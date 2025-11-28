import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
import json 
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from verl.trainer.metrics_calculator import RepresentationMetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument("--calculate_metrics", type=str, default="false", help="Enable hidden states metrics calculation.")
    parser.add_argument("--metrics_to_calc", type=str, default="Response Entropy 1", help="Comma-separated list of metrics to calculate.")
    parser.add_argument("--metric_orders", type=str, default="0,1,2", help="Orders of metrics to calculate, e.g., '0,1'.")
    parser.add_argument("--metric_stride", type=int, default=10, help="Stride for calculating metrics.")
    parser.add_argument("--num_test_sample_per_dataset", 
                        type=int, 
                        default=-1, 
                        help="Number of test samples per dataset to use for debugging")
    parser.add_argument("--dtype", default=" torch.bfloat16", type=str, help="Data type for the model (e.g., 'torch.float16', 'torch.bfloat16', 'auto')")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6, help="The fraction of GPU memory to be used by the vLLM engine. 1.0 means use all available memory.")
    
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    args.calculate_metrics = args.calculate_metrics.lower() in ['true', '1', 't', 'y', 'yes']

    if args.calculate_metrics and not args.metrics_to_calc:
        print("[INFO] --calculate_metrics is true but --metrics_to_calc is empty. Disabling metrics calculation.")
        args.calculate_metrics = False
   
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    if args.num_test_sample_per_dataset > 0:
        examples = examples[:args.num_test_sample_per_dataset]
        print(f"<<<<< DEBUG MODE: USING FIRST {args.num_test_sample_per_dataset} SAMPLES IN {data_name} >>>>>")

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    # if not os.path.exists(output_dir):
    #     output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    metrics_out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_repr_metrics.jsonl"

    stride_details_out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_stride_details.jsonl"
   
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]

    return examples, processed_samples, out_file, metrics_out_file, stride_details_out_file


def setup(args):

    # Helper function to convert string to torch.dtype
    def get_torch_dtype(dtype_str):
        if dtype_str in ["torch.float16", "float16"]:
            return torch.float16
        elif dtype_str in ["torch.bfloat16", "bfloat16"]:
            return torch.bfloat16
        elif dtype_str == "auto":
            return "auto"
        else:
            # Fallback to float16 for unrecognized strings
            return torch.float16

    # load model
    
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    data_list = args.data_names.split(",")
    need_eval_data_list = []
    if not args.overwrite:
        for data_name in data_list:
            out_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
            out_file =  f"{args.output_dir}/{data_name}/{out_prefix}_s{args.start}_e{args.end}.jsonl"
            out_metric_json = out_file.replace(".jsonl", f"_metrics.json")
            
            if os.path.exists(out_metric_json):
                print(f"Skipping {data_name} because {out_metric_json} already exists.")
                continue
            else:
                need_eval_data_list.append(data_name)
    
        if len(need_eval_data_list) == 0:
            print("All datasets already evaluated. Exiting.")
            exit(0)
        data_list = need_eval_data_list
    

    model_dtype = get_torch_dtype(args.dtype.strip())
    if args.use_vllm:

        print(f"[DEBUG] Initializing vLLM with gpu_memory_utilization = {args.gpu_memory_utilization}")
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            dtype=model_dtype,
            enforce_eager=True,
            enable_chunked_prefill=False,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True, dtype=model_dtype, enforce_eager=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
            dtype= torch.bfloat16,
        )

    # infer & eval
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):

    examples, processed_samples, out_file, metrics_out_file, stride_details_out_file = prepare_data(data_name, args)


    # 初始化 RepresentationMetricsCalculator
    metrics_calculator = None
    if args.calculate_metrics:
        print("Representation metrics calculation is enabled.")
        metrics_calculator = RepresentationMetricsCalculator(tokenizer=tokenizer)


    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
    elif "deepseek" in args.prompt_type:
        stop_words.extend(["\nProblem", "User:", "Assistant:", "</answer>", "</s>"])
    elif "qwen" in args.prompt_type:
        stop_words.extend(["assistant", "user", "_end", "_start"])
    elif "abel" in args.prompt_type:
        stop_words.extend(["Question:", "Answer:"])

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break
        # assert False
        # if epoch > 0 :
        #     assert False
        # get all outputs
        prompts = [item[1] for item in current_prompts]
        # Initialize stop_token_ids based on the model name
        stop_token_ids = None
        if "mistral" in args.model_name_or_path.lower():
            if "24b" in args.model_name_or_path.lower():
                stop_token_ids = [23836, 19464, 3263, 18993]  # _end, istant, user, _start
            elif "7b-v0.1" in args.model_name_or_path.lower():
                stop_token_ids = [22478, 24994, 26307, 9977]
        elif "qwen2" in args.model_name_or_path.lower():
            stop_token_ids = [151645, 151643]  
        
        if args.use_vllm:
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=stop_token_ids,
                ),
            )
            


            outputs = sorted(outputs, key=lambda x: int(x.request_id))  # sort outputs by request_id


            hidden_states_list = []
            if args.calculate_metrics:
                for output in outputs:
   
                    if output.hidden_states[0] is not None:
                        hidden_states_list.append(output.hidden_states[0].to(torch.bfloat16))  # 确保转换为 float16
                    else:
                        print("Warning: `hidden_states` not found in vLLM output. Metrics calculation will be skipped.")
                        args.calculate_metrics = False
                        break # 后续不再检查

            outputs = [(output.outputs[0].text, output.outputs[0].finish_reason) for output in outputs]

        else:
            outputs_text = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

            outputs = [(text, None) for text in outputs_text]



        # assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), (output, finish_reason) in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query, finish_reason))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query, finish_reason))
            # elif "boxed" not in output and output.endswith("```"):
            #     program = extract_program(query)
            #     remain_prompts.append((i, query))
            #     remain_codes.append(program)
            else:
                end_prompts.append((i, query, finish_reason))

        # execute the remain prompts
        # assert len(remain_codes)==0
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            # assert False
            i, query, finish_reason = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query,finish_reason)


    all_aggregated_metrics_data = []
    all_stride_details_data = [] # <--- 新增
    if args.calculate_metrics and metrics_calculator:
        print("Calculating representation metrics...")

        metrics_to_calc_list = [m.strip() for m in args.metrics_to_calc.split(',') if m.strip()]
        orders_to_calc = [int(o.strip()) for o in args.metric_orders.split(',') if o.strip()]

        
        for i, sample in enumerate(tqdm(samples, desc="Calculating Metrics")):
            hidden_states = hidden_states_list[i]
            if hidden_states is None or hidden_states.dim() != 3: # 确保是3D张量 (seq_len, num_layers, hidden_dim)
                print(f"Skipping sample {i} due to invalid hidden_states.")
                continue
            

            seq_len = hidden_states.shape[0] # 注意：现在 seq_len 在第0维
            attention_mask = torch.ones(seq_len, device=hidden_states.device)

 
            aggregated_metrics, stride_details = metrics_calculator(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                compute_diff=True,
                diff_stride=args.metric_stride,
                metrics_to_calc=metrics_to_calc_list,
                orders_to_calc=orders_to_calc 
            )
            
            all_aggregated_metrics_data.append({
                "idx": sample["idx"],
                "metrics": {
                    layer: {k: round(v.item(), 5) for k, v in metrics.items()}
                    for layer, metrics in aggregated_metrics.items()
                }
            })
            
            all_stride_details_data.append({
                "idx": sample["idx"],
                "stride_details": {
                    layer: {k: [round(val, 5) for val in v] for k, v in details.items()}
                    for layer, details in stride_details.items()
                }
            })




    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    finish_reasons = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt, finish_reason = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)
        finish_reasons.append(finish_reason)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        finish_reason_list = finish_reasons[i * args.n_sampling : (i + 1) * args.n_sampling]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports, "finish_reason": finish_reason_list  })
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)
        if all_aggregated_metrics_data:
            print(f"Saving aggregated representation metrics to {metrics_out_file}")
            save_jsonl(all_aggregated_metrics_data, metrics_out_file)
        if all_stride_details_data:
            print(f"Saving stride detail metrics to {stride_details_out_file}")
            save_jsonl(all_stride_details_data, stride_details_out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
