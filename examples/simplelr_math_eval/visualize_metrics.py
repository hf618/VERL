import argparse
import json
import os
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def parse_args():
    # 此函数无需修改，与3.0版本相同
    parser = argparse.ArgumentParser(description="Visualize and compare representation metrics with global scaling and rule-based sampling.")
    parser.add_argument("--base_eval_dir", type=str, required=True, help="评估结果的基础目录")
    parser.add_argument("--steps", type=str, required=True, help="要可视化的 global_step 列表，用逗号分隔")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--metric_to_visualize", type=str, required=True, help="要可视化的指标全名")
    parser.add_argument("--metric_stride", type=int, required=True, help="计算指标时使用的stride")
    parser.add_argument("--init_model_path", type=str, required=True, help="用于加载分词器的初始模型路径")
    parser.add_argument("--template", type=str, default="qwen-boxed", help="评估时使用的模板名称")
    parser.add_argument("--temperature", type=float, default=0.01, help="评估时使用的温度")
    parser.add_argument("--output_html_path", type=str, default="./visualization_comparison.html", help="输出HTML文件的路径")
    parser.add_argument("--sampling_mode", type=str, required=True, choices=['direct', 'rule'], help="问题采样模式：'direct' 或 'rule'")
    parser.add_argument("--problem_ids", type=str, help="直接采样模式下，用逗号分隔的问题 'idx' 列表")
    parser.add_argument("--sampling_rule", type=str, help="规则采样模式下，定义规则的JSON字符串")
    parser.add_argument("--num_samples", type=int, help="规则采样模式下，要采样的样本数量")
    return parser.parse_args()

def load_jsonl_as_dict(path):
    # 此函数无需修改，与3.0版本相同
    if not os.path.exists(path):
        return None
    data_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data_dict[str(item.get("idx"))] = item
    return data_dict

def get_problem_ids_by_rule(args):
    """根据规则筛选问题ID"""
    print("[*] 正在根据规则筛选问题...")
    try:
        rule = json.loads(args.sampling_rule)
    except json.JSONDecodeError:
        print(f"[!] 错误: 采样规则 '{args.sampling_rule}' 不是一个有效的JSON字符串。")
        exit(1)

    # ... (函数前半部分逻辑与3.0版本相同)
    first_step_in_rule = list(rule.keys())[0]
    base_filename = f"test_{args.template}_-1_seed0_t{args.temperature}_s0_e-1"
    step_dir = os.path.join(args.base_eval_dir, f"global_step_{first_step_in_rule}", args.dataset)
    output_text_file = os.path.join(step_dir, f"{base_filename}.jsonl")
    all_data = load_jsonl_as_dict(output_text_file)
    if not all_data:
        print(f"[!] 错误: 无法加载 {output_text_file} 以获取问题全集。")
        exit(1)
    all_problem_ids = list(all_data.keys())
    
    candidate_ids = []
    for problem_id in tqdm(all_problem_ids, desc="Filtering by rule"):
        matches_all_conditions = True
        for step, required_score in rule.items():
            step_dir_rule = os.path.join(args.base_eval_dir, f"global_step_{step}", args.dataset)
            output_text_file_rule = os.path.join(step_dir_rule, f"{base_filename}.jsonl")
            step_data = load_jsonl_as_dict(output_text_file_rule)
            
            if not step_data or problem_id not in step_data:
                matches_all_conditions = False
                break
            
            actual_score = step_data[problem_id].get('score', [False])[0] 
            if actual_score != required_score:
                matches_all_conditions = False
                break
        
        if matches_all_conditions:
            candidate_ids.append(problem_id)
            
    print(f"[*] 找到 {len(candidate_ids)} 个符合规则的问题。")
    
    # =======================================================
    #               【核心修改点】
    # =======================================================
    # 如果找不到任何样本，则打印错误并直接退出程序
    if not candidate_ids:
        print("\n[!] 错误: 没有找到任何符合采样规则的问题。")
        print("    请检查您的 --sampling_rule 或放宽条件。")
        print("    程序将中止，不会生成HTML文件。")
        exit(1) # 以错误码1退出，阻止后续所有操作
    # =======================================================
    
    if len(candidate_ids) < args.num_samples:
        print(f"[*] 警告: 符合条件的样本数 ({len(candidate_ids)}) 小于要求采样数 ({args.num_samples})。将使用所有符合条件的样本。")
        return candidate_ids
        
    return random.sample(candidate_ids, args.num_samples)

# --- 其他所有函数 (get_global_metric_range, generate_visualization_for_one_sample, main) ---
# --- 均无需任何修改，与3.0版本完全相同。为简洁起见，此处不再重复展示。            ---
# --- 您只需用上面的新 get_problem_ids_by_rule 函数替换掉旧版本中的对应函数即可。  ---

# 为了保证您能直接使用，下面仍提供完整的 main 函数及其他部分

def get_global_metric_range(problem_ids, steps, args):
    """【新】遍历所有指定样本，计算全局的指标最大最小值"""
    print("[*] 正在计算全局颜色刻度尺...")
    all_metric_values = []
    base_filename = f"test_{args.template}_-1_seed0_t{args.temperature}_s0_e-1"
    
    for problem_id in problem_ids:
        for step in steps:
            stride_details_file = os.path.join(args.base_eval_dir, f"global_step_{step}", args.dataset, f"{base_filename}_stride_details.jsonl")
            stride_data = load_jsonl_as_dict(stride_details_file)
            if stride_data and problem_id in stride_data:
                target_details = stride_data[problem_id]
                layer_details = target_details.get('stride_details', {}).get('1', {})
                metric_values = layer_details.get(args.metric_to_visualize, [])
                if metric_values:
                    all_metric_values.extend(metric_values)
                    
    if not all_metric_values:
        return 0, 1 # 返回一个默认范围
    
    global_min = min(all_metric_values)
    global_max = max(all_metric_values)
    print(f"[*] 全局指标范围: [{global_min:.4f}, {global_max:.4f}]")
    return global_min, global_max

def generate_visualization_for_one_sample(tokenizer, text, metric_values, args, vmin, vmax):
    cmap = plt.get_cmap('Reds')
    tokens = tokenizer.encode(text, add_special_tokens=False)
    num_strides = len(metric_values)
    html_parts = []
    for i in range(num_strides):
        segment_tokens = tokens[i * args.metric_stride:(i + 1) * args.metric_stride]
        segment_text = tokenizer.decode(segment_tokens)
        metric_value = metric_values[i]
        hex_color = mcolors.to_hex(cmap((metric_value - vmin) / (vmax - vmin) if vmax > vmin else 0.5))
        tooltip_text = f"{args.metric_to_visualize}: {metric_value:.4f}"
        segment_text = segment_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_part = f'<span style="background-color: {hex_color}; padding: 2px; border-radius: 3px;" title="{tooltip_text}">{segment_text}</span>'
        html_parts.append(html_part)
    remaining_tokens_start = num_strides * args.metric_stride
    if remaining_tokens_start < len(tokens):
        remaining_text = tokenizer.decode(tokens[remaining_tokens_start:])
        remaining_text = remaining_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_parts.append(f'<span>{remaining_text}</span>')
    return f"<div class=\"container\">{''.join(html_parts)}</div>"

def main():
    args = parse_args()
    
    if args.sampling_mode == 'direct':
        problem_ids_to_visualize = [p.strip() for p in args.problem_ids.split(',')]
    elif args.sampling_mode == 'rule':
        problem_ids_to_visualize = get_problem_ids_by_rule(args)
    else:
        print(f"不支持的采样模式: {args.sampling_mode}")
        return

    print(f"[*] 将为 Steps: {args.steps} 和 Problem IDs: {problem_ids_to_visualize} 生成对比可视化")

    steps_to_visualize = [s.strip() for s in args.steps.split(',')]
    
    global_vmin, global_vmax = get_global_metric_range(problem_ids_to_visualize, steps_to_visualize, args)

    tokenizer = AutoTokenizer.from_pretrained(args.init_model_path, trust_remote_code=True)
    all_problems_html = []
    base_filename = f"test_{args.template}_-1_seed0_t{args.temperature}_s0_e-1"

    for problem_id in tqdm(problem_ids_to_visualize, desc="Generating Visualizations"):
        problem_html_parts = []
        question_text = ""
        
        for step in steps_to_visualize:
            step_dir = os.path.join(args.base_eval_dir, f"global_step_{step}", args.dataset)
            output_text_file = os.path.join(step_dir, f"{base_filename}.jsonl")
            stride_details_file = os.path.join(step_dir, f"{base_filename}_stride_details.jsonl")

            all_outputs = load_jsonl_as_dict(output_text_file)
            all_stride_details = load_jsonl_as_dict(stride_details_file)
            
            if not all_outputs or not all_stride_details or problem_id not in all_outputs or problem_id not in all_stride_details:
                error_msg = f"<div class='step-container error'><h3>Step: {step}</h3><p>错误: 找不到 Step {step} 中关于问题 {problem_id} 的数据。</p></div>"
                problem_html_parts.append(error_msg)
                continue
            
            target_output = all_outputs[problem_id]
            target_details = all_stride_details[problem_id]
            
            if not question_text:
                 question_text = target_output.get('question', '无法加载问题文本。')

            is_correct = target_output.get('score', [False])[0]
            icon = "✅" if is_correct else "❌"
            correctness_text = "Correct" if is_correct else "Incorrect"
            step_title = f"<h3>{icon} Step: {step} ({correctness_text})</h3>"

            layer_to_visualize = '1'
            metric_values = target_details.get('stride_details', {}).get(layer_to_visualize, {}).get(args.metric_to_visualize)

            if not metric_values:
                viz_html = "<p>警告: 指标值列表为空。</p>"
            else:
                viz_html = generate_visualization_for_one_sample(tokenizer, target_output['code'][0], metric_values, args, global_vmin, global_vmax)
            
            problem_html_parts.append(f"<div class='step-container'>{step_title}{viz_html}</div>")

        question_html = f"<h2>问题 ID: {problem_id}</h2><p class='question'>{question_text.replace('<', '&lt;')}</p>"
        all_problems_html.append(f"<div class='problem-section'>{question_html}{''.join(problem_html_parts)}</div>")

    cmap = plt.get_cmap('Reds')
    global_legend = f"""
    <div class="global-legend">
        <strong>全局图例 (指标: {args.metric_to_visualize})</strong>
        <span class="legend-color" style="background-color: {mcolors.to_hex(cmap(0.0))};"></span> Low ({global_vmin:.4f})
        <span class="legend-color" style="background-color: {mcolors.to_hex(cmap(0.5))};"></span> Mid
        <span class="legend-color" style="background-color: {mcolors.to_hex(cmap(1.0))};"></span> High ({global_vmax:.4f})
    </div>
    """
    
    full_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>指标对比可视化: {args.metric_to_visualize}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; padding: 20px; background-color: #f4f4f9; color: #333; }}
            h1, h2 {{ text-align: center; color: #1a1a1a; }}
            h3 {{ color: #333; }}
            .global-legend {{ position: sticky; top: 0; background-color: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); z-index: 100; text-align: center; margin-bottom: 20px; }}
            .problem-section {{ background-color: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 30px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .question {{ background-color: #eef; padding: 15px; border-radius: 5px; font-style: italic; }}
            .step-container {{ margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 20px; }}
            .step-container.error p {{ color: #d9534f; }}
            .container {{ border: 1px solid #eee; padding: 15px; border-radius: 5px; background-color: #fdfdfd; white-space: pre-wrap; word-break: break-all; }}
            .legend-color {{ display: inline-block; width: 18px; height: 18px; border: 1px solid #ccc; vertical-align: middle; margin: 0 5px; }}
        </style>
    </head>
    <body>
        <h1>推理链指标对比可视化</h1>
        {global_legend}
        {''.join(all_problems_html)}
    </body>
    </html>
    """
    
    with open(args.output_html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"\n[+] 对比可视化HTML文件已成功保存到: {args.output_html_path}")

if __name__ == "__main__":
    main()