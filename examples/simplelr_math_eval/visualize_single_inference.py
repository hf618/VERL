import os
import sys
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- 关键修复：确保Python能够找到依赖的模块 ---
# 这个代码块会把您的工具脚本所在的目录添加到Python的搜索路径中
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    module_path = os.path.join(project_root, 'examples', 'simplelr_math_eval')
    if module_path not in sys.path:
        sys.path.append(module_path)
except NameError:
    # 为交互式环境（如Jupyter）提供备用路径
    module_path = os.path.abspath('./examples/simplelr_math_eval')
    if module_path not in sys.path:
        sys.path.append(module_path)

# 现在可以安全地导入了
from metrics_calculator import RepresentationMetricsCalculator
from data_loader import load_data
from utils import construct_prompt, set_seed


# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>推理链指标可视化</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 2em; line-height: 1.6; }}
        .container {{ border: 1px solid #e1e4e8; border-radius: 6px; margin-bottom: 2em; }}
        .header {{ background-color: #f6f8fa; padding: 10px 16px; border-bottom: 1px solid #e1e4e8; }}
        .content {{ padding: 16px; }}
        h1, h2, h3 {{ border-bottom: 1px solid #eaecef; padding-bottom: .3em; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 16px; border-radius: 3px; }}
        .segment {{ display: inline-block; transition: all 0.2s ease-in-out; border-radius: 3px; padding: 1px 0; }}
        .segment:hover {{ transform: scale(1.05); box-shadow: 0px 0px 15px rgba(0,0,0,0.2); }}
        .legend {{ display: flex; align-items: center; margin-top: 1em; }}
        .gradient-bar {{ width: 200px; height: 20px; border: 1px solid #ccc; background: linear-gradient(to right, {color_min}, {color_max}); }}
    </style>
</head>
<body>
    <h1>推理链指标可视化</h1>
    
    <div class="container">
        <div class="header"><h3>任务信息</h3></div>
        <div class="content">
            <strong>模型:</strong> {model_path}<br>
            <strong>数据集:</strong> {dataset}<br>
            <strong>问题ID:</strong> {question_idx}<br>
            <strong>可视化指标:</strong> {metric_name}
        </div>
    </div>

    <div class="container">
        <div class="header"><h3>问题</h3></div>
        <div class="content"><pre>{question}</pre></div>
    </div>

    <div class="container">
        <div class="header"><h3>可视化推理链 (基于: {metric_name})</h3></div>
        <div class="content">
            <div class="legend">
                <span>指标值: &nbsp;</span>
                <span>{min_val:.4f}</span>
                <div class="gradient-bar"></div>
                <span>{max_val:.4f}</span>
            </div>
            <hr>
            <p style="font-size: 1.1em;">{colored_answer}</p>
        </div>
    </div>
</body>
</html>
"""

def value_to_color(value, v_min, v_max):
    """Maps a value to a shade of red (HSL)."""
    if v_max == v_min:
        return "hsl(0, 70%, 85%)" # A neutral red if all values are the same
    
    # Normalize value to 0-1 range
    normalized = (value - v_min) / (v_max - v_min)
    
    # Lightness from 90% (light red) to 60% (darker red)
    lightness = 90 - (normalized * 30)
    
    return f"hsl(0, 80%, {lightness}%)"


def generate_html_visualization(data):
    """Generates the final HTML file from the processed data."""
    
    metric_values = data["metric_values"]
    v_min, v_max = min(metric_values), max(metric_values)
    
    # --- 分割文本并着色 ---
    generated_text = data["generated_text"]
    stride = data["stride"]
    segments = []
    
    for i, metric_val in enumerate(metric_values):
        start_char = i * stride
        end_char = (i + 1) * stride
        text_segment = generated_text[start_char:end_char]
        
        # HTML-safe text
        text_segment_html = text_segment.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        color = value_to_color(metric_val, v_min, v_max)
        tooltip = f"片段: '{text_segment}'\n{data['metric_name']}: {metric_val:.5f}"
        
        segments.append(
            f'<span class="segment" style="background-color: {color};" title="{tooltip}">{text_segment_html}</span>'
        )
        
    colored_answer = "".join(segments)

    # --- 渲染HTML模板 ---
    html_content = HTML_TEMPLATE.format(
        model_path=data["model_path"],
        dataset=data["dataset"],
        question_idx=data["question_idx"],
        metric_name=data["metric_name"],
        question=data["question_text"],
        colored_answer=colored_answer,
        min_val=v_min,
        max_val=v_max,
        color_min=value_to_color(v_min, v_min, v_max),
        color_max=value_to_color(v_max, v_min, v_max)
    )
    
    filename = f"visualization_{data['dataset']}_{data['question_idx']}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\n[SUCCESS] Visualization saved to: {os.path.abspath(filename)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize metrics for a single inference chain.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--init_model_path", type=str, required=True, help="Path to the initial model for tokenizer.")
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--question_idx", type=str, required=True)
    parser.add_argument("--metric_to_visualize", type=str, required=True)
    parser.add_argument("--order_to_visualize", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--max_tokens_per_call", type=int, default=1024)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)

    # 1. 加载模型和分词器
    print(f"Loading tokenizer from: {args.init_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.init_model_path, trust_remote_code=True)
    
    print(f"Loading model from: {args.model_name_or_path}")
    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        # enforce_eager=True, # 如果您的模型代码未按推荐方式修改，则需要保留此行
    )

    # 2. 加载单个问题并构建Prompt
    print(f"Loading dataset '{args.dataset}' to find question '{args.question_idx}'...")
    # 注意：这里的 data_dir 是硬编码的，与 math_eval.py 保持一致
    full_dataset = load_data(args.dataset, "test", "./data") 
    question_data = next((item for item in full_dataset if str(item.get("idx")) == args.question_idx), None)
    
    if not question_data:
        raise ValueError(f"Question with idx '{args.question_idx}' not found in dataset '{args.dataset}'.")
        
    prompt_args = argparse.Namespace(prompt_type=args.template, num_shots=0, adapt_few_shot=False)
    prompt = construct_prompt(question_data, args.dataset, prompt_args)

    # 3. 生成回答并获取Hidden States
    print(f"Generating answer for question {args.question_idx}...")
    sampling_params = SamplingParams(
        temperature=0.01,
        max_tokens=args.max_tokens_per_call,
    )
    # 假设您的vLLM有方式返回hidden_states, 例如通过修改模型代码实现
    vllm_output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
    
    completion_output = vllm_output.outputs[0]
    generated_text = completion_output.text
    hidden_states = getattr(completion_output, 'hidden_states', None)
    
    if hidden_states is None:
        raise RuntimeError("Could not retrieve hidden_states from vLLM output. Ensure your vLLM version/model is correctly configured to return them.")

    print(f"\nGenerated Text:\n---\n{generated_text}\n---\n")

    # 4. 计算指标
    print("Calculating metrics per stride...")
    metrics_calculator = RepresentationMetricsCalculator(tokenizer)
    
    # 假设 hidden_states 的形状是 (seq_len, num_layers, hidden_dim)
    seq_len = hidden_states.shape[0]
    attention_mask = torch.ones(seq_len, device=hidden_states.device)
    
    orders_to_calc = [0, 1, 2] # 计算所有阶，以便后续选择
    metrics_to_calc = [args.metric_to_visualize]

    _, stride_details = metrics_calculator(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        compute_diff=True,
        diff_stride=args.stride,
        metrics_to_calc=metrics_to_calc,
        orders_to_calc=orders_to_calc
    )
    
    # 5. 提取用于可视化的特定指标序列
    metric_name_to_visualize = args.metric_to_visualize
    if args.order_to_visualize == 1:
        metric_name_to_visualize += " diff"
    elif args.order_to_visualize == 2:
        metric_name_to_visualize += " diff 2"
        
    layer_to_visualize = '1' # 假设我们只可视化第一层的指标
    if not stride_details or layer_to_visualize not in stride_details or metric_name_to_visualize not in stride_details[layer_to_visualize]:
        raise ValueError(f"Could not find metric '{metric_name_to_visualize}' in calculated results. Available metrics: {stride_details.get(layer_to_visualize, {}).keys()}")
        
    metric_values = stride_details[layer_to_visualize][metric_name_to_visualize]

    if not metric_values:
        raise ValueError("The metric value list is empty. This might happen if the generated text is shorter than the stride.")

    # 6. 生成HTML可视化文件
    visualization_data = {
        "model_path": args.model_name_or_path,
        "dataset": args.dataset,
        "question_idx": args.question_idx,
        "metric_name": metric_name_to_visualize,
        "question_text": question_data['question'],
        "generated_text": generated_text,
        "metric_values": metric_values,
        "stride": args.stride
    }
    
    generate_html_visualization(visualization_data)


if __name__ == "__main__":
    main()