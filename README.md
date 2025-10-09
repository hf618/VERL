---

## 更新说明

### 2025.8.12

主要就是不用直接在vllm源码里魔改，直接把vllm继承到项目下，所以直接pull下来就可以跑，要不然每次更新代码到源码太抽象了；然后train.sh多了很多变量的开关。用新的版本之前更新一下vllm，重新pip install vllm=0.5.4

## ⚠️ 注意事项 (Important Notes)

### 关于 `train_grpo_math_tune_ray.sh`

这个脚本文件是一个**与特定机器相关的配置文件** (machine-specific)。

其中包含的路径、GPU数量、节点信息等参数，需要每一位开发者根据自己所使用的服务器环境进行修改，例如： `train_grpo_math_tune_ray.sh`中 `export VLLM_ATTENTION_BACKEND=FLASH_ATTN ` 只适用于安倍架构以上的GPUs


为了避免在协作中互相覆盖个人的配置，请遵循以下约定：

* **不要提交你对 `train_grpo_math_tune_ray.sh` 文件的任何本地修改。**
* 如果你修改了这个文件以适应你自己的环境，请在执行 `git add .` 命令前，确保这个文件没有被添加到暂存区。如果不小心添加了，可以使用 `git restore --staged train_grpo_math_tune_ray.sh` 将其移出暂存区。

### 关于 `V100`的缺陷

V100上不支持bfloat16和flash_attention，有些地方可能得手动调整，可以通过vscode遍历查询。典型文件路径是：`verl/workers/fsdp_workers.py`，得把类似`torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2'`的地方改为`torch.float16`和`attn_implementation='eager'`

（这个后续希望迭代一版为识别gpu类型的通用版本）



### 关于vllm源码库里hidden states的传递过程

| File Path | Description | 是否修改 |
| :--- | :--- | --- | 
| `model_executor/models/qwen2.py` | `Qwen2Model.forward()`中增加提取 hidden states 逻辑 | yes |
| `model_executor/models/llama.py` | 同上 | yes |
| `model_executor/models/mixtral.py` | 同上 | yes |
| `model_executor/models/gemma2.py` | 同上 | yes |
| `worker/model_runner.py` | ModelRunner.execute_model 中的 if self.return_hidden_states: 后面跟随接受 hidden states；有一个古法选择哪个 hidden states | yes |
| `engine/llm_engine.py` | LLMEngine.step 中的 output = self.model_executor.execute_model(execute_model_req=execute_model_req) | no |
| `entrypoints/llm.py` | LLM._run_engine 中的 step_outputs = self.llm_engine.step() | 不确定 |
| `outputs.py` | RequestOutput 增加初始化参数 hidden_states_decode 和 hidden_states_prefill；在from_seq_group方法中，从SequenceGroup 对象中获取 Prefill 阶段的隐藏状态，从每个Sequence 对象中获取 Decode 阶段的隐藏状态 | yes |
| `sequence.py` | Sequence 类增加 append_hidden_state 方法实现使用torch.cat实现增量存储 hidden_states_decode | yes |
| `engine/output_processor/single_step.py` | process_prompt_logprob 方法中添加 hidden_states_prefill， _process_sequence_group_outputs 方法中添加 hidden_states_decode | yes |
| `engine/output_processor/util.py` | 修改 create_output_by_sequence_group 方法处理 hidden states，方便 `engine/llm_engine.py` 调用 | yes |
| `worker/worker.py` | Work 类初始化的 self.model_runner: GPUModelRunnerBase = ModelRunnerClass 显式传入 return_hidden_states = False or True 作为总开关,如果FALSE,则decode 和 prefill都不输出 这里的 Work 类是测试eval的时候被调用，所以按需提取hidden states | yes |


---
