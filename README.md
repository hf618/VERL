<h1 align="center">
<br>
<!-- <br style="display: block; content: ''; margin-top: 0.5em;" /> -->
Velocity-Exploiting Rank-Learning (VERL)</span>
</h1>

<div align="center">

</div>


<br>

<p align="center">
    <img src="docs/_static/first.png" width="1000">
        <br>
    <em>Figure 1: Comparative analysis with the responses of <a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B">DeepSeek-R1-Distill-Qwen-7B</a> in <a href="https://github.com/hkust-nlp/simpleRL-reason">simpleRL-reason</a> test dataset (Level 3 to 5). (a) Traditional metrics for exploitation and exploration are constrained by negative coupling, leading to meandering progress for both capabilities. (b) Our metrics are mutually independent. (c) Training regularization with our metrics demonstrates stronger performance in both exploitation (small K) and exploration (large K).
    </em>
</p>


## 🔧Key Implementations

VERL extends [veRL](https://github.com/volcengine/verl) with specific components across the following modules:

**[`verl/trainer/main_ppo.py`](verl/trainer/main_ppo.py) & [`verl/trainer/reward_manager_versions.py`](verl/trainer/reward_manager_versions.py)**

- Main entry point with ray initialization
- `RewardManager` for reward distribution

**[`verl/trainer/metrics_calculator.py`](verl/trainer/metrics_calculator.py) & [`verl/trainer/metrics_utils.py`](verl/trainer/metrics_utils.py)**

- `RepresentationMetricsCalculator` for metrics calculation
- Hidden states metrics in [`metrics_utils.py`](verl/trainer/metrics_utils.py)

**[`verl/trainer/ppo/ray_trainer.py`](verl/trainer/ppo/ray_trainer.py)**

- Main RL training loop: data loading, LLM rollout, model updates, evaluation, checkpointing
- RL algorithm-specific advantage computation

**[`verl/workers/fsdp_workers.py`](verl/workers/fsdp_workers.py)**

- Source of core functions called in `ray_trainer.py`
- LLM model/optimizer initialization, `generate_sequences`, `update_actor`

VERL extends [vllm](https://github.com/vllm-project/vllm) with specific components across the following folder:

**[`hidden_vllm/`](hidden_vllm/)**

- Added the hidden states extraction feature
- Modified from the low-level LLM model classes all the way up to the worker

## 🚀 Quick Start

### ⚙️ Setup

Our code is implemented based on [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason). We recommend using [Conda](https://docs.conda.io/projects/miniconda) to manage your environment. We use [vLLM](https://github.com/vllm-project/vllm) (0.5.4) to accelerate inference. Run the following commands to setup your environment:

```sh
conda create -n verl python==3.10.16
conda activate verl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e . 
pip3 install -r requirements.txt
```

### ⚡️ Training

We also open-source our complete training scripts for the community. We follow the training data used in [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason).

The training process leverages Ray and vLLM for acceleration. So firstly, you need to launch the ray cluster using the command below:

```sh
# launch the master node of ray 
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8
```

To start training, configure the required environment variables and customize the experiment settings at the end of the [train.sh](train.sh) script. Then, from the master node, submit the training job by running the following command:

```sh
bash train.sh
```

For the details of experiment settings, you can refer to [here](TRAINING_CONFIG.md).

### 🪁 Evaluation

We provide a script for inference, simply config the `RUN_NAME_MAP` and `ACTIVE_CONFIG_SET`  in [eval.sh](eval.sh) and run the following command:

```sh
bash eval.sh
```

You can also add your own test datasets to [this fold](/examples/simplelr_math_eval/data).

<br>
