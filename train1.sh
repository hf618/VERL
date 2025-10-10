#!/bin/bash

# ==============================================================================
#                 MACHINE-SPECIFIC CONFIGURATIONS
# ==============================================================================
# --- Environment Exports ---
# These variables will be set for the train_grpo_math_tune_ray.sh script and its children processes.
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1
export PROJECT_NAME=verl_train_gong
export WANDB_API_KEY="8c84ddd422687515e5df25109f349a4f2c5df884" # Your WandB API Key
export WANDB_OFFICIAL=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=enp0s31f6 # IMPORTANT: Set to your network interface (e.g., eth0)
export RAY_pickling_fallback="True"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES="0,1"
export RAY_DEBUG=legacy
export REWORD_FUNCTION_TYPE="independent"
export TORCH_USE_CUDA_DSA=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# --- Path Configurations ---
export HDFS_DATA_PATH="/home/root1/Fanding/simpleRL-reason/custom/data" 
export HDFS_MODEL_PATH="/media/root1/4t/Models"
export HDFS_CHECKPOINT_PATH="/home/root1/Fanding/simpleRL-reason/custom/checkpoint"
export HDFS_LOG_PATH="/home/root1/Fanding/simpleRL-reason/custom/log"

# --- Ray Cluster & Hardware Configurations ---
export HEAD_IP="219.223.185.150"
export HEAD_PORT="6379"
export ARNOLD_WORKER_NUM=1 # Number of nodes you want to use
export NUM_GPUS=2 # Number of GPUs per node

# --- Working Directory for Ray ---
# This is the directory from which the job will be run inside the Ray cluster.
export WORKING_DIR="."

# --- Ray Runtime Environment ---
# We define the JSON content here and pass it to the next script.
# This makes it easier to manage complex configurations like 'excludes'.
export RAY_RUNTIME_ENV_JSON="{
    \"working_dir\": \"${WORKING_DIR}\",
    \"excludes\": [
      \"/.git/\",
      \"/checkpoint/\",
      \"/custom/checkpoint/\",
      \"/custom/log/\",
      \"/custom/data/\",
      \"/home/root1/Fanding/simpleRL-reason/examples/simplelr_math_eval/data/tabmwp/test.jsonl\"
    ],
    \"env_vars\": {
      \"http_proxy\": \"\",
      \"https_proxy\": \"\",
      \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
      \"CUDA_LAUNCH_BLOCKING\": \"${CUDA_LAUNCH_BLOCKING}\",
      \"NCCL_DEBUG\": \"${NCCL_DEBUG}\",
      \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
      \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"${RAY_OVERRIDE_JOB_RUNTIME_ENV}\",
      \"REWORD_FUNCTION_TYPE\": \"${REWORD_FUNCTION_TYPE}\",
      \"RAY_DEBUG\": \"${RAY_DEBUG}\",
      \"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES}\"
    }
}"

# ==============================================================================
#                 EXECUTION LOGIC
# ==============================================================================

bash train_grpo_math_tune_ray.sh \
    --model_name qwen/Qwen2.5-1.5B --max_prompt_length 512 --max_response_length 1536 \
    --critic_model_path "" --adv_estimator "grpo"  \
    --exp_name "GRPO_VERL" --add_reward False --add_adv False \
    --train_batch_size 8 --ppo_mini_batch_size 4 --val_batch_size 48  --rollout_n 4 \
    --ppo_micro_batch_size 1 --log_prob_micro_batch_size 12 --micro_rollout_batch_size 12 \
    --compute_global_metrics True --compute_cumulative_global_metrics True --global_diff_stride_train 20 --global_diff_stride_val 20 \
    --kl_loss_coef 0.001 --entropy_coefficient 0.001 --rollout_gpu_memory_util 0.65 --logger_config "['console','wandb']" \
    --rollout_tp 1 --save_freq 40 --except_save "" --test_freq 10 --total_epochs 2 --total_steps 161 \
    --dataset_name "simplelr_abel_level3to5"  \
    --val_before_train False --val_sample_size -1 --enable_calculator True --metric_indices "[1,2]" \
    --reward_weights "[0.0, 0.0, 1.0]" --reward_weights_exploit "[0.0, 1.0, 0.0]" \
    --reward_indicator_names "['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank']" \
    --diff_stride 40 --modulation_gain 2.0 --aux_reward_global_weight 1.0 --aux_fix True --reward_ema_alpha 0.3 --adv_shaping_kappa 2.0 \
    --return_hidden_states True --return_prefill False --return_decode True \

# model -> dataset_name
# exp_name -> add_reward -> add_adv
# critic_model_path -> adv_estimator -> rollout_n -> global_diff_stride_train
# exp_name -> reward_weights -> reward_weights_exploit -> hypothesis_type
