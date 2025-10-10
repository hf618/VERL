#!/bin/bash

# ==============================================================================
#                 MACHINE-SPECIFIC CONFIGURATIONS
# ==============================================================================
# --- Project and API Keys ---
export PROJECT_NAME="VELR_train"
export WANDB_API_KEY=FILL_IN_YOURS_HERE
export WANDB_OFFICIAL=1

# --- Hardware and Network ---
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NUM_GPUS=4
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export NCCL_SOCKET_IFNAME=FILL_IN_YOURS_HERE

# --- File Paths ---
export HDFS_DATA_PATH=FILL_IN_YOURS_HERE
export HDFS_MODEL_PATH=FILL_IN_YOURS_HERE
export HDFS_CHECKPOINT_PATH=FILL_IN_YOURS_HERE
export HDFS_LOG_PATH=FILL_IN_YOURS_HERE

# --- Ray Cluster & Debug Configurations ---
export HEAD_IP=FILL_IN_YOURS_HERE
export HEAD_PORT="6379"
export ARNOLD_WORKER_NUM=1 # Number of nodes you want to use
export WORKING_DIR="."

# --- Ray Environment Exports ---
export RAY_BACKEND_LOG_LEVEL="debug"
export RAY_DEDUP_LOGS=1
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG="INFO"
export RAY_pickling_fallback="True"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
export RAY_DEBUG="legacy"
export REWORD_FUNCTION_TYPE="independent"

# --- Ray Runtime Environment JSON ---
# This block defines the runtime environment for the Ray job itself.
# Using a variable with correct quoting avoids JSON parsing errors.
export RAY_RUNTIME_ENV_JSON="{
    \"working_dir\": \"${WORKING_DIR}\",
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
#                 CONFIGURATIONS FOR YOUR EXPERIMENTS
# ==============================================================================

bash train_grpo_math_tune_ray.sh \
    --model_name mistralai/Mathstral-7B-v0.1 --max_prompt_length 512 --max_response_length 1536 \
    --critic_model_path "" --adv_estimator "grpo"  \
    --exp_name "GRPO_ORIGIN" --add_reward False --add_adv True \
    --train_batch_size 48 --ppo_mini_batch_size 24 --val_batch_size 48  --rollout_n 4 \
    --ppo_micro_batch_size 1 --log_prob_micro_batch_size 12 --micro_rollout_batch_size 12 \
    --compute_global_metrics True --compute_cumulative_global_metrics True --global_diff_stride_train 20 --global_diff_stride_val 20 \
    --kl_loss_coef 0.001 --entropy_coefficient 0.001 --rollout_gpu_memory_util 0.65 --logger_config "['console','wandb']" \
    --rollout_tp 1 --save_freq 40 --except_save "" --test_freq 10 --total_epochs 2 --total_steps 161 \
    --dataset_name "simplelr_abel_level3to5"  \
    --val_before_train True --val_sample_size -1 --enable_calculator True --metric_indices "[1,2]" \
    --reward_weights "[0.0, 0.0, 1.0]" --reward_weights_exploit "[0.0, 1.0, 0.0]" \
    --reward_indicator_names "['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank']" \
    --diff_stride 40 --modulation_gain 2.0 --aux_reward_global_weight 1.0 --aux_fix True --reward_ema_alpha 0.3 --adv_shaping_kappa 2.0 \
    --return_hidden_states True --return_prefill False --return_decode True \

# model -> dataset_name
# exp_name -> add_reward -> add_adv
# critic_model_path -> adv_estimator -> rollout_n -> global_diff_stride_train
# exp_name -> reward_weights -> reward_weights_exploit -> hypothesis_type