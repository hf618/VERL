#!/bin/bash

# ==============================================================================
#                 MACHINE-SPECIFIC CONFIGURATIONS
# ==============================================================================
# --- Project and API Keys ---
export PROJECT_NAME="verl_train_Gnode57"
export WANDB_API_KEY="8c84ddd422687515e5df25109f349a4f2c5df884"
export WANDB_OFFICIAL=1

# --- Hardware and Network ---
export CUDA_VISIBLE_DEVICES="0,5,6,7"
export NUM_GPUS=4
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export NCCL_SOCKET_IFNAME="ens15f0"

# --- File Paths ---
export HDFS_DATA_PATH="/home/fdhuang/storage_net/simpleRL/custom/data"
export HDFS_MODEL_PATH="/home/fdhuang/storage_net/Models"
export HDFS_CHECKPOINT_PATH="/home/fdhuang/storage_net/simpleRL/custom/checkpoint_rebuttal"
export HDFS_LOG_PATH="/home/fdhuang/storage_net/simpleRL-reason/custom/log"

# --- Ray Cluster & Debug Configurations ---
export HEAD_IP="192.168.1.107"
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
# llama/Llama-3.2-3B-Instruct
# qwen/Qwen2.5-3B 
bash train_grpo_math_tune_ray.sh \
    --model_name llama/Llama-3.2-3B-Instruct --max_prompt_length 512 --max_response_length 4096 \
    --critic_model_path "" --adv_estimator "grpo"  \
    --exp_name "GRPO_origin_mrl4096" --add_reward False --add_adv False \
    --train_batch_size 48 --ppo_mini_batch_size 24 --val_batch_size 48  --rollout_n 4 \
    --ppo_micro_batch_size 1 --log_prob_micro_batch_size 12 --micro_rollout_batch_size 12 \
    --compute_global_metrics True --compute_cumulative_global_metrics True --global_diff_stride_train 20 --global_diff_stride_val 20 \
    --kl_loss_coef 0.001 --entropy_coefficient 0.001 --rollout_gpu_memory_util 0.85 --logger_config "['console','wandb']" \
    --rollout_tp 1 --save_freq 40 --except_save "" --test_freq 10 --total_epochs 2 --total_steps 161 \
    --dataset_name "simplelr_abel_level1to4"  \
    --val_before_train True --val_sample_size -1 --enable_calculator True --metric_indices "[1,2]" \
    --metric_indices_add "['avg_log_probs','avg_response_entropy','avg_logits_entropy']" \
    --reward_weights "[0.0, 0.0, 1.0, 0.0, 0.0]" --reward_weights_exploit "[0.0, 1.0, 0.0, 0.0, 0.0]" \
    --reward_indicator_names "['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank', 'avg_log_probs','avg_response_entropy']" \
    --diff_stride 128 --modulation_gain 2.0 --aux_reward_global_weight 1.0 --aux_fix True --reward_ema_alpha 0.3 --adv_shaping_kappa 2.0 \
    --return_hidden_states True --return_prefill False --return_decode True \
    --hypothesis_type "PlanB" --diff_calculator_method "optimized"
# model -> dataset_name
# exp_name -> add_reward -> add_adv
# critic_model_path -> adv_estimator -> rollout_n -> global_diff_stride_train
# exp_name -> reward_weights -> reward_weights_exploit -> hypothesis_type
