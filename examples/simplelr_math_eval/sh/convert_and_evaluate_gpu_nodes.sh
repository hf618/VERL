#!/bin/bash
set -x

# Parameters check
if [ "$#" -ne 21 ]; then 
    echo "Usage: $0 <eval_script_path> <base_checkpoint_path> <init_model_path> <template> [benchmarks] [temperature] [max_tokens] [top_p] [tp_size] [ckpt_list_file] [output_dir] [overwrite] [n_sampling] [visible_gpus] [calculate_metrics] [metrics_to_calc] [metric_stride] [metric_orders] [num_test_sample_per_dataset] [dtype] [gpu_memory_utilization]"
    exit 1
fi

# Get parameters
eval_script_path=$1
base_checkpoint_path=$2
init_model_path=$3
template=$4
benchmarks=$5
temperature=$6
max_tokens=$7
top_p=$8
tp_size=${9:-1} 
ckpt_list_file=${10:-""} 
output_dir_base=${11:-"eval_results"}
overwrite=${12:-false}
n_sampling=${13:-1}
# output_dir="${output_dir_base}_n${n_sampling}"
output_dir="${output_dir_base}"
actor_dir="actor"

visible_gpus=${14:-""}
calculate_metrics=${15:-"false"}
metrics_to_calc=${16:-""}
metric_stride=${17:-1}
metric_orders=${18:-"0,1,2"}
num_test_sample_per_dataset=${19:--1}  # The default value is -1, which means all samples are used.
dtype=${20:-"torch.bfloat16"}
gpu_memory_utilization=${21:-1.0}
# visible_gpus=${14:-""} 
# if [ -n "$visible_gpus" ]; then
#     export CUDA_VISIBLE_DEVICES="$visible_gpus"
#     echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES" >&2
# fi
VISIBLE_GPUS_ARRAY=()
if [ -n "$visible_gpus" ]; then
    IFS=',' read -r -a VISIBLE_GPUS_ARRAY <<< "$visible_gpus"
    export CUDA_VISIBLE_DEVICES="$visible_gpus"
    echo "CUDA_VISIBLE_DEVICES set to: $visible_gpus" >&2
else
    mapfile -t VISIBLE_GPUS_ARRAY < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
# Get the number of available GPUs
get_visible_gpus_count() {
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | xargs
    else
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^$' | wc -l | xargs
    fi
}
NUM_GPUS=$(get_visible_gpus_count)
NUM_GPUS=${#VISIBLE_GPUS_ARRAY[@]}
NUM_GPU_GROUPS=$((NUM_GPUS / tp_size))

copy_tokenizer_files() {
    local ckpt_path=$1
    local init_model_path=$2
    local files_to_copy=(
        "added_tokens.json"
        "config.json"
        "generation_config.json"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )
    if [ -f "$init_model_path/merges.txt" ]; then
        files_to_copy+=("merges.txt")
    fi
  
    if [ ! -d "$ckpt_path" ]; then
        mkdir -p "$ckpt_path"
        echo "Created checkpoint directory: $ckpt_path" >&2
    else
        echo "Checkpoint directory already exists: $ckpt_path" >&2
    fi

    for filename in "${files_to_copy[@]}"; do
        src="$init_model_path/$filename"
        dst="$ckpt_path/$filename"
        if [ -e "$src" ]; then
            cp "$src" "$dst"
            echo "Copied $src to $dst"
        else
            echo "Warning: $src does not exist."
        fi
    done
}

# Function: Retrieve all checkpoints that require evaluation and filter out those that have already been evaluated.
get_checkpoints_to_evaluate() {
    local base_path="$1"
    
    if [ -n "$ckpt_list_file" ] && [ -f "$ckpt_list_file" ]; then
        # Read checkpoints from the provided file
        cat "$ckpt_list_file"
    else
        # Original logic for getting all checkpoints
        local checkpoints=()
        for ckpt_dir in "$base_path"/global_step_*; do
            if [ -d "$ckpt_dir" ]; then
                step_tag=$(basename "$ckpt_dir")
                checkpoints+=("$step_tag")
            fi
        done
        
        if [ ${#checkpoints[@]} -eq 0 ]; then
            echo ""
        else
            printf "%s\n" "${checkpoints[@]}"
        fi
    fi
}

# Function: Processing a Single Checkpoint on a Specified GPU
process_checkpoint() {
    local start_idx=$((group_id * tp_size))
    local gpu_ids=""

    for ((i=0; i<tp_size; i++)); do
        physical_gpu=${VISIBLE_GPUS_ARRAY[$((start_idx + i))]}
        if [ -z "$physical_gpu" ]; then
            echo "Error: Not enough visible GPUs available for group $group_id." >&2
            exit 1
        fi
        if [ -n "$gpu_ids" ]; then
            gpu_ids="${gpu_ids},"
        fi
        gpu_ids="${gpu_ids}${physical_gpu}"
    done
    
    ckpt_path="$base_checkpoint_path/$step_tag/$actor_dir/huggingface"
    
    local port=29501

    echo "Evaluating checkpoint $step_tag on GPUs $gpu_ids" >&2
    
    output_path_new="$base_checkpoint_path/$output_dir/$step_tag"
    mkdir -p "$output_path_new"
    
    CUDA_VISIBLE_DEVICES=$gpu_ids bash "$eval_script_path" \
        ${template} "$ckpt_path" "$output_path_new" "$temperature" \
        "$max_tokens" "$top_p" "$benchmarks" "$overwrite" "$n_sampling" \
        "$calculate_metrics" "$metrics_to_calc" "$metric_stride" "$metric_orders" "$num_test_sample_per_dataset" \
        "$dtype" "$gpu_memory_utilization"


}

# Record the current working directory
original_dir=$(pwd)

# Main script modifications
# Get the checkpoints that need to be evaluated
readarray -t checkpoints_to_evaluate < <(get_checkpoints_to_evaluate "$base_checkpoint_path")

if [ ${#checkpoints_to_evaluate[@]} -eq 0 ]; then
    echo "No new checkpoints to evaluate." >&2
    exit 0
fi

# Check if the number of GPUs meets the tp_size requirement
if [ $((NUM_GPUS % tp_size)) -ne 0 ]; then
    echo "Error: Number of available GPUs ($NUM_GPUS) is not divisible by tp_size ($tp_size)" >&2
    exit 1
fi

echo "Found ${#checkpoints_to_evaluate[@]} checkpoints to evaluate:" >&2
printf '%s\n' "${checkpoints_to_evaluate[@]}" >&2
total_checkpoints=${#checkpoints_to_evaluate[@]}
eval_count=0
# Parallel processing checkpoint, allocated by GPU group.
for i in "${!checkpoints_to_evaluate[@]}"; do
    group_id=$((i % NUM_GPU_GROUPS))
    step_tag="${checkpoints_to_evaluate[i]}"

    # Start processing task in the background
    process_checkpoint "$step_tag" "$group_id" 
    
    # Wait for them to complete after starting every NUM_GPU_GROUPS tasks.
    if [ $(((i + 1) % NUM_GPU_GROUPS)) -eq 0 ]; then
        wait
    fi
    eval_count=$((eval_count + 1))
    echo "Evaluating $eval_count/$total_checkpoints checkpoints ..."
done

# Wait for all remaining background tasks to complete
wait

cd "$original_dir"
echo "All conversions and evaluations completed."