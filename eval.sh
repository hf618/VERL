#!/bin/bash

# =======================================================
#               Parameter Configuration Area
# =======================================================
DTYPE="torch.bfloat16"
HDFS_PATH=FILL_IN_YOURS_HERE
MODEL_BASE_PATH=FILL_IN_YOURS_HERE
GPU_MEMORY_UTILIZATION=0.75
 
# ==============================================================================
#                      1. CHOOSE YOUR CONFIGURATION HERE
# ==============================================================================
# Just change the value of this variable to switch between test sets.
# Available options: "FULL_PASS1", "SMALL_SAMPLES_PASS16", "TINY_SAMPLES_PASS256"

ACTIVE_CONFIG_SET="FULL_PASS1"
TEMPERATURES=(0.6)

# ==============================================================================
#                2. DEFINE ALL YOUR CONFIGURATION DATA SETS
# ==============================================================================

# --- Define the benchmark strings for each set ---
declare -A BENCHMARK_SETS
BENCHMARK_SETS=(
["FULL_PASS1"]="math24o,OlymMATH-EN-EASY,OlymMATH-EN-HARD,OlymMATH-ZH-EASY,OlymMATH-ZH-HARD,BeyondAIME,aime25,amc24,aime24,amc23,aqua,asdiv,carp_en,cmath,cn_middle_school,college_math,gaokao2023en,gaokao2024_I,gaokao2024_II,gaokao2024_mix,gaokao_math_cloze,gaokao_math_qa,gsm8k,math,math500,mawps,minerva_math,mmlu_stem,olympiadbench,sat_math,svamp,tabmwp"
["SMALL_SAMPLES_PASS16"]="math500,aqua,cn_middle_school,gaokao_math_qa,gaokao2023en,gaokao_math_cloze,gaokao2024_I,gaokao2024_II,gaokao2024_mix,minerva_math,sat_math"
["LITTLE_SAMPLES_PASS32"]="OlymMATH-EN-EASY,OlymMATH-EN-HARD,OlymMATH-ZH-EASY,OlymMATH-ZH-HARD,BeyondAIME"
["TINY_SAMPLES_PASS256"]="math24o,aime25,amc24,aime24,amc23"
)

# --- Define the N_SAMPLING value for each set ---
declare -A NSAMPLING_MAP
NSAMPLING_MAP=(
    ["FULL_PASS1"]=1
    ["SMALL_SAMPLES_PASS16"]=16
    ["LITTLE_SAMPLES_PASS32"]=32
    ["TINY_SAMPLES_PASS256"]=256
)

# ==============================================================================
#                  3. SETTINGS & GLOBAL PARAMETERS
# ==============================================================================

# --- Load the active configuration based on ACTIVE_CONFIG_SET ---
BENCHMARKS="${BENCHMARK_SETS[$ACTIVE_CONFIG_SET]}"
DEFAULT_N_SAMPLING="${NSAMPLING_MAP[$ACTIVE_CONFIG_SET]}"

# --- Sanity Check: Ensure the chosen configuration exists ---
if [ -z "$BENCHMARKS" ]; then
    echo "ERROR: Configuration set '$ACTIVE_CONFIG_SET' is not defined in BENCHMARK_SETS."
    echo "Available options are: ${!BENCHMARK_SETS[@]}"
    exit 1
fi

# --- Global Default Parameters ---
# Place all datasets that need to be tested here uniformly

DEFAULT_TEMPLATE="qwen-boxed"
DEFAULT_SPECIFIC_STEPS="120"

TOP_P=0.95
USE_WANDB="false"
CALCULATE_METRICS="false"
METRICS_TO_CALC="Effective Rank"
METRIC_ORDERS="0,1,2"
METRIC_STRIDE=20
RUN_COLLECT_RESULTS="false"

# =======================================================
#               Model and Run Configuration
# =======================================================
# --- Step 1: Define short "aliases" and control execution order with a regular array ---
RUN_ALIASES=(
    "GRPO_VERL"
    "GRPO_ORIGIN"
)

# --- Step 2: Use "aliases" as Keys to Define Configuration Dictionaries ---

# Alias -> Full Run Name
declare -A RUN_NAME_MAP
RUN_NAME_MAP=(
    ["GRPO_VERL"]="mistralai/Mistral-7B-v0.3_GRPOeradv_verl_max_prompt512_max_response1536_gae_batch48_ppomini24_valbatch48_rollout1_logprobbatch12_klcoef0.001_entcoef0.001_epochs2_simplelr_abel_gsm8k_level1_stride40_mgain2.0_auxgw1.0_ema0.3_critic-DeepSeek-R1-Distill-Qwen-1.5B"
    ["GRPO_ORIGIN"]="mistralai/Mathstral-7B-v0.1_GRPOorigin_verl_max_prompt512_max_response1536_grpo_batch48_ppomini24_valbatch48_rollout4_logprobbatch6_klcoef0.001_entcoef0.001_epochs2_simplelr_abel_level3to5_stride40_mgain2.0_auxgw1.0_ema0.3"

)

# Alias -> Base Model

declare -A BASE_MODEL_MAP
BASE_MODEL_MAP=(
    ["GRPO_ORIGIN"]="Mistral-7B-v0.3"
    ["GRPO_VERL"]="Mistral-7B-v0.3"
)

# Alias -> Specific Evaluation Steps

declare -A STEP_MAP
STEP_MAP=(
    ["GRPO_ORIGIN"]="120,160"
    ["GRPO_VERL"]="120,160"
)

# Alias -> Specific Template

declare -A TEMPLATE_MAP
TEMPLATE_MAP=(
    ["GRPO_ORIGIN"]="abel"
    ["GRPO_VERL"]="abel"

)

# Alias -> Maximum Response Length
declare -A MAX_RESPONSE_LENGTH_MAP
MAX_RESPONSE_LENGTH_MAP=(
    ["GRPO_ORIGIN"]=1536
    ["GRPO_VERL"]=1536
)

# =======================================================
#               Nested loop body
# =======================================================

# Outer loop: Iterate through the predefined sequence of the "alias" array.
for alias in "${RUN_ALIASES[@]}"
do
    # --- Retrieve configuration information from various dictionaries using aliases. ---
    run_name="${RUN_NAME_MAP[$alias]}"
    init_model_basename="${BASE_MODEL_MAP[$alias]}"
    max_response_length="${MAX_RESPONSE_LENGTH_MAP[$alias]}"
    
    if [ -z "$run_name" ] || [ -z "$init_model_basename" ] || [ -z "$max_response_length" ]; then
        echo "Warning: Configuration for alias '${alias}' is incomplete. Skipping."
        continue
    fi


    model_family_dir=$(dirname "${run_name}")
    init_model_relative_path="${model_family_dir}/${init_model_basename}"


    current_specific_steps=${STEP_MAP[$alias]:-$DEFAULT_SPECIFIC_STEPS}
    
    add_step_0="false"
    if [[ "${current_specific_steps}" =~ (^|,)0(,|$) ]]; then
        add_step_0="true"
    fi

    current_template=${TEMPLATE_MAP[$alias]:-$DEFAULT_TEMPLATE}

    # Loop 2: Iterate through each temperature value
    for temp in "${TEMPERATURES[@]}"
    do
        current_n_sampling=${DEFAULT_N_SAMPLING}

        echo "========================================================================"
        echo ">>>>>  RUNNING EVALUATION FOR ALIAS: ${alias}"
        echo ">>>>>  Run Name: ${run_name}"
        echo ">>>>>  Template: ${current_template}, N_Sampling: ${current_n_sampling}, STEPS: ${current_specific_steps}"
        echo ">>>>>  Max Response Length: ${max_response_length}"
        echo ">>>>>  Benchmarks: ${BENCHMARKS}"
        echo "========================================================================"

        FINAL_OUTPUT_DIR="eval_results_temp_${temp}_maxlen${max_response_length}_n${current_n_sampling}_calc${CALCULATE_METRICS}"
        
        bash eval_math_nodes.sh \
            --run_name "${run_name}" \
            --template "${current_template}" \
            --init_model "${init_model_relative_path}" \
            --tp_size 1 \
            --add_step_0 ${add_step_0}  \
            --temperature ${temp} \
            --top_p ${TOP_P} \
            --max_tokens ${max_response_length} \
            --benchmarks "${BENCHMARKS}" \
            --n_sampling ${current_n_sampling} \
            --visible_gpus 0 \
            --output_dir "${FINAL_OUTPUT_DIR}" \
            --use_wandb_arg ${USE_WANDB} \
            --calculate_metrics ${CALCULATE_METRICS} \
            --metrics_to_calc "${METRICS_TO_CALC}" \
            --metric_orders "${METRIC_ORDERS}" \
            --metric_stride ${METRIC_STRIDE} \
            --specific_steps "${current_specific_steps}" \
            --num_test_sample_per_dataset -1 \
            --hdfs_home "${HDFS_PATH}" \
            --init_model_base_path "${MODEL_BASE_PATH}" \
            --dtype "${DTYPE}" \
            --run_collect_results "${RUN_COLLECT_RESULTS}" \
            --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION}
    done
done

echo "========================================================================"
echo "All evaluations are complete."
echo "========================================================================"