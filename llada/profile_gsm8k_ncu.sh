#!/bin/bash
# NCU (Nsight Compute) Profiling script for GSM8K evaluation
# Profiles: 1) Baseline, 2) Prefix Cache + Parallel Factor
# NCU provides detailed kernel-level profiling

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=0,1

# Configuration
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'

# NCU output directory
PROFILE_OUTPUT_DIR="./ncu_profiles"
mkdir -p ${PROFILE_OUTPUT_DIR}

# Timestamp for unique filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# NCU profiling options
# --set full: Collect all available metrics
# --kernel-name-base: Use base kernel names
# --launch-skip: Skip first N kernel launches (warmup)
# --launch-count: Number of kernels to profile
# Note: Reduce --launch-count for faster profiling, increase for more comprehensive analysis
LAUNCH_SKIP=100      # Skip warmup kernels
LAUNCH_COUNT=50      # Profile this many kernels

echo "=========================================="
echo "NCU Profiling GSM8K Evaluation"
echo "Model: ${model_path}"
echo "Task: ${task}"
echo "Gen Length: ${length}"
echo "Block Length: ${block_length}"
echo "Steps (baseline): ${length}"
echo "Steps (parallel): ${steps}"
echo "Factor: ${factor}"
echo "Output Directory: ${PROFILE_OUTPUT_DIR}"
echo "Launch Skip: ${LAUNCH_SKIP}"
echo "Launch Count: ${LAUNCH_COUNT}"
echo "=========================================="

# ============================================
# Profile 1: Baseline
# ============================================
echo ""
echo "[1/2] NCU Profiling BASELINE configuration..."
echo "Steps: ${length}, No cache, No parallel factor"

ncu \
    -o ${PROFILE_OUTPUT_DIR}/baseline_${TIMESTAMP} \
    --force-overwrite \
    --set full \
    --nvtx \
    --nvtx-include "generate_baseline/,model_forward/" \
    --launch-skip ${LAUNCH_SKIP} \
    --launch-count ${LAUNCH_COUNT} \
    --kernel-name-base function \
    --target-processes all \
    python eval_llada.py \
        --tasks ${task} \
        --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code \
        --model llada_dist \
        --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,enable_nvtx=True \
        --limit 2

echo "Baseline NCU profiling completed: ${PROFILE_OUTPUT_DIR}/baseline_${TIMESTAMP}.ncu-rep"

# ============================================
# Profile 2: Prefix Cache + Parallel Factor
# ============================================
echo ""
echo "[2/2] NCU Profiling PREFIX CACHE + PARALLEL FACTOR configuration..."
echo "Steps: ${steps}, Cache: True, Factor: ${factor}"

ncu \
    -o ${PROFILE_OUTPUT_DIR}/baseline_${TIMESTAMP} \
    --force-overwrite \
    --set full \
    --nvtx \
    --nvtx-include "generate_prefix_cache/,model_forward_cached/" \
    --launch-skip ${LAUNCH_SKIP} \
    --launch-count ${LAUNCH_COUNT} \
    --kernel-name-base function \
    --target-processes all \
    python eval_llada.py \
        --tasks ${task} \
        --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code \
        --model llada_dist \
        --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,factor=${factor},show_speed=True,enable_nvtx=True \
        --limit 2

echo "Prefix Cache + Parallel Factor NCU profiling completed: ${PROFILE_OUTPUT_DIR}/prefix_cache_parallel_factor_${TIMESTAMP}.ncu-rep"

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "NCU Profiling Complete!"
echo "=========================================="
echo "Generated profile files:"
echo "  1. Baseline: ${PROFILE_OUTPUT_DIR}/baseline_${TIMESTAMP}.ncu-rep"
echo "  2. Prefix Cache + Parallel Factor: ${PROFILE_OUTPUT_DIR}/prefix_cache_parallel_factor_${TIMESTAMP}.ncu-rep"
echo ""
echo "To view the profiles, run:"
echo "  ncu-ui ${PROFILE_OUTPUT_DIR}/baseline_${TIMESTAMP}.ncu-rep"
echo "  ncu-ui ${PROFILE_OUTPUT_DIR}/prefix_cache_parallel_factor_${TIMESTAMP}.ncu-rep"
echo ""
echo "To generate summary reports:"
echo "  ncu --import ${PROFILE_OUTPUT_DIR}/baseline_${TIMESTAMP}.ncu-rep --page raw"
echo "  ncu --import ${PROFILE_OUTPUT_DIR}/prefix_cache_parallel_factor_${TIMESTAMP}.ncu-rep --page raw"
echo ""
echo "Common analysis pages: raw, details, source, launch, occupancy, memory, compute"
