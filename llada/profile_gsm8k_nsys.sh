#!/bin/bash
set -euo pipefail

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

export CUDA_VISIBLE_DEVICES=0,1

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'

# Use absolute path, avoid cwd change
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_OUTPUT_DIR="${SCRIPT_DIR}/nsys_profiles"
mkdir -p "${PROFILE_OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR}"
echo "TIMESTAMP=${TIMESTAMP}"

run_one () {
  local tag="$1"
  local model_args="$2"
  local out="${PROFILE_OUTPUT_DIR}/${tag}_${TIMESTAMP}"

  echo ""
  echo "==== Profiling ${tag} ===="
  echo "Output prefix: ${out}"

  nsys profile \
    --output="${out}" \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --cudabacktrace=all \
    --stats=true \
    bash -lc "
      accelerate launch --num_processes 2 --mixed_precision no --dynamo_backend no eval_llada.py \
        --tasks ${task} \
        --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code \
        --model llada_dist \
        --model_args ${model_args} \
        --limit 5
    "

  echo "Listing outputs:"
  ls -lh "${out}".* || true

  # Most common target file
  if [[ -f "${out}.nsys-rep" ]]; then
    echo "OK: ${out}.nsys-rep generated"
  else
    echo "WARN: .nsys-rep not found, searching related files:"
    ls -lh "${PROFILE_OUTPUT_DIR}" | grep "${tag}_${TIMESTAMP}" || true
  fi
}

# Baseline
run_one "baseline" "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,enable_nvtx=True"

# Prefix cache + parallel factor
run_one "prefix_cache_parallel_factor" "model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,factor=${factor},show_speed=True,enable_nvtx=True"

echo ""
echo "Done. To view:"
echo "  nsys-ui ${PROFILE_OUTPUT_DIR}/baseline_${TIMESTAMP}.nsys-rep"
echo "  nsys-ui ${PROFILE_OUTPUT_DIR}/prefix_cache_parallel_factor_${TIMESTAMP}.nsys-rep"
