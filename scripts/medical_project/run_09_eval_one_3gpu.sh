#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:-}"
PEFT_PATH="${2:-}"
if [[ -z "${LABEL}" ]]; then
  echo "Usage: bash scripts/medical_project/run_09_eval_one_3gpu.sh LABEL [PEFT_PATH]" >&2
  exit 2
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${PYTHON_BIN:=/home/zzc/miniconda3/envs/lora/bin/python}"
: "${BASE_MODEL:=models/Qwen2.5-7B-Instruct}"
: "${EVAL_OUTPUT_DIR:=outputs/medical_project/eval_3gpu}"
: "${MAX_PPL_SAMPLES:=200}"
: "${MAX_QA_SAMPLES:=100}"
: "${MAX_CEVAL_SAMPLES:=-1}"
: "${MAX_NEW_TOKENS:=256}"
: "${EMBEDDING_MODEL:=none}"
: "${SKIP_CEVAL:=false}"
export CUDA_VISIBLE_DEVICES

LOG_DIR="outputs/medical_project/logs"
LOG_FILE="${LOG_DIR}/eval_${LABEL}.log"
METADATA_FILE="${LOG_DIR}/eval_${LABEL}_timing.json"

EXTRA_EVAL_ARGS=()
if [[ "${SKIP_CEVAL}" == "true" ]]; then
  EXTRA_EVAL_ARGS+=(--skip_ceval)
fi

"${PYTHON_BIN}" tools/medical_project/run_timed_stage.py \
  --stage "eval_${LABEL}" \
  --log "${LOG_FILE}" \
  --metadata "${METADATA_FILE}" \
  -- "${PYTHON_BIN}" tools/medical_project/eval_model_suite.py \
  --label "${LABEL}" \
  --base_model "${BASE_MODEL}" \
  --peft_path "${PEFT_PATH}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --max_ppl_samples "${MAX_PPL_SAMPLES}" \
  --max_qa_samples "${MAX_QA_SAMPLES}" \
  --max_ceval_samples "${MAX_CEVAL_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --embedding_model "${EMBEDDING_MODEL}" \
  --torch_dtype float16 \
  --device_map auto \
  "${EXTRA_EVAL_ARGS[@]}"
