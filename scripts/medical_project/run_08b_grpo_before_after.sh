#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_08b_grpo_before_after.sh

Generate before/after cases for SFT vs GRPO outputs.

Common overrides:
  BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
  BEFORE_PEFT_PATH=outputs/medical_project/sft/qwen25_7b_lora_10k_v100
  AFTER_PEFT_PATH=outputs/medical_project/grpo/qwen25_7b_lora_10k_v100_grpo_500
  INPUT_FILE=data_processed/medical_project/grpo/medical_grpo_500.jsonl
  OUTPUT=outputs/medical_project/eval/grpo_before_after_cases.jsonl
  NUM_CASES=5
  TORCH_DTYPE=float16
EOF
  exit 0
fi

: "${BASE_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${BEFORE_PEFT_PATH:=outputs/medical_project/sft/qwen25_7b_lora_10k_v100}"
: "${AFTER_PEFT_PATH:=outputs/medical_project/grpo/qwen25_7b_lora_10k_v100_grpo_500}"
: "${INPUT_FILE:=data_processed/medical_project/grpo/medical_grpo_500.jsonl}"
: "${OUTPUT:=outputs/medical_project/eval/grpo_before_after_cases.jsonl}"
: "${NUM_CASES:=5}"
: "${MAX_NEW_TOKENS:=512}"
: "${TORCH_DTYPE:=float16}"
: "${DEVICE_MAP:=auto}"

echo "GRPO before/after generation"
echo "BASE_MODEL=${BASE_MODEL}"
echo "BEFORE_PEFT_PATH=${BEFORE_PEFT_PATH}"
echo "AFTER_PEFT_PATH=${AFTER_PEFT_PATH}"
echo "INPUT_FILE=${INPUT_FILE}"
echo "OUTPUT=${OUTPUT}"
echo "NUM_CASES=${NUM_CASES}, MAX_NEW_TOKENS=${MAX_NEW_TOKENS}, TORCH_DTYPE=${TORCH_DTYPE}, DEVICE_MAP=${DEVICE_MAP}"

PYTHONUNBUFFERED=1 python eval/medical_project/generate_grpo_before_after.py \
  --base_model "${BASE_MODEL}" \
  --before_peft_path "${BEFORE_PEFT_PATH}" \
  --after_peft_path "${AFTER_PEFT_PATH}" \
  --input_file "${INPUT_FILE}" \
  --output "${OUTPUT}" \
  --num_cases "${NUM_CASES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --device_map "${DEVICE_MAP}"
