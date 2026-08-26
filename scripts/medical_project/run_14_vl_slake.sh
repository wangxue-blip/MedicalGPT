#!/usr/bin/env bash
set -euo pipefail

# First Qwen2.5-VL medical experiment:
# 2,048 Chinese SLAKE image QA rows + 512 existing text-only MedicalGPT SFT rows.
# Base and LoRA-adapter performance are both measured on the untouched official
# Chinese SLAKE test split.

MODE="${1:-all}"
if [[ "${MODE}" != "prepare" && "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" ]]; then
  echo "Usage: bash scripts/medical_project/run_14_vl_slake.sh [prepare|train|eval|all]" >&2
  exit 2
fi

: "${CUDA_VISIBLE_DEVICES:=3}"
: "${PYTHON_BIN:=/home/zzc/miniconda3/envs/lora/bin/python}"
: "${MODEL_NAME_OR_PATH:=models/Qwen2.5-VL-3B-Instruct}"
: "${VL_DATA_DIR:=data_processed/medical_project/vl/slake_zh_textmix}"
: "${OUTPUT_DIR:=outputs/medical_project/vl/qwen25_vl_3b_slake_zh_textmix_lora_r8}"
: "${EVAL_OUTPUT_DIR:=outputs/medical_project/vl_eval}"
: "${MAX_VL_TRAIN_SAMPLES:=2048}"
: "${MAX_TEXT_TRAIN_SAMPLES:=512}"
: "${MAX_TRAIN_SAMPLES:=-1}"
: "${MAX_EVAL_SAMPLES:=256}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=8}"
: "${LEARNING_RATE:=2e-5}"
: "${LORA_RANK:=8}"
: "${LORA_ALPHA:=16}"
: "${MAX_NEW_TOKENS:=32}"
: "${MAX_TEST_SAMPLES:=-1}"
: "${SEED:=42}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

if [[ "${MODE}" == "prepare" || "${MODE}" == "all" ]]; then
  "${PYTHON_BIN}" tools/medical_project/prepare_slake_vl_data.py \
    --output_dir "${VL_DATA_DIR}" \
    --max_vl_train_samples "${MAX_VL_TRAIN_SAMPLES}" \
    --max_text_train_samples "${MAX_TEXT_TRAIN_SAMPLES}" \
    --seed "${SEED}"
fi

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "Qwen2.5-VL model is missing: ${MODEL_NAME_OR_PATH}" >&2
  echo "Download Qwen/Qwen2.5-VL-3B-Instruct to that directory before running train/eval." >&2
  exit 2
fi

if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  "${PYTHON_BIN}" training/supervised_finetuning_vl.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --train_file "${VL_DATA_DIR}/train.jsonl" \
    --validation_file "${VL_DATA_DIR}/validation.jsonl" \
    --output_dir "${OUTPUT_DIR}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --max_eval_samples "${MAX_EVAL_SAMPLES}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --target_modules q_proj,v_proj \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --gradient_checkpointing \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --cache_dir ./cache \
    --report_to none
fi

if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then
  "${PYTHON_BIN}" eval/medical_project/eval_vl_vqa.py \
    --label qwen25_vl_3b_base_slake_zh \
    --base_model "${MODEL_NAME_OR_PATH}" \
    --eval_file "${VL_DATA_DIR}/test.jsonl" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --max_samples "${MAX_TEST_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --torch_dtype bfloat16 \
    --cache_dir ./cache
  "${PYTHON_BIN}" eval/medical_project/eval_vl_vqa.py \
    --label qwen25_vl_3b_lora_slake_zh \
    --base_model "${MODEL_NAME_OR_PATH}" \
    --peft_path "${OUTPUT_DIR}" \
    --eval_file "${VL_DATA_DIR}/test.jsonl" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --max_samples "${MAX_TEST_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --torch_dtype bfloat16 \
    --cache_dir ./cache
fi
