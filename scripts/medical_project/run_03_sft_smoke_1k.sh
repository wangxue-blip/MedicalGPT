#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_03_sft_smoke_1k.sh

Run the 1k SFT 300-step smoke test for the medical project.

Defaults are V100-friendly:
  - fp16
  - flash_attn=False
  - model_max_length=1024
  - LoRA rank=8

Useful overrides:
  CUDA_VISIBLE_DEVICES=0
  NPROC_PER_NODE=1
  MEDICAL_SFT_PROFILE=v100|a100|auto
  MODEL_NAME_OR_PATH=models/Qwen2.5-7B-Instruct
  TORCHRUN_BIN=torchrun
  MAX_STEPS=300
  LOGGING_STEPS=10
  EVAL_STEPS=100
  SAVE_STEPS=200
  OUTPUT_DIR=outputs/medical_project/sft/qwen25_7b_lora_smoke_1k
EOF
  exit 0
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

: "${NPROC_PER_NODE:=1}"
: "${MEDICAL_SFT_PROFILE:=auto}"
: "${TORCHRUN_BIN:=torchrun}"
: "${MODEL_NAME_OR_PATH:=models/Qwen2.5-7B-Instruct}"
: "${TRAIN_FILE_DIR:=data_processed/medical_project/sft/sharegpt_1k}"
: "${VALIDATION_FILE_DIR:=data_processed/medical_project/sft/sharegpt_1k}"
: "${OUTPUT_DIR:=outputs/medical_project/sft/qwen25_7b_lora_smoke_1k}"
: "${MAX_TRAIN_SAMPLES:=1000}"
: "${MAX_EVAL_SAMPLES:=100}"
: "${MODEL_MAX_LENGTH:=1024}"
: "${MAX_STEPS:=300}"
: "${LOGGING_STEPS:=10}"
: "${EVAL_STEPS:=100}"
: "${SAVE_STEPS:=200}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=8}"
: "${LEARNING_RATE:=2e-5}"
: "${LORA_RANK:=8}"
: "${LORA_ALPHA:=16}"
: "${LORA_DROPOUT:=0.05}"
: "${TARGET_MODULES:=q_proj,v_proj}"
: "${PREPROCESSING_NUM_WORKERS:=4}"

if [[ "${MEDICAL_SFT_PROFILE}" == "auto" ]]; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  if [[ "${GPU_NAME}" == *"A100"* || "${GPU_NAME}" == *"H100"* || "${GPU_NAME}" == *"H200"* ]]; then
    MEDICAL_SFT_PROFILE="a100"
  else
    MEDICAL_SFT_PROFILE="v100"
  fi
fi

if [[ "${MEDICAL_SFT_PROFILE}" == "a100" ]]; then
  PRECISION_ARGS=(--torch_dtype bfloat16 --bf16)
  : "${FLASH_ATTN:=True}"
else
  PRECISION_ARGS=(--torch_dtype float16 --fp16)
  : "${FLASH_ATTN:=False}"
fi

echo "Medical SFT smoke profile: ${MEDICAL_SFT_PROFILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_STEPS=${MAX_STEPS}, LOGGING_STEPS=${LOGGING_STEPS}, EVAL_STEPS=${EVAL_STEPS}, SAVE_STEPS=${SAVE_STEPS}"

PYTHONUNBUFFERED=1 "${TORCHRUN_BIN}" --nproc_per_node "${NPROC_PER_NODE}" training/supervised_finetuning.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --train_file_dir "${TRAIN_FILE_DIR}" \
  --validation_file_dir "${VALIDATION_FILE_DIR}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --logging_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --eval_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_strategy steps \
  --save_total_limit 3 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --target_modules "${TARGET_MODULES}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  "${PRECISION_ARGS[@]}" \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --cache_dir ./cache \
  --flash_attn "${FLASH_ATTN}"
