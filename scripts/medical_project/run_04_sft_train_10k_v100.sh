#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_04_sft_train_10k_v100.sh

Run the 10k medical SFT training job on V100 32GB.

Default V100 profile:
  - fp16
  - flash_attn=False
  - LoRA, with QLoRA available as an override
  - output path uses a _v100 suffix to avoid A100 job conflicts

Useful overrides for job submission:
  CUDA_VISIBLE_DEVICES=0
  NPROC_PER_NODE=1
  MODEL_NAME_OR_PATH=models/Qwen2.5-7B-Instruct
  OUTPUT_DIR=outputs/medical_project/sft/qwen25_7b_lora_10k_v100
  NUM_TRAIN_EPOCHS=1
  MODEL_MAX_LENGTH=1024
  PER_DEVICE_TRAIN_BATCH_SIZE=1
  GRADIENT_ACCUMULATION_STEPS=8
  LEARNING_RATE=2e-5
  SFT_METHOD=lora|qlora
EOF
  exit 0
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

: "${NPROC_PER_NODE:=1}"
: "${TORCHRUN_BIN:=torchrun}"
: "${MODEL_NAME_OR_PATH:=models/Qwen2.5-7B-Instruct}"
: "${TRAIN_FILE_DIR:=data_processed/medical_project/sft/sharegpt_10k}"
: "${VALIDATION_FILE_DIR:=data_processed/medical_project/sft/sharegpt_10k}"
: "${OUTPUT_DIR:=outputs/medical_project/sft/qwen25_7b_lora_10k_v100}"
: "${MODEL_MAX_LENGTH:=1024}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${MAX_TRAIN_SAMPLES:=-1}"
: "${MAX_EVAL_SAMPLES:=500}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=8}"
: "${LEARNING_RATE:=2e-5}"
: "${WARMUP_RATIO:=0.03}"
: "${WEIGHT_DECAY:=0.01}"
: "${LOGGING_STEPS:=20}"
: "${EVAL_STEPS:=500}"
: "${SAVE_STEPS:=500}"
: "${SAVE_TOTAL_LIMIT:=3}"
: "${LORA_RANK:=8}"
: "${LORA_ALPHA:=16}"
: "${LORA_DROPOUT:=0.05}"
: "${TARGET_MODULES:=q_proj,v_proj}"
: "${PREPROCESSING_NUM_WORKERS:=4}"
: "${SFT_METHOD:=lora}"
: "${FLASH_ATTN:=False}"

PRECISION_ARGS=(--torch_dtype float16 --fp16)

QUANT_ARGS=()
if [[ "${SFT_METHOD}" == "qlora" ]]; then
  QUANT_ARGS=(--qlora True --load_in_4bit True)
  OUTPUT_DIR="${OUTPUT_DIR/_lora_/_qlora_}"
elif [[ "${SFT_METHOD}" != "lora" ]]; then
  echo "Unsupported SFT_METHOD=${SFT_METHOD}; use lora or qlora." >&2
  exit 2
fi

echo "Medical SFT 10k V100 profile: method=${SFT_METHOD}, fp16, flash_attn=${FLASH_ATTN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "TRAIN_FILE_DIR=${TRAIN_FILE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

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
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --logging_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --eval_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_strategy steps \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --target_modules "${TARGET_MODULES}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  "${QUANT_ARGS[@]}" \
  "${PRECISION_ARGS[@]}" \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --cache_dir ./cache \
  --flash_attn "${FLASH_ATTN}"
