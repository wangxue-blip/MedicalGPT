#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_04_sft_ablation_3gpu.sh 1k|10k|30k

Run Qwen2.5-7B medical LoRA SFT with one DDP worker per visible GPU.
The default configuration uses three GPUs; two-GPU DDP is also supported.

Useful overrides:
  CUDA_VISIBLE_DEVICES=0,1,2
  NPROC_PER_NODE=3
  # Two GPUs: CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2
  OUTPUT_DIR=outputs/medical_project/sft/qwen25_7b_lora_1k_3gpu
  MAX_TRAIN_SAMPLES=-1
  MAX_EVAL_SAMPLES=100
  NUM_TRAIN_EPOCHS=1
  SEED=42
  DATA_SEED=42
  RUN_NAME=sft_1k_3gpu
EOF
  exit 0
fi

SIZE="${1:-}"
if [[ "${SIZE}" != "1k" && "${SIZE}" != "10k" && "${SIZE}" != "30k" ]]; then
  echo "Usage: bash scripts/medical_project/run_04_sft_ablation_3gpu.sh 1k|10k|30k" >&2
  exit 2
fi

: "${CUDA_VISIBLE_DEVICES:=0,1,2}"
: "${NPROC_PER_NODE:=3}"
: "${PYTHON_BIN:=/home/zzc/miniconda3/envs/lora/bin/python}"
: "${TORCHRUN_BIN:=${PYTHON_BIN%/python}/torchrun}"
: "${MODEL_NAME_OR_PATH:=models/Qwen2.5-7B-Instruct}"
: "${VALIDATION_FILE_DIR:=data_processed/medical_project/eval/sharegpt_validation}"
: "${MODEL_MAX_LENGTH:=1024}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${MAX_TRAIN_SAMPLES:=-1}"
: "${MAX_EVAL_SAMPLES:=100}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=8}"
: "${LEARNING_RATE:=2e-5}"
: "${LORA_RANK:=8}"
: "${LORA_ALPHA:=16}"
: "${LORA_DROPOUT:=0.05}"
: "${TARGET_MODULES:=q_proj,v_proj}"
: "${PREPROCESSING_NUM_WORKERS:=4}"
: "${REPORT_TO:=none}"
: "${LOGGING_STEPS:=1}"
: "${SEED:=42}"
: "${DATA_SEED:=${SEED}}"
: "${RUN_NAME:=sft_${SIZE}_${NPROC_PER_NODE}gpu}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Training Python or torchrun is not executable: PYTHON_BIN=${PYTHON_BIN}, TORCHRUN_BIN=${TORCHRUN_BIN}" >&2
  exit 2
fi

IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_GPUS[@]}" -ne "${NPROC_PER_NODE}" ]]; then
  echo "CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_GPUS[@]} device(s), but NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 2
fi

DETECTED_GPUS="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${DETECTED_GPUS}" -ne "${NPROC_PER_NODE}" ]]; then
  echo "PyTorch detects ${DETECTED_GPUS} CUDA device(s), expected ${NPROC_PER_NODE}." >&2
  exit 2
fi

TRAIN_FILE_DIR="data_processed/medical_project/sft/sharegpt_${SIZE}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/medical_project/sft/qwen25_7b_lora_${SIZE}_${NPROC_PER_NODE}gpu}"
LOG_DIR="outputs/medical_project/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
METADATA_FILE="${LOG_DIR}/${RUN_NAME}_timing.json"

echo "DDP launch: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, workers=${NPROC_PER_NODE}"
echo "Training data: ${TRAIN_FILE_DIR}; output: ${OUTPUT_DIR}"

"${PYTHON_BIN}" tools/medical_project/run_timed_stage.py \
  --stage "${RUN_NAME}" \
  --log "${LOG_FILE}" \
  --metadata "${METADATA_FILE}" \
  -- "${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" training/supervised_finetuning.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --device_map none \
  --train_file_dir "${TRAIN_FILE_DIR}" \
  --validation_file_dir "${VALIDATION_FILE_DIR}" \
  --do_train --do_eval --use_peft True \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --logging_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --logging_first_step True \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --dataloader_num_workers 2 \
  --output_dir "${OUTPUT_DIR}" \
  --target_modules "${TARGET_MODULES}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --torch_dtype float16 \
  --fp16 \
  --report_to "${REPORT_TO}" \
  --gradient_checkpointing True \
  --ddp_find_unused_parameters False \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --include_tokens_per_second True \
  --cache_dir ./cache \
  --flash_attn False
