#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_08_grpo_ablation_3gpu.sh 500|1000

Run medical GRPO with one DDP worker per visible GPU. By default it starts
from the three-GPU 10k SFT LoRA adapter; two-GPU DDP is also supported.

Useful overrides:
  CUDA_VISIBLE_DEVICES=0,1,2
  NPROC_PER_NODE=3
  # Two GPUs: CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2
  PEFT_PATH=outputs/medical_project/sft/qwen25_7b_lora_10k_3gpu
  OUTPUT_DIR=outputs/medical_project/grpo/qwen25_7b_lora_10k_3gpu_grpo_500
  RUN_NAME=grpo_500_3gpu
  REWARD_MODE=separate
  SEED=42
  DATASET_SEED=42
EOF
  exit 0
fi

SIZE="${1:-}"
if [[ "${SIZE}" != "500" && "${SIZE}" != "1000" ]]; then
  echo "Usage: bash scripts/medical_project/run_08_grpo_ablation_3gpu.sh 500|1000" >&2
  exit 2
fi

: "${CUDA_VISIBLE_DEVICES:=0,1,2}"
: "${NPROC_PER_NODE:=3}"
: "${PYTHON_BIN:=/home/zzc/miniconda3/envs/lora/bin/python}"
: "${TORCHRUN_BIN:=${PYTHON_BIN%/python}/torchrun}"
: "${MODEL_NAME_OR_PATH:=models/Qwen2.5-7B-Instruct}"
: "${PEFT_PATH:=outputs/medical_project/sft/qwen25_7b_lora_10k_${NPROC_PER_NODE}gpu}"
: "${TRAIN_FILE:=data_processed/medical_project/grpo/medical_grpo_${SIZE}.jsonl}"
: "${OUTPUT_DIR:=outputs/medical_project/grpo/qwen25_7b_lora_10k_${NPROC_PER_NODE}gpu_grpo_${SIZE}}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${NUM_GENERATIONS:=2}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=2}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${MAX_COMPLETION_LENGTH:=256}"
: "${SIMILARITY_MODEL:=none}"
: "${SIMILARITY_DEVICE:=cpu}"
: "${REPORT_TO:=none}"
: "${LORA_TARGET_MODULES:=q_proj v_proj}"
: "${REWARD_MODE:=separate}"
: "${SEED:=42}"
: "${DATASET_SEED:=${SEED}}"
: "${RUN_NAME:=grpo_${SIZE}_${NPROC_PER_NODE}gpu}"

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

if [[ ! -d "${PEFT_PATH}" ]]; then
  echo "SFT adapter not found: ${PEFT_PATH}" >&2
  exit 2
fi

LOG_DIR="outputs/medical_project/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
METADATA_FILE="${LOG_DIR}/${RUN_NAME}_timing.json"

echo "DDP launch: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, workers=${NPROC_PER_NODE}"
echo "SFT adapter: ${PEFT_PATH}; training data: ${TRAIN_FILE}; output: ${OUTPUT_DIR}"

"${PYTHON_BIN}" tools/medical_project/run_timed_stage.py \
  --stage "${RUN_NAME}" \
  --log "${LOG_FILE}" \
  --metadata "${METADATA_FILE}" \
  -- "${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" training/grpo_training.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --peft_path "${PEFT_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --train_samples "${SIZE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --output_dir "${OUTPUT_DIR}" \
  --dtype float16 \
  --fp16 True \
  --bf16 False \
  --report_to "${REPORT_TO}" \
  --remove_unused_columns False \
  --gradient_checkpointing False \
  --beta 0.001 \
  --learning_rate 5.0e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --use_vllm False \
  --logging_steps 1 \
  --eval_strategy no \
  --save_strategy epoch \
  --save_total_limit 1 \
  --use_peft True \
  --qlora False \
  --load_in_4bit False \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --num_generations "${NUM_GENERATIONS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --preprocessing_num_workers 4 \
  --ddp_find_unused_parameters False \
  --seed "${SEED}" \
  --reward_type medical \
  --reward_mode "${REWARD_MODE}" \
  --dataset_seed "${DATASET_SEED}" \
  --similarity_model "${SIMILARITY_MODEL}" \
  --similarity_device "${SIMILARITY_DEVICE}"
