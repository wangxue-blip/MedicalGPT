#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_08_grpo_train.sh

Run the small medical GRPO training experiment.

Common overrides:
  MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
  PEFT_PATH=outputs/medical_project/sft/qwen25_7b_lora_10k
  TRAIN_FILE=data_processed/medical_project/grpo/medical_grpo_500.jsonl
  TRAIN_SAMPLES=500
  OUTPUT_DIR=outputs/medical_project/grpo/qwen25_7b_medical_grpo_500
  CUDA_VISIBLE_DEVICES=0
  NPROC_PER_NODE=1
  GRPO_DTYPE=float16
  GRPO_FP16=True
  GRPO_BF16=False
  NUM_GENERATIONS=2
  PER_DEVICE_TRAIN_BATCH_SIZE=2
  STEPS_PER_GENERATION=2
  SIMILARITY_MODEL=BAAI/bge-small-zh-v1.5
  SIMILARITY_DEVICE=cpu
EOF
  exit 0
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NPROC_PER_NODE:=1}"
: "${MODEL_NAME_OR_PATH:=Qwen/Qwen2.5-7B-Instruct}"
: "${PEFT_PATH:=outputs/medical_project/sft/qwen25_7b_lora_10k}"
: "${TRAIN_FILE:=data_processed/medical_project/grpo/medical_grpo_500.jsonl}"
: "${TRAIN_SAMPLES:=500}"
: "${OUTPUT_DIR:=outputs/medical_project/grpo/qwen25_7b_medical_grpo_500}"
: "${SIMILARITY_MODEL:=BAAI/bge-small-zh-v1.5}"
: "${SIMILARITY_DEVICE:=cpu}"
: "${GRPO_DTYPE:=float16}"
: "${GRPO_FP16:=True}"
: "${GRPO_BF16:=False}"
: "${NUM_GENERATIONS:=2}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=2}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${STEPS_PER_GENERATION:=}"
: "${MAX_COMPLETION_LENGTH:=512}"
: "${PREPROCESSING_NUM_WORKERS:=4}"
export CUDA_VISIBLE_DEVICES

peft_args=()
if [[ -n "${PEFT_PATH}" ]]; then
  peft_args=(--peft_path "${PEFT_PATH}")
fi

generation_args=()
if [[ -n "${STEPS_PER_GENERATION}" ]]; then
  generation_args=(--steps_per_generation "${STEPS_PER_GENERATION}")
fi

torchrun --nproc_per_node "${NPROC_PER_NODE}" training/grpo_training.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  "${peft_args[@]}" \
  --train_file "${TRAIN_FILE}" \
  --train_samples "${TRAIN_SAMPLES}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --output_dir "${OUTPUT_DIR}" \
  --dtype "${GRPO_DTYPE}" \
  --fp16 "${GRPO_FP16}" \
  --bf16 "${GRPO_BF16}" \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing False \
  --beta 0.001 \
  --learning_rate 5.0e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --use_vllm False \
  --logging_steps 5 \
  --eval_strategy no \
  --save_steps 100 \
  --save_strategy steps \
  --save_total_limit 3 \
  --use_peft True \
  --qlora False \
  --load_in_4bit False \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --num_generations "${NUM_GENERATIONS}" \
  "${generation_args[@]}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --reward_type medical \
  --similarity_model "${SIMILARITY_MODEL}" \
  --similarity_device "${SIMILARITY_DEVICE}"
