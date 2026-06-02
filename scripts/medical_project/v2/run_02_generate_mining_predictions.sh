#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/v2/run_02_generate_mining_predictions.sh

Generate v2 mining predictions for Base, SFT10k, and SFT30k.
This script loads 7B models and should be run manually on a GPU server.

Common overrides:
  CUDA_VISIBLE_DEVICES=0
  INPUT=data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl
  OUTPUT_DIR=outputs/medical_project/v2/mining
  BASE_MODEL=models/Qwen2.5-7B-Instruct
  TORCH_DTYPE=float16
  DEVICE_MAP=auto
  MAX_SAMPLES=-1
  MAX_NEW_TOKENS=512
  MODEL_MAX_LENGTH=1536
  TEMPERATURE=0.2
  TOP_P=0.9
  REPETITION_PENALTY=1.05
  RUN_BASE=true
  RUN_SFT10K=true
  RUN_SFT30K=true
  DRY_RUN=false
EOF
  exit 0
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${INPUT:=data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl}"
: "${OUTPUT_DIR:=outputs/medical_project/v2/mining}"
: "${BASE_MODEL:=models/Qwen2.5-7B-Instruct}"
: "${TORCH_DTYPE:=float16}"
: "${DEVICE_MAP:=auto}"
: "${MAX_SAMPLES:=-1}"
: "${MAX_NEW_TOKENS:=512}"
: "${MODEL_MAX_LENGTH:=1536}"
: "${TEMPERATURE:=0.2}"
: "${TOP_P:=0.9}"
: "${REPETITION_PENALTY:=1.05}"
: "${SEED:=42}"
: "${RESUME:=true}"
: "${CONTINUE_ON_ERROR:=false}"
: "${DRY_RUN:=false}"

: "${RUN_BASE:=true}"
: "${RUN_SFT10K:=true}"
: "${RUN_SFT30K:=true}"

: "${SFT10K_PEFT:=outputs/medical_project/sft/qwen25_7b_lora_10k_v100}"
: "${SFT30K_PEFT:=outputs/medical_project/sft/qwen25_7b_lora_30k_v100}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}"

run_model() {
  local label="$1"
  local peft_path="${2:-}"
  local peft_args=()
  if [[ -n "${peft_path}" ]]; then
    if [[ ! -f "${peft_path}/adapter_config.json" ]]; then
      echo "Missing adapter_config.json for ${label}: ${peft_path}" >&2
      return 1
    fi
    peft_args=(--peft_path "${peft_path}")
  fi

  echo "Generating mining predictions for ${label}"
  python tools/medical_project/v2/generate_mining_predictions.py \
    --model_name_or_path "${BASE_MODEL}" \
    "${peft_args[@]}" \
    --model_label "${label}" \
    --input "${INPUT}" \
    --output "${OUTPUT_DIR}/predictions_${label}.jsonl" \
    --summary_output "${OUTPUT_DIR}/predictions_${label}_summary.json" \
    --max_samples "${MAX_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --device_map "${DEVICE_MAP}" \
    --resume "${RESUME}" \
    --continue_on_error "${CONTINUE_ON_ERROR}" \
    --seed "${SEED}" \
    --dry_run "${DRY_RUN}"
}

if [[ "${RUN_BASE}" == "true" ]]; then
  run_model "base" ""
fi
if [[ "${RUN_SFT10K}" == "true" ]]; then
  run_model "sft_10k" "${SFT10K_PEFT}"
fi
if [[ "${RUN_SFT30K}" == "true" ]]; then
  run_model "sft_30k" "${SFT30K_PEFT}"
fi
