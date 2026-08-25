#!/usr/bin/env bash
set -euo pipefail

# Evaluate every completed adapter in the supplementary matrix with exactly
# the same 500 PPL / 500 generation held-out protocol.

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_13_eval_supplementary_ablations.sh [all|sft|grpo]

Completed adapters are evaluated on 500 held-out PPL rows and 500 held-out
generation rows using max_new_tokens=512. Missing planned adapters are
reported and skipped, so this runner can be reused while training progresses.
EOF
  exit 0
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${PYTHON_BIN:=/home/zzc/miniconda3/envs/lora/bin/python}"
: "${EVAL_OUTPUT_DIR:=outputs/medical_project/eval_supplementary}"
: "${MAX_PPL_SAMPLES:=500}"
: "${MAX_QA_SAMPLES:=500}"
: "${MAX_NEW_TOKENS:=512}"
: "${MAX_CEVAL_SAMPLES:=200}"
: "${EMBEDDING_MODEL:=none}"
: "${CEVAL_DEV_DIR:=data_raw/medical_project/ceval_dev}"

SELECTION="${1:-all}"
if [[ "${SELECTION}" != "all" && "${SELECTION}" != "sft" && "${SELECTION}" != "grpo" ]]; then
  echo "Usage: $0 [all|sft|grpo]" >&2
  exit 2
fi

has_ceval=false
if [[ -d "${CEVAL_DEV_DIR}" ]] && compgen -G "${CEVAL_DEV_DIR}/*.csv" >/dev/null; then
  has_ceval=true
fi

run_one() {
  local group="$1"
  local label="$2"
  local peft_path="$3"
  local skip_ceval=true
  if [[ "${SELECTION}" != "all" && "${SELECTION}" != "${group}" ]]; then
    return 0
  fi
  if [[ ! -f "${peft_path}/adapter_config.json" ]]; then
    echo "Skip ${label}: planned adapter not complete (${peft_path})"
    return 0
  fi
  if [[ "${has_ceval}" == true ]]; then
    skip_ceval=false
  fi

  echo "Unified held-out evaluation: ${label}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
  MAX_PPL_SAMPLES="${MAX_PPL_SAMPLES}" \
  MAX_QA_SAMPLES="${MAX_QA_SAMPLES}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  MAX_CEVAL_SAMPLES="${MAX_CEVAL_SAMPLES}" \
  EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
  SKIP_CEVAL="${skip_ceval}" \
  bash scripts/medical_project/run_09_eval_one_3gpu.sh "${label}" "${peft_path}"
}

run_one sft sft_10k_qvffn_baseline outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn
run_one sft sft_rank4 outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_rank4
run_one sft sft_rank16 outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_rank16
run_one sft sft_lr1e5 outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_lr1e5
run_one sft sft_lr5e5 outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_lr5e5
run_one sft sft_epoch2 outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_epoch2
run_one sft sft_epoch3 outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_epoch3
run_one sft sft_seed17 outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_seed17
run_one sft sft_seed73 outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_seed73

run_one grpo grpo_len256_separate outputs/medical_project/grpo/qwen25_7b_lora_10k_1gpu_qvffn_grpo_500
run_one grpo grpo_len384_separate outputs/medical_project/grpo/qwen25_7b_lora_10k_1gpu_qvffn_grpo_500_len384_separate
run_one grpo grpo_len512_separate outputs/medical_project/grpo/qwen25_7b_lora_10k_1gpu_qvffn_grpo_500_len512_separate
run_one grpo grpo_len384_combined outputs/medical_project/grpo/qwen25_7b_lora_10k_1gpu_qvffn_grpo_500_len384_combined

"${PYTHON_BIN}" tools/medical_project/build_supplementary_ablation_manifest.py
"${PYTHON_BIN}" eval/medical_project/summarize_metrics.py \
  --metrics_dir "${EVAL_OUTPUT_DIR}" \
  --output_csv "${EVAL_OUTPUT_DIR}/summary_metrics.csv" \
  --output_json "${EVAL_OUTPUT_DIR}/summary_metrics.json"
