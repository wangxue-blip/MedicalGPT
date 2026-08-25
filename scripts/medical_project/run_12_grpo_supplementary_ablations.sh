#!/usr/bin/env bash
set -euo pipefail

# Compare GRPO generation lengths and reward composition while fixing the SFT
# start point, GRPO data, number of updates, seed, and similarity backend.

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_12_grpo_supplementary_ablations.sh [all|length384|length512|reward_combined]

The existing 10k SFT -> GRPO-500 run at max_completion_length=256 is the
length baseline. This script adds 384 and 512, plus the documented weighted
composite reward at 384 (paired with the separate-reward 384 run).

All new runs use 10k Q/V+FFN SFT, 500 GRPO rows, one epoch, seed 42, and the
same char n-gram fallback similarity signal (SIMILARITY_MODEL=none).
EOF
  exit 0
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NPROC_PER_NODE:=1}"
: "${PYTHON_BIN:=/home/zzc/miniconda3/envs/lora/bin/python}"
: "${FORCE_RERUN:=false}"
: "${START_SFT_OUTPUT:=outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn}"
: "${SIMILARITY_MODEL:=none}"

SELECTION="${1:-all}"
VALID_SELECTIONS=(all length384 length512 reward_combined)
if [[ ! " ${VALID_SELECTIONS[*]} " =~ " ${SELECTION} " ]]; then
  echo "Unknown selection: ${SELECTION}" >&2
  exit 2
fi

if [[ ! -f "${START_SFT_OUTPUT}/adapter_config.json" ]]; then
  echo "Required 10k Q/V+FFN SFT adapter is missing: ${START_SFT_OUTPUT}" >&2
  exit 2
fi

run_case() {
  local label="$1"
  local max_completion_length="$2"
  local reward_mode="$3"
  local output_dir="outputs/medical_project/grpo/qwen25_7b_lora_10k_${NPROC_PER_NODE}gpu_qvffn_grpo_500_${label}"
  local run_name="grpo_10k_qvffn_500_${label}_${NPROC_PER_NODE}gpu"

  if [[ -f "${output_dir}/adapter_config.json" && "${FORCE_RERUN}" != "true" ]]; then
    echo "Skip ${label}: completed adapter exists at ${output_dir}"
    return 0
  fi

  echo "Starting GRPO supplementary ablation: ${label}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  PEFT_PATH="${START_SFT_OUTPUT}" \
  OUTPUT_DIR="${output_dir}" \
  RUN_NAME="${run_name}" \
  MAX_COMPLETION_LENGTH="${max_completion_length}" \
  REWARD_MODE="${reward_mode}" \
  SEED=42 \
  DATASET_SEED=42 \
  SIMILARITY_MODEL="${SIMILARITY_MODEL}" \
  bash scripts/medical_project/run_08_grpo_ablation_3gpu.sh 500
}

case "${SELECTION}" in
  length384) run_case len384_separate 384 separate ;;
  length512) run_case len512_separate 512 separate ;;
  reward_combined) run_case len384_combined 384 combined ;;
  all)
    run_case len384_separate 384 separate
    run_case len512_separate 512 separate
    run_case len384_combined 384 combined
    ;;
esac

"${PYTHON_BIN}" tools/medical_project/build_supplementary_ablation_manifest.py
