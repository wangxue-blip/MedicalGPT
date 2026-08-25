#!/usr/bin/env bash
set -euo pipefail

# Run the missing SFT ablations as single-factor comparisons against the
# existing 10k Q/V+FFN, r=8, lr=2e-5, one-epoch, seed=42 baseline.

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_11_sft_supplementary_ablations.sh [all|rank4|rank16|lr1e5|lr5e5|epoch2|epoch3|seed17|seed73]

Runs each selected experiment sequentially. The default `all` covers every
missing SFT parameter/seed ablation. Existing completed adapters are skipped.

Default controlled configuration:
  10k samples; q_proj,v_proj,gate_proj,up_proj,down_proj; r=8; alpha=16;
  lr=2e-5; one epoch; seed=42; 500 held-out validation samples.

Useful overrides:
  CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1
  MAX_EVAL_SAMPLES=500
  FORCE_RERUN=true  # only after manually removing/renaming an existing output
EOF
  exit 0
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${NPROC_PER_NODE:=1}"
: "${PYTHON_BIN:=/home/zzc/miniconda3/envs/lora/bin/python}"
: "${MAX_EVAL_SAMPLES:=500}"
: "${FORCE_RERUN:=false}"
: "${TARGET_MODULES:=q_proj,v_proj,gate_proj,up_proj,down_proj}"

SELECTION="${1:-all}"
VALID_SELECTIONS=(all rank4 rank16 lr1e5 lr5e5 epoch2 epoch3 seed17 seed73)
if [[ ! " ${VALID_SELECTIONS[*]} " =~ " ${SELECTION} " ]]; then
  echo "Unknown selection: ${SELECTION}" >&2
  exit 2
fi

run_case() {
  local label="$1"
  local lora_rank="$2"
  local learning_rate="$3"
  local epochs="$4"
  local seed="$5"
  local output_dir="outputs/medical_project/sft/qwen25_7b_lora_10k_${NPROC_PER_NODE}gpu_qvffn_${label}"
  local run_name="sft_10k_qvffn_${label}_${NPROC_PER_NODE}gpu"

  if [[ -f "${output_dir}/adapter_config.json" && "${FORCE_RERUN}" != "true" ]]; then
    echo "Skip ${label}: completed adapter exists at ${output_dir}"
    return 0
  fi

  echo "Starting SFT supplementary ablation: ${label}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  OUTPUT_DIR="${output_dir}" \
  RUN_NAME="${run_name}" \
  TARGET_MODULES="${TARGET_MODULES}" \
  LORA_RANK="${lora_rank}" \
  LORA_ALPHA=16 \
  LEARNING_RATE="${learning_rate}" \
  NUM_TRAIN_EPOCHS="${epochs}" \
  SEED="${seed}" \
  DATA_SEED="${seed}" \
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES}" \
  bash scripts/medical_project/run_04_sft_ablation_3gpu.sh 10k
}

case "${SELECTION}" in
  rank4) run_case rank4 4 2e-5 1 42 ;;
  rank16) run_case rank16 16 2e-5 1 42 ;;
  lr1e5) run_case lr1e5 8 1e-5 1 42 ;;
  lr5e5) run_case lr5e5 8 5e-5 1 42 ;;
  epoch2) run_case epoch2 8 2e-5 2 42 ;;
  epoch3) run_case epoch3 8 2e-5 3 42 ;;
  seed17) run_case seed17 8 2e-5 1 17 ;;
  seed73) run_case seed73 8 2e-5 1 73 ;;
  all)
    run_case rank4 4 2e-5 1 42
    run_case rank16 16 2e-5 1 42
    run_case lr1e5 8 1e-5 1 42
    run_case lr5e5 8 5e-5 1 42
    run_case epoch2 8 2e-5 2 42
    run_case epoch3 8 2e-5 3 42
    run_case seed17 8 2e-5 1 17
    run_case seed73 8 2e-5 1 73
    ;;
esac

"${PYTHON_BIN}" tools/medical_project/build_supplementary_ablation_manifest.py
