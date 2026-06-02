#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/v2/run_01_build_prompt_pool.sh

Build the v2 hard prompt pool for later Base/SFT mining.

Common overrides:
  MEDICAL_SFT=data_processed/medical_project/sft/medical_sft_top30000.jsonl
  CEVAL_DEV_DIR=data_raw/medical_project/ceval_dev
  OUTPUT=data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl
  SUMMARY_OUTPUT=data_processed/medical_project/v2/pool/prompt_pool_summary.json
  MEDICAL_SAMPLES=3000
  CEVAL_REWRITE_SAMPLES=1200
  SAFETY_RISK_SAMPLES=800
  SEED=42
EOF
  exit 0
fi

: "${MEDICAL_SFT:=data_processed/medical_project/sft/medical_sft_top30000.jsonl}"
: "${CEVAL_DEV_DIR:=data_raw/medical_project/ceval_dev}"
: "${OUTPUT:=data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl}"
: "${SUMMARY_OUTPUT:=data_processed/medical_project/v2/pool/prompt_pool_summary.json}"
: "${MEDICAL_SAMPLES:=3000}"
: "${CEVAL_REWRITE_SAMPLES:=1200}"
: "${SAFETY_RISK_SAMPLES:=800}"
: "${SEED:=42}"

python tools/medical_project/v2/build_hard_prompt_pool.py \
  --medical_sft "${MEDICAL_SFT}" \
  --ceval_dev_dir "${CEVAL_DEV_DIR}" \
  --output "${OUTPUT}" \
  --summary_output "${SUMMARY_OUTPUT}" \
  --medical_samples "${MEDICAL_SAMPLES}" \
  --ceval_rewrite_samples "${CEVAL_REWRITE_SAMPLES}" \
  --safety_risk_samples "${SAFETY_RISK_SAMPLES}" \
  --seed "${SEED}"
