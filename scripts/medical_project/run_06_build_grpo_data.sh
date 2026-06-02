#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_06_build_grpo_data.sh

Build 500 and 1000 sample medical GRPO datasets from the filtered SFT data.
EOF
  exit 0
fi

python tools/medical_project/build_grpo_data.py \
  --sft_data data_processed/medical_project/sft/medical_sft_top10000.jsonl \
  --output data_processed/medical_project/grpo/medical_grpo_500.jsonl \
  --num_samples 500 \
  --prefer_categories 急诊医学,内科学,药物禁忌,儿科学,妇产科学 \
  --summary_output data_processed/medical_project/grpo/grpo_data_summary_500.json

python tools/medical_project/build_grpo_data.py \
  --sft_data data_processed/medical_project/sft/medical_sft_top10000.jsonl \
  --output data_processed/medical_project/grpo/medical_grpo_1000.jsonl \
  --num_samples 1000 \
  --prefer_categories 急诊医学,内科学,药物禁忌,儿科学,妇产科学 \
  --summary_output data_processed/medical_project/grpo/grpo_data_summary.json
