#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_all_pipeline.sh

Run the main MedicalGPT medical SFT + GRPO pipeline in order.
Training and evaluation stages require prepared data and GPU resources.
EOF
  exit 0
fi

bash scripts/medical_project/run_00_env_check.sh
bash scripts/medical_project/run_01_prepare_data.sh
bash scripts/medical_project/run_02_filter_sft.sh
bash scripts/medical_project/run_03_sft_smoke_1k.sh
bash scripts/medical_project/run_04_sft_train_10k.sh
bash scripts/medical_project/run_06_build_grpo_data.sh
bash scripts/medical_project/run_07_grpo_smoke.sh
bash scripts/medical_project/run_08_grpo_train.sh
bash scripts/medical_project/run_09_eval_all.sh
bash scripts/medical_project/run_10_generate_report.sh
