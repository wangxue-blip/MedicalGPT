#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_10_generate_report.sh

Generate English and Chinese experiment reports from metrics and data summaries:
  docs/medical_project/EXPERIMENT_REPORT.md
  docs/medical_project/EXPERIMENT_REPORT_ZH.md
EOF
  exit 0
fi

python eval/medical_project/generate_report.py \
  --metrics_dir outputs/medical_project/eval \
  --data_summary data_processed/medical_project/embedding/filter_summary.json \
  --output docs/medical_project/EXPERIMENT_REPORT.md
