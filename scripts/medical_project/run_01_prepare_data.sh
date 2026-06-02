#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_01_prepare_data.sh

Convert shibing624/medical finetune train/valid files to the project raw
JSONL entry, then normalize raw medical QA data into
data_processed/medical_project/sft/medical_candidates.jsonl.
EOF
  exit 0
fi

if [[ ! -f data_raw/medical_project/medical_raw.jsonl ]]; then
  python tools/medical_project/convert_medical_json_to_raw.py \
    --inputs data_raw/medical_project/medical/train_zh_0.json data_raw/medical_project/medical/valid_zh_0.json \
    --output data_raw/medical_project/medical_raw.jsonl
fi

python tools/medical_project/prepare_raw_medical.py \
  --input data_raw/medical_project/medical_raw.jsonl \
  --output data_processed/medical_project/sft/medical_candidates.jsonl \
  --min_question_len 5 \
  --min_answer_len 20 \
  --max_answer_len 2048
