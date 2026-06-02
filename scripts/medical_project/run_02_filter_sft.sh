#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_02_filter_sft.sh

Stage 2-3 placeholder. It will build medical topic queries and filter
SFT samples by embedding similarity.
EOF
  exit 0
fi

python tools/medical_project/build_medical_queries.py \
  --output data_processed/medical_project/embedding/medical_topic_queries.jsonl

python tools/medical_project/embed_and_filter_sft.py \
  --candidates data_processed/medical_project/sft/medical_candidates.jsonl \
  --queries data_processed/medical_project/embedding/medical_topic_queries.jsonl \
  --embedding_model BAAI/bge-m3 \
  --output_dir data_processed/medical_project/sft \
  --topk_list 1000,10000,30000 \
  --batch_size 64 \
  --normalize_embeddings true
