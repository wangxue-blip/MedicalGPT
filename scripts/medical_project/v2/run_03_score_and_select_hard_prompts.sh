#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/v2/run_03_score_and_select_hard_prompts.sh

Score Base/SFT mining predictions and rank hard prompts for v2 GRPO data.
Run this after V2-2 prediction files are generated.

Common overrides:
  POOL=data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl
  BASE_PREDICTIONS=outputs/medical_project/v2/mining/predictions_base.jsonl
  SFT_PREDICTIONS=outputs/medical_project/v2/mining/predictions_sft_10k.jsonl
  EXTRA_SFT_PREDICTIONS=outputs/medical_project/v2/mining/predictions_sft_30k.jsonl
  OUTPUT=data_processed/medical_project/v2/mining/scored_prompt_pool.jsonl
  SUMMARY_OUTPUT=data_processed/medical_project/v2/mining/scored_prompt_pool_summary.json
  TOP_OUTPUT=data_processed/medical_project/v2/mining/hard_prompt_candidates_top1500.jsonl
  TOP_K=1500
  EMBEDDING_MODEL=models/bge-m3
  EMBEDDING_DEVICE=cpu
  EMBEDDING_BATCH_SIZE=32
  ALLOW_MISSING_PREDICTIONS=false
  MIN_PREDICTION_COVERAGE=0.95
EOF
  exit 0
fi

: "${POOL:=data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl}"
: "${BASE_PREDICTIONS:=outputs/medical_project/v2/mining/predictions_base.jsonl}"
: "${SFT_PREDICTIONS:=outputs/medical_project/v2/mining/predictions_sft_10k.jsonl}"
: "${EXTRA_SFT_PREDICTIONS:=outputs/medical_project/v2/mining/predictions_sft_30k.jsonl}"
: "${OUTPUT:=data_processed/medical_project/v2/mining/scored_prompt_pool.jsonl}"
: "${SUMMARY_OUTPUT:=data_processed/medical_project/v2/mining/scored_prompt_pool_summary.json}"
: "${TOP_OUTPUT:=data_processed/medical_project/v2/mining/hard_prompt_candidates_top1500.jsonl}"
: "${TOP_K:=1500}"
: "${EMBEDDING_MODEL:=models/bge-m3}"
: "${EMBEDDING_DEVICE:=cpu}"
: "${EMBEDDING_BATCH_SIZE:=32}"
: "${ALLOW_MISSING_PREDICTIONS:=false}"
: "${MIN_PREDICTION_COVERAGE:=0.95}"

python tools/medical_project/v2/score_mining_predictions.py \
  --pool "${POOL}" \
  --base_predictions "${BASE_PREDICTIONS}" \
  --sft_predictions "${SFT_PREDICTIONS}" \
  --extra_sft_predictions "${EXTRA_SFT_PREDICTIONS}" \
  --output "${OUTPUT}" \
  --summary_output "${SUMMARY_OUTPUT}" \
  --top_output "${TOP_OUTPUT}" \
  --top_k "${TOP_K}" \
  --embedding_model "${EMBEDDING_MODEL}" \
  --embedding_device "${EMBEDDING_DEVICE}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --allow_missing_predictions "${ALLOW_MISSING_PREDICTIONS}" \
  --min_prediction_coverage "${MIN_PREDICTION_COVERAGE}"
