#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_09_eval_all.sh

Run Base/SFT/GRPO medical evaluation scripts and summarize metrics.

Common overrides:
  BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
  EVAL_OUTPUT_DIR=outputs/medical_project/eval
  EMBEDDING_MODEL=models/bge-small-zh-v1.5
  EMBEDDING_DEVICE=cpu
  TORCH_DTYPE=float16
  DEVICE_MAP=auto
  MAX_PPL_SAMPLES=1000
  MAX_QA_SAMPLES=200
  MAX_CEVAL_SAMPLES=-1
  RUN_BASE=true
  RUN_SFT10K=true
  RUN_SFT30K=true
  RUN_GRPO10K500=true
  RUN_GRPO10K1000=true
  RUN_GRPO30K500=true
  RUN_GRPO30K1000=true
EOF
  exit 0
fi

: "${BASE_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${EVAL_OUTPUT_DIR:=outputs/medical_project/eval}"
: "${PPL_EVAL_FILE:=data_processed/medical_project/eval/medical_longtext_eval.jsonl}"
: "${QA_EVAL_FILE:=data_processed/medical_project/eval/medical_qa_eval.jsonl}"
: "${CEVAL_DEV_DIR:=data_raw/medical_project/ceval_dev}"
: "${EMBEDDING_MODEL:=models/bge-small-zh-v1.5}"
: "${EMBEDDING_DEVICE:=cpu}"
: "${TORCH_DTYPE:=float16}"
: "${DEVICE_MAP:=auto}"
: "${MAX_PPL_SAMPLES:=1000}"
: "${MAX_QA_SAMPLES:=200}"
: "${MAX_CEVAL_SAMPLES:=-1}"

: "${RUN_BASE:=true}"
: "${RUN_SFT10K:=true}"
: "${RUN_SFT30K:=true}"
: "${RUN_GRPO10K500:=true}"
: "${RUN_GRPO10K1000:=true}"
: "${RUN_GRPO30K500:=true}"
: "${RUN_GRPO30K1000:=true}"

: "${SFT10K_PEFT:=outputs/medical_project/sft/qwen25_7b_lora_10k_v100}"
: "${SFT30K_PEFT:=outputs/medical_project/sft/qwen25_7b_lora_30k_v100}"
: "${GRPO10K500_PEFT:=outputs/medical_project/grpo/qwen25_7b_lora_10k_v100_grpo_500}"
: "${GRPO10K1000_PEFT:=outputs/medical_project/grpo/qwen25_7b_lora_10k_v100_grpo_1000}"
: "${GRPO30K500_PEFT:=outputs/medical_project/grpo/qwen25_7b_lora_30k_v100_grpo_500}"
: "${GRPO30K1000_PEFT:=outputs/medical_project/grpo/qwen25_7b_lora_30k_v100_grpo_1000}"

mkdir -p "${EVAL_OUTPUT_DIR}"

if [[ ! -s "${PPL_EVAL_FILE}" || ! -s "${QA_EVAL_FILE}" ]]; then
  python tools/medical_project/build_eval_data.py \
    --candidates data_processed/medical_project/sft/medical_candidates.jsonl \
    --exclude_files data_processed/medical_project/sft/medical_sft_top30k.jsonl \
    --output_dir data_processed/medical_project/eval \
    --qa_samples 500 \
    --longtext_samples 1000
fi

run_one_model() {
  local label="$1"
  local peft_path="${2:-}"
  local model_path="${BASE_MODEL}"
  local model_id
  local peft_args=()
  if [[ -n "${peft_path}" ]]; then
    if [[ ! -f "${peft_path}/adapter_config.json" ]]; then
      echo "Skip ${label}: adapter_config.json not found under ${peft_path}"
      return 0
    fi
    peft_args=(--peft_path "${peft_path}")
    model_id="$(basename "${peft_path}")"
  else
    model_id="$(basename "${model_path}")"
  fi

  echo "Evaluating ${label}"
  python eval/medical_project/eval_ppl.py \
    --model_name_or_path "${model_path}" \
    "${peft_args[@]}" \
    --eval_file "${PPL_EVAL_FILE}" \
    --output "${EVAL_OUTPUT_DIR}/ppl_${label}.json" \
    --max_samples "${MAX_PPL_SAMPLES}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --device_map "${DEVICE_MAP}"

  python eval/medical_project/eval_qa_similarity.py \
    --model_name_or_path "${model_path}" \
    "${peft_args[@]}" \
    --eval_file "${QA_EVAL_FILE}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --embedding_device "${EMBEDDING_DEVICE}" \
    --output "${EVAL_OUTPUT_DIR}/qa_similarity_${label}.json" \
    --prediction_output "${EVAL_OUTPUT_DIR}/predictions_${label}.jsonl" \
    --max_samples "${MAX_QA_SAMPLES}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --device_map "${DEVICE_MAP}"

  python eval/medical_project/eval_structure_safety.py \
    --prediction_file "${EVAL_OUTPUT_DIR}/predictions_${label}.jsonl" \
    --output "${EVAL_OUTPUT_DIR}/structure_safety_${label}.json" \
    --model "${model_id}"

  python eval/medical_project/eval_ceval_medical.py \
    --model_name_or_path "${model_path}" \
    "${peft_args[@]}" \
    --ceval_dev_dir "${CEVAL_DEV_DIR}" \
    --output "${EVAL_OUTPUT_DIR}/ceval_medical_${label}.json" \
    --max_samples "${MAX_CEVAL_SAMPLES}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --device_map "${DEVICE_MAP}"
}

eval_jobs=(
  "RUN_BASE|base|"
  "RUN_SFT10K|sft_10k_v100|${SFT10K_PEFT}"
  "RUN_SFT30K|sft_30k_v100|${SFT30K_PEFT}"
  "RUN_GRPO10K500|grpo_10k_v100_500|${GRPO10K500_PEFT}"
  "RUN_GRPO10K1000|grpo_10k_v100_1000|${GRPO10K1000_PEFT}"
  "RUN_GRPO30K500|grpo_30k_v100_500|${GRPO30K500_PEFT}"
  "RUN_GRPO30K1000|grpo_30k_v100_1000|${GRPO30K1000_PEFT}"
)

for eval_job in "${eval_jobs[@]}"; do
  IFS="|" read -r run_flag label peft_path <<< "${eval_job}"
  if [[ "${!run_flag}" == "true" ]]; then
    run_one_model "${label}" "${peft_path}"
  fi
done

python eval/medical_project/summarize_metrics.py \
  --metrics_dir "${EVAL_OUTPUT_DIR}" \
  --output_csv "${EVAL_OUTPUT_DIR}/summary_metrics.csv" \
  --output_json "${EVAL_OUTPUT_DIR}/summary_metrics.json"
