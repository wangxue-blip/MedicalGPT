#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/agent/run_01_prepare_data.sh

Convert VQA-RAD and CheXlocalize into the unified evaluation format.
Sources that are not present are skipped.

Overrides:
  VQA_RAD_TRAIN=data_raw/medical_project/agent/vqa_rad/trainset.json
  VQA_RAD_TEST=data_raw/medical_project/agent/vqa_rad/testset.json
  VQA_RAD_IMAGES=data_raw/medical_project/agent/vqa_rad/images
  CHEXLOCALIZE_GT=data_raw/medical_project/agent/chexlocalize/gt_segmentations_test.json
  CHEXLOCALIZE_IMAGES=data_raw/medical_project/agent/chexlocalize/images
  OUTPUT_DIR=data_processed/medical_project/agent
EOF
  exit 0
fi

: "${VQA_RAD_TRAIN:=data_raw/medical_project/agent/vqa_rad/trainset.json}"
: "${VQA_RAD_TEST:=data_raw/medical_project/agent/vqa_rad/testset.json}"
: "${VQA_RAD_IMAGES:=data_raw/medical_project/agent/vqa_rad/images}"
: "${CHEXLOCALIZE_GT:=data_raw/medical_project/agent/chexlocalize/gt_segmentations_test.json}"
: "${CHEXLOCALIZE_IMAGES:=data_raw/medical_project/agent/chexlocalize/images}"
: "${OUTPUT_DIR:=data_processed/medical_project/agent}"

if [[ -f "${VQA_RAD_TRAIN}" || -f "${VQA_RAD_TEST}" ]]; then
  python tools/medical_project/agent/prepare_vqa_rad.py \
    --input_train "${VQA_RAD_TRAIN}" --input_test "${VQA_RAD_TEST}" \
    --image_dir "${VQA_RAD_IMAGES}" --output_dir "${OUTPUT_DIR}/vqa_rad"
else
  echo "Skip VQA-RAD: ${VQA_RAD_TRAIN} not found"
fi

if [[ -f "${CHEXLOCALIZE_GT}" ]]; then
  python tools/medical_project/agent/prepare_chexlocalize.py \
    --gt_json "${CHEXLOCALIZE_GT}" --image_dir "${CHEXLOCALIZE_IMAGES}" \
    --output "${OUTPUT_DIR}/chexlocalize/gt.jsonl" --keep_rle
else
  echo "Skip CheXlocalize: ${CHEXLOCALIZE_GT} not found"
fi
