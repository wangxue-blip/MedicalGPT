#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/agent/run_03_eval_agent.sh

Batch-run MedVision-Agent over a question file, then evaluate grounding and
system metrics. The question file is jsonl with {"question": ..., "image_path": ...}.

Overrides:
  QUESTIONS=data_processed/medical_project/agent/questions.jsonl
  PREDICTIONS=outputs/medical_project/agent/predictions.jsonl
  EVAL_DIR=outputs/medical_project/agent/eval
  MOCK=true
  DEVICE=cpu
  TEXT_PEFT=
  MODEL_LABEL=agent_mock
EOF
  exit 0
fi

: "${QUESTIONS:=data_processed/medical_project/agent/questions.jsonl}"
: "${PREDICTIONS:=outputs/medical_project/agent/predictions.jsonl}"
: "${EVAL_DIR:=outputs/medical_project/agent/eval}"
: "${MOCK:=true}"
: "${DEVICE:=cpu}"
: "${TEXT_PEFT:=}"
: "${MODEL_LABEL:=agent_mock}"

mkdir -p "${EVAL_DIR}" "$(dirname "${PREDICTIONS}")"

MEDAGENT_MOCK="${MOCK}" MEDAGENT_DEVICE="${DEVICE}" MEDAGENT_TEXT_PEFT="${TEXT_PEFT}" \
python - "${QUESTIONS}" "${PREDICTIONS}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from medical_agent import config
from medical_agent.pipeline import MedVisionAgent
from medical_agent.schema import read_jsonl, write_jsonl

questions, predictions = sys.argv[1], sys.argv[2]
agent = MedVisionAgent(mock=config.MOCK, device=config.DEVICE, text_peft=config.TEXT_PEFT)
rows = []
for item in read_jsonl(questions):
    rows.append(agent.run(item.get("question", ""), image_path=item.get("image_path", ""), lab_data=item.get("lab_data")))
print("wrote", write_jsonl(predictions, rows), "predictions ->", predictions)
PY

python eval/medical_project/agent/eval_grounding.py \
  --prediction_file "${PREDICTIONS}" --output "${EVAL_DIR}/grounding_${MODEL_LABEL}.json" --model "${MODEL_LABEL}"

python eval/medical_project/agent/eval_agent.py \
  --prediction_file "${PREDICTIONS}" --output "${EVAL_DIR}/agent_${MODEL_LABEL}.json" --model "${MODEL_LABEL}"
