#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/agent/run_02_demo.sh

Run one MedVision-Agent request. Defaults to mock backends so it works
without any model; set MOCK=false on the server.

Overrides:
  MOCK=true
  IMAGE=                    (empty -> text-only question)
  QUESTION="这张胸片有没有异常？"
  DEVICE=cpu
  TEXT_PEFT=                (medical LoRA/GRPO adapter dir)
  DUMP_JSON=outputs/medical_project/agent/demo_result.json
EOF
  exit 0
fi

: "${MOCK:=true}"
: "${IMAGE:=}"
: "${QUESTION:=这张胸片有没有异常？}"
: "${DEVICE:=cpu}"
: "${TEXT_PEFT:=}"
: "${DUMP_JSON:=outputs/medical_project/agent/demo_result.json}"

mock_args=()
if [[ "${MOCK}" == "true" ]]; then
  mock_args=(--mock)
fi
image_args=()
if [[ -n "${IMAGE}" ]]; then
  image_args=(--image "${IMAGE}")
fi

python demo/medical_agent_demo.py \
  "${mock_args[@]}" "${image_args[@]}" \
  --question "${QUESTION}" --device "${DEVICE}" --text_peft "${TEXT_PEFT}" \
  --dump_json "${DUMP_JSON}"
