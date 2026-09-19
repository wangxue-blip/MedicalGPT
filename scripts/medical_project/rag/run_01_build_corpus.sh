#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/rag/run_01_build_corpus.sh

RAG stage 1. Parse tiered medical sources (guidelines / drug labels /
encyclopedia QA / knowledge-graph QA) into a chunk corpus with full
provenance, written to CHUNKS_FILE plus a corpus summary JSON.

Only the sources whose variables are non-empty are passed to build_corpus.py,
so an unavailable corpus is simply skipped instead of failing the run.

L3 online-consultation QA is excluded by default: unreviewed answers retrieved
as authoritative evidence make wrong answers look credible. Set
INCLUDE_CONSULTATION=true only when you accept that trade-off.

Common overrides:
  GUIDELINE_DIR=data_raw/medical_project/rag/guideline
  DRUG_LABEL_DIR=data_raw/medical_project/rag/drug_label
  ENCYCLOPEDIA_JSONL=data_raw/medical_project/rag/huatuo_encyclopedia_qa.jsonl
  KNOWLEDGE_GRAPH_JSONL=data_raw/medical_project/rag/huatuo_knowledge_graph_qa.jsonl
  CONSULTATION_JSONL=data_processed/medical_project/sft/medical_candidates.jsonl
  INCLUDE_CONSULTATION=false
  MAX_GUIDELINE=-1
  MAX_DRUG_LABEL=-1
  MAX_ENCYCLOPEDIA=20000
  MAX_KNOWLEDGE_GRAPH=20000
  MAX_CONSULTATION=0
  QUESTION_FIELD=question
  ANSWER_FIELD=answer
  TITLE_FIELD=
  VERSION_TAG=
  RECURSIVE=true
  CHILD_CHARS=320
  CHILD_OVERLAP=64
  PARENT_CHARS=1200
  MIN_CHARS=40
  CHUNKS_FILE=data_processed/medical_project/rag/chunks.jsonl
  CORPUS_SUMMARY=data_processed/medical_project/rag/corpus_summary.json
  LOG_LEVEL=INFO
EOF
  exit 0
fi

: "${GUIDELINE_DIR:=data_raw/medical_project/rag/guideline}"
: "${DRUG_LABEL_DIR:=data_raw/medical_project/rag/drug_label}"
: "${ENCYCLOPEDIA_JSONL:=data_raw/medical_project/rag/huatuo_encyclopedia_qa.jsonl}"
: "${KNOWLEDGE_GRAPH_JSONL:=data_raw/medical_project/rag/huatuo_knowledge_graph_qa.jsonl}"
: "${CONSULTATION_JSONL:=data_processed/medical_project/sft/medical_candidates.jsonl}"
: "${INCLUDE_CONSULTATION:=false}"

: "${MAX_GUIDELINE:=-1}"
: "${MAX_DRUG_LABEL:=-1}"
: "${MAX_ENCYCLOPEDIA:=20000}"
: "${MAX_KNOWLEDGE_GRAPH:=20000}"
: "${MAX_CONSULTATION:=0}"

: "${QUESTION_FIELD:=question}"
: "${ANSWER_FIELD:=answer}"
: "${TITLE_FIELD:=}"
: "${VERSION_TAG:=}"
: "${RECURSIVE:=true}"

: "${CHILD_CHARS:=320}"
: "${CHILD_OVERLAP:=64}"
: "${PARENT_CHARS:=1200}"
: "${MIN_CHARS:=40}"

: "${CHUNKS_FILE:=data_processed/medical_project/rag/chunks.jsonl}"
: "${CORPUS_SUMMARY:=data_processed/medical_project/rag/corpus_summary.json}"
: "${LOG_LEVEL:=INFO}"

mkdir -p "$(dirname "${CHUNKS_FILE}")"

source_args=()

# 目录型来源：目录不存在就跳过而不是报错。四类来源通常不会同时齐备，
# 少一类只是语料窄一点，不该让整条流水线停下来。
if [[ -n "${GUIDELINE_DIR}" && -e "${GUIDELINE_DIR}" ]]; then
  source_args+=(--guideline_dir "${GUIDELINE_DIR}" --max_guideline "${MAX_GUIDELINE}")
else
  echo "Skip guideline: ${GUIDELINE_DIR} not found"
fi

if [[ -n "${DRUG_LABEL_DIR}" && -e "${DRUG_LABEL_DIR}" ]]; then
  source_args+=(--drug_label_dir "${DRUG_LABEL_DIR}" --max_drug_label "${MAX_DRUG_LABEL}")
else
  echo "Skip drug_label: ${DRUG_LABEL_DIR} not found"
fi

if [[ -n "${ENCYCLOPEDIA_JSONL}" && -s "${ENCYCLOPEDIA_JSONL}" ]]; then
  source_args+=(--encyclopedia_jsonl "${ENCYCLOPEDIA_JSONL}" --max_encyclopedia "${MAX_ENCYCLOPEDIA}")
else
  echo "Skip encyclopedia: ${ENCYCLOPEDIA_JSONL} not found"
fi

if [[ -n "${KNOWLEDGE_GRAPH_JSONL}" && -s "${KNOWLEDGE_GRAPH_JSONL}" ]]; then
  source_args+=(--knowledge_graph_jsonl "${KNOWLEDGE_GRAPH_JSONL}" --max_knowledge_graph "${MAX_KNOWLEDGE_GRAPH}")
else
  echo "Skip knowledge_graph: ${KNOWLEDGE_GRAPH_JSONL} not found"
fi

if [[ "${INCLUDE_CONSULTATION}" == "true" ]]; then
  if [[ -n "${CONSULTATION_JSONL}" && -s "${CONSULTATION_JSONL}" ]]; then
    source_args+=(--consultation_jsonl "${CONSULTATION_JSONL}" --include_consultation
                  --max_consultation "${MAX_CONSULTATION}")
  else
    echo "Skip consultation: ${CONSULTATION_JSONL} not found"
  fi
else
  echo "Skip consultation: INCLUDE_CONSULTATION=${INCLUDE_CONSULTATION}"
fi

if [[ ${#source_args[@]} -eq 0 ]]; then
  echo "No corpus source available. Set GUIDELINE_DIR / DRUG_LABEL_DIR /" >&2
  echo "ENCYCLOPEDIA_JSONL / KNOWLEDGE_GRAPH_JSONL to an existing path." >&2
  exit 1
fi

optional_args=()
if [[ -n "${TITLE_FIELD}" ]]; then
  optional_args+=(--title_field "${TITLE_FIELD}")
fi
if [[ -n "${VERSION_TAG}" ]]; then
  optional_args+=(--version_tag "${VERSION_TAG}")
fi
if [[ "${RECURSIVE}" != "true" ]]; then
  optional_args+=(--no_recursive)
fi

python tools/medical_project/rag/build_corpus.py \
  "${source_args[@]}" \
  "${optional_args[@]}" \
  --question_field "${QUESTION_FIELD}" \
  --answer_field "${ANSWER_FIELD}" \
  --child_chars "${CHILD_CHARS}" \
  --child_overlap "${CHILD_OVERLAP}" \
  --parent_chars "${PARENT_CHARS}" \
  --min_chars "${MIN_CHARS}" \
  --output "${CHUNKS_FILE}" \
  --summary_output "${CORPUS_SUMMARY}" \
  --log_level "${LOG_LEVEL}"

echo "Corpus written to ${CHUNKS_FILE}"
echo "Summary written to ${CORPUS_SUMMARY}"
echo "Next: bash scripts/medical_project/rag/run_02_build_index.sh"
