#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/rag/run_02_build_index.sh

RAG stage 2. Build the three retrieval routes (dense / bm25 / term) from
CHUNKS_FILE, persist the dense vectors under INDEX_DIR, and write
index_summary.json with the backend and build time of every route.

Only dense vectors are cached: bm25 and term are pure-Python inverted indexes
that rebuild in seconds, while embedding a large corpus is the expensive step.

Check index_summary.json after the run. When sentence_transformers or a local
embedding model is missing, the dense route silently degrades to hashed
character n-grams, which still retrieves but is noticeably weaker; the summary
is where that shows up.

Common overrides:
  CHUNKS_FILE=data_processed/medical_project/rag/chunks.jsonl
  INDEX_DIR=data_processed/medical_project/rag/index
  ROUTES=dense,bm25,term
  DENSE_MODEL=models/bge-m3
  TERMINOLOGY=data_processed/medical_project/rag/terminology.jsonl
  DEVICE=cpu
  MAX_ROWS=-1
  PROBE_QUERY=二甲双胍怎么吃，肾功能不好能用吗
  PROBE_TOPK=3
  SAVE_DENSE=true
  VERIFY_CACHE=true
  INDEX_SUMMARY=
  LOG_LEVEL=INFO
EOF
  exit 0
fi

: "${CHUNKS_FILE:=data_processed/medical_project/rag/chunks.jsonl}"
: "${INDEX_DIR:=data_processed/medical_project/rag/index}"
: "${ROUTES:=dense,bm25,term}"
: "${DENSE_MODEL:=models/bge-m3}"
: "${TERMINOLOGY:=data_processed/medical_project/rag/terminology.jsonl}"
: "${DEVICE:=cpu}"
: "${MAX_ROWS:=-1}"
: "${PROBE_QUERY:=二甲双胍怎么吃，肾功能不好能用吗}"
: "${PROBE_TOPK:=3}"
: "${SAVE_DENSE:=true}"
: "${VERIFY_CACHE:=true}"
: "${INDEX_SUMMARY:=}"
: "${LOG_LEVEL:=INFO}"

if [[ ! -s "${CHUNKS_FILE}" ]]; then
  echo "Chunk file not found: ${CHUNKS_FILE}" >&2
  echo "Run bash scripts/medical_project/rag/run_01_build_corpus.sh first." >&2
  exit 1
fi

mkdir -p "${INDEX_DIR}"

extra_args=()

# 模型目录不存在时不要把路径传下去：build_index.py 会因此去 HuggingFace 拉模型，
# 在离线机器上表现为长时间卡住。这里直接留空，让它走本地探测 + 哈希向量降级。
if [[ -n "${DENSE_MODEL}" && -d "${DENSE_MODEL}" ]]; then
  extra_args+=(--dense_model "${DENSE_MODEL}")
else
  echo "Dense model not found locally: ${DENSE_MODEL} (falling back to hashed vectors)"
fi

if [[ -n "${TERMINOLOGY}" && -s "${TERMINOLOGY}" ]]; then
  extra_args+=(--terminology "${TERMINOLOGY}")
else
  echo "Terminology file not found: ${TERMINOLOGY} (using the built-in table)"
fi

if [[ -n "${INDEX_SUMMARY}" ]]; then
  extra_args+=(--summary "${INDEX_SUMMARY}")
fi
if [[ "${SAVE_DENSE}" != "true" ]]; then
  extra_args+=(--no_save_dense)
fi
if [[ "${VERIFY_CACHE}" != "true" ]]; then
  extra_args+=(--no_verify_cache)
fi

python tools/medical_project/rag/build_index.py \
  --chunks "${CHUNKS_FILE}" \
  --index_dir "${INDEX_DIR}" \
  --routes "${ROUTES}" \
  --device "${DEVICE}" \
  --max_rows "${MAX_ROWS}" \
  --probe_query "${PROBE_QUERY}" \
  --probe_topk "${PROBE_TOPK}" \
  --log_level "${LOG_LEVEL}" \
  "${extra_args[@]}"

echo "Index written to ${INDEX_DIR}"
echo "Next: bash scripts/medical_project/rag/run_03_eval_rag.sh"
