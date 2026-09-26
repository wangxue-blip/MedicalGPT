#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/rag/run_03_eval_rag.sh

RAG stage 3. Run retrieval evaluation and citation-quality evaluation in
order, then merge both metric files into EVAL_OUTPUT_DIR/rag_eval_summary.json
and print them as one table.

eval_citation.py scores a predictions jsonl (query / answer / sources), so
when PREDICTION_FILE is missing this script first answers every eval query
through the MedicalRAG pipeline and writes that file.

Answers are only generated when GENERATOR_MODEL points at a real model;
otherwise the run is retrieval-only and the citation metrics stay empty. That
is deliberate: scoring citations of an empty answer would report a perfect
score for a pipeline that answered nothing.

Common overrides:
  CHUNKS_FILE=data_processed/medical_project/rag/chunks.jsonl
  INDEX_DIR=data_processed/medical_project/rag/index
  QUERY_FILE=data_processed/medical_project/rag/eval_queries.jsonl
  EVAL_OUTPUT_DIR=outputs/medical_project/rag_eval
  PREDICTION_FILE=
  MODEL_TAG=rag_baseline
  TOPK=5
  TOPK_LIST=1,3,5,10
  RECALL_K=50
  FUSION_K=60
  ROUTES=dense,bm25,term
  ROUTE_WEIGHTS=
  AUTHORITY_WEIGHT=0.15
  DENSE_MODEL=models/bge-m3
  RERANKER_MODEL=
  GENERATOR_MODEL=
  PEFT_PATH=
  TERMINOLOGY=data_processed/medical_project/rag/terminology.jsonl
  DEVICE=cpu
  MAX_QUERIES=-1
  SUPPORT_THRESHOLD=0.62
  CITATION_EMBEDDER=none
  WEAK_LABEL=true
  RUN_RETRIEVAL=true
  RUN_CITATION=true
  MAX_NEW_TOKENS=512
  LOG_LEVEL=INFO
EOF
  exit 0
fi

: "${CHUNKS_FILE:=data_processed/medical_project/rag/chunks.jsonl}"
: "${INDEX_DIR:=data_processed/medical_project/rag/index}"
: "${QUERY_FILE:=data_processed/medical_project/rag/eval_queries.jsonl}"
: "${EVAL_OUTPUT_DIR:=outputs/medical_project/rag_eval}"
: "${PREDICTION_FILE:=}"
: "${MODEL_TAG:=rag_baseline}"

: "${RETRIEVAL_PY:=eval/medical_project/rag/eval_retrieval.py}"
: "${CITATION_PY:=eval/medical_project/rag/eval_citation.py}"

: "${TOPK:=5}"
: "${TOPK_LIST:=1,3,5,10}"
: "${RECALL_K:=50}"
: "${FUSION_K:=60}"
: "${ROUTES:=dense,bm25,term}"
: "${ROUTE_WEIGHTS:=}"
: "${AUTHORITY_WEIGHT:=0.15}"
: "${DENSE_MODEL:=models/bge-m3}"
: "${RERANKER_MODEL:=}"
: "${GENERATOR_MODEL:=}"
: "${PEFT_PATH:=}"
: "${TERMINOLOGY:=data_processed/medical_project/rag/terminology.jsonl}"
: "${DEVICE:=cpu}"
: "${MAX_QUERIES:=-1}"
: "${SUPPORT_THRESHOLD:=0.62}"
: "${CITATION_EMBEDDER:=none}"
: "${WEAK_LABEL:=true}"
: "${RUN_RETRIEVAL:=true}"
: "${RUN_CITATION:=true}"
: "${MAX_NEW_TOKENS:=512}"
: "${LOG_LEVEL:=INFO}"

if [[ ! -s "${CHUNKS_FILE}" ]]; then
  echo "Chunk file not found: ${CHUNKS_FILE}" >&2
  echo "Run bash scripts/medical_project/rag/run_01_build_corpus.sh first." >&2
  exit 1
fi
if [[ ! -s "${QUERY_FILE}" ]]; then
  echo "Eval query file not found: ${QUERY_FILE}" >&2
  echo "Each line should be {\"query\": ..., \"gold_chunk_ids\": [...]} or {\"query\": ..., \"gold_answer\": ...}" >&2
  exit 1
fi

mkdir -p "${EVAL_OUTPUT_DIR}"

RETRIEVAL_JSON="${EVAL_OUTPUT_DIR}/rag_retrieval_${MODEL_TAG}.json"
CITATION_JSON="${EVAL_OUTPUT_DIR}/rag_citation_${MODEL_TAG}.json"
SUMMARY_JSON="${EVAL_OUTPUT_DIR}/rag_eval_summary.json"
if [[ -z "${PREDICTION_FILE}" ]]; then
  PREDICTION_FILE="${EVAL_OUTPUT_DIR}/predictions_${MODEL_TAG}.jsonl"
fi

# 模型目录不存在时不要把路径传下去：传一个不存在的路径会触发远端下载，
# 在离线机器上表现为长时间无输出地卡住，而不是一句清楚的报错。
model_args=()
if [[ -n "${DENSE_MODEL}" && -d "${DENSE_MODEL}" ]]; then
  model_args+=(--dense_model "${DENSE_MODEL}")
else
  echo "Dense model not found locally: ${DENSE_MODEL} (falling back to hashed vectors)"
fi
if [[ -n "${RERANKER_MODEL}" && -d "${RERANKER_MODEL}" ]]; then
  model_args+=(--reranker_model "${RERANKER_MODEL}")
fi

# ---------------------------------------------------------------- retrieval
if [[ "${RUN_RETRIEVAL}" == "true" ]]; then
  if [[ ! -f "${RETRIEVAL_PY}" ]]; then
    echo "Retrieval eval script not found: ${RETRIEVAL_PY}" >&2
    exit 1
  fi

  retrieval_args=()
  if [[ -n "${ROUTE_WEIGHTS}" ]]; then
    retrieval_args+=(--route_weights "${ROUTE_WEIGHTS}")
  fi
  if [[ -n "${TERMINOLOGY}" && -s "${TERMINOLOGY}" ]]; then
    retrieval_args+=(--terminology_file "${TERMINOLOGY}")
  fi
  # 评测集若只给了 gold_answer 而没有 gold_chunk_ids，必须开弱标注，
  # 否则每条 query 的 gold 集为空，recall 会恒等于 0 而看不出是标注缺失。
  if [[ "${WEAK_LABEL}" == "true" ]]; then
    retrieval_args+=(--weak_label_from_answer)
  fi

  echo "Running retrieval eval: ${RETRIEVAL_PY}"
  python "${RETRIEVAL_PY}" \
    --chunk_file "${CHUNKS_FILE}" \
    --eval_file "${QUERY_FILE}" \
    --output "${RETRIEVAL_JSON}" \
    --model "${MODEL_TAG}" \
    --topk_list "${TOPK_LIST}" \
    --routes "${ROUTES}" \
    --recall_k "${RECALL_K}" \
    --fusion_k "${FUSION_K}" \
    --authority_weight "${AUTHORITY_WEIGHT}" \
    --device "${DEVICE}" \
    --max_queries "${MAX_QUERIES}" \
    --dump_per_query "${EVAL_OUTPUT_DIR}/retrieval_per_query_${MODEL_TAG}.jsonl" \
    "${model_args[@]}" \
    "${retrieval_args[@]}"
else
  echo "Skip retrieval eval: RUN_RETRIEVAL=${RUN_RETRIEVAL}"
fi

# ---------------------------------------------------------------- citation
if [[ "${RUN_CITATION}" == "true" ]]; then
  if [[ ! -f "${CITATION_PY}" ]]; then
    echo "Citation eval script not found: ${CITATION_PY}" >&2
    exit 1
  fi

  if [[ ! -s "${PREDICTION_FILE}" ]]; then
    echo "Predictions not found, generating: ${PREDICTION_FILE}"
    export CHUNKS_FILE QUERY_FILE PREDICTION_FILE ROUTES DEVICE TOPK RECALL_K \
           FUSION_K AUTHORITY_WEIGHT DENSE_MODEL RERANKER_MODEL GENERATOR_MODEL \
           PEFT_PATH TERMINOLOGY MAX_QUERIES MAX_NEW_TOKENS LOG_LEVEL
    # 这段批量作答内联在这里而不是单独建脚本：它唯一的用途就是给 eval_citation.py
    # 喂输入，没有独立复用价值，多一个文件就多一处要跟着 pipeline 签名走的 CLI。
    python - <<'PY'
# -*- coding: utf-8 -*-
import io
import json
import logging
import os
import sys

sys.path.insert(0, os.getcwd())

from medical_rag.pipeline import MedicalRAG
from medical_rag.schema import read_jsonl

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(levelname)s %(name)s: %(message)s",
)

QUERY_FIELDS = ("query", "question", "instruction", "input")


def optional_dir(name):
    value = os.environ.get(name, "")
    return value if value and os.path.isdir(value) else None


def optional_file(name):
    value = os.environ.get(name, "")
    return value if value and os.path.isfile(value) else None


generator_model = optional_dir("GENERATOR_MODEL") or os.environ.get("GENERATOR_MODEL", "") or None
rag = MedicalRAG.from_jsonl(
    os.environ["CHUNKS_FILE"],
    dense_model=optional_dir("DENSE_MODEL"),
    reranker_model=optional_dir("RERANKER_MODEL"),
    terminology_path=optional_file("TERMINOLOGY"),
    device=os.environ.get("DEVICE", "cpu"),
    routes=tuple(r for r in os.environ.get("ROUTES", "").split(",") if r),
    recall_k=int(os.environ.get("RECALL_K", "50")),
    fusion_k=int(os.environ.get("FUSION_K", "60")),
    authority_weight=float(os.environ.get("AUTHORITY_WEIGHT", "0.15")),
    generator_model=generator_model,
    peft_path=optional_dir("PEFT_PATH"),
    max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "512")),
)
if rag.degraded:
    print("Pipeline degraded: %s" % "; ".join(rag.degraded))

topk = int(os.environ.get("TOPK", "5"))
limit = int(os.environ.get("MAX_QUERIES", "-1"))
generate = generator_model is not None

written = 0
out_path = os.environ["PREDICTION_FILE"]
parent = os.path.dirname(out_path)
if parent and not os.path.isdir(parent):
    os.makedirs(parent)
with io.open(out_path, "w", encoding="utf-8") as handle:
    for row in read_jsonl(os.environ["QUERY_FILE"], max_rows=limit):
        query = ""
        for field in QUERY_FIELDS:
            if row.get(field):
                query = str(row[field])
                break
        if not query:
            continue
        result = rag.answer(query, topk=topk, generate=generate)
        handle.write(json.dumps({
            "query": result["query"],
            "answer": result["answer"],
            "sources": result["sources"],
            "refused": result["refused"],
        }, ensure_ascii=False) + "\n")
        written += 1
print("Predictions written: %d -> %s" % (written, out_path))
PY
  fi

  citation_args=()
  if [[ "${CITATION_EMBEDDER}" == "dense" ]]; then
    citation_args+=(--embedder dense)
    if [[ -n "${DENSE_MODEL}" && -d "${DENSE_MODEL}" ]]; then
      citation_args+=(--dense_model "${DENSE_MODEL}")
    fi
  fi

  echo "Running citation eval: ${CITATION_PY}"
  python "${CITATION_PY}" \
    --prediction_file "${PREDICTION_FILE}" \
    --output "${CITATION_JSON}" \
    --model "${MODEL_TAG}" \
    --support_threshold "${SUPPORT_THRESHOLD}" \
    --device "${DEVICE}" \
    --dump_details "${EVAL_OUTPUT_DIR}/citation_details_${MODEL_TAG}.jsonl" \
    "${citation_args[@]}"
else
  echo "Skip citation eval: RUN_CITATION=${RUN_CITATION}"
fi

# ---------------------------------------------------------------- summarize
export RETRIEVAL_JSON CITATION_JSON SUMMARY_JSON INDEX_DIR

# 汇总用内联 Python 而不是再加一个脚本：这里只是把几份指标 JSON 的标量字段拍平
# 对齐打印，没有独立复用价值。
python - <<'PY'
# -*- coding: utf-8 -*-
import io
import json
import os


def load(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with io.open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print("Failed to read %s: %s" % (path, exc))
        return None


def flatten(data, prefix=""):
    # 只收标量：评测 JSON 里常混着逐条明细的长列表，全打出来会把终端刷没。
    flat = {}
    if isinstance(data, dict):
        for key, value in data.items():
            name = "%s.%s" % (prefix, key) if prefix else str(key)
            if isinstance(value, dict):
                flat.update(flatten(value, name))
            elif value is None or isinstance(value, (int, float, str, bool)):
                flat[name] = value
    return flat


sections = [
    ("retrieval", os.environ.get("RETRIEVAL_JSON", "")),
    ("citation", os.environ.get("CITATION_JSON", "")),
    ("index", os.path.join(os.environ.get("INDEX_DIR", ""), "index_summary.json")),
]

summary = {}
for name, path in sections:
    data = load(path)
    if data is None:
        print("")
        print("[%s] missing: %s" % (name, path))
        continue
    flat = flatten(data)
    summary[name] = flat
    print("")
    print("[%s] %s" % (name, path))
    print("-" * 72)
    for key in sorted(flat):
        value = flat[key]
        if isinstance(value, float):
            print("  %-46s %.4f" % (key, value))
        else:
            print("  %-46s %s" % (key, value))

out_path = os.environ.get("SUMMARY_JSON", "")
if out_path and summary:
    with io.open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2))
    print("")
    print("Summary written to %s" % out_path)
PY
