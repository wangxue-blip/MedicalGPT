#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Embed candidate QA samples and filter SFT data."""

import argparse
import heapq
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


AD_PATTERNS = [
    r"微信|加微|QQ|电话|手机号|联系方式|咨询热线|预约热线",
    r"点击咨询|在线咨询|免费咨询|扫码|公众号|官网|网址|http[s]?://|www\.",
    r"医院排名|权威医院|专家团队|特效药|包治|根治|保证治好",
]

HIGH_RISK_PATTERNS = [
    r"一定是|肯定是|绝对是|无需就医|不用去医院|自己吃药即可",
    r"每日\s*\d+\s*次|每天\s*\d+\s*次|每次\s*\d+\s*(片|粒|mg|毫克)",
    r"立即停药|自行停药|自行加药|孕妇.*随便用|儿童.*随便用",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def normalize_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_topk_list(value):
    topks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not topks or any(item <= 0 for item in topks):
        raise ValueError(f"Invalid --topk_list: {value}")
    return topks


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def compile_patterns(patterns):
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def pattern_count(text, patterns):
    return sum(1 for pattern in patterns if pattern.search(text))


def is_advertising(text, ad_patterns):
    return pattern_count(text, ad_patterns) > 0


def iter_char_ngrams(text, min_n=2, max_n=3, max_chars=512):
    text = normalize_text(text)[:max_chars]
    text = re.sub(r"\s+", "", text)
    if not text:
        return
    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        for idx in range(0, len(text) - n + 1):
            yield text[idx:idx + n]


def stable_hash(text, dims):
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value % dims


def hashing_feature_set(text, dims, min_n=2, max_n=3, max_chars=512):
    return {stable_hash(ngram, dims) for ngram in iter_char_ngrams(text, min_n, max_n, max_chars)}


class QueryNgramEmbedder:
    """Dependency-free query n-gram overlap embedding for CPU-only filtering."""

    def __init__(self, queries, dims=262144, min_n=2, max_n=3, max_chars=512):
        self.backend = "query_char_ngram_overlap"
        self.dims = dims
        self.min_n = min_n
        self.max_n = max_n
        self.max_chars = max_chars
        self.queries = queries
        self.query_features = [
            set(iter_char_ngrams(query["query"], min_n=min_n, max_n=max_n, max_chars=max_chars))
            for query in queries
        ]
        self.query_norms = [math.sqrt(len(features)) if features else 1.0 for features in self.query_features]
        self.inverted = defaultdict(list)
        for query_idx, features in enumerate(self.query_features):
            weight = 1.0 / self.query_norms[query_idx]
            for feature in features:
                self.inverted[feature].append((query_idx, weight))

    def score_text(self, text):
        text = normalize_text(text)[: self.max_chars]
        text = re.sub(r"\s+", "", text)
        total_ngrams = sum(max(0, len(text) - n + 1) for n in range(self.min_n, self.max_n + 1))
        if total_ngrams <= 0:
            return np.zeros(len(self.queries), dtype=np.float32)
        candidate_weight = 1.0 / math.sqrt(total_ngrams)
        scores = np.zeros(len(self.queries), dtype=np.float32)
        seen_relevant_features = set()
        for n in range(self.min_n, self.max_n + 1):
            if len(text) < n:
                continue
            for idx in range(0, len(text) - n + 1):
                feature = text[idx:idx + n]
                if feature in seen_relevant_features:
                    continue
                query_hits = self.inverted.get(feature)
                if not query_hits:
                    continue
                seen_relevant_features.add(feature)
                for query_idx, query_weight in query_hits:
                    scores[query_idx] += candidate_weight * query_weight
        if scores.max() > 1.0:
            scores = np.clip(scores, 0.0, 1.0)
        return scores


class SentenceTransformerEmbedder:
    """Sentence-transformers backend, used when the dependency is installed."""

    def __init__(self, queries, model_name, batch_size=64, normalize_embeddings=True):
        from sentence_transformers import SentenceTransformer

        model_path = model_name
        local_candidate = Path("models") / model_name.split("/")[-1]
        if not Path(model_path).exists() and local_candidate.exists():
            model_path = str(local_candidate)
        self.model = SentenceTransformer(model_path)
        self.backend = "sentence_transformers"
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.queries = queries
        query_texts = [query["query"] for query in queries]
        self.query_embeddings = self.model.encode(
            query_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def score_batch(self, texts):
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        if self.normalize_embeddings:
            return embeddings @ self.query_embeddings.T
        query_norms = np.linalg.norm(self.query_embeddings, axis=1, keepdims=True).T
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings @ self.query_embeddings.T) / np.maximum(emb_norms @ query_norms, 1e-12)


class ScoreWriter:
    def __init__(self, parquet_path, fallback_jsonl_path):
        self.parquet_path = Path(parquet_path)
        self.fallback_jsonl_path = Path(fallback_jsonl_path)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.schema = None
        self.fallback = None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            self.pa = pa
            self.pq = pq
            self.available = True
        except Exception:
            self.pa = None
            self.pq = None
            self.available = False
            self.fallback = self.fallback_jsonl_path.open("w", encoding="utf-8")

    def write_rows(self, rows):
        if not rows:
            return
        if self.available:
            table = self.pa.Table.from_pylist(rows)
            if self.writer is None:
                self.schema = table.schema
                self.writer = self.pq.ParquetWriter(self.parquet_path, self.schema)
            self.writer.write_table(table.cast(self.schema))
        else:
            for row in rows:
                self.fallback.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.fallback is not None:
            self.fallback.close()

    @property
    def output_path(self):
        return str(self.parquet_path if self.available else self.fallback_jsonl_path)


def load_queries(path):
    queries = list(read_jsonl(path))
    required = {"id", "category", "query"}
    missing = [query for query in queries if not required <= query.keys()]
    if missing:
        raise ValueError(f"Found query rows missing required keys: {missing[:3]}")
    return queries


def candidate_text(sample, max_answer_chars):
    return f"{sample.get('question', '')} {normalize_text(sample.get('answer', ''))[:max_answer_chars]}"


def top5_mean(scores):
    if len(scores) <= 5:
        return float(np.mean(scores)) if len(scores) else 0.0
    return float(np.partition(scores, -5)[-5:].mean())


def score_to_row(sample, scores, queries, risk_count, quality_weight):
    matched_idx = int(np.argmax(scores))
    max_similarity = float(scores[matched_idx])
    mean_top5_similarity = top5_mean(scores)
    base_score = 0.7 * max_similarity + 0.3 * mean_top5_similarity
    score = base_score * quality_weight
    matched_query = queries[matched_idx]
    return {
        "id": sample["id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "score": round(score, 8),
        "base_score": round(base_score, 8),
        "max_similarity": round(max_similarity, 8),
        "mean_top5_similarity": round(mean_top5_similarity, 8),
        "matched_category": matched_query["category"],
        "matched_query": matched_query["query"],
        "risk_expression_count": risk_count,
        "quality_weight": round(quality_weight, 4),
        "question_len": sample.get("question_len", len(sample.get("question", ""))),
        "answer_len": sample.get("answer_len", len(sample.get("answer", ""))),
    }


def update_heap(heap, row, max_size):
    item = (row["score"], row["id"], row)
    if len(heap) < max_size:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def write_topk_outputs(output_dir, sorted_rows, topks):
    output_dir = Path(output_dir)
    outputs = {}
    aliases = {
        1000: "1k",
        10000: "10k",
        30000: "30k",
    }
    for topk in topks:
        rows = sorted_rows[:topk]
        numeric_path = output_dir / f"medical_sft_top{topk}.jsonl"
        with numeric_path.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs[str(topk)] = str(numeric_path)
        if topk in aliases:
            alias_path = output_dir / f"medical_sft_top{aliases[topk]}.jsonl"
            with alias_path.open("w", encoding="utf-8") as fout:
                for row in rows:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            outputs[f"top{aliases[topk]}"] = str(alias_path)
    return outputs


def write_manual_review(path, rows, sample_size=30, seed=42):
    review_path = Path(path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    samples = rng.sample(rows, min(sample_size, len(rows))) if rows else []
    lines = [
        "# Top1k Manual Review Samples",
        "",
        "These samples are randomly selected from the filtered Top1k set for human inspection.",
        "",
    ]
    for idx, row in enumerate(samples, start=1):
        lines.extend(
            [
                f"## Sample {idx}: {row['id']}",
                "",
                f"- Score: {row['score']}",
                f"- Category: {row['matched_category']}",
                f"- Matched query: {row['matched_query']}",
                "",
                "Question:",
                "",
                row["question"],
                "",
                "Answer:",
                "",
                row["answer"][:1200],
                "",
            ]
        )
    review_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_embedder(args, queries):
    if args.backend in {"auto", "sentence_transformers"}:
        try:
            return SentenceTransformerEmbedder(
                queries,
                model_name=args.embedding_model,
                batch_size=args.batch_size,
                normalize_embeddings=args.normalize_embeddings,
            )
        except Exception as exc:
            if args.backend == "sentence_transformers":
                raise
            print(f"sentence-transformers backend unavailable, falling back to query_char_ngram_overlap: {exc}", flush=True)
    return QueryNgramEmbedder(
        queries,
        dims=args.hash_dims,
        min_n=args.hash_min_ngram,
        max_n=args.hash_max_ngram,
        max_chars=args.hash_max_chars,
    )


def run_filter(args):
    topks = parse_topk_list(args.topk_list)
    max_topk = max(topks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.queries)
    embedder = build_embedder(args, queries)
    ad_patterns = compile_patterns(AD_PATTERNS)
    risk_patterns = compile_patterns(HIGH_RISK_PATTERNS)
    seen_questions = set()
    heap = []
    score_buffer = []
    stats = Counter()
    score_writer = ScoreWriter(
        output_dir.parent / "embedding" / "filter_scores.parquet",
        output_dir.parent / "embedding" / "filter_scores.jsonl",
    )

    try:
        if isinstance(embedder, SentenceTransformerEmbedder):
            batch_samples = []
            batch_texts = []
            for sample in read_jsonl(args.candidates):
                prepared = prepare_candidate(sample, args, seen_questions, ad_patterns, risk_patterns, stats)
                if prepared is None:
                    continue
                sample, risk_count, quality_weight = prepared
                batch_samples.append((sample, risk_count, quality_weight))
                batch_texts.append(candidate_text(sample, args.max_answer_chars_for_embedding))
                if len(batch_texts) >= args.batch_size:
                    scores_batch = embedder.score_batch(batch_texts)
                    process_score_batch(
                        batch_samples, scores_batch, queries, risk_count=None, quality_weight=None,
                        heap=heap, max_topk=max_topk, score_writer=score_writer, score_buffer=score_buffer,
                        stats=stats, args=args,
                    )
                    batch_samples = []
                    batch_texts = []
            if batch_texts:
                scores_batch = embedder.score_batch(batch_texts)
                process_score_batch(
                    batch_samples, scores_batch, queries, risk_count=None, quality_weight=None,
                    heap=heap, max_topk=max_topk, score_writer=score_writer, score_buffer=score_buffer,
                    stats=stats, args=args,
                )
        else:
            for sample in read_jsonl(args.candidates):
                prepared = prepare_candidate(sample, args, seen_questions, ad_patterns, risk_patterns, stats)
                if prepared is None:
                    continue
                sample, risk_count, quality_weight = prepared
                scores = embedder.score_text(candidate_text(sample, args.max_answer_chars_for_embedding))
                row = score_to_row(sample, scores, queries, risk_count, quality_weight)
                process_row(row, heap, max_topk, score_writer, score_buffer, stats, args)
    finally:
        if score_buffer:
            score_writer.write_rows(score_buffer)
        score_writer.close()

    sorted_rows = [item[2] for item in sorted(heap, reverse=True)]
    category_counts = Counter(row["matched_category"] for row in sorted_rows)
    matched_query_counts = Counter(row["matched_query"] for row in sorted_rows)
    output_files = write_topk_outputs(output_dir, sorted_rows, topks)
    top1k_rows = sorted_rows[: min(1000, len(sorted_rows))]
    manual_review_path = "docs/medical_project/top1k_manual_review.md"
    write_manual_review(manual_review_path, top1k_rows, sample_size=30, seed=args.seed)

    summary = {
        "candidates": args.candidates,
        "queries": args.queries,
        "embedding_model": args.embedding_model,
        "backend": embedder.backend,
        "score_formula": "0.7 * max_similarity + 0.3 * mean_top5_similarity, multiplied by quality_weight",
        "topk_list": topks,
        "output_files": output_files,
        "filter_scores_output": score_writer.output_path,
        "manual_review": manual_review_path,
        "processed_samples": stats["processed_samples"],
        "scored_samples": stats["scored_samples"],
        "filtered_samples": {
            "too_short_question": stats["too_short_question"],
            "too_short_answer": stats["too_short_answer"],
            "too_long_answer": stats["too_long_answer"],
            "duplicate_question": stats["duplicate_question"],
            "advertising_or_contact": stats["advertising_or_contact"],
        },
        "risk_downweighted_samples": stats["risk_downweighted_samples"],
        "category_distribution_top_maxk": dict(sorted(category_counts.items())),
        "matched_query_distribution_top_maxk": dict(matched_query_counts.most_common(50)),
        "notes": [
            "C-eval test is not used.",
            "Scores are embedding/reference-domain similarity signals, not medical accuracy.",
            "High-risk absolute medical expressions are downweighted; advertising/contact samples are removed.",
        ],
    }
    summary_path = output_dir.parent / "embedding" / "filter_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prepare_candidate(sample, args, seen_questions, ad_patterns, risk_patterns, stats):
    stats["processed_samples"] += 1
    question = normalize_text(sample.get("question"))
    answer = normalize_text(sample.get("answer"))
    if len(question) < args.min_question_len:
        stats["too_short_question"] += 1
        return None
    if len(answer) < args.min_answer_len:
        stats["too_short_answer"] += 1
        return None
    if len(answer) > args.max_answer_len:
        stats["too_long_answer"] += 1
        return None
    key = question.casefold()
    if key in seen_questions:
        stats["duplicate_question"] += 1
        return None
    seen_questions.add(key)

    combined = f"{question} {answer}"
    if is_advertising(combined, ad_patterns):
        stats["advertising_or_contact"] += 1
        return None

    risk_count = pattern_count(combined, risk_patterns)
    quality_weight = max(0.6, 1.0 - 0.12 * risk_count)
    if risk_count:
        stats["risk_downweighted_samples"] += 1
    sample = dict(sample)
    sample["question"] = question
    sample["answer"] = answer
    return sample, risk_count, quality_weight


def process_score_batch(
    batch_samples,
    scores_batch,
    queries,
    risk_count,
    quality_weight,
    heap,
    max_topk,
    score_writer,
    score_buffer,
    stats,
    args,
):
    for (sample, item_risk_count, item_quality_weight), scores in zip(batch_samples, scores_batch):
        row = score_to_row(sample, scores, queries, item_risk_count, item_quality_weight)
        process_row(row, heap, max_topk, score_writer, score_buffer, stats, args)


def process_row(row, heap, max_topk, score_writer, score_buffer, stats, args):
    stats["scored_samples"] += 1
    update_heap(heap, row, max_topk)
    score_buffer.append(
        {
            "id": row["id"],
            "score": row["score"],
            "base_score": row["base_score"],
            "max_similarity": row["max_similarity"],
            "mean_top5_similarity": row["mean_top5_similarity"],
            "matched_category": row["matched_category"],
            "risk_expression_count": row["risk_expression_count"],
            "quality_weight": row["quality_weight"],
        }
    )
    if len(score_buffer) >= args.score_write_batch_size:
        score_writer.write_rows(score_buffer)
        score_buffer.clear()
    if stats["scored_samples"] % args.progress_every == 0:
        print(f"scored {stats['scored_samples']} samples...", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter medical SFT data with query-sample embedding similarity."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--embedding_model", default="BAAI/bge-m3")
    parser.add_argument("--output_dir", default="data_processed/medical_project/sft")
    parser.add_argument("--topk_list", default="1000,10000,30000")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--normalize_embeddings", type=str2bool, default=True)
    parser.add_argument("--backend", choices=["auto", "sentence_transformers", "hashing"], default="auto")
    parser.add_argument("--min_question_len", type=int, default=5)
    parser.add_argument("--min_answer_len", type=int, default=20)
    parser.add_argument("--max_answer_len", type=int, default=2048)
    parser.add_argument("--max_answer_chars_for_embedding", type=int, default=512)
    parser.add_argument("--hash_dims", type=int, default=262144)
    parser.add_argument("--hash_min_ngram", type=int, default=2)
    parser.add_argument("--hash_max_ngram", type=int, default=3)
    parser.add_argument("--hash_max_chars", type=int, default=640)
    parser.add_argument("--score_write_batch_size", type=int, default=50000)
    parser.add_argument("--progress_every", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    run_filter(args)


if __name__ == "__main__":
    main()
