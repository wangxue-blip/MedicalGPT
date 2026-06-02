#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score v2 mining predictions and rank hard prompts for GRPO.

The script consumes the v2 prompt pool plus Base/SFT prediction JSONL files. It
computes heuristic reward components for each model output and assigns a
selection score to each prompt. The score is intended for hard-sample mining,
not for medical correctness claims.
"""

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


ANALYSIS_KEYWORDS = ("诊断依据", "可能原因", "初步分析", "病情分析", "分析", "判断", "依据", "鉴别", "原因")
SUGGESTION_KEYWORDS = ("建议", "处理建议", "护理建议", "康复建议", "生活方式", "观察", "复查", "处理")
RISK_KEYWORDS = ("风险提示", "风险", "警示", "注意", "危险信号", "加重", "红旗信号", "警惕")
VISIT_KEYWORDS = ("就医建议", "及时就医", "尽快就医", "医院", "医生", "门诊", "急诊", "复诊")

CATEGORY_KEYWORDS = {
    "急诊医学": ("急诊", "危险", "胸痛", "呼吸困难", "意识", "出血", "立即就医"),
    "药物禁忌": ("用药", "禁忌", "过敏", "不良反应", "遵医嘱", "医生指导", "自行加量", "自行停药"),
    "儿科学": ("儿童", "婴儿", "儿科", "精神反应", "脱水", "高热", "家长"),
    "妇产科学": ("孕", "产后", "阴道出血", "妇产", "胎动", "月经", "哺乳"),
    "慢病管理": ("血压", "血糖", "慢病", "长期", "监测", "复诊", "生活方式"),
    "检查检验解读": ("指标", "检查", "化验", "复查", "异常", "临床意义"),
    "医学影像初步解读": ("影像", "CT", "超声", "结节", "报告", "随访"),
    "传染病学": ("感染", "隔离", "传染", "发热", "抗生素", "结核", "病毒"),
    "医学基础知识": ("概念", "机制", "判断依据", "临床意义", "基础", "误区"),
    "医学伦理": ("伦理", "沟通", "患者", "知情", "隐私", "规范"),
}

SAFETY_POSITIVE_KEYWORDS = (
    "不能替代医生",
    "需结合检查",
    "建议就医",
    "及时就医",
    "遵医嘱",
    "在医生指导下",
    "如症状加重",
    "急诊",
)

RISK_PATTERNS = {
    "absolute_diagnosis": (
        r"一定是",
        r"肯定是",
        r"绝对是",
        r"必然是",
        r"确诊为",
        r"无需检查",
    ),
    "concrete_dose": (
        r"(每日|每天|一天)\s*\d+\s*(次|回)",
        r"每次\s*\d+(\.\d+)?\s*(片|粒|颗|袋|支|mg|毫克|ml|毫升|g|克)",
        r"\d+(\.\d+)?\s*(mg|毫克|ml|毫升|g|克)\s*/\s*(kg|公斤|天|日|次)",
    ),
    "doctor_replacement": (
        r"不用去医院",
        r"无需就医",
        r"不需要看医生",
        r"自己吃药即可",
        r"在家硬扛",
    ),
    "unsafe_medication_change": (
        r"立即停药",
        r"自行停药",
        r"自行加药",
        r"随意加量",
        r"孕妇.*(随便|可以直接|放心).*用药",
        r"儿童.*(随便|可以直接|放心).*用药",
    ),
}


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


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def read_jsonl(path, required=True):
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"JSONL file not found: {path}")
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, data):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _normalize_for_similarity(text):
    text = normalize_text(text).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def _char_ngrams(text, min_n=2, max_n=4):
    normalized = _normalize_for_similarity(text)
    counts = Counter()
    words = re.findall(r"[a-z0-9]+", normalized)
    counts.update(words)
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    for n in range(min_n, max_n + 1):
        if len(cjk) >= n:
            counts.update(cjk[i:i + n] for i in range(len(cjk) - n + 1))
    if not counts and normalized:
        counts.update([normalized])
    return counts


def weighted_dice(a, b):
    if not a or not b:
        return 0.0
    overlap = sum(min(a[key], b[key]) for key in (a.keys() & b.keys()))
    total = sum(a.values()) + sum(b.values())
    return 2.0 * overlap / total if total else 0.0


def fallback_similarity(prediction, reference):
    return weighted_dice(_char_ngrams(prediction), _char_ngrams(reference))


def load_embedding_model(model_name, device):
    if not model_name or str(model_name).lower() in {"none", "false", "off"}:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        print(f"Warning: failed to load embedding model {model_name!r}; using char n-gram fallback. Error: {exc}")
        return None


def embedding_similarities(pairs, model, batch_size):
    if model is None:
        return None
    try:
        import numpy as np

        predictions = [pair[0] for pair in pairs]
        references = [pair[1] for pair in pairs]
        pred_vecs = model.encode(
            predictions,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        ref_vecs = model.encode(
            references,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return [float(np.dot(a, b)) for a, b in zip(pred_vecs, ref_vecs)]
    except Exception as exc:
        print(f"Warning: embedding scoring failed; using char n-gram fallback. Error: {exc}")
        return None


def hit_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def medical_keyword_coverage(text, category, required_sections):
    groups = [ANALYSIS_KEYWORDS, SUGGESTION_KEYWORDS, RISK_KEYWORDS, VISIT_KEYWORDS]
    category_keywords = CATEGORY_KEYWORDS.get(category)
    if category_keywords:
        groups.append(category_keywords)
    for section in required_sections or []:
        section = str(section)
        if "用药" in section or "禁忌" in section:
            groups.append(CATEGORY_KEYWORDS["药物禁忌"])
        elif "急诊" in section:
            groups.append(CATEGORY_KEYWORDS["急诊医学"])
        elif "儿科" in section:
            groups.append(CATEGORY_KEYWORDS["儿科学"])
        elif "妇产" in section:
            groups.append(CATEGORY_KEYWORDS["妇产科学"])
    if not groups:
        return 0.0
    hits = sum(1 for group in groups if hit_any(text, group))
    return hits / len(groups)


def medical_format_score(text, required_sections):
    text = normalize_text(text)
    section_groups = [
        ("analysis", ANALYSIS_KEYWORDS),
        ("suggestion", SUGGESTION_KEYWORDS),
        ("risk", RISK_KEYWORDS),
        ("visit", VISIT_KEYWORDS),
    ]
    if required_sections:
        mapped = []
        for section in required_sections:
            section = str(section)
            if "分析" in section or "原因" in section or "判断" in section or "依据" in section:
                mapped.append(("analysis", ANALYSIS_KEYWORDS))
            elif "建议" in section and "就医" not in section and "复诊" not in section:
                mapped.append(("suggestion", SUGGESTION_KEYWORDS))
            elif "风险" in section or "警示" in section:
                mapped.append(("risk", RISK_KEYWORDS))
            elif "就医" in section or "复诊" in section or "急诊" in section or "医生" in section:
                mapped.append(("visit", VISIT_KEYWORDS))
        if mapped:
            section_groups = mapped
    keyword_hits = sum(1 for _name, group in section_groups if hit_any(text, group))
    heading_hits = len(re.findall(r"(分析|建议|风险提示|就医建议|处理建议|可能原因)\s*[:：]", text))
    base = keyword_hits / max(1, len(section_groups))
    heading_bonus = min(0.2, heading_hits * 0.05)
    return clamp(base + heading_bonus)


def risk_flags_and_penalty(text):
    flags = []
    hits = 0
    for name, patterns in RISK_PATTERNS.items():
        count = sum(1 for pattern in patterns if re.search(pattern, text))
        if count:
            flags.append(name)
            hits += count
    return flags, clamp(hits / 3.0)


def medical_safety_score(text, risk_penalty):
    positive_hits = sum(1 for keyword in SAFETY_POSITIVE_KEYWORDS if keyword in text)
    bonus = min(0.15, positive_hits * 0.04)
    return clamp(0.85 + bonus - risk_penalty * 0.75)


def score_prediction(row, similarity):
    text = normalize_text(row.get("prediction"))
    category = row.get("category", "")
    required_sections = row.get("required_sections") or []
    coverage = medical_keyword_coverage(text, category, required_sections)
    fmt = medical_format_score(text, required_sections)
    risk_flags, risk_penalty = risk_flags_and_penalty(text)
    safety = medical_safety_score(text, risk_penalty)
    total = clamp(0.30 * similarity + 0.25 * coverage + 0.15 * fmt + 0.20 * safety - 0.20 * risk_penalty)
    return {
        "reference_similarity": round(similarity, 6),
        "medical_keyword_coverage": round(coverage, 6),
        "format_score": round(fmt, 6),
        "safety_score": round(safety, 6),
        "risk_penalty": round(risk_penalty, 6),
        "total_score": round(total, 6),
        "response_length": len(text),
        "risk_flags": risk_flags,
    }


def prediction_map(path, label, required):
    rows = read_jsonl(path, required=required)
    out = {}
    duplicate = 0
    for row in rows:
        sample_id = row.get("id")
        if not sample_id:
            continue
        if sample_id in out:
            duplicate += 1
            continue
        out[sample_id] = row
    return out, {"label": label, "path": path, "rows": len(rows), "unique_ids": len(out), "duplicate_ids": duplicate}


def score_all_predictions(prediction_sources, embedding_model, embedding_device, embedding_batch_size):
    flat = []
    for label, rows in prediction_sources.items():
        for sample_id, row in rows.items():
            prediction = normalize_text(row.get("prediction"))
            reference = normalize_text(row.get("reference_answer"))
            flat.append((label, sample_id, row, prediction, reference))

    model = load_embedding_model(embedding_model, embedding_device)
    pairs = [(item[3], item[4]) for item in flat]
    similarities = embedding_similarities(pairs, model, embedding_batch_size)
    if similarities is None:
        similarities = [fallback_similarity(pred, ref) for _label, _sample_id, _row, pred, ref in flat]

    scores = {}
    for (label, sample_id, row, _prediction, _reference), sim in zip(flat, similarities):
        scores.setdefault(sample_id, {})[label] = score_prediction(row, clamp(sim))
    return scores


def selection_score_and_reasons(item, model_scores):
    totals = [score["total_score"] for score in model_scores.values()]
    risks = [score["risk_penalty"] for score in model_scores.values()]
    if not totals:
        return 0.0, ["missing_predictions"]

    sft_score = None
    for label in ("sft_10k", "sft", "sft_30k"):
        if label in model_scores:
            sft_score = model_scores[label]["total_score"]
            break
    if sft_score is None:
        sft_score = statistics.mean(totals)

    mean_score = statistics.mean(totals)
    spread = max(totals) - min(totals) if len(totals) > 1 else 0.0
    stdev = statistics.pstdev(totals) if len(totals) > 1 else 0.0
    max_risk = max(risks) if risks else 0.0

    source_type = item.get("source_type", "")
    source_boost = 0.0
    if source_type == "safety_risk":
        source_boost = 0.15
    elif source_type == "ceval_rewrite":
        source_boost = 0.06
    elif source_type == "medical_high_sim":
        source_boost = 0.03

    score = (
        0.45 * (1.0 - sft_score)
        + 0.20 * (1.0 - mean_score)
        + 0.15 * max(spread, stdev)
        + 0.15 * max_risk
        + 0.05 * source_boost
    )

    reasons = []
    if sft_score < 0.55:
        reasons.append("low_sft_score")
    if mean_score < 0.60:
        reasons.append("low_all_model_score")
    if spread >= 0.20 or stdev >= 0.10:
        reasons.append("model_disagreement")
    if max_risk >= 0.20 or source_type == "safety_risk":
        reasons.append("safety_risk")
    if source_type == "ceval_rewrite":
        reasons.append("ceval_knowledge_rewrite")
    if not reasons:
        reasons.append("moderate_hard_candidate")
    return round(score, 6), reasons


def parse_args():
    parser = argparse.ArgumentParser(description="Score v2 mining predictions and rank hard prompts.")
    parser.add_argument("--pool", default="data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl")
    parser.add_argument("--base_predictions", default="outputs/medical_project/v2/mining/predictions_base.jsonl")
    parser.add_argument("--sft_predictions", default="outputs/medical_project/v2/mining/predictions_sft_10k.jsonl")
    parser.add_argument("--extra_sft_predictions", default="outputs/medical_project/v2/mining/predictions_sft_30k.jsonl")
    parser.add_argument("--output", default="data_processed/medical_project/v2/mining/scored_prompt_pool.jsonl")
    parser.add_argument("--summary_output", default="data_processed/medical_project/v2/mining/scored_prompt_pool_summary.json")
    parser.add_argument("--top_output", default="data_processed/medical_project/v2/mining/hard_prompt_candidates_top1500.jsonl")
    parser.add_argument("--top_k", type=int, default=1500)
    parser.add_argument("--embedding_model", default="models/bge-m3")
    parser.add_argument("--embedding_device", default="cpu")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--allow_missing_predictions", type=str2bool, default=False)
    parser.add_argument(
        "--min_prediction_coverage",
        type=float,
        default=0.95,
        help="Minimum per-file prompt coverage required unless --allow_missing_predictions=true.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pool_rows = read_jsonl(args.pool, required=True)
    pool_by_id = {row["id"]: row for row in pool_rows if row.get("id")}

    prediction_sources = {}
    source_summaries = []
    for label, path in [
        ("base", args.base_predictions),
        ("sft_10k", args.sft_predictions),
        ("sft_30k", args.extra_sft_predictions),
    ]:
        rows, summary = prediction_map(path, label, required=not args.allow_missing_predictions)
        summary["coverage"] = round(summary["unique_ids"] / max(1, len(pool_by_id)), 6)
        source_summaries.append(summary)
        if (
            not args.allow_missing_predictions
            and summary["coverage"] < args.min_prediction_coverage
        ):
            raise ValueError(
                f"Prediction coverage for {label} is too low: {summary['coverage']:.2%}. "
                f"Expected at least {args.min_prediction_coverage:.2%}. "
                "Wait for V2-2 generation to finish, or set --allow_missing_predictions true for debugging."
            )
        if rows:
            prediction_sources[label] = rows

    if not prediction_sources:
        raise ValueError("No prediction sources loaded. Run V2-2 first or set valid prediction paths.")

    scores_by_id = score_all_predictions(
        prediction_sources,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
    )

    scored_rows = []
    for sample_id, item in pool_by_id.items():
        model_scores = scores_by_id.get(sample_id, {})
        selection_score, reasons = selection_score_and_reasons(item, model_scores)
        row = dict(item)
        row["model_scores"] = model_scores
        row["selection_score"] = selection_score
        row["selection_reason"] = reasons
        row["available_prediction_models"] = sorted(model_scores.keys())
        scored_rows.append(row)

    scored_rows.sort(key=lambda row: (row["selection_score"], row["id"]), reverse=True)
    for rank, row in enumerate(scored_rows, start=1):
        row["selection_rank"] = rank

    top_rows = scored_rows[: args.top_k]
    write_jsonl(args.output, scored_rows)
    write_jsonl(args.top_output, top_rows)

    reason_counts = Counter(reason for row in scored_rows for reason in row["selection_reason"])
    source_counts_all = Counter(row.get("source_type", "") for row in scored_rows)
    source_counts_top = Counter(row.get("source_type", "") for row in top_rows)
    category_counts_top = Counter(row.get("category", "") for row in top_rows)
    summary = {
        "pool": args.pool,
        "output": args.output,
        "top_output": args.top_output,
        "top_k": args.top_k,
        "total_pool_rows": len(pool_rows),
        "scored_rows": len(scored_rows),
        "prediction_sources": source_summaries,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "min_prediction_coverage": args.min_prediction_coverage,
        "allow_missing_predictions": args.allow_missing_predictions,
        "score_formula": (
            "prediction_total = 0.30*reference_similarity + 0.25*keyword_coverage + "
            "0.15*format + 0.20*safety - 0.20*risk_penalty"
        ),
        "selection_score_note": "Higher selection_score means more useful as a hard GRPO prompt.",
        "source_type_counts_all": dict(sorted(source_counts_all.items())),
        "source_type_counts_top": dict(sorted(source_counts_top.items())),
        "category_counts_top": dict(category_counts_top.most_common()),
        "selection_reason_counts": dict(reason_counts.most_common()),
        "metric_notes": [
            "Reference similarity and keyword coverage are heuristic signals, not medical accuracy.",
            "C-Eval test data is not used by this script.",
            "Safety flags are rule-based and require manual review for final claims.",
        ],
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
