#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build small-scale medical GRPO alignment data."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_REQUIRED_SECTIONS = ["分析", "建议", "风险提示", "就医建议"]
DEFAULT_SAFETY_RULES = [
    "不提供具体处方剂量",
    "不做绝对化诊断",
    "不替代医生面诊",
    "高风险症状建议及时就医",
]

COMPLEX_KEYWORDS = {
    "symptom": [
        "发热", "咳嗽", "胸痛", "腹痛", "头痛", "头晕", "呕吐", "腹泻", "呼吸困难",
        "出血", "疼痛", "水肿", "心悸", "乏力", "皮疹", "抽搐", "昏迷",
    ],
    "differential": ["鉴别", "诊断", "可能", "原因", "区别", "排除", "检查", "体征", "影像", "化验"],
    "risk": ["风险", "危险", "严重", "急诊", "并发症", "恶化", "加重", "出血", "休克", "癌", "肿瘤"],
    "drug": ["药", "用药", "剂量", "抗生素", "禁忌", "过敏", "副作用", "停药", "孕妇", "儿童"],
    "visit": ["就医", "医院", "门诊", "急诊", "医生", "复查", "手术", "治疗"],
}

CATEGORY_REQUIRED_SECTIONS = {
    "急诊医学": ["初步判断", "处理建议", "风险提示", "急诊就医建议"],
    "药物禁忌": ["用药风险", "安全建议", "风险提示", "就医/咨询医生建议"],
    "儿科学": ["可能原因", "家庭观察", "风险提示", "儿科就医建议"],
    "妇产科学": ["可能原因", "处理建议", "风险提示", "妇产科就医建议"],
    "检查检验解读": ["指标解读", "可能原因", "风险提示", "复查/就医建议"],
    "医学影像初步解读": ["影像提示", "可能原因", "风险提示", "进一步检查建议"],
}


def normalize_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).strip()


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


def keyword_hits(text, keywords):
    return sum(1 for keyword in keywords if keyword in text)


def infer_category(sample):
    category = normalize_text(sample.get("matched_category"))
    if category:
        return category
    query = normalize_text(sample.get("matched_query"))
    if "儿童" in query or "儿科" in query:
        return "儿科学"
    if "孕" in query or "妇" in query or "月经" in query:
        return "妇产科学"
    if "药" in query or "禁忌" in query:
        return "药物禁忌"
    if "急诊" in query or "胸痛" in query or "呼吸困难" in query:
        return "急诊医学"
    return "内科学"


def complexity_score(sample, prefer_categories):
    question = normalize_text(sample.get("question"))
    answer = normalize_text(sample.get("answer"))
    text = f"{question} {answer}"
    category = infer_category(sample)
    score = float(sample.get("score") or 0.0)

    score += min(len(question), 180) / 180 * 0.25
    score += min(len(answer), 900) / 900 * 0.20
    score += min(keyword_hits(text, COMPLEX_KEYWORDS["symptom"]), 4) * 0.08
    score += min(keyword_hits(text, COMPLEX_KEYWORDS["differential"]), 4) * 0.09
    score += min(keyword_hits(text, COMPLEX_KEYWORDS["risk"]), 4) * 0.10
    score += min(keyword_hits(text, COMPLEX_KEYWORDS["drug"]), 4) * 0.08
    score += min(keyword_hits(text, COMPLEX_KEYWORDS["visit"]), 4) * 0.08
    if category in prefer_categories:
        score += 0.25
    if sample.get("risk_expression_count", 0):
        score -= min(float(sample.get("risk_expression_count", 0)), 3.0) * 0.08
    return score


def required_sections_for(category, sample):
    if category in CATEGORY_REQUIRED_SECTIONS:
        return CATEGORY_REQUIRED_SECTIONS[category]
    text = f"{sample.get('question', '')} {sample.get('answer', '')}"
    if keyword_hits(text, COMPLEX_KEYWORDS["drug"]):
        return CATEGORY_REQUIRED_SECTIONS["药物禁忌"]
    if keyword_hits(text, COMPLEX_KEYWORDS["risk"]) >= 2:
        return ["初步分析", "处理建议", "风险提示", "就医建议"]
    return DEFAULT_REQUIRED_SECTIONS


def make_grpo_sample(sample, category):
    return {
        "question": normalize_text(sample.get("question")),
        "answer": normalize_text(sample.get("answer")),
        "category": category,
        "required_sections": required_sections_for(category, sample),
        "safety_rules": DEFAULT_SAFETY_RULES,
        "source_id": sample.get("id"),
        "source_score": sample.get("score"),
        "matched_query": sample.get("matched_query"),
    }


def select_samples(samples, num_samples, prefer_categories):
    ranked = sorted(
        samples,
        key=lambda sample: complexity_score(sample, prefer_categories),
        reverse=True,
    )

    selected = []
    seen_questions = set()
    category_counts = Counter()
    min_per_prefer = 1 if num_samples < 100 else min(80, max(10, num_samples // 12))

    for category in prefer_categories:
        for sample in ranked:
            sample_category = infer_category(sample)
            question = normalize_text(sample.get("question"))
            if sample_category != category or question in seen_questions:
                continue
            selected.append(make_grpo_sample(sample, sample_category))
            seen_questions.add(question)
            category_counts[sample_category] += 1
            if category_counts[sample_category] >= min_per_prefer:
                break

    for sample in ranked:
        if len(selected) >= num_samples:
            break
        question = normalize_text(sample.get("question"))
        answer = normalize_text(sample.get("answer"))
        if not question or not answer or question in seen_questions:
            continue
        category = infer_category(sample)
        selected.append(make_grpo_sample(sample, category))
        seen_questions.add(question)
        category_counts[category] += 1

    return selected


def write_jsonl(path, records):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_data(args):
    prefer_categories = [item.strip() for item in args.prefer_categories.split(",") if item.strip()]
    samples = list(read_jsonl(args.sft_data))
    selected = select_samples(samples, args.num_samples, prefer_categories)
    if len(selected) < args.num_samples:
        raise ValueError(f"Only selected {len(selected)} samples, expected {args.num_samples}.")

    write_jsonl(args.output, selected)
    category_counts = Counter(item["category"] for item in selected)
    required_section_counts = Counter(section for item in selected for section in item["required_sections"])
    summary = {
        "sft_data": args.sft_data,
        "output": args.output,
        "num_samples": len(selected),
        "prefer_categories": prefer_categories,
        "category_counts": dict(sorted(category_counts.items())),
        "required_section_counts": dict(required_section_counts.most_common()),
        "safety_rules": DEFAULT_SAFETY_RULES,
        "construction_strategy": [
            "Rank filtered SFT samples by source score, question/answer length, and medical complexity keywords.",
            "Prefer emergency, internal medicine, drug contraindication, pediatrics, and obstetrics/gynecology categories.",
            "Keep the original reference answer as the similarity reference for GRPO reward.",
            "Add required_sections and safety_rules; chosen/rejected pairs are not required for GRPO.",
        ],
    }
    summary_path = Path(args.summary_output) if args.summary_output else Path(args.output).parent / "grpo_data_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select complex medical QA samples and annotate GRPO metadata."
    )
    parser.add_argument("--sft_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument(
        "--prefer_categories",
        default="急诊医学,内科学,药物禁忌,儿科学,妇产科学",
    )
    parser.add_argument("--summary_output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    build_data(args)


if __name__ == "__main__":
    main()
