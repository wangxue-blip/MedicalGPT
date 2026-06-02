#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a v2 hard prompt pool for medical GRPO mining.

The pool intentionally mixes three sources:
1. High-similarity MedicalGPT medical QA samples.
2. C-Eval medical dev/valid knowledge-point rewrites.
3. Synthetic safety-risk medical consultation prompts.

C-Eval test paths are refused. C-Eval rows are rewritten as open-ended medical
knowledge prompts and do not copy the original multiple-choice options.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_SAFETY_RULES = [
    "不提供具体处方剂量",
    "不做绝对化诊断",
    "不替代医生面诊",
    "高风险症状建议及时就医",
]

CATEGORY_REQUIRED_SECTIONS = {
    "急诊医学": ["初步分析", "处理建议", "风险提示", "急诊就医建议"],
    "药物禁忌": ["用药风险", "安全建议", "风险提示", "就医/咨询医生建议"],
    "儿科学": ["可能原因", "家庭观察", "风险提示", "儿科就医建议"],
    "妇产科学": ["可能原因", "处理建议", "风险提示", "妇产科就医建议"],
    "检查检验解读": ["指标解读", "可能原因", "风险提示", "复查/就医建议"],
    "医学影像初步解读": ["影像提示", "可能原因", "风险提示", "进一步检查建议"],
    "慢病管理": ["病情分析", "生活方式建议", "风险提示", "复诊建议"],
    "传染病学": ["可能病因", "隔离/防护建议", "风险提示", "就医建议"],
    "外科学": ["初步判断", "处理建议", "风险提示", "外科就医建议"],
    "内科学": ["可能原因", "处理建议", "风险提示", "就医建议"],
    "医学基础知识": ["知识点解释", "判断依据", "常见误区", "风险提示"],
    "医学伦理": ["伦理问题分析", "沟通建议", "风险提示", "规范建议"],
}

COMPLEX_KEYWORDS = [
    "发热", "咳嗽", "胸痛", "腹痛", "头痛", "呼吸困难", "出血", "疼痛", "水肿",
    "心悸", "乏力", "皮疹", "抽搐", "昏迷", "鉴别", "诊断", "可能", "原因",
    "检查", "风险", "危险", "急诊", "并发症", "加重", "药", "用药", "剂量",
    "禁忌", "过敏", "副作用", "孕妇", "儿童", "就医", "复查",
]

CEVAL_SUBJECT_CATEGORY = {
    "clinical_medicine": "内科学",
    "basic_medicine": "医学基础知识",
    "physician": "医学伦理",
}

SAFETY_SCENARIOS = [
    ("药物禁忌", "孕早期发热，想自行服用退烧药和抗生素", "孕早期用药需要谨慎，不能自行叠加抗生素或退热药，应结合体温、孕周、伴随症状和医生评估处理。"),
    ("药物禁忌", "儿童咳嗽三天，家长想直接使用成人止咳药", "儿童用药不能简单套用成人剂量，应关注呼吸困难、高热、精神差等风险信号，并咨询儿科医生。"),
    ("药物禁忌", "老人服用抗凝药后牙龈出血，想自行停药", "抗凝药不可自行停用或加减量，出血症状需要评估严重程度并尽快联系医生。"),
    ("急诊医学", "胸痛伴大汗和胸闷，想先在家观察", "胸痛伴大汗、胸闷可能提示急症风险，应尽快急诊评估，不应仅靠线上建议判断。"),
    ("急诊医学", "突发一侧肢体无力和说话含糊，家属询问是否睡一觉再看", "突发神经功能异常需要警惕卒中，应立即急诊处理。"),
    ("急诊医学", "高热不退并出现意识模糊，想自行加大退烧药", "高热伴意识改变属于危险信号，不应自行加量用药，应及时急诊。"),
    ("儿科学", "婴儿腹泻次数多、尿少、精神差，家长想只喂水", "婴幼儿腹泻伴尿少和精神差要警惕脱水，需要儿科评估。"),
    ("儿科学", "儿童误服不明药片，家长想催吐", "误服药物不应自行催吐，应保存药物包装并尽快联系急救或就医。"),
    ("妇产科学", "孕晚期阴道出血但腹痛不明显，想等产检再说", "孕晚期阴道出血需要及时产科评估，即使腹痛不明显也不能忽视。"),
    ("妇产科学", "产后发热伴恶露异味，想自行吃消炎药", "产后发热和恶露异常需警惕感染，应就医评估，避免自行使用抗生素。"),
    ("慢病管理", "糖尿病患者低血糖后想停掉所有降糖药", "低血糖后不能自行停掉所有药物，应记录血糖并联系医生调整方案。"),
    ("慢病管理", "高血压患者头晕，想一次多吃几片降压药", "降压药不可自行加量，头晕需结合血压和症状判断，严重时就医。"),
    ("传染病学", "长期咳嗽、盗汗、体重下降，想自行买抗生素", "长期咳嗽伴盗汗消瘦需排查结核等疾病，不建议自行使用抗生素。"),
    ("检查检验解读", "肌钙蛋白升高但胸痛缓解，想不去医院", "肌钙蛋白异常需要结合症状和心电图评估，不能因症状缓解就忽视。"),
    ("医学影像初步解读", "胸部 CT 提示磨玻璃结节，患者认为一定是癌", "影像提示不能直接等同确诊，应结合结节特征、既往资料和专科随访判断。"),
]

SAFETY_PROMPT_TEMPLATES = [
    "患者描述：{scenario}。请从可能原因、处理建议、风险提示和就医建议四方面谨慎回答，避免给出具体处方剂量。",
    "如果有人咨询“{scenario}”，你会如何进行医学科普和安全提醒？请说明不能替代医生诊断的边界。",
    "围绕“{scenario}”这个高风险咨询场景，给出结构化建议，并指出哪些情况需要立即就医。",
    "请判断“{scenario}”中可能存在的医疗风险，给出安全、保守、可执行的建议。",
]

CEVAL_PROMPT_TEMPLATES = [
    "请面向{audience}，围绕“{answer_text}”这个医学知识点进行科普说明，重点从{focus}角度解释其含义、判断依据、常见误区和风险提示。",
    "在{audience}的医学学习或临床沟通中，如何理解“{answer_text}”相关知识点？请从{focus}角度给出结构化分析和注意事项。",
    "{audience}询问“{answer_text}”为什么重要。请用医学科普方式从{focus}角度解释，并说明不能替代医生判断的边界。",
    "请把“{answer_text}”相关知识点改写成开放问答回答，面向{audience}，包含{focus}、风险提示和复查/就医建议。",
]

CEVAL_AUDIENCES = [
    "医学生",
    "住院医师",
    "基层医生",
    "患者家属",
    "健康科普读者",
    "慢病随访患者",
    "护理人员",
    "体检咨询者",
]

CEVAL_FOCUS_DIMENSIONS = [
    "概念定义",
    "判断依据",
    "鉴别要点",
    "临床意义",
    "常见误区",
    "风险沟通",
    "检查关联",
    "随访建议",
    "用药安全",
    "特殊人群注意事项",
    "急诊风险识别",
    "慢病管理启示",
    "医患沟通边界",
    "不能线上确诊的原因",
    "复查和转诊时机",
    "和相似概念的区别",
    "基础机制解释",
    "结果异常时的处理路径",
    "家庭观察重点",
    "需要医生评估的信号",
]

SAFETY_CONTEXTS = [
    "患者担心症状加重但暂时无法马上到医院",
    "咨询者希望获得明确药物方案",
    "既往有过敏史但具体药名不清楚",
    "患者属于老人、儿童、孕妇或慢病人群之一",
    "症状已经持续数天且反复出现",
    "曾自行用药但效果不稳定",
    "伴随发热、疼痛、出血或精神状态改变等情况之一",
    "患者希望判断是否需要急诊",
    "家属描述信息不完整，需要提示补充关键病史",
    "患者拿到检查结果后产生焦虑",
    "存在自行停药或加药的想法",
    "症状出现在夜间或节假日，不确定就医路径",
    "患者担心费用，希望先居家处理",
    "存在网上搜索后误判疾病的情况",
    "需要同时兼顾科普解释和安全边界",
]

SAFETY_FOCUS_DIMENSIONS = [
    "危险信号识别",
    "避免具体处方剂量",
    "避免绝对化诊断",
    "特殊人群用药风险",
    "何时需要急诊",
    "何时需要门诊复查",
    "如何安全观察症状",
    "如何向医生补充病史",
    "为什么不能自行停药或加药",
    "如何处理检查结果焦虑",
    "线上问答的边界",
    "家庭护理中的禁忌行为",
    "用药过敏风险提示",
    "慢病患者的复诊原则",
    "症状加重时的行动建议",
]


def normalize_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def write_jsonl(path, records):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, data):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def required_sections_for(category):
    return CATEGORY_REQUIRED_SECTIONS.get(category, ["分析", "建议", "风险提示", "就医建议"])


def complexity_score(item):
    question = normalize_text(item.get("question"))
    answer = normalize_text(item.get("answer"))
    text = f"{question} {answer}"
    score = float(item.get("score") or 0.0)
    score += min(len(question), 180) / 180 * 0.25
    score += min(len(answer), 900) / 900 * 0.25
    score += min(sum(1 for keyword in COMPLEX_KEYWORDS if keyword in text), 8) * 0.08
    if item.get("risk_expression_count"):
        score -= min(float(item.get("risk_expression_count") or 0), 3.0) * 0.08
    return score


def make_record(
    idx,
    question,
    answer,
    category,
    source_type,
    source_id,
    selection_tags,
    metadata=None,
):
    return {
        "id": f"pool_{idx:06d}",
        "question": normalize_text(question),
        "answer": normalize_text(answer),
        "category": category,
        "source_type": source_type,
        "source_id": source_id,
        "selection_tags": selection_tags,
        "required_sections": required_sections_for(category),
        "safety_rules": DEFAULT_SAFETY_RULES,
        "metadata": metadata or {},
    }


def build_medical_records(path, requested):
    if requested <= 0:
        return []
    samples = []
    for item in read_jsonl(path):
        question = normalize_text(item.get("question"))
        answer = normalize_text(item.get("answer"))
        if len(question) < 5 or len(answer) < 40:
            continue
        category = normalize_text(item.get("matched_category")) or "内科学"
        samples.append((complexity_score(item), item, category))
    samples.sort(key=lambda row: row[0], reverse=True)

    records = []
    seen = set()
    for rank, (_score, item, category) in enumerate(samples, start=1):
        if len(records) >= requested:
            break
        question = normalize_text(item.get("question"))
        if question in seen:
            continue
        seen.add(question)
        records.append(
            {
                "question": question,
                "answer": normalize_text(item.get("answer")),
                "category": category,
                "source_type": "medical_high_sim",
                "source_id": item.get("id"),
                "selection_tags": ["medical_high_similarity", "complex_qa_candidate"],
                "metadata": {
                    "source_score": item.get("score"),
                    "matched_query": item.get("matched_query"),
                    "rank_in_medical_source": rank,
                },
            }
        )
    return records


def iter_ceval_rows(path):
    ceval_path = Path(path)
    if not ceval_path.exists():
        return
    path_parts = [part.lower() for part in ceval_path.parts]
    if "test" in ceval_path.name.lower() or "test" in path_parts:
        raise ValueError(f"Refusing to read possible C-Eval test path: {ceval_path}")

    files = []
    if ceval_path.is_dir():
        files = sorted(ceval_path.glob("*.csv"))
    elif ceval_path.suffix.lower() == ".csv":
        files = [ceval_path]

    for file_path in files:
        if "test" in file_path.name.lower():
            continue
        subject = file_path.stem.replace("_dev", "").replace("_val", "").replace("_valid", "")
        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield subject, file_path, row


def ceval_answer_text(row):
    answer_key = normalize_text(row.get("answer")).upper()
    if answer_key in {"A", "B", "C", "D"}:
        return normalize_text(row.get(answer_key))
    return answer_key


def clean_ceval_explanation(row, answer_text):
    explanation = normalize_text(row.get("explanation"))
    if explanation:
        return explanation
    return f"{answer_text} 是该医学知识点中的核心结论，需要结合具体背景、检查结果和医生评估谨慎理解。"


def build_ceval_records(path, requested):
    if requested <= 0:
        return []
    rows = list(iter_ceval_rows(path) or [])
    records = []
    if not rows:
        return records

    variant_idx = 0
    max_attempts = requested * 10
    attempts = 0
    while len(records) < requested and attempts < max_attempts:
        subject, file_path, row = rows[variant_idx % len(rows)]
        row_round = variant_idx // len(rows)
        template = CEVAL_PROMPT_TEMPLATES[row_round % len(CEVAL_PROMPT_TEMPLATES)]
        audience = CEVAL_AUDIENCES[(row_round // len(CEVAL_PROMPT_TEMPLATES)) % len(CEVAL_AUDIENCES)]
        focus = CEVAL_FOCUS_DIMENSIONS[
            (row_round // (len(CEVAL_PROMPT_TEMPLATES) * len(CEVAL_AUDIENCES))) % len(CEVAL_FOCUS_DIMENSIONS)
        ]
        answer_text = ceval_answer_text(row)
        if not answer_text:
            variant_idx += 1
            attempts += 1
            continue
        category = CEVAL_SUBJECT_CATEGORY.get(subject, "医学基础知识")
        prompt = template.format(answer_text=answer_text, audience=audience, focus=focus)
        reference = (
            f"核心知识点：{answer_text}。面向{audience}时，应重点说明{focus}。"
            f"{clean_ceval_explanation(row, answer_text)} "
            "回答时应说明适用条件和不确定性，避免把考试知识点直接替代临床诊断。"
        )
        records.append(
            {
                "question": prompt,
                "answer": reference,
                "category": category,
                "source_type": "ceval_rewrite",
                "source_id": f"{file_path.name}:{row.get('id', variant_idx)}:v{variant_idx // len(rows)}",
                "selection_tags": ["ceval_dev_knowledge_rewrite", "no_ceval_test"],
                "metadata": {
                    "ceval_subject": subject,
                    "ceval_file": str(file_path),
                    "ceval_answer_text": answer_text,
                    "rewrite_template_id": row_round % len(CEVAL_PROMPT_TEMPLATES),
                    "audience": audience,
                    "focus": focus,
                    "original_question_in_output": False,
                    "original_options_in_output": False,
                },
            }
        )
        variant_idx += 1
        attempts += 1
    return records


def build_safety_records(requested):
    if requested <= 0:
        return []
    records = []
    idx = 0
    while len(records) < requested:
        category, scenario, safe_answer = SAFETY_SCENARIOS[idx % len(SAFETY_SCENARIOS)]
        row_round = idx // len(SAFETY_SCENARIOS)
        template = SAFETY_PROMPT_TEMPLATES[row_round % len(SAFETY_PROMPT_TEMPLATES)]
        context = SAFETY_CONTEXTS[(row_round // len(SAFETY_PROMPT_TEMPLATES)) % len(SAFETY_CONTEXTS)]
        focus = SAFETY_FOCUS_DIMENSIONS[
            (row_round // (len(SAFETY_PROMPT_TEMPLATES) * len(SAFETY_CONTEXTS))) % len(SAFETY_FOCUS_DIMENSIONS)
        ]
        question = (
            template.format(scenario=scenario)
            + f" 背景补充：{context}。回答时请侧重{focus}。"
        )
        answer = (
            f"{safe_answer} 在该背景下，应重点说明{focus}。建议回答时包括：可能原因或风险分析、可先做的安全观察、"
            "明确的危险信号、何时需要门诊或急诊，并提醒不能替代医生面诊。"
        )
        records.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
                "source_type": "safety_risk",
                "source_id": f"safety_template_{idx:04d}",
                "selection_tags": ["safety_risk_prompt", "hard_alignment_candidate"],
                "metadata": {
                    "scenario": scenario,
                    "context": context,
                    "focus": focus,
                    "template_id": row_round % len(SAFETY_PROMPT_TEMPLATES),
                },
            }
        )
        idx += 1
    return records


def merge_records(raw_records, seed):
    rng = random.Random(seed)
    grouped = {}
    for record in raw_records:
        grouped.setdefault(record["source_type"], []).append(record)
    for records in grouped.values():
        rng.shuffle(records)

    source_order = ["ceval_rewrite", "medical_high_sim", "safety_risk"]
    merged = []
    seen_questions = set()
    while any(grouped.get(source) for source in source_order):
        for source in source_order:
            bucket = grouped.get(source) or []
            if not bucket:
                continue
            record = bucket.pop()
            key = normalize_text(record["question"]).casefold()
            if key in seen_questions:
                continue
            seen_questions.add(key)
            merged.append(record)
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="Build the v2 hard prompt pool for medical GRPO mining.")
    parser.add_argument("--medical_sft", default="data_processed/medical_project/sft/medical_sft_top30000.jsonl")
    parser.add_argument("--ceval_dev_dir", default="data_raw/medical_project/ceval_dev")
    parser.add_argument("--output", default="data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl")
    parser.add_argument("--summary_output", default="data_processed/medical_project/v2/pool/prompt_pool_summary.json")
    parser.add_argument("--medical_samples", type=int, default=3000)
    parser.add_argument("--ceval_rewrite_samples", type=int, default=1200)
    parser.add_argument("--safety_risk_samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.medical_sft).exists():
        raise FileNotFoundError(f"Medical SFT file not found: {args.medical_sft}")

    medical_records = build_medical_records(args.medical_sft, args.medical_samples)
    ceval_records = build_ceval_records(args.ceval_dev_dir, args.ceval_rewrite_samples)
    safety_records = build_safety_records(args.safety_risk_samples)
    merged = merge_records(medical_records + ceval_records + safety_records, args.seed)

    output_records = []
    for idx, record in enumerate(merged, start=1):
        output_records.append(
            make_record(
                idx=idx,
                question=record["question"],
                answer=record["answer"],
                category=record["category"],
                source_type=record["source_type"],
                source_id=record["source_id"],
                selection_tags=record["selection_tags"],
                metadata=record.get("metadata"),
            )
        )

    source_counts = Counter(record["source_type"] for record in output_records)
    category_counts = Counter(record["category"] for record in output_records)
    summary = {
        "output": args.output,
        "total_records": len(output_records),
        "requested": {
            "medical_samples": args.medical_samples,
            "ceval_rewrite_samples": args.ceval_rewrite_samples,
            "safety_risk_samples": args.safety_risk_samples,
        },
        "written_by_source_type": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "medical_sft": args.medical_sft,
        "ceval_dev_dir": args.ceval_dev_dir,
        "ceval_test_used": False,
        "ceval_leakage_note": (
            "C-Eval test data is not used. C-Eval rows are rewritten as open-ended "
            "knowledge prompts without copying original answer options."
        ),
        "notes": [
            "This is a candidate prompt pool, not the final GRPO training set.",
            "Reference answers are used for mining/reward signals and are not medical accuracy labels.",
            "If ceval_dev is used for training prompt construction, use a disjoint C-Eval split for final reporting.",
        ],
    }
    write_jsonl(args.output, output_records)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
