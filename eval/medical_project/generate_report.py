#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the MedicalGPT medical project experiment report."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


MODEL_DISPLAY = {
    "Qwen2.5-7B-Instruct": "Base: Qwen2.5-7B-Instruct",
    "qwen25_7b_lora_10k_v100": "SFT 10k LoRA V100",
    "qwen25_7b_lora_30k_v100": "SFT 30k LoRA V100",
    "qwen25_7b_lora_10k_v100_grpo_500": "GRPO 500 from SFT10k V100",
    "qwen25_7b_lora_10k_v100_grpo_1000": "GRPO 1000 from SFT10k V100",
}

SFT_RUNS = {
    "SFT smoke 1k": "outputs/medical_project/sft/qwen25_7b_lora_smoke_1k",
    "SFT 10k V100": "outputs/medical_project/sft/qwen25_7b_lora_10k_v100",
    "SFT 30k V100": "outputs/medical_project/sft/qwen25_7b_lora_30k_v100",
    "SFT 10k A100 1GPU": "outputs/medical_project/sft/qwen25_7b_lora_10k_a100_1gpu",
    "SFT 30k A100 1GPU": "outputs/medical_project/sft/qwen25_7b_lora_30k_a100_1gpu",
}

GRPO_RUNS = {
    "GRPO smoke": "outputs/medical_project/grpo/qwen25_7b_lora_10k_v100_grpo_smoke",
    "GRPO 500 from SFT10k V100": "outputs/medical_project/grpo/qwen25_7b_lora_10k_v100_grpo_500",
    "GRPO 1000 from SFT10k V100": "outputs/medical_project/grpo/qwen25_7b_lora_10k_v100_grpo_1000",
    "GRPO 1000 from SFT30k A100 1GPU": "outputs/medical_project/grpo/qwen25_7b_lora_30k_v100_grpo_1000_a100_1gpu",
}


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path, limit=None):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=4, default="-"):
    value = as_float(value, None)
    if value is None:
        return default
    return f"{value:.{digits}f}"


def pct(value, digits=1, default="-"):
    value = as_float(value, None)
    if value is None:
        return default
    return f"{value * 100:.{digits}f}%"


def intfmt(value, default="-"):
    value = as_float(value, None)
    if value is None:
        return default
    return f"{int(value):,}"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def read_run_results(path):
    p = Path(path)
    train = load_json(p / "train_results.json", {})
    eval_ = load_json(p / "eval_results.json", {})
    all_ = load_json(p / "all_results.json", {})
    adapter_exists = (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()
    return {
        "exists": p.exists(),
        "adapter_exists": adapter_exists,
        "train": train or {},
        "eval": eval_ or {},
        "all": all_ or {},
    }


def training_rows(run_map):
    rows = []
    for name, path in run_map.items():
        result = read_run_results(path)
        if not result["exists"]:
            continue
        train = result["train"]
        eval_ = result["eval"]
        rows.append([
            name,
            "yes" if result["adapter_exists"] else "checkpoint only",
            intfmt(train.get("train_samples")),
            fmt(train.get("train_loss"), 4),
            fmt(eval_.get("eval_loss"), 4),
            fmt(eval_.get("perplexity"), 4),
            fmt(train.get("train_runtime"), 1),
            fmt(train.get("train_samples_per_second"), 3),
        ])
    return rows


def eval_rows(metrics):
    ordered = sorted(
        metrics,
        key=lambda row: [
            "Qwen2.5-7B-Instruct",
            "qwen25_7b_lora_10k_v100",
            "qwen25_7b_lora_30k_v100",
            "qwen25_7b_lora_10k_v100_grpo_500",
            "qwen25_7b_lora_10k_v100_grpo_1000",
        ].index(row.get("model")) if row.get("model") in {
            "Qwen2.5-7B-Instruct",
            "qwen25_7b_lora_10k_v100",
            "qwen25_7b_lora_30k_v100",
            "qwen25_7b_lora_10k_v100_grpo_500",
            "qwen25_7b_lora_10k_v100_grpo_1000",
        } else 99,
    )
    rows = []
    for row in ordered:
        rows.append([
            MODEL_DISPLAY.get(row.get("model"), row.get("model", "")),
            fmt(row.get("eval_loss"), 4),
            fmt(row.get("ppl"), 4),
            fmt(row.get("mean_similarity"), 4),
            pct(row.get("safety_pass_rate"), 1),
            pct(row.get("structure_hit_rate"), 1),
            pct(row.get("high_risk_expression_rate"), 1),
            fmt(row.get("avg_response_length"), 1),
            f"{pct(row.get('ceval_medical_dev_accuracy'), 1)} ({intfmt(row.get('ceval_medical_dev_samples'))})",
        ])
    return rows


def top_categories(filter_summary, limit=8):
    counts = filter_summary.get("category_distribution_top_maxk", {}) if filter_summary else {}
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]


def truncate(text, max_chars=500):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def case_sections(cases, limit=5):
    lines = []
    for idx, case in enumerate(cases[:limit], start=1):
        lines.extend([
            f"### Case {idx}: {case.get('category', 'unknown')}",
            "",
            f"**Question**: {truncate(case.get('question'), 260)}",
            "",
            f"**Reference answer**: {truncate(case.get('reference_answer'), 360)}",
            "",
            f"**Before ({case.get('before_model', 'SFT')})**: {truncate(case.get('before_response'), 500)}",
            "",
            f"**After ({case.get('after_model', 'GRPO')})**: {truncate(case.get('after_response'), 500)}",
            "",
            "**Observation**: This automatic case view is for qualitative inspection. It should be reviewed manually before making strong claims about medical correctness.",
            "",
        ])
    if not lines:
        lines.append("No before/after case file was found. This section should be regenerated after GRPO comparison inference.")
    return "\n".join(lines)


def case_sections_zh(cases, limit=5):
    lines = []
    for idx, case in enumerate(cases[:limit], start=1):
        lines.extend([
            f"### 案例 {idx}：{case.get('category', 'unknown')}",
            "",
            f"**问题**：{truncate(case.get('question'), 260)}",
            "",
            f"**参考答案**：{truncate(case.get('reference_answer'), 360)}",
            "",
            f"**GRPO 前（{case.get('before_model', 'SFT')}）**：{truncate(case.get('before_response'), 500)}",
            "",
            f"**GRPO 后（{case.get('after_model', 'GRPO')}）**：{truncate(case.get('after_response'), 500)}",
            "",
            "**观察**：该案例仅用于定性检查，需要人工复核后才能形成强结论，不能视为医学正确性证明。",
            "",
        ])
    if not lines:
        lines.append("未找到 before/after case 文件。GRPO 对比推理完成后应重新生成本节。")
    return "\n".join(lines)


def best_by(metrics, key, higher=True):
    vals = [(row, as_float(row.get(key), None)) for row in metrics]
    vals = [(row, val) for row, val in vals if val is not None]
    if not vals:
        return None
    return max(vals, key=lambda item: item[1])[0] if higher else min(vals, key=lambda item: item[1])[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a reproducible medical SFT + GRPO experiment report.")
    parser.add_argument("--metrics_dir", default="outputs/medical_project/eval")
    parser.add_argument("--data_summary", default="data_processed/medical_project/embedding/filter_summary.json")
    parser.add_argument("--output", default="docs/medical_project/EXPERIMENT_REPORT.md")
    parser.add_argument("--grpo_summary", default="data_processed/medical_project/grpo/grpo_data_summary.json")
    parser.add_argument("--eval_data_summary", default="data_processed/medical_project/eval/eval_data_summary.json")
    parser.add_argument("--cases", default="outputs/medical_project/eval/grpo_before_after_cases.jsonl")
    parser.add_argument("--output_zh", default="docs/medical_project/EXPERIMENT_REPORT_ZH.md")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    metrics = load_json(metrics_dir / "summary_metrics.json", [])
    if not metrics:
        metrics = load_csv(metrics_dir / "summary_metrics.csv")
    filter_summary = load_json(args.data_summary, {})
    grpo_summary = load_json(args.grpo_summary, {})
    eval_summary = load_json(args.eval_data_summary, {})
    cases = load_jsonl(args.cases, limit=5)

    best_ppl = best_by(metrics, "ppl", higher=False)
    best_similarity = best_by(metrics, "mean_similarity", higher=True)
    best_safety = best_by(metrics, "safety_pass_rate", higher=True)

    data_rows = [
        ["Raw candidate samples", intfmt(filter_summary.get("processed_samples"))],
        ["Scored samples after filtering", intfmt(filter_summary.get("scored_samples"))],
        ["Advertising/contact samples removed", intfmt((filter_summary.get("filtered_samples") or {}).get("advertising_or_contact"))],
        ["High-risk expression samples downweighted", intfmt(filter_summary.get("risk_downweighted_samples"))],
        ["SFT top-k outputs", ", ".join(str(x) for x in filter_summary.get("topk_list", []))],
        ["GRPO samples", intfmt(grpo_summary.get("num_samples"))],
        ["Held-out QA eval samples", intfmt(eval_summary.get("qa_samples"))],
        ["Held-out long-text eval samples", intfmt(eval_summary.get("longtext_samples"))],
    ]

    top_category_rows = [[name, intfmt(count)] for name, count in top_categories(filter_summary)]
    sft_rows = training_rows(SFT_RUNS)
    grpo_rows = training_rows(GRPO_RUNS)
    evaluation_rows = eval_rows(metrics)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report = []
    report.extend([
        "# MedicalGPT Medical SFT + GRPO Experiment Report",
        "",
        f"Generated at: {generated_at}",
        "",
        "## 1. Project Background",
        "",
        "This project extends the open-source MedicalGPT training framework for a medical-domain SFT and small-scale GRPO alignment experiment. The base model is Qwen2.5-7B-Instruct, adapted with LoRA on filtered medical QA data and then further aligned with a medical reward design for open-ended health consultation style prompts.",
        "",
        "The goal is to build a reproducible experimental pipeline for data filtering, SFT training, GRPO reward design, multi-metric evaluation, and report generation. This is a model training experiment, not a medical diagnosis system.",
        "",
        "## 2. MedicalGPT Baseline and Project Goal",
        "",
        "The baseline is the original Qwen2.5-7B-Instruct model evaluated with the same held-out medical QA and long-text data as the adapted models. The current report focuses on the V100 mainline evaluation results that have already completed. A100 SFT and ongoing 30k-GRPO evaluation results can be added in a later report refresh.",
        "",
        "Target comparison groups in this report:",
        "",
        "- Base: Qwen2.5-7B-Instruct",
        "- SFT 10k LoRA V100",
        "- SFT 30k LoRA V100",
        "- GRPO 500 from SFT10k V100",
        "- GRPO 1000 from SFT10k V100",
        "",
        "## 3. Data Source and Leakage Avoidance",
        "",
        "Source data comes from local MedicalGPT medical QA files converted into a normalized QA JSONL format. C-Eval test data was not used for filtering, training, tuning, or report selection. Optional C-Eval dev/valid data is treated only as a development/evaluation source where explicitly stated.",
        "",
        markdown_table(["Item", "Value"], data_rows),
        "",
        "Top category distribution among the selected Top30k SFT pool:",
        "",
        markdown_table(["Category", "Samples"], top_category_rows),
        "",
        "Leakage and wording constraints:",
        "",
        "- C-Eval test data is not used.",
        "- Embedding similarity and QA reference similarity are not medical accuracy.",
        "- Keyword or structure heuristics are not clinical correctness checks.",
        "- The project output is not intended for medical diagnosis or treatment.",
        "",
        "## 4. Embedding Similarity Filtering Method",
        "",
        f"Candidate QA samples were scored against medical topic queries with `{filter_summary.get('embedding_model', 'unknown')}` using the `{filter_summary.get('backend', 'unknown')}` backend. The filtering score was:",
        "",
        "```text",
        filter_summary.get("score_formula", "0.7 * max_similarity + 0.3 * mean_top5_similarity"),
        "```",
        "",
        "Filtering also removed advertising/contact/marketing content and downweighted samples containing high-risk absolute medical expressions. The Top1k manual review file was generated at `docs/medical_project/top1k_manual_review.md`.",
        "",
        "## 5. SFT Training Settings",
        "",
        "SFT uses Qwen2.5-7B-Instruct with LoRA. The main V100 runs use LoRA rank 8, alpha 16, dropout 0.05, and target modules `q_proj,v_proj`. A100 runs are included below as training artifacts but are not yet part of the main downstream evaluation table in this initial report.",
        "",
        markdown_table(
            ["Run", "Adapter", "Train samples", "Train loss", "Eval loss", "PPL", "Runtime(s)", "Samples/s"],
            sft_rows,
        ),
        "",
        "## 6. GRPO Reward Design",
        "",
        "The first-version medical GRPO reward combines structure, reference similarity, safety, and length/repetition signals:",
        "",
        "```text",
        "R = 0.35 * format",
        "  + 0.30 * reference_similarity",
        "  + 0.25 * safety",
        "  - 0.10 * repetition_penalty",
        "```",
        "",
        "The reference similarity term measures semantic closeness to the reference answer. It is not medical accuracy. The safety term rewards cautious medical wording and penalizes absolute diagnosis, concrete prescription dosage, replacing doctor visits, and unsafe medication changes.",
        "",
        "GRPO data construction summary:",
        "",
        markdown_table(
            ["Item", "Value"],
            [
                ["Source SFT data", grpo_summary.get("sft_data", "-")],
                ["GRPO output", grpo_summary.get("output", "-")],
                ["Samples", intfmt(grpo_summary.get("num_samples"))],
                ["Preferred categories", ", ".join(grpo_summary.get("prefer_categories", []))],
            ],
        ),
        "",
        "GRPO training artifacts:",
        "",
        markdown_table(
            ["Run", "Adapter", "Train samples", "Train loss", "Eval loss", "PPL", "Runtime(s)", "Samples/s"],
            grpo_rows,
        ),
        "",
        "## 7. Evaluation Sets and Metrics",
        "",
        "The evaluation script compares Base/SFT/GRPO models with the following metrics:",
        "",
        "- PPL / eval loss on held-out medical long-text QA samples.",
        "- QA reference similarity on generated answers. This is not medical accuracy.",
        "- Structure and safety heuristics on generated answers.",
        "- C-Eval medical dev/valid subset accuracy as a small auxiliary sanity check. C-Eval test is not used.",
        "",
        f"Current held-out QA samples: {intfmt(eval_summary.get('qa_samples'))}; held-out long-text samples: {intfmt(eval_summary.get('longtext_samples'))}. C-Eval samples in the current run are small, so the C-Eval result should be treated as a workflow sanity check rather than a strong benchmark claim.",
        "",
        "## 8. Experiment Results",
        "",
        markdown_table(
            [
                "Model",
                "Eval loss",
                "PPL",
                "Mean ref sim",
                "Safety pass",
                "Structure hit",
                "High-risk expr",
                "Avg len",
                "C-Eval acc(samples)",
            ],
            evaluation_rows,
        ),
        "",
        "Initial observations:",
        "",
        f"- The lowest PPL in the current V100 mainline is `{MODEL_DISPLAY.get(best_ppl.get('model'), best_ppl.get('model')) if best_ppl else '-'}` with PPL {fmt(best_ppl.get('ppl') if best_ppl else None, 4)}.",
        f"- The highest mean QA reference similarity is `{MODEL_DISPLAY.get(best_similarity.get('model'), best_similarity.get('model')) if best_similarity else '-'}` with mean similarity {fmt(best_similarity.get('mean_similarity') if best_similarity else None, 4)}.",
        f"- The highest safety pass rate is `{MODEL_DISPLAY.get(best_safety.get('model'), best_safety.get('model')) if best_safety else '-'}` with safety pass {pct(best_safety.get('safety_pass_rate') if best_safety else None, 1)}.",
        "- SFT and GRPO reduce PPL relative to the base model on the held-out medical text set.",
        "- SFT/GRPO answers are much shorter than base model answers in this run. This improves some safety heuristics but hurts explicit structure-hit metrics.",
        "",
        "## 9. Case Study: SFT vs GRPO",
        "",
        case_sections(cases, limit=5),
        "",
        "## 10. Error Analysis",
        "",
        "The current first-version experiment exposes several important failure modes:",
        "",
        "- Structure hit rate remains low for SFT/GRPO outputs. The models often provide concise answers that include partial advice but do not explicitly include all required sections such as analysis, suggestion, risk warning, and doctor-visit advice.",
        "- GRPO 500/1000 improves or preserves safety pass rate but does not yet reliably improve explicit structure compliance.",
        "- The QA reference similarity differences are small. They should be interpreted as weak reference-overlap signals rather than medical correctness.",
        "- The C-Eval subset in this report has only 15 samples, so its all-1.0 result is not a strong evidence of medical exam performance.",
        "- Some safety rules are heuristic. They can miss unsafe recommendations that do not match the current regex patterns and can also over-penalize cautious long responses.",
        "",
        "## 11. Limitations",
        "",
        "- This project is not a medical diagnosis system.",
        "- No automatic metric here verifies clinical correctness.",
        "- Reference similarity is not medical accuracy.",
        "- Keyword/structure coverage is not medical accuracy.",
        "- C-Eval test data was not used.",
        "- Current report uses completed V100 mainline evaluation. A100 SFT and ongoing 30k-GRPO downstream evaluation results should be added later.",
        "- GRPO reward design is heuristic and should be validated with manual review before making strong claims.",
        "",
        "## 12. Resume Project Description",
        "",
        "基于 MedicalGPT 框架完成 Qwen2.5-7B-Instruct 医疗问答场景 LoRA 微调与小规模 GRPO 对齐实验。通过医学主题 query 与 C-Eval dev/valid 辅助构建目标域语义中心，从开源医疗 QA 数据中筛选 1w-3w 条 SFT 样本，避免测试集泄漏；设计格式、参考答案语义相似度、安全表达和重复惩罚组成的医学 GRPO reward，在 500-1000 条复杂医疗问答上进行结构化回答对齐。最终在医学 QA held-out、C-Eval 医学 dev/valid、PPL、结构化回答命中率和安全规则通过率等指标上完成 Base/SFT/GRPO 多维评测，并明确区分 reference similarity 与医学准确率。",
        "",
        "## Appendix: Pending Updates",
        "",
        "- Add downstream evaluation results for A100 SFT 10k/30k if those models are selected for the final mainline.",
        "- Add the ongoing `qwen25_7b_lora_30k_v100_grpo_1000_a100_1gpu` evaluation after it finishes.",
        "- Refresh case studies after additional GRPO evaluation and manual review.",
        "- Consider the v2 hard-sample GRPO workflow to address short-answer and low-structure issues.",
        "",
    ])

    zh_report = []
    zh_report.extend([
        "# MedicalGPT 医疗 SFT + GRPO 实验报告",
        "",
        f"生成时间：{generated_at}",
        "",
        "## 1. 项目背景",
        "",
        "本项目基于开源 MedicalGPT 训练框架，围绕医疗问答场景构建一条可复现实验流程：先对本地医疗 QA 数据进行清洗和 embedding 相似度筛选，再使用 Qwen2.5-7B-Instruct 进行 LoRA SFT 微调，最后在小规模复杂医疗问答数据上进行 GRPO 对齐实验。",
        "",
        "本项目目标是展示数据筛选、SFT 训练、医学 reward 设计、统一评测与报告生成的完整闭环。它不是医疗诊断系统，也不声称自动指标能够验证真实临床正确性。",
        "",
        "## 2. Baseline 与改造目标",
        "",
        "Baseline 使用原始 Qwen2.5-7B-Instruct，并与 SFT/GRPO 模型在同一批 held-out 医疗 QA 和长文本样本上评测。当前中文报告主线采用已经完成的 V100 评测结果；A100 SFT 与正在补充的 30k-GRPO 评测结果后续可刷新进报告。",
        "",
        "当前报告对比模型：",
        "",
        "- Base：Qwen2.5-7B-Instruct",
        "- SFT 10k LoRA V100",
        "- SFT 30k LoRA V100",
        "- GRPO 500 from SFT10k V100",
        "- GRPO 1000 from SFT10k V100",
        "",
        "## 3. 数据来源与数据泄漏规避",
        "",
        "原始数据来自本地 MedicalGPT 医疗 QA 文件，先转换为统一 QA JSONL 格式。C-Eval test 没有用于数据筛选、训练、调参或报告结论选择。C-Eval dev/valid 仅在明确说明的开发或评测环节使用。",
        "",
        markdown_table(["项目", "数值"], data_rows),
        "",
        "Top30k SFT 数据中的主要类别分布：",
        "",
        markdown_table(["类别", "样本数"], top_category_rows),
        "",
        "数据与指标口径约束：",
        "",
        "- 未使用 C-Eval test。",
        "- embedding similarity 与 QA reference similarity 只能称为参考答案相似度，不能称为医学准确率。",
        "- 关键词覆盖和结构命中是启发式指标，不能代表临床正确性。",
        "- 本项目不是医疗诊断或治疗系统。",
        "",
        "## 4. Embedding 相似度筛选方法",
        "",
        f"候选医疗 QA 样本使用 `{filter_summary.get('embedding_model', 'unknown')}` 与医学主题 query 计算相似度，backend 为 `{filter_summary.get('backend', 'unknown')}`。筛选公式为：",
        "",
        "```text",
        filter_summary.get("score_formula", "0.7 * max_similarity + 0.3 * mean_top5_similarity"),
        "```",
        "",
        "筛选流程还会移除广告、联系方式、医院营销等内容，并对包含高风险绝对化医疗表达的样本进行降权。Top1k 人工抽样检查文件已生成：`docs/medical_project/top1k_manual_review.md`。",
        "",
        "## 5. SFT 训练设置",
        "",
        "SFT 阶段使用 Qwen2.5-7B-Instruct + LoRA。当前 V100 主线配置为 LoRA rank 8、alpha 16、dropout 0.05，target modules 为 `q_proj,v_proj`。A100 训练结果列入训练产物表，但当前下游主评测表仍采用已完成的 V100 结果。",
        "",
        markdown_table(
            ["训练任务", "Adapter", "训练样本", "Train loss", "Eval loss", "PPL", "耗时(s)", "Samples/s"],
            sft_rows,
        ),
        "",
        "## 6. GRPO Reward 设计",
        "",
        "第一版医学 GRPO reward 由格式、参考答案相似度、安全表达、长度/重复惩罚组成：",
        "",
        "```text",
        "R = 0.35 * format",
        "  + 0.30 * reference_similarity",
        "  + 0.25 * safety",
        "  - 0.10 * repetition_penalty",
        "```",
        "",
        "其中 reference similarity 只表示模型回答与参考答案的语义接近程度，不等价于医学准确率。Safety reward 奖励谨慎医疗表达，并惩罚绝对化诊断、具体处方剂量、替代医生面诊、自行停药或加药等风险表达。",
        "",
        "GRPO 数据构造摘要：",
        "",
        markdown_table(
            ["项目", "数值"],
            [
                ["来源 SFT 数据", grpo_summary.get("sft_data", "-")],
                ["GRPO 输出文件", grpo_summary.get("output", "-")],
                ["样本数", intfmt(grpo_summary.get("num_samples"))],
                ["优先类别", ", ".join(grpo_summary.get("prefer_categories", []))],
            ],
        ),
        "",
        "GRPO 训练产物：",
        "",
        markdown_table(
            ["训练任务", "Adapter", "训练样本", "Train loss", "Eval loss", "PPL", "耗时(s)", "Samples/s"],
            grpo_rows,
        ),
        "",
        "## 7. 评测集与评测指标",
        "",
        "统一评测包含以下指标：",
        "",
        "- PPL / eval loss：在 held-out 医疗长文本 QA 样本上计算。",
        "- QA reference similarity：模型生成答案与参考答案的相似度，不是医学准确率。",
        "- 结构化回答命中率与安全规则通过率：基于规则的启发式检查。",
        "- C-Eval 医学 dev/valid 子集 accuracy：当前作为小样本 sanity check，不使用 test。",
        "",
        f"当前 held-out QA 样本数：{intfmt(eval_summary.get('qa_samples'))}；held-out 长文本样本数：{intfmt(eval_summary.get('longtext_samples'))}。当前 C-Eval 样本数较少，因此只能作为流程验证和辅助观察，不能作为强 benchmark 结论。",
        "",
        "## 8. 实验结果",
        "",
        markdown_table(
            [
                "模型",
                "Eval loss",
                "PPL",
                "Mean ref sim",
                "Safety pass",
                "Structure hit",
                "High-risk expr",
                "Avg len",
                "C-Eval acc(samples)",
            ],
            evaluation_rows,
        ),
        "",
        "初步观察：",
        "",
        f"- 当前 V100 主线中 PPL 最低的是 `{MODEL_DISPLAY.get(best_ppl.get('model'), best_ppl.get('model')) if best_ppl else '-'}`，PPL 为 {fmt(best_ppl.get('ppl') if best_ppl else None, 4)}。",
        f"- 当前 mean QA reference similarity 最高的是 `{MODEL_DISPLAY.get(best_similarity.get('model'), best_similarity.get('model')) if best_similarity else '-'}`，均值为 {fmt(best_similarity.get('mean_similarity') if best_similarity else None, 4)}。",
        f"- 当前 safety pass rate 最高的是 `{MODEL_DISPLAY.get(best_safety.get('model'), best_safety.get('model')) if best_safety else '-'}`，通过率为 {pct(best_safety.get('safety_pass_rate') if best_safety else None, 1)}。",
        "- SFT 和 GRPO 相比 Base 明显降低了 held-out 医疗文本 PPL。",
        "- SFT/GRPO 的回答显著短于 Base，安全规则通过率更高，但显式结构命中率明显偏低。",
        "",
        "## 9. Case Study：SFT vs GRPO",
        "",
        case_sections_zh(cases, limit=5),
        "",
        "## 10. 错误分析",
        "",
        "第一版实验暴露出以下问题：",
        "",
        "- SFT/GRPO 输出的结构命中率偏低。模型常给出简短建议，但没有显式覆盖“分析、建议、风险提示、就医建议”等完整结构。",
        "- GRPO 500/1000 能保持或提升安全通过率，但尚未显著改善显式结构化表达。",
        "- QA reference similarity 的差异较小，只能作为参考答案重合度信号，不能说明医学正确性。",
        "- 当前 C-Eval 仅 15 个样本，所有模型 100% 不能说明真实考试能力，只能说明流程跑通。",
        "- 安全规则是启发式的，可能漏检未匹配正则的风险建议，也可能对谨慎长回答产生误判。",
        "",
        "## 11. 局限性",
        "",
        "- 本项目不是医疗诊断系统。",
        "- 当前没有任何自动指标能验证真实临床正确性。",
        "- Reference similarity 不是医学准确率。",
        "- 关键词覆盖和结构命中不是医学准确率。",
        "- 未使用 C-Eval test。",
        "- 当前报告采用已完成的 V100 主线评测；A100 SFT 与后续 30k-GRPO 下游评测结果需要后续补充。",
        "- GRPO reward 是启发式设计，强结论需要人工复核或专家评价支持。",
        "",
        "## 12. 简历项目描述",
        "",
        "基于 MedicalGPT 框架完成 Qwen2.5-7B-Instruct 医疗问答场景 LoRA 微调与小规模 GRPO 对齐实验。通过医学主题 query 与 C-Eval dev/valid 辅助构建目标域语义中心，从开源医疗 QA 数据中筛选 1w-3w 条 SFT 样本，避免测试集泄漏；设计格式、参考答案语义相似度、安全表达和重复惩罚组成的医学 GRPO reward，在 500-1000 条复杂医疗问答上进行结构化回答对齐。最终在医学 QA held-out、C-Eval 医学 dev/valid、PPL、结构化回答命中率和安全规则通过率等指标上完成 Base/SFT/GRPO 多维评测，并明确区分 reference similarity 与医学准确率。",
        "",
        "## 附录：待补充更新",
        "",
        "- 如果 A100 SFT 10k/30k 被选为最终主线，需要补充其下游评测结果。",
        "- 当前正在补跑的 `qwen25_7b_lora_30k_v100_grpo_1000_a100_1gpu` 评测完成后，需要刷新报告。",
        "- 新 GRPO 结果出来后，应重新生成并人工复核 case study。",
        "- 后续可使用 v2 hard-sample GRPO 流程改善短回答和结构命中率偏低的问题。",
        "",
    ])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    output_zh = Path(args.output_zh)
    output_zh.parent.mkdir(parents=True, exist_ok=True)
    output_zh.write_text("\n".join(zh_report), encoding="utf-8")
    print(f"Wrote report to {output}")
    print(f"Wrote Chinese report to {output_zh}")


if __name__ == "__main__":
    main()
