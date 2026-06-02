#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize stage 11 metrics into CSV and JSON."""

import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize medical project evaluation metrics.")
    parser.add_argument("--metrics_dir", default="outputs/medical_project/eval")
    parser.add_argument("--output_csv", default="outputs/medical_project/eval/summary_metrics.csv")
    parser.add_argument("--output_json", default="outputs/medical_project/eval/summary_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    by_model = {}
    for path in metrics_dir.glob("*.json"):
        if path.name in {"summary_metrics.json"}:
            continue
        data = load_json(path)
        model = data.get("model")
        if not model:
            continue
        row = by_model.setdefault(model, {"model": model})
        if path.name.startswith("ppl_"):
            row["eval_loss"] = data.get("eval_loss")
            row["ppl"] = data.get("ppl")
            row["ppl_samples"] = data.get("num_samples")
        elif path.name.startswith("qa_similarity_"):
            row["mean_similarity"] = data.get("mean_similarity")
            row["median_similarity"] = data.get("median_similarity")
            row["qa_samples"] = data.get("num_samples")
            row["prediction_file"] = data.get("prediction_file")
        elif path.name.startswith("structure_safety_"):
            keys = [
                "structure_hit_rate",
                "has_diagnosis_basis_rate",
                "has_suggestion_rate",
                "has_risk_warning_rate",
                "has_doctor_visit_advice_rate",
                "safety_pass_rate",
                "high_risk_expression_rate",
                "repetition_rate",
                "avg_response_length",
            ]
            for key in keys:
                row[key] = data.get(key)
        elif path.name.startswith("ceval_medical_"):
            row["ceval_medical_dev_accuracy"] = data.get("accuracy")
            row["ceval_medical_dev_samples"] = data.get("num_samples")
            row["ceval_test_used"] = data.get("ceval_test_used", False)

    rows = list(by_model.values())
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if "model" in fieldnames:
        fieldnames.remove("model")
        fieldnames = ["model"] + fieldnames

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rows)} model summaries to {args.output_csv} and {args.output_json}")


if __name__ == "__main__":
    main()
