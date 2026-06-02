#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build held-out medical QA/eval-loss files for stage 11 evaluation."""

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def collect_excluded_ids(paths):
    excluded = set()
    for path in paths:
        if not path or not Path(path).exists():
            continue
        for item in read_jsonl(path):
            if item.get("id"):
                excluded.add(item["id"])
            if item.get("source_id"):
                excluded.add(item["source_id"])
    return excluded


def write_jsonl(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build held-out medical evaluation JSONL files.")
    parser.add_argument("--candidates", default="data_processed/medical_project/sft/medical_candidates.jsonl")
    parser.add_argument("--exclude_files", default="data_processed/medical_project/sft/medical_sft_top30k.jsonl")
    parser.add_argument("--output_dir", default="data_processed/medical_project/eval")
    parser.add_argument("--qa_samples", type=int, default=500)
    parser.add_argument("--longtext_samples", type=int, default=1000)
    parser.add_argument("--min_answer_len", type=int, default=80)
    parser.add_argument("--max_answer_len", type=int, default=1800)
    return parser.parse_args()


def main():
    args = parse_args()
    exclude_files = [p.strip() for p in args.exclude_files.split(",") if p.strip()]
    excluded_ids = collect_excluded_ids(exclude_files)

    qa_rows = []
    longtext_rows = []
    seen_questions = set()
    for item in read_jsonl(args.candidates):
        sample_id = item.get("id")
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer or sample_id in excluded_ids:
            continue
        if question in seen_questions:
            continue
        if not (args.min_answer_len <= len(answer) <= args.max_answer_len):
            continue
        seen_questions.add(question)
        row = {
            "id": sample_id,
            "question": question,
            "answer": answer,
            "source": item.get("source", "medical_candidates_heldout"),
        }
        if len(qa_rows) < args.qa_samples:
            qa_rows.append(row)
        if len(longtext_rows) < args.longtext_samples:
            longtext_rows.append(row)
        if len(qa_rows) >= args.qa_samples and len(longtext_rows) >= args.longtext_samples:
            break

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "medical_qa_eval.jsonl", qa_rows)
    write_jsonl(output_dir / "medical_longtext_eval.jsonl", longtext_rows)
    summary = {
        "candidates": args.candidates,
        "exclude_files": exclude_files,
        "excluded_ids": len(excluded_ids),
        "qa_samples": len(qa_rows),
        "longtext_samples": len(longtext_rows),
        "note": "Held-out files are built without C-Eval test data.",
    }
    with open(output_dir / "eval_data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
