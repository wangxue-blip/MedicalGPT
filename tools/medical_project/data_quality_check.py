#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run quality checks for medical project data files."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def normalize_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).strip()


def percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct / 100))
    return ordered[idx]


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                yield line_no, None, "empty_line"
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, f"json_error:{exc}"


def extract_sharegpt_qa(record):
    conversations = record.get("conversations") if isinstance(record, dict) else None
    if not isinstance(conversations, list) or len(conversations) < 2:
        return "", "", "missing_conversations"
    first, second = conversations[0], conversations[1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        return "", "", "invalid_turn_type"
    if first.get("from") != "human" or second.get("from") != "gpt":
        return "", "", "invalid_roles"
    question = normalize_text(first.get("value"))
    answer = normalize_text(second.get("value"))
    if not question or not answer:
        return question, answer, "empty_question_or_answer"
    return question, answer, None


def extract_candidate_qa(record):
    if not isinstance(record, dict):
        return "", "", "invalid_record"
    question = normalize_text(record.get("question"))
    answer = normalize_text(record.get("answer"))
    if not question or not answer:
        return question, answer, "empty_question_or_answer"
    return question, answer, None


def check_file(args):
    total = 0
    valid = 0
    invalid = Counter()
    question_lens = []
    answer_lens = []
    seen_questions = set()
    duplicate_questions = 0
    role_counts = Counter()

    extractor = extract_sharegpt_qa if args.format == "sharegpt" else extract_candidate_qa

    for _line_no, record, error in read_jsonl(args.input):
        if error:
            invalid[error.split(":", 1)[0]] += 1
            continue
        total += 1
        if args.format == "sharegpt" and isinstance(record, dict):
            for turn in record.get("conversations", []):
                if isinstance(turn, dict):
                    role_counts[turn.get("from", "missing")] += 1
        question, answer, qa_error = extractor(record)
        if qa_error:
            invalid[qa_error] += 1
            continue
        valid += 1
        question_lens.append(len(question))
        answer_lens.append(len(answer))
        key = question.casefold()
        if key in seen_questions:
            duplicate_questions += 1
        else:
            seen_questions.add(key)

    summary = {
        "input": args.input,
        "format": args.format,
        "total_json_records": total,
        "valid_records": valid,
        "invalid_counts": dict(invalid),
        "duplicate_questions": duplicate_questions,
        "duplicate_question_rate": round(duplicate_questions / valid, 6) if valid else 0.0,
        "question_length": {
            "avg": round(sum(question_lens) / len(question_lens), 2) if question_lens else 0.0,
            "p50": percentile(question_lens, 50),
            "p95": percentile(question_lens, 95),
            "max": max(question_lens) if question_lens else 0,
        },
        "answer_length": {
            "avg": round(sum(answer_lens) / len(answer_lens), 2) if answer_lens else 0.0,
            "p50": percentile(answer_lens, 50),
            "p95": percentile(answer_lens, 95),
            "max": max(answer_lens) if answer_lens else 0,
        },
        "role_counts": dict(role_counts),
    }

    output = Path(args.output) if args.output else Path(args.input).with_suffix(".quality.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Check JSONL data quality.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument(
        "--format",
        choices=["candidate", "sharegpt", "grpo", "eval"],
        default="sharegpt",
        help="Expected data format.",
    )
    parser.add_argument("--output", default=None, help="Output quality summary JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    check_file(args)


if __name__ == "__main__":
    main()
