#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert filtered medical QA data to ShareGPT format."""

import argparse
import json
import re
from pathlib import Path


DEFAULT_SAFETY_PREFIX = (
    "请以科普和就医建议为主，避免替代医生诊断；如出现急重症或症状加重，应及时就医。\n\n"
)


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


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def build_human_text(question, use_safety_prefix):
    question = normalize_text(question)
    if not use_safety_prefix:
        return question
    return DEFAULT_SAFETY_PREFIX + question


def convert_record(record, use_safety_prefix):
    question = normalize_text(record.get("question"))
    answer = normalize_text(record.get("answer"))
    if not question or not answer:
        return None
    return {
        "conversations": [
            {"from": "human", "value": build_human_text(question, use_safety_prefix)},
            {"from": "gpt", "value": answer},
        ],
        "metadata": {
            "id": record.get("id"),
            "score": record.get("score"),
            "matched_category": record.get("matched_category"),
            "matched_query": record.get("matched_query"),
            "source_format": "medical_project_filtered_sft",
        },
    }


def write_jsonl(path, records):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_file(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_name
    stats_file = output_dir / "conversion_stats.json"

    converted = []
    skipped = 0
    for _line_no, record in read_jsonl(args.input):
        item = convert_record(record, args.system_safety_prefix)
        if item is None:
            skipped += 1
            continue
        if not args.keep_metadata:
            item.pop("metadata", None)
        converted.append(item)

    written = write_jsonl(output_file, converted)
    stats = {
        "input": args.input,
        "output": str(output_file),
        "written": written,
        "skipped_missing_qa": skipped,
        "system_safety_prefix": args.system_safety_prefix,
        "keep_metadata": args.keep_metadata,
        "format": "sharegpt",
    }
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert medical SFT JSONL to MedicalGPT ShareGPT JSONL format."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="medical_sft.jsonl")
    parser.add_argument("--system_safety_prefix", type=str2bool, default=True)
    parser.add_argument(
        "--keep_metadata",
        type=str2bool,
        default=True,
        help="Keep non-training metadata fields for traceability. MedicalGPT training ignores extra fields.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_file(args)


if __name__ == "__main__":
    main()
