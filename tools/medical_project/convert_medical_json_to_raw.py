#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert shibing624/medical finetune files to the project raw JSONL entry."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def infer_split(path):
    name = Path(path).name.lower()
    if "train" in name:
        return "train"
    if "valid" in name or "val" in name or "dev" in name:
        return "valid"
    return "unknown"


def first_non_ws_char(path):
    with Path(path).open("r", encoding="utf-8") as fin:
        while True:
            char = fin.read(1)
            if not char:
                return ""
            if not char.isspace():
                return char


def iter_json_array(path):
    with Path(path).open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if isinstance(data, list):
        for item in data:
            yield item, None
    elif isinstance(data, dict):
        yield data, None
    else:
        yield None, f"unsupported_json_root:{type(data).__name__}"


def iter_json_lines(path):
    with Path(path).open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), None
            except json.JSONDecodeError as exc:
                yield None, f"line_{line_no}:json_error:{exc}"


def iter_records(path):
    marker = first_non_ws_char(path)
    if marker == "[":
        yield from iter_json_array(path)
    elif marker == "{":
        yield from iter_json_lines(path)
    else:
        yield None, f"unsupported_file_start:{marker!r}"


def normalize_record(record, source_file, source_split, add_metadata):
    if not isinstance(record, dict):
        return None
    output = dict(record)
    if add_metadata:
        output.setdefault("source", "shibing624_medical")
        output.setdefault("source_file", source_file)
        output.setdefault("source_split", source_split)
    return output


def convert_files(args):
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "output": str(output_path),
        "inputs": [],
        "total_written": 0,
        "total_invalid": 0,
        "schema_counts": {},
        "conversion_logic": [
            "Read each source file as JSONL when it starts with '{', or as a JSON array/object when it starts with '['.",
            "Preserve original fields such as instruction/input/output without filtering or deduplication.",
            "Append source/source_file/source_split metadata for traceability.",
            "Write one compact JSON object per line to data_raw/medical_project/medical_raw.jsonl.",
            "Leave length filtering, empty sample removal, and duplicate question removal to prepare_raw_medical.py.",
            "Do not use any test split.",
        ],
    }
    schema_counts = Counter()

    with output_path.open("w", encoding="utf-8") as fout:
        for input_file in args.inputs:
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            if "test" in input_path.name.lower():
                raise ValueError(f"Refusing to convert possible test split: {input_path}")

            source_split = infer_split(input_path)
            file_stats = {
                "path": str(input_path),
                "source_split": source_split,
                "written": 0,
                "invalid": 0,
                "invalid_examples": [],
            }

            for record, error in iter_records(input_path):
                if error:
                    file_stats["invalid"] += 1
                    if len(file_stats["invalid_examples"]) < 5:
                        file_stats["invalid_examples"].append(error)
                    continue
                normalized = normalize_record(record, input_path.name, source_split, args.add_metadata)
                if normalized is None:
                    file_stats["invalid"] += 1
                    if len(file_stats["invalid_examples"]) < 5:
                        file_stats["invalid_examples"].append(f"non_object:{type(record).__name__}")
                    continue
                schema_counts.update([",".join(sorted(normalized.keys()))])
                fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                file_stats["written"] += 1
                stats["total_written"] += 1
                if args.progress_every > 0 and stats["total_written"] % args.progress_every == 0:
                    print(f"written {stats['total_written']} records...", file=sys.stderr)

            stats["total_invalid"] += file_stats["invalid"]
            stats["inputs"].append(file_stats)

    stats["schema_counts"] = dict(schema_counts.most_common(20))
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert shibing624/medical train/valid finetune files to medical_raw.jsonl."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data_raw/medical_project/medical/train_zh_0.json",
            "data_raw/medical_project/medical/valid_zh_0.json",
        ],
        help="Input JSONL/JSON files. Test splits are rejected.",
    )
    parser.add_argument(
        "--output",
        default="data_raw/medical_project/medical_raw.jsonl",
        help="Output raw JSONL path used by prepare_raw_medical.py.",
    )
    parser.add_argument(
        "--stats_output",
        default="outputs/medical_project/logs/medical_raw_conversion_stats.json",
        help="Output conversion stats JSON path.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    parser.add_argument(
        "--add_metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add source/source_file/source_split fields.",
    )
    parser.add_argument("--progress_every", type=int, default=100000)
    return parser.parse_args()


def main():
    args = parse_args()
    stats = convert_files(args)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
