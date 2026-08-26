#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build reproducible Chinese multimodal SFT files from the official SLAKE split.

The generated JSONL schema is intentionally small and model agnostic::

    {"id", "image" (or null), "question", "answer", "source", "metadata"}

``image`` is relative to the project root.  Image-backed SLAKE examples are
kept in their official train/validation/test split.  A bounded sample of the
existing Chinese MedicalGPT text SFT data can be mixed into *train only* to
preserve the project's answer style without contaminating image evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slake_dir", default="data_raw/medical_project/slake")
    parser.add_argument(
        "--text_sft_file",
        default="data_processed/medical_project/sft/sharegpt_1k/medical_sft.jsonl",
        help="Existing MedicalGPT ShareGPT JSONL to mix into train only.",
    )
    parser.add_argument(
        "--output_dir",
        default="data_processed/medical_project/vl/slake_zh_textmix",
    )
    parser.add_argument(
        "--max_vl_train_samples",
        type=int,
        default=2048,
        help="-1 keeps all official Chinese SLAKE train rows.",
    )
    parser.add_argument("--max_text_train_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(payload).__name__}")
    return payload


def limit_rows(rows: list[dict[str, Any]], maximum: int, rng: random.Random) -> list[dict[str, Any]]:
    if maximum < 0 or len(rows) <= maximum:
        return list(rows)
    selected = rng.sample(rows, maximum)
    selected.sort(key=lambda row: str(row["id"]))
    return selected


def slake_records(rows: list[dict[str, Any]], split: str, slake_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("q_lang") != "zh":
            continue
        image_path = slake_dir / "imgs" / row["img_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"SLAKE image missing for qid={row.get('qid')}: {image_path}")
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        if not question or not answer:
            continue
        records.append(
            {
                "id": f"slake_zh_{split}_{row['qid']}",
                "image": str(image_path.relative_to(PROJECT_ROOT)),
                "question": question,
                "answer": answer,
                "source": "SLAKE_zh_cc_by_4.0",
                "metadata": {
                    "split": split,
                    "qid": row["qid"],
                    "img_id": row["img_id"],
                    "img_name": row["img_name"],
                    "location": row.get("location"),
                    "modality": row.get("modality"),
                    "answer_type": row.get("answer_type"),
                    "content_type": row.get("content_type"),
                },
            }
        )
    return records


def first_message(conversations: list[dict[str, Any]], role: str) -> str | None:
    for message in conversations:
        if message.get("from") == role:
            value = str(message.get("value", "")).strip()
            if value:
                return value
    return None


def text_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            conversations = row.get("conversations", [])
            question = first_message(conversations, "human")
            answer = first_message(conversations, "gpt")
            if not question or not answer:
                continue
            metadata = row.get("metadata", {})
            records.append(
                {
                    "id": f"medicalgpt_text_{metadata.get('id', line_number)}",
                    "image": None,
                    "question": question,
                    "answer": answer,
                    "source": "MedicalGPT_text_sft",
                    "metadata": {"original_id": metadata.get("id"), "source_format": metadata.get("source_format")},
                }
            )
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["source"] for row in rows).items()))


def main() -> None:
    args = parse_args()
    slake_dir = (PROJECT_ROOT / args.slake_dir).resolve()
    source_dir = slake_dir / "source"
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    split_rows = {
        "train": slake_records(load_json(source_dir / "train.json"), "train", slake_dir),
        "validation": slake_records(load_json(source_dir / "validation.json"), "validation", slake_dir),
        "test": slake_records(load_json(source_dir / "test.json"), "test", slake_dir),
    }
    train_vl = limit_rows(split_rows["train"], args.max_vl_train_samples, rng)
    text_path = (PROJECT_ROOT / args.text_sft_file).resolve()
    train_text = limit_rows(text_records(text_path), args.max_text_train_samples, rng)
    train_rows = train_vl + train_text
    rng.shuffle(train_rows)

    outputs = {
        "train": train_rows,
        "validation": split_rows["validation"],
        "test": split_rows["test"],
    }
    for split, rows in outputs.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    stats = {
        "dataset": "SLAKE Chinese VQA + MedicalGPT text-only train mix",
        "license": {"SLAKE": "CC-BY-4.0", "MedicalGPT_text": "existing project dataset"},
        "seed": args.seed,
        "max_vl_train_samples": args.max_vl_train_samples,
        "max_text_train_samples": args.max_text_train_samples,
        "official_slake_zh_counts": {split: len(rows) for split, rows in split_rows.items()},
        "generated_counts": {split: len(rows) for split, rows in outputs.items()},
        "generated_source_counts": {split: source_counts(rows) for split, rows in outputs.items()},
    }
    (output_dir / "data_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
