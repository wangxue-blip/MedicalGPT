#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare aligned Qwen2.5-VL VQA prediction files without changing their metrics."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# This deliberately small alias table is an analysis aid, not an official VQA
# metric.  It exposes how much strict exact-match movement is answer-style
# alignment (e.g. "是" versus "包含") rather than a new visual prediction.
ALIASES = {
    "是": "positive",
    "是的": "positive",
    "有": "positive",
    "包含": "positive",
    "存在": "positive",
    "否": "negative",
    "不是": "negative",
    "没有": "negative",
    "不包含": "negative",
    "无": "negative",
    "mri": "mri",
    "核磁共振": "mri",
    "核磁共振成像": "mri",
    "磁共振": "mri",
    "磁共振成像": "mri",
    "ct": "ct",
    "ct扫描": "ct",
    "计算机断层扫描": "ct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_predictions", required=True)
    parser.add_argument("--lora_predictions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[\s\u3000]+", "", text)
    return re.sub(r"[，。！？；：、,.!?;:'\"()（）\[\]【】]", "", text)


def limited_alias(text: str) -> str:
    value = normalise(text)
    return ALIASES.get(value, value)


def rate(correct: int, samples: int) -> dict[str, float | int]:
    return {"correct": correct, "samples": samples, "exact_match": correct / samples}


def main() -> None:
    args = parse_args()
    base = read_jsonl(Path(args.base_predictions))
    lora = read_jsonl(Path(args.lora_predictions))
    if len(base) != len(lora) or [row["id"] for row in base] != [row["id"] for row in lora]:
        raise ValueError("Prediction files must contain the same rows in the same order.")

    pairs = list(zip(base, lora))
    samples = len(pairs)
    cross = Counter((bool(b["exact_match"]), bool(l["exact_match"])) for b, l in pairs)
    base_limited = sum(limited_alias(row["prediction"]) == limited_alias(row["reference"]) for row in base)
    lora_limited = sum(limited_alias(row["prediction"]) == limited_alias(row["reference"]) for row in lora)
    style_only = sum(
        not b["exact_match"]
        and l["exact_match"]
        and limited_alias(b["prediction"]) == limited_alias(b["reference"])
        for b, l in pairs
    )

    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"samples": 0, "base_correct": 0, "lora_correct": 0})
    for b, l in pairs:
        group = str(b.get("metadata", {}).get("content_type", "unknown"))
        by_type[group]["samples"] += 1
        by_type[group]["base_correct"] += int(bool(b["exact_match"]))
        by_type[group]["lora_correct"] += int(bool(l["exact_match"]))

    strict_base = sum(bool(row["exact_match"]) for row in base)
    strict_lora = sum(bool(row["exact_match"]) for row in lora)
    result = {
        "samples": samples,
        "strict_exact_match": {"base": rate(strict_base, samples), "lora": rate(strict_lora, samples)},
        "limited_alias_exact_match": {
            "aliases": ALIASES,
            "base": rate(base_limited, samples),
            "lora": rate(lora_limited, samples),
        },
        "strict_cross_outcomes": {
            "both_correct": cross[(True, True)],
            "both_wrong": cross[(False, False)],
            "lora_only_correct": cross[(False, True)],
            "base_only_correct": cross[(True, False)],
            "lora_only_strict_wins_already_equivalent_under_limited_alias": style_only,
        },
        "by_content_type_strict": {
            group: {
                **values,
                "base_exact_match": values["base_correct"] / values["samples"],
                "lora_exact_match": values["lora_correct"] / values["samples"],
            }
            for group, values in sorted(by_type.items())
        },
        "caveat": "The limited alias score is a transparent diagnostic, not an official SLAKE metric or clinical performance measure.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
