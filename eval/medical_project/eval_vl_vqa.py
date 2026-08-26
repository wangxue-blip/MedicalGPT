#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a Qwen2.5-VL base model or LoRA adapter on local Chinese VQA JSONL."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--peft_path", default=None)
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--torch_dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    return parser.parse_args()


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[\s\u3000]+", "", text)
    return re.sub(r"[，。！？；：、,.!?;:'\"()（）\[\]【】]", "", text)


def load_rows(path: Path, maximum: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("image"):
                continue
            image = PROJECT_ROOT / row["image"]
            if not image.is_file():
                raise FileNotFoundError(image)
            rows.append(row)
            if maximum >= 0 and len(rows) >= maximum:
                break
    if not rows:
        raise ValueError(f"No image-backed VQA rows in {path}")
    return rows


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
    )
    if args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path)
    model.to(device)
    model.eval()
    rows = load_rows(Path(args.eval_file), args.max_samples)
    prediction_path = Path(args.output_dir) / f"{args.label}_predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[bool]] = defaultdict(list)
    exact_matches: list[bool] = []

    with torch.inference_mode(), prediction_path.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc=f"VQA evaluation: {args.label}"):
            image_path = str((PROJECT_ROOT / row["image"]).resolve())
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": f"请根据这张医学影像简洁回答：\n{row['question']}\n只输出答案，不要解释。"},
                    ],
                }
            ]
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            with Image.open(image_path) as image:
                inputs = processor(text=[text], images=[image.convert("RGB")], padding=True, return_tensors="pt").to(device)
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            output_ids = generated[:, inputs.input_ids.shape[1] :]
            prediction = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            exact = normalise(prediction) == normalise(str(row["answer"]))
            exact_matches.append(exact)
            metadata = row.get("metadata", {})
            groups[str(metadata.get("content_type", "unknown"))].append(exact)
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "reference": row["answer"],
                        "prediction": prediction,
                        "exact_match": exact,
                        "metadata": metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {
        "label": args.label,
        "base_model": args.base_model,
        "peft_path": args.peft_path,
        "eval_file": args.eval_file,
        "samples": len(rows),
        "exact_match": sum(exact_matches) / len(exact_matches),
        "exact_matches": sum(exact_matches),
        "by_content_type": {
            key: {"samples": len(values), "exact_match": sum(values) / len(values)} for key, values in sorted(groups.items())
        },
    }
    metrics_path = Path(args.output_dir) / f"{args.label}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
