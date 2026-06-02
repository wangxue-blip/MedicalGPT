#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate C-Eval medical dev/valid subsets only. Test data is never used."""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on C-Eval medical dev/valid subsets. Test data is not used.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--peft_path", default=None)
    parser.add_argument("--ceval_dev_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def read_ceval_rows(ceval_dev_dir, max_samples):
    rows = []
    for path in sorted(Path(ceval_dev_dir).glob("*.csv")):
        subject = path.stem.replace("_dev", "").replace("_val", "").replace("_valid", "")
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["subject"] = subject
                rows.append(row)
                if max_samples > 0 and len(rows) >= max_samples:
                    return rows
    return rows


def build_prompt(row):
    return (
        "以下是医学相关单项选择题。请只回答 A、B、C 或 D，不要解释。\n\n"
        f"题目：{row['question']}\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n"
        "答案："
    )


def parse_choice(text):
    match = re.search(r"\b([ABCD])\b", text.upper())
    if match:
        return match.group(1)
    for ch in text.upper():
        if ch in "ABCD":
            return ch
    return ""


def main():
    args = parse_args()
    import torch
    from eval.medical_project.model_utils import (
        load_tokenizer_and_model,
        model_id_from_paths,
        write_json,
    )
    rows = read_ceval_rows(args.ceval_dev_dir, args.max_samples)
    if not rows:
        raise ValueError(f"No C-Eval dev/valid CSV files found in {args.ceval_dev_dir}")

    with torch.inference_mode():
        tokenizer, model = load_tokenizer_and_model(
            args.model_name_or_path,
            peft_path=args.peft_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )

        predictions = []
        for row in rows:
            prompt = build_prompt(row)
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[0][encoded["input_ids"].shape[-1]:]
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred = parse_choice(output_text)
            gold = str(row["answer"]).strip().upper()
            predictions.append({
                "subject": row["subject"],
                "id": row.get("id", ""),
                "prediction": pred,
                "answer": gold,
                "correct": pred == gold,
                "raw_output": output_text,
            })

    per_subject = {}
    for subject in sorted({row["subject"] for row in predictions}):
        subject_rows = [row for row in predictions if row["subject"] == subject]
        per_subject[subject] = {
            "accuracy": statistics.mean(row["correct"] for row in subject_rows),
            "num_samples": len(subject_rows),
        }
    result = {
        "model": model_id_from_paths(args.model_name_or_path, args.peft_path),
        "model_name_or_path": args.model_name_or_path,
        "peft_path": args.peft_path,
        "ceval_dev_dir": args.ceval_dev_dir,
        "accuracy": statistics.mean(row["correct"] for row in predictions),
        "num_samples": len(predictions),
        "per_subject": per_subject,
        "ceval_test_used": False,
        "note": "Only C-Eval dev/valid CSV files are evaluated; test files are not used.",
    }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
