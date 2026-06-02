#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate model perplexity or eval loss for medical held-out text."""

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPL/eval loss on medical held-out data.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--peft_path", default=None)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--model_max_length", type=int, default=1024)
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--batch_size", type=int, default=1, help="Reserved for compatibility; current eval is per-sample.")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from eval.medical_project.model_utils import (
        item_to_text,
        load_tokenizer_and_model,
        model_id_from_paths,
        read_jsonl,
        write_json,
    )
    with torch.inference_mode():
        tokenizer, model = load_tokenizer_and_model(
            args.model_name_or_path,
            peft_path=args.peft_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        rows = read_jsonl(args.eval_file, max_samples=args.max_samples)
        total_loss = 0.0
        total_tokens = 0
        used = 0

        for item in rows:
            text = item_to_text(item, tokenizer=tokenizer)
            if not text.strip():
                continue
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.model_max_length,
            )
            if encoded["input_ids"].shape[-1] < 2:
                continue
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            labels = encoded["input_ids"].clone()
            outputs = model(**encoded, labels=labels)
            token_count = int(encoded["attention_mask"].sum().item()) - 1
            total_loss += float(outputs.loss.item()) * max(1, token_count)
            total_tokens += max(1, token_count)
            used += 1

    if total_tokens == 0:
        raise ValueError(f"No valid samples found in {args.eval_file}")
    eval_loss = total_loss / total_tokens
    result = {
        "model": model_id_from_paths(args.model_name_or_path, args.peft_path),
        "model_name_or_path": args.model_name_or_path,
        "peft_path": args.peft_path,
        "eval_file": args.eval_file,
        "eval_loss": eval_loss,
        "ppl": math.exp(min(eval_loss, 20.0)),
        "num_samples": used,
        "num_tokens": total_tokens,
    }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
