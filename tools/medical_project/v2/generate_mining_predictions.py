#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate Base/SFT mining predictions for the v2 hard prompt pool.

This script loads a base model plus an optional PEFT adapter and writes one
prediction per prompt. It keeps v2 prompt metadata so later hard-mining stages
can score by source type, category, required sections, and safety rules.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path


MEDICAL_SYSTEM_PROMPT = (
    "你是一个医学科普与健康咨询助手。请基于用户描述给出结构化、谨慎的中文回答，"
    "包含可能原因或分析、处理建议、风险提示和就医建议。不要替代医生诊断，"
    "不要给出具体处方剂量，涉及急症、孕妇、儿童、老人或症状加重时应提示及时就医。"
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


def read_jsonl(path, max_samples=-1):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def read_done_ids(path):
    output = Path(path)
    if not output.exists():
        return set()
    done = set()
    with output.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id"):
                done.add(row["id"])
    return done


def write_json(path, data):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def torch_dtype_from_arg(dtype):
    if dtype in (None, "auto"):
        return "auto"
    import torch

    return getattr(torch, dtype)


def load_tokenizer_and_model(model_name_or_path, peft_path=None, torch_dtype="auto", device_map="auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype_from_arg(torch_dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    if peft_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required when --peft_path is set.") from exc
        model = PeftModel.from_pretrained(model, peft_path)
    model.eval()
    return tokenizer, model


def model_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def build_prompt(tokenizer, question, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\n用户：{question}\n助手："


def generate_answer(tokenizer, model, question, args):
    import torch

    prompt = build_prompt(tokenizer, question, args.system_prompt)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.model_max_length)
    input_device = model_input_device(model)
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    do_sample = args.temperature > 0
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def model_id_from_args(args):
    raw = args.peft_path or args.model_name_or_path
    return args.model_label or os.path.basename(os.path.normpath(raw)).replace("/", "_")


def build_output_row(item, prediction, args, error=None):
    return {
        "id": item.get("id"),
        "question": normalize_text(item.get("question")),
        "reference_answer": normalize_text(item.get("answer")),
        "prediction": prediction,
        "category": item.get("category", ""),
        "source_type": item.get("source_type", ""),
        "source_id": item.get("source_id", ""),
        "required_sections": item.get("required_sections", []),
        "safety_rules": item.get("safety_rules", []),
        "selection_tags": item.get("selection_tags", []),
        "metadata": item.get("metadata", {}),
        "model": model_id_from_args(args),
        "model_name_or_path": args.model_name_or_path,
        "peft_path": args.peft_path,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "model_max_length": args.model_max_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
        },
        "error": error,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate v2 mining predictions for Base/SFT models.")
    parser.add_argument("--model_name_or_path", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--peft_path", default=None)
    parser.add_argument("--model_label", default=None, help="Stable model id written to each prediction row.")
    parser.add_argument("--input", default="data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--model_max_length", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--torch_dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--system_prompt", default=MEDICAL_SYSTEM_PROMPT)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--continue_on_error", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry_run",
        type=str2bool,
        default=False,
        help="Validate IO and write placeholder predictions without loading a model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    rows = read_jsonl(args.input, max_samples=args.max_samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_done_ids(output) if args.resume else set()
    mode = "a" if args.resume and output.exists() else "w"

    tokenizer = None
    model = None
    if not args.dry_run:
        import torch

        torch.manual_seed(args.seed)
        tokenizer, model = load_tokenizer_and_model(
            args.model_name_or_path,
            peft_path=args.peft_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )

    generated_count = 0
    skipped_count = 0
    error_count = 0
    with output.open(mode, encoding="utf-8") as f:
        for idx, item in enumerate(rows, start=1):
            sample_id = item.get("id")
            if sample_id in done_ids:
                skipped_count += 1
                continue
            question = normalize_text(item.get("question"))
            try:
                if args.dry_run:
                    prediction = "[DRY_RUN] model generation is disabled."
                else:
                    prediction = generate_answer(tokenizer, model, question, args)
                out = build_output_row(item, prediction, args)
                generated_count += 1
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                out = build_output_row(item, "", args, error=repr(exc))
                error_count += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            if generated_count and generated_count % 50 == 0:
                print(f"generated {generated_count} predictions...", flush=True)

    summary_path = args.summary_output
    if summary_path is None:
        out_path = Path(args.output)
        summary_path = str(out_path.with_name(out_path.stem + "_summary.json"))
    summary = {
        "input": args.input,
        "output": args.output,
        "model": model_id_from_args(args),
        "model_name_or_path": args.model_name_or_path,
        "peft_path": args.peft_path,
        "max_samples": args.max_samples,
        "input_rows_loaded": len(rows),
        "generated_rows": generated_count,
        "skipped_existing_rows": skipped_count,
        "error_rows": error_count,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "model_max_length": args.model_max_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
