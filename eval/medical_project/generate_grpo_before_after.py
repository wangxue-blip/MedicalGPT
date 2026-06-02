#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate before/after cases for SFT vs GRPO medical QA comparison."""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


DEFAULT_SYSTEM_PROMPT = (
    "你是一个医学科普与健康咨询助手。请基于用户描述给出结构化、谨慎的中文回答，"
    "包含可能原因或分析、处理建议、风险提示和就医建议。不要替代医生诊断，"
    "不要给出具体处方剂量，涉及急症、孕妇、儿童、老人或症状加重时应提示及时就医。"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SFT vs GRPO before/after cases.")
    parser.add_argument("--base_model", required=True, help="Base model name or path.")
    parser.add_argument("--before_peft_path", default=None, help="SFT LoRA adapter path. Empty means merged/full model.")
    parser.add_argument("--after_peft_path", default=None, help="GRPO LoRA adapter path. Empty means merged/full model.")
    parser.add_argument("--before_model", default=None, help="Optional merged/full SFT model path.")
    parser.add_argument("--after_model", default=None, help="Optional merged/full GRPO model path.")
    parser.add_argument("--input_file", required=True, help="GRPO/eval JSONL file containing question/answer.")
    parser.add_argument("--output", default="outputs/medical_project/eval/grpo_before_after_cases.jsonl")
    parser.add_argument("--num_cases", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def read_cases(path, num_cases):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("question"):
                cases.append(item)
            if len(cases) >= num_cases:
                break
    return cases


def dtype_from_arg(dtype):
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def load_model(base_model, model_path=None, peft_path=None, torch_dtype="auto", device_map="auto"):
    target_model = model_path or base_model
    print(f"Loading tokenizer from {target_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {target_model} with dtype={torch_dtype}, device_map={device_map}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        target_model,
        torch_dtype=dtype_from_arg(torch_dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    if peft_path:
        if PeftModel is None:
            raise ImportError("peft is required when --before_peft_path or --after_peft_path is set.")
        print(f"Loading LoRA adapter from {peft_path}", flush=True)
        model = PeftModel.from_pretrained(model, peft_path)
    model.eval()
    print("Model ready", flush=True)
    return tokenizer, model


def build_prompt(tokenizer, system_prompt, question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\n用户：{question}\n助手："


@torch.inference_mode()
def generate_batch(tokenizer, model, cases, args):
    outputs = []
    for idx, item in enumerate(cases, start=1):
        print(f"Generating case {idx}/{len(cases)}", flush=True)
        prompt = build_prompt(tokenizer, args.system_prompt, item["question"])
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        do_sample = args.temperature > 0
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[0][encoded["input_ids"].shape[-1]:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def unload_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model unloaded", flush=True)


def main():
    args = parse_args()
    print(f"Reading {args.num_cases} cases from {args.input_file}", flush=True)
    cases = read_cases(args.input_file, args.num_cases)
    if not cases:
        raise ValueError(f"No valid cases found in {args.input_file}")

    print("=== BEFORE model generation ===", flush=True)
    before_tokenizer, before_model = load_model(
        args.base_model,
        model_path=args.before_model,
        peft_path=args.before_peft_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    before_outputs = generate_batch(before_tokenizer, before_model, cases, args)
    unload_model(before_model)

    print("=== AFTER model generation ===", flush=True)
    after_tokenizer, after_model = load_model(
        args.base_model,
        model_path=args.after_model,
        peft_path=args.after_peft_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    after_outputs = generate_batch(after_tokenizer, after_model, cases, args)
    unload_model(after_model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing output to {output_path}", flush=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item, before, after in zip(cases, before_outputs, after_outputs):
            row = {
                "question": item.get("question", ""),
                "reference_answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "required_sections": item.get("required_sections", []),
                "before_model": args.before_model or args.before_peft_path or args.base_model,
                "after_model": args.after_model or args.after_peft_path or args.base_model,
                "before_response": before,
                "after_response": after,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} before/after cases to {output_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
