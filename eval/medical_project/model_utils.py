#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared model and JSONL helpers for medical project evaluation."""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


MEDICAL_SYSTEM_PROMPT = (
    "你是一个医学科普与健康咨询助手。请基于用户描述给出结构化、谨慎的中文回答，"
    "包含可能原因或分析、处理建议、风险提示和就医建议。不要替代医生诊断，"
    "不要给出具体处方剂量，涉及急症、孕妇、儿童、老人或症状加重时应提示及时就医。"
)


def read_jsonl(path, max_samples=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def write_json(path, data):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def torch_dtype_from_arg(dtype):
    if dtype in (None, "auto"):
        return "auto"
    return getattr(torch, dtype)


def load_tokenizer_and_model(model_name_or_path, peft_path=None, torch_dtype="auto", device_map="auto"):
    tokenizer_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype_from_arg(torch_dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    if peft_path:
        if PeftModel is None:
            raise ImportError("peft is required when --peft_path is set.")
        model = PeftModel.from_pretrained(model, peft_path)
    model.eval()
    return tokenizer, model


def build_chat_prompt(tokenizer, question, system_prompt=MEDICAL_SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\n用户：{question}\n助手："


@torch.inference_mode()
def generate_answer(
    tokenizer,
    model,
    question,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.9,
    system_prompt=MEDICAL_SYSTEM_PROMPT,
):
    prompt = build_chat_prompt(tokenizer, question, system_prompt)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    do_sample = temperature > 0
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def item_to_question_answer(item):
    if "question" in item and "answer" in item:
        return str(item.get("question") or ""), str(item.get("answer") or "")
    conversations = item.get("conversations") or []
    if len(conversations) >= 2:
        question = conversations[0].get("value", "")
        answer = conversations[1].get("value", "")
        return str(question), str(answer)
    return str(item.get("input") or item.get("instruction") or ""), str(item.get("output") or "")


def item_to_text(item, tokenizer=None):
    question, answer = item_to_question_answer(item)
    if question and answer:
        if tokenizer is not None:
            prompt = build_chat_prompt(tokenizer, question)
            return prompt + answer
        return f"用户：{question}\n助手：{answer}"
    return str(item.get("text") or "")


def model_id_from_paths(model_name_or_path, peft_path=None):
    raw = peft_path or model_name_or_path
    return os.path.basename(os.path.normpath(raw)).replace("/", "_")
