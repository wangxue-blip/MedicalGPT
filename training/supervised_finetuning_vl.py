#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LoRA SFT for Qwen2.5-VL on local JSONL image-question-answer data.

The input records are produced by ``prepare_slake_vl_data.py``.  Each record
has an optional local image path, a question, and an answer.  Text-only rows
are supported so the existing MedicalGPT data can be mixed into training; use
``per_device_train_batch_size=1`` when media types are mixed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset

# This environment has a partially installed DeepSpeed package whose import
# requires a CUDA compiler (``nvcc``), even when DeepSpeed is not requested.
# Transformers/Accelerate only need its class for an ``isinstance`` check in
# Trainer initialisation.  This experiment uses native PyTorch + PEFT LoRA,
# so explicitly mark that optional backend unavailable rather than altering
# the shared Python environment or attempting to compile DeepSpeed ops.
from accelerate.utils import other as accelerate_other

accelerate_other.is_deepspeed_available = lambda: False

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments, set_seed


IGNORE_INDEX = -100
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--torch_dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--target_modules", default="q_proj,v_proj")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--report_to", default="none")
    return parser.parse_args()


def read_jsonl(path: Path, maximum: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not row.get("question") or not row.get("answer"):
                raise ValueError(f"{path}:{line_number} needs nonempty question and answer")
            image = row.get("image")
            if image and not (PROJECT_ROOT / image).is_file():
                raise FileNotFoundError(f"{path}:{line_number} image is missing: {PROJECT_ROOT / image}")
            rows.append(row)
            if maximum >= 0 and len(rows) >= maximum:
                break
    if not rows:
        raise ValueError(f"No usable rows loaded from {path}")
    return rows


class VQADataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def image_path(record: dict[str, Any]) -> str | None:
    image = record.get("image")
    return str((PROJECT_ROOT / image).resolve()) if image else None


@dataclass
class VLDataCollator:
    processor: Any

    def messages(self, record: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
        content: list[dict[str, str]] = []
        path = image_path(record)
        if path:
            content.append({"type": "image", "image": path})
        content.append(
            {
                "type": "text",
                "text": f"请根据{'这张医学影像' if path else '以下医疗问题'}简洁作答：\n{record['question']}\n只输出答案，不要补充无关内容。",
            }
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if include_answer:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": record["answer"]}]})
        return messages

    def render(self, records: list[dict[str, Any]], include_answer: bool) -> list[str]:
        return [
            self.processor.apply_chat_template(self.messages(record, include_answer), tokenize=False, add_generation_prompt=not include_answer)
            for record in records
        ]

    def load_images(self, records: list[dict[str, Any]]) -> list[Image.Image] | None:
        paths = [image_path(record) for record in records]
        if any(paths) and not all(paths):
            raise ValueError("Mixed image/text batch: set per_device_train_batch_size=1 for the text-image mixture.")
        if not any(paths):
            return None
        return [Image.open(path).convert("RGB") for path in paths if path]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = self.load_images(features)
        full_text = self.render(features, include_answer=True)
        prompt_text = self.render(features, include_answer=False)
        kwargs: dict[str, Any] = {"text": full_text, "padding": True, "return_tensors": "pt"}
        prompt_kwargs: dict[str, Any] = {"text": prompt_text, "padding": True, "return_tensors": "pt"}
        if images is not None:
            kwargs["images"] = images
            prompt_kwargs["images"] = images
        batch = self.processor(**kwargs)
        prompt_batch = self.processor(**prompt_kwargs)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = IGNORE_INDEX
        for row_index in range(labels.shape[0]):
            full_length = int(batch["attention_mask"][row_index].sum().item())
            prompt_length = int(prompt_batch["attention_mask"][row_index].sum().item())
            if prompt_length >= full_length:
                raise ValueError("No assistant tokens remain after prompt masking; inspect the chat template.")
            left_padding = labels.shape[1] - full_length if self.processor.tokenizer.padding_side == "left" else 0
            labels[row_index, left_padding : left_padding + prompt_length] = IGNORE_INDEX
        batch["labels"] = labels
        return batch


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def source_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Choose at most one of --fp16 and --bf16")
    if args.fp16 and args.torch_dtype != "float16":
        raise ValueError("--fp16 requires --torch_dtype float16")
    if args.bf16 and args.torch_dtype != "bfloat16":
        raise ValueError("--bf16 requires --torch_dtype bfloat16")
    train_rows = read_jsonl(Path(args.train_file), args.max_train_samples)
    eval_rows = read_jsonl(Path(args.validation_file), args.max_eval_samples) if args.validation_file else None
    if args.per_device_train_batch_size > 1 and any(row.get("image") for row in train_rows) and any(not row.get("image") for row in train_rows):
        raise ValueError("The train set mixes text and image rows; use --per_device_train_batch_size 1.")
    set_seed(args.seed)

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    processor.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype(args.torch_dtype),
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[name.strip() for name in args.target_modules.split(",") if name.strip()],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=eval_rows is not None,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="epoch" if eval_rows is not None else "no",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
        data_seed=args.data_seed,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=VQADataset(train_rows),
        eval_dataset=VQADataset(eval_rows) if eval_rows is not None else None,
        data_collator=VLDataCollator(processor),
    )
    run_config = {
        "model_name_or_path": args.model_name_or_path,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows) if eval_rows is not None else 0,
        "train_sources": source_summary(train_rows),
        "eval_sources": source_summary(eval_rows) if eval_rows is not None else {},
        "lora": {"target_modules": args.target_modules, "rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "image_pixels": {"min": args.min_pixels, "max": args.max_pixels},
        "seed": args.seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_result = trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if eval_rows is not None:
        metrics = trainer.evaluate()
        loss = metrics.get("eval_loss")
        if loss is not None and loss < 20:
            metrics["perplexity"] = math.exp(loss)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
