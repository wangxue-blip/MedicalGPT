#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a status manifest for the supplementary SFT and GRPO ablations."""

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "outputs/medical_project/reports/supplementary_ablation_manifest.csv"


def sft_row(label, output_dir, factor, treatment, seed=42, epochs=1, lr="2e-5", rank=8):
    return {
        "stage": "sft",
        "label": label,
        "factor": factor,
        "treatment": treatment,
        "output_dir": output_dir,
        "train_samples": 10000,
        "targets": "q_proj,v_proj,gate_proj,up_proj,down_proj",
        "lora_rank": rank,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "seed": seed,
        "max_eval_samples": 500,
    }


def grpo_row(label, output_dir, factor, treatment, length, reward_mode="separate"):
    return {
        "stage": "grpo",
        "label": label,
        "factor": factor,
        "treatment": treatment,
        "output_dir": output_dir,
        "train_samples": 500,
        "sft_start": "10k Q/V+FFN",
        "max_completion_length": length,
        "reward_mode": reward_mode,
        "seed": 42,
        "similarity_model": "none (char n-gram fallback)",
    }


EXPERIMENTS = [
    sft_row("sft_10k_qvffn_baseline", "outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn", "baseline", "r8_lr2e-5_epoch1_seed42"),
    sft_row("sft_rank4", "outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_rank4", "lora_rank", "r4", rank=4),
    sft_row("sft_rank16", "outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_rank16", "lora_rank", "r16", rank=16),
    sft_row("sft_lr1e5", "outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_lr1e5", "learning_rate", "1e-5", lr="1e-5"),
    sft_row("sft_lr5e5", "outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_lr5e5", "learning_rate", "5e-5", lr="5e-5"),
    sft_row("sft_epoch2", "outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_epoch2", "num_train_epochs", "2", epochs=2),
    sft_row("sft_epoch3", "outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_epoch3", "num_train_epochs", "3", epochs=3),
    sft_row("sft_seed17", "outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_seed17", "random_seed", "17", seed=17),
    sft_row("sft_seed73", "outputs/medical_project/sft/qwen25_7b_lora_10k_1gpu_qvffn_seed73", "random_seed", "73", seed=73),
    grpo_row("grpo_len256_separate", "outputs/medical_project/grpo/qwen25_7b_lora_10k_1gpu_qvffn_grpo_500", "max_completion_length", "256", 256),
    grpo_row("grpo_len384_separate", "outputs/medical_project/grpo/qwen25_7b_lora_10k_1gpu_qvffn_grpo_500_len384_separate", "max_completion_length", "384", 384),
    grpo_row("grpo_len512_separate", "outputs/medical_project/grpo/qwen25_7b_lora_10k_1gpu_qvffn_grpo_500_len512_separate", "max_completion_length", "512", 512),
    grpo_row("grpo_len384_combined", "outputs/medical_project/grpo/qwen25_7b_lora_10k_1gpu_qvffn_grpo_500_len384_combined", "reward_mode", "combined", 384, "combined"),
]


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def enrich(row):
    output_dir = PROJECT_ROOT / row["output_dir"]
    adapter = output_dir / "adapter_config.json"
    train = read_json(output_dir / "train_results.json")
    evaluation = read_json(output_dir / "eval_results.json")
    row["status"] = "completed" if adapter.is_file() else "pending"
    row["train_runtime_seconds"] = train.get("train_runtime", "")
    row["train_loss"] = train.get("train_loss", "")
    row["eval_loss"] = evaluation.get("eval_loss", "")
    row["eval_samples"] = evaluation.get("eval_samples", "")
    return row


def main():
    rows = [enrich(dict(row)) for row in EXPERIMENTS]
    fields = sorted({key for row in rows for key in row})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ablation rows to {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
