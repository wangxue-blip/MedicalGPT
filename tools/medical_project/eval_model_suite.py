#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the same held-out evaluation suite for one base or PEFT model."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--base_model", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--peft_path", default="")
    parser.add_argument("--output_dir", default="outputs/medical_project/eval_3gpu")
    parser.add_argument("--ppl_eval_file", default="data_processed/medical_project/eval/medical_longtext_eval.jsonl")
    parser.add_argument("--qa_eval_file", default="data_processed/medical_project/eval/medical_qa_eval.jsonl")
    parser.add_argument("--ceval_dev_dir", default="data_raw/medical_project/ceval_dev")
    parser.add_argument("--max_ppl_samples", type=int, default=200)
    parser.add_argument("--max_qa_samples", type=int, default=100)
    parser.add_argument("--max_ceval_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--embedding_model", default="none")
    parser.add_argument("--torch_dtype", default="float16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--skip_ceval", action="store_true", help="Skip C-Eval when dev/valid files are unavailable.")
    return parser.parse_args()


def run(command):
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_args = ["--peft_path", args.peft_path] if args.peft_path else []
    common_model_args = [
        "--model_name_or_path",
        args.base_model,
        *peft_args,
        "--torch_dtype",
        args.torch_dtype,
        "--device_map",
        args.device_map,
    ]

    run(
        [
            sys.executable,
            "eval/medical_project/eval_ppl.py",
            *common_model_args,
            "--eval_file",
            args.ppl_eval_file,
            "--output",
            str(output_dir / f"ppl_{args.label}.json"),
            "--max_samples",
            str(args.max_ppl_samples),
        ]
    )
    prediction_file = output_dir / f"predictions_{args.label}.jsonl"
    run(
        [
            sys.executable,
            "eval/medical_project/eval_qa_similarity.py",
            *common_model_args,
            "--eval_file",
            args.qa_eval_file,
            "--embedding_model",
            args.embedding_model,
            "--embedding_device",
            "cpu",
            "--output",
            str(output_dir / f"qa_similarity_{args.label}.json"),
            "--prediction_output",
            str(prediction_file),
            "--max_samples",
            str(args.max_qa_samples),
            "--max_new_tokens",
            str(args.max_new_tokens),
        ]
    )
    run(
        [
            sys.executable,
            "eval/medical_project/eval_structure_safety.py",
            "--prediction_file",
            str(prediction_file),
            "--output",
            str(output_dir / f"structure_safety_{args.label}.json"),
            "--model",
            args.label,
        ]
    )
    if not args.skip_ceval:
        run(
            [
                sys.executable,
                "eval/medical_project/eval_ceval_medical.py",
                *common_model_args,
                "--ceval_dev_dir",
                args.ceval_dev_dir,
                "--output",
                str(output_dir / f"ceval_medical_{args.label}.json"),
                "--max_samples",
                str(args.max_ceval_samples),
            ]
        )


if __name__ == "__main__":
    main()
