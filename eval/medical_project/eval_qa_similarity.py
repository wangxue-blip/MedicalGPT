#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate generated medical QA by reference similarity, not medical accuracy."""

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate answers and evaluate reference similarity. This is not medical accuracy."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--peft_path", default=None)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--embedding_model", default="models/bge-small-zh-v1.5")
    parser.add_argument("--embedding_device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction_output", default=None)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def load_embedding_model(model_name, device):
    if not model_name or model_name.lower() in {"none", "false", "off"}:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        print(f"Warning: failed to load embedding model {model_name!r}, using char n-gram fallback: {exc}")
        return None


def main():
    args = parse_args()
    from eval.medical_project.model_utils import (
        generate_answer,
        item_to_question_answer,
        load_tokenizer_and_model,
        model_id_from_paths,
        read_jsonl,
        write_json,
        write_jsonl,
    )
    from training.medical_grpo_rewards import reference_similarity_reward

    tokenizer, model = load_tokenizer_and_model(
        args.model_name_or_path,
        peft_path=args.peft_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    rows = read_jsonl(args.eval_file, max_samples=args.max_samples)
    embedding_model = load_embedding_model(args.embedding_model, args.embedding_device)
    predictions = []

    for item in rows:
        question, answer = item_to_question_answer(item)
        if not question or not answer:
            continue
        prediction = generate_answer(
            tokenizer,
            model,
            question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        score = reference_similarity_reward(
            [[{"content": prediction}]],
            [answer],
            embedding_model=embedding_model,
        )[0]
        predictions.append({
            "id": item.get("id") or item.get("source_id"),
            "question": question,
            "reference_answer": answer,
            "prediction": prediction,
            "similarity": score,
            "category": item.get("category", ""),
        })

    if not predictions:
        raise ValueError(f"No valid predictions generated from {args.eval_file}")
    scores = [row["similarity"] for row in predictions]
    pred_path = args.prediction_output
    if pred_path is None:
        output_path = Path(args.output)
        pred_path = str(output_path.with_name(output_path.stem.replace("qa_similarity", "predictions") + ".jsonl"))
    write_jsonl(pred_path, predictions)

    result = {
        "model": model_id_from_paths(args.model_name_or_path, args.peft_path),
        "model_name_or_path": args.model_name_or_path,
        "peft_path": args.peft_path,
        "eval_file": args.eval_file,
        "prediction_file": pred_path,
        "embedding_model": args.embedding_model,
        "mean_similarity": statistics.mean(scores),
        "median_similarity": statistics.median(scores),
        "min_similarity": min(scores),
        "max_similarity": max(scores),
        "num_samples": len(predictions),
        "note": "Reference similarity only; not a medical accuracy metric.",
    }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
