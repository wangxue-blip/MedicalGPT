#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare raw medical QA data for the MedicalGPT medical project."""

import argparse
import json
import random
import re
from pathlib import Path


DEFAULT_STATS_PATH = "outputs/medical_project/logs/data_prepare_stats.json"


def normalize_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_from_conversations(conversations):
    if not isinstance(conversations, list):
        return "", ""

    user_roles = {"human", "user", "患者", "病人"}
    assistant_roles = {"gpt", "assistant", "医生", "doctor"}
    question = ""

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = normalize_text(turn.get("from") or turn.get("role")).lower()
        value = normalize_text(turn.get("value") or turn.get("content"))
        if not value:
            continue
        if role in user_roles and not question:
            question = value
            continue
        if question and role in assistant_roles:
            return question, value

    values = [
        normalize_text(turn.get("value") or turn.get("content"))
        for turn in conversations
        if isinstance(turn, dict)
    ]
    values = [value for value in values if value]
    if len(values) >= 2:
        return values[0], values[1]
    return "", ""


def extract_question_answer(record):
    if not isinstance(record, dict):
        return "", ""

    question = normalize_text(record.get("question"))
    answer = normalize_text(record.get("answer"))
    if question and answer:
        return question, answer

    instruction = normalize_text(record.get("instruction"))
    input_text = normalize_text(record.get("input"))
    output = normalize_text(record.get("output"))
    if output and (instruction or input_text):
        if instruction and input_text:
            return f"{instruction}\n{input_text}", output
        return instruction or input_text, output

    question, answer = extract_from_conversations(record.get("conversations"))
    if question and answer:
        return question, answer

    question, answer = extract_from_conversations(record.get("messages"))
    if question and answer:
        return question, answer

    return "", ""


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                yield line_no, None, "empty"
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, f"json_error: {exc}"


def make_sample(sample_id, question, answer, source):
    return {
        "id": f"sample_{sample_id:06d}",
        "question": question,
        "answer": answer,
        "source": source,
        "question_len": len(question),
        "answer_len": len(answer),
    }


def write_jsonl(path, records):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_stats(path, stats):
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_candidates(args):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_count = 0
    invalid_json_count = 0
    empty_line_count = 0
    extracted_count = 0
    valid_count = 0
    too_short_question_count = 0
    too_short_answer_count = 0
    too_long_answer_count = 0
    missing_qa_count = 0
    duplicate_question_count = 0
    seen_questions = set()
    candidates = []

    for _line_no, record, error in iter_jsonl(input_path):
        if error == "empty":
            empty_line_count += 1
            continue
        raw_count += 1
        if error:
            invalid_json_count += 1
            continue

        question, answer = extract_question_answer(record)
        if not question or not answer:
            missing_qa_count += 1
            continue
        extracted_count += 1

        if len(question) < args.min_question_len:
            too_short_question_count += 1
            continue
        if len(answer) < args.min_answer_len:
            too_short_answer_count += 1
            continue
        if len(answer) > args.max_answer_len:
            too_long_answer_count += 1
            continue

        valid_count += 1
        dedupe_key = question.casefold()
        if dedupe_key in seen_questions:
            duplicate_question_count += 1
            continue
        seen_questions.add(dedupe_key)
        candidates.append(
            make_sample(
                sample_id=len(candidates) + 1,
                question=question,
                answer=answer,
                source=args.source,
            )
        )

    question_lens = [sample["question_len"] for sample in candidates]
    answer_lens = [sample["answer_len"] for sample in candidates]
    stats = {
        "input": str(input_path),
        "output": args.output,
        "source": args.source,
        "raw_samples": raw_count,
        "empty_lines": empty_line_count,
        "invalid_json_samples": invalid_json_count,
        "extracted_qa_samples": extracted_count,
        "valid_samples": valid_count,
        "deduped_samples": len(candidates),
        "duplicate_question_samples": duplicate_question_count,
        "filtered": {
            "missing_question_or_answer": missing_qa_count,
            "too_short_question": too_short_question_count,
            "too_short_answer": too_short_answer_count,
            "too_long_answer": too_long_answer_count,
        },
        "min_question_len": args.min_question_len,
        "min_answer_len": args.min_answer_len,
        "max_answer_len": args.max_answer_len,
        "avg_question_len": round(sum(question_lens) / len(question_lens), 2) if question_lens else 0.0,
        "avg_answer_len": round(sum(answer_lens) / len(answer_lens), 2) if answer_lens else 0.0,
    }
    return candidates, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize raw medical QA JSONL into the project candidate format."
    )
    parser.add_argument("--input", required=True, help="Input raw JSONL file.")
    parser.add_argument("--output", required=True, help="Output normalized JSONL file.")
    parser.add_argument("--min_question_len", type=int, default=5)
    parser.add_argument("--min_answer_len", type=int, default=20)
    parser.add_argument("--max_answer_len", type=int, default=2048)
    parser.add_argument("--source", default="medicalgpt_medical", help="Source name written to each sample.")
    parser.add_argument("--stats_output", default=DEFAULT_STATS_PATH, help="Output JSON stats path.")
    parser.add_argument("--print_samples", type=int, default=5, help="Number of random samples to print.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for printed samples.")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates, stats = build_candidates(args)
    write_jsonl(args.output, candidates)
    write_stats(args.stats_output, stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if candidates and args.print_samples > 0:
        rng = random.Random(args.seed)
        sample_count = min(args.print_samples, len(candidates))
        print("\nRandom samples for manual inspection:")
        for sample in rng.sample(candidates, sample_count):
            preview = {
                "id": sample["id"],
                "question": sample["question"][:160],
                "answer": sample["answer"][:240],
                "question_len": sample["question_len"],
                "answer_len": sample["answer_len"],
            }
            print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
