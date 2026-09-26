#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 VQA-RAD 转成统一评测格式。

VQA-RAD（Lau et al. 2018）：315 张放射影像、3515 条临床医生提出的问答，
下载：https://osf.io/89kps/ （CC BY 4.0）。官方给 trainset.json / testset.json，
每条含 image_name / question / answer / answer_type (CLOSED|OPEN) / phrase_type 等。

输出 jsonl：``{"qid", "image_path", "question", "answer", "answer_type": "closed"|"open"}``。
若只给一个文件，按 image_name 哈希 8:2 划分——**按图划分而不是按问答划分**，
同一张图的问答不能同时出现在训练与测试里，否则泄漏。
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_agent.schema import write_jsonl  # noqa: E402


def _get(row, *names, default=""):
    lowered = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name.lower() in lowered and lowered[name.lower()] not in (None, ""):
            return lowered[name.lower()]
    return default


def load_rows(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("questions") or list(data.values())
    return list(data)


def convert(rows, image_dir):
    out = []
    for index, row in enumerate(rows):
        image_name = str(_get(row, "image_name", "image", "img_name"))
        question = str(_get(row, "question")).strip()
        answer = str(_get(row, "answer")).strip()
        answer_type = str(_get(row, "answer_type", "type", default="open")).lower()
        if not image_name or not question:
            continue
        out.append({
            "qid": str(_get(row, "qid", "id", default=index)),
            "image_name": image_name,
            "image_path": str(Path(image_dir) / image_name) if image_dir else image_name,
            "question": question,
            "answer": answer,
            "answer_type": "closed" if answer_type.startswith("close") else "open",
        })
    return out


def split_by_image(rows, test_ratio=0.2):
    train, test = [], []
    for row in rows:
        digest = int(hashlib.md5(row["image_name"].encode("utf-8")).hexdigest(), 16)
        (test if (digest % 100) < int(test_ratio * 100) else train).append(row)
    return train, test


def main():
    parser = argparse.ArgumentParser(description="VQA-RAD -> 统一 jsonl")
    parser.add_argument("--input_train", default="", help="官方 trainset.json")
    parser.add_argument("--input_test", default="", help="官方 testset.json")
    parser.add_argument("--input", default="", help="单个 JSON（无官方划分时使用，按图哈希划分）")
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--output_dir", default="data_processed/medical_project/agent/vqa_rad")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    split_source = ""
    if args.input_train or args.input_test:
        train = convert(load_rows(args.input_train), args.image_dir) if args.input_train else []
        test = convert(load_rows(args.input_test), args.image_dir) if args.input_test else []
        split_source = "official"
    elif args.input:
        train, test = split_by_image(convert(load_rows(args.input), args.image_dir), args.test_ratio)
        split_source = "hash_by_image"
    else:
        parser.error("需要 --input_train/--input_test 或 --input")

    overlap = {r["image_name"] for r in train} & {r["image_name"] for r in test}
    out_dir = Path(args.output_dir)
    write_jsonl(str(out_dir / "train.jsonl"), train)
    write_jsonl(str(out_dir / "test.jsonl"), test)
    summary = {
        "source": "VQA-RAD (https://osf.io/89kps/, CC BY 4.0)",
        "split_source": split_source,
        "train": len(train),
        "test": len(test),
        "train_answer_types": dict(Counter(r["answer_type"] for r in train)),
        "test_answer_types": dict(Counter(r["answer_type"] for r in test)),
        "image_overlap_between_splits": len(overlap),
        "note": "按图划分防泄漏；image_overlap_between_splits 必须为 0。",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
