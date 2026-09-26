#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""医学视觉问答评测（VQA-RAD 风格）。

输入 predictions jsonl，每行：
    {"qid": ..., "answer_type": "closed"|"open", "gold": "...", "pred": "...", "parse_ok": true}

闭合题：yes/no 归一化后精确匹配。开放题：token 级 F1（中文按字、英文按词）与精确匹配。
``json_parse_rate`` 统计 VLM 是否按要求输出了合法 JSON——这是"结构化输出成功率"，
Phase 3 决定要不要做 LoRA 的核心依据之一。

这些是启发式自动指标，不等于临床正确性。
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_agent.schema import read_jsonl  # noqa: E402

YES_WORDS = ("yes", "y", "true", "是", "有", "对", "存在", "可见")
NO_WORDS = ("no", "n", "false", "否", "没有", "无", "不", "不存在", "未见")
_PUNCT_RE = re.compile(r"[\s\.,;:!?。，；：！？、\"'()（）\[\]【】]+")
_CJK_RE = re.compile(r"[一-鿿]")


def normalize_closed(text):
    t = _PUNCT_RE.sub("", str(text or "")).lower()
    if not t:
        return ""
    for word in NO_WORDS:
        if t == word or t.startswith(word):
            return "no"
    for word in YES_WORDS:
        if t == word or t.startswith(word):
            return "yes"
    return t


def tokens(text):
    t = str(text or "").lower().strip()
    if _CJK_RE.search(t):
        return [ch for ch in _PUNCT_RE.sub("", t)]
    return [w for w in _PUNCT_RE.split(t) if w]


def token_f1(pred, gold):
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred, gold):
    return float(_PUNCT_RE.sub("", str(pred or "")).lower() == _PUNCT_RE.sub("", str(gold or "")).lower())


def evaluate(rows):
    closed_hits, open_f1, open_em, parse_flags = [], [], [], []
    for row in rows:
        answer_type = str(row.get("answer_type", "open")).lower()
        gold, pred = row.get("gold", ""), row.get("pred", "")
        if "parse_ok" in row:
            parse_flags.append(bool(row["parse_ok"]))
        if answer_type.startswith("close"):
            closed_hits.append(float(normalize_closed(pred) == normalize_closed(gold)))
        else:
            open_f1.append(token_f1(pred, gold))
            open_em.append(exact_match(pred, gold))
    return {
        "n": len(rows),
        "n_closed": len(closed_hits),
        "n_open": len(open_f1),
        "closed_acc": statistics.mean(closed_hits) if closed_hits else None,
        "open_f1": statistics.mean(open_f1) if open_f1 else None,
        "open_em": statistics.mean(open_em) if open_em else None,
        "json_parse_rate": statistics.mean(parse_flags) if parse_flags else None,
    }


def main():
    parser = argparse.ArgumentParser(description="医学 VQA 评测（闭合题准确率 / 开放题 F1 / JSON 解析率）")
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="vlm_base")
    args = parser.parse_args()

    rows = list(read_jsonl(args.prediction_file))
    if not rows:
        raise ValueError("predictions 为空: %s" % args.prediction_file)
    result = {"model": args.model, "prediction_file": args.prediction_file}
    result.update(evaluate(rows))
    result["note"] = "启发式自动指标：闭合题精确匹配、开放题 token F1；不等于临床正确性。"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
