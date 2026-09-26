#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agent 系统级评测：工具调用成功率、端到端完成率、延迟、意图分布。

输入同 eval_grounding.py。端到端"完成"的定义：answer 非空、intent 不是
unsupported、且 safety flags 里没有绝对化诊断或剂量表达。

**mock_ratio > 0 时输出会带醒目 warning**：mock 结果验证的是编排逻辑，不是系统
能力，拿它当指标汇报是自欺。
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_agent.schema import INTENT_UNSUPPORTED, read_jsonl  # noqa: E402


def _p95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def evaluate(rows):
    per_tool = defaultdict(lambda: {"ok": 0, "total": 0})
    completed, latencies, intents, mock_flags = [], [], Counter(), []
    for row in rows:
        for result in row.get("tool_results") or []:
            bucket = per_tool[result.get("tool", "?")]
            bucket["total"] += 1
            bucket["ok"] += int(bool(result.get("ok")))
            mock_flags.append(float(result.get("backend") == "mock"))
        flags = (row.get("safety") or {}).get("flags") or []
        unsafe = any(f.startswith(("absolute_diagnosis", "dose_expression")) for f in flags)
        completed.append(float(bool(row.get("answer")) and row.get("intent") != INTENT_UNSUPPORTED and not unsafe))
        timing = row.get("timing_ms") or {}
        latencies.append(sum(v for k, v in timing.items() if str(k).startswith("node:")))
        intents[row.get("intent", "")] += 1
    return {
        "n": len(rows),
        "tool_success_rate": {
            tool: {"ok": b["ok"], "total": b["total"], "rate": b["ok"] / b["total"] if b["total"] else 0.0}
            for tool, b in sorted(per_tool.items())
        },
        "e2e_completion_rate": statistics.mean(completed) if completed else 0.0,
        "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "latency_ms_p95": _p95(latencies),
        "intent_distribution": dict(intents),
        "mock_ratio": statistics.mean(mock_flags) if mock_flags else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Agent 系统级评测")
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="agent")
    args = parser.parse_args()

    rows = list(read_jsonl(args.prediction_file))
    if not rows:
        raise ValueError("predictions 为空: %s" % args.prediction_file)
    result = {"model": args.model, "prediction_file": args.prediction_file}
    result.update(evaluate(rows))
    result["note"] = "工具成功率与完成率是系统工程指标，不等于医学正确性。"
    if result["mock_ratio"] > 0:
        result["warning"] = "含 mock 结果（mock_ratio=%.2f），不可作为真实系统指标" % result["mock_ratio"]
        print("=" * 60)
        print("WARNING: " + result["warning"])
        print("=" * 60)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
