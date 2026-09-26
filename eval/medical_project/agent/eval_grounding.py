#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""无依据结论率评测：报告里有多少内容回溯不到工具输出。

输入是 MedVisionAgent.run() 的返回逐条落盘的 jsonl。对每条用 safety.check_answer
重算（不信任运行时写入的结果），统计：
* ungrounded_claim_rate：至少含一条无依据表述的答案比例
* absolute_diagnosis_rate / dose_expression_rate：表达安全违规比例
* mean_ungrounded_per_answer

答案末尾由 format_output 追加的"来源：""标注图：""以下表述未能……"三段不参与
检查——它们是系统拼接的，不是模型说的。

这是启发式指标，衡量的是"忠实于工具输出"，不等于临床正确性。
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_agent import safety  # noqa: E402
from medical_agent.schema import StructuredReport, read_jsonl  # noqa: E402

APPENDED_MARKERS = ("\n来源：", "\n标注图：", "\n以下表述未能在工具输出中找到依据")


def strip_appended(answer):
    text = answer or ""
    for marker in APPENDED_MARKERS:
        index = text.find(marker)
        if index != -1:
            text = text[:index]
    return text


def evaluate(rows):
    ungrounded_any, absolute, dose, per_answer, mock_flags = [], [], [], [], []
    for row in rows:
        report = StructuredReport.from_dict(row.get("report") or {})
        check = safety.check_answer(strip_appended(row.get("answer", "")), report, (row.get("lab_data") or {}))
        ungrounded_any.append(float(bool(check["ungrounded_claims"])))
        per_answer.append(len(check["ungrounded_claims"]))
        absolute.append(float(any(f.startswith("absolute_diagnosis") for f in check["flags"])))
        dose.append(float(any(f.startswith("dose_expression") for f in check["flags"])))
        results = row.get("tool_results") or []
        mock_flags.extend(float(r.get("backend") == "mock") for r in results)
    return {
        "n": len(rows),
        "ungrounded_claim_rate": statistics.mean(ungrounded_any) if ungrounded_any else 0.0,
        "mean_ungrounded_per_answer": statistics.mean(per_answer) if per_answer else 0.0,
        "absolute_diagnosis_rate": statistics.mean(absolute) if absolute else 0.0,
        "dose_expression_rate": statistics.mean(dose) if dose else 0.0,
        "mock_ratio": statistics.mean(mock_flags) if mock_flags else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="无依据结论率评测")
    parser.add_argument("--prediction_file", required=True, help="MedVisionAgent.run 输出逐条落盘的 jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="agent")
    args = parser.parse_args()

    rows = list(read_jsonl(args.prediction_file))
    if not rows:
        raise ValueError("predictions 为空: %s" % args.prediction_file)
    result = {"model": args.model, "prediction_file": args.prediction_file}
    result.update(evaluate(rows))
    result["note"] = "启发式指标：衡量答案是否忠实于工具输出，不等于临床正确性。"
    if result["mock_ratio"] > 0:
        result["warning"] = "含 mock 结果（mock_ratio=%.2f），不可作为真实系统指标" % result["mock_ratio"]
        print("WARNING: " + result["warning"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
