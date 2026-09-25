# -*- coding: utf-8 -*-
"""MedVision-Agent 命令行 demo。

    # 无模型环境：mock 后端跑通编排链路
    python demo/medical_agent_demo.py --mock --image path/to/cxr.png --question "阴影在哪个位置"

    # 服务器：接真实后端
    MEDAGENT_TEXT_PEFT=outputs/medical_project/grpo/xxx \\
    python demo/medical_agent_demo.py --device cuda --image cxr.png --question "出一份报告" --dump_json out.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_agent import config  # noqa: E402
from medical_agent.pipeline import MedVisionAgent  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="MedVision-Agent demo")
    parser.add_argument("--question", default="这张胸片有没有异常？")
    parser.add_argument("--image", default="", help="胸片路径；留空则走纯文本问答")
    parser.add_argument("--symptoms", default="")
    parser.add_argument("--lab_json", default="", help='化验数据 JSON 字符串，如 {"WBC": 12.3}')
    parser.add_argument("--mock", action="store_true", help="全部工具走确定性 mock 后端")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--text_peft", default=config.TEXT_PEFT, help="医疗文本 LoRA/GRPO adapter 目录")
    parser.add_argument("--dump_json", default="", help="把完整结果落盘")
    parser.add_argument("--no_report", action="store_true", help="不打印结构化报告全文")
    return parser.parse_args()


def main():
    args = parse_args()
    lab_data = json.loads(args.lab_json) if args.lab_json else None
    agent = MedVisionAgent(mock=args.mock or config.MOCK, device=args.device, text_peft=args.text_peft)
    result = agent.run(args.question, image_path=args.image, symptoms=args.symptoms, lab_data=lab_data)

    print("=" * 72)
    print("意图: %s  (%s)" % (result["intent"], result["route_reason"]))
    print("运行时: %s" % result["runtime"])
    print("-" * 72)
    print("工具调用:")
    for item in result["tool_results"]:
        status = "ok " if item["ok"] else "FAIL"
        print("  [%s] %-24s backend=%-28s %.1f ms  %s" % (
            status, item["tool"], item["backend"] or "-", item["latency_ms"], item.get("error", "")))
    findings = result["report"].get("findings") or []
    if findings:
        print("-" * 72)
        print("检出发现:")
        for f in findings:
            print("  %-14s %3d%%  bbox=%s  %s" % (f["label_zh"], int(round(f["score"] * 100)), f["bbox"], f["source_tool"]))
    if result["sources"]:
        print("-" * 72)
        print("证据:")
        for index, src in enumerate(result["sources"], start=1):
            print("  [%d] %s" % (index, src["citation"]))
    print("-" * 72)
    print("答案:")
    print(result["answer"])
    print("-" * 72)
    print("安全标记: %s" % (result["safety"]["flags"] or "无"))
    print("无依据表述: %s" % (result["safety"]["ungrounded_claims"] or "无"))
    if result["visualization_path"]:
        print("标注图: %s" % result["visualization_path"])
    if result["degraded"]:
        print("降级项: %s" % "; ".join(result["degraded"]))
    if result["errors"]:
        print("错误: %s" % "; ".join(result["errors"]))
    if not args.no_report:
        print("-" * 72)
        print("结构化报告:")
        print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    if args.dump_json:
        Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已写入 %s" % args.dump_json)


if __name__ == "__main__":
    main()
