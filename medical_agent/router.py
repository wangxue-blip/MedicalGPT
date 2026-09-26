# -*- coding: utf-8 -*-
"""规则路由：识别用户意图，输出意图、理由与工具计划。

第一版刻意不用 LLM。意图空间只有四类，规则可解释、可测试、零延迟；更重要的
是，每一次路由决策都会追加写入 ``ROUTER_LOG``——这份日志就是将来评估
"规则 vs LLM 路由"的评测集。没有它，换成 LLM 路由时连"换了之后准不准"都
说不清。

规则优先级（自上而下，先命中先生效）：

1. 没图也没问题                          -> unsupported
2. 给了图但文件不存在/不可读              -> unsupported（带原因）
3. 没图、有问题                          -> text_qa
4. 有图 + 定位类关键词（在哪/位置/定位…）  -> localize
5. 有图 + 报告类关键词（报告/综合/全面…）  -> report
6. 有图 + 其他 / 空问题                   -> image_analysis
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from medical_agent import config
from medical_agent.schema import (
    INTENT_IMAGE_ANALYSIS,
    INTENT_LOCALIZE,
    INTENT_REPORT,
    INTENT_TEXT_QA,
    INTENT_TOOL_PLAN,
    INTENT_UNSUPPORTED,
    normalize_text,
)

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom", ".webp")

# 关键词组：命中即计入 matched_rules。顺序无关，优先级由 route() 决定。
LOCALIZE_KEYWORDS = (
    "在哪", "哪里", "哪个位置", "什么位置", "位置", "定位", "哪个区域", "哪一侧", "哪侧",
    "标出", "标注", "圈出", "框出", "where", "locate", "localize", "region",
)
REPORT_KEYWORDS = (
    "报告", "综合", "全面", "整体", "总结", "完整", "详细分析", "出一份", "写一份",
    "report", "summary", "comprehensive", "overall",
)
ANALYSIS_KEYWORDS = (
    "有没有", "是否", "异常", "什么问题", "看看", "看一下", "分析", "怎么样", "正常吗",
    "有什么", "发现", "abnormal", "finding", "analy",
)


def _has_keyword(text: str, keywords) -> List[str]:
    lowered = text.lower()
    return [kw for kw in keywords if kw.lower() in lowered]


def _image_status(image_path: str) -> Tuple[bool, str]:
    """(可用, 原因)。不读取像素，只看存在性与后缀，保持路由零成本。"""
    if not image_path:
        return False, "no_image"
    path = Path(image_path)
    if not path.exists():
        return False, "image_not_found:%s" % image_path
    if not path.is_file():
        return False, "image_not_a_file:%s" % image_path
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False, "unsupported_suffix:%s" % path.suffix
    try:
        if path.stat().st_size == 0:
            return False, "image_empty:%s" % image_path
    except OSError as exc:
        return False, "image_unreadable:%s" % exc
    return True, "ok"


def route(state: Dict[str, Any]) -> Dict[str, Any]:
    """返回 {intent, route_reason, tool_plan, matched_rules}。纯函数，不改 state。"""
    question = normalize_text(state.get("question"))
    image_path = state.get("image_path") or ""
    image_ok, image_reason = _image_status(image_path)
    matched: List[str] = []

    if not question and not image_path:
        intent, reason = INTENT_UNSUPPORTED, "empty_input"
    elif image_path and not image_ok:
        intent, reason = INTENT_UNSUPPORTED, image_reason
    elif not image_path:
        intent, reason = INTENT_TEXT_QA, "no_image_text_only"
    else:
        loc = _has_keyword(question, LOCALIZE_KEYWORDS)
        rep = _has_keyword(question, REPORT_KEYWORDS)
        ana = _has_keyword(question, ANALYSIS_KEYWORDS)
        if loc:
            matched = loc
            intent, reason = INTENT_LOCALIZE, "localize_keyword:%s" % ",".join(loc[:3])
        elif rep:
            matched = rep
            intent, reason = INTENT_REPORT, "report_keyword:%s" % ",".join(rep[:3])
        elif ana:
            matched = ana
            intent, reason = INTENT_IMAGE_ANALYSIS, "analysis_keyword:%s" % ",".join(ana[:3])
        elif not question:
            intent, reason = INTENT_IMAGE_ANALYSIS, "image_only_default"
        else:
            intent, reason = INTENT_IMAGE_ANALYSIS, "image_default"

    decision = {
        "intent": intent,
        "route_reason": reason,
        "tool_plan": list(INTENT_TOOL_PLAN.get(intent, [])),
        "matched_rules": matched,
    }
    _log_decision(question, bool(image_path), decision)
    return decision


def _log_decision(question: str, has_image: bool, decision: Dict[str, Any], log_path: Optional[str] = None) -> None:
    """尽力而为地追加路由日志；写失败只 warning，绝不影响主流程。"""
    path = log_path or config.ROUTER_LOG
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "question": question,
                "has_image": has_image,
                "intent": decision["intent"],
                "route_reason": decision["route_reason"],
                "matched_rules": decision["matched_rules"],
                "router": "rule_v1",
            }, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover
        logger.warning("路由日志写入失败: %s", exc)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="规则路由自检")
    parser.add_argument("--question", default="这张片子左肺有阴影吗，在哪个位置？")
    parser.add_argument("--image", default="")
    args = parser.parse_args()
    print(json.dumps(route({"question": args.question, "image_path": args.image}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
