# -*- coding: utf-8 -*-
"""安全与一致性校验：报告里的每个数字、位置、病灶，都要能回溯到工具输出。

这是图上生成之后的最后一道闸，对应方案第五节那句"不要让语言模型自行编造
病灶坐标、检查数值或文献来源"。做法与 medical_rag 的引用核查同源：

* **数字溯源**：从自然语言答案里抽出所有数字（含百分数），每个数字必须能
  在 ``StructuredReport`` 的 findings 分数 / bbox / 证据文本 / 化验数据里找到
  （允许 0.82 与 82% 这种尺度换算，允许四舍五入误差）。找不到的记为
  ``ungrounded_claims``。
* **病灶溯源**：答案里提到的病理名（中英文）必须出现在 grounded 的 findings、
  VQA 输出或证据文本里，否则是语言模型自己加的。
* **表达安全**：复用训练侧 ``ABSOLUTE_DIAGNOSIS_PATTERNS`` 与 ``DOSE_PATTERNS``。
  影像辅助分析报告里不应出现具体剂量；绝对化诊断（"肯定是"）直接标记。
* **结构合法性**：bbox 越界、掩码退化（占比过大/过小）、分数越界。

所有检查只标记不修改；改写或拦截由调用方决定。本模块只依赖标准库。
"""

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set

from medical_agent import config
from medical_agent.schema import PATHOLOGY_ZH, XRV_PATHOLOGIES, StructuredReport, normalize_text

logger = logging.getLogger(__name__)

try:  # 与训练侧口径对齐；不可导入时用本地镜像
    from training.medical_grpo_rewards import ABSOLUTE_DIAGNOSIS_PATTERNS, DOSE_PATTERNS
except Exception:  # pragma: no cover
    ABSOLUTE_DIAGNOSIS_PATTERNS = (r"一定是", r"肯定是", r"绝对是", r"必然是", r"确诊为", r"无需检查")
    DOSE_PATTERNS = (
        r"(每日|每天|一天)\s*\d+\s*(次|回)",
        r"每次\s*\d+(\.\d+)?\s*(片|粒|颗|袋|支|mg|毫克|ml|毫升|g|克)",
        r"\d+(\.\d+)?\s*(mg|毫克|ml|毫升|g|克)\s*/\s*(kg|公斤|天|日|次)",
    )

_NUMBER_RE = re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*(%|％)?")
_CITATION_RE = re.compile(r"[\[【［]\s*\d+(?:\s*[,，、]\s*\d+)*\s*[\]】］]")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")

# 中英文病理名 -> 规范英文名，供答案文本扫描
_LABEL_ALIASES: Dict[str, str] = {}
for _label in XRV_PATHOLOGIES:
    _LABEL_ALIASES[_label.lower()] = _label
    _LABEL_ALIASES[_label.replace("_", " ").lower()] = _label
for _label, _zh in PATHOLOGY_ZH.items():
    _LABEL_ALIASES[_zh] = _label


def extract_numbers(text: str) -> List[float]:
    """抽出答案里的数字；带 % 的按百分比记录原值（82% -> 82.0）。忽略引用编号。"""
    text = _CITATION_RE.sub(" ", text or "")
    values = []
    for match in _NUMBER_RE.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def _allowed_numbers(report: StructuredReport, lab_data: Optional[Dict[str, Any]] = None) -> Set[float]:
    """结构化报告里所有"可以被引用"的数字，含常见尺度换算。"""
    allowed: Set[float] = set()

    def add(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        allowed.add(round(v, 4))
        if 0.0 <= v <= 1.0:          # 分数/坐标 -> 百分数
            allowed.add(round(v * 100.0, 1))
            allowed.add(float(round(v * 100.0)))
        if 1.0 < v <= 100.0:         # 百分数 -> 小数
            allowed.add(round(v / 100.0, 4))

    for finding in report.findings:
        add(finding.score)
        if finding.bbox is not None:
            for coord in finding.bbox.to_list():
                add(coord)
        if finding.mask is not None:
            add(finding.mask.area_ratio)
    if report.vqa is not None:
        add(report.vqa.confidence)
        for value in extract_numbers(report.vqa.answer + " " + report.vqa.explanation):
            add(value)
    for evidence in report.evidence:
        for value in extract_numbers(evidence.text + " " + evidence.citation):
            add(value)
    add(report.confidence)
    for value in (lab_data or {}).values():
        if isinstance(value, (int, float)):
            add(value)
        elif isinstance(value, str):
            for number in extract_numbers(value):
                add(number)
    return allowed


def _is_grounded_number(value: float, allowed: Set[float]) -> bool:
    if value in allowed:
        return True
    tolerance = 0.5 if value > 1.0 else 0.005
    return any(abs(value - candidate) <= tolerance for candidate in allowed)


def mentioned_labels(text: str) -> List[str]:
    lowered = (text or "").lower()
    found = []
    for alias, label in _LABEL_ALIASES.items():
        if alias and alias in lowered and label not in found:
            found.append(label)
    return found


def _grounded_label_set(report: StructuredReport) -> Set[str]:
    labels: Set[str] = {f.label for f in report.findings if f.grounded}
    if report.vqa is not None:
        for name in report.vqa.findings:
            labels.update(mentioned_labels(name))
        labels.update(mentioned_labels(report.vqa.answer))
    for evidence in report.evidence:
        labels.update(mentioned_labels(evidence.text))
    return labels


def check_report_structure(report: StructuredReport) -> List[str]:
    """结构合法性：只看数据，不看文本。"""
    flags = []
    for index, finding in enumerate(report.findings):
        try:
            finding.validate()
        except ValueError as exc:
            flags.append("invalid_finding[%d]:%s" % (index, exc))
            continue
        if finding.bbox is not None and finding.bbox.area < config.MIN_BOX_AREA:
            flags.append("tiny_bbox[%d]:%s" % (index, finding.label))
        if finding.mask is not None and finding.mask.area_ratio > config.MAX_MASK_AREA:
            flags.append("degenerate_mask[%d]:%s area=%.2f" % (index, finding.label, finding.mask.area_ratio))
        if finding.label not in XRV_PATHOLOGIES and finding.grounded:
            flags.append("unknown_label[%d]:%s" % (index, finding.label))
    if not (0.0 <= float(report.confidence) <= 1.0):
        flags.append("confidence_out_of_range:%r" % report.confidence)
    return flags


def check_answer(answer: str, report: StructuredReport, lab_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """对自然语言答案做溯源与表达安全检查。返回 flags / ungrounded_claims / 统计。"""
    answer = answer or ""
    flags: List[str] = []
    ungrounded: List[str] = []

    allowed = _allowed_numbers(report, lab_data)
    numbers = extract_numbers(answer)
    ungrounded_numbers = [v for v in numbers if not _YEAR_RE.match(str(int(v)) if v == int(v) else "") and not _is_grounded_number(v, allowed)]
    for value in ungrounded_numbers:
        ungrounded.append("number_not_in_tool_output:%g" % value)

    grounded_labels = _grounded_label_set(report)
    for label in mentioned_labels(answer):
        if label not in grounded_labels:
            ungrounded.append("finding_not_in_tool_output:%s" % label)

    for pattern in ABSOLUTE_DIAGNOSIS_PATTERNS:
        if re.search(pattern, answer):
            flags.append("absolute_diagnosis:%s" % pattern)
    for pattern in DOSE_PATTERNS:
        if re.search(pattern, answer):
            flags.append("dose_expression:%s" % pattern)

    if not report.findings and not report.evidence and answer and not report.vqa:
        flags.append("answer_without_any_tool_output")

    flags.extend(check_report_structure(report))
    return {
        "ok": not ungrounded and not any(f.startswith(("absolute_diagnosis", "dose_expression", "invalid_finding")) for f in flags),
        "flags": flags,
        "ungrounded_claims": ungrounded,
        "numbers_checked": len(numbers),
        "numbers_ungrounded": len(ungrounded_numbers),
        "labels_mentioned": mentioned_labels(answer),
    }


def main():
    import argparse
    import json

    from medical_agent.schema import BBox, Evidence, Finding, TOOL_DETECT

    parser = argparse.ArgumentParser(description="grounding 校验自检")
    parser.add_argument("--answer", default="左下肺可见肺部阴影，置信度 82%[1]。肯定是肺炎，建议每次 2 片。另见结节 0.35。")
    args = parser.parse_args()
    report = StructuredReport(
        findings=[Finding(label="Lung Opacity", score=0.82, source_tool=TOOL_DETECT, bbox=BBox(0.1, 0.5, 0.45, 0.9))],
        evidence=[Evidence(chunk_id="c1", citation="指南 | 第二章", text="肺部阴影需结合临床。", source_tier="L1_guideline", score=0.7)],
    )
    print(json.dumps(check_answer(args.answer, report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
