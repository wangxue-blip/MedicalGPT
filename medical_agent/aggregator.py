# -*- coding: utf-8 -*-
"""aggregate：把各工具的输出合并成一份 StructuredReport。

这一步不调任何模型，只做数据合并与规则拼接。合并规则里最重要的一条：
localize 覆盖 detect 的同名 finding 时，**只写入 bbox / mask，不改分数与
source_tool**——分数来自检测器、位置来自 CAM，两者的出处要在数据里分得清。

caveats 是给读者看的免责与降级说明：任何工具失败、任何 mock 结果，都必须在
这里留下一行，报告生成器只能追加不能删除。
"""

from typing import Any, Dict, List

from medical_agent.schema import (
    MODALITY_CXR,
    PATHOLOGY_ZH,
    TOOL_DETECT,
    TOOL_LOCALIZE,
    TOOL_RETRIEVE,
    TOOL_VISUALIZE,
    TOOL_VQA,
    Evidence,
    Finding,
    StructuredReport,
    VQAAnswer,
)


def _merge_localized(findings: List[Finding], localized: List[Dict[str, Any]]) -> List[Finding]:
    by_label = {f.label: f for f in findings}
    for item in localized:
        try:
            loc = Finding.from_dict(item)
        except Exception:
            continue
        target = by_label.get(loc.label)
        if target is None:
            loc.source_tool = TOOL_LOCALIZE
            findings.append(loc)
            by_label[loc.label] = loc
            continue
        target.bbox = loc.bbox
        target.mask = loc.mask
        target.extra = dict(target.extra or {})
        target.extra.update({k: v for k, v in (loc.extra or {}).items()})
        target.extra["localized_by"] = loc.extra.get("localized_by", TOOL_LOCALIZE) if loc.extra else TOOL_LOCALIZE
    return findings


def aggregate(state: Dict[str, Any]) -> Dict[str, Any]:
    findings: List[Finding] = []
    localized: List[Dict[str, Any]] = []
    evidence: List[Evidence] = []
    vqa = None
    visualization_path = ""
    caveats: List[str] = []
    refused = False

    for result in state.get("tool_results") or []:
        tool = result.get("tool", "")
        payload = result.get("payload") or {}
        if not result.get("ok"):
            caveats.append("工具 %s 不可用：%s" % (tool, result.get("error") or "未知原因"))
            continue
        if result.get("backend") == "mock":
            caveats.append("工具 %s 的结果来自 mock 后端，不可用于真实分析" % tool)
        if tool == TOOL_DETECT:
            for item in payload.get("findings") or []:
                try:
                    finding = Finding.from_dict(item)
                    finding.validate()
                    findings.append(finding)
                except Exception as exc:
                    caveats.append("检测结果被丢弃（不合法）：%s" % exc)
        elif tool == TOOL_LOCALIZE:
            localized.extend(payload.get("findings") or [])
        elif tool == TOOL_VQA and payload.get("vqa"):
            try:
                data = payload["vqa"]
                vqa = VQAAnswer(**{k: data.get(k) for k in VQAAnswer.__dataclass_fields__ if k in data})
            except Exception as exc:
                caveats.append("VQA 结果解析失败：%s" % exc)
        elif tool == TOOL_RETRIEVE:
            refused = bool(payload.get("refused"))
            for item in payload.get("evidence") or []:
                try:
                    evidence.append(Evidence(
                        chunk_id=str(item.get("chunk_id", "")),
                        citation=str(item.get("citation", "")),
                        text=str(item.get("text", "")),
                        source_tier=str(item.get("source_tier", "")),
                        score=float(item.get("score", 0.0) or 0.0),
                    ))
                except Exception:
                    continue
        elif tool == TOOL_VISUALIZE:
            visualization_path = str(payload.get("visualization_path") or "")

    findings = _merge_localized(findings, localized)
    findings.sort(key=lambda f: f.score, reverse=True)

    if refused:
        caveats.append("检索未找到可靠医学依据，证据部分为空")
    if vqa is not None and not vqa.parse_ok:
        caveats.append("视觉问答未按结构化格式输出，其内容仅作参考")

    # summary 刻意不用阿拉伯数字写计数，避免 reporter 转述时引入报告里不存在的数字
    parts = []
    if findings:
        described = "、".join("%s(%d%%)" % (f.label_zh, int(round(f.score * 100))) for f in findings)
        parts.append("检出发现：%s" % described)
        located = [f.label_zh for f in findings if f.bbox is not None]
        if located:
            parts.append("已定位：%s" % "、".join(located))
    else:
        parts.append("未检出达到阈值的异常")
    if evidence:
        parts.append("检索到医学证据（见来源清单）")
    summary = "；".join(parts) + "。"

    confidence = 0.0
    if findings:
        confidence = max(f.score for f in findings)
    elif vqa is not None:
        confidence = float(vqa.confidence or 0.0)

    report = StructuredReport(
        findings=findings,
        evidence=evidence,
        vqa=vqa,
        summary=summary,
        confidence=confidence,
        caveats=caveats,
        visualization_path=visualization_path,
        modality=MODALITY_CXR,
    )
    return {
        "report": report.to_dict(),
        "findings": [f.to_dict() for f in findings],
        "evidence": [e.to_dict() for e in evidence],
        "vqa": vqa.to_dict() if vqa else None,
        "visualization_path": visualization_path,
    }
