# -*- coding: utf-8 -*-
"""LangGraph 编排：把方案里那张工作流图变成代码。

    validate_input -> route -> [unsupported -> format_output]
                             -> run_tools -> aggregate -> safety_check -> format_output -> END

每个节点只返回它改动的键（LangGraph 合并语义），列表字段先复制再追加再整体写回，
这样回退运行时与 LangGraph 行为一致，也不会出现跨次运行的引用共享。
每个节点都包在 try/except 里：异常写进 ``errors``，状态继续前进，``invoke``
永不抛异常——医疗辅助系统宁可给出"某工具失败"的不完整报告，也不能整个崩掉。
"""

import os
import time
from typing import Any, Callable, Dict, List

from medical_agent import config, router, safety
from medical_agent.aggregator import aggregate
from medical_agent.graph_runtime import END, HAS_LANGGRAPH, START, StateGraph
from medical_agent.schema import (
    INTENT_UNSUPPORTED,
    PATHOLOGY_ZH,
    TOOL_DETECT,
    TOOL_LOCALIZE,
    TOOL_RETRIEVE,
    TOOL_VISUALIZE,
    TOOL_VQA,
    AgentState,
    StructuredReport,
    normalize_text,
)

NODE_VALIDATE = "validate_input"
NODE_ROUTE = "route"
NODE_TOOLS = "run_tools"
NODE_AGGREGATE = "aggregate"
NODE_SAFETY = "safety_check"
NODE_FORMAT = "format_output"

UNSUPPORTED_EXPLANATIONS = (
    ("empty_input", "未提供问题或图像"),
    ("image_not_found", "图像文件不存在"),
    ("image_not_a_file", "图像路径不是文件"),
    ("unsupported_suffix", "不支持的图像格式"),
    ("image_empty", "图像文件为空"),
    ("image_unreadable", "图像文件不可读"),
)


def _explain_reason(reason: str) -> str:
    for prefix, text in UNSUPPORTED_EXPLANATIONS:
        if reason.startswith(prefix):
            return text
    return reason or "无法识别的请求"


def _wrap(name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """统一的节点包装：追加 trace、记录耗时、吞异常进 errors。"""

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        trace = list(state.get("trace") or []) + [name]
        try:
            update = dict(fn(state) or {})
        except Exception as exc:
            errors = list(state.get("errors") or [])
            errors.append("%s: %s: %s" % (name, type(exc).__name__, exc))
            update = {"errors": errors}
        timing = dict(state.get("timing_ms") or {})
        timing.update(update.get("timing_ms") or {})
        timing["node:%s" % name] = round((time.perf_counter() - started) * 1000.0, 3)
        update["timing_ms"] = timing
        update["trace"] = trace
        return update

    node.__name__ = name
    return node


def build_graph(registry, reporter, max_steps: int = config.MAX_GRAPH_STEPS):
    """构建并编译图。``registry`` 是 ToolRegistry，``reporter`` 是 Reporter。"""

    # ---- 节点 ---------------------------------------------------------------
    def validate_input(state: Dict[str, Any]) -> Dict[str, Any]:
        errors = list(state.get("errors") or [])
        question = normalize_text(state.get("question"))
        image_path = state.get("image_path") or ""
        lab_data = state.get("lab_data")
        if not question and not image_path:
            errors.append("validate_input: 问题与图像均为空")
        if image_path and not os.path.exists(image_path):
            errors.append("validate_input: 图像不存在 %s" % image_path)
        if lab_data is not None and not isinstance(lab_data, dict):
            errors.append("validate_input: lab_data 不是字典，已忽略")
            lab_data = {}
        return {"question": question, "errors": errors, "lab_data": dict(lab_data or {})}

    def route(state: Dict[str, Any]) -> Dict[str, Any]:
        decision = router.route(state)
        return {
            "intent": decision["intent"],
            "route_reason": decision["route_reason"],
            "tool_plan": list(decision["tool_plan"]),
        }

    def after_route(state: Dict[str, Any]) -> str:
        return "unsupported" if state.get("intent") == INTENT_UNSUPPORTED else "continue"

    def run_tools(state: Dict[str, Any]) -> Dict[str, Any]:
        plan: List[str] = list(state.get("tool_plan") or [])
        results = list(state.get("tool_results") or [])
        errors = list(state.get("errors") or [])
        timing: Dict[str, float] = {}
        image_path = state.get("image_path") or ""
        question = state.get("question") or ""
        detected_labels: List[str] = []
        findings_so_far: List[Dict[str, Any]] = []
        evidence_texts: List[str] = []
        heatmaps: Dict[str, str] = {}

        for tool in plan:
            if not registry.has(tool):
                errors.append("run_tools: 未注册的工具 %s" % tool)
                continue
            if tool == TOOL_DETECT:
                result = registry.run(tool, image_path=image_path)
                if result.ok:
                    findings_so_far = [dict(f) for f in result.payload.get("findings") or []]
                    detected_labels = [f.get("label") for f in findings_so_far if f.get("label")]
            elif tool == TOOL_LOCALIZE:
                if not detected_labels:
                    errors.append("run_tools: 跳过 %s（检测未给出标签）" % tool)
                    continue
                result = registry.run(tool, image_path=image_path, labels=list(detected_labels))
                if result.ok:
                    by_label = {f.get("label"): f for f in findings_so_far}
                    for loc in result.payload.get("findings") or []:
                        target = by_label.get(loc.get("label"))
                        if target is None:
                            findings_so_far.append(dict(loc))
                        else:
                            target["bbox"] = loc.get("bbox")
                            target["mask"] = loc.get("mask")
                    heatmaps = dict(result.payload.get("heatmap_paths") or {})
            elif tool == TOOL_VQA:
                result = registry.run(tool, image_path=image_path, question=question, context="\n".join(evidence_texts[:3]))
            elif tool == TOOL_RETRIEVE:
                query = (question + " " + " ".join(PATHOLOGY_ZH.get(label, label) for label in detected_labels)).strip()
                result = registry.run(tool, query=query)
                if result.ok:
                    evidence_texts = [str(e.get("text", "")) for e in result.payload.get("evidence") or []]
            elif tool == TOOL_VISUALIZE:
                result = registry.run(tool, image_path=image_path, findings=findings_so_far, heatmap_paths=heatmaps)
            else:
                result = registry.run(tool)
            results.append(result.to_dict())
            timing[tool] = round(result.latency_ms, 3)
            if not result.ok:
                errors.append("run_tools: %s 失败: %s" % (tool, result.error))
        return {"tool_results": results, "errors": errors, "timing_ms": timing}

    def aggregate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return aggregate(state)

    def safety_check(state: Dict[str, Any]) -> Dict[str, Any]:
        report = StructuredReport.from_dict(state.get("report") or {})
        answer = reporter.generate(report, state.get("question") or "", state.get("intent") or "")
        check = safety.check_answer(answer, report, state.get("lab_data") or {})
        report_dict = dict(state.get("report") or {})
        report_dict["ungrounded_claims"] = list(check["ungrounded_claims"])
        if check["ungrounded_claims"]:
            answer = answer + "\n以下表述未能在工具输出中找到依据，请以结构化结果为准：" + "；".join(check["ungrounded_claims"])
        return {"answer": answer, "safety_flags": list(check["flags"]), "report": report_dict}

    def format_output(state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("intent") == INTENT_UNSUPPORTED:
            return {"answer": "无法处理该请求：%s" % _explain_reason(state.get("route_reason") or "")}
        answer = state.get("answer") or ""
        evidence = state.get("evidence") or []
        if evidence:
            lines = ["", "来源："]
            for index, item in enumerate(evidence, start=1):
                lines.append("[%d] %s" % (index, item.get("citation", "")))
            answer = answer + "\n".join(lines)
        if state.get("visualization_path"):
            answer = answer + "\n标注图：%s" % state["visualization_path"]
        return {"answer": answer}

    # ---- 装配 ---------------------------------------------------------------
    graph = StateGraph(AgentState) if HAS_LANGGRAPH else StateGraph(AgentState, max_steps=max_steps)
    graph.add_node(NODE_VALIDATE, _wrap(NODE_VALIDATE, validate_input))
    graph.add_node(NODE_ROUTE, _wrap(NODE_ROUTE, route))
    graph.add_node(NODE_TOOLS, _wrap(NODE_TOOLS, run_tools))
    graph.add_node(NODE_AGGREGATE, _wrap(NODE_AGGREGATE, aggregate_node))
    graph.add_node(NODE_SAFETY, _wrap(NODE_SAFETY, safety_check))
    graph.add_node(NODE_FORMAT, _wrap(NODE_FORMAT, format_output))

    graph.add_edge(START, NODE_VALIDATE)
    graph.add_edge(NODE_VALIDATE, NODE_ROUTE)
    graph.add_conditional_edges(NODE_ROUTE, after_route, {"unsupported": NODE_FORMAT, "continue": NODE_TOOLS})
    graph.add_edge(NODE_TOOLS, NODE_AGGREGATE)
    graph.add_edge(NODE_AGGREGATE, NODE_SAFETY)
    graph.add_edge(NODE_SAFETY, NODE_FORMAT)
    graph.add_edge(NODE_FORMAT, END)
    return graph.compile()
