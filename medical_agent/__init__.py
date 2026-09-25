# -*- coding: utf-8 -*-
"""MedVision-Agent：多模态医疗辅助分析智能体。

视觉工具（胸片检测 / Grad-CAM 定位 / VQA）+ medical_rag 证据检索 + 医疗文本后训练
模型报告生成，由 LangGraph（或同语义回退运行时）编排。后训练模型只读经过校验
的工具输出。本系统是科研与辅助分析原型，不是诊断系统。
"""

from medical_agent.graph import build_graph
from medical_agent.graph_runtime import HAS_LANGGRAPH, RUNTIME_BACKEND
from medical_agent.pipeline import MedVisionAgent
from medical_agent.reporter import Reporter
from medical_agent.schema import (
    ALL_INTENTS,
    ALL_TOOLS,
    AgentState,
    BBox,
    Evidence,
    Finding,
    Mask,
    StructuredReport,
    ToolResult,
    VQAAnswer,
    new_state,
)
from medical_agent.tools.base import BaseTool, ToolRegistry

__all__ = [
    "MedVisionAgent", "build_graph", "Reporter", "BaseTool", "ToolRegistry",
    "AgentState", "BBox", "Mask", "Finding", "Evidence", "VQAAnswer", "ToolResult", "StructuredReport",
    "new_state", "ALL_INTENTS", "ALL_TOOLS", "HAS_LANGGRAPH", "RUNTIME_BACKEND",
]
