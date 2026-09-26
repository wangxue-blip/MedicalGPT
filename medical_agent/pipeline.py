# -*- coding: utf-8 -*-
"""MedVisionAgent：端到端门面。

组装五个工具 + 注册表 + 报告器 + 图；``run()`` 返回一个可直接 ``json.dumps``
的字典，里面除了答案还带：路由理由、每个工具的后端与耗时、结构化报告、
安全标记、降级清单、运行时后端、配置快照——评测脚本与 demo 都只消费这一个结构。

``mock=True`` 时全部工具走确定性假后端，整条编排链路在没有任何模型的机器上
可以完整跑通；这是 Phase 1 的验证方式，也是单测的基础。
"""

from typing import Any, Dict, Optional

from medical_agent import config
from medical_agent.graph import build_graph
from medical_agent.graph_runtime import RUNTIME_BACKEND
from medical_agent.reporter import Reporter
from medical_agent.schema import new_state
from medical_agent.tools.base import ToolRegistry
from medical_agent.tools.detect_abnormality import DetectAbnormalityTool
from medical_agent.tools.generate_visualization import GenerateVisualizationTool
from medical_agent.tools.localize_finding import LocalizeFindingTool
from medical_agent.tools.medical_vqa import MedicalVQATool
from medical_agent.tools.retrieve_evidence import RetrieveEvidenceTool


class MedVisionAgent(object):
    def __init__(
        self,
        mock: bool = False,
        device: str = config.DEVICE,
        rag_chunks: str = config.RAG_CHUNKS,
        rag_dense_model: str = config.RAG_DENSE_MODEL,
        vlm_model: str = config.VLM_MODEL,
        text_model: str = config.TEXT_MODEL,
        text_peft: str = config.TEXT_PEFT,
        output_dir: str = config.OUTPUT_DIR,
    ):
        self.mock = bool(mock)
        self.device = device or "cpu"
        self.detector = DetectAbnormalityTool(mock=self.mock, device=self.device)
        self.registry = ToolRegistry()
        self.registry.register(RetrieveEvidenceTool(mock=self.mock, chunks_path=rag_chunks, dense_model=rag_dense_model, device=self.device))
        self.registry.register(self.detector)
        self.registry.register(LocalizeFindingTool(mock=self.mock, detector=self.detector, output_dir=output_dir, device=self.device))
        self.registry.register(MedicalVQATool(mock=self.mock, model_path=vlm_model, device=self.device))
        self.registry.register(GenerateVisualizationTool(mock=self.mock, output_dir=output_dir))
        self.reporter = Reporter(model_path=text_model, peft_path=text_peft, device=self.device, mock=self.mock)
        self.graph = build_graph(self.registry, self.reporter)

    # -- 诊断信息：不强制探测未使用的后端（避免为一个纯文本问题加载 VLM） -------------
    def _describe_tools(self) -> Dict[str, Dict[str, Any]]:
        described = {}
        for name in self.registry.names():
            tool = self.registry.get(name)
            probed = tool._available is not None
            described[name] = {
                "name": name,
                "mock": tool.mock,
                "probed": probed or tool.mock,
                "available": True if tool.mock else (tool._available if probed else None),
                "backend": "mock" if tool.mock else (tool._backend if probed else ""),
                "unavailable_reason": tool.unavailable_reason,
            }
        return described

    def _degraded(self, tools: Dict[str, Dict[str, Any]]) -> list:
        degraded = []
        for name, info in tools.items():
            if info["mock"]:
                degraded.append("%s: mock" % name)
            elif info["probed"] and info["available"] is False:
                degraded.append("%s: 不可用(%s)" % (name, info["unavailable_reason"]))
        if self.reporter.mock:
            degraded.append("reporter: template(mock)")
        elif self.reporter._available is False:
            degraded.append("reporter: template(%s)" % self.reporter.unavailable_reason)
        if RUNTIME_BACKEND != "langgraph":
            degraded.append("graph_runtime: %s" % RUNTIME_BACKEND)
        return degraded

    def run(self, question: str, image_path: str = "", symptoms: str = "", lab_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = new_state(question=question, image_path=image_path, symptoms=symptoms, lab_data=lab_data, mock=self.mock)
        final = self.graph.invoke(state)
        tools = self._describe_tools()
        report = final.get("report") or {}
        return {
            "question": final.get("question", question),
            "image_path": image_path,
            "intent": final.get("intent", ""),
            "route_reason": final.get("route_reason", ""),
            "tool_plan": list(final.get("tool_plan") or []),
            "answer": final.get("answer", ""),
            "report": report,
            "sources": list(final.get("evidence") or []),
            "visualization_path": final.get("visualization_path", ""),
            "safety": {
                "flags": list(final.get("safety_flags") or []),
                "ungrounded_claims": list(report.get("ungrounded_claims") or []),
            },
            "tools": tools,
            "tool_results": list(final.get("tool_results") or []),
            "runtime": RUNTIME_BACKEND,
            "degraded": self._degraded(tools),
            "errors": list(final.get("errors") or []),
            "trace": list(final.get("trace") or []),
            "timing_ms": dict(final.get("timing_ms") or {}),
            "config": config.snapshot(),
        }

    def describe(self) -> Dict[str, Any]:
        tools = self._describe_tools()
        return {
            "mock": self.mock,
            "device": self.device,
            "runtime": RUNTIME_BACKEND,
            "tools": tools,
            "reporter_backend": "template" if self.reporter.mock else self.reporter._backend,
            "degraded": self._degraded(tools),
        }
