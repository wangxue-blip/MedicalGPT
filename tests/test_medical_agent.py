# -*- coding: utf-8 -*-
"""medical_agent 单测：只用标准库 + pytest，全部 mock，裸环境必须能过。

验证的正是"依赖全缺时编排、路由、汇总、安全校验仍然正确"。
"""

import json
import os
import struct
import sys
import zlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_agent import config, graph_runtime, router, safety  # noqa: E402
from medical_agent.aggregator import aggregate  # noqa: E402
from medical_agent.graph import build_graph  # noqa: E402
from medical_agent.pipeline import MedVisionAgent  # noqa: E402
from medical_agent.reporter import Reporter, describe_location  # noqa: E402
from medical_agent.schema import (  # noqa: E402
    INTENT_IMAGE_ANALYSIS,
    INTENT_LOCALIZE,
    INTENT_REPORT,
    INTENT_TEXT_QA,
    INTENT_UNSUPPORTED,
    TOOL_DETECT,
    TOOL_LOCALIZE,
    TOOL_RETRIEVE,
    TOOL_VISUALIZE,
    TOOL_VQA,
    BBox,
    Evidence,
    Finding,
    Mask,
    StructuredReport,
    new_state,
)
from medical_agent.tools.base import BaseTool, ToolRegistry  # noqa: E402
from medical_agent.tools.detect_abnormality import DetectAbnormalityTool  # noqa: E402
from medical_agent.tools.generate_visualization import GenerateVisualizationTool  # noqa: E402
from medical_agent.tools.localize_finding import LocalizeFindingTool  # noqa: E402
from medical_agent.tools.medical_vqa import MedicalVQATool, parse_vqa_json  # noqa: E402
from medical_agent.tools.retrieve_evidence import RetrieveEvidenceTool  # noqa: E402


# --------------------------------------------------------------------------
# 夹具：用标准库手写一张最小 PNG，不依赖 PIL
# --------------------------------------------------------------------------
def _write_png(path, width=16, height=16, value=128):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes([value]) * width for _ in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)
    return str(path)


@pytest.fixture
def png_path(tmp_path):
    return _write_png(tmp_path / "cxr.png")


@pytest.fixture(autouse=True)
def _isolate_router_log(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROUTER_LOG", str(tmp_path / "router.jsonl"))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
def test_bbox_validate_iou_and_pixels():
    box = BBox.from_list([0.1, 0.2, 0.5, 0.6])
    assert abs(box.iou(box) - 1.0) < 1e-9
    assert box.to_pixels(100, 200) == [10, 40, 50, 120]
    other = BBox(0.3, 0.4, 0.7, 0.8)
    assert 0.0 < box.iou(other) < 1.0


@pytest.mark.parametrize("values", [[0.1, 0.2, 1.5, 0.6], [0.5, 0.2, 0.1, 0.6], [0.1, 0.2, 0.1, 0.6], [0, 0, 1]])
def test_bbox_rejects_out_of_range_or_degenerate(values):
    with pytest.raises(ValueError):
        BBox.from_list(values)


def test_finding_validate_rejects_unknown_tool_and_bad_score():
    with pytest.raises(ValueError):
        Finding(label="Pneumonia", score=0.5, source_tool="nope").validate()
    with pytest.raises(ValueError):
        Finding(label="Pneumonia", score=1.5, source_tool=TOOL_DETECT).validate()


def test_structured_report_roundtrip_keeps_bbox_and_zh_label():
    finding = Finding(label="Lung Opacity", score=0.82, source_tool=TOOL_DETECT, bbox=BBox(0.1, 0.4, 0.5, 0.9), mask=Mask(path="", area_ratio=0.2))
    report = StructuredReport(findings=[finding], evidence=[Evidence("c1", "指南 | 第二章", "肺部阴影", "L1_guideline", 0.7)], confidence=0.82)
    restored = StructuredReport.from_dict(json.loads(json.dumps(report.to_dict(), ensure_ascii=False)))
    assert restored.findings[0].bbox.iou(finding.bbox) > 0.999
    assert restored.findings[0].label_zh == "肺部阴影"
    assert restored.evidence[0].chunk_id == "c1"


def test_new_state_normalizes_and_initializes_lists():
    state = new_state("  这张片子  有问题吗 ", image_path="", lab_data=None)
    assert state["question"] == "这张片子 有问题吗"
    assert state["tool_results"] == [] and state["errors"] == [] and state["trace"] == []


# --------------------------------------------------------------------------
# router
# --------------------------------------------------------------------------
def test_router_rules(png_path, tmp_path):
    assert router.route({"question": "", "image_path": ""})["intent"] == INTENT_UNSUPPORTED
    missing = router.route({"question": "看看", "image_path": str(tmp_path / "nope.png")})
    assert missing["intent"] == INTENT_UNSUPPORTED and missing["route_reason"].startswith("image_not_found")
    bad_suffix = _write_png(tmp_path / "x.txt")
    assert router.route({"question": "看看", "image_path": bad_suffix})["route_reason"].startswith("unsupported_suffix")
    assert router.route({"question": "二甲双胍禁忌", "image_path": ""})["intent"] == INTENT_TEXT_QA
    assert router.route({"question": "阴影在哪个位置", "image_path": png_path})["intent"] == INTENT_LOCALIZE
    assert router.route({"question": "出一份完整报告", "image_path": png_path})["intent"] == INTENT_REPORT
    assert router.route({"question": "有没有异常", "image_path": png_path})["intent"] == INTENT_IMAGE_ANALYSIS
    assert router.route({"question": "", "image_path": png_path})["intent"] == INTENT_IMAGE_ANALYSIS
    log_rows = [json.loads(l) for l in open(config.ROUTER_LOG, encoding="utf-8") if l.strip()]
    assert len(log_rows) == 8 and log_rows[-1]["router"] == "rule_v1"


# --------------------------------------------------------------------------
# safety
# --------------------------------------------------------------------------
def _report():
    return StructuredReport(
        findings=[Finding(label="Lung Opacity", score=0.82, source_tool=TOOL_DETECT, bbox=BBox(0.1, 0.5, 0.45, 0.9))],
        evidence=[Evidence("c1", "指南 | 第二章 | 2023年版", "肺部阴影需结合临床，随访 3 个月。", "L1_guideline", 0.7)],
        confidence=0.82,
    )


def test_safety_accepts_percent_and_fraction_forms():
    check = safety.check_answer("肺部阴影，模型分数 82%[1]；亦即 0.82。", _report())
    assert check["ungrounded_claims"] == []


def test_safety_flags_number_absent_from_tool_output():
    check = safety.check_answer("肺部阴影，置信度 95%[1]。", _report())
    assert any(c.startswith("number_not_in_tool_output:95") for c in check["ungrounded_claims"])


def test_safety_accepts_number_present_in_evidence_and_ignores_year():
    check = safety.check_answer("建议随访 3 个月[1]，参考 2023 年版指南。", _report())
    assert check["ungrounded_claims"] == []


def test_safety_flags_label_absent_in_zh_and_en():
    check = safety.check_answer("可见气胸。Also Pneumothorax.", _report())
    labels = [c for c in check["ungrounded_claims"] if c.startswith("finding_not_in_tool_output")]
    assert labels and all("Pneumothorax" in c for c in labels)


def test_safety_flags_absolute_diagnosis_and_dose():
    check = safety.check_answer("肯定是肺炎，每次 2 片。", _report())
    assert any(f.startswith("absolute_diagnosis") for f in check["flags"])
    assert any(f.startswith("dose_expression") for f in check["flags"])
    assert check["ok"] is False


def test_safety_structure_flags_degenerate_mask_and_tiny_bbox():
    report = StructuredReport(findings=[
        Finding(label="Edema", score=0.5, source_tool=TOOL_DETECT, bbox=BBox(0.1, 0.1, 0.9, 0.9), mask=Mask(path="", area_ratio=0.95)),
        Finding(label="Nodule", score=0.5, source_tool=TOOL_DETECT, bbox=BBox(0.5, 0.5, 0.51, 0.51)),
    ])
    flags = safety.check_report_structure(report)
    assert any(f.startswith("degenerate_mask") for f in flags)
    assert any(f.startswith("tiny_bbox") for f in flags)


# --------------------------------------------------------------------------
# BaseTool 纪律
# --------------------------------------------------------------------------
class _BrokenTool(BaseTool):
    name = TOOL_DETECT

    def _probe(self):
        raise ImportError("no backend here")

    def _run_real(self, **kwargs):
        return {"findings": [{"label": "REAL"}]}

    def _run_mock(self, **kwargs):
        return {"findings": [{"label": "MOCK"}]}


class _ExplodingTool(BaseTool):
    name = TOOL_VQA

    def _probe(self):
        return "boom"

    def _run_real(self, **kwargs):
        raise RuntimeError("kaboom")

    def _run_mock(self, **kwargs):
        return {}


def test_basetool_mock_marks_backend_and_degraded():
    result = _BrokenTool(mock=True).run()
    assert result.ok and result.backend == "mock" and result.degraded


def test_basetool_unavailable_never_returns_mock_data():
    tool = _BrokenTool(mock=False)
    result = tool.run()
    assert result.ok is False and result.payload == {} and result.degraded
    assert "no backend here" in result.error and tool.available is False


def test_basetool_swallows_runtime_exception():
    result = _ExplodingTool(mock=False).run()
    assert result.ok is False and "kaboom" in result.error and result.backend == "boom"


def test_registry_unknown_tool_returns_error_result():
    result = ToolRegistry().run("nothing")
    assert result.ok is False and "未注册" in result.error


# --------------------------------------------------------------------------
# 五个工具的 mock 输出符合契约
# --------------------------------------------------------------------------
def test_tool_mock_payloads_match_contract(png_path):
    detect = DetectAbnormalityTool(mock=True)
    det = detect.run(image_path=png_path)
    assert det.ok and [f["label"] for f in det.payload["findings"]] == ["Lung Opacity", "Cardiomegaly"]
    for finding in det.payload["findings"]:
        Finding.from_dict(finding).validate()

    loc = LocalizeFindingTool(mock=True, detector=detect).run(image_path=png_path, labels=["Lung Opacity", "Hernia"])
    assert loc.ok and len(loc.payload["findings"]) == 2
    assert loc.payload["findings"][0]["bbox"] == [0.1, 0.45, 0.48, 0.9]
    Finding.from_dict(loc.payload["findings"][1]).validate()

    vqa = MedicalVQATool(mock=True).run(image_path=png_path, question="有异常吗")
    assert vqa.ok and vqa.payload["vqa"]["parse_ok"] and vqa.payload["vqa"]["findings"] == ["Lung Opacity"]

    rag = RetrieveEvidenceTool(mock=True).run(query="肺部阴影")
    assert rag.ok and rag.payload["n"] == 2 and rag.payload["refused"] is False
    assert rag.payload["evidence"][0]["source_tier"] == "L1_guideline"

    vis = GenerateVisualizationTool(mock=True).run(image_path=png_path, findings=loc.payload["findings"])
    assert vis.ok and vis.payload["visualization_path"] == "" and vis.payload["boxes_drawn"] == 2


def test_parse_vqa_json_variants():
    clean = parse_vqa_json('{"answer":"无明显异常","confidence":0.9,"findings":[],"location":"","explanation":""}')
    assert clean.parse_ok and clean.confidence == 0.9 and clean.findings == []
    fenced = parse_vqa_json('好的，结果如下：\n```json\n{"answer":"可见阴影","confidence":1.7,"findings":["肺部阴影","气胸"]}\n```\n以上。')
    assert fenced.parse_ok and fenced.confidence == 1.0 and fenced.findings == ["Lung Opacity", "Pneumothorax"]
    broken = parse_vqa_json("我看不出来")
    assert broken.parse_ok is False and broken.answer == "我看不出来"


# --------------------------------------------------------------------------
# aggregator / reporter
# --------------------------------------------------------------------------
def test_aggregator_localize_overrides_bbox_but_keeps_detect_score(png_path):
    detect = DetectAbnormalityTool(mock=True)
    det = detect.run(image_path=png_path).to_dict()
    loc = LocalizeFindingTool(mock=True, detector=detect).run(image_path=png_path, labels=["Lung Opacity"]).to_dict()
    state = {"tool_results": [det, loc]}
    out = aggregate(state)
    opacity = [f for f in out["findings"] if f["label"] == "Lung Opacity"][0]
    assert opacity["bbox"] == [0.1, 0.45, 0.48, 0.9]
    assert opacity["score"] == 0.82 and opacity["source_tool"] == TOOL_DETECT
    assert any("mock" in c for c in out["report"]["caveats"])
    assert "检出发现" in out["report"]["summary"] and "已定位" in out["report"]["summary"]


def test_aggregator_records_failed_tool_in_caveats():
    state = {"tool_results": [{"tool": TOOL_RETRIEVE, "ok": False, "backend": "", "error": "语料不存在", "payload": {}}]}
    out = aggregate(state)
    assert any("retrieve_evidence" in c and "语料不存在" in c for c in out["report"]["caveats"])
    assert out["report"]["findings"] == []


def test_reporter_template_passes_safety(png_path):
    agent = MedVisionAgent(mock=True)
    result = agent.run("出一份完整报告", image_path=png_path)
    assert result["safety"]["ungrounded_claims"] == [], result["safety"]
    assert "放射科医生复核" in result["answer"]


def test_describe_location_uses_patient_side():
    left_image = Finding(label="Lung Opacity", score=0.5, source_tool=TOOL_DETECT, bbox=BBox(0.05, 0.6, 0.4, 0.95))
    assert describe_location(left_image) == "右下肺野"
    right_image = Finding(label="Lung Opacity", score=0.5, source_tool=TOOL_DETECT, bbox=BBox(0.6, 0.1, 0.9, 0.3))
    assert describe_location(right_image) == "左上肺野"
    assert describe_location(Finding(label="Cardiomegaly", score=0.4, source_tool=TOOL_DETECT, bbox=BBox(0.3, 0.4, 0.7, 0.8))) == "心影区"


# --------------------------------------------------------------------------
# graph 端到端（mock）
# --------------------------------------------------------------------------
def test_graph_text_qa_only_runs_rag():
    result = MedVisionAgent(mock=True).run("二甲双胍肾功能不全能用吗")
    assert result["intent"] == INTENT_TEXT_QA
    assert [r["tool"] for r in result["tool_results"]] == [TOOL_RETRIEVE]
    assert result["trace"] == ["validate_input", "route", "run_tools", "aggregate", "safety_check", "format_output"]
    assert "来源：" in result["answer"] and "[1]" in result["answer"]


def test_graph_image_analysis_runs_detect_vqa_rag(png_path):
    result = MedVisionAgent(mock=True).run("有没有异常", image_path=png_path)
    assert result["intent"] == INTENT_IMAGE_ANALYSIS
    assert [r["tool"] for r in result["tool_results"]] == [TOOL_DETECT, TOOL_VQA, TOOL_RETRIEVE]
    assert result["report"]["findings"][0]["label"] == "Lung Opacity"


def test_graph_localize_passes_detected_labels_and_visualizes(png_path):
    result = MedVisionAgent(mock=True).run("阴影在哪个位置", image_path=png_path)
    assert result["intent"] == INTENT_LOCALIZE
    assert [r["tool"] for r in result["tool_results"]] == [TOOL_DETECT, TOOL_LOCALIZE, TOOL_VISUALIZE]
    opacity = [f for f in result["report"]["findings"] if f["label"] == "Lung Opacity"][0]
    assert opacity["bbox"] is not None and opacity["source_tool"] == TOOL_DETECT


def test_graph_report_runs_all_five(png_path):
    result = MedVisionAgent(mock=True).run("出一份完整报告", image_path=png_path)
    assert result["intent"] == INTENT_REPORT
    assert [r["tool"] for r in result["tool_results"]] == [TOOL_DETECT, TOOL_LOCALIZE, TOOL_VQA, TOOL_RETRIEVE, TOOL_VISUALIZE]
    assert result["report"]["vqa"]["parse_ok"] is True
    assert all(r["backend"] == "mock" for r in result["tool_results"])


def test_graph_unsupported_short_circuits(tmp_path):
    result = MedVisionAgent(mock=True).run("看看", image_path=str(tmp_path / "missing.png"))
    assert result["intent"] == INTENT_UNSUPPORTED
    assert result["tool_results"] == []
    assert result["trace"] == ["validate_input", "route", "format_output"]
    assert "图像文件不存在" in result["answer"]


def test_graph_tool_failure_does_not_abort(png_path):
    registry = ToolRegistry()
    registry.register(_BrokenTool(mock=False))                       # detect 不可用
    registry.register(RetrieveEvidenceTool(mock=True))
    registry.register(MedicalVQATool(mock=True))
    graph = build_graph(registry, Reporter(mock=True))
    final = graph.invoke(new_state("有没有异常", image_path=png_path, mock=True))
    tools_run = [r["tool"] for r in final["tool_results"]]
    assert tools_run == [TOOL_DETECT, TOOL_VQA, TOOL_RETRIEVE]
    assert final["tool_results"][0]["ok"] is False
    assert any("detect_abnormality 失败" in e for e in final["errors"])
    assert final["answer"]                                             # 仍然给出了答案
    assert any("不可用" in c for c in final["report"]["caveats"])


def test_graph_never_raises_even_if_node_explodes(png_path, monkeypatch):
    agent = MedVisionAgent(mock=True)
    monkeypatch.setattr(agent.reporter, "generate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reporter down")))
    result = agent.run("有没有异常", image_path=png_path)
    assert any("safety_check: RuntimeError" in e for e in result["errors"])
    assert result["trace"][-1] == "format_output"


@pytest.mark.skipif(graph_runtime.HAS_LANGGRAPH, reason="仅回退运行时有 max_steps 保护")
def test_runtime_fallback_loop_guard():
    graph = graph_runtime.StateGraph(dict, max_steps=3)
    graph.add_node("spin", lambda s: {"n": s.get("n", 0) + 1})
    graph.add_edge(graph_runtime.START, "spin")
    graph.add_conditional_edges("spin", lambda s: "again", {"again": "spin"})
    with pytest.raises(RuntimeError):
        graph.compile().invoke({})


def test_agent_run_is_json_serializable(png_path):
    result = MedVisionAgent(mock=True).run("有没有异常", image_path=png_path, lab_data={"WBC": 12.3})
    dumped = json.dumps(result, ensure_ascii=False)
    assert "Lung Opacity" in dumped and result["runtime"] in ("langgraph", "fallback")
    assert any(d.startswith("detect_abnormality: mock") for d in result["degraded"])
