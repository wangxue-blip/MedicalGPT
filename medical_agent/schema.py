# -*- coding: utf-8 -*-
"""medical_agent 的数据契约。

整个智能体只有一条设计原则：**后训练文本模型只读经过校验的工具输出**。它不能
自己编病灶坐标、检验数值、文献来源。这条原则靠数据结构保证，不靠 prompt 自觉——
与 medical_rag 里"引用必须可核验"是同一个思路。

因此：
* 所有坐标归一化到 0~1，与图像分辨率解耦；越界即校验失败。
* 每条 ``Finding`` 都带 ``source_tool`` 与 ``grounded``，报告里任何一个病灶、
  数值、位置都能回溯到某个工具的输出。
* ``StructuredReport`` 是 reporter 唯一允许读取的输入；``ungrounded_claims``
  由 safety 节点写入，reporter 无权清空。

本模块只依赖标准库，任何环境都能导入。``AgentState`` 用 TypedDict 而不是
Pydantic，是为了在 LangGraph 缺失时也能用 ``graph_runtime`` 的回退运行时。
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

try:  # Python 3.8+
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict


# --------------------------------------------------------------------------
# 意图与工具名
# --------------------------------------------------------------------------
INTENT_TEXT_QA = "text_qa"                  # 医学文本问答 -> RAG
INTENT_IMAGE_ANALYSIS = "image_analysis"    # 影像异常分析 -> 检测 (+VQA)
INTENT_LOCALIZE = "localize"                # 目标位置查询 -> 定位
INTENT_REPORT = "report"                    # 综合报告 -> 多工具协作
INTENT_UNSUPPORTED = "unsupported"          # 无法处理（缺图、非胸片、越权）
ALL_INTENTS = (
    INTENT_TEXT_QA, INTENT_IMAGE_ANALYSIS, INTENT_LOCALIZE, INTENT_REPORT, INTENT_UNSUPPORTED,
)

TOOL_VQA = "medical_vqa"
TOOL_DETECT = "detect_abnormality"
TOOL_LOCALIZE = "localize_finding"
TOOL_RETRIEVE = "retrieve_evidence"
TOOL_VISUALIZE = "generate_visualization"
ALL_TOOLS = (TOOL_VQA, TOOL_DETECT, TOOL_LOCALIZE, TOOL_RETRIEVE, TOOL_VISUALIZE)

# 每种意图默认调用的工具序列。Router 只决定意图，工具编排由这张表决定，
# 这样"意图 -> 工具"的映射是可审计的数据而不是散落在代码里的 if/else。
INTENT_TOOL_PLAN: Dict[str, List[str]] = {
    INTENT_TEXT_QA: [TOOL_RETRIEVE],
    INTENT_IMAGE_ANALYSIS: [TOOL_DETECT, TOOL_VQA, TOOL_RETRIEVE],
    INTENT_LOCALIZE: [TOOL_DETECT, TOOL_LOCALIZE, TOOL_VISUALIZE],
    INTENT_REPORT: [TOOL_DETECT, TOOL_LOCALIZE, TOOL_VQA, TOOL_RETRIEVE, TOOL_VISUALIZE],
    INTENT_UNSUPPORTED: [],
}

MODALITY_CXR = "CXR"
SUPPORTED_MODALITIES = (MODALITY_CXR,)

# torchxrayvision densenet121-res224-all 的 18 个病理标签，作为检测器输出的
# 合法标签集合；VQA 与 reporter 提到的 finding 若不在此集合内，safety 会标记。
XRV_PATHOLOGIES = (
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture",
    "Lung Opacity", "Enlarged Cardiomediastinum",
)

# 中文别名，供 reporter 与 safety 做标签对齐；不在这里的按英文原样输出。
PATHOLOGY_ZH = {
    "Atelectasis": "肺不张", "Consolidation": "实变", "Infiltration": "浸润",
    "Pneumothorax": "气胸", "Edema": "肺水肿", "Emphysema": "肺气肿",
    "Fibrosis": "纤维化", "Effusion": "胸腔积液", "Pneumonia": "肺炎",
    "Pleural_Thickening": "胸膜增厚", "Cardiomegaly": "心影增大",
    "Nodule": "结节", "Mass": "肿块", "Hernia": "疝", "Lung Lesion": "肺部病变",
    "Fracture": "骨折", "Lung Opacity": "肺部阴影",
    "Enlarged Cardiomediastinum": "纵隔增宽",
}


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[ \t　]+", " ", text).strip()


# --------------------------------------------------------------------------
# 几何
# --------------------------------------------------------------------------
@dataclass
class BBox:
    """归一化边界框，(x1, y1) 左上，(x2, y2) 右下，全部落在 [0, 1]。"""

    x1: float
    y1: float
    x2: float
    y2: float

    def validate(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value != value:  # NaN
                raise ValueError("BBox.%s 不是数值: %r" % (name, value))
            if value < 0.0 or value > 1.0:
                raise ValueError("BBox.%s=%r 越界，坐标必须归一化到 [0,1]" % (name, value))
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("BBox 退化: (%r,%r)-(%r,%r)" % (self.x1, self.y1, self.x2, self.y2))

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_pixels(self, width: int, height: int) -> List[int]:
        return [
            int(round(self.x1 * width)), int(round(self.y1 * height)),
            int(round(self.x2 * width)), int(round(self.y2 * height)),
        ]

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_list(self) -> List[float]:
        return [round(self.x1, 6), round(self.y1, 6), round(self.x2, 6), round(self.y2, 6)]

    @classmethod
    def from_list(cls, values: Iterable[float]) -> "BBox":
        vals = [float(v) for v in values]
        if len(vals) != 4:
            raise ValueError("BBox 需要 4 个数，得到 %d" % len(vals))
        box = cls(*vals)
        box.validate()
        return box


@dataclass
class Mask:
    """分割掩码。存 PNG 路径而不是内联像素，避免 state 膨胀。

    ``area_ratio`` 是前景占整图比例，供 safety 判断"掩码是否退化"（全图或全空）。
    弥漫性阴影不能只用矩形表示，所以 Finding 允许同时带 bbox 与 mask。
    """

    path: str = ""
    area_ratio: float = 0.0
    rle: str = ""

    def is_empty(self) -> bool:
        return not self.path and not self.rle


# --------------------------------------------------------------------------
# 工具输出
# --------------------------------------------------------------------------
@dataclass
class Finding:
    """一条影像发现。``grounded`` 表示它来自视觉工具的实际输出而非语言模型的推测。"""

    label: str
    score: float
    source_tool: str
    bbox: Optional[BBox] = None
    mask: Optional[Mask] = None
    grounded: bool = True
    modality: str = MODALITY_CXR
    slice_index: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.label:
            raise ValueError("Finding.label 为空")
        if not (0.0 <= float(self.score) <= 1.0):
            raise ValueError("Finding.score=%r 必须在 [0,1]" % self.score)
        if self.source_tool not in ALL_TOOLS:
            raise ValueError("Finding.source_tool=%r 不是已注册工具" % self.source_tool)
        if self.bbox is not None:
            self.bbox.validate()

    @property
    def label_zh(self) -> str:
        return PATHOLOGY_ZH.get(self.label, self.label)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_list() if self.bbox else None
        data["mask"] = asdict(self.mask) if self.mask else None
        data["label_zh"] = self.label_zh
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        bbox = data.get("bbox")
        mask = data.get("mask")
        return cls(
            label=data.get("label", ""),
            score=float(data.get("score", 0.0)),
            source_tool=data.get("source_tool", ""),
            bbox=BBox.from_list(bbox) if bbox else None,
            mask=Mask(**{k: v for k, v in mask.items() if k in ("path", "area_ratio", "rle")}) if mask else None,
            grounded=bool(data.get("grounded", True)),
            modality=data.get("modality", MODALITY_CXR),
            slice_index=data.get("slice_index"),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class Evidence:
    """一条医学证据，字段与 medical_rag.RetrievalHit.to_dict() 对齐。"""

    chunk_id: str
    citation: str
    text: str
    source_tier: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_hit_dict(cls, data: Dict[str, Any]) -> "Evidence":
        return cls(
            chunk_id=str(data.get("chunk_id", "")),
            citation=str(data.get("citation", "")),
            text=str(data.get("text", "")),
            source_tier=str(data.get("source_tier", "")),
            score=float(data.get("final_score", data.get("score", 0.0)) or 0.0),
        )


@dataclass
class VQAAnswer:
    """视觉问答结果。``parse_ok`` 记录 VLM 是否按要求输出了合法 JSON，
    这是 Phase 3 要统计的"结构化输出成功率"。"""

    answer: str = ""
    confidence: float = 0.0
    findings: List[str] = field(default_factory=list)
    location: str = ""
    explanation: str = ""
    raw: str = ""
    parse_ok: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """所有工具的统一返回。``ok=False`` 时 payload 可为空，``error`` 必填。

    ``backend`` 与 ``degraded`` 让降级路径不可被悄悄略过：mock 模式下的结果
    永远带 backend="mock"，评测脚本会拒绝把它当真实结果统计。
    """

    tool: str
    ok: bool
    backend: str = ""
    latency_ms: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    degraded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# 结构化报告：reporter 唯一允许读取的输入
# --------------------------------------------------------------------------
@dataclass
class StructuredReport:
    findings: List[Finding] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    vqa: Optional[VQAAnswer] = None
    summary: str = ""
    confidence: float = 0.0
    caveats: List[str] = field(default_factory=list)
    ungrounded_claims: List[str] = field(default_factory=list)
    visualization_path: str = ""
    modality: str = MODALITY_CXR

    def grounded_labels(self) -> List[str]:
        return sorted({f.label for f in self.findings if f.grounded} | {f.label_zh for f in self.findings if f.grounded})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "evidence": [e.to_dict() for e in self.evidence],
            "vqa": self.vqa.to_dict() if self.vqa else None,
            "summary": self.summary,
            "confidence": round(float(self.confidence), 4),
            "caveats": list(self.caveats),
            "ungrounded_claims": list(self.ungrounded_claims),
            "visualization_path": self.visualization_path,
            "modality": self.modality,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredReport":
        vqa = data.get("vqa")
        return cls(
            findings=[Finding.from_dict(f) for f in data.get("findings") or []],
            evidence=[Evidence(**{k: e.get(k, "" if k != "score" else 0.0) for k in ("chunk_id", "citation", "text", "source_tier", "score")}) for e in data.get("evidence") or []],
            vqa=VQAAnswer(**{k: vqa.get(k) for k in VQAAnswer.__dataclass_fields__ if k in vqa}) if vqa else None,
            summary=data.get("summary", ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            caveats=list(data.get("caveats") or []),
            ungrounded_claims=list(data.get("ungrounded_claims") or []),
            visualization_path=data.get("visualization_path", ""),
            modality=data.get("modality", MODALITY_CXR),
        )


# --------------------------------------------------------------------------
# 图状态
# --------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """LangGraph 状态。节点返回的 dict 会被合并进来（无 reducer，后写覆盖）。

    列表类字段（tool_results / findings / evidence / errors / trace）由节点
    自己负责"读出-追加-写回"，不依赖 LangGraph 的 reducer，这样回退运行时
    的语义与 LangGraph 完全一致。
    """

    # 输入
    image_path: str
    question: str
    symptoms: str
    lab_data: Dict[str, Any]
    mock: bool
    # 路由
    intent: str
    route_reason: str
    tool_plan: List[str]
    # 工具输出（dict 形式，便于序列化；aggregate 时再还原成 dataclass）
    tool_results: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    vqa: Optional[Dict[str, Any]]
    visualization_path: str
    # 汇总与输出
    report: Optional[Dict[str, Any]]
    safety_flags: List[str]
    answer: str
    # 诊断
    errors: List[str]
    trace: List[str]
    timing_ms: Dict[str, float]


def new_state(
    question: str,
    image_path: str = "",
    symptoms: str = "",
    lab_data: Optional[Dict[str, Any]] = None,
    mock: bool = False,
) -> AgentState:
    return AgentState(
        image_path=image_path or "",
        question=normalize_text(question),
        symptoms=normalize_text(symptoms),
        lab_data=dict(lab_data or {}),
        mock=bool(mock),
        intent="",
        route_reason="",
        tool_plan=[],
        tool_results=[],
        findings=[],
        evidence=[],
        vqa=None,
        visualization_path="",
        report=None,
        safety_flags=[],
        answer="",
        errors=[],
        trace=[],
        timing_ms={},
    )


# --------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------
def read_jsonl(path: str, max_rows: int = -1) -> Iterator[Dict[str, Any]]:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON at %s:%d: %s" % (path, line_no, exc)) from exc
            count += 1
            if 0 < max_rows <= count:
                return


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written
