# -*- coding: utf-8 -*-
"""Core data contract for the medical RAG pipeline.

Everything in :mod:`medical_rag` is built on :class:`Chunk`. The single most
important property of a medical RAG chunk is that it never loses provenance:
a retrieved chunk must always be able to answer "which document, which
section, which page, which version did this come from".

The legacy ``demo/chatpdf.py`` demo stores chunks as bare strings, which makes
traceability impossible. This module exists to fix that.

Nothing here imports torch, transformers or any optional retrieval backend, so
it stays importable in every environment.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


# --------------------------------------------------------------------------
# Source tiers
# --------------------------------------------------------------------------
# Medical RAG lives or dies on corpus quality. Online-consultation QA is NOT an
# authoritative source: retrieving an unreviewed answer and citing it is worse
# than not using RAG at all, because the citation makes the error look credible.
# Tiers carry an authority weight that the reranker uses to break ties.

TIER_GUIDELINE = "L1_guideline"
TIER_DRUG_LABEL = "L1_drug_label"
TIER_TEXTBOOK = "L2_textbook"
TIER_ENCYCLOPEDIA = "L2_encyclopedia"
TIER_KNOWLEDGE_GRAPH = "L2_knowledge_graph"
TIER_CONSULTATION = "L3_consultation"

TIER_AUTHORITY = {
    TIER_GUIDELINE: 1.00,
    TIER_DRUG_LABEL: 0.95,
    TIER_TEXTBOOK: 0.75,
    TIER_ENCYCLOPEDIA: 0.60,
    TIER_KNOWLEDGE_GRAPH: 0.55,
    TIER_CONSULTATION: 0.30,
}

ALL_TIERS = tuple(TIER_AUTHORITY.keys())

CONTENT_TEXT = "text"
CONTENT_TABLE = "table"
CONTENT_QA = "qa"

# Recommendation-class / evidence-level markers used in Chinese clinical
# guidelines. A recommendation stripped of its class has lost its strength, so
# the chunker lifts these into metadata instead of leaving them in the body.
# 顺序敏感：必须长前缀优先。re.search 会在 "III类推荐" 内部从第 3 个字符起
# 匹配到 "I类推荐"，若把 I 放在前面，III 类（不推荐/有害）会被误标成 I 类（强烈推荐）
# ——这是会直接反转临床结论的错标，比缺失危险得多。
EVIDENCE_PATTERNS = (
    (r"(III\s*类推荐|Ⅲ\s*类推荐|3\s*类推荐)", "III"),
    (r"(IIa\s*类推荐|Ⅱa\s*类推荐)", "IIa"),
    (r"(IIb\s*类推荐|Ⅱb\s*类推荐)", "IIb"),
    (r"(II\s*类推荐|Ⅱ\s*类推荐|2\s*类推荐)", "II"),
    (r"(I\s*类推荐|Ⅰ\s*类推荐|1\s*类推荐)", "I"),
    (r"[（(]\s*证据级别\s*[:：]?\s*([ABCD])\s*[）)]", "level:{}"),
    (r"([ABCD])\s*级证据", "level:{}"),
)


def normalize_text(text: Optional[str]) -> str:
    """Collapse whitespace; keep punctuation intact."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[ \t　]+", " ", text).strip()


def stable_id(*parts: Any) -> str:
    """Deterministic id. Deterministic so re-indexing does not churn chunk ids."""
    hasher = hashlib.sha1()
    for part in parts:
        hasher.update(str(part).encode("utf-8", errors="ignore"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()[:16]


def detect_evidence_level(text: str) -> str:
    """Return a recommendation class / evidence level marker, or ''."""
    for pattern, label in EVIDENCE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            if "{}" in label:
                groups = [g for g in match.groups() if g and len(g) == 1]
                return label.format(groups[0]) if groups else ""
            return label
    return ""


@dataclass
class RawDoc:
    """A parsed document before chunking.

    ``blocks`` is an ordered list of dicts, each with at least ``text``; parsers
    add ``page``, ``heading_level``, ``is_heading`` and ``content_type`` when the
    format carries that information.
    """

    doc_id: str
    doc_title: str
    source_tier: str
    source_path: str = ""
    version: str = ""
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def authority(self) -> float:
        return TIER_AUTHORITY.get(self.source_tier, 0.3)


@dataclass
class Chunk:
    """A retrievable unit that never loses provenance.

    ``text`` is the child block used for retrieval (small and focused).
    ``parent_text`` is the surrounding block handed to the generator, so a
    citation keeps enough context to be readable. This is the "small-to-big"
    pattern: retrieve precisely, generate with context.
    """

    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    source_tier: str
    parent_text: str = ""
    source_path: str = ""
    version: str = ""
    section_path: List[str] = field(default_factory=list)
    page: Optional[int] = None
    evidence_level: str = ""
    content_type: str = CONTENT_TEXT
    authority: float = 0.3
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- provenance ---------------------------------------------------------
    def citation_label(self) -> str:
        """Human-readable provenance string shown under the answer."""
        parts = [self.doc_title or self.doc_id]
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        if self.page is not None:
            parts.append(f"第 {self.page} 页")
        if self.version:
            parts.append(self.version)
        if self.evidence_level:
            parts.append(f"推荐等级 {self.evidence_level}")
        return " | ".join(parts)

    def context_text(self) -> str:
        """Text handed to the generator (parent block when available)."""
        return self.parent_text or self.text

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        known = {f for f in cls.__dataclass_fields__}  # noqa: F821 - dataclass attr
        payload = {k: v for k, v in data.items() if k in known}
        payload.setdefault("chunk_id", stable_id(payload.get("text", "")))
        payload.setdefault("doc_id", "")
        payload.setdefault("doc_title", "")
        payload.setdefault("source_tier", TIER_CONSULTATION)
        if not payload.get("authority"):
            payload["authority"] = TIER_AUTHORITY.get(payload["source_tier"], 0.3)
        if payload.get("section_path") is None:
            payload["section_path"] = []
        return cls(**payload)


@dataclass
class RetrievalHit:
    """One retrieved chunk plus why it was retrieved.

    ``route_ranks`` records the 1-based rank each retrieval route gave this
    chunk (``None`` when a route did not return it), which makes fusion and
    debugging auditable rather than a black box.
    """

    chunk: Chunk
    score: float = 0.0
    route_ranks: Dict[str, Optional[int]] = field(default_factory=dict)
    route_scores: Dict[str, float] = field(default_factory=dict)
    rerank_score: Optional[float] = None
    final_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "citation": self.chunk.citation_label(),
            "source_tier": self.chunk.source_tier,
            "score": round(self.score, 6),
            "route_ranks": self.route_ranks,
            "route_scores": {k: round(v, 6) for k, v in self.route_scores.items()},
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "final_score": round(self.final_score, 6),
            "text": self.chunk.text,
        }


# --------------------------------------------------------------------------
# JSONL helpers
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
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
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


def load_chunks(path: str, max_rows: int = -1) -> List[Chunk]:
    return [Chunk.from_dict(row) for row in read_jsonl(path, max_rows=max_rows)]


def save_chunks(path: str, chunks: Iterable[Chunk]) -> int:
    return write_jsonl(path, (chunk.to_dict() for chunk in chunks))
