# -*- coding: utf-8 -*-
"""医疗 RAG 端到端管线：语料 -> 三路召回 -> RRF 融合 -> 精排 -> 生成 -> 引用核查。

这个模块本身不实现任何检索/生成算法，它只干一件事：把 index / fusion / rerank /
retriever / generator / verifier 这些各自独立的零件按医疗场景要求的顺序接起来，
并且保证**任何一环缺失都不会让整条链路崩掉**。

为什么"证据不足就拒答"要放在管线层而不是生成器里
-------------------------------------------------
通用 RAG 的失败模式是"检索没命中，模型照样编一段听起来很专业的话"。在医疗场景
这是最危险的一种错误：用户看到带引用格式的回答，会默认它有出处。所以拒答必须是
一个**在调用大模型之前**就做完的硬闸门——``should_refuse`` 判定为真时，
``answer()`` 根本不构造 prompt、不碰生成模型，直接返回建议就医的安全话术。
把这个判断交给模型自己"自觉不回答"是不可靠的。

为什么 generate=False 仍然要返回 prompt 和 sources
-------------------------------------------------
开发机通常没有 GPU（本仓库的 Windows 开发环境连 torch 都没有）。如果没有生成模型
就整条链路报错，检索质量和引用链路就永远没法在本地验证。因此本管线把"检索+组装
prompt"和"调用大模型"显式拆开：无生成能力时 ``answer`` 字段为空串，而
``prompt`` / ``sources`` / ``retrieval_explain`` 照常返回，离线就能回归检索效果。

降级链（每一级都只 warning，不抛异常）
-------------------------------------------------
  retriever  : medical_rag.retriever.MedicalRetriever -> 内置等价实现(RRF+精排+权威度)
  reranker   : CrossEncoderReranker(available=True)   -> None，直接用融合分排序
  generator  : HFGenerator(available=True)            -> answer 留空，只给 prompt
  verifier   : CitationVerifier                       -> verification 标记 skipped
  三路索引    : 见 medical_rag.index，单路挂掉不影响其余路
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # 允许 `python -m medical_rag.pipeline` 与 `python medical_rag/pipeline.py` 两种入口
    from medical_rag.schema import Chunk, RetrievalHit, load_chunks, normalize_text
except ImportError:  # pragma: no cover - 直接跑脚本时仓库根目录不在 sys.path 上
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from medical_rag.schema import (  # type: ignore
        Chunk,
        RetrievalHit,
        load_chunks,
        normalize_text,
    )

from medical_rag.fusion import reciprocal_rank_fusion
from medical_rag.index import build_indexes
from medical_rag.rerank import apply_authority_boost

logger = logging.getLogger(__name__)

__all__ = ["MedicalRAG", "REFUSAL_ANSWER", "to_jsonable"]


# --------------------------------------------------------------------------
# 兄弟模块探测
# --------------------------------------------------------------------------
# retriever / generator / verifier 与本模块是并行开发的，任何一个还没落地时
# `import medical_rag.pipeline` 都不能失败，否则整个包连检索都跑不起来。
# 这里统一按"真实实现优先，缺失则用本文件内的等价兜底"的方式解析。

try:
    from medical_rag.rerank import CrossEncoderReranker
except Exception as _exc:  # pragma: no cover
    CrossEncoderReranker = None  # type: ignore
    logger.warning("CrossEncoderReranker 不可用，精排环节将被跳过: %s", _exc)

try:
    from medical_rag.terminology import Terminology
except Exception as _exc:  # pragma: no cover
    Terminology = None  # type: ignore
    logger.warning("Terminology 不可用，术语归一将使用 index 内置兜底表: %s", _exc)

try:
    from medical_rag.retriever import MedicalRetriever as _MedicalRetriever
except Exception as _exc:  # pragma: no cover
    _MedicalRetriever = None  # type: ignore
    logger.warning("medical_rag.retriever 不可用，改用管线内置检索实现: %s", _exc)

try:
    from medical_rag.generator import HFGenerator as _HFGenerator
    from medical_rag.generator import build_prompt as _build_prompt
    from medical_rag.generator import format_sources as _format_sources
except Exception as _exc:  # pragma: no cover
    _HFGenerator = None  # type: ignore
    _build_prompt = None  # type: ignore
    _format_sources = None  # type: ignore
    logger.warning("medical_rag.generator 不可用，改用管线内置 prompt 组装: %s", _exc)

try:
    from medical_rag.verifier import CitationVerifier as _CitationVerifier
    from medical_rag.verifier import should_refuse as _should_refuse
except Exception as _exc:  # pragma: no cover
    _CitationVerifier = None  # type: ignore
    _should_refuse = None  # type: ignore
    logger.warning("medical_rag.verifier 不可用，改用管线内置拒答判据: %s", _exc)


# --------------------------------------------------------------------------
# 拒答话术
# --------------------------------------------------------------------------
# 拒答不是"我不知道"就完事：必须给出下一步可执行的动作，并显式列出需要立刻
# 就医的红旗症状，否则用户很可能把"模型没回答"误读成"问题不严重"。
REFUSAL_ANSWER = (
    "抱歉，在当前知识库中没有检索到足以支撑回答的可靠资料，为避免给出没有依据的医疗建议，"
    "这个问题我不能作答。\n"
    "建议及时就医：携带既往病历、检查报告和正在服用的药物清单，到正规医疗机构相应科室就诊，"
    "由医生结合你的具体情况判断。\n"
    "如果出现剧烈胸痛、呼吸困难、意识模糊、抽搐、持续高热、大量出血等情况，"
    "请立即拨打 120 或前往最近医院急诊。"
)


# --------------------------------------------------------------------------
# 内置兜底实现（仅在对应兄弟模块缺失时启用）
# --------------------------------------------------------------------------
class _FallbackRetriever(object):
    """与 medical_rag.retriever.MedicalRetriever 等价的最小实现。

    刻意保持与正式实现相同的构造签名和 retrieve 语义，这样兄弟模块一落地，
    管线只需换个类名，其余代码零改动。
    """

    def __init__(self, chunks, indexes, reranker=None, terminology=None, recall_k=50,
                 fusion_k=60, route_weights=None, authority_weight=0.15):
        self.chunks = list(chunks or [])
        self.indexes = dict(indexes or {})
        self.reranker = reranker
        self.terminology = terminology
        self.recall_k = max(1, int(recall_k or 1))
        self.fusion_k = int(fusion_k if fusion_k is not None else 60)
        self.route_weights = route_weights
        self.authority_weight = float(authority_weight or 0.0)

    def retrieve(self, query: str, topk: int = 5) -> List[RetrievalHit]:
        query = normalize_text(query)
        if not query or not self.chunks or not self.indexes:
            return []

        # 每一路单独 try：一路挂掉时剩下两路仍然构成有效召回，RRF 本来就容忍缺路。
        route_hits: Dict[str, List[Tuple[int, float]]] = {}
        for name, index in self.indexes.items():
            try:
                route_hits[name] = index.search(query, self.recall_k) or []
            except Exception as exc:
                logger.warning("召回路 %s 检索失败，已跳过: %s", name, exc)

        hits = reciprocal_rank_fusion(
            route_hits, self.chunks, k=self.fusion_k, weights=self.route_weights
        )
        # 精排代价高，只对召回窗口内的候选做，不能对全库做。
        hits = hits[:self.recall_k]

        if self.reranker is not None and hits:
            try:
                if self.reranker.available:
                    hits = self.reranker.rerank(query, hits)
            except Exception as exc:
                logger.warning("交叉编码器精排失败，保留融合序: %s", exc)

        hits = apply_authority_boost(hits, authority_weight=self.authority_weight)
        return hits[:max(1, int(topk or 1))]


def _fallback_build_context(hits: Sequence[RetrievalHit]) -> str:
    """把命中拼成带编号的证据块。

    用 chunk.context_text()（父块优先）而不是 text：检索要精准所以用子块，
    生成要可读所以补回父块上下文，这正是 schema 里 small-to-big 的用意。
    """
    blocks = []
    for idx, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        blocks.append(
            "[%d] 来源：%s\n%s" % (idx, chunk.citation_label(), chunk.context_text())
        )
    return "\n\n".join(blocks)


def _fallback_build_prompt(query: str, hits: Sequence[RetrievalHit]) -> str:
    """最小可用的中文 RAG prompt，硬性要求逐句标注 [编号]。

    要求标注编号不只是为了好看：verifier 需要靠编号把答案句子对回证据块，
    没有编号就无法做引用核查。
    """
    context = _fallback_build_context(hits)
    return (
        "你是一名严谨的临床医学助手。请只依据下面提供的资料回答问题。\n"
        "要求：\n"
        "1. 每一条结论后面必须用 [编号] 标注它依据的资料，编号对应下方资料块。\n"
        "2. 资料中没有提到的内容一律不要写，不要凭常识补充。\n"
        "3. 涉及具体用药、剂量、疗程时，必须提醒以医生处方和药品说明书为准。\n"
        "4. 如果资料不足以回答，直接说明无法回答并建议就医。\n\n"
        "资料：\n%s\n\n"
        "问题：%s\n\n"
        "回答：" % (context, normalize_text(query))
    )


def _fallback_format_sources(hits: Sequence[RetrievalHit]) -> str:
    lines = []
    for idx, hit in enumerate(hits, start=1):
        lines.append("[%d] %s" % (idx, hit.chunk.citation_label()))
    return "\n".join(lines)


def _fallback_should_refuse(hits, min_score: float = 0.0, min_hits: int = 1) -> bool:
    """证据不足判据：命中数不够，或最高分达不到阈值。"""
    hits = list(hits or [])
    if len(hits) < max(1, int(min_hits or 1)):
        return True
    best = max(float(getattr(h, "final_score", 0.0) or 0.0) for h in hits)
    return best < float(min_score or 0.0)


class _FallbackVerifier(object):
    """verifier 模块缺席时的占位：不做判断，只如实标记"没核查"。

    故意不返回 passed=True：把"没核查"伪装成"核查通过"，比不核查更危险。
    """

    def __init__(self, embedder=None, support_threshold=0.62, dose_exempt=True):
        self.embedder = embedder
        self.support_threshold = support_threshold
        self.dose_exempt = dose_exempt

    def verify(self, answer: str, hits) -> Dict[str, Any]:
        return _skipped_verification("medical_rag.verifier 不可用，本次未做引用核查")


def _skipped_verification(reason: str) -> Dict[str, Any]:
    """未执行核查时的统一返回结构，passed 一律为 False。"""
    return {
        "passed": False,
        "skipped": True,
        "reason": reason,
        "checks": [],
        "supported_ratio": 0.0,
        "unsupported": [],
    }


# --------------------------------------------------------------------------
# 管线
# --------------------------------------------------------------------------
class MedicalRAG(object):
    """医疗 RAG 端到端管线。

    典型用法::

        rag = MedicalRAG.from_jsonl("data/chunks.jsonl")
        result = rag.answer("二甲双胍怎么吃", topk=5, generate=False)
        print(result["prompt"])
        print(result["sources"])
    """

    def __init__(
        self,
        chunks: List[Chunk],
        indexes: Optional[Dict[str, Any]] = None,
        retriever: Optional[Any] = None,
        reranker: Optional[Any] = None,
        verifier: Optional[Any] = None,
        generator: Optional[Any] = None,
        terminology: Optional[Any] = None,
        min_score: float = 0.0,
        min_hits: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        degraded: Optional[List[str]] = None,
    ):
        self.chunks = list(chunks or [])
        self.indexes = dict(indexes or {})
        self.retriever = retriever
        self.reranker = reranker
        self.verifier = verifier
        self.generator = generator
        self.terminology = terminology
        self.min_score = float(min_score or 0.0)
        self.min_hits = max(1, int(min_hits or 1))
        self.max_new_tokens = int(max_new_tokens or 512)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.degraded: List[str] = list(degraded or [])

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_chunks(
        cls,
        chunks: List[Chunk],
        dense_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        terminology_path: Optional[str] = None,
        device: str = "cpu",
        routes: Sequence[str] = ("dense", "bm25", "term"),
        recall_k: int = 50,
        fusion_k: int = 60,
        route_weights: Optional[Dict[str, float]] = None,
        authority_weight: float = 0.15,
        generator_model: Optional[str] = None,
        peft_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        support_threshold: float = 0.62,
        dose_exempt: bool = True,
        min_score: float = 0.0,
        min_hits: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> "MedicalRAG":
        """从内存中的 chunk 列表装配整条管线。

        构造期不加载任何大模型：reranker / generator 的权重都走各自的惰性加载，
        所以在没有 GPU 的机器上 from_chunks 依然是秒级返回。
        """
        chunks = list(chunks or [])
        degraded: List[str] = []
        if not chunks:
            logger.warning("from_chunks 收到空语料，所有查询都会走拒答分支")
            degraded.append("语料为空")

        _repair_authority(chunks, degraded)

        terminology = _load_terminology(terminology_path, degraded)

        indexes = build_indexes(
            chunks,
            dense_model=dense_model,
            terminology=terminology,
            routes=tuple(routes),
            device=device,
        )
        missing_routes = [r for r in routes if r not in indexes]
        if missing_routes:
            degraded.append("召回路缺失: %s" % ",".join(missing_routes))

        reranker = _build_reranker(reranker_model, device, degraded)
        retriever = _build_retriever(
            chunks, indexes, reranker, terminology, recall_k, fusion_k,
            route_weights, authority_weight, degraded,
        )
        verifier = _build_verifier(support_threshold, dose_exempt, degraded)
        generator = _build_generator(
            generator_model, peft_path, device_map, torch_dtype, degraded
        )

        return cls(
            chunks=chunks,
            indexes=indexes,
            retriever=retriever,
            reranker=reranker,
            verifier=verifier,
            generator=generator,
            terminology=terminology,
            min_score=min_score,
            min_hits=min_hits,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            degraded=degraded,
        )

    @classmethod
    def from_jsonl(cls, path: str, max_rows: int = -1, **kwargs: Any) -> "MedicalRAG":
        """从 chunk JSONL 装配管线（schema.save_chunks 的产物）。"""
        chunks = load_chunks(path, max_rows=max_rows)
        logger.info("从 %s 载入 %d 条 chunk", path, len(chunks))
        return cls.from_chunks(chunks, **kwargs)

    # -- 主流程 -------------------------------------------------------------
    def answer(self, query: str, topk: int = 5, generate: bool = True) -> Dict[str, Any]:
        """检索 -> 拒答闸门 -> 生成 -> 引用核查，返回固定结构的结果字典。

        Returns:
            dict，固定包含 query / refused / answer / sources / verification /
            prompt / retrieval_explain 七个键。
        """
        query = normalize_text(query)
        timing: Dict[str, float] = {}

        started = time.perf_counter()
        hits = self._retrieve(query, topk)
        timing["retrieve_ms"] = _elapsed_ms(started)

        refused = self._should_refuse(hits)
        sources = [hit.to_dict() for hit in hits]

        if refused:
            # 拒答是硬闸门：不组装 prompt、不调用生成模型，避免"无依据却看似有依据"。
            logger.info("证据不足，触发拒答: %s", query[:40])
            return self._result(
                query=query, refused=True, answer=REFUSAL_ANSWER, sources=sources,
                verification=_skipped_verification("证据不足已拒答，未生成回答"),
                prompt="", hits=hits, topk=topk, timing=timing,
                refuse_reason=self._refuse_reason(hits),
            )

        prompt = self._build_prompt(query, hits)

        can_generate = bool(generate) and self._generator_available()
        answer_text = ""
        verification: Dict[str, Any] = _skipped_verification(
            "generate=False，未生成回答" if not generate else "生成模型不可用，未生成回答"
        )

        if can_generate:
            started = time.perf_counter()
            answer_text = self._generate(prompt)
            timing["generate_ms"] = _elapsed_ms(started)
            if answer_text:
                started = time.perf_counter()
                verification = self._verify(answer_text, hits)
                timing["verify_ms"] = _elapsed_ms(started)
            else:
                verification = _skipped_verification("生成失败或返回空，未做引用核查")

        return self._result(
            query=query, refused=False, answer=answer_text, sources=sources,
            verification=verification, prompt=prompt, hits=hits, topk=topk,
            timing=timing, refuse_reason="",
        )

    # -- 各环节的异常隔离 ----------------------------------------------------
    def _retrieve(self, query: str, topk: int) -> List[RetrievalHit]:
        if self.retriever is None:
            return []
        try:
            return list(self.retriever.retrieve(query, topk=topk) or [])
        except Exception as exc:
            # 检索炸了就当成"什么都没召回"，后续自然走拒答，比抛异常给上层安全。
            logger.warning("检索失败，按无召回处理: %s", exc)
            return []

    def _should_refuse(self, hits: List[RetrievalHit]) -> bool:
        func = _should_refuse or _fallback_should_refuse
        try:
            return bool(func(hits, min_score=self.min_score, min_hits=self.min_hits))
        except Exception as exc:
            logger.warning("拒答判据执行失败，保守按拒答处理: %s", exc)
            return True

    def _build_prompt(self, query: str, hits: List[RetrievalHit]) -> str:
        func = _build_prompt or _fallback_build_prompt
        try:
            return func(query, hits)
        except Exception as exc:
            logger.warning("prompt 组装失败，改用内置模板: %s", exc)
            return _fallback_build_prompt(query, hits)

    def _generator_available(self) -> bool:
        if self.generator is None:
            return False
        try:
            return bool(self.generator.available)
        except Exception as exc:
            logger.warning("生成模型可用性探测失败，按不可用处理: %s", exc)
            return False

    def _generate(self, prompt: str) -> str:
        try:
            text = self.generator.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            # 只去首尾空白，不做 normalize_text：生成结果里的缩进和换行是答案
            # 格式的一部分，压掉之后列表和分点会糊成一段。
            return (text or "").strip()
        except Exception as exc:
            # HFGenerator.generate 在后端不可用时会主动抛 RuntimeError，
            # 这里统一按"没生成出来"处理，让上层继续返回 prompt + sources。
            logger.warning("生成失败，返回空回答: %s", exc)
            return ""

    def _verify(self, answer_text: str, hits: List[RetrievalHit]) -> Dict[str, Any]:
        if self.verifier is None:
            return _skipped_verification("未配置引用核查器")
        try:
            result = self.verifier.verify(answer_text, hits)
            return result if isinstance(result, dict) else _skipped_verification(
                "引用核查返回值不是 dict，已忽略"
            )
        except Exception as exc:
            logger.warning("引用核查失败: %s", exc)
            return _skipped_verification("引用核查异常: %s" % exc)

    # -- 结果组装 -----------------------------------------------------------
    def _refuse_reason(self, hits: List[RetrievalHit]) -> str:
        if not hits:
            return "三路召回均无结果"
        best = max(float(getattr(h, "final_score", 0.0) or 0.0) for h in hits)
        if len(hits) < self.min_hits:
            return "命中数 %d 低于下限 %d" % (len(hits), self.min_hits)
        return "最高分 %.4f 低于阈值 %.4f" % (best, self.min_score)

    def _result(self, query, refused, answer, sources, verification, prompt,
                hits, topk, timing, refuse_reason) -> Dict[str, Any]:
        return {
            "query": query,
            "refused": bool(refused),
            "answer": answer,
            "sources": sources,
            "verification": verification,
            "prompt": prompt,
            "retrieval_explain": self._explain(hits, topk, timing, refuse_reason),
        }

    def _explain(self, hits, topk, timing, refuse_reason) -> Dict[str, Any]:
        """把"为什么是这几条、为什么是这个顺序"摊开。

        医疗 RAG 出问题时，"答案错"往往只是表象，真正要定位的是哪一路召回把
        无关段落顶了上来。把 backend、每路排名、加权前后的分数一起返回，
        线上排查才不用回放整条链路。
        """
        backends = {}
        for name, index in self.indexes.items():
            backends[name] = getattr(index, "backend", "unknown")

        ranking = []
        for rank, hit in enumerate(hits, start=1):
            ranking.append({
                "rank": rank,
                "chunk_id": hit.chunk.chunk_id,
                "source_tier": hit.chunk.source_tier,
                "authority": round(float(getattr(hit.chunk, "authority", 0.0) or 0.0), 4),
                "route_ranks": hit.route_ranks,
                "fusion_score": round(float(hit.score), 6),
                "rerank_score": (None if hit.rerank_score is None
                                 else round(float(hit.rerank_score), 6)),
                "final_score": round(float(hit.final_score), 6),
            })

        return {
            "corpus_size": len(self.chunks),
            "routes": sorted(self.indexes.keys()),
            "route_backends": backends,
            "reranker_enabled": self._reranker_enabled(),
            "generator_available": self._generator_available(),
            "verifier_enabled": self.verifier is not None,
            "topk": int(topk),
            "num_hits": len(hits),
            "refuse_rule": {"min_score": self.min_score, "min_hits": self.min_hits},
            "refuse_reason": refuse_reason,
            "ranking": ranking,
            "timing_ms": {k: round(v, 2) for k, v in timing.items()},
            "degraded": list(self.degraded),
        }

    def _reranker_enabled(self) -> bool:
        if self.reranker is None:
            return False
        try:
            return bool(self.reranker.available)
        except Exception:
            return False

    def format_sources(self, hits_or_result: Any) -> str:
        """把命中渲染成"参考资料"清单，供 CLI / 前端直接展示。"""
        hits = hits_or_result
        if isinstance(hits_or_result, dict):
            hits = hits_or_result.get("sources") or []
            return "\n".join(
                "[%d] %s" % (i, row.get("citation", row.get("chunk_id", "")))
                for i, row in enumerate(hits, start=1)
            )
        func = _format_sources or _fallback_format_sources
        try:
            return func(hits)
        except Exception as exc:
            logger.warning("来源渲染失败，改用内置模板: %s", exc)
            return _fallback_format_sources(hits)

    def __repr__(self) -> str:
        return "<MedicalRAG chunks=%d routes=%s rerank=%s gen=%s>" % (
            len(self.chunks), ",".join(sorted(self.indexes.keys())) or "-",
            self._reranker_enabled(), self._generator_available(),
        )


# --------------------------------------------------------------------------
# 装配辅助
# --------------------------------------------------------------------------
def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def to_jsonable(value: Any) -> Any:
    """把 answer() 的结果递归转成纯 JSON 可序列化结构。

    为什么不让 answer() 直接返回纯 dict：verifier.verify() 的 checks 里放的是
    CitationCheck 数据类，verifier.format_report() 依赖属性访问（check.reason）。
    如果管线在返回前就把它拍平成 dict，format_report 会当场坏掉。所以保持
    verification 原样透传，把"要发 HTTP / 落盘"这一步显式交给本函数：

        json.dumps(to_jsonable(rag.answer(q)), ensure_ascii=False)
    """
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _repair_authority(chunks: List[Chunk], degraded: List[str]) -> None:
    """把"从没被赋过值"的 authority 按来源分级补上。

    Chunk.authority 的 dataclass 默认值是 0.3，而 chunker 才会按 TIER_AUTHORITY
    写入真值。于是手工构造 Chunk、或从非 chunker 产出的 JSONL 载入时，一份指南
    会带着 0.3 进检索——apply_authority_boost 照算不误，只是"指南优先于问诊"这条
    规则被悄悄抹平了。这种错误不报错、不崩溃，只是排序变差，极难在线上发现，
    所以在管线入口处补一刀。

    只修"值恰好等于默认 0.3 且该分级本应更高"的情况：这精确对应"没人设过"，
    不会覆盖调用方刻意调低某条证据权重的意图（问诊本身就是 0.30，不受影响）。
    """
    try:
        from medical_rag.schema import TIER_AUTHORITY
    except ImportError:  # pragma: no cover
        return

    default_authority = 0.3
    fixed = 0
    for chunk in chunks:
        expected = TIER_AUTHORITY.get(getattr(chunk, "source_tier", ""), default_authority)
        current = float(getattr(chunk, "authority", default_authority) or 0.0)
        if abs(current - default_authority) < 1e-9 and expected > default_authority:
            chunk.authority = expected
            fixed += 1
    if fixed:
        logger.warning("%d 条 chunk 的 authority 仍是默认值，已按来源分级补齐", fixed)
        degraded.append("已按来源分级补齐 %d 条 chunk 的 authority" % fixed)


def _load_terminology(terminology_path: Optional[str], degraded: List[str]):
    """术语表：指定路径优先，失败回落内置表，再失败返回 None 交给 index 兜底。"""
    if Terminology is None:
        degraded.append("Terminology 模块不可用")
        return None
    if terminology_path:
        try:
            return Terminology.from_jsonl(terminology_path)
        except Exception as exc:
            logger.warning("加载术语表 %s 失败，改用内置表: %s", terminology_path, exc)
            degraded.append("术语表加载失败，已回落内置表")
    try:
        return Terminology.default()
    except Exception as exc:
        logger.warning("内置术语表构建失败: %s", exc)
        degraded.append("术语表不可用")
        return None


def _build_reranker(reranker_model: Optional[str], device: str, degraded: List[str]):
    """精排器构造失败不算致命：融合分本身就是一个可用的排序基线。"""
    if CrossEncoderReranker is None:
        degraded.append("精排模块不可用")
        return None
    try:
        reranker = (CrossEncoderReranker(model_name_or_path=reranker_model, device=device)
                    if reranker_model else CrossEncoderReranker(device=device))
    except Exception as exc:
        logger.warning("构造交叉编码器失败，已跳过精排: %s", exc)
        degraded.append("精排构造失败")
        return None
    try:
        if not reranker.available:
            degraded.append("精排模型不可用，按融合分排序")
    except Exception:
        degraded.append("精排可用性未知")
    return reranker


def _build_retriever(chunks, indexes, reranker, terminology, recall_k, fusion_k,
                     route_weights, authority_weight, degraded: List[str]):
    cls = _MedicalRetriever or _FallbackRetriever
    if _MedicalRetriever is None:
        degraded.append("使用管线内置检索实现")
    try:
        return cls(
            chunks, indexes, reranker=reranker, terminology=terminology,
            recall_k=recall_k, fusion_k=fusion_k, route_weights=route_weights,
            authority_weight=authority_weight,
        )
    except Exception as exc:
        logger.warning("构造 %s 失败，改用内置检索实现: %s", cls.__name__, exc)
        degraded.append("检索器构造失败，已回落内置实现")
        return _FallbackRetriever(
            chunks, indexes, reranker=reranker, terminology=terminology,
            recall_k=recall_k, fusion_k=fusion_k, route_weights=route_weights,
            authority_weight=authority_weight,
        )


def _build_verifier(support_threshold: float, dose_exempt: bool, degraded: List[str]):
    cls = _CitationVerifier or _FallbackVerifier
    if _CitationVerifier is None:
        degraded.append("引用核查模块不可用")
    try:
        return cls(support_threshold=support_threshold, dose_exempt=dose_exempt)
    except Exception as exc:
        logger.warning("构造引用核查器失败: %s", exc)
        degraded.append("引用核查构造失败")
        return _FallbackVerifier()


def _build_generator(generator_model: Optional[str], peft_path: Optional[str],
                     device_map: str, torch_dtype: str, degraded: List[str]):
    """没给模型路径就直接不装生成器——这是本地验证检索链路的默认姿态。"""
    if not generator_model:
        return None
    if _HFGenerator is None:
        degraded.append("生成模块不可用")
        return None
    try:
        generator = _HFGenerator(
            generator_model, peft_path=peft_path,
            device_map=device_map, torch_dtype=torch_dtype,
        )
    except Exception as exc:
        logger.warning("构造生成器失败，将只返回 prompt: %s", exc)
        degraded.append("生成器构造失败")
        return None
    try:
        if not generator.available:
            degraded.append("生成模型不可用，answer 将为空")
    except Exception:
        degraded.append("生成模型可用性未知")
    return generator


# --------------------------------------------------------------------------
# 自检 / 命令行
# --------------------------------------------------------------------------
def _demo_chunks() -> List[Chunk]:
    """几条带真实医疗特征的样例：中文术语、英文药名、剂量、推荐等级。"""
    from medical_rag.schema import (
        TIER_AUTHORITY, TIER_CONSULTATION, TIER_DRUG_LABEL, TIER_GUIDELINE,
        TIER_TEXTBOOK, stable_id,
    )

    raw = [
        ("二甲双胍是2型糖尿病的一线降糖药物，起始剂量通常为 500mg 每日两次，随餐服用可减轻胃肠道反应（I类推荐）。",
         "中国2型糖尿病防治指南", TIER_GUIDELINE, ["第四章 口服降糖药", "双胍类"], "2020"),
        ("估算肾小球滤过率低于 30 mL/min 的患者禁用二甲双胍，使用含碘对比剂前应暂停给药。",
         "盐酸二甲双胍片说明书", TIER_DRUG_LABEL, ["禁忌", "肾功能不全"], "2021"),
        ("低血糖是降糖治疗最常见的不良反应，表现为心悸、出汗、手抖，血糖低于 3.9mmol/L 即应处理。",
         "内科学", TIER_TEXTBOOK, ["内分泌系统", "糖尿病并发症"], "2018"),
        ("急性ST段抬高型心肌梗死确诊后应在 120 分钟内完成直接 PCI。",
         "急性心肌梗死诊疗规范", TIER_GUIDELINE, ["第二章 再灌注治疗"], "2019"),
        ("网友回复：我吃二甲双胍没啥事，你也可以试试加量。",
         "在线问诊记录", TIER_CONSULTATION, ["用药咨询"], ""),
    ]
    chunks = []
    for idx, (text, title, tier, section, version) in enumerate(raw):
        chunks.append(Chunk(
            chunk_id=stable_id(title, idx, text),
            text=text,
            doc_id=stable_id(title),
            doc_title=title,
            source_tier=tier,
            parent_text=text,
            section_path=section,
            version=version,
            page=idx + 1,
            # 显式写入，别让示例数据依赖 _repair_authority 兜底
            authority=TIER_AUTHORITY.get(tier, 0.3),
        ))
    return chunks


def parse_args():
    parser = argparse.ArgumentParser(description="medical_rag 端到端管线自检 / 离线问答")
    parser.add_argument("--chunks", default="", help="chunk JSONL 路径，留空则用内置示例数据")
    parser.add_argument("--query", default="二甲双胍怎么吃，肾功能不好能用吗", help="提问")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--recall_k", type=int, default=50)
    parser.add_argument("--routes", default="dense,bm25,term")
    parser.add_argument("--dense_model", default="", help="本地 embedding 模型目录")
    parser.add_argument("--reranker_model", default="", help="交叉编码器模型目录")
    parser.add_argument("--generator_model", default="", help="生成模型目录，留空则不生成")
    parser.add_argument("--peft_path", default="", help="LoRA 适配器目录")
    parser.add_argument("--terminology", default="", help="术语表 JSONL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min_score", type=float, default=0.0, help="拒答分数阈值")
    parser.add_argument("--min_hits", type=int, default=1, help="拒答命中数下限")
    parser.add_argument("--no_generate", action="store_true", help="只跑检索与 prompt 组装")
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--dump_json", default="", help="把完整结果写到该 JSON 文件")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")

    if args.chunks and Path(args.chunks).exists():
        rag = MedicalRAG.from_jsonl(
            args.chunks,
            max_rows=args.max_rows,
            dense_model=args.dense_model or None,
            reranker_model=args.reranker_model or None,
            generator_model=args.generator_model or None,
            peft_path=args.peft_path or None,
            terminology_path=args.terminology or None,
            device=args.device,
            routes=tuple(r.strip() for r in args.routes.split(",") if r.strip()),
            recall_k=args.recall_k,
            min_score=args.min_score,
            min_hits=args.min_hits,
        )
    else:
        if args.chunks:
            logger.warning("找不到 %s，改用内置示例数据", args.chunks)
        rag = MedicalRAG.from_chunks(
            _demo_chunks(),
            dense_model=args.dense_model or None,
            reranker_model=args.reranker_model or None,
            generator_model=args.generator_model or None,
            peft_path=args.peft_path or None,
            terminology_path=args.terminology or None,
            device=args.device,
            routes=tuple(r.strip() for r in args.routes.split(",") if r.strip()),
            recall_k=args.recall_k,
            min_score=args.min_score,
            min_hits=args.min_hits,
        )

    print(repr(rag))
    result = rag.answer(args.query, topk=args.topk, generate=not args.no_generate)

    explain = result["retrieval_explain"]
    print("\n查询: %s" % result["query"])
    print("召回路: %s | backend: %s" % (explain["routes"], explain["route_backends"]))
    print("命中 %d 条 | 拒答=%s %s" % (explain["num_hits"], result["refused"],
                                      explain["refuse_reason"]))
    if explain["degraded"]:
        print("降级项: %s" % "; ".join(explain["degraded"]))

    print("\n参考资料:")
    print(rag.format_sources(result) or "  (无)")

    print("\n回答:")
    print(result["answer"] or "  (未生成，当前为检索验证模式)")

    if not result["refused"]:
        print("\nprompt 预览(前 300 字):")
        print(result["prompt"][:300])

    print("\n耗时(ms): %s" % explain["timing_ms"])

    if args.dump_json:
        Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.dump_json).open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(result), handle, ensure_ascii=False, indent=2)
        print("完整结果已写入 %s" % args.dump_json)


if __name__ == "__main__":
    main()
