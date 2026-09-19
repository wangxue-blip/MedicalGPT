# -*- coding: utf-8 -*-
"""医疗 RAG 的检索编排：查询扩展 -> 多路召回 -> RRF 融合 -> 精排 -> 权威度加权。

index / fusion / rerank 三个模块各自只负责一件事，本模块把它们串成一条可复现、
可追溯的检索链路，并回答三个编排层面的问题。

一、为什么要在检索前做查询扩展
-------------------------------------------------
用户问"心梗要做什么检查"，指南原文写的是"急性ST段抬高型心肌梗死"。term 路靠
术语归一能命中，但 bm25 路字面对不上直接 0 分，dense 路对短别名的向量也不稳。
把 query 先用 ``Terminology.expand_query`` 改写成若干别名变体，让**每一路**都
拿着多个说法去召回，等于在不改任何索引的前提下给词面路补上了同义词能力。
代价是每路检索次数乘以变体数，所以变体数默认只取 4（见 DEFAULT_MAX_VARIANTS）。

二、同一路内跨变体为什么按"最好排名"合并，而不是分数取最大
-------------------------------------------------
变体之间的分数不可比：BM25 对"心梗"和对"急性ST段抬高型心肌梗死"给出的分数受
query 长度和 IDF 影响，量纲差一个数量级；dense 路各变体的余弦相似度也会整体
平移。直接取 max 等于让"最长的那个变体"独占结果。排名则是变体内部归一化过的
相对信号——"某个说法把它排到了第 1"这件事，不管用哪个说法问都同样可信。
合并后按最好排名重新排序输出，正好接上 fusion 以列表位置为 rank 的契约。

三、route_weights 为什么 term 路只有 0.6
-------------------------------------------------
术语路精度极高但召回面很窄：query 里一个已登录术语都没有时它直接返回空，命中
时又容易把"提到同一个术语但答非所问"的段落顶上来。给它满权重会让 RRF 里
"术语共现"这一个信号压过语义相关性。0.6 的定位是"加分项"而不是"决定项"。
dense 与 bm25 是互补的主力，保持平权。

可追溯性
-------------------------------------------------
返回的每个 RetrievalHit 都带齐 ``route_ranks``（每一路给的名次，没召回写 None）、
``route_scores``（该路原始分）、``rerank_score``（未精排时为 None）与
``final_score``。``explain()`` 进一步给出每路每个变体的 topN 以及融合前后的
名次变化，用来回答"这条证据是被哪几路、以第几名捞出来的，后来为什么升/降"。

降级行为
-------------------------------------------------
本模块自身只依赖标准库。术语表不可用时退化为不扩展（只用原始 query）；
reranker 为 None 或其 ``available`` 为 False 时跳过精排，由
``apply_authority_boost`` 自动改用融合分作为语义基线；单路索引抛异常只记
warning 并按空结果处理；索引为空或全部召回为空时返回空列表，不抛异常。
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # 作为包导入（import medical_rag.retriever）
    from .schema import (
        Chunk,
        RetrievalHit,
        TIER_CONSULTATION,
        TIER_GUIDELINE,
        TIER_TEXTBOOK,
        load_chunks,
        normalize_text,
        stable_id,
    )
    from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
    from .index import BaseIndex, build_indexes
    from .rerank import apply_authority_boost
except ImportError:  # pragma: no cover - `python medical_rag/retriever.py` 直接运行走这里
    import sys as _sys

    # 直接跑脚本时 sys.path[0] 是 medical_rag/ 本身，仓库根目录不在路径上。
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from medical_rag.schema import (  # type: ignore
        Chunk,
        RetrievalHit,
        TIER_CONSULTATION,
        TIER_GUIDELINE,
        TIER_TEXTBOOK,
        load_chunks,
        normalize_text,
        stable_id,
    )
    from medical_rag.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion  # type: ignore
    from medical_rag.index import BaseIndex, build_indexes  # type: ignore
    from medical_rag.rerank import apply_authority_boost  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "MedicalRetriever",
    "DEFAULT_ROUTE_WEIGHTS",
    "format_explain",
]

# term 路是加分项不是决定项，详见模块 docstring 第三节。
DEFAULT_ROUTE_WEIGHTS = {"dense": 1.0, "bm25": 1.0, "term": 0.6}

DEFAULT_RECALL_K = 50
DEFAULT_AUTHORITY_WEIGHT = 0.15
DEFAULT_RECENCY_WEIGHT = 0.05

# 变体数上限。每多一个变体，每一路就多跑一次检索；实测 4 个变体已能覆盖
# "全规范化 + 两三个主要别名"，再多的收益迅速趋近于 0。
DEFAULT_MAX_VARIANTS = 4

# explain / 命令行预览里每段正文截断长度，避免刷屏。
PREVIEW_CHARS = 48
EXPLAIN_TOP_N = 5


class _IdentityTerminology(object):
    """术语表不可用时的兜底：不扩展、不归一化。

    刻意不做任何猜测式改写——错误的同义词会把无关段落召回上来，在医疗场景里
    比少召回危险得多。少一路扩展，检索质量下降；乱扩展，答案可能出错。
    """

    def normalize(self, term: str) -> str:
        return term

    def expand_query(self, query: str, max_expansions: int = 8) -> List[str]:
        return [query]

    def terms_in(self, text: str) -> List[str]:
        return []


def _default_terminology():
    """加载内置术语表，失败则退化为恒等术语表。"""
    try:
        try:
            from .terminology import Terminology  # type: ignore
        except ImportError:
            from medical_rag.terminology import Terminology  # type: ignore
    except Exception as exc:  # 模块缺失或其自身导入期报错都算不可用
        logger.warning("medical_rag.terminology 不可用，查询扩展已关闭: %s", exc)
        return _IdentityTerminology()
    try:
        return Terminology.default()
    except Exception as exc:
        logger.warning("Terminology.default() 失败，查询扩展已关闭: %s", exc)
        return _IdentityTerminology()


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = (text or "").replace("\n", " ").replace("\r", " ")
    return text[:limit]


class MedicalRetriever:
    """把三路索引、融合、精排、权威度加权串成一次可追溯的检索。

    ``indexes`` 必须是在**同一份** ``chunks`` 上构建的（fusion 用下标回查 chunk），
    构造时会校验规模并对不一致的路发 warning。
    """

    def __init__(
        self,
        chunks: List[Chunk],
        indexes: Dict[str, BaseIndex],
        reranker=None,
        terminology=None,
        recall_k: int = DEFAULT_RECALL_K,
        fusion_k: int = DEFAULT_RRF_K,
        route_weights: Optional[Dict[str, float]] = None,
        authority_weight: float = DEFAULT_AUTHORITY_WEIGHT,
    ):
        self.chunks = list(chunks or [])
        self.indexes = dict(indexes or {})
        self.reranker = reranker
        self.terminology = terminology if terminology is not None else _default_terminology()

        self.recall_k = max(1, _safe_int(recall_k, DEFAULT_RECALL_K))
        # fusion_k 为负会让 RRF 除零，这里提前夹到 0；fusion 内部也有一层兜底。
        self.fusion_k = max(0, _safe_int(fusion_k, DEFAULT_RRF_K))

        # 以默认权重打底再覆盖：调用方只想调 term 一路时不必把三路都写全。
        self.route_weights = dict(DEFAULT_ROUTE_WEIGHTS)
        if route_weights:
            for route, weight in route_weights.items():
                # 传进来的权重不是数值时退回该路的默认值（而不是统一的 1.0），
                # 否则一个笔误会悄悄把 term 路从 0.6 提到 1.0，排序结果莫名其妙地变。
                fallback = DEFAULT_ROUTE_WEIGHTS.get(route, 1.0)
                self.route_weights[route] = _safe_float(weight, fallback)
        self.authority_weight = _safe_float(authority_weight, DEFAULT_AUTHORITY_WEIGHT)

        # 下面三个是调参旋钮。全系统 API 契约固定了 __init__ 签名，所以不进参数表，
        # 需要调时直接改属性（retriever.max_variants = 1 即关闭查询扩展）。
        self.recency_weight = DEFAULT_RECENCY_WEIGHT
        self.max_variants = DEFAULT_MAX_VARIANTS
        # 交叉编码器每个候选都要跑一次前向，候选池必须有上界；与召回规模对齐。
        self.rerank_candidates = self.recall_k

        self._check_alignment()

    # -- 构造期校验 ---------------------------------------------------------
    def _check_alignment(self) -> None:
        """索引规模与 chunk 数不一致时告警。

        fusion 靠"下标 -> chunks[i]"回查证据，一旦索引是在另一份语料上建的，
        检索还能跑，但引用会张冠李戴——这是医疗场景最不能接受的错误类型，
        所以宁可吵一点也要在构造期就喊出来。
        """
        if not self.chunks:
            logger.warning("MedicalRetriever 收到空语料，retrieve 将始终返回空列表")
            return
        if not self.indexes:
            logger.warning("MedicalRetriever 没有可用索引，retrieve 将始终返回空列表")
            return
        for route, index in self.indexes.items():
            size = _safe_int(getattr(index, "size", len(self.chunks)), len(self.chunks))
            if size != len(self.chunks):
                logger.warning(
                    "路由 %s 的索引规模 %d 与语料 %d 不一致，引用可能错位，请确认索引与 chunks 同源",
                    route, size, len(self.chunks),
                )

    # -- 查询扩展 -----------------------------------------------------------
    def expand(self, query: str) -> List[str]:
        """生成查询变体，首个恒为规整后的原始 query。"""
        query = normalize_text(query)
        if not query:
            return []
        variants = [query]
        limit = max(1, _safe_int(self.max_variants, DEFAULT_MAX_VARIANTS))
        if limit <= 1 or self.terminology is None:
            return variants
        try:
            raw = self.terminology.expand_query(query, max_expansions=limit) or []
        except Exception as exc:
            # 术语表出问题不该让检索停摆，退化成"只用原始 query"即可。
            logger.warning("查询扩展失败，本次仅使用原始 query: %s", exc)
            return variants
        for item in raw:
            item = normalize_text(item)
            if item and item not in variants:
                variants.append(item)
        return variants[:limit]

    # -- 单路召回 -----------------------------------------------------------
    def _recall_route(
        self,
        route: str,
        index: BaseIndex,
        variants: Sequence[str],
    ) -> Tuple[List[Tuple[int, float]], List[Dict[str, Any]]]:
        """在一路索引上跑完所有变体，按最好排名合并。

        返回 (合并后的 [(chunk 下标, 分数)], 每个变体的明细)。合并结果的**列表
        顺序**就是交给 fusion 的 rank，所以这里必须输出成已排序的形态。
        """
        best: Dict[int, Tuple[int, float, str]] = {}  # idx -> (最好排名, 该次分数, 变体)
        variant_traces: List[Dict[str, Any]] = []
        total = len(self.chunks)

        for variant in variants:
            try:
                raw_hits = index.search(variant, self.recall_k) or []
            except Exception as exc:
                # 三路挂一路仍然能检索（RRF 本就容忍缺路），不能让整次问答失败。
                logger.warning("路由 %s 检索变体 %r 失败，按空结果处理: %s", route, variant, exc)
                variant_traces.append({"query": variant, "hits": [], "error": str(exc)})
                continue

            pairs: List[Tuple[int, float]] = []
            for rank, item in enumerate(_iter_pairs(route, raw_hits), start=1):
                idx, score = item
                if idx < 0 or idx >= total:
                    logger.warning("路由 %s 返回越界下标 %d（语料共 %d 条），已跳过", route, idx, total)
                    continue
                pairs.append((idx, score))
                prev = best.get(idx)
                # 名次相同时取分数更高的那次，保证结果与变体遍历顺序无关（可复现）。
                if prev is None or (rank, -score) < (prev[0], -prev[1]):
                    best[idx] = (rank, score, variant)
            variant_traces.append({"query": variant, "hits": pairs})

        order = sorted(best.keys(), key=lambda i: (best[i][0], -best[i][1], i))
        merged = [(i, best[i][1]) for i in order[:self.recall_k]]
        return merged, variant_traces

    # -- 精排 ---------------------------------------------------------------
    def _rerank(self, query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        """交叉编码器精排；不可用或出错时原样返回融合顺序。"""
        if not hits or self.reranker is None:
            return hits
        try:
            available = bool(getattr(self.reranker, "available", True))
        except Exception as exc:
            logger.warning("探测 reranker.available 失败，已跳过精排: %s", exc)
            return hits
        if not available:
            return hits
        try:
            reranked = self.reranker.rerank(query, hits)
        except Exception as exc:
            logger.warning("精排失败，保持融合顺序: %s", exc)
            return hits
        # 数量对不上说明 reranker 实现有问题，此时它写回的 rerank_score 也不可信，
        # 宁可退回融合顺序，也不要让证据与分数错位。
        if not reranked or len(reranked) != len(hits):
            logger.warning("精排返回 %d 条，与候选 %d 条不符，已忽略", len(reranked or []), len(hits))
            return hits
        return list(reranked)

    # -- 主流程 -------------------------------------------------------------
    def _run(self, query: str, topk: int, collect_trace: bool = False):
        """检索主流程，retrieve 与 explain 共用，避免两条实现走偏。"""
        started = time.time()
        topk = max(0, _safe_int(topk, 0))
        trace: Dict[str, Any] = {
            "query": query if isinstance(query, str) else str(query or ""),
            "variants": [],
            "routes": {},
            "fused": [],
            "final": [],
            "route_weights": dict(self.route_weights),
            "recall_k": self.recall_k,
            "fusion_k": self.fusion_k,
            "topk": topk,
            "reranked": False,
            "n_chunks": len(self.chunks),
            "elapsed_ms": 0.0,
            "note": "",
        }

        def _finish(hits: List[RetrievalHit], note: str = "") -> Tuple[List[RetrievalHit], Dict[str, Any]]:
            if note:
                trace["note"] = note
            trace["elapsed_ms"] = round((time.time() - started) * 1000.0, 2)
            return hits, trace

        if topk <= 0:
            return _finish([], "topk<=0")
        if not self.chunks:
            return _finish([], "语料为空")
        if not self.indexes:
            return _finish([], "没有可用索引")

        variants = self.expand(query)
        trace["variants"] = list(variants)
        if not variants:
            return _finish([], "query 为空")

        route_hits: Dict[str, List[Tuple[int, float]]] = {}
        for route, index in self.indexes.items():
            merged, variant_traces = self._recall_route(route, index, variants)
            # 即使为空也保留这个 key：fusion 用 route_hits 的键集合决定 route_ranks
            # 覆盖哪几路，少一个键会让"这一路没召回"和"根本没这一路"混为一谈。
            route_hits[route] = merged
            if collect_trace:
                trace["routes"][route] = {
                    "backend": getattr(index, "backend", "unknown"),
                    "size": _safe_int(getattr(index, "size", 0), 0),
                    "weight": self.route_weights.get(route, 1.0),
                    "recalled": len(merged),
                    "variants": [
                        {
                            "query": item["query"],
                            "error": item.get("error", ""),
                            "top": [
                                {
                                    "rank": rank,
                                    "chunk_id": self.chunks[idx].chunk_id,
                                    "score": round(float(score), 6),
                                    "preview": _preview(self.chunks[idx].text),
                                }
                                for rank, (idx, score) in enumerate(
                                    item["hits"][:max(topk, EXPLAIN_TOP_N)], start=1
                                )
                            ],
                        }
                        for item in variant_traces
                    ],
                }

        if not any(route_hits.values()):
            logger.info("query=%r 在全部 %d 路索引上均无召回", query, len(route_hits))
            return _finish([], "全部路由均无召回")

        fused = reciprocal_rank_fusion(
            route_hits, self.chunks, k=self.fusion_k, weights=self.route_weights
        )
        if not fused:
            return _finish([], "融合后为空")

        # 用对象身份而非 chunk_id 记录融合名次：rerank / boost 都是原地改字段再
        # 重排同一批对象，chunk_id 在语料里重复时会串行，id() 不会。
        fusion_rank = {id(hit): rank for rank, hit in enumerate(fused, start=1)}
        if collect_trace:
            trace["fused"] = [
                {
                    "rank": rank,
                    "chunk_id": hit.chunk.chunk_id,
                    "rrf_score": round(float(hit.score), 6),
                    "route_ranks": dict(hit.route_ranks),
                    "route_scores": {k: round(float(v), 6) for k, v in hit.route_scores.items()},
                    "preview": _preview(hit.chunk.text),
                }
                for rank, hit in enumerate(fused[:max(topk, EXPLAIN_TOP_N)], start=1)
            ]

        # 候选池必须有上界：交叉编码器是按候选数线性耗时的。多路取并集后 fused
        # 可能远长于 recall_k，截断掉尾部不影响 topk（topk <= 候选池）。
        pool_size = max(topk, self.rerank_candidates)
        candidates = fused[:pool_size]
        reranked = self._rerank(normalize_text(query), candidates)
        trace["reranked"] = any(hit.rerank_score is not None for hit in reranked)

        ranked = apply_authority_boost(
            reranked,
            authority_weight=self.authority_weight,
            recency_weight=self.recency_weight,
        )
        result = ranked[:topk]

        if collect_trace:
            trace["final"] = [
                {
                    "rank": rank,
                    "chunk_id": hit.chunk.chunk_id,
                    "fusion_rank": fusion_rank.get(id(hit)),
                    # 正数表示精排/权威度把它往前提了，负数表示压下去了。
                    "rank_delta": _rank_delta(fusion_rank.get(id(hit)), rank),
                    "route_ranks": dict(hit.route_ranks),
                    "rrf_score": round(float(hit.score), 6),
                    "rerank_score": (
                        None if hit.rerank_score is None else round(float(hit.rerank_score), 6)
                    ),
                    "final_score": round(float(hit.final_score), 6),
                    "source_tier": hit.chunk.source_tier,
                    "authority": round(float(getattr(hit.chunk, "authority", 0.0) or 0.0), 4),
                    "citation": hit.chunk.citation_label(),
                    "preview": _preview(hit.chunk.text),
                }
                for rank, hit in enumerate(result, start=1)
            ]
        return _finish(result)

    # -- 对外接口 -----------------------------------------------------------
    def retrieve(self, query: str, topk: int = 5) -> List[RetrievalHit]:
        """检索 topk 条证据，按 final_score 降序。

        返回的每个 hit 都带齐 route_ranks / route_scores / final_score；
        rerank_score 仅在精排真正跑起来时非 None（None 表示"未精排"，不是 0 分）。
        任何一步失败都只降级不抛异常，最坏情况返回空列表。
        """
        hits, _ = self._run(query, topk, collect_trace=False)
        return hits

    def explain(self, query: str, topk: int = 5) -> Dict[str, Any]:
        """返回可打印的检索过程明细，供调参与排查使用。

        结构为纯 JSON 可序列化对象：variants / routes（每路每个变体的 topN）/
        fused（融合后名次与各路名次）/ final（最终名次及相对融合名次的变化）。
        配合 ``format_explain`` 可直接打印成文本。
        """
        _, trace = self._run(query, topk, collect_trace=True)
        return trace


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("参数 %r 不是整数，按 %d 处理", value, default)
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("参数 %r 不是数值，按 %s 处理", value, default)
        return default


def _iter_pairs(route: str, items: Sequence[Any]):
    """规整某一路的召回结果，跳过结构非法的元素。

    索引后端可能是第三方（faiss）直接吐出来的 numpy 标量，这里统一转成
    (int, float)，避免脏数据顺着链路一路传到引用里。
    """
    for item in items or []:
        if item is None:
            continue
        try:
            yield int(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            logger.warning("路由 %s 返回结构非法的召回项 %r，已跳过", route, item)
            continue


def _rank_delta(fusion_rank: Optional[int], final_rank: int) -> Optional[int]:
    if fusion_rank is None:
        return None
    return fusion_rank - final_rank


def format_explain(detail: Dict[str, Any], top_n: int = EXPLAIN_TOP_N) -> str:
    """把 explain() 的结果渲染成多行文本。"""
    if not detail:
        return "(无检索明细)"
    lines: List[str] = []
    lines.append("query: %s" % detail.get("query", ""))
    lines.append("变体: %s" % " | ".join(detail.get("variants") or ["(无)"]))
    lines.append(
        "语料=%s recall_k=%s fusion_k=%s topk=%s 精排=%s 耗时=%sms"
        % (
            detail.get("n_chunks"),
            detail.get("recall_k"),
            detail.get("fusion_k"),
            detail.get("topk"),
            detail.get("reranked"),
            detail.get("elapsed_ms"),
        )
    )
    if detail.get("note"):
        lines.append("说明: %s" % detail["note"])

    for route, info in (detail.get("routes") or {}).items():
        lines.append(
            "\n[%s] backend=%s size=%s weight=%s 合并后召回=%s"
            % (route, info.get("backend"), info.get("size"), info.get("weight"), info.get("recalled"))
        )
        for variant in info.get("variants") or []:
            if variant.get("error"):
                lines.append("  <%s> 失败: %s" % (variant.get("query"), variant["error"]))
                continue
            lines.append("  <%s>" % variant.get("query"))
            if not variant.get("top"):
                lines.append("    (无结果)")
            for row in (variant.get("top") or [])[:top_n]:
                lines.append(
                    "    %d. %.4f %s | %s"
                    % (row["rank"], row["score"], row["chunk_id"], row["preview"])
                )

    if detail.get("fused"):
        lines.append("\n[融合 RRF]")
        for row in detail["fused"][:top_n]:
            lines.append(
                "  %d. rrf=%.6f %s ranks=%s | %s"
                % (row["rank"], row["rrf_score"], row["chunk_id"], row["route_ranks"], row["preview"])
            )

    if detail.get("final"):
        lines.append("\n[最终排序：融合名次 -> 最终名次]")
        for row in detail["final"]:
            delta = row.get("rank_delta")
            if delta is None:
                move = "?"
            elif delta > 0:
                move = "+%d" % delta
            else:
                move = str(delta)
            lines.append(
                "  %d. (融合第 %s 名, %s) final=%.4f rerank=%s tier=%s"
                % (
                    row["rank"],
                    row.get("fusion_rank"),
                    move,
                    row["final_score"],
                    row.get("rerank_score"),
                    row.get("source_tier"),
                )
            )
            lines.append("     %s" % row.get("citation", ""))
            lines.append("     %s" % row.get("preview", ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def _demo_chunks() -> List[Chunk]:
    """构造带真实医疗特征的小语料：别名、英文药名、剂量、不同来源等级。"""
    raw = [
        ("二甲双胍是2型糖尿病一线降糖药物，通常起始剂量 500mg 每日两次，随餐服用可减轻胃肠道反应。",
         "中国2型糖尿病防治指南", TIER_GUIDELINE, "2020年版", ["第四章 口服降糖药", "双胍类"]),
        ("二甲双胍禁用于 eGFR 低于 30 ml/min 的患者，使用碘造影剂前应暂停。",
         "二甲双胍片说明书", TIER_TEXTBOOK, "2019年版", ["禁忌", "肾功能不全"]),
        ("急性ST段抬高型心肌梗死确诊后应在 120 分钟内完成直接 PCI（I 类推荐）。",
         "急性心肌梗死诊疗规范", TIER_GUIDELINE, "2019年版", ["第二章 再灌注治疗"]),
        ("心梗发作时首先要做心电图和心肌酶检查，怀疑心梗要立即拨打 120。",
         "在线问诊记录", TIER_CONSULTATION, "", []),
        ("高血压患者初始降压目标一般为收缩压低于 140mmHg，合并糖尿病者可考虑 130mmHg。",
         "中国高血压防治指南", TIER_GUIDELINE, "2018年版", ["第五章 降压目标"]),
        ("低血糖是降糖治疗最常见的不良反应，血糖低于 3.9mmol/L 即应处理。",
         "内科学", TIER_TEXTBOOK, "第9版", ["内分泌系统", "糖尿病并发症"]),
    ]
    chunks = []
    for idx, (text, title, tier, version, section) in enumerate(raw):
        chunks.append(
            Chunk(
                chunk_id=stable_id(title, idx, text),
                text=text,
                doc_id=stable_id(title),
                doc_title=title,
                source_tier=tier,
                parent_text=text,
                version=version,
                section_path=section,
                page=idx + 1,
            )
        )
    return chunks


def _parse_weights(raw: str) -> Optional[Dict[str, float]]:
    """解析 --weights dense=1.0,bm25=1.0,term=0.6。"""
    if not raw:
        return None
    weights: Dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        route, _, value = item.partition("=")
        route = route.strip()
        if not route:
            continue
        weights[route] = _safe_float(value.strip(), 1.0)
    return weights or None


def parse_args():
    parser = argparse.ArgumentParser(description="medical_rag 检索编排自检 / 离线检索")
    parser.add_argument("--chunks", default="", help="chunk JSONL 路径，留空则用内置示例数据")
    parser.add_argument("--query", default="心梗要做什么检查", help="检索查询")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--recall_k", type=int, default=DEFAULT_RECALL_K)
    parser.add_argument("--fusion_k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--routes", default="dense,bm25,term")
    parser.add_argument("--weights", default="", help="路由权重，形如 dense=1.0,bm25=1.0,term=0.6")
    parser.add_argument("--dense_model", default="", help="本地 embedding 模型目录，留空则自动探测")
    parser.add_argument("--authority_weight", type=float, default=DEFAULT_AUTHORITY_WEIGHT)
    parser.add_argument("--max_variants", type=int, default=DEFAULT_MAX_VARIANTS)
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--explain", action="store_true", help="打印每路召回与名次变化明细")
    parser.add_argument("--use_reranker", action="store_true",
                        help="启用交叉编码器精排（可能触发模型下载）")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.chunks and os.path.exists(args.chunks):
        chunks = load_chunks(args.chunks, max_rows=args.max_rows)
    else:
        if args.chunks:
            logger.warning("找不到 %s，改用内置示例数据", args.chunks)
        chunks = _demo_chunks()

    routes = tuple(r.strip() for r in args.routes.split(",") if r.strip())
    terminology = _default_terminology()
    indexes = build_indexes(
        chunks,
        dense_model=args.dense_model or None,
        terminology=terminology,
        routes=routes,
    )

    reranker = None
    if args.use_reranker:
        try:
            try:
                from .rerank import CrossEncoderReranker  # type: ignore
            except ImportError:
                from medical_rag.rerank import CrossEncoderReranker  # type: ignore
            reranker = CrossEncoderReranker()
        except Exception as exc:
            logger.warning("构造 reranker 失败，本次不精排: %s", exc)

    retriever = MedicalRetriever(
        chunks,
        indexes,
        reranker=reranker,
        terminology=terminology,
        recall_k=args.recall_k,
        fusion_k=args.fusion_k,
        route_weights=_parse_weights(args.weights),
        authority_weight=args.authority_weight,
    )
    retriever.max_variants = args.max_variants

    print("语料 %d 条 | 索引: %s" % (len(chunks), ", ".join(indexes.keys()) or "(无)"))
    if args.explain:
        print(format_explain(retriever.explain(args.query, topk=args.topk)))
        return

    hits = retriever.retrieve(args.query, topk=args.topk)
    print("查询: %s" % args.query)
    if not hits:
        print("  (无结果)")
        return
    for rank, hit in enumerate(hits, start=1):
        print("%d. final=%.4f rrf=%.6f rerank=%s ranks=%s"
              % (rank, hit.final_score, hit.score, hit.rerank_score, hit.route_ranks))
        print("   %s" % hit.chunk.citation_label())
        print("   %s" % _preview(hit.chunk.text, 80))


if __name__ == "__main__":
    main()
