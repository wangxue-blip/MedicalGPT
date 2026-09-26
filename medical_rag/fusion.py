# -*- coding: utf-8 -*-
"""多路召回结果的融合（Reciprocal Rank Fusion）。

为什么用 RRF 而不是把各路分数加权求和？
-------------------------------------------------
三路召回的分数处在完全不同的量纲上：

* dense 路是归一化向量的余弦相似度，天然落在 0~1（实际常年挤在 0.3~0.8）；
* bm25 路是 IDF 加权的词频得分，**没有上界**，长 query 命中多个高 IDF 术语时
  轻松到 20 以上；
* term 路是术语命中数折算的分数，量纲又是另一套。

如果直接 ``0.5 * dense + 0.3 * bm25 + 0.2 * term``，BM25 那一路的数值尺度会
淹没另外两路——名义上的权重 0.3 实际等于让 BM25 独裁，而且这个尺度还随语料
规模、query 长度漂移，今天调好的权重明天换个语料就失效。要修就得先做分数
归一化，而 min-max 归一化又强依赖候选集合本身（同一个 chunk 在不同 query 下
归一化结果不同），非常脆弱。

RRF 绕开了整个问题：它**只看排名，不看分数**。每一路给出的贡献是
``weight / (k + rank)``，rank 从 1 开始。于是：

1. 免调参、免归一化，任何打分函数都能直接接进来；
2. 对离群分数天然鲁棒（BM25 给出 200 分还是 20 分，只要排第一就都是 1/(k+1)）；
3. 被多路同时召回的 chunk 会累加多份贡献，自动体现"共识"——这正是混合检索
   想要的信号。

k 的作用是压平头部差距（k=60 是 Cormack 等人论文里的经验值）：k 越大，排名
第 1 和第 5 的差距越小，越倾向于奖励"多路都召回"而不是"单路排第一"。医疗
场景更怕漏（单路的偶然高分往往是噪声），所以保持默认 60。

``weighted_score_fusion`` 作为对照实现保留，供消融实验时定量说明"RRF 比加权
求和好多少"，不建议在线上使用。

本模块只依赖标准库，在裸环境下必定可导入。
"""

import argparse
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from medical_rag.schema import Chunk, RetrievalHit

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


def _route_weight(weights: Optional[Dict[str, float]], route: str) -> float:
    """取某一路的权重，缺省为 1.0。

    RRF 的权重不是"分数权重"而是"这一路的话语权"，所以缺省给 1.0（各路平权）
    比缺省给 0 更安全：调用方少配一路不会把那一路直接废掉。
    """
    if not weights:
        return 1.0
    try:
        return float(weights.get(route, 1.0))
    except (TypeError, ValueError):
        logger.warning("路由 %s 的权重不是数值，按 1.0 处理", route)
        return 1.0


def _iter_route_items(items: Optional[Sequence[Tuple[int, float]]]):
    """规整单路召回结果，跳过结构非法的元素。

    索引类模块可能是第三方后端（faiss）直接吐出来的，这里宽松处理，避免一条
    脏数据让整个检索链路崩掉。
    """
    for item in items or []:
        if item is None:
            continue
        try:
            idx = int(item[0])
            score = float(item[1])
        except (TypeError, ValueError, IndexError):
            logger.warning("忽略结构非法的召回项: %r", item)
            continue
        yield idx, score


def _collect(
    route_hits: Dict[str, List[Tuple[int, float]]],
    chunks: List[Chunk],
) -> Tuple[List[str], Dict[int, Dict[str, int]], Dict[int, Dict[str, float]]]:
    """把各路结果摊平成 {chunk 下标: {路由: rank}} 与 {chunk 下标: {路由: 原始分}}。

    rank 取元素在该路结果列表里的位置（1-based），因为 ``BaseIndex.search``
    的契约就是返回已排好序的 topk；这里不重新排序，以免破坏索引侧自己的
    打分与 tie-break 逻辑。
    """
    route_names = list(route_hits.keys()) if route_hits else []
    ranks: Dict[int, Dict[str, int]] = {}
    scores: Dict[int, Dict[str, float]] = {}
    total = len(chunks)

    for route in route_names:
        seen = set()
        for rank, (idx, score) in enumerate(_iter_route_items(route_hits.get(route)), start=1):
            if idx < 0 or idx >= total:
                logger.warning("路由 %s 返回越界下标 %d（chunks 共 %d 条），已跳过", route, idx, total)
                continue
            if idx in seen:
                # 同一路重复召回同一个 chunk，只认最靠前的那次，否则会被重复加分。
                continue
            seen.add(idx)
            ranks.setdefault(idx, {})[route] = rank
            scores.setdefault(idx, {})[route] = score
    return route_names, ranks, scores


def _make_hit(
    idx: int,
    chunks: List[Chunk],
    route_names: Sequence[str],
    ranks: Dict[int, Dict[str, int]],
    scores: Dict[int, Dict[str, float]],
    fused_score: float,
) -> RetrievalHit:
    """构造 RetrievalHit，并把"哪一路没召回"显式写成 None。

    route_ranks 必须覆盖全部路由（没召回写 None），这样 debug 时一眼能看出
    某条结果是三路共识还是单路孤证；route_scores 只放真正召回的那几路的原始
    分数，因为 RetrievalHit.to_dict() 会对它做 round()，None 会直接抛异常。
    """
    hit_ranks = ranks.get(idx, {})
    return RetrievalHit(
        chunk=chunks[idx],
        score=float(fused_score),
        route_ranks={route: hit_ranks.get(route) for route in route_names},
        route_scores=dict(scores.get(idx, {})),
    )


def reciprocal_rank_fusion(
    route_hits: Dict[str, List[Tuple[int, float]]],
    chunks: List[Chunk],
    k: int = DEFAULT_RRF_K,
    weights: Optional[Dict[str, float]] = None,
) -> List[RetrievalHit]:
    """用 RRF 融合多路召回结果。

    Args:
        route_hits: {路由名: [(chunk 下标, 该路分数), ...]}，每路按分数降序。
        chunks: 与下标对应的 chunk 全集。
        k: RRF 平滑常数，越大越奖励"多路共识"。
        weights: {路由名: 权重}，缺省各路平权。

    Returns:
        按 RRF 分降序排列的 RetrievalHit 列表；score 即 RRF 分，
        route_ranks / route_scores 记录每一路的贡献。
    """
    if not route_hits or not chunks:
        return []

    if k < 0:
        logger.warning("RRF 的 k=%s 为负会导致除零，已按 0 处理", k)
        k = 0
    k = int(k)

    route_names, ranks, scores = _collect(route_hits, chunks)

    totals: Dict[int, float] = {}
    for idx, per_route in ranks.items():
        total = 0.0
        for route, rank in per_route.items():
            total += _route_weight(weights, route) / float(k + rank)
        totals[idx] = total

    # 同分时按 chunk 下标升序，保证同一份语料多次检索的结果完全可复现。
    order = sorted(totals.keys(), key=lambda i: (-totals[i], i))
    return [_make_hit(i, chunks, route_names, ranks, scores, totals[i]) for i in order]


def _minmax(values: Sequence[float]) -> Dict[int, float]:
    """按位置返回 min-max 归一化结果；全相等时统一给 1.0。

    全相等说明这一路没有区分度，给 1.0（而不是 0）可以保留"它确实召回了"
    这个信息，不至于让该路的贡献凭空消失。
    """
    if not values:
        return {}
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        return {i: 1.0 for i in range(len(values))}
    return {i: (v - lo) / span for i, v in enumerate(values)}


def weighted_score_fusion(
    route_hits: Dict[str, List[Tuple[int, float]]],
    chunks: List[Chunk],
    weights: Optional[Dict[str, float]] = None,
    normalize: bool = True,
) -> List[RetrievalHit]:
    """加权分数融合，仅作为 RRF 的消融对照。

    ``normalize=True`` 时先在每一路内部做 min-max 归一化再加权（这是加权求和
    能勉强工作的前提）；``normalize=False`` 时直接对原始分加权，用来在实验里
    复现"BM25 尺度碾压 dense"的失败现象。

    注意归一化是在**当前候选集合内**做的，同一个 chunk 换个 query 就会得到
    不同的归一化分，这正是我们线上不用它的原因。
    """
    if not route_hits or not chunks:
        return []

    route_names, ranks, scores = _collect(route_hits, chunks)

    totals: Dict[int, float] = {}
    for route in route_names:
        members = [(idx, per_route[route]) for idx, per_route in scores.items() if route in per_route]
        if not members:
            continue
        weight = _route_weight(weights, route)
        if normalize:
            normed = _minmax([score for _, score in members])
            contributions = [(idx, normed[pos]) for pos, (idx, _) in enumerate(members)]
        else:
            contributions = members
        for idx, value in contributions:
            totals[idx] = totals.get(idx, 0.0) + weight * value

    order = sorted(totals.keys(), key=lambda i: (-totals[i], i))
    return [_make_hit(i, chunks, route_names, ranks, scores, totals[i]) for i in order]


def _demo_chunks() -> List[Chunk]:
    """构造几条最小可用 chunk，供 --demo 自检使用。"""
    from medical_rag.schema import TIER_CONSULTATION, TIER_GUIDELINE

    raw = [
        ("二甲双胍是 2 型糖尿病一线用药", TIER_GUIDELINE, "2023版"),
        ("二甲双胍常见不良反应为胃肠道反应", TIER_GUIDELINE, "2023版"),
        ("我血糖高能吃二甲双胍吗", TIER_CONSULTATION, ""),
    ]
    chunks = []
    for i, (text, tier, version) in enumerate(raw):
        chunks.append(
            Chunk(
                chunk_id="demo-%d" % i,
                text=text,
                doc_id="demo",
                doc_title="演示语料",
                source_tier=tier,
                version=version,
            )
        )
    return chunks


def parse_args():
    parser = argparse.ArgumentParser(description="多路召回融合（RRF / 加权求和对照）自检。")
    parser.add_argument("--k", type=int, default=DEFAULT_RRF_K, help="RRF 平滑常数")
    parser.add_argument("--mode", choices=["rrf", "weighted", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    chunks = _demo_chunks()
    # dense 分在 0~1，bm25 无上界，刻意制造尺度差异来对比两种融合方式。
    route_hits = {
        "dense": [(0, 0.81), (2, 0.76), (1, 0.55)],
        "bm25": [(2, 24.7), (0, 9.1)],
        "term": [(0, 2.0), (1, 1.0)],
    }
    if args.mode in ("rrf", "both"):
        print("[rrf]")
        for hit in reciprocal_rank_fusion(route_hits, chunks, k=args.k):
            print("  %.6f %s %s" % (hit.score, hit.chunk.chunk_id, hit.route_ranks))
    if args.mode in ("weighted", "both"):
        print("[weighted-raw] 原始分加权，观察 BM25 主导现象")
        for hit in weighted_score_fusion(route_hits, chunks, normalize=False):
            print("  %.6f %s %s" % (hit.score, hit.chunk.chunk_id, hit.route_scores))


if __name__ == "__main__":
    main()
