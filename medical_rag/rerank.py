# -*- coding: utf-8 -*-
"""重排序：交叉编码器精排 + 权威度/时效性加权。

为什么召回之后还要重排？
-------------------------------------------------
召回阶段的三路（dense / bm25 / term）都是"双塔"或词面匹配，query 与 chunk
各自独立编码，交互只发生在最后一次点积上，因此只能保证"大致相关"。医疗问答
里最典型的翻车是：query 问"二甲双胍**禁忌证**"，召回回来一堆讲"适应证"的段落
——词面高度重合、向量也很近，但语义正好相反。交叉编码器把 (query, chunk) 拼成
一条序列送进模型做全注意力交互，这类否定/限定关系才能被识别出来。

代价是算力：交叉编码器要对每个候选跑一次前向，所以只能放在召回之后对 topk
(recall_k≈50) 做精排，不能直接拿来检索全库。

为什么精排之后还要再加一层权威度加权？
-------------------------------------------------
交叉编码器只回答"这段文字回答了这个问题吗"，不回答"这段文字值不值得信"。
当语义相关度接近时（差距落在模型噪声范围内），医疗场景必须让来源可信度来
决定顺序：

    指南 > 药品说明书 > 教科书/百科 > 在线问诊

并且新版指南优先于旧版——引用一份 2015 年已被推翻的推荐，比不引用更危险。
``apply_authority_boost`` 就是把这条业务规则显式写进分数里：先把语义分归一化
到 0~1，再叠加一个小幅度的权威度/时效性偏置。偏置权重默认很小（0.15 / 0.05），
是"同分时的裁判"而不是"主排序依据"——否则一份跑题的指南会压过一段精准命中
的说明书。

降级行为
-------------------------------------------------
sentence_transformers 不可用（或模型加载失败）时 ``available`` 为 False，
``rerank()`` 原样返回输入且不抛异常；此时 ``apply_authority_boost`` 会自动改用
融合分作为语义基线，整条链路照常工作，只是精度下降。
"""

import argparse
import logging
import math
import re
from typing import Dict, List, Optional

from medical_rag.schema import RetrievalHit

logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"

# 交叉编码器的输入长度有限，过长的 chunk 截断后再打分，避免显存/耗时失控。
MAX_RERANK_CHARS = 1024

# 时效性映射区间：2010 年及更早记 0，2030 年及更晚记 1。
RECENCY_BASE_YEAR = 2010
RECENCY_SPAN_YEARS = 20.0
DEFAULT_RECENCY = 0.5

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


class CrossEncoderReranker:
    """bge-reranker-v2-m3 交叉编码器精排，依赖缺失时静默降级。"""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RERANKER,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self.model_name_or_path = model_name_or_path or DEFAULT_RERANKER
        self.device = device or "cpu"
        self.batch_size = max(1, int(batch_size or 1))
        self._model = None
        self._load_attempted = False

    # -- 后端加载 ----------------------------------------------------------
    def _ensure_model(self) -> None:
        """惰性加载模型，且只尝试一次。

        放在 __init__ 里加载会让"构造一个 pipeline"变成一次几 GB 的模型下载，
        所以推迟到真正要用（或调用方询问 available）时再做；失败后置位
        _load_attempted，避免每条 query 都重试一遍加载。
        """
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # ImportError 及其它环境异常都算不可用
            logger.warning("sentence_transformers 不可用，交叉编码器精排已关闭: %s", exc)
            return
        try:
            self._model = CrossEncoder(self.model_name_or_path, device=self.device, max_length=512)
            logger.info("交叉编码器加载成功: %s (device=%s)", self.model_name_or_path, self.device)
        except Exception as exc:
            # 模型不存在、无网络、显存不足等都归入降级，不能让检索链路中断。
            logger.warning("交叉编码器 %s 加载失败，已降级: %s", self.model_name_or_path, exc)
            self._model = None

    @property
    def available(self) -> bool:
        """当前是否真的能做精排（会触发一次惰性加载）。"""
        self._ensure_model()
        return self._model is not None

    # -- 精排 --------------------------------------------------------------
    def _predict(self, pairs: List[List[str]]) -> List[float]:
        """分批打分。异常不在这里吞掉，交由 rerank() 统一降级，避免把"模型坏了"
        和"模型认为都不相关"混成同一种结果。"""
        scores: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            raw = self._model.predict(batch, batch_size=self.batch_size, show_progress_bar=False)
            for value in raw:
                scores.append(float(value))
        return scores

    def rerank(self, query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        """对 (query, chunk.text) 打分写入 rerank_score，并按之降序返回。

        用 chunk.text（子块）而不是 parent_text 打分：父块是给生成器补上下文用的，
        塞进交叉编码器只会稀释信号并撑爆输入长度。
        """
        if not hits or not query:
            return hits
        if not self.available:
            return hits

        pairs = [[query, (hit.chunk.text or "")[:MAX_RERANK_CHARS]] for hit in hits]
        try:
            scores = self._predict(pairs)
        except Exception as exc:
            logger.warning("交叉编码器打分失败，本次保持召回顺序: %s", exc)
            return hits
        if not scores or len(scores) != len(hits):
            logger.warning("交叉编码器返回 %s 个分数，与候选数 %d 不符，已忽略", len(scores or []), len(hits))
            return hits

        for hit, score in zip(hits, scores):
            hit.rerank_score = score
        # 稳定排序：rerank_score 相同的候选保持融合阶段的相对顺序。
        return sorted(hits, key=lambda h: -(h.rerank_score or 0.0))


# --------------------------------------------------------------------------
# 权威度 / 时效性加权
# --------------------------------------------------------------------------
def parse_year(version: str) -> Optional[int]:
    """从版本串里抽 4 位年份；出现多个（如 "2018-2023版"）取最新的那个。"""
    if not version:
        return None
    matches = _YEAR_RE.findall(str(version))
    if not matches:
        return None
    return max(int(item) for item in matches)


def recency_score(version: str) -> float:
    """把版本年份映射到 0~1；解析不出年份时给中性值 0.5。

    给 0.5 而不是 0，是因为"没写版本"只代表元数据缺失，不代表这份文档过时，
    直接判 0 会系统性地惩罚解析不完整的语料。
    """
    year = parse_year(version)
    if year is None:
        return DEFAULT_RECENCY
    ratio = (year - RECENCY_BASE_YEAR) / RECENCY_SPAN_YEARS
    return max(0.0, min(1.0, ratio))


def _sigmoid(value: float) -> float:
    """数值安全的 sigmoid（两侧分支写法避免 exp 溢出）。"""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _normalized_bases(hits: List[RetrievalHit]) -> List[float]:
    """取语义基线并归一化到 0~1。

    只有当**全部**候选都有 rerank_score 时才用 rerank_score：精排分与融合分
    量纲完全不同（前者是 logits，可正可负；后者是 RRF 分，都在 0.0x 量级），
    混着用会让没被精排到的候选凭空占便宜。

    这里刻意不用 min-max：min-max 会把当前候选集里的最高分强行拉到 1、最低分
    压到 0，两条语义上几乎并列的结果被人为拉开成 1.0 与 0.0，权威度偏置再也
    翻不动——恰好废掉了本模块存在的理由。改用：

    * rerank_score 走 sigmoid，这是交叉编码器 logits 的标准用法，得到的是与
      候选集合无关的绝对相关度，两条都很像就都接近 1；
    * 融合分（非负）按最大值缩放，保留原本的相对差距（0.0318/0.0320≈0.99）。

    只有在出现负的融合分（自定义权重的消融实验可能造出来）时才退回 min-max，
    因为那时"除以最大值"没有意义。
    """
    if not hits:
        return []
    if all(hit.rerank_score is not None for hit in hits):
        return [_sigmoid(float(hit.rerank_score)) for hit in hits]

    values = [float(hit.score) for hit in hits]
    hi = max(values)
    if min(values) >= 0.0:
        if hi <= 0.0:
            # 全零：语义侧毫无信息，交给权威度/时效性排序。
            return [1.0] * len(values)
        return [v / hi for v in values]

    lo = min(values)
    span = hi - lo
    if span <= 0:
        return [1.0] * len(values)
    return [(v - lo) / span for v in values]


def apply_authority_boost(
    hits: List[RetrievalHit],
    authority_weight: float = 0.15,
    recency_weight: float = 0.05,
) -> List[RetrievalHit]:
    """final_score = 归一化语义分 + 权威度偏置 + 时效性偏置，并按 final_score 降序。

    chunk.authority 来自 schema.TIER_AUTHORITY（指南 1.00 / 说明书 0.95 /
    教科书 0.75 / 百科 0.60 / 图谱 0.55 / 问诊 0.30），所以同为满分语义相关时，
    一条指南比一条在线问诊多拿 0.15*(1.00-0.30)=0.105 分，足够翻转次序；而一条
    语义分归一化后差 0.3 的指南则翻不过来——这正是"裁判而非主排序"的设计意图。
    """
    if not hits:
        return []

    bases = _normalized_bases(hits)
    for hit, base in zip(hits, bases):
        chunk = hit.chunk
        authority = float(getattr(chunk, "authority", 0.3) or 0.0)
        recency = recency_score(getattr(chunk, "version", "") or "")
        hit.final_score = base + authority_weight * authority + recency_weight * recency
        # 归一化前的语义分单独留一份。final_score 里的 base 是按候选集归一过的
        # （第一名恒为 1.0），任何"绝对相关度"判断——尤其是拒答闸门——都不能读它。
        # 详见 verifier.absolute_relevance。
        setattr(hit, ABSOLUTE_SCORE_ATTR, _absolute_base(hit))
    return sorted(hits, key=lambda h: -h.final_score)


# 挂在 RetrievalHit 上的额外属性名。schema.RetrievalHit 是冻结的数据契约，
# 不能加字段，所以用 setattr 附着；读取方一律走 getattr(hit, ..., None)。
ABSOLUTE_SCORE_ATTR = "absolute_score"


def _absolute_base(hit: RetrievalHit) -> float:
    """未经集合归一的语义分：有精排分走 sigmoid（绝对量纲），否则用原始融合分。"""
    if hit.rerank_score is not None:
        return _sigmoid(float(hit.rerank_score))
    return float(hit.score)


def explain_boost(hit: RetrievalHit) -> Dict[str, float]:
    """拆解某条命中的加权构成，方便在 debug 输出里解释排序理由。"""
    return {
        "score": round(float(hit.score), 6),
        "rerank_score": None if hit.rerank_score is None else round(float(hit.rerank_score), 6),
        "authority": round(float(getattr(hit.chunk, "authority", 0.3) or 0.0), 6),
        "recency": round(recency_score(getattr(hit.chunk, "version", "") or ""), 6),
        # 归一化前的语义分：排查"为什么这条没被拒答闸门拦下"时要看的是它，不是 final_score
        "absolute_score": round(float(getattr(hit, ABSOLUTE_SCORE_ATTR, hit.score) or 0.0), 6),
        "final_score": round(float(hit.final_score), 6),
    }


def _demo_hits() -> List[RetrievalHit]:
    """构造几条语义分接近、来源等级不同的命中，演示权威度裁决。"""
    from medical_rag.schema import TIER_AUTHORITY, TIER_CONSULTATION, TIER_GUIDELINE
    from medical_rag.schema import Chunk

    raw = [
        ("c1", "我吃二甲双胍拉肚子，是不是不能吃了", "在线问诊", TIER_CONSULTATION, "", 0.0320),
        ("g1", "二甲双胍为 2 型糖尿病药物治疗的首选，eGFR<30 时禁用", "中国2型糖尿病防治指南",
         TIER_GUIDELINE, "2020年版", 0.0316),
        ("c2", "血糖高平时要注意饮食", "在线问诊", TIER_CONSULTATION, "", 0.0180),
    ]
    hits = []
    for chunk_id, text, title, tier, version, score in raw:
        chunk = Chunk(
            chunk_id=chunk_id,
            text=text,
            doc_id=title,
            doc_title=title,
            source_tier=tier,
            version=version,
            authority=TIER_AUTHORITY[tier],
        )
        hits.append(RetrievalHit(chunk=chunk, score=score))
    return hits


def parse_args():
    parser = argparse.ArgumentParser(description="交叉编码器精排与权威度加权自检。")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--authority_weight", type=float, default=0.15)
    parser.add_argument("--recency_weight", type=float, default=0.05)
    parser.add_argument("--query", default="二甲双胍的禁忌证有哪些")
    parser.add_argument("--probe_model", action="store_true", help="探测交叉编码器是否可用（可能触发模型下载）")
    return parser.parse_args()


def main():
    args = parse_args()
    reranker = CrossEncoderReranker(args.reranker, device=args.device, batch_size=args.batch_size)
    hits = _demo_hits()
    if args.probe_model:
        print("reranker available: %s" % reranker.available)
        hits = reranker.rerank(args.query, hits)
    ranked = apply_authority_boost(hits, args.authority_weight, args.recency_weight)
    for hit in ranked:
        print("%s %s" % (hit.chunk.chunk_id, explain_boost(hit)))


if __name__ == "__main__":
    main()
