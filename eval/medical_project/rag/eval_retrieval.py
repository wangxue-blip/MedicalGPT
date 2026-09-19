#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索质量评测：三路召回各自贡献了什么，融合与精排各自又加了多少分。

为什么要按"路"拆开评，而不是只看端到端的一个数
-------------------------------------------------
RAG 的端到端指标（答案对不对）把检索和生成揉在一起，检索掉了一条关键证据和
生成把证据读错，在最终分数上长得一模一样，没法据此决定下一步该调哪儿。本脚本
只评检索这一段，并且把链路切成可对照的三层：

  1. 单路 recall@k —— dense / bm25 / term 各自单独能捞回多少 gold。这是消融的
     基础：如果 term 路的 recall 和 bm25 完全重合，那它就只是在增加延迟；
     如果某一路在裸环境降级成哈希向量后 recall 掉到接近 0，也能立刻看出来。
  2. 融合后 recall@k —— 减去"最好的单路"就是 RRF 真正带来的增量。多路融合看起来
     很美，但当某一路显著强于其它路时，融合反而可能把它的头部结果压下去，
     这个差值是唯一能说明问题的证据。
  3. 精排 + 权威度加权后 —— 与融合层对比。这一层不改变候选集合，只改顺序，
     所以 recall@k(k 足够大时) 不该变，变的是 MRR 和 nDCG；若 recall 也掉了，
     说明精排把 gold 挤出了 topk，是明确的回退信号。

指标口径
-------------------------------------------------
recall@k = 命中的 gold 数 / gold 总数（按 query 求均值，不是全局微平均）；
MRR@k    = 首个 gold 的倒数名次，topk 内无 gold 记 0；
nDCG@k   = 二值相关性，IDCG 按 min(|gold|, k) 条满分算。

弱标注模式（--weak_label_from_answer）
-------------------------------------------------
没有人工相关性标注时，用参考答案与各 chunk 的字符 n-gram 重合度反推 gold。
必须清楚它的两个局限，脚本的输出里也会原样写上：

  * 它衡量的是**字面重合**，不是相关性。一条换了说法但同样正确的证据会被判为
    不相关，一条抄了题面却答非所问的 chunk 反而可能被选中；
  * 它对 bm25 / term 这类词面路有系统性偏好——这两路和弱标注用的是同一种信号。
    所以弱标注下的"各路贡献"只能用来看趋势，绝不能拿去论证 dense 路没用。

因此弱标注结果只适合做回归监控（同一套标注下比较改动前后），不能当成绝对的
检索质量，更不等于医学正确性。

降级行为
-------------------------------------------------
脚本本身只依赖标准库；medical_rag 各模块自带降级（sentence_transformers /
faiss / numpy / jieba / rank_bm25 缺失时分别退化为哈希向量、纯 Python 点积、
内置 BM25、字符 bigram），任何一路构建失败都只跳过该路并记入输出的 degraded
字段，不影响其余路继续评测。交叉编码器默认关闭，需要时用 --reranker_model 指定。
"""

import argparse
import json
import logging
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_rag.schema import Chunk, load_chunks, normalize_text, read_jsonl  # noqa: E402

logger = logging.getLogger("eval_retrieval")

# 所有自动指标共用的免责声明。检索评测最容易被误读成"模型医学水平"，
# 所以它同时写进 json 和 stdout，报告怎么截图都不会把这句话漏掉。
DISCLAIMER = (
    "本文件中的全部指标均为启发式自动指标：recall/MRR/nDCG 衡量的是"
    "'标注为 gold 的 chunk 有没有被检索回来'，不等于医学准确率，也不等于"
    "临床正确性；弱标注模式下的 gold 由字符 n-gram 重合度推得，更不等于"
    "人工相关性判断。任何对外结论都需人工复核。"
)

WEAK_LABEL_CAVEAT = (
    "weak_label：gold 由 gold_answer 与 chunk 的字符 n-gram 相似度自动推得，"
    "衡量字面重合而非相关性，且对 bm25/term 等词面路有系统性偏好，"
    "各路对比仅供趋势参考。"
)

DEFAULT_TOPK_LIST = "1,3,5,10"
DEFAULT_ROUTES = "dense,bm25,term"

# 弱标注候选池：先用字符 bigram 倒排做一次廉价预筛，再对池内候选算滑窗 Dice。
# 不预筛的话是 query 数 x chunk 数次 Dice，万级语料上要跑几十分钟。
DEFAULT_WEAK_POOL = 200
# 文档频率超过这个比例的 bigram（"患者""可以"之类）对区分毫无帮助，
# 却会让倒排的候选集退化成全量，直接丢弃。
MAX_DF_RATIO = 0.3

QUERY_FIELDS = ("query", "question", "input", "instruction")
GOLD_ID_FIELDS = ("gold_chunk_ids", "gold_ids", "relevant_chunk_ids", "gold")
GOLD_ANSWER_FIELDS = ("gold_answer", "answer", "reference", "output", "response")


# --------------------------------------------------------------------------
# 通用小工具
# --------------------------------------------------------------------------
def write_json(path: str, data: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def parse_int_list(raw: str, default: Sequence[int]) -> List[int]:
    values: List[int] = []
    for piece in re.split(r"[,\s]+", raw or ""):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError:
            logger.warning("忽略无法解析的 k 值: %r", piece)
            continue
        if value > 0 and value not in values:
            values.append(value)
    return sorted(values) if values else list(default)


def parse_str_list(raw: str, default: Sequence[str]) -> List[str]:
    values = [piece.strip() for piece in re.split(r"[,\s]+", raw or "") if piece.strip()]
    return values or list(default)


def parse_weights(raw: str) -> Optional[Dict[str, float]]:
    """解析 "dense=1.0,bm25=1.0,term=0.6"；解析不出来就返回 None 用默认权重。"""
    if not raw:
        return None
    weights: Dict[str, float] = {}
    for piece in re.split(r"[,\s]+", raw):
        if not piece.strip():
            continue
        if "=" not in piece and ":" not in piece:
            logger.warning("忽略无法解析的权重项: %r", piece)
            continue
        key, _, value = piece.replace(":", "=").partition("=")
        try:
            weights[key.strip()] = float(value)
        except ValueError:
            logger.warning("忽略无法解析的权重值: %r", piece)
    return weights or None


def first_field(row: Dict[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if row.get(field):
            return row[field]
    return None


def as_id_list(value: Any) -> List[str]:
    """gold 字段既可能是 list，也可能是单个 id 或逗号分隔的字符串。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [piece.strip() for piece in re.split(r"[,\s]+", text) if piece.strip()]


# --------------------------------------------------------------------------
# 排序指标
# --------------------------------------------------------------------------
def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log(pos + 2, 2) for pos, rel in enumerate(relevances) if rel)


def rank_metrics(ranked_ids: Sequence[str], gold: Sequence[str], ks: Sequence[int]) -> Dict[str, float]:
    """一条 query 的 recall@k / MRR@k / nDCG@k。

    gold 去重后参与分母：同一条 chunk 在标注里写两遍不应该让 recall 天然减半。
    """
    gold_set = set(gold)
    result: Dict[str, float] = {}
    if not gold_set:
        return result

    for k in ks:
        window = list(ranked_ids[:k])
        rels = [1 if cid in gold_set else 0 for cid in window]
        hit = sum(rels)
        result["recall@%d" % k] = hit / float(len(gold_set))

        mrr = 0.0
        for pos, rel in enumerate(rels, start=1):
            if rel:
                mrr = 1.0 / pos
                break
        result["mrr@%d" % k] = mrr

        ideal = _dcg([1] * min(len(gold_set), k))
        result["ndcg@%d" % k] = (_dcg(rels) / ideal) if ideal > 0 else 0.0
    return result


def mean_metrics(samples: List[Dict[str, float]], keys: Sequence[str]) -> Dict[str, float]:
    """按 key 求均值；没有样本时返回 0.0 而不是抛异常，便于空评测集也能出报告。"""
    out: Dict[str, float] = {}
    for key in keys:
        values = [sample[key] for sample in samples if key in sample]
        out[key] = statistics.mean(values) if values else 0.0
    return out


def metric_keys(ks: Sequence[int]) -> List[str]:
    keys: List[str] = []
    for k in ks:
        keys.extend(["recall@%d" % k, "mrr@%d" % k, "ndcg@%d" % k])
    return keys


def recall_keys(ks: Sequence[int]) -> List[str]:
    return ["recall@%d" % k for k in ks]


# --------------------------------------------------------------------------
# 弱标注
# --------------------------------------------------------------------------
def _ngram_scorer():
    """取 verifier 的滑窗 Dice，取不到则用本地等价实现。

    优先复用 verifier 是为了让"弱标注判定相关"和"引用校验判定支持"用同一把尺，
    否则 eval_retrieval 说命中、eval_citation 说不支持，两份报告会互相打架。
    """
    try:
        from medical_rag.verifier import ngram_support_score  # type: ignore
        return ngram_support_score, "verifier.ngram_support_score"
    except Exception as exc:
        logger.warning("medical_rag.verifier 不可用，弱标注改用本地 n-gram 实现: %s", exc)
        return _local_ngram_score, "local_ngram_dice"


def _fold(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^\w一-鿿]+", "", text)


def _grams(text: str, min_n: int = 2, max_n: int = 3) -> Dict[str, int]:
    folded = _fold(text)
    counts: Dict[str, int] = {}
    for token in re.findall(r"[a-z0-9]+", folded):
        counts[token] = counts.get(token, 0) + 1
    cjk = "".join(re.findall(r"[一-鿿]", folded))
    for n in range(min_n, max_n + 1):
        for i in range(len(cjk) - n + 1):
            gram = cjk[i:i + n]
            counts[gram] = counts.get(gram, 0) + 1
    return counts


def _local_ngram_score(query: str, evidence: str) -> float:
    """加权 Dice 的本地兜底实现（不滑窗，长度差会稀释分数，阈值需相应调低）。"""
    a = _grams(query)
    b = _grams(evidence)
    if not a or not b:
        return 0.0
    overlap = sum(min(count, b.get(gram, 0)) for gram, count in a.items())
    total = sum(a.values()) + sum(b.values())
    return 2.0 * overlap / total if total else 0.0


class WeakLabeler:
    """用参考答案反推 gold chunk。

    两段式：bigram 倒排预筛出候选池 -> 池内算滑窗 n-gram 相似度 -> 取 top-m 且
    过阈值。预筛只为省时间，不改变判定口径（池足够大时与全量扫描等价）。
    """

    def __init__(self, chunks: List[Chunk], topm: int = 3, threshold: float = 0.35,
                 pool: int = DEFAULT_WEAK_POOL):
        self.chunks = chunks
        self.topm = max(1, int(topm))
        self.threshold = float(threshold)
        self.pool = max(self.topm, int(pool))
        self.score_fn, self.backend = _ngram_scorer()
        self._postings: Dict[str, List[int]] = {}
        self._build_postings()

    def _build_postings(self) -> None:
        total = len(self.chunks)
        if not total:
            return
        for idx, chunk in enumerate(self.chunks):
            folded = _fold(chunk.text)
            seen = set()
            for i in range(len(folded) - 1):
                gram = folded[i:i + 2]
                if gram in seen:
                    continue
                seen.add(gram)
                self._postings.setdefault(gram, []).append(idx)
        cutoff = max(1, int(total * MAX_DF_RATIO))
        for gram in list(self._postings.keys()):
            if len(self._postings[gram]) > cutoff:
                del self._postings[gram]

    def _candidates(self, answer: str) -> List[int]:
        folded = _fold(answer)
        counts: Dict[int, int] = {}
        seen = set()
        for i in range(len(folded) - 1):
            gram = folded[i:i + 2]
            if gram in seen:
                continue
            seen.add(gram)
            for idx in self._postings.get(gram, ()):  # 高频 gram 已在建表时剔除
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            # 预筛为空（答案太短或全是高频词）时退回全量，宁可慢也不要漏标。
            return list(range(len(self.chunks)))
        order = sorted(counts.keys(), key=lambda i: (-counts[i], i))
        return order[:self.pool]

    def label(self, answer: str) -> List[Tuple[str, float]]:
        answer = normalize_text(answer)
        if not answer or not self.chunks:
            return []
        scored: List[Tuple[int, float]] = []
        for idx in self._candidates(answer):
            try:
                score = float(self.score_fn(answer, self.chunks[idx].text))
            except Exception as exc:
                logger.warning("弱标注打分失败，该 chunk 记 0 分: %s", exc)
                score = 0.0
            if score >= self.threshold:
                scored.append((idx, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [(self.chunks[idx].chunk_id, score) for idx, score in scored[:self.topm]]


# --------------------------------------------------------------------------
# 检索链路
# --------------------------------------------------------------------------
def build_terminology(path: Optional[str], degraded: List[str]):
    try:
        from medical_rag.terminology import Terminology  # type: ignore
    except Exception as exc:
        degraded.append("terminology:%s" % exc)
        logger.warning("术语表不可用，term 路与查询扩展将退化: %s", exc)
        return None
    try:
        return Terminology.from_jsonl(path) if path else Terminology.default()
    except Exception as exc:
        degraded.append("terminology_load:%s" % exc)
        logger.warning("术语表加载失败，改用内置默认表: %s", exc)
        try:
            return Terminology.default()
        except Exception:
            return None


def build_reranker(model: Optional[str], device: str, degraded: List[str]):
    """交叉编码器默认不开：裸环境下它必然降级，而每次构造都要试一遍加载。"""
    if not model:
        return None
    try:
        from medical_rag.rerank import CrossEncoderReranker  # type: ignore
        reranker = CrossEncoderReranker(model_name_or_path=model, device=device)
    except Exception as exc:
        degraded.append("reranker:%s" % exc)
        logger.warning("交叉编码器不可用，精排层退化为仅权威度加权: %s", exc)
        return None
    try:
        if not reranker.available:
            degraded.append("reranker_unavailable:%s" % model)
            logger.warning("交叉编码器 %s 未能加载，精排层退化为仅权威度加权", model)
    except Exception as exc:
        degraded.append("reranker_probe:%s" % exc)
        return None
    return reranker


def recall_routes(retriever, variants: Sequence[str]) -> Dict[str, List[Tuple[int, float]]]:
    """跑一次三路召回，融合前后共用同一批候选。

    复用 MedicalRetriever 的 _recall_route 是有意为之：评测必须和线上走同一套
    "多变体按最好排名合并"的逻辑，在评测脚本里另写一份迟早会和线上分叉，到时候
    "评测涨了线上没涨"的排查成本远高于这点耦合。该方法不存在时退化为直接用
    首个变体调 index.search。
    """
    route_hits: Dict[str, List[Tuple[int, float]]] = {}
    fallback = not hasattr(retriever, "_recall_route")
    for route, index in retriever.indexes.items():
        try:
            if fallback:
                raw = index.search(variants[0] if variants else "", retriever.recall_k) or []
                merged = [(int(idx), float(score)) for idx, score in raw]
            else:
                merged, _ = retriever._recall_route(route, index, variants)
        except Exception as exc:
            logger.warning("路由 %s 召回失败，按空结果处理: %s", route, exc)
            merged = []
        route_hits[route] = list(merged)
    return route_hits


def stage_rankings(
    retriever,
    query: str,
    route_hits: Dict[str, List[Tuple[int, float]]],
    chunks: List[Chunk],
    max_k: int,
) -> Tuple[List[str], List[str], Dict[str, List[str]], bool]:
    """把同一批候选分别跑到"融合后"和"精排+加权后"，返回两层的 chunk_id 序列。

    Returns:
        (fused_ids, final_ids, route_ids, cross_encoder_used)
    """
    from medical_rag.fusion import reciprocal_rank_fusion  # 延迟导入，保持脚本 import 轻量
    from medical_rag.rerank import apply_authority_boost

    route_ids = {
        route: [chunks[idx].chunk_id for idx, _ in pairs[:max_k] if 0 <= idx < len(chunks)]
        for route, pairs in route_hits.items()
    }

    if not any(route_hits.values()):
        return [], [], route_ids, False

    fused = reciprocal_rank_fusion(
        route_hits, chunks, k=retriever.fusion_k, weights=retriever.route_weights
    )
    if not fused:
        return [], [], route_ids, False
    # 先把融合顺序快照下来：apply_authority_boost 会原地改 final_score 并返回新列表，
    # 列表对象虽不变，但先取 id 更不易在后续改动里踩到共享引用的坑。
    fused_ids = [hit.chunk.chunk_id for hit in fused[:max_k]]

    pool = fused[:max(max_k, retriever.rerank_candidates)]
    reranked = retriever._rerank(normalize_text(query), pool) if hasattr(retriever, "_rerank") else pool
    cross_used = any(hit.rerank_score is not None for hit in reranked)
    ranked = apply_authority_boost(
        reranked,
        authority_weight=retriever.authority_weight,
        recency_weight=retriever.recency_weight,
    )
    final_ids = [hit.chunk.chunk_id for hit in ranked[:max_k]]
    return fused_ids, final_ids, route_ids, cross_used


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="评测医疗 RAG 的检索质量：单路 recall、融合增益、精排前后对比。"
    )
    parser.add_argument("--chunk_file", required=True, help="语料 chunks jsonl")
    parser.add_argument("--eval_file", required=True,
                        help="评测集 jsonl，每行 {query, gold_chunk_ids} 或 {query, gold_answer}")
    parser.add_argument("--output", required=True, help="指标 json 输出路径")
    parser.add_argument("--model", default=None, help="模型/配置标识，供汇总脚本归并")
    parser.add_argument("--topk_list", default=DEFAULT_TOPK_LIST, help="逗号分隔的 k，如 1,3,5,10")
    parser.add_argument("--routes", default=DEFAULT_ROUTES, help="参与评测的召回路")
    parser.add_argument("--recall_k", type=int, default=50, help="每路召回数量")
    parser.add_argument("--fusion_k", type=int, default=60, help="RRF 平滑常数")
    parser.add_argument("--route_weights", default="", help='如 "dense=1.0,bm25=1.0,term=0.6"')
    parser.add_argument("--authority_weight", type=float, default=0.15, help="来源权威度加权系数")
    parser.add_argument("--dense_model", default=None, help="稠密向量模型路径，缺省走内置探测/降级")
    parser.add_argument("--terminology_file", default=None, help="术语同义词 jsonl")
    parser.add_argument("--reranker_model", default="", help="交叉编码器模型；留空则不做精排")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_queries", type=int, default=-1)
    parser.add_argument("--max_chunks", type=int, default=-1)
    parser.add_argument("--weak_label_from_answer", action="store_true",
                        help="无人工标注时用 gold_answer 的字符 n-gram 相似度反推 gold")
    parser.add_argument("--weak_label_topm", type=int, default=3, help="弱标注每条 query 取多少条 gold")
    parser.add_argument("--weak_label_threshold", type=float, default=0.35, help="弱标注相似度下限")
    parser.add_argument("--weak_label_pool", type=int, default=DEFAULT_WEAK_POOL,
                        help="弱标注预筛候选池大小")
    parser.add_argument("--dump_per_query", default=None, help="逐条明细 jsonl 输出路径（可选）")
    return parser.parse_args()


def infer_model_name(args) -> str:
    if args.model:
        return args.model
    return Path(args.eval_file).stem


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    ks = parse_int_list(args.topk_list, [1, 3, 5, 10])
    max_k = max(ks)
    routes = parse_str_list(args.routes, ("dense", "bm25", "term"))
    degraded: List[str] = []

    chunks = load_chunks(args.chunk_file, max_rows=args.max_chunks)
    if not chunks:
        raise ValueError("语料为空: %s" % args.chunk_file)
    # chunk_id 重复会让 recall 凭空变高（一条 gold 被多条候选同时"命中"），
    # 这属于语料构建问题，必须显式喊出来而不是静悄悄算完。
    unique_ids = set(chunk.chunk_id for chunk in chunks)
    if len(unique_ids) != len(chunks):
        degraded.append("duplicate_chunk_id:%d" % (len(chunks) - len(unique_ids)))
        logger.warning("语料中有 %d 条重复 chunk_id，指标可能偏高", len(chunks) - len(unique_ids))

    terminology = build_terminology(args.terminology_file, degraded)
    reranker = build_reranker(args.reranker_model, args.device, degraded)

    from medical_rag.index import build_indexes  # 延迟导入：这一步才会去探测可选依赖
    from medical_rag.retriever import MedicalRetriever

    indexes = build_indexes(
        chunks, dense_model=args.dense_model, terminology=terminology,
        routes=routes, device=args.device,
    )
    missing = [route for route in routes if route not in indexes]
    if missing:
        degraded.append("routes_unavailable:%s" % ",".join(missing))
    if not indexes:
        raise ValueError("所有召回路构建失败，无法评测")

    retriever = MedicalRetriever(
        chunks, indexes, reranker=reranker, terminology=terminology,
        recall_k=args.recall_k, fusion_k=args.fusion_k,
        route_weights=parse_weights(args.route_weights),
        authority_weight=args.authority_weight,
    )

    rows = list(read_jsonl(args.eval_file, max_rows=args.max_queries))
    if not rows:
        raise ValueError("评测集为空: %s" % args.eval_file)

    labeler = None
    if args.weak_label_from_answer:
        labeler = WeakLabeler(chunks, topm=args.weak_label_topm,
                              threshold=args.weak_label_threshold, pool=args.weak_label_pool)

    valid_ids = set(chunk.chunk_id for chunk in chunks)
    all_keys = metric_keys(ks)

    fused_samples: List[Dict[str, float]] = []
    final_samples: List[Dict[str, float]] = []
    route_samples: Dict[str, List[Dict[str, float]]] = dict((route, []) for route in indexes)
    per_query: List[Dict[str, Any]] = []

    num_queries = 0
    num_with_gold = 0
    num_weak_labeled = 0
    num_gold_missing = 0
    cross_used_any = False

    for row in rows:
        query = normalize_text(str(first_field(row, QUERY_FIELDS) or ""))
        if not query:
            logger.warning("跳过缺少 query 的样本: %r", row)
            continue
        num_queries += 1

        gold = as_id_list(first_field(row, GOLD_ID_FIELDS))
        label_source = "manual"
        weak_scores: List[float] = []
        if gold:
            unknown = [cid for cid in gold if cid not in valid_ids]
            if unknown:
                # 标注里的 id 在语料里不存在 -> recall 分母虚高，是最隐蔽的一类评测错误。
                num_gold_missing += len(unknown)
                logger.warning("query=%r 的 %d 个 gold id 不在语料中", query[:24], len(unknown))
        elif labeler is not None:
            answer = str(first_field(row, GOLD_ANSWER_FIELDS) or "")
            labeled = labeler.label(answer)
            gold = [cid for cid, _ in labeled]
            weak_scores = [round(score, 6) for _, score in labeled]
            label_source = "weak_from_answer"
            if gold:
                num_weak_labeled += 1

        if not gold:
            logger.warning("query=%r 没有可用 gold，已跳过指标统计", query[:24])
            continue
        num_with_gold += 1

        variants = retriever.expand(query)
        route_hits = recall_routes(retriever, variants)
        fused_ids, final_ids, route_ids, cross_used = stage_rankings(
            retriever, query, route_hits, chunks, max_k
        )
        cross_used_any = cross_used_any or cross_used

        fused_metrics = rank_metrics(fused_ids, gold, ks)
        final_metrics = rank_metrics(final_ids, gold, ks)
        fused_samples.append(fused_metrics)
        final_samples.append(final_metrics)
        for route in route_samples:
            route_samples[route].append(rank_metrics(route_ids.get(route, []), gold, ks))

        if args.dump_per_query:
            per_query.append({
                "query": query,
                "label_source": label_source,
                "gold_chunk_ids": gold,
                "weak_label_scores": weak_scores,
                "variants": variants,
                "route_topk": dict((route, ids[:max_k]) for route, ids in route_ids.items()),
                "fused_topk": fused_ids,
                "final_topk": final_ids,
                "fused_metrics": dict((k, round(v, 6)) for k, v in fused_metrics.items()),
                "final_metrics": dict((k, round(v, 6)) for k, v in final_metrics.items()),
            })

    fused_mean = mean_metrics(fused_samples, all_keys)
    final_mean = mean_metrics(final_samples, all_keys)
    route_mean = dict(
        (route, mean_metrics(samples, recall_keys(ks)))
        for route, samples in route_samples.items()
    )

    # 融合增益 = 融合后 recall 减去"当时最好的那一路"。单路全挂时退化为 0，
    # 此时 fusion_gain 没有解释力，看 route_recall 全 0 即可判断。
    fusion_gain = {}
    best_route = {}
    for key in recall_keys(ks):
        per_route = [values.get(key, 0.0) for values in route_mean.values()]
        best = max(per_route) if per_route else 0.0
        best_route[key] = best
        fusion_gain[key] = fused_mean.get(key, 0.0) - best

    rerank_delta = dict((key, final_mean.get(key, 0.0) - fused_mean.get(key, 0.0)) for key in all_keys)

    label_mode = "weak_from_answer" if args.weak_label_from_answer else "manual"
    result = {
        "model": infer_model_name(args),
        "chunk_file": args.chunk_file,
        "eval_file": args.eval_file,
        "num_queries": num_queries,
        # num_samples 是 eval/medical_project 下其它指标文件的通用键名，
        # 这里给一个同义别名，汇总脚本不必为检索评测单开一条分支。
        "num_samples": num_queries,
        "num_queries_with_gold": num_with_gold,
        "num_queries_weak_labeled": num_weak_labeled,
        "num_gold_ids_not_in_corpus": num_gold_missing,
        "num_chunks": len(chunks),
        "topk_list": ks,
        "label_mode": label_mode,
        "metric_type": "heuristic_auto",
        "routes": sorted(indexes.keys()),
        "route_backends": dict(
            (route, getattr(index, "backend", "unknown")) for route, index in indexes.items()
        ),
        "route_weights": dict(retriever.route_weights),
        "recall_k": retriever.recall_k,
        "fusion_k": retriever.fusion_k,
        "authority_weight": retriever.authority_weight,
        "cross_encoder_enabled": bool(cross_used_any),
        "degraded": degraded,
        # metrics 取"精排+加权后"的最终排序，即线上真正返回给生成器的顺序。
        "metrics": dict((key, round(value, 6)) for key, value in final_mean.items()),
        "stages": {
            "fused": dict((key, round(value, 6)) for key, value in fused_mean.items()),
            "final": dict((key, round(value, 6)) for key, value in final_mean.items()),
        },
        "route_recall": dict(
            (route, dict((key, round(value, 6)) for key, value in values.items()))
            for route, values in route_mean.items()
        ),
        "best_single_route_recall": dict((key, round(value, 6)) for key, value in best_route.items()),
        "fusion_gain": dict((key, round(value, 6)) for key, value in fusion_gain.items()),
        "rerank_delta": dict((key, round(value, 6)) for key, value in rerank_delta.items()),
        "disclaimer": DISCLAIMER,
    }
    if args.weak_label_from_answer:
        result["weak_label"] = {
            "enabled": True,
            "topm": args.weak_label_topm,
            "threshold": args.weak_label_threshold,
            "pool": args.weak_label_pool,
            "backend": labeler.backend if labeler is not None else "unavailable",
            "caveat": WEAK_LABEL_CAVEAT,
        }

    write_json(args.output, result)
    if args.dump_per_query and per_query:
        detail_path = Path(args.dump_per_query)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w", encoding="utf-8") as handle:
            for item in per_query:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[免责声明] %s" % DISCLAIMER)
    if args.weak_label_from_answer:
        print("[弱标注] %s" % WEAK_LABEL_CAVEAT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
