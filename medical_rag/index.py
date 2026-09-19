# -*- coding: utf-8 -*-
"""医疗 RAG 的三路召回索引：dense（语义） / bm25（词面） / term（术语）。

为什么要三路而不是一路稠密召回：
  - 稠密向量擅长语义近似，但对"二甲双胍 vs 二甲双胍缓释片"、"I21.9 vs I21.0"
    这类只差几个字符却是完全不同临床含义的串，区分度反而不如词面匹配；
  - BM25 擅长精确串匹配（药名、ICD 编码、剂量），但遇到"降压药有哪些"这种
    完全不共词的提问就废了；
  - 术语路把口语说法（心梗 / 心肌梗死 / AMI）归一到同一个规范术语，补上
    前两路都覆盖不到的同义改写。
三路各自召回、RRF 融合，是医疗场景下性价比最高的组合。

依赖策略（对齐 tools/medical_project/embed_and_filter_sft.py 的双 backend 写法）：
本模块在"裸环境"（无 numpy / faiss / jieba / rank_bm25 / sentence_transformers）
下必须 import 成功并能跑通。所有可选依赖的 ImportError 一律降级处理，
每个索引对象都暴露 backend 属性说明当前实际走的是哪条路，便于线上排查。

降级链：
  DenseIndex : sentence_transformers(+faiss / +numpy) -> 字符 n-gram 哈希稀疏向量
  BM25Index  : rank_bm25.BM25Okapi                    -> 自实现 BM25(k1=1.5, b=0.75)
  分词        : jieba.lcut                             -> 中文字符 bigram + 英文数字整词
  TermIndex  : medical_rag.terminology.Terminology     -> 内置极简术语表兜底
"""

import argparse
import json
import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # 允许 `python -m medical_rag.index` 与 `python medical_rag/index.py` 两种入口
    from .schema import (
        Chunk,
        TIER_DRUG_LABEL,
        TIER_GUIDELINE,
        TIER_TEXTBOOK,
        normalize_text,
        stable_id,
    )
except ImportError:  # pragma: no cover - `python medical_rag/index.py` 这种直接运行走这里
    import sys as _sys

    # 直接跑脚本时 sys.path[0] 是 medical_rag/ 本身，仓库根目录不在路径上，
    # 补进去才能按包名导入，否则 main() 自检无法执行。
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from medical_rag.schema import (  # type: ignore
        Chunk,
        TIER_DRUG_LABEL,
        TIER_GUIDELINE,
        TIER_TEXTBOOK,
        normalize_text,
        stable_id,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "BaseIndex",
    "DenseIndex",
    "BM25Index",
    "TermIndex",
    "build_indexes",
    "tokenize",
    "TOKENIZER_BACKEND",
]


# --------------------------------------------------------------------------
# 可选依赖探测
# --------------------------------------------------------------------------
# 这里只做探测，不在导入期打 warning：模块被 import 的时候刷一屏告警很吵，
# 真正发生降级（build 的时候）再 warning，信息更有定位价值。
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    _HAS_FAISS = False

try:
    import jieba  # type: ignore

    _HAS_JIEBA = True
except ImportError:  # pragma: no cover
    jieba = None  # type: ignore
    _HAS_JIEBA = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _HAS_RANK_BM25 = True
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore
    _HAS_RANK_BM25 = False

logger.debug(
    "optional deps: numpy=%s faiss=%s jieba=%s rank_bm25=%s",
    _HAS_NUMPY, _HAS_FAISS, _HAS_JIEBA, _HAS_RANK_BM25,
)

TOKENIZER_BACKEND = "jieba" if _HAS_JIEBA else "char_bigram"

# 常见的本地 embedding 模型目录，按优先级排。医院内网机器基本都是离线的，
# 直接写 HuggingFace repo id 会在 build 阶段卡住，所以先找本地目录。
DEFAULT_DENSE_MODEL_DIRS = (
    "models/bge-m3",
    "models/bge-large-zh-v1.5",
    "models/bge-base-zh-v1.5",
    "models/bge-small-zh-v1.5",
    "models/text2vec-base-chinese",
)

HASH_DIMS = 1 << 18
BM25_K1 = 1.5
BM25_B = 0.75


# --------------------------------------------------------------------------
# 分词
# --------------------------------------------------------------------------
# 英文单词、数字、ICD 编码（I21.9）、剂量（500mg）、复合药名（bge-m3 式连字符）
# 必须作为整体保留。BM25 这一路的全部价值就在于精确匹配这些串，
# 一旦被切成 bigram，"I21.9" 和 "I21.0" 就会高度相似，等于把这路废掉。
_SEG_RE = re.compile(
    r"([A-Za-z0-9]+(?:[.\-+/][A-Za-z0-9]+)*)"      # 拉丁字母 / 数字 / 编码整词
    r"|([一-鿿㐀-䶿぀-ヿ]+)"  # 连续 CJK 片段
)
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-+/][A-Za-z0-9]+)*")
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")

# 中文停用词只挡最没有区分度的那几个，医疗语料里宁可多召回也不要漏
_CJK_STOPWORDS = frozenset(
    ["的", "了", "是", "在", "和", "与", "及", "或", "有", "为", "对", "等", "中", "上", "下",
     "可", "会", "被", "把", "就", "都", "也", "而", "其", "之", "一", "个", "这", "那"]
)


def _cut_cjk(segment: str) -> List[str]:
    """切分一段连续中文。jieba 不可用时退化为字符 bigram。"""
    if _HAS_JIEBA:
        try:
            words = [w.strip() for w in jieba.lcut(segment)]
            return [w for w in words if w and w not in _CJK_STOPWORDS]
        except Exception as exc:  # pragma: no cover - jieba 词典损坏之类
            logger.warning("jieba 分词失败，本次退化为字符 bigram: %s", exc)
    if len(segment) <= 1:
        return [segment] if segment and segment not in _CJK_STOPWORDS else []
    # bigram 而不是 unigram：单字在中文医疗语料里几乎没有区分度（"病""症""药"）
    return [segment[i:i + 2] for i in range(len(segment) - 1)]


def tokenize(text: str) -> List[str]:
    """混合分词：中文走 jieba 或 bigram，英文与数字整词保留（统一小写）。"""
    tokens: List[str] = []
    for match in _SEG_RE.finditer(normalize_text(text)):
        latin, cjk = match.group(1), match.group(2)
        if latin:
            tokens.append(latin.lower())
        elif cjk:
            tokens.extend(_cut_cjk(cjk))
    return tokens


def _iter_char_ngrams(text: str, min_n: int = 2, max_n: int = 3, max_chars: int = 1024) -> Iterable[str]:
    """中文字符 n-gram，去掉空白后再切，避免排版空格把词切散。"""
    text = re.sub(r"\s+", "", normalize_text(text))[:max_chars]
    if not text:
        return
    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        for idx in range(len(text) - n + 1):
            yield text[idx:idx + n]


def _stable_hash(text: str, dims: int) -> int:
    """FNV-1a，跨进程稳定。用 Python 内置 hash 会因 PYTHONHASHSEED 每次不一样，
    那样缓存下来的向量换个进程就对不上了。"""
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value % dims


def _minmax(pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """把 (下标, 分数) 列表按分数降序并 min-max 归一到 [0,1]。

    RRF 融合只用排名不用分数，归一化纯粹是为了日志和调试时三路分数可比
    ——原始 BM25 分数是 0~30，稠密余弦是 0~1，混在日志里根本没法看。
    """
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
    hi = pairs[0][1]
    lo = pairs[-1][1]
    span = hi - lo
    if span <= 1e-12:
        return [(idx, 1.0) for idx, _ in pairs]
    return [(idx, (score - lo) / span) for idx, score in pairs]


# --------------------------------------------------------------------------
# 术语表兜底
# --------------------------------------------------------------------------
def _load_terminology_class():
    """惰性拿 Terminology 类。terminology.py 还没落地时也不能让本模块崩掉。"""
    try:
        from .terminology import Terminology  # type: ignore

        return Terminology
    except Exception:
        pass
    try:
        from medical_rag.terminology import Terminology  # type: ignore

        return Terminology
    except Exception as exc:
        logger.warning("medical_rag.terminology 不可用，TermIndex 使用内置兜底术语表: %s", exc)
        return None


class _FallbackTerminology:
    """极简术语兜底，只在 medical_rag.terminology 缺失时启用。

    只做两件真正有用的事：拉丁词/编码整词直接当术语（药名、ICD、检验项缩写），
    外加一份很小的中文同义词种子表。它不追求覆盖率，只保证 TermIndex 这一路
    在依赖缺失时不会静默返回空结果。
    """

    SEED_GROUPS = (
        ("心肌梗死", "心梗", "急性心梗", "ami", "心肌梗塞"),
        ("高血压", "血压高", "hypertension"),
        ("2型糖尿病", "二型糖尿病", "糖尿病", "t2dm", "diabetes"),
        ("二甲双胍", "metformin", "格华止"),
        ("阿司匹林", "aspirin", "乙酰水杨酸"),
        ("他汀", "阿托伐他汀", "瑞舒伐他汀", "statin"),
        ("脑卒中", "中风", "脑梗死", "stroke"),
        ("慢性阻塞性肺疾病", "慢阻肺", "copd"),
        ("肺炎", "pneumonia"),
        ("冠心病", "冠状动脉粥样硬化性心脏病", "chd"),
        ("心力衰竭", "心衰", "heart failure"),
        ("低血糖", "hypoglycemia"),
        ("肾功能不全", "肾功能损害", "肾衰"),
        ("妊娠期", "孕期", "怀孕"),
        ("不良反应", "副作用", "adverse reaction"),
    )

    def __init__(self, groups=None):
        groups = groups or self.SEED_GROUPS
        self._alias2canon: Dict[str, str] = {}
        self._canon2aliases: Dict[str, List[str]] = {}
        for group in groups:
            canon = group[0]
            aliases = list(group)
            self._canon2aliases[canon] = aliases
            for alias in aliases:
                self._alias2canon[alias.lower()] = canon

    def normalize(self, term: str) -> str:
        key = normalize_text(term).lower()
        return self._alias2canon.get(key, normalize_text(term))

    def expand_query(self, query: str, max_expansions: int = 8) -> List[str]:
        out = [normalize_text(query)]
        for term in self.terms_in(query):
            for alias in self._canon2aliases.get(term, []):
                if len(out) >= max_expansions:
                    return out
                if alias not in out:
                    out.append(alias)
        return out

    def terms_in(self, text: str) -> List[str]:
        text_norm = normalize_text(text)
        lowered = text_norm.lower()
        found: List[str] = []
        for alias, canon in self._alias2canon.items():
            if alias and alias in lowered and canon not in found:
                found.append(canon)
        # 拉丁/数字整词本身就是最可靠的"术语"：药品英文名、ICD 编码、检验缩写
        for token in _LATIN_TOKEN_RE.findall(text_norm):
            token = token.lower()
            if len(token) < 2 or token.isdigit():
                continue
            canon = self._alias2canon.get(token, token)
            if canon not in found:
                found.append(canon)
        return found

    @classmethod
    def default(cls) -> "_FallbackTerminology":
        return cls()


def _default_terminology():
    """优先真实 Terminology.default()，失败则用兜底实现。"""
    cls = _load_terminology_class()
    if cls is not None:
        try:
            return cls.default()
        except Exception as exc:
            logger.warning("Terminology.default() 失败，改用内置兜底术语表: %s", exc)
    return _FallbackTerminology.default()


# --------------------------------------------------------------------------
# 索引基类
# --------------------------------------------------------------------------
class BaseIndex(object):
    """三路索引的公共接口。search 返回 (chunk 下标, 归一化分数) 降序列表。"""

    name = "base"

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.backend = "none"

    def build(self, chunks: List[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, topk: int) -> List[Tuple[int, float]]:
        raise NotImplementedError

    # -- 公共小工具 ---------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.chunks)

    def _index_text(self, chunk: Chunk) -> str:
        """索引用文本：正文 + 标题 + 章节路径。

        标题和章节路径带进来是因为医疗文档的关键限定词经常只在标题里出现
        （"第三章 妊娠期用药"），正文段落里反而没有，不带就召不回。
        """
        parts = [chunk.text or ""]
        if chunk.doc_title:
            parts.append(chunk.doc_title)
        if chunk.section_path:
            parts.append(" ".join(chunk.section_path))
        return normalize_text(" ".join(p for p in parts if p))

    def __repr__(self) -> str:
        return "<%s backend=%s size=%d>" % (self.__class__.__name__, self.backend, self.size)


# --------------------------------------------------------------------------
# DenseIndex
# --------------------------------------------------------------------------
class _HashingEmbedder(object):
    """无依赖的字符 n-gram 哈希向量，sentence_transformers 缺失时兜底。

    输出的是稀疏 dict（特征下标 -> 权重）而不是稠密数组：维度 2^18，
    真要展成稠密矩阵，1 万条 chunk 就是 10GB 浮点数，没有任何意义。
    稀疏 dict + 倒排点积在这个量级上又快又准。
    """

    def __init__(self, dims: int = HASH_DIMS, min_n: int = 2, max_n: int = 3, max_chars: int = 1024):
        self.dims = dims
        self.min_n = min_n
        self.max_n = max_n
        self.max_chars = max_chars
        self.backend = "hashing"

    def encode_one(self, text: str) -> Dict[int, float]:
        weights: Dict[int, float] = defaultdict(float)
        for ngram in _iter_char_ngrams(text, self.min_n, self.max_n, self.max_chars):
            weights[_stable_hash(ngram, self.dims)] += 1.0
        # 拉丁整词额外加权：药品英文名和编码在中文语料里出现次数少，
        # 但一旦命中就是强信号，不加权会被大量中文 n-gram 淹没。
        for token in _LATIN_TOKEN_RE.findall(normalize_text(text)[: self.max_chars]):
            weights[_stable_hash("W:" + token.lower(), self.dims)] += 3.0
        norm = math.sqrt(sum(value * value for value in weights.values()))
        if norm <= 0:
            return {}
        return {key: value / norm for key, value in weights.items()}

    def encode(self, texts: Sequence[str]) -> List[Dict[int, float]]:
        return [self.encode_one(text) for text in texts]


def _resolve_local_model(model_name_or_path: Optional[str]) -> Optional[str]:
    """把模型名解析成可用路径：显式路径优先，其次 models/ 下的本地目录。"""
    search_roots = [Path("."), Path(__file__).resolve().parent.parent]
    if model_name_or_path:
        candidate = Path(model_name_or_path)
        if candidate.exists():
            return str(candidate)
        # 传的是 repo id（BAAI/bge-m3）时，先看本地有没有同名目录
        leaf = model_name_or_path.rstrip("/").split("/")[-1]
        for root in search_roots:
            local = root / "models" / leaf
            if local.exists():
                return str(local)
        return model_name_or_path  # 交给 sentence_transformers 自己去下载/报错
    for root in search_roots:
        for rel in DEFAULT_DENSE_MODEL_DIRS:
            local = root / rel
            if local.exists():
                return str(local)
    return None


class DenseIndex(BaseIndex):
    """语义召回。三种后端结果格式完全一致，靠 backend 属性区分实际路径。

    backend 取值：
      sentence_transformers+faiss / sentence_transformers+numpy /
      sentence_transformers+python / hashing+sparse
    """

    name = "dense"

    def __init__(self, model_name_or_path: Optional[str] = None, device: str = "cpu", batch_size: int = 32):
        BaseIndex.__init__(self)
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.backend = "hashing+sparse"
        self.model = None
        self._embedder: Optional[_HashingEmbedder] = None
        self._dense_vectors = None          # numpy 矩阵或 list[list[float]]
        self._sparse_vectors: List[Dict[int, float]] = []
        self._inverted: Dict[int, List[Tuple[int, float]]] = {}
        self._faiss_index = None
        self._chunk_ids: List[str] = []
        self._dim = 0

    # -- 后端初始化 ---------------------------------------------------------
    def _init_backend(self) -> None:
        """尝试加载 sentence_transformers，失败一律降级到哈希向量。"""
        if self.model is not None or self._embedder is not None:
            return
        resolved = _resolve_local_model(self.model_name_or_path)
        if resolved and _HAS_NUMPY:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self.model = SentenceTransformer(resolved, device=self.device)
                self.backend = "sentence_transformers+faiss" if _HAS_FAISS else "sentence_transformers+numpy"
                logger.info("DenseIndex 使用 %s (model=%s)", self.backend, resolved)
                return
            except Exception as exc:
                logger.warning("加载 sentence_transformers 失败(model=%s)，降级为哈希向量: %s", resolved, exc)
        elif resolved and not _HAS_NUMPY:
            logger.warning("numpy 不可用，sentence_transformers 无法使用，降级为哈希向量")
        else:
            logger.warning("未找到本地 embedding 模型(%s)，降级为哈希向量", ", ".join(DEFAULT_DENSE_MODEL_DIRS))
        self._embedder = _HashingEmbedder()
        self.backend = "hashing+sparse"

    @property
    def is_dense(self) -> bool:
        return self.model is not None

    # -- 构建 ---------------------------------------------------------------
    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = list(chunks)
        self._chunk_ids = [c.chunk_id for c in self.chunks]
        if not self.chunks:
            logger.warning("DenseIndex.build 收到空 chunk 列表")
            return
        self._init_backend()
        texts = [self._index_text(c) for c in self.chunks]
        if self.is_dense:
            vectors = self._encode_dense(texts)
            self._install_dense(vectors)
        else:
            self._sparse_vectors = self._embedder.encode(texts)  # type: ignore[union-attr]
            self._build_inverted()
        logger.info("DenseIndex 构建完成: backend=%s size=%d", self.backend, self.size)

    def _encode_dense(self, texts: Sequence[str]):
        embeddings = self.model.encode(  # type: ignore[union-attr]
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,   # 归一化后点积即余弦，faiss IndexFlatIP 才成立
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _install_dense(self, vectors) -> None:
        """把稠密向量装进检索结构：优先 faiss，其次 numpy，最后纯 Python。"""
        self._dense_vectors = vectors
        self._dim = len(vectors[0]) if len(vectors) else 0
        self._faiss_index = None
        if _HAS_FAISS and _HAS_NUMPY and self._dim:
            try:
                index = faiss.IndexFlatIP(self._dim)
                index.add(np.ascontiguousarray(vectors, dtype=np.float32))
                self._faiss_index = index
                self.backend = "sentence_transformers+faiss" if self.is_dense else "cached+faiss"
                return
            except Exception as exc:
                logger.warning("faiss 建索引失败，退回 numpy 点积: %s", exc)
        if _HAS_NUMPY:
            self.backend = "sentence_transformers+numpy" if self.is_dense else "cached+numpy"
        else:
            self.backend = "sentence_transformers+python" if self.is_dense else "cached+python"

    def _build_inverted(self) -> None:
        inverted: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for idx, vector in enumerate(self._sparse_vectors):
            for feature, weight in vector.items():
                inverted[feature].append((idx, weight))
        self._inverted = dict(inverted)

    # -- 检索 ---------------------------------------------------------------
    def search(self, query: str, topk: int) -> List[Tuple[int, float]]:
        if not self.chunks or topk <= 0:
            return []
        query = normalize_text(query)
        if not query:
            return []
        if self.is_dense or self._dense_vectors is not None:
            pairs = self._search_dense(query, topk)
        else:
            pairs = self._search_sparse(query, topk)
        return _minmax(pairs)

    def _embed_query(self, query: str):
        if self.is_dense:
            vector = self.model.encode(  # type: ignore[union-attr]
                [query], batch_size=1, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
            return np.asarray(vector, dtype=np.float32)[0]
        return None

    def _search_dense(self, query: str, topk: int) -> List[Tuple[int, float]]:
        vector = self._embed_query(query)
        if vector is None:
            # 缓存装载了稠密向量但模型不在，无法编码 query
            logger.warning("DenseIndex 缺少可用模型，无法编码查询")
            return []
        limit = min(topk, self.size)
        if self._faiss_index is not None:
            scores, ids = self._faiss_index.search(
                np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32), limit
            )
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if int(i) >= 0]
        if _HAS_NUMPY and isinstance(self._dense_vectors, np.ndarray):
            scores = self._dense_vectors.dot(vector)
            order = np.argsort(-scores)[:limit]
            return [(int(i), float(scores[int(i)])) for i in order]
        # 纯 Python 兜底（只会在缓存 + 无 numpy 的极端组合下出现）
        scored = []
        for idx, row in enumerate(self._dense_vectors or []):
            scored.append((idx, sum(a * b for a, b in zip(row, vector))))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _search_sparse(self, query: str, topk: int) -> List[Tuple[int, float]]:
        if self._embedder is None:
            self._init_backend()
        query_vector = self._embedder.encode_one(query)  # type: ignore[union-attr]
        if not query_vector:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for feature, q_weight in query_vector.items():
            for idx, d_weight in self._inverted.get(feature, ()):  # 倒排只碰命中的 chunk
                scores[idx] += q_weight * d_weight
        pairs = [(idx, score) for idx, score in scores.items() if score > 0]
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs[:topk]

    # -- 缓存 ---------------------------------------------------------------
    def save(self, dirpath: str) -> str:
        """把向量和 chunk_id 一起落盘。

        chunk_id 必须存：只存向量的话，语料改了而缓存没删，检索会静默地
        返回上一版语料的内容——这种 bug 没有任何报错，靠人眼发现不了。
        """
        out = Path(dirpath)
        out.mkdir(parents=True, exist_ok=True)
        dense = self._dense_vectors is not None
        if dense and _HAS_NUMPY and isinstance(self._dense_vectors, np.ndarray):
            vector_file = "vectors.npy"
            np.save(str(out / vector_file), self._dense_vectors)
        elif dense:
            vector_file = "vectors.jsonl"
            with (out / vector_file).open("w", encoding="utf-8") as handle:
                for row in self._dense_vectors:
                    handle.write(json.dumps([float(v) for v in row]) + "\n")
        else:
            vector_file = "vectors.jsonl"
            with (out / vector_file).open("w", encoding="utf-8") as handle:
                for vector in self._sparse_vectors:
                    handle.write(json.dumps({str(k): v for k, v in vector.items()}) + "\n")
        meta = {
            "backend": self.backend,
            "dense": dense,
            "dim": self._dim,
            "model": self.model_name_or_path or "",
            "hash_dims": self._embedder.dims if self._embedder is not None else 0,
            "vector_file": vector_file,
            "chunk_ids": self._chunk_ids,
            "fingerprint": stable_id(*self._chunk_ids),
        }
        with (out / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False)
        logger.info("DenseIndex 缓存已写入 %s (%d 条)", out, len(self._chunk_ids))
        return str(out)

    def load(self, dirpath: str, chunks: Optional[List[Chunk]] = None) -> bool:
        """装载缓存。chunk_id 列表不一致时拒绝加载并返回 False，由调用方重建。"""
        out = Path(dirpath)
        meta_path = out / "meta.json"
        if not meta_path.exists():
            logger.info("DenseIndex 缓存不存在: %s", meta_path)
            return False
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception as exc:
            logger.warning("DenseIndex 缓存 meta 损坏，忽略: %s", exc)
            return False

        target = list(chunks) if chunks is not None else self.chunks
        if target:
            cached_ids = list(meta.get("chunk_ids") or [])
            current_ids = [c.chunk_id for c in target]
            if cached_ids != current_ids:
                logger.warning(
                    "DenseIndex 缓存与当前语料不匹配(缓存 %d 条 / 当前 %d 条)，拒绝加载并重建",
                    len(cached_ids), len(current_ids),
                )
                return False
            self.chunks = target
            self._chunk_ids = current_ids

        self._init_backend()
        cached_dense = bool(meta.get("dense"))
        if cached_dense != self.is_dense:
            # 缓存是哈希向量但现在有模型了（或反过来），两种向量空间不可混用
            logger.warning("DenseIndex 缓存后端(dense=%s)与当前(dense=%s)不一致，拒绝加载并重建",
                           cached_dense, self.is_dense)
            return False

        vector_file = out / str(meta.get("vector_file") or "vectors.jsonl")
        if not vector_file.exists():
            logger.warning("DenseIndex 缓存向量文件缺失: %s", vector_file)
            return False
        try:
            if cached_dense:
                if vector_file.suffix == ".npy":
                    if not _HAS_NUMPY:
                        logger.warning("缓存为 .npy 但 numpy 不可用，拒绝加载并重建")
                        return False
                    vectors = np.load(str(vector_file))
                else:
                    rows = []
                    with vector_file.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if line:
                                rows.append(json.loads(line))
                    vectors = np.asarray(rows, dtype=np.float32) if _HAS_NUMPY else rows
                self._install_dense(vectors)
                self._sparse_vectors = []
            else:
                sparse: List[Dict[int, float]] = []
                with vector_file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            sparse.append({int(k): float(v) for k, v in json.loads(line).items()})
                self._sparse_vectors = sparse
                self._dense_vectors = None
                self._build_inverted()
        except Exception as exc:
            logger.warning("DenseIndex 缓存加载失败，改为重建: %s", exc)
            return False
        logger.info("DenseIndex 缓存加载成功: %s (%d 条)", out, self.size)
        return True

    def build_or_load(self, chunks: List[Chunk], cache_dir: Optional[str] = None) -> None:
        """有可用缓存就装载，否则重建并写回。给 pipeline 用的便捷方法。"""
        if cache_dir and self.load(cache_dir, chunks=chunks):
            return
        self.build(chunks)
        if cache_dir:
            try:
                self.save(cache_dir)
            except Exception as exc:
                logger.warning("DenseIndex 缓存写入失败(不影响检索): %s", exc)


# --------------------------------------------------------------------------
# BM25Index
# --------------------------------------------------------------------------
class BM25Index(BaseIndex):
    """词面召回。rank_bm25 不可用时自实现 BM25(k1=1.5, b=0.75)。

    backend 取值：rank_bm25 / builtin_bm25
    """

    name = "bm25"

    def __init__(self):
        BaseIndex.__init__(self)
        self.backend = "builtin_bm25"
        self.k1 = BM25_K1
        self.b = BM25_B
        self._bm25 = None                                   # rank_bm25 对象
        self._inverted: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
        self._idf: Dict[str, float] = {}
        self._doc_len: List[int] = []
        self._avgdl = 0.0
        self._corpus_tokens: List[List[str]] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            logger.warning("BM25Index.build 收到空 chunk 列表")
            return
        self._corpus_tokens = [tokenize(self._index_text(c)) for c in self.chunks]
        self._doc_len = [len(tokens) for tokens in self._corpus_tokens]
        total = sum(self._doc_len)
        self._avgdl = float(total) / max(1, len(self._doc_len))

        if _HAS_RANK_BM25:
            try:
                self._bm25 = BM25Okapi(self._corpus_tokens, k1=self.k1, b=self.b)
                self.backend = "rank_bm25"
                logger.info("BM25Index 使用 rank_bm25 (tokenizer=%s, size=%d)", TOKENIZER_BACKEND, self.size)
                return
            except Exception as exc:
                logger.warning("rank_bm25 初始化失败，改用自实现 BM25: %s", exc)
        else:
            logger.warning("rank_bm25 不可用，使用自实现 BM25(k1=%.1f, b=%.2f)", self.k1, self.b)

        self._build_builtin()
        self.backend = "builtin_bm25"
        logger.info("BM25Index 构建完成: backend=%s tokenizer=%s size=%d",
                    self.backend, TOKENIZER_BACKEND, self.size)

    def _build_builtin(self) -> None:
        inverted: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for idx, tokens in enumerate(self._corpus_tokens):
            freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                freq[token] += 1
            for token, tf in freq.items():
                inverted[token].append((idx, tf))
        self._inverted = dict(inverted)
        n_docs = len(self._corpus_tokens)
        # Robertson/Sparck-Jones idf 的平滑版本：加 1 避免高频词得到负 idf，
        # 负 idf 会让"含这个词反而扣分"，在医疗语料里很容易踩到（如"患者"）。
        self._idf = {
            token: math.log(1.0 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for token, postings in self._inverted.items()
        }

    def search(self, query: str, topk: int) -> List[Tuple[int, float]]:
        if not self.chunks or topk <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        if self._bm25 is not None:
            pairs = self._search_rank_bm25(tokens, topk)
        else:
            pairs = self._search_builtin(tokens, topk)
        return _minmax(pairs)

    def _search_rank_bm25(self, tokens: List[str], topk: int) -> List[Tuple[int, float]]:
        scores = self._bm25.get_scores(tokens)  # type: ignore[union-attr]
        pairs = [(idx, float(score)) for idx, score in enumerate(scores) if score > 0]
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs[:topk]

    def _search_builtin(self, tokens: List[str], topk: int) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = defaultdict(float)
        for token in tokens:
            postings = self._inverted.get(token)
            if not postings:
                continue
            idf = self._idf.get(token, 0.0)
            for doc_idx, tf in postings:  # 只遍历命中文档，不是全量打分
                dl = self._doc_len[doc_idx] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self._avgdl or 1.0))
                scores[doc_idx] += idf * tf * (self.k1 + 1.0) / denom
        pairs = [(idx, score) for idx, score in scores.items() if score > 0]
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs[:topk]


# --------------------------------------------------------------------------
# TermIndex
# --------------------------------------------------------------------------
class TermIndex(BaseIndex):
    """术语召回：把 chunk 与 query 都映射到规范术语集合，按 idf 加权交集打分。

    这一路解决的是"同义改写"：用户问"心梗吃什么药"，指南里写的是
    "急性ST段抬高型心肌梗死"，dense 可能召回、bm25 一定召不回，
    而术语归一后两边都是"心肌梗死"，稳定命中。
    """

    name = "term"

    def __init__(self, terminology=None):
        BaseIndex.__init__(self)
        self.terminology = terminology if terminology is not None else _default_terminology()
        self.backend = type(self.terminology).__name__
        self._inverted: Dict[str, List[int]] = {}
        self._doc_terms: List[List[str]] = []
        self._idf: Dict[str, float] = {}

    def _terms_of(self, text: str) -> List[str]:
        try:
            terms = self.terminology.terms_in(text) or []
        except Exception as exc:
            logger.warning("Terminology.terms_in 失败，本条按无术语处理: %s", exc)
            return []
        out: List[str] = []
        for term in terms:
            term = normalize_text(term)
            if term and term not in out:
                out.append(term)
        return out

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            logger.warning("TermIndex.build 收到空 chunk 列表")
            return
        inverted: Dict[str, List[int]] = defaultdict(list)
        self._doc_terms = []
        for idx, chunk in enumerate(self.chunks):
            terms = self._terms_of(self._index_text(chunk))
            self._doc_terms.append(terms)
            for term in terms:
                inverted[term].append(idx)
        self._inverted = dict(inverted)
        n_docs = len(self.chunks)
        self._idf = {
            term: math.log(1.0 + (n_docs - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self._inverted.items()
        }
        covered = sum(1 for terms in self._doc_terms if terms)
        logger.info("TermIndex 构建完成: backend=%s size=%d 命中术语的 chunk=%d 术语数=%d",
                    self.backend, self.size, covered, len(self._inverted))

    def search(self, query: str, topk: int) -> List[Tuple[int, float]]:
        if not self.chunks or topk <= 0:
            return []
        query_terms = self._terms_of(query)
        if not query_terms:
            return []
        # 未登录术语给一个保底 idf，否则整条 query 的分母可能为 0
        default_idf = math.log(1.0 + len(self.chunks))
        query_weight = sum(self._idf.get(term, default_idf) for term in query_terms) or 1.0
        scores: Dict[int, float] = defaultdict(float)
        for term in query_terms:
            idf = self._idf.get(term, default_idf)
            for doc_idx in self._inverted.get(term, ()):  # 术语倒排
                scores[doc_idx] += idf
        pairs = []
        for doc_idx, hit_weight in scores.items():
            doc_terms = self._doc_terms[doc_idx]
            # 除以 query 权重得到覆盖率，再对 chunk 术语数做轻微惩罚：
            # 否则一个术语堆砌的长 chunk 会命中所有 query。
            coverage = hit_weight / query_weight
            penalty = 1.0 / (1.0 + math.log(1.0 + max(0, len(doc_terms) - len(query_terms))))
            pairs.append((doc_idx, coverage * penalty))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return _minmax(pairs[:topk])


# --------------------------------------------------------------------------
# 组装
# --------------------------------------------------------------------------
def build_indexes(
    chunks: List[Chunk],
    dense_model: Optional[str] = None,
    terminology=None,
    routes: Sequence[str] = ("dense", "bm25", "term"),
    device: str = "cpu",
) -> Dict[str, BaseIndex]:
    """按 routes 构建索引字典。

    单路失败只 warning 并跳过：三路里挂一路仍然能检索（RRF 本来就容忍缺路），
    直接抛异常会让整个问答服务不可用，这个代价在生产上不可接受。
    """
    indexes: Dict[str, BaseIndex] = {}
    for route in routes:
        try:
            if route == "dense":
                index: BaseIndex = DenseIndex(model_name_or_path=dense_model, device=device)
            elif route == "bm25":
                index = BM25Index()
            elif route == "term":
                index = TermIndex(terminology=terminology)
            else:
                logger.warning("未知召回路 %s，已跳过", route)
                continue
            index.build(chunks)
            indexes[route] = index
        except Exception as exc:
            logger.warning("构建 %s 索引失败，已跳过该路: %s", route, exc)
    if not indexes:
        logger.warning("所有召回路均构建失败，检索将返回空结果")
    return indexes


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def _demo_chunks() -> List[Chunk]:
    """造几条带真实医疗特征的假 chunk：中文术语、英文药名、ICD 编码、剂量。"""
    raw = [
        ("二甲双胍是2型糖尿病一线降糖药物，通常起始剂量 500mg 每日两次，随餐服用可减轻胃肠道反应。",
         "中国2型糖尿病防治指南", TIER_GUIDELINE, ["第四章 口服降糖药", "双胍类"]),
        ("Metformin should be withheld before iodinated contrast administration in patients with eGFR below 30.",
         "Metformin Drug Label", TIER_DRUG_LABEL, ["Warnings", "Renal impairment"]),
        ("急性ST段抬高型心肌梗死（ICD-10 编码 I21.0）确诊后应在 120 分钟内完成直接 PCI。",
         "急性心肌梗死诊疗规范", TIER_GUIDELINE, ["第二章 再灌注治疗"]),
        ("高血压患者的初始降压目标一般为收缩压低于 140mmHg，合并糖尿病者可考虑 130mmHg。",
         "中国高血压防治指南", TIER_GUIDELINE, ["第五章 降压目标"]),
        ("阿司匹林肠溶片用于心血管事件二级预防，常用维持剂量 75-100mg 每日一次，消化道出血高风险者需联用 PPI。",
         "临床药物治疗学", TIER_TEXTBOOK, ["抗血小板药物"]),
        ("低血糖是降糖治疗最常见的不良反应，表现为心悸、出汗、手抖，血糖低于 3.9mmol/L 即应处理。",
         "内科学", TIER_TEXTBOOK, ["内分泌系统", "糖尿病并发症"]),
    ]
    chunks = []
    for idx, (text, title, tier, section) in enumerate(raw):
        chunks.append(
            Chunk(
                chunk_id=stable_id(title, idx, text),
                text=text,
                doc_id=stable_id(title),
                doc_title=title,
                source_tier=tier,
                parent_text=text,
                section_path=section,
                page=idx + 1,
            )
        )
    return chunks


def _print_results(route: str, index: BaseIndex, query: str, topk: int) -> None:
    print("\n[%s] backend=%s" % (route, index.backend))
    hits = index.search(query, topk)
    if not hits:
        print("  (无结果)")
        return
    for rank, (idx, score) in enumerate(hits, start=1):
        chunk = index.chunks[idx]
        preview = chunk.text[:46].replace("\n", " ")
        print("  %d. score=%.4f | %s | %s" % (rank, score, chunk.doc_title, preview))


def main():
    parser = argparse.ArgumentParser(description="medical_rag 三路索引自检 / 离线检索")
    parser.add_argument("--chunks", default="", help="chunk JSONL 路径，留空则用内置示例数据")
    parser.add_argument("--query", default="二甲双胍怎么吃，会不会低血糖", help="检索查询")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--routes", default="dense,bm25,term")
    parser.add_argument("--dense_model", default="", help="本地 embedding 模型目录，留空则自动探测")
    parser.add_argument("--cache_dir", default="", help="DenseIndex 向量缓存目录，留空则不缓存")
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")

    if args.chunks and os.path.exists(args.chunks):
        try:
            from .schema import load_chunks
        except ImportError:
            from medical_rag.schema import load_chunks  # type: ignore
        chunks = load_chunks(args.chunks, max_rows=args.max_rows)
    else:
        if args.chunks:
            logger.warning("找不到 %s，改用内置示例数据", args.chunks)
        chunks = _demo_chunks()

    routes = tuple(r.strip() for r in args.routes.split(",") if r.strip())
    print("语料 %d 条 | 分词后端=%s | numpy=%s faiss=%s jieba=%s rank_bm25=%s"
          % (len(chunks), TOKENIZER_BACKEND, _HAS_NUMPY, _HAS_FAISS, _HAS_JIEBA, _HAS_RANK_BM25))
    indexes = build_indexes(chunks, dense_model=args.dense_model or None, routes=routes)
    print("查询: %s" % args.query)
    for route in routes:
        index = indexes.get(route)
        if index is None:
            print("\n[%s] 构建失败或未启用" % route)
            continue
        _print_results(route, index, args.query, args.topk)

    if args.cache_dir and "dense" in indexes and isinstance(indexes["dense"], DenseIndex):
        dense = indexes["dense"]
        dense.save(args.cache_dir)
        probe = DenseIndex(model_name_or_path=args.dense_model or None)
        print("\n缓存回读(同语料): %s" % probe.load(args.cache_dir, chunks=chunks))
        print("缓存回读(改语料): %s" % probe.load(args.cache_dir, chunks=chunks[:-1]))


if __name__ == "__main__":
    main()
