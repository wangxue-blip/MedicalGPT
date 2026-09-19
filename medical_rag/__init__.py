# -*- coding: utf-8 -*-
"""medical_rag：带来源可追溯与引用核查的医疗检索增强问答子系统。

模块分工::

    schema       数据契约（Chunk / RawDoc / RetrievalHit / 来源分级），零依赖
    parsers      PDF / DOCX / JSONL -> RawDoc
    chunker      结构感知切块，保留章节路径、页码、推荐等级
    terminology  医学术语归一与同义扩展
    index        三路召回：dense(语义) / bm25(词面) / term(术语)
    fusion       RRF 多路融合
    rerank       交叉编码器精排 + 来源权威度/时效性加权
    retriever    召回 -> 融合 -> 精排的检索门面
    generator    证据拼装、prompt 组装与 HF 生成
    verifier     逐句引用核查与证据不足拒答
    pipeline     端到端管线 MedicalRAG

快速开始::

    from medical_rag import MedicalRAG
    rag = MedicalRAG.from_jsonl("data/chunks.jsonl")
    print(rag.answer("二甲双胍怎么吃", topk=5, generate=False)["prompt"])

为什么大部分符号是惰性导出
-------------------------------------------------
本包的子模块依赖差异很大：``schema`` 零依赖，而 ``index`` / ``rerank`` 会去探测
numpy / faiss / sentence_transformers。如果 ``__init__`` 上来就把所有子模块
eager import 一遍，那么一句 ``import medical_rag.schema`` 也要付全套探测的代价，
并且任何一个子模块的问题都会污染整个包。所以这里只 eager 导入零依赖的
``schema``，其余符号走 PEP 562 的模块级 ``__getattr__``，首次访问时才真正加载。
"""

import logging
from typing import Any

# schema 不 import 任何可选依赖，可以安全地 eager 导入，
# 保证 `from medical_rag import Chunk` 在任何环境下都成立。
from medical_rag.schema import (
    ALL_TIERS,
    CONTENT_QA,
    CONTENT_TABLE,
    CONTENT_TEXT,
    TIER_AUTHORITY,
    TIER_CONSULTATION,
    TIER_DRUG_LABEL,
    TIER_ENCYCLOPEDIA,
    TIER_GUIDELINE,
    TIER_KNOWLEDGE_GRAPH,
    TIER_TEXTBOOK,
    Chunk,
    RawDoc,
    RetrievalHit,
    detect_evidence_level,
    load_chunks,
    normalize_text,
    read_jsonl,
    save_chunks,
    stable_id,
    write_jsonl,
)

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

# 惰性符号表：属性名 -> 所在子模块。
_LAZY_EXPORTS = {
    # pipeline
    "MedicalRAG": "medical_rag.pipeline",
    "REFUSAL_ANSWER": "medical_rag.pipeline",
    "to_jsonable": "medical_rag.pipeline",
    # retriever
    "MedicalRetriever": "medical_rag.retriever",
    # generator
    "HFGenerator": "medical_rag.generator",
    "build_context": "medical_rag.generator",
    "build_prompt": "medical_rag.generator",
    "format_sources": "medical_rag.generator",
    # verifier
    "CitationVerifier": "medical_rag.verifier",
    "CitationCheck": "medical_rag.verifier",
    "should_refuse": "medical_rag.verifier",
    # 检索底座
    "BaseIndex": "medical_rag.index",
    "BM25Index": "medical_rag.index",
    "DenseIndex": "medical_rag.index",
    "TermIndex": "medical_rag.index",
    "build_indexes": "medical_rag.index",
    "reciprocal_rank_fusion": "medical_rag.fusion",
    "CrossEncoderReranker": "medical_rag.rerank",
    "apply_authority_boost": "medical_rag.rerank",
    # 语料构建
    "StructureAwareChunker": "medical_rag.chunker",
    "chunk_documents": "medical_rag.chunker",
    "Terminology": "medical_rag.terminology",
    "load_document": "medical_rag.parsers",
    "load_qa_jsonl": "medical_rag.parsers",
}

__all__ = [
    # schema
    "Chunk",
    "RawDoc",
    "RetrievalHit",
    "load_chunks",
    "save_chunks",
    "read_jsonl",
    "write_jsonl",
    "normalize_text",
    "stable_id",
    "detect_evidence_level",
    "TIER_AUTHORITY",
    "TIER_GUIDELINE",
    "TIER_DRUG_LABEL",
    "TIER_TEXTBOOK",
    "TIER_ENCYCLOPEDIA",
    "TIER_KNOWLEDGE_GRAPH",
    "TIER_CONSULTATION",
    "ALL_TIERS",
    "CONTENT_TEXT",
    "CONTENT_TABLE",
    "CONTENT_QA",
    "__version__",
] + sorted(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    """PEP 562 惰性属性：首次访问时才导入对应子模块。

    子模块缺失时抛 AttributeError 而不是静默返回 None——``None`` 会让错误推迟到
    调用处才炸，堆栈里看不出是哪个模块没装好；这里把真实的导入异常带进消息里。
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))

    import importlib

    try:
        module = importlib.import_module(target)
    except Exception as exc:
        logger.warning("导入 %s 失败，medical_rag.%s 不可用: %s", target, name, exc)
        raise AttributeError(
            "medical_rag.%s 不可用：子模块 %s 导入失败(%s)" % (name, target, exc)
        )

    try:
        value = getattr(module, name)
    except AttributeError:
        raise AttributeError("子模块 %s 中没有 %r" % (target, name))

    # 缓存到包命名空间，后续访问不再走 __getattr__。
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(list(globals().keys()) + __all__))
