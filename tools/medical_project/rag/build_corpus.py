# -*- coding: utf-8 -*-
"""分层医疗语料构建：多来源 -> 统一的 chunks.jsonl。

医疗 RAG 的上限由语料质量决定，而不是由检索算法决定。把临床指南、药品说明书、
医学百科和在线问诊 QA 一股脑塞进同一个索引里，检索器无法区分"权威结论"和
"网友回答"，一旦附上引用，错误答案反而更有说服力。

所以本脚本给每一条来源打上 ``source_tier``（见 medical_rag.schema 的 TIER_*），
下游 rerank 用 ``TIER_AUTHORITY`` 做权威度加权，答案里的出处也会带上分级。

三条处理流水线::

    指南/说明书目录 (PDF/DOCX/MD/TXT) --load_document--> RawDoc --+
    百科 / 知识图谱 QA jsonl          --load_qa_jsonl-->  RawDoc --+--> chunk_documents --> Chunk
    在线问诊 QA jsonl (默认不入库)    --load_qa_jsonl-->  RawDoc --+

关于 L3 在线问诊
-------------------------------------------------
``--consultation_jsonl`` 默认**不会**进入主库，必须显式加 ``--include_consultation``。
这不是保守，而是因为把未经医学审核的问答当作权威证据召回，等于用引用格式给错误
背书——没有 RAG 还好，有了 RAG 反而更危险。它更适合做检索评测的负样本或对照组。

用法::

    python tools/medical_project/rag/build_corpus.py \
        --guideline_dir data_raw/medical_project/guidelines \
        --drug_label_dir data_raw/medical_project/drug_labels \
        --encyclopedia_jsonl data_raw/medical_project/huatuo_encyclopedia_qa.jsonl \
        --max_encyclopedia 20000 \
        --output data_processed/medical_project/rag/chunks.jsonl
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("build_corpus")


# --------------------------------------------------------------------------
# 仓库根目录自举
# --------------------------------------------------------------------------
# 本脚本在 tools/medical_project/rag/ 下，直接 `python tools/.../build_corpus.py`
# 运行时 sys.path[0] 是脚本所在目录，`import medical_rag` 会失败。这里把仓库根
# 目录补进去，让脚本既能被当模块 import，也能直接当命令行工具跑。
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------
# 依赖导入（失败也不能让本模块 import 不进来）
# --------------------------------------------------------------------------
# medical_rag.schema 零依赖，parsers 里的 pymupdf/pdfplumber/docx 都是可选的、
# 内部已做降级。即便如此仍然兜一层：只要本模块能 import 成功，自检脚本和上层
# 编排就能拿到明确的错误信息，而不是一个看不出所以然的 ImportError 堆栈。
_IMPORT_ERROR = ""  # type: str
try:
    from medical_rag.chunker import chunk_documents
    from medical_rag.parsers import load_document, load_qa_jsonl
    from medical_rag.schema import (
        TIER_AUTHORITY,
        TIER_CONSULTATION,
        TIER_DRUG_LABEL,
        TIER_ENCYCLOPEDIA,
        TIER_GUIDELINE,
        TIER_KNOWLEDGE_GRAPH,
        Chunk,
        RawDoc,
        normalize_text,
        save_chunks,
    )
except Exception as exc:  # pragma: no cover - 只在 medical_rag 不完整时触发
    _IMPORT_ERROR = str(exc)
    logger.warning("导入 medical_rag 失败，build_corpus 只能被 import 不能运行: %s", exc)

    # 占位常量，保证模块级定义（SOURCE_SPECS 等）不会连带崩掉。
    TIER_GUIDELINE = "L1_guideline"
    TIER_DRUG_LABEL = "L1_drug_label"
    TIER_ENCYCLOPEDIA = "L2_encyclopedia"
    TIER_KNOWLEDGE_GRAPH = "L2_knowledge_graph"
    TIER_CONSULTATION = "L3_consultation"
    TIER_AUTHORITY = {
        TIER_GUIDELINE: 1.00,
        TIER_DRUG_LABEL: 0.95,
        TIER_ENCYCLOPEDIA: 0.60,
        TIER_KNOWLEDGE_GRAPH: 0.55,
        TIER_CONSULTATION: 0.30,
    }
    Chunk = None
    RawDoc = None
    chunk_documents = None
    load_document = None
    load_qa_jsonl = None
    save_chunks = None

    def normalize_text(text):
        """降级版：只在 medical_rag 缺失时用于去重 key，不参与正式产出。"""
        if text is None:
            return ""
        return re.sub(r"[ \t　]+", " ", str(text)).strip()


# --------------------------------------------------------------------------
# 来源定义
# --------------------------------------------------------------------------
# kind="doc"  -> 目录/单文件，走 load_document，max_* 限制的是"文档篇数"
# kind="qa"   -> QA jsonl，走 load_qa_jsonl，max_* 限制的是"问答条数"
SourceSpec = namedtuple("SourceSpec", ["name", "tier", "kind", "arg", "max_arg", "label"])

# 顺序即去重优先级：先处理的来源占住去重槽位。按权威度从高到低排，
# 这样同一段文字在指南和问诊里都出现时，留下的是指南那一份。
SOURCE_SPECS = (
    SourceSpec("guideline", TIER_GUIDELINE, "doc", "guideline_dir", "max_guideline", "临床指南/专家共识"),
    SourceSpec("drug_label", TIER_DRUG_LABEL, "doc", "drug_label_dir", "max_drug_label", "药品说明书"),
    SourceSpec("encyclopedia", TIER_ENCYCLOPEDIA, "qa", "encyclopedia_jsonl", "max_encyclopedia", "医学百科 QA"),
    SourceSpec(
        "knowledge_graph", TIER_KNOWLEDGE_GRAPH, "qa", "knowledge_graph_jsonl",
        "max_knowledge_graph", "医学知识图谱 QA",
    ),
    SourceSpec(
        "consultation", TIER_CONSULTATION, "qa", "consultation_jsonl",
        "max_consultation", "在线问诊 QA（未经医学审核）",
    ),
)

# 文档目录里认这些后缀；.jsonl/.json 不在内，它们属于 QA 来源，
# 混进来会被当成纯文本解析出一堆 JSON 字面量噪声。
DOC_SUFFIXES = (".pdf", ".docx", ".md", ".markdown", ".mdown", ".txt", ".text")
QA_SUFFIXES = (".jsonl", ".json")

CONSULTATION_WARNING = (
    "警告：本次构建已把 L3_consultation（在线问诊 QA）纳入主检索库。"
    "这类问答未经医学审核，被检索并附上引用后，错误内容会显得更可信，"
    "风险高于不使用 RAG。仅建议用于离线评测/对照实验，不要用于对外问答。"
)

CONSULTATION_SKIPPED_NOTE = (
    "提示：已提供 --consultation_jsonl 但未加 --include_consultation，"
    "在线问诊 QA 未纳入主库（这是默认且推荐的行为）。"
)


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def _require_medical_rag() -> None:
    """运行期才检查依赖，保证 import 本模块永远成功。"""
    if _IMPORT_ERROR:
        raise RuntimeError(
            "medical_rag 不可用，无法构建语料：%s（请在仓库根目录运行）" % _IMPORT_ERROR
        )


def dedup_key(text: str) -> str:
    """去重 key：归一化 + 去掉全部空白后的 sha1。

    只 strip 空格是不够的——同一段话从 PDF 两栏排版和从 DOCX 取出来，换行位置
    不一样但内容相同，按原文哈希会当成两条。所以先抹掉所有空白再哈希。
    """
    compact = re.sub(r"\s+", "", normalize_text(text))
    return hashlib.sha1(compact.encode("utf-8", errors="ignore")).hexdigest()


def guess_version(path: str) -> str:
    """从文件名里猜版本号，如 "2023版" / "第9版"。

    指南的版本是引用的一部分：引 2016 版的降压目标去回答今天的问题就是错的。
    文件名里通常已经写了年份，顺手提取，避免每次都靠 --version_tag 手工指定。
    """
    name = Path(path).stem
    match = re.search(r"第\s*([0-9一二三四五六七八九十]+)\s*版", name)
    if match:
        return "第%s版" % match.group(1)
    match = re.search(r"(19|20)\d{2}", name)
    if match:
        return "%s版" % match.group(0)
    return ""


def iter_doc_files(input_path: str, recursive: bool = True) -> List[Path]:
    """收集文档来源里的可解析文件；输入是单文件就返回它自己。"""
    target = Path(input_path)
    if target.is_file():
        return [target]
    if not target.is_dir():
        logger.warning("路径不存在，已跳过: %s", input_path)
        return []
    pattern = "**/*" if recursive else "*"
    files = [p for p in sorted(target.glob(pattern)) if p.is_file()]
    return [p for p in files if p.suffix.lower() in DOC_SUFFIXES]


def iter_qa_files(input_path: str) -> List[Path]:
    """QA 来源允许直接给 jsonl 文件，也允许给一个装着 jsonl 的目录。"""
    target = Path(input_path)
    if target.is_file():
        return [target]
    if not target.is_dir():
        logger.warning("路径不存在，已跳过: %s", input_path)
        return []
    return [p for p in sorted(target.glob("*")) if p.is_file() and p.suffix.lower() in QA_SUFFIXES]


# --------------------------------------------------------------------------
# 载入 -> 切块
# --------------------------------------------------------------------------
def load_doc_source(
    input_path: str,
    source_tier: str,
    max_docs: int = -1,
    version_tag: str = "",
    recursive: bool = True,
) -> List["RawDoc"]:
    """文档目录/单文件 -> RawDoc 列表。

    load_document 自身不抛异常（坏文件返回空 blocks），这里只负责按 max_docs
    截断并丢掉解析为空的文档：空文档进切块器只会产出 0 个 chunk，留着徒增噪声。
    """
    _require_medical_rag()
    docs = []  # type: List[Any]
    files = iter_doc_files(input_path, recursive=recursive)
    if max_docs > 0:
        files = files[:max_docs]
    for file_path in files:
        version = version_tag or guess_version(str(file_path))
        doc = load_document(str(file_path), source_tier, version=version)
        if not doc.blocks:
            continue
        docs.append(doc)
    logger.info("文档来源 %s: %d 个文件 -> %d 篇有效文档", input_path, len(files), len(docs))
    return docs


def load_qa_source(
    input_path: str,
    source_tier: str,
    max_rows: int = -1,
    question_field: str = "question",
    answer_field: str = "answer",
    title_field: Optional[str] = None,
) -> List["RawDoc"]:
    """QA jsonl -> RawDoc 列表（一条 QA 一篇 doc）。

    max_rows 是整个来源的总预算而非每个文件的配额，否则给目录时上限会被悄悄
    放大成"文件数 × max_rows"。
    """
    _require_medical_rag()
    docs = []  # type: List[Any]
    remaining = max_rows
    for file_path in iter_qa_files(input_path):
        if max_rows > 0 and remaining <= 0:
            break
        part = load_qa_jsonl(
            str(file_path),
            source_tier,
            question_field=question_field,
            answer_field=answer_field,
            title_field=title_field,
            max_rows=remaining if max_rows > 0 else -1,
        )
        docs.extend(part)
        if max_rows > 0:
            remaining -= len(part)
    logger.info("QA 来源 %s: %d 条问答", input_path, len(docs))
    return docs


def deduplicate(chunks: List["Chunk"], seen: Optional[Dict[str, str]] = None) -> Tuple[List["Chunk"], int]:
    """按归一化文本 sha1 去重；``seen`` 跨来源复用以实现全局去重。

    返回 (保留的 chunks, 丢弃条数)。seen 的值存的是首次命中的 chunk_id，
    方便排查"这条去哪了"。
    """
    if seen is None:
        seen = {}
    kept = []  # type: List[Any]
    dropped = 0
    for chunk in chunks:
        key = dedup_key(chunk.text)
        if not key:
            dropped += 1
            continue
        if key in seen:
            dropped += 1
            continue
        seen[key] = chunk.chunk_id
        kept.append(chunk)
    return kept, dropped


def build_source_chunks(spec: SourceSpec, args: argparse.Namespace) -> Tuple[List["Chunk"], Dict[str, Any]]:
    """处理单个来源：载入 -> 切块，返回 (chunks, 该来源的统计)。"""
    input_path = getattr(args, spec.arg, None)
    max_items = int(getattr(args, spec.max_arg, -1) or -1)
    stat = {
        "source": spec.name,
        "label": spec.label,
        "source_tier": spec.tier,
        "authority": TIER_AUTHORITY.get(spec.tier, 0.3),
        "input_path": input_path or "",
        "max_items": max_items,
        "docs": 0,
        "chunks_before_dedup": 0,
        "duplicates_dropped": 0,
        "chunks": 0,
    }
    if not input_path:
        return [], stat

    if spec.kind == "doc":
        docs = load_doc_source(
            input_path,
            spec.tier,
            max_docs=max_items,
            version_tag=args.version_tag,
            recursive=not args.no_recursive,
        )
    else:
        docs = load_qa_source(
            input_path,
            spec.tier,
            max_rows=max_items,
            question_field=args.question_field,
            answer_field=args.answer_field,
            title_field=args.title_field,
        )

    stat["docs"] = len(docs)
    if not docs:
        return [], stat

    chunks = chunk_documents(
        docs,
        child_chars=args.child_chars,
        child_overlap=args.child_overlap,
        parent_chars=args.parent_chars,
        min_chars=args.min_chars,
    )
    # 记下来源名，summary 里按来源统计要靠它（source_tier 不够用：
    # 百科和知识图谱可能共用一个 tier 区间，但排查问题时要能分开看）。
    for chunk in chunks:
        chunk.extra["corpus_source"] = spec.name
    stat["chunks_before_dedup"] = len(chunks)
    return chunks, stat


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def _ratio(part: int, total: int) -> float:
    return round(part / float(total), 4) if total else 0.0


def summarize_corpus(
    chunks: List["Chunk"],
    source_stats: List[Dict[str, Any]],
    args: argparse.Namespace,
    warnings: List[str],
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """语料体检报告：分布 + 去重 + 可追溯性覆盖率。

    页码覆盖率和证据等级覆盖率是医疗 RAG 的两个关键健康指标：前者决定引用能不能
    定位到原文，后者决定回答能不能说清推荐强度。低了说明解析或切块丢了信息。
    """
    total = len(chunks)
    by_tier = {}  # type: Dict[str, int]
    by_source = {}  # type: Dict[str, int]
    by_content_type = {}  # type: Dict[str, int]
    tier_chars = {}  # type: Dict[str, int]
    lengths = []  # type: List[int]
    with_page = 0
    with_evidence = 0
    with_section = 0
    with_parent = 0

    for chunk in chunks:
        length = len(chunk.text)
        lengths.append(length)
        by_tier[chunk.source_tier] = by_tier.get(chunk.source_tier, 0) + 1
        tier_chars[chunk.source_tier] = tier_chars.get(chunk.source_tier, 0) + length
        name = str(chunk.extra.get("corpus_source") or "unknown")
        by_source[name] = by_source.get(name, 0) + 1
        by_content_type[chunk.content_type] = by_content_type.get(chunk.content_type, 0) + 1
        if chunk.page is not None:
            with_page += 1
        if chunk.evidence_level:
            with_evidence += 1
        if chunk.section_path:
            with_section += 1
        if chunk.parent_text and chunk.parent_text != chunk.text:
            with_parent += 1

    lengths.sort()
    duplicates = sum(int(s["duplicates_dropped"]) for s in source_stats)
    before = sum(int(s["chunks_before_dedup"]) for s in source_stats)

    summary = {
        "output": args.output,
        "chunks": total,
        "docs": sum(int(s["docs"]) for s in source_stats),
        "chunks_before_dedup": before,
        "duplicates_dropped": duplicates,
        "duplicate_ratio": _ratio(duplicates, before),
        "avg_chunk_chars": round(sum(lengths) / float(total), 1) if total else 0.0,
        "median_chunk_chars": lengths[len(lengths) // 2] if total else 0,
        "min_chunk_chars": lengths[0] if total else 0,
        "max_chunk_chars": lengths[-1] if total else 0,
        "with_page": with_page,
        "with_page_ratio": _ratio(with_page, total),
        "with_evidence_level": with_evidence,
        "with_evidence_level_ratio": _ratio(with_evidence, total),
        "with_section_path": with_section,
        "with_section_path_ratio": _ratio(with_section, total),
        "with_parent_context": with_parent,
        "by_source_tier": dict(sorted(by_tier.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_content_type": dict(sorted(by_content_type.items())),
        "avg_chars_by_tier": {
            tier: round(tier_chars[tier] / float(by_tier[tier]), 1) for tier in sorted(by_tier)
        },
        "include_consultation": bool(args.include_consultation),
        "chunk_params": {
            "child_chars": args.child_chars,
            "child_overlap": args.child_overlap,
            "parent_chars": args.parent_chars,
            "min_chars": args.min_chars,
        },
        "sources": source_stats,
        "warnings": warnings,
        "notes": list(notes or []),
    }
    return summary


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def build_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    """按 SOURCE_SPECS 顺序构建分层语料，写出 chunks.jsonl + corpus_summary.json。"""
    _require_medical_rag()

    configured = [spec for spec in SOURCE_SPECS if getattr(args, spec.arg, None)]
    if not configured:
        raise ValueError(
            "未指定任何语料来源，至少给一个："
            "--guideline_dir / --drug_label_dir / --encyclopedia_jsonl / "
            "--knowledge_graph_jsonl / --consultation_jsonl"
        )

    warnings = []  # type: List[str]   # 会影响答案可信度的问题，必须醒目
    notes = []  # type: List[str]       # 只是说明本次行为，不需要报警
    all_chunks = []  # type: List[Any]
    source_stats = []  # type: List[Dict[str, Any]]
    seen = {}  # type: Dict[str, str]

    for spec in SOURCE_SPECS:
        if not getattr(args, spec.arg, None):
            continue
        if spec.tier == TIER_CONSULTATION and not args.include_consultation:
            # 默认路径：读都不读，省掉 1GB 级文件的解析开销。
            notes.append(CONSULTATION_SKIPPED_NOTE)
            source_stats.append({
                "source": spec.name,
                "label": spec.label,
                "source_tier": spec.tier,
                "authority": TIER_AUTHORITY.get(spec.tier, 0.3),
                "input_path": getattr(args, spec.arg),
                "max_items": int(getattr(args, spec.max_arg, -1) or -1),
                "docs": 0,
                "chunks_before_dedup": 0,
                "duplicates_dropped": 0,
                "chunks": 0,
                "skipped": True,
                "skip_reason": "未启用 --include_consultation",
            })
            continue

        chunks, stat = build_source_chunks(spec, args)
        kept, dropped = deduplicate(chunks, seen=seen)
        stat["duplicates_dropped"] = dropped
        stat["chunks"] = len(kept)
        source_stats.append(stat)
        all_chunks.extend(kept)
        logger.info(
            "来源 %s(%s): %d 篇 -> %d 块（去重丢弃 %d）",
            spec.name, spec.tier, stat["docs"], len(kept), dropped,
        )

    if args.include_consultation and any(
        s["source_tier"] == TIER_CONSULTATION and s["chunks"] for s in source_stats
    ):
        warnings.insert(0, CONSULTATION_WARNING)

    written = save_chunks(args.output, all_chunks)
    summary = summarize_corpus(all_chunks, source_stats, args, warnings, notes)
    summary["written"] = written

    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else Path(args.output).parent / "corpus_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary["summary_output"] = str(summary_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for text in notes:
        print("")
        print(text)
    for text in warnings:
        # 警示单独再打一遍：summary 很长，滚屏之后没人会回头读 JSON 里的字段。
        print("")
        print("!" * 72)
        print(text)
        print("!" * 72)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a tiered medical RAG corpus (guidelines / drug labels / QA) into chunks.jsonl."
    )
    parser.add_argument("--guideline_dir", default=None, help="临床指南目录或单文件（PDF/DOCX/MD/TXT）-> L1_guideline")
    parser.add_argument("--drug_label_dir", default=None, help="药品说明书目录或单文件 -> L1_drug_label")
    parser.add_argument("--encyclopedia_jsonl", default=None, help="医学百科 QA jsonl（如 huatuo_encyclopedia_qa）-> L2_encyclopedia")
    parser.add_argument("--knowledge_graph_jsonl", default=None, help="知识图谱 QA jsonl（如 huatuo_knowledge_graph_qa）-> L2_knowledge_graph")
    parser.add_argument(
        "--consultation_jsonl",
        default=None,
        help="在线问诊 QA jsonl（如 data_processed/medical_project/sft/medical_candidates.jsonl）-> L3_consultation；"
             "默认不入库，需配合 --include_consultation",
    )

    parser.add_argument(
        "--include_consultation",
        action="store_true",
        help="把 L3 在线问诊 QA 纳入主库。未经医学审核的问答被当作权威证据召回会放大错误，默认关闭。",
    )

    parser.add_argument("--max_guideline", type=int, default=-1, help="最多处理多少篇指南文档，-1 不限")
    parser.add_argument("--max_drug_label", type=int, default=-1, help="最多处理多少篇说明书，-1 不限")
    parser.add_argument("--max_encyclopedia", type=int, default=-1, help="最多读多少条百科 QA，-1 不限")
    parser.add_argument("--max_knowledge_graph", type=int, default=-1, help="最多读多少条知识图谱 QA，-1 不限")
    parser.add_argument(
        "--max_consultation", type=int, default=-1,
        help="最多读多少条问诊 QA，-1 不限（该文件常有 GB 级，建议先给个小值验证）",
    )

    parser.add_argument("--question_field", default="question", help="QA jsonl 的问题字段名")
    parser.add_argument("--answer_field", default="answer", help="QA jsonl 的答案字段名")
    parser.add_argument("--title_field", default=None, help="QA jsonl 的标题字段名，缺省取问题前 40 字")
    parser.add_argument("--version_tag", default="", help="统一覆盖文档版本号；缺省从文件名猜（如 2023版）")
    parser.add_argument("--no_recursive", action="store_true", help="文档目录不递归子目录")

    parser.add_argument("--child_chars", type=int, default=320, help="子块目标长度")
    parser.add_argument("--child_overlap", type=int, default=64, help="子块重叠长度")
    parser.add_argument("--parent_chars", type=int, default=1200, help="父块目标长度")
    parser.add_argument("--min_chars", type=int, default=40, help="小于该长度的块丢弃")

    parser.add_argument("--output", required=True, help="chunks.jsonl 输出路径")
    parser.add_argument("--summary_output", default=None, help="默认写到 output 同目录的 corpus_summary.json")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_corpus(args)


if __name__ == "__main__":
    main()
