# -*- coding: utf-8 -*-
"""结构感知切片器（structure-aware chunking）。

`demo/chatpdf.py` 用的是固定 250 字滑窗：它不认识标题、不认识表格、也不认识
"（I类推荐）"这种把推荐强度写在句尾括号里的写法。医疗场景下这三件事都会直接
损坏答案——切掉章节路径等于丢掉"这句话是在讲哪个病的哪一节"，切碎表格等于把
剂量表拆成一堆无意义的数字，切掉推荐等级等于把"I类推荐"降格成一句普通建议。

本模块做三件事：

1. **标题栈**：按 ``heading_level`` 维护一个栈，正文 block 继承当前 ``section_path``，
   于是每个 chunk 都自带"文档 > 章 > 节"的坐标，引用时可直接回溯。
2. **父子切片（small-to-big）**：以 section 为单位聚合正文成 parent（供生成使用的
   大块上下文），parent 内部再切出带重叠的 child（供检索使用的小块）。检索要精准所
   以块小，生成要连贯所以给大块，两者分离才能同时满足。
3. **原子块保护**：表格（以及问答对）整块落成一个 chunk，绝不切碎。

本模块只依赖标准库与 :mod:`medical_rag.schema`，在任何环境下都能 import 成功。
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # 允许 `python medical_rag/chunker.py` 直接运行，而不只是作为包导入
    from medical_rag.schema import (
        CONTENT_QA,
        CONTENT_TABLE,
        CONTENT_TEXT,
        TIER_AUTHORITY,
        TIER_CONSULTATION,
        Chunk,
        RawDoc,
        detect_evidence_level,
        normalize_text,
        read_jsonl,
        save_chunks,
        stable_id,
    )
except ImportError:  # pragma: no cover - 仅在脚本直跑且未设置 PYTHONPATH 时触发
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from medical_rag.schema import (
        CONTENT_QA,
        CONTENT_TABLE,
        CONTENT_TEXT,
        TIER_AUTHORITY,
        TIER_CONSULTATION,
        Chunk,
        RawDoc,
        detect_evidence_level,
        normalize_text,
        read_jsonl,
        save_chunks,
        stable_id,
    )

logger = logging.getLogger(__name__)

# 这些 content_type 的 block 语义上不可分割：表格拆开就没有行列对应关系了，
# 问答对拆开就会出现"只有问题没有答案"的孤儿 chunk。它们整块成为一个 chunk。
ATOMIC_CONTENT_TYPES = frozenset({CONTENT_TABLE, CONTENT_QA})

# 中文断句：句末标点归属前一句，后面紧跟的右引号 / 右括号也一并带走，
# 避免出现以"）"开头的碎句。末尾无标点的残句由第二个分支兜底。
_SENTENCE_RE = re.compile(
    r"[^。！？；!?;\n]*[。！？；!?;\n]+[”’\"'）)\]】》]*"
    r"|[^。！？；!?;\n]+$"
)

# 章节序号前缀，用于兜底识别未被 parser 标成 is_heading 的短行标题。
#
# 序号标点（、．)）等）是必需的，不能可选：写成可选时 "数字开头 + 任意字符" 就算命中，
# 而中文指南里的剂量行与阈值行恰好长这样（"2型糖尿病患者血压应控制在130/80mmHg以下"、
# "100mg 阿司匹林每日一次长期维持"）。这类行被判成标题后不会产出任何 chunk——
# 被吞掉的正是本系统最该召回的具体数值句。点分编号（"3.1 一线用药"）后面习惯跟空格
# 而不是标点，所以单独给它一条分支。
_HEADING_HINT_RE = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百]+[章节篇部分]"
    r"|[0-9]+(?:\.[0-9]+)+\s+\S"
    r"|[0-9]+(?:\.[0-9]+)*[、.．)）]\s*\S"
    r")"
)

# 计量表达：命中的行一律不判为标题。标题不会写剂量、血压、检验阈值或数值区间，
# 而正文里的这类行恰恰是最不能被当成标题吞掉的内容。
_MEASURE_HINT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*"
    r"(?:mg|毫克|μg|ug|微克|IU|国际单位|ml|毫升|g|克|kg|公斤|mmHg|mmol|μmol|umol|mol|"
    r"%|‰|片|粒|颗|袋|支|滴|喷)"
    r"|\d\s*[~～至\-–]\s*\d"
    r"|\d\s*/\s*\d"
    r"|/\s*(?:L|l|kg|min|天|日|次|分)"
)

# 兜底识别出的标题没有可靠层级。给它一个比任何编号层级都深的值，
# _push_heading 才只会替换同为兜底级的标题，而不会把带编号的祖先章节整栈弹掉
# （弹掉的后果是 chunk.section_path 丢失"第四章"这类祖先，引用坐标直接指错）。
FALLBACK_HEADING_LEVEL = 9


# --------------------------------------------------------------------------
# 文本切分工具
# --------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    """按中文句末标点切句，标点保留在句尾。

    不做 strip，因为块间的 ``\\n`` 是段落边界信息，拼回 parent 时还要用。
    """
    if not text:
        return []
    return [piece for piece in _SENTENCE_RE.findall(text) if piece.strip()]


def _explode_long_units(
    units: Sequence[Tuple[str, Optional[int]]], max_chars: int
) -> List[Tuple[str, Optional[int]]]:
    """超长单句（常见于无标点的表格残文、目录行）硬切到 max_chars 以内。

    只有在句子本身就装不下时才硬切，正常句子仍然在标点处断开。
    """
    out = []  # type: List[Tuple[str, Optional[int]]]
    for text, page in units:
        if len(text) <= max_chars:
            out.append((text, page))
            continue
        for start in range(0, len(text), max_chars):
            out.append((text[start:start + max_chars], page))
    return out


def _pack_units(
    units: Sequence[Tuple[str, Optional[int]]],
    max_chars: int,
    overlap_chars: int = 0,
) -> List[List[Tuple[str, Optional[int]]]]:
    """把句子单元贪心打包成不超过 max_chars 的组，相邻组共享尾部句子作为重叠。

    重叠以"整句"为单位回退而不是按字符硬切，这样每个 child 的开头都是一句完整的话，
    embedding 不会被半截词污染。同时保证 next_start > start，避免退化成死循环。
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    units = _explode_long_units(units, max_chars)
    groups = []  # type: List[List[Tuple[str, Optional[int]]]]
    total = len(units)
    start = 0
    while start < total:
        size = 0
        end = start
        while end < total:
            piece = len(units[end][0])
            # 第一句无论多长都要收进来（已被 _explode_long_units 限长），否则无法前进
            if end > start and size + piece > max_chars:
                break
            size += piece
            end += 1
        groups.append(list(units[start:end]))
        if end >= total:
            break
        next_start = end
        if overlap_chars > 0:
            acc = 0
            probe = end - 1
            while probe > start and acc + len(units[probe][0]) <= overlap_chars:
                acc += len(units[probe][0])
                probe -= 1
            next_start = max(start + 1, probe + 1)
        start = next_start
    return groups


def _group_text(group: Sequence[Tuple[str, Optional[int]]]) -> str:
    return "".join(text for text, _ in group).strip()


def _merge_overlapping(prev: str, tail: str) -> str:
    """把尾段并回上一段，并消掉两者共享的重叠部分。

    child 之间本来就带 overlap_chars 的重叠，直接拼接会让同一句话在 chunk 里出现两遍，
    既污染 BM25 词频也让引用原文读起来错乱。所以先找 prev 的最长后缀 == tail 的前缀。
    """
    if not tail:
        return prev
    if tail in prev:
        return prev
    for size in range(min(len(prev), len(tail)), 0, -1):
        if prev.endswith(tail[:size]):
            return prev + tail[size:]
    return prev + tail


def _group_page(group: Sequence[Tuple[str, Optional[int]]]) -> Optional[int]:
    """组的页码取第一个有页码的单元——引用时指向内容开始的那一页最有用。"""
    for _, page in group:
        if page is not None:
            return page
    return None


def _coerce_page(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# 标题栈
# --------------------------------------------------------------------------
def _push_heading(stack: List[Tuple[int, str]], level: int, title: str) -> None:
    """按层级更新标题栈：弹出所有同级或更深的标题，再压入当前标题。

    这样 "1 概述 / 1.1 定义 / 2 诊断" 会正确地让 "2 诊断" 顶掉 1 和 1.1，
    而不是堆成 ["1 概述", "1.1 定义", "2 诊断"]。
    """
    level = max(1, int(level))
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


def _looks_like_heading(text: str, max_len: int = 40) -> bool:
    """兜底判断：短行 + 章节序号前缀 + 不含句读 + 不含计量表达，视为标题。

    纯文本来源（txt / md 之外）常常丢失结构信息，这里做一次低风险的补救；
    parser 已经给出 is_heading 时不会走到这里。

    误判的代价是不对称的：漏判一个标题只是少一层 section_path，误判一条正文却会
    让这条正文彻底不进索引（is_heading 的 block 只压栈不产出 chunk），并且污染
    后续所有 chunk 的章节路径。所以这里宁可收紧。
    """
    if not text or len(text) > max_len:
        return False
    if text[-1] in "。！？；;,，":
        return False
    # 标题不会在行内带句读；带了说明这是被截断的正文
    if re.search(r"[。！？；，]", text):
        return False
    if _MEASURE_HINT_RE.search(text):
        return False
    return bool(_HEADING_HINT_RE.match(text))


# --------------------------------------------------------------------------
# 证据等级
# --------------------------------------------------------------------------
# schema.EVIDENCE_PATTERNS 按"先 I 后 III"的顺序用 re.search 匹配，而 "III类推荐"
# 从第 3 个字符起就含有子串 "I类推荐"，于是 II / III 类推荐全被判成 I 类推荐——
# 在指南体系里 III 类的含义是"不应实施，有害"，翻成 I 类"强烈推荐"是最危险的一种错标。
# schema.py 是冻结的数据契约，不能改，所以在切片侧先做一次顺序正确的前置判定。
# 正则用单条有序 alternation：finditer 在同一起点会优先尝试更长的分支，
# 因此 "III类推荐" 只会产出一个 III，而不会既算 III 又算 II。
_REC_CLASS_RE = re.compile(r"(III|Ⅲ|IIa|Ⅱa|IIb|Ⅱb|II|Ⅱ|I|Ⅰ|[123])\s*类推荐")
_REC_CLASS_ALIAS = {
    "III": "III", "Ⅲ": "III", "3": "III",
    "IIa": "IIa", "Ⅱa": "IIa",
    "IIb": "IIb", "Ⅱb": "IIb",
    "II": "II", "Ⅱ": "II", "2": "II",
    "I": "I", "Ⅰ": "I", "1": "I",
}
_EVIDENCE_GRADE_RE = re.compile(
    r"[（(]\s*证据级别\s*[:：]?\s*([ABCD])\s*[）)]|([ABCD])\s*级证据"
)


def evidence_levels_in(text: str) -> List[str]:
    """返回文本里出现的全部推荐等级 / 证据级别标记（按出现顺序去重）。

    之所以要"全部"而不是"第一个"：同一段落里并列写多条不同强度的推荐是中文指南的
    常态，只取第一个就会把"不应实施"的段落标成"强烈推荐"。
    """
    found = []  # type: List[str]
    if not text:
        return found
    for match in _REC_CLASS_RE.finditer(text):
        level = _REC_CLASS_ALIAS.get(match.group(1))
        if level and level not in found:
            found.append(level)
    for match in _EVIDENCE_GRADE_RE.finditer(text):
        grade = match.group(1) or match.group(2)
        if grade:
            label = "level:%s" % grade
            if label not in found:
                found.append(label)
    return found


def _resolve_evidence_level(child_text: str, parent_text: str, section_path: Sequence[str]) -> str:
    """child -> parent -> section 标题，逐级回退找推荐等级；等级不唯一时宁缺毋错。

    推荐等级常写在段落开头或小标题上（"（一）I类推荐"），而检索命中的可能是
    段落中段的 child，所以需要逐级继承。但继承必须带约束：parent 默认 1200 字，
    里面同时出现 I 类与 III 类是常见写法，无条件继承会让"不应实施"的 child 被贴上
    "强烈推荐"。推荐等级是决定一条建议该不该执行的核心元数据，错标比缺失危险得多
    ——界面上它看起来还是经过溯源的。因此只有当某一级文本里的标记**唯一**时才采用。
    """
    for candidate in (child_text, parent_text, " ".join(section_path)):
        if not candidate:
            continue
        levels = evidence_levels_in(candidate)
        if not levels:
            # schema 里可能有本模块未覆盖的新标记，留一条兜底
            legacy = detect_evidence_level(candidate)
            if legacy:
                return legacy
            continue

        # 推荐类别(I/II/IIa/IIb/III)与证据级别(level:A~D)是两个正交维度，
        # "I 类推荐（证据级别：A）" 是中文指南里最标准的写法。唯一性约束必须
        # 在各维度**内部**判定：跨维度一起数会让这种标准写法命中"不唯一"而
        # 两个标记一起丢失。同一维度内出现多个取值才是真歧义，此时才宁缺毋错。
        classes = [lv for lv in levels if not lv.startswith("level:")]
        grades = [lv for lv in levels if lv.startswith("level:")]
        cls = classes[0] if len(set(classes)) == 1 else ""
        grade = grades[0] if len(set(grades)) == 1 else ""
        if cls and grade:
            return "%s（证据级别%s）" % (cls, grade.split(":", 1)[1])
        if cls:
            return cls
        if grade:
            return grade
        return ""
    return ""


# --------------------------------------------------------------------------
# 切片器
# --------------------------------------------------------------------------
class StructureAwareChunker(object):
    """按章节结构做父子切片。

    参数
    ----
    child_chars:    child（检索单元）目标字符数
    child_overlap:  相邻 child 的重叠字符数，防止答案句正好被切在边界上
    parent_chars:   parent（生成上下文）字符上限，section 超长时拆成多个 parent
    min_chars:      child 最短长度，短于此值的碎片丢弃（原子块不受此限制）
    """

    def __init__(self, child_chars=320, child_overlap=64, parent_chars=1200, min_chars=40):
        if child_chars <= 0 or parent_chars <= 0:
            raise ValueError("child_chars / parent_chars must be positive")
        if child_overlap < 0 or child_overlap >= child_chars:
            raise ValueError("child_overlap must satisfy 0 <= child_overlap < child_chars")
        if parent_chars < child_chars:
            # parent 比 child 还小说明配置写反了，直接抬到 child 大小，避免静默产出怪结构
            logger.warning(
                "parent_chars(%d) < child_chars(%d)，已自动抬升为 child_chars",
                parent_chars, child_chars,
            )
            parent_chars = child_chars
        self.child_chars = int(child_chars)
        self.child_overlap = int(child_overlap)
        self.parent_chars = int(parent_chars)
        self.min_chars = max(0, int(min_chars))

    # -- 对外主入口 ---------------------------------------------------------
    def chunk_document(self, doc: RawDoc) -> List[Chunk]:
        """把一个 RawDoc 切成带完整出处的 Chunk 列表。"""
        doc = _as_raw_doc(doc)
        chunks = []  # type: List[Chunk]
        stack = []  # type: List[Tuple[int, str]]
        pending = []  # type: List[Tuple[str, Optional[int]]]  # 当前 section 的正文单元
        counter = _Counter()

        for raw_block in doc.blocks or []:
            block = raw_block if isinstance(raw_block, dict) else {"text": raw_block}
            text = normalize_text(block.get("text"))
            if not text:
                continue
            content_type = block.get("content_type") or CONTENT_TEXT
            page = _coerce_page(block.get("page"))

            is_heading = bool(block.get("is_heading"))
            if not is_heading and content_type not in ATOMIC_CONTENT_TYPES:
                is_heading = _looks_like_heading(text)

            if is_heading:
                # 先用"旧"的 section_path 结算已累积的正文，再更新栈
                chunks.extend(self._flush_section(doc, stack, pending, counter))
                pending = []
                _push_heading(
                    stack, block.get("heading_level") or FALLBACK_HEADING_LEVEL, text
                )
                continue

            if content_type in ATOMIC_CONTENT_TYPES:
                # 原子块不进 pending，也不打断正文聚合：它自成一个 chunk，
                # parent_text 就是它自己（表格的上下文就是整张表）。
                # 代价是它在输出里会排在同 section 的正文之前（正文要攒到 flush 才结算）；
                # 这只影响 jsonl 的阅读顺序，不影响检索，换来的是 parent 不被表格切碎。
                chunks.append(
                    self._make_chunk(
                        doc=doc,
                        section_path=[title for _, title in stack],
                        text=text,
                        parent_text=text,
                        page=page,
                        content_type=content_type,
                        counter=counter,
                        parent_id="",
                        child_index=0,
                        atomic=True,
                    )
                )
                continue

            # 每个 block 末尾补 \n，让段落边界在拼回 parent 时仍然可见
            pending.extend((sent, page) for sent in split_sentences(text + "\n"))

        chunks.extend(self._flush_section(doc, stack, pending, counter))
        return chunks

    # -- section -> parent -> child ----------------------------------------
    def _flush_section(self, doc, stack, pending, counter):
        """把一个 section 累积的正文切成 parent / child 两层。"""
        if not pending:
            return []
        section_path = [title for _, title in stack]
        out = []  # type: List[Chunk]
        parent_groups = _pack_units(pending, self.parent_chars, overlap_chars=0)
        for parent_idx, parent_group in enumerate(parent_groups):
            parent_text = _group_text(parent_group)
            if not parent_text:
                continue
            parent_id = stable_id(doc.doc_id, " > ".join(section_path), "parent", parent_idx)
            child_groups = _pack_units(parent_group, self.child_chars, self.child_overlap)

            # 先把 child 文本定下来再建 chunk：短于 min_chars 的碎片必须**合并**而不是丢弃。
            # 丢弃会让这段文字在三路索引里彻底检索不到（它只剩在 parent_text 里，
            # 而 parent_text 不进索引）。医疗语料里"妊娠期妇女禁用本品"这类关键短句
            # 恰好是 20~35 字，整段禁忌就这么静默消失了。
            records = []  # type: List[Tuple[str, Optional[int]]]
            for child_group in child_groups:
                child_text = _group_text(child_group)
                if not child_text:
                    continue
                page = _group_page(child_group) or _group_page(parent_group)
                if len(child_text) < self.min_chars and records:
                    prev_text, prev_page = records[-1]
                    records[-1] = (_merge_overlapping(prev_text, child_text), prev_page)
                    continue
                # 本 parent 一条 child 都还没有时无条件保留：短 section 本身就是原子的
                records.append((child_text, page))

            for child_idx, (child_text, page) in enumerate(records):
                out.append(
                    self._make_chunk(
                        doc=doc,
                        section_path=section_path,
                        text=child_text,
                        parent_text=parent_text,
                        page=page,
                        content_type=CONTENT_TEXT,
                        counter=counter,
                        parent_id=parent_id,
                        child_index=child_idx,
                        atomic=False,
                    )
                )
        return out

    # -- Chunk 组装 ---------------------------------------------------------
    def _make_chunk(self, doc, section_path, text, parent_text, page, content_type,
                    counter, parent_id, child_index, atomic):
        seq = counter.next()
        extra = {
            "parent_id": parent_id,
            "child_index": child_index,
            "section_depth": len(section_path),
        }
        if atomic:
            extra["atomic"] = True
        return Chunk(
            chunk_id=stable_id(doc.doc_id, " > ".join(section_path), seq, text[:64]),
            text=text,
            doc_id=doc.doc_id,
            doc_title=doc.doc_title,
            source_tier=doc.source_tier,
            parent_text=parent_text,
            source_path=doc.source_path,
            version=doc.version,
            section_path=list(section_path),
            page=page,
            evidence_level=_resolve_evidence_level(text, parent_text, section_path),
            content_type=content_type,
            authority=TIER_AUTHORITY.get(doc.source_tier, 0.3),
            extra=extra,
        )


class _Counter(object):
    """文档内的单调序号，保证同文同节的重复文本也拿到不同 chunk_id。"""

    def __init__(self):
        self.value = 0

    def next(self) -> int:
        self.value += 1
        return self.value


def chunk_documents(docs: Iterable[Any], **kwargs: Any) -> List[Chunk]:
    """批量切片。kwargs 透传给 :class:`StructureAwareChunker`。"""
    chunker = StructureAwareChunker(**kwargs)
    out = []  # type: List[Chunk]
    for doc in docs:
        try:
            out.extend(chunker.chunk_document(doc))
        except Exception as exc:  # 单篇坏文档不应该毁掉整批离线建库
            logger.warning("切片失败，已跳过该文档: %s", exc)
    return out


# --------------------------------------------------------------------------
# RawDoc 反序列化（schema.py 只提供 Chunk.from_dict，这里补 RawDoc 的）
# --------------------------------------------------------------------------
def _as_raw_doc(doc: Any) -> RawDoc:
    if isinstance(doc, RawDoc):
        return doc
    if not isinstance(doc, dict):
        raise TypeError("doc must be a RawDoc or a dict, got %r" % type(doc))
    blocks = doc.get("blocks") or []
    if not isinstance(blocks, list):
        blocks = [blocks]
    doc_id = str(doc.get("doc_id") or stable_id(doc.get("source_path", ""), doc.get("doc_title", "")))
    return RawDoc(
        doc_id=doc_id,
        doc_title=str(doc.get("doc_title") or ""),
        source_tier=str(doc.get("source_tier") or TIER_CONSULTATION),
        source_path=str(doc.get("source_path") or ""),
        version=str(doc.get("version") or ""),
        blocks=blocks,
        extra=doc.get("extra") or {},
    )


# --------------------------------------------------------------------------
# 统计与 CLI
# --------------------------------------------------------------------------
def summarize_chunks(chunks: Sequence[Chunk]) -> Dict[str, Any]:
    """切片质量体检：长度分布、原子块数量、证据等级覆盖率、无章节路径比例。"""
    if not chunks:
        return {"chunks": 0}
    lengths = sorted(len(c.text) for c in chunks)
    by_type = {}  # type: Dict[str, int]
    by_tier = {}  # type: Dict[str, int]
    for chunk in chunks:
        by_type[chunk.content_type] = by_type.get(chunk.content_type, 0) + 1
        by_tier[chunk.source_tier] = by_tier.get(chunk.source_tier, 0) + 1
    return {
        "chunks": len(chunks),
        "docs": len({c.doc_id for c in chunks}),
        "parents": len({c.extra.get("parent_id") for c in chunks if c.extra.get("parent_id")}),
        "child_chars_min": lengths[0],
        "child_chars_median": lengths[len(lengths) // 2],
        "child_chars_max": lengths[-1],
        "child_chars_mean": round(sum(lengths) / float(len(lengths)), 1),
        "with_evidence_level": sum(1 for c in chunks if c.evidence_level),
        "with_page": sum(1 for c in chunks if c.page is not None),
        "without_section_path": sum(1 for c in chunks if not c.section_path),
        "max_section_depth": max(len(c.section_path) for c in chunks),
        "by_content_type": by_type,
        "by_source_tier": by_tier,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Structure-aware chunking: RawDoc jsonl -> Chunk jsonl."
    )
    parser.add_argument("--input", required=True, help="RawDoc jsonl 路径")
    parser.add_argument("--output", required=True, help="Chunk jsonl 输出路径")
    parser.add_argument("--child_chars", type=int, default=320)
    parser.add_argument("--child_overlap", type=int, default=64)
    parser.add_argument("--parent_chars", type=int, default=1200)
    parser.add_argument("--min_chars", type=int, default=40)
    parser.add_argument("--max_docs", type=int, default=-1, help="只处理前 N 篇，调试用")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    docs = list(read_jsonl(args.input, max_rows=args.max_docs))
    chunks = chunk_documents(
        docs,
        child_chars=args.child_chars,
        child_overlap=args.child_overlap,
        parent_chars=args.parent_chars,
        min_chars=args.min_chars,
    )
    written = save_chunks(args.output, chunks)
    stats = summarize_chunks(chunks)
    stats["input_docs"] = len(docs)
    stats["written"] = written
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
