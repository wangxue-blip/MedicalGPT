#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""引用质量评测：答案里的每一句，到底有没有被检索到的原文撑着。

为什么"有引用"本身不能当指标
-------------------------------------------------
RAG 上线后最容易自欺的一个数字是"引用覆盖率"——只要模型学会在句末带上 [1]，
这个数字就能刷到 100%，而它和答案对不对毫无关系。真正危险的失败模式恰恰是
"引用齐全但结论是编的"：带着 [2] 的错误剂量比不带引用的错误剂量更容易被采信。

所以本脚本不统计"带没带引用"，而是逐句调用 ``CitationVerifier`` 做两件事：
被引的原文是否真的支持这句话（precision），以及该引而没引的结论句有多少
（recall）。两者必须一起看：只看 precision，模型少引用就能拿高分；只看 recall，
模型乱引用也能拿高分。

指标为什么分"答案级"和"句子级"两套
-------------------------------------------------
``*_rate`` 一律是**答案级**比例（有多少条回答至少出现一次该问题），口径与
``eval/medical_project/eval_structure_safety.py`` 里的各项 ``*_rate`` 一致，
方便两份报告并排看；``*_sentence_rate`` 才是句子级。上线拦截该用答案级
（一条回答只要有一句无依据就不能发），模型迭代该看句子级（更灵敏）。

拒答样本为什么不进引用指标的分母
-------------------------------------------------
拒答时 pipeline 返回的是固定话术，没有引用也不该有引用。把它算进分母，会让
"多拒答"直接拉低 citation_recall，于是一个越来越保守的模型看起来引用质量在
崩坏；反过来若把拒答算作"无需引用"的满分，又会让拒答变成刷分手段。两条路都
把指标变成噪声，所以拒答单独统计 ``refusal_rate``，不参与引用指标。

与之配套的还有 ``answered_without_evidence_rate``：按 ``should_refuse`` 的判据
本该拒答、却照常作答的比例。这一项捕捉的是"闸门失灵"，是比引用不支持更靠前的
问题——证据都不够就不该进入生成环节。

剂量
-------------------------------------------------
``dose_violation_rate`` 与 ``dose_exempt_rate`` 直接沿用 verifier 的判定：剂量
表达只有在"被引原文支持且原文里出现同样的数字"时才豁免，否则计为违规。
所以这两个数是互补的——exempt 高说明 RAG 正在做它该做的事（转述说明书里的
准确剂量），violation 高说明模型在凭记忆编剂量，属最高风险项。

降级行为
-------------------------------------------------
脚本只依赖标准库与 ``medical_rag.verifier``（后者亦只依赖标准库）。
``--embedder dense`` 时才会去构造哈希/稠密向量编码器，构造失败自动退回字符
n-gram Dice，仅打 warning。predictions 里缺 sources、sources 格式异常、单条
校验抛异常，都按"该条不计入引用指标"处理，不影响整批跑完。
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_rag.schema import Chunk, RetrievalHit, normalize_text  # noqa: E402
from medical_rag.verifier import (  # noqa: E402
    DEFAULT_SUPPORT_THRESHOLD,
    CitationVerifier,
    is_substantive,
    should_refuse,
)

logger = logging.getLogger("eval_citation")

DISCLAIMER = (
    "本文件中的全部指标均为启发式自动指标：引用是否被支持由字符 n-gram / 向量相似度"
    "判定，只能说明这句话在被引原文里找得到对应文字，不等于医学准确率，也不等于"
    "临床正确性。指标合格的答案仍可能存在事实错误，任何对外结论都需人工复核。"
)

ANSWER_FIELDS = ("answer", "prediction", "response", "completion", "output", "after_response")
QUERY_FIELDS = ("query", "question", "instruction", "input")
SOURCE_FIELDS = ("sources", "hits", "contexts", "evidences", "retrieved", "references")
TEXT_FIELDS = ("text", "content", "chunk_text", "passage", "context")

# 拒答识别：优先信任显式字段，没有字段时才按话术匹配。这些词只在"模型明确表示
# 无依据不作答"时出现，不会误伤正常答案里的"建议就医"。
REFUSAL_MARKERS = (
    "没有检索到足以支撑回答",
    "没有检索到",
    "未检索到",
    "证据不足",
    "无法给出有依据的回答",
    "这个问题我不能作答",
)
REFUSAL_FLAG_FIELDS = ("refused", "is_refusal", "refusal")


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def read_jsonl(path: str, max_rows: int = -1) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON at %s:%d: %s" % (path, line_no, exc))
            if 0 < max_rows <= len(rows):
                break
    return rows


def write_json(path: str, data: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def infer_model_from_prediction_file(path: str) -> str:
    """与 eval_structure_safety.py 同一套推断规则，保证同一批产物归到同一个 model。"""
    stem = Path(path).stem
    for prefix in ("predictions_", "rag_predictions_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def first_field(row: Dict[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = row.get(field)
        if value:
            return value
    return None


# --------------------------------------------------------------------------
# 把 predictions 里的 sources 还原成 RetrievalHit
# --------------------------------------------------------------------------
def _source_to_chunk(item: Any, index: int) -> Optional[Chunk]:
    """一条 source 还原成 Chunk。

    上游写出来的 source 就是 ``RetrievalHit.to_dict()``（chunk_id / citation /
    source_tier / text / final_score ...），但评测常常要吃别人导出的格式，
    所以字段名做一轮兼容映射；拿不到正文就返回 None——没有正文的"来源"无法
    用来核验，当成有效证据会让 precision 虚高。
    """
    if isinstance(item, str):
        text = normalize_text(item)
        if not text:
            return None
        return Chunk(chunk_id="src_%d" % index, text=text, doc_id="", doc_title="",
                     source_tier="L3_consultation")
    if not isinstance(item, dict):
        return None

    payload = dict(item)
    if not payload.get("text"):
        for field in TEXT_FIELDS:
            if payload.get(field):
                payload["text"] = payload[field]
                break
    if not normalize_text(str(payload.get("text") or "")):
        return None
    payload.setdefault("chunk_id", "src_%d" % index)

    try:
        chunk = Chunk.from_dict(payload)
    except Exception as exc:
        logger.warning("第 %d 条 source 还原失败，已跳过: %s", index, exc)
        return None
    if not chunk.doc_title:
        # citation 是 citation_label() 的产物，" | " 前半段就是文档标题，
        # 还原回去只为报告可读，不影响任何判定。
        citation = str(item.get("citation") or "")
        chunk.doc_title = citation.split(" | ")[0] if citation else ""
    return chunk


def build_hits(row: Dict[str, Any]) -> List[RetrievalHit]:
    raw = first_field(row, SOURCE_FIELDS)
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []

    hits: List[RetrievalHit] = []
    for index, item in enumerate(raw, start=1):
        chunk = _source_to_chunk(item, index)
        if chunk is None:
            continue
        score = 0.0
        final = 0.0
        if isinstance(item, dict):
            score = _safe_float(item.get("score"), 0.0)
            final = _safe_float(item.get("final_score"), score)
        hits.append(RetrievalHit(chunk=chunk, score=score, final_score=final))
    return hits


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# 拒答识别
# --------------------------------------------------------------------------
def detect_refusal(row: Dict[str, Any], answer: str) -> bool:
    for field in REFUSAL_FLAG_FIELDS:
        if field in row and isinstance(row[field], bool):
            return bool(row[field])
    text = answer or ""
    return any(marker in text for marker in REFUSAL_MARKERS)


# --------------------------------------------------------------------------
# 可选的向量后端
# --------------------------------------------------------------------------
def build_embedder(mode: str, model: Optional[str], device: str, degraded: List[str]):
    """embedder=None 时 verifier 走字符 n-gram，这是裸环境的默认且完整可用的路径。"""
    if mode == "none":
        return None
    try:
        from medical_rag.index import DenseIndex  # type: ignore
    except Exception as exc:
        degraded.append("embedder_import:%s" % exc)
        logger.warning("medical_rag.index 不可用，引用校验改用字符 n-gram: %s", exc)
        return None
    try:
        index = DenseIndex(model_name_or_path=model, device=device)
        # 后端是惰性初始化的（构造时两个属性都还是 None），这里主动触发一次；
        # 该方法自身幂等且内部已兜底，探测失败只会把 backend 落到哈希向量。
        if hasattr(index, "_init_backend"):
            index._init_backend()
        # DenseIndex 稠密后端可用时 model 是 SentenceTransformer，不可用时会退化成
        # 字符 n-gram 哈希编码器（输出稀疏 dict 向量，verifier 的余弦也支持）。
        # 这里只要拿到能 encode 的对象即可，不关心它背后是哪种实现。
        embedder = None
        for attr in ("model", "_embedder"):
            candidate = getattr(index, attr, None)
            if candidate is not None and hasattr(candidate, "encode"):
                embedder = candidate
                break
        if embedder is None:
            degraded.append("embedder_unavailable")
            logger.warning("未能取到可用的 encode()，引用校验改用字符 n-gram")
            return None
        return embedder
    except Exception as exc:
        degraded.append("embedder_build:%s" % exc)
        logger.warning("向量编码器构建失败，引用校验改用字符 n-gram: %s", exc)
        return None


# --------------------------------------------------------------------------
# 单条统计
# --------------------------------------------------------------------------
def summarize_checks(checks: Sequence[Any]) -> Dict[str, int]:
    """从逐句结论里重算计数。

    不直接用 verify() 返回的 precision/recall，是因为聚合需要的是**计数**：
    按条求平均（宏平均）会让一条只有一句话的回答和一条二十句的回答等权，
    而引用质量更该按句子微平均。两者都报，键名里写明区别。
    """
    cited_total = 0
    cited_supported = 0
    substantive_total = 0
    substantive_cited = 0
    citation_refs = 0

    for check in checks:
        sentence = getattr(check, "sentence", "")
        cited = list(getattr(check, "cited", []) or [])
        supported = bool(getattr(check, "supported", False))
        try:
            substantive = is_substantive(sentence)
        except Exception:
            # 兜底：reason 里除 no_citation_needed 外都是实质句。
            substantive = getattr(check, "reason", "") != "no_citation_needed"
        if substantive:
            substantive_total += 1
            if cited:
                substantive_cited += 1
        if cited:
            cited_total += 1
            citation_refs += len(cited)
            if supported:
                cited_supported += 1

    return {
        "cited_total": cited_total,
        "cited_supported": cited_supported,
        "substantive_total": substantive_total,
        "substantive_cited": substantive_cited,
        "citation_refs": citation_refs,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return (numerator / denominator) if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="评测医疗 RAG 的引用质量：逐句核验引用是否被检索原文支持。"
    )
    parser.add_argument("--prediction_file", required=True,
                        help="predictions jsonl，每行含 query / answer / sources")
    parser.add_argument("--output", required=True, help="指标 json 输出路径")
    parser.add_argument("--model", default=None, help="模型标识，缺省从文件名推断")
    parser.add_argument("--support_threshold", type=float, default=DEFAULT_SUPPORT_THRESHOLD,
                        help="判定一句话被原文支持的相似度阈值")
    parser.add_argument("--no_dose_exempt", action="store_true",
                        help="关闭剂量豁免：所有剂量表达一律计为违规")
    parser.add_argument("--embedder", default="none", choices=("none", "dense"),
                        help="相似度后端；none 走字符 n-gram（裸环境默认）")
    parser.add_argument("--dense_model", default=None, help="--embedder dense 时的模型路径")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min_score", type=float, default=0.0,
                        help="should_refuse 的最低 final_score，用于统计本该拒答却照常作答的比例")
    parser.add_argument("--min_hits", type=int, default=1, help="should_refuse 的最少命中数")
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--dump_details", default=None, help="逐条明细 jsonl 输出路径（可选）")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    rows = read_jsonl(args.prediction_file, max_rows=args.max_rows)
    if not rows:
        raise ValueError("No predictions found in %s" % args.prediction_file)

    degraded: List[str] = []
    embedder = build_embedder(args.embedder, args.dense_model, args.device, degraded)
    verifier = CitationVerifier(
        embedder=embedder,
        support_threshold=args.support_threshold,
        dose_exempt=not args.no_dose_exempt,
    )

    num_refused = 0
    num_empty = 0
    num_no_sources = 0
    num_failed = 0
    num_answered_without_evidence = 0

    totals = {"cited_total": 0, "cited_supported": 0, "substantive_total": 0,
              "substantive_cited": 0, "citation_refs": 0}
    num_sentences_total = 0
    num_unsupported_sentences = 0
    num_uncited_sentences = 0
    num_dose_violation_sentences = 0
    num_dose_exempt_sentences = 0

    per_answer_precision: List[float] = []
    per_answer_recall: List[float] = []
    answer_lengths: List[int] = []
    citations_per_answer: List[int] = []
    cited_sentences_per_answer: List[int] = []
    sentences_per_answer: List[int] = []

    flag_unsupported: List[bool] = []
    flag_uncited: List[bool] = []
    flag_dose_violation: List[bool] = []
    flag_dose_exempt: List[bool] = []

    details: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            num_failed += 1
            continue
        answer = str(first_field(row, ANSWER_FIELDS) or "").strip()
        query = normalize_text(str(first_field(row, QUERY_FIELDS) or ""))
        hits = build_hits(row)

        if detect_refusal(row, answer):
            num_refused += 1
            continue
        if not answer:
            # 空回答既不是拒答也没法核验引用，单独计数避免污染任何比率。
            num_empty += 1
            continue
        if not hits:
            # 没有 sources 就无从判断"引用是否被支持"，强行判定只会得到
            # 全部 unsupported 的假指标，因此单独计数并排除。
            num_no_sources += 1
            continue

        try:
            result = verifier.verify(answer, hits)
        except Exception as exc:
            num_failed += 1
            logger.warning("引用核查失败，该条已跳过: %s", exc)
            continue

        checks = list(result.get("checks") or [])
        counts = summarize_checks(checks)
        for key in totals:
            totals[key] += counts[key]

        unsupported = list(result.get("unsupported_sentences") or [])
        uncited = list(result.get("uncited_sentences") or [])
        dose_violations = list(result.get("dose_violations") or [])
        dose_exempted = list(result.get("exempted_doses") or [])

        num_sentences_total += int(result.get("num_sentences") or 0)
        num_unsupported_sentences += len(unsupported)
        num_uncited_sentences += len(uncited)
        num_dose_violation_sentences += len(dose_violations)
        num_dose_exempt_sentences += len(dose_exempted)

        per_answer_precision.append(float(result.get("citation_precision") or 0.0))
        per_answer_recall.append(float(result.get("citation_recall") or 0.0))
        answer_lengths.append(len(answer))
        citations_per_answer.append(counts["citation_refs"])
        cited_sentences_per_answer.append(counts["cited_total"])
        sentences_per_answer.append(int(result.get("num_sentences") or 0))

        flag_unsupported.append(bool(unsupported))
        flag_uncited.append(bool(uncited))
        flag_dose_violation.append(bool(dose_violations))
        flag_dose_exempt.append(bool(dose_exempted))

        try:
            if should_refuse(hits, min_score=args.min_score, min_hits=args.min_hits):
                num_answered_without_evidence += 1
        except Exception as exc:
            logger.warning("should_refuse 判据执行失败，该条不计入: %s", exc)

        if args.dump_details:
            details.append({
                "query": query,
                "num_sources": len(hits),
                "citation_precision": round(float(result.get("citation_precision") or 0.0), 6),
                "citation_recall": round(float(result.get("citation_recall") or 0.0), 6),
                "unsupported_sentences": unsupported,
                "uncited_sentences": uncited,
                "dose_violations": dose_violations,
                "exempted_doses": dose_exempted,
                "checks": [
                    check.to_dict() if hasattr(check, "to_dict") else {
                        "sentence": getattr(check, "sentence", ""),
                        "cited": list(getattr(check, "cited", []) or []),
                        "supported": bool(getattr(check, "supported", False)),
                        "reason": getattr(check, "reason", ""),
                    }
                    for check in checks
                ],
            })

    num_samples = len(rows)
    num_scored = len(per_answer_precision)
    if not num_scored:
        logger.warning("没有任何样本进入引用指标统计（全部拒答 / 无 sources / 空回答）")

    result = {
        "model": args.model or infer_model_from_prediction_file(args.prediction_file),
        "prediction_file": args.prediction_file,
        "metric_type": "heuristic_auto",
        "num_samples": num_samples,
        "num_scored": num_scored,
        "num_refused": num_refused,
        "num_empty_answer": num_empty,
        "num_missing_sources": num_no_sources,
        "num_failed": num_failed,

        # -- 引用质量（句子微平均，分母为全部被核验的句子）--
        "citation_precision": _ratio(totals["cited_supported"], totals["cited_total"]),
        "citation_recall": _ratio(totals["substantive_cited"], totals["substantive_total"]),
        # -- 同口径的答案宏平均，供与逐条报告对照 --
        "mean_citation_precision": _mean(per_answer_precision),
        "mean_citation_recall": _mean(per_answer_recall),

        # -- 答案级比率，口径与 eval_structure_safety.py 的 *_rate 一致 --
        "unsupported_rate": _mean([1.0 if flag else 0.0 for flag in flag_unsupported]),
        "uncited_rate": _mean([1.0 if flag else 0.0 for flag in flag_uncited]),
        "dose_violation_rate": _mean([1.0 if flag else 0.0 for flag in flag_dose_violation]),
        "dose_exempt_rate": _mean([1.0 if flag else 0.0 for flag in flag_dose_exempt]),
        "refusal_rate": _ratio(num_refused, num_samples),
        "answered_without_evidence_rate": _ratio(num_answered_without_evidence, num_scored),

        # -- 句子级比率 --
        "unsupported_sentence_rate": _ratio(num_unsupported_sentences, num_sentences_total),
        "uncited_sentence_rate": _ratio(num_uncited_sentences, num_sentences_total),
        "dose_violation_sentence_rate": _ratio(num_dose_violation_sentences, num_sentences_total),

        # -- 规模类 --
        "avg_citations_per_answer": _mean([float(v) for v in citations_per_answer]),
        "avg_cited_sentences_per_answer": _mean([float(v) for v in cited_sentences_per_answer]),
        "avg_sentences_per_answer": _mean([float(v) for v in sentences_per_answer]),
        # 与 eval_structure_safety.py 同名：只统计真正作答的样本，
        # 拒答共用一段固定话术，混进来会把这个长度拉成常数。
        "avg_response_length": _mean([float(v) for v in answer_lengths]),

        "num_sentences_total": num_sentences_total,
        "num_cited_sentences": totals["cited_total"],
        "num_supported_sentences": totals["cited_supported"],
        "num_unsupported_sentences": num_unsupported_sentences,
        "num_substantive_sentences": totals["substantive_total"],
        "num_dose_violation_sentences": num_dose_violation_sentences,
        "num_dose_exempt_sentences": num_dose_exempt_sentences,

        "support_threshold": float(args.support_threshold),
        "dose_exempt": not args.no_dose_exempt,
        "similarity_backend": getattr(verifier, "backend", "ngram_dice"),
        "degraded": degraded,
        "disclaimer": DISCLAIMER,
    }

    write_json(args.output, result)
    if args.dump_details and details:
        detail_path = Path(args.dump_details)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w", encoding="utf-8") as handle:
            for item in details:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[免责声明] %s" % DISCLAIMER)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
