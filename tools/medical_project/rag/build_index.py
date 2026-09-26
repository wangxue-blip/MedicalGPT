#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""医疗 RAG 索引构建 CLI：读 chunks.jsonl，构建三路召回索引，DenseIndex 向量落盘。

为什么要把索引构建单独做成一个离线 CLI，而不是每次起服务时现建：
  - DenseIndex 是三路里唯一需要过模型的一路，几万条 chunk 的向量化动辄几分钟，
    放在服务启动路径上意味着每次重启都要重算一遍；
  - 三路的 backend 到底降级到了哪一级（有没有 sentence_transformers、有没有 faiss、
    分词走的 jieba 还是字符 bigram），必须在构建期就落到 index_summary.json 里。
    线上召回质量不对时，第一件事就是看这份 summary，而不是去猜环境里装了什么。
  - BM25 / term 两路重建极快（纯 Python 倒排），没有落盘价值，所以只缓存 dense 向量。

DenseIndex 的缓存由 medical_rag.index.DenseIndex.save/load 负责，缓存里带
chunk_id 指纹：语料换了而缓存没删时 load 会拒绝加载并触发重建，不会静默地
拿上一版语料去检索。本脚本在写完缓存后会立刻用一个新对象 load 一次做回读校验，
把结果记进 summary——缓存写坏了要在构建期暴露，不能等到线上检索才发现。

用法::

    # 用内置降级路径跑通（裸环境，无 sentence_transformers / faiss）
    python tools/medical_project/rag/build_index.py \\
        --chunks data_processed/medical_project/rag/chunks.jsonl \\
        --index_dir data_processed/medical_project/rag/index

    # 指定本地 embedding 模型，只建 dense 与 bm25 两路
    python tools/medical_project/rag/build_index.py \\
        --chunks data_processed/medical_project/rag/chunks.jsonl \\
        --index_dir data_processed/medical_project/rag/index \\
        --dense_model models/bge-m3 \\
        --routes dense,bm25
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 本脚本按 `python tools/medical_project/rag/build_index.py` 的方式从仓库根目录运行，
# 此时 sys.path[0] 是 tools/medical_project/rag/，仓库根不在路径上，import medical_rag 会失败。
# 这里补上仓库根目录（本文件的第 3 级父目录），让两种入口都能工作。
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from medical_rag import index as index_module  # noqa: E402
from medical_rag.index import BaseIndex, DenseIndex, build_indexes  # noqa: E402
from medical_rag.schema import Chunk, load_chunks, stable_id  # noqa: E402

logger = logging.getLogger("build_index")

DEFAULT_CHUNKS = "data_processed/medical_project/rag/chunks.jsonl"
DEFAULT_INDEX_DIR = "data_processed/medical_project/rag/index"
DEFAULT_ROUTES = "dense,bm25,term"
DEFAULT_PROBE_QUERY = "二甲双胍怎么吃，肾功能不好能用吗"


def parse_routes(value: str) -> Tuple[str, ...]:
    """解析 --routes，保序去重。未知路留给 build_indexes 去 warning，这里不拦。"""
    routes: List[str] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item and item not in routes:
            routes.append(item)
    if not routes:
        raise ValueError("--routes 不能为空")
    return tuple(routes)


def load_terminology(path: Optional[str]):
    """载入术语表。术语表是 term 路的输入，缺失时降级为内置默认表而不是让构建失败。"""
    try:
        from medical_rag.terminology import Terminology
    except Exception as exc:  # pragma: no cover - terminology 自身已做降级，这里只兜底
        logger.warning("术语模块不可用，term 路将使用 index 内置兜底表: %s", exc)
        return None

    if path and os.path.exists(path):
        try:
            terminology = Terminology.from_jsonl(path)
            logger.info("术语表已载入: %s", path)
            return terminology
        except Exception as exc:
            logger.warning("术语表 %s 载入失败，改用内置默认表: %s", path, exc)
    elif path:
        logger.warning("术语表不存在: %s，改用内置默认表", path)
    try:
        return Terminology.default()
    except Exception as exc:  # pragma: no cover
        logger.warning("内置术语表构造失败: %s", exc)
        return None


def _module_installed(name: str) -> bool:
    """只查模块是否可定位，不真正导入——import sentence_transformers 会连带拉起 torch，
    几秒起步，纯粹为了写一行 summary 不值得付这个代价。"""
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:  # pragma: no cover - 某些受限环境 find_spec 也会抛
        return False


def env_report() -> Dict[str, Any]:
    """记录本次构建实际可用的可选依赖。

    summary 里留这一段是为了让"同一份语料在两台机器上召回结果不同"这类问题
    一眼可查——多半就是一边有 sentence_transformers、另一边降级成哈希向量。
    """
    return {
        "python": sys.version.split()[0],
        "tokenizer_backend": getattr(index_module, "TOKENIZER_BACKEND", "unknown"),
        "numpy": bool(getattr(index_module, "_HAS_NUMPY", False)),
        "faiss": bool(getattr(index_module, "_HAS_FAISS", False)),
        "jieba": bool(getattr(index_module, "_HAS_JIEBA", False)),
        "rank_bm25": bool(getattr(index_module, "_HAS_RANK_BM25", False)),
        "sentence_transformers": _module_installed("sentence_transformers"),
    }


def corpus_report(chunks: Sequence[Chunk]) -> Dict[str, Any]:
    """语料画像：来源分级分布决定了权威度加权能不能起作用，构建期就要看一眼。"""
    tiers = Counter(chunk.source_tier for chunk in chunks)
    content_types = Counter(chunk.content_type for chunk in chunks)
    lengths = [len(chunk.text) for chunk in chunks]
    with_page = sum(1 for chunk in chunks if chunk.page is not None)
    with_evidence = sum(1 for chunk in chunks if chunk.evidence_level)
    report: Dict[str, Any] = {
        "num_chunks": len(chunks),
        "num_docs": len({chunk.doc_id for chunk in chunks}),
        "tier_distribution": dict(tiers.most_common()),
        "content_type_distribution": dict(content_types.most_common()),
        "with_page": with_page,
        "with_evidence_level": with_evidence,
        "fingerprint": stable_id(*[chunk.chunk_id for chunk in chunks]) if chunks else "",
    }
    if lengths:
        report["text_len_avg"] = round(sum(lengths) / float(len(lengths)), 2)
        report["text_len_min"] = min(lengths)
        report["text_len_max"] = max(lengths)
    return report


def build_one_route(
    route: str,
    chunks: List[Chunk],
    dense_model: Optional[str],
    terminology: Any,
    device: str,
) -> Tuple[Optional[BaseIndex], Dict[str, Any]]:
    """逐路构建，为的是拿到每一路单独的耗时。

    仍然走 build_indexes 而不是直接 new 各个 Index 类：单路构建失败时的
    "只 warning 不抛" 语义由 build_indexes 统一保证，这里不重复实现一遍。
    """
    started = time.perf_counter()
    indexes = build_indexes(
        chunks,
        dense_model=dense_model,
        terminology=terminology,
        routes=(route,),
        device=device,
    )
    elapsed = time.perf_counter() - started
    index = indexes.get(route)
    stat: Dict[str, Any] = {
        "route": route,
        "ok": index is not None,
        "backend": getattr(index, "backend", "") if index is not None else "",
        "size": int(getattr(index, "size", 0)) if index is not None else 0,
        "build_seconds": round(elapsed, 3),
    }
    if index is None:
        stat["error"] = "构建失败，详见日志中的 warning"
    return index, stat


def save_dense(index: DenseIndex, index_dir: str, verify: bool,
               dense_model: Optional[str], device: str) -> Dict[str, Any]:
    """落盘 dense 向量，并可选地用一个全新对象回读校验。

    回读用新对象而不是复用刚构建好的那个：复用的话向量本来就在内存里，
    load 成功与否说明不了缓存文件本身是否完好。
    """
    info: Dict[str, Any] = {"index_dir": index_dir, "saved": False}
    started = time.perf_counter()
    try:
        index.save(index_dir)
    except Exception as exc:
        logger.warning("dense 向量落盘失败(不影响本次构建结果): %s", exc)
        info["error"] = str(exc)
        return info

    info["saved"] = True
    info["save_seconds"] = round(time.perf_counter() - started, 3)
    meta_path = Path(index_dir) / "meta.json"
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        info["vector_file"] = meta.get("vector_file", "")
        info["dense_vectors"] = bool(meta.get("dense"))
        info["dim"] = meta.get("dim")
        info["fingerprint"] = meta.get("fingerprint", "")
        vector_path = Path(index_dir) / str(meta.get("vector_file") or "")
        if vector_path.exists():
            info["vector_file_bytes"] = vector_path.stat().st_size
    except Exception as exc:  # pragma: no cover - meta 刚写出来，正常不会读失败
        logger.warning("读取 dense 缓存 meta 失败: %s", exc)
        info["meta_error"] = str(exc)

    if verify:
        started = time.perf_counter()
        probe = DenseIndex(model_name_or_path=dense_model, device=device)
        ok = False
        try:
            ok = bool(probe.load(index_dir, chunks=index.chunks))
        except Exception as exc:
            logger.warning("dense 缓存回读异常: %s", exc)
            info["reload_error"] = str(exc)
        info["reload_ok"] = ok
        info["reload_seconds"] = round(time.perf_counter() - started, 3)
        if not ok:
            logger.warning("dense 缓存回读失败，下次检索会重建向量")
    return info


def probe_search(indexes: Dict[str, BaseIndex], query: str, topk: int) -> Dict[str, Any]:
    """构建完立刻用一条真实查询试一下：索引建成但检索返回空是很常见的哑失败。"""
    result: Dict[str, Any] = {"query": query, "topk": topk, "routes": {}}
    for route, index in indexes.items():
        try:
            hits = index.search(query, topk)
        except Exception as exc:
            logger.warning("[%s] 探测检索异常: %s", route, exc)
            result["routes"][route] = {"error": str(exc)}
            continue
        preview = []
        for rank, (idx, score) in enumerate(hits, start=1):
            if 0 <= idx < len(index.chunks):
                chunk = index.chunks[idx]
                preview.append({
                    "rank": rank,
                    "score": round(float(score), 6),
                    "chunk_id": chunk.chunk_id,
                    "doc_title": chunk.doc_title,
                    "text": chunk.text[:80].replace("\n", " "),
                })
        result["routes"][route] = {"num_hits": len(hits), "top": preview}
    return result


def print_route_table(stats: List[Dict[str, Any]]) -> None:
    """把每一路的 backend 与耗时打到 stdout，脚本日志里一眼能看出降级情况。"""
    print("")
    print("%-8s %-8s %-30s %10s %12s" % ("route", "ok", "backend", "size", "seconds"))
    print("-" * 72)
    for stat in stats:
        print("%-8s %-8s %-30s %10d %12.3f" % (
            stat["route"],
            "yes" if stat["ok"] else "NO",
            stat["backend"] or "-",
            stat["size"],
            stat["build_seconds"],
        ))
    print("-" * 72)


def print_probe(probe: Dict[str, Any]) -> None:
    print("")
    print("探测查询: %s" % probe["query"])
    for route, detail in probe["routes"].items():
        if "error" in detail:
            print("  [%s] 检索异常: %s" % (route, detail["error"]))
            continue
        print("  [%s] 命中 %d 条" % (route, detail["num_hits"]))
        for item in detail["top"]:
            print("    %d. score=%.4f | %s | %s" % (
                item["rank"], item["score"], item["doc_title"], item["text"]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="医疗 RAG 索引构建：chunks.jsonl -> 三路索引 + dense 向量缓存",
    )
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS, help="chunk JSONL 路径")
    parser.add_argument("--index_dir", default=DEFAULT_INDEX_DIR,
                        help="DenseIndex 向量与 index_summary.json 的输出目录")
    parser.add_argument("--routes", default=DEFAULT_ROUTES,
                        help="要构建的召回路，逗号分隔，可选 dense,bm25,term")
    parser.add_argument("--dense_model", default="",
                        help="本地 embedding 模型目录，留空则自动探测/降级为哈希向量")
    parser.add_argument("--terminology", default="", help="术语表 JSONL，留空用内置默认表")
    parser.add_argument("--device", default="cpu", help="embedding 推理设备，如 cpu / cuda")
    parser.add_argument("--max_rows", type=int, default=-1, help="只取前 N 条 chunk，-1 表示全量")
    parser.add_argument("--summary", default="",
                        help="summary 输出路径，留空则写到 index_dir/index_summary.json")
    parser.add_argument("--probe_query", default=DEFAULT_PROBE_QUERY,
                        help="构建完成后用于试检索的查询，置空则跳过")
    parser.add_argument("--probe_topk", type=int, default=3)
    parser.add_argument("--no_save_dense", action="store_true",
                        help="只构建不落盘 dense 向量")
    parser.add_argument("--no_verify_cache", action="store_true",
                        help="跳过 dense 缓存回读校验")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def run_build(args) -> Dict[str, Any]:
    chunks_path = args.chunks
    if not os.path.exists(chunks_path):
        raise SystemExit("chunk 文件不存在: %s（先跑 build_corpus.py 生成）" % chunks_path)

    started_all = time.perf_counter()
    chunks = load_chunks(chunks_path, max_rows=args.max_rows)
    if not chunks:
        raise SystemExit("chunk 文件为空: %s" % chunks_path)
    logger.info("载入 %d 条 chunk: %s", len(chunks), chunks_path)

    routes = parse_routes(args.routes)
    terminology = load_terminology(args.terminology or None)
    environment = env_report()
    logger.info("环境: %s", json.dumps(environment, ensure_ascii=False))

    index_dir = args.index_dir
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    indexes: Dict[str, BaseIndex] = {}
    route_stats: List[Dict[str, Any]] = []
    for route in routes:
        index, stat = build_one_route(
            route, chunks, args.dense_model or None, terminology, args.device,
        )
        route_stats.append(stat)
        if index is not None:
            indexes[route] = index
        logger.info("[%s] backend=%s size=%d 耗时 %.3fs",
                    route, stat["backend"] or "-", stat["size"], stat["build_seconds"])

    print_route_table(route_stats)

    dense_info: Dict[str, Any] = {"saved": False}
    dense_index = indexes.get("dense")
    if isinstance(dense_index, DenseIndex) and not args.no_save_dense:
        dense_info = save_dense(
            dense_index, index_dir, not args.no_verify_cache,
            args.dense_model or None, args.device,
        )
    elif dense_index is None and "dense" in routes:
        dense_info["error"] = "dense 路未构建成功，无向量可落盘"

    probe: Dict[str, Any] = {}
    if args.probe_query and indexes:
        probe = probe_search(indexes, args.probe_query, max(1, args.probe_topk))
        print_probe(probe)

    summary: Dict[str, Any] = {
        "chunks_path": os.path.abspath(chunks_path),
        "index_dir": os.path.abspath(index_dir),
        "requested_routes": list(routes),
        "built_routes": [stat["route"] for stat in route_stats if stat["ok"]],
        "failed_routes": [stat["route"] for stat in route_stats if not stat["ok"]],
        "dense_model": args.dense_model or "",
        "device": args.device,
        "corpus": corpus_report(chunks),
        "routes": route_stats,
        "dense_cache": dense_info,
        "environment": environment,
        "total_seconds": round(time.perf_counter() - started_all, 3),
    }
    if probe:
        summary["probe"] = probe

    summary_path = args.summary or str(Path(index_dir) / "index_summary.json")
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(summary_path).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("")
    print("summary 已写入: %s" % summary_path)
    if summary["failed_routes"]:
        # 不 exit(1)：RRF 容忍缺路，缺一路仍可检索，这里只把结论说清楚。
        print("注意: 以下召回路构建失败，检索会退化为剩余路 -> %s"
              % ",".join(summary["failed_routes"]))
    return summary


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    run_build(args)


if __name__ == "__main__":
    main()
