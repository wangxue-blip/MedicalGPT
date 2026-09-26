# -*- coding: utf-8 -*-
"""
@description: 医疗 RAG 子系统命令行 demo（medical_rag 包的端到端入口）。

与 demo/chatpdf.py 的差别在于"可追溯"：这里每条证据都带着文档标题、章节路径、
页码、版本与推荐等级，答案生成之后还会再跑一遍逐句引用核查，--explain 还能把
dense / bm25 / term 三路召回各自的排名摊开，便于定位"是哪一路把无关段落顶上来的"。

无 GPU 验证检索链路（不加载任何生成模型）：
python demo/medical_rag_demo.py --chunks data_processed/rag/chunks.jsonl \
    --query "二甲双胍肾功能不全还能用吗" --no_generate --explain

接本地生成模型后进入交互问答：
python demo/medical_rag_demo.py --chunks data_processed/rag/chunks.jsonl \
    --generator_model outputs-sft-qwen-v1 --peft_path outputs-dpo-qwen-v1 --device cuda

不带 --chunks 时会退回 medical_rag.pipeline 自检用的内置样例语料，方便零配置试跑。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_rag import pipeline as rag_pipeline
from medical_rag.pipeline import MedicalRAG, to_jsonable

logger = logging.getLogger("medical_rag_demo")

# 退出交互循环的关键字。留空行也退出，和 demo/inference.py 的习惯一致。
EXIT_WORDS = ("exit", "quit", "q", "退出")

SECTION_WIDTH = 72

# 证据正文在终端里的预览长度。全文打印会把来源清单冲掉，真要看全文用 --dump_json。
PREVIEW_CHARS = 120


# --------------------------------------------------------------------------
# 装配
# --------------------------------------------------------------------------
def build_rag(args):
    """按命令行参数装配 MedicalRAG。

    这里不做任何"依赖是否齐全"的前置检查：medical_rag 各模块自己会降级，
    降级项最终会出现在 retrieval_explain.degraded 里，由 print_environment 统一展示。
    把检查散到 demo 里反而会和库里的降级逻辑打架。
    """
    routes = tuple(item.strip() for item in args.routes.split(",") if item.strip())
    options = dict(
        dense_model=args.dense_model or None,
        reranker_model=args.reranker_model or None,
        generator_model=args.generator_model or None,
        peft_path=args.peft_path or None,
        terminology_path=args.terminology or None,
        device=args.device,
        routes=routes,
        recall_k=args.recall_k,
        fusion_k=args.fusion_k,
        authority_weight=args.authority_weight,
        support_threshold=args.support_threshold,
        dose_exempt=not args.no_dose_exempt,
        min_score=args.min_score,
        min_hits=args.min_hits,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.chunks and Path(args.chunks).exists():
        return MedicalRAG.from_jsonl(args.chunks, max_rows=args.max_rows, **options)

    if args.chunks:
        logger.warning("找不到 %s，改用内置样例语料", args.chunks)
    else:
        logger.info("未指定 --chunks，使用内置样例语料（仅供试跑，不是真实语料）")
    # 复用 pipeline 自检用的样例，避免 demo 里再维护一份会和库漂移的假数据。
    return MedicalRAG.from_chunks(rag_pipeline._demo_chunks(), **options)


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def _fmt_score(value, fmt="%.4f"):
    """None（如未精排）统一渲染成 '-'，不要打印成 'None' 混进数字列。"""
    if value is None:
        return "-"
    try:
        return fmt % float(value)
    except (TypeError, ValueError):
        return str(value)


def _preview(text, limit=PREVIEW_CHARS):
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def _field(obj, name, default=None):
    """兼容 CitationCheck 数据类与 to_jsonable 之后的 dict 两种形态。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def print_environment(rag, explain):
    """先把"这次是在什么条件下跑的"讲清楚，再看结果才有意义。"""
    print("=" * SECTION_WIDTH)
    print("语料 %d 条 chunk | 召回路 %s | topk=%d"
          % (explain["corpus_size"], ",".join(explain["routes"]) or "-", explain["topk"]))
    backends = explain.get("route_backends") or {}
    if backends:
        print("后端: %s" % "  ".join("%s=%s" % (k, v) for k, v in sorted(backends.items())))
    print("精排=%s | 生成=%s | 引用核查=%s"
          % (explain["reranker_enabled"], explain["generator_available"], explain["verifier_enabled"]))
    if explain.get("degraded"):
        # 降级不是错误，但必须显式说出来，否则很容易把"哈希向量召回"当成真语义召回。
        print("降级项: %s" % "; ".join(explain["degraded"]))
    print("=" * SECTION_WIDTH)


def print_sources(result):
    """来源清单：章节 / 页码 / 版本 / 推荐等级都在 citation_label 里，这里只补分数。"""
    sources = result.get("sources") or []
    print("\n【来源】")
    if not sources:
        print("  (无召回)")
        return
    for index, row in enumerate(sources, start=1):
        print("[%d] %s" % (index, row.get("citation") or row.get("chunk_id", "")))
        print("    分级=%-18s 融合分=%s 精排分=%s 最终分=%s"
              % (row.get("source_tier", "-"),
                 _fmt_score(row.get("score")),
                 _fmt_score(row.get("rerank_score")),
                 _fmt_score(row.get("final_score"))))
        print("    正文: %s" % _preview(row.get("text", "")))


def print_verification(verification):
    """引用核查结果。

    verify() 与"未执行核查"的返回结构不同（后者只有 passed/skipped/reason），
    所以全部走 .get()，不要按某一种结构硬取键。
    """
    print("\n【引用核查】")
    if not verification:
        print("  (未执行)")
        return
    if verification.get("skipped"):
        print("  已跳过: %s" % verification.get("reason", ""))
        return

    print("  citation_precision=%.3f  citation_recall=%.3f  句子数=%d"
          % (verification.get("citation_precision", 0.0),
             verification.get("citation_recall", 0.0),
             verification.get("num_sentences", 0)))
    for check in verification.get("checks") or []:
        print("    [%-22s] score=%s cited=%s %s"
              % (_field(check, "reason", ""),
                 _fmt_score(_field(check, "best_score"), "%.3f"),
                 _field(check, "cited", []),
                 _preview(_field(check, "sentence", ""), 60)))
    for key, title in (("unsupported_sentences", "无证据支撑"),
                       ("uncited_sentences", "缺引用"),
                       ("dose_violations", "剂量违规"),
                       ("exempted_doses", "剂量豁免")):
        items = verification.get(key) or []
        if items:
            print("  %s(%d): %s" % (title, len(items), _preview(items[0], 60)))


def print_explain(result):
    """三路检索明细：每条命中在 dense / bm25 / term 各自的名次与加权前后的分数。"""
    explain = result["retrieval_explain"]
    routes = explain["routes"] or []
    ranking = explain.get("ranking") or []

    print("\n【三路检索明细】")
    if not ranking:
        print("  (无召回) 原因: %s" % (explain.get("refuse_reason") or "未知"))
        return

    # 表头里的中文是双宽字符，占位宽度要比数据行少 3，两行才对得齐。
    print("  %-2s %-18s %s %7s %7s %7s" % (
        "排名", "chunk_id", " ".join("%-6s" % r for r in routes), "融合分", "精排分", "最终分"))
    for row in ranking:
        ranks = row.get("route_ranks") or {}
        # 某一路没召回这条时 route_ranks 里是 None，打成 '-'，一眼能看出是不是单路孤证。
        rank_cells = " ".join("%-6s" % ("-" if ranks.get(r) is None else ranks[r]) for r in routes)
        print("  %-4d %-18s %s %10s %10s %10s"
              % (row["rank"], row["chunk_id"], rank_cells,
                 _fmt_score(row.get("fusion_score"), "%.6f"),
                 _fmt_score(row.get("rerank_score")),
                 _fmt_score(row.get("final_score"))))
    print("  耗时(ms): %s" % explain.get("timing_ms", {}))


def print_result(rag, result, show_explain=False, show_prompt=False):
    explain = result["retrieval_explain"]
    print_environment(rag, explain)
    print("\n问题: %s" % result["query"])

    print("\n【回答】")
    if result["refused"]:
        print("  %s" % (result["answer"] or ""))
        print("  拒答原因: %s" % (explain.get("refuse_reason") or "证据不足"))
    elif result["answer"]:
        print(result["answer"])
    else:
        # 没接生成模型是常态（本地只验证检索），要说清楚不是模型答不出来。
        print("  (未生成：当前为检索验证模式，检索与 prompt 组装均已完成)")

    print_sources(result)
    print_verification(result.get("verification"))

    if show_explain:
        print_explain(result)

    if show_prompt and result["prompt"]:
        print("\n【prompt】")
        print(result["prompt"])


# --------------------------------------------------------------------------
# 单次 / 交互
# --------------------------------------------------------------------------
def answer_once(rag, query, args):
    result = rag.answer(query, topk=args.topk, generate=not args.no_generate)
    print_result(rag, result, show_explain=args.explain, show_prompt=args.show_prompt)
    return result


def interactive_loop(rag, args):
    """交互问答。空行或退出词结束，Ctrl+C / Ctrl+D 也当作正常退出。"""
    print("进入交互模式，输入问题开始提问；空行或 %s 退出。" % "/".join(EXIT_WORDS))
    while True:
        try:
            query = input("\n提问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in EXIT_WORDS:
            break
        try:
            answer_once(rag, query, args)
        except Exception as exc:
            # 单条问题炸掉不应该结束整个会话，医生在旁边试用时体验差别很大。
            logger.exception("本次提问处理失败: %s", exc)
            print("本次提问处理失败: %s" % exc)
    print("已退出。")


def dump_json(result, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(result), handle, ensure_ascii=False, indent=2)
    print("\n完整结果已写入 %s" % out)


def parse_args():
    parser = argparse.ArgumentParser(
        description="医疗 RAG demo：带出处的检索问答 + 引用核查。"
    )
    parser.add_argument("--chunks", default="", help="chunk JSONL 路径（medical_rag.chunker 的产物）")
    parser.add_argument("--query", default="", help="单次提问；不给则进入交互循环")
    parser.add_argument("--topk", type=int, default=5, help="送进 prompt 的证据条数")
    parser.add_argument("--explain", action="store_true", help="打印三路召回排名明细")
    parser.add_argument("--show_prompt", action="store_true", help="打印完整 prompt")
    parser.add_argument("--no_generate", action="store_true",
                        help="只跑检索与 prompt 组装，不加载生成模型（无 GPU 验证用）")

    parser.add_argument("--routes", default="dense,bm25,term", help="启用的召回路，逗号分隔")
    parser.add_argument("--recall_k", type=int, default=50, help="每路召回深度")
    parser.add_argument("--fusion_k", type=int, default=60, help="RRF 平滑常数")
    parser.add_argument("--authority_weight", type=float, default=0.15, help="来源权威度权重")
    parser.add_argument("--max_rows", type=int, default=-1, help="只载入前 N 条 chunk，调试用")

    parser.add_argument("--dense_model", default="", help="本地 embedding 模型目录")
    parser.add_argument("--reranker_model", default="", help="交叉编码器模型目录")
    parser.add_argument("--generator_model", default="", help="生成模型目录，留空则不生成")
    parser.add_argument("--peft_path", default="", help="LoRA 适配器目录")
    parser.add_argument("--terminology", default="", help="外部术语表 JSONL")
    parser.add_argument("--device", default="cpu", help="cpu / cuda")

    parser.add_argument("--min_score", type=float, default=0.10,
                        help="拒答闸门的最低绝对相关度(0~1)，0 表示关闭闸门")
    parser.add_argument("--min_hits", type=int, default=1, help="拒答命中数下限")
    parser.add_argument("--support_threshold", type=float, default=0.62, help="引用支持度阈值")
    parser.add_argument("--no_dose_exempt", action="store_true", help="关闭剂量豁免，任何剂量句都算违规")

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--dump_json", default="", help="把单次提问的完整结果写到该 JSON 文件")
    parser.add_argument("--log_level", default="WARNING", help="日志级别，排查降级原因时设为 INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    rag = build_rag(args)
    print(repr(rag))

    if args.query:
        result = answer_once(rag, args.query, args)
        if args.dump_json:
            dump_json(result, args.dump_json)
        return

    if args.dump_json:
        logger.warning("--dump_json 只在单次提问（--query）时生效，交互模式下已忽略")
    interactive_loop(rag, args)


if __name__ == "__main__":
    main()
