# -*- coding: utf-8 -*-
"""medical_rag 子系统单元测试。

刻意只用标准库 + pytest。本仓库的目标运行环境（医院内网机、无 GPU 的开发机）
经常没有 torch / transformers / sentence_transformers / faiss / jieba /
rank_bm25 / numpy，因此这些用例全部走降级路径：不加载任何外部模型、不联网、
不依赖可选后端。否则"优雅降级"就只是 docstring 里的口号，没人验证过。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_rag.chunker import StructureAwareChunker, chunk_documents
from medical_rag.fusion import reciprocal_rank_fusion
from medical_rag.index import BM25Index, build_indexes
from medical_rag.pipeline import MedicalRAG
from medical_rag.rerank import apply_authority_boost
from medical_rag.retriever import MedicalRetriever
from medical_rag.schema import (
    CONTENT_TABLE,
    CONTENT_TEXT,
    TIER_AUTHORITY,
    TIER_CONSULTATION,
    TIER_DRUG_LABEL,
    TIER_GUIDELINE,
    Chunk,
    RawDoc,
    RetrievalHit,
)
from medical_rag.terminology import Terminology
from medical_rag.verifier import CitationVerifier, should_refuse


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------
def make_chunk(chunk_id, text, tier=TIER_GUIDELINE, **kwargs):
    """构造一条元数据齐全的 chunk。authority 按来源分级取值，和 chunker 口径一致。"""
    kwargs.setdefault("doc_id", "doc-1")
    kwargs.setdefault("doc_title", "示例资料")
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_tier=tier,
        authority=TIER_AUTHORITY[tier],
        **kwargs
    )


def make_hit(text, tier=TIER_GUIDELINE, score=0.0, chunk_id="c1", **kwargs):
    return RetrievalHit(chunk=make_chunk(chunk_id, text, tier, **kwargs), score=score)


BODY = (
    "对于确诊的2型糖尿病患者，如无禁忌证，二甲双胍是一线首选降糖药物。"
    "用药期间应定期监测肾功能，估算肾小球滤过率低于45时需要考虑减量。"
    "估算肾小球滤过率低于30时应当停用，并由医生重新评估替代方案。"
)

TABLE = (
    "| 药物 | 起始剂量 | 最大日剂量 |\n"
    "| --- | --- | --- |\n"
    "| 二甲双胍 | 0.5g 每日2次 | 2.0g |\n"
    "| 格列美脲 | 1mg 每日1次 | 6mg |"
)


def guideline_doc():
    return RawDoc(
        doc_id="gl-1",
        doc_title="中国2型糖尿病防治指南",
        source_tier=TIER_GUIDELINE,
        source_path="data_raw/gl.pdf",
        version="2020",
        blocks=[
            {"text": "第三章 药物治疗", "is_heading": True, "heading_level": 1, "page": 10},
            {"text": "3.1 一线用药", "is_heading": True, "heading_level": 2, "page": 10},
            {"text": BODY, "page": 10, "content_type": CONTENT_TEXT},
            {"text": TABLE, "page": 11, "content_type": CONTENT_TABLE},
        ],
    )


# --------------------------------------------------------------------------
# schema：序列化与出处
# --------------------------------------------------------------------------
def test_chunk_roundtrip_keeps_every_provenance_field():
    chunk = make_chunk(
        "c-1",
        "二甲双胍是一线用药。",
        TIER_GUIDELINE,
        parent_text="完整段落……",
        source_path="data_raw/gl.pdf",
        version="2020",
        section_path=["第三章 药物治疗", "3.1 一线用药"],
        page=42,
        evidence_level="I",
        extra={"parent_id": "p-1", "child_index": 0},
    )
    restored = Chunk.from_dict(chunk.to_dict())
    # dataclass 的 __eq__ 逐字段比较：任何一个出处字段掉了这里就会红。
    assert restored == chunk
    assert restored.section_path == ["第三章 药物治疗", "3.1 一线用药"]
    assert restored.extra["parent_id"] == "p-1"


def test_chunk_from_dict_tolerates_extra_keys_and_fills_authority():
    restored = Chunk.from_dict({
        "chunk_id": "c-2",
        "text": "低血糖处理原则。",
        "doc_id": "d",
        "doc_title": "内科学",
        "source_tier": TIER_DRUG_LABEL,
        "section_path": None,
        "不认识的字段": "会被丢掉",
    })
    assert restored.section_path == []
    # authority 没给就按来源分级补齐，否则后续的权威度加权会整体失真。
    assert restored.authority == TIER_AUTHORITY[TIER_DRUG_LABEL]


def test_citation_label_carries_section_page_version_and_evidence_level():
    chunk = make_chunk(
        "c-3",
        "推荐使用二甲双胍。",
        TIER_GUIDELINE,
        doc_title="中国2型糖尿病防治指南",
        section_path=["第三章 药物治疗", "3.1 一线用药"],
        page=42,
        version="2020",
        evidence_level="I",
    )
    label = chunk.citation_label()
    assert "中国2型糖尿病防治指南" in label
    assert "第三章 药物治疗 > 3.1 一线用药" in label
    assert "第 42 页" in label
    assert "2020" in label
    assert "推荐等级 I" in label


def test_context_text_prefers_parent_for_small_to_big():
    chunk = make_chunk("c-4", "子块。", parent_text="父块提供完整上下文。")
    assert chunk.context_text() == "父块提供完整上下文。"
    assert make_chunk("c-5", "只有子块。").context_text() == "只有子块。"


# --------------------------------------------------------------------------
# chunker：章节继承与原子块
# --------------------------------------------------------------------------
def test_chunker_inherits_section_path_and_metadata():
    chunks = StructureAwareChunker(
        child_chars=60, child_overlap=10, parent_chars=300, min_chars=10
    ).chunk_document(guideline_doc())

    text_chunks = [c for c in chunks if c.content_type == CONTENT_TEXT]
    assert len(text_chunks) >= 2, "正文应当被切成多个 child"
    for chunk in text_chunks:
        assert chunk.section_path == ["第三章 药物治疗", "3.1 一线用药"]
        assert chunk.page == 10
        assert chunk.version == "2020"
        assert chunk.doc_title == "中国2型糖尿病防治指南"
        assert chunk.authority == TIER_AUTHORITY[TIER_GUIDELINE]
        # small-to-big：每个 child 的 parent_text 都要覆盖整段，
        # 否则生成时会丢掉"低于30应停用"这类限定条件。
        assert "一线首选降糖药物" in chunk.parent_text
        assert "重新评估替代方案" in chunk.parent_text


def test_chunker_pops_stack_when_sibling_heading_arrives():
    doc = RawDoc(
        doc_id="gl-2",
        doc_title="诊疗规范",
        source_tier=TIER_GUIDELINE,
        blocks=[
            {"text": "1 概述", "is_heading": True, "heading_level": 1},
            {"text": "1.1 定义", "is_heading": True, "heading_level": 2},
            {"text": "本病是一组以慢性高血糖为特征的代谢性疾病，由胰岛素分泌或作用缺陷引起。"},
            {"text": "2 诊断", "is_heading": True, "heading_level": 1},
            {"text": "空腹血糖达到七点零毫摩尔每升，或糖化血红蛋白达到六点五即可诊断本病。"},
        ],
    )
    chunks = StructureAwareChunker(child_chars=200, parent_chars=400, min_chars=10).chunk_document(doc)
    paths = [chunk.section_path for chunk in chunks]
    assert ["1 概述", "1.1 定义"] in paths
    # 同级标题必须把更深的层级顶掉，而不是堆成 ["1 概述", "1.1 定义", "2 诊断"]
    assert ["2 诊断"] in paths


def test_chunker_keeps_table_atomic():
    chunks = StructureAwareChunker(
        child_chars=60, child_overlap=10, parent_chars=300, min_chars=10
    ).chunk_document(guideline_doc())

    tables = [c for c in chunks if c.content_type == CONTENT_TABLE]
    assert len(tables) == 1, "表格应当整块保留，不能按 child_chars 切开"
    table = tables[0]
    assert len(table.text) > 60, "表格长度超过 child_chars 仍未被切碎"
    # 首行表头与最后一行数据都在，说明整张表完整
    assert "最大日剂量" in table.text
    assert "格列美脲" in table.text
    assert table.parent_text == table.text
    assert table.extra.get("atomic") is True
    assert table.section_path == ["第三章 药物治疗", "3.1 一线用药"]
    assert table.page == 11


def test_chunk_documents_accepts_rawdoc_list():
    chunks = chunk_documents([guideline_doc()], child_chars=80, parent_chars=300, min_chars=10)
    assert chunks
    assert all(chunk.doc_id == "gl-1" for chunk in chunks)


# --------------------------------------------------------------------------
# terminology
# --------------------------------------------------------------------------
def test_terminology_normalizes_alias_to_canonical():
    term = Terminology.default()
    assert term.normalize("心梗") == "心肌梗死"
    assert term.normalize("慢阻肺") == "慢性阻塞性肺疾病"
    # 表里没有的词原样返回，不做任何猜测
    assert term.normalize("某个不存在的词") == "某个不存在的词"


def test_terminology_accepts_custom_groups():
    term = Terminology([["二甲双胍", "metformin", "格华止"]])
    assert term.normalize("metformin") == "二甲双胍"
    assert term.normalize("格华止") == "二甲双胍"
    assert "二甲双胍" in term.terms_in("患者长期服用格华止控制血糖")


def test_terminology_expand_query_keeps_original_first_and_caps_size():
    term = Terminology.default()
    variants = term.expand_query("心梗后还能吃阿司匹林吗", max_expansions=5)
    assert variants[0] == "心梗后还能吃阿司匹林吗", "第一个变体必须是原始 query"
    assert len(variants) <= 5
    assert any("心肌梗死" in item for item in variants), "应当给出规范名变体"
    assert len(set(variants)) == len(variants), "变体之间不应重复"


def test_terminology_expand_query_without_known_term_returns_single():
    term = Terminology.default()
    assert term.expand_query("今天天气怎么样", max_expansions=5) == ["今天天气怎么样"]


def test_terminology_terms_in_takes_longest_match():
    term = Terminology.default()
    terms = term.terms_in("患者为2型糖尿病合并高血压")
    # 最长匹配：命中"2型糖尿病"后不应再把内部的"糖尿病"识别成第二个术语
    assert "2型糖尿病" in terms
    assert "糖尿病" not in terms
    assert "高血压" in terms


# --------------------------------------------------------------------------
# BM25：英文药名与数字的精确匹配
# --------------------------------------------------------------------------
def bm25_corpus():
    return [
        make_chunk("bm-0", "二甲双胍片 metformin 起始剂量 500mg，每日两次，随餐服用。", TIER_DRUG_LABEL),
        make_chunk("bm-1", "阿卡波糖 acarbose 起始剂量 50mg，每日三次，与第一口主食同服。", TIER_DRUG_LABEL),
        make_chunk("bm-2", "生活方式干预包括饮食控制、规律运动与体重管理。", TIER_GUIDELINE),
    ]


def test_bm25_matches_english_drug_name_exactly():
    index = BM25Index()
    index.build(bm25_corpus())
    assert index.backend in ("rank_bm25", "builtin_bm25")

    hits = index.search("metformin", topk=3)
    assert hits, "英文药名必须召回"
    assert hits[0][0] == 0

    hits = index.search("acarbose", topk=3)
    assert hits[0][0] == 1


def test_bm25_distinguishes_dose_numbers():
    index = BM25Index()
    index.build(bm25_corpus())
    # 500mg 与 50mg 是两个不同的 token，不能被切成 bigram 后混为一谈
    assert index.search("500mg", topk=3)[0][0] == 0
    assert index.search("50mg", topk=3)[0][0] == 1


def test_bm25_returns_empty_for_unmatched_query():
    index = BM25Index()
    index.build(bm25_corpus())
    assert index.search("metformin", topk=0) == []
    assert BM25Index().search("metformin", topk=3) == [], "未 build 的索引应返回空而不是抛异常"


# --------------------------------------------------------------------------
# RRF 融合
# --------------------------------------------------------------------------
def fusion_corpus():
    return [make_chunk("f-%d" % i, "第%d条证据内容。" % i) for i in range(3)]


def test_rrf_rewards_multi_route_consensus():
    chunks = fusion_corpus()
    route_hits = {
        "dense": [(2, 0.90), (0, 0.50)],
        "bm25": [(0, 0.80), (1, 0.40)],
        "term": [(0, 0.70)],
    }
    hits = reciprocal_rank_fusion(route_hits, chunks, k=60)

    # f-0 只在 dense 里排第 2，但三路都召回了它；f-2 是 dense 第 1 的单路孤证。
    assert [hit.chunk.chunk_id for hit in hits] == ["f-0", "f-2", "f-1"]
    assert hits[0].score > hits[1].score

    assert hits[0].route_ranks == {"dense": 2, "bm25": 1, "term": 1}
    # 没召回的路必须显式写成 None，否则 debug 时看不出是共识还是孤证
    assert hits[1].route_ranks["bm25"] is None
    assert hits[1].route_ranks["dense"] == 1
    assert hits[0].route_scores["bm25"] == 0.80


def test_rrf_weights_can_flip_the_order():
    chunks = fusion_corpus()
    route_hits = {"dense": [(2, 0.9), (0, 0.5)], "bm25": [(0, 0.8)], "term": [(0, 0.7)]}

    assert reciprocal_rank_fusion(route_hits, chunks, k=60)[0].chunk.chunk_id == "f-0"
    # 把词面与术语两路的话语权调到 0（消融实验的典型用法）后只剩 dense 的排序，
    # 原本靠三路共识上位的 f-0 就让位给 dense 第一名的 f-2。
    ablated = reciprocal_rank_fusion(route_hits, chunks, k=60, weights={"bm25": 0.0, "term": 0.0})
    assert ablated[0].chunk.chunk_id == "f-2"


def test_rrf_skips_out_of_range_index_and_empty_input():
    chunks = fusion_corpus()
    hits = reciprocal_rank_fusion({"dense": [(99, 0.9), (1, 0.5)]}, chunks, k=60)
    assert [hit.chunk.chunk_id for hit in hits] == ["f-1"]
    assert reciprocal_rank_fusion({}, chunks) == []
    assert reciprocal_rank_fusion({"dense": [(0, 1.0)]}, []) == []


# --------------------------------------------------------------------------
# 权威度加权
# --------------------------------------------------------------------------
def test_authority_boost_breaks_ties_in_favor_of_higher_tier():
    consultation = make_hit("同一句话。", TIER_CONSULTATION, score=0.05, chunk_id="low")
    guideline = make_hit("同一句话。", TIER_GUIDELINE, score=0.05, chunk_id="high")

    ordered = apply_authority_boost([consultation, guideline], authority_weight=0.15, recency_weight=0.05)
    assert ordered[0].chunk.chunk_id == "high"
    assert ordered[0].final_score > ordered[1].final_score
    # final_score 必须被写回，下游拒答闸门读的就是它
    assert all(hit.final_score > 0 for hit in ordered)


def test_authority_boost_is_a_tiebreaker_not_the_main_ranker():
    strong = make_hit("语义上高度相关。", TIER_CONSULTATION, score=0.05, chunk_id="relevant")
    weak = make_hit("语义上基本无关。", TIER_GUIDELINE, score=0.01, chunk_id="authoritative")
    ordered = apply_authority_boost([strong, weak], authority_weight=0.15, recency_weight=0.05)
    # 语义分差得够远时，指南也翻不过来——这正是"裁判而非主排序"的设计意图
    assert ordered[0].chunk.chunk_id == "relevant"


def test_authority_boost_prefers_newer_version_at_same_tier():
    old = make_hit("同一句话。", TIER_GUIDELINE, score=0.05, chunk_id="old", version="2012")
    new = make_hit("同一句话。", TIER_GUIDELINE, score=0.05, chunk_id="new", version="2023")
    ordered = apply_authority_boost([old, new], authority_weight=0.15, recency_weight=0.05)
    assert ordered[0].chunk.chunk_id == "new"


def test_authority_boost_handles_empty_input():
    assert apply_authority_boost([]) == []


# --------------------------------------------------------------------------
# 引用核查
# --------------------------------------------------------------------------
DOSE_EVIDENCE = "二甲双胍普通片起始剂量为每次0.5g，每日2次，随餐服用可减轻胃肠道反应。"


def test_verifier_marks_out_of_range_citation_as_unsupported():
    hits = [make_hit("二甲双胍常见不良反应为胃肠道反应，多在用药初期出现。")]
    result = CitationVerifier().verify(
        "二甲双胍常见不良反应为胃肠道反应，多在用药初期出现[3]。", hits
    )
    check = result["checks"][0]
    assert check.supported is False
    # 编号越界说明模型在编引用，内容碰巧正确也不能算可追溯
    assert check.reason == "citation_out_of_range"
    assert check.best_score == 0.0
    assert result["citation_precision"] == 0.0


def test_verifier_supports_sentence_backed_by_cited_evidence():
    hits = [make_hit(DOSE_EVIDENCE, TIER_DRUG_LABEL)]
    result = CitationVerifier().verify(DOSE_EVIDENCE[:-1] + "[1]。", hits)
    check = result["checks"][0]
    assert check.supported is True
    assert check.reason == "supported"
    assert result["citation_precision"] == 1.0


def test_verifier_exempts_dose_that_is_quoted_from_evidence():
    hits = [make_hit(DOSE_EVIDENCE, TIER_DRUG_LABEL)]
    answer = "二甲双胍普通片起始剂量为每次0.5g，每日2次，随餐服用可减轻胃肠道反应[1]。"
    result = CitationVerifier(dose_exempt=True).verify(answer, hits)
    # 剂量是从被引原文里转述来的，数字对得上，不算违规
    assert result["exempted_doses"], "应当命中剂量豁免"
    assert result["dose_violations"] == []


def test_verifier_flags_dose_numbers_not_present_in_evidence():
    hits = [make_hit(DOSE_EVIDENCE, TIER_DRUG_LABEL)]
    answer = "二甲双胍普通片起始剂量为每次2.5g，每日9次，随餐服用可减轻胃肠道反应[1]。"
    result = CitationVerifier(dose_exempt=True).verify(answer, hits)
    # 把 0.5g 说成 2.5g 是最危险的一类幻觉：句子读起来像转述，数字却是自创的
    assert result["dose_violations"], "证据里没有的剂量数字必须判违规"
    assert result["exempted_doses"] == []


def test_verifier_flags_uncited_dose_even_when_numbers_match():
    hits = [make_hit(DOSE_EVIDENCE, TIER_DRUG_LABEL)]
    result = CitationVerifier(dose_exempt=True).verify(
        "二甲双胍普通片起始剂量为每次0.5g，每日2次。", hits
    )
    # 没有引用的剂量必然是模型自己说的，属最高风险，豁免不适用
    assert result["dose_violations"]
    assert result["uncited_sentences"]


def test_verifier_flags_bare_dose_not_adjacent_to_frequency_word():
    """回归：动词隔开的裸剂量也必须被判违规。

    reward 那份 DOSE_PATTERNS 只认"每次<数字><单位>"的紧邻写法，而模型编剂量时
    常写成"每次加服500mg""建议服用500mg"——中间隔了动词。早期实现直接复用
    reward 正则，导致这类句子既进不了 exempted_doses 也进不了 dose_violations，
    凭空编的剂量被静默放行。
    """
    hits = [make_hit(DOSE_EVIDENCE, TIER_DRUG_LABEL)]
    result = CitationVerifier(dose_exempt=True).verify(
        "此外建议每次加服500mg以增强疗效。", hits
    )
    assert result["dose_violations"], "隔了动词的裸剂量同样属于自创剂量，必须判违规"
    assert result["exempted_doses"] == []


def test_verifier_does_not_treat_body_weight_as_dose():
    """体重/年份这类描述性数字不是剂量，放宽召回后不能误伤。"""
    evidence = "患者体重60kg，于2023年确诊，随访期间未出现不良反应。"
    hits = [make_hit(evidence, TIER_DRUG_LABEL)]
    result = CitationVerifier(dose_exempt=True).verify(evidence[:-1] + "[1]。", hits)
    assert result["dose_violations"] == []
    assert result["exempted_doses"] == []


def test_verifier_disabling_dose_exempt_flags_every_dose_sentence():
    hits = [make_hit(DOSE_EVIDENCE, TIER_DRUG_LABEL)]
    answer = "二甲双胍普通片起始剂量为每次0.5g，每日2次，随餐服用可减轻胃肠道反应[1]。"
    result = CitationVerifier(dose_exempt=False).verify(answer, hits)
    assert result["dose_violations"]
    assert result["exempted_doses"] == []


def test_verifier_handles_empty_answer_without_dividing_by_zero():
    result = CitationVerifier().verify("", [make_hit("任意证据。")])
    assert result["citation_precision"] == 0.0
    assert result["citation_recall"] == 0.0
    assert result["num_sentences"] == 0


# --------------------------------------------------------------------------
# 拒答闸门
# --------------------------------------------------------------------------
def test_should_refuse_on_empty_evidence():
    assert should_refuse([], min_score=0.0, min_hits=1) is True


def test_should_refuse_respects_score_and_hit_thresholds():
    hit = make_hit("足够相关的证据。")
    hit.final_score = 0.5
    assert should_refuse([hit], min_score=0.0, min_hits=1) is False
    assert should_refuse([hit], min_score=0.9, min_hits=1) is True
    assert should_refuse([hit], min_score=0.0, min_hits=2) is True


# --------------------------------------------------------------------------
# 端到端：裸环境下整条链路必须跑通
# --------------------------------------------------------------------------
def e2e_corpus():
    return [
        make_chunk(
            "e-0",
            "二甲双胍是2型糖尿病的一线降糖药物，起始剂量通常为500mg每日两次，随餐服用。",
            TIER_GUIDELINE,
            doc_title="中国2型糖尿病防治指南",
            section_path=["第四章 口服降糖药", "双胍类"],
            page=88,
            version="2020",
            evidence_level="I",
        ),
        make_chunk(
            "e-1",
            "估算肾小球滤过率低于30的患者禁用二甲双胍，使用含碘对比剂前应暂停给药。",
            TIER_DRUG_LABEL,
            doc_title="盐酸二甲双胍片说明书",
            section_path=["禁忌"],
            version="2021",
        ),
        make_chunk("e-2", "低血糖表现为心悸、出汗、手抖，需及时补充糖分。", TIER_CONSULTATION),
    ]


def build_offline_rag():
    """手工装配管线，刻意不走 MedicalRAG.from_chunks。

    from_chunks 会构造 CrossEncoderReranker 并探测 available，在装了
    sentence_transformers 的机器上那一步会去拉 bge-reranker 权重。单测必须
    离线可复现，所以这里显式传 reranker=None / generator=None。
    """
    chunks = e2e_corpus()
    terminology = Terminology.default()
    indexes = build_indexes(chunks, dense_model=None, terminology=terminology,
                            routes=("dense", "bm25", "term"))
    retriever = MedicalRetriever(chunks, indexes, reranker=None, terminology=terminology, recall_k=10)
    return MedicalRAG(
        chunks=chunks,
        indexes=indexes,
        retriever=retriever,
        verifier=CitationVerifier(),
        generator=None,
        terminology=terminology,
    )


def test_pipeline_retrieves_and_builds_prompt_without_heavy_deps():
    rag = build_offline_rag()
    result = rag.answer("二甲双胍肾功能不好还能用吗", topk=3, generate=False)

    assert set(result) >= {
        "query", "refused", "answer", "sources", "verification", "prompt", "retrieval_explain"
    }
    assert result["refused"] is False
    assert result["sources"], "三路降级后端至少应召回一条"
    assert "【证据】" in result["prompt"]
    # 出处必须一路带到 prompt 与来源清单里
    assert any("说明书" in row["citation"] for row in result["sources"])

    explain = result["retrieval_explain"]
    assert set(explain["routes"]) == {"dense", "bm25", "term"}
    assert explain["generator_available"] is False
    assert explain["ranking"]
    for row in explain["ranking"]:
        assert set(row["route_ranks"]) == {"dense", "bm25", "term"}


def test_pipeline_refuses_when_evidence_is_empty():
    rag = MedicalRAG(chunks=[], indexes={}, retriever=None, verifier=CitationVerifier())
    result = rag.answer("任何问题", topk=3, generate=False)
    assert result["refused"] is True
    assert result["prompt"] == "", "拒答时不应组装 prompt"
    assert result["verification"]["skipped"] is True


# --------------------------------------------------------------------------
# 回归：审查确认的缺陷
# --------------------------------------------------------------------------
def test_chunker_keeps_short_section_instead_of_dropping_it():
    """短于 min_chars 的 section 必须保留，不能整段从索引里消失。

    早先 _flush_section 直接 continue 丢弃短 child，结果是"孕妇禁用本品"这类
    20~35 字的关键禁忌句只存在于 parent_text 里，任何 query 都召不回它。
    """
    doc = RawDoc(
        doc_id="short-1",
        doc_title="说明书",
        source_tier=TIER_DRUG_LABEL,
        blocks=[
            {"text": "第一章 总则", "is_heading": True, "heading_level": 1},
            {"text": "孕妇禁用。"},
        ],
    )
    chunks = StructureAwareChunker().chunk_document(doc)
    assert chunks, "短 section 不能产出 0 条 chunk"
    assert any("孕妇禁用" in chunk.text for chunk in chunks)


def test_chunker_merges_short_tail_child_into_previous_one():
    """贪心打包后的尾组短于 min_chars 时要并入上一条，而不是丢掉。"""
    body = "甲" * 100 + "。" + "乙" * 100 + "。" + "丙" * 100 + "。" + "妊娠期妇女禁用本品。"
    doc = RawDoc(doc_id="tail-1", doc_title="说明书", source_tier=TIER_DRUG_LABEL,
                 blocks=[{"text": body}])
    chunks = StructureAwareChunker(child_chars=320, child_overlap=64,
                                   parent_chars=1200, min_chars=40).chunk_document(doc)
    assert any("妊娠期妇女禁用本品" in chunk.text for chunk in chunks)
    # 合并时要消掉重叠，同一句话不能在一条 chunk 里出现两遍
    for chunk in chunks:
        assert chunk.text.count("妊娠期妇女禁用本品") <= 1


def test_chunker_does_not_treat_dose_or_threshold_lines_as_headings():
    """以数字开头的剂量/阈值行不是标题：判成标题就等于把它从索引里删掉。"""
    from medical_rag.chunker import _looks_like_heading

    for line in ["2型糖尿病患者血压应控制在130/80mmHg以下",
                 "100mg 阿司匹林每日一次长期维持",
                 "1．阿司匹林75~100mg 每日1次"]:
        assert _looks_like_heading(line) is False, line
    # 真正的标题仍然要能认出来
    assert _looks_like_heading("第三章 药物治疗") is True
    assert _looks_like_heading("3.1 一线用药") is True

    doc = RawDoc(doc_id="dose-1", doc_title="指南", source_tier=TIER_GUIDELINE,
                 blocks=[{"text": "2型糖尿病患者血压应控制在130/80mmHg以下"}])
    chunks = StructureAwareChunker(min_chars=10).chunk_document(doc)
    assert any("130/80mmHg" in chunk.text for chunk in chunks), "血压目标值必须进索引"


def test_fallback_heading_does_not_pop_numbered_ancestors():
    """无编号小标题不能把带编号的祖先章节整栈弹掉。"""
    doc = RawDoc(
        doc_id="hd-1",
        doc_title="指南",
        source_tier=TIER_GUIDELINE,
        blocks=[
            {"text": "第四章 口服降糖药", "is_heading": True, "heading_level": 1, "page": 10},
            {"text": "双胍类", "is_heading": True, "page": 10},
            {"text": "二甲双胍为一线用药，应从小剂量开始并逐步加量以减轻胃肠道反应。", "page": 10},
        ],
    )
    chunks = StructureAwareChunker(min_chars=10).chunk_document(doc)
    assert chunks
    assert chunks[0].section_path == ["第四章 口服降糖药", "双胍类"]


def test_evidence_level_does_not_flip_class_three_into_class_one():
    """III / II 类推荐不能被读成 I 类——那是把"有害，不应实施"翻成"强烈推荐"。"""
    from medical_rag.chunker import evidence_levels_in

    assert evidence_levels_in("（III类推荐）") == ["III"]
    assert evidence_levels_in("（II类推荐）") == ["II"]
    assert evidence_levels_in("（IIa类推荐）") == ["IIa"]
    assert evidence_levels_in("（I类推荐）") == ["I"]
    assert evidence_levels_in("（Ⅲ类推荐）") == ["III"]

    doc = RawDoc(doc_id="ev-1", doc_title="指南2", source_tier=TIER_GUIDELINE, blocks=[
        {"text": "对于出血高危患者，不建议常规联合三联抗栓治疗（III类推荐），因为出血风险超过获益。"},
    ])
    chunks = StructureAwareChunker(min_chars=10).chunk_document(doc)
    assert chunks[0].evidence_level == "III"


def test_evidence_level_is_blank_when_parent_mixes_classes():
    """父块里同时有 I 类与 III 类时，无标记 child 宁可留空也不能瞎继承。"""
    from medical_rag.chunker import _resolve_evidence_level

    parent = "应尽早启动他汀治疗（I类推荐）。对于出血高危患者不建议三联抗栓（III类推荐）。"
    assert _resolve_evidence_level("某个没有标记的子块。", parent, []) == ""
    # 只有一种标记时照常继承，否则检索命中段落中段的 child 会丢掉强度信息
    single = "应尽早启动他汀治疗（I类推荐），并长期随访监测肝酶。"
    assert _resolve_evidence_level("某个没有标记的子块。", single, []) == "I"


def test_verifier_out_of_range_citation_does_not_count_as_recall():
    """编号全靠编的答案不能拿到 citation_recall = 1.0。"""
    hits = [make_hit("二甲双胍用于2型糖尿病。", chunk_id="v1"),
            make_hit("肾功能不全者慎用。", chunk_id="v2")]
    result = CitationVerifier().verify(
        "二甲双胍能治愈糖尿病并逆转肾损伤[7]。它还能显著改善认知功能与寿命[8]。", hits
    )
    assert result["citation_precision"] == 0.0
    assert result["citation_recall"] == 0.0
    assert len(result["fabricated_citations"]) == 2


def test_verifier_flags_dose_whose_unit_was_altered():
    """单位被篡改（0.5g -> 0.5mg，差 1000 倍）必须落进 dose_violations。"""
    evidence = "二甲双胍缓释片起始剂量为每次0.5g，每日1次，随晚餐服用。"
    hits = [make_hit(evidence, TIER_DRUG_LABEL)]
    result = CitationVerifier(dose_exempt=True).verify(
        "二甲双胍缓释片起始剂量为每次0.5mg，每日1次，随晚餐服用[1]。", hits
    )
    assert result["dose_violations"]
    assert result["exempted_doses"] == []


def test_verifier_rejects_dose_number_borrowed_from_non_dose_context():
    """原文里的病型编号 / 化验阈值不能给凭空编的剂量做担保。"""
    evidence = ("二甲双胍缓释片起始剂量为每次0.5g，每日1次，随晚餐服用。"
                "2型糖尿病患者eGFR低于30时禁用。")
    hits = [make_hit(evidence, TIER_DRUG_LABEL)]
    for answer in ["二甲双胍缓释片起始剂量为每次2片，每日1次，随晚餐服用[1]。",
                   "二甲双胍缓释片起始剂量为每次30mg，每日1次，随晚餐服用[1]。"]:
        result = CitationVerifier(dose_exempt=True).verify(answer, hits)
        assert result["dose_violations"], answer
        assert result["exempted_doses"] == [], answer


def test_should_refuse_blocks_chunk_unrelated_to_query():
    """与 query 完全无关的 chunk 必须被闸门拦下——闸门不能建立在归一化分上。"""
    from medical_rag.verifier import DEFAULT_MIN_SCORE

    hit = make_hit("完全不相关的内容，今天天气很好适合散步。", TIER_CONSULTATION, score=0.0163)
    apply_authority_boost([hit])
    # final_score 被归一化抬到 1.0 以上，正是它不能用来当闸门判据的原因
    assert hit.final_score > 1.0
    assert should_refuse([hit], min_score=DEFAULT_MIN_SCORE, min_hits=1,
                         query="二甲双胍肾功能不好还能用吗") is True

    good = make_hit("估算肾小球滤过率低于30的患者禁用二甲双胍，肾功能不全者应减量。",
                    TIER_GUIDELINE, score=0.0163, chunk_id="ok")
    apply_authority_boost([good])
    assert should_refuse([good], min_score=DEFAULT_MIN_SCORE, min_hits=1,
                         query="二甲双胍肾功能不好还能用吗") is False


def test_dense_index_load_without_chunks_is_refused(tmp_path):
    """不传 chunks 的 load 不能返回 True 却留下一个空索引。"""
    from medical_rag.index import DenseIndex

    chunks = [make_chunk("d-%d" % i, "第%d条内容，二甲双胍与糖尿病。" % i) for i in range(4)]
    index = DenseIndex()
    index.build(chunks)
    index.save(str(tmp_path))

    probe = DenseIndex()
    assert probe.load(str(tmp_path)) is False, "拿不到语料就不能声称加载成功"
    assert probe.search("糖尿病", topk=3) == []

    # 传了 chunks 才算数
    ok = DenseIndex()
    assert ok.load(str(tmp_path), chunks=chunks) is True
    assert ok.size == len(chunks)
    assert ok.search("糖尿病", topk=3)


def test_from_chunks_does_not_load_reranker_weights(monkeypatch):
    """from_chunks 必须是惰性的：构造期不能把交叉编码器的权重拉起来。"""
    import types

    from medical_rag.pipeline import _demo_chunks

    calls = []
    fake = types.ModuleType("sentence_transformers")

    class _CrossEncoder(object):
        def __init__(self, *args, **kwargs):
            calls.append(args)

        def predict(self, *args, **kwargs):
            return [0.0]

    class _SentenceTransformer(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("测试环境不加载稠密模型")

    fake.CrossEncoder = _CrossEncoder
    fake.SentenceTransformer = _SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    MedicalRAG.from_chunks(_demo_chunks())
    assert calls == [], "构造期不应实例化 CrossEncoder"


# --------------------------------------------------------------------------
# 术语表：上位词 / 反义词不得被归一成同一个概念
# --------------------------------------------------------------------------
# 逐组比对"别名是不是自身规范名的子串"会产出 39 条合法命中（肥胖/肥胖症、
# 卒中/脑卒中、肌酐/血肌酐……），当硬断言只会天天误报。真正要守住的是下面这批
# "归一过去就是临床错误"的配对，所以直接把它们写成显式清单。
MUST_STAY_DISTINCT = [
    ("酒精性脂肪肝", "非酒精性脂肪性肝病"),   # 病因、干预、预后完全不同
    ("NT-proBNP", "B型利钠肽"),             # 诊断切点差一个数量级
    ("腹膜透析", "血液透析"),
    ("睡眠障碍", "失眠症"),                  # 上位类别含 OSA / 不宁腿
    ("胃溃疡", "十二指肠溃疡"),              # 随访策略不同
    ("STEMI", "心肌梗死"),                   # 再灌注路径不同
    ("口眼歪斜", "面神经麻痹"),              # 也是卒中的典型表述
    ("功能性肠病", "肠易激综合征"),
    ("老年痴呆", "阿尔茨海默病"),            # 会吞掉血管性痴呆、路易体痴呆
    ("透析", "血液透析"),
    ("脂肪肝", "非酒精性脂肪性肝病"),
]


def test_terminology_keeps_clinically_distinct_concepts_apart():
    term = Terminology.default()
    for surface, must_not_become in MUST_STAY_DISTINCT:
        assert term.normalize(surface) != must_not_become, (surface, must_not_become)
        assert must_not_become not in term.terms_in(surface), (surface, must_not_become)


def test_terminology_still_normalizes_true_synonyms():
    """收紧词表不能把真正的同义关系也误删。"""
    term = Terminology.default()
    assert term.normalize("非酒精性脂肪肝") == "非酒精性脂肪性肝病"
    assert term.normalize("血透") == "血液透析"
    assert term.normalize("急性心梗") == "心肌梗死"
    assert term.normalize("BNP") == "B型利钠肽"


def test_terminology_skips_hypernym_hit_behind_negation_prefix():
    """命中区间左边紧邻"非"时跳过：否则"非 A"会被标成"A"。"""
    term = Terminology([["小细胞肺癌", "SCLC"]])
    assert term.terms_in("非小细胞肺癌的一线治疗") == []
    assert term.terms_in("小细胞肺癌的一线治疗") == ["小细胞肺癌"]


# --------------------------------------------------------------------------
# PDF：表格不得同时以原子块和拍扁正文两种形态入库
# --------------------------------------------------------------------------
def test_pdf_table_cells_do_not_also_appear_as_text_blocks():
    from medical_rag.parsers import _drop_table_echo, _make_block

    table = _make_block(
        "| 药物 | 起始剂量 | 最大日剂量 |\n| --- | --- | --- |\n"
        "| 二甲双胍 | 0.5g 每日2次 | 2.0g |",
        page=3,
        content_type=CONTENT_TABLE,
    )
    text_blocks = [
        _make_block("药物", page=3),
        _make_block("二甲双胍", page=3),
        _make_block("0.5g 每日2次", page=3),
        _make_block("本节讨论二甲双胍在肾功能不全患者中的剂量调整原则与监测要求。", page=3),
    ]
    kept = _drop_table_echo(text_blocks, [table])
    assert len(kept) == 1, "表格单元格的拍扁副本必须被丢掉"
    assert "剂量调整原则" in kept[0]["text"], "正文不能被误删"


def test_evidence_class_and_grade_coexist_instead_of_cancelling():
    """"I 类推荐（证据级别：A）" 是中文指南最标准的写法。

    推荐类别与证据级别是两个正交维度，唯一性约束若跨维度统计，会把这种标准
    写法判成"标记不唯一"而两个标记一起丢弃，导致最该被保留的强推荐反而没有等级。
    """
    from medical_rag.chunker import _resolve_evidence_level

    text = "无禁忌者均应尽早给予阿司匹林。I 类推荐（证据级别：A）。"
    assert _resolve_evidence_level(text, "", []) == "I（证据级别A）"
    # 同一维度内真出现歧义时，仍然宁缺毋错
    assert _resolve_evidence_level("既有I类推荐也有III类推荐", "", []) == ""


def test_detect_evidence_level_does_not_let_class_one_shadow_class_three():
    """re.search 会在 "III类推荐" 内部匹配到 "I类推荐"，顺序错了会反转临床结论。"""
    from medical_rag.schema import detect_evidence_level

    assert detect_evidence_level("（III类推荐）") == "III"
    assert detect_evidence_level("（II类推荐）") == "II"
    assert detect_evidence_level("（IIa类推荐）") == "IIa"
    assert detect_evidence_level("（I类推荐）") == "I"
    assert detect_evidence_level("A级证据") == "level:A"
    assert detect_evidence_level("（证据级别：B）") == "level:B"
