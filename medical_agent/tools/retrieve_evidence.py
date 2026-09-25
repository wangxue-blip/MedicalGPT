# -*- coding: utf-8 -*-
"""retrieve_evidence：把已完成的 medical_rag 子系统包成一个 Agent 工具。

不重复实现任何检索逻辑——三路召回、RRF、精排、拒答闸门全在 medical_rag 里。
本工具只做两件事：把问题送进去，把 ``RetrievalHit`` 转成契约里的 ``Evidence``。
``refused`` 直接透传 medical_rag 的 ``should_refuse`` 判定，aggregate 节点据此
决定是否在报告里声明"未检索到可靠依据"。
"""

import os
from typing import Any, Dict, Optional

from medical_agent import config
from medical_agent.schema import TOOL_RETRIEVE, Evidence
from medical_agent.tools.base import BaseTool


class RetrieveEvidenceTool(BaseTool):
    name = TOOL_RETRIEVE
    description = "从分层医学语料（指南/说明书/百科）检索带出处的证据片段"

    def __init__(
        self,
        mock: bool = False,
        chunks_path: str = config.RAG_CHUNKS,
        dense_model: str = config.RAG_DENSE_MODEL,
        topk: int = config.RAG_TOPK,
        device: str = "cpu",
    ):
        super(RetrieveEvidenceTool, self).__init__(mock=mock, device=device)
        self.chunks_path = chunks_path
        self.dense_model = dense_model or None
        self.topk = max(1, int(topk or 1))
        self._rag = None

    # -- 真实后端 ------------------------------------------------------------
    def _probe(self) -> str:
        if not self.chunks_path or not os.path.exists(self.chunks_path):
            raise FileNotFoundError("RAG 语料不存在: %s（先跑 tools/medical_project/rag/build_corpus.py）" % self.chunks_path)
        from medical_rag.pipeline import MedicalRAG

        self._rag = MedicalRAG.from_jsonl(self.chunks_path, dense_model=self.dense_model, device=self.device)
        return "medical_rag:%s" % self.chunks_path

    def _run_real(self, query: str, topk: Optional[int] = None) -> Dict[str, Any]:
        k = int(topk or self.topk)
        hits = self._rag.retriever.retrieve(query, topk=k) or []
        try:
            from medical_rag.verifier import should_refuse
            refused = bool(should_refuse(hits, query=query))
        except Exception:
            refused = len(hits) == 0
        evidence = [Evidence.from_hit_dict(hit.to_dict()).to_dict() for hit in hits]
        return {"evidence": evidence, "refused": refused, "n": len(evidence)}

    # -- mock ------------------------------------------------------------------
    def _run_mock(self, query: str, topk: Optional[int] = None) -> Dict[str, Any]:
        evidence = [
            Evidence(
                chunk_id="mock_c1",
                citation="胸部影像诊断指南（mock） | 第二章 肺部阴影 > 2.1 影像表现 | 2023年版",
                text="肺部阴影在胸片上表现为局部密度增高，需结合临床症状、化验与既往影像综合判断，必要时行 CT 进一步评估。",
                source_tier="L1_guideline",
                score=0.7,
            ).to_dict(),
            Evidence(
                chunk_id="mock_c2",
                citation="胸部影像诊断指南（mock） | 第三章 心影 > 3.1 心影增大 | 2023年版",
                text="心影增大指心胸比例超过正常范围，常见于心功能不全、心包积液等，应结合心脏超声评估。",
                source_tier="L1_guideline",
                score=0.6,
            ).to_dict(),
        ]
        k = int(topk or self.topk)
        evidence = evidence[:k]
        return {"evidence": evidence, "refused": False, "n": len(evidence)}
