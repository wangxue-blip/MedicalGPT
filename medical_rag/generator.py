# -*- coding: utf-8 -*-
"""生成层：带强制引用的 RAG prompt 组装 + HuggingFace 生成后端。

为什么生成层要单独做"强制引用"？
-------------------------------------------------
检索层辛苦保住的 provenance（哪份文档、哪一节、第几页、哪一版），只要生成层
把证据揉成一段没有编号的自然语言，就全部丢失了。医疗场景里这比不做 RAG 更
危险：模型的幻觉被一段真实证据"包装"过，读者反而更容易相信。

所以这里的核心约定只有一条——**编号是全链路唯一的锚点**：

    build_context()  把第 i 条证据渲染成 "[i] 正文 / 来源: ..."
    build_prompt()   要求模型每个结论后必须标注 [i]
    format_sources() 用同一份 hits、同一个 i 渲染答案下方的来源清单
    verifier         按答案里的 [i] 反查 hits[i-1] 做引用核查

三个函数都以"调用方传进来的 hits 列表顺序"为唯一编号依据，因此**绝不能**在
其中任何一个里面过滤、去重或重排 hits，否则 [3] 在 prompt 里和在来源清单里
会指向两条不同的证据——这是引用型系统最隐蔽也最致命的一类 bug。
``build_context`` 因此宁可截断单条证据的正文，也不丢弃任何一条证据。

医疗安全口径
-------------------------------------------------
prompt 里的安全措辞与 ``training/grpo_training.py`` 的 MEDICAL_SYSTEM_PROMPT
保持一致（不替代医生诊断、不做绝对化判断、急症/孕妇/儿童/老人/症状加重提示
及时就医），这样 SFT/GRPO 训出来的行为与 RAG 推理时的要求不会互相打架。

降级行为
-------------------------------------------------
本模块的 prompt 组装部分是纯字符串处理，不依赖任何第三方库，在裸环境下永远
可用。只有 HFGenerator 需要 transformers/torch：缺依赖或模型加载失败时
``available`` 为 False，``generate()`` 抛 RuntimeError（调用方应先查 available，
再决定走"只检索、不生成"的降级出口）。peft 缺失只影响 peft_path 分支。
"""

import argparse
import logging
from typing import Any, List, Optional

from medical_rag.schema import Chunk, RetrievalHit, normalize_text

logger = logging.getLogger(__name__)

# 证据不足时的标准答复。做成常量是为了让 pipeline / verifier / 评测脚本判定
# "模型是否正确拒答"时比对的是同一份字符串，而不是各写各的近义句。
REFUSAL_TEXT = "根据已有资料无法确定，建议就医"

NO_EVIDENCE_TEXT = "（未检索到相关证据）"

# 单条证据的正文上限。父块（parent_text）默认 1200 字，直接全量拼进 prompt，
# topk=5 就逼近 6000 字，小模型的上下文窗口会被证据挤占到放不下问题本身。
MAX_EVIDENCE_CHARS = 800
# 全部证据合计的软上限，按条数均摊后再与上面的硬上限取小。
MAX_CONTEXT_CHARS = 4800
# 均摊后的下限：再挤也要留够一句话，否则截断后的证据既占 token 又不可读。
MIN_EVIDENCE_CHARS = 200

TRUNCATE_MARK = "……（本条证据已截断）"

# 与 training/grpo_training.py 的 MEDICAL_SYSTEM_PROMPT 同口径，另加 RAG 特有的
# 引用纪律。两者措辞保持一致，避免"训练时教它别给剂量、推理时又逼它给"。
MEDICAL_RAG_SYSTEM_PROMPT = (
    "你是一个医学科普与健康咨询助手，现在基于检索到的资料回答问题。"
    "请给出结构化、谨慎的中文回答，包含可能原因或分析、处理建议、风险提示和就医建议。"
    "不要替代医生诊断，不要给出绝对化的结论，不要自行给出具体处方剂量，"
    "涉及急症、孕妇、儿童、老人或症状加重时应提示及时就医。"
)

RAG_PROMPT_TEMPLATE = """{system}

【回答规则】
1. 每一条结论后面必须标注其依据的证据编号，格式为 [n]；同时依据多条证据时写作 [1][3]。
2. 只依据下面给出的证据作答，不要引入证据之外的知识，也不要脑补证据中没有的数字、适应证或结论。
3. 证据不足、或证据与问题无关时，直接回答"{refusal}"，不要勉强作答。
4. 只能使用下面实际出现过的编号，不要编造不存在的来源编号。
5. 涉及具体用法用量时，必须以引用证据原文的形式给出，并注明"以说明书和医嘱为准"；证据里没有写剂量就不要写。
6. 用中文作答，先给结论再给依据，不要复述本规则。

【证据】
{context}

【问题】
{query}

【回答】
"""


# --------------------------------------------------------------------------
# prompt 组装
# --------------------------------------------------------------------------
def _evidence_budget(count: int) -> int:
    """按证据条数均摊单条正文预算。

    均摊而不是"前几条给足、后面直接砍掉"，是因为编号必须稳定：被砍掉的那条
    在来源清单里依然占着编号，prompt 里却是空的，模型就会把 [n] 用错。
    """
    if count <= 0:
        return MAX_EVIDENCE_CHARS
    share = MAX_CONTEXT_CHARS // count
    return max(MIN_EVIDENCE_CHARS, min(MAX_EVIDENCE_CHARS, share))


def _truncate(text: str, limit: int) -> str:
    """超长截断并显式标注，让模型知道"这条被截了"而不是"这条就这么短"。"""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + TRUNCATE_MARK


def build_context(hits: List[RetrievalHit]) -> str:
    """把命中列表渲染成带编号的证据块。

    编号 n 从 1 开始，严格等于 hits 的下标 +1，与 format_sources 一一对应。
    正文取 chunk.context_text()（有父块用父块），这是 small-to-big 的后半段：
    检索用子块保证精度，生成用父块保证可读性——只给子块，模型经常会把半句话
    的限定条件（"对于 eGFR<30 的患者"）丢在上文里读不到。
    """
    if not hits:
        return NO_EVIDENCE_TEXT

    budget = _evidence_budget(len(hits))
    blocks: List[str] = []
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        body = normalize_text(chunk.context_text())
        if not body:
            # 正文为空也要占住编号，否则后面的证据编号会整体前移。
            body = NO_EVIDENCE_TEXT
        blocks.append(
            "[%d] %s\n    来源: %s" % (index, _truncate(body, budget), chunk.citation_label())
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, hits: List[RetrievalHit]) -> str:
    """组装最终送进模型的 RAG prompt（含系统口径与引用纪律）。"""
    return RAG_PROMPT_TEMPLATE.format(
        system=MEDICAL_RAG_SYSTEM_PROMPT,
        refusal=REFUSAL_TEXT,
        context=build_context(hits),
        query=normalize_text(query) or "（空问题）",
    )


def format_sources(hits: List[RetrievalHit]) -> str:
    """渲染答案下方的来源清单，每行 "[n] <citation_label()>"。

    这里直接复用 Chunk.citation_label()，不自己拼字段：来源格式只有一处定义，
    将来加"发布机构"之类的字段时，prompt 侧和展示侧不会漂移。
    """
    if not hits:
        return ""
    return "\n".join(
        "[%d] %s" % (index, hit.chunk.citation_label())
        for index, hit in enumerate(hits, start=1)
    )


def format_answer_with_sources(answer: str, hits: List[RetrievalHit]) -> str:
    """答案 + 来源清单的标准展示形态（给 demo / API 直接输出用）。"""
    sources = format_sources(hits)
    answer = (answer or "").strip()
    if not sources:
        return answer
    return "%s\n\n【来源】\n%s" % (answer, sources)


# --------------------------------------------------------------------------
# HuggingFace 生成后端
# --------------------------------------------------------------------------
class HFGenerator:
    """本地 HF 模型生成器，支持加载本项目 SFT/GRPO 产出的 LoRA adapter。

    与 eval/medical_project/model_utils.py 的加载方式保持一致（同样的 pad_token
    兜底、同样的 peft 合并方式），这样 RAG 推理与离线评测跑的是同一套权重行为。
    """

    def __init__(
        self,
        model_name_or_path: str,
        peft_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        self.model_name_or_path = model_name_or_path
        self.peft_path = peft_path or None
        self.device_map = device_map or "auto"
        self.torch_dtype = torch_dtype or "auto"
        self._tokenizer = None
        self._model = None
        self._load_attempted = False
        self._load_error = ""

    # -- 后端加载 ----------------------------------------------------------
    def _resolve_dtype(self, torch_module: Any) -> Any:
        """把 "bfloat16" 这类字符串映射成 torch dtype；"auto" 原样透传。"""
        if self.torch_dtype in (None, "auto"):
            return "auto"
        if isinstance(self.torch_dtype, str):
            dtype = getattr(torch_module, self.torch_dtype, None)
            if dtype is None:
                logger.warning("未知的 torch_dtype=%r，已回退为 auto", self.torch_dtype)
                return "auto"
            return dtype
        return self.torch_dtype

    def _ensure_model(self) -> None:
        """惰性加载，且只尝试一次。

        构造 pipeline 时就加载模型会让"只想看看检索结果"也付出几 GB 的加载代价，
        所以推迟到真正生成（或调用方询问 available）时再做；失败后置位
        _load_attempted，避免每条 query 重试一次加载。
        """
        if self._load_attempted:
            return
        self._load_attempted = True

        if not self.model_name_or_path:
            self._load_error = "未指定 model_name_or_path"
            logger.warning("生成后端未配置模型路径，生成功能不可用")
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # ImportError 及其它环境异常都算不可用
            self._load_error = "transformers/torch 不可用: %s" % exc
            logger.warning("transformers 或 torch 不可用，生成功能已关闭: %s", exc)
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            # 大多数中文 base 模型没有 pad_token，不补会在 generate 时报错或
            # 产出错位的 attention_mask。
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self._resolve_dtype(torch),
                device_map=self.device_map,
                trust_remote_code=True,
            )
            if self.peft_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.peft_path)
                logger.info("已加载 LoRA adapter: %s", self.peft_path)
            model.eval()
        except Exception as exc:
            # 模型不存在、无网络、显存不足、peft 版本不匹配等都归入降级，
            # 不能让整条 RAG 链路（检索部分明明是好的）直接崩掉。
            self._load_error = "模型加载失败: %s" % exc
            logger.warning("生成模型 %s 加载失败，已降级: %s", self.model_name_or_path, exc)
            self._tokenizer = None
            self._model = None
            return

        self._tokenizer = tokenizer
        self._model = model
        logger.info("生成模型加载成功: %s", self.model_name_or_path)

    @property
    def available(self) -> bool:
        """当前是否真的能生成（会触发一次惰性加载）。"""
        self._ensure_model()
        return self._model is not None and self._tokenizer is not None

    # -- 生成 --------------------------------------------------------------
    def _apply_chat_template(self, prompt: str) -> str:
        """把 RAG prompt 作为一条 user 消息套进模型自己的对话模板。

        不再额外塞 system 消息：build_prompt() 已经把医疗安全口径写进正文了，
        再加一条 system 会出现两份措辞略有差异的规则，指令冲突时小模型往往两条
        都不遵守。没有 chat_template 的 base 模型则退化为朴素问答格式。
        """
        tokenizer = self._tokenizer
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                logger.warning("apply_chat_template 失败，改用朴素模板: %s", exc)
        return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        """生成回答文本。后端不可用时抛 RuntimeError（调用方需先查 available）。

        这里刻意**不**静默返回空串：生成失败和"模型认为无法回答"是两件事，
        混成同一种结果会让评测把环境问题统计成模型的拒答率。
        """
        if not self.available:
            raise RuntimeError(
                "生成后端不可用（%s）。请先检查 HFGenerator.available 再调用 generate()。"
                % (self._load_error or "未知原因")
            )

        import torch

        text = self._apply_chat_template(prompt)
        encoded = self._tokenizer(text, return_tensors="pt")
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}

        # temperature<=0 视为确定性解码：此时必须关掉 do_sample，否则
        # transformers 会对 temperature=0 报错或给出退化分布。
        do_sample = float(temperature) > 0
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
                top_p=float(top_p) if do_sample else None,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = generated[0][encoded["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------
# 自检 demo
# --------------------------------------------------------------------------
def _demo_hits() -> List[RetrievalHit]:
    """构造两条来源等级不同的命中，用来检查编号与来源清单是否对齐。"""
    from medical_rag.schema import TIER_AUTHORITY, TIER_DRUG_LABEL, TIER_GUIDELINE

    raw = [
        (
            "g1",
            "二甲双胍是 2 型糖尿病患者控制高血糖的一线用药（I 类推荐）。",
            "二甲双胍为 2 型糖尿病药物治疗的首选和全程用药，若无禁忌证应一直保留在治疗方案中。"
            "eGFR<45 ml/min/1.73m2 时需减量，eGFR<30 时禁用。",
            "中国2型糖尿病防治指南",
            TIER_GUIDELINE,
            "2020年版",
            ["第四章 高血糖的药物治疗", "口服降糖药"],
            76,
            "I",
        ),
        (
            "d1",
            "成人起始剂量为一次 0.5g，一日 2 次，随餐服用。",
            "盐酸二甲双胍片：成人开始一次 0.5g，一日 2 次，随餐服用；可根据血糖逐渐加量，"
            "一日最大剂量不超过 2g。",
            "盐酸二甲双胍片说明书",
            TIER_DRUG_LABEL,
            "2019修订",
            ["【用法用量】"],
            2,
            "",
        ),
    ]
    hits: List[RetrievalHit] = []
    for cid, text, parent, title, tier, version, section, page, level in raw:
        chunk = Chunk(
            chunk_id=cid,
            text=text,
            parent_text=parent,
            doc_id=title,
            doc_title=title,
            source_tier=tier,
            version=version,
            section_path=section,
            page=page,
            evidence_level=level,
            authority=TIER_AUTHORITY[tier],
        )
        hits.append(RetrievalHit(chunk=chunk, score=0.03, final_score=0.9))
    return hits


def parse_args():
    parser = argparse.ArgumentParser(description="RAG prompt 组装与生成后端自检。")
    parser.add_argument("--query", default="二甲双胍怎么吃，有什么禁忌", help="自检用问题")
    parser.add_argument("--model_name_or_path", default="", help="生成模型路径，留空则只打印 prompt")
    parser.add_argument("--peft_path", default="", help="LoRA adapter 路径（SFT/GRPO 产物）")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    hits = _demo_hits()
    prompt = build_prompt(args.query, hits)
    print(prompt)
    print("【来源】")
    print(format_sources(hits))

    if not args.model_name_or_path:
        return
    generator = HFGenerator(
        args.model_name_or_path,
        peft_path=args.peft_path or None,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    if not generator.available:
        print("生成后端不可用，已跳过生成。")
        return
    answer = generator.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(format_answer_with_sources(answer, hits))


if __name__ == "__main__":
    main()
