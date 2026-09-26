# -*- coding: utf-8 -*-
"""reporter：把 StructuredReport 转成自然语言报告。

两个后端：
* 真实——项目已有的医疗文本后训练模型（Qwen2.5-7B + 你的 LoRA/GRPO adapter），
  prompt 里只放 StructuredReport 的内容，规则与 medical_rag/generator.py 同口径；
* 模板——确定性渲染。mock 模式或模型不可用时使用，**输出必须能通过
  safety.check_answer**（数字只用报告里的、病灶只用 findings 里的），这样裸环境
  下也能验证整条 grounding 链路。

胸片左右：影像是从前向后投照的，**图像左侧对应患者右侧**。bbox 中心 x < 0.5
描述为"右"（患者侧），这是放射科的表述习惯，与图像坐标相反。
"""

import os
from typing import List, Optional

from medical_agent import config
from medical_agent.schema import INTENT_TEXT_QA, PATHOLOGY_ZH, Finding, StructuredReport

try:
    from medical_rag.generator import REFUSAL_TEXT
except Exception:  # pragma: no cover
    REFUSAL_TEXT = "根据已有资料无法确定，建议就医"

REVIEW_NOTICE = "以上为算法辅助分析结果，不构成诊断，需由放射科医生复核。"

REPORT_RULES = (
    "【回答规则】\n"
    "1. 只依据下面给出的结构化结果作答，不要引入结果之外的病灶、数字、位置或来源。\n"
    "2. 引用证据时在句末标注编号 [n]，编号必须来自证据清单。\n"
    "3. 病灶名称只能使用检出发现里的名称；分数按百分比表述且必须与结果一致。\n"
    "4. 不给出具体用法用量，不做绝对化诊断（不要出现\"一定是\"\"确诊为\"）。\n"
    "5. 结果不足以回答时直接回答\"%s\"。\n"
    "6. 末尾附一句：%s\n"
) % (REFUSAL_TEXT, REVIEW_NOTICE)


def describe_location(finding: Finding) -> str:
    """由归一化 bbox 推位置描述，用患者左右。"""
    if finding.bbox is None:
        return ""
    cx = (finding.bbox.x1 + finding.bbox.x2) / 2.0
    cy = (finding.bbox.y1 + finding.bbox.y2) / 2.0
    side = "右" if cx < 0.5 else "左"          # 图像左 = 患者右
    if cy < 0.4:
        level = "上"
    elif cy < 0.7:
        level = "中"
    else:
        level = "下"
    if finding.label in ("Cardiomegaly", "Enlarged Cardiomediastinum"):
        return "心影区"
    return "%s%s肺野" % (side, level)


class Reporter(object):
    def __init__(
        self,
        model_path: str = config.TEXT_MODEL,
        peft_path: str = config.TEXT_PEFT,
        device: str = "cpu",
        mock: bool = False,
        max_new_tokens: int = config.REPORT_MAX_NEW_TOKENS,
    ):
        self.model_path = model_path
        self.peft_path = peft_path or ""
        self.device = device or "cpu"
        self.mock = bool(mock)
        self.max_new_tokens = int(max_new_tokens or 512)
        self._available: Optional[bool] = None
        self._backend = "template"
        self.unavailable_reason = ""
        self._model = None
        self._tokenizer = None

    # -- 后端 ----------------------------------------------------------------
    @property
    def available(self) -> bool:
        if self.mock:
            return False
        if self._available is None:
            try:
                self._load()
                self._available = True
                self._backend = "hf:%s" % self.model_path + ("+peft" if self.peft_path else "")
            except Exception as exc:
                self._available = False
                self._backend = "template"
                self.unavailable_reason = "%s: %s" % (type(exc).__name__, exc)
        return bool(self._available)

    @property
    def backend(self) -> str:
        if self.mock:
            return "template"
        _ = self.available
        return self._backend

    def _load(self) -> None:
        if not self.model_path or not os.path.isdir(self.model_path):
            raise FileNotFoundError("文本模型目录不存在: %s" % self.model_path)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto", device_map=None if self.device == "cpu" else self.device, trust_remote_code=True,
        )
        if self.peft_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.peft_path)
        model.eval()
        self._model = model

    # -- prompt --------------------------------------------------------------
    def build_prompt(self, report: StructuredReport, question: str, intent: str) -> str:
        lines = ["你是医学影像辅助分析助手，现在基于以下结构化分析结果撰写中文说明。", "", REPORT_RULES, "【检出发现】"]
        if report.findings:
            for finding in report.findings:
                loc = describe_location(finding)
                lines.append("- %s，模型分数 %d%%%s" % (
                    finding.label_zh, int(round(finding.score * 100)), ("，位置：%s" % loc) if loc else ""))
        else:
            lines.append("- 无")
        if report.vqa is not None and report.vqa.parse_ok:
            lines.append("")
            lines.append("【视觉问答】%s（置信度 %d%%）" % (report.vqa.answer, int(round(report.vqa.confidence * 100))))
        lines.append("")
        lines.append("【证据清单】")
        if report.evidence:
            for index, evidence in enumerate(report.evidence, start=1):
                lines.append("[%d] %s\n    来源: %s" % (index, evidence.text[:400], evidence.citation))
        else:
            lines.append("（无）")
        if report.caveats:
            lines.append("")
            lines.append("【注意事项】")
            for caveat in report.caveats:
                lines.append("- %s" % caveat)
        lines.append("")
        lines.append("【问题】%s" % (question or "请描述影像分析结果"))
        lines.append("")
        lines.append("【回答】")
        return "\n".join(lines)

    # -- 生成 ----------------------------------------------------------------
    def generate(self, report: StructuredReport, question: str, intent: str) -> str:
        if not report.findings and not report.evidence and report.vqa is None:
            return REFUSAL_TEXT + "。" + REVIEW_NOTICE
        if self.mock or not self.available:
            return self._render_template(report, question, intent)
        try:
            return self._generate_hf(report, question, intent)
        except Exception as exc:
            self.unavailable_reason = "生成失败: %s" % exc
            return self._render_template(report, question, intent)

    def _generate_hf(self, report: StructuredReport, question: str, intent: str) -> str:
        import torch

        prompt = self.build_prompt(report, question, intent)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self._tokenizer.pad_token_id)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if REVIEW_NOTICE not in answer:
            answer = answer + "\n" + REVIEW_NOTICE
        return answer

    def _render_template(self, report: StructuredReport, question: str, intent: str) -> str:
        parts: List[str] = []
        if intent == INTENT_TEXT_QA and report.evidence and not report.findings:
            parts.append("根据检索到的资料：")
        if report.findings:
            for finding in report.findings:
                loc = describe_location(finding)
                sentence = "%s检出%s，模型分数 %d%%" % (("%s" % loc) if loc else "影像", finding.label_zh, int(round(finding.score * 100)))
                if report.evidence:
                    sentence += "[1]"
                parts.append(sentence + "。")
        if report.evidence:
            for index, evidence in enumerate(report.evidence, start=1):
                snippet = evidence.text.strip()
                if len(snippet) > 120:
                    snippet = snippet[:120] + "……"
                parts.append("%s[%d]" % (snippet.rstrip("。"), index) + "。")
        if report.vqa is not None and report.vqa.parse_ok and report.vqa.answer:
            parts.append("视觉问答：%s（置信度 %d%%）。" % (report.vqa.answer, int(round(report.vqa.confidence * 100))))
        parts.append(REVIEW_NOTICE)
        return "\n".join(parts)
