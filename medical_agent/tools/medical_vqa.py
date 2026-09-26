# -*- coding: utf-8 -*-
"""medical_vqa：Qwen2.5-VL 视觉问答，强制结构化 JSON 输出。

第一阶段直接用预训练权重不微调。这一步要回答的问题是"它在医疗图上稳不稳"：
能不能看懂胸片、能不能按 schema 吐 JSON、幻觉多不多。``parse_ok`` 字段就是
Phase 3 统计"结构化输出成功率"的依据，解析失败不抛异常，原文保留在 ``raw``。

VLM 的 findings 是"它认为看到了什么"，**不是** grounded 的检测结果。aggregate 与
safety 把它和 detect 的输出分开对待：VLM 提到而检测器没检出的病理会被 safety
标为无依据，这是刻意的——VLM 幻觉正是要抓的东西。
"""

import json
import os
import re
from typing import Any, Dict, List

from medical_agent import config
from medical_agent.schema import TOOL_VQA, XRV_PATHOLOGIES, VQAAnswer
from medical_agent.tools.base import BaseTool

VQA_PROMPT = (
    "你是医学影像辅助分析助手。请观察这张胸部 X 光片并回答问题。\n"
    "只输出一个 JSON 对象，不要输出任何其他文字，格式：\n"
    '{"answer": "对问题的直接回答", "confidence": 0到1之间的小数, '
    '"findings": ["观察到的异常，用英文病理名"], "location": "位置描述（用患者左右）", '
    '"explanation": "依据的影像特征"}\n'
    "候选病理名限于：%s。\n"
    "看不清或不确定时 confidence 给低值并在 answer 里说明；不要编造图上没有的发现。\n"
) % ", ".join(XRV_PATHOLOGIES)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_vqa_json(text: str) -> VQAAnswer:
    """把 VLM 输出解析成 VQAAnswer。容忍 ``` 围栏与前后杂文；失败时 parse_ok=False。"""
    raw = text or ""
    candidate = raw.strip()
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return VQAAnswer(answer=raw.strip()[:200], raw=raw, parse_ok=False)
    try:
        data = json.loads(candidate[start:end + 1])
    except (ValueError, TypeError):
        return VQAAnswer(answer=raw.strip()[:200], raw=raw, parse_ok=False)
    if not isinstance(data, dict):
        return VQAAnswer(answer=raw.strip()[:200], raw=raw, parse_ok=False)

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    from medical_agent.safety import mentioned_labels

    findings: List[str] = []
    raw_findings = data.get("findings") or []
    if isinstance(raw_findings, str):
        raw_findings = [raw_findings]
    for item in raw_findings:
        name = str(item).strip()
        if not name:
            continue
        mapped = mentioned_labels(name)
        for label in (mapped or [name]):
            if label not in findings:
                findings.append(label)

    return VQAAnswer(
        answer=str(data.get("answer", "") or "").strip(),
        confidence=confidence,
        findings=findings,
        location=str(data.get("location", "") or "").strip(),
        explanation=str(data.get("explanation", "") or "").strip(),
        raw=raw,
        parse_ok=True,
    )


class MedicalVQATool(BaseTool):
    name = TOOL_VQA
    description = "Qwen2.5-VL 医学视觉问答（预训练权重，强制 JSON 输出）"

    def __init__(
        self,
        mock: bool = False,
        model_path: str = config.VLM_MODEL,
        device: str = "cpu",
        max_new_tokens: int = config.VQA_MAX_NEW_TOKENS,
        max_pixels: int = config.VQA_MAX_PIXELS,
    ):
        super(MedicalVQATool, self).__init__(mock=mock, device=device)
        self.model_path = model_path
        self.max_new_tokens = int(max_new_tokens or 256)
        self.max_pixels = int(max_pixels or 640 * 640)
        self.model = None
        self.processor = None

    # -- 真实后端 ------------------------------------------------------------
    def _probe(self) -> str:
        if not self.model_path or not os.path.isdir(self.model_path):
            raise FileNotFoundError("VLM 模型目录不存在: %s" % self.model_path)
        import torch  # noqa: F401
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        device_map = None if self.device == "cpu" else self.device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map=device_map,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, max_pixels=self.max_pixels)
        return "qwen2.5-vl:%s" % self.model_path

    def _run_real(self, image_path: str, question: str, context: str = "") -> Dict[str, Any]:
        from PIL import Image

        prompt = VQA_PROMPT + "\n问题：%s" % (question or "这张胸片有无异常？")
        if context:
            prompt += "\n参考资料（仅供参考，不要复述）：\n%s" % context[:1500]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return {"vqa": parse_vqa_json(raw).to_dict()}

    # -- mock ------------------------------------------------------------------
    def _run_mock(self, image_path: str, question: str, context: str = "") -> Dict[str, Any]:
        payload = {
            "answer": "左下肺野可见斑片状阴影",
            "confidence": 0.7,
            "findings": ["Lung Opacity"],
            "location": "左下肺野",
            "explanation": "局部密度增高影，边界欠清",
        }
        raw = json.dumps(payload, ensure_ascii=False)
        return {"vqa": parse_vqa_json(raw).to_dict()}
