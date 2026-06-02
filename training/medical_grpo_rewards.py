# -*- coding: utf-8 -*-
"""Medical reward functions for GRPO alignment experiments.

These rewards are designed for open-ended medical QA alignment. The reference
similarity reward is only a semantic/reference-overlap signal; it is not a
medical accuracy metric.
"""

import argparse
import math
import re
from collections import Counter
from typing import Iterable, List, Sequence


ANALYSIS_KEYWORDS = ("诊断依据", "可能原因", "初步分析", "病情分析", "分析", "鉴别诊断")
SUGGESTION_KEYWORDS = ("建议", "处理建议", "护理建议", "康复建议", "生活方式", "观察")
RISK_KEYWORDS = ("风险提示", "警示", "注意", "危险信号", "加重", "红旗信号")
VISIT_KEYWORDS = ("就医建议", "及时就医", "尽快就医", "医院", "医生", "门诊", "急诊")

ABSOLUTE_DIAGNOSIS_PATTERNS = (
    r"一定是",
    r"肯定是",
    r"绝对是",
    r"必然是",
    r"确诊为",
    r"无需检查",
)
DOSE_PATTERNS = (
    r"(每日|每天|一天)\s*\d+\s*(次|回)",
    r"每次\s*\d+(\.\d+)?\s*(片|粒|颗|袋|支|mg|毫克|ml|毫升|g|克)",
    r"\d+(\.\d+)?\s*(mg|毫克|ml|毫升|g|克)\s*/\s*(kg|公斤|天|日|次)",
)
DOCTOR_REPLACEMENT_PATTERNS = (
    r"不用去医院",
    r"无需就医",
    r"不需要看医生",
    r"自己吃药即可",
    r"在家硬扛",
)
HIGH_RISK_ACTION_PATTERNS = (
    r"立即停药",
    r"自行停药",
    r"自行加药",
    r"随意加量",
    r"孕妇.*(随便|可以直接|放心).*用药",
    r"儿童.*(随便|可以直接|放心).*用药",
)
SAFETY_POSITIVE_KEYWORDS = (
    "不能替代医生",
    "需结合检查",
    "建议就医",
    "及时就医",
    "遵医嘱",
    "在医生指导下",
    "如症状加重",
)


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "") or ""
        if isinstance(first, str):
            return first
    if isinstance(completion, dict):
        return completion.get("content", "") or ""
    return ""


def _texts(completions: Iterable) -> List[str]:
    return [_completion_text(completion) for completion in completions]


def _as_sample_list(value, length: int):
    if value is None:
        return [None] * length
    if isinstance(value, list) and len(value) == length:
        return value
    return [value] * length


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def medical_format_reward(completions, **kwargs):
    """
    Reward answers containing medical structure:
    - diagnosis basis / possible reasons / analysis
    - advice / handling suggestions
    - risk warning
    - doctor visit advice
    """
    texts = _texts(completions)
    required_sections = _as_sample_list(kwargs.get("required_sections"), len(texts))
    rewards = []
    default_groups = [
        ANALYSIS_KEYWORDS,
        SUGGESTION_KEYWORDS,
        RISK_KEYWORDS,
        VISIT_KEYWORDS,
    ]

    for text, sample_required_sections in zip(texts, required_sections):
        groups = default_groups
        if isinstance(sample_required_sections, (list, tuple)) and sample_required_sections:
            custom_groups = []
            for section in sample_required_sections:
                section_text = str(section)
                if "诊断" in section_text or "分析" in section_text or "依据" in section_text:
                    custom_groups.append(ANALYSIS_KEYWORDS)
                elif "建议" in section_text and "就医" not in section_text:
                    custom_groups.append(SUGGESTION_KEYWORDS)
                elif "风险" in section_text or "警示" in section_text or "注意" in section_text:
                    custom_groups.append(RISK_KEYWORDS)
                elif "就医" in section_text or "医生" in section_text or "医院" in section_text:
                    custom_groups.append(VISIT_KEYWORDS)
            if custom_groups:
                groups = custom_groups

        hits = sum(1 for group in groups if _contains_any(text, group))
        rewards.append(hits / max(1, len(groups)))
    return rewards


def _normalize_for_similarity(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def _char_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> Counter:
    normalized = _normalize_for_similarity(text)
    counter = Counter()
    words = re.findall(r"[a-z0-9]+", normalized)
    counter.update(words)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    cjk_text = "".join(cjk_chars)
    for n in range(min_n, max_n + 1):
        if len(cjk_text) >= n:
            counter.update(cjk_text[i:i + n] for i in range(len(cjk_text) - n + 1))
    if not counter and normalized:
        counter.update(normalized)
    return counter


def _weighted_dice(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    overlap = sum(min(a[key], b[key]) for key in (a.keys() & b.keys()))
    total = sum(a.values()) + sum(b.values())
    return 2.0 * overlap / total if total else 0.0


def _cosine(vec_a, vec_b) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in vec_a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, (dot / (norm_a * norm_b) + 1.0) / 2.0))


def _embedding_similarity(texts: List[str], refs: List[str], embedding_model) -> List[float]:
    if embedding_model is None:
        raise ValueError("embedding_model is None")

    if hasattr(embedding_model, "encode"):
        text_vecs = embedding_model.encode(texts, normalize_embeddings=True)
        ref_vecs = embedding_model.encode(refs, normalize_embeddings=True)
    elif callable(embedding_model):
        text_vecs = embedding_model(texts)
        ref_vecs = embedding_model(refs)
    else:
        raise TypeError("embedding_model must provide encode() or be callable")

    return [_cosine(text_vec, ref_vec) for text_vec, ref_vec in zip(text_vecs, ref_vecs)]


def reference_similarity_reward(completions, answer, embedding_model=None, **kwargs):
    """
    Score closeness to reference answers.

    When an embedding model object is provided, use cosine similarity. Otherwise
    use a deterministic Chinese char n-gram fallback so tests and smoke checks do
    not load BGE-M3. This is reference similarity, not medical correctness.
    """
    texts = _texts(completions)
    refs = [str(item or "") for item in _as_sample_list(answer, len(texts))]

    if embedding_model is not None:
        try:
            return _embedding_similarity(texts, refs, embedding_model)
        except Exception:
            pass

    rewards = []
    for text, ref in zip(texts, refs):
        rewards.append(_weighted_dice(_char_ngrams(text), _char_ngrams(ref)))
    return rewards


def _pattern_hits(text: str, patterns: Sequence[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


def medical_safety_reward(completions, **kwargs):
    """
    Reward safe medical wording and penalize:
    - absolute diagnosis
    - concrete prescription dosage
    - replacing doctors / avoiding care
    - unsafe medication changes for high-risk groups
    """
    texts = _texts(completions)
    rewards = []
    for text in texts:
        penalty = 0.0
        penalty += 0.25 * _pattern_hits(text, ABSOLUTE_DIAGNOSIS_PATTERNS)
        penalty += 0.35 * _pattern_hits(text, DOSE_PATTERNS)
        penalty += 0.35 * _pattern_hits(text, DOCTOR_REPLACEMENT_PATTERNS)
        penalty += 0.35 * _pattern_hits(text, HIGH_RISK_ACTION_PATTERNS)

        positive_hits = sum(1 for keyword in SAFETY_POSITIVE_KEYWORDS if keyword in text)
        bonus = min(0.15, positive_hits * 0.04)
        rewards.append(max(0.0, min(1.0, 0.85 + bonus - penalty)))
    return rewards


def _sentence_repetition_ratio(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[。！？!?；;\n]+", text) if len(s.strip()) >= 6]
    if len(sentences) < 3:
        return 0.0
    counts = Counter(sentences)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(sentences)


def _ngram_repetition_ratio(text: str, n: int = 12) -> float:
    normalized = _normalize_for_similarity(text)
    if len(normalized) < n * 3:
        return 0.0
    grams = [normalized[i:i + n] for i in range(len(normalized) - n + 1)]
    unique = len(set(grams))
    return max(0.0, 1.0 - unique / len(grams))


def length_repetition_penalty(completions, **kwargs):
    """Return a penalty in [0, 1] for too-short, too-long, or repetitive answers."""
    texts = _texts(completions)
    penalties = []
    min_len = int(kwargs.get("min_response_chars", 80))
    max_len = int(kwargs.get("max_response_chars", 1200))
    hard_max_len = int(kwargs.get("hard_max_response_chars", 2200))

    for text in texts:
        text_len = len(text)
        short_penalty = 0.0
        if text_len < min_len:
            short_penalty = min(0.3, (min_len - text_len) / max(1, min_len) * 0.3)

        long_penalty = 0.0
        if text_len > max_len:
            long_penalty = min(0.5, (text_len - max_len) / max(1, hard_max_len - max_len) * 0.5)

        repetition_penalty = min(0.6, max(_sentence_repetition_ratio(text), _ngram_repetition_ratio(text)) * 1.2)
        penalties.append(min(1.0, short_penalty + long_penalty + repetition_penalty))
    return penalties


def combined_medical_reward(completions, answer, **kwargs):
    """
    Composite reward:
    R = 0.35 * format + 0.30 * similarity + 0.25 * safety - 0.10 * repetition_penalty
    """
    fmt = medical_format_reward(completions, **kwargs)
    sim = reference_similarity_reward(completions, answer, **kwargs)
    safety = medical_safety_reward(completions, **kwargs)
    rep = length_repetition_penalty(completions, **kwargs)
    rewards = []
    for f, s, safe, penalty in zip(fmt, sim, safety, rep):
        rewards.append(max(0.0, min(1.0, 0.35 * f + 0.30 * s + 0.25 * safe - 0.10 * penalty)))
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Medical GRPO reward helper module.")
    parser.add_argument("--demo", action="store_true", help="Print reward values for a small demo.")
    args = parser.parse_args()
    if args.demo:
        completions = [[{"content": "分析：可能为感染。建议：休息观察。风险提示：高热或胸痛需警惕。就医建议：症状加重请及时就医。"}]]
        print(combined_medical_reward(completions, ["感染需要结合体温、症状和检查评估，症状加重应及时就医。"]))


if __name__ == "__main__":
    main()
