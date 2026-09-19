# -*- coding: utf-8 -*-
"""引用校验：答案里的每一句话，到底有没有检索到的原文撑着。

为什么检索完、生成完还要再核验一遍？
-------------------------------------------------
RAG 常被当成"把原文塞进 prompt 就不会胡说"的银弹，但实际失败模式恰恰相反：
模型会把检索到的几段文字揉成一段读起来很顺、引用标记也齐全的答案，其中某一句
的关键结论（剂量、禁忌、阈值）却是它自己补的。带着 ``[2]`` 的错误结论比不带
引用的错误结论危险得多——引用让错误看起来可追溯。

所以本模块是链路上的最后一道闸：**逐句**把答案拆开，对每一句"它引了谁"和
"被引的原文是否真的支持它"做机器判定，给出 citation_precision / citation_recall
两个可量化的指标，而不是让"有引用"这件事本身成为可信度的证明。

判定用什么相似度？
-------------------------------------------------
``embedder`` 为 None（裸环境的默认路径）时走中文字符 2-4 gram 的加权 Dice，
实现直接复用 ``training/medical_grpo_rewards.py``（该文件不可导入时用本模块内置
的同款镜像），保证全项目"文本相似度"只有一套口径。

但有一个必须处理的细节：Dice 是对称系数，会被长度差稀释。一句 30 字的结论对
一段 320 字的 chunk，即使逐字抄原文，2*N/(N+10N) 也只有 0.18，阈值 0.62 永远
达不到——那样校验器只会输出"全都不支持"，毫无信息量。因此这里在证据文本上
**滑窗取最大值**：窗口宽度约为句长的 1.3 倍，逐段和句子算同一套加权 Dice，取
最高的那一段。语义上它回答的是"这句话能否在原文里找到一段高度重合的文字"，
既保住了和奖励函数一致的 n-gram 口径，又让阈值重新变得有意义。

``embedder`` 可用时改用余弦（不做 (cos+1)/2 的重标定——那会把毫不相关的两段
文字抬到 0.5，同样毁掉阈值）。embedder 只要提供 ``encode(list) -> 向量列表``
即可，稠密向量和 ``index._HashingEmbedder`` 的稀疏 dict 向量都支持；调用出错
自动退回 n-gram，不影响校验继续跑。

剂量豁免：为什么这里要为剂量开一个专门的口子
-------------------------------------------------
本项目里有两条规则天然打架：

* GRPO 的 ``medical_safety_reward`` 用 ``DOSE_PATTERNS`` 重罚"每日3次""每次
  0.5g"这类具体剂量表达——因为对齐阶段模型是在**凭记忆编**剂量，编错会直接
  伤人；
* 而 RAG 的价值恰恰在于能检索到说明书/指南里的**准确剂量**，此时把剂量说清楚
  是正确行为，一刀切地禁止反而逼着系统给出"请遵医嘱"式的无用答案。

两者的真正区别不在"说没说剂量"，而在"这个剂量是抄来的还是想出来的"。所以本
模块把命中 ``DOSE_PATTERNS`` 的句子分成两类：

* 有引用、且该句被引原文支持、且**被引 chunk 原文里出现了同样的剂量数字**
  -> 记入 ``exempted_doses``，视为合规（这正是 RAG 应该做的事）；
* 其余情况（没引用 / 引用不支持 / 原文里根本没有这个数字）
  -> 记入 ``dose_violations``，属最高风险项，调用方应据此拦截或改写。

于是安全约束从"禁止谈剂量"升级成"剂量必须可溯源"，准确性与安全性同时保住。

降级行为
-------------------------------------------------
本模块只依赖标准库，在裸环境（无 torch / transformers / numpy / jieba）下必定
可导入并完整可用；``training.medical_grpo_rewards`` 导入失败、``embedder`` 调用
失败都只打 warning 并降级，不抛异常。
"""

import argparse
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from medical_rag.schema import Chunk, RetrievalHit, normalize_text

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 与 training/medical_grpo_rewards.py 对齐的相似度口径
# --------------------------------------------------------------------------
# 本地镜像实现：仅在奖励模块不可导入（例如只拷了 medical_rag 目录出去用）时启用，
# 逻辑与奖励模块逐行一致，避免两处口径悄悄分叉。
_FALLBACK_DOSE_PATTERNS = (
    r"(每日|每天|一天)\s*\d+\s*(次|回)",
    r"每次\s*\d+(\.\d+)?\s*(片|粒|颗|袋|支|mg|毫克|ml|毫升|g|克)",
    r"\d+(\.\d+)?\s*(mg|毫克|ml|毫升|g|克)\s*/\s*(kg|公斤|天|日|次)",
)


def _fallback_normalize_for_similarity(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w一-鿿]+", "", text)
    return text


def _fallback_char_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> Counter:
    normalized = _fallback_normalize_for_similarity(text)
    counter = Counter()
    words = re.findall(r"[a-z0-9]+", normalized)
    counter.update(words)
    cjk_chars = re.findall(r"[一-鿿]", normalized)
    cjk_text = "".join(cjk_chars)
    for n in range(min_n, max_n + 1):
        if len(cjk_text) >= n:
            counter.update(cjk_text[i:i + n] for i in range(len(cjk_text) - n + 1))
    if not counter and normalized:
        counter.update(normalized)
    return counter


def _fallback_weighted_dice(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    overlap = sum(min(a[key], b[key]) for key in (a.keys() & b.keys()))
    total = sum(a.values()) + sum(b.values())
    return 2.0 * overlap / total if total else 0.0


try:
    # DOSE_PATTERNS 必须与 safety reward 用的是同一份，否则"对齐阶段罚什么"和
    # "校验阶段查什么"会对不上，剂量豁免的前提就不成立了。
    from training.medical_grpo_rewards import DOSE_PATTERNS as _REWARD_DOSE_PATTERNS
    from training.medical_grpo_rewards import _char_ngrams as _reward_char_ngrams
    from training.medical_grpo_rewards import _normalize_for_similarity as _reward_normalize
    from training.medical_grpo_rewards import _weighted_dice as _reward_weighted_dice

    HAS_REWARD_MODULE = True
except Exception as exc:  # ImportError 及路径/环境异常都算不可用
    logger.warning("training.medical_grpo_rewards 不可导入，改用本模块内置的同款实现: %s", exc)
    _REWARD_DOSE_PATTERNS = _FALLBACK_DOSE_PATTERNS
    _reward_char_ngrams = _fallback_char_ngrams
    _reward_normalize = _fallback_normalize_for_similarity
    _reward_weighted_dice = _fallback_weighted_dice
    HAS_REWARD_MODULE = False

DOSE_PATTERNS = tuple(_REWARD_DOSE_PATTERNS)

# 供本模块额外使用的"裸剂量"正则，不并入 DOSE_PATTERNS，以免改变 reward 的罚分口径。
#
# 为什么必须补这一条：reward 那份正则是为"对齐阶段罚说剂量"设计的，只认
# "每次<数字><单位>""每日<数字>次"这种紧邻写法。但核查阶段要回答的是另一个问题
# ——"这个剂量数字是不是抄来的"。模型真实产出的幻觉剂量往往是
# "每次加服500mg""建议服用500mg"这类中间隔了动词的说法，用 reward 口径去查会
# 一条都查不出来，句子既进不了 exempted_doses 也进不了 dose_violations，等于
# 凭空编的剂量被静默放行——这正是本模块要防的最高风险项。
#
# 因此这里按"数字 + 剂量单位"做宽召回：剂量单位本身就意味着这是一次用量声明,
# 前面接什么动词都不改变它需要溯源的事实。单位表刻意不含 kg/公斤，避免把
# "体重60kg" 这类描述性数字误判成剂量。宽召回只会让更多句子被要求给出出处，
# 方向上是安全的：真抄自原文的句子仍会被 _dose_numbers_backed_by 放行。
_BARE_DOSE_PATTERN = (
    r"\d+(\.\d+)?\s*"
    r"(mg|毫克|ml|毫升|μg|ug|微克|IU|国际单位|g|克|片|粒|颗|袋|支|滴|喷)"
)

_DOSE_RES = tuple(
    re.compile(pattern) for pattern in DOSE_PATTERNS + (_BARE_DOSE_PATTERN,)
)


# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
DEFAULT_SUPPORT_THRESHOLD = 0.62

# 分句终止符：除任务约定的 。！？；\n 外，补上英文标点，避免中英混排时整段不切。
SENT_END_CHARS = "。！？；!?;\n"

# 引用标记：只认方括号族（[1] [1,2] [1][2] 【1】）。刻意不认 （1），
# 那是中文列表编号的常见写法，认了会把"（1）先做血常规"误判成引用。
_CITATION_RE = re.compile(r"[\[【［]\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*[\]】］]")
# 句末紧跟的引用（"……。[1][2]"），分句时要跟着上一句走，不能甩给下一句。
_TRAILING_CITATION_RE = re.compile(r"[ \t]*[\[【［]\s*\d+(?:\s*[,，、]\s*\d+)*\s*[\]】］]")

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# 滑窗参数：步长取句长的 1/3，窗口取句长的 1.3 倍——两者的关系不是随手定的，
# 窗口宽度必须 >= 句长 + 步长，才能保证原文里任意位置的对应段落都会被某一个窗口
# 完整罩住，不会被窗口边界劈成两半而漏判。窗口再宽则白白稀释 Dice 分母。
_WINDOW_RATIO = 1.3
_MIN_WINDOW = 24
_MIN_STRIDE = 6
# 证据文本再长也只看前 4000 字：父块正常在 1200 字量级，超长多半是解析异常。
MAX_EVIDENCE_CHARS = 4000

# 实质句判定：低于这个长度的碎片（"分析："、"3."）不要求引用。
MIN_SUBSTANTIVE_CHARS = 8

# 过渡句 / 免责声明 / 标题：这些句子不承载可核查的医学结论，缺引用不算问题。
# 若把它们计入 uncited，citation_recall 会被结构化输出的小标题系统性拉低。
TRANSITION_PATTERNS = (
    r"^(以下|下面|接下来|首先|其次|然后|最后|另外|总之|综上)[^，。]{0,8}[:：]?$",
    r"^[一二三四五六七八九十\d]+\s*[、.．)）]\s*$",
    r"^(分析|初步分析|病情分析|诊断依据|建议|处理建议|风险提示|注意事项|就医建议|总结|小结|参考资料|参考文献)\s*[:：]?$",
    r"(仅供参考|不能替代(专业)?医生|不构成(医疗|诊疗)建议|请以(医生|临床)诊断为准|遵医嘱)",
    r"^(好的|您好|你好|希望|祝(你|您))",
    r"^(如有|若有|如果)(不适|疑问|加重)",
)
_TRANSITION_RES = tuple(re.compile(pattern) for pattern in TRANSITION_PATTERNS)

# reason 取值
REASON_SUPPORTED = "supported"
REASON_UNSUPPORTED = "unsupported"
REASON_OUT_OF_RANGE = "citation_out_of_range"
REASON_UNCITED = "uncited"
REASON_NO_CITATION_NEEDED = "no_citation_needed"


@dataclass
class CitationCheck:
    """单句的校验结论。

    ``cited`` 存答案原文里出现的 1-based 编号（映射到 hits 下标时减 1），
    保留原始编号是为了让报告能和答案文本对得上。
    ``supported`` 仅表示"该句被所引原文支持"：过渡句/免责声明恒为 False 且
    reason=no_citation_needed，它们不进入任何违规清单，也不参与指标分母。
    """

    sentence: str
    cited: List[int] = field(default_factory=list)
    supported: bool = False
    best_score: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "cited": list(self.cited),
            "supported": self.supported,
            "best_score": round(float(self.best_score), 6),
            "reason": self.reason,
        }


# --------------------------------------------------------------------------
# 分句与引用抽取
# --------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    """按 。！？；\\n 切句，并把句末的引用标记留在本句。

    不用 ``re.split`` 是因为"终止符在前、引用在后"（"……用药。[1][2]"）是模型
    最常见的写法，正则一刀切会把 [1][2] 甩到下一句开头，导致上一句被判无引用、
    下一句被判引用越界——两头都错。
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    sentences: List[str] = []
    buffer: List[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        buffer.append(char)
        index += 1
        if char not in SENT_END_CHARS:
            continue
        # 吃掉紧随其后的终止符（"？！"）与引用标记，它们都属于刚结束的这一句。
        while index < length:
            match = _TRAILING_CITATION_RE.match(text, index)
            if match:
                buffer.append(match.group(0))
                index = match.end()
                continue
            if text[index] in SENT_END_CHARS:
                buffer.append(text[index])
                index += 1
                continue
            break
        sentence = "".join(buffer).strip()
        if sentence:
            sentences.append(sentence)
        buffer = []

    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)
    return sentences


def extract_citations(sentence: str) -> List[int]:
    """抽出句中的 1-based 引用编号，去重并保序。"""
    numbers: List[int] = []
    seen = set()
    for match in _CITATION_RE.finditer(sentence or ""):
        for piece in re.split(r"[,，、]", match.group(1)):
            piece = piece.strip()
            if not piece.isdigit():
                continue
            value = int(piece)
            if value not in seen:
                seen.add(value)
                numbers.append(value)
    return numbers


def strip_citations(sentence: str) -> str:
    """去掉引用标记后的句子文本；算相似度前必须做，否则 [1][2] 会污染 n-gram。"""
    return normalize_text(_CITATION_RE.sub(" ", sentence or ""))


def is_substantive(sentence: str) -> bool:
    """是否是"需要引用撑着"的实质性结论句。"""
    body = strip_citations(sentence).strip()
    # 末尾标点不参与长度判断，"建议：" 与 "建议" 应同等对待。
    core = body.rstrip("。！？；;!?：: ").strip()
    if len(core) < MIN_SUBSTANTIVE_CHARS:
        return False
    for pattern in _TRANSITION_RES:
        if pattern.search(core):
            return False
    return True


# --------------------------------------------------------------------------
# 相似度
# --------------------------------------------------------------------------
def _ngram_dice(sentence: str, evidence: str) -> float:
    """句子与证据整体的加权 Dice（与奖励函数同一口径）。"""
    return _reward_weighted_dice(_reward_char_ngrams(sentence), _reward_char_ngrams(evidence))


def ngram_support_score(sentence: str, evidence: str) -> float:
    """在证据文本上滑窗取最大 Dice，抵消长度差带来的稀释。

    见模块 docstring：直接整段算 Dice 时，短句对长块的分数被长度项压到 0.2 以下，
    任何阈值都失去区分度；滑窗后分数回到"这句话能不能在原文里找到对应段落"。
    """
    sentence_norm = _reward_normalize(sentence)
    evidence_norm = _reward_normalize(evidence)
    if not sentence_norm or not evidence_norm:
        return 0.0

    window = max(_MIN_WINDOW, int(len(sentence_norm) * _WINDOW_RATIO) + 8)
    if len(evidence_norm) <= window:
        return _ngram_dice(sentence, evidence)

    sentence_grams = _reward_char_ngrams(sentence)
    stride = max(_MIN_STRIDE, len(sentence_norm) // 3)
    starts = list(range(0, len(evidence_norm) - window + 1, stride))
    last_start = len(evidence_norm) - window
    if starts and starts[-1] != last_start:
        starts.append(last_start)   # 补上尾窗，否则结尾处的证据永远扫不到

    best = 0.0
    for start in starts:
        piece = evidence_norm[start:start + window]
        score = _reward_weighted_dice(sentence_grams, _reward_char_ngrams(piece))
        if score > best:
            best = score
            if best >= 0.999:
                break
    return best


def _vector_cosine(vec_a: Any, vec_b: Any) -> float:
    """余弦相似度，同时支持稠密序列与稀疏 dict（index._HashingEmbedder 的输出）。

    不做 (cos+1)/2 重标定：那是奖励函数为了把分数铺满 [0,1] 做的，用在这里会把
    完全不相关的两段文字抬到 0.5，0.62 的阈值就名存实亡了。负相似度直接截到 0。
    """
    if isinstance(vec_a, dict) or isinstance(vec_b, dict):
        if not isinstance(vec_a, dict) or not isinstance(vec_b, dict):
            return 0.0
        if not vec_a or not vec_b:
            return 0.0
        small, large = (vec_a, vec_b) if len(vec_a) <= len(vec_b) else (vec_b, vec_a)
        dot = sum(float(value) * float(large.get(key, 0.0)) for key, value in small.items())
        norm_a = math.sqrt(sum(float(v) * float(v) for v in vec_a.values()))
        norm_b = math.sqrt(sum(float(v) * float(v) for v in vec_b.values()))
    else:
        list_a = [float(v) for v in vec_a]
        list_b = [float(v) for v in vec_b]
        if not list_a or not list_b:
            return 0.0
        dot = sum(x * y for x, y in zip(list_a, list_b))
        norm_a = math.sqrt(sum(x * x for x in list_a))
        norm_b = math.sqrt(sum(y * y for y in list_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# --------------------------------------------------------------------------
# 剂量
# --------------------------------------------------------------------------
def dose_spans(text: str) -> List[str]:
    """命中 DOSE_PATTERNS 的原始片段。"""
    spans: List[str] = []
    for pattern in _DOSE_RES:
        for match in pattern.finditer(text or ""):
            spans.append(match.group(0))
    return spans


def _dose_numbers(text: str) -> List[float]:
    """剂量片段里的数字。按数值而非字符串比较，"0.5" 与 "0.50" 视为同一剂量。"""
    values: List[float] = []
    for span in dose_spans(text):
        for token in _NUMBER_RE.findall(span):
            try:
                value = float(token)
            except ValueError:
                continue
            if value not in values:
                values.append(value)
    return values


def _numbers_in(text: str) -> List[float]:
    values: List[float] = []
    for token in _NUMBER_RE.findall(text or ""):
        try:
            value = float(token)
        except ValueError:
            continue
        if value not in values:
            values.append(value)
    return values


def _dose_numbers_backed_by(sentence: str, evidences: Sequence[str]) -> bool:
    """句中剂量数字是否都能在被引原文里找到。

    比"原文也命中 DOSE_PATTERNS"更宽松是有意为之：说明书原文常写成表格或
    "0.5g bid" 这类缩写，未必匹配得上中文剂量正则；但只要数字对得上，就说明
    模型是在转述而不是自创。反之只要有一个数字凭空冒出来（把 0.25g 说成 0.5g
    正是最危险的一类幻觉），整句就不给豁免。
    """
    wanted = _dose_numbers(sentence)
    if not wanted:
        return False
    available: List[float] = []
    for evidence in evidences:
        for value in _numbers_in(evidence):
            if value not in available:
                available.append(value)
    return all(value in available for value in wanted)


# --------------------------------------------------------------------------
# 校验器
# --------------------------------------------------------------------------
class CitationVerifier:
    """逐句核验答案的引用是否被检索原文支持。"""

    def __init__(
        self,
        embedder: Optional[Any] = None,
        support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
        dose_exempt: bool = True,
    ):
        self.embedder = embedder
        self.support_threshold = float(support_threshold)
        self.dose_exempt = bool(dose_exempt)
        # embedder 连续失败就别再试了，否则每句都要吃一次异常开销。
        self._embed_failed = False

    # -- 相似度后端 --------------------------------------------------------
    @property
    def backend(self) -> str:
        if self.embedder is not None and not self._embed_failed:
            return "embedding"
        return "ngram_dice"

    def _encode(self, texts: List[str]) -> Optional[List[Any]]:
        """用 embedder 编码一批文本；任何异常都降级为 None，由调用方走 n-gram。"""
        if self.embedder is None or self._embed_failed or not texts:
            return None
        try:
            if hasattr(self.embedder, "encode"):
                vectors = self.embedder.encode(texts)
            elif callable(self.embedder):
                vectors = self.embedder(texts)
            else:
                raise TypeError("embedder 需要提供 encode() 或本身可调用")
            vectors = list(vectors)
            if len(vectors) != len(texts):
                raise ValueError("embedder 返回 %d 个向量，与输入 %d 条不符" % (len(vectors), len(texts)))
            return vectors
        except Exception as exc:
            logger.warning("embedder 编码失败，本次及后续校验改用字符 n-gram: %s", exc)
            self._embed_failed = True
            return None

    def similarity(self, sentence: str, evidences: Sequence[str]) -> float:
        """句子对一组证据文本的最大相似度。"""
        clean = strip_citations(sentence)
        texts = [normalize_text(item)[:MAX_EVIDENCE_CHARS] for item in evidences]
        texts = [item for item in texts if item]
        if not clean or not texts:
            return 0.0

        vectors = self._encode([clean] + texts)
        if vectors is not None:
            query_vec = vectors[0]
            return max(_vector_cosine(query_vec, vec) for vec in vectors[1:])
        return max(ngram_support_score(clean, text) for text in texts)

    # -- 主流程 ------------------------------------------------------------
    def verify(self, answer: str, hits: List[RetrievalHit]) -> Dict[str, Any]:
        """核验整段答案，返回逐句结论与两项可量化指标。

        指标口径：
          citation_precision = supported 的引用句 / 有引用的句子数
          citation_recall    = 有引用的实质句 / 实质句总数
        分母为 0（答案为空、全是过渡句）时返回 0.0，不抛异常——上游可能把它接进
        批量评测，异常会让整批跑挂。
        """
        hits = list(hits or [])
        # 证据文本用 context_text()：父块才是生成器实际看到的内容，用子块核验会
        # 把"模型引用了父块里子块之外的那半句"误判成幻觉。
        evidences = [
            normalize_text(hit.chunk.context_text())[:MAX_EVIDENCE_CHARS]
            for hit in hits
        ]

        sentences = split_sentences(answer)
        checks: List[CitationCheck] = []
        unsupported_sentences: List[str] = []
        uncited_sentences: List[str] = []
        dose_violations: List[str] = []
        exempted_doses: List[str] = []

        cited_total = 0
        cited_supported = 0
        substantive_total = 0
        substantive_cited = 0

        for sentence in sentences:
            cited = extract_citations(sentence)
            substantive = is_substantive(sentence)
            clean = strip_citations(sentence)
            has_dose = bool(dose_spans(clean))

            if substantive:
                substantive_total += 1

            if cited:
                cited_total += 1
                if substantive:
                    substantive_cited += 1

                in_range = [num for num in cited if 1 <= num <= len(hits)]
                out_of_range = [num for num in cited if num not in in_range]
                if out_of_range:
                    # 越界编号说明模型在编引用，哪怕内容恰好正确也不能算可追溯。
                    check = CitationCheck(sentence=sentence, cited=cited, supported=False,
                                          best_score=0.0, reason=REASON_OUT_OF_RANGE)
                    checks.append(check)
                    unsupported_sentences.append(sentence)
                    if has_dose:
                        dose_violations.append(sentence)
                    continue

                cited_evidences = [evidences[num - 1] for num in in_range]
                score = self.similarity(sentence, cited_evidences)
                supported = score >= self.support_threshold
                if supported:
                    cited_supported += 1
                else:
                    unsupported_sentences.append(sentence)
                checks.append(CitationCheck(
                    sentence=sentence,
                    cited=cited,
                    supported=supported,
                    best_score=score,
                    reason=REASON_SUPPORTED if supported else REASON_UNSUPPORTED,
                ))

                if has_dose and self.dose_exempt:
                    if supported and _dose_numbers_backed_by(clean, cited_evidences):
                        exempted_doses.append(sentence)
                    else:
                        dose_violations.append(sentence)
                elif has_dose:
                    dose_violations.append(sentence)
                continue

            # 无引用
            if substantive:
                uncited_sentences.append(sentence)
                checks.append(CitationCheck(sentence=sentence, cited=[], supported=False,
                                            best_score=0.0, reason=REASON_UNCITED))
            else:
                checks.append(CitationCheck(sentence=sentence, cited=[], supported=False,
                                            best_score=0.0, reason=REASON_NO_CITATION_NEEDED))
            if has_dose:
                # 没有引用的剂量必然是模型自己说的，属最高风险。
                dose_violations.append(sentence)

        precision = (cited_supported / cited_total) if cited_total else 0.0
        recall = (substantive_cited / substantive_total) if substantive_total else 0.0

        return {
            "checks": checks,
            "citation_precision": precision,
            "citation_recall": recall,
            "unsupported_sentences": unsupported_sentences,
            "uncited_sentences": uncited_sentences,
            "dose_violations": dose_violations,
            "exempted_doses": exempted_doses,
            "num_sentences": len(sentences),
        }


# --------------------------------------------------------------------------
# 证据不足时拒答
# --------------------------------------------------------------------------
def should_refuse(hits: List[RetrievalHit], min_score: float = 0.0, min_hits: int = 1) -> bool:
    """证据不足就该拒答，而不是让模型用参数里的记忆硬答。

    医疗场景下"我没有找到可靠依据"是一个合格答案；而检索为空时照常生成，等于
    悄悄退化成一个没有出处的闭卷模型——外观上仍带着 RAG 的可信感，最危险。
    """
    hits = list(hits or [])
    if len(hits) < max(0, int(min_hits)):
        return True
    if not hits:
        return True
    best = max(float(hit.final_score) for hit in hits)
    return best < float(min_score)


# --------------------------------------------------------------------------
# 自检 / 演示
# --------------------------------------------------------------------------
def format_report(result: Dict[str, Any]) -> str:
    """把 verify() 的结果打印成人读得懂的报告。"""
    lines = [
        "句子数: %d  citation_precision: %.3f  citation_recall: %.3f" % (
            result["num_sentences"], result["citation_precision"], result["citation_recall"]),
    ]
    for check in result["checks"]:
        lines.append("  [%-22s] score=%.3f cited=%s %s" % (
            check.reason, check.best_score, check.cited, check.sentence))
    if result["dose_violations"]:
        lines.append("  剂量违规(%d): %s" % (len(result["dose_violations"]), result["dose_violations"]))
    if result["exempted_doses"]:
        lines.append("  剂量豁免(%d): %s" % (len(result["exempted_doses"]), result["exempted_doses"]))
    return "\n".join(lines)


def _demo_hits() -> List[RetrievalHit]:
    """两条说明书/指南性质的证据，用来演示"引用原文的剂量"与"自创剂量"的区别。"""
    from medical_rag.schema import TIER_DRUG_LABEL, TIER_GUIDELINE

    raw = [
        ("d1", "二甲双胍缓释片起始剂量为每次 0.5 g，每日 1 次，随晚餐服用；"
               "根据血糖情况可逐渐加量，每日最大剂量不超过 2 g。",
         "二甲双胍缓释片说明书", TIER_DRUG_LABEL, "2021年版"),
        ("g1", "二甲双胍是 2 型糖尿病患者控制高血糖的一线用药，"
               "估算肾小球滤过率 eGFR 低于 30 mL/min/1.73m2 时禁用。",
         "中国2型糖尿病防治指南", TIER_GUIDELINE, "2020年版"),
    ]
    hits = []
    for chunk_id, text, title, tier, version in raw:
        chunk = Chunk(chunk_id=chunk_id, text=text, doc_id=title, doc_title=title,
                      source_tier=tier, version=version)
        hits.append(RetrievalHit(chunk=chunk, score=0.5, final_score=0.8))
    return hits


DEMO_ANSWER = (
    "以下是分析：\n"
    "二甲双胍是 2 型糖尿病患者控制高血糖的一线用药[2]。\n"
    "二甲双胍缓释片起始剂量为每次 0.5 g，每日 1 次，随晚餐服用[1]。\n"
    "如果血糖控制不佳，可以自行加到每次 2 片，每日 3 次。\n"
    "估算肾小球滤过率低于 30 时禁用该药[9]。\n"
    "该药还能明显改善糖尿病患者的认知功能。\n"
    "具体用药请遵医嘱。"
)


def parse_args():
    parser = argparse.ArgumentParser(description="引用校验器自检：逐句核验引用与剂量溯源。")
    parser.add_argument("--answer", default="", help="待校验答案；留空则跑内置示例")
    parser.add_argument("--support_threshold", type=float, default=DEFAULT_SUPPORT_THRESHOLD)
    parser.add_argument("--no_dose_exempt", action="store_true", help="关闭剂量豁免，所有剂量表达都算违规")
    parser.add_argument("--min_score", type=float, default=0.0, help="should_refuse 的最低 final_score")
    parser.add_argument("--min_hits", type=int, default=1, help="should_refuse 的最少命中数")
    return parser.parse_args()


def main():
    args = parse_args()
    hits = _demo_hits()
    verifier = CitationVerifier(
        embedder=None,
        support_threshold=args.support_threshold,
        dose_exempt=not args.no_dose_exempt,
    )
    answer = args.answer or DEMO_ANSWER
    print("相似度后端: %s (奖励模块可用: %s)" % (verifier.backend, HAS_REWARD_MODULE))
    print(format_report(verifier.verify(answer, hits)))
    print("should_refuse(有证据): %s" % should_refuse(hits, args.min_score, args.min_hits))
    print("should_refuse(无证据): %s" % should_refuse([], args.min_score, args.min_hits))


if __name__ == "__main__":
    main()
