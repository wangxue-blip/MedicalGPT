# -*- coding: utf-8 -*-

from training.medical_grpo_rewards import (
    combined_medical_reward,
    length_repetition_penalty,
    medical_format_reward,
    medical_safety_reward,
    reference_similarity_reward,
)


def completion(text):
    return [[{"content": text}]]


GOOD_ANSWER = (
    "病情分析：发热、咳嗽可能与上呼吸道感染、支气管炎等有关，也需要结合体温和检查判断。"
    "处理建议：注意休息、补充水分，观察体温和呼吸情况。"
    "风险提示：如果出现持续高热、胸痛、呼吸困难、意识异常等加重表现，需要警惕。"
    "就医建议：症状持续或加重时建议及时就医，由医生结合查体和检查评估。"
)


def test_medical_format_reward_good_answer_is_high():
    reward = medical_format_reward(completion(GOOD_ANSWER))[0]
    assert reward == 1.0


def test_medical_format_reward_drops_without_risk_and_visit_advice():
    text = "病情分析：可能与感冒有关。处理建议：多休息，多饮水。"
    reward = medical_format_reward(completion(text))[0]
    assert 0.0 < reward < 1.0


def test_medical_format_reward_supports_required_sections():
    text = "分析：可能原因较多。风险提示：若明显加重需警惕。"
    reward = medical_format_reward(
        completion(text),
        required_sections=[["诊断依据", "风险提示"]],
    )[0]
    assert reward == 1.0


def test_reference_similarity_reward_shared_medical_terms_higher_than_unrelated():
    ref = "发热咳嗽可能与呼吸道感染有关，需要观察体温，症状加重时及时就医。"
    related = "发热、咳嗽常见于呼吸道感染，建议观察体温，若症状加重应及时就医。"
    unrelated = "今天适合整理电脑文件，备份照片并清理磁盘空间。"
    rewards = reference_similarity_reward(
        completion(related) + completion(unrelated),
        [ref, ref],
    )
    assert rewards[0] > rewards[1]
    assert rewards[0] > 0.25


def test_reference_similarity_reward_does_not_require_embedding_model():
    reward = reference_similarity_reward(completion(GOOD_ANSWER), ["发热咳嗽需要结合检查评估"])[0]
    assert 0.0 <= reward <= 1.0


def test_medical_safety_reward_penalizes_absolute_diagnosis():
    safe = "可能与感染有关，需结合检查判断，如症状加重建议就医。"
    unsafe = "这一定是肺炎，无需检查。"
    rewards = medical_safety_reward(completion(safe) + completion(unsafe))
    assert rewards[0] > rewards[1]


def test_medical_safety_reward_penalizes_concrete_dosage():
    safe = "用药应在医生指导下进行，不建议自行调整剂量。"
    unsafe = "建议每次2片，每天3次，自己吃药即可。"
    rewards = medical_safety_reward(completion(safe) + completion(unsafe))
    assert rewards[0] > rewards[1]
    assert rewards[1] < 0.7


def test_length_repetition_penalty_penalizes_repeated_template():
    concise = GOOD_ANSWER
    repeated = "建议及时就医。" * 80
    penalties = length_repetition_penalty(completion(concise) + completion(repeated))
    assert penalties[1] > penalties[0]
    assert penalties[1] > 0.2


def test_length_repetition_penalty_penalizes_overlong_answer():
    normal = GOOD_ANSWER
    overlong = GOOD_ANSWER + ("需要进一步结合医生面诊和检查判断。" * 120)
    penalties = length_repetition_penalty(completion(normal) + completion(overlong))
    assert penalties[1] > penalties[0]


def test_combined_medical_reward_good_answer_higher_than_bad_answer():
    bad = "一定是小问题，不用去医院。每次2片，每天3次。"
    rewards = combined_medical_reward(
        completion(GOOD_ANSWER) + completion(bad),
        [
            "发热咳嗽可能与呼吸道感染有关，需要观察风险信号，症状加重及时就医。",
            "发热咳嗽可能与呼吸道感染有关，需要观察风险信号，症状加重及时就医。",
        ],
    )
    assert rewards[0] > rewards[1]
