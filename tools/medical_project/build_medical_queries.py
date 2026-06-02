#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build medical topic queries for embedding-based filtering."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


CATEGORIES = [
    "内科学",
    "外科学",
    "妇产科学",
    "儿科学",
    "急诊医学",
    "传染病学",
    "药物禁忌",
    "慢病管理",
    "检查检验解读",
    "医学影像初步解读",
    "护理与康复建议",
    "风险提示与就医建议",
]

CATEGORY_SLUGS = {
    "内科学": "internal_medicine",
    "外科学": "surgery",
    "妇产科学": "obstetrics_gynecology",
    "儿科学": "pediatrics",
    "急诊医学": "emergency",
    "传染病学": "infectious_disease",
    "药物禁忌": "drug_contraindication",
    "慢病管理": "chronic_disease",
    "检查检验解读": "lab_test",
    "医学影像初步解读": "medical_imaging",
    "护理与康复建议": "nursing_rehabilitation",
    "风险提示与就医建议": "risk_visit_advice",
    "C-eval dev医学": "ceval_dev_medical",
}

CATEGORY_THEMES = {
    "内科学": [
        "发热、咳嗽、胸痛等常见内科症状的诊断依据、鉴别诊断和就医建议",
        "腹痛、腹泻、恶心呕吐等消化系统症状的可能原因、风险信号和处理建议",
        "头晕、头痛、乏力、心悸等非特异性症状的问诊要点和检查建议",
        "呼吸困难、喘息、咳痰等呼吸系统问题的病情判断和就医时机",
        "水肿、少尿、尿频尿急等肾脏和泌尿相关症状的初步分析",
        "贫血、白细胞异常、血小板异常等血液系统问题的解释和复查建议",
        "胃痛、反酸、黑便等消化道警示症状的风险提示和就医建议",
        "心前区不适、胸闷、心率异常等心血管症状的危险分层",
    ],
    "外科学": [
        "急性腹痛、右下腹痛、腹膜刺激征等外科急腹症的识别和就医建议",
        "外伤、扭伤、切割伤后的初步处理、感染风险和何时需要急诊",
        "术后发热、伤口红肿渗液、疼痛加重等并发症风险提示",
        "疝气、胆囊结石、阑尾炎等常见外科疾病的症状特点和处理路径",
        "骨折、关节脱位、活动受限等骨科问题的固定和就医建议",
        "乳腺肿块、甲状腺结节等外科门诊问题的检查和风险沟通",
        "肛周疼痛、便血、痔疮等肛肠外科问题的鉴别和护理建议",
        "烧烫伤分级、创面护理、破伤风风险和急诊就医指征",
    ],
    "妇产科学": [
        "孕期发热、腹痛、阴道出血等症状的风险提示和产科就医建议",
        "月经异常、痛经、经量过多或过少的可能原因和检查建议",
        "白带异常、外阴瘙痒、盆腔疼痛等妇科感染相关问题的处理建议",
        "备孕、孕早期用药、叶酸补充和禁忌用药风险提示",
        "妊娠期高血压、妊娠期糖尿病的监测、生活方式和复诊建议",
        "产后发热、恶露异常、乳房胀痛等产后问题的风险识别",
        "更年期潮热、睡眠差、情绪波动等症状的评估和就医建议",
        "妇科超声提示囊肿、肌瘤、内膜增厚时的初步解读和随访建议",
    ],
    "儿科学": [
        "儿童发热、咳嗽、喘息的病情观察、家庭护理和急诊指征",
        "婴幼儿腹泻、呕吐、脱水风险的识别和补液建议",
        "儿童皮疹、过敏、荨麻疹等问题的观察重点和就医建议",
        "儿童用药安全、退热药使用边界和避免自行叠加用药",
        "新生儿黄疸、喂养困难、精神反应差等风险提示",
        "儿童生长发育、身高体重、营养摄入和体检建议",
        "儿童腹痛、便秘、食欲差等消化问题的可能原因和处理建议",
        "儿童外伤、误服异物、误服药物后的紧急处理和就医建议",
    ],
    "急诊医学": [
        "胸痛、呼吸困难、大汗、濒死感等急危重症症状的立即就医建议",
        "意识障碍、抽搐、昏迷、言语不清等神经系统急症识别",
        "严重过敏、喉头水肿、全身皮疹伴呼吸困难的急救建议",
        "高热惊厥、持续高热、精神萎靡等需要急诊评估的情况",
        "中毒、误服药物、酒精或化学品暴露后的初步处置和急诊建议",
        "外伤大出血、头部撞击、骨折疑似等急诊处理原则",
        "剧烈腹痛、呕血、黑便、便血等消化系统急症风险提示",
        "孕妇腹痛、阴道出血、胎动异常等产科急诊指征",
    ],
    "传染病学": [
        "流感、新冠、肺炎等呼吸道传染病的症状识别、隔离和就医建议",
        "发热伴皮疹、淋巴结肿大等感染性疾病的风险提示",
        "腹泻、呕吐、食物中毒相关感染的补液、隔离和就医时机",
        "乙肝、丙肝等病毒性肝炎检查结果的初步解读和复查建议",
        "结核可疑症状、长期咳嗽咳痰、盗汗消瘦的检查建议",
        "动物咬伤、抓伤后的狂犬病暴露风险和疫苗接种建议",
        "旅行后发热、腹泻、皮疹等输入性传染病风险提示",
        "抗生素使用边界、耐药风险和避免自行使用抗菌药物",
    ],
    "药物禁忌": [
        "孕妇、哺乳期、儿童、老人常见用药禁忌和咨询医生建议",
        "药物过敏史、皮疹、呼吸困难等用药后不良反应风险提示",
        "降压药、降糖药、抗凝药等慢病药物漏服或重复服用的处理原则",
        "抗生素、止痛药、退热药自行使用的风险和边界说明",
        "肝肾功能不全患者用药调整风险和就医咨询建议",
        "多种药物联用、保健品和中成药叠加使用的相互作用风险",
        "服药后头晕、恶心、皮疹、心悸等不适的观察和就医建议",
        "不提供具体处方剂量时的安全替代表达和医生评估建议",
    ],
    "慢病管理": [
        "高血压家庭监测、生活方式调整、复诊和危险信号识别",
        "糖尿病血糖波动、低血糖风险、饮食运动和复查建议",
        "冠心病、心衰、房颤等心血管慢病的症状监测和急诊指征",
        "慢阻肺、哮喘长期管理、吸入药规范使用和急性加重识别",
        "慢性肾病尿检、肌酐、蛋白尿结果解读和随访建议",
        "高脂血症、脂肪肝、痛风等代谢问题的生活方式和复查建议",
        "甲状腺功能异常、甲亢甲减症状和化验复查建议",
        "长期服药患者的依从性、复诊计划和避免自行停药提示",
    ],
    "检查检验解读": [
        "血常规中白细胞、血红蛋白、血小板异常的初步解读和复查建议",
        "肝功能、肾功能、电解质异常的可能原因和就医建议",
        "尿常规蛋白、潜血、白细胞、酮体异常的解释和风险提示",
        "血糖、糖化血红蛋白、血脂、尿酸等代谢指标解读",
        "炎症指标 CRP、降钙素原、血沉升高的临床意义和局限性",
        "甲状腺功能、性激素、肿瘤标志物异常的谨慎解读和复查建议",
        "凝血功能、D-二聚体、心肌酶、肌钙蛋白异常的风险提示",
        "检查结果正常但症状持续时的复诊和进一步评估建议",
    ],
    "医学影像初步解读": [
        "胸片或胸部 CT 提示结节、炎症、磨玻璃影时的初步解释和随访建议",
        "腹部超声提示脂肪肝、胆囊结石、肾囊肿的初步解读",
        "甲状腺、乳腺超声结节分级的风险沟通和进一步检查建议",
        "头颅 CT 或 MRI 提示梗死、出血、占位可能时的风险提示",
        "骨片提示骨折、退变、骨质疏松相关表现的处理建议",
        "妇科超声提示卵巢囊肿、子宫肌瘤、内膜异常的随访建议",
        "影像报告中建议复查、增强扫描、专科就诊时的解释",
        "影像结果不能替代医生诊断时的谨慎表达和就医建议",
    ],
    "护理与康复建议": [
        "发热、咳嗽、腹泻等常见症状的家庭护理、观察指标和就医时机",
        "术后伤口护理、换药观察、疼痛管理和感染风险提示",
        "骨折、扭伤、关节疼痛后的康复训练边界和复诊建议",
        "慢病患者饮食、运动、睡眠、体重管理和自我监测建议",
        "卧床老人压疮预防、翻身护理、营养和感染风险观察",
        "产后护理、母乳喂养、乳腺胀痛和异常症状就医建议",
        "呼吸康复、咳痰训练、吸入装置使用和急性加重识别",
        "心理压力、睡眠问题、焦虑躯体化症状的支持性建议和求助路径",
    ],
    "风险提示与就医建议": [
        "哪些症状提示需要立即急诊：胸痛、呼吸困难、意识障碍、大出血",
        "儿童、孕妇、老人、免疫低下人群出现症状时的更低就医阈值",
        "症状持续不缓解、加重、反复发作时的复诊和进一步检查建议",
        "不能仅凭线上问答确诊疾病时的安全表达和就医建议",
        "出现高热不退、剧烈疼痛、呕血黑便、抽搐等危险信号的处理建议",
        "用药后过敏、皮疹、呼吸困难、面唇肿胀等警示症状",
        "避免绝对化诊断、避免具体处方剂量、避免替代医生的回答规范",
        "报告结论存在不确定性时如何建议专科门诊或急诊评估",
    ],
}


def normalize_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            yield row


def iter_ceval_records(path):
    ceval_path = Path(path)
    if not ceval_path.exists():
        return
    if "test" in ceval_path.name.lower() or "test" in str(ceval_path).lower().split("/"):
        raise ValueError(f"Refusing to read possible C-eval test path: {ceval_path}")

    files = []
    if ceval_path.is_dir():
        files.extend(sorted(ceval_path.glob("*.jsonl")))
        files.extend(sorted(ceval_path.glob("*.csv")))
    elif ceval_path.suffix.lower() in {".jsonl", ".csv"}:
        files.append(ceval_path)

    for file_path in files:
        if "test" in file_path.name.lower():
            continue
        iterator = read_jsonl(file_path) if file_path.suffix.lower() == ".jsonl" else read_csv(file_path)
        for record in iterator:
            yield file_path, record


def extract_ceval_question(record):
    question = normalize_text(
        record.get("question")
        or record.get("stem")
        or record.get("input")
        or record.get("query")
    )
    if not question:
        return ""

    choices = []
    for key in ["A", "B", "C", "D", "E"]:
        value = normalize_text(record.get(key))
        if value:
            choices.append(f"{key}. {value}")
    if choices:
        return question + " " + " ".join(choices)
    return question


def infer_category(text, file_path=None):
    text = f"{file_path or ''} {text}"
    rules = [
        ("妇产科学", ["孕", "胎", "产", "妇科", "月经", "阴道", "卵巢", "子宫"]),
        ("儿科学", ["儿童", "小儿", "婴", "新生儿", "儿科"]),
        ("急诊医学", ["急诊", "休克", "昏迷", "抽搐", "胸痛", "呼吸困难", "中毒"]),
        ("传染病学", ["感染", "传染", "病毒", "细菌", "结核", "肝炎", "隔离"]),
        ("药物禁忌", ["药", "禁忌", "剂量", "抗生素", "不良反应", "过敏"]),
        ("外科学", ["外科", "手术", "骨折", "创伤", "阑尾", "胆囊", "疝"]),
        ("检查检验解读", ["血常规", "尿常规", "肝功能", "肾功能", "指标", "检查", "检验"]),
        ("医学影像初步解读", ["CT", "MRI", "超声", "影像", "胸片", "X线"]),
        ("慢病管理", ["高血压", "糖尿病", "慢性", "冠心病", "哮喘", "慢阻肺"]),
    ]
    for category, keywords in rules:
        if any(keyword.lower() in text.lower() for keyword in keywords):
            return category
    return "C-eval dev医学"


def make_query(query_id, category, query, source):
    return {
        "id": query_id,
        "category": category,
        "query": query,
        "source": source,
    }


def build_manual_queries():
    queries = []
    for category in CATEGORIES:
        slug = CATEGORY_SLUGS[category]
        for idx, query in enumerate(CATEGORY_THEMES[category], start=1):
            queries.append(make_query(f"query_{slug}_{idx:03d}", category, query, "manual_medical_topics"))
    return queries


def build_ceval_dev_queries(path, max_queries):
    queries = []
    seen = set()
    for file_path, record in iter_ceval_records(path) or []:
        question = extract_ceval_question(record)
        if not question or question in seen:
            continue
        seen.add(question)
        category = infer_category(question, file_path=file_path)
        slug = CATEGORY_SLUGS.get(category, "ceval_dev_medical")
        queries.append(
            make_query(
                f"query_{slug}_ceval_dev_{len(queries) + 1:03d}",
                category,
                question,
                "ceval_dev",
            )
        )
        if len(queries) >= max_queries:
            break
    return queries


def write_jsonl(path, records):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_stats(path, stats):
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate medical topic queries without using C-eval test data."
    )
    parser.add_argument(
        "--output",
        default="data_processed/medical_project/embedding/medical_topic_queries.jsonl",
        help="Output query JSONL file.",
    )
    parser.add_argument(
        "--ceval_dev",
        default="data_raw/medical_project/ceval_medical_dev.jsonl",
        help="Optional C-eval medical dev JSONL/CSV file or directory. Test data is never used.",
    )
    parser.add_argument(
        "--stats_output",
        default="outputs/medical_project/logs/medical_queries_stats.json",
        help="Output JSON stats path.",
    )
    parser.add_argument("--max_ceval_dev_queries", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    manual_queries = build_manual_queries()
    ceval_dev_queries = build_ceval_dev_queries(args.ceval_dev, args.max_ceval_dev_queries)
    queries = manual_queries + ceval_dev_queries
    category_counts = Counter(query["category"] for query in queries)

    stats = {
        "output": args.output,
        "total_queries": len(queries),
        "manual_queries": len(manual_queries),
        "ceval_dev_queries": len(ceval_dev_queries),
        "ceval_dev_path": args.ceval_dev,
        "ceval_test_used": False,
        "leakage_note": "C-eval test was not used. Optional C-eval input is restricted to dev/valid files.",
        "category_counts": dict(sorted(category_counts.items())),
    }

    write_jsonl(args.output, queries)
    write_stats(args.stats_output, stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\nPreview:")
    for query in queries[:5]:
        print(json.dumps(query, ensure_ascii=False))


if __name__ == "__main__":
    main()
