# -*- coding: utf-8 -*-
"""中文医学术语归一化与查询扩展。

通用 RAG 只有稠密 + BM25 两路，放到医疗场景会漏掉一大类召回失败：
用户说"心梗""慢阻肺""血压高"，而指南原文写的是"心肌梗死""慢性阻塞性肺疾病"
"高血压"。稠密向量对这种别名并不可靠（短词、缩写的语义向量噪声大），
BM25 更是字面对不上直接 0 分。所以本模块提供第三路检索所需的术语能力：

1. ``normalize``  —— 任意别名映射到规范名，供 term 路建倒排索引；
2. ``terms_in``   —— 从文本抽出规范名（最长匹配优先，"2型糖尿病"不会被切成"糖尿病"）；
3. ``expand_query``—— 把查询改写成若干别名变体，交给 BM25 / 稠密路做多查询召回。

词表为什么要硬编码在代码里：医学同义词是强先验知识，不该依赖外部文件是否存在。
``from_jsonl`` 只做"增量合并"，用于补充领域词或从语料自动抽取的结果。

本模块只依赖标准库与 :mod:`medical_rag.schema`；jieba 缺失时退化为字符级规则，
不会导入失败。
"""

import argparse
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # 作为包导入（import medical_rag.terminology）
    from .schema import normalize_text, read_jsonl, write_jsonl
except ImportError:  # 作为脚本直接运行（python medical_rag/terminology.py）
    from schema import normalize_text, read_jsonl, write_jsonl


LOGGER = logging.getLogger(__name__)

# jieba 只用于自动建表时判断候选词是否是完整词，属于"锦上添花"。
# 缺失时退化为字符级长度/字符集规则，功能不减，只是抽出的词表噪声略多。
try:
    import jieba  # type: ignore
    _HAS_JIEBA = True
except Exception:  # pragma: no cover - 取决于运行环境
    jieba = None  # type: ignore
    _HAS_JIEBA = False
    LOGGER.warning("jieba 不可用，术语抽取退化为字符 bigram 规则")


# 单字别名（如"吐""痒"）会在任意词内部误命中，且几乎没有召回收益，
# 因此全表统一要求别名至少 2 个字符。
MIN_ALIAS_LEN = 2

# --------------------------------------------------------------------------
# 内置同义词表：每组第一个元素是规范名，其余是别名/俗称/英文缩写。
# 覆盖常见病名、症状、检查、药名、科室、缩写六类。
# 只收录真实存在的医学同义关系；把握不准的宁可不收，错误的同义词会污染召回。
# 注意：刻意不收录形如 "MI""AS""PD""IV""SC""AD" 这类在中文语料里歧义过大的
# 两字母缩写（AD 会命中广告、IV 会命中罗马数字），宁可少召回不可错召回。
# --------------------------------------------------------------------------
DEFAULT_SYNONYM_GROUPS = [
    # ---- 心血管 ----
    ["心肌梗死", "心梗", "心肌梗塞", "急性心肌梗死", "AMI", "急性心梗"],
    # STEMI / NSTEMI 是处理路径完全不同的亚型（前者要求即刻再灌注），
    # 并入"心肌梗死"会让问 STEMI 的查询召回 NSTEMI 的内容。
    ["ST段抬高型心肌梗死", "ST段抬高心肌梗死", "STEMI"],
    ["非ST段抬高型心肌梗死", "非ST段抬高心肌梗死", "NSTEMI"],
    ["高血压", "血压高", "原发性高血压", "高血压病", "HTN"],
    ["冠状动脉粥样硬化性心脏病", "冠心病", "缺血性心脏病", "CHD", "CAD"],
    ["心绞痛", "胸痹", "心前区疼痛", "angina"],
    ["心力衰竭", "心衰", "充血性心力衰竭", "心功能不全", "CHF"],
    ["心房颤动", "房颤", "心房纤颤", "AF"],
    ["心律失常", "心率不齐", "心跳紊乱", "arrhythmia"],
    ["动脉粥样硬化", "动脉硬化", "atherosclerosis"],
    ["主动脉夹层", "主动脉夹层动脉瘤", "aortic dissection"],
    ["深静脉血栓形成", "深静脉血栓", "DVT"],
    ["肺栓塞", "肺动脉栓塞", "PE"],
    # ---- 内分泌代谢 ----
    ["糖尿病", "消渴", "消渴病", "DM"],
    ["2型糖尿病", "二型糖尿病", "Ⅱ型糖尿病", "II型糖尿病", "T2DM", "成人发病型糖尿病", "非胰岛素依赖型糖尿病"],
    ["1型糖尿病", "一型糖尿病", "Ⅰ型糖尿病", "I型糖尿病", "T1DM", "胰岛素依赖型糖尿病"],
    ["糖尿病酮症酸中毒", "酮症酸中毒", "DKA"],
    ["糖尿病肾病", "糖尿病性肾病", "DKD"],
    ["糖尿病视网膜病变", "糖网", "DR"],
    ["高脂血症", "血脂异常", "高血脂", "血脂高"],
    ["高尿酸血症", "尿酸高", "血尿酸升高"],
    ["痛风", "痛风性关节炎", "gout"],
    ["肥胖症", "肥胖", "超重肥胖"],
    ["代谢综合征", "MS综合征", "metabolic syndrome"],
    ["甲状腺功能亢进症", "甲亢", "甲状腺功能亢进", "hyperthyroidism"],
    ["甲状腺功能减退症", "甲减", "甲状腺功能减退", "hypothyroidism"],
    ["桥本甲状腺炎", "慢性淋巴细胞性甲状腺炎", "桥本氏病", "桥本病"],
    ["骨质疏松症", "骨质疏松", "osteoporosis"],
    # ---- 呼吸 ----
    ["慢性阻塞性肺疾病", "慢阻肺", "慢性阻塞性肺病", "COPD"],
    ["支气管哮喘", "哮喘", "气喘病", "asthma"],
    ["肺炎", "肺部感染", "pneumonia"],
    ["社区获得性肺炎", "院外肺炎", "CAP"],
    ["肺结核", "肺痨", "痨病", "TB"],
    ["上呼吸道感染", "上感", "普通感冒", "伤风", "URTI"],
    ["流行性感冒", "流感", "influenza"],
    ["急性支气管炎", "支气管炎", "bronchitis"],
    ["支气管扩张症", "支气管扩张", "支扩"],
    ["呼吸衰竭", "呼衰", "呼吸功能衰竭"],
    ["阻塞性睡眠呼吸暂停低通气综合征", "睡眠呼吸暂停", "打鼾综合征", "OSAHS"],
    # ---- 神经/精神 ----
    ["脑卒中", "中风", "脑血管意外", "卒中", "CVA"],
    ["脑梗死", "缺血性脑卒中", "脑梗塞", "脑血栓形成", "缺血性卒中"],
    ["脑出血", "出血性脑卒中", "脑溢血", "ICH"],
    ["短暂性脑缺血发作", "小中风", "TIA"],
    # "老年痴呆"是俗称的上位词，会吞掉血管性痴呆、路易体痴呆，单独成组。
    ["阿尔茨海默病", "阿尔兹海默病", "阿尔茨海默症"],
    ["痴呆", "老年痴呆", "老年性痴呆"],
    ["帕金森病", "震颤麻痹", "帕金森氏病"],
    ["癫痫", "羊癫疯", "羊角风", "癫痫症", "epilepsy"],
    ["偏头痛", "血管性头痛", "migraine"],
    ["抑郁症", "抑郁障碍", "重度抑郁障碍", "MDD"],
    ["焦虑症", "焦虑障碍", "广泛性焦虑障碍", "GAD"],
    ["精神分裂症", "精神分裂", "schizophrenia"],
    ["失眠症", "失眠", "入睡困难"],
    # "睡眠障碍"是包含 OSA / 不宁腿 / 发作性睡病的大类，归一成失眠症会把
    # OSA 患者的提问导向镇静催眠药，方向上有害，且与上面的 OSAHS 组自相矛盾。
    ["睡眠障碍", "睡眠紊乱"],
    # "口眼歪斜"同样是脑卒中的典型民间表述，归一成周围性面瘫会延误卒中识别，
    # 因此不收这个别名。
    ["面神经麻痹", "面瘫", "贝尔麻痹"],
    # ---- 消化/肝胆 ----
    ["急性胃肠炎", "肠胃炎", "胃肠炎"],
    ["慢性胃炎", "胃炎", "gastritis"],
    # 胃溃疡需活检/复查胃镜除外恶性，十二指肠溃疡基本不恶变，随访策略不同，
    # 不能互设为同义；"消化性溃疡"只保留真正互指的"胃十二指肠溃疡"。
    ["消化性溃疡", "胃十二指肠溃疡"],
    ["胃溃疡", "良性胃溃疡"],
    ["十二指肠溃疡", "球部溃疡"],
    ["幽门螺杆菌", "幽门螺旋杆菌", "Hp感染", "H.pylori"],
    ["胃食管反流病", "反流性食管炎", "胃食道反流", "GERD"],
    ["炎症性肠病", "IBD"],
    ["溃疡性结肠炎", "慢性非特异性溃疡性结肠炎", "UC"],
    ["克罗恩病", "克隆病", "节段性回肠炎", "Crohn病"],
    ["肠易激综合征", "IBS"],
    # 功能性肠病还包含功能性便秘、功能性腹胀等，不等于 IBS。
    ["功能性肠病"],
    ["急性阑尾炎", "阑尾炎", "盲肠炎"],
    ["胆石症", "胆结石", "胆囊结石"],
    ["急性胰腺炎", "胰腺炎", "AP"],
    ["肝硬化", "肝硬变", "cirrhosis"],
    ["乙型病毒性肝炎", "乙肝", "乙型肝炎", "HBV感染"],
    ["丙型病毒性肝炎", "丙肝", "丙型肝炎", "HCV感染"],
    # "脂肪肝"是上位词：_scan 做字符最长匹配，"酒精性脂肪肝"里含有"脂肪肝"，
    # 收进 NAFLD 组会把酒精性脂肪肝的语料整体标成非酒精性——病因、干预、
    # 预后完全相反，是把"非 A"归一成"A"的定义级错误。
    ["非酒精性脂肪性肝病", "非酒精性脂肪肝", "NAFLD"],
    ["酒精性脂肪性肝病", "酒精性脂肪肝"],
    ["脂肪肝", "脂肪性肝病"],
    ["肝性脑病", "肝昏迷", "门体分流性脑病"],
    ["痔疮", "痔", "内痔外痔"],
    # ---- 泌尿/肾 ----
    ["慢性肾脏病", "慢性肾病", "慢性肾功能不全", "CKD"],
    ["急性肾损伤", "急性肾衰竭", "AKI"],
    ["尿毒症", "终末期肾病", "ESRD"],
    ["肾病综合征", "肾综", "肾病综合症"],
    ["泌尿系感染", "尿路感染", "尿感", "UTI"],
    ["泌尿系结石", "尿路结石", "肾结石", "输尿管结石"],
    ["良性前列腺增生", "前列腺增生", "前列腺肥大", "BPH"],
    # "透析"含腹膜透析，不能并入血液透析。
    ["血液透析", "血透", "HD"],
    ["腹膜透析", "腹透"],
    ["透析", "肾脏替代治疗"],
    # ---- 风湿免疫/骨科 ----
    ["类风湿关节炎", "类风湿性关节炎", "类风湿", "RA"],
    ["系统性红斑狼疮", "红斑狼疮", "狼疮", "SLE"],
    ["强直性脊柱炎", "强脊炎", "竹节样脊柱"],
    ["骨关节炎", "骨性关节炎", "退行性关节病", "增生性关节炎", "OA"],
    ["腰椎间盘突出症", "腰椎间盘突出", "腰突症", "腰间盘突出"],
    ["颈椎病", "颈椎综合征"],
    ["肩周炎", "冻结肩", "粘连性肩关节囊炎", "五十肩"],
    ["骨折", "骨断裂", "fracture"],
    # ---- 血液/肿瘤 ----
    ["贫血", "血亏", "anemia"],
    ["缺铁性贫血", "小细胞低色素性贫血", "IDA"],
    ["白血病", "血癌", "leukemia"],
    ["免疫性血小板减少症", "特发性血小板减少性紫癜", "ITP"],
    ["恶性肿瘤", "癌症", "恶性新生物", "cancer"],
    ["肺癌", "支气管肺癌", "肺恶性肿瘤"],
    ["胃癌", "胃恶性肿瘤"],
    ["原发性肝癌", "肝癌", "肝细胞癌", "HCC"],
    ["结直肠癌", "大肠癌", "CRC"],
    ["乳腺癌", "乳癌", "breast cancer"],
    ["宫颈癌", "子宫颈癌", "cervical cancer"],
    ["鼻咽癌", "NPC"],
    ["前列腺癌", "prostate cancer"],
    ["化学治疗", "化疗", "chemotherapy"],
    ["放射治疗", "放疗", "radiotherapy"],
    # ---- 症状 ----
    ["发热", "发烧", "体温升高", "fever"],
    ["咳嗽", "咳喘", "cough"],
    ["咯血", "咳血", "hemoptysis"],
    ["呼吸困难", "气短", "气促", "喘憋", "dyspnea"],
    ["胸痛", "胸部疼痛", "胸口痛", "chest pain"],
    ["心悸", "心慌", "心跳加快感", "palpitation"],
    ["腹痛", "肚子痛", "肚子疼", "abdominal pain"],
    ["腹泻", "拉肚子", "泄泻", "diarrhea"],
    ["便秘", "大便干结", "排便困难", "constipation"],
    ["恶心", "反胃", "想吐", "nausea"],
    ["呕吐", "呕逆", "vomiting"],
    ["头痛", "头疼", "脑袋痛", "headache"],
    ["头晕", "头昏", "昏沉感", "dizziness"],
    ["眩晕", "天旋地转", "vertigo"],
    ["水肿", "浮肿", "肿胀", "edema"],
    ["黄疸", "身目发黄", "皮肤黄染", "jaundice"],
    ["瘙痒", "皮肤痒", "痒感", "pruritus"],
    ["乏力", "疲乏", "倦怠", "全身无力", "fatigue"],
    ["消瘦", "体重下降", "体重减轻"],
    ["抽搐", "惊厥", "抽风", "convulsion"],
    ["昏迷", "意识丧失", "不省人事", "coma"],
    ["盗汗", "夜间出汗"],
    ["尿频", "小便次数多"],
    ["血尿", "尿血", "hematuria"],
    ["便血", "大便带血", "血便"],
    # ---- 检查/检验 ----
    ["心电图", "心电", "ECG", "EKG"],
    ["超声心动图", "心脏彩超", "心脏超声", "心超", "UCG"],
    ["计算机断层扫描", "电子计算机断层扫描", "CT检查", "CT"],
    ["磁共振成像", "核磁共振", "磁共振", "核磁", "MRI"],
    ["X线检查", "X光", "X线", "拍片子"],
    ["正电子发射计算机断层显像", "PET-CT", "PETCT"],
    ["血常规", "全血细胞计数", "血细胞分析", "CBC"],
    ["尿常规", "尿液分析", "小便常规"],
    ["糖化血红蛋白", "HbA1c", "糖化血红蛋白A1c"],
    ["空腹血糖", "空腹血浆葡萄糖", "FPG"],
    ["口服葡萄糖耐量试验", "糖耐量试验", "OGTT"],
    ["低密度脂蛋白胆固醇", "低密度脂蛋白", "坏胆固醇", "LDL-C"],
    ["高密度脂蛋白胆固醇", "高密度脂蛋白", "好胆固醇", "HDL-C"],
    ["甘油三酯", "三酰甘油", "TG"],
    ["总胆固醇", "TC"],
    ["血肌酐", "肌酐", "SCr"],
    ["估算肾小球滤过率", "肾小球滤过率", "eGFR", "GFR"],
    ["丙氨酸氨基转移酶", "谷丙转氨酶", "ALT", "GPT"],
    ["天门冬氨酸氨基转移酶", "谷草转氨酶", "AST", "GOT"],
    ["C反应蛋白", "CRP", "C-反应蛋白"],
    ["降钙素原", "PCT"],
    # BNP 与 NT-proBNP 是两种不同检测物：BNP 切点约 100 pg/mL，NT-proBNP 按年龄
    # 300/900/1800 pg/mL。归一后问 NT-proBNP 切点会召回 BNP 切点，是发生在心衰
    # 诊断决策上的 10 倍量级数值错误。
    ["B型利钠肽", "脑钠肽", "BNP"],
    ["N末端B型利钠肽原", "NT-proBNP", "NTproBNP"],
    ["肌钙蛋白", "心肌肌钙蛋白", "cTn", "cTnI"],
    ["D-二聚体", "D二聚体", "D-dimer"],
    ["国际标准化比值", "INR"],
    ["促甲状腺激素", "TSH"],
    ["胃镜", "上消化道内镜", "电子胃镜"],
    ["结肠镜", "肠镜", "电子肠镜"],
    ["骨密度检测", "骨密度", "BMD"],
    ["血氧饱和度", "氧饱和度", "SpO2"],
    ["体质指数", "身体质量指数", "BMI"],
    # ---- 药物 ----
    ["阿司匹林", "乙酰水杨酸", "ASA", "aspirin"],
    ["氯吡格雷", "波立维", "clopidogrel"],
    ["阿托伐他汀", "立普妥", "atorvastatin"],
    ["瑞舒伐他汀", "可定", "rosuvastatin"],
    ["二甲双胍", "盐酸二甲双胍", "格华止", "metformin"],
    ["胰岛素", "insulin"],
    ["硝苯地平", "心痛定", "nifedipine"],
    ["氨氯地平", "络活喜", "amlodipine"],
    ["美托洛尔", "倍他乐克", "metoprolol"],
    ["缬沙坦", "代文", "valsartan"],
    ["厄贝沙坦", "安博维", "irbesartan"],
    ["依那普利", "依苏", "enalapril"],
    ["氢氯噻嗪", "双氢克尿噻", "HCTZ"],
    ["呋塞米", "速尿", "呋喃苯胺酸", "furosemide"],
    ["螺内酯", "安体舒通", "spironolactone"],
    ["华法林", "苄丙酮香豆素钠", "warfarin"],
    ["利伐沙班", "拜瑞妥", "rivaroxaban"],
    ["低分子肝素", "低分子量肝素", "LMWH"],
    ["硝酸甘油", "三硝酸甘油酯", "nitroglycerin"],
    ["对乙酰氨基酚", "扑热息痛", "泰诺林", "acetaminophen", "paracetamol"],
    ["布洛芬", "芬必得", "ibuprofen"],
    ["阿莫西林", "羟氨苄青霉素", "amoxicillin"],
    ["头孢曲松", "菌必治", "ceftriaxone"],
    ["左氧氟沙星", "可乐必妥", "levofloxacin"],
    ["阿奇霉素", "希舒美", "azithromycin"],
    ["奥美拉唑", "洛赛克", "omeprazole"],
    ["泮托拉唑", "pantoprazole"],
    ["质子泵抑制剂", "PPI", "质子泵阻滞剂"],
    ["泼尼松", "强的松", "prednisone"],
    ["地塞米松", "氟美松", "dexamethasone"],
    ["甲氨蝶呤", "氨甲蝶呤", "MTX", "methotrexate"],
    ["别嘌醇", "别嘌呤醇", "allopurinol"],
    ["秋水仙碱", "colchicine"],
    ["左甲状腺素钠", "优甲乐", "左旋甲状腺素", "levothyroxine"],
    ["沙丁胺醇", "万托林", "salbutamol", "albuterol"],
    ["布地奈德", "普米克", "budesonide"],
    ["非甾体抗炎药", "非甾体类抗炎药", "解热镇痛抗炎药", "NSAIDs"],
    ["血管紧张素转换酶抑制剂", "普利类降压药", "ACEI"],
    ["血管紧张素受体拮抗剂", "沙坦类降压药", "ARB"],
    ["他汀类药物", "他汀", "HMG-CoA还原酶抑制剂"],
    # ---- 科室 ----
    ["心血管内科", "心内科", "心脏内科"],
    ["呼吸与危重症医学科", "呼吸内科", "呼吸科"],
    ["消化内科", "消化科", "胃肠内科"],
    ["内分泌科", "内分泌代谢科"],
    ["神经内科", "神经科"],
    ["肾内科", "肾脏内科"],
    ["血液内科", "血液科"],
    ["感染性疾病科", "感染科", "传染科"],
    ["普通外科", "普外科"],
    ["骨科", "骨伤科", "矫形外科"],
    ["泌尿外科", "泌尿科"],
    ["神经外科", "脑外科"],
    ["胸外科", "心胸外科"],
    ["妇产科", "妇科"],
    ["儿科", "小儿科", "儿童医学科"],
    ["急诊科", "急诊医学科", "急救科"],
    ["重症医学科", "重症监护病房", "加强医疗科", "ICU"],
    ["耳鼻咽喉科", "耳鼻喉科", "五官科"],
    ["皮肤科", "皮肤性病科"],
    ["眼科", "眼病专科"],
    ["肿瘤科", "肿瘤内科"],
    ["麻醉科", "麻醉学科"],
    ["康复医学科", "康复科"],
    ["精神心理科", "精神科", "心理科"],
    ["中医科", "中医内科"],
    ["口腔科", "牙科", "口腔医学科"],
    # ---- 通用临床概念 ----
    ["不良反应", "副作用", "药物不良反应", "ADR"],
    ["禁忌证", "禁忌症", "contraindication"],
    ["适应证", "适应症", "indication"],
    ["并发症", "合并症", "complication"],
    ["鉴别诊断", "differential diagnosis"],
    ["预后", "转归", "prognosis"],
    ["随访", "随诊", "follow-up"],
    ["肌肉注射", "肌注", "肌内注射"],
    ["静脉注射", "静注", "静脉推注"],
    ["口服给药", "口服", "内服"],
    ["血压", "动脉血压", "BP"],
    ["心率", "心跳次数", "HR"],
    ["妊娠期妇女", "孕妇", "怀孕妇女"],
    ["新生儿", "初生儿", "neonate"],
    ["老年人", "高龄患者", "老年患者"],
]


# --------------------------------------------------------------------------
# 文本折叠与边界判定
# --------------------------------------------------------------------------
def _fold(text: str) -> str:
    """大小写折叠，且严格保持长度。

    直接用 ``str.lower()`` 不安全：个别字符（如 'İ'）小写后长度会变，
    而扫描算法依赖"折叠串下标 == 原串下标"，长度一变命中区间就错位。
    """
    chars = []
    for ch in text:
        low = ch.lower()
        chars.append(low if len(low) == 1 else ch)
    return "".join(chars)


def _alias_key(term: str) -> str:
    return _fold(normalize_text(term))


def _is_ascii_token(key: str) -> bool:
    """判断别名是否是纯英文/缩写形式。"""
    if not key:
        return False
    if not key.isascii():
        return False
    return any(ch.isalpha() for ch in key)


# 否定前缀保护。中文医学词表里 "非X" 几乎总是与 "X" 不同的实体
# （非酒精性脂肪性肝病 vs 脂肪肝、非小细胞肺癌 vs 肺癌、非ST段抬高型心梗 vs STEMI），
# 而 _scan 做的是字符最长匹配、不认词边界，"非X" 里必然含有子串 "X"。
# 若 "非X" 本身在表里，最长匹配会先命中它；命中不到才会走到这里，此时把 X 标成
# 规范名就是把"非 A"归一成"A"。宁可少一个术语标签，也不能打反标签。
#
# 刻意只收 "非"，不收 "无/未/不"：后三者是对**所见**的否定（"无不良反应"仍然是在
# 讲不良反应），把它们也挡掉会无谓地损失召回。
_NEGATION_PREFIXES = ("非",)


def _negation_ok(text: str, start: int, key: str) -> bool:
    """命中区间左侧紧邻否定字时返回 False（纯 ASCII 缩写不适用）。"""
    if start <= 0 or _is_ascii_token(key):
        return True
    return text[start - 1] not in _NEGATION_PREFIXES


def _boundary_ok(folded: str, start: int, end: int) -> bool:
    """英文缩写必须独立成词，否则 ``ASA`` 会命中 ``ASAP``、``TC`` 会命中 ``TCM``。"""
    if start > 0:
        prev = folded[start - 1]
        if prev.isascii() and prev.isalnum():
            return False
    if end < len(folded):
        nxt = folded[end]
        if nxt.isascii() and nxt.isalnum():
            return False
    return True


def segment(text: str) -> List[str]:
    """分词：有 jieba 用 jieba，没有就退化为中文字符 bigram。

    本模块只在自动建表时用它做候选词过滤，检索路径不依赖分词结果，
    所以退化不会改变 normalize / terms_in / expand_query 的行为。
    """
    text = normalize_text(text)
    if not text:
        return []
    if _HAS_JIEBA:
        try:
            return [tok for tok in jieba.lcut(text) if tok.strip()]
        except Exception as exc:  # pragma: no cover - jieba 内部异常
            LOGGER.warning("jieba 分词失败，退化为字符 bigram: %s", exc)
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[i:i + 2] for i in range(len(compact) - 1)]


# --------------------------------------------------------------------------
# 术语表
# --------------------------------------------------------------------------
class Terminology(object):
    """同义词表 + 最长匹配术语识别 + 查询扩展。

    内部维护两张表：别名(折叠后) -> 规范名、规范名 -> 有序别名列表。
    构造时如果新组里的某个词已经属于已有组，则把两组合并而不是覆盖，
    这样 ``from_jsonl`` 的外部词表可以安全地"补充"内置表。
    """

    def __init__(self, groups: Optional[Iterable[Iterable[str]]] = None):
        if groups is None:
            groups = DEFAULT_SYNONYM_GROUPS
        self._alias2canonical = {}  # type: Dict[str, str]
        self._canonical2aliases = {}  # type: Dict[str, List[str]]
        self._max_alias_len = 0
        for group in groups:
            self.add_group(group)

    # -- 构建 ---------------------------------------------------------------
    def add_group(self, group: Iterable[str]) -> Optional[str]:
        """加入一组同义词，返回该组最终的规范名（无有效词时返回 None）。"""
        terms = []
        for raw in group or []:
            term = normalize_text(raw)
            if len(term) < MIN_ALIAS_LEN:
                continue
            if term not in terms:
                terms.append(term)
        if not terms:
            return None

        # 若任一词已存在，沿用旧规范名合并，避免同一概念分裂成两组。
        canonical = None
        for term in terms:
            existing = self._alias2canonical.get(_alias_key(term))
            if existing is not None:
                if canonical is None:
                    canonical = existing
                elif existing != canonical:
                    LOGGER.warning("术语组冲突：%s 同时属于 %s 与 %s", term, canonical, existing)
        if canonical is None:
            canonical = terms[0]
            self._canonical2aliases[canonical] = []

        bucket = self._canonical2aliases.setdefault(canonical, [])
        for term in terms:
            key = _alias_key(term)
            if not key:
                continue
            if key not in self._alias2canonical:
                self._alias2canonical[key] = canonical
                self._max_alias_len = max(self._max_alias_len, len(key))
            if term not in bucket and term != canonical:
                bucket.append(term)
        # 规范名自身也要能被识别
        self._alias2canonical.setdefault(_alias_key(canonical), canonical)
        self._max_alias_len = max(self._max_alias_len, len(_alias_key(canonical)))
        return canonical

    # -- 查询接口 -----------------------------------------------------------
    @property
    def canonical_terms(self) -> List[str]:
        return list(self._canonical2aliases.keys())

    def aliases_of(self, canonical: str) -> List[str]:
        """返回规范名 + 其全部别名（规范名在首位）。"""
        canonical = self.normalize(canonical)
        if canonical not in self._canonical2aliases:
            return [canonical] if canonical else []
        return [canonical] + list(self._canonical2aliases[canonical])

    def normalize(self, term: str) -> str:
        """别名 -> 规范名；表里没有就原样返回（不做任何猜测）。"""
        if not term:
            return term
        return self._alias2canonical.get(_alias_key(term), term)

    def terms_in(self, text: str) -> List[str]:
        """返回文本中出现的规范名（按出现顺序去重）。"""
        seen = []
        for _, _, canonical, _ in self._scan(text):
            if canonical not in seen:
                seen.append(canonical)
        return seen

    def expand_query(self, query: str, max_expansions: int = 8) -> List[str]:
        """把查询改写成多个别名变体，第一个恒为原始 query。

        扩展策略：先给一条"全部术语规范化"的变体（最容易命中指南原文），
        再按别名优先级轮转做单术语替换。轮转而不是深度优先，是为了在
        多术语查询里让每个术语都有机会被替换，而不是把名额耗在第一个词上。
        """
        query = query if isinstance(query, str) else str(query or "")
        if max_expansions <= 1:
            return [query]

        spans = self._scan(query)
        if not spans:
            return [query]

        variants = [query]
        seen_keys = set([_alias_key(query)])

        def _push(candidate: str) -> None:
            """去重后追加；重复的变体跳过即可，不能因此中断整轮扩展。"""
            key = _alias_key(candidate)
            if not key or key in seen_keys:
                return
            seen_keys.add(key)
            variants.append(candidate)

        # 1) 全规范化变体：最可能字面命中指南原文，优先级最高
        _push(_replace_spans(query, [(s, e, canon) for s, e, canon, _ in spans]))

        # 2) 单术语替换，按别名序轮转
        plans = []  # type: List[List[Tuple[int, int, str]]]
        for start, end, canonical, surface in spans:
            surface_key = _alias_key(surface)
            alts = [alias for alias in self.aliases_of(canonical) if _alias_key(alias) != surface_key]
            plans.append([(start, end, alias) for alias in alts])
        depth = max([len(plan) for plan in plans] or [0])
        for level in range(depth):
            for plan in plans:
                if len(variants) >= max_expansions:
                    return variants[:max_expansions]
                if level < len(plan):
                    _push(_replace_spans(query, [plan[level]]))
        return variants[:max_expansions]

    # -- 内部 ---------------------------------------------------------------
    def _scan(self, text: str) -> List[Tuple[int, int, str, str]]:
        """最长匹配扫描，返回 (起, 止, 规范名, 原文片段)。

        从每个位置尝试最长的别名，命中后直接跳过整段，
        所以"2型糖尿病"命中后不会再把内部的"糖尿病"识别成第二个术语。
        """
        if not text:
            return []
        folded = _fold(text)
        total = len(folded)
        results = []
        index = 0
        while index < total:
            hit = None
            longest = min(self._max_alias_len, total - index)
            for length in range(longest, MIN_ALIAS_LEN - 1, -1):
                key = folded[index:index + length]
                canonical = self._alias2canonical.get(key)
                if canonical is None:
                    continue
                if _is_ascii_token(key) and not _boundary_ok(folded, index, index + length):
                    continue
                if not _negation_ok(folded, index, key):
                    continue
                hit = (index, index + length, canonical, text[index:index + length])
                break
            if hit is None:
                index += 1
                continue
            results.append(hit)
            index = hit[1]
        return results

    # -- 工厂方法 -----------------------------------------------------------
    @classmethod
    def default(cls) -> "Terminology":
        return cls(DEFAULT_SYNONYM_GROUPS)

    @classmethod
    def from_jsonl(cls, path: str) -> "Terminology":
        """读取外部词表并与内置表合并。

        每行支持两种写法：``{"canonical": str, "aliases": [str]}``
        或 ``{"group": [str, ...]}``（首元素为规范名）。
        """
        groups = [list(group) for group in DEFAULT_SYNONYM_GROUPS]
        loaded = 0
        for row in read_jsonl(path):
            group = _row_to_group(row)
            if group:
                groups.append(group)
                loaded += 1
        LOGGER.info("从 %s 合并 %d 组外部术语", path, loaded)
        return cls(groups)


def _row_to_group(row: Dict[str, Any]) -> List[str]:
    """把一行 jsonl 转成同义词组，容忍字段缺失/类型不对。"""
    if not isinstance(row, dict):
        return []
    group = row.get("group")
    if isinstance(group, (list, tuple)):
        return [normalize_text(item) for item in group if normalize_text(item)]
    canonical = normalize_text(row.get("canonical") or "")
    if not canonical:
        return []
    aliases = row.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    out = [canonical]
    for alias in aliases:
        alias = normalize_text(alias)
        if alias and alias not in out:
            out.append(alias)
    return out


def _replace_spans(text: str, replacements: List[Tuple[int, int, str]]) -> str:
    """按区间替换文本；从后往前替换以免前面的替换改变后面的下标。"""
    pieces = text
    for start, end, value in sorted(replacements, key=lambda item: item[0], reverse=True):
        pieces = pieces[:start] + value + pieces[end:]
    return pieces


# --------------------------------------------------------------------------
# 从语料自动建表
# --------------------------------------------------------------------------
# 中文医学语料里别名几乎都出现在固定句式中，用规则抽取的准确率远高于
# 让模型去猜，而且零依赖、可复现。
_TERM_RE = r"[一-龥A-Za-z0-9\-·]{2,20}"
_CONNECTOR_RE = (
    r"(?:又称之为|又称为|又称|又名|又叫做|又叫|亦称为|亦称|亦名|"
    r"也称为|也称|也叫做|也叫|俗称为|俗称|简称为|简称|全称为|全称|"
    r"学名为|学名|医学上称为|临床上称为|通称为|通称)"
)
ALIAS_PATTERN = re.compile(
    r"(%s)\s*[，,、]?\s*%s\s*[：:\"“]?\s*(%s(?:\s*[、/／,，或]\s*%s)*)"
    % (_TERM_RE, _CONNECTOR_RE, _TERM_RE, _TERM_RE)
)
ABBR_PATTERN = re.compile(
    r"([一-龥]{2,20})\s*[（(]\s*(?:英文缩写|英文简称|英文名|英文|缩写|简称)?\s*"
    r"[：:]?\s*([A-Za-z][A-Za-z0-9\-]{1,15})\s*[）)]"
)
_SPLIT_RE = re.compile(r"[、/／,，或]")

# 这些词本身是句式残留或过于泛化，抽到了也只会污染词表。
_STOP_TERMS = frozenset([
    "本病", "该病", "此病", "本药", "该药", "疾病", "症状", "患者", "病人", "医生",
    "治疗", "检查", "诊断", "药物", "注意", "以上", "如下", "其中", "因此", "所以",
    "一般", "通常", "临床", "医学", "西医", "中医", "上述", "本品", "该品",
])


def _looks_like_term(term: str) -> bool:
    """候选词的基本合法性检查。

    有 jieba 时额外要求候选词能由完整词元拼出，用来挡掉"性肺炎""者常见"
    这类跨词边界的碎片；没有 jieba 时退化为长度 + 字符集规则。
    """
    term = normalize_text(term)
    if len(term) < MIN_ALIAS_LEN or len(term) > 20:
        return False
    if term in _STOP_TERMS:
        return False
    if re.search(r"[的了是和与及等就都很更也还把被为对于在]", term):
        return False
    if not re.search(r"[一-龥A-Za-z]", term):
        return False
    if re.fullmatch(r"[0-9\-·]+", term):
        return False
    if _HAS_JIEBA:
        tokens = segment(term)
        if tokens and "".join(tokens) == re.sub(r"\s+", "", term) and len(tokens) > 3:
            return False
    return True


def _iter_texts(row: Any, depth: int = 0) -> Iterable[str]:
    """递归收集一行记录里的所有字符串，兼容各种 QA / 知识图谱字段命名。"""
    if depth > 4:
        return
    if isinstance(row, str):
        if len(row) >= 4:
            yield row
    elif isinstance(row, dict):
        for value in row.values():
            for item in _iter_texts(value, depth + 1):
                yield item
    elif isinstance(row, (list, tuple)):
        for value in row:
            for item in _iter_texts(value, depth + 1):
                yield item


def build_synonym_table_from_kg(
    input_path: str,
    output_path: str,
    max_rows: int = -1,
    min_freq: int = 1,
    max_aliases: int = 8,
) -> Dict[str, Any]:
    """从医学 QA / 知识图谱语料里用规则抽同义词，写成可被 ``from_jsonl`` 读取的词表。

    识别两类句式：
      * ``X又称Y``、``X俗称Y、Z``、``X简称Y``（连接词后允许顿号并列）
      * ``X（ABC）``、``X（英文缩写：ABC）``  —— 括号内英文当作缩写别名

    返回统计信息 dict：scanned / pairs / groups / output。
    """
    pairs = {}  # type: Dict[str, Dict[str, int]]
    scanned = 0

    for row in read_jsonl(input_path, max_rows=max_rows):
        scanned += 1
        for text in _iter_texts(row):
            for canonical, alias_blob in ALIAS_PATTERN.findall(text):
                if not _looks_like_term(canonical):
                    continue
                for alias in _SPLIT_RE.split(alias_blob):
                    _record_pair(pairs, canonical, alias)
            for canonical, abbr in ABBR_PATTERN.findall(text):
                if not _looks_like_term(canonical):
                    continue
                _record_pair(pairs, canonical, abbr)

    rows = []
    for canonical in sorted(pairs.keys()):
        aliases = [alias for alias, freq in
                   sorted(pairs[canonical].items(), key=lambda kv: (-kv[1], kv[0]))
                   if freq >= min_freq][:max_aliases]
        if aliases:
            rows.append({"canonical": canonical, "aliases": aliases})

    written = write_jsonl(output_path, rows)
    stats = {
        "scanned": scanned,
        "pairs": sum(len(v) for v in pairs.values()),
        "groups": written,
        "output": output_path,
    }
    LOGGER.info("术语抽取完成: %s", stats)
    return stats


def _record_pair(pairs: Dict[str, Dict[str, int]], canonical: str, alias: str) -> None:
    canonical = normalize_text(canonical)
    alias = normalize_text(alias)
    if not _looks_like_term(alias):
        return
    if _alias_key(alias) == _alias_key(canonical):
        return
    bucket = pairs.setdefault(canonical, {})
    bucket[alias] = bucket.get(alias, 0) + 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Chinese medical terminology normalization and query expansion."
    )
    parser.add_argument("--query", default="", help="打印该查询的术语识别与扩展结果")
    parser.add_argument("--max_expansions", type=int, default=8)
    parser.add_argument("--table", default="", help="额外的同义词 jsonl，与内置表合并")
    parser.add_argument("--build_from", default="", help="从该 jsonl 语料抽取同义词表")
    parser.add_argument("--output", default="", help="建表输出路径，默认 <build_from>.synonyms.jsonl")
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--min_freq", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.build_from:
        output = args.output or (args.build_from + ".synonyms.jsonl")
        stats = build_synonym_table_from_kg(
            args.build_from, output, max_rows=args.max_rows, min_freq=args.min_freq
        )
        print("scanned={scanned} pairs={pairs} groups={groups} output={output}".format(**stats))
        return

    terminology = Terminology.from_jsonl(args.table) if args.table else Terminology.default()

    if not args.query:
        print("synonym groups: %d" % len(terminology.canonical_terms))
        print("aliases: %d" % len(terminology._alias2canonical))
        for canonical in terminology.canonical_terms[:5]:
            print("  %s <- %s" % (canonical, " / ".join(terminology.aliases_of(canonical)[1:])))
        return

    print("query: %s" % args.query)
    print("terms: %s" % ", ".join(terminology.terms_in(args.query)))
    for i, variant in enumerate(terminology.expand_query(args.query, args.max_expansions)):
        print("  [%d] %s" % (i, variant))


if __name__ == "__main__":
    main()
