# medical_agent：MedVision-Agent 多模态医疗辅助分析智能体

`medical_agent/` 把三样东西用 LangGraph 串起来：**视觉工具**（胸片检测 / Grad-CAM 定位 / VQA）、
已完成的 **`medical_rag` 证据检索**、以及本项目已有的**医疗文本后训练模型**（做报告生成）。

设计原则只有一条：**后训练文本模型只读经过校验的工具输出。** 它不能自己编病灶坐标、
检验数值、文献来源。这条原则靠数据结构（`schema.py`）和最后一道闸（`safety.py`）保证，
不靠 prompt 自觉。

本系统是科研与辅助分析原型，**不是诊断系统**。输出不构成诊断、治疗或用药建议，
任何实际使用都必须有放射科医生复核。

---

## 1. 架构

```
validate_input -> route -> [unsupported -> format_output]
                        -> run_tools -> aggregate -> safety_check -> format_output -> END
```

| 节点 | 职责 |
| --- | --- |
| `validate_input` | 图像存在性、问题非空、lab_data 类型；问题写 `errors` 不终止 |
| `route` | 规则路由（`router.py`），输出意图 + 理由 + 工具计划；每次决策追加到 `ROUTER_LOG` |
| `run_tools` | 按 `INTENT_TOOL_PLAN` 顺序调工具；某工具失败不中断，后续工具照跑 |
| `aggregate` | 合并工具输出成 `StructuredReport`；localize 只覆盖 bbox/mask，不改检测分数 |
| `safety_check` | reporter 生成文字 → `safety.check_answer` 逐项溯源；无依据表述追加声明而非静默删除 |
| `format_output` | 拼来源清单与标注图路径；unsupported 时给出中文原因 |

意图 → 工具计划（`schema.INTENT_TOOL_PLAN`，数据而非 if/else）：

| 意图 | 触发 | 工具序列 |
| --- | --- | --- |
| `text_qa` | 无图 | retrieve_evidence |
| `image_analysis` | 有图，默认 | detect → vqa → retrieve |
| `localize` | "在哪 / 位置 / 定位 / 标出" | detect → localize → visualize |
| `report` | "报告 / 综合 / 全面" | detect → localize → vqa → retrieve → visualize |
| `unsupported` | 无输入 / 图不存在 / 格式不支持 | 无（短路） |

## 2. 五个工具

| 工具 | 后端 | 输入 | 输出 | 解决什么 |
| --- | --- | --- | --- | --- |
| `retrieve_evidence` | `medical_rag.MedicalRAG` | query | Evidence 列表 + refused | 带出处的医学证据（零改动复用） |
| `detect_abnormality` | torchxrayvision DenseNet121（18 类胸片病理，预训练不训练） | 图像 | Finding（label, score） | 异常候选分类；COCO-YOLO 不能直接用于医学病灶 |
| `localize_finding` | Grad-CAM（复用检测器，手工 hook 实现） | 图像 + 标签 | Finding（bbox + mask PNG） | 可解释定位；弥漫性阴影需框 + 掩码并存 |
| `medical_vqa` | Qwen2.5-VL-7B-Instruct（预训练，强制 JSON） | 图像 + 问题 | VQAAnswer（parse_ok 记录 JSON 是否合法） | 看图理解；`parse_ok` 是"结构化输出成功率"的依据 |
| `generate_visualization` | Pillow | 图像 + findings | 标注 PNG | 只画 findings 里有的东西 |

所有工具继承 `tools/base.py::BaseTool`，三条纪律：`run()` 永不抛异常；**真实后端不可用时返回
`ok=False`，绝不静默换 mock**；只有显式 `mock=True` 才走假后端且结果永远带 `backend="mock"`。

## 3. grounding：报告里的每个数字和病灶都要能回溯

`safety.check_answer(answer, report)`：

- **数字溯源**：答案里每个数字必须能在 findings 分数 / bbox / 证据文本 / 化验数据里找到（82% 与 0.82 互认，年份除外）
- **病灶溯源**：答案里提到的病理名（中英文）必须在 grounded findings、VQA 输出或证据文本里出现
- **表达安全**：复用训练侧 `ABSOLUTE_DIAGNOSIS_PATTERNS` 与 `DOSE_PATTERNS`
- **结构合法性**：bbox 越界、掩码退化（占比 > 0.6）、微小框、未知标签

找不到依据的记入 `ungrounded_claims`，并在答案末尾追加"以下表述未能在工具输出中找到依据"——
保留原文让读者看到，而不是静默删掉。

已知边界：证据文本里恰好出现的数字会"担保"答案里的同一数字；VQA 提到而检测器没检出的病理，
若证据文本也提到，会被视为有依据。这是数字级溯源的固有精度，评测时用 `eval_grounding.py` 的
`ungrounded_claim_rate` 看整体趋势，不把单条判定当真值。

胸片左右：图像左侧对应患者右侧。`reporter.describe_location` 输出**患者侧**，与放射科表述一致。

## 4. 跑通命令

```bash
# 裸环境：mock 后端跑通编排
python demo/medical_agent_demo.py --mock --question "二甲双胍肾功能不全能用吗"
python demo/medical_agent_demo.py --mock --image path/to/cxr.png --question "阴影在哪个位置"
python -m pytest tests/test_medical_agent.py -q

# 服务器：真实后端（先 pip install langgraph torchxrayvision pillow；VLM 与文本模型放到 models/）
MEDAGENT_TEXT_PEFT=outputs/medical_project/grpo/<your_adapter> \
python demo/medical_agent_demo.py --device cuda --image cxr.png --question "出一份报告" --dump_json out.json

# 数据准备与评测
bash scripts/medical_project/agent/run_01_prepare_data.sh
bash scripts/medical_project/agent/run_03_eval_agent.sh
python eval/medical_project/agent/eval_vqa.py --prediction_file ... --output ...
python eval/medical_project/agent/eval_localization.py --gt_file ... --pred_file ... --output ...
```

配置项全部可用 `MEDAGENT_*` 环境变量覆盖（见 `config.py`），每次运行的返回里带 `config` 快照。

## 5. 数据集

| 数据集 | 用途 | 获取 |
| --- | --- | --- |
| VQA-RAD | 小规模验证 VLM 医学视觉问答；`eval_vqa.py` | https://osf.io/89kps/（CC BY 4.0） |
| CheXlocalize | 定位 IoU / Dice；十类发现的专家分割 | Stanford AIMI，需申请 |
| PMC-VQA | 筛选后做 Phase 4 训练样本 | 公开 |
| MIMIC-CXR | 胸片 + 报告 | PhysioNet 需资质认证，当前不依赖 |

训练与验证务必按患者/图像划分（`prepare_vqa_rad.py` 的 `image_overlap_between_splits` 必须为 0）。
患者影像与记录未经授权与脱敏不得上传第三方服务。

## 6. 评测指标

| 脚本 | 指标 | 含义 |
| --- | --- | --- |
| `eval_vqa.py` | closed_acc / open_f1 / json_parse_rate | VLM 答对率与结构化输出成功率 |
| `eval_localization.py` | mean_iou / hit@0.5 / mean_dice | 定位与专家标注的几何重合度 |
| `eval_grounding.py` | ungrounded_claim_rate / absolute_diagnosis_rate / dose_expression_rate | 报告是否忠实于工具输出 |
| `eval_agent.py` | tool_success_rate / e2e_completion_rate / latency p95 | 系统工程指标 |

**任何评测输出里 `mock_ratio > 0` 都会带 warning**：mock 验证的是编排逻辑，不是系统能力。
所有指标都是启发式自动指标，不等于临床正确性。

## 7. 分阶段路线

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 契约 | schema / config / graph_runtime / base / router / safety | 完成 |
| 1 编排 | 五工具接口 + mock、aggregate、reporter、graph、pipeline、单测 | 完成（本机可跑） |
| 2 真实工具 | torchxrayvision、Grad-CAM、Qwen2.5-VL、Pillow、文本模型 | 代码就绪，待服务器验证 |
| 3 评测基线 | VQA-RAD / CheXlocalize 上的 Base VLM 数字 | 脚本就绪，待数据 |
| 4 VLM LoRA | 仅在 Phase 3 暴露明确不足后做 | 未开始 |
| 5 多模态 GRPO | 仅在 Phase 4 后有量化错误模式再做 | 未开始 |

先用不微调，系统跑通、基线到手，再决定训不训。

## 8. 依赖与降级

| 依赖 | 缺失时 |
| --- | --- |
| langgraph | `graph_runtime` 同语义回退运行时，`RUNTIME_BACKEND="fallback"` |
| torch / torchxrayvision | detect / localize `available=False`，图照跑，报告 caveats 记录 |
| transformers | vqa / reporter 真实后端不可用；reporter 走确定性模板 |
| Pillow | visualize 不可用 |
| numpy / scipy | Grad-CAM 不可用；`eval_localization` 不算 Dice |

裸环境（只有标准库 + pytest）下 `tests/test_medical_agent.py` 必须全绿。
