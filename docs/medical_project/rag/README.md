# medical_rag：带出处与引用核查的医疗检索增强问答

`medical_rag/` 是一个独立于训练链路的医疗 RAG 子系统。它要解决的不是"能不能答"，
而是**"答的每一句话，能不能指回一份可核对的原文"**。

本仓库原有的 `demo/chatpdf.py` 是一个通用文档问答 demo：切片是定长字符窗口、
chunk 以裸字符串存在、召回只有向量与 BM25 两路、生成完就直接展示。
放到医疗场景里，这几点都会出问题——一条推荐脱离了"哪一版指南、第几章、
什么推荐等级"，读者无法判断它该不该信；而一个没有引用核查的回答，
一旦模型把 `0.25g` 说成 `0.5g`，外观上和正确答案没有任何区别。

---

## 1. 架构：两条链路

```
离线建库                                        在线查询
────────────────────────────────────────    ────────────────────────────────────────
PDF / DOCX / MD / TXT / QA-JSONL                 用户问题
        │                                             │
        │ parsers.load_document                       │ terminology.expand_query
        │  (pymupdf / pdfplumber / python-docx)       │  (心梗 -> 心肌梗死 / AMI ...)
        ▼                                             ▼
     RawDoc(blocks)                              查询变体 1..N
        │  每个 block 带 page / is_heading /           │
        │  heading_level / content_type               ├──────────┬──────────┐
        ▼                                             ▼          ▼          ▼
 chunker.StructureAwareChunker                   DenseIndex  BM25Index  TermIndex
        │  章节栈 -> section_path                  (语义)     (词面)     (术语)
        │  parent(1200字) / child(320字) 两层          │          │          │
        │  表格、QA 作为原子块不切碎                    └────┬─────┴────┬─────┘
        ▼                                                  ▼          │
   List[Chunk]  ──► chunks.jsonl                 fusion.reciprocal_rank_fusion
   （doc/章节/页码/版本/推荐等级/authority                   │ RRF：奖励多路共识
     全部随 chunk 落盘，检索后不会丢）                        ▼
                                                 rerank.CrossEncoderReranker
                                                          │ 交叉编码器精排（可缺省）
                                                          ▼
                                                 rerank.apply_authority_boost
                                                          │ 权威度 + 时效性偏置
                                                          ▼
                                                 verifier.should_refuse  ──► 证据不足则拒答
                                                          │
                                                          ▼
                                                 generator.build_prompt  (small-to-big：
                                                          │  检索用 child，送父块进 prompt)
                                                          ▼
                                                    HFGenerator.generate
                                                          │
                                                          ▼
                                                 verifier.CitationVerifier
                                                          │ 逐句核验 [n] 是否被原文支持
                                                          ▼
                                                 答案 + 来源清单 + 核查报告
```

两条链路由 `pipeline.MedicalRAG` 串起来，`demo/medical_rag_demo.py` 是它的命令行门面。

## 2. 模块职责

| 模块 | 职责 | 关键决策 |
| --- | --- | --- |
| `schema.py` | 全系统数据契约：`RawDoc` / `Chunk` / `RetrievalHit`、来源分级、JSONL 读写 | 零依赖，任何环境都能导入；`Chunk` 自带 `citation_label()`，来源格式只有一处定义 |
| `parsers.py` | PDF / DOCX / Markdown / TXT / QA-JSONL → `RawDoc` | PDF 双后端（pymupdf 取正文 + pdfplumber 抽表格）；靠字号与序号前缀还原标题层级 |
| `chunker.py` | 结构感知父子切片 | 标题栈产出 `section_path`；表格 / QA 作为原子块整块保留；推荐等级 child→parent→标题逐级继承 |
| `terminology.py` | 术语归一与同义扩展 | 内置同义词组 + 可合并外部词表；最长匹配扫描，"2型糖尿病"不会再被拆出一个"糖尿病" |
| `index.py` | 三路召回 `dense` / `bm25` / `term` | 统一返回 `(chunk 下标, 归一化分数)`；英文药名、剂量、ICD 编码按整词保留，不切 bigram |
| `fusion.py` | RRF 多路融合 | 只用名次不用原始分，避开三路分数量纲不可比的问题；`route_ranks` 把"哪一路没召回"显式写成 `None` |
| `rerank.py` | 交叉编码器精排 + 权威度/时效性加权 | `apply_authority_boost` 是**裁判不是主排序**：同分时指南压过问诊，语义分差得远时翻不过来 |
| `retriever.py` | 查询扩展 → 多路召回 → 融合 → 精排的门面 | 单路失败只跳过该路，不让整条链路中断 |
| `generator.py` | 证据拼装、prompt 组装、HF 生成 | 证据预算按条数均摊而非"前几条给足"，保证 `[n]` 编号稳定 |
| `verifier.py` | 逐句引用核查 + 证据不足拒答 | 相似度口径与 `training/medical_grpo_rewards.py` 对齐，训练与推理不分叉 |
| `pipeline.py` | 端到端管线 `MedicalRAG` | 每个环节异常隔离：检索炸了当作无召回 → 自然走拒答，而不是把异常抛给上层 |

## 3. 与 `demo/chatpdf.py` 的差异

| 维度 | `demo/chatpdf.py` | `medical_rag` |
| --- | --- | --- |
| chunk 形态 | 裸字符串，进 `corpus` 字典后只剩文本 | `Chunk` 数据类，doc / 章节 / 页码 / 版本 / 推荐等级 / 来源分级全程不丢 |
| 切片 | 定长 250 字符滑窗，按标点微调 | 结构感知：章节栈 + parent/child 两层；表格与 QA 原子不切碎 |
| 召回 | 向量 + BM25 两路（`EnsembleSimilarity`） | dense + bm25 + **term（术语归一）** 三路，解决"心梗 vs 急性ST段抬高型心肌梗死"这类同义改写 |
| 融合 | 加权分数求和 | RRF 按名次融合，天然容忍缺路；`route_ranks` 可审计 |
| 精排 | 无 | 交叉编码器精排 + 来源权威度/时效性偏置 |
| 拒答 | 靠 prompt 里一句"无法回答请说……"，模型可以不听 | `should_refuse` 是生成之前的**硬闸门**，不组装 prompt、不调用模型 |
| 引用 | 在证据块前加编号，是否引用全看模型自觉 | `CitationVerifier` 逐句核验：编号越界、无证据支撑、缺引用、剂量违规分别判定 |
| 降级 | 直接 `import torch / jieba / similarities`，缺一个就起不来 | 所有重依赖可选，缺失时降级并 warning；裸环境（只有标准库）可完整跑通检索链路 |
| 可解释 | 只输出答案与参考片段 | `retrieval_explain` 给出每路名次、融合分、精排分、最终分、耗时、降级项 |

## 4. 完整跑通命令序列

以下命令在**没有 GPU、没有 sentence_transformers / faiss / jieba / rank_bm25** 的机器上
也能全部跑完（走降级路径），结果可读但语义召回质量会下降——详见第 7 节。

### 4.1 建语料：分层原始文档 → chunks.jsonl

`tools/medical_project/rag/build_corpus.py` 把四类来源按 `source_tier` 打标后统一切片：

```bash
python tools/medical_project/rag/build_corpus.py \
  --guideline_dir      data_raw/medical_project/rag/guideline \
  --drug_label_dir     data_raw/medical_project/rag/drug_label \
  --encyclopedia_jsonl data_raw/medical_project/rag/huatuo_encyclopedia_qa.jsonl \
  --knowledge_graph_jsonl data_raw/medical_project/rag/huatuo_knowledge_graph_qa.jsonl \
  --max_encyclopedia 20000 --max_knowledge_graph 20000 \
  --child_chars 320 --child_overlap 64 --parent_chars 1200 --min_chars 40 \
  --output         data_processed/medical_project/rag/chunks.jsonl \
  --summary_output data_processed/medical_project/rag/corpus_summary.json
```

或者用封装好的 stage 脚本（缺失的来源自动跳过，不会让整条流水线失败）：

```bash
bash scripts/medical_project/rag/run_01_build_corpus.sh --help   # 看全部可覆盖变量
bash scripts/medical_project/rag/run_01_build_corpus.sh
```

**L3 在线问诊默认不入库**，必须显式 `--include_consultation`（或
`INCLUDE_CONSULTATION=true`）才会进主库，理由见第 5 节口径纪律第 1 条。

`corpus_summary.json` 里要重点看两个数：`without_section_path` 应接近 0
（否则说明 PDF 的标题层级没解析出来，`section_path` 整体为空，必须先修解析）；
`with_evidence_level` 是指南推荐等级的抽取覆盖率。这两个数字就是"元数据没丢"的量化证据。

也可以只跑底层的两步 CLI 做定位（build_corpus 内部就是这两步的组合）：

```bash
python -m medical_rag.parsers --input <目录> --source_tier L1_guideline \
  --version 2023 --output /tmp/docs.jsonl
python -m medical_rag.chunker --input /tmp/docs.jsonl --output /tmp/chunks.jsonl
```

### 4.2 建术语表（可选，但强烈建议）

`term` 路的召回质量直接取决于同义词表。不建表时用内置表，够用但覆盖面有限：

```bash
python -m medical_rag.terminology \
  --build_from data_raw/medical_project/rag/huatuo_knowledge_graph_qa.jsonl \
  --output     data_processed/medical_project/rag/terminology.jsonl

# 抽查某条 query 的术语识别与扩展效果
python -m medical_rag.terminology --query "心梗后还能吃阿司匹林吗" --max_expansions 6
```

### 4.3 建索引：dense 向量落盘

`bm25` 与 `term` 是纯 Python 倒排，重建只要几秒，没有落盘价值；只有 `dense`
需要过模型，所以只缓存它。

```bash
python tools/medical_project/rag/build_index.py \
  --chunks      data_processed/medical_project/rag/chunks.jsonl \
  --index_dir   data_processed/medical_project/rag/index \
  --routes      dense,bm25,term \
  --dense_model models/bge-m3 \
  --terminology data_processed/medical_project/rag/terminology.jsonl

# 或：
bash scripts/medical_project/rag/run_02_build_index.sh
```

跑完**必须看 `index_dir/index_summary.json`**：它记录每一路实际落到了哪个 backend。
`dense` 在缺 sentence_transformers 或本地模型时会静默降级成哈希字符 n-gram，
仍然能检索但明显更弱，这份 summary 是唯一能看出来的地方。

缓存里带 chunk_id 指纹：语料换了而缓存没删时 `load` 会拒绝加载并触发重建，
不会静默地拿上一版语料去检索。

抽查单路召回：

```bash
python -m medical_rag.index     --chunks data_processed/medical_project/rag/chunks.jsonl \
  --query "二甲双胍肾功能不全能用吗" --topk 5
python -m medical_rag.retriever --chunks data_processed/medical_project/rag/chunks.jsonl \
  --query "心梗要做什么检查" --topk 5 --explain
```

### 4.4 demo：问答

```bash
# 无 GPU：只跑检索与 prompt 组装，并打开三路检索明细
python demo/medical_rag_demo.py \
  --chunks data_processed/medical_project/rag/chunks.jsonl \
  --query "二甲双胍肾功能不好还能用吗" \
  --topk 5 --no_generate --explain

# 接本地模型后进入交互问答（不给 --query 即进入交互循环）
python demo/medical_rag_demo.py \
  --chunks data_processed/medical_project/rag/chunks.jsonl \
  --dense_model     models/bge-large-zh-v1.5 \
  --reranker_model  models/bge-reranker-v2-m3 \
  --generator_model outputs-sft-qwen-v1 \
  --peft_path       outputs-dpo-qwen-v1 \
  --device cuda --topk 5

# 单条问题落盘完整结果（含逐句核查明细），供离线分析
python demo/medical_rag_demo.py \
  --chunks data_processed/medical_project/rag/chunks.jsonl \
  --query "二甲双胍肾功能不好还能用吗" --no_generate \
  --dump_json outputs/medical_project/rag/case_metformin.json
```

`--no_generate` 是本地开发的默认姿态：它跑完检索、融合、精排、拒答判定与
prompt 组装，只是不加载生成模型，因此在 CPU 机器上是秒级的。
`--log_level INFO` 可以把各模块的降级原因打出来。

### 4.5 评测：检索质量 + 引用质量

评测集每行形如 `{"query": ..., "gold_chunk_ids": [...]}`；只有参考答案时写成
`{"query": ..., "gold_answer": ...}` 并开启弱标注。

```bash
# 检索侧：单路 recall@k / 融合增量 / 精排后的 MRR、nDCG
python eval/medical_project/rag/eval_retrieval.py \
  --chunk_file data_processed/medical_project/rag/chunks.jsonl \
  --eval_file  data_processed/medical_project/rag/eval_queries.jsonl \
  --output     outputs/medical_project/rag_eval/rag_retrieval_baseline.json \
  --model rag_baseline --topk_list 1,3,5,10 --weak_label_from_answer

# 引用侧：逐句核验 precision / recall / 剂量违规 / 拒答率
python eval/medical_project/rag/eval_citation.py \
  --prediction_file outputs/medical_project/rag_eval/predictions_rag_baseline.jsonl \
  --output          outputs/medical_project/rag_eval/rag_citation_baseline.json \
  --model rag_baseline --support_threshold 0.62

# 或者一次跑完两段并合并成一张表（predictions 不存在时会先批量作答生成）
bash scripts/medical_project/rag/run_03_eval_rag.sh
```

两项主指标：`citation_precision`（有引用的句子里被原文支持的比例）与
`citation_recall`（实质结论句里带了引用的比例）。两者必须一起看——
只看 precision，模型少引用就能拿高分；只看 recall，模型乱引用也能拿高分。
`dose_violations` 非空的样本必须逐条人工过一遍。

`run_03` 在没有 `GENERATOR_MODEL` 时只跑检索评测，引用指标留空。这是有意的：
给一个"什么都没答"的管线打引用满分，比没有指标更误导。

### 4.6 模块级自检

不需要任何语料，用各模块内置的样例数据验证装配与降级路径：

```bash
python -m medical_rag.pipeline --no_generate   # 端到端管线
python -m medical_rag.verifier                 # 引用核查
python -m medical_rag.fusion                   # RRF 融合
python -m medical_rag.rerank                   # 权威度加权
python -m pytest tests/test_medical_rag.py -q  # 单测（裸环境必须全绿）
```

## 5. 语料分层与口径纪律

| 分级 | 常量 | authority | 典型来源 | 是否进主库 |
| --- | --- | --- | --- | --- |
| L1 | `TIER_GUIDELINE` | 1.00 | 临床诊疗指南、专家共识 | 是 |
| L1 | `TIER_DRUG_LABEL` | 0.95 | 药品说明书 | 是 |
| L2 | `TIER_TEXTBOOK` | 0.75 | 教科书、规范化教材 | 是 |
| L2 | `TIER_ENCYCLOPEDIA` | 0.60 | 医学百科词条 | 是 |
| L2 | `TIER_KNOWLEDGE_GRAPH` | 0.55 | 结构化医学知识图谱 | 是（主要供术语表使用） |
| L3 | `TIER_CONSULTATION` | 0.30 | 在线问诊 QA、论坛问答 | **默认否** |

### 口径纪律（写报告、做演示前请逐条确认）

1. **在线问诊 QA 默认不进主库。** L3 语料没有经过医学审核，把它检索出来并配上
   一个 `[3]` 的引用编号，比不做 RAG 更危险——错误被引用格式包装成了可信的样子。
   确需引入（例如做"口语化提问 → 规范表述"的召回桥）时，必须：
   单独建库或显式标注分级、在界面上与 L1/L2 视觉区分、并在任何指标里分开统计。
2. **引用校验是启发式，不等于医学准确率。** `CitationVerifier` 判断的是
   "这句话在被引原文里找得到对应表述"，用的是字符 n-gram 滑窗 Dice（或可选的
   embedding 余弦）。它查得出"凭空捏造"和"引用编号错位"，查不出
   "原文本身就过时"、"适应证张冠李戴"、"结论在特定人群不成立"。
   `citation_precision = 1.0` 只意味着答案忠实于检索到的片段，
   **不意味着答案在医学上正确**。
3. **`authority` 是排序裁判，不是真值裁判。** 权威度只在语义分接近时决定先后，
   不代表 L1 的内容一定正确、L3 的内容一定错误。
4. **本系统不是医疗诊断系统。** 它是一个检索增强问答的工程实现与教学样例，
   输出不构成诊断、治疗或用药建议。任何实际使用都必须有执业医师在环复核。
   这一口径与 `docs/medical_project/README.md` 和仓库根目录的 `DISCLAIMER` 保持一致。
5. **降级路径要在结论里说清楚。** 裸环境下 `dense` 走的是字符 n-gram 哈希向量，
   不是真正的语义召回；`term` 路在没有 jieba 时用的是字符 bigram 兜底。
   拿降级路径跑出来的数字去和"完整依赖下的 RAG"作比较，是无效比较。
   `retrieval_explain.route_backends` 与 `retrieval_explain.degraded` 就是为了
   让这件事无法被悄悄略过。

## 6. 剂量豁免（dose exemption）

### 设计动机

训练侧的 `training/medical_grpo_rewards.py` 有一条安全奖励：**答案里出现具体
用法用量就扣分**。理由很直接——一个没有患者体重、肝肾功能、合并用药信息的模型，
自己说出"每次2片，每天3次"是危险的。

但这条规则直接搬到 RAG 场景会走向另一个极端：用户问"二甲双胍怎么吃"，
系统检索到了药品说明书原文"起始剂量为每次0.5g，每日2次"，
此时**拒绝复述说明书**并不会让回答更安全，只会让 RAG 退化成一个不敢念稿的搜索引擎。

差别在于剂量的**来源**：模型自己生成的剂量是幻觉风险，从被引原文里转述的剂量是引用行为。
剂量豁免就是把这两件事区分开。

### 规则

一条含剂量表述的句子（命中 `DOSE_PATTERNS`，与安全奖励是同一份正则），
**同时满足以下两条**才被判为"豁免"，否则一律计入 `dose_violations`：

1. 该句带引用编号，且编号在范围内，且相似度达到 `support_threshold`（默认 0.62）——
   即这句话确实是在转述被引证据；
2. 句中剂量片段里的**每一个数字**都能在被引证据原文中找到。

第 2 条刻意比"证据原文也要命中剂量正则"更宽松：说明书常写成表格或 `0.5g bid`
这类缩写，未必匹配得上中文剂量正则；但只要数字对得上，就说明模型在转述而不是自创。
反过来只要有一个数字凭空出现（把 `0.25g` 说成 `0.5g` 正是最危险的一类幻觉），
整句立刻失去豁免。

落到分支上：

| 情况 | 结果 |
| --- | --- |
| 有引用 + 被支持 + 数字全部对得上 | `exempted_doses` |
| 有引用 + 被支持 + 有数字对不上 | `dose_violations` |
| 有引用但编号越界 / 不被支持 | `dose_violations` |
| 无引用（模型自己说的剂量） | `dose_violations`（风险最高） |
| `CitationVerifier(dose_exempt=False)` | 任何剂量句一律 `dose_violations` |

关掉豁免（`--no_dose_exempt`）适用于面向患者的对外产品；保留豁免适用于
面向医务人员、以"把原文找出来并准确复述"为目标的场景。

同时，`generator.py` 的 prompt 规则第 5 条与此配套：
要求模型给用法用量时必须以引用原文的形式给出，并注明"以说明书和医嘱为准"。
训练端罚什么、prompt 端要求什么、校验端查什么，三处口径必须一致，
否则就会出现"训练时教它别给剂量、推理时又逼它给"的自相矛盾。

## 7. 依赖与降级矩阵

所有可选依赖的 `ImportError` 都只降级 + `logging.warning`，不会让模块导入失败。

| 依赖 | 有 | 无（降级行为） |
| --- | --- | --- |
| `sentence_transformers` | `DenseIndex` 真语义向量；交叉编码器精排可用 | 字符 n-gram 哈希稀疏向量；精排关闭，按融合分排序 |
| `faiss` | 稠密向量 ANN 检索 | numpy 暴力点积；再无 numpy 则纯 Python |
| `numpy` | 稠密矩阵运算 | 纯 Python 列表/字典运算 |
| `rank_bm25` | `BM25Okapi` | 自实现 BM25（k1=1.5, b=0.75），同一份倒排与 idf 平滑 |
| `jieba` | 中文分词 | 中文字符 bigram 切词 |
| `torch` / `transformers` | `HFGenerator.available == True` | `available == False`，`answer` 为空但检索与 prompt 照常返回 |
| `pymupdf` / `pdfplumber` | PDF 正文 + 表格 | 缺一个只丢对应能力；两个都缺则 PDF 不可解析，其他格式不受影响 |

判断当前实际走的是哪条路径，看 `retrieval_explain.route_backends`：
`dense=hashing+sparse`、`bm25=builtin_bm25`、`term=Terminology` 就是完整降级状态。

## 8. 测试

```bash
python -m pytest tests/test_medical_rag.py -q
```

`tests/test_medical_rag.py` 只用标准库 + pytest，覆盖：
`Chunk` 序列化往返与 `citation_label`、chunker 的 `section_path` 继承与标题栈弹出、
表格原子性、术语归一与查询扩展、BM25 对英文药名与剂量数字的精确匹配、
RRF 对多路共识的奖励与权重消融、权威度加权的同分翻转与"翻不过来"的边界、
引用越界判负、剂量豁免与剂量违规的各条分支、`should_refuse` 的空证据分支，
以及裸环境下 `MedicalRAG` 的端到端检索 + prompt 组装。

这些用例刻意不加载任何外部模型、不联网：它们验证的正是"依赖全缺时系统仍然可用"。
