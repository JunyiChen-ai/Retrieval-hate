# Gate 0 独立查新与性能 EV 复核：MLLM × RGCL

**复核日期：** 2026-07-10（Pacific/Auckland）  
**审阅身份：** 独立资深 ML reviewer；adversarial / cross-model-style xhigh 标准  
**边界：** 只审阅本地证据与 2024–2026 一手文献；未运行计算、未改实验代码、未接触测试集  
**硬目标：** 不是 localization / audit / encoder 贡献，而是同协议下相对最强 non-MLLM RGCL，在至少两个数据集、三个 paired seeds 上，**accuracy 与 macro-F1 各至少 +0.030 absolute**，并通过统计与 remove/shuffle 机制消融。

## 1. Executive verdict

先纠正一个会误导决策的口径：`TARGET_GATE0_LITERATURE.md` 中的“substantial ≥2 points”与当前 `TARGET_LOOP.md` / `TARGET_STATE.json` 冲突。**本复核只认 +3.0 acc 与 +3.0 macro-F1 的固定终局标准**；+2 只能是早期筛选信号，不能算目标完成。

候选池必须合并为以下**恰好三个**机制族：

| ID | 合并后的候选 | 合并了哪些原始提案 | novelty | 满足完整 +3/+3 终局的 ex-ante 概率 | 决策 |
|---|---|---|---:|---:|---|
| **A** | **EDCM-RGCL：Evidence-Directed Counterfactual Memory Geometry** | Evidence-Directed Memory Writing + 反事实模态联盟→梯度控制 | **5.5/10** | **8–12%** | 第二顺位；若退化为 segment weight / 普通 distillation，立即否决 |
| **B** | **SSR-MemRGCL：Gold-Signed Semantic Relation Memory Graph** | Label-Signed Semantic Pair Graph + MLLM 关系签名→relation-conditioned RGCL | **6.5/10** | **15–20%** | **唯一首跑路线** |
| **C** | **VCRouter-RGCL：Verified Clause-Conditioned Low-Rank Router** | Schema-Conditioned Modality Router + 可执行规则→低秩专家路由 | **3.0/10** | **3–7%** | 高风险储备；不应先实现 |

这些概率是 reviewer 的保守 EV，不是置信区间或已有结果。历史上 concat、score fusion、archive-kNN、LMM-RGCL、segment weighting 与 schema distillation 均未产生最终净增益，因此任何候选达到完整终局的先验概率都不应被写成“很可能”。**B 只是 least-bad / highest-EV，不是稳过。**

### 唯一首跑选择

**首跑 B：SSR-MemRGCL。** 理由不是它听起来最“语义化”，而是它同时满足三个必要条件：

1. 最终预测器是 train-memory kNN；B 直接改变训练 memory graph 与最终可见的几何，而不是再给 head 一条可被吸收的语义输入。
2. gold label 决定边符号，MLLM 只决定“关系类型”，避免重犯 P2 让 MLLM 猜 vote correctness 的错误。
3. 关系是跨样本条件量，不能由单样本 binary label 唯一恢复；它比 P4 schema auxiliary task 更有机会提供 conditional information gain。

但是 B 的 novelty 窗口很窄。论文不能声称首次 relation-conditioned multimodal learning、首次 hard-pair contrastive learning、首次 MLLM graph supervision；能够守住的只可能是：**在 hateful-video RGCL 中，以 gold label 定符号、以 MLLM 抽取 stance–target–mechanism 类型，对真实训练样本构造稀疏 typed hard-edge graph，并直接塑造最终 kNN memory geometry。**

## 2. 审阅依据与硬排除区

### 2.1 本地权威事实

- `src/model/evaluate_rac.py` 将 train embeddings 加入 FAISS index，并由训练样本标签作检索投票；因此最终读出不是一个可忽略的附属 head，而是 **train-memory kNN**。
- `CAMPAIGN_mllm_method_role.md` 与 `PAPER_MASTER_TABLES.md` 显示：简单 archive key concat、第三流 concat、score fusion、邻居 rerank、schema-field auxiliary distillation、MLLM segment pooling、LMM-SFT/RGCL 均未带来 substantial final acc/macro-F1 净增益。
- P9/P9b 的关键机制结论是 head 与 memory 发生 accuracy redistribution，而非 synergy。
- P3/P11 对 A 构成内部先验打击：MLLM density/localization 信号即使真实，也可能不比 video-label MIL 多出足够信息，且静态 segment weighting 会被融合头吸收。
- P2/P2b 对 B 既是警告也是机会：generic comparability 与 vote correctness 近乎正交；因此 B 必须证明 **typed relation 在控制 label、相似度与 query difficulty 后仍预测 wrong-neighbour attraction**，否则只是 P2 换名。

### 2.2 自动否决条件

以下任一出现，本轮 novelty 直接降到 ≤3/10，不应进入 full run：

- 把 MLLM rationale / schema / score / summary embedding 与现有表示做 concat；
- 在 test 时用 MLLM score 线性融合、veto、rerank 或 confidence-gated arbitration；
- 主要变量是换 32B/72B、更密 frames、更多 epoch、更多训练数据或大规模 MoE 容量；
- 只有 `method vs baseline`，没有 `remove-MLLM` 与 `shuffle-MLLM-information`；
- 最终只提升 native head，而 train-memory kNN 不提升；
- 把 +2pt、单 seed、单数据集或只提升 macro-F1 写成达到目标。

## 3. 候选 A：EDCM-RGCL

### 3.1 精确定义

**Evidence-Directed Counterfactual Memory Geometry for RGCL**：冻结 MLLM 只在 train split 上比较同一视频的确定性 temporal/modality interventions，得到相对 evidence signature；该 signature 选择或构造每个训练样本的 sufficient evidence view，作为 memory-writing teacher key。学生的 full-video query embedding 通过 memory-SupNCA / listwise rank loss 与 gold-signed memory keys 对齐，使 MLLM 直接决定哪些证据能够塑造最终 kNN geometry。

可接受的 A 必须同时有：

1. 同一样本 intervention 的**相对** evidence，而不是绝对 hate verdict；
2. evidence-selected teacher key / sufficient view；
3. 直接优化 query-to-memory ranking 或 NCA，最终以该 geometry 作 kNN readout；
4. teacher signature shuffle、random-view key、video-label MIL key 三个控制。

若只用已有 P6/P10 density score做 soft pooling，A 就是 P3/P11 的重演；若只按 evidence 加权 pair loss，A 很容易被视为 CGO 的任务定制版。

### 3.2 Novelty：5.5/10

**最近工作与威胁：**

- [Controlled Gradient Optimization for Harmful Video Detection (CGO, ACL ARR 2026)](https://openreview.net/forum?id=Z51RWOPKQQ) 已在**相同任务域**用扰动可靠性、梯度方向与跨模态收敛做 gradient control。这使“首次用干预控制 harmful-video gradient”不可主张。
- [Diagnosing and Mitigating Modality Interference in MLLMs (ICLR 2026 submission)](https://openreview.net/forum?id=0Cv0whP7l8) 已用 causal/perturbation intervention 与 consistency regularization缓解模态干扰。
- [BridgeVLM / From Prompts to Tokens (ICML 2026 OpenReview entry)](https://openreview.net/forum?id=NOoAIwF6bV) 已把外部 causal supervision 内化为可执行结构；“首次 internalize causal supervision”也不可主张。
- [CFPO (ICML 2026 OpenReview entry)](https://openreview.net/forum?id=BfWFCVjsNe) 已用 counterfactual suppression 强化多模态因果依赖。
- [More Than Sum of Its Parts / H-VLI + ARCADE (Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.974/) 已把 multimodal intent shift 与对立推理用于隐式仇恨理解。
- [RAMF (TMLR 2026)](https://openreview.net/forum?id=U9KnNiuMu1) 已把 objective / hate-assumed / non-hate-assumed reasoning 与语义融合用于 hateful-video，并报告 reasoning removal 的明显 macro-F1 cost。

**仍可守的核心差异：** 现有近邻分别覆盖 causal intervention、harmful-video gradient control、MLLM reasoning 和 causal internalization，但未检到“**MLLM 相对 intervention signature 选择持久 train-memory key，并以 memory-SupNCA/listwise objective 直接塑造最终 kNN classifier geometry**”这一完整组合。这个差异是 retrieval-specific 的；删掉 memory writing / kNN ranking 后 novelty 基本不成立。

### 3.3 性能 EV

- plausible mean effect（若机制成立）：MHC-EN acc `+1.0–2.5pt`、macro-F1 `+1.0–3.0pt`；MHC-ZH 两指标 `+0.5–2.0pt`。
- 达到两数据集、三 seed、双指标都 `≥+3pt` 的 ex-ante 概率：**8–12%**。
- 上行理由：它正面对齐最终 kNN readout，且 sufficient-view key 可能降低 hateful video 中 benign segment dilution。
- 下行理由：P11 已显示 density teacher 对同算子 video-label MIL 没有显著优势；query full view 与 evidence-selected memory key 可能产生 train/test geometry mismatch；CGO 已在同域做更直接的 gradient control。

### 3.4 可证伪预测

在任何 full training 前，A 必须同时满足：

1. train-only teacher evidence key 的 leave-one-out kNN label purity / positive rank 显著优于 whole-video key、random segment key 与 video-label MIL key；至少两个数据集同号。
2. 控制 gold label、原 cosine similarity 与 query difficulty 后，teacher sufficient-view agreement 仍显著预测 baseline wrong-neighbour attraction。
3. prompt paraphrase / repeated parse 下 relative intervention rank agreement `≥0.75`；绝对 verdict 好坏不能替代该门。
4. seed-0 minimal run 中，full A 相对 `signature-shuffle` 与 `random-view` 的 dev acc、macro-F1 均至少 `+1.0pt`，且 **kNN readout 本身**提升。

### 3.5 主要风险

- **信号不新：** 只复用 P6/P10 density 会被 P11 的负结果预先削弱。
- **方法不新：** 只做 loss reweighting 会被 CGO 覆盖。
- **几何错配：** 训练 memory key 变“干净”，test query 仍包含全视频噪声，反而检索不到正确邻居。
- **机制不可辨识：** memory key、loss、view selection 一次全开，无法判断收益来自哪里。

## 4. 候选 B：SSR-MemRGCL

### 4.1 精确定义

**Gold-Signed Semantic Relation Memory Graph for RGCL**：先从 baseline train-memory hard neighbours 与 hard errors 中取真实训练 pair；MLLM 只输出受限关系类型，例如 `target match`、`stance ∈ {endorse, quote, condemn, satire}`、`harm mechanism` 与 `evidence binding`。**边的正负号只由 gold label 与预注册模板决定，绝不由 MLLM verdict 决定。**

建议只保留三类高价值 typed edges：

- `counter-stance negative`：target/proposition 相近，但 stance 相反且 gold label 不同；
- `mechanism-invariant positive`：gold label 相同、harm mechanism 相同，但 surface/target/domain 不同；
- `topic-confound negative`：topic/lexical surface 相近，但 target 或 endorsed proposition 不同且 gold label 不同。

这些边通过 relation-specific margin / small relation adapter 进入 memory-SupNCA 或 listwise RGCL，直接改变训练 embedding 与最终 train-memory kNN topology。测试时无 MLLM rerank、无 schema concat、无额外 score channel。

### 4.2 Novelty：6.5/10

**最近工作与威胁：**

- [Multimodal Representation Learning Conditioned on Semantic Relations / RCML (2025–2026)](https://arxiv.org/abs/2508.17497) 已构建 many-to-many semantic relations，并以 relation-guided cross-attention 和 inter/intra-modal contrastive losses学习表示。一般意义上的 relation-conditioned multimodal representation 已被占据。
- [HateSieve (Findings NAACL 2025)](https://aclanthology.org/2025.findings-naacl.289/) 已在 hateful-meme 域用生成式 context-correlated triplets做 contrastive learning。一般意义上的“LLM/VLM 语义 hard triplet 改善 hate detection”不可主张。
- [RGCL (ACL 2024)](https://aclanthology.org/2024.acl-long.291/) 与 [RA-HMD / LMM-RGCL (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1215/) 已覆盖 retrieval-guided hard pairs 与 LMM contrastive adaptation。
- [CCLRec (ICML 2026 OpenReview entry)](https://openreview.net/forum?id=C2i7ciwxKQ) 已用 LLM semantic reasoning 与图结构共识选择 contrastive positives；跨领域 reviewer 会据此攻击“LLM graph supervision”的一般新颖性。
- [T-MAD (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.30/) 与 CASE 2025 系列工作已把 target/stance 作为 multimodal 学习对象；target 和 stance 字段本身并不新。

**仍可守的核心差异：** B 不是让 MLLM 生成样本、决定 label、选择 test 邻居或提供单样本字段，而是：

> gold label fixes polarity; MLLM supplies only the relation type; the typed edge is mined from real hard-neighbour pairs and directly changes the geometry used by the final kNN memory classifier.

“gold-signed + stance-target-mechanism typed + real hard pair + final memory geometry”是目前最窄、也最可辩护的 delta。少一个限定，都会靠近 RCML、HateSieve、P2 或 P4。

### 4.3 性能 EV

- plausible mean effect（若 gate 通过）：MHC-EN acc `+1.5–3.5pt`、macro-F1 `+2.0–4.0pt`；MHC-ZH 两指标 `+1.0–3.0pt`。
- 达到完整两数据集、三 seed、双指标都 `≥+3pt` 的 ex-ante 概率：**15–20%**，三者最高但仍低。
- 上行理由：P2 oracle membership edit 显示 memory topology 有显著 headroom；B 不让 MLLM 猜正确性，而用 gold sign把其职责压缩到更擅长的 relation extraction；typed relation 是 binary label 不能唯一恢复的跨样本信息。
- 下行理由：关系覆盖可能过稀；MLLM stance extraction 可能不稳定；关系类型可能在控制 gold label与 embedding similarity 后不再提供条件信息；P4 已警示 schema semantics 很容易与 label supervision 冗余。

### 4.4 可证伪预测

B 的 Gate 1 必须先做 train/validation-only pair audit，不能直接 full train：

1. 对 200–300 个 baseline hard-neighbour pairs，双向 pair order + prompt paraphrase 的 relation agreement `≥0.80`。
2. 人工盲审 typed edge precision `≥0.80`，且 eligible typed edges 覆盖 `≥15%` hard pairs、`≥20%` baseline train/validation errors；否则影响面不足以产生 +3pt。
3. 在 logistic / conditional analysis 中，控制 pair gold-label relation、cosine similarity、query margin 和 dataset 后，typed relation 对 wrong-neighbour attraction 有显著增量；必须明显优于 P2 generic comparability。
4. seed-0 minimal run：`typed graph` 同时赢 `label-only SupNCA` 与 `edge-type shuffle`，dev acc 和 macro-F1 均 `≥+1.0pt`；增益必须出现在 kNN readout，而不只是 native head。
5. full run 前冻结 relation vocabulary、edge generator、loss 与 seeds；禁止从 test error反推 relation type。

### 4.5 主要风险

- **RCML overlap：** 若 relation-specific margin 被简化成 natural-language relation embedding + cross-attention，novelty 会显著下降。
- **P4 redux：** 若只把 stance/target 当 per-sample auxiliary labels，B 直接退化成已失败路线。
- **pair-selection confound：** 改善可能来自选择更难 pair，而非 MLLM relation；必须有 difficulty-matched random hard-pair control。
- **覆盖不足：** quote/condemn/satire 类虽然重要，但可能占比不足以移动 overall acc 三点。
- **跨语言不稳：** EN/ZH relation ontology 与讽刺/引用标记可能不等价，两个数据集共同过线并不容易。

## 5. 候选 C：VCRouter-RGCL

### 5.1 精确定义

**Verified Clause-Conditioned Low-Rank Router for RGCL**：MLLM 从 train-only examples 中抽取受限原子谓词与 evidence binding，并提出少量可执行 clause，例如 `protected_target ∧ derogation ∧ endorse`、`quote_or_condemn ∧ ¬endorse`。只有经过 train-only support/precision/counterexample verifier 的 clause 才能保留。学生预测 predicates，clause activation 路由 3–4 个小型 low-rank fusion adapters；RGCL 与 classification loss训练最终 embedding。

这已合并 “Schema-Conditioned Modality Router” 与 “可执行规则→低秩专家路由”。两者不能再作为两个独立 novelty candidates，因为 schema router 只有在 schema 被组合成可执行 clause 并改变参数路径时才与 P4 有区别。

### 5.2 Novelty：3.0/10

**最近工作与直接威胁：**

- [BPDMoE-Hate (ACL 2026 Main)](https://aclanthology.org/2026.acl-long.480/) 已在 harmful-meme 域使用 VLM 生成 harmful/non-harmful viewpoints、adaptive viewpoint gating 与 dual-space MoE。它不再只是 pending submission，而是正式 ACL 2026 prior art；这对 C 是最严重威胁。
- [Mixture of Concept Bottleneck Experts (2026)](https://arxiv.org/abs/2602.02886) 已将 concept bottleneck 扩展为多 expert，并支持 symbolic expression experts；“可解释概念组合 + experts”已高度拥挤。
- [Graph-Integrated Multimodal Concept Bottleneck Model / MoE-SGT (2025)](https://arxiv.org/abs/2510.00701) 已以结构化 concept relations + graph transformer + dynamic expert selection处理多模态概念推理。
- [SyRHM (ACL ARR 2026)](https://openreview.net/forum?id=6tKlqJaBAQ) 与 [MemOracle (ICLR 2026 submission)](https://openreview.net/forum?id=0PYmHSugcj) 已在 harmful-content domain 结合 symbolic reasoning 与 retrieval。
- [TAM (ICML 2026 OpenReview entry)](https://openreview.net/forum?id=nVBt6ifvl0) 已用显式 task induction进行 video MoE routing；低秩/任务条件路由本身不是新机制。

**仅存差异：** verified hate clauses 不直接输出 label，而是路由 RGCL low-rank fusion adapters，并要求 clause intervention 对 kNN margin 产生预测性变化。这个 delta 是模块组合级空白，不是一个宽的新范式。Reviewer 很可能把它评价为 BPDMoE-Hate + concept bottleneck expert + LoRA router 的拼装。

### 5.3 性能 EV

- plausible mean effect：两指标 `0–2.5pt`，方差很大；容量增加可能偶然更高，但不能归因给 MLLM。
- 达到完整终局概率：**3–7%**。
- 上行理由：若 quotation/endorsement/target composition 真是主要错误源，条件子空间可能优于单一 head。
- 下行理由：predicate 可能泄漏 gold label；专家易 collapse；小数据下 route 不稳定；更大容量本身是强 confound；已有 BPDMoE-Hate 已提供更直接的 domain-specific competitor。

### 5.4 可证伪预测

1. 每条 clause train-only support `≥10%`、precision `≥0.80`，全部 clauses 合计覆盖 `≥25%` baseline errors。
2. predicate consistency `≥0.80`，且从 predicate 直接预测 gold label的性能不能证明机制；关键是 clause/router intervention 是否额外改变正确 kNN margin。
3. `verified clause router` 必须赢容量匹配的 learned router、uniform router、rule shuffle 与 independent-field auxiliary loss。
4. 每个 expert usage 非退化；若一个 expert 占比 >80%，视为 route collapse。
5. seed-0 minimal run 的 kNN acc 与 macro-F1 对 uniform router 均至少 `+1.0pt`，否则停止。

### 5.5 主要风险

- novelty 已被 2026 prior art 严重挤压；
- “规则”可能只是由 gold labels 反向总结出的 dataset-specific shortcut；
- router 的任何提升都难与额外参数/容量分离；
- 实现自由度大，极易形成事后调参和不可证伪的模块堆叠；
- 与最终 kNN memory geometry 的联系比 A/B 弱。

## 6. 首跑 B 的最小科学路线

本 Gate 0 只批准 **B 的信息价值审计**，不批准直接做大训练或把 A+B 合并。

### Stage B0：冻结 ontology 与 pair universe

- 数据只用 MHC-EN / MHC-ZH train 与 validation；test 不可见。
- pair universe 固定为 baseline top-k neighbours + baseline hard errors；不全局枚举，不靠更多数据堆规模。
- relation ontology 固定为 `counter-stance`、`mechanism-invariant`、`topic-confound` 三类；`unclear` 一律不建边。
- MLLM 输出 JSON 受限字段；模型规模固定，不做 7B/32B/72B sweep。

### Stage B1：先证 conditional information gain

在 200–300 个 pair 上同时报告 agreement、人工 precision、coverage、error enrichment，以及控制 label/similarity/margin 后的增量预测。**如果 typed relation 不优于 generic comparability，B 就是 P2 换名，必须停止。**

### Stage B2：最小机制训练

只比较四臂、同 backbone / data / epoch / seed：

1. exact RGCL baseline；
2. label-only memory-SupNCA；
3. B full typed graph；
4. B edge-type shuffle（同边数、同 label polarity、同难度分布）。

full B 必须同时赢 2 与 4，且提升出现在最终 kNN acc / macro-F1。若只有 head 赢、kNN 不赢，则与目标错位。

### Stage B3：只有 fast-fail 通过后才进入固定终局协议

- paired seeds 0/1/2；
- 至少 MHC-EN 与 MHC-ZH；
- 相对同 seed 最强 non-MLLM baseline，acc 和 macro-F1 各 `≥+0.030`；
- 3/3 paired deltas positive；hierarchical paired bootstrap 下界 >0；四个 dataset×metric tests Holm corrected；
- remove-MLLM、edge shuffle、difficulty-matched random hard-pair controls全部报告。

## 7. 最终 reviewer recommendation

**PROCEED WITH CAUTION，仅批准 B 的 Gate 1 fast-fail。**

- **A** 在 architecture alignment 上很强，但现有 P3/P11 负结果与同域 CGO 使其 EV/novelty 低于 B；保留为 B 失败后的第二路线，且必须是 counterfactual evidence-directed memory key，不得是 density pooling。
- **B** 是唯一首跑：它最直接利用最终 kNN memory，又将 MLLM 限制在 gold label不能提供的跨样本 relation typing；novelty 仍只有 6.5/10，必须用严格限定守住。
- **C** 不建议先做：ACL 2026 BPDMoE-Hate、M-CBE、MoE-SGT 已让 viewpoint/concept/rule-conditioned experts 过于拥挤；即使涨分，也很难证明不是容量与模块拼装。

最关键的否定性判断是：**不要再尝试 simple concat、score fusion、test-time judge、规模升级或“把三个候选全开”的大系统。** 当前真正值得验证的科学命题只有一个：

> 在控制 gold label、embedding similarity 与 pair difficulty 后，MLLM 提取的 stance–target–mechanism relation type 是否仍能识别并修复 RGCL train-memory 中的 wrong-neighbour attraction；若能，gold-signed typed edges 是否会因果性地改善最终 kNN acc 与 macro-F1。

若这个命题在 train/validation pair audit 中不成立，不应靠调 K、margin、边数、模型大小或更多 epoch 救活 B。

## 8. 检索范围与诚实声明

- 复核覆盖 2024–2026 ACL Anthology、OpenReview 与 arXiv 一手页面，并特别补查了 2026-01-10 至 2026-07-10 的近六个月工作。
- OpenReview submission 与正式 accepted paper 在文中明确区分；其中 BPDMoE-Hate 已更新为 ACL 2026 Main，RAMF 已更新为 TMLR 2026。
- 查新不是数学证明；匿名 ARR、索引延迟或标题变化稿仍可能遗漏。正式投稿前必须用同一 query pack 重扫。
- 外部工作的增益来自不同 split / backbone / protocol，不能用于声称本项目可达到相同数字；本报告的 EV 只用于路线排序。
