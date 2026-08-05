# TARGET REVIEW RAW

This file stores verbatim independent external-review outputs for the target-driven loop.

## Iteration 0

No external reviewer was invoked during state initialization. Gate 3 is pending; no reviewer verdict may be inferred from the internal campaign documents.

## Iteration 0 · Gate 0 independent novelty/EV review · 2026-07-10

The following is the complete verbatim reviewer output; formatting is preserved.

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

# Gate 0 Iteration 3 独立 adversarial novelty review

**冻结日期：** 2026-07-10（Pacific/Auckland）  
**审阅范围：** 本地权威产物、旧 Gate 0 报告、2024–2026 一手论文与官方会议信息；重点补查 2026-01-10 至 2026-07-10。  
**操作边界：** 只读；未修改文件、未提交作业、未调用 MLLM。  
**成功标准：** train-only MLLM 必须以可移除、机制级 privileged teacher 进入方法，并在 MHC-EN/MHC-ZH、paired seeds 0/1/2 上，相对移动的最强 non-MLLM RGCL baseline，使 final accuracy 与 macro-F1 各提升至少 `+0.030` absolute。

## 1. Executive verdict

原文学候选与代码候选存在明显机制重叠，不能作为六条路线并行推进。独立审阅后应合并为恰好三条：

| 最终候选 | 吸收/替代关系 | Novelty | 判断 | 顺序 |
|---|---|---:|---|---:|
| **A. CTE-RGCL：Counterfactual Tangent Evidence Geometry** | 吸收 RSDG 的 teacher-residual-geometry 主张；删除 diffusion/GW 大系统 | **5.5/10** | **PROCEED，唯一首跑** | 1 |
| **B. SQ-RGCL：Semantic Quotient RGCL** | 与 PSD 合并；保留 class-conditional nuisance quotient | **4.5/10** | **REVISE，第二储备** | 2 |
| **C. ECM-RGCL：Executable Constraint Modes** | 吸收 ESEC 的 executable-constraint 意图；弃用 rule/predicate energy，改为 OOF failure-mode constrained optimization | **4.0/10** | **REVISE，第三储备** | 3 |

唯一首跑是 **CTE-RGCL 的 strict-OOF zero-teacher capacity screen，随后才是最多 128 个 train video/库的 MLLM teacher-value pilot**。不批准三条并行实现，不批准先调用完整 teacher，不批准把 A+B+C 合成大系统。

原始 ESEC-RGCL 的 fuzzy-rule/predicate-energy 实现应视为 **ABANDON**；只有改写后的 ECM failure-mode constrained-gradient 版本可作为第三储备。

## 2. 监督来源红线

本项目没有片段级金标注。

本地权威证据明确写明：

- `artifacts/ssr/v1/b1/preflight_oracle_upper_bound.json`：`only_gold="video_level_binary_label"`，`segment_gold_used=false`；
- `artifacts/edcm/v1/a0/MHC/metrics.json`：`segment_gold_exists=false`，`segment_gold_used=false`；
- `artifacts/edcm/v1/a0/MHC_zh/metrics.json`：`segment_gold_exists=false`，`segment_gold_used=false`；
- 两个 EDCM A0 的 teacher/MLLM artifact count 均为零。

因此：

1. 均匀采样 frame、自动 subclip、ASR/OCR 都只是输入；
2. MLLM 的 target、stance、mechanism、environment、failure mode、constraint 或时间描述都是 weak/privileged pseudo-signal；
3. 不能把父视频标签广播后称为 segment label；
4. 不能写 segment annotation、span gold、oracle localization 或 dense supervision；
5. 外部 IARE 的 Ex-HateMM/Ex-ImpliHateVid 含 fine-grained harmful-element annotations，但这些标注不属于本项目，不能被当成本地可用监督。[IARE/arXiv](https://arxiv.org/abs/2606.11953)

最终三候选均只允许 whole-video teacher records；不得出现 segment loss、segment weighting 或 segment-level endpoint。

## 3. 本地 oracle / cost-screen 解释

### 3.1 SSR

SSR 的乐观预检结果为：

| 数据集 / family | 触及错例 | 乐观 Δacc | 乐观 ΔmF1 |
|---|---:|---:|---:|
| MHC / MI | 2 | +0.0036 | +0.0048 |
| MHC / SC | 7 | +0.0128 | +0.0176 |
| MHC-ZH / MI | 3 | +0.0052 | +0.0065 |
| MHC-ZH / SC | 15 | +0.0259 | +0.0307 |

这足以否定 SSR 注册的单邻居 candidate-arc universe。它不是所有表示学习方法的理论上界。

### 3.2 EDCM

| 数据集 | support | reachable errors | 乐观 Δacc | 乐观 ΔmF1 |
|---|---:|---:|---:|---:|
| MHC | 202/549 | 15 | +0.0273 | +0.0394 |
| MHC-ZH | 364/579 | 22 | +0.0380 | +0.0444 |

这些数字只覆盖：

- 冻结旧表示；
- 原 top-64 candidate pool；
- 每个 query 至多两次 swap；
- 不学习新 encoder geometry。

因此 EDCM A0 是 **cost/reachability screen，不是理论上界**。凡是共同移动 query 与 memory、允许旧 top-64 外样本成为邻居的方法，在 action-space 意义上都超出该 universe。

但“超出旧 universe”只代表没有被旧算术上界排除，不代表能达到 `+0.030/+0.030`。

## 4. 近六个月最重要的 novelty 压力

### 4.1 直接任务竞争

- [HVG​​uard, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.456/) 已覆盖 MLLM CoT + MoE hateful-video classifier。
- [RAMF, TMLR 2026](https://openreview.net/forum?id=U9KnNiuMu1) 已使用 objective / hate-assumed / non-hate-assumed reasoning 和语义融合。
- [DR-HM, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.2130/) 已覆盖 cognition-aware reasoning synthesis、两阶段 SFT 与 A-GRPO。
- [ExPO-HM, ICLR 2026](https://iclr.cc/virtual/2026/poster/10008633) 已覆盖 explain-then-detect policy optimization。
- [BPDMoE-Hate, ACL 2026](https://aclanthology.org/2026.acl-long.480/) 已覆盖 VLM adversarial viewpoints、adaptive gating 和双空间 MoE。

普通 reasoning feature、viewpoint fusion、MoE/router、rationale distillation 均不能再作为 novelty 主张。

### 4.2 A 的压力

- [TextTeacher, TMLR 2026](https://openreview.net/forum?id=Xwb0aEUwKh) 已证明训练时 semantic anchors 可塑造表示、推理时删除语言 teacher，并报告最高约 `+2.7` accuracy。
- [Geometric Knowledge Distillation / Neural Heat Kernel](https://openreview.net/forum?id=7WGNT3MHyBm) 已使用 heat kernel 对齐 teacher/student geometry。
- [EmbedDistill](https://openreview.net/forum?id=-aEuKX6zQKmr) 已直接蒸馏 retrieval embedding geometry。
- [UCMKD, ICML 2026](https://openreview.net/forum?id=z9aetU7Wfl) 已覆盖跨模态 distribution-level feature/label alignment。
- [Geometry-aware representational alignment](https://openreview.net/forum?id=Yzr27JSBiV) 已覆盖 Gram/Procrustes KD。
- [OptiMAG](https://arxiv.org/abs/2601.22856) 已使用 Fused Gromov-Wasserstein 做结构—语义对齐。

所以“semantic teacher 改 geometry”“heat kernel distillation”“OT/GW alignment”“inference-free language teacher”均不新。

### 4.3 B 的压力

- [Mitigating Spurious Correlations in Multi-modal Models during Fine-tuning, ICML 2023](https://proceedings.mlr.press/v202/yang23j/yang23j.pdf) 已自动发现语言属性，并用对比损失 decorrelate spurious attributes 与 class representation。
- [CDAL, ICML 2026](https://openreview.net/forum?id=wtsM3MPn2P) 已使用 semantic/sensitive 双子空间、正交约束和 HSIC。
- [CARE, ICLR 2026 workshop](https://openreview.net/forum?id=KzH1cLVU1R) 已分解 invariant 与 environment-specific concept directions。
- [Rethinking Disentanglement under Dependent Factors, TMLR 2026](https://openreview.net/forum?id=PgwkNC63CS) 明确指出真实因子通常相关，独立因子假设不可靠。

所以“语言属性去偏”“双子空间 + HSIC”“environment invariant representation”均不新。

### 4.4 C 的压力

- [DARTVAE](https://arxiv.org/abs/2509.20501) 已让 LLM-generated rules 通过 consistency/violation loss 直接塑造 multimodal latent clustering。
- [Formal Concept Lattices, ICML 2026](https://openreview.net/forum?id=AE8xCfWWL9) 已让结构化概念 scaffold 指导神经表示。
- [Energy-Based Constraint Networks](https://openreview.net/forum?id=gl6l8nTXBB) 已学习跨模态结构一致性能量。
- [DiLA](https://openreview.net/forum?id=uh3ZO2izyr) 与 [LogicFlow](https://openreview.net/forum?id=5XzVX6MXCp) 已覆盖 differentiable logic refinement。
- [The Lattice Representation Hypothesis, ICLR 2026](https://openreview.net/forum?id=5K1FG92m5s) 已把概念格与 embedding geometry 联系起来。

原 ESEC 的“MLLM rules → differentiable energy → latent correction”已经接近现有模块组合，无法维持 5.5/10 novelty。

## 5. 候选 A：CTE-RGCL

### 5.1 合并后的精确定义

冻结 MLLM 只处理 train video，并比较同一视频的三个 whole-video 条件：

- full；
- visual-neutralized；
- language-neutralized。

MLLM 不输出最终 hate verdict，只输出受限弱序：

- `preserve`；
- `weaken`；
- `reverse`；
- `unknown`。

弱序用于约束学生的 full-bank true-class retrieval margin：沿 deterministic counterfactual representation 的切向变化，应与 teacher 给出的 preserve/weaken/reverse 方向一致。训练改变 encoder/fusion representation；测试仍是普通 full-video embedding 和原 train-memory kNN。

该定义吸收 RSDG 中“teacher 提供 label/base geometry 之外的残差信息，并直接塑造 final kNN geometry”的核心，但删除：

- 全局 semantic kernel；
- heat-kernel matching；
- GW/OT；
- residual-kernel PSD 处理；
- 多套几何 loss。

### 5.2 为什么优于原 RSDG

原 RSDG 的 `K_sem^perp` 有未解决的数学问题：对 kernel 做 label/base regression 后，残差一般不再保证 positive semidefinite；将其直接当作 heat kernel target 可能不可实现。若再做 PSD projection，又可能重新引入已 residualize 的方向。

CTE 用可验证的局部弱序约束替代全局 residual kernel：

- teacher 信息更窄；
- 不需要把自由文本 embedding 当几何真值；
- 不增加新参数；
- 机制归因更直接；
- 与 final retrieval margin 的连接更清楚。

### 5.3 Closest prior

最接近的不是单篇，而是以下交叉：

- inference-free semantic supervision：[TextTeacher](https://openreview.net/forum?id=Xwb0aEUwKh)；
- retrieval geometry distillation：[EmbedDistill](https://openreview.net/forum?id=-aEuKX6zQKmr)；
- modality intervention/gradient control：旧 Gate 0 所列 [CGO](https://openreview.net/forum?id=Z51RWOPKQQ)；
- same-domain counter-reasoning：[RAMF](https://openreview.net/forum?id=U9KnNiuMu1)。

可守的 novelty 仅是：

> A train-only MLLM supplies ordinal whole-video modality-counterfactual evidence; that ordinal signal constrains the tangent response of the exact full-bank RGCL retrieval margin, while inference remains the unchanged full-video kNN classifier.

不能声称首次 counterfactual supervision、首次 semantic KD、首次 retrieval geometry distillation 或首次 gradient control。

### 5.4 硬条件审计

- 超越 top-64/2-swap universe：**是，结构上超越。** Encoder 与整库表示共同移动，可产生旧 top-64 外邻居。
- Meaningful train-only MLLM：**有条件成立。** `preserve/weaken/reverse` 必须胜 label-only、modality-energy heuristic 和 shuffle。
- 直接 final acc/mF1：**是，设计上直接。** Loss 作用于 full-bank true-class retrieval margin。
- Segment gold：**不使用。**
- Simple concat：**不使用。**
- Score fusion/test rerank：**不使用。**
- Segment weighting：**不使用。**
- Teacher-selected keys：**不使用。**
- BPDMoE low-rank router：**不使用。**

### 5.5 最大风险

1. neutralization intervention 未必语义有效，可能只造成 OOD artifact；
2. MLLM 弱序可能主要由输入缺失程度推断，而非理解 hate mechanism；
3. preserve/weaken/reverse 对多数样本可能退化为同一答案；
4. zero-parameter ordinal loss 仍可能被普通 label contrastive loss完全解释；
5. 若只改变 native head、不改变 kNN，则机制失败；
6. 若 label-only CTE-style loss 已大幅提升，它必须成为新的移动 non-MLLM baseline。

### 5.6 必要修订

1. neutralization 算子必须预注册并保持确定性；禁止按 MLLM 反馈反复设计；
2. teacher 不得见 segment、timestamp 或 test record；
3. strict OOF 时 outer-query 不得使用 teacher record、outer label 或 counterfactual loss；
4. 首先运行 zero-teacher capacity screen，证明该 action family 能产生 dense neighbor churn 和真实 OOF correction；
5. “非零梯度”门不能只检查浮点非零，必须相对 capacity-matched random/order control 有实质梯度与 correction effect；
6. teacher pilot 必须比较：
   - FULL；
   - REMOVE-MLLM；
   - order shuffle；
   - label-only ordinal proxy；
   - simple modality-energy heuristic；
   - strength-matched random orders；
7. [GRACE](https://openreview.net/forum?id=m276fke38H) 表明 directional coverage 单独并不足以判断 teacher 价值，因此不能只用 gradient effective rank 解锁 full run。

### 5.7 Verdict

**PROCEED，但只批准 CTE strict-OOF cost screen → 小规模 teacher-value pilot。**

不批准恢复 RSDG 的 diffusion/GW 组件，也不批准直接 full multi-seed run。

## 6. 候选 B：SQ-RGCL

### 6.1 合并后的精确定义

MLLM 为 train-only whole video 输出 presentation/context nuisance 的软环境分布。最终表示学习 class-conditional quotient：

- 同一 gold class、不同 nuisance environment 的样本拉近；
- 不同 gold class 即使处于同一 environment 也保持分离；
- 测试只使用 quotient representation 和普通 kNN。

它吸收 PSD 的 nuisance removal，但必须明确区分：

- 候选 nuisance：presentation style、reportage format、surface topic、speaker/layout/context；
- 不应自动删除：speaker endorsement、harm act、target evidence、cross-modal binding。

### 6.2 Closest prior

- [Yang et al., ICML 2023](https://proceedings.mlr.press/v202/yang23j/yang23j.pdf)：语言发现 spurious attributes，并直接 decorrelate multimodal representation；
- [CDAL, ICML 2026](https://openreview.net/forum?id=wtsM3MPn2P)：semantic/sensitive subspace + HSIC；
- [CARE](https://openreview.net/forum?id=KzH1cLVU1R)：invariant/environment-specific directions；
- [Dependent-factor disentanglement](https://openreview.net/forum?id=PgwkNC63CS)：相关因子下的 minimal/sufficient 表述。

可守差异很窄：

> MLLM supplies train-only whole-video soft nuisance environments for a class-conditional quotient of the exact RGCL memory space, and the teacher/environment information is absent at inference.

它不能被描述为 causal deconfounding；现有数据不足以识别因果 nuisance。

### 6.3 硬条件审计

- 超越旧 reachability universe：**是，结构上超越。**
- Meaningful train-only MLLM：**有条件。** 必须胜 label-blind cluster、cheap caption attributes 与 shuffle。
- 直接 final acc/mF1：**是，设计上直接。**
- Segment gold：**不使用。**
- concat/fusion/rerank/key selection/router：**均不使用。**

### 6.4 最大风险

1. 话题、target 与 stance 强相关，误删会擦除真实 label signal；
2. MLLM environment 可能只是已有 embedding cluster 的自然语言重述；
3. “同类跨环境拉近”可能牺牲真实的 hateful subclass geometry；
4. Yang 2023 与 CDAL 已覆盖大部分算法骨架；
5. 改进可能来自 quotient/HSIC 正则，而非 MLLM；
6. 小数据下软 environment 容易 collapse 或与 label近乎同构。

### 6.5 必要修订

1. 将名称从 causal deconfounding 收缩为 semantic quotient 或 conditional nuisance invariance；
2. 只允许 presentation/context nuisance 进入 quotient；
3. stance、harm act、endorsement、evidence binding 不得全部监督 `z_nuis`；
4. A0 的 label-blind base environment 若提升，必须成为移动 baseline；
5. 必须增加 cheap language-attribute detector control，以区分 MLLM world knowledge 与普通 caption/embedding clustering；
6. FULL 必须同时胜：
   - REMOVE；
   - base cluster；
   - field shuffle；
   - posterior noise；
   - Yang-style language attribute regularization；
   - P4-style auxiliary prediction。

### 6.6 Verdict

**REVISE。** 保留为 CTE 失败后的第二路线，不批准当前 PSD 表述直接实施。

## 7. 候选 C：ECM-RGCL

### 7.1 为什么原 ESEC 应废弃

原 ESEC 依赖：

- MLLM 归纳 fuzzy rules；
- student predicate heads；
- differentiable rule energy；
- 一至两步 latent proximal/gradient correction。

其核心已与 DARTVAE、concept lattice、energy constraint network 和 differentiable logic 工作高度重叠。更严重的是，原稿未清楚定义测试时在未知 gold label 下如何选择有方向的 energy correction；如果 rule energy 实际编码 hate conclusion，它很容易成为隐式 label shortcut。

因此原 ESEC：**ABANDON。**

### 7.2 仅保留的 ECM 改写

MLLM 读取 strict-OOF train residual records，为 whole video 给出受限 failure-mode posterior，例如：

- cross-modal binding failure；
- stance inversion；
- context/reportage confusion；
- target attribution failure；
- modality dominance；
- unknown。

MLLM 不生成规则、不预测最终 verdict、不产生 test-time predicate。

训练阶段把 failure modes 作为 mode-level constraints：

- constrained gradient projection，避免改善某 mode 时伤害另一 mode；
- 或 capacity-matched minimax curriculum，提高 worst-mode retrieval margin。

测试时没有 teacher、mode、rule、predicate 或额外 head；仍是普通 full-video embedding 与 kNN。

### 7.3 与 SQ 的边界

必须冻结以下区分：

- **SQ：** label-blind presentation/context nuisance environment，目标是 quotient invariance；
- **ECM：** 从 strict-OOF residual 中定义的 label-conditional failure mechanism，目标是 worst-mode optimization 和 gradient conflict control。

若 ECM modes 退化成 topic/style environment，它应并入 SQ，不得作为第三 novelty candidate。

### 7.4 Closest prior

- failure slice/environment discovery 和 GroupDRO 类方法；
- [CGO](https://openreview.net/forum?id=Z51RWOPKQQ) 的 harmful-video gradient control；
- [GRACE](https://openreview.net/forum?id=m276fke38H) 的 gradient-distribution teacher value；
- [CDAL](https://openreview.net/forum?id=wtsM3MPn2P) 与 invariant learning 的环境约束。

可守差异只可能是：

> Strict-OOF MLLM residual diagnosis defines train-only semantic failure modes, which enter capacity-matched constrained optimization of the final RGCL retrieval geometry; no mode information exists at inference.

### 7.5 硬条件审计

- 超越旧 reachability universe：**是，结构上超越。**
- Meaningful train-only MLLM：**有条件，且归因困难。**
- 直接 final acc/mF1：**是，若 constraint 直接作用于 retrieval loss。**
- Segment gold：**不使用。**
- rule energy/MoE/router/test rerank：**改写后均不使用。**

### 7.6 最大风险

1. MLLM modes 只是对 OOF error 和 gold label 的自然语言复述；
2. failure-mode curriculum 可能等价于普通 hard-example mining；
3. constrained gradient projection 的收益可能完全与语义 mode 无关；
4. SQ 与 ECM 的 environment 定义容易重叠；
5. MLLM 若看到预测、label 和完整 residual record，极易构造 label shortcut；
6. 当前 novelty 主要来自任务与 endpoint 组合，而非新优化原理。

### 7.7 必要修订

1. mode ontology 必须在 teacher pilot 前冻结；
2. mode induction 只能使用 inner-train strict-OOF residual，不能用 validation/test errors；
3. 必须提供：
   - random mode；
   - loss-only difficulty bins；
   - embedding cluster；
   - label×margin bins；
   - mode shuffle；
   - capacity-matched vanilla GroupDRO/minimax；
4. failure mode 不得包含时间 span、segment ID 或关键片段；
5. FULL 必须胜上述 cheap residual partitions，否则 MLLM 无不可替代性。

### 7.8 Verdict

**REVISE，第三储备。** 只有在 CTE 与 SQ 均 fast-fail 后才值得进入 teacher pilot。

## 8. 最终排序与唯一首跑

### 8.1 排序

1. **CTE-RGCL**
2. **SQ-RGCL**
3. **ECM-RGCL**

### 8.2 唯一首跑

只批准以下顺序：

1. **CTE-0：strict-OOF zero-teacher capacity screen**
   - 不调用 MLLM；
   - 验证 full-bank tangent/order constraint family 是否能产生 dense、稳定且真实的 OOF correction；
   - 不能把它写成理论上界；
   - 若 label-only 版本成为更强 non-MLLM 方法，它立即更新移动 baseline。

2. **CTE-1：最多 128 train videos/库的 teacher-value pilot**
   - whole-video full/neutralized inputs；
   - 双 prompt、双顺序；
   - unknown/fallback 明示；
   - 检查 order agreement、conditional information、gradient/value 增量；
   - outer query 不使用 teacher。

3. **CTE-2：仅在 CTE-0/1 均通过后做 seed-0**
   - FULL 同时胜 REMOVE、shuffle、label-only order、heuristic order 和 random order；
   - 两库 dev kNN accuracy/macro-F1 各至少 `+0.010`。

只有 seed-0 通过，才允许 paired seeds 0/1/2 和最终 `+0.030/+0.030` 检验。

## 9. 统一自动否决条件

出现以下任一项，应停止并将 novelty 降至 `<=3/10`：

- rationale/schema/score/summary embedding concat；
- MLLM score fusion、veto、rerank 或 test-time arbitration；
- segment weighting、segment pseudo-gold 或 localization 代替 classification；
- teacher-selected/replaced memory keys；
- 回到 SSR 单邻居 universe 或 EDCM top-64/two-swap universe 调参；
- 更大 teacher、更多 frame、更多 epoch、更多数据或 ensemble 作为主要变量；
- BPDMoE 式 viewpoint gate、MoE 或 low-rank router；
- 只提升 native head；
- 单 seed、单数据集、只提升一个指标；
- FULL 不胜 REMOVE 与 SHUFFLE；
- 把 zero-teacher A0 写成理论上界或 MLLM 成功证据。

## 10. 最终 reviewer recommendation

**唯一批准路线：CTE-RGCL，且只批准 staged fast-fail。**

理由不是 CTE 已证明能够涨三点，而是：

1. 它在 action-space 上真正离开冻结 top-64/two-swap universe；
2. MLLM 的职责被压缩为 gold label不能直接提供的 whole-video counterfactual ordinal relation；
3. loss 直接作用于最终 full-bank retrieval margin；
4. 不需要 segment gold；
5. 相比 RSDG，它去掉了不必要且已有强 prior 的 diffusion/GW/heat-kernel模块；
6. 相比 SQ，它更容易证明 MLLM 信息不是普通环境聚类；
7. 相比 ECM，它更容易区分语义 teacher 贡献与通用 robust optimization。

但 CTE 仍只是 **最值得先证伪的候选，不是已达到目标的方法**。当前不存在任何新结果证明最终 accuracy/macro-F1 已 substantial 提升，目标不能关闭。

## Iteration 4 · SQ-RGCL independent method refinement record · 2026-07-11

- **Canonical reviewer agent id:** `/root/sq_reviewer_replacement`
- **Discarded interrupted reviewer:** `/root/sq_method_refine/sq_reviewer` produced no output and is not part of the evidence.
- **Score path:** `6.88 REVISE → 7.90 REVISE → 8.46 REVISE → 9.12 READY`.
- **Final raw verdict:** READY only for experiment-plan/implementation audit; not a performance result and not project completion.
- **Anchor/drift:** problem anchor preserved verbatim; drift NONE.
- **Audits:** no-segment-gold PASS; CTE C0 interpretation PASS; method simplicity PASS; no remaining scientific blocker.
- **Full verbatim raw responses:** `refine-logs/sq/round-1-review.md`, `round-2-review.md`, `round-3-review.md`, `round-4-review.md`.
- **Full anchored revisions with anchor/simplicity checks:** `refine-logs/sq/round-1-refinement.md`, `round-2-refinement.md`, `round-3-refinement.md`.
- **Final reviewer restriction:** execute P0/SQ-0 only after independent planning/audit; SQ-0 failure is terminal and no new teacher call is allowed before SQ-0 GO.

## Iteration 6 · Independent novelty review · 2026-07-11

合并去重后保留恰好 3 条。关键修正：原 SCPT 把 gold label 给 MLLM，违反本轮“teacher label-blind”硬约束；必须改成 teacher 先生成无标签语义证书，cache 冻结后 compiler 才接入视频标签。原 RUF 不符合“先求目标 bank、再由 shared encoder 拟合”的指定主线，且太接近 function-space update projection，因此淘汰，由 RHT 替代。

### 1. LB-SCGP：Label-Blind Semantic-Certificate Gram Projection

唯一推荐首跑，合并原 SCPT 与 RGP。

- Teacher 输入：单个完整训练视频的 uniform frames、完整 ASR/OCR/title；绝不含 label、prediction、margin、error、neighbor、segment/span。
- Teacher 输出：受限 clause graph，例如 proposition、stance、quotation/condemnation/reportage exception、cross-modal binding、`supported/contradicted/unresolved`、confidence；无 verdict、score、rationale、key。
- 数学对象：当前归一化全库 `Z0` 及 `G0=Z0 Z0^T`。证书 compiler 产生全库 row-profile identities 与 exception-reflection 的 pair-of-pairs 约束 `A_sem vec(G) ~= b_sem`。标签只在 cache 冻结后用于 exact vote。
- Exact endpoint：

  `m_i(G) = c_i/210 * sum_{r=1}^{20} (21-r) c_{pi_i(r)} G_{i,pi_i(r)}`, `c_i=2y_i-1`，其中 self 排除，按 cosine 降序、ID 稳定打破平局，top-20 每个 proximal iteration 重算。

- Solver：

  `min_{G PSD, diag(G)=1} 0.5||G-G0||_F^2 + rho||A_sem vec(G)-b_sem||^2`

  subject to per-video exact-margin envelope、两类平均 margin 不下降、row/centroid trust region。采用 rank-cell sequential proximal projection；每步重排并用真实 evaluator 验证，失败回溯。因 `N<600<d=1024`，PSD factor 的 rank 不构成瓶颈。分解 `G*`、Procrustes 对齐为 `Z*`，再以 uniform `||normalize(f_theta(x_i))-stopgrad(z_i*)||^2` 拟合同一 encoder；test 不加载任何证书或 target。

- Preflight：五折 strict train-OOF 的 label-only target/fitting 必须在两库 acc、mF1 均达 `+0.05` 且每折同号；否则零 teacher 停止。随后每库最多 128 视频的证书稳定性/覆盖/conditional-value pilot，再做 seed-0。
- Controls：REMOVE、LABEL-ONLY-TARGET、CERT-SHUFFLE/NOISE、ERROR-PROPENSITY、P4-AUX、TextTeacher anchor、DARTVAE-style direct rule loss、generic pair/triplet、ECM archival mode target。
- 实现接口：`build_full_bank -> compile_certificate_constraints -> solve_gram_target -> factor_align -> fit_target_batch`；只在 epoch bank refresh 与 loss 处接入，原 evaluator 不改。
- 资源：SCGP-0 约 30–80 GPU-h；pilot 不超过 1,024 calls；完整两库四调用约 4,512 calls。
- Closest：LEAF、TextTeacher、DARTVAE、EmbedDistill/geometry KD、proof-obligation sidecars、ECM archival sketch。
- 评分：novelty `7.0/10`，feasibility `7.2/10`，达到 `+3/+3` likelihood `5.8/10`，综合 EV `6.6/10`。

### 2. LBOP：Label-Blind Lattice-Barycentric Order Projection

合并 LOP 与 SBT。

- Teacher 输出固定 moderation lattice 的置信 lower/upper set；不看 label，不输出 verdict/rationale。
- 数学对象：correlation matrix `G`、固定非数据型 lattice anchors，以及 meet/join 导出的 affine barycentric identities，例如 `u_{a meet b} ~= norm(alpha u_a + (1-alpha)u_b)`，并约束 lower/upper interval 的 isotonic order、meet/join row-profile compatibility，以及同一 exact top-20 margin envelope。
- Solver：交替执行 isotonic/order-polytope projection、nearest-correlation PSD projection和 exact-rank-cell verification；factor/Procrustes 后拟合同一 encoder。
- Preflight/controls：two-element label-only lattice OOF `+0.05/+0.05`；随后 interval coverage、四调用 lattice closure、meet/join held-out residual；对比 LABEL-ONLY-LATTICE、random/shuffled/noised intervals、caption anchors、P4 hierarchy、ordinal/triplet metric。
- 非等价性：meet/join 的 affine barycentric 坐标在保持全部 triplet distance ordering 不变的连续形变下仍可改变，因此不是 ordinal triplet loss 的函数。
- Closest：Lattice Representation Hypothesis、Polar Probe、partial-order/ordinal embedding、TextTeacher、concept bottleneck、Gram/Procrustes KD。
- 主要风险：lattice 人为、interval collapse、固定 anchors 退化为 concept bottleneck，可能擦除真实 label signal。
- 资源：40–100 GPU-h，teacher 成本与候选 1 接近。
- 评分：novelty `5.3/10`，feasibility `5.8/10`，`+3/+3` likelihood `4.0/10`，EV `4.8/10`。

### 3. RHT：Relational-Holonomy Targeting

保留 RHT，淘汰原 RUF。

- Teacher 输入：确定性、label-blind 的整视频 pair；候选 pair 由冻结 base bank 的 label-blind 规则预选，teacher 不选 key。
- 输出：有方向的语义变换类型及置信度，例如 quotation->endorsement、reportage->assertion、target-binding change、context inversion；无标签或 correctness。
- 数学对象：具有相同变换类型的 pair-of-pairs quadruple `(i,j,k,l)`。在目标球面 bank 上约束 relation vector 与 cycle holonomy：`L_hol = ||PT_{i->k} log_{u_i}u_j - log_{u_k}u_l||^2`，并要求闭环平行移动残差接近零，同时直接满足相同 exact top-20 vote-margin constraints。
- Solver：Riemannian trust-region/SQP 求最小位移 `U*`，每步 exact 重排、vote parity 与 accept/reject；随后 shared encoder 拟合 `U*`。
- Controls：同 pair 的 generic triplet、无方向 relation loss、TRANSFORM-SHUFFLE/NOISE、base-embedding transformation clusters、ERROR-PROPENSITY、LABEL-ONLY target、scalar reweight。
- 非等价性：两个配置可拥有完全相同的所有 pairwise distance order/triplet predicates，却具有不同的有向 relation-vector alignment 与 cycle holonomy，因此 generic triplet 无法表示该目标。
- Closest：relational KD、analogy/translation embedding、Polar Probe、geometry KD、TextTeacher；可守 delta 是 MLLM relation holonomy × exact-vote full-bank proximal target。
- 风险：可靠 quadruple 覆盖可能不足；pair 调用和图闭包昂贵；parallel transport 数值复杂。
- 资源：约 80–200 GPU-h，可能需要 5k–20k pair calls。
- 评分：novelty `7.3/10`，feasibility `4.4/10`，`+3/+3` likelihood `5.0/10`，EV `5.5/10`。

### 共同非重权证明

不能只报 NNLS residual。对每个最终 displacement `D*`，建立 registered ordinary RGCL per-example embedding-gradient cone 与 matched pair/triplet-gradient cone。求其投影并输出 Farkas dual separating certificate；要求两库：

- 相对投影残差 `>=0.25`；
- dual separation gap 显著大于数值容差；
- matched learned scalar-weight/triplet controls 不能满足相同 exact-vote 与语义约束，也不能匹配 OOF acc/mF1。

结构性非-triplet 证明分别来自：候选 1 的绝对 Gram/row-profile 与 rank-weighted vote magnitude、候选 2 的 affine meet/join barycentric identity、候选 3 的有向 holonomy；这些量都可在所有 triplet order 保持不变时改变。

**唯一首跑：LB-SCGP-0。** 原 gold-grounded SCPT、LOP 原版、RUF 原版均不得直接执行。全程只有父视频二分类 gold；不存在或使用任何 segment/timestamp/span/localization gold。

## Iteration 6 · LB-SCGP continuous method refinement · 2026-07-11

- Reviewer `/root/iter6_architect`, four continuous rounds: `6.74 REVISE -> 7.86 REVISE -> 8.69 REVISE -> 9.15 READY`.
- Final: anchor preserved; drift NONE; no method blocker; only parent-video binary gold; no segment/timestamp/span/localization gold; teacher label-blind/cache-before-label/test-clean PASS.
- Evolution: removed same-proposition/equivalence claims; kept one support-selected structural reflection; froze exact Dykstra/rank-cell projectors; scoped abstract/realized Farkas; added DIRECT-AEXC/STATE-MOMENT final controls; incorporated family selection in Rao-Wu inference.
- READY scope is implementation audit and sealed numerical microbenchmark only. No code/job/teacher call/performance result; `<160 GPU-hours` gate precedes zero-teacher SCGP-0.
- Full verbatim reviews: `refine-logs/lb_scgp/round-1-review.md` through `round-4-review.md`; final proposal/report under `refine-logs/lb_scgp/`.

## Iteration 8 candidate-registry recovery — 2026-07-28

No external reviewer was invoked for this documentation-only state recovery, so no verbatim reviewer output is added. The next raw entry must come from the independent reviewer gate for a candidate that survives the prospective novelty and minimal-pilot gates.

## Iteration 8 · C01 A0 external method review · Round 1 · reviewer `/root/idea_reviewer` · 2026-07-28

## 不可变 Problem Anchor

> 在不改协议、不加数据、不 scale、不 ensemble 的前提下，修复三个视频仇恨数据集 88–93% 跨 seed 稳定的高置信邻域反转；最终至少两个数据集在同协议 3 seeds 下，Acc 与 macro-F1 均提升不低于 0.030。

A0 只能服务这个 Anchor，不能把问题降格为“证明两个已有缓存可融合”或“找到更好的 prompt readout”。

## 核心判断

**当前结论：REVISE。**

A0 是一个合理、低成本、可严格证伪的 **readout-policy contrast 筛选器**，但还不是充分的 neutral–policy 机制验证。补上一个关键对照后，若通过预注册门槛，它足以授权生成同-pooling neutral/policy 缓存；A0 失败则只能 kill 当前两 endpoint 的这种代数读出，不能科学地否定 neutral–policy hypothesis。

## Claim / mechanism / falsifiable prediction / intervention

### Claim

应将 claim 收窄为：

> 同一样本的两个 MLLM readout endpoint 包含一种不能被单 endpoint、分数平均或普通 endpoint concat 利用的、实例配对的变化方向；对 common 与 displacement 分块归一化后，该变化方向能改善部署一致的检索几何。

A0 **不能**声称：

- displacement 是 safety direction；
- common 是 neutral content；
- 收益来自 prompt policy contrast；
- 已经证明 content/safety disentanglement。

原因是 standard 与 one-word 同时改变了 prompt 和 pooling。

### Mechanism

令两个 endpoint 各自单位化：

\[
s=\frac{z_s}{\|z_s\|},\qquad w=\frac{z_w}{\|z_w\|}.
\]

构造：

\[
c=s+w,\qquad d=w-s,
\]

最终表示为：

\[
r=[c/\|c\|,\ d/\|d\|].
\]

同一变换用于 train memory 与 dev query，沿用原 FAISS top-20 和标签协议，无训练、无新数据、无模型选择。

必须预先定义 `||d||` 或 `||c||` 接近零时的 epsilon/退化处理；否则 `s≈w` 时会把纯数值噪声放大成单位向量。

### Falsifiable prediction

同一个预注册 primary `common_displacement` 必须在 HateMM 与 MHC-ZH：

- 相对最强对照同时 `ΔAcc ≥ .020`、`Δmacro-F1 ≥ .020`；
- 两项指标 paired-bootstrap `p05 > 0`；
- 多重比较 Holm `p ≤ .05`；
- HateMM 净修复至少 3，ZH 至少 2；
- 胜过 best endpoint、avg-score、endpoint-concat、common-only、shuffled pairing 以及下述新增对照。

任何一项失败，kill A0。

### Minimal intervention

就“现有两个 endpoint 是否含值得继续解耦的配对几何信号”而言，A0 接近最小机制：零训练、零新缓存、部署路径一致。

就“neutral/policy 分解能修复稳定反转”而言，它不充分；同-pooling neutral/policy 缓存才是第一次真正隔离 policy intervention 的实验。

## 最强伪新颖性与归因风险

### 1. 未分块归一化时，它严格只是正交旋转

对 `[s,w]` 使用：

\[
T=\frac{1}{\sqrt 2}
\begin{bmatrix}
I&I\\
-I&I
\end{bmatrix}
\]

即可得到缩放后的 `[common, displacement]`。`T` 是正交变换，因此若不分别归一化 common/displacement，query–memory 内积和 kNN 排序与 endpoint concat 完全等价。

所以 A0 唯一实际机制是：

> **对 endpoint 和差两个块做等权单位化。**

不能把“sum/difference decomposition”本身作为新颖性。

### 2. 分块归一化可能只是放大 endpoint disagreement

当 `s≈w` 时，`||d||` 很小，单位化会把 prompt/pooling 噪声强制提升到与 common 同权。正结果可能来自一种数据依赖的角度重加权，而非 safety residual。

必须报告：

- `||d||` 分布；
- 收益与 `||d||` 的关系；
- 极小 displacement 样本是否主导翻票。

这些属于机制诊断，不应再变成可调 gate。

### 3. Prompt 与 pooling 不可辨识

即使 A0 强阳性，也只能说明“两 endpoint 的变化方向有信息”。它不能区分：

- prompt wording；
- one-word answer constraint；
- pooling/readout；
- 三者交互。

因此 A0 阳性可以授权下一阶段做 matched neutral/policy cache，但论文归因必须等同-pooling实验。

### 4. “不 ensemble”边界需要写清楚

两次 endpoint 表示若只在 score 层平均，显然是 ensemble；A0 将两个配对 readout 合成为单一表示，形式上可作为 representation mechanism。但 reviewer 仍会质疑它是 two-view ensemble。已有 avg-score 和同维 concat 是必要的 compute/information-matched 对照。

## 唯一必须增加的对照

增加：

> **随机正交 endpoint rotation + 相同分块 L2 normalization。**

具体地，对 endpoint 轴使用若干预注册的二维正交旋转 \(R_\theta\)，得到两个混合块，再执行与 common/displacement 完全相同的 block normalization 和 concat。

理由：

- endpoint concat 只证明“加第二个 endpoint”是否有效；
- shuffled pairing 只证明样本配对是否重要；
- 两者都不能证明固定的 sum/difference 轴比任意旋转轴更有意义；
- 随机旋转对照能直接区分“语义上特殊的 common/displacement basis”和“任意旋转后做块归一化都能涨”。

判定应针对预注册角度集合的最佳值或分布上界进行校正，不能事后挑一个弱旋转。

**不建议删除现有任何一个对照。**

## 是否足以授权同-pooling neutral/policy cache

**条件性足够。**

满足以下三点即可授权：

1. 通过原有全部统计与净修复门槛；
2. 胜过随机正交旋转 + 相同 block normalization；
3. 增益不是由极小 `||d||` 的噪声放大样本主导。

科学解释只能是：

> “paired prompt/readout change contains structured geometry，值得花下一笔预算去隔离 policy intervention。”

不能解释为：

> “neutral/policy safety displacement 已成立。”

另外，A0 是**非对称 gate**：

- **阳性**：足以授权 matched cache；
- **阴性**：可以按项目预算 kill C01 当前路线，但不能作为 neutral/policy hypothesis 的严格反证，因为当前 endpoints 没有隔离该变量。

## Gate 评分

| 维度 | Gate 0：A0 本身 | Gate 1：作为 matched-cache 授权依据 |
|---|---:|---:|
| Problem Fidelity | 8.5 | 8.5 |
| Method Specificity | 7.5 | 7.0 |
| Contribution Quality | 4.5 | 6.5 |
| Frontier Leverage | 6.0 | 8.0 |
| Feasibility | 10.0 | 8.5 |
| Validation Focus | 8.0 | 7.5 |
| Venue Readiness | 3.5 | 6.0 |
| **Overall** | **6.8** | **7.3** |

按指定权重计算，A0 的高价值主要来自可行性、问题忠实度和验证聚焦，而不是贡献新颖性。

## Verdict

**REVISE。**

需要的修订很小但关键：

- 把 claim 从“common/safety displacement”降为“block-normalized paired-endpoint contrast”；
- 增加随机正交旋转 + 同 block normalization 对照；
- 明确 epsilon 和小 displacement 诊断；
- 把 A0 阳性定义为生成同-pooling neutral/policy cache 的授权，不定义为机制成立。

若修订后通过全部门槛，则对“进入下一阶段”可判 **READY**；对论文方法本身仍不能判 READY。

## Iteration 8 · C01 A0 Gate1 external re-evaluation · Round 2 · reviewer `/root/idea_reviewer` · 2026-07-28

## 不可变 Problem Anchor

> 在不改协议、不加数据、不 scale、不 ensemble 的前提下，修复三个视频仇恨数据集 88–93% 跨 seed 稳定的高置信邻域反转；最终至少两个数据集在同协议 3 seeds 下，Acc 与 macro-F1 均提升不低于 0.030。

本轮只判断修订后的 A0 是否足以作为“生成同-pooling neutral/policy cache”的科学授权 gate；不把 A0 当作论文结果、Stage0 成功或 Gate0→Gate1 晋级。

## 静态复核结论

### 1. Claim 收窄：PASS

`configs/c01/c01_a0_v1.json:5-7` 将 claim 限定为 parameter-free、block-normalized paired-endpoint contrast，并明确：

- 阳性只授权后续 same-pooling neutral/policy cache；
- 不建立 safety、stance 或 discourse disentanglement；
- 阴性只淘汰当前 standard-L24/one-word-L24 endpoint route。

同一边界被硬绑定在 `scripts/analysis/c01_policy_contrast_a0.py:41-55`，并在 `refine-logs/C01_A0_RECORD.md:9-14` 明确承认 prompt 与 pooling 同时变化。脚本输出还固定禁止 prompt-only causality、safety/stance disentanglement 和 end-to-end gain 解释。上一轮要求的 claim 收窄已经落地。

### 2. Random orthogonal control：PASS

`configs/c01/c01_a0_v1.json:128-140` 预注册六个非 45° 角度、相同 block L2 和完整角度集最大值；`scripts/analysis/c01_policy_contrast_a0.py:970-987,1033-1067` 实现旋转，并用 `theta=0` 对齐 endpoint concat、`theta=45` 对齐 primary 的 algebra guard。

统计上，primary 必须：

- 在 accuracy 与 macro-F1 上严格超过完整固定旋转集的 observed upper bound；
- 对每个固定角度都有正 bootstrap lower bound；
- 通过 rotation-family Holm correction。

这已经是上一轮要求的“随机正交 endpoint rotation + 相同 block normalization”强对照，而不是弱化后的单角度对照。

### 3. 小 displacement 审计：PARTIAL / BLOCKING

已落地部分：

- `epsilon=1e-12`，零/退化 common 或 displacement fail closed；
- 报告 train/dev、image/text 的 raw displacement norm 分布；
- `1e-3` tiny threshold 与最大 5% tiny fraction gate；
- train-only 10th percentile 定义 small subset；
- 超过一半 fixes 来自 small subset 时 fail。

阻塞点在 `scripts/analysis/c01_policy_contrast_a0.py:1340-1345`：small-displacement fix concentration 固定比较 `common_displacement` 与 `endpoint_concat`。但最终净修 gate 在 `:1702-1719` 动态选择 strongest ordinary control，`+0.020` gate 在 `:1597-1617` 又按每个指标取 strongest control。

因此存在未封闭情形：primary 相对 endpoint concat 的全部 fixes 可由非-small 样本贡献而通过审计，但相对真正 strongest control 的净收益仍可能全部集中在 small-d 样本。此时 A0 仍会错误授权 matched cache，不能排除 reviewer 指出的 tiny-residual amplification。

最小修复不是增加新实验或新模块，而是让 displacement audit 同时使用 decision 中同一个 `strongest_control_name`，并要求相对该参照的 primary fixes 不被 small subset 主导。保留 endpoint-concat 版本作为机制诊断即可。

### 4. Same-pooling 授权语义：PASS

Config、canonical binding、decision enum 和 record 一致使用：

- `CONTINUE_SAME_POOLING_CACHE_ONLY`
- `KILL_CURRENT_ENDPOINT_ROUTE_ONLY`

阳性解释只允许生成 matched same-pooling neutral/policy cache；阴性不否定 same-pooling policy contrast。`C01_A0_RECORD.md:62` 进一步明确，即使 CONTINUE 也不是 Stage0 PASS 或 Gate0→Gate1 promotion。授权语义已经落地，没有 problem drift。

## 额外静态判断

- `common_interaction` 仅为 separately-corrected secondary，不参与 primary continue，未形成 contribution sprawl。
- 两个 sbatch 均无 `--time`，CPU-only，显式激活 `HateVideo`，符合 SLURM 约束。
- full-SHA256 preflight 在任何 `torch.load` 前执行 exact manifest match；本轮没有运行脚本、打开 cache、提交作业或产生指标。
- A0 的正结果仍只能支持“paired endpoint variation 值得被同-pooling 实验解混”，不能支持 policy/safety direction 已成立。

## Gate1 七维评分

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.2 |
| Method Specificity | 9.0 |
| Contribution Quality | 6.5 |
| Frontier Leverage | 8.0 |
| Feasibility | 9.3 |
| Validation Focus | 8.0 |
| Venue Readiness | 5.8 |
| **Overall** | **8.1** |

Contribution Quality 与 Venue Readiness 分数不是否定该 gate，而是因为 A0 被正确限定为 cache-only falsification/authorization audit，本身不是论文贡献。当前唯一 method-level blocker 是 small-d audit 与 strongest-control decision reference 不一致。

## Drift Warning

NONE。修订后的 claim 和 decision label 均保留 Problem Anchor；没有把 A0 生存误写成最终性能、Stage0 success 或 safety disentanglement。

## Simplification Opportunities

NONE。无需再加模型、head、数据集或指标；只需统一 small-d audit 的参照。

## Modernization Opportunities

NONE。A0 的职责是最小证伪，不应为提高“现代感”添加训练组件。

## Verdict

**REVISE。**

在 small-displacement gain-concentration audit 改为同时相对 decision 使用的 strongest ordinary control、并继续保留 endpoint-concat 机制诊断之前，A0 不足以授权同-pooling neutral/policy cache，也不应提交 hash preflight/runtime。

完成这一个参照一致性修复并通过静态复核后，A0 可判 **READY_TO_RUN_A0**。只有实际 A0 通过全部冻结门槛，才可授权 cache extraction；无论何种 A0 结果，都不构成论文结果或最终目标达成。

---

# C01 A0 v2 aligned structural-null adjudication（2026-07-28）

Reviewer: `/root/idea_reviewer`

Review type: independent evidence-bound method adjudication; no implementation review and no execution

Artifact under review: `artifacts/c01_policy_contrastive/v2/zero_contract_probe/C01-ZERO-PROBE-v1/zero_contract_probe.json`

Artifact SHA256: `bee4964ce7e4ca81cfdb72c3859f78196568badf982aef587bc14ee6dbe63526`

Probe job: `13717`

## Problem Anchor

C01 A0 仍然只是一个 **pre-Stage0、kill-only 的 paired readout-policy endpoint audit**：在 MHC-ZH 与 HateMM 上，用 strict train-memory → dev-query 协议检查现有 standard-L24 与 one-word-L24 endpoint 的 block-normalized common/displacement 几何是否比冻结 ordinary controls 提供足够的条件信息。由于两个 endpoint 同时改变 prompt 与 pooling，任何阳性都只能授权下一步 same-pooling cache preparation；不能解释为 prompt-only causality、safety/stance disentanglement、Stage0 success、Gate0→Gate1 promotion 或最终性能贡献。

本轮问题不是降低数值完整性门槛，而是判断：对历史 extractor 在唯一不可解码训练视频上写入、且被所有 endpoint/modality 与历史 R0 共同消费的 no-information sentinel，能否保持 exact zero，而不是把它单位化、插补、删除或让整个 A0 无法定义。

## Evidence chain

1. **Probe artifact 与 job log 一致。** Job `13717` stdout 明确记录 artifact 路径、上述完整 SHA256、`endpoint_zero_masks_exact_match=true` 与 `non_structural_tiny_absent=true`；stderr 为空。Probe 自身 exact-matches 已批准八文件 manifest SHA256 `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305` 后才加载 exact eight train/dev caches，且 test-like attempts/opens 均为 0。
2. **异常唯一且四向对齐。** Probe 的 HateMM/train 记录中，standard/one-word × img/text 唯一 exact-zero 都是 row `355`、raw ID `hate_video_95`、label `1`，另一 modality 也都是 `exact_zero`；standard-only 与 one-word-only zero 集均为空。MHC-ZH train/dev、HateMM dev 以及其余所有 cell 的 exact-zero、tiny-nonzero、nonfinite 均为 0。
3. **zero 的生成原因是解码失败，不是某个候选变换产生。** `src/utils/generate_VideoMLLM_embedding_readout_HF.py:255-278` 在 Decord 与 PyAV 都不能产生 frame 时返回 undecodable；同文件 `:422-429` 对每个 readout cell 的 img 与 text 写入同维 exact-zero，同时保留原始 ID 与 label。`refine-logs/W2A_EXTRACTION_RECORD.md:105-113` 进一步记录 `hate_video_95.mp4` 的 Decord partial-file/`av_read_frame` 失败、PyAV invalid-NAL 失败以及最终 `no decodable frames → ZERO-GUARD`。
4. **历史 deployed R0 确实消费同一个 zero，而不是 probe 的事后新产物。** `refine-logs/READOUT_SUBMIT_RECORD.md:166-185` 记录 HateMM train row 355 在全部四个 readout cells 中为 zero；R0 `ro_L28` 在两数据集全部三 split 上与 banked deployed cache full-cache bit-exact，img/text max absolute delta 都是 0、ID order exact。该记录明确说明 row 355 是 deployed cache 的 pre-existing guard，所有 cells 与 R0 共同消费。
5. **跨历史 extractor/readout 的身份稳定。** `refine-logs/MNTP_S1_RECORD.md:172-185,225-230` 记录同一 raw ID、同一 train index 在 causal、F72-bidir、S1 中均为 identical zero，并指出 `cosine(0,0)=0` 只是 belt artifact；`refine-logs/PROVENANCE_AUDIT_2026-07-28.md:185-194` 记录同一 label-1 row 在四个 banked caches 中 img/text 都为 zero。后者也揭示了最危险的混淆：deployed cutoff rule 会令 zero-query 的 `v=0` 预测为 1，因此 v2 绝不能依据 label 授权、也不能允许这个 sentinel 在不同 arm 中产生不同 retrieval influence。

证据链因此闭合为：**不可解码 → extractor 对所有 cells 写入同一个 no-information sentinel → exact eight current caches 保持同 ID/row/policy/modality 对齐 → historical R0 的 full-cache bit-exact 记录证明 deployed baseline 消费同一 sentinel。**

## Adjudication

**GO_CONTRACT_ONLY / NO_GO_EXECUTION。**

C01 A0 v2 可以只对这个 probe 与历史证据绑定的 aligned structural null 实行 zero-preserving normalization。这里的 “structural null” 只表示 extractor 的、协议历史中已经存在的 no-information sentinel；不表示一个有语义的零向量，更不是一个可泛化的 `allow_zero=true` 类别。

该 GO 仅批准下述合同作为 prospective implementation specification。当前没有 v2 implementation/static review/runtime authorization，也没有 v2 result、metric、CONTINUE 或 KILL。`allow_zero_block_in_a0=false` 对现有 v1 实现继续有效，直到新合同被实现、冻结并通过独立静态复核。

## Exact v2 contract

1. **Immutable authorization tuple.** 唯一授权 tuple 是 `(dataset=HateMM, split=train, raw_id=hate_video_95, row_index=355, policies={standard,oneword}, modalities={img,text})`，并绑定 probe artifact path、run ID `C01-ZERO-PROBE-v1`、artifact SHA256 `bee4964ce7e4ca81cfdb72c3859f78196568badf982aef587bc14ee6dbe63526`、approved eight-cache manifest SHA256 `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305` 与上述历史 evidence paths/ranges。V2 config 必须再冻结每个 evidence file 的完整 SHA256；任一 hash、ID order、row index、shape、policy、modality 或 split 漂移都 fail closed。
2. **Label is integrity-only.** Expected label `1` 可以作为 cache drift assertion，但不得参与 null-mask 构造、normalization 分支、shuffle 固定点选择或任何预测逻辑。任何 label mismatch 都停止；绝不允许 “因为它是正类所以保留 zero”。
3. **Exact state partition.** 对每一行先检查全部 finite，再按 norm 分类：`exact_zero` 仅指 norm 恰为 0；`normal_nonzero` 要求 norm `> epsilon=1e-12`；任何 `0 < norm <= epsilon` 都是 `tiny_nonzero` 并停止；任何 nonfinite 都停止。不得把 tiny round 到 zero，也不得把 zero 加 epsilon 后单位化。
4. **Mask is evidence-derived, not norm-discovered permission.** Runtime 可以重新观察 norm 来核验，但授权 mask 必须从冻结 tuple/probe 合同构造。一个未在 tuple 中的 exact-zero，即使数值完全相同，也必须停止。
5. **Zero-preserving normalization.** 唯一合法算子语义是：若且仅若 frozen null mask 为真且输入被核验为 exact-zero，则输出保持 bitwise exact-zero；若 mask 为假且 finite norm `> epsilon`，才输出 `x / ||x||`；其余情况全部停止。这个语义必须覆盖 raw img/text blocks、modal fusion、endpoint blocks 以及后续任何 normalization site，不能只修第一个 `l2_rows`。
6. **Four-way alignment before authorization.** HateMM/train 的授权 mask 必须在 standard/one-word × img/text 四个 cell 上 exact-equal，且恰有一个 true；HateMM/dev 与 MHC-ZH/train/dev 的 mask 必须全 false。任一 endpoint-only、modality-only、额外 zero、tiny 或 nonfinite 都停止。
7. **Derived-representation closure.** 对授权行，endpoint standard/one-word、endpoint concat、common、displacement、`common_displacement`、secondary `common_interaction` 以及每个 frozen orthogonal rotation 都必须保持 exact-zero；这些线性/interaction 路径不得生成方向。除授权行之外，任何 derived exact-zero/tiny/nonfinite 继续按原 v1 guard 停止。每个 arm 的输入/输出 null-mask equality 必须持久化。
8. **Shuffle control keeps the null fixed.** 256 个 label-blind ID-hash paired shuffles 只能在 non-null IDs 上置换；`hate_video_95` 必须是唯一冻结 fixed point。否则 shuffle 会人为产生 standard-zero/oneword-nonzero 与其反向 mismatch，既违反合同也污染 control。每个 permutation 必须记录 digest、permutation universe、fixed-null count；p95 shuffle gate 与 256 次数不变。
9. **Retrieval no-influence guard.** 对 HateMM 每个 dev query、每个 ordinary arm、primary/secondary、每个 rotation 和每个 shuffle，持久化 top-20 neighbor IDs。授权 null 不得进入任何 top-20；等价的 with-null 与 remove-null audit 必须得到 bit-exact scores/predictions 和完全相同的 gate booleans。若它被检索、移除后改变任一 score/prediction/gate，v2 必须发布 `HALT_STRUCTURAL_NULL_INFLUENCE`，不得发布科学 CONTINUE/KILL。这样 label-1/cutoff/tie behavior 不可能成为候选 arm 的隐藏信号。
10. **Historical parity is inclusion parity, not imputation.** R0 与所有 candidate/control arms 的主分析都保留同一 row、同一 order、同一 exact-zero。不得只从 candidate 删除、只从 R0 删除、替换为 mean/random/learned vector，或给它任意 unit direction。Remove-null 只作为上述 binding sensitivity audit，不替代 historical-parity primary protocol。
11. **Negative fail-closed fixtures are required before execution.** 静态/测试复核必须覆盖 unexpected zero、tiny-positive、nonfinite、endpoint-mask mismatch、modality-mask mismatch、ID/row/label drift、probe/evidence hash drift、shuffle 移动 null、derived new-zero 与 null entering top-20；每种情况都必须在 result/decision publication 之前停止。
12. **Output-schema binding.** Result/decision schema 必须持久化 probe/evidence/manifest hashes、授权 tuple、raw/derived mask counts、fixed-point shuffle audit、top-20 null occurrence count、with/remove-null parity、所有 fail-closed counters 与 fired validity rule。缺任一字段不得发布 decision。
13. **No scope expansion.** 此合同不授权 test access、其他 dataset/split/ID、未来 extractor 的 zero、missing rows、empty transcript、tiny vector、nonfinite vector 或任何自动 “structural_zero” 推断。新异常必须重新 probe 和独立裁决。

## Historical comparability and scientific gates

历史 R0 可比性在这个合同下保持，原因不是“zero 无害”的假设，而是：

- primary protocol 继续包含 deployed R0 已经包含的同一 raw row、同一位置和同一 exact-zero；
- zero 不被单位化或插补，因此不会被赋予历史中不存在的方向；
- 所有 real/control arms 对该 row 的 representation 都是 exact-zero；
- binding no-influence/remove-null audit 排除不同几何导致该 sentinel 进入 top-k 或通过 tie/displacement 产生 arm-specific 影响。

原 A0 的科学性能门槛保持不变：两个数据集上的 accuracy 与 macro-F1 `+0.020/+0.020`、harm boundary、paired bootstrap/Holm、net-fix 数、完整 fixed-rotation family、256-shuffle p95、small-displacement 与 exact strongest-control reference 均不得降低、改名或豁免。V2 只增加 data-integrity validity gates；这些 gate 只能 HALT，不能帮助 CONTINUE。Shuffle 的 sole-null fixed point 是对 decodable-row permutation universe 的必要限定，不改变 shuffle 次数或 p95 decision rule。

因此，**R0 comparability preserved；scientific thresholds unchanged；validity bar strengthened。**

## Strongest reviewer objection

最强反对是：这是在一次失败后，为一个 label-1 corruption row 增加 dataset/ID-specific post-hoc exception。历史 baseline 曾消费它并不自动使它科学有效；zero 在 cosine top-k 中会参与 tie/order，且 transformed arms 的非零邻居分布不同，所以该 row 即使自身 similarity 为 0，也可能通过占据第 20 位、挤掉负相似邻居或触发 cutoff 行为而产生 arm-specific gain。把它叫做 “structural” 还可能掩盖 missing-video bias。

这个反对成立，除非全部 v2 合同被满足。可接受的防线不是宣称 zero 有语义，而是把它限定为 **历史协议兼容的 no-information sentinel**，禁止 label-dependent branching，固定 shuffle 中的位置，并用 top-20 exclusion + exact remove-null parity 证明它对本次 A0 decision 没有作用。一旦 influence guard 失败，应停止并把 A0 判为对 corrupted memory item 敏感；不能放宽 guard、不能删掉 R0 中的 row 后继续声称历史可比，也不能把结果解释为 policy representation improvement。

## Final verdict and boundary

**Verdict: GO_CONTRACT_ONLY.**

允许实现一个 evidence-bound、single-tuple、zero-preserving A0 v2；不允许把 `allow_zero_block_in_a0` 全局改为 true。下一步只能是 prospective v2 config/code/schema/preregistration 实现，然后进行 fresh independent static review。该 review 通过之前不得提交 A0 v2；本裁决不预写任何运行结果，也不改变 target unmet、Gate 0 unpassed、0 qualifying datasets 的事实。

---

# C01 retrieval numerical-equivalence guard adjudication（2026-07-29）

Reviewer: `/root/idea_reviewer`

Review type: independent evidence-bound guard adjudication; no implementation review and no execution

Diagnostic artifact: `artifacts/c01_policy_contrastive/v2/retrieval_equivalence_probe/C01-RETRIEVAL-EQUIV-PROBE-v1/retrieval_equivalence_probe.json`

Artifact SHA256: `724c87cd2fbdb763180b663bc6492322887bc2077f378c5b21c4184c4ba80e6f`

Probe job: `13732`

## Problem Anchor

C01 A0 remains a pre-Stage0, kill-only paired readout-policy endpoint audit. Its scientific question, datasets, train-memory → dev-query protocol, primary/secondary arms, ordinary controls, rotations, shuffles, bootstrap/Holm, small-displacement audit, net-fix requirements and `+0.020 accuracy / +0.020 macro-F1` minimal-pilot thresholds do not change.

The guard question is narrower: when an authorized exact-zero memory row is physically removed, FAISS receives a `744×d` versus `743×d` finite float32 matrix. Must “no influence” mean byte-identical float32 dot products and float64 scores, or may it mean a principled numerical equivalence while all discrete retrieval and scientific outputs remain exact?

The guard must detect a null that changes the candidate set, rank order, prediction, metric or scientific gate. It must not mistake a bounded finite-reduction perturbation, caused solely by changing the matrix shape, for scientific influence.

## Evidence chain

1. Job `13732` completed with empty stderr and emitted the above artifact SHA256. The artifact is diagnostic-only and binds the exact approved eight-cache manifest SHA256 `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305`, zero-probe SHA256 `bee4964ce7e4ca81cfdb72c3859f78196568badf982aef587bc14ee6dbe63526`, A0 source SHA256 `d2b9c2ff909c07518ae35526db9550df655fb4af395cc7a0899f83e48db1b855`, HateMM standard train/dev cache hashes and exact thread shape.
2. Imported A0 and independent local four-step `endpoint_std` construction are byte-identical for train `(744,7168)` and dev `(107,7168)`. Thus the diagnostic is evaluating the same key bytes that failed job `13730`, not a simplified surrogate.
3. Removal mapping is exact for all 743 retained rows. The registered null `hate_video_95`, original row 355, appears in zero top-20 lists.
4. Raw FAISS neighbors and deterministic `(-float32 similarity, original train index)` neighbors have zero element, set, query-order and byte differences. This remains true despite one rank-20/rank-21 exact-tie query, so the observed event is not a candidate-set or tie-order change.
5. Exactly 22 of 2140 float32 similarity elements differ; maximum absolute difference is `2.980232238769531e-7` and maximum ordered-float32 distance is `5 ULP`. All values are finite.
6. The same exact neighbor IDs/order and labels produce 20 of 107 finite float64 score differences, maximum absolute difference `2.2706531321858847e-8`. Prediction arrays have zero differences and exact dtype/shape/bytes. Canonical typed metric bytes are exact: accuracy `0.8411214953271028`, macro-F1 `0.8390977443609022`, ROC-AUC `0.9164244186046512`.
7. The best-supported diagnosis is therefore `FLOAT_VARIATION_WITH_STABLE_NEIGHBOR_IDENTITY`: changing the IndexFlatIP memory shape from 744 to 743 changes a finite subset of float32 reduction results without changing any neighbor identity/order or downstream discrete/scientific output. This is evidence for a reduction-shape effect, not proof of a particular hidden FAISS kernel schedule; the prospective contract must remain fail-closed if the discrete invariants ever change.

## Adjudication

**GO_CONTRACT_ONLY / NO_GO_EXECUTION.**

The v2 bytewise retrieval guard should be replaced prospectively by a principled numerical-equivalence guard. Requiring byte equality of independently computed float32 dot products across different matrix shapes is stronger than “the registered null has no retrieval influence” and is not a scientifically meaningful invariant.

This is not permission to use `allclose` with an observed-value tolerance. Similarity and score envelopes must be derived before execution from IEEE-754 precision, the actual arm dimension, the frozen top-k weighted-vote formula and exact shared operands. Neighbor identity/order, null exclusion, predictions and canonical metrics remain exact.

## Principled numerical-equivalence contract

### 1. Immutable lineage and inputs

- Bind the diagnostic artifact path/SHA256, manifest SHA256, zero-probe SHA256, authorized-null tuple, exact cache hashes, exact A0 key construction and the approved FAISS/NumPy/thread/CPU environment.
- Rehash all inputs before load. Any identity, source, dependency or environment drift is a HALT.
- All raw keys, normalized retained memory rows and normalized query rows used by the with-null and remove-null paths must be dtype/shape/order compatible and byte-identical after mapping. The two paths must reuse one frozen float32 normalization result or prove retained-row/query byte parity before FAISS search. This closes normalization drift before a dot-product tolerance is considered.

### 2. Exact non-numerical invariants

Every real arm, secondary arm, fixed rotation and each of 256 fixed-null shuffles must satisfy all of the following:

- local-to-original mapping exact for every retained row;
- registered-null top-20 occurrence count exactly zero;
- raw FAISS neighbor ID arrays exact in dtype, shape, C-order bytes and rank order after mapping;
- deterministic `(-similarity, original index)` neighbor arrays exact in dtype, shape, C-order bytes and rank order;
- per-query neighbor set and order difference counts exactly zero;
- neighbor-label arrays exact;
- prediction arrays exact in dtype, shape and bytes;
- canonical typed accuracy, macro-F1 and ROC-AUC bytes exact;
- all scientific validity/gate booleans exact;
- no NaN, `+Inf`, `-Inf`, malformed index, signed-zero ambiguity or nonfinite bound at any stage.

Any failure is `HALT_NUMERICAL_EQUIVALENCE`, before result/decision publication. No tolerance applies to these fields.

### 3. Float32 similarity bound derived from dot-product arithmetic

Let `u32 = 2^-24`, let `d` be the actual final normalized key dimension of the arm, and require `d*u32 < 1`. Define the standard round-to-nearest dot-product factor

`gamma32(d) = d*u32 / (1 - d*u32)`.

After retained operands are byte-identical, two valid float32 reduction orders on the same operands satisfy

`|fl_1(x·q) - fl_2(x·q)| <= 2*gamma32(d)*sum_i |x_i q_i|`.

The implementation must compute a conservative finite operand scale

`rho = max_(compared q,r) ||x_r||_2 ||q||_2`

from the exact shared normalized float32 operands using an upward-safe binary64 audit. By Cauchy,

`sum_i |x_i q_i| <= rho`.

Freeze the arm-level absolute envelope as the next power of two:

`B_sim(d,rho) = 2^ceil(log2(2*gamma32(d)*rho))`.

This is a dimension/precision-derived envelope, not a fit to job `13732`. For illustration only, if `rho<=1` it gives `2^-10` at `d=7168` and `2^-9` at `d=14336`. Runtime must use the actual registered `d` and audited `rho`; it may not substitute the observed `2.98e-7`.

For every similarity element require:

- both values finite float32;
- `abs(with-remove) <= B_sim(d,rho)`;
- ordered-float32 ULP distance at most the number of representable float32 codes contained in the same derived interval. For reference value `a`, define

`B_ulp(a,B) = max(ord32(upper32(a+B))-ord32(a), ord32(a)-ord32(lower32(a-B)))`,

where `upper32`/`lower32` round outward, `ord32` is the already validated sign-aware monotone mapping, and the interval is clipped only to finite float32. Persist observed maximum ULP and derived maximum allowed ULP per arm.

This exponent-aware ULP rule avoids an arbitrary fixed “8 ULP” rule and remains meaningful near exponent boundaries. Both the ULP and absolute tests are binding; exceeding either HALTs.

### 4. Weighted-vote score bound

For top-k `k=20`, frozen descending integer weights `w_r = 20-r` for zero-based rank `r`, total `W=210`, and exact shared neighbor labels `sigma_r ∈ {-1,+1}`, the score is

`s = (1/W) * sum_r w_r sigma_r a_r`.

With identical neighbor order and labels,

`|Delta s_exact| <= (1/W) * sum_r w_r |Delta a_r| <= B_sim(d,rho)`.

For each query, require the tighter propagated audit

`B_score(q) = (1/W) * sum_r w_r |Delta a_qr| + 2^-45`.

The `2^-45` term is a conservative power-of-two allowance for fewer than 100 binary64 multiply/add/divide roundings at unit weighted-average scale (`u64=2^-53`); it is not derived from the observed score difference. Require

`abs(score_with(q)-score_remove(q)) <= B_score(q)`

and also the arm-level sanity bound

`max_q abs(Delta score_q) <= B_sim(d,rho) + 2^-45`.

Both bounds are binding. The cutoff must be robust under the propagated envelope: with/remove predictions must be byte-exact and the closed interval obtained by expanding either score by `B_score(q)` must not cross the frozen cutoff zero. Otherwise HALT even if the two observed prediction bits happen to agree.

### 5. Necessary independent reference ablation

For the exact agreed top-20 IDs, re-score both paths with one deterministic binary64 reference dot-product/vote implementation over the byte-identical shared normalized float32 operands, in frozen rank order. Require:

- reference neighbor inputs and labels exact;
- reference similarities/scores exact between with/remove, or within an independently derived binary64 `gamma64` envelope if the two reference computations are intentionally separate;
- reference predictions and canonical metrics exact;
- FAISS-with versus reference differences finite and within the float32 envelope above.

This ablation is necessary because the proposed guard otherwise uses the same FAISS implementation both to create and excuse the discrepancy. It establishes that the null changes neither the operands nor the mathematical retrieval result and that only finite float32 reduction varies.

### 6. Strongest objection and response

The strongest objection is post-hoc relaxation: after v2 failed, replacing bytes-exact with a tolerance could be tuned to make the failure disappear. Moreover, job `13732` covers only HateMM `endpoint_std`; other arms, rotations and shuffles have different dimensions and margins. A loose dot-product bound such as `2^-9` could conceal a genuine numerical instability near top-k or cutoff even while one diagnostic prediction vector stays unchanged.

This objection is valid unless the thresholds are formula-derived and every stronger discrete invariant remains exact. The response is therefore not “5 ULP is small.” It is:

1. derive the envelope from `u32`, actual `d`, audited operand norms and the exact weighted-vote formula;
2. require operand byte parity before applying it;
3. keep raw and stable neighbors exact for every arm/control;
4. keep null top-20 count zero;
5. require prediction and canonical metric bytes exact;
6. require cutoff-interval stability;
7. require an independent binary64 reference ablation;
8. HALT on any violation.

Under those conditions the repair narrows the guard to its scientific purpose and does not weaken the C01 decision.

## Versioning decision

This must be **a new v3 lineage, including a new config**:

- run ID `C01-A0-v3`;
- config `configs/c01/c01_a0_v3.json`;
- result schema `c01_a0_result_v3`;
- decision schema `c01_a0_decision_v3`;
- namespace `artifacts/c01_policy_contrastive/v3/a0/C01-A0-v3`;
- wrapper with a v3 identity.

Job `13730` executed the frozen v2 guard and halted. Mutating `c01_a0_v2.json`, reusing run ID `C01-A0-v2`, or writing into the v2 namespace would erase the reason for the observed HALT and break provenance. The analysis source may remain shared only if its canonical version dispatch, source SHA and v3 binding are independently reviewed; v1/v2 configs and empty failed namespaces remain immutable.

## Scientific thresholds and authorization boundary

No scientific CONTINUE/KILL threshold changes. The only permitted change is the HALT-only with/remove-null equivalence definition:

- exact bytes for operands, mapping, neighbors, predictions, metrics and scientific booleans;
- derived bounded equivalence for finite similarities and scores.

**Final verdict: GO_CONTRACT_ONLY / NO_GO_EXECUTION.**

This review authorizes only prospective v3 config/code/schema/preregistration work followed by fresh independent static review. It does not authorize a v3 submission, v2 retry, C02 job, test access, result, metric or scientific decision, and it does not prewrite any v3 observation.

## C01 A0 v3 scoped static re-review and runtime correction (2026-07-29)

### Static decision at authorization

The six requested repair families and frozen hash chain were reviewed without modifying the implementation. The v2→v3 allowdiff covered the exact-metric schema addition; the wrapper checked source and config hashes at runtime; the record pinned the wrapper; TARGET pinned the record; signed-zero position and signbit relations were explicit; the authorized tuple was checked against v2 and the diagnostic artifact; the decision validator covered the declared provenance/schema/result/scientific fields; public guard scalars were computed from runtime state; and v2 scientific arms and thresholds remained frozen.

The decision at that point was **GO (0 Critical / 0 High / 0 Important)**. That static decision authorized exactly one `sbatch scripts/slurm/c01_a0_cpu_v3.sbatch` submission and no C02.

### Runtime evidence and correction

Job `13735` was submitted once, waited for automatic `JobHeldUser` release, and failed closed at HateMM with:

`HALT_NUMERICAL_EQUIVALENCE: HateMM: derived public contract summary failed`

The original static decision missed a guaranteed cross-function schema mismatch:

- `evaluate_real_arms_v3` inserts `avg_score` into the same `validity` mapping as the actual retrieval arms.
- For HateMM, `avg_score_equivalence` returns `status=PASS`, `pass=true`, score/reference/evaluation evidence, but no `registered_null_top20_count` field (`scripts/analysis/c01_policy_contrast_a0_v3.py:1512-1539`).
- `derive_public_contract_guards` iterates over every entry in that mapping and requires each non-`NO_REGISTERED_NULL` audit to have `registered_null_top20_count == 0` (`:1998-2002`).
- The missing average-score field therefore evaluates as `None == 0`, making `registered_null_absent_from_all_top20=false`; the all-public-booleans guard then halts at `:2118`.

This is a **Critical runtime blocker** because it deterministically prevents the registered-null dataset from publishing any result. It is fail-closed and cannot create a false CONTINUE, but the earlier `0C/0H/0I` static verdict is invalidated. The failure does not show that the formula-derived numerical envelope, the registered-null top-20 invariant for actual retrieval arms, or any scientific threshold failed.

The namespace is empty and no result/decision was written, so there are no reportable raw metrics, public guards, fired scientific rules, dataset pass values, or CONTINUE/KILL verdict. No repair, retry, C02 or follow-up job is authorized by this correction.

## C01 A0 v4 terminal evidence and scientific adjudication (2026-07-29)

Job `13738` was submitted exactly once after independent static `GO (0 Critical / 0 High / 0 Important)` and exact preflight. It waited for automatic `JobHeldUser` release, ran on 8 CPU / 32 GB with no GPU or time limit, and completed `0:0`. No release, resubmission, dependency or chain occurred.

The result and decision artifacts are internally consistent: `C01_A0_DECISION.json` exact-pins result SHA256 `b45adf18cc35695618572abfbbc69571fe042fce9bdc48fce649bf7511e65f53`; run/config/schema identities match; the decision was exclusively created. The v4 six-case fail-closed schema self-test, v4/v3 lineage, frozen scientific base, runtime software/CPU/thread binding, 8/8 approved cache opens and all eleven aggregate halt-only validity guards passed. Both datasets pass all per-dataset typed-audit, zero-mask, registered-null, numerical-equivalence, binary64-reference, shuffle, displacement and scientific-basis guards. The execution therefore reaches a scientific decision and is not an engineering HALT.

The primary arm is `common_displacement`. Its MHC-ZH accuracy/macro-F1 is `0.8589743589743589 / 0.8479532163742689`, below selected strongest ordinary control `endpoint_concat` at `0.8846153846153846 / 0.8773370609820024`. Its HateMM accuracy/macro-F1 is `0.8598130841121495 / 0.8573713676352972`, below selected strongest ordinary control `common` at `0.8691588785046729 / 0.8671985815602836`. The binding gains are negative on both datasets.

Every primary-versus-control and primary-versus-orthogonal-rotation accuracy/macro-F1 bootstrap comparison has `holm_reject=false` and adjusted p-value `1.0`. The best rotations exceed the primary on both datasets. Net fixes versus the selected controls are `-2` on MHC-ZH and `-1` on HateMM, rather than the frozen required `+2` and `+3`. Gain, primary bootstrap, rotation and net-fix gates fail for both datasets.

All eight real-versus-shuffled accuracy/macro-F1 hypotheses do reject after Holm correction, and the displacement/history/contract gates pass. This establishes that the real pairing contains structure relative to the label-blind shuffled null, but it does not establish the required advantage over ordinary or equally normalized rotation controls.

The formal decision is **`KILL_CURRENT_ENDPOINT_ROUTE_ONLY`**, `continue=false`, with both dataset passes false. The interpretation is deliberately narrow: retire the current standard-L24 versus one-word-L24 endpoint-contrast route; do not infer that same-pooling policy contrast is false. C01 is frozen at this decision. C02 may proceed only to design and independent review; this adjudication does not authorize a C02 job.

## C02 independent design review and terminal adjudication (2026-07-29)

### Initial verdict: REVISE_DESIGN

The independent reviewer found two blocking flaws in the proposed A0. First, the
existing P3 mean/soft/mild representation banks alter image pooling while keeping
the text byte-identical, whereas final EDQ views alter transcript evidence density.
The proposed per-pair maximum over P3 variants is consequently an optimistic
image-routing upper bound, not a representation-matched oracle that can kill or
establish reachability for EDQ.

Second, deleting non-core transcript context is not guaranteed to preserve the
binary label. Quotations, counterspeech, archived material, lyrics, and reportage
can change meaning when their surrounding context is removed. The reviewer required
that any legal density view retain the complete native transcript as an ordered
subsequence and add only controlled repetition.

Before any S1 stage, the reviewer additionally required exact
`RANDOM_WINDOW_REPEAT`, `MIN_WINDOW_REPEAT`, `REPEAT_ONLY`, and
`LOCALIZED_REPEAT_ONLY` controls; frozen orbit radius, KRR metric,
retrieval-length-correlation formula, confidence/control thresholds, lambda
selection, Holm family, and full self-orbit exclusion; and an explicit reachability
account for empty/speech-poor HateMM cases whose text-density view is the identity.

The reviewer judged EDQ non-isomorphic to P3, SAV, and MECHFIX at the abstract level,
but not yet sharply distinguished from generic view consistency and CoLD-style
length debiasing.

### Follow-up asset evidence

A read-only audit found no train/dev representation bank on both HateMM and MHC-ZH
for native versus exact full-transcript repeat, localized repeat, prefix/suffix
repeat, echo, truncation, or another full-transcript-preserving density view.
`*p3pool_*` changes image pooling only; `*bidir/{meanpool,textpool}*` changes
attention/readout; `*nullop2merge*` is a PEFT merge-path null probe; and HateMM
`*curric-rep2*` is an independent SFT draw rather than an input-repeat view.
Echo/repeat embeddings were documented as an idea but never extracted.

### Final verdict: KILL_C02_DESIGN_COLLISION_OR_INFEASIBILITY

The candidate registry requires an existing-bank, representation-level oracle to
reach at least `+0.050` accuracy and `+0.050` macro-F1 on at least two datasets before
new extraction, teacher, or GPU spend. Because no representation-matched density
orbit bank exists, C02 cannot meet this binding gate. Reusing P3 would adjudicate a
different image-pooling hypothesis; extracting the needed text views would violate
the pre-extraction gate.

The reviewer therefore directed: freeze C02, do not implement or execute A0, do not
extract new views, and advance only to C03 design and independent review. This is a
design infeasibility under the current contract, not a scientific falsification of
abstract EDQ. No Python, pytest, cache opening, teacher call, GPU/test access, SLURM
submission, result, or metric occurred.

## C03 independent design review and terminal adjudication (2026-07-29)

### Concept review: REVISE — 3 Critical / 5 High / 4 Important

The binding Critical was the missing two-dataset representation-matched Stage-0
asset. The registry requires actual fold/deployed-head `+0.050` accuracy and
`+0.050` macro-F1 before teacher/GPU spend. Existing F72 raw/bidirectional, F92
readout, F93 transplant and C01 endpoint assets do not isolate native
policy-conditioned MNTP. The reviewer required a direct KILL if no such asset could
be demonstrated; training/extracting the missing bank first was explicitly
forbidden.

Other binding design requirements were: train-only label-blind mask/loss selection;
no hate/non-hate-conditioned policy text; no policy, teacher or pseudo-relation at
native inference; FULL versus `NATIVE_MNTP_ONLY`, REMOVE, SHUFFLE and matched NOISE;
outer-train-only unsupervised stopping; stream-collapse and fusion-synergy belts;
and current unified two-dataset thresholds rather than the retired crater-recovery
bars.

### Formal-file re-review: CONFIRM_KILL

The formal files were:

- `refine-logs/C03_ASSET_AUDIT.md`;
- `refine-logs/C03_REFINED_PROPOSAL.md`;
- `refine-logs/C03_EXPERIMENT_PLAN.md`;
- `refine-logs/C03_EXPERIMENT_TRACKER.md`.

The reviewer exact-matched all four submitted SHA256 values and returned
**`CONFIRM_KILL — 0 Critical / 0 High`**.

The core chain was confirmed: no HateMM+MHC-ZH bank matches native weight-point,
label-blind policy-conditioned MNTP under identical readout/path; producing it
requires training and re-extraction; that violates the pre-GPU Stage-0 rule. Old
caches may kill but cannot pass this missing mechanism.

Two non-blocking documentation points were incorporated. A fixed generic policy
prefix is only policy conditioning and does not identify per-example policy
relations. Also, matched-compute and control schemas would still require a formal
preregistration if the registry were ever reopened.

The final decision is **`KILL_C03_DESIGN_INFEASIBILITY`**. It is not a scientific
negative result for native MNTP. No Python, cache opening, teacher/GPU/test access,
SLURM submission, result or performance metric occurred. The next boundary is a
fresh literature/novelty Gate-0 review before C04; this record authorizes no C04
action.

## C04 independent design review (2026-07-29)

The reviewer exact-matched the five frozen file hashes recorded in
`refine-logs/C04_DESIGN_REVIEW.md` and returned:

**`REVISE_USER_AMENDMENT_REQUIRED — 2 Critical / 5 High / 4 Important`**

### Critical

1. The P8 proposition proxy preserves who is targeted and what could be hateful;
   K4 is only `0..3` evidence density with no target-act binding; deterministic
   S/T cue states are not the proposed teacher source/stance. DIRECT+STUDENT over
   those assets tests a different `summary+density+cue` tensor. It cannot PASS
   SPaSH and a null cannot scientifically KILL the stronger matched teacher.
   Therefore the current pre-teacher Stage-0 contract has no executable C04 path.
2. The hard supervision contract requires explicit reliability/confidence plus a
   deterministic missing fallback for every MLLM pseudo-signal. The draft has
   uncertain/missing values but no frozen reliability variable, conflict
   resolution, fallback tensor or required coverage/corruption reports.

### High

1. Add `CONCAT_ALL4_MLP`, retained independent-four-target/P4-strong, slotwise
   shuffle and role-permutation controls with exact q4/q<=3/capacity matching.
2. Close P4 and LEAF/C5-style knowledge-distillation collisions; discarded
   auxiliary P4 heads are not a strong control for a retained tensor branch.
3. DIRECT must be train-only OOF; STUDENT must be train-OOF and native-only dev.
   Remove the cross-arena 80% retention comparison.
4. Extend novelty review to RAMF, LEAF, TFN/LMF and DR-HM/Intent-style
   decomposition; absence of the exact conjunction is insufficient.
5. Freeze nested outer/inner folds and complete adaptation-plus-head paired
   lineages for seeds `0/1/2`.

### Important

Freeze primary/paraphrase prompt resolution; epsilon-safe zero-tensor behavior and
orthogonal-map hashes; 200-ID representativeness and resource accounting; and
exact tuple/slot/role/noise/remove-factor semantics.

### Required boundary

The existing proxy is nonbinding. Only the user may amend Stage-0 for C04 to
permit a bounded train-only local-teacher matched-signal pre-gate. The minimum
request is 200 label-blind SHA256-selected train IDs per dataset, no dev/test
teacher, one GPU at a time, `8 CPU / 64 GB`, aggregate first-tranche cap
2 GPU-hours. Only a small-gate PASS plus independent GO may complete the full
744 HateMM / 579 MHC-ZH train banks, under a total cap of 8 GPU-hours including
the first tranche. The original full-bank `+0.050/+0.050` gate is not waived.

This is not execution authorization. A user approval must be followed by design
repair and a fresh independent review. No Python, test, cache opening, teacher,
GPU, test access, implementation or SLURM submission is authorized.

## C04 V2 user amendment and design freeze (2026-07-29)

The user approved the bounded C04-only amendment, not execution. The exact V2
anchor/amendment/audit/proposal/plan/tracker/review-response hashes are registered
in `TARGET_STATE.json`. The response closes each prior 2C/5H/4I finding while
retaining a hard stop on implementation, teacher, Python/test, GPU, SLURM and
test access. Fresh independent design review is next; even a design GO would
authorize only prospective code/resource review.

## C04 V2 fresh review and V3 response (2026-07-30)

The reviewer exact-matched all V2 hashes and returned `REVISE (0C/4H/3I)`.
Accepted items require no new user contract. The exact verdict is archived in
`refine-logs/C04_V2_DESIGN_REVIEW.md`; V3 repair hashes are registered in
`TARGET_STATE.json`. No implementation or execution authority follows.

## C04 V3 fresh review and V4 response (2026-07-30)

The reviewer exact-matched V3 and returned `REVISE (0C/2H/1I)`. The read-only
verdict is archived in `refine-logs/C04_V3_DESIGN_REVIEW.md`; exact minimal V4
overlay hashes are registered in `TARGET_STATE.json`. The user contract and all
negative authority are unchanged.

## C04 V4 fresh design review (2026-07-30)

The reviewer exact-matched all V4 hashes, confirmed unchanged V3 bases and
returned **`GO (0 Critical / 0 High / 0 Important)`**. The read-only verdict is
archived in `refine-logs/C04_V4_DESIGN_REVIEW.md`.

This GO closes design review only. The next legal step is prospective
implementation plus fresh code/resource review. No implementation, teacher,
Python/test, GPU, SLURM, label/test access or execution is authorized.

## C04 implementation-v5 code/resource review and CPU-preflight authority (2026-07-30)

The independent implementation reviewer returned **`GO (0 Critical / 0 High /
0 Important)`** for the exact v5 config
`78e2ade7e91c74446eeba0d2965bc4675a804717857a0a202caabe1f80440a1b`
and implementation record
`aa49585cc27853793b349868bb6996ab5f051a44cfa1ff39c2fee0e7e1a704a1`.
The executor-side immutable transcription is
`refine-logs/C04_A0T_SMALL_V1_V5_CODE_RESOURCE_REVIEW.md`.

The next staged snapshot prepares only CPU preflight:

- authority manifest:
  `85a2ddc140ee523fdbdcd6764a736bdbd6b8c1731b7b76439207498d3d74d5a4`;
- authorized config:
  `5228c08fefcd41c354202dfd7a750b46e1a5d4d66a8b3562b736c9d943f526a6`;
- normalized config contract:
  `2bc1971e8b222e874a2000a2fca25b70e4391c41a0ecaf69b7040fdc7cb65f50`.

This prepared authority is not yet an unlock verdict. It must receive fresh
independent review before CPU preflight submission. Teacher, GPU, Slurm-GPU,
small-tranche execution, reconciliation and all forbidden surfaces remain
false; no Python/data/model/SLURM action occurred.

## C04 v5 CPU-preflight unlock verdict (2026-07-30)

The independent reviewer returned **`GO (0 Critical / 0 High / 0 Important)`**.
The exact transcription is
`refine-logs/C04_A0T_SMALL_V1_V5_CPU_PREFLIGHT_UNLOCK_REVIEW.md`
(`7b02c1ac67f447abc1f0a9501b1431c0e6d2ed289c311ca471a48327deda501e`).

The verdict authorizes only one invocation of
`scripts/slurm/c04_a0t_small_v1_v5_preflight.sbatch`. No job was submitted in
the authority/review update. All teacher/GPU/downstream stages remain blocked.

## C02 A0 preregistration — four independent static review rounds (2026-07-30)

C02 was revived under `STAGE0_BOUNDED_EXTRACTION_2026_07_30` and its A0 was
pre-registered and hash-frozen BEFORE any extraction, as the amendment requires.
Each round used a FRESH reviewer that had not seen the implementation reasoning and
was handed only the frozen files plus a review request. The full reviews are
persisted at the paths below; the verdicts and the load-bearing findings are
transcribed here.

| round | review file | sha256 | verdict | load-bearing findings |
|---|---|---|---|---|
| v1 | `refine-logs/C02_A0_PREREG_REVIEW.md` | `fd3209116e0853f25bfb208ba868d5a83655b93a9dc62e7db2278ba48d63fde5` | **`REVISE (0C/4H/20I)`** | leaking SHUFFLE derangement; false EMPTY_TEXT expectation; tautological net_fix_rate gate; unproved upper-bound claim |
| v2 | `refine-logs/C02_A0_V2_PREREG_REVIEW.md` | `f028eaa7170796676c8853c65489439ea9fcf2c6ba6b987456df79fafd5cebec` | **`REVISE (0C/2H/19I)`** | the retracted upper-bound claim survived in two frozen files; SHUFFLE still degraded under the design's own null |
| v3 | `refine-logs/C02_A0_V3_PREREG_REVIEW.md` | `373a82316ba1dc510044167bdc5aa000fe8188bd95f4b5d0718da529bfcedb26` | **`REVISE (0C/4H/23I)`** | retracted claim still in the arena docstring; derangement could hang then die outside the fail-closed path; 'exchangeable by construction' false; the SHUFFLE self-test could not fail |
| v4 | `refine-logs/C02_A0_V4_PREREG_REVIEW.md` | `b4698e85ab0d88f407b0109db812f39d23a8f73aaca515cc96ad3e0106bc0fd3` | **`GO (0C/0H/23I)`** | zero Critical, zero High; all four round-3 High findings verified REPAIRED in code; 23 Info findings remained open |

Three findings are worth transcribing because they changed the science, not the prose:

1. **The `SHUFFLE` control was measuring the wrong thing, twice.** Round 1 found the
   derangement was global over the train split, so ~80% of held-out queries carried the
   views of an item that was in the bank and retrieved a near-identity match with an
   unrelated label. Round 2 found that fixing the partition boundary did not fix the
   consequence: donating the donor's ABSOLUTE view keys still made bank row `j` a
   near-duplicate of bank row `pi(j)`, so `SHUFFLE` degraded under the design's own null
   and `FULL > SHUFFLE` was satisfiable when the orbit carries nothing. The repair is
   **displacement donation** — `z_i^v := NAT_i + (view_v(pi(i)) - NAT_pi(i))` — under which
   no component of the donor's position enters.

2. **`net_fix_rate` was a tautology presented as an independent gate.**
   `fixed - broken = n * delta_acc` exactly, so a `+0.030` net-fix conjunct can never bind
   under a `+0.050` accuracy bar. The registry amendment's net-fix clause is discharged
   **by the accuracy bar itself**, and the design now says so instead of restating the bar
   as a second gate. `precision_on_changed` was added as a genuinely independent fragility
   diagnostic and is **reported, not gated**.

3. **The claim that `s_Q` upper-bounds what any orbit-contracting representation could buy
   was withdrawn.** Max-cosine is one particular orbit-invariant similarity — the canonical
   max-matching quotient pseudo-metric — not a supremum over representations. A KILL from
   this A0 is therefore a **gate verdict under the registry's frozen Stage-0 rule**, not a
   proof that no orbit-contracting representation could ever help. That correction took
   three rounds to land in every frozen file, twice because a `str.replace` silently failed
   to match and the record asserted the repair without re-reading the file; every edit from
   round 4 on is applied by a helper that exits non-zero when its target does not match.

A fifth freeze (`C02-A0-v5`) closes the 23 Info findings that survived the round-4 `GO`
rather than running with them open. No threshold, bar, arm, metric or decision rule changed
in it.

## C04 implementation-v6 code/resource review (2026-07-30)

The v6 closure exists to repair one defect: the v5 CPU preflight could not pass
its own first static gate, which is why job `13805` died `FAILED 1:0` in
`00:00:00`. `c04_a0t_small_v1_v5_preflight.py:152` asserted
`prompt_hashes() == cfg["prompt_hashes"]` while
`configs/c04/c04_a0t_small_v1_v5.json:115-120` still held
`PENDING_CPU_PREFLIGHT_HASH_FREEZE` — the very values that preflight exists to
materialize. The v5 producer asserts the same equality at line 177.

Five independent static rounds, fresh reviewer each time, no Python run or
imported, no SLURM job touched, no label/video/weight opened, no file modified:

**`REVISE (0C/2H/2I)`** → **`REVISE (0C/1H/2I)`** → **`REVISE (0C/1H/2I)`** →
**`GO (0C/0H/1I)`** → **`GO (0C/0H/1I)`**, final Important closed in place.

The three High findings were substantive and are worth recording, because the
first two showed the initial repair had *displaced* the v5 impossibility rather
than removed it:

1. `config_contract_sha256` hashed `prompt_hashes` verbatim, so the contract
   baked into the authorization manifest, genesis ledger and resource ticket was
   computed over the pre-freeze config while downstream stages require the
   post-freeze one. The two states were mutually unsatisfiable: leaving the
   config unfrozen would burn the single GPU allocation before the producer's
   HALT, and amending it would permanently invalidate the pinned manifest.
2. The GPU-ledger claim stage was not literal-bound, so an unfrozen config
   passed `claim` end-to-end and consumed the single-use ticket before any
   consumer rejected it.
3. The fix for (2) did not protect the allocation either: the GPU wrapper arms
   its `EXIT` trap before the first Python call, so a claim-time HALT ran
   `mark-exit`, which bumped the genesis ledger's revision and state even with
   an empty job list, permanently breaking the ticket's
   `genesis_gpu_ledger_sha256` pin inside a no-clobber namespace.

The executor-side transcription is
`refine-logs/C04_A0T_SMALL_V1_V6_CODE_RESOURCE_REVIEW.md`.

## C04 v6 CPU-preflight unlock verdict (2026-07-30)

Four further independent unlock rounds over the complete authority snapshot:
**`GO (0C/0H/4I)`** → **`REVISE (0C/1H/2I)`** → **`GO (0C/0H/3I)`** →
**`GO (0 Critical / 0 High / 0 Important)`**.

Round 2's High is the one this campaign should remember: the implementation
record documented a reconstruction recipe for the reviewed `preflight.py`
revision that was true when written and false against the frozen bytes, yielding
`4d4dd033…` — a revision that never existed. A reconstruction claim a reviewer
cannot reproduce is the same class of unverified static assertion that put 13805
in the queue. It was corrected with an explicit erratum rather than overwritten.

The final reviewer recomputed 31 objects with zero mismatches, re-derived both
model tree hashes without opening a weight byte, and executed both claims that
would have been easiest to fake — the two-step `preflight.py` reconstruction and
the four-field config revert — reproducing each exactly. On "would any gate fail
again" it answered **no** from its own static reading.

The exact transcription is
`refine-logs/C04_A0T_SMALL_V1_V6_CPU_PREFLIGHT_UNLOCK_REVIEW.md`.

Frozen authority: config
`40ec6d97062498989ff9da21ebd6385aaee7fa3d2071d55b5664a1c5a135fc19`; normalized
config contract
`2b66775c44b727e35d52680d39eb838226d4f0a64fffd007d3e50ffcea79cdc5`; authority
manifest `5e56041adc5ef13527803f2c7950834cf59e38238a72dfb7a5c6a61b7e75b52f`;
closure `5375c39341933155286640563f5a3d588372acd2668a0cbbe3ba84639592639e`.

The verdict authorizes only one invocation of
`scripts/slurm/c04_a0t_small_v1_v6_preflight.sbatch`. Teacher, GPU, small
tranche, reconciliation, dev/test, OCR, API/network, labels, chain, release and
resubmit all remain false.

## C04 v6 CPU-preflight payload review (2026-07-31)

Fresh independent payload reviewer, no exposure to the v6 implementation
reasoning, read-only plus documentation. Single round.
Verdict: **`GO (0 Critical / 0 High / 3 Important)`**.

The reviewer verified by recomputation rather than by reading the job's own
`all_passed`, from a scratchpad outside the repository with
`PYTHONDONTWRITEBYTECODE=1` throughout, and decoded only `id`/`window_text`/
`language` from any ASR file, so no label value was materialized on the review
side either. The frozen `..._v6_common.py` was not imported from its repository
path: the prompt constants were extracted by static `ast` parse of the frozen
bytes, and the fixture spot-check ran against a byte-identical scratchpad copy.

What was recomputed and matched:

- **Prompt-hash freeze.** All four literal hashes reproduce from the frozen
  prompt sources — `system 1ffc0675…`, `A cecb3555…`, `B 9521bee7…`,
  `combined a42268e4…`; `payload_sha256 eb485b9a…` self-consistent; key set
  exactly `{A,B,combined,system}`; `downstream_binding LITERAL_BOUND`. Across
  the whole 15-file namespace the sentinel string appears only in
  `prompt_hashes.json`, only as the value of `pending_sentinel_token`. The 13805
  failure mode is closed as a contract, not bypassed.
- **Allowlists.** The exact 200 + 200 ID sequences reproduce from the frozen
  selection rule; ranks contiguous 0..199; every `selection_sha256` recomputes;
  strictly ascending `(digest, id)`; merkle roots `5897b44c…` / `24d40b0e…`
  recompute; `selection_contract` equals the config block. **Zero** dev/test
  contamination: selected ∩ dev = 0 and selected ∩ test = 0 on both datasets
  (HateMM dev 107 / test 215; MHC-ZH dev 78 / test 149), and every selected ID
  is in train.
- **Source manifests vs disk.** All 400 selected videos re-hashed —
  3,430,759,978 bytes read, **0 mismatches** on sha256, size, device, inode,
  resolved relative path and lexical path; all 400 transcript hashes, scalar
  counts and languages re-derived and matched; merkle roots `a8eab8ad…` /
  `af2f8d7a…` recompute. All 15 payload files match the per-file `bytes`/
  `sha256` table already in `TARGET_STATE.json`: 15/15 exact, no post-job drift.
- **Access ledger.** 402 events = 2 train-ASR opens + 400 video hashes;
  `label_field_syntactically_skipped 1323` (= 744 + 579) with
  `label_value_materialized 0`; events merkle `60f22f38…` recomputes; no
  dev/test/val path component anywhere; ledger events match the manifest rows
  1:1 in order.
- **Maps.** Both dense payloads rebuild **byte-exactly**; lengths are exactly
  256×3598×4 = 3684352 and 256×1024×4 = 1048576; every float32 is ±0.0625,
  equal to `maps.scale`. All four role maps rebuild byte-exactly with 256 unique
  indices in [0,3584), signs ⊆ {−1,1}, self-consistent `payload_sha256`, and
  pairwise-distinct index lists. The config's "documentation-only" `maps.*`
  geometry declarations were each checked against the module constants and the
  materialized bytes and agree on every count.
- **Resource state.** Ledger `GENESIS_UNCLAIMED`, revision 0, zero jobs, zero
  accounted and reconciled seconds, cap 7200 s = the amendment's 2 GPU-hour
  first-tranche ceiling; ticket unconsumed, single-use, 1 authorized allocation,
  0 completed seconds, watchdog 7080 = 7200 − 120, genesis pin equal to the
  ledger file on disk, `issued_by_slurm_job_id 13840`. Independently, `sacct`
  shows the only two C04 jobs ever are 13805 (FAILED) and 13840 (COMPLETED
  00:00:19) and **neither carries `gres/gpu` in AllocTRES**. Consumption,
  claim, entry-marker, lock, `seal/` and `checkpoints/` are all absent.
- **Authority chain.** 15/15 implementation hashes; config `40ec6d97…`;
  normalized contract `2b66775c…` recomputed independently and equal in all four
  in-payload copies; manifest `5e56041a…` with self-consistent closure
  `5375c393…`; records, review transcriptions and both job logs all match their
  pins. Contract neutrality tested directly: filling the sentinels in with the
  literals does not move the hash, while tampering with
  `small_cap_gpu_seconds` or `teacher_contract.num_frames` does.
- **Self-test honesty.** 13 prompt-hash fixtures + 7 legacy fixtures + 5 added
  by `run_self_tests` (`role_{S,P,T,H}_shape`, `no_test_paths`) = exactly the 25
  reported. Four checks probed for vacuity and found real: `valid_form` /
  `malformed_form` genuinely discriminate; `transcript_cap` is real arithmetic
  over a real branch (2061 = 1024+13+1024, short input takes the other branch);
  a sentinel-bearing payload with a *recomputed valid* `payload_sha256` is still
  rejected; and `_raises_runtime_error` returns False for a no-op and for a
  `ValueError`, True only for a `RuntimeError`.

The three Importants:

1. **The HateMM identifier is itself label-bearing.** IDs are `hate_video_*` /
   `non_hate_video_*`, and the sealed allowlist and source manifest store plain
   IDs, so any reader of the frozen payload holds the HateMM label of all 200
   selected videos — while the access ledger stores only `video_id_sha256`,
   which then buys nothing. This does **not** break the selection: the exact
   200-ID sequence reproduces from tag/dataset/id/suffix alone, and the draw is
   demonstrably unengineered (78 hate / 122 non-hate = 0.390, against a train
   prior of 298/446 = 0.401). It does **not** reach the teacher either —
   `producer.py:768` interpolates only `{transcript}`. It is filed because the
   seal must not be relied on downstream as label containment for HateMM, and
   "the teacher sees no label" is currently a property of one line of prompt
   assembly rather than a checked precondition.
2. **`selection_deterministic` is a tautology.** It asserts
   `selection_digest("HateMM","x") == selection_digest("HateMM","x")` — a pure
   function against itself, unfailable under any mutation of the rule, tag,
   suffix or digest payload. "25 checks, all_passed" is quoted as evidence in
   three documents; 24 of the 25 carry information. Non-blocking, since the
   property is independently proven by the exact allowlist reproduction.
3. **The amendment's 8 GPU-hour aggregate C04 ceiling is encoded nowhere.** The
   payload correctly encodes the 2 GPU-hour first-tranche cap; the conditional
   full-bank tranche's 8 GPU-hour aggregate — explicitly inclusive of first-
   tranche seconds — has no accumulator in any artifact and is enforced by prose
   only. Acceptable at this stage, but the conditional tranche's code/resource
   review must add one and carry the first tranche's actual spend into it.

One observation recorded and not filed: `frozen_payload.total_bytes = 5178606`
in `TARGET_STATE.json` is `du -sb artifacts/c04`, not the sum of the 15 file
sizes (5174184); every per-file `bytes` and `sha256` in that table is exact, so
this is field-labelling ambiguity with no integrity impact.

**Boundary.** A GO here means only that the payload is well-formed and faithful,
making C04 *eligible* for a separately-authorized teacher small tranche. It
authorizes no work. All GPU work remains blocked pending explicit main-dialogue
execution authorization; dev/test teacher remains forbidden forever under the
amendment; OCR, API, network, cross-dataset data, pre-seal label access, chained
submission, release and resubmission all remain false. This markdown verdict is
**not** the machine-checked authorization: `verify_payload_review` still needs a
`refine-logs/C04_A0T_SMALL_V1_V6_PAYLOAD_HASH_REVIEW.json` under schema
`c04_payload_review_v6`, a 64-hex config pin replacing the
`PENDING_CPU_PREFLIGHT_AND_PAYLOAD_REVIEW` sentinel, and an
`attested_closure_sha256` over the `C04-PAYLOAD-REVIEW-GO-v6` domain; none of it
exists. It must pin `preflight_manifest_sha256`
`06bf6b38f424dd53d142367abd029dfa1f485380fb1482d72beabb7f5943ad1a`.

The full review is `refine-logs/C04_A0T_SMALL_V1_V6_PAYLOAD_REVIEW.md`.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 1 (raw)

Reviewer: fresh independent worker, no exposure to the adjudicator reasoning.
Request: refine-logs/GATE0_REOPEN_2026-07-31_REVIEW_REQUEST.md
Verdict: REVISE - 1 Critical / 3 High / 10 Important. All findings applied;
the Critical moved C07 from strike to hold.

**REVISE — 1 Critical / 3 High / 10 Important**

The record's core work is real and largely holds up. I independently re-derived the census from `data/gt/*/{train,val}.jsonl` and every load-bearing `[M]` figure reproduces exactly (key-set `['id','label','text']` on all six files; HateMM whitespace-only 39/9; ZH tags 243/34; `em` 254//`em` 254 and 49 keywords on train; hate rate 141/243 = 0.5802 vs 39/336 = 0.1161, base 0.3109; val 20/34 vs 8/44; markup fraction median 0.0000, max 0.8621, 203 rows >10%, markup-bearing median 0.2604; `<76`-char rows 161/221/92; Archive = MHC + MHC_zh only; CLIP_Embedding 100/130/71/6/2). The C01 arm table verifies cell-for-cell against `C01_A0_OUT.json` with every accuracy recomputable from the stored confusion matrices. F55, F60, F70, F80, F82, F88, F96, F98, F99, F105, F112, F113, F114, EUM, BSY, TVB, LITSWEEP2/3/5/8, HEADCOV, GRADEDLBL, NCA_FORENSIC_RECON and the C05 comparator quotes are all verbatim-faithful. The C12 F55 leg (the decisive downgrade leg) is correct. I also stress-tested the C13 measurement — rows containing a harvested keyword *without* markup hate at only 10/140 = 0.0714 versus 141/243 = 0.5802 with it, so the tag itself carries the signal and the C13 strike's inference survives.

The defects below are scope, kind-of-record, and precision failures, not fabrication.

---

### Critical

**C-1 · C07 is struck on a precondition that was never attempted and on a screen that was never run.**
`refine-logs/GATE0_REOPEN_2026-07-31.md:230` states *"The reachability screen its boundary also demands **has been run and fails on both datasets**"* (mirrored at `TARGET_STATE.json:131`, `TARGET_FINDINGS.md:75`, `TARGET_LOOP.md` disposition table). No reachability screen of C07 has ever been run. What was run is F82's **vote-side** Offensive-reweighting oracle, and F82's own `ban_scope` (`directions_tried.json`, F82 entry) splits the two explicitly: *"vote-side Offensive reweighting closed both datasets … **head-side graded auxiliary = F44-capped + admissibility-gated, only revivable by user ruling WITH a new mechanism argument**."* C07 is a cone metric — a head-side/representation object — so the cited evidence sits on the other side of its own source's written boundary. The record concedes this three paragraphs later: its unblock at `:247` demands *"a **fresh** reachability screen at `+0.050`"*, which is unnecessary if the screen had already run. Leg 1 (`:227-229`) is weaker still: it says only that *"no delta has been written"* — an un-attempted precondition, whereas C05 is given `held_*` at `:445` for a precondition that was attempted **and demonstrated unwritable**. The record's own house rule against this substitution is the one it invokes to downgrade C12 (`LITSWEEP3_DATA_CENTRIC.md:80`, *"a headwind to price, not a coverage of this mechanism"*). C07 is therefore a strike carrying a HOLD's evidence and a HOLD's unblock, and the record's claim at `:222` that each confirmed strike *"survives on ban-free, faithfully-quoted evidence"* is false for it.
**Repair:** re-dispose C07 as `held_lattice_delta_unwritten_reachability_unscreened` with the same reversibility language as C05/C10/C11/C12; delete "has been run and fails"; restate F82 as a headwind priced from the vote-side channel and quote F82's head-side clause verbatim; or, to keep the strike, name a ban that reaches C07's object on its own text.

---

### High

**H-1 · C08 premise 2's categorical "no >=2-dataset substrate" is measured only for HTML *tags*.**
`refine-logs/GATE0_REOPEN_2026-07-31.md:256`: *"`0` on HateMM and `0` on MHC-EN — re-measured exactly. A `>=2`-dataset route has no substrate."* MHC-EN carries **64/549 train and 9/80 val rows with HTML entities** (`&#39;` x51, `&quot;` x22, `&amp;` x18 — recomputed by me) — un-cleaned scrape residue of the same provenance family. The recon's own census table (`C05PLUS_FORENSIC_RECON_2026-07-31.md:436-441`) carries this column; the record's §2 verification table (`:51-67`) silently drops it while certifying the rest "exact".
**Repair:** restate premise 2 as tag-scoped, record the MHC-EN entity counts in §2 and in `TARGET_STATE.json` `premise_2`, and either show entities are not a usable provenance substrate or soften "no substrate" to "no *highlight-marker* substrate".

**H-2 · The C11 downgrade truncates its own headwind citation before the clause that contradicts it.**
`:389` quotes `ERRPAT_MHC-ZH:405` as *"effectively LOCKED … +0.0738 if all 11 flipped, but §7.1 shows no better transcript exists; the deficit is signal absence"* and concludes it is *"a headwind to price rather than a closure."* The source row ends with a further clause the record drops: **"No legal unmeasured lever found."** That is the source's own verdict on the exact cluster the downgrade calls "unscreened", and it is the strongest text against the downgrade. Same truncation at `TARGET_STATE.json:182`.
**Repair:** quote the row in full in both places and re-argue why a missingness-representation lever is not covered by "no legal unmeasured lever found", or convert C11 back to a strike.

**H-3 · The C12 downgrade's narrow-reading precedent is quoted without its own blocking clause.**
`:414` cites *"F60/AUG rules **MLLM-as-data-generator admissible**"* and the unblock at `:435-436` offers stability-as-multi-view-target as *"governed by F60's admissible generator role"*. F60's `ban_scope` ends **"Do not re-propose without D7 generator-role sub-ruling"**, and F60's detail closes *"Revisit only under a user D7 generator-role sub-ruling AND acceptance of a weaker-than-tied prior."* D7 is an open user ruling (`progress.json` handoff: *"the five open USER RULINGS (D7 novelty boundary; …)"*). The record therefore offers C12 an unblock route that is itself blocked — the identical omission it charges the recon with for F80.
**Repair:** add F60's D7 clause verbatim to `gap_2` and to the unblock in both `GATE0_REOPEN_2026-07-31.md:414-418,435` and `TARGET_STATE.json:191,194`.

---

### Important

**I-1 · Both "cosmetic corrections" are median/percentile *convention* differences, not errors — and the correction is applied inconsistently.** `:69` and `:75`. The recon's `0.5155` is exactly numpy's `higher` p90 on MHC-ZH train (I reproduce it to 4 dp); `696` is exactly the upper median of HateMM train. The recon used the upper convention **consistently** — MHC-ZH val `111` and MHC-EN val `443` are also exact upper medians, and the record leaves both unremarked while certifying the layer "exact". Landed as errata in three files (`TARGET_STATE.json:37-47`, `TARGET_FINDINGS.md:75`, `TARGET_LOOP.md`). **Repair:** restate M-1/M-2 as convention notes ("upper vs interpolated"), not corrections, and either check or drop the remaining medians.

**I-2 · The C06 "bans do not reach C06" correction cites carve-outs whose object is C14, not C06.** `:509-515`. F80's *"multi-prompt ensembling remains a SEPARATE user-gated item"* and F70's *"Does NOT price: … multi-prompt ensembling"* both carve out **ensembling** — which C06's own dedup boundary forbids it from becoming (`TARGET_STATE.json:200`). The conclusion (F80's prompt-language ban does not reach a prompt-orbit geometry candidate) is right; the warrant conflates C06 with C14. **Repair:** re-base on object mismatch (C06 is not prompt-language matching, and F70 prices individual readout cells, not orbit geometry).

**I-3 · "A matched-block-L2 random orthogonal rotation matches or beats the real prompt displacement on both datasets" (`:504`) is a best-of-six selection.** Against the **primary** arm `common_displacement`, 4 of 6 HateMM rotations (`orthrot_17p6/29p1/60p4/72p7` at 0.8505/net +1) and 2 of 6 ZH rotations sit *below* it. Full verified spread: HateMM 0.8505–0.8692, ZH 0.8462–0.8974. **Repair:** state the rotation spread, or label the comparison "best of six" in the summary sentence as it is in the table.

**I-4 · "bit-identical … on 6/6 seeds" (`:177`, `TARGET_STATE.json:89`) overstates precision.** F113's banked arena artifacts store 4-dp values (`headspace_arena_hatemm_s0_OUT.json`: `head_deployed_acc: 0.8884`) against C02's `0.8884408602150538`. What is verifiable is identity **at the recorded 4-dp precision**. **Repair:** say "identical at the recorded 4-dp precision on 6/6 seeds".

**I-5 · F88 transcription slip.** `:551` gives HateMM 3/3-seed error invariance as *"(88–93 %)"*; F88 says **"(89-93%)"**. Propagated to `TARGET_STATE.json:222`. **Repair:** correct to 89–93 %.

**I-6 · OBS-1 attributes the validity gates to the wrong artifact.** `:183`: *"`C02_A0_DECISION.json` carries five named validity gates."* `C02_A0_DECISION.json` has no gates key at all (`bars, holm_family, interpretation_boundary, per_dataset, result_exists, run_id, schema_version, target_met, verdict`); the five gates live in `C02_A0_OUT.json` under `datasets.<ds>.gates`. Everything else in OBS-1 verifies exactly (4 with `pass: true` incl. per-seed ARENA2; ZERO_CONTRACT no `pass`; two `DOCUMENTARY_CITATION_NOT_COMPUTED`; ZH `banked_text_zero_rows: []`). **Repair:** change the path in `:183` and `TARGET_STATE.json:119`.

**I-7 · C14's landed status and its recorded prior status are both off-register.** `TARGET_STATE.json:327` sets `struck_gate0_2026_07_31_diagnostic_only_role_preserved`, while the disposition block at `TARGET_STATE.json:158` records `new_status: struck_gate0_2026_07_31` — a machine consumer reading the dispositions gets a different string than the registry. `TARGET_STATE.json:333` records `prior_status: "mechanism_gate_only_low_novelty"`, a string that appears nowhere else in the repository; C14 sat in `ordered_backlog` exactly as C05–C13 did. **Repair:** make the two status strings agree, and set C14's `prior_status` to `ordered_backlog` (its eligibility flag already carries the diagnostic-only fact).

**I-8 · C09's legality citation is one-sided in the way the record itself corrects elsewhere.** `:564` cites `progress.json:25` as affirmative legality. `LITSWEEP5_COMPLETENESS.md` §4(ii) is an on-point, post-ruling in-repo adjudication headed *"The contradiction (load-bearing)"* which states the ruling's two blessed classes — *"Trained SELECTOR on train labels"* and *"Trained symmetric RESHAPER on train labels"* — are *"both already measured dead"*, and that the ruling *"was written at lit-round-count 3 — before F75/F77/L1 sharpened the walls."* This does not defeat the legality verdict, but the record corrects exactly this one-sidedness at `:628` for `NCA_FORENSIC_RECON:110` and does not apply it here. **Repair:** cite LITSWEEP5 §4(ii) alongside `progress.json:25` and state that the legality holds while the ruling's viability premise is flagged stale in-repo.

**I-9 · C08 premise 3 is used past its source's written scope.** `:258-261` calls quoted-hate FPs *"measured at chance."* `ERRPAT_MHC-ZH` §5.4 is (a) TIER-2 CPU-re-mint proxy, (b) on the ZH **test** split, and (c) closes with *"Both are recorded as hypotheses the ZH test split is too small to settle, not as clusters"*; the cluster table at `:409` lists it as "not significant", not measured-null. The record names the MHC-ZH-only residual but not the tier or the source's own refusal to settle. **Repair:** add the tier/scope caveat, or rest the strike on premises 1–2 alone.

**I-10 · The paper-note keyword census is train-only but reads corpus-wide.** `:709` (and `TARGET_STATE.json:292`): *"49 distinct keywords over 254 occurrences (`em` 254 / `/em` 254, no other tag)"* — train-only; train+val is **50 keywords / 288 occurrences** (`em` 288 / `/em` 288). The table immediately above it is train **and** val. Same for "median text length of only 106 characters" (train; val is 108.5). **Repair:** label the census "train split".

---

### Downgrade verification (the record's headline claim)

- **C12 — justified.** F55's ban_scope and detail verify verbatim; "EN closed at all three levels" is unambiguously three levels *of the encoder-composition question* (frozen/F50, collapsed-adapted-deployed/B4-F53, healthy-img+adapted-text/F55). The recon's "MHC-EN is additionally closed at all three levels" was a real misread, and it was the leg that would have made the >=2-dataset arithmetic impossible. Downgrade correct, subject to H-3.
- **C11 — justified in direction, over-stated in support.** The disjunctive claim is verbatim in the registry; the thin-transcript cluster is real (p = 0.0048, robust at 0.0051) and the C02 fallback genuinely targets a different operator. Subject to H-2.
- **C10 — justified, but one clause is too generous.** EUM's ban does contemplate revival on three written preconditions, so `held_*` with those preconditions as the unblock is the faithful record. However `:343`'s *"C10 arguably **replaces** the bank object rather than adds rows"* is undercut by EUM's own measurement, quoted twenty lines later, that a flat unit bank puts only 10.6–11.3 distinct parent videos in a top-20 — i.e. it has more rows than the video bank. The `banned_constraints[3]` conditional in the unblock catches this, but the "arguably replaces" hedge should be withdrawn.

### Kind-of-record and reversibility

Reversibility language is present, correct and consistent on all ten entries (`"registry-level; reversible by a future user ruling. NOT a measured kill."`), the `what_this_reopen_does_not_do` block is accurate, the historical `ordered_backlog` is genuinely untouched, C09's prereg is a draft only, and nothing in C02 or C04 was modified. C05 (hold), C06 (gate), C13/C14 (strikes) are the correct kind of record for their evidence. C07 is not — see C-1.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 2 (raw)

Verdict: REVISE - 0 Critical / 3 High / 7 Important. All findings applied;
H-2 partly refuted on re-check (the phrase IS verbatim in directions_tried.json)
and corrected in the safer direction.

**Round-1 application audit (my first job).** Eleven of the fourteen findings are genuinely applied across all surfaces: C-1 (C07 is `held_lattice_delta_unwritten_reachability_unscreened` in the registry, the disposition block, `TARGET_FINDINGS.md` and `TARGET_LOOP.md`; "has been run and fails" survives only as an explicit withdrawal at `GATE0_REOPEN_2026-07-31.md:345`), H-1, H-2, H-3, I-1, I-4, I-5, I-6, I-8, I-9 all check out, and all ten registry `status` strings now match their `new_status` counterparts. **I-2, I-3, I-10 and the round-1 C10 hedge withdrawal were applied to the narrative record only** — see H-3. I-7 is half-applied — see I-4.

**Independent re-measurement.** I re-derived the census from the six gt files with no reference to either document. Every load-bearing `[M]` figure reproduces exactly: key-set `['id','label','text']` on all six; ws-only 39/9 and 0x4; ZH tags 243/34 with histogram `em` 254 / `/em` 254 and nothing else; 141/243 = 0.5802 vs 39/336 = 0.1161, base 0.3109; val 20/34 vs 8/44; 49 keywords / 254 occurrences train, 50/288 train+val; markup fraction median 0.0000, max 0.862069, 203 rows >10 %, markup-bearing median 0.2604; `<76`-char rows 161/221/92; HateMM medians 694.5 interp / 696 upper; p90 = 0.5051 (`lower`) / 0.5071 (`linear`) / 0.5155 (`higher`). The C01 arm table recomputes cell-for-cell from the stored confusions (HateMM rotation spread 0.8505-0.8692, 4/6 below primary; ZH 0.8462-0.8974, 2/6 below). §3.7's 6/6 4-dp identity is real. OBS-1 verifies. Asset claims verify: HateMM has only `-LoRA-curric` ro-caches, `MHC_zh` only `-LoRA`; `torch.save = _no_save` is exactly `headspace_mint.py:274-281`; all four `headspace_*.py` and all six arena `_OUT.json` exist. **No fabricated number found.**

**Downgrade verdicts.** All four are justified. C12: F55's ban_scope and detail verify verbatim and "all three levels" is unambiguously the three *encoder-composition* levels — decisive and correct. C10: EUM's `as of this recon` hedge is literally present, and EUM writes three revival preconditions, so HOLD-with-preconditions is the faithful kind of record where C14's ban (no revival path but a user ruling) is not; BSY's `bank-ADDITION` scoping is genuinely in its text (twice). C11: the registry claim is verbatim disjunctive. C07: F82's ban_scope does split vote-side from head-side. The three confirmed strikes (C08, C13, C14) rest on measured premise failures or registry text, not on bans.

## High

**H-1 · `GATE0_REOPEN_2026-07-31.md:417-426` — the C11 downgrade's load-bearing evidence is used past its written scope, in the exact way round-1's I-9 forced the record to fix C08's premise 3 from the same document.** ERRPAT_MHC-ZH §5.2 sits under `## 5. CONTENT COVARIATES (Tier 2)` (`:269`) and is computed on the **ZH test split, n = 149** (`:185`, `:272`) with a **CPU re-mint proxy head** (`:39`). Same tier, document and split as §5.4, which this record demotes to "a non-significant underpowered result". §5.2 is nonetheless recorded as **"measured positive"** with no qualifier — and it is the sole reason C11 is not struck. The record also drops `ERRPAT_MHC-ZH:307-308`: class-stratified, each half is underpowered (negatives p = 0.0506, positives p = 0.0668). Separately, the cross-dataset substrate figures (161/221/92 rows under 76 chars) apply a threshold derived from MHC-ZH's own quartiles to corpora whose medians are 696 and 369 characters. **Repair:** restate the §5.2 leg with tier/split/protocol/pooled-only limitation, note the `test_rule` tension, and replace or drop the 76-char cross-dataset census.

**H-2 · `TARGET_FINDINGS.md:75` and `TARGET_LOOP.md:1653` — "AGGNET held/carried the largest oracle ever measured on this object" is a misquote of F98, broadened in scope and landed as verified fact.** F98's actual text is *"C3 ENTERED THIS PREGATE WITH BY FAR THE LARGEST ORACLE CEILING ANY MEMBER OF THIS FAMILY HAS EVER HAD."* **Repair:** correct both surfaces to F98's own scope, and add the misquote to §3.

> **ADJUDICATION (partly refuted on re-check).** The phrase *"the LARGEST ORACLE
> CEILING EVER MEASURED ON THIS OBJECT"* **is verbatim** in
> `autoresearch/goal_mllm_plus3/state/directions_tried.json`'s F98 entry — round 2
> checked only `findings.jsonl`. So it is a real quote from one of two primary
> records, not a fabrication. However `findings.jsonl` F98 and
> `AGGNET_PREGATE_RECORD.md:678` are both narrower, and the conservative record is
> the family-scoped one. **Applied in the safer direction:** the family-scoped
> phrasing is now used in all landed surfaces and the disagreement between the two
> primary records is recorded in §6.

**H-3 · `TARGET_STATE.json` — four round-1 repairs were applied to the narrative record only; the machine-readable disposition block still carries the defective text, and on C10 it now directly contradicts the record.** C10 `gap_named` still asserts the "arguably REPLACES the bank object" hedge the record says is withdrawn; C06 `load_bearing_evidence` still carries I-3's unqualified best-of-six claim; C06 `why_gated_not_struck` still bases the warrant on the ensembling carve-outs (I-2 unapplied plus a live misstatement); the paper note `extent` still lacks I-10's split label. **Repair:** port the record's corrected text into all four JSON fields verbatim.

## Important

**I-1 · C07's unblock (c) imports F82's head-side clause onto C07's object**, the same over-application the Critical was about. F82's head-side clause governs a *"head-side graded auxiliary"*; C07 is a cone metric over a harm-act partial order, and no text shows those are the same object. **Repair:** state (c) as conditional.

**I-2 · `held_nonisomorphism_gate_unwritable` asserts unwritability from a three-source enumeration the record refuses to treat as exhaustive everywhere else** — the identical enumerative form the record calls "an EXTENSION" at C10 and disavows at C07. The disposition kind (hold) is right; the status string overclaims. **Repair:** rename to `held_nonisomorphism_gate_unwritten_as_posed`.

**I-3 · Two pinpoint citations do not exist, in a record whose subject is citation fidelity.** F99's actual citation for the exchange-rate claim is `:458-459`, not `:463`. `LITSWEEP8:218-222` is the §2.3 hubness table; the reduction argument lives at `:200-201` and `:268-273`. **Repair:** correct both.

**I-4 · `TARGET_STATE.json` — I-7's `prior_status_note` re-asserts as historical fact the very string round-1 found unsourced.** Repo-wide, `mechanism_gate_only_low_novelty` now appears only in round-1's own finding text and in this note; the candidate registry is entirely uncommitted. **Repair:** cite the artifact the string came from, or state it is not recoverable.

**I-5 · The C09 legality quote of `LITSWEEP3_DATA_CENTRIC.md:82` is truncated before its only qualifying clause** — *"…does not apply to the mechanism (though Wall-A still caps the achievable magnitude)"*, and the same section prices it at *"+3 any dataset: ~1-2%"* (`:94`) and *"at most +0.001-0.006"* (`:91`). This is the truncation pattern round-1 charged as High at C11, reproduced at the record's one promotion. Legality is unaffected; the prior is not. **Repair:** quote the parenthetical.

**I-6 · The MHC-EN entity histogram is a train+val occurrence count presented beside per-split row counts** (`&#39;` x51 / `&quot;` x22 / `&amp;` x18 is train+val; train-only is 43 / 17 / 16), inside a table certified "exact" — the mirror image of round-1's I-10. Related: the p90 convention label is self-contradictory across M-1, §7 and `TARGET_FINDINGS.md`. Numerically 0.5051 is numpy's `lower`, 0.5071 `linear`, 0.5155 `higher`. **Repair:** label the histogram and use one convention vocabulary.

**I-7 · The F82 headwind is quoted with the clause that limits its dataset coverage elided, and the GRADEDLBL ceiling's resolution is not stated.** F82's ban_scope ends *"; HateMM out of scope (no Offensive class)"*. And `GRADEDLBL_PREGATE_RECORD.md` states the ceiling is a **dev-label gold cheat on dev splits** (n = 80 EN / n = 78 ZH) whose ZH `+0.0256` is **2 dev items** (`:137`), and pre-declares at `:72-75` that the oracle *"does NOT bound the head's representation-reshaping"*. **Repair:** restore both.

## Not raised, checked and cleared

The three confirmed strikes (C08's premises 1-2 are genuine measured data failures with a named residual and unblock; C13 rests on a direct measurement with a short inference and an explicit "performance route only" scope; C14 rests on registry text alone with TVB correctly disowned); reversibility language, present and uniform on all ten entries; `ordered_backlog` genuinely untouched; `C09_A0_PREREG_DRAFT.md` present and described as a draft; F82/F55/F60/F80/F70/EUM/BSY/TVB/F78/F88/F99/F106/F107/F112/F113/F114/`banned_constraints[1,3,5,6,10]`/`hard_constraints` quotes all verbatim-faithful; LBOP verified as a real, distinct candidate at `research-wiki/TARGET_GATE0_ITER6_LITERATURE.md:246,271,284-288,444` and `TARGET_REVIEW_RAW.md:740-752`, with its order sourced from a label-blind MLLM; `prep_mhc.py:76` correct (the recon's `:72` was wrong); EDCM `+0.0273/+0.0394` and `+0.0380/+0.0444` exact at `TARGET_FINDINGS.md:9`; EUM's 83 %-contiguous-block figure exact; `generate_VideoMLLM_embedding_readout_HF.py` CELLS table confirms the `ow_` prompt/span confound.



==============================================================================
# C04-A0T-SMALL-v1 implementation-v7 — five independent code/resource review rounds
# Appended 2026-07-31. Verdicts: REVISE 2C/2H/3I -> REVISE 0C/2H/3I -> REVISE 0C/1H/3I
#                                 -> REVISE 0C/1H/0I -> GO 0C/0H/0I
==============================================================================


----- BEGIN refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_REQUEST.md (sha256 a5d026197fb3664b01c78a1a95ad710a44abaf08412d6b9679813147b3ec0b7f) -----
# C04-A0T-SMALL-v1 v7 — Fresh Independent Static Review Request

Date: 2026-07-31
Stage: `CPU_PREFLIGHT` code/resource review of implementation-v7
Requested verdict granularity: `GO / REVISE` with `nC / nH / nI` severity counts
Execution authority conferred by this review: **none**

You are a fresh independent reviewer. You have not seen the reasoning that
produced these files and you should not seek it out. Review the frozen bytes
against the contract below.

## Absolute reviewer boundary

- **Do not submit, hold, release, requeue or cancel any SLURM job.** `squeue`
  and `sacct` read-only are permitted.
- **Do not run any GPU, teacher, model-weight or frame-decode work.**
- **Do not create, modify or delete any file under `/data/jehc223/RGCL`** other
  than the single review file you are asked to write.
- **Do not materialize any dataset label value.** If you read either ASR file,
  decode only `id`, `window_text` and `language`. HateMM identifiers are
  themselves label-bearing; treat them as identifiers, not as labels to reason
  from.
- Work from a scratchpad outside the repository with `PYTHONDONTWRITEBYTECODE=1`
  set, so no `.pyc` is written anywhere. Prefer static `ast` parse or a
  byte-identical scratchpad copy over importing a module from its repository
  path.
- `artifacts/c04/a0t_small_v1_impl_v7/` must not exist when you finish, and you
  must not create it.

## Files under review (frozen; re-verify every hash against disk)

| File | SHA-256 |
|---|---|
| `scripts/analysis/c04_a0t_small_v1_v7_common.py` | `5fc5259ec4a98b47fa95851272b43fd6f8bdd7767d2c65a50d6c09889ebe2690` |
| `scripts/analysis/c04_a0t_small_v1_v7_preflight.py` | `ecdc8568dfab0a50e5f6701fba7c09fe939fcdba3af12e35243f6d11e9af873a` |
| `scripts/analysis/c04_a0t_small_v1_v7_gpu_ledger.py` | `944023b3aafc04dfdeee59fe920ed77b4ef882c4b56c39968bc0f12ee96758e3` |
| `scripts/analysis/c04_a0t_small_v1_v7_producer.py` | `7a3c3a794454c5856238234cb305a410b831ca624cb0e0e8391694992ebeae26` |
| `scripts/wrappers/c04_a0t_small_v1_v7_preflight.sh` | `914dd5df80ab45d5aa4102e6f4718c31282c9e48b9b7bee96e11dc6d54bf59b0` |
| `scripts/wrappers/c04_a0t_small_v1_v7.sh` | `645e501140690cece68e438422bfdc45005af1a5f392c70a86dc7c2e3713df5c` |
| `scripts/wrappers/c04_a0t_small_v1_v7_reconcile.sh` | `7af043225285f129e59ffd385782bed214e96cd39a296ecdeeb2a7f5fd16c8e0` |
| `scripts/slurm/c04_a0t_small_v1_v7_preflight.sbatch` | `919316c70ae79d9f019de6952acc761eaec065f60afec17da4dee573613ded39` |
| `scripts/slurm/c04_a0t_small_v1_v7.sbatch` | `00ddeeed57d1f585a7305738b259fcb6604eb8c545082650406ea6fe403aeacc` |
| `scripts/slurm/c04_a0t_small_v1_v7_reconcile.sbatch` | `d8f634ec88d762be8e797edf3bcc0433f033cbc1feb3c19faced71bde5ef44cd` |
| `schemas/c04/c04_a0t_small_v1_v7_prompt_record.schema.json` | `541d02455aee3af9293e978a8628f438bd78feca08d8b54f442e1ac8c77084f3` |
| `schemas/c04/c04_a0t_small_v1_v7_canonical_record.schema.json` | `bacbddaeba13806829e5dffa09fcab55a76a89420f03310a495ac6cccde578b3` |
| `schemas/c04/c04_a0t_small_v1_v7_stage_authorization.schema.json` | `2edac849da8a3bf4ebd3ad82e39de6f6d22e76bcfed3d8688379f349e08f10a8` |
| `schemas/c04/c04_a0t_small_v1_v7_payload_review.schema.json` | `7edebdfe81bb5180d9968d91ee797c6cf0ba14bbf71ec585d0b75a60d9cad81a` |
| `schemas/c04/c04_a0t_small_v1_v7_resource_final_state.schema.json` | `e2f9dca545874a4b2c8e65932b987b068563da560937965e11f684be255cd53d` |
| `configs/c04/c04_a0t_small_v1_v7.json` | `0af5b6bdc12eb641571de199b02530d31277343b666458aebb0f36265086dfcd` |
| `artifacts/c04/campaign/gpu_ledger.json` | `fc6ca12c32427625d0b80c16b7802ef9a574ced0dbf0288edc3938d217267414` |

## Round 2 — what changed since the round-1 review

Round 1 returned `REVISE (2C / 2H / 3I)`. All eight findings were accepted and
repaired; nothing was argued away. Re-test each repair independently, and treat
the round-1 findings as **unproven again** until you have re-derived them.

- **C-A** — the reconciler read `seal/provisional_gpu_usage.json` with strict key
  equality against a key set the writer had outgrown. Both halves now build and
  validate through single shared constants, and a fixture asserts they agree.
  Check that the two really cannot drift, and that the fixture is not vacuous.
- **C-B** — `proposition_cosine` could exceed the canonical schema's `maximum: 1`
  by one ulp exactly when the two prompt forms agreed. Verify the clamp is
  correct, that it changes no comparison outcome against the 0.80 agreement
  floor, and that the fixture proving the bound is real actually fails without it.
- **H-A** — the campaign accumulator's write side was reachable only after a
  successful seal. There is now a distinct `campaign-record` ledger mode keyed
  off `resource/allocation_claim.json`, run before `reconcile-terminal` in the
  same CPU allocation. Determine whether **every** path that burns GPU-seconds
  now reaches it, including exit 40, a watchdog TERM, an OOM and any HALT.
- **H-B** — the in-job guard is now anchored to the allocation entry marker's
  `/proc/uptime` reading and held `resources.guard_item_margin_seconds` ahead of
  the wrapper `timeout`, and no longer consults `SLURM_JOB_START_TIME` at all.
  Verify the guard now always leads the wrapper, that the margin covers a
  worst-case item, and that a breach record is therefore reachable.
- **I-C1** — the accumulator is phase-scoped. Confirm the effective ceiling is
  the tighter of the phase and aggregate caps, that it is 7200 s today, and that
  the phase cannot advance without an authorization the code refuses to invent.
- **I-C2** — assess whether the H-B margin actually buys enough headroom that a
  terminal sacct elapsed can no longer approach the hard 7200 s ceiling.
- **I-C3** — the CPU preflight now round-trips records against the downstream
  contracts that read them. Judge whether the coverage is real and whether the
  blind spot is genuinely closed, or merely narrowed.

Also re-check the two round-1 non-blocking observations: the pre-model-load
containment pass now assembles messages and extracts through the same path as
the per-forward call site, and `maps.expected_hashes` remains protected only by
inclusion in the contract hash.

## Round 3 — what changed since the round-2 review

Round 2 returned `REVISE (0C / 2H / 3I)` and confirmed both round-1 Criticals
closed. All five new findings were accepted and repaired. Re-derive each.

- **H-1** — the accumulator could brick itself: an over-cap row was written and
  *then* raised, after which every later load raised forever. Loading is now
  accounting and never refuses; every rejecting check runs before the write;
  an over-ceiling total is recorded with a flag, and refusal happens on the next
  allocation. Verify no code path can raise after the file is replaced, and that
  a recorded over-run really does refuse the next reservation.
- **H-2** — the reader now imports and uses `PROVISIONAL_USAGE_KEYS` and
  `BUDGET_GUARD_KEYS` rather than restating them. Verify one-sided drift is now
  impossible by construction, not merely detected.
- **I-1** — `cosine` moved into the module the CPU preflight actually exercises,
  with fixtures over the clamp. Verify that deleting the clamp now turns a
  fixture red.
- **I-2** — a new `resources.guard_seal_reserve_seconds` is required before the
  post-loop canonicalization and seal phase begins. Verify the reserve is
  actually large enough for that phase and that an insufficient remainder stops
  cleanly with a breach record instead of a SIGTERM.
- **I-3** — `campaign-record` now falls back to `allocation_entry_marker.json`
  when no claim was published. Verify that marker exists on every path where a
  GPU allocation was entered, and that the fallback cannot fabricate a row.

Treat every claim above as unproven until you re-derive it.

## Round 4 — what changed since the round-3 review

Round 3 returned `REVISE (0C / 1H / 3I)` and confirmed every round-1 and
round-2 finding closed. All four new findings were accepted and repaired.

- **H-1 (round 3)** — the GPU wrapper `mkdir`-ed the no-clobber namespace and
  wrote the entry marker before any code read an authorization flag. It now
  carries a `jq -e` authorization gate and a frozen-preflight-manifest existence
  test **ahead of** the `mkdir`. Verify no irreversible step precedes the gate,
  including via the EXIT trap armed earlier in the script.
- **I-1** — `watchdog_reserve_seconds` is 120 → 300, and an over-cap terminal
  elapsed is recorded and flagged rather than refused (bounded by a new
  `TERMINAL_SECONDS_HARD_MAX`; the final-state schema was widened and gained
  `terminal_elapsed_exceeded_cap`). Verify the headroom arithmetic and that a
  marginal over-run is now publishable.
- **I-2** — `verify_reconciliation_lineage` has a seal-free tail, and a terminal
  resource state is published on every terminal path with `seal_published`
  recorded. Verify the exit-40, 124/137/143, OOM and post-claim-HALT paths all
  reach a published final state, and that the seal-free path cannot forge one.
- **I-3** — `GPU_LEDGER_KEYS`, `ALLOCATION_CLAIM_KEYS` and
  `RESOURCE_TICKET_KEYS` are shared constants asserted on both the writer and
  reader side, and full `prompt_record` / `canonical_record` round-trip fixtures
  were added (four reliability regimes plus the zero-frame case, with a
  non-vacuity check). Verify no writer/reader key-set contract remains
  duplicated, and that the new fixtures fail when a record is malformed.

Treat every claim above as unproven until you re-derive it.

## Round 5 — what changed since the round-4 review

Round 4 returned `REVISE (0C / 1H / 0I)` and confirmed every earlier finding
closed except one it relocated. That single High is repaired.

- **H-1 (round 4)** — `stage_authorization.schema.json` still pinned the
  reconciliation `payload_binding.provisional_gpu_usage_sha256` to 64-hex while
  the code and the final-state schema had moved to admitting a
  `NO_SEAL_PUBLISHED` sentinel, so the seal-free reconciliation path was
  unsatisfiable and no terminal resource state could ever be published on a
  breach, watchdog-kill, OOM or post-claim-HALT path. The `anyOf` is now
  mirrored into the stage-authorization schema, and two fixtures round-trip a
  full reconciliation manifest through `validate_schema` in **both** seal
  regimes while confirming a foreign pin is still refused.

Verify the two schemas that describe this one field now agree, that both
regimes are genuinely satisfiable end to end against the real reader, and that
no third description of the same field remains anywhere.

## Authority chain (frozen, read-only, not under review)

- `refine-logs/C04_USER_AMENDMENT_V2.md` — the user-approved bounded-teacher
  amendment. This is the binding resource and scope contract.
- `refine-logs/C04_REFINED_PROPOSAL_V4.md`, `refine-logs/C04_EXPERIMENT_PLAN_V4.md`,
  `refine-logs/C04_V4_DESIGN_REVIEW.md` — the frozen design (`GO 0C/0H/0I`).
- `refine-logs/C04_A0T_SMALL_V1_V6_PAYLOAD_REVIEW.md` — the predecessor payload
  review (`GO 0C/0H/3I`) whose three Important findings v7 exists to close.
- The v6 implementation set (`*_v6_*`) and `artifacts/c04/a0t_small_v1_impl_v6/`
  are the predecessor. **They must not be modified.** Confirm they are not.

## What v7 claims, and what you must independently test

v7 is a full namespace rebuild. It claims to be byte-identical to v6 modulo the
`v6`→`v7` version-token rename, plus exactly four repairs. Test each claim; do
not accept any of them on assertion.

### Claim 0 — no scientific semantic changed

The selection rule, tag, suffix, per-dataset count, prompt **text** and its four
SHA-256 values, prompt-hash rule, frame rule, transcript normalization and cap,
reliability thresholds and the five-rate KILL taxonomy, fallback semantics,
role/JL map construction, resource caps and every authorization flag must be
unchanged in meaning from v6.

Strongest available test: re-derive the 200+200 selection from the v7 frozen
rule and confirm it reproduces the **v6 frozen allowlists** exactly. Also
recompute the four prompt hashes from the v7 sources and confirm they equal the
v6 frozen values. Diff the v7 tree against v6 modulo the version token and
account for **every** residual changed line.

### Claim C-1 — the prompt renderer

v6 rendered prompts with `PROMPTS[form].format(transcript=...)`. v7 replaces
that with a dedicated renderer. Determine for yourself:

1. whether the v6 form could ever have succeeded, given the frozen prompt text;
2. whether the v7 renderer produces exactly the substitution the frozen design
   intends, with every other character of the template literal;
3. whether any prompt byte changed (the four frozen hashes are the test);
4. whether any `.format(transcript=` call site survives anywhere in v7;
5. whether the guard rails inside the renderer are non-vacuous.

### Claim I-1 — teacher-visible containment is a fail-closed precondition

The amendment requires the teacher to see no label. v7 must make "no identifier
and no label-bearing string reaches any teacher-visible field" a **checked**
precondition that fires **before any teacher forward**, not a property of prompt
assembly. Test:

1. that the check runs before the model is loaded, and again per item before
   each forward;
2. that it is strict in both directions — an unrecognised message role, content
   part, payload type, frame count, or a frame carrying a string/path must
   raise, so a future edit adding a teacher-visible field cannot escape;
3. what exactly is banned, and whether the ban is wide enough (consider
   cross-item leakage, NFKC and case folding);
4. that it does **not** false-positive on the real 400 transcripts — verify this
   yourself, label-blind;
5. that the check cannot pass vacuously.

State explicitly in your review whether the HateMM ID-label asymmetry is handled
correctly: HateMM identifiers are `hate_video_*`/`non_hate_video_*` and therefore
*are* the label, so the sealed ID-only allowlist gives label containment for
MHC-ZH only.

### Claim I-2 — the selection self-test is a known-answer vector

v6's `selection_deterministic` compared a pure function to itself. Confirm the
replacement is a genuine known-answer test: that the pinned digests are literals
independent of the module's own code path, and that mutating the tag, the
suffix, the dataset term or the identifier breaks it.

### Claim I-3 — both ceilings are machine-checked and fail-closed

**Tranche ceiling (2 GPU-hours, 7200 s).** One absolute deadline computed once
at job start and never recomputed. The guard may only ever STOP work **before an
item begins** — it must never truncate, shorten or alter an output. A breach must
produce an accounting-only record (no metric, no teacher output, no reliability
rate, no CONTINUE/KILL verdict) and a **distinct** exit code, propagated
distinctly by the wrapper. Verify by tracing where the guard is and is not
called, and what state a breach leaves on disk.

**Campaign ceiling (8 GPU-hours, 28800 s).** The amendment's aggregate ceiling
covers "every GPU-second consumed by the first tranche and any later C04
extraction/adaptation job". Verify that the accumulator is checked at job start
**before** the single-use resource ticket is consumed, that a missing, malformed,
foreign, mis-capped or chain-broken ledger halts rather than defaulting to zero,
that no stage can create or reset it, that the write side is idempotent and
sacct-derived, and that its opening zero is evidence-backed rather than assumed
(check `sacct` yourself for jobs 13805 and 13840).

## Additional checks

- The `--time` directive, job arrays, dependencies, chained submission, release
  and resubmission must be absent from every v7 wrapper and sbatch file. The only
  `--gres` may be in the GPU producer sbatch. The preflight sbatch must request
  no GPU.
- Resources must be exactly 1 GPU / 8 CPU / 64 GB.
- No OCR entrypoint, no network or external API client, no dev/test path, no
  cross-dataset path, no label reader.
- Every authorization flag in the config must be in the correct pre-review
  state, and every review pin that is not yet earned must be a sentinel that the
  code rejects.
- Look specifically for the failure family this project has now hit three times:
  **an irreversible resource consumed before the check that would reject the
  run.** Enumerate any remaining instance you find.

## Deliverable

Write exactly one file, `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md`,
containing your verdict (`GO` or `REVISE`), severity counts, and one section per
claim recording what you recomputed and what you found. Report findings as
Critical / High / Important with the reasoning that makes each one actionable.
Do not soften a finding to reach `GO`, and do not accept a claim you did not
test.

----- END refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_REQUEST.md -----


----- BEGIN refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md (sha256 54ecfd731b73932da6b1316be6d271ce5190e75b98345551957baa898aa81e6f) -----
# C04-A0T-SMALL-v1 v7 — Fresh Independent Code/Resource Review

Reviewer: fresh independent static reviewer (no prior exposure to the authoring reasoning)
Date: 2026-07-31
Stage reviewed: `CPU_PREFLIGHT` code/resource review of implementation-v7
Execution authority conferred by this review: **none**

## Verdict

**REVISE — 2 Critical / 2 High / 3 Important (2C / 2H / 3I)**

Both Criticals are instances of the exact failure family the request asked me to
hunt: **an irreversible resource consumed before the check that would reject the
run.** In v7 the defect has moved one stage past the v6 render defect — the run
now gets *further*, spends the whole A100 allocation, and only then hits a gate
it cannot pass. Neither Critical is reachable by the CPU preflight self-test, so
neither would surface before the GPU is spent.

Everything the request asked me to verify about Claim 0, Claim C-1 and Claim I-2
reproduced exactly. Claim I-1 is substantially delivered. Claim I-3 is where the
findings concentrate.

## Method and reviewer-boundary compliance

- No SLURM job submitted, held, released, requeued or cancelled. `sacct` and
  `squeue` used read-only only.
- No GPU, teacher, model-weight or frame-decode work. Model *weights* were never
  read or loaded; only the 350-byte `preprocessor_config.json`,
  `generation_config.json` and `chat_template.json` metadata files were opened,
  plus a static read of the installed `transformers` source.
- No file under `/data/jehc223/RGCL` was created, modified or deleted other than
  this review file. Verified after every write-capable probe:
  `artifacts/c04/campaign/gpu_ledger.json` still hashes to
  `e84517d69ce7aa9a87c600b920882b1e19f118385fb571cc38acd544560dd14e`.
- `artifacts/c04/a0t_small_v1_impl_v7/` does not exist and was not created.
- No dataset label value was materialized. The ASR files were decoded only
  through the frozen `project_train_asr_line` projector (`id`, `window_text`,
  `language`). HateMM identifiers were treated as contained identifiers: they
  were compared and hashed, never printed and never reasoned from as labels.
- All work in `…/scratchpad/review-r1`, outside the repository, with
  `PYTHONDONTWRITEBYTECODE=1` on every invocation. `common.py` was imported from
  a byte-identical scratchpad copy, never from its repository path.

### Hash verification (all 17, re-verified against disk)

Every pinned SHA-256 in the request table matches disk exactly. All 15 entries of
`configs/c04/c04_a0t_small_v1_v7.json → implementation_hashes` match disk and
match the request table. All 15 entries of `frozen_design_hashes` verified with
`sha256sum -c`: 15/15 OK.

### v6 predecessor unmodified

The v6 sources and `artifacts/c04/a0t_small_v1_impl_v6/` are untouched. Strongest
available positive evidence: the v6 `freeze/preflight_manifest.json` is
self-consistent (`payload_sha256` reproduces) and **all 14** of its
`staged_output_hashes` still verify byte-for-byte against the v6 artifacts on
disk. All v6 artifacts carry a single mtime of `2026-07-31 05:11` (job 13840),
predating every v7 source file (`20:4x`).

---

## Claim 0 — no scientific semantic changed: **CONFIRMED**

### Selection re-derived from the v7 frozen rule reproduces the v6 frozen allowlists exactly

I re-read both train ASR files through the v7 projector, ranked by the v7
`selection_digest` with the v7 tie-break, and took the first 200:

| dataset | train N | recomputed order == v6 frozen allowlist | sha256(ordered id list) |
|---|---|---|---|
| HateMM | 744 | **True** | `6ff2917b4eaba00c7b92828be6f614cf5eb1b3c1fe8f728aabbd88eaadc76a5a` (identical for both) |
| MHC_zh | 579 | **True** | `2688edcb5a4f0beb228ee7d02c4bb47a20bf462a362f963ca3438e532cef86d3` (identical for both) |

Also reproduced, per dataset: every stored `selection_sha256` in the v6
allowlist (200/200), contiguous ranks 0..199, and — against the v6
`source_manifest.json` — all 200 `transcript_sha256` values and all 200
`transcript_scalar_count` values. That last check independently pins the
transcript normalization and the head/tail cap as unchanged.

### Prompt hashes recomputed from the v7 sources

```
system   1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048
A        cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b
B        9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314
combined a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a
```
Equal to the in-module fixture literals **and** to the values in the v6 frozen
artifact `artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json`. No
prompt byte changed.

### Version-token-normalized tree diff — every residual line accounted for

I copied both trees to the scratchpad, applied `s/v6/vX/g; s/V6/VX/g` and
`s/v7/vX/g; s/V7/VX/g`, and diffed file-by-file:

| file | changed lines | accounted for by |
|---|---|---|
| all 5 schemas | 0 | — |
| `*_preflight.sh`, `*_preflight.sbatch`, `*.sbatch`, `*_reconcile.sh`, `*_reconcile.sbatch` | 0 | — |
| `preflight.py` | 21 | I-3 only (campaign cap/path assertions + headroom call) |
| `gpu_ledger.py` | 68 | I-3 only (campaign imports, headroom at claim, `record_campaign_gpu_spend`) |
| `*_v7.sh` | 18 | I-3 only (exit-40 branch + breach-record jq + zero-exit breach guard) |
| `common.py` | 468 | C-1 `render_prompt` + I-1 containment block + I-3 campaign ledger + I-2 fixture replacement + new fixtures. **Pure additions** apart from the single I-2 fixture swap. |
| `producer.py` | 386 | C-1 call-site swap at `build_messages` + I-1 precondition/per-item check + I-3 `BudgetGuard`/breach + campaign assertions |
| `config.json` | 132 | version tokens, `v7_scope` prose, `campaign_aggregate_cap_gpu_seconds`, `paths.campaign_gpu_ledger`, `paths.budget_breach`, refreshed `implementation_hashes` |

Nothing outside the four declared repairs changed. `SYSTEM_PROMPT`,
`_SCHEMA_TEXT`, `PROMPT_A`, `PROMPT_B`, `SELECT_TAG`, `SELECT_SUFFIX`,
`SELECT_N`, `NUM_FRAMES`, the transcript cap/head/tail/separator, the confidence
and cosine thresholds, the five-rate KILL taxonomy, `render_slot`,
`build_slot_reliability`, `materialize_role_map`, `dense_rademacher_payload`,
the resource caps and every authorization flag are untouched.

### Self-test surface

`self_test_fixtures()` now returns **37** checks (v6: 25). All 37 pass on the
frozen bytes.

---

## Claim C-1 — the prompt renderer: **CONFIRMED, and the v6 form provably could never have succeeded**

1. **Could the v6 form ever have succeeded?** No. Executed against the frozen
   template text:
   ```
   PROMPT_A.format(transcript="X")  ->  KeyError: '"source_relation"'
   PROMPT_B.format(transcript="X")  ->  KeyError: '"source_relation"'
   ```
   and even supplying that field yields `KeyError: '"S"'` from the confidence
   literal. There is no argument set under which `str.format` returns; the
   defect is unconditional on both forms.
2. **Does v7 produce exactly the intended substitution?** Yes. For both forms and
   for a transcript containing non-ASCII, CJK, a newline and a literal
   `{braces}` sequence: `render_prompt(form, t) == PROMPTS[form][:-len("{transcript}")] + t`,
   with the prefix byte-identical and the tail byte-identical. A transcript that
   itself contains the literal `{transcript}` is substituted once only (the
   `count == 1` guard is on the *template*, and `str.replace` on the single
   trailing occurrence). The module-level templates are not mutated.
3. **Did any prompt byte change?** No — see Claim 0; all four hashes match the v6
   frozen artifact.
4. **Any surviving `.format(transcript=` call site?** None in the v7 tree. The one
   surviving textual occurrence is inside the deliberate regression fixture
   `str_format_render`, which asserts that the v6 form still raises `KeyError`.
5. **Are the renderer's guards non-vacuous?** Mostly yes, with one observation.
   `prompt_form not in PROMPT_FORMS`, `not isinstance(transcript, str)`,
   `count != 1` and `not endswith(placeholder)` are all genuine and the fixture
   `prompt_render_rejects_unknown_form_and_non_string` exercises two of them.
   The two post-hoc guards are weaker: `rendered.startswith(prefix)` is a tautology
   given `str.replace` on a unique terminal placeholder, and
   `rendered.endswith(transcript)` is vacuously true for an empty transcript
   (measured: an empty transcript renders and passes). Neither is load-bearing —
   the real protection is the `count == 1` + `endswith` pair on the template — so
   this is an observation, not a finding.

---

## Claim I-1 — teacher-visible containment: **substantially CONFIRMED**

1. **Runs before the model is loaded?** Yes.
   `assert_teacher_visible_precondition(inputs)` is `producer.py:1624`;
   `Qwen2_5_VLForConditionalGeneration.from_pretrained` is `producer.py:1633`.
   Measured on the real tranche: **800** renderings checked before model load
   (400 identifiers × 2 forms), ban list **402** tokens.
   The check is repeated per item at `producer.py:1656`, inside `one_forward`,
   after `build_messages` and before `apply_chat_template`/`generate`.
2. **Strict in both directions?** Yes. `teacher_visible_texts` raises on: a
   message list of the wrong length, unexpected message keys, an unexpected
   role, non-list/empty content, a content part that is not a dict or lacks
   `type`, a text part with extra keys or a non-string body, a video part with
   extra keys, a frame list whose length ≠ 8, **any frame that is a `str`,
   `bytes`, `bytearray` or `Path`**, any unknown content `type`, and a final
   census requiring exactly one video part and exactly two text parts. A future
   edit that adds a teacher-visible field cannot pass. Six of these are pinned by
   fixtures.
3. **What is banned, and is it wide enough?** All 200 HateMM + 200 MHC-ZH
   selected identifiers **plus** the two HateMM label-bearing prefixes, and both
   datasets' identifiers are banned in both datasets' prompts, so cross-item
   leakage is refused as firmly as self-leakage. Matching is done over
   `{token, NFKC(token), casefold(NFKC(token))}` against both `NFKC(text)` and
   `casefold(NFKC(text))`, so a fullwidth or case-altered identifier is caught.
   The wider protection is the equality `texts == [SYSTEM_PROMPT, render_prompt(form, transcript)]`:
   the only variable content reaching the teacher is the transcript, so the
   amendment's broader ban (prediction, neighbor, rank, margin, error status,
   dataset statistic, fold role, intended use) is structurally satisfied rather
   than enumerated. I also confirmed the pinned `chat_template.json` renders a
   video part as `<|vision_start|><|video_pad|><|vision_end|>` and discards the
   `video` value, so no path or identifier can leak through templating.
4. **Does it false-positive on the real 400 transcripts?** **No — 0 of 800.**
   I ran the frozen guard label-blind over all 400 selected transcripts × both
   forms: zero rejections. As a robustness margin I also scanned the same
   402-token ban list against **all 1323** train transcripts (744 + 579): **0**
   rows contain any banned token. Shortest banned token is 11 characters
   (`hate_video_`), so accidental substring collision is not a live risk.
5. **Can it pass vacuously?** Not at the per-forward call site. `if not forbidden
   or video_id not in forbidden: raise` makes an empty or incomplete ban list a
   halt, and the fixture `teacher_visible_unbanned_identifier_rejected` proves an
   identifier absent from the ban list is rejected. **Observation:** in the
   pre-model-load pass the `texts` argument is constructed inside
   `assert_teacher_visible_precondition` from the same `render_prompt` call the
   assertion compares against, so *that* half of the predicate is self-comparing
   there; the pre-load pass effectively verifies only "no banned token in the
   transcript". The message-assembly half becomes non-vacuous only at
   `producer.py:1656`, which does run before every forward, so the stated
   requirement is met — but the two passes are not equally strong and the review
   record should say so.

### The HateMM ID-label asymmetry — stated explicitly, as requested

**The asymmetry is handled correctly, and the code says so in the right place.**
Every HateMM training identifier is `hate_video_*` or `non_hate_video_*`; the
identifier *is* the binary label. MHC-ZH identifiers are opaque BiliBili `BV`
codes carrying no label information. Therefore the sealed ID-only allowlist
delivers label containment **for MHC-ZH only, and none at all for HateMM**.
`LABEL_BEARING_ID_SUBSTRINGS` encodes exactly this (`("hate_video_",
"non_hate_video_")` for HateMM, empty tuple for MHC-ZH), the `common.py` comment
block states the asymmetry, and `config.json → v7_scope.I1_teacher_visible_containment`
records it. The consequence is correctly drawn: for HateMM, selection
label-blindness is established by *hash reproduction of the selection rule*
(which I independently reproduced above), and teacher label-blindness by this
*runtime* check — not by the allowlist.

---

## Claim I-2 — the selection self-test is a known-answer vector: **CONFIRMED**

The v6 fixture `selection_digest("HateMM","x") == selection_digest("HateMM","x")`
was a tautology. The v7 replacement pins two literals. I recomputed both from
first principles, outside the module, with a hand-written concatenation:

```
sha256(utf8("C04-A0T-SMALL-v1" + "HateMM" + "c04-known-answer-vector" + "20260729"))
  = 871e0363e1b01a823f09a5a0bb9187749da74f1dfe8e454a733e21c218f6a384   [matches]
sha256(utf8("C04-A0T-SMALL-v1" + "MHC_zh" + "c04-known-answer-vector" + "20260729"))
  = 41bfb637f5cbb26bb0b9edfd44d19a3775d8df1712f9b49bde667863fbd37134   [matches]
```

The digests are literals independent of the module's own code path. Mutation
sensitivity, measured: tag → breaks; suffix → breaks; dataset term → breaks;
concatenation order → breaks; identifier → breaks. The companion fixture
`selection_dataset_and_id_sensitivity` additionally requires three distinct
digests. The chosen identifier is synthetic and belongs to neither dataset, so
the fixture pins the rule without naming a label-bearing real id — a good choice.

---

## Claim I-3 — both ceilings machine-checked and fail-closed: **PARTIALLY DELIVERED**

### Tranche ceiling (7200 s) — guard logic correct, guard *ordering* not enforced

Traced: `BudgetGuard.at_job_start` is constructed exactly once, in
`verify_claimed_resource`; the deadline is stored and never recomputed
(`check()` only compares, `accounting_snapshot()` only reports). There are
exactly two `deadline_check` call sites — `producer.py:1705` at the item
boundary, before frame-pack creation, and `producer.py:1651` inside
`one_forward`, before `build_messages`. The guard is never invoked during or
after a unit of work and has no path that truncates, shortens or rewrites an
output. `BudgetDeadlineReached` is caught only by the outer handler, which
publishes an accounting-only record and returns 40.

The breach record was inspected field by field: it carries lineage, job id,
terminal state, exit code, the two caps, the guard accounting snapshot,
per-dataset completed counts, teacher-call and frame-pack counters,
`outputs_truncated_or_altered: 0`, `seal_published: false`,
`no_scientific_verdict_is_published_by_a_budget_breach: true`. **No metric, no
teacher output, no reliability rate, no CONTINUE/KILL verdict.** Exit code 40 is
distinct from the watchdog codes and the wrapper propagates it distinctly
(dedicated branch, jq-verifies the record, exits 40) and also refuses a zero-exit
run that left a breach record behind. On-disk state after a breach: per-item
checkpoint JSONs and frame packs intact, breach record present, GPU ledger in
`EXIT_RECORDED_PENDING_SACCT`, **no seal**.

The defect is in the *ordering* between this guard and the wrapper's `timeout` —
see **High H-B**.

### Campaign ceiling (28800 s) — read side correct, write side unreachable

**Ordering.** `assert_campaign_aggregate_headroom` is called from
`validate_gpu_environment`, which is the first statement of `claim()`, i.e.
before `create_entry_marker`, before `verify_gpu_lineage`, and well before the
allocation claim and the single-use ticket-consumption record are published. The
check is genuinely before the ticket is consumed. It is also called by the CPU
preflight before the no-clobber namespace is materialized, and by the producer
before any model or data work.

**Fail-closed matrix**, executed against scratchpad copies (repo untouched):

| mutation | result |
|---|---|
| ledger absent | HALT `campaign ledger is absent` |
| aggregate ≠ Σ rows | HALT `payload mismatch` |
| `payload_sha256` tampered | HALT `payload mismatch` |
| foreign `schema_version` | HALT `foreign campaign ledger schema` |
| foreign `run_id` | HALT `foreign campaign ledger run id` |
| cap raised to 999999 | HALT `cap is not the amendment cap` |
| cap lowered to 7200 | HALT `cap is not the amendment cap` |
| row chain break (`previous ≠ GENESIS`) | HALT `chain break` |
| well-formed 25000 s already spent | HALT `would take the C04 campaign to 32200s` |
| well-formed 21601 s already spent | HALT `would take the C04 campaign to 28801s` |
| well-formed 20000 s already spent | accepted (27200 ≤ 28800) — correct |

It never defaults to zero. **No stage can create or reset it**: the only writer
is `append_campaign_gpu_job`, which calls `load_campaign_gpu_ledger()` first (so
it cannot bootstrap from nothing), and the preflight explicitly verifies the path
rather than staging it — the campaign path lies outside `ARTIFACT_ROOT` and would
be rejected by the staging loop's namespace check.

**Write side is idempotent and sacct-derived**: `record_campaign_gpu_spend`
verifies an already-present row instead of appending (so a recovery
reconciliation cannot double-count) and every numeric field originates from
`sacct_row` with `accounting_source: "sacct"`.

**Opening zero is evidence-backed.** I checked `sacct` myself, read-only:

```
13805|c04_a0t_small_v1_v5_preflight| 0|billing=8,cpu=8,mem=64G,node=1|FAILED
13840|c04_a0t_small_v1_v6_preflight|19|billing=8,cpu=8,mem=64G,node=1|COMPLETED
```

Both rows match the ledger's `genesis_evidence` verbatim, including the
`alloc_tres` strings, elapsed seconds and states. Neither carries `gres/gpu`, so
`gpu_seconds: 0` is correct for both. A full `sacct` sweep for `c04` job names
returns exactly these two rows, corroborating
`these_are_the_only_c04_jobs_in_the_accounting_record: true`. The live ledger
loads and fully verifies: aggregate 0 s, cap 28800 s, 0 jobs, head `GENESIS`,
revision 0.

The read side is therefore sound. The write side is not reachable — see
**Critical C-A** and **High H-A**.

---

## Additional checks

- **`--time` directive:** absent from all three sbatch files and all three
  wrappers. Each sbatch carries an explicit comment that its omission is
  deliberate.
- **Arrays / dependencies / chained submission / release / resubmission:**
  absent. There is no `sbatch`, `scontrol`, `scancel`, `srun` or `salloc`
  anywhere in the v7 set. The *only* `subprocess` call in the entire tree is
  `gpu_ledger.py:227`, `sacct -X -n -P -j <id> -o JobIDRaw,ElapsedRaw,AllocTRES,State`
  — read-only. All three wrappers and both Python entrypoints reject
  `SLURM_ARRAY_JOB_ID` / `SLURM_JOB_DEPENDENCY`.
- **`--gres`:** exactly one occurrence, `scripts/slurm/c04_a0t_small_v1_v7.sbatch:3`,
  `--gres=gpu:a100:1`. The preflight sbatch requests no GPU. The reconcile sbatch
  requests no GPU, and its wrapper additionally rejects a non-empty
  `CUDA_VISIBLE_DEVICES` or `SLURM_GPUS_ON_NODE`.
- **Resources:** GPU sbatch = 1 GPU / 8 CPU / 64 GB; preflight sbatch = 8 CPU /
  64 GB, no GPU; reconcile sbatch = 1 CPU / 4 GB, no GPU. `resources.gpu_count/cpus/ram_gb`
  = 1/8/64 and are asserted in the preflight, the GPU ledger and the producer.
- **No OCR entrypoint, no network/API client, no dev/test path, no cross-dataset
  path, no label reader:** confirmed by targeted grep. No `requests`, `urllib`,
  `http`, `socket`, or any OCR library is imported anywhere. The only `label`
  reference on the data path is `_skip_json_value`, which advances the parser
  past the `label` token syntactically and increments a skip counter without
  converting it to a Python value; the projector then requires the decoded key
  set to be exactly `{id, window_text, language}`. `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` are exported by the wrapper and *asserted* by the
  producer, and both `from_pretrained` calls pass `local_files_only=True`.
  `root_path` rejects any `dev`/`test`/`validation`-like path component.
- **Authorization flags in the correct pre-review state:** exactly one flag is
  `true` (`implementation_authorized`); all sixteen others, including
  `preflight_materialization_authorized`, `teacher_authorized`, `gpu_authorized`,
  `slurm_authorized`, `small_tranche_execution_authorized` and
  `post_job_reconciliation_authorized`, are `false`. The preflight wrapper
  blocks on `preflight_materialization_authorized != true`; the reconcile wrapper
  blocks on `post_job_reconciliation_authorized != true`.
- **Unearned review pins are sentinels the code rejects.** Executed against the
  frozen config:

  | entrypoint | result |
  |---|---|
  | `verify_historical_code_resource_authorization` | HALT `code/resource authorization SHA-256 is unpinned` |
  | `verify_payload_review` | HALT `payload hash verdict is not GO` |
  | `verify_gpu_execution_authorization` | HALT `GPU execution verdict is not GO` |
  | `verify_historical_gpu_execution_authorization` | HALT `authorization SHA-256 is unpinned` |
  | `verify_resource_reconciliation_authorization` | HALT `reconciliation verdict is not GO` |
  | `resolve_prompt_hashes(freeze=False)` | HALT `sentinel … outside the authorized freeze run` |
  | `resolve_prompt_hashes(freeze=True)` with materialization false | HALT (same) |

- **Config contract normalization is exactly as narrow as documented.** Filling
  the four prompt-hash keys in does not move `config_contract_sha256` (verified),
  so the v5 impossibility is genuinely closed rather than displaced. Mutating
  `selection.suffix`, `resources.small_cap_gpu_seconds`,
  `resources.campaign_aggregate_cap_gpu_seconds`, `maps.expected_hashes`,
  `teacher_contract.num_frames` or `model.snapshot_revision` *does* move it.
  Only the four review pins, the four review verdicts, the `authorization` block
  and `prompt_hashes` are normalized out — each of which is separately bound by a
  strict stage-authorization manifest. Minor observation: `maps.expected_hashes`
  is protected only by being inside the contract hash; no code asserts its literal
  value, so the prose invariant in `prompt_hash_contract` is not machine-checked.
- **Environment:** the pinned interpreter
  `/data/jehc223/miniconda3/envs/HateVideo/bin/python` (3.11.8) has jsonschema
  4.26.0 (Draft7Validator present), PIL 11.1.0, numpy 1.26.4, torch 2.6.0+cu124,
  transformers 4.49.0, decord 0.6.0, av 17.0.0; `jq` 1.6 is on PATH.
  `Qwen2_5_VLForConditionalGeneration` imports. `GenerationConfig.validate()`
  with `do_sample=False, temperature=0.0` warns but does not raise. Statically
  traced `make_batched_videos`: `videos=[frames]` with 8 PIL images is correctly
  interpreted as one video of eight frames. None of these produce a finding.

---

# Findings

## CRITICAL C-A — the terminal CPU reconciliation can never pass its own gate: the producer writes three fields the reconciler's exact-key check forbids

**Where.** `scripts/analysis/c04_a0t_small_v1_v7_producer.py:1840-1859` (the
`provisional` record) versus
`scripts/analysis/c04_a0t_small_v1_v7_gpu_ledger.py:436-452`
(`require_exact_keys(provisional, {...}, "provisional GPU usage")`).

**Measured (AST extraction of both key sets, v6 and v7):**

```
v6 producer provisional dict (11)  == v6 reconciler require_exact_keys (11)   MATCH
v7 producer provisional dict (14)  vs v7 reconciler require_exact_keys (11)   *** MISMATCH ***
    producer-only: budget_guard, campaign_aggregate_cap_gpu_seconds,
                   campaign_gpu_ledger_sha256_at_seal
```

v7's I-3 work added three fields to `seal/provisional_gpu_usage.json` and did not
update the reconciler's exact-key gate. `require_exact_keys` is strict equality
(`set(value) != expected → raise`), so the check fails unconditionally.

**Failure scenario.** The GPU job runs to completion, spends up to the full
2 GPU-hours on an A100, publishes the seal, and exits 0. The mandatory
`CPU_POST_JOB_RECONCILIATION` job then runs `reconcile_terminal` →
`verify_reconciliation_lineage`, which reaches the `provisional` check *before*
it touches the ledger, and raises
`HALT_…: provisional GPU usage exact-key failure`. Consequences, all of them
terminal:

- the per-namespace GPU ledger stays in `EXIT_RECORDED_PENDING_SACCT` with a
  7200 s reservation and is never reconciled to real sacct seconds;
- `resource_final_state.json` is never published, so the seal's
  `resource_final_state_required_before_any_downstream_review: true` and the
  config's `review.downstream_review_requires_terminal_resource_state: true`
  block every downstream review permanently;
- `record_campaign_gpu_spend` — the *write half of I-3* — is never reached, so
  the 8 GPU-hour accumulator never learns that the tranche happened.

The reconcile wrapper's recovery branch does not help: it retries only when the
ledger is already `SACCT_TERMINAL_RECONCILED`, which it never becomes, so the
wrapper propagates the failure. And the run cannot be repaired by editing
`gpu_ledger.py`, because that changes its SHA-256, which changes
`config.implementation_hashes`, which changes `config_contract_sha256`, which
invalidates the `config_contract_sha256` values already pinned inside the
no-clobber `artifacts/c04/a0t_small_v1_impl_v7/` preflight manifest, genesis GPU
ledger and resource ticket. This is the v5 impossibility displaced one stage
further — onto a fully consumed A100 allocation.

**What would close it.** Add the three new keys to the reconciler's
`require_exact_keys` set (and assert their types/values there, as the other
fields are asserted), *or* move the new accounting into a separate sibling
artifact and leave `provisional_gpu_usage.json` on its v6 key set. Then add a
CPU-preflight fixture that builds the producer's `provisional` dict and feeds it
to the reconciler's key set, so writer/reader drift cannot recur silently.

## CRITICAL C-B — `proposition_cosine` can exceed the canonical schema's `maximum: 1` whenever prompts A and B agree, halting the producer after all 800 forwards

**Where.** `producer.py:1096-1101` (`cosine`), `common.py:1468`
(`"proposition_cosine": proposition_cosine`), and
`schemas/c04/c04_a0t_small_v1_v7_canonical_record.schema.json`
`definitions.reliability.properties.proposition_cosine` →
`{"type":"number","minimum":-1,"maximum":1}`, enforced by
`validate_schema(record, cfg["schemas"]["canonical_record"], …)` at
`producer.py:1353`.

**Mechanism.** When prompt A and prompt B yield the same proposition after
`normalize_proposition`, the two mean embeddings are bit-identical. `cosine`
then computes `S / (sqrt(S) * sqrt(S))`, and `sqrt(S)**2` differs from `S` by up
to one ulp, so the result can exceed 1.0.

**Measured.** Over 2000 trials using bfloat16-rounded mean embeddings at the
pinned `TEACHER_DIM = 3584` and realistic magnitudes (mimicking
`embedding(ids).mean(dim=1)[0].float().cpu().tolist()`):

```
cosine(v, v) > 1.0  in  504 / 2000 trials  (25.2%)   max observed 1.0000000000000002
```

and feeding that value into a record built exactly as `canonicalize_dataset`
builds it:

```
FAIL -> ['slots','P','proposition_cosine']: 1.0000000000000002 is not valid under any of the given schemas
```

Probability that at least one of the 400 items trips it, as a function of the
number *k* of items where A and B agree exactly: k=1 → 0.25, k=5 → 0.77,
k=20 → 0.997, k=100 → 1.000. Exact agreement is not an edge case — it is the
outcome the design *rewards* (the `stable` state and the 0.80 cosine floor exist
to detect it), and decoding is greedy (`do_sample=False`, `num_beams=1`), so
short propositions will match verbatim across the two prompts routinely.

**Failure scenario.** All 800 teacher forwards complete and are checkpointed.
`canonicalize_dataset` then raises
`HALT: canonical HateMM/<id> schema failure: ['slots','P','proposition_cosine'] …`
at the first affected item — i.e. after essentially the entire 2 GPU-hour
allocation has been spent, and **before any seal is written**. No seal means no
reconciliation (see H-A), so this also silently loses the GPU-second accounting.
Nothing in the CPU preflight can catch it: `self_test_fixtures()` never builds a
canonical record and never invokes `validate_schema` on one. This is inherited
from v6 (the schema is byte-identical), but v6 could never reach it because of
the render defect; v7 is the first version in which it is reachable, so it is a
v7 finding.

**What would close it.** Clamp the returned cosine into `[-1.0, 1.0]` in
`cosine()` (a one-line `max(-1.0, min(1.0, …))`, which changes no science — the
0.80 threshold comparison is unaffected), or relax the schema bound. Independently,
add the CPU-preflight fixture described in Important I-C so that a canonical
record built from the `stable`/agreeing case is schema-validated before any GPU
is requested.

## HIGH H-A — the campaign accumulator's write side is reachable only on the fully-sealed path, so the 8 GPU-hour ceiling keeps reading a stale zero

**Where.** `gpu_ledger.py:1272` (`record_campaign_gpu_spend`) is called only from
`reconcile_terminal`, which first runs `verify_reconciliation_lineage`, which
opens `cfg["paths"]["provisional_gpu_usage"]` — a file that lives inside
`…/seal/` and is created only by the producer's final atomic seal publication.

**Failure scenario.** Every non-sealing outcome leaves the campaign accumulator
unwritten while GPU-seconds were genuinely burned:

- **budget breach (exit 40)** — the very path I-3 introduces: the guard stops
  before an item, no seal is published, so `provisional_gpu_usage.json` does not
  exist, `reconcile_terminal` dies with `FileNotFoundError`, and up to 2 GPU-hours
  are never recorded;
- **watchdog TERM/KILL (exit 124/137/143)** — same;
- **any producer HALT, OOM, decode failure, or C-B above** — same;
- **a fully successful run** — blocked instead by Critical C-A.

Taken together with C-A, `append_campaign_gpu_job` is unreachable on *every*
path in v7. A subsequent C04 namespace (v8, or a later extraction/adaptation job)
would call `assert_campaign_aggregate_headroom` and read `aggregate_gpu_seconds:
0`, authorizing a fresh 7200 s reservation as though nothing had been spent. The
accumulator that exists specifically to survive an implementation-version bump
therefore does not, in practice, remember anything.

**What would close it.** Make the terminal accounting stage independent of the
seal: derive the original job id from `resource/allocation_claim.json` (which
exists from claim time, before any teacher work) rather than requiring
`seal/provisional_gpu_usage.json`, and treat the seal/provisional checks as
conditional refinements. The campaign row should be appended from sacct for any
terminal GPU job that consumed the ticket, sealed or not.

## HIGH H-B — the in-job tranche guard is given no margin over the wrapper `timeout` it was introduced to replace, so the exit-40 breach path is usually pre-empted

**Where.** `producer.py:407-433` (`BudgetGuard.at_job_start`) versus
`scripts/wrappers/c04_a0t_small_v1_v7.sh:101-105` (`timeout … "${C04_ACTIVE_WATCHDOG_SECONDS}s"`).

Both deadlines are set to the same budget (`cap − reserve = 7080 s`), with no
enforced ordering between them.

**Failure scenario, `SLURM_JOB_START_TIME` present** (documented as exported by
this cluster's SLURM — `man sbatch` lists `SLURM_JOB_START_TIME`; version
25.11.4): the guard takes the `SLURM_ALLOCATION_START_EPOCH` branch and its
deadline lands at `allocation_start + 7080`. The wrapper's `timeout` was started
after the SLURM prolog, the bash preamble, the jq marker write and the `claim`
call, so it fires at `allocation_start + P + δ + 7080` — only `P + δ` (a few
seconds) later. But the guard is checked **once per item**, and an item is two
7B-VLM forwards (order 10-20 s at the 8-frame/256-new-token settings). The item
that straddles the deadline therefore runs past `timeout` and is SIGTERM'd
mid-forward, exit 124, **no breach record**, before the next item-boundary check
can fire. With a few seconds of margin against ~20 s items, the clean exit-40
path is taken only a small fraction of the time.

**Failure scenario, `SLURM_JOB_START_TIME` absent** (a config change, or any
invocation of the wrapper outside `sbatch`): the guard falls back to
`TICKET_WATCHDOG_REMAINDER`, `deadline = now_monotonic + watchdog_seconds`. But
`now_monotonic` is taken when `verify_claimed_resource` runs — after
`verify_model_snapshot` has SHA-256'd the **16.60 GB** pinned model/processor
tree (≈23 s of hashing at this node's measured ~710 MB/s, plus disk read), and
after three `verify_bound_file_map` sweeps and the full lineage chain. The
guard's deadline therefore lands *V ≈ 25-60 s later* than the wrapper's timeout,
and the guard can never fire at all. The tranche ceiling reverts to exactly the
v6 behaviour the I-3 repair exists to replace: `timeout` killing mid-forward.

**What would close it.** Anchor the guard to allocation entry rather than to
guard-construction time (the wrapper already computes and passes
`--allocation-start-uptime-seconds`; pass the same value to the producer and
derive the deadline from `/proc/uptime` rather than `time.monotonic()`), and give
the guard a hard margin over the wrapper's `timeout` of at least one worst-case
item — e.g. guard deadline `= allocation_start + cap − reserve − item_margin`
with the wrapper timeout left where it is, plus a startup assertion that the
guard deadline is strictly earlier than the remaining `timeout` budget.

## IMPORTANT I-C1 — the campaign accumulator enforces the conditional-tranche ceiling, not the currently binding first-tranche ceiling

`refine-logs/C04_USER_AMENDMENT_V2.md` places the 8 GPU-hour ceiling on the
**conditional full-bank tranche**, which is unlocked only by `PASS_C04_SMALL_V2`
plus a fresh result-to-claim `GO` plus a new code/resource review. The clause
binding *now* is stricter: the first tranche must observe "an aggregate maximum
of **2 GPU-hours** across both datasets **and all C04 jobs**". `CAMPAIGN_AGGREGATE_CAP_GPU_SECONDS`
is 28800 and `load_campaign_gpu_ledger` halts unless the on-disk cap equals
28800, so the accumulator cannot express the phase-1 limit at all.

**Failure scenario, measured.** With a well-formed ledger recording that the
first tranche already spent its full 7200 s, `assert_campaign_aggregate_headroom(7200)`
**accepts** a second C04 GPU allocation (7200 + 7200 = 14400 ≤ 28800). Given this
project has already produced three implementation namespaces (v5, v6, v7), a v8
rebuild after a v7 failure is the expected case, and it would take the campaign
to 4 GPU-hours under an amendment clause that caps it at 2 until the conditional
tranche is separately authorized. The v7 run *itself* is compliant (0 + 7200 ≤
7200), so this is not a blocker for this job — it is a gap in the ceiling that
outlives it.

**What would close it.** Carry a phase-scoped cap: keep 28800 as the campaign
ceiling but add `phase_cap_gpu_seconds: 7200` with a phase token in the ledger
that only the conditional-tranche authorization may advance, and check the
reservation against `min(phase_cap, campaign_cap)`. (Separately, `append_campaign_gpu_job`
takes no lock on the shared campaign file — it is protected only by the
*per-namespace* `gpu_ledger.lock`, which would not exclude a v8 reconciler; a
lock on the campaign path itself would close that.)

## IMPORTANT I-C2 — the hard 7200 s equality leaves ~90 s of margin, and exceeding it makes reconciliation permanently impossible

`reconcile_terminal` raises `HALT_RESOURCE_CAP: terminal sacct GPU seconds exceed
7200 cap` when `sacct` `ElapsedRaw` exceeds 7200;
`strict_validate_terminal_ledger` independently rejects seconds outside
`[0, 7200]`; and the `resource_final_state` schema pins
`terminal_sacct_gpu_seconds` / `aggregate_accounted_gpu_seconds` /
`aggregate_reconciled_terminal_gpu_seconds` to `maximum: 7200`.

**Failure scenario.** The wrapper's own budget is `7080 s` of `timeout`, plus
`--kill-after=30s`, plus the EXIT-trap `mark-exit` call, all measured from wrapper
start — while `sacct` measures from *allocation* start. The margin between the
worst-case wrapper wall time (`P + 7080 + 30 + ε`) and the hard 7200 s ceiling is
therefore only about 90 s, consumed by the SLURM prolog, node setup, `conda`/bash
startup and the `mark-exit` write. If it is exceeded — most plausibly on a
watchdog-terminated run, exactly the case where the reserve matters — the ledger
can never be reconciled, `resource_final_state.json` can never be published, and
the namespace is wedged for the same no-clobber reason described in C-A. A hard
ceiling whose breach is unrecoverable should not be defended by an unmeasured
90 s.

**What would close it.** Either lower the wrapper's effective budget so the
worst-case sacct elapsed is provably under 7200 s (e.g. reserve 300 s rather than
120 s, or subtract the measured allocation-start-to-wrapper-start offset from the
timeout), or make an over-cap terminal sacct row a *recorded* over-run — written
to the ledger, the campaign accumulator and a distinct final state — rather than
an unrecoverable halt.

## IMPORTANT I-C3 — the CPU preflight self-test never validates a producer record against its own JSON Schema, nor cross-checks writer/reader key sets

The 37 fixtures cover the prompt-hash contract, the containment guard, the
renderer, the selection rule, the parser, the transcript cap and the reliability
states. They do **not** construct a `prompt_record`, a `canonical_record`, a
`frame_pack_manifest`, a `provisional_gpu_usage` record or a
`resource_final_state` record and validate it against the schema/key-set the
downstream stage will apply. `validate_schema` is exercised at preflight only
against the stage-authorization manifest.

This is precisely the blind spot that let v6's `str.format` defect through — "no
v6 self-test ever rendered a prompt" — and it is the same blind spot that lets
C-A and C-B through in v7. I verified by construction that such fixtures would
have caught both: a synthetic canonical record with
`proposition_cosine = 1.0000000000000002` fails validation immediately, and a
one-line comparison of the producer's `provisional` key set against the
reconciler's `require_exact_keys` set flags the mismatch statically.

**What would close it.** Add CPU-preflight fixtures that (a) build one
`prompt_record` and one `canonical_record` — including the degenerate cases: all
slots `missing`, all `single_valid`, and the *agreeing* `stable` case with
`proposition_cosine` at and just above 1.0 — and run them through
`validate_schema` against the frozen schemas; and (b) assert that every
producer-written artifact's key set equals the key set the consuming stage's
`require_exact_keys` demands. Both are pure-CPU, cost nothing, and would have
turned both Criticals into preflight failures.

---

## Summary table

| # | Severity | Finding |
|---|---|---|
| C-A | Critical | Terminal CPU reconciliation can never pass: producer writes 3 keys the reconciler's `require_exact_keys` forbids; blocks final state, wedges the namespace after the A100 is spent, and makes the I-3 campaign write side unreachable |
| C-B | Critical | `proposition_cosine` exceeds the canonical schema's `maximum: 1` in 25.2% of exact A/B agreements; producer HALTs after all 800 forwards, before any seal |
| H-A | High | Campaign accumulator write side reachable only on the fully-sealed path; every failure mode (incl. the new exit-40 breach) leaves the 8 GPU-hour ceiling reading a stale zero |
| H-B | High | In-job tranche guard has no enforced margin over the wrapper `timeout`; the exit-40 accounting path is usually pre-empted by a mid-forward SIGTERM, and is unreachable entirely if `SLURM_JOB_START_TIME` is absent |
| I-C1 | Important | Accumulator enforces the 8 GPU-hour conditional-tranche ceiling, not the currently binding 2 GPU-hour "all C04 jobs" first-tranche ceiling; a v8 namespace would be authorized a second 7200 s |
| I-C2 | Important | ~90 s of margin between worst-case sacct elapsed and the hard 7200 s ceiling; exceeding it makes reconciliation permanently impossible |
| I-C3 | Important | CPU preflight never schema-validates a producer record nor cross-checks writer/reader key sets — the blind spot that produced C-A and C-B |

**Verdict: REVISE (2C / 2H / 3I). No execution authority is conferred. The
`preflight_materialization_authorized` flag must remain `false` and
`review.code_resource_verdict` must remain `PENDING` until at least the two
Criticals are closed and the closing fixtures are added to the CPU preflight.**

----- END refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md -----


----- BEGIN refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND2.md (sha256 7116d9489e6374f19f0a7d8cda65c5cf7f400d727356451e652e536adb79b6fc) -----
# C04-A0T-SMALL-v1 v7 — Fresh Independent Code/Resource Review, ROUND 2

Reviewer: fresh independent static reviewer (no exposure to the authoring reasoning)
Date: 2026-07-31
Stage reviewed: `CPU_PREFLIGHT` code/resource review of implementation-v7, second revision
Predecessor: `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md` (`REVISE 2C/2H/3I`), left intact
Execution authority conferred by this review: **none**

## Verdict

**REVISE — 0 Critical / 2 High / 3 Important (0C / 2H / 3I)**

Both round-1 Criticals and both round-1 Highs are genuinely repaired at the level
of what this run would do: I reproduced each defect's trigger and confirmed the
frozen bytes no longer fail. `I-C1` is fully closed. Claims 0, C-1, I-1 and I-2
all reproduced exactly.

The two Highs below are not soft. **H-1** is the same failure family the request
named — an irreversible resource written before the check that rejects it — and
the I-3 phase-cap repair *created* it: `campaign-record` now writes an unfiltered
sacct row into the one artifact designed to outlive every implementation
namespace, before the stage that would refuse that value runs, and an over-cap
write makes the campaign accumulator permanently unloadable for all future C04
work. **H-2** is a repair claim that is not delivered: the round-2 summary states
that the two halves of the `provisional_gpu_usage` contract "build and validate
through single shared constants, and a fixture asserts they agree." Neither is
true. The reader still carries a hand-maintained literal, and I reproduced the
exact v7-first-draft defect with all 45 preflight fixtures still green.

## Method and reviewer-boundary compliance

- No SLURM job submitted, held, released, requeued or cancelled. `sacct` and
  `squeue` used read-only only.
- No GPU, teacher, model-weight or frame-decode work. No model file was opened.
- No file under `/data/jehc223/RGCL` was created, modified or deleted other than
  this review file. `artifacts/c04/campaign/gpu_ledger.json` re-hashes to
  `fc6ca12c32427625d0b80c16b7802ef9a574ced0dbf0288edc3938d217267414` after every
  write-capable probe (all mutation experiments ran against a scratchpad copy
  with `common.ROOT` repointed outside the repository).
- `artifacts/c04/a0t_small_v1_impl_v7/` does not exist and was not created.
- No dataset label value was materialized. Both ASR files were read only through
  the frozen `project_train_asr_line` projector (`id`, `window_text`,
  `language`). HateMM identifiers were hashed and compared, never printed and
  never reasoned from as labels.
- All work in `…/scratchpad/review-r2`, outside the repository, with
  `PYTHONDONTWRITEBYTECODE=1` on every invocation. `common.py` was imported from
  a byte-identical scratchpad copy; `BudgetGuard` was executed by extracting its
  frozen `ClassDef` with `ast`, never by importing `producer.py` from the
  repository path. `py_compile` was not used.

### Hash verification (all 17, before and after)

Every pinned SHA-256 in the request table matches disk exactly, re-verified at
the end of the session. In addition:

- all 15 `configs/c04/c04_a0t_small_v1_v7.json → implementation_hashes` verify
  against disk (15/15) and equal the request table;
- all 15 `frozen_design_hashes` verify against disk (15/15).

### v6 predecessor unmodified

- `artifacts/c04/a0t_small_v1_impl_v6/freeze/preflight_manifest.json` is
  self-consistent (`payload_sha256` reproduces) and **all 14** of its
  `staged_output_hashes` verify byte-for-byte on disk.
- all 15 entries of `configs/c04/c04_a0t_small_v1_v6.json → implementation_hashes`
  verify against disk.
- every v6 artifact carries mtime `2026-07-31 05:11:50` (job 13840), predating
  every v7 source file.

---

## Round-1 findings — closure status

| # | Round-1 finding | Status | Evidence |
|---|---|---|---|
| C-A | reconciler's exact-key set had not grown with the writer | **CLOSED (live defect)** — see **H-2** for the unrepaired recurrence channel | reader set is now 14 keys (v6: 11) and matches the writer exactly; all **five** writer/reader pairs agree |
| C-B | `proposition_cosine` could exceed the schema `maximum: 1` | **CLOSED** | clamp present in `cosine()`; overshoot re-measured at 572/2000; no comparison outcome changes |
| H-A | campaign write side reachable only after a seal | **CLOSED for every named path** — residual gap in **I-3** | new `campaign-record` mode keyed on `allocation_claim.json`, run first in the reconcile wrapper |
| H-B | in-job guard had no margin over the wrapper `timeout` | **CLOSED** | measured lead = `300 + claim_duration` s, always ≥ 300 s; `SLURM_JOB_START_TIME` gone |
| I-C1 | accumulator enforced 28800 s, not the binding 7200 s | **CLOSED** | effective cap = `min(phase, aggregate)` = 7200 s today; no code path advances the phase |
| I-C2 | ~90 s margin to the hard 7200 s ceiling; breach unrecoverable | **NOT CLOSED, blast radius now larger** | see **H-1** |
| I-C3 | preflight never round-trips a record against a downstream contract | **PARTIALLY CLOSED** | see **I-1** |

Both round-1 non-blocking observations were re-checked:

- The pre-model-load containment pass now assembles through `build_messages` and
  extracts through `teacher_visible_texts`, i.e. the same path as the per-forward
  call site (`producer.py:1636-1646`). **The half that compares `texts` against
  `[SYSTEM_PROMPT, render_prompt(form, transcript)]` still self-compares there**,
  because `texts` is derived from the same `render_prompt` call; what the pass
  genuinely establishes before model load is "no banned token in any of the 400
  transcripts, and the assembled message shape is legal". The message-assembly
  half becomes non-vacuous only at `producer.py:1702`, which does run before
  every forward, so the stated requirement is met.
- `maps.expected_hashes` remains protected only by inclusion in the contract
  hash (verified: mutating it moves `config_contract_sha256`). No code asserts
  its literal value. Unchanged from round 1, still non-blocking.

---

## Claim 0 — no scientific semantic changed: **CONFIRMED**

### Selection re-derived from the v7 frozen rule reproduces the v6 frozen allowlists

Both train ASR files were read through the v7 projector, ranked by the v7
`selection_digest` with the v7 tie-break, first 200 taken:

| dataset | train N | recomputed order == v6 frozen allowlist | sha256(newline-joined ordered id list) |
|---|---|---|---|
| HateMM | 744 | **True** | `091fb1826cbc7f80…` |
| MHC_zh | 579 | **True** | `6c98c0d75891ce43…` |

Also reproduced, per dataset: every `selection_sha256` stored in the v6
allowlist (200/200), and — against the v6 `source_manifest.json` — all 200
`transcript_sha256` and all 200 `transcript_scalar_count` values. That last
check independently pins the transcript normalization, cap, head/tail split and
separator as unchanged.

### Prompt hashes

Recomputed from the v7 sources:

```
system   1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048
A        cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b
B        9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314
combined a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a
```

Equal to the in-module fixture literals **and** to
`artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json`. No prompt byte
changed.

### Version-token-normalized tree diff — every residual line accounted for

Both trees copied to the scratchpad, `s/v6|V6/vTOK/` and `s/v7|V7/vTOK/`
applied, diffed file by file:

| file | changed lines | accounted for by |
|---|---|---|
| **all 5 schemas** | **0** | — (so the canonical schema's `maximum: 1` is byte-identical to v6) |
| `*_preflight.sh`, `*_preflight.sbatch`, `*.sbatch`, `*_reconcile.sbatch` | 0 | — |
| `*_reconcile.sh` | 8 | H-A (`campaign-record` before `reconcile-terminal`) |
| `*_v7.sh` | 18 | I-3 (exit-40 branch, breach-record `jq`, zero-exit breach guard) |
| `preflight.py` | 30 | I-3 (campaign cap/path/margin/phase assertions + headroom call) |
| `gpu_ledger.py` | 145 | I-3 / H-A / C-A only |
| `common.py` | 769 | C-1 + I-1 + I-2 + I-3, **pure additions** |
| `producer.py` | 472 | C-1 + I-1 + I-3 |
| `config.json` | 135 | version tokens, `v7_scope` prose, three new `resources` keys, two new `paths`, refreshed `implementation_hashes` |

A function-level AST diff makes this exact. `common.py`: **32 additions, 0
removals, exactly one changed definition** (`self_test_fixtures`, the I-2 fixture
swap plus the new fixture groups). `producer.py`: 5 additions
(`BudgetGuard`, `BudgetDeadlineReached`, `BUDGET_BREACH_EXIT_CODE`,
`assert_teacher_visible_precondition`, `publish_budget_breach_record`) and 7
changed definitions (`build_messages` = C-1, `cosine` = C-B, `deadline_check`,
`verify_authorization`, `verify_claimed_resource`, `verify_execution_lineage`,
`main` = I-3). Nothing else moved: `SYSTEM_PROMPT`, `_SCHEMA_TEXT`, `PROMPT_A`,
`PROMPT_B`, `SELECT_TAG`, `SELECT_SUFFIX`, `SELECT_N`, `NUM_FRAMES`, the
transcript constants, the confidence/cosine thresholds, the five-rate KILL
taxonomy, `build_slot_reliability`, `render_slot`, `materialize_role_map`,
`dense_rademacher_payload`, `parse_teacher_response` are untouched.

`self_test_fixtures()` now returns **45** checks (round-1 v7: 37; v6: 25). All 45
pass on the frozen bytes, and no fixture name is duplicated (a duplicate name
would be silently dropped by `dict(self_test_fixtures())` in
`preflight.run_self_tests`).

---

## Claim C-1 — the prompt renderer: **CONFIRMED**

1. **Could the v6 form ever have succeeded?** No, executed against the frozen
   templates: `PROMPT_A.format(transcript="x")` and `PROMPT_B.format(...)` both
   raise `KeyError: '"source_relation"'`. Unconditional on both forms.
2. **Exact substitution?** Yes. For a transcript containing non-ASCII, CJK, a
   newline and a literal `{braces}` sequence, `render_prompt(form, t) ==
   PROMPTS[form][:-len("{transcript}")] + t` for both forms, with the module-level
   templates unmutated.
3. **Any prompt byte changed?** No — the four hashes match the v6 frozen artifact.
4. **Any surviving `.format(transcript=` call site?** None. The two textual
   occurrences are a docstring (`common.py:221`) and the deliberate regression
   fixture `prompt_render_regression_str_format_would_raise` (`common.py:2194`),
   which asserts the v6 form still raises `KeyError`.
5. **Guards non-vacuous?** The four pre-substitution guards
   (`prompt_form not in PROMPT_FORMS`, non-`str` transcript, `count != 1`,
   non-terminal placeholder) are genuine, and two are exercised by
   `prompt_render_rejects_unknown_form_and_non_string`. The two post-hoc guards
   remain weak (`startswith(prefix)` is a tautology given a unique terminal
   placeholder; `endswith(transcript)` is vacuous for an empty transcript).
   Unchanged from round 1, still an observation rather than a finding.

---

## Claim I-1 — teacher-visible containment: **CONFIRMED**

1. **Before model load?** Yes — `assert_teacher_visible_precondition(inputs)` at
   `producer.py:1670`, `from_pretrained` at `producer.py:1679`. Repeated per item
   inside `one_forward` at `producer.py:1702`, after `build_messages` and before
   `apply_chat_template`/`generate`.
2. **Strict in both directions?** Yes. `teacher_visible_texts` raises on a wrong
   message-list length, unexpected message keys, an unexpected role, non-list or
   empty content, a content part that is not a dict or lacks `type`, a text part
   with extra keys or a non-string body, a video part with extra keys, a frame
   list whose length ≠ 8, any frame that is `str`/`bytes`/`bytearray`/`Path`, any
   unknown content type, and a final census requiring exactly one video part and
   exactly two text parts. Six of these are pinned by fixtures.
3. **What is banned, and is it wide enough?** Measured on the real tranche: the
   ban list is **402** tokens — all 200 HateMM + all 200 MHC-ZH selected
   identifiers plus `hate_video_` and `non_hate_video_` — and both datasets'
   identifiers are banned in both datasets' prompts, so cross-item leakage is
   refused as firmly as self-leakage. Matching is over
   `{token, NFKC(token), casefold(NFKC(token))}` against both `NFKC(text)` and
   `casefold(NFKC(text))`. The wider protection is the equality
   `texts == [SYSTEM_PROMPT, render_prompt(form, transcript)]`: the only variable
   content reaching the teacher is the transcript, so the amendment's broader ban
   (prediction, neighbor, rank, margin, error status, dataset statistic, fold
   role, intended use) is satisfied structurally rather than by enumeration.
4. **False positives on the real 400 transcripts?** **None — 0 of 800.** I ran the
   frozen guard label-blind over all 400 selected transcripts × both forms: zero
   rejections. As a margin, the 402-token ban list was scanned against **all
   1323** train transcripts (744 + 579): **0** rows contain any banned token.
   Shortest banned token is 11 characters, so accidental substring collision is
   not a live risk.
5. **Can it pass vacuously?** Not at the per-forward call site:
   `if not forbidden or video_id not in forbidden: raise` makes an empty or
   incomplete ban list a halt, and
   `teacher_visible_unbanned_identifier_rejected` proves an identifier absent
   from the list is rejected. The pre-load pass is weaker in one half, as noted
   above.

### The HateMM ID-label asymmetry — stated explicitly, as requested

**The asymmetry is handled correctly.** Every HateMM training identifier is
`hate_video_*` or `non_hate_video_*`, so the identifier *is* the binary label;
MHC-ZH identifiers are opaque BiliBili `BV` codes carrying no label information.
Therefore **the sealed ID-only allowlist delivers label containment for MHC-ZH
only, and none at all for HateMM.** `LABEL_BEARING_ID_SUBSTRINGS` encodes exactly
this (`("hate_video_", "non_hate_video_")` for HateMM, `()` for MHC_zh), the
`common.py` comment block states it, and `config.json →
v7_scope.I1_teacher_visible_containment` records it. The consequence is drawn
correctly: for HateMM, selection label-blindness rests on *hash reproduction of
the selection rule* (independently reproduced above, 200/200 on both datasets)
and teacher label-blindness on this *runtime* check — not on the allowlist.

---

## Claim I-2 — the selection self-test is a known-answer vector: **CONFIRMED**

Both pinned digests recomputed from first principles outside the module, by hand
concatenation:

```
sha256(utf8("C04-A0T-SMALL-v1" + "HateMM" + "c04-known-answer-vector" + "20260729"))
  = 871e0363e1b01a823f09a5a0bb9187749da74f1dfe8e454a733e21c218f6a384   [matches]
sha256(utf8("C04-A0T-SMALL-v1" + "MHC_zh" + "c04-known-answer-vector" + "20260729"))
  = 41bfb637f5cbb26bb0b9edfd44d19a3775d8df1712f9b49bde667863fbd37134   [matches]
```

The digests are literals independent of the module's own code path. Mutation
sensitivity measured: tag → breaks; suffix → breaks; dataset term → breaks;
identifier → breaks; concatenation order → breaks. The companion fixture
`selection_dataset_and_id_sensitivity` additionally requires three distinct
digests. The chosen identifier is synthetic and belongs to neither dataset, so
the fixture pins the rule without naming a label-bearing real id.

---

## Claim I-3 — both ceilings machine-checked and fail-closed: **PARTIALLY DELIVERED**

### Tranche ceiling (7200 s) — guard now leads the wrapper (round-1 H-B closed)

`BudgetGuard.at_job_start` is constructed exactly once, in
`verify_claimed_resource`; the deadline is stored and never recomputed. There are
exactly two `deadline_check` call sites — `producer.py:1751` at the item boundary
and `producer.py:1697` inside `one_forward` — both strictly *before* a unit of
work. `BudgetDeadlineReached` is caught only by the outer handler, which
publishes an accounting-only record and returns 40.

`SLURM_JOB_START_TIME` appears nowhere in the v7 tree except one docstring line.
The anchor is now the allocation-entry `/proc/uptime` reading carried in
`allocation_claim.json`. I executed the frozen `BudgetGuard` class (extracted by
`ast`, not imported) against the frozen config:

| claim duration `c` | lineage+model-hash time `v` | guard fires | wrapper `timeout` | lead |
|---|---|---|---|---|
| 0 s | 0 s | entry + 6780 s | entry + 7080 s | **300 s** |
| 5 s | 30 s | entry + 6775 s | entry + 7080 s | **305 s** |
| 30 s | 60 s | entry + 6750 s | entry + 7080 s | **330 s** |
| 60 s | 120 s | entry + 6720 s | entry + 7080 s | **360 s** |

The guard's lead is `guard_item_margin_seconds + c`, i.e. never less than 300 s,
because the wrapper's `C04_ACTIVE_WATCHDOG_SECONDS` already has `c` subtracted
and `at_job_start` subtracts the elapsed-since-entry again. The margin comfortably
covers a worst-case item (two 7B forwards at 8 frames / 256 new tokens), so the
exit-40 breach record is reachable. **Round-1 H-B is closed.** What the margin
does *not* cover is the unguarded post-loop phase — see **I-2**.

The breach record was inspected field by field: lineage, job id, terminal state,
exit code, both caps, the guard snapshot, per-dataset completed counts, teacher
and frame-pack counters, `outputs_truncated_or_altered: 0`, `seal_published:
false`, `no_scientific_verdict_is_published_by_a_budget_breach: true`. No metric,
no teacher output, no reliability rate, no CONTINUE/KILL verdict. Exit 40 is
distinct and the wrapper propagates it distinctly, and refuses a zero-exit run
that left a breach record behind.

### Campaign ceiling — read side sound and phase-scoped (round-1 I-C1 closed)

`assert_campaign_aggregate_headroom` is called from `validate_gpu_environment`,
the first statement of `claim()`, i.e. before `create_entry_marker`, before
`verify_gpu_lineage`, and well before the allocation claim and the ticket
consumption record are published — genuinely before the single-use ticket is
consumed. It is also called by the CPU preflight *before* the no-clobber
namespace is materialized, and by the producer before any model or data work.

Fail-closed matrix, executed against a scratchpad copy (repository untouched):

| mutation | result |
|---|---|
| ledger absent | HALT `campaign ledger is absent` |
| aggregate ≠ Σ rows | HALT `payload mismatch` |
| `payload_sha256` tampered | HALT `payload mismatch` |
| foreign `schema_version` | HALT `foreign campaign ledger schema` |
| foreign `run_id` | HALT `foreign campaign ledger run id` |
| aggregate cap raised to 999999 | HALT `cap is not the amendment cap` |
| aggregate cap lowered to 7200 | HALT `cap is not the amendment cap` |
| `phase` advanced, phase cap left at 7200 | HALT `phase cap does not match the phase` |
| `phase_cap` raised alone | HALT `phase cap does not match the phase` |
| first-tranche phase carries an advance token | HALT `first-tranche phase carries an advance token` |
| row chain break | HALT `chain break` |
| well-formed, 1 s already spent | loads, next 7200 s reservation **REFUSED** |
| well-formed, 7000 s already spent | loads, next 7200 s reservation **REFUSED** |
| genesis (0 s) | loads, 7200 s reservation accepted (effective cap 7200) |

**The effective ceiling is 7200 s today**, `min(phase_cap, aggregate_cap)`, and it
is the amendment clause that actually binds ("an aggregate maximum of 2
GPU-hours across both datasets and all C04 jobs",
`C04_USER_AMENDMENT_V2.md:32`). Any recorded spend of even one second refuses the
next 7200 s reservation, so a v8 rebuild after a v7 failure is refused — which is
what round-1 I-C1 asked for. **No code path writes `phase` or
`phase_advance_authorization`**: `append_campaign_gpu_job` copies the ledger dict
and mutates only `jobs`, `aggregate_gpu_seconds`, `head_payload_sha256`,
`ledger_revision`, `payload_sha256`. The only route to 28800 s is a hand edit
that changes `phase` and `phase_cap_gpu_seconds` together and reseals the payload
— i.e. the human gate the amendment intends. **Round-1 I-C1 is closed.**

**Opening zero is evidence-backed.** `sacct` read read-only by me:

```
13805|c04_a0t_small_v1_v5_preflight|…|0 |billing=8,cpu=8,mem=64G,node=1|FAILED
13840|c04_a0t_small_v1_v6_preflight|…|19|billing=8,cpu=8,mem=64G,node=1|COMPLETED
```

Both rows match the ledger's `genesis_evidence` verbatim including the
`alloc_tres` strings, elapsed seconds and states; neither carries `gres/gpu`, so
`gpu_seconds: 0` is correct for both. A full accounting sweep for `c04` job names
returns exactly these two rows, corroborating
`these_are_the_only_c04_jobs_in_the_accounting_record: true`.

### Campaign ceiling — write side now seal-independent (round-1 H-A closed)

`campaign-record` is a distinct ledger mode keyed on
`resource/allocation_claim.json`, which `claim()` publishes before it appends the
ledger job row, and the reconcile wrapper runs it **first**, under `set -e`,
before `reconcile-terminal`. Every path the request named therefore reaches it:

| path | `allocation_claim.json` present? | campaign row written? |
|---|---|---|
| exit 40 budget breach | yes | **yes** |
| watchdog TERM/KILL (124/137/143) | yes | **yes** |
| OOM / decode failure / any producer HALT after claim | yes | **yes** |
| fully successful sealed run | yes | **yes** (and again idempotently from `reconcile_terminal`) |
| HALT *before* `claim()` publishes the claim | no | **no** — see **I-3** |

`record_campaign_gpu_spend` verifies an already-present row instead of appending,
so a recovery reconciliation cannot double-count, and every numeric field
originates from `sacct_row` with `accounting_source: "sacct"`.

---

## Additional checks

- **`--time` directive:** absent from all three sbatch files and all three
  wrappers; each sbatch carries an explicit comment that the omission is
  deliberate.
- **Arrays / dependencies / chained submission / release / resubmission:**
  absent. No `sbatch`, `scontrol`, `scancel`, `srun` or `salloc` anywhere. The
  only `subprocess` call in the entire tree is `gpu_ledger.py:236`,
  `sacct -X -n -P -j <id> -o JobIDRaw,ElapsedRaw,AllocTRES,State` — read-only.
  All three wrappers and both Python entrypoints reject `SLURM_ARRAY_JOB_ID` /
  `SLURM_JOB_DEPENDENCY`.
- **`--gres`:** exactly one occurrence, `scripts/slurm/c04_a0t_small_v1_v7.sbatch:3`,
  `--gres=gpu:a100:1`. The preflight sbatch requests no GPU; the reconcile sbatch
  requests no GPU and its wrapper additionally rejects a non-empty
  `CUDA_VISIBLE_DEVICES` or `SLURM_GPUS_ON_NODE`.
- **Resources:** GPU sbatch = 1 GPU / 8 CPU / 64 GB (exactly as required);
  preflight = 8 CPU / 64 GB, no GPU; reconcile = 1 CPU / 4 GB, no GPU.
  `resources.gpu_count/cpus/ram_gb` = 1/8/64 and are asserted in the preflight,
  the GPU ledger and the producer.
- **No OCR entrypoint, no network/API client, no dev/test path, no cross-dataset
  path, no label reader:** confirmed by targeted grep. No `requests`, `urllib`,
  `httpx`, `aiohttp`, `socket`, `boto3` or any OCR library is imported anywhere;
  the only textual hit is the word "requests" inside a frame-decode docstring.
  The only `label` reference on the data path is `_skip_json_value`, which
  advances the parser past the token syntactically and increments a skip counter;
  the projector then requires the decoded key set to be exactly
  `{id, window_text, language}`. `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
  are exported by the wrapper and asserted by the producer, and both
  `from_pretrained` calls pass `local_files_only=True`. `root_path` rejects any
  `dev`/`test`/`validation`-like path component.
- **Authorization flags in the correct pre-review state:** exactly one flag is
  `true` (`implementation_authorized`); all sixteen others — including
  `preflight_materialization_authorized`, `teacher_authorized`, `gpu_authorized`,
  `slurm_authorized`, `small_tranche_execution_authorized` and
  `post_job_reconciliation_authorized` — are `false`. The preflight wrapper
  blocks on `preflight_materialization_authorized != true`; the reconcile wrapper
  blocks on `post_job_reconciliation_authorized != true`.
- **Unearned review pins are sentinels the code rejects.** Executed against the
  frozen config:

  | entrypoint | result |
  |---|---|
  | `verify_historical_code_resource_authorization` | HALT `code/resource authorization SHA-256 is unpinned` |
  | `verify_payload_review` | HALT `payload hash verdict is not GO` |
  | `verify_gpu_execution_authorization` | HALT `GPU execution verdict is not GO` |
  | `verify_historical_gpu_execution_authorization` | HALT `authorization SHA-256 is unpinned` |
  | `verify_resource_reconciliation_authorization` | HALT `reconciliation verdict is not GO` |
  | `resolve_prompt_hashes(freeze=False)` | HALT `sentinel … outside the authorized freeze run` |
  | `resolve_prompt_hashes(freeze=True)` with materialization `false` | HALT (same) |

- **Config contract normalization is exactly as narrow as documented.** Filling
  the four prompt-hash keys does **not** move `config_contract_sha256` (measured),
  so the v5 impossibility is closed rather than displaced. Mutating
  `selection.suffix`, `resources.small_cap_gpu_seconds`,
  `resources.campaign_aggregate_cap_gpu_seconds`,
  `resources.campaign_first_tranche_phase_cap_gpu_seconds`,
  `resources.guard_item_margin_seconds`, `maps.expected_hashes`,
  `teacher_contract.num_frames`, `model.snapshot_revision` or
  `paths.campaign_gpu_ledger` **does** move it. Only the `authorization` block,
  the four review pins, the four review verdicts and `prompt_hashes` are
  normalized out, each separately bound by a strict stage-authorization manifest.
- **Writer/reader key-set census.** All five `require_exact_keys` contracts in
  `gpu_ledger.py` were extracted by `ast` and compared against the dict literal
  each writer builds:

  | contract | writer | reader | agree today |
  |---|---|---|---|
  | GPU ledger (15 keys) | `preflight.py:411` | `gpu_ledger.py:265` | yes |
  | resource ticket (16) | `preflight.py:432` | `gpu_ledger.py:731` | yes |
  | allocation claim (12) | `gpu_ledger.py:782` | `gpu_ledger.py:367` | yes |
  | provisional GPU usage (14) | `common.build_provisional_gpu_usage` | `gpu_ledger.py:445` | yes |
  | budget guard (7) | `BudgetGuard.accounting_snapshot` | `gpu_ledger.py:470` | yes |

  No live mismatch anywhere. All five are nevertheless maintained as **duplicated
  literals** with no mechanical cross-check — see **H-2**.

---

# Findings

## HIGH H-1 — `campaign-record` writes an unfiltered sacct row into the cross-version accumulator before the stage that would refuse it, and an over-cap write makes the campaign ledger permanently unloadable

**Where.** `gpu_ledger.py:1104-1141` (`campaign_record`, no cap check) and
`gpu_ledger.py:1256-1259` (`reconcile_terminal`, which *does* hard-refuse
`elapsed > 7200`), ordered by `scripts/wrappers/c04_a0t_small_v1_v7_reconcile.sh:63,66`;
`common.py:1264-1303` (`append_campaign_gpu_job`) and `common.py:1188`
(`load_campaign_gpu_ledger`).

**Mechanism, measured against a scratchpad copy of the ledger.** With the frozen
genesis ledger and a terminal sacct row of 7250 s:

```
append_campaign_gpu_job(row) RAISED: HALT_CAMPAIGN_AGGREGATE_CAP: campaign aggregate already exceeds the cap
  file on disk after the raise: aggregate=7250  jobs=1  ledger_revision=1
  subsequent load_campaign_gpu_ledger()          RAISES
  subsequent assert_campaign_aggregate_headroom() RAISES
  retrying the same idempotent append            RAISES
```

Three separate defects compose here:

1. **No cap check on the write side of `campaign-record`.** `reconcile_terminal`
   raises `terminal sacct GPU seconds exceed 7200 cap` for exactly this value,
   but the wrapper runs `campaign-record` *first*, and `campaign_record` calls
   `sacct_row` → `record_campaign_gpu_spend` → `append_campaign_gpu_job` with no
   bound at all. **The permissive stage is the irreversible writer, and it runs
   before the strict one.** This is the failure family the request asked me to
   hunt, applied to the one artifact that deliberately lives outside every
   no-clobber namespace so it can survive an implementation-version bump.
2. **`append_campaign_gpu_job`'s deliberate over-phase branch is dead code.**
   Its comment states "Accounting must never be refused — the spend is already
   real", prints a warning and continues; but the function ends with
   `return load_campaign_gpu_ledger()`, and that call re-applies
   `total > campaign_effective_cap(ledger)` and raises. The designed behaviour is
   unreachable by construction: the row is written and the process then dies.
3. **The write is unrecoverable.** After it, `load_campaign_gpu_ledger()` raises
   on every future call, so `assert_campaign_aggregate_headroom` halts for **every
   future C04 stage in every future namespace** (v8, extraction, adaptation), and
   `record_campaign_gpu_spend`'s idempotent verification path halts too, so the
   row cannot even be re-verified. Repair requires hand-editing a chained,
   payload-hashed ledger — precisely the tamper the chain exists to make evident.
   Meanwhile `campaign-record` exits non-zero, `set -e` aborts the reconcile
   wrapper before `reconcile-terminal`, the per-namespace ledger stays in
   `EXIT_RECORDED_PENDING_SACCT`, `resource_final_state.json` is never published,
   and `review.downstream_review_requires_terminal_resource_state: true` blocks
   every downstream review permanently.

**Failure scenario / reachability.** `sacct` `ElapsedRaw` is measured from SLURM
job start; the wrapper's `timeout` is measured from the wrapper's own
`/proc/uptime` read and is `cap − reserve = 7080 s`, plus `--kill-after=30s`,
plus the EXIT-trap `mark-exit` write. The whole defence is the fixed 120 s
`watchdog_reserve_seconds`, of which 30 s is already spoken for by `kill-after`.
Nothing in the code measures or bounds the job-start-to-wrapper-start offset.
This is round-1 **I-C2**, which the round-2 summary records as "subsumed by the
H-B margin"; it is not — the H-B margin moved the *in-job guard*, not the wrapper
`timeout`, and the wrapper-timeout path stays live whenever any single unit
between guard checks outruns the 300 s margin. I did measure the offset on this
cluster's most recent job (13840: `sacct` Start `05:11:31`, step stderr created
`05:11:31.87`), so today's prolog is sub-second and the realistic worst case is
≈ 7090 s — inside the cap. The trigger is therefore a tail event, not a
certainty, which is why this is High and not Critical. But the reserve is an
unmeasured constant, the run is unrepeatable, and the damage is unbounded and
cross-version.

**What would close it.** (a) Apply the same `> cap_gpu_seconds` refusal in
`campaign_record` that `reconcile_terminal` already applies, *before*
`append_campaign_gpu_job`, so the two stages cannot disagree; and/or (b) make
`append_campaign_gpu_job` genuinely honour its own comment — record the over-cap
spend, set an explicit `phase_over_cap: true` marker that `load_campaign_gpu_ledger`
accepts for reading while `assert_campaign_aggregate_headroom` refuses every
further reservation, and drop the trailing full re-load (or re-load with the
cap assertion relaxed for the just-written revision). (c) Independently, close
the I-C2 root: derive the wrapper's `timeout` from
`cap − (measured job-start-to-wrapper-start offset) − reserve`, or enlarge
`watchdog_reserve_seconds` so the worst-case sacct elapsed is provably < 7200 s.
A CPU-preflight fixture that appends a 7201 s row to an in-memory ledger and
asserts the result is still loadable would have caught the whole thing.

## HIGH H-2 — the writer/reader drift channel that produced round-1 C-A is still open, and the fixture named as the guarantee is a tautology

**Where.** `common.py:108-132` (`PROVISIONAL_USAGE_KEYS`, `BUDGET_GUARD_KEYS`),
`common.py:2035-2062` (`build_provisional_gpu_usage`), `common.py:2132-2137`
(the fixture `provisional_usage_writer_matches_reader_key_set`), versus
`gpu_ledger.py:445-482` (the reader).

**What the round-2 summary claims.** "Both halves now build and validate through
single shared constants, and a fixture asserts they agree."

**What the frozen bytes do.** `gpu_ledger.py`'s `from … common import (…)` list
contains 26 names; `PROVISIONAL_USAGE_KEYS` and `BUDGET_GUARD_KEYS` are **not
among them**. The reader validates against two hand-written set literals, exactly
as in v6. Only the writer half goes through a shared constant. And the fixture
compares `set(build_provisional_gpu_usage(...))` against `set(PROVISIONAL_USAGE_KEYS)`
— but `build_provisional_gpu_usage` already ends with
`require_exact_keys(record, set(PROVISIONAL_USAGE_KEYS), …)`, so the fixture
compares the constant with itself. It cannot observe `gpu_ledger.py` at all.

**Measured.** I reproduced the exact v7-first-draft defect on a scratchpad copy:
added one field to `PROVISIONAL_USAGE_KEYS` and to the builder, left
`gpu_ledger.py` byte-identical.

```
DRIFTED writer: fixture count 45   failing: []
writer now has 15 keys, reader still has 14   ->   drift present: True
=> CPU preflight self-test still PASSES on the drifted writer
```

**Failure scenario.** The current bytes have no live mismatch — I verified all
five writer/reader pairs agree today, so this run would not fail. The finding is
that the *recurrence* channel is intact and invisible: the next edit that adds an
accounting field to the seal record reproduces round-1 C-A verbatim, all 45
preflight fixtures stay green, and the mismatch surfaces only at
`CPU_POST_JOB_RECONCILIATION` — after the full 2 GPU-hour A100 allocation has
been spent, inside a no-clobber namespace that cannot be repaired without moving
`gpu_ledger.py`'s SHA-256, hence `config.implementation_hashes`, hence
`config_contract_sha256`, hence invalidating the values already pinned in the
preflight manifest, genesis ledger and resource ticket. A reviewer granting `GO`
on the strength of the round-2 summary would be granting it on a description that
does not match the code.

**What would close it.** Import `PROVISIONAL_USAGE_KEYS` and `BUDGET_GUARD_KEYS`
into `gpu_ledger.py` and pass them to `require_exact_keys` there, so the two
halves are literally the same object; and, for the four other duplicated
contracts (GPU ledger, resource ticket, allocation claim, and their writers),
either do the same or add a real CPU-preflight fixture that parses the reader's
key set — e.g. via `ast` over `gpu_ledger.py`, or by exposing each reader set as
a named constant the fixture can import — and asserts equality with the writer's.
The fixture must fail when only one side is edited; today it cannot.

## IMPORTANT I-1 — the preflight round-trip closes the `reliability` fragment only; the C-B clamp itself, and every other producer record, are still unexercised before the GPU is spent

`downstream_contract_fixtures()` is real work and I do not want to understate it:
it builds a `provisional_gpu_usage` record through the production builder, and it
validates `build_slot_reliability` output against the *actual* `reliability`
definition of the frozen canonical schema in six states (cosine 1.0, cosine
`1 + 2^-52`, `null` cosine, `missing`, `single_valid`, and all four slots). The
`reliability_rejects_an_unclamped_cosine` fixture does prove the schema bound is
real. But the coverage stops there:

- **`cosine()` lives in `producer.py`, which the CPU preflight never imports.**
  No fixture calls it. The fixture's `over_one = 1.0 + 2.0**-52` is a hand-written
  constant, so deleting `max(-1.0, min(1.0, …))` from `producer.cosine` leaves all
  45 fixtures green and re-opens round-1 C-B in full. The clamp is correct — I
  verified it, and re-measured the underlying overshoot at 572/2000 trials with
  bfloat16-rounded 3584-dim vectors — but it is protected by nothing.
- No fixture ever builds a **full** `canonical_record`, a `prompt_record`, a
  `frame_pack_manifest`, a `resource_final_state` or a `gpu_ledger`/`resource_ticket`/
  `allocation_claim` record and validates it against the contract its consumer
  applies. `validate_schema` at preflight is exercised only against the
  stage-authorization manifest.
- The full canonical record is first schema-validated at `producer.py:1395`,
  inside `canonicalize_dataset`, i.e. **only after all 800 forwards are paid
  for**. That structural exposure is unchanged; only its one known trigger is
  closed.

The blind spot is therefore narrowed, not closed, and it is the same shape as
v6's "no self-test ever rendered a prompt".

**What would close it.** Build one full `canonical_record` and one full
`prompt_record` in the preflight — including the degenerate cases (all slots
`missing`, all `single_valid`, the agreeing `stable` case) — and run them through
`validate_schema` against the frozen schemas; and give the producer's own pure
functions (`cosine` at minimum) a fixture reachable from the CPU preflight, e.g.
by moving `cosine` into `common.py` or by having the preflight import the
producer's pure-CPU helpers. Both are free.

## IMPORTANT I-2 — the mandatory post-loop canonicalization and seal phase is unguarded and must fit inside a margin sized for one item; overrun is a total, unresumable loss

`deadline_check` is called at exactly two sites, both inside the item loop's
`try` block. Everything after the loop — `canonicalize_dataset` for both datasets
(≈ 18 tokenizer + GPU-embedding round-trips per item × 400 items, plus two
`256×3598` projections per item, plus full canonical-record schema validation per
item), the Merkle roots over 400 canonical and 800 prompt records, the ~8 MB seal
write, and the post-publication `idempotent_complete` re-verification — runs with
**no deadline check at all**.

The only budget that phase has is whatever the item loop left, and the item loop
is permitted to run right up to the guard deadline. `guard_item_margin_seconds`
is documented and asserted as buying back "at least one worst-case item"
(`margin <= 0 or margin >= watchdog_seconds` is the only check on its size);
nothing states or enforces that it must also cover the seal phase.

**Failure scenario.** The last item completes just before the guard deadline —
which is the *expected* case if the tranche is sized near the budget (400 items ×
2 forwards against 6780 s implies ~8 s per forward). Canonicalization then has
`margin + claim_duration` ≈ 300-360 s. If it overruns, the wrapper's `timeout`
SIGTERMs mid-canonicalization: exit 124, **no seal, no breach record**, and the
run is unresumable — a fresh SLURM job gets a new job id, so `claim()` takes the
`ledger["jobs"]` non-empty branch and raises `single-use tranche already claimed`.
Up to 2 GPU-hours are lost with nothing but per-item checkpoints to show for it,
and (per H-1) the campaign accumulator then records that spend, which under the
7200 s phase cap forecloses any C04 retry. I could not measure the seal phase's
real duration without running GPU work, so I state the exposure rather than a
probability.

**What would close it.** Add a `deadline_check(guard, "canonicalization and seal")`
immediately before the canonicalization loop, with its own reserve, so an
insufficient remainder becomes a clean exit-40 breach with an accounting record
instead of a mid-phase SIGTERM; and size `guard_item_margin_seconds` (or a new
`guard_seal_phase_margin_seconds`) from a measured seal-phase cost rather than
from one item.

## IMPORTANT I-3 — the campaign accumulator is blind to GPU-seconds burned by an allocation that halts before `claim()` publishes the allocation claim

`campaign_record` keys entirely on `resource/allocation_claim.json`; if it is
absent it prints "no allocation claim; nothing to record" and returns 0. But the
allocation claim is published only at `gpu_ledger.py:796`, after
`validate_gpu_environment`, `create_entry_marker`, the full `verify_gpu_lineage`
chain (preflight manifest + 15 implementation hashes + 15 design hashes + payload
review + GPU authorization), the ledger load/validate/reconcile, and the ticket
validation. Every one of those can HALT — and an A100 allocation that halts there
has still consumed real GPU-seconds that `sacct` will bill and that the amendment
counts ("including **every** GPU-second consumed by the first tranche and any
later C04 extraction/adaptation job", `C04_USER_AMENDMENT_V2.md:45`). The same
applies to a wrapper-level exit before the first Python call. There is no other
code path that records them.

The amounts are small (tens of seconds per failed attempt, since the 16.6 GB
model tree is hashed later, in the producer), so this is Important rather than
High. But this project has now produced three implementation namespaces, and a
pre-claim HALT is exactly the outcome its recent history keeps producing; the
accumulator that exists to make the ceiling machine-checked should not be the one
artifact that forgets them.

**What would close it.** Have `campaign-record` fall back to
`resource/allocation_entry_marker.json` (written by the wrapper before any check)
when the allocation claim is absent, and record the sacct row for that job id
with an explicit `no_claim_published: true` marker; or require the reconcile
stage to run for every C04 GPU job id, claim or no claim.

---

## Non-blocking observations

1. **The pre-model-load containment pass is half self-comparing** (detailed under
   Claim I-1). Requirement met by the per-forward site; worth keeping in the
   record.
2. **`maps.expected_hashes` is protected only by inclusion in the contract hash.**
   No code asserts its literal value, so the prose invariant in
   `prompt_hash_contract` is not machine-checked. Unchanged from round 1.
3. **The campaign ledger's `phase` is not cryptographically bound to any
   authorization artifact.** No code advances it (verified), but an advanced
   ledger is indistinguishable from an authorized one; the amendment's gate is
   human-only. Consider binding `phase_advance_authorization` to the SHA-256 of
   the conditional-tranche code/resource authorization manifest.
4. **`append_campaign_gpu_job` takes no lock on the campaign file.** It is
   protected only by the *per-namespace* `resource/gpu_ledger.lock`, which would
   not exclude a future namespace's reconciler. A lock on the campaign path
   itself would close it.
5. **The reader does not validate `budget_guard` field *values*.** It checks the
   seven keys; `guard_may_only_stop_work_before_an_item: false` would pass.
   Likewise `campaign_gpu_ledger_sha256_at_seal` is recorded but never compared
   to anything.
6. **The GPU wrapper writes the allocation entry marker before any check**
   (`c04_a0t_small_v1_v7.sh:41-64`), so a HALT at the very first Python gate
   leaves a marker whose `jq -e` job-id assertion forbids any later GPU attempt
   in that namespace. Resubmission is forbidden by policy anyway, so this is
   consistent rather than wrong — but it means a config typo caught at
   `validate_gpu_environment` costs a full namespace rebuild.
7. **The exit-40 wrapper branch runs `jq -e` under `set -e`.** A malformed breach
   record would surface as exit 1 rather than exit 40, losing the distinct code
   the branch exists to propagate. Fail-closed, but the distinctness is lost.

---

## Summary table

| # | Severity | Finding |
|---|---|---|
| H-1 | High | `campaign-record` writes an unfiltered sacct row into the cross-version campaign accumulator before `reconcile-terminal`'s 7200 s refusal runs; an over-cap write is performed and *then* rejected, leaving the ledger permanently unloadable for all future C04 work. Subsumes round-1 I-C2, whose root (wrapper `timeout` anchored to a different clock than `sacct`, defended by an unmeasured 120 s reserve) is untouched. |
| H-2 | High | The round-1 C-A recurrence channel is still open: `gpu_ledger.py` validates against hand-maintained key-set literals and does not import `PROVISIONAL_USAGE_KEYS`/`BUDGET_GUARD_KEYS`; the fixture cited as the guarantee compares the writer's constant with itself. Reproduced the original defect with all 45 fixtures green. No live mismatch today. |
| I-1 | Important | Round-1 I-C3 narrowed, not closed: `cosine()`'s clamp — the entire C-B repair — is exercised by no preflight fixture, and no full `canonical_record`/`prompt_record`/`resource_final_state` is ever round-tripped. The full canonical record is still first validated only after all 800 forwards. |
| I-2 | Important | The unguarded post-loop canonicalization + seal phase must fit inside a margin sized for one item; overrun yields exit 124 with no seal, no breach record, and no resumption. |
| I-3 | Important | GPU-seconds burned by an allocation that HALTs before `claim()` publishes `allocation_claim.json` are never recorded in the campaign accumulator, against the amendment's "every GPU-second". |

**Verdict: REVISE (0C / 2H / 3I). No execution authority is conferred.
`authorization.preflight_materialization_authorized` must remain `false` and
`review.code_resource_verdict` must remain `PENDING`. H-1 in particular should be
closed before any GPU allocation, because its damage is not confined to this
namespace: it can brick the accumulator every later C04 stage depends on.**

----- END refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND2.md -----


----- BEGIN refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND3.md (sha256 fbd0829312dc8746c379b42e8418b18eef98c6d6bdd7512660b230ec43d2b212) -----
# C04-A0T-SMALL-v1 v7 — Fresh Independent Code/Resource Review, ROUND 3

Reviewer: fresh independent static reviewer (no exposure to the authoring reasoning)
Date: 2026-07-31
Stage reviewed: `CPU_PREFLIGHT` code/resource review of implementation-v7, third revision
Predecessors, both left byte-intact:
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md` (round 1, `REVISE 2C/2H/3I`)
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND2.md` (round 2, `REVISE 0C/2H/3I`)

Execution authority conferred by this review: **none**.

Note on the deliverable name: the request names
`C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md`, which is the round-1 file. This
review is written to the round-3 path so both earlier reviews survive unaltered.

## Verdict

**REVISE — 0 Critical / 1 High / 3 Important (0C / 1H / 3I)**

All ten prior findings (round-1's eight, round-2's five, with three shared) were
re-derived from the frozen bytes. **Both round-1 Criticals, both round-1 Highs,
round-1 I-C1, and both round-2 Highs are genuinely closed** — in every case I
reproduced the original trigger and confirmed the frozen bytes no longer fail,
rather than accepting the summary. Round-2 I-1 and I-3 are closed for the halves
the round-3 request names. Round-1 I-C2 is **still not closed** and is now three
rounds old.

The one High is **new**, and it is the failure family the request asked me to
enumerate, found in the one place none of the previous rounds looked: the GPU
wrapper's shell preamble. `scripts/wrappers/c04_a0t_small_v1_v7.sh` creates the
single-shot no-clobber namespace with `mkdir -p` **before any authorization
gate** — a gate its two sibling wrappers both carry and it does not. An
out-of-order or unauthorized GPU submission therefore permanently forecloses the
CPU preflight for v7 and forces a full namespace rebuild.

Claims 0, C-1, I-1 and I-2 all reproduced exactly, as in both prior rounds.

## Method and reviewer-boundary compliance

- No SLURM job submitted, held, released, requeued or cancelled. `sacct` and
  `squeue` used read-only only.
- No GPU, teacher, model-weight or frame-decode work. No model file was opened
  at all this session (not even metadata).
- **No file under `/data/jehc223/RGCL` was created, modified or deleted other
  than this review file.** Verified two ways at session end: all 17 pinned
  SHA-256 values re-verify (including
  `artifacts/c04/campaign/gpu_ledger.json` →
  `fc6ca12c32427625d0b80c16b7802ef9a574ced0dbf0288edc3938d217267414`), and
  `find … -newermt '2026-07-31 22:00'` over `scripts/`, `schemas/`, `configs/`
  and `artifacts/c04/` returns nothing. No `.pyc` was written for any v7 module
  (`__pycache__` contains only pre-existing entries, newest a v5 file dated
  2026-07-30).
- `artifacts/c04/a0t_small_v1_impl_v7/` does not exist and was not created.
- No dataset label value was materialized. Both ASR files were read only through
  the frozen `project_train_asr_line` projector (`id`, `window_text`,
  `language`). HateMM identifiers were hashed, counted and compared, never
  printed and never reasoned from as labels.
- All work in `…/scratchpad/review-r3`, outside the repository, with
  `PYTHONDONTWRITEBYTECODE=1` on every invocation. `common.py` was imported only
  from byte-identical scratchpad copies; mutation experiments ran against a copy
  whose sole edit was `ROOT`, repointed to a sandbox tree. `BudgetGuard` was
  executed by extracting its frozen `ClassDef` with `ast`, never by importing
  `producer.py`. `py_compile` was not used.

### Hash verification (all 17, before and after)

Every pinned SHA-256 in the request table matched disk at session start and
re-matched at session end (17/17, byte-for-byte diff against the request table).
In addition:

- all 15 `configs/c04/c04_a0t_small_v1_v7.json → implementation_hashes` verify
  against disk (15/15) and equal the request table;
- all 15 `frozen_design_hashes` verify against disk (15/15);
- the config is **not** listed inside its own `implementation_hashes`, so the
  authorization flips the pipeline requires between stages cannot break
  `verify_bound_file_map`.

### v6 predecessor unmodified

- `artifacts/c04/a0t_small_v1_impl_v6/freeze/preflight_manifest.json` is
  self-consistent (`payload_sha256` reproduces) and **all 14** of its
  `staged_output_hashes` verify byte-for-byte on disk.
- all 15 entries of `configs/c04/c04_a0t_small_v1_v6.json → implementation_hashes`
  verify against disk.
- every v6 artifact carries a single mtime, `2026-07-31 05:11` (job 13840),
  predating every v7 source file (`2026-07-31 21:45/21:46`).

---

## Round-1 findings — closure status

| # | Round-1 finding | Status | How I re-derived it |
|---|---|---|---|
| C-A | reconciler's exact-key set had not grown with the writer | **CLOSED, by construction** | `ast` census: `gpu_ledger.py` imports `PROVISIONAL_USAGE_KEYS` and `BUDGET_GUARD_KEYS` (28-name import list) and passes them to `require_exact_keys` at lines 453/459; the writer `build_provisional_gpu_usage` validates against the same objects. One-sided drift can no longer exist for this contract. |
| C-B | `proposition_cosine` could exceed schema `maximum: 1` | **CLOSED, and the fixture is real** | Deleting the clamp from a scratchpad copy turns `cosine_of_identical_vectors_is_within_the_schema_bound` **red** (measured, see Claim I-3 below). A full canonical record with the clamped value validates. |
| H-A | campaign write side reachable only after a seal | **CLOSED** | `campaign-record` is a distinct mode keyed on `allocation_claim.json` with a marker fallback, run first in the reconcile wrapper under `set -e`. |
| H-B | in-job guard had no margin over the wrapper `timeout` | **CLOSED** | Frozen `BudgetGuard` executed over a 5×4 grid of claim/verification durations: lead is `300 + claim_duration`, **never below 300 s**, and independent of verification time. `SLURM_JOB_START_TIME` appears nowhere except one docstring. |
| I-C1 | accumulator enforced 28800 s, not the binding 7200 s | **CLOSED** | Effective cap measured as `min(phase, aggregate) = 7200`; a recorded spend of 1 s refuses the next 7200 s reservation; ten distinct phase/cap mutations all halt. |
| I-C2 | ~90 s margin to the hard 7200 s ceiling; breach unrecoverable | **NOT CLOSED — three rounds old** | see **Important I-1**. The blast radius shrank (the campaign ledger no longer bricks, and the in-job guard makes the `timeout` path a tail event), so this is now Important rather than the round-2 High that subsumed it. |
| I-C3 | preflight never round-trips a record against a downstream contract | **PARTIALLY CLOSED** | see **Important I-3**. Narrowed a second time; two channels remain. |

Both round-1 non-blocking observations re-checked and unchanged: the
pre-model-load containment pass still self-compares in one half (the message
assembly half is non-vacuous only at `producer.py:1709`, which does run before
every forward), and `maps.expected_hashes` is protected only by inclusion in the
contract hash (mutating it moves `config_contract_sha256`; no code asserts its
literal value).

## Round-2 findings — closure status

| # | Round-2 finding | Status | How I re-derived it |
|---|---|---|---|
| H-1 | accumulator could brick itself on an over-cap write | **CLOSED** | Executed five append magnitudes (0/100/7200/7250/30000 s) against a sandbox ledger. See the table under Claim I-3. |
| H-2 | reader restated the writer's key set; the fixture self-compared | **CLOSED as scoped** | `ast` proof above. Residual: three *other* writer/reader contracts remain duplicated literals — all three agree today (measured) — see **Important I-3**. |
| I-1 | `cosine` unreachable from the preflight; no full-record round-trip | **CLOSED for the clamp; the wider half remains** | Clamp deletion now turns a fixture red (measured). No fixture yet builds a full `canonical_record`/`prompt_record`/`resource_final_state`. See **Important I-3**. |
| I-2 | post-loop canonicalization + seal phase unguarded | **CLOSED** | `guard.require_remaining(600, "the canonicalization and seal phase")` sits at `producer.py:1822`, inside the same `try` whose `except BudgetDeadlineReached` publishes the accounting-only breach record and returns 40. Measured effective budget for that phase is **≈900 s**, not 600 (see below). |
| I-3 | GPU-seconds burned before `claim()` were recorded nowhere | **CLOSED** | The wrapper writes the entry marker at `c04_a0t_small_v1_v7.sh:39-64`, before its first Python call; `campaign_record` falls back to it, and with neither artifact present it prints and returns 0 rather than fabricating a row. |

---

## Claim 0 — no scientific semantic changed: **CONFIRMED**

### Selection re-derived from the v7 frozen rule reproduces the v6 frozen allowlists

Both train ASR files read through the v7 projector, ranked by the v7
`selection_digest` with the v7 tie-break, first 200 taken:

| dataset | train N | recomputed order == v6 frozen allowlist | sha256(newline-joined ordered ids) |
|---|---|---|---|
| HateMM | 744 | **True** | `091fb1826cbc7f80…` |
| MHC_zh | 579 | **True** | `6c98c0d75891ce43…` |

Also reproduced per dataset, against the v6 frozen artifacts: every stored
`selection_sha256` (200/200), contiguous ranks 0..199, all 200
`transcript_sha256` and all 200 `transcript_scalar_count`. That last pair
independently pins the transcript normalization, cap, head/tail split and
separator as unchanged.

### Prompt hashes

Recomputed from the v7 sources:

```
system   1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048
A        cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b
B        9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314
combined a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a
```

All four equal `artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json`
and the in-module fixture literals. **No prompt byte changed.**

### Version-token-normalized tree diff — every residual line accounted for

Both trees copied to the scratchpad, `s/v6|V6/vTOK/` and `s/v7|V7/vTOK/`
applied, diffed file by file:

| file | changed lines | accounted for by |
|---|---|---|
| **all 5 schemas** | **0** | — (so the canonical schema's `maximum: 1` is byte-identical to v6) |
| `*_preflight.sh`, `*_preflight.sbatch`, `*.sbatch`, `*_reconcile.sbatch` | 0 | — |
| `*_reconcile.sh` | 8 | H-A only (`campaign-record` before `reconcile-terminal`) |
| `*_v7.sh` | 18 | I-3 only (exit-40 branch, breach-record `jq`, zero-exit breach guard) |
| `preflight.py` | 33 | I-3 only |
| `gpu_ledger.py` | 172 | I-3 / H-A / C-A / H-2 only |
| `common.py` | 813 | C-1 + I-1 + I-2 + I-3, **pure additions** |
| `producer.py` | 497 | C-1 + I-1 + I-3 |
| `config.json` | 137 | version tokens, `v7_scope` prose, three `resources` keys, two `paths`, refreshed `implementation_hashes` |

A function-level AST diff makes this exact:

- `common.py`: **33 additions, 0 removals, exactly one changed definition**
  (`self_test_fixtures`).
- `gpu_ledger.py`: 2 additions (`campaign_record`, `record_campaign_gpu_spend`),
  5 changed (`main`, `reconcile_terminal`,
  `validate_cpu_reconciliation_environment`, `validate_gpu_environment`,
  `verify_reconciliation_lineage`).
- `producer.py`: 5 additions (`BudgetGuard`, `BudgetDeadlineReached`,
  `BUDGET_BREACH_EXIT_CODE`, `assert_teacher_visible_precondition`,
  `publish_budget_breach_record`), **1 removal (`cosine`, moved to `common.py`)**,
  6 changed (`build_messages`, `deadline_check`, `main`, `verify_authorization`,
  `verify_claimed_resource`, `verify_execution_lineage`).
- `preflight.py`: 1 changed (`verify_static_config`).

Nothing else moved. `SYSTEM_PROMPT`, `_SCHEMA_TEXT`, `PROMPT_A`, `PROMPT_B`,
`SELECT_TAG`, `SELECT_SUFFIX`, `SELECT_N`, `NUM_FRAMES`, the transcript
constants, the confidence/cosine thresholds, the five-rate KILL taxonomy,
`build_slot_reliability`, `render_slot`, `materialize_role_map`,
`dense_rademacher_payload`, `parse_teacher_response`, `q_product`,
`safe_vector`, `merkle_root` are untouched — none appears in the added, removed
or changed sets.

`self_test_fixtures()` now returns **47** checks (round 2: 45; round 1: 37;
v6: 25). All 47 pass on the frozen bytes, and no fixture name is duplicated (a
duplicate would be silently dropped by `dict(self_test_fixtures())` in
`preflight.run_self_tests`).

---

## Claim C-1 — the prompt renderer: **CONFIRMED**

1. **Could the v6 form ever have succeeded?** No. Both `PROMPT_A.format(transcript="x")`
   and `PROMPT_B.format(...)` raise `KeyError: '"source_relation"'` against the
   frozen templates; the defect is unconditional on both forms.
2. **Exact substitution?** Yes: `render_prompt(form, t) == PROMPTS[form][:-len("{transcript}")] + t`
   for both forms, including transcripts containing CJK, newlines and literal
   `{braces}`; the module templates are not mutated.
3. **Any prompt byte changed?** No — the four hashes match the v6 frozen artifact.
4. **Any surviving `.format(transcript=` call site?** None. The two textual
   occurrences are a docstring (`common.py:221`) and the deliberate regression
   fixture `prompt_render_regression_str_format_would_raise` (`common.py:2262`),
   which asserts the v6 form still raises `KeyError`.
5. **Guards non-vacuous?** The four pre-substitution guards (unknown form,
   non-`str` transcript, `count != 1`, non-terminal placeholder) are genuine and
   two are exercised by `prompt_render_rejects_unknown_form_and_non_string`. The
   two post-hoc guards remain weak (`startswith(prefix)` is a tautology given a
   unique terminal placeholder; `endswith(transcript)` is vacuous for an empty
   transcript). Unchanged from rounds 1-2; still an observation.

---

## Claim I-1 — teacher-visible containment: **CONFIRMED**

1. **Before model load?** Yes. `assert_teacher_visible_precondition(inputs)` at
   `producer.py:1677`; `from_pretrained` at `producer.py:1686`. Repeated per item
   inside `one_forward` at `producer.py:1709`, after `build_messages` and before
   `apply_chat_template`/`generate`.
2. **Strict in both directions?** Yes. `teacher_visible_texts` raises on a wrong
   message-list length, unexpected message keys, an unexpected role, non-list or
   empty content, a content part that is not a dict or lacks `type`, a text part
   with extra keys or a non-string body, a video part with extra keys, a frame
   list whose length ≠ 8, any frame that is `str`/`bytes`/`bytearray`/`Path`, any
   unknown content type, and a final census requiring exactly one video part and
   exactly two text parts. Six are pinned by fixtures.
3. **What is banned, is it wide enough?** Measured on the real tranche: **402**
   tokens — all 200 HateMM + all 200 MHC-ZH selected identifiers plus
   `hate_video_` and `non_hate_video_` — with both datasets' identifiers banned
   in both datasets' prompts, so a cross-item leak is refused as firmly as a
   self-leak. Matching is over `{token, NFKC(token), casefold(NFKC(token))}`
   against both `NFKC(text)` and `casefold(NFKC(text))`. The wider protection is
   the equality `texts == [SYSTEM_PROMPT, render_prompt(form, transcript)]`: the
   only variable content reaching the teacher is the transcript, so the
   amendment's broader ban (prediction, neighbor, rank, margin, error status,
   dataset statistic, fold role, intended use) is satisfied structurally rather
   than by enumeration.
4. **False positives on the real 400 transcripts?** **None — 800 accepted, 0
   rejected.** As a margin, the 402-token ban list was scanned against **all
   1323** train transcripts (744 + 579): **0** rows contain any banned token.
   Shortest banned token is 11 characters, so accidental substring collision is
   not a live risk.
5. **Can it pass vacuously?** No, at the per-forward site. Measured: an empty ban
   list is rejected, and an identifier absent from the list is rejected.

### The HateMM ID-label asymmetry — stated explicitly, as requested

**The asymmetry is handled correctly.** Every HateMM training identifier is
`hate_video_*` or `non_hate_video_*`, so the identifier *is* the binary label;
MHC-ZH identifiers are opaque BiliBili `BV` codes carrying no label information.
Therefore **the sealed ID-only allowlist delivers label containment for MHC-ZH
only, and none at all for HateMM.** `LABEL_BEARING_ID_SUBSTRINGS` encodes exactly
this (`("hate_video_", "non_hate_video_")` for HateMM, `()` for `MHC_zh`), the
`common.py` comment block states it, and `config.json →
v7_scope.I1_teacher_visible_containment` records it. The consequence is drawn
correctly: for HateMM, selection label-blindness rests on *hash reproduction of
the selection rule* (independently reproduced above, 200/200 on both datasets)
and teacher label-blindness on this *runtime* check — not on the allowlist.

---

## Claim I-2 — the selection self-test is a known-answer vector: **CONFIRMED**

Both pinned digests recomputed from first principles outside the module by hand
concatenation, and both match:

```
sha256(utf8("C04-A0T-SMALL-v1" + "HateMM" + "c04-known-answer-vector" + "20260729"))
  = 871e0363e1b01a823f09a5a0bb9187749da74f1dfe8e454a733e21c218f6a384
sha256(utf8("C04-A0T-SMALL-v1" + "MHC_zh" + "c04-known-answer-vector" + "20260729"))
  = 41bfb637f5cbb26bb0b9edfd44d19a3775d8df1712f9b49bde667863fbd37134
```

Literals independent of the module's own code path. `selection_dataset_and_id_sensitivity`
additionally requires three distinct digests. The identifier is synthetic and
belongs to neither dataset, so the fixture pins the rule without naming a
label-bearing real id.

---

## Claim I-3 — both ceilings machine-checked and fail-closed: **SUBSTANTIALLY DELIVERED**

### Tranche ceiling (7200 s) — guard leads the wrapper, and the seal phase is now guarded

`BudgetGuard.at_job_start` is constructed exactly once, in
`verify_claimed_resource`; the deadline is stored and never recomputed. There are
exactly two `deadline_check` sites — `producer.py:1758` at the item boundary and
`producer.py:1704` inside `one_forward` — both strictly *before* a unit of work,
plus one `require_remaining` at `producer.py:1822` before the canonicalization
phase. The guard never truncates, shortens or rewrites an output;
`BudgetDeadlineReached` is caught only by the outer handler, which publishes an
accounting-only record and returns 40.

I executed the frozen `BudgetGuard` class (extracted with `ast`, never imported
from the repository path) over a grid of claim durations `c` and
lineage/model-hash durations `v`, with the ticket watchdog at
`cap − reserve = 7080 s`:

| claim `c` | verify `v` | guard fires @entry+ | wrapper timeout @entry+ | guard lead | seal phase may start until entry+ | seal budget before SIGTERM |
|---|---|---|---|---|---|---|
| 0 | 0…120 | 6780 | 7080 | **300 s** | 6180 | **900 s** |
| 5 | 0…120 | 6775 | 7080 | **305 s** | 6175 | **905 s** |
| 30 | 0…120 | 6750 | 7080 | **330 s** | 6150 | **930 s** |
| 60 | 0…120 | 6720 | 7080 | **360 s** | 6120 | **960 s** |
| 120 | 0…120 | 6660 | 7080 | **420 s** | 6060 | **1020 s** |

The lead is `guard_item_margin_seconds + c` and is **independent of `v`**,
because `at_job_start` subtracts elapsed-since-entry from a watchdog that already
had `c` subtracted. Round-1 H-B is closed. The degenerate cases all halt:
allocation entry in the future, no budget remaining, margin `0` or `≥ watchdog`,
seal reserve `0` or `≥ watchdog`.

**Round-2 I-2 is closed.** The post-loop phase is no longer unguarded, and its
effective budget is ~900 s rather than the ~300 s round 2 measured. The reserve's
size is still a human estimate rather than a measurement, but the CPU half is
cheap enough to make 900 s plausible: I built a **full** canonical record exactly
as `canonicalize_dataset` builds it (25.8 KB, real `q_product`/`f32le_b64`/
`apply_role`/`fixed_projection` shapes) and Draft7 validation costs **2.4 ms**,
so 400 records cost ≈1 s and the post-publication re-verification's ~2400
validations ≈6 s. The dominant term is the ~7200 small embedding lookups, not the
CPU work. Recorded as an observation, not a finding.

The breach record was inspected field by field: lineage, job id, terminal state,
exit code, both caps, the guard snapshot, per-dataset completed counts, teacher
and frame-pack counters, `outputs_truncated_or_altered: 0`,
`seal_published: false`, `no_scientific_verdict_is_published_by_a_budget_breach: true`.
No metric, no teacher output, no reliability rate, no CONTINUE/KILL verdict. The
wrapper's exit-40 branch `jq -e`s exactly the three fields the record carries,
and a zero-exit run that left a breach record is refused with exit 3.

### Campaign ceiling — read side fail-closed and phase-scoped

`assert_campaign_aggregate_headroom` is called from `validate_gpu_environment`,
the first statement of `claim()`, i.e. before `create_entry_marker`, before
`verify_gpu_lineage`, and well before the allocation claim and the ticket
consumption record are published — genuinely before the single-use ticket is
consumed. It is also called by the CPU preflight before the namespace is
materialized, and by the producer before any model or data work.

Fail-closed matrix, executed against a sandbox copy (repository untouched):

| mutation | result |
|---|---|
| pristine genesis (0 s spent) | **accepted** (7200 ≤ 7200) |
| ledger absent | HALT `campaign ledger is absent` |
| `payload_sha256` tampered | HALT `payload mismatch` |
| aggregate ≠ Σ rows | HALT `aggregate does not equal its rows` |
| foreign `schema_version` | HALT `foreign campaign ledger schema` |
| foreign `run_id` | HALT `foreign campaign ledger run id` |
| aggregate cap raised to 999999 | HALT `cap is not the amendment cap` |
| aggregate cap lowered to 7200 | HALT `cap is not the amendment cap` |
| phase advanced, phase cap left at 7200 | HALT `phase cap does not match the phase` |
| phase cap raised alone | HALT `phase cap does not match the phase` |
| unknown phase | HALT `unknown campaign phase` |
| first-tranche phase carries an advance token | HALT `first-tranche phase carries an advance token` |
| row chain break | HALT `chain break` |
| well-formed, 1 s already spent | loads; next 7200 s reservation **REFUSED** |
| well-formed, 7000 s already spent | loads; next 7200 s reservation **REFUSED** |
| well-formed, 7250 s (over-cap) already spent | loads; next 7200 s reservation **REFUSED** |
| non-positive reservation | HALT `requested a non-positive reservation` |

**Effective ceiling is 7200 s today.** The only route to 28800 s is a coordinated
hand edit that changes `phase`, `phase_cap_gpu_seconds` and
`phase_advance_authorization` together and reseals the payload — i.e. the human
gate the amendment intends. No code path writes any of the three. Round-1 I-C1
stays closed.

**Opening zero is evidence-backed.** I read `sacct` myself, read-only:

```
13805|c04_a0t_small_v1_v5_preflight|0 |billing=8,cpu=8,mem=64G,node=1|FAILED
13840|c04_a0t_small_v1_v6_preflight|19|billing=8,cpu=8,mem=64G,node=1|COMPLETED
```

Both rows match `genesis_evidence` verbatim including `alloc_tres`, elapsed
seconds and states; neither carries `gres/gpu`, so `gpu_seconds: 0` is correct
for both. A full accounting sweep for `c04` job names returns exactly these two
rows, corroborating `these_are_the_only_c04_jobs_in_the_accounting_record: true`.

### Write side — round-2 H-1 closed (the accumulator can no longer brick itself)

Executed against a sandbox ledger at five magnitudes:

| appended `gpu_seconds` | append raised | on-disk aggregate | over-cap flag | later `load` | next 7200 s reservation | idempotent re-verify | chained 2nd append |
|---|---|---|---|---|---|---|---|
| 0 | none | 0 | `false` | OK | accepted | OK | OK |
| 100 | none | 100 | `false` | OK | REFUSED | OK | OK |
| 7200 | none | 7200 | `false` | OK | REFUSED | OK | OK |
| **7250** | **none** | **7250** | **`true`** | **OK** | **REFUSED** | **OK** | **OK** |
| 30000 | none | 30000 | `true` | OK | REFUSED | OK | OK |

I also traced every check in `load_campaign_gpu_ledger` against what
`append_campaign_gpu_job` writes and confirmed **no check can fail on the
just-written file**: payload digest, schema, run id, aggregate cap, phase,
phase cap, advance token, chain links, row digests, aggregate-equals-rows and
head link are all satisfied by construction, and the cap check has been removed
from the load path entirely. Every rejecting check (head race, duplicate job,
non-integer seconds) runs before the write, and the new
`aggregate_exceeds_effective_cap` flag and both `campaign_effective_cap` calls
are evaluated before `os.replace`. The `load_campaign_gpu_ledger()` on the return
line is therefore genuinely non-raising, and the designed over-cap branch is no
longer dead code.

### Write side — round-1 H-A and round-2 I-3 closed

| path | `allocation_claim.json` | `allocation_entry_marker.json` | campaign row written? |
|---|---|---|---|
| exit 40 budget breach | yes | yes | **yes** |
| watchdog TERM/KILL (124/137/143) | yes | yes | **yes** |
| OOM / decode failure / any producer HALT after claim | yes | yes | **yes** |
| fully successful sealed run | yes | yes | **yes** (and again idempotently from `reconcile_terminal`) |
| HALT *before* `claim()` publishes the claim | no | **yes** (wrapper writes it at `c04_a0t_small_v1_v7.sh:39-64`, before the first Python call) | **yes, via the fallback** |
| allocation never entered | no | no | **no** — prints "no allocation entry; nothing to record", returns 0 |

The fallback cannot fabricate a row: with neither artifact present it records
nothing, and with the marker present it still requires `sacct` to show a terminal
row whose `gres/gpu` count is exactly 1. `record_campaign_gpu_spend` verifies an
already-present row instead of appending, so the two calls in one reconcile
wrapper run (once from `campaign-record`, once from `reconcile-terminal`) cannot
double-count.

---

## Additional checks

- **`--time` directive:** absent from all three sbatch files and all three
  wrappers; each sbatch carries an explicit comment that the omission is
  deliberate.
- **Arrays / dependencies / chained submission / release / resubmission:**
  absent. No `sbatch`, `scontrol`, `scancel`, `srun` or `salloc` token anywhere
  in the v7 set. The only `subprocess` call in the entire tree is
  `gpu_ledger.py:241`, `sacct -X -n -P -j <id> -o JobIDRaw,ElapsedRaw,AllocTRES,State`
  — read-only. All three wrappers and both Python entrypoints reject
  `SLURM_ARRAY_JOB_ID` / `SLURM_JOB_DEPENDENCY`.
- **`--gres`:** exactly one occurrence, `scripts/slurm/c04_a0t_small_v1_v7.sbatch:3`,
  `--gres=gpu:a100:1`. The preflight sbatch requests no GPU; the reconcile sbatch
  requests no GPU and its wrapper additionally rejects a non-empty
  `CUDA_VISIBLE_DEVICES` or `SLURM_GPUS_ON_NODE`.
- **Resources:** GPU sbatch = **1 GPU / 8 CPU / 64 GB** exactly; preflight = 8
  CPU / 64 GB, no GPU; reconcile = 1 CPU / 4 GB, no GPU. `resources.gpu_count/cpus/ram_gb`
  = 1/8/64 and are asserted in the preflight, the GPU ledger and the producer.
- **No OCR entrypoint, no network/API client, no dev/test path, no cross-dataset
  path, no label reader:** confirmed by targeted grep for `requests`, `urllib`,
  `httpx`, `aiohttp`, `socket`, `boto3`, `tesseract`, `easyocr`, `paddleocr`,
  `pytesseract` — the only textual hit is the word "requests" inside a
  frame-decode docstring. The only `label` reference on the data path is
  `_skip_json_value`, which advances the parser past the token syntactically and
  increments a skip counter; the projector then requires the decoded key set to
  be exactly `{id, window_text, language}`. `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1` are exported by the wrapper and asserted by the
  producer, and both `from_pretrained` calls pass `local_files_only=True`.
  `root_path` rejects any `dev`/`test`/`validation`-like path component.
- **Authorization flags in the correct pre-review state:** exactly one flag of
  seventeen is `true` (`implementation_authorized`); all sixteen others —
  including `preflight_materialization_authorized`, `teacher_authorized`,
  `gpu_authorized`, `slurm_authorized`, `small_tranche_execution_authorized` and
  `post_job_reconciliation_authorized` — are `false`. All four review pins are
  sentinels and all four verdicts are `PENDING`. `maps.expected_hashes` is the
  documented sentinel string.
- **Unearned review pins are sentinels the code rejects.** Executed against the
  frozen config:

  | entrypoint | result |
  |---|---|
  | `verify_historical_code_resource_authorization` | HALT `code/resource authorization SHA-256 is unpinned` |
  | `verify_payload_review` | HALT `payload hash verdict is not GO` |
  | `verify_gpu_execution_authorization` | HALT `GPU execution verdict is not GO` |
  | `verify_historical_gpu_execution_authorization` | HALT `authorization SHA-256 is unpinned` |
  | `verify_resource_reconciliation_authorization` | HALT `reconciliation verdict is not GO` |
  | `resolve_prompt_hashes(freeze=False)` | HALT `sentinel … outside the authorized freeze run` |
  | `resolve_prompt_hashes(freeze=True)`, materialization `false` | HALT (same) |
  | `resolve_prompt_hashes(freeze=True)`, materialization `true` | accepted — the single intended relaxation |

- **Config contract normalization is exactly as narrow as documented.** Measured:
  `authorization.*`, the four review pins, the four review verdicts and
  `prompt_hashes.*` do **not** move `config_contract_sha256` (including filling
  all four prompt hashes at once — the v5 impossibility stays closed). Fourteen
  other mutations — `selection.suffix`, all four resource caps/margins,
  `maps.expected_hashes`, `teacher_contract.num_frames`,
  `model.snapshot_revision`, `paths.campaign_gpu_ledger`,
  `run.implementation_version`, `schemas.canonical_record`,
  `reliability.proposition_agreement_cosine_min`,
  `datasets.HateMM.train_asr_sha256` — **do** move it.
- **No live writer/reader mismatch anywhere.** All five `require_exact_keys`
  contracts extracted by `ast` and compared against the dict literal each writer
  builds: GPU ledger 15/15, resource ticket 16/16, allocation claim 12/12,
  provisional GPU usage (shared constant), budget guard (shared constant).
- **No live schema mismatch anywhere.** A **full** canonical record built exactly
  as `canonicalize_dataset` builds it validates against the frozen schema in all
  four reliability regimes — all-`stable` with the clamped agreeing cosine,
  all-`missing`, `single_valid`, and `conflict` — and a full prompt record
  validates for both a normal decode and the zero-frame degenerate case
  (`total_frame_indices: 0`, `requested_indices: []`) at `sequence_index` 399.

---

# Findings

## HIGH H-1 — the GPU wrapper creates the single-shot no-clobber namespace before any authorization gate, so one out-of-order submission forecloses v7 permanently

**Where.** `scripts/wrappers/c04_a0t_small_v1_v7.sh:39-40`
(`readonly ENTRY_MARKER=…/resource/allocation_entry_marker.json` then
`mkdir -p "$(dirname "$ENTRY_MARKER")"`), versus
`scripts/analysis/c04_a0t_small_v1_v7_preflight.py:324-326`
(`namespace = root_path(ARTIFACT_ROOT); if namespace.exists(): raise FileExistsError`).

**Mechanism.** The GPU wrapper's only pre-Python checks are `SLURM_JOB_ID`
non-empty and no array/dependency (lines 31-38). It then unconditionally
`mkdir -p`s `artifacts/c04/a0t_small_v1_impl_v7/resource`, writes the allocation
entry marker, and only *afterwards* invokes `gpu_ledger.py --mode claim`, whose
`validate_gpu_environment` is the first code that ever looks at
`cfg["authorization"]`. **The two sibling wrappers both carry a config gate and
this one carries none** — measured by grep: `…_preflight.sh` has one
`authorization` reference (its `preflight_materialization_authorized` block),
`…_reconcile.sh` has four (a nine-clause `jq -e` requiring
`post_job_reconciliation_authorized`, every other flag false, the reconciliation
verdict `GO` and a 64-hex pin), and `…_v7.sh` has **zero**. Its three `jq -e`
calls are the entry-marker match, the breach-record check and the seal check —
none of them authorization.

**Measured.** I replayed the wrapper's lines 31-64 byte-faithfully against a
sandbox root, with `claim` stubbed:

```
namespace dir now exists : YES
entry marker written     : YES
...and only NOW would 'gpu_ledger.py --mode claim' run its first check.
frozen config says gpu_authorized = false, teacher_authorized = false

preflight.py:325  if namespace.exists(): raise FileExistsError -> namespace.exists() == True
=> CPU preflight would raise: FileExistsError("no-clobber namespace refusal: artifacts/c04/a0t_small_v1_impl_v7")
```

**Failure scenario.** Someone submits `scripts/slurm/c04_a0t_small_v1_v7.sbatch`
before the CPU preflight has run — today's state, with `gpu_authorized: false`
and `preflight_materialization_authorized: false`. SLURM grants an A100. The
wrapper creates the namespace and the marker. `claim` then halts within seconds
(`assert_literal_prompt_hash_binding` fires on the prompt-hash sentinel, and
`authorization.teacher_authorized` would fire immediately after). The EXIT trap
runs `mark-exit`, which finds the marker, stamps it, finds no ledger and
returns. The A100 is released having done nothing. But:

- `artifacts/c04/a0t_small_v1_impl_v7/` now exists, so
  `preflight.preflight()` raises `FileExistsError` on **every** future run. v7
  can never be materialized;
- deleting the directory is not a remedy the design admits — the whole no-clobber
  scheme rests on the namespace being created exactly once, atomically, by the
  preflight's `os.rename`;
- the surviving path is a full namespace rebuild to v8, which is precisely the
  cost this campaign has now paid three times (v5 → v6 → v7).

This is the failure family the request asked me to enumerate, in its purest form:
an irreversible, single-shot resource is consumed by an unconditional `mkdir -p`
**before** the check that would reject the run. It is also the cheapest one to
close and the only one where the correct guard already exists, verbatim, in two
sibling files.

**What would close it.** Put an authorization gate ahead of line 40 in
`scripts/wrappers/c04_a0t_small_v1_v7.sh`, in the same style the reconcile
wrapper already uses — `jq -e` on `.authorization.gpu_authorized`,
`.teacher_authorized`, `.slurm_authorized`, `.small_tranche_execution_authorized`
all `true`, `.review.gpu_execution_verdict == "GO"` and a 64-hex
`gpu_execution_authorization_sha256` — plus a plain `[[ -f
artifacts/c04/a0t_small_v1_impl_v7/freeze/preflight_manifest.json ]]` test, so a
GPU submission that precedes the preflight refuses before it touches the
filesystem. All of it is `exit 2` before any `mkdir`.

## IMPORTANT I-1 — round-1 I-C2 is still open: the wrapper `timeout` is anchored to a different clock than `sacct`, defended by an unmeasured 120 s reserve, and an over-run makes the final state permanently unpublishable

**Where.** `scripts/wrappers/c04_a0t_small_v1_v7.sh:14-15,101-105`
(`/proc/uptime` anchor, `timeout … --kill-after=30s "${C04_ACTIVE_WATCHDOG_SECONDS}s"`)
versus `gpu_ledger.py:1253-1256` (`terminal sacct GPU seconds exceed 7200 cap`),
`gpu_ledger.py:916-923` (`terminal GPU seconds outside [0,7200]`) and
`schemas/c04/c04_a0t_small_v1_v7_resource_final_state.schema.json:63-80`
(`maximum: 7200` on three fields, `cap_gpu_seconds: {"const": 7200}`).

**Mechanism, re-derived.** `claim()` returns `active = 7080 − c`, where `c` is
measured from the wrapper's own `/proc/uptime` read. `timeout` starts after the
claim and therefore fires at `P0 + 7080` where `P0` is the SLURM-prolog-to-wrapper
offset, plus 30 s of `--kill-after`, plus the EXIT-trap `mark-exit` write —
so worst-case wall time ≈ `P0 + 7113`. `sacct` `ElapsedRaw`, however, is measured
from SLURM job start, i.e. from `P0` earlier. Nothing in the code measures or
bounds `P0`; the entire defence is the fixed `watchdog_reserve_seconds = 120`, of
which 30 s is already spoken for by `--kill-after`. Round 2 measured `P0` at
sub-second on job 13840, so the realistic margin is ≈85 s.

**Failure scenario.** If terminal `sacct` elapsed exceeds 7200 s:
`campaign-record` correctly records the true spend (accounting is never refused —
that part is right), but `reconcile_terminal` then raises
`HALT_RESOURCE_CAP: terminal sacct GPU seconds exceed 7200 cap`;
`strict_validate_terminal_ledger` would reject it independently; and the
`resource_final_state` schema pins three fields at `maximum: 7200`, so no final
state can ever be published for that run. The reconcile wrapper's recovery branch
requires the ledger to already be `SACCT_TERMINAL_RECONCILED`, which it never
becomes, so it propagates the failure. `review.downstream_review_requires_terminal_resource_state: true`
then blocks every downstream review of that namespace permanently, and the run
cannot be repaired without editing `gpu_ledger.py`, which moves its SHA-256,
hence `config.implementation_hashes`, hence `config_contract_sha256`, hence the
values already pinned inside the no-clobber preflight manifest, genesis ledger
and resource ticket.

**Why this is Important and no longer High.** The round-2 High that subsumed it
was rated on the brick — an over-cap write making the cross-version accumulator
permanently unloadable. That is genuinely fixed (measured above). What remains is
confined to the namespace. And the in-job guard now stops the producer at
`entry + 6780 − c`, so the wrapper `timeout` fires only if a single item outruns
the 300 s margin or the seal phase outruns ~900 s — a tail, not a certainty. But
it is a hard ceiling whose breach is unrecoverable, defended by a constant nobody
has measured, and it is now three rounds old.

**What would close it.** Either derive the wrapper's `timeout` from
`cap − (measured job-start-to-wrapper-start offset) − reserve` — the offset is a
two-line `sacct -X -n -P -j "$SLURM_JOB_ID" -o Start` or, more simply, a larger
`watchdog_reserve_seconds` chosen so `reserve > kill_after + mark_exit + P0_max`
provably holds — or make an over-cap terminal sacct row a *recorded* over-run:
written to the per-namespace ledger and a distinct final state with the schema
bound relaxed to a flagged maximum, rather than an unrecoverable halt. The
campaign accumulator already demonstrates the second pattern working.

## IMPORTANT I-2 — `reconcile-terminal` is still seal-dependent, so the exit-40 breach path that I-3 introduced cannot complete its own mandated post-job stage

**Where.** `gpu_ledger.py:1188-1191` (`reconcile_terminal` → `verify_reconciliation_lineage`)
and `gpu_ledger.py:448-449`
(`provisional = load_json(cfg["paths"]["provisional_gpu_usage"])`, a file inside
`seal/` that only the producer's final atomic seal publication creates), ordered
by `scripts/wrappers/c04_a0t_small_v1_v7_reconcile.sh:66-83`.

**Mechanism.** Round-1 H-A asked for the terminal accounting stage to be made
independent of the seal. The repair delivered that for the *campaign* accumulator
by adding a separate `campaign-record` mode — which is correct and is why H-A is
closed — but left `reconcile-terminal` itself keyed on the seal. On any terminal
path that produces no seal, `verify_reconciliation_lineage` dies with an
uncaught `FileNotFoundError` on `seal/provisional_gpu_usage.json`. The wrapper's
recovery branch is guarded by `jq -e '.state == "SACCT_TERMINAL_RECONCILED"'`,
which is false (the ledger is still `EXIT_RECORDED_PENDING_SACCT`), so the
wrapper takes `exit "$C04_RECONCILE_STATUS"`.

**Failure scenario.** Every non-sealing terminal outcome:

- **exit 40 budget breach** — the path I-3 exists to create, and which the
  wrapper documents as "a terminal state" with "an accounting-only breach
  record";
- **watchdog TERM/KILL (124/137/143)**;
- **OOM, decode failure, or any producer HALT after `claim()`**.

In each, the campaign row is written correctly, and then the per-namespace GPU
ledger is left at `EXIT_RECORDED_PENDING_SACCT` holding a 7200 s reservation that
is never replaced by real sacct seconds, `resource_final_state.json` is never
published, and the mandated `CPU_POST_JOB_RECONCILIATION` job exits non-zero.
Because the campaign accumulator is now independent and correct, the cross-version
damage is nil — which is why this is Important and not High — but the design
intent of exit 40 ("a clean, accounting-complete terminal state, distinct from an
engineering failure") is not delivered end to end, and the namespace's resource
story is left permanently unfinalized on precisely the path the ceiling exists to
produce.

**What would close it.** Split `verify_reconciliation_lineage` into the parts
that need only the claim/marker/ledger (job identity, lineage chain,
reconciliation authorization) and the parts that need the seal, and make the seal
half conditional on `seal/seal_manifest.json` existing — exactly the
"conditional refinements" phrasing round 1 used. `reconcile_terminal` should then
reconcile the ledger from sacct and publish a `resource_final_state` on every
terminal path, carrying an explicit `seal_published: false` (and, where present,
`budget_breach_sha256`) so the absence of science is recorded rather than
inferred from a missing file.

## IMPORTANT I-3 — the preflight round-trip is narrowed a second time but two channels remain open, one of them with post-GPU blast radius

This subsumes round-1 I-C3, round-2 H-2's residual and round-2 I-1's wider half.
There is **no live defect today** — I measured that — so this is a recurrence
channel, which is why it is Important.

**(a) No full producer record is ever round-tripped before the GPU is spent.**
`downstream_contract_fixtures()` is real work and closes the specific hole that
produced round-1 C-B: it builds a `provisional_gpu_usage` record through the
production builder, and it validates `build_slot_reliability` output against the
*actual* `reliability` definition of the frozen canonical schema in six states.
And the C-B clamp is now genuinely protected — deleting
`max(-1.0, min(1.0, …))` from `common.cosine` turns
`cosine_of_identical_vectors_is_within_the_schema_bound` **red**, because two of
that fixture's three pinned vectors (`[1/3]*3584` and `[0.7,-0.3,0.2]*8`)
overshoot to exactly `1.0000000000000002` unclamped. Measured:

```
FROZEN bytes : 47 fixtures, failing: []
CLAMP DELETED: 47 fixtures, failing: ['cosine_of_identical_vectors_is_within_the_schema_bound']
```

But no fixture yet builds a **full** `canonical_record`, `prompt_record`,
`frame_pack_manifest` or `resource_final_state` and validates it against the
contract its consumer applies. `validate_schema` at preflight is exercised only
against the stage-authorization manifest. The full canonical record is first
schema-validated at `producer.py:1402`, i.e. **only after all 800 forwards are
paid for**; the full `resource_final_state` only at `gpu_ledger.py:1075`, after
the whole allocation. I closed the question of whether that exposure is live by
building both records exactly as the producer builds them: all four canonical
regimes and both prompt-record cases validate. The structural exposure is
unchanged; only its known triggers are closed.

**(b) Three writer/reader key-set contracts are still duplicated literals.**
Round-2 H-2 is closed for the two constants it named — `gpu_ledger.py` now
imports `PROVISIONAL_USAGE_KEYS` and `BUDGET_GUARD_KEYS` and passes them to
`require_exact_keys`, so one-sided drift is impossible by construction there.
The other three, from the same round-2 census, were not converted:

| contract | writer | reader | reader's set | agree today | when drift would surface |
|---|---|---|---|---|---|
| GPU ledger (15) | `preflight.py:414` | `gpu_ledger.py:270` | **literal** | yes | at `claim()` — GPU allocation entered, namespace poisoned |
| resource ticket (16) | `preflight.py:435` | `gpu_ledger.py:710` | **literal** | yes | at `claim()`, before ticket consumption |
| **allocation claim (12)** | `gpu_ledger.py:761` | `gpu_ledger.py:372` | **literal** | yes | **at `CPU_POST_JOB_RECONCILIATION` — after the full 2 GPU-hour spend** |

The third is the round-1 C-A shape verbatim: the writer is in `claim()` and the
reader is in `verify_reconciliation_lineage`, 390 lines apart in the same file,
and a one-sided edit would not be observable until the A100 was gone.

**What would close it.** (a) Add preflight fixtures that build one full
`canonical_record` — in the degenerate regimes: all `missing`, all
`single_valid`, and the agreeing `stable` case with `proposition_cosine` at and
just above 1.0 — one full `prompt_record` including the zero-frame case, and one
`resource_final_state`, and run each through `validate_schema` against the frozen
schema its consumer uses. My scratchpad harness shows this is ~40 lines and
2.4 ms per record. (b) Promote the remaining three key sets to named constants in
`common.py` and import them on both sides, exactly as was just done for the other
two; a fixture that only compares a constant with itself cannot substitute for
this, as round 2 demonstrated.

---

## Non-blocking observations

1. **The pre-model-load containment pass is half self-comparing.** `texts` is
   derived from the same `render_prompt` call the assertion compares against, so
   before model load the pass establishes "no banned token in any of the 400
   transcripts, and the assembled message shape is legal". The message-assembly
   half becomes non-vacuous only at `producer.py:1709`, which does run before
   every forward, so the stated requirement is met. Unchanged from rounds 1-2.
2. **`maps.expected_hashes` is protected only by inclusion in the contract hash.**
   Mutating it moves `config_contract_sha256` (verified), but no code asserts its
   literal value, so the prose invariant in `prompt_hash_contract` is not
   machine-checked. Unchanged from rounds 1-2.
3. **The campaign ledger's `phase` is not cryptographically bound to any
   authorization artifact.** No code advances it (verified), but an advanced
   ledger is indistinguishable from an authorized one. Unchanged from round 2.
4. **`append_campaign_gpu_job` takes no lock on the campaign file.** It is
   protected only by the *per-namespace* `resource/gpu_ledger.lock`, which would
   not exclude a future namespace's reconciler; the head-hash race check turns a
   collision into a halt rather than corruption, but a lock on the campaign path
   itself would close it. Unchanged from round 2.
5. **`campaign_record`'s marker-fallback branch validates nothing about the
   marker.** The claim branch verifies `claim_sha256`; the marker branch reads
   `["slurm_job_id"]` with no `schema_version`, `run_id` or self-hash check,
   unlike `verify_reconciliation_lineage:427-429`. The marker is written only by
   the wrapper and by `create_entry_marker`, and `sacct` must still show a
   terminal one-GPU row, so the risk is low — but the asymmetry is gratuitous.
6. **`BudgetGuard.at_job_start` checks `item_margin` and `seal_reserve`
   individually but not their sum.** `margin=300, seal_reserve=7000` builds a
   guard that can never let the seal phase start. Not live (300 + 600 = 900 ≪
   7080), but the sanity check is one-sided.
7. **The exit-40 wrapper branch runs `jq -e` under `set -e`.** A malformed or
   absent breach record surfaces as exit 1 rather than exit 40, losing the
   distinct code the branch exists to propagate. Fail-closed, but the
   distinctness is lost. Unchanged from round 2.
8. **The seal phase's 600 s reserve is still an estimate, not a measurement.**
   Its effective budget is ~900 s (reserve + item margin), and the CPU half is
   cheap (2.4 ms per canonical-record validation, ≈1 s for 400; ≈6 s for the
   ~2400 post-publication re-validations), so 900 s is plausible — but if it is
   exceeded the outcome is still exit 124 with no seal and no breach record.

---

## Summary table

| # | Severity | Finding |
|---|---|---|
| H-1 | High | `scripts/wrappers/c04_a0t_small_v1_v7.sh:40` `mkdir -p`s the single-shot no-clobber namespace and writes the entry marker before **any** authorization gate — a gate both sibling wrappers carry and this one does not. One out-of-order or unauthorized GPU submission makes the CPU preflight's `no-clobber namespace refusal` permanent and forces a full v8 rebuild. Demonstrated end to end. |
| I-1 | Important | Round-1 I-C2 untouched for a third round: the wrapper `timeout` is anchored to `/proc/uptime` while `sacct` measures from job start, defended by an unmeasured 120 s reserve of which 30 s is `--kill-after`; an over-7200 s terminal elapsed makes `reconcile_terminal` and the `resource_final_state` schema (`maximum: 7200`) permanently unsatisfiable. Blast radius reduced by the H-1 fix and the in-job guard, so no longer High. |
| I-2 | Important | `reconcile-terminal` is still seal-dependent, so exit 40, exit 124/137/143, OOM and any post-claim HALT cannot complete the mandated post-job stage: `resource_final_state.json` is never published and the namespace ledger keeps a 7200 s reservation forever. The campaign accumulator is independent and correct, so cross-version damage is nil. |
| I-3 | Important | The preflight blind spot is narrowed twice but two channels remain: no full `canonical_record`/`prompt_record`/`resource_final_state` is ever round-tripped before the GPU is spent (measured: no live mismatch), and three of five writer/reader key-set contracts are still duplicated literals (measured: all agree today), one of whose readers runs only after the full 2 GPU-hour spend. |

**Verdict: REVISE (0C / 1H / 3I). No execution authority is conferred.
`authorization.preflight_materialization_authorized` must remain `false` and
`review.code_resource_verdict` must remain `PENDING`. H-1 should be closed before
any GPU submission is even possible, because its trigger is a submission made in
exactly the config state the repository is in today, and its cost is the entire
v7 namespace.**

----- END refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND3.md -----


----- BEGIN refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND4.md (sha256 f85aad6ae81a6ff49cd140dc9caf7b147a1fc4e5b2685d0a07115ba6dcbac111) -----
# C04-A0T-SMALL-v1 v7 — Fresh Independent Code/Resource Review, ROUND 4

Reviewer: fresh independent static reviewer (no exposure to the authoring reasoning)
Date: 2026-07-31
Stage reviewed: `CPU_PREFLIGHT` code/resource review of implementation-v7
Execution authority conferred by this review: **none**

---

## Verdict

**REVISE (0 Critical / 1 High / 0 Important)**

`authorization.preflight_materialization_authorized` must remain `false` and
`review.code_resource_verdict` must remain `PENDING`.

Every finding from rounds 1, 2 and 3 is now closed — I re-derived all seventeen
of them rather than reading the repair claims, and eight of the nine round-4
repair sub-claims survive independent re-derivation. The ninth does not.

The single High is a **relocated defect, not a new one**: round-3 I-2 said the
seal-free terminal paths (exit 40, 124/137/143, OOM, post-claim HALT) could not
complete the mandated `CPU_POST_JOB_RECONCILIATION` stage. The round-4 repair
gave `verify_reconciliation_lineage` a seal-free tail and widened
`c04_a0t_small_v1_v7_resource_final_state.schema.json` to accept the
`NO_SEAL_PUBLISHED` sentinel — but did **not** widen
`c04_a0t_small_v1_v7_stage_authorization.schema.json`, whose reconciliation
`payload_binding` still pins `provisional_gpu_usage_sha256` to
`^[0-9a-f]{64}$`. The code therefore demands a reconciliation authorization
manifest that its own frozen schema forbids, and the seal-free path halts at
exactly the same place, for a new reason. I proved both branches of the
alternative unsatisfiable by execution.

This is worth one more cycle for one reason above all: the fix is a two-line
JSON widening that is **free today and impossible after the CPU preflight
runs**, because it moves the schema's SHA-256 → `implementation_hashes` →
`config_contract_sha256`, which is pinned inside the no-clobber preflight
manifest, genesis GPU ledger and resource ticket.

**Apart from that one schema line, the payload is sound enough to proceed.** The
scientific semantics are byte-provably unchanged from v6, both ceilings are
machine-checked and fail-closed, the recurring "irreversible resource before the
rejecting check" family is now closed at every entrypoint I could find, and the
preflight fixture suite is genuinely non-vacuous under twenty independent
mutations.

| # | Severity | Finding |
|---|---|---|
| H-1 | High | `stage_authorization.schema.json` cannot express the `NO_SEAL_PUBLISHED` binding that `verify_resource_reconciliation_authorization` requires on the seal-free path, so round-3 I-2 is relocated rather than closed; unfixable after the preflight materializes. |

---

## Method and reviewer-boundary compliance

- **No SLURM job was submitted, held, released, requeued or cancelled.** The
  only Slurm interaction was read-only `sacct` (two invocations, below).
- **No GPU, teacher, model-weight or frame-decode work.** No `.safetensors` was
  opened; no video was decoded; the only video-adjacent operation was reading
  file names/lengths of ASR JSONL rows.
- **No file under `/data/jehc223/RGCL` was created, modified or deleted** other
  than this review file. Verified by `git status --porcelain` (no tracked C04
  modification) and by re-hashing all seventeen frozen files plus the entire v6
  tree before and after.
- **No dataset label value was materialized.** Every ASR read went through the
  frozen `project_train_asr_line`, which decodes only `id`, `window_text` and
  `language` and syntactically skips `label`. HateMM identifiers were handled as
  contained identifiers: they are never printed, and where an identifier had to
  appear in output I printed a truncated SHA-256 instead.
- **All work in a scratchpad outside the repository**, with
  `PYTHONDONTWRITEBYTECODE=1` on every invocation. Modules were imported from
  byte-identical scratchpad copies (or ROOT-patched copies), never from the
  repository path. `python -m py_compile` was never used. A post-hoc
  `find … -name '*.pyc' -newermt '-3 hours'` over both the repository and the
  scratchpad returned nothing.
- **`artifacts/c04/a0t_small_v1_impl_v7/` does not exist** and was not created;
  confirmed at the end of the session.
- **`artifacts/c04/campaign/gpu_ledger.json` is byte-identical.** Its append path
  *was* exercised — eleven times, at six spend magnitudes — but only against
  copies inside the scratchpad sandbox root.

### Hash verification (all 17, before and after)

Every pinned SHA-256 in the request matched disk on first read and again after
all work. Truncated to 16 hex for legibility; full values were compared:

| File | pinned | measured (start) | measured (end) |
|---|---|---|---|
| `…v7_common.py` | `2e4272c4…` | match | match |
| `…v7_preflight.py` | `ecdc8568…` | match | match |
| `…v7_gpu_ledger.py` | `944023b3…` | match | match |
| `…v7_producer.py` | `7a3c3a79…` | match | match |
| `…v7_preflight.sh` | `914dd5df…` | match | match |
| `…v7.sh` | `645e5011…` | match | match |
| `…v7_reconcile.sh` | `7af04322…` | match | match |
| `…v7_preflight.sbatch` | `919316c7…` | match | match |
| `…v7.sbatch` | `00ddeeed…` | match | match |
| `…v7_reconcile.sbatch` | `d8f634ec…` | match | match |
| `…prompt_record.schema.json` | `541d0245…` | match | match |
| `…canonical_record.schema.json` | `bacbddae…` | match | match |
| `…stage_authorization.schema.json` | `b367eb03…` | match | match |
| `…payload_review.schema.json` | `7edebdfe…` | match | match |
| `…resource_final_state.schema.json` | `e2f9dca5…` | match | match |
| `configs/c04/c04_a0t_small_v1_v7.json` | `3f436ea2…` | match | match |
| `artifacts/c04/campaign/gpu_ledger.json` | `fc6ca12c…` | match | match |

### v6 predecessor unmodified

Concatenated SHA-256 over the 16 v6 source/config/schema files:
`180fc756bc11f8fa…`, identical at start and end. Concatenated SHA-256 over
every file under `artifacts/c04/a0t_small_v1_impl_v6/`:
`bf3f7a38701b1b84…`, identical at start and end (and identical to the value
round 3 recorded). Directory mtimes remain `Jul 31 05:11`.

---

## Round-1 findings — closure status

| # | Round-1 finding | Status | How I re-derived it |
|---|---|---|---|
| C-A | reconciler's exact-key set had not grown with the writer | **CLOSED by construction** | `ast` census of every `require_exact_keys` call in all four modules: the second argument is `set(PROVISIONAL_USAGE_KEYS)` / `set(BUDGET_GUARD_KEYS)` on **both** the writer (`common.py:2073,2090`) and the reader (`gpu_ledger.py:465,471`). Mutating the constant by deleting one member makes the *writer* raise inside the preflight fixture — i.e. the CPU preflight fails first, before any GPU. |
| C-B | `proposition_cosine` could exceed schema `maximum: 1` | **CLOSED, and the fixture is real** | Deleting `max(-1.0, min(1.0, …))` from a scratchpad copy of `common.cosine` turns `cosine_of_identical_vectors_is_within_the_schema_bound` red (measured). The clamped value round-trips through the frozen canonical schema's `reliability` definition and through a full canonical record. |
| H-A | campaign write side reachable only after a seal | **CLOSED** | `campaign-record` is a distinct ledger mode keyed on `resource/allocation_claim.json` with an `allocation_entry_marker.json` fallback, run **first** in the reconcile wrapper under `set -e`, and independent of `seal/`. Re-derived by executing `campaign_record`'s branch selection against a sandbox namespace. |
| H-B | in-job guard had no margin over the wrapper `timeout` | **CLOSED, with more margin than round 3 measured** | Executed `BudgetGuard.at_job_start` over an 18-point grid of claim/producer-start offsets. Guard lead over the wrapper SIGTERM is `300 + c` seconds and never below 300, independent of producer start time. `SLURM_JOB_START_TIME` appears nowhere in the v7 tree except one docstring sentence. |
| I-C1 | accumulator enforced 28800 s, not the binding 7200 s | **CLOSED** | `campaign_effective_cap` measured at `min(7200, 28800) = 7200` on the frozen ledger; a recorded spend of 100 s refuses the next 7200 s reservation; thirteen distinct read-side mutations all halt (table below). |
| I-C2 | ~90 s of margin to the hard 7200 s ceiling; breach unrecoverable | **CLOSED** (see round-3 I-1) | `watchdog_reserve_seconds` 120 → 300 lifts the worst-case margin from ≈85 s to ≈265 s, and an over-cap terminal elapsed is now publishable up to 7800 s with a flag. Both measured. |
| I-C3 | preflight never round-trips a record against a downstream contract | **CLOSED for the record types the GPU stage writes** | Full `prompt_record` (normal + zero-frame) and full `canonical_record` (four reliability regimes) are now built exactly as the producer builds them and validated against the frozen schemas inside the preflight fixture set. Mutating `parse_teacher_response`'s slot shape, `build_slot_reliability`'s key set, or `NUM_FRAMES` turns them red (measured). |

Round-1 non-blocking observations re-checked: the pre-model-load containment
pass now goes through `producer.build_messages` + `teacher_visible_texts`, the
same path as the per-forward call site (re-derived: I ran the real 800
renderings through it); `maps.expected_hashes` is still protected only by
inclusion in the contract hash (mutating it moves `config_contract_sha256` —
measured — but no code asserts its literal value).

---

## Round-2 findings — closure status

| # | Round-2 finding | Status | How I re-derived it |
|---|---|---|---|
| H-1 | accumulator could brick itself on an over-cap write | **CLOSED** | Six append magnitudes (0 / 100 / 7199 / 7200 / 7250 / 30000 s) executed against sandbox copies of the campaign ledger. No append raised; every post-append `load_campaign_gpu_ledger()` succeeded; the over-cap rows carry `aggregate_exceeds_effective_cap: true`; every one of them refuses the next 7200 s reservation; a duplicate append halts with `campaign row already recorded`. |
| H-2 | reader restated the writer's key set; the fixture self-compared | **CLOSED** | `ast` proof above. |
| I-1 | `cosine` unreachable from the preflight; no full-record round-trip | **CLOSED** | Clamp deletion turns a fixture red; both full-record round-trips exist and are non-vacuous. |
| I-2 | post-loop canonicalization + seal phase unguarded | **CLOSED** | `guard.require_remaining(600, "the canonicalization and seal phase")` at `producer.py:1822`, inside the `try` whose `except BudgetDeadlineReached` publishes the accounting-only breach record and returns 40. Measured effective budget for that phase: `900 + c` seconds. |
| I-3 | GPU-seconds burned before `claim()` were recorded nowhere | **CLOSED** | The wrapper writes the entry marker before its first Python call; `campaign_record` falls back to it; with neither artifact present it prints `no allocation entry; nothing to record` and returns 0 rather than fabricating a row. |

---

## Round-3 findings — closure status

| # | Round-3 finding | Status | How I re-derived it |
|---|---|---|---|
| H-1 | GPU wrapper `mkdir`-ed the no-clobber namespace before any authorization gate | **CLOSED — verified end to end** | See below. |
| I-1 | `watchdog_reserve` unmeasured; over-cap terminal elapsed unpublishable | **CLOSED** | See below. |
| I-2 | `reconcile-terminal` seal-dependent | **NOT CLOSED — relocated.** See **Finding H-1**. |
| I-3 | three duplicated writer/reader key sets; no full-record round-trip | **CLOSED** | See below. |

### Round-3 H-1 — closed, and I reproduced the closure

I replayed the frozen GPU wrapper byte-faithfully against a sandbox root with a
stub `python`, in three configurations:

| run | config state | preflight manifest | exit | filesystem effect |
|---|---|---|---|---|
| 1 | the **frozen** config (`gpu_authorized: false`) | absent | 1 (jq gate) | **nothing — `artifacts/` was never created** |
| 2 | fully GPU-authorized | absent | 2 (`HALT_REVIEW_LINEAGE: no frozen preflight manifest`) | **nothing** |
| 3 | fully GPU-authorized | present | proceeds | `resource/` + entry marker + serial lock, then `claim` |

Nothing irreversible precedes the gate. The statements before it are `set -euo
pipefail`, `cd`, `readonly` assignments, one read of `/proc/uptime`, the EXIT
trap arming, and two environment tests. **The EXIT trap is armed earlier than
the gate**, so I checked it specifically: `mark_exit` does
`marker_path = root_path(...)` (which never creates a directory) and returns
immediately when the marker is absent; the only preceding work is
`json.load` of the config. Run 1 confirms this empirically — the trap fired and
no directory appeared. The trap is additionally invoked with `|| true`.

I separately enumerated every other way the v7 namespace could be created before
the preflight: `campaign_record` never `mkdir`s; `reconcile_terminal`'s
`lock_path.parent.mkdir` is preceded by `validate_cpu_reconciliation_environment`
(requires `post_job_reconciliation_authorized: true`) and
`verify_reconciliation_lineage`; the reconcile wrapper's own nine-clause `jq`
gate precedes both; the preflight's `preflight()` tests `namespace.exists()` as
its first statement. All three wrapper gates were executed against the frozen
config and all three refuse it.

### Round-3 I-1 — closed; the arithmetic re-derived

`watchdog_reserve_seconds` is 300 in the config and asserted `== 300` in both the
preflight and the GPU ledger; `120` appears nowhere in the v7 tree. Chain:

- ticket `watchdog_seconds = 7200 − 300 = 6900`; `claim()` re-derives and rejects
  any other value, and rejects `watchdog ≤ minimum_submit_remaining_seconds (300)`.
- `claim()` returns `6900 − c` (`c` = claim duration); the wrapper's `timeout`
  therefore SIGTERMs at `entry + 6900`, SIGKILLs at `entry + 6930`.
- worst-case `sacct ElapsedRaw ≈ P0 + 6930 + mark_exit`, so the margin to the
  7200 s ceiling is ≈ **265 s** (round 3 measured ≈ 85 s). `P0` was measured
  sub-second on job 13840.
- an over-cap terminal elapsed is now **recorded and flagged** rather than
  refused, bounded by `TERMINAL_SECONDS_HARD_MAX = 7800`.

Publication behaviour, measured by driving
`publish_or_verify_resource_final_state` against a sandbox namespace:

| terminal sacct seconds | seal | result |
|---|---|---|
| 6000 | yes | PUBLISHED, `terminal_elapsed_exceeded_cap: false` |
| 6000 | **no** | PUBLISHED, `seal_published: false`, `provisional…: NO_SEAL_PUBLISHED` |
| 7250 | yes | **PUBLISHED**, `terminal_elapsed_exceeded_cap: true` |
| 7250 | no | **PUBLISHED**, both flags set |
| 7800 | yes | PUBLISHED (exact boundary) |
| 7801 | yes | HALT `terminal GPU seconds outside [0,7800]` |

The code constant and the schema maxima agree exactly at 7800 — there is no gap
in which `strict_validate_terminal_ledger` passes and the schema rejects. The
schema diff versus v6 is exactly and only: three maxima 7200 → 7800,
`provisional_gpu_usage_sha256` gains the `NO_SEAL_PUBLISHED` alternative, and two
new required booleans. No scientific field moved. (The remaining 120 diff lines
are pure re-indentation; verified by structural JSON comparison.)

### Round-3 I-3 — closed

`ast` census of all fifteen `require_exact_keys` call sites across the four
modules:

| contract | writer | reader | argument |
|---|---|---|---|
| GPU ledger (15 keys) | `preflight.py:436` | `gpu_ledger.py:306` | `set(GPU_LEDGER_KEYS)` — same object |
| resource ticket (16) | `preflight.py:463` | `gpu_ledger.py:741` | `set(RESOURCE_TICKET_KEYS)` — same object |
| allocation claim (12) | `gpu_ledger.py:785` | `gpu_ledger.py:388` | `set(ALLOCATION_CLAIM_KEYS)` — same object |
| provisional GPU usage | `common.py:2090` | `gpu_ledger.py:465` | `set(PROVISIONAL_USAGE_KEYS)` — same object |
| budget guard | `common.py:2073` | `gpu_ledger.py:471` | `set(BUDGET_GUARD_KEYS)` — same object |

`preflight.py` imports `GPU_LEDGER_KEYS` and `RESOURCE_TICKET_KEYS` from
`gpu_ledger.py`, so both writers are checked at CPU-preflight time, before any
GPU. One key-set literal survives — `{"index","filename","size","sha256"}` at
`producer.py:848` — but its writer is 80 lines away in the same file and the
first frame pack is re-validated immediately after creation, i.e. before the
first forward, so a drift there costs one model load and zero forwards.

The full-record fixtures are real. Twenty independent mutations of production
code, each run against the whole 50-fixture suite:

| mutation | fixtures that turn red |
|---|---|
| delete the `cosine` clamp | `cosine_of_identical_vectors_is_within_the_schema_bound` |
| `render_prompt` → `str.format` | suite **raises** `KeyError: '"source_relation"'` |
| `SELECT_TAG` / `SELECT_SUFFIX` mutated | `selection_known_answer_vector` |
| drop the dataset term from the digest | `selection_known_answer_vector`, `selection_dataset_and_id_sensitivity` |
| reorder the digest concatenation | `selection_known_answer_vector` |
| one whitespace byte changed in prompt A | `prompt_bytes_unchanged_by_the_render_repair` |
| `build_slot_reliability` gains/loses a key | 7 fixtures incl. `full_canonical_record_round_trips_in_every_reliability_regime` |
| `parse_teacher_response` slot dict gains a key | `full_prompt_record_round_trips_against_its_schema` |
| `NUM_FRAMES` 8 → 7 | both full-record fixtures |
| provisional writer gains a field | suite **raises** `provisional usage writer exact-key failure` |
| delete a member of `PROVISIONAL_USAGE_KEYS` | suite **raises** (writer side fires first) |
| drop the HateMM label-bearing prefixes | `teacher_visible_ban_list_covers_both_datasets` |
| accept an unknown content part | `teacher_visible_unknown_part_rejected` |
| accept string frame payloads | `teacher_visible_frame_path_rejected` |
| `TRANSCRIPT_CAP` 2048 → 4096 | `transcript_cap` |

Three mutations produced **no** red fixture and are recorded as observations
below: removing the case-folded haystack from the containment check, and moving
`RELIABLE_CONFIDENCE_MIN` or `PROPOSITION_COSINE_MIN`.

---

## Claim 0 — no scientific semantic changed: **CONFIRMED**

**Selection re-derived from the v7 frozen rule reproduces the v6 frozen
allowlists exactly**, label-blind (only `id`/`window_text`/`language` decoded):

| dataset | train N | ids == v6 allowlist | digests == v6 allowlist | sha256 of the ordered id list |
|---|---|---|---|---|
| HateMM | 744 | **True** | **True** | `091fb1826cbc7f80…` |
| MHC_zh | 579 | **True** | **True** | `6c98c0d75891ce43…` |

**Prompt hashes recomputed from the v7 sources equal the v6 frozen artifact and
the pinned fixture literals:**

```
system   1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048
A        cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b
B        9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314
combined a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a
```
equal to `artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json`: **True**.

**Version-token-normalized tree diff, every residual line accounted for:**

| file | changed lines | accounted for by |
|---|---|---|
| `prompt_record`, `canonical_record`, `stage_authorization`, `payload_review` schemas | **0** | — (so the canonical schema's `maximum: 1` is byte-identical to v6) |
| `resource_final_state.schema.json` | 128 | round-4 I-1/I-2 only; structural JSON comparison shows exactly 3 maxima + 1 `anyOf` + 2 new booleans, the rest re-indentation |
| all three `.sbatch`, `*_preflight.sh` | 0 | — |
| `*_reconcile.sh` | 8 | round-1 H-A only |
| `*_v7.sh` | 62 | round-2 I-3 (18) + round-4 H-1 authorization gate & manifest test (44) |
| `preflight.py` | 42 | round-2/3/4 I-3 |
| `gpu_ledger.py` | 332 | C-A / H-A / H-2 / I-1 / I-2 / I-3 |
| `common.py` | 958 | C-1 / I-1 / I-2 / I-3, pure additions |
| `producer.py` | 497 | C-1 / I-1 / I-3 |
| `config.json` | 140 | version tokens, `v7_scope` prose, `resources` keys, `paths`, refreshed `implementation_hashes` |

`config_contract_sha256` normalization measured independently: filling all four
prompt hashes, flipping any authorization flag, setting all four review pins and
setting all four verdicts do **not** move it (so the v5 impossibility stays
closed); eight other mutations — `watchdog_reserve_seconds`,
`guard_seal_reserve_seconds`, `maps.expected_hashes`, `selection.suffix`,
`reliability.proposition_agreement_cosine_min`, `teacher_contract.num_frames`,
an `implementation_hashes` entry, and
`review.downstream_review_requires_terminal_resource_state` — all **do** move it.

---

## Claim C-1 — the prompt renderer: **CONFIRMED**

1. **The v6 form could never have succeeded.** Both templates embed
   `_SCHEMA_TEXT`, whose literal `{"source_relation":…}` and `{"S":0,…}` braces
   `str.format` reads as replacement fields. Executing
   `PROMPTS[form].format(transcript="x")` raises `KeyError: '"source_relation"'`
   for both forms — pinned by the `prompt_render_regression_str_format_would_raise`
   fixture, which I verified fires (removing the repair makes the whole suite
   raise rather than merely fail).
2. **The substitution is exactly the frozen one.** `render_prompt(form, T)` equals
   `PROMPTS[form][:-len("{transcript}")] + T` for both forms, and the rendered
   text still contains both literal JSON brace groups.
3. **No prompt byte changed** — the four hashes above.
4. **No `.format(transcript=` call site survives**: two textual hits, one in the
   `render_prompt` docstring and one inside the regression fixture. `producer.py`
   has zero.
5. **The guard rails are non-vacuous** for the two cases a caller controls
   (unknown form, non-string transcript — both fixture-covered). The three
   template-shape guards (`count != 1`, non-terminal placeholder, prefix
   preserved) cannot fire on the frozen templates by construction; they are
   edit-detectors, which is their stated purpose.

---

## Claim I-1 — teacher-visible containment: **CONFIRMED**

1. **Runs before the model is loaded.** `assert_teacher_visible_precondition` is
   called at `producer.py:1677`, before `idempotent_complete` (1678) and before
   the `transformers` import and `from_pretrained` (1681-1694). It is repeated
   per item at `producer.py:1709`, inside `one_forward`, before
   `apply_chat_template`.
2. **Strict in both directions.** `teacher_visible_texts` raises on: a message
   list of the wrong length, wrong keys, wrong role, non-list/empty content, a
   part without `type`, a `text` part with extra keys or a non-string body, a
   `video` part with extra keys, a frame count ≠ 8, a frame that is a
   `str`/`bytes`/`bytearray`/`Path`, an unknown content type, and any part census
   other than exactly one video part and exactly two text parts. Four of these
   are fixture-pinned and I confirmed by mutation that removing the
   unknown-type and string-frame guards turns fixtures red.
3. **The ban is wide.** `forbidden_teacher_visible_tokens` bans **both**
   datasets' 400 identifiers in **both** datasets' prompts (so cross-item leakage
   is refused as firmly as self-leakage) plus the two HateMM label-bearing
   prefixes — 402 tokens, exactly `2·200 + 2`. Each token is expanded to
   `{raw, NFKC, NFKC.casefold}` and matched against `{NFKC(text),
   NFKC(text).casefold()}`. The check also refuses to run at all if the item's
   own identifier is not on the ban list.
4. **No false positive on the real 400 transcripts.** I ran all **800**
   renderings (400 items × 2 forms) through `build_messages` +
   `teacher_visible_texts` + `assert_teacher_visible_containment` with the real
   normalized transcripts, label-blind: **0 false positives**, 0.69 s. Positive
   controls, all caught: a self-identifier appended, a cross-dataset identifier
   appended, `HATE_VIDEO_99` (case), and a full-width `ｈａｔｅ_ｖｉｄｅｏ_` (NFKC).
   Shortest banned token is `hate_video_` (11 chars); shortest identifier is 12.
5. **It cannot pass vacuously.** The assertion first requires
   `texts == [SYSTEM_PROMPT, render_prompt(form, transcript)]`; appending
   `"\nVideo id: hate_video_3"` to the rendered text is rejected by that equality
   before the substring scan runs (fixture `teacher_visible_template_tamper_rejected`).

**The HateMM ID-label asymmetry is handled correctly and stated explicitly in the
code.** `common.py:909-926` records that MHC-ZH identifiers are opaque BiliBili
codes while every HateMM identifier is `hate_video_*` or `non_hate_video_*` and
therefore *is* the label, so **the sealed ID-only allowlist provides label
containment for MHC-ZH only, and none at all for HateMM**. HateMM label
containment is supplied instead by this runtime check (which bans both the
identifiers and the two prefixes from every teacher-visible field) plus the
selection rule's label-blindness, which I established independently by
reproducing the allowlists from `id` alone. `producer.py:1658-1662` records the
same asymmetry in the access ledger.

---

## Claim I-2 — the selection self-test is a known-answer vector: **CONFIRMED**

`SELECTION_KNOWN_ANSWER_DIGESTS` are two hard-coded literals over a synthetic
identifier (`c04-known-answer-vector`) belonging to neither dataset, so the
fixture pins the rule without naming a real, label-bearing video id. It is
independent of the module's own code path: mutating the tag, the suffix, the
dataset term, or the concatenation order each turns
`selection_known_answer_vector` red (measured, four separate mutations), and
dropping the dataset term additionally turns `selection_dataset_and_id_sensitivity`
red. The v6 tautology (`selection_digest(x) == selection_digest(x)`) is gone.

---

## Claim I-3 — both ceilings machine-checked and fail-closed

### Tranche ceiling (7200 s) — **CONFIRMED**

One absolute deadline, computed once in `BudgetGuard.at_job_start` and never
recomputed; `remaining_seconds()` only reads it. Measured over an 18-point grid:

| claim `c` | producer start `e` | guard fires at | wrapper SIGTERM | lead | latest seal-phase start | seal budget |
|---|---|---|---|---|---|---|
| 0 | 0 / 30 / 120 | entry+6600 | entry+6900 | **300 s** | entry+6000 | **900 s** |
| 5 | 5 / 35 / 125 | entry+6595 | entry+6900 | **305 s** | entry+5995 | **905 s** |
| 30 | 30 / 60 / 150 | entry+6570 | entry+6900 | **330 s** | entry+5970 | **930 s** |
| 60 | 60 / 90 / 180 | entry+6540 | entry+6900 | **360 s** | entry+5940 | **960 s** |
| 120 | 120 / 150 / 240 | entry+6480 | entry+6900 | **420 s** | entry+5880 | **1020 s** |
| 300 | 300 / 330 / 420 | entry+6300 | entry+6900 | **600 s** | entry+5700 | **1200 s** |

The lead is `300 + c` and is **independent of producer start time**, because the
deadline is anchored to the allocation-entry `/proc/uptime` reading rather than
to `time.monotonic()` at guard construction.

**Where the guard is and is not called.** `deadline_check(guard, "item …")` at
`producer.py:1758` — at the item boundary, before frame decode; `deadline_check`
at `producer.py:1704` — the first statement of `one_forward`, before
`build_messages`; `guard.require_remaining(600, …)` at `producer.py:1822` —
before the canonicalization and seal phase. It is called nowhere inside a decode,
a forward, a write, or the seal's atomic staging. `BudgetDeadlineReached`
subclasses `RuntimeError` and is caught only by the `except` at
`producer.py:1826`; I checked every `except` on the path and none swallows it.

**What a breach leaves on disk:** `publish_budget_breach_record` writes
`resource/budget_breach.json` carrying the lineage, the guard snapshot, the
per-dataset completed count, the teacher-call and frame-pack counters,
`outputs_truncated_or_altered: 0`, `seal_published: false`,
`no_performance_claim: true` and
`no_scientific_verdict_is_published_by_a_budget_breach: true`. It contains no
metric, no teacher output, no reliability rate and no CONTINUE/KILL verdict.
The producer returns **40**; the wrapper has a dedicated exit-40 branch that
re-asserts those three fields with `jq -e` and propagates 40 distinctly from the
124/137/143 branch and from a generic non-zero. A breach record on a zero-exit
run is itself an error (`HALT_INVALID_FREEZE`). All confirmed by reading the
frozen bytes and by the sandbox wrapper replay.

### Campaign ceiling (28800 s, effective 7200 s) — **CONFIRMED**

**Checked before the ticket is consumed.** `assert_campaign_aggregate_headroom`
is called at `gpu_ledger.py:190` inside `validate_gpu_environment`, which is the
**first** statement of `claim()` — before `create_entry_marker`, before
`verify_gpu_lineage`, and ~570 lines before the ticket is read. It is also called
at `preflight.py:167` (before the namespace is materialized) and at
`producer.py:213` (before any model or data work).

**Read side, executed against sandbox copies:**

| mutation | result |
|---|---|
| pristine genesis (0 s spent) | accepted (7200 ≤ 7200) |
| ledger absent | HALT `campaign ledger is absent` |
| `payload_sha256` tampered | HALT `payload mismatch` |
| `aggregate_gpu_seconds` ≠ Σ rows | HALT `payload mismatch` |
| foreign `schema_version` | HALT `foreign campaign ledger schema` |
| foreign `run_id` | HALT `foreign campaign ledger run id` |
| aggregate cap raised to 999999 | HALT `cap is not the amendment cap` |
| aggregate cap lowered to 7200 | HALT `cap is not the amendment cap` |
| phase advanced, phase cap left 7200 | HALT `phase cap does not match the phase` |
| phase cap raised alone | HALT `phase cap does not match the phase` |
| unknown phase | HALT `unknown campaign phase` |
| first tranche carries an advance token | HALT `first-tranche phase carries an advance token` |
| head link wrong | HALT `campaign ledger head link` |
| non-positive reservation | HALT `requested a non-positive reservation` |
| phase advanced **and** cap raised consistently | accepted — see observation 3 |

**No stage can create or reset it.** `preflight.py:162-169` verifies it and
comments that it is deliberately never created there; the only writer is
`append_campaign_gpu_job`, which appends and never truncates. A missing ledger
halts every stage rather than defaulting to zero.

**Write side is idempotent, sacct-derived and cannot brick:**

| appended `gpu_seconds` | append raised | on-disk aggregate | over-cap flag | later `load` | next 7200 s reservation | duplicate append |
|---|---|---|---|---|---|---|
| 0 | none | 0 | `false` | OK | accepted | HALT `already recorded` |
| 100 | none | 100 | `false` | OK | REFUSED | HALT `already recorded` |
| 7199 | none | 7199 | `false` | OK | REFUSED | HALT `already recorded` |
| 7200 | none | 7200 | `false` | OK | REFUSED | HALT `already recorded` |
| **7250** | **none** | **7250** | **`true`** | **OK** | **REFUSED** | HALT `already recorded` |
| 30000 | none | 30000 | `true` | OK | REFUSED | HALT `already recorded` |

Every rejecting check (`head race`, duplicate job id, non-integer seconds) runs
before the write; the two `campaign_effective_cap` evaluations and the flag are
computed before `os.replace`; nothing after the write can raise. Re-recording the
same job verifies `gpu_seconds` and `sacct_state` instead of appending, so the
two calls in one reconcile run (`campaign-record`, then `reconcile_terminal`)
cannot double-count.

**Its opening zero is evidence-backed.** I ran `sacct` myself:

```
13805|c04_a0t_small_v1_v5_preflight|0 |billing=8,cpu=8,mem=64G,node=1|FAILED
13840|c04_a0t_small_v1_v6_preflight|19|billing=8,cpu=8,mem=64G,node=1|COMPLETED
```

exactly matching the ledger's `genesis_evidence.rows`, including
`gres_gpu_present: false` and `gpu_seconds: 0`. A sweep of the entire accounting
record (`sacct -X -S 2020-01-01`) returns **two** C04 rows in total — these two —
and **zero** C04 rows with any `gres/gpu` allocation. The claim
`these_are_the_only_c04_jobs_in_the_accounting_record: true` is true.

**Write side reaches every path that burns GPU-seconds:**

| path | claim | marker | campaign row |
|---|---|---|---|
| exit 40 budget breach | yes | yes | **yes** |
| watchdog TERM/KILL (124/137/143) | yes | yes | **yes** |
| OOM / decode failure / any producer HALT after claim | yes | yes | **yes** |
| fully successful sealed run | yes | yes | **yes** (idempotently twice) |
| HALT before `claim()` publishes | no | **yes** (wrapper writes it) | **yes, via the fallback** |
| allocation never entered | no | no | **no** — prints and returns 0 |

---

## Additional checks

- **`--time`:** absent from all three sbatch files and all three wrappers; each
  sbatch carries an explicit comment that the omission is deliberate.
- **Arrays / dependencies / chained submission / release / resubmission:**
  absent. Zero occurrences of `sbatch`, `scontrol`, `scancel`, `srun`, `salloc`,
  `--array`, `--dependency`, `afterok` or `requeue` anywhere in the v7 set; the
  only textual hits for "release"/"resubmit" are authorization-flag names and
  prose. All three wrappers and all three Python entrypoints reject
  `SLURM_ARRAY_JOB_ID` / `SLURM_JOB_DEPENDENCY`.
- **`--gres`:** exactly one occurrence, `scripts/slurm/c04_a0t_small_v1_v7.sbatch:3`,
  `--gres=gpu:a100:1`. The preflight sbatch requests no GPU. The reconcile sbatch
  requests no GPU and its wrapper additionally rejects a non-empty
  `CUDA_VISIBLE_DEVICES` or `SLURM_GPUS_ON_NODE`, as does
  `validate_cpu_reconciliation_environment`.
- **Resources:** GPU sbatch = **1 GPU / 8 CPU / 64 GB** exactly; preflight = 8
  CPU / 64 GB, no GPU; reconcile = 1 CPU / 4 GB, no GPU.
  `resources.gpu_count/cpus/ram_gb = 1/8/64` and are asserted in the preflight,
  the GPU ledger and the producer.
- **No OCR entrypoint, no network/API client, no dev/test path, no cross-dataset
  path, no label reader:** zero hits for `requests`(module), `urllib`, `httpx`,
  `aiohttp`, `socket`, `boto3`, `tesseract`, `easyocr`, `paddleocr`,
  `pytesseract`; the single textual "requests" is the word inside a frame-decode
  docstring. The only `subprocess` in the whole tree is `gpu_ledger.py:277`,
  `sacct -X -n -P -j <id> -o JobIDRaw,ElapsedRaw,AllocTRES,State` — read-only.
  The only `label` reference on the data path is `_skip_json_value`, which
  advances the parser past the token and increments a skip counter; the projector
  then requires the decoded key set to be exactly `{id, window_text, language}`.
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are exported by the wrapper and
  asserted by the producer; both `from_pretrained` calls pass
  `local_files_only=True`. `root_path` rejects any `dev`/`test`/`validation`-like
  path component, `train_asr_path` rejects symlinks and out-of-root ASR, and
  `video_path` pins each dataset's physical root separately.
- **Authorization flags in the correct pre-review state:** exactly one of
  seventeen is `true` (`implementation_authorized`); all sixteen others are
  `false`, including `preflight_materialization_authorized`. All four review pins
  are `PENDING_*` sentinels and all four verdicts are `PENDING`.
  `maps.expected_hashes` is the documented sentinel string.
- **Unearned pins are rejected.** Against the frozen config: the preflight's
  static gate halts at `HALT_INVALID_FREEZE: preflight authorization is false`;
  `resolve_prompt_hashes(freeze=False)` halts on the sentinel;
  `resolve_prompt_hashes(freeze=True)` with materialization `false` halts;
  only `freeze=True` + materialization `true` is accepted, the single intended
  relaxation. Reverting `watchdog_reserve_seconds` to 120 halts
  (`reserve: 120 != 300`); zeroing `guard_seal_reserve_seconds` halts; setting
  `gpu_authorized: true` at preflight halts; a wrong prompt hash halts.
- **Every stage manifest shape the code can demand, checked against its frozen
  schema:**

  | manifest the code requires | validates? |
  |---|---|
  | `CPU_PREFLIGHT` (`payload_binding: "NO_PREFLIGHT_PAYLOAD_YET"`) | VALID |
  | `GPU_TEACHER_PRELABEL_SEAL` (4-key payload binding) | VALID |
  | `CPU_POST_JOB_RECONCILIATION`, **sealed** run | VALID |
  | `CPU_POST_JOB_RECONCILIATION`, **seal-free** run | **SCHEMA FAILURE** — see H-1 |
  | payload-hash review (incl. the attestation identity) | VALID |

---

# Findings

## HIGH H-1 — the seal-free reconciliation is unsatisfiable: the code demands an authorization manifest that the frozen `stage_authorization` schema forbids, so round-3 I-2 is relocated, not closed

**Where.**
`scripts/analysis/c04_a0t_small_v1_v7_gpu_ledger.py:58` (`NO_SEAL_SENTINEL = "NO_SEAL_PUBLISHED"`),
`:447-460` (the seal-free tail sets `provisional_sha = NO_SEAL_SENTINEL`),
`:540-551` (`_reconciliation_lineage_tail` passes it into
`verify_resource_reconciliation_authorization`),
`scripts/analysis/c04_a0t_small_v1_v7_common.py:735-746` (the manifest must equal
that value exactly) and `:479` (`_verified_review_file` schema-validates the
manifest first), versus
`schemas/c04/c04_a0t_small_v1_v7_stage_authorization.schema.json:159`
(`"provisional_gpu_usage_sha256": {"$ref": "#/definitions/sha256"}`, i.e.
`^[0-9a-f]{64}$`).

**Mechanism.** Round 4 widened the *final-state* schema to accept the sentinel:

```
resource_final_state.schema.json : "anyOf": [ {"$ref": "#/definitions/sha256"},
                                              {"const": "NO_SEAL_PUBLISHED"} ]
stage_authorization.schema.json  : {"$ref": "#/definitions/sha256"}       <-- not widened
```

`payload_binding` in the stage-authorization schema is a `oneOf` of three
variants. The reconciliation binding cannot match variant 1 (a const string) or
variant 2 (`additionalProperties: false` over four other keys), and variant 3
rejects a non-hex `provisional_gpu_usage_sha256`. So a manifest carrying the
sentinel matches **zero** variants.

**Measured.** I built both manifests a reviewer could possibly sign for a
seal-free run and drove the real `verify_reconciliation_lineage` (only the four
upstream verifiers stubbed; the reconciliation-authorization path left intact)
against a sandbox namespace with the claim, the entry marker and no seal:

```
(a) provisional_gpu_usage_sha256 = "NO_SEAL_PUBLISHED"   (what the CODE demands)
    -> RuntimeError: resource reconciliation authorization schema failure:
       ['payload_binding']: … is not valid under any of the given schemas
(b) provisional_gpu_usage_sha256 = <64 hex>              (what the SCHEMA demands)
    -> RuntimeError: HALT_REVIEW_LINEAGE: reconciliation authorization mismatch payload_binding

control: sealed run, 64-hex provisional  ->  passes the manifest schema and the
         binding check, and proceeds to the seal-content checks
```

Both options fail; the intersection is empty. The failure is raised twice over —
first inside `reviewed_pre_reconciliation_ledger_sha256`, again inside
`_verified_review_file` — so it is not routed around.

**Failure scenario.** The GPU tranche runs and terminates on any non-sealing
path — the exit-40 budget breach (the very path the tranche ceiling exists to
create), a watchdog TERM/KILL at 124/137/143, an OOM, a decode failure, or any
producer HALT after `claim()`. `campaign-record` correctly writes the true
sacct spend into the cross-version accumulator, so the amendment's 8 GPU-hour
ceiling stays honest — that half is genuinely fixed. Then `reconcile-terminal`
dies in `verify_reconciliation_lineage` no matter which manifest the reviewer
signed. The reconcile wrapper's recovery branch is guarded by
`jq -e '.state == "SACCT_TERMINAL_RECONCILED"'`, which is false (the ledger is
still `EXIT_RECORDED_PENDING_SACCT` or `CLAIMED_ACTIVE`), so it takes
`exit "$C04_RECONCILE_STATUS"`. The per-namespace ledger keeps its 7200 s
reservation forever, `resource/resource_final_state.json` is never published, and
`review.downstream_review_requires_terminal_resource_state: true` then blocks
every downstream review of that namespace permanently. This is, line for line,
the outcome round-3 I-2 described.

**Why this warrants another cycle rather than an Important.** The repair is two
JSON lines. But it is a change to a file whose SHA-256 is in
`config.implementation_hashes`, hence in `config_contract_sha256` (I measured
that an `implementation_hashes` edit moves the contract hash), hence in the
values the CPU preflight bakes into the preflight manifest, the genesis GPU
ledger and the single-use resource ticket — inside a no-clobber namespace. So
the window in which this is a two-line edit closes the moment the preflight runs;
after that the only remedy is a full v8 namespace rebuild, which is the cost this
campaign has already paid three times. It is also a claim the review request
asserts as delivered ("a terminal resource state is published on every terminal
path"), and it is not: the assertion holds for the code and fails for the frozen
contract the code must satisfy.

**What would close it.** Widen the third `payload_binding` variant of
`schemas/c04/c04_a0t_small_v1_v7_stage_authorization.schema.json` exactly as
`resource_final_state.schema.json` was already widened:

```json
"provisional_gpu_usage_sha256": {
  "anyOf": [ {"$ref": "#/definitions/sha256"}, {"const": "NO_SEAL_PUBLISHED"} ]
}
```

then refresh `config.implementation_hashes` for that file. To make the repair
non-recurring rather than point-fixed, add a preflight fixture that builds a
`CPU_POST_JOB_RECONCILIATION` authorization manifest in **both** regimes — with a
64-hex provisional digest and with `NO_SEAL_PUBLISHED` — and runs each through
`validate_schema` against `schemas.stage_authorization`, with a non-vacuity case
(a third value such as `"MAYBE"`) that must fail. That is the same round-trip
discipline the round-4 I-3 repair applied to `prompt_record` and
`canonical_record`; extending it to the manifests is what would have caught this
before I did. Consider extending it to `resource_final_state` at the same time
(see observation 1).

---

## Non-blocking observations

1. **No fixture round-trips a `resource_final_state` record.** Round 3 asked for
   three record types; two were added. The final state is schema-validated at
   `gpu_ledger.py:1094`, immediately before publication — i.e. after the whole
   allocation. I closed the live question by executing the writer across its full
   range (sealed/seal-free × 0/6000/7250/7800/7801 s): writer and schema agree
   everywhere, with no gap at the 7800 boundary. Structural exposure only.
2. **Removing the case-folded haystack from `assert_teacher_visible_containment`
   turns no fixture red.** The frozen code is correct — I verified a mixed-case
   leak (`HATE_VIDEO_99`) is caught — but the only leaking fixture uses an
   exact-case token, which the raw variant already catches. A mixed-case fixture
   would close the coverage gap.
3. **The campaign ledger's `phase` is not cryptographically bound to any
   authorization artifact.** No code advances it (verified: `CAMPAIGN_PHASE_CAPS`
   is read-only everywhere), and every inconsistent advance halts — but a
   hand-edited ledger with `phase: CONDITIONAL_FULL_BANK` **and**
   `phase_cap_gpu_seconds: 28800` loads and raises the effective ceiling to 28800.
   Unchanged from rounds 2-3.
4. **`append_campaign_gpu_job` takes no lock on the campaign file.** It is
   protected only by the per-namespace `resource/gpu_ledger.lock`, which would not
   exclude a future namespace's reconciler. The head-hash race check turns a
   collision into a halt rather than corruption. Unchanged from rounds 2-3.
5. **`campaign_record`'s marker-fallback branch validates nothing about the
   marker** — it reads `["slurm_job_id"]` with no `schema_version`, `run_id` or
   self-hash check, unlike the claim branch. `sacct` must still show a terminal
   one-GPU row, so the risk is low, but the asymmetry is gratuitous. Unchanged
   from round 3.
6. **`BudgetGuard.at_job_start` checks `item_margin` and `seal_reserve`
   individually but not their sum.** Not live (300 + 600 = 900 ≪ 6900).
   Unchanged from round 3.
7. **The exit-40 wrapper branch runs `jq -e` under `set -e`**, so an absent or
   malformed breach record surfaces as exit 1 rather than 40. The EXIT trap still
   records 40 in the ledger and marker (`C04_FINAL_STATUS` is already set), so the
   loss is cosmetic. Unchanged from rounds 2-3.
8. **`TERMINAL_SECONDS_HARD_MAX = 7800` lives only in `gpu_ledger.py` and,
   duplicated, in three schema maxima.** Every other cap is also in
   `config.resources`. Both files are hash-pinned so drift requires a review, but
   the config no longer fully describes the resource contract.
9. **The reconcile wrapper's closing `jq -e` does not surface
   `terminal_elapsed_exceeded_cap` or `seal_published`.** Both are printed to
   stderr by the Python stages and recorded in the final state, so a downstream
   reader can see them; the wrapper exits 0 either way.
10. **`RELIABLE_CONFIDENCE_MIN`, `PROPOSITION_COSINE_MIN` and the other
    reliability thresholds are never cross-checked against their
    `config.reliability` copies** (unlike `resources`, which is asserted key by
    key). Mutating either constant turns no fixture red. Both the config and
    `common.py` are inside `config_contract_sha256`, so drift requires a review,
    but the duplication is unguarded. Same shape as the `maps.expected_hashes`
    observation carried since round 1.
11. **The CPU preflight now transitively imports `gpu_ledger.py`, and therefore
    `subprocess`,** in order to share `GPU_LEDGER_KEYS` / `RESOURCE_TICKET_KEYS`.
    An `ast` reachability check confirms the preflight uses only those two
    constants and calls no `gpu_ledger` function, so `sacct` remains unreachable
    from it — but the preflight's `slurm_submit_release_resubmit_entrypoint_present:
    false` assertion is now a statement about reachability rather than about the
    import graph.
12. **Any GPU allocation entry, even one that HALTs before `claim()`, forecloses
    the namespace's single GPU opportunity**, because the wrapper writes the entry
    marker before its first Python call and then refuses any later job id. This is
    a deliberate trade — accountability for "every GPU-second" (round-2 I-3)
    against reversibility — and the round-4 gate removes the config/authorization
    state as a possible cause. Recorded so the trade stays visible, not as a
    defect.

---

## Summary

**Verdict: REVISE (0C / 1H / 0I). No execution authority is conferred.**
`authorization.preflight_materialization_authorized` must remain `false`;
`review.code_resource_verdict` must remain `PENDING`.

Everything else in this payload is, as far as four rounds of adversarial
recomputation can establish, sound: the science is byte-identical to v6, the two
ceilings are machine-checked and fail-closed, the accounting reaches every path
that burns a GPU-second, the fixture suite is non-vacuous under twenty
mutations, and the "irreversible resource before the rejecting check" family —
this campaign's signature failure — is closed at every entrypoint I could reach.
Close H-1 and I would expect the next round to be a GO.

----- END refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND4.md -----


----- BEGIN refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND5.md (sha256 3e182e22a6095c8fe3673e913c57514123921ff5881e9950e5e650ebd80587cc) -----
# C04-A0T-SMALL-v1 v7 — Fresh Independent Code/Resource Review, ROUND 5

Reviewer: fresh independent static reviewer (no exposure to the authoring reasoning)
Date: 2026-07-31
Stage reviewed: `CPU_PREFLIGHT` code/resource review of implementation-v7, fifth revision
Predecessors, all four left byte-intact:
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md` (round 1, `REVISE 2C/2H/3I`)
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND2.md` (round 2, `REVISE 0C/2H/3I`)
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND3.md` (round 3, `REVISE 0C/1H/3I`)
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND4.md` (round 4, `REVISE 0C/1H/0I`)

Execution authority conferred by this review: **none**.

Note on the deliverable name: the request names
`C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md`, which is the round-1 file. This
review is written to the round-5 path so all four earlier reviews survive
unaltered.

---

## Verdict

**GO — 0 Critical / 0 High / 0 Important (0C / 0H / 0I)**

The round-4 High is closed, and I closed it by recomputation rather than by
reading the repair claim: I drove the **real** reader
(`verify_resource_reconciliation_authorization`, with the manifest on disk, the
pin computed from its bytes, and `_verified_review_file`'s schema validation
intact) in both seal regimes against a sandbox root, and both now pass. I also
re-narrowed the schema back to its round-4 form in a mutation sandbox and
confirmed the new fixture turns red, so the repair carries a regression guard
rather than being a point fix.

Every finding from rounds 1, 2, 3 and 4 — eighteen in total — holds closed at
the **current** hashes. Three files moved since round 4
(`c04_a0t_small_v1_v7_common.py`, `…_stage_authorization.schema.json`,
`configs/c04/c04_a0t_small_v1_v7.json`); the other fourteen are byte-identical
to what round 4 reviewed. I re-derived the closures anyway rather than inherit
them, and I accounted for every line of the three that moved.

I found no new Critical, High or Important. What remains are fourteen
non-blocking observations, most of them inherited and explicitly carried by
earlier rounds; two are new and both are fixture-coverage gaps over code I
verified behaves correctly as frozen. None of them can cost a GPU allocation,
corrupt or wedge an artifact, or invalidate a scientific claim in the payload as
frozen, so none is rated above an observation.

**The payload is ready.** `authorization.preflight_materialization_authorized`
and `review.code_resource_verdict` are still `false` / `PENDING`, which is the
correct pre-authorization state; this review does not flip them.

---

## Method and reviewer-boundary compliance

- **No SLURM job was submitted, held, released, requeued or cancelled.** The
  only Slurm interaction was read-only `sacct` (three invocations) and one
  read-only `squeue`.
- **No GPU, teacher, model-weight or frame-decode work.** No `.safetensors` or
  any model file was opened, not even metadata; no video was decoded. The only
  video-adjacent operation was reading `id` / `window_text` / `language` out of
  the two train ASR JSONLs.
- **No file under `/data/jehc223/RGCL` was created, modified or deleted** other
  than this review file. Verified by re-hashing all seventeen frozen files
  before and after (17/17 identical), by re-fingerprinting the whole v6 artifact
  tree before and after, and by `find … -newermt` over `scripts/`, `schemas/`,
  `configs/` and `artifacts/c04/`, which returns nothing.
- **No dataset label value was materialized.** Both ASR files were read only
  through the frozen `project_train_asr_line` projector. Measured counters over
  the full files: `label_field_syntactically_skipped` = 744 (HateMM) / 579
  (MHC-ZH), `label_value_materialized` = **0 / 0**. HateMM identifiers were
  hashed, counted and compared, never printed and never reasoned from as labels.
- **All work in a scratchpad outside the repository**
  (`…/scratchpad/review-r5`), with `PYTHONDONTWRITEBYTECODE=1` on every
  invocation. Modules were imported only from scratchpad copies — byte-identical
  ones for read-only work, and for sandbox work copies whose sole edit is `ROOT`
  (verified by reverse-substituting the patch and diffing to zero against the
  frozen file). `python -m py_compile` was never used. `find … -name '*.pyc'
  -newermt` over the repository returns nothing.
- **`artifacts/c04/a0t_small_v1_impl_v7/` does not exist** and was not created.
- **`artifacts/c04/campaign/gpu_ledger.json` is byte-identical**
  (`fc6ca12c…`, start and end). Its append path *was* exercised — six spend
  magnitudes plus duplicate-append attempts — but only against copies inside a
  scratchpad sandbox root.

### Hash verification (all 17, before and after)

Every pinned SHA-256 in the request matched disk on first read and again after
all work. Truncated to 16 hex; full values compared.

| File | pinned / measured (start) | measured (end) |
|---|---|---|
| `…v7_common.py` | `5fc5259ec4a98b47` | match |
| `…v7_preflight.py` | `ecdc8568dfab0a50` | match |
| `…v7_gpu_ledger.py` | `944023b3aafc04df` | match |
| `…v7_producer.py` | `7a3c3a794454c585` | match |
| `…v7_preflight.sh` | `914dd5df80ab45d5` | match |
| `…v7.sh` | `645e501140690cec` | match |
| `…v7_reconcile.sh` | `7af043225285f129` | match |
| `…v7_preflight.sbatch` | `919316c70ae79d9f` | match |
| `…v7.sbatch` | `00ddeeed57d1f585` | match |
| `…v7_reconcile.sbatch` | `d8f634ec88d762be` | match |
| `…prompt_record.schema.json` | `541d02455aee3af9` | match |
| `…canonical_record.schema.json` | `bacbddaeba138068` | match |
| `…stage_authorization.schema.json` | `2edac849da8a3bf4` | match |
| `…payload_review.schema.json` | `7edebdfe81bb5180` | match |
| `…resource_final_state.schema.json` | `e2f9dca545874a4b` | match |
| `configs/c04/c04_a0t_small_v1_v7.json` | `0af5b6bdc12eb641` | match |
| `artifacts/c04/campaign/gpu_ledger.json` | `fc6ca12c32427625` | match |

All 15 `implementation_hashes` and all 15 `frozen_design_hashes` verify against
disk (15/15 and 15/15), and the `implementation_hashes` entry for
`stage_authorization.schema.json` is `2edac849…`, i.e. the round-5 schema edit
was propagated into the config as it had to be. The config is **not** listed
inside its own `implementation_hashes`.

### What moved since round 4

Round 4 recorded `…common.py = 2e4272c4…`, `…stage_authorization.schema.json =
b367eb03…`, `config = 3f436ea2…`. Those three are the only files whose hash
differs today. Every other file under review is byte-identical to the round-4
payload, so round 4's byte-level findings on them transfer — but I re-derived
them independently regardless (below).

### v6 predecessor unmodified

- `artifacts/c04/a0t_small_v1_impl_v6/freeze/preflight_manifest.json` is
  self-consistent (`payload_sha256` reproduces) and **all 14** of its
  `staged_output_hashes` verify byte-for-byte.
- All 15 entries of `configs/c04/c04_a0t_small_v1_v6.json →
  implementation_hashes` verify.
- Concatenated fingerprint over every file under
  `artifacts/c04/a0t_small_v1_impl_v6/`: `bf3f7a38701b1b84e310c2c7950d17e0226f…`,
  identical at start and end of this session and identical to the value round 4
  recorded.

---

## Round-4 H-1 — CLOSED, verified end to end and guarded against recurrence

**The finding.** `stage_authorization.schema.json` pinned the reconciliation
`payload_binding.provisional_gpu_usage_sha256` to `^[0-9a-f]{64}$` while the
code and the final-state schema had moved to a `NO_SEAL_PUBLISHED` sentinel, so
the seal-free reconciliation path was unsatisfiable: no terminal resource state
could be published on a breach, watchdog kill, OOM or post-claim HALT.

**1. The two schemas that describe this field now agree — byte-checked.**
Structural JSON comparison of the v7 stage-authorization schema against its v6
predecessor, version-token-normalized, shows **exactly one** difference in the
entire file:

```
REMOVED /properties/payload_binding/oneOf[2]/properties/provisional_gpu_usage_sha256/$ref
        = "#/definitions/sha256"
ADDED   /properties/payload_binding/oneOf[2]/properties/provisional_gpu_usage_sha256/anyOf
        = [{"$ref": "#/definitions/sha256"}, {"const": "NO_SEAL_PUBLISHED"}]
```

which is character-for-character the same construct already present at
`resource_final_state.schema.json:88-97`. Nothing else in the schema moved.

**2. Both regimes are satisfiable end to end against the real reader.** I built
a sandbox root containing the 15 implementation files, the 15 frozen design
files and the config at their true relative paths, wrote a real reconciliation
authorization manifest to
`refine-logs/C04_A0T_SMALL_V1_V7_RESOURCE_RECONCILIATION_AUTHORIZATION.json`,
set the config pin to that file's SHA-256, and called the **unstubbed**
`verify_resource_reconciliation_authorization` — so the manifest went through
`_verified_review_file` (64-hex pin check → file-hash equality →
`validate_schema` against the frozen stage-authorization schema), then
`verify_closure_hash`, then the full `expected`-dict comparison including
`payload_binding`, then both `verify_bound_file_map` sweeps:

```
(a) sealed regime,   provisional = <64 hex>          -> PASS (pin 460760b0a75a…)
(b) seal-free regime, provisional = NO_SEAL_PUBLISHED -> PASS (pin 37c0cf66d718…)
(c) foreign sentinel "MAYBE"                          -> RuntimeError: schema failure ['payload_binding']
(d) seal-free, reconciliation verdict PENDING         -> RuntimeError: reconciliation verdict is not GO
```

Round 4's measured `(a) schema failure / (b) binding mismatch` — the empty
intersection — is gone. The `NO_SEAL_SENTINEL` the seal-free tail passes
(`gpu_ledger.py:456`) is now exactly what the schema admits.

**3. The widening did not loosen anything else.** `payload_binding` is a
`oneOf` of three variants; a value matching two of them would now fail. Variant
2 (`additionalProperties: false` over four other keys) and variant 3 (eight
keys) are disjoint, and variant 1 is a const string. I validated every manifest
shape the code can demand:

| manifest the code requires | result |
|---|---|
| `CPU_PREFLIGHT` (`payload_binding: "NO_PREFLIGHT_PAYLOAD_YET"`) | VALID |
| `GPU_TEACHER_PRELABEL_SEAL` (4-key payload binding) | VALID |
| `CPU_POST_JOB_RECONCILIATION`, **sealed** (64-hex) | VALID |
| `CPU_POST_JOB_RECONCILIATION`, **seal-free** (sentinel) | **VALID** (was SCHEMA FAILURE in round 4) |
| reconciliation binding with a foreign `"MAYBE"` | REJECTED |
| reconciliation binding missing a required key | REJECTED |
| reconciliation binding with an extra key | REJECTED |

**4. No third description of the field remains.** A repository-wide grep for
`provisional_gpu_usage_sha256` returns, inside the v7 set, exactly two schema
descriptions (stage-authorization and resource-final-state, now identical
`anyOf`s) and the code sites that carry the value. No other schema constrains
it; `cfg["schemas"]` has only five entries and the seal manifest has none. The
one residual duplication is the string constant itself, defined in both
`common.py:2097` and `gpu_ledger.py:58` rather than imported — recorded as
observation 1, not a finding, because the two agree today and the fixture binds
the common-side spelling to the schema.

**5. The repair is guarded, not point-fixed.** Two new fixtures (#35, #36 of
52) round-trip a full reconciliation manifest through `validate_schema` in both
regimes and require a foreign pin to fail. I re-narrowed the schema back to its
round-4 form in a mutation sandbox:

```
FROZEN                       -> 52 fixtures, failing: []
schema re-narrowed to 64-hex -> 52 fixtures, failing:
                                ['reconciliation_manifest_round_trips_in_both_seal_regimes']
```

The fixture is therefore non-vacuous and would have caught the round-4 defect.
Renaming `common.NO_SEAL_SENTINEL` one-sidedly also turns it red, so the
common-side constant is pinned to the schema's `const`.

**6. The seal-free path now publishes a real terminal state.** I built the
`resource_final_state` record exactly as `publish_or_verify_resource_final_state`
builds it and validated it against the frozen schema across the full range:

| terminal sacct seconds | sealed | seal-free |
|---|---|---|
| 0 / 6000 / 7200 / 7250 / 7800 | VALID | VALID |
| 7801 | REJECTED | REJECTED |

`TERMINAL_SECONDS_HARD_MAX = 7800` and all three schema maxima are 7800, so
there is no gap in which `strict_validate_terminal_ledger` passes and the schema
rejects. `seal_published` and `terminal_elapsed_exceeded_cap` are both in the
schema's 29 required keys.

---

## Rounds 1-3 findings — re-derived at the current hashes

| # | Finding | Status | How I re-derived it this round |
|---|---|---|---|
| R1 C-A | reconciler's exact-key set had outgrown the writer | **CLOSED by construction** | `gpu_ledger.py:27-56` imports `PROVISIONAL_USAGE_KEYS` and `BUDGET_GUARD_KEYS` from `common.py` and passes them to `require_exact_keys` at `:465,:471`; the writer validates against the same objects. Deleting one member of `PROVISIONAL_USAGE_KEYS` makes the whole preflight suite **raise** `provisional usage writer exact-key failure` — i.e. the CPU preflight fails first, before any GPU. |
| R1 C-B | `proposition_cosine` could exceed schema `maximum: 1` | **CLOSED, fixture real** | Deleting `max(-1.0, min(1.0, …))` from `common.cosine` turns `cosine_of_identical_vectors_is_within_the_schema_bound` red (measured). |
| R1 H-A | campaign write side reachable only after a seal | **CLOSED** | `campaign_record` keys on `resource/allocation_claim.json` with an `allocation_entry_marker.json` fallback and is `--mode campaign-record`, run **first** in the reconcile wrapper under `set -e`, independent of `seal/`. |
| R1 H-B | in-job guard had no margin over the wrapper `timeout` | **CLOSED** | Executed the frozen `BudgetGuard` (extracted by `ast`, `/proc/uptime` faked) over an 18-point grid: guard lead over the wrapper SIGTERM is `300 + c` and never below 300, independent of producer start time. Table below. `SLURM_JOB_START_TIME` appears in the v7 set only in one producer docstring and one config prose string. |
| R1 I-C1 | accumulator enforced 28800 s, not the binding 7200 s | **CLOSED** | `campaign_effective_cap` = `min(7200, 28800)` = 7200 on the frozen ledger; reserve 7200 accepted, **7201 refused**; a 100 s recorded spend refuses the next 7200 s reservation. |
| R1 I-C2 | ~90 s margin to the hard ceiling; breach unrecoverable | **CLOSED** | `watchdog_reserve_seconds = 300` (asserted `== 300` in the preflight and the GPU ledger; `120` appears nowhere). Worst-case sacct elapsed ≈ `P0 + 6900 + 30 + mark_exit`, so ≈265 s of headroom; and an over-cap terminal elapsed is now recorded and flagged, publishable to 7800 s. |
| R1 I-C3 | preflight never round-trips a record against a downstream contract | **CLOSED for the record types the GPU stage writes** | Full `prompt_record` (normal + zero-frame) and full `canonical_record` (four reliability regimes) are built as the producer builds them and validated against the frozen schemas; `NUM_FRAMES 8→7` turns both red. |
| R2 H-1 | accumulator could brick itself on an over-cap write | **CLOSED** | Six append magnitudes against sandbox copies: no append raised, every post-append load succeeded, over-cap rows carry `aggregate_exceeds_effective_cap: true`, every one refuses the next 7200 s reservation, duplicate append halts. Table below. |
| R2 H-2 | reader restated the writer's key set | **CLOSED** | Same import proof as R1 C-A; a `require_exact_keys` census across the four modules shows both sides naming the same `frozenset` objects for the provisional-usage, budget-guard, GPU-ledger, resource-ticket and allocation-claim contracts. |
| R2 I-1 | `cosine` unreachable from the preflight | **CLOSED** | `cosine` is now in `common.py` (AST diff confirms it moved out of `producer.py`) and clamp deletion turns a fixture red. |
| R2 I-2 | post-loop canonicalization + seal phase unguarded | **CLOSED** | `guard.require_remaining(600, …)` at `producer.py:1822`, inside the `try` whose `except BudgetDeadlineReached` publishes the accounting-only breach record and returns 40. Measured effective budget for that phase: `900 + c` seconds. |
| R2 I-3 | GPU-seconds burned before `claim()` recorded nowhere | **CLOSED** | The wrapper writes the entry marker before its first Python call; `campaign_record` falls back to it; with neither artifact present it prints `no allocation entry; nothing to record` and returns 0. |
| R3 H-1 | GPU wrapper `mkdir`-ed the no-clobber namespace before any authorization gate | **CLOSED — reproduced end to end** | Byte-faithful wrapper replay, table below. |
| R3 I-1 | `watchdog_reserve` unmeasured; over-cap terminal elapsed unpublishable | **CLOSED** | See R1 I-C2 and the final-state matrix above. |
| R3 I-2 | `reconcile-terminal` seal-dependent | **CLOSED** | Relocated by round 4 into its H-1, which is now closed (section above). |
| R3 I-3 | duplicated writer/reader key sets; no full-record round-trip | **CLOSED** | Both halves re-derived above. |
| R4 I-1 | `watchdog_reserve` 120→300, over-cap terminal recorded | **CLOSED** | Arithmetic re-derived; 7800 code constant and 7800 schema maxima agree exactly. |
| R4 I-2 | terminal resource state on every terminal path | **CLOSED** | The final blocker was R4 H-1, closed above; both seal regimes now publish. |
| R4 I-3 | shared key-set constants + full-record fixtures | **CLOSED** | `ast` census plus the mutation table below. |

### R3 H-1 — the GPU wrapper gate, replayed

I replayed the frozen GPU wrapper against a sandbox root (only `cd` and
`PYTHON_BIN` repointed — verified by reverse-substitution diffing to zero), with
a stub `python`:

| run | config state | preflight manifest | exit | filesystem effect |
|---|---|---|---|---|
| 1 | the **frozen** config (`gpu_authorized: false`) | absent | 1 (jq gate) | **nothing — `artifacts/` never created** |
| 2 | fully GPU-authorized | absent | 2 (`HALT_REVIEW_LINEAGE: no frozen preflight manifest`) | **nothing** |
| 3 | fully GPU-authorized | present | proceeds past the gate | `…/freeze` + `…/resource` created, marker written |

Nothing irreversible precedes the gate: the statements before it are
`set -euo pipefail`, `cd`, `readonly` assignments, one `/proc/uptime` read, the
EXIT trap arming and two environment tests. The EXIT trap is armed earlier, so I
checked it specifically — `mark_exit` computes `root_path(...)` (which never
creates a directory) and **returns immediately when the marker is absent**;
run 1 confirms empirically that the trap fired and nothing appeared. Every
`mkdir` in the four modules is at `common.py:760`, `gpu_ledger.py:134/632/1218`,
`producer.py:921/1116/1942/1963`, `preflight.py:519/533`; none is reachable
before its stage's gate. `preflight.preflight()` tests `namespace.exists()` as
its first statement, and `claim()`'s first statement is `validate_gpu_environment`
(authorization flags + campaign headroom) **before** `create_entry_marker`.

### R1 H-B / R3 I-1 — guard arithmetic, executed

`cap=7200, reserve=300, ticket watchdog=6900, item margin=300, seal reserve=600`.

| claim `c` | producer start `e` | guard fires @entry+ | wrapper SIGTERM @entry+ | lead | latest seal start | seal budget |
|---|---|---|---|---|---|---|
| 0 | 0 / 30 / 120 | 6600 | 6900 | **300** | entry+6000 | **900** |
| 5 | 5 / 35 / 125 | 6595 | 6900 | **305** | entry+5995 | **905** |
| 30 | 30 / 60 / 150 | 6570 | 6900 | **330** | entry+5970 | **930** |
| 60 | 60 / 90 / 180 | 6540 | 6900 | **360** | entry+5940 | **960** |
| 120 | 120 / 150 / 240 | 6480 | 6900 | **420** | entry+5880 | **1020** |
| 300 | 300 / 330 / 420 | 6300 | 6900 | **600** | entry+5700 | **1200** |

The lead is `300 + c`, independent of producer start time, because the deadline
is anchored to the allocation-entry `/proc/uptime` reading. Every degenerate
case halts: entry in the future, margin 0 or ≥ watchdog, seal reserve 0 or
≥ watchdog, no budget remaining. `claim()` independently rejects a ticket whose
`watchdog != remaining − reserve`, and `verify_claimed_resource` rejects a
`C04_WATCHDOG_SECONDS` larger than the ticket's.

### R2 H-1 — campaign write side, executed on sandbox copies

| appended `gpu_seconds` | append raised | on-disk aggregate | over-cap flag | later `load` | next 7200 s reservation | duplicate append |
|---|---|---|---|---|---|---|
| 0 | none | 0 | `false` | OK | accepted | HALT |
| 100 | none | 100 | `false` | OK | REFUSED | HALT |
| 7199 | none | 7199 | `false` | OK | REFUSED | HALT |
| 7200 | none | 7200 | `false` | OK | REFUSED | HALT |
| **7250** | **none** | **7250** | **`true`** | **OK** | **REFUSED** | HALT |
| 30000 | none | 30000 | `true` | OK | REFUSED | HALT |

Every rejecting check (head race, duplicate job id, non-integer seconds) runs
before the write; the over-cap flag and both `campaign_effective_cap`
evaluations are computed before `os.replace`; nothing after the write can raise;
`load_campaign_gpu_ledger` has no cap check at all. `record_campaign_gpu_spend`
verifies an already-present row instead of appending, so `campaign-record` and
`reconcile-terminal` in one reconcile run cannot double-count.

---

## Claim 0 — no scientific semantic changed: **CONFIRMED**

**Selection re-derived from the v7 frozen rule reproduces the v6 frozen
allowlists exactly**, label-blind:

| dataset | train N | ids == v6 allowlist | digests == v6 allowlist | ranks 0..199 | transcript sha256 + scalar count == v6 source manifest (200/200) | sha256 of ordered id list |
|---|---|---|---|---|---|---|
| HateMM | 744 | **True** | **True** | True | **True** | `091fb1826cbc7f80…` |
| MHC_zh | 579 | **True** | **True** | True | **True** | `6c98c0d75891ce43…` |

Identical to the values rounds 3 and 4 measured. The transcript check
independently pins normalization, cap, head/tail split and separator as
unchanged.

**Prompt hashes recomputed from the v7 sources equal the v6 frozen artifact:**

```
system   1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048
A        cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b
B        9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314
combined a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a
```
equal to `artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json`: **True**.

**Version-token-normalized tree diff, every residual line accounted for:**

| file | changed lines | accounted for by |
|---|---|---|
| `prompt_record`, `canonical_record`, `payload_review` schemas | **0** | — (so the canonical schema's `maximum: 1` is byte-identical to v6) |
| `stage_authorization.schema.json` | **7** | the round-5 repair **only** — structural JSON comparison shows exactly one `$ref` → `anyOf` substitution and nothing else |
| `resource_final_state.schema.json` | 128 | structural comparison: 3 maxima 7200→7800, 1 `anyOf`, 2 new booleans, `required` 27→29; the rest re-indentation |
| all three `.sbatch`, `*_preflight.sh` | 0 | — |
| `*_reconcile.sh` | 6 | R1 H-A only (`campaign-record` before `reconcile-terminal`) |
| `*_v7.sh` | 60 | R2 I-3 + R3 H-1 (the authorization gate and manifest test) |
| `preflight.py` | 42 | R2/R3/R4 I-3 |
| `gpu_ledger.py` | 320 | C-A / H-A / H-2 / I-1 / I-2 / I-3 |
| `common.py` | 952 | C-1 / I-1 / I-2 / I-3 + the round-5 fixtures, pure additions |
| `producer.py` | 473 | C-1 / I-1 / I-3 |
| `config.json` | 141 | structural comparison below |

**AST function-level diff makes this exact.** Nothing scientific moved:

- `common.py`: **20 added definitions, 0 removed, exactly one changed
  (`self_test_fixtures`)**; 20 added module constants, **0 removed, 0 changed**.
  So `SELECT_TAG`, `SELECT_SUFFIX`, `SELECT_N`, `NUM_FRAMES`, `TRANSCRIPT_CAP`
  / `HEAD` / `TAIL` / `SEPARATOR`, `MAX_NEW_TOKENS`, `RELIABLE_CONFIDENCE_MIN`,
  `PROPOSITION_COSINE_MIN`, `ROLE_DIM`, `TEACHER_DIM`, `Q_DIM`, all three map
  tags, `SYSTEM_PROMPT`, `_SCHEMA_TEXT`, `PROMPT_A`, `PROMPT_B` are provably
  untouched, and `render_slot`, `build_slot_reliability`, `materialize_role_map`,
  `dense_rademacher_payload`, `parse_teacher_response`, `q_product`,
  `safe_vector`, `merkle_root`, `normalize_transcript`, `normalize_proposition`,
  `selection_digest` appear in none of the added/removed/changed sets.
- `preflight.py`: 2 changed (`verify_static_config`, `preflight`), 0 added, 0 removed.
- `gpu_ledger.py`: 3 added (`campaign_record`, `record_campaign_gpu_spend`,
  `_reconciliation_lineage_tail`), 9 changed, 0 removed.
- `producer.py`: 10 added (the `BudgetGuard` family, the containment
  precondition, the breach publisher), **1 removed (`cosine`, moved into
  `common.py`)**, 6 changed.

**Config, structural comparison v6 → v7**, every difference accounted for:
`preflight_materialization_authorized true → false` (correct pre-review state),
`code_resource_verdict GO → PENDING` and its pin → sentinel, 15 refreshed
`implementation_hashes`, 2 added `paths` (`budget_breach`,
`campaign_gpu_ledger`), 4 added `resources` keys (campaign aggregate/phase caps,
guard item margin, guard seal reserve), `watchdog_reserve_seconds 120 → 300`,
and the `v7_scope` prose block. Nothing else.

**`config_contract_sha256` normalization, measured independently:** filling all
four prompt hashes, flipping every authorization flag, setting all four review
pins and setting all four verdicts to `GO` all leave it unmoved (the v5
impossibility stays closed); `watchdog_reserve_seconds`,
`guard_seal_reserve_seconds`, `maps.expected_hashes`, `selection.suffix`,
`reliability.proposition_agreement_cosine_min`, `teacher_contract.num_frames`,
an `implementation_hashes` entry, `schemas.stage_authorization` and
`review.downstream_review_requires_terminal_resource_state` all move it.

---

## Claim C-1 — the prompt renderer: **CONFIRMED**

1. **The v6 form could never have succeeded.** Replacing
   `template.replace(TRANSCRIPT_PLACEHOLDER, transcript)` with
   `template.format(transcript=transcript)` makes the whole fixture suite
   **raise** `KeyError: '"source_relation"'` — both templates embed
   `_SCHEMA_TEXT`, whose literal JSON braces `str.format` reads as replacement
   fields.
2. **The substitution is exactly the frozen one.** Fixtures
   `prompt_render_places_transcript_at_the_tail` and
   `prompt_render_keeps_the_literal_json_schema_braces` pass on the frozen bytes,
   and both prompt hashes reproduce.
3. **No prompt byte changed** — the four hashes above.
4. **No `.format(transcript=` call site survives.** Two textual hits: the
   `render_prompt` docstring and the deliberate regression fixture
   `prompt_render_regression_str_format_would_raise`. `producer.py` has zero.
5. **Guard rails non-vacuous for the two cases a caller controls** (unknown
   form, non-string transcript — fixture-covered). The three template-shape
   guards cannot fire on the frozen templates by construction; they are
   edit-detectors, which is their stated purpose.

---

## Claim I-1 — teacher-visible containment: **CONFIRMED**

1. **Runs before the model is loaded.** `assert_teacher_visible_precondition` at
   `producer.py:1677`, before `from_pretrained` (`:1686`, `:1694`); repeated per
   item at `:1709` inside `one_forward`, before `apply_chat_template`.
2. **Strict in both directions.** I fed `teacher_visible_texts` ten malformed
   message structures — wrong message count, unknown role, unknown content type,
   string frame payload, frame count ≠ 8, text part with an extra key, message
   with an extra key, two video parts, non-string text body, empty content list —
   and **all ten raise**. Mutation-wise, accepting an unknown content part or a
   string frame payload turns fixtures red.
3. **The ban is wide.** Measured on the real tranche: **402** tokens = all 200
   HateMM + all 200 MHC-ZH selected identifiers + `hate_video_` and
   `non_hate_video_`, with both datasets' identifiers banned in both datasets'
   prompts, so cross-item leakage is refused as firmly as self-leakage. Each
   token is expanded to `{raw, NFKC, NFKC.casefold}` and matched against
   `{NFKC(text), NFKC(text).casefold()}`. Shortest banned token is 11 characters.
   Dropping `non_hate_video_` turns `teacher_visible_ban_list_covers_both_datasets`
   red.
4. **No false positive on the real 400 transcripts.** All **800** renderings
   (400 items × 2 forms) through the real `build_messages` +
   `teacher_visible_texts` + `assert_teacher_visible_containment`, label-blind:
   **800 accepted, 0 rejected**, 0.70 s. Positive controls, all caught: a
   self-identifier appended, a cross-dataset identifier appended, `HATE_VIDEO_99`
   (case), a full-width `ｈａｔｅ_ｖｉｄｅｏ_` (NFKC), and a post-render template tamper.
5. **It cannot pass vacuously.** An empty ban list and an identifier absent from
   the ban list are both rejected (`identifier missing from ban list`), and
   disabling that precondition turns `teacher_visible_unbanned_identifier_rejected`
   red.

**The HateMM ID-label asymmetry is handled correctly, and the code says so.**
Every HateMM training identifier is `hate_video_*` or `non_hate_video_*`, so the
identifier *is* the binary label; MHC-ZH identifiers are opaque BiliBili codes
carrying no label information. **The sealed ID-only allowlist therefore provides
label containment for MHC-ZH only, and none at all for HateMM.**
`LABEL_BEARING_ID_SUBSTRINGS` encodes exactly this — measured live as
`{'HateMM': ('hate_video_', 'non_hate_video_'), 'MHC_zh': ()}`. HateMM label
containment is supplied instead by (a) this runtime check, which bans both the
identifiers and the two prefixes from every teacher-visible field, and (b) the
label-blindness of the selection rule, which I established independently by
reproducing both allowlists from `id` alone with
`label_value_materialized == 0`.

---

## Claim I-2 — the selection self-test is a known-answer vector: **CONFIRMED**

Both pinned digests recomputed outside the module by hand concatenation:

```
sha256(utf8("C04-A0T-SMALL-v1" + "HateMM" + "c04-known-answer-vector" + "20260729"))
  = 871e0363e1b01a82…   matches SELECTION_KNOWN_ANSWER_DIGESTS["HateMM"]
sha256(utf8("C04-A0T-SMALL-v1" + "MHC_zh" + "c04-known-answer-vector" + "20260729"))
  = 41bfb637f5cbb26b…   matches SELECTION_KNOWN_ANSWER_DIGESTS["MHC_zh"]
```

Mutating `SELECT_TAG` or `SELECT_SUFFIX` each turns `selection_known_answer_vector`
red (measured). The identifier is synthetic and belongs to neither dataset, so
the fixture pins the rule without naming a real, label-bearing video id.

---

## Claim I-3 — both ceilings machine-checked and fail-closed: **CONFIRMED**

### Tranche ceiling (7200 s)

One absolute deadline, computed once in `BudgetGuard.at_job_start` and never
recomputed (`remaining_seconds()` only reads it). Grid table above.

**Where the guard is and is not called.** `deadline_check` at `producer.py:1758`
(item boundary, before frame decode) and `:1704` (first statement of
`one_forward`, before `build_messages`); `guard.require_remaining(600, …)` at
`:1822` before the canonicalization and seal phase. Nowhere inside a decode, a
forward, a write or the seal's atomic staging. It may only ever STOP work before
a unit begins; there is no path that truncates, shortens or alters an output.

**What a breach leaves on disk.** `publish_budget_breach_record` writes
`resource/budget_breach.json` with the lineage, the guard snapshot, per-dataset
completed counts, teacher-call and frame-pack counters,
`outputs_truncated_or_altered: 0`, `seal_published: false`,
`no_performance_claim: true` and
`no_scientific_verdict_is_published_by_a_budget_breach: true` — no metric, no
teacher output, no reliability rate, no CONTINUE/KILL verdict. The producer
returns **40**; the wrapper has a dedicated exit-40 branch that `jq -e`-asserts
those fields and exits 40, distinctly from the 124/137/143 branch and from a
generic non-zero; a breach record on a zero-exit run is itself refused (exit 3).

### Campaign ceiling (28800 s aggregate, 7200 s effective)

**Checked before the ticket is consumed.** `assert_campaign_aggregate_headroom`
is called inside `validate_gpu_environment`, which is the **first** statement of
`claim()` — before `create_entry_marker` (`:627`), before `verify_gpu_lineage`
(`:628`), and ~130 lines before the ticket is read and consumed. It is also
called at `preflight.py:167`, before the namespace is materialized, and in the
producer before any model or data work.

**Read side, executed against sandbox copies:**

| mutation | result |
|---|---|
| pristine genesis, reserve 7200 | accepted |
| pristine genesis, reserve **7201** | HALT `would take the C04 campaign to 7201s against the 7200s effective ceiling` |
| ledger absent | HALT `campaign ledger is absent` |
| `payload_sha256` tampered | HALT `payload mismatch` |
| foreign `schema_version` | HALT `foreign campaign ledger schema` |
| foreign `run_id` | HALT `foreign campaign ledger run id` |
| aggregate cap raised to 999999 | HALT `cap is not the amendment cap` |
| aggregate cap lowered to 7200 | HALT `cap is not the amendment cap` |
| phase advanced, phase cap left 7200 | HALT `phase cap does not match the phase` |
| phase cap raised alone | HALT `phase cap does not match the phase` |
| unknown phase | HALT `unknown campaign phase` |
| first tranche carries an advance token | HALT `first-tranche phase carries an advance token` |
| head link wrong | HALT `campaign ledger head link` |
| aggregate ≠ Σ rows | HALT `aggregate does not equal its rows` |
| row chain break | HALT `chain break` |
| non-positive reservation | HALT `requested a non-positive reservation` |
| phase advanced **and** cap raised consistently | accepted — observation 5 |

**No stage can create or reset it.** `preflight.py:162-169` verifies and
deliberately never creates it; the only writer is `append_campaign_gpu_job`,
which appends and never truncates; the campaign path lies outside `ARTIFACT_ROOT`
and would be rejected by the preflight's staging namespace check.

**Its opening zero is evidence-backed.** I ran `sacct` myself, read-only:

```
13805|c04_a0t_small_v1_v5_preflight|0 |billing=8,cpu=8,mem=64G,node=1|FAILED
13840|c04_a0t_small_v1_v6_preflight|19|billing=8,cpu=8,mem=64G,node=1|COMPLETED
```

matching `genesis_evidence.rows` verbatim including `alloc_tres`, elapsed and
state. A full accounting sweep (`sacct -X -S 2020-01-01`) returns exactly **two**
C04 rows — these two — and **zero** C04 rows carrying any `gres/gpu` allocation,
so `gpu_seconds: 0` for both and
`these_are_the_only_c04_jobs_in_the_accounting_record: true` are both true.
`squeue` shows no queued or running job for this user.

**Write side reaches every path that burns GPU-seconds:**

| path | claim | marker | campaign row |
|---|---|---|---|
| exit 40 budget breach | yes | yes | **yes** |
| watchdog TERM/KILL (124/137/143) | yes | yes | **yes** |
| OOM / decode failure / any producer HALT after claim | yes | yes | **yes** |
| fully successful sealed run | yes | yes | **yes** (idempotently twice) |
| HALT before `claim()` publishes | no | **yes** (wrapper writes it) | **yes, via the fallback** |
| allocation never entered | no | no | **no** — prints and returns 0 |

---

## Additional checks

- **`--time`:** zero occurrences anywhere in the v7 set; each sbatch carries an
  explicit comment that the omission is deliberate.
- **Arrays / dependencies / chained submission / release / resubmission:**
  zero occurrences of `scontrol`, `scancel`, `srun`, `salloc`, `--array`,
  `--dependency`, `afterok`, `requeue`. The three textual `sbatch` hits are
  `implementation_hashes` keys naming the sbatch files. All three wrappers and
  all three Python entrypoints reject `SLURM_ARRAY_JOB_ID` /
  `SLURM_JOB_DEPENDENCY`.
- **`--gres`:** exactly one occurrence, `scripts/slurm/c04_a0t_small_v1_v7.sbatch:3`,
  `--gres=gpu:a100:1`. The preflight sbatch requests **no GPU**. The reconcile
  sbatch requests no GPU and its wrapper additionally rejects a non-empty
  `CUDA_VISIBLE_DEVICES` or `SLURM_GPUS_ON_NODE`, as does
  `validate_cpu_reconciliation_environment`.
- **Resources:** GPU sbatch = **1 GPU / 8 CPU / 64 GB** exactly; preflight = 8
  CPU / 64 GB, no GPU; reconcile = 1 CPU / 4 GB, no GPU.
  `resources.gpu_count/cpus/ram_gb = 1/8/64`, asserted in the preflight, the GPU
  ledger and the producer.
- **No OCR entrypoint, no network or external API client, no dev/test path, no
  cross-dataset path, no label reader:** zero hits for `import requests`,
  `urllib`, `httpx`, `aiohttp`, `import socket`, `boto3`, `openai`,
  `huggingface_hub`, `tesseract`, `easyocr`, `paddleocr`, `pytesseract`. The
  only `subprocess` in the whole tree is `gpu_ledger.py:277`,
  `sacct -X -n -P -j <id> -o JobIDRaw,ElapsedRaw,AllocTRES,State` — read-only.
  The only `label` reference on the data path is `_skip_json_value`, which
  advances the parser past the token and increments a skip counter; the projector
  then requires the decoded key set to be exactly `{id, window_text, language}`
  (measured: 1323 rows, 0 label values materialized). `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1` are exported by the wrapper and asserted by the
  producer; both `from_pretrained` calls pass `local_files_only=True`.
- **Authorization flags in the correct pre-review state:** exactly **one of
  seventeen** is `true` (`implementation_authorized`); all sixteen others are
  `false`, including `preflight_materialization_authorized`. All four review
  pins are `PENDING_*` sentinels; all four verdicts are `PENDING`; all four
  `prompt_hashes` are the freeze sentinel; `maps.expected_hashes` is the
  documented sentinel.
- **Unearned pins are rejected.** Against the frozen config:

  | entrypoint | result |
  |---|---|
  | `verify_historical_code_resource_authorization` | HALT `code/resource authorization SHA-256 is unpinned` |
  | `verify_payload_review` | HALT `payload hash verdict is not GO` |
  | `verify_gpu_execution_authorization` | HALT `GPU execution verdict is not GO` |
  | `verify_historical_gpu_execution_authorization` | HALT `authorization SHA-256 is unpinned` |
  | `verify_resource_reconciliation_authorization` | HALT `reconciliation verdict is not GO` |
  | `resolve_prompt_hashes(freeze=False)` | HALT `prompt hash A is unfrozen…` |
  | `resolve_prompt_hashes(freeze=True)`, materialization `false` | HALT (same) |
  | `resolve_prompt_hashes(freeze=True)`, materialization `true` | accepted — the single intended relaxation |
  | `verify_static_config` on the frozen config | HALT `preflight authorization is false` |
  | …with `watchdog_reserve_seconds` reverted to 120 | HALT `reserve: 120 != 300` |
  | …with `guard_seal_reserve_seconds` = 0 | HALT `guard seal reserve: 0 != 600` |
  | …with `guard_item_margin_seconds` = 120 | HALT `guard item margin: 120 != 300` |
  | …with the phase cap raised to 28800 | HALT `campaign first-tranche phase cap: 28800 != 7200` |
  | …with `gpu_authorized: true` | HALT `preflight authorization.gpu_authorized` |

- **Fixture suite:** 52 checks (round 4: 50; round 3: 47; v6: 25), all pass on
  the frozen bytes, no duplicate name (a duplicate would be silently dropped by
  `dict(self_test_fixtures())`).

### Mutation battery — the suite is non-vacuous

Fifteen independent mutations of production code, each run against the whole
frozen suite from a scratchpad sandbox:

| mutation | fixtures that turn red |
|---|---|
| re-narrow `stage_authorization` `provisional_gpu_usage_sha256` to 64-hex | `reconciliation_manifest_round_trips_in_both_seal_regimes` |
| rename `common.NO_SEAL_SENTINEL` | `reconciliation_manifest_round_trips_in_both_seal_regimes` |
| delete the `cosine` clamp | `cosine_of_identical_vectors_is_within_the_schema_bound` |
| `render_prompt` → `str.format` | suite **raises** `KeyError: '"source_relation"'` |
| `SELECT_TAG` mutated | `selection_known_answer_vector` |
| `SELECT_SUFFIX` mutated | `selection_known_answer_vector` |
| drop the `non_hate_video_` prefix | `teacher_visible_ban_list_covers_both_datasets` |
| `NUM_FRAMES` 8 → 7 | `full_canonical_record_round_trips_in_every_reliability_regime`, `full_prompt_record_round_trips_against_its_schema` |
| `TRANSCRIPT_CAP` 2048 → 4096 | `transcript_cap` |
| delete a `PROVISIONAL_USAGE_KEYS` member | suite **raises** `provisional usage writer exact-key failure` |
| disable the ban-list-membership precondition | `teacher_visible_unbanned_identifier_rejected` |
| accept an unknown content part | `teacher_visible_unknown_part_rejected` |
| accept string frame payloads | `teacher_visible_frame_path_rejected` |
| **`RELIABLE_CONFIDENCE_MIN` 3 → 2** | **none** — observation 3 |
| **`PROPOSITION_COSINE_MIN` 0.80 → 0.70** | **none** — observation 3 |

Two further mutations produced no red fixture and are recorded as observations 2
and 4: removing the case-folded haystack, and disabling the template-equality
check inside `assert_teacher_visible_containment`.

### The "irreversible resource before the rejecting check" family

I enumerated every entrypoint again and found **no remaining instance**:

- **GPU wrapper** — the `jq -e` authorization gate and the frozen-preflight-manifest
  existence test both precede the first `mkdir`; the EXIT trap armed earlier
  cannot create anything (replayed, run 1 above).
- **`claim()`** — `validate_gpu_environment` (config identity, prompt-hash
  binding, all seventeen authorization flags, resource caps, and
  `assert_campaign_aggregate_headroom`) is its first statement, before
  `create_entry_marker`, before the lock, before the ticket.
- **CPU preflight** — `verify_static_config` (which itself calls the campaign
  headroom check) and `verify_code_resource_authorization` run in `main()` before
  `preflight()`; `preflight()` tests `namespace.exists()` first, stages
  everything into a temp directory, and materializes with a single `os.rename`
  as its last statement.
- **Reconcile wrapper** — a nine-clause `jq -e` gate precedes the first Python
  call; `campaign_record` and `reconcile_terminal` both begin with
  `validate_cpu_reconciliation_environment`.
- **`campaign-record` before `reconcile-terminal`** is the one place where an
  irreversible write (the campaign append) precedes a stage that can halt. That
  ordering is deliberate and correct: accounting must never be refused, the
  append is idempotent, over-cap totals are recorded and flagged rather than
  rejected, and no later check can un-record. Round 2's H-1 concern (the append
  bricking the reader) is measurably gone.

---

## Non-blocking observations

Ordered roughly by how much I would want them closed before a v8, not by risk.
None is a finding.

1. **`NO_SEAL_SENTINEL` is defined twice rather than imported** —
   `common.py:2097` and `gpu_ledger.py:58`, both `"NO_SEAL_PUBLISHED"`.
   `gpu_ledger.py`'s 28-name import list from `common` does not include it. The
   two agree today (verified), and the new fixture binds the *common*-side
   spelling to the schema's `const`; but a one-sided edit to the
   *gpu_ledger*-side constant — the one the production seal-free tail actually
   uses — would turn no fixture red and would resurface exactly the round-4
   High, post-GPU. This is the last instance of the duplicated-literal shape the
   round-2 H-2 and round-4 I-3 repairs eliminated everywhere else; a one-line
   import into the existing `from … common import (…)` list closes it. It is
   *not* a finding because the frozen bytes agree and both files are inside
   `config_contract_sha256`, so drift requires a fresh review.
2. **NEW: disabling the template-equality check in
   `assert_teacher_visible_containment` turns no fixture red.** The check
   (`texts == [SYSTEM_PROMPT, render_prompt(form, transcript)]`) is the
   structural half of the containment argument — it is what makes the
   amendment's broader ban (prediction, neighbour, rank, margin, dataset
   statistic, fold role) satisfied by construction rather than by enumeration.
   `teacher_visible_template_tamper_rejected` does not discriminate it, because
   its tamper string appends `hate_video_3`, which the substring scan catches
   independently. A tamper fixture whose injected text contains **no** banned
   token would close the gap. The frozen code is correct — I verified the
   post-render tamper is rejected with `rendered text is not the frozen
   template`.
3. **`RELIABLE_CONFIDENCE_MIN` and `PROPOSITION_COSINE_MIN` are never
   cross-checked against their `config.reliability` copies** (unlike
   `resources`, which is asserted key by key). Mutating either turns no fixture
   red. Both the config and `common.py` are inside `config_contract_sha256`.
   Carried from round 4 (observation 10).
4. **Removing the case-folded haystack turns no fixture red** — the only
   leaking fixture uses an exact-case token. The frozen code is correct: I
   verified `HATE_VIDEO_99` is caught. Carried from round 4 (observation 2).
5. **The campaign ledger's `phase` is not cryptographically bound to any
   authorization artifact.** No code advances it (verified: `CAMPAIGN_PHASE_CAPS`
   is read-only everywhere), and every inconsistent advance halts — but a
   hand-edited ledger with `phase: CONDITIONAL_FULL_BANK` **and**
   `phase_cap_gpu_seconds: 28800` loads and raises the effective ceiling to
   28800. That is the human gate the amendment intends, but it is
   indistinguishable from an authorized one. Carried from rounds 2-4.
6. **`append_campaign_gpu_job` takes no lock on the campaign file.** It is
   protected only by the per-namespace `resource/gpu_ledger.lock`, which would
   not exclude a future namespace's reconciler; the head-hash race check turns a
   collision into a halt rather than corruption. Carried from rounds 2-4.
7. **`campaign_record`'s marker-fallback branch validates nothing about the
   marker** — it reads `["slurm_job_id"]` with no `schema_version`, `run_id` or
   self-hash check, unlike the claim branch. `sacct` must still show a terminal
   one-GPU row. Carried from rounds 3-4.
8. **No fixture round-trips a `resource_final_state` record.** I closed the live
   question by executing the writer's exact shape across sealed/seal-free ×
   {0, 6000, 7200, 7250, 7800, 7801}: writer and schema agree everywhere, with
   no gap at the 7800 boundary. Structural exposure only. Carried from round 4.
9. **NEW (and very small): the window between `claim()` publishing the
   allocation claim (`gpu_ledger.py:788`) and appending the ledger job row
   (`:814`).** A hard kill inside those ~26 in-lock, pure-CPU statements would
   leave a claim and a consumption record with an empty `ledger["jobs"]`, and
   `reconcile_terminal` would then halt on `reconciliation requires one GPU job`
   with no publishable final state. The campaign accounting is unaffected (the
   claim exists, so `campaign-record` works). `mark_exit`'s comment shows the
   window is known and deliberately handled on the exit side. I record it only
   because it is the sole remaining route to an unpublishable final state I
   could find; it needs a SIGKILL landing between two adjacent statements.
10. **`BudgetGuard.at_job_start` checks `item_margin` and `seal_reserve`
    individually but not their sum.** Not live (300 + 600 = 900 ≪ 6900). Carried
    from rounds 3-4.
11. **The exit-40 wrapper branch runs `jq -e` under `set -e`**, so an absent or
    malformed breach record surfaces as exit 1 rather than 40. The EXIT trap
    still records 40 in the ledger and marker, so the loss is cosmetic. Carried
    from rounds 2-4.
12. **`TERMINAL_SECONDS_HARD_MAX = 7800` lives only in `gpu_ledger.py` and,
    duplicated, in three schema maxima** — every other cap is also in
    `config.resources`. Both files are hash-pinned. Carried from round 4.
13. **`maps.expected_hashes` is protected only by inclusion in the contract
    hash.** Mutating it moves `config_contract_sha256` (measured), but no code
    asserts its literal value. Carried from rounds 1-4.
14. **Any GPU allocation entry, even one that HALTs before `claim()`, forecloses
    the namespace's single GPU opportunity**, because the wrapper writes the
    entry marker before its first Python call and then refuses any later job id.
    This is a deliberate trade — accountability for "every GPU-second" against
    reversibility — and the round-3 H-1 gate removes the config/authorization
    state as a possible cause. Recorded so the trade stays visible. Carried from
    round 4.

Round 4's observation 11 also holds: the CPU preflight transitively imports
`gpu_ledger.py` (hence `subprocess`) to share `GPU_LEDGER_KEYS` and
`RESOURCE_TICKET_KEYS`; it calls no `gpu_ledger` function, so `sacct` stays
unreachable from it, but
`slurm_submit_release_resubmit_entrypoint_present: false` is a statement about
reachability rather than about the import graph.

---

## Summary

**Verdict: GO (0C / 0H / 0I). No execution authority is conferred by this
review.**

What this GO means: the code and resource contract of implementation-v7 is
sound, at the seventeen hashes pinned in the round-5 request. The round-4 High
is closed by recomputation and carries a regression fixture; every finding from
rounds 1-4 holds closed at the current hashes; the scientific semantics are
byte-provably unchanged from v6 (both allowlists, all four prompt hashes, all
400 transcript digests and scalar counts reproduce, and the AST diff shows no
scientific definition or constant moved); both ceilings are machine-checked and
fail-closed; the accounting reaches every path that burns a GPU-second; the
"irreversible resource before the rejecting check" family — this campaign's
signature failure — is closed at every entrypoint I could reach; and the
52-fixture preflight suite is non-vacuous under fifteen independent mutations.

What this GO does **not** do: it does not flip
`authorization.preflight_materialization_authorized`, it does not set
`review.code_resource_verdict`, and it authorizes no GPU, no SLURM submission
and no teacher work. Those remain the owning stage's acts, and the payload-hash
review and GPU execution review are still `PENDING` by design.

----- END refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND5.md -----

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 3 (raw)

Verdict: REVISE - 1 Critical / 1 High / 4 Important. All findings applied;
the Critical moved C08 from strike to hold after the reviewer located source
annotation files carrying a separable Title on 891/891 and 897/897 rows.

**Prior-round application audit (my first job).** All fourteen round-1 and all ten round-2 findings are genuinely applied on all four surfaces, with two exceptions charged below (R1 I-9 on `TARGET_LOOP.md`; R2 I-6 on `TARGET_STATE.json` and record §4.1). I re-checked the JSON explicitly: all ten registry `status` strings match their `new_status` counterparts (C14 reconciled to `struck_gate0_2026_07_31` with the diagnostic role in a separate `scope` field); `held_nonisomorphism_gate_unwritten_as_posed` is in all four surfaces; R2 H-3's four JSON ports are all present verbatim; C14's `prior_status_note` now states its provenance; `ordered_backlog` is untouched; reversibility language is present and uniform on all ten entries.

**Round-2 H-2 adjudication — confirmed handled honestly.** `"the LARGEST ORACLE CEILING EVER MEASURED ON THIS OBJECT"` *is* verbatim in `directions_tried.json`'s F98 entry (`dead[65]`), and `findings.jsonl` F98 is narrower. The record adopts the family-scoped phrasing everywhere and records the two-record disagreement in §6. Correct and conservative.

**Independent re-measurement.** Census re-derived from scratch on the six gt files: key-set `['id','label','text']`, no `title` field; ws-only 39/9 and 0x4; ZH tags 243/34, histogram `em` 254 / `/em` 254; 141/243 = 0.5802 vs 39/336 = 0.1161, base 0.3109; val 20/34 vs 8/44; 49/254 train, 50/288 train+val; markup median 0.0000, max 0.862069, 203 rows >10 %, markup-bearing median 0.2604; p90 0.50505 `lower` / 0.50713 `linear` / 0.51546 `higher`; medians 106 / 108.5 / 694.5-interp-696-upper / 369 / 439.5. C01 arm table recomputes cell-for-cell from the stored confusions (all 28 arms, both datasets); rotation spread and 4-of-6 / 2-of-6 counts correct under strict `<`. Arena `head_deployed_acc` 4 dp; C02 `ARENA2.pooled_native_acc` matches on 6/6. Asset claims verify. **No fabricated number found.** All four downgrades are justified — F82's vote/head split, EUM's `"as of this recon"`, BSY's `bank-ADDITION` scoping, C11's verbatim disjunctive claim, and F55's three *encoder-composition* levels all verify verbatim.

## Critical

**C-1 · The C08 strike's premise 1, as restated, is refuted by primary evidence in this repository — and the falsifying evidence is in the same paragraph that states it.**
`GATE0_REOPEN_2026-07-31.md:262-272` (mirrored in `TARGET_STATE.json` `premise_1_corrected`, `TARGET_FINDINGS.md:75`, `TARGET_LOOP.md:1540`) concludes: *"the correct premise is 'no separable title channel without re-deriving source metadata' — a data-collection act declined LOW/~0 in litsweep2."*

The title is already on local disk as a separate field on exactly the two datasets a `>=2`-dataset route needs:

- `/data/jehc223/Multihateclip/English/annotation(new).json` — **891/891 rows carry a non-empty `Title`**; `/data/jehc223/Multihateclip/Chinese/annotation(new).json` — **897/897**. (HateMM: 0/1066, consistent with the record.)
- `scripts/prep_mhc.py:72-85` — the very function the record cites at `:76` — reads `title = (entry.get("Title") or "").strip()` and `transcript` as **separate variables** before concatenating. `scripts/prep_video_dataset.py:126-139` is byte-identical logic for MHC-ZH. Emitting a title-separated gt is a re-run of an existing CPU-only, deterministic script.
- F88 ledger correction (c) — quoted in this same paragraph with an ellipsis exactly where the numbers sit — reports *"medians: title 15 chars, transcript 76, composed 96"*. That measurement is only possible from a separated title.

`LITSWEEP2_INPUT_FIDELITY.md:56`'s `title_present = 0` is true **of the gt-jsonl key schema only**; the record then imports LITSWEEP2 §3.3's inference (*"absent from source … recovering it means re-scraping YouTube metadata"*), which is factually wrong for both MHC datasets. Nothing in `hard_constraints` or `banned_constraints` bans a title channel. So premise 1 — one of the two premises the record says the strike rests on — is false in the direction that matters, and C08's own written unblock (*"exhibit a provenance artifact present on >=2 datasets"*) is arguably already met by the title itself. What remains is premise 2, scoped to the `<em>` **marker** only, plus a premise 3 the record itself demotes to non-significant corroboration.

This is a stretched strike of the same kind round-1's Critical caught at C07.

**Required repair (any one):** (a) re-dispose C08 as `held_*` with the same reversibility language as the other holds and the title half named as the unscreened residual; or (b) retain the strike but restate it as scoped to the **provenance-marker** route only, delete *"without re-deriving source metadata / a data-collection act"*, record that MHC-EN 891/891 and MHC-ZH 897/897 source rows carry a separable `Title` reachable by a CPU re-prep, and name the title-source half as a separate open candidate. Apply on all four surfaces.

## High

**H-1 · `GATE0_REOPEN_2026-07-31.md:63` — the round-2 I-6 repair introduced a transposed number inside the table certified "exact", and was not applied to the other two surfaces.** The record states *"(train-only 43 / 17 / 16)"* in the order `&#39;` / `&quot;` / `&amp;`. Recomputed: **`&#39;` 43, `&quot;` 16, `&amp;` 17** — the last two are swapped. (Train+val 51 / 22 / 18 is correct.) Round 2 asserted the same transposed triple, so it propagated unchecked. Separately, R2 I-6's repair landed only in §2: §4.1 and `TARGET_STATE.json` `premise_2` still present the train+val histogram beside per-split row counts with no label. Also: the `1` MHC-ZH train entity row is regex-convention-dependent — a hex-inclusive `&#?\w+;` gives **2** rows. **Repair:** correct to `43 / 16 / 17`; label the histogram and add the train-only triple on the other surfaces; state the entity regex convention.

## Important

**I-1 · `TARGET_LOOP.md:1540` — round-1's I-9 is unapplied on this surface, and it contradicts the record.** The disposition table records C08's basis as *"three measured premise failures"*; §4.1 demotes premise 3 to non-significant. **Repair:** correct the count.

**I-2 · `GATE0_REOPEN_2026-07-31.md:376-377` and `:584-586` — `banned_constraints[5]`/`[6]` applied as a blanket "MLLM output" ban, unengaged by the counter-precedent this record itself identifies at C12.** `[5]`'s literal text is four words — *"MLLM-scores-as-training-signal"* — and `research-wiki/TARGET_GATE0_ITER6_LITERATURE.md:260` says LBOP's MLLM emits lower/upper policy sets and *"不输出 label、score、memory pair/key 或 rationale"*. `[6]` is *"P1-P5 re-proposals"*. **Repair:** state the attribution at C05 and C07 as construction-dependent under the same EUM-vs-F60 tension recorded at C12.

**I-3 · Wrong pinpoint citation introduced by the round-2 I-5 repair.** *"+3 any dataset: ~1-2 %"* is `LITSWEEP3_DATA_CENTRIC.md:95`, not `:94`; `:69` carries a different *"~2 %"* belonging to §3 (ELR). **Repair:** `:94` → `:95`.

**I-4 · Note M-1 asserts something about §7 that is not true of §7.** M-1 states the three percentile labels are used in §7; §7 gives only `lower` and `linear` — the `higher`/`0.5155` label is absent, and `0.5155` is precisely the recon figure M-1 exists to explain. **Repair:** add it to §7.

## Checked and cleared

The four downgrades (all justified, all quotes verbatim-faithful at their cited scope); the C13 and C14 strikes (measurement-plus-plausibility and registry text respectively, with TVB correctly disowned as a prediction); C06's `gated_on_zero_cost_falsifier` (correct kind of record — C01's evidence is real, its supporting bans genuinely miss C06's object on F80's/F70's own text, and the ro-cache/span-confound design constraints verify); C09's promotion and its legality verdict (`progress.json:25` and `LITSWEEP3:82` verbatim, counter-text carried, three HALT boundaries stated); the strategic finding §6 (every figure traced); the `$0`/zero-touch boundary; and the three-surface agreement on all ten statuses.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 4 (raw)

Verdict: REVISE - 0 Critical / 2 High / 2 Important. All findings applied.
All THIRTY prior-round findings confirmed genuinely applied on all four surfaces.

**Prior-round audit (first job).** I checked all 30 prior findings (R1 14, R2 10, R3 6) against all four surfaces. **All are genuinely applied**, including in the JSON: R1 I-2/I-3/I-10 and the C10 "arguably replaces" withdrawal are all in `gate0_reopen_2026_07_31` verbatim (R2 H-3); `held_nonisomorphism_gate_unwritten_as_posed` is on all four surfaces (R2 I-2); the entity triple reads `43 / 16 / 17` with both regex conventions stated (R3 H-1 — I recomputed: `&#39;` 43, `&quot;` 16, `&amp;` 17 train-only; 51/22/18 train+val; 1 strict / 2 hex-inclusive MHC-ZH rows); `[5]`/`[6]` are construction-dependent at C05 and C07 (R3 I-2); `LITSWEEP3:95` (R3 I-3); `0.5155`/`higher` is in §7 (R3 I-4); `TARGET_LOOP.md` no longer says "three measured premise failures" (R3 I-1); C14's `prior_status` is `ordered_backlog` with a provenance note; all ten registry `status` strings match their `new_status`; reversibility language present and uniform on all ten; `ordered_backlog` untouched.

**Independent verification.** Census re-derived from scratch on the six gt files — key-set `['id','label','text']` on all six, no title field; ws-only 39/9 and 0x4; ZH tags 243/34, `em` 254 / `/em` 254 only; 141/243=0.5802 vs 39/336=0.1161, base 0.3109; val 20/34 vs 8/44; 49/254 train, 50/288 train+val; markup median 0.0000, max 0.862069, 203>10 %, bearing-median 0.2604; p90 0.5051 `lower`/0.5071 `linear`/0.5155 `higher`; medians 106/108.5, 694.5-interp/696-upper, 369, 439.5/443; R1's stress test reproduces (10/140 = 0.0714). C01 arm table recomputes cell-for-cell from the stored confusions, both datasets, spreads 0.8505-0.8692 and 0.8462-0.8974, 4-of-6 / 2-of-6 correct. C02 `ARENA2.pooled_native_acc` matches the six arena `head_deployed_acc` at 4 dp on 6/6. Assets verify. Every load-bearing quote I checked is verbatim at its cited scope. **No fabricated number found.** All five downgrades and both surviving strikes' underlying evidence hold; every hold and the gate name a usable unblock.

## High

**H-1 · A hold is filed under the heading "Strikes CONFIRMED" in the primary record — the only surface that gets the grouping wrong.** §4.1 heads *"Strikes CONFIRMED"* and then carries **three** entries, the first being C08, whose disposition is `held_title_channel_separable_route_unscreened` and whose own text says the strike is withdrawn. §4.2 heads *"Strikes DOWNGRADED to HOLD"* with only four entries, against the record's own headline that **five** strikes were downgraded; a later cross-reference points C08 to "(§4.2)", where it does not appear. The other three surfaces are correct. **Repair:** move the C08 block into §4.2 as its fifth entry and restore §4.1 to C13 + C14 only.

**H-2 · The only quantitative pricing in C08's unblock is an unqualified secondary transcription of an MHC-ZH *test-split*, markup-stripped median, applied to an MHC-EN + MHC-ZH route where the EN value is 3.4x larger.** The unblock reads *"price the Stage-0 oracle — noting that the title's median length is 15 characters against a composed median of 96, so the channel is thin."* That pair traces to `ERRPAT_MHC-ZH_2026-07-26.md:270-271` — *"Medians **on test**: title 15 chars, transcript 76 chars, composed text 96 chars"* — reaching the record second-hand through F88 ledger correction (c). It is MHC-ZH-only, test-split, markup-stripped. Recomputed on train/val ids joined to the source annotation: MHC-ZH title median **27 raw / 13 markup-stripped**; **MHC-EN title median 51 characters** against a transcript median of 322. The EN leg is exactly the half that makes the `>=2`-dataset route the hold is built on, and it is unpriced while the "thin" verdict generalises a number 3.4x smaller. The record applies the tier/split qualifier scrupulously to the two *other* ERRPAT reads it uses (C08 premise 3 and C11) and not to this one. **Repair:** on all three surfaces, qualify the figure and either add the MHC-EN figure or state that the EN half is unpriced. Apply the same qualifier where ERRPAT §5.3's Offensive/Hateful read is used at C07.

## Important

**I-1 · Two quote-attribution errors, in a record whose subject is quote fidelity.** (a) *"C08's own written unblock (\"exhibit a provenance artifact present on >=2 datasets\")"*. That string exists nowhere in the repository except round 3's own review text. C08's registry `dedup_boundary` contains no such clause, and the record's current C08 unblock is a different three-part text — so a phrase from a superseded draft is quoted in the present tense as the candidate's own written requirement. (b) The F113 honesty clause is attributed to *"`findings.jsonl` F114/F113"*. F114 contains no such text; the clause is verbatim in **F113 only** (the JSON's `V-4` gets this right). **Repair:** (a) drop the quotation marks and state the point directly; (b) cite F113.

**I-2 · C13's strike is recorded as resting on "measurement alone", but the operative step is an unmeasured plausibility inference — and the ban-free structural leg the record already owns went unused.** The measurement is real and I reproduce it exactly, but the step from it to a strike is the record's own next sentence — *"a **plausible** regression, not a plausible gain"* — an inference, not a measurement of C13. The record elsewhere refuses precisely this shape as a closure (`LITSWEEP3:80`'s *"a headwind to price, not a coverage of this mechanism"* at C12; F82 demoted to *"a headwind to price, not a screen"* at C07). Meanwhile a stronger, ban-free, purely measured leg is present and unused: the `<em>` marker is `0` on HateMM and `0` on MHC-EN, so C13's phenomenon exists on **one** of three datasets against `unified_pilot_gate.stage_0_reachability`'s *"at least two datasets"* — the identical arithmetic the record uses to close C11's literal-null disjunct. **Repair:** re-base the C13 strike on the substrate arithmetic, keep the hate-rate measurement as the corroborating headwind, and delete "measurement alone"/"self-sufficient" — or label the inference as an inference.

## Not raised, checked and cleared

All five downgrades are justified at their sources' written scope (F82's vote/head split, EUM's *"as of this recon"* + three revival preconditions, BSY's *"bank-ADDITION"* scoping, C11's verbatim disjunctive registry claim, F55's three *encoder-composition* levels, and C08's premise 1 refuted by `annotation(new).json` 891/891 + 897/897 and `prep_mhc.py:72-85` / `prep_video_dataset.py:126-139`, all verified directly). **No over-cautious hold:** C10's ban carries a written revival path so HOLD-with-preconditions is the faithful record where C14's registry ineligibility is not; C12's ban attribution genuinely turns on a construction nobody has named; C05's precondition was attempted and undischargeable, and `unwritten_as_posed` is the correct strength. C14's strike rests on registry text alone with TVB correctly disowned as a *predicted* rank. C06's gate is the right kind of record. C09's promotion, its zero-GPU Stage-0, its F88 numbers, its legality verdict on two written texts with the stale-premise counter-text carried, and its four kill-risk verification notes all verify. The strategic finding traces figure by figure, including the honestly recorded two-record disagreement over F98's superlative. The `$0` / zero-touch / C04-untouched / C02-unmodified boundary is intact.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 5 (raw)

Verdict: REVISE - 0 Critical / 2 High / 2 Important. All findings applied.
33 of 34 prior-round findings confirmed applied; C13 re-based on registry text.

## What I verified independently (all clean)

**Census, re-derived from scratch** on `data/gt/{HateMM,MHC_zh,MHC}/{train,val}.jsonl` with no reference to either document: key-set `['id','label','text']` on all six, no `title` field; ws-only `39/9` + `0x4`; ZH tags `243/34`, histogram `em` 254 / `/em` 254 only; `141/243 = 0.5802` vs `39/336 = 0.1161`, base `0.3109`; val `20/34` vs `8/44`, base `0.3590`; `49/254` train, `50/288` train+val; markup median `0.000000`, max `0.862069`, `203 > 10 %`, bearing-median `0.2604`; p90 `0.505051` (`lower`) / `0.507133` (`linear`) / `0.515464` (`higher`); medians `106 / 108.5 / 694.5-interp-696-upper / 369 / 439.5-443`; MHC-EN entity rows `64/9`, train-only `&#39;` 43 / `&quot;` 16 / `&amp;` 17, ZH `1` strict / `2` hex-inclusive; R1's stress test `10/140 = 0.0714`. **Every figure reproduces exactly.**

**C01 arm table** recomputed cell-for-cell from the stored confusions: `0.8411/0.8505/0.8598/0.8224/0.8692/0.8505` and `0.8590/0.8846/0.8590/0.8333/0.8974/0.8974`, net-fix all match, spreads `0.8505-0.8692` and `0.8462-0.8974`, 4-of-6 / 2-of-6 correct under strict `<`. `configs/c01/c01_a0_v1.json` confirms `orthogonal_rotation_control.same_block_l2: True` with ex-ante frozen angles.

**C02** `gates.ARENA2.pooled_native_acc` matches the six arena `head_deployed_acc` at 4 dp on 6/6. OBS-1 verifies exactly.

**C08 premise-1 refutation**: `Title` non-empty on `891/891` and `897/897` raw rows; `prep_mhc.py:72-85` and `prep_video_dataset.py:126-139` read title/transcript as separate variables; per-dataset medians recomputed over train+val ids — EN title `51` / transcript `322`, ZH `27` raw / `13` stripped / transcript `78`. All exact.

**Quote fidelity**, checked at cited scope and found verbatim across F82, F80, F70, F60, EUM, BSY, TVB, F55, F99, F113 (**F113 only** — F114 does not contain the honesty clause), F114, F75, F78, F88, HEADCOV, LITSWEEP8, GRADEDLBL, ERRPAT-ZH, LITSWEEP2/3/5, NCA, `progress.json`, LBOP, SSR, EDCM, AGGNET. **No fabricated number found.**

**Kind-of-record**: all ten registry `status` strings equal their `new_status`; reversibility string present and uniform on all ten; historical `ordered_backlog` intact. Both strikes are registry-level, not measured kills; the five downgrades are justified at their sources' written scope; **no hold is over-cautious** — each names a usable unblock.

**Prior-round audit**: 33 of the 34 findings (R1 14, R2 10, R3 6, R4 4) are genuinely applied on all four surfaces. One is not — see H-1.

## High

**H-1 · Round 4's H-2 is unapplied on `TARGET_LOOP.md`, the surface its own repair named — and the stale text contradicts the same file twenty lines later.** `TARGET_LOOP.md` still read: *"price the title channel's Stage-0 oracle knowing its median length is 15 characters against a composed median of 96."* That is exactly the figure round 4 charged as an **MHC-ZH, test-split, markup-stripped** median inherited second-hand via F88 from `ERRPAT_MHC-ZH:270-271`, generalised to a route whose EN leg is `3.4x` larger. R4 H-2's repair was explicit: *"on all three surfaces."* The record, `TARGET_STATE.json` and `TARGET_FINDINGS.md` all carry the corrected per-dataset pricing; `TARGET_LOOP.md` did not, and the same file then described the correction the reader had just been given the superseded version of. **Repair:** port the qualified per-dataset text.

**H-2 · The C13 strike's sole ban-free leg states a categorical no-substrate result that was measured for HTML *tags* only — the identical scoping defect round 1 forced out of C08's premise 2, now carrying a surviving strike alone.** The record read: *"The `<em class="keyword">` markup exists on one of three datasets … the phenomenon C13 acts on has no `>=2`-dataset substrate."* C13's registry claim is not scoped to the `<em>` highlight — it is *"Removing sensitivity to native **HTML/title markup**"* — and **MHC-EN carries HTML markup on `64/549` train and `9/80` val rows**, a fact this record's own §2 table certifies as exact. The step from "`<em>` is on one dataset" to "the phenomenon C13 acts on is on one dataset" is therefore an inference, and it was the *only* basis the strike rested on after round 4 correctly demoted the hate-rate leg to a labelled inference. **Repair (any one):** (a) scope the leg explicitly to the `<em>` harvest-highlight and say why MHC-EN's entity rows are not a substrate for *nuisance invariance*; or (b) re-base on C13's own written self-scoping — its claim says *"a **ZH-specific** extraction nuisance"* — and record that the census confirms rather than establishes; or (c) C13 returns to HOLD.

## Important

**I-1 · The record's provenance attestation "the only files opened for measurement were the six gt files" is false after the round-3 and round-4 repairs, on three surfaces.** Rounds 3 and 4 added measurements taken from `/data/jehc223/Multihateclip/{English,Chinese}/annotation(new).json` — the `891/891` and `897/897` Title counts and the per-dataset medians — and §4.4 recomputes every C01 accuracy from the stored confusion matrices in `C01_A0_OUT.json`. Those are measurements, not reads. Nothing improper happened (both are `$0`, non-test and permitted), but in a record whose subject is provenance fidelity this is a false statement about its own conduct. **Repair:** extend the attestation on all three surfaces.

**I-2 · The machine-readable disposition block files the gated candidate inside an array named `held`, contradicting its own tally.** `dispositions.held` contained C05 **and C06** (`gated_on_zero_cost_falsifier`), while `effective_order_post_c04.tally` reads *"six holds (C05, C07, C08, C10, C11, C12), one gate (C06)"*. A machine consumer reading `dispositions.held` gets two entries, neither set matching the tally. Each entry's own `new_status` is correct, so no status is wrong — but the grouping is, and round 4 treated exactly this class of defect as material because a consumer reads the group. **Repair:** add a `gated` array containing C06.

**One note, not counted as a finding.** The record was internally inconsistent about its own review depth — *"three rounds"* in four places against *"four rounds"* in one. Both statements were true and no disposition depended on it; the three-round phrasing was simply stale.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 6 (raw)

Verdict: REVISE - 0 Critical / 1 High / 2 Important. All findings applied.
The High moved C13 from strike to hold, leaving C14 the only surviving strike.

I independently re-derived the census from the six gt label files (key-set `['id','label','text']`, no `title`; ws-only 39/9 + 0x4; ZH tags 243/34 with histogram `em` 254 / `/em` 254 only; 141/243=0.5802 vs 39/336=0.1161, base 0.3109; val 20/34 vs 8/44; 49/254 train, 50/288 train+val; markup median 0.000000, max 0.862069, 203>10 %, bearing-median 0.2604; p90 0.505051 `lower` / 0.507133 `linear` / 0.515464 `higher`; entity rows 64/9, ZH 1 strict / 2 hex; round 1's stress test 10/140=0.0714 reproduces only with the **train-only** keyword set, as stated), recomputed the C01 arm table cell-for-cell from the stored confusions (spreads 0.8505-0.8692 and 0.8462-0.8974; 4-of-6 / 2-of-6 correct), re-measured the C08 title medians (EN 51/322; ZH 27 raw / 13 stripped / 78) and the `Title` counts (891/891, 897/897), re-verified the C02 `ARENA2.pooled_native_acc` / arena `head_deployed_acc` identity at 4 dp on 6/6, and checked every load-bearing quote at its cited scope. **No fabricated number.** All prior findings are applied on all four surfaces except the two charged below. Both surviving strikes are registry-level, the reversibility string is present and uniform on all ten entries, `ordered_backlog` is untouched, all ten registry `status` strings equal their `new_status`, the downgrades are justified at their sources' written scope, and **no hold is over-cautious**.

## High

**H-1 · The C13 strike's basis is stated three incompatible ways, and round 5's H-2 repair is half-applied — the surviving basis is an unmet, proponent-satisfiable precondition, which this record's own C07 Critical says cannot carry a strike.**
The record still read *"the dedup-boundary leg is withdrawn, and the strike rests on the substrate arithmetic above"* — but that arithmetic was withdrawn eighteen lines earlier (*"was measured for tags only and is withdrawn as a standalone leg"*), and the next line says the opposite: *"The strike rests on the registry text above."* Worse, the registry-text leg is itself built on the dedup boundary (*"Its dedup boundary then **forbids** the only thing that could rescue a two-dataset route"*) which the same section declares *"conditions a two-dataset claim; it does not prohibit one"*. Read as the record itself reads it, the surviving basis is: an un-named cross-dataset pairing. That is a precondition a proponent can supply — and the record writes C13's unblock as exactly that proponent task, which is the record's own definition of the **hold** column, and the same shape as C08's *"re-pose C08 around the half that has a substrate"*.

Two further claims do not survive checking: (a) *"That is registry text, not a ban and not an **inference**"* — the step from "the claim names a ZH-specific nuisance and the entry names no pairing" to "cannot meet `stage_0_reachability`'s two-dataset bar" **is** an inference, of exactly the kind round 4's I-2 forced the record to label at this same candidate; (b) *"the same basis as C14's"* — C14 carries `eligible_for_primary_target: false` plus `hard_constraints[4]` (both verified verbatim) and its unblock is *"a user ruling… Nothing else"*, whereas C13 has no eligibility flag and sits in `ordered_backlog` like C05-C12.

**Repair (all of):** delete the stale sentence; reconcile "forbids" with "does not prohibit"; drop *"not an inference"* and *"the same basis as C14's"*; then **either** state the principle that distinguishes C13's un-named pairing from C07's un-attempted delta **or** re-dispose C13 as `held_zh_scoped_no_cross_dataset_pairing_named`. Apply on all four surfaces.

## Important

**I-1 · The provenance attestation is still incomplete after round 5's I-1 repair: §3.7's load-bearing number is measured from a banked artifact the attestation does not name.** The attestation enumerates exactly three sources, but §3.7 states *"`headspace_arena_hatemm_s0_OUT.json` stores `head_deployed_acc: 0.8884`"* and asserts identity **on 6/6 seeds**, which requires reading all six `scripts/analysis/headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json` (I confirmed the values: 0.8884/0.8858/0.8858 and 0.8929/0.8895/0.8946). That is a fourth measurement source, and it carries §3.7's correction. Nothing improper occurred; this is the same defect round 5 charged at Important, repaired only partially. **Repair:** add the six files on all four surfaces.

**I-2 · "The strongest lexical shortcut in the ZH text channel" is an unranked superlative recorded inside the measured layer.** What is measured is that markup-bearing rows hate at 0.5802 vs 0.1161 against a 0.3109 base, and that keyword-bearing rows *without* the markup hate at 0.0714 — i.e. the marker beats the base rate and beats bare keyword presence. **No ranking against any other lexical feature was computed**, so "the strongest" is not established, and this is paper-bound limitations material where the superlative will be read as measured. Same class the record handled carefully at F98. **Repair:** soften to "a strong lexical shortcut — 5x the no-markup rate and 8x the bare-keyword rate", or add the ranking.

## Checked and cleared (not raised)

C14's basis (registry text alone, `eligible_for_primary_target: false` + `hard_constraints[4]` verbatim, TVB correctly disowned as a *predicted* rank); all downgrades at their sources' written scope (F82's vote/head split incl. "HateMM out of scope"; EUM's "as of this recon" + three revival preconditions; BSY's "bank-ADDITION" scoping; C11's verbatim disjunctive registry claim with ERRPAT §5.2 correctly tier/split/pooled-scoped and the `test_rule` tension disclosed; F55's three *encoder-composition* levels; C08's premise 1 refuted by `annotation(new).json` and the prep scripts); C05's `unwritten_as_posed` strength and its four comparator quotes; C06's gate (C01's own decision block independently fails the primary arm against `frozen_rotation_upper_bound 0.86916`, the ro-cache inventory is one lineage per dataset, and the `ow_` prompt/span confound is real); C09's promotion, F88 numbers, zero-GPU Stage-0, legality with the stale-premise counter-text carried, and all four kill-risk verification notes — **F78's *"'$0 banked keys' premise false — head space never persisted"* is verbatim in `directions_tried.json`'s F78 `status`, and F78 contains no random control, so the misattribution correction is right**; the strategic finding figure by figure; the `$0`/zero-touch/C04-untouched/C02-unmodified boundary; `ordered_backlog` intact; all five prior raw reviews appended.

**Three notes, not counted as findings:** (i) the ERRPAT medians cited at `:270-271` are at `:272` — findable, content correct. (ii) `TARGET_FINDINGS.md` said "four rounds" before enumerating five; the total is conveyed. (iii) §3.8/OBS-1 says four gates carry a machine `"pass": true` while singling out `ZERO_CONTRACT` — `ARENA2` also has no root-level `pass` (its three per-seed `pass: true` are the fourth); round 1 verified and endorsed this reading, and C02 is explicitly out of scope.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 7 (raw)

Verdict: REVISE - 0 Critical / 2 High / 3 Important. All findings applied.
Both Highs were round-6 repairs lagging on a surface. No hold is over-cautious.

I re-derived the measured layer from scratch (six gt files, both `annotation(new).json`, `C01_A0_OUT.json` confusions, `C02_A0_OUT.json` gates, the six `headspace_arena_*_OUT.json`, `data/Archive/`, `CLIP_Embedding/`, the ro-cache inventory) and re-checked every load-bearing quote at its cited scope. **Every `[M]` figure in §2 reproduces exactly**, the C01 arm table recomputes cell-for-cell (rotation spreads `0.8505-0.8692` / `0.8462-0.8974`; 4-of-6 and 2-of-6 below the primary arm), the C02 `ARENA2.pooled_native_acc` / arena `head_deployed_acc` identity holds at 4 dp on 6/6, the per-dataset title medians (EN 51/322, ZH 27 raw / 13 stripped / 78) reproduce, and **no fabricated number was found**. The C14 strike is faithful and inside its own text (`eligible_for_primary_target: false` + the dedup boundary + `hard_constraints[4]`, all verbatim; TVB correctly disowned as a *predicted* rank). All ten registry statuses equal their disposition-block `new_status`, reversibility is present and uniform on all ten, `ordered_backlog` is intact, and **no hold is over-cautious** — including C10, where EUM's ban does name the object but supplies its own three revival preconditions, so HOLD-with-preconditions is the faithful record and a strike would over-read a conditional closure as an absolute one.

## High

**H-1 · The record itself still records C13 as STRUCK in its operative sections, contradicting §4.2, the registry and all three other surfaces.** §5 read *"`C13`, `C14` struck"*; *"**Two strikes, six holds**, one gate, one promotion"*; the hold enumeration omitted C13; and *"only **two** survived adversarial verification"*. §4.2 holds C13, §13 says one strike survives, and all three other surfaces record the hold correctly. **§5 is the *effective post-C04 order* — the section a consumer reads to schedule** — so a live, evidence-less strike assertion sat in the operative output. Round 6's repair said "apply on all four surfaces" and was unapplied on the primary one. **Repair:** move C13 into the unblock-conditional list; "One strike, seven holds"; add C13 to the hold enumerations; "two" → "one".

**H-2 · Round 6's I-2 (unranked superlative) is unapplied on `TARGET_FINDINGS.md`.** It still read *"the ZH `<em class="keyword">` markup is **the strongest lexical shortcut** in that text channel"*, with neither the `5x`/`8x` ratios nor the disclosure that no ranking against other lexical features was computed. The record, `TARGET_LOOP.md` and `TARGET_STATE.json` all carry the softened form. This is paper-bound limitations material with an unmeasured superlative inside the measured layer. **Repair:** replace with "a strong lexical shortcut", add the ratios, state that no ranking was computed.

## Important

**I-1 · `TARGET_LOOP.md` mis-counts the downgrades and omits C13's gap from the section that promises it.** It read *"**five** of its seven recommended strikes … are downgraded to HOLD"* — six were. The heading reads *"### The **six** downgrades, and the gap in each"* but the section contained only five entries; C13's gap was written nowhere on that surface except the one-line table row. **Repair:** correct the count and add a C13 paragraph mirroring §4.2.

**I-2 · The C09 counter-text propagates a premise the repository has retracted — the one stale-premise class this record polices everywhere else.** The record carries `LITSWEEP5_COMPLETENESS.md` §4(ii) verbatim to weaken C09's prior. §4(ii)'s first blessed-class death reads *"Trained SELECTOR on train labels = F47's train-supervised source. DEAD: the deployed kNN vote memorizes train (**CLIP LOO 0.998**)"* — and F114 rules that exact premise a CLIP number ("deployed Qwen heads are 0.9406/0.8915/0.8154", remaining train-side headroom 30x-92x larger), with effect "**DOWNGRADED, NOT VACATED**". `LITSWEEP5_COMPLETENESS.md` is **not** among F114's nine corrected records. The record quotes that very retraction elsewhere for the Feldman leg, so it applies F114 inconsistently — and here the error runs *against* the candidate it promotes. **Repair:** note that §4(ii)'s selector leg rests on the retracted premise, that LITSWEEP5 was not among the nine corrected records, and that the counter-text is downgraded-not-vacated (its independent leg — train-disagreement `0/109, 0/102, 0/92` — is untouched).

**I-3 · C10's unblock transcribes EUM's compressed statistic rather than the two measurements behind it, and drops EUM's own grade qualifier.** The record states in its own voice *"(on HateMM it is a median 83 % of the video in a single contiguous block)"*. The primary measures two separate quantities — *"gold hate-span coverage MEDIAN 0.8289, mean 0.7174, **74.2% a single contiguous span**"* — so the compound is EUM's ban_scope compression, not a measurement; and the same entry's `record` field self-labels all of (a)/(b)/(c) as *"recon-grade inline reads of banked TRAIN-split caches, **not gate-grade** frozen-script output"*, a qualifier the record applies scrupulously to its ERRPAT and F88 reads but not here. **Repair:** state the two statistics separately (`n = 298` train hate videos) and attach the recon-grade qualifier.

## Checked and cleared (not raised)

The C14 strike's faithfulness, scope ("performance backlog only", diagnostic role preserved) and unblock; the seven holds' kind-of-record and usable unblocks (C05 `unwritten_as_posed` with its four comparator quotes verbatim; C07's un-attempted delta vs F82's vote-side/head-side split incl. "HateMM out of scope", the dev-split gold cheat and ZH's `+0.0256 = 2 dev items`; C08's premise 1 refuted by `891/891` + `897/897` and `prep_mhc.py:72-85`; C10 against EUM's "as of this recon" and BSY's "bank-ADDITION" scoping; C11's verbatim disjunctive claim with ERRPAT §5.2 tier/split/pooled-scoped, the `test_rule` tension disclosed and `:405` quoted in full; C12's F55 leg and the `[5]`-vs-F60/D7 construction dependence; C13's registry text); C06's gate incl. the one-lineage-per-dataset ro-cache correction and the `ow_` prompt/span confound; C09's promotion, F88 numbers, zero-GPU Stage-0, legality on `progress.json:25` + `LITSWEEP3:82` with the Wall-A parenthetical and `:91`/`:95` pricing, and the four kill-risk notes incl. the F78→F88(3) re-attribution (F78 is a `$0` park containing no random control — confirmed); the strategic finding figure by figure incl. the honestly recorded F98 superlative disagreement; C02's `+0.0009 / -0.0012` and `0.5040 / 0.4881`; and the `$0` / zero-touch / C04-untouched / C02-unmodified boundary, with all six raw reviews appended.


================================================================================
C04-A0T-SMALL-v1 impl-v8 CODE/RESOURCE REVIEW (round 1) -- raw
Appended 2026-08-01. Verdict: GO (0C/0H/0I). Stage authorized: CPU_PREFLIGHT only.
Reviewer: fresh independent Opus reviewer, zero exposure to the build reasoning.
Full file: refine-logs/C04_A0T_SMALL_V1_V8_CODE_RESOURCE_REVIEW.md
================================================================================
# C04-A0T-SMALL-v1 impl-v8 — Independent Code/Resource Review (round 1)

Date: 2026-08-01
Reviewer: fresh independent Opus reviewer, zero exposure to the implementation
reasoning; inputs were the frozen bytes plus
`refine-logs/C04_A0T_SMALL_V1_V8_CODE_RESOURCE_REVIEW_REQUEST.md`.
Stage authorized: `CPU_PREFLIGHT` only.

---

# VERDICT: `GO (0C/0H/0I)`

Authorizes the CPU preflight only. All 15 SHA-256s match the review request byte-for-byte, and the live config's `implementation_hashes` is identical to that same 15-row set; all 15 `frozen_design_hashes` also verify on disk.

No finding at any severity. I hunted the stated failure family specifically and could not construct an instance in the v8 tree.

---

## What I RECOMPUTED

| Quantity | Config claim | My recomputation | |
|---|---|---|---|
| `sacct -X -n -P -j 13852` | FAILED / 1:0 / 1978 / `billing=8,cpu=8,gres/gpu=1,mem=64G,node=1` | `13852\|c04_a0t_small_v1_v7\|FAILED\|1:0\|1978\|billing=8,cpu=8,gres/gpu=1,mem=64G,node=1` | OK |
| v7 prompt-record count | 400 | 400 files, 400 `elapsed_seconds` | OK |
| `hatemm_forward_seconds_sum` | 1769.9 | 1769.9382669688 -> 1769.9 | OK |
| `hatemm_forward_seconds_mean` | 4.4248 | 4.42484566742 -> 4.4248 | OK |
| median / max | 4.457 / 10.15 | 4.456705 / 10.149983 | OK |
| `non_forward_overhead_seconds` | 208.1 (1978 - 1769.9) | 208.0617 -> 208.1 | OK |
| Campaign ledger payload digest | `b19629de...` | recomputed identical | OK |
| Hash chain | GENESIS -> `e8db5f88...` = head | prev-link OK, row digest OK, head OK | OK |
| Aggregate | 1978 | sum rows = 1978 = field | OK |
| Effective cap / headroom | 7200 | min(7200, 28800)=7200; 1978+5222=7200 <= cap; 1978+5223=7201 > cap | OK |
| OOM closed form 43056^2*16*4 B | 110.50 GiB | 118,644,424,704 B = 110.49623 GiB -> 110.50 | OK |
| `project_gpu_window` | window 4022, 800 fwd, 3807.9 s, 5.3 % | 4022 / 800 / **3807.9** / margin **214.1** / **0.0532** | OK |
| secondary capped projection | — | 2588.1 s (36 % margin) | OK |
| max affordable mean/forward | — | 4.6924 s (basis 4.4248, capped-regime measurement 2.9) | OK |
| Max frame-pack load seconds that still fits | — | 274.1 s (builder's basis: 60 s) | — |

Also recomputed: `guard_item_margin` 300, `guard_seal_reserve` 600, `watchdog_reserve` 300 all present in config **and** asserted in all three of preflight/gpu_ledger/producer; `TERMINAL_SECONDS_HARD_MAX` = 5822 (5222+600), coherent — strictly tighter than the schema's 7800 ceiling, so no unpublishable state; wrapper `timeout` ceiling = 4922.

Independently re-derived facts the request did not ask for but that the argument rests on: v7's `elapsed_seconds` timer starts *after* `load_or_create_frame_pack` (`v7_producer.py:1759-1768`), so the 201 packs v7 built really are inside the 208.1 s residual — the over-count claim is true, worth ~111 s.

MHC_zh frame-pack footprint sampled on 8 real selected videos (incl. one 3840x2160): mean 8.48 MB -> ~1.70 GB; HateMM measured from v7 = 0.32 GB. Total ~2.0 GB against ~10 GB of soft-quota headroom (280 G used / 290 G soft / 3000 G hard). Fits.

---

## Negative fixtures I EXECUTED

All under `.../scratchpad/reviewer_r1`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=""`, no GPU, no SLURM, no v8 namespace, no write outside scratch (verified after: `artifacts/c04/a0t_small_v1_impl_v8` still absent, v6/v7 untouched, campaign ledger untouched, all 15 frozen hashes unchanged).

| # | Fixture | Result |
|---|---|---|
| 1 | `self_test_fixtures()` | **65 fixtures, 0 failed** — matches the claim exactly |
| 2 | `run_self_tests(cfg)` on the live config | **74 checks, all pass** (65 + 4 role-map + 4 AST + `no_test_paths`) |
| 3 | Preflight vs producer visual-geometry paths, real processor, 7 geometries (1920x1080, 1080x1920, 640x360, 224x224, **3840x2160**, 333x247, 720x1280) | **grids identical on all 7**; `max_pixels=151200` propagates; max 2880 pre-merge tokens, all < 4096 |
| 4 | Config-contract neutrality of the authority flip (`preflight_materialization_authorized`->true, verdict->GO, pin filled, prompt hashes frozen) | contract hash **unmoved** |
| 5 | Contract tamper probes: `projection_basis.mean`, `teacher_contract.max_pixels`, `resources.small_cap_gpu_seconds` | contract hash **moves** in all three — the basis and the cap are inside the pinned contract, not decoration |
| 6 | Projection gate at 4096 tokens | `geometry_fits=False`, `fits=False` — HALT |
| 7 | Projection gate at 600 s frame-pack load | `projected=4347.9`, `time_fits=False` — HALT |
| 8 | `assert_campaign_aggregate_headroom` vs **live** ledger | 5222 accepted, 5223 refused — non-vacuous |
| 9 | `write_frame_pack` -> `strict_validate_frame_pack` round trip, real PNG bytes | manifests identical; key set = `{schema_version} u FRAME_PACK_BINDING_KEYS u FRAME_PACK_METADATA_KEYS u {frames, payload_sha256}` |
| 10 | Reader vs wrong `teacher_max_pixels` / wrong `code_resource_authorization_sha256` / 1 flipped PNG byte / stray file in pack dir / re-write of an existing pack | **all 5 rejected**, distinct HALT messages; clean pack re-validates after cleanup |
| 11 | Zero-frame path (`total=0` -> `requested_indices=[]`) written, validated, and pushed through the **prompt-record schema** as the real producer would emit it | **accepted** (schema `minItems: 0`) — the fixture's `[0]*8` does not cover this, so I covered it; not empty |
| 12 | `visual_patch_tokens` = 4096 / 4097 vs schema | 4096 accepted, 4097 rejected; `assert_visual_token_ceiling` agrees; preflight gate is strictly tighter (`<`) |
| 13 | `resource_final_state` built as the code builds it, validated in **4 regimes**: sealed, no-seal (`NO_SEAL_PUBLISHED`), at `TERMINAL_SECONDS_HARD_MAX`, recovery publication | **all publishable**; required-set == builder key-set exactly |
| 14 | Same with a stale `cap_gpu_seconds: 7200` | **rejected** by the `const: 5222` — CHANGE-1 reached the schema |
| 15 | AST decode-guard on a mutated producer (`import decord`, `x.save()`, `from ...common import write_frame_pack`) | 3 of 4 sub-checks go **red** |
| 16 | Same with `.asnumpy()` injected | `producer_calls_no_decoder_attribute` goes **red** — all 4 sub-checks proven non-vacuous |
| 17 | Template-equality clause deleted from `assert_teacher_visible_containment` | `teacher_visible_benign_template_tamper_rejected` goes **red**; the plain tamper stays red via the ban scan. The carried-forward round-5 observation is real, not cosmetic |
| 18 | `NO_SEAL_SENTINEL` | no assignment in `gpu_ledger.py`; `G.NO_SEAL_SENTINEL is C.NO_SEAL_SENTINEL` -> **True** |
| 19 | **Selection reproduction** from the live ASR | 200+200; reproduces the **v6 and v7 frozen allowlists exactly on both datasets**; `label_value_materialized = 0`, `label_field_syntactically_skipped = 744 + 579` |
| 20 | **800 real containment renderings** on the real selected transcripts | **800/800 pass, 0 halts**, 402 banned tokens (400 ids + 2 HateMM label-bearing prefixes) |
| 21 | **v8 decode+PNG-encode vs v7's frozen packs**, 10 random HateMM items | **10/10 byte-identical**, incl. backend / total / requested_indices / decode-failure vector |
| 22 | **Two-pass vs one-pass pyav on the real fallback item** `MHC_zh/BV18N4y1B7qA` | decord genuinely fails (`DECORDError`); 2879 frames @1920x842; **48.5 s -> 7.0 s**; **PNG bytes identical**, `n` identical |
| 23 | Path containment | every staged path + frame-pack manifest path is inside the no-clobber namespace; the campaign ledger is outside it |

---

## Answers to (a)-(d)

**(a) `max_pixels = 151200` — legitimate at code/resource level; no design re-review needed to authorize a CPU preflight.**

I verified the authority claim rather than accepting it: all seven cited entrypoints do default to `360*420` at the exact cited lines (`generate_c02_density_view_text_embedding_HF.py:157`, `predict_target_qwen.py:299`, `generate_vision_summary.py:69`, `p10_score_segments.py:147`, `p10c_score_segments.py:137`, `score_segments_mllm.py:87`, `role3/arbitrate_qwen.py:363`), and `generate_VideoMLLM_embedding_bidir_textpool_HF.py:29` does call 151200 "the deployed max_pixels". Two of the config's paths are under `scripts/analysis/` rather than `src/utils/`, but the line numbers are all correct.

The decisive point is that the frozen design is *silent* on visual resolution — it fixes the frame count, the index rule, the black-frame rule, the transcript rule and the decoding parameters, none of which move. v7's "no `max_pixels`" was therefore not a reviewed choice either; it was an unreviewed default that made this teacher's visual input unlike every other Qwen2.5-VL call in the project. Choosing the deployed cap moves *toward* the reviewed protocol, not away. And it is not optional: at 43,056 pre-merge tokens the run is physically impossible on an A100.

It is also fully auditable: declared as a teacher-input change in the config, inside `config_contract_sha256`, inside all 400 frame-pack bindings, and `const: 151200` in the prompt-record schema's `provenance.teacher_max_pixels`, so every sealed record states what the teacher saw. Uniformity is preserved by refusing to salvage any v7 record. This is a no-performance-claim survival screen, and the amendment's non-waived gate binds the full bank, not this tranche.

Forward note, not a finding: if this tranche passes and a full-bank tranche is requested, the cap is by then part of the frozen teacher protocol and should be named explicitly in the proposal text at that review.

The fail-closed ceiling is correct on all three counts you asked about — `assert_visual_token_ceiling` runs at `producer.py:1536`, `model.generate` at 1545; the count is `visual_patch_tokens(prepared["video_grid_thw"][0])`, i.e. **pre-merge**, the quantity vision SDPA is quadratic in (the merged count would have been wrong by 4x); and it runs before `prepared.to(model.device)`, so nothing oversized ever reaches the card.

**(b) 5.3 % is acceptable — but the more important resource fact is that this is the last first-tranche allocation.**

Four independent reasons the margin is larger than 5.3 %: (i) the per-forward basis is v7's **native-resolution** mean, and the config's own least-squares fit (`1.9913 + 3.567e-4*tokens`) prices a capped v8 forward at ~3.0 s against 4.4248 budgeted, i.e. ~1140 s of unbudgeted slack; (ii) `non_forward_overhead_seconds` double-counts ~111 s of v7 frame-pack work that v8 provably does not do (I confirmed against v7's own timer placement); (iii) the corroborating capped-regime projection is 2588 s, a 36 % margin, and the 46 forwards it rests on are a *direct* measurement in the capped regime, correctly reported as corroboration only; (iv) crucially, **3807.9 is a prediction, not the gate** — the gate runs on measured inputs at preflight time, so a worse-than-predicted measurement HALTs before the namespace exists at zero GPU cost. The breach path is clean: exit 40, guard stops before an item, no output truncated, every completed checkpoint intact, accounting-only breach record, and the wrapper jq-verifies that record.

What the authorizer should know regardless: `1978 + 5222 = 7200` **exactly**. This reservation consumes the FIRST_TRANCHE phase ceiling to the second. If it breaches, there is no headroom for a v9 under the current phase — only `PASS_C04_SMALL_V2` plus a fresh result-to-claim GO plus a new code/resource review could raise it. That is a consequence of the amendment, not a defect, and the code enforces it correctly (5223 is refused against the live ledger).

The one arithmetic gap I found is immaterial and I state it for completeness: the gate compares against 4022 measured from job start, whereas the true bound is `4022 - claim_elapsed` (the guard deadline sits `300 + claim_elapsed` before the wrapper kill, because `BudgetGuard.at_job_start` receives the claim-time *remainder* and subtracts elapsed-since-entry again). `claim_elapsed` is a few seconds of small-file hashing — 412 staged files, of which 400 are ~1 KB manifests — against 111 s of double-counted slack. The double-subtraction errs *safe* on the guard side.

MHC_zh priced at the HateMM mean is the one unmeasured input, and declaring it is enough: the geometry it depends on is measured per item at preflight, both datasets land at the same capped grid (I measured 1080x1920 -> 2880 tokens, identical to HateMM's capped median), MHC-ZH transcripts are far shorter than HateMM's, and the basis it borrows is a native-resolution upper bound. The failure mode is a clean halt, not a corrupted bank.

**(c) Nothing is unbound. I walked the chain end to end and executed it.**

The chain: each pack's 8 PNG digests are pinned in its own `manifest.json`, and `strict_validate_frame_pack` re-hashes all 8 every time (I proved it catches a single flipped byte and a stray file); all 400 manifests enter `staged_output_hashes`; `verify_preflight_manifest` **re-hashes every staged entry at every stage** (claim, producer, reconciliation); `verify_payload_review` asserts `staged_output_hashes` and `preflight_manifest_sha256` equality; `verify_gpu_execution_authorization` pins `payload_review_sha256`. The three dropped fields are dropped because they *cannot exist* at preflight time, and the allocation binding is not lost — it lives in `provenance.allocation_claim_sha256`, which is in the prompt-record schema's `required` list, so every record still ties allocation -> pack -> frames.

Key-set drift is structurally impossible: `write_frame_pack` and `strict_validate_frame_pack` both derive their key set from the single `FRAME_PACK_BINDING_KEYS` / `FRAME_PACK_METADATA_KEYS` constants, and the only producer of a binding dict is `frame_pack_binding`, which `require_exact_keys`-checks it. Round-tripped for real. (`write_frame_pack` will accept a hand-built binding carrying an extra key, but the reader then rejects the pack and no such call site exists — not exploitable.)

The two-pass pyav is byte-identical, independently reproduced on the real fallback item; the second pass cannot select a different frame because it enumerates the same decode order and raises if any requested index is missing (which degrades to the frozen black-frame rule, now visible in the preflight manifest *before* the payload review instead of inside the GPU job — an improvement).

A failed gate leaves nothing: the namespace is created only at `os.rename(temp_namespace, namespace)` at `preflight.py:951`, everything before it lives in a `tempfile.mkdtemp` sibling that the `except` clause `rmtree`s, and `namespace.parent.mkdir` only touches the already-existing `artifacts/c04`. The projection HALT at line 903 raises inside that `try`.

No remaining path lets the producer decode a video: the AST guard parses the producer's frozen bytes for decoder imports, decoder attributes, PIL `save`, and the five frame-writing symbols from the common module (closing the transitive route) — and I proved all four sub-checks go red under mutation.

**(d) Strict equality is the right trade; warn-and-continue would be wrong.**

The equality cannot false-positive within a fixed environment: I measured that in transformers 4.49 `Qwen2_5_VLProcessor.__call__` routes videos to `self.image_processor(images=None, videos=videos, ...)` — literally the call the preflight makes — and the two paths produced identical grids on all 7 geometries I tested. The frames are byte-identical by hash, so the computation is deterministic end to end.

Its real value is therefore not memory safety (that is the separate `assert_visual_token_ceiling`, which runs regardless) but **desync detection**: it fires only if the processor tree, the pack bytes, or the transformers version drifted between the preflight and the GPU job. In exactly that case the teacher's visual input is no longer what the payload review approved, and continuing would silently seal a bank whose teacher input differs from the reviewed one. Losing the tranche is the cheaper failure. A warn-and-continue would convert an auditable halt into an unauditable bank.

---

## Non-blocking observations (no action required to proceed)

1. `preflight.py:848` — the freeze-stage `access_ledger.json` snapshots the audit *before* `materialize_frame_packs` runs, so its `guarded_runtime_evidence` holds 402 events while the same file claims `frame_packs_frozen_by_this_program: 400`. The preflight *manifest*'s `guarded_access_audit` (line 998) is complete (802 events) and is the artifact the payload review pins, so the trail exists; nothing compares the two. Cosmetic.
2. `resource_final_state.schema.json:111` — `terminal_sacct_gpu_seconds.maximum` is still 7800 (the v7-era 7200+600) while the code's `TERMINAL_SECONDS_HARD_MAX` is 5822. The code is strictly tighter, so the intersection is non-empty and every reachable state is publishable (verified at 5822). A stale bound, not a defect.
3. `assert_teacher_visible_precondition` runs in the producer, after the allocation and the single-use ticket are consumed, and has no preflight counterpart — structurally the shape of the hunted family. It cannot fire here: its inputs are the transcripts the preflight freezes and hash-pins, and I executed the check on exactly those frozen inputs (800/800 pass, item 20 above). Carried over from v7 unchanged. Worth moving into the preflight in any future version, purely as belt-and-braces.
4. Disk: the preflight will add ~2.0 GB of native-resolution frame packs and ~3,600 inodes. Usage is 280 G against a 290 G **soft** quota (3000 G hard), so it fits with ~8 G to spare, but the soft quota is close enough to be worth knowing before a job that writes for ~20-30 minutes.

---

## Builder's disposition of the four non-blocking observations

All four are **accepted as recorded, not repaired**, because repairing any of them
would move the reviewed bytes and void this `GO` — the same disposition v7 used
for its round-5 observations, one of which (`NO_SEAL_SENTINEL`) v8 then closed.

1. Accepted. The complete 802-event audit is in the preflight manifest, which is
   the artifact the payload review pins. Cosmetic.
2. Accepted. Verified independently: the code bound (5822) is strictly tighter
   than the schema bound (7800), so every reachable terminal state is publishable
   and the round-4 High shape (an empty code-schema intersection) does not recur.
3. Accepted and **carried forward as a named instruction for any v9**: move
   `assert_teacher_visible_precondition` into the CPU preflight. It cannot fire
   in v8 (its inputs are the preflight-frozen, hash-pinned transcripts, and it was
   executed on exactly those, 800/800), but it is the right structural fix.
4. Verified independently by the builder before submission: `quota -s` reports
   280 G used against a 290 G soft quota and a 3000 G hard limit, and the v7 tree's
   201 frame packs occupy 328 MB, consistent with the reviewer's ~2.0 GB estimate
   for 400 packs. Proceeding.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 8 (raw)

Verdict: REVISE - 0 Critical / 2 High / 4 Important. NO DISPOSITION CHANGED.
Exhaustive cross-surface sweep; all findings applied.

I re-derived the whole `[M]` layer from `data/gt/{HateMM,MHC_zh,MHC}/{train,val}.jsonl` and `/data/jehc223/Multihateclip/{English,Chinese}/annotation(new).json` from scratch, recomputed the C01 arm table from the stored confusion matrices, re-read the six banked `headspace_arena_*_OUT.json` and C02's `gates.ARENA2`, and checked every load-bearing quote against `directions_tried.json` / `findings.jsonl` / the cited `refine-logs`. **No fabricated number was found.** Every census figure reproduces exactly (including `10/140 = 0.0714` under the train-only keyword set, the entity triples `51/22/18` and `43/16/17`, EN title median 51 / transcript 322, ZH 27 raw / 13 stripped / 78, p90 `0.5051`/`0.5071`/`0.5155`, and the 6/6 fold-head floors). The **C14 strike is faithful and in-scope**. **All six downgrades are justified** — I independently confirmed C10 (EUM's "EMPTY **as of this recon**" over a three-item enumeration is verbatim, and BSY's own text scopes itself to "every future **bank-ADDITION** candidate"), C11 (the claim is literally disjunctive and ERRPAT §5.2's `p=0.0048` cluster is real), C12 (F55's "all three levels" are enumerated one field over as encoder/feature-composition levels — and `REDTEAM_BAN_SCOPE_AUDIT.md:227-230` already ruled the broad reading an "INDUCTIVE LEAP"). Every hold and the gate names a usable unblock; none is over-cautious. C09's promotion, its legality texts, and both zero-cost boundaries verify.

**New corroboration the record did not have:** the ZH `<em>` markup lives in **391 `Title` fields and 0 `Transcript` fields** of the source annotation — independent confirmation of §7's provenance claim that the marker rides on the harvested title.

What fails is the cross-surface sweep, again, and again on round 7.

## High

**H-1 · `TARGET_LOOP.md` — the headline of the surface a consumer schedules from states the wrong downgrade count, in the same sentence as its own correction.** It read: *"**But five of its seven recommended strikes** … are downgraded to HOLD — **six** of the seven."* Six were downgraded, as this file's own `### The six downgrades` section and all three other surfaces say. Round 7's I-1 is recorded as "count corrected", but the repair appended "— six of the seven" without deleting "five", leaving a self-contradiction and broken emphasis markers. **Repair:** rewrite cleanly.

**H-2 · Round 7 does not exist on three of the four surfaces.** The record says *"Seven rounds"*, lists seven verdicts and carries §14. But `TARGET_LOOP.md` said *"Six rounds"*, *"survives six rounds"*, Records `…ROUND{1..6}.md`, and had no Round-7 subsection at all; `TARGET_FINDINGS.md` said *"six rounds"* with exactly six verdict strings and the same truncated Records list; and `TARGET_STATE.json`'s `verification.headline` and `effective_order_post_c04.tally` both said *"six rounds"* — contradicting the same file's own `independent_review.round_7` block. The raw review **is** appended, so the omission is purely in the narrated history — but a consumer of either markdown surface is told the record survived six rounds and pointed at a file list omitting `ROUND7.md`. This is the identical single-surface-lag defect rounds 2-7 each charged, here on three surfaces at once. **Repair:** update all three, add round 7's verdict and a Round-7 paragraph, extend both Records lists, correct both JSON summary strings.

## Important

**I-1 · `TARGET_FINDINGS.md` — a derived figure is attributed to registry text.** The bullet read *"in the currency `banned_constraints[10]` already names (NET ITEMS, `37.2/29.0/27.5` for `+0.050`)"*. `banned_constraints[10]` names only *"NET ITEMS against 22.3 / 17.4 / 16.5 … for **+0.030**"*; the `+0.050` triple is a scaling. The other three surfaces label it correctly. **Repair:** restate as the `+0.030` figures scaling to the `+0.050` ones.

**I-2 · A house rule applied past its own written subject — the exact fault the record charges the recon with.** The record said *"The repo carries an explicit house rule against exactly this move — `LITSWEEP3_DATA_CENTRIC.md:80`"*. That bullet sits inside `LITSWEEP3 §4` (*"Memory-bank curation learned from TRAIN LABELS"*), and its "this mechanism" is curation; its stated ground for excluding the `C−A=0` null is *"that used an **MLLM two-vote** signal; here the deletion signal is **gold train labels + kNN geometry only**. Different information source."* C12's stability statistic is archive-version-derived, i.e. on the MLLM side of that very distinction — so the source's own ground does not transfer. The downgrade is unaffected: gap_1's independent leg stands on its own. **Repair:** state LITSWEEP3:80's own subject and cite it as a general-form precedent, or rest gap_1 on the inference-time/training-time leg alone.

**I-3 · Round 7's I-2 repair did not reach the two consumer surfaces.** The record and `TARGET_STATE.json` now record the C09 counter-text as *"downgraded, not vacated"*, because `LITSWEEP5 §4(ii)`'s selector leg rests on the *"CLIP LOO 0.998"* premise F114 retracted and LITSWEEP5 is not among F114's nine corrected records — both verified. `TARGET_LOOP.md` and `TARGET_FINDINGS.md` still stated it at full strength, i.e. carried an overstated headwind against the candidate the record promotes. **Repair:** add the downgrade and §4(ii)'s untouched independent leg (`0/109`, `0/102`, `0/92`) to both.

**I-4 · "D7 is an open user ruling" is wrong as written, and inconsistent with the record's own unblock 20 lines later.** `research-wiki/DECISION_MEMO_pending.md:211` and `:134` record D7 as *"✅ RESOLVED 2026-07-14 (RESOLVED-NEGATIVE)"*, binding. What is open is the **D7 generator-role sub-ruling** — which is what F60 actually demands and what the record's own unblock says correctly. (`HANDOFF_2026-07-28.md:76` does list "D7's novelty boundary" as open, so the repo is not unanimous — which is precisely why the narrow form should be used.) The conclusion is unaffected. **Repair:** narrow the sentence on both surfaces.

## Checked and cleared (not raised)

The C14 strike's basis, scope, reversibility language and unblock; all ten registry `status` strings matching their disposition-block counterparts; `ordered_backlog` genuinely untouched; `hard_constraints`, `unified_pilot_gate` and `serial_execution` unaltered; C02's entry unedited and §3.8's `OBS-1` accurate; `dispositions.gated` correctly separate from `held`; C09's prereg a DRAFT, not frozen, not submitted; the `$0`/zero-touch/C04-untouched attestation with all four provenance sources; §3.1's coverage-bound scope correction and §3.2's "same table" claim; the recon's recommendations as characterised; §3.5 (all LBOP line numbers verified individually), §3.6, §3.7, §3.9; C06's arm table cell-for-cell including both spreads and the 4-of-6 / 2-of-6 counts; the ro-cache inventory and the `ow_` prompt/span confound; C08's premise-1 refutation; and "§1.3's `+0.0286`" correctly referring to the *recon's* §1.3.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 9 (raw)

Verdict: REVISE - 0 Critical / 1 High / 3 Important. NO DISPOSITION CHANGED.
Three of round 8's six repairs had not landed; all now applied.

I re-derived the whole `[M]` layer from scratch (six gt files, both `annotation(new).json`, `C01_A0_OUT.json` confusions, `C02_A0_OUT.json` gates, the six banked `headspace_arena_*_OUT.json`, `data/Archive/`, `CLIP_Embedding/`, the `ro_*` cache inventory) and checked every load-bearing quote against `directions_tried.json` / `findings.jsonl` / the cited `refine-logs`.

**What holds.** Every census figure reproduces exactly (key-set `['id','label','text']`, ws-only 39/9 + 0x4, ZH tags 243/34 with `em` 254/`/em` 254, 141/243=0.5802 vs 39/336=0.1161 base 0.3109, val 20/34 vs 8/44, 49/254 train and 50/288 train+val, p90 `0.5051`/`0.5071`/`0.5155`, max 0.862069, 203>10%, bearing-median 0.2604, entity rows 64/9 and ZH 1 strict/2 hex, `10/140 = 0.0714`, EN title 51/transcript 322, ZH 27 raw/13 stripped/78). The C01 arm table recomputes cell-for-cell on both datasets, including both rotation spreads and the 4-of-6 / 2-of-6 counts. The 6/6 fold-head identity holds at 4 dp. **No fabricated number was found.** The **C14 strike is faithful and inside its own text**. **All six downgrades are justified at their sources' written scope** — I independently re-derived C10 (EUM's "EMPTY **as of this recon**" over a three-source enumeration; BSY's own "bank-**ADDITION**" scoping), C11 (the registry claim is literally disjunctive; ERRPAT §5.2's `p=0.0048` cluster is real and correctly tier/split/pooled-scoped), and C12 (F55's three levels are enumerated one field over as encoder-composition levels). Every hold and the gate names a usable unblock; **none is over-cautious**. All ten registry `status` strings equal their disposition-block `new_status`; reversibility is present and uniform; `ordered_backlog`, `hard_constraints`, `unified_pilot_gate` and `serial_execution` are untouched; C02 unedited; all eight raw reviews appended.

What fails is the author's claim that **all** eight rounds' findings are applied — three of round 8's six are not.

## High

**H-1 · Round 8's H-1 was never applied. `TARGET_LOOP.md` still carries the exact self-contradiction it charged, in the headline of the surface a consumer schedules from.** The file reads: *"**But five of its seven recommended strikes** rest on a ban … and are downgraded to HOLD — **six of the seven**. Exactly **one** survives…"* — including the broken emphasis markers R8 described. R8's finding was: *"round 7's count repair … appended '— six of the seven' without deleting 'five' … **Repair:** rewrite cleanly."* Both the record and `TARGET_STATE.json` record the disposition as **"rewritten cleanly"**. It was not. The same file's own table 20 lines later says "One strike, seven holds", so `TARGET_LOOP.md` contradicts itself. This is the third consecutive round in which a repair claimed applied is absent from the operative scheduling surface. **Repair:** rewrite once, cleanly, with balanced emphasis markers.

## Important

**I-1 · Round 8's I-4 did not reach `TARGET_STATE.json`. The machine-readable surface still asserts a false fact about a user ruling.** C12 `gap_2` ends: *"**D7 is an OPEN USER RULING**, so the narrow reading is not a free pass."* `research-wiki/DECISION_MEMO_pending.md:134` and `:211` record D7 as **`RESOLVED 2026-07-14 (RESOLVED-NEGATIVE)`** — I verified both lines. What is open is the **D7 generator-role sub-ruling** F60 demands, which the *same JSON object's* `unblock` field states correctly, so the block is internally inconsistent. The record, `TARGET_LOOP.md` and `TARGET_FINDINGS.md` all carry the narrowed form. **Repair:** narrow `gap_2`.

**I-2 · Round 8's I-2 landed only on the record. Three of four surfaces still read `LITSWEEP3_DATA_CENTRIC.md:80` past its own written subject — the precise fault the record charges the recon with.** `TARGET_STATE.json` C12 `gap_1`: *"LITSWEEP3_DATA_CENTRIC.md:80 **rules explicitly against this move**…"*; `TARGET_LOOP.md`: *"against `LITSWEEP3_DATA_CENTRIC.md:80`'s **explicit ruling**…"*; `TARGET_FINDINGS.md`: the same. I verified the source: that bullet sits inside `LITSWEEP3 §4` (heading *"Memory-bank curation learned from TRAIN LABELS"*), its "this mechanism" is **curation**, and its stated ground is that the null *"used an **MLLM two-vote** signal; here the deletion signal is **gold train labels + kNN geometry only**. Different information source."* — the side C12's archive-derived statistic is *not* on. The record itself now states this correctly. **Repair:** re-scope on all three surfaces, or rest C12's `gap_1` on the inference-time-vs-training-time leg alone (which stands unaided).

**I-3 · The provenance attestation mis-states the scope of the `annotation(new).json` measurements, on all four surfaces, and the mis-statement runs against the record's own "zero test-split contact" line.** All four attest that the annotation files were *"**joined only to train+val ids**, for the `Title` presence counts and the per-dataset title/transcript medians"*. The medians are indeed join-scoped (I reproduce EN 51/322 and ZH 27 raw / 13 stripped / 78 exactly over train+val ids). **The `Title` presence counts are not**: `891/891` and `897/897` are the *whole-file* row counts, against a train+val join of **629** and **657** (which give `629/629` and `657/657`). Round 8's new corroboration has the same problem: `391 Title / 0 Transcript` is whole-file; join-scoped it is **277 / 0** (= 243 train + 34 val, matching the gt census). The files carry a `Label` field on every row including test ids, so a whole-file pass is exactly what the join was declared to avoid. Nothing improper follows — only `Title` presence was read, no label was consumed, and the C08 downgrade is unaffected because the join-scoped counts make the identical point — but this is the same class of attestation defect rounds 5 (I-1) and 6 (I-1) each forced a repair for, and it is now false again. **Repair:** restate the counts at the attested scope, or amend the attestation to say the counts were taken over all rows reading only `Title`/`Transcript` with no label — and reconcile with the "zero test-split contact" cost line.

## Not raised, checked and cleared

The C14 strike's basis, scope, reversibility and unblock; all six downgrades (C07's F82 vote-side/head-side split incl. *"HateMM out of scope (no Offensive class)"*, the dev-split gold cheat at n=80/78 and ZH's `+0.0256` = 2 dev items; C08's premise-1 refutation via `prep_mhc.py:72-85` and `prep_video_dataset.py:126-139`, verified directly; C10 vs EUM/BSY; C11's disjunctive claim with `:405` quoted in full; C12's F55 leg, corroborated by `REDTEAM_BAN_SCOPE_AUDIT.md:227-230`'s "INDUCTIVE LEAP" ruling; C13's registry text); C05's `unwritten_as_posed` strength and its four comparator quotes; C06's gate incl. the one-lineage-per-dataset ro-cache correction and the `ow_` prompt/span confound; C09's promotion, F88 numbers, zero-GPU Stage-0, legality on `progress.json:25` + `LITSWEEP3:82` with the Wall-A parenthetical and `:91`/`:95` pricing, the F78→F88(3) re-attribution, and the F114/LITSWEEP5 downgrade-not-vacated note; §3.1-§3.9; the strategic finding figure by figure incl. the honestly recorded F98 superlative disagreement and `banned_constraints[10]` correctly labelled as scaling; and the `$0` / zero-touch / C04-untouched / C02-unmodified boundary.


================================================================================
C04-A0T-SMALL-v1 impl-v8 PAYLOAD REVIEW (round 1) -- raw
Appended 2026-08-01. Verdict: GO (0C/0H/5I). Subject: the materialised namespace
from CPU job 13855. Full file: refine-logs/C04_A0T_SMALL_V1_V8_PAYLOAD_REVIEW.md
================================================================================
# C04-A0T-SMALL-v1 impl-v8 — Independent Payload Review (round 1)

Date: 2026-08-01
Reviewer: fresh independent Opus payload reviewer; zero exposure to the build
reasoning and zero exposure to the code/resource review.
Subject: the materialised namespace `artifacts/c04/a0t_small_v1_impl_v8/`
produced by CPU SLURM job `13855` (COMPLETED, 0:0, 976 s, no GPU in AllocTRES).

## VERDICT: `GO (0C / 0H / 5I)`

No defect was found that the GPU stage would re-read and reject. Every hash,
every derivation, and every measured number in the frozen payload reproduced
independently, including several recomputed from scratch rather than checked
against the project's own code.

## What was RECOMPUTED, and what each gave

### 1. Hashes (all 414 + self-digests)
| quantity | result |
|---|---|
| `freeze/preflight_manifest.json` file SHA-256 | `bd93adf8...d50c1b` — equals the request anchor |
| manifest `payload_sha256` recomputed over body | `07ab0ef3...f187f6` — matches claim |
| `staged_output_hashes` re-hashed against disk | **414/414 match, 0 missing, 0 mismatched** (400 frame-pack manifests + 8 freeze artifacts + 4 map files + 2 resource files) |
| `resource/resource_ticket.json` `payload_sha256` | `023f6f47...0b489c` — matches |
| ticket `genesis_gpu_ledger_sha256` = `7b7b71ce...41e4a` | equals sha256 of the namespace `resource/gpu_ledger.json` bytes |
| ns `gpu_ledger.json` `payload_sha256` | matches; state `GENESIS_UNCLAIMED`, `jobs: []`, `ledger_revision: 0` |
| campaign ledger | payload digest OK, single row (job 13852, 1978 s), `previous_payload_sha256 = GENESIS`, row digest OK, head OK, aggregate == rows |
| campaign headroom | effective cap `min(7200, 28800) = 7200`; `1978 + 5222 = 7200 <= 7200` passes; `+5223` refused |

### 2. The 200+200 selection — reviewer's own implementation, no project imports
| | HateMM | MHC-ZH |
|---|---|---|
| ASR file SHA-256 | `d47d4062...f24124` = config pin | `1c3ce3d2...4b21a5d` = config pin |
| pool | 744 unique | 579 unique |
| **exact match to frozen allowlist incl. rank order** | **True, 0 diffs** | **True, 0 diffs** |
| rank 0 | `hate_video_334` / `003d3d31...` | `BV1vs4y127aA` / `001d1c3c...` |
| rank 199 | `hate_video_56` / `44f82322...` | `BV1Tp4y1k7HG` / `587975847...` |
| merkle_root | `5897b44c...` match | `24d40b0e...` match |
| source-manifest merkle | `a8eab8ad...` match | `af2f8d7a...` match |
| cross-dataset ID overlap | empty | |

`projected_field_counters`: `label_field_syntactically_skipped = 1323` — exactly
`744 + 579`, one per ASR line; `label_value_materialized = 0`. Zero fields named
`label*` anywhere in the 413 namespace JSONs.

### 3. Prompt hashes — derived from the module source via `ast`, not by calling their function
All four (`system` `1ffc0675...`, `A` `cecb3555...`, `B` `9521bee7...`,
`combined` `a42268e4...`) match the v6 freeze, the manifest and
`freeze/prompt_hashes.json`. Freeze artifact `payload_sha256` `9e47a9dc...e719d5`
recomputed and matches; file SHA-256 `671fddab...5ff651` matches both the
manifest attestation and its staged entry. `guarded_access_audit.events_merkle_root`
`1df335bc...b0dffe` recomputed over all 802 events — match.

### 4. Frame packs — all 400 manifests and all 3200 PNGs
- **3200/3200 PNG SHA-256 and byte sizes match** their manifest rows; no symlinks.
- 400/400 manifest `payload_sha256` recomputed; exact key set; all 13 binding
  fields re-derived independently — 0 mismatches.
- `requested_indices` re-derived as `min(N-1, floor((i+0.5)*N/8))` with the
  `N==0 -> []` branch: 400/400 match.
- Directory contents exactly `{manifest.json, 00..07.png}` for all 400; pack-root
  entry sets equal the allowlists. Every manifest's file hash equals its
  `staged_output_hashes` entry.
- Backends: decord 388, pyav 11, none 1. The 11 pyav items are all MHC_zh
  (`BV18N4y1B7qA`, `BV16M4y1c7eo`, `BV1oS4y1c7v3`, `BV1iD4y1B7Sw`, `BV1Yh411B7XD`,
  `BV1tq4y1G74f`, `BV1UK4y1j7N9`, `BV1CN4y1h79K`, `BV13j411R79M`, `BV1fU4y1X7Gu`,
  `BV1AT411y7ER`); all eight frames verified for each.
- `HateMM/hate_video_95`: `total_frame_indices: 0`, backend `none`,
  `frame_decode_failed: [true]x8`, `requested_indices: []`; all eight PNGs are
  336x336 with per-channel extrema `(0,0)` — pure black, byte-verified.
- **v7 vs v8 HateMM frame bytes: 0 differences over 1600 PNGs.**

### 5. Visual geometry — independently re-measured for ALL 400 items
Ran the frozen PNGs through
`AutoProcessor.from_pretrained(snapshot, local_files_only=True, max_pixels=151200).image_processor`
on CPU (transformers 4.49.0, torch 2.6.0).

- **400/400 exact match** on `video_grid_thw`, `patch_tokens`, `frame_size`,
  `merged_tokens`, `vision_sdpa_fp32_bytes`.
- HateMM min 1456 / median 2880 / max 3072 / mean 2848.16 (manifest 2848.2)
- MHC_zh min 480 / median 2880 / max 3072 / mean 2865.44 (manifest 2865.4)
- Global max 3072 -> `3072^2*16*4 = 603,979,776 B = 0.562 GiB`.

**The check that mattered most.** The preflight measures through
`processor.image_processor(...)`; the producer measures through the full
`processor(text=[chat], ...)` and hard-halts on
`tokens != expected_visual_tokens`. On transformers >= 4.5x these can diverge (a
separate `video_processor`). Both paths were run on **38 items** (incl.
`hate_video_95`, all 11 pyav items, both datasets' extremes, 20 random):
`Qwen2_5_VLProcessor` in 4.49.0 has **no `video_processor` attribute** and routes
videos to `self.image_processor` with the same `max_pixels`. **A-vs-B
disagreements: 0. Path-B vs frozen mismatches: 0.** The per-forward equality
assert is satisfiable.

### 6. Projection arithmetic — recomputed from inputs
```
window     = 5222 - 300 - 300 - 600           = 4022    (claim 4022)
forwards   = 800 x 4.4248                     = 3539.8  (claim 3539.8)
total      = 208.1 + 69.7 + 3539.8            = 3817.6  (claim 3817.6)
margin     = 4022 - 3817.6                    = 204.4   (claim 204.4)
fraction   = 204.4 / 4022                     = 0.0508  (claim 0.0508)
affordable = (4022 - 208.1 - 69.7) / 800      = 4.6803  (claim 4.6803)
capped     = 208.1 + 69.7 + 800 x 2.9         = 2597.8  (claim 2597.8)
had the 834.7 s of CPU work stayed in-job: 4652.3 > 4022 -> the gate would have FAILED
```

**The basis was also refit from raw v7 data.** Native geometry re-measured for
v7's 200 HateMM packs (no `max_pixels`) and its 400 recorded `elapsed_seconds`
regressed:
- v7 native tokens: min 1456 / median 8160 / max 9248 / mean 6823.1 — matches the
  `basis_note` exactly.
- Least squares: `elapsed = 1.9913 + 3.567e-04 x native_tokens` — identical to the
  claimed fit.
- v7's 46 forwards on the 23 items where native geometry already equalled capped
  geometry: mean **2.9002 s** — exactly the claimed 2.9.
- Regression evaluated at v8's capped geometry: **3.0072 s/forward -> 800 forwards
  = 2405.7 s**, total ~2683.5 s, margin ~**1338.5 s (33.3 %)**.
- Residual sd 0.834 s, p95 +1.06 s, max +5.25 s.
- Sum of v7 forwards = 1769.9 s; job 13852 elapsed 1978 s => residual 208.1 s. v7
  has exactly 200 HateMM + 1 MHC_zh frozen packs, so that residual already
  contains both datasets' full `load_selected_inputs` and v7's 201 in-allocation
  pack builds (~110 s) that v8 does not repeat.

### 7. Label containment and boundary hygiene
- **800/800 prompt renderings** re-assembled independently and scanned against
  the full ban set — 402 tokens expanded to 602 NFKC/casefold variants, against
  NFKC and casefolded haystacks. **0 violations.**
- All 400 transcripts re-derived from the raw ASR: 400/400 `transcript_sha256`
  match, 0 language mismatches, 0 scalar-count mismatches.
- All 400 videos: `resolve()` lands in the pinned physical root, symlink policy
  holds, live `st_dev`/`st_ino`/`st_size` match the frozen manifest for all 400.
  **All 400 `video_sha256` recomputed over 3.43 GB — 0 mismatches.**
- `_forbidden_path_component` re-implemented and applied to every path the GPU
  stage will `root_path()`: 414 staged + 30 bound-file-map + 400 lexical/resolved
  video paths — 0 forbidden components.
- Namespace: 3615 files = 3200 `.png` + 413 `.json` + 2 `.f32le`. No dev/test
  path, no cross-dataset path, no OCR artifact, no network/API artifact.

### 8. Maps, contract, bound files, model snapshot
- The four role maps re-implemented from scratch (HashStream + Fisher-Yates +
  sign-bit extraction) -> objects byte-identical to the frozen files.
- `le3_256x3598.f32le` (3,684,352 B) and `additive_256x1024.f32le` (1,048,576 B)
  re-derived byte-for-byte identical.
- `config_contract_sha256` re-implemented -> `e4d041df...6fe294`, equal to the
  manifest, the ticket and the ns ledger. Verified invariant under the exact
  post-review config amendment the pipeline requires. Control: amending
  `maps.expected_hashes` does move it, so the normalization is not vacuous.
- `implementation_hashes` (15) and `frozen_design_hashes` (15): 30/30 live files
  match their pins.
- Code/resource authorization `c0a12d1a...150344`: file hash = config pin =
  manifest pin; `closure_sha256` recomputed OK.
- Model snapshot: all 14 files re-hashed; `model_tree_sha256 = 55705d03...e28c22`
  and `processor_tree_sha256 = f77f6022...20710e` recomputed — both match.
- `self_test`: 74 checks, all true.

## GPU-stage satisfiability trace

Order of irreversibility in the GPU wrapper: jq authorization gate ->
preflight-manifest existence -> entry marker -> `claim` -> **ticket consumption**
-> `flock` -> producer.

`claim()` runs `validate_gpu_environment` (env + `assert_campaign_aggregate_headroom(5222)`)
and then **`verify_gpu_lineage` — which re-hashes all 414 staged outputs,
`verify_historical_code_resource_authorization` (including both
`verify_bound_file_map` sweeps), `verify_payload_review` and
`verify_gpu_execution_authorization` — BEFORE the ticket is consumed.** The
ordering is correct for this failure family.

Checks that fire only after consumption, and their status:

| post-consumption check | status |
|---|---|
| `verify_authorization` (env, constants, headroom, 30 bound files, literal prompt-hash path) | satisfied |
| `verify_model_snapshot` (14 files, two tree hashes) | satisfied |
| `verify_claimed_resource` (ticket digest, genesis pin, `consumed==False`, `4922 == 5222-300`, `5222 <= 5222`) | satisfied |
| `load_selected_inputs` x2 (file hashes, rank order, per-row digest, sorted replay, ASR hash, 400 transcript hashes, 400 st_dev/st_ino/video_sha256, lexical path) | all recomputed, satisfied |
| `assert_teacher_visible_precondition` (800 renderings vs 402-token ban list) | 0 violations |
| `strict_validate_frame_pack` x400 (manifest digest, key set, 13 binding fields, backend enum, index rule, decode vector, 8 rows, dir entries, 3200 hashes+sizes, inode identity, staged pin) | 400/400 satisfied |
| `visual_patch_tokens(...) != expected_visual_tokens` x800 | 400/400 satisfied via both processor paths; `visual_geometry.items` covers exactly the 400 selected IDs (no KeyError) |
| `assert_visual_token_ceiling` (<= 4096) | max 3072 |

Pre-state is clean for `claim()`: `resource/` holds only `gpu_ledger.json` and
`resource_ticket.json`; `seal/` absent; neither checkpoint directory exists, so
`load_checkpoint` returns `{}` and `idempotent_complete` short-circuits False.
`verify_reconciliation_lineage` also passes with `allow_claimed_gpu_ledger=True`,
so the terminal CPU stage is satisfiable.

**Residual, unverifiable-now dependency:** `payload_review` and
`gpu_execution_authorization` do not yet exist. `verify_payload_review` requires
`body["staged_output_hashes"] == preflight["staged_output_hashes"]` (all 414
verbatim), `preflight_manifest_sha256 == bd93adf8...d50c1b`, plus the two-layer
digest `sha256_obj(core) == reviewed_payload_sha256` and
`attested = sha256("C04-PAYLOAD-REVIEW-GO-v8\n" + reviewed_payload_sha256)`. The
payload-review schema is consistent with that reader and constructible.

## Findings — five Informational, none a repair blocker

**I-1 — a breach after 400 completed HateMM forwards yields no seal and no verdict.**
`run.dataset_execution_order = "strictly_serial"` with `run.datasets =
["HateMM","MHC_zh"]`; the seal requires all 800 records. The exit-40 path
preserves per-item checkpoints, but no authorized stage can consume a 400-record
partial, and `resubmit_authorized: false` plus `single_allocation_only: true`
close the namespace. A breach converts a fully completed HateMM dataset into zero
scientific output. No repair available inside a no-clobber namespace; the
GPU-execution authorization should record this explicitly as the accepted
downside.

**I-2 — `guard_seal_reserve_seconds = 600` is the only wholly unmeasured constant
in the window derivation.** v7 never reached the seal. If it is too small the
overrun is killed by the wrapper `timeout` (exit 124, no seal AND no breach
record — strictly worse than exit 40). Mitigating measurements: the guard
deadline sits at entry+4622 while `timeout` fires at entry+4922, so the seal has
>= 900 s even pessimistically (~2200 s in the corroborated case); the seal's
heaviest measurable component is ~1600 `strict_validate_frame_pack` calls
(~6.8 GB of PNG re-hashing), clocked at 3200 hashes in 1.6 s. No repair needed.

**I-3 — MHC-ZH transcript poverty in the frozen selection.** Measured from
`freeze/MHC_zh.source_manifest.json`: MHC_zh p10 = 8 unicode scalars, median
63.5, and **24/200 items carry fewer than 10 scalars**, against HateMM p10 = 38 /
median 999. Zero items are empty, so every prompt renders and nothing halts. If
the tranche returns `KILL_C04_TEACHER_SEMANTIC_RELIABILITY` on MHC_zh, the
failure is not separable from input poverty at this sample size. No repair
permitted (no redraw, no replacement under the amendment); the result-to-claim
stage must carry this caveat.

**I-4 — `HateMM/hate_video_95` is a frozen all-black pack.** Contract-conformant
(`teacher_contract.zero_frame_rule`) and byte-identical to v7's pack. Its
transcript is full (2061 scalars), so the item is not a total void. 2 of 800
forwards see no visual evidence. Named so the usable-rate statistics can be read
with it in view.

**I-5 — the audit ledger pseudonymises inconsistently.** Each `HASH_TRAIN_VIDEO`
event stores `video_id_sha256` (hashed) but `resolved_train_relative` in the
clear (e.g. `"hate_video_334.mp4"`), i.e. the label. **Not** a contract breach —
the amendment binds what a *teacher forward* can read, and the teacher-visible
surface was verified exhaustively clean — but the audit artifact hashes one copy
of the identifier and prints the other. Zero runtime cost; the risk is a later
stage mistaking the audit ledger for a label-free artifact. No repair inside the
no-clobber namespace; recorded here.

## Judgement on the 5.1 % margin

**Acceptable — but the 5.1 % is not the number the decision rests on, and the
record should say so.**

204.4 s is ~46 forwards at the basis rate. As a safety margin it is thin enough
to be consumed by any unmodelled 5 % effect, and if 4.4248 s were the honest
expectation it would be inadequate. It is not the honest expectation; it is a
deliberately wrong-in-the-safe-direction bound, and the wrongness was verified
rather than accepted:

- The basis was measured at **native** resolution (mean 6823.1 pre-merge tokens,
  independently re-measured). The run executes at **capped** geometry (mean
  2848.2 / 2865.4, max 3072) — 2.4x fewer tokens, on the quantity vision
  attention is quadratic in.
- An independent least-squares fit over v7's 400 forwards reproduces
  `1.9913 + 3.567e-04*tokens` exactly. At v8's frozen geometry: **3.0072 s/forward
  -> 2683.5 s total -> 1338.5 s margin (33.3 %)**.
- The 46 v7 forwards where native geometry already equalled capped geometry are a
  direct, unextrapolated measurement in the target regime: **2.9002 s** ->
  2597.8 s -> **35.4 % margin**. Two independent estimators agreeing within 3 %.
- `projected_fixed_overhead_seconds = 208.1` still contains v7's 201
  in-allocation frame-pack builds (~110 s) that v8 does not repeat.
- The one genuinely unmeasured input, MHC-ZH forward cost, is bounded on the
  favourable side by two measurements: its capped geometry is essentially
  identical to HateMM's, and its prompts are **shorter** in tokens (mean 287.5 vs
  482.8; median 253 vs 476), so prefill is smaller, not larger.

To breach, the true mean per forward would have to exceed **4.6803 s** — exceed
v7's *native* mean while running on 2.4x fewer visual tokens. Even pricing every
forward at the regression mean plus one residual sd (3.84 s) lands at ~3350 s, a
17 % margin. No mechanism produces the breach regime.

Two things further reduce the cost of being wrong, verified rather than assumed:
the failure is a clean exit-40 that never truncates or alters an output; and the
campaign accumulator records **actual sacct seconds, not the reservation** — a
clean ~2700 s run leaves ~2500 s of FIRST_TRANCHE headroom for a future C04
namespace rather than burning 7200/7200. The 5222 s is a reservation, not a spend.

The countervailing fact the record must state plainly is I-1: because the seal
requires all 800 records and the namespace cannot be resubmitted, a breach at any
point destroys the scientific output of a fully completed HateMM dataset. That
raises the stakes of the tail; it does not change the arithmetic. **GO.**

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 10 (raw)

Verdict: REVISE - 0 Critical / 1 High / 3 Important. NO DISPOSITION CHANGED.
18/18 [M] claims independently re-derived and matching exactly.

## H-1 · Round 9 does not exist on two of the four surfaces — the third consecutive recurrence of the defect rounds 7-9 each charged

The record and `TARGET_STATE.json` plus `independent_review.round_9` all state **nine** rounds. Both narrative surfaces still stated **eight**: `TARGET_LOOP.md` in four places (*"survives eight rounds"*, *"Eight rounds of fresh independent review"*, *"survives eight rounds"*, Records `ROUND{1..8}.md`) and `TARGET_FINDINGS.md` in four (*"through eight rounds"*, *"survives eight rounds"*, *"survived eight rounds"*, Records `ROUND{1..8}.md`). Grep for `ROUND9|round_9|round 9` returns **0 hits** in both, against 2 in the record and 3 in the JSON. This is the identical defect round 8's H-2 charged for round 7, recurring for round 9, and the review request makes cross-surface agreement on **counts** a GO criterion.

*(Round 9's raw review **is** appended, and round 9's own four findings all landed: H-1's "five of its seven" is gone from `TARGET_LOOP.md`; I-1's D7 narrowing is in `TARGET_STATE.json`; I-2's `LITSWEEP3:80`-as-general-form is on all four; I-3's attestation is restated on all four with `629/629`, `657/657`, `277 / 0` and the no-test-id clause. Only round 9 **itself** is missing.)*

**Repair:** sync both surfaces, add a round-9 paragraph and verdict, extend both Records lists.

## I-1 · The C08 title-median provenance cites a line range that does not contain the figure

Both the record and `TARGET_STATE.json` say the *"title 15 chars, transcript 76, composed 96"* median was inherited *"via F88 ledger correction (c) from `ERRPAT_MHC-ZH_2026-07-26.md:270-271`"*. Read directly: `:271` is the composition line (*"Deployed ZH text = `Title + " . " + Transcript` … (`scripts/prep_mhc.py:73-78`)"*); the medians are at **`:272`** (*"Medians on test: title 15 chars, transcript 76 chars, composed text 96 chars"*). The scope qualifiers the record attaches (Tier-2, test split, markup-stripped) are all correct — only the anchor is wrong. Same class as the pinpoint errors rounds 2 and 3 charged, introduced by round 4's H-2 repair. **Repair:** `:270-271` → `:272` on both surfaces.

## I-2 · EUM's broad reading of `banned_constraints[5]` is attributed to `[5]` alone; EUM stacks four authorities

The record: *"Broad-reading precedent exists (EUM glosses it as covering 'MLLM-derived boundaries or weights'), so if the stability statistic weights or selects training examples the ban applies."* EUM's precondition (2), verbatim: *"WITHOUT MLLM-derived boundaries or weights (that is **P3 / P11 plus** banned_constraints[5] 'MLLM-scores-as-training-signal' **and [6]** 'P1-P5 re-proposals')"*. EUM does not gloss `[5]` as reaching boundaries/weights — it reaches that object through a four-authority stack. In a record whose entire methodological finding is that authorities must be read at their own written scope, and whose C12 unblock turns on *"lands on `[5]` under EUM's gloss, and is then dead"*, the attribution matters. (The error runs in the conservative direction — the stacked authorities make the closure stronger, not weaker — so the C12 disposition is unaffected.) **Repair:** state the stack on both surfaces.

## I-3 · F80 is quoted at partial scope in the C06 warrant

The record: *"F80's object is prompt language … on MHC_zh, and its prohibition is conditional: 'do NOT re-propose prompt-language matching elsewhere without new mechanism' — the recon truncates the qualifier."* F80's ban_scope in full opens with an **unconditional** on-dataset closure the record omits: *"extraction-instruction language variations (**any language, any stream, either encoder arm**) on MHC_zh; prompt-language axis measured null-to-negative; do NOT re-propose prompt-language matching **elsewhere** without new mechanism (HateMM/EN are English-content = no mismatch exists)"*. The conditionality attaches only to *elsewhere*. The C06 warrant survives intact — it rests on **object mismatch** (orbit geometry is not extraction-instruction language), which is unaffected either way — but the record charges the recon with truncating this same entry's qualifier and then truncates its other half. **Repair:** quote F80's opening clause alongside the "elsewhere" clause on both surfaces.

## Checked and cleared (not raised)

- **The C14 strike is faithful and in-scope.** `eligible_for_primary_target: false` and the dedup-boundary sentence are verbatim; `hard_constraints[4]` is verbatim. The strike is scoped to the performance backlog with the diagnostic role preserved, the unblock is a user ruling, and TVB's *"7 of 7 at ~0"* is correctly identified as a **prediction** and explicitly not relied on.
- **All six downgrades justified**, C10/C11/C12 re-derived from primary text. C12's decisive leg holds; `REDTEAM_BAN_SCOPE_AUDIT.md:230` independently rules the broad reading an *"INDUCTIVE LEAP"*. C10's EUM ban does name the object but supplies three revival preconditions — conditional closure implies HOLD. C11's disjunct is genuinely open, with ERRPAT's *"No legal unmeasured lever found"* quoted in full and a written strike path.
- **Kind of record and reversibility:** all ten registry entries carry the reversibility string verbatim; ten `status` strings match their `new_status`; `dispositions` arrays sum to 10 and match the tally; `ordered_backlog`, `hard_constraints`, `unified_pilot_gate`, `serial_execution` unaltered; C01-C04 carry no `gate0_reopen` key; C09's prereg is a DRAFT.
- **Nothing recorded as measured that is not.** I re-derived the entire `[M]` layer independently — **18/18 claims MATCH exactly**, including `891/891`, `897/897`, join-scoped `629/629`, `657/657`, `391`/`0` and `277`/`0`, title medians `51`/`322` and `27`/`13`/`78`, `10/140 = 0.0714`, `49`/`254` and `50`/`288`, the p90 triple, `0.2604`, `203/579`, the entity triples, and both regex conventions. The C01 arm table recomputes cell-for-cell including both rotation spreads and the 4-of-6 / 2-of-6 claim. C02's `ARENA2.pooled_native_acc` equals the six banked `head_deployed_acc` at 4 dp on **6/6**, and `0.8875`/`0.8912` are confirmed to be `summary_3seed.FULL.acc_3seed_mean` — §3.7's correction is exact. The "≈ 36 heads" estimate is right (`K_FOLDS=5` plus the `fold == -1` deployed head). F113's honesty clause is byte-identical and in F113 only. F60's *"un-enumerated generator role is real"* is verbatim. The `3 x 3 x 2 x 2 = 36 cells` decomposition is written at `MECHNOV_PAIRVERIFY_PREGATE.md:202,308`. **No fabricated number found.**
- **No over-cautious hold.** Every hold and the gate names a proponent-actionable unblock; C11 additionally names the condition under which it should be struck without measurement.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 11 (raw)

Verdict: REVISE - 0 Critical / 1 High / 2 Important. NO DISPOSITION CHANGED.
All 18 [M] claims re-derived exactly; C01 arm table exact to <1e-9 on 28 cells.

## What I verified and cleared

**Independent re-derivation (from scratch).** All 18 `[M]` census claims reproduce exactly: key-set `['id','label','text']` with no `title` field; whitespace-only `39`/`9` HateMM and `0` on all four MHC splits; `243`/`34` ZH tag rows; entity rows `64`/`9` MHC-EN, `1` strict / `2` hex-inclusive MHC-ZH; tag histogram `em 254 / /em 254` and nothing else; hate rates `0.5802` (141/243) vs `0.1161` (39/336), base `0.3109`, val `0.5882` (20/34) vs `0.1818` (8/44), base `0.3590`; `49`/`254` train and `50`/`288` train+val with identical top-5; markup fraction median `0.0000`, max `0.8621`, `203/579` above 10 %, p90 `0.5051`/`0.5071`/`0.5155`; `0.2604` median among markup rows; HateMM train median `694.5` linear / `696` upper. **The round-1 stress test reproduces ONLY under the train-only keyword set: `10/140 = 0.0714` (train+val keywords give `10/146`)** — so the record's figure is right and its scoping is the one that works. Entity triples `43/16/17` (train) and `51/22/18` (train+val) exact. Title presence `891/891`, `897/897`; join-scoped `629/629`, `657/657`; `<em>` in `391` Title / `0` Transcript whole-file, `277 / 0` join-scoped (= 243+34). Title medians measured directly: **MHC-EN 51 / transcript 322; MHC-ZH 27 raw, 13 stripped / transcript 78**. Cache inventory `100/130/71/6/2`; `data/Archive/` = `MHC`, `MHC_zh` only. **No fabricated number found.**

**Strike fidelity.** C14 is the only strike and it is faithful: `eligible_for_primary_target: false` and the dedup-boundary sentence verbatim, reinforced by `hard_constraints[4]`, scoped to the performance backlog with the diagnostic role preserved, and TVB's support correctly disowned as a **prediction**.

**All six downgrades justified**, re-derived from primary text: F82's ban_scope splits vote-side from head-side and ends *"; HateMM out of scope (no Offensive class)"*; C08's premise 1 genuinely refuted; **C10** — EUM's ban_scope literally reads *"Any revival carries THREE preconditions"*, a conditional closure self-hedged *"as of this recon"*, so HOLD is faithful and a strike would over-read; **C11** — disjunctive claim, `ERRPAT §5.2` real, pooled-only limitation carried; **C12** — F55's own detail names the three levels as *encoder-composition* levels, independently corroborated by `REDTEAM_BAN_SCOPE_AUDIT.md:230`'s "INDUCTIVE LEAP" ruling.

**Kind of record.** All ten registry statuses match across surfaces; the reversibility string is identical on all ten; `dispositions` sums to 10 with the gate separate; `ordered_backlog`, `hard_constraints`, `unified_pilot_gate`, `serial_execution` unaltered; C01-C04 carry no `gate0_reopen` key; the prereg is a draft. **No hold is over-cautious.**

**Spot checks that could have broken the record and did not:** C01 arm table recomputed cell-for-cell from stored confusions (**all 14 arms x 2 datasets to `<1e-9`**, full rotation spreads exact); `head_deployed_acc` identical to C02's `ARENA2.pooled_native_acc` at 4 dp on 6/6; `K_FOLDS = 5` + deployed head gives 36 heads; `torch.save` no-op verified; the ro-cache grid is diagonal exactly as the record's correction says; `banned_constraints[10]`'s `22.3/17.4/16.5` scales to `37.2/29.0/27.5`; all ten raw reviews appended.

## High

**H-1 · `TARGET_FINDINGS.md` — round 10's I-3 repair did not land on this surface, which still asserts the characterization the record explicitly withdrew.** The bullet read: *"the two supporting bans do not reach C06's object (F80's object is prompt language and **its prohibition is conditional on "without new mechanism"**; F70's object is individual readout cells)"*. Primary text: *"extraction-instruction language variations (any language, any stream, either encoder arm) on MHC_zh; prompt-language axis measured null-to-negative; do NOT re-propose prompt-language matching **elsewhere** without new mechanism…"* — the on-MHC_zh closure is **unconditional**; only the *elsewhere* re-proposal carries the condition. This makes `TARGET_FINDINGS.md` the one surface still stating a ban is conditional when its operative half is not, in a record whose methodological finding is that authorities must be read at their own scope. Fourth consecutive recurrence of the lag class. **Repair:** replace with the full-scope statement the other three surfaces carry; no disposition changes.

## Important

**I-1 · "markup-stripped" is an inference recorded as a property of a primary measurement, and the record's own named intermediary says the opposite.** The C08 unblock states the superseded figure *"is an MHC-ZH, test-split, **markup-stripped** median inherited second-hand via F88 ledger correction (c) from `ERRPAT_MHC-ZH_2026-07-26.md:272`"*. `ERRPAT:271-272` says only *"Deployed ZH text = `Title + " . " + Transcript` … Medians on test: title 15 chars, transcript 76 chars, composed text 96 chars"* — **no stripping qualifier appears anywhere in that document**. F88's ledger correction (c), the intermediary the record names, describes the title as *"the harvested TITLE (**carrying** the `<em class="keyword">` search-term markup, medians: title 15 chars…)"* — i.e. as markup-bearing. My own measurement makes the label *plausible* but not established: ZH title median is 27 raw / 13 stripped, so 15 sits near the stripped value, and 15/96 ≈ 15.6 % tracks the stripped ratio (13/92 ≈ 14 %) rather than the raw (28/104 ≈ 27 %). That is a reconciliation argument, not a reading of either source. Nothing depends on it — the load-bearing claim (EN title median **51**, `3.4x`, **unpriced**) is a direct measurement. **Repair:** mark the attribute as inferred on all four surfaces, or drop it.

**I-2 · "F112 carries the same caveat independently" is half true.** §3.4 quotes F113's two-clause honesty clause verbatim and then asserts F112 carries **the same** caveat. `findings.jsonl` F112 carries only the first clause: *"LIMITATION: raw-space, train-side screen; a raw-space null does not entail a head-space null."* There is no TEST-transfer clause in F112. The clause the record actually leans on (raw ≠ head) *is* independently in F112, so the C06 gating rationale is unaffected — but the sentence claims corroboration F112 does not supply, in the record's own corrections layer. **Repair:** narrow it.

## Not raised, on the round's scoping rules

The `:270-271` anchors inside the §§11/§17 review-history log are historical record, not live claims; the §5.2/§5.3 anchors are off by one to three lines with no change to what is asserted; `MECHNOV_PAIRVERIFY_PREGATE.md:458-459` is disclosed by the record itself as F99's own pointer with the true location given; "fourteen lines later" in §3.3 is really 27 lines in the recon, which changes nothing the record asserts.


================================================================================
C04-A0T-SMALL-v1 impl-v8 TEACHER-OUTPUT RELIABILITY REVIEW -- raw
Appended 2026-08-01. Verdict: KILL_C04_TEACHER_SEMANTIC_RELIABILITY.
Instrument: the frozen five-rate taxonomy. Independent recomputation from the
800 raw teacher outputs AGREES with the sealed manifest with zero discrepancies.
Full file: refine-logs/C04_A0T_SMALL_V1_V8_TEACHER_OUTPUT_REVIEW.md
================================================================================
# C04-A0T-SMALL-v1 impl-v8 — Independent Teacher-Output Reliability Review

Date: 2026-08-01
Reviewer: fresh independent Opus reviewer; zero exposure to the implementation
reasoning and zero exposure to the code/resource and payload reviews.
Instrument: the **frozen** five-rate semantic-reliability taxonomy. The reviewer
was instructed that the taxonomy is the verdict instrument and that inventing
criteria, relaxing a threshold or substituting judgement for it is forbidden.
Subject: `artifacts/c04/a0t_small_v1_impl_v8/seal/` from GPU job `13857`
(COMPLETED, 0:0, 2668 s, 1xA100).

## VERDICT: `KILL_C04_TEACHER_SEMANTIC_RELIABILITY`

**Scope.** This kills the C04 teacher's *semantic reliability* under the frozen
v8 contract: Qwen2.5-VL-7B-Instruct, 8 frames + native transcript, two frozen
prompt forms A/B, greedy decoding, `max_new_tokens=256`, `max_pixels=151200`, on
the sealed 200+200 pre-label tranche. It does **not** claim anything about
accuracy, label correlation, downstream utility, or the C04 method (`FULL_Q4` and
its controls are untested). It does not claim a 7B VLM cannot ever produce
SPaSH-style roles — only that *this frozen prompt/model/decode/agreement
configuration* does not, by margins far outside any threshold quibble.
`labels_opened: false` holds; no label was read.

## Recomputed rate table (reviewer's own numbers vs the sealed manifest)

Frozen thresholds (`configs/c04/c04_a0t_small_v1_v8.json` -> `reliability`,
unchanged since the v6 freeze): usable >= 0.85, missing <= 0.10,
conflict <= 0.20, max_nonfallback_value_frequency <= 0.90, joint >= 0.60.

**HateMM** (n=200) — `prompt_parse_rate` 0.6525; `joint_all_four_usable` 0.2100
(**FAIL**, bar 0.60)

| slot | stable/single/conflict/missing | usable | missing | conflict | maxfreq | cell |
|---|---|---|---|---|---|---|
| S | 57/61/66/16 | 0.590 **F** | 0.080 P | 0.330 **F** | 0.5593 P | **FAIL** |
| P | 43/35/119/3 | 0.390 **F** | 0.015 P | 0.595 **F** | 0.0256 P | **FAIL** |
| T | 74/40/82/4 | 0.570 **F** | 0.020 P | 0.410 **F** | 0.3860 P | **FAIL** |
| H | 66/49/75/10 | 0.575 **F** | 0.050 P | 0.375 **F** | 0.3130 P | **FAIL** |

**MHC_zh** (n=200) — `prompt_parse_rate` 0.5425; `joint_all_four_usable` 0.2100
(**FAIL**)

| slot | stable/single/conflict/missing | usable | missing | conflict | maxfreq | cell |
|---|---|---|---|---|---|---|
| S | 32/56/89/23 | 0.440 **F** | 0.115 **F** | 0.445 **F** | 0.4886 P | **FAIL** |
| P | 29/52/95/24 | 0.405 **F** | 0.120 **F** | 0.475 **F** | 0.0123 P | **FAIL** |
| T | 46/47/98/9 | 0.465 **F** | 0.045 P | 0.490 **F** | 0.4624 P | **FAIL** |
| H | 44/45/100/11 | 0.445 **F** | 0.055 P | 0.500 **F** | 0.5056 P | **FAIL** |

**8/8 slot cells fail; 2/2 joint gates fail.** The nearest miss is HateMM-S usable
0.590 against 0.85 — **0.26 absolute short**. Nothing is borderline.

Slot mapping, per the frozen design (the review request stated it wrongly and the
reviewer corrected it): **S = source_relation, P = proposition,
T = presenter_stance, H = the ordered pair protected_target;harm_act.** The
recomputation follows the frozen mapping. This was an error in the request text,
not in the data or the code.

## Independent recomputation vs the sealed manifest: AGREES exactly, 0 discrepancies

The reviewer re-implemented the parse and the four-state slot logic from the
frozen design description rather than importing the project's module.

- 800/800 raw strings re-parsed: **0** disagreements across 7,200 compared fields.
- 1,600/1,600 slot states re-derived: **0** disagreements against `canonical_bank.jsonl`.
- All 20 rates and all 32 state counts reproduce bit-for-bit.
- P-slot cosines independently recomputed on CPU from the frozen snapshot's
  `model.embed_tokens.weight` (152064x3584 bf16) with the reviewer's own
  normalizer: 214 cosines, **max |delta| = 1.7e-05, 0 threshold flips**.
- Sealed file SHA-256s match `seal_manifest.sealed_output_hashes`.
- `build_slot_reliability` implements the `C04_REFINED_PROPOSAL_V2.md` four-state
  rule exactly, including the "any available valid form has confidence <3 ->
  conflict" clause.

## Mechanism: this is the teacher, not the harness

Three independent, additive failure modes, quoted verbatim from `raw_output`.

**(1) Verbatim schema echo / multi-value enum picks — the largest single defect.**
The teacher copies the prompt's pipe-alternation *as the value*:

> `{"source_relation":"current_presenter|quoted_or_embedded|performed_or_lyric|mixed|uncertain","proposition":"The current presenter is questioning the actions of the other participants.","presenter_stance":"reject_or_counter|report_or_describe|perform_without_clear_commitment|uncertain","protected_target":"race|ethnicity|religion|...","harm_act":...}` — `hate_video_334` form B

Attribution over 400 forms per dataset: HateMM 72 verbatim-echo + 62 pipe-subset
+ 11 invented values; MHC_zh 11 verbatim-echo + 40 pipe-subset + 0 invented.
Sub-cases like `"current_presenter|performed_or_lyric"` (15 HateMM / 22 MHC_zh)
are the model refusing to commit. This is a prompt-comprehension failure by the
teacher, unrecoverable without inventing a disambiguation rule.

**(2) The `confidence` map is simply omitted.** 33 HateMM + 52 MHC_zh forms drop
it entirely, invalidating all four slots at once:

> `{"source_relation":"performed_or_lyric","proposition":"The speaker describes a group of people who use various tactics to win arguments.","presenter_stance":"report_or_describe","protected_target":"uncertain","harm_act":"none"}` — `non_hate_video_450` form A

**(3) Genuine A/B semantic disagreement — the binding wall.** Among pairs where
*both* forms are fully valid, the two prompt forms disagree at 0.351/0.638/
0.391/0.374 (HateMM S/P/T/H) and 0.380/0.421/0.453/0.389 (MHC_zh). Example, both
forms fully valid, `hate_video_103`:

> A: `{"source_relation":"uncertain","proposition":"A train is moving along a track.","presenter_stance":"uncertain","protected_target":"race","harm_act":"attack","confidence":{"S":3,...}}`
> B: `{"source_relation":"quoted_or_embedded","proposition":"The train is moving along the tracks.","presenter_stance":"uncertain","protected_target":"uncertain","harm_act":"none","confidence":{"S":1,...}}`

Identical visual content, opposite harm reading, self-reported confidence 3 -> 1.
The P-cosine distribution is bimodal, not thresholded away: HateMM median 0.633,
23.9% at >= 0.999. Relaxing the 0.80 cosine bar to **0.00** — accepting *any* two
propositions as agreeing — still leaves P usable at 0.765 (HateMM) / 0.545
(MHC_zh), both below 0.85.

### Refuting "the harness mis-parsed a fine teacher" — three diagnostic-only counterfactuals

- **Lenient parser** (strip ```json fences, outermost-brace slice, truncation
  repair) recovers 106 of 800 forms and lifts `prompt_parse_rate` to 0.7100 /
  0.6750 — but **joint FALLS**: 0.21 -> 0.175 (HateMM) and 0.21 -> 0.135
  (MHC_zh), because recovered forms convert `single_valid` into `conflict`. Being
  more permissive makes it worse.
- **Drop the confidence >= 3 gate entirely**: best case usable 0.720-0.760, joint
  0.330/0.440. Still fails 0.85 and 0.60 everywhere.
- **Perfect-parser ceiling**: if every form parsed, `single_valid` is by
  construction 0 and usable == A/B agreement == 0.362-0.649. **Structurally below
  0.85 in all 8 cells.**

Truncation at `max_new_tokens=256` is a minor real contributor: 4 HateMM + 8
MHC_zh forms have unbalanced braces, on long hate-lyric transcripts or degenerate
repetition. It explains <2% of the loss.

## The MHC-ZH transcript-poverty confound: real, but NOT explanatory

Transcript lengths re-derived through the frozen label-skipping projection
(`label_value_materialized == 0`); all 400 reconstructed transcripts hash-match
`input.transcript_sha256`, 0 mismatches. MHC_zh p10 = 8, median 63.5, 24 items
< 10 scalars; HateMM p10 = 38, median 999, 5 items < 10.

A monotone length gradient exists (and exists on HateMM too, so it is not
ZH-specific):

| MHC_zh stratum | n | S usable | P usable | T usable | H usable | joint |
|---|---|---|---|---|---|---|
| < 10 scalars | 24 | 0.25 | 0.21 | 0.33 | 0.33 | 0.17 |
| 10 <= s < 64 | 76 | 0.36 | 0.32 | 0.37 | 0.39 | 0.18 |
| s >= 64 | 100 | 0.55 | 0.52 | 0.57 | 0.51 | 0.24 |

Exclusion counterfactuals: drop all 24 short items (n=176) -> best slot usable
0.483, joint 0.216. Keep only >= median (n=100) -> best 0.570, joint 0.240. Keep
only >= p75 = 171 scalars (n=51) -> best 0.667, joint 0.255. **No stratum, at any
cut, reaches 0.85 on any slot or 0.60 joint.** The confound moves the numbers by
~0.15-0.25 and the bar is 0.26-0.45 away. **The MHC-ZH kill survives deleting
every poor-transcript item**, so it is separable from the confound.

Adjacent finding: 78/314 parsable MHC-ZH propositions are English despite the
`chinese_proposition_max_64_unicode_scalars` contract, and 24 of the 39 MHC-ZH
proposition-bounds violations are English sentences overflowing a scalar cap
sized for Chinese. This inflates MHC-ZH `P.missing_rate` (0.120) somewhat, but P
usable is 0.405 — not load-bearing for the verdict.

`HateMM/hate_video_95` (frozen all-black pack, both forwards): the teacher
answered fluently from transcript alone. 2 of 800 forwards; no effect on any rate
at the third decimal.

## What this licenses

**Licensed:** recording `terminal_state = KILL_C04_TEACHER_SEMANTIC_RELIABILITY`
as a measured, independently reproduced result; closing the v8 teacher
configuration; citing the tranche as evidence that a 7B open VLM under a
strict-schema two-form self-consistency protocol yields 0.39-0.59 slot usability
and 0.21 four-way joint coverage on hateful-video train items. The seal is
scientifically clean — pre-label, hash-lineage intact, exactly reproducible.

**Not licensed:** opening labels
(`label_access_allowed_after_this_seal_only_if_reliability_passes: false` — the
gate did not pass); any accuracy, C1 or C2 claim; any statement about the SPaSH
tensor, its role maps, or `FULL_Q4` vs `CONCAT_ALL4_MLP` /
`RETAINED_INDEPENDENT4`; any re-derivation of a verdict from these same 800
forwards under a rewritten parser, relaxed threshold or retuned prompt
(`reliability.prompt_model_threshold_rewrite_after_failure: false` and
`no_retry_redraw_prompt_rewrite: true` forbid it, and the counterfactuals show it
would not help anyway).

**If C04 is to continue**, three changes each requiring a *new* pre-registered
tranche, not a re-scoring of this one: (a) constrained/grammar-forced decoding or
per-field single-token classification instead of free-form JSON — kills
mechanisms (1) and (2), worth roughly +0.13 to +0.18 usable; (b) a different or
larger teacher — mechanism (3), the 0.36-0.65 A/B agreement wall, is the only one
that matters and is untouched by any harness fix; (c) dropping the two-form
self-consistency requirement, which would eliminate the `conflict` state but also
eliminate the reliability instrument itself. Given (3) alone caps usable at 0.65
against a 0.85 bar, the reviewer would not fund (a) without first cheaply
measuring A/B agreement for a candidate replacement teacher.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 12 (raw)

Verdict: GO - 0 Critical / 0 High / 2 Important. FIRST GO.
All four scope items cleared explicitly; both Importants and all observations applied.

I verified the record against primary sources with four independent passes (three fact-check workers plus my own recomputation) and found the adjudication sound on all four scope items.

## What I verified independently

**Census / `[M]` layer — re-derived from scratch, 100% reproduction.** All six gt files (key-set `['id','label','text']`, single set); HateMM whitespace-only `39`/`9`, MHC `0/0/0/0`; ZH tags `243`/`34`, tag histogram `em 254 / /em 254` and nothing else; entities `64`/`9` MHC-EN with train-only `&#39;x43, &quot;x16, &amp;x17`; MHC-ZH `1` strict / `2` hex-inclusive; hate rates `0.5802 (141/243)` vs `0.1161 (39/336)` base `0.3109`, val `0.5882 (20/34)` vs `0.1818 (8/44)` base `0.3590`; keywords `49/254` train, `50/288` train+val, top-5 identical; markup fraction median `0.000000`, max `0.862069`, `203` rows >10%, p90 `0.505051` lower / `0.507133` linear / `0.515464` higher — **Note M-1's convention claim is correct**; the `10/140 = 0.0714` bare-keyword stress test reproduces **only** train-scoped (train+val gives `10/146`), as the record states. Medians: HateMM train `694.5`/`696`, MHC-EN train `369`, val `439.5`/`443`, ZH val `108.5`/`111` — **Note M-2's upper-convention claim is correct**.

**Title census.** Whole-file `891/891` and `897/897`; `<em>` in `391` Title / `0` Transcript. Join-scoped: `629/629`, `657/657`, `277 / 0`; **EN title median 51 (transcript 322), ZH 27 raw / 13 markup-stripped (transcript 78)** — exact. `scripts/prep_mhc.py:72-85` and `scripts/prep_video_dataset.py:126-139` do read `title` and `transcript` as separate variables. `LITSWEEP2_INPUT_FIDELITY.md:56` is gt-schema-scoped and §3.3's "re-scraping YouTube metadata" inference is indeed wrong for both MHC datasets. **Round 3's Critical stands.**

**§3.7 fold-head identity — recomputed.** Identical at 4 dp on **6/6**, exactly as claimed and at exactly the precision claimed.

**C01 arm table** recomputed from stored confusions across all 14 arms x 2 datasets (`<1e-12`); `n_dev` 107/78; six rotations; spreads `0.8505-0.8692` / `0.8462-0.8974`; **4 of 6** HateMM and **2 of 6** ZH rotations strictly below the primary arm — all exact.

**Scope item (a) — the one strike is faithful and in-scope.** C14's `eligible_for_primary_target: false`, its `dedup_boundary` and `hard_constraints[4]` are verbatim; the strike is confined to the performance backlog and preserves the diagnostic role; the TVB "7 of 7 at ~0" support is correctly identified as a **prediction** and explicitly not relied on.

**Scope item (b) — statuses are the correct kind.** All ten registry entries carry the byte-exact reversibility string; no candidate was run; `dispositions` sums to 10; historical `ordered_backlog` intact; `new_jobs` and `new_metrics` empty; C09's prereg is a `DRAFT`.

**Scope item (c) — no inference recorded as a measurement.** Every figure traced resolves to a primary source or a stated re-measurement. The inferential steps are all labelled: "markup-stripped" (inferred), TVB (predicted), C13's regression step (plausibility inference), C07's supervision-source enumeration (supporting inference, explicitly non-load-bearing), C10's "space is EMPTY" (extension), EUM's compressed `median 83%` (disclosed as a compression against the true `0.8289` / `0.7174` / `74.2%`), F98's superlative (conservative family-scoped wording with the two-registry disagreement recorded).

**Scope item (d) + the three headline downgrades.** All eight unblocks are concrete and proponent-actionable; **none is over-cautious**. C10 — EUM's ban is a *conditional* closure with three written revival preconditions, hedged "as of this recon" over a three-item enumeration excluding a rule-based gold-free MLLM-free boundary; BSY's block is textually scoped to "bank-ADDITION" and procedural. C11 — the claim is verbatim disjunctive; `ERRPAT §5.2` makes the second disjunct non-empty; the hold carries a written self-destruct clause. C12 — F55's ban_scope and detail confine "EN closed at all three levels" to the encoder-composition question. C07, C08 and C13 likewise.

## Findings

**I-1 · Live text still calls C07 struck.** §3.5 reads *"This strengthens rather than weakens the C07 strike"* and §3.6 *"the strike is recorded with both numbers"*; the JSON mirrors both in `V-5`/`V-6`. C07 is `held_lattice_delta_unwritten_reachability_unscreened` on all four surfaces, and §4.2 states the opposite conclusion explicitly. This is the residue of round 1's Critical never being swept out of §3, and it survives on the machine-readable surface. It changes no disposition, status or asserted fact — the LBOP facts are correct and are used correctly as C07's unblock (a). **Repair:** replace "the C07 strike" with "C07's first unblock condition" / "the hold is recorded with both numbers."

**I-2 · The C12 unblock re-asserts the attribution the record corrects 35 lines earlier.** The unblock reads *"stability-as-weight (lands on `[5]` **under EUM's gloss**, and is then dead)"*, while §4.2 leg 2 says *"it is a **stack, not a gloss**"* — EUM reaches "MLLM-derived boundaries or weights" via `"P3 / P11 plus banned_constraints[5] … and [6]"`, four authorities (verified verbatim). Round 10's I-2 was raised against this exact unblock phrasing and landed only in `gap_2`. The error runs conservative, so C12's disposition is unaffected. **Repair:** "lands on `[5]` under the EUM four-authority stack (P3/P11 + `[5]` + `[6]`)".

## Observations (not findings)

- `§54` — *"Findings are enumerated in §§8-13"*; rounds 7-11 are in §§14-18.
- `TARGET_LOOP.md` — the C12 unblock still ends *"(then F60 governs, subject to D7)"*, ten lines after the same paragraph correctly narrows it to the open **generator-role sub-ruling** (round-8 I-4 lag on one surface only).
- `TARGET_FINDINGS.md` — the attestation gives `629/629` and `657/657` but not `277 / 0`; internally consistent because that surface never makes the whole-file claim the figure join-scopes.
- `§96` labels `0.5051` "nearest-rank" while M-1 and §7 use numpy's `lower` (same value; round 2's I-6 asked for one vocabulary).
- `:417-419` carries a duplicated clause and `:380` begins lowercase — formatting residue from the round-4 and round-11 inserts.
- C07 `gap_2`'s F82 quote elides `(any monotone weighting, any tau)` behind the ellipsis; the elision makes the vote-side ban look *narrower* than written, i.e. runs against the record's own argument.
- On MHC-ZH `orthrot_83p8`'s `0.8974` ties `orthrot_72p7`; the table prints both rows, so "best of six" is transparent.
- C14's `prior_status` field holds a backlog position rather than the historical status string; the substitution is disclosed in `prior_status_note`.
- `generate_VideoMLLM_embedding_readout_HF.py` lives at `src/utils/`, not `scripts/`; the cited line range does contain the `ro_ow_L24` tuple, and the prompt/readout-span confound is real and confirmed at source.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 13 (raw)

Verdict: GO - 0 Critical / 0 High / 1 Important. SECOND GO.
All four scope items cleared; the Important and all 14 observations applied.

## What I cleared

**Scope item (a) — the one strike is faithful and in-scope.** C14 carries `eligible_for_primary_target: false` and the quoted `dedup_boundary` verbatim; `hard_constraints[4]` is verbatim. The strike is confined to the performance backlog and preserves the diagnostic role. TVB's "7 of 7 at ~0" support is correctly identified as a **prediction** (`LITSWEEP5_COMPLETENESS.md:120` contains the literal word "predicted") and explicitly not relied on; TVB's own "flagged-not-banned" is verbatim.

**Scope item (b) — statuses are the correct kind of record.** All ten `gate0_reopen_2026_07_31` blocks carry the **byte-identical** string `"registry-level; reversible by a future user ruling. NOT a measured kill."`. `dispositions` sums to exactly 10 (1 strike + 6 downgraded + 1 held + 1 gated + 1 promoted). C01-C04 carry no gate0 key. The historical `ordered_backlog` is intact. `new_jobs`/`new_metrics` empty; the prereg is a DRAFT.

**Scope item (c) — no inference recorded as a measurement.** I recomputed the whole `[M]` layer independently: key-set `['id','label','text']`; whitespace-only 39/9 and 0/0/0/0; ZH tags 243/34 with histogram `em 254 / /em 254` and nothing else; EN entities 64/9 with train-only 43/16/17; ZH strict 1 / hex-inclusive 2; hate rates 0.5802 (141/243) vs 0.1161 (39/336), base 0.3109, val 0.5882/0.1818/0.3590; bare-keyword 10/140 = 0.0714 (train-scoped only); keywords 49/254 train, 50/288 train+val; markup fraction median 0.000000, max 0.862069, 203 rows >10 %, p90 0.505051/0.507133/0.515464 — Note M-1's convention claim is exactly right; medians 106 / 108.5 / 694.5-696 / 369 / 439.5-443 — Note M-2 holds; markup-bearing median 0.2604; `CLIP_Embedding` 100/130/71/6/2; `Archive` = `MHC`, `MHC_zh` only. **100 % reproduction, no fabricated number.** The C01 arm table recomputes to `<1e-12` on all 14 arms x 2 datasets with spreads and the 4-of-6 / 2-of-6 counts exact. §3.7's fold-head identity is identical at 4 dp on 6/6 at exactly the claimed precision. Every inferential step is labelled.

**Scope item (d) + the three headline downgrades.** All eight unblocks are concrete and proponent-actionable; **none is over-cautious**. C10 — EUM's ban is a *conditional* closure supplying three written revival preconditions, with precondition (2) hedged "as of this recon" over a three-item enumeration excluding a rule-based gold-free MLLM-free boundary; BSY's block is procedural and scoped to bank-addition. C11 — the claim is verbatim disjunctive; `ERRPAT:301-306` makes the second disjunct non-empty; `:405` is verbatim and correctly **cluster-scoped** — the record in fact applies it **narrower** than ERRPAT's own broadest claim at `:415`. C12 — F55's ban_scope and detail confine "EN closed at all three levels" to encoder/feature-composition objects. C07, C08, C13 likewise.

**Round-12 application.** I-1 landed on all four places. Eight of nine observations landed.

## Finding

**I-1 · The C12 unblock in `TARGET_STATE.json` still attributes the broad reading of `banned_constraints[5]` to "EUM's gloss", and the round-12 block certifies the repair as applied.** The JSON's C12 `unblock` read `stability-as-weight (lands on [5] under EUM's gloss and is then dead)` — the exact string round 12 charged at I-2 and round 10 charged at I-2 before it. It contradicts the same object's `gap_2` three fields earlier (*"Broad-reading precedent exists but it is a STACK, NOT A GLOSS"*) and misstates the primary source: EUM reaches "MLLM-derived boundaries or **weights**" through four authorities, verbatim *"that is P3 / P11 plus banned_constraints[5] … and [6]"*. `banned_constraints[5]`'s own literal text reaches neither boundaries nor weights. The repair landed on the markdown record but not on the machine-readable surface — while the round-12 block states *"corrected"*. That certification is false as written. Nothing about C12's disposition changes: the error runs conservative, and the record says so. But in a record whose stated methodological finding is that authorities must be read at their own written scope, a live disposition field attributing a four-authority closure to a single ban's gloss is a misstatement of the evidence. **Repair:** correct the JSON string, and do not re-certify round 12's I-2 as applied until it is gone.

## Observations (not findings)

1. §54's "Findings are enumerated in §§8-18"; round 12 is §19.
2. `TARGET_LOOP.md` compresses the same unblock to "stability-as-weight (then `[5]` applies and it is dead)" — asserts `[5]` applies with no stack qualifier.
3. The JSON's `M-1` uses "nearest-rank / linear interpolation / upper-higher" while the other three surfaces use numpy's `lower`/`linear`/`higher`.
4. `banned_constraints[5]` is described as "four words"; hyphen-split it is five tokens. The substantive point is right.
5. `LITSWEEP5_COMPLETENESS.md` §4's table has eight rows (ranks 1-7 plus a parenthesized `(8)`); the record calls it a "seven-row priority table". The rank claim ("7 of 7") is exact.
6. **`banned_constraints[10]`'s `22.3 / 17.4 / 16.5` are train-arena net-item requirements** (`n = 744/579/549`, `LITSWEEP7_LANDING_SITE.md:107-111`). Neither the ban nor the record states the arena, so a proponent applying the recommended Gate-0 currency to a test-sized arena would mis-scale by ~3.5x. Inherited from the ban's own text, not introduced here.
7. `banned_constraints[2]` bans only "cross-seed ensembles" — which is why TVB can call multi-prompt "flagged-not-banned". The literal multi-prompt ban carrying C14 lives in `hard_constraints[4]`. Both ledgers are accurate in their own terms and the strike stands independently on C14's own eligibility flag.
8. C14 is still a member of the `ordered_backlog` array; "struck from the performance backlog" is a status-string change. The record discloses that the historical array is deliberately untouched, so the two are consistent.
9. §4.2 C12 cites archive-as-key `dAcc -0.0014 +/- 0.0313, zero vote flips` without dataset scope — those are MHC-ZH-only (5 ZH seeds); the EN arm is `-0.0062 +/- 0.0051` with 0-2 flips/seed. Inert, because the leg's argument is that the measurement is the wrong *object*.
10. **"C07 is a cone metric — a head-side/representation object" is the record's reading**; C07's registry entry contains no head-side language. Well-founded, and the record explicitly hedges the adjacent graded-auxiliary step, but it is the one unlabelled inferential step inside the C07 downgrade.
11. LBOP-0's gate is reported as ">= +0.050 on both datasets"; `TARGET_GATE0_ITER6_LITERATURE.md:284` also requires macro-F1, per-fold sign agreement and a Farkas/gradient-cone audit. The record **understates** the comparator's bar, against its own argument.
12. Global-R2's quoted epitaph "oracle@coverage 10x under bar" compresses arms of `+0.0044` and `+0.0277` against the `+0.040` bar; the `+0.0277` arm is 1.44x under. The compression predates the record and is quoted, not asserted.
13. The route-scoping sentence quoted for C01 has its exact home in the superseded `c01_a0_v1.json:7` / `v2.json:7`; the executed v4 config has no `negative_scope` key. The substance survives verbatim-in-substance at `TARGET_LOOP.md:649`.
14. `data/CLIP_Embedding` has a sixth directory `MHCsmoke` (0 entries) absent from the `100/130/71/6/2` row.

==============================================================================
## Gate-0 Reopen 2026-07-31 — Independent Review, Round 14 (raw) — CLOSING GO

Verdict: GO - 0 Critical / 0 High / 0 Important. THE TARGET VERDICT.
All four bar items cleared with zero findings at any severity. Reviewer on the
dispositions: "No hold should have been a strike, and no unblock is vacuous."
All fourteen observations applied anyway; five tighten the record against its
own arguments.

## Verdict

**GO — 0 Critical / 0 High / 0 Important**

No disposition should move; the one strike is faithful; every status is the correct kind of record; nothing measured is an inference; every unblock is usable and none is over-cautious.

## The four bar items, as cleared

**(a) The one strike is faithful and applied within its evidence's own scope.** C14 carries `eligible_for_primary_target: false` — the **only** entry in all fourteen carrying that field — and the `dedup_boundary` quote is byte-exact, as is `hard_constraints[4]`. The strike is confined to the performance backlog and preserves the diagnostic role. TVB's *"7 of 7 at ~0"* is correctly identified as a **prediction** (`LITSWEEP5_COMPLETENESS.md:120` contains the literal word "predicted"), TVB's own *"flagged-not-banned"* is verbatim, and the record explicitly does not rely on it.

**(b) Kind-of-record and reversibility.** All ten `gate0_reopen_2026_07_31` blocks carry the byte-identical string `"registry-level; reversible by a future user ruling. NOT a measured kill."`; `dispositions` sums to exactly 10 (1 strike + 6 downgraded + 1 held + 1 gated + 1 promoted); C01-C04 carry no gate0 key; `new_jobs`/`new_metrics` empty; the C09 prereg is a `DRAFT`. **The JSON's C12 `unblock` no longer says "under EUM's gloss"** — it now reads *"lands on `[5]` under the EUM FOUR-AUTHORITY STACK (P3 / P11 plus [5] and [6])"*, and round 12's certification is amended rather than left standing. Round 12's I-1 landed on all four places.

**(c) Nothing measured that isn't.** Census re-derived from scratch **twice, independently**, with no reference to any prior document: **100 % reproduction on all 18 `[M]` claims** — including the `10/140 = 0.0714` stress test reproducing **only** train-scoped (train+val gives `10/146`), and Note M-1/M-2's percentile and median conventions. The C01 arm table recomputed from the stored confusion matrices at **every one of 28 arm-cells** (`<1e-9`), with spreads `0.8505-0.8692` / `0.8462-0.8974` and the 4-of-6 / 2-of-6 counts exact. I recomputed §3.7 myself: C02's `gates.ARENA2.pooled_native_acc` against the banked `head_deployed_acc` — identical at 4 dp on **6/6**, exactly the precision claimed, with bit-equality correctly disclaimed. §6's AGGNET figures are verbatim, including the **conservative family-scoped** superlative. Every inferential step is labelled.

**(d) Unblocks usable; the three headline downgrades justified.** Re-derived from primary text: **C10** — EUM's precondition (2) enumerates exactly three illegal sources and concludes emptiness *"as of this recon"*; a rule-based gold-free MLLM-free boundary is outside that enumeration; BSY's block is textually scoped to *"bank-ADDITION"* and blocks a **prereg** pending a user ruling — procedural, not a scientific kill. **C11** — the claim is verbatim disjunctive; `ERRPAT_MHC-ZH:301-308` makes the second disjunct measured-positive (`p = 0.0048`, every figure exact); `:405` is genuinely a **cluster-scoped table row**, not a document verdict. **C12** — both `directions_tried.json` and `findings.jsonl` F55 confine *"EN closed at all three levels"* to the **encoder-composition** question. C05's `unwritten_as_posed` is the honest string; C07's registry boundary is conjunctive and its delta genuinely un-attempted; C08's premise 1 is refuted at source; C13's surviving basis is a proponent-satisfiable precondition and C13 carries no `eligible_for_primary_target` field. **No hold should have been a strike, and no unblock is vacuous.**

## Observations (none is a finding)

1. **A direct single-authority gloss of `banned_constraints[5]` exists and is not engaged.** F103/OCR's ban_scope glosses `[5]` directly, on an **archive field**: *"It is Qwen-2.5-VL GENERATED TEXT and falls under banned_constraints[5] (MLLM-scores-as-training-signal / the P4-P11 family boundary)."* It runs **conservative**, F60 conflicts with it head-on, and the C12 downgrade rests on gap_1/gap_3 — both verified — so the disposition is untouched. Worth adding, since it raises the burden on *both* branches of the fork.
2. **F108 is not named in C08's unblock.** F108 bans *"any change to WHICH STREAMS OR IN WHAT PROPORTION enter the RETRIEVAL KEY … CLOSED BY CONSTRUCTION, SO A RENAME CANNOT EVADE IT"*. C08 lands in F108's carve-out (ii) (content, not weight) **as written**, and the record's *"Nothing in hard_constraints or banned_constraints bans a title channel"* is literally true — F108 is a finding-level ban_scope. But a proponent realising unblock (a) as "expose the title as its own key block" walks into it.
3. **C06's six "random rotations" are angles on the same one-parameter family as the primary.** `c01_policy_contrast_a0.py:1272`'s `orthogonal_blocks()` is a Givens mixing of the two endpoint blocks; the code's own guards confirm theta=45deg **is** `common_displacement` (max abs diff `8.9e-08`-`1.2e-07`) and theta=0 **is** `endpoint_concat`. The record's *"a random direction with matched norm"* reads more diffuse than the object is — but this **sharpens** the adverse reading. Relatedly, the arm table omits two arms that also beat the primary — ZH `endpoint_concat` `0.8846`/+2 and HateMM `common` `0.8692`/+3, the decision block's own named strongest controls, with `gain_over_strongest_control` `-0.0256` / `-0.0094`, `pass: false`, `decision.continue = false`. Both omissions **understate** the record's own adverse case.
4. **Two round-13 observations certified applied landed on one surface only** — the "seven-row priority table" and "four words" descriptors. Both are wording/count descriptors on text either explicitly not relied on (TVB) or quoted verbatim adjacent (`[5]`), so neither misstates evidence.
5. **Round-13 observation 12 reached no surface.** Global-R2's quoted epitaph compresses arms of `+0.0044` and `+0.0277` against the `+0.040` bar; the second is 1.44x under. The compression is the source's and is quoted, and C05's leg rests on conditional information `<= 0` with both arms sub-bar regardless.
6. **C07's unblock (a) understates LBOP-0's bar at the point of use.** `:284` also requires macro-F1, per-fold sign agreement and a joint Farkas/gradient-cone audit. Recorded in §20 and the JSON but not where a proponent reads it — and it runs against the record's own argument.
7. **EUM's own status field records that a legal rule-based unit was already built and measured negative** — *"The best LEGAL evidence unit was already built (EXP_mm_segment_keys.md:195, final-epoch dF1 -0.0116, 3/3 seeds negative)"* — uniform K=4 windows with Whisper word-level timestamps, i.e. exactly the rule-based gold-free MLLM-free boundary C10's unblock (2) posits. EN-only and on consensus **vote** keys rather than the retrieval-bank object, so it does not close C10 — but it is an unpriced headwind.
8. **`ERRPAT_MHC-ZH:415` is the stronger headwind and is uncited** — but its section header scopes it to what is open *"in-box, at `$0`"*, and C11 is a training-time representation change. Round 13 already noted the record applies `:405` **narrower** than ERRPAT's broadest claim, i.e. against its own interest.
9. **The `LITSWEEP2` title error has a second uncaught instance** at `LITSWEEP3_ZH_SPECIFIC.md:39-40`; the record names only the LITSWEEP2 instance.
10. Minor: the `ro_*` caches are files, not directories; F88 null (3)'s "does not beat random deletion" is a val-sel loss and a final-epoch win, all under half a test item per seed, so "indistinguishable" is the exact reading; C14 remains a member of the historical `ordered_backlog` array as do C05-C13, disclosed deliberately; the AGGNET epitaph is lightly re-cast inside quotation marks, semantically identical.

---

*Read-only. Zero GPU, SLURM, Modal, model loads or `.pt` opens; no test-split file was opened by me or any worker; nothing was written; the C04 lineage was not touched and nothing in C02 was modified.*

==============================================================================
## C09 Stage-0 (A0) DRAFT — Independent Design Review, Round 1 (raw)

Verdict: REVISE - 4 Critical / 8 High / 10 Important. DRAFT NOT READY TO FREEZE.
One round of review was the authorized scope; repairs are NOT attempted.

## VERDICT: **REVISE** — 4 Critical / 8 High / 10 Important

The design is cheap, well-scoped on legality, and the arena/instrument choice is right. But its central instrument (`D-FELDMAN`) as specified **cannot decide the question it is built to decide**, and one of its frozen features is a direct read of the scored item's own gold label. Both defects push in the same direction — toward a false CONTINUE. Every finding is repairable at zero cost.

**Verified as sound** (so it is not re-litigated): both legality quotes are exact at the cited lines (`progress.json:25`, `LITSWEEP3_DATA_CENTRIC.md:82`); the F113 fold-head floors are exactly the banked values; `headspace_mint.py:274-281` is precisely the `torch.save` no-op; `n=744/579` are the true train sizes and the true pooled query counts; MHC-EN genuinely has no instrument; the F98 epitaph and its delivered figures are faithful; C02's A0 did reproduce `pooled_native_acc`.

## CRITICAL

**C-1 — §5.2 feeds the scored item's own gold label into `D-FELDMAN`.** The feature list opens with *"top-20 purity"* and then asserts *"none reads the scored item's own label"* — but §4.4 defines purity as *"fraction of `i`'s top-20 carrying **`i`'s gold label**"*. The leak is nearly the target itself: `ERRPAT_HateMM:143-145` measures gold-purity `< 0.5` for **24-27 of 26-28** errors in every cell. AUC would land near 1.0 by construction and the run returns CONTINUE regardless of the science. **Repair:** delete gold-purity; if a purity-like feature is wanted define it against the **predicted** class; add a `GATE-BLIND` validity gate enumerating every feature, naming the arrays it indexes, and asserting the query-side label array is never indexed at `i`, emitted as per-feature integer counts.

**C-2 — `D-FELDMAN` is not decidable: H-MEMORISATION does not predict AUC ~ 0.5.** Feldman's long-tail singletons are *by definition* the low-density, no-analogue, weak-margin items — exactly what the feature set measures. The separation is already measured: `ERRPAT_HateMM:130` gives median `|vote|` **0.7267 for errors vs 0.9873 for always-correct**. A label-blind feature set will separate at AUC ~ 0.85-0.95 under *either* hypothesis, so `K-FELDMAN` essentially cannot fire and `K-NET` carries the entire decision. §6's table has **no row** for the realistic outcome (AUC ~ 0.8, net far under bar) — precisely the AGGNET/F98 pattern the draft opens with. **Repair:** make `D-FELDMAN` **conditional and incremental** — negative class = `CONFIG-MATCHED-CORRECT`; primary statistic `dAUC = AUC(full) - AUC(configuration-only baseline)` with the CI on the difference; threshold pre-declared as the **conversion-equivalent** AUC, not 0.5; §6 restated three-valued, with the middle band a KILL under the F98 epitaph and explicitly **not** a Feldman confirmation. State that conditioning on a gold-defined stratification makes this an **upper bound** on identifiability.

**C-3 — cross-seed item leakage: the target is per-item, the rows are per (item, seed).** Stability is the 3-seed intersection, a property of the *item*; features are per-seed; `GATE-NESTED` asserts disjointness only from the arena fold, and results are pooled over seeds. Fitting on an item's seed-1 row and scoring its seed-0 row leaks the target exactly. **Repair:** group the nested split by **item id across both seeds and arena folds**; make the bootstrap resampling unit the item, not the (item, seed) row — otherwise the CI is anti-conservatively narrow by ~sqrt(3).

**C-4 — `NET`'s break accounting is non-conservative and mis-scoped.** Selected items in neither class — unstable errors, and correct items outside the matched stratum — are **uncosted**, inflating the promoting quantity. And the currency is mis-scoped: `22.3/17.4/16.5` is an accuracy delta over the **full** arena. **Repair:** `net = |{selected AND currently wrong}| - |{selected AND currently correct}|` over all `n` query items, with `CONFIG-MATCHED-CORRECT` retained as reporting stratification only; add a self-test asserting `net == n * dacc`.

## HIGH

**H-1 — `SHUFFLE-POP` is blind to the two leaks it is offered against.** Permuting the target destroys the feature-target association, so an estimator leaking through C-1 or C-3 **passes** cleanly. Also self-contradictory: a plain permutation does not preserve configuration marginals. Pin which permutation, and add `GATE-BLIND` as the actual leak detector.

**H-2 — undefined outcome in the decision rule.** `K-FELDMAN` fires only if the bar fails on **both** datasets; CONTINUE requires all three rules on **both**. Clear on HateMM, fail on ZH, everything else clearing → **neither** KILL nor CONTINUE. **Repair:** make every K-rule dataset-conjunctive in the same direction so KILL and CONTINUE partition the outcome space.

**H-3 — the vote and confidence definitions are wrong, and the sensitivity ladder is vacuous.** `mechfix_ops.py:91-95` is a **signed-cosine x rank-weighted** vote **already divided by sum(w)**, so `c_i = |score|/sum(w)` double-normalises and the declared ladder selects the empty set; on the correct scale it selects everything. Either way *"high-confidence"* is never tested. **Repair:** define `score` by literal reference to `mechfix_ops.py:94` and freeze tau as **quantiles of the in-run OOF `|vote|` distribution among stable inversions**.

**H-4 — §4.3's monotonicity argument is advertised far beyond where it holds.** It is arithmetic for `O1` only. Raising tau redefines the target and both AUC and precision can rise on a purer subpopulation — and `tau = 0` is **not the registered population**, which is *"high-confidence"* inversions. **Repair:** make the registered high-confidence point a co-primary for `K-FELDMAN`/`K-NET`, or restrict §9's KILL scope to `tau = 0` in terms.

**H-5 — the net-item bar is contested on three surfaces and the draft silently picks the hardest.** `unified_pilot_gate.stage_0_reachability` ties the net requirement to the **+0.030 final bar** (22.3/17.4); C02's own A0 ran `net_fix_rate: 0.03`; the C09 registry `bar` field says 37.2/29.0/27.5. **Repair:** adjudicate explicitly, report both currencies, name which governs — and discharge the R13 instruction to state the **train arena** whenever the figures are quoted.

**H-6 — the Holm family is named but no test exists, and there is no macro-F1 currency.** No test statistic, p-value source, resampling unit, `B` or `alpha`; *"2 metrics"* undefined for a net count; the Stage-0 bar has a macro-F1 leg the net currency does not price; *"paired bootstrap"* for a single AUC is not a paired quantity; and `K-FELDMAN` never says which estimator it reads — a live forking path.

**H-7 — F47's ban_scope literally covers `D-FELDMAN`'s feature family.** *"Decision-level meta-features (vote margins, purity, sub-votes, confidence differential, transcript stats) carry NO per-item routing signal, GBM or linear. Do NOT re-propose per-item selectors over frozen channels regardless of feature family or nonlinearity unless the selector input is a genuinely NEW information source not derivable from banked features/votes."* The draft cites F47 as a *protocol model* without registering its measured **null** as the prior or adjudicating the ban. **Repair:** add a fourth §3 boundary adjudicating `D-FELDMAN` against F47's text — the defensible distinction is the **target** and the fact that a Stage-0 probe is never deployed — and set §6's prior from F47's null. Carry the symmetric correction: F47's train-supervised leg rests on the F114-retracted *"CLIP LOO 0.998"* premise, which weakens the ban in C09's favour.

**H-8 — kill-risk (i) is never addressed, and `GATE-FID` names a gate that already exists as something else.** (a) The reopen's first quoted kill-risk (F75/NCA) appears **nowhere** in the draft, and it is a Stage-**0** problem: `H-L1` bans any query-time locator, so the only legal Stage-1 realisation is a train-label metric pull during encoder training — i.e. the F75 object the dedup boundary says C09 is *not*. The A0 would prove the existence of a target it has pre-banned itself from addressing. (b) `headspace_fidelity.py` already owns the name `GATE-FID` for a different check (the CPU-minted `fold == -1` head against the banked GPU floor's dev curve). Rename the draft's gate, and **also run the real one** — the six `fold == -1` heads are already inside the 36-head budget.

## IMPORTANT

**I-1** F88/ERRPAT provenance is a **test-split, deployed-head** measurement presented as a measurement of this arena; restate the stability rate as a transferred expectation. The configuration stratum is also frozen on test-derived buckets — say so. **I-2** `GATE-FID` pins only accuracy floors; the bar has a macro-F1 leg — pin the banked macro-F1s. **I-3** the two mandatory pregate clauses (`PREGATE_DETERMINISM_CLAUSE.md` DET-1..4, `PREGATE_CALIBRATION_CLAUSE.md` CAL-1..4) are not cited though `pregate_conventions` makes them mandatory; the fold contract is unpinned. **I-4** head-count arithmetic is wrong (30 total, not per sweep; 36 = 30 + six fidelity heads) and the wall-clock is optimistic — declare a resume path. **I-5** `GATE-NULL` is internally contradictory: with-null and remove-null cannot agree on every metric because `n` moves. **I-6** several declared features are measured to have **no dynamic range** in this space (cosine saturated at ~0.9999). **I-7** the raw arena is asserted but never specified. **I-8** estimator hyperparameters, bootstrap `B`, `alpha` and RNG seed are unfrozen. **I-9** §9 misses two scope items: `O1` is a **label-flip** oracle, not the registry's representation-level oracle; and with MHC-EN out of scope the two-dataset requirement has **zero slack**. **I-10** two carried corrections are missing from §3 — the LITSWEEP5 counter-text's *"downgraded, not vacated"* status, and round 14's *"indistinguishable"* reading of F88 null (3). Both omissions run **against** C09.

## Answers to the draft's six §10 open items

**1. Label-blind? NO.** The bank-side features are fine and are *not* the problem — *"first neighbour whose bank label differs from the majority"* is non-circular independently of the nested split, because the arena's folds are item-disjoint. The actual leaks are gold-purity (C-1) and cross-seed pooling (C-3), neither caught by `SHUFFLE-POP` or `GATE-NESTED`.

**2. `UNSTABLE-POP` on MHC-ZH? Do not declare it HateMM-only in advance** — that would import a test-split measurement into a train-arena design as a frozen scope decision. Declare a **data-independent power rule applied identically to both datasets**: `n_unstable < N_MIN` or CI width `> W` emits `CONTROL_UNDERPOWERED`. Once C-2's repair lands the control becomes genuinely informative.

**3. The scaling? Arithmetic and denominators right; premise contested.** `0.050 x 744 = 37.2`, `0.050 x 579 = 28.95`; `22.3/17.4/16.5` are exactly `0.030 x 744/579/549`; the denominators are the same population and the arena is right. But three surfaces disagree on which bar governs — see H-5.

**4. Six operating points? The count is not the problem; the absence of a test is.** Four of six cannot pass by arithmetic; the 95th percentile can at best tie the bar and only at precision 1.000. Use **three** points on the *selected-count* scale, `k in {|P|, 1.5|P|, 2|P|}`, each with a one-sided item-level bootstrap lower bound, Holm across 3 x 2 = 6, plus a declared macro-F1 currency.

**5. Monotonicity for `NET`? No — `O1` only.** Neither AUC nor precision is monotone in tau. Consequence §9 does not state: a KILL fired by `K-FELDMAN` or `K-NET` at `tau = 0` closes **only** `tau = 0`, which is not the population the registry claim registers.

**6. `GATE-FID` bit-exactness? Achievable at 4 dp, with three changes.** DET-1 is already hard-asserted in the instrument, so a thread mis-export fails fast. Required: (a) say **"equal at 4 decimal places"** — the banked anchor is a 4-dp value and asserting beyond it is the engineering-HALT trap that killed three C01 runs; (b) the residual risk is the **version/node** quartet — pin the interpreter/library versions and node from the banked `meta.runtime` and make a mismatch a documented HALT with a re-run path; (c) rename the gate and additionally run the real `headspace_fidelity.py` read on the six `fold == -1` heads.

## Bottom line

Legality holds and the three HALT boundaries are adequate for **Stage-0 as scoped** — nothing here becomes per-test-instance selection, because `D-FELDMAN` never scores a test item and `H-L1` forecloses query-time consultation. The live legality risk is at the **seam**: `H-L1` bans the only region-locator this A0 builds, so the CONTINUE it can produce has no legal successor unless the encoder-training route is named — and that route is F75's measured object.

Fix C-1 through C-4 and H-1 through H-4 and this becomes a genuinely decisive `$0` pregate: cheap, fast-failing, and — with the conditional/incremental discriminator — the first design in this channel that could actually separate a fixable topology defect from a memorisation-necessary error rather than re-measuring atypicality.



===== RAW REVIEW: refine-logs/C09_A0_PREREG_DRAFT_REVIEW_ROUND1.md =====

# C09 Stage-0 (A0) DRAFT — Independent Design Review, Round 1

**Reviewer.** Fresh independent worker, no exposure to the author's reasoning.
**Target.** `refine-logs/C09_A0_PREREG_DRAFT.md`
**Verdict.** `REVISE — 4 Critical / 8 High / 10 Important`
**Status.** The draft is **NOT ready to freeze or implement.** One round of review
was the authorized scope; the repairs are not attempted here.

---

## VERDICT: **REVISE** — 4 Critical / 8 High / 10 Important

The design is cheap, well-scoped on legality, and the arena/instrument choice is right. But its central instrument (`D-FELDMAN`) as specified **cannot decide the question it is built to decide**, and one of its frozen features is a direct read of the scored item's own gold label. Both defects push in the same direction — toward a false CONTINUE. Every finding is repairable at zero cost.

**Verified as sound** (so it is not re-litigated): both legality quotes are exact at the cited lines (`progress.json:25`, `LITSWEEP3_DATA_CENTRIC.md:82`); the F113 fold-head floors are exactly the banked values; `headspace_mint.py:274-281` is precisely the `torch.save` no-op; `n=744/579` are the true train sizes and the true pooled query counts; MHC-EN genuinely has no instrument; the F98 epitaph and its delivered figures are faithful; C02's A0 did reproduce `pooled_native_acc`.

## CRITICAL

**C-1 — §5.2 feeds the scored item's own gold label into `D-FELDMAN`.** The feature list opens with *"top-20 purity"* and then asserts *"none reads the scored item's own label"* — but §4.4 defines purity as *"fraction of `i`'s top-20 carrying **`i`'s gold label**"*. The leak is nearly the target itself: `ERRPAT_HateMM:143-145` measures gold-purity `< 0.5` for **24-27 of 26-28** errors in every cell. AUC would land near 1.0 by construction and the run returns CONTINUE regardless of the science. **Repair:** delete gold-purity; if a purity-like feature is wanted define it against the **predicted** class; add a `GATE-BLIND` validity gate enumerating every feature, naming the arrays it indexes, and asserting the query-side label array is never indexed at `i`, emitted as per-feature integer counts.

**C-2 — `D-FELDMAN` is not decidable: H-MEMORISATION does not predict AUC ~ 0.5.** Feldman's long-tail singletons are *by definition* the low-density, no-analogue, weak-margin items — exactly what the feature set measures. The separation is already measured: `ERRPAT_HateMM:130` gives median `|vote|` **0.7267 for errors vs 0.9873 for always-correct**. A label-blind feature set will separate at AUC ~ 0.85-0.95 under *either* hypothesis, so `K-FELDMAN` essentially cannot fire and `K-NET` carries the entire decision. §6's table has **no row** for the realistic outcome (AUC ~ 0.8, net far under bar) — precisely the AGGNET/F98 pattern the draft opens with. **Repair:** make `D-FELDMAN` **conditional and incremental** — negative class = `CONFIG-MATCHED-CORRECT`; primary statistic `dAUC = AUC(full) - AUC(configuration-only baseline)` with the CI on the difference; threshold pre-declared as the **conversion-equivalent** AUC, not 0.5; §6 restated three-valued, with the middle band a KILL under the F98 epitaph and explicitly **not** a Feldman confirmation. State that conditioning on a gold-defined stratification makes this an **upper bound** on identifiability.

**C-3 — cross-seed item leakage: the target is per-item, the rows are per (item, seed).** Stability is the 3-seed intersection, a property of the *item*; features are per-seed; `GATE-NESTED` asserts disjointness only from the arena fold, and results are pooled over seeds. Fitting on an item's seed-1 row and scoring its seed-0 row leaks the target exactly. **Repair:** group the nested split by **item id across both seeds and arena folds**; make the bootstrap resampling unit the item, not the (item, seed) row — otherwise the CI is anti-conservatively narrow by ~sqrt(3).

**C-4 — `NET`'s break accounting is non-conservative and mis-scoped.** Selected items in neither class — unstable errors, and correct items outside the matched stratum — are **uncosted**, inflating the promoting quantity. And the currency is mis-scoped: `22.3/17.4/16.5` is an accuracy delta over the **full** arena. **Repair:** `net = |{selected AND currently wrong}| - |{selected AND currently correct}|` over all `n` query items, with `CONFIG-MATCHED-CORRECT` retained as reporting stratification only; add a self-test asserting `net == n * dacc`.

## HIGH

**H-1 — `SHUFFLE-POP` is blind to the two leaks it is offered against.** Permuting the target destroys the feature-target association, so an estimator leaking through C-1 or C-3 **passes** cleanly. Also self-contradictory: a plain permutation does not preserve configuration marginals. Pin which permutation, and add `GATE-BLIND` as the actual leak detector.

**H-2 — undefined outcome in the decision rule.** `K-FELDMAN` fires only if the bar fails on **both** datasets; CONTINUE requires all three rules on **both**. Clear on HateMM, fail on ZH, everything else clearing → **neither** KILL nor CONTINUE. **Repair:** make every K-rule dataset-conjunctive in the same direction so KILL and CONTINUE partition the outcome space.

**H-3 — the vote and confidence definitions are wrong, and the sensitivity ladder is vacuous.** `mechfix_ops.py:91-95` is a **signed-cosine x rank-weighted** vote **already divided by sum(w)**, so `c_i = |score|/sum(w)` double-normalises and the declared ladder selects the empty set; on the correct scale it selects everything. Either way *"high-confidence"* is never tested. **Repair:** define `score` by literal reference to `mechfix_ops.py:94` and freeze tau as **quantiles of the in-run OOF `|vote|` distribution among stable inversions**.

**H-4 — §4.3's monotonicity argument is advertised far beyond where it holds.** It is arithmetic for `O1` only. Raising tau redefines the target and both AUC and precision can rise on a purer subpopulation — and `tau = 0` is **not the registered population**, which is *"high-confidence"* inversions. **Repair:** make the registered high-confidence point a co-primary for `K-FELDMAN`/`K-NET`, or restrict §9's KILL scope to `tau = 0` in terms.

**H-5 — the net-item bar is contested on three surfaces and the draft silently picks the hardest.** `unified_pilot_gate.stage_0_reachability` ties the net requirement to the **+0.030 final bar** (22.3/17.4); C02's own A0 ran `net_fix_rate: 0.03`; the C09 registry `bar` field says 37.2/29.0/27.5. **Repair:** adjudicate explicitly, report both currencies, name which governs — and discharge the R13 instruction to state the **train arena** whenever the figures are quoted.

**H-6 — the Holm family is named but no test exists, and there is no macro-F1 currency.** No test statistic, p-value source, resampling unit, `B` or `alpha`; *"2 metrics"* undefined for a net count; the Stage-0 bar has a macro-F1 leg the net currency does not price; *"paired bootstrap"* for a single AUC is not a paired quantity; and `K-FELDMAN` never says which estimator it reads — a live forking path.

**H-7 — F47's ban_scope literally covers `D-FELDMAN`'s feature family.** *"Decision-level meta-features (vote margins, purity, sub-votes, confidence differential, transcript stats) carry NO per-item routing signal, GBM or linear. Do NOT re-propose per-item selectors over frozen channels regardless of feature family or nonlinearity unless the selector input is a genuinely NEW information source not derivable from banked features/votes."* The draft cites F47 as a *protocol model* without registering its measured **null** as the prior or adjudicating the ban. **Repair:** add a fourth §3 boundary adjudicating `D-FELDMAN` against F47's text — the defensible distinction is the **target** and the fact that a Stage-0 probe is never deployed — and set §6's prior from F47's null. Carry the symmetric correction: F47's train-supervised leg rests on the F114-retracted *"CLIP LOO 0.998"* premise, which weakens the ban in C09's favour.

**H-8 — kill-risk (i) is never addressed, and `GATE-FID` names a gate that already exists as something else.** (a) The reopen's first quoted kill-risk (F75/NCA) appears **nowhere** in the draft, and it is a Stage-**0** problem: `H-L1` bans any query-time locator, so the only legal Stage-1 realisation is a train-label metric pull during encoder training — i.e. the F75 object the dedup boundary says C09 is *not*. The A0 would prove the existence of a target it has pre-banned itself from addressing. (b) `headspace_fidelity.py` already owns the name `GATE-FID` for a different check (the CPU-minted `fold == -1` head against the banked GPU floor's dev curve). Rename the draft's gate, and **also run the real one** — the six `fold == -1` heads are already inside the 36-head budget.

## IMPORTANT

**I-1** F88/ERRPAT provenance is a **test-split, deployed-head** measurement presented as a measurement of this arena; restate the stability rate as a transferred expectation. The configuration stratum is also frozen on test-derived buckets — say so. **I-2** `GATE-FID` pins only accuracy floors; the bar has a macro-F1 leg — pin the banked macro-F1s. **I-3** the two mandatory pregate clauses (`PREGATE_DETERMINISM_CLAUSE.md` DET-1..4, `PREGATE_CALIBRATION_CLAUSE.md` CAL-1..4) are not cited though `pregate_conventions` makes them mandatory; the fold contract is unpinned. **I-4** head-count arithmetic is wrong (30 total, not per sweep; 36 = 30 + six fidelity heads) and the wall-clock is optimistic — declare a resume path. **I-5** `GATE-NULL` is internally contradictory: with-null and remove-null cannot agree on every metric because `n` moves. **I-6** several declared features are measured to have **no dynamic range** in this space (cosine saturated at ~0.9999). **I-7** the raw arena is asserted but never specified. **I-8** estimator hyperparameters, bootstrap `B`, `alpha` and RNG seed are unfrozen. **I-9** §9 misses two scope items: `O1` is a **label-flip** oracle, not the registry's representation-level oracle; and with MHC-EN out of scope the two-dataset requirement has **zero slack**. **I-10** two carried corrections are missing from §3 — the LITSWEEP5 counter-text's *"downgraded, not vacated"* status, and round 14's *"indistinguishable"* reading of F88 null (3). Both omissions run **against** C09.

## Answers to the draft's six §10 open items

**1. Label-blind? NO.** The bank-side features are fine and are *not* the problem — *"first neighbour whose bank label differs from the majority"* is non-circular independently of the nested split, because the arena's folds are item-disjoint. The actual leaks are gold-purity (C-1) and cross-seed pooling (C-3), neither caught by `SHUFFLE-POP` or `GATE-NESTED`.

**2. `UNSTABLE-POP` on MHC-ZH? Do not declare it HateMM-only in advance** — that would import a test-split measurement into a train-arena design as a frozen scope decision. Declare a **data-independent power rule applied identically to both datasets**: `n_unstable < N_MIN` or CI width `> W` emits `CONTROL_UNDERPOWERED`. Once C-2's repair lands the control becomes genuinely informative.

**3. The scaling? Arithmetic and denominators right; premise contested.** `0.050 x 744 = 37.2`, `0.050 x 579 = 28.95`; `22.3/17.4/16.5` are exactly `0.030 x 744/579/549`; the denominators are the same population and the arena is right. But three surfaces disagree on which bar governs — see H-5.

**4. Six operating points? The count is not the problem; the absence of a test is.** Four of six cannot pass by arithmetic; the 95th percentile can at best tie the bar and only at precision 1.000. Use **three** points on the *selected-count* scale, `k in {|P|, 1.5|P|, 2|P|}`, each with a one-sided item-level bootstrap lower bound, Holm across 3 x 2 = 6, plus a declared macro-F1 currency.

**5. Monotonicity for `NET`? No — `O1` only.** Neither AUC nor precision is monotone in tau. Consequence §9 does not state: a KILL fired by `K-FELDMAN` or `K-NET` at `tau = 0` closes **only** `tau = 0`, which is not the population the registry claim registers.

**6. `GATE-FID` bit-exactness? Achievable at 4 dp, with three changes.** DET-1 is already hard-asserted in the instrument, so a thread mis-export fails fast. Required: (a) say **"equal at 4 decimal places"** — the banked anchor is a 4-dp value and asserting beyond it is the engineering-HALT trap that killed three C01 runs; (b) the residual risk is the **version/node** quartet — pin the interpreter/library versions and node from the banked `meta.runtime` and make a mismatch a documented HALT with a re-run path; (c) rename the gate and additionally run the real `headspace_fidelity.py` read on the six `fold == -1` heads.

## Bottom line

Legality holds and the three HALT boundaries are adequate for **Stage-0 as scoped** — nothing here becomes per-test-instance selection, because `D-FELDMAN` never scores a test item and `H-L1` forecloses query-time consultation. The live legality risk is at the **seam**: `H-L1` bans the only region-locator this A0 builds, so the CONTINUE it can produce has no legal successor unless the encoder-training route is named — and that route is F75's measured object.

Fix C-1 through C-4 and H-1 through H-4 and this becomes a genuinely decisive `$0` pregate: cheap, fast-failing, and — with the conditional/incremental discriminator — the first design in this channel that could actually separate a fixable topology defect from a memorisation-necessary error rather than re-measuring atypicality.


===== RAW REVIEW: refine-logs/C09_A0_V2_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v2 — Independent Design Review, Round 2

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V2_RECORD.md` (v2, claiming repair of all 22 round-1 findings).
**Verdict.** `REVISE — 1 Critical / 8 High / 10 Important`

---

v2 is a large, genuine improvement on v1. I re-opened every cited file and re-derived every number; the ledger is substantially honest and most of the 22 repairs really landed. But the C-1 repair — the one the whole design rests on — is defeated by v2's own feature 11, and the discriminator's inferential rule (`K-FELDMAN`) is not computable as written. Everything below is repairable at zero cost.

---

## Verified as sound (do not re-litigate)

**Instrument citations are exact to the line.** `mechfix_ops.py:94` is literally `votes = ((lab * 2 - 1) * sim * w).sum(1) / w.sum()`; `:95` is `(votes >= 0).astype(int)`; `_rank_weights(20)` is `[20…1]`, `Σw = 210`; `deployed_vote` returns `(votes, preds, I, sim)`. H-3's double-normalisation repair is correct and the vote scale is now right. `headspace_mint.py:106-116` (torch.load `test_seen`//`test` guard), `:192-194` (skip-if-exists), `:203-216` (StratifiedKFold parity assert against `vsw_ckpt/<ds>/f*.npz` `ho_idx`), `:274-281` (`torch.save` no-op) are all exact to the line — four for four. `mechnov_pairverify.K_FOLDS = 5`, `FOLD_SEED = 0`, `StratifiedKFold(shuffle=True, random_state=0)` ✓. `headspace_mint.CLI` has only `{hatemm, zh}`, so MHC-EN's out-of-scope claim is stronger than stated (minting it would require editing a frozen-sha artifact under `frozen_artifact_policy`).

**Every number re-derived.** All twelve `GATE-FLOOR` floors are exact against `result.acc_deployed`/`result.mF1_deployed` (HateMM `0.8884/0.8858/0.8858`, `0.8838/0.8811/0.8812`; ZH `0.8929/0.8895/0.8946`, `0.8747/0.8710/0.8765`). Fold sizes `149×4+148 = 744` and `116×4+115 = 579` are exact `per_fold.n_ho_items`. `posrate_bank 0.4005/0.3109` and the majority rates `0.5995/0.6891` reproduce from the caches directly. `raw_deployed_acc 0.8441/0.8480` ✓, and the raw space really is `l2n(concat(l2n(img), l2n(txt)))`, 7168-d, seed-free (`headspace_arena.py:7`, `mechnov_pairverify.py:124`). `B_fid_abs_3seedmean 0.0093/0.0086`, `STOP_RULE_TRIGGERED: false` ✓. `22.3/17.4 = 0.030 × 744/579`; `37.2/29.0 = 0.050 × 744/579`; `π* = (1+bar/k)/2` is the correct inversion of `net = k(2π−1)`; `net_s = 2|S∩wrong_s| − k ≡ n·Δacc_s` is an identity, so `GATE-SELFTEST` is well-posed; the `√3` bootstrap-narrowness claim is the right order.

**`GATE-NULL`'s re-measurement is true.** I re-ran it independently: HateMM zero-img `[355]`, zero-txt `[355]`, id `hate_video_95`, label `1`; MHC-ZH `[]`/`[]`. And clause 4 is correct in mechanism — `img_proj`/`text_proj` are `nn.Linear` with bias and `embed = self.mlp[:-2](x)` applies no final normalisation, so the zero row is genuinely non-zero in head space and "the item's own L2 norm before normalisation" is a non-degenerate feature.

**Monotonicity is sounder than §4.3 defends.** Flipping any error weakly raises *both* per-class F1s, so `ΔmF1_{O1}` is monotone in the flipped set too — `K-REACH`'s "closes every `τ ≥ 0` by arithmetic" holds for both legs, not just accuracy.

**C-3 and C-4 are genuinely repaired.** The arena fold assignment is a function of the label vector alone, so it is identical across head seeds and grouping the nested split by fold does group all three seed-rows of an item; the cross-seed leak is closed. `NET`'s accounting leaves nothing uncosted.

**Legality quotations are exact.** `progress.json:25`, `LITSWEEP3_DATA_CENTRIC.md:82` (including the Wall-A parenthetical), `:91`, `:95`, the registry `claim`/`dedup_boundary`, `stage_0_reachability`, `banned_constraints[10]`, `HEADCOV_PREGATE_RECORD.md:305-310` (including "0.82–0.94"), `NCA_FORENSIC_RECON.md:110` and `:112`, `LITSWEEP5 §4(ii)` and `§2`, F47's and F75's `ban_scope`, F113's fitted-relation-score sentence, round 14's "indistinguishable" reading, F114's `0.9406/0.8915/0.8154` and "30×–92×" — all verbatim at their cited locations.

**Cost and hardware are real.** `sacct` confirms job 13847 = 8 CPU / 32 G / `00:29:49` / COMPLETED. The 36 banked C02 mints run `33.2–60.0 s` (median 41.9), so "25–60 s/head" and a 36-CPU-minute ceiling are right. `sinfo` shows a single node (`foscsmlprd01`), and the banked arena, the fidelity run and C02's independent re-mint all ran on it — Tier-B 4-dp parity on this exact object has already been demonstrated across sessions, so `GATE-FLOOR` is achievable and `RUNTIME_DRIFT` is the correct residual.

---

## CRITICAL

**C-1 — §5.2's blanket label-blindness claim is false, and `GATE-BLIND` is structurally blind to the one feature that carries the target.** §5.2 asserts: *"Every feature reads only: the query item's own key, the bank keys, and the bank labels. **No feature reads the query-side gold-label array.**"* `FULL`'s sixth structural feature — *"the number of the item's top-20 members that are themselves stable inversions"* — is a function of the query-side gold labels of other items (their inversion status is defined when they were queries in their own folds). `GATE-BLIND` is specified as `query_label_reads_during_features == 0` with the array withheld from the builder's signature; since the `is_stable_inversion` array is a *derived* quantity, the gate either **HALTs the run** (if the derived array is built inside the phase) or **passes vacuously** (if it is pre-materialised and handed in). Either way the design's only structural leak detector cannot see the only feature that touches the target. Worse, the justification — *"computed from training-fold items exclusively and therefore disjoint from the scored item's own target"* — is **false for the model's fitting rows**: a training row `i′ ∈ fold g` has neighbours in every fold except `g`, *including fold f*, so item `i`'s own target enters `i′`'s feature, `i′` is in the fit set for the model that scores `i`, and `i`'s target reaches `i`'s score through the fitted coefficients. The magnitude is `O(1/n)` and will not fake a large `ΔAUC` — but the repair v2 claims (structural enforcement) does not exist, and `ΔAUC` is the primary. **Repair, pick one:** (a) delete feature 11, after which the §5.2 blanket claim becomes true and `GATE-BLIND` becomes meaningful; or (b) keep it and recompute it **per scoring fold**, masking fold-`f` neighbours for every fitting row and normalising by the eligible-neighbour count so scored and fitting rows share a support — and re-specify `GATE-BLIND` to count reads of the **target-derived** arrays (`is_inversion[seed]`, `is_stable_inversion`) with the per-fold mask asserted as an integer count, not only reads of the raw label array.

---

## HIGH

**H-1 — `τ_hi`, the registered co-primary threshold, is not a well-defined number.** `τ_hi = median(|score_i| : i ∈ P_0)`, but `|score_i|` is a per-`(item, seed)` quantity and there are three of them per stable inversion. The design never says whether the median is taken over the 3-seed **mean** vote (ERRPAT's convention), over per-seed values pooled, or per seed separately — and the answer changes `P_{τ_hi}` and therefore `K-REACH`, `K-FELDMAN` and `K-NET` at the co-primary. `q25`/`q75` inherit the same hole. In a preregistration whose entire claim is that thresholds are frozen before the run, this one is not frozen. **Repair:** pin it as `τ_hi = median over items in P_0 of mean_s |score_{i,s}|`, matching §5.3's item score, and state it in the same terms for `q25/q75` and `q_max`.

**H-2 — the `τ_hi` branch can die at `K-REACH`, and §8's KILL-scope attribution has no bullet for that.** §8's second clause of rule 1 requires `K-REACH` to clear at `τ_hi` too. Because `|P_{τ_hi}| ≈ |P_0|/2`, that is the inequality `|P_0|/n ≥ 0.10` in disguise — roughly the plausible value (HateMM ≈ 0.10, ZH ≈ 0.095 on the transferred F88 rates), so the co-primary sits on a knife edge decided by an arithmetic identity, not by identifiability or conversion. §8's scope bullets cover only "`K-REACH` fired at `τ_0`" (closes all `τ`) and "`K-FELDMAN`/`K-NET` fired" (closes `{τ_0, τ_hi}`); the realistic case *"clears at `τ_0`, fails reach at `τ_hi`"* is in neither, yet the record would report a KILL scoped to `τ_hi` where nothing about `D-FELDMAN` or `NET` was ever adjudicated. **Repair:** drop `K-REACH`-at-`τ_hi` (population size is already established at `τ_0`; `K-NET` at `τ_hi` is not arithmetically dead and is the co-primary's real content), or add a third scoping bullet stating explicitly that a reach failure at `τ_hi` closes nothing about identifiability or conversion there.

**H-3 — F97 and F98 are mis-registered as a "null"; the correct banked prior is sharper, is *band B*, and sits below `K-NET`'s bar.** §3.1 and §6 say *"F97 (K-VGA-3: F47-family features beat the trained relation profile as gating features **and the whole family still failed**)"* and conclude *"the registered prior for `D-FELDMAN` is therefore a null"*. F97's own `ban_scope` says the opposite about identifiability: *"HONEST POSITIVE DATUM… F47-features-as-adjudication-gate is REAL and permutation-validated — +0.0269 on HateMM (p=0.0050, fold signs +++++), +0.0104 on ZH (p=0.0050), +0.0182 on EN (p=0.0100) — a genuine refinement of F47's epitaph (dead as a per-item CHANNEL SELECTOR, not dead as a per-item ADJUDICATION GATE)"*, and F98 records *"the F47 features have a measured, already-banked ceiling of +0.0269"*. That is not a null: it is **identifiability real, conversion sub-bar** — precisely band B — and `+0.0269` sits *below* `K-NET`'s `+0.030`/22.3-item bar. It is also a **raw-arena** number, which under F113's 9/9 non-transfer makes it *more* adverse in head space, not less. §6 registers F113's pair-level `d_AUC` as "the nearest thing to" a closure while omitting the item-level anchor that is both closer and arithmetically adjacent to the bar. **Repair:** restate F97 accurately, register `+0.0269 / +0.0104` (with its raw-arena scope and F113's one-sided optimism) as the pre-registered prior for `K-NET`, and state that the campaign's own base rate makes band B the pre-declared expectation.

**H-4 — F98's `ban_scope` clause (b) is never adjudicated, and it names `NET`'s object.** F98 bans, verbatim: *"(b) ANY per-item selector, router or adjudication gate over the same neighbourhood **WITH ANY FEATURE FAMILY**"*. §5.3's `NET` is literally that object's arithmetic — top-`k` by a per-item classifier score over the deployed top-20 neighbourhood, predictions flipped. §3.1 adjudicates F47 at length and F98 appears only as an epitaph and a prior; the ban is read **narrower than its own text**, which is exactly the failure mode the reopen's verification pass was instructed to hunt. The scientific consequence matters more than the governance one: `NET` prices a *banned per-item selector*, while §10's Stage-1 successor is a *global symmetric reshaper* whose conversion `NET` does not measure — and F66/law-I say the symmetric object converts far less. **Repair:** add an F98 adjudication beside §3.1's F47 one, stating (i) the ban's measured object and its `+0.0269` ceiling, (ii) that `H-L4` forecloses building the banned selector, and (iii) that `NET` is therefore an **optimistic upper bound** on the Stage-1 global operator's conversion — so a `K-NET` failure is a fortiori a KILL, and a `K-NET` pass licenses nothing about the successor.

**H-5 — there is no threshold-degeneracy control, in a campaign that has twice measured this operator class collapsing into one.** F96 measured its operator agreeing with a **pure global threshold shift** on `95.03 / 97.75 / 99.45 %` of items; F98's DEG-A measured `0.9570 (HateMM) / 0.9508 (EN)` against a frozen `0.95` kill line, with bare `THRESH_best` scoring `+0.0188` — *more* than the learned operator. F98's `ban_scope` (d) closes *"re-deriving the HateMM train-arena threshold observation"*, and C09's own registry dedup boundary excludes *"thresholding"*. `D-FELDMAN`'s `BASE` is led by `|score_i|`, so the selected set `S` can be a threshold move in costume, and nothing in §6.1 or §7.2 checks it. **Repair:** add a DEG-style control — the agreement between `S` and the best bare global-threshold shift of the same size `k` (and against a fixed-`k` neighbourhood rule), with the frozen `0.95` kill line the campaign already uses, gating a CONTINUE; and report F113's head-space `THRESH_best +0.0041` as the anchor.

**H-6 — `K-FELDMAN` is not computable as written: Holm is applied to confidence bounds with no p-value defined.** §5.2 says *"the Holm-corrected one-sided 95 % lower bound on `ΔAUC` is > 0"* over a family of 4 at `α = 0.05`. Holm is a step-down procedure on **p-values**; the design defines only bootstrap quantiles, and "Holm-corrected" and "95 %" are mutually inconsistent (Holm's first threshold here is `α/4 = 0.0125`, i.e. a 98.75 % bound). C02's own A0 `DECISION.json` carries an explicit `holm_family` with `p` and per-hypothesis `threshold` — the precedent exists and v2 does not follow it. Separately, the dataset conjunction is an intersection-union test and needs **no** correction, so a family of 4 is over-conservative and burns power on the two `τ_hi` hypotheses that H-2 shows may be pre-dead. **Repair:** define the bootstrap ASL (`p = fraction of resamples with ΔAUC ≤ 0`, with the `1/(B+1)` floor), order it, apply Holm over the **2 `τ`** hypotheses per dataset, and require rejection on both datasets as an IUT; report the correspondingly adjusted one-sided bounds rather than a fixed 95 %.

**H-7 — §8's clause 4 contradicts §7.2's own gate definitions and can block a CONTINUE the design says it should not.** §7.2 is headed *"Validity gates — HALT-only; a failure publishes no verdict"*, but `GATE-DEVFID` is declared *"Reported, and NOT a HALT"* (a stop-rule trigger *"publishes the verdict with a `PROXY_FIDELITY_FLAG`"*), `GATE-SEED` is an emission with no pass predicate, and `GATE-NULL`'s route disagreement is *"published as a first-class finding"* rather than a HALT. §8 then makes CONTINUE conditional on *"all ten validity gates pass"*. Read literally, a `STOP_RULE_TRIGGERED: true` on `GATE-DEVFID` converts a CONTINUE into a KILL — the opposite of §7.2. **Repair:** split the ten explicitly into HALT gates (`GATE-FLOOR`, `GATE-PARITY-λ0`, `GATE-BLIND`, `GATE-LEDGER`, `GATE-NESTED`, `GATE-SELFTEST`, `GATE-ARENA`) and reporting/scoping instruments (`GATE-DEVFID`, `GATE-SEED`, `GATE-NULL`), and reword §8 clause 4 to name only the HALT set.

**H-8 — CAL-3 is mandatory, is triggered by §8's own raw leg, and is silently omitted.** §2 states *"`PREGATE_CALIBRATION_CLAUSE.md` **CAL-0 … CAL-5** are adopted by citation and are binding on this run"*, then applies CAL-0, CAL-1, CAL-2, CAL-4 and CAL-5 — CAL-3 appears nowhere. CAL-3 is *"mandatory whenever a raw Δ ≥ +0.010 is reported"* and requires the raw Δ to be reported *"together with the deployed space's own gold-cheating ceiling for the same operator family"*, with `RAW-ARENA ARTEFACT` labelling if the raw legal number exceeds the deployed oracle. §8 recomputes the identical battery on the raw arena, where `O1` and the operating-point Δs will certainly exceed `+0.010`. **Repair:** apply CAL-3 explicitly — either name the banked deployed oracle for the flip-a-stable-inversion family, or discharge it on the record with "no deployed oracle is banked for this operator family, so the CAL-3 comparison is unavailable and no raw Δ is escalated," which is the honest form.

---

## IMPORTANT

**I-1 — §6's three bands and §8's decision rule disagree.** Band C says the defect *"is a locatable topology defect… and C09 earns Stage-1"* on `ΔAUC` and `NET` alone; §8 additionally requires `K-REACH` and the gates. These can diverge: `K-NET` at `k = 2|P_τ|` can pick up unstable errors and clear `22.3` with `|P_0| < 37.2`. Restate the bands as a function of the same three rules §8 uses.

**I-2 — `AUC_strat` is under-specified in four ways that are each a forking path.** The `|score|` tercile is defined *"in-run over all `n` query items of the (dataset, seed) cell"*, so strata are seed-specific — yet the text says *"each of the 12 configuration strata"*, which is only true if strata are pooled across seeds. The pooling unit of the Mann-Whitney (rows vs items) is unstated. Whether strata (and hence tercile edges) are **recomputed on each bootstrap resample** is unstated. And there is no rule for strata with zero positives or zero negatives (weight 0 is the obvious answer, but a cell where *every* stratum is single-class leaves `AUC_strat` undefined). Pin all four.

**I-3 — `SHUFFLE-POP`'s domain and evaluation point are unstated.** *"A uniform random permutation of the per-item target vector over the query items of one dataset"* — over all `n` query items, or over the analysis set `P ∪ CONFIG-MATCHED-CORRECT`? The two give different nulls, because unstable errors are excluded from `D-FELDMAN`. §8 clause 4 also does not say at which `τ` the `[0.45, 0.55]` band is checked. Pin both.

**I-4 — provenance defects (bundle).** (a) §4.2 quotes the registry's F88 line as *"ZH 22 of the 25-item union wrong 3/3 **(88 %)** with NOTHING at exactly 2/3…"* — `(88 %)` is **not** in the source (`dispositions.promoted.supporting_evidence_verified`); it is an interpolation inside quotation marks. The HateMM `(89-93 %)` *is* in the source. (b) §4.4 cites `ERRPAT_HateMM:130` for median rank `3.0` and `6/27`; those are at `:134` and `:135`. (c) §5.2 cites `ERRPAT_HateMM:130-136` for *"Distance-based abstention/gating has essentially no dynamic range to work with"*; that sentence is at `:141` (the `~0.9999` cosines are at `:131`). (d) §8 quotes F113's caveat as ending *"…cannot be a head-space positive."* — the source continues *"(F95's own limitation L1, untouched here); that any of this transfers to TEST."* (e) §10 presents kill-risk (i) as a closed quotation but drops *"and section 1.3's +0.0286"*. (f) ERRPAT's row is *"median |vote| (3-seed **mean vote**)"*; the qualifier is dropped. Fix the line numbers, mark the ellipses, and remove the interpolation.

**I-5 — the CAL-2 substitution is called *"strictly stronger"* and is not.** CAL-2's hard half is that `FIXK_20` **changes 0 items and gives `d_acc = 0.0000`** — a check that the *treatment harness* returns the floor when parameterised to the deployed rule. `GATE-FLOOR` checks that the *floor itself* reproduces. Different objects. Add C09's actual analogue: with `S = ∅` (or `k = 0`), assert `Δacc = ΔmF1 = 0.0000` and `net_s = 0` exactly, HALT otherwise.

**I-6 — executability nits that would become engineering HALTs at freeze.** (a) `headspace_fidelity.py:66` hard-codes `mint_{dataset}_s{seed}_ffull.npz` under a single `--mintdir`; the six deployed-configuration heads must be written under exactly that name or `GATE-DEVFID` cannot run. Not stated. (b) `GATE-PARITY-λ0` says *"bit-for-bit at 4 dp"* — the exact oxymoron the round-1 review flagged as the trap that killed three C01 runs; `GATE-FLOOR` gets it right ("at 4 decimal places") and this one should match. (c) `GATE-ARENA`'s band edges are unpinned where C02's `ARENA2` used `majority_rate + 0.02` and `0.98`. (d) Two `FULL` features (the rank-1 per-class similarity gap; 50-NN local density) need full-row / `k = 50` faiss searches that `deployed_vote`'s `(I, sim)` does not return — say so, since §2 claims the analysis script is the only new code.

**I-7 — the registry's own submission preconditions are not recorded.** `gate0_reopen_2026_07_31.c09_next_step` and `dispositions.promoted.sequencing` both state that the CPU job *"waits for C04's tranche to terminate (serial-execution precedent) and for main-dialogue authorization"*. The STATUS block records what has not been done but not what must be true before submission. Add both as explicit preconditions.

**I-8 — the cost budget prices the mints and not the battery.** *"36 CPU-minutes, and the analysis adds minutes"* covers 36 head mints. It does not cover 2 datasets × 2 `τ` × [200-draw `SHUFFLE-POP` (which **refits** `FULL` and `BASE` over 5 folds each draw) + 200-draw `RANDOM-POP` + `B = 10000` bootstrap] plus `UNSTABLE-POP` plus the raw-arena leg. This is still cheap, but state the estimate rather than assert it.

**I-9 — the estimator is fit on a filtered population and applied to the full one.** `D-FELDMAN` excludes unstable errors from both classes, yet §5.3 ranks **all `n`** items by the `FULL` model's OOF probability. Unstable errors (~7-8 % of the arena, and the items whose `NET` contribution is partial) are out-of-support for the ranker that decides where they land. Not a leak; declare it, and report the composition of `S` by class (stable inversion / unstable error / always correct) at every operating point.

**I-10 — `LITSWEEP5 §4(ii)`'s independent leg is carried as numbers without its conclusion, and `ΔAUC > 0` has an unhandled third explanation.** (a) §3.2 records the counts `0/109, 0/102, 0/92` as *"untouched"* but omits what the source draws from them: *"Training labels **cannot** supervise the test-time selection decision in this pipeline — a data-generating-process obstacle upstream of any selector capacity."* That sentence bears directly on §10's train-label-supervised successor. State it, and give the answer F113 supplies (the fold-head arena's train error rate is comparable to test, so the inverse-base-rate obstacle does not transfer) rather than leaving it out. (b) §6's table opposes H-TOPOLOGY to H-MEMORISATION only. A `ΔAUC > 0` driven by feature 11 is equally consistent with **clustered annotation noise** — a data defect no encoder operator can fix — and this repository has documented data defects on both datasets (MHC-ZH's `<em>` shortcut at hate-rate `0.5802` vs `0.1161`; HateMM's 39/744 whitespace-only transcripts). Add the third column and say which reported quantity, if any, separates it.

---

## Bottom line

The legality spine is solid: the two written texts are exact, `H-L1`…`H-L4` are the right boundaries, `H-L4` and §10's void-if-no-operator precondition are real improvements, and §3.1's F47 adjudication is honest in both directions. C-3 and C-4 are properly repaired, the instrument exists and runs, and the arithmetic checks out end to end. The two things that must change before freeze are narrow and cheap: **the C-1 repair does not actually hold** (feature 11 defeats `GATE-BLIND` and falsifies §5.2's blanket claim), and **`K-FELDMAN` cannot be evaluated as written** (no p-value, an inconsistent Holm/95 % pairing, and a `τ_hi` threshold that is not a number). Alongside those, register the prior the repository actually banked — F97/F98's `+0.0269` conversion ceiling for exactly this feature family, sitting below the `+0.030` bar — and adjudicate F98's `ban_scope` (b) against `NET`. Do that and this becomes what it claims to be: a `$0` pregate whose most likely outcome, band B, is written in before the run and is a complete and valuable result.


===== RAW REVIEW: refine-logs/C09_A0_V3_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v3 — Independent Design Review, Round 3

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V3_RECORD.md`
**Verdict.** `REVISE — 1 Critical / 8 High / 11 Important`

---

## Prior-round audit

**Round 1 (4C/8H/10I) — genuinely discharged, 22/22.** I re-checked each against v3 rather than the ledger. C-1 (gold-purity leak) is gone: §4.4's `pred_purity` is now defined against the *predicted* class and is label-blind. C-2 is answered by the conditional/incremental `ΔAUC` plus §6.1's three-valued taxonomy, and round-1's "conversion-equivalent AUC threshold" request is *correctly refused* in §5.2 with a sound reason (AUC does not determine precision at fixed `k`) and replaced by `π* = (1 + bar/k)/2`, which is the right inversion of `net = k(2π−1)`. C-3 is closed: the arena fold assignment is a function of the label vector alone (`headspace_mint.py:204`, exact), so folding by arena fold does group all three seed-rows. C-4 is closed and `GATE-SELFTEST`'s `net_s = 2|S∩wrong_s| − k ≡ n·Δacc_s` is a true identity. H-3's vote-scale repair is exact against `mechfix_ops.py:94-95`. I-1's "test-derived buckets" complaint is *better* than repaired — the strata are now computed in-run and are label-blind.

**Round 2 (1C/8H/10I) — 13 fully discharged, 6 partial.**

- **C-1 (feature 11) — DISCHARGED.** Feature 11 is deleted (`FULL` 13→12), and `GATE-BLIND` now guards `lab_query`, `is_inversion[seed]`, `is_stable_inversion` as integer counts. The *structural* half (the builder's signature admits neither array) is the part that actually works. But the blanket sentence it licenses is still over-broad (I-4 below), and the same class of residual path survives at `τ_hi` (H-4 below).
- **H-1 (`τ_hi`) — DISCHARGED as a number**, but the pinning introduced a new full-sample-transductive dependency (H-4).
- **H-2 — DISCHARGED.** §9's third scope bullet is present and correctly says the `τ_hi` reach failure closes nothing about identifiability or conversion.
- **H-3 — DISCHARGED.** F97's `HONEST POSITIVE DATUM` is verbatim exact at its `ban_scope`, `+0.0269 / +0.0104 / +0.0182` correct, band B declared as the prior.
- **H-4 (F98 (b)) — PARTIAL.** The ban is quoted exactly and the governance half is honest; the *scientific* consequence drawn from it is unsound (H-5 below).
- **H-5 (`K-DEG`) — PARTIAL.** The control exists and gates the verdict, and F96/F98's texts are exact. But the twins are built on the wrong confidence scale (H-1 below), `FIXK`'s "best" is undefined, and the twin's own vote path has no parity check (H-8, I-7).
- **H-6 (`K-FELDMAN` computability) — PARTIAL.** Holm-over-2-τ, IUT-across-datasets and the `1/(B+1)` floor are all correct now; the *bootstrap itself* is the wrong null (H-2).
- **H-7 (gate/decision contradiction) — NOT DISCHARGED.** The split into 8 HALT + 3 reporting gates landed, but §9 re-creates the contradiction in a new and worse form (Critical below).
- **H-8 (CAL-3) — DISCHARGED** in the honest form; only its premise needs a correction (I-10).
- **I-1 — PARTIAL.** Bands restated but still not exhaustive, and the hole is precisely the divergence round 2 named.
- **I-2, I-3, I-5, I-7, I-9, I-10 — DISCHARGED.** All four `AUC_strat` pins are present; `SHUFFLE-POP`'s domain is the analysis set; `GATE-ZEROOP` is added and the "strictly stronger" claim withdrawn; both re-measured data-defect counts reproduce exactly.
- **I-4 (six provenance defects) — PARTIAL 4/6.** (a), (b), (e), (f) fixed; (c) fixed the line number but introduced a *space* mislabel (I-2 below); (d) claims "in full" and is not (I-3 below).
- **I-6 — PARTIAL 3/4.** (b), (c), (d) fixed; (a)'s mint-directory statement is factually wrong against the banked precedent (I-6 below).
- **I-8 — PARTIAL.** Itemised, but the table's own arithmetic is wrong by 2× on the two largest lines (I-5 below).

---

## Verified as sound (do not re-litigate)

**Every quotation I opened is exact at its cited location**, with three exceptions listed under IMPORTANT. Specifically verified verbatim: `stage_0_reachability`; `banned_constraints[10]`; the registry `claim` and `dedup_boundary` at `candidate_registry[8]`; F47's, F75's, F96's, F97's and F98's `ban_scope`; the F98 epitaph and `+0.1492/+0.1520/+0.2186 → +0.0134/−0.0069/+0.0000`; the reopen's "a large oracle is no longer evidence for a candidate in this channel — it is the precondition every failed candidate already met" (`GATE0_REOPEN_2026-07-31.md:1005-1006`); `supporting_evidence_verified`'s F88 line with no interpolation; kill-risk (i) with all four F114 qualifiers; `progress.json:25`; `LITSWEEP3_DATA_CENTRIC.md:82` (with the Wall-A parenthetical), `:91`, `:95`; `LITSWEEP5_COMPLETENESS.md:128` and §2's "generalizes past its named-loss letter"; `HEADCOV_PREGATE_RECORD.md:305-310`; `NCA_FORENSIC_RECON.md:110`, `:112`; `ERRPAT_HateMM:130/:134/:135` and `:141`; `ERRPAT_MHC-ZH:234-235`; the ERRPAT curation null `+0.0016` vs `+0.0031/+0.0000`; round 14's "indistinguishable is the exact reading" (`GATE0_REOPEN_2026-07-31_REVIEW_ROUND14.md:38`); DET-1…DET-4 and §1.3's `auc_logistic` 4-dp invariance; CAL-0…CAL-5.

**Every number re-derived from source.** All twelve `GATE-FLOOR` floors are exact against `result.acc_deployed` / `result.mF1_deployed`. Fold sizes `149×4+148` / `116×4+115` exact. `posrate_bank 0.4005/0.3109` → majority `0.5995/0.6891`, and `GATE-ARENA`'s bands `[0.6195, 0.98]` / `[0.7091, 0.98]` reproduce C02's `ARENA2` `lower` to the digit (`0.6194623…`, `0.709119…`). `raw_deployed_acc 0.8441/0.8480`; raw space is `l2n(concat(l2n(img), l2n(txt)))`, 3584+3584 = 7168-d, seed-free. `B_fid 0.0093/0.0086`, `STOP_RULE_TRIGGERED: false`. `22.3/17.4 = 0.030×744/579`; `37.2/29.0 = 0.050×744/579`. I re-measured the data defects from `data/gt/*/train.jsonl` myself: MHC-ZH `243/579` `<em` rows, hate rate `141/243 = 0.5802` vs `39/336 = 0.1161` against base `180/579 = 0.3109` — all four exact; HateMM `39/744` whitespace-only, `0` on ZH (independently corroborated by C02's `VIEW_SUPPORT.degenerate_causes.EMPTY_TEXT = 39`). `GATE-NULL` re-measured from the caches: HateMM zero-img `[355]`, zero-txt `[355]`, `hate_video_95`, label `1`; MHC-ZH `[]`/`[]`.

**Executability is real.** `mechfix_ops.deployed_vote` returns `(votes, preds, I, sim)`; `_norm32`/`_flat_ip` exist and are the same faiss engine, so the extra `k=50` and per-class top-1 searches are genuinely artefact-free. `headspace_mint.py:106-116` (`torch.load` `test_seen`//`test` guard), `:192-194` (skip-if-exists), `:203-216` (fold parity vs `vsw_ckpt/<ds>/f*.npz`) and `:274-281` (`torch.save` no-op) are exact to the line, four for four. `headspace_fidelity.py:66` is literally `mint_{}_s{}_ffull.npz`; `headspace_arena.load_mint` uses `mint_{ds}_s{seed}_f{full|0..4}.npz` — v3's naming is right. `CLI` admits only `{hatemm, zh}`. `mechnov_pairverify.py`'s on-disk sha256 still equals `FROZEN_PAIRVERIFY_SHA`, so the mint will not HALT on its own assert. All ten `vsw_ckpt` fold files and all six fidelity floor trainlogs are present.

**`GATE-FLOOR` at 4 dp is demonstrably achievable, not aspirational.** C02's independent re-mint (`C02_A0_OUT.json`, `ARENA2.pooled_native_acc`) reproduces `0.8884408602 / 0.8857526882 / 0.8857526882` and `0.8929188 / 0.8894646 / 0.8946459` — the banked fold-head values *exactly*, not merely to 4 dp — under the same node and the same `python 3.11.8 / numpy 1.26.4 / scipy 1.17.1 / sklearn 1.5.2 / torch 2.6.0+cu124` quintet the banked `meta.runtime` records. This is the single most important executability fact in the design and it holds.

**Cost is real.** `sacct` confirms job 13847 = 8 CPU / 32 G / `00:29:49` / COMPLETED. The 36 banked C02 mints run `33.2–60.0 s` (median `41.85`), so the 36-CPU-minute mint ceiling is exact. The total ≈70 min is the right order (see I-5 for two arithmetic slips inside the table that do not change the conclusion).

**Legality holds.** Both blessed texts are exact; `H-L1`…`H-L4` are the right boundaries; §3.1's F47 adjudication is honest in both directions and correctly confines the F114 correction to F47's *train-supervised* leg without touching the decision-level-meta-features sentence; §3.3 carries LITSWEEP5's conclusion, not only its counts, and answers it with the arena's own `0.105–0.114` train error rate (which I re-derived from the floors) while keeping the residual headwind; §11's symmetric-reshaper argument is correctly grounded in `LITSWEEP3:82`'s *inference-time* symmetry test, correctly concedes that a region-weighted triplet+BCE is hard-example weighting and therefore fails C09's own dedup boundary, and correctly makes a CONTINUE void absent a nameable operator. **I found no path by which a query item's own gold label reaches its own features or its own model's fitting rows as a target — except the one at H-4.**

---

## CRITICAL

**C-1 — §9's "two-valued, exhaustive" rule converts every HALT into a published KILL, contradicting §8.1, §6.3 and §5.2 in terms; and §9 clause 5 states the `SHUFFLE-POP` predicate differently from §6.3 and from the ledger.** §8.1 is headed *"HALT gates — a failure publishes **no** verdict"*; §6.3's `SHUFFLE-POP` rule ends *"outside that band the estimator is leaking and **no verdict is published**"*; §5.2's degenerate case says *"the run **HALTs** with that cell named"*; and §8.1's `RUNTIME_DRIFT` paragraph says a `GATE-FLOOR` failure under drift is *"an **engineering HALT** … **not a scientific result**"*. §9 then makes clause 5 — *"`SHUFFLE-POP`'s permutation-null mean … and **all eight HALT gates of §8.1 pass**"* — a **CONTINUE condition**, and closes with *"`KILL` in every other case. `KILL` and `CONTINUE` are complements by construction."* Read literally, and §9 is the frozen rule that governs at verdict time: a thread mis-export, a version drift, a `GATE-BLIND` trip (i.e. **a detected leak**), or a single all-single-class cell publishes a **KILL of C09**. That is a wrong verdict produced by a non-scientific cause, and it is the exact failure round-2 H-7 charged, re-created with the polarity reversed. Separately, §6.3 and the §12 ledger (I-3) require the `[0.45, 0.55]` band *"at **both** `τ` on **both** datasets"*, while §9 clause 5 requires it *"at **that** `τ` on both datasets"* — two different predicates for one frozen rule. **Repair:** (i) delete "and all eight HALT gates of §8.1 pass" from clause 5 and replace the closing sentence with "The run publishes a verdict only if all eight HALT gates of §8.1 pass, the `SHUFFLE-POP` band holds at both `τ` on both datasets, and no §5.2 degeneracy HALT fired; **conditional on a completed run**, `KILL` is the complement of `CONTINUE`. A HALT publishes no verdict and is an engineering result with a diagnose-repair-resubmit path."; (ii) restate clause 5's `SHUFFLE-POP` predicate as "at both `τ` on both datasets", matching §6.3 and the ledger; (iii) change §9's heading from "two-valued, exhaustive" to "two-valued on completed runs".

---

## HIGH

**H-1 — `K-DEG`'s degenerate twins are built on the per-seed `|score_{i,s}|` scale while `S` is built on the per-item mean, which systematically depresses agreement and biases the campaign's only degeneracy gate toward not firing — and it contradicts §4.3's own pinning sentence.** §4.3 pins *"there is exactly one confidence scale in this design and it is per item"* (`c_i ≡ mean_s |score_{i,s}|`). §5.3 builds `S` on the per-item mean OOF probability, so `S` is seed-independent. §6.2 then defines `THRESH-SYM` as *"the `k` items with the smallest `|score_{i,s}|`"* and `FIXK` as truncated *"to the `k` with the largest `|Δscore|`"* — both per seed — and takes agreement *"averaged over seeds"*. Any seed-to-seed jitter in `|score|` shrinks `|S ∩ twin|` for reasons that have nothing to do with degeneracy, against a hard `0.95` line that F96/F98 calibrated on same-scale comparisons. Since `D-FELDMAN`'s `BASE` is led by `|score|`, the most likely real-world outcome is exactly the one this gate exists to catch — a threshold move in costume — and the scale mismatch is a systematic tilt toward letting it through to CONTINUE. **Repair:** define all three twins on the same per-item scale `S` is built on (`THRESH-SYM` = the `k` items with smallest `c_i`; `THRESH-BEST` on the seed-mean signed score; `FIXK` on the seed-mean `|Δscore|`), or run both scales and take `K-DEG`'s agreement as the **maximum** over per-item and per-seed twins.

**H-2 — `K-FELDMAN`, the design's only inferential rule, gets its p-value from a bootstrap that holds the fitted model fixed, which omits the dominant source of null variability for a nested-model `ΔAUC` — while a refit-based null is already computed and is not used for the primary.** §5.2 states, and §2's budget confirms, *"re-scoring the banked OOF predictions (**no refits**) … which is what keeps `B = 10000` cheap."* `ΔAUC = AUC_strat(FULL) − AUC_strat(BASE)` is a *difference of two estimated models*; under H0 the five structural features are noise and the entire question is whether this particular fit found spurious structure. A bootstrap that conditions on that fit measures only the item-sampling variance of a fixed score vector and therefore understates the sampling distribution's spread, inflating type-I error in the CONTINUE direction. The design already pays for the correct null: `SHUFFLE-POP` performs 200 refitted permutation draws of `FULL` **and** `BASE` on the same folds and the same frozen strata — that is exactly the permutation distribution of `ΔAUC` under H0 — but it is consumed only as a `[0.45, 0.55]` band on `AUC_strat(FULL)`. **Repair:** define `K-FELDMAN`'s `p` as the `SHUFFLE-POP` permutation ASL `p = (1 + #{d : ΔAUC_d ≥ ΔAUC_obs}) / (D + 1)` with `D = 200` (resolution `1/201 = 0.00498`, adequate against Holm's `α/2 = 0.025`), keep the item bootstrap for the reported interval only, and state explicitly that the bootstrap is conditional on the fit.

**H-3 — nothing in the design declares stratum occupancy or effective overlap, so `AUC_strat` can return a null from *absence of two-class overlap* and the record would publish it as band A, i.e. as evidence for H-MEMORISATION; and `SHUFFLE-POP` structurally cannot detect this.** The banked marginals make this concrete, not hypothetical: `ERRPAT_HateMM:130` gives median `|vote|` `0.7267` for errors vs `0.9873` for correct, and `:133` gives median rank-weighted purity `1.0000` for always-correct vs `0.1667` for errors (ZH `:220`: `1.00` vs `0.15`). Both stratification axes therefore separate the two classes almost completely, so most of the 12 strata will be single-class and receive weight 0, and `AUC_strat` will be carried by two or three strata — worst at `τ_hi`, where the positives are the *high*-`|score|` half and move into terciles that contain very few of them. §5.2 HALTs only in the extreme case where *every* stratum in a cell is single-class; the far more likely intermediate case is unhandled and is read as a scientific null. `SHUFFLE-POP` cannot see it: permuting the target spreads the "positives" uniformly across strata, destroying the concentration that creates the risk, so a clean `[0.45, 0.55]` band is uninformative about the real regime — the mirror of the honest concession v3 already makes about the withdrawn "preserves all configuration marginals" claim. **Repair:** add a pre-declared `STRATUM_OCCUPANCY` emission (per dataset × seed × `τ`: number of strata with weight > 0, `n_pos` and `n_neg` in each, and total `Σ n_pos·n_neg`) and a frozen power rule — e.g. tag the `ΔAUC` verdict `IDENTIFIABILITY_UNDERPOWERED` rather than band A whenever fewer than 3 strata carry weight or `Σ n_pos·n_neg` falls below a number frozen now — and make band A's optional statement *"H-MEMORISATION is consistent with this object"* conditional on that tag being absent.

**H-4 — `τ_hi` is a full-sample statistic over the gold-defined set `P_0`, so the scored item's own gold label reaches the target vector used to fit the model that scores it, violating §5.2's own nested-CV contract.** §4.3 pins `τ_hi = median over items i ∈ P_0 of c_i`; `P_0` is defined by gold labels, so item `i`'s membership shifts `τ_hi`, and `τ_hi` defines the positive class for the fitting rows in the other four folds. §5.2 asserts without qualification *"An item's score comes from a model fit on items from the other four arena folds only"* and `GATE-NESTED` asserts *"this item's arena fold was excluded from the model that scored it"* — both are true of *rows* and false of the *threshold*. `GATE-BLIND` cannot see it: it guards the **feature-construction phase**, and this is target construction, where reading `is_stable_inversion` is legal. The magnitude is `O(1/|P_0|) ≈ 1/70 ≈ 1.4 %` — roughly **ten times** the `O(1/n)` leak round 2 rated Critical in feature 11 — and it sits on the registered high-confidence co-primary. **Repair (either):** (a) compute `τ_hi` per scoring fold as the median of `c_i` over `P_0 ∩ (the four fitting folds)`, emitting the five values and their spread, so the threshold is as out-of-fold as the model; or (b) keep the full-sample `τ_hi`, declare it explicitly as a transductive constant in §4.3 and §5.2, amend `GATE-NESTED`'s assertion text to say what it does and does not cover, and report the leave-one-out spread of `τ_hi` so a reader can bound the effect.

**H-5 — §3.2 and §10 assert that `NET` is an optimistic upper bound on the Stage-1 successor's conversion, and pre-declare that a `K-NET` failure is *a fortiori* a KILL of the successor. Neither follows.** The argument offered is a *capability class* argument — *"the selector may act on each item individually, the reshaper must apply one map to all"* — but `NET` does not measure the class ceiling; it measures the realised net of **one specific, weak selector**: top-`k` by a 12-feature `lbfgs` logistic probe fit on hand-built geometry features. A trained reshaper optimises `φ` end-to-end with the vote in the loop and is not nested inside that selector's reachable set. The two effects run in opposite directions (higher-capability class, lower-information instrument), so `NET` bounds the successor in **neither** direction. The design already contains the honest upper bound and labels it correctly: `O1`, the label-flip oracle over `P_τ`, which §10 already scopes as *"an upper bound, used only in the closing direction"*. The over-claim is therefore also redundant. This matters because it is a **pre-declared corollary** that would license writing a KILL wider than the measurement supports, and because §10 is the section whose job is scope honesty. **Repair:** delete the "a fortiori KILL of the successor" corollary from §3.2(3) and the "optimistic upper bound on the successor's conversion" bullet from §10; replace both with "`NET` prices one specific per-item selector. Its operator class dominates the successor's, but this particular selector may be weaker than the successor, so `NET` bounds the successor's conversion in neither direction. `O1` is the only upper bound this A0 produces." Keep, unchanged, the correct half already in §11: *"a `K-NET` pass is not evidence the successor converts."*

**H-6 — the net-item currency adjudication adopts the softer of two written figures for the only conversion rule, against C09's own authorising registry entry, and supports the choice with an argument that is true of `O1` and false of `S`.** `gate0_reopen_2026_07_31.dispositions.promoted.bar` reads, verbatim: *"net-items currency **37.2/29.0/27.5** (HateMM/MHC-ZH/MHC-EN) scaled from `banned_constraints[10]`'s 22.3/17.4/16.5 for +0.030"*, and `consequence_for_gate_0` repeats *"NET ITEMS against 22.3/17.4/16.5 for +0.030, **i.e. 37.2/29.0/27.5 for +0.050**"*. §5.3 binds `K-NET` to `22.3/17.4` instead, and demotes `37.2/29.0` to a secondary that *"can never create or block"* a CONTINUE. The stated reason is *"for `O1`, which breaks nothing, `net ≡ n · Δacc`, so a `+0.050` net screen on `O1` would restate the accuracy screen"* — correct, but `K-NET` is not applied to `O1`; it is applied to `S`, where breaks are real and a `37.2` screen is fully independent of the reach screen. The justification therefore does not reach the rule it governs, and the gap is decisive: at `k = |P_0| ≈ 80` the two bars differ by ~7.5 percentage points of required precision. **Repair (either):** (a) make `37.2 / 29.0` binding on `K-NET` — the figure C09's own authorising entry names — with `22.3 / 17.4` reported as the secondary; or (b) keep `22.3 / 17.4` and replace the justification with one addressed to `S`, state plainly that the design is departing from its authorising entry's explicit figure, and record that departure in the freeze record.

**H-7 — one of the five `FULL` structural features is undefined for the majority of the negative class, with no frozen convention.** *"rank of the first neighbour whose bank label differs from the top-20 majority bank label"* has no value when all 20 neighbours carry the same bank label. `ERRPAT_HateMM:133` measures median rank-weighted top-20 purity toward the true label at **`1.0000`** for always-correct items (`ERRPAT_MHC-ZH:220`: **`1.00`**), i.e. at least half of the negative class has a uniform-label neighbourhood; the fold-head arena is *purer* still (`ERRPAT_MHC-ZH:237-240` measures head-space purity `0.1167` vs raw `0.400` on errors and `0.9833` vs `0.85` on correct items). The sentinel is not a nuisance: whatever value is chosen (21, `NaN`-imputed, `k+1`) becomes a near-perfect class indicator and materially determines `AUC_strat(FULL)` — the numerator of the primary. In a preregistration whose premise is that every threshold is frozen before the run, this one is not defined at all. **Repair:** freeze the convention in §5.2 — e.g. "rank `21` when the top-20 bank labels are uniform" — and add the uniform-neighbourhood *fraction*, per class and per cell, to the `FEATURE_DEGENERACY` block so a reader can see how much of `ΔAUC` this single indicator carries.

**H-8 — CAL-2's mandatory hard gate is declared binding, is available in this arena at zero marginal cost, is not run, and its absence leaves the one verdict-gating code path with no parity check.** §2 states *"`PREGATE_CALIBRATION_CLAUSE.md` **CAL-0 … CAL-5**, binding on this run"*. CAL-2(1) reads: *"`FIXK_20` **must change 0 items and give `d_acc = 0.0000`** — the arena's k=20 rule *is* the deployed rule. Miss ⇒ **harness VOID** … (This half is a hard gate and it is sound.)"* v3 substitutes `GATE-ZEROOP` and declines the rest on the ground that *"this arena is not that arena"* — which is a valid reason for CAL-2's **leg (2)** (the F94 Spearman on the raw k-curve) and is not a reason for leg (1): a fixed-`k` rank-weighted vote at `k = 20` is perfectly well-defined in the fold-head arena, and §6.2 **already computes exactly this object** for `k′ ∈ {1,2,3,5,7,10,15}` to build the `FIXK` twin. Adding `k′ = 20` costs one grid point. The gap is not merely formal: `GATE-FLOOR` checks the floor, `GATE-PARITY-FOLD` checks the deployed vote, `GATE-ZEROOP` checks the flip accounting — **nothing checks the fixed-`k` vote implementation**, and `K-DEG`, which reads it, can KILL. **Repair:** add `FIXK_20` to §8.1's HALT gates ("`k′ = 20` must change 0 items and give `Δacc = 0.0000` exactly on every dataset × seed × fold; miss ⇒ harness VOID"), retain the `{1,2,3,5,7,10,15}` grid for the twin itself, and note in §2 that CAL-2 leg (2) alone is out of scope and why.

---

## IMPORTANT

**I-1 — §6.1's three bands are still not exhaustive, and the hole is exactly the divergence round-2 I-1 named.** Band A = `K-FELDMAN` fires; band B = `K-FELDMAN` clears at some `τ` but `K-NET` or `K-DEG` fires *there*; band C = all four clear at some `τ`. The case "`K-FELDMAN`, `K-NET` and `K-DEG` all clear at `τ_0` but `K-REACH` fails there" falls in none — and it is precisely round 2's scenario (`K-NET` at `k = 2|P_τ|` picking up unstable errors and clearing `22.3` with `|P_0| < 37.2`). §9 still returns a well-defined KILL, so no verdict is undefined, but the published taxonomy is incomplete and §6.1 claims otherwise. **Repair:** add "Band A′ — `K-REACH` fails at every `τ`: the population is too small regardless of identifiability or conversion. KILL, and (at `τ_0`) closing every `τ ≥ 0` by arithmetic", and state that the bands partition completed runs.

**I-2 — the ZH "right analogue" measurement is a *pre-head raw fused space* number presented under a "test split, proxy head" label, and the source's next paragraph shows the head space is materially worse.** §4.4's parenthetical reads *"(Transferred motivating measurements, test split, proxy head: `ERRPAT_MHC-ZH_2026-07-26.md:234-235` — …)"*. The source sentence begins at `:233`: *"In the **pre-head raw fused space** over the full 579-row bank, the first same-gold-class train neighbour sits at median rank 1.5…"* — raw space, over the full **train** bank, not the proxy head. `:237-240` then measures the head space collapsing that same population's purity from `0.400` (raw fused, 5 of 22 still majority-correct) to `0.1167` (0 of 22), so "the right analogues are present and top-ranked" is exactly the kind of raw-space fact F113 exists to stop transferring. The HateMM companion (`:134`, median rank `3.0`) *is* head-space (`§2`: "All from the saved per-item top-20 neighbour lists"), so the mislabel is ZH-specific. It gates nothing — §4.4 correctly excludes the right analogue from every feature set — but it is a provenance error in C09's favour on the motivating mechanism. **Repair:** split the parenthetical: label the HateMM figures "test split, proxy head, deployed head space" and the ZH figure "test-split errors, **pre-head raw fused space** over the full 579-row bank", and add `:237-240`'s head-space contrast beside it.

**I-3 — F113's caveat is said to be "carried in full" and is the reopen's abridged rendering, not F113's primary text.** §9 and ledger I-4 both claim fullness. The primary at `HEADSPACE_TRANSFER_PREGATE.md:920-923` reads *"NOT established: that a raw-space negative cannot be a head-space positive (**§8.1**); that any of this transfers to the test split (all arenas here query train-split items held out from their own head, **which is closer to deployment than raw but is still not deployment**); **that the CPU-minted proxy head equals the CUDA floor to better than ±0.0093 (3-seed) / ±0.0280 (single seed)**."* v3 substitutes "(F95's own limitation L1, untouched here)" — the reopen's wording, not F113's — drops the "still not deployment" qualifier, and omits the third clause entirely, which is the one that bears on `GATE-DEVFID`'s own `0.0093 / 0.0086`. Separately, §6.1's *"F113 measured 9 of 9 raw-space positives failing to transfer"* uses F113's operator-cell count where F113 itself scopes the finding at `:917-919`: *"**Established on ONE cell only** … F105/VSW is the only raw-space positive the campaign ever produced."* (That over-read runs *against* C09, so it is not harmful, but it is an over-read.) **Repair:** quote `HEADSPACE_TRANSFER_PREGATE.md:920-923` at its own location with all three clauses, or delete "in full"; and re-scope "9 of 9" as "9 of 9 operator cells, all descending from F105/VSW, which F113 itself records as one cell".

**I-4 — §5.2's boxed blanket label-blindness claim is false as written, and §8.1 contradicts it.** The box reads *"`FULL` now contains no feature that reads any **query item's** gold label, directly or derived."* In this arena every train item is a query in its own fold and a **bank member** in the other four, so `pred_purity`, `|score|`, the first-differing-label rank, the runs count and the per-class rank-1 gap all read other query items' gold labels — legally, exactly as the deployed vote does, which is why `GATE-BLIND` reports `bank_label_reads` as a *"(nonzero, legal) integer"*. The substance is sound and the structural enforcement is real; the sentence that C-1's discharge is *claimed* on is not. This matters because round-2 C-1 turned on precisely this reading of "query item". **Repair:** replace the box with "`FULL` contains no feature that reads the **scored item's own** gold label and no feature that reads any **target-derived** array (`is_inversion[seed]`, `is_stable_inversion`). Bank labels — the gold labels of the other four folds' items — are read, legally and by design, exactly as the deployed vote reads them, and are counted by `GATE-BLIND` as `bank_label_reads`."

**I-5 — the budget table's own arithmetic is wrong by 2× on its two largest lines, and one line double-counts the raw leg.** *"`SHUFFLE-POP` (200 draws × 2 sets × 5 folds × 2 `τ` × 2 ds) | **16 000** LR fits"* — the stated factors give `200 × 2 × 5 × 2 × 2 = 8 000`; same for `RANDOM-POP`. And *"deployed vote + features, all cells | **30 fold-cells × 2 spaces**"* double-counts the raw arena, which has its own line and, being seed-free (`headspace_arena.py:7`), has 5 fold-cells per dataset, not 30. The errors are conservative and the ≈70-minute total survives, but a preregistration under this campaign's numeric-provenance discipline should not carry a factor-2 slip in its own cost table. **Repair:** correct to `8 000` on both lines, and change the vote/features line to "30 head fold-cells + 10 raw fold-cells".

**I-6 — §2's mint-directory statement is false against the banked precedent and against `headspace_mint.py` itself.** §2 states *"All 36 mints are written into **one** directory, `<scratch>/mint/` … **Nothing else is written there.**"* `headspace_mint.py:286-287` writes the run_rac output tree to `os.path.join(a.scratch, "mint", tag)` — i.e. into `<scratch>/mint/` — and the banked convention (`headspace_drive.sh:20`, and C02's own scratch) puts the `.npz` files at the **scratch root** with `<scratch>/mint/` holding 36 run_rac output directories. Nothing breaks (both readers resolve by exact filename), but "nothing else is written there" is wrong, and the I-6a repair was specifically about getting this right. **Repair:** either follow the precedent (`--out <scratch>/mint_{ds}_s{seed}_f{tag}.npz`, `--scratch <scratch>`, `--mintdir <scratch>`) and say so, or keep `<scratch>/mint/` for the `.npz` and pass `--scratch <scratch>/rac` so the run_rac trees land elsewhere; either way delete "Nothing else is written there" or scope it to the `.npz` files.

**I-7 — `FIXK`'s selection criterion is undefined and its size cannot be guaranteed, in a control that gates the verdict.** §6.2 says the twins are *"each of size exactly `k`"* and defines `FIXK` as *"the set of items whose deployed decision flips under **the best** alternative fixed neighbourhood size `k′ ∈ {1, 2, 3, 5, 7, 10, 15}` … truncated to the `k` with the largest `|Δscore|`."* "Best" is never defined — maximum agreement with `S` (F98's DEG-B convention), maximum `net_s`, and maximum flip count give different twins and different `K-DEG` outcomes. And if fewer than `k` items flip under the chosen `k′`, "truncated to the `k`" is vacuous and the twin is smaller than `k`, silently capping agreement below `0.95` so the gate cannot fire. **Repair:** pin "`best` = the `k′` maximising `|S ∩ flip(k′)| / k`, reported per cell" (F98's own convention, and the conservative one), and add a top-up rule ("if fewer than `k` items flip, pad with the next-largest `|Δscore|` items") or redefine agreement as `|S ∩ twin| / min(k, |twin|)` — and say which.

**I-8 — `H-L3` as written would HALT the run on the instrument's own contract; `GATE-LEDGER` already carries the correct predicate.** `H-L3` bans *"**Any** read of a dev **label** … at any stage, by any code path"*, then carves out dev *features* and dev *accuracy*. But `headspace_mint.py` loads the `dev_seen` split whole and stores `lab_dev = dv[3].numpy()` into every `fold == -1` mint, and `run_rac`'s dev evaluation computes the `eval_curve` `GATE-DEVFID` consumes directly from dev labels. So dev labels are materialised, not merely derived-from. `GATE-LEDGER`'s operative predicate — *"dev **label** materialisations **into any decision quantity** (must be `0`)"* — is the right one and is met. **Repair:** restate `H-L3` as "Any read of a dev or test label **into any decision quantity**, and any read of a test path, at any stage, by any code path", so the HALT boundary matches the gate that enforces it and the instrument does not violate its own preregistration on line one.

**I-9 — `UNSTABLE-POP` is arithmetically pre-dead and the design does not say so, though §4.3 extends exactly this courtesy to `τ_hi`.** From the re-derived floors, per-seed error counts are 83–85 (HateMM) and 61–64 (ZH). On the transferred F88 stability rates (~89–93 % HateMM, 22/25 ZH) that implies `n_unstable ≈ 7–9` on both datasets — far under the declared `n_unstable < 20` trigger. So `CONTROL_UNDERPOWERED` will fire on both datasets by construction, and the control that tests the registry claim's *own premise* ("stability carries information") will return nothing, while §6.3 promises it *"reported as a mechanism finding regardless of the verdict"*. The rule is correctly data-independent and should stay; the pre-declaration is what is missing. **Repair:** add to §6.3 the sentence §4.3 already models: "On the transferred F88 rates `n_unstable ≈ 7–9`, so `CONTROL_UNDERPOWERED` is the expected outcome on both datasets. That is declared now, is not an artefact, and means the registry claim's stability premise is **not** tested by this A0."

**I-10 — CAL-3's discharge rests on a premise that is stronger than the record supports.** §9 states *"no deployed-space gold-cheating oracle is banked for the 'flip a nominated stable-inversion population' operator family — the ERRPAT oracles cover the threshold, curation, length-de-bias and stream families, none of which is this one."* `ERRPAT_HateMM §7` ("SOLUTION MAPPING PER CLUSTER") banks per-cluster fix ceilings on the deployed proxy head — FN1 `+0.0326`, FN2 `+0.0140`, FN3 `+0.0233`, FP1 `+0.0233`, FP2 `+0.0233`, FP3 `+0.0093` — which *are* deployed-space upper bounds for flipping a nominated error sub-population, i.e. `O1`'s object on the test split. CAL-3's own hedge (*"wherever one is banked"*) still licenses the outcome, and the raw `Δ` will not exceed those ceilings, so no `RAW-ARENA ARTEFACT` label is triggered either way. Only the premise is wrong. **Repair:** cite `ERRPAT_HateMM §7`'s cluster ceilings as the nearest deployed-space analogue and conclude that no arm is escalated — instead of asserting unavailability.

**I-11 — residual under-specifications and provenance nits (bundle).** (a) §9 clause 3 requires *"a **single** `k`"* clearing `K-NET` and clause 4 then reads `K-DEG` *"at that `(τ, k)`"*; when several `k` clear `K-NET` and only some fire `K-DEG`, the intended reading (`∃k: K-NET ∧ ¬K-DEG`) should be written out. (b) §5.3's item score is *"the OOF predicted probabilities from **the** `FULL` logistic model"* — there are two, one per `τ`; say "the `τ`-matched fit". (c) §8.1's *"the engineering-HALT trap that killed **three C01 runs**"* is a round-1 reviewer's assertion repeated as fact; I found no primary citation for the count in `refine-logs/C01*`. Either cite it or drop the number. (d) §8.1 calls `meta.runtime.versions` a *"quartet"*; it has five members (`python, numpy, scipy, sklearn, torch`) plus the node. (e) §4.3 defines `q25` and `q75` on the `c_i` scale and no section uses them — vestigial from v2; delete or give them a reporting role. (f) §5.5's *"(**no label is read for this diagnostic**)"* is true of the flag construction but not of the diagnostic, whose output is the enrichment of the gold-defined `P_τ`; scope the parenthetical to the flag. (g) §4.3's `median` is not pinned to an estimator; `numpy.median`'s linear interpolation on an even `|P_0|` should be named, since it determines `|P_{τ_hi}|` and therefore `K-REACH` at the co-primary.

---

## Bottom line

The legality spine is solid and I could not break it: both blessed texts are exact, the four HALT boundaries are the right ones, §3.1's F47 adjudication and §3.2's F98 adjudication both concede the ban rather than narrowing it, §3.3 carries LITSWEEP5's conclusion and answers it with a measurement rather than an argument, and §11's void-if-no-operator precondition is a genuine and unusual piece of discipline. The instrument exists, runs, and — uniquely among the things I checked — has *already* been demonstrated to reproduce its own floor exactly, which is the fact `GATE-FLOOR` rests on. Round 1 is fully discharged and most of round 2 is.

What must change before freeze is narrow. One clause in §9 currently turns every HALT — including a *detected leak* — into a published KILL, which is the round-2 H-7 defect with its polarity reversed and is the only Critical. Three of the Highs are places where a repair landed structurally but on the wrong object: `K-DEG`'s twins are on the per-seed scale while `S` is per-item, contradicting §4.3's own pinning sentence; `K-FELDMAN`'s p-value comes from a fit-conditional bootstrap when the design already pays for the refit-based null two lines away; and `NET` is asserted in §10 as an upper bound it is not, when the design's real upper bound (`O1`) is already correctly scoped one bullet earlier. Two more are gaps the banked marginals make concrete rather than hypothetical — the stratum-occupancy question, where `ERRPAT`'s own `0.7267/0.9873` and `0.1667/1.0000` predict that most strata will be single-class, and the undefined feature value that follows from the same measurement. `τ_hi`'s full-sample threshold is a small but real hole in the OOF contract the design states. And the net-item currency should not be softened against C09's own authorising entry on an argument about a different quantity.

None of this requires a new measurement, a new instrument, or a GPU. Fix the §9 clause, put the twins and the null on the right objects, declare stratum occupancy, freeze the sentinel and `FIXK`'s "best", re-open the currency choice, and correct the three provenance items — and this becomes what it claims to be: a `$0` pregate whose pre-declared most-likely outcome, band B, is written in before the run and is a complete and valuable result.


===== RAW REVIEW: refine-logs/C09_A0_V4_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v4 — Independent Design Review, Round 4

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V4_RECORD.md`
**Verdict.** `REVISE — 1 Critical / 4 High / 9 Important`

---

## Prior-round audit

I re-checked every R1, R2 and R3 finding against v4 itself, not against §12.

**Round 1 (4C/8H/10I) — discharged, and it holds in v4.** C-1: `pred_purity` is defined against the *predicted* class (§4.4) and is label-blind. C-2: the discriminator is conditional (`AUC_strat`) and incremental (`ΔAUC`), with a three-hypothesis table, and round-1's "conversion-equivalent AUC threshold" is still correctly refused in favour of `π* = (1+bar/k)/2` — which I re-derived as the exact inversion of `net = k(2π−1)`. C-3: the arena fold assignment is a function of the label vector alone (`headspace_mint.py:204`, exact), so folding by arena fold groups all three seed-rows; the bootstrap unit is the item. **But PERM-STRUCT reintroduces the row as a resampling unit — see H-2.** C-4/H-3/H-5/H-8: net accounting, the vote scale against `mechfix_ops.py:94-95`, the currency adjudication and the real `headspace_fidelity.py` read are all present and correct.

**Round 2 (1C/8H/10I) — the six partials R3 recorded are closed, with two exceptions.** C-1's blanket sentence is properly rewritten (§5.2's box now says *scored item's own* gold label + target-derived arrays, and concedes `bank_label_reads` as a nonzero legal integer — which is the true state of this arena). H-4's over-claim is withdrawn in full. H-6's null is replaced. H-7 is discharged by §9's publication precondition — **but it over-corrects (H-4 below)**. I-1's band A′ lands, and I verified the four bands do partition completed runs. I-6's mint layout is now correct: `headspace_drive.sh:20` and `headspace_mint.py:285-286` confirm `.npz` at the scratch root and `run_rac` trees under `<scratch>/mint/`. I-8's budget arithmetic is fixed (`200 × 2 × 5 × 2 × 2 = 8 000`, twice). **H-5 (`K-DEG`) is still not discharged** — the twins moved to the right scale but the agreement *statistic* is still the wrong object (Critical).

**Round 3 (1C/8H/11I).** C-1 ✓. H-1 partial (Critical). **H-2 — the deliberate deviation: correct in diagnosis, regressive in execution (H-2 below).** H-3 ✓ (`STRATUM_OCCUPANCY` added; thresholds unjustified — I-9). **H-4 ✗ NOT discharged (H-1 below).** H-5 ✓ fully. H-6 ✓ (`37.2 / 29.0` verbatim from the registry `bar` field) — but its arithmetic consequence is undeclared (H-3) and its worked figure is wrong (I-1). H-7 ✓ (sentinel frozen at 21 + `FEATURE_DEGENERACY`). H-8 ✓ (`GATE-FIXK20`), reason misstated (I-3). I-1…I-11 ✓ except I-3's re-scoping (I-2) and I-7's residuals (I-4). **I-11(c) verified against `sacct`:** job `13738` = `c01_a0_v4`, 8 CPU/32 G, `00:06:05`, COMPLETED, and `TARGET_FINDINGS.md:52` records it a valid scientific KILL; job `13805` = `c04_a0t_small_v1_v5_preflight`, FAILED, and `serial_execution.current_design_boundary` still reads `C04_IMPL_V5_CPU_PREFLIGHT_ENGINEERING_HALT_JOB_13805_V6_REPAIR_REQUIRED`. The "three C01 runs" deletion is correct.

---

## Verified as sound (do not re-litigate)

**Every quotation I opened is exact at its cited location**, with the exceptions under IMPORTANT. Verbatim-verified: `unified_pilot_gate.stage_0_reachability`; `banned_constraints[10]` (`directions_tried.json`); `gate0_reopen_2026_07_31.dispositions.promoted.bar` and `.sequencing` and `.supporting_evidence_verified` and `.c09_next_step`; `strategic_finding.consequence_for_gate_0`; F47's, F66's, F75's, F96's, F97's and F98's `ban_scope` (F97's `HONEST POSITIVE DATUM` with `+0.0269 / +0.0104 / +0.0182` and its p-values exact; F98's (b) and (d) exact; F96's standing-gate sentence exact); the F98 epitaph and `+0.1492/+0.1520/+0.2186 → +0.0134/−0.0069/+0.0000`; `GATE0_REOPEN_2026-07-31.md:1004-1006`; `progress.json:25`; `LITSWEEP3_DATA_CENTRIC.md:82` (with the Wall-A parenthetical), `:91`, `:95`; `LITSWEEP5_COMPLETENESS.md:125`, `:127`, `:128` and §2's `:84`; `HEADCOV_PREGATE_RECORD.md:305-310`; `NCA_FORENSIC_RECON.md:110`, `:112`; `ERRPAT_HateMM:126/130/133/134/135/141`; `ERRPAT_MHC-ZH:220/233-235/237-240`; round 14's *"indistinguishable is the exact reading"*; DET-1…DET-4 and CAL-0…CAL-5. **`HEADSPACE_TRANSFER_PREGATE.md:920-923` is now quoted with all three clauses including `±0.0093 / ±0.0280` — R3 I-3's main leg is genuinely repaired.**

**Every number re-derived from source.** All twelve `GATE-FLOOR` floors are exact against `result.acc_deployed` / `result.mF1_deployed`. `fold_acc_deployed` arrays present for all 30 fold-cells. `posrate_bank 0.4005 / 0.3109` → majority `0.5995 / 0.6891` → `GATE-ARENA` bands `[0.6195, 0.98]` / `[0.7091, 0.98]`. `raw_deployed_acc 0.8441 / 0.8480`. `B_fid_abs_3seedmean 0.0093 / 0.0086`, `STOP_RULE_TRIGGERED: false`. `22.3/17.4 = 0.030 × 744/579`; `37.2/29.0 = 0.050 × 744/579`. `π* = 0.7325 / 0.6394` at `k = 80`. Per-seed error counts `83/85/85` and `62/64/61` reproduce exactly from the floors (so §6.3's `83–85` / `61–64` and §3.3's `0.1054–0.1142` are right). ERRPAT §7's six cluster ceilings `+0.0326 / +0.0140 / +0.0233 / +0.0233 / +0.0233 / +0.0093` with their `n` are exact, and `1 test item = 0.00465` ⇒ `n = 215`. F113's head-space `THRESH_best +0.0041` ≈ 3 items on 744 ≈ 1 per 244 ✓. I re-measured the data defects myself: MHC-ZH `243/579` `<em`, `141/243 = 0.5802` vs `39/336 = 0.1161` against base `180/579 = 0.3109`; HateMM `39/744` whitespace-only, `0` on ZH, corroborated by `C02_A0_OUT.json` `VIEW_SUPPORT.degenerate_causes.EMPTY_TEXT = 39`. `GATE-NULL` re-measured from the frozen caches: HateMM zero-img `[355]`, zero-txt `[355]`, `ids[355] = hate_video_95`, `labels[355] = 1`; MHC-ZH `[]`/`[]`.

**Executability is real, and I verified the sha chain.** On-disk `headspace_mint.py` = `cefdf8dc…` = the banked `meta.mint_script_sha256`; `headspace_arena.py` = `761b256b…` = banked `script_sha256`; `mechnov_pairverify.py` = `77b0defd…` = `FROZEN_PAIRVERIFY_SHA` at `headspace_mint.py:123`; `mechfix_ops.py` = `635c1312…` = the banked frozen entry. All four match, so the mint will not self-HALT. `headspace_mint.py:106-116/192-194/203-216/274-281/285-286/322-323` are exact to the line; `headspace_fidelity.py:66` is literally `mint_{}_s{}_ffull.npz`; `headspace_arena.py:7` and `mechnov_pairverify.py:124` carry the raw-space definition; `CLI` has exactly `{hatemm, zh}`. `GATE-FIXK20` will not spuriously VOID: `k′=20` with profile `[k..1, 0…]` over the deployed top-20 *is* `[20..1]`, and the banked `degeneracy.B_agree_fixk["20"]` equals `agree_deployed` on both datasets (`0.9919` / `1.0`), consistent with `FIXK_20 ≡ deployed`.

**The budget is sound and conservative — I timed it.** At the design's own scale (≈2 190 rows × 12 features, 4/5 fitting), one `lbfgs` fit + OOF score is **4.1 ms**, so 20 000 `PERM-STRUCT` refits cost **1.4 min**, not the 12 min budgeted; the GBM arm is **0.10 min**, not 3. A vectorised stratified AUC is **1.17 ms/cell**, so `B = 10 000` item-bootstrap re-scorings cost **≈4.7 min**. C02's 36 banked mints total **24.5 min** (min `33.2 s`, max `60.0 s`, median **`41.85 s`** — exact), and `sacct` confirms job `13847` = 8 CPU / 32 G / `00:29:49` / COMPLETED. Realistic total ≈45 CPU-min against the declared ≈70. **The ≈70-minute figure is achievable with slack.**

**Legality holds and I could not break it.** Both blessed texts are exact; `H-L1`…`H-L4` are the right boundaries and `H-L3`'s restatement matches `GATE-LEDGER`'s predicate; §3.1's F47 adjudication confines the F114 correction to F47's *train-supervised* leg without touching the decision-level-meta-features sentence; §3.3 answers LITSWEEP5's independent leg with a measurement (`0.1054–0.1142` vs the saturated arena's `0/109, 0/102, 0/92`) and keeps the residual headwind; §11's four-part Stage-1 precondition and its void-if-no-operator clause are genuine discipline. **No test path or test label can reach any decision quantity** (`torch.load` guard at `:111` catches both `test_seen` and `/test`), and I found no path by which a scored item's own gold label reaches its own *features* — the only surviving path is through the *target* construction at `τ_hi`, which is H-1.

---

## CRITICAL

**C-1 — `K-DEG` applies the campaign's `0.95` degeneracy line to a statistic it was never calibrated on, and the mis-match is large enough that an operator agreeing with a bare threshold shift on three-quarters of its selections passes the gate.** §6.2 defines agreement as *"`|S ∩ twin| / k` with no `min()` ambiguity"* — **selected-set overlap** — and fires at `≥ 0.95`, asserting *"The `0.95` line is the campaign's own."* It is not. F96's own number is a fraction of **items**: *"C1 agrees with a PURE GLOBAL THRESHOLD SHIFT … on 95.03% / 97.75% / 99.45% of items"*. F98's DEG-A is `(c3 == coll["THRESH_best"]).mean()` — prediction agreement over all `n` — giving `0.9570` (HateMM) / `0.9508` (EN), and DEG-B is the same construction (`B_agree_fixk`, which I read directly out of the banked arena JSONs). The two statistics are related by `pred_agree = 1 − 2k(1−ov)/n`. At HateMM's `n = 744` with `k = |P_0| ≈ 76`, `pred_agree = 1 − 0.204(1−ov)`: **F98's `0.95` line corresponds to set-overlap `ov = 0.755`, while v4 requires `ov ≥ 0.95`, i.e. `pred_agree ≥ 0.9898`.** An `S` that is 76 % identical to `THRESH-SYM` — the exact "dead lever in an item-level costume" F96 made a standing gate against, and the failure mode this family has now collapsed into twice — passes v4's `K-DEG` and can carry a CONTINUE. The v4 repair of R3 H-1 (twins on both scales, maximum taken) moved the gate onto the right *scale* and left it on the wrong *statistic*, so the campaign's only degeneracy screen is disabled at exactly the operating point where it matters. **Repair:** compute the induced prediction-vector agreement `1 − |S Δ twin| / n` for each twin on each scale, apply the frozen `0.95` line to **that** quantity (F96/F98's own object), and report `|S ∩ twin| / k` beside it as a descriptive figure; or, if set-overlap is preferred, re-derive the line explicitly (`ov* = 1 − (1−0.95)·n/(2k)`, per dataset and per `k`), state that the campaign constant has been re-mapped, and record the mapping in the freeze record. Do not leave `0.95` attached to set-overlap while claiming it is the campaign's line.

---

## HIGH

**H-1 — R3's H-4 is not discharged: the per-fold `τ_hi` closes the scored item's *self*-target path and leaves open the path R3 actually named — the scored item's gold label still reaches the *fitting rows'* targets — and §4.3 asserts the leak has been removed.** §4.3 defines `τ_hi^(f) = median(c_i : i ∈ P_0, fold(i) ≠ f)` and `P_{τ_hi} ≡ {i ∈ P_0 : c_i ≥ τ_hi^(fold(i))}`, i.e. **one global positive set in which every item is labelled by its own fold's threshold**. Now take the model that scores fold `f`. Its fitting rows are items `j ∈ fold g ≠ f`, and `j`'s target is `c_j ≥ τ_hi^(g)` — and `τ_hi^(g)` is the median over `P_0 \ fold g`, which **contains fold `f`, hence contains item `i`**. So `i`'s `P_0` membership still perturbs the thresholds that define the positive class for every fitting row of the model that scores `i`. R3 H-4 stated exactly this path, and R3's repair (a) put **one** threshold per *scoring* fold on **all** rows in that fit; v4 instead put one threshold per *item's own* fold, which is a different construction. The magnitude is unchanged at `O(1/|P_0|) ≈ 1.4 %`, and the direction is toward CONTINUE. §4.3's sentence *"for fold `f`, both the threshold and the fitting rows exclude fold `f` entirely, and the scored item's own label enters only as its own target"* is false as written, and `GATE-NESTED`'s new clause is true of the scored item and silent on the fitting rows, so the gate cannot detect it. **Repair:** decouple the two uses. For the model that scores fold `f`, label **every** row in that fit — fitting and scored — with `τ_hi^(f)`, and assert this in `GATE-NESTED` as a per-item check. Keep the global `P_{τ_hi}` (each item at its own fold's threshold) only for `|P_τ|`, `K-REACH`, `O1` and `NET`'s `k`, and say in §4.3 that the *population* and the *fitting target* are two objects computed two ways, with the reason.

**H-2 — the deliberate deviation on R3 H-2 is right about the hypothesis and wrong about the permutation unit, and `PERM-STRUCT` is declared "the exact null" for a conditional hypothesis for which it is not exact.** Two separate defects. **(i) Unit.** §5.2 permutes the five structural columns *"jointly across rows"* under a single global row permutation. Rows are `(item, seed)`; the target is per item; and round-1 C-3's repair — which v4 keeps everywhere else, including the item-level bootstrap and its explicit `√3` argument — established the item as this design's resampling unit. A row-level permutation gives each item's three seed-rows three unrelated structural vectors (and mixes rows across `(dataset, seed)` cells entirely), so the permuted noise averages out over `3n` effectively independent rows instead of `n`. The fitted null coefficients shrink toward zero, the null `ΔAUC` distribution is **narrower than it should be**, and the resulting `p` is **too small** — anti-conservative, in the CONTINUE direction, by the same mechanism round 1 rated Critical. **(ii) Exactness.** §5.2 states *"Permuting the structural block alone is the exact null for 'the structural block adds nothing'."* A naive permutation of a covariate block is exact only for the **marginal** null `structural ⊥ (target, BASE)`. `K-FELDMAN`'s stated hypothesis is the **conditional** one, and the two coincide only when the blocks are independent — which they emphatically are not here: structural feature 1 (first-differing-label rank, sentinel 21) is nearly a deterministic function of `pred_purity`, which is in `BASE`. Under misspecification of the linear logit — the exact possibility the GBM capacity arm exists to probe — a conditionally-uninformative but correlated block can raise OOF AUC, and `PERM-STRUCT` will reject. The design cannot then distinguish "adds conditional information" from "re-encodes `BASE` better". **Repair:** (a) permute at the **item** level (one donor item supplies all three of an item's structural rows) — or, better, within each `(dataset, seed)` cell — and state the unit; (b) delete "the exact null" and replace with an explicit marginal-vs-conditional statement, then scope `K-FELDMAN`'s CONTINUE language to what the test actually rejects; or (c) replace it with a conditional randomisation / residual-permutation scheme that preserves `structural | BASE`.

**H-3 — the frozen `2 τ × 3 k` grid, offered as the design's forking-path control, contains cells that cannot pass by arithmetic under the new `37.2 / 29.0` currency, and this is undeclared while §4.3 and §6.3 extend exactly this courtesy to `τ_hi` and `UNSTABLE-POP`.** `mean_s net_s = 2·mean_s|S ∩ wrong_s| − k ≤ 2·mean_s|wrong_s| − k`, and the banked floors fix `mean_s|wrong_s| = 84.33` (HateMM, from errors `83/85/85`) and `62.33` (MHC-ZH, from `62/64/61`). So **`K-NET` is unreachable for any `k > 131.5` (HateMM) or `k > 95.7` (MHC-ZH), and for any `k < 37.2 / 29.0`, whatever the selector does.** With the design's own transferred F88 rate (`|P_0| ≈ 76 / 55`): at `τ_0`, `k = 2|P_τ| ≈ 152 / 110` is **arithmetically dead on both datasets**, and `k = 1.5|P_τ| ≈ 114 / 83` survives only at recall `≥ 0.89 / 0.88` *and* precision `≥ 0.66 / 0.68` simultaneously. At `τ_hi` (`|P_{τ_hi}| ≈ 38 / 28`), `k = |P_{τ_hi}|` needs precision `0.99` on HateMM and is **strictly impossible on MHC-ZH whenever `|P_{τ_hi}| < 29`**. §5.4 rests its multiplicity argument on the grid being *"frozen, exhaustive and reported in full"*, and §9 promises every quantity at all three `k` — but a KILL written "at every `k`" would read as three tested operating points when one or two could never have passed. The direction is conservative, so no verdict is wrong; the **scope** of the KILL and the honesty of the grid are. **Repair:** add to §5.3, beside the currency adjudication, the closed-form caps `k ≤ 2·mean_s|wrong_s| − bar` (`131` / `95`) and `k ≥ bar`; pre-declare which of the six `(τ, k)` cells are arithmetically dead under the expected `|P_τ|`; and require the KILL record to name the live cells rather than say "every `k`".

**H-4 — §9's publication precondition over-corrects R3's C-1: a purely *data-driven* degeneracy is routed to a no-verdict HALT with a "diagnose-repair-resubmit path" that does not exist, and the design's own cited marginals make that outcome plausible.** §5.2: *"if in some cell every stratum is single-class, `AUC_strat` is undefined there and the run **HALTs** with that cell named"*; §9 then makes "no §5.2 degeneracy HALT fired" a precondition for publishing **any** verdict, and characterises a HALT as *"an engineering result with a diagnose-repair-resubmit path"*. But single-class strata are a property of the data under frozen strata, not of the harness: `ERRPAT_HateMM:130/133` gives `|vote|` `0.7267` vs `0.9873` and rank-weighted purity `0.1667` vs `1.0000`, `ERRPAT_MHC-ZH:220` gives `0.15` vs `1.00`, and `:237-240` says the head space is *purer still* (`0.9833` for correct items) — the design cites all of this itself to justify `STRATUM_OCCUPANCY`. One degenerate cell out of the twelve `(dataset, seed, τ)` cells — most likely at `τ_hi` — converts a completed, fully-informative `$0` run into **no verdict at all**, with nothing to repair and nothing gained by resubmitting the identical frozen design. **Repair:** demote the all-single-class case from a HALT to a data outcome: drop the affected cell from the seed-mean with the drop emitted, force `IDENTIFIABILITY_UNDERPOWERED` on that dataset, and let §9 publish a KILL scoped as *"not identifiable at this power"*. Reserve HALT status for the eight instrument gates plus `SHUFFLE-POP`'s band, all of which genuinely have a repair path.

---

## IMPORTANT

**I-1 — §5.3's decisive worked figure contradicts the two numbers printed beside it.** *"At `k = |P_0| ≈ 80` the two bars differ by ~7.5 points of required precision (`π* = 0.7325` vs `0.6394`)"*. Both `π*` values are exactly right (`(1+37.2/80)/2 = 0.7325`; `(1+22.3/80)/2 = 0.639375`), and their difference is **`0.0931` = 9.3 points, not 7.5** (7.5 points is the gap at `k = 100`). The error is inherited verbatim from R3 H-6 and was not re-derived. **Repair:** state `~9.3 points`, or give the `k` at which `7.5` holds.

**I-2 — §6.1 re-scopes F113's "9 of 9" into a claim F113 does not make, and the error weakens a headwind that runs against C09.** v4 writes *"9 of 9 raw-space **operator cells** failing to transfer to head space — **all nine descending from F105/VSW**"*. `HEADSPACE_TRANSFER_PREGATE.md:858-863` enumerates them: `VSW_pow`, `VSW_exp`, `VSW_lin`, **`THRESH_best`**, **`CTRL_cos_pow`**, **`FIXK_15`**, **`FIXK_10`**, **F95 `mlp_mean3`**, and the λ-oracle ceiling — four distinct operator families, not one lineage. `:917-919`'s *"Established on ONE cell only … F105/VSW is the only raw-space positive the campaign ever produced"* is a statement about campaign-level raw positives, not about the nine arms. **Repair:** "9 of 9 raw arms that scored positive in the raw arena fail to transfer (`:858-863`); F113 separately records that only one of them, F105/VSW, was ever a campaign-level raw-space positive, so as a *transfer* result it is established on one cell (`:917-919`)."

**I-3 — §2 misdescribes CAL-2 leg (2)'s reference arena, so the stated reason for out-of-scoping it is wrong, and the leg is available at zero marginal cost.** v4: *"it is a Spearman against F94's banked **raw**-arena k-curve (`ksweep_OUT.json`), and this arena is not that arena"*. `PREGATE_CALIBRATION_CLAUSE.md:80-81` reads *"the Spearman against F94's banked **deployed** k-curve (`scripts/analysis/ksweep_OUT.json`; primary arms HateMM final / MHC-ZH final / MHC-EN ARM-V) over `k ∈ {5,7,10,15}` ONLY"* — the comparator is the **deployed** curve, which does not change with the arena. §6.2 already computes `FIXK` at `k′ ∈ {1,2,3,5,7,10,15}`, so the required grid exists. The quoted fragments *"deliberately no threshold"* and *"not as a validity gate"* are both verbatim and the non-gating conclusion is defensible; only the reason is wrong. **Repair:** correct "raw" to "deployed", and either report the Spearman (free) or say it is skipped because CAL-2 is scoped by its own header to *"the raw banked train-space arena"*.

**I-4 — two frozen quantities in `K-DEG`'s `FIXK` twin are still undefined, in the control that gates the verdict.** (a) `|Δscore|` appears three times in §6.2's size contract and is never defined — presumably `score(k′) − score(20)`, but a preregistration must say. (b) `best` is pinned as *"the `k′` maximising `|S ∩ flip(k′)| / k`"* with **no tie-break**, and the banked degeneracy shows `flip(k′)` can be empty in this space (`B_agree_fixk["15"] = 1.0000` and `agree_deployed = 1.0000` on ZH seed 0, i.e. `FIXK_15 ≡ deployed`), so all seven ratios can be `0` and the argmax is undefined; the twin then becomes pure `|Δscore|` padding, which is a threshold construct wearing the `FIXK` label. **Repair:** define `Δscore` explicitly; add "ties broken by the smallest `k′`"; and emit `|flip(k′)|` per cell so a reader can see when the twin is padding rather than flipping.

**I-5 — §9 tags a CONTINUE `ROBUST` or `POINT_ESTIMATE_ONLY` and the predicate is never defined anywhere in the document.** The only occurrence is at §9's tagging bullet. Since the item bootstrap is explicitly *"conditional on the fitted model"* and *"not `K-FELDMAN`'s p-value"*, the rule cannot be inferred. **Repair:** freeze it, e.g. "`ROBUST` iff the one-sided item-bootstrap lower bound on `ΔAUC` is `> 0` on both datasets at the CONTINUE's `τ`, else `POINT_ESTIMATE_ONLY`", and restate that the interval is fit-conditional.

**I-6 — §3.2(1)'s "The banned object is not built" contradicts §5.3, which builds and fits it.** §5.3 fits the `FULL` logistic probe, ranks **all `n`** query items by its OOF probability, takes the top `k`, and costs the flips — that is F98(b)'s object, constructed and measured. What is *not* done is deploying it, which `H-L4` forecloses. Calling it *"an accounting instrument applied to an **idealised** operator"* understates what runs. The legality conclusion is unaffected, but the wording is the kind that a later reader would rely on. **Repair:** "The banned object is **constructed and measured as a pricing instrument on the train arena** and is never deployed, never consulted at query time, and foreclosed as a component by `H-L4`."

**I-7 — §9's raw-arena leg says "the identical battery" is recomputed, but the battery is not identical-able there and its inferential cost is unbudgeted.** The raw space is seed-free (`headspace_arena.py:7`), so there are no `(dataset, seed)` cells to average `AUC_strat` over, `K-DEG`'s two-scale maximum collapses to one scale, and — as §9 itself concedes — stability is undefined. Meanwhile every permutation line in §2's table is scoped `× 2 ds`, not `× 2 spaces`, so the raw leg's `PERM-STRUCT` / `SHUFFLE-POP` / `RANDOM-POP` / bootstrap fits are outside the ≈70 minutes. I measured this: at raw scale it adds only ≈5–8 CPU-min, so it is **not** budget-breaking — but the table should say so, and the raw leg's degenerate cases should be specified. **Repair:** state which quantities the raw leg computes and which are undefined there, and add the raw-leg line to the budget table.

**I-8 — provenance and specification nits (bundle).** (a) `headspace_drive.sh:20` and `c02_a0_cpu_v9.sbatch:85` are `out=` / `OUT=` **assignments**; the `--out` flag is at `:24-25` and `:89`, and `c02_a0_cpu_v9.sbatch` invokes `c02_a0_mint.py`, not `headspace_mint.py` — the layout precedent is right, the citation is not. (b) §3's `H-L3` says `headspace_mint.py:322-323` stores `lab_dev` *"in every `fold == -1` mint"*; the `np.savez` at `:322-323` stores it in **every** mint, folds `0–4` included — a narrower claim than the truth, so `GATE-LEDGER`'s expected dev-materialisation count should be set accordingly. (c) §5.2 attributes *"cosine saturated at ~`0.9999`"* to `ERRPAT_HateMM:131`; `:131` carries `0.999852 / 0.999976` and the `~0.9999` characterisation is at `:139-140`. (d) §0 calls `STRATUM_OCCUPANCY` one of *"two further gates"* while §5.2 and §7 correctly make it a non-gating emission, and §2/§5.6/§8.1 alternate between `FIXK_20` and `GATE-FIXK20`. (e) `GATE-FLOOR`'s *"6/6 seeds"* is six `(dataset, seed)` cells, not six seeds. (f) F75's *"First measured negative…"* is re-cased to lowercase inside quotation marks.

**I-9 — `STRATUM_OCCUPANCY`'s frozen thresholds carry no power justification, and the tag they raise does not block a CONTINUE.** `Σ n_pos·n_neg < 200` is a very weak floor: at `n_pos = 10, n_neg = 20` (`Σ = 200`) the Hanley–McNeil standard error of `AUC` at `A = 0.5` is **`0.114`**, an order of magnitude above the `ΔAUC` effects the design is looking for, so a dataset can clear the tag and still be unable to resolve the primary. Conversely, with `|P_0| ≈ 76` positives and `≈ 654` negatives per cell, `Σ n_pos·n_neg` will be in the thousands unless separation is near-total, so the tag will rarely fire and the rule may be close to inert. And §9 lets a CONTINUE be published *with* `IDENTIFIABILITY_UNDERPOWERED` present, merely tagged. **Repair:** justify the floor against a target `ΔAUC` resolution, and state explicitly whether a CONTINUE carrying the tag is permitted — and if so, why.

---

## Bottom line

The instrument is real and I verified it end to end: four sha256s match their banked anchors, all twelve floors and thirty fold arrays are exact, C02's independent re-mint reproduces the pooled accuracies to full precision, and my own timing at the design's scale puts the run at ≈45 CPU-minutes against a declared ≈70. Legality is the strongest part of the document. R1 is fully discharged, R2's six partials are closed except `K-DEG`, and R3's C-1, H-3, H-5, H-6, H-7, H-8 and I-1…I-11 all land, with I-11(c) verified against `sacct`.

What must change is narrow and, as before, sits on *objects* rather than on substance. The Critical is that `K-DEG` carries F96/F98's `0.95` constant on a statistic that is systematically ~0.18 lower at this `k/n`, so it fires only at `pred_agree ≥ 0.99` and would let a three-quarters-threshold-shift through. Of the Highs, R3's `τ_hi` finding is not discharged but relocated. The deliberate H-2 deviation is right that a target permutation tests the wrong (joint) null and wrong to permute at row level and to call the result exact for a conditional hypothesis — both errors lean toward CONTINUE. And two arithmetic facts the design already had in hand are undeclared: the new `37.2 / 29.0` currency makes `k = 2|P_τ|` unreachable on both datasets, and a plausible data outcome is routed to a no-verdict HALT with no repair path.

None of this needs a new measurement, a new instrument, or a GPU.


===== RAW REVIEW: refine-logs/C09_A0_V5_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v5 — Independent Design Review, Round 5

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V5_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 3 High / 10 Important` *(the first round with no Critical)*

---

## Prior-round audit

I re-checked every R1–R4 finding against v5 itself, not against §12.

**Round 1 (4C/8H/10I) — discharged, and it still holds.** C-1: `pred_purity` is against the *predicted* class (§4.4). C-2: the discriminator is conditional (`AUC_strat`) + incremental (`ΔAUC`) with a three-hypothesis table, and `π* = (1 + bar/k)/2` is the exact inversion of `net = k(2π−1)` (re-derived). C-3: the arena fold assignment is a function of the label vector alone — `headspace_mint.py:204` is literally `skf.split(np.zeros((n,1)), lab)` — so folding by arena fold groups all three seed-rows; the bootstrap unit is the item; **and `PERM-STRUCT`'s row-level unit, which R4 reopened as H-2(i), is now the item.** C-4/H-3/H-5/H-8: net accounting, the vote scale against `mechfix_ops.py:94-95`, the currency adjudication and the real `headspace_fidelity.py` read are all present and correct.

**Round 2 (1C/8H/10I) — closed, including R4's one surviving exception.** Feature 11 stays deleted; the `(88 %)` interpolation is gone from §4.2's F88 quote; H-7's gate/decision split holds and §9 no longer over-corrects; I-6's mint layout is verified on disk (36 `.npz` at the C02 scratch **root**, 36 `run_rac` trees under `<scratch>/mint/`); I-8's budget arithmetic is right on every line (`1000×1×5×2×2 = 20 000`, `200×2×5×2×2 = 8 000` twice). **R2 H-5 / R4 C-1 (`K-DEG`) is now genuinely discharged.**

**Round 3 (1C/8H/11I) — all discharged.** C-1 ✓ (§9's publication precondition, and §9's `SHUFFLE-POP` predicate now reads "at both `τ` on both datasets", matching §6.3). H-1…H-8 ✓. I-11 in full: (c) "three C01 runs" is deleted, (d) the runtime **quintet** is right (`python/numpy/scipy/sklearn/torch` — I read `meta.runtime.versions`), (e) `q25`/`q75` are deleted with a note, (g) `numpy.median`'s linear interpolation is pinned.

**Round 4 (1C/4H/9I).**
- **C-1 — DISCHARGED, and the arithmetic is correct.** `pred_agree = 1 − |S △ twin|/n = 1 − 2k(1−ov)/n` is exact for `|S| = |twin| = k`. At `n = 744, k = 76`: `2k/n = 0.20430` ✓, so the `0.95` line maps to `ov = 0.7553` ✓ and v4's `ov ≥ 0.95` demanded `pred_agree = 0.98978` ✓. `aggnet_pregate.py:534` is literally `(c3 == coll["THRESH_best"]).mean()` and `:537` is the same construction for `B_agree_fixk`, so v5 is now on F96/F98's own statistic.
- **H-1 — DISCHARGED on the path R4 named.** Under (b) the model that scores fold `f` uses one threshold `τ_hi^(f)` computed over `P_0 \ fold f`; the `τ_hi^(g)` route is closed. (A *different*, arena-intrinsic route survives — Important I-4 below, not a re-opening of H-1.)
- **H-2 — (i) and (ii) DISCHARGED; (c) ATTEMPTED BUT INCOMPLETE.** The item-level unit and the withdrawal of "the exact null" are correct and correctly reasoned. The **deliberate deviation is the right call**: a target permutation tests the joint null, and adding a residualised conditional null that a CONTINUE must *also* clear is strictly stronger than switching. But the residualisation is **linear**, and the dependence the document itself names is a step function — HIGH H-1 below. This is a regression in execution, not in design intent.
- **H-3 — DISCHARGED.** `mean_s|wrong_s| = 84.33 / 62.33` re-derived exactly from the banked floors (`661/659/659` of 744; `517/515/518` of 579); caps `131.47 / 95.67` ✓; `k = 2|P_τ|` dead on both ✓; `k = |P_{τ_hi}|` impossible on ZH below 29 ✓.
- **H-4 — DISCHARGED.** HALT → data outcome, `IDENTIFIABILITY_UNDERPOWERED` forced, KILL scoped "at this power"; §8.1's "nine HALT gates + `SHUFFLE-POP`" count is self-consistent with §9.
- **I-1 ✓ 9.3 points** (`0.7325 − 0.639375 = 0.093125`; 7.5 is the `k = 100` gap ✓). **I-2 ✓ split** (line-range nit below). **I-3 ✓** `PREGATE_CALIBRATION_CLAUSE.md:80-81` is verbatim "**deployed** k-curve". **I-4 ✓** `Δscore`, tie-break, `|flip(k′)|`, `DEGENERATE_ALL_EMPTY`. **I-5 ✓** `ROBUST` frozen. **I-6 ✓** reworded. **I-7 ✓** raw-leg line + three declared changes. **I-8 (a)–(f) ✓** all six verified on disk. **I-9 ✓** Hanley–McNeil is exactly right: `sqrt((0.25 + (p+q−2)/12)/(pq)) → sqrt(1/(12p))`, `p=30 ⇒ 0.05270`, `p=10 ⇒ 0.09129`.

---

## Verified as sound (do not re-litigate)

**Every quotation I opened is exact at its cited location**, with the line-range exceptions in I-5. Verbatim-verified against source: `stage_0_reachability`; `banned_constraints[10]`; `gate0_reopen_2026_07_31.dispositions.promoted.{bar, sequencing, supporting_evidence_verified}` and `.c09_next_step` and `new_status`; `strategic_finding.consequence_for_gate_0`; F47's, F66's, F75's (both clauses, including the upper-case "First measured negative…" and the reopen's lower-case rendering), F96's, F97's and F98's (b)+(d) `ban_scope`; F96's `95.03% / 97.75% / 99.45% of items`; the AGGNET/F98 epitaph and its delivered figures; `progress.json:25`; `LITSWEEP3_DATA_CENTRIC.md:82/:91/:95`; `LITSWEEP5_COMPLETENESS.md:125/:127/:128` and §2's `:84`; `HEADCOV_PREGATE_RECORD.md:305-310`; `NCA_FORENSIC_RECON.md:110/:112`; `ERRPAT_HateMM:126/130/131/133/134/135/139-141` and §7's six cluster ceilings; `ERRPAT_MHC-ZH:220/233-235/237-240`; `HEADSPACE_TRANSFER_PREGATE.md:917-919` and `:920-923` **with all three clauses including `±0.0093 / ±0.0280`**; `PREGATE_CALIBRATION_CLAUSE` CAL-0/CAL-2(1)/CAL-2(2)/CAL-3/CAL-5; `PREGATE_DETERMINISM_CLAUSE §1.3`'s `auc_logistic` 4-dp invariance.

**Every number re-derived from source.** All twelve `GATE-FLOOR` floors exact; `fold_acc_deployed` present at 4 dp for all 30 fold-cells; fold sizes; `posrate_bank 0.4005/0.3109` → `GATE-ARENA` bands `[0.6195, 0.98]` / `[0.7091, 0.98]`; `raw_deployed_acc 0.8441/0.8480`; `B_fid 0.0093/0.0086` with `STOP_RULE_TRIGGERED: false`; F98's `DEG_KILL = 0.95`, `A_agree_threshold_shift = 0.957 / 0.9508`, `THRESH_best d_acc = 0.0188` vs C3's `+0.0134`; ZH seed-0 `B_agree_fixk["15"] = 1.0 = agree_deployed`; `22.3/17.4` and `37.2/29.0`. I re-measured the data defects myself: MHC-ZH `243/579` `<em`, `141/243 = 0.5802` vs `39/336 = 0.1161` against base `180/579 = 0.3109`; HateMM `39/744` whitespace-only, `0` on ZH. `GATE-NULL` re-measured: HateMM zero-img `[355]`, zero-txt `[355]`, `hate_video_95`, label `1`; MHC-ZH `[]`/`[]`.

**Executability is real and I verified the whole chain.** Four sha256s match their banked anchors. `headspace_mint.py:106-116 / 192-194 / 203-216 / 274-281 / 285-286 / 322-324` exact to the line; `CLI` keys exactly `['hatemm','zh']` (AST-parsed); `headspace_fidelity.py:66` literal; `mechnov_pairverify.K_FOLDS = 5` at `:56`, `FOLD_SEED = 0` at `:57`; `headspace_drive.sh:20 / :24-25` and `c02_a0_cpu_v9.sbatch:85 / :89` are assignments-then-flags exactly as §2 now says. **The live environment matches the banked `meta.runtime` exactly** — `python 3.11.8 / numpy 1.26.4 / scipy 1.17.1 / sklearn 1.5.2 / torch 2.6.0+cu124` on `foscsmlprd01`, a single-node cluster — so `GATE-FLOOR` at 4 dp is achievable and `RUNTIME_DRIFT` will not fire. `lab_dev` is present in **all 36** banked mints, so `GATE-LEDGER`'s expected count of 36 is right. `sacct` confirms job `13847` = 8 CPU / 32 G / **no GPU** / `00:29:49` / COMPLETED.

**The budget is sound.** The 36 banked C02 mints ran `33.2–60.0 s`, median `41.85`, total `24.51 min` — exact. Every line of §2's table reproduces from its own factors, and ≈90 min is conservative against R4's measured ≈4.1 ms per `lbfgs` fit-and-score.

**Decidability is real, and I checked the arithmetic in both directions.** `K-REACH` at `τ` forces `|P_τ| ≥ 37.2 / 28.95`, hence `2k/n ≥ 0.10`, so `K-DEG`'s vacuous-fire regime (`2k/n < 0.05`) is **unreachable on the CONTINUE path**. On the live cells the `0.95` line maps to set-overlap `ov ≥ 0.755 / 0.837` (HateMM) and `0.737 / 0.826` (ZH) against a floor `1 − 2k/n` of `0.796 / 0.810` — so the gate can both fire and clear. The four bands partition completed runs. Holm-within-family and the IUT across datasets are correctly specified. `net_s ≡ n·Δacc_s` is an identity. `c_i` is label-blind for `i` (the vote reads only bank labels, and `i` is never in its own bank).

**Legality holds.** No test path can be opened; dev labels are materialised 36× by the instrument's own contract and correctly kept outside every decision quantity. §3.1's F47 adjudication concedes the ban. §3.2's F98 adjudication is **sound and now honest about what runs**. §3.3 answers LITSWEEP5's independent leg with a measurement and keeps the residual headwind. §11's Stage-1 seam is sound. §10 claims nothing the design does not support.

---

## HIGH

**H-1 — `PERM-STRUCT-COND` residualises on `BASE` with OLS, so it does not implement the conditional null it was added to supply, in exactly the dependence the document itself names.** §5.2 justifies the new null as preserving "each structural feature's conditional mean given `BASE`". An OLS fit preserves the *linear projection* onto `BASE`, not `E[struct | BASE]` when the relation is nonlinear — and the relation the document nominates as load-bearing is a step function: structural feature 1 takes the frozen sentinel `21` **iff** all 20 top-20 bank labels are identical, i.e. iff `pred_purity = 1.0`, and `pred_purity` is in `BASE`. Under OLS that censoring lands almost entirely in the residual. Permuting the residual therefore destroys both (i) the block's conditional association with the target — intended — and (ii) the block's *nonlinear* `BASE` content, which is not. Under H0 with a misspecified linear logit (the exact possibility the GBM capacity arm exists to probe, and the exact mechanism §5.2 quotes to withdraw "the exact null"), the observed `FULL` retains that nonlinear `BASE` content and can beat `BASE`-only on OOF AUC while every permuted draw cannot, so `p_cond` rejects under the conditional null. The conjunction `p ∧ p_cond` does not repair this: under a `BASE`-dependent block the marginal null is false anyway, so the conjunction's level is governed by the conditional test, which is the anti-conservative one — in the CONTINUE direction. **Repair:** replace the linear residualisation with a permutation that is exact for the estimand `AUC_strat` already conditions on — permute the structural block **within the frozen 12 strata**, which is an exact permutation null for *"struct ⊥ target | stratum"*; or, if full-`BASE` conditioning is wanted, residualise on a flexible representation of `BASE` and say so. Either way, state which conditioning set `p_cond` is exact for, and scope `MARGINAL_ONLY_NOT_CONDITIONAL` to that set.

**H-2 — the `τ_hi` co-primary is at or past the arithmetic edge of the *new* `p_w ≥ 30` rule, and the interaction with `K-REACH` is undeclared, so `τ_hi` is reported as a tested identifiability cell when the design's own numbers make it unpassable.** `p_w` counts positives lying in weighted strata, so `p_w ≤ |P_τ|` identically. §5.2's own pre-declared expectation is `|P_{τ_hi}| ≈ 38 / 28`; on MHC-ZH `28 < 30` therefore **forces** `IDENTIFIABILITY_UNDERPOWERED`, which R4 I-9's repair defines `K-FELDMAN` to fail on, which by §9's both-datasets conjunction kills the `τ_hi` CONTINUE branch outright — not "plausibly", as §5.2 says, but by arithmetic. Simultaneously `K-REACH` at `τ_hi` needs `|P_{τ_hi}| ≥ 28.95`, so on ZH the two rules jointly require `|P_{τ_hi}| ≥ 30`, i.e. `|P_0| ≳ 60` against a declared expectation of 55. On HateMM the same rule needs `p_w ≥ 30` out of `≈ 38` — ≥ 79 % of positives in two-class strata — against §5.2's own statement that "most strata may be single-class … worst at `τ_hi`". This is precisely the class of fact R4's H-3 forced §5.3 to pre-declare for `K-NET`, and it is left undeclared for `K-FELDMAN`. **Repair:** apply §5.3's treatment to §5.2 — compute `p_w ≤ |P_τ|` as a closed-form cap in-run, mark each `(τ, dataset)` identifiability cell `LIVE` or `ARITHMETICALLY_DEAD_AT_THIS_POWER`, pre-declare that on the transferred rates the `τ_hi` branch cannot produce a CONTINUE on MHC-ZH, and require the KILL record to name it as unreachable rather than as tested-and-failed.

**H-3 — at `τ_hi` the positive class is restricted by a threshold on the confidence scale while `CONFIG-MATCHED-CORRECT` is not, so §5.2's "matching is achieved by the stratification itself" does not hold at the co-primary and `ΔAUC(τ_hi)` is a different estimand from `ΔAUC(τ_0)` — undeclared, and §9 lets a `τ_hi`-only rejection carry a CONTINUE.** Positives at `τ_hi` are `P_{τ_hi} = {i ∈ P_0 : c_i ≥ τ_hi^{fold(i)}}`; negatives are *all* three-seed-correct items, with no `c` restriction. The stable inversions with `c < τ_hi` are dropped from the analysis set entirely. So the contrast at `τ_hi` is "high-confidence inversions vs all correct items", selected on a quantity — `c_i` — that `BASE` carries only as its **per-seed** component and that the strata discretise only into terciles. That is selection on a collider, which distorts the feature–target association inside the selected sample in a direction the design does not determine, and it means the two `τ` hypotheses Holm treats as one family are not two instances of one estimand. §6.1's hypothesis table is stated without any `τ` qualifier, and §9's `∃τ` lets the co-primary alone license Stage-1 spend. **Repair (cheap, and it costs almost no negatives because correct items are the high-`c` class):** apply the same out-of-fold threshold to the negative class; and add `c_i` itself to `BASE` so the definitional quantity is fully absorbed by the baseline. If neither is adopted, state in §5.2 and §10 that `ΔAUC(τ_hi)` is a confidence-restricted contrast, not the `τ_0` object.

---

## IMPORTANT

**I-1 — `PERM-STRUCT` / `PERM-STRUCT-COND` carry three specification gaps, one of them with an anti-conservative branch.** (a) §5.2 says "within each `(dataset, seed)` cell, a permutation `π` of the cell's **items** is drawn" and then "the **same `π`** is applied in all three seed cells" — two incompatible descriptions of the drawing unit. (b) `PERM-STRUCT-COND` leaves undecided whether one `π` is shared across the five features (conservative) or five independent `π` are drawn (anti-conservative). (c) The OLS is fit "on the training folds only", but there are five fold-specific fits per `(τ, dataset)`, and the permutation *pool* is unstated; nor is it said that the z-scoring is re-fit per draw. **Repair:** pin the unit as one `π` per `(dataset, τ, draw)` applied identically across the three seed cells; pin one shared `π` across the five structural columns; state that the reconstruction is per scoring fold and that standardisation is re-fit on the training folds within each draw.

**I-2 — the RNG substream policy is unpinned, and `default_rng(20260801)` is named for five different procedures.** Five independent instantiations of the same seed give five *identical* streams; one shared generator makes every draw order-of-execution dependent. **Repair:** pin one policy — e.g. `default_rng(20260801).spawn(k)` with a frozen named child index per procedure — and record the indices.

**I-3 — `round(1.5·|P_τ|)` and `round(2·|P_τ|)` are unpinned as to rounding rule, and the document's own worked example disagrees with Python/NumPy's default.** With `|P_0| = 55`, `1.5 × 55 = 82.5`; banker's rounding gives **82**, while §5.3's worked example uses **83**. `k` enters the caps, `K-NET`'s bar and `K-DEG`'s mapping. **Repair:** pin the rule explicitly (e.g. `int(np.floor(x + 0.5))`) and re-state the worked example under it.

**I-4 — §4.3's and §12's "item `i`'s label enters that fit nowhere" is false as written.** The `τ_hi^(g)` path R4 named *is* closed. But a fitting row `j ∈ fold g ≠ f` carries target `[j ∈ P_0] ∧ [c_j ≥ τ_hi^{(f)}]`, and both `[j ∈ P_0]` and `c_j` are computed from `j`'s deployed vote, whose bank is fold `g`'s fitting pool — which contains fold `f`, hence contains item `i`. So `i`'s gold label reaches `j`'s target through the same bank channel `GATE-BLIND` already books as `bank_label_reads`. The effect is diffuse (≈ 20/595 at weight ≤ 20/210) and, decisively, **identical for `BASE` and `FULL`**, so it cancels in the paired `ΔAUC`. `GATE-NESTED`'s assertion text is accurate; only the prose over-claims. **Repair:** replace the universal with a statement naming the bank channel and its cancellation.

**I-5 — provenance bundle.** (a) `HEADSPACE_TRANSFER_PREGATE.md:858-863` — the nine-arm enumeration runs `:859-864`; `:858` is blank and the quoted clause "three **invert sign**" sits on `:864`. (b) "**four** distinct operator families" is not re-derivable from the list quoted; it is at least six. (c) `GATE0_REOPEN_2026-07-31.md:1005-1006` — the quoted sentence begins "**A large**" on `:1004`. (d) §2 attributes *"every `$0` pregate run in the raw banked train-space arena"* to "CAL-2 … its own header"; it is the **clause document's** header at `PREGATE_CALIBRATION_CLAUSE.md:3`, which scopes CAL-0…CAL-5 alike — so it cannot out-scope CAL-2(2) while CAL-0/1/3/5 are treated as binding. Reason (ii) is independently sufficient; drop or restate reason (i). (e) §2 calls DET-1…DET-4 "binding"; DET-4 is headed *"recommended, not mandatory"*. (f) `GATE-NULL` cites a glob; the operative caches are the two named in `mechnov_pairverify.DATASETS`.

**I-6 — `banned_constraints[9]` is never named, though it is the campaign's most recent ban and the design contains its closest shapes.** `[9]` closes *"COMPOSITION RE-ORDERING of the deployed top-20 decision — aggregate-then-compare / **per-class prototype or centroid comparison** / … **at any rank or ridge** (F112)"*. `FULL`'s feature 5 is a per-class rank-1 similarity comparison. In my reading the ban is **not** engaged — its object is the *deployed decision operator* — but every other engaged-adjacent ban gets a named adjudication. **Repair:** add one sentence naming `banned_constraints[9]` and stating why it is not engaged.

**I-7 — §5.3's `LIVE` / `ARITHMETICALLY_DEAD` marking is derived from the net leg only, but `K-NET` has two legs.** The caps bound `mean_s net_s` and say nothing about `mean_s ΔmF1_s ≥ +0.050`. **Repair:** label the marking `LIVE_ON_NET` / `ARITHMETICALLY_DEAD_ON_NET` and say the mF1 leg carries no closed-form cap.

**I-8 — `THRESH-BEST` is selected by "the higher `net_s`", so the run necessarily computes a bare global-threshold operator's net on the HateMM train arena — the object F98 clause (d) bans re-deriving — and the design does not say what happens to that number.** **Repair:** state in §6.2 that the twins' own `net` and `Δacc` are computed solely to select and score the twin, are not reported as findings, and that F98(d)'s observation is cited from F113's banked head-space `+0.0041`, not re-derived.

**I-9 — §5.3's ZH recall figure is not re-derivable, and the HateMM one rounds the wrong way.** At `k = 1.5|P_0| ≈ 83` with `mean_s|wrong_s| = 62.33` and `bar = 29.0`, the requirement is `|S ∩ wrong| ≥ 56`, i.e. recall `≥ 56/62.33 = 0.8984` and precision `≥ 56/83 = 0.6747`; §5.3 states `0.88 / 0.68`. At `k = 114` on HateMM the requirement is `75.6/84.33 = 0.8965`, stated as `0.89`. **Repair:** re-derive both pairs in place, and state the `k` convention used.

**I-10 — §2's out-of-scoping of MHC-EN names only "an independent reason" and never states the primary one.** The decisive fact is that **no `headspace_arena_en_*_OUT.json` exists** — MHC-EN has no banked fold-head floor, so `GATE-FLOOR` has no anchor and the arena cannot be gated at all. And `headspace_mint.py` is not in a `frozen_artifact_policy` list; what pins it is `meta.mint_script_sha256 = cefdf8dc…` in the banked arena outputs. **Repair:** state the missing-floor reason first, and say precisely what pins the mint.

---

## Bottom line

The instrument, the legality spine and the executability are the strongest parts of this document, and I could not break any of them. Four sha256s match their banked anchors, all twelve floors and thirty fold arrays are exact, the live library quintet and node are byte-identical to the banked `meta.runtime` so `GATE-FLOOR` is achievable rather than aspirational, the 36-mint timing reproduces exactly, and every line of the ≈90-minute budget reproduces from its own factors with real slack. R1, R2 and R3 are fully discharged. Of R4's fourteen findings, twelve are discharged cleanly — including the Critical, whose `pred_agree = 1 − 2k(1−ov)/n` mapping I re-derived — and I confirmed that `K-DEG` is non-vacuous in both directions on every live cell.

The three Highs are narrow and all sit on the same seam: what the `τ_hi` co-primary and the new conditional null actually measure. The deliberate H-2 deviation is the **right** call, but the residualisation is linear while the document's own named dependence is a censored step function. The co-primary is separately squeezed from two sides that the design has each number for and has not multiplied together. None of the three changes the expected verdict — band B remains the pre-declared expectation and is a complete and valuable result — but each affects what a `τ_hi` reading would *mean*, and two of the three lean toward CONTINUE.

None of this needs a new measurement, a new instrument, or a GPU.


===== RAW REVIEW: refine-logs/C09_A0_V6_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v6 — Independent Design Review, Round 6

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V6_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 2 High / 10 Important`

---

## Prior-round audit

I re-checked every R1–R5 finding against **v6 itself**, not against §12, and re-opened every source.

**Round 1 (4C/8H/10I) — discharged, and it still holds.** C-1: `pred_purity` is against the *predicted* class (§4.4), label-blind for the scored item. C-2: the discriminator is conditional (`AUC_strat`) and incremental (`ΔAUC`), with `π* = (1+bar/k)/2` re-derived as the exact inversion of `net = k(2π−1)`. C-3: `headspace_mint.py:205` is literally `splits = list(skf.split(np.zeros((n,1)), lab))`, so the fold assignment is a function of the label vector alone and is identical across head seeds; the bootstrap and both permutations are now item-level. C-4: `net_s = 2|S∩wrong_s| − k ≡ n·Δacc_s` re-derived as an identity. H-3: `mechfix_ops.py:93-95` exactly as quoted. H-5/H-8/I-9 all land.

**Round 2 (1C/8H/10I) — closed.** Feature 11 stays deleted; F97's `+0.0269 / +0.0104` is registered as band B and as the design's *expectation*; F98(b) is adjudicated rather than narrowed; `K-FELDMAN` is computable; the gate/decision split holds; CAL-3 is discharged on `ERRPAT_HateMM §7`'s six cluster ceilings (all twelve numbers exact, `:410/:432/:444/:460/:477/:486`). I-6's mint layout is right on disk. Budget arithmetic re-verified line by line.

**Round 3 (1C/8H/11I) — all discharged.** §9's publication precondition, the two-scale twins, the permutation null, `STRATUM_OCCUPANCY`, the per-fold `τ_hi`, the withdrawn `NET` over-claim, `37.2/29.0`, the frozen sentinel `21`, `GATE-FIXK20`, the ZH raw-space relabelling, F113's three-clause caveat verbatim, the runtime quintet, deleted `q25/q75`, pinned `numpy.median`.

**Round 4 (1C/4H/9I) — all discharged.** C-1's `pred_agree = 1 − |S △ twin|/n = 1 − 2k(1−ov)/n` is exact; `aggnet_pregate.py:534` is literally `(c3 == coll["THRESH_best"]).mean()` and `:537` the same for `B_agree_fixk`. H-3's caps, H-4's HALT→data-outcome demotion, I-1's `9.3` points, I-2's "at least six", I-4, I-5, I-7, I-8(a)–(f), I-9's Hanley–McNeil all verified independently.

**Round 5 (0C/3H/10I) — 9 of 13 clean, 4 partial.**
- **H-1 — DISCHARGED as specified.** The OLS residualisation is withdrawn; `PERM-STRUCT-COND` now permutes within `ITEM-STRATUM`, which *is* an exact permutation null for `struct ⊥ target | ITEM-STRATUM`. R5's own primary repair, adopted, with the conditioning set named and `MARGINAL_ONLY_NOT_CONDITIONAL` scoped to it.
- **H-2 — DISCHARGED.** `p_w ≤ |P_τ|` computed in-run, cells marked, the joint ZH requirement written out, §9 adds the liveness conjunct.
- **H-3 — DISCHARGED in substance, but the repair created a new gap** (HIGH H-1 below).
- **I-1 — 2 of 3.** The permutation *pool* is still unstated (IMPORTANT I-3).
- **I-2 — PARTIAL.** The substream policy is not propagated to §6.3 (IMPORTANT I-2).
- **I-3 — PARTIAL.** §9 still writes `round(...)` (IMPORTANT I-4).
- **I-9 — attempted, one figure wrong** (IMPORTANT I-5).
- **I-4, I-5(a)–(f), I-6, I-7, I-8, I-10 — DISCHARGED**, with one line-number drift.

---

## Verified as sound (do not re-litigate)

**Executability is real; I re-opened every cited line.** `headspace_mint.py:106-116`, `:126` (`CLI` keys exactly `['hatemm','zh']`, enforced at `:177`), `:188-189`, `:192-194`, `:203-216`, `:274-281`, `:285-286`, `:322-324` (`lab_dev` in **every** mint). `mechnov_pairverify.py:56-57`, `:124`; `headspace_arena.py:7`; `headspace_fidelity.py:66` and its `--mintdir` at `:57`. The four sha256s match their banked anchors; the live `python 3.11.8 / numpy 1.26.4 / scipy 1.17.1 / sklearn 1.5.2 / torch 2.6.0+cu124` on `foscsmlprd01` is byte-identical to the banked `meta.runtime`. `squeue -u jehc223` is empty and C04's tranche has terminated (F117, job `13857`, COMPLETED), so the first sequencing precondition is satisfied in fact. **I found nothing that would turn this into an engineering HALT.**

**Every floor and count re-derived.** All twelve `GATE-FLOOR` values exact; 30 `fold_acc_deployed` cells present; fold sizes; `raw_deployed_acc 0.8441/0.8480`; `posrate_bank` → bands; `B_fid 0.0093/0.0086`; **`B_agree_fixk["20"] == agree_deployed` in 6/6 cells and `FIXK_20.d_acc = 0.0`, so `GATE-FIXK20` is corroborated rather than merely asserted**; per-seed error counts `83/85/85` and `62/64/61` integer-exact → `84.33 / 62.33`; caps; `DEG_KILL = 0.95`. **No `headspace_arena_en_*` exists anywhere in the repo** — §2's primary EN reason is true. Data defects re-measured independently.

**Decidability holds in both directions.** `K-REACH` at `τ_0` forces `2k/n ≥ 0.10`, so `K-DEG`'s vacuous regime is unreachable on the CONTINUE path; the `0.95` line maps to set-overlap `0.755–0.878` (HateMM) and `0.737–0.868` (ZH) across the live `k`. The four bands partition completed runs. Holm within dataset and family is correct; the across-dataset IUT correctly takes no correction; `p ∧ p_cond` is conservative. `ΔmF1_{O1}` monotonicity is right. `net_s ≡ n·Δacc_s`. `c_i` is label-blind for `i`.

**On the residual leak path, the answer is: none beyond the one §4.3 names.** Features are structurally blocked from `lab_query`, `is_inversion[seed]` and `is_stable_inversion`; standardisation is fit on training folds only; strata are label-blind; `τ_hi^{(f)}` excludes fold `f`. The only surviving channel is the arena's own bank channel, which §4.3 states exactly, prices, and correctly observes cancels in the paired `ΔAUC`.

**On `p_cond`'s exactness.** Within-`ITEM-STRATUM` permutation is exact for the stated null. It is not exact for `struct ⊥ target | BASE` in full, but v6 declares this and scopes it in §10 in the precise terms R5 asked for. Declared limitation, not a finding.

**Scope honesty and legality.** §10 claims nothing the design does not support. No test path can be opened; dev-label materialisation is bounded at 36; `H-L1`…`H-L4` are the right boundaries; §3's new `banned_constraints[9]` adjudication is correct. I found no ban read *narrower* than its own text.

---

## HIGH

**H-1 — §5.2's analysis-set definition and §4.3(b)'s fitting target use two different `τ_hi` conventions, so at the co-primary the design does not determine which rows are positives and which are negatives in the model that scores fold `f`.** §4.3 pins **(b) the FITTING TARGET** as *"for the model that scores fold `f`, every row in that fit … is labelled with that fit's own `τ_hi^(f)`"*. But §5.2 defines the analysis set with the **(a)** convention on both sides: *"Positives: `P_τ`"* — and `P_{τ_hi} ≡ { i ∈ P_0 : c_i ≥ τ_hi^{(fold(i))} }` — and *"Negatives: … `c_i ≥ τ_hi^{(fold(i))}`"*. The v5→v6 H-3 repair was written in the (a) convention while the v4→v5 H-1 repair had already moved the target to (b), and nothing reconciles them. Read literally, a stable inversion `j ∈ fold g` with `τ_hi^{(g)} ≤ c_j < τ_hi^{(f)}` is **in** the analysis set but carries `y_j^{(f)} = 0`, i.e. sits in the *negative* class of the fold-`f` fit, contradicting §5.2's own definition of that class and §6.1's statement of the contrast. Both readings are conservative and the magnitude is a handful of items, so this is not a wrong-verdict risk — but it is an under-determined frozen quantity on a **live** decision path, and a preregistration cannot leave the composition of its estimator's two classes to the implementer. **Repair:** state in §5.2 that the analysis set is **recomputed per scoring fold** under `τ_hi^{(f)}` for both classes, keep `P_{τ_hi}` only for `|P_τ|`, `K-REACH`, `O1` and `NET`'s `k`, extend `GATE-NESTED`'s per-item check to assert it, and note that the emitted negative counts are per-fold.

**H-2 — §11's named successor is the GAP-7 head/key-map class, and the campaign has a banked finding on exactly that class — F99 — carrying both a novelty closure and a closed-form arithmetic cap on the successor's principal channel. Neither appears among §11's headwinds, and F99 is absent from §11's own pre-registered Stage-1 adjudication list.** §11 names the successor as *"a global, symmetric, train-label-supervised reshaping of the head map `φ₀ → φ′`"*, encoder frozen — i.e. `REDTEAM_BAN_SCOPE_AUDIT.md` GAP-7's *"adapt the retrieval key-map / head recipe only, encoder frozen"*. §11 then makes the Stage-1 precondition binding, requiring *"(d) accompanied by a fresh ban-scope adjudication against F75, F66 and F98"* — and F99 is not in that list, nor named anywhere in v6. Two things F99 banks run directly against C09. **(1)** `dead[F99].ban_scope`, verbatim: *"D7 additionally kills the NOVELTY CLAIM regardless of the number (GAP-7 head/key-map class)."* **(2)** The load-bearing half — a closed-form ceiling on the successor's set-preserving channel, computed from the same marginals v6 already cites: from `w = [20…1]`, `Σw = 210` and the measured cone collapse, the best permutation flips a prediction iff top-20 purity `≥ 6/20` (hate) / `≥ 7/20` (non-hate); crossed with the measured error purity (`ERRPAT_HateMM:143-144`, `ERRPAT_MHC-ZH:222-223`), **at most 6 of 27 HateMM errors and 7 of 22 ZH errors are permutation-flippable, capping any set-preserving re-metrication at `≤ +0.0279` / `≤ +0.0470` under a zero-break assumption "the campaign has never met."** That is a statement about **C09's own target population**. §10's *"`O1` … is the only upper bound this A0 produces"* is true of *this A0* but leaves the reader without the far tighter banked bound on the object a CONTINUE would license. This design registers weaker headwinds at length; omitting F99 is the one asymmetry I could find in an otherwise scrupulously two-sided document, and it runs against C09. **Repair.** Add F99 to §11's headwind list and to precondition (d)'s adjudication set, quoting the GAP-7/D7 sentence and the `+0.0279 / +0.0470` permutation cap with its zero-break and test-split-purity provenance; and add one bullet to §10 recording that the set-preserving channel of the successor carries a banked closed-form ceiling far below `O1`.

---

## IMPORTANT

**I-1 — the frozen feature-set cardinalities are stale in three places.** §5.2's header reads *"Two frozen feature sets — 7 and 12"* while its own enumerations give `BASE` = **8** and `FULL` = **13**; §3.2(3) and §10 both say *"a 12-feature `lbfgs` logistic probe"*.

**I-2 — R5 I-2's RNG repair did not propagate to §6.3.** §6.3 still specifies a bare `default_rng(20260801)` for `SHUFFLE-POP` and `RANDOM-POP` — the exact construction R5 I-2 found unacceptable, in the section that defines a HALT gate.

**I-3 — the permutation *pool* is still unstated for both nulls, and R5 I-1(c) named it.** §5.2 never says whether `π` runs over all `n` items or over the `D-FELDMAN` analysis set. At `τ_hi`, where the analysis set is `c`-restricted, permuting over all `n` would import structural vectors from low-`c` items and change the null distribution.

**I-4 — §9 reverts to the word `round`, which §5.3 pinned `R(x)` precisely to avoid.** §9 is the rule that governs at verdict time; it must use `R(·)`.

**I-5 — the R5 I-9 re-derivation carries an arithmetic error, twice.** `56.0 / 62.33 = 0.89844` → **`0.898`**, not `0.899`, in §5.3 and in §12's I-9 row. (The HateMM companion and both precisions are right.)

**I-6 — `GATE-NULL`'s remove-null sensitivity has no line in the budget.** Clause 3 requires agreement on every K-rule outcome, so the sensitivity requires re-running both 1000-draw nulls plus `SHUFFLE-POP`, `RANDOM-POP`, the bootstrap and `K-DEG` on HateMM — ≈24 CPU-min, ~27 % of the declared total. No re-mint is implied and the job carries no `--time`, so this is completeness rather than feasibility.

**I-7 — provenance bundle (six items).** (a) §6.2 presents the `95.03% / 97.75% / 99.45%` sentence as "F96's number" with no citation; F96's `ban_scope` contains no such sentence — the nearest true text is `RESTRANS_PREGATE_RECORD.md:409`. (b) §1's F98 epitaph is exact only against `TARGET_STATE.json`'s rendering. (c) §6.3 calls F88's null (3) *self*-labelled; the quoted string is the reopen's compression at `GATE0_REOPEN:243-244`, and `ERRPAT_HateMM:395-396` phrases it differently. (d) Three sentences attributed to **F113** are not in `HEADSPACE_TRANSFER_PREGATE.md`; they trace to `directions_tried.json:525` and, for the KILL/PROMOTE clause, to `unified_pilot_gate.arena`, where every source writes lower-case "promote". F113's `dead[]` entry has no `ban_scope` and closes *"STANDING RULE PROPOSED (not yet ruled)"*. (e) The *"9 of 9 raw positives…"* clause **begins on `:863`**. (f) `C02_A0_OUT.json`'s real paths are `datasets.<ds>.gates.ARENA2.seed<n>.pooled_native_acc` and `datasets.hatemm.gates.VIEW_SUPPORT.degenerate_causes.EMPTY_TEXT`.

**I-8 — the header's authorisation line contradicts the record it cites, and drops one of that record's own preconditions.** The reopen's `what_this_reopen_does_not_do[2]` reads *"it authorizes no work…"* and `c09_next_step` ends *"NOTHING IS AUTHORIZED BY THE REOPEN"*. Separately, `c09_next_step` opens *"the draft must be repaired against the round-1 review and re-reviewed before anything is frozen"* — the precondition box drops this clause.

**I-9 — `banned_constraints[2]` / `hard_constraints[4]` (cross-seed ensembles) are never named, though the design's three most central objects are cross-seed aggregates.** The defence is clean and short — nothing is deployed, `H-L4` forecloses the selector, and `[4]`'s own qualifier is *"as a final performance method"* — but v6 gives named adjudications to `[9]`, `[10]`, F47, F96, F97 and F98 and is silent here.

**I-10 — three residual scoping precisions.** (a) CAL-2(2)'s sole surviving skip reason is over-broad: `ksweep_OUT.json`'s `curves[...]` has a `dev` per-`k` sub-key beside the `test` one, so *"the per-`k` payload is a TEST-SPLIT read"* is true of the sub-key CAL-2 leaves unnamed, not of the file. (b) §2's EN out-of-scoping does not mention `queued.headspace_arena_EN`'s *"~15 CPU-min"* price. (c) §5.3's caps are stated *looser* than the true values (`131.4667 / 95.6667` written `131.5 / 95.7`), and `28.95` vs `29.0` are used for the same ZH figure in different sections (neither can move an outcome, but they should be reconciled).

---

## Bottom line

This is the strongest document in the lineage and I could not break the parts that matter most. The instrument is real and executable end to end; the legality spine is sound in both directions; the only surviving route by which a scored item's gold label touches its own model is the arena's own bank channel, which §4.3 names, prices, and correctly shows cancels. R1–R4 are fully discharged; nine of R5's thirteen land cleanly and four partially. Band B remains the pre-declared expectation and a KILL scoped *"not identifiable at this power"* is a complete and valuable result.

Two things must change before freeze. The v6 repair of R5's H-3 was written in the `τ_hi` convention that the v5 repair of R4's H-1 had already replaced, so the composition of the estimator's two classes at the co-primary is under-determined. And §11, the clause that gives a CONTINUE its only forward meaning, omits F99 — the one banked finding written on exactly the object it names. The ten Importants are precision and propagation.


===== RAW REVIEW: refine-logs/C09_A0_V7_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v7 — Independent Design Review, Round 7

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V7_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 1 High / 4 Important`

---

## Prior-round audit

I re-checked every R1–R6 finding against **v7 itself**, and re-opened every source rather than trusting §12. I also diffed v6→v7 line by line to confirm that each repair landed where the ledger says and that nothing else moved.

**Rounds 1–5 — discharged, and they still hold in v7.** R1 C-1 (`pred_purity` against the *predicted* class); C-2 (conditional `AUC_strat` + incremental `ΔAUC`, `π*` re-derived); C-3 (`headspace_mint.py:205` is literally `splits = list(skf.split(np.zeros((n, 1)), lab))`, so the fold map is a function of the label vector alone and is seed-invariant; bootstrap and both permutations item-level); C-4 (`net_s ≡ n·Δacc_s`). R2 C-1, H-3, H-5, H-6, H-7, H-8. R3 C-1, H-1…H-8, I-1…I-11. R4 C-1 (`pred_agree = 1 − |S△twin|/n = 1 − 2k(1−ov)/n`; `aggnet_pregate.py:534/:537`), H-1…H-4, I-1…I-9. R5 H-1/H-2/H-3 and I-1…I-10.

**Round 6 (0C/2H/10I) — all twelve land.**
- **H-1 — DISCHARGED as specified, but the repair opened a new seam.** §5.2 now recomputes `A^{(f)}` per scoring fold for both classes at `τ_hi^{(f)}`; the old text is gone; `P_{τ_hi}` survives only for `|P_τ|`, `K-REACH`, `O1`, `NET`'s `k`; `GATE-NESTED`'s check is extended; per-`(dataset, seed, fold, τ)` counts are emitted. The interaction with I-3's pool pin is HIGH H-1 below.
- **H-2 — DISCHARGED, and both F99 legs are exact.** `dead[F99].ban_scope` reads verbatim as quoted, GAP-7 is at `REDTEAM_BAN_SCOPE_AUDIT.md:293` with its cell text at `:303-304` and the D7 sentence at `:305-308` — both citations correct. The arithmetic checks: `6/215 = 0.02791`, `7/149 = 0.04698`, and `ksweep_OUT.json` independently confirms `n_test = 149` for MHC-ZH.
- **I-1 … I-10 — all DISCHARGED.** 8/13 everywhere; children `2`/`3` in §6.3; `R(x)` in §9 with no `round(` anywhere; `56.0/62.33 = 0.898`; the `~24 min` `GATE-NULL` line and the ≈115 total; all six provenance items; the authorisation line and all three `c09_next_step` clauses; the `banned_constraints[2]` / `hard_constraints[4]` adjudication; and I-10's three precisions including the `[28.95, 29.0)` proof.

---

## Verified as sound (do not re-litigate)

**Every quotation I opened is exact at its cited location.** Verbatim-verified this round: `unified_pilot_gate.{arena, stage_0_reachability}`; `hard_constraints[4]`; `banned_constraints[2]/[9]/[10]`; `candidate_registry[8].{claim, dedup_boundary}`; the reopen's promoted/sequencing/`c09_next_step`/`what_this_reopen_does_not_do[2]`/`strategic_finding` fields; `ban_scope` for F47, F75, F96, F97, F98(b)+(d), F99; `dead[F113].status`'s two measurement sentences; `queued.headspace_arena_EN`; `progress.json:25`; `LITSWEEP3_DATA_CENTRIC.md:82/:91/:95`; `LITSWEEP5_COMPLETENESS.md:84/:125/:127/:128`; `HEADCOV_PREGATE_RECORD.md:305-310`; `NCA_FORENSIC_RECON.md:110/:112`; `RESTRANS_PREGATE_RECORD.md:409`; `GATE0_REOPEN_2026-07-31.md:243-244` and `:1004-1006`; `HEADSPACE_TRANSFER_PREGATE.md:859-864`, `:871-873`, `:917-919`, `:920-923`; `ERRPAT_HateMM:126/130/131/133/134/135/139-141/143-144/395-396` and §7's six cluster ceilings; `ERRPAT_MHC-ZH:220/222-223/233-235/237-240`; `PREGATE_CALIBRATION_CLAUSE` CAL-2(1)/(2)/CAL-3 and the `:3` header; `PREGATE_DETERMINISM_CLAUSE §1.3`, DET-3 Tier B, DET-4's *"recommended, not mandatory"*.

**Every number re-derived from source.** All twelve `GATE-FLOOR` floors; fold sizes; `raw_deployed_acc`; `posrate_bank` → bands; `B_fid`; per-seed errors `83/85/85` and `62/64/61` → `84.3333 / 62.3333`; caps `131.4667 / 95.6667`; `π*` ⇒ 9.3 points; `0.896 / 0.663 / 0.898 / 0.675`; Hanley–McNeil; the `0.95 ↔ ov 0.755` mapping; `0.1054–0.1142`; F99's `6/215` and `7/149`. Data defects and `GATE-NULL` re-measured independently.

**Executability is real, end to end.** All four sha256s match their banked anchors. `mechfix_ops.py:94-95` literal; `deployed_vote` returns only `(votes, preds, I, sim)`, so the extra `k=50` and per-class top-1 searches are genuinely needed and `_norm32`/`_flat_ip` exist to do them on the same engine. `headspace_mint.py`'s six cited line ranges are exact; `lab_dev` in every mint; `CLI` keys exactly `['hatemm','zh']`. All ten `vsw_ckpt` fold files, both `dev_seen_*.pt`, all six floor trainlogs and `headspace_report.py` are present. **No `headspace_arena_en_*` exists anywhere in the repo.** The live environment is byte-identical to the banked `meta.runtime`. `GATE-FIXK20` is corroborated, not merely asserted: banked `FIXK_20` has `changed = 0`, `dacc = 0.0` and all-zero `folddeltas`. `sacct` confirms job `13847` = 8 CPU / 32 G / no GPU / `00:29:49`; `squeue -u jehc223` is empty and C04's tranche has terminated (job `13857` COMPLETED). **I found nothing that would turn this into an engineering HALT.**

**The budget is sound.** Every table line reproduces from its own factors and the lines sum to 114–116 against the declared ≈115, generous by an order of magnitude on the permutation lines.

**Legality holds in both directions.** No test path can be opened; dev labels are materialised exactly 36× and kept outside every decision quantity; the F47, F98(b)+(d), F96, `[9]` and the new `[2]`/`[4]` adjudications all **concede** the ban rather than narrowing it. F99's ban is formally about distilling the F95 verifier into the key space — C09's successor is not that — yet v7 imports F99's cap and novelty leg anyway; that is reading a ban wider than its letter *against* C09, the conservative direction, and is what R6 asked for.

**Decidability.** `K-REACH` forces `2k/n ≥ 0.10`, so `K-DEG`'s vacuous regime is unreachable on the CONTINUE path. `LIVE_ON_NET` leaves two of three `k` live on both datasets at `τ_0`. `p_w ≥ 30` is reachable and can also fail, so it is neither inert nor fatal by construction. The bands partition completed runs; no single rule can produce a CONTINUE. Holm within dataset and family is correct; the IUT takes no correction; `p ∧ p_cond` is conservative. `R(x)` sends `82.5 → 83`.

**On the residual leak path: none beyond the one §4.3 names.** `i`'s own bank never contains `i`; `τ_hi^{(f)}` excludes fold `f`; both stratifications are label-blind; standardisation is fit on training folds only; the feature builder admits neither `lab_query` nor either target-derived array; the `K-DEG` twins feed no feature or target.

**Scope honesty.** §10 claims nothing the design does not support and now carries F99, the headwind that runs hardest against C09. Both "re-measured this session" claims are real `$0` train-side reads and I reproduced both.

---

## HIGH

**H-1 — the R6 I-3 pool pin and the R6 H-1 per-fold analysis set contradict each other at the co-primary, so the null distribution of the primary test is under-determined, and one of the two readings is anti-conservative.** §5.2 pins the permutation pool as *"the `D-FELDMAN` analysis set only … Since the analysis set is now per scoring fold, the pool is `A^{(f)}` for the fold-`f` refit"* and then, in the next sentence, asserts *"**The drawing unit is unambiguous:** per `(dataset, τ, draw)` **one** permutation `π` of that pool's **items** is drawn."* At `τ_0` these agree, because `A^{(f)} = P_0 ∪ {all three-seed-correct}` for every `f`. At `τ_hi` they cannot both hold: `A^{(f)}` is a different set for each of the five folds (§5.2 says so in terms), and a single permutation of items cannot be a permutation of five different pools. An implementer must therefore choose:

- **(a) five `π` per draw, one per scoring fold** — the literal reading. Under (a) each null draw's pooled OOF score vector is built from **five independent** structural re-assignments, so the between-draw variance of `ΔAUC_d` shrinks, the null narrows, and `p` is **too small** — anti-conservative, in the CONTINUE direction. This is the same averaging mechanism R1 C-3 rated Critical for the bootstrap and R4 H-2(i) rated High for the row-level permutation.
- **(b) one `π` per draw over `∪_f A^{(f)}`, restricted to `A^{(f)}` inside each refit** — coherent and conservative, and what the "one `π`" sentence intends — but the document never names the union pool.

The same hole propagates to `PERM-STRUCT-COND` and to `SHUFFLE-POP`. Exposure is confined to `τ_hi` — but `τ_hi` is a registered co-primary that §9's `∃τ` makes CONTINUE-capable, and the design *measures* rather than assumes that `τ_hi` is power-dead. This is exactly the class of defect v7's own §0 rationale for the H-1 repair forbids, applied to the resampling scheme of the only rule that tests. **Repair (one sentence, no re-measurement):** pin the pool as `P^{(τ)} ≡ ∪_f A^{(f)}` (equivalently `{j : c_j ≥ min_f τ_hi^{(f)}}` at `τ_hi`, `P_0 ∪ {3-seed-correct}` at `τ_0`), draw **one** `π` of `P^{(τ)}`'s items per `(dataset, τ, draw)`, and state that inside the fold-`f` refit only the rows of `A^{(f)}` are used, with donors drawn from `P^{(τ)}`; delete "the pool is `A^{(f)}` for the fold-`f` refit"; and apply the identical wording to `PERM-STRUCT-COND` and `SHUFFLE-POP`.

---

## IMPORTANT

**I-1 — `PERM-STRUCT-COND`'s exactness claim omits `BASE` from the conditioning set, and the test statistic reads `BASE`.** §5.2 asserts *"This is an exact permutation null for `struct ⊥ target | ITEM-STRATUM`."* A within-stratum permutation resamples `struct` from its exchangeable conditional law given `ITEM-STRATUM` only; validity for a statistic `T(struct, target, BASE, stratum)` requires `struct ⊥ (target, BASE) | ITEM-STRATUM`. The difference is not idle: the dependence §5.2 itself nominates — structural feature 1's sentinel fires exactly when `pred_purity = 1.0` — is **not** absorbed by the `[0.95, 1.0]` bucket, so the observed `FULL` retains a within-stratum `struct↔BASE` coupling every permuted draw loses. `p_cond` can then reject under the null it was added to protect. §10's scoping is the right honest statement and largely contains the damage; the word "exact" attached to the wrong null is not. **Repair:** state the joint form, and that under the weaker independence a rejection may reflect within-stratum non-linear re-encoding of `BASE`.

**I-2 — §5.3's out-of-support declaration is stale against the R6 H-1 repair and, at `τ_hi`, materially incomplete.** (a) The frozen fit set is now `A^{(f)}`, not `P_τ ∪ CONFIG-MATCHED-CORRECT`. (b) At `τ_hi` **both** classes are `c`-restricted, so the out-of-support population is not just the ~7–9 unstable errors: it is every item with `c_j < τ_hi^{(f)}`, i.e. roughly half of `P_0` plus the whole low-`c` tail of the correct class — while `NET` still ranks all `n`. That is a much larger extrapolation than the one declared, on a rule that can produce a CONTINUE. **Repair:** restate the fit set and add the `c`-restricted items to the named out-of-support population at `τ_hi`.

**I-3 — CAL-2(2)'s secondary skip reason quotes a HateMM-only `n_test` for all three named arms.** In the file, `n_test` is `215` for HateMM, **`149`** for `MHC_zh` and **`161`** for `MHC_EN_ARM-V`. Nothing gating turns on it, but the ZH figure is load-bearing elsewhere in this very document (F99's `+0.0470 = 7/149`).

**I-4 — `THRESH-SYM`'s per-seed scale is grammatically ambiguous in a KILL-capable gate.** Read as "average the scores", the per-seed twin becomes identical to the per-item twin, collapsing the two-scale maximum §6.2 explicitly adopts as *"the conservative direction for a gate whose job is to fire."* **Repair:** "…the `k` items with the smallest `|score_{i,s}|` **within each seed**; `pred_agree` is computed per seed and averaged over the three seeds."

---

## Bottom line

This remains the strongest document in the lineage and I could not break the parts that matter most. The instrument is real, on disk, sha-pinned and demonstrably reproducible in the live environment; the legality spine holds in both directions; the only route by which a scored item's gold label touches its own model is the arena's own bank channel, which §4.3 names and prices; §10 claims nothing the design does not support and now carries F99. R1–R5 are fully discharged and all twelve of R6's findings land.

One thing must change before freeze, and it is a one-sentence fix. R6's H-1 repair made the analysis set fold-dependent; R6's I-3 repair pinned the permutation pool to that same fold-dependent set — and then asserted a single permutation per draw. At `τ_0` the two agree. At `τ_hi` they cannot, and the reading the letter forces narrows the null in the CONTINUE direction. Name the union pool and the contradiction disappears. The four Importants are precision, provenance and completeness.


===== RAW REVIEW: refine-logs/C09_A0_V8_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v8 — Independent Design Review, Round 8

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V8_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 1 High / 3 Important` *(all four in the front matter; the science passed)*

---

## Prior-round audit

I re-checked every R1–R7 finding against **v8 itself**, not against §12, and re-opened every source. I also diffed v6 → v7 → v8 to see what actually moved.

**Rounds 1–5 — discharged, and they hold in v8.** R1 C-1; C-2 (`π*` re-derived as the exact inversion of `net = k(2π−1)`); C-3 (`headspace_mint.py:204-205` literal, fold map seed-invariant; bootstrap and both permutations item-level); C-4 (`net_s ≡ n·Δacc_s`). R2 C-1, H-3, H-5, H-6, H-7, H-8. R3 C-1, H-1…H-8, I-1…I-11. R4 C-1 (`pred_agree = 1 − |S△twin|/n = 1 − 2k(1−ov)/n`; `aggnet_pregate.py:534/:537`), H-1…H-4, I-1…I-9. R5 H-1/H-2/H-3 and I-1…I-10.

**Round 6 (0C/2H/10I) — all twelve land.** `A^{(f)}` per scoring fold; F99 in §10 and precondition (d) with `6/215 = 0.02791`, `7/149 = 0.04698` and `n_test = 149` independently confirmed; 8/13 everywhere; children `2`/`3` in §6.3; no `round(` anywhere; `56.0/62.33 = 0.898`; the `~24 min` `GATE-NULL` line; all six I-7 provenance items; the authorisation line and all three `c09_next_step` clauses; `[2]`/`[4]`; caps `131.4667 / 95.6667`; the `[28.95, 29.0)` proof.

**Round 7 (0C/1H/4I) — all five land in the body.**
- **H-1 — DISCHARGED and correctly.** §5.2 names `P^{(τ)} ≡ ∪_f A^{(f)}`, one `π` per `(dataset, τ, draw)`, `A^{(f)}`'s rows only inside the fold-`f` refit, donors from `P^{(τ)}`. I verified the set identity at both `τ`. The scheme is exact for the marginal null over the pool and the union pool cannot narrow it.
- **I-1 … I-4 — DISCHARGED.** The joint exactness form; the `A^{(f)}` fit set and the `c`-restricted out-of-support population; `n_test` `215 / 149 / 161` (and `curves` really is keyed by the flat string with `test` and `dev` sub-keys, so §2's notation is literal); `THRESH-SYM` per seed with `pred_agree` averaged.

**What no prior round audited:** the STATUS block and §0's "retained repairs" list. Both are **verbatim v6 text**, carried unchanged through v7 into v8. That is where the one High and the first Important sit.

---

## Verified as sound (do not re-litigate)

**Every quotation I opened is exact at its cited location.** Verbatim-verified this round against source: `unified_pilot_gate.{arena, stage_0_reachability}`; `hard_constraints[4]`; `banned_constraints[2]/[9]/[10]`; `candidate_registry[8].{claim, dedup_boundary}`; the reopen's promoted fields, `what_this_reopen_does_not_do[2]`, `c09_next_step`, `strategic_finding`, kill-risk (i) with all four F114 qualifiers, and F88's CPU-floor caveat; `ban_scope` for F47, F75, F96, F97, F98(b)+(d), F99; `RESTRANS_PREGATE_RECORD.md:409`; F98's DEG-A and `DEG_KILL = 0.95`; `dead[F113].status` and its absent `ban_scope`; `queued.headspace_arena_EN`; `progress.json:25`; `LITSWEEP3_DATA_CENTRIC.md:82/:91/:95`; `LITSWEEP5_COMPLETENESS.md:84/:125/:127/:128`; `HEADCOV_PREGATE_RECORD.md:305-310`; `NCA_FORENSIC_RECON.md:110/:112`; `REDTEAM_BAN_SCOPE_AUDIT.md:303-304` and `:305-308`; `GATE0_REOPEN_2026-07-31.md:243-244` and `:1004-1006`; `HEADSPACE_TRANSFER_PREGATE.md:859-864`, `:871-873`, `:917-919`, `:920-923`; `ERRPAT_HateMM` (nine cited locations plus §0.1 and §7's six ceilings, summing to n=27, with `1/215 = 0.004651`); `ERRPAT_MHC-ZH:220/222-223/233-235/237-240`; CAL-0…CAL-5 and the `:3` header; DET-1…DET-4 including DET-4's *"recommended, not mandatory"*.

**Every number re-derived from source.** All twelve floors; fold sizes; 30 `fold_acc_deployed` cells; bands; `raw_deployed_acc`; `B_fid`; per-seed errors → `84.3333 / 62.3333`; caps ⇒ `k ≥ 132 / 96` dead; `0.8965 / 0.663 / 0.898 / 0.675`; `π*` ⇒ 9.3 points; Hanley–McNeil; the `pred_agree ↔ ov` mapping; `0.1054–0.1142`; F99's `6/215` and `7/149`; `p_w ≥ 30` out of ≈38 = 79 %. Data defects and `GATE-NULL` re-measured independently.

**Executability is real, end to end, and I found nothing that would turn this into an engineering HALT.** All four sha256 match their banked anchors. `mechfix_ops.py:91/:94/:95` literal; `deployed_vote` returns only `(votes, preds, I, sim)`. `headspace_mint.py`'s cited ranges exact; `lab_dev` on the common `np.savez` path, hence in all 36 mints; `det1_assert` at `:75` fires at `:187`; `CLI` exactly `['hatemm','zh']`. `headspace_fidelity.py:57/:66` literal. All ten `vsw_ckpt` files, both `dev_seen` caches and six floor trainlogs present. **No `headspace_arena_en_*` exists anywhere in the repo.** `load_split` is called only with `"train"` and `"dev_seen"`, and `run_rac.load_feats_from_CLIP` is monkeypatched to a closure returning train/dev/dummy-from-fitting-pool — no test path can be reached. The live environment is byte-identical to the banked `meta.runtime`. **`np.random.default_rng(20260801).spawn(6)` works on this numpy — I ran it.** `int(np.floor(82.5+0.5)) = 83` against `round(82.5) = 82`, so the `R(x)` pin is load-bearing and correct. `GATE-FIXK20` is corroborated: banked `FIXK_20` has `changed = 0`, `dacc = 0.0`, `B_agree_fixk["20"] == agree_deployed`. `sacct` confirms `13847` = 8 CPU / 32 G / no GPU / `00:29:49`; `squeue` empty; C04's tranche terminated (`13857`, `13862` COMPLETED).

**The budget is sound and very conservative.** Lines sum to **116** against ≈115. I timed one `lbfgs` fit-and-score at the design's scale under DET-1 threads: **3.6 ms**, against the 35 ms assumed.

**Statistical soundness.** `AUC_strat`'s weighting, zero-weight rule, frozen strata and row-pooling are pinned; `ΔAUC` is genuinely paired. `PERM-STRUCT` is exact for the marginal null over `P^{(τ)}`. `PERM-STRUCT-COND` is exact for the joint form now claimed, and degenerate small strata can only push `p_cond` toward 1 — conservative. Holm within dataset and family, both families required, is conservative; the IUT takes no correction. `SHUFFLE-POP`'s band cannot be silently degenerate. `p_w ≥ 30` is neither inert nor fatal by construction. `K-DEG`'s vacuous regime is unreachable on the CONTINUE path. `ΔmF1_{O1}` monotonicity is genuine arithmetic. The four bands partition completed runs. No single rule can carry a CONTINUE.

**On the residual leak path: none beyond the one §4.3 names.** `i` is never in its own bank; every feature reads only `i`'s key, bank keys and bank labels; `τ_hi^{(f)}` excludes fold `f`; every row in the fold-`f` fit carries `τ_hi^{(f)}`; both stratifications are label-blind; standardisation is fit on training folds only; the `K-DEG` twins feed no feature and no target and can only make the gate fire.

**Legality holds in both directions.** No test path; dev labels materialised exactly 36× and kept outside every decision quantity; the F47, F96, F97, F98(b)+(d), `[9]`, `[2]`/`[4]` and F99 adjudications all concede their bans. `headspace_mint.py` is genuinely not in `frozen_artifact_policy` — §2's statement about what pins the mint is exactly right.

---

## HIGH

**H-1 — §0 asserts, under the heading "retained unchanged in v8", a conditional null that §5.2 explicitly withdraws, and a permutation unit that §5.2 explicitly replaces.** `C09_A0_V8_RECORD.md:69` heads the list *"Rounds 3–6's repairs, retained unchanged in v8:"*. Item 3 (`:84-89`) closes: *"v5 permutes at the **item** level **within each `(dataset, seed)` cell**, states the null honestly as **marginal**, and adds a **residualised conditional null** that a CONTINUE must also clear."* Both italicised clauses are contradicted by the governing body: `:844` reads *"`PERM-STRUCT-COND` … (R5 H-1 — **v5's OLS residualisation is withdrawn**)"* with the reason at `:845-853`; and `:809-823` pins *"**One** permutation `π` of `P^{(τ)}`'s items … per `(dataset, τ, draw)`"*, *"applied identically in all three seed cells"*, so §0's *"within each `(dataset, seed)` cell"* is the v5 phrasing R5 I-1(a) killed and is incompatible with the union pool that is v8's entire reason for existing. The whole four-item list is verbatim v6 text; v7 and v8 re-labelled it without editing its contents, and neither R6 nor R7 audited it. §5.2 governs and is internally consistent, so this is not a wrong-verdict certainty — but a hash-frozen preregistration whose purpose is to remove implementer discretion may not contain, in its own change summary, an affirmative statement that it retains a resampling scheme its body rejects. **Repair:** rewrite item 3 to the v8 object, or relabel the list as history and state that §5.2 governs.

---

## IMPORTANT

**I-1 — the STATUS block is two revisions stale and misdescribes the document's own contents.** `:19-29` is verbatim v6 text. The reading order runs *"… → R5 (0C/3H/10I) → this file"*, omitting v6/R6 and v7/R7; *"v1–v5 are superseded in full"* leaves **v6 and v7 un-superseded** in the only place the document addresses supersession — and `C09_A0_V7_RECORD.md` is on disk carrying the pool specification R7 rated High; *"The v6 repair ledger is §12"* is false in v8, whose §12 heading reads *"the 5 round-7 findings"*. Relatedly §0's tail still ends *"and nine provenance and specification items are corrected"* — R4's Important count.

**I-2 — §1's summary sentence states the strongest possible negative-existence claim, which is the opposite of what A0 does.** `:132-133`: *"A0 trains no encoder, touches no test split, and establishes that no operator exists."* Read as written that asserts a proof of nonexistence — precisely the over-claim §10's first bullet forbids. The intended meaning is *"establishes no operator's existence"*. Present since v2 and never flagged.

**I-3 — §11 names an *encoder-frozen* successor while C09's registry claim and dedup boundary are *encoder-level*; the substitution is unremarked, and F51 appears nowhere.** §11 identifies the successor as GAP-7's cell, *"adapt the retrieval key-map / head recipe only, encoder frozen"*. Those are different objects, and the repository says so at the lines §11 quotes from: GAP-7 is headed *"F51 two-object closure"*, and `REDTEAM_BAN_SCOPE_AUDIT.md:302-304` reads *"F51's 'no third object' is airtight for adapting the MLLM … But the tasking's candidate … is a **different** object F51 does not address."* So F51 does not bind §11's successor — but it does bind the object C09's own claim registers, since an encoder-level region-targeted retrieval term is `dead[F51]`'s *"= P9b's adapted object … do not re-propose."* Nothing here makes the A0 illegal — it trains no encoder — but a CONTINUE would license an operator narrower than C09's registered claim, and the design does not say so.

---

## Bottom line

The science is finished. I could not break the instrument, the legality spine, the executability, the budget, or the inference. Four sha256s match their banked anchors, all twelve floors and thirty fold arrays are exact, the live environment is byte-identical to the banked runtime so `GATE-FLOOR` is achievable rather than aspirational, `spawn(6)` and `R(82.5) = 83` both behave as pinned, the run costs a tenth of what it budgets, and no test path can be opened by any code path I could find. R7's High is genuinely repaired: the union pool `P^{(τ)}` is the right object and one `π` over it is exact for the marginal null. The only route by which a scored item's gold label reaches its own model remains the arena's own bank channel, which §4.3 names, prices and shows cancels.

What blocks the freeze is not science but the document's front matter, which is two revisions behind its body and was never audited by R6 or R7. §0 states in the affirmative that v8 retains the OLS-residualised conditional null — the construction §5.2 withdraws as anti-conservative — and re-imports the per-`(dataset, seed)`-cell permutation unit the union pool replaced; and the STATUS block still reads as v6's. These are edits to four paragraphs, need no re-measurement and no GPU. Fix them and this is ready to freeze and submit as a single CPU-only SLURM job.


===== RAW REVIEW: refine-logs/C09_A0_V9_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v9 — Independent Design Review, Round 9

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V9_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 3 Important` *(the first round with neither a Critical nor a High)*

---

## Prior-round audit

I diffed v8 → v9 line by line and re-checked every R1–R8 finding against **v9 itself**, re-opening every source rather than trusting §0 or §12.

**The diff is exactly what the ledger claims it is, and nothing else moved.** v9 touches only: the title, the STATUS block, §0, one sentence of §1, one bullet added to §10, one paragraph added to §11, and §12. `§§2–9` and `§11`'s pre-existing text are byte-identical to v8. No rule, threshold, feature set, null, gate or arithmetic changed.

**Round 8 (0C/1H/3I) — all four repairs land, and none of them disturbed the body.**
- **H-1 (§0's "retained unchanged" list) — DISCHARGED.** Rewritten as five history bullets under a heading that says *"§§1–11 govern. Nothing in this subsection is a rule."* I checked each bullet against its governing section: `K-DEG`→§6.2 ✓; `τ_hi`→§4.3(b)+§5.2 ✓; the conditional-null thread now ends *"the OLS variant is withdrawn and the per-`(dataset, seed)`-cell unit is superseded"*, matching §5.2 exactly ✓; caps/stratum-degeneracy→§5.3+§5.2 ✓; `STRATUM_OCCUPANCY`→§5.2 ✓. The affirmative re-assertion R8 rated High is gone.
- **I-1 (stale STATUS block) — DISCHARGED.** Reading order runs through R8; `v1–v8` declared superseded; §12 named the v9 ledger; §0's tail no longer carries R4's Important count.
- **I-2 (§1's negative-existence claim) — DISCHARGED.** *"establishes the existence of no operator"* takes the negative-raising reading, and the inline parenthetical removes residual ambiguity.
- **I-3 (encoder-frozen vs encoder-level, F51) — DISCHARGED, and the new text verifies at source.** `candidate_registry[8].claim` reads *"can be corrected at encoder level"*; `REDTEAM_BAN_SCOPE_AUDIT.md:302-304` and `:303-304` are verbatim; GAP-7's heading at `:293` begins *"F51 two-object closure"*; `dead[F51].ban_scope` is quoted exactly. The declared consequence — a CONTINUE licenses an operator narrower than C09's registered claim — is correct and runs against the candidate.

**Round 7 (1H/4I)** — re-verified independently. The union-pool identity holds at both `τ`; because only `struct` is permuted and the evaluation set is fixed by `target` and `c`, the scheme is exact for the marginal null over the pool. The joint exactness form, the `A^{(f)}` fit set with its `c`-restricted out-of-support population, `n_test` `215 / 149 / 161`, and the `THRESH-SYM` disambiguation all land.

**Rounds 1–6** — spot-verified from source. `aggnet_pregate.py:534/:537` literal; `pred_agree = 1 − 2k(1−ov)/n` re-derives; `6/215 = 0.027907`, `7/149 = 0.046980`; 8/13 everywhere; RNG children `2`/`3`/`4`; no bare `round(` anywhere; caps `131.4667 / 95.6667`; no multiple of `1/3` in `[28.95, 29.0)`.

**One R5 finding did not survive re-checking, and it is the source of I-2 below.** R5's I-5(b) asserted the source's "four" was not re-derivable and "it is at least six." That assertion is wrong at source, and v6–v9 have carried it forward.

---

## Verified as sound (do not re-litigate)

**Registry and ban quotations, re-opened at source.** `unified_pilot_gate.{arena, stage_0_reachability}`; `hard_constraints[4]`; `banned_constraints[2]/[9]/[10]`; `candidate_registry[8].{claim, dedup_boundary}`; the reopen's promoted fields, `what_this_reopen_does_not_do[2]`, all three `c09_next_step` clauses, `strategic_finding`, and kill-risk (i) with all four F114 qualifiers — all verbatim. `ban_scope` for F47, F51, F66, F75, F96, F97, F98(b)+(d), F99, F112 verbatim. **`dead[F113]` genuinely carries only `name` and `status` — no `ban_scope`** — so §2's statement that the binding surface is the registry clause is exactly right. All the `.md` citations verify, including `AGGNET_PREGATE_RECORD.md:690`, which does phrase the epitaph differently as §1 says.

**Every number re-derived.** All twelve floors; 30 `fold_acc_deployed` cells; fold held-out counts recomputed from `StratifiedKFold(5, shuffle=True, random_state=0)` on the real label vectors; `raw_deployed_acc`; `B_fid`; bands; per-seed errors → `84.3333/62.3333`; caps ⇒ `k ≥ 132/96` dead; `k = R(2|P_0|) = 152/110` dead, `R(1.5|P_0|) = 114/83` live; `0.896/0.663` and `0.898/0.675`; `π*` gap `0.093125` at `k=80` and 7.5 at `k=100`; Hanley–McNeil; `0.1054–0.1142`; `0.050 × 743 = 37.15`. Both "re-measured this session" claims reproduce exactly on my own reads, as does `GATE-NULL`.

**Executability is real and I found nothing that would turn this into an engineering HALT.** `sha256(headspace_mint.py) = cefdf8dc…` matches the banked `meta.mint_script_sha256` in all six arena outputs and `C02_A0_OUT.json::frozen_modules`; `headspace_mint.py` is genuinely absent from `frozen_artifact_policy`. Every cited line range is exact. `lab_dev` is written on the unconditional `np.savez` path, hence in all 36 mints. `load_split` is called only with `"train"` and `"dev_seen"`; for `fold ≥ 0` the dev/test splits are a stratified slice of the *fitting pool* (`:221-226`), so **no test path is reachable by any code path**. The live environment is byte-identical to the banked `meta.runtime`. `default_rng(20260801).spawn(6)` works — I ran it. `int(np.floor(82.5+0.5)) = 83` against `round(82.5) = 82`. `GATE-FIXK20` is corroborated: banked `FIXK_20` has `changed = 0`, `dacc = 0.0`, `B_agree_fixk["20"] = agree_deployed = 1.0`; and `B_agree_fixk["15"] = 1.0` with `changed = 0` on ZH seed 0 confirms §6.2's `DEGENERATE_ALL_EMPTY` anchor. `sacct`/`squeue` confirm the job precedent and the empty queue; C04's tranche has terminated.

**The budget is sound and very conservative.** The 36 banked C02 mint wall times: min `33.2 s`, max `60.0 s`, median `41.85`, total `24.51 min` — §2's figures are exact. Every table line reproduces and the lines sum to **116** against ≈115. I timed one `lbfgs` fit-and-score at the design's scale under DET-1 threads: **4.6 ms** against the 35 ms assumed.

**Statistical soundness.** `AUC_strat` fully pinned; `ΔAUC` genuinely paired. `PERM-STRUCT` exact for the marginal null over `P^{(τ)}`; only `struct` is permuted, so the evaluation set is invariant across draws — the property exactness needs. `PERM-STRUCT-COND` exact for the joint form claimed; small item-strata can only push `p_cond` toward 1. Holm within dataset and family with both families required is conservative; the IUT takes no correction; the `1/1001` floor leaves `α/2` reachable. `p_w ≥ 30` neither inert nor fatal. `K-DEG`'s vacuous regime unreachable on the CONTINUE path; the gate can both fire and clear. `net_s ≡ n·Δacc_s`; `Δacc_{O1} = |P_τ|/n` identically per seed; `ΔmF1_{O1}` monotonicity genuine. The four bands partition completed runs, and **no single rule can carry a CONTINUE**.

**On the residual leak path: none beyond the one §4.3 names.** Verified feature by feature, threshold by threshold, stratum by stratum.

**Legality holds in both directions.** F47, F96, F97, F98(b)+(d), `[9]`, `[2]`/`[4]`, F66, F99 and now F51 all conceded rather than narrowed — several read wider than their own text *against* C09.

**Scope honesty.** §10 claims nothing the design does not support, and a KILL is genuinely available.

---

## IMPORTANT

**I-1 — v9's own change summary understates v9's own changes, in both places that state them, and §12 contradicts its own table.** §0 reads *"v9 changes nothing in §§1–11's substance"* and is contradicted two lines later by its own items 3 and 4, which change §1's summary sentence, add a bullet to §10 and add a paragraph to §11. §12 reads *"All four findings are in §0, the STATUS block, one sentence of §1 and one paragraph of §11. **Nothing in §§2–10's substance changed in v9.**"* — but **§10 gained a five-line scope bullet**, and §12's own I-3 row names its `where` as *"§10, §11"*, and §0's item 4 says *"a bullet to §10"*. This is the identical failure class R8 rated High, differing only in direction, which is why it is Important and not High. **Repair:** state v9's true surface (§0, STATUS, §1, §10, §11) in both places, with the invariant spelled out — no rule, threshold, feature set, null, gate or arithmetic changed; §§2–9 byte-identical to v8.

**I-2 — §6.1's "at least six distinct operator families" is contradicted by the cited source's own heading, and the accompanying claim that "four" is not re-derivable is false.** The enumeration is verbatim ✓ — but `HEADSPACE_TRANSFER_PREGATE.md:623` heads the ladder *"### 4.10 The transfer ladder (§2.10) — **four operator families**, one arena swap"*, and that is the only family count anywhere in the file. Its rung column places the nine positives in four groupings: rung 1 = VSW (`VSW_pow/exp/lin`, plus `ORACLE_lambda_pow` *"(hindsight ceiling)"* and `CTRL_cos_pow` *"(DEG-D, no verifier)"* — VSW's own oracle and control arms), rung 2 = F95, rung 3 = FIXK, and un-runged `THRESH_best` *"(global recalibration)"*. Reaching six requires promoting VSW's hindsight ceiling and its degeneracy control to independent families, a regrouping the source declines. R5's I-5(b) introduced the error and v6–v9 have carried it. Nothing gating turns on it and its direction is conservative, but it is a stated count the cited source contradicts, in the section whose only job is to register the prior faithfully. **Repair:** restore the source's taxonomy, and delete the "not re-derivable" clause.

**I-3 — §6.1's decimal-place characterisation of `HEADSPACE_TRANSFER_PREGATE.md:871-873` is wrong.** The two attributions are right — the sentence does live in `dead[F113].status` and `:871-873` does omit the `30/30` fold count (it appears at `:847`/`:901`) — but `:872-873` reads `(d_AUC −0.064 / −0.129), inverting the raw arena's +0.157 / +0.230`, which is **3 decimal places**. **Repair:** "at 3 dp".

---

## Bottom line

The science is finished and I could not move it. R8's four repairs all land and none disturbed the body: the diff confirms §§2–9 are byte-identical to v8, the new §11 paragraph and §10 bullet verify word-for-word against source, and the §1 correction reads as intended. The instrument is on disk, sha-pinned to its banked anchor, and reproducible in a live environment byte-identical to the banked runtime; the budget is ~7× conservative against a measured 4.6 ms fit; the legality spine concedes every adjacent ban including one (F51) that now runs against C09's own registered claim; the nulls are exact for the nulls they name and conservative where they are not; the four decision rules are jointly decidable, individually non-vacuous, and none can carry a CONTINUE alone; and the only route by which a scored item's gold label reaches its own model remains the arena's own bank channel, which §4.3 names, prices and shows cancels.

What I am not able to pass is three statements about the document's and the record's own contents. Two are provenance defects inherited from R5 and R6 that four rounds of review have re-attested without re-opening the source line that contradicts them. The third is the same defect class R8 rated High, one revision later and pointing the other way. These are edits to three sentences. They need no re-measurement, no GPU and no re-derivation of anything in §§2–11. Fix them and freeze.


===== RAW REVIEW: refine-logs/C09_A0_V10_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v10 — Independent Design Review, Round 10

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V10_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 2 Important`

---

## Prior-round audit

I diffed v9 → v10 line by line and re-verified R1–R9 against **v10 itself**, re-opening every source rather than trusting §0 or §12.

**The diff confirms the scope claim, with one exception.** v10 changes exactly: line 1 (title), line 17 (STATUS), lines 28–37, lines 55–91 (§0), lines 1161–1170 and 1180–1182 (two loci in §6.1), and §12. `§1`, `§§2–5`, `§7`, `§8`, `§9`, `§10`, `§11` and the rest of `§6` are byte-identical to v9. §12's scope statement is exactly right. §0's is not — see I-1.

**Round 9 (0C/0H/3I) — two of three repairs land cleanly, one re-commits its own defect.**
- **I-2 — DISCHARGED, re-derived at source.** `HEADSPACE_TRANSFER_PREGATE.md:623` is character-exact: *"### 4.10 The transfer ladder (§2.10) — four operator families, one arena swap"*. The rung column at `:630–643` places the nine positives of `:859–864` into exactly four groupings. The "at least six" and the false "not re-derivable" clause are gone, and the "not one lineage" point is retained.
- **I-3 — DISCHARGED.** `:872-873` reads `(d_AUC −0.064 / −0.129), inverting the raw arena's +0.157 / +0.230` — 3 dp confirmed; the `30/30` fold count is at `:847` and `:901`. `dead[F113]` genuinely carries only `name` and `status`.
- **I-1 — NOT DISCHARGED IN §0.** §12's half is repaired correctly; §0's half re-commits the identical defect one revision later.

**Rounds 1–8 — every Critical and every High lands in v10.** Checked against the body, not the ledger: R1 C-1…C-4 and H-1…H-8; R2 C-1 and H-1…H-8; R3 C-1 and H-1…H-8; R4 C-1 and H-1…H-4; R5 H-1/H-2/H-3; R6 H-1/H-2; R7 H-1; R8 H-1.

---

## What I verified as sound

**Quotation fidelity: 36/36 exact.** Every registry field, every `ban_scope` (F47, F51, F66, F75, F96, F97, F98(b)+(d), F99, F112), every `.md` citation. No ellipsis hides material that changes a meaning — several hide clauses that read *more* adversely to C09. The disowned upper-case rendering ("may not PROMOTE") appears in no source outside C09's own earlier drafts, so that parenthetical is accurate.

**Every number re-derived from source.** All twelve floors; 30 `fold_acc_deployed` arrays; fold counts recomputed independently from `StratifiedKFold(5, shuffle=True, random_state=0)`; `raw_deployed_acc`; `B_fid`; per-seed errors exactly `83/85/85` and `62/64/61` from integer correct-counts; caps ⇒ `k ≥ 132/96` dead; `R(1.5×55) = 83` against banker's `82`; `0.896/0.663` and `0.898/0.675`; `π*` gap `9.3125` pt; Hanley–McNeil; `28.95`, `37.15`, the `[28.95, 29.0)` emptiness; `6/215`, `7/149`; `pred_agree` mapping; majority rates and both bands; `n_test` `215/149/161`. Both "re-measured this session" claims reproduce on my own reads, as does `GATE-NULL`.

**Executability is real; nothing would turn this into an engineering HALT.** `sha256(headspace_mint.py)` matches its banked anchor in all six arenas and `C02_A0_OUT.json::frozen_modules`; the real `frozen_artifact_policy` list (`progress.json:30`) genuinely does not contain it. Every cited line range is exact. `lab_dev` is on the unconditional `savez` path. `load_split` is called only with `"train"` and `"dev_seen"`; for `fold ≥ 0` the dev/test splits are a stratified slice of the *fitting pool*, so **no test path is reachable by any code path**. **The mint trains 30 fixed epochs with `torch.save` monkeypatched to a no-op and `best_epoch_path` never re-loaded — no dev-based model selection anywhere**, which is what makes the whole label-use spine hold. The live env is byte-identical to the banked `meta.runtime`. `spawn(6)` works; `int(np.floor(82.5+0.5)) = 83`. `GATE-FIXK20` is corroborated in all six banked arenas, and ZH seed 0's `B_agree_fixk["15"] = 1.0000` is exactly the `DEGENERATE_ALL_EMPTY` anchor §6.2 cites. `squeue` empty; C04's tranche terminated.

**The budget is sound and conservative.** Job `13847` ran 8 CPU / 0 GPU / 32 G in `00:29:49`. The 36 banked mint times are min `33.2`, max `60.0`, median `41.85`, total `24.51 min` — exact. Lines sum to 116 against "≈115", ~7× conservative.

**Statistical soundness and decidability.** `AUC_strat` fully pinned; `ΔAUC` paired. `PERM-STRUCT` exact for the marginal null over `P^{(τ)}`: only `struct` is permuted and `A^{(f)}` is measurable w.r.t. `(target, BASE)`, so the evaluation sets are invariant across draws — precisely the property exactness needs. `PERM-STRUCT-COND` exact for the joint form claimed. Requiring both families is an intersection, hence conservative; Holm within dataset is correct and the `1/1001` floor leaves `α/2` reachable; the dataset conjunction is a genuine IUT; the existential over `τ` is covered because a false CONTINUE requires a false rejection inside one dataset's Holm family. I re-derived the macro-F1 monotonicity — each FN→TP flip raises class-1 F1 iff `2(2TP+FP+FN) > 2TP`, always true, and raises class-0 F1 by shrinking its denominator — so `K-REACH` at `τ_0` genuinely closes every `τ ≥ 0`. `K-DEG` can both fire and clear at every live `k`. **No single rule can carry a CONTINUE.** A KILL is fully available and is the pre-declared expectation.

**On the residual leak path: none beyond the one §4.3 names.** I walked all 13 features. The right analogue does read `lab_query` and is excluded by `GATE-BLIND`'s guard on exactly that array. That last point is stronger than the document claims: because both permutations hold `target`, `BASE` and `A^{(f)}` fixed and permute only `struct`, any target-side contamination is present identically in the observed and permuted draws and is therefore inside the null's own calibration.

**Legality holds, in both directions.** F47 conceded outright; the F114 correction used only where it applies; F98(b) conceded as constructed-and-measured; `[9]` and `[2]`/`[4]` named; F96 treated on the statistic it was calibrated on; F51 now runs against C09's own registered claim.

**Scope honesty.** §10 claims nothing the design does not support and several things that cut against the candidate.

---

## IMPORTANT

**I-1 — §0's own scope sentence is false, is self-contradicted by the clause that follows it, and re-commits the exact defect R9 I-1 named — in the repair that closes R9 I-1.** Line 77 reads *"**Nothing in §§2–9 changed in v10, and nothing in §§1–11 changed in substance.** The edits are three sentences in §0/§6.1, the STATUS block, and §12."* §6.1 is inside §§2–9 (§6 begins at line 1119), and two of its sentences changed — as the very next clause says, as §12's scope statement says, and as the diff shows. R9 I-1's finding was "v9's change summary understates v9's own changes, in both places that state them"; the repair fixed §12 and re-committed the understatement in §0. §12's version is correct and is the one to mirror. **Repair:** "Nothing in §§1–5 or §§7–11 changed in v10; the edits are two sentences of §6.1, the STATUS block, §0 and §12, and no decision rule, threshold, feature set, null, gate or arithmetic has changed since v8."

**I-2 — §7 names `CONFIG-MATCHED-CORRECT` as a live control and rests the registry claim's break-exposure conjunct on it, but v10 nowhere defines the term and §5.3 calls the phrasing that contains it superseded.** The term appears exactly three times in v10: twice in §7, and once at line 980, where §5.3 calls `P_τ ∪ CONFIG-MATCHED-CORRECT` *"the superseded … phrasing"*. §5.2 and §6.3 — the sections §7 points at — do not contain it. The object also changed under the name: in v5 it was the *unrestricted* "query items correct at all three seeds"; after R5's H-3 and R6's H-1 repairs it is confidence-restricted **and** fold-dependent. The phrase "break-exposure stratification" likewise has no referent anywhere else in the document. This changes no rule, threshold or verdict, and break exposure genuinely *is* instrumented (§5.3's per-seed `net_s`, `GATE-SELFTEST`, and the reported class composition of `S`) — but the one conjunct of C09's registered claim that §7 asserts is "measured rather than asserted" is asserted, in a summary section, by reference to a name the frozen specification retired. This is the residue of the R7 I-2 / R6 H-1 repairs, which were scoped to §5.3 and §5.2 and never followed into §7; no round has audited §7. **Repair:** either delete the term and name the object §5.2 actually specifies, or re-introduce it as an explicit alias at its §5.2 definition; and attribute break exposure to what measures it.

---

## Checked and deliberately not counted

- **§6.1's "four operator families".** The source's four at §2.10 are VSW / F95 / fixed-k / **F89**, with `THRESH_best` un-runged; the nine positives occupy VSW / F95 / FIXK / un-runged `THRESH_best`. The two fours differ by one member. But v10's operative enumeration is the second one, stated exactly and labelled "un-runged", the `:623` quote is verbatim, and "the count is the source's four" is true as a count.
- **§2's ksweep parenthetical.** `curves[…]["dev"]` exists on every curve but is `null` for all four `MHC_EN_ARM-V/*/valsel` curves. The concession errs *against* the design, on a parenthetical to a secondary reason for skipping a clause CAL-2 itself declares non-gating.
- **§2's budget.** Lines sum to 116 (or 114 reading "< 1 min" as 0); "≈115" is the midpoint and the table is ~7× conservative.
- **§5.3's caps.** `2 × 84.3333 − 37.2` is literally `131.4666`; the printed `131.4667` is the exact-mean (`253/3`) value. Inert at integer `k`.
- **§5.3's "7.5 points at `k = 100`"** — true value `7.45`.
- **§5.3's "the six `(τ, k)` cells"** — the caps are per-dataset, so there are really 12 markings.
- **The R9 I-2 repair leaves a sentence fragment.** Prose, not science.
- **F66's own `ban_scope` closes *"Re-open requires an operator that converts symmetric structure — the beta decomposition proves none of the tried classes can"*,** a live headwind against §11's symmetric successor that the document never quotes. §11's precondition (d) already requires a fresh F66 adjudication, so this A0 does not turn on it — but the F99 leg is spelled out and the F66 leg is not.

---

## Bottom line

The science is finished and I could not move it, and neither of my two findings touches it. I re-derived every number from the named source on disk, re-opened all 36 quotations, walked all 13 features and all five thresholds for a residual gold-label path, re-checked the exactness argument for both permutation nulls, confirmed the four decision rules are jointly decidable and individually non-vacuous with no single rule able to carry a CONTINUE, confirmed the mint does no dev-based selection and that no test path is reachable, matched the mint's sha256 to its banked anchor, and confirmed the live environment is byte-identical to the banked runtime. The legality spine concedes every adjacent ban, including two — F51 and F99 — that run against C09's own registered claim.

What I cannot pass is two statements about the document's own contents. **I-1 is the one that matters**, because it is R9 I-1 re-committed inside the repair that closed R9 I-1. **I-2** is a stale object name in the controls summary, on the one conjunct of the registry claim whose instrumentation §7 asserts by reference to it.

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not yet — but only barely not.** Both repairs are edits to three sentences in §0 and §7. They need no re-measurement, no GPU, no re-derivation of anything in §§1–11, and they do not touch a single rule, threshold, feature set, null, gate or arithmetic. Fix them and the frozen set is ready: the instrument is on disk and sha-pinned, the queue is empty, C04's tranche has terminated, and the only remaining precondition outside the document is explicit main-dialogue authorization.


===== RAW REVIEW: refine-logs/C09_A0_V11_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v11 — Independent Design Review, Round 11

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V11_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 2 Important`

---

## Prior-round audit

I diffed v10 → v11 line by line (`diff -u`, 257 lines, 12 hunks) and re-verified R1–R10 against **v11 itself**, re-opening every source on disk rather than trusting §0 or §12.

**The diff, exactly.** v11 changes: line 1 (title); line 17 (STATUS); lines 27–38; lines 54–99 (§0); one sentence in §5.2; three sentences in §5.3; **one** sentence in §6.1; **one** sentence in §7; one sentence in §11; and §12. `§§1, 2, 3, 4, 5.1, 5.4–5.6, 6.2, 6.3, 8, 9, 10` are byte-identical to v10. No rule, threshold, feature set, null, gate or arithmetic moved.

**Round 10 (0C/0H/2I) — both findings discharged.**
- **I-1 — DISCHARGED.** v10's false *"Nothing in §§2–9 changed in v10"* is gone; §0 now mirrors §12's form.
- **I-2 — DISCHARGED, both halves.** §5.2 defines `CONFIG-MATCHED-CORRECT` at the point of definition with both changes of object named; §7 names `net_s` (§5.3, line 978), `GATE-SELFTEST` (§8.1, line 1427) and the reported class composition (§5.3, lines 1003–1005). All three referents check out.

**Round 10's five below-the-line items — all five fixed, all five correct.** The fragment; `7.45`; "twelve `(dataset, τ, k)` cells, six per dataset"; the exact means `253/3` and `187/3` (both re-derived, `k ≥ 132 / 96` dead either way); and F66's re-open clause quoted **verbatim** against `directions_tried.json` `dead[]` index 40 — character-exact, em-dash included.

**Rounds 1–9 — every Critical and every High still lands in v11.** In particular R4 C-1, R4 H-1, R6 H-1 and R7 H-1 are intact and were not disturbed by the §5.2 alias insertion.

---

## What I verified as sound

**Quotations: every one exact at its cited location.** All registry fields, all `ban_scope`s (F47, F51, F66, F75, F96, F97, F98(b)+(d), F99; F113's entry genuinely carries **no** `ban_scope`), all `.md` citations including `HEADSPACE_TRANSFER_PREGATE.md:623/:630-643/:859-864/:863-864/:871-873/:917-919/:920-923`, `ERRPAT_*`, `REDTEAM_BAN_SCOPE_AUDIT.md:293/:302-304/:305-308`, `RESTRANS:409`, both standing clauses, `progress.json:25/:30`. Every ellipsis I checked hides material that is neutral or adverse to C09. The disowned renderings are correctly attributed to the rendering source rather than the primary.

**Every number re-derived from source.** All twelve floors; `raw_deployed_acc`; per-seed errors `83/85/85` and `62/64/61`; `mean_s|wrong_s| = 253/3` and `187/3`; both caps; `R(1.5×55) = 83`; `114/83`, `152/110`, `75.6`, `56.0`, `0.896/0.663`, `0.898/0.675`; `π*` at `k=80` and `k=100`; the `[28.95, 29.0)` emptiness; the Hanley–McNeil SE re-derived from `Q1`/`Q2` at `A=0.5`; the `pred_agree` mapping; `6/215`, `7/149`; `n_test 215/149/161`; the fidelity anchors; F98's DEG-A; majority rates and both bands. Both "re-measured this session" claims reproduce on my own reads, as do the per-fold held-out counts from `StratifiedKFold(5, shuffle=True, random_state=0)`.

**Executability: nothing here becomes an engineering HALT.** `sha256(headspace_mint.py)` matches its banked anchor everywhere; **the mint's other run-time assertion — `sha256(mechnov_pairverify.py) == FROZEN_PAIRVERIFY_SHA` at `:188-189` — also matches on disk**, so the mint will not refuse to run. `--dataset` admits exactly `{hatemm, zh}`. `load_split` is called only with `"train"` and `"dev_seen"`; the `torch.load` guard bars `test_seen` and `/test`; `torch.save` is a no-op and `best_epoch_path` is re-loaded only on an EM branch this recipe never takes, so **there is no dev-based model selection anywhere** — the load-bearing fact behind the whole label-use spine. `GATE-FIXK20` is corroborated in all six banked arenas. The live environment is byte-identical to the banked `meta.runtime`. `squeue -u jehc223` is empty.

**Budget: sound and conservative.** Job `13847` ran 8 CPU / 32 G / no GPU in `00:29:49`. The 36 banked mint times re-derive to min `33.2`, max `60.0`, median `41.85`, total `24.51` min — exact. Every fit-count multiplies out; lines sum to 116 against "≈115".

**Statistics and decidability.** `AUC_strat` fully pinned; `ΔAUC` paired. `PERM-STRUCT` exact for the marginal null over `P^{(τ)}`: only `struct` is permuted and `A^{(f)}`/`P^{(τ)}` are measurable w.r.t. `(target, BASE)`, so the evaluation sets are invariant across draws. `PERM-STRUCT-COND` exact for the joint form claimed. Both families required = intersection = conservative; Holm correct; IUT correct; the existential over `τ` covered. I re-derived the macro-F1 monotonicity independently. All four rules individually non-vacuous and jointly decidable; **no single rule can carry a CONTINUE**; a KILL is fully available and is the pre-declared expectation.

**Residual gold-label paths: none beyond the one §4.3 names.** All 13 features walked. The one remaining channel is named, priced, and identical for `BASE` and `FULL`, so it is inside the paired difference and inside both nulls' own calibration.

**Legality holds in both directions.** F51, F99 and now F66 all run **against** C09's own registered claim or successor and are carried anyway.

---

## IMPORTANT

**I-1 — §6.3's and §5.3's pre-declared `n_unstable ≈ 7–9` is the *per-seed* count of non-stable errors, not the item-level unstable population, and it is arithmetically incompatible with the document's own `|P_0| ≈ 76 / 55`.** §4.2 defines an unstable error at the **item** level and `D-FELDMAN`'s target is per item, so `n_unstable` is an item count. Write `E_s` for the seed-`s` inversion set, `a` = #(wrong in exactly 2), `b` = #(exactly 1). Then `Σ_s |E_s| = 3|P_0| + 2a + b`, with `n_unstable = a + b`. The per-seed counts are **measured**: `83+85+85 = 253` (HateMM), `62+64+61 = 187` (MHC-ZH).

- HateMM at `|P_0| = 76`: `2a + b = 25` ⇒ `n_unstable ∈ [13, 25]`. The declared `7–9` is **unreachable** — the minimum is 13.
- MHC-ZH at `|P_0| = 55`: `2a + b = 22` ⇒ `n_unstable ∈ [11, 22]`; and the very F88 datum §4.2 quotes (*"NOTHING at exactly 2/3"*) forces `a = 0`, i.e. **`n_unstable = 22`, above the `20` trigger**, so `CONTROL_UNDERPOWERED` would *not* fire on that leg.
- Conversely, `n_unstable ≤ 9` forces `|P_0| ≥ 78.3` / `≥ 56.3`, contradicting `≈ 76 / 55` and shifting `|P_{τ_hi}|` off the `38 / 28` on which §5.2's *"the `τ_hi` branch cannot produce a CONTINUE at all"* and §10's ZH bullet both rest.

The origin is visible: `84.33 − 76 = 8.33` and `62.33 − 55 = 7.33` — the per-seed excess. **Why Important and not High:** no decision rule reads it, `UNSTABLE-POP` is explicitly non-gating and its power rule is data-driven, and each of the two incompatible statements is individually conservative in its own context. But the document cannot pre-declare both, and one of the two things §10 will publish with the verdict is arithmetically impossible. **Repair:** keep `|P_0| ≈ 76 / 55` and declare `n_unstable ∈ [13, 25] / [11, 22]`, noting that on the ZH transfer the control may in fact be powered on the count leg, with §10's stability-premise bullet re-worded to depend on the realised tag.

**I-2 — v11's own scope statement, in both places, mis-describes v11's own contents.** (1) **§6.1: one sentence changed, not two** — the preceding sentence is byte-identical; "two" is v10's edit count carried into v11's description of v11. (2) **§7: one of the two `CONFIG-MATCHED-CORRECT` sentences was edited**; the controls-enumeration sentence is byte-identical between v10 and v11. (3) **§0 says "one line of §5.2"; §12 says "one sentence"** — of the same edit, in the two statements §0 says it is mirroring. This changes no rule and the substantive invariant attached to it is **true** — I verified it against the diff — but it is the fourth consecutive round in which the change summary misdescribes its own contents, and it is the one sentence whose entire job is to be checkable.

---

## Checked and deliberately not counted

§5.3's "superseded phrasing" note (still true after the alias insertion, since `P_τ` and `A^{(f)}` remain different objects); §11's F66 paragraph in mild tension with the NCA counterweight (runs **against** the candidate, Stage-1-scoped, already covered by precondition (d)); F75's semicolon truncation (nothing adverse hidden); typographic normalisation inside quotations; §4.3's `≈ 1.4 %` for a withdrawn v4 construction; §2's budget summing to 116 against "≈115"; §2's `ksweep` parenthetical erring against the design; §5.3's R10 attribution slack.

---

## Bottom line

I could not move the science, and neither of my findings is a defect in it. I re-derived every number in §§1–11 from the named source on disk, re-opened every quotation, walked all 13 features and all five thresholds for a residual gold-label path, re-derived both permutation nulls' exactness arguments and the Hanley–McNeil algebra from scratch, confirmed the four decision rules are jointly decidable and individually non-vacuous with none able to carry a CONTINUE alone, confirmed the mint performs no dev-based model selection and that no test path is reachable, matched **both** of the mint's run-time sha256 assertions to their on-disk targets, and confirmed the live environment is byte-identical to the banked runtime. The legality spine concedes every adjacent ban, including three — F51, F99 and now F66 — that run against C09's own registered claim or successor.

**I-1 is the one that matters**, and it is the first arithmetic error found in this document since round 6. **I-2** is the change summary mis-describing its own diff for the fourth round running.

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not yet — but the gap is one paragraph and one sentence.** Neither repair needs re-measurement, GPU, or any change to a rule, threshold, feature set, null, gate or arithmetic. With those made, the frozen set is ready.


===== RAW REVIEW: refine-logs/C09_A0_V12_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v12 — Independent Design Review, Round 12

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V12_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 1 High / 2 Important`

---

## Prior-round audit

I diffed v11 → v12 (15 hunks) and re-verified R1–R11 against **v12 itself**, re-opening every source on disk.

**v12's scope claim checks out on its load-bearing half.** `§§1, 2, 3, 4, 5.1–5.2, 5.4–5.6, 6.1, 6.2, 7, 8, 9 and 11 are byte-identical to v11` — verified hunk-by-hunk, **exactly right for the first time in five rounds**. §0 and §12 carry the identical scope paragraph. No rule, threshold, feature set, null, gate or arithmetic moved.

**Round 11 (0C/0H/2I).**
- **I-2 — DISCHARGED.** Stated once, in one convention, in both places, with a correct and checkable byte-identical enumeration.
- **I-1 — the repair is internally valid but corrects the wrong term.** The identity, the intervals and the ZH `a = 0` step all re-derive exactly as written. But the input it anchors on (`|P_0| ≈ 76 / 55`) is itself the mis-derived quantity, and R11's `7–9` was approximately right. See HIGH.

**Rounds 1–10 — every Critical and every High still lands in v12**, including R4 C-1, R4 H-1/H-3/H-4, R5 H-1/H-2/H-3, R6 H-1/H-2, R7 H-1, R8/R9/R10's items. R10's five below-the-line fixes survive unchanged.

---

## What I verified as sound

**Every number re-derived from source**, including all twelve floors, per-seed errors `83/85/85` and `62/64/61`, `Σ = 253 / 187`, the caps from the exact means, `R(1.5×55) = 83`, the `π*` figures, the `[28.95, 29.0)` emptiness, Hanley–McNeil re-derived from `Q1`/`Q2`, the `pred_agree` mapping, **F98's DEG-A re-read from `aggnet_main_*_OUT.json` = `0.9570 / 0.9508`**, F113's `THRESH_best +0.0041`, `n_test 215/149/161`, the fidelity anchors, both data-defect measurements, `GATE-NULL`, `sacct` for job `13847`, and the 36 banked mint times.

**Quotations.** Every registry field, every `ban_scope`, every `.md` citation verifies character-exact modulo dash/quote normalisation. **F113's `dead[]` entry is the only one of 76 with keys `["name","status"]` only** — no `ban_scope`. The disowned renderings are all correctly attributed to the rendering source.

**Executability: nothing becomes an engineering HALT.** Both of the mint's run-time sha256 assertions match on disk. `--dataset` choices are exactly `['hatemm','zh']`. `best_epoch_path` is reloaded only inside the EM branch which `--seg_mode full` never enters, so **the keys are final-epoch and there is no dev-based model selection anywhere**. `GATE-FIXK20` corroborated in all six banked arenas and independently in F98's three. Live environment byte-identical to the banked `meta.runtime`. `squeue` empty.

**Statistics and decidability.** `AUC_strat` fully pinned. `PERM-STRUCT` exact for the marginal null (`A^{(f)}`, `P^{(τ)}` and the frozen strata are all measurable w.r.t. `(target, BASE)`, so evaluation sets are draw-invariant). `PERM-STRUCT-COND` exact for the joint form. Both families = intersection = conservative; Holm correct; IUT correct. Macro-F1 monotonicity re-derived independently. **No single rule can carry a CONTINUE**; a KILL is fully available at every branch.

**Residual gold-label paths: none beyond the one §4.3 names.** All 13 features walked. No test-like cache reachable.

**Legality holds in both directions.** F51, F66 and F99 all run against C09's own claim or successor and are carried anyway. I enumerated both constraint arrays independently.

---

## HIGH

**H-1 — v12's headline repair corrects the wrong term: the error was never `n_unstable ≈ 7–9`, it is `|P_0| ≈ 76 / 55`, and the "corrected" figures are now wrong on both datasets.**

`|P_0|` and `n_unstable` are two ratios of the *same* F88 seed-consensus table and must be denominated on the only quantity measured in the C09 arena, the per-seed error count. Transferred that way from the primary tables (both **final-epoch**, the protocol C09's mint produces):

| | `ERRPAT_HateMM §1.1` (final) | `ERRPAT_MHC-ZH §1` |
|---|---|---|
| per-seed errors | `26/27/27` ⇒ `Σ = 80`, mean `26.667` | `23/24/22` ⇒ `Σ = 69`, mean `23` |
| `3/3` (`\|P_0\|`) | `25` | `22` |
| `2/3` (`a`) / `1/3` (`b`) | `2 / 1` ⇒ `n_unstable = 3` | `0 / 3` ⇒ `n_unstable = 3` |
| `\|P_0\|` / per-seed mean | `0.9375` | `0.9565` |
| scale to C09 (`84.333` / `62.333`) | `×3.1625` | `×2.7101` |
| **⇒ `\|P_0\|`** | **`79.1`** | **`59.6`** |
| **⇒ `n_unstable`** | **`9.5`** | **`8.1`** |

Both reproduce `Σ_s|E_s|` exactly: `3(79.06)+15.81 = 253.0` and `3(59.62)+8.13 = 187.0`.

So **`n_unstable ≈ 7–9` was right to within rounding, and `|P_0| ≈ 76 / 55` is the mis-derived term.** The ZH origin is diagnosable to the character: `55 = 0.88 × 62.333`, where `0.88 = 22/25` is F88's **union**-denominated rate applied to a **per-seed** base; the per-seed rate is `22/23 = 0.9565 ⇒ 59.6`. HateMM's `76 ≈ 0.90 × 84.333` takes the low end of a `"≈89-93%"` band against a fuzzy `"~26-28"`; the protocol-matched rate is `25/26.667 = 0.9375 ⇒ 79.1`. The two datasets were computed by *different* conventions, which is exactly why the incompatibility exists.

R11 correctly proved the two could not coexist, then resolved it by keeping the wrong term. v12 has written that into four sections: **§6.3** (`n_unstable ∈ [13,25]/[11,22]`, and *"on MHC-ZH … `n_unstable = 22`, above the `20` trigger"* — the coherent figures are `9.5 / 8.1`, both **under**, i.e. v11's retracted *"expected on both datasets"* was correct); **§5.3**; **§10**; **§12**. The ZH combination implies a 3/3-stability fraction of `55/77 = 71 %` against the `88 %` F88 states; HateMM's implies `75–85 %` against `89–93 %`.

And the same `|P_0|` propagates through `|P_{τ_hi}| ≈ |P_0|/2`: **`38 / 28` becomes `≈ 40 / 30`**, inverting four further pre-declarations v12 left untouched — §5.2's *"On MHC-ZH `28 < 30`, so the tag fires by arithmetic … the `τ_hi` branch cannot produce a CONTINUE at all"* (at `30` the cell is LIVE); §5.2's `K-REACH`-at-`τ_hi` arithmetic; §5.2's *"`|P_0| ≳ 60` against a declared expectation of `≈ 55`"* (the corrected expectation *is* `≈ 60`); §5.3's *"strictly impossible on MHC-ZH whenever `|P_{τ_hi}| < 29`"*; and §10's *"expected to be closed on power on MHC-ZH"*.

**Why High and not Critical.** No decision rule reads an F88 number; every cap, tag and trigger is computed in-run. **Why High and not Important.** These are the design's pre-registered statement of *what it can adjudicate*, on one of exactly two datasets in a design §10 calls "zero slack", and §9 requires the KILL record to be scoped by them. Eleven rounds could not check it because **the document nowhere shows the derivation of `|P_0| ≈ 76 / 55`**.

**Repair (text only).** Transfer both ratios per-seed-denominated from the primary tables, show the arithmetic, and restate: `|P_0| ≈ 79 / 60`, `|P_{τ_hi}| ≈ 40 / 30`, `n_unstable ≈ 9 / 8`. Then re-word §6.3, §5.3's out-of-support figure, §10's bullet, and the four `τ_hi` sentences that turned on `28 < 30`. Keep §10's stronger new framing — *"decided by the realised tag, not in advance"* — which is right under either transfer.

---

## IMPORTANT

**I-1 — §6.1's *"the count is the source's four"* equivocates between two different fours; the source's four does not contain the nine.** `:623`'s rung column assigns **rung 1 = VSW, rung 2 = F95, rung 3 = FIXK, rung 4 = F89** (`:644-647`), with `THRESH_best` deliberately **un-runged** (`:643`). The nine raw positives occupy rungs 1, 2, 3 and the un-runged arm — **no rung-4 arm at all** (F89's RAW column is `—`). So the document's four (`VSW/F95/FIXK/THRESH_best`) and the heading's four (`VSW/F95/FIXK/F89`) are *different sets of the same cardinality*, and the appeal to `:623` does not license the count. The substantive claim (*"not one lineage"*) stands; the sourcing sentence fails. Worth noting alongside: **rung 4 is the four closed-form geometric head-key transforms — the family nearest to C09's own structural block — and it reads `+0.0000 / −0.0004 / +0.0027 / +0.0054`, i.e. null-to-sub-noise with a cross-dataset sign flip.** That is adverse to C09 and is carried nowhere.

**I-2 — `F114` is cited three times with no locator, and the bare number resolves to a different finding in the repository's own findings file.** The finding meant is the CLIP-LOO erratum at `autoresearch/goal_mllm_plus3/state/findings.jsonl:115`. But `TARGET_FINDINGS.md:79` is headed **`### F114 — the v6 teacher producer could never have produced a teacher response`** — an unrelated C04-lineage finding. Every other finding the document leans on is given with a path. The three numbers themselves verify. **Repair:** one locator at first use.

---

## Checked and deliberately not counted

"three sentences of §6.3" understating the *result* by ~3× while being true of the edit's footprint; `GATE-LEDGER`'s trainlog reads being of mixed files whose safety rests on the hard `VAL_RE` filter (never named, but no test label is materialised and `GATE-DEVFID` gates nothing); §5.6's "30-epoch Adam" where `run_rac.py:684` uses `AdamW`; `load_split` calling `_ORIG_TORCH_LOAD` (safe because both call sites are literals); `ITEM-STRATUM` occupancy not emitted the way the row strata are (direction conservative); the ellipsis omissions that all run **against** C09 (F98's head clause and (c), F97's *"DO NOT PROMOTE…"*, `banned_constraints[9]`'s parenthetical, ERRPAT's train-LOO recalibration); `queued.screening_arena_switch`; §2's budget and `ksweep` parenthetical; §4.3's `≈ 1.4 %`; §2's "~25–60 s" against a measured `33.2`.

---

## Bottom line

The instrument, the legality spine, the executability, the budget and the inference are all intact, and I could not move any of them. Every gate is corroborated on disk, both of the mint's run-time hashes match, the runtime is byte-identical, no test path is reachable, and the four decision rules remain jointly decidable with none able to carry a CONTINUE alone. A KILL remains fully available and is still the honest expectation.

But **v12's one substantive repair is aimed one term to the left.** The pre-declared pair `(|P_0|, n_unstable)` has to be transferred jointly, per-seed-denominated; done that way it gives `(79, 9.5)` and `(60, 8.1)` and reproduces `Σ_s|E_s| = 253 / 187` exactly. R11 proved the pair could not coexist and chose to keep `76/55`; the arithmetic says the opposite. v12 propagated that choice into §5.3, §6.3, §10 and §12, and left four further `τ_hi` pre-declarations standing on `≈ 28` when the corrected figure is `≈ 30` — the exact value at which "arithmetically dead" flips.

**Is this ready to hash-freeze and submit? No — not because of the science, which is finished, but because the round-12 repair itself needs re-doing.** The fix is arithmetic on paper: four lines of derivation shown once, `|P_0| ≈ 79 / 60` substituted, and seven sentences restated. It needs no re-measurement, no GPU, and no change to any rule, threshold, feature set, null, gate or operating point. With those made — and with the derivation *shown*, so that a thirteenth reviewer can check in two minutes what eleven could not check at all — the frozen set is ready.


===== RAW REVIEW: refine-logs/C09_A0_V13_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v13 — Independent Design Review, Round 13

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V13_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 5 Important`

---

## Prior-round audit

I diffed v12 → v13 (nine content hunks plus title/STATUS) and re-verified R1–R12 against **v13 itself**, re-opening every source on disk.

**v13's scope claim: the load-bearing half is exact.** The enumeration `§§1, 2, 3.2–3.4, 4.1, 4.3, 4.4, 5.1, 5.4–5.6, 6.2, 7, 8, 9 and 11 are byte-identical to v12` is **verified hunk-by-hunk and correct**. No decision rule, threshold, feature set, null, gate or operating-point definition moved. The prose *describing* the edits does not check out; see I-5.

**Round 12 (0C/1H/2I).**
- **H-1 — DISCHARGED on substance, with residue.** The joint per-seed-denominated transfer is derived in §4.2, and I re-derived every step independently. `|P_0| ≈ 79 / 60`, `n_unstable ≈ 9 / 8`, `|P_{τ_hi}| ≈ 40 / 30` are **correct**. §5.2's knife-edge, §5.3's whole `k` grid, §6.3's UNSTABLE-POP expectation and §10's stability bullet all re-derive exactly. Residue: the derivation block mis-transcribes and mis-locates its own primary source (I-1); one knife-edge leg is misattributed (I-4); four statements elsewhere still stand on the retired `76 / 55` (I-2).
- **I-1 — DISCHARGED with a new defect.** The two-fours equivocation is retired and rung 4's adverse numbers are carried; the new paragraph mis-describes its source (I-3).
- **I-2 — DISCHARGED.** `findings.jsonl:115` is the CLIP-LOO erratum; `TARGET_FINDINGS.md:79` is the unrelated C04-lineage `F114`. The locator sits at the genuine first use.

**Rounds 1–11 — every Critical and every High still lands in v13.** One discharged Important has regressed: **R11 I-2** (I-5).

---

## What I verified as sound

**The central derivation — re-derived from scratch, and it is right.** `ERRPAT_HateMM §1.1` final row: `187 / 1 / 2 / 25`, union `28`, sums to `n_test = 215` ✓; identity `3(25)+2(2)+1 = 80` ✓. `ERRPAT_MHC-ZH` consensus: `22 / 0 / 3 / 124`, union `25`, sums to `149` ✓; identity `3(22)+0+3 = 69` ✓. Means `26.6667` / `23` ✓; rates `0.9375`, `0.95652`, `0.1125`, `0.13043` ✓. Arena per-seed errors `83/85/85` and `62/64/61`, means `84.3333` / `62.3333` ✓ — floors re-read from all six arena JSONs and corroborated by `C02_A0_OUT.json`. Scaling: `79.0625 / 59.6232`; `9.4875 / 8.1304`. Closure: `3(79.0625)+15.8125 = 253.0` ✓ and `3(59.6232)+8.1304 = 187.0` ✓.

**The protocol choice is right and consistently applied.** The mint runs `--epochs 30` with checkpoint writes suppressed and `best_epoch_path` reachable only on an EM branch this recipe never takes — **final-epoch, no checkpoint selection**. HateMM's `final` row is the matching row; MHC-ZH's inventory is headed *"(Tier 2, final-epoch protocol)"*.

**Every downstream statement re-derives.** §5.2 reach `0.05376 / 0.05181`; power `30/40 = 75 %` and ZH needing all `30`; the withdrawal of *"cannot produce a CONTINUE at all"* is correct. §5.3 caps `131.4667 / 95.6667`; `R(2|P_0|) = 158/120` dead; `R(118.5) = 119`; `(119+37.2)/2 = 78.1 ⇒ 0.9261 / 0.6563`; `(90+29)/2 = 59.5 ⇒ 0.9545 / 0.6611`; at `k = |P_0|` `0.6889/0.7354` and `0.7139/0.7417`; τ_hi cells all inside the caps at `π* = 0.965/0.810/0.7325` and `0.9833/0.8222/0.7417` — **every one of the twelve figures reproduces.** §5.3's `≈ 9 / 8`, §6.3's ratios and §10's bullet are consistent.

**Everything else numeric** re-derived, including the Hanley–McNeil algebra, the π* gaps, the `[28.95, 29.0)` emptiness, F98's DEG-A from `aggnet_main_*_OUT.json`, `n_test 215/149/161`, both data defects re-measured this session, and `GATE-NULL` re-measured from the two operative caches.

**Quotations.** Every registry field and every `ban_scope` verifies character-exact modulo dash/quote normalisation. **F113's `dead[]` entry carries keys `["name","status"]` only.** All `.md` locators land. Every ellipsis omission still runs **against** C09.

**Executability: nothing becomes an engineering HALT.** All three relevant sha256 match (mint against the banked `meta.mint_script_sha256`; `mechnov_pairverify` against the run-time assertion; `mechfix_ops` against F113's record). Every cited line number is exact. `GATE-FIXK20` corroborated. `sacct` confirms job `13847`. `squeue` empty. Budget sums to ≈116 against ≈115.

**Statistics and decidability.** `AUC_strat` fully pinned; both nulls exact for the nulls they name; both families intersected ⇒ conservative; Holm and the IUT correct; `p`-floor `1/1001 ≪ α/2`. Macro-F1 monotonicity re-derived independently. **No single rule can carry a CONTINUE**; at least one `LIVE_ON_NET` cell exists per (dataset, τ).

**Residual gold-label paths: none beyond the one §4.3 names.** All 13 features, both stratifications, `τ_hi^{(f)}`, the standardisation, the permutation pools and the head-training partition walked.

**Legality holds in both directions.** The three that run **against** C09 (F51, F66's re-open sentence, CAL-5) are carried anyway.

---

## IMPORTANT

**I-1 — §4.2's derivation block, whose whole purpose is checkability, mis-transcribes and mis-locates its primary source in three places.** (a) **The HateMM per-seed triple is wrong.** v13 writes `26 / 27 / 27`; `ERRPAT_HateMM §1`'s cell table gives, for the **final** protocol, `s0 = 28`, `s1 = 26`, `s2 = 26` — i.e. **`28 / 26 / 26`**. The sum `80` and mean `26.6667` are right and the row is pinned by `(25, 2, 1)`, so nothing downstream moves. (b) **The ZH locator is off by a section:** the seed-consensus table is `ERRPAT_MHC-ZH` **§2** (*"ERROR INVENTORY (Tier 2, final-epoch protocol)"*), not §1, which *"avoids per-item claims entirely"*. (c) **The protocol note's counterfactual is wrong:** the val-sel mean is `79/3 = 26.3333`, so `24/26.3333 × 84.3333 = 76.86` ⇒ **`≈ 77`**, not `78`. (d) Minor: the val-sel row is rendered `24 / 3 / 2` in the source's order two rows below a table using `(|P_0|, a, b)`.

**I-2 — four statements still stand on the retired `|P_0| ≈ 76 / 55`.** §5.2's *"the same ≈76 values"*; §6.2's `k ≈ 76` mapping (`1 − 0.204(1−ov)`, `ov ≈ 0.755`, `0.9898` — corrected: `79`, `0.2124`, `0.765`, `0.9894`); §5.3's `R(x)` example on `1.5 × 55 = 82.5` (ZH is now the integer `90`; the live hazard is HateMM's `1.5 × 79 = 118.5`); §0's history bullet `k/n ≈ 76/744`. None is read by a rule, which is why this is Important — but round 12's High was precisely that `|P_0|` was inconsistent, and v13 leaves the inconsistency live in three specification-adjacent places.

**I-3 — §6.1's new rung-4 headwind mis-describes the table it cites.** §4.10 is headed *"3-seed means, head arena. RAW column re-read from `vsw_main_hatemm_OUT.json`"* — a **HateMM-only** ladder whose three-entry column is **per seed**, not per dataset, so *"per-dataset first entries"* is wrong and **no cross-dataset comparison exists at `:644-647`**. A cross-dataset sign flip does exist and is **stronger**: `:784` gives ZH `F89_T4 −0.0063` against HateMM's `+0.0054`, and `:786-790` records the MECHFIX head-space test reads against this arena's, concluding *"T2b/T4 are sub-0.007 and sign-unstable across the two head-space arenas"*; `:902` books it as *"same magnitude, opposite sign."*

**I-4 — §5.2's reach bullet misattributes one knife-edge leg: on MHC-ZH a one-item shortfall closes the branch on *power*, not reach.** `K-REACH` uses the exact `+0.050` rate, so the integer bar is `|P| ≥ 29`; at `29` reach still clears (`0.05009`) and it takes **two** to fail on reach. What a one-item shortfall does is fire `p_w < 30` ⇒ `K-FELDMAN` fails by construction — the **power** leg the next bullet describes.

**I-5 — the scope statement diverges between §0 and §12 again (R11 I-2 regression), and §0's counts understate the diff.** §0 uses "four sentences of §5.2 / four of §5.3 / three of §6.1 / three of §6.3"; §12 uses "one paragraph of §5.2 / one paragraph + one list item of §5.3 / two paragraphs of §6.1 / one block of §6.3". Different units, different counts, different ordering — and both understate the actual hunks. The **byte-identical enumeration is identical in both places and correct**, which is the half that matters.

---

## Checked and deliberately not counted

§10's *"expected to be closed on power on MHC-ZH"*; §4.3's `|P_0|/n ≥ 0.10` remark (now `0.1062 / 0.1036`, **better** than under the retired figures); the `≈ 1.4 %` bank-channel magnitude (`1/79 = 1.27 %`); `GATE-LEDGER`'s trainlog reads of mixed files, safe on the hard `VAL_RE` filter; §5.6's *"30-epoch Adam"* where `src/run_rac.py:684` uses `AdamW`; `ITEM-STRATUM` occupancy not emitted the way the row strata are; `queued.screening_arena_switch`; §2's *"~25–60 s"* against a measured `33.2`; the mint-timing figures; the `(τ, k)`-cell wording.

---

## Bottom line

**The science is finished and I could not move it — for the fifth round running.** The instrument, the legality spine, the label-use discipline, the executability, the budget and the inference all hold under independent re-derivation. A KILL is fully available and remains the honest expectation.

**Round 12's High is discharged.** I re-derived `|P_0| ≈ 79 / 60`, `n_unstable ≈ 9 / 8` and `|P_{τ_hi}| ≈ 40 / 30` myself; the identity closes exactly on both datasets; the final-epoch protocol is right and consistently applied; and every downstream figure reproduces.

What remains is five text-level defects, four of them **inside the material v13 added this round**, none of which touches a rule, threshold, feature set, null, gate, operating point or verdict. Three are provenance failures of exactly the kind the round-12 High was about.

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not yet — but only barely not.** Every repair is a text edit. With those made, v14 is ready to freeze and submit as one 8-CPU / 32-GB / no-GPU / no-`--time` job of ≈115 CPU-minutes, subject to the three submission preconditions the STATUS block names and the immediately-prior `squeue` empty-check, sha256 re-verification and namespace-absence check.


===== RAW REVIEW: refine-logs/C09_A0_V14_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v14 — Independent Design Review, Round 14

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V14_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 3 Important`

---

## Prior-round audit

I diffed v13 → v14 (21 raw hunks over ten edit sites), re-opened every source on disk, and re-derived §4.2's transfer table and every downstream figure myself.

**v14's scope claim — the load-bearing half is exact.** The enumeration `§§1, 2, 3, 4.1, 4.3, 4.4, 5.1, 5.4–5.6, 6.3, 7, 8, 9, 10 and 11 are byte-identical to v13` is **verified correct and complete**. The *count* is not (I-3).

**"No decision rule … has changed since v8" — verified directly.** I diffed v8 → v14: **§8 (all ten gates) and §9 (the whole decision rule) are byte-identical to v8.** Every v8→v14 hunk elsewhere is prose or a transferred expectation.

**Round 13's five findings.**
- **I-1 — DISCHARGED in full, verified at source.** `ERRPAT_HateMM §1`'s cell table gives final rows `28 / 26 / 26`, Σ = 80, mean 26.6667 ✓; §1.1's final row `187/1/2/25` sums to 215 ✓; identity `3(25)+2(2)+1 = 80` ✓. `ERRPAT_MHC-ZH §2` is headed exactly *"ERROR INVENTORY (Tier 2, final-epoch protocol)"* ✓; consensus `22/0/3/124`, union 25, sum 149 ✓. Val-sel `(24, 2, 3)`, per-seed `26/27/26`, Σ = 79, mean 26.3333, `24/26.3333 × 84.3333 = 76.86 ⇒ ≈ 77` ✓, in the block's own order ✓.
- **I-2 — DISCHARGED for three of four.** §5.2, §5.3 (`np.round(118.5) = 118` vs `R(118.5) = 119` confirmed in the live env) and §6.2 all re-derive. §0's history bullet had its numerator re-based and its derived figure left behind → I-1.
- **I-3 — DISCHARGED on substance, new locator defect.** `:623`, `:625`, `:643`, `:644-647`, `:784`, `:902` all verify. `:786-790` does not → I-2.
- **I-4 — DISCHARGED in full.** `37/744 = 0.049731`, `38/744 = 0.051075`, ZH bar `28.95 ⇒ ≥ 29`, `28/579 = 0.048359`, `29/579 = 0.050086`. All four re-derive; the reach/power attribution is now correct.
- **I-5 — DISCHARGED on convergence, not on count.** The two scope paragraphs are byte-identical apart from the intentional cross-reference (diffed programmatically). The count is wrong → I-3.

**Round 12's H-1 and every Critical and High from R1–R11 still land in v14.**

---

## What I verified as sound

**The transfer, re-derived from scratch.** Rates `0.9375`, `0.95652`, `0.1125`, `0.13043`; arena per-seed errors `83/85/85` and `62/64/61`, means `84.3333` / `62.3333`; scaling `79.0625 / 59.6232` and `9.4875 / 8.1304`; closure exact on both datasets; `|P_{τ_hi}| ≈ 40 / 30`.

**Every downstream figure.** Caps `131.4667` / `95.6667`; `R(2|P_0|) = 158/120` dead; `R(1.5 × 79) = 119`; `78.1 ⇒ 0.9261 / 0.6563`; `59.5 ⇒ 0.9545 / 0.6611`; at `k = |P_0|`, `0.6889/0.7354` and `0.7139/0.7417`; all six τ_hi cells inside the caps at `π* = 0.965/0.810/0.7325` and `0.9833/0.8222/0.7417`. **All twelve reproduce.** π* gaps `9.31` / `7.45`; `[28.95, 29.0)` empty of multiples of 1/3; Hanley–McNeil `0.0527` / `0.0913`; §12's `0.1062 / 0.1036` and `1/79 = 1.27 %`.

**Statistical soundness and decidability.** `AUC_strat` fully pinned; both nulls exact for the nulls they name with the weaker-independence caveat carried; one shared `π` is conservative in both places; Holm + IUT level-controlled; `p`-floor `≪ α/2`. **Bands A′/A/B/C are exhaustive and mutually exclusive on completed runs and map exactly onto §9's CONTINUE.** At least one `LIVE_ON_NET` cell exists per `(dataset, τ)` and no identifiability cell is arithmetically dead at the expected sizes — nothing is undecidable by construction. `IDENTIFIABILITY_UNDERPOWERED` cannot appear on a CONTINUE.

**Legality and ban-scope.** Every `ban_scope` quotation verifies character-exact. **F113's `dead[]` entry carries keys `["name","status"]` only.** The three texts that run **against** C09 are carried anyway.

**Executability, instrument, budget.** All three sha256 match on disk. `frozen_artifact_policy` names four modules and **not** the mint — §2's distinction is right. Every cited line lands, including `VAL_RE` (Test_Retrieval lines cannot be parsed, so `GATE-LEDGER`'s six trainlog reads are safe). `GATE-FLOOR`'s 12 anchors, the fold arrays, `raw_deployed_acc`, the fidelity anchors, `C02_A0_OUT.json`'s `ARENA2` and `EMPTY_TEXT = 39`, `aggnet_pregate.py:534`, DEG-A `0.9570 / 0.9508`, and ZH seed-0's `B_agree_fixk["15"] = 1.0` all re-read exactly. `sacct` confirms job `13847`. `squeue` empty. `default_rng(20260801).spawn(6)` works. Budget sums to ≈116 against ≈115.

**Re-measured this session, independently:** `243/579`, `0.5802 / 0.1161 / 0.3109` (5.0×), `39/744` and `0` on ZH, the majority rates and both `GATE-ARENA` bands, and `GATE-NULL` on both caches.

**Label-use discipline holds.** No test path is opened; no test-derived quantity is materialised anywhere, including CAL-2(2)'s omitted comparator. The one residual bank channel is correctly identified as cancelling in the paired `ΔAUC`. The 36 dev-label materialisations are declared and fenced.

---

## IMPORTANT

**I-1 — §0's `K-DEG` bullet was half-repaired: the numerator was re-based on `79`, the figure it derives was not.** At `k = 79`, `2k/n = 0.212366` and `ov = 0.95` gives `0.98938 ⇒ 0.9894`. `0.9898` is the retired-`76` value. §6.2 states this correctly **and explicitly labels `0.9898` as what "the retired `76`" gives** — so §0 now derives from `79` a number §6.2 attributes to `76`. No rule reads it; §0's own header says *"Nothing in this subsection is a rule."*

**I-2 — the I-3 repair cites the MECHFIX cross-check to lines that do not contain it, in three places.** `:786-790` is the ZH degeneracy block. The MECHFIX paragraph — the `T1/T2a/T2b/T4` test reads, this arena's `−0.0006 / +0.0000 / −0.0040 / −0.0063`, and the *"sign-unstable across the two head-space arenas"* conclusion — is at **`:791-796`**. The quoted text and every number are exact; only the locator is wrong, at §0, §6.1 and §12. **Note the provenance: round 13's own review supplied `:786-790`, and the repair adopted the reviewer's locator without re-opening the file — which is precisely how R13 I-3 arose.** `:784` and `:902` are correct.

**I-3 — the unified scope statement miscounts its own diff.** Both copies read *"**Nine hunks:** …"* and then enumerate **ten** items. It also omits an eleventh edit site: the document trailer (`*v13.` → `*v14.`), the counterpart of "the title". R13 I-5's substantive requirement — one convention, byte-identical in both places — **is met**; what regressed is the arithmetic in the sentence that reports the count.

---

## Checked and deliberately not counted

§4.3's `≈ 1.4 %` (now `1/79 = 1.27 %`, disclosed); §2's `~25–60 s` and the ≈115-vs-116 rounding; §5.6's *"30-epoch Adam"* vs `run_rac.py:684`'s `AdamW`; `ITEM-STRATUM` occupancy not emitted the way the row strata are; `queued.screening_arena_switch`; the `(τ, k)`-cell wording; §6.2's DEG-A `0.9570` being the HateMM **seed-0** cell (3-seed mean `0.9575`, a headwind either way); §3's rendering of `LITSWEEP3:91`, a compression inherited verbatim from the registry and running against C09; §9's raw-arena leg with CAL-3 discharged by a comparator of record, correctly labelled.

---

## Bottom line

**The science is finished and I could not move it — for the sixth round running.** The instrument, the arena, the legality spine, the label-use discipline, the executability, the sha256 pins, the budget and the whole inferential apparatus hold under independent re-derivation from the primary artifacts. Round 12's High and all five of round 13's Importants are discharged on substance. **A KILL remains fully available and is still the honest expectation**, most likely on the τ_hi power leg on MHC-ZH, with τ_0 as the branch where identifiability is actually adjudicated.

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not yet — but only barely not, and for the third round running the blockers are entirely outside §§1–11's governing text.** All three findings are one-line text edits: one stale derived figure in a subsection the document itself declares non-governing (`0.9898 → 0.9894`), one line-range locator wrong by five lines in three places (`:786-790 → :791-796`), and one count wrong by one in a sentence that reports the diff. None touches a rule, threshold, feature set, null, gate, operating point, verdict or scope of any conclusion.

With those three edits made, **v15 is ready to freeze and submit as one 8-CPU / 32-GB / no-GPU / no-`--time` job of ≈116 CPU-minutes**, subject to the three submission preconditions the STATUS block names and the immediately-prior `squeue` empty-check, sha256 re-verification of the frozen set, and namespace-absence check.


===== RAW REVIEW: refine-logs/C09_A0_V15_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v15 — Independent Design Review, Round 15

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V15_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 5 Important`

---

## Prior-round audit

I diffed v14 → v15 (7 raw hunks), re-opened every source on disk, and re-derived the arithmetic myself.

**Round 14's three findings — all three DISCHARGED, verified at source.**
- **R14 I-1.** `2×79/744 = 0.2123655914`; `1 − 0.2123655914 × 0.05 = 0.9893817 ⇒ 0.9894`. The retired `76` gives `0.9897849 ⇒ 0.9898`. §0 now reads `0.9894`; §6.2 already carried it and labelled `0.9898` as the retired-`76` value.
- **R14 I-2.** `:786-790` is the ZH degeneracy block; the MECHFIX paragraph begins at **`:791`** and ends at `:796`, carrying `T1 +0.0000, T2a +0.0000, T2b +0.0000, T4 +0.0067`, this arena's `−0.0006 / +0.0000 / −0.0040 / −0.0063`, and *"sign-unstable across the two head-space arenas"* — all quoted exactly. Corrected in all three places; the surviving `:786-790` strings are the description of the defect. The provenance claim checks out: `C09_A0_V13_PREREG_REVIEW.md:52` is where it originated.
- **R14 I-3.** v15 says **six edit sites** and enumerates six, including the trailer. Against the diff, six is correct.

**Round 14's byte-identity claim, re-verified independently.** I hashed `## 8. Gates` through the line before `## 10.` in v8 and in v15: **`md5 a17b56954ee6955013327f82a03904f7` on both.** §8 and §9 are byte-identical between v8 and v15.

**v15's scope claim.** *"§§1–11 are byte-identical to v14 except for that single §6.1 locator"* — **verified true**.

**Findings from R1–R13 that remain open in v15: none.**

---

## What I verified as sound

**The F88 transfer, re-derived from the primary tables.** `ERRPAT_HateMM §1`'s final rows `28 / 26 / 26` (Σ 80, mean 26.6667); §1.1's final row `187/1/2/25` ⇒ `(25, 2, 1)`, identity `= 80` ✓. `ERRPAT_MHC-ZH §2`, headed exactly *"ERROR INVENTORY (Tier 2, final-epoch protocol)"*, gives `23/24/22` (Σ 69) with `(22, 0, 3)`, identity `= 69` ✓. Arena per-seed errors re-derive from the **unrounded** banked accuracies (`744 × 0.8884408602150538 = 661.0`). Scaling gives `|P_0| ≈ 79 / 60`, `n_unstable ≈ 9 / 8`, `|P_{τ_hi}| ≈ 40 / 30`, closing exactly. Val-sel counterfactual `≈ 77` ✓.

**Every k-grid figure.** Caps `131.4667` / `95.6667`; `R(2|P_0|) = 158/120` dead; `R(1.5 × 79) = 119` (and `np.round(118.5) = 118`, so the half-up pin is load-bearing); `78.1 ⇒ 0.9261/0.6563`; `59.5 ⇒ 0.9545/0.6611`; `0.6889/0.7354` and `0.7139/0.7417`; all six `τ_hi` cells inside the caps at `π* = 0.965/0.810/0.7325` and `0.9833/0.8222/0.7417`. **All twelve reproduce.**

**The knife edge, checked beyond the document.** §5.2 analyses the `τ_hi` reach leg on accuracy only, while `K-REACH` is a conjunction with `ΔmF1 ≥ +0.050`. I reconstructed both confusion matrices from the banked floors and the ERRPAT FP/FN splits: HateMM `(251, 47, 36, 410) ⇒ mF1 0.8831` against banked `0.8838`; MHC-ZH `(148, 32, 30, 369) ⇒ 0.8747` exactly. Flipping `|P_{τ_hi}|` gives `ΔmF1 ≈ +0.0565` (vs `Δacc 0.0538`) and `≈ +0.0606` (vs `0.0518`). **The accuracy leg binds on both datasets**, so §5.2's acc-only knife edge is correct rather than incomplete.

**Statistical soundness and decidability.** `AUC_strat` fully pinned; both nulls exact for the nulls they name with the caveats carried; one shared `π` conservative in both places; Holm over `m = 2` correct; requiring both families is a pure additional hurdle; the dataset conjunction is a genuine IUT; `p`-floor `≪ α/2`. **Bands A′/A/B/C exhaustive over completed runs.** At `τ_0`, `k = 2|P_0|` dead on both and the other two live on both; at `τ_hi` all three live — **nothing is undecidable by construction, and a KILL is fully reachable.**

**Legality.** `GATE-BLIND` enforces `H-L1`–`H-L4` structurally; `GATE-LEDGER`'s expected `36` matches the unconditional `np.savez`; the `torch.load` guard and `headspace_fidelity.py`'s `VAL_RE` filter close every test path; CAL-2 leg (2) is correctly omitted. The three texts that run against C09 are carried rather than dropped.

**Executability and budget.** `sha256(headspace_mint.py) = cefdf8dc…` identical to `meta.mint_script_sha256` in all six banked arenas. Every cited line lands. `sacct -j 13847` confirms 8 CPU / 0 GPU / 32 G / `00:29:49`; the 36 mints re-derive to min `33.2`, max `60.0`, median `41.85`. Budget sums to ≈116 against ≈115.

**Every other number** re-measured this session and exact, including `GATE-NULL` on both named caches and the two data-defect counts.

---

## IMPORTANT

**I-1 — §12 asserts a measured quantity (`0.9575`) that exists in no artifact, transcribed from the round-14 review without re-opening the source.** There is no 3-seed DEG-A. `aggnet_pregate.py:534` computes it once — `(c3 == coll["THRESH_best"]).mean()` — from the `NET_SEED = 0` primary arm; the stability arms `C3_net_s1 / C3_net_s2` **never enter the `DEG` block**, and their per-item prediction vectors are not banked, so the figure is not recomputable. Exactly one `A_agree_threshold_shift` exists per `(dataset, space)`; HateMM/fused `= 0.957`. The string `0.9575` appears nowhere in the repository except round 14's review and this document. Two claims are unsupported: that `0.9570` is a *seed-0 cell* of a per-seed family, and that a 3-seed mean of `0.9575` exists. Nothing moves — `K-DEG` reads realised `pred_agree`, and the `0.95` line is separately sourced to `RESTRANS_PREGATE_RECORD.md:409` — but §0 states the repository's rule in this document's own voice (*"re-read the source, never transcribe a locator from a review"*), and v15 then transcribed a **number** from the same review into the same paragraph type. **Strike or re-source.**

**I-2 — the ledger claims a repair whose text does not exist in v15.** §0 and §12 both say v14's scope paragraph was *"corrected to eleven edit sites, in both copies"*. That paragraph does not survive into v15 — it was deleted and replaced by v15's own six-site paragraph. There is no copy to have corrected, and nothing in v15 states eleven. A reader auditing the ledger will search for the repaired text and not find it.

**I-3 — §5.5 presents a composite paraphrase as a verbatim quotation from the reopen.** §5.5 attributes, in quotation marks: *"so part of the reported ZH 0.8537 floor rests on how the corpus was harvested rather than on video content."* `GATE0_REOPEN_2026-07-31.md:1050-1051` reads: *"Part of the reported ZH floor rests on a marker of how the corpus was collected rather than on video content"*. Three deviations: `0.8537` is **spliced in from `:1047-1048`**; `a marker of` is dropped — the source's own hedge; `collected` becomes `harvested`. No locator is given, which is why it has survived since v3. Every surrounding number is exact and the distortion runs *against* C09, but it is a sentence in quotation marks the cited record does not contain. **Note:** this is the only one of the five inside §§1–11, so fixing it retires v15's *"§§1–11 are byte-identical to v14 except…"* claim.

**I-4 — "§8 (all ten gates)" contradicts §8.1's own count, in both new copies.** §8 names **twelve** `GATE-*` objects: nine in §8.1 and three in §8.2. §8.1's own closing sentence is careful about exactly this — *"Nine HALT gates, and exactly one further HALT condition elsewhere… These ten conditions are the complete publication precondition of §9"* — and §9 reads *"all nine HALT gates of §8.1 … and the `SHUFFLE-POP` band."* The underlying claim is **true** (verified by hash), so this is a gloss defect, but it is a statement about the document's own contents that its own §8.1 refutes, and it is new in v15.

**I-5 — the two copies of the scope paragraph are not verbatim identical, against the paragraph's own claim.** v14's two copies differed only by the intentional cross-reference; v15's differ by that **and** by the phrase `this section, `. Semantically inert, but R11 I-2 and R13 I-5 established divergence between these copies as a reviewable convention and R14 verified byte-identity programmatically as the discharge.

---

## Checked and deliberately not counted

Band B's quantifier not formally excluding Band C in one corner (§9 is two-valued and unambiguous); §5.2's acc-only knife edge (verified correct above); §5.6's *"30-epoch Adam"* against `AdamW`; §2's *"~25–60 s"* and the ≈115-vs-116 rounding; `ITEM-STRATUM` occupancy not emitted the way the row strata are (direction conservative — `K-FELDMAN` requires both families, so the conditional null can only remove CONTINUEs); the `(τ, k)`-cell wording; §3's rendering of `LITSWEEP3:91` (inherited from the registry, running against C09); `dead[Fxx]` not being a literally resolvable path (repository convention); `ksweep_OUT.json`'s `dev` sub-key being `null` on the four `MHC_EN_ARM-V/*/valsel` curves; minor typographic drift inside italic quotes.

---

## Bottom line

**The science is finished and I could not move it — for the seventh round running.** The instrument, the arena, the legality spine, the label-use discipline, the executability, the sha256 pins, the budget and the whole inferential apparatus hold under independent re-derivation, and §8 and §9 are byte-identical to v8 by hash. Round 14's three findings are genuinely discharged at source. **A KILL remains fully available and is still the honest expectation.**

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not yet.** All five findings are one-line text edits and none touches a rule, threshold, feature set, null, gate, operating point, verdict or scope. But two are new in v15, and I-1 is a number asserted as measured that exists in no artifact — not a defect a preregistration should carry across a hash freeze, and the exact failure mode §0's own repair note names as forbidden. Four sit in §0/§12; I-3 sits in §5.5 and therefore requires v16's scope statement to be restated rather than carried.

With those five edits made, **v16 is ready to freeze and submit as one 8-CPU / 32-GB / no-GPU / no-`--time` job of ≈116 CPU-minutes**, subject to the three preconditions the STATUS block names plus the immediately-prior `squeue` empty-check, sha256 re-verification and namespace-absence check.


===== RAW REVIEW: refine-logs/C09_A0_V16_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v16 — Independent Design Review, Round 16

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V16_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 3 Important`

---

## Prior-round audit

I diffed v15 → v16 (7 raw hunks), hashed the document section by section, re-opened every source on disk, and re-derived the arithmetic myself.

**Round 15's five findings — four DISCHARGED at source, one repaired-but-misdescribed.**
- **R15 I-1 (`0.9575`) — DISCHARGED.** `aggnet_pregate.py:534` is exactly `"A_agree_threshold_shift": round(float((c3 == coll["THRESH_best"]).mean()), 4),` with `c3 = coll["C3_net"]` (`:533`), the `NET_SEED = 0` arm; `C3_net_s1/s2` are built at `:391` and never enter the `DEG` block, and no per-arm agreement against `THRESH_best` is emitted. **No per-seed or 3-seed DEG-A exists.** `0.9575` now appears only inside the finding's own description. The `0.95` line is correctly sourced.
- **R15 I-2 — DISCHARGED.** `eleven` appears twice in v15, both as the *claim*; v16 restates it accurately.
- **R15 I-3 — DISCHARGED.** `GATE0_REOPEN_2026-07-31.md:1050-1051` reads exactly as v16 quotes it; `0.8537` sits at `:1047-1048` inside a nested quote of `LITSWEEP3_ZH_SPECIFIC.md:36-37`. All three deviations named. `corpus was harvested` first entered at v3, so "carried since v3" is right.
- **R15 I-5 — DISCHARGED, verified by hash.** With the cross-reference normalised, `md5` of the two scope paragraphs is identical (`50fc261144c5f5b2d15af82e63e28836`).
- **R15 I-4 — the *defect* is gone; the *ledger's description of the repair* is false.** See I-1.

**Byte-identity of §8 and §9, re-verified independently.** `## 8. Gates` through the line before `## 10.`: v8 `a17b56954ee6955013327f82a03904f7`; v15 same; v16 same. **§8 and §9 are byte-identical between v8 and v16.**

**v16's scope claim, verified against the diff.** Section-level hashing: §§1, 2, 3, 4, 6, 7, 8, 9, 10, 11 **SAME**; §0, §5, §12 and head/trailer **DIFF**; §5's only hunk is the single §5.5 site. The six enumerated edit sites match exactly.

**Findings from R1–R14 that remain open in v16: none.**

---

## What I verified as sound

**The F88 transfer, re-derived from the primary tables, not from any review.** `ERRPAT_HateMM §1`'s final rows `28 / 26 / 26` (Σ 80, mean 26.6667); `§1.1`'s final row `187/1/2/25` ⇒ `(25, 2, 1)`, identity `= 80` ✓. `ERRPAT_MHC-ZH §2` gives `23/24/22` (Σ 69) and `(22, 0, 3)`, identity `= 69` ✓. The banked accuracies give integer error counts exactly (`744 × 0.8884408602150538 = 661.0`; `579 × 0.8929188255613126 = 517.0`). Scaling gives `|P_0| ≈ 79 / 60`, `n_unstable ≈ 9 / 8`; closure holds on both. Val-sel counterfactual `≈ 77` ✓.

**All twelve `k`-grid figures, re-derived**, together with the knife-edge arithmetic (`40/744 = 0.0538`, `37/744 = 0.0497`, `38/744 = 0.0511`, `30/579 = 0.0518`, `29/579 = 0.05009`, `28/579 = 0.0484`, ZH bar `28.95 ⇒ 29`), Hanley–McNeil, and the π* gaps.

**The knife edge, checked past the document.** I reconstructed the **exact** confusion matrices from `posrate_deployed` / `posrate_bank`: HateMM `(256, 42, 41, 405) ⇒ mF1 0.88378 = banked 0.8838`; MHC-ZH `(148, 32, 30, 369) ⇒ 0.87466 = banked 0.8747`. Flipping `|P_{τ_hi}|` gives `ΔmF1 ≈ +0.056` vs `Δacc 0.0538` and `≈ +0.061` vs `0.0518`. **The accuracy leg binds at `τ_hi` on both datasets** — §5.2's accuracy-only analysis is complete rather than partial.

**Statistical soundness and decidability.** `AUC_strat` fully pinned; both nulls exact for the nulls they name with honest caveats; one shared `π` per `(dataset, τ, draw)` over `P^{(τ)}`, item-level everywhere; Holm over `m = 2` correct; both families required is a pure additional hurdle; the dataset conjunction is a genuine IUT; `p`-floor `≪ α/2`; `IDENTIFIABILITY_UNDERPOWERED` fails `K-FELDMAN`, the conservative direction. §9 is two-valued and complete; **a KILL is fully reachable** and nothing is undecidable by construction.

**Legality / ban-scope.** `H-L1`–`H-L4` enforced structurally by `GATE-BLIND` and `GATE-LEDGER` (expected `36` matches the unconditional `np.savez` at `headspace_mint.py:322-324`); the `torch.load` guard and `VAL_RE` close every test path; CAL-2 leg (2) correctly omitted. Every registry ban quoted verbatim and adjudicated; **F113's `dead[]` entry has no `ban_scope` key** and closes *"STANDING RULE PROPOSED (not yet ruled)"*. The three texts that run **against** C09 are carried.

**Executability and budget.** `sha256(headspace_mint.py)` identical to `meta.mint_script_sha256` in all six banked arenas. `headspace_mint.py:288` forces `--device cpu`; `det1_assert` fires at `:187`; `CLI` admits exactly `{hatemm, zh}`, so the MHC-EN scope argument is structural. `sacct -j 13847` confirms 8 CPU / 32 G / no GPU / `00:29:49`. I recomputed the 36 banked mint durations from their own `meta.secs`: **min 33.2, max 60.0, median 41.85**, total 24.5 CPU-min against a budgeted ≤ 36. Every gate anchor re-read exactly; both data defects re-measured.

**Scope honesty.** **No C09 namespace exists anywhere on disk** — the trailer's claim is literally true. §10's withdrawals are all consistent with §§3.2, 5.1, 6.3 and 11.

---

## IMPORTANT

**I-1 — §12's ledger claims a repaired text that appears nowhere in v16, one row after repairing exactly that defect.** Row **I-4**'s repair column reads *"corrected in **both** copies to `"its nine HALT gates and its three reporting instruments"`."* That string occurs **zero** times in v16. What happened is that both sentences carrying `§8 (all ten gates)` were **deleted and rewritten** to describe the hashed span. The rewrite is a *better* repair than the one claimed and the defect is genuinely gone — but a reader auditing the ledger will search for the quoted corrected text and not find it, which is verbatim the failure round 15 raised as I-2 and which row I-2 immediately above now describes as a lesson.

**I-2 — §12's quotation of round 15's bottom line is not verbatim; the trailing clause is spliced in from round 14's review.** §12 attributes to round 15 *"…verdict or **the scope of any conclusion**."*; `C09_A0_V15_PREREG_REVIEW.md` reads *"…verdict or **scope**."* The phrase `scope of any conclusion` appears **0 times** in round 15's review and **1 time** in round 14's. v15's §12 quoted round 14 accurately; v16 swapped the counts and the round number and carried round 14's trailing words into round 15's mouth. The ellipsis joining round 15's two separated sentences is legitimate; the five-word expansion is not. New in v16, and the same species as R15 I-3.

**I-3 — the STATUS block points a reader at "the v13 ledger" for a §12 that is now v16's ledger.** *"Each carries its own ledger; **the v13 ledger is §12**."* §12's own heading is *"the 5 round-15 findings"*. The sentence names **this** document's ledger and froze at `v13` when v14 forked; it has been false in v14, v15 and v16, and no round has caught it.

---

## Checked and deliberately not counted

Band B's quantifier (§9 is two-valued and complete — I re-derived exhaustiveness myself); §5.6's *"30-epoch Adam"* against `src/run_rac.py:684`'s `torch.optim.AdamW`; §2's *"~25–60 s"* against a measured minimum of `33.2`; the budget summing to ≈116 against ≈115; `ITEM-STRATUM` occupancy not emitted the way the row strata are (conservative); the `(τ, k)`-cell wording; §3's rendering of `LITSWEEP3:91` (inherited verbatim, running against C09); `dead[Fxx]` not being a literally resolvable path; `ksweep_OUT.json`'s `dev` sub-key being `null` on exactly the four `MHC_EN_ARM-V/*/valsel` curves; §6.1's *"`:644-647` … headed"* where the heading is at `:625`; §12's report of round 15's approximate `mF1 0.8831` (my exact reconstruction lands on `0.8838` and confirms the conclusion); typographic drift inside italic quotes.

---

## Bottom line

**The science is finished and I could not move it — for the eighth round running.** The instrument, the arena, the fold contract, the legality spine, the label-use adjudications, the two nulls, the Holm/IUT structure, the power rule, the closed-form caps, the executability, the three sha256 pins and the budget all hold under independent re-derivation, and **§8 and §9 are byte-identical to v8 by hash on my own recomputation**. Round 15's five findings are discharged in substance — including the one that mattered most, the non-existent `0.9575`, which I verified against `aggnet_pregate.py` line by line. **A KILL remains fully available and is still the honest expectation.**

**Does any remaining finding change a verdict, a rule, a threshold, a gate, an operating point, or the scope of any conclusion? No — none of the three does.** All three are statements the document makes about its own repair history: one claims a corrected string that does not exist, one misquotes the review it is answering, one points at the wrong ledger version. §§1–11's governing text is untouched by all three, and for the fourth round running the blockers sit entirely outside it.

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not quite.** Two of the three are *new in v16*, and both reproduce — inside the very ledger that names them as lessons — the two defect species round 15 raised. A preregistration whose audit trail is its ledger should not carry a false description of its own repairs across a hash freeze. They are three one-line text edits, all in §0/§12.

With those made, **v17 is ready to freeze and submit as one 8-CPU / 32-GB / no-GPU / no-`--time` job of ≈116 CPU-minutes** — subject to the three preconditions the STATUS block names, the immediately-prior `squeue` empty-check, sha256 re-verification and namespace-absence check (all four namespaces confirmed absent today), and the separate step this document is honest about but which is not yet done: **the analysis script and the sbatch driver do not exist yet and will need their own implementation and code review before anything is submitted.**


===== RAW REVIEW: refine-logs/C09_A0_V17_PREREG_REVIEW.md =====

# C09 Stage-0 (A0) v17 — Independent Design Review, Round 17

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V17_RECORD.md`
**Verdict.** `GO` — **0 Critical / 0 High / 0 Important**

---

## Prior-round audit

I diffed v16 → v17 (6 hunks), section-hashed both, re-opened every source on disk and re-derived the arithmetic myself.

**Round 16's three Importants — all three DISCHARGED at source.**
- **R16 I-1.** The two sentences carrying `§8 (all ten gates)` sat at `C09_A0_V15_RECORD.md:73` and `:1879`; in v16 both were **deleted and rewritten** to describe the hashed span — exactly what round 16 said, and a better repair than v16's ledger claimed. v17's row I-1 now describes that. `all ten gates` occurs **once** in v17, inside row I-1's quotation of the defective string, and nowhere in governing text.
- **R16 I-2.** `C09_A0_V15_PREREG_REVIEW.md:68` ends *"…verdict or scope."*; `scope of any conclusion` occurs **0 times** in round 15's review and **1 time** in round 14's. v17 quotes round 15 exactly and names the splice.
- **R16 I-3.** Now reads *"the ledger of THIS version is always §12"*, with the freeze history named. Version-independent; cannot restale.

**v17's scope claim, verified against the actual diff.** The diff touches exactly: title; STATUS tag; STATUS reading-order + ledger pointer; §0 (including the new *"What is NOT yet done"* paragraph); §12; trailer. **Nothing between v17:114 and v17:1887 changes — §§1–11 are byte-identical to v16.** Five edit sites, exactly as declared.

**The "character-identical apart from the cross-reference" claim, verified mechanically.** Word-diffing both copies: the *only* difference is `copied into §12` vs `copied into §0`.

**§8 + §9 byte-identity, re-verified by my own hash.** `## 8. Gates` through the line before `## 10.`: v8 `a17b56954ee6955013327f82a03904f7`; v15 same; v16 same; **v17 same**. No gate, no publication precondition and no decision rule has moved in nine versions.

**Findings from R1–R16 still open in v17: none.**

---

## What I verified as sound

**Instrument and executability.** All six analysis modules, `headspace_drive.sh`, `c02_a0_cpu_v9.sbatch`, `ksweep_OUT.json`, the six arena outputs, both fidelity outputs and `C02_A0_OUT.json` are present. `sha256(headspace_mint.py) = cefdf8dc…6612`, **identical to `meta.mint_script_sha256` in all six banked arenas**; the mint's own `FROZEN_PAIRVERIFY_SHA = 77b0defd…240d` matches `mechnov_pairverify.py` on disk today, so the mint runs unmodified. `CLI` admits exactly `{hatemm, zh}` — the MHC-EN scope argument is structural. `:288` forces `--device cpu`; `det1_assert` fires at `:187`; `:192-194` gives the resume path; `:203-216` asserts the fold partition against the banked `vsw_ckpt` and refuses on mismatch; `:274-281` no-ops `torch.save`; `:285-286` matches the layout §2 describes. `headspace_fidelity.py:66` hard-codes the `ffull` name; its six FLOOR trainlogs exist; `VAL_RE` drops every non-`Val_Retrieval` line. `mechfix_ops.py:94/95` literal; `Σw = 210`.

**Gate anchors, re-read.** All twelve `GATE-FLOOR` values reproduce exactly, and C02's independent re-mint reproduces them to full precision. Fold counts, `raw_deployed_acc`, `GATE-DEVFID`'s references, `GATE-ARENA`'s bands, `GATE-NULL` (re-measured on both operative caches) and `GATE-FIXK20`'s premise all check out.

**The F88 transfer, re-derived from the ERRPAT tables.** `28/26/26` (Σ 80) with `(25, 2, 1)`; `23/24/22` (Σ 69) with `(22, 0, 3)`; both identities close. Scaling gives `|P_0| ≈ 79 / 60`, `n_unstable ≈ 9 / 8`; closure exact. Val-sel counterfactual `≈ 77`.

**The knife edge, checked past the document.** I reconstructed both confusion matrices exactly from `posrate_deployed`/`posrate_bank`: HateMM `256/41/42/405 ⇒ mF1 0.883777 = 0.8838`; MHC-ZH `148/30/32/369 ⇒ 0.874659 = 0.8747`. **The accuracy leg binds at `τ_hi` on both datasets**, so §5.2's accuracy-only knife edge is complete. All twelve `k`-grid figures, the caps, the required recalls/precisions, the `π*` values, the reach knife edges, the Hanley–McNeil SEs and the `K-DEG` algebra all re-derive.

**Statistical soundness and decidability.** `AUC_strat` fully pinned. `PERM-STRUCT` is an exact marginal permutation null with the item as the unit and the conservative choices in every case. `PERM-STRUCT-COND` is exact for `struct ⊥ (target, BASE) | ITEM-STRATUM`, with the residual coupling declared. Holm over `m = 2` correct, both families required; the dataset conjunction is a genuine IUT; the `p`-floor `≪ α/2`. `IDENTIFIABILITY_UNDERPOWERED` *fails* `K-FELDMAN`. I independently verified §4.3's monotonicity claim (`(A+2)/(A+B+1) > A/(A+B)`), so `K-REACH` at `τ_0` closes every `τ ≥ 0` by arithmetic, and it is correctly declared false for `NET` and `ΔAUC`. `GATE-SELFTEST`'s identity is exact. §9 is two-valued and exhaustive on completed runs; **a KILL is fully reachable at both `K-FELDMAN` and `K-NET`.**

**Legality / ban-scope.** Zero GPU. Train + `dev_seen` only; the `torch.load` guard and `VAL_RE` close every test path; the expected dev-label count of `36` matches the unconditional `np.savez`. Every ban quoted **verbatim** and adjudicated at source. **F113's `dead[]` entry has keys `['name','status']` only.** CAL-0…CAL-5 check out. The three texts that run **against** C09 are carried at their adjudicated weight; F114 is correctly distinguished from its C04-lineage homonym.

**Budget.** `sacct -j 13847` confirms 8 CPU / 32 G / **no GRES** / `00:29:49`. The 36 banked mint durations recompute to min `33.2`, max `60.0`, median `41.85`, total `24.5` CPU-min against a budgeted `≤ 36`. The permutation lines are generous by a factor of ~2–5.

**Scope honesty.** `find` returns **no C09 namespace of any kind**; the analysis script and sbatch driver genuinely do not exist, exactly as §0 says. §10's ten scope bullets are each consistent with the section they cite, and every withdrawal is honoured in the governing text.

---

## Bottom line

**The science is finished and I could not move it — for the ninth round running.** The instrument, the arena and its fold contract, the legality spine, the four label-use adjudications, the two nulls and their honest caveats, the Holm/IUT structure, the power rule, the closed-form caps, the degeneracy control, the executability, the three sha256 pins and the budget all hold under independent re-derivation, and **§8 and §9 are byte-identical to v8 on my own hash**. A KILL is fully available at `K-FELDMAN` and at `K-NET`, and remains the honest expectation given the registered `+0.0269 / +0.0104` prior.

**I raise no finding.** Three things I deliberately did not count, so the record shows they were seen: (i) row I-1's *"a string that appears nowhere in v16"* is self-referentially loose — the intended and only sensible reading ("nowhere as the corrected text") is true, and round 16's own *"zero times"* was the same looseness; (ii) row I-2's *"round 15 ends"* refers to how that sentence ends; (iii) everything on v17's own below-the-line list, none of it load-bearing on any rule.

**The DESIGN is ready to hash-freeze.** What remains is not design work: implement the analysis script and the sbatch driver, hash-freeze the frozen set, and pass a **separate** independent code/resource review. Then the STATUS block's three preconditions plus the immediately-prior `squeue` empty-check, sha256 re-verification and namespace-absence check. For the record, `squeue -u jehc223` is empty as of this review and no C09 namespace exists on disk; neither fact authorizes anything.

---

# C09 A0 CLOUD PORT — PORT-DELTA REVIEW, ROUND 1 (2026-08-02)

Fresh independent reviewer (opus), no exposure to the port reasoning. Scope: the PORT
DELTA only (substrate fidelity, image pin, gate portability classification, tie-break
rule, data boundary). The science was already GO 0C/0H/0I at design round 17 and GO
0C/0H/0I at code round 7 and is unchanged by the port; it was explicitly out of scope.

## VERDICT: REVISE — 2C / 2H / 8I

"The port is a careful, largely faithful transliteration, and its gate-portability
classification is substantively correct. But the driver as written cannot start a
container -- and this is not a theoretical claim: a preflight was already launched and
crash-looped 10+ times. There is also a second, independent container-start blocker that
the crash-loop masked. Neither can produce a wrong verdict, but the artifact under review
is currently non-functional and one frozen submission-discipline precondition has been
dropped in the move."

C-1  image env() sets PYTHONPATH, which breaks Modal's own container bootstrap. MEASURED,
     not predicted: every container died at /pkg/modal/exception.py:42 with
     "ModuleNotFoundError: No module named 'grpclib'" and crash-looped. Cause: the image
     ENV replaces the PYTHONPATH Modal's runtime uses to reach its vendored client deps,
     and forces site to import the C09 sitecustomize into Modal's own runner process,
     monkeypatching builtins.open before Modal is loaded. Fix: delete ONLY the PYTHONPATH
     key from .env(); keep the four thread variables and CUDA_VISIBLE_DEVICES there
     (DET-1 requires them before an interpreter starts). _env() already sets PYTHONPATH
     for every child; prefer the sbatch's ${PYTHONPATH:+:$PYTHONPATH} prepend semantics.

C-2  module-level `from modal_probe_runner import guard_reason` ModuleNotFoundErrors
     inside the container; independent of C-1 and not yet reached. Modal 1.5.2 executes
     importlib.import_module("c09_a0_modal") inside the container, so ALL module-level
     code runs container-side. Automounting is fully removed in Modal 1.x -- only
     /root/c09_a0_modal.py is mounted, and scripts/cloud/ has no __init__.py so the
     PACKAGE branch does not apply. Same defect makes build_manifest() re-hash 46 MB and
     build_stage() re-copy the closure on every container start. Fix: wrap all three in
     `if modal.is_local():`; the container already receives the manifest as an argument.

H-1  the frozen "exactly ONE submission / never resubmit" precondition (freeze record
     preconditions 6 and 7) is not carried into the port, and the tie-break presumes a
     single cloud event. `modal run` has no approval gate, RUN_ID is hard-coded, and the
     resume-skip makes a re-run cheap. Because Modal offers no CPU-SKU control, repeated
     attempts are independent host draws -- a literal retry-until-GATE-FLOOR-passes loop,
     selecting the host post hoc on a criterion the prereg does not bound. Does NOT reach
     a biased verdict (GATE-FLOOR is verdict-blind) but a dropped frozen precondition in
     a hash-freeze ceremony is not a nit. Fix: pre-register one invocation, a spent-on-
     HALT rule, and an explicit infrastructure-failure boundary.

H-2  the image is not actually pinned and the record claims a pin it does not have.
     tqdm unpinned (the build resolved tqdm-4.70.0 vs local 4.67.3); debian_slim pins no
     base digest and no Python patch level (local 3.11.8); transitive deps unpinned
     (joblib 1.5.3, packaging 26.2, networkx, fsspec...). CLAUDE.md's 2026-07-31 ruling
     requires the image be nailed down IN THE PREREGISTRATION, yet the record reads
     "image digest: TBD" while asserting in the present tense that the digest covers the
     input closure byte-for-byte. VERIFIED SOUND: every version-pinned member matches the
     local env exactly (torch 2.6.0 -> same PyPI CUDA wheel, so torch.__version__ reports
     2.6.0+cu124 both sides; numpy 1.26.4, scipy 1.17.1, sklearn 1.5.2, faiss-cpu 1.13.2,
     pandas 2.3.3, pillow 11.1.0, torchmetrics 1.9.0, wandb 0.28.0, threadpoolctl 3.6.0,
     easydict 1.13, rank-bm25 0.2.2). The local env is entirely pip (pypi_0), so the
     pip-vs-conda BLAS-provenance trap does not apply. Thread pinning at 8 preserved.

I-1  c09guard's current-vs-stale ledger partition is keyed on SLURM_JOB_ID and silently
     collapses in a container (both writer and aggregator see "nojob"), so a previous
     attempt's processes count toward GATE-LEDGER's n_processes_reporting>=1 conjunct --
     the exact conjunct the design added so that "a ledger that reports zero because NO
     process ever reported is not evidence of a clean run". Not classified in the port
     record's audit.
I-2  the 36 mints and the ledger files never leave the container; the winning run's
     instrument would be unauditable. The local sbatch leaves both on disk indefinitely.
I-3  the upload allowlist widens by four extensions while the code declares three
     (.sbatch undeclared). Harmless in substance, but an undeclared relaxation inside a
     media guard is precisely what this ceremony exists to prevent.
I-4  the port record and TARGET_STATE are stale on two facts: status still reads NOT YET
     SUBMITTED though a preflight was submitted and crash-looped, and job_13885 is
     recorded as PENDING/JobHeldUser though squeue now shows 13885 RUNNING on
     foscsmlprd01. Given the ~45-115 min budget the local job is very likely to take the
     tie-break before any cloud fix lands, which the caller should weigh.
I-5  PORT-CHECK-5 is described as an in-container hash verification but compares the
     locally-computed manifest against FROZEN_SHA256; it never hashes a container file.
     The property holds transitively via _verify_upload, but the wording overstates it.
I-6  the numerics preflight constrains nothing about the host the real run receives
     (Modal exposes no CPU-SKU selector; cpu=8 is a core reservation), and its retry count
     is unbounded. Fine as cost-avoidance; weaker than it reads as a "mitigation".
I-7  "the port cannot produce a wrong verdict; it can only produce a HALT" is very nearly,
     but not exactly, true. GATE-FLOOR/GATE-PARITY-FOLD compare AGGREGATES at 4dp, not
     item-level identity: a compensating pair of flips inside one fold preserves that
     fold's accuracy exactly. The mF1 conjunct kills most such cases (hence Important, not
     High), but the residual is not zero. Fix: soften the claim (adding an item-level pred
     digest would be a science change and is out of scope for the port).
I-8  the tie-break's ordering procedure compares a duration against a log: "sacct elapsed"
     is not a wall-clock instant and cannot be ordered against a container-log timestamp.
     Fix: absolute clock times on both sides.

## WHAT THE REVIEWER CHECKED HARDEST AND FOUND SOUND

- Media guard unweakened. guard_reason (modal_probe_runner.py:100-112) evaluates the
  media-extension blocklist FIRST, then the forbidden-media-dir check, then the allowlist.
  The port's tolerance clause matches only reason.startswith("extension ") AND requires a
  declared extra suffix, so it can tolerate only the third clause. NO MEDIA FILE CAN PASS.
  Upstream assert_uploadable is untouched; the added C09-TEST-GUARD is strictly stronger.
- GATE-LEDGER's pass is not contaminated by the coverage emission. c09_a0_arena.py:1958:
  pass = bool(tot["test_path_opens"] == 0 and len(procs) >= 1). The coverage block and
  n_processes_expected_fresh_run:39 are read by no predicate. Classification CORRECT.
- The guard has not become tautological in a container with no test files:
  c09guard.is_test_like is purely path-based and never stats the file, so an attempted
  test open still raises and still increments test_path_opens.
- The REPO-path invariant is structurally guaranteed AND runtime-checked, not asserted in
  prose: add_local_dir targets the absolute literal, and PORT-CHECK-2 checks
  c09guard.REPO == REPO, root existence, arena presence, plus a POSITIVE probe
  (.../test_seen_x.pt -> True) and a NEGATIVE probe (operative train cache -> False),
  failing closed. "This is the port's best piece of work."
- Step-for-step fidelity: order, loop nesting (hatemm,zh x 0,1,2 x 0..4,-1), the full/f{n}
  tag, output filenames, --threads omission on the mints and --threads 8 on the arena,
  allow_fail=True on both GATE-DEVFID runs matching `if ! ...; then`, the resume-skip, and
  the fail-closed rc!=0 semantics of set -euo pipefail all match. The expected reporting-
  process count of 39 is preserved exactly (1 startup child + 36 mints + 2 devfid; the
  driver writes no ledger because C09_LEDGER_DIR is set on the child env only -- the same
  asymmetry the sbatch's shell has). cwd=REPO + absolute script paths give the same
  sys.path[0] as the sbatch's relative invocation.
- DET-1: OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8 as image ENV, i.e. before any interpreter
  starts, and _env() re-asserts them for every child. det1_assert("8") fires in the mints,
  in headspace_fidelity and in the arena.
- ONE container: all 36 mints, both GATE-DEVFID runs and the arena in a single
  @app.function invocation. The preflight is a separate container but computes no decision
  quantity, writes to a separate namespace, and discards its mints.
- Input closure complete AND minimal, verified against the code not the prose. The mint's
  run_rac chain needs only pinned packages (no transformers/decord/av/peft/matplotlib on
  the import path). --path defaults to ./data/ but is never dereferenced because
  load_feats_from_CLIP is monkeypatched at headspace_mint.py:241; the archive and TARC
  branches that read data/gt/** are off under the frozen CLI. headspace_fidelity needs
  exactly the 6 trainlogs. The arena's complete read set is the 6 banked arena JSONs, 2
  train caches, 2 data/gt/*/train.jsonl, the config and the 36 mint .npz -- all present.
  src/ contributes 43 .py files and nothing else. Excluding src/logging and src/moka is
  safe under PEP 420 namespace-portion resolution. NOTHING UPLOADED THAT SHOULD NOT BE:
  no test_seen, no _shards, no data/gt/*/test.jsonl, no media, 46.1 MB total.

## DISPOSITION

All 12 findings addressed; see refine-logs/C09_A0_CLOUD_PORT_REVIEW.md and
refine-logs/C09_A0_CLOUD_PORT_RECORD.md. C-1 and C-2 fixed and verified (container-side
import now succeeds with MANIFEST {} / STAGED [] and both functions present). I-4 acted
on: job 13885 went RUNNING at 2026-08-02T08:14:15 and takes the pre-registered tie-break,
so the cloud A0 was NOT launched.

---

# C09 A0 CLOUD PORT — PORT-DELTA REVIEW, ROUND 2 (2026-08-02)

Fresh independent reviewer (opus), scoped to the PORT DELTA only. Formed an independent
view FIRST (substrate diff, gate-by-gate trace, closure verification, live check of job
13885), then read round 1 LAST and re-derived each round-1 fix rather than taking the
record's word.

## VERDICT: REVISE — 1C / 2H / 9I

"Round 1's twelve findings are genuinely closed -- I re-derived each fix rather than
taking the record's word. But I found one new Critical that neither round has addressed,
and two Highs where a round-1 repair was written as prose without a mechanism."

C-1 (NEW, DECISIVE) Modal clamps this account's function timeout to ~3600 s. The port
    budgets ~115 min in ONE container, and a cap-kill lands on the "invocation is SPENT"
    side of the tie-break's boundary with ZERO artifacts.
    Refuted by this repo's own measurements: W2A_PROBE_RECORD.md:34-38 -- "Modal clamps
    the effective function timeout to ~3600 s server-side (VERIFIED: MODAL_PROBE_TIMEOUT
    =43200 reaches the child and computes 43200 -- no stray 3600 in code -- yet every
    single-container attempt was killed at ~3600 s function-time)." Two apps died at
    ~62 min. W2A_CHUNK_LOG.md:3-6 attributes it to the Starter-plan cap; CLAUDE.md
    confirms this account is on Modal Starter. So timeout=28800 being ACCEPTED by the
    decorator proves nothing -- W2-A verified exactly that failure mode.
    MEASURED RUNTIME from the live local job: 13885 started 08:14:15, all 36 mints and
    both C09_FIDELITY_*.json written by 08:42 -> ~28 min of mints on AMD EPYC 7742 at 8
    threads, with a 20-30 min arena still to go. Total ~50-58 min ON THE FASTER SUBSTRATE.
    A Modal 8-vCPU container 1.3x slower puts the run at ~70-75 min, i.e. OVER the
    measured 62-min wall.
    Why Critical not operational: a kill at ~62 min lands after the first mint, so the
    single pre-registered cloud invocation is SPENT -- no verdict, no relaunch. And
    because the only outvol.commit() sat on the success path, the kill leaves NOTHING:
    no mints, no ledger, no manifest, no proof the invocation was consumed. The
    workaround this repo already built for exactly this cap (modal_probe_runner._execute's
    soft-budget chunk loop) restarts in a NEW CONTAINER, which violates the port's own
    invariant 2 (one container, one host) -- so it is unavailable here.
    Fix: record the cap in section 5; verify effective container lifetime >= 2x the
    budgeted wall before committing the one shot; if ~3600 s stands, the port CANNOT
    execute A0 under invariant 2 and must be abandoned or re-preregistered on a limit
    that admits ~2 h (chunked resume is a substrate change that breaks
    same-table-same-hardware and would need its own review); independently, commit the
    volume after the mints and after DEVFID so any kill leaves an auditable trace.

H-1 "Exactly ONE cloud invocation" is prose only. The code has no mechanism, and a second
    invocation silently ERASES THE EVIDENCE OF THE FIRST. (a) dest = "/c09out/{RUN_ID}"
    is a constant and is overwritten -- invocation #2 overwrites #1's decision, manifest
    (incl. ledger_job_token), mints and ledger; the one artifact that would prove the rule
    was broken is destroyed by breaking it. job_token is already unique per invocation --
    key dest on it. (b) No spent-invocation sentinel: SLURM has sacct as a permanent
    third-party record plus an approval gate as a second party; `modal run` has neither.
    Write /c09out/INVOCATION_*.json before the first mint and refuse if one exists.
    (c) `retries` is not pinned; a platform default change or operator-set retry would
    silently re-run the science on a FRESH INDEPENDENT HOST DRAW -- the exact thing the
    preregistration forbids. retries=0 costs nothing.
    Note the residual is asymmetric in the right direction on the gaming path -- a
    GATE-FLOOR HALT completes and commits, so it leaves an artifact. The traceless path is
    the crash/kill path, which is also the path C-1 makes likely.

H-2 The section-7 authorisation gate is real as a preregistration but the port collects
    NONE of the three things it demands, and one cannot be obtained from anything the port
    runs. _hostinfo() records only cpu_model/n_cpu_online/uname; C09_PORT_MANIFEST.json
    carries no pip freeze, no sys.version, no image id; preflight's runtime_block() gives
    the Python patch version and threadpool_info() but pip freeze is collected nowhere and
    the image id only from CLI build output. And it is not sufficient even when filled: a
    post-hoc pip freeze is a DESCRIPTION, not a pin -- debian_slim carries no base digest,
    so a cache-evicted rebuild can produce a different image and nothing would detect it,
    since PORT-CHECK-3 and PORT-CHECK-5 hash the input closure, never the environment. The
    gate is also unmechanised: run_a0() calls a0.remote() unconditionally.

I-1 the record mis-describes where PORT-CHECK-1 runs and in what order the checks fire.
    Section 6 says the guard assert happens "in the driver process before the mints"; it
    does not and must not -- the code says it runs in a CHILD, and the driver is
    deliberately unguarded, which IS the C-1 repair. Section 6 as written describes the
    state that crash-looped. Separately, section 2 lists "startup guard/zero-GPU assert,
    then the four sha256 checks", matching the sbatch; the code inverts it. Both are before
    the mints and both fail closed, so nothing is weakened -- but a fidelity record should
    not assert an order the code does not have.
I-2 round 1's I-7 residual is still mis-enumerated. GATE-FLOOR/GATE-PARITY-FOLD constrain
    only VOTE-DERIVED DISCRETE AGGREGATES. Every CONTINUOUS decision quantity the arena
    computes from the same keys is unconstrained by either gate at any tolerance:
    D-FELDMAN's dAUC, the PERM-STRUCT / PERM-STRUCT-COND p-values, the fit-conditional
    item-bootstrap lower bound (c09_a0_arena.py:1577-1581), the per-fold tau_hi medians
    (:668-678), and the dens50 / class_gap design columns. A key perturbation too small to
    move any vote but large enough to move a statistic across a bar is NOT a coincidence --
    it is the generic consequence of a different BLAS reduction order. What actually bounds
    it is the magnitude argument the record omits: a head that reproduces 6 pooled-acc +
    6 pooled-mF1 + 30 fold-acc cells at 4 dp is numerically near-identical, so the residual
    perturbation is ~1e-12-scale. State that.
I-3 outvol.commit() runs only on the success path, with no try/finally. Holds for
    CONTINUE/KILL/HALT (a HALT exits 0), but any crash, OOM, preemption or timeout-kill
    loses all 36 mints and the whole ledger -- precisely the state the tie-break calls
    spent. Wrap in try/finally and commit incrementally after the mints and after DEVFID.
I-4 no dependency/import probe before the first mint, so an import failure lands on the
    "spent" side of the boundary. The startup heredoc imports only torch/numpy/faiss/
    sklearn/scipy; the mints additionally need wandb, pandas, PIL, easydict, rank_bm25,
    tqdm, torchmetrics, threadpoolctl plus run_rac, mechfix_ops, mechnov_pairverify,
    headspace_mint. The pin list IS complete (transformers genuinely not needed; it appears
    only in run_linear.py and utils/generate_*, none imported) -- but the port PROVES none
    of it before burning the invocation. A missing threadpoolctl would surface only at
    runtime_block(), after ~52 s of training in mint #1.
I-5 PYTHONOPTIMIZE is neither neutralised nor classified. Every guard in headspace_mint.py
    is an assert: DET-1 (:77), fold parity vs banked vsw_ckpt (:215), frozen-pairverify sha
    (:188), torch.load test guard (:111). Under -O they all vanish. c09_a0_arena.py:47
    refuses to run under -O, so the job fails closed AT THE ARENA -- but only after 36
    unprotected mints, and that refusal appears in NO row of the portability tables.
I-6 module-level side effects: importing the driver locally rmtree's and rebuilds a 46 MB
    tree, and STAGE defaults to a hard-coded session-specific scratchpad path.
I-7 the preflight retry budget is unmechanised and its evidence is overwritten (fixed
    output path, no counter, ambiguous per-cell). Auditability, not bias -- the run draws
    an independent host regardless.
I-8 the container environment is a superset of the sbatch's, carried into every child, and
    never captured. WANDB_DISABLED is set in the image and NOT by the sbatch -- harmless
    (both disable wandb) but it contradicts _env's "verbatim" docstring.
I-9 section 5's same-table-same-hardware compliance claim omits section 4.3's own
    disclosure that Modal exposes NO CPU-SKU selector, and instead claims compliance on a
    different ground (one container, one host). That substituted property is the operative
    protection and does hold, but it is not what the ruling literally asks for.

## WHAT ROUND 2 CHECKED HARDEST AND FOUND SOUND

- The C-1 guard trace end to end. PYTHONPATH absent from image env, set per child by
  _env() with the sbatch's prepend semantics; every job process spawned through _run with
  that dict: heredoc, 36 mints, 2 fidelity runs, arena. sitecustomize.py is found because
  PYTHONPATH precedes site-packages, and `find` confirms the repo contains NO OTHER
  sitecustomize.py to shadow it. PORT-CHECK-1 runs before the first mint. The driver is
  unguarded -- correctly, in exact parity with the sbatch's bash, whose sha256sum/cat are
  equally unguarded and equally uncounted. Grandchildren inherit via os.environ;
  --num_workers 0 means there are none. n_processes_reporting = 39 preserved exactly.
- The C-2 container-side import path. Every module-level statement walked. guard_reason
  import, MANIFEST, STAGED and image.add_local_dir all under `if modal.is_local()`; the
  container-side stub raises. REPO_LOCAL, STAGE and TIMEOUT_S are the only other
  module-level statements and none touches the filesystem. _port_checks(manifest) consumes
  the passed argument, never MANIFEST. NOTHING ELSE AT MODULE LEVEL READS THE LOCAL FS.
- The SLURM_JOB_ID token genuinely restores aggregate's partition. _ledger_path:118-120
  names files led_{job}_{pid}_{ms}.json; aggregate:157-158 keeps only
  fn.startswith("led_{job}_") and routes the rest to stale. job_token contains hyphens but
  NO UNDERSCORE, so the prefix match is exact and unambiguous; writers and reader share one
  env dict; the self-exclusion d["pid"]==os.getpid() and d["t0"]==_T0 still works; pid
  reuse cannot collide because the filename also carries int(_T0*1000)%10**9.
- PORT-CHECK-5 hashes container bytes, before the first mint. Re-ran sha256sum on all nine
  live files -- 9/9 MATCH the freeze record section 3 and the port's table.
- Input closure complete. All 37 enumerated files exist, 46.1 MB. Traced the read path
  rather than trusting the prose: load_feats_from_CLIP is monkeypatched, and under the
  frozen CLI (--sparse_dictionary None, --lambda_seg 0, --archive_feats None,
  --tarc_target_source off) run_rac opens NO OTHER FILE; torch.save is a no-op. src/moka
  correctly excluded (imported only by utils/generate_VideoMLLM_embedding_lora_HF.py);
  src/logging holds no .py. headspace_fidelity.FLOOR's six trainlog patterns match the six
  uploaded filenames exactly. The image's pip set covers the entire chain.
- Data boundary intact. `git diff HEAD` on modal_probe_runner.py is EMPTY -- the upstream
  guard is byte-identical to the committed version. guard_reason evaluates media-extension
  -> forbidden-media-dir -> allowlist in that order, and c09_assert_uploadable's tolerance
  clause matches only reason.startswith("extension ") AND a declared suffix, so it can
  forgive only the third clause; NO MEDIA CAN PASS. The added C09-TEST-GUARD is strictly
  stronger than upstream and mirrors c09guard.is_test_like's site-packages exemption
  exactly. Walked the live staging tree: 80 files, 45 MB, ZERO test-like names, ZERO media
  (json 7 / jsonl 2 / npz 10 / pt 4 / py 50 / sbatch 1 / trainlog 6). The six banked arena
  JSONs contain no test-split number (only token matching `test` is `test_contact`).
- NO GATE WEAKENED, RELAXED, MADE TAUTOLOGICAL OR DROPPED. Section 4.1's ten rows match
  adjudicate's halts dict (c09_a0_arena.py:1474-1484) one-for-one, and
  ok_pub = all(halts) and _ledger["pass"] (1514-1515). GATE-LEDGER's pass is
  test_path_opens == 0 and len(procs) >= 1 (1958) -- the coverage emission and
  n_processes_expected_fresh_run: 39 are read by no predicate, exactly as classified. The
  guard does NOT go tautological in a container with no test files: is_test_like is purely
  path-based and never stats, so an attempted open still raises and still increments.
  GATE-FLOOR/GATE-PARITY-FOLD are carried VERBATIM and not re-derived in-container; the
  preflight uses the arena's own build_features/acc/mf1 with an NRM/X construction
  identical to main()'s (1737-1740), writes to a separate namespace, and its PROCEED is
  correctly declared non-licensing.
- The tie-break is sound, unambiguous and genuinely pre-registered. "COMPLETE"
  (verdict in {CONTINUE, KILL}) is exactly adjudicate's two-valued outcome --
  HALT_NO_VERDICT is returned iff ok_pub is false, so the definition is the code's, not a
  paraphrase. Ordering is now instant-vs-instant (I-8 closed). Registration precedes any
  result. NOT LAUNCHING DOES NOT VIOLATE ANYTHING THE TIE-BREAK PRE-REGISTERS -- the rule
  binds IF two runs complete; it never requires a second run to start, and "the strongest
  anti-gaming position is the one taken, since no cloud number exists to be peeked at." It
  cannot be gamed by preflight sweeping either, because the run draws its host
  independently. The one thing that does NOT close is the mechanism (H-1), not the logic.

## DISPOSITION

C-1 is decisive and the port STOPS on it: `a0` must not be invoked on the present Modal
plan. Recorded as section 9 of the port record and at the TIMEOUT_S definition in the
driver. H-1 mechanised (job_token-keyed destination, pre-mint INVOCATION_ sentinel with a
hard refusal, retries=0 pinned). H-2 partially mechanised (_envpin() captures pip freeze,
sys.version, Modal task/image ids and threadpool_info into the manifest); the base-digest
gap is declared, not closed. I-3 fixed (incremental snapshots + finally). I-4 fixed (full
import chain probed in the startup heredoc). I-5 fixed (PYTHONOPTIMIZE scrubbed in _env,
__debug__ asserted in the heredoc). I-1/I-2/I-6/I-7/I-8/I-9 recorded.
