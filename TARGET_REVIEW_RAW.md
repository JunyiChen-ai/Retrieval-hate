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
