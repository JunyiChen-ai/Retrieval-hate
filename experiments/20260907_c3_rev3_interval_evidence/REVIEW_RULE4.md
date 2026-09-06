# 规则 4 复核：候选 3 修订 3「证据路由的跨模态注意力 + 区间证据 HMM」（2026-09-07，独立 fable agent）

审的对象：本目录 `README.md` 第 0–4 节；代码 `model.py`（`EvidenceEncoder`、`BiasedMultiHeadAttention`、`EvidenceRoutedCMA`、`ERCA`）、`train.py`、`src/interval_evidence_hmm.py`。上一版外部审稿 `experiments/20260904_evidence_guided_attention/REVIEW_NOVELTY_GPT6ASTRA.md`，上一版规则 4 记录 `experiments/20260904_evidence_guided_attention/REVIEW_RULE4.md`（其领域内逐篇核对结果本次沿用，不重复列）。只按 `RESEARCH_ITERATION_RULES.md` 第 4 条的四种 STOP 情形判定；不按"可能退化""可识别性""可能有 shortcut"判定。

## 结论

**PASS（放行）。** 四种 STOP 情形均不触发：

| 情形 | 触发？ | 依据 |
|---|---|---|
| (1) 来源方法已用于 hateful video detection / localization | 否 | 本次 14 组检索加上一版逐篇核对：没有任何 hateful video 论文把外部 VLM 裁定做成 query 门控的注意力分数偏置，也没有任何一篇用区间观测的连续时间 HMM 融合 VLM 逐段裁定。2026 年新出的 CLARA（ACM MM 2026）、WWW 2026 Companion 的 agentic evidence attribution、ARCADE（arXiv 2603.21298）、MUWS'25 temporal label noise 均核对，见"领域内检查"。 |
| (2) 纯 training/test ensemble | 否 | 单一骨干、单一 VLM 裁定源（与候选 1 同一输入）；HMM 是对单一来源裁定的标签模型，不聚合多个模型的预测。 |
| (3) 纯 calibration / 后处理 / 平滑 | 否 | B.1 的 c 是从输入（裁定编码均值）算出、与骨干联合训练、进 MIL 损失的模型项，不是对训练后输出的事后校准；B.2 改的是注意力分数；A 改的是生成模型的观测结构与转移，且它的输出进训练（先验、块标签），不是推理时对分数做平滑。规则 3 禁止的是 inference 时的后处理，这里没有。 |
| (4) 纯工程（只调超参 / 换特征 / 加增强 / 改训练配置） | 否 | 新增带参数的结构部件（query 门 g_h、logit 校准头）与新的概率模型（8 状态增广链、连续时间转移、区间 OR 观测、正视频约束 EM），每项都有预注册的消融臂与可证伪预期。删 w_fine、固定融合配置属于配置改动，提案没有把它们当作 novelty 主张。 |

放行不等于认可 novelty 表述。三个改动里，B.1 和"正视频约束"单独都不构成方法贡献；B.2 在结构上是 WavLM 门控相对位置偏置把偏置来源换成证据；A 的可主张点是"任意查询区间 + 缺失裁定 + 视频归一化时间"的统一推断，不是连续时间 HMM 本身。论文写法要求见末节。

## 检索记录

工具：WebSearch（14 条查询）、WebFetch（论文全文 / 摘要页 5 篇）。只列与判定相关的命中。

| # | 查询串 | 相关命中（venue/年） |
|---|---|---|
| 1 | hateful video localization weakly supervised cross-modal attention VLM evidence bias HateMM HateClipSeg | MultiHateLoc（WWW 2026, arXiv 2512.10408）；LELA（arXiv 2602.09637）；HateClipSeg（arXiv 2508.01712）；MM-HSD（2025） |
| 2 | weakly supervised video anomaly detection query-conditioned attention bias gated evidence prompt-guided cross-modal attention 2025 | STPrompt（ACM MM 2024）；WSVAD-CLIP（J. Imaging 2025）；SESAD "Structured Evidence Selection for WSVAD"（arXiv 2607.10298, 2026-07）；Cross-Modal Fusion and Attention for WSVAD（arXiv 2412.20455）；Language-guided open-world VAD（arXiv 2503.13160） |
| 3 | continuous-time HMM interval observations weak supervision label model temporal localization LLM VLM verdicts | MLLM4WTAL（arXiv 2411.08466）；Temporal Label-Refinement for WS-AVEL（arXiv 2307.06385）；无组合命中 |
| 4 | Dugong weak supervision multi-resolution sequential data label model Snorkel video | Dugong（Sala et al., NeurIPS 2019, arXiv 1910.09505） |
| 5 | "evidence-routed attention" OR "evidence routing" attention hateful video multimodal | Interpretable Agentic Framework with Explicit Evidence Attribution（WWW 2026 Companion）；CLARA（arXiv 2608.15905, ACM MM 2026）；ARCADE / Intent Shifts（arXiv 2603.21298）；IARE（SIGIR 2026）；MultiHateLoc。**"evidence-routed attention" 这个词组没有命中。** |
| 6 | hateful video detection localization MLLM LLM segment-level verdict fusion HMM 2025 2026 HateClipSeg HateMM temporal | HateClipSeg；LELA；MultiHateLoc；无 HMM/标签模型命中 |
| 7 | query-dependent gated attention bias "gate" attention logits prior modulation transformer | Gated Attention for LLMs（Qiu et al., NeurIPS 2025 Oral, arXiv 2505.06708）：门在 SDPA 输出上，不在分数上；Forgetting Transformer（arXiv 2503.02130）：遗忘门作为数据依赖的加性 logit 偏置 |
| 8 | continuous-time hidden Markov model interval-censored noisy observations "at least one" OR noisy-OR aggregate observation HMM | CT-HMM for disease progression（Liu et al., NeurIPS 2015；arXiv 2110.13998, 2021）；Learning HMMs from aggregate observations（Singh et al., arXiv 2011.11236 / Automatica 2022，人群聚合，不是区间 OR） |
| 9 | hidden Markov model multiple instance learning constraint positive bag at least one positive EM "constrained" HMM | EM-MIL for WS action localization（Luo et al., ECCV 2020）；CHMM（Li et al., ACL 2021）；Sparse-CHMM（KDD 2022） |
| 10 | CHMM "conditional hidden Markov model" BERTifying weakly supervised NER | 同上，确认 CHMM 是 token 条件转移/发射，NER 域 |
| 11 | hateful video "hidden Markov" OR "label model" OR "probabilistic fusion" VLM segment verdicts temporal localization 2026 | CMFusion（arXiv 2505.12051）；Revealing Temporal Label Noise（MUWS'25, arXiv 2508.04900）；无 HMM/标签模型命中 |
| 12 | WavLM "gated relative position bias" query-dependent gate on attention bias | WavLM（Chen et al., IEEE JSTSP 2022, arXiv 2110.13900）；已从 PDF 原文确认公式 |
| 13 | LAVAD Holmes-VAD VERA training-free VAD LLM scores temporal fusion HMM smoothing Markov | LAVAD（CVPR 2024）：相似度加权平滑；VERA（CVPR 2025）：粗到细分数；VADTree（arXiv 2510.22693）：层次树；均无 HMM |
| 14 | EM HMM "at least one" positive state constraint conditional likelihood weakly labeled sequences | Constrained Baum-Welch for partial labels（BMC Bioinformatics 2021）；Prediction-constrained HMM（Sudderth 组, ICML 2021 workshop）；Neural-Hidden-CRF（arXiv 2309.05086） |
| 15 | hidden Markov model window-level labels weak supervision "any" frame in window positive sound event detection | HSMM 后处理用于弱监督 SED（Interspeech 2025）；无"区间 OR 观测因子"形式 |
| 16 | normalized video time transition duration-invariant HMM video segmentation | RNN-HMM for WS temporal action segmentation（arXiv 1906.01028）的相对长度模型；无"按视频比例跑连续时间链"的先例 |
| 17 | hateful video Qwen2.5-VL window verdicts fusion temporal localization weakly supervised 2026 | MultiHateLoc；TANDEM（arXiv 2601.11178）；无新命中 |

全文/摘要核对（WebFetch）：CLARA（arXiv 2608.15905 HTML）；SESAD（arXiv 2607.10298 HTML）；ARCADE（arXiv 2603.21298 abs；PDF 超限未读全文）；Revealing Temporal Label Noise（arXiv 2508.04900 HTML）；WavLM（arXiv 2110.13900 PDF，用 pdftotext 抽公式）。WWW 2026 Companion 那篇 ACM DL 返回 403，只能依据检索摘要与同题 MARS（arXiv 2601.15115）判断。

## 三个改动的最近先例与新旧边界

### B.1 视频级校准只加在 logit 上（c = Linear(hid→1)(mean_t e_t)，不进表示）

最近先例：
- 上一版规则 4 记录已指出修订 2 的 C 在线性头下恰等于每视频一个 logit 偏移；B.1 只是把这个等价式落实到代码，把 c 从表示里挪到 logit 上，使 CMAL 与块级 MIL 不再看到它。
- MIL 里的 bag 级摘要广播回实例分数、GIG-VAD（Lv et al., 2021）的全局模式向量、PEL4VAD（TIP 2024）的全局-局部聚合，仍是最近的结构先例。
- 从形式上看，它就是一个由输入侧特征算出的、与骨干联合训练的加性 logit 项。

新与不新：
- 不新：加性视频级偏移本身；"把上下文项从表示移到输出"是普通设计选择。
- 新的只有它在本方法里的角色：它使"对比损失与块级监督只作用于内容"这句机制陈述在代码里成立，并有 `ctx_in_rep` 臂对照。它不能单独作为 novelty 主张，只能作为机制一致性修正报告。
- 不属于规则 4 情形 (3)：它不是对训练后分数的事后校准，输入是裁定分布，训练时进 MIL 损失。

### B.2 query 门控的证据路由（score_h(i,j) += g_h(e_i)·β_h(e_j)，g_h = 2σ(Linear(e_i))，零初始化 g ≡ 1）

最近先例（按结构接近程度）：
- **WavLM 门控相对位置偏置**（Chen et al., IEEE JSTSP 2022, arXiv 2110.13900，§III 公式已核对）：g_i^{update}, g_i^{reset} = σ(q_i·u), σ(q_i·w)；r̃_{i−j} = w·g_i^{reset}·d_{i−j}；r_{i−j} = d_{i−j} + g_i^{update}·d_{i−j} + (1−g_i^{update})·r̃_{i−j}，加到注意力 logit。**这就是"query 侧 sigmoid 门 × 加性偏置"的模板**，B.2 与它结构相同，差别是门的输入从 query 向量 q_i 换成 query 秒的证据编码 e_i，偏置从相对位置 d_{i−j} 换成 key 秒的证据 β_h(e_j)。
- **Forgetting Transformer**（Lin et al., 2025, arXiv 2503.02130）：数据依赖的门累积成加性 logit 偏置；同属"数据依赖偏置"一支。
- **Gated Attention**（Qiu et al., NeurIPS 2025 Oral, arXiv 2505.06708）：query 依赖的 sigmoid 门，但作用在每头输出上，不作用在分数上；不是同构先例。
- Graphormer（NeurIPS 2021）、ALiBi、T5 偏置：上一版已记录的静态加性偏置先例，修订 2 的 key 偏置属于这一类；B.2 相对它们的改动正是引入 query 依赖。
- WSVAD/WTAL 域：MLLM4WTAL（arXiv 2411.08466）用 MLLM 文本先验与视频特征内积调制注意力，query 依赖来自内容相似度，不是外部裁定的门；SESAD（arXiv 2607.10298，2026-07，全文核对）的"scene/action query 引导证据选择"是对内部多分支语义表示的 softmax 加权，不是外部证据、不是注意力分数偏置、无 HMM。
- hateful video 域：CLARA（ACM MM 2026，全文核对）把 Qwen3-VL 生成的 rationale 经门控后与 clip 表示拼接进 Transformer，做视频级分类；门在源级别（rationale 分支 vs clip 分支），不是注意力分数上的 query×key 证据项，无定位、无 HMM。MultiHateLoc 的 CMA 无偏置。

新与不新：
- 不新：query 门 × 加性偏置这个算子（WavLM）；证据作为 key 偏置（修订 2 已有，Graphormer/ALiBi 谱系）。
- 新：门与偏置都由**外部冻结 VLM 裁定（经 HMM 后验）的编码**算出，query 秒自己的证据决定它从证据秒取多少内容；两个跨模态方向共用；在弱监督 MIL 骨干里、在 hateful video 任务上都没有先例。这是"把 WavLM 的门控偏置从位置信号迁到证据信号"的迁移，符合第 4 条"迁移自其他任务、没在 hateful video 用过"的口径。
- `key_bias`（g ≡ 1）与 `shared_bias` 两个臂能把"query 条件化"和"逐头"分别隔离，是上一版审稿人要求的对照，这里已预注册。

### A 区间证据 HMM（真实区间 OR 观测、视频归一化时间的两态连续时间转移、正视频约束 EM、缺失裁定不发射）

最近先例（按部件）：
- **区间/多分辨率弱监督标签模型**：Dugong（Sala et al., NeurIPS 2019）——多分辨率标注源对序列元素集合投票，Ising/因子图，参数可辨识；Snorkel；CHMM（Li et al., ACL 2021）、Sparse-CHMM（KDD 2022）——多源噪声标签的 HMM 标签模型，token 条件转移。这些是上一版审稿人点名的先例。它们都不把一条观测写成"区间内任一隐状态为 1 的带噪 OR 因子"并在增广状态 (s, h_fine, h_coarse) 上做精确前向后向。本项目 `src/verdict_hmm.py`（修订 2）已经有 OR 因子与 EM，A 的改动是把索引层次换成真实区间。
- **连续时间 HMM**：CT-HMM（Liu et al., NeurIPS 2015；Efficient learning of CT-HMM, arXiv 2110.13998, 2021）——两态链 exp(QΔt) 闭式、不规则间隔观测；教科书内容。A 用的正是这个闭式；Δt 用视频比例而非秒是 4.1 节的经验取舍（按秒在 HateMM 掉 .014 AP / .028 ROC），检索 16 没找到"按视频归一化时间跑连续时间链"的先例，但这是一个建模选择，不是新算法。
- **聚合观测 HMM**：Learning HMMs from aggregate observations（Singh et al., Automatica 2022）——人群级聚合计数，不是同一序列的区间 OR。
- **正视频约束**：MIL 的"正 bag 至少一个正实例"假设（Dietterich et al. 1997）在 EM 里的用法——EM-MIL（Luo et al., ECCV 2020）；constrained Baum-Welch（BMC Bioinformatics 2021）对部分标注序列限制隐路径；prediction-constrained HMM（ICML 2021 workshop）。A 的做法（从后验里减去全零路径的质量）是这一假设在 HMM 后验上的直接实现。
- **缺失观测不发射**：HMM 标准处理。
- **视频级随机效应**：本修订试了、EM 发散、未采用，记为负结果，不进主张。
- VAD 域的 LLM 分数时间融合：LAVAD（CVPR 2024）相似度加权平滑、VERA（CVPR 2025）粗到细、VADTree（2025）层次树，均无概率标签模型。
- hateful video 域：LELA、TANDEM、MARS、IARE、SafeLens、HVGuard 直接用 VLM/LLM 输出或做 CoT/RL，无标签模型；CLARA 无；检索 6/11/17 无 HMM 命中。

新与不新：
- 不新：连续时间两态链；EM；正 bag 约束；缺失观测处理；OR 因子（修订 2 已有）。
- 新（可主张的）：把每条 VLM 裁定绑定到它真正看过的时间区间，在全部区间边界的并集上做精确推断，因此不要求粗细嵌套、允许缺失裁定、允许任意查询区间；同一个模型在嵌套网格上与索引模型后验一致到 1e-9（README 第 4 节）。这与上一版审稿人建议方向 ④ 一致，在 hateful video 定位上没有先例，在 WSVAD 上也没找到同形式先例。
- 必须如实写：它是 Dugong/CHMM 谱系里一个针对"VLM 区间查询"的具体观测结构，不是新的标签模型理论。

## 领域内检查（本次新增核对；上一版已核对的 12 篇不重复列）

| 论文 | 核对到的结构 | 有无 B.2 / A |
|---|---|---|
| CLARA（arXiv 2608.15905，ACM MM 2026，全文 HTML） | utterance 对齐 clip，Whisper/ViT/BERT，MoE clip 编码器；Qwen3-VL-8B 两步提示生成 rationale，经源级门控后与 clip 表示拼接进 Transformer；局部-全局 InfoNCE；视频级分类；HateMM / MultiHateClip / DeHate | 无（rationale 是拼接进序列，不是注意力分数上的 query×key 证据项；无定位；无 HMM） |
| Interpretable Agentic Framework with Explicit Evidence Attribution（WWW 2026 Companion；ACM DL 403，据检索摘要） | agentic、可解释、证据归因 | 无（无训练骨干、无标签模型；与 MARS 同一路线） |
| ARCADE / Intent Shifts（arXiv 2603.21298，2026-03/04，摘要页） | 控辩双方 agent 辩论，H-VLI benchmark，视频级 | 无 |
| Revealing Temporal Label Noise in Multimodal Hateful Video Classification（MUWS'25, arXiv 2508.04900，全文 HTML） | 用已有时间戳截取 hate/non-hate 片段做分类对比分析 | 无 |
| SESAD（arXiv 2607.10298，WSVAD，全文 HTML） | scene/action query 对内部多分支语义表示加权；双原型 | 无（非 hateful video；query 引导的是内部表示，不是外部证据偏置；无 HMM） |

## 对论文写法的要求（不阻断，供第 14 条与 novelty 复查用）

1. B.2 的相关工作必须引 WavLM 门控相对位置偏置（Chen et al., IEEE JSTSP 2022）作为结构模板，写明本方法 = 门与偏置的输入都换成证据编码；并保留上一版要求的 Graphormer / ALiBi / T5 / MLLM4WTAL 对照。novelty 只能落在"证据来源 + query 秒证据门控 + 弱监督跨模态 MIL 骨干"，不能落在"门控注意力偏置"本身。
2. B.1 只作为机制一致性修正报告（"校准项不进表示"），不列为贡献。
3. A 的相关工作必须引 Dugong（NeurIPS 2019）、CHMM（ACL 2021）、CT-HMM（NeurIPS 2015 / 2021）、EM-MIL（ECCV 2020）；贡献表述为"任意查询区间与缺失裁定下的统一精确推断 + 视频归一化时间转移"，并给出 4.1 节按秒 vs 归一化的数字；不能写成"新的标签模型"。视频级随机效应的负结果如实报告。
4. `key_bias`、`shared_bias`、`index_hmm`、`no_constraint`、`seconds_time` 五个臂是本次放行依据的一部分：B.2 与 A 各自只有在预注册第 2 节第 3/5 条成立时才进主张。
