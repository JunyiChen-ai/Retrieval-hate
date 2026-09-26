# 规则 4 proposal review（第 4 版）— 20260925_query_paradigm 第 15 节

**判定：GO。** 规则 4 的四种 STOP 情形均不成立：(1) 新部件（树节点先验势：把视频头的注意力池化沿二分问题树自底向上算到每个节点、零初值线性头给每个内部节点的 OR 状态 z_n 一个势 φ_n、作为因子 exp(φ_n z_n) 进入精确 OR 树推断、由 VLM 答案的树似然训练）以及"神经网络参数化的树结构概率模型节点势 + VLM 提问答案"这一组合，在 hateful video detection / localization 文献中检索不到；(2) 不是 training/test ensemble；(3) 不是 calibration / 后处理 / 平滑；(4) 不是纯工程技巧。词级 ASR 时间戳属于规则 5 允许方法自带的输入，不单独过审，且不构成规则 3 的 ensemble。

- 审稿人：独立 agent（Fable 5.1），2026-09-27，一次性。
- 依据：`RESEARCH_ITERATION_RULES.md` 第 4 条（唯一四种 STOP）、第 3 条（ensemble / 后处理禁令）、第 5 条（方法可自带输入）。未运行任何训练或 GPU 任务，未改动审稿文件以外的任何文件。
- 未以"可识别性 / 可能退化 / shortcut 风险 / 预期涨幅"作为判定依据（规则 4 明文排除）。

## 1. 审阅了什么

- `README.md` 第 2 节（方法）、第 3 节（来源与 novelty）、第 7–10 节（第 1–3 次修改，含链先验、零膨胀、锚定答案模型）、第 11–13 节（第 4 次修改的设计检查、两个对照、传播检查）、第 14 节（审稿 concern C1–C6 的诊断、比较式问法、转录时间错位）、第 15 节（第 4 版提案，本次审阅对象）。
- 上一次规则 4 审稿 `REVIEW_RULE4.md`（格式、34 次检索记录、已定的最近先例 VADTree / FV-Action / Learning to Watch / Cuturi 分组检测 BOED）。
- 代码 `git show 3cf8d34`（`ctree.py`、`model.py`、`policy.py`、`cpolicy.py`、`train.py`）与 `data.py:load_answers`、`scripts/asr_words.py` 头部说明，用于确认方法实际做了什么（第 2 节）。

## 2. 代码确认：第 4 版实际做了什么

| 第 15 节的表述 | 代码位置 | 确认 |
|---|---|---|
| 节点池化向量 e_n = Σ_{t∈n} softmax_{t∈n}(w_t) h_t，w_t 是视频头的注意力 logit，h_t = a_out_t + v_out_t，沿树自底向上 | `model.py:PriorNet.node_logits`：`w = self.att(h)`（与视频头同一 `att`），逐层从叶到根，父节点向量 = 两子节点向量按 `exp(lL − l)`、`exp(lR − l)` 加权，权重对数和 `l = logaddexp(lL, lR)` | 一致；根节点的向量就是视频头 `vid` 读的池化向量 |
| φ_n = w_φ·e_n + b_φ，线性层零初值；叶节点无势 | `self.node = nn.Linear(hid, 1)`，权重与偏置 `zeros_`；`phi` 只在 `nodes[n_leaf:]`（内部节点）上 `index_put`，叶为 0 | 一致；训练起点与第 3 版完全相同 |
| 因子 exp(φ_n z_n) 只挂在单个节点的 OR 状态上，G = 1 下生效，也进入无答案归一化 | `ctree._up`：`a1 = a1 + stack([0, phi])`，与 `eta` 同一位置（z = 1 列）；`ctree.up` 在零膨胀分支对 `v1`（有答案）与 `v1_prior`（无答案）都传 `phi`；`assert phi is None or chain.zero_inflated` | 一致；树上和积仍精确（提交说明：T ≤ 9 穷举误差 < 1e-15） |
| 训练目标不变，φ 由树似然训练 | `train.py`：`phi = model.node_logits(v_out, a_out, fo)` 来自同一次前向，传入 `ctree.log_evidence(..., phi)`；`loss_ans`、`loss_g` 形式未改 | 一致 |
| 测试时后验与 EIG 都带节点势 | `policy.video_prior_nodes` 五 crop 平均 φ；`cpolicy.run_batch` 把 `PHI` 传给每一步的 `ctree.marginals`，包括 EIG 钳住节点状态的两次推断 | 一致 |
| 只换转录，问法、帧、模型不变 | `data.load_answers(corpus, source)`：`"words"` 读 `answers_words_qwen7b_mod5*.jsonl`；`train.py` DEFAULTS 增 `answer_source "k30"`、`node_prior False` | 一致 |
| 词级转录 = 单一 ASR + 强制对齐 | `scripts/asr_words.py`：whisper-large-v3 按 30 秒窗转写（贪心、只出文字），torchaudio `WAV2VEC2_ASR_BASE_960H` CTC 强制对齐给每个词时间；输出 `data/ASR_words/<C>/words.jsonl` | 一致 |
| 配置闸门 | `train.py`：`node_prior` 要求 `backbone macil`、`prior chain`、`chain_form zero_inflated`、`objective tree`、`answer_model != refit`、`query_level 0` | 两语料同一配置（规则 13） |

## 3. 检索记录

工具：WebSearch 32 次、WebFetch 8 次（读全文或摘要）。检索日期 2026-09-27。上一次审稿的 34 次检索（问题树、noisy-OR、EIG、带噪标注者、零膨胀）不重复，本次只查第 4 版新元素。

### 3.1 hateful video 侧（STOP 情形 1 的直接依据）

| # | 查询 | 结果 |
|---|---|---|
| 1 | hateful video localization hierarchical attention pooling multi-scale segments tree nodes weakly supervised | MultiHateLoc、MM-HSD、STPN、弱监督 video re-localization；hate 侧无分层 / 树上池化 |
| 2 | hateful video detection "binary tree" OR "binary partition" OR "hierarchical tree" temporal segments prior multimodal | TANDEM、label-noise 分析（2508.04900）、HateClipSeg、Buffalo MLLM capstone；无树结构 |
| 3 | hate speech video localization multi-scale span prior segment-level potential neural CRF HateMM MultiHateClip | HateClipSeg、TANDEM、MM-HSD、LELA；无 span 势 / CRF |
| 4 | hateful video WhisperX word-level timestamps forced alignment transcript segment localization hate | 只有 WhisperX 工具页与教程；无 hateful video 应用 |
| 20 | hierarchical attention network hate speech video segments frames multimodal hate detection HateMM hierarchical | RAMF（两级注意力，但层级在模态 / 推理之间，不在时间轴）、MM-HSD、HateMM 原文；无时间层级 |
| 26 | hateful video multi-scale temporal pyramid MIL hierarchical segments weakly supervised HateMM MultiHateClip 2026 localisation coarse fine | 只有 MultiHateLoc（单一帧级分辨率）；无多尺度 / 分层 hate 定位 |
| 27 | hateful video detection "conditional random field" OR "hidden Markov" OR "graphical model" OR "Bayesian network" temporal segments neural potentials | 无 hateful video 概率图模型工作；命中的 CRF 势句子出自 Li–Mahadevan–Vasconcelos TPAMI 2014 人群异常检测（第 31 条核对） |
| 30 | spoken hate speech detection word-level timestamps localization audio transcript alignment frame-level toxic speech span | "An Investigation into Explainable Audio Hate Speech Detection"（SIGDIAL 2024：级联 ASR→文本判定→映射回音频帧，对比端到端）、MUTEX（乌尔都语 toxic span）、词级 Whisper 打码管线（GitHub）。音频 hate 定位有用 ASR 时间戳的级联管线；无视频、无 VLM、无树 |

全文核对（WebFetch）：MultiHateLoc `arxiv.org/html/2512.10408v1`——组件为模态专属 Transformer 时间编码器、动态模态选择 + 跨模态注意力、跨模态对比、模态感知 top-K MIL；**无分层 / 树 / 多尺度池化、无段级势或图模型、无词级 ASR 时间戳（句级特征扩展填充）、无 VLM / LLM**。

### 3.2 其他任务中新元素的先例（STOP 情形 1 的补充，并给出最近先例）

| # | 查询 | 结果 |
|---|---|---|
| 5 | tree-structured CRF neural node potentials video anomaly detection temporal localization exact inference | VADTree；"tree structured CRF with unary potential ... action unit"（专利，面部动作单元）；"Structured Learning of Tree Potentials in CRF"（TNNLS 2017，这里的 tree 指决策树势，名称撞车）；无 VAD 树 CRF + 神经节点势 |
| 6 | hierarchical attention pooling binary partition tree temporal action localization bottom-up node representation | 分层自注意力动作定位、分层 LSTM（CVPRW 2020）、HR-Pro、HAN 动作分割；无二分树池化 |
| 7 | weakly supervised TAL multi-scale segment potentials semi-Markov CRF neural span scores | CPMN、RefineLoc、多分辩率 WTAL（2506.18261）；WTAL 无 semi-Markov / 神经 span 势 |
| 8 | hierarchical MIL nested bags multi-scale video weak labels OR aggregation tree | HAMIL（二叉合并树 + 卷积聚合单元，显微图像）、EM-MIL、multi-attention networks；无视频 OR 树 |
| 9 | noisy-OR tree neural network node potentials exact sum-product learned prior video localization | 图诱导 SPN、noisy-OR BN 的 max-product 学习（2302.00099）；无视频 |
| 10 | VADTree hierarchical tree node scores learned prior training-free follow-up | VADTree（NeurIPS 2025）、Probe-VAD、"Is VAD Misframed?"；无带学习节点先验的后续 |
| 11 | coherent hierarchical multi-label classification tree constraint OR semantics neural node scores max constraint | C-HMCNN（NeurIPS 2020：MCM 父输出 = 子树输出的 max，MCLoss）、硬逻辑约束多标签（2103.13427）、HMC 稀有节点（2602.08986） |
| 12 | learned prior network combined with VLM yes/no answers Bayesian fusion hierarchical windows video anomaly localization | STPrompt、Parser-Free VLM Verification（2609.07455）、"Zoom In, Reason Out"（2604.23724，运动学引导的贝叶斯 + VLM）、联邦二值门控；无"学习的树节点先验 + VLM 答案"融合 |
| 13 | span-based neural CRF constituency parsing span scores tree potentials exact inference | Stern–Andreas–Klein ACL 2017、Kitaev–Klein ACL 2018、Zhang–Zhou–Li IJCAI 2020（神经 CRF 成分句法：LSTM/自注意力特征给每个 span 一个势，树上精确 inside 推断） |
| 14 | recursive bottom-up attention pooling tree children merge log-sum-exp softmax pooling Tree-LSTM video segments | Tree-LSTM（ACL 2015）、Tree-Structured Attention with Hierarchical Accumulation（ICLR 2020）、attentive recursive trees；全部 NLP 句法树 |
| 15 | weakly supervised VAD multi-scale temporal MIL hierarchical segments nested windows pyramid pooling | HiTESS（ESWA 2025）、Adaptive Multi-Granularity Temporal Modeling（2609.05066）、DE-Net 多尺度、temporal resolution feature learning；均无树上精确推断、无 VLM |
| 16 | video anomaly detection tree structure learned node anomaly score hierarchical segments weakly supervised 2025 2026 | SESAD（2607.10298）、VADTree、双曲空间 WSVAD、事件完整性；无学习的树节点势 |
| 17 | vision-language model answers as noisy observations factor graph learned neural prior potentials exact inference tree localization | 只有机器人定位的因子图（Ground Encoding）、VLM-Loc；无视频时间定位 |
| 18 | 2D temporal adjacent network moment retrieval span-level pooling multi-scale span scores hierarchical span prior | MS-2D-TAN（TPAMI 2021）、弱监督多尺度 2D 表示（2111.02741）、多尺度高斯先验（PLOS One）；全监督 / 无 OR 语义、无 VLM |
| 19 | dyadic interval tree OR segment tree neural network temporal event localization multi-scale span classifier | Hypotheses Tree Building（MHST，2301.01871）、moment localization 时间关系（1908.03846）、MSST；无二分区间树概率模型 |
| 21 | noisy-or pooling hierarchical multi-scale weak label sound event detection tree | He et al. Interspeech 2019 分层池化（帧→段→片，无参数）、Wang et al. 五种 MIL 池化比较（noisy-or 池化不能定位）、HiPool |
| 22 | anomaly prior network guides which video segments to query VLM coarse-to-fine learned prior hierarchical query selection | VERA、HVLMCLD-VAD、细粒度 prompting（2510.02155）；无学习先验驱动的查询选择 |
| 23 | multi-granularity anomaly scores hierarchical video anomaly understanding VLM tree HolmesVAU | Holmes-VAU（CVPR 2025：异常打分器 + 密度采样器给 MLLM 选帧，无概率融合）、GuardReasoner-Omni |
| 24 | "hierarchical" OR "tree" Bayesian "at least one" constraint neural potentials exact inference weakly supervised localization | 只有弱监督语义分割（SEC、constrained CNN）；无时间轴 OR 树 |
| 25 | learning deep structured models neural network potentials CRF joint training exact inference | Chen–Schwing–Yuille–Urtasun ICML 2015、Zheng et al. ICCV 2015（CRF-as-RNN）、Arnab et al. 2018 综述 |
| 28 | train localization network to predict VLM answers at multiple temporal scales hierarchical windows likelihood weakly supervised | OV-TAL、TOGA、FIAN；无多尺度 VLM 答案似然训练 |
| 29 | multiple instance learning nested bags "sub-bag" multi-level bag hierarchy video anomaly detection | 多层 MIL 视频概念检测（专利）、PU-MIL 异常检测；无嵌套袋树 |
| 31 | "multi-scale" scores "potentials of a conditional random field" anomaly judgments | Li–Mahadevan–Vasconcelos TPAMI 2014（人群场景异常，多尺度 MDT 分数作 CRF 势）；非 hate、非树、非 VLM |
| 32 | "Hierarchical Temporal Sequence Segmentation" weakly supervised VAD | HiTESS：分层时间分段 + BiLSTM/MHA，分层 MIL，段级与序列级分数聚合；XD-Violence / UCF-Crime |

摘要 / 全文核对（WebFetch）：VADTree `arxiv.org/pdf/2510.22693`（全部节点分数来自 VLM + LLM，**无任何训练的节点先验**，融合为节点相关的启发式而非概率模型，节点非自适应查询）；SESAD 2607.10298（原型结构上的证据选择，未提树 / VLM）；Adaptive Multi-Granularity 2609.05066（snippet 级 + 事件级两粒度，相似度融合，无概率模型、无 VLM）；MHST 2301.01871（按视觉-语言语义合并帧成树、产生片段假设排序，无树上概率推断）；HiTESS（ScienceDirect 403，摘要由检索 32 取得）；C-HMCNN 2010.10151（摘要页；MCM = max 的细节来自 NeurIPS 评审页片段 "MCM_B = max(h_B, h_A)"）；HAMIL 2103.09764（卷积聚合单元、显微图像；二叉合并树的说法来自检索 8 的二手摘要）。

## 4. 四种 STOP 情形逐条核对

| 情形 | 判定 | 依据 |
|---|---|---|
| 来源方法已用于 hateful video detection / localization | 否 | 第 3.1 节 8 条检索 + MultiHateLoc 全文核对：hate 侧现有方法（MultiHateLoc 帧级 MIL、TANDEM RL 时间戳、LELA / CLARA 逐帧或逐 clip 提示、HVGuard / RAMF / MARS / IARE 视频级推理、SafeLens 逐段策略）没有一篇在时间轴的树 / 多尺度区间上放学习的先验势，也没有一篇用神经网络参数化的概率图模型节点势与 VLM 答案做精确融合。第 3.2 节 24 条检索：树节点势的最近先例在其他任务（神经 CRF 句法分析的 span 势、C-HMCNN 的层级 max 约束、HAMIL / 分层池化 SED 的树上聚合、VADTree 的树 + VLM 节点打分），无一是 hateful video；"学习的树节点势 + VLM 提问答案"组合在任何任务上都没有找到 |
| 纯 training / test ensemble | 否 | 节点势与每秒势、视频头来自同一个网络的同一次前向（`node_logits` 复用 `att`、`a_out + v_out`），是单模型的新输出头；最终分数仍是一个后验。词级转录：whisper-large-v3 产生词，wav2vec2 CTC 只做强制对齐给这些词标时间（不是自由预测），两者都不输出任何关于 hate 的 prediction / feature / posterior / pseudo-label / decision，转录只作为唯一 VLM 的输入；规则 5 明文把"词级 ASR 时间戳"列为方法可自带的输入，规则 3 的"单一预训练编码器 / VLM 作为特征来源不算 ensemble"覆盖此情形。旧 K30 缓存本身也来自 Whisper。不构成 ensemble |
| 纯 calibration / 后处理 / 平滑 | 否 | φ_n 是生成模型先验里训练得到的因子，在推断内部起作用并进入无答案归一化项与 EIG；不是对输出分数的事后变换，无按语料路由。转录替换是输入变更 |
| 纯工程技巧 | 否 | 节点势改变了模型结构（新因子 + 新头）、推断（每个节点后验与 EIG 都变）与训练信号的去向（答案似然的梯度经树进入每个节点的池化），不是只调超参 / 只换特征 / 只加增广 / 只改训练配置。单独看词级转录属于"换输入"，按规则 5 不构成 novelty 也不需过审；第 4 版的 novelty 主张应落在节点势与整体 QTL 框架上，第 15.2 节预注册的"原转录答案（k30）"消融正好把两项改动的贡献分开 |

## 5. 非阻断说明

### 5.1 论文必须对照与引用的最近工作（按与节点势的关系分组）

1. **神经网络参数化的概率图模型势（一般先例）**：Lafferty et al. 2001（CRF）；Chen, Schwing, Yuille & Urtasun ICML 2015 "Learning Deep Structured Models"（深度特征作 MRF 势、联合训练）；Zheng et al. ICCV 2015 CRF-as-RNN；Arnab et al. 2018（CRF + DNN 综述）。第 4 版的定位：树结构、OR 语义、观测是 VLM 答案而非标签。
2. **树结构 CRF 的神经 span 势 + 精确推断**：Stern, Andreas & Klein ACL 2017；Kitaev & Klein ACL 2018；Zhang, Zhou & Li IJCAI 2020 "Fast and Accurate Neural CRF Constituency Parsing"。区别：句法树是隐变量、span 势决定选哪棵树；本方法树固定为二分，节点变量是叶的 OR，势由弱标签 + VLM 答案的边际似然训练。
3. **层级上的 OR / max 约束 + 神经节点分数**：Giunchiglia & Lukasiewicz NeurIPS 2020 C-HMCNN（父类输出 = 子树输出的 max，MCLoss）。区别：确定性约束层、标签层级、全监督；本方法是概率模型里的软因子、时间轴分区、弱监督。
4. **树 / 层级聚合的 MIL 与弱标签池化**：HAMIL（Tu et al., Pattern Recognition 2022；二叉合并树 + 卷积聚合单元）；He et al. Interspeech 2019 "Hierarchical Pooling Structure for Weakly Labeled SED"；Wang et al. ICASSP 2019 池化函数比较（noisy-or 池化不能定位，本方法的 OR 是概率模型的结构而不是池化函数，应说明区别）；HiPool。
5. **树上自底向上的注意力 / 递归组合**：Tai, Socher & Manning ACL 2015 Tree-LSTM；Nguyen et al. ICLR 2020 Tree-Structured Attention with Hierarchical Accumulation。本方法的节点池化是同一注意力 logit 的逐层 log-sum-exp 合并，无新参数（只在读出头有 hid + 1 个参数），可以引这些工作说明"沿树递归池化"不是新的，新的是它作为 OR 树因子并由答案似然训练。
6. **多尺度区间打分（temporal grounding / VAD）**：2D-TAN（AAAI 2020）与 MS-2D-TAN（TPAMI 2021）对全部 (起点, 时长) 区间池化打分；Li, Mahadevan & Vasconcelos TPAMI 2014 多尺度分数作 CRF 势；HiTESS（ESWA 2025）分层分段 + 段级 / 序列级分数；Adaptive Multi-Granularity Temporal Modeling（2609.05066）；Multi-Scale VAD by Multi-Grained Spatio-Temporal Representation（CVPR 2024）。区别：全监督或 MIL 打分，无树上精确推断，无 VLM 观测。
7. **树 + VLM 节点查询**：VADTree（NeurIPS 2025，全文核对：节点分数全部来自 VLM/LLM、全节点查询、启发式融合、无学习先验）——第 4 版与它的差别比第 3 版更大（多了学习的节点先验），论文里应把"节点先验 + 精确融合 + 自适应提问"三点并列对照；Holmes-VAU（CVPR 2025，异常打分器只用于给 MLLM 选帧）；MHST（2301.01871，语义合并树生成片段假设）；GtS（2608.11260）；以及上一次审稿列出的 FV-Action、Learning to Watch、Parser-Free VLM Verification、Cuturi 等分组检测 BOED。
8. **词级转录的动机引用**："An Investigation into Explainable Audio Hate Speech Detection"（SIGDIAL 2024）用 ASR 时间戳做音频 hate 帧级定位（级联 vs 端到端）；WhisperX（Bain et al. Interspeech 2023）是本方法词级对齐的做法来源。这两篇只支撑第 14.4 节的转录错位发现与修正，不是 novelty 依据。

### 5.2 规则 3 / 规则 13 相关说明

- 词级 ASR 不是 ensemble（第 4 节）。但方法管线里现在有两份转录：VLM 节点问法用词级转录，骨干的 BERT 文本行仍用旧 K30 转录（第 15.1 节"骨干的特征不变"）。两者都是输入而非模型对 hate 的预测，不触规则 3；论文必须写明，且审稿人会问"为何 BERT 行不用同一份转录"。若日后把 BERT 行换成词级转录，属于规则 5 的输入变更，不需再过规则 4。
- 两语料同一配置（`answer_source words`、`node_prior true`），无按语料开关（规则 13）。

### 5.3 值得记录的实现观察（不阻断，供 code review 与写作参考）

1. **根节点的势是恒等因子。** `node_logits` 对包括根在内的全部内部节点算 φ；零膨胀下 G = 1 时 z_root ≡ 1，exp(φ_root) 同时乘进 V1(有答案) 与 V1(无答案)，在 `up` 的 `g + v1 − v1_prior` 中抵消，在 `marginals` 的对数导数中也抵消。φ_root 与 `self.node` 对根的那次读出无任何效果、无梯度。无害；写作时"每个内部节点一个势"应改为"每个非根内部节点"，或在代码里跳过根，以免读者以为根势替代了 g。
2. **节点势的训练信号只来自正例视频的答案项。** 零膨胀下负例视频 G = 0，没有 z 因子；标签项只训练 g。所以 φ_n 学的是"正例视频里 VLM 在该区间说有的倾向"，与 s_t 同源。这是设计使然，不是问题；但解释 Q4 消融时要明确它和 s_t、链的分工（φ 给的是区间尺度的信息，s_t 与链只给秒级和相邻秒）。
3. **φ 与视频头共享注意力 logit `att`。** φ 的梯度会改变 `att`，进而改变 g 的池化。这正是第 15.1 节"骨干的计算图就是问题树"的含义，但意味着"去掉节点势"消融同时也去掉了这一路对 `att` 的训练；消融结论应表述为"节点势（含其对注意力的训练）"。
4. 第 15.2 节的四个臂（词级 + 节点势 / 词级无节点势 / K30 + 节点势 / 第 3 版）构成完整 2×2，可以分开报两项改动各自的贡献，设计恰当。
5. 上一次审稿第 5 节第 4 条（训练期用全部嵌套节点答案、测试期只用 ≤ 8 个，条件独立假设在训练期被更强地违反）在第 4 版依旧成立，且节点势也由这些嵌套答案训练；第 14.2 节的 oracle 与上限诊断已部分回答了这一点，论文写作时应保留该说明。
