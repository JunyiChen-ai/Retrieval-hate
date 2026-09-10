# 规则 4 复核：机制 D（证据分解 E_t = ell_fine + x_t + v）（2026-09-10）

审稿人：独立 agent（Claude Fable 5.1），只读，不跑训练；依据 `RESEARCH_ITERATION_RULES.md` 第 4 条（四类阻断：(a) 来源方法已用于 hateful video detection / localization；(b) 纯 ensemble；(c) 纯 calibration / 后处理 / 平滑；(d) 纯工程技巧）。前两份复核（`REVIEW_RULE4.md` 审 A–D 四部件，`REVIEW_RULE4_T.md` 审文本证据流 T）已录的先例不重复；本份只审 README 第 8 节的 D。

读过：README 第 7、8 节；`src/hier_evidence_common.py`（`text_centre`、`text_logit_seconds`、`scaffold_rows_interval` 的 `decomposed` 分支、`decomposed_logodds`、`make_masked_scaffold_fn` 的 `evidence == "decomp"` 分支）；`src/interval_evidence_hmm.py`（`any_hate_logodds`、`_log_all_zero`）；`experiments/20260910_online_query_within/model.py` 第 218–236 行（`av_log = av_log + prior_scale * ell / ELL_SCALE`）；`search.py` 第 45 行（`prior_scale` 在 0.5–8.0 对数空间由 Optuna 搜索）。

## 结论：PASS-with-phrasing（放行，带第 5 节措辞要求）；另有一条规则 3 边界必须由用户裁定（第 4 节）

代码核对：D 由三步组成。(1) `hmm.posterior(..., w_coarse=0)` 得细窗 + 转移先验的段级后验对数几率 ell_fine；(2) `any_hate_logodds` 用同一链、只喂粗块裁定、细窗全 MISSING，取 logit(1 − P(全 0 路径)) 作视频内常数 v；(3) `text_logit_seconds` 把冻结文本分类器的 P(HATE) 取 logit 减训练集中位数（`text_centre`，不读标签），ASR / OCR 取大，加到逐秒 ell 上后截断到 ±13.8。E_t 同时作骨干输入列（ell、P(s) = σ(E_t)）和输出先验项 α·E_t / 13.8；HMM 拟合不含文本；粗块裁定的逐秒信息只通过 v 与 P(h_j) 列进入。README 描述与代码一致。

四类逐条：(a) 检索到的 hateful video 工作没有一篇做"视频级项 + 视频内项"的分数分解（第 3 节）；(b) 不是纯 ensemble：骨干在含 E_t 的输入上训练，α 与骨干联合决定输出，但 x_t 部分与规则 3 有边界问题（第 4 节）；(c) 不是后处理：改的是训练输入与先验项，训练前即生效；(d) 三步各自都小（一个 w_coarse 开关、一个常数项、一列外部分数），单独任何一步都属"只换特征 / 只改配置"，**放行的依据是分解原则本身**："粗粒度证据只在视频级起作用，细粒度证据逐秒起作用"是一个可证伪的结构主张，臂 `evidence=hmm`（粗块逐秒）与 `no_text_term` 已能证伪两步；建议补 `no_video_term`（v = 0）臂，否则 v 对 pooled 的贡献无法与 ell_fine 分开。

## 1. 检索记录（2026-09-10，WebSearch / WebFetch）

| 查询意图 | 命中（与判定相关） |
|---|---|
| WSVAD 视频级概率调制 snippet 分数 | **MSL（Li, Liu, Wu, AAAI 2022）**：Transformer 同时学视频级异常概率与 snippet 级分数，推理期"用视频级异常概率抑制 snippet 分数的波动"——与 v 的作用最接近（视频级项调制视频内分数）。Sensors 2023（Sharif et al.）综述句子核对来源为此文。 |
| WSVAD 粗→细两阶段 | TCVADS（arXiv 2412.20201，2024–25）：先视频级二分类，判异常才触发细粒度阶段（硬门）。 |
| 多粒度 VLM 裁定融合 | **VADTree（NeurIPS 2025，arXiv 2510.22693）**：训练自由，VLM 在层次树的粗节点与细节点分别打分，按子节点分数方差加权做父 / 子分数的凸组合 ā = ½(1−βŵ)â_parent + ½(1+βŵ)â_child；无视频级项、无生成模型。 |
| TAL 视频级类别分数 × CAS | 标准推理流程（CoLA CVPR 2021、W-TALC ECCV 2018、UntrimmedNet CVPR 2017 等）：先按视频级类别分数选类，再对该类 T-CAS 阈值出提议；Lee et al. 2020（不确定性建模）明写"段级后验 = 段级 softmax × 动作概率"。视频级分数是硬门或乘子，不是对数几率加项。 |
| 视频级 = 至少一段为正 | Noisy-OR MIL：Maron & Lozano-Pérez NeurIPS 1998（框架）、Viola, Platt, Zhang NeurIPS 2005（MILBoost，noisy-OR 包概率）；Wang et al. arXiv 1804.01146（max 与 noisy-OR 池化比较）。序列模型侧：keyword / filler HMM（关键词路径 vs. 背景路径似然比）；Shirahama, Grzegorzek, Uehara, IJMIR 2015（HCRF，视频级事件存在 = 对隐帧状态边际化）。 |
| 分类器对数几率相加 | Kittler, Hatef, Duin, Matas, TPAMI 1998（sum / product rule）；Hinton 2002 product of experts（概率相乘 = logit 相加）；logarithmic opinion pool。x_t 的"减中位数后相加"是固定权重的对数意见池，减中位数是先验偏置校正（同 Saerens et al. 2002 先验修正 / Menon et al. ICLR 2021 logit adjustment 的一类）。 |
| hateful video 基线 | HateMM ICWSM 2023、HVGuard EMNLP 2025、MM-HSD ACM MM 2025：视频级，无定位。MultiHateLoc WWW 2026（抓取核对）：逐帧 σ(W·F)，MA-MIL 只选 top-K 帧算损失，推理无视频级门、无文本分类器分数。HateClipSeg ACM MM 2025：ActionFormer 单模态 + 晚期融合。LELA（2602.09637，抓取核对）：五模态逐帧 LLM 打分取 max，"video context"只是另一条逐帧字幕流，无视频级项。VERA CVPR 2025（抓取核对）：段级 VLM 分数经相似段加权、高斯平滑、以视频中点为中心的位置权重；无视频级项。DSANet AAAI 2026：粗粒度正常原型重建 + 细粒度对比对齐，"粗 / 细"指特征层次，不是分数分解。CMHKF ACL 2025：视频-文本-音频特征融合，无分解。TANDEM（2601.11178）：RL 输出时间戳。 |

## 2. 问题 (b)：最近先例与 D 的差别

1. **视频级项调制视频内分数**：MSL AAAI 2022（视频级异常概率乘 snippet 分数）、TAL 视频级类别门（硬门）、TCVADS（硬门）。D 的差别：v 不是乘子或门而是对数几率加项，且 v 来自生成模型的"全 0 路径"概率，不是另一个判别头；D 的视频级项由粗块 VLM 裁定单独决定，而 MSL / TAL 的视频级分数与 snippet 分数来自同一网络的同一特征。
2. **多分辨率证据进不同层次**：Dugong NeurIPS 2019（上轮已录）把多分辨率弱源放进同一生成模型逐帧融合——这正是第 1 轮融合 ell 的做法；VADTree NeurIPS 2025 把粗 / 细 VLM 裁定按子节点方差加权凸组合。D 与二者的差别：粗裁定被**移出**逐秒发射、只保留视频级信息，理由是粗块在 HCS 上逐秒反向（`within_oracle` 数据）。这是 D 唯一可写成"设计洞察"的地方，且只能以两语料的诊断数据支撑，不能写成普遍原则。
3. **P(至少一段为正) 作视频级分数**：noisy-OR MIL 是定义，keyword / filler HMM 与 HCRF（Shirahama 2015）是序列模型上的同一量。D 的差别：该量由已拟合的区间 HMM 在部分观测（只有粗块裁定）下精确算出，与视频内后验共享参数；不能写"提出用 P(any) 作视频级分数"。
4. **文本分类器 logit 相加**：Kittler 1998 sum rule / PoE。这是已知做法，单独不构成任何 claim；本仓库 T 轮复核已裁定文本分数来源属实现细节（规则 5）。

## 3. 问题 (c)：基线是否做视频级 + 局部分解

没有。九个基线中做定位的（MultiHateLoc、HateClipSeg-ActionFormer、LELA、VERA、DSANet、CMHKF）都只输出逐帧 / 段级分数；MultiHateLoc 的 MIL 只在损失里用视频标签，推理不带视频级项；LELA 的"video"模态是逐帧字幕；VERA 的位置权重是时间先验不是视频级证据。视频级分类基线（HateMM、HVGuard、MM-HSD）不做定位。(a) 类阻断不成立。

## 4. 规则 3 边界（不属规则 4，须用户裁定后 D 才能作论文 claim）

`model.py` 第 233 行：最终逐秒分数 = 骨干 content logit + α·E_t / 13.8，E_t 含 x_t，即**冻结文本仇恨分类器的 centred logit 以搜索得到的权重 α 直接加进最终输出**。T 轮复核已把"第二个冻结模型的预测分数"提交用户裁定（README 第 7 节末），当时理由是 HMM 用发射表建模、不与 VLM 平均；D 去掉了发射表，x_t 以固定权重直接相加，这个理由不再成立。字面上规则 3"inference 阶段组合多个独立模型的 prediction"可被读作命中。审稿人判断：x_t 同时是骨干输入（列 ell、P(s)、COL_TEXT），骨干在其上训练，输出项与输入项共用 α，不是事后平均两个模型——按规则 4(b)"纯 ensemble"不阻断；按规则 3 由用户定。建议无论裁定如何，补一个 `text_prior_off` 臂（x_t 只进输入列、不进 α·E_t 输出项），以便论文能报告"文本分数不直接加到输出"的版本。

## 5. 措辞要求（写作前必须遵守，不阻断开跑）

1. D 命名为 "evidence decomposition by granularity"：逐秒对数几率 = 视频内项（细窗裁定 + 转移先验的 HMM 后验）+ 逐秒文本项 + 视频级项（粗块裁定下"至少一段为仇恨"的对数几率）。引 MSL AAAI 2022（视频级概率调制 snippet 分数）、TAL 视频级门（CoLA / W-TALC / UntrimmedNet）、TCVADS；明写差别：对数几率加项、视频级项来自生成模型的全 0 路径、粗 / 细证据来源分离。
2. 与 Dugong、VADTree 对照时写"coarse verdicts are removed from the per-second emission and retained only as video-level evidence"，依据是两语料诊断（粗块逐秒 within HCS .47）；不写成普遍原则。
3. v 写 "logit P(at least one hateful segment | coarse verdicts) from the same interval HMM (noisy-OR / keyword-filler likelihood ratio in sequence form)"；引 Viola 2005、Shirahama 2015；不写"提出"。
4. x_t 写 "a centred log-odds of a frozen text hate classifier added as a per-second term (sum-rule fusion, Kittler et al. 1998)"，作实现细节；不写"文本证据流"为贡献；规则 3 裁定结果写进 README。
5. 不写"首个层次化分解 hateful video 定位分数"（MSL、TAL 已有视频级 × 局部）；只能写"首个把 VLM 粗 / 细裁定按粒度分配到视频级 / 逐秒两层，并由同一生成模型给出两层的量"。
6. 消融必须齐：`evidence=hmm`、`no_text_term`、`no_video_term`（建议新增）、`text_prior_off`（建议新增）；缺 `no_video_term` 时不得 claim 视频级项有效。

## 6. 书目补录（并入 `docs/20260908_adaptive_query_survey.md` 补录表）

| 短名 | 题目 | 第一作者 | 出处 | 标识 | 核实 |
|---|---|---|---|---|---|
| MSL | Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection | Shuo Li | AAAI 2022 | ojs.aaai.org/index.php/AAAI/article/view/20028 | 已核实（视频级概率抑制 snippet 波动） |
| VADTree | VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity-Aware Tree | — | NeurIPS 2025 | arXiv 2510.22693 | 已核实（html 抓取，父子节点凸组合公式） |
| TCVADS | Injecting Explainability and Lightweight Design into Weakly Supervised Video Anomaly Detection Systems | — | arXiv 2024-12（v2 2025-09） | arXiv 2412.20201 | 已核实存在，只读摘要 |
| CoLA | CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning | Can Zhang | CVPR 2021 | arXiv 2103.16392 | 已核实 |
| W-TALC | W-TALC: Weakly-supervised Temporal Activity Localization and Classification | Sujoy Paul | ECCV 2018 | arXiv 1807.10418 | 已核实 |
| UntrimmedNet | UntrimmedNets for Weakly Supervised Action Recognition and Detection | Limin Wang | CVPR 2017 | arXiv 1703.03329 | 已核实（PDF 第 4 节） |
| Lee 2020 | Weakly-supervised Temporal Action Localization by Uncertainty Modeling | Pilhyeon Lee | AAAI 2021（arXiv 2020） | arXiv 2006.07006 | 已核实存在；"softmax × 动作概率"来自搜索摘要 |
| MILBoost | Multiple Instance Boosting for Object Detection | Paul Viola | NeurIPS 2005 | — | 已核实 |
| Maron 1998 | A Framework for Multiple-Instance Learning | Oded Maron | NeurIPS 1998 | — | 已核实 |
| Shirahama 2015 | Weakly Supervised Detection of Video Events Using Hidden Conditional Random Fields | Kimiaki Shirahama | Int. J. Multimedia Information Retrieval 2015 | doi 10.1007/s13735-014-0068-6 | 已核实 |
| Kittler 1998 | On Combining Classifiers | Josef Kittler | IEEE TPAMI 20(3) 1998 | — | 已核实 |
| PoE | Training Products of Experts by Minimizing Contrastive Divergence | Geoffrey Hinton | Neural Computation 2002 | — | 经典，未再核 |
| Logit adjustment | Long-tail Learning via Logit Adjustment | Aditya Menon | ICLR 2021 | arXiv 2007.07314 | 凭记忆，未核 |
| Saerens 2002 | Adjusting the Outputs of a Classifier to New a Priori Probabilities | Marco Saerens | Neural Computation 2002 | — | 凭记忆，未核 |
| VERA | VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models | Muchao Ye | CVPR 2025 | arXiv 2412.01095 | 已核实（html 抓取，分数细化公式） |
| DSANet | Learning to Tell Apart: WSVAD via Disentangled Semantic Alignment | — | AAAI 2026 | arXiv 2511.10334 | 已核实 |
| CMHKF | CMHKF: Cross-Modality Heterogeneous Knowledge Fusion for WSVAD | Guohua Wang | ACL 2025 | aclanthology 2025.acl-long.1524 | 已核实 |
| MultiHateLoc | MultiHateLoc: Towards Temporal Localisation of Multimodal Hate Content in Online Videos | — | WWW 2026 | arXiv 2512.10408 | 已核实（html 抓取，推理无视频级门） |
| Sharif 2023 | CNN-ViT Supported Weakly-Supervised Video Segment Level Anomaly Detection | Md Haidar Sharif | Sensors 2023 | PMC10537718 | 已核实（仅作 MSL 的转引来源） |
