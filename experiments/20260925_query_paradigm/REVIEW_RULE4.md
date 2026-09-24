# 规则 4 proposal review — 20260925_query_paradigm（Query-Tree Localization）

**判定：GO。** 四种 STOP 情形均不成立（来源方法未在 hateful video detection / localization 中使用；不是 train/test ensemble；不是纯 calibration / 后处理 / 平滑；不是纯工程技巧）。

- 审稿人：独立 agent（Fable 5.1），2026-09-25，一次性。
- 审阅对象：`README.md` 第 2.1–2.5、3 节；代码 `qtree.py`、`policy.py`、`model.py`、`data.py`、`train.py`（用于确认方法实际做了什么）。
- 依据：`RESEARCH_ITERATION_RULES.md` 第 4 条；第 3 条（ensemble 判定）。未运行任何训练或 GPU 任务。

## 1. 结论摘要

1. 五个来源思想（带噪二分 / 带噪 20 问；noisy-OR / 分组检测贝叶斯解码；期望信息增益选问题；带噪标注者 + 神经先验的潜在真值模型；零膨胀模型）均未在 hateful video detection 或 localization 文献中出现。hateful video 侧现有的 VLM/MLLM 用法是：逐帧或逐 clip 打分（LELA、CLARA、SafeLens）、CoT 推理做视频级分类（HVGuard、RAMF、MARS、IARE）、把 VLM 问答当特征喂给分类器（MemeScouts，meme）、RL 联合训练 VLM+ALM 输出时间戳（TANDEM）。没有一篇做自适应提问、树上精确后验、EIG 选问题或学习 VLM 答案噪声模型。
2. 组合方法（对时间轴二分树节点向冻结 VLM 提问、五类答案经零膨胀 OR 树精确后验融合、内容先验由答案的精确边际似然训练、EIG 选问题并按价格停止）作为整体在任何任务上都没有找到相同先例。最近的结构先例是 VAD 的 VADTree（时间二分树 + 每节点 VLM 打分，但训练无关、全节点查询、启发式融合）和 temporal grounding 的 FV-Action（coarse-to-fine yes/no 扫描，固定预算、排序融合）。
3. 规则 3：只用一个冻结 VLM（Qwen2.5-VL-7B）作为观测源，骨干输入 I3D/VGGish/BERT 与 MACIL-SD 基线及上游 it5 相同；最终分数是单一后验，没有组合多个独立模型的 prediction / posterior。不构成 ensemble。停止阈值 c 是预算旋钮而非分数变换（见第 5 节第 2 条的非阻断提醒）。

## 2. 检索记录

工具：WebSearch（34 次）、WebFetch（20 次，读摘要或全文）。检索日期 2026-09-25。

### 2.1 hateful video 侧（判定 STOP 情形 1 的直接依据）

| # | 查询 | 结果 |
|---|---|---|
| 1 | hateful video localization VLM query information gain adaptive questioning | CLARA、LELA、ReQuest/A.I.R.（通用 VideoQA 帧选择）。无自适应提问的 hate 定位工作。 |
| 2 | hateful video detection noisy-OR Bayesian generative model temporal segments | 只有 label-noise 分析（2508.04900）、TANDEM、MM-HSD；无 noisy-OR / 生成模型。 |
| 3 | hateful video detection bisection binary search timeline temporal localization multimodal LLM | MultiHateLoc、LELA；检索结果明确无 bisection / binary search。 |
| 4 | hateful video Dawid-Skene noisy annotator latent truth neural prior weak supervision | 只返回通用 crowdsourcing 文献（Dawid–Skene、Raykar、CROWDLAB）；无 hateful video 应用。 |
| 5 | HateMM HateClipSeg MultiHateLoc temporal localization hateful video methods 2025 2026 | HateClipSeg（ActionFormer / LSTR 基线）、MultiHateLoc（MIL）、LELA。 |
| 6 | hateful meme detection Bayesian active querying information gain VLM questions multimodal hate | MemeScouts（VLM 问答作特征 → Random Forest）、FBHM、IntMeme；无 Bayesian active querying。 |
| 7 | hateful video MLLM coarse-to-fine hierarchical segment querying budget calls HVGuard RAMF moderation | HVGuard（EMNLP 2025）、RAMF、CLARA、"Distributed Implicit Harm"；无分层/预算式提问。 |
| 8 | zero-inflated model video anomaly detection or hateful content localization | 无零膨胀模型；返回 LELA、AnyAnomaly 等。 |
| 9 | hateful video moderation agent iterative tool calls MLLM segment inspection adaptive budget 2026 | SafeLens（AAAI-26 demo）、WWW'26 companion agentic framework、UNIVID、VideoSeek（通用）。 |
| 10 | "hateful video" OR "hate video" "information gain" OR "mutual information" query selection VLM | 无命中（只有 RAMF、CLARA、MultiHateClip、VIBE 的 PMI 摘要选择）。 |
| 11 | Bayesian fusion MLLM judgments segment posterior hateful video video-level label prior network marginal likelihood | 无命中；返回 RAMF、label-noise 分析、HateClipSeg。 |
| 12 | MultiHateLoc method modality-aware MIL HateMM MultiHateClip frame-level AUC AP results VLM baseline | MultiHateLoc 数字（HateMM mAP .645 / AUC .799）；无 VLM 查询基线。 |
| 13 | "hateful video" OR "hate speech video" "question tree" OR "decision tree" OR "hierarchical questions" VLM prompting segments | MemeScouts、IPS（短视频审核的 in-prompt process supervision）、CLARA；无问题树。 |
| 14 | "Interpretable Agentic Framework for Multimodal Hate Video Analysis" evidence attribution arxiv | 只找到 ACM DL 页（全文 403）；连带 MARS（2601.15115）、IARE（2606.11953）。 |
| 15 | "zero-inflated" hate speech OR toxicity OR harmful content detection model | 全部是 zero-shot，无 zero-inflated。 |

全文/摘要核对（WebFetch）：CLARA 2608.15905；LELA 2602.09637；TANDEM 2601.11178；HateClipSeg 2508.01712v1（基线与指标）；MemeScouts 2604.24179（聚合方式）；SafeLens AAAI 42390；"Now You See the Hate" 2607.19061；MARS 2601.15115；IARE 2606.11953；UNIVID 2606.05748；WWW'26 agentic（403，仅摘要索引）。

### 2.2 来源思想在其他任务的先例（判定 STOP 情形 1 的补充，并给出最近先例）

| # | 查询 | 结果 |
|---|---|---|
| 16 | video anomaly detection VLM querying expected information gain active frame selection | Learning to Watch（ICML 2026，RL 主动获取证据，"expected information value" 作 reward proxy）、VERA、LAVAD、AnyAnomaly。 |
| 17 | weakly supervised video anomaly detection noisy-OR multiple instance learning Bayesian tree exact inference | UMIL、Bayesian nonparametric submodular partition；无 noisy-OR 树。 |
| 18 | "twenty questions" OR "noisy binary search" video temporal grounding vision-language model query oracle | FV-Action "Your VLM Already Knows When"（2608.08315）、T*、ReVisionLLM。 |
| 19 | tree search long video LLM temporal localization hierarchical coarse-to-fine VideoTree T* | VideoTree（CVPR 2025）。 |
| 20 | group testing temporal event localization video segments Bayesian decoding noisy tests | Cuturi 等 "Noisy Adaptive Group Testing using Bayesian Sequential Experimental Design"（2004.12508）；视频侧只有 Bayesian order priors。 |
| 21 | expected information gain question selection vision-language model active perception Bayesian posterior video event localization | FOVEA / S-BOED（2605.01345，ICML 2026，图像 crop）、Video Active Perception（2605.01662）、Active Video Perception（2512.05774）。 |
| 22 | noisy-OR multiple instance learning weakly supervised temporal action localization video-level label generative model | 只有 MIL/伪标签 WTAL 文献；无 noisy-OR 生成模型。 |
| 23 | VLM pseudo-labels as noisy annotators confusion matrix latent truth weakly supervised video anomaly detection Dawid-Skene | TFPLG、Parser-Free VLM Verification（2609.07455）；无 confusion-matrix / Dawid–Skene。 |
| 24 | zero-inflated Bernoulli multiple instance learning bag label "at least one" positive instance likelihood | 标准 MIL Bernoulli 假设（Ilse 等）、ProMIL；无零膨胀 MIL。 |
| 25 | probabilistic bisection noisy oracle video event change point localization LLM agent | Waeber–Frazier–Henderson 原文、generalized PBA；无视频应用。 |
| 26 | LLM judge as noisy annotator learned confusion matrix latent true label joint training neural network Raykar 2025 2026 | "Calibrate, Don't Curate"（2605.09702）、REALM、Tanno 等 CVPR 2019；均非视频、非分层。 |
| 27 | hierarchical OR tree binary partition timeline Bayesian sum-product exact inference multi-scale weak labels temporal localization | 通用统计文献（BayesBreak、Bayesian context trees、tree-structured change-point）；无视频弱监督应用。 |
| 28 | group testing pooled queries LLM oracle noisy tests identify positive items information gain | 经典 noisy group testing；另有 group testing 用于图像审核（2305.07639，非自适应 CS 解码）。 |
| 29 | budgeted MLLM calls per video adaptive number of queries content moderation cost accuracy curve early stopping | AdaptToken、COEF-VQ cascade、Budget-Aware Tool Use；无每视频自适应 VLM 调用数的定位方法。 |
| 30 | expected information gain twenty questions large language model uncertainty of thoughts Bayesian question selection | Uncertainty of Thoughts（2402.03271）、Learning to Ask Informative Questions（2406.17453）。 |
| 31 | interval yes/no oracle queries temporal action localization active learning coarse interval labels posterior | Boundary-centric active learning（2604.15173）；无区间 yes/no 神谕 + 后验。 |
| 32 | video anomaly detection coarse-to-fine zoom-in VLM queries hierarchical temporal windows localization training-free | **VADTree**（2510.22693）、GtS（2608.11260）、VERA、LATERN。 |
| 33 | noisy-or pooling hierarchical multiple instance learning multi-scale segments video weak label likelihood tree | MIL pooling 比较（1810.09050）、EM-MIL；无层级 noisy-OR 树。 |
| 34 | latent variable model EM learning from VLM segment verdicts weakly supervised video localization observation model learned reliability | EM-MIL（ECCV 2020）；无 VLM 观测模型。 |
| 35 | Bayesian search event in video sequence noisy detector queries optimal interval selection posterior information theoretic | 经典 noisy binary search 理论；无视频 + VLM 应用。 |

全文/摘要核对（WebFetch）：VADTree 2510.22693（全文）；FV-Action 2608.08315（全文）；ReVisionLLM 2411.14901；Learning to Watch 2607.00622；FOVEA 2605.01345；Active Video Perception 2512.05774；GtS 2608.11260；Parser-Free VLM Verification 2609.07455；group testing 图像审核 2305.07639。

## 3. 最近先例（含链接）与为何不触发 STOP

### 3.1 hateful video / multimodal hate 内

| 工作 | 与本提案的关系 | 为何不是同一来源 |
|---|---|---|
| [MultiHateLoc](https://arxiv.org/abs/2512.10408) | 弱监督 hate 定位，HateMM/MHC | modality-aware MIL；不查询 VLM。 |
| [LELA: Training-free Multimodal Hate Localisation with LLMs](https://arxiv.org/abs/2602.09637) | LLM 做 hate 定位 | 逐帧多阶段 prompting 打分，训练无关；无自适应提问、无贝叶斯融合、无学习部件。 |
| [CLARA](https://arxiv.org/abs/2608.15905) | VLM 输出进 hateful video 模型 | 每个 clip 一条 VLM rationale 经 gated Transformer 进编码器；检测任务；非自适应、非贝叶斯。 |
| [HVGuard](https://aclanthology.org/2025.emnlp-main.456/)、[RAMF](https://arxiv.org/abs/2512.02743)、[MARS](https://arxiv.org/abs/2601.15115)、[IARE](https://arxiv.org/abs/2606.11953) | MLLM 推理做 hateful video 检测 | 视频级 CoT / 对抗推理；不定位、不选问题、不建答案噪声模型。 |
| [TANDEM](https://arxiv.org/abs/2601.11178) | 带时间戳的多模态 hate 检测 | VLM+ALM 联合 RL；无树、无后验、无 EIG。 |
| [SafeLens](https://ojs.aaai.org/index.php/AAAI/article/view/42390) | 段级 hate 审核系统（AAAI-26 demo） | 固定每段一次 policy LLM；无自适应、无贝叶斯。 |
| [Interpretable Agentic Framework for Multimodal Hate Video Analysis](https://dl.acm.org/doi/10.1145/3774905.3796488) | agentic hate video 分析（WWW'26 companion） | 摘要：确定性证据排序；全文 403 未读。摘要索引无自适应提问 / 树 / 后验。 |
| [HateClipSeg](https://arxiv.org/abs/2508.01712) | 本项目主数据集 | 定位基线 ActionFormer 晚融合、在线分类 LSTR；无 VLM 查询。 |
| [MemeScouts (LT-EDI@ACL 2026)](https://arxiv.org/abs/2604.24179) | VLM 问答做 hate 弱监督（meme） | 89 个问题答案作整型特征 → Random Forest；原文明确**不**学 label model；无时间维。 |
| [Now You See the Hate](https://arxiv.org/abs/2607.19061) | "adaptive view retrieval"，hate 图像 | CLIP 视图库 + 校准；图像、非贝叶斯、非视频。 |
| [Revealing Temporal Label Noise](https://arxiv.org/abs/2508.04900) | hateful video 时间标签噪声 | 按时间戳裁剪做分析；不是 VLM 答案噪声模型。 |

### 3.2 其他任务中机制最接近的先例

| 工作 | 相同点 | 不同点 |
|---|---|---|
| [VADTree](https://arxiv.org/abs/2510.22693)（VAD，训练无关） | 时间轴二叉层级树，每节点由 VLM 描述 + LLM 打分，节点分数传到帧 | 树由 GEBD 边界递归切分；**全部节点**都查询（无选择、无预算）；融合是余弦相似 + 方差加权启发式，不是概率模型；无学习先验、无答案噪声模型、无 EIG。 |
| [FV-Action: Your VLM Already Knows When](https://arxiv.org/abs/2608.08315)（temporal grounding） | 用 yes/no 提问代替时间戳回归，coarse-to-fine 扫描 | 两层均匀网格（K_c + N_p·K_f 次固定调用），几何平均排序融合；原文明确拒绝每视频自适应预算；无训练。 |
| [ReVisionLLM](https://arxiv.org/abs/2411.14901)（CVPR 2025） | 由粗到细递归缩小区间 | 训练的 VLM 自己递归；无贝叶斯后验。 |
| [Learning to Watch](https://arxiv.org/abs/2607.00622)（ICML 2026，VAU） | 主动获取时间证据，信息价值驱动 | RL（iDPO）策略 + 时间原子操作；信息价值是 reward proxy，非精确 EIG；无树、无生成模型。 |
| [Active Video Perception](https://arxiv.org/abs/2512.05774)、[Video Active Perception](https://arxiv.org/abs/2605.01662) | 迭代选看哪里 | agent 规划/反思或生成模型"意外度"；无精确后验。 |
| [FOVEA / S-BOED for VLMs](https://arxiv.org/abs/2605.01345)（ICML 2026） | 把 VLM 主动感知写成 sequential BOED，EIG 代理 | 图像 crop，非视频、非时间定位。 |
| [Noisy Adaptive Group Testing using Bayesian Sequential Experimental Design](https://arxiv.org/abs/2004.12508) | noisy-OR 测试 + 后验 + 信息增益选下一组 | SMC 近似后验，检测个体感染；无视频、无神经先验。 |
| [Group testing for image moderation](https://arxiv.org/abs/2305.07639) | 分组检测用于内容审核 | 图像池 + 压缩感知解码，非自适应。 |
| [Uncertainty of Thoughts](https://arxiv.org/abs/2402.03271)、[Learning to Ask Informative Questions](https://arxiv.org/abs/2406.17453) | EIG 选问题（LLM 20 问） | 文本对话。 |
| [Parser-Free VLM Verification for Federated WSVAD](https://arxiv.org/abs/2609.07455)（AVSS 2026） | 冻结 VLM yes/no logit 验证 MIL 片段 | 级联后验证，无噪声模型、无贝叶斯融合、无预算。 |
| [VERA](https://arxiv.org/abs/2412.01095)（CVPR 2025） | 冻结 VLM + 学到的引导问题做 VAD | 问题文本由 verbalized learning 得到，每窗固定问；已是本项目基线。 |
| [Tanno et al. CVPR 2019](https://arxiv.org/abs/1902.03680)、[Calibrate, Don't Curate](https://arxiv.org/abs/2605.09702) | 神经分类器 + 学习的标注者/LLM judge 混淆矩阵（Raykar 一脉） | 非视频、非层级、非按区间查询。 |
| [EM-MIL](https://arxiv.org/abs/2004.00163)（ECCV 2020） | 隐变量 + 似然训练的弱监督定位 | 无外部神谕观测。 |

结论：来源思想只在 hateful video 之外出现；组合方法（树 + 精确零膨胀 OR 后验 + 由答案边际似然训练的内容先验 + 精确 EIG 与价格停止）没有完整先例。写论文时 VADTree、FV-Action、Learning to Watch、Cuturi 等分组检测 BOED 应作为最近工作对照说明差异。

## 4. 四种 STOP 情形逐条核对

| 情形 | 判定 | 依据 |
|---|---|---|
| 来源方法已用于 hateful video detection / localization | 否 | 第 2.1 节 15 条检索 + 11 篇核对，无一使用二分 / 20 问、noisy-OR 树、EIG 选问题、带噪标注者潜在真值模型或零膨胀模型。 |
| 纯 training / test ensemble | 否 | 一个冻结 VLM（观测源，规则 3 明示允许）+ 一个训练网络（先验）+ 一个生成模型给出的单一后验；`model.py` 的 PriorNet 只吃 I3D/VGGish/BERT 特征；无多模型 prediction / posterior 组合，无多 teacher。骨干输入集合与 MACIL-SD 基线、上游 it5 相同。 |
| 纯 calibration / 后处理 / 平滑 | 否 | 方法主体是生成模型、训练目标、提问策略与骨干；无分数平滑、无按语料路由。停止阈值 c 是"问几次"的预算旋钮，不变换分数（见第 5 节第 2 条）。 |
| 纯工程技巧 | 否 | 新的生成模型（零膨胀 OR 树 + 三状态、含长度项的答案模型）、新训练目标（答案边际似然，`TreeBatch.log_evidence`）、新推理流程（EIG 策略）；不是只调超参 / 换特征 / 加增广。 |

未以"可识别性 / 退化 / shortcut"类实现前推理作为阻断依据。

## 5. 非阻断问题（≤5，供实现与 code review 参考）

1. **训练目标与 README 2.3 的"精确边际似然"表述不一致（归一化）。** `train.py` 第 168 行把答案项除以该视频已答节点数 `n_obs`（`loss_ans = -(where(label, lp1, lp0) / n_obs).mean()`），所以实际优化的是"每答案平均对数似然 + 视频标签 BCE"，长视频（节点数 ≈ T/2）的答案权重被压到与短视频相同，且答案项与 BCE 之间隐含 1:1 权重。这是合法设计，但 README 应写明这一归一化，否则"精确边际似然训练"的 claim 与代码不符；若不想引入隐含权重，就去掉 `/ n_obs`。

2. **门数字要预先钉死：README 第 5 节的 "B = 8" 没有说是固定 8 次还是自适应均值 8 次。** `train.py` 的 `metrics.json` 只写自适应规则（`primary_budget` 的 `adaptive` 结果），而自适应阈值 c 是在 validation 上标定后用于 test 的推理期标量（`policy.calibrate`），且同一 validation 已用于选 checkpoint。建议 README 明确规则 8 的过门数字用哪一个（建议固定 B=8 为门、自适应为曲线），并在论文里把 c 表述为部署预算而非 calibration，以免撞规则 3 的"禁止推理期 calibration"字面。

3. **消融 (b) 不能单独隔离 OR 树融合。** README 写"每秒取覆盖它的已问节点答案的平均"，这个臂里内容先验 π_t 完全不进分数，因此对照差异 = "融合规则" + "是否用先验" 两个因素之和。要隔离融合，需规定 π_t 如何进入平均臂（例如把先验当一个额外观测一起平均，或只在无答案覆盖的秒用先验）。消融 (d) BFS 同理：自适应停止按臂各自标定 c 只保证均值相同，应同时报固定 B 的版本（代码已算 `fixed_budgets`，只需写进消融表）。

4. **训练期与测试期观测集合不同，答案模型的条件独立假设在训练期被强烈违反。** 训练用一个视频**全部** ≈T/2 个嵌套节点的答案（`train_obs` 来自所有缓存节点），这些节点共享帧与转录，答案高度相关；测试期只观测 ≤8–32 个 EIG 选出的节点。学到的 θ/ω 在训练期会被重复计数的证据推得过于自信，用到测试的稀疏观测上时后验尖锐度可能失真。不阻断（规则 4），但建议在 validation 上按测试预算（8 次）报告答案模型的对数似然 / 校准，作为解释结果用的诊断。

5. **EIG 目标 I(o_n; (G, y)) 中视频级项占主导时，预算可能全花在长节点上。** 任一节点的答案都通过状态 0 与状态 1/2 表的差异强烈区分 G，在 p_G 不确定的视频里 EIG 最大的往往是根附近的长节点，8 次调用可能都用于判定"视频是否有害"，视频内定位仍主要靠先验 π。`evaluate` 已保存 `res["test_runs"][v]["asked"]`，建议在 STATUS 里按预算统计被问节点的长度分布与深度分布，用数据确认或否定这一点，而不是事先改策略。

正确性核对（无问题，记录以便 code review 少走一步）：`TreeBatch.log_evidence` 与 `VideoTree.infer` 的上行消息（lP1 = a₁ + logaddexp(luL, luR + lvL) 即 1 − (1−uL)(1−uR)）、"至少一秒为 1"的归一化（lz_root + lu_root − log(1 − Π(1−π_t))）、下行外消息（子 = 0 需父 = 0 且兄 = 0，或父 = 1 且兄 = 1；子 = 1 需父 = 1）、P(G=1|o) 的混合、EIG = H(o|已问) − Σ_s P(s_n) H(o|s) 等价于 I(o_n; (G, y) | 已问)（o_n ⟂ (G, y) | s_n），均与 README 2.2、2.4 一致。`model.py` 的 PriorNet 前向只接收 f_a、f_v，"骨干不读任何 VLM 答案"成立；`init_theta`、长度标准化统计只用训练集节点；test 答案缓存只在策略选中节点时被读取，解析失败计为一次调用。
