# 规则 4 复核：单次训练在线查询 + 视频内监督（2026-09-10）

审稿人：独立 agent（Claude Fable 5.1），只读，不跑训练；依据 `RESEARCH_ITERATION_RULES.md` 第 4 条。第 4 条原文只挡四类：(a) 来源方法已用于 hateful video detection / localization（审稿必须实际检索并记录）；(b) 纯 training/test ensemble；(c) 纯 calibration / 后处理 / 平滑；(d) 纯工程技巧而非完整科研方法（只调超参、只换特征、只加数据增强、只改训练配置）。明文禁止以"可识别性 / 可能退化 / 可能 shortcut"等实现前推理 STOP。本复核只回答"有没有先例、什么是新的、论文怎么说"，不评估预期性能。

读过：本目录 `README.md` 第 0–5 节、`train.py`（`train()` 全部：允许集合、`refit`、`sampler`、`window_targets`、获取事件、损失、评估）、`acquire.py`（`Acquirer.run_video` 的 eoc 分支、`window_bags`）、`src/hier_evidence_common.py`（`TrainDataset`、`block_bag_loss`、`window_bag_loss`）、`src/interval_evidence_hmm.py`（模块 docstring、`_emissions`、`_posterior_video`、`fit`、`infer` 的 regimes 部分）；上一轮 `experiments/20260908_adaptive_vlm_query/REVIEW_RULE4.md` 与 README 第 1、6 节；`docs/20260908_adaptive_query_survey.md`；修订 3 README 第 22–37 行（块级 MIL 来源）。

## 结论：PASS（放行），带四条措辞要求（第 6 节）；无须改代码

四个部件都不落在四个阻断类别里。A 与 C 各有清楚的文献先例族（训练集主动特征获取 + 期望模型输出变化；掩码标签预测 / 从众包学习），但把它们用在"冻结 VLM 的区间裁定"这种既是输入又是监督的观测上，且在弱监督逐秒定位里，没有先例，也没有任何 hateful video 工作做过。B 单独不是贡献，只能作 A 的一个设计选择（消融臂）。D 门未过、不进默认方法，只需要在诊断报告里按先例措辞。

## 1. 检索记录（2026-09-10，WebSearch / WebFetch）

查询与命中（只列与判定相关的）：

| 查询意图 | 关键词 | 命中 |
|---|---|---|
| hateful video + 本轮四部件 | hateful video localization HateMM HateClipSeg active query / training-time acquisition / masked window loss | HateClipSeg（ACM MM 2025，2508.01712）、MultiHateLoc（2512.10408）、CLARA（2608.15905）、LELA（2602.09637）——上轮已审；**新命中 TANDEM（arXiv 2601.11178v3，AAAI-ICWSM 2027）**：VLM + 音频语言模型"tandem RL"输出结构化推理与时间戳，HateMM 上做 target 识别；无自适应查询、无掩码裁定预测、无训练期获取、无可靠性建模（抓取其摘要页核对）。 |
| 训练期主动获取 | active feature acquisition during training / weakly supervised temporal localization VLM | 只命中普通 WTAL + VLM 工作（VLPO、Ju et al. CVPR 2023 蒸馏、STPrompt），无"训练期逐步揭示查询"。 |
| A 的准则先例 | Freytag Rodner Denzler expected model output changes | Freytag et al. ECCV 2014（EMOC）已核实；Käding et al. 2016 后续。 |
| A 的训练集获取先例 | Melville Saar-Tsechansky Provost active feature-value acquisition | Melville et al. ICDM 2004；Saar-Tsechansky, Melville, Provost, Management Science 2009 已核实。 |
| C 的结构先例 | masked label prediction UniMP | Shi et al. IJCAI 2021 已核实（标签作输入、随机遮蔽、预测被遮蔽标签）。 |
| C 的另一族 | Rodrigues & Pereira deep learning from crowds | AAAI 2018 已核实（crowd layer：网络额外预测每个标注者的带噪标签）。 |
| C 在 VAD 的近邻 | weakly supervised VAD VLM pseudo labels snippet-level supervision | TFPLG（IEEE 2025，训练-free CLIP 伪标签）、STPrompt（2408.05905）、pseudo-label 完备性/不确定性（ICME 2023）——VLM 分数当 snippet 伪标签，全量、不遮蔽、不作输入。 |
| D 的先例 | Dawid–Skene / HMM / per-item annotator regime | HMM-Crowd（Nguyen et al. ACL 2017）、BSC（Simpson & Gurevych TACL 2019）、CommunityBCC（Venanzi et al. WWW 2014）、GLAD（Whitehill et al. NeurIPS 2009）已核实。 |

没有任何 hateful video 工作做过 A–D 中任何一项。阻断类别 (a) 不成立。

## 2. 部件 A：单次训练在线获取

**机制（代码核对与 README 1.A 一致）**：`allowed[v]` 从空集起；epoch {5,10,15,20} 结束后 `Acquirer.run_split(train_ids, "eoc", 1, initial=allowed)` 对每个训练视频算一次 EOC、揭一个窗（`bf[w] = bf_true[w]` 只在选中后读）；`refit()` 用扩大后的观测重拟合 HMM 并换 `masked_fn`；`sampler` 在允许集合内做证据 dropout（一半随机子集、一半获取顺序前缀）。训练期每视频 4 粗 + 4 细 = 8 次，`calls["train_total_per_video"] = 4 + len(acq_epochs)` 与 README 一致。

**最近先例（按相似度）**：
1. **训练集主动特征值获取**：Melville, Saar-Tsechansky, Provost, Mooney, "Active feature-value acquisition for classifier induction", ICDM 2004；Saar-Tsechansky, Melville, Provost, "Active feature-value acquisition", Management Science 55(4) 2009。训练样本的特征值缺失、可付费获取，由当前分类器的期望效用决定先买哪个实例的哪个特征，迭代获取、迭代重训。这正是 A 的"问题定义"：获取对象是训练集里缺失的观测，不是标签。差别：他们的效用是期望分类性能提升（用当前模型估计），特征无结构；我们的效用是 EOC。
2. **期望模型输出变化准则**：Freytag, Rodner, Denzler, "Selecting influential examples: Active learning with expected model output changes", ECCV 2014（EMOC）；Käding et al. 2016 的大规模近似。"揭示这条观测后模型输出的期望变化"作为获取价值，就是 EMOC 的定义；上轮复核已指出它同时是 DIME（ICLR 2024）目标量的非摊销版本。A 把 EMOC 从"选哪个样本去标注"用到"选哪个训练视频的哪个窗去问 VLM"。
3. **推断期动态特征选择**（上轮已审）：Covert et al. ICML 2023；Gadgil et al. DIME ICLR 2024；Shim et al. NeurIPS 2018（获取与分类联合训练，单次训练，但训练集全观测、以 RL 环境模拟获取）。
4. **从带噪 oracle 主动学习**（因为 A 揭示的裁定又被 C 当监督目标）：Donmez & Carbonell, "Proactive learning", CIKM 2008；Yan, Chaudhuri, Javidi, "Active learning from imperfect labelers", NeurIPS 2016。差别：他们的 oracle 输出只作训练目标；我们的裁定同时是测试期输入。
5. 时序多模态主动获取 A2MT（Kossen et al. TMLR 2023，上轮补录）。

**相对文献的新点**：(i) 获取对象是训练视频内部的细窗裁定，既是骨干输入（测试期也要问）又是 C 的监督目标——文献里"主动特征值获取"（只当输入）与"主动学习"（只当标签）是两条线，A+C 把同一条被获取的观测同时用在两边；(ii) 缺失观测的预测分布与骨干输入都由区间 HMM 在每次事件后重拟合、精确重算（DFS 与 AFA 文献里训练集缺失靠掩码或代理模型）；(iii) 输出是弱监督逐秒密集分数，价值在时间轴上平均；(iv) 获取事件嵌在训练循环里成为"由学习者驱动的证据课程"，训练期与测试期预算完全对称（8 次）。

**相对本仓库上一轮的新点**：上轮是两轮（训练 M0 → 一次性选窗 → 重训 M1），骨干只在两轮之间决定一次；本轮单次训练、四次事件、每次事件后 HMM 重拟合与 scaffold 换新。这是训练流程结构的改变，不是超参调整。

**不能写的**："新的获取准则"（EMOC / DIME）；"首创训练集获取"（Saar-Tsechansky 2009）；"训练期调用如实计数是方法贡献"（上轮已裁：协议贡献）。另注意：训练期 34 条裁定本来就已缓存，"只揭 8 条"是协议而非成本节省本身，论文必须说清这是 simulated acquisition（`acquire.py` docstring 已如此写）。

**建议措辞**："training-time active acquisition of VLM window verdicts: an expected-model-output-change criterion (Freytag et al. 2014; DIME) selects, per training video and at fixed epochs, which cached verdict to reveal; the revealed set is the only fine evidence the backbone sees (evidence dropout inside it) and the only fine supervision the window loss uses, so train-time and test-time budgets coincide."

**阻断类别核对**：(a) 无 hateful video 先例；(b) 一个骨干、一个 HMM、一个冻结 VLM，反事实前向是同一模型两种输入；(c) 改变的是输入集合，不动输出分数；(d) 训练流程结构改变并带机制主张，`fixed_uniform_train` 臂可证伪。**PASS**。

## 3. 部件 B：问询权重来自骨干

**机制（代码核对）**：`Acquirer.weight == "model"` 时 `pred = window_bags(clog, window_rows, k, topk_div)` = σ(该窗行上五 crop 均值 content logit 的 top-⌈n_w/16⌉ 均值)，作 EOC 里 p(b_w = 1 | E)；`"hmm"` 时用 `infer()["pred_fine"]`（上轮做法）。与 C 的 bag 定义相同。

**先例**：用当前模型自己的预测分布对未知结果加权，是 Roy & McCallum ICML 2001（期望误差缩减）、EMOC（ECCV 2014）、Covert 2023 / DIME 的标准做法——这些方法本来就没有第二个模型给预测分布。上轮用 HMM 预测才是偏离常规的那一版；B 是回到常规。

**新点**：相对文献没有。相对本仓库：换了 p(b_w) 的来源，且这个 bag 被 C 训练成裁定预测器，所以 B 与 C 是配套的（同一个 bag，C 训、B 用）。

**必须写明的一点（不阻断）**：σ(top-k content logit) 只有在 C 开着时才是 p(b_w = 1) 的估计；`no_window_loss` 臂里这个量只是骨干的仇恨内容分数，不是裁定预测。若日后报 `no_window_loss` 与 `hmm_weight` 两臂，读表时要说明 `no_window_loss` 同时改变了 B 的权重语义（两处差异）。

**措辞**：B 不单独立贡献，写成 A 的设计选择："the predictive p(b_w = 1 | state) is the backbone's own window bag, trained by the window loss (C); the interval-HMM predictive (iteration 1) is an ablation."

**阻断类别核对**：单独看 B 接近 (d)"只改配置"，但它不是独立方法，是 A 的一个组件，随 A 放行。**PASS，要求不单独 claim**。

## 4. 部件 C：被遮蔽裁定的窗级损失

**机制（代码核对与 README 1.C 一致）**：`window_targets(vid, b_input)`：正例视频里 `hidden = allowed[v] ∩ {b_input == MISSING}`，目标 = 缓存裁定（`verdict`）或 `hmm.infer(b_input ∪ {b_w})["p_hf"][w]`（`posterior` 臂）；负例视频所有 30 窗目标 0（来自视频标签，与允许集合无关）。`window_bag_loss`：每个有目标的窗取 top-⌈n_w/16⌉ 均值 content logit 作 bag，BCE，先视频内均值再跨视频均值；权重 λ_block 与块级 MIL 共用。目标窗的裁定不在该样本输入里（`b_input[w] == MISSING`），HMM 列 ℓ_t、P(s_t)、P(h_j) 也是在遮蔽输入上算的；粗块裁定始终可见，测试期同样如此，不构成泄漏。

**最近先例**：
1. **掩码标签预测**：Shi et al., "Masked label prediction: Unified message passing model for semi-supervised classification", IJCAI 2021（UniMP）。标签作为输入嵌入送进网络，随机遮蔽一部分，训练时预测被遮蔽的标签，目的正是防止"标签直接从输入抄到输出"。C 的结构与之同构：裁定作输入（六格嵌入 + HMM 列），随机遮蔽（证据 dropout），预测被遮蔽的裁定。更早的同类：BERT 式 masked modeling；"label masked autoencoder"用于分割标签补全。
2. **从众包学习 / 预测标注者标签**：Rodrigues & Pereira, "Deep learning from crowds", AAAI 2018（crowd layer：网络额外输出每个标注者的带噪标签作辅助监督）；Raykar et al. JMLR 2010。VLM = 单个带噪标注者，C 让骨干预测这个标注者的窗级判断。
3. **VLM/CLIP 分数当 snippet 伪标签**（弱监督 VAD）：TFPLG（IEEE TCSVT? 2025，会议未核实）、STPrompt（arXiv 2408.05905）、Ju et al., "Distilling vision-language pre-training to collaborate with weakly-supervised temporal action localization", CVPR 2023；hateful video 侧 CLARA（2608.15905）把 VLM rationale 当 clip 级输入而非目标。这些都是"全量伪标签、不遮蔽、不作输入"。
4. **本仓库自身**：修订 3 的块级 MIL（`block_bag_loss`，目标 = HMM 粗块后验 P(h_j)，权重 |2P−1|）已是"用 VLM 派生的段级目标做 bag BCE"。C 是它的细窗版本，两处不同：目标是被遮蔽的原始裁定（`verdict`）而不是后验，且目标窗的观测不在输入里。

**新点**：相对文献——把掩码标签预测用在"一个冻结 VLM 的时间区间裁定"上，且遮蔽集合不是随机全集而是主动获取得到的允许集合（A 决定哪些窗有目标）；目标粒度（细窗）细于视频标签，提供视频内排序监督，这是 README 第 0 节第 4 点的动机。相对仓库——细窗粒度、遮蔽目标、原始裁定目标。三者都是设计决定，可被 `no_window_loss`、`window_target_posterior` 臂证伪。

**不能写的**："自监督"要谨慎：目标来自 VLM 裁定，是辅助带噪标签，不是从数据本身派生的 pretext；建议写 "masked modeling of auxiliary (VLM) labels" 或 "masked verdict prediction"，不写 self-supervised。也不能写"首创用 VLM 段级判断做监督"（VAD 伪标签一族、Ju et al. CVPR 2023、本仓库块级 MIL）。

**建议措辞**："masked verdict prediction: verdicts inside the acquired set are randomly hidden from the input and predicted by a window-level bag of the content logit (BCE toward the hidden verdict; all windows toward 0 on non-hateful videos), in the manner of masked label prediction (Shi et al. 2021) and annotator-label heads in learning from crowds (Rodrigues & Pereira 2018); this is the only within-video supervision the backbone receives."

**阻断类别核对**：(a) 无；(b) 无；(c) 训练损失，不是后处理；(d) 新损失项带机制主张，不是只改配置。**PASS**。

## 5. 部件 D：视频级可靠性混合 HMM（门 D1 未过，诊断臂）

**机制（代码核对）**：`regimes=R`：每视频隐变量 z_v，各档独立 (q_f, r_f, q_c, r_c)、混合权 π，转移率与 p0 共享；`_posterior_video` 对每档做一遍前向后向按 π_z·边际似然混合；`fit` 对正、负视频都算责任度 ρ，闭式加权计数 M 步；`infer()["pred_fine"]` 按 ρ 混合各档预测。R = 1 时 `q_f_z[0]` 等即原参数。与 README 1.D 一致。

**先例**：
1. **Dawid & Skene**, Applied Statistics 1979：标注者混淆矩阵 EM。
2. **按"类型"聚类的标注者可靠性**：Venanzi, Guiver, Kazai, Kohli, Shokouhi, "Community-based Bayesian aggregation models for crowdsourcing", WWW 2014（CommunityBCC：标注者属于少数几个社区，每社区一套混淆矩阵，混合先验）。D 的三档 = 三个"社区"，只是分档单位是视频而不是标注者（同一个 VLM 在不同视频上像不同的标注者）。
3. **按 item 变化的噪声**：Whitehill et al., GLAD, NeurIPS 2009（item 难度 × 标注者能力）；Li, Rubinstein, Cohn, "Exploiting worker correlation for label aggregation", ICML 2019。
4. **序列标注上的众包模型**：Nguyen et al., "Aggregating and predicting sequence labels from crowd annotations", ACL 2017（HMM-Crowd）；Simpson & Gurevych, "A Bayesian approach for sequence tagging with crowds", TACL 2019（BSC）——隐 Markov 序列 + 标注者混淆，是 D 在结构上最近的：D = 单标注者、混淆矩阵按 item（视频）混合、时间轴是连续时间二态链。

**新点**：把"标注者社区混合"移到"单个 VLM 标注者、按视频分档"，嵌入区间 HMM，EM 精确。**审稿人的文献层面说明（不是阻断，是解释 D1 失败的已知原因）**：Dawid–Skene 一族的可识别性来自每个 item 有多个标注者；单标注者时，item 级的可靠性档只能靠时间先验（转移率）与视频标签区分"整段仇恨"与"整段误报"。README 第 4 节的判断（需要视频级标签参与 z 的推断）与这一点一致；CommunityBCC / GLAD 都在多标注者设定下工作，不能直接引来支持单标注者可识别。

**措辞（若进论文诊断节）**："a video-level annotator-reliability mixture inside the interval HMM (Dawid–Skene / CommunityBCC-style regimes, one latent regime per video); it lowers predictive log-loss of held-out verdicts but hurts pooled localization because, with a single annotator, 'all-hateful' and 'over-triggering' videos are not separable without the video label — reported as a negative result."

**阻断类别核对**：不进默认方法，规则 4 对其只要求措辞。作为方法本身，四类都不命中（模型结构改变，非 ensemble、非后处理）。**PASS（诊断臂）**。

## 6. 措辞要求（论文写作前必须遵守，不阻断开跑）

1. A 的准则写 EMOC（Freytag et al. ECCV 2014）/ DIME 的非摊销版；问题定义引 Saar-Tsechansky et al. 2009；贡献落在"同一条被获取的观测同时是输入与监督、HMM 精确重算、弱监督逐秒输出、训练/测试预算对称"，不写"新准则"。
2. B 不单独 claim；写明 B 的预测器由 C 训练，`no_window_loss` 臂同时改变 B 的语义。
3. C 写 "masked verdict prediction / masked modeling of auxiliary VLM labels"，引 Shi et al. IJCAI 2021 与 Rodrigues & Pereira AAAI 2018，说明与本仓库块级 MIL 的关系（粒度、遮蔽目标、原始裁定）；不写 self-supervised。
4. D 只作负结果或诊断，引 Dawid–Skene、CommunityBCC、HMM-Crowd / BSC，明说单标注者不可识别是失败原因。

## 7. 书目补录（请并入 `docs/20260908_adaptive_query_survey.md` 补录表）

| 短名 | 题目 | 第一作者 | 出处 | 标识 | 核实 |
|---|---|---|---|---|---|
| Melville 2004 | Active Feature-Value Acquisition for Classifier Induction | Prem Melville | ICDM 2004 | cs.utexas.edu/~ml/papers/afa-tr-04.pdf | 已核实（搜索命中） |
| Saar-Tsechansky 2009 | Active Feature-Value Acquisition | Maytal Saar-Tsechansky | Management Science 55(4) 2009 | doi 10.1287/mnsc.1080.0952 | 已核实 |
| EMOC | Selecting Influential Examples: Active Learning with Expected Model Output Changes | Alexander Freytag | ECCV 2014 | doi 10.1007/978-3-319-10593-2_37 | 已核实 |
| UniMP | Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification | Yunsheng Shi | IJCAI 2021 | arXiv 2009.03509 | 已核实 |
| CrowdLayer | Deep Learning from Crowds | Filipe Rodrigues | AAAI 2018 | arXiv 1709.01779 | 已核实 |
| CommunityBCC | Community-Based Bayesian Aggregation Models for Crowdsourcing | Matteo Venanzi | WWW 2014 | doi 10.1145/2566486.2567989 | 已核实 |
| GLAD | Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise | Jacob Whitehill | NeurIPS 2009 | — | 已核实 |
| HMM-Crowd | Aggregating and Predicting Sequence Labels from Crowd Annotations | An T. Nguyen | ACL 2017 | PMC5662012 | 已核实 |
| BSC | A Bayesian Approach for Sequence Tagging with Crowds | Edwin Simpson | TACL 2019（arXiv 1811.00780） | arXiv 1811.00780 | 会议版本未逐字核实 |
| Dawid–Skene | Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm | A. P. Dawid | Applied Statistics 28(1) 1979 | — | 经典，未再核 |
| Proactive learning | Proactive Learning: Cost-Sensitive Active Learning with Multiple Imperfect Oracles | Pinar Donmez | CIKM 2008 | — | 未核实（凭记忆） |
| Yan 2016 | Active Learning from Imperfect Labelers | Songbai Yan | NeurIPS 2016 | — | 未核实（凭记忆） |
| Ju 2023 | Distilling Vision-Language Pre-training to Collaborate with Weakly-Supervised Temporal Action Localization | Chen Ju | CVPR 2023 | arXiv 2212.09335 | 已核实 |
| TANDEM | TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech | — | AAAI-ICWSM 2027（摘要页所写） | arXiv 2601.11178 | 已核实存在；方法细节只读摘要 |
| TFPLG | Training-Free VLM-Based Pseudo Label Generation for Video Anomaly Detection | — | IEEE 期刊 2025 | ieeexplore 11015429 | 期刊名未核实 |

## 8. 不阻断的观察

- `window_targets` 对负例视频给全部 30 窗目标 0，包括从未揭示的窗；这是视频标签派生的监督（同块级 MIL 对负例的处理），不是裁定泄漏，但论文描述 C 时要写清"负例的窗目标来自视频标签"。
- README 第 2 节 W1 门用 VERA .562 作 HCS 下限，与 `RESEARCH_ITERATION_RULES.md` "within 只报告不作门"（2026-09-06）并存；这是本轮自设的终止条件，不是规则门，README 已写明是预注册，建议在 STATUS 里也按"自设终止条件"称呼。
