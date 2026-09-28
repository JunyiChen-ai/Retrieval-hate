# 第 5 次修改前的文献检索（2026-09-29）

用户问题：第 4 版的问题 1（答案可靠性表对正例视频短区间不对）、3（自动停止不如固定次数）、4（HateMM 问 8 次 within 不升）、5（同一视频内答案错误相关），文献里有没有不用逐秒标注的解法。四个独立检索 agent（Claude Fable 5.1，WebSearch / arXiv / Semantic Scholar，论文均经核实），报告原文见下四节，汇总见第 5 节。

## 1. 可靠性表：不用逐秒标注能否估三状态

核心理论事实（Liu, Cheng & Zhang, "Identifiability of Label Noise Transition Matrix", ICML 2023）：每个实例只有一个噪声标签时，实例级转移矩阵不可辨识——论文给的反例正是我们的情况：P(Y=1|x)=1, e=0.3 与 P(Y=1|x)=0.7, e+=0.1, e-=0.233 产生完全相同的 P(Ỹ|x)。定理 4.2（基于 Kruskal 1977）：三个条件独立、信息性的噪声标签是充要条件。所以 EM / 联合学习漂移不是调参问题，是不可辨识；任何可行路线都必须引入第三方"重复观测"。按适用性排序：

1. **三视图三阶共识（首选）**。Liu et al. ICML 2023；Zhu, Song & Liu, "Clusterability as an Alternative to Anchor Points When Learning with Noisy Labels", ICML 2021 (HOC)；Fu et al., "Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods", ICML 2020；Pepe & Janes, Biostatistics 2007（3 个测试的闭式解）。额外信息 = 同一节点上 3 个给定状态后条件独立的答案；无需锚点。做法：对正例视频全部 4–8 s 节点各问 3 次（仅帧、仅转录、第二 VLM 或同 VLM 不同帧子集）；统计一阶 / 二阶 / 三阶一致率，闭式解出含仇恨节点比例和每视图的 (FPR_pos, TPR_short)；负例视频节点单独给 FPR_neg，三状态模型成立。风险：每视频相关错误直接破坏条件独立（Kim et al., "Correlated Errors in Large Language Models", ICML 2025：不同厂商模型同错 60% 一致）。缓解：HOC 式把"重复观测"换成跨视频特征 2-NN 节点，天然切断视频内相关；用 Jaffe et al., AISTATS 2016 的协方差残差检验先验证独立性。未试过；已试的同深度 EM 是单标签情形，恰是不可辨识的那种。
2. **带随机效应的潜类模型**。Qu, Tan & Kutner, Biometrics 1996；Dendukuri & Joseph 2001；Hui & Walter 1980。≥3 个测试 + 视频级随机效应。是已试"每视频倾向"的多视图版；单视图下它解释掉视频信号。
3. **正例视频内部锚点 + T-revision（零额外调用）**。Patrini et al. CVPR 2017；Liu & Tao TPAMI 2016；Xia et al. NeurIPS 2019。假设正例视频里存在几乎必然无仇恨的节点，只用正例短节点训 g(x)=P(o=yes|x)，取分位数作 FPR_pos / TPR_short。风险：极值分位挑中"什么都说 no"的视频；T-revision 不冻结骨干就回到已试的联合漂移。
4. **VolMinNet 最小体积约束**。Li et al. ICML 2021。骨干冻结下对正例节点拟合 2×2 T_pos。与已试联合学习同族。
5. **树结构自复制（Hidden Markov Tree）**。Crouse, Nowak & Baraniuk 1998。全树同视频，相关错误无法切断；已试同深度 EM 是其子情形，风险高。
6. **未知噪声率的噪声二分**。Rodriguez & Ludkovski 2020。同节点多次改写提示多数票估局部正确率；47% / 57% 接近随机，低适用。
7. **MPE / PU 与 LLM-judge 校准（均低适用）**。Ramaswamy 2016；Scott 2015；Garg 2021；Bekker & Davis 2020：要求分量样本与混合内分量同分布，18% / 47% 已否定。Berthon et al. ICML 2021：VLM token 概率不是所需的量。Prediction-powered inference（Angelopoulos et al. Science 2023）需要小标注集——若允许 validation 逐秒标签拟合答案模型，它是最直接的修法。

结论：有可辨识性证明的只有"每个正例短节点 ≥3 个条件独立观测"；其余要么需要标签，要么是已试方法换皮。首测：正例视频 4–8 s 节点做仅帧 / 仅转录 / 第二 VLM 三视图，先做独立性检验，再与 47% / 57% 核对。独立性不过则改用 HOC 的跨视频 2-NN。

## 2. 自动停止

硬事实：答案模型假设的每答证据量 KL(Bern .96‖Bern .18) ≈ 2.1 bit，实测短节点 KL(Bern .57‖Bern .47) ≈ 0.03 bit，相差约 70 倍；按真实率 8 个答案总共约 0.2 bit。任何"基于后验信息量"的停止规则在当前答案模型下都在读一个假的量，这是 EIG 阈值与 VOI 两条规则同时失败的根因。

1. **先修似然温度，再复用 VOI（广义贝叶斯 / SafeBayes）**。Bissiri, Holmes & Walker JRSS-B 2016；Grünwald & van Ommen 2017；Wu & Martin 2023。P(answer|·)^η，η 一个标量，可只用训练集视频级标签估计（取使视频级对数似然最大者），或 validation 选。η 是全局常数，不能表达每视频倾向。
2. **预测稳定性停止（不依赖校准）**。Bloodgood & Vijay-Shanker CoNLL 2009；Zhou et al. PABEE NeurIPS 2020；综述 Pullar-Strecker et al. 2024。停在输出本身稳定：连续 t 步秒级排序 / top-k 集合一致即停，(t, τ) validation 选。风险：EIG 选题就是选"最能改变后验"的题，与稳定性对立；每视频倾向造成"稳定但错"。
3. **重复提问一致性停止 + 估每视频偏置**。Aggarwal et al. EMNLP 2023；Li et al. ICLR 2024。同一节点换提示 / 换帧问 m 次得 yes 率；根节点与随机节点的 yes 率估视频基线；假设采样独立。
4. **用视频级标签学停止策略（RL）**。Shim, Hwang & Yang NeurIPS 2018；Janisch et al. AAAI 2019；FrameExit CVPR 2021；AdaFrame CVPR 2019；Cheng & Huan 2025。奖励是视频级、指标是秒级：策略会学"正例视频立即停"；Kossen et al. TMLR 2023 报告 RL 学不出自适应获取。
5. **成本惩罚的 CMI / VOI 与非短视前瞻**。DIME (Gadgil, Covert & Lee ICLR 2024)；Covert et al. ICML 2023；Frazier, Powell & Dayanik 2008；Chick & Frazier 2012；Valancius et al. ICML 2024；Foster et al. ICML 2021。全部依赖校准后验。
6. **未知噪声率的序贯检验**。Kaufmann & Koolen JMLR 2021；Gretta & Price ICALP 2024。每节点需几十次调用。
7. **VLM 自报置信停止**。VideoAgent ECCV 2024；VideoTree CVPR 2025；TraveLER EMNLP 2024。与每视频倾向同源。
8. 序贯共形（COINS 2026）需带标签校准集，仅作记录。

裁定：先修答案模型（1 或第 1 节的路线）再复用 EIG / VOI；不修校准也可能有效的只有 2 和 3，而 3 的重复提问其实就是在无标签地估计答案模型。

## 3. 答案错误相关（同一视频内）

核心诊断：单个标注者时，"该 item 让人倾向说是"与"该 item 真的是"在答案上不可分；GLAD / Dawid–Skene 类的 item 效应只有多个准确率不同的标注者共标同一 item 时才可识别。可行路线三类：(i) 相关性建成全局参数而非逐视频潜变量；(ii) 用额外调用测量逐视频倾向；(iii) 用协变量从负例视频预测倾向后插入。

1. **嵌套答案的"复制型持久噪声"似然（全局标量）**。Bach et al. ICML 2017；Varma et al. ICML 2019；Rühling Cachay et al. 2021（忽略依赖导致系统性过度自信的界）；MACE (Hovy et al. NAACL 2013) 复制机制。改似然：P(a_C | s_C, a_P) = π·1[a_C = a_P] + (1−π)·P(a_C | s_C)，π 每层一个全局标量（validation 选）。树上精确推断不变；子答案与父答案相同时证据被稀释；EIG 自动降低"很可能只是复制父答案"的子节点价值。只吸收嵌套冗余，兄弟区间共享的倾向吸收不了。零额外调用。
2. **对照查询：测得逐视频"说是"倾向**。Zhao et al., "Calibrate Before Use", ICML 2021；Leng et al. VCD CVPR 2024。每视频加 1–2 次调用：同一提示、同一转录上下文，帧换成已知负例片段或强失真帧；观测改为 log-odds(yes|真实) − log-odds(yes|对照)。风险：对照仍泄露仇恨语义（转录含仇恨）会一并减掉真信号。
3. **用负例视频拟合协变量驱动的假"是"率**。Yan et al. AISTATS 2010；GLAD NeurIPS 2009。负例视频每个"是"都是假阳，用视频级特征回归逐层假"是"率，正例视频插入预测值。参数来自协变量而非本视频答案，不会解释掉视频级信号。风险：正例的假"是"率可能被仇恨本身抬高。
4. **采集规则：相关噪声下的联合信息增益**。Golovin, Krause & Ray NeurIPS 2010 (EC2)；Chen, Hassani & Krause AISTATS 2017 (ECED)；BatchBALD NeurIPS 2019；Boczkowski et al. 2016（永久噪声树搜索）。最小改动：EIG(node) × (1 − ρ·overlap)，或"父已问则子答案替换父答案而非叠加"。只改选问不改似然。
5. **两通道打分：视频内中心化只用于视频内排序**。Zhou et al., Batch Calibration, ICLR 2024。与已失败方案同源。
6. **平衡（反向计分）问题对**。Billiet & McClendon 2000；反面证据 Alhamoud et al. CVPR 2025（VLM 不懂否定）。
7. **多"标注者"分散化**。Welinder et al. NeurIPS 2010；Paun et al. TACL 2018；CARE 2026；警示 Kim et al. ICML 2025。≥3 个错误画像不同的"标注者"（纯帧、纯转录、第二家族 VLM）。
8. **区间长度依赖的答案模型**。Kaspi, Shayevitz & Javidi 2016；Atia & Saligrama 2012。缓解而非解决。

裁定：先试 1（零额外调用、不动采集器、直接对准嵌套重复计数）；within 仍无改善再试 2（+1–2 调用 / 视频），3 是其零调用替代；4 的 overlap 折扣可与 1 叠加。

## 4. 同类 VLM 定位方法怎么处理

记法：P1 短段可靠性 / P3 调用次数与停止 / P4 within 排序 / P5 重叠窗相关错误。★ 报告调用数–精度曲线；◆ 报告 within。

1. VADTree (Li et al. 2025, 2510.22693)：P1 子节点分数方差作一致性代理，父子融合权重随方差变；P3 节点数由 GEBD + K-means 决定；P4 不处理；P5 RemoveDup + 父子融合。
2. Holmes-VAU (CVPR 2025, 2412.06171)：定位分数来自帧级标签训练的打分器，MLLM 不参与定位；每视频 1 次调用。
3. LAVAD (Zanella et al. CVPR 2024)：固定网格每 16 帧一次；相似帧平滑；◆外部审计 within 61.75 vs micro 80.28。
4. AnomalyRuler (Yang et al. ECCV 2024)：规则归纳随机批只留一致元素；EMA + 多数投票；假设错误独立。
5. VERA (Ye et al. CVPR 2025, 2412.01095)：视频级标签学引导问题；固定 10 s 段；平滑非去相关。
6. QVAD (Bekit et al. 2026, 2604.03040) ★：LLM 置信 p < .7 触发追问，最多 2 轮；AUC vs 轮数 77.2 → 82.2 → 84.3。
7. Huang, Devereux & Wang 2026, "Your VLM Already Knows When: Training-Free Temporal Grounding by Asking Yes or No", 2608.08315 ★◆：首 token logits 得 P(Yes)，只用于视频内排序不用于阈值；GT 片段在 80.5% 成对比较中胜出尽管绝对概率低；粗扫 12 + 细扫 36 次；Kc=8 仅 −1.1pp。
8. Probe-VAD (Gu et al. 2026, 2609.17211)：10 个有序严重度阈值的 YES/NO 似然 + 等渗回归；解码答案 70.2 vs 似然 84.4 vs Probe 86.3 AUC (UCF)；明言不假设校准。
9. Song & Lee 2026, "A VLM Answer Is Not an Anomaly Score", 2608.21244：解码答案导致秩压缩；改用 Σ v(a) p(a)：UCF 72.2 → 85.3，XD AP 49.4 → 69.0。
10. Song & Lee 2026, "Frame-Level Evaluation in WSVAD Mostly Measures Video-Level Ranking", 2608.21854 ◆：Micro-AUROC = w·Within + (1−w)·Cross，同视频对仅占 0.07–0.39%；VadCLIP within 72.9 / 84.3、LAVAD 61.8 / 64.8。
11. CEAVAD (Yin et al. 2026, 2608.09908)：每段同时对三假设（危害 / 通用正常 / 机制专属良性对照）打分取 margin；Past/Target/Future 分区；固定 2 s 网格约 50 次 / 视频。
12. SlowFastVAD (Ding et al. 2025, 2504.10320) ◆：快检测器分数熵决定是否送 VLM；报 Macro AUC。
13. PANDA (Yang, Gao & Shou NeurIPS 2025, 2509.26386)：固定 1 次 / 秒；"Insufficient" 触发反思最多 3 轮。
14. LELA (Sun et al. 2026, 2602.09637)：唯一训练无关的 hateful 定位；逐帧 caption + LLM；HateMM ROC .726 vs LAVAD .578；固定逐帧网格。

其余核查：MultiHateLoc (2512.10408) 纯 MIL；HateClipSeg (MM'25) 段级标签训 ActionFormer；DeHate / ImpliHateVid 只视频级；MARS (2601.15115) 视频级对抗提示；CLARA (2608.15905) 分类；VideoAgent ★ 置信 1–3 停止、轮数曲线 53.8 / 58.6 / 60.2 / 59.8；VideoTree ★；TraveLER；TimeChat / VTimeLLM / Momentor 全监督；Moment-Video (2606.02522) MLLM 对瞬时事件随视频长度急降。

可移植：P1 (a) 不用解码 yes/no，取首 token P(Yes) 或有序阈值似然（#7 / #8 / #9，+13–16 AUC）；(b) VLM 分数只当视频内排序用；(c) #4 随机批一致元素、#1 子节点方差作无标签可靠性代理。P3：#12 用骨干分数熵决定是否问；#6 / #7 置信阈值停止 + 轮数曲线。P5：#11 三假设对照与 Past/Target/Future 分区；#1 父子一致性融合。P4 无人正面解决，仅 #10 给出分解。"固定网格 + 无可靠性模型"：LAVAD、VERA、AnomalyRuler、Probe-VAD、CEAVAD、PANDA、LELA、#9。

## 5. 汇总与建议（主 agent）

- 问题 1 的结论是定理级的：一个 VLM、每节点一个答案、只有视频标签，三状态表不可辨识（Liu et al. ICML 2023）。不用逐秒标注的唯一有证明的路线是每个正例短节点 ≥3 个条件独立的答案（多视图），前提是视图间错误独立，而 Kim et al. ICML 2025 表明这很可能不成立，需先做独立性检验。
- 问题 3 的前提是后验可信；现在每个答案的信息量被高估 70 倍，先修表再谈停止。不看后验的停法只有"输出稳定即停"和"重复提问一致即停"。
- 问题 5 有一条零调用的改法：嵌套答案的复制型似然（每层一个全局标量），可离线用现有缓存答案检查。
- 同类方法都不建可靠性模型；三篇 2026 论文一致表明用 VLM 首 token 概率代替解码 0–3 分、且只用于视频内排序，比解码答案高 13–16 AUC。我们现在用的是解码分（几乎只有 0 / 3）。这是零标注、可直接试的改动，对问题 1 和 4 都有针对性。
