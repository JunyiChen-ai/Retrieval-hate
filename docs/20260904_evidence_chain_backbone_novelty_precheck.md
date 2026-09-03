# 新候选骨干(证据链骨干)novelty 预检(规则 4)

日期:2026-09-04。范围:只读文献检索,不改代码、不跑实验。请求方:主会话(retrieval-hate-95);执行:本会话四个独立 fable 检索 agent 并行,每个部件 ≥ 25 次 WebSearch 加原文 WebFetch 核对,检索记录与链接逐条列在各附录。判据:`RESEARCH_ITERATION_RULES.md` 第 4 条(只挡四类:源方法已用于 hateful video、纯 ensemble、纯后处理/校准/平滑、纯工程),以及 STATUS 里的三条 novelty 条件。

背景(来自主会话):修订 1 的机制分析显示骨干靠三件事起作用——视频级仇恨密度估计(片段 logit 方差 92%/79% 在视频之间)、两粒度裁定可靠性校正(K30 单独触发在 HateMM 不可信,固定噪声率 EM 学不到)、视频内排序主要来自 HMM 时间后验;CMAL 对比损失是唯一有用的训练项,EMA 自蒸馏无用。计划新骨干(超参 0)五个部件。

## 总表

| 部件 | 最接近的先例(差别要点) | 已用于 hateful video? | 规则 4 STOP? | 改造非平凡? |
|---|---|---|---|---|
| 1 链式结构化输出头(神经一元势 + VLM 固定发射势进马尔可夫链,训练目标 = 视频标签边际似然"至少一段为 1") | MI-DORF(TIP 2018,链 + "至少一个"边际似然 + 前向后向,线性一元势、无外部噪声势);GLWS(ICML 2024,MIL 的 2 状态 NFA 前向后向,实例条件独立、无转移);Wang–Li–Metze 2018(noisy-or 池化在序列上不定位——直接风险证据);MI-HMM(TGRS 2015);Dugong/Linked HMM/CHMM(多源噪声标注链,无视频级约束) | 否(MultiHateLoc、LELA、TANDEM、HateClipSeg 均无) | 四类均不触发 | 是,新在组合:时间转移 + 视频级"至少一个"似然 + 神经一元势 + 两粒度外部裁定发射势;无先例同时具备。必须预注册两项:去掉转移势/去掉 VLM 势的消融(否则退化为 noisy-or,已知失败);EM 噪声率是 P(裁定∣视频标签) 而非 P(裁定∣段标签) 的近似,要显式写出或做敏感性分析 |
| 2 视频级证据分布编码器(K30/K4 裁定分布 + 池化内容 → 密度 d_v,条件化链的初始分布与转移) | 层次 HMM(Xie et al. 2003;顶层是随时间切换的离散状态、无监督);IOHMM(Bengio & Frasconi 1995;条件变量是逐时刻输入);Absorbing Markov Chain WSVAD(CVIU 2023;转移来自特征相似度图,无视频级量);基数势 Markov 网络 MIL(UAI 2013;比例固定参数化、无时间链);MultiHateLoc(无视频级先验、无链) | 否 | 四类均不触发;第三类需消融自证:若 d_v 只进初始分布或转移对 d_v 不敏感,退化成"每视频一个偏置"= 逐视频校准,必须加"d_v 只作 logit 偏置"对照 | 是:三要素(上层决定下层链参数 / 转移是输入的函数 / 包级比例约束)各有先例但从未组合;要解决 d_v 参数化、转移矩阵对 d_v 的函数形式、防止 d_v 变成纯视频分类器 |
| 3 粒度可靠性门(内容 + 上下文 → 两粒度裁定势的权重) | WeaSEL(NeurIPS 2021,样本依赖的多源精度,无标签自举、无序列);CoNAL(AAAI 2021)、Li et al. ML 2022 / TPAMI 2024(实例依赖混淆矩阵,需逐实例噪声标签);CrowdAttention(Sensors 2025,自洽加权);CANE(2026,LLM 标注可靠性随特征区域变化,按簇统计) | 否(HVGuard 的 MoE 是特征空间专家选择;LELA/TANDEM 直接用 VLM 输出) | 四类均不触发 | 是,但要说清三点:只有视频级标签、门经链边际化由 MIL 反传(可识别性问题,需用 HateMM .16 / HateClipSeg .73 的门权重分布验证);"同一标注者不同粒度"是新的可靠性轴;需消融"门只读内容 vs 读内容+裁定""固定权重 vs 学习权重"。风险:门学成常数即退化为 EM 固定噪声率,预注册应把"两语料对短窗粒度的门权重均值差异显著"设为必要条件 |
| 4 后验引导的视频内对比(链后验选正负片段,替代 CMAL) | MACIL-SD CMAL(MM 2022,自身 top-k 选、跨模态对);CoLA SniCo(CVPR 2021,自身分数 + 膨胀腐蚀);LAP(2024,文本相似度独立阈值伪标签进 BCE,不选对比对);PVLR(2024);DSCIL(PR 2025) | 否(MultiHateLoc 的对比是跨模态同时间戳对齐,不选段) | 四类均不触发;注意若消融显示对比损失无贡献、后验只在推理起作用,会被归为后处理 | 非平凡但边际:新在"网络证据 + VLM 裁定 + 时间转移的联合后验做选择",断开自选自确认回路;与 LAP 只差时间耦合与用途,须配消融(后验选 / 自身 top-k 选 / VLM 独立阈值选) |
| 5 后验蒸馏到单模态分支(可选) | Structural/Posterior KD(ACL 2021,CRF 后验蒸馏);Linked HMM noise-aware loss(AAAI 2020);RNN-HMM Viterbi 自训练(CVPR 2017/18);DAKD(WACV 2025,集成教师);MACIL-SD 自蒸馏经核对是"视觉单模态→音视频、EMA 参数融合",不是损失蒸馏 | 否 | 四类均不触发;边界风险"把后处理蒸进网络" | 中等:损失形式已有;新在弱监督多模态闭环与 VLM 锚。**单独 novelty 偏弱,建议只作部件 4 的配套并消融**;需处理自增强循环(用户两次抓过"证据认定步 = 原判断本身"),设计期须说明 VLM 势作为固定外部锚如何阻断,并消融"去掉 VLM 势后蒸馏是否退化" |

## 跨部件要点

1. 五个部件都没有在 HateMM / MultiHateClip / HateClipSeg 文献里出现;同任务已发表方法核对:MultiHateLoc(WWW 2026)、LELA(2026,training-free)、TANDEM(ICWSM 2027,RL)、HateClipSeg(MM 2025,ActionFormer)、HVGuard(EMNLP 2025,视频级)、SafeLens(AAAI 2026)。
2. 部件 1 是核心且先例最近(MI-DORF 的 ζ_t 构造与"至少一个"前向后向等价);novelty 必须靠"VLM 两粒度发射势 + 神经一元势 + 视频级似然"的组合和可靠性门(部件 3)一起支撑,单独的"链 + 至少一个"不能作主张。
3. 三处必须写进预注册的退化对照:(a) 去转移势 / 去 VLM 势(部件 1,noisy-or 失败模式);(b) d_v 只作偏置(部件 2);(c) 门固定权重 / 门只读内容(部件 3)。
4. 两处理论近似要明说:EM 噪声率的条件对象是视频标签而非段标签;K4 块裁定并入链需要块级观测结构(现有 verdict_hmm 的 (s,h) 增广状态可复用)。
5. 部件 5 若做,只作部件 4 的配套,不单列。

## 附录:四份检索报告原文(A = 部件 1;B = 部件 2;C = 部件 3;D = 部件 4 与 5)


---

# 附录 A

# 文献预查报告:链式结构输出头(Chain-structured output head)

日期:2026-09-04。只读检索,未改任何文件。

## 1. 检索记录(逐条原文,均于 2026-09-04 执行)

WebSearch(27 条):
1. `multiple instance learning hidden Markov model sound event detection weakly labeled`
2. `noisy-OR "at least one" multiple instance learning sequence marginal likelihood forward-backward`
3. `CRF multiple instance learning weakly supervised sequence labeling bag-level label "multiple instance" "conditional random field"`
4. `weakly supervised temporal action localization Markov chain forward-backward differentiable video-level label structured MIL`
5. `weakly supervised video anomaly detection HMM CRF video-level label structured output temporal`
6. `hateful video localization HateMM MultiHateClip HateClipSeg segment-level`
7. `"presence-only" OR "presence labels" Markov model weak supervision sequence "video-level" latent segment labels EM noisy annotator`
8. `"MIL" "structured" "HMM" weakly supervised temporal segmentation video "bag" "sequence" Viterbi marginal`
9. `MultiHateLoc temporal localisation multimodal hate content online videos arXiv`
10. `"hidden Markov" "multiple instance" bag sequence "at least one" audio OR bird OR "sound event" MI-HMM`
11. `weakly supervised localization vision-language model pseudo labels noisy labeler confusion matrix EM hidden Markov model temporal segment "video-level"`
12. `"multiple instance learning" "Markov" "video-level" "forward-backward" OR "forward algorithm" anomaly OR action "at least one segment" likelihood neural`
13. `"multi-instance" OR "multiple instance" "cardinality potentials" OR "cardinality potential" temporal chain video segments weakly supervised localization`
14. `hateful video detection HateMM "Markov" OR "CRF" OR "HMM" OR "multiple instance learning" weakly supervised localization`
15. `Dugong multi-resolution weak supervision sequential data video Varma Ré NeurIPS 2019 labeling functions different resolutions`
16. `weakly supervised sequence tagging from noisy rules linked hidden Markov model Safranchik AAAI 2020`
17. `"multiple instance hidden Markov model" OR "multiple-instance hidden Markov model" landmine GPR sequence bag`
18. `"absorbing Markov chain" weakly supervised video anomaly detection end-to-end label estimation`
19. `"multiple instance" "conditional random field" OR "hidden conditional random field" video event detection temporal localization weakly supervised latent segments`
20. `"multiple instance learning" forward-backward algorithm log-space "at least one positive" bag probability sequence neural network implementation`
21. `BERTifying the Hidden Markov Model multi-source weakly supervised NER conditional hidden Markov model CHMM ACL 2021`
22. `"learning from weak supervision" GLWS forward-backward "multiple instance" temporal action localization OR video anomaly extension 2025`
23. `"A General Framework for Learning from Weak Supervision" Chen Wang Feng Sugiyama Raj ICML 2024`
24. `weakly supervised video anomaly detection "noisy-or" OR "noisy or" video-level probability "at least one" snippet likelihood instead of top-k MIL`
25. `"hidden Markov" OR "Markov chain" head neural network "video-level labels" frame posterior forward-backward weakly supervised anomaly OR violence OR hate localization 2024 2025 2026`
26. `Wu Wang Ji "Multi-instance hidden Markov model for facial expression recognition" FG 2015 video-level label at least one frame`
27. `Liu Chen Zhu Liu Metaxas "Video classification via weakly supervised sequence modeling" CVIU 2015 hidden CRF multiple instance`

WebFetch(原文核对):arXiv 1804.01146(abs + PDF 全文,pdftotext)、1309.6833、1803.00907、1609.01465(PDF 全文,pdftotext)、2309.05086、2108.06816、2602.09637、2601.11178、2004.00163、2512.10408、2402.01922(abs + HTML 方法节)、1910.09505、2508.01712(abs + HTML v2 全文)、Springer s13735-014-0068-6(重定向,改用 Semantic Scholar API)、Hacettepe MI-HMM PDF(pdftotext)。

## 2. 最接近的工作(5 项)

**(1) Multi-Instance Dynamic Ordinal Random Fields (MI-DORF)** — Ruiz, Rudovic, Binefa, Pantic。arXiv 1609.01465(2016);期刊版 IEEE TIP 2018,arXiv 1803.00907。
做了什么:bag = 一段视频(帧序列),帧标签 h_t 是隐变量,构成线性链:节点势 Ψ^N(x_t,h_t)(线性 ordinal probit,手工特征)+ 转移势 Ψ^E(h_t,h_{t+1})(可学习 L×L 矩阵)+ 高阶 MIL 势 Ψ^M(h,y):当 max(h)=y 时取 w·#{t:h_t=y},否则 −∞(即"至少一帧等于 bag 标签")。训练目标是 max Σ log P(y|X;θ),对全部 h 求边缘。推断:引入辅助二值变量 ζ_t="前缀 h_{1:t} 是否已出现过 y",把高阶势改写成链上的节点/边势,然后用标准 forward-backward,复杂度 O(T·(2L)^2)。数据:UNBC 疼痛强度。
与计划组件的差异:链结构和"至少一个"边缘似然这两点几乎完全相同(ζ_t 的构造就是计划里 forward-backward 求 P(至少一段为正) 的做法)。不同点:(a) 一元势是线性 ordinal 模型而非神经骨干;(b) 没有外部噪声标注者的势——它只有输入特征的节点势,没有"冻结 VLM 判定 + EM 估计的噪声率"这一类固定发射势,更没有两粒度(K=30 / K=4)窗口判定;(c) bag 势带一个"正例比例"项 w·count,计划里是纯"至少一个";(d) 任务是 ordinal 强度估计,不是二分类定位。

**(2) A General Framework for Learning from Weak Supervision (GLWS)** — Chen, Wang, Feng, Li, Wang, Xie, Sugiyama, Singh, Raj。ICML 2024(PMLR v235),arXiv 2402.01922。
做了什么:把各类弱监督统一写成 label 序列上的 NFA;MIL 的 NFA 只有 2 个状态(q0→q1 必须经由一个正标签,之后任意),用 forward-backward 计算每个实例的后验 p(y^j | x^{1:L}, w),训练目标是 EM:无监督项让预测贴近后验,监督项最大化弱监督 w 的概率。
差异:GLWS 明确假设实例条件独立 p(y^{1:L}|x)=∏_j p(y^j|x)(Assumption 3.1),没有可学习/固定的实例间转移势——它的"链"只是 NFA 状态,不是时间马尔可夫链;不支持外部噪声标注者势(文中留作 future work);实验全是图像 bag(CIFAR/STL/ImageNet-100 随机成 bag),无视频/时序。计划组件 = GLWS 的 MIL 边缘似然 + MI-DORF 式的时间转移 + 额外的 VLM 发射势。

**(3) Comparing the Max and Noisy-Or Pooling Functions in MIL for Weakly Supervised Sequence Learning Tasks** — Wang, Li, Metze。arXiv 1804.01146(2018,ICASSP 系)。
做了什么:在语音(TEDLIUM,phoneme 有无标签)和 SED 上比较 max pooling 与 noisy-or pooling y=1−∏(1−y_i)。结论:max 能定位,noisy-or 完全失败(PER>90%,预测几乎全为空)。给出的原因:(1) 相邻帧强相关,违反独立假设;(2) 连乘使得所有帧都很低时序列仍被判为正。
差异:这是一个对计划组件的直接风险证据,而不是同构方案。计划里的 P(至少一段为正) 在无转移势、无外部势时退化为 noisy-or;计划靠马尔可夫转移 + VLM 固定势改变这一点,但"链结构是否消除 noisy-or 的定位失败"没有先例支持,需要作为消融项显式验证(去掉转移势、去掉 VLM 势各一组)。

**(4) Multiple-Instance Hidden Markov Models with Applications to Landmine Detection** — Yuksel, Bolton, Gader,IEEE TGRS 53(12) 2015;同类:**Multi-instance Hidden Markov Model for Facial Expression Recognition** — Wu, Wang, Ji,FG 2015。
做了什么:MIL + HMM 的生成式版本。TGRS:bag = 一组序列(不同深度的 GPR 时间序列),bag 正 ⇔ 至少一条序列是目标序列,用 noisy-OR 把各序列的 HMM 似然合成 bag 概率,stochastic EM 训练。FG 2015:视频先聚类切段,段为实例,序列级标签,MIL 定表情峰值。
差异:HMM 状态在每条序列内部,"至少一个"作用在序列集合上,不是在同一条时间链的段之间;生成式、手工特征、EM 训练;没有噪声标注者势;不做帧级 AUC/AP 评测。

**(5) 噪声标注者链模型一族**:Dugong — Sala, Varma, Fries, Fu, Sagawa, Khattar, Ramamoorthy, Xiao, Fatahalian, Priest, Ré,NeurIPS 2019,arXiv 1910.09505;Linked HMM — Safranchik, Luo, Bach,AAAI 2020;CHMM — Li, Shetty, Liu, Zhang, Song,ACL 2021,arXiv 2105.12848;Neural-Hidden-CRF — Chen et al.,KDD 2023,arXiv 2309.05086。
做了什么:隐真标签链(HMM/CRF),观测 = 多个弱标注源的输出,标注源的准确率/混淆矩阵在无真标签条件下用 EM 或矩方法估计;Dugong 专门处理"多分辨率"源(逐帧源与整窗口源同时存在),并在视频任务上验证。
差异:这些模型没有 bag/视频级"至少一个"约束,监督全部来自标注源本身;输出是训练用的概率标签(Dugong、linked HMM)或链本身即模型(CHMM/Neural-Hidden-CRF),不是与神经骨干一元势联合训练的输出头。计划组件的"K=30 与 K=4 两粒度 VLM 判定 + EM 估噪声率"对应 Dugong 的多分辨率源,但计划里噪声率是在视频标签上估的,且判定作为固定发射势并入带视频级似然的链。

补充(相关但更远):Shirahama, Grzegorzek, Uehara, "Weakly supervised detection of video events using hidden conditional random fields," IJMIR 2014——视频级标签、隐状态区分相关/不相关 shot、边缘似然训练,但目标是事件检测(视频级),非神经、无外部势。Hajimirsadeghi et al., UAI 2013 (arXiv 1309.6833)——cardinality 势 Markov 网络 + max-margin,实例无序无链。"End-to-end learning for WSVAD using Absorbing Markov Chain," CVIU 2023——AMC 只用于 GCN 内的标签清洗,无 forward-backward、无边缘似然。EM-MIL (Luo et al., ECCV 2020, arXiv 2004.00163)——EM 对 key-instance 分配,片段间无马尔可夫结构。

## 3. 是否已用于 hateful video 检测/定位

**否。** 核对的全部 hateful video 定位工作:
- MultiHateLoc(Sun et al., WWW 2026, arXiv 2512.10408):模态感知时序编码器 + 跨模态融合 + "modality-aware MIL objective";摘要与方法描述中无 Markov/CRF/HMM/forward-backward,无 VLM 判定势。
- LELA(Sun, Yang, Jiao, Fu, arXiv 2602.09637, 2026):training-free,多阶段 prompting 逐帧打分;无训练、无结构化模型。
- TANDEM(Koushik, Treharne, Kanojia, ICWSM 2027, arXiv 2601.11178):VLM/ALM 的 RL 联合优化;无 HMM/CRF。
- HateClipSeg(arXiv 2508.01712 v2):定位任务唯一 baseline 是 ActionFormer,指标 tIoU@0.3/0.5/0.7;明确无 Markov/CRF/HMM 方法。
- HateMM 原论文与 MultiHateClip 原论文只做视频级分类。
- 检索 14 号查询("hateful video" + Markov/CRF/HMM/MIL)未返回任何结构化输出头用于 hateful video 的记录。

## 4. 规则 4 四类 STOP 判定

- **源方法已用于 hateful video**:不适用(第 3 节)。
- **纯 ensemble**:不适用。VLM 判定不是和骨干分数做加权平均,而是作为固定发射势进入同一个联合概率模型,噪声率由 EM 估计,训练目标是链的视频级边缘似然。需要注意的边界:如果最终实现退化成"骨干 logit + λ·VLM logit 再平滑",评审会归为 ensemble;方案里 EM 噪声率、转移势、forward-backward 后验三者缺一都会往这个方向滑。
- **纯后处理/校准/平滑**:不适用。链参与训练(梯度经 forward-backward 回传到骨干一元势,目标由 top-k MIL 换成边缘似然),不是推断时对现成分数做 HMM 平滑。同样的边界:若只在推断期把 VLM 势和转移势套在训练好的 MIL 骨干上,则属于后处理。
- **纯工程(超参/特征/增强/训练配置)**:不适用。改变的是输出结构与训练目标。

结论:四类均不触发 STOP。

## 5. 改编是否 non-trivial

是,但新颖性来自组合而非任何单一部件。三个构件各有先例:链 + "至少一个"边缘似然 + forward-backward(MI-DORF 2016/2018 的 ζ_t 构造与之等价;GLWS 2024 的 2 状态 NFA 亦同);神经一元势 + MIL 后验 EM(GLWS);多分辨率噪声标注源 + 无标签估噪声率的隐标签链(Dugong 2019、CHMM 2021)。没有任何一篇同时具备"时间转移势 + 视频级至少一个似然 + 神经骨干一元势 + 两粒度外部判定的固定发射势"。把 K=4 窗口判定并入链需要设计(窗口势覆盖约 7–8 个相邻段,不是单段发射,要么扩展状态、要么把窗口判定当作窗口内任一段为正的观测),这不是套用现成公式。两个必须在预注册里写清的技术问题:(a) Wang–Li–Metze 2018 证明无结构 noisy-or 在序列上不定位,计划组件在去掉转移势和 VLM 势时正是 noisy-or,所以消融必须包含这两组,且要说明为什么链结构能避开该失败模式;(b) 噪声率"只用训练视频标签估计"——视频标签不是段标签,EM 估到的是 P(判定 | 视频标签) 而非链里需要的 P(判定 | 段标签),这个近似要么在模型里显式写出(例如把负视频的段全视为负、正视频用后验加权做 EM),要么承认是偏差并给出敏感性分析。


---

# 附录 B

# 文献预检报告:视频级证据分布编码器(hate density d_v → 条件化分段 Markov 链)

日期:2026-09-04。只读预检,未改任何文件。30 次检索、12 次页面抓取。

## 1. 查询记录(全部 2026-09-04 执行,原文)

WebSearch(30 条):
1. `weakly supervised video anomaly detection video-level prior conditioning snippet scores global branch`
2. `hierarchical hidden Markov model video segmentation top-level state conditions transition matrix`
3. `input-dependent transition matrix HMM "input-output HMM" conditional transition neural network video`
4. `multiple instance learning label proportion estimation positive rate prior bag-level weakly supervised temporal localization`
5. `"video-level" branch "snippet-level" branch weakly supervised anomaly detection RTFM MGFN UR-DMU global context`
6. `hateful video localization HateMM MultiHateClip HateClipSeg weakly supervised temporal`
7. `Markov chain snippet labels weakly supervised temporal action localization HMM CRF transition weak supervision`
8. `"switching" state space model video-level latent conditions segment dynamics anomaly detection hierarchical latent`
9. `"Towards Training-free Multimodal Hate Localisation with Large Language Models" arxiv`
10. `VadCLIP dual branch video-level classification frame-level anomaly "coarse-grained" "fine-grained" weakly supervised video anomaly detection`
11. `hidden Markov model weakly supervised video anomaly detection transition probability snippet anomaly scores temporal smoothing learned`
12. `anomaly ratio estimation prior weakly supervised video anomaly detection "proportion" abnormal snippets adaptive top-k`
13. `hierarchical conditional random field video-level label segment labels weakly supervised "hidden conditional random field" temporal segmentation global latent`
14. `neural HMM differentiable forward algorithm weakly supervised action segmentation transcript video-level supervision`
15. `"Learning Where and When" patch-based spatiotemporal localization weakly supervised video anomaly detection Gaussian mixture prior video-level parameters`
16. `distribution of segment prediction scores as video-level feature histogram of instance scores MIL "score distribution" video classification`
17. `weakly supervised video anomaly detection video-level anomaly probability multiplied with snippet scores "video-level" classifier guides frame-level predictions two-stage coarse-to-fine`
18. `HMM sequence-specific transition matrix conditioned on global latent variable per-sequence "mixture of hidden Markov models" hierarchical Bayesian transition parameters`
19. `multiple instance learning Markov dependency between instances "hidden Markov" bag label sequential instances MIL-HMM structured MIL`
20. `predict per-video anomaly proportion regression weakly supervised anomaly detection "anomaly ratio" video-specific dynamic k selection`
21. `hateful video detection hidden Markov model OR "Markov chain" OR "transition matrix" HateMM temporal segments`
22. `"Absorbing Markov Chain" weakly supervised video anomaly detection end-to-end Computer Vision and Image Understanding 2023 transition`
23. `HiTESS "Hierarchical Temporal Sequence Segmentation" weakly supervised video anomaly detection Expert Systems with Applications`
24. `"Weakly supervised detection of video events using hidden conditional random fields" two-layer activity segments max-margin`
25. `Xie Chang Divakaran Sun "hierarchical hidden Markov model" video structure discovery top-level states 2003 abstract`
26. `CRF transition scores conditioned on global sequence representation "global context" neural CRF sequence labeling conditional transition matrix per-sequence`
27. `weakly supervised video anomaly detection global video representation modulates snippet anomaly scores FiLM conditioning video-level embedding "global" "local" MIL`
28. `"Learning Event Completeness for Weakly Supervised Video Anomaly Detection" video-level prior temporal model`
29. `"learning from label proportions" video temporal segments weakly supervised localization proportion prior frames`
30. `"multiple-instance hidden Markov model" MI-HMM Manandhar Collins landmine bag-level label sequences transition`

WebFetch(12 次):MultiHateLoc arXiv 2512.10408(成功);HateClipSeg arXiv 2508.01712(成功);LELA arXiv 2602.09637(成功);BN-WVAD arXiv 2311.15367(成功);Hajimirsadeghi et al. arXiv 1309.6833(成功);GIG-VAD arXiv 2104.06813(成功);Temporal label noise arXiv 2508.04900(成功);Count-based WSL arXiv 2311.13718(成功);AMC-WSVAD ScienceDirect(403);HiTESS ScienceDirect(403);Shirahama HCRF academia.edu(403);Bui AAAI'04 PDF(二进制,未解析——下文对 HHMM 的描述依据 Fine/Singer/Tishby 1998 定义的标准 HHMM 结构,非该 PDF)。

## 2. 最接近的先行工作(5 篇)

**(a) Hierarchical HMM 用于视频结构发现** — Xie, Chang, Divakaran, Sun, "Unsupervised discovery of multilevel statistical video structures using hierarchical hidden Markov models", ICME 2003(https://www.semanticscholar.org/paper/1f8196633563a381c07326a4375b7c22bd91c303);通用 HHMM 定义见 Fine, Singer, Tishby 1998、Bui AAAI 2004(https://cdn.aaai.org/AAAI/2004/AAAI04-052.pdf)。
做法:两层 Markov 依赖,顶层状态决定底层子链的初始分布和转移矩阵,顶层状态本身随时间切换,参数用 EM 无监督估计,用于体育/新闻视频的结构分割。
与计划组件的差别:(1) 顶层是离散潜变量、随时间变化;计划中 d_v 是**每个视频固定的连续标量**(密度),不是切换状态。(2) HHMM 的顶层状态从观测无监督推断;计划中 d_v 由一个网络从**冻结 VLM 的 K=30/K=4 窗口判定分布 + 池化内容特征**回归出来,只用视频级标签监督。(3) HHMM 的子链参数是按顶层状态查表;计划中 π 和 A 是 d_v 的连续函数(网络输出)。

**(b) Input-Output HMM** — Bengio & Frasconi, "An Input Output HMM Architecture", NIPS 1995(讲义引用:https://pages.discovery.wisc.edu/~sroy/teaching/network_biology/fall2018/lectures/IOHmms_Lecture10.pdf);近期神经版:"End-to-End Training of a Neural HMM with Label and Transition Probabilities", arXiv 2310.02724(https://arxiv.org/abs/2310.02724)。
做法:转移概率 a_kl = P(s_{t+1}=l | s_t=k, u_{t+1}) 由每个时刻的外部输入 u_t 决定,神经网络把上下文映射到随机矩阵。
差别:IOHMM 的条件变量是**逐时刻输入**;计划组件的条件变量是**视频级全局量**(d_v),并且这个量本身是从分段判定的分布中学出来的中间量,再反过来条件化分段链——层级方向不同(video → chain → segment),不是 segment input → transition。

**(c) Absorbing Markov Chain 端到端弱监督 VAD** — "End-to-end learning for weakly supervised video anomaly detection using Absorbing Markov Chain", Computer Vision and Image Understanding, 2023(https://www.sciencedirect.com/science/article/abs/pii/S1077314223001789;摘要来自 ResearchGate 检索结果)。
做法:把 MIL 检测器给出的噪声分段分数当作初值,在分段构成的图上用吸收 Markov 链递归传播,相当于弱监督下的 label propagation,端到端训练。
差别:(1) 它的"转移"来自分段间**特征相似度图**,不是时间链;(2) 没有任何视频级状态或密度参与转移矩阵;(3) 它的作用是分数精修,计划组件的作用是把视频级密度当作链的先验(初始分布 + 转移矩阵)。这是弱监督 VAD 里唯一找到的"Markov 链进主干"的工作,值得在相关工作里正面引用。

**(d) 基于基数(cardinality)的 Markov 网络 MIL** — Hajimirsadeghi, Li, Mori, Zaki, Sayed, "Multiple Instance Learning by Discriminative Training of Markov Networks", UAI 2013(https://arxiv.org/abs/1309.6833);扩展版 TPAMI 2017。
做法:在包级别引入基数团(cardinality clique)显式建模"包内正例比例/歧义程度",max-margin 训练,实例标签推断受该团约束。
差别:(1) 它的"比例"是一个固定参数化的势函数,不是从外部证据(VLM 判定分布)预测出来的每视频量;(2) 实例之间无时间转移结构,比例只约束正例个数,不约束正例在时间轴上的聚集方式;(3) 计划组件用 d_v 同时条件化初始分布和转移矩阵,即同时控制"有多少"和"怎么连成段"。这是"label proportion 先验进 MIL"方向上最近的先例。

**(e) MultiHateLoc** — Sun, Yang, Jiao, Fu 等, "MultiHateLoc: Towards Temporal Localisation of Multimodal Hate Content in Online Videos", arXiv 2512.10408, 2025-12(https://arxiv.org/html/2512.10408v1)。
做法:每模态各一个 Transformer 时序编码器,动态跨模态融合,top-k(K=3,取前 33% 帧)MIL,跨模态对比损失 + 时间平滑正则;HateMM 与 MultiHateClip 上报帧级 mAP/AUC。
差别:全文核对确认——**没有视频级分支或视频级密度估计条件化帧分数,没有 Markov 链/HMM/CRF/转移矩阵,不用冻结 VLM 分段判定**。它是同任务最强的已发表训练式基线,但结构上与计划组件无重叠。

其他相关但更远的工作(一句话):
- BN-WVAD(arXiv 2311.15367):用 batch 级异常比例做自适应 top-k 选择——比例在 **batch 级**而非视频级,不进时序模型。
- VadCLIP(AAAI 2024):粗粒度视频级分类分支 + 细粒度帧级分支,但两分支并列,视频级分支不条件化帧级分支。
- GIG-VAD(Lv, Xu, Cui, arXiv 2104.06813, 弱监督):视频级标签挖出"全局线索向量"来选空间特征,不是时间链条件化。
- Shirahama, Grzegorzek, Uehara, IJMIR 2015 两层 HCRF:视频级事件标签 + 隐藏 shot 状态,max-margin 训练;隐藏状态判"相关/不相关 shot",但没有连续密度量条件化转移。
- MI-HMM(Manandhar et al., IEEE TGRS 2015):包级标签下学 HMM 参数,链参数与包无关。
- LEC-VAD(ICML 2025):用 anomaly-aware 高斯混合学事件边界,是时间位置的参数化先验,不是密度条件化链。

## 3. 是否已被用于 hateful video 检测/定位?

**否。** 证据:
- MultiHateLoc(2025-12)全文核对:top-k MIL + Transformer,无视频级先验、无 Markov 链、无 VLM 判定输入。
- LELA(arXiv 2602.09637, 2026-02, training-free):每帧分数由 LLM 独立打分后跨模态取 max,明确**无视频级条件化、无任何时序模型**。
- HateClipSeg(arXiv 2508.01712):定位基线是 ActionFormer,在线基线是 LSTR,分类是 LLaMA-3.2-11B,无 HMM/先验。
- "Revealing Temporal Label Noise in Multimodal Hateful Video Classification"(arXiv 2508.04900):只做分析,不提出模型。
- 查询 21(HateMM + HMM/Markov/transition matrix)无任何命中。

## 4. 规则 4 四类 STOP 裁定

| STOP 情形 | 是否适用 | 依据 |
|---|---|---|
| 源方法已用于 hateful video | 不适用 | 见第 3 节,四篇同任务论文均无此结构 |
| 纯 ensemble | 不适用 | 单一模型;VLM 判定是输入特征(已在现有模型中),d_v 是学出的潜变量,不是多模型投票 |
| 纯后处理/校准/平滑 | **不适用,但有一个必须防住的退化点** | 链与 d_v 编码器与主干联合训练、只用视频级标签,不是对冻结 logit 做事后平滑。**风险**:若 d_v 只进初始分布、或转移矩阵对 d_v 不敏感,组件就退化成"每视频一个偏置项",那就等价于逐视频校准。消融必须包含"d_v 只作 logit 偏置"对照,证明转移矩阵条件化带来的增量 |
| 纯工程(超参/特征/增强/训练配置) | 不适用 | 改的是概率结构(新增视频级潜变量 + 其对分段链参数的函数依赖),不是配置 |

结论:四类 STOP 均不触发;第三类需要用消融设计自证。

## 5. 改造是否非平凡?

是。现有文献里三个要素各自存在但从未组合:(i) HHMM 有"上层状态决定下层链参数",但上层是随时间切换的离散状态、无监督估计;(ii) IOHMM 有"转移矩阵是输入的函数",但条件变量是逐时刻的;(iii) 基数 Markov 网络 MIL 有"包级正例比例约束实例推断",但没有时间链。计划组件把三者合成一个方向明确的层级:从冻结 VLM 多尺度判定的**分布**(K=30 与 K=4 的直方图/统计量)加池化内容特征回归出一个每视频连续密度 d_v,再让 d_v 同时决定分段链的初始分布和转移矩阵,整个链只靠视频级标签通过 MIL/边际似然训练。需要自己解决的问题至少有:d_v 的参数化(标量还是低维向量)、转移矩阵对 d_v 的函数形式(要保证随机矩阵约束且梯度可用)、前向算法与 MIL 目标的耦合方式、以及防止 d_v 直接从视频级标签泄漏成"视频分类器"而让链失去作用。动机数据(分段 logit 92%/79% 的方差在视频间)是本项目自己的观测,文献里没有对应的"先估密度再条件化链"的设计,所以不是把现成方法换个数据集。

主要来源:MultiHateLoc https://arxiv.org/html/2512.10408v1;LELA https://arxiv.org/html/2602.09637;HateClipSeg https://arxiv.org/html/2508.01712v1;AMC-WSVAD https://www.sciencedirect.com/science/article/abs/pii/S1077314223001789;Hajimirsadeghi UAI 2013 https://arxiv.org/abs/1309.6833;IOHMM 讲义 https://pages.discovery.wisc.edu/~sroy/teaching/network_biology/fall2018/lectures/IOHmms_Lecture10.pdf;Neural HMM https://arxiv.org/abs/2310.02724;Xie et al. HHMM https://www.semanticscholar.org/paper/1f8196633563a381c07326a4375b7c22bd91c303;Bui AAAI 2004 https://cdn.aaai.org/AAAI/2004/AAAI04-052.pdf;BN-WVAD https://arxiv.org/abs/2311.15367;VadCLIP https://arxiv.org/abs/2308.11681;GIG-VAD https://arxiv.org/abs/2104.06813;Shirahama HCRF https://link.springer.com/article/10.1007/s13735-014-0068-6;MI-HMM https://ieeexplore.ieee.org/document/7152896/;LEC-VAD https://arxiv.org/pdf/2506.13095;Temporal label noise https://arxiv.org/abs/2508.04900。


---

# 附录 C

# 文献预查报告:Granularity reliability gate(输入依赖的两粒度 VLM 判决可靠性门)

日期:2026-09-04。只读检索,未改任何文件。

## 1. 检索记录(原文,全部于 2026-09-04 执行)

WebSearch(25 条):
1. `learning from crowds instance-dependent confusion matrix neural network annotator reliability conditioned on input features`
2. `Dawid-Skene with features "annotator" reliability depends on input "neural" crowd layer`
3. `learning when to trust LLM annotations instance-dependent label noise LLM pseudo labels confidence weighting`
4. `adaptive pseudo-label weighting multi-teacher learned teacher weighting weakly supervised video temporal action localization`
5. `multi-granularity multi-scale temporal proposals reliability weighting learned gating weakly supervised temporal localization`
6. `hateful video localization HateMM MultiHateClip HateClipSeg vision-language model segment verdict weakly supervised`
7. `VLM as noisy labeler input-dependent trust gating video moderation pseudo label source reliability network`
8. `"instance-dependent" annotator noise "common noise adaptation layers" OR "CoNAL" OR "Learning from noisy labels by regularized estimation of annotator confusion"`
9. `MultiHateLoc weakly-supervised multimodal hate localisation arxiv`
10. `weak supervision label model instance-dependent labeling function accuracy "features" learned (WeaSEL OR "Snorkel" OR "data programming") end-to-end neural`
11. `mixture of experts gating over multiple pseudo-label sources input-dependent source weighting weakly supervised learning "source reliability" learned from downstream loss`
12. `weakly supervised video anomaly detection VLM pseudo labels reliability "coarse" "fine" granularity weighting learned gate`
13. `"CrowdAttention" instance-dependent reliability scores annotators cross-attention end-to-end`
14. `learning from multiple LLM annotators weighted aggregation learned per-instance weights "LLM annotators" Dawid-Skene features 2025 2026`
15. `adaptive multi-teacher knowledge distillation instance-level teacher weighting learned "AMTML-KD" OR "CA-MKD" OR "adaptive multi-teacher"`
16. `hateful video detection VLM zero-shot verdict segments trust reliability HVGuard RAMF HateMM 2026 localization weakly supervised MLLM evidence`
17. `Markov chain OR CRF temporal potentials learned input-dependent potential weights noisy pseudo-label multiple instance learning video-level labels`
18. `"Learning to trust" OR "when to trust" vision-language model zero-shot pseudo labels learned gating network weak supervision temporal video`
19. `hierarchical coarse-to-fine MLLM anomaly verdicts video "Holmes-VAU" OR "hierarchical" VLM pseudo labels multiple window sizes fusion learned confidence weakly supervised video anomaly detection 2025 2026`
20. `learning from crowds video temporal segment labels annotator reliability instance-dependent weakly supervised video "video-level" labels noisy segment pseudo labels MLLM annotator confidence estimation`
21. `"Beyond confusion matrix" learning from multiple annotators awareness of instance features Machine Learning 2022 Li Sun`
22. `learned confidence network for LLM-generated labels "meta-learning" reweight pseudo-labels from foundation model per-instance weight trained via downstream validation loss weak supervision`
23. `hateful video temporal localization MLLM segment-level verdicts sliding windows different lengths fused confidence 2026 HateClipSeg baseline`
24. `"learning from crowds" OR "multiple annotators" temporal sequence labeling video annotator reliability "instance-dependent" hidden Markov OR "sequence" 2023 2024 2025`
25. `"Revealing Temporal Label Noise in Multimodal Hateful Video Classification" arxiv`

WebFetch(全文/摘要核对):arXiv 2012.13052(CoNAL,PDF 已本地抽文本)、2106.15146、2306.03116、2605.27913(CANE)、2602.09637(LELA,HTML)、2601.11178v3(TANDEM,HTML)、2107.02233(WeaSEL,PDF 已本地抽文本)、2512.10408(MultiHateLoc)、2511.13891、2605.09702、2506.18261、ACL Anthology 2025.emnlp-main.456(HVGuard,PDF 已本地抽文本)、AAAI-26 SafeLens PDF(已本地抽文本)、github.com/autonlab/weasel。

## 2. 最接近的工作(4 篇 + 2 篇领域内对照)

**(a) WeaSEL — "End-to-End Weak Supervision", Rühling Cachay, Boecking, Dubrawski, NeurIPS 2021.** https://arxiv.org/abs/2107.02233
做什么:多个弱标注源(labeling functions)、无真标签。一个 encoder MLP 读入**样本特征 x 与各源的投票 λ(x)**,输出每个源的样本依赖精度 θ(λ,x) = τ2·softmax(e(λ,x)τ1);软标签 = 各源投票按 θ 加权的归一化线性组合;encoder 与下游模型用对称交叉熵(对目标 stop-grad)互相当作目标训练,无任何标签。消融证明"不看特征、只看投票"会掉点。
与本组件差别:(1) 监督信号不同——WeaSEL 完全无标签,靠两网络"一致性"自举;本组件有视频级标签,门权重由 MIL/视频级目标反传得到,不存在 WeaSEL 那类退化解需要靠温度约束的问题。(2) 输出对象不同——WeaSEL 给出一个独立样本的软标签;本组件输出的是 Markov 链中两个判决势(potential)的输入依赖权重,门权重进入序列推断(前向/后向),秒级判决还受相邻秒和转移势约束。(3) 弱源结构不同——WeaSEL 的源是任意异构 LF;本组件的两个源是**同一 VLM 在两个时间粒度**上的判决,门要学的是"粒度可靠性随内容变化"(短窗判决在 HateMM 上假阳率高、在 HateClipSeg 上可信),这是一维的"粒度"变量,不是多源精度向量。(4) 领域:文本/表格分类,没有视频/序列。

**(b) CoNAL — "Learning from Crowds by Modeling Common Confusions", Chu, Ma, Wang, AAAI 2021.** https://arxiv.org/abs/2012.13052
做什么:每个 (实例 i, 标注者 r) 对,一个辅助网络用实例嵌入 v_i 和标注者嵌入 u_r 算 ω_ir = σ(u_r^T v_i),按 ω 在"公共混淆矩阵"与"个体混淆矩阵"两层噪声适配层之间插值;对观测到的众包标签做交叉熵端到端训练,再加 L2 正则强制两层不同。
与本组件差别:CoNAL 的监督是**每个实例上的噪声标签本身**(把噪声标签当预测目标,分类器是潜变量);本组件没有秒级标签,只有视频级标签,门是通过 MIL 目标间接学的。CoNAL 门选的是"噪声来源属于哪个混淆矩阵",不改变标注者的投票被使用的程度;本组件的门直接调节两粒度判决对秒级后验的贡献强度。CoNAL 是 i.i.d. 图像分类,无时序结构。

**(c) "Learning from Multiple Annotators by Incorporating Instance Features" / "Beyond confusion matrix: learning from multiple annotators with awareness of instance features", Li, Sun, Li 等, Machine Learning 2022(arXiv 2106.15146).** https://arxiv.org/abs/2106.15146
做什么:把标注者的噪声转移矩阵做成实例特征的函数(神经网络输出),与分类器联合成一个网络,直接在众包标签上端到端训练。同一谱系还有 Li, Xia, Deng, Ge, Liu, "Transferring Annotator- and Instance-dependent Transition Matrix for Learning from Crowds", TPAMI 2024(https://arxiv.org/abs/2306.03116),用全局到个体、个体到个体的知识迁移解决标注稀疏。
与本组件差别:同 (b),它们建模的是完整的实例依赖转移矩阵 T(x),目标是从**逐实例的多标注者标签**恢复分类器;本组件只需一个标量(或二维)权重表示"此秒此粒度判决应信多少",且没有逐秒标签可作监督,监督来自视频级弱标签经序列模型的边际化。它们没有时间序列、没有多粒度同源标注者的设定。

**(d) CrowdAttention, Sensors 2025.** https://doi.org/10.3390/s25206435
做什么:分类网络的预测分布做 query,各标注者 one-hot 标签做 key/value,cross-attention 输出实例依赖的标注者可靠性权重,产生可靠性加权的伪标签,再对分类网络做交叉熵。
与本组件差别:可靠性权重由"标注者标签与当前模型预测的一致程度"决定,是一种自洽式加权,很容易走向确认偏差;本组件的门读**内容特征与上下文**(不读模型自己对该秒的预测),并由视频级损失校正。同样无序列、无多粒度、有逐实例标签。

**(e) 补充:Where LLM Annotators Fail (CANE), arXiv 2605.27913, 2026.** https://arxiv.org/abs/2605.27913
做什么:指出 LLM 标注错误是"特征空间区域依赖"的,按特征聚类估计簇条件的 LLM 可靠性(无真标签),决定信哪些伪标签、纠哪些。图节点分类。
与本组件差别:可靠性是按簇的离散统计量、非学习网络、不进入序列推断、不区分同一标注者的不同粒度;它的动机("同一 LLM 可靠性随输入区域变化")与本组件一致,可作为动机引用。

**领域内对照(hateful video):**
- **MultiHateLoc (WWW 2026, arXiv 2512.10408)**:弱监督 hate 定位,模态感知时序编码 + 动态跨模态融合 + 模态感知 MIL;**不用 LLM/VLM 判决,不建模任何标注源可靠性,无多粒度加权**。
- **LELA (arXiv 2602.09637, 2026)**:training-free,GPT-4o mini 对每帧五模态字幕打分,**帧级单粒度**,融合为各模态取 max(式 7),**无任何学习组件**;数据 HateMM、MultiHateClip,不含 HateClipSeg。
- **TANDEM (AAAI-ICWSM 2027, arXiv 2601.11178)**:VLM 与音频 LM 直接输出 XML 时间戳,视频切 30 s 块顺序处理,**单一粒度,无可靠性加权/门控**,且**用段级真标签做 SFT + RL**,不是弱监督。
- **HVGuard (EMNLP 2025)**:MoE 门在**拼接的多模态嵌入(含 MLLM 推理文本)上选专家**,只做视频级分类,无时间定位,无"MLLM 输出可靠性"建模。
- **SafeLens (AAAI-26 demo)**:HateClipSeg 上做段级检测,策略 LLM 全监督,单粒度,无可靠性门。
- **Revealing Temporal Label Noise in Multimodal Hateful Video Classification (MUWS@ACM MM 2025, arXiv 2508.04900)**:只分析视频级标签的时间噪声对分类的影响,不建模标注源可靠性。

## 3. 是否已在 hateful video 检测/定位中使用?

**否。** 证据:上面六篇 2025–2027 年的 hateful video 工作(MultiHateLoc、LELA、TANDEM、HVGuard、SafeLens、Temporal Label Noise)没有一篇(i)让同一 VLM 在两个时间粒度给判决,(ii)用输入依赖的学习门决定各粒度判决的可信度,(iii)只用视频级标签训练该门。最接近的 HVGuard MoE 是特征空间上的专家选择,不是判决可靠性;LELA/TANDEM 的 VLM 输出直接当结果,无可靠性学习。检索 6、16、23 三条针对领域的查询均未命中。

## 4. 规则 4 四个 STOP 条件裁定

- **源方法已用于 hateful video**:不成立(第 3 节)。
- **纯 ensemble**:不成立。两粒度判决不是被平均/投票,而是作为 Markov 链的两个势,权重由内容条件的网络给出并经视频级目标学习;固定权重 EM(现有 hier_evidence_mil 的固定噪声率)正是被替换的对象。
- **纯后处理/校准/平滑**:不成立。门在训练图内、与骨干联合训练,改变的是序列模型的势函数,不是对最终分数做事后校准或平滑。
- **纯工程(超参/特征/增广/训练配置)**:不成立。它引入一个新的可学习模块并改变概率模型结构(噪声率从常数变为 x 的函数)。

**结论:四个 STOP 条件均不触发。**

## 5. 适配是否非平凡?

非平凡,但要在写作与消融中把三点说清:(1) 现有实例依赖标注者可靠性方法(WeaSEL、CoNAL、Li et al. 2022/2024、CrowdAttention)全部依赖**逐实例的噪声标签**作监督或自举,本组件只有视频级标签,门权重必须经 Markov 链前向边际化后由 MIL 损失反传,能否学到"HateMM 短窗不可信、HateClipSeg 短窗可信"这种语料差异是一个真实的可识别性问题,需要用 HateMM(.16)/HateClipSeg(.73)的门权重分布直接验证。(2) "同一标注者、不同时间粒度"是新的可靠性轴——先前工作的轴是"不同标注者";粒度可靠性依赖的是局部内容与上下文长度的关系,这是本任务特有的现象驱动。(3) 与 WeaSEL 那类自举方法不同,本组件有外部弱标签锚定,理论上不需要温度约束防退化,但需要消融"门只读内容 vs 门读内容+两粒度判决本身"以及"固定权重 vs 学习权重",否则审稿人会把它读成 WeaSEL 的一个变体。风险点:若门权重在两个语料上学出来接近常数(等价于 EM 固定噪声率),则该组件的贡献就退化为工程调参,预注册时应把"门权重在 HateMM 与 HateClipSeg 上对短窗粒度的均值差异显著"设为必要条件。

Sources: [WeaSEL arXiv](https://arxiv.org/abs/2107.02233), [CoNAL arXiv](https://arxiv.org/abs/2012.13052), [Li et al. instance features](https://arxiv.org/abs/2106.15146), [AIDTM TPAMI](https://arxiv.org/abs/2306.03116), [CrowdAttention](https://doi.org/10.3390/s25206435), [CANE](https://arxiv.org/abs/2605.27913), [MultiHateLoc](https://arxiv.org/abs/2512.10408), [LELA](https://arxiv.org/abs/2602.09637), [TANDEM](https://arxiv.org/abs/2601.11178), [HVGuard](https://aclanthology.org/2025.emnlp-main.456/), [SafeLens](https://ojs.aaai.org/index.php/AAAI/article/download/42390/46351), [Temporal Label Noise](https://arxiv.org/abs/2508.04900), [HateClipSeg](https://arxiv.org/abs/2508.01712), [Holmes-VAU](https://arxiv.org/abs/2412.06171), [AMTML-KD](https://arxiv.org/abs/2103.04062), [Calibrate Don't Curate](https://arxiv.org/abs/2605.09702)


---

# 附录 D

# 文献预检报告:后验引导的视频内对比 (A) 与后验蒸馏 (B)

日期:2026-09-04。只读检索,未改任何文件。WebSearch 20 次、WebFetch 15 次,全部列在下方。

---

## 0. 检索记录(全部 2026-09-04 执行)

WebSearch(逐字):
1. `MACIL-SD modality-aware contrastive instance learning self-distillation weakly supervised audio-visual violence detection CMAL`
2. `CoLA weakly supervised temporal action localization snippet contrast SniCo hard snippet mining`
3. `pseudo-label guided contrastive learning weakly supervised temporal action localization`
4. `teacher-guided instance selection contrastive weakly supervised video anomaly detection`
5. `UR-DMU uncertainty regulated dual memory units weakly supervised video anomaly detection contrastive`
6. `HMM posterior self-training weakly supervised sequence labeling distillation`
7. `CRF posterior teacher distillation structured prediction knowledge distillation`
8. `HateMM MultiHateClip HateClipSeg hateful video localization weakly supervised contrastive distillation`
9. `VLM guided pseudo labels weakly supervised video anomaly detection distillation single modality branch teacher posterior`
10. `HateClipSeg segment-level hateful video dataset 2025 2026`
11. `MultiHateLoc temporal localisation multimodal hate content online videos cross-modal contrastive alignment modality-aware MIL`
12. `DSCIL dynamic selected contrastive instance learning weakly supervised video anomaly detection Pattern Recognition`
13. `SafeLens segment-level hate speech detection online videos AAAI 2026`
14. `hidden Markov model OR Markov chain temporal weakly supervised video anomaly detection posterior pseudo label MIL`
15. `"contrastive" "multiple instance learning" top-k hard instance mining video anomaly detection pseudo label selection`
16. `multimodal fusion teacher distill unimodal branch audio visual text weakly supervised video posterior soft label mutual distillation`
17. `Distilling Aggregated Knowledge for Weakly-Supervised Video Anomaly Detection DAKD 2024`
18. `weakly supervised temporal action localization CRF OR HMM OR "structured" pseudo labels contrastive snippet selection 2024 2025`
19. `weakly supervised action segmentation HMM Viterbi pseudo labels iterative RNN Richard Kuehne fine-to-coarse`
20. `"Distilling Privileged Knowledge for Anomalous Event Detection From Weakly Labeled Videos" audio teacher RGB student`
21. `hateful video detection knowledge distillation OR self-distillation HateMM unimodal branch teacher`
22. `EMA mean teacher self-distillation weakly supervised video anomaly detection snippet pseudo label 2024 2025`
23. `"Weakly Supervised Multimodal Video Anomaly Detection Based on Knowledge Distillation" soft labels teacher audio-guided`
24. `HateMM hateful video localization frame-level MIL baseline HTMM MHCL HVGuard weakly supervised segment`
25. `vision-language model segment verdict potentials Markov chain weakly supervised localization posterior guided contrastive`
26. `Safranchik Luo Bach "Weakly Supervised Sequence Tagging from Noisy Rules" linked HMM posterior train neural network soft labels`
27. `frozen VLM per-segment judgments as unary potentials CRF weakly supervised video localization "vision-language" pseudo labels contrastive selection anomaly`
28. `"hate" video localization HMM OR "Markov" OR "CRF" temporal segment posterior weakly supervised`
29. `Neural Network-Viterbi weakly supervised video learning Richard Kuehne Iqbal Gall CVPR 2018 posterior pseudo labels`
30. `"hateful video" OR "hate video" segment contrastive learning pseudo label MIL weakly supervised 2026`

WebFetch(全文/摘要核对):arxiv 2207.05500(abs、pdf、ar5iv)、MACIL_SD GitHub、arxiv 2512.10408v3(MultiHateLoc)、2601.11178v3(TANDEM)、2602.09637(LELA)、aclanthology 2021.acl-long.46(Structural KD,pdftotext 抽取)、Safranchik AAAI-20 pdf(pdftotext 抽取)+ OJS 页、2403.01169(LAP,abs+html)、1906.01028、2406.02831v1(DAKD)、2408.05955v1(PVLR)、2505.02179(ProDisc-VAD)、2606.11953、2508.01712v1(HateClipSeg)、2508.06570(ImpliHateVid)、ieeexplore 11015429(TFPLG,HTTP 418 失败)。

---

## (A) 后验引导的视频内对比

### A.2 最接近的工作

**1. MACIL-SD — Modality-Aware Contrastive Instance Learning with Self-Distillation (ACM MM 2022)**
[arXiv 2207.05500](https://arxiv.org/abs/2207.05500) | [代码](https://github.com/JustinYuu/MACIL_SD)
经 ar5iv 全文核对:对比对的选择完全来自模型自身的音视频 logits(单模态 logits 之和)。视频级预测 p>0.5 的视频里取 **top-K logits 实例**作 violent semi-bag,p≤0.5 的取 top-K 作 normal semi-bag,整个 mini-batch 的 **bottom-K** 作 background semi-bag;正对 = 音频 violent semi-bag 与视觉 violent semi-bag,负对 = violent semi-bag 与另一模态的 normal/background 实例,InfoNCE,权重 λ(t)=min(0.1t,1.5) 线性增长。
**差异**:MACIL-SD 的选择器是被训练网络自己的分数(自我确认回路);计划方案的选择器是链后验,后验里包含冻结 VLM 的段级 verdict potential 和时间耦合项,两者是不同来源的信号。MACIL-SD 的正负对是跨模态(音频对视觉);计划方案是视频内段级(同一视频内正段对负段)。

**2. CoLA — Snippet Contrastive Learning for WS-TAL (CVPR 2021)**
[arXiv 2103.16392](https://arxiv.org/abs/2103.16392)
用模型自身的 T-CAM/actionness 分数排序取 easy action(top)/easy background(bottom),再用时间膨胀/腐蚀在边界附近挖 hard snippet,SniCo loss 把 hard 拉向 easy。
**差异**:选择器仍是模型自身分数加固定形态学规则;没有外部教师,没有概率时间模型。计划方案的时间耦合是链的转移项,不是膨胀/腐蚀。

**3. LAP — Learn Suspected Anomalies from Event Prompts (arXiv 2403.01169, 2024)**
[arXiv 2403.01169](https://arxiv.org/html/2403.01169)
用字幕与异常事件 prompt 的余弦相似度算每段的"疑似异常"分数 c,以 batch 统计 mean+τ·std 阈值化成伪标签 p,再加一个段级 BCE(L_PAL)和 multi-prompt triplet(L_MPL)。这是我找到的最接近"外部文本模型决定哪些段是正例"的先例。
**差异**:LAP 的伪标签是独立阈值化,每段独立,没有时间耦合,也不与网络证据合成后验;伪标签直接进 BCE,不是用来挑对比对(MIL 的 top-k 仍是模型自己选)。计划方案是"网络证据 + VLM verdict + 转移"的联合后验来挑对比对。

**4. PVLR — Probabilistic Vision-Language Representation for WS-TAL (2024)**
[arXiv 2408.05955](https://arxiv.org/html/2408.05955v1)
冻结 CLIP 做知识蒸馏(对齐 snippet 分布均值),对比样本的选择沿用 CoLA(自身 attention top/bottom-k + 膨胀腐蚀)。
**差异**:VLM 只进表示对齐,不参与选谁进对比;计划方案里 VLM verdict 直接影响选择。

**5. DSCIL — Dynamic Selected Contrastive Instance Learning for WSVAD (Pattern Recognition 2025)**
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320325013196)
动态实例选择模块在异常视频里同时选"可能异常"和"可能正常"的实例进对比学习。
**差异**:选择依据仍是模型自身特征空间/分数;无外部教师、无时间链。

补充:UR-DMU(AAAI 2023,[arXiv 2302.05160](https://arxiv.org/abs/2302.05160))的双记忆单元用自身分数拉开正常/异常表示,不是显式对比对选择;DAKD(WACV 2025)的特征级 InfoNCE 由教师伪标签划分正负子集,但教师是多骨干集成 MIL 网络,不是后验模型(详见 B)。

### A.3 是否已用于 hateful video?
**否。** 证据:
- MultiHateLoc(WWW 2026,[arXiv 2512.10408v3](https://arxiv.org/html/2512.10408v3)):唯一的弱监督 hate 定位方法。其 CM-Contrast 正对 = 同一时间戳不同模态特征,负对 = 错位时间戳或其他视频,**对所有时间步做,不做段选择**;MA-MIL 用各分支 sigmoid 分数 top-K。全文核对:无蒸馏、无 HMM/CRF/Markov、无 VLM 伪标签。
- LELA([arXiv 2602.09637](https://arxiv.org/html/2602.09637)):训练无关的 LLM 定位,无对比无蒸馏。
- TANDEM([arXiv 2601.11178v3](https://arxiv.org/html/2601.11178v3)):RL(GRPO/GSPO)联合训练 VLM+ALM,无对比、无蒸馏、无 Markov。
- HateClipSeg(MM 2025,[arXiv 2508.01712](https://arxiv.org/html/2508.01712v1)):定位任务只跑全监督 ActionFormer。
- ImpliHateVid([arXiv 2508.06570](https://arxiv.org/abs/2508.06570)):视频级分类,跨模态对比。
- SafeLens(AAAI 2026)、MM-HSD(MM 2025)、IARE(SIGIR 2026,[arXiv 2606.11953](https://arxiv.org/html/2606.11953)):均为段/视频分类或 LLM 推理,无上述机制。

### A.4 规则 4 四种 STOP 情形
- 源方法已用于 hateful video:**不适用**(MACIL-SD/CoLA/LAP 均未在 HateMM/MHC/HCS 上用过;MultiHateLoc 的对比是跨模态时间戳对齐,不是选段对比)。
- 纯集成:**不适用**(不是多模型平均)。
- 纯后处理/校准/平滑:**不适用**(后验进训练损失,改变学到的表示;但注意:如果最终消融显示后验只在推理时起作用、训练时的对比损失无贡献,审稿人会把它归为后处理,需要"去掉对比损失"的消融来证明)。
- 纯工程(超参/特征/增强/训练配置):**不适用**(替换了损失的样本选择机制)。
**结论:无 STOP。**

### A.5 改造是否非平凡?
非平凡,但边际。文献里"谁选对比对"有三类:模型自身分数(MACIL-SD、CoLA、DSCIL、UR-DMU)、外部文本相似度独立阈值(LAP)、模型自身分数加固定形态学规则(CoLA/PVLR)。计划方案用一个把网络证据、冻结 VLM verdict 和时间转移联合起来的链后验做选择,这在 WSVAD/WS-TAL/hate 定位文献里没有直接先例;它断开了 MACIL-SD 式"自己选自己确认"的回路,且时间耦合是概率模型内生的而不是膨胀腐蚀。风险点:与 LAP 的 L_PAL 只差"伪标签是否含时间耦合、是否用于挑对比对";写作时必须把这两点作为设计动机并配消融(后验选 vs 自身 top-k 选 vs VLM verdict 独立阈值选)。

---

## (B) 后验蒸馏到单模态分支(可选)

### B.2 最接近的工作

**1. MACIL-SD 的自蒸馏 (ACM MM 2022)**
[ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2207.05500)
核对结果:教师是一个**共同训练的单模态视觉网络**(标准 transformer,BCE 视频级损失,小学习率 8e-5),学生是音视频模型;知识通过 **EMA 参数融合** θ_av ← mθ_av + (1−m)θ_v 传递,m 按 cosine 从初值升到 1;没有 KL/MSE 蒸馏损失。方向是"单模态→多模态"。
**差异**:计划方案方向相反(多模态链后验→各单模态分支),教师不是网络参数而是后验分布(含 VLM verdict 与转移),传递方式是输出层蒸馏而不是参数 EMA。

**2. Structural Knowledge Distillation (Wang et al., ACL 2021)** + **Structure-Level KD for Multilingual Sequence Labeling (ACL 2020)**
[ACL 2021](https://aclanthology.org/2021.acl-long.46/) | [arXiv 2004.03846](https://arxiv.org/pdf/2004.03846)
pdftotext 核对:"Posterior KD" 用 CRF 教师的单位置边缘后验训练学生,"Struct. KD" 用相邻标签对的边缘,还有 Top-K 序列版本;有无标注数据的实验(学生可超教师)。这是"链模型后验作为蒸馏目标"的正规先例。
**差异**:NLP 全监督/半监督序列标注,教师是训练好的 CRF;计划方案教师是弱监督下由网络证据+VLM verdict 合成的链后验,学生是三个模态分支,目标是提升单分支证据质量再反馈给链。

**3. Linked HMM — Weakly Supervised Sequence Tagging from Noisy Rules (AAAI 2020)**
[AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/6009)
pdftotext 核对:弱规则→linked HMM 估计后验→用"noise-aware loss"(相对后验的期望对数似然,含 unary 与 pairwise marginals)训练 BiLSTM-CRF。这是"HMM 后验做软标签自训练"的弱监督先例。
**差异**:文本域,规则固定不学习;HMM 与神经网络两阶段而非闭环;无多模态分支。

**4. RNN-HMM / NeuralNetwork-Viterbi (Richard, Kuehne, Gall, CVPR 2017/2018; TPAMI 2019)**
[CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Richard_Weakly_Supervised_Action_CVPR_2017_paper.pdf) | [arXiv 1906.01028](https://arxiv.org/abs/1906.01028)
HMM Viterbi 对齐产生帧级伪标签,RNN 用帧级交叉熵训练,交替迭代。视频域"结构化推断→帧级自训练"的经典先例。
**差异**:监督是动作转录(有序标签序列),HMM 有语法约束,伪标签是硬 Viterbi 路径;计划方案是仅视频级二值标签、软后验、多模态分支。

**5. DAKD — Distilling Aggregated Knowledge for WSVAD (WACV 2025)**
[arXiv 2406.02831](https://arxiv.org/html/2406.02831v1)
多骨干(I3D/S3D/CLIP)聚合教师→单骨干学生,预测级用教师伪标签的 CE,特征级用 InfoNCE。
**差异**:教师是 MIL 网络的集成,不是后验模型;学生是单骨干视觉,不是三模态分支。类似的还有 Privileged KD(TNNLS 2023,[IEEE](https://ieeexplore.ieee.org/document/10098140/)):音视频教师→RGB 学生;ChinaCom 2024 的 soft-label logits 蒸馏([Springer](https://link.springer.com/chapter/10.1007/978-3-032-03215-7_7))。

### B.3 是否已用于 hateful video?
**否。** 检索 21 与全文核对显示:hate 领域的蒸馏只有 meme/文本方向(DDML 多学生蒸馏、hateful meme KD),视频方向 HateMM/MHC/HCS 上没有任何"融合或后验教师→单模态分支"的工作;MultiHateLoc 明确无蒸馏。

### B.4 规则 4 四种 STOP 情形
- 源方法已用于 hateful video:**不适用**。
- 纯集成:**不适用**(蒸馏改变分支参数)。
- 纯后处理/校准/平滑:**不适用**,但有边界风险——如果后验教师只把链的平滑结果回灌给分支、而分支输出只在推理时进链,审稿人可能视之为"把后处理蒸进网络"。需要消融证明蒸馏后单分支在独立评测上提升,且链最终 pooled AP/ROC 提升由此导致。
- 纯工程:**不适用**。
**结论:无 STOP,但 (B) 若单独作为贡献,与 Posterior KD(ACL 2021)和 Linked HMM 的 noise-aware loss 在形式上高度重合;建议只作为 (A) 的配套机制并做消融,不单列为 novelty。**

### B.5 改造是否非平凡?
中等。把链后验作为软目标蒸馏到分支,损失形式(逐段 KL/期望对数似然)在 Posterior KD 和 Linked HMM 里已有;新的部分是:教师后验不是外部训练好的模型,而是学生自身证据与冻结 VLM verdict 的联合;三个分支同时接收同一后验(多模态自训练闭环);后验随分支更新而变化,需要处理自增强(confirmation)风险——这一点在自训练文献里有明确对策(温度、EMA 教师、置信过滤),也是用户记忆里两次抓到的"证据认定步=原判断本身"循环的风险所在。若要做,设计期必须说明 VLM verdict 作为固定外部锚如何阻断这个循环,并给出"去掉 VLM potential 后蒸馏是否退化"的消融。

---

## 总结
- (A) 和 (B) 都未在 HateMM/MultiHateClip/HateClipSeg 上出现;规则 4 四种 STOP 均不触发。
- (A) 最近的先例是 MACIL-SD(自身 top-k 选)和 LAP(文本相似度独立阈值选);差异在"联合后验含时间耦合与 VLM verdict 做选择"。
- (B) 最近的先例是 Structural/Posterior KD(ACL 2021)、Linked HMM noise-aware loss(AAAI 2020)、RNN-HMM Viterbi 自训练;差异在弱监督多模态闭环与 VLM 锚;单独 novelty 偏弱,建议作为 (A) 的配套,并用消融防"后处理蒸进网络"的质疑。
- MACIL-SD 的自蒸馏方向是视觉单模态→音视频、机制是 EMA 参数融合而非损失蒸馏(已核对全文),与计划方案的教师定义不同,写 related work 时应准确表述。
