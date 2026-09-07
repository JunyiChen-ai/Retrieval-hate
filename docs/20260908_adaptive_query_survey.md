# 自适应 VLM 查询模块：文献调研（2026-09-08）

依据：`experiments/20260907_c3_rev3_interval_evidence/README.md` 第 1、7.3、8 节。所有引文经 arXiv API / ACL Anthology / NeurIPS 官网 / Springer 核对题目、第一作者、年份与会议；未能核对的标"未核实"。完整书目见文末表格，正文只写短名与 arXiv 号。

## 1. 六个相关方向

**1.1 主动特征获取 / 动态特征选择。** Shim 等（NeurIPS 2018，1709.05964）把"逐个买特征、何时停止并分类"建模为 RL，联合训练分类器与策略，用集合编码表示已获取子集。Janisch 等（AAAI 2019，1711.07364）用深度 Q 学习做同一问题，动作含"终止并预测"。EDDI（Ma 等，ICML 2019，1809.11142）不用 RL：训练一个 Partial VAE 对缺失特征建模，每步选信息增益（对目标的互信息）最大的特征，贪心获取。Li & Oliva（ICML 2021，2010.02433）用生成式代理模型同时估算缺失特征分布和获取价值。Covert 等（ICML 2023，2301.00557）证明贪心最大化条件互信息可以用两个网络实现：预测器在随机掩码子集上训练，选择器学习预测"加哪个特征让预测器输出变化最大"，无需显式互信息估计；DIME（Gadgil 等，ICLR 2024，2306.03301）改为直接回归条件互信息，得到可解释的每步价值并支持变预算停止。与我们的差别：这些方法的特征是无结构的标量向量、目标是单个分类标签、缺失分布靠 VAE 或掩码训练近似；我们的"特征"是带区间的 0/1 裁定，缺失下的后验由区间 HMM 精确计算，输出是每秒密集分数，监督只有视频标签。

**1.2 用 LLM/VLM 迭代选帧做长视频理解。** VideoAgent（Wang 等，ECCV 2024，2403.10517）让 LLM 先看少量帧的字幕，判断是否足以回答，不够就检索新帧，平均 8.4 帧作答。VideoAgent（Fan 等，ECCV 2024，2403.11481）改为记忆库加工具调用。TraveLER（EMNLP 2024，2404.01476）分"遍历-定位-评估-回放"多 LMM 角色迭代。VideoTree（CVPR 2025，2405.19209）按问题相关度对聚类树自适应展开，粗到细取帧；LVNet（2406.09396，预印本）用分层关键帧选择器；Frame-Voyager（ICLR 2025，2410.03226）用 Video-LLM 损失给帧组合排序作监督训练查询器；DrVideo（2406.12846，预印本）把视频转成文档后做检索增强。AKS（CVPR 2025，2502.21271）按相关度与覆盖度选关键帧。"Sequential frame selection"未找到同名论文，未核实。与我们的差别：这些工作的目标是回答一个问题（视频级输出），选帧标准是"够不够回答"，由 LLM 自评或检索相似度决定，没有可训练的骨干与之交互，也没有对缺失观测的概率模型。

**1.3 LLM 级联 / 预算查询。** FrugalGPT（Chen 等，2023，2305.05176）用打分器决定是否升级到更贵的模型。Yue 等（ICLR 2024，2310.03094）用答案一致性触发升级；Gupta 等（2404.10136，预印本）研究 token 级不确定性作级联信号；AutoMix（NeurIPS 2024，2310.12963）用小模型自验证加 POMDP 路由。与我们的差别：级联是"同一输入问哪个模型"，我们是"同一模型问哪些片段"；级联的决策信号是当前答案的置信度，我们的是骨干输出的反事实变化。

**1.4 主动学习获取函数。** 期望模型变化 / 期望梯度长度：Settles & Craven（EMNLP 2008，ACL D08-1112）。BALD：Houlsby 等（2011，1112.5745），选预测分布与参数后验互信息最大的样本。期望误差缩减：Roy & McCallum（ICML 2001）用采样估计标注后测试误差的下降。贝叶斯实验设计：Foster 等 DAD（ICML 2021，2103.02438）把序贯设计策略摊销到网络；Rainforth 等（Statistical Science 2023，2302.14545）综述期望信息增益。与我们的差别：主动学习是训练期挑样本改参数，我们是推断期挑观测改单个视频的输出，且候选价值需要在缺失裁定的预测分布下加权——这正是 EDDI/DIME 与经典 AL 的分界。

**1.5 学习拒答 / 选择性预测；主动弱监督。** Madras 等（NeurIPS 2018，1711.06664）与 Mozannar & Sontag（ICML 2020，2006.01862）学习"什么时候把样本交给专家"，一次性、单专家、视频级决策。主动弱监督：Active WeaSuL（Biegel 等，ICLR 2021 workshop，2104.14847）用主动学习查真标签修正标签模型；Interactive Weak Supervision（Boecking 等，ICLR 2021，2012.06046）让人判断候选标注函数是否有用；Nemo（Hsieh 等，VLDB 2022，2203.01382）引导用户给哪些样本写标注函数。与我们的差别：这些工作问的是人类真标签或标注函数质量，用于训练集；我们的标签模型（区间 HMM）本身是被查询的对象，查询发生在每个测试视频内部。

**1.6 预算下对 VLM/oracle 的粗到细时间查询。** VideoTree、ZoomV（ACM MM 2026，2504.01407）、T*（CVPR 2025，2504.02259）都做"先粗后细"的时间缩放，但目标是问答或找针（needle）。视频异常检测侧：LAVAD（CVPR 2024，2404.01014）训练-free 用 LLM 给字幕打分并做时间聚合；Holmes-VAU（CVPR 2025，2412.06171）的 ATS 用异常打分器决定 MLLM 看哪些帧；VADTree（NeurIPS 2025，2510.22693）用事件边界建粗细层次树，逐节点问 VLM，再融合多粒度分数；VIBES（2604.23724，预印本）用贝叶斯运动模型触发 VLM 只看可疑片段。Heilbron 等（ECCV 2018）研究动作定位的主动标注，属训练期。未找到"按视频自适应决定 VLM 调用次数并有停止规则"的弱监督定位工作。与我们的差别：以上层次查询的展开规则是固定阈值或事件边界，不由训练后的定位骨干反事实决定，也不报告"调用数 vs 定位 AP"曲线。

## 2. 最接近的先例与审稿人视角

1. **Covert 等 2023 + DIME 2024**。可指"重用"：随机掩码训练预测器 = 我们的证据 dropout；"加哪个特征输出变化最大"= 我们的反事实增益；变预算停止 = 我们的阈值。真正不同：他们的缺失分布是无结构掩码，我们用区间 HMM 精确给出未问窗口的预测概率和后验；价值定义在每秒密集输出而非单标签；候选有粗/细层次。
2. **EDDI 2019**。可指：用生成模型对缺失观测积分再算价值。不同：EDDI 需要训练 VAE 作代理，我们的 HMM 7 参数、推断精确，且价值以骨干输出而非信息增益衡量。
3. **Shim 2018 / Janisch 2019**。可指：停止动作与两阶段（先获取再训练）。不同：我们不做 RL，策略是无参贪心，训练期预算靠两轮重训。
4. **VideoAgent（Wang 2024）**。可指：先看少量、不够再问。不同：判断者是训练过的定位骨干而非 LLM 自评，输出是逐秒分数。
5. **VADTree / Holmes-VAU ATS**。可指：粗到细问 VLM 做异常定位。不同：它们的展开由边界检测或固定打分器决定，无交互、无停止、无每视频预算报告。

## 3. 建议

不建议整体照搬某篇方法；建议在自有设计上明确借用两处机制并作为对照：(a) 把 Covert 2023 的"学一个选择器回归输出变化"作为我们两次反事实前向的摊销替代（30 候选 × 2 次前向，骨干便宜，先跑反事实版本，选择器版本作消融）；(b) 论文写法上把模块定位为"结构化观测上的动态特征选择"，引 DIME 的条件互信息视角解释停止阈值。理由：这些方法在无结构特征上已证明贪心可行，我们的贡献点必须放在"HMM 精确缺失推断 + 每秒输出价值 + 粗细层次"三处，而不是贪心本身。风险须先记：README 第 8 节显示不训练时 HateMM 只用 4 粗块的 HMM 后验已超过 34 次全观测，所以模块的价值只能通过训练后的骨干体现，对照必须含"4 粗块 + 训练"臂。

## 4. 查新应核对的引文

| 短名 | 题目 | 第一作者 | 会议/年份 | 标识 | 核实 |
|---|---|---|---|---|---|
| Shim 2018 | Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding | Hajin Shim | NeurIPS 2018 | arXiv 1709.05964（预印本题目不同） | 已核实 |
| Janisch 2019 | Classification with Costly Features using Deep Reinforcement Learning | Jaromír Janisch | AAAI 2019 | arXiv 1711.07364 | 已核实 |
| EDDI | EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE | Chao Ma | ICML 2019 | arXiv 1809.11142 | 已核实 |
| Li & Oliva | Active Feature Acquisition with Generative Surrogate Models | Yang Li | ICML 2021 | arXiv 2010.02433 | 已核实 |
| Covert 2023 | Learning to Maximize Mutual Information for Dynamic Feature Selection | Ian Covert | ICML 2023 | arXiv 2301.00557 | 已核实 |
| DIME | Estimating Conditional Mutual Information for Dynamic Feature Selection | Soham Gadgil | ICLR 2024 | arXiv 2306.03301 | 已核实 |
| VideoAgent-W | VideoAgent: Long-form Video Understanding with Large Language Model as Agent | Xiaohan Wang | ECCV 2024 | arXiv 2403.10517; DOI 10.1007/978-3-031-72989-8_4 | 已核实 |
| VideoAgent-F | VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding | Yue Fan | ECCV 2024 | arXiv 2403.11481 | 已核实 |
| TraveLER | TraveLER: A Modular Multi-LMM Agent Framework for Video Question-Answering | Chuyi Shang | EMNLP 2024 | arXiv 2404.01476 | 已核实 |
| VideoTree | VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos | Ziyang Wang | CVPR 2025 | arXiv 2405.19209 | 已核实 |
| LVNet | Too Many Frames, Not All Useful: Efficient Strategies for Long-Form Video QA | Jongwoo Park | 预印本 2024 | arXiv 2406.09396 | 已核实（会议未核实） |
| Frame-Voyager | Frame-Voyager: Learning to Query Frames for Video Large Language Models | Sicheng Yu | ICLR 2025 | arXiv 2410.03226 | 已核实 |
| DrVideo | DrVideo: Document Retrieval Based Long Video Understanding | Ziyu Ma | 预印本 2024 | arXiv 2406.12846 | 已核实（会议未核实） |
| AKS | Adaptive Keyframe Sampling for Long Video Understanding | Xi Tang | CVPR 2025 | arXiv 2502.21271 | 已核实 |
| SeViLA | Self-Chained Image-Language Model for Video Localization and Question Answering | Shoubin Yu | NeurIPS 2023 | arXiv 2305.06988 | 已核实 |
| TimeSearch-R | TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification RL | Junwen Pan | 预印本 2025 | arXiv 2511.05489 | 已核实（原版 TimeSearch 未核实） |
| ZoomV | ZoomV: Temporal Zoom-in for Efficient Long Video Understanding | Yuan Zhang | ACM MM 2026 | arXiv 2504.01407 | 已核实 |
| T* | T*: Re-thinking Temporal Search for Long-Form Video Understanding | Jinhui Ye | CVPR 2025 | arXiv 2504.02259 | 已核实 |
| FrugalGPT | FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance | Lingjiao Chen | 预印本 2023 | arXiv 2305.05176 | 已核实 |
| MoT cascade | Large Language Model Cascades with Mixture of Thoughts Representations for Cost-efficient Reasoning | Murong Yue | ICLR 2024 | arXiv 2310.03094 | 已核实 |
| Gupta 2024 | Language Model Cascades: Token-level uncertainty and beyond | Neha Gupta | 预印本 2024 | arXiv 2404.10136 | 已核实 |
| AutoMix | AutoMix: Automatically Mixing Language Models | Pranjal Aggarwal | NeurIPS 2024 | arXiv 2310.12963 | 已核实 |
| Settles 2008 | An Analysis of Active Learning Strategies for Sequence Labeling Tasks | Burr Settles | EMNLP 2008 | ACL D08-1112 | 已核实 |
| BALD | Bayesian Active Learning for Classification and Preference Learning | Neil Houlsby | 2011 | arXiv 1112.5745 | 已核实 |
| Roy 2001 | Toward Optimal Active Learning through Sampling Estimation of Error Reduction | Nicholas Roy | ICML 2001 | dblp conf/icml/RoyM01 | 已核实 |
| DAD | Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design | Adam Foster | ICML 2021 | arXiv 2103.02438 | 已核实 |
| Rainforth 2023 | Modern Bayesian Experimental Design | Tom Rainforth | Statistical Science 2023 | arXiv 2302.14545 | 已核实 |
| Madras 2018 | Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer | David Madras | NeurIPS 2018 | arXiv 1711.06664 | 已核实 |
| Mozannar 2020 | Consistent Estimators for Learning to Defer to an Expert | Hussein Mozannar | ICML 2020 | arXiv 2006.01862 | 已核实 |
| Active WeaSuL | Active WeaSuL: Improving Weak Supervision with Active Learning | Samantha Biegel | ICLR 2021 WSL workshop | arXiv 2104.14847 | 已核实 |
| IWS | Interactive Weak Supervision: Learning Useful Heuristics for Data Labeling | Benedikt Boecking | ICLR 2021 | arXiv 2012.06046 | 已核实 |
| Nemo | Nemo: Guiding and Contextualizing Weak Supervision for Interactive Data Programming | Cheng-Yu Hsieh | VLDB 2022 | arXiv 2203.01382 | 已核实 |
| LAVAD | Harnessing Large Language Models for Training-free Video Anomaly Detection | Luca Zanella | CVPR 2024 | arXiv 2404.01014 | 已核实 |
| Holmes-VAU | Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity | Huaxin Zhang | CVPR 2025 | arXiv 2412.06171 | 已核实 |
| VADTree | VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity-Aware Tree | Wenlong Li | NeurIPS 2025 | arXiv 2510.22693 | 已核实 |
| VIBES | Zoom In, Reason Out: Efficient Far-field Anomaly Detection in Expressway Surveillance Videos via Focused VLM Reasoning Guided by Bayesian Inference | Xiaowei Mao | 预印本 2026 | arXiv 2604.23724 | 已核实 |
| VERA | VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models | Muchao Ye | CVPR 2025 | arXiv 2412.01095 | 已核实 |
| Heilbron 2018 | What do I Annotate Next? An Empirical Study of Active Learning for Action Localization | Fabian Caba Heilbron | ECCV 2018 | DOI 10.1007/978-3-030-01252-6_13 | 已核实 |
