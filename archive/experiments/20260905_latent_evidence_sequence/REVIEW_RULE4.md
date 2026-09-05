# 候选6独立 proposal review（规则4）

截至 2026-09-05；审查对象为本目录 README 初始提案，以及主 agent 明确提出的“两个粒度拼成单个8维高斯、观测损失按8T归一化”修订方向。评审人：独立 agent `proposal_review_c6`。本次只作一次规则4评审，不实现代码、不运行实验。

## 结论：GO，但必须修正数学叙述与模块证据对应

实际检索未发现所审具体组合——单一冻结VLM四种内容输入的连续联合观测密度、内容条件状态转移、视频是否出现目标事件的序列约束联合训练——已发表于 hateful video detection/localization。这个结论是本次检索范围内“未发现”，不是证明文献不存在，也不授予已成立的三模块 novelty。

四个 STOP 条件均未命中：没有查到具体来源方案已用于本任务；不是多个独立模型预测的 ensemble；状态模型从训练开始定义最终预测，不是对既有分数的纯后处理；改变监督、概率模型与推断，不是仅换特征/调参的工程技巧。高斯不合适、前景可能不可识别、转移可能恒定、长序列事件饱和等风险不作为 STOP 理由，交完整实验判断。

允许继续实现修订后的候选；下述修正属于科学定义和贡献边界，不增设 smoke、理论门或额外 proposal review。

## 1. 实际文献检索与来源边界

检索包括 `hateful video hidden Markov model`、`hate video hidden Markov`、`hateful video Gaussian Markov`、`hateful video hidden state`、`HateMM Markov`、`HateClipSeg Gaussian`、`hate localization Markov likelihood`、`input-output HMM Bengio Frasconi`、`weakly supervised action segmentation forward loss`，并核对相关论文方法内容。未把搜索无结果等同于绝对首创。以下只引用一手论文或作者资源。

### 迁移来源已有，不能声称通用数学首创

- **Input-output HMM** 已明确将隐状态序列模型条件化于输入序列，不能把输入条件转移本身称为新概率模型。来源：[Bengio 与 Frasconi, Input-output HMMs for sequence processing, 1996](https://pubmed.ncbi.nlm.nih.gov/18263517/)。
- **视频弱监督HMM与观测密度学习** 已用于动作分割：利用动作顺序而非逐帧标签训练，观测模型采用GMM。与此提案二值视频事件监督不同，但“弱标注训练HMM定位”不是新范式的充分理由。来源：[Kuehne、Richard、Gall, Weakly supervised learning of actions from transcripts](https://arxiv.org/abs/1610.02237)。
- **约束路径求和与神经序列模型联合训练** 已用于弱监督动作分割；CDFL区分合法和不合法标注路径并以递归求和训练HMM/GRU。不能把“对所有合法序列求和”“logadd动态规划”本身作为发明。来源：[Li 等, Weakly Supervised Energy-Based Learning for Action Segmentation, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_Weakly_Supervised_Energy-Based_Learning_for_Action_Segmentation_ICCV_2019_paper.html)。
- **仅已出现动作集合的弱监督** 也已有HMM工作，监督比有序transcript更接近“某类至少出现一次”，但其Viterbi生成伪GT流程不同于本提案精确事件概率训练。来源：[Anchor-Constrained Viterbi for Set-Supervised Action Segmentation, 2021](https://arxiv.org/abs/2104.02113)。

### hateful video 的相邻占位

- **MultiHateLoc** 已有时间编码、动态跨模态融合和视频标签MIL；其第3节采用Transformer/跨模态注意力/Top-K目标，而非此处连续发射与事件约束状态模型。因此“首个视频弱监督局部定位”不可主张。来源：[MultiHateLoc 方法原文](https://arxiv.org/html/2512.10408v3)。
- **RAMF、MARS、MATCH** 已使用支持/反对hate解释的多角度语义证据；不能泛称首次引入证据/反证据或VLM推理与融合。它们的文本理由/多agent验证不同于对同一VLM内容删减输入的连续联合密度建模。来源：[RAMF](https://arxiv.org/html/2512.02743v2)、[MARS](https://arxiv.org/html/2601.15115v1)、[MATCH 作者论文](https://jianlang.org/papers/MATCH.pdf)。
- **CLARA** 已有clip级多模态表示、局部/全局目标与VLM理由引导，故“clip级VLM引导的统一视频模型”也是过宽表述。其MoE/contrastive/gated Transformer方案不同于本提案。来源：[CLARA](https://arxiv.org/html/2608.15905v1)。
- **LELA** 已做多阶段LLM推理和组合匹配的training-free定位；**TANDEM** 已做结构化仇恨判断/时间定位及SFT/RL训练。不能声称首次端到端结构化hate定位或首次多阶段语义融合。它们不是此处连续观测IOHMM与精确视频事件边缘化来源。来源：[LELA 作者预印本](https://arxiv.org/abs/2602.09637)、[TANDEM](https://arxiv.org/html/2601.11178v1)。
- 也检查了任务数据集论文：[HateClipSeg](https://arxiv.org/html/2508.01712v2)。在其方法/任务内容及上述主要相邻来源中未找到本提案对应的高斯状态发射与事件约束训练方案。对CLARA、RAMF、HCS、TANDEM、MARS全文作Markov关键词核查未命中；关键词核查仅为辅助，不代替方法比较。

因此可检验的窄主张是：将相关连续VLM观测、内容驱动状态变化与视频事件训练放入同一任务概率模型后，是否改善局部hateful证据学习。不是重新发明HMM、Gaussian、MIL的存在性语义或forward–backward。

## 2. 必须修正的数学问题

初稿对两个四维密度取log均值，相当于

`q(o30,o4|s) = sqrt(p30(o30|s) p4(o4|s))`。

这是8维空间上的未归一化势。一般 `∫q do30 do4 != 1`，遗漏的归一化量依赖状态与可学习协方差，不能视为对参数无影响的常数。即使对状态路径归一化后可定义 `p(s|o,x)`，`Z=sum_s weight(s,o|x)` 也不能直接称为归一化观测密度；`-log Z/T` 不因此成为观测NLL。这不是“可能退化”的猜测，而是初稿等式的确定性解释错误。

主 agent 提议的修正有效：固定train统计标准化后，把两粒度各四个logits拼成 `o_t∈R^8`，每个状态使用一个正规化8维高斯 `N(o_t; μ_s, Σ_s)`，`Σ_s=L_s L_s^T`，Cholesky正对角并保留完整logdet与常数项。这样行归一化转移、归一化初始概率及归一化发射构成 `p(s,o|x)`；状态求和的 `Z=p(o|x)` 确有生成密度含义。

对应损失明确写为

`L = -log p(y|o,x) - log p(o|x)/(D*T)`，完整模型 `D=8`。

这是**判别事件损失+按维度/长度归一化生成损失的混合目标**，不是未经加权的 `-log p(y,o|x)`。两项固定等权及D/T归一化仍是设计选择，不能称无超参数。仅av消融若使用两个粒度的av则D=2，需要按自身D归一化并记录；不要丢一粒度同时更改输入比较。train标准化引入的固定Jacobian对同一臂的参数优化无影响；不同观测维度的NLL绝对值不应直接横比。

8维高斯能表示同token跨粒度/跨内容输入相关性，但不能消除跨token重复窗口所造成的观测依赖。时间对齐插值后的向量是模型定义的观测，不宜宣称它恢复了真实窗口独立生成机制；这是模型假设和后续诊断，不作阻断。

三状态自动机需严格编码“从未出现正状态背景 / 当前正状态 / 已出现正状态背景”。初始正状态必须进入已命中事件集合；正状态可以返回已出现背景并再次进入正状态。padding不能改变事件或T；无标签test边缘化必须覆盖所有路径。如此计算logZ0/logZ1无需用近等量相减，属于正确实现已有精确求和算法。

## 3. 三模块贡献应怎样对应消融

| 模块 | 可以检验的主张 | 对照与解释限制 |
|---|---|---|
| M1 | 建模8维VLM观测的状态条件相关性有用 | `diagonal_emission` 保留同8维观测、同标准化、同转移/目标/归一化，仅去协方差非对角项。它同时去跨内容和跨粒度相关性，因此只证明联合相关性建模，不分别证明两者。`full_input_emission` 是输入删减对照，不替代这个主对照。 |
| M2 | 内容条件转移优于静态状态转移 | `static_transition` 只令2×2转移时间共享，保留相同内容条件初始概率、发射、事件目标与后验输出。不要同步删初始内容支路、改损失或更换推断。该对照支持这条转移支路的贡献，不证明“内容只能用于转移”优于所有content-emission设计。 |
| M3 | 用精确视频事件概率训练优于top-k弱监督读出 | 当前 `independent_state` 不够：它保留了事件约束与精确边缘化，只移除上一状态依赖。应增加 `event_to_topk` 主对照：同完整状态模型、同生成NLL、同forward–backward后验，仅将事件NLL替换为后验Top-K池化视频BCE。Top-K只用于训练视频损失，test仍输出原局部后验，不添加推理后处理。 |

M3建议的Top-K divisor应在开跑前固定为既有16并记录；这不是为主方法增加搜索。若采用其它事件损失对照，须同样只改变要归因的训练机制。`independent_state` 可保留作结构依赖辅助对照；`no_observation_likelihood` 只隔离生成项，不是事件约束的替代消融。不能将两者的提升重复分配给M2/M3，或从其结果宣称forward–backward算法本身novel。

三个对应机制仍分别需要两语料三seed满足规则14(g)才可作为novelty主张；本评审不保证有效。最强baseline+同输入依然是最终声明缺口，不能仅因缓存复用而略去。无需先增加任何新理论或运行前实验门。

## 4. 执行边界

按以上定义修正README并实现后进入既定一次code review；两语料同架构/损失/推断，validation仅选trial内checkpoint，test(AP+ROC)/2选trial，完整固定Optuna预算。记录固定设计参数，不把“只搜索lr/dropout/max_seqlen”表述成方法无超参数。训练不可使用test标签、test统计或test条件化后验。

本评审结论为 **GO（修正定义后按规则推进）**，不触发STOP，不增加评审轮次；“整体novel paradigm”与三模块独立贡献仍待实证。
