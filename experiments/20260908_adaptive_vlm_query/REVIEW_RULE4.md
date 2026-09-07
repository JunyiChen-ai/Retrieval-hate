# 规则 4 复核：自适应 VLM 查询模块（2026-09-08）

审稿人：独立 agent（Claude Fable 5.1），只读；依据 `RESEARCH_ITERATION_RULES.md` 第 4 条（只挡四类：来源方法已用于 hateful video 检测/定位、纯 ensemble、纯校准/后处理/平滑、纯工程技巧），不评估预期性能。

读过：本目录 `README.md` 第 0–3 节、`model.py`、`acquire.py`、`train.py`、`search.py`、`launch/run_search.sh`；`docs/20260908_adaptive_query_survey.md`；`src/hier_evidence_common.py`（`TrainDataset.mask_sampler`、`EvalDataset.masks`、`make_masked_scaffold_fn`、`fit_hmm`）、`src/interval_evidence_hmm.py`（`_emissions`、`fit`、`predictive_fine`、`summarize_gamma`）；`experiments/20260907_c3_rev3_interval_evidence/README.md` 第 1、7.3、8 节；`experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/README.md` 第 2 节（P2）；`research-wiki/STATUS.md`。

## 结论：PASS（放行），但下面第 4 节的 6 项协议修正必须在第一个 trial 开跑前写进 README

不落在四个阻断类别里。它是把动态特征选择（Covert 2023 / DIME 2024 一支）的贪心获取框架迁移到"一个冻结 VLM 的区间裁定"这种观测上，观测模型、缺失推断、输出形式、监督形式都换了，不是重新实现。规则 4 允许"从其他任务迁移的方法"。

## 1. 四个阻断类别逐条核对

**(a) 来源方法已用于 hateful video 检测/定位？没有。** 实际检索（2026-09-08，WebSearch，三组关键词：hateful video + adaptive querying / active acquisition / dynamic feature selection；HateMM / HateClipSeg + coarse-to-fine VLM query）。命中的 hateful video 工作：
- LELA，arXiv 2602.09637（training-free，五模态字幕 + 多阶段 prompting 给**每一帧**打分，HateMM / MultiHateClip）：不训练定位网络、对全部帧问、无预算、无停止规则、不报调用数曲线。
- CLARA，arXiv 2608.15905（clip 级 MoE 编码器 + VLM rationale 经门控 Transformer 融合，视频级分类）：rationale 对每个 clip 取，无自适应查询。
- 本仓库自己的回放（修订 3 README 第 8 节）：只用 HMM 后验、无骨干、无停止规则。
- HateClipSeg 论文（ACM MM 2025）、MultiHateLoc、Reasoning-Aware Multimodal Fusion（2512.02743）：都是固定输入、无交互式查询。
没有任何 hateful video 工作做"逐视频、由训练后的定位网络决定问哪几个窗口、带停止规则、报调用数–AP 曲线"。调研文档第 1.6 节结论成立。

**(b) 纯 ensemble？不是。** 一个骨干、一个 HMM（修订 3 已放行的组合）、一个冻结 VLM。EOC 的两次反事实前向是同一个模型在两种输入下的输出，不是模型组合。五 crop 均值沿用 MACIL-SD baseline 的做法。

**(c) 纯校准/后处理/平滑？不是。** 停止规则决定"再问不问"，改变的是骨干的输入（哪些裁定可见），不动骨干输出的分数；tau、B_max 不按语料改。

**(d) 纯工程技巧？不是。** 三处有机制主张的改动：训练分布（证据 dropout 于允许集合 + HMM 缺失重算）、编码器状态（"没问过"单独一格）、推断流程（骨干驱动的贪心获取 + 两轮训练预算）。搜索空间与修订 3 相同的 5 个标量，方法级新超参（b_max、seed 窗、prefix_mix、tau 网格）全部固定不搜。

## 2. 与 Covert 2023 / DIME / EDDI 的精确对照（论文写作必须按这个说）

**相同的（不能当 novelty 写）：**
1. 预测器在随机掩码子集上训练 = 我们的证据 dropout（Covert 2023、DIME 的预测器训练；EDDI 的 Partial VAE 也用随机掩码）。
2. 贪心逐个选、每步价值 = "揭示这个观测后预测器输出的期望变化，期望在缺失值的预测分布下取"。这正是 DIME 要摊销估计的目标量（期望 KL）、EDDI 的信息奖励。我们用的是逐秒 sigmoid 输出的 L1 均值而不是 KL/条件互信息，是同一量的一个变体，不是新准则。
3. 价值低于阈值就停 = DIME 的变预算停止。
4. **"没问过"显式状态：Covert 2023 与 DIME 的预测器输入本来就是 (x ⊙ m, m)，掩码 m 直接进网络。** 所以六格编码相对动态特征选择文献**不是新的**，只相对修订 3 的四格编码是新的。论文里只能写"沿用 DFS 的掩码输入惯例"，消融臂 `no_missing_state` 回答的是"在本骨干上这个惯例有没有用"。

**不同的（这是贡献的落点）：**
1. 观测是一个冻结 VLM 在真实时间区间上的 0/1 裁定，两个粒度（4 粗块永远在、30 细窗可选）；未观测裁定的预测分布 p(b_w = 1 | E) = q_f·P(h_w | E) + r_f·(1 − P(h_w | E)) 与缺失下的后验由 7 参数区间 HMM 精确前向后向给出（`predictive_fine`、`_emissions` 缺失不发射），不用 VAE 代理也不用摊销的价值网络。
2. **预测器的输入本身是缺失下重算的后验量**（ℓ_t、P(s_t)、块后验），DFS 文献里预测器只看原始特征加掩码，没有这一层。
3. 输出是弱监督（只有视频标签）下的逐秒密集分数，价值在时间轴上取平均；DFS 文献是单标签分类。
4. 价值在推断时非摊销计算（每候选两次前向，30 候选 × 2 × 5 crop），不是学出来的选择器；调研建议的"学一个选择器"作为消融是合理的备选。
5. 训练期观测集受限（Round 0 每视频 8 次）、HMM 只在部分观测上拟合、策略扩展观测集后重训（两轮）。DFS 文献默认训练集全观测。附注：训练期选窗只依赖已观测裁定与内容特征，不依赖未观测裁定的值，属于 MAR，EM 忽略选择机制是合法的，论文可以明说。
6. 最接近的时序先例 Kossen 等 A2MT（TMLR 2023，arXiv 2211.05039，多模态时序数据的主动获取，Perceiver IO，Kinetics/AudioSet 分类）：选的是模态、输出是分类、非弱监督定位。**调研文档漏了这篇，必须补进书目并在论文引用。**

## 3. README 与代码是否一致（第 1–3 节主张能否被臂证伪）

**一致的：**
- 1.1：`sampler` 取 m ~ U{0..|S_v|}（`rng.randint(0, len(al)+1)`），粗块永不遮蔽（`bc` 全观测），六格 `Emb[3·b_coarse + (b_fine+1)]`，`fit` 跳过缺失，每轮用新 HMM 重建 `ScaffoldCache`。
- 1.2：EOC 公式、预测概率来源、argmax、`stop_index` 的 tau/cap 语义、分数 = 五 crop 均值 sigmoid(完整 logit 含先验与 c) 都与代码一致；反事实前向经 `cache.build(vid, bf2)` 真的重跑 HMM。
- 1.3：seed 窗 [0,15,7,22] = bit-reversal 前四；训练期策略 eoc、b_max 步、tau 0；Round 1 prefix_mix .5；测试从 4 粗块起。
- 泄漏：策略只在 train 视频上跑；validation 只选 checkpoint；test 只做规则 7 允许的目标值。

**臂能否隔离各主张：**

| 主张 | 臂 | 判定 |
|---|---|---|
| 选窗有用 | eoc vs uniform/random/entropy/localization/conflict，同一模型同一 pick 数 | 可证伪。注意 checkpoint 是在 uniform 前 8 窗上选的，这对 uniform 对照有利、对 eoc 不利，是保守方向，可以接受但要写明。 |
| 反事实前向有必要 | conflict | 可证伪，但 conflict 同时去掉了 HMM 预测概率加权，两处差异；若想单独主张"HMM 加权有用"需再加一个 0.5/0.5 加权的 eoc 变体（非规则 4 要求）。 |
| 显式缺失状态有用 | no_missing_state | 可证伪（ℓ、P(s) 列仍隐含缺失信息，臂只回答"显式那一格"）。 |
| 预算鲁棒训练有用 | no_dropout | **对照对象写错**：no_dropout 与 full 差三处（无 dropout、允许集合全 30、单轮）。隔离 dropout 必须是 no_dropout vs train34（两者只差 dropout）。README 第 3 节要改。 |
| 策略观测集上重训有用 | round0_only | 可证伪。 |
| 训练期诚实预算的代价 | train34 | 可证伪（full vs train34）。 |
| 细窗经训练买到了什么 | coarse4_train | 可证伪，但**读数位置未写**：该臂的 `summary.json["test"]` 是 eoc 策略结果（代码对该臂仍跑 eoc/uniform），臂的主张要读 `metrics_test_coarse4.json`。README 必须写明。 |
| 停止规则有用 | no_stop（tau = 0） | **当前不可证伪**：预注册操作点就是 tau = 0，no_stop 与 full 是同一个数。见第 4 节第 2 项。 |

## 4. 会让日后主张不合法的地方（开跑前必须修）

1. **checkpoint 选择：README 与代码不一致。** README 第 2 节写"checkpoint 按 validation（同操作点）选"；`train.py` 实际用固定 uniform 掩码（bit-reversal 前 b_max 窗）在 validation 上选，从不跑策略。代码做法可以接受（便宜、且对提案保守），但 README 必须改成代码的做法，规则 14(d) 才对得上。
2. **tau 没有预注册的选择规则。** 操作点 tau = 0，`no_stop` 臂 = 操作点本身；`train.py` 只算 validation 的 eoc k = b_max 一个点，不算 validation 的 (cap, tau) 网格。因此任何 tau > 0 的主张只能从 test 网格里挑，属于 test 选择，不合法。二选一，搜索前定：(i) 预注册一条 validation 规则（例如：cap = b_max 下，validation AP 与 ROC 都不低于 tau = 0 值 − .005 的最大 tau），并在 `train.py` 加 validation 网格；(ii) 停止规则只报曲线，删掉"停止规则有没有用"这一行主张。
3. **E1 按构造成立。** cap 8、tau 0 时每个视频恰好 4 + 8 = 12 次，E1"平均 ≤ 12"不是检验。改写：E1 是操作点定义；效率主张 = E2（12 次不低于 34 次）+ 调用数–指标曲线。
4. **E2 的对照没钉死。** "固定 34 次的起点模型"要写路径与数字：修订 4 过 P2 则用 `runs/20260908_c3_rev4_rev2_backbone_interval_hmm/` 三 seed best-trial，否则用修订 3 `runs/20260907_c3_rev3_interval_evidence/` 的 .6409 ± .0174 / .8421 ± .0080、.7045 ± .0053 / .6924 ± .0089；std 用起点模型的 seed std。同理**骨干变体（bias_mode / ctx_mode）必须在本实验第一个 trial 之前写定并注明日期**，不能看了结果再选修订 2 还是修订 3 骨干。本实验自己的 `fixed34` 行（dropout 训练的模型看全部裁定）是另一件事，可作附加行，不能当 E2 的对照。
5. **训练期调用数记法。** `calls["train_total_per_video"] = 8 + mean(picks)`，而训练期策略从"细窗全缺失"起跑，picks 可能包含已付费的 seed 窗，计数偏高（tau = 0 时恒为 16）。偏高是保守方向不违规，但 README 写的是"8 + 平均新增"，二者要对齐：改成计 |picks \ seeds|，或让训练期策略从 seed 集合起跑。同时写明 tau = 0 时训练期预算是常数 16，不是自适应。
6. **E3 的"同实际调用数"在 tau > 0 时未定义。** uniform 曲线只有 pick 数 {0,2,4,8,12,18,30}；tau > 0 的平均调用非整数。写明比较规则（线性插值，或取不小于平均调用的最近预算点）。tau = 0 时 12 vs uniform k = 8 没问题。

另外两条不阻断、写清即可：eoc 曲线 `eval_max_picks` = 18 只到 22 次，第 2 节曲线表的 34 次点 = `fixed34` 行；训练臂（no_missing_state / no_dropout / train34 / round0_only / coarse4_train）按修订 3 惯例用各 seed best-trial 超参跑三 seed，写进第 3 节。

## 5. 论文必须引什么、贡献怎么写才不过头

**必引：** Covert 等 ICML 2023（随机掩码预测器 + 贪心）、DIME ICLR 2024（期望输出变化 / 条件互信息、变预算停止）、EDDI ICML 2019（对缺失观测积分再算价值）、Shim 2018 / Janisch 2019（停止动作、获取–分类联合）、**Kossen 等 A2MT TMLR 2023（时序多模态主动获取，调研漏项）**、VideoAgent ECCV 2024（LLM 判断"够不够"再取帧）、VADTree NeurIPS 2025 与 Holmes-VAU CVPR 2025（粗到细问 VLM 做异常定位）、LELA 2602.09637 与 CLARA 2608.15905（hateful video 用 VLM 的现状：全量问、无预算）、HateClipSeg / MultiHateLoc（数据与 baseline）。

**建议措辞：** "结构化 VLM 观测上的动态特征选择，用于弱监督时序定位：贪心获取以骨干逐秒输出的期望变化为价值，未观测裁定的预测分布与缺失下的输入后验由区间 HMM 精确给出；骨干在受限观测集上用证据 dropout 训练，训练期与测试期的 VLM 调用数都如实报告。"

**不能写的：**
- "新的贪心准则"——它是 DIME 目标量的非摊销版本；
- "首创缺失状态输入"——DFS 预测器本来就吃掩码；
- "学习到的查询策略"——策略无参数；
- "训练期预算诚实是方法贡献"——它是评测协议与报告，只能作协议贡献；
- 若 E3 不达，不能写"选窗有用"，主张退成"预算鲁棒训练 + 停止规则"（README 已预留）。

**可以写的：** hateful video 定位上首个逐视频、带停止规则、由训练后的定位网络驱动的 VLM 查询；用精确 HMM 缺失推断替代生成式代理；价值定义在逐秒密集输出；粗/细两粒度层次；调用数–AP/ROC 曲线与对照。

## 6. 风险记录（不阻断）

修订 3 第 8 节：HateMM 不训练时 4 粗块的 HMM 后验已高于 34 条全观测。若训练后的骨干也偏好少问，E2 在 HateMM 会"容易过"，此时效率主张必须与 `coarse4_train` 和同模型 `coarse4` 行并排报，否则审稿人会问"是不是根本不该问细窗"。臂表已含，只需保证报表里不省略。
