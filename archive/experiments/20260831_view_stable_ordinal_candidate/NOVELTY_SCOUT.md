# View-deletion-stable ordinal distillation：focused novelty review

**查新截止：2026-08-31。** 本报告仅做 proposal-stage novelty review，不实现方法。项目依据 commit `13257e004fc1d306e2dfbadb4e93317062513f83`；性能动机只依据 `runs/20260831_multiview_consensus_probe/analysis.json`，该文件标明 split=`val`、test_used=`false`、status=`ensemble_upper_bound_only`。

## 1. 候选与 claim 边界

每个语料独立在 train-only 产生四个局部 ordering views：bag-label audio、visual、text probe，加 sparse VERA。对同视频 window pair `(i,j)`，每个 view 只投相对次序符号 `s_k(i,j) ∈ {-1,+1}`。仅当四票至少三票同向，即“删除任意一个 view 后仍保持多数方向”时，才把该方向作为 ordinal pseudo-pair；其余 abstain。student 使用这些 pairs、原 video MIL 和 negative dense supervision。推理无 teachers，只用 student ordering 重排 frozen POWA 每视频 score multiset。

按任务要求，本轮不 claim：

- teacher score ensemble；
- multi-teacher KD 一般范式；
- rank transport、score-multiset preservation 或 differentiable sorting；
- VLM-free inference；
- 单个 audio/visual/text/VERA teacher。

唯一拟 claim 是 **view-deletion-stable abstaining ordinal distillation**。

## 2. 结论

### **STOP 作为 novelty core（约 3/10）；可保留为低成本 diagnostic baseline**

没有检到 hateful-video localization 中完全相同的四-view 实现，但核心并不开放：

1. 对四个完整二元 votes，“删除任一 view 后方向仍为多数”与普通 `3-of-4 majority` 严格等价，并没有额外 certification 或新 stability operator。
2. 多 view 只在 agreement/majority 时生成 pseudo labels，是 co-training、multi-view learning 和 disagreement filtering 的标准做法。
3. 更直接地，Usunier et al. 早在 ECML-PKDD 2011 的 multi-view semi-supervised ranking 中，就对未标注**样本对**比较各 view ranker 的相对次序，只选择 view-specific rankers 一致的 pairs 加入训练。这已经占据“multi-view ordinal agreement → abstaining pseudo-pairs”。
4. 把多数 pseudo-pairs 蒸馏到 test-time 单 student，是标准 multi-teacher/ensemble distillation；teacher 不在推理出现并不能使训练时 majority ensemble 变成新机制。
5. WTAL 已有 two-stream/cross-modal consensus pseudo targets，WSVAD 已有 multi-backbone teacher-to-student KD，ranking 领域已有 teacher-order distillation。

因此，“不 claim teacher ensemble”只是写作约束，不能改变 pseudo-pair generator 本身就是无权重 teacher ensemble 的事实。按项目规则“ensemble 只能作为 baseline/upper bound，不能作为论文主方法”，这个候选不应作为下一轮论文 core。

## 3. 数学等价：deletion stability 没有增加内容

令四票为 `s_1,...,s_4 ∈ {-1,+1}`，总票差 `S = Σ_k s_k`。候选要求存在方向 `d ∈ {-1,+1}`，使删除任一 view `j` 后仍有：

`d · (S - s_j) > 0`，对所有 `j=1,...,4` 成立。

- 若四票 `4-0` 同向，条件显然成立。
- 若 `3-1`，则 `S=2d`。删除任一多数票后剩余票差为 `d`；删除少数票后为 `3d`，条件成立。
- 若 `2-2`，`S=0`。删除 `+1` 后方向为负，删除 `-1` 后方向为正，不可能存在稳定的 `d`。

所以：

`view-deletion-stable ⇔ |S| ≥ 2 ⇔ 至少 3/4 同向`。

这只是 strict majority with abstention。若未来允许 view 缺失或 view 自己 abstain，规则会变成另一个机制，必须重新定义并查新；当前四个 view 都投符号时不能把等价的 majority 重新命名为鲁棒认证。

此外，这个条件只保证**单个 pair**对删一票稳定，不保证全局 ranking 稳定。多数 pair relation 可能非传递。四个各自完全传递的 rankings 仍能产生 3-of-4 的四环，例如 `A>B, B>C, C>D, D>A` 每条各获三票。student 的单标量 ordering 无法同时满足环内所有 pseudo-pairs；若再做 Kemeny/图排序/循环消解，那是另一个已有的 rank aggregation 机制，不能静默加入。

## 4. upper bound 能说明什么、不能说明什么

权威分析文件中的 validation 结果确实支持“多 view 含有互补 ordering 信息”：

| corpus | POWA within ROC | all-view transport | 增益 | 更强单/少 view 结果 |
|---|---:|---:|---:|---:|
| HateMM | 0.57193 | 0.60065 | +0.02872 | audio 0.62774；audio+VERA 0.62076 |
| HateClipSeg | 0.52707 | 0.55611 | +0.02904 | VERA 0.55738；concat+VERA 0.57052 |

两个 all-view 结果都通过预设的 `within_gain_at_least_0.020` 与 pooled feasibility gate，且 score multiset error 为 0。这足以把 multi-view 方案保留为 feasibility/upper-bound baseline。

但它不支持 proposed core：

- `transport_all_view` 是直接 score-level all-view upper bound，不是 3-of-4 sign-consensus pair precision，也不是 student distillation。
- HateMM all-view 明显弱于 audio；HateClipSeg all-view 也不优于 VERA，并弱于 concat+VERA。等权增加 view 没有显示出“一票删除鲁棒性”优势。
- score averaging 保留 magnitude；候选把每 view 压成符号再筛 pairs，信息更少。upper bound 不能被当作 sign-majority student 的性能上界保证。
- 两个数据集的强 view 不同，恰好增加 majority 被相关弱 views 稀释、或隐式按 corpus 选 view 的风险。

## 5. 最接近 prior art

| 工作 | 与候选的重合 | 剩余差异 |
|---|---|---|
| [Usunier, Amini & Goutte, “Multiview Semi-supervised Learning for Ranking Multilingual Documents,” ECML-PKDD 2011](https://nrc-publications.canada.ca/eng/view/object/?id=cbacfa57-40f8-42ad-bf51-5e1417b443bf) | 多个 view-specific rankers 对未标注样本对给相对次序；只选择 views 一致的 pairs 作为 pseudo labels，迭代训练 ranker。 | 使用全体一致而非 3/4 多数，任务是文档 ranking。它比一般 co-training 更直接地占据 ordinal consensus pseudo-pairs。 |
| [TSCN, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510035.pdf) | two-stream consensus、迭代 frame pseudo-GT、消除 WTAL false positives。 | 两个 streams，输出 frame pseudo labels而非四-view pair signs。 |
| [CO2-Net, ACM MM 2021](https://arxiv.org/abs/2107.12589) | appearance/motion cross-modal consensus；每个 CCM 的 attention 互作另一个 CCM 的 pseudo target。 | mutual consistency 而非 majority ordinal pair，但 temporal multi-modal consensus 已明确存在。 |
| [AICL, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25237) | 比较两分支 activation，将 snippets 分成 consistent/inconsistent，以一致区域监督困难区域。 | 不做四-teacher ordinal vote；已占 disagreement filtering/consistent-region trust。 |
| [Self-paced Multi-view Co-training, JMLR 2020](https://jmlr.csail.mit.edu/papers/v21/18-794.html) | 多 view 交换、筛选 pseudo labels，并显式处理错误 pseudo labels和多 view 扩展。 | 分类/检测等通用任务，不是 video ordinal pairs；但 agreement-based pseudo-label selection 是成熟范式。 |
| [Confidence-Aware Multi-Teacher KD, ICASSP 2022](https://arxiv.org/abs/2201.00007) | 多 teachers 的 sample-wise reliability/aggregation后蒸馏到 student，动机正是避免低质量 teacher 误导。 | 有标签辅助的连续权重，不是无权重符号 majority。说明 multi-teacher selective KD 不新。 |
| [DAKD, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Dalvi_Distilling_Aggregated_Knowledge_for_Weakly-Supervised_Video_Anomaly_Detection_WACV_2025_paper.html) | WSVAD 中聚合 I3D/S3D/CLIP 多 backbone teacher，再做 feature/prediction-level KD 到 single-backbone student。 | 不输出 ordinal consensus pairs；直接占 multi-view expensive teacher → cheap VAD student。 |
| [Ensemble-Based KD for VAD, 2024](https://www.mdpi.com/2076-3417/14/3/1032) | VAD 中 ensemble teacher / multi-teacher knowledge 被压入单 student。 | 非弱监督 hate localization，非 pairwise order；部署故事已有。 |
| [RankDistil, AISTATS 2021](https://proceedings.mlr.press/v130/reddi21a.html) | student 匹配 teacher 的 item order，用 ranking loss 蒸馏。 | 单/聚合 teacher 的 top-k ranking，不是 multi-view abstention。ordinal distillation 本身不新。 |
| [Ranking Distillation for VideoQA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Ranking_Distillation_for_Open-Ended_Video_Question_Answering_with_Insufficient_Labels_CVPR_2024_paper.html) | 用 teacher pairwise priorities 训练 student，并对不确定 pairs 放松/降权。 | VideoQA answer ranking，不是 temporal localization；已占 noisy/uncertain teacher pair filtering + pairwise KD。 |

相邻的 [Ju et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ju_Distilling_Vision-Language_Pre-Training_To_Collaborate_With_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html) 和 [MLLM4WTAL, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Weakly_Supervised_Temporal_Action_Localization_via_Dual-Prior_Collaborative_Learning_Guided_CVPR_2025_paper.html) 还分别占据 VLP/MLLM guidance 与 WTAL collaborative/interactive distillation。项目内 [RIFT preregistration](../../docs/duplex/PREREG_RIFT_MACIL.md) 已明确占据 train-only semantic teacher、confidence-gated within-video ordinal distillation和 VLM-free test。

## 6. 为什么“多模态 views”不自动提供可靠性

多数投票只有在 individual views 至少略优于随机且错误不高度相关时，才有常见的降错直觉。当前四个 views 不满足已证条件：

- audio/visual/text probes 使用同一语料 bag labels，可能共享 video identity、class prior 与 MIL shortcut；
- VERA 虽模型来源不同，但输入语义与 text/visual evidence 重叠，不是独立 annotator；
- HateMM validation 上 visual within ROC 0.49922、VERA 0.49263，不能被默认视作可靠 voter；
- view 的票数不是信息独立性的度量。复制同一个 view 三次即可制造“删除一票仍稳定”，却没有新增证据。

所以 3/4 agreement 至多是启发式置信度，不能称为 certified correctness、robustness guarantee 或 causal consensus。

## 7. 若作为 diagnostic baseline，最小 controls

所有条件必须使用同一 student、MIL/negative dense loss、训练预算和 frozen POWA transport readout：

1. `POWA + rank-transport-only`，无 teacher pairs；
2. 四个 single-view ordinal KD，分别报告 audio/visual/text/VERA；
3. `best-single-view KD`，但 best 只能按 train-only 规则或预注册，不能看 validation span GT 后挑；
4. unweighted mean-score/mean-rank multi-teacher KD，作为普通 ensemble distillation；
5. `3-of-4` sign majority，即候选本身；
6. `4-of-4` unanimity，直接对照 ECML-PKDD 2011 式 view agreement；
7. matched-coverage controls：从每个 single view 按 pair margin 选取与 3-of-4 相同数量的 pairs，排除“少而容易”带来的收益；
8. leave-one-view-out 四次完整训练/评估，测试真实依赖，而不是只证明单 pair 的票数恒等式；
9. view-sign/time shuffle；若 student 增益不消失，说明是 pair 数、正则或 coverage 效应；
10. duplicate-view control：用某一 view 的精确副本替换另一 view并仍计一票。若结果显著改变，证明规则把相关副本误当独立证据；不得称 view-robust。

还必须报告：

- candidate pair coverage、每视频 pair 数分布；
- 3-1 与 4-0 pairs 分开后的 validation pair correctness；
- 每个 view 的边际 vote、dissent rate、两两 Kendall disagreement/error correlation；
- pseudo-pair directed graph 的 cycle rate、最大冲突子图，以及 student 无法满足的 pairwise loss 比例；
- accepted pairs 对真实 within-video GT order 的 precision，按相同 coverage 与 single-view/margin controls 比较。

## 8. diagnostic success gate 与论文 gate 的区别

作为一个工程 baseline，`3-of-4` 值得保留的最低条件是：

- 在 HateMM、HateClipSeg 都比 rank-transport-only 和四个 single-view KD 提高 within-video macro ROC；
- 在 matched pair coverage 下，accepted-pair correctness 高于每个 single view 与 mean-rank KD；
- pooled Frame AP/ROC 相对 frozen POWA 每项下降不超过 0.01 absolute；
- view/time shuffle 消除增益，cycle rate低且不能解释主要 loss；
- leave-one-view-out 不出现只删某一 view 就崩溃的情况。

即使全部通过，也只说明 consensus pseudo-pair baseline 有效；**不会自动把 novelty verdict 从 STOP 变成 GO**。要成为论文 core，必须出现一个不等价于 majority/agreement filtering 的新机制，例如显式建模 view 相关错误、在可验证假设下给出 worst-case ordinal risk，且实验表明有效部分来自该机制。那将是新候选，需重新查新。

## 9. 致命 anti-pattern / 一票否决

1. 把 `3-of-4 majority` 重命名为 deletion-stable certification，却不写上述等价证明。
2. 声称“删除一个 view 鲁棒”，但只在 pair vote 上做代数检查，不运行 leave-one-view-out student training/evaluation。
3. 把 audio/visual/text/VERA 当独立 voters，不测 error correlation；或把同源/重复 view 数成多份证据。
4. 不做 matched-coverage control。consensus pairs 天然更少、更容易，涨分可能只来自筛掉难 pair。
5. 用 validation/test span labels选择 view、pair margin、3/4 vs 4/4、teacher combination或 corpus-specific fallback。
6. HMM 实际依赖 audio、HCS 实际依赖 VERA，然后按语料切换 teacher，却统一包装成 view-stable method。
7. 不审 pair graph cycles，或静默加入 graph projection/Kemeny aggregation来修环。
8. direct all-view score ensemble 与 student majority结果混报，或把 upper bound 数字写成已实现 student evidence。
9. rank transport 在不同 controls 使用不同配置，导致无法归因；或重新 claim transport novelty。
10. teacher 在 validation/test media 上生成训练监督，或不同主数据集 train set混合。

## 10. 最终建议

**停止把该候选作为新方法推进。** `view-deletion-stable` 在当前定义下只是 3-of-4 majority，而 multi-view ranking agreement pseudo-pairs、temporal consensus、multi-teacher selective KD 和 ordinal distillation均已有直接先例。

如果实现成本很低，可以按第 7 节把它跑成一个 diagnostic baseline，回答“train-only 多 view 共识能否压入 student”。但它不能占用一次声称有 novelty 的正式方法迭代，也不能因 teacher-free inference 或 score-multiset transport 改名为新 core。现有 upper bound 支持继续研究 **view error structure**，不支持继续包装 **majority vote**。

## 11. 检索范围与局限

检索覆盖 ensemble/multi-teacher KD、confidence-aware teacher aggregation、co-training、multi-view majority/agreement pseudo-labeling、uncertainty/disagreement filtering、multi-view semi-supervised ranking、WTAL two-stream/cross-modal consensus、WSVAD teacher-student KD、pairwise/ordinal ranking distillation，以及 2025--2026 相关工作。主要入口为 CVF Open Access、ECVA、AAAI Proceedings、JMLR/PMLR、NRC 官方论文档案、arXiv 和出版方论文页；结论仅由论文或官方页面支撑。

“未见 hateful-video 中相同四-view 代码”不等于组合可 claim。这里的 STOP 主要来自数学等价与 2011 年 multi-view ordinal pseudo-pair 直接先例，不依赖检索是否穷尽所有近期论文。
