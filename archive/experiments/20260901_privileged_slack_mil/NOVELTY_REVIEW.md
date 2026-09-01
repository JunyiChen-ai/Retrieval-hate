# Privileged-slack temporal MIL：独立 novelty review

截至 2026-09-01。审查对象仅为本目录 `README.md` 中的候选机制；未审代码、未参与实现。本审查按 `RESEARCH_ITERATION_RULES.md` Rule 12，并结合 Rule 14、16、21 和 `research-wiki/STATUS.md` / failure ledger 裁定。

## 裁定

**STOP，4.1/10。** 三门结论为：Gate 1 **PASS**；Gate 2 **PASS（窄口径）**；Gate 3 **FAIL**。不得进入实现或正式训练。

最窄 claim“train-only privileged correcting function parameterizes asymmetric negative-frame and latent-positive MIL slack，部署时删除 corrector”仍然是把既有 SVM+/LUPI correcting-slack 原理放入标准 positive-witness / all-negative MIL。已有工作还更直接地把训练期文本 privileged information 与视频 bag MIL 结合，用于带 loose/noisy label 的 action/event recognition。当前 task delta 没有越过“直接套用已有 MIL-PI recipe + 更换数据集与 privileged features”的边界。

## 实际检索范围与结果

检索词覆盖：`learning using privileged information hateful video detection/localization`、`SVM+ hateful video`、`privileged slack MIL video`、`multi-instance learning with privileged information`、`teacher-guided MIL hateful video`、`hateful video knowledge distillation`、`hateful video pseudo label`，并检查相邻的 weakly supervised temporal action localization / video anomaly detection。

### 来源方法与最接近跨任务先例

- Vapnik 与 Vashist 的 SVM+ 直接令 privileged-space correcting function 参数化训练样本 slack；这正是候选声称的来源核心，而不是只共享“训练时有额外信息”这一宽泛思想。[Vapnik and Vashist, Neural Networks 2009](https://doi.org/10.1016/j.neunet.2009.06.042)
- Vapnik 与 Izmailov 将 LUPI 概括为 similarity correction 和 teacher-student knowledge transfer；前者仍以 correcting/slack function 为核心。[JMLR 2015](https://www.jmlr.org/papers/v16/vapnik15b.html)
- Niu、Li、Xu 已提出 **MIL-PI**：训练 web videos 带有测试时不可用的文本描述，以 privileged information 进入 multi-instance learning，并同时处理 loose/noisy video labels；应用是 action/event recognition。它已经覆盖“视频 bag + train-only text PI + MIL 弱标签/噪声”的方法骨架。[IJCV 2016](https://doi.org/10.1007/s11263-015-0862-5)，[机构摘要](https://hub.hku.hk/handle/10722/321652)
- MIML-FCN+ 进一步以第二个 `slack-FCN` 建模 privileged bags，并用 PI-modelled loss 约束主网络 loss。它说明深度网络中“privileged branch 只在训练期建模 loss/slack”也不是新的构造。[CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_MIML-FCN_Multi-Instance_Multi-Label_CVPR_2017_paper.pdf)
- Lapin、Hein、Schiele 证明 weighted SVM 总能复现某个 SVM+ solution，虽然反向不总成立。因此“不是显式 loss weighting”不等于机制上已经与 instance weighting 拉开距离；必须由 adaptation 或 control 显示额外结构。[Neural Networks 2014](https://doi.org/10.1016/j.neunet.2014.02.002)

### hateful-video 目标任务占用

在上述精确检索中，**没有找到 LUPI/SVM+ correcting-slack 或 MIL-PI 已用于 hateful-video detection/localization 的公开方法**。因此 Gate 2 不能因为普通 teacher 方法存在而判 FAIL。

但训练期 teacher/auxiliary evidence 指导 hateful-video student 已被占用，故候选不能把“训练期用、推理删除”本身当 novelty：

- LEAF 在 hateful video detection 中由 LMM teacher 生成 supervision，再以 stage-wise distillation 训练部署 student。[Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.604/)
- CLARA 把 VLM-derived rationales 经 gated Transformer 注入 clip-level hateful-video model。[ACM MM 2026 preprint](https://arxiv.org/abs/2608.15905)
- MVKD 已对包含 hateful content 的 harmful micro-video detection 使用 multi-view knowledge distillation。[Information Fusion 2026](https://doi.org/10.1016/j.inffus.2026.104735)

这些是 ordinary KD / rationale guidance 的占用证据，不是 exact SVM+ 占用证据。

## Rule 12 三门逐项判断

### Gate 1：允许 adaptation 已有方法 — PASS

候选明确引用 LUPI/SVM+，没有声称 correcting-slack 原理由本项目从零提出。使用既有来源本身合规。

### Gate 2：来源方法未用于 hateful-video detection/localization — PASS（窄口径）

精确检索未发现 SVM+、privileged correcting function、MIL-PI 或数学等价的 privileged-slack MIL 已进入 hateful-video detection/localization。普通 KD、VLM rationale guidance 和 teacher supervision 已进入该任务，但不能据此把不同的 exact source 判为已占用。

### Gate 3：non-trivial hateful temporal localization adaptation — FAIL

1. `negative bag` 的全部实例为负、`positive bag` 至少存在一个 positive witness，是 canonical MIL 语义；用 soft-min 替代 hard latent minimum 是标准可微松弛。它们不是本任务新增的结构。
2. “corrector 只在训练期读取文本/辅助特征并预测 slack，部署时删除”就是 SVM+ 的核心；而“视频 bag + train-only text privileged information + loose labels”已由 MIL-PI 明确覆盖。把文本描述换成 OOF lexical locality 与 frozen VERA，不足以形成 non-trivial adaptation。
3. 候选没有给出超出 MIL-PI 的新约束：negative-frame/latent-positive 的所谓 asymmetry 由 bag labels 本身直接决定，是二元 MIL 的既有正负不对称，而不是针对 hate timing 新设计的监督关系。
4. 它虽不直接拟合 teacher probability、hard pseudo-label 或 pair order，因此**字面上不是 ordinary KD 或 pseudo-labeling**；但 privileged corrector 改变各帧违反约束的代价和 latent witness 选择，功能上仍是 auxiliary-evidence-dependent instance loss weighting。Lapin 等人的 SVM+/weighted-SVM 关系使这一等价风险尤其直接。
5. failure ledger 已关闭过 `Privileged Rank Transfer` 的同一二分：若 auxiliary signal给出时间方向，就退化为 teacher order/weight guidance；若只忠实预测 slack 而不提供方向，则不能迫使 raw student 形成正确 within-video localization。当前只是把同一条链放进 standard MIL positive-witness loss，没有新增能够越过该二分的约束。
6. RESET6 candidate 1 已关闭 asymmetric lexical support 驱动的训练期机制。当前加入 VERA 并把 lexical/teacher evidence 从 region admission 移到 slack cost，属于换 loss 接入位置；README 没有提供 Rule 16 所要求的、新独立 premise evidence 来重开这条 lexical/teacher source chain。

因此，这不是“ordinary KD”的简单同义词，但仍是**已知 MIL-PI/SVM+ recipe 的直接 task application**，并与本项目已关闭的 lexical/teacher-guided instance-selection/weighting 链同构。Gate 3 失败属于 Rule 12 允许的 pre-run STOP：直接套用，以及与 failure ledger 已关闭机制同构而无新增约束。

## shifted control 能否证伪机制

逐视频 half-video circular shift 保留 privileged sequence 的边际分布、模型容量和训练量。若 aligned 在 HMM/HCS 的 within ROC 均胜 shifted，它可以证伪“privileged evidence 的时间位置完全无关”，所以是有意义的 timing control。

但它**不能**证伪以下替代解释：同一 aligned lexical/VERA signal 仅作为普通 per-frame loss weight、soft witness selector、置信度门或 pseudo-target，也会胜 shifted。该 control 没有区分 privileged correcting-slack 与 ordinary aligned loss weighting，更不能建立相对既有 MIL-PI 的 task-adaptation novelty。negative bags 本身所有有效帧均为 benign，shift 对其也不是“正确 hate timing vs 错误 hate timing”的对照。

要检验最窄机制，至少还需一个 matched aligned auxiliary-evidence weighting/selection control，与 corrector 使用完全相同的 signal、预算和主损失，只移除 learned slack semantics；但即使该 control 通过，也只能验证实现机制，不会消除 MIL-PI 已覆盖其方法骨架这一 novelty 问题。

## 最终结论

- 可 claim：未检出 exact SVM+/MIL-PI 用于 hateful-video；候选与直接 score KD、hard pseudo-label 和 inference fusion 不同。
- 不可 claim：把 train-only lexical/VERA corrector 接到 binary temporal MIL slack 构成新的 non-trivial hateful-localization adaptation。
- 执行裁定：**STOP；不实现、不训练、不用 shifted test 结果倒推 novelty。**
