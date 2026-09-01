# Certified-negative error routing：focused novelty review

**查新截止：2026-08-31。** 本报告只做 proposal-stage prior-art review，不实现方法。项目依据 commit `13257e004fc1d306e2dfbadb4e93317062513f83`。检索优先使用论文原文、CVF/AAAI/ACL/NeurIPS/PMLR 官方页面、arXiv 原稿和作者官方项目页。

## 1. 最终被审机制

本报告以最后窄化的版本为准，不再审查已否决的 corpus-level scalar trust：

- 每个主数据集独立训练；VLM/VERA 只稀疏评分该语料的 train windows。
- 按标准 MIL 语义，negative video 的所有局部单元在该数据集 label policy 下都为 negative。因此，negative video 上任何 `teacher-positive` 都是 **policy-certified local false positive**。
- 只用这些 certified errors 建立 feature-dependent teacher false-positive risk。positive-video 的 `teacher-positive` 只有在预测 risk 低、且 POWA policy primitives 不冲突时，才提供正向局部/顺序蒸馏；其余一律 abstain。
- `teacher-zero` 在 positive video 中保持 unlabeled，绝不当 benign 或 forced negative。
- student 同时保留原始 video-level MIL 和 negative-video dense negative supervision；测试不调用 VLM。
- inference 使用 student ordering 和原 POWA 每视频 score multiset 的固定 transport/readout。rank transport 只负责 anchor preservation，**不计入本候选 novelty**。

候选唯一拟 claim 的机制应是：

> 从 weak negative bags 中抽取可认证的 teacher 局部错误，学习 feature-dependent error risk，并把该 risk 迁移到 positive bags，对 VLM 正伪标签做 policy-aware、positive-only、abstention-based selective distillation。

## 2. Stop / go verdict

### **有条件 GO：只值得做最小 pilot；proposal novelty 约 6/10，不是强 novelty**

截至截止日，未检到 WS-VAD、WTAL 或 hateful-video localization 工作完整采用：

`negative-bag certified VLM false positives → feature-dependent teacher-error risk → positive-bag teacher-positive selective distillation → teacher-zero remains unlabeled → policy-conflict abstention → VLM-free student`。

这个窄版本比普通 confidence-weighted KD 多出一个可识别的监督非对称性：teacher confidence 不等于 teacher correctness；候选直接利用 negative bags 的 instance labels 是确定的这一事实，收集 teacher 的已知错误，并只学习“什么时候不要信 teacher-positive”。它也避免了两个常见错误：把 positive-bag teacher-zero 当 negative，以及根据 corpus/validation 手选 teacher。

但它仍处在拥挤边界：normality guidance、confident-instance mining、pseudo-label noise correction、selective KD、PU risk estimation、VLM temporal pseudo labels、policy primitive filtering 均有先例。能否晋级取决于一个非常具体的假设：

> **negative-video 中 VLM false-positive 的特征条件风险，能迁移到 positive video 内的 hard-negative windows。**

如果该 transfer 不成立，risk head 只会学习 negative/positive video identity、背景、语言、频道或 teacher score，方法就退化为 feature-dependent corpus routing。故这里的 GO 只是允许一个严格 pilot，不是允许完整工程实现或论文 claim。

### 已否决版本

用 train 上 VLM bag score 对 video label 的区分力估一个 scalar trust，再按 trust 加权 KD/回退 MIL，仍是常规 confidence-weighted distillation。bag-level discrimination 测的是 between-video separation，不证明 within-video ordering；POWA 当前失败本身已经说明两者可脱钩。该版本应保持 **STOP**，不得作为 control 以外的主方法。

## 3. 最接近 prior art

| 工作 | 已占据的部分 | 与窄候选的剩余差异 |
|---|---|---|
| [TPWNG, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html) | 从 normal videos 聚合 normal visual prompt，将 CLIP normal/anomaly text-image evidence 融入 frame pseudo-label generation，再训练 temporal classifier；[补充材料](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Yang_Text_Prompt_with_CVPR_2024_supplemental.pdf)说明测试移除 text/pseudo-label branch。 | 最接近的单篇工作。它用 normal similarity 直接指导 pseudo labels，不把 negative-video `teacher-positive` 显式标为已知 teacher error，也不学习条件 error risk。 |
| [Ju et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ju_Distilling_Vision-Language_Pre-Training_To_Collaborate_With_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html) | 让 CBP 提供 confident background、VLP 提供 confident foreground，按来源擅长区域协同蒸馏。 | 已占“不同证据负责不同局部标签”。其 source trust 是设计先验，不是从 negative-bag 已知 VLP 错误学习局部 risk。 |
| [NG-MIL, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Park_Normality_Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_WACV_2023_paper.html) | 从 noise-free normal videos 学 normal prototypes，以 similarity classifier 修正普通 MIL score。 | 已占 certified normal evidence / normal-density refinement；没有外部 VLM teacher-error model，也不是 selective positive-only KD。 |
| [UMIL, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Lv_Unbiased_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2023_paper.html) | 将 snippets 划分为 confident normal/abnormal 与 ambiguous，学习 invariant classifier，并用 pseudo labels 自训练 ambiguous set。 | 已占 confident-instance/ambiguity 处理；没有“teacher 在 negative bags 上的已知 FP”这一监督对象，也不估 teacher conditional error。 |
| [AICL, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25237) | 比较 class-aware/class-agnostic 两分支，按 activation agreement 划分 consistent/inconsistent snippets。 | 已占 cross-evidence agreement；没有 certified teacher errors、one-sided pseudo-positive selection 或 policy compiler。 |
| [Rethinking Pseudo-Label Guided Learning, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33094) | online teacher-student、ambiguous/missing instance correction、高质量 pseudo-label mining 与加权。 | 已占 noisy pseudo-label correction/confidence weighting；risk 来源不是 weak negative bags 中可认证的外部-teacher FP。 |
| [Towards Better Utilization of Pseudo Labels, Information Sciences 2023](https://doi.org/10.1016/j.ins.2022.12.044) | 分析高置信错误 pseudo labels，并用 overconfidence suppression 减轻自蒸馏错误。 | 说明“高 teacher score 不等于可靠”已知；没有 feature-dependent certified-error transfer。 |
| [DAKD, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Dalvi_Distilling_Aggregated_Knowledge_for_Weakly-Supervised_Video_Anomaly_Detection_WACV_2025_paper.html) | WSVAD 中将多 backbone teacher 蒸馏到 single-backbone student。 | 已占 expensive-teacher to cheap-student；不处理外部 teacher 的局部错误选择。 |
| [MLLM4WTAL](https://arxiv.org/abs/2411.08466) | 用 MLLM key semantics 和 complete semantic priors 指导异构 WTAL 模型。 | 已占 MLLM privileged temporal semantics；没有 negative-certified error risk。 |
| [nnPU, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html) | 从 positive + unlabeled 样本估风险，并修正 flexible PU model 的负经验风险/过拟合。 | 本候选的 risk-learning 数学上接近“known teacher-errors + unlabeled teacher-positives”的 PU 问题；视频 MIL、teacher error、policy gating 和 selective KD 是应用差异。不能把 PU estimator 本身 claim 为新。 |

同域 hate teacher 也已存在：[VERA](https://arxiv.org/abs/2412.01095) 直接输出 segment/frame anomaly score；[LELA](https://arxiv.org/abs/2602.09637) 用五类模态 caption 与 LLM 输出细粒度 hate score；[TANDEM](https://arxiv.org/abs/2601.11178) 做多模态 hate temporal grounding。故不能 claim “首次 VLM hate teacher”或“首次 VLM-free student”。

## 4. 与项目内已有方法的 collision

- [POWA preregistration](../../docs/duplex/PREREG_POWA_MACIL.md) 已占 `train-only MLLM primitive evidence → dataset policy compiler → dense MIL → test without MLLM`，并已要求 policy/primitive permutation ablations。
- [RIFT preregistration](../../docs/duplex/PREREG_RIFT_MACIL.md) 已占 `train-only semantic teacher → confidence-gated within-video ordinal distillation → frozen-base residual`。

因此当前候选不能重新 claim policy compilation、primitive teacher、ordinal KD、VLM-free inference 或 anchor-preserving readout。真正的内部增量也只有：**用 negative bags 观察到的外部-teacher 局部错误，学习可迁移的 feature-dependent false-positive risk。**

## 5. 数学边界与必须说清的识别问题

记 frozen VLM teacher 在被采样 window 上的正预测为 `T=1`，video bag label 为 `Y`：

- `Y=0, T=1`：在该 corpus policy 下是已知 teacher error，记 `E=1`；
- `Y=1, T=1`：`E` 未知，可能是真 positive，也可能是 positive video 内的 normal/hard-negative window；
- `T=0`：不进入正向蒸馏，也不被赋成 negative。

这里不能把 `Y=1, T=1` 全部标成 `E=0` 训练普通二分类 risk head；这会循环假设 teacher-positive 正确，并主要学习 bag identity。合法的最小实现只能是以下之一：

1. **one-class error proximity**：只对 `Y=0,T=1` 建模，输出与已知 FP manifold 的相似度/密度；
2. **PU error estimator**：把 `Y=0,T=1` 当 labeled errors，把 `Y=1,T=1` 当 unlabeled mixture，使用有明示假设和 class-prior 处理的 PU objective；
3. **density-ratio risk proxy**：训练区分 error-source 与 unlabeled-source，但只把输出称为相对 risk proxy，不声称校准后的 `P(E=1|x)`。

三者都依赖 transfer/selection assumption：negative-video FP 与 positive-video 内 FP 在所用 feature space 中具有可迁移结构。特别是 PU 的常见 selected-completely-at-random 假设在这里并不自然成立；negative-bag error 明显是按 bag context 选择出来的。报告必须把它作为假设，而不是定理。

policy gate 也必须是可审计逻辑：仅当 low-risk teacher-positive 与 POWA 的 policy-required primitives 不冲突时提供 positive order/consistency；冲突和未覆盖都 abstain。不得给 policy 一个自由 dataset embedding 或 per-corpus learned head。

## 6. Rank transport 的位置

推理时将 frozen POWA 每视频 score multiset 按 student ordering 重排，是合理的 anchor-preserving readout：每视频 values、mean、quantiles 保持不变，改变的只是 timestamp assignment。但它不是本轮 novelty，也不是证明 risk routing 有效的证据。

如需可微排序，[SoftSort](https://proceedings.mlr.press/v119/prillo20a.html) 等只是工具先例。pilot 中必须固定同一 transport/readout 用于所有 ordering controls；否则无法区分增益来自 teacher-risk routing，还是来自 score rearrangement。还应单独报告 `rank-transport-only` control。

注意：保持每视频 score multiset 不等于严格保持 pooled frame AP/ROC，因为 scores 与 frame labels 的配对被改变；它只严格保持每视频及跨视频的 score-value distribution。因此 pooled 指标仍必须实测，不能由守恒性质宣称不变。

## 7. 最小可证伪 pilot

### 7.1 数据与训练纪律

- 至少独立跑 HateMM 与 HateClipSeg；不得混合不同数据集的 train set。两者现有 VERA feasibility 呈相反强弱，适合检验 risk gate 是否比“挑强 teacher”更一般。
- 所有 VLM windows、certified errors、risk model、threshold/coverage rule 只用各自 train split。
- risk model 必须按 video 做 K-fold cross-fitting：某视频的 pseudo-positive 权重由未见过该视频的 risk model 产生，防止记忆 video/channel identity。
- risk threshold 由 train-negative false-positive rejection/coverage 规则预先确定，不能看 validation span GT 调阈值。
- POWA、VLM teacher、policy primitive extractor和 rank-transport readout全部冻结；pilot 唯一可变机制是 teacher-positive selection rule。

### 7.2 必须 controls

所有条件使用同一 student、MIL、negative dense loss、训练预算和 transport readout：

1. `POWA + rank-transport-only`：student ordering 不用 VLM teacher；
2. raw sparse teacher-positive KD：所有 `Y=1,T=1` 均蒸馏；
3. teacher-score threshold/confidence KD：排除普通 confidence filtering；
4. normal-density/prototype gate：排除 NG-MIL/TPWNG-style normality filtering；
5. POWA-policy-only conflict gate：排除只靠已有 policy primitives；
6. learned FP-risk gate，不加 policy conflict；
7. 完整 `FP-risk + policy-conflict + abstain`；
8. scalar bag-trust KD（已否决版本，仅作 baseline）；
9. risk-label shuffle、risk-feature shuffle 或用 random negative windows 替换 certified VLM-FP 的 negative controls。

若预算必须缩小，1/2/3/4/6/7/9 不能删；否则不能判断增益是否只是 confidence、normal density 或一般 feature filtering。

### 7.3 两层可证伪 gate

**先验 gate A：risk 是否真的转移。** 在完全未参与 risk 训练、且 threshold 已由 train 冻结的 validation 上，利用 evaluation-only span GT 审计：

- 对 positive videos 的 teacher-positive windows，risk 应能把错误 pseudo-positive 排到更高 risk；报告 error-detection ROC/AP、risk decile 的 empirical precision、accepted-teacher precision/coverage curve。
- 在相同 accepted coverage 下，learned FP-risk 必须优于 teacher score、normal density、POWA primitive conflict 和 scalar trust。
- 该审计仅评估，不把 validation span labels回流到 risk 训练、threshold 或 feature 选择。

若 gate A 不成立，立即停止 student 训练扩展；因为核心 transfer assumption 已被否证。

**下游 gate B：定位是否由该 risk 机制改善。** 在至少两个主数据集：

- 完整方法的 within-video macro ROC 高于 rank-transport-only、raw teacher KD、score-confidence、normal-density、risk-only 和 policy-only；建议预注册最小实质差异 `+0.01 absolute`。
- pooled Frame AP 与 ROC 相对 frozen POWA 每项下降不超过 `0.01 absolute`。
- risk-label/feature shuffle 至少消除完整方法相对最佳非完整 control 的一半增益。
- full 必须优于 risk-only；否则 policy-conflict gate没有增量。full 也必须优于 policy-only；否则 learned error risk没有增量。
- 报告 teacher-positive raw/accepted coverage、negative-video FP rejection、positive-video accepted precision、abstain rate、每种约束产生的 pairs/gradient mass。

## 8. 一票否决项

以下任一情况出现，应淘汰该候选，不继续堆 gate 或 corpus-specific calibration：

1. 把 positive-video teacher-positive 当作 risk-negative 训练普通二分类器，却仍声称学到 calibrated false-positive probability。
2. held-out validation 上，risk 对 positive-video teacher errors 的排序不优于 teacher score 或 normal density；这直接否证 transfer assumption。
3. risk head 可仅凭 video/bag identity、语言、频道或 teacher scalar score复现；去掉 local features 不降性能。
4. 只有每语料单独换 risk family、阈值方向、feature set 或 teacher priority 才提高。
5. full 不胜过 risk-only、policy-only、confidence-only 或 normal-density gate；组合中的新交互没有必要性。
6. `teacher-zero` 被当 benign，或 teacher-positive 未通过 risk/policy gate仍进入正蒸馏。
7. risk/teacher 使用 validation/test labels、test VLM scores，或风险阈值看过 validation span GT 后选择。
8. 只在一个主数据集提高，或 within-video 提高但 pooled AP/ROC 明显下降。
9. risk-label/feature shuffle 不影响结果，说明 student 用的是 coverage、class prior 或训练正则，而不是 certified-error structure。
10. rank transport 的实现/超参随 control 改变，或把 transport 本身包装为本轮 risk-routing novelty。

## 9. 可主张与不可主张

若且仅若两个 gate 都通过，可使用最窄 claim：

> We learn a feature-dependent external-teacher false-positive risk from policy-certified errors in weak negative bags, and transfer it as an abstaining, positive-only distillation gate for local ordering in positive bags.

不能 claim：

- first VLM/MLLM supervision for WTAL/WSVAD/hate localization；
- first normality-guided pseudo-labeling；
- first confidence-weighted/selective KD；
- first PU risk estimator；
- first policy-conditioned hate localization（POWA 已占）；
- rank transport / differentiable sorting 本身；
- score multiset preservation 严格保证 pooled AP/ROC 不变。

## 10. 最终建议

**GO 做一个单机制、两数据集、先审 risk-transfer 再训 student 的 pilot。** 不建议先实现 flexible router、MoE 或多种 fallback。最小版本应采用 one-class error proximity 或明确的 PU/density-ratio risk proxy，positive-only 蒸馏、conflict abstention、固定 transport。

最值得先跑的不是最终 localization，而是 gate A：如果 negative-video certified VLM false positives 学出的 risk，不能在 held-out positive videos 上优于 teacher confidence 与 normal density 地识别错误 teacher-positive，这个 idea 的唯一新机制已经失败，应立即 STOP。若 gate A 和两个数据集的 gate B 同时通过，proposal novelty 才足以进入实现后的独立深度查新；届时真正可审的贡献是 **certified-error transfer**，不是组件组合。

## 11. 检索范围与局限

检索覆盖 WSVAD、WTAL、hateful-video temporal localization、normality/negative prototypes、MIL/PU learning、teacher error/confidence、pseudo-label denoising、selective KD、branch agreement、VLM/MLLM temporal teachers、policy/logic conditioning，以及排序/transport 工具。主要入口为 CVF Open Access、AAAI Proceedings、ACL Anthology、NeurIPS/PMLR、arXiv 和作者官方项目页；结论只引用原论文或官方页面。

“未发现完整同构”不等于不存在；本审查不能覆盖未公开稿件、专利或未索引工作。该方向与 PU/selective distillation 相邻且容易独立出现，因此 pilot 有效后仍必须按项目规则第 12 步重新查新，并把真正有效的 risk estimator 与最接近工作逐公式比较。
