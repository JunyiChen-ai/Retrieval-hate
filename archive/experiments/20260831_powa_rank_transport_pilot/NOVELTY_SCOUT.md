# Novelty scout: frozen-score temporal rank transport

**截至日期：2026-08-31。**  本记录针对本轮候选做独立查新，不构成穷尽式专利检索。
检索只采纳论文原文、出版社论文页、会议论文页和官方项目页等一手来源。检索范围包括
WSVAD、WTAL、weakly supervised temporal assignment、learning-to-rank、differentiable
sorting、optimal transport/permutation learning、exact histogram/eCDF matching，以及视频插入和
segment shuffling。重点检索词包括 `video anomaly score permutation`、`temporal localization
differentiable sorting`、`score reassignment`、`score multiset`、`histogram-preserving
rearrangement`、`NeuralSort/SoftSort/Sinkhorn temporal localization`。

## Candidate under review

给定同语料训练得到并冻结的 POWA 逐帧分数

\[
b_v=(b_{v1},\ldots,b_{vT_v}),
\]

将其排序后仅作为不可学习的 values：

\[
z_v=\operatorname{sort}(b_v).
\]

另一个 temporal rank head 从同一多模态序列输出顺序分数
\(q_v=h_\theta(x_v)\)。由 \(q_v\) 产生时间位置与 \(z_v\) 的一一置换
\(P(q_v)\)，最终输出

\[
s'_v=P(q_v)^\top z_v.
\]

训练信号只来自同语料 train split：从 negative train video 取出在 negative-bag 假设下
label-certified benign 的连续窗口，插入 positive train video，形成 donor-low/order 约束；不使用
任何 frame/span annotation。训练可使用 differentiable sorting、Sinkhorn/OT 或 permutation
relaxation，推理使用带显式 tie rule 的 stable hard assignment。

## Verdict

**查新不否决，建议进入 pilot。领域/机制 novelty 中高，通用算法 novelty 低，总体判断为
medium，约 6--7/10。**

截至检索日期，没有找到 WSVAD、WTAL 或 hateful-video localization 方法同时满足以下四点：

1. 冻结 same-corpus detector 的逐视频 score multiset；
2. 只学习 content-conditioned 的 score-to-timestamp permutation；
3. hard inference 时逐视频 empirical score distribution 精确不变；
4. permutation 由同语料、train-only 的 benign insertion/order constraint 学习。

但不能把 score-multiset reassignment、differentiable sorting、OT、permutation learning、视频插入
或 temporal-order supervision 中的任何一项单独主张为新。最接近的 WSVAD 工作已经使用完全相同
的“逐视频 score multiset 在时间位置间重分配”作为随机诊断；跨领域的 exact distribution
matching 也已经使用“一个信号给 order、另一个信号给 values”的 sort-matching 结构。因此本轮
可辩护的新意只存在于这些部件针对弱监督 temporal localization 的受约束组合。

如果方法最终只是独立训练一个 rank head，再在 inference 做 per-video quantile mapping，而没有
证明冻结 marginal 约束是性能和稳定性的必要组成，reviewer 可以合理地把它归类为 post-hoc
reranking。若论文声称提出了新的 OT 或 differentiable sorting 算法，则 novelty claim 不成立。

## Closest works

### 1. Same-domain closest: random within-video score-multiset reassignment

[Song and Lee, *Frame-Level Evaluation in Weakly Supervised Video Anomaly Detection Mostly
Measures Video-Level Ranking*, arXiv:2608.21854, 2026](https://arxiv.org/abs/2608.21854)，
Appendix E，把每个视频已保存的 score multiset 均匀随机赋给该视频的时间位置。该干预严格保持
完整的 per-video score histogram，并解析计算随机重排后的期望 Micro-AUROC。

这是单篇最接近工作：其 invariant 和操作对象与本候选相同。关键差异是该论文只做 random
permutation metric audit，不训练 content-conditioned permutation，不使用 benign insertion，
也不提出 localization method。因此，“within-video score-multiset reassignment”本身已经被
占位；“learned temporal assignment under a frozen score marginal”尚未被该工作占位。

### 2. Same abstract mechanism outside temporal localization

[Zhang et al., *Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain
Generalization*, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Exact_Feature_Distribution_Matching_for_Arbitrary_Style_Transfer_and_Domain_CVPR_2022_paper.html)
使用 Sort-Matching：由 content features 的 rank 决定位置，由 reference features 提供排序后的
values，从而精确匹配 empirical CDF。它在抽象结构上非常接近本候选的 order/value 分离，但对象
是图像 feature distribution，任务是 style transfer/domain generalization，不是冻结异常分数的
时间定位。

更早的 exact histogram specification 也已证明，按一个信号的严格顺序赋予另一个信号的 order
statistics 可以精确保留目标 histogram。因此“通过排序实现 empirical-distribution preservation”
不是新数学原语。

### 3. Differentiable permutation/sorting tools

- [Mena et al., *Learning Permutations with Gradient Descent and the Sinkhorn Operator*, ICLR
  2018](https://research.google/pubs/learning-permutations-with-gradient-descent-and-the-sinkhorn-operator/)
  用 Sinkhorn relaxation 学习 deterministic/stochastic permutations。
- [Grover et al., *Stochastic Optimization of Sorting Networks via Continuous Relaxations*, ICLR
  2019](https://openreview.net/pdf?id=H1eSS3CcKX) 提出 NeuralSort。
- [Cuturi, Teboul and Vert, *Differentiable Ranking and Sorting using Optimal Transport*, NeurIPS
  2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html)
  把 sorting 写成 assignment/OT 并以 entropic regularization 和 Sinkhorn 求连续近似。
- [Prillo and Eisenschlos, *SoftSort: A Continuous Relaxation for the argsort Operator*, ICML
  2020](https://proceedings.mlr.press/v119/prillo20a.html) 给出 argsort 的简单连续 relaxation 和
  hard projection 路径。

这些工作完整占位了 differentiable sorting、bistochastic assignment 和 hard/soft permutation
工具。它们可以作为实现手段，不能成为本轮 novelty claim。

### 4. Weakly supervised temporal assignment and order supervision

[Bojanowski et al., *Weakly Supervised Action Labeling in Videos Under Ordering Constraints*,
2014](https://arxiv.org/abs/1407.1208) 把视频分块后的 action-label assignment 与 ordered action
list constraints 联合优化。它证明 weak temporal supervision 下学习 assignment/order 早已有先例，
但赋的是 action labels，而不是冻结 detector 的 score values，也不保持 score marginal。

[Zhang et al., *Action Shuffling for Weakly Supervised Temporal Localization*,
2021](https://arxiv.org/abs/2105.04208) 通过 intra/inter-action shuffling、order prediction 和
global-local adversarial learning 学习 temporal representation。它是 temporal-order pretext 的
直接先例，但没有 score transport。

### 5. Shuffling/insertion in WSVAD and WTAL

[AlMarri, Zaheer and Nandakumar, *A Multi-Head Approach With Shuffled Segments for
Weakly-Supervised Video Anomaly Detection*, WACV Workshops
2024](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/AlMarri_A_Multi-Head_Approach_With_Shuffled_Segments_for_Weakly-Supervised_Video_Anomaly_WACVW_2024_paper.html)
从 normal/anomaly video pair 随机拼接连续 segments，制造 virtual events，并训练 anomaly、boundary
和 center heads。它是最接近的 WSVAD synthetic temporal-order supervision，但既不重排已有 score
values，也不保持 score distribution。

[Yun et al., *VideoMix: Rethinking Data Augmentation for Video Classification*,
2020](https://arxiv.org/abs/2012.03457) 向另一个视频插入 spatio-temporal cuboid/temporal segment，
按插入体积混合 video label，并报告对 WTAL 的收益。它占位了视频插入增强，不占位
label-certified benign donor、rank assignment 或 score-mass preservation。

WSVAD 的 outer/inner-bag ranking 也早已有大量先例。例如
[Xu et al., *Discriminative Score Suppression for Weakly Supervised Video Anomaly Detection*,
WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Xu_Discriminative_Score_Suppression_for_Weakly_Supervised_Video_Anomaly_Detection_WACV_2025_paper.html)
结合 outer-bag ranking、positive-bag top/bottom score-sensitive inner-bag loss 和 normal
prototypes。故“在 MIL 中学同视频排序”不能单独 claim novelty。

### 6. Internal closest: V20 centered local judge

内部 V20 先例见 [`docs/V20_V26_FINAL_ITERATION_ARCHIVE.md`](../../docs/V20_V26_FINAL_ITERATION_ARCHIVE.md)。
V20 使用 frozen identity/consensus base、video-constant global prior 和
`local score - per-video frame mean` 的 centered local correction，再用 \(\alpha,\beta\) 相加。
该方法只固定 local correction 的均值；加法会改变 variance、quantiles、top-k 和完整 multiset。
本候选使用 hard permutation，保留所有 permutation-invariant statistics，约束严格强于 V20，因而
可以在机制上区分；但必须用直接消融证明这种更强约束是必要的。

## Narrowest defensible claim

建议只主张：

> **Frozen-score empirical-distribution-preserving temporal assignment**：在不生成、缩放、
> 平移或混合任何新 score value 的前提下，由同语料弱监督干预学习冻结 score mass 与时间位置的
> content-conditioned coupling；hard inference 精确保留每个视频的 empirical score marginal，
> 只改变 score-to-timestamp association。

同义但更直白的名称是 **frozen-score temporal permutation** 或 **per-video quantile
assignment**。“Rank transport”可以作为解释，但不应暗示提出了新的 OT 算法。

可主张的保证是：若 \(P\) 是 hard permutation，则

\[
\operatorname{multiset}(s'_v)=\operatorname{multiset}(b_v),
\]

从而对任意 permutation-invariant video functional \(\phi\)，

\[
\phi(s'_v)=\phi(b_v).
\]

这包括 mean、max、min、variance、所有 order statistics、quantiles、top-\(k\) mean、histogram、
log-sum-exp 和任何对称 MIL bag readout。候选不是 calibration：calibration 通常改变 values 并保持
order；这里保持 values 并学习 order。候选也不是 ensemble：没有融合两个 head 的 score
magnitudes，rank head 只提供 permutation。

## Mathematical and methodological boundaries

### Hard permutation is required for exact preservation

严格 multiset/quantile preservation 只对 hard one-to-one permutation 成立。若训练使用
doubly-stochastic matrix \(A\)，\(Az_v\) 通常只是 convex combinations：在双随机条件下可保总和和
均值，但不保 variance、quantiles 或 multiset。标准 NeuralSort/SoftSort relaxation 通常只保证
row-stochastic，不能一般性声称连总和也保持。

若需要声称 training forward 也 exact，必须使用 hard-forward permutation 加 STE 或其他明确的
surrogate backward。否则准确表述只能是“hard inference exact；soft training relaxation
approximate”。必须同时报告 soft-to-hard gap。

### Preserving score mass does not preserve pooled AP/ROC

hard permutation 精确保留每视频及全测试集的无标签 score histogram，也保留所有视频级对称
统计量，但 frame labels 固定在 timestamps 上。score-label association 改变后，pooled ROC-AUC 和
AP 可以改善或恶化。因此不能写“严格保持 pooled AP/ROC”；只能写“严格保持 score marginal 和
permutation-invariant video evidence”。上一轮约 0.053 的 pooled 降幅是否恢复仍是经验问题。

全 normal 或全 positive 视频的 score-label multiset 在置换后不变；真正被修改的是 mixed-label
视频中的 score-to-label association。这使机制与 within-video localization 对齐，但不构成 pooled
metric 非劣保证。

### The transported within-video order is the rank head order

若冻结 POWA values 无 ties，hard assignment 后 \(s'_v\) 与 \(q_v\) 在视频内具有完全相同的
ordering，因此二者的 within-video ROC-AUC 和 within-video AP 必须相同。若 POWA values 有 ties，
transported order 是 \(q_v\) order 的 coarsening，指标可能因 ties 略低，但不能凭 transport 本身
创造更好的 order。

因此 rank transport 的作用不是提高 rank head 的 within 指标，而是把 rank head 的 within order
与 POWA 的 absolute per-video marginal 组合起来。若实验中 full transport 的 within ROC/AP 显著
高于 raw \(q_v\)，首先应排查 tie convention、不同 frame grids 或评测错误。

### Ordinary MIL cannot identify the permutation

hard permutation 下，max、mean、top-\(k\) mean、log-sum-exp 等对称 bag aggregator 对 \(P\) 是
常数。标准 POWA MIL/BCE/outer-bag ranking 若只读取这些统计量，对 rank head 没有梯度和可识别性。
仅有 video labels 时，所有 permutation 对这类目标等价。

若 soft relaxation 让这些对称 MIL losses 看似能训练 \(P\)，训练信号来自 relaxation 不再严格保
multiset 的 leakage，hard inference 时可能消失。rank head 必须主要由真正
position/order-sensitive 的 train-only 信号训练，例如 donor-position pair/listwise loss、边界排除、
temporal consistency；这也是 benign insertion 在本方法中不可替代的 identifiability 来源。

### Identifiability, ties, and deployment

rank head 只通过 order 可识别；任何严格单调变换 \(g(q_v)\) 给出同一 hard output。损失和模型选择
应基于 order/listwise quantities，而不应把未经定义的 \(q\) magnitude 当成 calibrated anomaly
probability。

rank ties 或接近 ties 会使 hard assignment 对微小扰动敏感。stable tie rule 若按 timestamp index
处理，会在 rank collapse 时隐式注入片头/片尾 prior。必须固定并审计 reverse/random tie rules。
hard global sorting 还要求看到整段视频；本候选是 offline locator，不能声称 online/causal。

## Required ablations and falsifiable tests

### Core assignment controls

1. **POWA identity assignment**：保持 POWA 原始 order，作为严格起点。
2. **Uniform random within-video permutation**：复刻 Song and Lee 的诊断 control，验证仅保
   histogram 不足以定位。
3. **Raw rank head**：直接评测 \(q_v\)。在无 base-score ties 时，其 within ROC/AP 必须与 hard
   transport 相同。
4. **Full hard rank assignment**：冻结 POWA values，仅由 \(q_v\) 决定 stable permutation。
5. **Oracle GT assignment**：用 test GT 把最高 values 赋给正秒，只作为不可达到的 test upper
   bound；不得用于选择任何超参、checkpoint 或 tie rule。
6. **Reverse/random rank controls**：确认收益来自学到的内容排序而非位置先验。

### Constraint and architecture controls

7. **Unconstrained additive/blend head**：相同参数、特征和 insertion data，与 full 比较；证明收益
   来自 exact marginal constraint，而不是额外 head 或训练数据。
8. **V20 centered residual**：直接比较只保 mean 与保完整 multiset。
9. **Frozen versus jointly fine-tuned POWA**：若 joint fine-tuning 才有效，核心 invariant 与 claim
   失效。
10. **Hard-forward/STE versus soft Sinkhorn versus pairwise-trained hard inference**：说明
    differentiable OT 是否真的必要，并量化 relaxation/hardening gap。
11. **No-insertion/standard inner-bag ranking**：证明已知 donor positions 提供了标准 MIL 没有的
    identifiability。

### Insertion and shortcut controls

12. negative-donor versus positive-donor control；
13. donor edge versus interior，且 donor boundary 单独 mask；
14. 保留/去除插入 seam 的对照，防止 boundary detector shortcut；
15. donor identity、source video、长度和插入位置平衡；
16. non-donor recipient prediction/order consistency，防止模型只学“不是 donor 就应更高”；
17. same-corpus、train-only provenance audit；不得引入其他主数据集或 test 信息。

### Invariance and metric audits

每个视频都应自动 assert：

- `sort(output)` 与 `sort(frozen_powa)` 在规定精度下完全相同；
- mean/max/min/variance/top-\(k\)/quantiles 一致；
- rank head 与 transported output 的 Kendall/Spearman 关系符合设计；
- frozen POWA 的 unique-value ratio、tie rate 和 dynamic range 已报告；
- hard/soft outputs 分开保存，不混用评测。

最终至少报告固定三指标：pooled frame AP、pooled frame ROC-AUC、within-video macro ROC-AUC；
另报告 Cross-AUC 或跨视频 pair audit，说明 temporal reassignment 是否破坏 POWA 的 absolute
separation。所有 temperature、loss weight、tie rule 和 hardening policy 只能在 validation 选择。

### Falsification criteria

以下任一现象直接削弱或证伪核心解释：

- hard output 未精确保留 POWA multiset；
- transported within gain 超过 raw rank-head within gain，且不能由 base ties 解释；
- unconstrained additive head 在同等 pooled 约束下同样有效，full constraint 无独立贡献；
- 只有 soft Sinkhorn 有效，hard assignment 后收益消失；
- no-insertion 或 positive-donor control 与 negative-donor 同样有效；
- 收益只出现在 donor boundaries，interior 或 boundary-masked 后消失；
- rank head collapse，结果由 stable timestamp tie-break 决定；
- HMM within gain恢复但 pooled AP/ROC 仍出现与上一轮同量级下降。

## Fatal anti-patterns

1. **声称 soft OT 严格保 score distribution。** 双随机矩阵只保 mass/mean，不保原 values；
   row-stochastic relaxation 保证更弱。
2. **声称 multiset preservation 保证 pooled AP/ROC 不变。** 指标依赖 score-label association，
   该说法错误。
3. **用普通 permutation-invariant MIL loss 声称训练了 hard permutation。** hard 条件下该损失对
   permutation 是常数；soft 条件下可能只是 relaxation leakage。
4. **把 train-only augmentation 退化为 test-time fitting。** 禁止用 test GT/span、test metric、
   oracle assignment 或 per-test-video tuning 选择任何设计。
5. **把 negative-bag seconds 写成 ground-truth benign。** 合规称谓只能是“label-certified under
   the negative-bag assumption”；必须承认 video-label noise 的可能性。
6. **稳定排序偷带位置先验。** rank ties 时按原 timestamp index 分配会系统性偏向片头或片尾，
   必须做 reverse/random tie audit。
7. **学习剪辑 seam 或 donor identity。** 若 edge/interior、boundary mask、positive donor 和来源
   平衡 controls 不过，不能解释为 benign semantics/order learning。
8. **把独立训练 rank head + inference quantile mapping 包装成新 OT 模型。** 若没有 constraint
   ablation 和训练机制证据，它只是 post-hoc reranking。
9. **混用不同语料或含 span 标注的 supervision。** values、rank-head training、checkpoint
   selection 必须全部 same-corpus，并遵守 train/validation/test 边界。
10. **忽略 base-score ties/低动态范围。** exact reassignment 无法创造新 values；大量 ties 会形成
    不可突破的排序上限。
11. **宣称 online localization。** 全视频 hard sort 是非因果操作，除非另行提出并验证 causal
    assignment。

## Final novelty position

本候选相对 V20 的关键进步不是另一个 local head，而是把“local learner 不得改写 absolute score
mass”从均值级软约束提升为完整 empirical distribution 的硬不变量。它也把上一轮 benign
insertion 已观察到的 local ordering signal 限定为只控制时间关联，而不再自由改变 score
magnitude。

最安全的论文表述是：

> We learn a content-conditioned temporal permutation of a frozen detector's within-video empirical
> score distribution from train-only weak temporal interventions.

不要声称首次提出 score permutation、rank transport、exact histogram matching、video insertion、
inner-video ranking、NeuralSort 或 Sinkhorn。最终 novelty 是否成立，取决于 hard invariant 能否在
直接对照 V20、unconstrained head、random permutation 和 standard inner-bag ranking 时，保住
POWA pooled evidence 并继承可复现的 within-video ordering gain。
