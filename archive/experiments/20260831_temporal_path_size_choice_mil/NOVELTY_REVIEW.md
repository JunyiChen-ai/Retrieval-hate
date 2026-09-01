# Temporal Path-Size Choice MIL：独立查新与机制审查

截至 2026-08-31。审查对象：本目录 `README.md`。本轮只查新与审机制，未实现、未训练、未生成 prediction。

## 裁定

**GO（仅进入修订后的 frozen test premise；尚不批准 formal training）**  
**novelty：6.6 / 10**

按当前 novelty 标准，它勉强但明确过线：Path-Size Logit（PSL）是可迁移的外部 source；截至本次检索，未发现
PSL、按 candidate-set link occupancy 定义的等价 size correction，或带该 correction 的 latent proposal
likelihood 已用于 hateful-video detection/localization。把原本具有 observed route choice 的 PSL 改成只有 bag
label 的 latent temporal choice，并用 negative bag 的 outside option闭合 likelihood，再把同一 posterior 边缘化成
唯一 frame score，是有任务含义的 adaptation，不只是换应用名。

但新意很窄。普通 proposal MIL、outside/background class、proposal posterior、coverage marginal、overlap relation
和多样性学习都已有充分先例；真正可保留的新点只有：**用整个 temporal candidate set 的秒级占用关系修正 latent
proposal choice mass，以抑制弱监督下由近重复枚举产生的 evidence-multiplicity bias**。这是一项新的 task
adaptation，不是新的离散选择模型或通用 MIL 原理。

当前 README 的 premise gate 不能充分证伪 whole-video 风险，而且 learned `beta` 与文中 duplication-invariance
陈述不一致。因此 GO 只允许先修订并运行下面固定的 premise；若不修订，应视为 **STOP**。

## 查新范围与结论

只采用作者论文、出版社/会议论文页或作者机构版本作为方法依据；通用网页检索仅用于定位，不作为方法证据。
检索覆盖 PSL / route-choice、MIL / weakly supervised detection、WS-TAL、temporal grounding / proposal
detection，以及截至当前可检索的 hateful-video detection/localization。未发现直接占用不能证明绝对不存在，故
任何首次性表述都必须带 `to our knowledge` 并限定任务与公式。

### 1. Source method：已有，且候选所用式子确属经典 PSL 家族

PSL 把 choice-set 中共享 link 的程度写为 route utility correction。Duncan 等对 PSL 系列的梳理明确指出：path
size 衡量 route distinctiveness，共享 link 越多越受罚；也明确记录了 choice set 中不合理长 route 会反向改变
其他 route 的 path-size term。这既支持候选来源，也直接支持当前担心的 candidate-set composition 风险：

- Duncan et al., *Path Size Logit route choice models: Issues with current models...*, TR-B 2020，
  [作者机构版本](https://eprints.whiterose.ac.uk/id/eprint/156992/)，
  [DOI](https://doi.org/10.1016/j.trb.2020.02.006)。
- Frejinger and Bierlaire, *Capturing correlation with subnetworks in route choice models*, TR-B 2007，
  [DOI](https://doi.org/10.1016/j.trb.2006.06.003)。该工作把 PSL/C-logit 放在显式 route-correlation
  建模的谱系中；它不是 video 或 MIL 方法。

因此不能 claim 新的 PSL、overlap correction、choice-set correction 或 duplicate-alternative correction。

### 2. MIL / weakly supervised proposal learning：未发现同式 core，但问题和组件已有

- Ilse et al. 的 attention MIL 已用归一化 attention 对 bag instances 聚合，但不按 instance 间几何重叠或
  choice-set occupancy 修正重复 evidence：
  [ICML 2018 / PMLR](https://proceedings.mlr.press/v80/ilse18a.html)。
- Tang et al. 的 PCL 把同一物体附近的重叠 proposals 聚成小 bags，以减少 proposal ambiguity；这是最接近的
  weakly supervised spatial precedent之一，但它以 seed/IoU clustering 产生伪 instance supervision，不是
  `log PS` 修正的 bag likelihood：
  [CVPR 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Tang_PCL_Proposal_Cluster_CVPR_2018_paper.html)。
- Azadi et al. 已把 proposal similarity / overlap 放进可微 DPP likelihood，在训练和推理中鼓励非重复检测集合。
  因而“首次在训练中考虑 overlapping proposal correlation”不可成立。它依赖 box supervision、学习 subset
  diversity，而本候选只有 video label、建模一个 latent choice 的枚举质量：
  [CVPR 2017 author preprint](https://arxiv.org/abs/1704.03533)。

结论：未找到 PSL 等价式进入 MIL bag likelihood 的直接先例；但“弱监督 proposal clustering / 可微 overlap
diversity”已被占用，所以 adaptation 的新意只能落在集合依赖的 choice-mass correction 与 hate-localization
机制上。

### 3. WS-TAL / temporal proposal detection：proposal relation 很拥挤，未发现等价 size correction

- Ren et al. 的 P-MIL 已在 train/test 都直接分类 candidate proposals；PCE 先以 NMS 式过程选 pseudo instances，
  IRC 再对每个重叠 proposal cluster 的相对分数做跨模态一致性，推理仍用 Soft-NMS。它显式处理 overlap cluster，
  但没有按候选枚举占用修正整包 choice mass，也没有 outside-option latent likelihood：
  [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html)。
- Li et al. 的 2026 ACL-Net/PACL 用 overlapping proposals 的相对 temporal geometry 构造 process-completeness
  reference score并校准 proposal quality。这是当前 WTAL 最近邻，已占用“训练时由重叠关系改善 proposal
  scoring”的宽 claim；其目标不是候选集复制偏差，也不是 PSL-normalized bag evidence：
  [TIP 2026, DOI 10.1109/TIP.2026.3697621](https://doi.org/10.1109/TIP.2026.3697621)。
- P-GCN/后续 proposal graph 把 overlapping 与 nearby proposals 连边做上下文传播；同样是 relation modeling，
  不是 alternative-count correction：
  [ICCV 2019 paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Zeng_Graph_Convolutional_Networks_for_Temporal_Action_Localization_ICCV_2019_paper.html)。
- Soft-NMS 按当前最高分 proposal 与其余 proposal 的 IoU 顺序衰减分数，是 inference suppression；它既不形成
  同一个 bag likelihood，也不具有 PSL 的全 choice-set、link-occupancy dependence：
  [ICCV 2017 paper](https://openaccess.thecvf.com/content_ICCV_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html)。

结论：不能 claim 首次 temporal overlap modeling、首次 proposal relation、首次 overlap-aware training 或无需
NMS。可以说检索未发现以 PSL/等价 occupancy factor 修正 WS-TAL latent bag choice mass。

### 4. Temporal grounding：密集候选与邻接关系已有，未发现同式归一

- 2D-TAN 把 `(start,end)` candidates 放在 2D temporal map 上并联合编码 adjacent candidate relations，还专门
  比较 sparse sampling 与 dense enumeration；它解决 representation/context 和计算冗余，不校正一个候选被
  多次枚举后在 bag log-sum-exp 中的总质量：
  [AAAI 2020 paper](https://ojs.aaai.org/index.php/AAAI/article/view/6984)。
- BANet-APR 明确指出 overlapping/nearby proposals 的 pooled representations 难区分，用边界特征、proposal
  interaction 和 adaptive selection refine proposals；仍不是 choice-set size correction：
  [ACCV 2022 paper](https://openaccess.thecvf.com/content/ACCV2022/html/Dong_Boundary-aware_Temporal_Sentence_Grounding_with_Adaptive_Proposal_Refinement_ACCV_2022_paper.html)。

### 5. Hateful video：未发现 source/core 被使用

检查的直接相邻工作包括：

- MultiHateLoc：modality-aware temporal encoder、dynamic fusion、contrastive alignment 和 modality-wise top-k
  MIL；无 temporal proposal choice-set correction：
  [arXiv:2512.10408](https://arxiv.org/abs/2512.10408)。
- LELA：training-free LLM/captioning localization；无 proposal likelihood：
  [arXiv:2602.09637](https://arxiv.org/abs/2602.09637)。
- TANDEM：跨模型 reinforcement-learning temporal grounding；无 PSL proposal MIL：
  [arXiv:2601.11178](https://arxiv.org/abs/2601.11178)。
- CLARA：clip-level MoE、local-global contrast与 VLM rationale，用于 hateful-video detection；无 proposal
  choice-set correction：
  [arXiv:2608.15905](https://arxiv.org/abs/2608.15905)。
- HateClipSeg 提供 segment-level hateful-video tasks/benchmark，本身不占用该训练机制：
  [arXiv:2508.01712](https://arxiv.org/abs/2508.01712)。

在这个目标领域，source/core 的直接占用未检出。

## 它是否只是 overlap penalty / NMS 重写

**不是 NMS 的数学重写，但确实可精确写成一个固定结构先验加到普通 MIL aggregator。**

候选 bag log-odds 为：

`logit P(y=1|V) = logsumexp_p(u_p + beta log PS_p) - v_0`。

所以：

1. `beta log PS_p` 是由候选集合决定的 additive utility bias；除这个 bias 外，likelihood 是普通 categorical
   latent-choice / log-sum-exp MIL，`v_0` 是普通 background/outside alternative。
2. 它与 NMS/Soft-NMS 不等价：NMS 是按分数排序后的离散删除，Soft-NMS 是相对当前 winner 的顺序衰减；PSL
   同时依赖整个候选集合，直接改变 bag probability、所有 proposal gradient 和训练/测试共用的 posterior。
3. 它也不同于固定 pairwise IoU penalty：当前 `PS_p` 按 proposal 内每秒的覆盖数聚合，能区分具有同样 pairwise
   IoU 但秒级 coverage multiplicity 不同的 candidate sets。
4. 但如果没有 candidate duplication/density stability、cluster/dedup、length prior 和 Soft-NMS controls，结果只能
   说明“某个固定 overlap prior 有用”，不能归因于 choice-set multiplicity correction。

### 当前数学陈述必须修正

README 说 `beta >= 0` 可学习，同时又说 `J` 个完全重复 proposal 不会获得 `J` 倍总 mass。后者只在相同
`u_p` 且 **`beta=1`** 的理想重复组内精确成立：其总质量按 `J^(1-beta)` 缩放。若 `beta` 学到 0.4 或 1.6，
分别仍会膨胀或过度收缩；部分重叠 proposal 的复制还会改变其他 proposal 的 `n_t` 和 `PS`。因此：

- 不得把 learned-beta core 宣称为严格 duplication invariant；
- `beta=1` 必须是 formal arm，learned beta 是额外 arm；
- 若核心 claim 需要精确 invariance，应固定 `beta=1`，否则 claim 改成“reduces enumeration sensitivity”。

此外，只有 video label 时，`u_p` 可以补偿 `beta log PS_p`；bag likelihood 对 beta 的机制识别弱于 observed-route
PSL。需要报告 beta、其梯度和跨 seed 稳定性，不能只根据最终指标断言模型学会了 overlap correction。

## Frozen test premise gate 审查

### 当前版本：不足以证伪 whole-video shortcut

`whole-video top` 减少而 within ROC 不降并不充分，原因有四个：

1. top 可以从精确 whole-video proposal 移到 `0.8T` 的 near-whole proposal，风险未消失；
2. PSL 对 edge seconds 的低 `n_t` 会奖励含稀有边缘的长 proposal；只数 exact whole 无法看见这种 mass shift；
3. frozen P-MIL 的 proposal score 未被定义为 `u_p` 的 logit，`v_0` 也不存在；不同转换会改变 correction 结果；
4. README 未固定 premise 的 frame readout。proposal max、posterior coverage sum 和长度归一 coverage 会产生不同
   within ranking，不能在分析时任选。

### GO 前必须冻结成无歧义 gate

不扫描 beta、不按 corpus 选规则；一次性固定以下定义：

1. **Utility**：明确 frozen P-MIL 哪个 proposal arm 是 `s_p`，固定
   `u_p = logit(clamp(s_p, eps, 1-eps))`；报告 finite/coverage。若原 score 不是概率，直接固定使用其 pre-sigmoid
   logit，不能事后换变换。
2. **Premise posterior/readout**：对 proposals 用 `softmax_p(u_p + log PS_p)`；frame score固定为覆盖该秒的
   posterior sum。premise 内不要凭空拟合 outside logit；outside 不影响 proposal-conditional ranking，应留到训练
   arm验证 negative-bag likelihood。
3. **正确/错误 proposal**：预先给 IoU 判据，例如仅对有 oracle proposal 的 positive videos，`top IoU < .3`
   定义错误 top，`oracle IoU >= .5` 定义可纠正；比较错误 top 与该视频 best-IoU proposal 的 `log PS`，同时报告
   有效视频数。不能把“correct top”留成分析后解释。
4. **whole/length 风险同时过门**：两语料均要求 exact-whole top fraction下降；top-duration ratio 的中位数不升，
   `duration >= 2T/3` 的 top fraction下降，且 long-proposal posterior mass 不升。至少一项 near-whole 量必须严格
   改善，防止只换成长 proposal。
5. **定位方向**：HMM/HCS 的 within ROC 均不下降；pooled ROC/AP 同时完整报告但不设可被视频级 offset 轻易满足
   的替代门槛。
6. **枚举稳定性**：对同一 frozen utility 做预注册的 exact-duplicate、near-duplicate 和 proposal-grid
   thinning/densification perturbation；`beta=1` 相对 `beta=0` 必须降低 bag logit 与 frame ranking 的变化。

该 premise 使用已经冻结的 test prediction 与 test GT，按项目规则可以直接 inform development；结论必须标成
developmental test evidence，不能称未揭盲验证。以上所有定义需在看结果前写入 README。任一语料方向失败即
`STOP_BEFORE_FORMAL_METHOD`，不允许扫描 beta、换 PS 公式、换 length cutoff 或按语料设规则。

## Formal pilot 必须 controls

premise 通过后，最小 formal pilot 除 README 三臂外必须包含：

1. `beta=0 + outside`（严格 capacity control）；
2. `beta=1` PSL core；learned-beta 只能作为额外 arm，不能替代固定 core；
3. proposal-count correction：`logsumexp(u)-log|P|`，排除只是 generic bag-size normalization；
4. matched length prior：给 utility 加同容量的 `a log|p| + b`，排除 duration/edge bias；
5. IoU cluster/dedup control：同一 frozen threshold、每 cluster 一个代表或 cluster-level log-mean-exp，排除任何
   proposal 去重都会有效；
6. Soft-NMS 或固定 pairwise-overlap suppression control，明确训练中与纯 inference 去重的差异；
7. exact/near duplication 与 candidate-density perturbation stability，不允许只在原候选集报指标；
8. outside-only negative-bag诊断：正负 bag logit、outside posterior、proposal posterior entropy；
9. whole、near-whole、top-duration、long-proposal posterior mass，以及多段 GT 视频分层。单 categorical witness
   对多段 hate event 没有计数可识别性，不能只看全体平均；
10. 四主语料独立 train、validation 仅选 checkpoint、训练完成后立即 test 三固定指标；test-informed 方法结果
    均标 developmental。不得跨主数据集训练，不得做 ensemble/calibration/routing。

机制 gate 应比较 `beta=1 core` 与完全同 backbone/proposals/readout 的 `beta=0 + outside`，而不是与原 P-MIL
不同 readout 的结果直接归因。若只赢 pooled 而 within 不赢，或 whole-top下降但 near-whole mass不降，机制失败。

## 允许的 claim

若 premise、controls 和正式 test gate 全部通过，最多允许：

> To our knowledge, this is the first adaptation of route-choice path-size correction to weakly supervised
> hateful-video temporal localization. It treats overlapping temporal proposals as a latent, correlated choice set,
> trains an outside-option bag likelihood from video labels, and marginalizes the same posterior to a single frame
> score, reducing sensitivity to proposal enumeration density.

中文等价表述：**据我们所知，首次把 route-choice path-size correction 改造成弱监督 hateful-video temporal
localization 的 latent proposal choice likelihood，用 candidate-set occupancy 修正重叠 proposal 枚举质量，并
将同一 posterior 边缘化为唯一帧分数。**

除非固定 `beta=1` 且 invariance controls 通过，不得使用“duplication invariant”；应使用“降低对枚举密度的
敏感性”。不得 claim 首次 proposal MIL、首次 outside option、首次 overlap-aware proposal learning、首次
temporal proposal relation、首次无需 NMS，或通用的新 MIL/choice model。

## 最终决策含义

候选值得花一次很小的、预注册的 frozen premise 成本，因为它直接针对当前 test error analysis 发现的重叠枚举
与 whole-video readout 风险，且目标领域未发现 core 占用。它不值得跳过 premise 直接实现：PSL 的已知
choice-set composition问题在这里可能正好奖励长 proposal；若修订后的 HMM/HCS gate 任一失败，立即 STOP，
不做公式搜索。
