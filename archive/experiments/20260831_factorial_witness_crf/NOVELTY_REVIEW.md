# Novelty review: modality-subset latent CRF

**截至日期：2026-08-31**  
**审查对象：** `experiments/20260831_factorial_witness_crf/README.md`  
**结论：PASS（仅限极窄的任务 adaptation claim）；ownership/可识别性 claim 不通过。**

## 核心裁定

这个 proposal 不是新的 factorial HMM、latent CRF、MIL bag constraint、exact partition 或 forward-backward 方法。最接近的先例 MI-DORF 已经把“时间序列 latent instance states + bag/instance cardinality potential + 对 latent paths 精确求和的条件似然与 forward-backward inference”组合起来。候选的 binary collapsed 版本——负 bag 只允许全空路径，正 bag 允许至少一个非空状态——本质上就是 standard dynamic MIL cardinality semantics。

在新标准下仍给 **PASS**，原因只在于一个更窄、且对任务有实质作用的 adaptation：把标量 latent state 改成 audio/visual/text 三个有类型 bit 的 8-state modality-subset chain，并在同语料 binary video labels 下联合学习 temporal localization 和 modality-subset posterior。检索未发现这套具体状态空间与合法路径似然已用于 hateful-video detection/localization。它改变了训练时的 latent hypothesis space 和 partition，不只是把一个额外 loss 接到现有网络，因此可以算 non-trivial adaptation。

但 README 中“解决 modality ownership 错配”的机制故事目前不成立。binary bag label 只监督 `empty` 与 `nonempty`，并不监督非空状态里究竟是哪一个 modality bit。该模型可以减少显式的逐模态 label copying，却不能仅凭此识别真实 ownership。论文若把 per-modality posterior 表述为 identifiable、causal 或 ground-truth ownership，应判 **FAIL**。

## 最近的直接先例与边界

| 工作 | 已覆盖的核心 | 与本候选的剩余差异 |
|---|---|---|
| Deselaers & Ferrari, [A Conditional Random Field for Multiple-Instance Learning, ICML 2010](https://icml.cc/2010/papers/87.pdf) | CRF 中联合 bag/instance latent assignment；正负 bag 的 MIL 语义 | 图结构不是视频时间链，也没有 typed modality subsets |
| Hajimirsadeghi et al., [Multiple Instance Learning by Discriminative Training of Markov Networks, UAI 2013](https://www.cs.sfu.ca/~mori/research/papers/hajimirsadeghi-uai13.pdf) | binary latent instance labels、高阶 cardinality clique；正 bag 至少一个 positive、负 bag 全 negative；可 tractable inference | 没有 temporal chain，也没有 modality ownership states |
| Ruiz et al., [Multi-Instance Dynamic Ordinal Random Fields, ACCV 2016](https://arxiv.org/abs/1609.01465) | 时间 latent states、bag-level weak label、cardinality potential、精确 conditional likelihood 和 forward-backward | 这是最接近的算法核心；候选只新增 typed three-bit subset state 及 hateful-video application |
| Li et al., [Weakly Supervised Energy-Based Learning for Action Segmentation, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Weakly_Supervised_Energy-Based_Learning_for_Action_Segmentation_ICCV_2019_paper.pdf) | HMM/GRU energy、对符合/不符合弱约束的 paths 做动态规划求和 | 监督是 action transcript/order，不是 binary bag；没有 modality subsets |
| Tian et al., [Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/513_ECCV_2020_paper.php) | 从 modality-agnostic video labels 学 temporal 与 audible/visible/both parsing；已占据 weak time×modality ownership 问题 | attentive MMIL，不是 exact temporal CRF；两模态且非 hateful video |
| Cheng et al., [JoMoLD, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940424.pdf) | 明确处理 overall event label 在单个 modality 中缺失造成的 modality-specific label noise | 用 modality-label denoising，不是 subset-state path partition |
| [VALOR, NeurIPS 2023](https://arxiv.org/abs/2305.17343) | 从缺少 timestamp/modality 的 video labels 构造模态与片段监督 | 借助 modality-specific pretrained teachers；不是仅靠 binary bag partition |

因此不能主张“首次用 latent CRF 做 weak temporal localization”“首次 exact marginalization weak paths”“首次从 video labels 学 modality ownership”或“新 factorial inference”。枚举 8 个 joint states 并运行普通 linear-chain DP，也不构成新的 factorial inference algorithm；更准确的名称是 **three-bit modality-subset linear-chain CRF** 或 **typed dynamic-MIL CRF**。

## 可识别性 blocker

令 `Z` 为全部 8-state paths 的 partition，`Z0` 为唯一 all-empty path 的权重。README 的 bag likelihood 等价于：

```text
P(y=0|x) = Z0 / Z
P(y=1|x) = (Z - Z0) / Z = 1 - P(all-empty|x)
```

这个目标只识别“是否有某个非空 witness”，不识别非空 witness 的模态类型：

1. 若同时置换 audio/visual/text bit、相应 unary heads 和参数，bag likelihood 不变。Hamming transition 完全 permutation-symmetric；若 coalition cost 只依赖 subset cardinality，它也不破坏该对称性。
2. 即使输入 feature streams 有类型，灵活的 unary heads 仍可利用相关性把一个模态的证据归给另一个 state。typed input 是 inductive bias，不是 ownership supervision。
3. 每个 positive video 永远选 visual-only、audio-only 或任何一个容易分类的 branch，都能满足 bag label。coalition penalty 最多把“所有模态同时亮”变成“某一个模态独占”，不能决定哪一个才正确。
4. 因而 union frame posterior 可以被 bag labels 学到；per-modality posterior 的语义则存在 label switching 和 shortcut collapse。没有额外 anchor 时，不能把它作为真实 ownership 证据。

这是 performance risk，也是 claim blocker，但不是极窄 adaptation novelty 的 blocker。

## 退化与反模式

- coalition cost 趋于很大时，模型退化为空状态加三个互相竞争的 singleton MIL heads；趋于相反方向时，all-modal state 主导，重新产生 label copying。
- Hamming transition 为零时，退化为逐时刻 subset MIL/noisy-OR；很大时，路径近乎整段常量，退化为 video classifier/broadcast score。
- 任一 unary branch 占优时，退化为 single-modality two-state dynamic MIL。
- 只有一条 negative legal path，却有 `8^T - 1` 条 positive legal paths。相同能量下，`P(y=1)=1-8^{-T}`，模型天然带有极强的长度与 path-entropy 偏置。它可通过 empty-state bias 部分抵消，但需要随长度变化；也可能通过产生大量低置信非空路径优化 bag likelihood，而不是形成稀疏、局部 witness。
- 当前项目 test error analysis 已显示固定 temporal smoothing 对 HateMM 与 HateClipSeg 方向相反。Hamming persistence 是 learned generic smoothness，不应在没有 retrained attribution controls 前被叙述为跨语料 fragmentation solution。

## 必须做的 attribution controls

以下都应按主数据集独立训练，仅由 validation 选择各自 checkpoint，随后立即在 test 报固定三指标。只在同一 checkpoint 上切换 inference 不足以归因训练机制。

1. **Collapsed 2-state dynamic-MIL CRF**：同一 encoder/capacity，状态仅 empty/nonempty。证明 8-state typed subsets 超过 generic temporal MIL。
2. **Independent unary union**：移除 transition，用相同 bag likelihood。证明 temporal path structure 有贡献。
3. **Three independent per-modality 2-state chains + union/noisy-OR**：证明 joint subset partition 与 coalition interaction 超过三个普通 branch 的组合。
4. **Zero-coalition、zero-transition、两者同时为零**：分别归因 coalition 与 temporal structure。
5. **Cardinality-only states**：只保留 0/1/2/3 个 active modalities，不保留 modality identity；与 typed 8-state model 区分“subset identity”与单纯更多状态/容量。
6. **Parameter-matched temporal encoder + noisy-OR/top pooling**：排除收益仅来自 encoder 和参数量。
7. **Exact marginalization 对 hard Viterbi/MAP training**：证明 partition over paths，而非任意 latent sequence learner，是 load-bearing choice。
8. **模态 permutation 与 ablation**：置换输入 streams、置换 state names、遮掉各模态，报告 union score 变化、各 bit occupancy、singleton/multimodal state 占比和跨 seed label switching。若 ownership 语义真实，结果应随输入语义而非 state index 稳定变化。
9. **长度/path-entropy tests**：同一 benign sequence 的重复、padding 和 temporal resampling；按视频长度分层报告 false-positive score。必须证明 positive probability 不由序列长度主导。
10. **localization shortcut tests**：temporal shuffle/reversal、全局复制 feature、transition-strength 与预测 span length。证明模型不是 video-level evidence broadcast。
11. **ownership developmental analysis**：允许在 test predictions 上比较 modality posterior 与可得的 span/modality evidence，明确标记为 iterative test evidence；不得把该分析用于 checkpoint selection，也不得据此声称 confirmatory ownership identification。

## 最窄可主张表述

> To our knowledge, this is the first adaptation to weakly supervised hateful-video temporal localization of a three-bit audio/visual/text modality-subset linear-chain CRF, trained from same-corpus binary video labels by exactly contrasting the unique all-empty path with the partition over paths containing at least one non-empty state.

建议紧接限制句：binary bag supervision identifies non-empty temporal evidence but does not by itself identify the semantic modality owner; modality posteriors are structured latent attributions, not ground-truth ownership.

## 最终 PASS/FAIL

**PASS：** 作为“MI-DORF/dynamic-MIL semantics 向三模态 subset-state hateful localization 的 non-trivial adaptation”，并严格使用上述最窄 claim。没有找到相同机制已用于 hateful-video detection/localization 的直接先例。

**FAIL：** 若 claim 是新的 CRF/MIL/exact-partition/factorial inference，或声称 binary video labels 足以识别真实 modality ownership。若 collapsed 2-state、independent-chain、cardinality-only controls 中任一 matched control 达到同等结果，也应撤回 8-state ownership mechanism claim，将收益归为 generic temporal MIL、容量或平滑效应。
