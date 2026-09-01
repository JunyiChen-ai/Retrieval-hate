# Independent novelty review: Policy-constrained cluster transport MIL

截至 2026-09-01。审查对象仅为本目录 `README.md` 中冻结的候选机制；未审代码、实现质量或训练结果。

## Verdict

**GO，6.5/10。** 三道 novelty 硬门逐项为 `PASS / PASS / PASS`。允许进入最小实现与正式 HMM/HCS validation search + test，但必须在运行前把下述 **binary/unconstrained CASE control** 写进冻结实验设计；这不是再开启一轮 novelty review。

最窄可成立 claim 是：

> 将 CASE 的 binary foreground/background OT self-labeling 改造成由同语料 negative bags 锚定 background、由 moderation policy 限定可行 primitive states、并允许 positive-unselected snippets abstain 的 partial-label transport；transport assignment直接监督同一个最终 policy compiler 所使用的 primitive logits。

不能 claim 首次 snippet clustering、首次 optimal transport、首次 prototype learning、首次 background modeling、首次 policy-aware hateful-video model，或首次 latent primitive learning。

## 实际检索与目标任务占用

检索了以下组合及其同义表达：`CASE hateful video`、`Clustering-Assisted F&B Separation HateMM`、`snippet clustering hateful video`、`optimal transport self-labeling hateful video`、`policy-constrained latent clustering hateful video`、`policy-valid primitive clustering HateMM/HateClipSeg/MultiHateClip`。同时核对当前目标任务的主要 temporal/localization 工作与数据集论文。

- CASE 的正式来源是 Liu et al., ICCV 2023，[Revisiting Foreground and Background Separation in Weakly-supervised Temporal Action Localization: A Clustering-based Approach](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Revisiting_Foreground_and_Background_Separation_in_Weakly-supervised_Temporal_Action_Localization_ICCV_2023_paper.html)。其核心是 multi-cluster snippet clustering、cluster foreground/background classification，以及带分布约束的 OT self-labeling；实验任务是 THUMOS14 与 ActivityNet WTAL，而非 hateful video。
- 当前 hateful-video temporal 工作中，[MultiHateLoc](https://arxiv.org/abs/2512.10408) 使用 modality-aware temporal encoders、dynamic fusion、cross-modal contrastive alignment 与 modality-aware MIL；[LELA](https://arxiv.org/abs/2602.09637) 是 training-free multi-stage prompting 与 composition matching；[CLARA](https://arxiv.org/abs/2608.15905) 使用 clip encoder、local-global segment contrast 与 VLM rationale；[HateClipSeg](https://arxiv.org/abs/2508.01712) 提供多类 segment annotation 与 temporal benchmark。检索到的这些目标任务工作没有采用 CASE 式 snippet-cluster/cluster-classification OT self-labeling，也没有采用本候选的 policy-feasible latent transport。
- 检索到名为 TOT 的 hate 方法是 meme 场景中的 cross-modal topology-aware OT alignment，不是 hateful video、不是 temporal snippet clustering，也不是 CASE self-labeling。因此不能据“hate + OT”字面重合判定来源已占用。

这是截至审查日的未检出结论，不是对所有未公开工作的绝对不存在声明。

## 三道硬门

### Gate 1：允许 adaptation 已有方法

**PASS。** 候选明确承认并保留 CASE 的 load-bearing clustering + OT self-labeling，不把已有组件冒充从零提出的方法。

### Gate 2：来源方法未被 hateful-video task 占用

**PASS。** 实际检索未发现 CASE、其 snippet-cluster/cluster-classification OT 核心，或精确的 policy-constrained latent cluster transport 被用于 hateful-video detection/localization。目标任务中存在 MIL、contrastive learning、policy categories、prototype/semantic evidence 与其他用途的 OT，但这些都不等于占用所 adaptation 的 CASE 核心。

### Gate 3：task adaptation non-trivial 且机制可证伪

**PASS，边界偏窄。** 本候选不是只把 action 类名改成 hate：它改变了 OT 可行域和监督语义。

1. negative bags 提供 certified background admission，而不是 CASE 在 action-only bag 内依据当前 foreground prediction自举背景；
2. positive bag不是一个 generic foreground state，而是 `background + corpus-policy-valid primitives + abstain` 的 partial-label feasible set；
3. policy-invalid primitive不能接收 mass，clause group约束决定列容量，因而同一 representation geometry 在不同 policy 下会产生不同 pseudo assignment；
4. assignment监督 POWA 最终 compiler实际读取的 primitive logits，cluster/OT在 inference 删除，所以若训练约束不起作用，不存在额外 inference branch掩盖失败。

这些改动共同对应 hateful temporal localization 的具体结构：positive bag可能由 targeted hostility 的组合 clause，或 HCS 的 abuse/violence/sexual/self-harm alternatives 成立；未被选择的正视频片段不能被强迫成 harmful。只要实现确保 harmful clause 获得非零的 required/feasible mass，而不只是设置一个从不激活的上限，这个 delta 就是 load-bearing 的。

这里登记一个实现前必须保持的 claim boundary：如果所谓 clause-level constraint最终只有“禁止 policy-invalid states”，同时 background/null容量足以吸收全部 positive mass，那么 core 可退化为全 background/null，policy transport不再 load-bearing；这种实现不属于本次 GO 覆盖的候选。

## 与项目内最近方法逐项比较

| 最近方法/失败链 | 与本候选相同处 | 不同且 load-bearing 的部分 | 是否严格同构 |
|---|---|---|---|
| Multimodal P-MIL / policy-complete proposal MIL | 都是 weak bag label 下产生 latent supervision | P-MIL 的单位是 interval proposal，核心为 completeness/PCE/IRC；本候选不生成 proposal，以 snippet-to-global prototype transport改变 representation 与 primitive logits | 否 |
| Policy-Simplex Background-Uncertainty MIL | 都使用 POWA typed primitives、background动机与 policy permutation | 旧候选删除来源的 feature-norm/pseudo-background/magnitude核心，只剩 entropy/energy head拼接；本候选完整保留 CASE clustering + constrained OT，约束对象是 assignment polytope，不是另加 scalar residual | 否；但最接近，必须由 control 证明 policy constraint而非普通 clustering有效 |
| Background-anchored event-slot MIL | 都有 background 与多个 latent states | event-slot 是每视频 exchangeable slots、noisy-OR与slot marginal，已因slot collapse/普通normal prototype失败；本候选是跨数据全局 typed prototypes与逐snippet mass conservation，不 claim 多事件slot分工 | 否 |
| 普通 background prototype / carrier-energy chain | 都利用 negative bags | 旧链本质是 binary normalized head/prototype distance；这里 background只是七状态 constrained transport的一列，核心比较对象是 policy-feasible assignment，不是单一异常距离 | 否；若去掉 typed feasibility 后效果不变，则实际有效部分会退化到该失败链 |
| POWA 原 primitive BCE / sparse grounding | primitive logits、semantic anchor与固定 compiler相同 | 原训练是逐点 sparse BCE、negative sparsity与bag BCE；新信号是满足 row mass、column/clause feasibility和abstention的联合 assignment，跨 snippets竞争，不能分解成独立逐点 BCE | 否；若 OT target等价于对当前 logits逐点 threshold，则 novelty claim失败 |

因此当前候选没有通过换术语、换数学工具严格重开一个已关闭机制。它仍复用 POWA semantic primitives，但新增约束是 CASE-faithful的 joint transport，不是 lexical DCC、self-derived top-K responsibility、proposal completeness、slot decomposition或普通 prototype head replacement。

## Permuted control 是否足够

**单独使用不够。** 循环置换六个 primitive 到 policy leaf 的映射能测试语义对应关系，但不能可靠隔离“policy-constrained transport”本身：

- trainable cluster prototypes与primitive heads可能协同重命名，部分抵消置换；
- HCS admissible alternatives覆盖多个 harmful primitive，循环置换后 feasible set可能大体不变；
- 即使 core胜置换，也可能只是 frozen semantic anchors与错误leaf冲突，而不是 clause-level mass feasibility带来的增益。

正式 test 至少要增加一个 matched **`binary_or_unconstrained_case`** control：完全相同 POWA backbone、七个prototype容量、OT求解、loss weight、训练量与checkpoint选择，但把 positive transport改为 CASE式 generic foreground/background，或允许全部 harmful primitives采用相同非policy列预算；保留background和abstain总预算。Core必须在 HMM/HCS within 都胜该 control，并且两语料方向一致。这个 control直接回答 task delta是否 load-bearing。原 permuted control可以保留为语义归因的第二 control，但不能作为唯一 matched control。

## 可证伪预期与最终裁定边界

当前预期可证伪：正确 policy-feasible transport应同时改善 HMM/HCS within ordering；negative-anchored background应使每个语料至少一个 pooled指标胜 matched POWA anchor；core还应胜 binary/unconstrained CASE control。若只胜 permuted、不胜 binary/unconstrained，结论只能是 semantic anchoring敏感，不能支持 policy-constrained transport claim。若普通 clustering control追平、OT harmful mass接近零、或最终有效部分只是 background prototype/primitive BCE，深度 novelty复审应判定真正有效部分不 novel。

## 最终结论

`GO 6.5/10`。来源未被目标任务占用；相对 CASE 与 POWA 的 adaptation目前足够 non-trivial，也不与 failure ledger中的 P-MIL、policy-simplex、event-slot或 ordinary background prototype严格同构。分数未更高的原因是 policy可行域按语料固定、primitive identity存在重参数化风险，且原 brief 的 permuted control不足。补上 binary/unconstrained CASE matched control后应立即进入实现、一次基础 technical review、完整 validation search与双数据集 test，不继续扩展 novelty 论证。
