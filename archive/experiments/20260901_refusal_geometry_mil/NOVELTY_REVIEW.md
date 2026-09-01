# Independent Novelty Review — Negative-Recentered Refusal Geometry MIL

截至 2026-09-01。审查对象仅为本目录 `README.md` 中已冻结的候选；未审代码、未实现、未运行实验，也未提出候选修补方案。

## 最终裁定

**STOP — 4.3/10。**

三门结论为：

1. **Gate 1（允许 adaptation 既有方法）：PASS。** Refusal direction、multimodal activation re-centering 和 weakly supervised MIL 都可作为跨任务来源被 adaptation。
2. **Gate 2（来源方法未被 hateful-video detection/localization 占用）：PASS（窄口径）。** 实际联网检索未发现 Arditi refusal direction、MARS 的 textual refusal steering 或 ReGap 的 safety-geometry drift correction 已被用于 hateful-video detection/localization。MARS 虽包含 video jailbreak safety evaluation，但任务是 MLLM refusal/guardrail safety，不是 hateful-video detection 或 temporal localization。
3. **Gate 3（task adaptation non-trivial 且机制成立）：FAIL。** 候选声称 task-load-bearing 的 `negative-video recentering` 对定义出的标准化 refusal coordinate 只是全语料常数平移，随后被标准化严格消除；对 orthogonal content 分量也只产生一个可由 head bias 吸收的常数平移。去掉这个无效差异后，方法只剩 frozen MLLM activation 的一个 scalar projection、一个普通 content head 和标准 positive-top-k/negative-dense MIL。这正是本门明令不能接受的“frozen embedding 加 scalar + 普通 global/local MIL”，而不是 non-trivial hateful temporal localization adaptation。

任一门失败必须 STOP，因此该候选不得实现或训练。

## 一手来源核查

### Refusal geometry 与多模态来源

- [Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS 2024](https://arxiv.org/abs/2406.11717) 证明在多个 chat model 中存在一维 refusal-mediating direction，并通过加法与擦除干预验证其对拒绝行为的因果作用。候选用 harmful/harmless 文本 activation mean difference 构造 `d`，在思想上与该来源一致。
- [D'Incà et al., *Harnessing Textual Refusal Directions for Multimodal Safety*](https://arxiv.org/abs/2606.31876) 的 MARS 已经把 textual refusal direction 扩展到 image/video，使用 activation re-centering、adaptive steering、layer selection，并在 first generated token 处干预。因而“文本 refusal direction 跨模态 + re-centering + first-token activation”本身是来源已有部分，不能算本候选 novelty。
- [Guo et al., *Safety Geometry Collapse in Multimodal LLMs and Adaptive Drift Correction*](https://arxiv.org/abs/2605.18104) 明确研究 text-aligned refusal direction 与 modality-induced drift，指出多模态输入会压缩 refusal separability，并以 drift correction 恢复 safety behavior；ReGap 是 training-free inference-time safety correction。该工作进一步占用了“multimodal safety geometry collapse/drift correction”的来源概念。

固定 harmful/harmless 文本集合形成 mean-difference direction，在高层定义上是 source-faithful；在 assistant first-token 读取 activation 也与 MARS 的 operational site 相符。但 brief 没有 Arditi 式的 addition/erasure 因果 qualification，因此它最多能称为按来源方法构造的 candidate direction，不能预先称为 Qwen2.5-VL 中已经验证的 refusal mediator。这个问题削弱来源忠实度，但不是本次 STOP 的唯一依据。

### Hateful-video 目标领域占用检索

- [LELA](https://arxiv.org/abs/2602.09637) 使用五模态 caption、多阶段 prompt、frame-level hate scoring 与 composition matching 做 training-free hate localization。
- 同名但不同方法的 hate-reasoning [MARS](https://arxiv.org/abs/2601.15115) 使用 objective description、hate-supporting evidence、counter-evidence 和最终综合判断做 training-free hateful-video detection。
- [SafeLens: Segment-Level Hate Speech Detection in Online Videos](https://ojs.aaai.org/index.php/AAAI/article/view/42390) 在 segment level 融合 speech、text、visual，并由 policy LLM 输出 label、confidence、reason 与 harm category。
- [CLARA](https://arxiv.org/abs/2608.15905) 使用 fine-grained clips、MoE multimodal alignment、local-global segment contrast 和 VLM-derived rationale gated Transformer。
- [LEAF](https://aclanthology.org/2026.findings-acl.604/) 使用 self-grounding CoT 与 stage-wise distillation，把 LMM explanations 蒸馏到轻量模型。

这些目标领域工作已经覆盖 frame/segment prompting、hate/counter-evidence reasoning、clip-level local-global modeling、policy LLM rationales 与 explanation distillation，但上述一手资料中未见 residual-stream refusal direction、activation ablation/addition、textual refusal steering 或 modality-drift correction 被用于 hateful-video detection/localization。因此 Gate 2 不因广义“用了 MLLM safety”而误判失败；其结论是窄口径 PASS。

## Gate 3 机制审查

### 1. Negative recentering 在当前定义下不是 load-bearing

令 `d` 为单位向量，候选定义：

`c = mean(a_neg) - mean(a_text-neutral)`

`r(a) = <a - c, d> = <a, d> - <c, d>`

`u(a) = (a - c) - r(a)d = (I - dd^T)a - (I - dd^T)c`

这里 `c` 对该语料所有 train/validation/test window 都是同一个常数向量。

- 对 `r`，recenter 只减去常数 `<c,d>`。只要 `standardize(r)` 是通常的 train-locked z-score，训练均值也同减该常数，core 与 uncentered 的标准化 coordinate **逐 window 完全相同**；标准差也不变。
- 即使 `standardize` 不是去均值的 z-score而只做固定尺度缩放，`c` 仍只改变所有 window 的共同 intercept，不改变任何视频内或全池 pairwise ordering。可训练 logit bias 可以吸收该常数。
- 对 `u`，recenter 同样只增加固定的 orthogonal translation `-(I-dd^T)c`。带 bias/normalization 的小 temporal head 可以吸收它；它没有提供随 window 变化的 negative-derived correction signal。

因此，“negative train video 的每个局部 window 都被认证为 non-hateful”这一 weak-supervision 事实并没有以局部约束进入 refusal coordinate。它只被压缩成一次全局 translation，而 translation 在后续标准化或 bias 中消失。候选声称用 same-corpus negative geometry 修正 modality drift，但公式并未产生 sample-dependent 或 time-dependent drift correction。

这不是一般性的 shortcut 猜测，而是候选已冻结公式的直接等价关系，符合 Rule 12 允许 pre-run STOP 的“adaptation 明确只是直接套用/组件拼接”情形。

### 2. 单调 `beta` 没有让 refusal direction 成为不可绕开的任务机制

`z_t = q_t + softplus(beta) * standardize(r_t)` 只限制 refusal scalar 的符号：

- `softplus(beta)` 可以趋近于零，训练可以实质关闭该分支；
- 高容量 `q_t` 仍可完全承担 bag classification 和 temporal ranking；
- negative-all BCE 与 positive-top-k MIL 都是标准 binary MIL 监督，没有要求正视频的 selected witness 必须由 refusal coordinate 提供，也没有要求 `r_t` 对局部预测贡献达到非零下界；
- `q_t` 使用 `u_t` 而非完整 `a_t`，确实删除了对 `d` 的线性分量，但这只形成 feature decomposition，不阻止 orthogonal content features 独立完成任务。

因此 monotonicity 防止“把方向翻转后当任意 feature”，但没有把该方向变成 task-identifying constraint。它不足以把剩余方案从“额外 scalar feature + 普通 MIL”提升为 non-trivial adaptation。

### 3. Matched controls 只能测边际效果，不能恢复 recenter novelty

- `content_only` 能检查 scalar branch 是否对结果有边际贡献。
- `random_direction` 能检查 true direction 是否优于一个 norm-matched arbitrary coordinate。
- 但是 `uncentered` 无法隔离声称的 recentering：按上述公式，标准化 `r` 与 core 相同，而 `u` 仅差可吸收常数。它预期不是一个能产生机制差异的 matched arm。
- 即使未来 core 胜 `content_only` 与 `random_direction`，也只能说明 true refusal projection 作为 frozen scalar feature 有预测价值；不能证明 negative-video recentering 修复了 modality drift。
- 四秒 overlapping window 与 triangular overlap-add 只是固定 temporal sampling/readout。它可以把 window score还原到 1fps，但不提供新的 supervision、局部识别约束或 refusal-geometry adaptation。

### 4. 与项目已关闭机制链的关系

`research-wiki/STATUS.md` 已记录：

- `context-residual ASM-Loc` 因 global constant/gated local branch 不能建立 within ordering 而关闭；
- `dense-negative marked splat` 与 `Negative-anchored D2` 已表明 negative-all dense supervision 本身是标准 binary MIL，不能作为新机制；
- `policy-patch spatial excess` 已因 zero-shot/frozen prompt scalar 接普通 MIL 而在第三门停止；
- `LELA`、hate-reasoning `MARS`、POWA contextual negation、LB-SCGP stance semantics 与 Exception-Competitive Prompt MIL 已覆盖 prompt/counter-evidence 目标领域邻域。

本候选的新 activation information source 与这些旧输入不完全相同，所以不据此判 Gate 2 失败；但在 task-specific recentering 被代数消去后，剩余结构正好退化为项目已明确拒绝的 frozen scalar + global/local head + dense-negative/top-k MIL 组合。新术语 `refusal geometry` 没有改变这条训练和 readout 信息链。

## 逐项结论

| 审查项 | 结论 | 理由 |
|---|---|---|
| 固定文本模板形成 direction 是否 source-faithful | **有限 PASS** | mean-difference 与 first-token site 符合 Arditi/MARS 高层做法；但未做来源式 causal qualification，不能预先把它视为已验证 mediator。 |
| Negative recenter 是否 task-load-bearing | **FAIL** | 它是 dataset-wide constant translation；在标准化 coordinate 中严格消失，对 content head 也只是可吸收的常数。 |
| `beta` 单调是否阻止绕开 direction | **FAIL** | 只能约束符号，不能保证非零贡献；content head 和普通 MIL 可独立完成训练目标。 |
| Controls 是否隔离机制 | **部分 FAIL** | random/content-only 可测 scalar 边际价值；uncentered 与 core 在核心 coordinate 上等价，不能隔离 recentering。 |
| 4秒 overlap-add 是否构成 novelty | **FAIL** | 仅是 deterministic temporal readout。 |

## 最终评分

- 跨任务来源清晰度：`1.5/2.0`
- 目标领域未占用证据：`1.5/2.0`
- task-specific mechanism 的非平凡性：`0.6/3.0`
- 机制 load-bearing 与 controls：`0.4/2.0`
- 可证伪性与边界陈述：`0.3/1.0`

**总分：4.3/10，STOP。**

该裁定只关闭当前冻结定义的候选；按任务要求，不在本审查中提出替代设计或修补路径。
