# Independent novelty and mechanism review: policy-complete proposal MIL

**截至日期：2026-08-31**  
**审查对象：** `experiments/20260831_policy_complete_proposal_mil/README.md`  
**裁定：STOP，novelty 4.1/10。**  
**范围：** 仅审查 proposal，不实现、不训练、不运行新评测。

## 总结裁定

截至本次检索，没有发现 P-MIL 已用于 hateful-video detection 或 temporal localization。因此，“把 P-MIL 跨任务迁移到 hateful localization”满足用户关于来源方法的最低条件。

但当前 adaptation 仍不合格。它没有把 policy clause 变成 proposal 的新 latent semantics 或结构化合法性约束，而是：

1. 用 frozen prototypes计算 interval 内外的 role coverage；
2. 把编译后的标量作为 stop-gradient pseudo-label 的必要因子；
3. 再把 completeness log-odds以一个全局标量加到 P-MIL-style base logit。

这在实现层面是 **P-MIL completeness head + POWA-style policy feature/pseudo-label gate**。P-MIL 已占据 proposal classification、surrounding contrast和 completeness refinement；POWA 已在 hateful localization 中占据 hostile/target/context等 primitives、policy AST 与 MIL。候选没有提出新的 clause assignment、跨时刻 role binding、可识别的 clause ownership或新的 weak-label likelihood。因此它更接近组件拼接和 feature-derived pseudo-label substitution，而不是用户标准要求的 non-trivial adaptation。

对 HateMM，`hostile ∧ target ∧ ¬context` 至少提供一个有任务意义的 relational hypothesis；但候选 coverage 只要求两个 role 在同一个 interval 中出现，不证明它们构成同一攻击关系。对 HateClipSeg，violence/sexual/self-harm等 unary clauses使“policy-clause completeness”退化为普通 semantic saliency/completeness；此时与 `flat_role_max` control近乎同构，不存在统一的跨语料新核心。当前以 HateMM + HateClipSeg 作为首轮共同机制验证，因此这个退化是设计 blocker，而不是等结果出来再解释的风险。

项目中可用于方法判断的既有 **test** 证据也不支持 frozen policy primitives能承担 temporal ordering：corpus-specific POWA 的四个 within-video test门均未过；coalition witness在 HateMM/HateClipSeg test未胜 matched controls；dense Qwen3 pointwise teacher在正式 HateMM/HateClipSeg test的 within ROC也均低于 SOTA。这些结果不严格证明“proposal-level training永远不能利用 primitives”，但已经否定了“primitive/policy signal自身具有可靠双语料局部排序”的前提。当前候选使用更简单的 frozen semantic prototypes，却没有新增能修复该前提的监督来源。

因此本轮应在实现前停止。可保留为 `P-MIL + policy-feature` baseline proposal，不能作为主 novelty candidate。

## 1. P-MIL 是否已经用于 hateful task

### 检索结论

**未发现。** Ren et al., [Proposal-Based Multiple Instance Learning for Weakly-Supervised Temporal Action Localization, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html) 应用于 THUMOS14 与 ActivityNet。其核心包括：

- 训练和测试都直接处理 candidate temporal proposals；
- Surrounding Contrastive Feature Extraction；
- 用 pseudo instances与 proposal IoU产生 completeness pseudo labels；
- RGB/flow instance rank consistency。

2025–2026 的直接 hateful-video 文献中：

- [MultiHateLoc](https://arxiv.org/abs/2512.10408) 使用 modality-aware temporal encoders、dynamic fusion、cross-modal contrast和 modality-aware snippet MIL，没有 interval proposal completeness。
- [LELA](https://arxiv.org/abs/2602.09637) 是 training-free frame prompting/composition matching，没有 proposal MIL训练。
- [CLARA](https://arxiv.org/abs/2608.15905) 使用 utterance clips、MoE、local-global contrast和 VLM rationales做 video detection，没有 proposal completeness localization。
- [MATCH](https://jianlang.org/papers/MATCH.html) 的“proposer”生成 hate/non-hate evidential clues，由 LMM verifier核查；它不是 temporal interval P-MIL。
- [MoRE](https://jianlang.org/assets/papers/WWW-2025-MoRE.pdf) 使用 retrieval-augmented multimodal experts做短视频 detection，没有 proposal localization。

所以可以主张“P-MIL 尚未被直接应用到 hateful localization”，但不能由此自动推出当前 policy adaptation新颖。

## 2. 当前实现并不是 faithful P-MIL

README 将 `faithful_pmil` 描述为固定 `1/2/4/.../128s + whole-video` proposal grid、smooth-max bag pooling和自定义 normalized overlap readout。官方 P-MIL 则由 S-MIL生成 candidate proposals，并使用 classification/attention branches、top-k video aggregation、pseudo-instance IoU completeness和 RGB/flow rank consistency。

除非复现这些 load-bearing步骤，当前 control只能称为 **P-MIL-inspired exhaustive proposal MIL**，不能称 faithful P-MIL。这个差别影响 novelty归因：

- 固定 dense grid本身改变 proposal distribution和正负比例；
- smooth max替代 top-k；
- whole-video proposal会鼓励 video-level broadcast；
- 没有官方 pseudo-instance generation时，“原 completeness pseudo target”缺少可执行定义；
- normalized proposal-to-frame weights不是 P-MIL的 action-detection readout。

若 core胜过这个 control，不能确定收益来自 policy completeness，还是来自一个偏离 P-MIL的自定义 proposal system与其 pseudo targets互动。

## 3. Policy-clause completeness 是否 non-trivial

### 有效的任务故事

HateMM 中，短 proposal只覆盖 target mention或只覆盖 hostile predicate时，不应被视为完整 targeted-hate interval；同时出现 target、hostility且 surrounding不成立，确实比 generic action completeness更符合 label definition。这个动机是清楚且可证伪的。

### 为什么当前算子没有实现这个故事

README 的 targeted clause大致为：

```text
min(coverage_hostile(I), coverage_target(I)) * (1 - coverage_context(I))
```

这只验证两个 prototype responses共存于 interval。它没有验证：

- hostile predicate是否指向该 target；
- 两个证据是否属于同一说话者、引述或事件；
- target/context的 modality和时间归属；
- interval扩张后偶然纳入的两个高分峰是否构成关系。

因此 longer proposal天然更容易“完成”clause。inside-minus-surrounding可以抑制一部分扩张，但不能解决 interval内错误绑定。POWA原本至少使用 hostile-to-target transport表达跨时间绑定；候选删掉 binding 后只保留 AST pooling，是对现有目标任务机制的简化，不是更强的 adaptation。

此外，policy score只作为 stop-gradient pseudo target factor，最终又通过一个 scalar加回 base logit。模型可以令该 scalar接近零，或者 completeness head只复现 base confidence。policy结构没有进入 bag legal set、proposal latent assignment或可识别的 likelihood。它在优化上不是不可替代的核心。

### 与其他先例的边界

P-MIL及其后续 action-completeness工作已占“抑制只覆盖显著局部的短 proposal”；semantic-prototype WTAL与 cross-modal WTAL已占“用文本/semantic prototypes改善 action completeness”；weak video grounding和 relation grounding已占“用 compositional semantic roles选择完整 moment”。例如 [Video Object Grounding Using Semantic Roles, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sadhu_Video_Object_Grounding_Using_Semantic_Roles_in_Language_Description_CVPR_2020_paper.pdf) 明确指出独立累计 agent/verb/patient分数不能保证关系，并通过 role replacement构造 contrastive disambiguation；[Inverse Compositional Learning for Weakly-supervised Relation Grounding, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Inverse_Compositional_Learning_for_Weakly-supervised_Relation_Grounding_ICCV_2023_paper.pdf) 已处理 relation semantic integrity与弱监督 grounding。

这些工作不是 hateful localization，也没有当前 policy AST，但它们进一步压窄 claim：不能主张 compositional/role-complete proposal本身新；最多只能主张特定 moderation clauses 的 task adaptation。由于 POWA 已在目标任务使用这些 clauses，剩余变化仅是把 score送入 P-MIL completeness head。

## 4. HateClipSeg unary-clause 退化

HateClipSeg允许 violence、sexual、self-harm及可能的 untargeted abuse等单角色 clauses。令每个 unary clause为 `c_k(I)`，candidate score成为：

```text
max_k [c_k(I) - c_k(S)]
```

这就是多个 prototype saliency的 inside-versus-surrounding maximum。它不包含 clause composition，也不测试“证据全部出现才完整”。在该语料上：

- `policy-clause completeness` 与 flat harmful-role completeness是同一方法族；
- `flat_role_max` 若使用相同 inside/surrounding顺序，可能数学相同；若先 max再相减，也只是 `max(a_k-b_k)` 与 `max(a_k)-max(b_k)` 的次序差；
- 任意收益只能归因于 semantic prototype或 surrounding contrast，不能归因于 policy composition。

因此 README 要求 core在 HCS严格超过 `flat_role_max`，同时又不给 core任何非 unary结构，形成了近乎不可满足或不可解释的 attribution gate。若通过，最可能来自实现细节/聚合次序而不是 clause mechanism。

若把 HCS从主 claim拿掉，可把候选缩成 targeted-hate-only HateMM study；但项目目标是四个固定主语料、相同主机制，不允许结果后按 corpus切换 semantic rule。即使公开 label definitions不同，手工为每个 corpus加载不同 AST仍是 corpus-conditioned semantic routing，不能声称一个统一的跨语料 mechanism。

## 5. 既有 primitive/policy 负结果如何使用

按项目规则，validation performance不能 inform performance-oriented method development。因此，早期 validation-only 的 policy-gated recurrence、dense primitive qualification和 witness-path probe不作为本次 STOP的性能依据；它们只能说明已有设计重复度和需预注册的 controls。

可用于判断的是已经记录的 developmental **test** evidence：

1. Corpus-specific POWA policy pipeline在四语料 within-video test指标均未过当前门，HCS-only三 seed within约 `.521`。
2. Coalition witness test中，candidate在 HateMM/HCS均未胜 matched nonminimal controls；typed latent composition没有形成稳定局部排序。
3. Qwen3 dense pointwise teacher完整 test诊断中，HateMM/HCS within分别约 `.562/.540`，均低于各自 SOTA；更密的 semantic teacher也没有提供足够定位上限。
4. MultiHateLoc test error analysis显示真实问题是 per-video modality ownership错配，但 fixed policy prototypes没有提供 ownership supervision。

这些结果不逻辑排除“用 proposals重新训练后会改善”，所以不是单独的 impossibility proof。它们却使 candidate必须提出一个能改变 primitive semantics的机制。当前方法只是把相同 primitives变成 pseudo-label gate，没有这样的修复，因此 premise缺乏支持。

## 6. 不可识别性与退化

1. **Circular pseudo-labeling：** positive pseudo target要求 base confidence高；completeness head再影响同一 proposal logit。模型会确认 base head最初偏好的 interval，而不是发现完整 clause。
2. **Label broadcasting：** positive bag BCE只要求一个 proposal；whole-video proposal可同时获得最高 base confidence与最高 role coverage，成为稳定捷径。
3. **Length bias：** role coverage对 interval长度通常单调或近单调；多尺度 grid使长 proposal更易收集所有 roles。
4. **Prototype leakage：** `hostile`等 prototype本身直接编码 outcome；所谓 completeness可能只是另一个 hate classifier score。
5. **Context failure：** protected/reporting context用 `(1-context)`乘法会把 context覆盖整个 proposal，无法区分“视频包含报道语境”与“被引述片段本身是否违规”。
6. **Modality ownership未解决：** A/V/T先进入共享 encoder，role prototypes没有明确映射到各模态 witness；候选不能针对 MultiHateLoc已观察到的 branch ownership错误提出可识别修复。
7. **Scalar bypass：** final global coefficient可归零；若不归零，可能只做 score calibration。`base_only_readout`不足以证明训练期 policy pseudo labels load-bearing。
8. **Proposal-to-frame coverage：** 即使 constant logits平坦，不同长度和重叠 proposals的 learned weights仍可能使中心秒获得更高分；必须与 position/coverage-only readout比较。

## 7. 若作为 baseline 实现，最低必要 controls

STOP结论表示不建议把它作为下一主方法；若项目仍需要把它作为 P-MIL adaptation baseline，至少必须冻结以下 controls：

1. **真正的官方 P-MIL port或准确改名：** 若不复现官方 candidate generation、attention/classification、PCE与相关训练，只能称 P-MIL-inspired。
2. **Parameter-matched direct role feature：** 将同一六维 role coverage直接拼入 base proposal head，不生成 completeness pseudo labels。core必须超过它，才不是 feature substitution。
3. **Direct clause-logit control：** `base + alpha*clause_score`，不训练 completeness head。区分 pseudo-label learning与直接加分。
4. **Geometric PCE + clause feature control：** 保留原 P-MIL completeness，仅把 clause作为普通 input。区分“重新定义 completeness”与语义 feature。
5. **Flat unary equality audit：** 在 HCS逐 proposal验证 core clause score与 `flat_role_max`的代数/数值差异；若只差 max/subtract次序，HCS不计 mechanism evidence。
6. **Targeted-only subset control：** 在有 independent target/hostility evidence的 HateMM developmental test子集检查 `min(hostile,target)`是否超过 hostile-only、target-only、product、flat max。
7. **Role binding controls：** target与hostile时间独立 shuffle、跨视频 target replacement、同视频错配 target峰；共同 time shuffle不够，因为它保留二者共现关系。
8. **Length controls：** matched role mass下比较短/长 interval；role coverage须做长度校正，并报告 whole-video proposal成为 top proposal的比例。
9. **Pseudo-label dependence controls：** base-confidence threshold去除、clause threshold去除、stop-gradient去除，以及 pseudo target与 base score相关性。
10. **Training attribution：** 各 ablation独立重训。`base_only_readout`与`completeness_only_readout`只说明输出分支，不能归因训练时 pseudo targets。
11. **Position/coverage controls：** length-only、center-only、coverage-count以及 proposal weights在时间内循环移位。
12. **Missing modality/role calibration：** 分模态删除、prototype循环置换、role score norm与语言版本控制；不得按 corpus选择最有利 modality。

所有实现完成后应按项目规则立即在 test运行固定三指标；validation只选本次训练 checkpoint。test error analysis可 inform后续设计，但这些数字必须标记为 iterative/developmental evidence。

## 8. Claim 边界

当前版本不能主张：

- 新的 proposal MIL、proposal completeness或 SCFE；
- 首次 semantic/compositional proposal completeness；
- 新的 policy primitives、policy AST或 targeted-hate logic；
- 解决 modality ownership；
- 统一适用于 targeted hate与 HCS unary harm的 relational clause mechanism；
- faithful P-MIL adaptation，除非复现官方 load-bearing pipeline。

若未来加入真正的 learned/latent role binding，并通过 controls，最窄可考虑的 claim是：

> A task-specific adaptation of proposal-based weak temporal localization in which proposal completeness is defined by jointly bound moderation-policy roles, rather than by geometric overlap with a pseudo action instance.

当前方法不满足“jointly bound”：它只有 interval内 role pooling。因此当前最多只能写：

> A P-MIL-inspired baseline whose proposal-completeness pseudo labels are filtered by frozen moderation-role prototype coverage.

这不是足够的主方法 novelty claim。

## 9. 最终决定

**STOP，4.1/10。**

- P-MIL本身尚未进入 hateful localization，这一点通过。
- HateMM的 policy-completeness动机合理，但当前 coverage没有建模 role binding。
- POWA primitives/AST已经用于同一目标任务；将其标量输出接入 P-MIL completeness pseudo labels是组件拼接。
- HCS unary clauses把核心压成 ordinary prototype saliency，与 flat-role control无法形成清晰机制差异。
- 已有 test evidence表明 primitive/policy signals没有可靠双语料 temporal ordering；候选没有新增能修复该信号的 supervision或 latent structure。
- 当前所谓 faithful P-MIL control实际上偏离官方 proposal generation、aggregation与 refinement，进一步削弱可归因性。

不建议实现或运行正式 pilot。若要救这个方向，必须先提出不依赖 corpus-specific unary/relational routing、能显式绑定 role instances且在 HCS仍有非退化语义的 proposal likelihood；那将是另一个候选，不是对当前 README 的小修。
