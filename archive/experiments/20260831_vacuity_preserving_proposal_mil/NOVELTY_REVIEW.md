# Independent novelty and mechanism review: vacuity-preserving proposal MIL

**截至日期：2026-08-31**  
**审查对象：** `experiments/20260831_vacuity_preserving_proposal_mil/README.md`  
**裁定：STOP，novelty 4.8/10。**  
**范围：** 仅文献与机制审查；未实现、未训练、未运行新评测。

## 结论先行

截至本次检索，没有发现 Cascade Evidential Learning（CEL）、subjective-logic vacuity或 evidential proposal completeness 已用于 hateful-video temporal localization。P-MIL和 CEL均来自 WTAL/Open-World WTAL，不是 hateful-video 方法；因此跨任务来源资格本身通过。

但是当前 proposal 不满足“non-trivial且机制统一”的标准，原因不是 evidential learning过于常见，而是 README 声称的核心 algebra 与训练语义并不成立：

1. `e=softplus(g)` 对所有有限 `g` 严格大于零；模型不会产生 README 所称的 exact zero-evidence opinion。因此 vacuous modality不是严格 identity，除非另有 hard availability mask把 evidence显式置零。
2. README把三件不同的事都称为 ignorance：视频边界外的 context是**已知物理缺失**；observed modality里没有 hateful cue可能是**background evidence**；positive bag里某 modality是否不承担事件是**未标注 latent ownership**。后二者不能从 video labels仅靠 Dirichlet strength自动区分。
3. Negative bags被要求对每 modality/proposal产生 background evidence。这与“modality-local absence of hate evidence应保持 vacuous”直接冲突：无 hate的 observed modality会被训练成高 `e_b`，不是 `e_h=e_b=0`。
4. Masked SCFE只是正确处理 padding的 deterministic geometry fix；evidential PCE/fusion是 learned uncertainty机制。两者没有共享状态、共享守恒量或共同 likelihood。把二者都解释为 ignorance不能使它们成为一个算法核心。
5. PCE teacher仍由同一模型的 opinion产生，再用其 reliability训练自身 completeness，保留 circular self-training。Vacuity可以改变谁有资格当 teacher，但没有外部信号保证低 vacuity等于正确 localization。

因此当前版本更准确地说是：**zero-padding bug fix + P-MIL + evidential pseudo-label weighting + evidence-level multimodal fusion**。这些模块各有直接先例，组合有合理工程动机，但当前没有形成可主张的新统一机制。

可以保留为强 baseline proposal。若要重新进入 novelty review，至少需要先把 physical missingness、observed benign evidence和 latent modality non-ownership形式化为不同状态，并给出能在 binary bags下识别或约束它们的训练目标。

## 1. 文献占位：是否已进入 hateful video

### Cascade Evidential Learning 与 WTAL

Chen et al., [Cascade Evidential Learning for Open-World Weakly-Supervised Temporal Action Localization, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Cascade_Evidential_Learning_for_Open-World_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html) 已用 evidential learning、multi-scale temporal context和 knowledge-guided prototypes逐步收集 known action、unknown action和 background evidence。它已占据：

- WTAL中的 evidential known/unknown/background separation；
- subjective uncertainty用于 temporal localization；
- 多尺度 context与 prototype evidence的联合建模。

同一作者的 [Uncertainty-Aware Dual-Evidential Learning for WTAL, TPAMI 2023](https://pubmed.ncbi.nlm.nih.gov/37624714/) 又直接处理 action-background ambiguity与弱监督 localization uncertainty。因而不能主张“首次用 vacuity/evidence解决弱时序定位中的 background ambiguity”。

检索未发现 CEL/UDEL被应用到 hateful-video detection/localization。这一点符合用户的跨任务 adaptation规则，但只留下目标任务 adaptation，而不是方法原语 novelty。

### Hateful content中的 uncertainty/evidence

Yang et al., [Uncertainty-Aware Cross-Modal Alignment for Hate Speech Detection, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.1475/) 已在 hateful memes上用跨模态分布差异估计 uncertainty，并据此动态平衡 unimodal和cross-modal features。2026年的 MHM-DS也已用 Dempster–Shafer evidential fusion进行 hateful-meme classification。它们不是 hateful **video localization**，但已经占据“uncertainty-aware multimodal hate fusion”的宽 claim。

2026年的 [An Interpretable Agentic Framework for Multimodal Hate Video Analysis with Explicit Evidence Attribution](https://doi.org/10.1145/3774905.3796488) 甚至已把多模态信号表示为独立 evidence，并把 modality absence标为 null；其方法不是 Dirichlet/P-MIL，也不做本项目的弱监督 dense localization，但进一步表明“hateful video中把缺失模态视为非贡献 evidence”不是开放的宽叙述。

因此可以主张的目标空缺只可能是：**proposal-level weak hateful localization中，以 evidential reliability同时控制 PCE teacher eligibility和 multimodal proposal fusion。** 当前 README还没有把这个窄点做成可识别机制。

## 2. 最近、最直接的跨任务先例

| 工作 | 已占据的部分 | 与候选剩余差异 |
|---|---|---|
| [P-MIL, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html) | proposal-level training/testing、SCFE、pseudo-instance PCE、跨 view rank consistency | 没有 evidential opinion/vacuity，也非 hateful video |
| [CEL, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Cascade_Evidential_Learning_for_Open-World_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html) | weak temporal localization中的 evidence、unknown/background与 uncertainty | snippet/action级 open-world WTAL；没有 multimodal proposal PCE |
| UDEL, TPAMI 2023 | dual evidential WTAL和 action/background ambiguity | 没有缺失模态 identity或 proposal completeness |
| [CO2-Net, ACM MM 2021](https://arxiv.org/abs/2107.12589) | appearance/motion cross-modal consensus；两支互作 pseudo targets | 候选删除强制 mutual teaching，以 vacuity允许 abstention |
| [JoMoLD, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940424.pdf) | video label在某 modality缺失造成的 modality-specific noise；动态移除 noisy modal labels | 不用 subjective opinion，也不处理 proposal PCE |
| [NREP, TNNLS 2025](https://pubmed.ncbi.nlm.nih.gov/40030688/) | weak AVVP中用 modalitywise/temporalwise evidential learning抵抗 noisy pseudo labels，并做 foreground-background consistency | 与候选非常接近；候选剩余差异仅是 interval proposals、PCE与 hateful adaptation |
| [UWAV, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lai_UWAV_Uncertainty-weighted_Weakly-supervised_Audio-Visual_Video_Parsing_CVPR_2025_paper.html) | weak AVVP中 uncertainty-weighted segment pseudo labels、temporal dependencies、模态事件缺失 | 候选将 uncertainty放到 proposal/pseudo-instance completeness，而非 segment parsing |
| [Trusted Multi-View Classification, ICLR 2021](https://arxiv.org/abs/2102.02051) | per-view Dirichlet evidence、subjective opinions、evidence-level dynamic fusion与 uncertainty | 非 temporal/MIL；但占据 evidential multi-view fusion原语 |
| [Conformal Fusion Under Missing Modalities, 2026](https://arxiv.org/abs/2608.07183) | absent modality提供 vacuous evidence并被融合规则结构性忽略；per-modality evidential decomposition | 几乎完全占据“missing modality = vacuous identity”的宽 claim；非 hateful/P-MIL，且另有 conformal calibration |

NREP和 UWAV尤其重要：它们已经把 uncertainty/evidential reliability用于 weak video parsing的 modalitywise/temporal pseudo supervision。当前候选不能只说“vacuity让无关 modality不当 teacher”；必须证明 proposal completeness产生了不同于已有 uncertainty-weighted pseudo-labeling的训练语义。

## 3. Masked SCFE 与 evidential机制是否统一

### Masked SCFE本身合理

README正确识别了 zero padding的语义错误。whole-video proposal两侧没有观测 context；把缺失侧写成零向量后计算 `inside-zero`，确实可能制造强 contrast。显式 `a_L,a_R` 并令缺失侧 contrast为零，是正确的 representation fix。

但这个 fix不需要 subjective logic：一个 binary/continuous availability mask已完全知道缺失程度。它既不是 epistemic uncertainty，也不需要从数据学习。reflection padding、replicate padding、只使用存在侧、masked pooling等简单方法都能处理同一 bug。

### Evidential PCE/fusion是另一机制

modality/proposal vacuity是模型从弱标签学习的 latent confidence。它是否对应真实缺失或错误 teacher并不知道。`masked_scfe_only`和`evidential_only_zero_context` controls能测两项增益，但即便 core超过两个单因素，也只说明 interaction/synergy；它不证明二者属于同一 ignorance algebra。

如果要主张统一，availability必须进入同一个 opinion定义，例如 physical missingness对相应 evidence施加硬零约束，并证明 fused opinion在 temporal-context mask与modality mask下满足同一明确不变量。当前 SCFE availability只进入 descriptor，不进入 opinion algebra，所以“两个轴上的同一机制”是叙述，不是模型性质。

## 4. Vacuous identity 的数学问题

README定义：

```text
e_h,e_b = softplus(g(x))
S = e_h + e_b + 2
b_h = e_h/S, b_b = e_b/S, u = 2/S
```

对于有限 `g(x)`，`softplus(g)>0`。特别地，`g=0`时每类 evidence约为 `.693`，不是零。因此：

- `u<1`始终成立；
- 单个 modality永远不是 exact vacuous opinion；
- cumulative evidence fusion中它永远不是严格 identity；
- 只有 `g→-∞`的极限才接近 identity，训练没有保证达到该极限。

若 physical modality确实缺失，必须用 observed availability mask显式定义 `e_m=a_m*softplus(g_m)`。这只能为**物理缺失**提供 exact identity，不能为“该 modality没有 hateful event”提供 hard mask，因为后者没有 modality label。

使用 ReLU可以产生 exact zero，但 2026年的 [Generalized Regularized Evidential Deep Learning Models](https://pubmed.ncbi.nlm.nih.gov/41632661/) 已分析 non-negative evidential activations在 low-evidence区域的 learning-freeze问题。激活函数和 regularizer不是无关实现细节，必须冻结为方法定义。

## 5. Weak labels 下的可识别性与 collapse

### All-vacuous collapse

若 positive bag loss直接作用于 hate belief，完全 all-vacuous不能满足 positive label：`b_h→0`。negative bag若要求高 background belief，也不能以全 vacuity获得最优。因此严格 all-vacuous global solution不是主要风险。

### 真正的退化

1. **Single-modality monopoly：** positive bag只要求一个 proposal/modality产生 hate evidence。一个易学 branch可承担所有 positive bags，其余始终 vacuous或background；bag loss无法识别真实 modality ownership。
2. **Background domination：** fused evidence为逐模态相加。如果一个 branch给 `e_h=H`，另两个给大 `e_b`，则 `b_h=H/(H+Σe_b+2)`；observed但无事件的 branches不是 identity，而会压低有效 hate evidence。
3. **Correlated evidence double counting：** audio、ASR text与visual不是条件独立来源。简单相加会重复累计相关 evidence；这不是 conflict-aware Dempster–Shafer fusion，也没有去冗余保证。
4. **Unbounded evidence scale：** 正确类 evidence可无限增大，使 `u→0`。vacuity取决于 logit尺度和 regularizer，而非独立可识别的 epistemic state。
5. **Low-evidence pseudo-label starvation：** early training时所有 proposals的 `Σr`小，PCE没有 pseudo instances；completeness head失去正监督，最终 score又乘 completeness，形成自我维持的低分状态。
6. **Circular confidence：** opinion决定 pseudo instances和teacher reliability，PCE再强化这些 proposals。低 vacuity可能只是模型早期自信，而不是正确性。
7. **Whole-video persistence：** masked SCFE移除 artificial boundary contrast，但 whole-video inside embedding仍最稳定地携带 bag label。MIL和 PCE仍可能选择 whole-video proposal。

### `q_I` 的双重 evidence-strength weighting

README令 `r=1-u=E/(E+2)`，再计算 `r*b_h`，其中 `b_h=e_h/(E+2)`。分子实际为：

```text
e_h * E / (E+2)^2
```

即 evidence strength被使用两次。这个量不是标准 Dirichlet expected hate probability，也不是纯 belief；它会强烈压低中低 evidence teacher。可以有工程理由，但必须承认是自定义 confidence heuristic，并与仅用 `b_h`、expected probability和单次 reliability weighting比较。

## 6. 相对各方法的最窄边界

- **P-MIL：** 候选不拥有 proposal/PCE/SCFE；只可能拥有 evidential teacher eligibility和fusion的目标任务 adaptation。
- **CEL/UDEL：** 候选不拥有 evidential weak temporal localization或 background uncertainty；区别只在 multimodal proposals、PCE和 missing-view semantics。
- **UCA：** UCA已在 hate detection中用 uncertainty平衡 unimodal/cross-modal features。候选不能主张首次 uncertainty-aware hate fusion；区别是 video proposal localization和 zero-evidence abstention。
- **NREP/UWAV：** 已在 weak multimodal video parsing中使用 evidential/uncertainty weighted temporal pseudo labels。它们是最接近的跨任务 mechanism controls；候选必须证明 proposal completeness而非普通 pseudo-label weighting是 load-bearing。
- **JoMoLD：** 已占“video label不应复制到每个 modality”。候选的区别是 abstention opinion而非 loss-based label removal，但不能主张发现 modality-specific label noise。
- **CO2-Net：** 已占 cross-modal mutual pseudo teaching；候选删除 all-pair consensus是合理差异，但“不要让无信息 view当 teacher”本身已被 JoMoLD/NREP/UWAV占据。
- **Missing-modality evidential fusion：** TMC、MCCF等已占 per-view evidence、vacuity和 missing view identity。候选只能主张这些原理在 proposal PCE/hateful localization中的 adaptation。

## 7. 当前 controls/gates 的评价

### 已经做得对的部分

- `masked_scfe_only`、`evidential_only_zero_context`与 core形成必要的两因素分解。
- `forced_nonvacuous`、`vacuity_time_shuffle`和`probability_average`试图分别检验 abstention、定位对应关系和fusion algebra。
- `no_pce`能检查 PCE是否仍有价值。
- whole-video top比例、常数 view、pseudo-instance贡献和 proposal geometry诊断直接对应现有 test错误。
- 要求各 arm独立重训，避免同 checkpoint删组件冒充训练归因。

### 仍不足以支持 claim

1. `vacuity_time_shuffle`会把 evidence strength与 class belief拆成训练中不可能出现的组合，不能替代 entropy/max-probability confidence gate。
2. `probability_average`只排除一种fusion；没有排除 sum logits、max single-view、TMC/Dempster–Shafer conflict-aware fusion或简单 learned reliability gate。
3. 没有 hard physical availability mask control，无法验证 exact identity。
4. 没有 simple padding controls，不能证明 masked SCFE超过 reflection、replicate、one-sided SCFE或learned missing token。
5. 没有 modality-dropout/known missingness测试，无法检查 `u`是否真的识别 missing view。
6. 没有 fixed-total-evidence control，不能区分收益来自 class belief还是总 logit/evidence scale。
7. 没有 NREP/UWAV-style confidence-weighted pseudo-label baseline，无法证明 PCE-specific adaptation。
8. 只报告 `u`高于 shuffled不等于 uncertainty calibrated；需要 missing/constant/corrupted modality detection AUROC、risk-coverage或 error-vacuity association。
9. 首轮单 seed可作淘汰 gate，但不能支持 uncertainty机制的稳定性 claim。

## 8. 若降格为 baseline，必须增加的 controls

1. **Padding family：** zero、masked、replicate、reflection、one-sided valid context和learned missing token；同一 P-MIL训练预算。
2. **Hard-mask evidential identity：** known missing modality直接令 evidence为零；对比仅依赖 learned `u`。
3. **Confidence-gate baselines：** softmax entropy、max probability、energy/logit norm和learned scalar reliability，以相同方式控制 PCE teacher。
4. **Fusion baselines：** sum logits、probability mean、max branch、TMC/DS conflict-aware fusion和简单 availability-masked mean。
5. **Pseudo-label baselines：** confidence-weighted segment/proposal pseudo labels但无 PCE IoU head；区分 NREP/UWAV式 weighting与 proposal-completeness贡献。
6. **Evidence-scale controls：** fixed total evidence、temperature rescaling、只打乱总 strength不打乱 class ratio、只打乱 class ratio不打乱 strength。
7. **Synthetic missing/corruption：** train/test中独立删除、置常数、错位每个 modality；检查 vacuity是否随已知 corruption单调变化。
8. **Single-view collapse：** 报告每语料正 bag winner modality、每 branch evidence mass和删除 winner后的性能；与 best single-modality retrained model比较。
9. **PCE starvation：** 每 epoch有资格 proposals比例、positive pseudo-instance coverage、无正 PCE监督视频比例和 whole-video pseudo-instance比例。
10. **Reliability formula：** `b_h`、Dirichlet expected probability、`r*b_h`和一次/二次 strength weighting直接比较。
11. **Conflict tests：** 构造一个 view高 hate evidence、一个 view高 background evidence的proposal，检查 fusion是否简单由总 evidence较大者支配。
12. **Localization shortcuts：** whole-video proposal删除、length-matched proposal、position/coverage-only score和 temporal shuffle。

所有 trainable controls应独立训练；validation只用于各自 checkpoint selection，完成后立即在 test运行固定三指标。test error analysis可继续用于方法开发，但结果属于 developmental evidence。

## 9. 最窄可保留 claim

当前不可使用 README中的强 claim，因为“vacuous modalities remain identity elements”在 `softplus`公式下不成立，“missing context与modality absence是同一机制”也没有模型约束支持。

若仅作为 baseline，准确表述是：

> A P-MIL adaptation that masks unavailable temporal context and uses proposal-level evidential strength to weight multimodal pseudo-instance generation and proposal fusion for weak hateful-video localization.

这描述的是两个改动，不声称统一 vacuity algebra。

若未来显式 hard-mask physical missing evidence、区分 benign/background与 latent modality absence，并使同一 opinion operator控制 context和view两个轴，才可考虑：

> A proposal-level evidential adaptation in which explicitly unavailable observations contribute a vacuous identity, while learned proposal opinions control both completeness supervision and multimodal fusion.

即便如此，也不能主张新的 subjective logic、evidential fusion、uncertainty-weighted pseudo labels或 missing-modality identity；只能主张它们在 hateful proposal localization中的 non-trivial adaptation。

## 10. 最终决定

**STOP，4.8/10。**

- CEL/subjective evidential WTAL未被 hateful-video localization直接使用，来源资格通过。
- UCA和 evidential hateful-meme methods已经占据 uncertainty-aware hate fusion；NREP/UWAV占据 weak multimodal video中的 evidential/uncertainty pseudo supervision；TMC/MCCF占据 vacuous missing-view fusion。
- 当前唯一潜在新组合是 proposal PCE + hateful adaptation，但 masked SCFE与 learned vacuity是两个不同机制。
- `softplus`使 exact zero evidence/identity claim在数学上不成立。
- Negative background supervision与“无 hate modality应 vacuous”语义冲突，binary bags不能识别 modality-local absence。
- Cumulative evidence容易出现 background domination、correlated-view double counting、single-modality monopoly和 PCE starvation。
- 现有 controls方向良好，但不足以排除 padding fix、ordinary confidence gating、evidence-scale和已有 uncertainty-weighted pseudo-label方法。

不建议按当前 README进入正式实现/训练。修复需要重新定义 observation states与evidence constraints，不是补一个 control或改一个阈值即可完成。
