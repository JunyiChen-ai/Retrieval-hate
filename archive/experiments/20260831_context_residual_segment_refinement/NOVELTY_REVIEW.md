# Context-residual segment refinement：独立 novelty / source / identifiability review

截至 2026-08-31。审查对象为本目录 `README.md`。本轮只做联网查新、来源核验、数学与机制审查；未实现、未训练、未运行 prediction。

## 最终裁定

**STOP，novelty 4.2/10。**

三门结果：

| 硬门 | 结果 | 理由 |
|---|---|---|
| 允许跨任务 adaptation | PASS | ASM-Loc 是合规的跨任务来源，adaptation 本身不要求从零发明 |
| ASM-Loc / 相同核心未用于 hateful video | PASS | 未找到 ASM-Loc 被用于 hateful-video detection/localization 的 primary source；官方论文与代码只报告 THUMOS-14、ActivityNet-v1.3 |
| adaptation 必须 non-trivial、任务特定、可识别 | **FAIL** | action/context separation、completeness modeling、prototype residual 与 ASM-Loc refinement均已有直接先例；当前目标没有约束 `g`/`r` 的语义，存在关闭 residual、复制 topic shortcut和 score-scaling等退化解 |

这个候选有合理的任务动机，但动机没有被写进可识别的训练约束。当前实现定义最多是 **ASM-Loc + learned background-prototype subtraction + two-head MIL**。精确代码组合可能未见于 hateful video，不足以跨过“非组件拼接”硬门。

## 联网检索记录

实际使用的核心 query如下，检索日为 2026-08-31：

1. `ASM-Loc "hateful video" localization`
2. `"Action-Aware Segment Modeling" hate video`
3. `site:openaccess.thecvf.com ASM-Loc Action-Aware Segment Modeling CVPR 2022`
4. `hateful video localization topic context residual prototype subtraction temporal`
5. `weakly supervised temporal localization context residual background prototype subtraction`
6. `weakly supervised temporal action localization foreground background prototype residualization`
7. `hateful video "global" "local" segment contrast context 2026`
8. `SafeLens segment-level hate speech detection online videos primary paper`
9. `"Completeness Modeling and Context Separation" official paper`

Primary evidence：

- He et al., [ASM-Loc, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_ASM-Loc_Action-Aware_Segment_Modeling_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2022_paper.html) 与[官方仓库](https://github.com/boheumd/ASM-Loc)只列 THUMOS-14 和 ActivityNet-v1.3。其核心是 dynamic segment sampling、intra/inter-segment attention、pseudo instance supervision与multi-step proposal refinement。
- 上述 ASM-Loc + hate queries没有返回把 ASM-Loc用于 hateful video 的论文。目标领域检索到的是 [MultiHateLoc](https://arxiv.org/abs/2512.10408)、[LELA](https://arxiv.org/abs/2602.09637)、[CLARA](https://arxiv.org/abs/2608.15905)、[SafeLens](https://ojs.aaai.org/index.php/AAAI/article/view/42390) 等不同机制。
- 未检出直接应用不能证明整个机制新颖。反而，扩大到 action/context separation 后出现两个比 README 当前 related-work 更直接的先例：Liu et al., [Completeness Modeling and Context Separation, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Completeness_Modeling_and_Context_Separation_for_Weakly_Supervised_Temporal_Action_CVPR_2019_paper.html)，以及 Min and Corso, [Learning Explicit Subspaces for Action and Context, AAAI 2021](https://cdn.aaai.org/ojs/16323/16323-13-19817-1-2-20210518.pdf)。

因此第二门可以写成“未发现 ASM-Loc 已进入目标任务”，不能写成“未发现相同上位机制”。

## 最近 primary work 逐项比较

### ASM-Loc：来源合法，但 adaptation 不是自动 novelty

[ASM-Loc](https://openaccess.thecvf.com/content/CVPR2022/papers/He_ASM-Loc_Action-Aware_Segment_Modeling_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2022_paper.pdf) 已占据：

- 用 video labels训练的 segment-centric WTAL；
- short-segment dynamic sampling；
- segment内/间attention；
- 从当前模型proposal生成 pseudo instance supervision；
- 多轮 proposal refinement。

README 正确识别出其 action-aware background loss不适合直接迁移：ASM-Loc明确让 background-attended logits也预测 action category，因为台球桌等 action context有类别信息。去掉这一 loss 是必要的 domain correction，但 **删除一个不合适的 source loss不是独立机制贡献**。

### CVPR 2019：context separation 与 completeness 已经被联合提出

[Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Completeness_Modeling_and_Context_Separation_for_Weakly_Supervised_Temporal_Action_CVPR_2019_paper.pdf) 已在同一 WTAL 方法中同时解决：

- MIL只抓最显著部分导致的不完整 localization；
- 与目标类别共现的 context 被错误定位；
- 用额外 branch / hard-negative context supervision扩大完整 action并排除context。

这直接占据候选的上位结构：“先区分context，再做完整segment”。把 action换成 local hostile event，把 action context换成 video topic，是合理领域映射，不足以单独形成新机制。

### AAAI 2021：两个语义变量、context只辅助训练、localization只读事件子空间均已有

[Weakly Supervised Temporal Action Localization Through Learning Explicit Subspaces for Action and Context](https://cdn.aaai.org/ojs/16323/16323-13-19817-1-2-20210518.pdf) 是最致命先例。它明确指出 video classification会依赖贯穿视频的类别context，学习 action/context 两个feature subspaces，并在 test localization时只使用 action subspace。论文还加入 temporal residual module。

本候选的 `g_t` / `r_t`、topic branch不写回localization、只对 residual localizer做时序细化，和该先例在监督语义与信息流上高度同构。差异是用“低分位背景prototype的 gated subtraction”代替显式subspace/triplet训练，再把 ASM-Loc接到 residual后面。这个差异目前更像实现替换与组件串接。

### Prototype residual不是空白

Zhang et al., [Prototypical Residual Networks for Anomaly Detection and Localization, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Prototypical_Residual_Networks_for_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf) 已从 normal prototypes构造 feature-prototype residual用于 anomaly localization。它不是视频MIL，也没有 hate topic语义，因此不直接占用目标任务；但它占据“prototype subtraction产生local anomaly residual”这个组件。候选不能把 prototype residual本身列为贡献。

### 目标领域已有 global/local 与 context-sensitive segment modeling

- [MultiHateLoc](https://arxiv.org/abs/2512.10408) 已占据 video-label-only multimodal temporal hate localization、modality-aware temporal encoder、dynamic fusion和MIL。
- [CLARA](https://arxiv.org/abs/2608.15905) 已在 hateful video detection中使用fine-grained clips、local-global segment contrast，并明确以跨clip contextual buildup解释局部中性、全局可恨的现象。它不做 residual-only ASM refinement，也不输出本项目的frame posterior，但占据 broad “local/global context modeling for hateful video” claim。
- [LELA](https://arxiv.org/abs/2602.09637) 已把 video context与逐帧多模态hate score结合用于training-free localization；机制不同，但再次压缩“global context + local hate”的宽叙事。
- [SafeLens](https://ojs.aaai.org/index.php/AAAI/article/view/42390) 是segment-level multimodal moderation demo，不使用video-label-only ASM refinement；它只说明segment-level hate本身不是贡献。

没有一篇目标领域工作与当前精确公式相同；STOP来自机制不可识别和强组件先例，不是“ASM-Loc 已经被目标论文直接使用”。

## 数学与可识别性 blocker

### 1. `g_t` 与 `r_t` 没有语义可识别性

README称 `g_t` 是topic/context、`r_t` 是local hostile-event residual，但两头最终都只接受同一个video label监督。没有 topic label、hostility predicate、orthogonality、adversarial invariance、互信息约束或可验证干预来区分两个潜变量。因此以下解对训练目标同样合法：

- `g` 与 `r` 都编码video topic；
- `g` 编码局部hate，`r` 编码topic；
- 任一分支承担全部video classification，另一分支失效；
- 两分支按任意可逆变换交换表示。

给变量起语义名称不产生disentanglement。AAAI 2021至少用stream agreement分组与triplet/subspace losses约束 action/context；当前候选比直接先例的语义约束更弱。

### 2. gated subtraction 有多个零作用退化解

`r_m(t)=P_m x_m(t)-gate_m(t)P_m b_m` 可以通过以下方式退回普通特征：

- `gate_m(t)→0`；
- learned projection令 `P_m b_m→0`，同时保留分类方向；
- downstream bias重新加回被减去的近常数分量；
- local encoder从未被prototype消除的topic方向重建video identity/topic。

若 gate 只由 similarity决定，在线性 local classifier下第二项主要表现为一个随similarity变化的score offset。除非证明 residual在保持local evidence的同时去除了topic信息，这与learned centering / attention calibration无法区分。

### 3. stop-gradient不解决语义循环

背景prototype由当前local head的低分位秒决定；local head又在这些prototype产生的residual上训练。stop-gradient只阻断单次forward中穿过prototype数值的梯度，不能阻断：

- 下一轮分位选择随模型改变；
- seed model的topic shortcut决定哪些秒被视为background；
- 正视频内漏检的hateful秒进入prototype；
- ASR fragment复制产生的大块常数score使低分位选择近似任意。

这仍是self-training feedback loop，不是独立的background evidence。

### 4. global × local 对 within-video ranking严格不起作用

对固定视频 `V`，`p(video positive|V)` 是所有秒共享的正常数。因此

`rank_t p(y_t=1|V) = rank_t p(local_t=1|V,V+)`。

video multiplier不能改善within-video排序。它只能改变不同视频间的尺度，从而改善或损害 pooled AP/ROC。若两头只通过最终乘积训练，还存在分解不唯一：一头放大、另一头缩小可以保持相同frame posterior。若两头各自另有video BCE，它们仍可学习同一topic shortcut。

所以该factorization是一个概率恒等式，不是topic/event separation机制。任何只来自multiplier的gain都应称为cross-video calibration；README对此已有部分警觉，但当前目标没有防止训练收益全部走这条路径。

### 5. residual-only ASM refinement仍会自我确认

pseudo segments来自 residual localizer自己的train predictions。若 residual head已学习topic或whole-video shortcut，ASM-Loc的segment attention和multi-step refinement只会把错误proposal结构化并重复写回。ASM-Loc原论文也明确展示：base proposal严重错位时会漏掉真实action。把refinement限制在 `r_t` 上，只有在 `r_t` 已被独立证明为topic-free时才有意义；当前恰好没有这项保证。

### 6. missing-evidence mask是独立工程修正

不把缺失ASR的零向量当作negative text evidence是正确的，但它解决physical missingness，不解决topic/event separation。它与 prototype residual、ASM refinement没有共同约束，不能用来提高核心机制novelty。

## 与项目已有负结果的关系

已有developmental test evidence没有直接证明本公式一定失败，但显著提高了其举证门槛：

- [target-conditioned normal proposal MIL](../../archive/experiments/20260831_target_conditioned_normal_proposal_mil/README.md) 中，conditional topic residual相对unconditional在HateMM/HCS的within均下降，HCS topic support也很低。当前候选改用同视频低分位prototype，避开跨视频support问题，但“topic conditioning自然改善local hate排序”的前提已有反证。
- [POWA span marginal pilot](../../archive/experiments/20260831_powa_span_marginal_pilot/README.md) 的连续span机制未胜matched controls，HateMM shuffled-span反而更强。不能从“hate应是segment”直接推出ASM-style连续refinement有效。
- [multimodal P-MIL baseline](../../archive/experiments/20260831_multimodal_pmil_baseline/README.md) 的proposal oracle有空间，但learned scoring在HCS严重失败并偏向whole-video proposal。它证明的是候选集合存在，不证明self-generated pseudo segment可被正确refine。
- [semantic-neighbor probe](../../archive/experiments/20260831_semantic_neighbor_probe/README.md) 只支持semantic recurrence+persistence作为calibration upper bound，不能为本训练机制背书。

候选当前没有提供一个新信息源来打破这些循环：仍是同一video label、同一模型低分位和同一模型pseudo proposals。

## README controls为何不足

`no-residual ASM-Loc`、`no-stop-gradient`、`unconditional subtraction` 可以判断若干部件是否影响性能，但不能排除最关键替代解释：

1. 多一个local head / projection带来的容量提升；
2. 任意per-video score centering都能达到同样结果；
3. backgroundprototype身份不重要，随机或全视频均值也一样；
4. ASM refinement单独带来全部提升；
5. global multiplier只做pooled calibration；
6. residual表示仍可线性预测video topic。

若未来提出一个具有新识别约束的版本，最低归因矩阵必须包含：faithful ASM-Loc port（去掉action-aware background loss）、two-head ASM但无residual、prototype residual但无ASM、scalar score-centering + ASM、random/time-shuffled prototype、全视频均值prototype、只训练projection的capacity control，以及global multiplier固定为1。还必须直接测 residual上的video/topic可预测性与local GT切换，而不能只看最终三指标。

这些是未来重审要求，不是对当前版本的实现批准。

## 是否只是 calibration

结论分两部分：

- `p(video positive|V)` 乘法对within-video排序 **严格只是无效常数**，对pooled指标是video-level scale calibration。
- gated feature subtraction若进入非线性temporal encoder，原则上能改变within ranking，不必然只是calibration；但当前无约束版本可以退化成相似度相关的score offset，且没有证明它去除topic而保留hostility。

因此不能笼统说整个方法必然是post-hoc calibration；可以确定的是 global multiplier没有定位机制价值，而 residual branch目前没有可识别证据超越learned centering。

## 最窄可能描述与为什么仍不批准

当前最多可以如实描述为：

> A target-task port of ASM-Loc in which segment refinement is applied to modality features centered by a train-time, low-score, per-video prototype, while a separate video head rescales the final frame scores.

这句话是实现描述，不是足够强的method claim。它不能声称topic被分离、hostile-event residual被识别、global/local probability被解耦，或首次context-aware segment refinement。

## 最终意见

**STOP，4.2/10；无批准实现边界。**

第二门通过：ASM-Loc没有被查到直接用于 hateful video。第三门失败才是淘汰原因：CVPR 2019已经联合提出context separation与completeness，AAAI 2021已经实现action/context双空间且只用action空间定位，prototype residual与ASM refinement也都有直接来源；本候选又缺少迫使 `g`/`r` 获得所声明语义的训练约束。继续实现只会测试一个不可归因的组件组合。

若要重新提交，必须先增加一个由任务语义产生、能排除上述退化解的监督或干预约束，并说明它提供了video label与self-generated proposal之外的什么新信息；仅增加orthogonality loss、更多ablation或更换prototype定义不构成足够的新核心。
