# Interval Witness Transport：独立 novelty review

截至 2026-08-31。依据仓库 commit `13257e004fc1d306e2dfbadb4e93317062513f83`、本目录
`README.md`、`runs/20260831_powa_witness_path_probe/analysis.json`，以及下列论文与官方项目页面。
本文只评审候选核心，不批准训练，也不把 dense primitive VLM supervision、Sinkhorn、POWA
既有 PEF/AWB、span pooling、knowledge distillation 或 score transport 计作 novelty。

## Verdict

**Conditional GO，仅允许最小可证伪 pilot；novelty 5.4/10。**

截至检索日，没有找到与以下完整机制相同的工作：在 video-level weak hateful-video
localization 中，把 policy-typed hostile-predicate timestamp 与 protected-target timestamp
之间的异步 OT witness edge distribution，经固定 interval-incidence operator 转成 dense
interval occupancy，并在推理时不使用 teacher。

但这不是一个宽 novelty。核心算子与三类成熟做法高度相邻：start/end pair 的二维 proposal
map、relation-to-temporal-span prediction、弱监督 anchor/snippet temporal propagation；此外，
OT-to-dense temporal segmentation 已有明确先例。若去掉 “policy-typed asynchronous
predicate–target witness” 这一语义限定，剩余的 “把一对时间索引覆盖到二者之间的闭区间”
是初等的固定 incidence transform，单独不足以形成强方法贡献。

**Novelty 与 performance 分开裁定。** Novelty 上可 conditional GO；目前的有效性证据则未过关：

- HateMM：POWA within `.57193`，path `.62866`，但 fixed center-first `.76550`，后者明显更高；
  当前 path gain 完全可能只是 edge distance / timestamp position 引出的中心先验。
- HateClipSeg：POWA `.52707`，path `.49878`，方向相反。
- 因此这不是“已有双语料可行性上的方法增量”，而只是值得用严格 matched-prior control
  杀一次的机制假设。若未胜过 matched center/edge controls，立即 STOP，不应以 teacher
  补强后继续包装。

## 被评审的唯一数学核心

设 AWB transport plan 为 `P[i,j]`，其中 `i` 是 hostile-predicate timestamp，`j` 是
protected-target timestamp。定义固定 interval-incidence tensor：

```text
R[t,i,j] = 1[min(i,j) <= t <= max(i,j)]
o[t]     = sum_{i,j} P[i,j] R[t,i,j]
```

最窄的候选是将 `o[t]` 作为 policy-typed dense witness evidence。它不是学习边界，不是从
关系显式预测 span，也不是在真实图上推理一条 path；它把两个语义 witness timestamp 的
一维 convex hull 当作占用区间。

这一定义有四条必须公开的边界：

1. **未归一化版本不守恒。** 令 `L[i,j] = |i-j| + 1`，则
   `sum_t o[t] = sum_{i,j} P[i,j] L[i,j]`。长 edge 的总影响被复制 `L` 次，不能称为
   “mass transport” 或 “mass preserving”。长度归一化版本应定义为
   `o_norm[t] = sum P[i,j] R[t,i,j] / L[i,j]`，并作为必做 control。
2. **区间算子本身无方向。** `R[t,i,j] = R[t,j,i]`。对固定 plan 仅做转置，rasterized
   occupancy 不变。因此所谓 direction-shuffle control 必须交换 predicate/target primitive
   roles 并重新计算 AWB；对 `P` 做 post-hoc transpose 是无效 control。
3. **中心位置有组合偏置。** 在近似均匀或无信息的 endpoint pairs 下，跨过中心 timestamp
   的区间数最多；未归一化版本还叠加长边偏置。HateMM center-first `.76550 > .62866`
   已把这一风险从理论可能变成当前首要混杂因素。
4. **语义端点不是事件边界。** hostile cue 在 `i`、target cue 在 `j` 并不推出二者之间
   每一秒都 hateful。convex-hull fill 可能跨越 benign islands；这正是 within-video ROC
   会惩罚而 pooled 指标可能掩盖的错误。

## 最接近的工作

| 工作 | 已占据的机制 | 与本候选的实质差异 | 风险等级 |
|---|---|---|---|
| [BMN, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_BMN_Boundary-Matching_Network_for_Temporal_Action_Proposal_Generation_ICCV_2019_paper.html) | 以 start/end boundary pair 表示 proposal，构造 dense 2D boundary-matching confidence map，并在区间内采样聚合特征 | 两端是被监督/预测的 action boundaries；输出是 proposal confidence，不是 policy relation edge 的 frame occupancy | **最高 operator-level collision** |
| [2D-TAN, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6984) | 用二维 start-time × end-time map 枚举、表示和关联不同长度的 moments | query-grounded moment map；没有 weak MIL、typed predicate–target witness 或 OT edge rasterization | 高 |
| [TimePLE, arXiv 2026](https://arxiv.org/abs/2607.23951) | 不再独立预测 endpoints，而直接预测所有 valid intervals 上的 joint distribution | interval-native VTG，依赖 grounded samples/interval annotations；没有把异步 semantic edge mass 转成 dense occupancy | 高；限制宽泛的“interval-native evidence” claim |
| [TSPN, arXiv 2021 / ESWA](https://arxiv.org/abs/2107.07154) | 为 object-pair relationness 与 relation category 联合预测 temporal start/end span | 从完整视频上下文学习 relation span；本候选不是 span predictor，只取两个 evidence timestamps 的 convex hull | **最高 semantic-level collision** |
| [CLASP, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38374) | 在 weakly supervised dense audio-visual localization 中找 cross-modal salient timestamps，并通过 Anchor-based Temporal Propagation 增强全时间特征 | 已占据 weak anchor-to-dense propagation；但不是 pair-edge incidence，也不表达 hostile–target policy relation | 高 |
| [RSKP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Weakly-Supervised_Temporal_Action_Localization_via_Representative_Snippet_Knowledge_Propagation_CVPR_2022_paper.html) | 从代表 snippets 向 intra-/inter-video snippets 传播知识并产生 pseudo labels | 通用 WTAL snippet propagation；没有两个 typed endpoints、interval fill 或 OT plan | 中高 |
| [Temporally Consistent Unbalanced OT, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Temporally_Consistent_Unbalanced_Optimal_Transport_for_Unsupervised_Action_Segmentation_CVPR_2024_paper.html) | 从 frame–action affinity/cost 经 temporal OT 解码 dense action segmentation，并用于 self-training pseudo-labels | OT 的两轴是 frame × action class，temporal consistency 在 OT objective 内；不是 timestamp × timestamp relation edge 的区间 rasterization | **最高 OT-to-dense collision** |
| [CLOT, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Bueno-Benito_CLOT_Closed_Loop_Optimal_Transport_for_Unsupervised_Action_Segmentation_ICCV_2025_paper.html) | 多个 OT problem 联合 frame/segment embeddings 与 dense pseudo-label refinement | action-segmentation assignment，不是 policy witness edge；进一步说明“OT 产生 segment/dense labels”不是空白 | 中高 |
| [P-GCN, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Zeng_Graph_Convolutional_Networks_for_Temporal_Action_Localization_ICCV_2019_paper.html) | proposal graph 的边用于 node message passing、分类和 boundary regression | graph edge 连接 proposal nodes，并未把边渲染为两 timestamp 间的每一帧；反证本候选不应借用“graph path attribution”术语 | 中 |
| [MultiHateLoc, arXiv 2025](https://arxiv.org/abs/2512.10408) | video-level weak supervision 下的 multimodal temporal encoders、dynamic fusion、contrastive alignment 与 modality-aware MIL，产生 frame scores | 领域最接近；虽强调跨模态 cue 异步，但没有 policy-typed relation edges 或 interval occupancy | 领域邻近 |
| [LELA, arXiv 2026](https://arxiv.org/abs/2602.09637) | training-free multimodal captions/LLM 推理并产生细粒度 frame hateful scores | 直接逐时刻 scoring，无 edge/interval mechanism | 领域邻近 |
| [TANDEM, arXiv 2026](https://arxiv.org/abs/2601.11178) | 输出带 timestamp intervals 与 target identities 的 structured hate result | 直接生成/学习 temporal grounding 与 target；无 AWB relation plan 或 edge-to-occupancy operator | 领域与 policy-output 邻近 |

## 按文献族的 novelty 判断

### WTAL / weak dense localization

snippet/anchor mining、传播、pseudo-label refinement 和显式 segment modeling 已很拥挤。RSKP、
CLASP 等已足以否定“从稀疏局部证据传播出 dense temporal evidence”这一宽 claim。本候选的差异
只能落在：传播的输入不是单 anchor，而是带 policy role 的 predicate–target pair distribution；
传播支撑集被固定为两端点的一维闭区间。

### Boundary / interval representation

BMN、2D-TAN、TimePLE 已分别覆盖 boundary pair map、2D interval map、interval-native joint
distribution。故不能 claim “首次联合建模起止时刻”“首次把 pair 表示为 interval”或“首次
differentiable interval evidence”。可区分点是候选两端不是 start/end boundary，而是异步语义
witness；但这同时也是科学风险，因为语义 witness 并不保证是区间边界。

### Relation localization 与 graph/path attribution

TSPN 已表明 object relation 可以直接联合预测 temporal span。候选没有学习 relation 的 start/end，
只是把两个 evidence timestamps 的 convex hull 视为 span。时间轴上 `i` 与 `j` 之间的 indices
不是由关系 edge 观测到的图路径，所以 “edge-to-path attribution” 会夸大机制。正式名称应使用
**interval-incidence rasterization / convex-hull occupancy**，不使用 graph path、causal chain、
relation duration 等措辞。

### Optimal transport temporal alignment / segmentation

ASOT、CLOT 以及相关 temporal OT segmentation 已占据 “用 OT 获得 temporally consistent dense
assignment/segmentation”。本候选不能把 Sinkhorn 或 OT-to-dense 当贡献。唯一剩余差异是 OT plan
的两轴都是视频 timestamps，且二者承担不对称的 policy primitive roles；dense support 由固定
interval incidence 从 relation edge 构造，而不是 OT 本身解码。

### Hateful-video localization

MultiHateLoc、LELA、TANDEM 尚未出现上述完整 readout。领域空白有助于组合的新颖性，但“首次在
hateful video 使用已有 interval operator”本身不够；必须证明 policy typing 和 asynchronous pair
structure 对定位有独立价值，而非把通用中心/长度 prior 换了应用域。

## 可 defend 的最窄 claim

若且仅若必做 controls 通过，可主张：

> 在仅有 video-level hate labels 的 temporal localization 中，使用固定、可微的
> interval-incidence operator，将 policy-typed asynchronous predicate–target edge
> distribution 转换为 dense interval-valued witness evidence；实验证明收益来自 typed edge
> structure，而非 endpoint、teacher、区间长度或 timestamp-position prior。

不能主张：

- 首次 edge-to-span、pair-to-interval、interval-native localization 或 temporal propagation；
- 首次 OT temporal localization / segmentation；
- learned boundary、relation duration、graph path attribution 或 causal interval；
- 未归一化版本 mass preserving；
- dense VLM supervision、KD、POWA、Sinkhorn、span pooling、score transport 是本文 novelty；
- 在 matched center/edge controls 未通过时仍称 interval witness 有效。

## 最小可证伪 pilot 与必要 controls

### 1. 四臂 factorial，隔离真正增量

在完全相同 POWA anchor、训练预算、evaluator、checkpoint rule 下运行：

1. 原 POWA endpoint readout；
2. dense primitive supervision + endpoint readout；
3. interval occupancy + 原 sparse supervision；
4. dense primitive supervision + interval occupancy（full）。

Full 必须在 HateMM 与 HateClipSeg 都显著胜过 2、3 两个单因素 arms；否则结果属于 teacher
或 rasterizer 的普通独立增益，不能支持完整机制。teacher protocol 不得按语料切换。

### 2. 位置、距离和边缘先验：硬性 gate

- fixed center-first 与 fixed edge-first；二者的 score multiset、占用预算和 selection fraction
  必须与候选严格 matched；README 已记录 center-first `.76550`，它是必须击败的 baseline。
- duration-matched deterministic prior：仅使用 AWB edge-length histogram，不使用 primitive/edge
  semantics。
- marginal-preserving null plan：在 lag/distance bins 内置换 edge weights，尽量保持 row/column
  marginals、edge-length distribution 与 occupancy budget。普通 edge-time shuffle 会同时改变距离和
  位置分布，不足以排除混杂。
- 报告 occupancy 与 normalized time position、edge length 的相关性，并按 interval length 分层
  报告 within-video 结果。

**否决线：** 若 typed plan 不显著优于 matched center-first、edge-first 和 marginal-preserving
null，机制 STOP，即使 absolute within 数字上升也不得归因于 witness transport。

### 3. 算子 controls

- endpoint-only；
- same-time / diagonal plan；
- 未归一化 `R` 对比 length-normalized `R/L`；
- role swap 后重新计算 PEF/AWB plan；不能用固定 `P` 的 transpose 充当 direction control；
- predicate-role permutation、target-role permutation 和 edge-weight permutation；
- `sum` 与一个预注册的饱和 aggregator（如 noisy-OR/hazard），但不得看 test 后挑选；
- 诊断 overlapping edges 的 multiplicity，避免中心位置仅因重复覆盖获得高分。

### 4. 语义有效性 controls

- 用冻结 GT 只做最终诊断：统计 endpoint 之间含 benign islands 的 interval，比较这些秒的假阳性；
- 分别报告 same-modal / cross-modal、短 lag / 长 lag、hostile-before-target / target-before-hostile；
- edge direction 的验证来自 role-aware plan recomputation，而非对称 rasterizer；
- 检查收益是否只来自极少数长边或整段正例广播。

### 5. 晋级标准

沿用 README 的预注册门槛：HateMM、HateClipSeg 都需 within `+.020`，pooled ROC/AP 满足 feasibility，
并且 full 显著优于两个单因素 arms与所有 matched priors，才进入 test。任何按语料选择 endpoint/path、
normalization、kernel 或 teacher 的做法均为 STOP。

## 致命 anti-pattern

以下任一项出现即否决该 core，而不是继续增加模块：

1. **中心先验伪增益：** path 不胜 matched center-first；尤其当前 `.76550 > .62866` 已是直接警报。
2. **长度复制伪增益：** 只报未归一化 occupancy，把长边重复铺开的 gain 称为 transported mass。
3. **teacher 掩盖：** full 的增益不能超过 teacher + endpoint，或 HCS 只有 teacher arm 上升。
4. **语料特判：** HMM 用 interval、HCS 用 endpoint，或使用不同 teacher / normalization / lag policy。
5. **无效方向消融：** 对固定 `P` 转置后因 `R` 对称得到相同结果，却据此宣称 direction robustness。
6. **把 cue 当 boundary：** 叙述中把 hostile/target evidence timestamps 称为 event start/end，或默认
   convex hull 全段为 hate，而没有 benign-island error analysis。
7. **graph/causal 过度叙述：** 把时间轴填充说成图路径、因果链或关系持续时间。
8. **null plan 未匹配：** random shuffle 改变 edge length、marginals、position histogram 或 score budget，
   导致 typed plan 与 null 不可比。
9. **只看 pooled：** pooled ROC/AP 上升但 within-video 不升；这不能支持 dense localization。
10. **test 后调算子：** 根据 val/test 选择 interval kernel、clip、saturation、length normalization、lag
    cutoff 或 per-corpus branch。
11. **重复计 novelty：** 把原 POWA PEF/AWB、Sinkhorn、VLM/KD 或 score transport 并入本轮 claim。
12. **应用域替换：** 仅证明一个已知 pair-to-interval operator 能在 hate 数据上运行，却未证明
    policy-typed asynchronous edge 比 matched positional prior 提供额外信息。

## 查新范围与置信度

检索截至 2026-08-31，优先使用 CVF、AAAI、ECCV、PMLR、arXiv 原论文/官方项目页面。检索族包括：
weakly supervised temporal action/event localization、snippet/anchor propagation、boundary matching、
2D temporal maps、relation temporal span prediction、optimal-transport action segmentation/alignment、
proposal graphs/path attribution、asynchronous audio-visual event localization、multimodal hateful-video
temporal grounding。

对“未找到完整等价机制”的置信度为**中等**，不是法律意义的穷尽式专利检索。对“宽泛的 pair-to-
interval、anchor propagation、relation-to-span 与 OT-to-dense claims 已被占位”的置信度为**高**。
这支持一次严格的 kill pilot，不支持直接把候选写成 publication-ready novelty。
