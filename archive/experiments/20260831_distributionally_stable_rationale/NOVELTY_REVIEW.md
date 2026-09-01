# Independent novelty and mechanism review

截至 2026-08-31。审查对象为本目录 `README.md` 中的
distributionally stable sufficient-and-necessary temporal rationale 候选。只做文献、数学与项目内证据审查；
未实现、未训练、未生成 prediction。

## Verdict

**STOP。Novelty：4.8/10。**

三项硬门分别为：

1. **允许跨任务 adaptation：PASS。** SIS、selective rationalization 与 perturbation explanation 可以作为来源。
2. **来源核心未进入 hateful-video detection/localization：窄义 PASS。** 本次检索未发现已有 hateful-video
   方法求“跨多个 benign replacements 同时 sufficient 且 necessary 的最小共享 temporal mask”。已有 hateful-video
   rationale 工作生成或融合自然语言理由、显式线索或 clip 表示，不是该约束优化器。
3. **adaptation 必须 non-trivial、任务机制成立且可证伪：FAIL。** 多 replacement 是非零改动，但候选仍只识别
   frozen video classifier 的最小稳定决策子集，不识别 hateful span。它有解析的单帧 shortcut、topic shortcut、
   replacement artifact、不可行/全视频解与非唯一解。随后 student 只是把 teacher 产生的 mask 当伪标签蒸馏，
   不再执行 sufficiency/necessity 约束。项目 V26 已直接表明“强 video discrimination + exact temporal replacement
   effect”不足以给正确的 video 内排序；本候选没有引入独立于同一个 video-label classifier 的方向性观测。

该结论不是“已有人做过完全相同组合所以停止”，而是：**剩余可主张的组合差异不提供所需的可识别性**。

## 1. 实际检索范围与占位结论

使用论文主页、会议/期刊官方页面、ACL Anthology、PMLR、CVF Open Access、OpenReview 与作者公开预印本。
核心 query 包括：

- `sufficient input subsets minimal subset classifier`；
- `selective rationalization sufficiency comprehensiveness degeneration`；
- `meaningful perturbation extremal perturbation preservation deletion mask`；
- `video explanation temporal mask perturbation preservation deletion`；
- `robust rationale multiple replacements baseline`；
- `hateful video rationale explanation temporal localization`；
- `MultiHateLoc CLARA MATCH explainable hateful video detection localization`。

公开索引的负检索不能证明绝对不存在先例；这里的结论是，在截至日期可检索的一手材料中没有发现 exact hateful-task
占位，而最相邻的跨任务算子已有明确先例。

### 1.1 直接来源与最邻近先例

| 工作 | 已有核心 | 与候选的真实差异 | 裁定 |
|---|---|---|---|
| [Carter et al., *Sufficient Input Subsets*, AISTATS 2019](https://proceedings.mlr.press/v89/carter19a.html) | 对 black-box 决策寻找仍超过阈值的 minimal-cardinality observed feature subset；明确是“从模型视角”的充分解释。 | 候选把缺失值换成同语料 negative-video references，并增加 removal necessity 与跨 replacement 最坏约束。 | minimal sufficient subset 不能 claim；只能 claim robust temporal adaptation。 |
| [Lei et al., *Rationalizing Neural Predictions*, EMNLP 2016](https://aclanthology.org/D16-1011/) | 用短、连贯、单独足以预测的 latent extractive rationale 训练 predictor。 | 候选冻结 OOF predictor、采用 post-hoc constrained selector，且要求 complement 变 normal。 | sparse sufficient rationale 已占位。 |
| [Bastings et al., *Interpretable Neural Predictions with Differentiable Binary Variables*, ACL 2019](https://aclanthology.org/P19-1284/) | HardKuma binary/continuous latent mask与可微稀疏选择。 | 候选采用 lexicographic feasibility，而非固定 selection-rate penalty。 | soft/binary mask与稀疏优化不新。 |
| [Yu et al., *Rethinking Cooperative Rationalization: Introspective Extraction and Complement Control*, EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1420/) | 明确定义 sufficiency、comprehensiveness、compactness，并用 rationale complement control；论文也给出只选择首/尾 token 以编码标签的 degeneration。 | 候选的 remove constraint 用同一个 frozen classifier和多 replacements，不是另训 complement adversary。 | “keep 足够 + complement 不足 + compact”主体已占位；而且该论文直接预示 mask/position shortcut。 |
| [DeYoung et al., *ERASER*, ACL 2020](https://aclanthology.org/2020.acl-main.408/) | 将 sufficiency 与 comprehensiveness 作为 faithfulness 度量，并与 human rationale alignment 分开。 | 候选把两项从评测量变成硬约束。 | 满足 faithfulness 只说明忠实于 classifier，不等于和人工 span 对齐。 |
| [Fong & Vedaldi, *Meaningful Perturbation*, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Fong_Interpretable_Explanations_of_ICCV_2017_paper.html) | 以连续 mask 和 reference input 做 preservation/deletion 式 black-box explanation；原文明确警告 network artifact 会吸引 explanation。 | 候选的 reference 是多个 negative videos，且 mask 只有时间维并共享三模态。 | replacement mask算子与 artifact 风险已有直接先例。 |
| [Fong, Patrick & Vedaldi, *Extremal Perturbations*, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Fong_Understanding_Deep_Networks_via_Extremal_Perturbations_and_Smooth_Masks_ICCV_2019_paper.html) | 在固定 area/最小区域意义下优化 preservation response，避免任意 loss 权重。 | 候选以 margin feasibility 后 lexicographic 最小质量替代 area sweep。 | “约束优先、再最小 mask”的结构非常接近，不足以把整个 optimizer claim 为新。 |
| [Li et al., *Towards Visually Explaining Video Understanding Networks With Perturbation*, WACV 2021](https://openaccess.thecvf.com/content/WACV2021/html/Li_Towards_Visually_Explaining_Video_Understanding_Networks_With_Perturbation_WACV_2021_paper.html) | STEP/EP-3D 已把 extremal perturbation 扩展到视频时空 mask，联合限制 preservation ratio 与 temporal smoothness。 | 候选是 1D temporal mask、三模态 feature replacement、同时做 keep/remove。 | video temporal perturbation explanation 已占位；多 negative references与双约束是窄差异。 |
| [Yoon et al., *INVASE*, ICLR 2019](https://iclr.cc/virtual/2019/poster/1022) | actor-critic instance-wise variable selection，选择数量可按样本变化。 | 候选不联合训练 selector/predictor，采用显式 feasibility。 | instance-wise variable-size selection 不新。 |

补充地，[Zhou et al., *Interventional Rationalization*, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.700/)
已以干预来处理 selector/predictor 依赖；[Hase et al., ACL 2023](https://aclanthology.org/2023.acl-long.707/)
则系统讨论 generator 利用 predictor bias 造成 rationale degeneration。它们不构成 hateful-video 直接占位，但否定了
“只要 OOF/frozen 并做 input replacement，rationale 就自然可信”的推断。

### 1.2 Hateful-video 领域是否已有该核心

- [MultiHateLoc](https://arxiv.org/abs/2512.10408) 用 modality-aware encoders、dynamic fusion、cross-modal
  contrastive alignment 与 MIL 从 video labels 输出 frame scores；没有 SIS 或 keep/remove rationale optimizer。
- [CLARA](https://arxiv.org/abs/2608.15905) 使用 VLM-derived video rationale、clip encoder 与 gated Transformer；
  其 rationale 是语义指导，不是最小 temporal subset，也没有跨 benign replacements 的 necessity test。
- [IARE / Ex-HateMM](https://arxiv.org/abs/2606.11953) 学习生成融合 harmful elements 的 contextual natural-language
  rationale，并使用细粒度 harmful-element/rationale annotations；监督与输出都不同。
- [MATCH](https://jianlang.org/papers/MATCH.html) 用多 agent 提议 hate/non-hate clues，再做 spatiotemporal
  evidence-grounded verification；不是 video-label-only mask selection。
- [LELA](https://arxiv.org/abs/2602.09637) 是 training-free prompt/caption composition localization；没有本候选约束。
- [SafeLens](https://ojs.aaai.org/index.php/AAAI/article/download/42390/46351) 输出有 timestamp 和理由的 harmful
  segments，但属于 MLLM/规则化 segment analysis，不是 post-hoc SIS。

因此第二门可以在很窄的意义上通过。但不能声称“首个 hateful-video rationale”：IARE、MATCH、CLARA、SafeLens
已经占用；也不能声称“首个 temporal perturbation rationale”：WACV 2021 STEP 已占用。

## 2. 与项目 V26 和 deletion-carrier 的精确边界

### V26 Counterfactual Temporal Witnesses

项目冻结记录 `docs/V20_V26_FINAL_ITERATION_ARCHIVE.md` 显示：V26 用 negative-only OOF decoder 给每个训练视频
产生 counterfactual background，以替换某一秒后对有限 receptive field 重算的**精确输出变化**作为 local score。
它的 video AP 为 `.888242`，within-video ROC 为 `.559938`；real、permuted 与 negative-mean replacement controls
也显示 learned negative reference 并非必要。项目结论是 replacement effect 没有对齐人工 span。

当前候选相对 V26 的增量是：

- 从逐秒 deletion effect 改成 whole-mask constrained search；
- 同时要求 kept sufficiency 与 removed necessity；
- 对多个 frozen benign replacements 取共同可行解；
- OOF teacher rationale 再监督 student。

这不是字面重复 V26，但没有引入新观测。两者都只问同一个 video-label classifier 在 synthetic replacement 下如何变化。
V26 的失败不能逻辑上证明所有集合干预必败，却直接否定了候选的关键跳步：**replacement faithfulness 并不推出
within-video hate-span alignment**。集合搜索能发现 feature interaction，但也能更稳定地发现 classifier shortcut。

### deletion-carrier-abstaining ItS2CLR

项目 `research-wiki/STATUS.md` 与归档实验记录显示，deletion-carrier core 相对 capacity-matched broadcast 的
within 增益只有 HMM `+.00313`、HCS `+.00105`；core/broadcast 的逐视频 frame-ranking Spearman 分别为
`.97568/.99723`。该失败说明 auxiliary attribution 即使被计算和迭代，也可能不进入最终 localizer ranking。

当前候选让 student 直接监督 rationale，形式上比 auxiliary contrastive loss 更强；但 student 阶段没有再 forward
teacher 的 keep/remove constraints。若 student 只拟合一个 hard/soft mask，它就是 OOF teacher pseudo-label
distillation。要归因于 rationale 机制，必须与同架构 student 拟合 teacher attention、raw frame score、V26 deletion、
same-mass top-k 等 controls 比较；仅“最终 head 读取 student posterior”不能消除伪标签解释。

## 3. 解析退化与不可识别性

### 3.1 最小质量目标系统性偏好 top-1

取一个合法的 frozen classifier：

\[
F(x)=\max_t h(x_t).
\]

若某个 topic/logo/slur frame `t*` 对每个 replacement 都满足：保留它使 `F` 超过 positive margin，替换它使
`F` 低于 normal margin，则 `m_{t*}=1`、其余为零同时满足两项约束，且是 lexicographic 全局最优。该反例
不依赖联合训练、数值误差或 replacement 数量。增加更多 benign replacements 只会增强这个 classifier witness
的稳定性，不会把它扩展为人工 hateful span。

README 将“单点退化”设为事后 kill condition，但目标本身奖励最短决策证据。也就是说，gate 可能发现预期退化，
却没有机制排除它。若加入最小持续时间/TV/边界先验来救，会与“无固定事件长度”的 claim 冲突，并把新意改成普通
temporal regularization。

### 3.2 OOF 不排除 topic shortcut

OOF 只排除 `F` 在生成某个 train rationale 时见过同一视频；它不排除 corpus-level shortcut。若正例中某片头、
speaker、channel logo、宗教/政治主题或固定 OCR pattern 与 label 相关，`F` 可以跨 fold 学到它。上述特征在多种
negative replacements 下恰好会非常稳定，因此 distributional constraint 可能**偏爱**而不是排除 topic shortcut。

在只有 video labels 的观察分布里，可构造两个数据生成过程：一个让该稳定帧是真实 hate witness，另一个让它只是
与 hate span 共现的 topic marker；二者对所有 train video labels 与候选 interventions 给出相同 `F` 响应，却有不同
frame GT。故目标 span 对当前观测不可识别。

### 3.3 Replacement artifact 与 off-manifold soft mask

`m*x+(1-m)*r` 是 feature-space 插值，不是观测到的真实视频：

- 不同视频的 speaker、scene、audio、OCR 与节奏在 mask 边界发生同步跳变；
- fractional `m_t` 生成训练分布外的 convex mixture；
- 三模态共用 mask 不等于三模态 replacement 在语义上同步；
- classifier 可对 source identity、尺度或边界不连续敏感。

Fong & Vedaldi 2017 已明确指出 network artifacts 是 explanation 的主要吸引子。对多个 replacement 取最坏约束
只能要求 artifact response 在所选有限集合上一致；不能证明 intervention on-manifold。若 mask pattern或边界本身被
`F` 感知，它还可能成为 label code，这与 Yu et al. 给出的首/尾 token degeneration 属于同一类问题。

### 3.4 可行性没有保证

令 `m=1`，removed input 就是 `r`。只要 frozen negative set 中存在一个被 `F` 判到 normal margin 以上的 false
positive，**任何 mask 都可能无法满足对所有 `r` 的 necessity**；negative video label 不保证 classifier margin 为负。
若预筛 `R(x)` 只保留 `F` 认为 normal 的 replacement，则 reference set 被 teacher 选择，结论变成对 teacher-friendly
背景的条件 faithfulness。

另外：

- distributed positive bias 可使只有 full-video mask 可行；
- 两段冗余 evidence 会迫使 necessity mask 包含两段，即使其中一段不是 GT；
- 不同 replacement 诱导互斥最小解释时，共同解是其 union、近全视频或不存在；
- positive prior/bias 即使在纯 replacement 上也高时，necessity 永远失败。

所以“不退化为全视频/不可行”不是优化器质量检查，而是当前模型族和 margin 下未必成立的额外数据假设。

### 3.5 Soft mask、fold margin 与非唯一解

- `m_t in [0,1]` 加 `L1` 最小化得到的是最小 mixture mass，不是 minimal subset。非线性 `F` 可通过许多小数
  membership 达标；阈值化后约束可能失效。
- 若改成 binary exact subset，组合搜索不可忽略；近似求解不能再声称 lexicographic minimum。
- “原 prediction 的固定 fraction”若作用于 logit，不对 logit 平移不变；若作用于 probability，饱和会使质量解
  高度依赖 fold calibration。不同 OOF folds 的 margin 不天然可比。
- 对称/冗余 frames 会产生多个等质量共同可行 mask。README 的“replacement disagreement tie-break”尚未定义：
  当前写法本来就对所有 `r` 求**一个共享 mask**，因此不存在 per-replacement mask disagreement；若先对每个 `r`
  分别求 mask 再聚合，则又是另一套问题。
- 单个共享解若是 binary，frame score ties 会很重；“minimal feasible membership/stability”没有给出唯一、可重算的
  数值定义。若它来自多次优化频率，随机性和 solver path 就成了未冻结的额外算法。

### 3.6 Student 是伪标签蒸馏，不保存约束

正式 test 输出 `student(x)_t`，不再求 keep/remove feasibility。因此 test score 没有构造上的 sufficiency、necessity
或 replacement stability 保证。student 可以：

- 平滑、广播或反转 teacher mask局部排序；
- 学到与 OOF masks 共现的 topic shortcut；
- 在 negative frames 数量占优时近常数；
- 只复现 mask的伪标签误差。

所以 student 不构成第二个新机制；在最窄 claim 中它只能被描述为 computational amortization / pseudo-label
distillation。若 selector premise 成功、student 失败，不能说正式方法成功；若 student 胜过 selector，则必须另证收益
不是普通 temporal student 的 inductive bias。

## 4. 为什么现有 premise controls 不足

README 的 control 列表方向正确，但仍缺以下决定性隔离：

1. **同一 classifier、同一 mask mass、不同 teacher target：** raw local logit、attention、V26 deletion、单 replacement
   SIS、suff-only、necessity-only、joint core分别蒸馏到完全相同 student。否则无法区分结构化 rationale与 dense pseudo-label。
2. **classifier shortcut controls：** temporal shuffle/reversal、片头/片尾移位、video-global feature广播、topic/channel/speaker
   matched strata。OOF 不能替代这些。
3. **replacement artifact controls：** same-label neutral splice、同一视频 temporal donor、跨模态 donor identity独立打乱、
   hard copy 对 soft interpolation、边界宽度匹配。必须比较 mask与 source boundary、modality discontinuity的关系。
4. **replacement-set intervention：** topic-matched negative、topic-mismatched negative、teacher-normal预筛 negative分别运行。
   README 的“replacement identity 均衡打乱”在 replacements 本来可交换时可能是无效 null control。
5. **可行性全量报告：** 每个视频报告 infeasible、only-full-feasible、single-frame、fractional-mass、非唯一解比例；
   不能只在可行子集上算指标。
6. **约束复验：** 对 hard-threshold mask、未参与优化的新 negative replacements 与独立 classifier重新 forward。
   只在优化用的有限 `R(x)` 和同一 `F` 上复验是 in-sample faithfulness。
7. **student mechanism retention：** teacher mask 与 student score的逐视频 rank agreement、student top region重新做
   keep/remove test、student vs broadcast capacity control。否则重演 deletion-carrier“机制存在但不进最终 ranking”。
8. **人工 span只用于 test evaluation/error analysis：** 对 single-point/full-video/fixed-boundary 与真实 GT 的关系可在固定
   test prediction 后分析，但不能把 GT 用入 mask optimization、teacher选择或 checkpoint selection。

即使补齐 controls，它们也只能经验上筛掉已知 shortcut，不能把“classifier-faithful rationale”升级为 causal hate span。

## 5. 最窄可辩护 claim

若仅讨论方法来源而不宣称成功，最窄表述是：

> A cross-task adaptation of sufficient-input and perturbation explanations to weakly supervised multimodal hateful video:
> an OOF frozen video classifier is queried for a shared temporal mask that satisfies both preservation and deletion margins
> under a fixed set of negative-train video replacements, and the resulting post-hoc masks are amortized into a frame student.

不能主张：

- 首个 hateful-video rationale、首个 temporal perturbation explanation或首个 sufficiency/comprehensiveness rationale；
- causal、counterfactual-ground-truth、identified hateful witness；
- distributional robustness（有限、经验 replacement set至多是 empirical worst-case robustness）；
- student 输出在 test 时仍是 sufficient/necessary；
- OOF 消除了 shortcut/collusion；
- 最小 rationale 等于完整 hateful event。

## 6. 停止理由与后续边界

本候选无需先运行 premise 才能裁定当前 specification 失败。直接 STOP 的解析反例是：max-over-time classifier 对一个
稳定 topic frame作出正判时，single-frame mask严格满足所有候选约束并成为全局最优，而该 frame可以与真实 hateful
span无关。该反例正好落在项目当前最核心的失败模式——video classifier强、within-video方向错误——且多 replacement
不会排除它。

若未来出现**独立于同一 video-label classifier**的新观测，例如受控且 on-manifold 的语义保真编辑、同语料可靠的
target/predicate relation监督，或不由 teacher score定义的 temporal correspondence，才值得重新立项。仅增加 replacement
数量、margin、mask temperature、continuity penalty、solver精度或更强 student，仍是在当前不可识别目标上调参，不能
解除本次 STOP。

