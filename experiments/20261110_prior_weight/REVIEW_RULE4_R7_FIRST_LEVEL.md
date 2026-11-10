# 判定：GO — 规则 4 proposal review（第 7 版 + 在 validation 上选首层深度 d）— 20261110_prior_weight 第 10 节

**判定：GO。** 规则 4 的四种 STOP 情形均不成立（第 1、2 节）。d 只能以"提问策略的一个超参数"的身份加入已过审的候选，不能作为机制或 novelty 主张；它单独拿出来就是第 4 种情形（第 2 节）。规则 13：按条文在规则内，把握约 65%，需要用户裁定后才能按规则 14(c)、14(e) 写成"已满足"（第 3 节）。规则 3 一致；规则 7 目的上一致，条文有两处没有覆盖，要写明（第 4 节）。开跑前 README 须改 8 处（5.1）。Stage A 表与原始文件一致（第 6 节）。

- 审稿人：独立 agent（Claude Fable），2026-11-10，一次性。未运行实验（只读取已有结果文件做核对），未改动本文件以外的任何文件。
- 审阅：`README.md` 第 5、7–10 节（提案 = 第 10 节后半）；`RESEARCH_ITERATION_RULES.md` 规则 3、4、7、8、11、13、14；`REVIEW_RULE4_R7.md`（第 7、8 节的位置权重 c 已过审，本次不重审，已查过的文献不重复）；分支 `r7-first-level` 相对 `main` 的改动（`src/qtl/train.py`、`src/qtl/cpolicy.py`、`search_first_level.py`、`launch/sbatch_search_first_level.sh`），以及 `src/qtl/qtree.py:depth_nodes`、`src/qtl/search.py`、`policy_select.py`；结果文件 `runs/20261110_prior_weight/all_trials/policy_select_b8_first.json`、`policy_select_b8_eig_level.json`、`policy_select_b8_3.json`、`policy_select_b8_5.json`、`all_trials/<corpus>/seed*/trial*.json`；`experiments/20261109_tree_within/README.md` 第 7 节（对照数字的来源）；`research-wiki/STATUS.md` 目标 1（停止规则）的记录。
- 代码版本：分支上最后四次提交，提交说明与时间为 "revision 7, first level: breadth-first start of depth d chosen on validation after training …"（2026-11-10 21:56）、"CPU replay launcher: run the code of its own checkout"（21:57）、"first level: validated per reported budget (candidates d <= floor(log2 B)); replay policies first4 / first5"（22:09）、"README: stopping rule against the single level, goal (b) on HateMM (section 9), proposal for the validated first level (section 10); search entry renamed"（22:30）。
- 依据的用户裁定：2026-11-10（无标签估计先验权重不违反规则 3）；2026-09-29（validation 逐秒标注不进梯度即可用）；搜索目标为 test（规则 7、10）。
- 未以可识别性、退化、shortcut 或预期涨幅作为判定依据（规则 4 明文排除）。
- 第 8 节搜索的输出不在本机（`runs/20261110_prior_weight/search/` 不存在），本审稿没有核对它的进度。

代码确认（与 README 第 10 节的表述一致）：

- `cpolicy.run_batch(first=)`：每一步，候选节点里只要还有未问的首层节点，就只在这些节点里按 EIG 取最大；首层问完后在全部剩余节点上贪心。`first=None`（d = 0）时这一段不执行，即 d = 0 是"没有强制首层"，不是"先问根节点"。
- `train.py`：训练循环与选 checkpoint 没有改（validation 上贪心策略 8 次、该 epoch 的 c）。`first_level: "validated"` 只在训练结束后的 `evaluate` 里起作用：对 d = 1..5 各跑一遍 validation 与 test 的 32 次轨迹；每个报告预算 B（0、1、2、4、8、16、32）的候选是 d = 0 与满足 2^d ≤ B 的 d，取 validation AP + ROC 最大者（并列取小的 d），test 数字取该 d 的轨迹。选择只读 validation 指标；各 d 的 test 数字都算出并存入 `depths`，不参与选择。
- Optuna 目标值读 `summary["test"]` = 预算 8、所选 d 的 test (AP + ROC) / 2；"validation 选 trial"读所选 d 的 validation 值。
- 停止规则（`adaptive`、`adaptive_voi`）仍读 d = 0 的轨迹。
- 视频太短、深度 d 的窗不能全部提问时，`qtree.depth_nodes` 退到更浅的一层，之后贪心补到预算；所以短视频上"d = 3"不等于"8 个等长窗"。
- 改动里没有按语料分支的代码；`search_first_level.py` 的 `EXTRA_DEFAULTS` 三语料相同。

## 1. 第 1 种情形：检索记录（WebSearch 27 次；WebFetch 5 次，其中 2 次失败；2026-11-10）

要查的问题：hateful video 检测 / 定位里，有没有人在 validation 上在"自适应"与"非自适应（等长窗 / 逐层）"的 VLM 提问顺序之间做选择，或者给粗到细的时间提问树设一个在 validation 上选的"首层深度"。

| # | 查询 | 结果 |
|---|---|---|
| 1 | hateful video detection VLM query schedule adaptive versus uniform segments chosen on validation HateMM | HateMM 原文（1 fps、每视频 100 帧）、MARS（2601.15115）、SCANNER（2602.00132）、MultiHateLoc（2512.10408）、SAGE、MAESTRO（DSTA 报告）、TANDEM（2601.11178）；无命中 |
| 2 | hateful video temporal localization coarse-to-fine tree VLM questions segments weakly supervised 2026 | 只有 MultiHateLoc 各版本；无树、无 VLM 提问 |
| 3 | hate video detection multimodal LLM "uniform" segments OR "equal-length" clips versus "adaptive" segment selection keyframe hateful | VMID（2411.10032，假信息，固定时长分段）、MAESTRO、SafeLens（AAAI-26 demo）、LELA（2602.09637）、HVGuard；无两种取段方式的对比 |
| 4 | harmful video moderation VLM hierarchical coarse-to-fine temporal search breadth-first depth hyperparameter | 2609.26233、LELA、SafeLens（2605.17610）、Filter-And-Refine（2507.17204，视频级级联）、TANDEM、2508.10974（稀疏均匀取帧会漏掉有害片段）；无首层深度 |
| 5 | MAESTRO multimodal agentic model efficient synergistic text reasoning optimisation hateful video detection frame selection | MAESTRO（DSTA 报告）：视频级仇恨检测，迭代推理、不逐帧处理；全文未能获取（见下） |
| 6 | TANDEM temporal-aware neural detection multimodal hate speech arXiv 2601.11178 method segments temporal grounding | TANDEM（Koushik, Treharne, Kanojia，2026-01）；全文核对见下 |
| 7 | adaptive versus non-adaptive group testing dense positives prevalence threshold individual testing optimal | Bay–Price–Scarlett（阳性稠密时逐个检测在非自适应设计里渐近最优）、Aldridge–Baldassini–Johnson（1306.6438）、McMorrow–Scarlett；检索摘要级 |
| 8 | long video question answering VLM tree search coarse-to-fine adaptive breadth depth expansion VideoTree keyframe uniform sampling comparison | VideoTree（2405.19209，CVPR 2025）：按查询自适应的广度 / 深度展开，对比均匀取帧的 LLoVi |
| 9 | Bayesian optimization initial design size number of initial random points hyperparameter "initial design" adaptive acquisition after space-filling effect | Cheon–Koh–Tsay 2026（2607.04356，"How Many Initial Points Does Bayesian Optimization Need?"：初始点数与总代价呈 U 形）；经验规则 4d、10d；检索摘要级 |
| 10 | video anomaly detection VLM hierarchical tree coarse-to-fine segments granularity VADTree OR "HiProbe" OR "Holmes-VAU" adaptive temporal sampling vs uniform | VADTree（2510.22693，NeurIPS 2025）：按事件边界建粗到细的树；动机是固定窗长不适合时长不一的异常 |
| 11 | hateful OR harmful video localization VLM "binary search" OR "divide-and-conquer" OR "hierarchical" temporal segments query budget number of VLM calls | 2508.10974、Triage（2601.22959）、FV-Action（2608.08315）、2604.08966、LELA、TANDEM、SafeLens、TempJail（2608.19737）；任务内无命中 |
| 12 | SafeLens segment-level hate speech detection online videos arXiv 2605.17610 fast slow inference segmentation method | 两篇同名：AAAI-26 demo（基于 HateClipSeg 的逐段检测）与 2605.17610（通用视频护栏，快慢两级筛查）；都没有时间上的提问顺序 |
| 13 | two-stage hybrid group testing non-adaptive first stage then adaptive, number of stages chosen, "two-stage" adaptive group testing noisy tests | GROTESQUE（1307.2811）、Scarlett（1803.05099，两到三个阶段）、Aldridge（2005.06617）、Teo–Scarlett（2106.12193）：第一阶段非自适应、之后自适应是常见设计，阶段数是设计参数 |
| 14 | active learning OR Bayesian experimental design fixed initial non-adaptive design then sequential adaptive; when is adaptivity beneficial "adaptivity gap" noisy observations uniform design matches adaptive | Golovin–Krause–Ray（1010.3091：有噪声时贪心二分可以很差）、NeurIPS 2019 带噪声的最优决策树（自适应差距随实例变化）、Castro–Nowak（Allerton 2006） |
| 15 | "Your VLM Already Knows When" training-free temporal grounding asking yes or no uniform probes coarse fine scan | FV-Action（2608.08315）、2605.21954、HiTeA（ICLR 2026） |
| 16 | hateful video detection LLM agent iterative frame retrieval adaptive sampling "hateful video" keyframe selection ablation uniform sampling 2026 | MAESTRO、MARS（均匀取帧，对取帧率做参数分析）、LELA、SAGE；无均匀对自适应的消融 |
| 17 | video temporal grounding VLM search policy selected per dataset on validation set coarse-to-fine levels hyperparameter "number of levels" OR "initial granularity" sliding window scales | FV-Action（各语料的粗探针数按时长手设）、ScanFocus（2607.13421）、Zoom-Zero（2512.14273）、2506.18883；无"在 validation 上选粗层粒度" |
| 18 | "Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning" MARS frame sampling arXiv | MARS（2601.15115，ICASSP 2026）：每视频均匀取 16 帧，视频级 |
| 19 | weakly supervised temporal action localization OR video anomaly detection VLM queries "information gain" sequential segment selection warm start uniform coverage then adaptive refinement budget | MuST-VAD（2608.06913）、"The Road Less Seen"（CVPR 2026）、InfoShot（2603.17374）；无命中 |
| 20 | information pursuit OR "twenty questions" active testing localization coarse-to-fine dyadic tree first level fixed queries then adaptive Geman Jedynak "coarse-to-fine" search depth | Blanchard–Geman（Annals of Statistics）、Sznitman–Jedynak（1003.5249）、Jedynak–Frazier–Sznitman 2012、Chattopadhyay et al.（ICML 2024） |
| 21 | hate video span localization zero-shot VLM sliding window segments cost number of calls "hateful" video "segment-level" LLM 2026 arXiv DeHate OR HateClipSeg OR ImpliHateVid | OZ-TAL（2605.09976）、2501.07972、TANDEM、LELA、HateClipSeg（2508.01712）；无提问次数或取窗方式的分析 |
| 22 | DeHate dataset hateful video localization benchmark temporal segments deepfake hate paper baselines VLM | DeHate（ACM MM 2025，6689 个视频，段级标注，benchmark 是检测模型） |
| 23 | hateful video detection "cost-aware" OR "budget" OR "early exit" OR "adaptive computation" multimodal LLM calls video moderation selective querying 2026 | LEAF（Findings of ACL 2026，蒸馏）、StreamSense（2601.22738，直播流，按需路由到 VLM）、SafeLens、MAESTRO、LELA；无按预算安排的提问顺序 |
| 24 | Jedynak Frazier Sznitman "twenty questions with noise" dyadic policy non-adaptive versus probabilistic bisection adaptive equal entropy | J. Appl. Probab. 2012：有噪声的二十问，熵损失下自适应的概率二分与非自适应的 dyadic 策略都是最优 |
| 25 | hateful video detection temporal localization "breadth-first" OR "level-wise" OR "equal windows" OR "uniform partition" VLM questions tree adaptive "expected information gain" | LELA、MAESTRO、MultiHateLoc；无命中 |
| 26 | model selection between adaptive and static query strategy on held-out validation data "policy selection" active feature acquisition OR sequential testing choose acquisition strategy by validation performance | 主动特征获取：2006.07701（动态一般好于静态）、2407.13429 与 2508.02566（静态策略在部分任务上不输或更好）、AFABench（2508.14734）；未见把"在 validation 上在自适应与静态之间选"作为方法的工作 |
| 27 | multimodal hate speech video 2026 new method "temporal localization" OR "temporal grounding" LLM segments arXiv September OR October OR November 2026 hateful video | LELA、TANDEM、SafeLens demo、2508.04900；未检索到 2026 年 9–11 月的新方法 |

全文 / 摘要核对（WebFetch）：

- TANDEM（HTML，前 10 万字符）：固定 30 秒分块（"we partition each input video into manageable 30-second chunks"），块内按镜头切换选关键帧，失败时退回约 1 fps 均匀取帧。原文："for the VL model, we omit the temporal segmentation as it degraded the classification and target identification performance. However, for MHC, we enable temporal segmentation for the VL model"——按数据集手动决定 VL 模型是否做时间分段，没有给数字，没有说在 validation 上选。没有树、没有自适应提问、没有调用预算。这是任务内最接近的一条：按数据集开关分段，但不是按规则在 validation 上选的提问顺序。
- FV-Action（2608.08315；摘要页，加 HTML 前 10 万字符）：任务外（Charades-STA、ActivityNet Captions、TACoS、QVHighlights）。先在 [0, T] 上均匀放 K_c 个 yes/no 探针（12 / 16 / 16 / 96，按视频时长定间隔，原文称除窗宽 W 外不用目标标签调），取 3 个峰，再在每个峰附近放 8 个细探针。第 5.5 节：把均匀探针换成按内容选的关键帧，Charades-STA 的 R@0.5 从 56.8 降到 43.7；原文 "Given a fixed budget and no reliable prior over where the event lies, spreading probes uniformly is the better strategy"；按视频自适应的探针数在 ActivityNet 上没有增益。没有在 validation 上选粗层数量。
- MAESTRO（DSTA 报告）：PDF 两次请求都返回 403，未读到全文；只有检索摘要级的信息（视频级仇恨检测，"Adaptive Global-Local Reasoning Loop"，不逐帧处理）。摘要级未见时间定位、树、调用预算或在 validation 上选策略。
- 其余条目为检索摘要级，未读全文。LELA、SCANNER、SAGE、LEAF 的全文 / 摘要核对见 `REVIEW_RULE4_R7.md` 第 2 节，不重复。

结论：hateful video 检测 / 定位文献里没有"在 validation 上在自适应与等长窗的 VLM 提问顺序之间选择"，也没有"时间提问树的首层深度"这一对象。第 1 种情形不成立。任务外的相关做法（论文引用时用）：均匀探针对按内容放置的探针（FV-Action 第 5.5 节）；分组检测里阳性稠密时逐个检测最优、两阶段设计（第 7、13 条）；有噪声二十问里非自适应 dyadic 策略同样最优（第 24 条）；贝叶斯优化的初始设计点数（第 9 条）；主动特征获取里静态对动态策略（第 26 条）。任务内需要提到的是 TANDEM 的按数据集开关分段。

## 2. 四种 STOP 情形，重点是第 4 种

1. **来源方法已用于 hateful video**：否（第 1 节）。
2. **纯 training/test ensemble**：否。单一网络、单一 VLM、单一答案表、单一后验；每个视频只用所选 d 的一条轨迹出分数，不同 d 的预测不合并。
3. **纯 calibration / 后处理 / 平滑**：否。d 决定问哪些节点（读取哪些 VLM 回答）；分数仍由同一次精确推断直接输出，之后没有任何作用在分数上的映射。
4. **纯工程技巧**：否，但要分两层说。
   - (a) 第 10 节相对第 8 节新增的内容本身就是"只调超参"：一个整数、最多 4 个候选（8 次时）、按 validation 指标选。如果它单独作为一个候选或一个 novelty 点提出，属于第 4 种，应 STOP。README 自己也写了"it is a hyper-parameter"。
   - (b) 规则 4 判的是候选整体。候选整体 = 第 6 版（树上的后验模型、答案表、EIG 提问；已过审）+ 位置权重 c（已过审）+ d。给一个已过审的完整方法加一个按预先声明的规则选出的超参数，不会让整体变成"纯工程技巧"；规则 13 本身就预设方法带有按 validation 选的标量超参数。所以不 STOP。
   - (c) 这个放行带四个条件。前两个 README 已写，后两个要补（5.1 第 3、8 条）：
     - d 不作为机制或 novelty 主张，不读规则 14(g)。
     - 汇报和论文里，HateMM 相对第 6 版的涨幅要写明来自提问策略的超参数 d，不来自 c。Stage A：c 在 d = 0 下 HateMM 的配对差是 AP +.002 / ROC +.004；选 d 之后相对 d = 0 的配对差是 AP +.021。
     - c 的 14(g) 消融要在候选自己的配置（所选 d）上做，不能只在 d = 0 上做。
     - 所选 d = log2 B 的语料上，"由 EIG 选节点"这一部件不起作用（8 次、d = 3 时问的就是 8 个等长窗，EIG 只决定顺序）。因此在这个候选里，EIG 自适应提问在 HateMM 上按 14(g) 得不到"去掉后下降"的消融结果，按 14(g) 的条文不能列为三语料都成立的 novelty 机制。这不是本提案造成的（第 9 节：d = 0 时在 HateMM 上把 EIG 提问换成 8 个等长窗，AP 反而高 .021），但本提案把它固定进了方法。树是用户 11-09 指定的核心 novelty，这一点要在汇报时交给用户。本条不是 STOP 依据。

## 3. 规则 13 与 14(e)

规则原文（节选）："一个方法 = 一套架构、一套损失、一套训练流程、一套推理流程，四个语料完全相同。语料之间只允许标量超参数不同（学习率、epoch、top-K、时间窗长度、损失权重），且必须由同一份预先声明的 validation 搜索空间和选择规则自动选出。以下按语料改动一律视为不同方法……：手写的、按语料不同的规则或 policy 表达式；……模块开关不同；……按语料挑不同分支或 head 报数。" 14(e)："无 inference 后处理、无 ensemble、无按语料分支选择"。

支持"在规则内"的事实：

- (a) d 是一个整数；候选集合（0..floor(log2 B)）和选择规则（validation AP + ROC 最大，并列取小）三语料相同，写在搜索前；代码里没有语料名分支。条文对"允许"给的两个条件（标量；同一份预先声明的 validation 规则自动选出）都满足。
- (b) 条文自己举的允许项里有 top-K 和时间窗长度，都是改变结构的整数。d 决定首轮窗长 T / 2^d，性质最接近"时间窗长度"。
- (c) 被列为反模式的几项，限定语是"按语料改动"，举的例子（POWA）是手写的、按语料不同的设定。这里返回的 d 是选择规则的输出，同一语料的不同 trial 也会不同（Stage A：HCS 上四个值都被选到过）。
- (d) 架构、损失、训练流程、答案表、后验推断、c 的估计三语料相同；d 只在训练结束后起作用。

支持"在规则外"的事实：

- (e) 候选集合的两端不是同一机制的强弱两档。d = 0 时没有强制首层（`first=None`）；2^d = B 时问哪些节点完全由 d 决定，EIG 只定顺序，对固定预算下的分数没有影响。在主报告点（8 次）上，d = 0 与 d = 3 的区别就是"EIG 选节点"这个部件开或关，d = 1、2 是两者的混合。README 也记录了二选一 {d = 0, 单层} 给出同样的结果（`policy_select_b8_eig_level.json`）。这与清单里的"模块开关不同"效果相同，区别只是开关由 validation 规则决定，不由人决定。
- (f) 预期结果（Stage A）：HateMM 90% 的 trial 选 d = 3，HCS 82% 选 d = 0 或 1。读者看到的是：HateMM 用 8 个等长窗加后验模型，HCS 用自适应提问。"一套推理流程"能否涵盖这两种，取决于把推理流程定义成"带一个整数参数的提问程序"还是"实际执行的提问方式"。
- (g) 树上的自适应提问是用户 11-09 指定的核心 novelty。一个语料的主表数字来自该部件不起作用的配置，与"按语料挑不同分支报数"关注的是同一件事；区别是这里由预先声明的 validation 规则挑，不是按 test 或人工挑。

读法：**按条文，d 在规则 13 允许的范围内**——条文给的是两个可检查的条件，两个都满足，(b) 是条文内的直接类比。按规则的目的（不把按语料不同的做法合并成一个方法 claim），它在边界上：成立的前提是方法主张里不包含"自适应提问在每个语料都起作用"，README 的 "Not claimed" 一段已经这样收窄。**把握：约 65% 认为用户会认定在规则内**；不到可以当作定论的程度，原因是 (e) 和 (g)。

**需要用户裁定：是。** 规则 11 写明流程规则只由用户改；清单里的"模块开关不同"与本提案在 (e) 上重叠；且涉及用户指定的核心 novelty。裁定要回答两件事：① 按 validation 选的整数 d（两端分别是自适应提问和等长窗）算规则 13 允许的标量超参数，还是模块开关；② 若算超参数，"HateMM 上所选策略就是单层"是否接受为目标 (b) 的结论。裁定之前，本候选不能按 14(c)、14(e) 写成"已满足"。规则 13 不在规则 4 的四种 STOP 情形里，所以不影响本审稿的 GO；是否等裁定后再开跑由主 agent 按用户 11-10 的自主迭代指令决定（5.2 第 1、2 条）。

如果用户裁定在规则外，统一的做法只剩两种：三语料固定同一个 d（d = 0 就是第 8 节候选；d = 3 时 HCS 低于第 6 版，Stage A 各自最优 trial .677 / .667 对 .691 / .680），或找到一个不需要按语料选的策略（第 5、9 节试过的都不行）。这是 5.1 第 1 条要求保留第 8 节搜索的原因。

## 4. 规则 3 与规则 7

**规则 3**（"禁止 inference 时的后处理（平滑、calibration、按语料路由）"）：一致，把握高。d 在 test 推理之前由 validation 定下，是部署模型的一个常数；它改变读取哪些 VLM 回答，不改变分数的计算，分数之后没有任何映射；每个视频只走一条轨迹。"按语料路由"说的是推理时按语料把样本送进不同分支的后处理；这里每个语料本来就各自训练一个模型，d 是该模型的一个设置。"按语料不同"这一点归规则 13 管（第 3 节）。用 validation 逐秒标注选 d 不进梯度，符合 09-29 裁定（与选阈值同类）。

**规则 7**（"每个 trial 的流程一样：train set 训练，validation set 选 checkpoint……然后用该 checkpoint 在 test 上跑全部三项指标"；"Validation 只用于 trial 内选 checkpoint，不用于比较方法或决定方向"）：

- 目的上一致，把握约 80%。在 trial 内用同一判据（validation AP 与 ROC 的均值）再选一个 d，位置与选 checkpoint 相同（训练后、test 前、每个 trial 各自做），不是拿 validation 数字去比较候选方法或决定研究方向。
- 条文有两处没有覆盖，要在 README 写明，不能默认（5.1 第 8 条）：① 规则 7 只写了"选 checkpoint"，d 是 trial 内的第二次 validation 选择；② 规则 14(d) 写"超参数来自第 7 条的固定 Optuna 搜索"，d 不在 Optuna 搜索空间里。另一种做法是把 d 放进 Optuna、按 test 目标选，更贴 14(d) 的字面，但那是按 test 选策略，而且每个 seed 各选各的；按 validation 选与规则 13 的条文（"由同一份预先声明的 validation 搜索空间和选择规则自动选出"）一致，本审稿认为是两者中更合规的一种。
- Optuna 目标值 = 所选 d 在 8 次的 test (AP + ROC) / 2：符合规则 7，目标值是候选部署配置的 test 数字。
- checkpoint 仍按贪心策略（d = 0）的 validation 判据选，之后才选 d，所以 HateMM 上选 checkpoint 用的策略与 test 用的策略不同。这是预先声明的设计，不违规；开跑后不得改成联合选（epoch, d）。
- 中途停掉第 8 节搜索，与规则 7 "机器被占可以暂停，不可以缩减"和 14(d)"目标值定义……不得事后改"的关系，见 5.1 第 1、2 条。

## 5. 第 10 节的协议

### 5.1 开跑前 README 必须改的点（每条一句）

1. 把"第 8 节搜索停止"改成"暂停"：保留 `runs/20261110_prior_weight/search/` 的全部目录和 `optuna.db`（同一命令可续跑），并写明续跑条件——用户裁定 d 不在规则 13 内，或 R1 / R2 不过时，按第 8 节原协议跑满登记的 trial 数并按第 8 节读数。
2. `depths["0"]` 的读数改称"固定 d = 0 的消融行"，不称为第 8 节候选按规则 7 / 14(a) 的搜索结果（trial 10–19 的超参数由所选 d 的目标值引导，不是 d = 0 自己的目标值）；并预先写明这一行报三种读法：各自最优 trial、全部 trial 的配对差、只用 trial 0–9（超参数与目标值无关）的配对差。
3. c 的规则 14(g) 消融补"候选自身配置"一行：在本次搜索按所选 d 的 test 目标选出的 trial 上，同一 checkpoint 令 c = 1，d 在 c = 1 下重新按 validation 选和 d 保持不变两种都报（Stage A 没有"不带权重的单层"的全 trial 重放，c 在 d = 3 下的作用目前没有数字）。
4. R1 的对照数字按一条规则取，并写来源文件（`runs/20261109_tree_within/naive_questions/<corpus>.json`）：现在 HCS 用的是"alpha 按 test 取最好"的一行（.693 / .651），DeHate 用的是 alpha = .5 的一行（.243 / .773；按 test 取最好是 .243 / .777），HCS 换另一个 baseline 的同一行是 .696 / .653；同时写明"高于"需要的幅度（规则 8 / 14(b)：不小于两者 seed 标准差中较大者，最低 .005）。
5. R2 写明容差（第 8 节 Q1 是"低不超过 .005"，R2 只写"not below"），并把规则 8 的主门（六个数对 baseline 表）单列一行，不与目标 (c) 写在同一条里。
6. R3 写明：所选 d = 3 的 trial 上，"所选 d 对单层"是同一策略自己比自己，汇报为"所选策略就是单层，目标 (b) 在该语料未达成"，不汇报为"目标 (b) 前半成立"；所选 d = 0 时对贪心策略同理。
7. 写明停止规则在新候选里的地位：`adaptive` 的数字属于 d = 0 的轨迹；在所选 d ≠ 0 的语料上，固定预算的报告配置与停止规则是两个不同的提问策略，要写清论文报哪一个，以及停止规则的代价（STATUS 目标 1：相对固定 8 次 AP / ROC 降不超过 .01）对哪个固定预算参照读（第 5 节：HateMM 上带停止规则的贪心在平均 8 次时比单层低 AP .017）。
8. "Why a validated hyper-parameter" 一段里关于规则 13 的句子改为"主 agent 的读法，待用户裁定"，并写明 d 不在 Optuna 搜索空间内、不按 test 选，是 trial 内继 checkpoint 之后的第二次 validation 选择（汇报时按 14(d) 照此写）；"What is and is not claimed" 补一句：HateMM 相对第 6 版的涨幅来自 d，以及第 2 节 (c) 最后一条关于 EIG 提问与 14(g) 的后果。

### 5.2 非阻断备注

1. **两种裁定下都不浪费的顺序**是先让第 8 节搜索跑完，再开本搜索：第 8 节搜索在"d 在规则外"时是候选本身，在"d 在规则内"时是固定 d = 0 的、由自己目标值引导的消融行（比 `depths["0"]` 干净）。用户 09-06 说过"既然你已经开始跑了 先跑完再看"。代价是多占 GPU 约一天。是否这样做由主 agent 决定；选择先停的话，5.1 第 1、2 条是最低要求。
2. README 第 9 节的结论是目标 (b) 在 HateMM 上达不到。按用户 11-10 的指令（达到目标或证明不可能之前不汇报），这个结论本身就是该汇报的时点；规则 13 的问题和树的主张可以放进同一次汇报。
3. 逐预算选 d：每个预算从最多 6 条固定轨迹里选一条，不是每个预算重跑一次（与第 7 节否掉 plan 的理由不冲突）；但预算曲线上相邻两点可能来自不同的 d，画曲线时要标出每点的 d。Stage A 只在 8 次上检验过按 validation 选 d；16、32 次没有（本机的结果文件里没有 `first4|16`，也没有 8 次以外预算的 `@val` 重放），逐预算选择的依据只有 HateMM 上"单层 8 个再贪心 8 次"对"16 个等长窗"这一个比较。
4. 选择的噪声：所选 d 等于 test 最好的 d 的比例是 HateMM 77%、HCS 38%、DeHate 18%（`chosen_is_test_best`；4 个候选随机选是 25%）。HCS 上按各自最优 trial 读，选 d 比固定 d = 0 低 AP .005 / ROC .008（.703 / .684 对 .708 / .692），比 README 第三条引用的全 trial 均值差（-.004 / -.005）大；R4 汇报时两种读法都给。并列规则（取小的 d）与代码一致；选择没有设最小差距，这一点照现在的声明执行，不要在搜索后加。
5. 单 trial 耗时：每个 trial 多 10 遍 32 次的策略运行（5 个深度 × validation 和 test），第一个 trial 也包含在内，它决定 trial 数。参照：第 6 版 DeHate seed 234 trial 0 从写出 `model.pth` 到第一个 test 评测文件相隔约 100 秒（两遍带 VOI 记录的运行，三个进程共用一块 GPU）。按规则 7，某个 seed 首个 trial 超过 3600 秒就是 5 个 trial，不改；出现这种情况时，R2 与第 6 版（20 个 trial）的比较要注明 trial 数不同。
6. 开跑后的复现检查：第 8 节搜索已完成的 trial（0–9 的超参数相同、训练相同）应与本搜索同号 trial 的 `depths["0"]`（8 次）数字一致；把差异报出来。
7. 对照在 HateMM、DeHate 上平均只用 7.13 / 6.56 次（短视频不足 8 个窗），候选在短视频上会贪心补到 8 次；R1 旁边并列报 `test_mean_calls`。
8. 措辞："d = 0 is the greedy policy"之外补一句 d = 0 表示没有强制首层（不是先问根节点）；"d = 3（single level）"在短视频上不严格等于 8 个等长窗。
9. 论文须引用并划清：FV-Action（均匀探针优于按内容放置，结论与第 9 节的原因 (2) 同类）；分组检测的稠密区间与两阶段设计；Jedynak–Frazier–Sznitman 2012；主动特征获取的静态对动态；TANDEM 的按数据集开关分段（本方法是按预先声明的 validation 规则选，不是手设）。

## 6. 第 10 节 Stage A 表与 `policy_select_b8_first.json` 核对

逐格核对了表中 HateMM、HCS、DeHate 三列的全部 11 行，均与文件一致（3 位小数）：

- 所选 d 的占比：文件 .05 / .033 / .017 / .90，.50 / .317 / .10 / .083，.267 / .317 / .317 / .10 → 表 5 / 3 / 2 / 90%，50 / 32 / 10 / 8%，27 / 32 / 32 / 10%。
- test 最好的 d 的占比：.133 / .033 / .05 / .783，.567 / .35 / .083 / 0，.15 / .367 / .317 / .167 → 表一致。
- 所选 d 的全 trial 均值：.6928 / .8838 / .8092，.6812 / .6639 / .5849，.2473 / .7739 / .6801 → 表一致。
- 对第 6 版的配对差：+.0225 / +.0062 / +.0455，+.0072 / +.0064 / +.0308，+.0202 / +.0213 / +.0111 → 表一致。
- 对 d = 0 的配对差：+.0207 / +.0021 / +.0272，-.0041 / -.0048 / +.0022，+.0017 / .0000 / +.0007 → 表一致。
- 对 d = 3 的配对差：-.0005 / +.0005 / -.0025，+.0168 / +.0182 / -.0011，+.0008 / +.0048 / -.0074 → 表一致。
- 各自最优 trial（3 seed 均值）：所选 .7220 / .8982 / .8312，.7028 / .6838 / .6171，.3045 / .7924 / .6939；d = 0 .6977 / .8890 / .7736，.7076 / .6922 / .6171，.2981 / .7929 / .6929；d = 3 .7220 / .8982 / .8312，.6771 / .6670 / .6030，.2911 / .7842 / .7006；第 6 版 .7017 / .8890 / .7827，.6915 / .6797 / .5630，.2780 / .7698 / .6810 → 表一致。
- 表下文字：二选一文件 `policy_select_b8_eig_level.json` 的 .722 / .899 / .829，.704 / .686 / .611，.298 / .793 / .693 一致；"单层 8 个再贪心 8 次".688 / .884 / .791、"16 个等长窗".700 / .887 / .797、"8 个等长窗".693 / .883 / .811，由 `all_trials/hatemm/seed*/trial*.json` 的 60 个 trial 重新求均值得 .6885 / .8842 / .7910、.7000 / .8866 / .7970、.6932 / .8834 / .8107，一致。

需要改的小处（不影响结论）：

- 第二条里相对第 6 版的差是用四舍五入后的数相减得到的，有三处与文件直接相减差 .001：HCS AP 写 +.012，文件 .7028 − .6915 = +.011；DeHate ROC 写 +.022，文件 +.023；HateMM within 写 +.048，文件 +.049。
- 同一条写"ahead of … revision 6 on all six AP / ROC numbers"：HCS ROC 的差是 +.004，在规则 8 的 .005 噪声带内，应写"不低于"。
- 第三条"选 d 的代价"只引了全 trial 均值；按各自最优 trial，HCS 是 -.005 / -.008（见 5.2 第 4 条），表里两行都在，文字补一句即可。
