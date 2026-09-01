# Benign-Anchored Antisymmetric Robust Hodge Grounding

> **淘汰：双独立审查为 `CONDITIONAL GO 6.7/10` 与 `STOP 4.8/10`；novelty硬门未全过。** 新版消除了旧 unary/edge 冗余，但在 source-faithful difference judge 下，共同 reference 只改变所有target分数的全局常数，rank指标不变；within edges又由reference-star edges确定，完整图只剩标准 robust rank aggregation。Negative medoid未成为load-bearing hate语义约束。未实现、未运行 VLM query、未生成 prediction。

截至 2026-08-31。状态：novelty硬门STOP；未实现、未运行 VLM query、未生成 prediction。

## 与被淘汰版本的唯一关系

`archive/experiments/20260831_cycle_selective_hodge_grounding/` 因 `y_ij=u_i-u_j` 的代数冗余被双审 STOP。本候选不是修改其 gate：彻底删除 `A_ONLY/B_ONLY/BOTH/NEITHER`、per-edge unary、hard edge deletion与neutral fill。任何 edge只观察相对强度；跨视频 gauge由一个独立、train-negative、同域 reference node提供。

## 来源与 novelty 三门

允许 adaptation 的跨任务来源：

1. HodgeRank/robust Hodge regression：从 noisy pairwise edge flow恢复全局 potential；
2. comparative judgment 的 fixed reference/gold-standard anchoring：用共同 reference把不同 item set放在同一尺度；
3. LLM judge swapped-order antisymmetrization：消去 presentation-order 的对称 bias component。

初查未发现 fixed benign reference + antisymmetrized robust Hodge用于 hateful-video detection/localization。目标任务最近邻 LELA/TANDEM 已占 training-free VLM hate scoring/grounding；一般 pairwise judge、HodgeRank、robust regression、negative exemplar和medoid selection也都不能 claim 新。

拟主张的窄 adaptation：**利用 weak supervision 中 negative video 的全窗 benign certificate构造同域公共 reference；reference duels提供跨视频尺度，同视频 duels提供独立 temporal order，并在同一个 anchored robust-Hodge objective中通过 reference–i–j cycles互相约束。** 这针对 hate localization 的两个具体问题：absolute VLM score跨视频受topic/style影响，而纯within rank无法通过 pooled指标。独立 reviewer必须判定它是 non-trivial task mechanism，还是 `negative exemplar prompting + pairwise VLM + standard HodgeRank` 的简单组件拼接。

## Train-only benign reference

四语料完全独立。对当前 corpus 的 negative-train videos切固定 8秒/4秒stride windows。使用已有 frozen MultiHateLoc input features，对每种 modality先L2-normalize，再concat并再次normalize。选择使所有 negative-train windows平均 cosine distance最小的**唯一 observed medoid window**；tie按 `(video_id,start)` 排序取首个。

reference必须是一个真实 window，保留它自己的两帧、ASR和OCR；不平均媒体、不看validation/test、不用其他主语料。selection artifact记录输入路径、train IDs、窗口数、最终 `(video_id,start,end)`、feature coverage与命令。若任一corpus没有negative train window或媒体不可读，fail closed。

## 纯相对 VLM edge observation

Test video按同一8秒/4秒stride切窗，节点为所有 target windows与reference `r`。固定 edge set：

- 每个target window连接同一`r`，建立公共跨视频尺度；
- 同视频 window offsets `{1,2}`，建立局部与稍长程顺序并产生 `r-i-j` triangles。

每条 edge查询同一个 frozen VLM两次，媒体展示次序分别AB与BA。Prompt只问“哪个 window 含更强的明确 hateful evidence”，输出固定五级相对选择 `{A_STRONG,A_WEAK,TIE,B_WEAK,B_STRONG}`，不输出任一 window 的 absolute hate label。

实现读取五个固定 choice token 的 next-token probability，不解析自由文本。映射值 `v={+2,+1,0,-1,-2}`。若AB orientation的期望为`q_AB`，BA映回原identity后的期望为`q_BA_mapped`，最终 edge flow固定为：

`y_ij = (q_AB + q_BA_mapped)/2`。

presentation-order bias diagnostic为两者差的一半，只报告，不做删边、阈值或routing。所有 finite edge保留。

## Anchored robust Hodge potential

固定reference potential `z_r=0`，其余window score解：

`min_z sum_edges huber((z_i-z_j)-y_ij; delta=1)`。

所有edge等权，`delta`固定，不按corpus/test调整。reference star保证每个window与同一gauge连通，解不存在component平移不定。Robust loss让冲突edge留下可审计residual，而不是output-dependent删除。最终每秒raw logit只由覆盖它的window potential做固定算术平均；不存在其他branch、CDF、temperature scaling、POWA/VERA blend、per-video centering或dataset routing。

## 机制识别与反例

- **Pooled-only broadcast**：若所有reference edges相同、within edges全tie，所有`z_t`可为常数，within=`.5`。方法不声称解析上无条件排除；它要求真实同视频duels提供非零local flow，并由test premise直接否证。
- **Within-only scale failure**：去掉reference edges后，每视频potential只确定到平移，不能支持pooled AP/ROC；anchor-only与within-only都是必做controls。
- **Reference artifact**：单一medoid可能携带topic/style。必须比较固定第二近medoid、最远negative window与blank/silence reference controls；它们只做机制诊断，不允许按corpus选择anchor。
- **Stable wrong judge**：swap antisymmetry与低Hodge residual不保证edge正确。time-shuffle、media-identity swap和GT test evaluation必须直接检验。
- **Hodge decoration**：若anchor-only或逐edge直接平均追平完整图，within edges/Hodge claim失败。

## 双独立 novelty/identifiability gate

Reviewer必须给三门verdict，并回答：

1. reference-anchored comparative scaling、robust Hodge或等价机制是否已进入 hateful-video detection/localization；
2. certified-negative common reference是否只是普通 few-shot exemplar prompt；
3. reference edges与within edges是否为真正独立 observations，还是仍有隐藏的确定性冗余；
4. single reference star + robust Hodge是否形成任务特定机制，或只是标准ranking pipeline；
5. fixed medoid selection是否会把负视频身份/topic变成shortcut；
6. next-token five-choice probabilities、AB/BA averaging与Huber是否构成 calibration/ensemble/routing。

任一 novelty硬门失败立即归档，不实现。

## 若review通过：test-first premise

只先运行HMM/HCS。各自从test按sorted ID固定取前8个eligible positive与前8个negative videos；同一配置，不看validation。固定 gates：

1. media、ASR/OCR、choice-token、finite probability与graph coverage全部100%；
2. AB/BA mapped expectation Pearson `>=.70`，mean absolute order-bias `<=.50`（五级值范围4）；
3. positive test videos中至少75%具有至少20%的within edges满足`|y|>=.25`；
4. mean-repeated visual + time-shuffled ASR/OCR使within-edge `mean |y|` 相对原输入下降至少20%；
5. full potential在两语料subset within都`>.5`，且都胜anchor-only与matched-query pointwise，至少一边`>=+.020`；
6. negative subset的reference-edge potential均值低于positive subset，且pooled ROC方向在两语料都`>.5`。

任一失败即归档，不调window、edge offsets、prompt、choice mapping、Huber或reference。

## 正式两语料 evaluation

Premise通过后冻结全部设置，完整HMM/HCS test评测pooled AP、pooled ROC与within-video macro ROC。Controls：pointwise、anchor-only、within-only、ordinary least squares、single-order、第二近medoid、最远negative、blank reference、time-shuffled媒体、mean-repeated视觉+shuffled文本。

Mechanism gate：core在两语料within都胜pointwise、anchor-only、within-only与single-order，至少一边`>=+.020`；second-medoid不得改变结论，far/blank不得追平；shuffle/mean-repeat不得追平且mean-repeat within在`.5±.01`。Performance gate：HMM/HCS六项全部严格超过SOTA。失败即归档，不接任何teacher blend或calibration。

通过后才扩MHC-EN/ZH与多decoding seed；最终晋级仍要求四主语料全部固定指标严格SOTA。

## Primary sources

- Jiang et al., *Statistical Ranking and Combinatorial Hodge Theory*, Mathematical Programming 2011.
- Rajkumar and Agarwal, *A Statistical Convergence Perspective of Algorithms for Rank Aggregation from Pairwise Data*, ICML 2014.
- Li et al., *Split and Merge: Aligning Position Biases in LLM-based Evaluators*, EMNLP 2024.
- Sun et al., *Towards Training-free Multimodal Hate Localisation with Large Language Models* (LELA), 2026.
