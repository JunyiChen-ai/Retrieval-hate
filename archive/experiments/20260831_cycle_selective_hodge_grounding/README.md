# Cycle-Selective Hodge Grounding candidate

> **淘汰：双独立 novelty review 均为 STOP（4.2/10、5.0/10）。** Gate 1 PASS、Gate 2 窄 PASS、Gate 3 FAIL。每条 categorical query 同时产生 unary 与 edge，定义上恒有 `y_ij=u_i-u_j`；因此 Hodge edge没有独立局部观测，只是 LELA式 absolute judgment 的标准图平滑。稳定 all-BOTH 可通过 consistency、curl、coverage并得到整段常数，within为`.5`。未实现、未运行新 VLM query、未生成 prediction。

截至 2026-08-31。状态：双独立 novelty/identifiability review 均已 STOP。未实现、未运行新 VLM query、未生成 prediction。

## 研究问题与证据边界

任务仍是四主语料各自独立的 weakly supervised hateful-video temporal localization。候选为 training-free localizer，不读取任何 corpus 的 span label、validation performance 或其他主数据集 train set。若进入实验，所有 performance evaluation 直接在 test 上；test GT 只用于 evaluator 与后续 developmental error analysis。

设计依据只来自当前允许的 developmental test artifacts：

- `runs/20260831_pair_alignment_attribution/main/metrics.json` 表明正确的 frame timing 在 HMM 对 local ordering 是 load-bearing，在 HCS 的高 base-tie stratum 也有贡献；
- `runs/20260831_powa_test_error_taxonomy/analysis.json` 记录 HCS 的 coarse/context-dependent boundary noise，说明孤立 frame 的绝对打分不可靠；
- `runs/20260831_test_signal_complementarity/main/metrics.json` 证明现有 frame-local signals 有真实 headroom，但 ensemble/blend 只可作 upper bound。

历史 `archive/experiments/20260831_powa_listwise_teacher_pilot/` 使用 validation，因此其 performance 永远不用于本候选设计、晋级或淘汰；本候选也不复用其 validation score。LLM judge 的 presentation-order bias 与 pairwise comparison的 cycle inconsistency由外部方法文献建立。

## 来源与 novelty 边界

跨任务来源是 HodgeRank：把 pairwise edge flow 正交分解为可由全局势函数解释的 gradient component 与 cyclic inconsistency；以及 LLM judge 的 swapped-order consistency test。初查未发现 Hodge decomposition、cycle-selective pairwise inference或其等价方法进入 hateful-video detection/localization。

目标任务已有 LELA 的逐 frame multimodal caption→LLM pointwise score，也已有一般 VLM temporal grounding、pair/listwise ranking和 HodgeRank。不能 claim VLM prompting、pairwise comparison、order swap、least squares ranking、captioning或 training-free inference 本身新。

允许的窄 claim 是：**把 hate window comparison建模成带绝对占用语义的 antisymmetric edge flow；只有同时通过 AB/BA 反对称检查与三角 cycle check 的 edge 才能进入 Hodge potential，从而让同一 VLM 的可审计局部比较而非 presentation order或不自洽 context承担最终 temporal score。**

独立 reviewer 必须判断这是否仍只是 `LELA pointwise + swap filter + standard HodgeRank` 的简单组件拼接；若第三道 non-trivial adaptation 门不成立，立即归档。

## 单一核心机制

视频按固定 8 秒 window、4 秒 stride切分。每个 window使用固定两帧、该时间段 ASR 与 OCR；所有 corpus 使用同一个 frozen VLM、同一个 hate-policy prompt、同一 decoding config。

在 window graph 中固定连接 offset `{1,2}`，因此相邻三元组产生可检查的 triangles。每条无向 edge `(i,j)` 查询两次：一次按 `A=i,B=j`，一次交换显示顺序。输出只能为：

- `A_ONLY`：A有明确 hateful evidence，B没有；
- `B_ONLY`；
- `BOTH`；
- `NEITHER`；
- `UNCERTAIN`。

交换后的 label 先映回原 window identity。两次不完全一致则该 edge abstain，不做投票或平均。映射后的 consistent label产生：

- unary occupancy targets `u_i,u_j in {-1,+1}`；
- antisymmetric comparison `y_ij=+2/-2/0`，其中 `BOTH/NEITHER` 是 tie，但绝对 unary 符号相反。

在所有 retained triangles 上计算 edge curl。任何参与非零离散 curl 的 decisive edge均 abstain；不以 GT、corpus identity或 test performance决定删除。最终唯一 window logit `z` 是固定等权目标的最小二乘势函数：

`min_z sum_(i,j) (z_i-z_j-y_ij)^2 + sum_i n_i (z_i-mean(u_i))^2`。

没有 pointwise branch、teacher blend、score calibration、dataset-specific weight、routing或第二模型。重叠 window 的 `z` 对覆盖秒做固定平均，未覆盖秒为该视频 retained unary 的中性值；这条规则在运行前冻结。

## 为什么不是 broadcast

只要 retained graph 中存在一个 `A_ONLY/B_ONLY` edge，常量 `z_i=c` 就产生固定非零 edge residual；同时绝对 unary要求两个端点符号相反。与只含 bag label 的 MIL不同，局部方向来自同一 VLM 对两个真实时间窗的显式比较。

这不保证 VLM edge正确。若 VLM输出大多为 `BOTH/NEITHER`、交换顺序不一致、cycle很多，或 edge方向与真实时序无关，方法必须 fail closed，不得靠 pointwise fallback、POWA/VERA blend或 dataset-specific阈值救回。

## Test-first premise 与固定 gates

若双 novelty review 都通过，先在 HMM/HCS 各自 test 按 sorted video ID 固定取前 8 个 eligible positive与前 8 个 negative视频，运行同一 sparse graph：

1. JSON parse rate `>=.98`；
2. AB/BA identity-consistency `>=.70`；
3. cycle filtering后，positive videos平均至少 `60%` windows有 retained unary，且最大 connected component平均覆盖至少 `50%` windows；
4. 至少 `30%` positive videos有一个 retained `A_ONLY/B_ONLY` edge；
5. mean-repeated visual frames + time-shuffled ASR/OCR control 的 decisive-edge rate必须比原输入低至少 `.10`；
6. 在这批 developmental test videos 上，recovered score 的 within ROC 必须在 HMM/HCS 都 `>.5`，且相对同 VLM、同 windows、matched query-count 的 repeated pointwise control至少一边 `>=+.020`、另一边不下降。

任一 gate失败即归档，不调 window、stride、offset、prompt、temperature或 abstention规则。

## 若 premise 通过：正式最小 evaluation

冻结全部设置，在 HMM/HCS完整 test直接评测统一 evaluator的 pooled AP、pooled ROC、within-video macro ROC。固定 controls：

1. matched-query pointwise VLM；
2. single-order pair graph；
3. AB/BA-consistent但不做cycle filter；
4. within-video time-shuffled window identities；
5. `BOTH/NEITHER` unary-only，去掉decisive edge order；
6. mean-repeated visual + shuffled ASR/OCR。

Mechanism gate：core在两语料within都胜 controls 1、2、3、5，至少一边 `>=+.020`；control 4与6不得追平，且后者within绝对偏离`.5`不超过`.01`。Performance gate：HMM/HCS六个固定指标全部严格超过 SOTA。任一失败即归档，不接 POWA/VERA、不调权重、不扩 MHC。

只有双语料机制与performance gate都通过，才扩 MHC-EN/ZH与多次独立decoding seed；最终仍要求四语料全部固定指标SOTA。

## Primary sources

- Jiang et al., *Statistical Ranking and Combinatorial Hodge Theory*, Mathematical Programming 2011: pairwise edge-flow Hodge decomposition and cyclic inconsistency.
- Rajkumar and Agarwal, *A Statistical Convergence Perspective of Algorithms for Rank Aggregation from Pairwise Data*, ICML 2014.
- Li et al., *Split and Merge: Aligning Position Biases in LLM-based Evaluators*, EMNLP 2024.
- Sun et al., *Towards Training-free Multimodal Hate Localisation with Large Language Models* (LELA), 2026 target-task nearest work.
