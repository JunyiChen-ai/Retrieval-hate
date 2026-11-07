## 1. **Score**

**6/10 — not ready。**

E2、E3 已补上关键归因证据，C1 仍有支持。主要实验缺口是 E1：实际实现没有完成上一轮要求的 baseline-conditioned fusion。另有两处结果表述需要修正。

## 2. **Round-1 points**

1. **partly settled**：E1 确实学习了全部答案，但 [e1_fusion.py:196](/home/jehc223/Retrieval-hate/experiments/20261108_astra_review/e1_fusion.py:196) 只训练独立 answer head，baseline 在训练后加入；DeHate ROC 的例外已正确披露。
2. **settled**：`hard2/hard2q` 保持同一缓存、prompt 和训练流程；原始结果中 HCS 的 `hard2` AP/ROC 降幅为 **.015704/.050465**。复合臂的配对分析也支持现在披露的 HCS answer-training 例外。
3. **settled**：核查 replay 后，同 checkpoint、逐视频调用数匹配成立；[HCS 原始结果](/home/jehc223/Retrieval-hate/runs/20261108_astra_review/e3/hateclipseg/summary.json) 的 L4 AP/ROC 差值为 **+.006473/−.002428**，例外已如实承认。
4. **partly settled**：E4 的相邻阈值随机化正确，validation 期望调用达到 B；但“指标变化均 ≤.002”与[原始结果](/home/jehc223/Retrieval-hate/runs/20261108_astra_review/e4/val_selected/hatemm/summary.json) 不符。
5. **settled**：新版 C2 接受局部回落，与[原始曲线](/home/jehc223/Retrieval-hate/runs/20261101_r6_length_table/final_curves/val_selected/hateclipseg/summary.json) 一致；不需要再用实验追求逐点单调。

## 3. **Weaknesses**

**1. E1 仍未检验上一轮指定的 baseline 与答案联合学习。**

**(a)** [Head.forward:83](/home/jehc223/Retrieval-hate/experiments/20261108_astra_review/e1_fusion.py:83) 明确不使用 baseline logit `z`；训练 loss 只作用于 `h`。随后 [validation:207](/home/jehc223/Retrieval-hate/experiments/20261108_astra_review/e1_fusion.py:207) 才选择 `alpha`，组成 `alpha*z+h`。`feat` 变体输入的是 r6 的原始 I3D/VGGish/BERT features，也不是 frozen baseline 的 learned features。

因此，这是 **独立训练的 answer head 加 scalar late fusion**。它补上了“全部答案参与学习”，但没有补上 baseline-conditioned fusion。[HCS DSANet 的逐 seed 文件](/home/jehc223/Retrieval-hate/runs/20261108_astra_review/e1/hateclipseg/dsanet/seed_234.json) 中，三个 seed、两个变体的 `alpha` 全为零，最终预测实际上舍弃了 baseline。

内部 review 根据 baseline 的 train/test AP 差异推断联合 head 必然过度依赖 baseline；这不是该 fusion 实验的实测结果，不能据此认为原实验已经完成。

**(b)** 损害 C4 的机制归因。现有数字支持对这两种指定 late-fusion 实现的比较。

**(c)** **EXPERIMENT**。

**(d)** 最小补实验：冻结官方 baseline，从其最终 classification layer 前提取逐时刻 hidden features，按现有时间网格对齐；将这些 features 与现有全部答案 token 的聚合表示共同送入轻量 MIL head，**对最终融合输出计算视频级 loss**。沿用当前节点抽样、两个 learning rates、30 epochs 和 validation 选取规则；validation/test 使用相同 r6 query prefixes。运行：

- HateMM：MACIL-SD；
- HCS：Fed-WSVAD、DSANet 官方分支；
- DeHate：Fed-WSVAD；
- 每项 seed **234、2025、3407**。

报告逐 seed 配对差值。若 AP 优势在任一 corpus 消失，应进一步收窄 C4；若三语料 AP、HateMM/HCS ROC 优势保留，则该缺口得到支持性证据。无需要求 DeHate ROC 获胜。新增 VLM 调用 **0**，baseline 不重训。

**2. C5 的“.002 内变化”被原始结果直接否定。**

**(a)** [E4 HateMM 原始结果](/home/jehc223/Retrieval-hate/runs/20261108_astra_review/e4/val_selected/hatemm/summary.json)，`seed3407_trial12`、B=7：

- validation 调用从 **4.715596** 调整到期望 **7**；
- AP：**.642085 → .647208**，变化 **+.005123**；
- ROC：**.859131 → .866590**，变化 **+.007459**；
- 三 seed 平均 ROC 变化仍为 **+.002486**。

这不是随机化实现错误；[代码:76](/home/jehc223/Retrieval-hate/experiments/20261108_astra_review/e4_budget_audit.py:76) 正确地按视频选择一个 stopping threshold，没有混合预测。但五次 random draws 的均值也不能充当统一误差上界。

**(b)** 损害 C5 的数值准确性，不否定其 validation 期望预算性质。

**(c)** **WRITING**。删除统一的 `.002` 上界，报告实际差值及 Monte Carlo variability；明确 exact expected validation budget 属于 randomized variant，原 deterministic r6 仍可能 underspend。已有结果足够，无需追加实验。

**3. 旧 HCS fixed-fusion 行仍用了错误的 DSANet 分支。**

**(a)** [baseline_plus_vlm.py:39](/home/jehc223/Retrieval-hate/experiments/20261101_r5_completion/baseline_plus_vlm.py:39) 读取 `score_align`，而[官方表配置:19](/home/jehc223/Retrieval-hate/scripts/reproduction_baselines/aggregate_official_val.py:19) 指定 `score_mlp`。新 E1 已修正，旧 fixed-fusion 原始行仍是 **AP=.599402、ROC=.567837**。

我用官方分支、同三 seed、同 query prefixes、原 alpha grid 和共享 evaluator 只读复算：最佳 alpha 为 **.2**，AP/ROC 为 **.597099/.582805**，仍低于 r6 的 **.691467/.679673**。因此，修正不改变该比较的结论方向。

**(b)** 影响 C4 中“官方 baseline 的 fixed fusion”的准确标识。

**(c)** **WRITING**。更正该行，或明确旧行使用 `score_align`；本轮复算已解决其是否改变结论的问题，不需要追加训练实验。

## 4. **Experiment list**

1. **E1 补做：frozen baseline hidden features 与全部答案共同训练的 MIL fusion head。** 上述四个 baseline–corpus 配对，各 seed 234/2025/3407；保持相同 test queries，报告配对 AP/ROC 差值。新增 VLM 调用为 0。