## 1. **Score**

**7/10 — almost。**

当前限定后的 claims 已有实证支持。必要对照已补齐；未发现 E1b 中影响结论的实现错误或 test leakage。

## 2. **Round-2 points**

1. **settled**：核查 hidden features、训练集标准化、[联合 head 与 loss](/home/jehc223/Retrieval-hate/experiments/20261108_astra_review/e1_fusion.py:304) 及 24 份原始评测文件，汇总完全一致。E1b、E1b_val 各配对的三 seed 平均 AP 差距最小分别为 **+.089565、+.095242**。
2. **settled**：[E4 原始结果](/home/jehc223/Retrieval-hate/runs/20261108_astra_review/e4/val_selected/hatemm/summary.json) 的最大 AP/ROC 变化确为 **+.005123/+.007459**；文档已纠正，并区分 randomized variant 与 deterministic 方法。
3. **settled**：[DSANet 重算结果](/home/jehc223/Retrieval-hate/runs/20261108_astra_review/fixed_fusion_mlp/hateclipseg_r6_mlp_k8.json) 正确：**α=.3，AP=.594649、ROC=.571719**。上一轮错误在我：通配路径误取 seed2025 的 trial16，正确应为 trial18；明确指定路径后复现你们的数字，撤回原来的 .597/.583。

## 3. **Weaknesses**

**1. E1b 失败原因仍被写成了已经证实的机制。**

**(a)** [README:108](/home/jehc223/Retrieval-hate/experiments/20261108_astra_review/README.md:108) 将联合 head 较差、较早选中 checkpoint 解读为已经测得“过度依赖 baseline”。这些结果没有直接识别依赖程度。`hid_val` 同时改变训练样本、样本量及 checkpoint 选择程序，也没有隔离该原因；其 AP 在 HateMM、DeHate 上反而从 **.612→.603、.188→.183**。

**(b)** 影响 C4 的机制解释，**不否定当前“优于所有已试 fusion”的数值主张**。

**(c)** **WRITING**。将“已经证实过度依赖”改为“结果与该解释相容，但原因未被隔离”。保留 C4 的“under every fusion we tried”限定。当前 claims 不需要为此追加实验。

## 4. **Experiment list**

NO_MORE_EXPERIMENTS