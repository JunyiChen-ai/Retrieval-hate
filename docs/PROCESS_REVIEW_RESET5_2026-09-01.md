# Process Review RESET5 — 2026-09-01

独立只读 process review；未审代码、未提出candidate、未修改实验。依据`research-wiki/STATUS.md`、`RESEARCH_ITERATION_RULES.md`、failure ledger，以及RESET4三次正式方法的README与权威summary。

## 裁定

**RESET**。整体研究继续，但必须先修正starting-point drift、failure-target churn与过重的validation搜索。累计连续performance failure保持`8`；落实后新process-review窗口=`0/3`。

## 诊断

RESET4三次方法并非简单换名，但formal candidate同时改变过多基础条件。Temporal Residual control相对seed-234 MultiHateLoc within在HMM/HCS为`-.0245/+.0133`；Local-Quotient control为`-.0739/-.0329`，严重starting-point drift；Sparse-Scan control仅`-.0032/+.0113`，是三者中唯一较干净的机制反证。Local-Quotient在HCS虽相对弱control within`+.0223`，仍低于anchor且pooled崩溃，不能把整体下降精确归因于GRL。

RESET4锁定的是MultiHateLoc modality-selection/fusion failure，但Sparse-Scan转向occupancy、Local-Quotient转向identity/position，形成failure-target churn。Policy-Simplex novelty STOP也说明候选生成开始由跨任务术语驱动，而不是固定failure驱动。本epoch没有premise churn，premise不是瓶颈。

## 已落实的流程修正

1. 下一epoch只允许攻击`runs/20260831_multihateloc_test_error_analysis/main/metrics.json`记录的MultiHateLoc modality-selection/fusion failure；不得中途切换occupancy、position、teacher、producer、raw statistic或generic regularization。
2. 强制anchor-compatible first：保留原MultiHateLoc forward、raw fused test score、基础loss与training schedule；只加一个result-relevant机制；机制关闭时必须退化为原MultiHateLoc。首轮禁止同时重写representation、readout、MIL functional与optimizer schedule。
3. 每轮正式报告同一harness/selection policy下的MultiHateLoc control及core，并报告control相对官方seed-234 anchor的test偏差。明显漂移记scaffold failure，不把下降归因于机制。
4. Validation默认每语料6 trials：两个learning rate×三个机制强度；最多两个新增result-affecting超参数，确有第二个独立超参数时上限8 trials。只有双test共同正向并进入Rule18唯一corrective时才增加4–6个定向trial。
5. Validation selection必须在候选brief中预先固定：within为主，同时对validation pooled AP/ROC设置明确非劣化约束；它只在已定义方法内部选择超参数和checkpoint，不用于生成或更换机制。
6. 固定执行顺序：failure→novelty三门→anchor-compatible实现→一次technical review→每语料6–8 validation trials→HMM/HCS test三指标→test error analysis后关闭或唯一corrective。

## 方向裁定

- CONTINUE：整体研究；MultiHateLoc modality-selection/fusion failure。
- PAUSE：需要整体重写backbone、forward、基础loss或training schedule的候选；新teacher、producer、premise及无关跨任务adaptation。
- STOP：Temporal Residual、Sparse-Mixture Scan、Local-Quotient Adversarial、Policy-Simplex当前版本、marked-splat/duration-field与ledger全部已关闭变体。

