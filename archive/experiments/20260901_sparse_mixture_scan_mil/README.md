# Sparse-Mixture Scan MIL

截至 2026-09-01。RESET4 candidate 3。无 premise；现有四语料 test error evidence 直接支持 failure。

## Failure

权威 artifact 为 `runs/20260831_multihateloc_test_error_analysis/main/metrics.json`。MultiHateLoc 固定读取 top `1/K` 秒，但真实 positive occupancy 跨视频剧烈变化：HMM/EN/ZH/HCS 的 median absolute mismatch 分别为 `.362/.533/.332/.307`。在 HMM，positive fraction `<=1/3` 的视频 fused within 仅 `.542`，中/高 occupancy 为`.663/.668`；HCS 对应为`.481/.495/.563`。HMM/HCS 的 within 与真实positive fraction均显著正相关（`.255/.387`），说明固定 witness count 对稀疏 hateful moments 系统性过聚合，而不是单一模态融合问题。

## Cross-task source

来源为 sparse heterogeneous mixture detection 的 Higher Criticism / Berk-Jones scan。它把一组局部检验的有序 p-values 与 global-null 下的期望数量比较，并在多个可能 sparsity levels 上扫描；因此不需要预先知道 non-null proportion。Donoho & Jin 证明 Higher Criticism 对未知稀疏度自适应；Walther 的 average likelihood ratio / Berk-Jones 系列兼顾 very sparse 与 moderately sparse alternatives。

来源：[Donoho & Jin, Annals of Statistics 2004](https://doi.org/10.1214/009053604000000265)；[Walther, 2013](https://arxiv.org/abs/1111.0328)。当前检索未发现 Higher-Criticism/Berk-Jones sparse-mixture scan 已用于 hateful-video detection/localization。

## Non-trivial task adaptation

保留 MultiHateLoc local encoders、原fixed top-K bag BCE与唯一 raw `score_fused`，避免丢失其跨视频判别。新增的local scan只读取每个video内去均值后的frame logits；negative train bags的同样去均值logits定义benign local-variation null。每个positive bag的centered logits转换为相对该null的one-sided tail probabilities。对候选rank`k`计算observed top-tail fraction`k/T`与第`k`个ordered tail probability之间的one-sided binomial KL（Berk-Jones evidence），再对从very sparse到moderately sparse的固定rank grid做temperature-controlled weighted log-average likelihood ratio。为防止长视频仅因包含更多检验而获得更大training signal，最终使用`log-ALR / T`作为per-frame evidence rate；这是本任务视频时长可变所需的明确adaptation。Positive bags要求scan evidence rate高于negative bags，并继续接受原dense benign BCE。Inference不做scan、threshold、CDF、calibration或post-processing，只输出训练后网络的raw per-frame `score_fused`。

关键task delta不是“换pooling”：negative hateful-video bags给出同语料benign local-variation null；positive video 的未知hateful occupancy被表述为相对该null的sparse-mixture alternative；Berk-Jones rank scan把video label的局部梯度自动分配到最有证据的occupancy scale。去均值使whole-video broadcast offset无法满足scan，原top-K bag BCE单独保留pooled discrimination。固定top-K只能在一个预设occupancy上产生witness gradient，而scan在同一bag内比较多个candidate occupancies并包含binomial multiplicity penalty，避免永远选择最小`k`的max shortcut。每个主数据集独立估计train null；validation只选训练超参数和checkpoint；test label不进训练或选择。

## Falsification and matched control

Matched control使用完全相同的MultiHateLoc architecture、原fixed top-K loss、dense-negative loss、optimizer budget和validation search，但没有centered negative-null sparse-mixture scan；core唯一差异是该occupancy-adaptive local scan loss。Inference两者都只评raw fused frame score。

HMM/HCS test上core within必须都胜matched control与seed-234 MultiHateLoc anchor，至少一边`>=+.020`；post-test occupancy strata中低occupancy改善必须大于高occupancy改善，否则机制失败。最终晋级仍要求两语料pooled AP、pooled ROC、within ROC全部SOTA。方法含scan temperature、scan weight、margin、null EMA与rank-grid上限等超参数，必须先在每个语料做12个validation-only trials联合选择配置和checkpoint；core直接加载选中trial checkpoint做test，不重新训练，control在同配置下独立用validation选checkpoint后test。

## Novelty gates

独立 verdict：`GO 6.9/10`。Gate 1 PASS：Higher Criticism/Berk-Jones/ALR可adapt。Gate 2 PASS：未检出ordered-tail binomial-KL sparse-mixture scan用于hateful-video detection/localization；Auto-pool、Sparse Temporal Pooling与AdaScan只占用宽泛adaptive pooling claim。Gate 3 PASS：negative-train benign null、ordered tail probabilities、binomial count likelihood与raw-score inference形成non-trivial task adaptation，不与duration、ensemble/calibration或旧interval scan严格同构。视频帧有时间相关性，因此不主张继承经典independent-p-value显著性或最优检测边界。

唯一 formal technical review 在修正per-frame evidence-rate定义、独立scan temperature、12-trial fail-closed选择、selected-checkpoint直接test与occupancy-strata gate后为`PASS`。

## 正式结果与去向

HateMM与HateClipSeg各完成12个validation-only trial，并分别按validation within-video ROC联合选择超参数和checkpoint。选择记录为`runs/20260901_sparse_mixture_scan_mil/val_search/{hatemm,hateclipseg}/selection.json`；锁定后立即在test评测，权威汇总为`runs/20260901_sparse_mixture_scan_mil/formal_val_selected_seed234/summary.json`。

HateMM matched control/core 的 pooled AP、pooled ROC、within ROC 分别为`.485738/.728018/.625220`与`.483127/.720623/.627010`；core within仅`+.001791`，相对MultiHateLoc anchor为`-.001446`。HateClipSeg control/core为`.566257/.529873/.534994`与`.520455/.493017/.526921`；core within为`-.008073`，相对anchor仅`+.003220`。两语料core都未通过三指标SOTA门。

目标机制也失败：low-occupancy视频的core-minus-control within在HateMM/HateClipSeg分别为`-.005186/-.009139`，并未优于high-occupancy的`+.003791/+.009328`。因此scan没有修复它声称针对的fixed-top-K稀疏occupancy失败，且不存在跨两语料一致的post-test corrective依据。按Rule 18不使用唯一修补轮，不调scan weight、temperature、rank grid、null或margin；该family关闭并归档。
