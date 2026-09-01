# Novelty review B

截至 2026-08-31。只读独立审查；裁定 `STOP`，novelty `4.6/10`。

Gate 1 PASS；Gate 2 窄 PASS，未检出完整的 ItS2CLR + deletion/abstain + signed prototype energy 合取进入 hateful-video detection/localization，但 prototype-MIL/metric readout 邻域拥挤；Gate 3 FAIL。

Direct energy 确实把 instance loss 接入最终 scalar score 的计算图，但“代数上不可旁路”不等于获得新的定位识别性。旧 carrier relation 已经正式失败；把 auxiliary SupCon 换成标准 prototype classifier，再以 smooth-max 聚合，只能作为 implementation-fix control，不构成新方法。

最强反例是让一个全 abstain 的 modality 在 positive video 全时刻输出恒定高 energy，其他 modality 在较低能量区间拟合 carrier/background。smooth-max 始终由前者主导，bag BCE 和 instance loss都可低，within-video ROC 仍为 `.5`。现有 corpus-level coverage gate排除不了此解。未归一化 logsumexp 还会把 available modality 数量变成 score shortcut。

正式 pilot 没有保留逐行 OOF artifact，所以“复用旧 frozen cache”在当前工作树不可执行。Smoke cache不能代替正式 artifact。

若只作为工程 baseline，最低要求是重新生成并保存完整 train-only OOF rows、改为 log-mean-exp、对所有可能成为 winner 的 modality 棠查 same-video/same-modality carrier-background pair，并加入相同 pseudo labels 的 normalized-linear direct-head control和 winner responsibility audit。这些修复仍不足以恢复 novelty；新候选必须加入能解析排除 abstaining-modality dominance 的任务特定局部观察或 final-score constraint。

近邻来源：ItS2CLR（CVPR 2023）、TPMIL（MIDL 2024）及 prototype-based MIL 文献。
