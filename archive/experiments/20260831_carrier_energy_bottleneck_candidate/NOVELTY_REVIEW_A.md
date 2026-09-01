# Novelty review A

截至 2026-08-31。只读独立审查；裁定 `STOP`，novelty `4.0/10`。

## 三道硬门

- 允许 adaptation：PASS。
- 来源未占用 hateful-video detection/localization：窄 PASS。未检出 ItS2CLR 或 deletion-carrier direct proxy readout 进入目标任务，但 prototype-based weakly supervised temporal localization 已有邻近先例。
- non-trivial task adaptation：FAIL。

单位归一化后，`e_tm = z_tm^T (p_m+ - p_m-) / tau`，严格等价于无 bias 的 normalized binary linear head。对 carrier/background 做 logistic loss、忽略 abstain，是普通 pseudo-label BCE。它修复 auxiliary head 与 readout 脱离的工程问题，但新增部分只是 direct classifier/head replacement，没有形成新的定位识别性。

## 解析反例

每个 positive video 让一个 modality 全时刻输出 `+C` 并承担 carrier，另一个 modality 全时刻输出 `-C` 并承担 background；跨视频轮换 modality，仍可通过当前全局 coverage 门。instance loss 与 bag loss可低，但 smooth-max 在整段恒定，within-video ROC 为 `.5`。即使同一 modality 内补齐 carrier/background，另一个恒高、全 abstain modality 仍可支配 smooth-max。

Smooth-max 是单模型固定 soft-OR aggregation，不是 ensemble、离散 routing 或 post-hoc calibration；但未归一化 LSE 带来 available-count offset，小温度也会近似 winner-take-all。

## Artifact blocker

正式 HMM/HCS run 当前没有逐行 `core/oof/pseudo.pt`、OOF log 或 diagnostics，只保留 aggregate pseudo-state counts。它们不能恢复 `(video,time,modality,state)`，所以不能执行 proposal 的 coverage premise。若重做，只能按旧固定 producer/config 重新生成并登记为新 supervision，不能声称复用了旧 formal cache。

近邻来源：ItS2CLR（CVPR 2023）、Relational Prototypical Network（AAAI 2020）、Prototypical Networks（NeurIPS 2017）、Proxy Anchor（CVPR 2020）。
