# Novelty review — negative-reference unmatched-mass witness

截至 2026-08-31。依据独立文献查新与 2026-08-31 起生效的新 novelty 标准。

## 结论

**PASS，可进入 pilot。** 不主张发明 UOT、partial transport、normal prototype、outlier mass 或
MIL pooling。查新的最近工作分别位于 open-set learning、robust OT、PU learning、弱监督视频异常
定位、无监督动作分割和通用 multimodal MIL；未发现 typed negative-reference unmatched-mass
transport 已用于 hateful video detection / localization。

本项目允许有机制的跨任务 adaptation，因此真正 claim 仅为：

> 在只有 video label 的 multimodal hateful temporal localization 中，用只由 negative train videos
> 更新的 typed normal references 解释 time×modality token；不同模态只能匹配同类型 reference，
> 但共享 normal-slot capacity。无法解释的固定 source mass 是唯一 local witness，并由同一 witness
> 产生 bag likelihood 和 frame score。

## 非 trivial 的任务机制

现有 hateful localization MIL 把 positive video label 同时广播到全部秒和全部模态，并以固定 top-K
假设事件占比；MultiHateLoc test error analysis 还显示其 modality ownership 与最佳局部分支严重
错位。本 adaptation 用 negative-only reference 定义“可被正常内容解释”的质量，并让 time×modality
token 竞争共享解释容量：只有无法被正常 reference 解释的质量承担 positive evidence。它直接针对
错误 label broadcast 与 latent modality/time ownership，而不是把相邻任务模块原样接到分类器后面。

## 最近先例与 claim 边界

- POT-OSSL、Outlier-Robust OT、partial-OT PU：占据 transported/unmatched mass 作 outlier 或 PU
  signal；本项目不主张该数学原语新。
- NG-MIL：占据 normal-video prototype 弱监督异常定位；本项目不主张 normal reference 新。
- OT-WSVAD、MG-TVMF、ASOT：占据 OT 在视频异常定位或无监督时序分割中的使用，但不是 hateful
  video detection / localization。
- VALOR/MMIL：占据无 modality label 的 time×modality ownership 问题，但没有本候选的 typed
  negative-reference reject transport。

## 必须证伪的退化

首轮同 checkpoint 强制输出 independent per-modality transport 与 nearest-normal controls。core 若
不在两个主数据集同时超过两者的 within-video ROC，说明共享 transport 没有贡献，立即淘汰。若 core
接近晋级，再补 balanced transport、pooling、无时间约束、模态置换、缺失模态、长度重采样和容量/
atom/temperature sensitivity；不得把 calibration、ensemble、corpus routing 或固定事件比例包装成
novelty。
