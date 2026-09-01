# Utterance-boundary premise

截至 2026-08-31。该目录只做 developmental test error analysis，不训练模型、
不选择 checkpoint，也不生成候选方法 prediction。

## 问题

检验一个此前未验证的结构前提：hateful span 的进入/退出边界是否跨 HateMM 与
HateClipSeg 都更靠近 Whisper utterance/chunk 边界。若成立，后续才考虑把来自
speech/NLP 的 segmental latent-structure 方法非平凡地适配到弱监督 hateful temporal
localization；若不成立，立即关闭 utterance-unit / segmental-MIL 方向。

## 协议

- 只分析 test predictions/GT 允许范围内的结构证据；test GT 不进入梯度、模型拟合或
  checkpoint selection。
- 仅保留同时含正负秒的正例视频，使用已有 Whisper-large-v3 chunk timestamps。
- 原缓存混有 Whisper word/chunk timestamps。为避免把词边界误当 utterance，全部记录先经
  同一 deterministic grouping：相邻 entry 仅在前一文本没有句末标点且 silence gap
  `<0.8s` 时合并；否则切分。该规则在看结果前冻结，不按语料调整。
  Whisper 的 finite zero-duration point token 仍参与 grouping，以保留独立输出的句末标点；
  只丢弃反向时间区间或完全落在 `[0,T]` 外的 entry。
- 对每个视频计算 GT transition 到最近非空 ASR utterance 内部边界的普通线性时间距离。
- control 将同一视频的整组 ASR 边界作 30 个非零 modulo shifts，保持边界数量、相对间距
  和视频长度；旋转后仍计算普通线性距离，不把视频首尾视为相邻。先在每个视频内平均，
  再跨视频 macro 平均，避免长视频支配结果。
- GT 的 1fps 长度定义分析区间 `[0,T)`；ASR 与 GT 都从视频起点计时，超出该区间的 ASR
  timestamp 丢弃，并显式报告 ASR metadata duration 与 `T` 的差值。
- 固定 premise gate：两语料都要求 observed mean distance 小于 shift mean，且两语料
  `recall@2s` 相对 shift 至少 `+0.05`。任一失败即 `STOP_DIRECTION`，不做参数扫描。

## 输入与输出

- GT 由共享数据层 `scripts/reproduction_baselines/hate_common/data.py` 读取。
- ASR：`data/ASR/HateMM/test_seen_asrK4_whisper-large-v3.jsonl` 与
  `data/ASR/HateClipSeg/test_seen_asrK4_whisper-large-v3.jsonl`。
- 权威输出：`runs/20260831_utterance_boundary_premise/main/analysis.json`。

## 结论

固定 gate 结论为 `STOP_DIRECTION`，独立结果链审计见 `POST_RUN_AUDIT.md`，结论
`PASS`。

- HateMM：85/85 eligible videos 有效；observed/shift mean nearest distance 为
  `20.4583/22.2579s`，`recall@2s` 为 `.53829/.34521`，gain `+.19308`，两门通过。
- HateClipSeg：67/67 有效；observed/shift distance 为 `31.8987/25.5979s`，方向反转；
  `recall@2s` 为 `.23771/.23437`，gain 仅 `+.00335`，两门失败。

因此 utterance boundary 不是跨语料共同的 hateful-span 结构约束，不进入 segmental
latent-structure novelty review，不训练 utterance-unit MIL，不调 grouping threshold，
也不按语料 routing。
