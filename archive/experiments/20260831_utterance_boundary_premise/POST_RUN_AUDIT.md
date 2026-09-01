# Independent post-run audit

日期：2026-08-31

## Verdict

**PASS。** 正式结果链完整且可独立复算，`STOP_DIRECTION` 联合结论正确。

权威产物：`runs/20260831_utterance_boundary_premise/main/analysis.json`。

## Completion and artifact consistency

- `run.pid` 记录的进程已经结束，未发现仍在运行的本实验分析进程。
- `run.log` 无 traceback，包含正式输出路径、`STOP_DIRECTION` verdict 和两个 corpus 的完整 summary；日志数值与 `analysis.json` 一致。
- Artifact 明确记录本轮为 developmental test error analysis：没有训练、梯度、checkpoint selection 或方法 prediction。

## Independent recomputation

独立重新读取冻结 test GT 与对应 ASR，重新执行固定 punctuation + `<0.8s` gap grouping、zero-duration point-token 处理、线性最近距离和 30 个非零 modulo shifts。逐视频统计、macro aggregation 和 gate 均与正式 artifact 一致。

| Corpus | Coverage | GT transitions | Grouped internal boundaries | Source granularity |
|---|---:|---:|---:|---|
| HateMM | 85/85 eligible videos | 344 | 1593 | 33 word / 52 chunk |
| HateClipSeg | 67/67 eligible videos | 466 | 1723 | 22 word / 45 chunk |

- 两个 corpus 均无 missing ASR、audio failure 或 grouping 后无内部边界的视频。
- 所有 per-video 数值均 finite；transition 和 boundary 均处于 GT `[0,T)` 时间网格。
- Artifact 中逐视频记录的 source timestamp granularity、ASR metadata duration 与 ASR−GT duration 差均可由输入复算。HateMM duration 差范围为 `[-1.011, +0.520]` 秒，HateClipSeg 为 `[-0.990, +6.900]` 秒；正式代码按冻结规则使用 GT 区间并裁剪超界 timestamps。
- 30 个 controls 只 modulo 旋转完整 boundary set，旋转后使用普通线性距离；先对每视频 transitions/shifts 平均，再对视频 macro 平均。

## Metrics and fixed gate

| Corpus | Observed mean distance | Shift mean distance | Observed recall@2s | Shift recall@2s | Recall gain | Gate |
|---|---:|---:|---:|---:|---:|---|
| HateMM | 20.458326 | 22.257904 | 0.538293 | 0.345214 | +0.193079 | PASS |
| HateClipSeg | 31.898706 | 25.597905 | 0.237712 | 0.234366 | +0.003346 | FAIL |

- HateMM 同时满足 observed distance 小于 shift mean，以及 recall@2s gain `>= .05`。
- HateClipSeg 的 observed distance 大于 shift mean，且 recall@2s gain 只有 `+0.003346`，两项要求均失败。
- 冻结 gate 要求两个 corpus 同时通过，因此 `joint_pass=false` 与 `STOP_DIRECTION` 严格成立。

## Reliable conclusion

当前固定协议支持 HateMM 中存在 utterance-boundary alignment，但该现象没有在 HateClipSeg 复现。双语料结构前提失败，应按预注册结论关闭 utterance-unit / segmental-MIL 方向；不能以 HateMM 单语料结果声称 premise 通过。

