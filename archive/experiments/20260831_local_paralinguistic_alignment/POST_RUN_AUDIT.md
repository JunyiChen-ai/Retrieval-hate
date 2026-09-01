# Independent post-run result-chain audit

日期：2026-08-31

## Verdict

**Integrity PASS；premise FAIL。** 正式结果链完整、可复算，冻结联合决策 `STOP_DIRECTION` 正确。

权威目录：`runs/20260831_local_paralinguistic_alignment/premise/`。

## Completion and provenance

- `run.pid` 记录的进程已经结束；未发现仍在运行的本实验 producer/evaluator。
- `run.log` 完整包含两个 corpus 的 producer 汇总和 evaluator 最终行，无 traceback。`run.sh` 使用 `set -euo pipefail`，`metrics.json` 是最后一个 evaluator 成功完成后才写出的产物。
- `config.json` 与冻结 README 一致：两个 corpus 独立、`dense4fps_w2vemo`、固定 4 fps→1 fps sampling、固定 weighted scaler/logistic recipe、无 validation 或参数选择。
- 当前 producer 的 scoped loader 只为 frozen train IDs 解析 video-level labels；HateMM/HateClipSeg train-test ID 交集均为空。HateClipSeg 使用官方 `video_level_annotation.csv`，没有读取 segment labels 或 timestamps。Test score 生成前只使用 GT archive key names 确定 exact cohort，没有打开 label arrays。
- `producer_report.json` 与输入复核一致：HateMM 744 个 train videos、113110 train seconds、class counts 446/298，两类 weight sums 均为 372；HateClipSeg 251 个 train videos、59759 train seconds、class counts 32/219，两类 weight sums均为 125.5。两个 fit 分别在 172/109 iterations 结束，未触及 500 上限；train/test feature 中均无全零视频。

## Scores and evaluator recomputation

- `hatemm_scores.npz` exact 覆盖冻结 GT 214/214 videos、29269 frames；`hateclipseg_scores.npz` exact 覆盖 79/79 videos、18839 frames。所有数组均为一维、逐视频长度与 GT 完全一致且 finite，无 missing/extra IDs。
- 冻结 label-blind 1 fps grid 与 score 长度逐视频一致。报告的 tail padding 可复核为 HateMM 52 秒、HateClipSeg 11 秒；container/audio-grid mismatch 计数分别为 9 和 3，且 container duration 没有参与 score length。
- 用保存的 scaler mean/scale 与 logistic coefficients 逐帧重建正式 scores，最大绝对差为 HateMM `1.17e-6`、HateClipSeg `9.90e-7`，符合数值运算精度，确认 scores 来自对应模型产物。
- 直接重新调用唯一共享 `evaluate_scores`，两个 corpus 的 original 指标与 `metrics.json` 完全一致：

| Corpus | pooled AP | pooled ROC | within-video ROC |
|---|---:|---:|---:|
| HateMM | 0.5079137661 | 0.7762374687 | 0.5424723948 |
| HateClipSeg | 0.5344248825 | 0.4954902532 | 0.5122357684 |

## Circular-shift and gate

- 32 个联合 circular-shift controls 均由 `np.roll` 产生，保持各视频 score multiset。其 pooled AP、pooled ROC 与 within-video ROC 逐项重算均与正式文件一致。
- Gate 使用预注册的 equal-video/equal-unique-shift aggregation。独立重算覆盖 HateMM 85 个 both-class videos、2564 个 video-shift evaluations，shift mean `0.5008873945`；HateClipSeg 67 个 videos、2144 evaluations，shift mean `0.5002005806`。
- HateMM：original within `0.5424723948 > .52`，相对 shift gain `+0.0415850003 >= .020`，coverage PASS，因此 corpus gate PASS。
- HateClipSeg：original within `0.5122357684 <= .52`，相对 shift gain `+0.0120351878 < .020`，因此 corpus gate FAIL，尽管 coverage PASS。
- 两 corpus 必须同时通过。正式 `joint_pass=false` 与 `decision=STOP_DIRECTION` 严格符合冻结 gate。

## Usable conclusion

当前固定协议只支持：该 train-video-label paralinguistic readout 在 HateMM 有内部时间对齐信号，但没有在 HateClipSeg 复现。双语料 premise 失败，按预注册规则关闭本方向；不能将 HMM 单语料结果表述为通过，也不能将本 diagnostic 表述为方法或 SOTA。

