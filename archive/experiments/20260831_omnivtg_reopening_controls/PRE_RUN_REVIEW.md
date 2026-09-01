# Technical pre-run review

截至 2026-08-31。独立 technical reviewer：`metrics_audit`。

Verdict：`FAIL — DO NOT RUN FORMAL`。

## 决定性 protocol failure

不读取frame/span GT的structural precheck确认：HMM frozen cohort中的
`hate_video_329` media duration为`7.105011s`，固定8秒block rotation只有一个
block；其旧raw prediction为`parse_ok=true`、interval `[2,6]`。当前冻结protocol
明确规定`N<2` fail closed，且任何raw-success→corrupted-failure使integrity gate失败。
因此joint reopening gate在任何新模型输出前已经不可能PASS，继续GPU推理没有信息价值。

## 可执行 bug

HMM `hate_video_184` 的score grid/media duration为`46/45.0`，`hate_video_89`为
`157/155.989`。若corrupted prediction成功，当前`inverse_mapped_interval_score`
会在最后grid秒因`second >= duration`抛错，使`produce_controls.py`整体崩溃，而不是
产生可审计的fail-closed artifact。未来若复用block corruption基础设施，必须先冻结
grid/media边界映射规则；本轮因上述protocol failure无需修复后运行。

## 其余核验

- 32秒synthetic 8秒half-rotation/inverse mapping PASS；
- raw/corrupted schema、exact cohort及raw-success→failure gate实现正确；
- OCR upstream覆盖HMM `85/85`、HCS `69/69`，OCR feature length/finite通过；
- static threshold只读取同语料positive-train scoped labels；八个strata gate与README一致；
- GT只由`evaluate.py`读取；指标只调用canonical `evaluate_scores`与
  `frame_eval_common` primitives；
- 未使用任何hash/checksum/digest。

另：ffmpeg CRF18重编码不能字面声称frame multiset原样保留，但不影响当前STOP裁定。
未启动正式producer、GPU或test evaluation，无run artifact。
