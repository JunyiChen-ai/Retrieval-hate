# Active-speaker focused test error analysis

截至日期：2026-09-01。依据代码：`experiments/20260901_active_speaker_test_error_analysis/analyze.py`；权威结果：`runs/20260901_active_speaker_test_error_analysis/main/metrics.json`。

本轮不是新方法、premise或训练运行，而是 RESET7 process review 要求的唯一一次聚焦 post-test error analysis。它复用 active-speaker 正式运行已经产生的 HMM/HCS test `anchor`、`permuted`、`core` predictions、冻结 test ground truth 与无标签 active-speaker cache，量化 multi-face eligible 覆盖、机制实际改变输出的范围，以及 eligible subgroup 上的 core-minus-permuted 效果。

运行：

```bash
/home/jehc223/miniconda3/envs/HateVideo/bin/python experiments/20260901_active_speaker_test_error_analysis/analyze.py
```

Test predictions 与 GT 明确用于 iterative/developmental error analysis；未用于梯度训练、validation 超参数选择或 checkpoint selection。

## 结论

Multi-face eligible 秒仅占 HMM `681/29269=2.33%`、HCS `900/18839=4.78%`。在含 eligible 秒且同时含两类 GT 的视频上，core 相对 permuted 的 within ROC 为 HMM `-0.001277`（改善/持平/恶化 `13/2/15`）与 HCS `+0.000249`（`18/0/19`）。Eligible 秒 pooled AP 的变化为 HMM `+0.010833`、HCS `+0.000169`，但 pooled ROC 同时分别下降 `-0.000953/-0.000879`。因此正确 active-speaker assignment 的覆盖过低，且在真正受机制定义约束的 subgroup 上仍无跨语料一致排序收益；该 family 保持关闭。
