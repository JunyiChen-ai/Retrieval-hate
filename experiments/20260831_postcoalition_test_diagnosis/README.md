# Post-coalition test diagnosis

截至 2026-08-31。此轮按 test-first 规则读取 HateMM 与 HateClipSeg 的 test predictions 和
test GT，诊断 coalition witness 失败后下一机制应解决的问题。test label 不参与梯度或
checkpoint selection；所有输出属于 iterative/developmental evidence。

固定 structured 输入在两个 corpus 统一使用上一轮的 `mobius_nonminimal`。它是 prior test
pilot 中两语料 within-video ROC 都最高的 coalition structured control；这个 source selection
已经看过 test，因而只允许用于本轮 error analysis，不能成为按语料/视频选择 branch 的方法。
另一个固定输入是各自独立训练的 POWA pooled anchor。
分析只测量：预测相对 GT 的过度碎片化、固定窗口 temporal smoothing 的诊断上限，以及两个
现有模型排序的互补性。smoothing、test occupancy mask、best-of-two oracle 和 rank mean 都
不是候选方法，不允许直接晋级或部署。

Rank mean 对相同分数使用 average rank，不以时间索引拆 tie。GT-occupancy fragmentation 使用
cutoff-score 的 tie-inclusive superlevel set；若 cutoff 落在 plateau，会记录实际选中比例和
扩张秒数，而不是任意挑选 plateau 中较早或较晚的秒。

运行：

```bash
/home/jehc223/miniconda3/bin/conda run -n HateVideo \
  python experiments/20260831_postcoalition_test_diagnosis/analyze.py
```

权威输出：`runs/20260831_postcoalition_test_diagnosis/main/metrics.json`。

## 结果与设计影响

两个语料均 exact-cover test。HateMM/HateClipSeg 的 GT 平均 transition count 分别为
`4.05/6.96`，structured score 在 test-GT occupancy superlevel 下分别为 `20.48/38.93`，
中位 transition inflation 为 `4.5×/5.5×`：跨语料共同错误是高分时间点过度碎片化。

但固定 smoothing 不构成共同修复。HateMM 7 秒窗口把 AP/ROC/within 从
`.5346/.7516/.6338` 提到 `.5487/.7655/.6583`；HateClipSeg 同一窗口却从
`.5643/.5509/.5365` 降到 `.5548/.5421/.5221`。因此下一轮不得做 generic smoothing、CRF
或固定 duration prior。POWA 与 structured score 的 rank mean 胜过两者的比例也只有
HateMM `.212`、HateClipSeg `.075`，不支持 ensemble/routing 路线。

受此 test error analysis 影响，下一步只检验非局部 semantic recurrence 的 multiplicity bias，
且先设双语料 premise gate；不把任何诊断 readout当作方法。
