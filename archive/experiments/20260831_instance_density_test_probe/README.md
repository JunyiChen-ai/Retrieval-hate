# Same-corpus instance-density test probe

**截至 2026-08-31；状态：TEST FAIL，前提淘汰。淘汰原因：没有任何同一 channel 在
HMM/HCS test 同时提供有效局部排序，且所有分支均未同时达到三项 SOTA。**

## Test 结果

权威输出：
`runs/20260831_instance_density_test_probe/{hatemm,hateclipseg}/metrics.json`。

- HMM POWA within `.58342`；concat probe `.60843`，但 pooled AP/ROC 只有
  `.57912/.78048`，且 within 仍低于 SOTA `.63153`。
- HCS POWA within `.51597`；最好 visual probe仅 `.51816`，concat `.49753`；所有 pooled
  指标也远低于 SOTA。
- transport 只作上限，HMM concat within `.60729`，HCS `.49754`，不改变结论。

不得按 corpus 选择不同 channel。结论：普通同语料 bag-label instance density不能解决 HCS
局部可识别性，也不足以支撑跨语料 PU/MIL instance-risk 候选；不进入 novelty check 或训练。

Rule-10 iterative/developmental diagnostic。每个 corpus 只用自身 train video labels，把负视频
snippet 视为 reliable benign、正视频 snippet 视为 noisy positive mixture，训练固定线性
audio/visual/text/concat density probe；随后立即在 test 上用唯一共享 evaluator 报 pooled AP、
pooled ROC、within-video ROC。

同时报告 POWA multiset 按 probe order 的 tie-neutral transport 上限，但 transport 是 calibration，
绝不作为候选方法。teacher tie 内保持 POWA anchor 原排序，全 tie 必须严格返回 anchor。

这个 probe 本身不具 novelty。它只决定下一候选是否值得采用同语料 PU/MIL instance-risk 学习；
不得按 corpus 选择不同模态。只有同一个 channel 在 HMM/HCS test 同时显著改善 within 且保持
pooled 指标，才进入独立 novelty check。
