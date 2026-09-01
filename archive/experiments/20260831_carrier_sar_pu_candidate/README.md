# STOP — Carrier-SAR-PU temporal MIL

截至 2026-08-31。未实现、未训练。

候选拟适配 Selected-at-Random positive-unlabeled learning：同语料负视频帧作为 certified negatives；正视频中固定选择 speech-present 且 lexical score 位于视频 top-15% 帧作为 observed positives；carrier covariate 预测选择 propensity，单一学生预测 frame hate，test 只输出学生 raw score。

机制假设是 verbal hate 更容易被 lexical selector 观测，而 silent/visual hate 是 selection-biased missing positives；propensity correction 应避免把未被选中的 visual hate 当成负例。

## Test-first premise

只读使用既有同语料 lexical test predictions 与 test GT。每个同时含两类秒的正例视频固定选择至多 `ceil(.15*T)` 个 speech-present frames，按 lexical score 取最高；先逐视频计算 selected precision 与视频 positive base rate，再 macro 平均。

权威输出：`runs/20260831_carrier_sar_pu_premise/metrics.json`。

HateMM selected precision `.703735`，全 cohort base `.607315`，在 84 个 selection-defined videos 上的 paired 差 `+.094923`；HateClipSeg selected precision `.575347`，base `.586666`，paired 差 `-.011319`。cutoff ties 较多，但 tie-neutral expectation 与 include-all-cutoff-ties 下 HCS 差仍约 `-.021699/-.017908`，停止结论稳健。HCS 的 observed-positive selector 没有 positive enrichment，统一入口假设失败。

## Novelty verdict

两路独立审查均 STOP，`4.0/10` 与 `4.7/10`。SAR-PU 来源未检出已用于 hateful-video temporal localization，但 adaptation 的关键假设不成立：lexical top-q 是含 false positive 的 deterministic bag-relative pseudo-label，不满足 `s=1 => y=1`；speech=0 的 silent/visual positives 选择概率为零，违反 overlap；`f(x)g(c)` 在 bag label 下仍不可识别，whole-video broadcast 仍可满足 loss。禁止调 q 或按语料 routing 挽救。
