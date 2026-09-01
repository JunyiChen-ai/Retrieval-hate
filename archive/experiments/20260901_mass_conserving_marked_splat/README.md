# Mass-Conserving Marked Temporal Splat

截至 2026-09-01。RESET3 后对已成立 duration-field 方向的一次正式方法内迭代，不是新 candidate，不运行 premise，不重做 novelty review。

## Evidence and single change

`runs/20260901_marked_splat_test_error_analysis/main/metrics.json`显示首版 splat 的 leave-one-video-out position-only within 已达 HMM/HCS `.709890/.530324`，raw 只高`.017819/.003757`；同时 HMM 正帧对负视频帧 separation 与 anchor 几乎相同。首版 renderer 使用全局归一 Gaussian 后直接截断视频边界，导致靠近边界的 center 丢失部分 kernel mass，同一 amplitude 在不同位置、不同长度视频中不再可比。

本轮唯一 result-relevant change：对每个 source center、每个 duration scale，只在该视频有效 target 秒上重新归一 kernel，使其渲染质量和为 1。Amplitude/duration heads、固定 `1/2/4/8s` bank、跨 center/modality noisy-OR、top-K MIL、smoothness 与 contrastive loss 均保持不变。旧首版 splat 是 matched control。

## Selection and evaluation

每个语料只用各自 train 训练；validation 只选择预先列出的训练配置与 checkpoint，选择完成前不生成 test prediction。随后对所选配置立即生成 HMM/HCS test prediction，并调用冻结 canonical evaluator 输出三个固定指标。若仍未通过双数据集 performance gate，RESET3 窗口记 `1/3`；不追加 renderer、kernel 或 loss 扫描。

权威输出根目录：`runs/20260901_mass_conserving_marked_splat/formal_seed234/`。

## Formal result and decision

Validation 在任何 test prediction 前完成：HMM 选择 `low_regularization`（validation video AP `.852651`，epoch 21），HCS 选择 `bag_focus`（`.957669`，epoch 61）。权威汇总为`runs/20260901_mass_conserving_marked_splat/formal_seed234/summary.json`。

- HMM AP/pooled ROC/within 为`.512140/.755154/.686462`，相对首版 splat 为`+.016774/+.014423/-.041247`。
- HCS 为`.572049/.549973/.527733`，相对首版为`-.003420/-.003854/-.006348`。

双数据集 performance gate 失败，RESET3窗口记`1/3`。边界质量守恒回收了部分 HMM pooled separation，但移除了大量首版位置增益，且没有改善 HCS；该固定 renderer 修正停止，不扫描归一方式或 kernel。下一轮不做 premise，回到已审过的原始 renderer，只补做正常 validation 配置选择并立即双 test，以隔离配置选择与 renderer 改动。
