# Marked Splat Formal-Test Error Analysis

截至 2026-09-01。该实验不是 premise 或新 candidate；它是 RESET3 要求的正式方法 post-test error analysis，只读取已有 HateMM/HateClipSeg anchor、point-control、splat test predictions 与 test GT，不训练模型，不选择 checkpoint，不生成可部署 prediction。

目标固定为回答四件事：pooled 缺口属于哪类 separation；splat 增益集中在哪些 occupancy/span/carrier strata；HCS 小增益是 boundary ordering 还是无差别 score spreading；time-shift 与 leave-one-video-out position controls 是否支持 duration-field 的时间对齐归因。分析完成后不得继续扩展 statistic sweep，只能选择一个由结果直接支持的方法内改动进入正式训练。

运行：

```bash
conda run -n HateVideo python experiments/20260901_marked_splat_test_error_analysis/analyze.py
```

权威输出：`runs/20260901_marked_splat_test_error_analysis/main/metrics.json`。

## Result

- HateMM splat 的 within 为`.727709`，但正帧对负视频全部帧的 ROC 为`.776200`，与 anchor `.776121`几乎相同；video-label mean-score ROC 还从`.872047`降到`.849248`。Pooled 缺口主要是跨视频/bag separation，不是正视频内部排序。
- HateMM leave-one-video-out position-only within 已有`.709890`，splat raw 只高`.017819`；去公共位置轮廓后 splat within 为`.591156`，仍高于 point `.539834`，说明 duration field 有内容相关贡献，但头条增益中有大量共同位置/边界效应。
- HCS splat 同时改善正帧对负视频帧 ROC（point `.583558`→splat `.657375`）与正视频内部 ROC（`.496339`→`.517679`），不是单纯无差别扩散；但 position-only 已有`.530324`，raw 只高`.003757`，去位置后 splat/point within 仅`.512890/.510991`。HCS 的 duration 增益几乎没有超出共同位置轮廓。
- 时间循环错位的平均 within 为 HMM `.461049`、HCS `.494136`，低于 aligned `.727709/.534081`；但 HCS 的 `0.2` 周期错位反而到`.540768`，进一步说明 HCS 时间归因不稳。
- Carrier-count 分层没有形成决定性解释：eligible 视频几乎都有三个可变 modality carrier（HMM `80/85`，HCS `65/67`），因此不作为门。

## Decision

分析到此停止，不追加 probe。下一正式方法内改动固定为 **per-center mass-conserving splat renderer**：每个 evidence center 在真实视频边界截断后重新归一其 kernel mass，使同一 amplitude 不因位于开头/结尾而系统性变小。该单一改动直接针对已观察到的共同位置/边界轮廓，并使跨视频 event amplitude 更可比；其余 marked duration representation、noisy-OR readout 与训练目标保持不变。训练配置与 checkpoint 可由 validation 正常选择，随后立即 HMM/HCS test。
