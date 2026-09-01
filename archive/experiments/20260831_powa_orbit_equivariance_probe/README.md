# 淘汰：POWA temporal-orbit equivariance probe

淘汰原因：双语料 within 增益门和 cyclic-vs-random attribution 门均失败。权威输出
`runs/20260831_powa_orbit_equivariance_probe/analysis.json`（SHA256
`f2601289c74823c7edcac47699d5bf0b3e1b988aef623fb773d49323dfafa562`）。HateMM
within `.57193→.56985`，HCS `.52707→.52724`；cyclic 与 matched random 的差仅
`.00016/-.00004`。不训练、不查新、不读 test。

截至 2026-08-31。零训练、validation-only，不读 test。循环平移多次推理后反变换并平均
属于 test-time ensemble，因此严格只作 upper bound，不能成为方法。它只回答：训练期
强制同一个 POWA 对 temporal-origin 变化保持 frame equivariance，是否有跨 HMM/HCS
的性能空间。

固定使用 corpus-specific POWA seed-234 validation checkpoints；每个视频取原序列与
`T/4,T/2,3T/4` 三个循环平移，逐个推理、反平移 frame scores 后无权重平均。matched
control 使用原序列加三个由 `(corpus,video_id)` 固定种子生成的任意 frame permutations，
同样反置换平均；它检查增益是否只是多次推理把分数抹平，而非 temporal-origin group。
所有 modality 使用完全相同的 permutation，不制造 modality desynchronization。

进入 novelty review 的冻结门必须在 HateMM 和 HateClipSeg 同时满足：

1. cyclic orbit mean 的 within-video ROC 比原 POWA 至少 `+.020`；
2. pooled AP 与 pooled ROC 各不低于原 POWA `.005` 以上；
3. cyclic within 比 matched random-permutation mean 至少高 `.010`。

任一失败即归档，不训练、不调 shift 数量、不增加 reverse/smoothing/score transport。
若通过，真正候选也只能是 single-pass student 的 train-time equivariance loss；本 probe
本身永远不计方法或 novelty。

## 结果解释

反变换后的 cyclic-view frame score 与原分数平均绝对误差只有 HMM `.00161`、HCS
`.00048`，任意 permutation ensemble 也几乎相同。POWA 对 temporal origin 已近似不敏感；
此前 center/edge correlations 不是可由 shift-equivariance loss 修复的 absolute-position
encoding failure。继续加 reverse、更多 shifts 或 smoothing 只会变成 test-time averaging
调参，按冻结规则停止。
