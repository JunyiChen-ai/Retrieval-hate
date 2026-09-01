# 淘汰：Monotone-warp rank equivariance

淘汰原因：独立 novelty review 给出 `STOP, 4.6/10`。来源 Gong et al. temporal transforms
未被检到已用于 hateful video task，但其已在 video-label WTAL 占用 local warp、pullback
localization consistency 与 adversarial transform selection。当前 sigmoid pairwise-JS 在
pair graph 连通时等价于 warped logits 与 clean logits只差一个 video常数，不是真正新的
ordinal invariant；若改用纯符号/order consistency，又只能保存 clean模型已有错误排序，并
存在 constant/broadcast与scale saturation退化。未实现、未训练、未生成 prediction；完整
依据见 `NOVELTY_REVIEW.md`。

截至 2026-08-31。当前已在 novelty/identifiability 阶段淘汰；未实现、未训练、未生成
prediction。

## 直接失败证据

跨语料 trainable starting architecture 仍为 MultiHateLoc。它以每秒独立 head + 固定
top-third MIL训练，HCS fused branch 超过全部单模态 branch 的视频比例仅 `.154`；HMM/HCS
structured scores还表现出高分区相对 GT transition 数的 `4.5×/5.5×` 碎片膨胀。固定 7 秒
smoothing 虽改善 HMM，却损害 HCS，说明不能再加入 generic duration prior 或 smoothing。

既有 `powa_orbit_equivariance_probe` 只检查 cyclic origin shift，并已证明 POWA 对起点近似
equivariant；它否定 absolute-position shift 作为 POWA 的瓶颈。本候选不是 cyclic shift、
不是 POWA score averaging，也不使用固定 smoothing；它检查 MultiHateLoc 在局部快慢变化的
严格单调时间重参数化下，是否保持对应内容的 frame ordering。

依据：

- `runs/20260831_multihateloc_test_error_analysis/main/metrics.json`
- `runs/20260831_postcoalition_test_diagnosis/main/metrics.json`
- `runs/20260831_powa_orbit_equivariance_probe/analysis.json`

## 跨任务来源

来源方法是 Gong et al., IJCAI 2021, *Self-Supervised Video Action Localization with
Adversarial Temporal Transforms*。它在 weakly-supervised temporal action localization 中
使用 temporal downsampling、attention-guided time warping、equivariant localization
consistency 和 adversarial transform policy。当前检索尚未发现该方法用于 hateful video
detection/localization；必须由独立 reviewer 再查证。

## 非 trivial adaptation

### 1. 从 pointwise equivariance 改为 metric-aligned ordinal equivariance

设固定、严格单调、分段线性的时间映射为 `w:[0,1]→[0,1]`。干净输入产生秒级 logit
`s(t)`；warp 后输入产生 `s_w(w(t))`，再按已知 `w^{-1}` 拉回原 1 fps 网格。来源方法的
pointwise consistency要求两条 score curve 数值接近。本候选只要求对应 frame pair 的排序
概率一致：

`q_ij = sigmoid((s_i-s_j)/tau)`，

`L_rank-eq = mean JS(stopgrad(q_ij), q^w_ij)`。

pair 从同一视频的有效秒中均匀抽取，不使用 frame label、test GT、top/bottom oracle 或
语料特定规则。clean side stop-gradient，避免两边共同移动掩盖不一致。这个目标直接对应
within-video ROC只依赖相对排序的评价语义，不强迫不同 duration/edit realization 具有相同
calibration，也不把相邻秒拉平。

### 2. Adversarial warp 选择直接攻击排序 shortcut

每个 train video 使用一个预先固定的有限 warp bank；knots 和最大位移在 test 前写死，所有
语料相同。每一步选择使当前 clean/pullback pairwise disagreement 最大的 warp，再最小化
video MIL + rank-equivariance。warp policy不读取 label以外的信息，不学习 corpus-specific
参数，也不在 test 使用。

严格单调 warp保持内容顺序和 video label，但改变局部 duration、采样密度与片段边界相对
网格的位置。若模型依赖固定 top-third、fragment length、重复帧数或编辑速度 shortcut，其
frame ordering会随 warp改变；真正绑定到内容的 ordering应在 pullback 后保持。

### 3. 最终输出

推理是一次原始视频 1 fps forward，直接输出连续 frame posterior。没有 test-time
augmentation、score averaging、branch routing、threshold search、calibration或平滑。

## 可证伪假设与 controls

首轮只在 HateMM/HateClipSeg 独立训练，seed 234；validation只为每个固定 arm选 checkpoint，
随后立即test三项固定指标。

训练 arms：

1. `core_rank_eq`：adversarial monotone warp + ordinal equivariance；
2. `source_pointwise_eq`：相同 warp/预算，改为来源式 pointwise score consistency；
3. `warp_bce_only`：相同 warp与双 forward，只做两边 video BCE，不做 consistency；
4. `no_warp`：相同架构与总 optimizer steps，不做 warp。

机制成立必须同时满足：

1. core 对 `source_pointwise_eq` 与 `warp_bce_only` 的 within ROC 在 HMM/HCS 两边同向提高，
   且至少一边 `>= .020`；
2. clean/pullback Kendall disagreement 显著下降，但 clean score temporal total variation
   不得整体塌缩；若靠近常数曲线满足，机制失败；
3. core 的改善不能只来自 video score尺度；within不提高即失败；
4. core 必须在 HMM/HCS 的 pooled AP、pooled ROC、within ROC 六个格全部严格超过当前
   SOTA，才扩 EN/ZH；最终四语料每项都必须 SOTA。

附加 diagnostic：对同样位移幅度的非单调 permutation做 pullback，rank consistency不应被
要求成立；若模型对非单调破坏也同样“稳定”，说明输出已塌缩或忽略内容。该 diagnostic
不参与训练和 checkpoint selection。

## Anti-pattern guard

- 不扫描 warp幅度、knot数、pair margin、loss weight或语料特定设置来追 test。
- 不把 temporal augmentation本身、IJCAI来源方法或 within-oriented loss单独 claim novelty。
- 不使用 test prediction生成训练 pair；全部 pair与warp只来自该语料 train forward。
- 不把 clean/warped predictions做 test ensemble。
- 不把低 disagreement当 performance；必须同时过机制 control与三指标 SOTA。
- 正式运行前独立代码审查必须验证 warp/pullback方向、padding mask、pair采样、clean
  stop-gradient、train/validation/test isolation和共享 evaluator调用。

## Novelty review 必答

1. 来源 temporal-transform方法是否已用于 hateful video detection/localization？
2. `adversarial monotone warp + metric-aligned ordinal pullback consistency` 是否是针对本任务
   within-video ranking与duration shortcut的 non-trivial adaptation，还是普通 augmentation、
   consistency regularization与pairwise ranking loss的拼接？
3. 在只有 video labels时，该约束是否能提供新信息，还是只保持当前错误排序/存在常数解？
4. 是否已被 WTAL rank consistency、temporal equivariance、ranking distillation或项目既有
   rank-transport/orbit实验实质占用？
