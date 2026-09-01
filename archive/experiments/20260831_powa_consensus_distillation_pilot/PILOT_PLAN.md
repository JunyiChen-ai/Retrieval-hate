# Frozen pilot plan

**截至 2026-08-31；revision 3，冻结对象：HateMM + HateClipSeg test evaluation。**

Revision 3 服从最新协议裁定：validation 只用于每个训练臂内部选择 checkpoint，不再用
validation performance、upper bound 或 gate 决定方法方向。每个臂训练完成后立即在 test 上
评测全部固定指标，方法修改与淘汰只依据 test evidence；test 标签不进入梯度或 checkpoint
选择。此前 validation upper bound 仅作为历史诊断，不再作为运行门槛。

本 recipe 是在查看多个 validation upper-bound arm 后确定的 validation-adaptive design，
不是预注册选择。audio+VERA 是已知 multi-teacher percentile calibration + ordinal KD，
不承担 novelty claim。未来任何 test 结果均按 Rule 10 作为 iterative/developmental evidence。

Revision 3 删除全部哈希、qualification 和 validation gate。VERA 固定使用 batch 2；HCS 使用
已有完整 238-video K16 cache，HMM 用相同 batch 2 producer 补齐 744-video K16 cache。两者均
通过实际 JSON 解析、完整 coverage、时间索引和 score 范围检查，不做哈希校验。

## 固定输入

- seed：`234`。
- corpus-specific POWA anchor：沿用
  `runs/20260831_powa_starting_point/summary.json` 所绑定的 checkpoint。
- VERA：`OpenGVLab/InternVL2-8B`，`torch_attention`，代码内唯一固定 hate-domain 五问
  prompt，10 秒窗口，batch size 2；运行时不读取 validation prompt-selection 文件。
- train VERA support：
  `unique(round(linspace(0,L-1,min(16,L))))`。
- audio probe：L2-normalized VGGish 1fps，`SGDClassifier(loss=log_loss,
  penalty=l2, alpha=1e-4, average=True)`，5 epochs，5-fold stratified OOF，fold seed
  `234`。每 corpus 独立拟合；拟合时每视频最多 uniform 取 200 行，生成 teacher 时在完整
  student 时间网格 score。正式 train teacher 对每个 train 视频只用未见该视频的 OOF fold
  model。
- teacher order：先在完整 student 时间网格上计算 `percentile(audio_oof)`；VERA 的 K16
  raw score 先按时间线性插值到同一完整网格，再计算 `percentile(vera_interp)`；最后取
  `0.5 * audio_percentile + 0.5 * vera_percentile`。在该完整网格上冻结绝对差至少
  `0.20` 的 pair endpoints；每个 anchor point 最多取一个更高和一个更低 pair，避免长视频
  二次加权。core、shuffled、audio-only、VERA-only 全部复用这些 endpoints 和 pair weights。
- student：frozen POWA + `src/` 共享 residual head；hidden size 沿用 POWA，两个
  kernel-3 temporal conv，zero-init output，masked zero-mean residual。训练和 residual
  推理均使用相同 fixed 200-bin context grid；测试另跑 native dense-grid POWA anchor，将
  fixed-grid residual 线性 lift 后按 crop 在 native grid 重新减均值，再与 dense anchor logit
  直接相加。即同一 frozen POWA 两次 forward；不隐瞒该推理开销。
- loss：`pairwise_softplus(margin=0.25) + 0.5 * anchor_centered_smooth_l1 +
  0.5 * original_POWA_video_MIL`。不按 corpus 调权重。
- optimizer：AdamW，lr `2e-4`，weight decay `1e-4`，gradient clip `5.0`，5 epochs，
  batch size 24。

## 运行顺序

1. 使用相同本地 InternVL2-8B 模型、prompt 和 batch size 2；复用完整 HCS train K16 cache，
   生成 HMM train K16 cache，并为 HMM/HCS test 各自实际生成 fixed K16 cache。test diagnostic
   直接读取这些 K16 segments，不从旧 dense prediction 抽点。只做解析、coverage、时间索引和
   score 范围检查。
2. 跑 synthetic/property tests：pair orientation、mask、zero-mean、zero-init identity、
   no-teacher inference、split isolation、score/GT alignment、shared evaluator。
3. 独立 code/evaluation review 得到 PASS 后，每个 corpus 运行以下 6 臂：
   `powa_anchor`、`powa_residual_no_teacher`、`powa_shuffled_teacher`、
   `powa_audio_only`、`powa_vera_only`、`powa_audio_vera`。除 pair direction/loss 开关外，
   eligibility、head、optimizer、epochs、MIL/anchor loss、checkpoint rule 完全相同。
4. 每个 epoch 都导出 validation direct-additive score，由唯一共享 evaluator 直接产生
   `metrics.json`。
5. 优先在 pooled-feasible checkpoint 中选 within-video ROC 最高者；若没有可行 epoch，仍选
   within-video ROC 最高者并立即跑 test，同时记录 feasibility 失败；不得用 test 选择。

## 冻结 test evaluation gate

每个 corpus 的 core test 都必须同时超过权威 baseline/SOTA 门：

- HMM pooled AP/ROC/within ROC `>= .5938316/.8161838/.6315317`；
- HCS pooled AP/ROC/within ROC `>= .6193711/.6050225/.5619079`；
- 三指标同时高于相同 seed-234 corpus-specific POWA anchor；
- core 的 within 增益至少比 shuffled_teacher 高 `.010`；
- core 的 within 必须高于 residual_no_teacher（两 corpus 都成立）；
- zero-init epoch 0 与 anchor score 最大绝对误差 `<= 1e-6`；
- inference graph 不读取任何 VERA/audio-probe teacher artifact。

HMM/HCS 每个训练臂选定 checkpoint 后立即跑 test，不设置 test 前 validation gate。是否扩展
EN/ZH 以及下一轮如何修改，只依据 HMM/HCS test 全部固定指标和归因 control；仍使用完全
相同的 prompt、K、teacher construction、student、loss 和 checkpoint rule。

audio-only 与 VERA-only 是在 core 固定 endpoints 上的 conditional-direction 归因臂；某一来源
两端打平时该 pair 不施加该来源的 loss，并报告 active coverage。它们不是独立 single-teacher
endpoint 方法，也不允许按 corpus 选择它们替代固定 core。若 core 不胜两者，
不得声称 teacher complementarity；即使 absolute gate 通过，也只能把固定双 teacher 如实记为
已知辅助 recipe。

HMM 所有 arms 使用完全相同的 744-video eligible cohort；HCS 所有 arms 使用既有 label-free
media audit 确认的 238 个可解码 train 视频，并统一排除 13 个无可解码 visual stream 的视频。
coverage 不能成为 arm difference。

## 扩大实验前的 load-bearing control

最小 pilot 的 test 结果显示值得继续后，必须完成：

1. `MACIL/base × {no KD, same audio+VERA KD}` 与
   `POWA × {no KD, same audio+VERA KD}` 的 2×2 matched control；普通 backbone 不得减少
   输入、容量或训练预算。加入同一 auxiliary 后，POWA+KD 必须仍优于 MACIL+KD，才能把最终
   performance 与 POWA novel core 一起 claim。
2. 在 HMM/HCS 当前 within-video ROC 上复核 full POWA、same-time/pointwise binder、
   flat/anonymous head、policy/teacher-channel permutation。
3. 多 seed 与 paired uncertainty；单 seed 只作 kill pilot，不作论文结论。

## 性能解释边界

audio+VERA transport 是上限，不是方法结果。单学生即使达到上限，也只能说明已知辅助
排序可被 POWA representation 内化；不能把 consensus、rank distillation、teacher density
或 transport 写成新的算法贡献。

zero-mean 约束只作用于 residual logit；它不严格保持 sigmoid 后的均值、quantile 或 score
multiset。pooled performance 只按 test evaluation 报告，validation 只参与 checkpoint selection。
