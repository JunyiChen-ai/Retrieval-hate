# Independent pre-run review

日期：2026-08-31

## Verdict

**PASS。** 两个会改变正式观察或 gate 的问题已在正式 test 运行前修复；当前实现可以运行冻结 premise。评审没有启动正式 producer 或读取性能结果。

## 运行前发现并修复

1. 原 producer 以 raw container duration 的 ceiling 作为 score 长度，但共享 evaluator 的冻结 1 fps 网格来自 audio clock。全 cohort 只读检查发现 HateMM 9/214、HateClipSeg 3/79 个视频不一致，原实现会在 evaluator 的 shape 检查直接失败。现改为使用既有、无标签的 `results/reproduction/features/clip_b16_1fps/<corpus>/<video>.npy` 行数定义 score 网格；container duration 仅记录差异。修后该网格与冻结 GT 长度逐视频一致：HateMM 214/214，HateClipSeg 79/79。
2. 原 32-arm circular-shift 调度会在短视频上用 modulo 重复 shift，额外偏重较小 offset。HateMM test 中有 42 个长度不超过 33 秒的视频，因此可能改变 `.020` alignment gate。现 gate 的 shift mean 先对每个可算视频的最多 32 个均匀 unique nonzero shifts 等权平均，再对视频等权平均；每次 per-video ROC 仍由共享 evaluator 产生。32 个联合 controls 只负责 pooled/global control 的均值和范围，短视频重复也改为均匀分配。
3. HateClipSeg 训练标签读取改为官方 `video_level_annotation.csv`，且只在 ID 属于 frozen train split 时解析标签；不再打开 segment labels/timestamps。README 同步明确实际权重：类内视频等权、两类总权重相等、总 sample weight 等于 train 视频数，因此 `C=1` 的尺度固定。

## Result-chain checks

- `produce.py` 的模型拟合只使用各自 corpus 的 train IDs、train video labels 与 `dense4fps_w2vemo`。HateMM/HateClipSeg train-test ID 交集均为空；scoped labels 完整覆盖 744/251 个 train videos，并与现有数据定义一致。
- Test score 生成前只从冻结 GT archive 读取 key names，以取得共享 evaluator 的 exact cohort；没有打开其中任何 label array。模型、scaler、score length 和 feature sampling 均不依赖 test label values。
- 4 fps 到 1 fps 映射是固定的 `feature[4*t]`，对应 dense cache 的整数秒时间戳。若 dense tail 短于冻结网格则重复最后一行并显式计数；只读预检为 HateMM 52 秒、每视频最多 4 秒，HateClipSeg 11 秒、每视频最多 1 秒。
- Producer 对 feature shape、feature finiteness、score finiteness 和输出长度 fail-closed。Evaluator 对 score/GT shape、finite、missing IDs 和 extra IDs 再次 fail-closed。
- 正式三指标及所有 circular-shift 指标只调用 `scripts/reproduction_baselines/eval_baseline_scores.py::evaluate_scores`；实验目录没有复制 pooled AP、pooled ROC 或 within-video ROC 的实现。
- Gate 与 README 一致：每个 corpus 都要求 original within-video ROC `> .52`、相对 equal-video/equal-unique-shift mean 的增益 `>= .020`、exact coverage；两 corpus 都通过才输出 `PROCEED_TO_NOVELTY`。
- Logistic recipe 固定为 weighted `StandardScaler` 与 `LogisticRegression(C=1, solver=lbfgs, max_iter=500, random_state=234)`，无 validation、checkpoint selection 或参数扫描。

## Checks run

- 两个 Python 文件 `py_compile`：PASS。
- `run.sh` syntax check：PASS。
- train/test isolation、scoped train-label coverage、class weight sums：PASS。
- 两个已知 audio/container clock 差异样例的 canonical grid regression：PASS。
- synthetic short-video unique-shift、score-multiset preservation、equal-shift aggregation 与 shared-evaluator recomputation：PASS。

