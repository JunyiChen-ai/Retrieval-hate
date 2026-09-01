# Premise probe pre-run review

截至 2026-08-31。审查范围：`README.md`、`probe.py`、`decide_premise.py`、
`test_probe.py`、`run_probe.sh`、`launch_probe.sh` 及其直接调用的共享 feature、split、label 与
evaluator 入口。

## 裁定

**PASS，可以启动冻结的 HateMM/HateClipSeg premise probe。** 本次仅运行了 synthetic/静态检查和
只读规模检查；没有启动 probe，没有生成 test prediction。

review 中修复了自动 gate 的三个边界问题：不再直接信任 `support.json` 中预写的布尔值，而是由
support fraction 重新计算并核对；拒绝非有限的 within ROC/support 输入；`+.020` 边界按数值容差
处理，避免十进制差值的表示误差。README 中与“两个语料都严格上升”重复且逻辑无效的下降容忍条款
已删除。`run_probe.sh` 也增加了固定 `HateVideo` Python 的可执行性检查。

## 1. Producer 数据边界

- `probe.py` 只从目标 corpus 的 frozen train split 取 IDs，并只调用一次
  `scoped_video_labels(corpus, "train", train_ids)`；随后只保留 label 为 0 的视频。HateMM 实查为
  446 个 negative-train videos，HateClipSeg 为 32 个。
- test 阶段只通过 `hdata.load_split(corpus, "test")` 与 `evaluator_test_ids` 取得 blind IDs，再读取
  A/V/T features。producer 没有导入或调用 test labels、frame labels、temporal GT 或 evaluator。
- `aligned_local_features` 的直接依赖只做 audio/visual/text feature path、时间对齐与行归一化；其
  producer 路径不读取标签。policy primitive 文件由固定文本语义 prototypes 构成，shape 为
  `en/zh: [6,768]`；本 probe 使用英文 frozen prototype，只把 hostile、violence、sexual、self-harm、
  context span 投影掉，保留 target/topic 方向。它不包含当前 train/test 样本。
- test GT 只在 `probe.py` 完成 blind scores 后由共享 evaluator 读取。`decide_premise.py` 只读取
  evaluator 的汇总指标和 producer 的 support 汇总，不读取 prediction 内容或逐帧 GT。

结论：producer 符合“只用目标 corpus negative-train video labels 拟合，test labels/temporal GT 不进入
producer”的约束。

## 2. Conditional/unconditional 数学与公平性

- topic branch 先把 frozen text sentence feature 投影到五个 harm/context directions 的正交补，再只
  用 negative-train frames 拟合 PCA-16。`np.linalg.qr(harm.T)` 产生正交 basis，
  `x - (xB)B^T` 的 projection 公式正确。
- residual branch 对相同 frozen concat A/V/T frames、只用 negative-train frames 拟合 PCA-32；proposal
  residual 是 projected inside mean 减去可用 surrounding mean。PCA、proposal grid、residual、训练
  proposal集合在 conditional 与 unconditional 两臂完全共享。
- conditional arm 在 standardized topic descriptor 上以 ridge `1.0` 预测同一 residual，随后用
  negative-train conditional errors 的逐维方差计算 diagonal Gaussian dimension-mean NLL。
  unconditional arm用同一 residual 的全局 mean/variance计算 matched diagonal Gaussian NLL。
  忽略的 `log(2π)` 是 proposal-independent 常数，且两臂各自随后标准化，不影响其 frame ranking。
- 两种 energy 都只用各自在同一 negative-train proposals 上的 energy mean/std 标准化。这个 affine
  normalization 不使用 test distribution；scale 有 `1e-6` 下界，Gaussian variance同样有下界。
- conditional 多出的 ridge 参数正是 premise 要检验的 target conditioning，不存在额外 residual feature、
  proposal budget或 test-fitted normalization。该 probe 不声称校准 likelihood，也不把其数值与
  unconditional raw NLL直接比较；只比较由相同 readout形成的 test within-video ordering。

README 的正式方法段写 proposal-level PCA，而冻结 probe 段明确写 negative-train **frames** 拟合 PCA；
当前代码与后者及 config 一致。premise 结论只能归于这个 frozen frame-PCA probe，不能偷换成尚未运行的
formal flow。

## 3. Proposal、surrounding 与 frame readout

- 对每个长度 `T`，代码枚举每个合法起点的 `1/2/4/8/16/32/64/128` 秒非空 interval；若
  `[0,T)` 尚未在固定尺度中则补一个 whole-video proposal，不重复已有 whole-video interval。
  bounds 均满足 `0 <= start < end <= T`，1 秒 proposals 保证每秒被覆盖。
- inside/topic mean 使用 prefix sums，分母为 interval width。surrounding 取最多一个 proposal width 的
  left 与 right 可用 frames；两边存在时按全部可用 surrounding frames 求均值，只有一边时只用该边。
  whole-video 的 outside count 为零，outside 明确为零，因此 residual 等于 inside descriptor。
- `frame_readout` 以 difference arrays累加所有覆盖 interval 的 standardized energy 与覆盖数，最后做
  覆盖 proposal 的算术平均。它不是 coverage sum；constant proposal energy 在长度 1、7、129 上均
  产生严格平坦 score。任一未覆盖 second 会直接报错。
- whole-video proposal 当前包含在 frozen main score 中；synthetic test 已验证其 zero-surrounding 数值。
  premise 没有 sparsemax 或 proposal mass，因而 README 要求的 whole-video mass/active-count 属于通过
  premise 后 formal method 的报告项，不能从本 probe 推断。

## 4. Support 与 direction gate

- topic support 仅在 standardized negative-train proposal topics 上拟合最多 64 个 centroids；门限是
  negative-train proposal 到最近 centroid 距离的 95th percentile。test support 只做 blind feature
  distance计算，总 support fraction 写入各 corpus 的 `support.json`。
- shared evaluator 对 `score_conditional` 和 `score_unconditional` 都输出固定 test 三指标。
  direction gate只取 evaluator 原生 `per_video.macro_auc`：两个 corpus 的
  `conditional - unconditional` 都必须严格大于 0，且至少一个必须达到 `+.020`。
- `decide_premise.py` 在两个 evaluator 都完成后生成唯一 `verdict.json`。它同时要求每个 corpus
  support fraction `>= .80`，重新计算并核对 producer 的 support decision，检查 corpus/split scope 和
  数值有效性。因此 gate 可由 `metrics.json + support.json` 无歧义复算；support 不是 evaluator metric，
  不应误称为 evaluator 自身输出。
- 正例、反例、精确 `+.020` 边界与 support `.80/.79` 边界均有 synthetic test。任一语料 support、
  严格方向或大增益条件失败时，decision 固定为 `STOP_BEFORE_FORMAL_METHOD`。

## 5. 数值、内存与运行风险

- 所有 heavy arrays 当前是 CPU float64。按 frozen negative train feature lengths 只读估算：HateMM
  约 96,914 frames、686,594 proposals，最长视频约 8,712 seconds；HateClipSeg 约 11,501 frames、
  84,136 proposals，最长约 479 seconds。HateMM cached frame arrays、PCA workspace、proposal matrices
  与 64-centroid distance matrix预计为数 GB，低于本机约 60 GiB RAM，但不是轻量任务。
- 最明显的峰值来自同时保留 per-video float64 cache 与 concatenated frame matrices，以及
  `kmeans.transform` 的 `N×64` distance matrix。当前规模可运行；若未来扩到 MHC 或增密 proposal grid，
  必须改为分块 transform/在线统计，不能直接照搬。
- prefix sums、PCA、ridge、variance floor、standardization floor 和 finite gate共同避免除零/非有限结果
  静默晋级。若 PCA 输入样本不足、feature shape异常、某秒无 proposal、共享 evaluator覆盖不全或某指标
  无定义，脚本会停止而不会产生通过 verdict。
- probe 是 CPU/内存任务，两个 corpus 串行运行。`launch_probe.sh` 使用 `setsid` 与 SSH session 解耦，
  root 写 `run.log` 和 `run.pid`；shell 使用 `set -euo pipefail`，producer/evaluator/gate 任一步失败都会
  停止。

## 6. 共享 evaluator 与输出

- `run_probe.sh` 直接调用 `research-wiki/STATUS.md` 登记的唯一共享 evaluator：
  `scripts/reproduction_baselines/eval_baseline_scores.py`，split 强制为 test，并启用
  `--require-full-coverage`。
- 未指定 branch 时，共享 evaluator从 blind score records 评测两个 `score_*` branches；输出包含 pooled
  AP、pooled ROC-AUC 与 within-video macro ROC-AUC。缺视频、多视频、长度不匹配或非有限 score 都会失败。
- 输出全部进入 `runs/20260831_target_conditioned_normal_proposal_mil/premise/`：每语料 config、blind
  scores、support、evaluator metrics，以及 root log、PID、最终 verdict。`data/` 不被写入。

## 7. 已执行检查

- 6 项 `unittest`：bounds/coverage/whole-video、inside-surrounding exact value、constant-energy flat
  readout、producer source boundary、gate正反例、gain/support边界，全部通过。
- `probe.py`、`decide_premise.py`、`test_probe.py` compile 通过；两个 shell scripts syntax 通过。
- frozen primitive shape、negative-train counts 与运行规模只读检查通过。

最终裁定：**PASS FOR FROZEN PREMISE PROBE**。probe 结果仅为 iterative/developmental test evidence；
不得作为正式方法 prediction 或 SOTA 数字。只有 `verdict.json` 的 `premise_pass_both=true` 才允许进入
formal method 实现。
