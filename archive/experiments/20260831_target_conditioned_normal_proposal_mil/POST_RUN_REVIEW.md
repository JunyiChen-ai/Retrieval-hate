# Target-conditioned normal proposal premise：独立 post-run audit

日期：2026-08-31  
范围：`experiments/20260831_target_conditioned_normal_proposal_mil/` 与
`runs/20260831_target_conditioned_normal_proposal_mil/premise/`。  
结果链结论：**PASS。** Producer、shared evaluation、support 汇总和冻结 gate 均可独立复算；
premise 本身 **FAIL**，唯一合规结论是 `STOP_BEFORE_FORMAL_METHOD`。

## 进程与正式产物

- Detached PID `3908286` 已退出，不存在仍运行的 producer/evaluator 进程。
- `run.log` 依次包含 HateMM producer、两个共享 evaluator branches、HateClipSeg producer、两个
  branches，以及最终 verdict；末行明确为 `STOP_BEFORE_FORMAL_METHOD`，无 traceback 或中途截断。
- 两语料均有固定 config、完整 `scores.jsonl`、`support.json` 与共享 evaluator 原生
  `metrics.json`；root 有最终 `verdict.json`。
- 两个 config 均为 seed 234 和冻结 recipe：proposal lengths `1/2/4/8/16/32/64/128` 秒、加入
  whole-video proposal、topic PCA-16、residual PCA-32、ridge 1、64 topic clusters、negative-train
  support P95 与 `.80` support gate。

## Producer test-isolation 边界

- `probe.py` 仅从同一 corpus train manifest 读取 scoped train video labels，并只选择 negative-train
  videos 拟合 PCA、conditional/unconditional Gaussian、ridge 与 topic support model。
- Test 阶段只读取 frozen evaluator-test IDs 和 aligned local features。静态调用检查确认 producer
  没有 `gt_arrays`、聚合 label loader 或 test scoped-label 调用；唯一 scoped-label 调用的 split 是
  literal `train`。
- Temporal GT/test labels 只由随后独立执行的共享 evaluator 读取；没有训练 checkpoint，也不存在
  test label 参与梯度或 selection。`support.json` 与 verdict 的 isolation 声明符合实际调用链。

## Full coverage、finite/length 与共享指标复算

- HateMM scores 精确按 frozen test manifest 覆盖 214 个唯一视频、29,269 帧；HateClipSeg 覆盖
  79 个唯一视频、18,839 帧。两者无 missing、extra 或 duplicate。
- 每行 schema 仅含 video ID、conditional 和 unconditional 两个 branches；每个 score vector 都是
  一维、finite，并逐视频与 1 fps GT 长度完全一致。
- 在内存中对两个语料 × 两个 branches 只调用仓库唯一
  `eval_baseline_scores.evaluate_scores`。四份完整结果与两个 `metrics.json` 逐字段一致，shared
  evaluator coverage 均为零 missing/extra：

| corpus | branch | pooled AP | pooled ROC-AUC | within-video ROC-AUC | within n |
|---|---|---:|---:|---:|---:|
| HateMM | conditional | 0.19171743597917137 | 0.40396710030648364 | 0.2933098868382792 | 85 |
| HateMM | unconditional | 0.19295717933079562 | 0.40846281292226794 | 0.29632935540479394 | 85 |
| HateClipSeg | conditional | 0.4901004520497286 | 0.4599788917251721 | 0.3849434412682865 | 67 |
| HateClipSeg | unconditional | 0.4906651483563242 | 0.46078228306106145 | 0.39528471104187385 | 67 |

这些是 developmental premise diagnostics，不是正式方法 prediction 或 SOTA 结果。

## Support fraction 独立复算

依据每个 test 视频的 1 fps 长度重新枚举冻结 proposal grid，再把 `per_video_support` 还原为整数
supported proposal counts 并跨视频汇总：

- HateMM：`167607 / 192740 = 0.8696015357476393`，严格复现 `support.json`，且
  `>= .80`，所以 support PASS。
- HateClipSeg：`37898 / 131278 = 0.28868508051615654`，严格复现 `support.json`，且
  `< .80`，所以 support FAIL。

两个 `n_test_proposals`、每视频 support fractions、aggregate fraction 和 `support_pass` boolean 均
一致；因此 `support_pass_both=false` 正确。

## Frozen gate 与最终裁定

- HateMM conditional-minus-unconditional within 为
  `0.2933098868382792 - 0.29632935540479394 = -0.003019468566514749`，严格方向门 FAIL。
- HateClipSeg 对应差为
  `0.3849434412682865 - 0.39528471104187385 = -0.010341269773587347`，严格方向门 FAIL。
- 两个差值都不是正数，更没有任一 corpus 达到 `+.020`；因此
  `strict_direction_pass_both=false` 与 `one_corpus_gain_at_least_020=false` 正确。
- 三项冻结前提中，双语料 support、双语料正方向、至少一边 `+.020` 全部没有联合成立。独立按
  原公式重算得到 `premise_pass_both=false`、`continue_to_formal_method=false`、
  `decision=STOP_BEFORE_FORMAL_METHOD`，与 `verdict.json` 完全一致。

**最终 verdict：PASS（result-chain integrity）；premise：FAIL。** 不得实现或运行 formal
method，不得按 corpus 改 topic bandwidth/flow，也不能把本 premise probe 当作候选方法。本审查
未修改任何结果，也未启动新实验。
