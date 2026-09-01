# Formal pre-run review

截至 2026-08-31。审查对象：本目录 `README.md`、`model.py`、`run.py`、
`test_port.py`、`run_pilot.sh`、`launch_pilot.sh`，以及直接调用的 frozen MultiHateLoc、
scoped split/label helpers 和共享 evaluator。

## 裁定

**PASS，可以启动登记的 HateMM/HateClipSeg baseline pilot。** 这是 P-MIL 的跨任务 baseline port，
本裁定只确认代码、数据边界和 evaluation protocol 足以产生可解释的 baseline 数字，不赋予 novelty
结论。

本次 review 修复了会实质改变观察的公式与协议错误；修复后 9 项 synthetic tests、两个 archived
source checkpoint 的 strict structure load，以及 HateMM 真实 negative/positive train-video smoke 均通过。
没有启动正式 pilot，没有读取 test GT，也没有生成正式 test prediction。

## Review 中修复的 blockers

1. **训练/validation split 曾错误重切。** 初版从 official train 内另切 10% validation，与项目 frozen
   official validation manifest 及 source MultiHateLoc protocol 不一致。现完整 official train 只用于
   gradient，official validation labels 只用于每次训练选择 checkpoint；两者 scoped labels 分开读取并
   强制无 overlap。
2. **test producer 顺序过早。** 初版在 P-MIL checkpoint 选择前预生成 test proposals。虽然不读 test
   labels 且这些 proposals 未进入 gradient，这仍不符合 README 的“训练完成后 blind test”。现只在
   训练前生成 train/validation proposals；checkpoint 选定后才在一个 blind loop 中逐视频生成 test
   proposals并立即评分。
3. **三模态 proposal score 存在交叉项。** 初版先分别平均 hate、attention、completeness，再把三个均值
   相乘，会混合例如 visual hate、audio attention、text completeness。现严格计算每模态
   `hate_probability × attention × completeness`，再对三模态平均。video score也改为每模态 attention-
   weighted hate score后再平均。
4. **negative bag 的 completeness 监督相反。** 初版也从 negative bag 的任意高 attention proposal生成
   foreground pseudo instance，迫使至少一个 negative proposal completeness 接近 1，与 background
   supervision冲突。现 positive bag按 PCE生成 IoU pseudo labels；negative bag没有 foreground instance，
   completeness直接回归 0。
5. **官方 P-MIL loss 系数与 target 被改弱。** 对照 CVPR 2023 公式，positive base target 应为 multi-hot
   `[foreground=1, background=1]`，不是归一化后的 `[.5,.5]`；`gamma=.8`、
   `lambda_comp=20`、`lambda_IRC=2`。这些现均已恢复，ramp仍只作用于 PCE/IRC。
6. **optimizer schedule 与官方规模不一致。** 初版每视频立即 Adam step、学习率 `1e-4`；官方设置为
   Adam `5e-5`、bag batch size 10。现以十个变长 bags 顺序 forward/backward、loss除以当前 batch size后
   累积 gradient，再统一 step，保持官方 bag batch而不增加显存峰值。
7. **frame readout 静默接受错误输入。** 初版 `zip` 会静默截断 proposal/score 数量不一致，float bounds
   也会被直接截整。现显式检查 length、shape、数量、有限性、1fps整数 bounds、范围和非空 interval；
   未完整覆盖仍立即失败。

## 1. Producer 与 split isolation

- gradient 路径只遍历 official train IDs，labels 仅来自
  `scoped_video_labels(corpus, "train", train_ids)`。official validation labels通过独立 scoped 调用读取，
  只进入 `validation_ap` 和 epoch checkpoint selection，不进入 train loop。
- train/validation/test ID manifests 在 HateMM 和 HateClipSeg 实查两两无交集。HateMM evaluator test cohort
  通过固定 no-localization exclusion得到 214 videos；HateClipSeg 为 79。producer只读这些 blind IDs和
  frozen features。
- `run.py` 不导入 localization GT，不调用 corpus-wide label loader，也没有 test scoped-label调用。
  test blind loop发生在 `best_state` 恢复后；其输入只有 model、frozen proposal model、test IDs和三模态
  features。
- validation video AP 是唯一 checkpoint criterion；没有 validation localization GT、validation method
  gate或 validation-driven hyperparameter branch。test labels与 temporal GT只由 `run_pilot.sh` 在
  `scores.jsonl` 关闭后调用的 evaluator读取。

结论：训练监督、checkpoint selection 和 blind test producer职责分离正确。

## 2. Frozen MultiHateLoc 与 candidate proposals

- source path固定为 archived corpus-specific official-val MultiHateLoc seed-234 checkpoint。两个实际文件均
  存在且是 raw state dict；HateMM结构解析为 hidden 512/embed 64，HateClipSeg为 hidden 512/embed 256，
  两者 strict load均无 missing/unexpected keys。
- archived train logs显示 checkpoint由 official validation video AP选择。虽然 source run随后曾生成 test
  scores，当前 port只读取 checkpoint参数；其 selected epoch不由 test决定。
- source model的 hidden/embed从 weight shapes解析；dropout、MIL K与temperature不影响 eval-mode fused
  forward。所有 train/validation/test proposals均由同一个 frozen fused frame probability branch产生。
- candidate rule完全冻结：每视频 score min/max间的 9 个相对 thresholds、连通 components、stable top-16
  peaks的固定 `1/2/4/8/16/32/64` 秒邻域、最多 256 proposals。按当前视频 min/max应用固定公式是 blind
  per-video inference，不是用 test performance拟合 threshold。
- priority只由 source score的 interval max/mean、长度和起点构成；whole-video proposal始终保留。
  proposal rule不读取当前 PMIL label，train/validation/test一致。random sampling只发生在 P-MIL train
  loss，不改变 frozen proposal producer。

该 proposal generator 是 README 已明确披露的 hateful-task adaptation，不应表述为官方 P-MIL candidate
generator 的逐字复现。

## 3. SCFE、CAS/attention MIL、PCE 与 IRC

- `_roi` 把时间轴作为 RoIAlign height，并对每个 interval左右各扩 `.25 × width`。`roi_size=12`、edge=2
  形成官方 2/8/2 left/inside/right bins；每区 max-pool后使用
  `[inside-left, inside, inside-right]`，与 SCFE Eq. (4) 一致。视频边界外由显式 zero padding提供，所有
  modalities共享同一 proposal geometry但各自保留独立 LayerNorm/branch。
- 每模态 CAS为 `[hate, background]` 两类 logits。base与 attention-suppressed CAS分别做 proposal top-`M/8`
  pooling，再 softmax。positive base target为 `[1,1]`、suppressed target为 `[1,0]`；negative两者均为
  `[0,1]`，与 action/background公式及真实 negative-bag适配一致。
- positive PCE先对另外两 modalities 的平均 attention使用 `.8 × max` threshold，再贪心保留最高
  attention proposal并删除所有有 overlap的 candidates。每个 proposal的 target是到 pseudo instances的
 最大 temporal IoU；predicted sigmoid completeness用 MSE训练。teacher attention已 detach。negative
  completeness回归零，不产生虚假 foreground pseudo instance。
- IRC仅用于 positive bags。对三模态六个有序 student/teacher pairs，在每个 retained teacher anchor的
  overlapping proposal neighborhood内对 hate-CAS做 normalized distribution KL；teacher distribution
  detach，student保持 gradient。无 retained anchor时返回有图连接的零损失。
- PCE与IRC使用官方渐增 ramp，并分别乘 20与2。synthetic positive/negative backward及真实 train-video
  backward显示 loss与所有实际 gradient有限。

这是 binary tri-modal adaptation：官方只有 action videos与 RGB/flow。本 port 对 negative completeness和
三模态互教的定义已明确，不应称为未经修改的官方复现。

## 4. Validation 与 test evaluation

- 每个 corpus单独初始化 P-MIL model与 Adam，完整 official train训练15 epochs；每 epoch结束在 official
  validation上只算 video AP并复制最佳 state。没有 performance gate或跨 corpus routing。
- best state恢复后立即运行 blind test proposals与唯一 `score_pmil`，随后
  `run_pilot.sh` 调用仓库登记的唯一共享 evaluator：
  `scripts/reproduction_baselines/eval_baseline_scores.py`。
- evaluator split固定为 test并启用 `--require-full-coverage`，直接输出 pooled AP、pooled ROC-AUC与
  within-video macro ROC-AUC。缺视频、多余视频、score/GT长度不一致或非有限 score均会终止。
- baseline没有 SOTA或 validation gate；两个语料的 evaluator原生 `metrics.json` 均完成后，才算 port
  pilot完成。后续 test error analysis属于 iterative/developmental evidence。

## 5. Proposal-to-frame readout

- inference不做 proposal subsampling，使用全部 frozen proposals。每个 proposal confidence在 `[0,1]`，
  为三模态各自 hate probability、attention probability、completeness probability乘积的平均。
- frame score是所有覆盖该 second proposals的最大 confidence。whole-video interval保证 dense coverage；
  constant proposal confidences在 synthetic test中产生逐秒严格平坦 score。
- readout现在验证 proposal/score一一对应、整数 bounds与有限输入。真实 train-only smoke的输出长度分别
  与 5-second negative、2-second positive视频精确一致且全部有限。

## 6. Random proposal sampling

- train视频 proposals超过128时，使用 seeded `torch.randperm`选一个无重复 subset。`forward`同时返回
  `used` bounds；三个 modality branches均只对这一相同 subset做 RoIAlign。
- MIL、PCE与IRC全部接收同一个 `outputs, used`，不存在用 sampled logits配 full proposal geometry的错位。
  synthetic test强制 10→4 subsampling，验证三个 branches长度与 `used`一致且全 loss有限。
- validation/test固定 `training_sample=False`，不会随机丢 proposal。训练 subsampling不保证保留
  whole-video proposal，但 train loss不需要 dense frame coverage；inference完整 proposal集合始终含
  whole-video。

## 7. 内存、数值与长任务风险

- 只读 feature shape统计：HateMM train+validation+evaluator-test共1067 videos、156,082 seconds，长度
  2–5,809（median 108）；HateClipSeg共393 videos、93,487 seconds，长度181–350（median 239）。
- 单次 source forward只处理一个视频；最长 HateMM原始三模态 float32输入约数十 MB。P-MIL train最多对
  128 proposals做三路 12-bin RoIAlign；gradient accumulation顺序释放每个 bag graph，batch size 10不会
  把十个长视频同时留在显存。RTX 5090显存风险低。
- 主要风险是 wall time与磁盘 I/O：15 epochs逐视频重复加载三组 feature，HateMM约11,160 train-video
  forwards，另有每 epoch validation。脚本必须按 detached方式运行并监控 root log；不应在前台 SSH
  session启动。
- LayerNorm、softmax/sigmoid、IoU denominator floor、attention denominator floor、finite feature/readout
  checks降低数值崩溃风险。IRC用有限负 mask而非负无穷，短 proposal neighborhoods也可计算。
- 当前脚本没有自动 resume。中断后不得在同一 run directory把旧 checkpoint/metrics与新运行混用；应
  检查 log并使用新的 run name重启。这是运维限制，不是本次 correctness blocker。

## 8. 输出与运行入口

- `run_pilot.sh`固定使用 `HateVideo` Python并检查可执行性；两 corpus串行运行，任一步失败即停止。
  `launch_pilot.sh`用 `setsid`与 SSH session解耦，pilot root写 `run.log`和`run.pid`。
- 每 corpus输出 readable config/source checkpoint path、proposal diagnostics、model、train/validation
  history、blind scores与 evaluator metrics，全部位于 `runs/`；不写 `data/`。

## 已执行检查

- 9项 synthetic unit tests全部通过：proposal determinism/bounds/whole coverage、constant-score readout、
  readout错误输入、scoped label source、finite forward/backward、positive multi-hot target、negative
  completeness、per-modality product score、sampled proposal/loss一致性。
- 两 corpus archived MultiHateLoc state dict strict load通过。
- HateMM真实 train-only negative/positive视频：feature load、source fused scoring、proposal generation、
  SCFE/P-MIL forward、完整 loss backward、proposal/frame inference全部通过，frame length与有限性正确。
- Python compile和两个 shell scripts syntax通过；未访问 test GT。

最终裁定：**PASS FOR FORMAL BASELINE PILOT**。
