# 跑前基础 Technical Review — Lexically Anchored DCC

截至 2026-09-01。审查对象现归档于 `archive/experiments/20260901_lexically_anchored_dcc/`
以及 `scripts/reproduction_baselines/powa_macil/model.py` 中为本候选暴露既有
`shared_rep` 的改动。本记录是正式运行前唯一一次基础 technical review；没有运行 smoke、
训练、缩小数据或缩短 epoch。

## 裁定

**PASS。没有剩余的 result-affecting blocker，可以进入完整 validation hyperparameter
search、checkpoint selection 和 HMM/HCS test evaluation。**

## 限定范围内的核查结果

### 1. 机制进入 shared representation 与 final score：PASS

- `POWAMACIL` 暴露的 `shared_rep` 是 PEF `fuse` 的既有输出；同一张量随后进入
  `primitive_head`、hostile/target binding、policy compiler 和最终 `frame_prob`。
- `CrossVideoRegionMemory` 直接对 region mean 做 `F.normalize` 后计算 class-region
  contrast，不设置独立 learned projection head。Contrastive gradient 实测非零地进入
  `pef.fuse`，因此会更新 final policy head 的上游共同表示；memory 没有 inference readout。
- `anchor` arm 将 contrast 项严格置零。`aligned` 与 `shifted` 使用相同 POWA forward、
  原 POWA loss、memory 和 contrast 实现，唯一 control 差异是 lexical/speech timing 的
  half-video circular shift。
- 当前权威 POWA starting point 是 mask-fix 版本。MACIL backbone 与 text attention 的
  `valid_mask` propagation 在本候选前已经属于该 starting point，不是本候选机制改动，必须并已
  保留。本候选在共享模型上的实际新增只有：PEF 多返回既有 `shared`、caller 解包它、输出字典
  新增 `shared_rep`；没有新增参数，也没有改变已有 score 公式。

### 2. Split isolation 与 OOF lexical producer：PASS

- 实际 evidence artifact 来自
  `runs/20260831_lexical_posterior_regularization/stage_a_fix2/evidence/`。生成代码对 HMM/HCS
  分别执行五折 `StratifiedKFold`；每个 train video 只由不含该视频的 fit fold产生 lexical
  evidence。Producer 只读取本语料 train video labels、train ASR文本和时间戳。
- 实际解析检查确认 HMM `744/744`、HCS `251/251` 个 train video 被精确覆盖；evidence key
  与当前可用 train cohort完全一致，train/validation 与 train/test ID交集均为零。
- 候选 dataset只打开各语料的 `train_evidence.npz` 与 `train_speech.npz`。artifact目录中即使
  存在旧实验生成的 val/test evidence，本候选训练代码也没有读取路径。
- Validation GT只进入每 epoch 的 metrics 和 checkpoint选择；test GT只在锁定配置后的
  test cohort/alignment检查及统一 evaluator中读取，不进入梯度、超参数或 checkpoint选择。

### 3. Lexical、POWA时间网格、crop与标签对齐：PASS

- 实际解析检查确认所有 HMM/HCS train evidence 均为一维有限数组，speech shape完全一致，
  且每个数组长度与该 video 的 canonical `n_seconds` 完全相等。
- Evidence先从明确的 1 fps interval grid resample到该视频的 POWA snippet grid，再使用与
  visual/audio/text完全相同的 deterministic `process_feat(..., is_random=False)`。当序列超过
  `max_seqlen=200` 时，各流使用相同 uniform index规则；不足时使用相同右侧 padding。训练
  `valid` mask来自同一个 visual sequence length，padding不进入 admission或contrast。
- 五个 visual crops共享同一视频的 audio、text、lexical、speech与稳定 `video_index`；crop
  item按 `index // crop_repeat` 映射回正确 video，没有 crop/video错配。
- Shifted control在 resample之后、padding/截取之前，仅对有效未填充的 lexical与speech序列
  做确定性半视频 circular roll，padding不参与 shift。

### 4. Asymmetric admission、memory与 same-video exclusion：PASS

- 只有 `label=1` 且有 speech support、达到预注册 quantile的时间点进入 hateful regions；
  positive-unselected 时间点不会进入 benign memory，也不形成反向 pseudo-label。
- Benign admission严格为 `valid & label<0.5`，因此只来自 negative train bags；按固定 width
  切 region，不读取 positive bag 的未选时间点。
- 当前 batch与历史 FIFO memory的候选均携带稳定 `video_index`。每个 anchor的 numerator和
  denominator都应用 `candidate_id != anchor_id`；针对性单元检查确认 same-video candidate
  被排除，而存在跨视频同类及异类candidate时 contrast loss和anchor gradient均为非零。
- Memory entry在 enqueue前 detach，历史视频不会保留计算图；当前 anchor保持梯度。Memory
  只在训练调用，未写入 model checkpoint，test inference不实例化也不读取它。

### 5. Validation search、checkpoint与 matched control：PASS

- 每语料运行 `2 learning rates × 3 contrast weights × 2 support quantiles = 12` 个 aligned
  full-epoch trial，另有两个相同 learning-rate 的 no-contrast anchor；不存在 video/epoch limit。
- 每个 trial按 validation `within_roc → pooled_ap → pooled_roc → earlier epoch` 选择 checkpoint。
  跨 trial先要求 aligned 相对同 learning-rate anchor 的 pooled AP与ROC delta均不低于
  `-.005`，再最大化 validation within；若无 feasible配置，按两项 pooled delta的较小值、
  within依次选择。README中的符号已针对性修正为与代码和 formal config一致的 `-.005`。
- 两个语料的 core配置都锁定后，shifted control才使用各自 selected learning rate、contrast
  weight与support quantile进行完整训练；它使用同一 checkpoint rank规则。正式 test同时评测
  matched anchor、shifted与aligned，不以 validation决定方法晋级或淘汰。

### 6. Test inference 与 canonical evaluator：PASS

- `infer.py`只加载 validation-selected POWA `model.pt`，输出单一 raw `frame_prob` 的五 crop
  mean；不读取 lexical posterior、region memory、teacher score、校准、routing或其他模型分数。
- Prediction使用 `PowaTestDataset` 的 canonical second-to-snippet `index_map` 回到 1 fps；每个
  video同时检查 score长度等于 `n_seconds` 和 gold长度。
- `run_formal.sh`对 HMM/HCS 的三个 arm统一调用
  `scripts/reproduction_baselines/eval_baseline_scores.py`，指定 `split=test`、固定 branch
  `score_method`、`--require-full-coverage`，直接生成各自 `metrics.json`。`summarize.py`只读取这些
  evaluator输出，不重新实现任何指标。

## 执行过的非训练检查

- Python syntax compilation：PASS。
- `bash -n` 检查 `run_formal.sh` 与 `launch_formal.sh`：PASS。
- `test_method.py` 的 asymmetric admission、positive abstention、memory与shared gradient测试：PASS。
- 针对性 same-video exclusion / cross-video active-loss测试：PASS。
- 针对性完整 POWA forward + region contrast backward测试：contrast gradient进入 shared
  representation producer，且不存在独立 projection：PASS。
- 实际 HMM/HCS OOF archive 的 cohort、split isolation、shape、finite value和时间长度解析：PASS。
