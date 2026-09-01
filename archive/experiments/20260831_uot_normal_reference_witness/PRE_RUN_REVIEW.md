# PRE-RUN REVIEW — typed shared-capacity normal-reference witness

**日期：2026-08-31**  
**范围：** `README.md`、`model.py`、`protocol.py`、`train.py`、`evaluate.py`、`test_model.py`  
**结论：PASS。** 已发现的两个 blocker 均已修复，快速 CPU 测试通过。未启动正式 GPU 训练，未生成正式 prediction。

## Blocker 及修复

1. **原 shared capacity 耦合了没有共同语义的 atom 编号。** 原实现为每个模态维护独立
   `[M,K,D]` reference bank，却按相同 `k` 汇总跨模态 load；三个 bank 可独立置换，因此该共享容量
   可能只是任意数组编号碰撞。现改为一个 `[K,D]` latent normal atom bank，audio/visual/text 使用各自
   projector 进入同一 embedding space。atom identity 现在确实共享，typed 差异由 projector 保留。
2. **正式 evaluator cohort 不完整。** 原 exclusion 只覆盖 HateMM，遗漏 MHC-EN 与 MHC-ZH 各 4 个
   test manifest 中无 localization gold 的视频，导致 `evaluate.py` 必然拒绝正式 artifact。现已把四语料
   frozen test manifest 与 gold 逐项对齐，并将可读的固定 exclusion ID 写入 `protocol.py`。

当前没有剩余 blocker。

## 1. Transport 数学、shape、梯度与 controls

- typed cost 为 `[B,T,M,K]`，mask 为 `[B,T]`。每个有效 time×modality source row 在 K 个 normal
  atoms 与一个 reject state 之间归一化，row mass 为 1；padding 的 reject 与最终 score 均为 0。
- shared 分支对 `(T,M)` 汇总每个 atom 的 load，上限为 `N_valid/K`；independent control 只在每个
  modality 内沿 T 汇总，上限为 `T_valid/K`。reject 不设容量，因此不是固定 unmatched 比例。
- nearest-normal control 不做 column-capacity projection，只使用相同 cost 的 normal kernels 与 reject
  kernel。它与 independent/shared 两个 transport 分支均有实际计算差异。
- synthetic collision case 明确得到 `shared != independent`。另在两个 HateMM train 视频和两个
  HateClipSeg train 视频上做未训练 CPU forward：core 相对 independent 的 mean absolute score delta
  分别为 `.12116/.09261`，相对 nearest 为 `.09501/.05581`。controls 未退化为重复输出。
- 所有 transport 操作可微；短程 30-step synthetic optimization 的 loss 下降，全部参数梯度有限。
  temperature、维度、步数、mask 和 feature shape 已增加显式输入检查。
- controls 是同一已训练 checkpoint 的 inference ablation，因此严格隔离 shared-capacity readout；不是
  三个独立训练模型。冻结 gate 只用 core 是否超过两 control 的 within-video ROC 判定容量机制是否
  load-bearing。

## 2. Negative-only normal references

- `reference_gradient_mask=(labels == 0)` 在逐 batch-item reference path 上执行：negative item 使用正常
  reference，positive item 使用 detached reference。
- positive-only bag loss、temporal loss和 positive features 对 normal atom bank 的梯度严格为 0；
  negative bags 的 BCE 与 negative compactness 对 bank 产生非零梯度。两项均有自动测试。
- positive bags仍可更新 modality projectors，使无法由 frozen-for-positive normal bank 解释的 token
  进入 reject；不能把 normal bank 改成 positive prototype。
- transport load 按视频独立计算，不跨 batch item，共 batch 的 positive item 不可能通过容量影响
  negative item 的 reference gradient。

## 3. Split、HCS producer 与 test 时序

- 四语料 train/validation/test ID 两两隔离；train/validation labels 只从对应 scoped video-label JSON
  读取，coverage 与二值类型均严格检查。
- `aligned_local_features` 只生产 audio/visual/text 特征。静态调用审计确认 `train.py` 不调用 temporal
  gold API，也不调用聚合全语料标签的 API；HateClipSeg producer 不读取 segment annotations、test
  labels 或 temporal GT。
- validation 采用 frozen validation manifest，只计算 video AP 选择一次 checkpoint。已修正文档和
  config 中原来不准确的“stratified 10% carve”描述。validation 不参与方法比较或设计。
- best checkpoint 保存并写出 `training_complete` marker 后，代码才进入 test feature prediction；
  prediction 阶段使用全 0 placeholder labels，既不加载 test video labels，也不加载 temporal GT。
  `evaluate.py` 是唯一读取 test gold 的本实验文件，并只接受状态为 `prediction_complete` 的 run。

## 4. Exact cohort、共享 evaluator 与 SOTA gate

- 四语料 `evaluator_test_ids(...)` 与 `gt_arrays(corpus, "test")` 的 key 集合完全一致，顺序来自 frozen
  test manifest；duplicate、missing、extra、score/GT length、有限值和 `[0,1]` 范围均硬失败。
- `evaluate.py` 不实现指标，唯一调用
  `scripts/reproduction_baselines/eval_baseline_scores.py::evaluate_scores`。
- 固定输出 pooled AP、pooled ROC-AUC、within-video macro ROC-AUC。所有 gate 使用严格 `>`，不是
  `>=`。
- HateMM gate 精确为 `.5938315566328208/.8161837922270064/.631531717970362`；HateClipSeg 为
  `.6193710949898349/.6050224699167533/.5619078936355938`。同时要求 core within 严格超过 independent
  与 nearest controls。任一语料失败即停止，不扩 MHC-EN/ZH。

## 5. 输出、provenance、恢复与长任务安全

- run 目录写 `config.json`、`code_version.txt`、每 epoch 原子 `training_state.pt`、最终 `model.pt`、
  `train_history.json`、`training_complete.json`、`predictions.jsonl`，评测后写 `metrics.json`。
- config 记录可读的 train/validation/test split、scoped label、feature producer 路径、参数、语料与
  test-isolation 声明。新 run 不允许覆盖已有正式 artifacts。
- `--resume` 要求同一 config、代码版本说明与 training state；恢复 model、optimizer、best checkpoint、
  history、随机状态与 DataLoader generator state。中断发生在 training 完成后的 prediction 阶段时，
  也可从 completion marker 恢复。
- README 的正式命令使用 `setsid`、独立 `run.log` 与 `run.pid`，符合 SSH 解耦要求。

## 6. 数值与学习可行性

- cosine cost 范围受控；默认 temperature `.10`、reject cost `1.0` 对应有限 kernels。所有分母均有
  下界，mask 至少含一个有效秒，bag probability 在 BCE 前限制到开区间。
- synthetic padding、有限值、范围、positive/negative reference gradient、shared/control 区别及短优化
  均通过。真实 train 特征的 CPU forward 也产生有限且非恒定的 core/control scores。
- 该审查只确认“可以学习且不会因明显 shape/数值错误崩溃”，不预判正式 test performance。

## 快速测试

运行：

```bash
/home/jehc223/miniconda3/envs/HateVideo/bin/python -m unittest \
  experiments/20260831_uot_normal_reference_witness/test_model.py -v
/home/jehc223/miniconda3/envs/HateVideo/bin/python -m py_compile \
  experiments/20260831_uot_normal_reference_witness/*.py
```

结果：10/10 tests passed；所有目标文件语法检查通过。

## 最终裁定

**PASS。** 可以按 README 启动冻结的 seed-234 HateMM 与 HateClipSeg pilot。该裁定只授权既定实现与
既定 test-first gate，不授权根据后续 test 数字调参、选择语料分支或改变 control/gate。
