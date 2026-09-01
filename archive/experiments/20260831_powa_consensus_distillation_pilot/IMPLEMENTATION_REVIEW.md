# 正式训练前独立实现与评测审查

**截至 2026-08-31；结论：PASS。**

本结论审查当前工作树中的 `train.py`、`prepare_vera_k16.py`、
`test_teacher_diagnostic.py`、`test_method.py`、`src/powa_residual.py`，以及实际 import 的
`powa_macil`/`macilsd` dataset、alignment、model 与唯一共享 evaluator
`scripts/reproduction_baselines/eval_baseline_scores.py`。审查未修改实现代码，也未计算、记录、
比较或依赖任何哈希、checksum 或 digest。

## 裁定

当前代码没有仍会改变该 pilot 观察或结论的已知实现错误，可以按以下顺序启动：

1. 用 `prepare_vera_k16.py` 的同一固定 recipe 生成 HMM/HCS test K16 cache；
2. 运行 `test_teacher_diagnostic.py`；
3. 只有其权威输出 `runs/20260831_powa_consensus_distillation_pilot/test_teacher_diagnostic/analysis.json`
   中 `continue_to_student=true`，才生成 HMM train K16 cache并启动学生训练；否则直接停止；
4. HCS train 只复用已审计的 238-video cache；HMM 与 HCS 分语料运行五个训练臂，validation
   只选各臂 checkpoint，选定后同一进程立即跑 test 三项固定指标。

这里的 PASS 是代码与评测协议放行，不是 performance PASS。正式 test K16 cache、teacher
diagnostic 和学生训练在本次审查时均尚未运行；其结果不能由本审查预先保证。

## 关键审查结论

### 数据与监督隔离：PASS

- HMM/HCS 的 student、audio probe 与 POWA anchor 均按 corpus 分开；训练 teacher 只索引该
  corpus 的 train IDs 和 video-level labels，不读取 train span GT。
- audio teacher 是严格 video-level 5-fold OOF：每个 train 视频的 teacher score由不含该视频
  的 fold model产生；test diagnostic 则只用全体 train 视频拟合 audio model后预测 test。
- validation GT 只参与每个训练臂五个 epoch之间的 checkpoint selection。test 在选定 state
  后立即评测，不参与梯度或 checkpoint selection。
- test teacher diagnostic 明确属于 Rule 10 developmental evidence。test GT只由共享 evaluator
  计算三指标并决定是否继续该候选，不进入 audio/VERA 预测、学生梯度或 checkpoint selection。
- 代码内固定 hate-domain prompt，不在运行时读取 validation prompt-selection 产物；历史
  validation exposure 已在 README/plan 披露，不冒充未揭盲预注册证据。

### VERA、audio 与时间网格：PASS

- K16 producer 对 HMM train/test 与 HCS test 使用同一模型名、固定五问 prompt、
  `torch_attention`、10 秒窗口、batch 2 和相同的
  `ceil(min(audio_1fps_length, media_duration))` support/index rule。
- resume 会实际解析每个 JSON，并逐字段核对 corpus、split、模型版本说明、prompt、batch、
  window、K、媒体路径、duration、start/end、score 和 response；不以文件名或缓存存在性放行。
- test cache cohort与共享 evaluator一致：HMM 214、HCS 79。HMM train 为 744；HCS train 为
  238 个 label-free media-audit 可解码视频，统一排除 13 个不可解码视频。
- diagnostic 直接读取实际 batch-2 K16 segments，不再从旧 dense VERA score抽样。audio 与 VERA
  order 都先经过训练同款 uniform fixed-200 grid，再线性 lift 回 native snippet grid后组成
  audio、VERA 与 consensus transport branch。
- 对既有 HCS train cache做了实际解析审计：238 个文件均属于 train，13 个排除项之外 coverage
  完整；固定 prompt、模型名、backend、K、每段 start/end 和二值 score 全部符合当前 recipe。
  审查不依赖该缓存内的旧校验字段。

### residual、padding、crop 与冻结 anchor：PASS

- `load_corpus_powa` 要求 corpus-specific checkpoint；已核对 seed-234 HMM/HCS anchor 的
  `train_meta.json` 均只包含各自 corpus，当前 checkpoint 均走 strict state load。
- POWA 所有参数 `requires_grad=False`，wrapper 的 `train()` 会把 POWA重新置为 eval；optimizer
  只持有 residual-head 参数，teacher artifact不进入推理图。
- 训练与推理 residual context 均在同一 fixed-200 grid。test 另跑一次 native dense POWA作为
  原始 anchor，将 fixed-grid residual 线性 lift 到 native grid、逐 crop重新减均值后再直接加到
  dense anchor logit；最终仍按五 crop 概率均值和共享 snippet-to-1fps map导出。
- residual temporal conv按每个样本自己的真实有效长度做 replicate boundary，不再让短视频尾部
  依赖同 batch 的最长视频。真实/随机张量回归确认同一有效序列 padded 与 unpadded 输出一致。
- zero-init output 保持 anchor identity。对完整 HCS validation 63 个视频实测，逐帧
  candidate/anchor 最大绝对误差为 `2.3842e-7`，通过 `1e-6` 门；覆盖与 GT 完整一致。

### pair、control 与 loss：PASS

- core、shuffled、audio、VERA 共用由 consensus gap冻结的 endpoint候选；shuffled 只改变
  direction，不在 shuffle 后重新 threshold。
- source tie现在不产生 pair direction。真实 HCS 238 cohort审计曾发现旧实现中 VERA 对
  `55,652` 个 core endpoints中的 `40,492` 个打平；当前代码会丢弃这些无偏好 pair，避免把它们
  变成“较晚时间高于较早时间”的位置 teacher。文档也已将 audio/VERA臂准确限定为
  conditional-direction controls，并要求报告 active coverage。
- pair softplus、zero-residual Smooth-L1、video MIL、有效长度 mask与 crop temporal index一致。
  使用两个真实 HCS train 视频跑过一个完整 CPU batch：loss有限、backward/optimizer成功，
  residual head发生更新；POWA 的可训练参数数和梯度 tensor数均为 0。

### test score与共享 evaluator：PASS

- `score_split` 对 val/test frozen GT coverage fail-closed，并逐视频检查导出长度与 1fps GT长度；
  `scores.jsonl` 同时保存 `score_candidate` 与 `score_anchor`。
- `metrics.json` 直接保存共享 `evaluate_scores` 的完整输出，包含 pooled ROC-AUC、within-video
  macro ROC-AUC 和 pooled AP；没有复制、重写或改变指标实现。
- test gate现使用 test 上的权威 baseline/SOTA 三指标门，并另要求超过同 seed-234 POWA anchor；
  已删除把旧 validation anchor数值当作 test门的错误。

## 实际执行的检查

- `py_compile`：实验目录全部 Python 文件与 `src/powa_residual.py` 通过。
- `test_method.py`：12/12 PASS，覆盖 percentile、gap pair、方向 loss、source tie、MIL padding、
  differentiable zero、padding invariance、long-video fixed-grid/lift、diagnostic fixed-grid，以及
  transport 的全 tie恒等、部分 tie内 anchor排序保持、无 tie严格 rank assignment。
- 共享 evaluator CPU smoke：全部 PASS，包括 pooled ROC/AP、within-video macro、缺失视频报告与
  score/GT length rejection。
- HMM 744 个 train媒体逐个用正式 `ReusableVideoReader` 打开：744 成功、0 失败。
- 两语料的 train/val/test 交集实测为 0；HMM/HCS val和冻结 GT coverage完整，test cache cohort按共享 evaluator
  的 GT cohort固定。
- 真实首 batch backward与完整 HCS validation epoch-0 identity检查均通过，如上所述。
- Transformers 新版兼容补丁经过真实 InternVL2-8B batch-2 推理：动态兼容类成功提供
  `GenerationMixin`，缺失的 `GenerationConfig` 从原模型 config初始化，并明确保持 InternLM2
  原有 legacy KV-cache语义；模型保持 eval，两个真实 10 秒窗口均完成生成并返回可解析响应。
  补丁未关闭 cache，也未改变模型权重、prompt、batch或 generation参数。
- `test_audio_only_diagnostic.py` 通过独立审查：audio probe只用同语料完整 train label拟合，
  test order严格走 fixed-200→native lift→1fps，POWA test score multiset由共享 tie-neutral
  transport原样保留，三指标直接调用唯一共享 evaluator。随机 80 组输入的 multiset最大误差为
  0。真实 test audio order存在不可忽略的 tie，因此共享 transport已改为在 teacher tie内保留
  anchor原排序；全 tie时严格返回原 anchor，不再引入时间索引先验。该脚本和正式 core
  diagnostic均可按 Rule 10 developmental evidence运行。

## 正式运行约束

- 必须先完整运行 producer；diagnostic 的 cache路径固定为
  `runs/20260831_powa_consensus_distillation_pilot/teacher_cache/{hatemm,hateclipseg}_test/raw/`。
- HCS train必须使用已审计的
  `results/reproduction/official_val/final/vera/hateclipseg/seed_234/train_sparse_k16/raw/`；HMM train
  必须使用本 producer 完整生成并复核的 744-video root。
- 正式训练必须使用 plan冻结的 seed、anchor、batch、epoch、loss和 optimizer参数；所有 arm使用
  相同 corpus cohort与 VERA root。任何改动都形成新实验，不继承本 PASS。
- cache生成与训练均为长任务，按项目规则用脱离 SSH 的后台进程运行，并把日志与 PID写到对应
  `runs/` 目录。
