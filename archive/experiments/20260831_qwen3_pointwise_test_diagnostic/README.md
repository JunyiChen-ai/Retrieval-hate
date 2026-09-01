# Qwen3-VL pointwise test diagnostic

> **取消并归档：process review 裁定为 duplicate-after-test diagnostic。** 已有 `runs/20260831_qwen3_test_teacher_diagnostic/formal/metrics.json` 使用同一 Qwen3-VL、16秒/8秒stride完成HMM/HCS positive test localization，within仅`.561760/.539628`且零生成故障，已足够否定pointwise observation premise。本目录未运行、未生成任何run artifact；不得用2帧或choice-token readout重开同一失败输入源。

截至 2026-08-31。状态：process review后取消；未运行正式test。

## 目的与边界

这是 target-task 已有 LELA-style pointwise VLM scoring 的 baseline/diagnostic，不是 novelty候选，也不算方法迭代。它回答一个当前缺失的问题：同一 frozen Qwen3-VL local observation 在完整 HateMM/HateClipSeg test 上是否有足够的三指标上限，值得继续设计新的训练机制。

所有 performance evaluation直接在test。没有训练、validation、checkpoint selection、跨主数据集数据、span supervision、ensemble、calibration或routing。Test GT只由共享 evaluator读取；生成器不读取GT labels，只读取 frozen test manifest来确定cohort，并从feature length取得1fps长度。

## 冻结配置

- model：`Qwen/Qwen3-VL-8B-Instruct`，本机已有权重；
- windows：16秒宽、8秒stride；每窗固定取2帧并加入timestamped ASR；
- prompt：只判断当前window，区分protected-group hate与generic profanity；
- readout：下一token在整数`0..10`对应token上的logits，只在这11个choice内部softmax，取期望后除以10；不自由生成、不解析文本、不调threshold；
- densify：覆盖同一秒的window score做算术平均；
- corpora：先完整HMM与HCS；同一配置。

输入媒体来自`data/frames_1fps/`与`results/reproduction/asr/`；`data/`只读。输出写入`runs/20260831_qwen3_pointwise_test_diagnostic/<corpus>/`。

## 评测与结论规则

生成结束立即调用唯一共享 evaluator `scripts/reproduction_baselines/eval_baseline_scores.py`，输出`metrics.json`。必须100%覆盖frozen test cohort，并报告pooled AP、pooled ROC、within-video macro ROC。

这组结果无论是否SOTA都只能作为developmental test diagnostic：

- 若HMM/HCS三指标均有接近或超过SOTA的local observation，再生成不属于pointwise VLM/ensemble/calibration/KD的新候选并重新novelty审查；
- 若任一语料within接近chance或pooled严重失败，则停止围绕Qwen pointwise observation包装新方法。

不得用这个baseline本身claim novelty或晋级。
