# MultiHateLoc test error analysis：独立运行前评审

日期：2026-08-31  
评审阶段：正式运行前  
结论：**PASS，可以正式运行。**

## 评审范围

- `analyze.py`
- `run.sh`
- `README.md`
- 四个主语料、三个 seed 的既有 MultiHateLoc test predictions、`frame_eval.json` 和
  `train_log.json`
- MultiHateLoc 的 `data.py`、`model.py`、`train.py` 及唯一共享 frame evaluator

本评审只进行了只读检查和内存复算，没有正式执行 `analyze.py`，也没有生成本实验的
正式结果目录。

## 发现并在正式运行前修复的三点

1. **DMS modality 顺序错位。** 初版分析中的 score branch 顺序是
   `audio, visual, text`，而模型的 DMS weight 顺序是 `visual, audio, text`，导致
   selector agreement 把 audio 与 visual 对错位置。修订版将
   `MODALITY_BRANCHES` 统一为模型顺序，并删除后续二次映射。
2. **top-third occupancy mismatch 未使用真实 top-K 取整。** 模型实际使用
   `ceil(T/3)`，初版却固定与 `1/3` 比较。修订版逐视频记录
   `top_k_count = ceil(T/3)` 和 `top_k_fraction = top_k_count/T`，所有 mismatch、
   correlation 和逐视频记录均使用真实 fraction。
3. **并列最优 modality 的 selector agreement 处理不正确。** 初版 `argmax` 只认
   model order 中第一个并列项。修订版把 DMS 选中任一并列最优 modality 都计为正确，
   单列 tie pair 数；用于展示的 best-modality counts 明确采用固定 model-order
   tie break。

## 数据、评测与覆盖证据

共检查 12 个 corpus/seed artifacts：四个语料各三个 seed。每个 artifact 均满足：

- evaluator artifact 明确标注 `split=test` 和正确 corpus；
- evaluator 记录的 score 文件路径与分析实际读取的文件一致；
- predictions video IDs、test GT IDs 和 DMS test state IDs 完全一致；
- 六个 score branches 全部存在，每个视频的 score shape 与 1 fps GT 完全一致，且全部为
  finite；
- 使用唯一共享 `evaluate_scores` 在内存中复算六个 branches 后，pooled ROC-AUC、pooled
  AP、within-video macro ROC-AUC、逐视频 AUC、视频数、帧数和 missing/extra 计数均与既有
  `frame_eval.json` 一致。

覆盖规模：

| corpus | test videos | test frames | seeds checked |
|---|---:|---:|---:|
| HateMM | 214 | 29,269 | 3 |
| MHC-EN | 158 | 5,601 | 3 |
| MHC-ZH | 153 | 4,818 | 3 |
| HateClipSeg | 79 | 18,839 | 3 |

另外，用每个视频记录的 `visual, audio, text` DMS weights 重新组合三个 modality score，
得到的逐帧结果与存量 `score_dms` 一致；仅存在源 JSON 六位小数写出造成的舍入误差。
这同时验证了 weight 顺序、score branch 顺序和 train-log/test-score 对应关系。

## 计算与结论语义

- 所有 performance 数字只读取四语料 test 上既有的共享 evaluator 输出；分析不读取或比较
  validation performance。历史 validation 只承担原 baseline 训练内部的 checkpoint
  selection。
- occupancy correlation 先对每个视频跨三个 seed 求 fused AUC 均值，再以视频为独立单位
  计算 Spearman correlation，未把 seed-video pair 当作独立样本。
- occupancy strata 中每个视频具有相同的三个 seed，因此 seed-video pair 平均不会给某些
  视频额外权重。
- best-modality oracle、best-branch oracle 和 DMS selector agreement 明确使用 test GT，
  仅作为 Rule-10 iterative/developmental error analysis，不是 prediction、候选后处理或
  可部署选择规则。
- 本实验不计算、记录或依赖内容校验标识。

## 非阻断措辞提醒

`stable_worst_10` 和 `stable_best_10` 实际按三个 seed 的 **mean fused AUC** 排序，并同时
报告 sample SD；它们没有按低方差筛选，也不表示统计显著的稳定最好或最差。正式解读时应
将其理解为“按三 seed 均值排序的 best/worst 个案”，不要由字段名推出稳定性结论。

