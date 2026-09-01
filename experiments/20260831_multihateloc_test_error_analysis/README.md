# MultiHateLoc test error analysis

截至 2026-08-31。只读诊断；不训练、不选择 checkpoint、不生成候选 prediction。

## 目的

按当前 test-first 规则，读取 MultiHateLoc 三个 seed 的四语料 test predictions、统一
evaluator 输出和 test GT，回答三个与下一轮训练机制直接相关的问题：

1. 固定 `ceil(T/3)` MIL witness count 是否与真实正例 span occupancy 的偏差共同出现；
2. video-global Dynamic Modality Selection 是否真的选中 within-video 排序更好的模态；
3. fused 分支相对单模态的收益或损失是否跨语料一致。

test GT 只用于 error analysis。它不参与梯度、checkpoint selection、pseudo-label、
threshold 或 inference rule。受本分析影响的后续 test 数字均属于
iterative/developmental evidence。

## 输入

- MultiHateLoc test artifacts：
  `/home/jehc223/Hate-follow-up/results/reproduction/official_val/final/multihateloc/`
- 固定 test GT：`/home/jehc223/Hate-follow-up/results/reproduction/gt/`
- 指标只读取每个 run 已由共享 evaluator 生成的 `frame_eval.json`；本目录不复制评测逻辑。

## 输出

`runs/20260831_multihateloc_test_error_analysis/main/metrics.json`

其中 `best_branch_oracle` 和 DMS selector agreement 都是 test-label-informed diagnostic，
不得作为模型结果、候选后处理或可部署机制。

## 结论

权威输出：`runs/20260831_multihateloc_test_error_analysis/main/metrics.json`。
四语料 × 三 seed 的 test coverage、score/GT alignment 与 finiteness 全部通过；独立
pre-run review 见 `INDEPENDENT_REVIEW.md`。

- primary fused 的 pooled AP / pooled ROC / within ROC 均值：HateMM
  `.49602/.72875/.63153`，MHC-EN `.37216/.65336/.57426`，MHC-ZH
  `.36652/.65213/.51205`，HateClipSeg `.53905/.52441/.52423`。
- DMS 最高权重与 test-GT 最佳单模态相符的 seed-video 比例仅为
  `.216/.333/.375/.323`。HateMM/HCS 各 255/201 个 eligible seed-video pair 中，
  DMS 分别有 252/198 次选择 visual，却不存在相应的 visual oracle 优势。
- fused 超过全部单模态的比例仅 `.345/.159/.042/.154`；best-branch test oracle
  相对 fused 的 within AUC 缺口为 `.106/.171/.211/.106`。Oracle 只证明存在尚未被
  DMS 捕获的 modality responsibility，不授权 test routing 或 ensemble。
- `ceil(T/3)` witness fraction 与真实 occupancy 的偏差没有形成统一的负相关失败：
  HMM/HCS 的低 occupancy strata 反而最差，MHC-ZH 仅 8 个 eligible 视频。
  因此不进入 adaptive-cardinality/top-K 方向。

设计影响：POWA 不再是默认 backbone；从 MultiHateLoc 出发，下一候选只针对一个核心
错误——正视频标签被复制给每个 modality branch，加上全视频 unconditional alignment，
导致 video-global DMS 学成 visual shortcut。候选必须在训练中学习 time×modality 的
latent witness ownership，并在 test 输出单一 fused localizer；不得使用 test-GT branch
selection、按语料路由、score ensemble 或 calibration。

## 运行

```bash
bash experiments/20260831_multihateloc_test_error_analysis/run.sh
```
