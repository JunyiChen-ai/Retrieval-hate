# Teacher-scale transfer diagnostic

> 最终裁定：fivefold video-heldout scale-transfer premise PASS；本轮是 developmental test diagnostic，不是方法或 SOTA claim，已归档。

截至 2026-08-31。独立 pre-run review、正式运行与 post-run 重算均完成。

## 问题

`runs/20260831_universal_teacher_simplex_diagnostic/main/metrics.json` 在完整 test-frame pool 上分别对 lexical、POWA、VERA、MultiHateLoc 做 global ECDF，再找到 7 个 HMM/HateClipSeg 共同 all-SOTA 权重 tuple。该结果是有效的 developmental inference upper bound，但完整 test 分布参与了 score-scale mapping，不能证明统一关系能迁移到未见视频，更不能证明 train-only student target 可学。

本轮只检查这 7 个已经冻结的 tuple 对 scale mapping 的依赖，不重新搜索权重，不提出方法或 novelty claim。

## 冻结设计

- 输入、branch、seed、test cohort 和 SOTA 门与 universal simplex diagnostic 完全相同。
- 权重只允许原 artifact 中冻结的 7 个 joint tuples；代码会逐项核对，不接受命令行替换。
- 对每个 corpus/signal 构造三种不看 GT 的 mapping：
  1. `fivefold_video_heldout_ecdf`：按排序后的 video ID 轮转分成 5 folds；每个 fold 的每个分数只用其他 4 folds 的 frame scores 建 empirical mid-CDF。这模拟 scale map 对未见视频的迁移，target video 不参与自己的 mapping。
  2. `per_video_ecdf`：只用目标视频自身 score multiset；这是无需 corpus reference 的部署可用 control，但会删除跨视频 offset。
  3. `raw_identity`：完全不做 mapping，直接使用原 branch 数值。
- 每个 tuple 调用唯一共享 `evaluate_scores`，报告完整 test pooled AP、pooled ROC、within-video macro ROC。
- 主 gate：`fivefold_video_heldout_ecdf` 至少保留一个完全相同的 tuple，使 HMM/HCS 都三项过 SOTA。若失败，关闭“现有 shared tuple 可直接迁移为 train-only target”的解释；不得靠重新搜索 crossfit 权重救回。
- `per_video_ecdf` 与 `raw_identity` 只做 scale-sensitivity controls，不决定主 gate。

这是允许的 test-informed error analysis。GT 只由共享 evaluator 和 SOTA gate读取，不参与 mapping、训练、gradient 或 checkpoint selection。本轮没有 student training；即使 gate 通过，也只允许继续寻找新的、非普通 KD 的 target-construction mechanism。

正式输出固定为 `runs/20260831_teacher_scale_transfer_diagnostic/main/metrics.json`。

## 正式结果

主 gate PASS。原 7 个冻结 tuple 在两个语料的 `fivefold_video_heldout_ecdf` 下全部仍为 all-SOTA：

- HateMM：7/7；
- HateClipSeg：7/7；
- 完全相同 tuple 的双语料 joint：7/7。

代表 tuple `[.10,.25,.40,.25]`（lexical、POWA、VERA、MultiHateLoc）：

- HateMM AP/ROC/within = `.597022/.825070/.667356`；
- HateClipSeg = `.627108/.620614/.566527`。

Controls：`per_video_ecdf` 在两个语料各 0/7 all-SOTA；`raw_identity` 在 HateMM 为 7/7、HateClipSeg 为 0/7，因此两个 control 的 joint 都是 0。说明原结果不依赖 target video参与自己的 scale reference，但依赖一个可跨视频迁移的 corpus-level score geometry；删除跨视频 offset或直接混合不可比 raw scales都不能保持六项门。

独立 post-run reviewer从 run config声明输入重建三种 mapping并重算全部 42 rows，最大数值差为 0；逐 fold target/reference ID集合不相交，coverage为 HateMM 214视频/29,269帧、HateClipSeg 79视频/18,839帧。最弱 heldout margin 仍合法通过，但很窄：HMM tuple `[.05,.30,.30,.35]` pooled AP `.593856`，仅约 `1.0000413×` SOTA；HCS同tuple within `.562075`，约 `1.0002974×` SOTA。

去向：这支持下一机制必须学习 train-only、跨视频共享的局部证据尺度；不授权 inference ensemble、test calibration、ordinary multi-teacher KD或把 heldout test CDF当部署方法。
