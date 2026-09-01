# Developmental test error analysis post-run review

截至 2026-08-31。审查对象为权威输出
`runs/20260831_owner_abstaining_its2clr/pilot_seed234/test_error_analysis.json`、生成脚本
`analyze_test_errors.py`，以及同一 run 下 HateMM/HateClipSeg 的正式 `scores.jsonl`、`metrics.json`、
`model.pt` 与 `train_log.json`。本审查未训练模型、未修改方法或正式结果。

## 最终裁定

**PASS。** Error-analysis 结果链与冻结的六个正式 test score、共享 evaluator metrics 和 core checkpoint
一致；test GT 的使用符合允许的 developmental error-analysis 边界。数据支持“core 几乎没有改变
broadcast 的帧排序，观察到的 carrier-dependent gain 很弱，因此淘汰当前机制而不是围绕本轮 test 调参”。

这里的“几乎不改变”只应表述为**排序几乎不变**。HateMM 的 raw score 数值仍有可见 calibration/scale
差异，不能扩大成“逐帧分数本身几乎相同”。

## 1. 正式 artifacts 与 checkpoint 一致性

- JSON 中登记的两语料 `anchor/broadcast/core` 路径均指向本次正式 pilot 的对应 score 和 metrics。
- 六个 score 文件重新解析后均精确覆盖共享 evaluator test cohort：HateMM 214 videos / 29,269 frames，
  HateClipSeg 79 videos / 18,839 frames；video ID 无 missing/extra，逐视频长度与 1 fps GT 一致，全部值
  finite 且在 `[0,1]`。
- 重新调用唯一共享 evaluator
  `scripts/reproduction_baselines/eval_baseline_scores.py::evaluate_scores`，六个完整结果对象均与各自
  `metrics.json` 一致。Error-analysis JSON 转录的 pooled AP、pooled ROC 与 within-video macro ROC 也与
  正式 metrics 一致。
- 两个 core checkpoint 的 corpus/arm 与登记值一致；HateMM selected epoch 为 10，HateClipSeg 为 1。
  两者都与各自 60-epoch train log 的最大 validation video-AP epoch/value 一致；train log 明确记录 test
  未用于梯度或 checkpoint selection。
- 对当前冻结 core checkpoint 重新运行只读 carrier diagnostic，得到的 core score 与正式 score 最大绝对误差
  分别为 HateMM `4.47e-7`、HateClipSeg `1.19e-7`，均低于脚本的 `1e-6` 硬门。诊断没有替换正式
  prediction，也不是新 inference arm。
- 独立执行 `analyze_corpus` 并逐字段比较后，两语料全部统计值与权威 JSON 一致。

## 2. “排序几乎不变”的证据

比较的是 exact within-video AUC eligible cohort 上，core 与 capacity-matched broadcast 的逐视频帧分数
Spearman：

| corpus | eligible videos | mean rho | median rho | 高相关覆盖 |
|---|---:|---:|---:|---:|
| HateMM | 85 | 0.97568 | 0.98116 | 84/85 的 rho ≥ 0.90；81/85 ≥ 0.95 |
| HateClipSeg | 67 | 0.99723 | 0.99752 | 67/67 的 rho ≥ 0.95 |

HateMM 最低 rho 仍为 `0.83289`，HateClipSeg 最低为 `0.98950`。因此“两种训练关系产生的帧排序几乎相同”
有直接支持。

需要保留的限定是 raw score 差异：HateMM pooled absolute difference 的 mean/median 为
`0.05199/0.02770`，并不能称数值预测逐点相同；HateClipSeg 才接近逐点相同，mean/median 仅
`0.000372/0.000389`。这不影响排序结论，但排除了更宽的 score-identity 表述。

## 3. within gain 与 carrier 关联

core 相对 broadcast 的 per-video AUC 变化为：

| corpus | mean delta | median delta | improved | worsened |
|---|---:|---:|---:|---:|
| HateMM | +0.003129 | +0.002983 | 0.5294 | 0.4000 |
| HateClipSeg | +0.001048 | 0.000000 | 0.4627 | 0.4925 |

这与正式 macro-within 差完全一致，且远低于冻结的至少一语料 `+0.020` 门。HateClipSeg 中改善视频比例还
低于恶化比例，正均值只来自很小的不对称变化。

以冻结 core checkpoint、同语料 train negative centroid 与 fixed top-third 定义的 test carrier rate，和
per-video AUC delta 的 Spearman 为：

| corpus | visual | audio | text |
|---|---:|---:|---:|
| HateMM | -0.1359 | +0.1388 | +0.2397 |
| HateClipSeg | +0.0873 | +0.1781 | +0.0165 |

所有绝对相关都不超过 `0.24`，visual 的方向跨语料翻转，text 的弱正相关也不复现。GT positive fraction
与 delta 的 rho 仅 HateMM `+0.0787`、HateClipSeg `-0.1577`；三个 occupancy strata 的均值也没有共同方向。
因此没有可复现的 carrier-rate 或 positive-occupancy 子群能解释一个足够大的 gain。

该诊断 carrier 是最终 core model 在“全体同语料 train negative centroid + sole-neighbor replacement”下的
局部 deletion sensitivity，不是 test label、真实 modality owner、因果归因，也不是逐项重放每个 OOF fold
训练时的 pseudo mask。故可支持的是“按冻结诊断定义，carrier-dependent gain 的观察关联很弱”，不能声称
carrier 在所有定义下无作用。

## 4. Test-label 使用边界

- 正式 `predict.py` 只读取 evaluator test membership，并给 dataset 传零占位 video label；它不加载 frame GT。
- Error analysis 在三个 arm 的正式 score 和 core checkpoint 已冻结后才调用 test GT，用于核 coverage/length、
  per-video AUC delta、GT occupancy strata 与相关性。
- carrier rate 本身的 model forward 不读 test GT；replacement centroid 只由当前 corpus 的 train negative
  video frames及 train video labels生成。test GT 只在随后对齐和统计关联阶段使用。
- 脚本不执行 optimizer、backward、checkpoint selection、threshold search、test-time routing、ensemble 或
  calibration，输出也明确标记 `developmental_error_analysis=true` 和
  `test_labels_used_for_gradient_or_checkpoint_selection=false`。

因此该使用符合 `RESEARCH_ITERATION_RULES.md` Rule 10。它可以作为生成**新机制**的 developmental evidence；
任何受其影响的后续 test 都必须继续标记 iterative/developmental，不能冒充未揭盲 confirmatory 结果。

## 5. 决策是否被数据支持

正式结果已经显示 core 在两个语料的三项 SOTA 门全部失败，core-vs-broadcast within gain 又只有
`+0.00313/+0.00105`。本次分析进一步显示：两种方法排序高度相似，改善与 carrier rate 的关联弱且不跨语料
稳定，没有一个预先定义的子群给出足以支持当前机制的共同信号。

所以“淘汰当前 deletion-carrier-abstaining ItS2CLR，不围绕已揭盲 test 调 carrier margin、replacement、
self-paced ratio 或按语料路由”是由冻结 gate 和 error analysis 共同支持的正确结论。分析不能证明所有
deletion 或 abstaining 机制都无效，但足以否决当前已实现、已冻结的候选；若继续研究，应提出新的机制假设，
而不是用这些 test strata 对本候选做 post-hoc 调参。

最终结论：**result-chain 与 test-use integrity PASS；“排序近似不变、carrier-dependent gain 弱、淘汰而非
调参”的窄结论成立。**
