# Factorial witness CRF for hateful temporal localization

**淘汰原因：HateClipSeg 三项 test SOTA gate 全败，HateMM pooled AP/ROC 失败，且 HateMM zero-transition within 高于 core，双语料机制不成立。**

截至 2026-08-31；状态：两语料正式 test pilot 已完成并淘汰；不扩 MHC-EN/ZH。

## 跨任务 adaptation

候选从 factorial HMM/latent CRF 与 weakly supervised energy-based action segmentation 迁移“对所有合法
latent sequences 做 exact dynamic-programming marginalization”的方法，但不使用 action transcript、
顺序标签或边界标签。查新必须确认该机制尚未用于 hateful video detection/localization。

## 任务机制

每个 1fps 时刻有 3 个二值 latent variables，分别表示 audio、visual、text 是否承担 hateful
witness；因此联合状态空间固定为 8 个 modality subsets。模态 encoder 产生 unary evidence，CRF
energy 还包含：

- 相邻时刻 subset 的 Hamming transition cost，表达 witness 的持续与异步切换；
- 同时激活多个模态的 coalition cost，防止把一个 binary video label 无条件复制给所有模态。

negative video 的唯一合法路径是全时刻空 subset；positive video 的合法集合是除此路径之外的所有
状态序列。训练用 forward recursion 精确计算两集合的 log partition，并从输入相关的 positive
partition 减去同长度、同 transition、零 unary 的 null positive partition。zero-transition 8-state
情形下 null 项就是 `log(8^T-1)`；一般情形下它还消除 learned transition 自身带来的长度基线，
比较输入证据相对同一 chain prior 的增量，而不是让指数级路径数量天然把长视频判成 positive。
模型既不选择 top-K，也不生成硬 pseudo-span。test 时不使用 video label，forward-backward 输出每秒
任一模态激活的 posterior 作为 frame score，并输出各 bit posterior 作为 structured latent
attribution；binary bag label 不能使它成为 causal 或 ground-truth modality ownership，本项目不作该
claim。

它直接针对当前已证实的 failure：MultiHateLoc 把 bag label 广播到全部分支且 DMS 几乎总选 visual；
typed REBA 的视频级 BiAlign 仍没有 time×modality ownership；UOT 的共享 normal capacity 又在长
hate 段压平排序。这里监督约束作用于完整 latent time×modality path 集，而不是视频级 late gate。

## Anti-pattern 与 controls

- source supervision 仅为同语料 train video labels；无跨主数据集训练、无 test gradient/ckpt selection。
- 无固定 top-K、固定事件比例、post-hoc smoothing、ensemble、calibration 或 corpus routing。
- 必须分别重训 zero-transition 8-state control 与忽略 modality subset 的 collapsed 2-state CRF
  control；只切换同 checkpoint inference 不算训练机制归因。首个最小 pilot 不把所有扩展 control
  一次塞入核心；若 core 晋级，再补 independent-chain、cardinality-only、parameter-matched pooling
  等完整归因。core within-video ROC 若不在
  HateMM/HateClipSeg 同时严格超过两个 controls，说明 typed dynamic partition 不是 load-bearing，
  立即淘汰。
- 首轮固定 seed 234；validation 仅在一次训练内选择 checkpoint，随后立即 test 三项固定指标。
  两语料 core 必须全部三指标严格 SOTA 才可扩 MHC-EN/ZH。

## 可证伪风险

binary bag label 不足以识别哪个非空 subset 是真实 owner；这是已确认限制，不再作为待证明 claim。
若 union posterior 不能超过 collapsed/zero-transition controls，或长度复制仍系统性抬高 positive
probability，就判定 typed path hypothesis 没有定位价值，不得用 CRF/DP 的形式复杂度包装成贡献。

## 实现与运行

- `model.py`：8-state typed subset chain、稳定的非空路径 exact partition、forward-backward posterior。
- `train.py`：仅用同语料 train/val scoped video labels；validation video AP 只选本次固定训练 checkpoint。
- `predict.py`：blind test producer，只读 test ID/特征并使用零占位标签，不加载 temporal GT/test labels。
- `evaluate.py`：只调用全仓库共享 evaluator；不复制指标。
- `test_model.py`：DP 与穷举一致性、null 长度归一化、posterior 数值测试。
- `run_pilot.sh`：HateMM/HateClipSeg × core/zero-transition/collapsed，各自独立重训后立即 test 三指标。
- `launch_pilot.sh`：正式 pilot 的 detached launcher，在 run root 写 `run.log` 与 `run.pid`。

## 正式结果与结论

权威 verdict：`runs/20260831_factorial_witness_crf/pilot_seed234/verdict.json`；逐臂数字来自同目录下
各 corpus/arm 的 evaluator 原生 `metrics.json`。

| corpus | arm | pooled AP | pooled ROC | within ROC |
|---|---|---:|---:|---:|
| HateMM | core | .43352 | .71442 | .63437 |
| HateMM | zero-transition | .47423 | .71334 | .63460 |
| HateMM | collapsed | .41934 | .69022 | .63172 |
| HateClipSeg | core | .57449 | .54655 | .52105 |
| HateClipSeg | zero-transition | .57128 | .53774 | .52012 |
| HateClipSeg | collapsed | .58081 | .54447 | .51775 |

HateMM core 只在 within ROC 以 `+.00283` 越过旧门，但 pooled AP/ROC 大幅失败，而且
zero-transition 的 within 反而高 `.00023`；HateClipSeg core 三项均未过 SOTA，虽然 within 比两个
controls 高，但只有 `+.00094/+.00330`。因此 SOTA gate 与双语料 mechanism gate 均失败，结论
`FAIL_AND_STOP`。

按现行规则读取全部六个 test predictions 与 test GT 做 developmental error analysis，输出为
`runs/20260831_factorial_witness_crf/pilot_seed234/test_error_analysis.json`。三 modality bit posterior
帧均值在两语料都集中于约 `.46–.52`，argmax 又大致均衡，表明失败不是单一 modality branch
collapse，而是 binary bag label 下对巨大 non-empty path 集形成高熵弥散 attribution。core 相对
zero-transition 的 per-video AUC delta 与 GT transition rate 的 Spearman 仅 HMM `-.044`、HCS
`-.079`，learned persistence 没有针对真实 boundary 发挥作用。下一轮不得继续调 transition、CRF
或 path entropy；应改用显式 proposal-level completeness，使训练对象与输出 interval 对齐且不平均
全部非空 paths。
