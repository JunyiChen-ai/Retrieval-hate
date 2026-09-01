# RESET6 HMM/HCS goal-gap audit

**截至 2026-09-01。仅复用既有 test artifacts；不是 premise、candidate 或 performance
evaluation 新运行。依据路径列于各节。**

## 六指标缺口

当前可复现 starting point 仍取 official MultiHateLoc seed 234；固定 SOTA threshold 取现行
三指标表。数值来源：
`runs/20260831_multihateloc_test_error_analysis/main/metrics.json`、
`runs/20260831_powa_starting_point/summary.json` 与
`runs/20260831_universal_teacher_simplex_diagnostic/main/metrics.json`。

| corpus | metric | starting point | SOTA threshold | required gain |
|---|---:|---:|---:|---:|
| HateMM | pooled AP | .493000 | .593832 | +.100831 |
| HateMM | pooled ROC | .738258 | .816184 | +.077925 |
| HateMM | within ROC | .628456 | .631532 | +.003076 |
| HateClipSeg | pooled AP | .553021 | .619371 | +.066350 |
| HateClipSeg | pooled ROC | .544072 | .605022 | +.060950 |
| HateClipSeg | within ROC | .523701 | .561908 | +.038207 |

结论：HMM 的主要未解目标是跨视频 pooled separation；within 已接近门。HCS 三项都缺，不能
只优化 within。任何下一 brief 必须解释其 signal 如何同时触及这些主缺口，不能再以
matched-control 千分量级 within 变化作为接近最终目标的证据。

## 已观察 headroom 与共同失败组

MultiHateLoc 三 seed test error analysis 显示：HMM/HCS 的 best-branch oracle-minus-fused
within 分别为 `+.10627/+.10591`，但这是使用 test GT 的 oracle，不是训练或推理时可用信号。
RESET5 三个正式方法已经证明，fused top-K、masked-coalition credit、普通 branch confidence、
GCE failure与retain gate不能把该 oracle headroom转成共同增益；当前 self-derived modality
responsibility信息链关闭。

两语料都有 low-occupancy failure：positive fraction `<=1/3` 组的 MultiHateLoc within 为
HMM `.54150`、HCS `.48103`，明显低于高 occupancy 组。但真实 occupancy 来自 test GT，部署
时不可见；且既有 Sparse-Mixture Scan 在 low-occupancy 组已反向失败。因此这里只把它记录为
error subgroup，不能把它当可用 correction signal 或重开 occupancy scan。

## 训练与推理时实际可用的独立信号

1. Train-video-label lexical locality 是独立于当前 fused scorer 的本语料信号，test within
   为 HMM `.632629`、HCS `.522700`；HMM 有正确 timing evidence，HCS 仅弱且 pooled ROC低。
   来源：`runs/20260831_video_label_lexical_locality/premise/metrics.json`。
2. 固定 test complementarity 显示完整目标有数值 headroom：HMM 的 lexical 与
   MACIL-SD/POWA family、HCS 的 lexical/DSANet/MultiHateLoc/POWA 与 VERA family存在
   all-SOTA pair。来源：`runs/20260831_test_signal_complementarity/main/metrics.json`。
3. 四个既有 signal 的共同固定 simplex 与 label-free fold scale transfer 在 HMM/HCS均存在
   all-SOTA tuple。来源：
   `runs/20260831_universal_teacher_simplex_diagnostic/main/metrics.json` 与
   `runs/20260831_teacher_scale_transfer_diagnostic/main/metrics.json`。

边界：第2、3项是 developmental test ensemble/upper bound，不是方法；ensemble/calibration
不能作为主方法，普通 multi-teacher KD/knowledge amalgamation 又已被目标任务占用。它们只证明
“独立于 MultiHateLoc self-confidence 的信息确实存在”，尚缺一个通过 novelty三门、单一raw
final score、非普通KD/ensemble的利用机制。下一 candidate若依赖新teacher/producer，仍须按
Rule14处理；不得把这些upper bound直接包装成方法。

## HCS matched-anchor pooled drift 已核清

RESET5正式 harness 中常见 HCS matched anchor `.523714/.497501/.525970`，相对 official
seed-234 `.553021/.544072/.523701` 的 pooled AP/ROC低`.029307/.046571`。这不是模型代码
scaffold bug：

- official seed-234 使用 learning rate `.0001819082`，以 validation video AP 选择 epoch 64；
- RESET5 candidate-selected matched chain 使用 learning rate `.0000909541`，并按固定方法内部
  的 validation within-primary规则选择 anchor epoch 10；
- 两者 train/validation cohort完全相同（HMM为744/109，HCS为251/63，validation ID集合相同）。

因此 drift 来自 candidate-specific超参数/checkpoint选择链，不是数据split或forward漂移。
以后正式summary必须同时报告：(a) 同配置matched control，用于机制归因；(b) official starting
point，用于绝对performance判断。不得把 matched anchor整体差异归因于新增机制。

## 恢复方法探索的约束

- Observed headroom 与 available correction signal 必须在brief中分开写。
- 不再从当前 fused top-K、DMS、masked self-credit或普通branch confidence派生局部责任。
- Failure target由HMM/HCS六项缺口和实际可用signal共同决定，不再锁死 modality selection。
- 可以改变 representation/backbone，但每轮仍只改变一个核心机制，并定义matched control。
- Validation只选既定方法的超参数与checkpoint；随后立即HMM/HCS test。
- 每轮正式test后做一次聚焦error analysis，随后关闭或使用Rule18唯一corrective，不追加无界probe。

本audit已满足RESET6恢复条件中的bounded starting-point/goal-gap audit；它没有批准任何具体
candidate。
