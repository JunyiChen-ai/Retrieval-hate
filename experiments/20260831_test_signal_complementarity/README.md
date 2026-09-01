# Test signal complementarity diagnostic

截至 2026-08-31。该目录是 Rule 10 允许的 developmental test error analysis，
不是方法、ensemble 候选或 SOTA claim。

## 目的

lexical posterior regularization 在 HMM train support 上满足约束却未改善 test
ranking，HCS 连机制门都失败。这里检查现有独立信号是否存在足以同时越过三项
SOTA 门的互补性。这里只审计两个预先固定的构造族：global empirical-CDF
pairwise convex blend，以及每个视频整条选择一个信号的 within-oracle readout。
positive 结果可证明对应构造存在 headroom；negative 结果只关闭对应的固定构造族，
不能否定一般的 nonlinear/per-frame fusion、distillation 或其他 selector objective。

固定信号：POWA、MultiHateLoc、VERA、同语料 lexical locality，以及 HMM 的
MACIL-SD / HCS 的 DSANet。每个信号只取一个预先指定 branch、seed 234。

## 分析

1. 逐信号完整 test 三指标；
2. 每个信号在整个 test frame pool 做 label-free empirical-CDF rank normalization，
   对所有 signal pairs 扫固定权重 `{0,.05,...,1}`，共享 evaluator 计算三指标；
3. 对每个 both-class positive video，用 test GT 选择 within AUC 最高的单信号，
   其他视频使用该语料 pooled ROC 最强的单信号，形成明确使用 GT 的
   whole-video single-signal within-oracle readout，再由共享 evaluator 评测；
4. 输出每个 signal 被该 selector 选择的 video 数、其实际三指标，以及固定
   pairwise grid 是否存在 all-metric SOTA。

该 within-oracle 只最大化 eligible positive video 的 within AUC；它不最大化 pooled
AP/ROC，也不覆盖 pair、逐帧或 nonlinear fusion。因此它不是一般 signal-headroom
upper bound。只有其 within 本身低于门槛时，才说明这个特定 whole-video
single-signal selector family 不可能三项同时过门；其 pooled 指标失败不能给其他
selector objective 提供上界。

所有结果均属于已揭盲 developmental evidence。不得把任何 blend/oracle 作为
主方法、校准、routing 或 ensemble 晋级。

## 正式运行

唯一正式入口为 `run.sh`。它只允许新建
`runs/20260831_test_signal_complementarity/main/`，并保存 `run.pid`、`run.log`、
`config.json`、可读的 `code_version.txt` 与共享 evaluator 直接产生的
`metrics.json`。本诊断固定 seed 234，不做选择或重跑。

## 结果与去向

权威输出：`runs/20260831_test_signal_complementarity/main/metrics.json`；独立
post-run 重算审计 PASS。HateMM 的 210 个固定 pair-weight entries 中有 20 个
all-SOTA，来自 lexical+MACIL-SD 与 lexical+POWA 两个 pair family；代表性的
lexical `.35` + POWA `.65` 为 AP/ROC/within
`.647732/.836151/.659823`。HateClipSeg 的 210 个 entries 中有 24 个 all-SOTA，
来自四个全部包含 VERA 的 family；best-within passing 的 lexical `.05` + VERA
`.95` 为 `.629456/.617913/.595685`。

whole-video selector readout 在 HateMM 为 `.611664/.826918/.794703`，三门通过；
HateClipSeg 为 `.624822/.602475/.695980`，pooled ROC 未过。该 selector 结果只描述
这一特定构造，不能作为其他 selector 的上界。

结论：两个语料都存在“强 pooled carrier + 补充 within ordering signal”的构造性
互补，因而不应把当前失败解释成现有观测完全没有定位信息。但这些 test-GT-informed
blend/selector 不是方法，禁止晋级。下一轮只允许把该现象转成同语料、video-label-only
训练的单学生机制，并需另过三项 novelty 硬门。
