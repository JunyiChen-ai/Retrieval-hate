# LB-SCGP Round 4 Re-evaluation

<details open>
<summary>Full raw reviewer response</summary>

## Executive verdict

**Weighted overall: 9.15 / 10**  
**Verdict: READY**  
**Problem Anchor: preserved**  
**Drift warning: NONE**  
**Unresolved method blocker: NONE**

Round 3 的三个剩余 blocker 均已关闭。LB-SCGP 现在已达到方法 specification READY：这表示可以进入独立实现审计与 SCGP-0 numerical microbenchmark，不表示已有性能提升，更不表示最终目标已经实现。

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.8 |
| Method Specificity | 9.3 |
| Contribution Quality | 9.0 |
| Frontier Leverage | 9.0 |
| Feasibility | 8.3 |
| Validation Focus | 9.5 |
| Venue Readiness | 9.0 |
| **Weighted overall** | **9.15** |

## Anchor / supervision audit

**PASS。**

- 最终 endpoint 始终是 unchanged ordinary full-video train-memory kNN。
- 最终门槛仍是 MHC-EN/MHC-ZH、seeds 0/1/2、accuracy/macro-F1 各 moving baseline `+0.030`。
- 唯一 gold 是 parent-video binary label。
- Teacher 严格 label-blind，不见 prediction、margin、error、correctness、neighbor、loss 或 gradient。
- Teacher cache 先以 Merkle root 关闭，compiler 后读 train labels。
- Certificate atoms始终是 weak/privileged pseudo-signals。
- 不存在 segment/timestamp/span/localization gold、loss、weight或endpoint。
- Validation/test 不加载 certificate、compiler、`G*` 或 `Z*`。
- FULL 无 teacher key、direct rule loss、sample/group reweight、concat、router、rerank或test-time MLLM。

## Exact projector audit

Round 3 blocker已关闭。

在完整实矩阵 Frobenius ambient space中：

- row extraction满足 `L_iL_i*=I`，radial formula确为preimage-ball Euclidean projection；
- class-mean operator满足 `L_cL_c*=(1/n_c)I`，给出的缩放公式正确；
- semantic set使用标准KKT形式 `g(mu)=y-mu A^T(I+mu AA^T)^(-1)Ay` 并通过单调标量root求解；
- `r=0` 使用 Moore–Penrose nullspace projection；
- symmetry作为独立affine set，与row projector在Dykstra交集中协调，定义合法；
- KKT stationarity、complementarity、idempotence、variational inequality、finite difference与dense reference parity均成为binding tests。

Product-space Dykstra现在是唯一solver，且每个步骤是真实闭凸集合的Euclidean projection，因此对固定rank cell的projection/convergence claim成立。Independent float64 verifier仍然binding。

Implementation audit应确认PSD步骤显式对输入取对称部分，但这是代码审计项，不再是method blocker。

## Rank-cell与local target audit

**PASS。** 完整19条top-20内部order约束；20th-vs-all-outsider约束；self exclusion；同一对称 `G` 上的全局coupled boundary；canonical-ID tie；完整orientation enumeration；超过orientation/pivot预算或未处理tie统一映射REMOVE；只有 `LOCAL_STATIONARY_CERTIFIED` target可训练。

方案只声明已探索相邻cells上的numerically certified local target，没有暗示global nearest optimum，表述诚实。

## Direct-control attribution audit

Round 3 blocker已关闭。

- `DIRECT-AEXC`使用与FULL完全相同的 `A_exc`；
- `STATE-MOMENT`覆盖最近的conditional moment/prototype解释；
- 每个aux step的eval-mode、dropout/BatchNorm、complete differentiable outer-memory bank和train-mode恢复顺序明确；
- 一次解析系数由十个seed-0 fold首refresh的full-memory parameter-gradient RMS确定；
- 系数EN/ZH共享、后续不rematch、不读endpoint；
- epoch累计gradient strength仅诊断；
- wall time只报告，不虚称compute matching；
- seed-0按frozen worst-cell rule选一个全局最强direct control；
- final两库三seed实际运行该control；
- FULL-minus-direct要求所有seed/metric/dataset正、hierarchical paired lower bound `>0`、Holm通过。

因此“proximal realization优于直接使用同一semantic moment”已成为最终可证伪claim，而非seed-0印象。

## Farkas audit

**PASS。** Example/pair/triplet/SupCon cones均有明确合法宇宙；pair cone不含同pair双方向；triplet oracle joint扫描 `(p,n)` 并包含hinge activity；small-bank brute-force parity；machine-readable universe、active-mask、generated columns及termination global minimum；abstract与realized displacement均审；conclusion只覆盖registered cones；actual AdamW归因仍依靠learned matched controls。

## Pilot design audit

Round 3 blocker已关闭。

- family选择使用固定 `Q<C<R` first-passing support rule；
- main sample和每个Rao–Wu replicate均重新计算HT totals、ESS、family selection、A/B reference profiles和effect；
- selection failure按零增益处理；
- selection success要求 `>=95%`；
- A/B cross-fit阻止identity自用于reference；
- unprocessed与rejected严格区分；
- uncertainty覆盖family selection，而非条件化于事后family。

该estimand现在完整且无outcome循环。

## Factor、fit与rollback

**PASS。** PSD factor维度与zero padding正确；repeated eigenspace/nullspace使用coordinate-projector canonical basis；backend/version/thread count冻结；target与realized collapse均检查；one block定义清楚；完整恢复model、AdamW、scheduler、scaler、RNG、sampler和epoch cursor；REMOVE replay hash验证；target从不成为inference key。

## Contribution and novelty

当前claim已正确收窄为：

> label-blind structural reflection → exact-vote-safe proximal full-bank target → uniform encoder fit → ordinary kNN。

它不再声称发明certificate、moment alignment、Gram KD、target fitting或通用metric non-equivalence。相对最近解释：vs direct semantic moment用 `DIRECT-AEXC`；vs conditional prototype/moment用 `STATE-MOMENT`；vs P4/schema prediction用P4 control；vs pair/triplet/SupCon用cone audit与learned controls；vs ECM没有pseudo-group risk、sample weighting或gradient surgery；vs teacher geometry KD，target由compiler、video labels和exact vote共同求解，不复制teacher embedding。

该单一组合贡献已经足够集中、可守且可证伪。

## Simplification opportunities

**NONE。** 当前一个exception family、一个solver、一个target-fit path和一个final strongest-direct control已经足够精简。

## Modernization opportunities

**NONE。** 不应加入learned compiler、differentiable sorter、larger teacher、router或actual-update projection。

## Remaining implementation checks

以下是执行审计项，不是方法修订 blocker：

1. 逐projector复算KKT和reference parity。
2. 检查PSD projection显式symmetrization与Dykstra correction存储。
3. 小bank brute-force验证triplet oracle和rank pivots。
4. 验证direct-control pooled coefficient不读取任何endpoint。
5. 验证Rao–Wu replicate完整重选family。
6. 先执行sealed synthetic + one-real-fold microbenchmark；`<160 GPU-hours`失败即STOP。
7. 只有microbenchmark和独立code review通过后才允许SCGP-0；teacher仍保持零调用。

**Final verdict: READY。** 下一步仅是独立实现审计与 SCGP-0 numerical microbenchmark。最终科研目标仍未满足，必须等两库三seed的最终 `+0.030/+0.030` 和完整归因门实际通过。

</details>
