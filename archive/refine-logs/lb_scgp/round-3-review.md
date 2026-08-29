# LB-SCGP Round 3 Re-evaluation

<details open>
<summary>Full raw reviewer response</summary>

## Executive verdict

**Weighted overall: 8.69 / 10**  
**Verdict: REVISE**  
**Anchor status: preserved**  
**Drift warning: NONE**

Round 2 blockers基本关闭。当前方案已成为一个聚焦、可证伪且明显区别于 ECM/SQ/P4 的方法：

- 唯一语义机制收窄为一个 support-selected structural exception reflection；
- `DIRECT-AEXC` 与 `STATE-MOMENT` 正面覆盖了“只是条件矩匹配”的最近解释；
- solver 已统一为 product-space Dykstra；
- 只有 `LOCAL_STATIONARY_CERTIFIED` target 可训练；
- triplet cone、realized displacement、rollback、pilot design inference均显著加强；
- teacher label-blind、cache-before-label、test-clean、no-segment-gold全部保持。

尚余三个窄但真实的 specification blockers，因此不能按 `overall>=9` 规则给 READY。

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.8 |
| Method Specificity | 8.8 |
| Contribution Quality | 8.5 |
| Frontier Leverage | 8.5 |
| Feasibility | 7.5 |
| Validation Focus | 9.0 |
| Venue Readiness | 8.3 |
| **Weighted overall** | **8.69** |

## Blocker 1：两个 trust-set projection 尚未证明是欧氏投影

Dykstra 只有在每个步骤确实计算对应闭凸集合的 Euclidean projection 时才有当前收敛含义。

- 单行 trust ball 若在完整非对称 ambient `G` 上定义，可以作 row radial projection；
- 但最终还与 symmetry 集合相交，而且“symmetric row coefficient”会同时影响 `G_ij/G_ji`，其范数缩放必须明确；
- class-mean-row trust 是线性映射 `L_cG` 的二范数球。除非证明 `L_cL_c^T=alpha I`，其投影不是普通 radial clipping。

Semantic ellipsoid的思路正确，但应把公式完整写出：

`g(mu)=y-mu A^T(I+mu AA^T)^(-1)Ay`,

由标量 `mu>=0` 解 `||Ag(mu)||=r`。当前 “16-dimensional dual + unique multiplier” 尚略含糊。

**Required fix — CRITICAL**

为 row trust、class-mean trust 和 semantic set 分别给出：ambient inner product；linear operator `L`；exact KKT projection；scalar root equation；synthetic finite-difference/projection optimality parity。

若 class-mean set 无简洁闭式，就像 semantic set 一样通过小型 `LL^T` dual solve，不要称为 radial projection。

## Blocker 2：matched direct controls 的执行和 final attribution尚未完全冻结

`DIRECT-AEXC` 是正确且必要的最近控制，但仍需明确：

1. coefficient 是每 fold解析匹配，还是一个 EN/ZH共享全局值；
2. `eval()` full-bank graph、dropout/BatchNorm状态和随后恢复 `train()` 的确切顺序；
3. 每个 fit batch都重建 complete differentiable bank，还是每个 refresh只算一次；
4. first-batch norm matching 后，整个epoch的累计辅助梯度强度可能严重漂移；
5. FULL、DIRECT-AEXC、STATE-MOMENT 的计算量不同，至少应报告而不能写成 compute-matched。

最重要的是，正文开头说 direct controls 在 pilot/seed-0/final 都 binding，但 final gate只明确要求 FULL 显著胜 REMOVE/SHUFFLE。

**Required fix — CRITICAL**

- 冻结一个无 outcome 的解析 coefficient protocol；
- 记录整epoch累计 auxiliary-gradient norm和wall-time，但不得二次调参；
- seed-0选出 `max(DIRECT-AEXC, STATE-MOMENT)`；
- final三 seeds必须运行该最强直接控制；
- FULL-minus-strongest-direct 在两库两指标上也须3/3同号且paired CI lower bound `>0`。否则只证明MLLM有用，未证明 proximal realization 是贡献来源。

## Blocker 3：pilot family selection与design inference需最后闭合

`e*=argmax min ESS` 不使用性能结果，因此不存在普通 outcome tuning；但它仍由同一pilot sample的随机state support选择，而随后又在该样本上估计reflection effect。

**Required fix — IMPORTANT**

二选一：采用固定 `Q<C<R` 中第一个在两库两half均通过ESS门的family；或在每个Rao–Wu replicate内重复support-only family selection，并对selection failure作预注册处理。

同时明确 Rao–Wu replicate同时重建 HT cell totals；selected family；A/B reference profiles；held-out reflection residual；correction-direction statistic。这样 lower bound才覆盖完整pilot procedure，而不是把随机选出的family当固定。

## Dykstra与rank-cell审计

除上述 projector 公式外，本轮已基本闭合：

- primal `x=(G,xi)` 与 projection objective明确；
- symmetry、PSD、box、rank、vote、slack、semantic集合分离；
- cell内部完整19条order及20th-vs-outsider inequalities；
- 同一对称G上的全局boundary event；
- 超过orientation/pivot预算映射REMOVE；
- 只有全部相邻orientation检查完成才称 `LOCAL_STATIONARY_CERTIFIED`；
- unresolved tie不能训练；
- independent float64 exact evaluator仍binding。

该表述诚实地只主张 explored union上的local target，不再暗示global nearest solution。

## Farkas审计

Round 2 blocker已关闭：pair cone只有label-legal单方向；triplet oracle joint扫描 `(p,n)` 并包含hinge activity；small-bank brute-force parity；machine-readable universe/active-mask/column manifest；abstract和realized displacement均审；claim明确限于registered cones；learned AdamW controls仍binding。

Implementation plan仍应验证column-generation终止时的全宇宙minimum witness inner product，而不能只记录最后新增列。除此之外不再构成method blocker。

## Structural exception compiler

当前compiler不再冒充content identity。它明确检验一个更窄的科学假设：direct-speaker reading变成closed exception时，label-conditioned full-bank state profile应发生固定D↔e反射。

这仍是label-conditioned moment structure，但 `DIRECT-AEXC` 和 `STATE-MOMENT` 已把该最近解释变成binding controls。因此当前 novelty边界是可守的，前提是 final也显著胜最强direct control。

信心只进入operator估计，不进入encoder sample loss；这不是ECM式sample/group risk reweighting。

## Factor、fit与rollback

此前 blocker已关闭：factor维度、repeated subspace canonical basis、backend freeze、abstract/realized collapse、block定义、完整rollback及REMOVE replay均明确。

## Supervision与test-clean审计

**PASS。** 唯一gold始终是parent-video binary label；frames/ASR/OCR只是整视频输入；teacher看不到label/prediction/margin/error/neighbour/loss；post-cache compiler才可读train labels；certificate states始终是weak/privileged pseudo-signals；schema/compiler/loss均无segment/timestamp/span/localization对象；validation/test不加载certificate/compiler/target；inference保持ordinary full-video train-memory kNN。

## Simplification opportunities

1. 保持最多一个exception family；不要在后续因support不足合并多个family。
2. final仅携带seed-0最强的一个direct control，而非两个都跑三seed。
3. Farkas继续作为审计，不扩写成通用理论贡献。
4. Dykstra projector统一使用同一种“小型dual root solve”表述，减少特殊情况。

## Modernization opportunities

**NONE。**

## Minimal remaining action items

1. 写出row/class-mean/semantic集合的exact Euclidean projection与parity test。
2. 冻结DIRECT-AEXC/STATE-MOMENT coefficient、mode、累计强度和final strongest-control gate。
3. 让Rao–Wu inference包含support-only family selection。
4. 同一reviewer做最后一轮复评。

**Final verdict: REVISE。** 方法本体已接近READY；剩余问题是三个可局部修正的执行/归因定义，不需要改方向、增加模块或放宽目标。

</details>
