# Failure Equivalence Ledger

**2026-09-02 起本文件只作参考，不再是实现前阻断依据；其中"重开所需新证据"一列与末尾"计数"节作废。** 依据 `RESEARCH_ITERATION_RULES.md` 第 4 条。

截至 2026-08-31。依据 `docs/PROCESS_REVIEW_2026-08-31.md`、近期候选review及其引用的`runs/` test artifacts。用途：在novelty检索前拒绝与既有失败机制等价的新命名。

| 等价类 | 决定性反例/证据 | 已关闭代表 | 重开所需新证据 |
|---|---|---|---|
| Broadcast/constant score | video bag可正确但positive-video内score恒定，within=`.5` | 多个MIL、CTC、carrier、Hodge | 新增可观测local direction，双语料胜time-shuffle`>=.020` |
| Video identity + position | global feature复制到每秒，再用position任意放置latent event | Evidence-Program Graph CTC、TCC scout | mean-repeated与position-only均失败，真实内容时间对齐成立 |
| Teacher-order KD | teacher给pair方向即普通order/margin distillation | knowledge amalgamation、Privileged Rank Transfer、SDR | 非teacher direction的新train-only observation；不能只换distillation loss |
| Ensemble/calibration/scale map | 多signal blend或CDF改善数值但不是主方法 | universal simplex、scale transfer | 单一模型raw score学出同一信息；不得inference blend/CDF/routing |
| Auxiliary-head bypass | auxiliary relation优化但final ranking几乎不变 | deletion-carrier ItS2CLR | load-bearing readout且不能退化为普通direct head；需新observation |
| Direct-head replacement | prototype energy等价normalized linear BCE head | carrier-energy bottleneck | 不是换head/loss；必须新增task-specific information |
| Reference-only gauge | common reference只给全部target score加常数 | benign-anchored Hodge | negative evidence进入非平移不变的任务约束；但不能变成calibration |
| Unary/edge algebraic redundancy | `y_ij=u_i-u_j`，Hodge项无独立信息 | cycle-selective Hodge v1 | relational observation在固定unary下独立改变正确ranking |
| Single-carrier/branch dominance | 一个恒高或abstain modality支配final score | carrier-energy、ownership lines | same-video/same-modality局部对照且winner责任可审计 |
| Gradient-only modality balancing | branch gradient确实被调制，但HMM final ranking几乎不变；HCS改变ranking时pooled崩溃 | witness-conditional DGM | 新机制必须直接改变可审计的time×modality final-score responsibility；不得调gamma/competence/ckpt续命 |
| Capacity-forced modality ownership | per-modality assignment数严格均衡，但bag label不能识别哪个modality在何时可信；弱模态被强制写入final score | temporal expert-choice MIL | 新机制需提供独立于负载均衡的local competence evidence；不得调capacity、temperature或checkpoint rule续命 |
| Carrier-absence constant | local carrier缺失时输入退化为同一空表示，score恒为intercept、within=`.5` | raw video-label lexical locality | 同一statistic必须在预先定义的carrier-absent区间仍有native变化；不得事后拼OCR/visual branch |
| Stable-but-wrong local judge | swap/curl/consistency只证明自洽，不证明hate timing | listwise VLM、Hodge | GT developmental test direction与corruption controls同时成立 |
| Generic temporal regularization | smoothing/duration对HMM/HCS方向不统一 | post-coalition smoothing、semi-Markov | 同一native boundary statistic双语料成立，非固定平滑 |
| Duration field improves local order but not full performance | 可学习 marked splat 在 HMM within 大幅提升、HCS 小幅提升，证明 duration field 可改变正确局部排序；但 HMM pooled 与 HCS 三项仍未达 SOTA | marked temporal splat MIL | 保留已成立的 duration-field 证据；新方法必须解释并直接改善跨视频 pooled discrimination 与 HCS 弱增益，不能只扫描 kernel、top-K 或按语料调参 |
| Boundary mass conservation | 每个center在有效视频边界内重归一后，HMM pooled AP/ROC仅回升`.0168/.0144`但within下降`.0412`，HCS三项全降 | mass-conserving marked splat | 不扫描renormalization/reflection/kernel；只有新的内容相关证据能同时替代位置增益并改善HCS才可重开 |
| Dense certified-negative suppression | 对negative bags全部有效帧施加BCE只使HMM pooled AP/ROC变化`+.0006/+.0047`且within降`.0090`；HCS pooled AP/ROC降`.0223/.0154` | dense-negative marked splat | 不调dense-loss权重、margin或negative mining续命；需新的局部监督机制而非更强压低负视频 |
| Alternating temporal residual reconcilement | 12-trial validation selection后，exact top-K bag-gradient residual在HMM仅比matched cyclic control within高`.0151`且仍低于MultiHateLoc，pooled明显下降；HCS还比control低`.0093` | Temporal Residual Reconcilement MIL | 不再扫描residual weight、lr、K、stage order或cycle；只有post-test证据明确指出一个可修复且跨HMM/HCS共同的residual分配错误，才允许唯一一次corrective iteration |
| Negative-null sparse-mixture scan | 12-trial validation selection后，HMM core相对matched fixed-top-K control within仅`+.0018`且pooled两项下降；HCS三项全降、within`-.0081`。声称应改善的low-occupancy组在HMM/HCS反而`-.0052/-.0091`，均劣于high-occupancy组 | Sparse-Mixture Scan MIL | 不调scan weight、temperature、rank grid、null EMA、margin或tail变换；只有新的test证据证明negative-null ordered-tail evidence在low-occupancy视频跨HMM/HCS一致有效才可重开 |
| Local video-ID/position adversarial suppression | 12-trial validation selection后，HMM core/control within为`.5542/.5546`，video-ID probe不降且high-position-risk增益为负；HCS虽降低两个probe且high-risk组`+.0576`，但core仍低于anchor且pooled AP/ROC较control下降`.0330/.0595` | Local-Quotient Adversarial MIL | 不调GRL权重/schedule、position bins、local scale或backbone共享续命；只有新test证据支持一个跨HMM/HCS共同的nuisance与pooled-preserving约束才可重开 |
| Self-coalition temporal modality credit | validation-selected aligned相对anchor在HMM/HCS三项均微升，但HMM within低于等训练量circular-shift control`.00036`；高oracle-gap组aligned-minus-shifted为HMM`-.00370`、HCS仅`+.00012`，正确credit timing不是共同增益来源 | Temporal Coalition-Credit MIL | 不调alpha、positive-part映射、shift、router width或按branch disagreement gating；需新的、独立于当前fused模型自确认credit的local modality-competence evidence才可重开 |
| Latent-witness failure debiasing | 14-trial validation selection后，relative-vs-uniform within在HMM仅`+.00058`、HCS为`-.00056`；HCS relative的AP/ROC也均低于uniform，且两语料都远未all-SOTA | Witness-Failure Debiasing MIL | 不调GCE、bias聚合、relative-weight公式、support producer或强度续命；只有新的test证据表明shortcut-expert failure在HMM/HCS都能识别正确local witness才可重开 |
| Witness-preserving temporal token substitution | 14-trial validation selection后，aligned相对time-shifted within在HMM/HCS仅`+.00013/+.00215`，均未达机制门；HMM aligned pooled ROC还下降`.00247`，两语料all-SOTA全败 | Witness-Preserving Temporal TokenFusion | 不调retain gate、projection、budget、shift或fusion strength；只有新的test证据表明aligned donor replacement在HMM/HCS有实质共同增益才可重开 |
| Cross-corpus supervision | 其他主数据集span提升但违反独立训练 | LOCO-ST/span transfer | 永不重开为主方法；只保留diagnostic |

## Pre-novelty failure-equivalence screen（不是独立 identifiability 硬门）

Ledger 只在以下情形于实现前 STOP：候选与表中失败机制严格同构且没有新增约束，或候选核心项在代数上完全不能进入 final score。一般性 broadcast、position、video-identity、branch bypass 等 shortcut 只登记为风险，并在最小端到端 HMM/HCS test 后用 matched control 判断；不要求弱监督方法在实现前解析排除所有可能 shortcut。

候选 brief 最多一页，只需写明：`STATUS.md` 已证实的具体 failure、跨任务来源、task adaptation delta、进入 final score 的路径、一个可证伪 test 预期和一个 matched control。Rule 14 负责 observation evidence，Rule 12 负责 novelty；两者通过后立即实现，不再追加 pre-method proof。

## Pre-novelty minimum-evidence gate

当前 gate 以 `RESEARCH_ITERATION_RULES.md` Rule 14 为准：premise 只验证局部变化存在、不是纯 position/broadcast/video-identity、且 HMM/HCS 方向一致；使用与假设最直接的一个 matched control 即可。Carrier strata 只作诊断，不作全档硬门；raw statistic 不需要在学习机制前独立承担方法级定位证明。通过后立即进入 novelty 三门与最小 end-to-end 方法。连续两个 premise 失败即停止 raw-statistic 搜索并重新审视 failure mode、gate 或信息源。

## 计数

RESET3中mass-conserving与dense-negative marked splat是两次result-relevant performance failure；validation-selected original只是配置校正，不占candidate failure。RESET4的三个正式失败依次为validation-selected Temporal Residual Reconcilement MIL、Sparse-Mixture Scan MIL与Local-Quotient Adversarial MIL。`docs/PROCESS_REVIEW_RESET5_2026-09-01.md`已裁定`RESET`并落实anchor-compatible first。RESET5三个正式失败依次为Temporal Coalition-Credit MIL、Witness-Failure Debiasing MIL与Witness-Preserving Temporal TokenFusion。累计连续performance failure=`11`，只有通过performance gate才清零；RESET5窗口=`3/3`，已停止生成新candidate并触发process review。Diagnostic、scout、premise、配置补选、matched control、初始未调参test或novelty PASS不增加或清零任一performance计数。
