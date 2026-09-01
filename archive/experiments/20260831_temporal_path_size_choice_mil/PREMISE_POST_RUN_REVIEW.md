# Frozen developmental test premise post-run audit

截至 2026-08-31。审查对象为权威输出
`runs/20260831_temporal_path_size_choice_mil/premise/analysis.json`、`run.log`，冻结规则与实现
`README.md`、`analyze_frozen_premise.py`、`src/proposal_mil.py`，以及输入的正式 multimodal P-MIL artifacts。
本审查未训练模型、未生成新arm、未修改正式 premise 输出。

## 最终裁定

**Result-chain integrity PASS；premise FAIL。`STOP_BEFORE_FORMAL_METHOD` 正确。**

权威 JSON 的 frozen score reconstruction、共享 evaluator metrics、top/length、long posterior mass、
wrong-top path-size、三种candidate-set perturbation与11项failure均由当前冻结实现完整只读复算并一致。
失败不是missing cohort、undefined Spearman、浮点边界或布尔门错误造成的。

该机制不得进入formal训练；也不得根据本次test结果扫描beta、改path-size公式、换utility/readout、调整IoU或
length cutoff，或按语料选择规则。若继续研究，必须作为新的机制与新的预注册，而不能把当前premise修到
通过。

## 1. 运行与输出完整性

- `run.log`按固定顺序处理HateMM 214个test视频和HateClipSeg 79个test视频，随后打印完整verdict；无
  traceback、abort、non-finite或partial-write迹象。
- `run.pid`对应进程已经结束，系统中没有该premise的存活分析或训练进程。
- `analysis.json`可完整解析，顶层固定为`split=test`、`developmental_test_evidence=true`、
  `test_labels_used_for_gradient_or_checkpoint_selection=false`，并登记冻结P-MIL source run与唯一fixed policy。
- 两个corpus均存在，`n_test_videos`分别为214和79；gate对stored corpus payload重放后得到与JSON逐项相同的
  `pass=false`、decision和failure列表。

## 2. Frozen P-MIL重建与metrics复算

全cohort重新加载两个 frozen source MultiHateLoc checkpoint、两个正式 P-MIL checkpoint和1 fps features，
按冻结proposal recipe与full P-MIL score重建正式`score_pmil`。两语料所有视频的
`formal_frozen_score_reconstruction_max_abs_error`均为`0.0`。因此premise确实作用于产生正式P-MIL结果的
同一checkpoint、candidate set和proposal utility，不是相邻实现或另一branch。

Beta0/beta1 frame marginal重新通过唯一共享 evaluator评测，完整结果与JSON一致：

| corpus | beta | pooled AP | pooled ROC-AUC | within-video macro ROC-AUC | within n |
|---|---:|---:|---:|---:|---:|
| HateMM | 0 | 0.3451967786 | 0.6484400832 | 0.7007441816 | 85 |
| HateMM | 1 | 0.3441060880 | 0.6631668985 | 0.6827907558 | 85 |
| HateClipSeg | 0 | 0.4973728600 | 0.4500169668 | 0.4776126935 | 67 |
| HateClipSeg | 1 | 0.5249470214 | 0.5172594329 | 0.4700641105 | 67 |

Core beta1使定位主指标within在HateMM下降`0.01795343`，在HateClipSeg下降`0.00754858`，所以两个
`within_non_decrease`均明确失败。HCS pooled AP/ROC虽分别提高约`0.02757/0.06724`，也不能覆盖预注册的
within硬门；HMM pooled AP同时略降。这里的pooled值仍只是未拟合outside option的proposal-conditional
diagnostic，不是formal method performance。

## 3. Top、duration与long posterior mass

按完整test cohort独立还原统计：

| corpus | beta | exact whole top | near-whole top (`width>=2T/3`) | median width/T | mean long mass |
|---|---:|---:|---:|---:|---:|
| HateMM | 0 | 3/214 | 33/214 | 0.070438 | 0.033329 |
| HateMM | 1 | 19/214 | 33/214 | 0.048553 | 0.050015 |
| HateClipSeg | 0 | 48/79 | 63/79 | 1.000000 | 0.031328 |
| HateClipSeg | 1 | 34/79 | 43/79 | 0.922780 | 0.124829 |

- HateMM虽缩短top duration中位数，但whole-top由3增至19，near-whole完全不降，long mass也上升约50%。
- HCS的whole、near-whole与duration三项改善，但long posterior mass约增至原来的4倍。

这些是计数与明显幅度变化，不是strict comparison的数值噪声。对应HMM三个length failure和HCS
`long_mass_nonincrease=false`均正确，也直接验证README预警的“path size可能奖励覆盖稀有边缘秒的长proposal”
没有被排除。

## 4. Wrong-top correctability

Half-open GT intervals、proposal IoU和固定`best IoU>=0.5 / raw top IoU<0.3`条件重算一致：

- HateMM有53个correctable wrong-top cases，mean
  `logPS_error-logPS_best=-0.318565`，方向正确；
- HateClipSeg有29个cases，但mean为`+0.843849`，方向强烈相反。

因此HCS中PSL平均会比best-IoU proposal给予错误top更大的path size，而不是更强惩罚错误top。
`hateclipseg:wrong_top_has_lower_path_size`失败正确；不能因HMM方向正确而按语料路由或删除该双语料硬门。

## 5. Candidate-set perturbations

### Duplicate all

Beta1严格重复性质在完整cohort成立：两语料bag log-evidence maximum absolute error均为
`8.88e-16`；frame maximum absolute error为HMM `3.89e-16`、HCS `5.55e-16`，远低于`1e-10`门。
两个duplicate exactness checks均通过。Beta0 bag evidence按理论增加`log(2)=0.69314718`，而frame posterior
ranking保持不变，进一步确认统计语义正确。

### Near duplicate与grid thinning

| corpus | perturbation | bag abs change beta0→beta1 | frame rho beta0→beta1 |
|---|---|---:|---:|
| HateMM | near duplicate | 0.379722 → 0.078629 | 0.970912 → 0.967134 |
| HateMM | thin grid | 0.668009 → 0.106735 | 0.940221 → 0.921680 |
| HateClipSeg | near duplicate | 0.328018 → 0.061795 | 0.968162 → 0.961351 |
| HateClipSeg | thin grid | 0.674819 → 0.085719 | 0.881376 → 0.848040 |

Beta1确实显著稳定了bag log evidence，所以四个bag gates通过；但四个frame ranking comparisons全部变差，
故相应frame gates全部失败。每个比较使用完整paired cohort：HMM `214/214`、HCS `79/79`，
`paired_frame_undefined=0`。不存在两beta使用不同denominator或undefined被丢弃后造成的误判。

这说明PS correction满足全局choice-evidence的重复代数，却没有把这种性质转化成预注册要求的frame ranking
稳定性；不能只报告通过的bag项而忽略frame项。

## 6. Verdict与test使用边界

当前verdict的11个failure逐项成立：

- HMM：within、whole top、near-whole top、long mass、near frame stability、thin frame stability失败；
- HCS：within、long mass、wrong-top方向、near frame stability、thin frame stability失败。

任何一个failure已足以触发STOP；当前是两个语料、多组独立门同时失败。通过的duplicate identity、bag
stability、HMM wrong-top方向与部分length项只能说明PSL公式按定义运行，不能救回formal eligibility。

程序没有optimizer、backward或checkpoint selection。Test GT只在冻结prediction后用于共享metrics、IoU
correctability和developmental error analysis，未进入梯度或checkpoint选择，符合Rule 10。由此看到的
beta、语料差异、length与wrong-top结果不能用于当前候选的post-hoc parameter/formula search；后续受其影响的
test证据也必须继续标为iterative/developmental。

最终结论：**POST-RUN RESULT-CHAIN PASS；FROZEN PREMISE FAIL；维持
`STOP_BEFORE_FORMAL_METHOD`，禁止beta扫描或公式修补。**
