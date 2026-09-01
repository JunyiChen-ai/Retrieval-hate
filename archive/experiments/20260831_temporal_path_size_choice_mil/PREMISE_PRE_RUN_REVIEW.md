# Frozen developmental test premise pre-run review

截至 2026-08-31。审查范围：`README.md`、`analyze_frozen_premise.py`、`test_premise.py`、
`src/proposal_mil.py`，以及 frozen multimodal P-MIL artifacts
`runs/20260831_multimodal_pmil_baseline/pilot_seed234/{hatemm,hateclipseg}/`。

## 最终裁定

**PASS，可以运行一次冻结的 HateMM + HateClipSeg 完整 developmental test premise。**

本 PASS 只授权当前固定 `beta=0/1`、utility、path-size、frame marginal、IoU/length定义和三种candidate-set
perturbation的无训练分析；不授权训练 formal method、扫描 beta、修改 utility/readout/cutoff、按 corpus 选
规则或把 premise 的 pooled 指标称为正式方法/SOTA。运行前没有生成
`runs/20260831_temporal_path_size_choice_mil/` 输出。

审查中发现的 verdict 可靠性问题已经在首次完整 premise 前修复，修复后的 compile、8项单元测试、
frozen artifact coverage复算和小规模真实 reconstruction 均通过。没有剩余会改变当前 premise 观察或结论的
blocker。

## 1. Frozen P-MIL 兼容性与正式 score identity

- 共享 `MultimodalPMIL` 的 module层级、参数名、tensor shape、ROI padding/coordinates、三模态顺序
  `visual/audio/text` 与归档 P-MIL baseline实现一致。两个 frozen `model.pt` 均能 strict-load，所有参数键
  匹配。
- proposal producer仍使用各 corpus config登记的 corpus-specific MultiHateLoc seed-234 checkpoint、
  hidden/embed、1 fps features与原始 proposal recipe：9个relative thresholds、16 peaks、固定 widths、
  maximum 256并保留whole proposal。
- frozen utility前先按原实现计算每个 modality的
  `softmax(cas)[hate] * sigmoid(attention) * sigmoid(completeness)`，再跨三模态求均值。没有改成CAS-only、
  分量均值后再相乘或使用validation选择的其他branch。
- `analyze_frozen_premise.py` 对每个正式test视频用共享模型重建原 P-MIL max-over-covering-proposals score；
  任一逐帧误差大于 `1e-6`立即停止，不会继续产生 premise verdict。
- 实际只读检查了每语料首个test视频及proposal数最大的test视频，共4个视频；HateMM含125/256 proposals，
  HCS含105/248 proposals，四条重建score与正式 `score_pmil` 的最大绝对误差均为0。
- 两个 frozen score文件也重新通过完整共享 evaluator cohort核查：HateMM 214 videos / 29,269 frames，
  HCS 79 / 18,839；无missing/extra、逐视频length/finite通过，共享 evaluator完整结果与正式metrics一致。
  Frozen checkpoint分别由official-validation在epoch 15与5选定。该已审计baseline是本 premise唯一输入，
  premise不重新训练或重选checkpoint。

因此共享实现既能加载 frozen checkpoint，也有正式prediction identity硬门；不会在不同模型或不同proposal
集合上分析后仍写 verdict。

## 2. Path-size公式与 beta=1 duplication性质

实现对每个半开区间proposal `[start,end)`逐秒累计occupancy `n_t`，再计算

`PS_p = mean_{t in p}(1 / n_t)`，

与README固定的离散公式一致。bounds必须是1 fps整数、位于`[0,T]`且非空；utility数量必须和proposal数量
相等并全部finite。

`choice_readout`固定使用 `v_p=u_p+beta*log(PS_p)`、稳定log-sum-exp、proposal softmax posterior；
`beta=1`时把**整个候选集**复制`J`次会使每个occupancy乘`J`、每个PS除`J`，从而bag log evidence与
覆盖posterior marginal严格不变。测试同时覆盖纯重复组和存在交叠alternatives时的全候选集复制，均在
`1e-12`数值精度通过。README已收窄措辞：只复制某个subgroup时不声称严格不变。

审查修复了一个共享边界错误：旧 `choice_readout`先把proposal转成整数再验证，fractional endpoint会被静默
截断。现在先验证原始bounds再转换，并补充fractional、count mismatch与non-finite utility拒绝测试。正式
producer本来就生成整数bounds，因此这项修复不改变frozen score，只避免perturbation或未来消费者悄悄改变
interval geometry。

## 3. Utility、posterior与frame readout

- `s_p`固定为frozen三模态full score，先clamp到`[1e-6,1-1e-6]`，再用
  `log(s_p)-log(1-s_p)`得到utility；没有拟合参数或per-corpus arm。
- `beta=0`与`beta=1`都只在同一个proposal conditional choice set做softmax。frame score是覆盖该秒的所有
  proposal posterior之和；posterior总和为1，所以frame marginal有限且在`[0,1]`。
- Outside option在frozen premise不拟合，因为它只会给同一视频全部proposal conditional posterior乘同一
  hate-choice概率，不改变within-video ranking、top proposal或conditional long mass。故当前pooled AP/ROC
  只能作为conditional diagnostic完整报告，不能视为含outside formal model的performance或SOTA证据；
  README也没有把它们放进premise通过门。
- 原 P-MIL max readout只用于score identity检查；premise beta0并不冒充原baseline frame score。真正比较的
  beta0/beta1都走同一个posterior marginal，因此差异只来自固定path-size项。

## 4. IoU、length统计与 perturbations

- GT interval由1 fps二值数组的连续正段形成半开区间，temporal IoU使用标准intersection/union。correctable
  case只来自positive test videos，要求best proposal IoU `>=0.5`且frozen raw-score top IoU `<0.3`。
  两语料都必须至少有一个case，并且mean `logPS_error-logPS_best < 0`；这一方向门已在README明确冻结。
- exact whole top严格等于`[0,T)`；top duration ratio为`width/T`；near-whole top与long proposal都固定为
  `width >= 2T/3`；long mass是该集合posterior之和。四项按完整test cohort聚合，没有positive-only或
  per-corpus threshold选择。
- near duplicate只加入每个proposal合法且尚不存在的右移1秒interval；grid thinning按lexicographic排序取
  偶数位置并强制保留whole interval；duplicate-all保持每条utility逐项复制。新proposal utility仍由同一个
  frozen P-MIL独立计算。
- Near-duplicate/thinning的bag stability比较两beta的video-level mean absolute evidence change；frame
  stability在同一逐视频paired cohort比较mean Spearman。现在强制这个paired cohort等于完整test cohort；
  任一beta出现undefined就fail-closed，不能分别删除不同视频后比较两个均值。
- 全体复制的beta1 bag与frame invariance现在都用全cohort **maximum** absolute error `<=1e-10`。审查前bag门
  只看mean，理论上可能掩盖单视频违例；该问题已修复并有专门单测。

## 5. Metrics、双语料 STOP gate 与无结果后自由度

- Beta0/beta1的pooled AP、pooled ROC、within-video macro ROC全部直接调用仓库唯一
  `eval_baseline_scores.py::evaluate_scores`。within cohort由共享GT与hate IDs确定，未复制指标实现。
- 两语料均必须同时存在；gate现在显式拒绝partial corpus dictionary。每个corpus逐项检查within不降、whole
  top严格下降、median duration不升、near-whole top严格下降、long mass不升、wrong-top path-size方向、
  duplicate exactness，以及near/thin的bag与frame双稳定性。
- 所有门对None/NaN/Inf和undefined Spearman fail-closed；不会发生`None > None`异常后留下无verdict，也不会
  把不同denominator的Spearman均值当作通过。任一失败只输出`STOP_BEFORE_FORMAL_METHOD`。
- 当前代码只构造`beta in (0,1)`，CORPORA固定为HateMM/HCS，utility、EPS、IoU阈值、2/3 length cutoff、
  perturbation recipe和比较方向均在完整run前冻结。没有beta scan、best branch、per-corpus routing、
  post-hoc calibration或用pooled指标救within failure的入口。

## 6. Test-GT边界与依赖隔离

- 该程序没有optimizer、backward、训练dataset或checkpoint selection。它读取的模型、proposal utility和正式
  scores都已冻结。
- Test GT只用于共享evaluation、oracle IoU correctability与统计分层；不进入proposal generation、utility、
  model forward、parameter update或checkpoint选择。输出固定标记`split=test`、
  `developmental_test_evidence=true`、`test_labels_used_for_gradient_or_checkpoint_selection=false`。
- 按Rule 10，这个结果只能inform后续新机制；任何后续test结果属于iterative/developmental evidence，不能
  表述为未揭盲confirmatory结果。
- 正式分析脚本只import `src/`共享模块和`scripts/reproduction_baselines/`的数据、模型及唯一evaluator；没有
  import其他实验目录。单测仅通过模块名加载本实验自己的gate，不构成跨实验依赖。

## 7. 实际执行的检查

- `python -m unittest -v .../test_premise.py`：8/8 PASS。
- 范围内三个Python文件compile PASS；diff-check PASS。
- 两个 frozen P-MIL checkpoint strict state-dict load PASS。
- 4-video真实 frozen score reconstruction：逐帧max error均为0；包含两语料各自proposal-count最大test视频。
- Frozen score exact coverage、finite/length与共享 evaluator复算 PASS。
- Small perturbation contract检查确认：全体复制时beta1 bag evidence误差约`4.44e-16`，frame max error不超过
  `2.22e-16`；未把该小样本的near/thin数值当作premise结果或用于修改规则。
- 未运行完整premise、未写任何 premise run输出、未训练模型。

最终结论：**PASS FOR ONE FROZEN DEVELOPMENTAL TEST PREMISE RUN。**
