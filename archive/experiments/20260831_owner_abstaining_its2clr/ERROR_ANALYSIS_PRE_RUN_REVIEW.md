# Frozen test error analysis pre-run review

截至 2026-08-31。审查对象：`analyze_test_errors.py`、新增
`test_error_analysis.py`，以及它们直接调用的 frozen anchor/broadcast/core artifacts、core model、
train-only centroid helper与共享数据协议。

## 裁定

**PASS，可以运行 frozen post-training developmental test error analysis。** review中修复了 formal cohort
静默取交集、carrier重算未对齐frozen prediction、Spearman denominator不透明、字段scope含混及输出目录未
约束等会改变解释的问题。修复后 compile、2项synthetic tests和两个语料的formal artifact只读覆盖检查
均通过。

没有启动新训练、没有选择 checkpoint，也没有运行完整 test analysis。只读检查中使用test GT确认正式
cohort/length/eligible集合；该用途属于允许的developmental error analysis。

## Review 中修复的 blockers

1. **core/broadcast per-video AUC曾只取集合交集。** 缺少某视频时原代码会静默缩小eligible cohort，改变
   delta、occupancy和correlation。现要求anchor/broadcast/core score maps与test GT exact-cover，所有长度
   与finite检查通过；core和broadcast `per_video_auc` keys还必须精确等于GT同时含两类frames的集合。
2. **deletion-carrier重算未证明使用同一frozen predictor。** 现逐test视频比较core checkpoint重算的
   fused score与正式core `scores.jsonl`，shape必须一致且maximum absolute error不超过`1e-6`；输出记录
   corpus-level maximum error和`prediction_changed=false`。因此carrier诊断不能悄然变成另一条prediction。
3. **selected-checkpoint provenance未检查。** 现要求core checkpoint corpus/arm正确，checkpoint中的
   `selected_epoch`与formal `train_log.json`一致，且train log明确test未用于gradient或checkpoint selection。
4. **per-video frame Spearman静默含NaN。** 原实现把所有statistics加入list，再用`nanmean/nanmedian`，没有
   denominator。现统一使用对齐/finite/constant检查，输出finite mean/median、`n_finite`和`n_undefined`。
5. **carrier字段scope不准确。** rate只在within-AUC eligible videos上与delta对齐，现字段明确改为
   `core_test_carrier_rate_top_third_on_within_eligible_videos`；top divisor从frozen checkpoint读取，不再
   硬编码3。
6. **输出位置可任意指定。** 现要求analysis JSON直接写在传入的frozen run root下；不能把正式结果写进
   `data/`、源码目录或无关run。

## 1. Frozen artifacts 与 test information boundary

- anchor/broadcast/core的正式 `metrics.json` 均必须声明目标corpus、split `test`、branch `score_core`且
  evaluator coverage无missing/extra。三臂正式 `scores.jsonl` 都只读加载并与GT exact-cover。
- 实查 HateMM三臂均为214 test videos、85个within eligible videos；HateClipSeg三臂均为79/67。所有
  score arrays与相应GT shape一致且finite。
- deletion carrier只加载formal core `model.pt`；checkpoint metadata与formal train log selected epoch
  强制一致。anchor/broadcast不重新推理，只读取正式scores/metrics。
- 脚本没有optimizer、loss、backward、train mode或checkpoint写入。`pseudo_for_video`受`torch.no_grad()`
  保护，core backbone设为eval。
- test GT只用于：确定per-video AUC eligible cohort、计算GT positive occupancy、验证score coverage/length，
  以及解释frozen core-vs-broadcast结果。GT不进入model carrier deletion、centroid、gradient或selection。

## 2. Deletion-carrier诊断与train centroid

- `negative_centroids`的IDs/labels只来自`supervised_split(corpus,"train")`；helper仅遍历label为0的目标语料
  train videos，并分别对visual/audio/text所有train-negative frames求均值。不读取validation/test labels、
  GT或其他主数据集。
- 每test视频用frozen core backbone重算原始fused score。对每个modality，centroid replacement和同视频相邻
  秒replacement都只重算该秒对应modality embedding与fused head；原始DMS weights冻结。stable carrier定义
  严格为两个deletion effects都大于0。
- carrier rate只在frozen core fused score的top-`ceil(T/k)`秒上计算，`k`从checkpoint读取。modality name
  顺序来自checkpoint dims，与model state的visual/audio/text顺序一致。
- replacement logits仅用于诊断stable mask；正式frame prediction仍是未替换的core fused score。逐视频
  frozen-score一致性门会在任何prediction变化时停止。
- 该rate不是test-time carrier selector、输出branch、calibration或routing rule。输出新增
  `carrier_diagnostic_policy`明确centroid来源、prediction未改变和diagnostic-only身份。

## 3. Delta、occupancy、strata 与Spearman

- `delta`逐同一sorted eligible ID计算
  `core formal per-video AUC - broadcast formal per-video AUC`；mean、median、improved/worsened fractions的
  denominator均为exact eligible cohort。相等视频不计入improved或worsened。
- occupancy是同一视频1fps GT array的positive mean。strata为`<=1/3`、`(1/3,2/3]`、`>2/3`；空stratum
  输出`n=0, mean=null`，不会产生非有限值。
- `delta_vs_gt_positive_fraction`与每modality
  `carrier_rate_vs_delta_auc`均保持同一eligible ID顺序，并通过`safe_spearman`检查shape、finite和constant。
  correlation输入少于2或任一侧constant时输出`rho=null`及实际n。
- `core_vs_broadcast_per_video_score_spearman`对每个eligible视频的两条完整frame score vector计算；finite与
  undefined视频分别计数，不能用隐式drop后的均值冒充全cohort结果。
- `pooled_absolute_score_difference`在三臂coverage检查后，对全部test videos/frames拼池计算core与broadcast
  absolute difference mean/median。它不是within-video量，字段名已明确pooled。

## 4. Formal fields、diagnostic身份与输出

- 输出保存三臂正式scores/metrics路径及其正式三指标；另保存core checkpoint/train log路径与selected
  epoch。formal数字直接转录共享evaluator，不在analysis内重写AUC/AP算法。
- anchor只作为正式baseline参照；核心诊断仅比较frozen core和broadcast。carrier rates、删除effects、
  score differences与correlations均不会写回scores或产生新method arm。
- 顶层固定写`split=test`、`developmental_error_analysis=true`、
  `test_labels_used_for_gradient_or_checkpoint_selection=false`，并明确design use只判断tiny gains是否对应
  carrier-dependent reranking，禁止test-time routing。
- output必须直接位于`runs/20260831_owner_abstaining_its2clr/...`的传入frozen run root。脚本只新增一份
  analysis JSON，不修改formal score、metric或checkpoint artifacts。

## 5. 运行与数值风险

- 最重步骤是每语料扫描negative-train features求三个centroids，并对每个test视频执行原始forward及每模态
  两种local replacements。视频逐个处理，返回CPU deletion arrays，不累计整套GPU graph；显存风险低，
  但HateMM会有明显feature I/O。
- formal score、GT、carrier rate与correlation输入均新增shape/finite checks。carrier rate由boolean mean
  得到，必在`[0,1]`；absolute difference/delta由已验证finite正式scores/metrics产生。
- 若frozen run不完整、checkpoint与train log不一致、test cohort/length变化或model重算不能恢复正式core
  score，analysis会在写output前失败。

## 6. 已执行检查

- 2项synthetic tests通过：Spearman反序/constant/shape/非有限输入；delta与occupancy按ID对齐。
- `analyze_test_errors.py`与`test_error_analysis.py` compile通过。
- 只读formal artifact检查通过：HateMM anchor/broadcast/core均214 scores与85 AUC entries；HateClipSeg均
  79/67；三臂score/GT shape和finite检查全部通过。
- 未调用`model_carrier_rates`全test循环或`main`，因此未生成正式analysis artifact。

最终裁定：**PASS FOR FROZEN DEVELOPMENTAL TEST ERROR ANALYSIS**。
