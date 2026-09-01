# DSANet semantic-alignment raw-statistic reopening premise

截至 2026-08-31。本轮只审计一个 target-train-only local statistic，不是method
candidate，不作novelty claim，不替代MultiHateLoc starting architecture，也不增加
RESET epoch候选计数。

**淘汰原因（2026-08-31）**：正式双语料test control已完成，权威输出为
`runs/20260831_dsanet_alignment_reopening/main/metrics.json`。HMM/HCS raw
AP/pooled ROC/within分别为`.445647/.725550/.573931`与
`.576017/.545827/.568621`。两语料time-shuffle、mean-repeated与reversal-within
均通过，但两者carrier-strata均失败；HMM reversal Spearman失败，HCS
positive-train position-only margin为`-.032212`并失败。联合gate失败，裁定
`KEEP_CANDIDATE_FREEZE`；不扫seed、branch或control参数，不进入novelty/method。

## 资格依据

只读developmental test筛查发现，历史DSANet seed-234 `score_align`在HMM/HCS的
within ROC为`.574108/.568624`，比其binary-collapse headline `score_mlp`更稳定地保留
局部语义排序。现行规则允许在一个已定义方法内部用validation选择训练超参数、训练配置
和checkpoint，因此本轮直接复用这两个各自独立训练、validation-selected的official
seed-234 checkpoint，不重新搜索参数。审计`score_align`这一预先存在的raw branch来自
允许的test error analysis，不由validation branch比较决定。

冻结raw statistic是同一checkpoint的
`score_align(t)=1-softmax(logits2(t))[normal]`。不使用headline MLP、branch blend、
calibration、smoothing或按语料routing。

## Producer / evaluator隔离

既有训练各自使用本语料train labels，validation用于训练配置与checkpoint选择。Score producer读取选定
checkpoint、train/test split membership与label-blind CLIP/ASR/OCR cache；test cohort用
`src/scoped_video_protocol.py::evaluator_test_ids`固定排除无localization gold的manifest
成员，不读取test video labels或frame/span GT。Evaluator才读取test GT与test video
labels，结果均为iterative/developmental test evidence。

## 冻结 controls

1. **Time-shuffle**：每个both-class positive test video最多16个均匀、唯一、非零
   circular shifts；先视频内均值再视频间等权。每语料raw-minus-shift `>=.020`。
2. **Mean-repeated**：每视频raw temporal mean重复到全部秒，保留video level、删除local
   ordering；raw-minus-control `>=.020`。
3. **Positive-train position-only**：用选定checkpoint对本语料positive-train videos做
   natural full-sequence inference。相对位置分20个bins，先每视频bin mean、再跨视频
   等权；任一train bin无观测即fail。template应用于全部test视频，不读取test label。
   raw-minus-position within `>=.020`。
4. **Input-reversal equivariance**：对每个test视频把1fps CLIP sequence完全reverse，
   用同一checkpoint推理，再把输出reverse回原index。content-aligned statistic应随内容
   反转并在inverse后恢复；pure position/video-identity shortcut会被镜像。每语料要求
   inverse-reversal within不低于raw超过`.020`，且在eligible videos上raw与inverse score
   的equal-video Spearman中位数`>=.50`。这不是“raw必须打败corruption”的错误方向。
5. **Carrier strata**：ASR coverage由已冻结`score_speech`定义；OCR coverage只在上游
   OCR window cache存在时由`ocr_bert_1fps`非零row定义；visual static/dynamic用
   positive-train CLIP相邻秒cosine-distance视频中位数冻结阈值。ASR sparse/mixed/dense、
   OCR sparse/mixed/dense、visual static/dynamic共8档，每档至少5个both-class positive
   test videos，且raw-minus-time-shuffle均`>=.020`。不足即coverage FAIL，不合并档位。

所有完整score maps调用canonical `evaluate_scores`；shift与rank correlation只调用
canonical `frame_eval_common` primitives或明确的non-performance correlation诊断，绝不
复制ROC/AP。Exact cohort、shape、finite、train/test isolation与OCR upstream missing均
fail closed。

## Reopening gate

HMM/HCS必须同时通过：aggregate time-shuffle、mean-repeated、positive-train
position-only、input-reversal equivariance、8个carrier strata与integrity gates。
任一失败即`KEEP_CANDIDATE_FREEZE`，不扫seed、loss、temperature、window、head或branch。
通过只记`REOPENING_EVIDENCE_PASS_ONLY`，下一步才允许对真正跨任务source做独立novelty
premise；DSANet及其alignment branch已经进入本任务baseline，绝不能作为novelty来源。

正式训练、推理与evaluation前必须由独立technical reviewer PASS。长任务以detached方式
写入`runs/20260831_dsanet_alignment_reopening/main/`，不记录或使用任何hash/checksum。
