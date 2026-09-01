# 已停止：OmniVTG raw-statistic reopening controls

**淘汰原因：独立 technical review 前的 label-free structural precheck 已证明冻结
protocol不可能通过。HMM `hate_video_329` 只有`7.105011s`，固定8秒plan只有一个
block；其旧raw prediction成功且interval为`[2,6]`，因此必然触发预注册的`N<2`
与raw-success→corrupted-failure integrity FAIL。未启动GPU、未读取新增test GT、无
run artifact。**

截至 2026-08-31。该轮是 Process RESET 后的 identifiability audit，不是 method
candidate，不增加候选计数，不训练或修改 OmniVTG，也不授权 teacher distillation。

## 为什么允许审计这一个既有 statistic

固定 OmniVTG diagnostic 已经用完全相同的 checkpoint、query、2 fps protocol 与
interval parser覆盖 HMM/HateClipSeg positive test cohort。其 raw interval score 的
within ROC 为 `.626380/.539010`，但旧 gate 要求超过另一个 structured control，故
当时正确裁定 `STOP_BEFORE_STUDENT`。RESET reopening gate 问的是更早的一层：该
external/native visual local observation 是否真的携带跨语料、content-aligned temporal
direction。旧实验没有回答 time-shuffle、mean-repeated、position-only 与 carrier
strata，因此这里只复用冻结 predictions 补控制，不重跑模型、不改 query。

输入固定为：

- `runs/20260831_omnivtg_grounder_diagnostic/formal/hatemm/predictions.jsonl`
- `runs/20260831_omnivtg_grounder_diagnostic/formal/hateclipseg/predictions.jsonl`

合法 interval 仍按原 protocol 转为 1 fps binary score；推理/解析失败仍是全零。
test GT 只进入 evaluator，结果属于 iterative/developmental evidence。

## 冻结 controls

### 1. Time-shuffle

每个 both-class positive video 对 raw score 做最多 16 个均匀、唯一、非零 circular
shifts；先在视频内平均，再等权平均视频。每个 corpus 要求
`raw within - shift mean >= .020`。

### 2. Mean-repeated

把每视频 raw score temporal mean 重复到全部秒，保留预测覆盖率和 video identity、
删除 local ordering。完整 score map 必须调用 canonical `evaluate_scores`；每个 corpus
要求 `raw within - mean-repeated within >= .020`。

### 3. Target-preserving temporal-corruption position control

主 position control 保留 target video identity、global content、帧 multiset、长度、模型、
query 与 parser，只破坏 content 和原时间位置的对应。对每个 target source video 按固定
8 秒 blocks 切分；若有 `N>=2` 个 blocks，使用同一排列
`[floor(N/2),...,N-1,0,...,floor(N/2)-1]`，即按 block 做半周 rotation。排列不使用 GT、
video ID seed、随机数或哈希。对 permuted video 用同一冻结 OmniVTG 重跑一次，再把预测
interval 按 inverse block mapping 转回原 1 fps grid。`N<2`、media转换失败、或 raw
成功但 corrupted inference/parse失败均 fail closed。

每个 corpus 要求 `raw within - inverse-mapped corrupted within >= .020`。如果模型只
由 target identity/topic 决定固定位置，permutation 后仍会输出该位置，inverse mapping
会破坏原排序；若输出真正跟随 content，inverse mapping 应恢复原排序。

原先的 cross-test donor position control只保留为补充 transductive diagnostic：对每个
target，把其他 frozen positive-test outputs 按相对位置缩放到 target 长度，按 sorted ID
取最多16个均匀唯一donors。它显式记录
`donor_pool_source=frozen positive test cohort`、
`transductive_diagnostic_only=true`、
`used_for_score_generation_after_old_test_video_label_selection=true`，不承担主 gate，
不进入训练、部署或 checkpoint selection。

### 4. Carrier strata

Stratum membership 在 evaluator 读取 frame GT 前由冻结、label-blind cache 定义：

- ASR coverage：既有 lexical premise 的 `score_speech` 均值，固定 sparse `<=1/3`、
  mixed `(1/3,2/3)`、dense `>=2/3`；
- OCR coverage：`ocr_bert_1fps` 每秒 feature 是否为非零向量的比例，使用相同三档；
- visual dynamics：`clip_b16_1fps` 相邻秒 L2-normalized cosine distance 的视频中位数；
  static/dynamic 阈值只用对应 corpus 的 positive-train videos计算中位数，test 不参与
  threshold。每视频只计算 `d_t=1-cosine(x_t,x_{t-1})`,`t=1..T-1`；任一row zero-norm
  或 `T<2` 都 fail closed。保存threshold、positive-train视频数及每个test视频statistic。

OCR presence 的上游依据是冻结 OCR window JSONL；required train/test video 不在上游
cache 时是 `missing_upstream` 并使 coverage fail，不能当作 empty OCR。只有上游存在且
`ocr_bert_1fps` row 为零才算 observed empty；OCR、ASR、CLIP 与 raw score长度必须完全
一致。

这些是 video-level carrier strata，不把“无 ASR”等同于 static/OCR/visual-only。
每个 corpus 的 ASR 三档、OCR 三档、static/dynamic 两档都必须各有至少 5 个
both-class positive test videos；每一档都要求 raw macro ROC 减 matched per-video
time-shuffle macro ROC `>=.020`。不足 5 个本身就是 coverage FAIL，不合并档位、不看
GT 改边界。

## 完整 reopening gate

HMM 与 HateClipSeg 必须全部通过：

1. exact prediction/cohort/length/finite contract；
2. aggregate time-shuffle margin `>=.020`；
3. mean-repeated margin `>=.020`；
4. target-preserving temporal-corruption margin `>=.020`，且不存在 raw-success 到
   corrupted-failure；
5. 上述八个 carrier strata 均满足 `n>=5` 且 raw-minus-shift `>=.020`。

任一失败即 `KEEP_CANDIDATE_FREEZE`。通过也只记
`REOPENING_EVIDENCE_PASS_ONLY`，旧 OmniVTG `STOP_BEFORE_STUDENT` 仍然有效；不能自动
恢复 teacher/student/distillation。任何后续 source adaptation 仍须单独解释为何不是
teacher-order KD 或在旧失败source上替换head/loss，并通过独立 novelty premise review。

正式代码与 evaluation 必须先经独立 pre-run review，且只能调用 canonical evaluator
及其 rank/macro primitives，不得复制 ROC/AP 实现。输出固定为
`runs/20260831_omnivtg_reopening_controls/main/`。

## Technical verdict 与去向

独立 technical reviewer `metrics_audit` 裁定 `FAIL / DO NOT RUN`，详见
`PRE_RUN_REVIEW.md`。只读 structural precheck 还发现 HMM `hate_video_184` 的1fps
grid为46而media duration为45.0秒、`hate_video_89`为157与155.989秒；当前inverse
mapping会在尾秒抛错而不是生成可审计结果。这是实现bug，但无需为本premise修复后重跑，
因为`hate_video_329`已经使冻结gate确定失败。

裁定`KEEP_CANDIDATE_FREEZE`。该轮没有生成method candidate，RESET epoch计数保持
`0/3`；旧OmniVTG `STOP_BEFORE_STUDENT`继续有效。
