# 已停止：Lexical-locality reopening controls

**淘汰原因：独立 pre-run review 证明该 statistic 在 ASR-carrier-absent 秒上恒为
classifier intercept，within ROC 解析上固定为 `.500`，不可能通过 Rule 14；未写
正式代码、未读取新增 test GT、未运行。**

截至 2026-08-31。该轮只补齐 Process RESET 要求的 identifiability evidence，
不是 method candidate，不增加新 epoch 的候选计数，也不授权训练方法。

## 固定 observation

沿用已经独立审计的同一 raw statistic：每个语料独立用本语料 train video
labels 和 whole-video ASR 拟合 `char_wb` TF-IDF 3--5 gram logistic probe，再对
test 的 `[t-2,t+3)` ASR window 输出 1 fps decision score。不得改变窗口、模型、
特征或按语料选择分支。已有 raw test scores 来自
`runs/20260831_video_label_lexical_locality/premise/`。

test GT 只由共享 evaluator 及本轮 control evaluator 读取；不进入训练、参数选择
或 checkpoint selection。全部结果属于 iterative/developmental test evidence。

## 预注册 controls

1. **Time-shuffle**：沿用每视频最多 16 个均匀、唯一、非零 circular shifts，
   先在视频内平均再等权平均视频。raw lexical within 必须比它高至少 `.020`。
2. **Mean-repeated**：把每个 test 视频的 raw lexical temporal mean 重复到全部秒。
   这保留 video identity/level、删除 local ordering。raw lexical within 必须比它高
   至少 `.020`。
3. **Position-only**：只用 train 视频的 raw lexical score，按相对时间固定分成
   20 个等宽 bins；先求每视频 bin mean，再跨 train 视频等权平均。test 只按相对
   位置查该 train-derived template。raw lexical within 必须比它高至少 `.020`。
4. **Carrier strata**：`score_speech` 是与 local ASR window 同定义的 carrier
   availability。固定报告 test both-class positive videos 的 sparse (`<=1/3`)、
   mixed (`>1/3,<2/3`) 和 dense (`>=2/3`) coverage strata，各自 raw lexical、
   time-shuffle 与 margin。每个非空 stratum 都报告，但不因 test 分布改变边界。
5. **Carrier-absent necessity test**：在每个视频内只保留 `score_speech=0` 的秒；
   至少要有 5 个仍同时含两类 GT 的 positive videos，且其 raw lexical macro ROC
   必须 `>=.520`。这项只直接检验同一 observation 能否覆盖无 ASR carrier 的
   intervals；它本身不能把样本进一步认定为 static、OCR-only 或 visual-only。
   若 empty-text lexical score在这些秒上为常数，
   必须如实失败，不能用 OCR/visual 新分支补救本轮 statistic。

## Reopening gate

HateMM 与 HateClipSeg 必须各自同时满足：time-shuffle margin `>=.020`、
mean-repeated margin `>=.020`、position-only margin `>=.020`、carrier-absent
necessity test，以及精确 cohort/长度/finite checks。任一项失败即
`KEEP_CANDIDATE_FREEZE`；不得进入 novelty naming、查新或实现。

正式代码和 evaluation 在运行前必须由独立 reviewer 审查。运行只允许 CPU；
输出固定在 `runs/20260831_lexical_reopening_controls/main/`，并直接引用既有 raw
score artifact，不覆盖旧结果。

## Pre-run verdict 与去向

独立 reviewer 裁定 `FAIL — DO NOT RUN`，详见 `PRE_RUN_REVIEW.md`。由
`local_texts` 定义，`score_speech=0` 等价于该秒的 local text 为空；冻结的
TF-IDF 对空文本输出零向量，logistic `decision_function` 因而对所有这类秒输出
相同 intercept。只要 masked GT 同时有两类，其 within ROC 必为 `.500`，不可能
达到预注册的 `.520`。

因此本方向裁定 `KEEP_CANDIDATE_FREEZE`。没有创建 producer/evaluator/run
artifact，没有生成 method candidate，RESET epoch 候选计数保持 `0/3`。不得通过
加入 OCR/visual branch 续命，因为那将成为不同的组合 observation，必须重新从
Rule 14 premise 开始。
