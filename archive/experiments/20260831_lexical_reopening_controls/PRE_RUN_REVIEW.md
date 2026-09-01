# Pre-run protocol review

截至 2026-08-31。独立 reviewer：`process_reviewer`。

Verdict：`FAIL — DO NOT RUN`。

决定性 blocker：冻结 statistic 在 `score_speech=0` 秒的 local text 恒为空，
TF-IDF 向量恒为零，logistic score 恒等于 classifier intercept。因此 carrier-absent
masked cohort 只要包含两类 GT，ROC 解析上固定为 `.500`，无法满足 `.520` gate。
这已经证明它不能提供 ASR-carrier-absent local direction，因而不能建立 reviewer
要求的完整 HCS carrier coverage；该 mask 本身不区分 static/OCR/visual-only，
无需读取新增 test GT 或运行形式化 control。

Reviewer 还指出，若未来审查另一项 statistic，protocol 必须同时修正：

- sparse/mixed/dense 每层至少 5 个 both-class positive videos，且各层 raw-minus-
  matched-shift 都须 `>=.020`，不足覆盖即 fail；
- position-only 主 control 只用 positive-train videos 构造 20-bin template，先视频
  内 bin mean、再跨视频等权平均，空 train bin fail closed；
- “无 ASR”不能被过度表述成分别证明 static/OCR/visual-only；这些 strata 必须由
  冻结、label-blind 信息单独定义；
- producer 不得读取 test GT/test labels；完整 score maps 调 canonical
  `evaluate_scores`，masked diagnostics 只能调用 canonical rank/macro primitives，
  不得复制 ROC/AP 实现。

推荐处置已执行：不补代码、不运行，直接记录 `KEEP_CANDIDATE_FREEZE`。
