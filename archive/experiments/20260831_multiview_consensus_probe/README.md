# 完成：Multiview local-order consensus upper bound

截至 2026-08-31。validation-only ensemble upper bound，不是候选方法，不读 test。
按 Rule 4，任何直接 consensus/fusion 都不能作为论文主方法。

输入固定为此前同语料 train bag-label linear probes 的 audio/visual/text/concat scores、
VERA neighbor score 与 POWA anchor。每个 local score 先在视频内转为 percentile rank，
再做无参数等权平均，最后按 consensus ordering 重排原 POWA score multiset。报告单源、
all-view、density-view 和 pair controls；不根据 validation GT 选权重。

只有 HMM/HCS 的同一 `all_view` consensus 都比 POWA within 至少 `+.020`、保住
pooled AP/ROC `-.010`，才说明值得研究单一 student；否则多专家路线停止。

权威输出 `runs/20260831_multiview_consensus_probe/analysis.json`，verdict
`MULTIVIEW_STUDENT_FEASIBLE`。同一 `all_view`：HMM AP/ROC/within
`.75894/.87467/.60065`（POWA within `.57193`）；HCS
`.51168/.60917/.55611`（POWA `.52707`）。逐视频 score multiset error 为 `0`。
该结果只授权查新/单-student 候选，不授权把 consensus 本身报告为方法。
