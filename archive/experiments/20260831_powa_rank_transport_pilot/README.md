# REJECTED — POWA score-mass-preserving temporal assignment pilot

淘汰原因（2026-08-31）：独立复审 PASS 后完成六个正式 Stage-V validation
run，但两个 pilot 语料均未通过冻结机制门，supervisor 裁定
`KILL_BEFORE_TEST`；未运行候选 test。权威裁定：
`runs/20260831_powa_rank_transport_pilot/stage_v_summary.json`。冻结方案见
`PILOT_PLAN.md`，独立查新见 `NOVELTY_SCOUT.md`，审查见
`PRE_RUN_REVIEW.md`。

唯一起点是各语料独立训练的 POWA。首个 benign-insertion 候选证明 HateMM
存在可学习的局部排序信号，但因同时改写绝对 frame-score mass 而使 pooled
AP/ROC 大幅下降。本轮考察是否能把两种职责结构性分离：POWA 提供每视频的
固定 score values，新的 temporal rank head 只决定这些 values 的时间归属。

`val_rank_diagnosis.py` 只在 validation split 将 POWA values 按上一失败模型的
rank 重排，用于确认失败归因；它不是候选方法，不训练、不选模，也不读取 test。

验证集诊断结果（权威 JSON 在
`runs/20260831_powa_rank_transport_pilot/`）：HateMM 固定 POWA values 后，
采用失败模型的 rank，pooled AP/ROC `.7577/.8744 → .7614/.8764`，within
`.5719 → .6435`；HateClipSeg 为 `.5064/.5985/.5271 →
.5087/.6032/.5340`。两者逐视频 1 fps score multiset 最大误差均为 `0`。
这只证明分解诊断，不证明新的 rank head 能学到该排序。

## 正式 Stage-V 结果与结论

六个 run 的 artifact/provenance/evaluator/source-snapshot integrity 全部通过。
HateMM negative-donor core 在 validation 选中 epoch 3：POWA 的 pooled
AP/ROC/within ROC 为 `.757659/.874425/.571931`，rank transport 为
`.757242/.872993/.584379`。它保持 pooled feasibility，但 within 只增
`.012448`，改善视频比例 `.5349`；未达到冻结的 `+.020/.55` 门。更关键的是
固定 center-first control 达 `.765501` within，远高于 learned candidate，
且 positive-donor arm 没有 feasible checkpoint，所以机制归因失败。

HateClipSeg negative-donor core 在 epoch 5：POWA 为
`.506391/.598539/.527072`，rank transport 为
`.507572/.600540/.532753`，within 只增 `.005681`，改善比例 `.4898`。
高正例率视频 within 从 `.547956` 到 `.562907`，增量 `.014951`，仍略低于
冻结的 `+.015`；core 未比 shifted-mask 高 `.010`，direct additive 与
transport within 几乎相同且 pooled 也可行，score-marginal constraint 的归因
不成立。

结论：固定 POWA score multiset 确实避免了上一轮的 pooled score-mass collapse，
但同语料 insertion/order supervision 没有学到足够强、非位置先验且可归因的
内容排序。按预注册不调 loss weight、window、epoch 或 gate，整轮淘汰。
