# Independent novelty reassessment

截至 2026-08-31。审查范围仅为 novelty 三项硬门；未实现、未训练、未生成 prediction。

## Verdict

**STOP。Novelty：3.8/10。**

1. **允许跨任务 adaptation：PASS。** Quote attribution、speaker/addressee attribution 和 conversational stance 可以作为来源。
2. **来源核心不得已经用于 hateful-video detection/localization：FAIL。** 项目旧候选
   `archive/refine-logs/lb_scgp/FINAL_PROPOSAL.md` 已把 direct-speaker endorsement、quotation/condemnation/reportage
   exception、cross-modal target-predicate binding 与 speaker-source/stance binding 共同适配到 hateful-video detection。
   因此当前候选不能再以“为 hostile proposition 识别 accountable source 和 endorsement scope”作为新的任务适配。
3. **新 adaptation 必须 non-trivial 且有独立机制：FAIL。** 当前新增的 proposition identity、timestamp 和
   temporal complete-path posterior 尚未定义出区别于“把旧 SCGP semantic state 按时间展开，再接 video-label MIL”的
   新学习原理。若 posterior 只是 frozen atom confidence 的乘积、路径聚合或普通 structured MIL，它仍是已占用语义证书的
   localization granularity extension，不足以构成不可拆的新核心。

LB-SCGP 虽明确不做 segment/span localization，但它已经占用了本候选原先声称的 task-specific semantic adaptation。
“旧方法是 whole-video、当前方法是 temporal”只能说明 endpoint 不同，不能自动通过来源占用门。

只有未来提出一种不依赖上述 SCGP state semantics、且 proposition identity conservation 本身产生可证明不同于普通
time-expanded MIL 的监督约束，才值得作为新候选重新审查。当前 specification 直接停止，不运行 frozen premise，
不实现模型。

