---
type: idea
node_id: idea:segment-keyed-retrieval-purity-loop
title: "Segment-keyed retrieval-purity closed loop (segment-keyed RGCL)"
stage: archived
outcome: negative
added: 2026-08-07T13:36:11Z
based_on: []
target_gaps: ["gap:G-A", "gap:G-B"]
tags: ["G-A", "G-B", "killed-by-pilot"]
---

# Segment-keyed retrieval-purity closed loop (segment-keyed RGCL)

**stage:** `archived`  ·  **outcome:** `negative`

KILLED BY PILOT: purity-selected segments land below chance on gold spans and cost macro-F1.

> **[勘误 2026-08-09]** 0.544 低于 chance 系 argmax 并列破序 artifact(随机破并列后 0.768 ≈ chance
> 0.762);NO-GO 维持(ratio 1.008 < 1.3×),−0.59 pt 为 null(CI [−2.16,+0.99])。详见
> `idea-stage/P2_FORENSIC_MEMO.md`。

## Thesis
A hard top-1 segment would be the only retrieval key, and the selector would be trained by the label purity of the neighbourhood its key retrieves - selection and retrieval supervising each other with no span gold. Ranked #1 by the cross-model jury on paper.

## Key risks
KILLED by pilot P2 (2026-08-08, HateMM-train 5-fold OOF, frozen thresholds): purity-selected segment hit rate 0.544 vs a chance rate of 0.762 (ratio 0.71, boot LB 0.487) - i.e. BELOW chance; adding the selected segment to the head moved macro-F1 0.8231 -> 0.8172 (-0.59 pt). Re-scored post-hoc against the coders' minimal sufficient intervals it reaches only 1.23x chance (0.444 vs 0.361, CI [0.343,0.545] straddling chance, random-selector control 0.354). Lesson: neighbourhood label purity does not identify hateful evidence segments in frozen CLIP space.

> **[勘误 2026-08-09]** 上述 "BELOW chance" / "cost macro-F1" 的表述需修正:0.544 低于 chance 系
> `pilots.py:175` 的 `np.argmax` 并列破序 artifact(K=20 邻居的离散统计量高度并列,按最低下标破并列
> 把 51.3% 仇恨视频送到 k=0,而 k=0 恰是 gold span 命中率最低的位置 0.339);随机破并列后同一选择器
> hit **0.768** ≈ chance **0.762**,即 *at* chance。minimal-interval 目标上去掉破并列为 0.410 vs
> 0.361(lift +0.050,CI [−0.009,+0.109])。−0.59 pt 的 CI 为 [−2.16,+0.99],是 null 而非实证伤害。
> **NO-GO / KILLED 判决维持**(ratio 1.008 < 1.3× 杀线),且被独立加强:`p_j` 的 within-video AUROC
> 仅 0.511 [0.488,0.533](视频级标签 AUROC 0.782)——"neighbourhood label purity does not identify
> hateful evidence segments" 这一课依然正确,只是依据换成机制性空档而非"低于 chance"。详见
> `idea-stage/P2_FORENSIC_MEMO.md`。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

