---
type: idea
node_id: idea:counter-narrative-dissent-auxiliary
title: "Counter-Narrative dissent as an auxiliary target"
stage: proposed
outcome: pending
added: 2026-08-08T20:50:02Z
based_on: []
target_gaps: ["gap:G2"]
tags: ["counter-speech", "disagreement", "multihateclip"]
---

# Counter-Narrative dissent as an auxiliary target

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
MultiHateClip raw votes contain a Counter Narrative class (63 EN + 76 ZH) that majority aggregation destroys; predict dissent probability as an auxiliary head alongside the binary decision, targeting the field's worst failure mode (counter-speech / reportage judged hateful).

## Key risks
NOVELTY (2026-08-09): ADJACENT. arXiv 'counter speech AND (video OR visual OR meme)' = 0 hits; counterspeech DETECTION in video/multimodal is empty. BUT arXiv:2404.01651 (NAACL 2024, use-vs-mention censors counterspeech) owns the exact problem statement and is the rejection cite; FC-CONAN 2601.01350 owns paired hate-counterspeech data (post-response pairs, evaluation only); ImpliHateVid 2508.06570 already does contrastive learning on hateful video with label-defined pairs. FEASIBILITY was the flagged risk and P-B ANSWERED IT NEGATIVELY: content-matched opposite-stance pairs do not exist at usable density (1 verified pair). Ship as a component of agreement-shaped-retrieval-memory, not standalone. Must be framed as probability of annotator dissent, never objective stance truth.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

