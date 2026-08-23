---
type: idea
node_id: idea:agreement-shaped-retrieval-memory
title: "Agreement-shaped retrieval memory (human-agreement retrieval)"
stage: proposed
outcome: pending
added: 2026-08-08T20:50:02Z
based_on: []
target_gaps: ["gap:G1", "gap:G2"]
tags: ["retrieval", "disagreement", "contrastive", "multihateclip"]
---

# Agreement-shaped retrieval memory (human-agreement retrieval)

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
Define contrastive pair topology by expected inter-annotator agreement instead of majority-label identity, and make memory entries carry the annotator vote distribution so the kNN read-out returns a distribution; binary decision = Hateful+Offensive mass, with contestedness and Counter-Narrative mass as separate beyond-accuracy outputs. Data substrate: official MultiHateClip release raw per-annotator votes (EN 21.3% / ZH 29.9% non-unanimous).

## Key risks
NOVELTY (2026-08-09 sweep): OPEN only for the COMPOSITE. Leg 'vote-distribution memory' pre-empted in weak form by UAKNN arXiv:2504.01508 (kNN over label distributions); leg 'distributional read-out' by Opt-ICL arXiv:2510.07105 (LeWiDi-2025 winner, retrieves rater examples in-context) and DeMeVa arXiv:2509.09524. ONLY leg with zero occupants in any field = agreement-defined contrastive PAIR TOPOLOGY - the paper must rest on that. Rejection cites: 2504.01508, 2510.07105, RGCL 2311.08110. No annotator IDs, so QuMAB/LPI-RIT-style annotator modelling is unavailable. PILOT P-A GO (AUROC 0.686 EN / 0.709 ZH; delta vs label-hardness +0.058/+0.091) BUT EN paired CI includes zero, ~80% of EN items have only 2 votes, and the hardness baseline was a crude 20-NN label fraction - FIRST follow-up must re-test delta against a TRAINED hardness baseline; if it vanishes, close.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

