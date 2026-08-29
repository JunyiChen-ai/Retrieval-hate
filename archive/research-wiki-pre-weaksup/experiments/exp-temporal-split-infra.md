---
type: experiment
node_id: exp:exp-temporal-split-infra
title: "Temporal split infrastructure for MHClip EN/ZH: upload-date collection + survivor-bias audit (INFRA)"
idea_id: "idea:cross-dataset-knn-memory"
verdict: yes
confidence: high
date: "2026-07-03"
hardware: ""
duration: ""
provenance: "slurm/logs/collect_dates_{12170,12171}.out; data/gt/temporal_split_stats.json"
added: 2026-07-02T21:20:20Z
tags: ["hateful-video", "temporal-split", "evolving-memory", "survivor-bias", "infrastructure", "MHClip", "iteration-3", "wave-1"]
---

# Temporal split infrastructure for MHClip EN/ZH: upload-date collection + survivor-bias audit (INFRA)

**verdict:** `yes`  ·  **confidence:** `high`  ·  tests `idea:cross-dataset-knn-memory`

## Metrics

**Datability (upload-date collection, yt-dlp / Bilibili API):**

| Corpus | queried | datable | rate | job | elapsed |
|---|---|---|---|---|---|
| MHC (EN, YouTube) | 890 | 781 | **87.8%** | 12170 | 59.3 min |
| MHC_zh (ZH, Bilibili) | 897 | 804 | **89.6%** | 12171 | 33.0 min |

Within our split universe (train+val+test): EN 771/790 = **97.6%** datable, ZH 796/806 = **98.8%**;
undatable pinned into train per protocol (EN 19, ZH 10).

**Survivor bias — dead-link rate by label (FULL-count, supersedes the n=5 probe):**

| Label | MHC (EN) dead % | MHC_zh (ZH) dead % |
|---|---|---|
| Hateful | 11.1 (8/72) | **18.8 (21/112)** |
| Offensive | **21.2 (46/217)** | 11.1 (20/180) |
| Normal | 9.2 (55/601) | 8.6 (52/605) |

The reconnaissance-time "ZH-Hateful ~60% dead" estimate (5 probe samples) was a large
OVERESTIMATE; true value 18.8%. Bias direction confirmed (harmful classes die faster than
Normal) but magnitude is moderate. EN failure modes: unavailable/deleted 68, private 21,
age-gated 10; ZH: invisible(62002) 46, private(62012) 30, deleted 17.

**Temporal splits produced** — `data/gt/{MHC,MHC_zh}_temporal/{train,val,test}.jsonl`
(same sizes as the random split: EN 549/80/161, ZH 579/78/149):

| Split | EN dates | EN pos% | ZH dates | ZH pos% |
|---|---|---|---|---|
| train | 2009-11-03 → 2023-06-18 | 33.9 | 2013-06-10 → 2023-08-10 | 34.0 |
| val | 2023-06-18 → 2024-01-16 | 21.2 | 2023-08-11 → 2023-11-03 | 24.4 |
| test | 2024-01-16 → 2024-05-12 | **24.2** | 2023-11-03 → 2024-05-13 | **24.8** |

Full statistics: `data/gt/temporal_split_stats.json`.

## Reasoning
Verdict=yes: infrastructure goal met — datability HIGH on both languages, temporal splits
produced, survivor bias quantified exactly (the reconnaissance 60% ZH-Hateful estimate was a
fragile n=5 overestimate). Unblocks the W4 evolving-memory protocol (DESIGN_iter3 Method B).

## Caveats
- **Label prior shift across time must be declared in any W4 result:** temporal-test positive
  rate ~24% vs temporal-train ~34% (both languages). A model can gain accuracy on the temporal
  test just by predicting more negatives; report macro-F1 and compare against a
  same-prior-shift floor, not against the random-split numbers.
- Survivor bias is now an exact, reportable number, but it means the temporal test slice
  under-represents the harmful content of its era (dead links skew Hateful/Offensive).
- Undatable samples (EN 19 / ZH 10) are forced into train — a conservative choice (cannot leak
  future into test) but slightly inflates train heterogeneity.
- Datability collected on 2026-07-02/03; YouTube/Bilibili availability decays, numbers are a
  snapshot.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

