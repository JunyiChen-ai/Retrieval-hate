# RESEARCH_BRIEF — Hateful Video Detection (Round 6, 2026-08-17)

> Rewritten 2026-08-17 for round 6 of idea discovery. The previous version (written during the
> SLURM-cluster era, before rounds 3-5 and before the 2026-08-13/14 kill series) described a
> machine, a goal state and a novelty story that no longer exist. Superseded content that is still
> true is folded in below; everything else is deleted rather than left to mislead.

---

## 1. Problem

Binary classification of short videos as hateful/harmful vs normal, from video frames + title +
transcript (audio optional). Multimodal and multilingual (English + Chinese). The dataset caption /
annotation rationale is **not** a model input.

**Paper type is fixed by user constraint: a METHOD paper that shows a real gain.** Benchmark
papers, audit papers, metric papers and evaluation-protocol papers are permanently out of scope,
even when the literature slot for them is empty.

---

## 2. Datasets — fixed, four, no additions

| dataset | train | val | test | test positives | language |
|---|---|---|---|---|---|
| HateMM | 744 | 107 | 215 | 86 | EN |
| MultiHateClip-EN (`MHC`) | 549 | 80 | 161 | 49 | EN |
| MultiHateClip-ZH (`MHC_zh`) | 579 | 78 | 149 | 45 | ZH |
| ImpliHateVid | 1283 | 325 | 401 | 200 | EN |

`HateClipSeg` exists on disk (395 rows) but has **no train/val split at all** and can never serve
as a second dataset for anything (`IDEA_REPORT` §8.4). No new dataset may be introduced.

**Consequence of the test sizes**: one flipped test item is worth roughly 0.5-0.6 macro-F1 points on
MHC-EN/ZH and 0.25 on ImpliHateVid. Seed std on a head is 0.004. Any claimed effect below ~0.01
is inside noise on the small three.

---

## 3. The contrast line (what a candidate has to beat)

Best **single-encoder bare head**, test macro-F1:

| dataset | contrast line | which encoder |
|---|---|---|
| HateMM | **0.8774** | LoRA-Qwen |
| MHC-EN | **0.7331** | frozen Qwen2.5-VL-7B |
| MHC-ZH | **0.7821** | LoRA-Qwen |
| ImpliHateVid | **0.9118** | CLIP ViT-L/14-336 |

The three-encoder ensemble of pairwise-trained heads (0.8732 / 0.7776 / 0.8183 / 0.9276) is
recorded as a **trick line, not a method** — it may be reported but cannot be the contribution.
Pairwise/AUC objective beats BCE on ROC in 4/4 cells; ensembling beats the val-best single encoder
by +1.3 to +5.3 macro-F1. Both are already banked and neither counts as novelty.

**Round-8 correction to both (2026-08-17, `idea-stage/R8_BLR_RESULT.md`, `R8_DECOMP_MEMO.md`).**
Neither is what it looked like. With sampling and the pointwise term held identical, the **pairwise
ranking term itself is worth +0.0003 / −0.0008 / +0.0011 / +0.0007 macro-F1 (all CIs contain zero)
and ≈0 ROC** — the whole banked effect is the equal-positive/negative sampling a pairwise objective
performs incidentally, which is worth **+0.0065 / −0.0113 / +0.0163 / +0.0106** and is a per-dataset
default, not a universal one. And **equal-weight cross-encoder averaging loses to the best single
encoder on 3 of 4 datasets** (−0.0068 / −0.0427 / −0.0191 / +0.0103); the recorded ensemble gain is
measured against the *val-selected* encoder and is largely encoder selection plus head estimation
variance. Same-encoder seed averaging is positive 4/4 (+0.0153 / +0.0016 / +0.0048 / +0.0033).

**User ruling on gain size**: "one step must gain 5 points" is *not* the kill line. A real,
stackable, honestly-measured gain is worth pursuing even if incremental.

---

## 4. Method substrate (what physically exists)

- Frozen encoders → precomputed per-video embeddings → small (~5M) MLP head (HateClipper-style
  element-wise "align" fusion) → BCE and/or retrieval-guided contrastive/pairwise loss →
  optional FAISS kNN retrieval at inference. **Only the head trains.**
- Feature cache layout: `data/CLIP_Embedding/<DS>/{train,dev_seen,test_seen}_<tag>.pt`, each a dict
  `{ids, img_feats [N,D], text_feats [N,D], labels [N]}`. D = 1024 (CLIP) / 3584 (Qwen).
- Encoder tags present for all four datasets: `openai_clip-vit-large-patch14-336_HF`,
  `Qwen2.5-VL-7B-Instruct_HF`, plus dataset-specific LoRA-Qwen variants, `Molmo2-8B_HF` (HateMM),
  `Qwen2.5-VL-32B-Instruct_HF`, and ~99 pooling/readout variants.
- Trainer: `src/run_rac.py`. **Measured cost: 11 seconds per head training run on HateMM**
  (12-run grid = 127 s wall). A 3-seed × 8-arm pilot is under 5 minutes of GPU. This is the single
  most important feasibility fact for round 6: head-level pilots are effectively free, so a pilot
  should be *rejected for weak premise, never for cost*.

### Positive assets a candidate may build on
- Three-encoder feature caches on all four datasets (above).
- Pairwise objective: 4/4 wins over BCE on ROC (banked).
- OCR cache: `data/OCR/{HateMM,MHC_test,MHC_zh_test,HateClipSeg}/ocr_windows_K30.jsonl`,
  plus `data/OCR/HateMM/test_ocr_window_vecs.npz` (2111×768).
- 1034 parsed six-field perceptual video descriptions (HateMM, ¥8.30 already spent):
  `idea-stage/desc_channel/descriptions_hatemm.jsonl`.
- 108 test errors coded item-by-item into S/O/M/A/D/X buckets: `idea-stage/r5_buckets.json`,
  `idea-stage/r5_error_dump.json`.
- MHC per-annotator votes: `data/gt/mhc_votes/mhc_{English,Chinese}_{train,valid,test}.tsv`.
- HateMM official hate time-spans: `data/gt/HateMM/hate_spans.json` (1083 entries).
- Raw videos on this machine for HateMM / MHC / MHC_zh (`~/data`); **ImpliHateVid raw video is
  gone** — every encoder-level or LoRA candidate is at most 3 of 4 datasets.
- RTX 5090, 32 GB, currently idle. QLoRA fits at ~15 GiB.
- MultiClimate is downloadable and usable as an out-of-domain capability test bed (inference only).
- Frozen-perception + small-head consumption topology has a positive external precedent
  (`2509.08024`, +3.1).

---

## 5. Budget and hard rules for round 6

- **API budget for the entire direction-finding stage: ≤ ¥60** (DashScope etc.). Spent so far: ¥0.
  Any pilot that calls a paid API must state a per-pilot cap ≤ ¥10 **inside its frozen design**
  before the first call. Key lives at `~/.dashscope_api_key`, never written to a file.
- **GPU**: local RTX 5090 only (check `nvidia-smi` first). ≤ 2 h per pilot, ≤ 8 h total.
- **Four red lines, never waived**: (1) zero test-label tuning; (2) decision rule frozen before
  results are seen; (3) blindness — no candidate metric computed during design/implementation;
  (4) the frozen run is submitted exactly once.
- **Ceremony scales with re-run cost** (user ruling 2026-08-05). CPU-level / ≤1 h experiments get at
  most one review round, and only defects that would produce a wrong verdict or touch the test set
  may block. Documentation self-consistency is never a blocker.
- **Test protocol** (user ruling 2026-08-09): test *inputs* are unsealed — transductive / TTA-family
  methods are legal. Test *labels* remain sealed; nothing tuned on them may be reported as held-out.
- **Novelty standard** (user ruling): occupying a slot someone else has touched is not an automatic
  kill. Doing it demonstrably better, with a real mechanism insight, is legitimate novelty.
- **Reporting language**: standard technical vocabulary. No metaphors, no coined jargon, no
  nicknames. State what was run, what number came out, and what it means for the decision.

---

## 6. Constraint map — what is already dead

Cumulative through round 5: **65 candidates generated, 0 live mechanism candidates**
(`idea-stage/IDEA_REPORT.md` §9.11). The 2026-08-13/14 series added 9 more kills. A round-6
candidate that is isomorphic to anything below is rejected before scoring.

### 6.1 Three framing killers (any mechanism must route around these)
1. **SAGE** (ACL 2026, HateMM 0.8710/0.8628) — the HateMM accuracy race is closed; a pure accuracy
   claim is not publishable on its own.
2. **HCG-MPB** (ICMR 2026) — argues in its motivation that instance-based retrieval is a flawed
   design; every RGCL-family hateful-video paper must now rebut it.
3. **`2607.23304` + `2602.05152`** — under squared loss + linear head + fixed features, explicit
   parameter adaptation and implicit routing are the same kernel ridge regression, and query
   expansion ≡ key expansion. "Our retrieval module is a form of test-time adaptation" and "we
   improved the query/key construction" are both formally absorbed and cannot be claims.

### 6.2 Retrieval structure — closed
Segment-level retrieval keys; multi-segment complementarity; single-segment selection; visual-purity
segment selection; type-hard-partitioned memory; streaming/continual memory; cross-lingual
EN-rescues-ZH; CVoI acquisition; OCR−ASR residual keys; near-duplicate/label-conflict memory
(conflicting pairs 5 vs bar 10, observed 5 vs permutation-expected 24.1 — near-duplicates are label
*concordant*); late-interaction segment retrieval (−0.043 macro-F1). Human-Agreement Retrieval is
**permanently closed** on three legs (vote-feature arm −0.0174 EN / −0.0105 ZH vs a *trained*
baseline; contrastive leg −0.00506 with the shuffled-vote placebo capturing 87 % of the gain).
XBUCKET adds: for the residual error set, retrieval/kNN repair is foreclosed — top-20 gold purity
0.255-0.517 for errors, and non-X errors sit *closer* to train than X errors in 9/9 encoder cells.

### 6.3 Temporal / segment / pooling — axis closed
`refine-logs/LITSWEEP5_TEMPORAL.md` closes the axis at four independent levels: order kernels
(soft-DTW Δacc +0.0059 = exactly the shuffle null p95), retrieval over frame groups (+0.0035 vs bar
+0.05), causal-prefix conditional information (**exactly 0.0000**), segment granularity (+0.0012 /
+0.0032, with 91-98 % of the oracle inside banned per-item selection), frame count (8→16 = −0.0077).
`EUM_FORENSIC_RECON` kills the dilution premise outright: HateMM hate-span coverage **median
0.8289**, 74 % single contiguous span, only 22 % below 0.5 coverage. The hateful evidence is not
sparse in time. Sub-video units are also not a different object (unit↔own-video pooled cosine
0.95).

### 6.4 Audio / prosody — 0 for 4, family closed
eGeMAPS, LAUD, CLAP, and the C8 prosody-as-operator estimand (Δ −0.0436 / −0.0392 vs bar +0.010,
0/3 seeds both arms). Failure mode is **redundancy, not weakness**: label-permuted prosody adds
*more* to a text head (+0.0294/+0.0448) than real prosody (+0.0031/+0.0122). Closes FiLM/gating/
bilinear audio conditioning and any "audio modulates text" successor.

### 6.5 OCR / on-screen text — one-dataset at best
Provenance typing: −0.0020, 0/3 seeds, label-permuted null = 90 % of the real effect. Mean fusion:
+0.0094 (bar +0.015). Same vector through the learned fusion MLP: **−0.0246**, 3/3 seeds, sign flip.
Proposition-mass firewall: `rho_obs` −0.0345 vs bar 0.24, negative in 5/5 seeds. Measured
evidentiary fact: OCR is complementary on HateMM and **redundant on MultiHateClip** — 95 % EN / 99 %
ZH of MHC videos carry ≥20 chars of screen text, and not one MHC error is decidable from screen text
the transcript lacks. External occupant: MM-HSD (ACM MM 2025) macro-F1 **0.874** on HateMM with
PaddleOCR-as-CMA-query.

### 6.6 Annotation disagreement / votes — closed
Vote-distribution retrieval, dissent prototypes, contested-item abstention, annotation-escalation
prediction, vote-fraction soft targets. MHC has only 2.2 annotators; the flagship multi-annotator
architecture ranks last on a 6-annotator corpus. `Counter Narrative` votes: 139 videos, **never a
majority, never even a tie**, and only **1 of 15** S-bucket false positives carries one (bar 25 %);
error rate on CN-voted videos is *lower*, not higher.

### 6.7 MLLM in front of the head — all five access points measured negative (2026-08-13/14)
| access point | result |
|---|---|
| new input stream (768-d description as 3rd stream) | −0.0371, 0/3 seeds |
| text merge (description into transcript pre-encoder) | −0.0105, 0/3 seeds |
| pre-classifier stance labelling | see 6.8 |
| decision arbitration / uncertainty-gated deferral | −0.0135, 0/3 seeds; the judge is **less accurate than the head in 21 of 24 in-band cells and better in 0** |
| training-data augmentation (counterfactual de-hating) | −0.0507, 0/3 seeds, and **0.0444 worse than a random-negative control** |

Scope note the code review attached to the last row: the CAD construction was defective (25 of 284
rewrites still matched the slur list, 24 were identical), so the verdict is "*this* construction is
worthless", not "counterfactual augmentation is disproven for video".

### 6.8 Zero-supervision stance extraction — six routes, all negative
Direct 5-way prompt 0.257 → masked 0.371 → symmetric 2-way **0.469 (chance 0.50, p = 0.86)**;
perception questionnaire gate-0 **2/18 = 0.111** vs bar 0.30 (content described accurately 15/18,
but a direction-bearing fact present only 2/18); CN votes (above); synthetic attribution pairs —
0.98-1.00 synthetic dev, **AUC 0.441/0.467 on real ASR**, sign inverted in 6/6 cells, and the
mechanism is decisive: **only 10 of 99 real transcripts contain any attribution marker at all**, so
the source cue is not present in speech; likelihood read-out — all four arms emit a **constant
OPPOSE**, base vs instruct margin correlation **r = 0.980**, template moves the margin 3.8× more
than the video. The RLHF/alignment explanation is refuted.

Standing counter-fact: matched violation/exemption policy clauses embed at cosine **0.920** (CLIP),
and clause directions **lose to random directions by −0.046 mean ROC on 3 of 4 datasets** — stance
is provably not linearly present in the frozen features.

### 6.9 Training-level families (`IDEA_REPORT` §9.6)
F1 rationale-then-verdict SFT — **occupied** (IARE, LEAF, ExPO-HM), and naive explain-then-detect
*loses* (Direct-SFT 75.0 > CoT-SFT 74.5 > GRPO 74.5 on Qwen2.5-VL-7B).
F2 generative-MLLM-as-classifier — adjacent + nulled in-house; small-n is a loss.
F4 votes-as-target — dropped. F5a/b OCR integration — occupied.
**F3 stance-as-supervision is the only OPEN family, and its supervision does not exist.**
F5c text-bearing frame selection — open, but relevance sampling still misses >90 % of harmful
content (`2508.10974`).

### 6.10 Other standing negatives a candidate must not re-derive
- **Cross-encoder complementarity is additive.** A monotone non-additive lattice over per-encoder
  OOF logits extracts nothing: ΔROC mean −0.0000, bootstrap LCB95 −0.00253, with a positive control
  that recovered a planted interaction.
- **Decision-rule / calibration mechanisms are capped** at **+0.25 to +1.2 points** (round 8, D3:
  a global threshold oracle fitted on the 629-1608-item train+val pool). The older figure of
  +1.2 to +4.6 was measured with an oracle fitted on the 149-215-item **test** splits and is roughly
  4x inflated by that oracle overfitting a small evaluation set. No realistic rule reaches even the
  corrected cap: a dev-fitted threshold is negative on 3 of 4 datasets and prior-matching ranges
  −0.0002 to +0.0104. (`idea-stage/R8_DECOMP_MEMO.md` §3.)
- **No measurable train/test covariate shift on any of the four datasets** (domain-classifier AUC
  0.42-0.56, MMD p 0.17-0.96). Any distribution-alignment / shift-correction TTA premise is dead on
  arrival.
- **Random projections are a strong baseline** (val ROC up to 0.88); any "K interpretable
  directions" claim must beat random directions averaged over draws.
- **A large oracle ceiling is not evidence for a candidate** — it is the precondition every failed
  candidate already met. AGGNET carried the largest oracle the project ever measured (+0.149/
  +0.152/+0.219) and delivered +0.013/−0.007/+0.000. Gate on demonstrated *conversion*, in net
  items.
- **Law III / F47: per-item selection is banned** — any operator that chooses, per test item, which
  member/segment/encoder to believe is foreclosed.
- **Within-hard-label permutation nulls are invalid** (deviation D1): they manufacture the
  conditionally-independent ideal case and would force KILL for every possible input.

---

## 7. Where the remaining headroom actually is

Annotation noise is **not** the binding constraint: ≥ +7 macro-F1 of purchasable headroom on every
dataset (MHC-EN +15.0, MHC-ZH +12.0 to the panel-resample ceiling; HateMM +12.7, ImpliHateVid +7.2).

Error taxonomy over all 108 test errors of the round-4 comparator:

| bucket | n | share | oracle-fix value (mean macro-F1) | status |
|---|---|---|---|---|
| **S** stance / use-vs-mention | 49 | 45.4 % | **+6.46** | open only via paid supervision |
| **O** decisive evidence in screen text | 5 | 4.6 % | (in +7.91 with S+M) | HateMM-only |
| **M** empty / music-only transcript | 5 | 4.6 % | " | tiny |
| **A** annotators split / label conflict | 9 | 8.3 % | — | not purchasable |
| **D** duplicate / degenerate item | 3 | 2.8 % | — | not purchasable |
| **X** ordinary ranking error | 37 | 34.3 % | — | **SEALED** — diffuse, deterministic, 28/37 wrong under all 3 seeds, no attackable structure on 5 hypotheses |

Purchasable mass after removing X: **+0.0635 / +0.1199 / +0.1031 / +0.0300**.

The S bucket is the prize and the only open novelty family, and every zero-cost route into it has
now been measured and failed. The standing jury ruling is: **fund ~750 human stance judgements
(375 items × 2 blind annotators + adjudication, α ≥ 0.80, machine-vs-human macro-F1 ≥ 0.80) or
close the stance direction.** Model-checking-model does not satisfy the trigger. The LDC BeSt
corpus (`LDC2023T13`, label `ROB` = "writer reports another source's belief without revealing her
own", 10,777 EN instances) is the one off-the-shelf supervision source that matches the
distinction; it is paid and the user has not yet checked institutional membership.

---

## 8. What round 6 should therefore look for

A mechanism that (a) is not isomorphic to §6, (b) can be piloted on cached features in minutes,
(c) has a premise that is *checkable before the pilot* against the measured facts in §6-§7, and
(d) can beat the §3 contrast line on **more than one dataset**. Failing that, the honest output of
round 6 is a precise statement of what is left and what it would cost.

Explicitly recorded as still-open by the project's own documents:
1. F3 stance-as-supervision (blocked on funding).
2. F5c text-bearing frame selection for moderation.
3. B-SRTD named-intervention response-tensor distillation — never killed by a mechanism failure,
   an occupant or a null; blocked on a bounded balanced data build.
4. EAPD accountability-path distillation — blocked on a 330-video annotation build.
5. Transductive / TTA in the **binary moderation** regime — every sub-branch has a top-venue owner,
   but all assume many-class, text-anchored, balanced pools; the counter-literature's demand
   ("prove your adaptation cannot damage the un-adapted model") is currently unstated in this
   regime. Note the two live headwinds: killer 3 in §6.1 absorbs the retrieval≡TTA claim, and §6.10
   shows there is no covariate shift to correct.
6. A trained probe on **intermediate hidden states** of the MLLM — explicitly not ruled out by the
   likelihood probe, which measured only the output distribution.
7. The `Metadata` (title/description) channel — the largest human-annotated contributing modality
   in ZH (294) and second in EN (197), and **no model in this project consumes it**. Recorded as an
   observation, not a candidate: "adding a modality is not a mechanism".

---

## 9. Non-goals

No new datasets. No benchmark/audit/metric/evaluation-protocol paper. No caption as model input.
No claim built on the ensemble or on the pairwise objective. No re-skin of anything in §6.
