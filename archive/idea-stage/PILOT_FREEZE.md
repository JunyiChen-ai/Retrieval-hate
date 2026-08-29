# Pilot pre-specification — FROZEN before any pilot number was computed

Written 2026-08-08, after the cross-model triage returned its ranking and threshold corrections,
and **before** any pilot script was executed. These are exploratory probes, not a registered
verdict; but the decision rules below are frozen and are not to be edited after results appear.

Common protocol for all three pilots:
- **HateMM-train only** (744 videos, 298 hateful / 446 not). `dev_seen` (val) and `test` are never
  opened. No cross-dataset data.
- Folds: the frozen seed-20260807 5-fold split at
  `artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/folds/fold_{0..4}/{train_ids,query_ids}.json`.
  Every retrieval memory is built from that fold's `train_ids` only.
- Head, wherever a head is trained: `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=2000,
  solver='lbfgs')` on L2-normalized per-modality blocks. Decision threshold chosen on the fold's
  **training** predictions to maximize macro-F1, then applied to that fold's query videos.
  This head is deliberately independent of the Gate-0 arm hyperparameters.
- Bootstrap: 2000 video-level resamples, seed 20260808.
- Metric: macro-F1 over the 744 out-of-fold predictions unless stated otherwise.

## Disclosure

While retrieving the A0 head configuration I opened
`folds/fold_0/selected_hparams.json` and thereby saw fold-0 **inner-OOF hyperparameter-selection**
macro-F1 values for arms A0-A4. Those numbers are adjacent to the sealed Gate-A family. They are
**not used** anywhere in these pilots, in the report, or in any decision; the pilots define their
own views and their own head and measure their own numbers. The pilot designs below were frozen in
`idea-stage/codex_triage_bundle.md` and in the triage reply *before* that file was opened.

---

## P1 — Trim-gain decomposition (idea I1), corrected per reviewer

**Question.** Is the published claim that trimming hateful videos to gold spans improves
classification a *dilution* effect (temporal headroom) or a *label-alignment* artifact (trimming
is only applied to positives, so it leaks the oracle)?

**Three views.** Each view produces a 1792-d vector = [window-mean segment visual (1024) ||
whole-video text (768)].
- **A "full"** — mean over all 30 segments.
- **B "random-window"** — a contiguous window of m segments, m drawn per video from the empirical
  distribution of gold-span segment counts over hateful train videos, window position uniform.
  Applied to **both** classes. RNG seed 20260808.
- **C "gold-aligned"** — hateful videos: mean over segments overlapping any gold span.
  Non-hateful videos: the **same** window as view B (they have no gold spans), so the two views
  differ only in how the positives' windows are chosen.

**Reported quantities.**
- generic-trim term = macroF1(B) − macroF1(A)
- oracle-alignment term = macroF1(C) − macroF1(B)

**Decision (frozen).**
- **GO** if oracle-alignment ≥ **+1.5** macro-F1 points AND the paired-bootstrap 95% lower bound
  is > 0 → dilution/temporal headroom is real, the G-A family is revived.
- **GO-AS-NEGATIVE** if the paired-bootstrap 95% **upper** bound of the oracle-alignment term is
  ≤ **+0.5** points → gold-span trimming carries no headroom under frozen features; the
  decomposition itself plus the invariance regularizer becomes the contribution, and purely
  temporal branches are deprioritized.
- **AMBIGUOUS** otherwise → the idea is dropped from the top ranking.

## P2 — Segment-keyed retrieval purity (idea I4)

**Question.** Does a segment chosen purely by the *label purity of the neighbourhood its key
retrieves* land on hateful evidence, and does it carry classification signal?

**Selector (label-free at query time).** For each query video, for each of its 30 segments, take
the top-20 cosine neighbours among the fold-train segments (same-parent excluded); p_j = fraction
of those 20 neighbours whose parent video is labelled hateful; j* = argmax_j p_j.

**Metric 1 (selection quality).** Over the 298 hateful train videos evaluated as queries in their
own fold: hit rate = fraction whose j* segment overlaps a gold span. Chance = the per-video
expected hit rate = (number of gold-overlapping segments)/30, averaged over the same videos.
- **GO** if hit ≥ 2× chance AND ≥ **0.35** absolute AND bootstrap 95% lower bound > chance.
- **NO-GO** if hit < 1.3× chance.

**Metric 2 (classification).** OOF macro-F1 of a head on [whole-video visual || text || segment-j*
visual] minus a head on [whole-video visual || text].
- **GO** if ≥ **+0.015**; **NO-GO** if < **+0.005**.

Both metrics must pass for the idea to keep its rank.

## P3 — Typed-evidence routability probe (idea I8)

**Question.** Is "this video's decision requires on-screen text" predictable from frozen CLIP
features, and does hard-partitioning the retrieval memory by that predicted type improve
neighbourhood purity? If not, no MLLM annotation budget is ever spent.

**Part (a) — probe.** Label = `on_screen_text ∈ required_modalities` from the adjudicated Gate-C
audit (133 videos). Features = [whole-video visual || whole-video text || max-pool over the 30
segment visual vectors] (2816-d). Cross-validation groups = the video's frozen outer fold.
- **GO** if OOF AUROC ≥ **0.68** with bootstrap 95% lower bound > **0.55**;
  **NO-GO** if AUROC ≤ 0.60.

**Part (b) — typed routing.** Apply the OOF probe to all 744 train videos to obtain a predicted
type bit for every memory entry. For the 133 audited videos as queries, compare top-20 neighbour
**label purity** (fraction of neighbours whose label equals the query's true label) under
(i) unrestricted memory and (ii) memory restricted to entries with the same predicted type bit.
- **GO** if typed purity exceeds unrestricted purity by ≥ **0.05** in at least **4 of 5** folds.

Both parts must pass for the Claude annotation budget to be authorized.
