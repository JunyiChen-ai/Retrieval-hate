---
type: experiment
node_id: exp:exp-tarc-t0
title: "TARC (target-aware retrieval-contrastive): pre-registration + cheap GT-target oracle probe (B-line)"
idea_id: ""
verdict: pre-registered
confidence: n/a
date: "2026-07-13"
status: DESIGN ONLY — no runs, no SLURM, no commit. HARD CONSTRAINT: the method proper uses NO gold annotations (target OR time-span); gold labels are used ONLY to probe the ceiling (oracle) or to score MLLM predictions. See §5.
hardware: "planned: 1x A100 (SLURM), frozen CLIP features cached -> ~21-25 s/run (enc3seed precedent)"
provenance: "data facts verified live 2026-07-13 against data/gt/HateMM/HateMM_annotation.csv, data/gt/HateMM/{train,val,test}.jsonl, data/gt/MHC{,_zh}/{train,val,test}.jsonl; code change points cite src/ file:line read live; cost/protocol precedent = exp-encoder-3seed.md (job 12850)"
added: 2026-07-13T00:00:00Z
tags: ["hateful-video", "MLLM", "target-aware", "retrieval-contrastive", "pre-registered", "oracle-probe", "HateMM", "MHC", "B-line", "TARC"]
---

# TARC — Target-Aware Retrieval-Contrastive (B-line pre-registration)

**One-line mechanism.** Use the *attacked-community / content-community* variable
`target` (a categorical label, e.g. Blacks/Jews/LGBTQ/…) to **reshape the RGCL
retrieval graph itself** — which examples become mined hard-negatives, how kNN votes
are weighted, and how the contrastive space is regularised — rather than injecting any
MLLM output as a feature, score, prior, or field.

**Gold-annotation isolation (hard user constraint, 2026-07-13).** The *method proper* uses
**no gold annotation of any kind** — not `target`, not the `hate_snippet` time-spans. Gold
`target` (HateMM only) is admitted **only** on two isolated, non-main-table paths: a cheap
**oracle ceiling probe** (**G1**) that gates the whole line *before* any MLLM inference, and
**scoring** the MLLM's target prediction (**G2**). The **final method (G3) reads
MLLM-predicted target on every dataset, including HateMM** — GT target never enters
training or inference. The MLLM's irreplaceable role: on MHC EN/ZH `target` has **no** GT
at all (§4), and on HateMM the method still refuses the GT and consumes the MLLM's
prediction, so the MLLM is load-bearing on every dataset. Full per-gate ledger + code-level
enforcement in **§5**.

> Status: **design only.** This file pre-registers hypotheses, mechanisms, gates, kill
> numbers, seeds, and protocol. No experiment is run and nothing is committed here.

---

## 1. Hypothesis

**H (exploitability of target structure).** In hateful-video detection the hardest
discriminations are *same-community, opposite-label* pairs — a benign video *about* a
community vs. a hateful video *attacking* the same community (e.g. a documentary about
Black history vs. an anti-Black hate clip). The frozen-CLIP+transcript fused embedding
is dominated by *topic/community* signal, so the standard label-only RGCL objective
mines and votes over neighbours that are largely community-matched *by accident*.
Making the `target` community an **explicit conditioning variable** on the
mining / voting / regularisation structure should (a) sharpen the within-community
benign-vs-hateful boundary and (b) suppress cross-community false neighbours, yielding
**≥ +0.03 accuracy** over the strongest non-MLLM baseline (frozen-CLIP RGCL) on at
least one dataset under a 3-seed paired test, both protocols.

**H0 (the null this line most likely dies on).** The fused CLIP space *already*
encodes community topic so strongly that the label-only nearest opposite-label neighbour
is *already* community-matched in practice; conditioning on `target` is therefore close
to a no-op and the oracle probe (**G1**, GT target, zero MLLM cost) lands within seed
noise (**< +0.015 acc**). See §8.

---

## 2. Mechanism variants (with exact code change points)

All three variants add a single global, id-keyed **`target_pack`** and thread it through
the *same* plumbing already used by `aux_pack` (P4), `cf_pack` (P5), and `segment_cache`
(segment-RGCL): built in `main()` from `train_set.ids`, passed through `model_pass` into
`compute_loss` / the evaluator, and **exactly no-op when absent or when its
weight/flag is off**. `target_pack` = `{ "id_to_target": {video_id -> int target-code},
"num_targets": K, "row_target": LongTensor aligned to train rows }`. Target codes come
from the normalised label set (§4): `{Blacks, Jews, Whites, Others, LGBTQ, Muslims,
Sexits, Asian}` (single-label = primary/first-listed target; multi-target handling in
§4). **Target source is a hard-gated config field** `--tarc_target_source
{mllm_pred, gt_oracle}`: main-table runs (G3) may use **only** `mllm_pred`; `gt_oracle`
is refused unless `--oracle_probe True` is also set and stamps `ORACLE_CEILING` into the
run name (§5). No variant initialises prototypes, centroids, or codes from GT — all three
read `target_pack` identically whatever its source, so swapping GT→MLLM-pred is a
one-flag change with no hidden GT dependency.

### V1 — target-matched hard-negative mining  *(retrieval structure, TRAIN-time)*

**What changes.** The RGCL triplet mines, per anchor, the *nearest opposite-binary-label*
train embedding as its hard negative (`src/utils/retrieval.py:446`, condition
`train_labels[I[i,iter]] != query_labels[i]`). V1 adds a **target-match preference**:
among opposite-label candidates, prefer those that **share the anchor's `target`
community**, i.e. mine the *benign-about-T vs. hateful-about-T* confusable pair. A knob
`--tarc_hn_mode {off, prefer, require}` controls it (`off` = bit-identical baseline;
`prefer` = take same-target opposite-label if present in the top-`k*multiple`, else fall
back to current behaviour; `require` = same-target opposite-label only, else zero-fill
as the code already does for "not enough hard negatives").

**Where.**
- Candidate-selection loop: `src/utils/retrieval.py:433-478`
  (`dense_retrieve_hard_negatives_pseudo_positive`). Add a target check alongside the
  existing label check.
- The index is rebuilt by iterating `train_dl` (`src/utils/retrieval.py:330-362`), which
  is **shuffled**, and currently collects only `train_feats`/`train_labels`. V1 must also
  accumulate `batch["ids"]` in that loop and map to `target_pack["id_to_target"]` so
  `train_targets` is **row-aligned to `train_feats`**. Query-side targets come from
  `batch["ids"]` of the current batch. (This is the one non-trivial edit; ~15 lines.)
- Pseudo-gold-positive (same-label) selection at `src/utils/retrieval.py:463` can
  optionally be target-matched too (`--tarc_pp_mode`), but the pre-registered probe holds
  it `off` to isolate the hard-negative effect.

**MLLM dependency:** train-side targets only → on a target-less dataset, MLLM predicts
`target` for **train** videos; inference vote is unchanged.

### V2 — target-consistency-weighted kNN vote  *(retrieval structure, TEST-time)*

**What changes.** The final decision is a similarity/rank-weighted kNN vote over
retrieved neighbours' binary labels (`src/utils/metrics.py:214-312`,
`compute_metrics_retrieval`; the `use_sim` branch `:262-278` already multiplies each
neighbour's ±1 vote by its similarity). V2 adds a **target-consistency multiplier**: a
neighbour whose `target` matches the query's `target` gets weight `w_hi`, otherwise
`w_lo` (`--tarc_vote_gamma`, with `gamma=0` = identity = baseline). Rationale: a
"Blacks"-topic query should be decided mostly by other "Blacks"-topic memory entries,
not by cross-community neighbours that are near only through generic toxicity cues.

**Where.**
- `src/model/evaluate_rac.py:429-468` (`retrieve_evaluate_RAC_`): the per-query
  `logging_dict` already carries `retrieved_ids` and `retrieved_label`; add
  `retrieved_target` (look up `target[retrieved_id]`) and `query_target`.
- `src/utils/metrics.py:262-278` (new `use_target` branch): fold the target-match
  multiplier into the existing `retrieved_labels_map * retrieved_sims` weighting.

**MLLM dependency:** needs the **query** (test) target too → heavier. At G1 the query
target is GT (HateMM test annotation, oracle ceiling); at G3 it must be MLLM-predicted
per test video.

### V3 — intra-target separation regulariser  *(contrastive structure, TRAIN-time)*

**What changes.** Add one additive loss term (same slot/pattern as `lambda_seg` /
`lambda_aux`, `src/model/loss.py:556-573`) that, **within each target community present
in the batch**, pushes the benign centroid and the hateful centroid apart by a margin:
`L_tarc = mean_T relu(m - d(mu_{T,hate}, mu_{T,benign}))`, where `mu_{T,·}` are the
batch's per-target per-label mean fused embeddings and `d` is the run's metric
(`_pair_similarity`, `src/model/loss.py:654-663`). Gated by `--lambda_tarc` (`0.0` =
exact no-op; the term only reads the grad-tracked `feats` already produced by the forward
pass, so `lambda_tarc=0` is byte-identical to baseline, exactly as P4's `lambda_aux`
guarantee `src/model/loss.py:565-573`). This makes "same community, opposite intent"
explicit geometry rather than leaving it to accident.

**Where.**
- New `compute_target_loss(feats, batch_ids, target_pack, labels, args)` in
  `src/model/loss.py` (mirrors `compute_aux_loss` `:578-608`).
- Call site: `src/model/loss.py:573`, immediately after the `lambda_aux` block:
  `if lambda_tarc > 0 and target_pack is not None: total_loss += lambda_tarc * compute_target_loss(...)`.
- `target_pack` add to `compute_loss` signature (`src/model/loss.py:12-23`) and the
  `model_pass` call (`src/run_rac.py:548-565`), plus `model_pass` signature
  (`src/run_rac.py:487-501`) and build in `main()` (precedent: aux_pack
  `src/run_rac.py:1054-1082`, using `train_set.ids` `:1062`).

**MLLM dependency:** train-side targets only (like V1).

**Pre-registered variant order.** V1 first (cheapest, cleanest MLLM story, train-only),
then V3, then V2. G1 tests all three under GT target; a variant is carried to G2/G3 only
if it clears G1.

---

## 3. Mechanism differentiation vs. the 13 closed routes (hard constraint #1)

The 13 closed routes all treat an MLLM output as an **additive signal consumed by an
otherwise-unchanged pipeline** (a feature, a score, a field, a prior, a twin). TARC
instead uses a **categorical `target` variable to change the retrieval/contrastive
*graph structure*** — which pairs are contrasted, which neighbours vote, how the space is
partitioned. The `target` category is never a model input feature, never a per-sample
regression/classification training target, and never rescales a logit.

| Closed route | Its mechanism | Why TARC is distinct |
|---|---|---|
| **P1** prior recalibration | MLLM prior shifts the decision threshold/prior | TARC never touches the decision prior/threshold or logit scale; it changes *which examples* are mined/voted. |
| **P2/P2b/P2c** neighbour rerank | MLLM *comparability score* re-ranks the retrieved neighbour list post-hoc | V2 reweights by a **discrete target-category match**, not a learned MLLM score; and V1/V3 act at **train time** on mining/loss — P2 never modifies training. |
| **P3** evidence-density pooling | MLLM per-segment hate density → weighted temporal pooling of frames | TARC has no pooling and no temporal weighting; it operates on a **video-level community category**. |
| **P4** schema-distill | MLLM structured fields (incl. `target_group`) distilled as **auxiliary prediction heads** (predict-the-field from the fused embed) | V3 does **not predict** `target` as an output. `target` is a **grouping variable** that defines contrastive centroids / pair selection. P4 = "add a prediction task" (redundant); TARC = "restructure the metric-learning graph". Distinct locus even though both can read a `target_group` field. |
| **P5** counterfactual twins | Sanitise text with MLLM, push anchor from its twin | TARC generates no counterfactuals and adds no twin negative; the negative in V1 is a **real, mined, same-community opposite-label train video**. |
| **P11** MLLM-scores-as-training-signal | MLLM per-segment scores as pseudo-labels/regression targets for training | TARC uses a **discrete category**, never a score, and uses it to **condition pair selection**, never as a per-sample training target. |

**Through-line to state in the paper:** every earlier route asked "what extra number can
the MLLM hand the classifier?"; TARC asks "what *structure* does the MLLM let us impose on
the retrieval memory that we could not impose without it?" — and on MHC that structure
(community identity) is *unobtainable* without the MLLM (§4).

---

## 4. Data fact-check (verified live 2026-07-13)

### HateMM — `target` column exists, is dirty, and covers BOTH classes

`data/gt/HateMM/HateMM_annotation.csv`: header
`video_file_name,label,hate_snippet,target` (col 4 = `target`); 1084 lines = 1 header +
**1083 data rows** = **431 Hate + 652 Non Hate**.

**Format is dirty (must normalise):** `target` mixes plain strings and Python-list
literals: among the 431 hate rows, **70** are list literals (e.g. `"['Blacks', 'Jews']"`),
the rest are plain (`Blacks`, `Blacks,Jews`), and **1** is empty. Normalisation:
`str.startswith('[')` → `ast.literal_eval`; else split on `,`; strip; dedupe.

**Hate-video per-target presence (video counted once per distinct target), verified:**

| target | Blacks | Jews | Whites | Others | LGBTQ | Muslims | Sexits | Asian |
|---|---|---|---|---|---|---|---|---|
| # hate videos | **329** | **90** | **18** | **12** | **12** | **10** | **5** | **1** |

(Matches the counts in the task brief exactly.) **Multi-target reconciliation:** I count
**42** multi-target *hate* videos (raw comma/list-len>1 and normalised agree). The brief's
"55 multi-target" = **all** videos: 42 hate + **13** non-hate multi-target (e.g.
`('Blacks','Whites')`×6, `('Blacks','Others')`×4, `('Asian','Blacks')`×1, …) = 55. So "55"
is dataset-wide, "42" is hate-only — both correct, different scopes.

**Primary (first-listed) target, hate videos:** Blacks 321, Jews 67, Whites 15, Others 10,
Muslims 7, Sexits 5, LGBTQ 4, Asian 1.

**Non-hate videos ALSO carry `target`** (this reshapes the mechanism): Others 389, **Blacks
150**, **Jews 39**, Whites 8, Muslims 7, plus 42 empty and 13 multi. So `target` is a
**content-community** label available for *both* classes — it is *not* a
hate-only "victim" tag. This is exactly what V1/V2/V3 need: they can pair a hateful and a
benign video *about the same community* (the confusable case), which is impossible if the
tag existed only on hate videos.

**Severe skew (the mechanism's structural risk):** Blacks = 329/430 = **76.5%** of hate
videos with a target; Blacks+Jews = 419/430 = **97.4%**. Effectively ≈ 2 informative target
classes with a long thin tail (Asian n=1). Low target entropy ⇒ small headroom (§8).

**ID join (verified):** jsonl `id` (e.g. `hate_video_98`) = CSV `video_file_name` **stem**
(strip `.mp4`); **744/744** train ids match by stem, 0/744 by full name. So `target_pack`
is built by stem-join. HateMM `{train,val,test}.jsonl` schema = `{id, text, label}` only
(744 / 107 / 215 rows) — target is **not** in the jsonl, it must be joined from the CSV.

### MHC EN and ZH — NO target field anywhere (⇒ MLLM is the ONLY source)

Verified across every split:

| file | n | keys |
|---|---|---|
| `data/gt/MHC/train.jsonl` | 549 | `{id, text, label}` |
| `data/gt/MHC/val.jsonl` | 80 | `{id, text, label}` |
| `data/gt/MHC/test.jsonl` | 161 | `{id, text, label}` |
| `data/gt/MHC_zh/train.jsonl` | 579 | `{id, text, label}` |
| `data/gt/MHC_zh/val.jsonl` | 78 | `{id, text, label}` |
| `data/gt/MHC_zh/test.jsonl` | 149 | `{id, text, label}` |

No `target` / `group` / `community` / `victim` field in any MHC EN or ZH annotation
(grep over `data/gt/` returns only HateMM + the RGCL split files). **Consequence:** on MHC,
`target` **can only** come from an MLLM prediction — there is no GT to fall back on. This
is the concrete source of MLLM irreplaceability for the B-line: HateMM is the oracle
sandbox (GT target ⇒ free ceiling test), MHC is where a positive result would *require*
the MLLM. (The MultiHateClip source paper is catalogued at
`research-wiki/papers/wang2024_multihateclip_multilingual_benchmark.md`; its release does
not ship a per-video target-community field in our `data/gt`.)

---

## 5. Gold-annotation isolation declaration (method uses NO gold labels)

**Rule (user hard constraint, 2026-07-13).** Anything whose output can feed a number in the
paper's main results MUST NOT consume any gold annotation — not `target`, not the
`hate_snippet` time-spans (`data/gt/HateMM/hate_spans.json` / the `hate_snippet` CSV column),
not any other GT field. Gold labels are admissible ONLY to (i) probe the method's **ceiling**
(oracle) or (ii) **score** an MLLM prediction. TARC touches **no** time-span annotation
anywhere — it is a video-level method and never reads `hate_snippet`. The only gold field it
could touch is `target`, and it does so only on the two isolated paths below.

**Per-gate GT-contact ledger:**

| gate | reads GT `target`? | how it is used | enters main table? |
|---|---|---|---|
| G0 normalisation | yes | builds `target_map.json` (a data-prep cache); consumed at main-table scale by NOTHING except the oracle probe | no |
| G1 oracle probe | yes (in the RGCL pipeline) | **ceiling estimate only**, stamped `ORACLE_CEILING`; establishes headroom | **NO — never tabulated as a result** |
| G2 prediction quality | yes (as an eval reference) | scores MLLM-pred vs GT (macro-F1); the RGCL runs inside G2 consume MLLM-pred target, never GT | no |
| G3 final method | **NO** | MLLM-predicted target on **every** dataset incl. HateMM; GT read by neither training nor inference | **YES — the only main-table numbers** |

**Code-level enforcement (auditable):**
- One config field `--tarc_target_source {mllm_pred, gt_oracle}` selects the source and is
  threaded into the run/output name and trainlog header, so every log self-records provenance.
- `mllm_pred` (loads `data/gt/<ds>/target_pred_<mllm>.json`) is the **default and the only
  value a main-table run may use**.
- `gt_oracle` (loads GT `target_map.json`) is accepted **only** together with an explicit
  `--oracle_probe True`; `run_rac.py` asserts `not (target_source=="gt_oracle" and not
  oracle_probe)` and stamps `ORACLE_CEILING` into the run name + trainlog, so an oracle number
  can never be silently reported as a result.
- The MLLM-pred JSON is produced by a **separate offline** step (G2); the training pipeline
  therefore has *no* code path from GT `target` into a `mllm_pred` run.
- HateMM's GT `target` / `hate_snippet` stay in `data/gt/HateMM/` but are read at exactly two
  call sites — the `target_map.json` builder (G0) and the G2 scorer — never by a G3 method run.
- **No implicit GT dependency in any variant:** V1 mining, V2 vote, and V3 centroids all read
  `target_pack` identically regardless of source; none uses GT for prototype init, code
  assignment, or normalisation. GT→MLLM-pred is a single-flag swap. (Audit: grep a G3
  trainlog for `target_source=mllm_pred` and the *absence* of `ORACLE_CEILING`; grep the code
  for GT-`target` reads → only the two allowed sites.)

---

## 6. Gate sequence (kill numbers + cost)

**Strongest non-MLLM baseline (control for all deltas):** frozen-CLIP RGCL on HateMM,
3-seed, from `exp-encoder-3seed.md` (job 12850): val-selected Test acc **0.8279 / 0.8279 /
0.8047** (seed 0/1/2; `enc3s_HateMM_openai_clip-...seed{0,1,2}_12850.trainlog`), final-epoch
**0.8186 / 0.8047 / 0.8140**. TARC arms are the identical command **plus** the target knob,
so every delta is paired within seed and attributable only to target-conditioning.

### G0 — data availability & normalisation  *(cost: 0 GPU, ~5 min; DONE in §4)*
- **Pass** (already met): HateMM `target` normalised to 8 codes, stem-join to jsonl 744/744,
  non-hate coverage confirmed, MHC confirmed target-less.
- **Kill:** would only fire if the join failed or target were hate-only — neither holds.
- Deliverable to persist before G1: `data/gt/HateMM/target_map.json`
  `{video_stem -> {"targets": [...], "primary": code}}` (built once, no MLLM).

### G1 — GT-target oracle **ceiling** probe  *(cost: HateMM RGCL ~21-25 s/run; see §7)*
- **Nature:** this is a **ceiling estimate, not a result.** It runs `gt_oracle` +
  `--oracle_probe True` (stamped `ORACLE_CEILING`, §5); its numbers **never enter the main
  table**. It answers only "does perfect target even have headroom to exploit?" — if not, no
  MLLM-predicted (noisier) target can, and the line dies here at zero MLLM cost.
- **Design:** for each variant V1/V3 (train-only, no test-target needed) and V2 (test-target
  = GT), run frozen-CLIP RGCL on **HateMM**, **3 seeds** (0/1/2), knob ON vs the byte-identical
  knob-OFF control, **paired within seed**, GT target for all videos.
- **Metric:** Val_Retrieval **acc** (the selection metric) and Test acc, both protocols.
- **Kill number (pre-registered):** a variant is **killed** unless its 3-seed mean paired
  **Δacc ≥ +0.015** on the **validation** split under **at least one** protocol with sign
  **≥ 2/3 seeds positive**. (Rationale: +0.015 is the minimal ceiling that could plausibly
  survive MLLM-target noise into a +0.03 test effect; below it, a real MLLM-predicted target
  — which is strictly noisier than GT — cannot reach the +0.03 bar. The probe is judged on
  **val** to keep the single test-touch for G3.)
- **Sweep budget:** V1 `{prefer, require}` × 3 seeds = 6 runs + 3 OFF controls; V3
  `lambda_tarc ∈ {0.1, 0.5}` × 3 seeds = 6 + shared OFF; V2 `gamma ∈ {0.5, 1.0}` × 3 = 6.
  ≈ **24-30 runs × ~25 s ≈ 10-13 min** total on one A100, one serial sbatch. **No MLLM.**
- **If G1 fails for all variants → kill the entire B-line** (do not spend any MLLM inference).

### G2 — MLLM target-prediction quality  *(only if G1 passes; cost: MLLM inference)*
- **Design:** the same MLLM already used as encoder (Qwen2.5-VL-7B; optionally 32B) predicts
  `target` per HateMM video (transcript + sampled frames, closed 8-way + "none"). GT is used
  **only to score** the prediction (macro-F1); the RGCL runs in this gate consume the
  MLLM-pred target, never GT (§5). Output cached to `data/gt/HateMM/target_pred_<mllm>.json`.
- **Kill number:** predicted-vs-GT **macro-F1 over the target classes ≥ 0.60** on the
  effective label set (Blacks/Jews/Other-merged, given the skew) **AND** substituting
  predicted target into the winning G1 variant retains **≥ 60%** of the GT-target val Δacc
  (i.e. predicted-target Δacc ≥ 0.6 × GT-target Δacc, 3-seed mean). If the MLLM cannot
  predict target well enough to keep most of the oracle gain → kill (the ceiling was real but
  unreachable).
- **Cost:** ~1083 HateMM videos × one MLLM pass; reuse cached frames/transcripts. Bounded,
  no training.

### G3 — real 3-seed paired confirmation  *(only if G2 passes; the single test-touch)*
- **Design:** winning variant, **MLLM-predicted target on EVERY dataset including HateMM**
  (train-side for V1/V3; train+test for V2) — **the method never reads GT target** (`gt_oracle`
  is not permitted here, §5), so HateMM is a genuine MLLM-integration testbed, not a GT-target
  shortcut. frozen-CLIP RGCL, **HateMM and MHC-EN**, 3 seeds, knob ON vs OFF, **both
  protocols**, judged by the **exact** enc3seed decision rule (`exp-encoder-3seed.md:73-85`).
- **Pass number (pre-registered, unchanged from the campaign bar):** mean paired
  **Δacc ≥ +0.030 AND Δpaired-macro-F1 ≥ +0.030 AND sign 3/3 positive**, on **≥ 1 dataset**,
  under a **stated** protocol (each protocol judged separately). Both datasets are now
  fully MLLM-driven results (neither touches GT target). **MHC is the strongest headline**
  because it has no GT target even in principle; a **HateMM pass** is equally GT-free but
  weaker as a novelty claim (there GT *could* have supplied target, and the MLLM merely
  substitutes for it). A HateMM-only pass is reported as "MLLM-predicted target helps HateMM,
  MHC fails", mirroring the encoder result.
- **Test touch:** this is the **only** time the test set is read (val used for all G1/G2
  selection). One serial sbatch, ~12-16 runs × ~25 s + the MHC MLLM-target inference.

---

## 7. G1 probe execution checklist (concrete)

1. **Script/template.** Clone `scripts/slurm/enc3seed.sbatch` → `scripts/slurm/tarc_g1.sbatch`
   (do NOT edit enc3seed). Each config = the enc3seed python command (`--dataset HateMM
   --model openai_clip-vit-large-patch14-336_HF --batch_size 64 --lr 1e-4 --epochs 30
   --topk 20 --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align
   --hard_negatives_loss True --no_hard_negatives 1 --metric cos --loss triplet
   --hybrid_loss True --warmup 5 --majority_voting arithmetic --no_pseudo_gold_positives 1
   --lambda_seg 0 --Faiss_GPU False`) **plus** exactly one TARC knob per arm.
2. **Target injection (minimal code, all flag-gated, all no-op at default):**
   - Build `data/gt/HateMM/target_map.json` once (stem-join CSV→codes; primary target for
     single-label training; multi-target hate video → its primary/first-listed code at G1,
     with a `--tarc_multitarget {primary, any}` knob reserved for later). **This oracle map is
     read only under the `gt_oracle` path.**
   - Add the gated source flag `--tarc_target_source {off, mllm_pred, gt_oracle}` (default
     `off` = byte-identical baseline) + `--oracle_probe` (required for `gt_oracle`; asserts and
     stamps `ORACLE_CEILING`, §5) + the per-variant knobs `--tarc_hn_mode {off,prefer,require}`
     (V1), `--lambda_tarc` (V3), `--tarc_vote_gamma` (V2) in the arg block
     `src/run_rac.py:305-408`. **G1 arms run `--tarc_target_source gt_oracle --oracle_probe
     True`; every G3 arm will run `--tarc_target_source mllm_pred` (no `--oracle_probe`).**
   - Build `target_pack` in `main()` from `train_set.ids` (precedent aux_pack
     `src/run_rac.py:1054-1082`); thread through `model_pass`
     (`src/run_rac.py:487-501`, `:548-565`) into `compute_loss` (`src/model/loss.py:12-23`)
     and into `retrieve_evaluate_RAC_` for V2 (`src/model/evaluate_rac.py:429-468`).
   - Implement the three edits of §2. **Verify no-op:** with all knobs at default the run must
     reproduce the enc3seed HateMM CLIP seed0 numbers **to 4 decimals** (val-sel 0.8172 F1 /
     0.8279 acc; final 0.7997 / 0.8186) — this is the pre-commit correctness gate before any
     ON arm is trusted (same discipline as the enc3seed code-version audit).
3. **Seeds:** 0 / 1 / 2, paired ON-vs-OFF within seed.
4. **Selection/protocol:** identical parser to the sbatch (`warmup≥5`, val-acc, roc
   tie-break) for protocol A; final epoch 29 for protocol B. Judge on **val** Δacc at G1.
5. **Estimated cost:** ~24-30 runs × ~25 s ≈ **10-13 min**, one serial sbatch, 1 GPU, **no
   MLLM inference**.
6. **Recommended code-review before submit:** the retrieval.py:330-362 id-collection edit
   (V1) touches the shuffled-index build — route through `codex-code-review` (touches model
   internals / retrieval) before any GPU submit, per project convention.

---

## 8. Where this line most likely dies (honest prior)

**Most likely: G1 (the oracle probe), via H0.** The fused CLIP+transcript space is
topic-dominated, so the label-only nearest opposite-label neighbour is *already* usually
community-matched; target-conditioning the mining (V1) and the vote (V2) then buys little,
and the probe lands **< +0.015 val Δacc** within seed noise. This is the same failure shape
that killed P2 neighbour-rerank (comparability ⊥ vote-correctness) and P3 pooling (clean
probe, flat training) — a plausibility check that passes need not move the trained metric.

**Second: the target skew.** Blacks alone = 76.5% and Blacks+Jews = 97.4% of hate targets,
so `target` carries ≈1 bit of information; "condition on community" ≈ "condition on
is-it-about-Blacks". Even a perfect oracle has little structure to exploit, and V3's
per-target centroids are unstable for the tail classes (Asian n=1, Sexits n=5) at batch 64.

**Third (if G1 somehow passes): G2 on MHC.** HateMM has GT target so a HateMM G1 pass is
cheap, but the headline requires MHC, where target is MLLM-predicted from scratch. Predicting
fine-grained victim community from a short multilingual clip is hard; noise likely erases a
GT-sized ceiling before it reaches the +0.03 G3 bar — and the whole campaign has met that bar
only once (the encoder swap, HateMM only). My point estimate: **G1 kills V2, G1 marginal for
V1/V3 on HateMM, G2/G3 fail to reach +0.03 on MHC.** The value of the design is that G1 costs
~12 min and **zero MLLM inference**, so the line is falsified cheaply either way.

---

## 9. Implementation record (G0+G1)

Implemented 2026-07-13 (code + data prep only; **no SLURM job submitted, nothing
committed**). All source paths cite the post-change line numbers.

### 9.1 Changed / new files (file:line)

**New files**
- `scripts/analysis/build_tarc_target_map.py` — G0 builder (normalise → `target_map.json`
  + self-check). Ran live on the login node (pure data, no GPU).
- `data/gt/HateMM/target_map.json` — G0 output, 1083 video-stem entries + `_meta` header
  (code dict, num_targets=8, primary rule). Read **only** on the `gt_oracle` path.
- `scripts/slurm/tarc_g1.sbatch` — G1 oracle-probe runner (clone of `enc3seed.sbatch`;
  original untouched).

**`src/run_rac.py`** (all flag-gated; default = no-op)
- `parse_args` arg block: `--tarc_target_source` (:415), `--oracle_probe` (:424),
  `--tarc_hn_mode` (:429), `--lambda_tarc` (:437), `--tarc_vote_gamma` (:441),
  `--tarc_multitarget` (:446), `--tarc_mllm` (:452).
- `main()`: `gt_oracle`⇒`oracle_probe` **assertion** + run-name stamping incl.
  `_ORACLE_CEILING` (:883–897); trainlog provenance print (:913–922);
  **`target_pack` builder** (:1219–1263, reads `data/gt/<ds>/target_map.json` via
  `os.path.join(args.path,"gt",args.dataset,...)`; `mllm_pred` raises `FileNotFoundError`
  this round); pass into non-EM `model_pass` (:1349).
- `model_pass`: `target_pack=None` param (:548); threaded into `compute_loss` (:612),
  both `retrieve_evaluate_RAC_` calls (:670,:688), and `tarc_vote_gamma` into both
  `compute_metrics_retrieval` calls (:675,:693). The EM (`consensus/selfscore`)
  `model_pass` call is intentionally **not** wired (TARC arms use `lambda_seg 0`, non-EM).

**`src/model/loss.py`**
- `compute_loss`: `target_pack=None` param (:23); passed to both
  `dense_retrieve_hard_negatives_pseudo_positive` call sites with `query_ids=ids`
  (:261–262, :286–287).
- **V3 additive term** after the P4 `lambda_aux` block (:580–588): `if lambda_tarc>0 and
  target_pack is not None`.
- **`compute_target_loss(feats, batch_ids, target_pack, labels, args)`** new (:626),
  mirrors `compute_aux_loss`; reuses the grad-tracked `feats`, no new params, no RNG.

**`src/utils/retrieval.py`** (`dense_retrieve_hard_negatives_pseudo_positive`, the one
non-trivial edit)
- `target_pack=None, query_ids=None` params (:318); `tarc_active` gate (:324).
- Index-rebuild loop collects `batch["ids"]` (:346,:349) and caches the row-aligned
  primary-target vector `target_pack["_train_targets"]` (:381–384), so `train_targets`
  is aligned to the shuffled FAISS row order and persists across the epoch's cached
  index (the draft-flagged shuffle hazard).
- Mining loop split: **original loop kept verbatim** under `if not tarc_active:` (:466);
  target-preferred `else` branch (:515) with `_store_hn`/`_store_pp` helpers, phase-A
  (same-target opposite-label) + phase-B (`prefer` fallback to any opposite-label);
  `require` leaves unfilled slots zero (baseline behaviour). Pseudo-gold-positive
  selection is byte-identical to baseline (pp preference held off, per §2).

**`src/model/evaluate_rac.py`** (`retrieve_evaluate_RAC_`)
- `target_pack=None` param (:323); `tarc_v2` gate = target_pack set **and**
  `tarc_vote_gamma>0` (:329); attaches `query_target` + `retrieved_target` to each
  logging-dict entry (:475–480).

**`src/utils/metrics.py`** (`compute_metrics_retrieval`)
- `tarc_vote_gamma=0.0` param (:214); in the `use_sim` branch, multiply each neighbour's
  vote by `(1+gamma)` when its target matches the query's (:274–279). `gamma==0` (or no
  target keys) ⇒ identity.

### 9.2 `target_map.json` class-count cross-check (G0 self-check output)

Stem-join coverage **train 744/744, val 107/107, test 215/215** (all hit). Hate-video
per-target presence (video counted once per distinct target) reproduces §4 exactly:

| target | Blacks | Jews | Whites | Others | LGBTQ | Muslims | Sexits | Asian |
|---|---|---|---|---|---|---|---|---|
| built | **329** | **90** | **18** | **12** | **12** | **10** | **5** | **1** |
| §4 expected | 329 | 90 | 18 | 12 | 12 | 10 | 5 | 1 |

Hate multi-target = **42** (§4: 42), hate empty-target = **1** (§4: 1). Hate primary
(first-listed): Blacks 321, Jews 67, Whites 15, Others 10, Muslims 7, Sexits 5, LGBTQ 4,
Asian 1 — matches §4. Code dict (draft order): Blacks0 Jews1 Whites2 Others3 LGBTQ4
Muslims5 Sexits6 Asian7. Train-split primary histogram: {Blacks 332, Others 270, Jews 72,
Whites 18, Muslims 7, LGBTQ 6, Sexits 5, Asian 2, no-target −1 ×32} = 712/744 with a
target. **G0 SELF-CHECK: PASS.** (Note: §4's *non-hate* prose figures — "Others 389,
Blacks 150…" — are approximate; the authoritative live counts are non-hate presence
Others 395 / Blacks 161 / Jews 39 / Whites 15 / Muslims 7 / LGBTQ 3 / Sexits 2 / Asian 1.
The hate table §4 cross-checks exactly; the small non-hate divergence does not affect any
gate.)

### 9.3 G1 sbatch arms (`scripts/slurm/tarc_g1.sbatch`)

**21 arms**, serial, HateMM / frozen-CLIP, run order OFF → V1 → V3 → V2:

| block | arm label | flags (added to the enc3seed command) | seeds |
|---|---|---|---|
| control | `off` ×3 | `--tarc_target_source off` | 0/1/2 |
| V1 | `v1prefer` ×3 | `gt_oracle --oracle_probe True --tarc_hn_mode prefer` | 0/1/2 |
| V1 | `v1require` ×3 | `… --tarc_hn_mode require` | 0/1/2 |
| V3 | `v3lt0.1` ×3 | `… --lambda_tarc 0.1` | 0/1/2 |
| V3 | `v3lt0.5` ×3 | `… --lambda_tarc 0.5` | 0/1/2 |
| V2 | `v2vg0.5` ×3 | `… --tarc_vote_gamma 0.5` | 0/1/2 |
| V2 | `v2vg1.0` ×3 | `… --tarc_vote_gamma 1.0` | 0/1/2 |

The V2 arms read GT target for the **query** (test/val) side too — legal at G1 (oracle
probe). Each arm ≈ 25 s ⇒ **≈ 9–12 min total** on one A100, 1 GPU, no `--time`, conda
`HateVideo`, no MLLM. Every arm auto-prints its VAL-selected (warmup≥5) + FINAL-epoch
readout and a `RESULT_ROW` line (VAL acc/F1 for the G1 kill decision + TEST acc/F1).
OFF-arm expectation (verified live from job 12850 trainlogs): seed0 VALSEL acc 0.8279 /
F1 0.8172, FINAL acc 0.8186 / F1 0.7997; seed1 VALSEL 0.8279 / 0.8163, FINAL 0.8047 /
0.7822; seed2 VALSEL 0.8047 / 0.7920, FINAL 0.8140 / 0.7988.

### 9.4 No-op discipline self-check

- **py_compile**: all 5 edited source files + the builder compile clean.
- **argparse**: defaults are a full no-op (`tarc_target_source=off`, `oracle_probe=False`,
  `tarc_hn_mode=off`, `lambda_tarc=0.0`, `tarc_vote_gamma=0.0`, `tarc_multitarget=primary`).
  `gt_oracle` without `--oracle_probe True` raises `AssertionError` **before** any file/
  data side effect.
- **GT-read audit (grep)**: `target_map.json` is read at exactly **one** training-pipeline
  site (`run_rac.py:1232`, the `gt_oracle` branch) plus the builder that writes it — no
  other reader. `gt_oracle` never enters a `mllm_pred` code path.
- **Branch gating (grep)**: every TARC branch is guarded — V1 `tarc_active`
  (target_pack set ∧ `tarc_hn_mode≠off` ∧ source≠off ∧ query_ids given), V3
  `lambda_tarc>0 ∧ target_pack`, V2 evaluator `tarc_v2` (target_pack ∧ `gamma>0`), V2
  metrics `gamma>0 ∧ "query_target" in value`. With `--tarc_target_source off`,
  `target_pack` is `None` ⇒ no target array is built, no logging-dict key is added, no
  loss term is computed, and the mining runs the original loop verbatim. No new torch
  tensor and no torch-RNG draw occurs on any default path (the id→code work is pure
  numpy/python), so seed reproduction is preserved.
- **CPU unit test** (`scratchpad/tarc_unit_test.py`, forced `CUDA_VISIBLE_DEVICES=""`, no
  training): 10/10 pass — V3 value matches hand calc + gradient flows + single-class
  target skipped; V2 `gamma=0`≡baseline and `gamma=1` flips the intended vote; V1 `off`
  (with and without target_pack present) ≡ baseline nearest-opposite-label, `require`/
  `prefer` select the same-target opposite-label neighbour.

### 9.5 Deviations from the pre-registration

1. **V3 loss sign (load-bearing).** §2 wrote `L = relu(m − d(μ_hate, μ_benign))` naming
   `d = _pair_similarity`, but `_pair_similarity` is a **similarity** (higher = closer),
   so `relu(m − sim)` would *pull* the centroids together — the opposite of the stated
   "push apart". `compute_target_loss` instead hinges on similarity directly:
   `L = mean_T relu(margin + sim(μ_{T,hate}, μ_{T,benign}))`, which drives the same-
   community hate/benign centroids to similarity ≤ −margin (i.e. apart), realising the
   stated intent. `margin` reuses `--triplet_margin` (0.1); no new hyperparameter.
2. **`target_map.json` built over all 1083 CSV rows** (keyed by stem), not only the 1066
   split ids, so `id_to_target` covers every query the pipeline can raise (train + val +
   test) and the §4 hate counts reproduce exactly; the 17 non-split stems are never
   queried.
3. **`--tarc_multitarget any` is a stub**: selecting it raises `NotImplementedError` (G1
   uses `primary` only, §4). The primary code is fully wired.
4. **Missing-target videos** (primary code −1) never target-match anything (match test is
   `a==b ∧ a≥0`): in V1 `require` they get zero-filled hard-negative slots (baseline
   behaviour for too-few HN); in V2 they get the identity multiplier.
5. **EM path not wired** (§9.1): combining `gt_oracle` with `seg_mode consensus/selfscore`
   would silently no-op TARC; out of scope for G1/G3 which use `lambda_seg 0`.

---

## 10. G1 RESULTS (job 12975, 2026-07-13)

**Job:** `12975` (`tarc_g1`), node `foscsmlprd01`. Submitted 04:39, released from
`JobHeldUser` ~05:00, all 21 arms computed by ~05:12 (elapsed ~11:54 of compute; job then
held ~8 min on the trailing `b2_push.sh` rclone upload — irrelevant to results). Out file:
`slurm/logs/tarc_g1_12975.out`; per-arm trainlogs
`slurm/logs/tarc_g1_<arm>_seed<s>_12975.trainlog`. 21/21 `RESULT_ROW` emitted.

**Judgment metric (§6, re-stated):** VAL-split Δacc, variant − OFF, paired within seed.
Two protocols judged separately — **A = val-selected** (warmup≥5, val-acc, roc tie-break),
**B = final epoch 29**. A variant **passes** iff, under **≥1 protocol**, 3-seed mean
Δacc **≥ +0.015 AND sign ≥ 2/3 seeds positive**. TEST numbers below are **recorded only,
never used for the G1 decision** (test-touch reserved for G3).

### 10.1 OFF no-op reproduction gate (MUST match enc3seed job-12850 to 4 dp)

`RESULT_ROW` TEST acc/F1 from the out file vs the sbatch header (`tarc_g1.sbatch:19-21`):

| seed | valsel TEST acc/F1 (this run) | header expect | final TEST acc/F1 (this run) | header expect | src |
|---|---|---|---|---|---|
| 0 | 0.8279 / 0.8172 | 0.8279 / 0.8172 | 0.8186 / 0.7997 | 0.8186 / 0.7997 | `out:319` |
| 1 | 0.8279 / 0.8163 | 0.8279 / 0.8163 | 0.8047 / 0.7822 | 0.8047 / 0.7822 | `out:633` |
| 2 | 0.8047 / 0.7920 | 0.8047 / 0.7920 | 0.8140 / 0.7988 | 0.8140 / 0.7988 | `out:947` |

**All 6 pairs reproduce to 4 dp ⇒ no-op gate PASS ⇒ batch is valid, G1 decidable.**

### 10.2 VAL-split acc, both protocols (the delta baseline is OFF)

Protocol-A source = `VALSEL_VAL` line in the out file; Protocol-B source = the
`Val_Retrieval Epoch 29 … acc:` line in each trainlog.

| arm | A s0 | A s1 | A s2 | A src | B s0 | B s1 | B s2 | B src (`…seed{0,1,2}…trainlog:ln`) |
|---|---|---|---|---|---|---|---|---|
| **off**   | 0.8411 | 0.8505 | 0.8692 | `out:318/632/946` | 0.7944 | 0.8131 | 0.8224 | `:303/:303/:303` |
| v1prefer  | 0.8411 | 0.8411 | 0.8598 | `out:1257/1570/1883` | 0.8037 | 0.8411 | 0.8411 | `:300/:301/:302` |
| v1require | 0.8318 | 0.8411 | 0.8505 | `out:2194/2508/2820` | 0.8131 | 0.8224 | 0.8224 | `:300/:303/:301` |
| v3lt0.1   | 0.8505 | 0.8411 | 0.8411 | `out:3132/3447/3762` | 0.8224 | 0.8411 | 0.8318 | `:301/:304/:304` |
| v3lt0.5   | 0.8037 | 0.7944 | 0.8131 | `out:4074/4386/4700` | 0.8037 | 0.7664 | 0.8037 | `:300/:301/:303` |
| v2vg0.5   | 0.8505 | 0.8505 | 0.8692 | `out:5009/5320/5634` | 0.7944 | 0.8131 | 0.8318 | `:298/:300/:303` |
| v2vg1.0   | 0.8505 | 0.8692 | 0.8692 | `out:5943/6254/6566` | 0.7944 | 0.8131 | 0.8318 | `:298/:300/:301` |

### 10.3 Paired Δacc (variant − OFF) and per-variant verdict

**Protocol A (val-selected):**

| arm | Δ s0 | Δ s1 | Δ s2 | mean | pos | ≥+.015 ∧ ≥2/3 |
|---|---|---|---|---|---|---|
| v1prefer  | +0.0000 | −0.0094 | −0.0094 | −0.0063 | 0/3 | ✗ |
| v1require | −0.0093 | −0.0094 | −0.0187 | −0.0125 | 0/3 | ✗ |
| v3lt0.1   | +0.0094 | −0.0094 | −0.0281 | −0.0094 | 1/3 | ✗ |
| v3lt0.5   | −0.0374 | −0.0561 | −0.0561 | −0.0499 | 0/3 | ✗ |
| v2vg0.5   | +0.0094 | +0.0000 | +0.0000 | +0.0031 | 1/3 | ✗ |
| v2vg1.0   | +0.0094 | +0.0187 | +0.0000 | +0.0094 | 2/3 | ✗ (mean<0.015) |

**Protocol B (final epoch 29):**

| arm | Δ s0 | Δ s1 | Δ s2 | mean | pos | ≥+.015 ∧ ≥2/3 |
|---|---|---|---|---|---|---|
| **v1prefer**  | +0.0093 | +0.0280 | +0.0187 | **+0.0187** | 3/3 | **✓ PASS** |
| v1require | +0.0187 | +0.0093 | +0.0000 | +0.0093 | 2/3 | ✗ (mean<0.015) |
| **v3lt0.1**   | +0.0280 | +0.0280 | +0.0094 | **+0.0218** | 3/3 | **✓ PASS** |
| v3lt0.5   | +0.0093 | −0.0467 | −0.0187 | −0.0187 | 1/3 | ✗ |
| v2vg0.5   | +0.0000 | +0.0000 | +0.0094 | +0.0031 | 1/3 | ✗ |
| v2vg1.0   | +0.0000 | +0.0000 | +0.0094 | +0.0031 | 1/3 | ✗ |

**Per-variant G1 verdict (pass if either protocol passes):**

| variant | verdict | via |
|---|---|---|
| **v1prefer** (V1 target-matched HN mining, prefer) | **PASS** | protocol B |
| v1require (V1 mining, require) | KILL | — |
| **v3lt0.1** (V3 intra-target separation, λ=0.1) | **PASS** | protocol B |
| v3lt0.5 (V3 separation, λ=0.5) | KILL | hurts (over-regularised) |
| v2vg0.5 (V2 vote reweight, γ=0.5) | KILL | — |
| v2vg1.0 (V2 vote reweight, γ=1.0) | KILL | — |

### 10.4 G1 total verdict + honest caveats

**G1 PASSES for 2 of 6 variants (v1prefer, v3lt0.1). B-line is NOT killed** (pre-registration
§6: kill the whole line only if *all* variants fail). Per §6/§8 flow, the surviving variants
are eligible for **G2 (MLLM target-prediction quality)**.

Three caveats the verdict must carry forward — the pass is real but fragile:

1. **Protocol-B-only.** Both passes come *entirely* from the final-epoch protocol; under
   val-selected (peak) both are flat-to-negative (v1prefer −0.0063, v3lt0.1 −0.0094). The
   mechanism is late-epoch **regularisation**: OFF decays from val-peak ≈0.854 (mean) down
   to 0.810 (mean) by ep29, while the target-conditioned arms hold ≈0.829 (v1prefer) /
   0.832 (v3lt0.1). Target-conditioning slows OFF's late overfit; it does **not** raise the
   peak. Whether the "correct" protocol is peak or final is the same open protocol question
   flagged elsewhere in the campaign.
2. **TEST is flat (recorded, not judged).** Final-epoch TEST Δacc: v1prefer **+0.0031**
   (2/3 pos), v3lt0.1 **−0.0016** (1/3 pos); val-selected TEST Δacc both negative (v1prefer
   −0.0046, v3lt0.1 −0.0248). The oracle-ceiling val gain does **not** transfer to test —
   exactly the "clean probe, flat trained metric" shape §8-H0 predicted (cf. P3 pooling).
3. **This is an ORACLE ceiling.** Every passing arm read GT target (`ORACLE_CEILING`
   stamped, never a main-table number). A real MLLM-predicted target is strictly noisier;
   §6 set +0.015 as the *minimum* ceiling that could plausibly survive into a +0.03 test
   effect — these clear +0.015 on val but the test signal at the ceiling is already ~0, so
   the headroom for a noisier predictor to reach the G3 +0.03 test bar looks slim.

**Recommendation:** formally the pre-registration routes v1prefer + v3lt0.1 to G2, so the
line is not dead. But the ceiling is protocol-fragile and test-flat, so whether to spend the
G2 MLLM inference is a **user decision** (proceed to G2 on the two survivors, or stop the
line here on the weak-ceiling read). No test-touch was spent; G3 remains untouched.
