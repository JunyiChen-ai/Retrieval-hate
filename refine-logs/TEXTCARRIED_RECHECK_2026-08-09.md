# TEXT-CARRIED RECHECK — HateMM CLIP text-only vs image-only kNN AUC, empty transcripts removed

**Date** 2026-08-09. **CPU only, zero GPU, zero test-set touch.** Only
`data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt` and
`data/gt/HateMM/train.jsonl` are read. No dev/test metric enters the judgement.

**Trigger:** `refine-logs/EMPTY_TEXT_AUDIT_2026-08-09.md` §2d + §3b RECHECK-1 + §3d.
39/744 HateMM train rows carry `" "` as transcript; the CLIP text tower encodes them as
`[BOS, EOS]` → one identical 768-d vector, and that single point is 36/39 non-hate (P(hate)=0.077)
against a 0.401 split base rate. Any text-inclusive kNN read-out gets a near-free correct answer
on 5.2 % of the memory bank from a constant.

**Conclusion under test:** "HateMM is TEXT-carried" — CLIP text-only train-LOO kNN AUC
**0.8471** ≥ image-only **0.8255**, margin **+0.0216**, recorded in
`refine-logs/HATEMM_LORA_STREAM_DECOMP.md` §Q1 / `scripts/analysis/hatemm_lora_stream_decomp_out.json`
(`geometry.CLIP.{text,img}.train_loo_knn.auc`), propagated into `DRAFT_analysis_chapter.md` and
`DRAFT_experiments_chapter.md`.

---

## FROZEN DECISION RULE (written before any 705-row number was computed)

Primary footing = **train-LOO kNN AUC, k=20, cosine-weighted signed vote**, exactly the protocol of
`scripts/analysis/encoder_swap_geometry.py::loo_knn` reused by `hatemm_lora_stream_decomp.py`
(per-stream L2-norm, `np.fill_diagonal(S, -inf)`, `score = Σ_j cos_ij · (2y_j−1)` over the top-20).
Clean condition = the 39 empty-transcript rows deleted from **both** the query and the memory side
(n = 705), per `EMPTY_TEXT_AUDIT_2026-08-09.md` §3d.

Let `Δ705 = AUC_text(705) − AUC_img(705)`.

- **SURVIVES (conclusion stands, footnote added)** iff `Δ705 > 0` **and** `Δ705 ≥ 0.0216/2 = 0.0108`.
- **RETRACTED** iff `Δ705 ≤ 0` (text advantage gone or reversed), **or** `0 < Δ705 < 0.0108`
  (advantage survives in sign but is more than half manufactured by the constant cluster —
  the recorded margin is not defensible as stated).
- In the RETRACTED branch, mark the conclusion RETRACTED and annotate every downstream item on
  the `EMPTY_TEXT_AUDIT_2026-08-09.md` §3 RECHECK list.

Alignment check (must pass before the judgement is read): the 744-row rerun must reproduce
0.8471 / 0.8255 to ±0.0005. If it does not, the protocol is not the original one and the
judgement is void.

Script: `scripts/analysis/textcarried_recheck.py` → `scripts/analysis/textcarried_recheck_OUT.json`.

---

## RESULTS

Run: `logging/runs/textcarried_recheck/run.{log,pid}`, ~40 s CPU, single pass.
Raw: `scripts/analysis/textcarried_recheck_OUT.json`.

### 0. Empty-row identification — two independent routes agree exactly

| route | n |
|---|---|
| `data/gt/HateMM/train.jsonl` rows whose `text.strip() == ""` | **39** |
| rows sharing the modal duplicated `text_feats` vector (the `[BOS, EOS]` constant) | **39** |
| set agreement | **exact** (`empty_id_sets_agree: true`) |

744 rows → 695 unique text vectors, matching `EMPTY_TEXT_AUDIT_2026-08-09.md` §0.
The cluster is 3/39 hate, **P(hate) = 0.077** vs split base rate 0.401. Dropping it lifts the
train positive rate 0.4005 → 0.4184.

### 1. Alignment check — the 744-row rerun reproduces the historical pair

| stream | historical (`hatemm_lora_stream_decomp_out.json`) | this rerun (744) | diff |
|---|---|---|---|
| text-only | 0.8471 | **0.84715** | +0.00013 |
| image-only | 0.8255 | **0.82553** | +0.00003 |
| concat | 0.8667 | 0.86672 | +0.00002 |

Both inside the ±0.0005 tolerance ⇒ **protocol aligned, judgement is readable.**
(Residual is float32→float64 / `argpartition` tie-order only.)

### 2. Headline — train-LOO kNN AUC (k=20), full 744 vs clean 705

| stream | 744 (with the 39) | 705 (39 deleted, query **and** memory) | Δ(705−744) |
|---|---|---|---|
| **image-only** | 0.8255 | **0.8202** | **−0.0054** |
| **text-only** | 0.8472 | **0.8477** | **+0.0004** |
| concat | 0.8667 | 0.8648 | −0.0020 |
| **margin text − image** | **+0.0217** | **+0.0275** | **+0.0058 (margin WIDENS)** |

Secondary read-outs on the same key (LOO accuracy / balanced accuracy), for completeness:

| condition | img acc / bal | text acc / bal | concat acc / bal |
|---|---|---|---|
| 744 | 0.7634 / 0.7687 | 0.7406 / 0.7608 | 0.7715 / 0.7782 |
| 705 | 0.7589 / 0.7646 | 0.7319 / 0.7524 | 0.7660 / 0.7745 |

### 3. Verdict — **SURVIVES** (frozen rule, SURVIVES branch)

`Δ705 = +0.0275 > 0` and `+0.0275 ≥ 0.0108` (half the original 0.0216 margin). Both clauses of the
frozen rule are met, with margin to spare — the threshold is cleared by 2.5×.

**The "HateMM is text-carried" conclusion is NOT retracted. It is confirmed and, on this footing,
mildly strengthened.**

The mechanism ran opposite to the audit's hypothesis, and this is the substantive finding:

- The constant text vector did **not** inflate the text stream. Text-only AUC is flat to four
  decimals when the 39 rows go (+0.0004). The degenerate cluster is a single point that all 39
  rows retrieve from each other; within a *rank-based* AUC over the whole train set it neither
  helps nor hurts materially, because those rows were already being ranked as one block.
- What the 39 rows were actually propping up is the **image** stream (−0.0054 when removed). They
  are 93 % non-hate rows that the CLIP *image* key happened to rank correctly, i.e. free easy
  negatives for the image read-out. Removing them raises the positive rate to 0.418 and takes that
  freebie away.
- So the audit §2d prediction ("can *inflate* a text-inclusive read-out") is **not borne out for
  this particular statistic**. The audit's core census facts (39 rows, one vector, P(hate)=0.077)
  all reproduce exactly; only the directional inference about this AUC pair was wrong.

### 4. Honest caveat that the recheck does surface (new, not in the original record)

Query-side paired stratified bootstrap (2000 resamples, memory bank held fixed — descriptive
spread, **not** an inferential test; the frozen rule judged the point estimate):

| condition | Δ(text−img) | 95 % bootstrap interval | P(Δ>0) |
|---|---|---|---|
| 744 | +0.0217 | [−0.0163, +0.0577] | 0.870 |
| 705 | +0.0275 | [−0.0096, +0.0637] | 0.917 |

**Both intervals straddle zero.** The text > image ordering is a consistent point estimate but is
not separated from zero at n = 705 by row resampling. This was never stated when 0.847 vs 0.826
was recorded. The claim should be phrased as an *ordering that holds on every footing measured*
(and it does: CLIP 0.847/0.837, frozen-Qwen 0.888/0.875, LoRA-Qwen 0.920/0.899, train-LOO and dev,
six of six cells in `HATEMM_LORA_STREAM_DECOMP.md` §Q1), **not** as a resolved single-comparison
gap. The 6/6 consistency, not the 0.021 CLIP margin alone, is what carries the conclusion.

Scope note: this recheck cleaned **HateMM train only** (39 rows). Dev (9) and test (26) empty rows
were deliberately not touched — dev is out of scope for the frozen rule and test is off-limits.

---

## 5. Chained annotations — `EMPTY_TEXT_AUDIT_2026-08-09.md` §3 RECHECK list

| audit item | prior verdict | now |
|---|---|---|
| §3b row 1 — **"HateMM is TEXT-carried", CLIP text-only 0.847 ≥ image-only 0.826** (`HATEMM_LORA_STREAM_DECOMP.md` §Q1 → `DRAFT_analysis_chapter.md`, `DRAFT_experiments_chapter.md`) | RECHECK (highest priority) | **CLOSED — CONFIRMED.** Clean-705 margin +0.0275 ≥ +0.0217. Attach the §4 caveat (bootstrap straddles zero; lean on the 6/6 cross-encoder/footing consistency) wherever the 0.847/0.826 pair is quoted. No retraction, no number change. |
| §3b row 2 — **CLIP-vs-Qwen encoder delta +4.2 acc / +4.4 mF1** (magnitude only; direction already judged safe) | RECHECK | **PARTIALLY ADDRESSED — direction confirmed, magnitude still open.** The relevant new fact is that the CLIP *text* stream is unharmed by the 39 rows (+0.0004), so the "CLIP is handicapped on its text stream" story that would have inflated the Qwen delta is **not supported at the geometry level**. The CLIP *image* stream is the one that loses 0.0054 on clean rows. Closing this properly needs a 705-row-restricted frozen-CLIP-vs-frozen-Qwen head comparison, which is a GPU/head-retrain item, not this CPU pass. Downgrade to **FOOTNOTE**: quote +4.2 as measured, note the CLIP text stream is confirmed not to be the source of any deflation. |
| §3b row 5 — **`ERRPAT_HateMM_2026-07-26.md` FN1 mechanism story** (0-for-30 empty-transcript test items, "retrieval geometry / length-conditional class prior") | RECHECK (mechanism, rewrite not re-run) | **UNCHANGED — still a rewrite.** This recheck confirms the enabling fact (the 39 train rows are literally one point at cosine 1.0, P(hate)=0.077), which is exactly the simpler explanation the audit proposed. Nothing here contradicts it; it remains a documentation edit, not a re-run. Out of scope for this pass. |
| §3b `Z_best` KILL cluster, FA/premise-(d), `C3_NONTARGET_PILOT_RECORD`, `ROUTER_GATE_RECORD`, the inherited 0.8279/0.8172 tail, `DEGEN_FEATURE_FIX §4` | FOOTNOTE | **UNCHANGED.** All were footnote-grade on conservativeness/upper-bound arguments that this pass neither strengthens nor weakens. One relevant addition for the `Z_best` cluster: the constant text block's contribution to the conditioning key is smaller than assumed, since removing it does not move a text-only read-out — the "raises the bar ⇒ kills are conservative" argument still holds and is now, if anything, closer to neutral. |
| §3a HateClipSeg items | CLEAR / one FOOTNOTE (OP-2 rationale) | **UNCHANGED.** No HateMM-train evidence bears on them. |

**Net effect on the audit's §3c summary count: RECHECK 3 → RECHECK 1** (the ERRPAT mechanism
rewrite), with the top-priority text-carried item **closed as confirmed** and the encoder-delta
item **downgraded to FOOTNOTE** pending a head-level 705-row rerun if the magnitude is ever
load-bearing for a paper claim.

Items **not** created by this pass and still open from the audit §4 (unchanged): the
`hateclipseg_prep.py:106` text wiring, the `EMPTY_TEXT` flag in
`generate_VideoCLIP_embedding_HF.py:244-257`, the 14 recoverable HateMM transcripts, and the
Gate-0 appendix OP-2 correction.

