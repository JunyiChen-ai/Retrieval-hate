# Independent review — TERA Gate-0 implementation appendix (v1-draft)

> **Review round 1 of 1** (project rule: CPU-level experiments get at most one review round).
> **Reviewed artifacts:** `research-wiki/EXP_tera_gate0_impl_appendix.md` (v1-draft, 1218 lines),
> `research-wiki/tera_gate0_frozen_config.draft.json` (draft, `payload_sha256 = TO-FILL-AT-FREEZE`).
> **Frozen upstream:** `research-wiki/EXP_tera_gate0_prereg.md` (registered 2026-08-07).
> **Review date:** 2026-08-07. **Reviewer blinding:** no real feature, label, span or split
> content was reduced to a metric. Only source code, directory listings, file byte sizes, and
> structural key/length inspection of `p11_split.json` / `hate_spans.json` / `gold_segments.json`
> (key names and record counts, never values) were read. No candidate metric was computed.

## Review standard actually applied

Per the project's proportional-ceremony rule, only two defect classes may block:

- **(a)** defects that can produce a **wrong verdict** (including: a binding criterion left
  undefined so that the implementer can pick a reading *after* seeing numbers — this breaks the
  hard red line "decision rules frozen before results are seen");
- **(b)** defects that touch the **test set**, break **blinding**, or breach the
  **weak-supervision boundary**.

"A clean run would HALT" defects are accepted (cost = one rerun). Style, redundancy, prose and
documentary-drift issues are recorded as NOTE and never block.

---

## 1. Item-by-item adjudication

### 1.1 Scope compliance (appendix may instantiate, not legislate)

**PASS.** The appendix adds no arm, removes no arm, and changes no endpoint, threshold, split,
or decision rule. Arm list A0/A1/A2/A3/A4/O1/O2 and B0–B5 is exactly the prereg's. §12's
"decision_rules_copied_from_prereg" block in the draft JSON reproduces prereg §4.3, §5.2, §6
verbatim, including the `+0.050 / +0.050 / +0.020 / CI-excludes-zero / 0.60 / 0.53` thresholds
and the four Gate-C criteria. The verdict vocabulary matches prereg §9. No candidate result
appears anywhere in either file.

### 1.2 O1 / O2 vs prereg §5.1 — oracle rules

**PASS.**

- **O1** (`§5.1` pseudocode): branches only on `spans.get(v, [])` and `D[v]`. The video label
  array is not in scope of `o1_video_logit`. The prereg's exact words — "may inspect span
  presence but may not branch directly on the video label" — are honoured literally. Overlap is
  strict positive duration (`min(hi,b) - max(lo,a) > 0.0`), matching prereg §5.1's
  "positive-duration overlap". The empty-`W` fallback is **exactly** registered A1 mean pooling
  over all K windows, as prereg §5.1 requires, and because the A1 head is affine the two
  admissible readings of "pool the segment scores" (pool logits vs. pool representations then
  apply the head) coincide identically — the appendix's proof of this is correct and removes a
  real ambiguity in the prereg.
- The appendix's disclosure that negatives typically take the fallback (span presence correlates
  with the label) is the correct, non-deployable reading, and `o1_fallback` is reported per row.
- **O2** (`§5.2`): `argmax_k z` for `y=1`, `argmin_k z` for `y=0` — verbatim what prereg §5.1
  names ("may choose max for a positive and min for a negative"). Marked `label_leaking: true`
  *and* `oracle_or_eval_only: true`. The singleton-optimality argument (the label-aware optimum
  of `mean_{k∈S} z` over non-empty subsets is attained at a singleton) is correct, so no subset
  search is needed and the rule is deterministic.
- Neither oracle can select a deployable arm: `arm_D` is registered as
  `argmax over {A2,A3,A4}` from inner-OOF only, and the JSON carries
  `may_select_a_deployable_arm: false` on both O1 and O2.
- Using the **A1 fold-trained head** as the shared "fixed fold-trained segment scorer" for both
  oracles is a defensible and in fact the *tightest* matched choice: A1 / O1 / O2 then form an
  exact nested family differing only in the selected window set. The prereg does not name a
  scorer, so this is inside the appendix's authority.

### 1.3 Nested OOF protocol vs prereg §7

**PASS on structure, BLOCKING-FIX on two undefined selection steps (see B-2, B-3).**

Verified correct: 5 outer video-stratified folds seed `20260807`; 4 inner folds seed `20260808`;
`StratifiedKFold(shuffle=True, random_state=seed)` over the **sorted** id list (which correctly
makes fold membership independent of cache row order); all 30 segments travel with their video
(asserted); preprocessing is `"none"` so §7 step 1 is satisfied vacuously and no fold-fitted
object exists to leak; threshold comes **only** from pooled inner-OOF (§3.3), is carried
unchanged to the outer-query fold, and every row records `threshold_source`; refit **once** on
full outer-train at the selected fixed `epoch*` with `inner = -1` seed; scores emitted **once**
for outer-query; the primary metric concatenates all outer-query predictions rather than
averaging fold metrics; the four overlap assertions (one-query-fold-per-video, video/segment/
derived-id disjointness, inner nested in outer-train) are mandatory manifest fields that must all
be `true`. `HALT_FOLD_INFEASIBLE` reproduces prereg §7's measurement-failure clause.

The threshold candidate set of §3.3 is complete: midpoints of consecutive unique scores plus
`u_min − 1e-6` and `u_max + 1e-6` under `ŷ = 1 iff s ≥ θ` enumerates every distinct partition,
including all-positive and all-negative. The tie rule (`smallest |θ − 0.5|`, then smallest `θ`)
matches prereg §8.1 word for word.

### 1.4 Within-video second-level AUROC vs prereg §5.2 item 5 and §8.2

**PASS.**

- Midpoint rule: second `t` positive iff `t + 0.5` lies in a gold span (§3.2) — verbatim prereg
  §8.2. Second→window map `w(t) = min(29, floor((t+0.5)·30/D_v))` is consistent with the
  half-open window rule `[kD/K, (k+1)D/K)` and the last window closed at `D`.
- Eligible video set frozen **once** from gold spans before any arm's temporal metric, written to
  `eligible_videos.json`, SHA256 in `manifest.json`, reused unchanged by every arm (§8.1) — this
  is exactly prereg §5.2 item 5's "frozen once from gold spans and shared across arms".
  The added conditions (`≥1 parsed span`, valid `D_v`) are necessary preconditions, not a
  narrowing of the registered set: a video with no span has no positive second and would be
  excluded by the both-classes condition anyway.
- A0 broadcast = AUROC exactly 0.5 per eligible video, so the registered comparator mean is 0.5
  and the prereg's "+0.03 over the A0 broadcast" becomes `≥ 0.53`. The appendix checks **both**
  `≥ 0.60` and `≥ 0.53` explicitly instead of assuming the first subsumes the second. Correct.
- Per-video AP is not averaged; pooled full second-level AP/AUROC is kept as secondary and
  explicitly labelled as containing between-video separability (prereg §8.2). Gold-span
  recall@{1,2,4} and the separation/std diagnostics are all present.

### 1.5 Gate-C weighting and bootstrap vs prereg §4.1

**PASS.**

- Prediction source is the **A0 whole-video baseline on HateMM-train only**, through the §7
  protocol with the same folds/seeds; val and test untouched (prereg §4.1). ✔
- Terciles are computed on the OOF positive-class score **within the FN population**, half-open,
  from `numpy.quantile(..., method="linear")` — deterministic. ✔
- `|FN| ≤ 120` → audit all, weights all 1.0. Otherwise 40/40/40 with deterministic deficit
  redistribution in ascending tercile index. ✔ (Redistribution is a departure from "stratified
  *equally*", but it is forced by finite tercile sizes, is fully deterministic, and the weight
  formula absorbs it exactly — see next bullet.)
- **Weight `w[v] = N_{tercile(v)} / n_{tercile(v)}`** = prereg §4.1's "its tercile's
  full-population count divided by its sampled count", verbatim, and **frozen at sampling time**.
  Coverage is the weighted ratio over `audit_FN` only. ✔
- **Bootstrap replays the frozen weights**: "within each tercile draw `n_t` audited items with
  replacement; reapply the **frozen** `w[v]` (not recomputed from the resample)". This is exactly
  prereg §4.1's "resamples within tercile and reapplies these frozen weights", and the
  "not recomputed" clause is the non-obvious part that the appendix gets right. ✔
- Controls (30 TP + 30 FP, 10 per tercile within each control population, drawn from the same
  `rngC` **after** the FN draw so consumption order is fixed) are **never** in the FN coverage
  denominator. ✔ Unweighted proportions reported separately, labelled `diagnostic`. ✔
- Mechanism presence uses **primary-or-secondary** for the union *and* for
  `multi_segment_complementary`, matching prereg §4.3's parenthetical exactly. ✔
- Blinding (prereg §4.1): the item list is shuffled and carries no category field; the FN/TP/FP
  mapping is kept in a separate file not given to annotators; score, correctness category,
  retrieval output and TERA output are all hidden; transcript is shown only because it is an
  ordinary model input. ✔ Hashing order §11.5 puts `annotation_protocol.json` and the blinded
  item list under SHA256 **before** any label is entered (prereg §4.2). ✔ 20% double-coding by
  seeded permutation, raw pre-adjudication labels retained, adjudication appended not rewritten,
  raw agreement + Cohen's kappa on `primary_cause`. ✔

### 1.6 Seeds vs the prereg-pinned values

**PASS.** Every prereg-pinned seed is reproduced exactly and used where the prereg puts it:

| purpose | prereg | appendix §7.6 + JSON `seeds` | verdict |
|---|---|---|---|
| outer folds | §7 `20260807` | `20260807` | ✔ |
| inner folds | §7 `20260808` | `20260808` | ✔ |
| video bootstrap | §8.3 `20260809` | `20260809` | ✔ |
| Gate-C sampling | §4.1 `20260807` | `20260807` | ✔ |
| Gate-C double-coding assignment | (unpinned) | `20260807` | ✔ admissible |
| B4 order permutation | §6 `20260807` | `20260807` (`rng4`) | ✔ |
| B5 donor draw | §6 `20260807` | `20260807`, **separate generator instance** | ✔ |
| Gate-C coverage bootstrap | (unpinned) | `20260809` | ✔ consistent with §8.3 |
| model init base | (unpinned) | `20260810`, SHA256-derived per scope | ✔ appendix authority |
| fixtures base | (unpinned) | `424242`, deliberately far from run seeds | ✔ |

The two `20260807` consumers (B4, B5) use **separate `default_rng` instances**, so B5's draw
cannot be perturbed by B4's consumption — this is the failure mode a shared generator would have
created, and the appendix closes it. Both are consumed in ascending sorted video-id order, making
the assignment independent of iteration order. Bootstrap indices are generated once over the
sorted OOF id list, saved to `bootstrap_indices.npz`, and **reused identically by every arm**,
which is what makes the deltas genuinely paired (prereg §8.3). Macro-F1 recomputed inside each
resample at the frozen threshold; percentile [2.5, 97.5]; no multiple-testing adjustment. ✔

### 1.7 Test seal and the HateClipSeg `test_*` whitelist

**PASS on the whitelist logic; BLOCKING-FIX on load-time id filtering (see B-4).**

The whitelist reasoning is **correct and verified**. HateClipSeg's *entire* surviving corpus was
extracted under `--splits test`, and `SPLIT_TO_OUTNAME` in
`src/utils/generate_subclip_embedding_HF.py` maps `test → test_seen`; so
`data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_*.pt` and
`data/gt/HateClipSeg/test.jsonl` hold all 395 videos, not a test split. I confirmed
`data/gt/HateClipSeg/p11_split.json` has `train: 237, val: 39, test: 119` (= 395), so
HateClipSeg's sealed split is indeed defined **inside** `p11_split.json` and not by filename.
Whitelisting those three paths **by exact path** (never by glob) is right, and the whitelist is
inside the hashed payload so it cannot be widened silently.

HateMM's seal is genuine: `data/gt/HateMM/test.jsonl` and
`data/CLIP_Embedding/HateMM/test_seen_*` are forbidden, `test_contact_count` must be 0 and
`opened_test_paths` must be `[]`, and F14 exercises the guard.

The residual hole is B-4 below: the guard is path-based, but the two *gold* files needed by
Gate-C/O1 and by the HateClipSeg endpoint span the **whole corpus including sealed ids**, so a
path guard alone does not seal them.

### 1.8 B5 support matching vs prereg §6 / §11.1

**PASS.**

- `Dpred` is **D's frozen binary prediction** at D's frozen inner-OOF threshold, never the
  query's true label; the appendix additionally asserts the routine's closure holds no reference
  to the label array. ✔ prereg §6 "whose **D-predicted** video label matches the query's
  D-predicted label ... never the query's true label".
- `legal_support(v, f) = sorted(set(outer_train_ids[f]) − {v})` for **both** training rows and
  held-out query rows. No draw crosses an outer fold; a query video never donates to another
  query video. ✔ prereg §6 "replacements never cross into the held-out outer fold" and §11.1
  "train and held-out-query predictions must each use replacements drawn only from their legally
  available support partition".
- The donor's contribution is the donor's **own `e_second`** under D's top-two selection — a
  legitimately selected second evidence unit, not a random window. ✔
- Empty stratum → uniform draw from the full legal pool, `b5_fallback: true` per row plus an
  aggregate `b5_fallback_count`. ✔ prereg §6's exact fallback clause. F6 exercises it.
- Replacing only the second slot's **content** while retaining the query's own `φ` is the correct
  disentanglement: it keeps B5 (identity of the second unit) orthogonal to B4 (temporal
  relation). Without the split the two lesions would be confounded and the two `+0.015`
  criteria would not be testing different things. Good call.

### 1.9 Weak/no-span supervision boundary (prereg §3)

**PASS for Gate-A/Gate-B deployable arms.** Gold spans enter only `o1_video_logit` and the
temporal-evaluation routines, both marked `oracle_or_eval_only: true` with a sidecar marker on
any file containing a non-null gold-derived field. No gold-span-trained arm exists.
`retrieval_memory: "none"` for every arm, with the single registered exception of B5's donor draw
(a lesion, not a retrieval component) — the run asserts no arm builds an index over other videos,
which also satisfies prereg §3's "if a retrieval memory is used, the held-out video's segments
and all derivatives are absent from that memory" vacuously. Gate-B's top-two selection is a pure
function of D, which never saw a span or a segment label. ✔

One deliberate, correct tension is worth recording: **O1 and O2 receive their own thresholds from
their own pooled inner-OOF scores**, i.e. gold-span-derived information reaches a threshold.
Prereg §3's sentence "They must not affect ... thresholds" sits in the paragraph governing
*deployable* arms and is immediately followed by "Gold spans may be read only by ... the
explicitly named non-deployable oracle/evaluation routines". Giving O1 a gold-blind threshold
(e.g. A1's) would make the oracle bound meaningless, since O1's scores live on a different scale.
The appendix's reading is the only one under which criteria 1 and 2 are informative. **PASS**,
recorded here so the reading is on the record before unblinding.

### 1.10 Verified factual assertions about the assets and source

I re-derived every asset-level claim the appendix makes. All of the following are **confirmed**:

- **Environment.** `nvidia-smi -L` → one `NVIDIA GeForce RTX 5090`; `sbatch` and `squeue` both
  absent from `PATH`; `/data/jehc223` does not exist. §0.2's "single-GPU exemption applies,
  `scheduler: none`" is factually correct, and §2.5's "raw media absent" is correct.
- **Cache presence and sizes.** `train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt` =
  **91,800,918 B** (matches the JSON's `bytes_observed_at_draft` exactly);
  `train_openai_clip-vit-large-patch14-336_HF.pt` = **5,359,881 B** (exact);
  `HateClipSeg/test_seen_subclipK30_...pt` = **48,739,122 B** (exact).
  `HateMM/dev_seen_subclipK30_...pt` and `HateClipSeg/test_seen_openai_...pt` are genuinely
  **ABSENT** (only `dev_seen_subclipK4_...` and `test_seen_subclipK4_...` exist). §2.2/§2.4 are
  accurate.
- **Segment-cache schema.** The extractor's own contract (lines 27–40) is
  `{video_ids, subclip_img_feats, subclip_parent, labels, num_subclips, num_frames}` —
  the appendix reproduces it exactly, including `subclip_parent` as a parent row index and the
  zero-vector guard for undecodable videos.
- **Text sharing.** Lines 22–25 state the sub-clip text stream is not re-extracted and sub-clips
  share the parent's video-level `text_feats`. The appendix's §2.3 claim is exactly right.
- **Frame grid.** `p11_hatemm_subclipK30.sbatch:36-37` pins `--num_subclips 30 --num_frames 120`.
  `_window_bounds(120, 30)` gives `base = 4, rem = 0`, so window `k` receives sampled frames
  `[4k, 4k+4)` = indices `4k … 4k+3` — exactly as claimed. I additionally verified
  `scripts/slurm/hateclipseg_subclip.sbatch` runs the loop `"4 16"` and `"30 120"`, so the
  **HateClipSeg K=30 cache also uses M=120** and the same alignment argument covers it.
- **Alignment inequality.** `_sample_frame_indices` is `round(linspace(0, N-1, M))`, so sampled
  frame `j` sits at normalized position `j/(M-1) = j/119`. Re-derived independently:
  left edge `4k/119 ≥ 4k/120 = k/30` for all `k ≥ 0`; right edge `(4k+3)/119 ≤ (k+1)/30`
  ⟺ `120k + 90 ≤ 119k + 119` ⟺ `k ≤ 29`. Both hold for all 30 windows. The appendix's proof
  is **correct**.
- **Visual/text readout.** `encode_frames_pooled` uses `out.pooler_output`; the zero-guard
  fallback uses `vision_model.config.hidden_size`, i.e. the **vision tower hidden size**, not the
  768-d joint projection. `generate_VideoCLIP_embedding_HF.py` takes `CLIPTextModel.pooler_output`
  chunked and mean-pooled. §2.1 is correct. (Independent size check: `744 × (1024+768) × 4 B`
  ≈ 5.33 MB against the observed 5.36 MB whole-video cache, and `V × 30 × 1024 × 4 B` against
  both K=30 caches, are consistent with `Dv = 1024, Dt = 768`. These remain
  `TO-FILL-AT-ASSET-AUDIT` as they should be.)
- **Init convention.** `scripts/analysis/p11_probe_hatemm.py:181-183` is
  `nn.Linear` → `zeros_(bias)` → `normal_(weight, std=0.01)`, and `:186-195` is the
  `topk(...).values.mean(1)` MIL proxy. §4's stated convention and §4.3's rationale (ii) are
  accurate. (The probe uses `Adam`, not `AdamW`; the appendix only claims init parity, so this
  is not a misstatement.)
- **Canonical JSON.** `scripts/analysis/edcm_a0.py:47-58` is exactly the
  `ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False` convention §10.2
  cites. ✔
- **Split files.** `data/gt/HateMM/{train,val,test}.jsonl` all exist (dated 2026-07-01), so §2.7's
  "the ongoing B2 restore has since materialized them" is correct and branch 1 of the resolution
  order will be the live branch. `hate_spans.json` holds **1083** records with fields
  `duration / spans / label` (plus `clipped` and `anomaly` on 2 records each);
  `gold_segments.json` holds **395**; `p11_split.json` is `237 / 39 / 119`.
- **Arithmetic.** Independently recomputed: `params(P) = 1792·128 + 128 = 229,504`;
  `params(B2) = 229,504 + (390·64 + 64) + 65 = 254,593`; `130·H3 + 229,505 = 254,593`
  → `H3 = 192.98` → `H3* = 193` → `params(B3) = 254,595`, `|Δ|/params(B2) = 7.855e-6` ✔ ("7.9e-6");
  `params(B0) = params(B1) = 237,825`, which is `6.586%` below B2 ✔ ("6.6%"). A3:
  `128·1792 + 128 + 128 + 1792 + 1 = 231,425` ✔. B2 input dim `3r + 6 = 390` ✔.
  Budget: `5·(6·4+1)·3 = 375`, `5·(18·4+1)·2 = 730`, total `1105`; B-stage `5·25·6 = 750` ✔.

Factual errors found are listed as **F-1 … F-4** in §4 below. None of them can produce a wrong
verdict.

---

## 2. BLOCKING-FIX items (must be resolved before the freeze)

All five are one-to-three-sentence registrations. All are free right now because nothing has been
computed; all become material deviations under prereg §12 once a candidate metric exists.

---

### **B-1 — The val-confirmation protocol is entirely undefined, and two of the confirmations are binding criteria.** *(class (a) + (b))*

Prereg §5.2 item 6 and §6 bullet 4 are **binding pass conditions**: "on the one-time HateMM-val
confirmation, item 3 remains positive; and on HateClipSeg-val the matched `has-hateful-segment`
delta is positive" and "the B2 delta remains positive on HateMM-val and HateClipSeg-val". The
appendix specifies the caches these need (§2.4) and the split-source rule for `val.jsonl` (§2.7),
but **never specifies how a val number is produced**. A grep of the appendix for `val` returns
only asset-gap and split-source lines. Specifically undefined:

1. **Which model scores HateMM-val** — the five outer-fold models (averaged scores? averaged
   logits? majority vote?), or a single new refit on the whole of HateMM-train.
2. **Which threshold is applied on val** — the five folds' `θ*` (which one?), a pooled
   inner-OOF `θ*` from a fresh full-train inner split, or something else. Re-deriving `θ` on val
   would be threshold selection on a confirmation set and is forbidden by prereg §8.1 and §2.1
   ("it may not select an arm, threshold, epoch, granularity, or decision rule"), but the
   appendix never says so.
3. **How the HateClipSeg confirmation is fitted at all.** Prereg §2.2 makes HateClipSeg-**train**
   the development partition, so the arms must be fitted on it — but with which protocol (the
   same 5×4 nested loop? a single fit? which hyperparameters — re-selected on HateClipSeg-train
   inner-OOF, or transferred from the HateMM selection?). "Transfer the HateMM model" and "train
   on HateClipSeg-train" are both defensible readings that give different numbers.
4. **The supervision status of the HateClipSeg binding endpoint.** The endpoint
   (`has ≥ 1 segment labelled hateful`) is *derived from gold segment labels*. Prereg §3 allows
   "video-level binary label" and forbids "gold segment labels" as supervision. Using the derived
   video-level binary as a **training target** is admissible under that reading, but it is a
   weak-supervision-boundary call that must be stated, not inferred — and it must be stated that
   no per-segment gold label reaches any deployable path.

Because these are binding criteria, leaving the procedure open lets the implementer choose a
reading after seeing the outer-OOF result. That breaks the "decision rules frozen before results"
red line and can flip a Gate-A or Gate-B verdict.

**Minimal fix.** Add one subsection (e.g. §7.10 "Confirmation-set protocol, registered") fixing,
for both HateMM-val and HateClipSeg-val: (i) the scoring model — recommended: **the mean of the
five outer-fold models' scores**, or equivalently a single refit on all of HateMM-train at the
modal `(cfg*, epoch*)`; pick one and write it down; (ii) the threshold — **the median (or the
fold-0) frozen `θ*` from the outer folds; never re-derived on val**, with an explicit statement
that val may not select anything; (iii) for HateClipSeg, the complete fitting procedure on
HateClipSeg-**train** (arms fitted, hyperparameter source, threshold source) run **once**;
(iv) an explicit sentence that the HateClipSeg binding endpoint is consumed **only** as a
video-level binary target/label and that per-segment gold labels never enter a deployable path.
Nothing in this fix touches an arm, endpoint, threshold value or decision rule.

---

### **B-2 — "Pooled inner-OOF macro-F1" is not defined across the five outer folds, and it is the sole selector of `D`.** *(class (a))*

§5.3 registers `D = argmax_{arm ∈ {A2,A3,A4}} (pooled inner-OOF macro-F1)`. §7.2 defines
"pooled inner-OOF macro-F1" only **within one outer fold** (pool the 4 inner-held-out prediction
sets of that fold). It never says how the five outer folds' inner-OOF results are combined into
the single per-arm number that `argmax` ranges over. At least three readings are available:
concatenate all five folds' inner-OOF predictions (each at its own fold's `θ*`) and compute one
macro-F1; average the five folds' inner-OOF macro-F1 values; or use only the pooled inner-OOF of
one nominated fold.

`D` determines Gate-A criterion 3, criterion 5, and the entire basis of Gate-B. An undefined
aggregation lets `D` be chosen after the per-fold numbers are visible. This is a wrong-verdict
defect regardless of which reading is eventually correct.

Note the same gap appears one level down in §6.1: "train videos: seg scores from D's **inner-OOF**
predictions within outer-train(f)" does not say which config's inner-OOF models (D's `cfg*` for
that outer fold, presumably).

**Minimal fix.** In §5.3 add: *"'Pooled inner-OOF macro-F1' for arm selection is computed by
concatenating the inner-held-out predictions of all 5 outer folds, each fold contributing its own
selected `(cfg*, epoch*)` at its own `θ*`, and computing one macro-F1 on that concatenation
(never a mean of fold metrics — same convention as prereg §7's primary metric)."* In §6.1 add:
*"D's inner-OOF segment scores for outer-train videos come from D's inner-fold models at that
outer fold's selected `cfg*` and `epoch*`."*

---

### **B-3 — §7.2 and §7.4 give two conflicting epoch-selection rules.** *(class (a))*

§7.2's pseudocode says: *"apply early-stopping rule (§7.4) → best (cfg, epoch, θ) candidate"* and
then *"select (cfg\*, epoch\*, θ\*) = **argmax over all cfg/epoch**"*. §7.4 defines a
patience-truncated best (`patience = 40`, `min_delta = 1e-4`, break on stall). These are different
selection rules whenever the inner-OOF macro-F1 curve has a later peak after a 40-epoch plateau —
`argmax over all epochs` would take it, §7.4's rule would not. §7.4 asserts early stopping "must
not change the selection", which is exactly what it *can* do.

Epoch is a selected hyperparameter and part of the frozen selection rule; leaving two readings
open lets the implementer pick after seeing curves.

**Minimal fix.** In §7.2 replace *"select (cfg\*, epoch\*, θ\*) = argmax over all cfg/epoch"*
with *"select `(cfg*, epoch*, θ*)` = argmax over the per-cfg candidates `(best_epoch, best)`
produced by §7.4, ties by §3.5"*, and delete the "must not change the selection" sentence from
§7.4 (it is false; the truncation *is* part of the registered rule, applied identically to every
arm/config/fold, which is what matters).

---

### **B-4 — The two gold files span the whole corpus, including sealed test ids; the path-based guard cannot seal them.** *(class (b))*

Verified: `data/gt/HateMM/hate_spans.json` contains **1083** records — the full HateMM corpus,
i.e. train **+ val + the sealed official test set** — and every record carries a `label` field
alongside `duration` and `spans`. Prereg §2.3 is stricter than a path guard: *"TERA design,
implementation, debugging, arm selection, and thresholding must not load any official test
labels, predictions, spans, or per-example artifacts."* Loading `hate_spans.json` wholesale loads
the sealed test set's labels **and** spans into the process. The same applies on the HateClipSeg
side: `gold_segments.json` (395), `video_durations.jsonl`, `test.jsonl` and the whitelisted K=30
cache all contain the 119 `p11_split["test"]` videos.

The appendix's mitigation ("the P11 test-split IDs are additionally filtered out in code with an
assertion that no P11-test id enters any training, selection, or thresholding path") is a
downstream promise, not a seal, and it says nothing at all about HateMM's sealed test ids inside
`hate_spans.json` — those are not covered by any registered filter because `hate_spans.json` is
correctly *not* on `forbidden_paths` (Gate-C and O1 need it).

This is not hypothetical bookkeeping: the whole point of prereg §2.3's prospective rule and of
`test_contact_count` is that sealed per-example artifacts never enter the run's address space.

**Minimal fix.** Register a **load-time id restriction** as the only admissible way to open a
corpus-spanning gold file:

> Every corpus-spanning gold artifact (`data/gt/HateMM/hate_spans.json`,
> `data/gt/HateClipSeg/gold_segments.json`, `data/gt/HateClipSeg/video_durations.jsonl`,
> `data/gt/HateClipSeg/test.jsonl`, and the whitelisted HateClipSeg feature caches) is loaded
> through a single registered reader that **immediately** restricts the object to the currently
> authorized id set (HateMM-train for development; HateMM-train ∪ HateMM-val at confirmation
> time; `p11_split["train"]` / `["train"] ∪ ["val"]` for HateClipSeg) and discards all other
> entries **before returning**. The reader asserts that no HateMM `test.jsonl` id and no
> `p11_split["test"]` id survives the restriction, and records
> `sealed_ids_dropped{hatemm, hateclipseg}` and `authorized_id_hash` in `manifest.json`.
> Any code path holding an unrestricted handle to these files is a HALT.

Add `F15` to the §9 battery: a synthetic corpus-spanning gold file containing sealed ids must come
back id-restricted with a non-zero `sealed_ids_dropped` count and zero sealed ids reachable.

---

### **B-5 — The Gate-B false-positive side-condition is undefined on the `multi_segment_complementary` subset.** *(class (a))*

Prereg §6 bullet 3: *"B2 rescues at least 20% of B0 false negatives **without increasing false
positives on that subset by more than 10%**."* The appendix (§6.7 and the JSON's
`gate_b.decision_inputs.rescue`) defines the rescue numerator/denominator correctly but leaves
the FP term undefined in two ways:

1. **Membership.** Gate-C codes FNs *and* 30 TP + 30 FP controls. Is "the frozen Gate-C
   `multi_segment_complementary` subset" the msc-flagged audited **FNs only**, or msc-flagged
   audited videos of **all** categories? If FNs only, the subset is entirely label-1 and the FP
   condition is structurally vacuous (0 → 0). If controls are included, it is a real and much
   stricter test. Both readings remain available after results are seen.
2. **The 0/0 case.** Expressed as a ratio, "increase by more than 10%" on a subset with zero
   baseline false positives is `0/0`. Undefined, and in code a `nan` that compares `False` under
   every comparison — i.e. an accidental auto-pass or an accidental crash depending on
   formulation.

**Minimal fix.** In §6.7 register, in one sentence: *"The frozen Gate-C
`multi_segment_complementary` subset is the set of **audited videos of any category** (FNs and
controls) carrying `multi_segment_complementary` as primary or secondary cause, per prereg §4.3's
presence rule. Rescue rate is computed over the label-1 members
(`|{B0 FN ∧ B2 TP}| / |{B0 FN}|`, with exact numerator/denominator and a Wilson interval);
the false-positive side condition is computed over the label-0 members as
`FP_{B2} ≤ FP_{B0} + max(1, ceil(0.10 · FP_{B0}))` in counts, and if the subset contains no
label-0 member the condition is recorded as `not_evaluable` and treated as satisfied."*
(Any single explicit convention is acceptable; what is not acceptable is leaving it open.)

---

## 3. Open-point adjudications

| OP | topic | ruling | basis |
|---|---|---|---|
| **OP-1** | B4 lesion semantics | **RATIFIED** — keep the per-video Bernoulli(0.5) swap | see below |
| **OP-2** | text stream in the feature family | **RATIFIED** — keep `concat(l2n(img), l2n(text))` at both levels | see below |
| **OP-3** | Gate-A capacity asymmetry | **CONFIRMED as a recorded limitation** — do **not** change the design | see below |
| **OP-4** | missing caches (HateMM-val K=30, HateClipSeg whole-video) | **RATIFIED** — the registered `HALT_MISSING_ASSET` is correct; not a reviewer decision, a user resource decision | prereg §12 makes a missing asset a HALT, never a performance negative; substituting anything would be a §12 material deviation |
| **OP-5** | raw video for Gate-C | **RATIFIED** — Gate-C cannot run here; no in-repo substitute is admissible; user resource decision | prereg §4.1 requires annotators to watch the video; transcript-only or MLLM-described auditing changes the registered protocol (§12) |
| **OP-6** | lr × wd grid, `E_max`, patience | **RATIFIED as registered**, with a free non-blocking recommendation | see below |
| **OP-7** | `r = 128`, `H = 64`, `H_att = 128` → `H3 = 193` | **RATIFIED** — arithmetic independently verified | see below |

### OP-1 — B4: is the Bernoulli(0.5) reading compatible with prereg §6 and §11.1? **Yes. RATIFIED.**

The two prereg clauses genuinely pull apart for a 2-element permutation, and the appendix is right
to flag it rather than absorb it silently.

- Prereg **§6** says the order is "**randomly** permuted within video". A deterministic
  always-swap is not random; the Bernoulli(0.5) per-video draw is the literal reading of §6.
- Prereg **§11.1** says the swap must be "a genuine lesion (swap the selected pair's order/
  relative-time encoding while retaining both segment contents), not an arbitrary permutation
  that sometimes leaves order unchanged."

The author's three assertions are **sufficient**, and more importantly the author's scientific
argument is **correct**, which is what decides it:

Under an unconditional swap, B4's presented order is deterministically the reverse of the true
order (`iA = b, iB = a` always). The mapping is a bijection on the input space, so a B4-trained
model simply relearns the reversed convention: `φ_B4 = [t_b, t_a, −δ, |δ|, −sin πδ, cos πδ]`
carries exactly the same information as `φ_B2`, merely re-encoded. B2 − B4 would then be ≈ 0 by
construction, the criterion `B2 − B4 ≥ +0.015` would fail almost surely **regardless of whether
temporal order matters**, and Gate-B would return a near-certain false `NO-GO-B`. That is a
wrong-verdict-producing design. Always-swap does not lesion the information; it relabels it.

The Bernoulli(0.5) draw, fixed per video and reused identically in training and evaluation,
destroys the **dataset-level correspondence** between presented slot order and true temporal
order — which is precisely the quantity H-B is about — while retaining both segments' contents
exactly as §11.1 demands. §11.1's warning is best read as guarding against *degenerate*
permutations (a single global draw, a per-epoch resample the model averages out, or a swap
probability so small the lesion is toothless). The appendix's three assertions close exactly
those: (1) realized swap fraction in `[0.45, 0.55]` proves the coin is fair and roughly half the
dataset is genuinely swapped; (2) `sign(δ_B4) = −sign(δ_B2)` and `in_B4 ≠ in_B2` on every swapped
video proves the swap propagates into `φ` and into the model input rather than being cosmetically
absorbed; (3) the no-downstream-re-sort assertion (`iA > iB` on swapped videos) closes the one
silent-cancellation path. F12 verifies all three on F4's data, and F4 itself is constructed so
that a real ordered-pair signal exists — so if B4 failed to bite, F4 would fail before the freeze.

Note also that the appendix's F12 assertion "unswapped videos have `in_B4 == in_B2` bitwise" is
not a defect but the correct verification that the lesion is applied *selectively and exactly*.

The alternative the appendix names (zeroing `φ`'s order-carrying components) is correctly
identified as a **different lesion**, not a permutation, and therefore a §12 material deviation
requiring a re-freeze. I do not request it.

**Ratified.** Record in the appendix §12 that OP-1 was adjudicated by this review, before any
candidate metric, so the record shows the reading was frozen ex ante.

### OP-2 — text stream: reasonable default, violates no frozen clause? **Yes, and no. RATIFIED.**

- **No frozen clause is touched.** Prereg §3 lists "raw input modalities" as allowed supervision;
  §5.1 requires only that all arms share "the same frozen segment encoder/features"; §12 makes
  the feature family material *once frozen*, and this appendix **is** the freeze. Nothing in the
  prereg pins the modality combination, so the choice is squarely inside §11.1's grant.
- **The two levels are isomorphic**, which is the load-bearing requirement:
  `s_{v,k} = concat(l2n(subclip_img_feats[v,k]), l2n(text_feats[v]))` and
  `x_v = concat(l2n(img_feats[v]), l2n(text_feats[v]))`. Same `d = Dv + Dt`, same per-stream
  normalizer, same concatenation order. Every arm consumes the same family. ✔ This is confirmed
  correct against the extractor: `generate_subclip_embedding_HF.py:22-25` states sub-clips share
  the parent's video-level text embedding by construction, so there is no per-segment text to
  use even in principle.
- **The constant-across-`k` consequence is benign and correctly disclosed.** Because
  `text_feats[v]` does not vary with `k`, for the linear-head arms the text contribution
  `w_text · l2n(text_v)` is a per-video additive constant on every `z_{v,k}`. I checked each arm:
  A2's top-k is rank-invariant to a per-video constant, so `k` selection is unaffected and the
  constant passes straight through to the video logit; A4's
  `τ(logsumexp((a+c)/τ) − log K) = c + τ(logsumexp(a/τ) − log K)` likewise passes it through
  exactly; A3's attention sees it inside a `tanh` so it modulates rather than shifts, which is
  fine. Crucially, **the within-video second-level AUROC and gold-span recall are unaffected**,
  because a per-video additive constant cannot change a within-video ranking. So including text
  cannot inflate the temporal criterion (§5.2 item 5) — it only strengthens the video-level
  baselines, which is the conservative direction for criteria 1–3.
- **Excluding text would be the worse choice.** It would make A0 weaker than the repository's own
  standard HateMM whole-video baseline (prereg §2.3 forbids historical headline numbers as
  comparators, but a self-inflicted weak baseline still inflates every delta), and it would make
  the Gate-C `cross_modal` category untestable by any Gate-A arm — a coherence failure between C
  and A.

**Ratified**, with the disclosure sentence in §2.3 decision 1 kept verbatim so the "no temporal
text capability" caveat is on the record, and with NOTE N-2 below (the A0/A1 frame-budget
asymmetry) added to the same section.

### OP-3 — A3's 231k vs ~1.8k parameters: limitation or design change? **Limitation. CONFIRMED.**

The appendix's handling is correct and must not be "repaired". A Gate-A capacity control would be
a **new arm**, which prereg §11.1 forbids the appendix from adding and §12 classes as a material
deviation. The asymmetry is inherent to the registered arm A3 ("learned attention pooling"): any
attention pooler over `d = 1792` costs `O(H·d)` parameters, and shrinking `H_att` to chase
parameter parity would be unregistered, self-serving tuning of the primary recoverability arm
(and would not even reach parity — `H_att = 1` still gives ~3.6k).

Two things make this limitation **bounded rather than fatal**, and both should be written into
the appendix's §4.4 limitation paragraph:

1. Prereg §5.2 **criterion 5 is itself the capacity/selection discriminator**. A pure capacity
   effect improves video-level macro-F1 without localizing; criterion 5 requires D's mean
   within-video second-level AUROC ≥ 0.60 and ≥ A0-broadcast + 0.03. A capacity-only pass on
   criterion 3 that is not accompanied by a criterion-5 pass cannot promote the route, because
   Gate-A requires **all six** criteria.
2. `D` is chosen among A2/A3/A4 by inner-OOF macro-F1, and A2/A4 are ~1.8k-parameter arms. If D
   turns out to be A2 or A4, OP-3 is moot for the promoted arm entirely.

**Confirmed as a recorded limitation.** Required wording addition (documentary, non-blocking):
state in §4.4 that criterion 5 bounds the capacity confound, and that any Gate-A pass with
`D = A3` must carry the sentence "the A3 advantage over A0/A1 is not capacity-controlled at
Gate-A" in `verdict.json`.

### OP-6 — grid width: is it skewed enough to produce a wrong verdict? **No. RATIFIED, with a free recommendation.**

Checked for the three ways a grid can corrupt a verdict:

1. **Skew favouring one arm.** The `lr ∈ {3e-3, 1e-3, 3e-4} × wd ∈ {1e-4, 1e-2}` grid, `E_max`,
   `patience`, `min_delta`, `batch_size`, loss, dtype, device and loop structure are **identical
   for every arm**, including A0 and A1. No arm gets a private grid point.
2. **Budget inequality.** A2 and A4 evaluate 18 configs while A0/A1/A3 evaluate 6. This is
   **mandated by the prereg itself** (§5.1 registers `k ∈ {1,2,4}` and `τ ∈ {0.1,0.3,1.0}` as
   inner-OOF-selected), so the appendix cannot equalize it without deleting a registered
   hyperparameter. The optimistic-selection effect is contained by proper nesting: the extra
   configs are selected on inner-OOF and *evaluated* on outer-OOF. The appendix's
   `budget_report` in `metrics.json` makes the asymmetry visible rather than hidden, which is the
   right remedy. Note it runs *against* the promoted arm's comparator only if D is A2/A4, and
   criterion 3's comparator `max(A0,A1)` is the *larger* of two baselines, which partly offsets.
3. **Under-fitting the primary recoverability arm.** The narrowest genuine risk: a 1-decade lr
   span could under-fit A3's 231k-parameter attention MLP while fully fitting the 1.8k linear
   arms, biasing toward `NO-GO-A-SELECTOR`. This is a **conservative-direction** error and lands
   inside prereg §10's explicit claim boundary ("a negative result falsifies only the registered
   granularity, frozen representation, weak/no-span learners, data, and thresholds"). Under the
   review standard it does not block. Against a full-batch-ish regime (≈12 steps/epoch × up to
   200 epochs ≈ 2400 AdamW steps) on L2-normalized CLIP features, `3e-4 … 3e-3` is a sane span
   for both head classes.

**Ratified as registered.** *Free non-blocking recommendation, worth taking because it costs
nothing now and nothing later:* add `1e-2` as a fourth lr point. That makes the A-stage
`5·(8·4+1)·3 + 5·(24·4+1)·2 = 495 + 970 = 1465` head trainings instead of 1105 — a few extra
CPU-minutes — and permanently removes the "you under-fit A3" objection from any future
`NO-GO-A-SELECTOR` writeup. If the author declines, then a `NO-GO-A-SELECTOR` verdict must state
the grid span as an explicit scope limit in `verdict.json`.

### OP-7 — `r = 128`, `H = 64`, `H_att = 128`, `H3 = 193`: any wrong-verdict risk? **No. RATIFIED.**

Arithmetic independently reproduced (see §1.10): `H3* = 193`, `params(B3) = 254,595` vs
`params(B2) = 254,593`, relative difference `7.86e-6` — three orders of magnitude inside prereg
§6's 5% requirement, and `H3` is recomputed at run time from the observed `d` rather than
hard-coded, with the 5% assertion as a HALT. F10 checks it at both `d_fix` and `d = 1792`.

One observation that I checked and that turns out **not** to be a problem: `params(P) = 229,504`
is ~90% of every B arm's parameter count, so one might worry the "within 5%" match is trivially
satisfied by the shared projection rather than by matched discriminative capacity. It is not:
the *head* capacities are matched too — B2's `390→64→1` is 25,024 parameters against B3's
`128→193→1` at 25,090 (0.26% apart). B3 is a genuine capacity control, and B0/B1 at 237,825
(6.59% below B2) are correctly identified as *not* capacity-matched, which is exactly why
prereg §6 requires B3 separately. The construction is sound.

`H_att = 128` is a standard non-gated ABMIL width and is the direct cause of OP-3's asymmetry;
as argued under OP-3, tuning it downward to chase parameter parity would be worse than recording
the limitation. `r = 128` and `H = 64` are unconstrained by the prereg and uniform across all six
B arms. No arm gets a width advantage. **Ratified.**

---

## 4. Factual corrections (source-verified; none blocks)

- **F-1 — Whole-video cache schema is wrong in §2.2 (and consequentially in §2.7 step 1).**
  The appendix states `{"ids": [V] str, "img_feats": [V, Dv], "text_feats": [V, Dt], "labels": [V] long}`.
  The extractor's actual contract (`src/utils/generate_VideoCLIP_embedding_HF.py`, save block) is
  commented `# CONTRACT: ids is a list containing ONE sublist of all string ids.` and writes
  `"ids": [ids]` — a **nested** list-of-one-list. The repository-wide read convention is
  `d["ids"][0]` (`c3_nontarget_probe.py:73`, `apx_g0cond_gate.py:64,77`, `s2s_probe.py:95,102`,
  `eval_localization_ours.py:82`, `clap_cache_verify.py:55,68`, `w2b_probe.py:128`, and
  `errpat_hatemm_forensics.py:203` which defensively handles both). As written, §2.7 step 1's
  `set(ids) == set(cache["ids"])` raises `TypeError: unhashable type: 'list'`.
  **Fix:** write the schema as `{"ids": [[V] str], ...}` and dereference `cache["ids"][0]`.
  *Non-blocking: a clean run crashes immediately; it cannot mis-verdict.*
- **F-2 — `num_frames_expected: 8` on `hatemm_train_wholevideo` is not verifiable at asset audit.**
  The whole-video cache stores only `{ids, img_feats, text_feats, labels}` — there is no
  `num_frames` key (unlike the segment cache, which does store `num_subclips` and `num_frames`).
  The asset audit cannot assert this field against the artifact.
  **Fix:** mark it `provenance_only` (it is the extractor's default and is not recoverable from
  the file), or drop it. Do **not** silently assert it.
- **F-3 — Source citation `generate_subclip_embedding_HF.py:112-124` is wrong.** Lines 112–124
  are the tail of `parse_args_sys` and the head of `read_gt`. `_sample_frame_indices`
  (`np.round(np.linspace(0, num_total-1, num_frames))`) is at approximately lines **134–146**.
  The companion citation `:238-259` for `_window_bounds` is **correct**, as are `:22-25`,
  `:27-40`, `:316`, `p11_hatemm_subclipK30.sbatch:36-37`, `p11_probe_hatemm.py:181-183` and
  `:186-195`, and `edcm_a0.py:47-58`. **Fix:** correct the one line range.
- **F-4 — `hate_spans.json` schema line lists a field that does not occur.** The registered schema
  string includes `parse_error?: str`; the current file's 1083 records carry only
  `duration / spans / label`, plus `clipped` and `anomaly` on 2 records each. Harmless (the field
  is optional in the schema string), but record the observed field set at asset audit so the
  hash-time schema assertion does not encode a phantom key.

---

## 5. NOTEs (non-blocking; take the cheap ones now, none is a gate)

- **N-1 — Frame-time convention should be stated.** The §2.6 alignment proof normalizes sampled
  frame `j` to `j/119`, i.e. it treats frame index `m` as sitting at time `m/(N-1)`. Under the
  alternative frame-**midpoint** convention `(m + 0.5)/N`, the slack of window `k` is
  `(29 − k)/3570` of `D` — so for `k = 28` the last sampled frame could nominally spill into
  window 29 by ≤ 0.03% of the duration (≈ 8 ms on a 30 s clip, half a frame at 30 fps). Utterly
  immaterial to any criterion, but the appendix should say **which convention its inequality
  assumes** so the asset-audit re-assertion is unambiguous. The `k = 29` right edge is exact
  (`119/119 = 1.0`) and the last window is closed at `D`, so no violation occurs there.
- **N-2 — A0 and A1 do not share a frame budget; say so.** A0's `img_feats` come from the
  whole-video cache built at `--num_frames 8` (extractor default), while A1–A4 consume the
  segment cache built at `M = 120`. Prereg §5.1 asks for "the same frozen segment
  encoder/features"; the encoder is identical but the *sampling budget* is not, and A0 also
  L2-normalizes a plain 8-frame mean whereas A1 averages 30 per-window-normalized vectors. This
  cannot inflate any criterion, because criteria 1–3 all compare against `max(A0, A1)` and A1 is
  the 120-frame comparator — the max() absorbs it. But it should be disclosed in §2.3 alongside
  the OP-2 disclosure, and `num_frames` for each cache should be recorded in
  `feature_manifest.json`.
- **N-3 — A3's registered per-segment score is `head(s_{v,k})`, not the attention weight `α_k`.**
  I checked this closely because it feeds criterion 5 and Gate-B's pair selection. It is
  **defensible and I do not require a change**: because `p_v = Σ_k α_k s_{v,k}` and the head is
  affine, `logit_v = Σ_k α_k · head(s_{v,k})` exactly — so `head(s_{v,k})` *is* segment `k`'s own
  contribution to A3's video logit, and the singleton rule keeps §11.1's "same definition in
  training and evaluation" and cross-arm comparability. But `α_k` is A3's *selector*, and prereg
  §6 freezes D as "the segment-scoring/**selection** basis". **Recommended, register now:** keep
  the singleton rule as the binding score, and additionally register the α-based within-video
  second-level AUROC as a **pre-registered diagnostic** (it is already recorded as
  `attention_weights` in `segment_scores.jsonl`). It may never rescue a failed criterion 5, but
  if D = A3 fails criterion 5 while the α-based AUROC is high, the record will contain the
  evidence that the failure is metric-definitional rather than substantive.
- **N-4 — A2's singleton bag is ill-typed for `k ∈ {2,4}`.** `torch.topk(z, k=2)` on a
  one-element tensor raises. §3.1 already states the reduction ("`f` on a singleton reduces to the
  linear segment logit `z_{v,k}`"), so the implementation must special-case it rather than call
  the generic pooling function. Worth one explicit sentence so the fixture harness does not
  discover it.
- **N-5 — B4's swap-fraction HALT has no legal escape.** With `V ≈ 744` the realized fraction has
  SE ≈ 0.018, so `[0.45, 0.55]` is ±2.7σ and trips ~0.7% of the time. If it trips there is **no
  recovery**: the seed is prereg-pinned at `20260807` and reseeding would be a §12 material
  deviation. Recommended: widen to `[0.40, 0.60]` (still ±5.5σ against a broken/degenerate draw,
  which is what the assertion is actually for) or demote it to a recorded warning with the exact
  realized fraction in `metrics.json`. Free now; a dead end later.
- **N-6 — Decode-failure accounting covers only the segment cache.** `decode_failures` is defined
  as "all 30 segment vectors exactly zero". A video can be zero-guarded in the whole-video cache
  (extracted separately, on a different date) without being zero in the segment cache, or vice
  versa; A0 and A1–A4 would then not be evaluated on matched inputs, which prereg §5.1 requires.
  Recommended: define `zero_vector_videos` as the **union** across both caches, count the union
  in the >1% HALT rule, and register that such videos are **kept** (with their zero vectors) in
  every arm so the evaluated video set is identical across arms.
- **N-7 — The temporal-metric bootstrap needs its own index set.** `bootstrap_indices.npz` is
  `int32[10000, N]` over the full outer-OOF video list, but §8.2's within-video AUROC CI is a
  bootstrap over the (smaller) **eligible** video set. Register whether it is a separate draw from
  `default_rng(20260809)` over the sorted eligible list, or a filtering of the master indices.
  Non-blocking: criterion 5 is a point-estimate threshold; the CI is reported, not binding.
- **N-8 — Refit step count differs from the inner-fit step count.** `batch_size = 64`,
  `drop_last = false`, and the outer-train partition is ~33% larger than an inner-train partition,
  so `epoch*` epochs means more gradient steps at refit than during selection. This is standard
  nested practice and prereg §7 step 3 says "using the selected fixed epoch/budget", so it is
  compliant; it is applied identically to every arm. Recorded for the limitations section only.
- **N-9 — `_hash_note` and §10.2 describe the hash exclusion set differently.** The JSON note says
  keys beginning with `_` are excluded; §10.2 lists only `hash_algorithm, payload_sha256,
  schema_version, status`. Both are consistent with the operative rule ("the hash covers **only**
  `cfg["payload"]`"), so nothing is ambiguous in practice — align the wording at freeze.
- **N-10 — §7.4's "compute saver" framing does not match §7.2's loop order.** §7.2 runs all
  `E_max` epochs for every inner fold before pooling, so §7.4's `break` saves nothing; it is
  purely a selection rule over an already-computed curve. Fold this into the B-3 fix.
- **N-11 — Fixture battery is unusually strong; two additions requested.** F1–F14 cover both the
  pass path and the clean-failure path (F2, F5), both HALT paths (F7b, F14), determinism (F13),
  the exact threshold tie rules (F8), leakage integrity (F9), the Gate-C weighting arithmetic
  against an analytic value (F11), and B4 genuineness (F12) on data (F4) constructed so that a
  toothless lesion would be caught. This is well above what the proportional-ceremony rule
  requires. Requested additions: **F15** (sealed-id restriction, per B-4) and an assertion inside
  F4/F12 that the *unswapped* half of B4 is bitwise identical to B2 **and** that B4's realized
  swap set is identical between the training pass and the evaluation pass (the "fixed dataset,
  not stochastic augmentation" property is asserted in prose but not in a fixture).
- **N-12 — Gate-C's `multi_segment_complementary` subset may be small.** Prereg §4.3 requires
  msc ≥ 15% of FNs to pass C; with ≤120 audited FNs the msc subset could be ~18 videos, so Gate-B
  bullet 3's 20% rescue rate is a ratio over a very small denominator. The prereg sets no minimum
  and the appendix correctly reports exact numerator/denominator plus a Wilson interval, which is
  the right treatment. Recorded so the eventual writeup does not over-read it.
- **N-13 — Documentary.** §2.7's "At registration `data/gt/HateMM/{train,val,test}.jsonl` were
  absent" is now stale in the present tense — all three exist (dated 2026-07-01), so resolution
  branch 1 (`split_source = "gt_jsonl"` with exact-match-or-`HALT_SPLIT_MISMATCH`) is the live
  branch. Per project rule, fix on sight without a review round; no re-review.

---

## 6. Final determination

# APPROVE-WITH-FIXES

The appendix is a faithful and unusually careful instantiation of the pre-registration. It adds
no arm, changes no endpoint, threshold, split, or decision rule; it contains no candidate result;
its blinding statement holds against everything I could check; and every asset-level and
source-level assertion I verified is correct except the four minor factual items in §4. The
oracle rules, the nested-OOF protocol, the temporal-metric definition, the Gate-C weighting and
bootstrap, the seed register, and B5's support partition all match the prereg clause by clause.
OP-1's Bernoulli(0.5) reading is not merely acceptable, it is the reading that avoids a
guaranteed-false `NO-GO-B`.

**Freeze is authorized once, and only once, the five BLOCKING-FIX items below are registered.**
All five are one-to-three-sentence additions that pin down something already intended; none
requires re-deriving anything, none touches the registered science, and all are free today.
Per the project's proportional-ceremony rule this is the **only** review round: the author applies
the minimal fixes, re-runs the §9 fixture battery (plus F15) to a full pass, and proceeds directly
to the freeze procedure of the appendix header (resolve `TO-FILL-AT-ASSET-AUDIT`, compute the
canonical payload hash, embed `appendix_sha256`, then execute). **No re-review.** Author self-test
evidence — a full fixture-battery pass — is sufficient to release.

### Minimal must-fix set

| id | class | one-line summary |
|---|---|---|
| **B-1** | (a)+(b) | Register the complete HateMM-val / HateClipSeg-val confirmation protocol: scoring model, threshold source (never re-derived on val), the HateClipSeg fitting procedure, and the statement that the HateClipSeg binding endpoint is consumed only as a video-level target. |
| **B-2** | (a) | Define "pooled inner-OOF macro-F1" across the five outer folds — the sole selector of `D` — and name the config whose inner-OOF scores feed §6.1. |
| **B-3** | (a) | Resolve the §7.2 (`argmax over all cfg/epoch`) vs §7.4 (patience-truncated best) conflict in favour of one rule. |
| **B-4** | (b) | Load every corpus-spanning gold artifact through a reader that id-restricts to the authorized split **before returning**; assert zero surviving sealed ids; record `sealed_ids_dropped`. Add fixture F15. |
| **B-5** | (a) | Define the Gate-B `multi_segment_complementary` subset's membership and the 0/0 convention for the "false positives increase by ≤ 10%" side condition. |

### Reminder on scope, unchanged by this approval

An approved freeze does **not** authorize execution. Gate-C remains blocked on raw HateMM video
(OP-5) and two binding confirmations remain blocked on absent caches (OP-4); both are user
resource decisions, and per prereg §9 and §12 the correct behaviour without them is
`HALT_MISSING_ASSET`, never a substituted endpoint and never a performance negative.

---

*Reviewed against `EXP_tera_gate0_prereg.md` §§0–13. No candidate metric was computed during this
review. This document is the ex-ante record of the OP-1 through OP-7 adjudications.*
