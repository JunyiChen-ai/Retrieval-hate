# S2S Pre-Registration Review — set-to-set retrieval over frozen Qwen per-frame token sets (round-3 lead)

**Reviewer:** fresh zero-prior-context independent pre-registration reviewer (read-only; CPU
verification only; NO GPU / NO SLURM / NO commits except this deliverable).
**Date:** 2026-07-14.
**Under review:** `research-wiki/experiments/exp-s2s-r3.md` (prereg, `DRAFT-UNREVIEWED`) +
`refine-logs/S2S_PROBE_DESIGN.md` (executable spec, `DRAFT-UNREVIEWED`).
**Context read (to judge, not obey):** `research-wiki/ROUND3_NOVELTY_CANDIDATES_2026-07-14.md`,
`autoresearch/goal_mllm_plus3/state/directions_tried.json`,
`research-wiki/REFLECTION_mllm_integration_failures.md` §4, `refine-logs/B5_PREREG_REVIEW.md`,
`refine-logs/B3_PREREG_REVIEW.md`.
**Method:** every load-bearing code fact re-read directly from `src/` and from the installed
`transformers==4.49.0` Qwen2.5-VL modeling source; the banked cache shapes / counts / zero-guard rows /
L2-norm independently re-derived by `torch.load` on CPU. Non-isomorphism claims checked against the
22-route graveyard.

**VERDICT: APPROVED-WITH-AMENDMENTS.** Five BLOCKING amendments (A1 grouping-gate overclaim + missing
grouping control; A2 sim-magnitude vote confound; A3 near-duplicate leakage; A4 under-specified oracle
statistic; A5 hash-freeze + the extraction/probe scripts do not yet exist). Seven non-blocking. The
*design* is code-faithful, honestly scoped, escapes D1/D2 on a genuinely untouched component, and its
central §4 correction (the banked pooled cache is **not** a per-frame mean) is exactly right and
independently confirmed. The blocking items are all cheap and mechanical; none touches a number. A
**CONDITIONAL AUTHORIZATION for the DESIGN of the EXTRACTION + PROBE stages only** is granted in §5,
gated on the five amendments AND on the not-yet-written scripts being authored-to-spec,
independently code-reviewed, and hash-pinned before the single submit. Head-training is NOT authorized.

---

## 0. Checklist result summary

| # | Review item | Ruling |
|---|---|---|
| 1 | Extraction parity (byte-parity vs banked; under-specification that could drift past the gate) | **PASS w/ BLOCKING A1** |
| 2 | Probe validity (zero test-touch; arm completeness; set-size / video-length confound) | **PASS** (set-size confound defused; A3 blocking on near-dup) |
| 3 | Kill bars (raw +0.05 honestly priced; oracle-headroom; non-shoppable dataset rule; frame budgets) | **PASS w/ BLOCKING A4** (oracle statistic) |
| 4 | Statistics (bootstrap/perm as distributions; multiple-comparison discipline; deterministic seeds) | **PASS** (non-blk N1, N3) |
| 5 | Veto / scope (single-dataset; no OCR; no gold in-method; no ensembles; no API; storage) | **PASS** |
| 6 | Ceremony (hash-freeze; single-submit; JobHeldUser; raw-only; separate probe vs head-train gates) | **PASS w/ BLOCKING A5** |
| 7 | Novelty-paragraph honesty (mechanism transfer; D7 user ruling pending) | **PASS** |
| 8 | Missed false-PASS / false-KILL surfaces | A1 + **A2** + **A3** (blocking) + N2/N5 |

---

## 1. Load-bearing verification (independent re-derivation)

### 1.1 The banked pooled cache is NOT a per-frame mean — §4 is EXACT (and sharper than the brief)
`src/utils/generate_VideoMLLM_embedding_HF.py:290-303` (span `"prefix"`): `im_start_id` positions →
`end = positions[-1]` (the **last** `<|im_start|>`, i.e. the assistant header) → `pooled =
last_hidden[:end].mean(dim=0)` → `:321-322` `pooled.float()`, `F.normalize(p=2)`. So banked
`img_feats[v] = L2normalize( mean over tokens[0:end] )`, and `[0:end]` = system turn + `<|im_start|>user`
header + **all** vision-pad tokens + `IMG_INSTRUCTION` text + `<|im_end|>`. The designer's §4/§1 claim —
"NOT a mean over per-frame vectors; NOT even a pure visual pool; it folds in the non-vision text tokens"
— is **verbatim correct**. Note `video_pad_id` is computed (`:288`) but **unused** for the span
(`:320` `_ = video_pad_id`), confirming the banked pool never separated vision from non-vision.
**The brief's framing ("mean over the FULL last_hidden sequence") is imprecise**: it is `[0:end]`,
*excluding* the trailing assistant header; the designer correctly narrowed it. (Minor cite blemish N7:
prereg provenance says `:320-322`; the pooling is at `:303`.)

### 1.2 G-decomp is a mathematically SOUND identity — but see A1 for what it does NOT prove
`(Σ_t n_t·g_t + p_S)/end` with `g_t` = group mean, `n_t` = group token count, `p_S` = non-vision prefix
sum, `end` = span length is a true count-weighted-recombination identity for `mean[0:end]`, bit-exact
within a forward. The arithmetic (`S2S_PROBE_DESIGN.md:108-111`) is correct.

### 1.3 The temporal-major equal-partition grouping is CORRECT for this exact model+config (I verified it)
`transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`: `get_window_index` (`:466-505`) builds the
canonical token layout `index = arange(grid_t·llm_grid_h·llm_grid_w).reshape(grid_t, llm_grid_h,
llm_grid_w)` — (t,h,w) row-major. The vision encoder reorders tokens for window attention
(`:529-530 hidden_states[window_index]`) then **restores original order** at the end (`:561-562
reverse_indices = argsort(window_index); hidden_states[reverse_indices]`). So the merged vision tokens
emitted into the LLM sequence are temporal-major, each temporal group a **contiguous
`llm_grid_h·llm_grid_w` block**. With `temporal_patch_size=2` and 8 frames, `grid_t = T = 4`; each `g_t`
is a 2-frame-group spatial-token mean. The designer's "T=4, frame-group vectors, temporal-major"
(§2 `:71-76`) is **correct by construction**, and `T = video_grid_thw[0][0]`
(`S2S_PROBE_DESIGN.md:92`) reads it robustly. **This is the fact the prereg asserts but does not cite,
and — critically — does not gate (A1).**

### 1.4 Set size is FIXED (the reviewer's set-size / video-length confound does NOT bite)
`_sample_frame_indices` (`generate_VideoMLLM_embedding_HF.py:146-152`) always returns exactly
`num_frames=8` indices (`np.linspace` + round + clip, duplicates for short videos). So **every** video
yields `grid_t = 4` → **every** set has exactly T=4 frame-group vectors, independent of video length or
resolution (`max_pixels` varies `grid_h,grid_w` but never `grid_t`). MeanMaxSim therefore cannot confound
with frame count / video length — the primary is safe from that surface. (Zero-guard rows get an empty
set, handled identically in both arms, §8.) This is a genuine strength of the design; no amendment.

### 1.5 Banked cache provenance — independently re-derived EXACT (CPU `torch.load`)
| dataset | train | dev | test | zero-img rows (train) | L2-normed |
|---|---|---|---|---|---|
| HateMM | 744 | 107 | 215 | **1** | yes (‖img₀‖=1.0000) |
| MHC-EN | 549 | 80 | 161 | 0 | yes |
Dv=3584 confirmed. Memory sizes match §6.1: HateMM train∪val = **851**, MHC-EN = **629**. Every number
in prereg §1 provenance is exact. The G-recon anchor files exist and are consumable.

### 1.6 Vote code — the sim is a WEIGHT, not just a ranking key (load-bearing for A2)
`src/utils/metrics.py:268-270`: `retrieved_labels_map = (np.array(retrieved_labels)*2-1) *
retrieved_sims` — the neighbor's signed label is **multiplied by the arm's pairwise score**; `:284`
rank-weights and normalizes; `:300` cut = `sigmoid(vote) ≥ 0.5 ⇔ vote ≥ 0`; `:309` macro-F1. So the
vote is a **sim-magnitude-weighted** rank-weighted signed sum. Substituting a different-scaled score
(MeanMaxSim vs pooled cosine) changes the vote even at fixed retrieved neighbors → A2.

**Nothing in the prereg's code or numeric claims is wrong.** The blocking items are about gate
completeness, an uncontrolled confound, a leakage surface, an under-specified statistic, and ceremony —
not about any incorrect number.

---

## 2. Per-item rulings

### Item 1 — EXTRACTION PARITY · PASS w/ BLOCKING A1
The config-parity table (`S2S_PROBE_DESIGN.md:40-64`) pins every forward-affecting knob (model, class,
bf16, sdpa, device, processor `max_pixels=151200`, transformers 4.49.0, `num_frames=8`, verbatim
sampler / decode / messages / `IMG_INSTRUCTION` / chat template / span end) — byte-parity is correctly
scoped and G-recon (cos ≥ 0.9999, max-abs ≤ 1e-3 vs the banked `img_feats`, `:139-145`) is a genuine
parity anchor that catches wrong frames / preprocessing / model / span. **The gap is A1:** the frame-set
grouping and the vision/non-vision split — the actual scientific object — pass **both** declared gates
regardless of whether they are correct (see A1). Under-specification that could silently drift: the
`video_pad_id` identity and the temporal contiguity are asserted, not gated.

### Item 2 — PROBE VALIDITY · PASS (set-size confound defused; A3 blocking on near-dup)
- **Zero test-touch confirmed.** Stage E caches all splits but scores none (`exp-s2s-r3.md:311`); Stage P
  builds memory from train∪val only, LOO (`:199-201`, `S2S_PROBE_DESIGN.md:166`); gold used only for
  Fano + oracle ceiling. The ledger (§10) is honest. **N4 (non-blk):** add a fail-closed guard in the
  probe script asserting the `test_seen` cache is never loaded into the memory nor voted (defense-in-depth).
- **Arms.** POOLED (visual-isolated null), SET/MeanMaxSim (primary), SET-Chamfer (1 sensitivity),
  PIPELINE-ANCHOR pooled, WITH-TEXT, oracle ceiling, Fano, permutation null. The paired
  pool-then-match vs match-then-pool on **identical** g_t (§5) is the right seed/representation-noise
  canceller. A CLIP-pooled reference arm is **not** required — the paired design already cancels the
  encoder (optional only).
- **Set-size / video-length confound: DEFUSED** (§1.4). No amendment.
- **A3 (BLOCKING) near-duplicate leakage** — see §3.

### Item 3 — KILL BARS · PASS w/ BLOCKING A4
- **Raw bar** (§6.5 `:236-247`): HateMM Δacc ≥ +0.05 **AND** ΔmF1 ≥ +0.05 (AND-rule, not shoppable),
  bootstrap 5th-pct > 0, observed Δ > 95th-pct null. The 1.7× pessimism pricing (+0.03 goal → +0.05 raw)
  is explicitly derived from the P3 "probe passes, training flat" graveyard lesson (≥4× repeats) and is
  demanding but inside the cited +3–8pt band. Honest and non-gameable.
- **Dataset rule** (§6.6 `:255-266`): four rows (a/b/c/d) fixed before results; HateMM-primary /
  MHC-EN-binding-gap; explicit "no post-hoc dataset shopping"; the honest carry-over that a HateMM-only
  win does **not** advance the ≥2-dataset goal (HateMM already passes via encoder-swap). Correct.
- **Frame budgets:** exactly 2 (8→T=4 primary, 16→T=8 sensitivity, `:267-269`), the 16-frame arm gated
  by its **own** internal G-decomp (cannot G-recon against the 8-frame banked cache) — correct.
- **A4 (BLOCKING):** the oracle-ceiling statistic (whether ANY head GPU is spent) is under-specified —
  see §3.

### Item 4 — STATISTICS · PASS (non-blocking N1, N3)
Bootstrap ≥1000 and permutation ≥100 are correctly framed as **distributions** (report 5/50/95 pct;
observed Δ vs 95th-pct null). The permutation null (shuffle frame sets across videos, preserving marginal
frame-vector distribution) is a defensible test of set↔label structure.
- **N1 (non-blk):** pre-declare the null seed set (e.g. 0..99) and require the **same** permutation
  applied to both arms within a seed so the paired Δ is preserved. Optionally add a finer per-frame
  shuffle null to separate the "alignment" mechanism from a generic "richer-key" effect (recommended,
  not required).
- **N3 (non-blk):** state explicitly that the sensitivity arms (Chamfer, WITH-TEXT, 16-frame) **cannot
  rescue a failed primary**; the single pre-declared primary is MeanMaxSim visual-isolated paired Δ (acc
  AND F1) under the §6.6 dataset rule — no OR-ing across arms/metrics beyond the four fixed rows.
- **N2 (non-blk):** specify the exact Fano-arm "sim" value (e.g. +1 same-label / −1 diff-label) so the
  ≥0.99 target is reproducible.

### Item 5 — VETO / SCOPE · PASS
Single-dataset own-train∪val memory ✓; no OCR ✓; **no gold in-method** — the oracle uses **video-level**
gold labels only, as a probe ceiling (§6.4, §10), and needs **no** time-span / per-frame annotation
(A4 will make this explicit) ✓; no cross-seed ensemble ✓; no external API ✓; no MLLM-score-as-signal ✓
(matching is data-driven, no score); no kNN-pool expansion ✓; local Qwen-7B only ✓; not a P1–P5
re-proposal ✓. Storage ~80–210 MB, sub-GB, 471 G free ✓ (`S2S_PROBE_DESIGN.md:219-221`). The
non-isomorphism claims (§3 of the prereg) hold: vs P3 (no score, changes the metric not the weighting),
vs SAV (matching-geometry not information-content — and the prereg **carries SAV's MHC-EN falsification
forward honestly** as a weaker MHC-EN prior, §9/§13), vs encoder-swap (composable, changes pooled→set),
vs P6 (matches vs scores), vs MoRE (changes the distance not the pipeline). Genuinely open cell.

### Item 6 — CEREMONY · PASS w/ BLOCKING A5
Single-submit, no `--time`, JobHeldUser=wait, one sbatch with two dataset invocations, `--limit 1` smoke
to a throwaway path, raw-only transcription with **NO pass/fail interpretation** (`S2S_PROBE_DESIGN.md:189`,
matching the house independent-verdict rule), and a **separate** post-probe authorization for head
training (§7, §11) — all correct. **The gap is A5:** no hash-freeze clause, and the three scripts that
would actually run **do not yet exist** (`S2S_PROBE_DESIGN.md:257-259`). B3/B5 authorizations pinned
existing, reviewer-verified, sha256-hashed artifacts; this review can authorize only the *design*.
- **N5/N6 (non-blk):** state the oracle-vs-raw ordering expectation (oracle Δ should generally ≥ raw Δ;
  a raw Δ materially exceeding oracle → oracle-construction bug, investigate not auto-pass), and name the
  independent-verdict condition explicitly (mirror B5 §5.6).

### Item 7 — NOVELTY HONESTY · PASS
§9 `:295-305` is honest: the raw mechanism (OTAM/DeepEMD/ColBERT/ColPali/Video-ColBERT) is established;
the claim is **domain + representation transfer** (first set-to-set retrieval in hateful-video, first
over MLLM video-language frame tokens); whether it clears the novelty clause is the **same pending D7
user ruling** as B3-LoRA, explicitly NOT decided here; this file decides only the performance clause's
G0-cond screen. The binding-gap honesty is repeated. No overclaim.

### Item 8 — MISSED false-PASS / false-KILL surfaces
A1 (grouping ungated), **A2 (sim-magnitude vote confound)**, **A3 (near-duplicate leakage)** — all in §3.

---

## 3. Required amendments

### BLOCKING (fold in + re-hash before any authorization is exercised)

**A1 — G-decomp and G-recon are both grouping-invariant; the frame set (the scientific object) is
un-gated, and §4/§13 overclaim what G-decomp proves.**
In the probe pseudocode (`S2S_PROBE_DESIGN.md:91-111`) `vis_pos = vis_mask[:end].nonzero()` and
`nonvis_pos = (~vis_mask[:end]).nonzero()` are **complementary within `[0:end]` by construction**.
Therefore `Σ_t n_t·g_t + p_S = Σ(vision) + Σ(non-vision) = Σ[0:end]` and `(…)/end = mean[0:end]`
**for ANY `vis_mask` and ANY grouping of the vision tokens**. Consequences: (i) a **wrong
`video_pad_id`** (wrong vision/text boundary) still passes G-decomp; (ii) a **wrong temporal grouping**
(spatial-major misread, interleaved frames, wrong per) still passes G-decomp; (iii) G-recon is also
grouping-invariant (it checks the pooled aggregate). So the prereg §4 claim "G-decomp proves the frame
set is exactly the banked representation, decomposed … any residual > 1e-5 means the token→frame
assignment … is wrong" and §13 K1 are **false as stated** — neither gate protects the per-frame set.
*(For the record: I verified the grouping IS correct for this model+config, §1.3 — but "correct by
construction, verified against source" is not "gated," and a future code change or an unexpected
`video_grid_thw` shape would drift silently.)*
REQUIRED — all cheap:
1. **Correct the overclaim** in prereg §4 (`:150-153,:160-163`) and §13 K1: G-decomp proves only
   (span `end` matches) + (the vision/non-vision partition is complete) + (arithmetic self-consistency);
   it does **not** verify `video_pad_id` correctness or the temporal grouping.
2. **Add a HARD grid-consistency gate** (free, from the already-saved `grid_thw`): assert
   `n_vis == grid_t·(grid_h//spatial_merge_size)·(grid_w//spatial_merge_size)` **and**
   `per == (grid_h//spatial_merge_size)·(grid_w//spatial_merge_size)`. This catches a wrong
   `video_pad_id` (mask count ≠ grid count) and a wrong per-group size — strictly stronger than the
   `n_vis % T == 0` check.
3. **Add a temporal-structure positive control** on ≥1 synthetic clip (e.g. 8 frames = 4 distinct
   solid-color pairs): verify each `g_t` is nearest its intended temporal slab and that permuting input
   frame order permutes `g_t` as expected. HALT on failure. (This is the only check that actually
   exercises the grouping.)
4. **Cite the layout proof** (`modeling_qwen2_5_vl.py:466-505` window_index; `:529-534,:560-562`
   reverse_indices ⇒ (t,h,w) row-major) so the executor/verdict can confirm temporal-major contiguity.

**A2 — the paired SET−POOLED Δ is confounded by sim-magnitude weighting in the vote.**
`metrics.py:270` uses the arm's pairwise score as a multiplicative vote **weight**, not merely a ranking
key (§1.6). MeanMaxSim (a mean of maxes) has a compressed, upward-shifted range vs pooled cosine, so at
even *identical* retrieved neighbors the two arms produce different votes → the paired Δacc/ΔmF1
conflates retrieval-neighborhood quality with sim-scale weighting. This is a live **false-PASS** (SET
wins by scaling) and **false-KILL** (SET's real edge masked by scaling) surface.
REQUIRED (cheap): add a **rank-only** co-diagnostic vote (constant sim, or per-arm rank-normalized
scores, so both arms weight identically and the only difference is *which* neighbors are retrieved),
computed for both arms; require the primary Δ's sign and permutation/bootstrap significance to be
**corroborated** in the rank-only arm; report both. Keep the sim-weighted arm as the reported primary
(it mirrors the downstream vote), but the rank-only arm de-confounds the mechanism claim.

**A3 — near-duplicate / same-source retrieval leakage is uncontrolled and specifically inflates SET.**
SET matches on shared segments; if train∪val contains near-duplicate or same-source clips (plausible for
MHC YouTube/Bilibili clips and HateMM re-uploads), SET can "win" by re-discovering duplicates rather than
by aligning hateful segments across genuinely distinct videos — inflating the probe beyond what
generalizes and mis-attributing the mechanism. The permutation null does **not** catch this (near-dups
satisfy true set↔label structure).
REQUIRED (cheap, CPU): (i) a **pre-declared near-duplicate audit** reported per dataset — count of
distinct-video pairs whose frame-set max-cosine (and pooled cosine) exceeds a pre-declared threshold
(e.g. ≥ 0.98); (ii) a **pre-declared same-source/near-dup-excluded retrieval sensitivity** (drop
above-threshold neighbors from the LOO retrieval) confirming the SET advantage survives. If either
dataset is provably distinct-source, a one-line statement of that provenance discharges (ii).

**A4 — the oracle-ceiling frame-selection statistic is under-specified for a gating decision.**
§6.4 `:220-224` "the single frame-group that maximizes retrieval vote separation" does not define what
is maximized, over what data, per-query or global, or the tie-break — yet the +0.04 oracle Δ decides
whether ANY head GPU is spent (§7 gate 4). This is the exact class as B5's blocking A1 (an ambiguous
gating rule).
REQUIRED (cheap): pin the exact deterministic oracle procedure — the objective function, evaluated on
train∪val LOO, per-query vs global selection, using **only video-level gold labels** (state explicitly:
no time-span / per-frame gold), and the tie-break — pre-declared before results. Restate the
oracle-vs-raw ordering expectation (N5) so a deflated oracle is not a silent false-KILL.

**A5 — no hash-freeze; the extraction sbatch, extraction script, and probe script DO NOT YET EXIST.**
`S2S_PROBE_DESIGN.md:257-259` lists `src/utils/generate_VideoMLLM_frameset_HF.py`,
`scripts/slurm/s2s_extract.sbatch`, `scripts/analysis/s2s_g0cond_probe.py` as "to be created at
execution, after review." I cannot authorize submission of code I have not seen.
REQUIRED: (i) add a hash-freeze clause pinning the two design files now (sha256, re-verified at submit,
B3/B5 style); (ii) the three scripts must be authored to this spec, **independently code-reviewed**
(house practice for model-internal / hook code — the G-decomp/G-recon/grid gates, the real-vote reuse
via `compute_metrics_retrieval`, and the A4/N4 guards verified), and sha256-pinned **before** the single
Stage-E submit; (iii) a `--limit 1` smoke must show all HARD gates (A1 grid gate, G-decomp, G-recon,
temporal positive control) green on a throwaway path before the real run. This review authorizes the
**design**; the code review of the written scripts is a **separate gate**.

### NON-BLOCKING (fold in at execution / verdict; do not re-open the design authorization)
- **N1** — pre-declare null seeds (0..99); same permutation per seed across both arms; optional per-frame null.
- **N2** — specify the exact Fano-arm sim value (±1 by label agreement).
- **N3** — state "sensitivity arms cannot rescue a failed primary" explicitly.
- **N4** — probe-script fail-closed guard: `test_seen` never in the retrieval memory / never voted.
- **N5** — oracle Δ ≥ raw Δ ordering expectation; raw > oracle ⇒ investigate oracle construction.
- **N6** — name the independent-verdict condition (executor writes raw only; verdict processing separate).
- **N7** — provenance cite `:320-322` → pooling is `:303` (do NOT edit a hashed file for this; correct at next edit, B3 F-1 precedent).

---

## 4. Final verdict — APPROVED-WITH-AMENDMENTS
The S2S pre-registration rests on a correct, independently confirmed core insight (the banked pooled
cache is a prefix-span mean over vision∪text tokens, not a per-frame mean), targets the one structurally
untouched component (the retrieval object) via a representation-geometry lever (D2 — the only class that
ever cleared +3), is honestly scoped (performance clause only; novelty a pending D7 ruling; MHC-EN
treated as the binding-gap co-primary with SAV's falsification carried forward), prices probe→train
pessimism into a demanding +0.05 raw bar, and keeps a zero-test-touch ledger with a genuine oracle
kill-switch. The five blocking amendments close a mis-stated and grouping-blind correctness gate (A1),
a vote confound that could flip the verdict (A2), a leakage surface that could inflate it (A3), an
under-specified gating statistic (A4), and a ceremony/authorization gap where the executable code does
not yet exist (A5) — all cheap, none touching a number. Seven non-blocking items strengthen statistical
and reporting discipline.

---

## 5. Conditional authorization — DESIGN of the EXTRACTION + PROBE stages ONLY

**Granted 2026-07-14 by the S2S pre-registration reviewer**, scoped to the **design of Stage E
(one frozen extraction sbatch, ~1–2 GPU-h) + Stage P (zero-GPU, zero-test-touch CPU probe)**. The
downstream head-training formal stage (prereg §11) is **NOT** authorized — it is a separate,
independently-reviewed pre-registration gated behind the Stage P oracle kill-switch.

Conditions (ALL required before the single Stage-E submit is authorized):
1. **A1–A5 folded into both files and re-hashed.** N1–N7 SHOULD be folded now; only N1/N2/N4 are
   probe-correctness relevant.
2. **Scripts authored-to-spec, independently code-reviewed, and sha256-pinned.** The extraction script,
   sbatch, and probe script do not yet exist; this authorization does **not** cover their submission
   until they are written, code-checked (A1 grid gate + G-decomp + G-recon + temporal positive control
   + real-vote reuse + N4 test-touch guard), and hash-frozen. That code review is a **separate gate**.
3. **HARD gate order, HALT-on-failure:** A1 grid-consistency + temporal positive control → G-decomp
   (≤1e-5) → G-recon (cos ≥ 0.9999 AND max-abs ≤ 1e-3 vs the banked `img_feats`, per non-zero-guard
   video) → Fano (≥ 0.99 both datasets, else VOID) → oracle kill-switch (A4 rule; < +0.04 every dataset
   ⇒ DEAD, no head GPU) → raw bar + permutation null + bootstrap → dataset rule (a/b/c/d).
4. **Single-submit discipline:** one `sbatch`, no `--time`, `PENDING (JobHeldUser)` = WAIT (never
   force-release), `--limit 1` smoke to a throwaway path permitted first, no resubmission after any
   terminal state.
5. **Gold labels used ONLY for the bounded Fano + oracle-ceiling probes (train∪val, video-level);**
   test frame sets are cached but **never scored** at Stage P; the oracle number is NEVER a reported
   result.
6. **Executor writes raw numbers + gate distributions (G-decomp max-residual, G-recon cos/max-abs, Fano,
   oracle ceiling, null percentiles, bootstrap percentiles, per dataset per arm) with line-numbered
   provenance and applies NO pass/fail interpretation** — verdict processing is independent.
7. **No SLURM/GPU beyond the single extraction job; no config/state/CLAUDE.md mutation;** writes limited
   to the three scripts, the sub-GB frame-set caches (guard-excluded path), and the probe output table.

**Out of scope:** the head-training formal stage; any second submission; any change to encoder / dataset
/ frames beyond the two pre-declared budgets; any test-label use beyond the pre-declared Fano/oracle
ceiling; treating the oracle or any sensitivity arm as the reported result.
