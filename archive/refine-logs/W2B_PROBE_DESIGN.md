# W2-B Probe — Executable Design Spec (sub-clip set-matching, cloud features-only)

**Line name:** W2-B (multimodal sub-clip set-matching). **Companion:** `refine-logs/W2B_FORENSIC_RECON.md`
(cache reality + non-isomorphism), `research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md §W2-B`. **Adapts:**
`refine-logs/S2S_PROBE_DESIGN.md §5` + `scripts/analysis/s2s_probe.py` (vote reuse, permutation-null
distribution ≥100 seeds, bootstrap, Fano/oracle calibration, near-dup audit, rank-only co-diagnostic,
fail-closed no-test-touch). **Status:** **r1 — APPROVED-WITH-AMENDMENTS (blocking B1–B3 + non-blocking
N1–N5 folded in place from `refine-logs/W2B_PREREG_REVIEW.md`, 2026-07-15).** Cloud execution is
conditionally authorized (review §8) **only after** the authored `scripts/analysis/w2b_probe.py` passes a
**separate** independent code review + hash-freeze; **the real probe is NOT authorized to run** until that
code re-check clears. **Execution target:** Modal, features-only, CPU, **~$0**.

**One-paragraph honest framing (binding on the verdict).** W2-B is a **zero-training, frozen-feature,
paired LOO kNN** measurement asking one thing: *does set-matching (MeanMaxSim) over a video's K sub-clip
vectors beat pooling (their mean) as the retrieval metric, on hate-video kNN?* It is **not a D7-novel
contribution** (mechanism ≈ S2S) and its **prior on a positive is weak** — the recon establishes that the
*trained* version of essentially this idea (Delta-1 / seg-mode multi-granularity) was already a
**killed analysis** (`exp-seg-mode-ablation.md`, verdict NO / high-confidence: no config beat whole-video
on **both** languages; "do not re-attempt segment-level temporal retrieval without gold spans").
**(r1: N4) Honest refinement on the binding dataset:** that kill was driven by **MHC_zh** (the sign-flip,
−0.066 F1); on the **binding MHC-EN** dataset the trained precedent was a **marginal positive-below-bar**
(+0.0149 F1 / +0.0062 acc), *not* a negative. So the correct prior is "weak-positive-below-bar on MHC-EN,
sign-flipped on ZH," not "already-negative everywhere" — the framing must not over-pessimize the binding
dataset. W2-B's value is a **~$0 family de-risker for the S2S LEAD**: it removes Delta-1's training-signal
confound (no training, no MIL pseudo-labels), and it tests the question S2S rests on before S2S's Qwen
frame-set extraction GPU is spent. The most likely, most useful outcome is a **clean negative that revises
the whole don't-pool family's prior down**, bounded by the **CLIP<Qwen encoder asymmetry** (a CLIP-null
**lowers but does not veto** the Qwen-token S2S prior — see §4.8).

---

## 1. What the caches are (correctness reference)
See `W2B_FORENSIC_RECON.md §1`. In brief: `subclip_img_feats [V*K,1024]` = frozen CLIP-ViT-L/14-336
vision-pooler **mean over 4 contiguous frames**, unnormalized; `subclip_parent` = contiguous K-blocks;
`labels` per-subclip (MIL parent inheritance); K=4 primary, K=30 (HateMM-train-only), `_mm` adds
per-segment Whisper-ASR CLIP-text `[V*K,768]` (MHC/MHC_zh-train-only, 71% text coverage). Ids and labels
are index-identical to the pooled caches. HateMM train has **1 zero-guard video** (K zero rows).

**Availability → protocol partition (pre-declared, from recon §1.2):**
- **PRIMARY (K4):** memory = **train ∪ dev_seen, LOO**. Sizes **HateMM 851, MHC-EN 629** — identical to
  S2S `EXPECTED_MEM`; the S2S N4 size-guard transfers verbatim.
- **K30 SENSITIVITY:** HateMM has K30 for **train only** → **within-train LOO (744)**, paired against a
  **K4-train-only** arm on the identical 744 videos. NEVER mixed with the 851 primary.
- **`_mm` SENSITIVITY:** MHC-EN has `_mm` for **train only** → **within-train LOO (549)**, paired against a
  **K4-visual-train-only** arm on the identical 549 videos.
- **MHC_zh:** OPTIONAL third-dataset K4 arm (train∪val = 657); ZH is out of the binding gap, carried only
  for language-robustness color, never a primary gate. **(r1: N1)** its caches were NOT independently
  re-derived at review (out-of-goal, non-gating); if MHC_zh is ever promoted past robustness color, its
  caches must be re-derived (shapes/ids/labels) first.

---

## 2. Loader adaptation (the only real delta vs `s2s_probe.py`)
`s2s_probe.py`'s loader reads `fs["ids"][0]`, `fs["g"] [N,T,D]`, per-video `fs["labels"] [N]`,
`fs["zero_guard"]`. The sub-clip cache differs (recon §1.1). A W2-B loader must:
1. Read `video_ids` (flat), `subclip_img_feats [V*K,D]`, `subclip_parent`, `labels [V*K]`,
   `num_subclips=K`.
2. **Assert** `subclip_parent` is the contiguous K-block pattern (recon-verified) and `V*K == rows`;
   then `g = subclip_img_feats.view(V, K, D)`.
3. Per-video label = `labels.view(V,K)[:,0]`; assert row-constant within a video.
4. Per-video `zero_guard = (g.norm(dim=-1) < 1e-6).all(dim=1)`.
5. Memory assembly + **N4 fail-closed guard**: train + dev_seen ONLY; assert
   `len(memory) == {851 HateMM / 629 MHC}` for the primary (or `{744/549}` for the train-only sensitivity
   arms). Never construct/open a `test_seen` path (mirror `s2s_probe.py:_frameset_path` guard).
6. `_mm` arm additionally loads `subclip_txt_feats [V*K,768]` + `subclip_txt_has_text [V*K]`; zero-mask
   empty-text subclips.

Everything downstream (score matrices, the REAL vote, nulls, bootstrap, oracle, Fano, near-dup) is the
S2S machinery with `T→K` and `g_t→` sub-clip vectors. **The vote is reused, never reimplemented:**
`compute_metrics_retrieval(logging_dict, labels, majority_voting='arithmetic', topk=20, use_sim=True)` from
`src/utils/metrics.py` (identical to `s2s_probe.py:196`).

**(r1: B1) Video-level LOO / parent-exclusion is the structure — stated explicitly (the flagged biggest
false-PASS risk).** The memory is **V video-level sets** (`g = subclip_img_feats.view(V, K, D)`, one
K-element set per video); retrieval and the vote are **video-to-video** (query video's K-set scored against
each memory video's K-set by POOLED / SET / ASYM), and **LOO holds out the WHOLE query set** (the vote's
diagonal exclusion is at the **video** index, not the sub-clip index). There is **NO flat `[V·K,D]`
sub-clip retrieval bank anywhere** — that would be catastrophic trivial leakage (a query sub-clip would
retrieve its K−1 sibling sub-clips: same parent, adjacent windows, near-identical, same inherited label).
Because retrieval is video-level and LOO holds out the entire query set, a query's own sub-clips are
**structurally un-retrievable**, so **parent-exclusion is subsumed by video-level LOO by construction.** The
pre-declared `len(memory) == 851/629` assert (§2 step 5) is a **video-count** guard that would itself trip
(→ 3404/2516) if a flat sub-clip bank were ever built. The future code review (§10 code-review gate) MUST
verify this holds in `w2b_probe.py`.

---

## 3. Arms (per dataset; memory = train∪val for K4 primary; LOO; ĝ = L2-normed sub-clip vecs)
All computed on the **same frozen sub-clip vectors**, run through the **same LOO vote**, paired, same
seeds — mirroring `s2s_probe.py:build_matrices/probe_dataset`.

1. **POOLED** (primary null) — `cos(mean_k g^Q_k, mean_k g^M_k)`, normalized pooled sub-clip means. The
   "pooling destroys segment alignment" baseline.
2. **SET** (primary) — `MeanMaxSim(Q,M) = (1/K) Σ_q max_m cos(ĝ^Q_q, ĝ^M_m)`. Late interaction over the
   sub-clip sets.
3. **SET-Chamfer** (single sensitivity) — `0.5[MMS(Q→M)+MMS(M→Q)]`.
4. **ASYM** (folded, mirrors the S2S C2 cell) — `max_m cos(ĝ^Q_pooled, ĝ^M_m)`, the `|Q|=1` reduction of
   MeanMaxSim (pooled-query × set-memory). Same frozen vectors, same vote.
5. **PIPELINE-ANCHOR** — pooled cosine over the **banked whole-video `img_feats`** (pooled parent CLIP
   cache). Internal reference tying the probe to the real pipeline cache; **not** a primary null.
6. **WITH-TEXT** (K4) — arms 1/2 visual score **+** `cos(text^Q, text^M)` over the **video-level** pooled
   `text_feats` (the plain K4 sub-clips share video-level text; identical channel both arms). Sensitivity.
7. **RANK-ONLY** (A2 co-diagnostic, MANDATORY) — retrieve by the arm's own score but set every
   `retrieved_scores` entry to **1.0**, so the vote reduces to identical rank-position weighting and the
   only remaining difference is *which* neighbours are retrieved. De-confounds sim-scale from
   neighbour-quality. **Credit rule (pre-declared):** the sim-weighted primary Δ is credited only if the
   rank-only Δ **matches its sign AND is itself significant** (rank-only observed Δ > its own perm-null
   95th pct AND rank-only bootstrap 5th pct > 0). (Verbatim from `s2s_probe.py` B3 fix.)
8. **ORACLE** (A4 ceiling, video-level gold ONLY) — **(r1: B3) computed on the K4 PRIMARY arm** (train∪val
   memory; the K30 oracle, if reported at all, is a sensitivity number ONLY and never feeds the
   kill-switch). Per query `Q` with gold `y_Q`, for each sub-clip index `t` form
   `s_t(Q,M)=max_m cos(ĝ^Q_t, ĝ^M_m)`, run the real vote → margin `v_t(Q)`, pick
   `t*(Q)=argmax_t (2y_Q−1)·v_t(Q)` (smallest-index tie-break); oracle score = `s_{t*(Q)}(Q,·)`; memory
   keeps full sets (no double-dip). Upper-bounds how much sub-clip alignment structure pooling discards
   *could* buy. **Ordering expectation:** oracle Δ ≥ raw Δ; a raw Δ materially above oracle ⇒ oracle bug,
   investigate (do NOT auto-kill). Never reported as a result.
9. **FANO** (N2 calibration) — retrieve by ±1 gold-label agreement key; vote acc must reach **≥0.99** both
   datasets, else the vote machine is VOID and no negative verdict is admissible.
10. **`_mm` SET** (multimodal sensitivity, MHC-EN train-only) — the genuine `_mm` sliver: score each
    sub-clip **pair** by `cos(ĝ^Q_q,ĝ^M_m) + cos(t̂^Q_q,t̂^M_m)` with the text term **masked to 0 where
    either segment lacks ASR** (`has_text=False`), then MeanMaxSim over the K sub-clips. Paired against
    `_mm`-visual-only SET and against POOLED on the identical 549 train videos.

---

## 4. Gates, in order (mechanical arithmetic only; independent verdict reviewer rules — house rule)
Mirrors `s2s_probe.py:mechanical_gate_check`. The executor writes **RAW numbers only**; a machine JSON
carries a `mechanical_gate_check` block stamped *"pre-registered arithmetic, NOT the binding verdict."*

1. **Fano** — ±1 gold-label key LOO vote acc ≥ **0.99** each dataset, else **VOID**.
2. **Oracle kill-switch** — **(r1: B3) computed on the K4 PRIMARY arm only** (a K30-inflated 30-way-max
   ceiling must not bypass the early kill). Per-query oracle-sub-clip MaxSim; paired Δ(oracle − POOLED);
   **DEAD if < +0.04 on EVERY dataset** (pooling discards nothing convertible → whole don't-pool family
   de-risked down; no further arm can rescue).
3. **Raw bar (anchor)** — **HateMM** mean paired Δ(SET−POOLED) **acc ≥ +0.05 AND macro-F1 ≥ +0.05**,
   **corroborated by the rank-only arm** (§3.7). This is the S2S "P3-priced" shrinkage bar for a
   zero-training frozen probe.
4. **MHC-EN survival bar** — MHC-EN paired Δ(SET−POOLED) **acc ≥ +0.03 AND macro-F1 ≥ +0.03** (the goal's
   cross-dataset increment; the weakest defensible "could matter for the goal" threshold).
5. **Permutation null** — seed set **0..99** (≥100), shuffle sub-clip **sets across videos**, the **same**
   permutation applied to every arm within a seed (paired Δ preserved); observed Δ > **95th pct**. Optional
   secondary **per-sub-clip-vector shuffle** null (separates "alignment" from a generic "richer-key"
   effect), reported not gating.
6. **Bootstrap** — ≥**1000** query resamples; **5th pct of paired Δ > 0** else D3-FRAGILE.
7. **Near-dup audit + excluded sensitivity** — flag a distinct memory pair near-duplicate if
   `pooled_cos ≥ 0.995 OR MeanMaxSim ≥ 0.995` (report the full distribution at 0.98/0.99/0.995 for both
   metrics + the single-sub-clip max-cos distribution). MeanMaxSim ≥ 0.995 requires **all K** query
   sub-clips to near-match ⇒ global near-duplication, not single-segment sharing (a lone shared hateful
   segment gives MMS ≈ 1/K = 0.25 ≪ 0.995, so the flag cannot swallow the signal). Re-run LOO dropping
   flagged neighbours; the SET advantage must **survive**. Zero-guard rows excluded from the audit.
8. **Dataset rule (single-vs-both)** — assign the outcome:
   - **(a) DEAD-family:** oracle < +0.04 on every dataset → set structure is not convertible even under an
     oracle → **strong negative prior update for the don't-pool family** (S2S/C2/W2-C priors revised down,
     bounded by CLIP<Qwen). This is the pre-registered *expected* outcome.
   - **(b) BOTH:** HateMM raw bar (§3, corroborated) AND MHC-EN survival bar both cleared → set-matching
     beats pooling on frozen CLIP on both anchor datasets → **S2S prior revised UP**; escalate to the
     Qwen-token S2S version (which carries the novelty).
   - **(c) SINGLE:** exactly one of {HateMM raw bar, MHC-EN survival bar} clears → partial; language/dataset
     -specific, mirrors Delta-1's sign-flip; not a family green-light.
   - **(d) NEGATIVE:** neither raw bar clears but oracle survived on ≥1 dataset → set ≈ pool at the
     *decision* level on frozen CLIP despite some oracle headroom → weak-negative family update.

   **(r1: N5) Forward-direction consolidation (W2-B result → S2S GPU decision), in one place.** A W2-B
   **negative** (a/d) **lowers** the S2S/C2/W2-C priors but **does NOT veto** S2S: the pre-declared
   **CLIP<Qwen** encoder asymmetry (§9 threat #1) means a weaker-encoder null cannot close the Qwen-token
   version — the negative is a cheap prior-down update that argues *against* spending S2S's Qwen GPU, not a
   proof S2S is dead. A W2-B **positive** (b) **raises** the S2S prior and escalates to the novelty-carrying
   Qwen-token version (W2-B itself is not D7-novel). A **single** (c) mirrors Delta-1's language sign-flip
   and greenlights nothing. This is the entire load-bearing purpose of the probe.

**K-budget / no-shopping rule.** The verdict is decided on the **K4 PRIMARY** (train∪val). **K30 and `_mm`
are SENSITIVITY arms and cannot rescue a failed K4 primary** (mirror S2S N3). K30 is reported only as the
matched **K4-train-only vs K30-train-only** granularity contrast on HateMM's 744; `_mm` only as the matched
**K4-visual vs K4-mm** contrast on MHC-EN's 549. No sweeping K beyond {4, 30}; no per-arm cherry-picking.

---

## 5. Pre-declared kill logic (the scout's kill, made exact — r1: B2 reconciled with §4 K-budget)
**(r1: B2) The K4 PRIMARY is the SOLE survival-determining arm.** The verdict (outcome a/b/c/d, §4 gate 8)
is decided **only** by the K4-primary paired Δ on the two anchor datasets. The K30-train-only contrast can
**never** rescue a failed K4 primary and can **never** convert a survival into a kill; it only **modulates
the breadth** of a *negative* (whether the negative is K4-specific or persists across granularity). This
removes the §4-vs-§5 ambiguity flagged in the review (a state where "K4 fails both datasets but K30 clears
+0.05" is still outcome (a)/(d) NEGATIVE — K30 does not rescue).

> **KILL (survival-determining, K4 primary ONLY):** `HateMM K4 primary` paired Δacc AND ΔmF1 **< +0.05**
> (rank-only-uncorroborated counts as below) **AND** `MHC-EN K4 primary` paired Δ **< +0.03/+0.03** →
> set-matching does not beat pooling at the sub-clip granularity on banked CLIP features → **strong
> negative prior update for the whole don't-pool family** (outcome (a)/(d)).

> **Breadth-modifier (reported, NON-determining):** if the KILL above fires, the **K30-train-only** contrast
> on HateMM's 744 (paired vs K4-train-only) is reported to characterise the negative's *breadth* — a K30
> that is **also < +0.05/+0.05** says the negative persists across granularity (broad family de-risk); a
> K30 that clears is a *reported* granularity note that does **NOT** rescue the K4-primary kill.

> **SURVIVE / ESCALATE (K4 primary ONLY):** outcome (b) → the Qwen-token S2S version is the escalation
> (W2-B itself is not D7-novel); outcome (c) → report the single-dataset advantage with the Delta-1
> sign-flip caveat, no family green-light.

All bars are frozen **before** any number is computed. No arm/metric OR-ing beyond the four fixed
dataset-rule rows (S2S N3); K30/`_mm`/Chamfer/WITH-TEXT/MHC_zh are sensitivity/breadth reports only.

---

## 6. No-test-touch, determinism, machine validity
- **Fail-closed (N4):** the probe never constructs/opens any `test_seen` path; after loading it asserts
  `len(memory) == EXPECTED_MEM` for the active protocol (851/629 primary; 744/549 train-only sensitivity).
  Any deviation is a HARD failure. (Verbatim from `s2s_probe.py`.)
- **Determinism:** `CUDA_VISIBLE_DEVICES=""` (CPU), `torch/np` seeded (20260714), deterministic index
  tie-break (smaller memory idx wins), null seeds 0..99 fixed, bootstrap seed fixed.
- **Probe self-test:** the synthetic shared-segment positive control (`s2s_probe.py:synthetic_set_control`)
  — plant a shared sub-clip in two otherwise-different videos; MeanMaxSim must exceed POOLED, else HALT
  (set-metric bug). Transfers verbatim (T→K).

---

## 7. Execution target — Modal, features-only, CPU (~$0)
- **Volume:** `rgcl-features` (`scripts/cloud/modal_probe_runner.py:VOLUME_NAME`), mounted `/root/data`.
- **Are the sub-clip caches on the volume?** They are **cloud-eligible** (derived `.pt` CLIP floats; pass
  `assert_uploadable`) and are **included automatically** by `sync` (uploads the whole
  `data/CLIP_Embedding/<dataset>/` dir). **Execution step-0 (idempotent, ~$0, a few MB):**
  ```
  modal run scripts/cloud/modal_probe_runner.py::sync --dataset HateMM
  modal run scripts/cloud/modal_probe_runner.py::sync --dataset MHC
  # (optional third-dataset arm) --dataset MHC_zh
  ```
  This pushes pooled + **sub-clip** (K4, K30, `_mm`) caches + `data/gt` labels in one shot.
- **Run (CPU):**
  ```
  modal run scripts/cloud/modal_probe_runner.py::run \
      --script scripts/analysis/w2b_probe.py \
      --args "--data_root /root/data --datasets HateMM,MHC"
  ```
  The runner auto-mounts `scripts/analysis/` and `src/`; the probe script MUST (a) live in
  `scripts/analysis/`, (b) take `--data_root` (default local repo `data/`; `/root/data` on Modal) so the
  same file runs locally and on cloud, (c) import the vote from `src/utils/metrics.py` (present on the
  image). **CPU is sufficient** — CLIP 1024-d, K=4, all-pairs frame-frame tensor is tiny (HateMM 851²·16·
  1024 ≈ 1e10 MACs → seconds).
- **Cost:** CPU minutes within Modal free credits → **~$0. No GPU.**
- **G-repro note:** per the cloud rule, cloud results are **exploratory triage only**; if W2-B were ever to
  feed a paper number it would be re-run locally on the same hardware as its table. For a *negative*
  family-de-risk update the cloud number is sufficient (no local re-run needed to decline to spend GPU).

---

## 8. Storage / output layout
- Human: `refine-logs/W2B_PROBE_RESULTS.md` — raw per-arm AUC/acc/mF1, paired Δ tables (sim-weighted AND
  rank-only), Fano, oracle ceiling, near-dup audit table, null percentiles, bootstrap percentiles, the
  K4/K30/`_mm` sensitivity contrasts, per dataset, **NO pass/fail interpretation**.
- Machine: `refine-logs/w2b_probe_results.json` — same raw numbers + a `mechanical_gate_check` block doing
  the §4/§5 threshold arithmetic, stamped "NOT the binding verdict."
- No new heavy artifacts (the caches already exist); results are two small files.

---

## 9. What-would-kill-this (design-integrity table for the reviewer)
| # | threat | why it would invalidate a W2-B verdict | mitigation in this design |
|---|---|---|---|
| 1 | **CLIP<Qwen encoder asymmetry** | a CLIP-null does not close the Qwen-token S2S version | pre-declared; a W2-B negative revises the family prior *down* but is explicitly stated NOT to fully kill S2S |
| 2 | **near-dup self-retrieval** (MHC re-uploads, HateMM re-uploads) | SET "wins" by re-discovering duplicates, not aligning hateful segments | pooled-OR-MMS ≥0.995 flag + excluded-retrieval sensitivity must survive; MMS≥0.995 needs global dup (1/K floor) |
| 2b | **(r1: N2) coarse-sub-clip PARTIAL-overlap re-upload** — a query and a *distinct* memory video sharing ONE viral/re-uploaded segment produce ONE near-identical sub-clip pair (≈1/K = 0.25 to MeanMaxSim), **below** the 0.995 global-near-dup flag → NOT flagged | a single shared segment could inflate SET without genuine cross-video hateful-segment alignment; the same surface S2S carries (approved there), but a sub-clip is a *coarser* unit than an S2S frame-group | **explicitly named residual**; discharged by **reporting the single-sub-clip max-cos distribution** per dataset (review N2) so the reviewer can see the partial-overlap mass; not a blocking surface (matches the approved S2S treatment) |
| 3 | **sim-scale artifact** (MeanMaxSim range ≠ pooled-cos range) | raw paired Δ conflates neighbour-quality with sim weighting | mandatory rank-only co-diagnostic + sign-AND-significance credit rule |
| 4 | **probe-passes / train-flat over-read** | a frozen probe over-reads vs a trained head | +0.05/+0.05 P3-priced HateMM bar + oracle ceiling gate |
| 5 | **Delta-1 anti-repeat** (already tried, killed) | re-running a killed idea | non-isomorphism established (zero-training, no MIL pseudo-labels, isolates the confound Delta-1 was blamed on); honest weak prior; value = de-risk not novelty |
| 6 | **guard/zero rows** (1 HateMM-train video) | zero vectors pollute normalization/near-dup | eps-floored norm; per-video `zero_guard`; guard rows excluded from near-dup |
| 7 | **train-only sensitivity confound** (K30/`_mm`) | different memory size vs the 851/629 primary | K30/`_mm` are matched-memory contrasts vs a K4-train-only arm on the identical videos; never mixed with the primary; cannot rescue a failed primary |
| 8 | **test leakage** | any test row in the memory | fail-closed N4 guard: never open test_seen; assert exact memory size |

---

## 10. Scope / veto (executable restatement)
- **Stage P only** — no extraction, no GPU, no training, **ZERO test touch** (train∪val memory; gold labels
  used only for Fano + oracle ceiling).
- **Vetoes honored:** single-dataset own-train memory; **no OCR** (`_mm` = native Whisper ASR); no gold in
  the method; no cross-seed ensemble; no external API; no MLLM-score-as-signal; **CLIP-only** (local, no
  Qwen forward). Not a P1–P5 re-proposal.
- **Not authorized here:** any downstream head-training stage, and running the probe itself — a fresh
  reviewer gates execution; the authored `scripts/analysis/w2b_probe.py` takes a **separate** independent
  code review + hash-freeze before any Modal dispatch (S2S precedent).

### 10.1 Code-review gate items (r1: B1 + N3) — MUST be verified in `w2b_probe.py` before any Modal run
The separate code review of `scripts/analysis/w2b_probe.py` MUST confirm, hunk by hunk:
- **(B1-a) contiguous-parent reshape** — the loader asserts `subclip_parent == repeat_interleave(arange(V),
  K)` and `V*K == rows` before `view(V, K, D)`; a non-contiguous cache HALTS.
- **(B1-b) video-level LOO diagonal exclusion** — the vote's self-exclusion is at the **video** index
  (`np.fill_diagonal(S, NEG_INF)` on the `[V,V]` score matrix), holding out the whole query set.
- **(B1-c) no flat sub-clip retrieval index** — there is **no** `[V·K, D]` bank fed to any retrieval/vote
  anywhere; all scoring is video-to-video over the reshaped sets. The `len(memory)==851/629` (or 744/549)
  **video-count** assert is present and would trip on a flat bank.
- **(N3-i) NEG_INF filter survives** — before the vote, excluded/self entries (`row[idx] > NEG_INF/2`) are
  dropped so a near-dup-excluded neighbour can never multiply a label by ~−1e30 (S2S NB-a).
- **(N3-ii) `_mm` zero-text eps-guard** — empty-text sub-clips (verified zero-norm text rows) are
  eps-guarded / `has_text`-masked so `cos(0,·)` never produces a 0/0 NaN.
- **(N3-iii) real-vote reuse** — the vote is `compute_metrics_retrieval(..., use_sim=True, topk=20,
  majority_voting='arithmetic')` from `src/utils/metrics.py`, never reimplemented.
- **no-test-read** — no `test_seen` path is ever constructed/opened (fail-loud loader guard).

---

## 11. Hash-freeze
A file cannot embed its own hash (recorded in the commit message). The recon is unchanged by r1; the design
is re-hashed; the authored probe script is hashed here and **remains AWAITING an independent code review** —
a mismatch after a review-driven edit is expected and the table is re-pinned at that point.

**r0 (STEP-3, 2026-07-15) — retained for the audit trail:**
<!-- W2B-HASH-TABLE-R0-START -->
| artifact | sha256 (r0) |
|---|---|
| `refine-logs/W2B_FORENSIC_RECON.md` | `17f18bf597b654d50a5eb77246b1760f248e855e2d37f9f57fbc1c408e43b7e5` |
| `refine-logs/W2B_PROBE_DESIGN.md` (r0) | `04785113f8a318456f0c892cb2e8f8aa3063912901089cec9dc4f3d500a8bc8e` |
<!-- W2B-HASH-TABLE-R0-END -->

**r1 (2026-07-15, B1–B3 + N1–N5 folded; probe script authored) — CURRENT freeze, re-verify at submit:**
<!-- W2B-HASH-TABLE-R1-START -->
| artifact | sha256 (r1) | status |
|---|---|---|
| `refine-logs/W2B_FORENSIC_RECON.md` | `17f18bf597b654d50a5eb77246b1760f248e855e2d37f9f57fbc1c408e43b7e5` | **UNCHANGED** vs r0 |
| `refine-logs/W2B_PROBE_DESIGN.md` (this file, r1) | recorded in the r1 commit message | changed (B1–B3+N1–N5) |
| `scripts/analysis/w2b_probe.py` | `d22aac02b4c50f2952e1aa06b4609dd158d69ff54dd184cd9885fec1d3a15776` | authored; py_compile+synthetic self-test PASS; **AWAITING code review** |
<!-- W2B-HASH-TABLE-R1-END -->

---

## 12. Revision history
- **r0 (2026-07-15) DESIGN DRAFT.** Initial executable spec + recon (commit `1f8265a`).
- **r1 (2026-07-15) APPROVED-WITH-AMENDMENTS, amendments applied** (`refine-logs/W2B_PREREG_REVIEW.md`).
  Folded in place:
  - **B1** — made video-level LOO / parent-exclusion explicit (§2 new paragraph): memory = V video-level
    sets, video-to-video retrieval, LOO holds out the whole query set, **no flat `[V·K,D]` sub-clip bank**,
    `len(memory)==851/629` is a video-count guard; added the §10.1 code-review gate items (a/b/c).
  - **B2** — reconciled the §5 KILL with the §4 K-budget rule: the **K4 primary is the sole
    survival-determining arm**; the K30-train-only contrast is demoted from a KILL conjunct to a **reported
    breadth-modifier** that can never rescue a failed K4 primary.
  - **B3** — pinned the **oracle kill-switch to the K4 primary arm** (§3.8 + §4 gate 2); a K30-inflated
    ceiling cannot bypass the early DEAD-family kill.
  - **N1** — MHC_zh optional-arm caches not independently re-derived; re-derive if ever promoted (§1).
  - **N2** — named the coarse-sub-clip partial-overlap re-upload residual (§9 threat 2b), discharged by the
    reported single-sub-clip max-cos distribution.
  - **N3** — added the code-review guards to §10.1 (NEG_INF filter, `_mm` zero-text eps-guard, video-level
    diagonal exclusion, real-vote reuse).
  - **N4** — refined the framing so the **binding MHC-EN** prior is "weak-positive-below-bar" (+0.0149 F1),
    not "already-negative"; the sign-flip was MHC_zh (top framing paragraph).
  - **N5** — consolidated the forward-direction matrix (W2-B result → S2S GPU decision; CLIP-negative
    **lowers but does not veto** the S2S prior) into the §4.8 interpretation rows.
  - Authored `scripts/analysis/w2b_probe.py` to the amended spec; re-hashed (§11 r1 table). **Still AWAITING
    an independent code review — no Modal submission authorized.**
