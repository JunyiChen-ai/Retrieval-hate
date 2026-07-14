# W2-B Probe — Executable Design Spec (sub-clip set-matching, cloud features-only)

**Line name:** W2-B (multimodal sub-clip set-matching). **Companion:** `refine-logs/W2B_FORENSIC_RECON.md`
(cache reality + non-isomorphism), `research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md §W2-B`. **Adapts:**
`refine-logs/S2S_PROBE_DESIGN.md §5` + `scripts/analysis/s2s_probe.py` (vote reuse, permutation-null
distribution ≥100 seeds, bootstrap, Fano/oracle calibration, near-dup audit, rank-only co-diagnostic,
fail-closed no-test-touch). **Status:** DESIGN DRAFT — **NOTHING authorized to run**; a fresh reviewer
gates execution and (if a probe script is authored from this spec) an independent code review + hash-freeze
precedes any Modal dispatch. **Execution target:** Modal, features-only, CPU, **~$0**.

**One-paragraph honest framing (binding on the verdict).** W2-B is a **zero-training, frozen-feature,
paired LOO kNN** measurement asking one thing: *does set-matching (MeanMaxSim) over a video's K sub-clip
vectors beat pooling (their mean) as the retrieval metric, on hate-video kNN?* It is **not a D7-novel
contribution** (mechanism ≈ S2S) and its **prior on a positive is weak** — the recon establishes that the
*trained* version of essentially this idea (Delta-1 / seg-mode multi-granularity) was already **killed
high-confidence** on MHClip ("sign-flips by language, no consistent gain; do not re-attempt without gold
spans"). W2-B's value is a **~$0 family de-risker for the S2S LEAD**: it removes Delta-1's
training-signal confound (no training, no MIL pseudo-labels), and it tests the question S2S rests on before
S2S's Qwen frame-set extraction GPU is spent. The most likely, most useful outcome is a **clean negative
that revises the whole don't-pool family's prior down**, bounded by the **CLIP<Qwen encoder asymmetry** (a
CLIP-null does not fully close the Qwen-token S2S version).

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
  for language-robustness color, never a primary gate.

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
8. **ORACLE** (A4 ceiling, video-level gold ONLY) — per query `Q` with gold `y_Q`, for each sub-clip index
   `t` form `s_t(Q,M)=max_m cos(ĝ^Q_t, ĝ^M_m)`, run the real vote → margin `v_t(Q)`, pick
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
2. **Oracle kill-switch** — per-query oracle-sub-clip MaxSim; paired Δ(oracle − POOLED); **DEAD if <
   +0.04 on EVERY dataset** (pooling discards nothing convertible → whole don't-pool family de-risked
   down; no further arm can rescue).
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

**K-budget / no-shopping rule.** The verdict is decided on the **K4 PRIMARY** (train∪val). **K30 and `_mm`
are SENSITIVITY arms and cannot rescue a failed K4 primary** (mirror S2S N3). K30 is reported only as the
matched **K4-train-only vs K30-train-only** granularity contrast on HateMM's 744; `_mm` only as the matched
**K4-visual vs K4-mm** contrast on MHC-EN's 549. No sweeping K beyond {4, 30}; no per-arm cherry-picking.

---

## 5. Pre-declared kill logic (the scout's kill, made exact)
> **KILL:** `HateMM K4 primary` paired Δacc AND ΔmF1 **< +0.05** (rank-only-uncorroborated counts as
> below) **AND** `MHC-EN K4 primary` paired Δ **< +0.03/+0.03** **AND** the **K30-train-only** contrast on
> HateMM also **< +0.05/+0.05** → set-matching does not beat pooling at the sub-clip granularity on banked
> CLIP features → **strong negative prior update for the whole don't-pool family** (outcome (a)/(d)).

> **SURVIVE / ESCALATE:** outcome (b) → the Qwen-token S2S version is the escalation (W2-B itself is not
> D7-novel); outcome (c) → report the single-dataset advantage with the Delta-1 sign-flip caveat, no
> family green-light.

All bars are frozen **before** any number is computed. No arm/metric OR-ing beyond the four fixed
dataset-rule rows (S2S N3).

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
  reviewer gates execution; if a `scripts/analysis/w2b_probe.py` is authored from this spec it takes a
  separate independent code review + hash-freeze before any Modal dispatch (S2S precedent).

---

## 11. Hash-freeze
Both design docs are pinned at the STEP-3 commit (a file cannot embed its own hash; recorded in the commit
message / below). A probe script, if authored later, is hashed at its own code-review gate.

<!-- W2B-HASH-TABLE-START -->
| artifact | sha256 |
|---|---|
| `refine-logs/W2B_FORENSIC_RECON.md` | `17f18bf597b654d50a5eb77246b1760f248e855e2d37f9f57fbc1c408e43b7e5` |
| `refine-logs/W2B_PROBE_DESIGN.md` (this file) | _(recorded in the STEP-3 commit message — a file cannot embed its own hash)_ |
<!-- W2B-HASH-TABLE-END -->
