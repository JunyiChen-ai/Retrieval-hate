# W2-A INDEPENDENT CODE REVIEW (r2)

**Reviewer:** fresh, zero-prior-context independent code reviewer. **Date:** 2026-07-15.
**Scope:** READ-ONLY except this deliverable. NO GPU, NO submission.
**Under review (commit 654720b, r2):** `scripts/analysis/w2a_extract.py`, `scripts/slurm/w2a_extract.sbatch`,
`scripts/analysis/w2a_probe.py`.
**Governing specs (r1, verified byte-identical vs cb59a94):** `research-wiki/experiments/exp-w2a-grounded.md`,
`refine-logs/W2A_FORENSIC_RECON.md`, `refine-logs/W2A_PREREG_REVIEW.md`.
**Reference read:** `src/utils/generate_VideoMLLM_embedding_HF.py` (parity-by-import), `src/utils/metrics.py`
(the vote), `scripts/analysis/c3_fusion_probe.py` (the K9 conditional-info precedent).

---

## VERDICT: **APPROVED FOR SMOKE**

The extractor is correct against the banked parity anchor and the re-authored novel surfaces
(transcript-first message builder + vision-pad pool) are right — both CPU self-tests pass and the K9
machinery is a faithful port of the approved C3 template. **No BLOCKING defect.** Four NON-BLOCKING items
and several NOTEs apply; three of the four live in `w2a_probe.py` and must be fixed **before probe
execution** (a later, separate authorization) but do **not** block the extract SMOKE, which exercises only
gate 0/1/4. The extract `SMOKE=1 --limit 1` job is cleared.

---

## 1. HASH-FREEZE VERIFICATION (all match; docs unchanged)

Recomputed on-disk sha256 (this review) vs `W2A_HASH_FREEZE_r2.md`:

| artifact | frozen sha256 (prefix) | on-disk | match |
|---|---|---|---|
| `scripts/analysis/w2a_extract.py`  | `2e79599a…` | `2e79599a92d227d9…` | ✓ |
| `scripts/slurm/w2a_extract.sbatch` | `9ed04c14…` | `9ed04c14d16799d2…` | ✓ |
| `scripts/analysis/w2a_probe.py`    | `72e25d24…` | `72e25d246890ecd2…` | ✓ |
| `research-wiki/experiments/exp-w2a-grounded.md` | `076bfa5e…` | `076bfa5eff14fe13…` | ✓ |
| `refine-logs/W2A_FORENSIC_RECON.md` | `fedc7e67…` | `fedc7e6726385be8…` | ✓ |
| `refine-logs/W2A_PREREG_REVIEW.md` | `b7f6ee09…` | `b7f6ee09bb4eaf69…` | ✓ |

`git diff cb59a94 -- <exp> <recon> <review>` is **empty** — the three governing docs are byte-identical to
r1. HEAD = 654720b. Hash-freeze record is accurate.

## 2. VERIFICATION PERFORMED THIS REVIEW (beyond source-read)

- **Extractor CPU self-test** (`--self_test`, no GPU/model): **PASS** — message-block order `[text,video,text]`,
  empty→`"(none)"` sentinel, synthetic span indexing (`locate_spans` `end`/`vis_pos`), `pool_grounded`
  (`grd`=vision-pad mean, `grd_pfx`=`[first_vis:end]`), `pool_control`, gate-0 raises on non-contiguous +
  grid-count mismatch, placebo pairing distinct/length-comparable.
- **Probe synthetic self-test** (`--self_test_only`, `CUDA_VISIBLE_DEVICES=""`): **PASS** — real
  `metrics.py` vote (planted grd beats noise concat), Fano=1.000, oracle≥concat, covered-subset path,
  rank-only path, and the full K9 conditional-info probe (calibration arm reached ≥0.99, informative-A
  `dacc=+0.5417`, `VERDICT=CONDINFO_PROCEED`, perm-null populated).
- **Cache contracts** (HateVideo `torch.load`, this review): CLIP img **1024** + CLIP text **768** + Qwen img
  **3584** + Qwen text **3584** = **8960-d** on **both** datasets/splits; CLIP↔Qwen ids **aligned** per split;
  train∪val = **851 / 629** exact. So `build_Zbest`'s `assert Z.shape[1]==8960` and the `EXPECTED_MEM` guard
  are satisfied by the real caches — the probe will neither crash on load nor mis-shape Z_best.
- **K9 machinery diff vs `c3_fusion_probe.py`:** faithful port (see Lesson 3).

---

## 3. PER-RESOLUTION RULINGS (implementer's 11 ambiguity resolutions)

### #1 — K5 oracle = per-query gold **grd-vs-CONCAT** (prereg §6.4 frozen constant), NOT grd-vs-Z_best. **RULING: IMPLEMENTER CORRECT. This is the binding reading. CONCAT stands; no Z_best-referenced form needed.**

This is the one ruling that could change the kill topology; I worked it through end to end.

- **The prereg constant governs.** §6.4 and the §16 hash-frozen constants define K5 verbatim as the
  per-query choice `choose grd iff (2y−1)·v_grd(Q) > (2y−1)·v_cat(Q)` (tie→CONCAT), report `Δ(oracle−CONCAT)`,
  DEAD if `Δacc < +0.04` on **every** dataset. `v_grd`/`v_cat` are **kNN LOO vote margins** from the real
  `metrics.py` vote. The orchestrator's task text ("grd-vs-Z_best") is not the frozen constant.
- **A kNN-margin oracle over Z_best is ill-defined — the implementer is right.** Z_best (8960-d) enters the
  design **only** as the K9 conditional-info **logistic** baseline; it produces per-video CV **correctness
  (0/1)**, not a signed cosine vote margin. There is no commensurable `v_Zbest(Q)` to plug into
  `(2y−1)·v`. A `predict_proba`-distance surrogate would be a supervised-fit quantity on a different scale
  from a LOO kNN cosine margin — mixing them in the per-query `>` comparison is apples-to-oranges. The
  oracle-ceiling is a *kNN* upper bound; its natural, well-defined baseline is the *kNN* CONCAT.
- **The 50/50 handicap does NOT corrupt the kill decision — it makes K5 harder to fire, which is the safe
  direction.** `Δ(oracle−CONCAT) = Pr[CONCAT wrong ∧ grd rescues]`. A handicapped (poor-geometry) CONCAT is
  wrong on *more* queries → *more* rescue opportunities → Δ **inflates** → K5 is **less** likely to fire
  DEAD. K5 is a conservative **kill-switch** (its only job is to avoid killing something with headroom), so
  a low false-kill rate is exactly what you want. The prereg states this explicitly ("inflates the oracle
  gap … it never wrongly kills"). Verified in code: `oracle_ceiling(a_grd["votes"], a_cat["votes"], y)` with
  strict `>` tie→concat (`w2a_probe.py:263-270`), `d_orc_acc = orc_acc − a_cat["acc"]`,
  `oracle_all_below = all(Δacc < 0.04)` (`:695`).
- **Why this doesn't corrupt anything given K9 is the sole binding perf gate.** A conservative K5 simply
  **defers to K9** more often. K9 (§K9) is weighting-invariant (its logistic head re-weights the img/text
  halves freely) and CLIP-augmented (Z_best) — it is immune to the 50/50 handicap and is the instrument that
  killed C3-nontarget. So the worst case of a handicapped K5 is a false-SURVIVE that hands the decision to
  the un-foolable gate — no head GPU is spent regardless (the whole Stage-P′ probe is CPU-only; the head
  stage §11 is unauthorized). Kill topology is sound.

Verified against the self-test: the noise-grd branch produced `Δ(oracle−concat)=+0.375` (kill branch
exercised, oracle ≥ concat by construction).

### #2 — Z_best = concat(CLIP img[1024], CLIP text[768], Qwen img[3584], Qwen text[3584]) = 8960-d. **RULING: CORRECT.** `build_Zbest` order matches the C3 record and §16 exactly (CLIP block first, then Qwen); `assert Z.shape[1]==8960` present; verified against the real caches (§2). Qwen-only 7168-d = secondary, point-arms-only (see #10).

### #3 — Empty-transcript block = un-labelled `"(none)"` (SPEC-AMBIGUITY 1). **RULING: ACCEPTABLE; correctly disclosed.** §4 zero-guard prose writes `"Transcript: (none)"` while §2/§4 message-content writes the block as `{text: transcript}` with the team-lead `[{text:"(none)"}]` fix. The implementer took the un-labelled `"(none)"` and flagged the divergence in-code. This matters only on the empty-transcript rows, which are **excluded from the binding grounding-live present-set median** and routed to the diagnostic empty-set distribution (`empty_gcos`), and whose grounded key is vacuous by design. The choice cannot affect the binding gate. Disclosed, defensible — ACCEPTABLE. (NOTE: the `img_control` forward has no transcript block at all, so this never touches the G-recon parity anchor.)

### #4 — grd_pfx span = `[first_vis:end]` (SPEC-AMBIGUITY 2). **RULING: CORRECT.** `end` = last `<|im_start|>` (assistant-header start), matching the banked `_encode` prefix boundary. `[first_vis:end]` = vision + trailing IMG_INSTRUCTION (+im_end), **excluding** the leading transcript — exactly §2/§4 "vision + trailing-instruction span". Confirmed by self-test (`exp_pfx` allclose).

### #5 (task #4/#5) — gates 2/3 = per-split aggregate **median VOID flags**, not per-video HALTs. **RULING: CONSISTENT with prereg §4.** §4 gate 2 is explicitly "…median cos ≥ 0.999 → silent no-op → probe **VOID** (recorded flag; the keys are still cached)" and gate 3 likewise a subset-median VOID; neither is a HALT. `assemble_split` computes present-set/placebo medians and records `grounding_void`/`placebo_void` booleans + full distributions in the gatelog (`:531-549`). Correct reading. **Caveat → see NON-BLOCKING B:** these VOID flags are surfaced in the gatelog JSON and the probe markdown, but are **absent from `mechanical_gate_check`**, whose `any_ci_pass` "SURVIVES" line therefore does not condition on a no-op grounding.

### #6 — placebo subset = first-50 present (gt order) + length-comparable partner. **RULING: SELECTION-BIAS-FREE (accept), with one cosmetic wrap-around NOTE.** The subset is the first `PLACEBO_N` present-transcript videos in **native gt order** (uncorrelated with grounding strength), and each partner is the char-length neighbour drawn from the **full** present set (`build_placebo_partners`), so length is held comparable while content is mismatched — exactly Amdt 3. No bias. **NOTE C:** the sort is *cyclic* (`order[(rank+1)%m]`), so the single longest present video is paired with the shortest — length-comparability breaks for that one pair. Harmless to the median VOID (≥50 subset) and in the safe direction (a bigger `grd` move lowers cos, away from the VOID trigger), but a non-cyclic nearest-by-|Δlen| partner would be cleaner.

### #7 — within-video token-shuffle secondary placebo NOT implemented. **RULING: WAIVE (acceptable), fix the comment.** §4/§12 list it as a **SECONDARY, non-gating diagnostic**; the **binding** cross-video mismatched placebo (K3) **is** implemented (`placebo_grd`). Omitting a non-gating diagnostic is fine. **NOTE D:** the module docstring (`w2a_extract.py:40`) and gate-3 doc say the token-shuffle is "kept as a secondary diagnostic," which overstates — it is *not* implemented. Change "kept" → "deferred (non-gating)" so the comment matches the code.

### #8 (task #10) — Qwen-only secondary conditional-info runs **without** perm null. **RULING: CORRECT, matches the c3 precedent.** `c3_fusion_probe.PERM_CELLS` runs the ≥150-perm null only on the Z_best cells; Qwen cells are point-arms-only. `conditional_info_probe(build_Zqwen(mem), …, run_perm=False)` → `VERDICT=SECONDARY_NO_PERM`, reported non-binding (`_ci_summary`, markdown). Faithful.

### #9 (task #11) — K9 checkpoint/resume config signature. **RULING: ADEQUATE for the intended single-frozen-run workflow, but under-covers a re-extraction footgun → NON-BLOCKING A.** `ci_meta = {grounded_dir, ci_nseed, Zbest_dim}` invalidates on a changed grounded-key version, perm count, or Z dimension — the three things that change across normal re-runs. Gaps: (i) it does **not** hash the **values** of Z_best/grd (a re-extraction into the **same** `grounded_dir` at fixed dim keeps the signature valid → both the cached `point` arms *and* the perm seeds are silently reused, because `_ci_point` is skipped when `"point" in cell`); (ii) it does not cover the covered-mask composition (keyed by cell name, not value); (iii) module constants (C_GRID/SCALE/KS/seeds) are not in the signature — but they are hash-frozen with the script, so a change re-hashes the script and the run is re-authored. The freeze discipline (single Stage-E′ submit, hash-pinned) makes (i)/(ii) unlikely, but it is a real defense-in-depth gap. Fix in NON-BLOCKING A.

### #10 (task, implicit) — grid-gate `spatial_merge_size` read from `model.config.vision_config.spatial_merge_size`. **RULING: CORRECT.** Not hardcoded to 2 (`_run_forward:242`), matching §16 ("read from the model config"). NOTE: gate-0 also reads `inputs["video_grid_thw"]` — a Qwen2.5-VL processor output with **no banked precedent** (the banked `_encode` uses only `im_start` spans). Wrong key/attr → fail-loud KeyError, not a silent pass; SMOKE validates it (see smoke terms).

### #11 (implicit) — grounded transcript block = raw text, no `"Transcript:"` label (`GROUNDED_TRANSCRIPT_PREFIX=""`). **RULING: ACCEPTABLE.** §2/§4 message content is `{text: transcript}` (raw); the banked `text_feats` `"Transcript:"` label lives in a *different* forward and is not part of W2-A's mechanism. Consistent with the prereg. NOTE: this is a deliberate departure from the banked text prompt and is correctly scoped to the grounded forward only.

---

## 4. PER-LESSON FINDINGS

### Lesson 1 — novel surfaces (deepest scrutiny)

- **Transcript-first message builder** (`_build_grounded_messages`): one user turn, content order
  `[{text:transcript},{video:frames},{text:IMG_INSTRUCTION}]`; `apply_chat_template(add_generation_prompt=
  True)` renders `…<|im_start|>user\n<transcript><vision…><instruction><|im_end|>\n<|im_start|>assistant\n`.
  Vision tokens sit **after** the transcript, **before** the assistant header → they causally attend back to
  the transcript (the mechanism) and remain a single contiguous block (gate-0 holds). Order/roles/IMG
  placement correct. The `"(none)"` branch is the only empty-case rendering and is excluded from the binding
  median. **Correct.**
- **Vision-span indexing** (`locate_spans`/`pool_grounded`/`pool_control`): `grd`=`_pool_idx_norm(vis_pos)`
  over `input_ids==video_token_id`; `grd_pfx`=`[first_vis:end)`; `ungrd_vis`=vision-pad pool of the
  img-control forward; `img_recon`=`[0:end)` prefix pool = the **banked** span. Walked with the synthetic
  layout `[IM,TXT,TXT,TXT][VID×6][TXT,TXT][IM,TXT]`: `end=8`, `vis_pos=[4..9]`, `grd_pfx` over `[4:8)`, all
  allclose. No off-by-one; `im_end` inclusion matches the banked `_encode` (both stop `end` at the *last*
  `<|im_start|>`). **Correct.**
- **im_end / `end` robustness to injected specials:** even if an ASR transcript literally contained
  `"<|im_start|>"`, `end=positions[-1]` is the assistant header (always last via
  `add_generation_prompt=True`), so `grd`/`grd_pfx`/`img_recon` spans are unaffected; and the banked
  `text_feats` forward already survived the full HateMM transcript distribution with the same boundary
  logic. Non-issue.
- **M-RoPE** handled as spec'd: `rope_vis_offset` = `vis_pos.min()` logged as a **diagnostic** only
  (`:245`, gatelog medians), never gates. Matches Amdt 6.
- **Placebo partner pairing:** deterministic, distinct, length-comparable (self-test-checked). See NOTE C.

### Lesson 2 — gate walk (HALT vs VOID, exit codes)

- **gate 0** (`grid_contiguity_gate`): grid-count `n_vis==grid_t·(grid_h//merge)·(grid_w//merge)` **and**
  single-contiguous-block `(hi−lo+1)==n_vis`; `raise RuntimeError` in **both** forwards → non-zero exit
  under `set -e`. Self-test confirms it raises on both failure modes. **HALT correct.**
- **gate 1** (`encode_video`): `img_recon` vs banked `img_feats[v]`, `cos<0.9999 OR max-abs>1e-3 → raise`;
  runs whenever `banked_vec is not None`, i.e. **under `--limit` smoke too** (`bvec=banked.get(vid)` from the
  real cache). Parity is genuine: img-control forward = banked `_build_messages(frames,IMG_INSTRUCTION)`,
  same processor call, same `[0:end)` prefix pool, same `.float()`→L2 path; only residual is A100 bf16 kernel
  drift (S2S precedent, within tol). **HALT correct.**
- **gate 4** (`_run_forward`): `last_hidden.shape[0]!=input_ids.numel() → raise` in both forwards. **HALT
  correct.**
- **gates 2/3** (`assemble_split`): computed as **aggregate median VOID flags** + full distributions in the
  gatelog; never HALT (matches Amdt 4/6). Recorded and surfaced to the probe (`stage_e_gatelog`) and markdown.
  **VOID recording correct** — but see NON-BLOCKING B for the missing mechanical-table surfacing.

### Lesson 3 — K9 machinery vs c3_fusion_probe precedent

Line-by-line, the K9 block is a faithful port of the **approved** C3 template:
- `_ci_pick_C` / `_ci_pick_C_combined` — StratifiedKFold(5,shuffle,rs=0), C_GRID=[.001,.01,.1,1], max_iter=2000 — identical.
- `_ci_baseline_cor` / `_ci_oracle_cor` / `_ci_full_cor` / `_ci_arm_cor_allk` — rs=1000+rep, 5×5, train-fold
  PCA sliced from kmax, aux ×`SCALE_A=50` (un-penalized) at `C_Z` — identical.
- **label-oracle calibration** appends 2-col one-hot(y)×50, `PASS = accZA≥0.99` else `MACHINERY_INVALID` —
  reaches full Fano headroom; matches REFLECTION §4. Self-test hit PASS.
- **+0.040 triple rule:** C1 `best_dacc≥CI_BAR(0.040)`, C2 per-video bootstrap `ci[0]>0` (B=5000, resample of
  the per-video correctness vector = example-clustered), C3 `real_max_over_kdec > all` of the ≥150-perm
  **max-over-k** distribution — the family-corrected form, verbatim from C3.
- **null as a DISTRIBUTION** with fresh seeds `CI_PERM_BASE(70000)+si`, resumable, checkpointed every 10.
- **Z_best composition** 8960-d verified against real caches; **covered-rows** variant slices `Z[sel]/A[sel]/
  y[sel]` and runs `run_perm=True` (Amdt 5 requires the binding Δ on covered rows — correct).
- Only cosmetic divergences: `CI_BOOT_SEED=20260715` (c3 used 20260714) — arbitrary bootstrap seed, no parity
  meaning; and w2a **reuses** the cached baseline (`pt["base"]`, `pt["C_Z"]`) in the perm null instead of
  recomputing it as c3 does — deterministically equivalent (same rs) and strictly safer (one base for point
  and perm). **No defect.**

### Lesson 4 — imports / handoff / fail-loud / determinism / no-test-read / sbatch

- **Deferred imports:** `transformers`/`AutoProcessor`/`Qwen2_5_VL…` imported **inside `main()`** after the CPU
  self-test (`w2a_extract.py:681`); `compute_metrics_retrieval` imported once at probe top (the real vote).
  Parity-by-import loads only forward-neutral helpers (`_build_messages`, `read_gt`, `load_video_frames`,
  `IMG_INSTRUCTION`, `SPLIT_TO_OUTNAME`) — the novel `_build_grounded_messages`/pools are **re-authored**, as
  the review required.
- **Handoff chain:** extractor writes per-split `{outname}_grounded.pt` + `{outname}_gatelog.json` under
  `<out_root>/<ds>/grounded_qwen7b_<F>f/`; probe reads exactly those paths via the shared
  `GROUNDED_DIR_TMPL` and `_grounded_path`. Consistent. `_load_bank` reorders banked CLIP/Qwen to the
  grounded ids (KeyError if an id is missing — fail-loud). Aligned.
- **Fail-loud:** no bare `except` around any forward or gate. The only `except` is `shard_ok`'s narrow
  `torch.load` guard (corrupt/truncated shard on resume → recompute, the safe direction) — documented,
  not around a forward. A-line lesson honored.
- **Determinism:** `set_determinism` (random/np/torch, seed 20260715); probe seeds np/torch + per-stage RNGs
  (`default_rng`). no_grad + eval + bf16 + sdpa.
- **No-test-read guard + size asserts:** `_guard_outname` asserts `outname in ("train","dev_seen")` on
  **every** grounded/bank path; `load_memory` never iterates `test_seen`; `EXPECTED_MEM` size guard fail-
  closed (851/629, verified). Zero-guard rows carried with a `-2` sentinel in `grounding_cos` and excluded
  from present/empty medians; near-dup `valid` mask excludes guard rows. **Compliant.**
- **sbatch:** `--gres=gpu:a100:1` (matches the banked producer, justified in-comment), `--cpus-per-task=8`
  ≤16, `--mem=64G` ≤128, **no `--time`**, `set -euo pipefail`, `conda activate HateVideo`, `HF_HUB_OFFLINE=1`,
  `sha256sum` of extractor+sbatch+parity-source echoed, `nvidia-smi` banner, SMOKE=1 → `--limit 1` to a
  throwaway `--out_root` under `slurm/logs/`, both datasets sequentially in one job, `JobHeldUser`→wait noted.
  `disk_guard.sh` exists. **Correct.**

### Lesson 5 — EXECUTION ROUTING

- The extractor is the single **local A100** GPU job (raw video, license-sensitive) — correctly local, via
  `w2a_extract.sbatch`.
- The **probe is multi-hour CPU** (≥150-perm null × 5×5 logistic on 8960-d Z_best over 851 rows, ×2–3 cells).
  It **must not** run on the login shell (reaping). Its inputs are **features-only, derived floats** — the
  grounded `.pt` keys plus the banked CLIP/Qwen `.pt` (no raw video) — so it is **Modal-compatible** and
  equally runnable as a **CPU-only SLURM job** (no `--gres`). **Recommended routing:** a dedicated CPU SLURM
  sbatch (no GPU, no `--time`) **or** a Modal CPU app; never `python w2a_probe.py` on the login node.
  **There is no probe runner in the frozen set** (only `w2a_extract.sbatch`) — see remaining authorizations.
- `CUDA_VISIBLE_DEVICES=""` and BLAS thread caps are set at probe import time (cloud-friendly). Good.

---

## 5. DEFECTS

### BLOCKING — none.

### NON-BLOCKING (fix before **probe execution**; do NOT block the extract SMOKE)

- **A — K9 checkpoint signature under-covers re-use (`w2a_probe.py:930-947`, `_ci_point:497`).** A probe
  re-run into the same `grounded_dir` at fixed `Zbest_dim` silently reuses cached `point` arms **and** perm
  seeds. **Fix (pick one):** (i) add the extractor script/grounded-cache identity to `ci_meta` — e.g.
  `"grd_sha": sha256(train_grounded.pt)+sha256(dev_seen_grounded.pt)` and `"probe_sha": sha256(__file__)`;
  **or** (ii) have the probe runner `rm -f refine-logs/w2a_ci_ckpt.json` at the start of the authoritative
  run and only resume within a single interrupted run. (ii) is the simplest and sufficient given the freeze
  discipline.
- **B — grounding/placebo VOID absent from `mechanical_gate_check` (`w2a_probe.py:680-745`).** The binding
  K2/K3 VOID flags are in the gatelog + markdown, but the one-glance mechanical table (and its terminal
  echo) reports `CondInfo BINDING (any dataset PROCEED) → SURVIVES` **without** conditioning on a no-op
  grounding — the exact "silent no-op" trap the design exists to prevent. **Fix:** add two rows per dataset
  to `mechanical_gate_check` reading `results[…]["stage_e_gatelog"][split]["grounding_void_present_median_ge_
  0.999"]` and `["placebo_void_median_ge_0.999"]` (result `VOID`/`LIVE`), and append to the K9 note
  "VOID nullifies K9". (`mechanical_gate_check` is explicitly non-binding, but the verdict reviewer scans
  this table — surfacing it here removes a foot-gun.)
- **C — placebo cyclic wrap pairs longest↔shortest (`build_placebo_partners:325-334`).** Harmless to the
  median VOID and in the safe direction, but replace the cyclic successor with the nearest-by-|Δlen| partner
  (or drop the wrap for the extreme ranks) for a clean length control. *Extraction-side, but it does not run
  under `--limit 1` smoke, so it does not gate SMOKE.*
- **D — comment overstates the token-shuffle placebo (`w2a_extract.py:40`, gate-3 docstring).** Change "kept
  as a secondary diagnostic" → "deferred (non-gating secondary)" to match the code. *Doc-only.*

### NOTE (no action required to proceed)

- Gate-0 depends on `inputs["video_grid_thw"]` (no banked precedent) — validated by SMOKE (fail-loud if
  wrong).
- Empty-transcript sentinel `"(none)"` (#3) and un-labelled grounded block (#11) diverge from the banked
  `text_feats` prompt by design; both correctly scoped to the grounded forward and both excluded from / not
  touching the binding gate.
- The covered-rows K9 runs a **second** ≥150-perm null (HateMM covered cell) — expected per Amdt 5; budget
  the probe runtime accordingly.
- `retrieved_scores` are `np.float64` scalars; `metrics.py:266` calls `.item()` on each — valid (verified in
  the self-test that drives the real vote).

---

## 6. SMOKE TERMS (single `SMOKE=1` sbatch; what must show GREEN)

`SMOKE=1 sbatch scripts/slurm/w2a_extract.sbatch` (→ `--limit 1`, `--out_root` throwaway under
`slurm/logs/w2a_smoke_out_<jobid>`). The job passes iff **all** of:

1. **Config echo + provenance:** the extractor banner prints `script sha256`, dataset/splits/frames/device/
   limit/out_root, model + `max_pixels=151200` + dtype bf16 + attn sdpa + transformers version, and the
   sbatch prints `sha256sum` of `w2a_extract.py`, `w2a_extract.sbatch`, and
   `generate_VideoMLLM_embedding_HF.py` (the parity-by-import source).
2. **CPU self-test PASS** ("[self_test] PASS …") before model load.
3. **gate 0** — no `[gate0 grid/*]` or `[gate0 contiguity/*]` RuntimeError in either forward, for both
   datasets (validates `video_grid_thw` + `spatial_merge_size` + vision-pad contiguity on real inputs).
4. **gate 1 G-recon-IMG** — at least one non-guard video per dataset prints a finite `grecon_cos` and the
   gatelog shows `grecon_cos_min ≥ 0.9999` and `grecon_maxabs_max ≤ 1e-3` (the img-control-vs-banked parity
   anchor; **must** fire — if `--limit 1` lands on the single undecodable HateMM row it is zero-guarded and
   gate 1 is skipped, so confirm `grecon_n_checked ≥ 1` per dataset; if 0, re-run SMOKE with `--limit 3`).
5. **gate 4** — no `[gate4 len-parity/*]` error.
6. **No writes to the real cache path** — all artifacts under the throwaway `w2a_smoke_out_*`; nothing under
   `data/CLIP_Embedding/*/grounded_qwen7b_8f/`.
7. Job **exits 0** (under `set -e`, reaching `end=…` means success).

The ≥50-video placebo (gate 3) and the grounding-live present-set median (gate 2) do **not** exercise under
`--limit 1` (expected — real-run gates); their code paths are covered by the two self-tests.

## 7. SEPARATE AUTHORIZATIONS THAT REMAIN

1. **Full Stage-E′ extraction** (`sbatch scripts/slurm/w2a_extract.sbatch`, no SMOKE) — local A100, single
   submit, no `--time`, `JobHeldUser`→wait-never-force. Authorize **after** a green SMOKE + re-freeze of the
   three hashes.
2. **Probe execution** (`w2a_probe.py`) — **no runner exists yet**; author a **CPU-only SLURM sbatch (no
   `--gres`, no `--time`) or a Modal CPU app** (features-only inputs → Modal-eligible), never the login
   shell. Land NON-BLOCKING A + B first. Then an independent **verdict review** (raw-only transcription)
   renders the binding DEAD/PASS ruling — and must cross-check the gate-2/gate-3 VOID flags before honoring
   any K9 PROCEED.
3. **Head-training formal stage (§11)** — remains **NOT authorized**; separate prereg behind the K5/K9
   outcome.

---

## §RE-CHECK — r2b (commit 2bf00cb, one-line-diff scope)

**Verdict: CLEARED-FOR-PROBE-EXECUTION.** Both NON-BLOCKING fixes A and B land correctly; no remaining
blockers. Probe sha256 = `af4a2f9f5b35461173fd82c176bd52c6fc84bf8fc0d09736f938d38d8f6fe06d` (matches the
new hash-freeze). Extractor `2e79599a…` + sbatch `9ed04c14…` are **byte-unchanged** since 654720b
(`git diff 654720b 2bf00cb -- <extractor> <sbatch>` empty).

**Diff scope confirmed — only `w2a_probe.py` + `W2A_HASH_FREEZE_r2.md`; three code hunks, no drive-by
edits, no binding computation touched.** load_memory, run_vote, oracle_ceiling, the entire K9 `_ci_*`
machinery, build_Zbest, probe_dataset, and write_markdown are identical to r2. The only new logic is the
checkpoint signature (A) and the reporting/surfacing layer (B); the `datasets` hoist in `main()` is a pure
refactor (compute the split list once, reuse it in the grd_sha comprehension and the loop). Probe
synthetic self-test re-run: still **PASS** (parse + machinery intact).

**Fix A — checkpoint signature hardening: CORRECT, and it INVALIDATES (not merely records).** `ci_meta` now
carries `probe_sha = sha256(this script)` and `grd_sha[ds] = sha256(train_grounded.pt)+sha256(dev_seen_
grounded.pt)` per dataset. The resume/fresh gate is unchanged (`if loaded["_meta"] == ci_meta: resume else:
start fresh`), so any mismatch — a probe edit OR a re-extraction into the same `grounded_dir` — makes
`loaded["_meta"] != ci_meta`, drops to the else branch, and reinitialises `ci_ckpt = {"_meta": ci_meta}`
with empty cells → `_ci_point` recomputes and the perm null restarts. The r2 footgun (silent reuse of
cached point-arms + perm seeds after re-extraction) is closed. `grd_sha` hashes only `train`/`dev_seen` via
`_grounded_path` (which asserts `_guard_outname`), so it respects the N4 no-test-touch guard. If a grounded
cache is absent at probe start it fails loud (the probe can't run without it anyway).

**Fix B — K2/K3 VOID surfacing + nullification: CORRECT (verified on synthetic rows).** `_stage_e_void`
reads the extractor gatelog for the two probe memory splits (train + dev_seen) and returns True if EITHER
split tripped the flag, False if present-and-untripped, None if no gatelog carries it. Confirmed the prefix
match is exact (`"grounding_void"` → `grounding_void_present_median_ge_0.999`; `"placebo_void"` →
`placebo_void_median_ge_0.999`; no collision with `grounding_present`/`placebo`/`placebo_rows`). Synthetic
exercise:
- A dataset with `VERDICT=CONDINFO_PROCEED` but `grounding_void=True` → its K9 row relabelled
  **`VOID(K2/K3-nullified)`**, and the aggregate `CondInfo BINDING (any LIVE dataset PROCEED)` = **BELOW**
  (SURVIVES=False) — the VOID PROCEED cannot count.
- Add a LIVE dataset that also PROCEEDs → aggregate **SURVIVES=True**.
- New per-dataset `GroundingLive[ds] (K2)` and `Placebo[ds] (K3)` rows emit VOID/LIVE/N/A.
This closes the r2 "silent no-op reads as PASS" gap — the exact trap the design exists to prevent.
*Soft spot (NOTE, non-blocking):* a **missing** gatelog yields `None`, which `bool(None)=False` treats as
not-voided, so an N/A dataset with a PROCEED would still count toward SURVIVES. In a completed extraction
the gatelog always carries both flags (so this is unreachable), and the N/A is **surfaced** in the
mechanical table for the verdict reviewer — acceptable for a non-binding table. The independent verdict
reviewer must still confirm both flags read `LIVE` (not `N/A`) before honoring any K9 PROCEED.

**Ruling on the C/D deferral: ACCEPTABLE — the team lead's read is right.** C (placebo cyclic-wrap) is
extractor-side, harmless and safe-direction on the median VOID, and does not run under the `--limit 1`
smoke; D is a doc-only comment. The extractor is byte-frozen at `2e79599a` and the PENDING smoke job 13166
reads that exact file at execution time, so editing it now would (a) break the hash the smoke is validating
and (b) mutate a file a queued job will run — strictly worse than deferring. C/D correctly ride the
post-smoke re-freeze cycle; neither touches any binding number. **Requirement:** C/D must actually land in
that post-smoke extractor re-hash (don't drop them).

**Remaining authorizations after this re-check:** unchanged from §7, except probe execution is now
**cleared** to be wired into a CPU-only SLURM sbatch (no `--gres`/`--time`) or a Modal CPU app (features-
only → eligible), never the login shell; the independent raw-only verdict review still gates the DEAD/PASS
ruling and must cross-check the K2/K3 rows read LIVE.

---

## §r2c — extractor fixes C+D re-check (commit 9470b64, post-green-SMOKE re-freeze)

**Verdict: CLEARED-FOR-STAGE-E′.** Fixes C and D land exactly as specified; no blockers; the green SMOKE
(job 13166, 7/7) **carries over — no GPU re-smoke required**. New extractor sha256 =
`9e984d61e2bf91d58f15af5e54f14d45a3fabe4e0701ce4492645399d810fa31`; sbatch `9ed04c14…` + probe `af4a2f9f…`
**byte-unchanged** since r2b (`git diff 2bf00cb 9470b64 -- <sbatch> <probe>` empty).

**Diff scope — exactly 2 hunks in `w2a_extract.py`, no drive-by edits.**
- **D (comment, doc-only):** module docstring gate-3 line "…kept as a secondary diagnostic." → "…deferred
  (non-gating secondary; not implemented)." Matches the code (the within-video token-shuffle placebo is not
  implemented).
- **C (`build_placebo_partners`):** cyclic-successor partner → **nearest-by-|Δlen| among the ADJACENT ranks
  in the length-sorted order, NON-cyclic** (candidates = `[rank−1, rank+1]` in-bounds only; tie → later
  rank). This is precisely the fix my r2 review specified. Verified on synthetic data (60 present + 5
  empties): the **longest transcript is never paired with the shortest** (no wrap); every partner is a
  length-**adjacent** rank (`|rankΔ|==1` → length-comparable); all `pid≠vid` (distinctness holds for m≥2,
  because non-cyclic adjacents are distinct entries ⇒ distinct ids); output is **deterministic**
  (rebuild-identical); the `len(present)<2 → {}` guard is unchanged. Extractor CPU `--self_test` (which
  exercises `build_placebo_partners` in case (e)) re-run: **PASS**.

Nothing else changed semantically — the forward runner, `pool_grounded`/`pool_control`, gate 0/1/4, the
message builders, `encode_video`, `placebo_grd`, shard I/O, and `assemble_split` are byte-identical. The
`W2A_HASH_FREEZE_r2.md` edit is record-keeping only (records the new extractor sha, notes sbatch/probe
unchanged, and the carry-over rationale).

**Carry-over ruling: ACCEPT — no GPU re-smoke.** Verified from the diff that **no smoke-exercised code path
changed.** The SMOKE (`--limit 1`) validated model load, gate 0 (grid+contiguity), gate 1 (G-recon-IMG),
gate 4 (len-parity), the grounded + img-control forwards, and the four keys (grd/grd_pfx/img_recon/
ungrd_vis) — all in code the r2c diff does not touch. The only functional change is gate-3 placebo **partner
selection**, whose output is consumed solely via `partners`/`placebo_subset`; under `--limit 1` that subset
is empty (`len(partners) < 2 → placebo_subset = set()`), so the changed logic is **never exercised in the
smoke path** and the ≥50-subset placebo is a real-run-only gate regardless. The one startup touch of the
changed function is the **CPU** `self_test` correctness assertion, which I re-ran (PASS). A GPU re-smoke
would only re-validate byte-identical gates 0/1/4 + G-recon — pointless spend; the CPU self-test fully
covers the changed logic. **No re-smoke required.**

**All prior clearances stand:** extractor APPROVED (r2) → probe CLEARED-FOR-PROBE-EXECUTION (r2b) → extractor
**CLEARED-FOR-STAGE-E′** (r2c). The single Stage-E′ submit runs the r2c extractor `9e984d61…` (local A100, no
`--time`, `JobHeldUser`→wait-never-force); probe execution and the independent raw-only verdict review (which
must confirm the K2/K3 rows read LIVE before honoring any K9 PROCEED) remain the downstream gates.
