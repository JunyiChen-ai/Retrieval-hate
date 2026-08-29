# S2S Stage-E/Stage-P — Independent Code Review

**Reviewer:** fresh zero-prior-context independent code reviewer (read-only; NO GPU / NO SLURM / NO
submissions; NO edits to the reviewed files). **Date:** 2026-07-14.

**Under review (r1):**
- `scripts/analysis/s2s_extract.py` — Stage-E per-frame-group Qwen2.5-VL extractor (HateMM + MHC-EN).
- `scripts/slurm/s2s_extract.sbatch` — Stage-E sbatch.
- `scripts/analysis/s2s_probe.py` — Stage-P CPU zero-training G0-cond probe.

**Governing specs:** `research-wiki/experiments/exp-s2s-r3.md` (r1), `refine-logs/S2S_PROBE_DESIGN.md`
(r1, incl. §12 spec-ambiguity resolutions + the G-decomp-f32 / G-recon-bf16 dtype note),
`refine-logs/S2S_PREREG_REVIEW.md` (A1–A5 amendments). **Reference:**
`src/utils/generate_VideoMLLM_embedding_HF.py` (banked producer, parity-by-import),
`src/utils/metrics.py` `compute_metrics_retrieval` (`use_sim` vote path, reused not reimplemented).

**Method:** static walk of every gate with concrete values; deferred-import + env audit against the
HateVideo site-packages; full-chain handoff trace (sbatch → extract → shards → probe); parity diff of
the extractor forward against the banked `_encode(span="prefix")`; the probe's entire vote/oracle/
near-dup/null/bootstrap pipeline **executed end-to-end on synthetic data** (it drove the REAL
`utils.metrics.compute_metrics_retrieval`); the cross-device hypothesis for G-recon tested directly in
torch.

**VERDICT: APPROVED AFTER FIXES.** Structure is sound and parity-by-import is genuine, but the
extractor **crashes on the first real video** (G-recon compares a CUDA tensor to a CPU tensor —
confirmed cross-device ops raise `RuntimeError`), and the mandated `--limit 1` smoke as wired **cannot
exercise G-recon** (so it cannot catch that crash). One probe defect (A2 rank-only corroboration is
under-implemented) must land before Stage P. All three are surgical. The current hashes are **NOT
submittable**; re-hash after the fixes.

---

## 1. Hash-verification (on-disk vs `S2S_PROBE_DESIGN.md` §10 r1 freeze) — ALL MATCH

| artifact | frozen sha256 (r1) | on-disk sha256 | match |
|---|---|---|---|
| `research-wiki/experiments/exp-s2s-r3.md` | `ffcc3a67…06e3fe5b` | `ffcc3a679b628e32d600bc0ecaeda0a0d7ac2d8da6387fbc6de2143506e3fe5b` | ✅ |
| `scripts/analysis/s2s_extract.py` | `91637ccd…059c4b74` | `91637ccd52ce8fec5e29c8c8a30621e046d19aca3a5096d7e116beab059c4b74` | ✅ |
| `scripts/slurm/s2s_extract.sbatch` | `818e0cd2…dddb47561` | `818e0cd2f865fcb17e864e2519574abcb9ce3f8894b15d7a72e2811dddb47561` | ✅ |
| `scripts/analysis/s2s_probe.py` | `53c6a6c8…4753c9c4` | `53c6a6c80e3db336070521c5cf3daf5877e82cb24cfbb749162862ad4753c9c4` | ✅ |

All four reviewed artifacts are the r1-frozen bytes. (Any fix below produces a mismatch — expected, and
the §10 table is re-pinned at that point per its own note.)

---

## 2. House-lesson findings

### Lesson 1 — Runtime cross-check simulation of every gate
Walked each gate with concrete values; every gate raises `RuntimeError` on violation and the exception
is **uncaught**, so it propagates to a non-zero process exit (real HALT, not a print). Specifics:

- **Gate 0a temporal positive control** (`s2s_extract.py:239-280`) — SOUND and it is the *only* gate
  that exercises the token→temporal-group assignment. Clip A frame-pairs carry colours `[0,1,2,3]`;
  clip B carries `σ=[2,0,3,1]`. `M[i,j]=cos(A_i,B_j)`; `match=M.argmax(dim=0)` must equal `σ`
  (`:266-267`). If grouping were spatial-major/scrambled the per-group colour identity collapses and
  either `match≠σ` or the within-clip distinctness check (`off.max()>0.999`, `:274-278`) fires — both
  HALT. Runs before any real video (`:496`). Corroborated the temporal-major layout it relies on in the
  installed source: `modeling_qwen2_5_vl.py` `get_window_index` (`:466-505`), window reorder (`:530`),
  `reverse_indices=argsort(window_index)` (`:561-562`) → merged tokens are (t,h,w) row-major, each
  temporal group a contiguous block. ✅
- **Gate 0b grid-consistency** (`:164-180`) — asserts `n_vis == grid_t·(grid_h//merge)·(grid_w//merge)`
  and `(n_vis//T)==per_expected` with `merge=model.config.vision_config.spatial_merge_size`. Verified
  `vision_config` is a real sub-config (`configuration_qwen2_5_vl.py:32,175`) and `spatial_merge_size`
  defaults 2. Strictly stronger than `n_vis % T == 0`, matches spec §2 exactly. HALT. ✅
- **Gate 1 G-decomp** (`:198-203`) — `L2norm((Σ n_t·g_t + p_S)/end)` vs the **f32** prefix-mean
  (`banked_formula_vec`, `:151`), max-abs ≤ 1e-5. Both sides are pure f32 from `prefix=last_hidden[:end]
  .float()` (`:150`) → exact identity up to summation order. HALT. ✅
- **Gate 2 G-recon** (`:205-216`) — **DEFECT, see BLOCKING-1.** Logic (cos ≥ 0.9999 AND max-abs ≤ 1e-3
  vs banked `img_feats[v]`) is correct, but the two tensors live on different devices → it raises a
  device error instead of running.
- **Probe N4 no-test guard** (`s2s_probe.py:71-83,113-116`) — `_frameset_path`/`_bank_path` assert
  `outname ∈ {train,dev_seen}`; `load_memory` loops only those; size guard `N==851/629`. Un-bypassable.
  Exercised on synthetic. ✅
- **Synthetic set-matching self-test** (`s2s_probe.py:349-368`) — planted shared frame gives
  `MMS>POOLED`; ran, passed (`0.2594 > 0.2479`). HALT on failure. ✅

### Lesson 2 — Deferred-import / env audit (HateVideo site-packages, read-only)
All resolvable: `importlib.util` (stdlib, loads `src/utils/generate_VideoMLLM_embedding_HF.py`),
`transformers 4.49.0`, `Qwen2_5_VLForConditionalGeneration`/`AutoProcessor`, `decord 0.6.0`, `av 17.0.0`,
`numpy 1.26.4`, `torch 2.6.0+cu124`, `PIL 11.1.0`. Probe's `from utils.metrics import
compute_metrics_retrieval` pulls `metrics.py`'s top-level `import wandb / torchmetrics / easydict /
pandas / sklearn` — **all present** (`wandb 0.28.0`, `torchmetrics 1.9.0`, `easydict`, `pandas 2.3.3`,
`sklearn 1.5.2`), and the deferred `from sklearn.metrics import f1_score` in `bootstrap_delta` resolves.
`inputs["video_grid_thw"]` is a documented processor output key (`processing_qwen2_5_vl.py:114,131`). ✅

### Lesson 3 — Full-chain handoff / resumable integrity
sbatch passes `--dataset/--splits/--num_frames/--out_root/--device`; extractor writes
`data/CLIP_Embedding/<ds>/frameset_qwen7b_8f/{train,dev_seen,test_seen}_frameset.pt` + `<outname>_gatelog
.json`; probe reads `frameset_dir` default `frameset_qwen7b_8f` (matches) and the banked
`{train,dev_seen}_Qwen2.5-VL-7B-Instruct_HF.pt`. All inputs verified on disk (gt train/val/test for both
datasets, `data/video/HateMM/All`, all four banked anchor caches). **Atomicity:** `atomic_save`
(`:325-329`) writes `.tmp` then `os.replace` (atomic rename) — a partial write never becomes the final
path. **Corrupt-shard safety:** `shard_ok` (`:304-322`) requires existence + all keys + `g.shape==(grid_t
,3584)` + stored `decomp_res≤1e-5`; a truncated shard raises inside the one sanctioned narrow `except`
(`:310-312`) which prints and returns False → recompute (safe direction). A shard can only exist if its
video passed G-decomp+G-recon (both HALT before `atomic_save`), so skip-if-exists cannot admit a failed
video. Assembly re-loads every item's shard in gt order and re-asserts constant T (`:420-423`) → HALT if
not. **No test-path leak constructible** in the probe (traced). ✅

### Lesson 4 — Fail-loud
Extractor: no bare/broad except wraps the forward or gates 0a/0b/1/2 — all raise and propagate. The only
broad except is the sanctioned corrupt-shard reload (`:310`), printed and safe-direction. The imported
`load_video_frames` decode-fallback excepts are pre-existing banked code and resolve to the zero-guard
policy, not a swallowed OOM. sbatch is `set -euo pipefail`. Probe: no silent excepts. ✅

### Lesson 5 — Parity-by-import (diff vs banked `_encode(span="prefix")`)
The extractor forward is **byte-identical** to the banked one: same imported `_build_messages` +
`IMG_INSTRUCTION` (`:124`), same `apply_chat_template(add_generation_prompt=True)` + `processor(text,
images=None, videos=[frames])` (`:125-127`), same `model(**inputs, output_hidden_states=True,
use_cache=False)` (`:129`), same span-end = last `<|im_start|>` (`:140-143`), same model construction
(`bfloat16/sdpa/device_map=None`, `:489-491`) and `AutoProcessor.from_pretrained(MODEL,
max_pixels=151200)` (`:493`), same imported `load_video_frames`/`_sample_frame_indices`/`read_gt`/
`SPLIT_TO_OUTNAME`. **Below-tolerance divergence check:** none found — the G-recon vector
(`grecon_pooled=last_hidden[:end].mean(0)` bf16 → `.float()` → normalize, `:146-147`) replicates the
banked `:303,321-322` line-for-line (bf16 accumulation, not the f32 accumulation used for G-decomp). The
implementer's precision note (header `:34-44`) is **accurate**: G-decomp is an f32-vs-f32 exact identity,
G-recon is bf16-vs-banked-cache — the correct split. `set_determinism` doesn't touch the (dropout-free,
sampling-free) forward, so it doesn't perturb parity. ✅ (gated by the G-recon device fix landing).

### Lesson 6 — Vote reuse (real `compute_metrics_retrieval`, symmetric `use_sim`)
`run_vote` (`s2s_probe.py:166-190`) builds `logging_dict` and calls the REAL
`compute_metrics_retrieval(..., majority_voting="arithmetic", topk=20, use_sim=True)` — 8-value unpack
matches `metrics.py:320`. **Executed on synthetic data: it drives the real metrics fn** (module
`utils.metrics`), Fano ±1 key → acc 1.000 (machine-validity converts), oracle > pooled. `retrieved_scores`
= `list(np.float64[...])`; the `use_sim` path's `sim.item()` (`metrics.py:266`) works on np.float64
(verified). **Rank-only** neutralises sim to constant `1.0` (`:182`) while still retrieving by the arm's
own score → both arms weight identically, differ only in *which* neighbours (correct A2 de-confound). The
**permutation null applies the SAME permutation to both arms** within a seed (`:277-283`, `ix=np.ix_(perm,
perm)` on both `mms` and `spool`, labels fixed = shuffle framesets across label slots) — N1 compliant.
**But the A2 corroboration is only half-built — BLOCKING-3.**

### Lesson 7 — Set-metric + oracle correctness
- **MeanMaxSim** (`build_matrices:145-147`): `Sff.max(axis=3).mean(axis=1)` = mean over query frames of
  max over memory frames, direction Q→M, on L2-normed frame vecs (`ghat`, `:143`), T from `grid_t`.
  Matches spec §5 exactly. Chamfer `0.5(mms+mms.T)` correct (`:388`).
- **Oracle A4** (`:200-234`): margin `V[i,t]` inlines the exact `metrics.py:262-284` weighted signed-sim
  vote (verified term-by-term: `lab_signed·row[idx]·weight/wsum`, `weight=arange(1,k+1)[::-1]`); frame
  select `t*=argmax_t (2y_i−1)·V[i,t]` with `np.argmax` (smallest-index tie-break) (`:228-229`); memory
  keeps full sets (`:232`); the FINAL oracle number uses the REAL vote (`run_vote(Sorc)`, `:233`).
  **Video-level gold only** — `labels` are video labels; **no time-span / per-frame annotation is read
  anywhere** in either script (grep-confirmed: gold enters only via `labels` in Fano + oracle). Matches
  A4 verbatim. ✅

### Lesson 8 — sbatch
`conda activate HateVideo` (`:25`); single GPU `--gres=gpu:a100:1` (`:3`); **no `--time`** (`:8`);
`PENDING (JobHeldUser)=wait` noted (`:9`); prints config echo + `sha256sum` of both scripts (`:38-39`);
SMOKE=1 → `--limit 1 --out_root <throwaway>` (`:44-51`) vs full run (`:52-58`); one job runs both
datasets sequentially. Log path pinned. Two issues: the smoke path **disables G-recon** (BLOCKING-2), and
the `a100`-typed gres differs from siblings' `gpu:1` (NON-BLOCKING-b).

### Lesson 9 — false-PASS / false-KILL / silent-corruption surfaces
The frameset cache cannot be silently corrupted (atomic write + integrity check + HALT gates before
save). The false-KILL surface is BLOCKING-1 (a device crash masquerading as the run). The near-dup
exclusion has a theoretical NEG_INF-leak (NON-BLOCKING-a). The A2 gap (BLOCKING-3) is a false-PASS/KILL
surface for the *verdict* (the pre-registered corroboration cannot be applied from the numbers produced).

---

## 3. Defects

### BLOCKING

**B1 — G-recon compares a CUDA tensor against a CPU tensor → `RuntimeError` on the first real video.**
`s2s_extract.py:147` `grecon_vec = F.normalize(grecon_pooled.float(), …)` is on `device` (CUDA in the
real run); `:209` `bv = banked_vec.float()` is on CPU (banked loaded with `map_location="cpu"`, `:291`).
`:210` `F.cosine_similarity(grecon_vec, bv, dim=0)` and `:211` `(grecon_vec - bv)` are cross-device. I
**tested torch directly**: cross-device `cosine_similarity` and subtraction both raise
`RuntimeError: Tensor on device … is not on the expected device …`. So the real (`--limit`-free)
extraction crashes on the first banked-matched video — a hard failure that looks like a gate error, not a
parity result. It manifests ONLY on CUDA and ONLY when `banked_vec is not None`, so a `--device cpu`
dry-run or the current smoke would not surface it.
**Fix:** compare on one device, e.g. `gv = grecon_vec.detach().cpu()` then
`F.cosine_similarity(gv, bv, dim=0)` / `(gv - bv).abs().max()` (or `bv = banked_vec.float().to(grecon_vec
.device)`).

**B2 — the mandated smoke cannot exercise G-recon, so it cannot catch B1.**
`s2s_extract.py:344` `banked = None if args.limit else load_banked_imgfeats(...)` disables G-recon
whenever `--limit` is set; the smoke is `--limit 1` (sbatch `:48-49`), so gate 2 never runs in the smoke.
This follows `S2S_PROBE_DESIGN.md §3` ("G-recon skipped" in smoke) but **contradicts the governing
authorization** `S2S_PREREG_REVIEW.md §5 condition (iii)`: *"a `--limit 1` smoke must show all HARD gates
(A1 grid gate, G-decomp, G-recon, temporal positive control) green on a throwaway path before the real
run."* The first `--limit 1` video's id IS in the banked cache, so G-recon can run in the smoke.
**Fix:** decouple G-recon from `--limit` — load the banked cache and run G-recon on the limited
video(s) too (keep the throwaway `--out_root`). This makes the smoke fulfil its pre-registered purpose
and is the only way to prove B1 is fixed before the full run.

**B3 — A2 rank-only corroboration is under-implemented (probe).** The pre-registered A2 rule
(`exp-s2s-r3.md:215-223`, `S2S_PROBE_DESIGN.md:269-281`) credits the primary Δ only if the rank-only Δ
"matches its sign **AND is itself significant** (observed Δ > 95th-pct permutation null AND bootstrap
5th-pct > 0)". The probe computes only `rankonly_corroborates_sign` (`s2s_probe.py:427-428`) and the
mechanical gate checks only sign (`:487-490`); `permutation_null` (`:270-294`) and `bootstrap_delta`
(`:324-343`) operate **only** on the sim-weighted `mms`/`spool` arms, never on the rank-only arms. The
binding verdict therefore cannot apply the pre-registered corroboration from the numbers the probe emits.
**Fix:** extend the permutation null + bootstrap to also cover the rank-only arms (retrieve by arm
score, sim≡1.0) and surface their null-p95 / bootstrap-5th in the results + JSON. (Stage-P only — does
not block the Stage-E smoke, but must land + be re-checked before the probe runs.)

### NON-BLOCKING

**NB-a — NEG_INF sim can leak into the near-dup-excluded vote.** `run_vote` sets excluded/self entries
to `NEG_INF=-1e30` (`:174-176`); if a query had ≥ N−k flagged neighbours, `np.argpartition(-row,k)[:k]`
could return NEG_INF-scored entries whose `row[topk_idx]` sim then multiplies the label in the vote
(`:182`). Practically impossible at threshold 0.995 (would need ~830 flagged neighbours for one query on
HateMM), but add a guard (drop NEG_INF entries from `topk_idx`, or assert `≥k` finite candidates).

**NB-b — `--gres=gpu:a100:1` vs siblings' `gpu:1`.** The A100 pin is intentional (bf16 kernel parity for
G-recon), but sibling sbatch files use `gpu:1`; confirm `a100` is a schedulable gres *type* on
`slurmpartition`, else the job may hold/reject for a reason unrelated to the normal `JobHeldUser`. If the
cluster exposes only `gpu:1`, express the A100 requirement via a node constraint instead.

### NOTE

- **N-i** `s2s_probe.py:117-118` — `if "test" in "".join(ids).lower() and False:` is an always-false
  no-op; delete it (the `N==851/629` size guard is the real, sound N4 check).
- **N-ii** `s2s_extract.py:304` — `shard_ok(path, T_nominal)`: `T_nominal` param is unused (body uses the
  stored `grid_thw[0]`, which is actually the more correct check). Harmless.
- **N-iii** `s2s_extract.sbatch:61` — `echo "… exit=$?"` reports the preceding `echo`'s status, not
  python's (with `set -e` a python failure exits before this line). Cosmetic.
- **N-iv** Design §4 says G-decomp is "recomputable offline to ≤1e-5 from saved {g,n_t,p_S}", but g/p_S
  are stored **fp16** (`:427,429`) → an offline f16 recompute lands ~1e-3; the authoritative record is
  the inline f32 `decomp_res` in each shard + `decomp_res_max` in the gatelog. Wording only.
- **N-v** `s2s_probe.py:419` strips the per-query vote arrays from the JSON, so the design §5 "raw arrays
  so one cell can be hand-recomputed" is only partially met; the probe is fully deterministic
  (`CUDA_VISIBLE_DEVICES=""`, fixed seeds) so a re-run reproduces them. Minor reporting gap.

### Positives worth recording
Parity-by-import is real (forward byte-identical to banked `_encode`); the f32-G-decomp / bf16-G-recon
dtype split is correct and the header precision note is accurate; the temporal positive control's
σ-tracking is the one gate that truly exercises grouping and is logically sound; grid gate matches spec;
oracle A4 implemented exactly with video-level gold only and no time-span annotation read anywhere; the
N4 fail-closed path is un-bypassable; the vote is the real metrics fn used symmetrically with the same
permutation on both arms in the null; shard writes are atomic and resume is corruption-safe.

---

## 4. Final verdict — APPROVED AFTER FIXES

The current r1 hashes are **NOT authorized for submission**. Land **B1 + B2** (extractor/sbatch) and
re-run `sha256sum`, then re-pin the `S2S_PROBE_DESIGN.md §10` table. **B3** (probe) must land before Stage
P but does not block the Stage-E smoke. NB-a/NB-b and the NOTES SHOULD be folded at the same edit
(cheap). A one-line reviewer re-check of the three fixed hunks is sufficient — no full re-review.

### Smoke authorization terms (Stage E only)
After B1 + B2 land and the three scripts are re-hashed:
1. **One** `SMOKE=1 sbatch scripts/slurm/s2s_extract.sbatch` submission — single job, no `--time`,
   `PENDING (JobHeldUser)` = wait (never force-release), throwaway `--out_root` under `slurm/logs/`.
2. The smoke log MUST show, on ≥1 real video per dataset, **all four HARD gates green**:
   - gate 0a temporal positive control PASS (`match==σ`, groups distinct),
   - gate 0b grid-consistency PASS (`n_vis == grid_t·(grid_h//2)·(grid_w//2)`),
   - gate 1 G-decomp residual ≤ 1e-5,
   - gate 2 **G-recon cos ≥ 0.9999 AND max-abs ≤ 1e-3** vs the banked `img_feats` — this is the proof
     that B1 is fixed and REQUIRES the B2 decoupling, so it must actually appear in the smoke log.
   plus the config echo + both script sha256 lines, and **no artifact under the real
   `data/CLIP_Embedding/<ds>/frameset_qwen7b_8f/` path**.
3. Only after the smoke shows all four green (G-recon especially) is the **full Stage-E extraction**
   submit **separately** authorized (still one job, no `--time`).
4. Stage P (`s2s_probe.py`) stays gated behind the full extraction; B3 must be fixed and re-checked
   before it runs. Stage P remains CPU-only, zero-test-touch (N4 guard verified).

**Out of scope / unchanged:** the downstream head-training formal stage (not authorized); any second
Stage-E submission; any change to encoder/dataset/frames beyond the two pre-declared budgets; any
test-label use beyond the Fano/oracle ceiling.

---

## 5. r2 re-check (commit `0a88d73`, 2026-07-14) — VERDICT: SMOKE AUTHORIZED

One-line re-check of the fixed hunks only (not a full re-review). All three BLOCKING fixes, both
NON-BLOCKING fixes, and the NOTES are correctly applied; the r2 §10 hash table matches on-disk.

**Hash re-verification (on-disk vs `S2S_PROBE_DESIGN.md §10` r2 table) — ALL MATCH.**

| artifact | r2-table sha256 | on-disk | match |
|---|---|---|---|
| `research-wiki/experiments/exp-s2s-r3.md` | `587f9b9b…504811c7` | `587f9b9b8e103758c34ffbb4c81aaa6796f231528b4612cca7c3d513504811c7` | ✅ |
| `scripts/analysis/s2s_extract.py` | `41979f6a…051cd23a` | `41979f6a41c95e38a3cd875e11dc54a5a48eac9a5b908f295bad4d8d051cd23a` | ✅ |
| `scripts/slurm/s2s_extract.sbatch` | `2dc0f90b…d56665dc` | `2dc0f90b03a44f45945cab3194f78ec97012fe7b157727cd50f64d88d56665dc` | ✅ |
| `scripts/analysis/s2s_probe.py` | `949ebbdd…f826f209` | `949ebbdd432c9d72b1b164bc715da1cbba9fafc7f363337893f9813ff826f209` | ✅ |

**B1 — FIXED** (`s2s_extract.py:211-214`). `gv = grecon_vec.detach().cpu()` and
`bv = banked_vec.detach().float().cpu()` — both operands on CPU before `F.cosine_similarity` / abs-diff.
Cross-device crash eliminated.

**B2 — FIXED** (`s2s_extract.py:352`). `banked = load_banked_imgfeats(dataset, outname)` is now loaded
**unconditionally** (the `if args.limit` guard is gone), so `bvec = banked.get(vid)` is non-None under
`--limit` and gate 2 (G-recon) runs in the smoke. The smoke's per-split assembly line prints
`grecon_cos_min=… grecon_maxabs_max=…` (`:458-461`), so G-recon evidence appears in the smoke log.
Running G-recon on the test split during extraction is not a test-touch (embedding parity check, no
labels/scoring; extracting+caching test frame sets was already authorized, prereg §10). ✅

**B3 — FIXED** (`s2s_probe.py:275-304, 427, 434-438`). `permutation_null` now computes the rank-only
arm's own null with the **same per-seed permutation applied to both rank-only arms**
(`mms_s`/`spool_s` reused for the `rank_only=True` votes, `:294-295`); `boot_rank =
bootstrap_delta(a_set_r["votes"], a_pool_r["votes"], …)` is a paired bootstrap on the rank-only votes
(`:427`); and the credit rule is `rank_corroborates = rank_sign_ok AND rank_null_ok AND rank_boot_ok`
(`:438`) = sign-match (acc AND F1) AND rank-only obs Δacc > rank-only null-95th AND rank-only
bootstrap-5th > 0 — exactly the pre-registered A2 rule. Surfaced in the primary dict, the results MD,
the JSON, and three mechanical-gate lines (`:518-525`). Drove it on synthetic: all keys populate,
`boot_rank["dacc_p5_gt0"]` present. ✅

**NB-a — FIXED** (`s2s_probe.py:184-188`). After top-k selection, `topk_idx =
topk_idx[row[topk_idx] > NEG_INF/2]` drops excluded/self sentinels before the vote, and it raises
`RuntimeError("degenerate retrieval: no finite neighbours …")` if none remain. Stress-tested on
synthetic: a NEG_INF value can no longer enter the vote (verified it fail-loud-raises instead of
multiplying a label by −1e30); the arithmetic vote's `weight[:length]` handles the shortened list. This
is a no-op for the non-excluded arms (only the diagonal is NEG_INF there, never in the top-20). ✅

**NB-b — RESOLVED** (`s2s_extract.sbatch:3-5`). `gpu:a100:1` kept, with the evidence I asked for: the
banked-cache producer `scripts/slurm/gen_embed_mllm.sbatch:3` uses the identical `--gres=gpu:a100:1` and
ran successfully, so the a100 gres type is schedulable on this partition (and the A100 pin is what
preserves bf16 kernel parity for G-recon). ✅

**NOTES** — N-i dead `and False` no-op removed from the probe; N-ii unused `T_nominal` param dropped
from `shard_ok` (now called `shard_ok(shard_path)`, `:365`); N-iv design §4 offline-G-decomp wording
corrected. N4 fail-closed guard (train/dev_seen assert + 851/629 size guard) intact after the edits. All
three scripts `py_compile` clean; sbatch `bash -n` clean.

**One residual (NOTE, non-blocking) — stale smoke echo.** `s2s_extract.sbatch:20` and `:49` still print
"(G-recon skipped)", which now **contradicts** the B2 fix (the smoke *does* run G-recon). Functionally
harmless — the authoritative evidence is the `grecon_cos_min=`/`grecon_maxabs_max=` numbers in each
per-split assembly line — but the echo is misleading and should be corrected. **When reading the smoke
log, verify gate 2 via the `grecon_cos_min` assembly lines and ignore the "(G-recon skipped)" echo.**

### Verdict: SMOKE AUTHORIZED
Per the §4 smoke terms: **ONE** `SMOKE=1 sbatch scripts/slurm/s2s_extract.sbatch` — single job, no
`--time`, `PENDING (JobHeldUser)`=wait (never force), throwaway `--out_root`, no artifact under the real
`data/CLIP_Embedding/<ds>/frameset_qwen7b_8f/` path. The smoke log must show **all four HARD gates green**
on ≥1 real video per dataset: gate 0a temporal control (`match==σ`, groups distinct), gate 0b grid
(`n_vis == grid_t·(grid_h//2)·(grid_w//2)`), gate 1 G-decomp ≤ 1e-5, and **gate 2 G-recon
`grecon_cos_min ≥ 0.9999` AND `grecon_maxabs_max ≤ 1e-3`** (read from the assembly lines; the stale echo
is a doc bug), plus the config echo + both script sha256s. The **full Stage-E extraction is a SEPARATE
authorization** granted only after the smoke shows all four green. Stage P stays gated behind the full
extraction (B3 now landed and re-checked; N4 zero-test-touch verified). The stale-echo NOTE may be fixed
opportunistically (it does not block the smoke).

---

## 6. r3 re-check (commit `2408347`, C2→ASYM fold, 2026-07-15) — VERDICT: STAGE-P CLEARED

Diff-only re-check of the C2-candidate fold into the probe as the ASYM ablation arm. **Probe-only
amendment; the extractor + sbatch are byte-identical to the r2 pins.** All five checks pass against the
amended `exp-s2s-r3.md §5` + `S2S_PROBE_DESIGN.md §5`.

**Hash / scope.** §10 r3 table matches on-disk for every artifact: `s2s_extract.py` `41979f6a…` and
`s2s_extract.sbatch` `2dc0f90b…` are **UNCHANGED** from r2 (so the queued smoke 13159 validity is
untouched), `s2s_probe.py` `141a0441…`, `exp-s2s-r3.md` `3f1f5b09…`, `S2S_PROBE_DESIGN.md`
`d40684dc…` (self-hash confirmed). The diff touches only `build_matrices`, `permutation_null`,
`probe_dataset`, `mechanical_gate_check`, `write_markdown` — an additive fold, no rewrite.

**1. ASYM math — CORRECT.** `s2s_probe.py:150` `asym = (pooled @ G.t()).reshape(N,N,T).max(dim=2).values`
with `pooled = normalize(g.mean(dim=1))` (the POOLED arm's query vector) and `G = ghat.reshape(N*T,D)`
(the SET arm's L2-normed memory frame vectors). I confirmed numerically that `asym[i,j] == max_m
cos(ĝ^Q_pooled_i, ĝ^M_j[m])` to 6e-8 — exactly the `|Q|=1` reduction of MeanMaxSim on the SAME frozen
frame vectors, i.e. the pooled-query × set-memory off-diagonal cell. The matrix is (correctly)
asymmetric; its diagonal is self-similar (0.635 vs 0.173 off-diagonal) and LOO-excluded by `run_vote`'s
`fill_diagonal(NEG_INF)`. It flows through `add("ASYM", M["asym"])` → the IDENTICAL `run_vote` →
`compute_metrics_retrieval` — **no bespoke vote code**. ✅

**2. Machinery symmetry — CORRECT.** `permutation_null` now evaluates ASYM under the **same per-seed
permutation** as every other arm (`asym_s = asym[ix]`, `ix` shared with `mms_s`/`spool_s`, `:293-297`),
producing both Δ(ASYM−POOLED) (mirrors SET) and the Δ(ASYM−SET) adjudication contrast, each with its own
null-95th. Bootstrap adds `boot_asym` (ASYM vs POOLED) and `boot_asym_vs_set` (ASYM vs SET), `:446-447`.
The `c2_asym` results/JSON block exposes `asym_vs_set_d_acc/d_f1` (Δ(ASYM−SET)),
`asym_vs_set_null_p95_acc` (null-95th), `asym_vs_set_boot_dacc_p5` (boot-5th), and `asym_beats_set`; the
MD writer surfaces the same. ✅

**3. C2 kill logic — matches the recon's two branches, non-binding.** `mechanical_gate_check` (`:571-593`):
branch **(a)** — if `oracle_all_below` (S2S oracle Δacc < +0.04 on every dataset, the same variable that
drives the S2S kill-switch) → `KILL(DEAD-with-S2S-family)` and **no** per-dataset ASYM adjudication;
branch **(b)** — else per-dataset `asym_beats_set = (Δacc_{ASYM−SET} > 0 AND Δf1_{ASYM−SET} > 0)`, then
`any_beats` across datasets → `SURVIVES(escalate-to-§11-asym)` if ASYM beats SET on ≥1 dataset, else
`KILL(DEAD-route)`. I drove all three states on synthetic results: (a) fires and suppresses adjudication,
(b)-fail → DEAD-route, (b)-pass → SURVIVES. This is exactly `exp-s2s-r3.md:242-247` /
`S2S_PROBE_DESIGN.md:277-283`. The C2 checks append to the same `checks` list emitted under the
"MECHANICAL … NOT the binding verdict" banner (`:13, :522` intact) — stays non-binding. ✅

**4. Extractor/sbatch/hash-table — CONFIRMED** (above). Smoke 13159 runs the r2-pinned extractor,
unaffected by the probe-only r3. ✅

**5. No regression — CONFIRMED.** `run_vote` (incl. the NB-a `row[topk_idx] > NEG_INF/2` guard,
`:187-189`), `load_memory` (N4 asserts + 851/629 size guard, `:73/:80/:115`), and the B3 credit rule
(`rank_corroborates = sign AND null AND boot`, `:467`) are all **outside the diff** and grep-confirmed
intact; B1/B2 live in the byte-identical extractor. `py_compile` clean. ✅

**Residual (unchanged from r2):** the sbatch still echoes "(G-recon skipped)" (`:20/:49`) — stale, but
the extractor is byte-identical to r2 by design, so this is the same non-blocking NOTE; verify gate 2 in
the smoke via the `grecon_cos_min` assembly lines.

### Verdict: STAGE-P CLEARED
The ASYM fold is correct, symmetric, and faithful to the amended pre-registration; it is a pure additive
ablation cell on the existing zero-test-touch probe with no new GPU, no new vote code, and no regression
to any prior fix. Stage P remains cleared to run **after** the Stage-E smoke passes and the full
extraction is separately authorized (unchanged gating). The §4/§5 smoke terms stand verbatim.

---

## 7. r3a re-check (commit `fabb49f`, smoke-13159 device fix, 2026-07-15) — VERDICT: CLEARED FOR SMOKE RESUBMIT

Tight one-line-diff re-check after smoke 13159 failed in 13 s at gate 0a (temporal positive control),
pre-data, with `RuntimeError: … cuda:0 and cpu` in `encode_frameset`'s G-decomp assembly. All four checks
pass.

**1. Functional change is exactly the one line.** `git show fabb49f -- s2s_extract.py`: the only code
change is `n_t_t = torch.tensor(n_t, dtype=torch.float32)` → `n_t_t = torch.tensor(n_t,
dtype=torch.float32, device=g.device)` (`:194`) plus a 3-line explanatory comment. `g` is
`torch.stack(prefix[idx].mean(0)…)` on the model device, so the old CPU-default `n_t_t` made
`recon_mean = (g * n_t_t[:,None]).sum(0).add(p_S)…` (`:202`) mix cuda+cpu. Building `n_t_t` on `g.device`
fixes the cuda path and is **behavior-identical on CPU** (`g.device==cpu` there, so `n_t_t` lands on CPU
exactly as before). ✅

**2. No other semantic change rode the commit.** `git show --stat`: three files — `s2s_extract.py`
(7 lines, the fix + comment), `S2S_PROBE_DESIGN.md` (+29, **purely additive**: §10 r3a hash table + §11
revision entry, record-keeping only, 0 deletions), `S2S_SMOKE_RECORD.md` (+56, new postmortem, record).
`s2s_probe.py` `141a0441…` UNCHANGED, `s2s_extract.sbatch` `2dc0f90b…` UNCHANGED, prereg
`exp-s2s-r3.md` `3f1f5b09…` UNCHANGED — confirmed on-disk and restated in the r3a table. The probe (incl.
r3 ASYM fold, B3, NB-a, N4) and the sbatch are untouched. ✅

**3. Sibling-device audit — CONFIRMED, `n_t_t` was the only one.** I traced every tensor in
`encode_frameset` and scanned the whole file for CPU-default constructors: `torch.eye(4)` (`:281`) sits in
`temporal_positive_control`, which operates on the returned `g.detach().cpu()` (CPU) — no mix;
`torch.zeros` (`:378-380`) and `torch.tensor` (`:441-445`) build the CPU shard/assembled `.pt` from
CPU-loaded data — no mix. After the fix, every device-mixing op is consistent: the **G-decomp assembly**
(`g`, `n_t_t`, `p_S`, `banked_formula_vec` all on device, `:190-203`) and the **B1 G-recon compare**
(`gv`, `bv` both `.cpu()`, `:214-217`). No sibling device bug remains to burn a second smoke. ✅

**4. Hash.** On-disk `s2s_extract.py` = `07fd162196a7e61e8e83f1a181408fe7b8080cf475cb59ecd58a1dc035b3740a`
= the §10 r3a freeze table entry; the table also correctly records sbatch/probe/prereg as UNCHANGED.
`py_compile` clean. ✅

**Reviewer self-correction (honest note).** This device bug lived in the G-decomp assembly since r1, and
my r1 review (§2, Lesson 1 gate-1) asserted the G-decomp operands were "both on device" — that was
**wrong**: `n_t_t` was CPU-default, so the assembly mixed cuda+cpu. It is a GPU-only fault that the
CPU-only synthetic drive available to this review structurally could not surface, and the mandated smoke
(the intended backstop) caught it in 13 s pre-data — but I under-caught it, and the correction is recorded
here. No data or meaningful GPU was consumed (crash was in the synthetic control before any real video);
banked caches untouched.

### Verdict: CLEARED FOR SMOKE RESUBMIT
The r3a fix is correct, complete (full-function device audit done), and strictly extractor-scoped; the
probe/sbatch/prereg pins are intact. Authorize **one** `SMOKE=1 sbatch scripts/slurm/s2s_extract.sbatch`
resubmit under the unchanged §4/§5 terms — it must now show **all four HARD gates green** on ≥1 real video
per dataset (0a temporal control, 0b grid, 1 G-decomp ≤ 1e-5, 2 G-recon cos ≥ 0.9999 & max-abs ≤ 1e-3 via
the `grecon_cos_min` assembly lines; the stale "(G-recon skipped)" echo remains a cosmetic NOTE). The full
Stage-E extraction stays a separate authorization after the smoke passes; Stage P stays gated behind it.

---

## 8. r4 re-check (commit `36bd7a3`, gate 0a→0a′ onset-invariance rewrite, 2026-07-16) — VERDICT: CLEARED FOR SMOKE RESUBMIT

Diff-only re-check of the gate-0a replacement mandated by the independent amendment ruling
`refine-logs/S2S_GATE0A_AMENDMENT_RULING.md` (commit `20c0bf2`, disposition **(B)-REPLACE**), after smoke
13169 failed the old permutation-equivariance/argmax control **scientifically** (the σ test is invalid by
construction for cumulative-causal `g_t`). I read the ruling §D myself and diffed its spec against the
code rather than trusting the transcription. All five checks pass.

**1. Code implements ruling §D.1 VERBATIM.** The only functional change is `temporal_positive_control` →
`causal_prefix_control` (gate 0a′) with the assertions factored into `_assert_onset_invariance(gp,gq)`
and the call-site renamed. Diffed line-by-line against §D.1:
- **Clips:** `order_p=[0,1,2,3]` → P pairs R,G,B,Y; `order_q=[0,1,3,2]` → Q pairs R,G,Y,B (`colours[3],
  colours[2]`=Y,B). Frames 0–3 (groups 0,1) identical; frames 4–7 (groups 2,3) differ. Exactly the
  ruling's P=[R,G,B,Y]/Q=[R,G,Y,B], shared prefix {0,1}, changed {2,3}. ✅
- **Encode:** both clips `banked_vec=None`, `T==4` guard, `gp/gq = F.normalize(r["g"], dim=1)` (per-group
  L2-norm). ✅
- **Assertion 1 (prefix-invariance):** `cross=(gp*gq).sum(1)`; for `k∈{0,1}` HALT if `c[k] < 0.999`
  (⇔ requires cos ≥ 0.999). ✅ verbatim.
- **Assertion 2 (onset-divergence):** HALT unless `max(c[2],c[3]) < min(c[0],c[1]) − 0.002`. ✅ verbatim.
- **Assertion 3 (within-clip distinctness):** for **each** clip, `max off-diag of (gn@gn.t()) < 0.999`
  (`torch.eye(4, device=gn.device)` — device-safe). ✅ verbatim (in fact stricter than old 0a, which
  checked only one clip; the ruling says "for each clip").
- **HALT + placement:** every violation `raise RuntimeError` (uncaught → non-zero exit); call site is in
  `main()` before the splits loop (before any real video), comment updated to `[gate 0a']`. ✅

**2. No other semantic change.** The extractor diff is two hunks only — the gate-0a function region and
the two call-site lines. `encode_frameset` (incl. the r3a device fix, G-decomp, B1 G-recon compare),
`process_split`, assembly, shard I/O, `main`'s model load — untouched. `s2s_extract.sbatch` `2dc0f90b…`
and `s2s_probe.py` `141a0441…` are byte-unchanged (hashes confirmed; `grep "0a'" s2s_probe.py` → 0). ✅

**3. Doc REPLACE-in-place, Stage-P bars untouched.** `exp-s2s-r3.md` + `S2S_PROBE_DESIGN.md` reworded to
"cumulative causal group summaries" (§2/§4, §D.2/§D.3); the `g_t` definition gains the cumulative-causal
clause; the §7 anchor table, prereg §7 gate-0, §13 K0, and both gate tables now read **0a′**
(`exp:365/451/534`, `design:153/207/424`); r4 revision entries + §10 r4 hash table added. **No Stage-P
bar/arm/threshold moved** — verified verbatim: oracle-ceiling `Δacc < +0.04` (`exp:314,372,455`;
`design:358`) and raw bar `Δacc ≥ +0.05 AND ΔmF1 ≥ +0.05` (`exp:324,456`; `design:360`) are unchanged,
and old equivariance wording ("permutes … identically" / "nearest its slab" / `sigma=[2,0,3,1]`) is
**fully gone** from both docs. ✅

**4. CPU self-test exercises the SAME code path the GPU run hits.** The assertions are factored into
`_assert_onset_invariance`, which `causal_prefix_control` (the GPU path) calls with the real
Qwen-derived `gp,gq`; the CPU self-test calls the **same** helper with synthetic tensors. I ran the
committed helper: a valid PASS case returns `[1.0,1.0,~0.07,~-0.02]` (matching the implementer's
`[1.0,1.0,0.31,−0.07]` shape), and all three targeted HALT paths fire (`PREFIX-INVARIANCE FAILED` when a
prefix group differs, `ONSET-DIVERGENCE FAILED` when the changed groups don't diverge,
`WITHIN-CLIP DISTINCTNESS FAILED` when groups collapse). Only the *source* of `gp,gq` differs between the
CPU test and the GPU run; the risky assertion logic is identical and shared. ✅

**5. Hash.** On-disk `s2s_extract.py` = `ce23dfe6810ee74a7311606b6992a747a7267e8754fc0554cd8c1f43d83ff677`
= the §10 r4 freeze table entry = the commit-message hash-freeze; the r4 table's UNCHANGED rows
(sbatch/probe) and the two changed doc hashes (`exp` `64a489f2…`, `design` `ab641369…`) all match
on-disk. `py_compile` clean. ✅

**Science soundness (confirming no false-KILL).** Prefix-invariance is guaranteed by the causal mask
(group-{0,1} tokens attend only to the identical shared frames 0–3 → their states are ~bit-exact between
P and Q, cos≈1.0 ≫ 0.999), so a correct extraction cannot fail it — the exact defect that sank old 0a is
gone. The changed groups will diverge far below the 0.998 onset ceiling (the 13169 matrix showed
different-content cumulative cos ≈ 0.79–0.90), so onset-divergence has a large margin, while a
spatial-major/reversed/interleaved grouping leaks changed frames into an "early" group and breaks
prefix-invariance → HALT. Valid under cumulation AND discriminative, as the ruling requires.

### Verdict: CLEARED FOR SMOKE RESUBMIT
The 0a′ rewrite is faithful to ruling §D.1, strictly extractor-scoped, changes no Stage-P bar, and leaves
every prior fix (B1/B2/r3a device-align, B3, NB-a, N4, r3 ASYM fold) intact. Authorize **one**
`SMOKE=1 sbatch scripts/slurm/s2s_extract.sbatch` resubmit (ruling §D.5) — single job, no `--time`,
`JobHeldUser`=wait, throwaway `--out_root`, no real-path artifact. The smoke log must show all four hard
gates green on ≥1 real video per dataset: **0a′** (prefix groups invariant cos≥0.999, changed groups
diverge, groups distinct), 0b grid, 1 G-decomp ≤ 1e-5, 2 G-recon `grecon_cos_min ≥ 0.9999 &
grecon_maxabs_max ≤ 1e-3` (via the assembly lines; the stale "(G-recon skipped)" echo is still a cosmetic
NOTE). Full Stage-E extraction remains a separate authorization after the smoke passes; Stage P stays
gated behind it.
