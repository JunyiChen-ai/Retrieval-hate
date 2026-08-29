# C02 A0 — evidence-density orbit reachability, preregistration record

**Status:** `V1_FROZEN_READY_NOT_SUBMITTED_PENDING_INDEPENDENT_REVIEW`
**Date:** 2026-07-30 (Pacific/Auckland)
**Candidate:** `C02 Evidence-Density Quotient Geometry`
**Run IDs:** `C02-DEN-v1` (extraction), `C02-A0-v1` (A0)

This record is **prospective**. At freeze time no extraction job, no A0 job, no result,
no decision, no metric and no CONTINUE/KILL verdict exists, and none is claimed. Every
number below that is not a *threshold* is either a hash, a resource figure, a count of
lines in an already-banked file, or a measurement on **synthetic arrays** explicitly
labelled as such.

---

## 1. Authority and what changed

- Registry authority:
  `TARGET_STATE.json::iteration_8_stage0_bounded_extraction_amendment`
  (`STAGE0_BOUNDED_EXTRACTION_2026_07_30`), landed from the user's 2026-07-30 ruling
  that extraction is authorised.
- C02's registry status is
  `revived_under_stage0_bounded_extraction_amendment_a0_prereg`. The 2026-07-29
  `KILL_C02_DESIGN_COLLISION_OR_INFEASIBILITY` is **retained verbatim** in
  `TARGET_STATE.json` and in `refine-logs/C02_DESIGN_REVIEW.md`, is **not overturned on
  its own terms**, and remains the authoritative account of why C02 could not be
  adjudicated under the pre-amendment gate.
- What the kill actually ruled: a **sequencing and evidence-availability failure**. It
  said so in writing. Its two genuine scientific objections had already been repaired
  before it was issued, and both repairs are implemented here:
  1. *A0 proxy-target mismatch* — the old A0 used P3 image-pooling banks as a proxy for
     a text-density orbit. **Repaired:** the orbit is now defined on the native text
     channel and the views are extracted, so the A0 representation is the one the
     mechanism is defined on.
  2. *Unsafe evidence-core deletion* — **repaired:** every view retains the complete
     native text as an **ordered subsequence** and adds only controlled repetition.
     Nothing is deleted, reordered, summarised or paraphrased, and this is *proved per
     item per view at run time* before any forward pass.

Everything the reviewer left standing is carried forward and is listed in §7 with the
place each requirement is discharged.

---

## 2. The question, and the honest status of the answer

> Does the controlled evidence-density orbit of a video's own text channel contain
> enough reachable signal, **in the deployed head key space**, for the quotient
> similarity to clear `+0.050` accuracy **and** `+0.050` macro-F1 over the paired native
> floor on **both** HateMM and MHC-ZH?

The quotient (orbit) metric is

```
s_Q(i, j) = max_{a in A_i, b in A_j} cos(z_i^a, z_j^b)
```

`s_Q` is **deliberately optimistic**: it is the orbit metric of the quotient space and
upper-bounds what any representation that contracts this orbit could buy. It is **not a
deployable router**. Therefore a failure is decisive, and a pass authorises Stage-1
design plus a fresh review — nothing else.

---

## 3. The views

Module `src/utils/c02_density_views.py`. `T` is the `text` field of
`data/gt/<DS>/<split>.jsonl` — exactly the string the deployed extractor puts in the
Transcript slot (`generate_VideoMLLM_embedding_lora_HF.py:438-442`). On HateMM that is
the ASR transcript; on MHC-ZH it is MultiHateClip's harvested title + `" . "` + its own
speech transcript. Defining the orbit on the field the encoder is actually fed is the
only definition that is protocol-faithful on **both** datasets, needs no time alignment
and depends on no external asset.

| view | string | note |
|---|---|---|
| `NAT` | `T` | re-extracted this session; the floor |
| `RFULL` | `T + " " + T` | exact-content quantity doubling |
| `RW1..RW4` | `T[:c_k] + " " + T[c_{k-1}:c_k] + T[c_k:]` | window `k` duplicated in place |

Windows are the `K=4` contiguous character quarters at `c_k = (k*len(T))//4`. There is
**no snapping heuristic and no tunable parameter** — one integer expression, identical
on whitespace-rich English and on non-whitespace Chinese.

**Label-preservation contract.** Duplicating a contiguous block in place can only ever
leave `T` as an ordered subsequence. `c02_density_views.assert_subsequence` proves this
per item per view inside the extractor **before** the forward pass, and it is itself
tested against a deletion (self-test case `deletion_rejected`).

**Identity (degenerate) causes — declared, counted, fail-closed.**

| cause | rule | why |
|---|---|---|
| `EMPTY_TEXT` | all views `:= T` | the deployed prompt substitutes `(none)` for falsy text; `"" + " " + ""` is `" "`, which is truthy, so a repeat would *edit the prompt* rather than the density |
| `LENGTH_GUARD` (`len(T) > 12000`) | all views `:= T` | unbounded sequence growth; `C02_EXPERIMENT_PLAN.md §3.1` already required such items be excluded from the view and counted |
| `EMPTY_WINDOW` | `RW_k := T` for that `k` only | only reachable when `len(T) < 4` |

For an identity view the `NAT` vector is computed **once and copied** into the slot, so
the identity is **bit-exact by construction**, not by tolerance.

*Design-time facts used to set the length guard, read with shell text utilities from
`train`/`val` gt files only (no cache, no model, no test):* HateMM train line lengths
are p50 745, p90 3089, p95 3990, p99 12710, max 80784 bytes; 9 of 744 rows exceed 12000
and 1 exceeds 20000. MHC-ZH train max is 756 bytes. `grep -c '"text": ""'` returns 0 on
all four train/val files, so `EMPTY_TEXT` is expected to be empty — the guard exists
because it must, not because it is expected to fire.

---

## 4. Extraction — the bounded Stage-0 spend

- Script `src/utils/generate_c02_density_view_text_embedding_HF.py`; wrapper
  `scripts/slurm/c02_density_extract.sbatch`; namespace
  `artifacts/c02_edq/v1/extract/C02-DEN-v1` (**absent at freeze time**).
- **Reuse, not rewrite.** The encoder, frame sampler, chat template, pooling span,
  instruction constants and prompt assembly are imported **unmodified** from
  `src/utils/generate_VideoMLLM_embedding_lora_HF.py`, whose sha256 is asserted at
  start-up. The only new numerical surface is *which string goes into the transcript
  slot* and the output contract.
- **Text stream only.** `img_feats` never see the transcript
  (`generate_VideoMLLM_embedding_lora_HF.py:427-430`), so they are invariant across
  views by construction; re-extracting them would inject GPU non-determinism into a
  quantity that must be identical across arms. The arena pairs the **banked** native
  `img_feats` with per-view `text_feats`.
- **Weight points** are the deployed ones: HateMM `logging/lora/HateMM_curric` ->
  `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`; MHC-ZH `logging/lora/MHC_zh` ->
  `Qwen2.5-VL-7B-Instruct-LoRA_HF`. These are exactly the caches
  `mechnov_pairverify.DATASETS` names, i.e. the paired strongest baseline's own
  representation.
- **Splits: `train` + `dev_seen` only.** Asserted three ways — the split map admits only
  `train`/`val`, every input and output path is refused if it contains a test-like
  token, and `torch.load` is wrapped by the head-space instrument's guard. **A0 consumes
  the train split only**, asserted again in `c02_a0_mint.py`. `dev_seen` views are
  extracted under the same frozen contract so a future Stage-2 needs no second GPU
  spend; **no A0 code path opens them.**
- **Cost model.** 6 views x 1 text forward per item, one video decode per item, over
  744+107 (HateMM) + 579+78 (MHC-ZH) = 1508 items. Comparable single-GPU extraction jobs
  measured by `sacct`: `13295` 00:24:23, `13329` 00:27:18, `13302` 00:34:37, `13648`
  00:46:27, `13352` 00:59:57, `13470` 01:01:44, `13468 gen_embed_readout` 02:00:08 —
  all 8 CPU / 1 GPU / 48-64 G. **Projected ~2 GPU-hours, capped at the amendment's 4.0.**
  Actual elapsed will be read back from `sacct` and recorded in §10.
- **Output contract.**
  `data/CLIP_Embedding/<DS>/{split}_{base_tag}-c02den-{VIEW}.pt` with
  `ids / text_feats / labels / c02_view` and **deliberately no `img_feats` key**, so
  `run_rac` cannot silently consume a view file as a full cache. No-clobber enforced on
  every file and on the manifest.

---

## 5. The arena

**Primary — fold-head / deployed-head, because a raw arena may not promote (F113).**
`StratifiedKFold(5, shuffle=True, random_state=0)` over the train split, asserted
item-for-item against the banked `scripts/analysis/vsw_ckpt/<ds>/f{0..4}.npz` inside
every mint. Bank = the fitting pool's head keys from a head trained on that same fitting
pool; queries = the held-out fifth, **never seen by that head in any role**. Bank and
query index sets are disjoint and asserted per fold — this is **full self-orbit
exclusion**: a query's own orbit can never be retrieved. 3 head seeds (0/1/2), final
epoch 29, pooled over every train item exactly once; the **3-seed mean is the primary
read**.

**Secondary — raw fused key space** `l2n(concat(l2n(img), l2n(text_view)))`, same folds.
Under F113 it **may corroborate a KILL and may never promote a lead**, and it is
labelled as such in the output.

The head recipe is imported unchanged from `headspace_mint.py::CLI` (the deployed
`enc3seed_lora_curric` / `enc3seed_zh_b3` CLI), and `headspace_mint.py`'s fold logic,
monkeypatches, dummy-dataloader construction and DET-1 contract are reused with its
sha256 asserted. The only addition is: forward the trained head over
`(banked native img_feats, view text_feats)` for each view and store those keys.

---

## 6. Arms — every one a sub-orbit of the single extraction

| arm | orbit | role |
|---|---|---|
| `NATIVE` | `{NAT}` | **floor**; must reproduce the frozen deployed vote bit-exactly |
| `FULL` | `{NAT, RFULL, RW1..RW4}` | **PRIMARY treatment** — the complete controlled-density orbit |
| `REPEAT_ONLY` | `{NAT, RFULL}` | quantity without localization |
| `LOCALIZED_REPEAT_ONLY` | `{NAT, RW1..RW4}` | localization without whole-text doubling |
| `RANDOM_WINDOW_REPEAT` | `{NAT, RW_r(i)}`, `r = blake2b(salt+id) mod 4 + 1` | is any localized repeat as good as a chosen one? |
| `MIN_WINDOW_REPEAT` | `{NAT, RW_argmin P3}` | the anti-core localization |
| `MAX_WINDOW_REPEAT` | `{NAT, RW_argmax P3}` | the evidence-core localization (**secondary**) |
| `SHUFFLE` | `NAT` + item `pi(i)`'s non-native views, deterministic derangement | does the gain need the **correct** within-video orbit? |
| `NOISE` | `NAT` + a deterministic random tangent per item and view, **norm-matched** to the true displacement | does it need the **semantic** displacement, or just one of that size? |

`SHUFFLE` and `NOISE` are load-bearing, not decoration: a `max` over a 36-way view-pair
product mechanically inflates similarities, so an orbit oracle **must** be shown to
depend on the *correct* orbit rather than on having more vectors to maximise over.

**The primary treatment uses no P3 at all.** P3 enters only `MIN`/`MAX`, and the
P3-window-to-text-window correspondence is a **declared positional approximation** (P3
windows are temporal Whisper-ASR windows; the text channel is a string). That
approximation is confined to two control arms and is why they are controls, not the
treatment. An item with no P3 row falls back to the identity orbit in those two arms
only, and the count is reported.

---

## 7. Gates — every reviewer requirement, and where it is discharged

| requirement (2026-07-29 review) | discharged |
|---|---|
| `RANDOM_WINDOW_REPEAT` | arm, §6 |
| `MIN_WINDOW_REPEAT` | arm, §6 |
| `REPEAT_ONLY` | arm, §6 |
| `LOCALIZED_REPEAT_ONLY` | arm, §6 |
| frozen orbit radius | `orbit_radius_median_oof`, strict OOF (each item read from its own fold's head) |
| frozen KRR metric | `krr_length_probe`, RBF gamma = 1/d, ridge = 1, strict OOF, one declared repair (§9) |
| retrieval-length correlation | Spearman rho(query `len(T)`, median `len(T)` of its top-20), **per arm** |
| confidence / control thresholds | paired item bootstrap B = 10000, seed 20260730, percentile 95% CI; `FULL` must strictly beat `SHUFFLE` and `NOISE` in both metrics on both datasets |
| lambda selection | **NOT APPLICABLE AT A0** — the A0 oracle has no `lambda_orbit`; it first exists at Stage-1, selected inside outer-train folds only. Stated in config and emitted in the result. |
| Holm family | `{hatemm_acc, hatemm_mf1, zh_acc, zh_mf1}`, alpha = 0.05 |
| full self-orbit exclusion | bank/query index disjointness asserted per fold, §5 |
| empty / speech-poor identity orbits | §3 identity causes + `VIEW_SUPPORT` gate + the zero contract below |

**Additional fail-closed gates.**

- **`PARITY-NAT`.** The oracle run on the `{NAT}` orbit must reproduce
  `mechfix_ops.deployed_vote` **bit-equally** in predictions and in the sorted top-20
  similarity vector, on all 15 (seed x fold) cells per dataset. Neighbour IDs are
  asserted equal on every row whose top-20 similarities are all distinct; rows with an
  exact float32 tie are exempt from the *ID* assert only — their vote is invariant to
  tie order — and are counted and reported.
- **`ARENA-1`.** Fold parity against the banked `vsw_ckpt` npz, hard assert in every
  mint.
- **`ARENA-2`.** Pooled head-space `NATIVE` accuracy must lie in
  `[majority_rate + 0.02, 0.98]`; outside means the arena is saturated (the F47 0.998
  problem) or collapsed, and no operator could show an effect in it.
- **`GATE-FID`.** The deployed-configuration CPU proxy head is compared against the
  banked GPU floor's **dev** `Val_Retrieval` curve by the **frozen, unmodified**
  `scripts/analysis/headspace_fidelity.py` (job `13241` HateMM, `13150` MHC-ZH; the
  reader hard-filters `Test_Retrieval` lines at the point of read). **Stop rule:** if the
  3-seed instrument band `B_fid >= 0.050` — the Stage-0 accuracy bar — the measurement is
  inadmissible and the job halts **before the arena runs**.
- **`GATE-EXT`.** Re-extracted `NAT` vs banked native `text_feats`: median row cosine
  must be `>= 0.99`; min/mean cosine and max absolute difference are reported. Every
  delta in the A0 is paired against the **re-extracted** `NAT` floor from the same
  session, so treatment and floor come off one GPU, one driver and one process.
- **Zero contract.** The four `C01_ZERO_CONTRACT_PROBE.md` criteria are applied
  verbatim: (1) documented structural null — the video-decode-failure zero-guard, HateMM
  train row 355 `hate_video_95`, label 1, recorded in
  `PROVENANCE_AUDIT_2026-07-28.md:187-193`; (2) **exact zero-mask match** across the
  banked native and all six views, asserted; (3) no non-structural row with
  `0 < norm <= 1e-12`, asserted; (4) the same deployed baseline consumed that row.
  Structural nulls are **retained identically in every arm**, so no arm can gain or lose
  from them, and a sensitivity read **excluding** them is reported separately. Any
  deviation HALTs.
- **`VIEW_SUPPORT`.** Fraction of train items whose orbit is not the full identity must
  be `>= 0.60`, else HALT.
- **Non-finite** anywhere HALTs. **Any test-like path** HALTs before a handle opens.

---

## 8. Decision rule — frozen

`PASS_C02_DENSITY_ORBIT_REACHABLE` requires, on **both** HateMM and MHC-ZH, in the
**primary head arena**, 3-seed mean:

- `FULL - NATIVE >= +0.050` accuracy **and** `>= +0.050` macro-F1;
- `net_fix_rate = (fixed - broken)/n >= 0.030` — the "enough net corrected-minus-broken
  items for the `+0.030` final bar" clause;
- `FULL` strictly beats **both** `SHUFFLE` and `NOISE` in both metrics;
- both paired-bootstrap 95% lower bounds `> 0`;
- both Holm-corrected null rejections at `alpha = 0.05`.

Anything weaker is `KILL_C02_DENSITY_ORBIT_UNREACHABLE`. Any gate failure is
`HALT_FAIL_CLOSED_NO_DECISION` and is **not** a scientific verdict.

**Interpretation boundary.** A PASS is *reachability* evidence only. It authorises
Stage-1 design plus a fresh independent review. It is not a training gain, not a
development result and not a test result. A KILL closes C02 and the serial loop
advances.

---

## 9. Two defects found and repaired at freeze time, before any real data was touched

Both were found by a **synthetic-array dry run** (random matrices only — no project
cache, no model, no label, no test split, no GPU) and are recorded here because they
would otherwise have silently changed the verdict.

1. **An exhaustive `k = n_bank` faiss search is NOT bit-equal to the deployed `k = 20`
   call.** faiss selects a different code path for large `k`; measured max
   `|delta sim| = 1.5e-07`, enough to break `PARITY-NAT`. The oracle therefore searches
   `k = 20` **per view pair**, which is exact for the top-20 (if `tau` is the 20th
   largest per-item maximum, any row `>= tau` forces its item into the top-20, so each
   view pair contributes at most 20 such rows and its own top-20 already holds all of
   them) **and** is the literal deployed call for a singleton orbit. Verified on
   synthetic data: predictions, sorted similarities and neighbour IDs all bit-equal to
   `deployed_vote`, vote max-diff `0.0`; and the top-20 item sets agree with a
   brute-force max over view pairs on 40/40 synthetic queries.
2. **`mechfix_ops._norm32` can alias its input.** `np.ascontiguousarray(np.asarray(X,
   float32))` returns the same buffer for an already-float32 C-contiguous array and
   `faiss.normalize_L2` works in place, so a second normalisation of the same buffer
   shifts similarities at float32 ulp level. The frozen module is not modified; the
   arena's own `_norm32` **always copies**, so every array handed to faiss is private.

One further **declared repair**, not a defect: `C02_EXPERIMENT_PLAN.md §5` specified the
KRR length probe as "gamma = 1/d, ridge = 1". `gamma = 1/d` is the sklearn convention for
**per-dimension standardised** features; on L2-normalised 1024-d keys squared distances
lie in `[0, 4]`, so `gamma*dist^2 < 0.004`, the kernel is numerically constant and the
probe is uninformative *by construction* — measured `R^2 = 0.0087` on a synthetic planted
signal. Features are therefore z-scored on the **fitting fold only**; `gamma` and `ridge`
are untouched, nothing is tuned and nothing is selected on the held-out fold. After the
repair the same synthetic planted signal gives `R^2 = 0.8788` at `n = 744, d = 1024`, and
a synthetic null gives `-0.0163`.

---

## 10. Frozen identity — sha256

**New, frozen by this record:**

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v1.json` | `0b8a8289e7438396ce081fdf872f7d18017f870640fa33a687099de4066b53d1` |
| `src/utils/c02_density_views.py` | `e0cd2d2b920a4f5133f30d174d36865843fe23977ff1f8639eea0400d12eab72` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `9ebb80f48d27fd14278b15692d45d3c925efc84ae61941fae1488574bc96832b` |
| `scripts/slurm/c02_density_extract.sbatch` | `e5c29338fab4b0ac1af4c57826e11bde9d96f29b111bd806b98ccc1658acafbc` |
| `scripts/analysis/c02_a0_mint.py` | `3b1b602b145fa362f270ba08a604a1b284ae153f0d22f9a15dafa5c3a0abbfa7` |
| `scripts/analysis/c02_a0_arena.py` | `92abe7d8157a54f89a47657fb1edaf4a8f90e55b873c3fd03840aa940593fa41` |
| `scripts/slurm/c02_a0_cpu.sbatch` | `2b55c67834fc6dfdaf9a932be634c735b5362edcd128cfd5aa6e3829fc82c281` |

**Imported unmodified, sha256 asserted at run time:**

| path | sha256 | asserted by |
|---|---|---|
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | `75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399` | the extractor |
| `scripts/analysis/headspace_mint.py` | `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612` | mint + arena |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` | mint + arena |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` | arena |
| `scripts/analysis/headspace_fidelity.py` | `72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598` | GATE-FID reader, unmodified |

Chain: the view module pins nothing; the extractor pins the view module and the deployed
extractor; the mint and arena pin the frozen analysis modules and the view module; this
record pins config, sources and both wrappers; the TARGET records pin this record after
its bytes are frozen. **No file pins its own sha256.**

**Namespace absence at freeze time (verified):** `artifacts/c02_edq` does not exist; zero
`*-c02den-*` files exist in either `data/CLIP_Embedding/HateMM` or
`data/CLIP_Embedding/MHC_zh`.

---

## 11. Execution boundary

- **Two SLURM submissions, one each, in this order:**
  1. `sbatch scripts/slurm/c02_density_extract.sbatch` — 8 CPU / 1 A100 / 64 G, no
     `--time`.
  2. `sbatch scripts/slurm/c02_a0_cpu.sbatch` — 8 CPU / 0 GPU / 32 G, no `--time`.
- Neither wrapper has a `--time`, dependency, array, singleton, requeue, chain, force or
  release path. `PENDING (JobHeldUser)` is normal, may last hours and **must never be
  force-released**.
- 8 CPU each, so the submit-time 16-CPU aggregate-cap wedge cannot occur, and the two
  jobs are strictly serial.
- The registry's `one_candidate_at_a_time` and `parallel_gpu_or_teacher_pilots_forbidden`
  are honoured: `squeue`/`sacct` must show no other candidate's GPU or teacher pilot
  running before the extraction is submitted.
- **What has already been executed at preparation time, on the login node, and nothing
  else:** `python -m py_compile` on the four new sources (byte-compile only, no module
  execution); `bash -n` on both wrappers; `json.load` on the config; the view module's
  pure-string `self_test()`; and the synthetic-array dry run of §9. **No project cache,
  model, video, label, teacher, GPU, SLURM job or test path was opened, and no
  scientific quantity exists.**
- Execution requires a fresh **independent static review** returning `GO (0C/0H/0I)`.

---

## 12. Post-run fields — EMPTY at freeze time

| field | value |
|---|---|
| extraction job id | *(not submitted)* |
| extraction elapsed / GPU-hours vs the 4.0 cap | *(not measured)* |
| A0 job id | *(not submitted)* |
| GATE-FID `B_fid` per dataset | *(not measured)* |
| view support per dataset | *(not measured)* |
| `FULL - NATIVE` per dataset | *(not measured)* |
| verdict | *(none)* |
