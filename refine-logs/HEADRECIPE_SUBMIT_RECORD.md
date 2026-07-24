# HEAD-RECIPE (SAM + modality-dropout) — SUBMIT RECORD (submit executor)

**Role:** submit executor for the frozen HEAD-RECIPE family (F68 wave-2). ZERO user interaction. NO push.
NO test metric read for decisions. NO verdict / NO gates / NO deltas / NO pass-fail language on the head
numbers. RAW-ONLY at the head stage: the executor transcribes raw both-protocol per-seed numbers
(line-numbered); the verdict is rendered by an independent 0-context reviewer against the prereg VERBATIM.
**Date:** 2026-07-25 NZST.
**Prereg:** `refine-logs/HEADRECIPE_PREREG.md`, FROZEN sha256
`68be61ac5ad77d2cfc66f3b355b574855bb426ed2c2a6e6b89f8050347ba232d`.
**Freeze:** `refine-logs/HEADRECIPE_FREEZE.md` (reviewer verdict APPROVED-WITH-NOTES, 4 non-blocking notes).
**Review:** `refine-logs/HEADRECIPE_PREREG_REVIEW.md`.
**House precedent:** `refine-logs/READOUT_SUBMIT_RECORD.md`, `refine-logs/FRAME16_SUBMIT_RECORD.md`.

Authorization derives from the freeze and is VOID on any sha mismatch.

**Four review notes that travel (non-blocking; recorded for the verdict reviewer):**
1. Mod-dropout perturbs the retrieval query as well as the classifier output (within-mechanism; mining
   INDEX still clean, encoded under `model.eval()`).
2. SAM clips/steps on the PERTURBED gradient (`grad_clip=0.1` acts on the ascended grad — standard SAM).
3. The §4.4.2 mask-rate reference numbers (0.2953 / 0.1525 / 0.1427 / 0) are one seed's draw, not a fixed
   target.
4. The re-mine assert is conservatively over-broad (would also crash a hypothetical no-mining config; the
   deployed config always mines, so it never trips; can only ever block, never fabricate a pass).

---

## 1. Sha re-verification at submit time — ALL MATCH (authorization intact)

Re-ran `sha256sum` on the prereg (self-sha), artifacts A/B/C, and the reused-unchanged machinery. **Every
hash matches the frozen block in `HEADRECIPE_FREEZE.md`; authorization is intact.**

### Prereg (self-sha) + frozen artifacts A/B/C
```
FROZEN 68be61ac5ad77d2cfc66f3b355b574855bb426ed2c2a6e6b89f8050347ba232d  refine-logs/HEADRECIPE_PREREG.md          [MATCH]
A      1012c9e378905e5c10a0447475560de4a32904af691e457bf4ce77a3d36cc20d  src/run_rac.py                            [MATCH]
B      e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py                   [MATCH]
C      c88f685f68f83611fde3f91751f330d30b6be278693a405f4b9fb80f53ebb009  scripts/slurm/headrecipe_family.sbatch    [MATCH]
```

### Reused-unchanged machinery (NOT edited; SAM re-uses / FAISS gate; do NOT edit)
```
loss.py       48796638fdd60fcfb313e97e7f89d73226d96f23369f8c8ebb61ca5814f9cd64  src/model/loss.py                          [MATCH]
retrieval.py  d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57  src/utils/retrieval.py                     [MATCH]
anchor        19c76b177f7dc883a9e03524450ad2e6cb302cdd0a6704d69da68a62188a06fc  scripts/slurm/enc3seed_lora_hatemm.sbatch  [MATCH]
```

Header verification (prereg §6 resource plan): `headrecipe_family.sbatch` requests `--cpus-per-task=8`,
`--mem=64G`, `--gres=gpu:a100:1`, and carries **NO `--time`** (L2-8). ONE combined job (12 head runs
sequential inside); peak footprint 8 CPU / 64 G / 1 GPU — within the 16/128/2 cap, never two 16-CPU jobs.

## 2. Collision-safety re-check at submit — CLEAN (all ABSENT); banked inputs PRESENT

- `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_headrecipe*` — ABSENT (0) ⇒ fresh verdict group.
- `slurm/logs/hr_*.trainlog` — ABSENT (0) ⇒ no trainlog collision.
- `_smoke_hr` groups / `hr_smoke_*` logs — ABSENT (0) ⇒ no smoke residue.
- Banked ZH cache `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`
  — PRESENT (3/3), read-only paired input.
- Banked HateMM cache
  `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` —
  PRESENT (3/3), read-only paired input.
- Floor trainlogs (13150 ZH generic-LoRA, 13241 HateMM curric-LoRA) — PRESENT (6/6), read-only.

`--force False` would hard-abort (never overwrite) if a `RAC_video_headrecipe` path ever pre-existed; the
`hr_${ARM}_…` prefix + arm-tagged `exp_comment` keep ARM A / ARM B (same dataset/seed) distinct.

## 3. MANDATORY CODEX GATE (prereg §4.5) — CLEARED (no P1; 2 P2 inactive under deployed config)

Ran the house `codex-code-review` pattern (CLI, NOT MCP — the skill mandates `codex exec`), model
**`gpt-5.4`**, `model_reasoning_effort=xhigh`, `--full-auto`, session `019f954e-865a-73b3-9b22-12704f423f56`
(codex-cli 0.144.1). Codex READ the actual source (`run_rac.py`, `classifier.py`, `loss.py`,
`retrieval.py`) with full context and cited file:line for every conclusion. Focus per §4.5: the SAM
double-step + re-mine-reuse (F0.6) + classifier identity-fill. Full transcript:
session scratchpad `codex_headrecipe_review1.txt`.

**VERDICT: NO P1 findings.** Codex verbatim: *"No P1s found under the deployed dense CPU-FAISS path. The
core reuse invariant holds ... the SAM second pass reuses those same variables, the `train_feats is None or
train_labels is None` re-encode gate stays closed, and `seg_mode=full` with `lambda_seg=0` stays on the plain
`model_pass` branch."*

Codex's independently VERIFIED list (all load-bearing invariants confirmed with citations):
- `_sam_ascend`: correct **global 2-norm** over params-with-grad, `eps = rho·g/(‖g‖+1e-12)`, in-place
  `add_`/`sub_` under `torch.no_grad()`, `_sam_restore` undoes EXACTLY via the cached `e` (not a fresh grad);
  no device/dtype bug.
- **SAM ordering correct**: loss@w → zero_grad → backward → ascend → second loss@w+eps → zero_grad →
  backward → restore → clip → step. Restoring BEFORE `step()` correctly applies the w+eps grad to w; the
  second `zero_grad()` prevents double-counting.
- **Re-mine-reuse invariant SATISFIED** under the deployed config: the same cached `train_feats/train_labels`
  are passed to the second call, `retrieval.py:341` gate stays False, the train re-encoding loop is skipped
  at w+eps; the assert is correctly placed before perturbation and fails loudly rather than silently
  re-mining; the post-training E-step path is dead (`segment_cache` None when `lambda_seg==0`, EM driver only
  for `seg_mode consensus/selfscore`).
- **No-flag byte-identity preserved**: `--sam` default False ⇒ else-branch is the plain
  zero_grad→backward→clip→step; `--mod_dropout` default False + the `training ∧ mod_dropout ∧ align` gate ⇒
  the classifier forward skips all extra RNG/tensor ops. Floors need NO re-run.
- **Mod-dropout correct**: ones-fill (not zeros) — correct under Hadamard align (zero-fill would zero the
  fused vector); at-most-one-stream guaranteed (`drop&coin` vs `drop&~coin`); OFF at eval (`self.training`);
  masks on `img_feats.device`, `ones_like` preserves device/dtype/shape.

**Two P2s — BOTH inactive under this family's deployed config (codex-agreed); NOT blocking; NOT fixed
(fixing would edit frozen artifact A and force a re-freeze, and neither changes any behaviour here):**
1. *P2 (aux_pack)*: if `aux_pack` were enabled, `_sam_ascend`/`clip_grad_norm_` would skip the aux-head
   params. **Inactive:** the sbatch passes NO `--lambda_aux` (0 occurrences) ⇒ default `0.0` ⇒
   `aux_pack = None` (`run_rac.py:1219`) ⇒ optimizer is built over `model.parameters()` only (the else branch
   `run_rac.py:607-608`), which is EXACTLY the full trainable set SAM/clip walk. Codex's own Assumptions:
   *"I assumed these head-only runs do not enable aux_pack."* Latent-only; does not fire here.
2. *P2 (SAM assert crash in unsupported retrieval modes)*: the assert would crash if `train_feats/labels`
   were None (non-dense/sparse modes). **Inactive + by design:** the sbatch uses dense CPU-FAISS
   (`--hard_negatives_loss True --no_hard_negatives 1 --no_pseudo_gold_positives 1`, no `--sparse_dictionary`)
   which ALWAYS mines ⇒ non-None ⇒ assert satisfied. Codex verbatim: *"Under your deployed config this assert
   is satisfied."* This is **exactly review Note 4** — the assert is conservatively over-broad, the safe
   crash-loudly direction (DEV-5), and *"can only ever block, never fabricate a pass."*

Six P3 nits (documentation wording of the reuse invariant, floating-point roundoff in restore, `mod_dropout_p`
not range-checked, faiss index object still rebuilt/searched at w+eps though corpus embeddings reused, etc.) —
all cosmetic / out-of-config; none affect the deployed head-recipe result.

**Both sides reviewed the final (frozen, unchanged) code; codex reports no P1s; I agree (no P1 missed, no P1
falsely dismissed); both P2s explicitly accepted as inactive-under-deployed-config with documented
justification. Per prereg §4.5 there are NO blocking findings ⇒ no code fix ⇒ no re-freeze ⇒ gate CLEARED.**
Artifacts A/B/C shas UNCHANGED by the gate (still `1012c9e3…`/`e7b61df4…`/`c88f685f…`; authorization intact).

## 4. Smoke (prereg §4.4) — PASS

### 4.1 $0-CPU checks (login-node, CUDA_VISIBLE_DEVICES="", seconds) — PASS

Script: session scratchpad `hr_cpu_smoke.py` (imports the CURRENT frozen `run_rac.parse_args`; replicates the
`classifier.py:129-136` masking exactly).

- **(A) mod-dropout mask-rate (§4.4.2b), training mode, p=0.3, n=200000:** `drop-rate=0.2965`,
  `img=0.1487`, `text=0.1478`, **BOTH-dropped=0** (at-most-one-stream invariant holds), fill-applied to the
  dropped rows (ones), tensors changed in training. **Eval-mode gate (self.training=False) ⇒ NO fill**
  (any-change=False). Matches the prereg reference draw (0.2953 / 0.1525 / 0.1427 / 0) within stochastic
  tolerance (review Note 3: reference is one seed's draw, not a fixed target). `MASK_RATE_SMOKE: PASS`.
- **(B) no-flag Namespace equivalence (§4.4.3 / §4.1b):**
  - (B1) headrecipe with `ARM_FLAGS=""` vs the floor command, BOTH parsed by the current `run_rac`: differ
    ONLY in `{group_name, exp_comment}`; the 4 new keys are inert-default in both
    (`sam=False, sam_rho=0.05, mod_dropout=False, mod_dropout_p=0.3`).
  - (B2) headrecipe no-flag (current code) vs the BANKED 13150 seed0 Namespace (old code, no new keys):
    the ONLY keys present now but absent in banked are EXACTLY the 4 inert keys
    `{sam, sam_rho, mod_dropout, mod_dropout_p}`; every other shared arg is byte-identical; the sole residual
    shared-key difference is `output_path` (a path string DERIVED in `main()`, captured pre-derivation at
    parse time — inert, not a training knob). `NO_FLAG_NAMESPACE_EQUIV: PASS`.

  This confirms F0.7: flags-off ⇒ Namespace byte-identical to the banked floor modulo the 4 inert keys +
  derived-inert `group_name`/`exp_comment`/`output_path` ⇒ the banked floors need NO re-run.

### 4.2 GPU smoke (prereg §4.4.1 / §4.4.2a) — PASS; artifacts deleted

Throwaway smoke sbatch (session scratchpad `hr_smoke.sbatch`, mirroring the family env block; `--epochs 3`,
seed 0, `--group_name _smoke_hr`, on the ZH LoRA cache; 3 runs = no-flag baseline + ARM A `--sam` + ARM B
`--mod_dropout`). **Job 13477** (`hr_smoke`): `sbatch` (NO `--time`); auto-released from `JobHeldUser` (never
forced; running aggregate was zero); **COMPLETED** exit 0:0, Elapsed 00:00:31 (A100).

| arm (flags) | per-epoch tqdm (3/3) | loss lines w/ NaN·Inf | assert-hits | tracebacks | epochs done |
|---|---|---|---|---|---|
| baseline (none) | `1.19 it/s` (≈0.84 s/epoch) | 0 | 0 | 0 | 3 ✓ |
| ARM A (`--sam True --sam_rho 0.05`) | **`1.19 s/it`** (≈1.19 s/epoch) | 0 | 0 | 0 | 3 ✓ |
| ARM B (`--mod_dropout True --mod_dropout_p 0.3`) | `1.17 it/s` (≈0.85 s/epoch) | 0 | 0 | 0 | 3 ✓ |

- **(i) loss finite:** every loss line finite across all 3 epochs for all 3 arms (0 NaN/Inf; sample baseline
  0.878880 → 0.813780 → 0.621143).
- **(ii) completes / re-mine-reuse assert does NOT trip:** ARM A ran to completion with **0 AssertionError**
  and **0 Traceback** — the SAM re-mine-reuse `assert` (F0.6) executed on the SAM path and did NOT trip; FAISS
  re-mine fired once/epoch (reindex_every_step=False) with no crash.
- **(iii) SAM double-step VISIBLE:** ARM A per-epoch wall is `1.19 s/it` vs the flag-off baseline `1.19 it/s`
  (= 0.84 s/it) — tqdm flips from `it/s` to `s/it` exactly at the 1-second boundary; SAM's per-epoch time is
  **≈1.42× the baseline**, the signature of the second forward-backward. ARM B (`1.17 it/s`) ≈ baseline (no
  extra optimizer step; only the in-forward mask), as expected.

**Cleanup:** `logging/Retrieval/MHC_zh/_smoke_hr` + `slurm/logs/hr_smoke_*` (3 trainlogs + `.out`) **deleted**;
§4.3 collision targets re-verified ABSENT after deletion (RAC_video_headrecipe / hr_*.trainlog / _smoke_hr /
hr_smoke_* all 0); banked ZH + HateMM caches re-checked UNTOUCHED (4/4 spot-check sha16 MATCH freeze §5.2 — the
distinct `_smoke_hr` group never clobbered the deployed tag). Throwaway `hr_smoke.sbatch` + `hr_cpu_smoke.py`
live only in the session scratchpad.

**SMOKE_VERDICT: PASS.** Cleared to submit the real family job.

## 5. Real family — single-submitted (prereg §6 / §1.1; NO `--time`; 12 head runs sequential)

Final `sha256sum` re-verified at the submit instant — A `1012c9e3…`, B `e7b61df4…`, C `c88f685f…`,
loss.py `48796638…`, retrieval.py `d43e3bc4…`, prereg `68be61ac…` [ALL MATCH]; `bash -n` C = SYNTAX_OK;
queue EMPTY; authorization intact.

| job | id | script | runs | CPU/mem/GPU | ~cost |
|---|---|---|---|---|---|
| headrecipe family | **13478** | `headrecipe_family.sbatch` (2 arms × 2 datasets × 3 seeds = 12 head runs sequential, hardcoded CONFIGS) → `slurm/logs/hr_{SAM_rho0.05,MODDROP_p0.3}_{MHC_zh,HateMM}_<MODEL>_seed{0,1,2}_13478.trainlog` | 12 | 8 CPU / 64 G / 1×A100 | < 0.15 GPU-h |

Peak footprint 8 CPU / 64 G / 1 GPU (within the 16/128/2 cap; never two 16-CPU jobs — queue was EMPTY at
submit). ONE sbatch = ONE family = ONE multiplicity bite.

### 5.1 Queue state at submit — PENDING (JobHeldUser); WAIT never force

- **13478 PENDING (JobHeldUser)** (no dependency). Per CLAUDE.md the hold is **waited out, NEVER forced**
  (the smoke 13477 auto-released from the identical hold in seconds; aggregate was zero). If held > 2 h a
  status line is committed and the turn ends PENDING-JOB (orchestrator resumes). **On COMPLETE:** transcribe
  RAW both-protocol per-seed numbers (line-numbered) for all 12 runs — NO gates/deltas/pass-fail.

### 5.2 Chain outcome — 13478 COMPLETED (exit 0:0)

**13478 headrecipe (12 head runs):** auto-released from `JobHeldUser` (never forced), RUNNING on
`foscsmlprd01`, **COMPLETED** exit 0:0, Elapsed 00:51:33; `======== headrecipe ALL DONE (13478) ========`.
All 12 trainlogs written; **0 AssertionError / 0 Traceback** across all 12 (the SAM re-mine-reuse assert did
NOT trip in any real run); derived logs B2-pushed (videos never left the node).

## 6. RAW per-seed both-protocol numbers (executor transcription — NO gates / NO deltas / NO pass-fail)

Val-selected protocol = epoch ≥ warmup 5 with max `Val_Retrieval` acc (roc tie-break) → that epoch's
`Test_Retrieval` line; final-epoch = max epoch (29). Values are the **macroF1-format** `Test_Retrieval` line
(the protocol's macro-F1 line, same regex the sbatch's embedded parser uses). Each `(:N)` is the 1-based line
number (grep -n / sed compatible) of that `Test_Retrieval Epoch` line in the named trainlog. **Cross-checked
three ways: (i) an independent parser, (ii) the sbatch's OWN embedded RESULT_ROW parser — bit-identical
values, (iii) direct `sed` at each cited line.** The executor applies NO gates/deltas/interpretation; the
independent 0-context reviewer renders the verdict against the prereg VERBATIM (and will re-parse the
trainlogs itself).

Trainlogs: `slurm/logs/hr_{SAM_rho0.05,MODDROP_p0.3}_{MHC_zh,HateMM}_<MODEL>_seed{0,1,2}_13478.trainlog`
(ZH `<MODEL>`=`Qwen2.5-VL-7B-Instruct-LoRA_HF`; HateMM `<MODEL>`=`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`).

### 6.1 ARM A — SAM (`--sam True --sam_rho 0.05`)

**MHC_zh** (`hr_SAM_rho0.05_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13478.trainlog`):

| seed | val-sel ep | val-sel acc/mF1 | (Test line) | final ep | final acc/mF1 | (Test line) |
|---|---|---|---|---|---|---|
| 0 | 7 | 0.7852 / 0.7385 | :92 | 29 | 0.7987 / 0.7612 | :269 |
| 1 | 8 | 0.8255 / 0.8002 | :100 | 29 | 0.8054 / 0.7646 | :269 |
| 2 | 5 | 0.8121 / 0.7893 | :74 | 29 | 0.8054 / 0.7784 | :267 |
| **mean** | | **0.8076 / 0.7760** | | | **0.8032 / 0.7681** | |

**HateMM** (`hr_SAM_rho0.05_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13478.trainlog`):

| seed | val-sel ep | val-sel acc/mF1 | (Test line) | final ep | final acc/mF1 | (Test line) |
|---|---|---|---|---|---|---|
| 0 | 25 | 0.8791 / 0.8735 | :262 | 29 | 0.8884 / 0.8828 | :299 |
| 1 | 28 | 0.8884 / 0.8828 | :294 | 29 | 0.8837 / 0.8776 | :304 |
| 2 | 29 | 0.8791 / 0.8724 | :300 | 29 | 0.8791 / 0.8724 | :300 |
| **mean** | | **0.8822 / 0.8762** | | | **0.8837 / 0.8776** | |

### 6.2 ARM B — modality-dropout (`--mod_dropout True --mod_dropout_p 0.3`)

**MHC_zh** (`hr_MODDROP_p0.3_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13478.trainlog`):

| seed | val-sel ep | val-sel acc/mF1 | (Test line) | final ep | final acc/mF1 | (Test line) |
|---|---|---|---|---|---|---|
| 0 | 24 | 0.8322 / 0.8023 | :229 | 29 | 0.7919 / 0.7577 | :270 |
| 1 | 6 | 0.8523 / 0.8202 | :83 | 29 | 0.8389 / 0.8090 | :268 |
| 2 | 29 | 0.8121 / 0.7771 | :275 | 29 | 0.8121 / 0.7771 | :275 |
| **mean** | | **0.8322 / 0.7999** | | | **0.8143 / 0.7813** | |

**HateMM** (`hr_MODDROP_p0.3_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13478.trainlog`):

| seed | val-sel ep | val-sel acc/mF1 | (Test line) | final ep | final acc/mF1 | (Test line) |
|---|---|---|---|---|---|---|
| 0 | 5 | 0.8512 / 0.8450 | :80 | 29 | 0.8651 / 0.8567 | :297 |
| 1 | 5 | 0.8512 / 0.8476 | :80 | 29 | 0.8837 / 0.8776 | :297 |
| 2 | 29 | 0.8698 / 0.8620 | :299 | 29 | 0.8698 / 0.8620 | :299 |
| **mean** | | **0.8574 / 0.8515** | | | **0.8729 / 0.8654** | |

**3-seed means are raw arithmetic summaries of the transcribed per-seed values only** (as in the prereg §2
floor tables); the executor computes NO Δ-vs-floor, applies NO promote/kill bar, and renders NO pass/fail or
KILLED language. The banked floors (prereg §2.1 ZH / §2.2 HateMM; freeze table) and the §3 bars are the
independent reviewer's inputs.

## 7. Closeout — CHAIN COMPLETE (raw numbers transcribed; verdict deferred to independent reviewer)

Sha re-verify (submit + submit-instant) ALL MATCH; codex gate CLEARED (no P1; 2 P2 inactive under deployed
config, freeze preserved); $0-CPU smoke PASS (mask-rate 0.2965 / both-dropped 0 / no-flag Namespace adds
exactly the 4 inert keys); GPU smoke 13477 PASS (exit 0:0, SAM double-step visible, artifacts deleted);
collisions CLEAN throughout; family 13478 COMPLETED exit 0:0 (00:51:33), 12/12 trainlogs, 0 assert/traceback;
RAW both-protocol per-seed numbers transcribed line-numbered for all 4 arm×dataset cells (§6),
cross-verified three ways. **NO gates/deltas/pass-fail applied; NO `state/` mutation; nothing pushed.** The
independent 0-context reviewer renders the verdict (KS-arm-dead → FORMAL +0.030/+0.030 both-protocol, per
arm×dataset) against the prereg VERBATIM.

