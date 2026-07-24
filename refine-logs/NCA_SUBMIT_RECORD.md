# NCA / soft-kNN HEAD-LOSS family — SUBMIT RECORD (submit executor)

**Role:** submit executor for the frozen NCA head-loss family (litsweep2 wave-3). ZERO user
interaction. NO push. NO test metric read for decisions. NO verdict / NO gates / NO deltas /
NO pass-fail language on the head numbers. RAW-ONLY at the head stage: the executor transcribes
raw both-protocol per-seed numbers (line-numbered); the verdict is rendered by an independent
0-context reviewer against `refine-logs/NCA_PREREG.md` VERBATIM.
**Date:** 2026-07-25 NZST.
**Prereg (frozen object):** `refine-logs/NCA_PREREG.md`, FROZEN sha256
`7607863c15bef1d40c1f1f0a5b980123dd84aba260ced6db1fa993f277db5591`.
**Freeze:** `refine-logs/NCA_FREEZE.md` (reviewer verdict APPROVED-WITH-NOTES, 4 non-blocking notes).
**Review:** `refine-logs/NCA_PREREG_REVIEW.md`.
**House precedent:** `refine-logs/HEADRECIPE_SUBMIT_RECORD.md`.

Authorization derives from the freeze and is VOID on any sha mismatch.

**Four review notes that travel (non-blocking; recorded for the verdict reviewer):**
1. §1.2 / §5.1 mislabel loss.py as "+147/−6"; the true numstat is 141 insertions / 6 deletions
   (git's `--stat` bar total transcribed as the insertion count). Deletion count (6) + identity
   (the BCE block re-emitted verbatim under the new `else:`) correct; no hash/bar/gate affected.
2. F0.2 over-states A2 (SupCon)'s RNG divergence — conservatively. run_rac builds the per-epoch
   bank ONLY for `head_loss=='nca'`, and `_supcon_head_loss` draws no RNG, so A2 iterates
   `train_dl` once per epoch (like the floor) and stays in RNG lockstep — A2 is in fact the
   cleanest paired arm. Direction conservative; touches no bar.
3. A1 (NCA) and A3 (mixup) DO carry the disclosed treatment-arm RNG divergence (A1's per-epoch
   `_build_nca_bank` consumes a `train_dl` shuffle before the step loop; A3 draws a `Beta` λ +
   permutation each step). Their paired delta carries a seed-noise-level data-order/regularisation
   component on top of the objective swap (mod-dropout precedent). Head-INIT matched; averages
   over 3 seeds. Read A1/A3 as "floor-objective vs +arm-objective under matched init,
   treatment-shuffled order."
4. A3's `_manifold_mixup_bce` re-forwards the classifier a second time (the mixup BCE logit comes
   from a fresh forward on the mixed rep, so line-32's `output` is computed-but-unused-for-loss in
   A3 — a few dead flops). Inherent to manifold mixup; harmless; does not affect the triplet term
   (real feats) or the floor. Informational.

---

## 1. Sha re-verification at submit time — ALL MATCH (authorization intact)

Re-ran `sha256sum` on the prereg (self-sha), artifacts A/B/C, and the reused-unchanged machinery.
**Every hash matches the frozen block in `NCA_FREEZE.md`; authorization is intact.**

### Prereg (self-sha) + frozen artifacts A/B/C
```
FROZEN 7607863c15bef1d40c1f1f0a5b980123dd84aba260ced6db1fa993f277db5591  refine-logs/NCA_PREREG.md               [MATCH]
A      e1244adadf16b47c24b05786d1ee4e153fd9c696e3be0924eae43c82f1c3b75b  src/model/loss.py                       [MATCH]
B      b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py                          [MATCH]
C      baf41be8a1264445d6bbc63d2eb54966abed6da33c52176ae62d93bf79c14b94  scripts/slurm/ncafam_family.sbatch      [MATCH]
```

### Reused-unchanged machinery (NOT edited; do NOT edit)
```
classifier.py  e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py                    [MATCH]
retrieval.py   d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57  src/utils/retrieval.py                     [MATCH]
anchor         00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02  scripts/slurm/enc3seed_lora_curric.sbatch  [MATCH]
```

Header verification (prereg §6 resource plan): `ncafam_family.sbatch` requests `--cpus-per-task=8`,
`--mem=64G`, `--gres=gpu:a100:1`, and carries **NO `--time`** (L2-8). ONE combined job (24 head runs
sequential inside); peak footprint 8 CPU / 64 G / 1 GPU — within the 16/128/2 cap, never two
16-CPU jobs.

## 2. MANDATORY CODEX GATE (prereg §4.5) — **BLOCKING A3 FINDING; SUBMISSION HALTED**

Ran the house `codex-code-review` pattern (CLI, NOT MCP — `codex exec`), model **`gpt-5.4`**,
`model_reasoning_effort=xhigh`, `--full-auto`, two iterative rounds (sessions
`019f95d0-628d-7f61-a288-09630fe0632a` round 1, `019f95d9-b88b-7040-a3d6-254c5df0af22` round 2;
codex-cli 0.144.1). Codex READ the actual source (`loss.py`, `run_rac.py`, `classifier.py`,
`retrieval.py`, `metrics.py`) with full context and cited file:line for every conclusion, and ran a
runtime probe in the pinned env. Transcripts: session scratchpad `codex_nca_review1.txt`,
`codex_nca_review2.txt`.

**ROUND 1 VERDICT: NO P1 bugs; 6 P2 + 2 P3 findings.** Codex verbatim: *"No P1 bugs showed up in the
new flag-gated NCA/SupCon/mixup path."* All load-bearing invariants for A1 (NCA) and A2 (SupCon)
verified with citations: `_build_nca_bank` unique-id assert + mode restore; LOO id->row `-inf`
self-mask correct + fails loud on missing id (self-mass exactly 0 in the runtime probe); bank
stop-grad complete both layers (hostile `requires_grad=True` bank grad == None); once-per-epoch bank
cadence gated to `head_loss=='nca'`; A1/A2 early-return 7-tuple matches the normal return and leaves
mining inert; eval/selection routes through `retrieve_evaluate_RAC_` (training-only self-mask); A3
re-derives the align Hadamard fusion; flags-off byte-identity holds.

**ROUND 2 (disposition confirmation against the exact deployed sbatch flags): codex REFUTED the
"zero live findings" bottom line and escalated ONE finding to BLOCKING for the A3 (mixup) arm.**

Disposition of every round-1 finding under the frozen deployed config (codex round-2, agreed by me
after independent source re-verification):

| # | finding | disposition under this config | status |
|---|---|---|---|
| 1 | early-return skips `lambda_seg`/`lambda_aux`/`lambda_tarc` | `lambda_seg 0`, no aux_pack, no target_pack -> skipped tail is all no-op guards (run_rac.py:1177/1287/1378); A3 keeps `head_loss='triplet'` so does not early-return | INACTIVE |
| 2 | `--sam` incompatible with mining-inert A1/A2 | `--sam` not passed (default False, run_rac.py:535/742); safe fail-loud assert anyway | INACTIVE |
| 3 | `_supcon_head_loss` detached-zero no-grad | `hybrid_loss=True`, `ce_weight=0.5` -> BCE keeps total grad-connected (loss.py:52); batch-64 binary -> `valid.any()` always true | INACTIVE |
| 4 | mixup first `output` forward advances dropout RNG | subsumed by the blocker below (round 1 framed it as RNG-only; round 2 found the stronger eval-mode bug) | **see BLOCKER** |
| 5 | `dense_retrieve_...` leaves model in eval mode | pre-existing frozen reused-unchanged; benign on the FLOOR (its BCE uses the train-mode `output` from loss.py:32, mining runs after); but see BLOCKER — A3 does a SECOND head forward AFTER mining | **see BLOCKER** |
| 6 | GPU FAISS rebuild not fully detached | `--Faiss_GPU False` -> CPU path detaches (retrieval.py:337/363/377) | INACTIVE |
| 7 | NCA clamp all-`-inf` singleton-class | binary datasets, hundreds/class -> every anchor keeps same-class neighbours after LOO | INACTIVE (P3) |
| 8 | mixup mode `assert` stripped under `-O` | plain `python` (no `-O`), `fusion_mode=align` set -> assert passes | INACTIVE (P3) |

### 2.1 THE BLOCKER — A3 (mixup) BCE forward runs with the classifier's dropout DISABLED (eval mode)

**Confirmed by independent source re-read (all three legs), not accepted on codex's word alone:**

1. **The mixup forward uses dropout-bearing head submodules.** `classifier_hateClipper`:
   `img_proj`/`text_proj` each `= nn.Sequential(nn.Linear, nn.Dropout(dropout[0]=0.2))`
   (classifier.py:81-82); `mlp` starts `nn.Dropout(dropout[1]=0.4)` then per-layer
   `nn.Dropout(dropout[2]=0.1)` (classifier.py:96,103). `_manifold_mixup_bce` forwards through exactly
   `model.img_proj`, `model.text_proj`, `model.mlp`, `model.output_layer` (loss.py:709-717). These
   dropouts are train/eval-mode sensitive.
2. **Mining leaves the model in eval mode and never restores it.**
   `dense_retrieve_hard_negatives_pseudo_positive` calls `model.eval()` (retrieval.py:330) and returns
   at retrieval.py:582/584 WITHOUT restoring the prior mode.
3. **In A3 the mixup forward runs AFTER mining with no intervening `model.train()`.** For A3
   (`head_loss='triplet'`, `mixup=True`) the ordering inside `compute_loss` is:
   `model.train()` (loss.py:31) -> train-mode forward (loss.py:32) -> [head_loss=='triplet' so NO
   early return] -> `dense_retrieve_...` mining (loss.py:310, since `no_pseudo_gold_positives=1>0`) ->
   **model now in EVAL mode** -> triplet assembly (pure tensor ops) -> hybrid block ->
   `_manifold_mixup_bce` (loss.py:585). `grep` confirms the ONLY `model.train()` in `loss.py` are
   line 31 (before mining) and line 973 (inside `compute_segment_loss`, NOT called at `lambda_seg=0`).
   So the A3 mixup BCE forward executes with img_proj/text_proj/mlp **dropout OFF**.

**Why this is BLOCKING (not inactive, not a shared-with-floor no-op):**
- The FLOOR's BCE (`mixup=False`, else-branch loss.py:588-594) uses `output` from the **train-mode**
  forward at loss.py:32 -> classifier dropout **ON**. The mixup arm's BCE forward has dropout **OFF**.
  So A3 differs from its floor in TWO ways: (intended) manifold-mixup interpolation, AND (UNINTENDED)
  the classifier BCE path loses its dropout regularisation. A3 is therefore NOT a clean "floor + mixup"
  ablation — the measured A3 delta conflates mixup with dropout-disabling. It is A3-specific (A1/A2
  early-return before mining; the floor has no second post-mining head forward inside `compute_loss`),
  so it is NOT shared-identically with the paired control.
- It directly contradicts the prereg's binding claim (DEV-3 §11, echoed in the review §7): that
  `_manifold_mixup_bce` reproduces "**exactly the deployed align forward**." The deployed align forward
  during training runs in train mode with dropout ON; the mixup re-forward runs in eval mode with
  dropout OFF. The eval mode is an ACCIDENT of the mining leak, not a declared design choice (the
  prereg neither sets eval nor wraps the mixup forward in `no_grad`).
- It fires in EVERY A3 run under the exact frozen config, does NOT crash/NaN, and only manifests in the
  full `compute_loss` ordering — so the §4.4 GPU smoke (loss finite/decreasing, completes) would have
  PASSED without catching it, and codex's round-1 UNIT probe of `_manifold_mixup_bce` in isolation
  (model in train mode) also missed it. This is exactly the confound the mandatory codex gate exists to
  catch.

Per prereg §4.5 ("Blocking findings ⇒ fix the code + re-freeze the shas (§5) + re-run this gate") and
the executor mandate ("Blocking findings = STOP and report"): **SUBMISSION HALTED. No sbatch submitted.
Frozen code NOT edited by the executor** (any edit to loss.py changes sha A `e1244ada…` and VOIDS the
freeze-derived authorization; the fix + independent-reviewer re-freeze is a separate ceremony). A1/A2
(NCA/SupCon) are clean; the blocker is A3-specific, but the family is ONE frozen sbatch = ONE bite, so
the whole submission is blocked pending the fix + re-freeze.

**Suggested minimal fix (for the prereg author / re-freeze ceremony, NOT applied here):** restore train
mode before the mixup forward — e.g. `model.train()` at the top of `_manifold_mixup_bce`, or reset
`model.train()` in `compute_loss` after the mining call and before the hybrid block. Either changes
sha A and mandates a re-freeze + a re-run of this codex gate.

---

## RESUME (REFREEZE-1) — team-lead authorized 2026-07-25; codex STOP confirmed correct

The blocking A3 finding (§2.1) was fixed (commit `8f08e9f`, `refine-logs/NCA_REFREEZE_FIX.md`),
re-frozen (`NCA_FREEZE.md` REFREEZE-1, new sha A), and independently re-reviewed APPROVED (commit
`467a6f4`, `refine-logs/NCA_REFREEZE_REVIEW.md`, R1-R7 all PASS). Resuming the §4 chain from the top
against REFREEZE-1 values. Two non-blocking re-review notes travel to this gate/executor:
(a) the mixup mode-restore is a plain sequential block, NOT `try/finally` — harmless (an exception on
that path is a fatal per-seed `run_rac` crash that kills the process; there is no in-process recovery
that could silently reuse a train-mode-leaked model);
(b) `NCA_REFREEZE_FIX.md` E2/E3 evidence harness used `num_layers=2` (5 dropouts) vs the deployed
`num_layers=3` (6 dropouts); the fix enumerates dropouts dynamically (count-agnostic) and R3 re-verified
all 6 toggle ON + restore at the true config.

### 2.2 Sha re-verification at RESUME — ALL MATCH REFREEZE-1 (authorization intact)

```
FROZEN 7607863c15bef1d40c1f1f0a5b980123dd84aba260ced6db1fa993f277db5591  refine-logs/NCA_PREREG.md               [MATCH, prereg NOT modified]
A      2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b  src/model/loss.py                       [MATCH REFREEZE-1 new sha A]
B      b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py                          [MATCH original frozen B]
C      baf41be8a1264445d6bbc63d2eb54966abed6da33c52176ae62d93bf79c14b94  scripts/slurm/ncafam_family.sbatch      [MATCH original frozen C]
reused e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py                 [MATCH]
reused d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57  src/utils/retrieval.py                  [MATCH]
anchor 00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02  scripts/slurm/enc3seed_lora_curric.sbatch [MATCH]
```

### 2.3 MANDATORY CODEX RE-GATE (prereg §4.5; code changed ⇒ FULL gate) — CLEARED (no P1; 2 P2 inactive under deployed config)

Ran the house `codex-code-review` pattern (`codex exec`, model `gpt-5.4`, `xhigh`, `--full-auto`),
FULL gate (not a delta) on the patched `loss.py` — session `019f95fa-803c-79f1-9ef5-75c45c643b23`.
Codex was given the diff context (the 18-line REFREEZE-1 mixup dropout-mode guard), the two travel
notes, the exact deployed flags, and instructed to (A) vet the fix, (B) confirm floor/A1/A2 unbroken,
(C) re-confirm the original A1/A2 invariants, (D) confirm count-agnosticism. It READ the actual source
and RAN a runtime probe in the pinned env. Transcript: session scratchpad `codex_nca_regate.txt`.

**VERDICT: NO P1 blocker remains.** Codex verbatim: *"No P1 blocker remains. Under the stated deployed
flags, this fix cleanly repairs the prior A3 dropout-off bug without breaking the floor or A1/A2."* and
*"I agree the A3 fix resolves the prior blocker with no new blocking issue for this unattended job."*

Codex's independently VERIFIED list (citations):
- **The fix does exactly what it claims** (loss.py:708 assert align; :721 enumerate all `nn.Dropout`;
  :722 snapshot prior `.training`; :723 force-train Dropout only; :725/:726/:733 the dropout-bearing
  forward; :734 restore each Dropout's EXACT prior mode; restore point AFTER the last dropout op and
  BEFORE the BCE math at :736).
- **Runtime probe on the ACTUAL `_manifold_mixup_bce`** on `classifier_hateClipper(align,
  batch_norm=False, num_layers=3)` forced into eval mode (post-mining leak): **all 6 dropout hooks
  `training=True` DURING the mixup forward, all 6 restored to `False` after**, loss finite,
  `lambda=0.6748 in [0,1]`, grads reach the first MLP linear + `output_layer`.
- **A4**: triplet term still reads REAL un-mixed feats (loss.py:31/32 → mining :285/:310 → triplet
  :486); mixup only rewrites the BCE side (:578/:585/:596); no detach between `x_mix` and mlp/output.
- **A5 (try/finally note)**: acceptable, NOT blocking — `set -euo pipefail` (sbatch:26), each arm a
  separate `python ./src/run_rac.py` process (sbatch:79), one process at a time (sbatch:117) ⇒ any
  exception in the enable/restore window is a fatal process exit, no in-process reuse of a leaked model.
- **B1/B2**: floor (mixup=False) + A1/A2 (early-return at loss.py:43/63 BEFORE mining and the mixup
  hook) NEVER call `_manifold_mixup_bce` ⇒ the 18-line fix cannot touch their mode/RNG; flag-off BCE
  still uses the train-mode `output` from loss.py:32.
- **C**: original A1/A2 invariants all re-confirmed — `_build_nca_bank` once/epoch gated `head_loss=='nca'`
  + mode-restore + unique-id assert; `_nca_head_loss` bank-detach + id→row `-inf` self-mask (KeyError on
  missing id; hostile `requires_grad` bank got zero grad in the probe); `_supcon_head_loss` sound;
  eval/selection routes through `retrieve_evaluate_RAC_`, not the training-only mask.
- **D**: fix is count-agnostic (`model.modules()` enumeration), so the fix-record harness's
  `num_layers=2` under-count does not weaken correctness at the deployed `num_layers=3`.

**Two P2s — BOTH inactive/latent under this family's frozen config (codex-agreed; NOT blocking; NOT
fixed — fixing would edit frozen artifact A and force another re-freeze, and neither changes behaviour
here):**
1. *P2 (whole-model eval not restored — only Dropout is)*: the fix restores `nn.Dropout` submodules
   but not the model's global `.training` flag. **Inactive**: `--batch_norm False` (sbatch:86) ⇒ no
   `nn.BatchNorm1d` is constructed (classifier.py:100 branch), so Dropout is the only mode-sensitive
   module; and the global flag is reset by `model.train()` at loss.py:31 at the next step's
   `compute_loss` (no forward runs between the mixup hook and that reset). The Dropout-only variant is
   deliberately chosen (fix record §2.1) as strictly robust even if `--batch_norm` were flipped.
2. *P2 (nca|supcon + mixup combo silently bypasses mixup via the early return)*: latent only — the
   frozen sbatch never combines `--head_loss nca|supcon` with `--mixup True` (A1/A2 rows set no
   `--mixup`; the A3 row sets no `--head_loss`). Cannot fire under this family.

**Both sides reviewed the final (REFREEZE-1, unchanged) code; codex reports no P1; I agree (no P1
missed, no false dismissal); the fix is verified to resolve the prior blocker; both P2s accepted as
inactive-under-deployed-config with documented justification. Per prereg §4.5 there are NO blocking
findings ⇒ no code fix ⇒ no re-freeze ⇒ RE-GATE CLEARED.** Artifacts A/B/C shas UNCHANGED by the gate
(A still `2ae7a73f…`, B/C original; authorization intact).

## 3. Collision-safety re-check at RESUME — CLEAN (all ABSENT); banked inputs PRESENT

- `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_ncafam*` — ABSENT ⇒ fresh verdict group.
- `slurm/logs/nca_*.trainlog` — ABSENT (0) ⇒ no trainlog collision.
- `_smoke_nca` groups / `nca_smoke_*` logs — ABSENT (0) ⇒ no smoke residue at re-check time.
- Banked ZH cache `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`
  — PRESENT (3/3), read-only paired input.
- Banked HateMM cache
  `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` —
  PRESENT (3/3), read-only paired input.
- Floor trainlogs (13150 ZH generic-LoRA, 13241 HateMM curric-LoRA) — PRESENT (6/6), read-only.

`--force False` would hard-abort (never overwrite) if a `RAC_video_ncafam` path ever pre-existed; the
`nca_${ARM}_…` prefix + arm-tagged `exp_comment` keep the four arms (same dataset/seed) distinct.

## 4. Smoke (prereg §4.4, + A3 dropout-ON assertion) — CPU PASS; GPU in progress

### 4.1 $0-CPU checks (login node, `CUDA_VISIBLE_DEVICES=""`) — PASS

Script: session scratchpad `nca_smoke_cpu.py` (imports the CURRENT frozen `run_rac.parse_args`).
- **`python -m py_compile src/model/loss.py src/run_rac.py` = COMPILE_OK.**
- **(1) 4 new argparse keys INERT by default** on the ncafam base command (ARM_FLAGS=""):
  `head_loss='triplet'`, `nca_tau=0.1`, `mixup=False`, `mixup_alpha=0.0`. PASS.
- **(2) no-flag Namespace equivalence (§4.1b / §4.4.2):** ncafam base vs the `enc3seed_lora_curric`
  floor command (same model/seed) differ in **ONLY** `{group_name}` (`RAC_video_lora_curric` →
  `RAC_video_ncafam`); every other arg byte-identical, the 4 new keys present at inert defaults in
  both ⇒ flags-off Namespace is floor-identical ⇒ banked floors need NO re-run. PASS.
- **(3) additive-gating:** each arm's flags flip EXACTLY the intended keys and nothing else —
  A1a `head_loss→nca`; A1b `head_loss→nca, nca_tau→0.2`; A2 `head_loss→supcon`; A3
  `mixup→True, mixup_alpha→2.0`. PASS.

`CPU_SMOKE: PASS`.

### 4.2 GPU smoke (prereg §4.4.1 + the added A3 dropout-ON assertion) — PASS; artifacts deleted

Throwaway smoke sbatch (session scratchpad `nca_smoke.sbatch`; group `_smoke_nca`, `--epochs 3`, seed 0,
ZH cache; 5 runs = no-flag baseline + the 4 arms) + the targeted harness (`nca_smoke_harness.py`,
temporary forward-hooks NEVER in frozen code). Job **13480** (`nca_smoke`): auto-released from
`JobHeldUser` (never forced; queue empty), **COMPLETED exit 0:0**, elapsed 01:13:04 — of which the 5 NCA
runs took ~2 s each (tqdm) and the ~70-min wall was `disk_guard.sh` housekeeping (usage 272G > 250G
threshold ⇒ push-verify-prune of 417 old checkpoints to B2, each ~10 s; derived artifacts only, videos
never leave). `======== nca_smoke ALL DONE (13480) ========`.

**Run-level (§4.4.1 i/ii), all 5 runs — finite, complete, 0 NaN/Inf, 0 Traceback/Assert (3 epochs each,
6 Test_Retrieval evals each):**

| arm (flags) | Train Loss ep0→ep1→ep2 (step-0 snapshots) | NaN/Inf | Err | completes |
|---|---|---|---|---|
| baseline (none) | 0.878880 → … → 0.621143 | 0 | 0 | ✓ |
| A1a nca_tau0.1 | 0.639782 → 0.620198 → 0.664219 | 0 | 0 | ✓ |
| A1b nca_tau0.2 | 0.639783 → … → 0.664206 | 0 | 0 | ✓ |
| A2 supcon_tau0.1 | 2.418714 → … → 2.410412 | 0 | 0 | ✓ |
| A3 mixup_a2.0 | 0.878757 → … → 0.624600 | 0 | 0 | ✓ |

baseline/mixup decrease clearly (BCE-driven); the NCA/SupCon contrastive terms are finite and
non-diverging in a narrow band at the sparse per-epoch step-0 prints (only ~9 batches/epoch, log_interval
10 ⇒ one print/epoch on a per-epoch-rebuilt-bank moving-target loss). The head IS training: nca_tau0.1
Val_Retrieval acc 0.8205→0.8205→0.8333, Test_Retrieval 0.8121→0.8523→0.8054 (metrics move across the 3
epochs). A1/A2 build the per-epoch bank and do NOT enter FAISS mining (harness C4); A3 runs the
triplet+mining path and completes.

**Targeted harness (§4.4.1 iii + §4.4 asserts + the team-lead A3 dropout-ON add) — HARNESS_VERDICT: PASS.**
Real model `classifier_hateClipper(align, batch_norm=False, num_layers=3)` (6 Dropout), real ZH batch,
REAL frozen `_manifold_mixup_bce` / `_nca_head_loss` / `compute_loss`:
- **C1 A3-dropout (the REFREEZE-1 fix, in the real function):** `training_DURING_mixup_forward=[True×6]`,
  `restored_to_eval_after=True (after=[False×6])`, `mixup_loss=0.692520 finite`, `lam=0.721763 in[0,1]`.
  **PASS.**
- **C2 NCA-LOO:** `max_retained_self_softmax_mass=0.000e+00 (<1e-6)` at max self-similarity (anchor==own
  row); real nca loss 0.710227 finite. **PASS.**
- **C3 NCA bank stop-grad:** `hostile_bank.grad=None` (zero); `anchor.grad_sum=0.1218 (>0)`. **PASS.**
- **C4 mining-inert:** `dense_retrieve calls under head_loss=nca=0`; `train_feats=None train_labels=None`;
  total_loss 0.701431 finite. **PASS.**

**Cleanup:** `logging/Retrieval/MHC_zh/_smoke_nca` + `slurm/logs/nca_smoke_*_13480.trainlog` + the `.out`
**deleted**; §4.3 collision targets re-verified ABSENT after deletion (`_smoke_nca` / `nca_smoke_*` /
`RAC_video_ncafam` / `nca_*.trainlog` all 0); banked ZH + HateMM caches re-checked PRESENT (6/6, read-only
inputs — the smoke read them, never wrote). Throwaway `nca_smoke.sbatch` / `nca_smoke_harness.py` /
`nca_smoke_cpu.py` live only in the session scratchpad.

**SMOKE_VERDICT: PASS.** No C1-C4 failure. Cleared to submit the real family job.

## 5. Real family — single-submitted (prereg §6 / §1.1; NO `--time`; 24 head runs sequential)

Submit-instant `sha256sum` re-verified — prereg `7607863c…`, A `2ae7a73f…` (REFREEZE-1), B `b85eb72a…`,
C `baf41be8…`, classifier.py `e7b61df4…`, retrieval.py `d43e3bc4…` [ALL MATCH]; `bash -n` C = SYNTAX_OK;
queue EMPTY (0 jobs — no 16-CPU conflict, standing infra rule satisfied); authorization intact.

| job | id | script | runs | CPU/mem/GPU | ~cost |
|---|---|---|---|---|---|
| ncafam family | **13482** | `ncafam_family.sbatch` (4 arms × 2 datasets × 3 seeds = 24 head runs sequential, hardcoded CONFIGS, group `RAC_video_ncafam`) → `slurm/logs/nca_{nca_tau0.1,nca_tau0.2,supcon_tau0.1,mixup_a2.0}_{MHC_zh,HateMM}_<MODEL>_seed{0,1,2}_13482.trainlog` | 24 | 8 CPU / 64 G / 1×A100 | ~0.33 GPU-h |

Peak footprint 8 CPU / 64 G / 1 GPU (within the 16/128/2 cap; never two 16-CPU jobs — queue EMPTY at
submit). ONE sbatch = ONE family = ONE multiplicity bite. `PENDING (JobHeldUser)` → waited out, NEVER
forced (per CLAUDE.md); if held > 2 h a status line is committed and the turn ends PENDING-JOB.

## 6. RAW per-seed both-protocol numbers (executor transcription — NO gates / NO deltas / NO pass-fail)

**13482 COMPLETED exit 0:0, elapsed 00:19:19; 24/24 trainlogs written; 0 AssertionError / 0 Traceback
across all 24.** Val-selected = epoch ≥ warmup 5 with max `Val_Retrieval` acc (roc tie-break) → that
epoch's `Test_Retrieval` line; final-epoch = max epoch (29). acc AND macroF1 are read from the SAME raw
`Test_Retrieval` line (no companion-metric fabrication). Each `(:N)` is the 1-based **grep -n / sed
compatible** (`\n`-counted) line number of that `Test_Retrieval` line in the named trainlog.
**Cross-verified two ways: (i) an independent `\n`-line parser (session `nca_transcribe.py`), (ii) the
sbatch's OWN embedded `RESULT_ROW` parser in `slurm/logs/ncafam_13482.out` — bit-identical (epoch, F1,
acc) for all 24 runs, both protocols; (iii) `grep -n` sed spot-checks of cited lines.** The executor
applies NO gates/deltas/interpretation; the independent 0-context reviewer renders the verdict against the
prereg VERBATIM (and re-parses the trainlogs itself). Floor values (prereg §2.1 ZH / §2.2 HateMM) are the
reviewer's inputs, shown here for reference only — the executor computes NO Δ.

Trainlogs: `slurm/logs/nca_{ARM}_{MHC_zh,HateMM}_<MODEL>_seed{0,1,2}_13482.trainlog`
(ZH `<MODEL>`=`Qwen2.5-VL-7B-Instruct-LoRA_HF`; HateMM `<MODEL>`=`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`).

### 6.1 A1a — NCA τ=0.1 (`--head_loss nca --nca_tau 0.1`)

| dataset | seed | val-sel ep | val-sel acc/mF1 (Test line) | final ep | final acc/mF1 (Test line) |
|---|---|---|---|---|---|
| MHC_zh | 0 | 11 | 0.8389 / 0.8090 (:125) | 29 | 0.8322 / 0.8023 (:270) |
| MHC_zh | 1 | 9 | 0.8322 / 0.7997 (:109) | 29 | 0.8456 / 0.8158 (:270) |
| MHC_zh | 2 | 15 | 0.8591 / 0.8295 (:157) | 29 | 0.8389 / 0.8090 (:270) |
| MHC_zh | **mean** | | **0.8434 / 0.8127** | | **0.8389 / 0.8090** |
| HateMM | 0 | 17 | 0.8884 / 0.8823 (:192) | 29 | 0.8837 / 0.8765 (:301) |
| HateMM | 1 | 27 | 0.8744 / 0.8678 (:280) | 29 | 0.8837 / 0.8771 (:299) |
| HateMM | 2 | 23 | 0.8605 / 0.8547 (:247) | 29 | 0.8744 / 0.8684 (:302) |
| HateMM | **mean** | | **0.8744 / 0.8683** | | **0.8806 / 0.8740** |

### 6.2 A1b — NCA τ=0.2 (`--head_loss nca --nca_tau 0.2`)

| dataset | seed | val-sel ep | val-sel acc/mF1 (Test line) | final ep | final acc/mF1 (Test line) |
|---|---|---|---|---|---|
| MHC_zh | 0 | 12 | 0.8322 / 0.8023 (:133) | 29 | 0.8322 / 0.8023 (:270) |
| MHC_zh | 1 | 10 | 0.8255 / 0.7931 (:118) | 29 | 0.8389 / 0.8090 (:271) |
| MHC_zh | 2 | 15 | 0.8523 / 0.8226 (:156) | 29 | 0.8389 / 0.8113 (:269) |
| MHC_zh | **mean** | | **0.8367 / 0.8060** | | **0.8367 / 0.8075** |
| HateMM | 0 | 28 | 0.8744 / 0.8672 (:290) | 29 | 0.8744 / 0.8666 (:300) |
| HateMM | 1 | 29 | 0.8744 / 0.8678 (:301) | 29 | 0.8744 / 0.8678 (:301) |
| HateMM | 2 | 8 | 0.8837 / 0.8776 (:108) | 29 | 0.8791 / 0.8730 (:298) |
| HateMM | **mean** | | **0.8775 / 0.8709** | | **0.8760 / 0.8691** |

### 6.3 A2 — neighborhood-SupCon τ=0.1 (`--head_loss supcon --nca_tau 0.1`)

| dataset | seed | val-sel ep | val-sel acc/mF1 (Test line) | final ep | final acc/mF1 (Test line) |
|---|---|---|---|---|---|
| MHC_zh | 0 | 9 | 0.8255 / 0.7956 (:109) | 29 | 0.8725 / 0.8436 (:270) |
| MHC_zh | 1 | 12 | 0.8322 / 0.8023 (:132) | 29 | 0.8389 / 0.8113 (:269) |
| MHC_zh | 2 | 13 | 0.8523 / 0.8226 (:141) | 29 | 0.8456 / 0.8158 (:270) |
| MHC_zh | **mean** | | **0.8367 / 0.8068** | | **0.8523 / 0.8236** |
| HateMM | 0 | 15 | 0.8791 / 0.8730 (:175) | 29 | 0.8837 / 0.8776 (:302) |
| HateMM | 1 | 22 | 0.8791 / 0.8724 (:238) | 29 | 0.8791 / 0.8724 (:302) |
| HateMM | 2 | 14 | 0.8791 / 0.8724 (:166) | 29 | 0.8744 / 0.8660 (:302) |
| HateMM | **mean** | | **0.8791 / 0.8726** | | **0.8791 / 0.8720** |

### 6.4 A3 — manifold mixup α=2.0 (`--mixup True --mixup_alpha 2.0`)

| dataset | seed | val-sel ep | val-sel acc/mF1 (Test line) | final ep | final acc/mF1 (Test line) |
|---|---|---|---|---|---|
| MHC_zh | 0 | 20 | 0.8322 / 0.8023 (:199) | 29 | 0.8725 / 0.8458 (:272) |
| MHC_zh | 1 | 21 | 0.8255 / 0.7931 (:206) | 29 | 0.8523 / 0.8226 (:271) |
| MHC_zh | 2 | 29 | 0.8523 / 0.8202 (:269) | 29 | 0.8523 / 0.8202 (:269) |
| MHC_zh | **mean** | | **0.8367 / 0.8052** | | **0.8590 / 0.8295** |
| HateMM | 0 | 15 | 0.8651 / 0.8580 (:174) | 29 | 0.8791 / 0.8730 (:301) |
| HateMM | 1 | 24 | 0.8837 / 0.8781 (:256) | 29 | 0.8791 / 0.8730 (:302) |
| HateMM | 2 | 10 | 0.8744 / 0.8678 (:129) | 29 | 0.8791 / 0.8730 (:301) |
| HateMM | **mean** | | **0.8744 / 0.8680** | | **0.8791 / 0.8730** |

**3-seed means are raw arithmetic summaries of the transcribed per-seed values only** (as in the prereg §2
floor tables); the executor computes NO Δ-vs-floor, applies NO promote/kill bar, and renders NO pass/fail
or KILLED language. Floors (reviewer inputs, prereg §2): ZH val-sel `0.8322/0.8015`, final `0.8456/0.8173`;
HateMM val-sel `0.8775/0.8711`, final `0.8791/0.8726`. The §3 bars + the verdict are the independent
0-context reviewer's job.

## 7. Closeout — CHAIN COMPLETE (raw numbers transcribed; verdict deferred to independent reviewer)

STOP→fix→re-freeze→resume executed cleanly. Sha re-verify at every stage MATCH (initial freeze, then
REFREEZE-1 A `2ae7a73f…`, B/C/reused unchanged, submit-instant re-verify MATCH). Codex STOP (§2.1) confirmed
correct by the team lead; REFREEZE-1 fix + independent re-review APPROVED; **codex re-gate CLEARED** (no P1;
runtime probe confirms the A3 mixup forward now runs with all 6 head dropouts train-on + restored; 2 P2
inactive under config). Collision CLEAN throughout. **CPU smoke PASS** (inert defaults / no-flag Namespace
floor-identical mod group_name / additive-gating). **GPU smoke 13480 PASS** (5 runs finite+complete;
harness C1 A3-dropout-ON, C2 LOO self-mass 0, C3 bank stop-grad 0, C4 mining-inert — all PASS; artifacts
deleted). **Family 13482 COMPLETED exit 0:0** (00:19:19), 24/24 trainlogs, 0 assert/traceback; RAW
both-protocol per-seed numbers transcribed line-numbered for all 8 arm×dataset cells (§6), cross-verified
two ways (independent `\n`-parser == embedded `RESULT_ROW`) + grep -n spot-checks. **NO gates/deltas/pass-fail
applied; NO `state/` mutation; nothing pushed. Zero test-touch beyond the 24 budgeted head reads.** The
independent 0-context reviewer renders the verdict (KS-arm-dead → FORMAL +0.030/+0.030 both-protocol, per
arm×dataset) against `NCA_PREREG.md` VERBATIM.
