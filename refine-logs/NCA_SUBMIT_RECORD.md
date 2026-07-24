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

## 3. Collision-safety re-check at submit — NOT REACHED (submission halted at the codex gate)

## 4. Smoke (prereg §4.4) — NOT REACHED (submission halted at the codex gate)

## 5. Real family — single-submit — NOT SUBMITTED (blocked by the codex gate)

## 6. RAW per-seed both-protocol numbers — NONE (no job submitted; zero test-touch spent)

## 7. Closeout — HALTED PENDING FIX + RE-FREEZE

Sha re-verification PASS (§1, all A/B/C + reused hashes MATCH — authorization was intact at gate
time). Mandatory codex gate (§2): NO P1; A1/A2 clean; **one BLOCKING A3 finding** (mixup BCE forward
runs with classifier dropout disabled due to the mining eval-mode leak). Per §4.5 the executor STOPS
and reports; the fix requires editing frozen artifact A (loss.py) and therefore a re-issued freeze
block + a re-run codex gate before any submission. **Zero GPU/SLURM spent. No job submitted. No
`state/` mutation. Zero test-touch. Nothing pushed.**
