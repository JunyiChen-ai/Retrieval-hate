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

## 2. MANDATORY CODEX GATE (prereg §4.5) — pending

## 3. Collision-safety re-check at submit — pending

## 4. Smoke (prereg §4.4) — pending

## 5. Real family — single-submit — pending

## 6. RAW per-seed both-protocol numbers — pending

## 7. Closeout — pending
