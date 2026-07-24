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
