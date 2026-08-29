# HEAD-RECIPE (SAM + modality-dropout) — HASH-FREEZE

**Frozen by:** independent 0-context pre-registration reviewer.
**Date:** 2026-07-25 NZST.
**Verdict:** **APPROVED-WITH-NOTES** (see `refine-logs/HEADRECIPE_PREREG_REVIEW.md`; four non-blocking notes).
**Prereg:** `refine-logs/HEADRECIPE_PREREG.md` (commit `83bb76e`, clean/unmodified at freeze).
**Recon:** `refine-logs/HEADRECIPE_FORENSIC_RECON.md` (`44918e0`).
**Prereg NOT modified** (freeze recorded here, per review mandate).

## FROZEN block (re-verified on disk at freeze time — all match)

```
FROZEN 68be61ac5ad77d2cfc66f3b355b574855bb426ed2c2a6e6b89f8050347ba232d  refine-logs/HEADRECIPE_PREREG.md
A 1012c9e378905e5c10a0447475560de4a32904af691e457bf4ce77a3d36cc20d  src/run_rac.py
B e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py
C c88f685f68f83611fde3f91751f330d30b6be278693a405f4b9fb80f53ebb009  scripts/slurm/headrecipe_family.sbatch
```

## Reused-unchanged machinery (re-verify at submit; do NOT edit)

```
48796638fdd60fcfb313e97e7f89d73226d96f23369f8c8ebb61ca5814f9cd64  src/model/loss.py            (compute_loss; SAM re-uses as-is)
d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57  src/utils/retrieval.py       (FAISS re-mine gate :341; the F0.6 invariant)
19c76b177f7dc883a9e03524450ad2e6cb302cdd0a6704d69da68a62188a06fc  scripts/slurm/enc3seed_lora_hatemm.sbatch  (same-code anchor)
```

Banked paired-input caches (read-only; NOT clobbered by this family):
- ZH floor: `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`
- HateMM floor: `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt`

Banked floor trainlogs (paired controls, re-derived to 4dp; NOT re-run):
- `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog`
- `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`

## Independently re-derived floors (4dp; both protocols; bit-match prereg §2.1/§2.2)

| leg | protocol | mean acc/mF1 |
|---|---|---|
| ZH 13150 (generic-LoRA) | val-sel | 0.8322 / 0.8015 |
| ZH 13150 | final | 0.8456 / 0.8173 |
| HateMM 13241 (curric-LoRA) | val-sel | 0.8775 / 0.8711 |
| HateMM 13241 | final | 0.8791 / 0.8726 |

## Submit conditions (binding on the executor)

1. **Re-verify all shas** in the FROZEN + reused-unchanged blocks with `sha256sum` at submit time; any
   mismatch = authorization VOID.
2. **Codex gate FIRST** (§4.5) on the SAM double-step + re-mine-reuse branch, iterative until Claude+Codex
   agree. Any blocking finding ⇒ fix code + re-freeze A/B/C shas here + re-run the gate before submit.
3. **Smoke** (§4.4): ARM A double-step visible + assert does not trip + loss finite; ARM B loss finite +
   $0 mask-rate line; no-flag Namespace equivalence. Delete all `_smoke_hr` groups / `hr_smoke_*` logs.
4. **Re-check collisions absent** (`logging/Retrieval/{MHC_zh,HateMM}/RAC_video_headrecipe*`,
   `slurm/logs/hr_*.trainlog`, `_smoke_hr*`).
5. **ONE** `sbatch scripts/slurm/headrecipe_family.sbatch` — 12 head runs sequential, 8 CPU / 64 G / 1 A100,
   **NO `--time`**; `PENDING (JobHeldUser)` → wait for auto-release, **never force**.
6. Executor transcribes RAW both-protocol per-seed numbers (line-numbered); applies NO gates/interpretation.
   Verdict (KS-arm-dead → FORMAL +0.030/+0.030 both-protocol, per arm×dataset) rendered by an independent
   0-context reviewer against the prereg VERBATIM.

**Reviewer statements:** ZERO GPU/SLURM/Modal spent; no test metric produced; `state/` and
`autoresearch/goal_mllm_plus3/state/` not touched; prereg NOT modified; no job submitted; not pushed.
