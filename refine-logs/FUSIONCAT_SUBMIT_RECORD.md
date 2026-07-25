# FUSIONCAT — SUBMIT EXECUTION RECORD

Executor: fusioncat submit executor (Opus 4.8). Working tree `/data/jehc223/RGCL`, conda `HateVideo`.
Frozen family: `refine-logs/FUSIONCAT_FREEZE.md` (APPROVED-WITH-NOTES). Prereg: `refine-logs/FUSIONCAT_PREREG.md`.
Chain script = prereg §4 / §6. NO state/ mutation, NO push. Raw facts only; verdict is an independent reviewer's job.

---

## Authorization (S0) — VERIFIED @ HEAD `aa132f9`

Re-ran `sha256sum` + `git status --porcelain src/` at submit-executor start. ALL match the freeze block byte-exact:

| artifact | expected (freeze) | measured | match |
|---|---|---|---|
| `refine-logs/FUSIONCAT_PREREG.md` | `c88332b8…433c0830` | `c88332b8972e3270081600d0a8cb892a8d24afefbc73e378a5a3104a433c0830` | ✓ |
| `scripts/slurm/fusioncat_family.sbatch` | `62bfb773…2517fc` | `62bfb773beeec325096362c86bae1aca8b90c94804ba3798973bbc88e82517fc` | ✓ |
| `src/model/classifier.py` | `e7b61df4…` | `e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378` | ✓ |
| `src/run_rac.py` | `b85eb72a…` | `b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3` | ✓ |
| `src/model/loss.py` | `2ae7a73f…` | `2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b` | ✓ |
| `src/utils/retrieval.py` | `d43e3bc4…` | `d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57` | ✓ |
| `git status --porcelain src/` | empty | empty (CLEAN) | ✓ |

VOID-on-edit clause (freeze §26-39): none of the 3 VOID conditions triggered ⇒ authorization VALID.
NO codex gate (prereg §4.5: zero-code flag-only family, reviewer-confirmed exemption).

---

## S1 — CPU smoke (prereg §4.4.1) — PASS

- `bash -n scripts/slurm/fusioncat_family.sbatch` → **SYNTAX_OK**.
- CONFIGS word-count = **6 rows** (MHC_zh×{0,1,2} + HateMM×{0,1,2}). ✓
- Collision re-check (all must be ABSENT):
  - `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_fuscat*` → **ABSENT** ✓
  - `slurm/logs/*fuscat*.trainlog` → **ABSENT** ✓
  - `logging/Retrieval/*/RAC_video_smoke_fuscat*` → **ABSENT** ✓
- Banked LoRA feature caches present (read-only inputs):
  - ZH `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` ✓
  - HateMM `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` ✓
- Banked floors present (paired controls):
  - ZH `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog` ✓
  - HateMM `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog` ✓
- Queue at S1: `squeue -u jehc223` = EMPTY (no 16-CPU concurrency concern; never-2×16-CPU trivially clear).

S1 verdict: **PASS** — proceed to S2 GPU smoke.

---
