# S2S Stage-E — Smoke Record

Kept as a **separate** file so the hash-frozen `S2S_PROBE_DESIGN.md` (r2) is not disturbed.
Authorization: `S2S_CODE_REVIEW.md` §5 (SMOKE AUTHORIZED after B1–B3 confirmed). Single submission;
throwaway `--out_root`; no `--time`; `PENDING (JobHeldUser)` = wait, never force. Executor transcribes
RAW gate values; the binding verdict is rendered independently.

## Submission

| field | value |
|---|---|
| job id | **13159** |
| submit ts (UTC) | 2026-07-14T11:14:03Z |
| command | `SMOKE=1 sbatch scripts/slurm/s2s_extract.sbatch` |
| out_root (throwaway) | `/data/jehc223/RGCL/slurm/logs/s2s_smoke_out_13159` |
| log | `/data/jehc223/RGCL/slurm/logs/s2s_extract_13159.log` |
| `scripts/analysis/s2s_extract.py` sha256 (r2 pin) | `41979f6a41c95e38a3cd875e11dc54a5a48eac9a5b908f295bad4d8d051cd23a` |
| `scripts/slurm/s2s_extract.sbatch` sha256 (r2 pin) | `2dc0f90b03a44f45945cab3194f78ec97012fe7b157727cd50f64d88d56665dc` |
| initial state | `PENDING (JobHeldUser)` — awaiting auto-release (never force) |

Note: the sbatch echoes "(G-recon skipped)" at lines 20/49 are a **known-cosmetic stale residual**
(flagged by the reviewer); after fix B2 the banked cache loads unconditionally, so **G-recon DOES run in
this smoke**. Gate 2 is read from the per-split assembly line `grecon_cos_min=... grecon_maxabs_max=...`,
not from those echoes.

## Attempt 1 — job 13159: FAILED at gate 0a (RAW)

| field | value |
|---|---|
| final state | **FAILED**, ExitCode 1:0, Elapsed 00:00:13 |
| where | `s2s_extract.py:199` in `encode_frameset`, called from `temporal_positive_control` (`:259`, `banked_vec=None`) |
| raw error | `recon_mean = (g * n_t_t[:, None]).sum(0).add(p_S).div(float(end))` → `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` |
| gate 0a temporal positive control | **DID NOT COMPLETE** — crashed before the σ check |
| gates 0b / 1 / 2 | not reached |
| model load / sha echo | OK (model loaded; echoed `41979f6a…` / `2dc0f90b…`, matching the r3 extractor pin) |
| real-path artifact | NONE under `data/CLIP_Embedding/<ds>/frameset_qwen7b_8f/` (crash was pre-data, in the synthetic control) |

**Postmortem.** `n_t_t = torch.tensor(n_t, dtype=torch.float32)` defaulted to CPU while `g`/`p_S` are on
the model device → the G-decomp assembly mixed cuda+cpu. GPU-only bug the CPU synthetic test could not
surface; the smoke caught it in 13 s, pre-data. Fix (r3a): build `n_t_t` on `g.device`. Different site
from the B1 review fix (that aligned the G-recon *compare*). The smoke did its job — no submission wasted
beyond one held-then-13s run, banked caches untouched.

## r3a re-freeze (extractor-only; awaiting reviewer diff re-check before resubmit)

| field | value |
|---|---|
| `scripts/analysis/s2s_extract.py` sha256 (**r3a**, NEW) | `07fd162196a7e61e8e83f1a181408fe7b8080cf475cb59ecd58a1dc035b3740a` |
| `scripts/slurm/s2s_extract.sbatch` sha256 | `2dc0f90b03a44f45945cab3194f78ec97012fe7b157727cd50f64d88d56665dc` (UNCHANGED) |
| status | fix committed (r3a); **NOT resubmitted** — awaiting the team-lead's GO after the reviewer's diff re-check |

## Attempt 2 — job 13169 (r3a fix; reviewer CLEARED §7, team-lead GO)

| field | value |
|---|---|
| job id | **13169** |
| submit ts (cluster UTC) | 2026-07-14T23:49:52Z |
| command | `SMOKE=1 sbatch scripts/slurm/s2s_extract.sbatch` |
| out_root (throwaway) | `/data/jehc223/RGCL/slurm/logs/s2s_smoke_out_13169` |
| log | `/data/jehc223/RGCL/slurm/logs/s2s_extract_13169.log` |
| `s2s_extract.py` sha256 (r3a, pre-submit verified on-disk) | `07fd162196a7e61e8e83f1a181408fe7b8080cf475cb59ecd58a1dc035b3740a` |
| `s2s_extract.sbatch` sha256 (verified on-disk) | `2dc0f90b03a44f45945cab3194f78ec97012fe7b157727cd50f64d88d56665dc` |
| initial state | `PENDING (JobHeldUser)` — waiting for auto-release (never force) |
| waiter | background `bavwhxjrj` (120s sacct poll to terminal) |

### TERMINAL RESULT (RAW)

**FAILED**, ExitCode 1:0, Elapsed 00:00:18. NOT a crash — the r3a device fix held (execution passed the
old line-199 site). Gate 0a failed as a **scientific gate** (`s2s_extract.py:274`):

| gate | expected | observed (raw) |
|---|---|---|
| 0a temporal positive control | `match==σ` with σ=[2,0,3,1] | **FAIL** — `match=[1,0,3,3]` (matrix position-dominated 0.61–0.94; see postmortem) |
| 0b grid-consistency | `n_vis == grid_t·(grid_h//2)·(grid_w//2)` | not reached (0a HALTs first; 0a is pre-real-video) |
| 1 G-decomp | `decomp_res_max` ≤ 1e-5 | not reached |
| 2 G-recon | `grecon_cos_min` ≥ 0.9999 AND `grecon_maxabs_max` ≤ 1e-3 | not reached |

Config echo + sha256 (`07fd1621…`/`2dc0f90b…`) match the r3a pins: **YES**. Model loaded fine.
No artifact under the real `data/CLIP_Embedding/<ds>/frameset_qwen7b_8f/`: **CONFIRMED ABSENT** (0a runs
before any real video; nothing written to the throwaway either).

**Disposition:** zero-GPU forensic postmortem in `refine-logs/S2S_GATE0A_POSTMORTEM.md`. Verdict
hypothesis: **CONTROL-DESIGN FLAW (invalid by construction) + PREMISE-REFRAME (not falsification)** —
`g_t` are cumulative causal-prefix summaries (Qwen LLM `is_causal=True`), so the control's frame-local
permutation-equivariance assumption cannot hold for ANY stimulus; orientation-bug REFUTED
(`match ≠ σ⁻¹=[1,3,0,2]`, and `match` isn't even a permutation). NO patch-and-resubmit; awaiting
independent review before any amendment/resubmit/kill.
