# VERA and CMHKF on DeHate (2026-11-18)

Two cells of the main table are empty: VERA and CMHKF on DeHate. Both baselines ran in August, before DeHate was
added (2026-09-26, external validation; main corpus since 2026-11-01), and only four baselines (MACIL-SD,
MultiHateLoc, DSANet, Fed-WSVAD) were run on DeHate then. The user asked on 2026-11-18 to fill both cells. No method
change; only the main table changes.

## 1. What runs

- **CMHKF** (ACL 2025, upstream `ssp-seven/CMHKF` at `3b07707`, adapter `scripts/reproduction_baselines/cmhkf_adapter.py`
  unchanged). The same protocol as the four DeHate baselines (`experiments/20260926_dehate_external/README.md`
  section 2): 5 Optuna trials at seed 234 selected on validation video AP (user, 2026-09-26), then the winner
  retrained at seeds 234 / 2025 / 3407 and scored on test once by `scripts/reproduction_baselines/eval_baseline_scores.py`.
  - Launch: `launch/cmhkf_dehate.sbatch` on uoa-lab1 (Slurm partition `local-sc474397`). Outputs
    `runs/20261118_dehate_vera_cmhkf/cmhkf/{tuning,final}`.
  - One deviation from `20260926_dehate_external/launch/baseline.sh`: `--max-new-attempts 40`. CMHKF's guard prunes
    every 128-step temporal choice (its fusion block is fixed at 256 steps), about half of DeHate's four temporal
    choices, and the default budget of 10 attempts would stop before 5 trials complete with probability about .38.
    Pruned attempts train nothing.
  - Cost: DeHate has 4680 training videos (HateMM 744); a CMHKF trial took about 40 min on HateMM, so about 4 h
    here, about 32 h for 5 trials and 3 seeds. The job limit is 1 day; the job resumes where it stopped.
- **VERA** (CVPR 2025): see section 3.

## 2. Machines (2026-11-18)

- uoa-lab2 (local): GPU held by another user's job until at most 2026-11-19 00:06; holds the DeHate videos.
- uoa-lab3: Slurm node DOWN since 2026-11-16 16:46 ("Not responding").
- uoa-lab1: GPU idle; DeHate features, splits and gold in `results/reproduction`; disk 14 GB free; no DeHate videos.
- lab-server: Slurm node down.
- `third_party/cmhkf` copied to uoa-lab1 from `~/Hate-follow-up/third_party/cmhkf` (upstream clone at `3b07707`).
