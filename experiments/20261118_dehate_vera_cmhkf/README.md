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

## 3. VERA

- Code: `vera_dehate.py` is a copy of `~/Hate-follow-up/scripts/reproduction_baselines/vera_adapter.py`, the adapter
  that produced the HateMM and HateClipSeg VERA cells (`docs/duplex/OFFICIAL_VAL_RESULTS.md`). Changed: DeHate in
  `VIDEO_DIRS` (`~/data/DeHate/video`), the CLIP feature root and `hate_common` taken from this repository, the default
  output directory. Unchanged: InternVL2-8B, the three prompt banks (UCF released, XD released, hate-domain), 8
  frames per 10 s window every second, the `Output: 1 / 0` question, greedy decoding, selection on validation video AP,
  visual-neighbour propagation and Gaussian smoothing.
- Protocol as on HateMM / HateClipSeg: `select` asks each of the three prompt banks once per validation video (668
  videos, 212 hateful, video labels) and keeps the one with the highest video AP; `infer` runs the chosen bank on every
  test second (1151 videos, 110,839 windows); `postprocess`; `scripts/reproduction_baselines/eval_baseline_scores.py`.
  Single run (VERA is deterministic and trains nothing), as on the other corpora.
- transformers: InternVL2-8B's remote code (`modeling_internlm2.py`, `InternLM2ForCausalLM` without
  `GenerationMixin`) has no `generate()` under transformers >= 4.50. The HateVideo environment has 4.57.6, installed
  2026-08-29, after the HateMM / HateClipSeg VERA runs (2026-08-26). The job therefore puts transformers 4.45.2 and
  tokenizers 0.20.3 first on `PYTHONPATH` from `third_party/_venv/vera_transformers_4.45.2` (pip `--no-deps --target`,
  log `runs/_setup_uoa-lab2/pip_vera_transformers_4.45.2.log`); the environment itself is not changed.
- Machine: uoa-lab2 (DeHate videos, 9 GB for test + val; InternVL2-8B 16 GB in the Hugging Face cache, download log
  `runs/_setup_uoa-lab2/hf_internvl2_8b.log`). uoa-lab1 has 14 GB free and no DeHate videos. The job waits in the
  `local-sc474399` queue behind another user's job (end at most 2026-11-19 00:06).
- Cost: on HateMM 29,269 windows took about 4 h (raw files 2026-08-26 10:30-14:34), about 2 windows / s; DeHate's
  110,839 windows are about 15 h, plus about 2,000 selection calls.
- Inputs restored on uoa-lab2: `results` was a dangling link to `~/Hate-follow-up/results` (removed); it is now a
  directory like on uoa-lab1 / uoa-lab3, with `results/reproduction/{splits,gt}` and
  `results/reproduction/features/clip_b16_1fps/dehate` copied from uoa-lab1 (rsync, 2026-11-18).
- Launch: `launch/vera_dehate.sbatch`. Outputs `runs/20261118_dehate_vera_cmhkf/vera/{selection,final}`.

## 4. Results

### CMHKF (done 2026-11-18 01:11, uoa-lab1, Slurm job 2424; outputs copied back to uoa-lab2)

- Search: 13 attempts, 5 complete, 8 pruned at once by the 256-step guard, none failed
  (`runs/20261118_dehate_vera_cmhkf/cmhkf/tuning/cmhkf/dehate/best.json`). Best validation video AP .534 (trial 5:
  temporal 256:32, lr 1.18e-4, batch 32, 10 epochs, loss_mil .53, loss_align .83, loss_text 6.1e-4, prompt 20 / 5).
- A trial took about 5 min, not the 4 h estimated in section 1; the whole job took 42 min.
- Test, branch `score_align` (the branch of the HateMM / HateClipSeg CMHKF cells, `aggregate_official_val.py`),
  `final/cmhkf/dehate/seed_<s>/frame_eval.json` (evaluator output), AP / ROC / within:

| seed | AP | ROC | within |
|---|---|---|---|
| 234 | .1354 | .6583 | .5740 |
| 2025 | .1388 | .6702 | .5188 |
| 3407 | .1263 | .6693 | .4902 |
| mean (std) | **.134** (.005) | **.666** (.005) | **.528** (.035) |
