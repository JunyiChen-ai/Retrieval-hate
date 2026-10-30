# Code review: all-trial evaluation, per-setting selection, budget curves (2026-10-31)

One independent reviewer agent (Claude Fable 5.1), rule 6 scope (bugs that change numbers or conclusions only).
Files: `experiments/20261002_complete_io_method/{evaluate.py (--all-trials/--with-val), launch/run_corpus.py,
launch/sbatch_all_trials.sh, budget_curve.py, select_by_setting.py}`.

**Verdict: PASS.** Checked and found correct:
- `evaluate.py` without the new flags behaves exactly as before; with `--all-trials` the two selected trials are not
  evaluated twice; `--with-val` validation calls use the validation-calibrated threshold and validation-only quantile
  reference, nothing from test; the unified evaluator accepts `--split val` with the same GT arrays.
- `select_by_setting.py`: validation basis selects by `val_results` and reports test `results`; checkpoint epochs were
  chosen by validation only during training (`src/qtl/train.py`); glob layout matches what `run_corpus.py` writes;
  original-result lookup paths match the existing outputs.
- `sbatch_all_trials.sh`: source roots give the correct study paths for main corpora and DeHate.
- `budget_curve.py`: the target-8 recalibration follows the same code path as the passed DeHate audit; dedup of
  saturated fixed-k points loses nothing (equal mean calls imply identical per-video calls); adaptive mean calls lie
  inside the fixed range, so interpolation never extrapolates.

Noted, not bugs: exact-equality reproduction asserts could fail on GPU nondeterminism (would block, not corrupt; use a
tolerance and record the deviation if it trips). Applied the two metadata suggestions: `selection_protocol` text
depends on `--all-trials`; assert validation mean calls of the adaptive rule equal the calibration value.
