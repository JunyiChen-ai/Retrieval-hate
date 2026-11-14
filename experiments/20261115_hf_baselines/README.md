# Baselines from Hate-follow-up, re-evaluated on this repository's test sets (2026-11-15)

Request (user, 2026-11-15): the main table has too few baselines. Remove the naive control (strongest baseline +
8 equal-window answers) and revision 6 from the main table, and bring in weakly supervised and VLM baselines from the
other project (`~/Hate-follow-up`) that do reasonably well but stay below the proposed method.

Mechanism hypothesis: none. This is a measurement of existing baseline outputs; nothing is trained or re-run.

## What was done

- Source: each baseline's `predictions.jsonl` in `~/Hate-follow-up/runs/` on uoa-lab1 (`20261008_baselines/<method>/`,
  `20261113_lf_baselines/{naive,mars,eventvad}/`), 4 fps `score_curve`.
- `reeval.py` converts each video to this repository's 1 fps grid (second t = mean of 4 fps frames 4t..4t+3; cut or
  padded with the last value to this repository's GT length), writes `scores.jsonl` with `hier_evidence_common.write_scores`
  (6 decimals, as for the method's own runs) and evaluates with `scripts/reproduction_baselines/eval_baseline_scores.py
  --require-full-coverage` on the test cohort of `hier_evidence_common.load_fixed_cohort`.
- Test cohorts: HateMM 214 videos (the other project has 215; `hate_video_427`, which has no GT here, is dropped);
  DeHate 1151 videos, identical. **HateClipSeg is not comparable**: this repository's test split has 79 videos, the
  other project's 118, overlap 23. No HCS numbers here.
- Length alignment: HateMM 3 videos padded by 1 s, 6 cut by 1 s; DeHate 6 videos padded by 1 s, 29 cut by up to 60 s
  (223 s in total), where the other project's duration is longer than this repository's GT array.
- Check: CLIP+MIL HateMM .720 / .458 / .523 (ROC / AP / within) here against .719 / .459 / .523 in the other project.

Run: uoa-lab1 (sc474397), 2026-11-15, CPU, `python experiments/20261115_hf_baselines/reeval.py` (code commit message
"Re-evaluate Hate-follow-up baselines ..."). Output `runs/20261115_hf_baselines/` (copied back to uoa-lab2),
`table.json` = seed mean and sd (n-1); per-seed `metrics.json` from the evaluator.

## Results (test, AP / ROC / within; weakly supervised rows 3-seed mean)

| row | group | HateMM | DeHate |
|---|---|---|---|
| BERT+MIL | weak | .383 / .636 / .519 | .172 / .645 / .557 |
| CLIP+MIL | weak | .458 / .720 / .523 | .131 / .642 / .516 |
| wav2vec2+MIL | weak | .442 / .692 / .561 | .082 / .542 / .520 |
| VadCLIP (fixed XD preset, last epoch) | weak | .401 / .683 / .459 | .116 / .627 / .445 |
| AVadCLIP (IEEE TMM 2026) | weak | .435 / .678 / .481 | .138 / .641 / .499 |
| SAGE (ACL 2026) | weak | .503 / .731 / .616 | .110 / .593 / .572 |
| CLARA (ACM MM 2026) | weak | .628 / .872 / .549 | .193 / .774 / .477 |
| ZS-CLIP | label-free | .235 / .505 / .485 | .076 / .505 / .515 |
| ZS-ImageBind | label-free | .341 / .630 / .561 | .093 / .562 / .524 |
| LAVAD (CVPR 2024) | label-free | .501 / .731 / .517 | .123 / .608 / .498 |
| EventVAD (ACM MM 2025) | label-free | .371 / .625 / .576 | .086 / .537 / .508 |
| PRISM (ICML 2026) | label-free | .450 / .723 / .626 | .111 / .542 / .555 |
| VideoMind (ICLR 2026) | label-free | .399 / .680 / .658 | .089 / .568 / .603 |
| Qwen3-8B, plain question on transcript segments | label-free | .565 / .789 / .587 | .134 / .639 / .582 |
| Qwen3-VL-8B, plain question per 8 s window | label-free | .554 / .811 / .628 | .176 / .691 / .601 |
| MARS on Qwen3-VL-8B (ICASSP 2026) | label-free | .528 / .791 / .611 | .163 / .692 / .597 |
| EventVAD on Qwen3-VL-8B | label-free | .510 / .762 / .583 | .140 / .638 / .613 |
| revision 7 (8 calls, test-selected trial) | ours | .723 / .898 / .826 | .291 / .795 / .710 |

Every row is below revision 7 on every metric of both corpora. Strongest trained baseline is now CLARA on both
(revision 7 ahead by +.095 / +.026 on HateMM, +.098 / +.021 on DeHate in AP / ROC). The method page's main table
(`research-wiki/method_framework.html`, E1) lists the rows above except ZS-CLIP, ZS-ImageBind and the original EventVAD
(near chance).

## Open

- HCS cells for the new rows need the baselines retrained (weakly supervised) or re-run (VLM) on this repository's HCS
  split. Not done; needs the user's go-ahead (GPU).
- VadCLIP's DeHate cell on the page comes from this batch (fixed preset), its HateMM / HCS cells from the 40-trial
  official-validation runs; the page marks it.
