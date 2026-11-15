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

- LAVAD has no HateClipSeg cell (see the rerun section below).
- CMHKF has no DeHate cell (never run on DeHate).
- VadCLIP's DeHate cell on the page comes from this batch (fixed preset), its HateMM / HCS cells from the 40-trial
  official-validation runs; the page marks it.

## HateClipSeg reruns on this repository's split, and an InternVL3.5-8B naive row (2026-11-15 / 16)

What: baseline outputs only (4 fps `predictions.jsonl` in the other project's schema); nothing is evaluated here and
`reeval.py` / the table are not touched. Full record on the other side:
`~/Hate-follow-up/experiments/20261115_rh_hcs_split/README.md` (branch `follow-up`, commits de3b361, 54bdd9b, 338f7dd,
1dcbd9d, 2a732db).

- `export_hcs_split.py` (uoa-lab1) writes `runs/20261115_hf_baselines/hcs_split/split.json` from
  `load_fixed_cohort('hateclipseg')`: train 251 (219 hateful) / val 63 (55) / test 79, train + val labels only.
- In `~/Hate-follow-up` the baseline scripts read the split, the test cohort and the run root from environment
  variables (unset = their behaviour as before; no existing result or cache file changed) and stop after
  `predictions.jsonl`, because their evaluator knows only their 118-video cohort. Test durations = ffprobe format
  duration (equal to their manifest on the 23 shared test videos).
- Label check: 314 train / val labels of this repository against the other project's HateClipSeg labels: 219 against
  its train / val labels, 95 (its test videos) against "any positive frame" of its 4 fps GT. 0 disagreements.
- `collect_sources.py` (uoa-lab1) copies the rows of this repository's test cohorts into
  `runs/20261115_hf_baselines/sources/<method_key>/<Dataset>/<seed<k> | single>/predictions.jsonl` (one row per cohort
  video, curve finite and of length ceil(4 · duration); report `sources/collect_report.json`); copied to uoa-lab2 with
  `rsync -a`.

| method_key | corpora | files | settings |
|---|---|---|---|
| `BERTpMIL`, `CLIPpMIL`, `wav2vec2pMIL` | HateClipSeg | seed2025 / 234 / 3407 | their MIL recipe, epoch on val video AP |
| `AVadCLIP_align`, `AVadCLIP_mlp` | HateClipSeg | 3 seeds each | fixed XD preset, last epoch. Their file holds both branches per video; `reeval.py` keeps the last row per video, so the existing "AVadCLIP" HateMM / DeHate cells are the `score_mlp` branch (`AVadCLIP_mlp` here); `score_align` is the other project's headline |
| `SAGE` | HateClipSeg | 3 seeds | epoch on val macro-F1 (15 / 16 / 16) |
| `CLARA` | HateClipSeg | 3 seeds | rationale mode both |
| `Qwen3-VL-8B` | HateClipSeg | single | naive `vlm` arm |
| `MARS_Qwen3-VL-8B`, `EventVAD_Qwen3-VL-8B` | HateClipSeg | single | read-out `prob` |
| `PRISM`, `VideoMind` | HateClipSeg | single | `prism_av`, `videomind_av` |
| `InternVL3.5-8B` | HateMM (214), HateClipSeg (79), DeHate (1151) | single | the naive `vlm` arm with `OpenGVLab/InternVL3_5-8B-HF` (new arm `vlm_internvl` of `naive_prompt.py`): 8 s windows, 1 fps frames ≤ 8 + Whisper words, the plain question, fp32 Yes / No log-odds through HF transformers (the existing `vlm` rows are read the same way, not through vLLM); one 448 tile per frame |

Hosts and jobs (RTX 5090, one GPU job at a time): uoa-lab1 (sc474397) Slurm 2319 MIL + AVadCLIP, 2320 SAGE, 2343
VideoMind, 2344 EventVAD; uoa-lab3 (sc474398) Slurm 2333 CLARA, 2334 Qwen3-VL-8B naive + MARS + InternVL3.5 (three
corpora), 2337 / 2341 / 2342 PRISM. Failed: 2322 (CLARA on uoa-lab1, encoder weights missing there, 14 s), 2337 and
2341 (PRISM stopped on two undecodable webm files). GPU time 10.7 h in all.

Deviations:
- **LAVAD has no HateClipSeg file.** It needs the visual-only LAVAD chain on the 56 new test videos (about 3 to 4
  GPU-h) before the AV scoring; BLIP-2 and Llama-2-13b weights are no longer on any lab machine, its env exists only on
  uoa-lab2 (GPU held by other jobs), and four more scripts would need the cohort override. Stopped, not improvised.
- PRISM: `yt_hkNgzMgCA9E` and `yt_p_qvu46d1AU` (webm) fail in the full decord read; the other project's rule stops a
  corpus with more than 1 % constant-filled videos, so the two were re-encoded to H.264 (the script's own decode-only
  fallback, switched on by a new opt-in variable) and embedded normally. Window embeddings of the 23 test videos shared
  with the other project's cohort are its existing ones (H200); the other 56 were computed on uoa-lab3.
- EventVAD: event boundaries of the 23 shared test videos are the existing ones; 56 segmented on uoa-lab1 with the
  same released config; all 79 scored in this run.
- SAGE / CLARA: per-video input caches reused for videos whose role (train / val or test) is the same in both splits;
  CLARA rationales (video-level, exist for all videos) reused as is; CLARA's w/o-rationale check not rerun.
- InternVL3.5-8B HateMM / DeHate ran on the other project's cohorts (215 / 1151); `collect_sources.py` drops
  `hate_video_427`.
- Commits of this section carry the trailer of the model that made them (Claude Fable 5.2).

## Evaluation of the reruns (2026-11-16)

`reeval.py` now reads, for the (row, corpus) pairs in its `SRC` table, `runs/20261115_hf_baselines/sources/` instead of
`~/Hate-follow-up/runs/`: every HateClipSeg cell of the new rows and all three corpora of the new row
"InternVL3.5-8B". AVadCLIP's HateClipSeg cell is `AVadCLIP_mlp`, the branch of its HateMM / DeHate cells. HateClipSeg
cells of rows without a rerun are skipped. Run on uoa-lab1 (CPU), 2026-11-16:
`python experiments/20261115_hf_baselines/reeval.py --corpora hateclipseg --rows <12 rows>` and
`--corpora hatemm dehate --rows InternVL3.5-8B`; output copied back to uoa-lab2, numbers in `table.json`.
The Qwen3-8B transcript row stays in `table.json` but is removed from the method page (user, 2026-11-15).

| row | HateClipSeg (AP / ROC / within) |
|---|---|
| BERT+MIL | .550 / .514 / .490 |
| CLIP+MIL | .522 / .495 / .515 |
| wav2vec2+MIL | .481 / .432 / .476 |
| AVadCLIP (mlp) | .555 / .531 / .561 |
| SAGE | .553 / .545 / .506 |
| CLARA | .549 / .534 / .522 |
| Qwen3-VL-8B, plain question per 8 s window | .704 / .688 / .618 |
| InternVL3.5-8B, same | .703 / .677 / .617 |
| MARS on Qwen3-VL-8B | .690 / .677 / .621 |
| EventVAD on Qwen3-VL-8B | .646 / .649 / .597 |
| PRISM | .666 / .662 / .571 |
| VideoMind | .618 / .572 / .565 |
| revision 7, 8 calls, test-selected | .704 / .688 / .605 |
| revision 7, validation-selected | .681 / .657 / .596 |

InternVL3.5-8B on HateMM .580 / .809 / .647, DeHate .198 / .704 / .627.

Findings:
- Weakly supervised baselines are near chance on this repository's HateClipSeg split (ROC .43 to .55; train has 219
  hateful videos of 251). Revision 7 leads the strongest trained baseline by +.142 AP (Fed-WSVAD) / +.143 ROC (SAGE).
- **On HateClipSeg the plain per-window VLM question matches or beats revision 7**: Qwen3-VL-8B .7039 / .6881 / .6178
  against .7036 / .6878 / .6053 (test-selected trials); the validation-selected revision 7 is lower on all three.
  The plain question asks every 8 s window, 30.2 calls per video on average on HateClipSeg (HateMM 17.5, DeHate 12.5),
  revision 7 asks 8. On HateMM and DeHate revision 7 is ahead of every label-free row (+.143 / +.087 / +.168 and
  +.093 / +.091 / +.083 against the best row per column).

## Settings of the plain question (2026-11-16, user request)

Request: try other windows and settings for the two plain-question rows (Qwen3-VL-8B, InternVL3.5-8B), because on
HateClipSeg they match revision 7. Other project: `naive_prompt.py` gained `--window` (seconds, 0 = whole video),
`--no-text` (frames only) and `--question tax` (the HateClipSeg five-kind question: hateful, insulting, sexual,
violence, self-harm); defaults unchanged (follow-up 16104f5). Slurm 2356, uoa-lab3, task order HateClipSeg, HateMM,
DeHate; outputs `~/Hate-follow-up/runs/20261115_rh_hcs_split/lf/naive/<arm>__<tag>/`, copied to
`runs/20261115_hf_baselines/sources/<model>__<tag>/`, evaluated with `reeval.py --settings` on uoa-lab1.

HateClipSeg, test 79 videos (AP / ROC / within):

| setting | Qwen3-VL-8B | InternVL3.5-8B |
|---|---|---|
| 8 s window (main-table row) | .704 / .688 / .618 | .703 / .677 / .617 |
| 4 s window (at most 4 frames) | .687 / .670 / .597 | .687 / .660 / .599 |
| 16 s window (8 frames) | .708 / .689 / .623 | .702 / .671 / .612 |
| 32 s window (8 frames) | .695 / .668 / .581 | .678 / .648 / .564 |
| whole video, one call (8 frames) | .671 / .620 / .500 | .645 / .604 / .500 |
| 8 s, frames only | .679 / .647 / .594 | .667 / .634 / .590 |
| 8 s, five-kind question | **.748 / .726 / .657** | **.743 / .722 / .641** |

Revision 7 on HateClipSeg: .704 / .688 / .605 (test-selected), .681 / .657 / .596 (validation-selected). Its VLM is
Qwen2.5-VL-7B with 4 frames per node, and on HateClipSeg it already asks with the five-kind definition
(`experiments/20260925_query_paradigm/README.md`, extraction of 2026-09-29); the plain-question rows use Qwen3-VL-8B
with up to 8 frames and ask every window (30.2 calls per video at 8 s).

- Window length changes the result by at most .02 between 4 and 16 s; 16 s is the best window for Qwen3-VL-8B
  (+.004 AP over 8 s). One call per video drops to .671 / .620 with within .500 (one score per video).
- Removing the transcript costs .025 to .041.
- The five-kind question, which matches the HateClipSeg labels (a union of the five kinds), adds .044 / .038 / .039
  (Qwen3-VL-8B) and is .044 / .038 / .052 above revision 7.
- HateMM and DeHate runs of the same settings are in progress (job 2356).
