# DeHate external validation (2026-09-26)

Status (2026-09-27 02:10): complete. Every result is on uoa-lab2 under `runs/20260926_dehate_external/`.

## 1. Why this run

The user asked, 2026-09-26, to add DeHate. DeHate is local on uoa-lab2 at `~/data/DeHate`, with an official split
of train 4680 / val 668 / test 1341. Every video is at most 300 s long, 179 h in total. The run covers:

- the query-tree method, revision 3 (`experiments/20260925_query_paradigm`, README sections 9 and 10);
- the four strongest trained baselines of the fixed table: MACIL-SD, MultiHateLoc, DSANet, Fed-WSVAD with 3 clients.

It answers two questions.

1. How does the method do against the baselines on a third corpus?
2. Do the findings of `experiments/20260925_query_paradigm/README.md` sections 10.4 and 12 carry over? The
   findings are:
   - (a) the content backbone is needed on HateMM but not on HateClipSeg (no-backbone control);
   - (b) the tree beats one-level windows on ROC only, and scores fall when more than 8 questions are asked, with
     the tree only;
   - (c) short-node answers barely separate harmful from harmless seconds inside hateful videos, and the anchored
     answer table over-trusts the all-zero answer;
   - (d) the proposed fix is a per-video correction of the backbone scoring head, which spreads each VLM answer to
     seconds with similar backbone features. Its check is the offline upper bound with true spans, plus the
     version with the 8 real answers.

DeHate is external validation only (CLAUDE.md: new corpora never gate the method and never enter the main table).

## 2. Protocol (user decisions, 2026-09-26)

- **Search protocol:** each method keeps its own, as on HateMM and HateClipSeg.
  - Baselines: 5 Optuna trials at seed 234 (user, 2026-09-26; HateMM and HateClipSeg had 40), selected on
    validation video AP. Optuna's TPE sampler starts from 10 random trials, so the 5 configurations are random
    draws from each method's search space
    (`scripts/reproduction_baselines/tune_official_val.py`), then the selected configuration is retrained at seeds
    234 / 2025 / 3407 and scored on test once.
  - The retraining uses `experiments/20260926_dehate_external/confirm_baselines.py`. It is the same procedure as
    `confirm_official_val.py`, without the file digests that script records (CLAUDE.md, no hashing).
  - Query-tree method: 20-trial Optuna search per seed, seeds 234 / 2025 / 3407. The objective is test (pooled AP +
    pooled ROC) / 2, with a budget of 20 trials if the first trial takes at most 1 h, else 5
    (`experiments/20260925_query_paradigm/search.py`).
- **Gold:** frame protocol plus the DeHate amendment (`docs/duplex/FRAME_EVAL_PROTOCOL_DEHATE.md`).
  - 983 of the 2120 hateful videos have no usable span. They are almost all title- or description-only hate.
    Rule (b) excludes them from evaluation, and training keeps them with their video label.
  - Test cohort: 234 hateful + 917 non-hateful videos.
- **Inputs:** the same inputs as on the other corpora, extracted by the same scripts:
  - I3D 5-crop, VGGish, CLIP ViT-B/16, ImageNet ViT-B/16, BERT sentence rows, 1-fps frames;
  - Whisper large-v3 sentence chunks (`results/reproduction/asr/dehate_all`).
  - One deviation: the VLM node transcript uses the same sentence chunks. The word-level Whisper run
    (`src/utils/generate_segment_asr_HF.py`) runs out of memory on the 32 GB GPU for DeHate's 5-minute videos. On
    HateMM and HateClipSeg it had already fallen back to sentence chunks for 51% and 62% of videos.
- **One training video has no I3D features.** 8bAfN6vXoIZp (non-hateful) has 4 frames (0.14 s) of video over 134 s
  of audio, so the I3D extractor forms no 16-frame snippet. Every other DeHate video has at least 3 s of video.
  - The two I3D methods (MACIL-SD and the query-tree method) train without it. This is MACIL-SD's existing rule
    (`scripts/reproduction_baselines/macilsd/train.py`, `usable_ids`); for the query-tree method it is
    `src/hier_evidence_common.load_fixed_cohort`, which still raises on any other missing feature.
  - CLIP and ViT features exist for it (the last frame repeated), so DSANet, Fed-WSVAD and MultiHateLoc keep it.
  - It is a training video, so the evaluation cohort does not change.
- **Machines:** uoa-lab2 holds the raw videos and extracts media and features. VLM answers and searches are spread
  over uoa-lab1, uoa-lab2 and uoa-lab3. Only derived files (frames, wav, features, the label file) are copied; the
  raw videos stay on uoa-lab2.

## 3. How to run

1. `python scripts/dehate/prepare_dehate.py splits`, then `... media` (CPU): split files, flat symlink directory,
   wav, 1-fps frames.
2. `bash experiments/20260926_dehate_external/launch/prep_local.sh <stage>` for each stage:
   - `asr_chunks`;
   - `vggish`, `clip`, `vit`, `i3d`, `bert`;
   - `gt`.

   Logs go to `runs/20260926_dehate_external/prep/`.
3. `python experiments/20260925_query_paradigm/build_tree_manifest.py --corpus dehate`, then VLM answers with
   `extract_tree_answers.py --corpus dehate` (sharded over the three machines).
4. Searches and baselines: see section 4, filled in as the runs start.

## 4. Runs

Baselines (`launch/baseline.sh <method>`; logs in `runs/20260926_dehate_external/baselines/`):

| Method | Host | Started | Finished | Log |
|---|---|---|---|---|
| DSANet | uoa-lab1 | 2026-09-26 10:32 | 12:59 | `dsanet_sc474397.out` |
| Fed-WSVAD, 3 clients | uoa-lab3 | 2026-09-26 10:33 | 14:21 | `fed_wsvad_3client_sc474398.out` |
| MACIL-SD | uoa-lab2 | 2026-09-26 11:50 | 14:14 | `macilsd_sc474399.out` |
| MultiHateLoc | uoa-lab2 | 2026-09-26 15:58 | 16:25 | `multihateloc_sc474399.out` |

Fed-WSVAD's first launch failed: all 10 attempts stopped within 12 s with `ModuleNotFoundError: No module named
'ftfy'`, because the HateVideo environment on uoa-lab3 lacked the package.
- ftfy 6.3.1 and wcwidth 0.8.2 were installed to match uoa-lab2 (`runs/_setup_uoa-lab3/pip_ftfy.log`).
- The failed study was deleted, so the relaunch starts from the same sampler seed, 234.
- The first log is kept as `fed_wsvad_3client_sc474398_ftfy_missing.out`.

Transcripts (Whisper large-v3 sentence chunks, `launch/asr_shard.sh`, logs in `runs/20260926_dehate_external/prep/`):
- The duration-sorted video list was split into 3 shards; shard 0 was split again so that idle GPUs could take parts
  of it.
- Parts and hosts:
  - uoa-lab2: 448 videos in `shard0of3`, 891 in `shard0of3.sub0of2` and 446 in `shard0of3.sub1of4`.
  - uoa-lab1: 2230 videos in `shard1of3`.
  - uoa-lab3: 2229 videos in `shard2of3` and 445 in `shard0of3.sub3of4`.
- `prepare_dehate.py merge_asr` merged them on 2026-09-26 at 14:40: 6689 videos, each exactly once, 0 error records.
- A first attempt at `shard0of3.sub1of2` on uoa-lab1 ran out of GPU memory for all 891 videos while shard 1 was
  running on the same GPU. Its output was deleted; its log is kept as `asr_shard0of3_sub1of2_sc474397_oom_attempt.log`.

Features: CLIP, ViT, VGGish and BERT have 6689/6689 videos; I3D has 6688 (section 2).

Query-tree nodes and VLM answers (`data/vlm_tree/DeHate/PROVENANCE.md`):
- Manifest: 6688 videos and 268,990 queryable nodes, 4.2 times HateMM's 63,963.
- Answers: shards 0/3, 1/3 and 2/3 on uoa-lab2, uoa-lab1 and uoa-lab3, from 14:45 to 15:58 on 2026-09-26.
  - There are 268,990 answers and 0 are unparsed.
  - A duplicate launch of shard 0/3 on uoa-lab2 was stopped at 14:43, before it wrote any answer.

Query-tree searches (`launch/run_search.sh dehate <seed> runs/20260926_dehate_external/qtl`), started 2026-09-26 at
16:28:
- seed 234 on uoa-lab2, seed 2025 on uoa-lab1, seed 3407 on uoa-lab3;
- logs: `runs/20260926_dehate_external/qtl/launch_dehate_seed<seed>_<host>.out`;
- MultiHateLoc (`launch/baseline.sh multihateloc`) runs next to seed 234 on uoa-lab2 from 15:58.
  - The first run stopped at 16:16 with 4/5 completed trials.
  - `tune_official_val.py` allows max(2 x trials, trials + 5) attempts, which is 10 at 5 trials (80 at the 40 trials
    used on HateMM and HateClipSeg).
  - Six of the 10 random draws had batch size 64. The script skips that setting before training because it does
    not fit in GPU memory, the same guard as on the other corpora.
  - The run was resumed as the script is designed to be: the Optuna study keeps the 4 completed trials, and the
    sampler is re-seeded from the attempt count.
  - The first log is kept as `multihateloc_sc474399_first_10_attempts.out`.
- The first search launch (16:00) failed at the end of every first trial, after training and evaluation had finished.
  - The cause was `KeyError: 'dehate'` in the silent-group comparison of `train.py`, which reads the old K30
    verdicts; DeHate has none.
  - `train.py` now skips that comparison when a corpus has no K30 verdicts.
  - The seed directories were deleted and the searches restarted at 16:28, so each seed starts from a new study.
  - The logs of the failed launch are in `runs/20260926_dehate_external/qtl/first_attempt_keyerror/`.

## 5. Results

Numbers are test means over seeds 234 / 2025 / 3407 on the 1-fps frame grid. Each triple is pooled AP / pooled ROC /
within-video ROC. The DeHate frame base rate is .076 (8,386 hateful of 110,839 test seconds). Sources:
- baselines: `runs/20260926_dehate_external/baselines/final/<method>/dehate/seed_<s>/frame_eval.json`;
- controls: `runs/20260926_dehate_external/diag/controls_summary.json`, and
  `runs/20260925_query_paradigm_diag/controls_summary.json` for HateMM and HateClipSeg;
- answer rates: `runs/20260926_dehate_external/answer_rates/dehate.json`;
- propagation check: `runs/20260926_dehate_external/propagation_check/{abl_full,best_trials}/dehate_propagation.json`;
- searched query tree: `runs/20260926_dehate_external/summary.json` (from `summarize.py`, which reads each seed's
  `study_summary.json` and best-trial `summary.json`).

### 5.1 Baselines and the query tree with default hyperparameters

| Method | AP | ROC | within |
|---|---|---|---|
| Fed-WSVAD, 3 clients | .174 ± .017 | .701 ± .010 | .508 |
| MultiHateLoc | .129 | .611 | .546 |
| DSANet | .120 | .632 | .486 |
| MACIL-SD | .088 | .562 | .530 |
| **Query tree, 20-trial search, best trial per seed, 8 calls** | **.215 ± .005** | **.733 ± .009** | **.647** |
| Query tree, 20-trial search, trial selected on validation | .201 ± .005 | .731 ± .010 | .646 |
| Query tree, default hyperparameters (`abl_full`), 8 calls | .198 | .730 | .601 |
| Query tree, default hyperparameters, 0 calls (backbone only) | .179 | .691 | .593 |

The searched method is above the strongest baseline, Fed-WSVAD:
- best trial: AP +.041, ROC +.033, within +.139;
- trial selected on validation: AP +.026, ROC +.031.

The best trials are 11, 13 and 11 (seeds 234, 2025, 3407); the validation-selected trials are 18, 17 and 18.

### 5.2 Do the query-paradigm findings carry over?

The control arms use default hyperparameters and 8 calls per video.

| Arm | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| full | .594 / .847 / .649 | .661 / .672 / .536 | .198 / .730 / .601 |
| no backbone | .543 / .822 / .679 | .658 / .634 / .571 | .142 / .700 / .633 |
| one level, 4-8 s | .563 / .803 / .598 | .665 / .659 / .550 | .172 / .696 / .574 |
| one level, 8-16 s | .587 / .827 / .618 | .657 / .649 / .554 | .191 / .734 / .562 |
| one level, 16-32 s | .590 / .826 / .661 | .662 / .654 / .557 | .191 / .732 / .586 |

**(a) Backbone.** Full minus no-backbone:

| Corpus | AP | ROC | within |
|---|---|---|---|
| HateMM | +.051 | +.026 | -.030 |
| HateClipSeg | +.003 | +.038 | -.035 |
| DeHate | +.056 | +.031 | -.032 |

- DeHate behaves like HateMM: the backbone raises both pooled metrics.
- On all three corpora the backbone lowers within-video ROC.

**(b) Tree versus one-level windows.**
- On HateMM and HateClipSeg the tree beats the best single level on ROC only: +.020 and +.013.
- On DeHate it does not: ROC -.002 to -.003, and AP +.007, below the .01 threshold. It beats every single level on
  within ROC instead: +.015 against 16-32 s, +.039 against 8-16 s.
- DeHate videos are short, so a single level runs out of nodes: 8-16 s windows used 5.3 calls on average and
  16-32 s windows 4.0, against 7.3 for the tree.

**More calls.** Full arm at 8, 16 and 32 calls:

| Corpus | AP | ROC |
|---|---|---|
| HateMM | .594, .576, .566 | .847, .839, .827 |
| HateClipSeg | .661, .668, .664 | .672, .674, .665 |
| DeHate | .198, .196, .196 | .730, .725, .723 |

- DeHate falls in the same direction, but by less than .01.
- The one-level arms on DeHate do not fall.
- Within-video ROC rises with more calls on DeHate: .601, .609, .612.

**(c) Short-node answers.** Nodes of 4-8 s on test:

| Measure | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| All-zero answers on nodes that contain harm | .41 | .50 | .61 |
| Separation inside hateful videos, logit(yes \| harm) - logit(yes \| no harm) | .37 | .68 | .75 |

- The answer table's harmful row comes from the roots of positive training videos. On DeHate, those roots give the
  all-zero answer .29 of the time (HateMM .04-.10).
- So the table still counts an all-zero answer as stronger evidence against harm than it is. The gap is smaller on
  DeHate than on HateMM, because many DeHate roots are themselves short.

**(d) Proposed fix: propagating an answer through backbone features.** The measure is within-video ROC of the
remaining seconds, averaged over the three default models.

| Pairs | Corpus | prior | time | feat | feat_c | raw |
|---|---|---|---|---|---|---|
| oracle | HateMM (336 pairs) | .592 | .692 | .643 | .655 | .712 |
| oracle | HateClipSeg (335) | .513 | .653 | .602 | .602 | .632 |
| oracle | DeHate (773 pairs, 166 videos) | .544 | .804 | .749 | .761 | .802 |
| real VLM answers | HateMM (281) | .581 | .549 | .549 | .539 | .551 |
| real VLM answers | HateClipSeg (314) | .514 | .531 | .523 | .534 | .546 |
| real VLM answers | DeHate (878 pairs, 191 videos) | .553 | .514 | .528 | .526 | .532 |

- As on HateMM and HateClipSeg, backbone features spread a true answer worse than distance in time does (.749-.761
  against .804). The fix needs features to beat time, so its precondition does not hold.
- With the real answers every method is near .5. The backbone's own ranking (.553) is the best on DeHate too.
- On the three best search trials (the model choice used for HateMM and HateClipSeg) the numbers are the same:
  - oracle pairs: prior .538, time .804, feat .742, feat_c .755, raw .802;
  - real-answer pairs: prior .553, time .514, feat .521, feat_c .524, raw .532.

### 5.3 Searched query tree: calls per video

Best trial of each seed, three-seed mean, from `runs/20260926_dehate_external/summary.json`:

| Calls per video | AP | ROC | within |
|---|---|---|---|
| 0 (backbone only) | .200 | .710 | .647 |
| 2 | .198 | .737 | .638 |
| 4 | .213 | .738 | .647 |
| 8 | .215 | .733 | .647 |
| 16 | .215 | .729 | .645 |
| 32 (21.96 on average) | .207 | .719 | .649 |

- Asking helps the pooled metrics: from 0 to 8 calls, AP rises by .015 and ROC by .023.
- It does not help within-video ROC on DeHate: .647 at both 0 and 8 calls.
- Beyond 8 calls the scores fall, as on HateMM: at 32 calls AP is .009 lower and ROC .014 lower.
  - With default hyperparameters the fall stays under .01 (section 5.2).
