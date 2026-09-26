# DeHate external validation (2026-09-26)

Status: preparation in progress (media, transcripts, features). No results yet.

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
- Answers: shards 0/3, 1/3 and 2/3 on uoa-lab2, uoa-lab1 and uoa-lab3, started 2026-09-26 at 14:45.

## 5. Results

(to be filled)
