## P. Method as run — CLAP (CVPR 2024)

CLAP is the label-free end of Wave 2: it is given a pool of videos with **no labels of any kind** —
not even a "these are normal" pool — and has to invent its own supervision. It does that in two
stages. A *coarse* stage clusters each participant's videos into a normal and an abnormal group
from two hand-designed video-level statistics. A *fine* stage fits a Gaussian to the snippet-level
squared feature norms of the normal group, mixes the participants' Gaussians into one density, and
marks, inside every video the coarse stage called abnormal, the one window of fixed relative length
(0.2 of the video) across which that density changes fastest. Those marks are the pseudo-labels. A
small MLP is then trained on them with BCE, federated across participants with FedAvg.

**Code path.** `third_party/CLAP` @ `3dcaadc1` (AnasEmad11/CLAP, CVPR 2024). Campaign patch
`scripts/repro_campaign/patches/CLAP.patch`; data-adaptation layer and driver
`scripts/repro_campaign/run_clap.py`; corpus chain `scripts/repro_campaign/run_clap_chain.sh`; logs
`logging/runs/repro_clap/run.log`. The entry point executed is the repo's own
`src/server/fedavg.py` with the repo's own `train.sh` argument set. The clustering, the Gaussian
mixture, the pseudo-label window search (`data/utils/datasets.py::C2FPL_client`, `::gmm_PL`), the
scorer (`src/config/models.py::C2FPL_XD`), the local BCE fit (`src/client/fedavg.py::fit_ucf`) and
the FedAvg aggregation (`src/server/fedavg.py::aggregate`) all run unmodified.

`MODEL_ASSETS_STATUS` row 10 / §3.5 marked CLAP **BLOCKED** because the repo is a federated-learning
codebase that reads UCF-Crime concatenated features plus scene-partition `.pkl` files and nothing
else. Supplying exactly that is what the adaptation layer below does; the block is lifted.
`visdom` is plot-only, reachable only under `--visible 1`, which is off, and was never installed.
`wandb` *is* called unconditionally by both server and client, so an offline stub
(`scripts/repro_campaign/shim/clap/wandb.py`) is put ahead of the real package on `PYTHONPATH`.

### P.1 What the adaptation layer does

CLAP's input contract is four objects per dataset, and `run_clap.py --stage build` writes all four
into `third_party/CLAP/data/hate_<DS>_<config>/`:

1. **`concat_train.npy`** — every snippet of every training-pool video, concatenated in video order,
   shape `(N, ncrops, D)`. UCF-Crime supplies 10-crop I3D, `(N, 10, 2048)`.
2. **`concat_score.npy`** — the same for the videos to be scored.
3. **`partition.pkl` / `video_num_partition.pkl`** — per client, the list of *per-video* snippet
   index arrays and the matching global video numbers. The coarse stage reads these.
4. **`partition_chain.pkl`** — per client, the same indices flattened, which is what the
   `DataLoader` subsets are built from.

**Snippet convention.** CLAP's UCF-Crime features are the "complete" (variable-length, not
32-segment) I3D stream: one snippet per 16 decoded frames, which at UCF-Crime's ~30 fps is 0.53 s,
and the repo's own evaluator upsamples one snippet score to 16 frames. The nearest thing our frozen
4 fps cache supports is **one snippet = 2 consecutive 4 fps frames = 0.50 s**, mean-pooled over the
1024-d CLIP-L/336 vectors, trailing remainder frames dropped as a fixed-stride clip extractor drops
them. **CLAP's native output rate is therefore 2 samples per second**, recorded per video in the
npz and broadcast piecewise-constant onto the 4 fps grid by the shared evaluator (freeze §1). No
feature was re-extracted.

**Client partition.** `train.sh` uses `scene_partition_11_V3.pkl`: 11 participants, partitioned by
surveillance scene. Our corpora carry no scene metadata, so the rule — written down here and
nowhere else — is **`client = crc32(video_id) mod 11`**: deterministic, reproducible from the video
id alone, independent of every label. It gives 11 IID participants where CLAP's own split is 11
non-IID ones (deviation D-b). Resulting client sizes: HateMM 58–82 videos, MHC-EN 40–61,
MHC-ZH 41–64, HateClipSeg 16–31.

**Scorer.** Our features are 1024-d, so the head used is the repo's **`c2fpl_XD`**
(`Linear(1024, 512) → 512 → 32 → 1`, dropout 0.6, sigmoid, mean over the crop axis), not
`c2fpl_ucf`, which is the same architecture with a 2048-d first layer for I3D. `c2fpl_XD` is the
repo's own head for its own 1024-d dataset and is used verbatim, including the fact that the
released file leaves `self.apply(weight_init)` commented out on this head, so it carries PyTorch's
default Linear init rather than the Xavier init `c2fpl_ucf` gets.

**Pool sizes actually built** (train pool = train split only; score set = every video with a
feature file, so val and test curves come out of the same forward pass and the split is chosen by
the evaluator):

| dataset | train videos | train snippets | scored videos | scored snippets |
|---|---|---|---|---|
| HateMM | 743 | 225,549 | 1,081 | 311,832 |
| MHC-EN | 548 | 38,386 | 792 | 55,329 |
| MHC-ZH | 579 | 36,361 | 814 | 50,937 |
| HateClipSeg | 237 | 111,721 | 395 | 187,606 |

**Supervision column.** Freeze §9 pre-classified CLAP as `one-class`. It is not: the configuration
run here is the paper's own `--train_mode US`, in which no pool of known-normal video exists and the
normal set is a *product* of the coarse clustering. The row is reported as **`unlabelled`**. No
test-split video and no label of any split entered training, clustering, or knob selection.

### P.2 Deviations forced by the transplant

- **D-a — features.** Single-view CLIP-L/336 at 0.5 s snippets replaces 10-crop I3D. The crop axis
  is kept as a singleton (`ncrops = 1`), which makes the model's `x.mean(dim=1)` and
  `get_matrix`'s mean over crops no-ops. Everything CLAP computes from the features — squared L2
  norm per snippet, its first difference, its per-dimension variance over time — is defined on any
  feature stream and is computed here exactly as released.
- **D-b — partition.** IID `crc32 mod 11` instead of a scene partition. CLAP's paper argues the
  scene split is the *hard* case for collaboration, so this deviation is not adverse to the method.
- **D-c — the `eta_clustering=1` code path does not run, and is not ours to fix.** `train.sh` sets
  `--eta_clustering 0`, and that is what was run. The alternative path, `C2FPL_client_eta`,
  initialises `normal_set = {}` / `abnormal_set = {}`, fills `set_n` / `set_a` instead, and has the
  four lines that would choose between them commented out — so under `--load 0` it always proceeds
  with two empty dictionaries and fits a Gaussian to an empty array. Recorded as a defect of the
  release, not worked around.
- **D-d — model selection.** The repo's `evaluate_ucf` loads UCF-Crime frame labels from
  `labels/gt-ucf-RTFM.npy`, and `test_ucf` reports the `max` AUC over client models before and
  after local training — i.e. it selects on test labels. Freeze §10 red line 1 forbids that, so the
  in-loop readout is disabled and the row reports the **single aggregated global model** at the
  frozen round, scored once, through the campaign's own evaluator.
- **D-e — covariance diagonal computed directly.** `covariance_mat` materialises the full `D × D`
  covariance of the time series and the caller reads only `np.diagonal` of it — ~1.2 × 10¹⁰ flops
  per video at `D = 1024` and 11.5k snippets. It now returns `np.diag(np.var(X, axis=0, ddof=1))`,
  which is *exactly* the diagonal `np.cov` would have produced. Numerically identical on every
  entry the algorithm reads.
- **D-f — five videos have no row.** Two HateMM containers have no video stream at all (freeze D2:
  `hate_video_147`, `hate_video_292`) so no feature exists. One HateMM and one MHC-EN training
  video are shorter than 4 snippets (2 s) — `hate_video_272`, `XScP1AiMkNM` — and the coarse stage
  needs `np.diff` of a length-≥2 norm curve plus a non-degenerate window, so they are excluded from
  the *training pool only*; both are still scored. No zero curve is fabricated for anything. None
  of the five is in any test split, so **every test row has coverage 1.000 and zero missing videos.**
- **D-g — plotting, logging, and where it ran.** `visdom` not installed (plot-only, unreachable at
  `--visible 0`); `wandb` replaced by an offline stub. `--global_testset 1` (rather than
  `train.sh`'s 0) so the ordered score-set `Subset` exists for the score dump; with the in-loop AUC
  readout disabled this changes nothing that is trained. The run is on the shared RTX 5090 under a
  hard `torch.cuda.set_per_process_memory_fraction(0.10)` cap added by the patch — CLAP's scorer is
  a 4-layer MLP over cached feature vectors and peaks at a few hundred MB. An earlier CPU-only
  start was abandoned because HateMM projected to ~12 h; nothing from it was evaluated and no test
  call was made, so no result is carried across the CPU/GPU boundary (CUDA and CPU dropout draw
  from different generators, so the two are not interchangeable and are not mixed).
- **D-h — the training matrix is loaded into RAM, not memory-mapped.** Purely an I/O choice
  (≤ 0.9 GB for the largest corpus); no value changes.

### P.3 Frozen knobs

Every hyper-parameter is the published `train.sh` value: `--train_mode US`, `--local_epoch 10`,
`-bs 128`, SGD `lr = 1e-2`, momentum 0, weight decay 0, `MultiStepLR([5, 10], γ = 0.1)`,
`--join_ratio 1`, `--gmm_pl 1`, `--eta_clustering 0`, 2-component
`GaussianMixture(max_iter=150, random_state=0)`, pseudo-label window fraction 0.2.

The one free knob is the **number of FedAvg global rounds**. The selection rule was written to
`idea-stage/repro_clap/knob_rule.json` *before any number existed*: candidates {1, 2, 5, 10}, chosen
on the **val** split by `frame_ROC_AUC` (the metric CLAP's own paper reports) of the `fedavg11`
configuration at seed 20250819, ties to the smaller count, one value per dataset, then reused for
the `central` configuration and all three seeds.

| dataset | val ROC r=1 | r=2 | r=5 | r=10 | **frozen** |
|---|---|---|---|---|---|
| HateMM | **0.7408** | 0.7328 | 0.7122 | 0.7030 | **1** |
| MHC-EN | 0.4810 | 0.5328 | **0.5709** | 0.5546 | **5** |
| MHC-ZH | 0.2363 | 0.2401 | 0.2871 | **0.3200** | **10** |
| HateClipSeg | **0.4591** | 0.4507 | 0.4476 | 0.4474 | **1** |

Frozen in `idea-stage/repro_clap/run_record.json` at 2026-08-21T23:48:15+12:00, repo commit
`4fcb487`, **before** the single test call. Note that the published `train.sh` value is
`--global_epoch 1`, which the val rule independently selects on two of the four datasets.

Model init, dropout and client sampling are stochastic, so freeze §6 applies: **3 seeds**
(20250819 / 20250820 / 20250821), mean ± sd. The `normality` ablation has no trained component and
`GaussianMixture(random_state=0)` is fixed, so it is deterministic and carries `seeds = 1`.

### P.4 Variants reported

| config | what it is |
|---|---|
| `collaborative (11 clients, FedAvg)` | the faithful port of `train.sh`: 11 participants, FedAvg |
| `centralized (1 client)` | the paper's own "Centralized" configuration — one participant holding the whole pool |
| `normal-Gaussian score, no MLP (ablation)` | CLAP's aggregated normal-density evaluated per snippet and negated, with **no MLP trained at all**. CLAP uses this density only to *locate a window* inside a video the coarse stage already called abnormal; reading it directly as a score separates how much of the localisation lives in the density and how much in the trained scorer. Ours, not the paper's. |

`F1@tIoU` reads `n/a` for every CLAP row: the method emits a score curve, and freeze §2 forbids
inventing a threshold to turn it into intervals. `transplant` is `n/a`: freeze §7 lists no
published CLAP number on any of our datasets.

Reproduce: `python scripts/repro_campaign/run_clap.py --stage build`;
`bash scripts/repro_campaign/run_clap_chain.sh`;
`python scripts/repro_campaign/clap_select_knob.py` (val only);
`python scripts/repro_campaign/run_clap.py --stage curves`;
`python scripts/repro_campaign/eval_frame.py --method curves --curve-dir idea-stage/repro_clap/curves --variants fedavg11_s0,fedavg11_s1,fedavg11_s2,central_s0,central_s1,central_s2,normality --method-name "CLAP" --wave 2 --supervision unlabelled --split test`.

### P.5 Headline rows — test split

| method | wave | dataset | split | supervision | variant | config | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOLD_BROADCAST | — | HateMM | test | control | control | n/a | video | 0.8857 | 0.5829 | n/a | n/a | n/a | 1.0000 | 116975 | 0.2421 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateMM | test | control | control | n/a | 4 fps | 0.5003 ± 0.0019 | 0.2423 ± 0.0013 | n/a | n/a | n/a | 0.0000 | 116975 | 0.2421 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| GOLD_BROADCAST | — | MHC-EN | test | control | control | n/a | video | 0.9427 | 0.7664 | n/a | n/a | n/a | 1.0000 | 22337 | 0.2734 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-EN | test | control | control | n/a | 4 fps | 0.5004 ± 0.0034 | 0.2737 ± 0.0026 | n/a | n/a | n/a | 0.0000 | 22337 | 0.2734 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| GOLD_BROADCAST | — | MHC-ZH | test | control | control | n/a | video | 0.9842 | 0.9191 | n/a | n/a | n/a | 1.0000 | 18199 | 0.2648 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-ZH | test | control | control | n/a | 4 fps | 0.4985 ± 0.0052 | 0.2646 ± 0.0038 | n/a | n/a | n/a | 0.0000 | 18199 | 0.2648 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| GOLD_BROADCAST | — | HateClipSeg | test | control | control | n/a | video | 0.6260 | 0.5437 | n/a | n/a | n/a | 1.0000 | 114097 | 0.4712 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateClipSeg | test | control | control | n/a | 4 fps | 0.5009 ± 0.0021 | 0.4721 ± 0.0016 | n/a | n/a | n/a | 0.0000 | 114097 | 0.4712 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| CLAP | 2 | HateMM | test | unlabelled | base | collaborative (11 clients, FedAvg) | 2 fps | 0.6015 ± 0.0147 | 0.3570 ± 0.0050 | n/a | n/a | n/a | 0.3359 ± 0.0146 | 116832 | 0.2424 | 3 | n/a | §4 | idea-stage/repro_clap/ | E=1 global rounds (val-selected); coverage 1.000, 0 missing |
| CLAP | 2 | HateMM | test | unlabelled | base | centralized (1 client) | 2 fps | 0.6008 ± 0.0030 | 0.3369 ± 0.0081 | n/a | n/a | n/a | 0.2772 ± 0.0236 | 116832 | 0.2424 | 3 | n/a | §4 | idea-stage/repro_clap/ | E=1 global rounds (val-selected) |
| CLAP | 2 | HateMM | test | unlabelled | base | normal-Gaussian score, no MLP (ablation) | 2 fps | 0.6210 | 0.3137 | n/a | n/a | n/a | 0.2093 | 116832 | 0.2424 | 1 | n/a | §4 | idea-stage/repro_clap/ | no training; CLAP's coarse-to-fine normal model only |
| CLAP | 2 | MHC-EN | test | unlabelled | base | collaborative (11 clients, FedAvg) | 2 fps | 0.4784 ± 0.0134 | 0.2599 ± 0.0163 | n/a | n/a | n/a | -0.0282 ± 0.0330 | 22298 | 0.2738 | 3 | n/a | §4 | idea-stage/repro_clap/ | E=5 global rounds (val-selected) |
| CLAP | 2 | MHC-EN | test | unlabelled | base | centralized (1 client) | 2 fps | 0.5746 ± 0.0023 | 0.3174 ± 0.0025 | n/a | n/a | n/a | 0.0881 ± 0.0049 | 22298 | 0.2738 | 3 | n/a | §4 | idea-stage/repro_clap/ | E=5 global rounds (val-selected) |
| CLAP | 2 | MHC-EN | test | unlabelled | base | normal-Gaussian score, no MLP (ablation) | 2 fps | 0.4682 | 0.2511 | n/a | n/a | n/a | -0.0461 | 22298 | 0.2738 | 1 | n/a | §4 | idea-stage/repro_clap/ | no training; CLAP's coarse-to-fine normal model only |
| CLAP | 2 | MHC-ZH | test | unlabelled | base | collaborative (11 clients, FedAvg) | 2 fps | 0.3213 ± 0.0103 | 0.1872 ± 0.0027 | n/a | n/a | n/a | -0.1188 ± 0.0040 | 18136 | 0.2648 | 3 | n/a | §4 | idea-stage/repro_clap/ | E=10 global rounds (val-selected); **below the random floor** |
| CLAP | 2 | MHC-ZH | test | unlabelled | base | centralized (1 client) | 2 fps | 0.4421 ± 0.0134 | 0.2291 ± 0.0065 | n/a | n/a | n/a | -0.0546 ± 0.0099 | 18136 | 0.2648 | 3 | n/a | §4 | idea-stage/repro_clap/ | E=10 global rounds (val-selected); below the random floor |
| CLAP | 2 | MHC-ZH | test | unlabelled | base | normal-Gaussian score, no MLP (ablation) | 2 fps | 0.5383 | 0.2958 | n/a | n/a | n/a | 0.0474 | 18136 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_clap/ | no training; CLAP's coarse-to-fine normal model only |
| CLAP | 2 | HateClipSeg | test | unlabelled | base | collaborative (11 clients, FedAvg) | 2 fps | 0.4693 ± 0.0014 | 0.4506 ± 0.0020 | n/a | n/a | n/a | -0.2796 ± 0.0277 | 113987 | 0.4709 | 3 | n/a | §4 | idea-stage/repro_clap/ | E=1 global rounds (val-selected); AP_norm denominator 0.072, unreliable |
| CLAP | 2 | HateClipSeg | test | unlabelled | base | centralized (1 client) | 2 fps | 0.4379 ± 0.0007 | 0.4251 ± 0.0008 | n/a | n/a | n/a | -0.6332 ± 0.0118 | 113987 | 0.4709 | 3 | n/a | §4 | idea-stage/repro_clap/ | E=1 global rounds (val-selected); AP_norm denominator 0.072, unreliable |
| CLAP | 2 | HateClipSeg | test | unlabelled | base | normal-Gaussian score, no MLP (ablation) | 2 fps | 0.4915 | 0.4587 | n/a | n/a | n/a | -0.1682 | 113987 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_clap/ | no training; CLAP's coarse-to-fine normal model only; AP_norm unreliable |

`AP_norm` denominators (broadcast − base on the evaluated pool): HateMM 0.341, MHC-EN 0.494,
MHC-ZH 0.654, **HateClipSeg 0.072**. The HateClipSeg `AP_norm` column is carried for completeness
and is flagged `AP_norm_reliable = false` by the evaluator; a 0.01 wobble in AP moves it by 0.14.

### P.6 Stratified sub-tables — single-span vs multi-span

Same convention as the rest of the campaign: zero-span videos supply negative frames and appear in
both strata. MHC-ZH's test split contains no multi-span video with any positive frame, so that
stratum is single-class and its metrics are undefined for every method, CLAP included.

| dataset | config | stratum | frame_ROC_AUC | frame_PR_AUC | AP_norm | n_frames | base_rate |
|---|---|---|---|---|---|---|---|
| HateMM | collaborative | single_span | 0.6425 ± 0.0163 | 0.3766 ± 0.0110 | 0.3185 ± 0.0199 | 93183 | 0.2010 |
| HateMM | centralized | single_span | 0.6134 ± 0.0023 | 0.3190 ± 0.0087 | 0.2141 ± 0.0158 | 93183 | 0.2010 |
| HateMM | normality (ablation) | single_span | 0.6181 | 0.2685 | 0.1225 | 93183 | 0.2010 |
| HateMM | collaborative | multi_span | 0.5687 ± 0.0256 | 0.1755 ± 0.0096 | 0.2363 ± 0.0319 | 91940 | 0.1043 |
| HateMM | centralized | multi_span | 0.6093 ± 0.0098 | 0.1722 ± 0.0103 | 0.2255 ± 0.0343 | 91940 | 0.1043 |
| HateMM | normality (ablation) | multi_span | 0.6613 | 0.1632 | 0.1956 | 91940 | 0.1043 |
| MHC-EN | collaborative | single_span | 0.4906 ± 0.0100 | 0.2555 ± 0.0156 | -0.0150 ± 0.0302 | 21630 | 0.2632 |
| MHC-EN | centralized | single_span | 0.5894 ± 0.0025 | 0.3147 ± 0.0017 | 0.0993 ± 0.0032 | 21630 | 0.2632 |
| MHC-EN | normality (ablation) | single_span | 0.4630 | 0.2396 | -0.0456 | 21630 | 0.2632 |
| MHC-EN | collaborative | multi_span | 0.2735 ± 0.0846 | 0.0176 ± 0.0019 | -0.0167 ± 0.0032 | 15013 | 0.0274 |
| MHC-EN | centralized | multi_span | 0.4010 ± 0.0329 | 0.0216 ± 0.0016 | -0.0099 ± 0.0027 | 15013 | 0.0274 |
| MHC-EN | normality (ablation) | multi_span | 0.5743 | 0.0299 | 0.0041 | 15013 | 0.0274 |
| MHC-ZH | collaborative | single_span | 0.3213 ± 0.0103 | 0.1872 ± 0.0027 | -0.1188 ± 0.0040 | 18136 | 0.2648 |
| MHC-ZH | centralized | single_span | 0.4421 ± 0.0134 | 0.2291 ± 0.0065 | -0.0546 ± 0.0099 | 18136 | 0.2648 |
| MHC-ZH | normality (ablation) | single_span | 0.5383 | 0.2958 | 0.0474 | 18136 | 0.2648 |
| MHC-ZH | any | multi_span | n/a | n/a | n/a | 12909 | 0.0000 |

### P.7 What the coarse stage actually produced

The pseudo-labels are the whole method, so they are reported directly. All of this is measured on
the **train** split only and is descriptive; nothing here selected anything.

| dataset | config | normal videos | abnormal videos | pseudo-positive snippet rate | P(gold video = 1 \| abnormal cluster) | P(gold = 1 \| normal cluster) | train base rate |
|---|---|---|---|---|---|---|---|
| HateMM | 11 clients | 466 | 277 | 0.067 | 0.516 | 0.330 | 0.400 |
| HateMM | centralized | 594 | 149 | 0.048 | **0.772** | 0.306 | 0.400 |
| MHC-EN | 11 clients | 365 | 183 | 0.055 | 0.284 | 0.301 | 0.296 |
| MHC-EN | centralized | 354 | 194 | 0.048 | 0.294 | 0.297 | 0.296 |
| MHC-ZH | 11 clients | 359 | 220 | 0.068 | 0.336 | 0.329 | 0.332 |
| MHC-ZH | centralized | 392 | 187 | 0.053 | 0.348 | 0.324 | 0.332 |
| HateClipSeg | 11 clients | 166 | 71 | 0.061 | 0.901 | 0.861 | 0.873 |
| HateClipSeg | centralized | 170 | 67 | 0.059 | 0.881 | 0.871 | 0.873 |

Two facts fall straight out. First, the coarse video-level clustering **separates on HateMM and on
nothing else**: centralized, a video the clusterer calls abnormal is hateful 77.2% of the time
against a 40.0% base rate, while on MHC-EN (0.294 vs 0.296), MHC-ZH (0.348 vs 0.332) and
HateClipSeg (0.881 vs 0.873) the abnormal cluster is indistinguishable from a random draw. Second,
the fine stage marks **4.8–6.8% of snippets positive** on every dataset, against true frame base
rates of 24–47%; the 0.2-of-the-video window applied to a minority of videos cannot reach the
amount of positive time these corpora actually contain, whatever it points at.

### P.8 What the numbers say

**Against the random floor.** HateMM is the only dataset on which every configuration clears the
floor by a clear margin: ROC 0.60–0.62 and AP 0.31–0.36 against a floor of 0.5003 / 0.2423, an
`AP_norm` of 0.21–0.34, i.e. between a fifth and a third of the way from a random score to the
zero-temporal-resolution ceiling. Elsewhere one cell of nine is above the floor: MHC-EN centralized
(0.5746 / 0.3174, `AP_norm` 0.088). MHC-EN collaborative is just below it (0.4784 ± 0.0134), and
HateClipSeg is at or just below it everywhere (0.4379–0.4915). **MHC-ZH collaborative is materially below
the floor** (0.3213 ± 0.0103), which is the same inverted-ranking pattern the val sweep showed
(0.2363 at one round rising to 0.3200 at ten) and is not seed noise: the score is
anti-correlated with the gold spans, i.e. CLAP's normal model calls the *hateful* stretches the
most normal-looking ones on Chinese short videos.

**Against the broadcast ceiling.** No cell comes close. The best CLAP row on HateMM, AP 0.3570,
sits well under the 0.5829 a perfect video-level classifier with no localisation ability at all
would score. On MHC-ZH the best CLAP row reaches AP 0.2958 against a ceiling of 0.9191. This is the
campaign's recurring finding and CLAP does not change it.

**Collaboration versus centralization.** The paper's ordering (centralized > collaborative) holds on
MHC-EN (+0.096 ROC) and MHC-ZH (+0.121) and reverses slightly on HateClipSeg (−0.031); on HateMM
the two are indistinguishable (0.6015 vs 0.6008) even though the centralized *pseudo-labels* are far
purer (0.772 vs 0.516). That last pair is the most informative number in the section: a large
improvement in pseudo-label purity bought no improvement at all in frame-level ranking, because
purity here is a *video-level* property and the frame-level metric is asking a question the fine
stage never answers.

**The trained scorer adds nothing.** The `normality` ablation — CLAP's aggregated normal density
read directly, no MLP, no FedAvg, no pseudo-labels — is the best of the three configurations on
HateMM ROC (0.6210), on MHC-ZH (0.5383, the only MHC-ZH row above the floor), and on HateClipSeg
(0.4915). It loses to the trained model only on HateMM AP and on MHC-EN. So the small amount of
frame-level signal CLAP has in this domain lives in **the feature-norm density**, not in anything
the coarse-to-fine pseudo-labels or the federated training add on top; the pipeline's own
contribution is at best neutral and on two datasets negative.

**Why HateMM and not the others — an interpretation, not a measurement.** CLAP's two coarse
statistics are the maximum first difference of the squared feature norm and the entropy of the
per-dimension variance of the feature over time. Both measure how much the visual stream changes
within a video, which is a property of the *shot structure*, not of what is being said or shown.
The one corpus where the coarse cluster separates is also the one with the widest duration spread —
HateMM's training pool has median 108.6 s and 90th percentile 255.8 s, against 35.8 s / 59.2 s
(MHC-EN) and 31.2 s / 52.2 s (MHC-ZH); HateClipSeg is uniformly long (median 235.6 s, p90 284.2 s)
and equally uniformly cut. A statistic that ranks videos by how much they change has more to
separate on when the pool contains both 20-second clips and five-minute uploads. That reading is
consistent with the numbers but is not tested here; the measured facts are the P.7 cluster purities
and the P.5 rankings.

**Does the literature mechanism work in the hate domain, yes or no, on what evidence** — No: CLAP's
coarse-to-fine pseudo-labelling and federated training clear the random floor on only one of four
corpora (HateMM, ROC 0.6015 ± 0.0147, AP 0.3570 ± 0.0050 against 0.5003 / 0.2423), fall below it on
MHC-ZH (0.3213 ± 0.0103), never approach the video-level broadcast ceiling, produce a video-level
cluster indistinguishable from chance on three of four corpora (P.7), and are beaten by the
ablation that deletes the pseudo-labels and the trained scorer entirely — CLAP's normal-feature
density read directly — on frame ROC-AUC on three of four corpora and on frame PR-AUC on two of
four.
