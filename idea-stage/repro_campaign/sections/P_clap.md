## P. Method as run — CLAP (CVPR 2024)

CLAP is the unsupervised half of the campaign's Wave 2: it is given a pool of videos with **no
labels of any kind** — not even a "these are normal" pool — and has to invent its own supervision.
It does that in two stages. A *coarse* stage clusters each participant's videos into a normal and
an abnormal group from two hand-designed video-level statistics; a *fine* stage fits a Gaussian to
the snippet-level feature norms of the normal group, mixes the participants' Gaussians into one
density, and marks, inside every video the coarse stage called abnormal, the one window of fixed
relative length across which that density changes fastest. Those marks are the pseudo-labels. A
small MLP is then trained on them with BCE, federated across participants with FedAvg.

**Code path.** `third_party/CLAP` @ `3dcaadc1` (AnasEmad11/CLAP, CVPR 2024). The campaign patch is
`scripts/repro_campaign/patches/CLAP.patch`; the data-adaptation layer and driver are
`scripts/repro_campaign/run_clap.py`, the corpus chain is
`scripts/repro_campaign/run_clap_chain.sh`, logs in `logging/runs/repro_clap/run.log`. The entry
point that is actually executed is the repo's own `src/server/fedavg.py`, with the repo's own
`train.sh` argument set. The clustering, the Gaussian mixture, the pseudo-label window search
(`data/utils/datasets.py::C2FPL_client`, `::gmm_PL`), the scorer (`src/config/models.py::C2FPL_XD`),
the local BCE fit (`src/client/fedavg.py::fit_ucf`) and the FedAvg aggregation
(`src/server/fedavg.py::aggregate`) all run unmodified.

`MODEL_ASSETS_STATUS` row 10 marked CLAP **BLOCKED** because the repo is a federated-learning
codebase that reads UCF-Crime concatenated features plus scene-partition `.pkl` files and nothing
else. That is exactly what the adaptation layer below supplies. `visdom` is plot-only and is
reached solely under `--visible 1`, which is off; it was never installed. `wandb` *is* called
unconditionally by both the server and the client, so an offline stub
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
and the repo's own evaluator upsamples a snippet score to 16 frames. The nearest thing our frozen
4 fps cache supports is **one snippet = 2 consecutive 4 fps frames = 0.50 s**, mean-pooled over the
1024-d CLIP-L/336 vectors, trailing remainder frames dropped as a fixed-stride clip extractor drops
them. **CLAP's native output rate is therefore 2 samples per second**, recorded per video in the
npz and broadcast piecewise-constant onto the 4 fps grid by the shared evaluator (freeze §1). No
feature was re-extracted.

**Client partition.** `train.sh` uses `scene_partition_11_V3.pkl`: 11 participants, partitioned by
surveillance scene. Our corpora carry no scene metadata, so the rule — written down here and
nowhere else — is **`client = crc32(video_id) mod 11`**: deterministic, reproducible from the video
id alone, and independent of every label. It gives 11 IID participants where CLAP's own split is
11 non-IID ones; that is deviation D-b below.

**Scorer.** Our features are 1024-d, so the head used is the repo's **`c2fpl_XD`** (`nn.Linear(1024,
512) → 512 → 32 → 1`, dropout 0.6, sigmoid, mean over the crop axis), not `c2fpl_ucf`, which is the
same architecture with a 2048-d first layer for I3D. `c2fpl_XD` is the repo's own head for its own
1024-d dataset and is used verbatim, including the fact that the released file leaves
`self.apply(weight_init)` commented out on this head — so it carries PyTorch's default Linear init
rather than the Xavier init `c2fpl_ucf` gets.

**Supervision column.** Freeze §9 pre-classified CLAP as `one-class`. It is not: the configuration
run here is the paper's own `--train_mode US`, in which no pool of known-normal video exists and the
normal set is a *product* of the coarse clustering. The row is reported as **`unlabelled`**. The
training pool is every **train-split** video of the dataset with at least 4 snippets; no test-split
video and no label of any split enters training, clustering, or knob selection.

### P.2 Deviations forced by the transplant

- **D-a — features.** Single-view CLIP-L/336 at 0.5 s snippets replaces 10-crop I3D. The crop axis
  is kept as a singleton (`ncrops = 1`), which makes the model's `x.mean(dim=1)` a no-op and makes
  `get_matrix`'s mean over crops a no-op. Everything CLAP computes from the features — squared L2
  norm per snippet, its first difference, its per-dimension variance over time — is defined on any
  feature stream and is computed here exactly as released.
- **D-b — partition.** IID `crc32 mod 11` instead of a scene partition. CLAP's paper argues the
  scene split is the *hard* case for collaboration; an IID split is the easier one, so this
  deviation is not adverse to the method.
- **D-c — the `eta_clustering=1` code path does not run, and is not ours to fix.** `train.sh` sets
  `--eta_clustering 0`, and that is what was run. The alternative path,
  `C2FPL_client_eta`, initialises `normal_set = {}` / `abnormal_set = {}`, fills `set_n` / `set_a`
  instead, and has the four lines that would choose between them commented out — so with
  `--load 0` it always proceeds with two empty dictionaries and fits a Gaussian to an empty array.
  Reported as a defect of the release, not worked around.
- **D-d — model selection.** The repo's `evaluate_ucf` loads UCF-Crime frame labels from
  `labels/gt-ucf-RTFM.npy` and `test_ucf` reports `max` AUC over client models before and after
  local training — i.e. it selects on test labels. The campaign forbids that (freeze §10 red
  line 1), so the in-loop readout is disabled and the row reports the **single aggregated global
  model** at the frozen round, scored once, through the campaign's own evaluator.
- **D-e — covariance diagonal computed directly.** `covariance_mat` materialises the full `D × D`
  covariance of the time series and the caller reads only `np.diagonal` of it. At `D = 1024` and up
  to 11.5k snippets that is ~1.2 × 10¹⁰ flops per video. It now returns
  `np.diag(np.var(X, axis=0, ddof=1))`, which is *exactly* the diagonal `np.cov` would have
  produced. Numerically identical on every entry the algorithm reads.
- **D-f — five videos have no row.** Two HateMM containers have no video stream at all (freeze D2:
  `hate_video_147`, `hate_video_292`) so no feature exists. One HateMM and one MHC-EN training video
  are shorter than 4 snippets (2 s) — `hate_video_272`, `XScP1AiMkNM` — and the coarse stage needs
  `np.diff` of a length-≥2 norm curve plus a non-degenerate window, so they are excluded from the
  *training pool only*; both are still scored. No zero curve is fabricated for anything.
- **D-g — plotting and logging.** `visdom` not installed (plot-only, unreachable at `--visible 0`);
  `wandb` replaced by an offline stub. `--global_testset 1` (rather than `train.sh`'s 0) so the
  ordered score-set `Subset` exists for the score dump; with the in-loop AUC readout disabled this
  changes nothing that is trained.

### P.3 Frozen knobs

Every hyper-parameter is the published `train.sh` value: `--train_mode US`, `--local_epoch 10`,
`-bs 128`, SGD `lr = 1e-2`, momentum 0, weight decay 0, `MultiStepLR([5, 10], γ = 0.1)`,
`--join_ratio 1`, `--gmm_pl 1`, `--eta_clustering 0`, 2-component `GaussianMixture(max_iter=150,
random_state=0)`, pseudo-label window fraction 0.2.

The one free knob is the **number of FedAvg global rounds**. The selection rule was written to
`idea-stage/repro_clap/knob_rule.json` *before any number existed*: candidates {1, 2, 5, 10}, chosen
on the **val** split by `frame_ROC_AUC` (the metric CLAP's own paper reports) of the `fedavg11`
configuration at seed 20250819, ties to the smaller count, one value per dataset, then reused for
the `central` configuration and all three seeds. The selected values and the val numbers behind
them are in `idea-stage/repro_clap/run_record.json`.

Because model init, dropout and client sampling are stochastic, freeze §6 applies: **3 seeds**
(20250819 / 20250820 / 20250821), mean ± sd. The `normality` ablation has no trained component and
`GaussianMixture(random_state=0)` is fixed, so it is deterministic and carries `seeds = 1`.

### P.4 Variants reported

| key | what it is |
|---|---|
| `collaborative (11 clients, FedAvg)` | the faithful port of `train.sh`: 11 participants, FedAvg |
| `centralized (1 client)` | the paper's own "Centralized" configuration — one participant holding the whole pool |
| `normality (ablation, ours)` | CLAP's aggregated normal-Gaussian density evaluated per snippet, negated, with **no MLP trained at all**. CLAP uses this density only to *locate a window* inside a video the coarse stage already called abnormal; reading it directly as a per-snippet score isolates how much of the method's localisation is in the density and how much is in the trained scorer. |

`F1@tIoU` reads `n/a` for every CLAP row: the method emits a score curve, and freeze §2 forbids
inventing a threshold to turn it into intervals.

Reproduce:
`python scripts/repro_campaign/run_clap.py --stage build`, then
`bash scripts/repro_campaign/run_clap_chain.sh`, then
`python scripts/repro_campaign/clap_select_knob.py` (val only), then
`python scripts/repro_campaign/run_clap.py --stage curves`, then
`python scripts/repro_campaign/eval_frame.py --method curves --curve-dir idea-stage/repro_clap/curves --variants ... --method-name "CLAP" --wave 2 --supervision unlabelled --split test`.

*(P.5 headline table, P.6 strata and P.7 "what the numbers say" are filled once the corpus run and
the single test call are complete.)*
