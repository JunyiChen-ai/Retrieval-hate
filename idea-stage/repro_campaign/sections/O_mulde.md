## O. Method as run — MULDE (CVPR 2024)

MULDE's premise is that anomaly is **low density under a model of normality**. It fits a
noise-conditioned log-density network to a pool of "these are normal" feature vectors by denoising
score matching, then reads a per-sample anomaly score off that network at a ladder of noise scales
and aggregates across the ladder. It is the campaign's first `one-class` row: it is the only method
so far that consumes target-dataset labels at all, and it is reported in its own supervision column,
never mixed with the label-free rows.

**Code path.** `third_party/MULDE` @ `f821b965` (jakubmicorek/MULDE). The model
(`models.py::MLPs`, `ScoreOrLogDensityNetwork`), the denoising-score-matching loss with its `σ²`
weighting, the log-uniform noise sampling, the `β`-weighted noise-free log-density regulariser, the
`Adam(lr=4e-5, betas=(0.5, 0.9))` optimiser, the component-wise standardisation by the training
pool's own mean and sd, and the multiscale read-out are all upstream's and were transcribed without
change of arithmetic. `scripts/repro_campaign/run_mulde.py` is the corpus loop around that code.

**What the read-out is, and where it comes from.** Upstream's `calculate_scores` evaluates each
clean sample at each of `L = 16` noise scales linspaced over `[σ_low, σ_high]` and keeps two
quantities per scale: the network's own output (`log_density`) and the squared norm of its score,
`‖∇ₓ(−log p)‖²` (`score_norm`). It then aggregates the resulting 16-dimensional vector four ways,
all of which are computed here:

- **max / median / mean** of the vector after standardising each of the 16 coordinates by the
  **training pool's** mean and sd for that coordinate;
- **GMM negative log-likelihood** — a full-covariance Gaussian mixture is fitted to the *training*
  16-dimensional vectors and the evaluated sample is scored by `−log p_GMM`. Upstream fits 1, 3 and
  5 components; all three are computed here.

Higher = more anomalous in every case, which is upstream's own sign convention (its
`roc_auc_score(labels_test, …)` calls pass these quantities unmodified). One fidelity detail worth
recording: upstream applies the `λ = σ²` weight only to the per-scale quantity it prints, not to the
per-scale arrays it actually aggregates (`scores_by_sigma`), so the aggregated `score_norm` here is
unweighted, matching what upstream aggregates.

**Training pool.** For each dataset, every 4 fps frame of every **train-split video whose
video-level gold is non-hateful** (`y_video == 0` in `data/gt/frame_gt_4fps/<DS>.npz`). Nothing
else. No test-split video and no test label reaches training, hyper-parameter selection or the GMM
fit at any point.

| dataset | train videos fitted | train frames | val frames | val positive rate |
|---|---|---|---|---|
| HateMM | 446 | 258,807 | 53,287 | 0.3078 |
| MHC-EN | 387 | 54,460 | 11,430 | 0.3045 |
| MHC-ZH | 387 | 50,672 | 9,964 | 0.2341 |
| HateClipSeg | 30 | 27,038 | 37,764 | 0.5088 |

HateClipSeg's pool is the one to watch: only **30** videos in the corpus are non-hateful at video
level, so MULDE's model of "normal" there is built from 7.6 % of the corpus.

**Features.** The frozen campaign cache `data/CLIP_Embedding/<DS>/dense4fps_clipL336/<vid>.npy`,
`(T, 1024)` float32, native rate 4 fps. Nothing was re-extracted. Native rate equals the evaluation
grid, so no broadcast is involved and `native_rate = 4 fps`.

**Output.** One `.npz` per video under `idea-stage/repro_mulde/curves/<DS>/`, holding
`clipL336_s0/_s1/_s2` (the three seeds) and `rate = 4.0`. MULDE emits no intervals, so there is no
intervals file and every `F1@tIoU` cell reads `n/a`; freeze §2 forbids inventing a threshold for a
score-curve method.

### O.1 Frozen hyper-parameters and where they came from

Fixed at upstream's published values and never searched: `lr = 4e-5`, `batch_size = 2048`,
`β = 0.1`, `σ_low = 1e-3`, `L = 16`, Adam betas `(0.5, 0.9)`, no scheduler, no dropout, no
layernorm, no gradient clipping, component-wise standardisation by the train pool. These are the
"essential parameters" line of the upstream README.

Searched on the **val split only**, by frame ROC-AUC, at seed 20250819, and written into
`idea-stage/repro_mulde/run_record_<DS>_clipL336.json` **before** the single test call: hidden width
`{[4096,4096], [1024,1024]}` × top noise scale `σ_high {0.5, 1.0}` × epoch budget `{25, 50, 100}` ×
read-out `{log_density, score_norm}` × aggregation `{max, median, mean, gmm1, gmm3, gmm5}` = 144 val
cells per dataset. The epoch budget is not a separate training run: one 100-epoch run is
checkpointed at 25/50/100 and each checkpoint scored on val.

| dataset | units | σ_high | epochs | read-out | aggregation | val ROC | **test ROC** | val → test |
|---|---|---|---|---|---|---|---|---|
| HateMM | [4096, 4096] | 1.0 | 50 | `log_density` | `gmm1_nll` | 0.6460 | 0.5989 | −0.047 |
| MHC-EN | [1024, 1024] | 1.0 | 100 | `score_norm` | `gmm3_nll` | 0.5754 | 0.4737 | −0.102 |
| MHC-ZH | [1024, 1024] | 1.0 | 25 | `score_norm` | `gmm3_nll` | 0.5769 | 0.5102 | −0.067 |
| HateClipSeg | [4096, 4096] | 0.5 | 50 | `score_norm` | `gmm5_nll` | 0.5241 | 0.5276 | +0.004 |

Two things in this table are findings rather than bookkeeping.

**No read-out is stable across datasets.** On HateMM `log_density` dominates (val range across the
grid 0.5942–0.6460) and `score_norm` is the weaker and wildly variable half (0.3931–0.6384, i.e. it
can land well *below* chance). On both MHC datasets the ordering reverses: `log_density` is flat and
useless (EN 0.5017–0.5306, ZH 0.4044–0.5062) while `score_norm` supplies whatever signal exists
(EN up to 0.5754, ZH up to 0.5769). MULDE's multiscale aggregation has no dataset-independent best
choice in this domain, so any single frozen read-out would have been wrong on half the corpora.

**Val selection does not transfer.** Three of the four datasets lose more val→test than the
across-seed sd of the test number (which is 0.003–0.012). MHC-EN loses 0.102 and crosses from above
chance to below it. With 144 val cells and val pools of 10k–53k frames, part of the val maximum is
val noise rather than a property of the configuration.

Stochastic elements (weight init, noise sampling, batch order, GMM init) mean freeze §6 applies:
**3 seeds, 20250819 / 20250820 / 20250821, mean ± sd reported.** The chosen configuration is frozen
once from the selection seed and reused for all three.

Wall clock: **79 minutes** for the whole headline variant across all four datasets on a shared
RTX 5090 (51 min of it selection: HateMM 2,049 s, MHC-EN 395 s, MHC-ZH 377 s, HateClipSeg 266 s).
Peak GPU memory ~1.1 GiB under a hard `set_per_process_memory_fraction(0.10)` cap, so the
concurrent UniTime corpus run was never at risk of an OOM.

### O.2 Headline rows — test split

Control rows are the campaign's standard §3 controls, quoted from §M.1 (same GT, same pools).

| method | wave | dataset | split | supervision | variant | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | coverage | missing | seeds | transplant | gt_convention | run_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOLD_BROADCAST | — | HateMM | test | control | control | video | 0.8857 | 0.5829 | n/a | n/a | n/a | 1.0000 | 116975 | 0.2421 | 1.0000 | 0/215 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ |
| RANDOM_UNIFORM | — | HateMM | test | control | control | 4 fps | 0.5003 ± 0.0019 | 0.2423 ± 0.0013 | n/a | n/a | n/a | 0.0000 | 116975 | 0.2421 | 1.0000 | 0/215 | 20 | n/a | §4 | idea-stage/repro_campaign/ |
| **MULDE** | 2 | HateMM | test | one-class | clipL336 | 4 fps | **0.5989 ± 0.0031** | **0.3078 ± 0.0048** | n/a | n/a | n/a | 0.1923 ± 0.0142 | 116908 | 0.2422 | 1.0000 | 0/215 | 3 | n/a | §4 | idea-stage/repro_mulde/ |
| GOLD_BROADCAST | — | MHC-EN | test | control | control | video | 0.9427 | 0.7664 | n/a | n/a | n/a | 1.0000 | 22337 | 0.2734 | 1.0000 | 0/161 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ |
| RANDOM_UNIFORM | — | MHC-EN | test | control | control | 4 fps | 0.5004 ± 0.0034 | 0.2737 ± 0.0026 | n/a | n/a | n/a | 0.0000 | 22337 | 0.2734 | 1.0000 | 0/161 | 20 | n/a | §4 | idea-stage/repro_campaign/ |
| **MULDE** | 2 | MHC-EN | test | one-class | clipL336 | 4 fps | **0.4737 ± 0.0117** | **0.2530 ± 0.0048** | n/a | n/a | n/a | −0.0414 ± 0.0098 | 22336 | 0.2734 | 1.0000 | 0/161 | 3 | n/a | §4 | idea-stage/repro_mulde/ |
| GOLD_BROADCAST | — | MHC-ZH | test | control | control | video | 0.9842 | 0.9191 | n/a | n/a | n/a | 1.0000 | 18199 | 0.2648 | 1.0000 | 0/149 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ |
| RANDOM_UNIFORM | — | MHC-ZH | test | control | control | 4 fps | 0.4985 ± 0.0052 | 0.2646 ± 0.0038 | n/a | n/a | n/a | 0.0000 | 18199 | 0.2648 | 1.0000 | 0/149 | 20 | n/a | §4 | idea-stage/repro_campaign/ |
| **MULDE** | 2 | MHC-ZH | test | one-class | clipL336 | 4 fps | **0.5102 ± 0.0028** | **0.2490 ± 0.0009** | n/a | n/a | n/a | −0.0242 ± 0.0014 | 18195 | 0.2649 | 1.0000 | 0/149 | 3 | n/a | §4 | idea-stage/repro_mulde/ |
| GOLD_BROADCAST | — | HateClipSeg | test | control | control | video | 0.6260 | 0.5437 | n/a | n/a | n/a | 1.0000 | 114097 | 0.4712 | 1.0000 | 0/119 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ |
| RANDOM_UNIFORM | — | HateClipSeg | test | control | control | 4 fps | 0.5009 ± 0.0021 | 0.4721 ± 0.0016 | n/a | n/a | n/a | 0.0000 | 114097 | 0.4712 | 1.0000 | 0/119 | 20 | n/a | §4 | idea-stage/repro_campaign/ |
| **MULDE** | 2 | HateClipSeg | test | one-class | clipL336 | 4 fps | **0.5276 ± 0.0066** | **0.4890 ± 0.0095** | n/a | n/a | n/a | 0.2505 ± 0.1317 † | 114021 | 0.4709 | 1.0000 | 0/119 | 3 | n/a | §4 | idea-stage/repro_mulde/ |

Coverage is 1.0000 everywhere and no video is missing from any test pool: MULDE scores every frame
it is given, it cannot refuse, and the CLIP cache is complete on all four test splits. The two
HateMM videos absent from the full-corpus row are the D2 audio-only containers, neither of which is
in any split.

† HateClipSeg's `AP_norm` carries the evaluator's `AP_norm_reliable = False` flag: the
broadcast-minus-random denominator there is only 0.0724, so a 0.01 wobble in AP moves `AP_norm` by
0.14. That is why its sd (0.13) is twenty times the sd of its own AP (0.0095). Read the raw AP on
that dataset, not the normalised column. The other three denominators are 0.34 / 0.49 / 0.65 and
their `AP_norm` values are usable.

### O.3 Stratified sub-tables — single-span vs multi-span (test split)

Convention as elsewhere in this file: zero-span videos supply the negative frames and appear in both
strata, otherwise a stratum is single-class and its ROC/AP are undefined.

**Single-span videos**

| method | dataset | split | variant | frame_ROC_AUC | frame_PR_AUC | AP_norm | n_frames | base_rate |
|---|---|---|---|---|---|---|---|---|
| MULDE | HateMM | test | clipL336 | 0.6329 ± 0.0045 | 0.3055 ± 0.0080 | 0.1900 ± 0.0145 | 93251 | 0.2008 |
| MULDE | MHC-EN | test | clipL336 | 0.4758 ± 0.0121 | 0.2442 ± 0.0057 | −0.0361 ± 0.0112 | 21668 | 0.2628 |
| MULDE | MHC-ZH | test | clipL336 | 0.5102 ± 0.0028 | 0.2490 ± 0.0009 | −0.0242 ± 0.0014 | 18195 | 0.2649 |

**Multi-span videos**

| method | dataset | split | variant | frame_ROC_AUC | frame_PR_AUC | AP_norm | n_frames | base_rate |
|---|---|---|---|---|---|---|---|---|
| MULDE | HateMM | test | clipL336 | 0.5693 ± 0.0007 | 0.1350 ± 0.0028 | 0.1021 ± 0.0092 | 91996 | 0.1043 |
| MULDE | MHC-EN | test | clipL336 | 0.4963 ± 0.0290 | 0.0280 ± 0.0029 | 0.0009 ± 0.0049 | 15036 | 0.0274 |
| MULDE | MHC-ZH | test | clipL336 | n/a | 0.0000 ± 0.0000 | n/a | 12952 | 0.0000 |

MHC-ZH's multi-span test stratum contains **zero positive frames** — the corpus is 98.2 % single-span
and the test split happens to contain no multi-span positive at all — so ROC and `AP_norm` are
undefined there and the cell reads `n/a`. This is a property of the split, not of MULDE.

On HateMM, MULDE is better on single-span videos (0.6329) than on multi-span ones (0.5693), a 0.064
gap that is twenty times the across-seed sd. On MHC-EN both strata sit at or below chance and the
multi-span stratum's 0.029 sd is the largest in the table, driven by a 2.7 % base rate.

### O.4 Full corpus — contaminated by construction, reported for one reason only

**This block includes the very train-split videos MULDE was fitted on.** It is not a headline and
must never be quoted as one. It is printed because the size of the gap between it and the test row
is itself the cleanest measurement in this section.

| method | dataset | split | variant | frame_ROC_AUC | frame_PR_AUC | AP_norm | n_frames | base_rate | missing |
|---|---|---|---|---|---|---|---|---|---|
| MULDE | HateMM | all | clipL336 | 0.8011 ± 0.0043 | 0.5531 ± 0.0024 | 0.6883 ± 0.0060 | 623788 | 0.2858 | 2/1083 |
| MULDE | MHC-EN | all | clipL336 | 0.8314 ± 0.0042 | 0.4931 ± 0.0146 | 0.4674 ± 0.0273 | 110728 | 0.2441 | 0/792 |
| MULDE | MHC-ZH | all | clipL336 | 0.7091 ± 0.0078 | 0.4482 ± 0.0110 | 0.3241 ± 0.0182 | 102130 | 0.2538 | 0/814 |
| MULDE | HateClipSeg | all | clipL336 | 0.5576 ± 0.0031 | 0.4960 ± 0.0064 | 0.4914 ± 0.0975 † | 375250 | 0.4635 | 0/395 |

Put the three pools side by side, using the same frozen configuration and the same seed family:

| dataset | share of corpus videos MULDE was fitted on | val ROC (unseen videos, but selection-optimistic) | test ROC (untouched) | full-corpus ROC | corpus − test |
|---|---|---|---|---|---|
| MHC-EN | 387/792 = 48.9 % | 0.5754 | 0.4737 | 0.8314 | **+0.358** |
| MHC-ZH | 387/814 = 47.5 % | 0.5769 | 0.5102 | 0.7091 | **+0.199** |
| HateMM | 446/1081 = 41.3 % | 0.6460 | 0.5989 | 0.8011 | **+0.202** |
| HateClipSeg | 30/395 = 7.6 % | 0.5241 | 0.5276 | 0.5576 | **+0.030** |

The corpus-minus-test gap tracks the share of the pool that was fitted, and collapses to noise on
the one dataset where almost nothing was fitted. The full-corpus number is memorisation of the
negatives the model was trained on, not localisation. Anyone reading a one-class anomaly detector's
"full corpus" AUC in this domain is reading the training set.

### O.5 Deviations

Every item below is a departure from the published pipeline, named as required.

1. **The data front-end is ours; upstream has none.** `third_party/MULDE/dataset.py` is a 2-D toy
   mixture generator. The repo ships no video loader, no feature loader, no per-video output and no
   corpus loop. Pool construction, the corpus scoring loop and the `.npz` writer are
   `scripts/repro_campaign/run_mulde.py`. The model and its arithmetic are untouched.
2. **Read-out selection moved from test to val.** Upstream's own evaluation loop selects the best
   noise scale and the best aggregation by **test** ROC-AUC (`_roc_auc_best/*` in its tensorboard,
   and the `best_auc_roc_*` arg-max in `main.py`). Freeze §10 red line 1 forbids that, so the same
   selection is run on **val** and frozen into a run record before the single test call. Reported
   MULDE numbers are therefore *lower* than a naive port of the upstream evaluation loop would
   print, and the difference is not a bug in either.
3. **Epoch budget capped at 100.** Upstream's README example runs 1,000 epochs on a 2-D toy set. The
   budget here is a val-selected choice among {25, 50, 100}. Reason: shared GPU. The chosen budget
   was interior to the grid on three of four datasets (50, 100, 25, 50), so the cap is not binding
   except on MHC-EN, where val preferred the largest budget offered.
4. **Selection restricted to the repo's six aggregate read-outs × two score types.** Upstream also
   reports 16 per-scale "individual" read-outs per score type. Those are computed and logged on val
   as diagnostics but are not selectable, to keep the val selection space at 144 rather than 656
   cells. The logged per-scale val maxima are in the run log.
5. **GMM `random_state` pinned to the run seed.** Upstream leaves it unset, which makes the GMM
   read-outs irreproducible run to run. Pinning it is what makes the ± sd here a seed effect rather
   than an unseeded-init effect.
6. **Scope cut: the audio and concatenated feature variants were dropped.** The brief made
   `dense4fps_w2vemo` and a `clip+w2vemo` concatenation conditional on being cheap. Measured, they
   were not: the headline visual variant alone took 79 min on a card already holding a UniTime
   corpus run whose throughput had fallen from 0.064 to 0.042 gen/s under contention, and the two
   extra variants were projected at a further ~4.6 h. The cut was decided **before any test number
   existed and without looking at any metric** — only wall-clock and GPU-contention measurements —
   so it touches no red line. Neither variant produced any partial result, so nothing is being
   withheld. Consequence: MULDE has no audio row, and the question of whether one-class density
   estimation behaves differently on an audio feature space in this domain is open.
7. **Batching by `randperm` rather than `DataLoader(shuffle=True)`.** Arithmetically equivalent;
   done so the epoch ordering is reproducible from the seed alone.
8. **Run-mechanics, no effect on any number:** a 0.10 per-process GPU memory cap; and one clobbered
   result file — a stale duplicate background task re-ran the test scorer from the wrong working
   directory, found no curves and overwrote `eval_MULDE_test.json` with `[]`. The scorer was re-run
   on the same frozen curves and reproduced the twelve rows identically. This is a file being
   restored, not a second test call: the curves were already frozen on disk and `eval_frame.py` is
   deterministic.

Reproduce: `python scripts/repro_campaign/run_mulde.py --datasets HateMM,MHC,MHC_zh,HateClipSeg
--variants clipL336 --device cuda:0 --mem-frac 0.10`, then `scripts/repro_campaign/eval_frame.py
--method curves --curve-dir idea-stage/repro_mulde/curves --variants
clipL336_s0,clipL336_s1,clipL336_s2 --method-name "MULDE" --wave 2 --supervision one-class --split
{test,all}`. Log: `logging/runs/repro_mulde/run.log`. Frozen choices:
`idea-stage/repro_mulde/run_record_<DS>_clipL336.json`. Environment:
`idea-stage/repro_mulde/run_meta_<DS>_clipL336.json`.

### O.6 What the numbers say

**MULDE clears the random floor on two of four datasets, and only one of those margins is large
enough to matter.** Against the 20-seed uniform floor: HateMM +0.099 ROC (0.5989 vs 0.5003, ~32
floor-sd), HateClipSeg +0.027 (0.5276 vs 0.5009, ~13 floor-sd), MHC-ZH +0.012 (0.5102 vs 0.4985,
~2 floor-sd, i.e. at chance), MHC-EN **−0.027** (0.4737 vs 0.5004, below chance). The AP column says
the same thing: `AP_norm` is 0.19 on HateMM, 0.25 on HateClipSeg (unreliable denominator), and
negative on both MHC datasets.

**Even where it works, it is far below the zero-temporal-resolution ceiling.** On HateMM the
gold video-level broadcast — a method with a perfect video classifier and no localisation ability
whatsoever — reaches ROC 0.8857 / AP 0.5829, against MULDE's 0.5989 / 0.3078. MULDE recovers 19 % of
the distance from random to a detector that cannot localise at all. This is the campaign's recurring
finding, not a MULDE-specific one, but MULDE is the first method to reach it while consuming
target-dataset labels.

**The mechanism's failure mode is legible.** A density model of normality answers "is this frame
unlike the normal pool", and on these corpora the frames inside a hateful span are not visually
unlike the normal pool. Both a hateful and a non-hateful video of these datasets are overwhelmingly
a person talking to camera, a studio backdrop, on-screen text. The hate is in what is said and
written, not in the visual statistics of the frame — which is exactly the modality gap the project's
own OCR ruling identified. What MULDE's density model does separate well is *which video a frame
came from*, which is why the full-corpus number leaps to 0.80–0.83 the moment fitted videos re-enter
the pool, and why that leap scales with the fitted share (O.4).

**Two secondary observations worth keeping.** (a) The best read-out flips between datasets —
`log_density` on HateMM, `score_norm` on the other three — and the losing read-out can land far below
chance (down to val 0.371). There is no dataset-independent setting of MULDE's own multiscale
aggregation in this domain. (b) Val→test transfer is poor on three of four datasets (up to −0.102),
which is a caution for any future one-class candidate here: a val-selected knob over a 144-cell grid
on a 10k-frame val pool buys less than it appears to.

**Does the literature mechanism — one-class density estimation of normality — work in the hate
domain? No.** On the frozen test splits it beats the random floor on only two of four datasets
(HateMM ROC 0.5989 ± 0.0031, HateClipSeg 0.5276 ± 0.0066), sits at chance on MHC-ZH (0.5102 ± 0.0028)
and below chance on MHC-EN (0.4737 ± 0.0117); its best result recovers 19 % of the gap between random
and a detector with no temporal resolution at all; and the one number that looks strong — 0.80–0.83
on the full corpus — is produced by the training videos being inside the evaluated pool, as shown by
that gap shrinking to +0.030 on HateClipSeg, the one dataset where almost nothing was fitted.
