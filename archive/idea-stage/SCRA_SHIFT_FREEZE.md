# SCRA shift-measurement freeze (2026-08-10)

Decision rules frozen BEFORE `idea-stage/scra_shift_probe.py` was written and before any number
was computed. Zero-label on test: **only test *inputs* are read**; test labels are never loaded
by the probe (the loader drops `y` for the test split).

## What is measured (per dataset, per encoder)

Cells = the four R4 cells (`HateMM/LoRA`, `MHC/Qwen`, `MHC_zh/LoRA`, `ImpliHateVid/CLIP`) plus
CLIP on all four as a second encoder view.

1. **M1 domain-classifier OOF AUC**, val-vs-test and train-vs-test, on L2-normalised `[img;txt]`,
   5-fold cross-fitted logistic regression. Reported at full dimension and after PCA-32 (PCA fitted
   on the pooled unlabelled inputs). This is the standard proxy for "is there a detectable shift".
2. **M2 density-ratio statistics** from the cross-fitted val-vs-test classifier:
   `w_i = (n_val/n_test) * s_i/(1-s_i)` for val items; report `max`, `q95`, `ESS/n`.
   Clipping for downstream use: `w` clipped to its own [1st, 99th] percentile (declared here).
3. **M3 MMD^2** (RBF, median heuristic) with a 200-permutation p-value, val vs test.
4. **M4 support coverage**: for each test item, cosine distance to nearest train item; fraction of
   test items whose NN distance exceeds the 95th percentile of the train-to-train NN distance.
5. **M5 the AUC-relevance of the shift** — the load-bearing measurement. Train the deployed bare
   head (r4_harness, BCE, 3 seeds, val-macro-F1 epoch selection) and compute on the **val** split:
   - `AUC_plain` = ordinary ROC of the head on val;
   - `AUC_iw` = importance-weighted pairwise AUC with weights M2. Under covariate shift
     (`P(y|x)` invariant) this is a consistent estimate of the head's **test** AUC.
   - `Delta = AUC_iw - AUC_plain` = how much AUC the covariate shift is worth for this head.
   - `se(AUC_iw)` by 200-replicate bootstrap over val items (weights held fixed).

## Frozen decision rules

Let `Delta_d` and `se_d` be the seed-mean values for dataset `d` (encoder = the R4 cell encoder).

- **R1 (no room).** If `median_d |Delta_d| < 0.01`, the total AUC the shift moves for the deployed
  head is smaller than the already-banked pairwise-head baseline gain (+0.008..+0.017, §8.8).
  Any "safe" adaptation must fit inside this, so the safe set has nothing worth having.
- **R2 (uncertifiable).** If in >= 3 of 4 datasets `se_d > |Delta_d|`, then the very estimator any
  label-free certificate must be built on has a standard error larger than the effect it is
  supposed to certify. A certificate must subtract that error, so it can never clear zero.
- **VERDICT VACUOUS** if R1 or R2 fires.
- **VERDICT space-exists (proceed to theory)** only if some dataset has
  `|Delta_d| >= 0.02` AND `se_d <= |Delta_d| / 2`.
- Anything else (mixed) -> report as AMBIGUOUS and let the theory step decide.

No grid, no re-runs, no metric added after seeing output. Seeds 0,1,2. If the probe crashes it may
be fixed and re-run; the rules above do not change.
