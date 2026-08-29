# SCRA shift probe — deviation D1 (2026-08-10)

## What changed
`SCRA_SHIFT_FREEZE.md` measures M5 (`Delta = AUC_iw - AUC_plain`) but gives no **null calibration**
for it. After M1/M3 came back at chance (domain-classifier OOF AUC 0.42–0.56, MMD permutation
p = 0.43/0.52/0.96), the observed nonzero `Delta` values could be either (a) real shift, or
(b) pure density-ratio estimation noise at n_val ≈ 80–107. The frozen rules cannot separate these.

## Added measurement (negative control, `--null` mode)
Split the **train** split into two random halves A and B. By construction there is **zero**
covariate shift between A and B. Run the identical pipeline: cross-fitted PCA-32 domain classifier
A-vs-B, ratio, clip, `Delta_null = AUC_iw(A;w) - AUC_plain(A)` using the same deployed head's
scores. 40 random half-splits per dataset, seed-0 head only.

## Rule, frozen before running
- **R3 (noise floor).** If the observed `|Delta_d|` lies inside the central 90 % interval of the
  `|Delta_null|` distribution in **>= 3 of 4** datasets, then the AUC movement attributed to
  covariate shift is statistically indistinguishable from the movement produced by weights fitted
  to no shift at all. R3 firing supports **VACUOUS**.
- If instead observed `|Delta_d|` exceeds the null 95th percentile in **>= 2 of 4** datasets, the
  shift is real and R3 argues **against** VACUOUS; the theory step must then decide on its own.

This control is two-sided: it can rescue the candidate as well as bury it. No other rule changes.
