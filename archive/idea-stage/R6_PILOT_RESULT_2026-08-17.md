# Round-6 pilot results — 2026-08-17

Design frozen in `idea-stage/R6_PILOT_FREEZE_2026-08-17.md` (commit `753bb08`) before any arm cache
was built or any head was trained. Decision rules applied verbatim below.

**Cost: ¥0.00 API (of a ¥60 round budget), under 15 minutes of GPU across both pilots.**

---

## R6-1 — Multi-layer readout fusion — **KILL**

24 head runs (2 datasets × 4 arms × 3 seeds), 0 failures, **203 s** wall.
Artifacts: `idea-stage/r6_readout/{build_arms.py,run_arms.sh,analyze.py,build_meta.json,results.json}`,
logs `logging/runs/r6_readout/`.

Provenance guards, all passed: `ids` and `labels` identical between the L24 and L28 sources on all
6 dataset×split pairs; the random projection R is a single (3584,3584) draw from
`default_rng(20260817)`, sha256 `a64c3e16fefbbb0c593b424b264318e31ad4bb46fc22590abf12ac182bf3386c`,
reused for every split, dataset and stream; source rows were already L2-normalised
(max |‖row‖−1| = 4.17e-07), so re-normalisation was a verified no-op.

Baseline reproduction check: **HateMM A0 = 0.8774 ± 0.0041**, which reproduces the banked contrast
line for HateMM to four decimals. The instrument is the same one the 2026-08-13/14 series used.

### Test macro-F1 at the val-selected epoch

| dataset | A0 (L28) | L24 | CAT (L28‖L24) | RANDCAT (L28‖R·L28) |
|---|---|---|---|---|
| HateMM | **0.8774** ± 0.0041 | 0.8628 ± 0.0013 | 0.8759 ± 0.0016 | 0.8712 ± 0.0023 |
| MHC_zh | 0.7603 ± 0.0531 | 0.7798 ± 0.0235 | 0.7873 ± 0.0291 | 0.7854 ± 0.0069 |

### Paired per-seed deltas

| dataset | delta | s0 | s1 | s2 | mean | #pos |
|---|---|---|---|---|---|---|
| HateMM | CAT−A0 | −0.0046 | −0.0006 | +0.0005 | **−0.0016** | 1/3 |
| HateMM | CAT−RANDCAT | +0.0082 | +0.0030 | +0.0028 | +0.0047 | 3/3 |
| HateMM | L24−A0 | −0.0185 | −0.0158 | −0.0097 | −0.0147 | 0/3 |
| HateMM | RANDCAT−A0 | −0.0128 | −0.0036 | −0.0023 | −0.0062 | 0/3 |
| MHC_zh | CAT−A0 | +0.1160 | −0.0147 | −0.0203 | **+0.0270** | 1/3 |
| MHC_zh | CAT−RANDCAT | +0.0263 | −0.0276 | +0.0068 | +0.0018 | 2/3 |
| MHC_zh | L24−A0 | +0.0511 | +0.0174 | −0.0100 | +0.0195 | 2/3 |
| MHC_zh | RANDCAT−A0 | +0.0897 | +0.0129 | −0.0271 | +0.0252 | 2/3 |

### Verdict
- HateMM: clause 1 false (−0.0016), clause 2 false (1/3), clause 3 false (+0.0047). Fails.
- MHC_zh: clause 1 true (+0.0270), clause 2 false (1/3), clause 3 false (+0.0018). Fails.

**Zero of two datasets pass → KILL.** No subgroup rescue, no reinterpretation.

### What the numbers mean
1. **The MHC_zh mean is seed luck, and the control proves it.** A0 on MHC_zh has a seed std of
   **0.0531** (seed 0 = 0.7016 vs seed 2 = 0.8050). The +0.0270 CAT−A0 mean is carried entirely by
   seed 0's +0.1160, where A0 landed badly; seeds 1 and 2 are both negative. RANDCAT−A0 on the same
   dataset is **+0.0252** — nearly the same size. What moved was dimensionality and seed variance,
   not layer information. Without RANDCAT this would have read as a +2.7-point result.
2. **The L24 half does carry a little more than a random projection of L28, and it is not enough.**
   On HateMM, CAT − RANDCAT = +0.0047 with 3/3 seeds positive — a real but sub-threshold signal —
   while CAT − A0 = −0.0016. Layer 24 adds a small amount of non-random information and adding it
   still does not beat layer 28 alone. This is Law I in miniature at the smallest possible scale.
3. **L24 alone is clearly worse on HateMM** (−0.0147, 0/3), which is the first head-trained
   measurement of the quantity the 2026-07 readout screen tried to estimate with a kNN proxy whose
   permutation null was 8-9 points wide.
4. **The readout axis is now closed on the arena that counts.** The prior closure was on the raw
   retrieval arena that F111 later declared unvalidated as a predictor. This one is on the deployed
   head path, which satisfies F113's rule that a promotion must be rendered on the fold-head path.

### Incidental finding
`hate_video_95` (HateMM train index 355) is an **all-zero row in both `img_feats` and `text_feats`**
in the source `ro_L24` and `ro_L28` caches. This is a pre-existing extraction failure in those
caches, propagating identically into all four arms, so it cannot bias this comparison. It is
consistent with the 2026-08-09 degenerate-feature audit, which recorded `hate_video_95` as repaired
in the `-degenfix1` caches — the repair was never propagated to the `ro_` family.

---

## R6-2 — Transductive pool refinement — **AMBIGUOUS by the frozen rule; the mechanism is dead on
the leakage-free split**

12 head runs + 24 split dumps + the full grid, **~8 minutes** wall.
Artifacts: `idea-stage/r6_trans/{run_heads.sh,dump_r6.py,em_r6.py,results.json,dumps/}`,
logs `logging/runs/r6_trans/`.

Dump fidelity: macro-F1 recomputed from the dumped probabilities matched the trainlog to < 5e-4 on
**all 24 split-dumps**.

### Test macro-F1

| dataset | IND | TRANS | SHUF | mean T−IND | per-seed T−IND | 3/3 pos | mean T−SHUF | passes |
|---|---|---|---|---|---|---|---|---|
| HateMM | 0.8541 | 0.8828 | 0.5178 | +0.0287 | +0.0431, −0.0005, +0.0434 | 2/3 | +0.3649 | no |
| MHC-EN | 0.7273 | 0.7235 | 0.4808 | −0.0038 | −0.0057, 0.0000, −0.0057 | 0/3 | +0.2427 | no |
| MHC-ZH | 0.7776 | 0.8110 | 0.4528 | +0.0334 | +0.0190, +0.0525, +0.0288 | 3/3 | +0.3582 | **yes** |
| ImpliHateVid | 0.9118 | 0.9109 | 0.5079 | −0.0009 | −0.0026, −0.0000, 0.0000 | 0/3 | +0.4030 | no |

**Frozen rule applied verbatim: exactly one dataset satisfies all three conditions → AMBIGUOUS.**

### Two defects, found and reported rather than patched around

**Defect 1 — λ and ρ are numerically inert across the entire frozen grid.** At d = 1024 the Gaussian
log-odds term has magnitude ~3164 (median |Δ| across items) while the KL anchor `λ·log(p₁/p₀)` has
median 1.3 and p90 4.2, i.e. at most ~17 at λ=4. The posteriors saturate to hard 0/1 after one
E-step (**0 of all test items land in (0.01, 0.99)**), EM converges in 2-3 iterations, and **all 15
grid cells return bit-identical dev macro-F1 in all 12 runs**. So what actually ran is **hard
spherical 2-means over the pool, seeded by the head's predictions** — not the KL-anchored TransCLIP
operator the freeze specified. The frozen run did not test the intended mechanism.

**Defect 2 — the SHUF control is degenerate and therefore non-discriminating.** Because λ is inert,
destroying the geometry leaves clustering with nothing to anchor to, so SHUF collapses to chance
(0.45-0.53). The `TRANS − SHUF ≥ +0.005` clause is then free on all four datasets (+0.24 to +0.40)
and never discriminates as designed. Only two of the three clauses were ever binding.

### The corrected experiment, run on dev only — and it is negative everywhere

The anchor term was re-scaled by S ∈ {1, 10, 100, 300, 1000, 3000, 10000} so it can actually
compete, sweeping the whole continuum from pure clustering (S→0) to pure inductive (S→∞). Scored on
**dev labels only**, which is legal and changes nothing frozen:

| dataset | best-over-grid dev delta vs IND, across all S |
|---|---|
| HateMM | −0.0140 → −0.0074 |
| MHC-EN | −0.0040 (flat) |
| MHC-ZH | −0.0088 (flat) |
| ImpliHateVid | −0.0051 → −0.0010 |

**At every anchor strength, on every dataset, the best cell in the grid is negative or zero on the
split where selection is legal, approaching 0 from below.** Both endpoints of the continuum and
everything between fail. The mechanism has no operating point.

### Why the single AMBIGUOUS dataset is not a near-miss
- **MHC-ZH's pass rests on 11 flipped test items across 3 seeds** (2, 7, 3 flips; 11 corrections,
  1 break) on n=149 where one item is worth ~0.005-0.006 macro-F1. Its **dev-side delta on the same
  configuration was 0.000 / −0.0263 / 0.000** — the leakage-free evidence for the cell that passed
  on test is neutral-to-negative.
- **HateMM's near-pass is a threshold artifact, not transduction.** On seeds 0 and 2 the
  val-selected threshold came out at 0.207 and 0.178 (buying +0.0012 on dev) and that threshold
  costs IND about 0.034 on test; TRANS's posteriors are saturated and therefore
  threshold-insensitive, so it simply does not pay that cost. Held at 0.5, HateMM deltas would be
  +0.0088 / −0.0005 / +0.0093 — still not 3/3, still a fail. Diagnostic only, no rescue.
- **The operator is nearly a no-op**: TRANS agrees with the plain inductive prediction at 0.5 on
  95.3-100 % of test items on every dataset.

### Disposition
The frozen verdict stands as **AMBIGUOUS** and is recorded as such. It is not upgraded and not
softened. But it should not be read as "worth another run", for two reasons stated together:
the frozen instrument did not test the intended operator, and the corrected instrument — swept
across its entire hyperparameter continuum on the only split where selection is legal — is negative
on all four datasets. **A corrected test-side re-run is not warranted**; it would spend a test read
on a mechanism whose legal-split evidence is negative at every operating point.

This is consistent with, and independent of, **F63** (multi-hop label propagation over the frozen
kNN graph, killed on all three datasets, monotone-negative in the diffusion coefficient). Two
mechanically different transductive operators — edge-wise label propagation and pool-density
class re-estimation — now both fail on this substrate. The likely common cause is on record: there
is **no measurable train/test covariate shift on any of the four datasets** (domain-classifier AUC
0.42-0.56, MMD p 0.17-0.96), so there is nothing for a pool-level correction to correct.

---

## Round-6 pilot ledger

| pilot | frozen verdict | mechanism reading | cost |
|---|---|---|---|
| R6-1 multi-layer readout fusion | **KILL** (0 of 2 datasets) | concatenating a second decoder layer is a dimensionality increase; the control captures nearly all of the apparent gain | 203 s GPU, ¥0 |
| R6-2 transductive pool refinement | **AMBIGUOUS** (1 of 4 datasets) | instrument defective; corrected instrument negative at every operating point on dev, all four datasets | ~8 min GPU, ¥0 |

Neither pilot produces a candidate. Total round API spend: **¥0.00 of ¥60.**
