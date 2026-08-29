# R6-1C — powered confirmation freeze, 2026-08-17

Frozen **before any seed in the confirmation range is run**. Zero API cost.

## Why this run exists

`idea-stage/R6_PILOT_RESULT_2026-08-17.md` recorded pilot R6-1 as **KILL** under its frozen 3-seed
rule. The measurement-protocol audit (`idea-stage/r6_audit/results.json`, 180 runs, 30 seeds,
pre-registered as a variance diagnostic) then established that **the instrument that rendered that
KILL cannot resolve the effect it was judging**:

| dataset | protocol | pair | 30-seed mean | std | n* for MC SE ≤ 0.0025 | P(3-seed rule fires GO) |
|---|---|---|---|---|---|---|
| HateMM | P1 | CAT−A0 | +0.0019 | 0.0067 | 7.1 | 0.129 |
| HateMM | P1 | CAT−RANDCAT | +0.0071 | 0.0093 | 13.9 | 0.368 |
| HateMM | P1 | RANDCAT−A0 | −0.0052 | 0.0130 | 27.0 | 0.035 |
| HateMM | P2 | CAT−A0 | +0.0072 | 0.0156 | 38.7 | 0.325 |
| HateMM | P2 | CAT−RANDCAT | +0.0068 | 0.0122 | 23.8 | 0.440 |
| MHC_zh | P1 | CAT−A0 | **+0.0145** (t=3.78, p=0.0007) | 0.0211 | 71.0 | 0.435 |
| MHC_zh | P1 | CAT−RANDCAT | +0.0167 | 0.0189 | 57.0 | 0.499 |
| MHC_zh | P1 | RANDCAT−A0 | −0.0022 | 0.0142 | 32.4 | 0.085 |
| MHC_zh | P2 | CAT−A0 | +0.0108 | 0.0156 | 39.0 | 0.378 |
| MHC_zh | P2 | CAT−RANDCAT | +0.0137 | 0.0195 | 61.0 | 0.498 |

Three facts follow. (1) **CAT − A0 is positive in all four dataset × protocol cells** and
**CAT − RANDCAT is positive in all four**, while the dimension-matched random control **RANDCAT − A0
is flat or negative** in all four — so the ordering is not a dimensionality artifact, which is what
the 3-seed read had suggested. (2) The 3-seed rule needs **7 to 71 seeds** to reach half-bar
resolution and uses 3; it fires GO 12.9 % of the time on a below-bar HateMM effect and misses a
genuinely above-bar MHC_zh effect 56.5 % of the time. (3) A **protocol deviation** was found: the
freeze specified epoch selection by validation macro-F1, but `idea-stage/r6_readout/analyze.py`
inherited `scripts/rgcl_ablation_analyze.py::parse_run`, which selects on `(dev acc, dev roc)`.
That key is what produced the 0.0531 MHC_zh seed std; under it n* rises to 201-357.

**The R6-1 KILL is not withdrawn. It stands as the frozen verdict of an underpowered instrument.**
This run is a separate, properly powered, independently pre-registered confirmation.

## Independence guard
The audit consumed seeds **0-29**. This confirmation uses seeds **30-89** (60 fresh seeds, disjoint).
The hypothesis, arms, protocols, quantities and decision rule below are fixed before any seed in
30-89 is executed.

## Design
- **Datasets**: HateMM (LoRA-curric encoder) and MHC_zh (LoRA encoder). The only two with `ro_`
  caches; both are required to be reported.
- **Seeds**: 30..89, 60 per arm per dataset.
- **Arms** (4): `A0` = L28; `CAT` = concat(l2norm L28, l2norm L24); `RANDA` = concat(l2norm L28,
  l2norm(L28·R_A)); `RANDB` = concat(l2norm L28, l2norm(L28·R_B)). R_A and R_B are **two independent**
  Gaussian 3584×3584 draws, `default_rng(20260817001)` and `default_rng(20260817002)`, sha-pinned.
  Two draws, not one, because a single random matrix estimates the redundant-view null from a
  sample of one — the reviewer's objection, adopted.
- **Grid**: 2 × 4 × 60 = **480 head runs**, ~80 minutes at the measured 9-10 s per run.
- **Read-out protocols**, both computed from the same runs: **P1 (primary)** = epoch selected on val
  by validation macro-F1, ties to earliest epoch, epochs ≥ warmup 5, test macro-F1 at threshold 0.5.
  **P2 (corroboration)** = final epoch (29), threshold 0.5. The `(dev acc, dev roc)` key is **not**
  used and is reported only if it comes free.
- Hyperparameters otherwise identical to `idea-stage/r6_readout/run_arms.sh`.

## Quantities
Per dataset per protocol, paired seed-wise over the 60 seeds:
`CAT − A0`, `CAT − RAND` where `RAND` is the per-seed mean of RANDA and RANDB, `RANDA − A0`,
`RANDB − A0`, and `RANDA − RANDB`. For each: mean, std, MC standard error, and a **paired bootstrap
95 % CI over seeds** (20 000 resamples).

## Frozen decision rule
Primary protocol is **P1**. A dataset **passes** iff:
`mean(CAT − A0) ≥ +0.005` with its paired-bootstrap 95 % CI excluding 0, **and**
`mean(CAT − RAND) ≥ +0.005` with its paired-bootstrap 95 % CI excluding 0.

- **CONFIRMED-2DS** — both datasets pass under P1, and P2 agrees in sign on both.
- **CONFIRMED-1DS** — exactly one dataset passes under P1, P2 agrees in sign on it, and the other
  dataset shows `mean(CAT − A0) ≥ −0.002` under P1 (no material harm).
- **NOT CONFIRMED** — anything else.

`RANDA − RANDB` is a sanity quantity: it should be centred on zero. If |mean(RANDA − RANDB)| ≥ 0.005
the random-control construction is itself unstable and the run is declared **VOID**, not passed.

## Expected outcome, stated before the run
CONFIRMED-1DS (MHC_zh passes, HateMM does not reach +0.005 on CAT−A0 under P1). The 30-seed
estimates put MHC_zh CAT−A0 at +0.0145 and HateMM at +0.0019, and the confirmation is powered to
separate those.

## What a pass would and would not mean
It would mean: **on this substrate, the frozen final-layer readout is not the best frozen readout,
and a fixed global two-layer concatenation buys roughly one to one-and-a-half macro-F1 points on
MHC_zh over the final layer alone, beating a dimension-matched random control.** It would be a
component-level gain of the size the user's "incremental gains are acceptable" ruling admits, on
1-2 of 4 datasets, in a family (multi-layer probing) that is occupied in general machine learning
(`2605.10494` ICASSP 2026, `2601.09322`, `2507.17394` ACM MM 2025) and unoccupied in hateful
content. It would **not** by itself be a method paper, and it must not be reported as one.
