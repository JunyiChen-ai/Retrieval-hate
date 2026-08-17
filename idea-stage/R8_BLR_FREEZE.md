# R8-1 BLR — boundary-localised ranking objective: frozen design

Frozen and committed **before any seed in the range 200-229 is executed**. Zero API cost.
Code `idea-stage/r8_blr/blr.py` exists but has been run only in `--smoke` mode, which prints
wall-clock, step count, first/last training loss and a NaN flag, and **no metric of any arm**.

## 1. Why this run exists

Three diagnostics run today (`idea-stage/R8_DECOMP_MEMO.md`, raw in `idea-stage/r8_decomp/`,
5-fold CV over train+val, no test contact) establish:

- **D2** Averaging predicted probabilities over training epochs 20-29 beats the deployed
  val-selected-epoch read-out on ROC in 4/4 datasets (+0.0171 / +0.0059 / +0.0150 / +0.0032) and on
  macro-F1 by essentially nothing (+0.0005 / +0.0024 / −0.0040 / +0.0010).
- **D3** Evaluating both arms at their own oracle thresholds, that ROC advantage is still worth only
  **+0.0051 / +0.0033 / +0.0008 / +0.0001** macro-F1 — so the failure is not the operating point.
  The operating point itself carries only +0.0025 to +0.0119 of headroom on a properly sized pool.

macro-F1 at a fixed threshold is a function of the ordering **local to that threshold**. A global
AUC surrogate spreads its capacity over all positive x negative pairs, most of which are already
far apart and contribute nothing to the metric. That is the most economical explanation both for D2
and for the project's banked-but-never-converted result that a pairwise/AUC objective beats BCE on
**test ROC in 4/4 cells** (+0.0080 / +0.0167 / +0.0115 / +0.0020, `IDEA_REPORT` §8.8) with its
macro-F1 effect never measured.

This run tests the implied prediction — that ranking pressure concentrated on boundary-local pairs
converts where global ranking pressure does not — and simultaneously closes the measurement the
project owes on the banked pairwise objective.

**Prior stated before the run.** The external reviewer (gpt-5.6-sol, xhigh) scored this candidate
4.5/10 composite, called it a falsification run rather than a paper, and predicted fewer than two
datasets above +0.005 with the anchored pointwise term doing most of the work. The executor's own
expectation is **KILL**. The run proceeds because it costs ~35 minutes of an idle local GPU, it
is the only remaining live hypothesis on the round's search axis, and its secondary quantities are
owed regardless of the verdict.

## 2. Design

- **Datasets and encoders** (the deployed best single encoder per dataset, unchanged):
  HateMM / `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`; MHC-EN and MHC-ZH /
  `Qwen2.5-VL-7B-Instruct-LoRA_HF`; ImpliHateVid / `Qwen2.5-VL-7B-Instruct_HF`.
- **Seeds**: 200-229, 30 per arm per dataset. Disjoint from every previously consumed range
  (0-29 protocol audit, 30-89 R6-1C confirmation, 100-129 R7).
- **Head / optimiser**: `idea-stage/r4_harness.py::Head`, the deployed `classifier_hateClipper`
  geometry (map 1024, proj 1024, 3 layers, align fusion, dropout 0.2/0.4/0.1), AdamW lr 1e-4,
  batch 64, 30 epochs, warm-up 5. Identical across all five arms.
- **Fixed a priori, not swept**: `NPAIR = 1024` sampled pairs per optimisation step,
  retained fraction `Q = 0.25`, anchor `tau = 0` (i.e. probability 0.5, the reported threshold).

### Arms (five), differing only in the training objective

With `s_p`, `s_n` the logits of a sampled positive and negative, and
`anchor = 0.5 * (mean softplus(-s_p) + mean softplus(s_n))` (the fixed threshold treated as a
virtual item every positive must outrank and every negative must be outranked by; at `tau = 0` this
term is exactly balanced BCE):

| arm | objective | role |
|---|---|---|
| `A0` | BCE over all training items, ordinary shuffled minibatches | deployed baseline |
| `BALBCE` | `anchor` alone | isolates implicit class balancing (`2512.01766`, `2607.09832`) |
| `PAIRG` | `mean softplus(-(s_p - s_n)) + anchor` | global pairwise; the banked recipe, anchored |
| `PAIRL` | `mean of the top-Q hardest softplus(-(s_p - s_n)) + anchor` | **the candidate** (two-way partial-AUC surrogate) |
| `RANDL` | `mean of a uniformly random Q subset + anchor` | matched pair count, no localisation |

`PAIRL` vs `PAIRG` isolates localisation; `PAIRL` vs `RANDL` isolates localisation from having
fewer, noisier pairs; `PAIRG` vs `BALBCE` isolates the ranking term from the balancing; `BALBCE`
vs `A0` isolates the balancing from everything else.

Grid: 4 datasets x 5 arms x 30 seeds = **600 head runs**, ~50 minutes at the smoke-measured
1.0-7.4 s per run. Single background job, single submission.

## 3. Read-out

Both computed from the same runs.
- **P1 (primary)**: epoch = `argmax_{e >= 5}` val macro-F1 at threshold 0.5, ties to the earliest
  epoch; report test macro-F1 at threshold 0.5.
- **P2 (corroboration)**: last epoch (29), test macro-F1 at threshold 0.5.

Test ROC at both epochs is recorded as a secondary quantity. Test labels are read only for the
final metric; no threshold, epoch rule, arm, hyper-parameter or dataset is selected on them.

Paired over the 30 seeds; every reported difference carries a **paired bootstrap 95 % CI over
seeds**, 20 000 resamples, `default_rng(20260817)`.

## 4. Frozen decision rule

A dataset **passes** under P1 iff **all four** hold:

1. `mean(PAIRL − A0) >= +0.005` and its 95 % CI excludes 0;
2. `mean(PAIRL − PAIRG) >= +0.005` and its 95 % CI excludes 0;
3. `mean(PAIRL − BALBCE) >= +0.005` and its 95 % CI excludes 0;
4. `mean(PAIRL − RANDL) > 0` (sign only).

Verdict:
- **GO** — at least **2 of 4** datasets pass under P1, and P2 agrees in sign on `PAIRL − A0` for
  every passing dataset.
- **WEAK** — exactly 1 dataset passes. Recorded, not promoted, no follow-up run authorised by this
  freeze.
- **KILL** — 0 datasets pass.

The verdict is rendered by `idea-stage/r8_blr/analyze.py`, run **exactly once** on the complete
600-run grid. If any run fails or produces NaN the whole grid is declared VOID rather than
partially analysed.

## 5. Secondary quantities — reported, with no decision attached

Owed by the project regardless of the verdict, and not permitted to change the verdict:

- `mean(PAIRG − A0)` and `mean(BALBCE − A0)` in **test macro-F1**, P1 and P2, per dataset, with CIs
  — the never-measured macro-F1 effect of the banked pairwise objective, and how much of it is
  merely class balancing.
- The same two contrasts in **test ROC**, to check whether the banked "+0.008 to +0.017 ROC in 4/4
  cells" reproduces at 30 seeds with the anchored pointwise term.

## 6. Integrity

Four red lines: (1) no test-label tuning — test labels enter only the final metric; (2) this
document is committed before the first frozen seed runs; (3) blindness — the only prior execution
of `blr.py` was `--smoke`, which emits no arm metric; (4) the frozen grid is submitted exactly once.

## 7. What a pass would and would not mean

A pass would mean: **on frozen features with 549-1283 training items, concentrating a pairwise
ranking objective on boundary-local pairs converts into fixed-threshold macro-F1 where a global
ranking objective does not** — with the global objective, the balancing, and the pair-count
reduction all separately controlled.

It would **not** be a method contribution on its own. The mechanism is two-way partial AUC, which
is occupied in general machine learning (`2012.03173`, LibAUC pAUC), and ranking-plus-pointwise
coupling is occupied by `2208.06164` (JRC, KDD 2023) and `2211.01494`. The external reviewer's
novelty verdict, adopted in advance, is **known mechanism, new domain**. What would be new is the
*diagnosis* — that global ranking gains provably fail to convert here, measured at the oracle
threshold — and the diagnosis alone is not a method paper under the standing constraint.
