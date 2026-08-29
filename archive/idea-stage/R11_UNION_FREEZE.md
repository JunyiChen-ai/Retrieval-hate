# R11-UNION — freeze: can any legal mechanism buy part of the CAT ∪ LL fix-set?

Frozen before any candidate arm metric on the R11 seed range exists. One
deviation is filed alongside this freeze and must be read with it:
`idea-stage/R11_UNION_DEVIATION_D1.md`.

## 1. The question this pilot answers

R10-COMBO (`idea-stage/R10_COMBO_RESULT.md`, freeze `33580d4`, result `2200f24`)
left one concrete open item. Against the deployed readout `A0`:

| per seed, MHC-ZH / HateMM | |
|---|---|
| A0 errors fixed by `CAT` (token axis) | 5.9 / 4.7 |
| A0 errors fixed by `LL` (layer axis) | 6.5 / 2.9 |
| fixed by both | 3.0 / 1.9 |
| **fixed by either (the union)** | **9.3 / 5.7** |
| new errors introduced by `CAT` | 3.9 / 2.9 |
| new errors introduced by `LL` | 4.8 / 3.5 |

Fix-set Jaccard 0.325 / 0.341 — the two axes are only about a third the same
items. A combination that kept the union of the fixes without the breakage would
be −9.3 errors on MHC-ZH ≈ **+0.05 macro-F1**. Every feature-level combination
R10-COMBO tried (raw concatenation, PCA-512 of the union, layer-on-img ×
token-on-text, additive fusion, all 14 blocks compressed) landed at or below
`CAT`, because each axis also breaks ~4 items and the breakage does not cancel.

**Question: is there a legal mechanism that buys part of that union?**

Explicitly out of scope, and not attempted: per-item oracle selection (Law III),
cross-encoder ensembling (already explained away as encoder selection + seed
variance), anything that reads test labels.

## 2. Substrate, arms and caches

Head-level only, one machine (RTX 5090), one extraction pass. Everything comes
from caches already on disk; **no new extraction, no GPU encoder work, zero API**.

Reused verbatim from R10-COMBO (sha256 re-verified against
`idea-stage/r10_combo/build_meta_<DS>.json` at build time, HALT on mismatch):

| arm | img | text | text dim |
|---|---|---|---|
| `A0` | `i28` | `a28` | 3584 |
| `LL` | `[i28‖i24]` | `[a28‖a24]` | 7168 |
| `CAT` | `i28` | `[a28‖t28]` | 7168 |

One new cache, built by `idea-stage/r11_union/build_r11.py`:

| arm | img | text | text dim |
|---|---|---|---|
| `MC` | `i28` | `n(a28 − mean_train(a28))` | 3584 |

`MC` is the cheap follow-up R10-COMBO §5 named (separating mean-centring from
compression in the `PC0` observation). It is **recorded separately and takes no
part in the union judgement**.

### 2.1 Trained runs

Recipe byte-identical to `idea-stage/r10_combo/run_combo_grid.sh` (3-layer
HateClipper-align MLP, `--contrast_mode none` BCE-only rung, 30 epochs, lr 1e-4,
batch 64, warmup 5). Runner `idea-stage/r11_union/run_union_grid.sh` adds only
two optional CLI fields (anchor teacher path, λ); with the teacher field `-` the
command line is the R10-COMBO command line.

Per dataset, per seed: `A0`, `LL`, `CAT`, `MC`, `ANCA_l{01,03,10}`,
`ANCL_l{01,03,10}`, `LBL_l{01,03,10}` — 13 runs. Plus `CATB` = the `CAT` arm at
seed +50000, 1 run. 14 × (30 + 15) = **630 runs**, ~70 min at the measured
6.2–7.2 s/run.

Seeds: MHC-ZH **700–729**, HateMM **700–714**; `CATB` at **50700–50729** /
**50700–50714**. All disjoint from every consumed range (0–119, 30–89, 100–129,
200–229, 300–329, 400–429, 500–529, 600–629, 41000–41029).

## 3. The five candidate mechanisms

All are frozen here in full; nothing about them may change after this commit.

### M1 `AVG` — decision-level averaging, fixed weight
`z = 0.5·z_CAT + 0.5·z_LL`, each head at **its own** dev-selected epoch,
threshold `z ≥ 0` (= sigmoid ≥ 0.5, the deployed threshold). Nothing fitted.
This is the cheapest possible way to ask how much of the union survives
averaging.

### M2 `WAVG` — decision-level averaging, dev-fitted weight
`z = w·z_CAT + (1−w)·z_LL`, `w ∈ {0.00, 0.05, …, 1.00}`. `w` is chosen **per
dataset and per protocol** as the argmax of the mean-over-seeds **dev** macro-F1
(ties → the `w` nearer 0.5). Test is never consulted.

### M3 `SEL` — selective trust under disagreement
A global, dev-fitted reliability rule; not per-item oracle selection. For each
head `h ∈ {CAT, LL}`, pool over seeds and dev items the pair `(|z|, correct)`.
Bucket `|z|` into 3 buckets at the terciles of the pooled dev `|z|`. Bucket
accuracy `a_{h,b} = (n_correct + 1)/(n + 2)`, clipped to [0.51, 0.99]; weight
`w_{h,b} = log(a/(1−a))`. Combine
`S = sign(z_CAT)·w_{CAT,b} + sign(z_LL)·w_{LL,b}`, predict 1 iff `S > 0`, ties
broken by `z_CAT ≥ 0`. Bucket edges and weights are fitted on dev only and
applied unchanged to test.

### M4 `ANCA` — churn-anchored training, deployed-readout teacher
The R9 slate's never-run ANCHOR-TRAIN (composite 3.70). Train the `CAT` head with
an added term

    L = L_main + λ · BCEWithLogits( z_i , q_i )   over train items,

`q_i` = a **frozen out-of-fold** teacher probability for train item `i`. Teacher
= 5-fold stratified logistic regression (C=1.0, lbfgs, max_iter 5000, fold seed
20260818) on `[i28 ‖ a28]`, i.e. the deployed readout's own feature set. λ ∈
{0.1, 0.3, 1.0}, selected on mean-over-seeds **dev** macro-F1.

Mechanism under test: keep the items the deployed readout already gets right
(suppress churn) while the new token-axis features stay free to fix the ones it
gets wrong. R9 ANCHOR-INT died because repair and breakage sat on the same knob
at inference; whether a *training-time* anchor has the same disease is the point.

The teacher is out-of-fold precisely so it is not a restatement of the train
labels. Measured at build time: OOF macro-F1 0.766 (MHC-ZH) / 0.864 (HateMM),
mean |q − y| 0.309 / 0.275, and only 0.2 % / 0.3 % of items have `q` within 0.05
of their label.

### M5 `ANCL` — churn-anchored training, layer-axis teacher
Identical to M4 except the teacher is fitted on `[i28‖i24‖a28‖a24]` — the `LL`
arm's feature set. This is the union-targeted version: it distils layer-axis
knowledge into a head whose **input** is the token axis, so a single model can in
principle carry both. Teacher OOF macro-F1 0.810 (MHC-ZH) / 0.875 (HateMM).

### Controls
- `ECTL` = `0.5·z_CAT + 0.5·z_CATB`, two `CAT` heads differing only in seed.
  **Separates "averaging two heads helps" from "the union is real".** This is the
  R10-TOKPOS `RAND` precedent applied at the decision level.
- `LBL_l{01,03,10}` = M4/M5's loss with `q_i = y_i` (hard train labels), λ chosen
  on dev the same way. **Separates the teacher's soft knowledge from "more BCE".**
- `A0`, `LL` — the reference points; `MC` — the side observation.

## 4. Measurement protocol

Unchanged from the project standard. P1 (primary) = epoch `argmax_{e≥5}` dev
macro-F1, ties → earliest; test macro-F1 at threshold 0.5. P2 (corroboration) =
epoch 29. Paired bootstrap over seeds, B = 20000, rng seed 20260817.

Every arm's macro-F1 — trained and derived alike — is recomputed from the
per-item head logits dumped by `--dump_head_scores`, so both families go through
identical metric code. **Belt:** the recomputed test macro-F1 must match the
trainlog to 1e-4 for every trained run, else HALT.

**Belt already passed at design time:** with λ = 0 the new code path is an exact
no-op — arm `A0` at seed 600 reproduces the R10-COMBO trainlog metric lines *and*
the dumped per-item logits **byte for byte**.

REAUDIT_NCA clause: the **dev**-side paired contrast is computed and reported for
every arm and every contrast, alongside test. Any arm whose test gain comes with
a negative dev contrast whose CI excludes zero is flagged selection-rule-bound in
the result document.

## 5. Decision rule — frozen

Reference arm: **`CAT`**. Candidates: `AVG`, `WAVG`, `SEL`, `ANCA`, `ANCL`.

A candidate **STANDS** iff there is an ordering (D1, D2) of (MHC-ZH, HateMM) such
that, under **P1 on test**:

- **(a)** mean(arm − `CAT`) on D1 ≥ **+0.005** and the paired-bootstrap 95 % CI
  excludes zero on the positive side;
- **(b)** mean(arm − `CAT`) on D2 ≥ **−0.002**;
- **(c)** family control clause on D1 —
  - `AVG`/`WAVG`/`SEL`: mean(arm − `ECTL`) ≥ +0.005, CI excluding zero;
  - `ANCA`/`ANCL`: mean(arm − `LBL`) ≥ +0.005, CI excluding zero;
- **(d)** P2 agrees in sign on D1: mean(arm − `CAT`) > 0 under P2.

If more than one stands, the headline is the one with the larger D1 mean.

**If none stands**, the pre-committed conclusion is: *the union is not
purchasable by these mechanisms; `CAT` alone remains the entry, and the ~+0.05
headroom identified by R10-COMBO stays unclaimed.*

Applied mechanically by `idea-stage/r11_union/verdict.py`, which was written
before any R11 number existed.

## 6. Union accounting — reported either way, no verdict power

From the P1 test per-item logits, per seed, for every arm: the A0 error pool, the
union fix-set `{A0 errors that CAT or LL gets right}`, the fraction of that union
the arm retains, the number of A0-correct items the arm newly breaks, and the net
errors saved against A0. Read **after** the mechanical verdict is produced, and
reported as the answer to "what fraction of the union did each mechanism eat".

## 7. Scope limits, declared in advance

- MHC-ZH and HateMM only; layers 28 and 24; spans `A0`/`TXT` only; one head, one
  hyperparameter set, one fusion mode.
- Same-machine, same-extraction-pass. **No absolute number here is comparable to
  the project ledger** (those were A100); only within-table contrasts are results.
- `AVG`/`WAVG`/`SEL`/`ECTL` cost 2× head training at inference-equivalent
  parameter count; `ANCA`/`ANCL` cost 1× and are single-model at deployment. If
  both families stand, that difference is a deployment consideration, not part of
  the rule.
- +0.005 is ≈ 0.7 test items of 149 (MHC-ZH) and ≈ 1.1 of 215 (HateMM).
- Zero API calls, zero cost, local only.

## 8. Artefacts

| what | where |
|---|---|
| this freeze | `idea-stage/R11_UNION_FREEZE.md` |
| deviation D1 (blindness slip) | `idea-stage/R11_UNION_DEVIATION_D1.md` |
| cache + teacher builder | `idea-stage/r11_union/build_r11.py`, `build_meta_<DS>.json` |
| frozen teachers | `idea-stage/r11_union/teacher_<DS>_{A0,LL,LBL}.json` |
| anchor loss | `src/model/loss.py::compute_anchor_loss`, `src/run_rac.py` `--anchor_logits/--lambda_anchor` |
| grid runner | `idea-stage/r11_union/run_union_grid.sh` |
| single submission | `idea-stage/r11_union/run_all.sh` |
| read-out | `idea-stage/r11_union/analyze_union.py` → `{zh,hm}_union.json` |
| mechanical verdict | `idea-stage/r11_union/verdict.py` → `verdict.json` |
| logs | `logging/runs/r11_union/{run.log,run.pid,zh/,hm/}` |
