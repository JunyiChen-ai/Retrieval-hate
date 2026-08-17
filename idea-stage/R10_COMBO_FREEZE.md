# R10-COMBO freeze — best combination of the banked span × layer blocks

**Written and committed BEFORE any arm metric exists.** Round 10 follow-up to
`idea-stage/R10_TOKPOS_RESULT.md` (`d811796`), whose own freeze is `f182877`.
Budget ¥0 (no API), local RTX 5090, no SLURM on this machine.

Four hard red lines apply and are restated as operational commitments:

1. **No test-label tuning.** Every epoch rule, arm definition, PCA basis, threshold and
   winner rule is fixed in this document or fitted on `train` only. Test labels are read
   only by the final metric.
2. **The decision rule (§3) is frozen here, before the run.**
3. **No candidate metric was computed during design.** The arms were built (§2) and one
   1-epoch shape smoke test was run on arm `K1` at seed **999** (outside every frozen seed
   range) with its output inspected only for exit code, epoch-line count and dump-file
   shape — no metric value was read.
4. **Single submission**: `idea-stage/r10_combo/run_all.sh`, run once.

Date: 2026-08-17. Light ceremony (head-level, ~1 h GPU, cheap) per `CLAUDE.md` §实验流程 —
one freeze document, no external review.

---

## 0. What is already established (inputs, not questions)

From `R10_TOKPOS_RESULT.md`, all on this machine, same extraction pass, seeds 500–529 /
500–514, protocol P1:

| configuration | MHC-ZH P1 | HateMM P1 |
|---|---|---|
| `A0` — deployed: img L28, text = assistant-header readout L28 | 0.8075 | 0.8660 |
| `CAT` — img L28, text = `[A0_28 ‖ TXT_28]` (token axis) | 0.8151 (+0.0076) | 0.8761 (+0.0101) |
| `L24⊕L28` — img `[i28‖i24]`, text `[A0_28‖A0_24]` (layer axis) | 0.8130 | not run |
| `L24⊕L28` + token axis on text (leg 2 `C1`) | 0.8033 (−0.0097 vs `C0`) | not run |

So: each axis buys ≈ +0.01 alone; naively stacking them on the text stream costs −0.010.
The open question handed back by §4 of that result document is **why**, and **what the
best available combination of the banked blocks actually is**.

### 0.1 Substrate — nothing needs re-extracting

`data/CLIP_Embedding/<DS>/<split>_<BASE>-tp.pt` holds, for both datasets and both splits
sets, a dict `{ids, labels, spans, meta}` with
`spans[layer][span] -> [N, 3584]` for `layer ∈ {28, 24}` and
`span ∈ {A0, TXT, S1, S2, S3, S4, ALL}` — 14 pooled text-stream blocks per video, all from
**one** frozen causal Qwen2.5-VL-7B forward per video. The img stream comes from the banked
`-ro_L28` / `-ro_L24` caches (a different forward, held constant per arm width).

Bases: MHC_zh `Qwen2.5-VL-7B-Instruct-LoRA_HF`, HateMM
`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`. Split sizes MHC_zh 579/78/149 (test 45 pos),
HateMM 744/107/215 (test 86 pos).

**Consequence for PCA:** the train split has only 579 (MHC_zh) / 744 (HateMM) rows, so any
PCA fitted on train has rank ≤ 578 / 743. Projecting "down to A0's width" (3584) is
impossible; the frozen target dimension is **512**, below the rank limit of both datasets.

---

## 1. Question

Given the banked blocks, what is the best combination, and is any combination better than
the better of the two single-axis configurations already measured?

Notation, all row-L2-normed: `i28`, `i24` = img stream at L28/L24; `a28`, `a24` = span `A0`
at L28/L24; `t28`, `t24` = span `TXT` at L28/L24. `P512(·)` = PCA to 512 components,
**mean and basis fitted on the train split only**, mean-centred, no scaling, no whitening,
applied unchanged to dev_seen and test_seen, followed by row L2-norm.

---

## 2. Arms (frozen; built by `idea-stage/r10_combo/build_combo.py`)

Four controls, six candidates. `--model` tag = `R10CB-<arm>`.

| arm | role | img stream | text stream | text dim |
|---|---|---|---|---|
| **A0** | control — deployed | `i28` | `a28` | 3584 |
| **LL** | control — layer axis (= R10 leg-2 `C0`) | `[i28‖i24]` | `[a28‖a24]` | 7168 |
| **CAT** | control — token axis (= R10 leg-1 `CAT`) | `i28` | `[a28‖t28]` | 7168 |
| **PC0** | control — PCA family width/compression control | `i28` | `P512(a28)` | 512 |
| **K1** | candidate — low-rank fusion of BOTH axes | `[i28‖i24]` | `P512([a28‖a24‖t28‖t24])` | 512 |
| **K2** | candidate — low-rank fusion of the token axis | `i28` | `P512([a28‖t28])` | 512 |
| **K3** | candidate — layer axis on **img only** + token axis on text | `[i28‖i24]` | `[a28‖t28]` | 7168 |
| **K4** | candidate — layer × span cross | `i28` | `[a28‖t24]` | 7168 |
| **K5** | candidate — additive fusion, zero extra width | `i28` | `n(a28 + t28)` | 3584 |
| **K6** | candidate — all 14 blocks, compressed | `[i28‖i24]` | `P512(` all 7 spans × 2 layers `)` | 512 |

Rationale, one line each (written before any number):

- **K1** tests whether R10 leg 2's failure was **width** (14336-d text on 579 train rows)
  rather than redundancy: the same four blocks, compressed to 512.
- **K2** tests whether the `CAT` gain survives a 14× narrower text stream — the cheapest
  possible deployment of the token axis.
- **K3** is the one obvious combination R10 never tried: leg 2 stacked the token axis onto a
  text stream that **already carried L24**; K3 uses the layer axis on the **img** stream
  only, where it may still be independent of the token axis.
- **K4** asks whether the transcript span is better read at L24 than at L28, at `CAT`'s
  exact width.
- **K5** is the parameter-free version of `CAT` (a linear head over `[a‖b]` can represent
  `W₁a + W₂b`; the sum forces `W(a+b)`), i.e. the cheapest implementation if it holds.
- **K6** is the maximal-information candidate: if the extra signal really is one small
  shared pool, throwing every extracted block at a 512-d bottleneck will not beat `CAT`.

**Belts, all run at build time and all passed before this freeze was committed** (they
compute no metric): `R10CB-A0`, `R10CB-CAT` and `R10CB-LL` must be bit-identical to R10's
`R10TP-A0`, `R10TP-CAT` and `R10L2-C0` caches (max abs diff < 1e-5). Result: text streams
exactly 0.0, img 1.19e-07 (float round-trip). Build metadata with per-arm sha256 in
`idea-stage/r10_combo/build_meta_{MHC_zh,HateMM}.json`.

---

## 3. Run protocol and decision rule (frozen — this is the judgement)

### 3.1 Run

`idea-stage/r10_combo/run_combo_grid.sh` is a fork of `idea-stage/reaudit/run_grid.sh`
whose **only** functional change is the addition of `--dump_head_scores` (the R7-2
read-out-only flag: it writes per-item head logits after the metric is computed, feeds
nothing back into training and draws no RNG). Verify with
`diff idea-stage/reaudit/run_grid.sh idea-stage/r10_combo/run_combo_grid.sh` — one added
line plus the header comment. All hyperparameters therefore remain byte-identical to
`r6_confirm/run_confirm.sh` → `r6_readout/run_arms.sh`.

- **MHC_zh, 30 seeds 600–629** — primary.
- **HateMM, 15 seeds 600–614** — run **unconditionally in the same submission**, not as a
  conditional leg. Deviation from R10's leg structure, taken because the whole HateMM grid
  costs ≈ 18 min here: running it up front removes any possibility that the second dataset
  is chosen after seeing the first, and lets the "at least one dataset passes, the other is
  harmless" clause below be evaluated symmetrically.
- Both seed ranges are disjoint from every consumed range in this project (30–89, 100–129,
  200–229, 300–329, 400–429, 500–529, 41000–41029).
- Read-out: `idea-stage/reaudit/analyze_grid.py`, unchanged. P1 = epoch
  `argmax_{e≥5}` dev macro-F1 (primary); P2 = epoch 29 (corroboration); test macro-F1 @ 0.5;
  paired bootstrap B = 20000, seed 20260817. Run once.

Note in advance: `analyze_grid.py` prints its own aggregate verdict, a conjunction over
*every* listed contrast. With 19 contrasts listed (including deliberately-negative ones)
that aggregate is meaningless here, exactly as in R10 leg 1. The per-contrast numbers are
what this freeze uses.

### 3.2 Decision rule

For candidate `K` and dataset `D`, all under **P1**, seed-paired:

- `REF(D)` = whichever of `LL`, `CAT` has the higher P1 mean test macro-F1 on `D`.

**`K` STANDS** iff there exists a dataset `D` with **all** of:

1. `mean(K − LL) ≥ +0.005` and its paired-bootstrap 95 % CI excludes 0;
2. `mean(K − CAT) ≥ +0.005` and its paired-bootstrap 95 % CI excludes 0;
3. P2 agrees in sign with P1 on both (1) and (2);
4. **PCA-family clause** — if `K ∈ {K1, K2, K6}`, additionally
   `mean(K − PC0) ≥ +0.005` with its 95 % CI excluding 0. (`PC0` is the matched control for
   "PCA compression by itself helps on 579 rows", the analogue of R10's `RAND` clause.
   Declared limitation: `PC0` compresses 3584→512 while `K1`/`K6` compress 14336/50176→512,
   so the control is not width-matched, only family-matched.)

**and** on the other dataset `D′`: `mean(K − REF(D′)) ≥ −0.002` (harmlessness).

**Selection-rule-bound demotion (REAUDIT_NCA lesson).** If `K` satisfies everything above
but its **dev-side** paired contrast against `REF(D)` — dev macro-F1 at the P1-selected
epoch, `idea-stage/r10_combo/analyze_dev_panel.py` — is negative with its 95 % CI excluding
zero, then `K` is reported as **AMBIGUOUS / selection-rule-bound**, not STANDS. The dev
panel, P2 and the dev−test gap at the selected epoch are reported for every arm regardless.

**AMBIGUOUS**: some candidate reaches `mean ≥ +0.005` against both references but fails a
CI, the P2-sign clause, the PCA clause, the harmlessness clause, or is demoted above.

**Winner, if more than one STANDS**: the largest `min over the two datasets` of
`mean(K − REF(D))`. Declared now so it cannot be chosen later.

**If nothing STANDS** — the pre-committed conclusion is: *the token-position axis and the
layer axis are substitutes on this substrate; adopt the cheapest implementation as the
default.* "Cheapest" is fixed here, before the numbers, by (a) fewest hidden-state layers
required, then (b) smallest total feature width. That is **`CAT`** (layer 28 only; 3584 img
+ 7168 text = 10752) over **`LL`** (layers 28 and 24; 7168 + 7168 = 14336) — unless a
candidate that is *statistically indistinguishable* from `REF` is strictly cheaper by the
same two-key ordering, in which case that candidate is recorded as the default instead.
"Statistically indistinguishable" = `mean(K − REF) ≥ −0.002` on **both** datasets. By that
ordering the cheapest arms in the grid are, in increasing cost:
`PC0`/`K2` (3584+512) < `K6`/`K1` (7168+512) < `K5` (3584+3584) < `CAT`/`K4` (3584+7168)
< `LL`/`K3` (7168+7168).

---

## 4. Diagnostics — where the redundancy comes from (NO verdict power)

Neither diagnostic can pass, fail, revive or kill any arm. Both are reported in
`idea-stage/R10_COMBO_RESULT.md`.

### 4.1 Error-set overlap — `idea-stage/r10_combo/diag_errors.py`

For each seed, each arm's P1 epoch is selected from its **own dev** curve. The per-item
head logits dumped at that epoch give the exact prediction set behind the reported
macro-F1; a belt recomputes macro-F1 from the dumped logits and **HALTs if it disagrees
with the trainlog by more than 1e-4**. Reported for the pairs among `{A0, LL, CAT, K3, K5}`:
seed-mean Jaccard of the test error sets, against an independence null (two random subsets
of the same observed sizes drawn from the test split, 2000 draws per seed), and the
observed/null ratio.

### 4.2 Representation redundancy — `idea-stage/r10_combo/diag_repr.py`

**Train split only, no labels.** Blocks `a28, a24, t28, t24`.

1. Linear CKA and mean cosine for every block pair.
2. 5-fold **out-of-fold** kernel-ridge R² between blocks (`n < d`, so an in-sample linear
   map fits exactly and in-sample partialling is degenerate — hence out-of-fold). λ chosen
   from a fixed grid by the same 5-fold CV, train only.
3. The quantity the "same pool of signal" claim is actually about:
   `CKA( resid(t28 | a28), resid(a24 | a28) )` on out-of-fold residuals — how aligned is
   the part of the token axis `a28` cannot explain with the part of the layer axis `a28`
   cannot explain. Reported with a random-Gaussian floor and with each residual's energy
   fraction.

Interpretation is written after the numbers and is explicitly non-decisional.

---

## 5. What a STANDS verdict would and would not license

**Would**: "on `<dataset>`, `<arm>` beats both single-axis configurations by ≥ 0.005 test
macro-F1 under P1 with paired-bootstrap CIs excluding zero, and is not harmful on the other
dataset" — a head-level, same-machine, same-extraction-pass statement.

**Would not**: any absolute-number comparison to the project ledger (those numbers were
extracted on A100; per D1 of R10 only within-table contrasts are results here); any claim
about MHC-EN or ImpliHateVid; any claim about layers other than 28/24 or spans other than
the seven banked; any claim that the effect survives a different head, fusion mode or loss.

## 6. Artefacts

| what | where |
|---|---|
| this freeze | `idea-stage/R10_COMBO_FREEZE.md` |
| arm builder | `idea-stage/r10_combo/build_combo.py` (+ `build_meta_*.json`) |
| grid runner (fork, 1 added line) | `idea-stage/r10_combo/run_combo_grid.sh` |
| single submission | `idea-stage/r10_combo/run_all.sh` |
| judgement read-out | `idea-stage/reaudit/analyze_grid.py` (unchanged) → `zh_grid.json`, `hm_grid.json` |
| dev/epoch panel | `idea-stage/r10_combo/analyze_dev_panel.py` → `*_devpanel.json` |
| diagnostics | `diag_errors.py` → `*_errors.json`; `diag_repr.py` → `*_repr.json` |
| logs | `logging/runs/r10_combo/{run.log,run.pid,zh/,hm/}` |
| result | `idea-stage/R10_COMBO_RESULT.md` |
