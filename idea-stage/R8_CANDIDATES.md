# Round 8 — candidate slate and adversarial review

**Search axis fixed by the principal:** downstream head architecture + training objective /
optimisation dynamics + view-combination mechanism. Rationale: the project's only three measured
positive effects all sit on this axis (pairwise objective 4/4 test-ROC wins; three-encoder ensemble
+1.3 to +5.3 macro-F1; L24 || L28 concatenation +1.85 on MHC-ZH, 60 seeds, CI excludes zero) and
the axis has never been searched systematically.

**Budget:** ≤ ¥10 API for the round. **Spent: ¥0.00** — every diagnostic, the literature recon and
the pilot ran on the local GPU and on free search.

## 1. Evidence assembled before scoring

- Three zero-cost diagnostics, `idea-stage/R8_DECOMP_MEMO.md` (5-fold CV over train+val, no test
  contact): the ensemble gain is encoder selection plus head estimation variance, not
  complementarity; trajectory averaging buys ROC in 4/4 and macro-F1 in ~0; and at the oracle
  threshold that ROC gain is still worth +0.0051 / +0.0033 / +0.0008 / +0.0001.
- Three parallel literature sweeps on the axis (frozen-feature objectives; multi-view combination;
  optimisation dynamics and small-validation selection). Every arXiv id was opened before use.

## 2. Slate, scored by the external reviewer (gpt-5.6-sol, xhigh, instructed to be hostile)

Composite = 0.3·Premise + 0.3·Novelty + 0.3·Gain + 0.1·Cost, each 0-10.

| # | candidate | P | N | G | C | comp | sharpest kill reason |
|---|---|---|---|---|---|---|---|
| C1 | **BLR** — boundary-localised ranking objective (two-way partial-AUC surrogate + threshold anchor) | 6 | 2 | 4 | 9 | **4.5** | falsification only; `2208.06164` already owns ranking-plus-pointwise coupling |
| C8 | FROFA — frozen-feature augmentation on the cached embeddings | 4 | 0 | 3 | 9 | 3.0 | direct application of `2403.10519`; a gain is still someone else's method |
| C9 | ALIGNRC — mutually aligned ranking + pointwise objective, no λ | 4 | 1 | 3 | 8 | 3.2 | `2208.06164` owns the compatibility construction |
| C12 | PAIRSAMP — pair count as effective sample size | 3 | 1 | 2 | 9 | 2.7 | D2: ROC +0.003…+0.017 with ~0 macro-F1; more dependent pairs create no information |
| C5 | LAYERATT — learned attention over decoder layers | 3 | 0 | 3 | 7 | 2.5 | `2601.09322` occupies attentive multilayer fusion over frozen backbones |
| C11 | DUALENS — dual-space ensembling of the head | 3 | 0 | 2 | 7 | 2.2 | `2206.10566` is the method; seed averaging cleared +0.005 on only 1/4 datasets |
| C7 | LOGADJ — logit adjustment / balanced-softmax retraining | 2 | 0 | 2 | 9 | 2.1 | `2607.09832` + `2007.07314` |
| C10 | OOBDR — bagged out-of-fold estimation of epoch *and* threshold, dev folded back into training | 2 | 1 | 2 | 6 | 2.1 | D3: dev-fitted thresholds are negative on 3/4; oracle headroom +0.0025…+0.0119; protocol engineering |
| C3 | QAT — quantile-anchored training | 1 | 1 | 1 | 9 | 1.8 | D3: prior matching is −0.0002…+0.0104 and fails the two-dataset bar |
| C6 | MIMO — implicit multi-output ensemble in one head | 1 | 0 | 1 | 8 | 1.4 | `2601.16936` (Jan 2026): implicit ensembles track a single model |
| C2 | XVC — cross-view co-regularisation on unlabelled inputs, single-view deployment | 1 | 0 | 1 | 6 | 1.2 | D1: cross-encoder averaging loses −0.0068 / −0.0427 / −0.0191; agreement cannot manufacture complementarity. Also `1905.11866` (finite-unlabelled impossibility) and the Balcan-Blum arithmetic: m_u ≈ 300, VCdim ≈ 10 → ε ≈ 0.18 |
| C4 | NCLV — negative-correlation training over views | 1 | 0 | 1 | 5 | 1.1 | `2301.11323` (NeurIPS 2023): jointly optimising an ensemble objective makes base learners collude to inflate apparent diversity, widening the generalisation gap. *Scope note: the "worse at small n" qualifier used in the review is an inference, not a claim of that paper's abstract, which is verified only for "a range of standard machine learning tasks and architectures".* |

**Reviewer's headline, verbatim:** *"No candidate clears the bar. Do not spend GPU on this slate.
C1 is the only hypothesis worth a cheap cached-feature falsification run, but it is not a defensible
method-paper contribution."*

Asked directly whether any objective-level or view-combination mechanism can plausibly move test
macro-F1 at threshold 0.5 by ≥ +0.005 on ≥ 2 datasets, the reviewer answered: *"No — there is
currently no evidence-backed objective-level or view-combination mechanism plausibly clearing
+.005 fixed-threshold macro-F1 on at least two datasets… This is not a mathematical impossibility:
a new objective could learn a genuinely better boundary rather than merely reorder or recenter
existing scores. Nothing here supplies such a mechanism."*

On C1's novelty: *"Insufficient. It is 'known two-way partial AUC mechanism plus a threshold
anchor, applied to a new domain.'"*

## 3. Disposition

One pilot authorised: **R8-1 BLR** (`idea-stage/R8_BLR_FREEZE.md`), run as a falsification with the
two control arms the reviewer named as indispensable (`PAIRG` matched for pair count and gradient
steps; `BALBCE` for the implicit class balancing), plus a pair-count control (`RANDL`) and the
secondary measurement the reviewer required — *"The existing pairwise+BCE objective must also have
its fixed-0.5 macro-F1 measured first; leaving that result unknown makes the entire motivation
incomplete."*

No second pilot. C2 was the only other candidate the literature called dormant, and both its
premise (D1) and its theory (finite-unlabelled impossibility, Balcan-Blum sample arithmetic) price
it below the noise floor; spending a second frozen run on it would be search without a premise.
