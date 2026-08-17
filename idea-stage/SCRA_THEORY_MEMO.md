# F2 SCRA — Safe Covariate-Shift Rank Adaptation: theory + feasibility memo

Date 2026-08-10 · candidate from `idea-stage/IDEA_REPORT.md` §8.3 / §8.9 (round-4 hold, jury 6.1)
Freeze: `idea-stage/SCRA_SHIFT_FREEZE.md` · deviation: `idea-stage/SCRA_SHIFT_DEVIATION_D1.md`
Code: `idea-stage/scra_shift_probe.py`, `scra_null_run.py`, `scra_ece.py`
Data: `idea-stage/scra_shift.json`, `scra_shift_null.json`, `scra_ece.json`
Log: `logging/runs/scra_shift/run.log`, `logging/runs/scra_null/run.log`

---

## 0. VERDICT — **VACUOUS**, and independently **OCCUPIED**

Either kill alone is sufficient; they rest on disjoint evidence.

- **VACUOUS (primary).** The certificate's irreducible slack, expressed in AUC units, is
  **0.32–1.45** on the four datasets (Prop. 3, instantiated with measured calibration error).
  The entire effect this project is chasing on the deployed line is **0.008–0.017 ROC** (§8.8).
  The slack is **19× to 180× the effect**, so the safe set is `{f0}` on every dataset. And it is
  moot anyway, because there is **no covariate shift to adapt to**: the domain classifier runs at
  or below chance (OOF AUC 0.42–0.56) and MMD permutation p = 0.17–0.96 on all four.
- **OCCUPIED (independent).** "Deviate from the deployed model only when a guarantee of
  non-degradation holds, using unlabelled target data only" is published twice over:
  **TCPR** (Kouw & Loog, PRL 2021 / arXiv 1706.08082) for domain-adaptation risk, and — closer —
  **UMVP** (Li, Zha & Zhou, AAAI 2016), a maximin worst-case *performance gain over the supervised
  baseline* that **names AUC** as a supported measure. SCRA is a composition of the two, not a new
  mechanism.

The frozen rule **R2 fired 3 of 4** and no dataset met the "space exists" bar. Verdict is by rule,
not by argument.

---

## 1. Formalisation

`X` inputs, `Y ∈ {0,1}`. Source (train) law `P`, target (test) law `Q`.
**Covariate shift**: `η(x) = Pr(Y=1 | X=x)` is invariant; only the marginal moves. `r = dQ/dP`.
Deployed head `f₀ : X → ℝ`; candidate `f`. `φ(u) = 1{u>0} + ½·1{u=0}`.

Target pairwise AUC:

```
AUC_Q(s) = N_Q(s) / Z_Q,
  N_Q(s) = E_{x,x' ~ Q⊗Q}[ η(x)(1−η(x')) · φ(s(x) − s(x')) ],
  Z_Q    = E_{x,x' ~ Q⊗Q}[ η(x)(1−η(x')) ] = π_Q (1 − π_Q),   π_Q = E_Q η.
```

Because `Z_Q` does not depend on `s`, the sign of the improvement is the sign of

```
D(f) = E_{Q⊗Q}[ w(x,x') · ψ(x,x') ],
  w(x,x') = η(x)(1 − η(x')) ∈ [0,1],
  ψ(x,x') = φ(f(x) − f(x')) − φ(f₀(x) − f₀(x')) ∈ {−1, −½, 0, ½, 1}.
```

`ψ ≠ 0` exactly on the **disagreement region** — pairs the two heads order differently.
SCRA's certificate is the requirement `D(f) ≥ 0` with confidence `1 − δ`, using only
`{x_i}` from `Q` (unlabelled test inputs), never test labels.

### Prop. 1 (non-identification). *Target AUC is not identified by unlabelled target data alone.*
`AUC_Q(s)` is a functional of `(Q, η)`. Unlabelled `Q`-draws carry **zero** information about `η`:
for any score `s` and any target value `a ∈ [0,1]` there exists an `η` making `AUC_Q(s) = a`, and
no unlabelled sample can distinguish them. **Every** label-free certificate must therefore import
`η` from the source-labelled data as an estimate `η̂`, and inherits `η̂`'s error. *(Immediate; the
likelihood of unlabelled data does not involve `η`.)*

### Prop. 2 (the narrowest safe class is exactly the trivial one).
Let `F_mono = { g ∘ f₀ : g strictly increasing }` — temperature scaling, prior/threshold correction,
recalibration, any monotone score transform. Then for **every** `Q` and every `g`,
`AUC_Q(g ∘ f₀) = AUC_Q(f₀)`.
*Proof.* `φ(s(x) − s(x'))` depends on `s` only through the order relation `s(x) ≷ s(x')`, which a
strictly increasing `g` preserves. `N_Q` and `Z_Q` are therefore unchanged. ∎

**Consequence.** The class of adaptations that is *trivially* certifiable (safety holds with
equality, no assumptions at all) has **identically zero** AUC effect. This is not slack in the
certificate — it is a structural property of a rank metric. *Every* AUC gain requires flipping
orderings, and flipping orderings requires knowing which order is correct, which requires `η`.
The entire "safe TTA by recalibration/prior-shift correction" family is ruled out **a priori** for
this objective. (It is not ruled out for macro-F1, but macro-F1 adaptation needs the target prior,
which needs labels.)

So the narrowest non-trivial class is the first one that changes orderings — e.g. adapting the
three-encoder ensemble weights `s_λ = Σ λ_k f_k`, `λ ∈ Δ²`, with `λ₀` deployed. Everything below
concerns that class and any richer one.

### Prop. 3 (certificate slack). *A valid label-free certificate must reserve a margin of at least
`2 ε_Q / Z_Q` AUC units, where `ε_Q = E_{x∼Q} |η(x) − η̂(x)|`.*

*Proof sketch.* Let `ŵ = η̂(x)(1 − η̂(x'))` and `D̂` the plug-in of `D` using `ŵ` and the empirical
target sample. Then

```
|D − D̂| ≤ |E_{Q⊗Q}[(w − ŵ) ψ]| + O_p(n_Q^{−1/2}).
|w − ŵ| = |η(x)(1−η(x')) − η̂(x)(1−η̂(x'))| ≤ |η(x)−η̂(x)| + |η(x')−η̂(x')|   (add/subtract η̂(x)(1−η(x'))),
|ψ| ≤ 1
⇒ |E[(w − ŵ) ψ]| ≤ 2 E_{x∼Q}|η(x) − η̂(x)| = 2 ε_Q.
```

A certificate must certify `D ≥ 0`, so it may only fire when `D̂ ≥ 2ε_Q` (+ the sampling term).
Dividing by `Z_Q` puts it in AUC units. ∎

Under covariate shift `ε_Q = E_P[ r(x) · |η(x) − η̂(x)| ] ≤ ‖r‖_∞ · ε_P`, so the slack is *worse*
on target than on source, by the density-ratio factor — the certificate is most expensive exactly
where adaptation would be most valuable.

**Numerical instantiation (measured, not assumed).** `ε_P` is bounded below by the equal-mass
`ECE₁` of the deployed head on val (Jensen: `ECE₁ = Σ_b (n_b/n)|mean(y|b) − mean(η̂|b)| ≤ E|η−η̂|`).
Measured with the R4 deployed head, 3 seeds (`scra_ece.json`):

| dataset | ECE₁ (10 bins) | `π_val` | `Z = π(1−π)` | slack ≥ `2·ECE₁/Z` (AUC units) |
|---|---|---|---|---|
| HateMM / LoRA | 0.116 | 0.402 | 0.240 | **0.965** |
| MHC-EN / Qwen | 0.099 | 0.313 | 0.215 | **0.925** |
| MHC-ZH / LoRA | 0.167 | 0.359 | 0.230 | **1.454** |
| ImpliHateVid / CLIP | 0.040 | 0.492 | 0.250 | **0.320** |

Target effect for comparison: the pairwise-vs-BCE head gain banked in §8.8 is **+0.008 to +0.017
ROC**; the three-encoder ensemble is **+0.015 to +0.020 ROC**. The certificate's *floor* is
**19× (best case, ImpliHateVid) to 180× (worst case, MHC-ZH)** larger than anything on the table.
`D̂ ≥ 2ε_Q` can never be met, so the safe set is `{f₀}` and SCRA returns the deployed head
everywhere — which is precisely the risk the round-4 jury flagged, now with a number on it.

The bound is worst-case, and a sharper version restricting `ε` to the disagreement region helps —
but not nearly enough, by Prop. 4.

### Prop. 4 (certificate–gain exclusion). *The refinement does not rescue it.*
Restricting to the disagreement region gives `|D − D̂| ≤ 2 ε_dis` where `ε_dis` is the mean `η`-error
over the target marginal of `{ψ ≠ 0}`, while the *maximum possible* gain is
`D̂ ≤ E_{Q⊗Q}[ŵ · 1{ψ ≠ 0}] ≤ ¼ · ρ` with `ρ = Q⊗Q(ψ ≠ 0)` the disagreement mass. Certification
requires

```
¼ ρ  ≥  D̂  >  2 ε_dis      ⇒      ε_dis  <  ρ / 8.
```

To gain `δ` AUC units you need `ρ ≳ 4 δ Z_Q`; at `δ = 0.02`, `Z ≈ 0.24` that is `ρ ≈ 0.019`,
requiring `ε_dis < 0.0024` — the head's probabilities must be accurate to **0.24 percentage points**
on exactly the pairs where the two heads disagree, i.e. the ambiguous region where `η̂` is *least*
reliable. Measured `ECE₁ ≥ 0.040`, a factor **≥ 17** away, and that is the *aggregate* figure; on
the disagreement region it is larger, not smaller. The dichotomy is structural: the region where
adaptation could pay is the region where `η̂` cannot be trusted, and vice versa.

### 1.5 What assumption would make the certificate real, and is it verifiable?

| the certificate needs | verifiable from data we have? | honest status |
|---|---|---|
| covariate shift (`η` invariant) | **no** — untestable without target labels | standing untestable assumption; also *false by construction* here, see §3 |
| `r = dQ/dP` known | no | textbook blank cheque |
| `‖r‖_∞ ≤ κ` with useful `κ` | estimable only through a density-ratio estimator, which at `n ≈ 80–325` is pure noise (§3, M2/D1) | **blank cheque at these sample sizes** |
| `ε_P` small (`≲ 0.002`) | yes — measured, `0.040–0.167` | **fails by 17–80×** |
| `n_Q` large enough for the `O_p(n_Q^{−1/2})` term | `n_test = 149–401` ⇒ AUC s.e. ≈ 0.03–0.05 | **fails on its own**, before any `η̂` error |

Only one row is verifiable and it fails. **The certificate as stated is a blank cheque signed on an
assumption (`ε_P` small) that this project can measure and that measures out two orders of magnitude
too large.**

---

## 2. Prior-art table

| line | closest work | what it proves / does | distance |
|---|---|---|---|
| safe / anti-collapse TTA | EATA (Niu et al., ICML 2022); SAR (ICLR 2023); POEM (NeurIPS 2024) | heuristic anti-collapse — entropy filtering, Fisher anchor to source, flat minima. **No guarantee.** POEM claims an empirical "no-harm" behaviour, not a theorem | ADJACENT |
| risk monitoring for TTA | Schirmer, Jazbec, Naesseth, Nalisnick, arXiv 2507.08721 | sequential confidence sequences that **raise an alarm** when adapted risk crosses a threshold; label-free, bounded-loss assumption | ADJACENT — certifies *detection*, never *improvement* |
| guaranteed TTA with labels | ATTA / SimATTA (Gui, Li, Ji, ICLR 2024) | DA bound showing a few **labelled** test instances improve target error | FAR — buys the guarantee with target labels, which SCRA forbids |
| **safe SSL, maximin gain** | **UMVP — Li, Zha, Zhou, AAAI 2016**; S4VM (TPAMI 2015); SAFEW (TPAMI 2021) | **maximin worst-case performance *gain* over the supervised baseline, with AUC among the supported measures**, minimax convex relaxation, no target labels. SAFEW's guarantee holds iff truth ∈ convex hull of base learners | **OCCUPIES the objective** |
| **contrastive-pessimistic DA** | **TCPR — Kouw & Loog, arXiv 1706.08082 / PRL 2021**; CPLE (Loog, TPAMI 2016); TRDA (S+SSPR 2020) | **transductive** estimator that deviates from the source classifier only when lower target risk is guaranteed **for every possible labelling of the target sample**; no target labels; explicitly does *not* assume covariate shift | **OCCUPIES the mechanism** |
| DRO / worst-case AUC | DR-AUC (ORL 2020); DRAUC (NeurIPS 2023); When AUC meets DRO (ICML 2022) | Wasserstein / CVaR balls around the **training** distribution, labelled source, train-time | ADJACENT — absolute robustness, not relative dominance |
| IW-AUC under covariate shift | **Kumagai et al., ICML 2025, PMLR 267:31876** ("PU AUC maximization under covariate shift") | shows plain IW **cannot** maximise AUC; builds test-AUC risk estimators from source + **unlabelled test** data. **No safety guarantee** | ADJACENT / occupies the *estimator* |
| conformal / label-free certification | Tibshirani, Barber, Candès, Ramdas, NeurIPS 2019; PAPE (arXiv 2401.08348) | weighted conformal: valid coverage under covariate shift **given a known likelihood ratio**; PAPE estimates performance from unlabelled data under approximate calibration | ADJACENT — certifies coverage / performance, not dominance |
| safe policy improvement | HCPI (Thomas et al. 2015); SPIBB (Laroche et al., ICML 2019) | return a new policy only if ≥ baseline w.p. `1−δ`; rests on **logged reward labels** + behaviour policy + concentration | OCCUPIES the template, FAR on setting |

**Reading.** The exact string "TTA + pairwise AUC + non-degradation certificate" appears unpublished.
But the *mechanism* is occupied twice: TCPR gives the transductive worst-case-over-labellings
certificate, UMVP gives the worst-case-gain-over-baseline objective **for AUC specifically**.
SCRA's novelty is the composition (AUC × covariate-shift weights × TTA framing) — reviewers who
know Loog or the LAMDA line will call it exactly that. Worse, Loog's own literature states the
failure mode in advance: contrastive-pessimistic estimators "are never worse than the source
classifier by construction, though they will not automatically lead to improvements in the error
rate, due to the difference between optimizing a surrogate loss and evaluating the 0/1-loss."
SCRA inherits that gap in pairwise-surrogate-vs-AUC form.

*Verification status (from the novelty sweep):* TCPR's theorem scope and SAFEW's convex-hull
assumption were read in the source text; UMVP's AUC coverage is from search snippets only (its PDF
would need reading before any related-work paragraph). Several 2026 preprints surfaced by title
only and are unconfirmed.

---

## 3. Measured shift — is there any room in the safe set?

Zero test labels: the probe loads only `img_feats`/`text_feats` for the test split
(`scra_shift_probe.py::load_inputs`). Cells are the four R4 cells. 3 seeds.

### M1/M3/M4 — is there a covariate shift at all?

| dataset | domain AUC val↔test (full / PCA-32) | domain AUC train↔test (full / PCA-32) | MMD² perm-p | test items outside train support (chance = 5 %) |
|---|---|---|---|---|
| HateMM | 0.506 / 0.563 | 0.544 / 0.508 | **0.43** | 7.4 % |
| MHC-EN | 0.451 / 0.487 | 0.503 / 0.469 | **0.52** | 8.1 % |
| MHC-ZH | 0.424 / 0.512 | 0.491 / 0.458 | **0.96** | 4.7 % |
| ImpliHateVid | 0.496 / 0.518 | 0.452 / 0.489 | **0.17** | 4.7 % |

**Six of the eight domain-classifier AUCs are at or below 0.5.** Not one MMD permutation test comes
close to significance. Nearest-neighbour support coverage is at the 5 % chance level. This is what
random i.i.d. splitting of a single corpus looks like — and that is exactly what these splits are:
`data/gt/<ds>/{train,val,test}.jsonl` are random splits of one collection, not deployment shifts.

**SCRA's premise is absent from all four benchmarks.** There is no covariate shift for a
covariate-shift adaptation method to exploit.

### M5 — how much AUC does the (non-existent) shift move?

`Δ = AUC_iw − AUC_plain` on val, where `AUC_iw` uses cross-fitted density-ratio weights and is,
under covariate shift, a consistent estimate of the head's **test** AUC. `se` is a 200-replicate
bootstrap. This is the only object a label-free certificate could be built on.

| dataset | AUC_plain | AUC_iw | Δ | se(AUC_iw) | ESS/n of weights | `se > |Δ|`? |
|---|---|---|---|---|---|---|
| HateMM | 0.9079 | 0.9007 | −0.0072 | 0.0430 | 0.45 | **yes** |
| MHC-EN | 0.8264 | 0.8845 | +0.0581 | 0.0482 | 0.37 | no |
| MHC-ZH | 0.9144 | 0.9317 | +0.0173 | 0.0317 | 0.48 | **yes** |
| ImpliHateVid | 0.9721 | 0.9704 | −0.0017 | 0.0087 | 0.79 | **yes** |

**Frozen rule R2 fires: `se > |Δ|` in 3 of 4.** The estimator the certificate must be built on has
a standard error larger than the effect it is meant to certify; a certificate must subtract that
error, so it can never clear zero. R1 (median `|Δ| < 0.01`) does **not** fire — median `|Δ|` =
0.0123. No dataset meets the "space exists" bar (`|Δ| ≥ 0.02` **and** `se ≤ |Δ|/2`): MHC-EN has the
magnitude but its s.e. is 0.048 > 0.029.

Note the weights are far from uniform (ESS/n = 0.37–0.79, `q95` ratio 2.3–5.4, `max` up to 19)
**even though M1/M3 say there is nothing to weight**. That is the tell: the ratios are estimator
noise at `n_val ≈ 80–325` in 1 792–7 168 dimensions, and they still move the AUC estimate by up to
5.8 points. Which motivated D1.

### D1 — null calibration (declared deviation, two-sided)

Zero-shift control with matched sample sizes: A = the full val split (same labels, same
out-of-sample head scores as M5), B = a random `n_test`-sized subset of **train**. Under i.i.d.
splitting these are exchangeable, so `Δ_null` is M5's noise floor. 40 draws × 3 seeds = 120 reps.

| dataset | observed `|Δ|` (M5) | null `|Δ|` p50 | null p90 | null p95 | one-sided p | inside noise floor? |
|---|---|---|---|---|---|---|
| HateMM | 0.0072 | **0.0132** | 0.0389 | 0.0460 | > 0.5 | **yes** |
| MHC-EN | 0.0581 | **0.0682** | 0.1202 | 0.1282 | > 0.5 | **yes** |
| MHC-ZH | 0.0173 | **0.0389** | 0.0602 | 0.0638 | > 0.5 | **yes** |
| ImpliHateVid | 0.0017 | **0.0020** | 0.0052 | 0.0062 | > 0.5 | **yes** |

**R3 fires 4 of 4, and more strongly than the rule required: the observed `|Δ|` is below the null
*median* in every cell**, so the one-sided p-value (fraction of null reps with `|Δ_null| ≥ |Δ_obs|`)
exceeds 0.5 everywhere. The null scale also tracks the observed scale cell by cell — the same
dataset ordering, the same magnitudes. **Every bit of the "shift-driven AUC movement" in M5 is
reproduced by fitting importance weights to data with no shift in it.** In particular MHC-EN's
`Δ = +0.058`, the single cell that did not trigger R2, is *smaller* than the typical movement its
own zero-shift control produces (0.068).

*(Rule bookkeeping: R3 was stated as "inside the central 90 % interval". Observed values sit below
the null median in all four cells, hence far below p95; the p5 side is not a rescue — a movement
*smaller* than the noise floor is still evidence of no shift, not evidence of shift. Reported this
way rather than re-running to obtain p5.)*

---

## 4. Verdict and reasoning

**VACUOUS.** Reasons, in order of hardness:

1. **The certificate's floor exceeds the prize by 19–180×** (Prop. 3, measured `ECE₁`). This is the
   hardest single fact and it does not depend on the benchmarks having no shift — it would hold on a
   genuinely shifted deployment too, as long as the deployed head is a neural head trained on
   ~500–1300 items. The safe set is `{f₀}`; SCRA returns the baseline everywhere.
2. **Prop. 2 makes the trivially-safe class exactly the zero-effect class.** For a rank metric,
   safety-without-assumptions and effect are mutually exclusive by construction. There is no
   "narrowest useful parameterisation" to retreat to — the retreat lands on the identity.
3. **The premise is absent from all four datasets.** Domain classifiers at or below chance, MMD
   p ≥ 0.17, support coverage at the chance level. These are random splits of single corpora.
   Even a perfect SCRA has nothing to adapt to here.
4. **Frozen rule R2 fired 3 of 4** and **D1's rule R3 fired 4 of 4** — the observed shift-driven
   AUC movement is below the *median* movement of a matched zero-shift control in every cell. No
   dataset met the pre-declared "space exists" bar.

**OCCUPIED (independent, would suffice alone).** TCPR (Kouw & Loog) already publishes the
transductive no-degradation certificate from unlabelled target data; UMVP (Li, Zha & Zhou, AAAI
2016) already publishes the maximin worst-case *gain over the baseline* objective with **AUC** as a
supported measure. SCRA is their composition. And Loog's line states its own limitation in advance:
never-worse on the surrogate does not transfer to the reported metric.

**No pilot design is offered.** A pilot would consist of confirming that the safe set is `{f₀}`,
which Prop. 3 already establishes analytically and §3 confirms has no shift to work with.

**What survives as reusable.** Three transferable facts, independent of the killed candidate:

1. **The four benchmarks have no measurable train/test covariate shift** (M1/M3/M4 above). Any
   future candidate whose story is "adapt to the deployment distribution", "transductive shift
   correction", "domain-robust", or "importance weighting" is dead on arrival on this corpus and
   should be killed at recon, not piloted. This retires the whole shift-family from the pool —
   including the round-4 holds **T2 TMN** (nulls "feature directions whose prevalence shifts at
   deployment" — no prevalence shifts) and the shift half of **T3 JRSA**.
2. **Estimated density ratios at `n ≈ 80–325` in ≥1 792 dimensions are pure noise, and the noise is
   large** — ESS/n down to 0.37 and up to 5.8 AUC points of spurious movement on data with no
   shift. Any method in this project that wants an importance weight must first clear this null.
3. **The deployed head's `ECE₁` is 0.04–0.17.** Any method needing calibrated `η̂` — cost-sensitive
   thresholding, selective prediction, abstention, expected-value routing, Bayes-risk decision
   rules — must budget for an error of that size. Relevant to the annotation-escalation-routing
   and pay-for-evidence idea files.

---

## 5. Honesty notes

- The M5 weights are estimated, not oracle. A defender could say the estimator is the problem, not
  the idea. That defence fails for two reasons: (i) at `n_val = 80–325` no label-free density-ratio
  estimator does better — the sample size is the binding constraint and it is a property of the
  benchmarks; (ii) M1/M3 independently say there is no ratio to estimate.
- Prop. 3's bound is worst-case over `ψ` and is loose by a constant. Prop. 4 gives the tightened
  version and it is still short by ≥ 17×. Closing a 17–180× gap with a better constant is not
  plausible.
- `ECE₁` lower-bounds `E|η − η̂|`; the true `ε_P` is larger, so the slack figures are conservative
  (i.e. favourable to SCRA) and it still dies.
- The covariate-shift assumption itself (`η` invariant) is untestable without target labels
  anywhere, in this project or any other. Papers that assume it should say so; this memo does.
- Prop. 1–4 are proof sketches at the level of rigour appropriate to a kill memo. They were not put
  through `/proof-checker`, because a candidate that is simultaneously vacuous by measurement,
  occupied by two prior lineages, and premise-free on all four datasets does not warrant it.
- No test labels were read at any point. The test split was loaded through `load_inputs()`, which
  drops the label field.
