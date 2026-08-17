# Round-4 triage verdict

## Executive verdict

**Pilot two candidates, not three: F1 MDL first, then B1 JLR.** Do not spend the third slot merely because it exists.

The disk facts remove the former number-three candidate, T1 PRES, from methods-paper contention: it can run only on HateMM, precisely the dataset whose test split and CLIP-text channel are compromised. I1 IPPO is executable but similarly cannot escape one dataset, and its method/error hypothesis was designed after reading that same dataset's test decomposition. Both can answer local diagnostic questions; neither can currently carry a top-venue method claim.

F1 remains the only clear pilot because it runs across all four datasets and attacks the strongest measured signal. B1 is a conditional second: it is multi-dataset, shares much of F1's fold-training infrastructure, and has a strict comparison against ordinary pairwise loss and an identically sized ensemble. Its broader novelty is doubtful, so a merely positive result is still a KILL. B2 is the best reserve idea, but the proposed policy cone is not yet mathematically pinned to the deployed nonlinear Hadamard-fusion head; implementing it now would make a design choice after the gate was supposed to be frozen.

## Confirmation status

- **F1 MDL:** confirmatory-by-construction is **survivable for this disclosed pilot**, but the present four test sets cannot be described as untouched confirmation because their complementarity motivated the method. The design repair is to pre-register one mechanism and one grid now, require the cross-dataset rule below rather than a single favourable cell, and obtain a genuinely untouched external dataset or new held-out split before a paper-level claim. Current test results can falsify MDL and estimate whether an asset build is justified; they cannot alone establish unbiased generalization.
- **I1 IPPO:** this is more serious and **requires a specific design change for any methods claim**. Both the error target (NH false positives) and the EX/IM functional form came from ImpliHateVid test recon, and no second subtype-annotated dataset exists locally. A publishable confirmation needs a new untouched split or a second EX/IM/NH video dataset. On the current test set, even a pre-registered run is exploratory/diagnostic evidence with disclosure, not independent confirmation.
- **ID leakage rule for any ImpliHateVid diagnostic:** parse subtype into a train-only label table, then discard all string IDs before folds or tensors are built. Folds must come from a seeded permutation of row indices, never ID sorting or ID hashing. At inference the loader must expose only numeric features; subtype is joined back after prediction solely for reporting EX/IM/NH metrics. An assertion should fail if any ID or subtype field enters the model batch.

## Adversarial pass over the eight runnable candidates

### 1. F1 MDL — Monotone Disagreement Lattice

**Strongest reviewer objection.** This is monotone stacking with a hate-video case study, and the opportunity was discovered by inspecting the same test sets. That objection is **differentiable for a pilot but potentially fatal for a paper**: MDL must beat mean logit/probability ensembles, validation-weighted averaging, logistic stacking, and a matched MLP stacker across datasets, and it still needs untouched external confirmation and a novelty sweep over monotone ensemble aggregation.

**Most likely failure mode.** The lattice will match logistic stacking or the mean ensemble in ROC, overfit the tiny validation sets, and fail to convert its score into macro-F1. That result is informative: it would show that the observed complementarity is already captured by additive stacking and that the remaining macro-F1 gap is decision-rule/calibration, not an exploitable disagreement geometry. A gain against mean averaging but not the trained stackers is also a KILL.

**Single-dataset constraint.** No. It can use three encoders on HateMM/MHC-EN/MHC-ZH and two on ImpliHateVid. MHC-EN is the clean primary cell; HateMM must remain supporting evidence. Cross-dataset feasibility is the main reason it stays first.

### 2. B1 JLR — Jackknife Lower-Bound Rank Head

**Strongest reviewer objection.** Robust pairwise AUC, influence/stability regularization, and deep-ensemble lower confidence bounds are crowded, and the original two-stage wording did not specify how the jackknife variance remains coupled to the trained head. This is **differentiable for the pilot only after freezing the joint leave-one-block formulation below**; even a successful pilot requires a literature sweep before novelty can be claimed.

**Most likely failure mode.** The standard-deviation term will either collapse diversity among the five heads or add nothing beyond an identically sized BCE ensemble and an ordinary mean-margin pairwise objective. That is informative: it directly rejects the claim that unstable train-pair order, rather than ordinary model variance, is limiting the frozen-feature head. A small gain over a single head that disappears against the five-head BCE comparator is the expected weak-comparator failure and is a KILL.

**Single-dataset constraint.** No. The pilot can run the pre-declared best-encoder cell on each of the four datasets. This is weaker coverage than F1's full encoder grid but enough to test whether the stability mechanism replicates across data regimes.

### 3. B2 PCD — Policy-Cone Discriminant Head

**Strongest reviewer objection.** It is prompt engineering plus a cone constraint, and the policy-clause directions live in the raw text feature space while the deployed classifier makes its decision after learned image/text projections, Hadamard fusion, and a nonlinear MLP. An unconstrained visual residual can also bypass the cone. This objection is **differentiable in principle but fatal to an immediate frozen pilot**: the exact score function and where the constraint acts must be resolved before implementation, and the result must beat LP++/CLAP-like single-anchor and prompt-ensemble controls plus random clauses.

**Most likely failure mode.** The policy directions will be weak or encoder-dependent, random semantically rich sentences will regularize similarly, and the constrained head will underfit the nonlinear BCE baseline. That would inform whether policy sentences constitute a usable anchor at all, but it would not validate the cone mechanism. Conversely, a gain from adding anchor similarities without a cone-specific gain is also a KILL.

**Single-dataset constraint.** No. The same pre-registered policy atoms can be encoded for all four datasets, with frozen translations for ZH. Its problem is mechanism definition and novelty, not coverage.

### 4. I1 IPPO — Incomparable-Positive Partial-Order Head

**Strongest reviewer objection.** The loss is group-DRO or balanced pairwise AUC with EX and IM used as two positive groups, presented as an implicitness mechanism after the test error decomposition was already read. Combined with one-dataset evidence, this is **fatal to a top-venue methods claim in the current asset state**, though the loss remains a valid diagnostic.

**Most likely failure mode.** IPPO will reproduce balanced pairwise/group-DRO results: NH FPR may fall at a fixed operating point, but EX or IM recall will fall too and full ROC will not rise by one point. The random positive partition will perform similarly, showing that any effect comes from rebalancing rather than true implicitness structure. That is informative about the axis, but it leaves no method paper.

**Single-dataset constraint.** Yes, decisively. A positive result cannot show that an EX/IM partial order generalizes beyond ImpliHateVid, and the source paper already supplies the three-way labels that define the mechanism. In the current project IPPO is a **diagnostic wearing a method's clothes**. It should not consume one of the methods-pilot slots.

### 5. I3 CNV — Cross-Fitted Non-Hate Veto

**Strongest reviewer objection.** This is two-stage residual boosting or hard-negative mining: the base model defines its own difficult negatives, and a one-sided second head corrects them. The objection is **probably fatal to novelty** unless a cross-dataset, no-harm result separates CNV from matched residual boosting, focal loss, and hard-negative training.

**Most likely failure mode.** The veto will learn high base score rather than independent negative evidence, subtract from genuinely hateful examples under shift, and trade hate recall for a lower NH FPR without improving ROC. A matched residual booster will do the same or better. This is informative about whether the false-positive tail is learnably structured, but not about a novel mechanism.

**Single-dataset constraint.** Not computationally: the cross-fitted over-fire target can be defined on all four binary datasets. However, its distinctive EX/IM safety analysis exists only on ImpliHateVid; extending it elsewhere makes it a generic boosting method and removes the implicitness claim. It escapes the resource constraint only by weakening its novelty.

### 6. B3 NTC — Negative-Tail CVaR Rank Head

**Strongest reviewer objection.** CVaR, partial-AUC optimization, and hard-negative pairwise losses are established machinery; moderation supplies an application-specific operating point, not a new method. This is **fatal unless a prior-art search uncovers a genuinely new guarantee or formulation**, which the current candidate does not contain.

**Most likely failure mode.** It will reduce FPR near the chosen 90%-recall point while leaving full ROC flat or worse, and it will be sensitive to the tail fraction at these sample sizes. Ordinary pairwise AUC or hard-negative mining will match it. That is a useful loss-function benchmark but cannot satisfy the methods-only constraint.

**Single-dataset constraint.** No. It can run across all four datasets, but broad feasibility does not repair occupied novelty.

### 7. I2 SHC — Shared-Hate Cone Head

**Strongest reviewer objection.** This is a max-min/group-robust form of shrinkage LDA using known positive subtypes, and HatePrototypes already weakens the premise that EX and IM require special geometry. Together with one-dataset evidence, the objection is **fatal to a top-venue claim as currently framed**.

**Most likely failure mode.** The single common direction will be overly conservative, lose the nonlinear BCE head's ROC, and behave no better than a randomly partitioned positive cone. That null would be informative: it would show that subtype-specific directions are not the source of NH false positives. A win only over LDA but not the trained BCE head is a KILL under the project's weak-comparator lesson.

**Single-dataset constraint.** Yes. The true EX/IM cone can be formed only on ImpliHateVid. Random partitions elsewhere would test generic group robustness, not the proposed implicitness mechanism. One result cannot support the claimed method.

### 8. T1 PRES — Pool-Relative Evidence Sparsification

**Strongest reviewer objection.** It is embedding-space TF–IDF/top-k pooling evaluated only on a compromised split, and the supposedly transductive part may be unnecessary because a train-only background can do the same thing. With no second dataset, this is **fatal to the methods claim**, not merely a weakness to discuss.

**Most likely failure mode.** Pool-relative surprisal will select rare OCR errors, logos, or junk rather than label evidence; train-background and test-background variants will tie; learned attention or the trained fusion MLP will remain stronger; and any macro-F1 movement will lack a matching ROC gain. A large HateMM-only gain could still diagnose window dilution, but the transcript-shift and duplicate findings prevent it from establishing a general method.

**Single-dataset constraint.** Yes, decisively. HateClipSeg has no train/test structure, and no other dataset has OCR windows. HateMM is not merely the sole cell but the contaminated one. PRES is now a **diagnostic wearing a method's clothes**; it should be removed from the pilot queue rather than demoted one place.

## Re-ranking of the eight runnable candidates

Scores are post-feasibility judgments about methods-paper upside, not ease of execution.

| rank | candidate | score / 10 | triage verdict |
|---:|---|---:|---|
| 1 | **F1 MDL** | **6.8** | Only clear pilot; four-dataset mechanism test, with confirmation and generic-stacking risks explicitly bounded. |
| 2 | **B1 JLR** | **5.2** | Pilot conditionally, using the exact joint formulation and ensemble controls below; novelty remains unproven. |
| 3 | **B2 PCD** | **4.8** | Best reserve, but do not implement until the cone is mathematically attached to the deployed head and prompt/LP++ occupancy is checked. |
| 4 | **I1 IPPO** | **4.2** | Clean local diagnostic; single-dataset and confirmatory-by-construction facts kill the current method claim. |
| 5 | **I3 CNV** | **4.0** | Multi-dataset executable, but likely residual boosting; implicitness distinctiveness disappears outside ImpliHateVid. |
| 6 | **B3 NTC** | **3.8** | Broadly executable but occupied by CVaR/partial-AUC work; a pilot cannot repair novelty. |
| 7 | **I2 SHC** | **3.4** | Single-dataset shrinkage/group-LDA variant with a hostile trained comparator. |
| 8 | **T1 PRES** | **2.0** | Structurally one-dataset, and that dataset is compromised; remove from methods-pilot consideration. |

## Final pilot selection and frozen rules

### Run 1 — F1 MDL

**Scope frozen before implementation.** Run seeds 0/1/2 on HateMM, MHC-EN, MHC-ZH, and ImpliHateVid; use CLIP, frozen Qwen, and LoRA-Qwen where cached, and CLIP+Qwen on ImpliHateVid. Generate five stratified out-of-fold train logits per encoder. Fit one four-knot-per-axis monotone piecewise-linear lattice (knots at train-OOF empirical 0, 1/3, 2/3, and 1 quantiles) with pairwise logistic loss, BCE calibration, and the concordant-region identity penalty fixed at weight 1.0. No lattice-size or loss-weight grid is allowed. Select epochs on validation macro-F1 exactly as in the bare-head harness.

**Comparator frozen before implementation.** The comparator set is: validation-best single encoder, mean probability, mean logit, non-negative validation-weighted logit average, logistic stacker, and a two-layer ReLU MLP stacker with parameter count no smaller than the lattice. For each dataset, choose one comparator by highest mean validation ROC over the three seeds, breaking ties in the conservative order `MLP > logistic > weighted > mean-logit > mean-probability > single`; freeze that comparator before reading test metrics. Every comparator's test result is still reported.

**Exact decision quantities.** For dataset \(d\), let

`DeltaROC_d = mean_seed[test_ROC(MDL) - test_ROC(frozen_comparator_d)]`

and define `DeltaF1_d` analogously after each method/seed chooses its threshold by maximum validation macro-F1 (ties go to the threshold closest to 0.5). Let `MeanDeltaROC` and `MeanDeltaF1` be unweighted means over the four datasets.

**Exact null.** Run 200 deterministic null repetitions after the primary rules are frozen. In each split and hard-label stratum, independently permute every non-reference encoder's logit rows while leaving the validation-best reference encoder fixed; refit the identical lattice. This preserves each encoder's class-conditional score distribution and ROC while destroying item-level complementarity. Test labels are used only to construct/report this post-hoc null and never to choose the model. Let `Null95` be the 95th percentile of `max(0, MeanDeltaROC_null)`.

**GO, all clauses required.**

1. `DeltaROC_MHC-EN >= +0.010`.
2. `MeanDeltaROC >= +0.010` and `MeanDeltaROC >= 3 * Null95`.
3. At least three of four `DeltaROC_d` values are strictly positive, and none is below -0.005.
4. `MeanDeltaF1 >= +0.010`, with no dataset below -0.005.

**KILL.** Failure of any clause is a KILL for MDL as the round-4 mechanism. In particular, beating mean averaging but not the validation-selected trained stacker is a KILL; a macro-F1-only gain with ROC below the bar is a KILL; and an MHC-EN miss cannot be rescued by HateMM.

**Confirmation interpretation.** A GO licenses an untouched-external-dataset build and a novelty sweep, not a paper claim from these four tests. A KILL closes the proposed disagreement-lattice mechanism while leaving the descriptive encoder-complementarity finding intact.

### Run 2 — B1 JLR

Run only after F1 predictions and folds are complete so the same data-loading and fold assignments can be reused.

**Scope frozen before implementation.** Use seeds 0/1/2 and exactly four pre-declared cells: HateMM/LoRA-Qwen, MHC-EN/frozen-Qwen, MHC-ZH/LoRA-Qwen, and ImpliHateVid/CLIP. Use five seeded stratified folds. Maintain five heads; head \(k\) receives BCE and pairwise gradients only from items outside fold \(k\). For each sampled positive–negative pair, compute margins over the heads eligible for both items and optimize

`softplus(-(mean_eligible_margin - 1.0 * sd_eligible_margin)) + 0.1 * mean_eligible_BCE`.

At inference average the five logits. The coefficient 1.0, BCE weight 0.1, fold count, architecture, and sampling rule are fixed; there is no grid. Epoch selection remains validation macro-F1.

**Comparator frozen before implementation.** Compare with: (a) the ordinary single BCE head; (b) a five-head leave-one-fold-out BCE ensemble with identical inference averaging; (c) the same joint pairwise ensemble with the standard-deviation coefficient set to zero; and (d) a single head with ordinary pairwise-AUC loss plus 0.1 BCE. Per dataset, freeze the comparator having the highest mean validation ROC across seeds; ties prefer the higher-capacity five-head BCE ensemble, then coefficient-zero pairwise ensemble, then single pairwise, then single BCE. Report all test cells.

**Exact decision quantities.** Define `DeltaROC_d`, `DeltaF1_d`, `MeanDeltaROC`, and `MeanDeltaF1` exactly as for F1, relative to B1's frozen comparator and with validation-tuned thresholds.

**Exact null.** On MHC-EN run 20 pre-seeded null trainings. For the lower-confidence-bound term only, independently permute each eligible head's item scores within hard label before assembling cross-head margins; keep each head's BCE data, label marginals, architecture, and inference averaging unchanged. This destroys item-specific stability while preserving class-conditional member behavior. Let `Null95` be the 95th percentile of `max(0, DeltaROC_MHC-EN_null)`.

**GO, all clauses required.**

1. `DeltaROC_MHC-EN >= +0.010` and `DeltaROC_MHC-EN >= 3 * Null95`.
2. `MeanDeltaROC >= +0.010` across the four fixed cells.
3. At least three of four `DeltaROC_d` values are strictly positive, and none is below -0.005.
4. `MeanDeltaF1 >= +0.005`, and no dataset loses more than 0.005.

**KILL.** Anything short of all four clauses is a KILL. A gain over a single BCE head that vanishes against either five-head comparator is explicitly a KILL. A GO is only permission for a targeted novelty search; if robust-AUC or ensemble-LCB prior art already contains the mechanism, the candidate remains dead regardless of its numbers.

### No run 3

Do not spend GPU on a third candidate in this round. T1 and I1 cannot produce multi-dataset methods evidence; B3 and I3 are recognizable occupied baselines; I2 has both the one-dataset and weak-comparator problems; and B2 needs a mathematical design decision before it can be frozen. The next legitimate action for B2 is a paper-and-pencil specification plus novelty check, not a pilot.
