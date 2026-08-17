# §9 deliverable — round-4 method candidates

## Gatekeeping assumptions

I treat the four bare-head numbers in the bundle as the baselines to beat, MHC-EN as the primary clean benchmark, and HateMM as supporting evidence only because its CLIP-text channel and official split are compromised. Candidates motivated by §4.1 or §4.3 are **confirmatory-by-construction**: the relevant test-set pattern has already been observed and must be disclosed. All proposed test runs therefore mean one pre-registered grid, train-only fitting, validation-only epoch/hyperparameter selection, a frozen decision rule, three or more seeds, and every test cell reported. Unless a candidate says otherwise, its null is an auxiliary-signal or structure permutation that preserves the hard labels and feature marginals; a GO additionally requires the gain to exceed three times that null's 95th-percentile gain. A mechanism that raises macro-F1 without raising ROC is a KILL even if its headline number looks good.

My honest base-rate view is that only six of the fourteen candidates clear roughly 5.5/10 before a pilot. The bare-head-only and ImpliHateVid groups are particularly thin: most obvious inhabitants are old AUC losses, LDA/QDA, hard-negative mining, or trivial three-class heads. I include the best falsifiable versions below, not fourteen claims that all deserve papers.

## Group 1 — mechanisms native to frozen features plus a head

### B1. Jackknife Lower-Bound Rank Head (JLR)

1. **One-sentence summary.** Replace pointwise BCE with a pairwise head that maximizes the leave-block-out lower confidence bound of every hate–non-hate margin, so rankings supported only by a few training items are deliberately discounted.
2. **Core hypothesis.** At n ≈ 10³, the flexible deployed head obtains part of its ranking from high-influence examples; optimizing the mean pair margin minus λ times its jackknife standard deviation should improve test ROC by retaining stable rather than merely large margins.
3. **Minimum viable experiment.** On every available dataset×encoder `.pt` cell, obtain five cross-fitted logits and per-pair margin means/standard deviations, train the same head with `softplus[-(mean_margin - lambda*sd_margin)] + 0.1*BCE` for a pre-registered \(\lambda\in\{0,1,2\}\) selected on validation, and compare with bare BCE, ordinary pairwise-AUC loss, and a five-head probability ensemble; permuting the standard deviations across opposite-label pairs is the null, and GO requires at least +0.010 test ROC on clean MHC-EN, positive mean ROC delta across all four datasets, and a non-negative fixed-rule macro-F1 delta.
4. **Contribution type.** new method.
5. **Risk.** MEDIUM.
6. **Effort.** days.
7. **Death-list check.** The closest dead direction is **retrieval-guided contrastive pairing**, but JLR neither forms embedding neighborhoods nor applies a contrastive representation loss: it robustifies the supervised ordering produced by the final head itself.
8. **Absorption check.** Not applicable: JLR is an inductively trained objective with no test context, routing, or adaptation.

### B2. Policy-Cone Discriminant Head (PCD)

1. **One-sentence summary.** Represent the binary policy not by one class-name prompt but by a convex cone of paired violation/safe-use clause directions, then constrain a covariance-whitened frozen-feature head to lie in that cone.
2. **Core hypothesis.** A heterogeneous policy class has no single semantic centroid, but directions such as protected-target attack versus quotation, endorsement versus condemnation, and dehumanization versus neutral description can regularize the small-sample head toward policy-relevant axes and improve ranking without a larger network.
3. **Minimum viable experiment.** Pre-register 12 paired policy clauses without inspecting test errors, encode them with the matching frozen text encoder, whiten their difference vectors with train covariance, fit non-negative cone weights plus an unconstrained visual residual using train labels, and compare against bare BCE, ridge LDA/GDA, one positive/negative prompt anchor, and an equal-size random-sentence cone; clause-pair sign permutation is the null, and GO is +0.010 mean test ROC on MHC-EN and ImpliHateVid with no dataset below -0.005 and fixed-rule macro-F1 not worse on average.
4. **Contribution type.** new method.
5. **Risk.** HIGH.
6. **Effort.** days.
7. **Death-list check.** The closest entry is **C3 vote-constrained semantic polytope**, but PCD uses no votes or item-level feasible regions; it constrains one global supervised head by pre-declared policy-clause directions.
8. **Absorption check.** Not applicable: policy anchors are fixed before training and no target context changes the fitted predictor.

### B3. Negative-Tail CVaR Rank Head (NTC)

1. **One-sentence summary.** Train the head against the conditional-value-at-risk of the worst hate–non-hate pairwise margins, allocating capacity to the high-scoring negative tail that determines moderation false positives.
2. **Core hypothesis.** BCE averages away the small negative tail that controls both partial AUC and many macro-F1 errors, whereas a CVaR pairwise objective should move those negatives below positives and therefore change ROC rather than merely the threshold.
3. **Minimum viable experiment.** For each cached-feature cell train `BCE + beta*CVaR_q(pairwise logistic margin)` with \(q\in\{0.1,0.2\}\) and \(\beta\in\{0.25,1\}\) frozen as a one-shot grid, comparing bare BCE, focal loss, ordinary pairwise AUC, and online hard-negative mining; randomizing tail membership among non-hate train examples is the null, and GO requires +0.010 test ROC and at least a 15% relative reduction in test FPR at a validation-frozen 90% hate-recall operating point on two datasets.
4. **Contribution type.** new method.
5. **Risk.** HIGH.
6. **Effort.** days.
7. **Death-list check.** The closest entry is **threshold/prior recalibration under label shift**, but NTC changes pair order during training and is evaluated at a frozen operating point; it never estimates a prior or shifts a threshold.
8. **Absorption check.** Not applicable: this is source-supervised inductive training with no test-set context.

**Group verdict.** B1 is worth a cheap pilot. B2 has a real structural distinction from LP++/CLAP but a reviewer may still call it prompt-anchored regularization. B3 is probably occupied in the broader pAUC/CVaR literature and is included mainly because it is the cleanest falsification of the measured negative-tail story; it is below 5/10 until a full novelty sweep says otherwise.

## Group 2 — transduction/TTA crossed with an exclusive asset

### T1. Pool-Relative Evidence Sparsification (PRES)

1. **One-sentence summary.** Use the unlabeled deployment pool only to estimate a background distribution over the 30 cached OCR windows, then pool each video from the few windows with the largest conditional surprisal relative to that background.
2. **Core hypothesis.** The OCR sign flip is caused by dilution—3 of 30 windows already deliver 61% of the gain—so a label-free pool estimate of what text is routine should expose item-specific evidence and improve ROC where learned fusion attends to ubiquitous or junk windows.
3. **Minimum viable experiment.** For HateMM and HateClipSeg, fit a train-pool shrinkage Gaussian to OCR-window embeddings after train-only linear residualization on the video visual vector, compute leave-video-out Mahalanobis surprisal, aggregate the top \(k\in\{1,3,5\}\) windows for train/validation and independently use the unlabeled test-pool background at test, then train the standard head with this fixed extra block; compare all-window mean, top-norm, learned attention/MLP fusion, and train-background-only PRES, use within-pool window-to-video permutation as the null, and GO is +0.010 test ROC over the bare head and over the best trained OCR comparator with gain at least three times the null bound.
4. **Contribution type.** new method.
5. **Risk.** MEDIUM.
6. **Effort.** days.
7. **Death-list check.** The closest entries are **provenance-typed OCR fusion** and **OCR−ASR residual**; PRES neither types/fuses provenance nor subtracts ASR, but attacks the still-live measured cause—which windows survive pooling—with a label-free background statistic.
8. **Absorption check.** The §5.3-3 equivalence assumes fixed features, a linear head, and squared loss; PRES changes the per-video representation through a nonlinear order statistic of a test-pool density and uses BCE, so it is outside the theorem's assumptions, although the paper must frame the novelty as sparse evidence extraction rather than generically "using context." It also is not TransCLIP: no class assignments, graph, prior, or test-label likelihood is fitted.

### T2. Test-Specified Metadata Nuisance Nulling (TMN)

1. **One-sentence summary.** Use unlabeled train/test title-description embeddings to identify metadata-predictable feature directions whose prevalence shifts at deployment, then null only those directions while constraining the source class-separation vector to be preserved.
2. **Core hypothesis.** MultiHateClip metadata is a major annotation cue and likely a selection channel, so the frozen video representation may inherit topic/platform directions that correlate with the source label but change in the test pool; removing only shifted metadata-predictable directions should improve target ranking without feeding metadata as another predictor.
3. **Minimum viable experiment.** Cache one metadata embedding per MHC item, estimate on unlabeled train+test a low-rank cross-covariance basis between metadata and the existing fused `.pt` features, solve for the smallest projection that removes basis components with train–test standardized mean shift above a pre-registered cutoff while preserving at least 99% of the train Fisher class direction, and refit the same head; compare bare BCE, adding metadata as a modality, CORAL, and removing the top feature PCs, permute metadata rows within split as the null, and GO is +0.010 test ROC in clean MHC-EN, the same sign in ZH, and no more than 0.005 validation ROC loss.
4. **Contribution type.** new method.
5. **Risk.** HIGH.
6. **Effort.** days.
7. **Death-list check.** The closest entry is **modality-attributed retrieval decomposition**; TMN has no retrieval and never treats metadata as contributing evidence—metadata defines a deployment-specific nuisance subspace and is discarded before classification.
8. **Absorption check.** TMN is a target-conditioned representation projection and therefore TTA-flavoured, but its constrained cross-covariance nulling is not squared-loss parameter adaptation or a router over fixed joint features; §5.3-3 can mimic some predictions with a kernel but does not supply the nuisance-identification constraint or preservation certificate. The strongest reviewer objection remains that this distinction may be formal rather than practically meaningful.

### T3. Jury-Robust Global Safe Adaptation (JRSA)

1. **One-sentence summary.** Convert train-only MultiHateClip vote fractions into a label-ambiguity polytope and permit a single global target-adapted head only when it weakly dominates the bare head for every labeling and density ratio inside that polytope.
2. **Core hypothesis.** TTA failures are amplified when ambiguous policy labels are treated as certain; using the otherwise-burned votes only to define a worst-case safety constraint should allow useful covariate-shift correction while forcing an exact fallback to the unadapted head when improvement cannot be certified.
3. **Minimum viable experiment.** On MHC-EN/ZH, restrict vote data to train/validation, estimate a clipped train-to-test density ratio from unlabeled frozen features, solve a convex importance-weighted pairwise-risk problem subject to `worst_case_risk(candidate) <= worst_case_risk(base)` over Bernoulli label intervals derived from votes, and compare no adaptation, unconstrained importance weighting, BBSE/EM, and the same robust program with hard labels only; permuting vote fractions within hard label is the null, and GO requires a non-vacuous certificate, activation on at least one language, +0.010 test ROC where active, and no active cell worse than -0.005.
4. **Contribution type.** theoretical result.
5. **Risk.** HIGH.
6. **Effort.** weeks.
7. **Death-list check.** The closest entry is **Human-Agreement Retrieval**; JRSA never retrieves by agreement, shapes a representation topology, or uses test votes—train votes specify uncertainty in a global safety constraint and disappear after fitting.
8. **Absorption check.** JRSA uses test context, but its object is a minimax pairwise-risk dominance certificate with a global fallback, not squared-loss adaptation or per-instance routing; the kernel-ridge equivalence does not absorb the robust feasible set or its no-degradation theorem. Unlike StatA-style anchoring, the constraint is relative to the deployed moderation head and explicit over label ambiguity.

**Group verdict.** T1 is the only cheap candidate in this group with both an exclusive asset and a directly measured mechanism. T2 is plausible but may collapse to ordinary nuisance regression. T3 is intellectually cleaner and answers the TTA counter-literature, but it is a theory project whose vote-based guarantee may be too conservative to activate.

## Group 3 — ImpliHateVid and the implicitness axis

All three candidates below target the observed binary error budget—false positives on NH—rather than claiming that IM recall is the problem. Their motivation is test-recon-derived and therefore confirmatory-by-construction. I do not think this group currently contains a 7/10 idea.

### I1. Incomparable-Positive Partial-Order Head (IPPO)

1. **One-sentence summary.** Treat EX and IM as two incomparable positive strata and optimize only the partial order `NH < EX` and `NH < IM`, so easy examples in either stratum cannot hide violations against non-hate and no gradient is wasted separating EX from IM.
2. **Core hypothesis.** Binary BCE averages EX and IM, while three-class training spends capacity on the irrelevant EX↔IM confusion; a worst-stratum partial-order loss should push high-scoring NH below both types of hate and reduce false positives while retaining a binary decision.
3. **Minimum viable experiment.** Using subtype labels only on ImpliHateVid train/validation, train the existing head with `max(mean_pairloss(NH,EX), mean_pairloss(NH,IM)) + 0.1*BCE`, compare bare BCE, balanced pairwise AUC, group-DRO BCE, and three-class softmax collapsed as `p(EX)+p(IM)`, and use repeated random two-way partitions of positive train items matched to the EX/IM counts as the subtype-label null; GO requires +0.010 test ROC, at least 15% relative reduction in NH FPR at validation-frozen 90% hate recall, at most 0.01 absolute loss in both EX and IM recall, and more than three times the random-partition null gain.
4. **Contribution type.** new method.
5. **Risk.** MEDIUM.
6. **Effort.** days.
7. **Death-list check.** The closest entry is **type-hard-partitioned memory**; IPPO uses no memory and does not route examples to subtype experts—the subtypes impose two inequalities on one shared score.
8. **Absorption check.** Not applicable: subtype labels affect source training only and the predictor has no target context or routing.

### I2. Shared-Hate Cone Head (SHC)

1. **One-sentence summary.** Learn one max-margin direction constrained to have positive class separation from NH for both EX and IM, discarding subtype-specific directions that cannot support the shared binary policy decision.
2. **Core hypothesis.** Hate-adjacent NH may align with an EX-only lexical or an IM-only contextual direction, whereas a direction lying in the intersection of the two positive-separation half-spaces should retain evidence common to both hate types and rank those negatives lower.
3. **Minimum viable experiment.** Standardize and train-only PCA the concatenated frozen image/text features, estimate shrinkage means/covariances for NH/EX/IM, solve a second-order-cone problem maximizing the minimum standardized margin against NH, and compare binary ridge-LDA/GDA, the standard nonlinear BCE head, separate EX and IM heads combined by max/mean, and IPPO; randomly repartitioning positives into EX/IM-sized groups is the null, and GO requires +0.010 test ROC plus lower NH FPR at fixed hate recall than every closed-form comparator, with positive bootstrap lower bound over three seeds.
4. **Contribution type.** new method.
5. **Risk.** HIGH.
6. **Effort.** days.
7. **Death-list check.** The closest entry is **type-hard-partitioned memory**, but SHC explicitly avoids a hard subtype partition by finding one direction in the intersection of both subtype constraints and contains neither a datastore nor retrieval.
8. **Absorption check.** Not applicable: SHC is a closed-form source-trained head and does not adapt or route at inference.

### I3. Cross-Fitted Non-Hate Veto (CNV)

1. **One-sentence summary.** Add a one-sided residual function that may only subtract from the cross-fitted bare-head logit and is trained specifically on non-hate examples the base head over-fires on, with a penalty that forbids subtraction on either hate subtype.
2. **Core hypothesis.** ImpliHateVid's remaining binary error is an asymmetric residual—hate-adjacent NH receives spurious positive evidence—so a constrained negative-evidence veto can repair their ordering without asking a second unrestricted head to relearn the whole task.
3. **Minimum viable experiment.** Produce five-fold out-of-fold train logits, mark the upper-quartile scored NH items as veto positives, fit `r(x)=softplus(h(x))` with loss `BCE(veto_target,r)+gamma*E_positive[r]`, and predict `s(x)=base_logit(x)-alpha*r(x)` with \(\alpha,\gamma\) selected on validation; compare a matched-capacity deeper BCE head, focal loss, online hard-negative mining, and residual boosting, permute the cross-fitted over-fire indicator within NH for the null, and GO requires +0.010 test ROC, 20% relative NH-FPR reduction at 90% hate recall, and no more than one-point recall loss on EX or IM.
4. **Contribution type.** new method.
5. **Risk.** HIGH.
6. **Effort.** days.
7. **Death-list check.** The closest entry is **per-sample abstention / escalation routing**; CNV never abstains or chooses a route, and emits one continuous inductive score whose correction is learned entirely from source residuals.
8. **Absorption check.** Not applicable: despite the word "veto," CNV is a fixed source-trained score function, not a context-dependent router or adapter.

**Group verdict.** I1 is the only member I would pilot without further ideation because it encodes the EX/IM distinction without optimizing the irrelevant 3-class problem. I2 is elegant but may just rediscover conservative LDA. I3 is likely to be read as asymmetric boosting or hard-negative mining. None should be sold as "solving implicit hate"; the binary test evidence does not support that claim.

## Group 4 — revival re-ranking

### R1. C4 revived: Balanced Semantic Response-Tensor Distillation (B-SRTD)

1. **One-sentence summary.** Build a balanced two-axis semantic intervention lattice over both hard labels and distil a teacher's named-intervention Jacobian and mixed partial into the bare head rather than distilling its logits.
2. **Core hypothesis.** A head that matches how a strong teacher responds to target substitution and endorsement/condemnation reversal will learn policy-relevant local geometry that ordinary counterfactual augmentation or logit distillation omits, improving original-item ranking even with frozen encoders.
3. **Minimum viable experiment.** Before any new test inspection, use Claude to construct at least 200 train and 80 validation 2×2 lattices balanced by hard label and intervention validity, embed all variants with one frozen encoder into `.pt`, train the standard head with BCE plus losses on the two finite differences and one mixed partial, and compare bare BCE, ordinary logit distillation, counterfactual augmentation, Jacobian-only matching, and a matched number of teacher calls; independently permute intervention names and tensor coordinates within hard label for the null, then make one three-seed test submission, with GO requiring +0.010 ROC on at least two datasets, positive mean fixed-rule macro-F1, response-sign accuracy above both comparators, and a gain over three times the null bound.
4. **Contribution type.** new method.
5. **Risk.** MEDIUM.
6. **Effort.** weeks.
7. **Death-list check.** This is the explicitly revival-eligible **C4 semantic response-tensor distillation**, not a re-skin of dead C5: the old blocker was a one-axis, all-positive asset, and the proposed balanced two-axis lattice removes that precondition failure; the claim remains narrowly the named-intervention response tensor, not generic counterfactual or Jacobian distillation.
8. **Absorption check.** Not applicable: B-SRTD trains an inductive head on generated source interventions and performs no test-time adaptation, context routing, or parameter update.

### R2. C6 revived: Executable Accountability-Path Distillation (EAPD)

1. **One-sentence summary.** Distil a typed agency graph into separately supervised proposition, quotation, endorsement, condemnation, and accountable-speaker edges, then compute hate only when an attack path reaches an accountable endorsing agent.
2. **Core hypothesis.** The dominant NH false positives are plausibly use–mention, reporting, counterspeech, or quoted hate, and an executable path product will suppress these cases more reliably than an unconstrained rationale embedding or another fused classifier.
3. **Minimum viable experiment.** Pre-register an edge schema, have Claude annotate 250 train and 80 validation videos balanced across NH/EX/IM without exposing IDs, train small edge-prediction heads on the existing frozen features, and define the final logit from a fixed differentiable path expression such as `attack * accountable * (endorse - condemn)` plus a residual capped in magnitude; compare the bare head, same-parameter multitask edge prediction with a free MLP readout, teacher-logit/rationale distillation, and C1's scalar stance features, permute edge labels within hard class for the null, and GO requires +0.010 test ROC and at least 20% relative NH-FPR reduction at fixed 90% hate recall with less than one-point hate-recall loss.
4. **Contribution type.** new method.
5. **Risk.** MEDIUM.
6. **Effort.** weeks.
7. **Death-list check.** The closest dead entry is **C1 target-conditioned attack/defence stance algebra**, but C1 reduced an item to target/stance scalars and failed its double dissociation; EAPD is the separately held C6 mechanism, whose load-bearing object is a multi-hop accountable-agent path that distinguishes asserting, quoting, endorsing, and condemning.
8. **Absorption check.** Not applicable: the graph student and executable readout are fixed after source training; there is no test context, router, or adaptation.

### R3. C9 re-gated: Sparse Rank-Copula Pooling (SRCP)

1. **One-sentence summary.** Apply differentiable within-video ranks to the few pre-selected visual/OCR/transcript observations and classify from their cross-stream copula, testing whether relative co-occurrence survives scale and encoder mismatch better than mean/max pooling.
2. **Core hypothesis.** Pooling is a measured culprit and rank normalization may expose rare cross-stream dependence, but the 8.2% multi-segment rate and frozen-CLIP's 0.511 within-video localization AUROC make the identifying signal doubtful unless sparsification happens before the copula.
3. **Minimum viable experiment.** On existing K=30/K=4 cached segment features, pre-register three stream summaries (visual, OCR, transcript where available), retain the top three windows by a train-only non-label score, compute soft empirical ranks and all pairwise rank products, and fit a parameter-matched head; compare mean, max, learned attention, the original dense C9, and rank features with stream alignment permuted within hard label, with a strict GO of +0.010 validation and test ROC over the strongest trained pooling comparator and at least three times the permutation-null 95th percentile in two datasets.
4. **Contribution type.** new method.
5. **Risk.** HIGH.
6. **Effort.** days.
7. **Death-list check.** The closest dead direction is **multi-segment complementarity**; SRCP does not claim that segments contain complementary class evidence or select a segment, but tests cross-stream dependence after ranks remove marginal scales—nonetheless the same weak within-video signal may kill it.
8. **Absorption check.** Not applicable: SRCP is fixed pooling inside an inductive classifier and uses no target-pool adaptation or context routing.

**Revival verdict.** R1 moves from 6.0/10 to about 7.0/10 because the bare head makes the student cheap and the Claude exemption makes the missing balanced lattice buildable. R2 moves from unscored to about 6.4/10 because §4.1 supplies a matching false-positive failure mode, albeit confirmatory-by-construction. R3 remains below 4/10 and should run only as a one-day gate. C7 should stay dead because its order-sensitive parameters remain unidentified by an 8.2% multi-segment regime. C2, C3, C11, C13, and C14 should also stay dead: the foundation reduces engineering cost but supplies none of their missing causal signal or novelty defense.

## Group 5 — free candidates

### F1. Monotone Disagreement Lattice (MDL)

1. **One-sentence summary.** Fit a monotone lattice over out-of-fold encoder logits that is anchored to the best single encoder on concordant examples and learns non-additive corrections only where encoders disagree.
2. **Core hypothesis.** The +1.5–2.0 ROC points from mean ensembling prove useful complementary order information exists, but averaging assumes a globally additive, equally scaled score; a monotone interaction surface can learn which disagreement geometries are reliable while preserving coordinate-wise consensus.
3. **Minimum viable experiment.** Train each CLIP/Qwen/LoRA base head in five folds to obtain leakage-free train logits and ordinary validation/test logits, fit a 2- or 3-D piecewise-linear lattice with non-negative finite differences using pairwise logistic rank loss plus an identity penalty to the validation-best encoder when all logits are concordant, and compare best single, mean probability/logit, validation-weighted average, logistic stacking, and a matched MLP stacker; independently permute one encoder's logits within hard label to destroy item-level complementarity while retaining its marginal accuracy for the null, and GO requires +0.010 test ROC on clean MHC-EN, positive ROC delta on at least three of four datasets, mean fixed-validation-rule macro-F1 gain of at least +0.010, and more than three times the null gain.
4. **Contribution type.** new method.
5. **Risk.** MEDIUM.
6. **Effort.** days.
7. **Death-list check.** The closest entry is **rank–vote decoupling**, but MDL uses neither human votes nor retrieval ranks; it learns a supervised monotone interaction among independently trained encoder scores and tests complementarity against strong stacking baselines.
8. **Absorption check.** Not applicable: MDL is trained inductively from cross-fitted source predictions and consumes no unlabeled test context; it is an ensemble head, not TTA or context routing.

### F2. Safe Covariate-Shift Rank Adaptation (SCRA)

1. **One-sentence summary.** Refit the head for the unlabeled target pool by maximizing worst-case target-weighted pairwise AUC subject to a formal constraint that its worst-case rank risk cannot exceed the deployed bare head's.
2. **Core hypothesis.** A train–test density ratio can change which source pairs matter and therefore improve target ROC, while an ambiguity set around that ratio can turn the TTA literature's informal "do no harm" demand into an exact fallback guarantee under stated covariate-shift assumptions.
3. **Minimum viable experiment.** For each dataset, train a cross-fitted discriminator between train and unlabeled test fused features, convert its clipped odds into pair weights, and solve a linear-last-layer distributionally robust pairwise-logistic program over an \(f\)-divergence ball with the bare head included as a feasible point; compare bare BCE, unconstrained importance weighting, GroupDRO, BBSE/EM, and an unweighted robust head, use train/test domain-label permutation as the null, and GO requires a non-vacuous certified constraint, +0.010 test ROC in MHC-EN or two other cells, no cell below -0.005, and fixed-rule macro-F1 positive on average.
4. **Contribution type.** theoretical result.
5. **Risk.** HIGH.
6. **Effort.** weeks.
7. **Death-list check.** The closest entry is **threshold/prior recalibration under label shift**; SCRA estimates a covariate density ratio and changes the pairwise ranking function under a worst-case constraint, not the class prior or decision threshold.
8. **Absorption check.** SCRA is adaptation, but its minimax pairwise objective and baseline-dominance feasible set violate the squared-loss/linear-fixed-feature assumptions of §5.3-3; a kernel representation of its outputs would not absorb the contribution, which is the target-rank no-degradation guarantee. The guarantee must be stated conditionally—if covariate shift or ratio coverage fails, the theorem does not promise safety.

**Group verdict.** F1 is the strongest cheap candidate in the round because it directly attacks the cleanest measured signal with an inductive mechanism and strong trivial baselines. F2 is the cleanest top-venue answer to the TTA counter-literature, but it is high risk: the ambiguity set may make the solution equal the bare head everywhere, and its moderation-specific empirical gains may be too small for a general theory paper.

## Overall ranking

The scores below are pre-pilot research judgments, not predicted F1 gains.

| rank | candidate | score / 10 | why it sits here | immediate action |
|---:|---|---:|---|---|
| 1 | **F1 MDL** | **7.4** | Directly attacks the strongest clean cross-encoder signal; cheap and falsifiable; novelty versus monotone stacking is the main risk. | RUN |
| 2 | **R1 B-SRTD** | **7.0** | Highest prior jury score, and its sole asset blocker is now removable; mechanism remains distinguishable from logits and generic augmentation. | BUILD SMALL LATTICE, THEN RUN |
| 3 | **T1 PRES** | **6.6** | Crosses new transductive freedom with an exclusive cache and the surviving measured OCR cause; very cheap. | RUN |
| 4 | **R2 EAPD** | **6.4** | Best semantic match to ImpliHateVid's NH false positives, but annotation validity and C1 adjacency are serious. | ANNOTATION RELIABILITY GATE |
| 5 | **F2 SCRA** | **6.1** | Could make the demanded no-harm property a theorem and move ranking, but may return the baseline everywhere. | DERIVE FEASIBILITY GATE |
| 6 | **I1 IPPO** | **5.8** | Correctly uses implicitness without chasing IM recall or EX↔IM classification; likely incremental relative to group-DRO ranking. | RUN AFTER TOP 3 |
| 7 | **B1 JLR** | **5.4** | Cheap robust-ranking test with a clear null; broad influence/robust-AUC prior art may own it. | NOVELTY CHECK, THEN RUN |
| 8 | **T3 JRSA** | **5.3** | Exclusive votes enter as a safety object rather than retrieval, but the certificate may be vacuous. | THEORY TOY CHECK |
| 9 | **T2 TMN** | **5.1** | Clever use of untouched metadata without adding a modality; likely to look like ordinary nuisance regression. | RUN ONLY IF SHIFT BASIS IS STRONG |
| 10 | **B2 PCD** | **5.0** | Occupies the policy-anchor gap, but reviewers may reduce it to prompt ensembles plus a cone constraint. | LITERATURE CHECK |
| 11 | **I2 SHC** | **4.7** | A clean closed-form use of the strata, but likely conservative LDA and possibly null because prototypes transfer across EX/IM. | CHEAP NULL ONLY |
| 12 | **I3 CNV** | **4.4** | Targets the correct error side, but asymmetric residual boosting is not obviously a top-venue mechanism. | HOLD |
| 13 | **B3 NTC** | **4.2** | Mechanically aligned with false positives but almost certainly crowded by partial-AUC/CVaR work. | HOLD PENDING NOVELTY |
| 14 | **R3 SRCP** | **3.8** | Pooling is live, but the project's own identifiability evidence is hostile. | STAY DEAD UNLESS ONE-DAY GATE IS LARGE |

## The top three I would actually run

### 1. F1 MDL

It is a cached-logit experiment measured in hours, uses MHC-EN as a clean primary cell, and asks the round's sharpest question: can a constrained decision surface convert genuine encoder complementarity rather than merely average it? **Strongest reviewer objection:** this is monotone stacking with a hate-video case study, and the apparent opportunity was discovered on the same test sets used for confirmation. The defense must therefore be a mechanism-level constraint/guarantee, strong logistic/MLP/weighted-ensemble baselines, the within-label complementarity null, and explicit confirmatory disclosure; a one-dataset gain is a KILL.

### 2. R1 B-SRTD

This is the only prior high-ranked idea whose death was purely an asset failure, and the new foundation plus Claude exemption reduce the asset and training cost enough to test the actual hypothesis. **Strongest reviewer objection:** named response tensors are just Jacobian matching or counterfactual augmentation with LLM-authored examples, so any gain may come from more teacher calls or synthetic labels rather than the tensor. The pilot must equalize calls/examples, include logits-only, augmentation-only, Jacobian-only, and coordinate-permutation controls, and show original-item ROC gain coupled to held-out intervention-response accuracy.

### 3. T1 PRES

It is the most economical exclusive-asset×transduction experiment and directly targets the only OCR explanation that survived prior pilots: dilution by window choice. **Strongest reviewer objection:** pool-relative surprisal is embedding-space TF–IDF/top-k pooling, while using the test pool as a background is an unnecessary transductive flourish; any gain might be reproduced with a train-only corpus statistic. The decisive comparison is therefore test-background PRES versus train-background-only PRES, top-norm, attention, and the trained fusion MLP; without both a ROC gain over the trained comparator and an additional target-background gain, it is not a paper.

## Final gate recommendation

Run F1 and T1 first because both fit cached assets and can die cheaply. In parallel, build only a small balanced C4 lattice and verify intervention validity and teacher response variance before generating the full asset. Do not launch R2's larger graph annotation until those three gates report. Treat all improvements on HateMM as supporting only, lead with clean MHC-EN, and describe every §4-derived result as confirmatory-by-construction.
