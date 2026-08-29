# R4 deviation D1 — jury ruling

## 1. Status of the frozen null

**Yes. The null is a false-KILL generator and is a blocking defect. Do not apply it literally.**

Within-label shuffling preserves each encoder's marginal class-conditional score distribution and marginal ROC, but it replaces the observed inter-encoder copula with conditional independence. Conditional independence is not universally the mathematically most complementary coupling—a countermonotone error coupling can be more extreme—but it is a new and often much more favourable evidence-combination regime. It therefore does not represent "the same encoders with item-level complementarity removed." The synthetic demonstration and the real-data null deltas of +0.0549 and +0.0692 establish that, in this pilot, the operation manufactures combination gain and makes `3 * Null95` unattainable. Applying it would make the verdict turn on a known-invalid control, so the defect blocks the primary run until this ruling is frozen.

## 2. Replacement for R4-1 clause 2

### Impossibility ruling

**There is no non-arbitrary permutation that simultaneously (a) holds every encoder's empirical class-conditional score distribution and marginal test ROC fixed, (b) destroys item-level complementarity, and (c) yields a canonical "no-complementarity" distribution.**

With the marginal distributions fixed, the remaining object is the joint dependence structure, or copula. Item-level complementarity is a property of that copula. Shuffling within label chooses the independence copula and can increase useful diversity; aligning ranks chooses a comonotone copula and suppresses diversity; preserving observed rank correlation preserves part of the very complementarity being tested; a Gaussian-copula replacement adds an unjustified parametric assumption. None is a neutral null. I therefore decline to replace one arbitrary copula with another.

### Exact substitute: paired stratified joint-row bootstrap

The permutation null and `Null95` are **retired for R4-1 only**. Replace the second half of clause 2 with a one-sided paired bootstrap lower-confidence bar on MDL's improvement over the already frozen comparator. This tests whether the observed incremental ROC gain survives test-sample uncertainty without modifying the observed encoder dependence.

Implement exactly as follows:

1. Complete all 12 primary cells under the unchanged mechanism, comparator, epoch, and threshold rules. For every dataset \(d\), seed \(s\), and test item \(i\), retain the hard label, MDL test score, and score from that dataset's already frozen comparator. Do not refit either method inside the bootstrap.
2. Use **10,000 repetitions** from `numpy.random.default_rng(20260810)` in a single serial RNG stream. Dataset iteration order is exactly `HateMM`, `MHC-EN`, `MHC-ZH`, `ImpliHateVid`; within each dataset draw the positive stratum before the negative stratum.
3. In repetition \(b\), and independently for each dataset, sample with replacement exactly \(n_{d,1}\) indices from that dataset's positive test indices and exactly \(n_{d,0}\) indices from its negative test indices. Use the **same sampled joint rows** for MDL, the frozen comparator, all three seeds, and—if retained for diagnostics—all member encoders. Never permute one encoder or one method independently of another.
4. For each dataset and seed, compute ROC AUC on the resulting stratified bootstrap sample for MDL and for the frozen comparator using `sklearn.metrics.roc_auc_score`. Form

   `DeltaROC_d_b = mean over seeds [AUC(MDL bootstrap) - AUC(comparator bootstrap)]`.

   Then form

   `MeanDeltaROC_b = unweighted mean of DeltaROC_d_b over the four datasets`.

5. Collect the 10,000 values `MeanDeltaROC_b`. Define

   `LCB95 = numpy.quantile(MeanDeltaROC_boot, 0.05, method="linear")`.

   This is a one-sided 95% percentile-bootstrap lower bound. **Do not truncate values at zero. Do not multiply the bound by three. Do not form `Null95`.**

The resampling population is the observed joint test row, so all encoder scores, within-item alignments, and observed inter-encoder dependence remain intact. Individual bootstrap replicates naturally vary in their empirical ROC and weighted class-conditional distributions; that sampling variation is the uncertainty being measured. This is deliberately a weaker but valid guarantee, not a claim to have generated a complementarity-free world.

### Amended clause 2, verbatim

Replace R4-1 clause 2 with:

> **2. `MeanDeltaROC >= +0.010` and `LCB95 > 0.000`.**

Failure of either conjunct is a KILL under the existing all-clauses-required rule. Clauses 1, 3, and 4; the comparator selection; model; scope; thresholds; and every other GO/KILL statement remain unchanged. The 200 defective permutation repetitions must not be run or reported as a gate. If any were already produced during smoke, retain them only in the deviation record labelled **invalid diagnostic; excluded from verdict**.

## 3. Partial unblinding ruling

**The round survives with disclosure. No dataset, seed, threshold, comparator, bar, or mechanism is to be replaced because of the partial unblinding.**

The exposure is limited to MHC-ZH seed 0, occurred during a mandatory pre-primary smoke that discovered a verdict-changing defect, and happened before the remaining 11 cells. The replacement above is a standard paired uncertainty calculation, is not tuned to the observed −0.0017 value, and does not relax clauses 1, 3, or 4. Excluding or replacing the seen cell would create a worse selection problem.

Apply these procedural requirements:

1. Freeze this ruling and the current primary implementation hash before continuing.
2. Carry the already generated MHC-ZH seed-0 primary predictions forward unchanged as 1 of the 12 cells; do not discard, substitute, or selectively rerun that cell. If the exact prediction artifact was not saved and deterministic reconstruction is mechanically necessary, rerun it once under the identical frozen code and seed, suppress its output until final aggregation, and record that recovery explicitly.
3. Run the remaining 11 cells exactly once under the unchanged primary protocol. Suppress per-cell test metrics from interactive output until all primary predictions and frozen comparator choices are written to disk.
4. In the final result, reproduce the §4 disclosure: identify MHC-ZH seed 0 as viewed before D1, report the already seen numbers verbatim, state that only the invalid null was replaced by this jury ruling, and link both D1 and this ruling.

The pilot is therefore **partially unblinded but still adjudicable**. Its standing confirmatory-by-construction limitation remains exactly as frozen; this incident adds disclosure but does not require a new split or restart.
