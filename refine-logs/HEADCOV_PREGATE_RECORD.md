# HEADCOV — $0 falsification of the LITSWEEP-8 label-tuple theorem **in the deployed head space**

**Date:** 2026-07-28 NZST · **Agent:** litsweep-8 (pathology lens), Task 1 · **Cost: $0** — CPU only,
≤8 threads, **zero GPU, zero SLURM, zero Modal, zero training, zero test-split contact**.

**What this is.** A falsification test of the two measurements that carry
`refine-logs/LITSWEEP8_PATHOLOGY_MATCH.md` §2. Those were taken in the **raw** train-LOO fused arena.
The deployed system retrieves in the **trained head** space. This record asks whether the theorem
transfers. **A negative result here is the point of the exercise, not a failure of it.**

**What this is not.** Not a verdict, not a prereg for a GPU cell, not a performance claim. It reads no
test metric and produces none.

**Complementary, non-overlapping:** a separate agent owns
`refine-logs/INSTRUMENT_VALIDATION_RECON.md` (the broad instrument-validation question). This record
answers only the narrow question *does the kernel argument hold in head space*. Not written to, and not
read from, this sweep. Untouched per tasking: `VSW_PREGATE_RECORD.md`, `scripts/analysis/vsw_*`,
`MEMBANK_C4_PREGATE_RECORD.md`, `scripts/analysis/*membank_c4*`, `STREAMCOMP_FORENSIC_RECON.md`,
`LITSWEEP7_LANDING_SITE.md`, `PROVENANCE_AUDIT_2026-07-28.md`.

---

## §1. INSTRUMENT INVENTORY — established BEFORE any bar was written

This is the load-bearing section. The tasking asked me to verify what is replayable rather than assume it.

### 1.1 Deployed head checkpoints: **effectively all gone**

`find logging/Retrieval -name "*.pt"` → **228 files, in 6 run directories, all of them
`mntp_s1_cpuhead`** (the F92 MNTP stage-1 **bidir** heads, i.e. heads over
`…-LoRA-curric-bidir_HF` / `…-LoRA-bidir_HF` features). `find logging/Retrieval -type d -name ckpt -empty`
→ **97 empty ckpt directories.** Every deployed-configuration head — HateMM `RAC_video_lora_curric`,
MHC-ZH `errpat_zh_remint_v2`, MHC-EN/`MHC_zh` `RAC_video_archive{,_seeds}` — has an **empty** `ckpt/`.

* The 9 P2-era deployed ckpts named in `scripts/analysis/p2_rerank_eval.py:55-63` (`CKPT_FILE`) were
  each existence-checked: **0 of 9 exist.** This **confirms and extends F78** ("6/6 deployed HateMM floor
  head ckpts MISSING") — it is not HateMM-specific; it is the whole deployed head inventory.
* The surviving `mntp_s1_cpuhead` heads are **not** a usable instrument: F92 closed the bidir readout
  route at zero training and F72 measured the naked mask-flip cratering −10 to −14 pts. A head over a
  measured-dead encoder variant cannot stand in for the deployed head.

**Consequence:** the version of HEADCOV sketched in `LITSWEEP8_PATHOLOGY_MATCH.md` §5.3 — which assumed
the MECHFIX-era ERRPAT proxy heads were loadable — **cannot be run as sketched.** Re-projecting keys
through a head, and therefore any pool-expansion measurement at `M > 20`, is unavailable at $0.

### 1.2 Banked head-space per-item artifacts, and their split

| artifact | space | split | n | depth | usable here? |
|---|---|---|---|---|---|
| `scripts/analysis/errpat_remint_dumps/errpat_zh_remint_seed{0,1,2}.pkl` | **deployed MHC-ZH head key space** (CPU re-mint heads, `errpat_zh_remint_v2`) | **`dev` and `test`, 30 epochs each** | dev **78** / test 149 | top-**20** (`nb_ids`, `nb_sim`, `nb_lab`) + `vote`, `pred`, `gold` | **YES — `dev` records only** |
| `scripts/analysis/errpat_hatemm_peritem.csv` | deployed HateMM head space (ERRPAT CPU proxy heads) | **TEST ONLY** | 215 | rank-weighted purity + top-1 sim + vote | **NO — consuming it is a test touch** |
| `scripts/analysis/p2_out/cache_{MHC,MHC_zh}_s*.json` | deployed archive-kNN key space | **TEST ONLY** (`samples` = 149 for ZH) | test | top-60 | **NO — test touch** |

### 1.3 Scope declaration — **frozen here, before any number**

* **MHC-ZH: IN SCOPE.** `dev` split, n = 78, seeds 0/1/2, all 30 epochs. Instrument = the ERRPAT CPU
  **re-mint** heads. Disclosed limitation: these are a re-mint proxy for the deployed ZH head, and
  `MECHFIX_PREGATE_2026-07-27.md:154` records that only their **final epoch** is device-reproducible.
  **PRIMARY READ = final epoch (29).** All 30 epochs are reported as a stability read and cannot carry a bar.
* **HateMM: OUT OF SCOPE.** Its only head-space per-item artifact is **test-only**. Declared out of scope
  now, before any number exists, so it cannot be shopped for later.
* **MHC-EN: OUT OF SCOPE.** All 4 deployed head ckpts deleted; its only banked head-space artifact
  (`p2_out/cache_MHC_s*.json`) is test-only. Exactly as pre-declared in `LITSWEEP8` §5.3.
* **The cost of extending to HateMM/EN, reported rather than spent** (per tasking): a CPU head re-mint,
  which `ERRPAT_HateMM_2026-07-26.md:6.1` prices at ~52 s CPU per seed. That is $0 in GPU but it is
  **training a new instrument**, which needs its own fidelity/parity gate (MECHFIX spent §2 on exactly
  that). **I do not mint an instrument inside a falsification test of my own claim.** Recommended as a
  separate, gated job if the parent wants 3-dataset coverage.

### 1.4 Budget compliance

`research-wiki/experiments/exp-encoder-3seed.md` §"Decision rule (pre-registered)" / §"Judgment" define
the val-selection and reporting discipline for *test-bearing* verdicts. **No test touch is consumed by
this record**: the loader filters `split == 'dev'` and a hard assert refuses any record whose split is not
`dev`. No held-out test metric is read, computed, or reported.

---

## §2. WHAT IS UNDER TEST

The deployed decision (`src/utils/metrics.py:262-301`, recomputed inside
`scripts/analysis/errpat_zh_remint.py:60-84` under an assert that it matches `metrics.py`'s own vote list
to `< 1e-12`) is

```
v = Σ_i (2·lab_i − 1)·sim_i·w_i / Σ_i w_i ,  w = [20,19,…,1],  predict 1 iff sigmoid(v) ≥ 0.5 ⟺ v ≥ 0
```

`LITSWEEP8` §2 Result A defines the **label-only** vote, with every similarity discarded:

```
M = Σ_i (2·lab_i − 1)·w_i / Σ_i w_i ,                        predict 1 iff M ≥ 0
```

and Result B defines coverage `C(k) = 1[ ∃ i ≤ k : lab_i = gold ]`.

**The bound that makes a $0 test decisive despite depth-20 lists.** The free-weight oracle at pool `M` is
exactly `coverage(M) − acc`, because an unconstrained non-negative reweighting is correct iff the pool
contains ≥1 correct-label item. Since `coverage(M) ≤ 1` for all `M`, the **marginal** oracle of expanding
the pool beyond 20 obeys

```
oracle(∞) − oracle(20) = coverage(∞) − coverage(20) ≤ 1 − coverage(20).
```

**So measuring head-space `coverage(20)` upper-bounds the entire pool-expansion / re-ranking family
without ever retrieving past rank 20.** This is what makes the truncated dumps sufficient.

---

## §3. FROZEN GATES AND BARS

**All of §3 was written before any dev number in §4 existed.** No arm, threshold, split or protocol below
was changed after a result was seen.

### 3.1 PARITY-HC — machinery gate (mandatory; a failure VOIDS the record)

Recompute `v` from the dumped `(nb_sim, nb_lab)` via the deployed arithmetic above and compare to the
dumped `vote`; recompute `pred` and compare to the dumped `pred`.
**Bar: max |Δvote| < 1e-9 AND pred agreement = 1.0000, on 100 % of (seed × epoch × dev item) cells
(3 × 30 × 78 = 7 020).** Anything less and every number below is void and must be reported as void.

### 3.2 K-HC-3 — **integrity / does Result A transfer?** (read first; it gates the meaning of the others)

Decision identity between the deployed vote `v` and the label-only vote `M`, on ZH dev, head space.

* **PASS (≥ 0.98):** the kernel argument holds in the deployed space. `LITSWEEP8` §2 Result A is a
  **deployed-space** theorem and the campaign's closure of the metric-value axis is principled.
* **FAIL (< 0.98):** `LITSWEEP8` §2 Result A is **raw-space only**, its C1-C4 corollaries must be
  re-scoped to the raw arena, and the head-space metric axis **re-opens**. *This record must say so
  loudly and without hedging.*

### 3.3 K-HC-1 / K-HC-2 — **does Result B transfer?**

Head-space `coverage(20)`, and the derived bound `1 − coverage(20)` on the pool-expansion oracle.

* **K-HC-1 (confirm, closes CAND-A in the deployed space):** `1 − coverage(20) < 0.030` ⇒ the marginal
  oracle of *any* candidate-set change (k-reciprocal, mutual proximity, local scaling, NICDM, QB-Norm,
  NNN, larger k, any re-metrication that only re-ranks) is **arithmetically** under the +0.030 bar on ZH.
  `LITSWEEP8` §5.1's kill stands, in the deployed space, for this dataset.
* **K-HC-2 (falsify):** `1 − coverage(20) ≥ 0.030` ⇒ **`LITSWEEP8` §5.1's kill is WITHDRAWN** and exactly
  one re-ranker (k-reciprocal encoding, the canonical set-changing operator that is neither CSLS nor
  label-propagating) earns a further $0 pregate under the F98 degeneracy-control discipline.

### 3.4 DEG-HC — **triviality control** (mandatory; declared now, not post-hoc)

K-HC-3 could pass trivially if the head-space cosines within a query were numerically identical, in which
case the identity would be an artefact of collapse rather than a property of the decision rule. Report:

1. the head-space within-query similarity spread `sim(rank 1) − sim(rank 20)`, at the final epoch and at
   **epoch 0** (an essentially untrained head, before any collapse);
2. K-HC-3's agreement **at epoch 0** specifically.

**Reading rule, frozen:** if the identity holds at epoch 0 *as well*, where the head is not collapsed, the
result is a property of the **decision rule** (strong). If it holds only at late epochs, it is a property
of **collapse** (weaker, and must be stated as such).

### 3.5 Anti-shopping rule

Primary read = **MHC-ZH, dev, final epoch (29), 3 seeds.** The 30-epoch curves and epoch-0 reads are
stability/mechanism reads and **cannot carry a bar**. No other dataset, split, epoch or protocol may be
promoted into the primary read after the fact.

<!-- EVERYTHING BELOW THIS LINE WAS WRITTEN AFTER THE RUN -->

---

## §4. RESULTS

Loaded: **90 dev cells** = 3 seeds × 30 epochs, n = 78 each. **Zero test records were read** — the loader
filters `split == 'dev'` and asserts it.

### 4.1 PARITY-HC — **PASS, exactly**

| quantity | bar | measured |
|---|---|---|
| max \|Δvote\| (recompute from `nb_sim`,`nb_lab` vs dumped `vote`) | < 1e-9 | **0.000e+00** |
| pred agreement | = 1.0000 | **1.0000** |
| cells | 7 020 | **7 020** |

*Determinism note (per the confirmed drift defect):* this is a **closed-form** recompute over banked
arrays — no trained estimator is re-fitted anywhere in this record — so bit-exactness is a legitimate
claim here. Nothing in HEADCOV re-mints a head; the F95-module non-determinism on trained quantities does
not apply to any number in this record.

### 4.2 K-HC-3 — **PASS at 1.0000. Result A transfers to the deployed head space.**

Decision identity between the deployed vote `v` and the label-only vote `M`:

| read | cells | identity | min over cells |
|---|---|---|---|
| **FINAL epoch 29 (PRIMARY)** | 3 | **1.0000** | 1.0000 |
| epoch 0 | 3 | 1.0000 | 1.0000 |
| all 30 epochs (stability) | 90 | 0.9989 | **0.9872** |

Per seed at the primary read: **0/78, 0/78, 0/78 items differ.** Deployed acc 0.8462 / 0.8333 / 0.8462;
label-only acc **identical, 0.8462 / 0.8333 / 0.8462**. Every one of the 90 stability cells clears the
0.98 bar (min 0.9872).

> **In the deployed head space, discarding every cosine and voting on the retrieved labels alone changes
> nothing at all.** `LITSWEEP8_PATHOLOGY_MATCH.md` §2 Result A is a **deployed-space** result, not a
> raw-space artefact. Corollaries C1-C4 stand as written.

### 4.3 K-HC-1 — **FIRES. Result B transfers; CAND-A is closed in the deployed space (ZH).**

| read | coverage(20) | **1 − coverage(20)** = bound on the pool-expansion oracle | vs bar 0.030 |
|---|---|---|---|
| **FINAL epoch 29 (PRIMARY)** | **0.9829** | **+0.0171** | **under, 1.8×** |
| worst single seed (s1) | 0.9744 | +0.0256 | under, 1.2× |
| all 30 epochs | 0.9905 | +0.0095 | under, 3.2× |

Per seed at the primary read: **1 / 2 / 1 of 78 dev items** have no correct-label neighbour anywhere in
the deployed top-20.

> **K-HC-1 fires on the mean and on every individual seed.** The marginal oracle of *any* candidate-set
> change — k-reciprocal, mutual proximity, local scaling, NICDM, QB-Norm, NNN, larger `k`, or any
> re-metrication that only re-ranks — is **arithmetically ≤ +0.0171** on ZH in the deployed space.
> `LITSWEEP8` §5.1's kill **stands**, now confirmed in the space the system actually retrieves in.
> **K-HC-2 does not fire.** No re-ranker earns a pregate.

### 4.4 DEG-HC — **the control DID NOT DISCRIMINATE, and its premise was measured FALSE**

My frozen reading rule assumed epoch 0 is an *uncollapsed* reference. It is not:

| read | median within-query sim spread, rank 1 − rank 20 |
|---|---|
| epoch 0 | **9.34e-07** |
| epoch 29 | 1.95e-04 |

The untrained head is *more* numerically degenerate than the trained one (the head at epoch 0 is
essentially a random projection producing near-identical cosines), so "identity holds at epoch 0" carries
**none** of the evidential weight §3.4 assigned it. **Reporting this as a failed control rather than as a
pass.**

What this means, stated precisely and without hedging: **in head space the label-only identity is
essentially *forced*** — the similarity spread (1.95e-04) is ~49× smaller than `M`'s smallest possible
step (2·w₂₀/210 = 0.0095), so `sign(v) = sign(M)` is near-arithmetically guaranteed. The evidence that
the identity is a property of the **decision rule** rather than of collapse therefore lives in the
**raw** arena, where `LITSWEEP8` §2 measured 99.6-100 % identity *despite* a real spread of 0.021-0.025 —
and in `VSW_ASYMMETRY_RECON.md` §5, which independently deletes cosine magnitude in the raw fused arena
and measures Δacc **−0.0013 / +0.0000 / −0.0018** at **99.60 / 99.65 / 99.82 %** agreement. **Three
independent measurements, two arenas, same conclusion.** The head-space read confirms *transfer*; the
raw-space reads carry the *mechanism*.

---

## §5. CONNECTION TO F105/VSW — the Samworth citation becomes a measurement

`LITSWEEP8` §3.1 imported Samworth (2012, Ann. Statist. 40(5):2733-2763, arXiv:1101.5783): the regret
ratio of the *optimally* weighted NN classifier to the unweighted k-NN classifier "depends asymptotically
only on the dimension `d` … The improvement is greatest when `d = 4`, but thereafter **decreases as
`d → ∞`**." Our `d` is 1024 (head) / 7168 (raw).

F105/VSW (commit `e9a17fe`) measured the finite-`n` shadow of exactly that statement, and in doing so
**refuted the campaign's own exchange-rate law**: K-VSW-2's predeclared outcome (b) is FALSE — the
exchange rate reaches **6.0** on HateMM (against F95's best-of-36-cells 1.1667) and VSW still fails. The
correct law is `net = changed × (2·precision − 1)`, with precision decaying monotonically as sharpness
rises (HateMM 0.8571 at 21 changed → 0.5696 at 79 changed), pinning net to **+11…+21 items across a
16 384× λ range**.

**These are the same statement.** VSW's λ-continuum spans the deployed rank profile at one end and
single-best-neighbour adjudication at the other — i.e. it is a search over the weighted-NN family that
Samworth optimises analytically. Samworth says the family's best member beats uniform by an amount that
vanishes with `d`; VSW measures that the family's best member beats the deployed profile by an amount
pinned below the bar at *every* point of the continuum, *including* points where the per-item exchange
rate is excellent. **A high exchange rate with a capped net is precisely what "the optimal weight vector
exists but its advantage → 0" looks like when you can only spend it on a shrinking population.** The
citation is no longer decoration: F105 is its finite-`n` measurement, and F105's own corrected law is the
empirical form of the theorem.

---

## §6. TASK-2 RULING — the metric channel (raise top-20 purity for unseen queries)

The tasking asked whether the one channel the kernel argument leaves live — *changing which items land in
the top-20, via the metric itself* — is (1) genuinely left open by my own theory, (2) isomorphic to
F75/NCA, (3) bounded, and (4) if not, what its cheapest falsifier is.

### 6.1 Q1 — is the channel genuinely left open by the theory? **Partly open by the kernel argument, but closed by neural collapse.**

The kernel argument (§4.2 / `LITSWEEP8` §2 C1) closes only *metric-value* changes that leave the retrieved
label tuple fixed. A metric that **changes the tuple** is outside the kernel, so the kernel argument alone
does **not** close it. The coverage oracle (§4.3) does not close it either — coverage bounds *pool
expansion*, not *purity within a fixed pool*.

**But neural collapse does close it, and the closure is exact rather than analogical.** The objective a
purity-raising metric optimises is train-side neighbourhood purity. Its optimum — every train item's
neighbourhood pure — *is* NC1: "cross-example within-class variability of last-layer training activations
collapses to zero, as the individual activations themselves collapse to their class-means"
(Papyan, Han & Donoho, PNAS 117(40):24652-24663, arXiv:2008.08186). **The family's own optimum is the
0.9999 cone this campaign has spent three records fighting.** F47 records the deployed head's train LOO
accuracy at **0.998** — the objective is already at its optimum on its own training signal, with ≤0.002
of headroom there, while held-out error purity sits at 0.12-0.22. **That is a generalisation gap, not an
optimisation gap, and no objective evaluated on the train split addresses it.**

**On Feldman and label-freeness (the tasking's specific challenge).** I wrote that Feldman predicts "no
**label-free** operator can fix it", and metric learning is label-*using*. The theory still binds, and the
reason is not label-freeness: Feldman's claim is that the correct label of a long-tail singleton is not
determined by the training distribution, so achieving it **requires memorising that example**. A
label-using metric can and does memorise the train split — that is exactly what 0.998 is — and
memorisation of a train item does not transfer to an unseen test item drawn from the same one-member
sub-population. **The theory binds label-using metric learning specifically, and it predicts the 0.998
number.**

### 6.2 Q2 — is it isomorphic to F75/NCA? **Yes under the only estimator anyone can actually optimise; formally distinguishable only under a cross-fitted estimator.**

`NCA_FORENSIC_RECON.md:5` states the axis verbatim: *"replace/augment the head's `triplet(m=0.1)+0.5·BCE`
objective with a loss that DIRECTLY optimizes the deployed top-20 signed-cosine kNN vote."* That **is**
"raise top-20 purity", and the recon's own §1 line 35 says so: *"the deployed decision is a rank-decay AND
cosine-magnitude weighted SIGNED top-20 kNN vote… No objective in the campaign ever optimized this
object."* F75 then measured it: **0/8 formal, 7/8 KS-arm-dead**, at ~0.33 GPU-h.

* **Estimated by train LOO** (the only estimator available without held-out labels): **ISOMORPHIC to
  NCA.** Same target, same estimator, same optimum. Re-proposing it is re-proposing F75.
* **Estimated by cross-fitted / episodic held-out folds** (optimise purity on a fold the metric did not
  see): **formally distinguishable** from NCA, which is transductive-on-train. This is the *only*
  non-isomorphic residual I can construct, and I am recording it rather than hiding it.

Being strict as instructed: the residual is **D7-novelty-dead** (head-loss engineering), it is bounded by
§6.3 below, its optimum is still NC1, and the few-shot literature carries a published negative for exactly
this move — Tian et al., *Rethinking Few-Shot Image Classification: A Good Embedding Is All You Need?*
(arXiv:2003.11539, **id/venue UNVERIFIED this sweep**) and Laenen & Bertinetto (arXiv:2012.09831,
**UNVERIFIED**) both report that episodic meta-learning adds nothing over a well-trained embedding.
**I do not recommend it.**

### 6.3 Q3 — is there a non-vacuous bound? **YES — measured here, $0.**

The naive oracle is vacuous (a perfect metric drives purity → 1 and accuracy → 1). The tasking asked for a
principled bound instead. The 30-epoch × 3-seed ZH dev trajectory supplies one, because the head's own
training **is** a purity-raising metric intervention, and we can watch what its purity gains buy.

| epoch | dev top-20 purity toward gold | dev acc | coverage(20) | median sim spread |
|---|---|---|---|---|
| 0 | 0.7207 | 0.8376 | 1.0000 | 9.3e-07 |
| 10 | 0.7897 | 0.8419 | 0.9915 | 6.7e-05 |
| 18 | 0.8015 | 0.8590 | 0.9872 | 3.3e-04 |
| 26 | 0.8267 | 0.8462 | 0.9915 | 2.2e-04 |
| **29** | **0.8250** | **0.8419** | 0.9829 | 2.0e-04 |

> **Over the full training run, held-out neighbourhood purity rises +0.1043 (0.7207 → 0.8250, correlation
> with epoch +0.9237 — the metric channel genuinely transfers) and held-out accuracy rises +0.0043.**
> Observed conversion ratio **d(acc)/d(purity) = +0.0410**. Coverage *falls* −0.0171 over the same run.

**The bound.** Pooled regression of dev accuracy on dev purity over all 90 cells:
slope **+0.0661**, 95 % bootstrap CI **[−0.0221, +0.1637]**, r = +0.1642 (R² = 0.027), per-seed slopes
+0.0884 / +0.0897 / +0.0441. Applying it to the entire remaining purity gap (1 − 0.8250 = 0.1750):

| estimator | implied accuracy from driving held-out purity to a **perfect 1.000** |
|---|---|
| observed whole-run conversion ratio (+0.0410) | **+0.0072** |
| pooled regression point slope (+0.0661) | **+0.0116** |
| **upper end of the 95 % bootstrap CI (+0.1637)** | **+0.0286** |

> **Even at the upper end of the 95 % confidence interval, a metric that achieves PERFECT held-out
> neighbourhood purity buys +0.0286 accuracy — under the +0.030 bar.** That is the non-vacuous bound the
> tasking asked for, and it closes the family on its own terms rather than by analogy.

**Honest limitations of this bound, stated plainly.** (i) R² = 0.027 — the association is weak and the CI
includes zero; the bound is deliberately read at its *conservative* upper end. (ii) ZH dev only, n = 78,
one instrument (CPU re-mint proxy heads), single dev split. (iii) It is an **observational**
within-trajectory association, confounded with everything else that changes during training; it is **not**
a causal estimate of what a purity-targeting objective would achieve. (iv) Linearity is assumed and could
fail near purity = 1. **A purity-targeting objective is not logically forbidden from beating this rate —
but it would have to beat the head's own measured conversion by a wide margin, and F75 measured that it
does not.**

### 6.4 Q4 — instantiation and cheapest falsifier

Not required: the family is isomorphic to F75 under the practical estimator (§6.2) and bounded under the
measured conversion rate (§6.3). **I therefore rule the metric channel CLOSED at the theory level, which
is the outcome the tasking said would close the campaign.** For completeness, the cheapest falsifier of
*my own* ruling, at $0: re-run §6.3's trajectory on HateMM and MHC-EN once a head-space dev instrument
exists, and fire if the upper-95 %-CI implied bound exceeds +0.030 on either. That needs a CPU head
re-mint (~52 s/seed, `ERRPAT_HateMM:6.1`) plus its own fidelity gate — **priced, not spent**, per §1.3.

---

## §7. VERDICT

| gate | bar | measured | verdict |
|---|---|---|---|
| **PARITY-HC** | max\|Δvote\| < 1e-9, pred agreement 1.0000, 7 020 cells | 0.000e+00, 1.0000, 7 020 | **PASS (exact)** |
| **K-HC-3** | decision identity ≥ 0.98 | **1.0000** primary (0/78 × 3 seeds); 0.9989 over 90 cells, min 0.9872 | **PASS — Result A transfers** |
| **K-HC-1** | 1 − coverage(20) < 0.030 | **+0.0171** primary; +0.0256 worst seed; +0.0095 over 30 epochs | **FIRES — CAND-A closed on ZH in the deployed space** |
| **K-HC-2** | 1 − coverage(20) ≥ 0.030 | not met | **does not fire — no re-ranker earns a pregate** |
| **DEG-HC** | epoch-0 discrimination | epoch-0 spread 9.3e-07 < epoch-29 1.95e-04 | **CONTROL FAILED — premise false; head-space identity is forced by collapse, mechanism evidence lives in the raw arena** |

**Bottom line.** The label-tuple theorem **transfers to the deployed head space** on the one dataset with a
zero-test-touch head-space instrument. The re-ranking / candidate-set family stays killed. The metric
channel — the one thing the kernel argument left open — is **isomorphic to F75 under its only practical
estimator, and bounded at +0.0286 under the upper 95 % CI of its own measured conversion rate.**

**Scope, restated so nothing is over-claimed:** ZH dev, n = 78, 3 seeds, CPU re-mint proxy heads, one
split. HateMM and MHC-EN are **out of scope by instrument availability**, declared before any number
existed (§1.3), and their extension cost is priced, not spent. **This is a pregate, not a verdict.**

---

## §8. LEDGER ERRATUM EVIDENCE (stated for the provenance-audit agent; ledger rows NOT edited here)

The certified law-I count: `findings.jsonl` F50 says "5th better-signal-no-conversion instance", F63 says
"SEVENTH", F65's title says "**8th law-I instance**", and **F87 says "9th law-I NOT certified"** — twice,
once in the title and once in the body ("image AMBIGUOUS (9th law-I not certified…)"). F95 calls itself
"the sharpest instance of law-I yet recorded" without claiming an ordinal. **On that evidence the ledger
certifies EIGHT.** litsweep-7 says nine and a figure of ten circulated earlier. I have **not** edited any
historical row; adjudication belongs to the provenance-audit agent.

Separately: the cand-2 datum quoted to me as "a targeted train-LOO move of −0.0538 buying +0.0132" could
not be reconciled. **+0.0132 is verified** (`CAND2_REP2_VERDICT_REVIEW.md:120,182`); the only `0.0538` in
`refine-logs/` is HateMM's **LOO-disagreement rate** in an unrelated curation experiment
(`ERRPAT_HateMM_2026-07-26.md:390`).

---

## §9. FILE MANIFEST AND COMPLIANCE

**Read-only inputs:** `scripts/analysis/errpat_remint_dumps/errpat_zh_remint_seed{0,1,2}.pkl` (**`dev`
records only**); `scripts/analysis/errpat_zh_remint.py` (vote arithmetic, lines 57-96);
`scripts/analysis/p2_rerank_eval.py:45-113` (`CKPT_FILE`, `augment`, `sim_vote`);
`scripts/analysis/p2b_train_benchmark.py`; `src/utils/metrics.py:258-307`;
`logging/Retrieval/**` (existence checks only, no checkpoint loaded).
**Read for context:** `MECHFIX_PREGATE_2026-07-27.md`, `ERRPAT_{HateMM,MHC-ZH,MHC-EN}_2026-07-26.md`,
`NCA_FORENSIC_RECON.md`, `VSW_ASYMMETRY_RECON.md`, `LITSWEEP8_PATHOLOGY_MATCH.md`,
`research-wiki/experiments/exp-encoder-3seed.md`.
**Not opened:** `VSW_PREGATE_RECORD.md` (this pass), `MEMBANK_C4_PREGATE_RECORD.md`,
`scripts/analysis/*membank_c4*`, `STREAMCOMP_FORENSIC_RECON.md`, `INSTRUMENT_VALIDATION_RECON.md`,
`PROVENANCE_AUDIT_2026-07-28.md`, `DISK_FORENSICS_2026-07-28.md`, `PREGATE_DETERMINISM_CLAUSE.md`,
`LITSWEEP7_LANDING_SITE.md`.

**Required statements.** ZERO GPU / SLURM / Modal / training / test-touch spent. **No test-split record
was read** (hard `split == 'dev'` filter + assert). No held-out test metric read or produced. No `state/`
prereg, config, `research-wiki/`, or frozen artifact mutated. No checkpoint loaded, minted, or deleted.
All numbers are closed-form recomputes over banked arrays; nothing trained was re-fitted, so the
confirmed trained-estimator drift defect does not touch any quantity here.
