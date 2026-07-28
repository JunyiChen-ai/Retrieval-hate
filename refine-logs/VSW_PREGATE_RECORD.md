# VSW — $0 pregate on VERIFIER SOFT RE-WEIGHTING of the deployed vote (LITSWEEP-6 C4)

**Date:** 2026-07-28 NZST · **Agent:** vsw-pregate · **Cost: $0** (CPU only, ≤8 threads,
**zero GPU, zero SLURM, zero Modal, zero training of any deployed arm**).
Repo sha at freeze time `b4800d7` (working tree dirty). Env: conda `HateVideo`, python 3.11.8,
numpy 1.26.4, scipy 1.17.1, scikit-learn 1.5.2, torch 2.6.0+cu124 (CPU), faiss.

**Test-split contact: NONE.** The only data files opened by any script in this record are
`data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_<model>.pt`. `dev_seen` and `test_seen` are never
loaded, and no test label appears anywhere in this document.

**Binding design source:** `refine-logs/LITSWEEP6_RELGEN.md` §2 candidate **C4 (VSW)** and §5, read
in full before any code was written. **Machinery source:** `refine-logs/MECHNOV_PAIRVERIFY_PREGATE.md`
(F95) — the frozen verifier, its constants and its arena. **Record-format and control precedents,
both read in full first:** `refine-logs/VGA_PREGATE_RECORD.md` (F97 — the emitter pattern, the
78/78 parity-assert pattern, the permutation-null spec) and `refine-logs/AGGNET_PREGATE_RECORD.md`
(F98 — the degeneracy-control discipline: DEG-A threshold twin, DEG-B fixed-k twin, class balance,
and the "an arm that agrees with a closed lever on ≥95 % of items is that lever in a costume" rule).

---

## §0. FRAMING — THIS IS A DOOR-CLOSER, NOT A GOAL BET

`LITSWEEP6_RELGEN.md` §5 states the conditional this record executes:

> If C1 fails, the relational asset is settled as **analysis-grade only**, and the campaign should
> stop trying to convert it.

**C1 failed** (F97 / `VGA_PREGATE_RECORD.md` §6: K-VGA-1 missed by a factor of three, K-VGA-2 gave
p = 0.8706 / 0.5174 / 0.9751 for the primary arm, and K-VGA-3 **fired** — the F47 features beat the
verifier features on 3/3 datasets). C4 (VSW) is the one arm of the litsweep-6 relgen sweep that was
**never tasked and never run** — recorded as such in `VGA_PREGATE_RECORD.md` §7.7 and in F97's
`ban_scope`, which also notes that "the emitter now exists so its λ-sweep exchange-rate curve is a
much cheaper rider than the sweep record priced it".

It is being run **as the arm that closes the aggregation axis arithmetically**, not as a lever.
The sweep record prices it itself:

> **Honest kill risk — near-certain death as a performance bet.** P(clear K-VSW-1 on ≥2 datasets)
> ≈ **2 %**. … Recommendation: run as a rider on C1, budget it as analysis, never as a lever.

**The expected outcome of this record is a KILL that closes an axis.** The deliverable that carries
the value is K-VSW-2, which the sweep record calls "a diagnostic that cannot fail": it upgrades the
paper's law-I datum from the **two-point** exchange-rate read F95 currently carries (max and
mean-top-3) to a **full curve over aggregation sharpness**. A positive on K-VSW-1 would be a
surprise against a pre-registered 2 % prior and would require escalation to a formal prereg with
independent review, not a promotion inside this record.

---

## §1. WHAT IS UNDER TEST

The deployed decision (`src/utils/metrics.py:262-301`, `src/model/evaluate_rac.py:405-465`, replayed
bit-faithfully by the F89-frozen `mechfix_ops.deployed_vote`) is

```
v = Σ_i (2·lab_i − 1)·cos_i·w_i / Σ_i w_i ,   top-20 own-train neighbours,  w = [20,19,…,1]
predict 1 iff v ≥ 0
```

`LITSWEEP6_RELGEN.md` §2 C4 specifies the treatment, quoted verbatim:

> **Mechanism.** Keep the deployed rank-weighted sum over the top-20 exactly as it is, and multiply
> each neighbour's rank weight by a monotone function of its verifier score, with an interpolation
> coefficient λ such that **λ=0 reproduces the deployed vote bit-exactly**. The deployed vote is
> first-order (labels × fixed rank weights); the verifier supplies second-order information (how
> genuinely each retrieved item *relates* to this query), which is structurally the OW/ISP move.
> Data flow: `S[q, ·] → per-neighbour multiplier → λ-blend with rank weights → same sum → label`.

Operationalised:

```
v(λ) = Σ_i (2·lab_i − 1)·cos_i·w_i·m_i(λ) / Σ_i w_i·m_i(λ) ,   m_i(0) ≡ 1
predict 1 iff v(λ) ≥ 0
```

where `m_i(λ) ≥ 0` is a **monotone non-decreasing function of `p_i`**, the F95 pair verifier's
`P(same-class)` score for the pair `(query, i-th deployed neighbour)`. Retrieval, the key space,
`k = 20`, the candidate set, the neighbour labels `lab_i`, the cosines and the decision threshold are
all **untouched**; only the weight each already-retrieved neighbour receives changes, and it changes
per query as a function of the verifier's relational judgement.

**Distinctness, and where it is weakest — carried verbatim from the sweep record rather than
finessed** (`LITSWEEP6_RELGEN.md` §2 C4):

> It does not change the aggregation shape, so the −0.029/−0.044 control-2b cost is zero at λ=0 and
> grows continuously. It is not a change of k, so F94's measured content does not touch it. **But
> F94's ban text says in terms: "no truncation *or re-weighting* of the retrieved list reaches it."**
> That clause was generalised from F49/F66/F86, not measured for *learned* weights — so this
> candidate does not violate a measurement, but it does contradict a stated generalisation, and that
> must be flagged to the reviewer up front rather than finessed. F63 is a second charge: diffusion
> over the frozen graph is monotone-negative in α, and while F63 explicitly does not price learned
> edge weights, a one-hop verifier-weighted vote is close enough that the burden of proof is ours.

**A third charge, added by F98 after the sweep record was written and binding on this pregate.**
`AGGNET_PREGATE_RECORD.md` §7.1(a) closes "any learned re-weighting, soft-mixture-over-k, attention,
or gating **over the deployed top-20**" on the grounds that C3's 1316-parameter conditional
aggregator spans that class and converged to a threshold shift (DEG-A 0.9570) and a fixed k=15 vote
(DEG-B 0.9610). **VSW is inside the functional form of that closure but outside its information
content**: C3's input was the `(cosine, label)` profile of the top-20 and contained **no verifier
feature by construction** (F98 §1.3, binding restriction); VSW's multiplier is a function of the
**trained relation score**, which F95 control 1 measured to carry ordering information the cosine
does not (+0.1572 / +0.2302 / +0.1785 within-query AUC, 5/5 fold signs, 18/18 cells). VSW is
therefore the *one* member of the re-weighting family F98's closure does not price — and it is being
run precisely so that the closure becomes exhaustive rather than nearly so. **DEG-A and DEG-B are
imported from F98 unchanged and at the same 0.95 threshold**, and a firing kills this arm exactly as
it killed C3.

---

## §2. FROZEN BARS

### 2.1 K-VSW-1 and K-VSW-2, quoted **verbatim** from `LITSWEEP6_RELGEN.md` §2 C4

> **Frozen kill bar.** **K-VSW-1:** net ≥ +0.030 on ≥2 of 3 datasets at a λ selected on inner folds.
> Given the exchange-rate law I expect this to fail. **The reason to run it anyway is K-VSW-2, a
> diagnostic that cannot fail:** sweep λ from 0 to 1 and record the exchange rate as a function of
> aggregation sharpness. F95 measured the exchange rate at exactly **two** points (max and mean-top-3).
> A full curve either finds a sharpness regime where the rate exceeds 1.2 — which no cell in a 36-cell
> battery reached — or it shows the rate is bounded below 1 across the entire continuum, which
> **closes the aggregation axis arithmetically** and is a materially stronger law-I datum than the
> two-point read F95 currently carries into the paper.

### 2.2 K-VSW-0 — the pregate-level bar (declared here, weaker than K-VSW-1)

Because a result between the house *interest* threshold and the house *decision* bar must be
**visible rather than hidden** (the F98 §2.2 convention):

> **K-VSW-0 (interest threshold):** pooled item-disjoint Δacc ≥ **+0.010** on ≥1 dataset, with
> **5/5 fold signs Δ ≥ 0 and ≥3/5 strictly positive**, at a λ selected on inner folds.

This is the F95 control-2 bar form verbatim. Clearing K-VSW-0 while missing K-VSW-1 is **not** a
pass and licenses nothing; it is reported so the boundary is legible.

### 2.3 The escalation condition for K-VSW-2, declared before any curve exists

K-VSW-2 is a diagnostic, so it has an *outcome*, not a pass/fail. The sweep record names the two
possible outcomes; they are operationalised here before the run:

* **Outcome (a) — sharpness regime found.** Some λ in the declared grid, on the PRIMARY family, with
  **≥ 20 changed decisions** (so the rate is not a small-count artefact), attains **exchange rate
  > 1.2** on ≥2 of 3 datasets. ⇒ the aggregation axis is **not** closed; escalate to a formal prereg
  with independent review. **No promotion is made inside this record under any outcome.**
* **Outcome (b) — bounded below the law.** No such λ exists. ⇒ **the aggregation axis is closed
  arithmetically across the sharpness continuum**, and the paper's law-I datum becomes a curve.

The ≥20-changed-decisions guard is declared now because at small λ only a handful of decisions flip
and `fixed/broken` on 3 items is noise, not a rate. Rates at <20 changed items are reported but
cannot trigger outcome (a).

### 2.4 Anti-shopping rule

Three multiplier families are declared (§3.3). **K-VSW-1 must be met by one and the same family on
≥2 of 3 datasets**; a pass assembled from different families on different datasets does not count.
(The VGA §2.5 rule, transplanted.)

---

## §3. FROZEN DESIGN

### 3.1 Arena — the F95 harness verbatim

Banked **RAW fused** encoder key space (`L2norm(concat(L2norm(img), L2norm(text)))`, 7168-d),
**train split only**, item-disjoint `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` —
`mechnov_pairverify.py`'s frozen `K_FOLDS` / `FOLD_SEED`. Verifier = the F95 **PRIMARY** cell:
fused × **MLP** × the frozen constants (`PCA_DIM=256`, `PCA_SOLVER='full'`, `PAIR_FIT_CAP=150000`,
`PAIR_SUBSAMPLE_SEED=0`, `MLP_HIDDEN=128`, `MLP_EPOCHS=30`, `MLP_BATCH=1024`, `MLP_LR=1e-3`,
`MLP_WD=1e-4`, `MLP_SEED=0`). The **logistic** verifier arm is **not** used: F95 control 4 fired on
it (collapse to positive rate 0.0237-0.0604).

Feature caches, train split only: HateMM `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` (n=744,
bank pos-rate **0.4005**), MHC-ZH `Qwen2.5-VL-7B-Instruct-LoRA_HF` (n=579, **0.3109**), MHC-EN
`Qwen2.5-VL-7B-Instruct_HF` (n=549, **0.3060**).

**Inherited limitation L1, stated once and assumed throughout** (F95 §6, restated in
`LITSWEEP6_RELGEN` §0(b) and in both sibling records): this is a raw-space, train-split arena, not
the deployed head space and not test. A raw-space null does not logically entail a head-space null,
and the campaign's history (F47, F66, F89) is that raw-space oracles do not survive that trip. This
cuts *against* a negative verdict and is stated first for that reason.

### 3.2 Reuse, not rewrite — the frozen modules are imported unmodified

Per `LITSWEEP6_RELGEN.md` §0(a) and the VGA precedent, the F95 arms module is **not edited**. A new
script imports it and asserts both frozen sha256 values before running:

* `scripts/analysis/mechnov_pairverify.py` — sha256
  `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` (F95), used for `load_cache`,
  `build_space`, `l2n`, `all_unordered_pairs`, `pair_features`, `fit_mlp`, `predict_mlp`, `acc`,
  `DATASETS` and **every** frozen constant.
* `scripts/analysis/mechfix_ops.py` — sha256
  `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` (F89, 15/15 floor-parity gates
  at 4 dp), used for `deployed_vote` and `macro_f1`.

**One efficiency deviation, disclosed** (identical to VGA §0): F95 scored every
(held-out × in-fold) pair; this script scores only the **20 deployed neighbours** per query, plus the
**20 nominated candidates** (top-10 per class) needed to reproduce F95's own parity quantities. The
frozen MLP is a deterministic pointwise function of its fitted parameters, and the fit set, PCA,
standardisation statistics, seeds and fold assignment are bit-identical, so scores on any subset are
bit-identical. This is **asserted, not argued** — see §3.6.

### 3.3 The multiplier families — declared in full, no post-hoc family

Let `p_i ∈ (0,1)` be the frozen MLP verifier's `P(same-class)` for the pair `(query, i-th deployed
neighbour)`, `i = 1..20` in deployed rank order; clipped to `[1e-12, 1]` for numerical safety.
`v(λ)`'s sign is invariant to any positive rescaling of the whole weight vector, so each family is
written in a scale-free form.

| id | multiplier `m_i(λ)` | λ grid | status |
|---|---|---|---|
| **`pow`** | `(p_i / max_j p_j)^λ` | `0, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 4096` and the exact endpoint `∞` | **PRIMARY** |
| `exp` | `exp(λ·(p_i − max_j p_j))` | `0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024` | SECONDARY |
| `lin` | `(1 − λ) + λ·(p_i / mean_j p_j)` | `0, 0.1, 0.2, …, 1.0` | SECONDARY |

**Why `pow` is PRIMARY.** It is the only family that spans the **entire aggregation-sharpness
continuum** the K-VSW-2 diagnostic asks for: at `λ = 0` it is the deployed rank-weighted vote
bit-exactly; over `λ ∈ (0, 1]` it contains the literal "λ from 0 to 1" interpolation the sweep record
names; and as `λ → ∞` it converges to *"emit the label of the single best-verified neighbour in the
deployed top-20"*, i.e. the F95 **max** endpoint transplanted onto the deployed candidate set. The
`λ = ∞` cell is computed exactly (arg-max indicator) rather than by extrapolation, so the curve has a
closed right endpoint. `lin` is the literal reading of the sweep's "interpolation coefficient" and is
retained as a SECONDARY family for exactly that reason; `exp` is a monotone map of `p` rather than of
`log p` and prices whether the curve's shape is an artefact of the link function.

**Every family satisfies `m_i(0) ≡ 1` identically**, which is what makes §3.6's parity gate a hard
assert rather than a tolerance. The `exp` family is written against the **row max** rather than the
row mean purely for overflow safety at λ = 1024; `v`'s *sign* is invariant to a positive global
rescaling of the weight vector, so the two forms emit identical decisions at every λ. All three
families are verified numerically to be **monotone non-decreasing in `p_i`** at every declared λ
before the freeze (§3.8).

### 3.4 λ selection — inner folds only, never the evaluated fold

Every train item has exactly one record — its deployed top-20 (indices, cosines, neighbour labels)
and the 20 verifier scores — produced in the F95 fold in which **it** was held out, against the bank
of the other four folds. That record is the unit of everything below.

For outer fold `f` (the frozen F95 assignment): the fitting pool is the items in folds ≠ `f`. An
**inner `StratifiedKFold(n_splits=5, shuffle=True, random_state=17)`** is run inside that pool (the
VGA §2.4 / F98 §1.4 inner constants); for each λ in the family's grid, the mean accuracy over the
five inner held-out portions is computed, and **λ\* = argmax**, with **ties broken toward the
smallest λ** (i.e. toward the deployed rule — F98's "ties break toward the deployed rule"). λ\* is
then applied to fold `f`. **λ is never chosen on the fold it is evaluated on, and no held-out item
contributes to any selection.** λ\* is reported per fold.

*Residual coupling, disclosed (limitation L3, inherited verbatim from VGA §2.4):* a fitting-pool
item's verifier scores came from the F95 fold in which **it** was held out, and that fold's verifier
was fitted on a set that included fold-`f` items. The λ selection therefore never sees fold `f`'s
outcomes, but the features it selects on were produced by verifiers that saw fold `f`'s items.
Removing this would require a doubly-nested verifier (25 fits per dataset instead of 5), which the
spec does not ask for. The bias direction would, if anything, **flatter** the treatment arm.

### 3.5 Arms and controls — all declared here

| id | status | what |
|---|---|---|
| `deployed` | FLOOR | `mechfix_ops.deployed_vote` over the same fitting-fold bank |
| `VSW_pow` | **PRIMARY** | λ\* selected on inner folds over the `pow` grid |
| `VSW_exp`, `VSW_lin` | SECONDARY | same, over their own grids |
| `VSW_pow@λ` … | K-VSW-2 curve | every λ in every grid, evaluated on all held-out items — **fixed λ, no selection**, so the curve is a property of the operator and not of a selector |
| `THRESH_best` | **DEG-A** | deployed vote with a single **global threshold** τ chosen on the fitting-fold items (exact optimum over all thresholds, F98's `best_threshold`) |
| `FIXK_{1,2,3,5,7,10,15,20}` | **DEG-B** | the eight F94 grid profiles `[k..1, 0×(20−k)]` over the identical deployed top-20 |
| `CTRL_cos@λ` | **DEG-D** | the identical `pow` family with `p_i` **replaced by `cos_i`** — a re-weighting that uses **no verifier information at all**, only extra sharpening toward rank 1 |
| `ORACLE_lambda` | CEILING | the best λ chosen on the **held-out** fold. Reported as a ceiling only; never selects an arm, never carries a pass |

### 3.6 Mandatory controls — declared before the run; any firing is a KILL

1. **PARITY-λ0 (hard assert, aborts the run).** For every dataset × fold, the VSW vote engine
   evaluated at **λ = 0** must reproduce `mechfix_ops.deployed_vote` **bit-exactly** — identical
   neighbour index matrix, identical cosine matrix, `np.array_equal` on the vote vector and on the
   prediction vector — and the pooled accuracy must match at 4 dp. A single mismatch aborts before
   any treatment number is written. (`m_i(0) ≡ 1` exactly, and `w_i · 1.0 == w_i` exactly in IEEE
   754, so this is an exact-equality gate, not a tolerance.)
2. **F95 PARITY (hard assert, aborts the run) — TWO TIERS, 26 quantities per dataset, 78 gates
   total, exactly the key set VGA asserted 78/78 on.** The 26 keys are `acc / mF1 / posrate` of the
   deployed vote, of the cosine-shape control (2b), of `mlp_max` and of `mlp_mean3`;
   `n_deployed_wrong`; `n_pathology_pop`; `median_sc_rank_all`; `median_sc_rank_deployed_wrong`; and
   `fixed / broke / net / exchange_rate / pathology_fixed` for both aggregations. They are split by
   **what they depend on**, because §4.1 measures that those two things are not equally reproducible:
   * **Tier 1 — the 10 closed-form quantities** (retrieval, the deployed vote, the cosine-shape
     control, the ERRPAT rank statistics) are asserted against the **recorded** F95 cell in
     `mechnov_pairverify_{hatemm,zh,en}_OUT.json`. This gates the claim *"the regenerated arena **is**
     the F95 arena."*
   * **Tier 2 — the 16 quantities that depend on the torch-fitted MLP verifier** are asserted
     against `vsw_f95anchor_{ds}_OUT.json`, produced by `--stage anchor`, which calls the **frozen
     F95 module's own `run_space` unmodified** (sha asserted) in **this** session. This gates the
     claim *"scoring only the 20 nominated pairs reproduces the frozen module's full-eval-matrix
     scoring exactly"* — the efficiency deviation of §3.2, which is the only thing this pregate is
     entitled to be checked on.
   * The **recorded-vs-anchor difference** on the 16 trained quantities is **reported as a measured
     drift table (§4.1), never asserted away.**
3. **DEG-A (threshold twin).** Pooled agreement between `VSW_pow`'s held-out decisions and
   `THRESH_best`'s. **≥ 0.95 ⇒ FIRES ⇒ KILL.** (F96/RESTRANS measured 95-99 % here; F98 measured
   0.9570 HateMM / 0.9508 MHC-EN.)
4. **DEG-B (fixed-k twin).** `max_k` pooled agreement between `VSW_pow` and `FIXK_k` over the eight
   F94 grid profiles. **≥ 0.95 ⇒ FIRES ⇒ KILL.** (F98 measured 0.9610 at k=15 / 0.9964 at k=20.)
   *Note declared in advance:* `FIXK_20` **is** the deployed rule, so an arm whose inner CV selects
   λ\* = 0 on every fold agrees with it at 1.0000 and DEG-B fires at the ceiling. That is the correct
   verdict form, not a technicality — an arm that falls back to the deployed rule **is** a member of
   the closed family (F98 recorded exactly this on MHC-EN at 0.9964).
5. **CLASS BALANCE.** Positive rate of the emitted decision must lie within **0.10** of the bank
   positive rate (0.4005 / 0.3109 / 0.3060). Outside ⇒ the permutation nulls for that cell are
   **VOID** and its Δ is a collapse artefact, per F95 control 4 and VGA K-VGA-4.
6. **DEG-D (cosine twin), reported not auto-killing.** `CTRL_cos` at the same λ grid. Firing
   condition, declared now: if `CTRL_cos` **matches or beats** `VSW_pow` at the selected λ on ≥2 of 3
   datasets, then the verifier contributes nothing to the re-weighting beyond generic sharpening
   toward rank 1, and the "second-order information" claim in §1 is refuted by measurement. This is
   the VSW analogue of K-VGA-3 and is treated as a distinctness verdict (F98's DEG-C convention).

### 3.7 Permutation null — mandatory

`N_PERM = 200`, `PERM_SEED = 12345` (the VGA constants). **The verifier's fit targets are shuffled
within the fitting pool only**: for each fold, the fitting-fold **item labels** are permuted by
`RandomState(PERM_SEED)` and the pair targets `y = 1[lab̃_i == lab̃_j]` are derived from the permuted
labels, so the pair-target structure and its class balance are preserved. **The bank labels `lab_i`
used in the vote, the retrieval, the cosines, the deployed floor and every held-out gold label are
untouched.** The full pipeline — verifier fit at the frozen budget, scoring of the deployed top-20,
inner-fold λ selection, held-out evaluation — is re-run per draw, i.e. the null is at the **same
fitting budget**, as the spec requires. Run on the **PRIMARY cell** (fused × MLP × `pow`); because a
draw yields a complete `n × 20` verifier score table, the null distribution is obtained for **every λ
and every family at no extra fit cost**, so the whole K-VSW-2 curve carries a null.

Reported `p = (1 + #{null ≥ observed}) / (N_PERM + 1)`; significance at `p < 0.05`; resolution
1/201 = 0.0050.

**Why this null is expected to be informative here, unlike F98's.** `AGGNET_PREGATE_RECORD.md` §6
recorded that its null was uninformative because **no null draw could reach the arm's floor
fallback** — 0 of 300 draws reached zero, so anything that fell back to the deployed rule scored
p = 0.0099 automatically. Here the fallback **is** reachable by the null: λ = 0 is in every grid and
is the tie-break winner, so a null draw whose verifier carries nothing will select λ = 0 and score
Δ = 0.0000 exactly, the same value the observed arm would score. The null therefore tests the right
proposition — *does the verifier's relation information contribute anything to the re-weighting* —
and non-significance will be an honest non-significance, as it was in VGA §4.2.

### 3.8 Machinery validity — synthetic positive control, run **BEFORE** the freeze

Per the tasking and the VGA §2.7 / F98 §2.5 precedent, the harness is exercised on **synthetic data
only** before the sha is frozen, with two arms:

Geometry, identical in both arms: `n = 350`, `d = 106` — **5 shared nuisance factors of sd 10 that
dominate the cosine** (their contribution to the inner product has sd `√5·10² ≈ 224`), 100 isotropic
noise dimensions, and **one noiseless class coordinate** of magnitude `sig = 1.5`. The planted term
is only `sig² = 2.25` against that 224, so the deployed top-20 retrieval is essentially class-blind
(tail-enrichment ratio `≈ exp(2·sig²·t/σ²) ≈ 1.03`), while the class coordinate's variance (2.25)
still exceeds the isotropic coordinates' (1.0), so the PCA keeps it as its own component and a
relation function on `[|z−z'|, z⊙z']` can read its **signed product**.

* **Arm A (signal planted).** The class coordinate carries the true label, so verifier re-weighting
  of the deployed top-20 should repair the vote. **Required: a clear positive Δacc and a significant
  permutation p.**
* **Arms B0 and B (nothing learnable).** The class coordinate carries a **label-independent**
  grouping and the labels are drawn independently of every feature, so the verifier's fit target
  `1[lab_i == lab_j]` is unlearnable. They differ **only** in the class rate, and that difference is
  the point:
  * **B0, rate 0.40 (imbalanced)** is the F98 §2.5 arm-B lesson made concrete. A re-weighting arm's
    function class contains a drift toward the neighbourhood **label ratio**: the verifier memorises
    its own fitting-fold items, so `p(q, bank_j)` drifts toward `P(class of q == lab_j) ≈` the class
    **prior**, which upweights majority-class neighbours. Where the deployed vote sits *below* the
    majority rate, that drift buys free accuracy with nothing learnable present — **and the
    permutation null does not absorb it**, because shuffling the fit targets destroys the
    correspondence with the real bank labels the vote consumes. B0 is therefore expected to return a
    **spurious positive with a significant p**, and the **frozen class-balance control (§3.6.5) must
    catch it**, since the drift is precisely a collapse away from the bank rate. **B0 exists to
    demonstrate that the declared control does its job**; it is not a failure of the harness.
  * **B, rate 0.50 (balanced)** removes the prior asymmetry the drift feeds on. **This is the arm on
    which an honest null is REQUIRED: a non-significant permutation p.**

  On the real datasets the deployed vote sits far *above* the base rate (0.8441 / 0.8480 / 0.7796 vs
  bank 0.4005 / 0.3109 / 0.3060), so this drift can only *cost* accuracy there — but it is named here
  before the run so that any real-data positive accompanied by a class-rate collapse is read as the
  artefact it would be.

If the harness cannot return a positive on arm A, the harness is fixed **before** any real cell runs.
The self-test uses `N_PERM_SELFTEST = 60` (p-resolution 1/61 = 0.0164 — the VGA self-test precedent);
this is disclosed rather than presented as the full budget. **No real-dataset number is computed
before the sha256 below is frozen.**

**Measured, before the freeze** (`vsw_selftest_OUT.json`; re-run and reproduced bit-identically under
every sha, §3.9). The multiplier check passes first: all **48 (family × λ) cells** are monotone
non-decreasing in `p` on sorted probes and `m(0) == 1.0` **exactly**. PARITY-λ0 is 18/18 on every arm.

| arm | deployed acc | **Δacc** | fold signs | ER (fixed/broken) | pos-rate vs bank | **class balance** | neighbour-score AUC cos → verifier | **permutation p** | null mean ± sd (max, frac ≥ 0) |
|---|---|---|---|---|---|---|---|---|---|
| **A — signal planted** | 0.6143 | **+0.2771** | `+++++` | 9.8182 (108/11) | 0.3457 vs 0.4143 | **PASS** (0.0686) | 0.5117 → **0.8477** | **0.0164** | −0.0062 ± 0.0095 (max +0.0143, 45 %) |
| **B0 — noise, imbalanced** | 0.5171 | +0.0657 | `+++++` | 1.9583 (47/24) | 0.1229 vs 0.3857 | **FAIL → nulls VOID** (0.2628) | 0.4894 → 0.5290 | 0.0164 | +0.0056 ± 0.0277 (max +0.0629, 58 %) |
| **B — noise, balanced** | 0.4886 | **+0.0057** | `-+++-` | 1.0556 (38/36) | 0.5029 vs 0.5000 | **PASS** (0.0029) | 0.5068 → 0.5056 | **0.5410** | +0.0071 ± 0.0299 (max +0.0657, 63 %) |

**All three requirements are met.** (i) **The harness is not structurally incapable of returning a
positive**: arm A returns +0.2771 at the smallest p the 60-draw design can produce, with 5/5 fold
signs and an exchange rate of 9.8182, driven by a verifier whose neighbour-score AUC is 0.8477
against the cosine's 0.5117. (ii) **It returns an honest null when nothing is learnable**: arm B
gives +0.0057 at **p = 0.5410**. (iii) **The predicted prior-fallback artefact appears exactly where
it was predicted to and is caught by exactly the control declared to catch it**: arm B0's spurious
+0.0657 at p = 0.0164 comes with a positive rate of 0.1229 against a bank rate of 0.3857, and the
frozen class-balance control **fires and voids it**. A null below is therefore a property of the
data, not of the code.

One further property worth recording, because it is the one F98 §6 lacked: **the null can reach the
arm's fallback.** λ = 0 is in every grid and is the tie-break winner, so a draw whose verifier
carries nothing scores exactly 0.0000 — and **45-63 % of null draws land at or above zero** in the
self-test. F98's null could not reach its floor (0 of 300 draws), which made its p = 0.0099
automatic and uninformative. This null does not have that defect.

> **Disclosed rather than silently fixed: one superseded pre-freeze self-test construction.**
> The first version of arm A planted the class signal as a *small mean shift added to a noisy
> coordinate* in a 64-d problem (`shift = 0.55`, `n = 300`). It was measured, **before the freeze and
> before any real-dataset number existed**, to be too weak to exercise the mechanism: verifier
> neighbour-score AUC **0.5478** against cosine **0.5167**, Δacc **+0.0067**, permutation
> **p = 0.0820** — i.e. the harness did *not* demonstrate that it can return a positive, which is
> precisely the condition under which the tasking requires the harness to be fixed rather than run.
> The construction above replaces it (noiseless concentrated signal coordinate, more nuisance
> dimensions). **No bar, no arm, no constant of the treatment changed** — only the synthetic
> generator used to validate the machinery. The superseded numbers are recorded here so the
> replacement is visible rather than hidden.

### 3.9 Frozen script sha256

| path | sha256 |
|---|---|
| `scripts/analysis/vsw_pregate.py` | `ba9982dba98fb14dd53297ac6c087f2e1a4aa068490879d2121c50cf1f932eea` |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` (F95, imported unmodified, sha asserted at run time) |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` (F89, imported unmodified, sha asserted at run time) |

`scripts/analysis/vsw_pregate_report.py` is **reporting only** — it merges the per-stage JSON and
re-reads every number at 4 dp; it computes no arm and is not part of the freeze.
`scripts/analysis/vsw_drive.sh` is **orchestration only** — it re-invokes the frozen script after a
reap (§3.10) and changes no arm.

> **Disclosed rather than silently fixed: TWO re-freezes, both forced by a gate firing, both before
> any treatment number existed anywhere.** No VSW arm was computed under any earlier sha — each run
> aborted at a gate, which is precisely what the gates are for. The self-test was re-run and
> **re-verified bit-identically under every sha** (arm A +0.2771 p = 0.0164; arm B0 +0.0657
> p = 0.0164 class-balance VOID; arm B +0.0057 p = 0.5410), and **all real-data checkpoints were
> deleted and recomputed from scratch after each re-freeze**, so every number in §4 onward was
> produced by `ba9982db…932eea` and by nothing else.
>
> **Re-freeze 1 — `9dbd35de…c2c2bf` → `bf7d16f1…e5a1`: a defensive assert that was not a declared
> control.** The first frozen version carried a sanity assert — *all deployed top-20 cosines > 0* —
> that is **not** among the §3.6 controls. It fired on HateMM and aborted. The cause was measured,
> not guessed: **exactly one train item has a zero-norm key** (`mechnov_pairverify.l2n` leaves an
> all-zero row as the zero vector), so all 20 of its top-20 inner products are exactly `0.000000` —
> **20 of 14 880 cosine cells, 0.13 %, one item**. That is not a defect: the deployed rule itself
> votes `0/210 = 0` there and predicts 1, and this harness replays it bit-exactly. The assert became
> a **reported diagnostic** (`cos_diagnostic`, §4.1). Diff: one abort → one counted report.
>
> **Re-freeze 2 — `bf7d16f1…e5a1` → `ba9982db…932eea`: the F95 parity gate fired, and it was
> right to.** Under `bf7d16f1`, HateMM returned **F95-PARITY 11/26**: every closed-form quantity
> matched the recorded F95 cell exactly, and **all 15 quantities depending on the torch-fitted MLP
> verifier missed** (e.g. `acc_mlp_max` recorded 0.8401, emitted 0.8468). Four diagnostics were run
> before touching the harness, and they exonerate it — see §4.1, where the finding is reported as a
> campaign-level erratum. The gate was therefore **split into the two tiers described in §3.6.2**,
> and `--stage anchor` added to produce the tier-2 reference by re-running the frozen F95 module
> unmodified. **No bar, arm, constant, seed, multiplier family, λ grid, nesting rule, null spec or
> degeneracy control changed in either re-freeze** — the diff is entirely in what the *gates*
> compare against and in what is *reported*.

### 3.10 Reap resilience — process boundaries only, no arm changed

The login node SIGTERMs sustained non-SLURM CPU processes: this pregate's **first self-test attempt
was killed at 70 s wall / ~9 min CPU with `exit 143`**, exactly the failure `MECHNOV_PAIRVERIFY_PREGATE`
§3 recorded for F95 and `LSMI_GATE_RECORD` §2.7 characterised in general. The harness was therefore
restructured **before the freeze and before any real-data number existed** so that every unit of work
— one fold's arena, one (fold × permutation draw) verifier fit, one draw's evaluation — is serialised
to `scripts/analysis/vsw_ckpt/` the instant it completes, via a temp file plus `os.replace` so a reap
cannot leave a half-written file; a re-run skips completed units, and a retry loop drives the stage to
completion. **Every draw's RNG is seeded from `(PERM_SEED, draw, fold)`, so the draw sequence is
identical to an uninterrupted run.** This is the `mechnov_pairverify_runner.py` (F95) and LSMI
per-draw-checkpoint precedent: same arms, same constants, same seeds, same fold assignment, same
budget — only the process boundary differs.

Gate order: §0-§3 written and the bars above frozen → synthetic self-test → sha frozen into the table
→ F95 parity gate (78 cells) + PARITY-λ0 gate (per fold) → arms, curve and degeneracy controls →
permutation null → verdict. Machine output: `scripts/analysis/vsw_pregate_OUT.json`. **Every number
in §4 onward is re-read from that JSON at report time at 4 dp, never transcribed from a run log.**

<!-- ============ EVERYTHING BELOW THIS LINE WAS WRITTEN AFTER THE RUN ============ -->

---

## §4. GATES

Every number from here on is re-read at report time from `scripts/analysis/vsw_pregate_OUT.json`
via `scripts/analysis/vsw_pregate_report.py` at 4 dp. All declared arms ran; none was dropped.

### 4.1 PARITY-λ0 — **54/54 PASS**, bit-exact on every family × fold × dataset

18 gates per dataset (3 multiplier families × 5 folds bit-exact + 3 pooled-accuracy-at-4dp),
**18/18 on HateMM, MHC-ZH and MHC-EN**. At λ = 0 the VSW vote engine returns a vote vector and a
prediction vector **`np.array_equal` to `mechfix_ops.deployed_vote`** on every fold. The treatment
therefore differs from the deployed floor **in the neighbour weighting and in nothing else**, and
every Δ in this record is a paired Δ against a floor the harness reproduces exactly.

Reproduced floors (pooled over all held-out train items, each held out exactly once):

| dataset | n | bank pos-rate | deployed acc | deployed mF1 | deployed pos-rate | per-fold deployed acc |
|---|---|---|---|---|---|---|
| HateMM | 744 | 0.4005 | **0.8441** | 0.8419 | 0.4812 | 0.7987, 0.8322, 0.8926, 0.8255, 0.8716 |
| MHC-ZH | 579 | 0.3109 | **0.8480** | 0.8281 | 0.3489 | 0.8534, 0.8534, 0.9224, 0.8017, 0.8087 |
| MHC-EN | 549 | 0.3060 | **0.7796** | 0.7286 | 0.2605 | 0.7091, 0.7545, 0.8091, 0.8182, 0.8073 |

### 4.2 F95 PARITY — **78/78 PASS** (tier-1 30/30, tier-2 48/48)

26 gates per dataset, exactly the key set VGA asserted 78/78 on.
**Tier 1 (10 closed-form quantities per dataset, vs the recorded F95 cell): 30/30 PASS.**
**Tier 2 (16 trained-arm quantities per dataset, vs the same-session anchor): 48/48 PASS.**
Beyond the asserted set, the per-fold PCA explained-variance sequences reproduce F95 fold for fold
(HateMM `[0.9459, 0.9447, 0.9442, 0.9454, 0.9447]`, ZH `[0.9627, 0.9628, 0.9626, 0.9619, 0.9627]`,
EN `[0.9546, 0.9546, 0.9548, 0.9548, 0.9549]`), as do the fitted-pair counts
(150 000 / 106 953 / 96 141).

**Cosine diagnostic** (§3.6, re-freeze 1): HateMM `20 / 14 880` neighbour cosines ≤ 0 on **one** item,
`min_cos = 0.000000`, **1 zero-norm key**; MHC-ZH `0 / 11 580`, `min_cos = 0.790597`, 0 zero-norm
keys; MHC-EN `0 / 10 980`, `min_cos = 0.843110`, 0 zero-norm keys.

### 4.3 ERRATUM (campaign-level): the frozen F95 module's TRAINED arm is not reproducible across sessions; every closed-form quantity is

This was surfaced by the tier-1/tier-2 split and is reported because it changes how F95 and F97
numbers may be cited, not because it changes anything in this record.

**Measured fact.** `scripts/analysis/mechnov_pairverify.py` (sha `77b0defd…b7240d`), executed
**unmodified** today via `--stage anchor` — same node `foscsmlprd01`, same conda env `HateVideo`,
same torch 2.6.0+cu124 / numpy 1.26.4 / MKL 2024.2 (package directories unchanged since 2026-03-27),
same feature caches, same seeds — reproduces **every closed-form quantity of its own recorded cell
exactly at 4 dp** and **fails to reproduce its torch-fitted MLP arm on 44 of 48 trained quantities**:

| dataset | `acc_deployed` | `acc_cos_shape` | `acc_mlp_max` recorded → **anchor** | `mlp_max` fixed/broke recorded → **anchor** | `mlp_max` ER recorded → **anchor** | trained quantities drifted |
|---|---|---|---|---|---|---|
| HateMM | 0.8441 ✓ | 0.8024 ✓ | 0.8401 → **0.8468** | 54/57 → **55/53** | 0.9474 → **1.0377** | **15/16** |
| MHC-ZH | 0.8480 ✓ | 0.8187 ✓ | 0.8014 → **0.8152** | 31/58 → **32/51** | 0.5345 → **0.6275** | **16/16** |
| MHC-EN | 0.7796 ✓ | 0.7359 ✓ | 0.7650 → **0.7614** | 49/57 → **49/59** | 0.8596 → **0.8305** | **13/16** |

**Four diagnostics were run before the harness was touched, and all four exonerate it.**
1. `fit_mlp` is **bit-deterministic within a process**: fitting twice on identical inputs gives
   `max |Δ score| = 0.0`.
2. **Call ordering is irrelevant**: fitting before vs after the faiss `deployed_vote` call and the
   numpy GEMM gives `max |Δ score| = 0.0` (fold-0 `acc_mlp_max` 0.7987 either way).
3. **Thread count is irrelevant**: `torch.set_num_threads` ∈ {1, 4, 8} all give fold-0
   `acc_mlp_max = 0.7987`.
4. **The §3.2 efficiency deviation is exact**: scoring only the 20 nominated pairs and scoring the
   full 149 × 595 eval-pair matrix then indexing give **identical** fold-0 predictions (0.7987 both).
   This is the one thing this pregate's emitter could have got wrong, and it did not.

**Magnitude and direction.** The drift is small per quantity — HateMM `acc_mlp_max` moves 0.0067
(5 items on 744), pooled pair-AUC `auc_mlp` moves 0.7753 → 0.7747 (0.0006) — but it is **neither
sign-preserving nor uniformly signed**: HateMM's `mlp_max` net goes **−3 → +2** (exchange rate
0.9474 → 1.0377), MHC-ZH's `mlp_max` Δ improves **−0.0466 → −0.0328**, and MHC-EN's *worsens*
**−0.0146 → −0.0182**. The residual cause is most likely oneDNN/MKL kernel selection on this
256-core AMD EPYC varying between sessions; it was not isolated further because it is invariant to
everything this pregate controls.

**Is it load-bearing? Partly — and the honest answer is worse than "4-dp provenance only".**
Checked against F95's own bar (Δacc ≥ +0.010 with 5/5 folds Δ ≥ 0 and ≥3/5 strictly positive):

| cell (fused) | recorded Δ / signs | clears F95's bar? | **anchor Δ / signs (today)** | **clears?** |
|---|---|---|---|---|
| HateMM × MLP × max (**PRIMARY**) | −0.0040 `-0-+-` | no | +0.0027 `0-0+-` | **no** |
| **HateMM × MLP × mean3 (SECONDARY)** | +0.0054 `+-+++` | no | **+0.0107 `0++++`** | **YES** |
| MHC-ZH × MLP × max (**PRIMARY**) | −0.0466 `-----` | no | −0.0328 `+----` | no |
| MHC-ZH × MLP × mean3 | −0.0345 `0----` | no | −0.0207 `0----` | no |
| MHC-EN × MLP × max | −0.0146 `+-+--` | no | −0.0182 `+-0--` | no |
| MHC-EN × MLP × mean3 | −0.0383 `+----` | no | −0.0273 `+----` | no |

**So F95's stated count — "`Δ ≥ +0.010` is achieved by 0 of 36 cells" (F95 §3.2) — does not reproduce
in this session: one cell now clears it** (HateMM × fused × MLP × **mean-top-3**, +0.0107 with 5/5
folds Δ ≥ 0 and 4/5 strictly positive). That is a **secondary** aggregation, on **one** dataset, at
**8 items on 744**, a hair over a +0.010 bar.

**What this does and does not overturn.**
* **F95's VERDICT (KILL) stands, and does not depend on the drifted count.** Its load-bearing terms
  are all closed-form and all reproduce bit-exactly: control 2b's shape cost (−0.0417 / −0.0293 /
  −0.0437, `acc_cos_shape` exact on 3/3) and the deployed floor. Under today's anchor **all four of
  F95's PRIMARY cells still fail** (+0.0027 / −0.0328 on the two primary datasets at max aggregation),
  and the promotion bar was never met on a secondary aggregation anyway.
* **F95's headline sentence needs a caveat when quoted.** "0 of 36" is a session-dependent count; the
  session-independent statement is "0 of 4 primary cells, and the aggregation-shape cost alone is
  −0.029 to −0.044 before any verifier runs".
* **F97's "78/78 parity" was true when made and would not re-assert today** — it fails on the 15
  trained cells of the HateMM emitter. F97's *conclusions* are unaffected: K-VGA-3 (its decisive bar)
  is a **relative** comparison between two feature families measured inside one session, which is
  exactly the kind of claim this drift cannot touch.
* **Consequence for the campaign, stated as a rule:** any future record gating against F95's MLP arm
  must gate against a **same-session re-run** (`--stage anchor`), never against the recorded JSON; and
  any *count* of F95 cells crossing a threshold must be treated as session-dependent.

**Consequence for THIS record: none.** Every VSW quantity is a paired Δ against a floor computed in
the *same* session, and that floor is closed-form and reproduced bit-exactly (§4.1). No number in
§5-§9 is compared against a cross-session trained quantity. The one place a cross-record comparison
appears is §7.4's benchmark table, and it is flagged there.

> ### ▸ APPENDED 2026-07-28 by the closeout/hardening agent — §4.3's CAUSE is corrected; its NUMBERS stand
>
> *Nothing above this box is altered. The measured table in §4.3 is correct and reproduces. What is
> corrected is the attributed **cause** and the **rule** drawn from it. Full evidence:*
> **`refine-logs/PREGATE_DETERMINISM_CLAUSE.md`.**
>
> The drift is **not** oneDNN/MKL kernel selection and is **not** non-determinism. It is **one unpinned
> environment variable**. `scripts/analysis/mechnov_drive.sh:9` exports
> `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`; `scripts/analysis/vsw_drive.sh` exports nothing. Re-running the
> frozen module unmodified today reproduces **the recorded F95 cell** at `OMP=8` and **this record's
> `--stage anchor` cell** with the variable unset — **6 of 6 predicted fold-0 `auc_mlp` values hit**
> (0.7589 / 0.7908 / 0.6900 vs 0.7584 / 0.7911 / 0.6902). Under `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`
> the frozen `run_space` reproduces the recorded cell on **630 of 630** emitted quantities across all
> three datasets — **including all 186 torch-trained ones** — and a fresh-process repeat is bit-exact on
> the score array.
>
> The perturbation enters **upstream of the estimator**: `sha256(Phi_fit)` takes a distinct value at
> every thread count, because `PCA(svd_solver="full")` → LAPACK/OpenBLAS blocks its reduction by thread
> count. Closed-form quantities absorb it below 4 dp (hence "bit-exact"; they are 4-dp-exact, not
> bit-exact) and the convex logistic arm absorbs it too (0 of 186 quantities moved); only the 4 500-step
> Adam MLP amplifies it above 4 dp. §4.3's four diagnostics are all correct and all tested the wrong
> knob — `torch.set_num_threads` does not reach the BLAS/LAPACK path.
>
> **Consequences.** (i) §4.3's rule *"must gate against a same-session re-run, never against the
> recorded JSON"* is **withdrawn as a general rule** — with the environment pinned and recorded (clause
> DET-1/DET-2) banked JSON is a valid anchor; the same-session rule survives only where the anchor's
> environment is unknown. (ii) §4.3's and §10.5's *"F97's 78/78 parity … would not re-assert today"* is
> **too strong**: F97's 48 trained cells matched F95's recorded values, which is only possible at
> `OMP=8`, so F97 ran pinned and its parity **re-asserts whenever DET-1 is honoured**. (iii) F95's
> *"0 of 36"* remains environment-conditioned and is exact under its own `OMP=8` configuration.
> (iv) **No verdict in F95/F96/F97/F98/F105 moves** — blast-radius table in the clause file, banked as a
> ledger erratum.

---

## §5. K-VSW-1 and K-VSW-0 — the λ-selected arms

λ selected on inner folds only (`StratifiedKFold(5, shuffle=True, random_state=17)` inside the
fitting pool), ties toward λ = 0, applied to the held-out fold. PRIMARY family `pow`.

| dataset | family | **Δacc** | ΔmF1 | fold signs | fixed | broken | **ER** | changed | pos-rate (bank) | λ* per fold |
|---|---|---|---|---|---|---|---|---|---|---|
| **HateMM** | **pow (PRIMARY)** | **+0.0255** | +0.0242 | `+++++` | 36 | 17 | **2.1176** | 53 | 0.4368 (0.4005) | 3, 2, 3, 2, 3 |
| HateMM | exp | +0.0255 | +0.0239 | `+++++` | 39 | 20 | 1.9500 | 59 | 0.4315 | 4, 4, 4, 4, 8 |
| HateMM | lin | +0.0188 | +0.0182 | `+0+++` | 26 | 12 | 2.1667 | 38 | 0.4570 | 0.8, 0.9, 0.8, 0.9, 0.8 |
| **MHC-ZH** | **pow (PRIMARY)** | **−0.0017** | −0.0056 | `++-0-` | 8 | 9 | 0.8889 | 17 | 0.3230 (0.3109) | 0.25, 0.25, 0.25, **0**, 0.5 |
| MHC-ZH | exp | −0.0138 | −0.0177 | `0--0-` | 1 | 9 | 0.1111 | 10 | 0.3351 | **0**, 0.5, 1, **0**, 1 |
| MHC-ZH | lin | +0.0000 | −0.0044 | `+0-+-` | 11 | 11 | 1.0000 | 22 | 0.3178 | 0.4, 0.4, 0.5, 0.5, 0.6 |
| **MHC-EN** | **pow (PRIMARY)** | **+0.0018** | −0.0015 | `+-+-+` | 22 | 21 | 1.0476 | 43 | 0.2477 (0.3060) | 0.25, 2, 0.5, 0.5, 0.5 |
| MHC-EN | exp | −0.0036 | −0.0067 | `+--0-` | 22 | 24 | 0.9167 | 46 | 0.2532 | 1, 2, 1, 1, 16 |
| MHC-EN | lin | +0.0073 | +0.0058 | `+-00+` | 22 | 18 | 1.2222 | 40 | 0.2495 | 0.5, 0.9, 0.8, 0.9, 0.9 |

### 5.1 K-VSW-1 (net ≥ +0.030 on ≥2 of 3, one and the same family) — **FAIL**

**0 of 3 datasets reach +0.030 under any family.** The best λ-selected number anywhere in the
battery is **+0.0255** (HateMM, and it is reached by `pow` and `exp` alike), i.e. **85 % of the bar
on one dataset and ~0 on the other two.** The bar requires two datasets; it is missed by two.

### 5.2 K-VSW-0 (interest threshold: ≥ +0.010 on ≥1 dataset, 5/5 fold signs ≥ 0, ≥3/5 strict) — **PASS on HateMM only**

HateMM `pow` returns **+0.0255 with fold deltas `+0.0336, +0.0067, +0.0268, +0.0268, +0.0338`** —
**5/5 strictly positive**, the cleanest fold-sign pattern any treatment arm has produced in this
family. MHC-ZH (−0.0017) and MHC-EN (+0.0018) do not clear it. **This licenses nothing** (§2.2); it
is reported so the boundary is visible rather than hidden.

### 5.3 The arithmetic that actually closes K-VSW-1: it is unreachable **with hindsight**

The λ-selected numbers above could in principle be blamed on the selector. They cannot. Below is the
**best fixed λ chosen on the evaluation data itself** — a strict upper bound on what *any* selection
rule over the declared grids could deliver — for every dataset × every declared family: <!-- anchor -->

| dataset | `pow` | `exp` | `lin` | **best over the whole declared operator space** |
|---|---|---|---|---|
| HateMM | +0.0282 (λ = 3) | **+0.0309** (λ = 4) | +0.0228 (λ = 0.9) | **+0.0309** |
| MHC-ZH | +0.0052 (λ = 0.25) | +0.0017 (λ = 0.5) | +0.0069 (λ = 0.4) | **+0.0069** |
| MHC-EN | +0.0128 (λ = 0.5) | +0.0128 (λ = 1) | +0.0164 (λ = 0.9) | **+0.0164** |

**Even with full hindsight over 3 families × 48 λ values, only one dataset reaches +0.030.**
MHC-ZH tops out at **23 % of the bar** and MHC-EN at **55 %**. K-VSW-1 is therefore not merely unmet
— it is **arithmetically unreachable on ≥2 of 3 datasets for the entire declared operator space**.
That is the door-closer this pregate was run to produce.

### 5.4 The selection lock, arriving at the smallest possible selection surface

| dataset | nested λ (deployable) | best fixed λ, pooled hindsight | per-fold ORACLE λ (hindsight per fold) | **fraction of the pooled-hindsight ceiling the selector keeps** |
|---|---|---|---|---|
| HateMM | **+0.0255** | +0.0282 | +0.0349 | **90 %** |
| MHC-ZH | **−0.0017** | +0.0052 | +0.0121 | **negative** |
| MHC-EN | **+0.0018** | +0.0128 | +0.0310 | **14 %** |

The object being selected here is **one scalar from a 24-point grid**, chosen on 440-595 items by
5-fold inner CV — the smallest selection surface anywhere in this campaign. On HateMM the selector
works (keeps 90 %). On MHC-ZH and MHC-EN it **destroys essentially all of the available headroom**,
turning +0.0128 into +0.0018 on EN and +0.0052 into −0.0017 on ZH. F66's selection lock — hitherto
demonstrated on per-item routers and multi-parameter gates — is here reproduced on a **single global
hyperparameter**, which is as small as the object can get. Note the per-fold oracle would clear
+0.030 on HateMM (+0.0349) and MHC-EN (+0.0310); the deployable selector recovers +0.0255 and
+0.0018 of those. **The gap between what is present and what is selectable is the whole story.**

---

## §6. K-VSW-2 — THE DELIVERABLE: exchange rate as a function of aggregation sharpness

This is the diagnostic the sweep record called *"a diagnostic that cannot fail"*, and it is the
reason this arm was run. **λ = 0 is the deployed vote; λ → ∞ is "emit the label of the single
best-verified neighbour in the deployed top-20".** Fixed λ, **no selection** — the curve is a
property of the operator, not of a selector. `precision = fixed / changed`; `net = fixed − broken`;
`Δacc = net / n`.

### 6.1 PRIMARY family `pow`, full continuum, all three datasets

**HateMM** (floor 0.8441, n = 744)

| λ | Δacc | fixed | broken | net | **ER** | changed | **precision** | fold signs |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.0000 | 0 | 0 | +0 | n/a | 0 | n/a | `00000` |
| 0.25 | +0.0202 | 18 | 3 | +15 | **6.0000** | 21 | **0.8571** | `+++++` |
| 0.5 | +0.0188 | 24 | 10 | +14 | 2.4000 | 34 | 0.7059 | `+0+++` |
| 1 | +0.0215 | 29 | 13 | +16 | 2.2308 | 42 | 0.6905 | `+0+++` |
| 2 | +0.0269 | 35 | 15 | +20 | 2.3333 | 50 | 0.7000 | `+++++` |
| **3** | **+0.0282** | 38 | 17 | **+21** | 2.2353 | 55 | 0.6909 | `+++++` |
| 4 | +0.0269 | 39 | 19 | +20 | 2.0526 | 58 | 0.6724 | `+++++` |
| 6 | +0.0255 | 40 | 21 | +19 | 1.9048 | 61 | 0.6557 | `+++++` |
| 8 | +0.0255 | 41 | 22 | +19 | 1.8636 | 63 | 0.6508 | `+++++` |
| 12 | +0.0228 | 42 | 25 | +17 | 1.6800 | 67 | 0.6269 | `+++++` |
| 16 | +0.0255 | 43 | 24 | +19 | 1.7917 | 67 | 0.6418 | `+++++` |
| 24 | +0.0228 | 40 | 23 | +17 | 1.7391 | 63 | 0.6349 | `+++++` |
| 32 | +0.0228 | 41 | 24 | +17 | 1.7083 | 65 | 0.6308 | `+++++` |
| 48 | +0.0215 | 42 | 26 | +16 | 1.6154 | 68 | 0.6176 | `+++++` |
| 64 | +0.0215 | 42 | 26 | +16 | 1.6154 | 68 | 0.6176 | `+++++` |
| 96 | +0.0202 | 41 | 26 | +15 | 1.5769 | 67 | 0.6119 | `+++++` |
| 128 | +0.0202 | 40 | 25 | +15 | 1.6000 | 65 | 0.6154 | `+++++` |
| 192 | +0.0188 | 40 | 26 | +14 | 1.5385 | 66 | 0.6061 | `++++0` |
| 256 | +0.0175 | 40 | 27 | +13 | 1.4815 | 67 | 0.5970 | `++++0` |
| 384 | +0.0175 | 40 | 27 | +13 | 1.4815 | 67 | 0.5970 | `++++0` |
| 512 | +0.0175 | 40 | 27 | +13 | 1.4815 | 67 | 0.5970 | `++++0` |
| 1024 | +0.0148 | 41 | 30 | +11 | 1.3667 | 71 | 0.5775 | `++++-` |
| 4096 | +0.0148 | 44 | 33 | +11 | 1.3333 | 77 | 0.5714 | `++++-` |
| **∞** | +0.0148 | 45 | 34 | +11 | **1.3235** | 79 | **0.5696** | `++++-` |

**MHC-ZH** (floor 0.8480, n = 579)

| λ | Δacc | fixed | broken | net | **ER** | changed | **precision** | fold signs |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.0000 | 0 | 0 | +0 | n/a | 0 | n/a | `00000` |
| **0.25** | **+0.0052** | 13 | 10 | **+3** | **1.3000** | 23 | **0.5652** | `++-+-` |
| 0.5 | +0.0035 | 15 | 13 | +2 | 1.1538 | 28 | 0.5357 | `++-+-` |
| 1 | +0.0000 | 16 | 16 | +0 | 1.0000 | 32 | 0.5000 | `++-+-` |
| 2 | −0.0086 | 15 | 20 | −5 | 0.7500 | 35 | 0.4286 | `00-+-` |
| 3 | −0.0086 | 16 | 21 | −5 | 0.7619 | 37 | 0.4324 | `0+-+-` |
| 4 | −0.0104 | 17 | 23 | −6 | 0.7391 | 40 | 0.4250 | `00-+-` |
| 6 | −0.0138 | 17 | 25 | −8 | 0.6800 | 42 | 0.4048 | `-0-+-` |
| 8 | −0.0121 | 17 | 24 | −7 | 0.7083 | 41 | 0.4146 | `-0-+-` |
| 12 | −0.0086 | 19 | 24 | −5 | 0.7917 | 43 | 0.4419 | `-0-+-` |
| 16 | −0.0052 | 20 | 23 | −3 | 0.8696 | 43 | 0.4651 | `-0-+-` |
| 24 | −0.0035 | 21 | 23 | −2 | 0.9130 | 44 | 0.4773 | `-0-+-` |
| 32 | −0.0086 | 21 | 26 | −5 | 0.8077 | 47 | 0.4468 | `---+-` |
| 48 | −0.0121 | 22 | 29 | −7 | 0.7586 | 51 | 0.4314 | `---+-` |
| 64 | −0.0104 | 23 | 29 | −6 | 0.7931 | 52 | 0.4423 | `0--+-` |
| 96 | −0.0138 | 24 | 32 | −8 | 0.7500 | 56 | 0.4286 | `0--0-` |
| 128 | −0.0104 | 25 | 31 | −6 | 0.8065 | 56 | 0.4464 | `0--0-` |
| 192 | −0.0138 | 26 | 34 | −8 | 0.7647 | 60 | 0.4333 | `+--0-` |
| 256 | −0.0155 | 26 | 35 | −9 | 0.7429 | 61 | 0.4262 | `+--0-` |
| 384 | −0.0155 | 27 | 36 | −9 | 0.7500 | 63 | 0.4286 | `+--0-` |
| 512 | −0.0155 | 27 | 36 | −9 | 0.7500 | 63 | 0.4286 | `0--0-` |
| 1024 | −0.0138 | 28 | 36 | −8 | 0.7778 | 64 | 0.4375 | `+--+-` |
| 4096 | −0.0225 | 29 | 42 | −13 | 0.6905 | 71 | 0.4085 | `+--0-` |
| **∞** | **−0.0225** | 30 | 43 | −13 | **0.6977** | 73 | **0.4110** | `+----` |

**MHC-EN** (floor 0.7796, n = 549)

| λ | Δacc | fixed | broken | net | **ER** | changed | **precision** | fold signs |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.0000 | 0 | 0 | +0 | n/a | 0 | n/a | `00000` |
| 0.25 | +0.0109 | 18 | 12 | +6 | **1.5000** | 30 | **0.6000** | `++0-+` |
| **0.5** | **+0.0128** | 24 | 17 | **+7** | 1.4118 | 41 | 0.5854 | `+0+-+` |
| 1 | +0.0128 | 27 | 20 | +7 | 1.3500 | 47 | 0.5745 | `+-+-+` |
| 2 | +0.0128 | 29 | 22 | +7 | 1.3182 | 51 | 0.5686 | `+-+-+` |
| 3 | +0.0036 | 29 | 27 | +2 | 1.0741 | 56 | 0.5179 | `+-+-+` |
| 4 | +0.0018 | 30 | 29 | +1 | 1.0345 | 59 | 0.5085 | `+-+-0` |
| 6 | +0.0000 | 31 | 31 | +0 | 1.0000 | 62 | 0.5000 | `+-+--` |
| 8 | +0.0000 | 32 | 32 | +0 | 1.0000 | 64 | 0.5000 | `+-+--` |
| 12 | +0.0018 | 32 | 31 | +1 | 1.0323 | 63 | 0.5079 | `+-+--` |
| 16 | +0.0055 | 34 | 31 | +3 | 1.0968 | 65 | 0.5231 | `+-+--` |
| 24 | −0.0055 | 34 | 37 | −3 | 0.9189 | 71 | 0.4789 | `+-+--` |
| 32 | −0.0055 | 35 | 38 | −3 | 0.9211 | 73 | 0.4795 | `+-+--` |
| 48 | −0.0018 | 36 | 37 | −1 | 0.9730 | 73 | 0.4932 | `+-+--` |
| 64 | −0.0036 | 36 | 38 | −2 | 0.9474 | 74 | 0.4865 | `+-+--` |
| 96 | −0.0018 | 37 | 38 | −1 | 0.9737 | 75 | 0.4933 | `+-+--` |
| 128 | +0.0018 | 40 | 39 | +1 | 1.0256 | 79 | 0.5063 | `+-+0-` |
| 192 | −0.0073 | 38 | 42 | −4 | 0.9048 | 80 | 0.4750 | `+-+--` |
| 256 | −0.0073 | 38 | 42 | −4 | 0.9048 | 80 | 0.4750 | `+-+--` |
| 384 | −0.0018 | 41 | 42 | −1 | 0.9762 | 83 | 0.4940 | `+-+--` |
| 512 | −0.0018 | 41 | 42 | −1 | 0.9762 | 83 | 0.4940 | `+-+--` |
| 1024 | −0.0018 | 42 | 43 | −1 | 0.9767 | 85 | 0.4941 | `+-+--` |
| 4096 | −0.0036 | 43 | 45 | −2 | 0.9556 | 88 | 0.4886 | `+-+--` |
| **∞** | **−0.0073** | 44 | 48 | −4 | **0.9167** | 92 | **0.4783** | `+-+--` |

Secondary families, curve extremes (full tables in `vsw_pregate_OUT.json`):

| family | HateMM best Δacc | best ER at ≥20 changed | MHC-ZH best Δacc | best ER | MHC-EN best Δacc | best ER |
|---|---|---|---|---|---|---|
| `exp` | +0.0309 (λ = 4) | 2.4375 (λ = 4) | +0.0017 (λ = 0.5) | 1.0833 (λ = 1) | +0.0128 (λ = 1) | 1.5385 (λ = 1) |
| `lin` | +0.0228 (λ = 0.9) | 2.6000 (λ = 0.8) | +0.0069 (λ = 0.4) | 1.4000 (λ = 0.5) | +0.0164 (λ = 0.9) | 1.5455 (λ = 0.4) |

### 6.2 OUTCOME (a), on the frozen terms of §2.3 — the rate is **NOT** bounded below 1.2

The sweep record's hoped-for reading was outcome (b): *"it shows the rate is bounded below 1 across
the entire continuum, which closes the aggregation axis arithmetically."* **That reading is false and
is hereby withdrawn.** Under the §2.3 test declared before the run (some λ with **≥ 20 changed
decisions** attaining **ER > 1.2**, on ≥2 of 3 datasets), the condition is met on **3 of 3**:

| dataset | max ER at ≥20 changed decisions | at λ | changed | number of qualifying λ |
|---|---|---|---|---|
| HateMM | **6.0000** | 0.25 | 21 | **23 of 23 non-zero λ** |
| MHC-ZH | **1.3000** | 0.25 | 23 | 1 |
| MHC-EN | **1.5000** | 0.25 | 30 | 4 |

For context, **F95's best exchange rate anywhere in a 36-cell battery was 1.1667**, and its two
measured points were 0.9474 (max) and 1.0851 (mean-top-3) on HateMM. VSW's continuum reaches
**6.0000** on HateMM and stays above 1.3 for its entire length there. **A sharpness regime in which
verifier re-weighting buys fixes at well above par demonstrably exists.**

### 6.3 …and the law that replaces the withdrawn one: **rate and volume are in a binding trade-off, and the NET is capped**

The curve's shape is the finding. On every dataset, **precision decays monotonically as sharpness
rises**, and it decays at almost exactly the rate that cancels the rise in volume:

| dataset | precision at the sharpest useful λ → at λ = ∞ | changed at those λ | **net across the whole continuum** | max net |
|---|---|---|---|---|
| HateMM | 0.8571 (λ = 0.25) → **0.5696** (λ = ∞) | 21 → 79 | **+11 to +21**, never outside | **+21** (λ = 3) = **+0.0282** |
| MHC-ZH | 0.5652 (λ = 0.25) → **0.4110** (λ = ∞) | 23 → 73 | **+3 down to −13** | **+3** (λ = 0.25) = **+0.0052** |
| MHC-EN | 0.6000 (λ = 0.25) → **0.4783** (λ = ∞) | 30 → 92 | **+7 down to −4** | **+7** (λ = 0.5) = **+0.0128** |

`net = changed · (2·precision − 1)`. On HateMM the two factors move in opposite directions across a
**16 000×** range of λ and their product varies by less than a factor of two, staying pinned between
+11 and +21 items — **0.0148 to 0.0282 accuracy, i.e. bounded strictly under the +0.030 bar at every
single point of the continuum.** On MHC-ZH and MHC-EN the product is under +7 items everywhere and
goes negative over most of the range. **Precision falls below 0.50 — worse than a coin — at 20 of
MHC-ZH's 23 non-zero λ and at 12 of MHC-EN's 23**, which is the same statement F95 made at two
points, now made at 24.

**The corrected law-I datum, stated for the paper.** *In a retrieval-memory classifier whose relation
scorer is measurably better than the deployed cosine (+0.16 to +0.23 within-query AUC, F95 §4.1), the
exchange rate of verifier-weighted aggregation is **not** bounded below par — it reaches 6.0 — but
the rate is purchasable only by shrinking the population it acts on, and the product of rate and
volume is capped below the improvement bar at **every** point of a continuum spanning the deployed
vote at one end and single-best-neighbour adjudication at the other. Sharpening the aggregation does
not trade off against anything except its own coverage.* That is a strictly stronger and more
falsifiable statement than the two-point read F95 carries, and it is what this pregate was run to
produce.

---

## §7. MANDATORY CONTROLS

### 7.1 DEG-A (threshold twin) and DEG-B (fixed-k twin) — **BOTH FIRE ON MHC-ZH; neither fires on HateMM or MHC-EN**

Pooled agreement between `VSW_pow`'s held-out decisions and each twin. Frozen kill line **≥ 0.95**.

| dataset | **DEG-A** global threshold shift | **DEG-B** best single fixed k | agree with deployed | verdict |
|---|---|---|---|---|
| HateMM | 0.9220 | 0.9328 (k = 15) | 0.9288 | **neither fires** |
| **MHC-ZH** | **0.9516 → FIRES** | **0.9706 (k = 20) → FIRES** | 0.9706 | **KILL on this dataset** |
| MHC-EN | 0.9126 | 0.9217 (k = 20) | 0.9217 | neither fires |

**MHC-ZH is a degenerate cell and is killed on its own terms.** DEG-B's arg-max is **k = 20**, which
*is* the deployed rule, and the agreement with the deployed rule is the identical 0.9706 — the
consequence of the inner CV selecting λ\* ∈ {0.25, 0.25, 0.25, **0**, 0.5}, i.e. **falling back to
the deployed vote outright in one fold and to near-zero sharpening in the rest, changing 17 items out
of 579**. This is exactly the verdict form §3.6.4 declared in advance and exactly what F98 recorded
for C3 on MHC-EN (DEG-B 0.9964 at k = 20).

**HateMM is not a degenerate cell, and that is a genuine difference from F98's C3.** F98's C3 fired
both controls on the *only* dataset where it was positive (DEG-A 0.9570, DEG-B 0.9610 at k = 15) and
was additionally *outscored* by its own threshold twin (`THRESH_best` +0.0188 vs C3 +0.0134). Here
the ordering is reversed on both counts:

| quantity | F98 C3 (HateMM) | **VSW (HateMM)** |
|---|---|---|
| DEG-A agreement | 0.9570 **FIRES** | **0.9220** |
| DEG-B agreement | 0.9610 (k = 15) **FIRES** | **0.9328** (k = 15) |
| arm Δacc | +0.0134 | **+0.0255** |
| `THRESH_best` Δacc | **+0.0188** (beats the arm) | **+0.0148** (the arm beats it by +0.0107) |
| fold signs | `-0+++` (4/5) | **`+++++`** (5/5) |
| exchange rate | 1.8333 | **2.1176** |

So on HateMM, VSW is **not** a threshold move in a costume and **not** a fixed-k truncation in a
costume — the two degeneracies that killed every previous member of this family. It is a distinct
operator that beats both twins. It is nonetheless **under the bar**, and §5.3 shows no λ anywhere
would put it over on a second dataset.

Reference twins, for completeness (Δacc vs the same floor):

| dataset | `THRESH_best` | `FIXK_1` | `FIXK_3` | `FIXK_10` | `FIXK_15` | `FIXK_20` (= deployed) | `ORACLE_lambda` |
|---|---|---|---|---|---|---|---|
| HateMM | +0.0148 | −0.0430 | −0.0430 | +0.0027 | +0.0040 | +0.0000 | +0.0349 |
| MHC-ZH | −0.0052 | −0.0294 | −0.0294 | −0.0121 | −0.0121 | +0.0000 | +0.0121 |
| MHC-EN | −0.0128 | −0.0437 | −0.0437 | −0.0036 | −0.0055 | +0.0000 | +0.0310 |

(The `FIXK` column independently re-confirms F94 in this arena: every truncation of the deployed
top-20 is ≤ 0 on all three datasets, and `k ≤ 3` costs −0.0294 to −0.0437.)

### 7.2 DEG-D (cosine twin — identical machinery, **no verifier information**) — **does not fire**, and it prices the verifier's contribution

Firing condition (§3.6.6): `CTRL_cos` matches or beats `VSW_pow` on ≥2 of 3 datasets.
**Measured: 1 of 3.** The control does not fire.

| dataset | `VSW_pow` Δacc (ER) | `CTRL_cos_pow` Δacc (ER) | **verifier increment** | agreement | ctrl ≥ VSW |
|---|---|---|---|---|---|
| HateMM | **+0.0255** (2.1176) | +0.0067 (1.7143) | **+0.0188** | 0.9382 | no |
| MHC-ZH | −0.0017 (0.8889) | −0.0155 (0.1818) | +0.0138 | 0.9551 | no |
| MHC-EN | +0.0018 (1.0476) | **+0.0200** (3.2000) | **−0.0182** | 0.9199 | **yes** |

This is the VSW analogue of K-VGA-3, and unlike K-VGA-3 it **does not fire**: re-weighting by the
*cosine itself* — pure extra sharpening toward rank 1, carrying no learned relation information at
all — recovers only +0.0067 of HateMM's +0.0255. **On HateMM the trained relation score contributes
+0.0188 of the +0.0255, i.e. roughly three quarters of it, and that increment is not obtainable from
the cosine ordering.** This is the first measurement in this campaign in which the verifier profile
beats a like-for-like cosine control at the *decision* level rather than the relation level, and it
should be recorded as such — while noting that on MHC-EN the cosine twin **wins** (+0.0200 vs
+0.0018), so the effect does not replicate across datasets.

### 7.3 CLASS BALANCE — **PASS on all three; no nulls are void**

| dataset | bank pos-rate | `VSW_pow` pos-rate | deviation | tolerance | verdict |
|---|---|---|---|---|---|
| HateMM | 0.4005 | 0.4368 | **0.0363** | 0.10 | PASS |
| MHC-ZH | 0.3109 | 0.3230 | **0.0121** | 0.10 | PASS |
| MHC-EN | 0.3060 | 0.2477 | **0.0583** | 0.10 | PASS |

No arm collapses toward either class. This is what licenses reading everything above as measurement
rather than artefact — and it is the control that, on the §3.8 synthetic arm B0, correctly **voided**
a spurious +0.0657 at a positive rate of 0.1229 against a bank rate of 0.3857 (deviation 0.2628).

### 7.4 Where this sits against the family's standing benchmark

Re-read from source (`vga_pregate_OUT.json`, `aggnet_pregate_OUT.json`), not transcribed, and all in
the identical raw train-split arena against the identical deployed floor (`acc_deployed`
0.8441 / 0.8480 / 0.7796):

| member | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| F95 pair-verification, ungated | −0.0040 | −0.0466 | −0.0146 |
| F97 verifier-profile gate (PRIMARY) | +0.0108 (gbm); logistic p = 0.8706 | +0.0000 | +0.0018 (gbm) |
| **F97 F47-feature adjudication gate — the standing benchmark** | **+0.0269** (p = 0.0050) | **+0.0104** (p = 0.0050) | **+0.0182** (p = 0.0100) |
| F98 C3 learned aggregation-profile net | +0.0134 (DEG-A/B both FIRE) | −0.0069 | +0.0000 |
| **VSW (this record), λ selected on inner folds** | **+0.0255** (DEG-A/B clear) | **−0.0017** (DEG-A/B FIRE) | **+0.0018** |

**VSW is below the F47-gate benchmark on all three datasets** (+0.0255 < +0.0269; −0.0017 < +0.0104;
+0.0018 < +0.0182), which is the same question F98 posed for C3 and got the same answer to: a richer
operator over the same neighbourhood extracts **less** than the cheap F47 gate already banked. It is
however the **best-behaved** member the family has produced — it is the only one whose positive
dataset survives both degeneracy controls, and it beats its own threshold and fixed-k twins there.

*Footnote, disclosed so it is not read as a contradiction:* this record's `THRESH_best` on HateMM is
**+0.0148**, F98's is **+0.0188**. They are different estimators of the same idea: F98 fitted τ on
within-fitting-pool leave-one-out votes (`exclude_self=True`), this record fits τ on each fitting-pool
item's **own out-of-fold** deployed vote — the identical records the λ-selector consumes, which is
what makes DEG-A an apples-to-apples twin *here*. Each record's DEG-A comparison is internally
consistent; the two τ values are not comparable across records.

---

## §8. PERMUTATION NULL

**Complete: 200/200 draws on all three datasets** (`vsw_perm_{hatemm,zh,en}_OUT.json`, 600 draws =
**3 000 F95 verifier refits**, ~7 h of ≤8-thread CPU). `PERM_SEED = 12345`; per draw and fold the
**fitting-fold item labels** are permuted and the pair targets `y = 1[lab̃_i == lab̃_j]` re-derived, so
the verifier is refitted at the identical frozen budget while the bank labels the vote consumes, the
retrieval, the cosines, the deployed floor and every held-out gold label are untouched. The full
pipeline **including the inner-fold λ selection** is re-run per draw.
`p = (1 + #{null ≥ observed}) / 201`; resolution 0.0050; significance at `p < 0.05`.

### 8.1 The λ-selected arms — **HateMM significant, MHC-ZH and MHC-EN honest nulls**

| dataset | arm | observed | null mean ± sd | null q95 | null max | **frac of draws ≥ 0** | **p** |
|---|---|---|---|---|---|---|---|
| **HateMM** | **`pow` (PRIMARY)** | **+0.0255** | −0.0005 ± 0.0053 | +0.0068 | +0.0134 | **0.5700** | **0.0050** |
| HateMM | `exp` | +0.0255 | +0.0004 ± 0.0051 | +0.0094 | +0.0148 | 0.5900 | **0.0050** |
| HateMM | `lin` | +0.0188 | +0.0007 ± 0.0047 | +0.0081 | +0.0148 | 0.6300 | **0.0050** |
| **MHC-ZH** | **`pow` (PRIMARY)** | −0.0017 | −0.0038 ± 0.0052 | +0.0000 | +0.0069 | 0.5250 | **0.5522** |
| MHC-ZH | `exp` | −0.0138 | −0.0038 ± 0.0053 | +0.0001 | +0.0086 | 0.4450 | 0.9403 |
| MHC-ZH | `lin` | +0.0000 | −0.0038 ± 0.0052 | +0.0035 | +0.0104 | 0.3500 | 0.3532 |
| **MHC-EN** | **`pow` (PRIMARY)** | +0.0018 | −0.0036 ± 0.0070 | +0.0091 | +0.0164 | 0.4750 | **0.1194** |
| MHC-EN | `exp` | −0.0036 | −0.0037 ± 0.0064 | +0.0073 | +0.0146 | 0.3400 | 0.5423 |
| MHC-EN | `lin` | +0.0073 | −0.0034 ± 0.0070 | +0.0091 | +0.0182 | 0.3450 | 0.1144 |

**These nulls are INFORMATIVE, which is what makes them readable** — the property §3.7 predicted and
F98 §6 lacked. **34-63 % of null draws reach ≥ 0**, because λ = 0 is in every grid and is the
tie-break winner, so a draw whose verifier carries nothing falls back to the deployed rule and scores
exactly 0.0000. Contrast F98, where **0 of 300** draws reached zero and every arm therefore scored
p = 0.0099 automatically — including a two-item no-op. Nothing of that kind is happening here: the
null means sit at −0.0038 to +0.0007, i.e. essentially at the floor, and the test discriminates.

**Reading.** HateMM's +0.0255 beats **all 200** label-shuffled draws under all three families — the
effect is real and is not an artefact of fitting one scalar. **MHC-ZH and MHC-EN return honest
non-significance on every family** (p = 0.1144 to 0.9403); in particular MHC-EN's PRIMARY +0.0018 at
**p = 0.1194** is a null, so **VSW is 1-for-3, not 1-of-2-with-one-pending.** Combined with §5.3, the
picture is unambiguous: one dataset carries a real but sub-bar effect and two carry nothing.

### 8.2 A defect that DOES apply — the fixed-λ curve nulls are uninformative, and must not be banked

The same 600 draws also yield a null for every point of the K-VSW-2 curve. **Those p-values are not
usable above small λ, and saying so is required rather than optional.** PRIMARY family, `pow`:

| dataset | λ | observed | null mean ± sd | null max | **frac ≥ 0** | p | usable? |
|---|---|---|---|---|---|---|---|
| HateMM | 0 | +0.0000 | +0.0000 ± 0.0000 | +0.0000 | **1.0000** | 1.0000 | degenerate by construction |
| HateMM | 0.25 | +0.0202 | +0.0013 ± 0.0039 | +0.0148 | 0.6950 | 0.0050 | **yes** |
| HateMM | 3 | +0.0282 | −0.0140 ± 0.0081 | +0.0067 | 0.0600 | 0.0050 | marginal |
| HateMM | 16 | +0.0255 | −0.0498 ± 0.0116 | −0.0202 | **0.0000** | 0.0050 | **no** |
| HateMM | 256 | +0.0175 | −0.0880 ± 0.0131 | −0.0524 | **0.0000** | 0.0050 | **no** |
| HateMM | ∞ | +0.0148 | −0.0961 ± 0.0130 | −0.0578 | **0.0000** | 0.0050 | **no** |
| MHC-ZH | 0.25 | +0.0052 | −0.0056 ± 0.0055 | +0.0086 | 0.1850 | 0.0597 | yes (**n.s.**) |
| MHC-ZH | ∞ | −0.0225 | −0.1283 ± 0.0158 | −0.0794 | **0.0000** | 0.0050 | **no** |
| MHC-EN | 0.25 | +0.0109 | −0.0034 ± 0.0074 | +0.0164 | 0.3350 | 0.0398 | yes |
| MHC-EN | ∞ | −0.0073 | −0.1281 ± 0.0196 | −0.0729 | **0.0000** | 0.0050 | **no** |

**Why.** At a *fixed* λ > 0 the operator has **no fallback**: it re-weights by whatever the verifier
emits, so a null verifier produces a heavy meaningless re-weighting and the draw lands far below the
floor (null mean −0.0961 at λ = ∞ on HateMM). Any observed value near zero then beats all 200 draws
automatically. **MHC-ZH at λ = ∞ is the proof: Δacc = −0.0225, an unambiguous LOSS of 13 items,
"significant" at p = 0.0050.** A test that certifies a 13-item regression as significant is measuring
the null's inability to reach the floor, not the operator's signal.

**This is precisely the F98 §6 defect, and it is recorded here because §3.7 predicted this null would
not have it.** That prediction was **correct for the λ-selected arms** (§8.1, fallback reachable,
34-63 % of draws ≥ 0) and **wrong for the fixed-λ curve** (fallback unreachable above λ ≈ 3). The
distinction is exactly the one F98 identified — *whether the null can reach the arm's fallback* — and
it separates the two halves of this same battery. **Consequence: §8.1's p-values carry the verdict;
the per-λ p-values in `vsw_pregate_OUT.json` are reported for completeness and are NOT evidence above
λ ≈ 3.** The K-VSW-2 curve does not need them — it is an arithmetic decomposition of fixed, broken and
changed counts (§6.3), not a significance claim.

---

## §9. VERDICT

# **KILL as a performance lever — and the door-closer only half closed.**

**The performance bet dies exactly as the sweep record priced it at ~2 %. The diagnostic that "cannot
fail" returned the OPPOSITE of the outcome the sweep hoped for**, and that is the finding.

| bar | requirement | measured | verdict |
|---|---|---|---|
| **K-VSW-1** (decisive) | net ≥ **+0.030** on **≥2 of 3** datasets, one and the same family, λ selected on inner folds | best λ-selected number **anywhere** = **+0.0255** (HateMM); **0 of 3** datasets reach the bar; and **with full hindsight over 3 families × 48 λ** only HateMM reaches it (+0.0309), MHC-ZH topping out at **+0.0069** and MHC-EN at **+0.0164** | **FAIL — and arithmetically unreachable** |
| **K-VSW-0** (interest threshold) | Δacc ≥ +0.010 on ≥1 dataset, 5/5 fold signs ≥ 0, ≥3/5 strict | HateMM `pow` **+0.0255**, fold deltas `+0.0336, +0.0067, +0.0268, +0.0268, +0.0338` = **5/5 strictly positive** | **PASS on HateMM only — licenses nothing** |
| **K-VSW-2** (diagnostic) | outcome (a) if some λ with ≥20 changed reaches ER > 1.2 on ≥2 of 3; outcome (b) otherwise | **OUTCOME (a) on 3 of 3.** Max ER at ≥20 changed: **6.0000 / 1.3000 / 1.5000**, against F95's best-of-36-cells 1.1667 | **OUTCOME (a) — the sweep's outcome (b) is FALSE and is withdrawn** |
| **PARITY-λ0** | λ = 0 bit-exact vs `mechfix_ops.deployed_vote`, every fold × dataset | **54/54**, `np.array_equal` on votes and predictions | **PASS** |
| **F95 PARITY** | 26 quantities × 3 datasets at 4 dp | **78/78** (tier-1 30/30 vs recorded, tier-2 48/48 vs same-session anchor) | **PASS** |
| **DEG-A** (mandatory) | agreement with a global threshold shift **< 0.95** | HateMM 0.9220, MHC-EN 0.9126, **MHC-ZH 0.9516** | **FIRES on MHC-ZH → KILL there** |
| **DEG-B** (mandatory) | agreement with any single fixed k **< 0.95** | HateMM 0.9328 (k=15), MHC-EN 0.9217 (k=20), **MHC-ZH 0.9706 (k=20 = the deployed rule)** | **FIRES on MHC-ZH → KILL there** |
| **DEG-D** (cosine twin) | fires if the cosine-only twin matches or beats VSW on ≥2 of 3 | **1 of 3** (MHC-EN only). On HateMM the verifier contributes **+0.0188 of the +0.0255** | **does not fire** |
| **CLASS BALANCE** (mandatory) | pos-rate within 0.10 of the bank rate | deviations **0.0363 / 0.0121 / 0.0583** | **PASS 3/3 — nulls are valid** |
| **PERMUTATION NULL** (mandatory) | beat a label-shuffled null at the same fitting budget, 200 draws | **HateMM p = 0.0050 on all three families** (beats all 200 draws); **MHC-ZH p = 0.5522 / 0.9403 / 0.3532**; **MHC-EN p = 0.1194 / 0.5423 / 0.1144**. Nulls **informative** — 34-63 % of draws reach ≥ 0 | **PASS on HateMM only; honest NULL on 2 of 3** |

### 9.1 What died

**K-VSW-1 is not merely missed — it is out of reach.** The strongest statement this record can make
is §5.3's: *even choosing λ on the evaluation data itself, over the entire declared operator space of
three monotone multiplier families and 48 λ values spanning the deployed vote to single-best-neighbour
adjudication, two of three datasets cannot be brought within half the bar* (MHC-ZH +0.0069 = 23 % of
it, MHC-EN +0.0164 = 55 %). No selector, no λ schedule and no third family can change that; it is a
property of the operator space, not of the search over it.

**MHC-ZH additionally dies of degeneracy**, on both mandatory controls, with DEG-B's arg-max at
**k = 20 — the deployed rule itself**. The inner CV chose λ\* = 0 outright in one fold and ≤ 0.5 in
the rest, changing 17 items in 579. That is F98's C3 verdict form arriving on a third independent
operator.

**And the selection lock arrives at the smallest object it has ever been measured on.** One scalar,
one 24-point grid, 5-fold inner CV on 440-595 items — and it converts +0.0128 into +0.0018 on MHC-EN
and +0.0052 into −0.0017 on MHC-ZH (§5.4). F66's law does not need a per-item router or a
multi-feature gate to bite; a single global hyperparameter is already too much selection for this
arena.

### 9.2 What did NOT die, and must be recorded honestly

1. **The sweep's hoped-for arithmetic closure is FALSE.** `LITSWEEP6_RELGEN` §2 C4 offered outcome
   (b) — *"it shows the rate is bounded below 1 across the entire continuum"* — as the way this arm
   would close the axis. **It is not bounded.** The exchange rate reaches **6.0000** on HateMM and
   exceeds 1.2 at every one of the 23 non-zero λ there, against F95's best-in-36-cells of 1.1667.
   Anyone citing "the exchange rate never exceeds ~1.2" as a law of this system must stop; the
   correct law is §6.3's.
2. **The axis closes on the NET, not on the RATE**, and that is the better datum: rate and volume are
   in a binding trade-off (`net = changed · (2·precision − 1)`), precision decays monotonically with
   sharpness on all three datasets, and the product is pinned below the bar at every point of a
   16 000× λ range.
3. **On HateMM, VSW is the best-behaved member this family has produced**: +0.0255 with 5/5 strictly
   positive fold signs, exchange rate 2.1176, class balance clean, **both** degeneracy controls clear,
   **permutation-validated at p = 0.0050 against an informative null**, beating its threshold twin by
   +0.0107 and its cosine twin by +0.0188. Every previous member either failed a degeneracy control on
   its positive dataset (F98 C3), lost to a free baseline (F97 C2), or was net-negative outright (F95).
   **It is still below the F47-gate benchmark of +0.0269 on that same dataset (§7.4), still one dataset
   out of three — MHC-ZH p = 0.5522 and MHC-EN p = 0.1194 are honest nulls — and still in the raw
   train-split arena.**
4. **DEG-D did not fire**, so the trained relation score is doing real work in the HateMM
   re-weighting (+0.0188 of +0.0255 is not obtainable from the cosine ordering). This does **not**
   reopen F97: K-VGA-3 closed verifier features as *gating/selection/routing* inputs by beating them
   with F47 features head to head. Re-weighting is a different use, it is measured here, and it is
   measured **under bar on 3/3 and unreachable on 2/3**.

### 9.3 The §2.3 escalation clause is triggered, and is handed up rather than resolved here

§2.3, frozen before the run, says of outcome (a): *"⇒ the aggregation axis is **not** closed;
escalate to a formal prereg with independent review. **No promotion is made inside this record under
any outcome.**"* Outcome (a) is met on 3 of 3 datasets. **The clause is therefore triggered and is
recorded as triggered.** It is not being talked away.

What escalation *means* here must be stated with equal precision, because the same record contains
the arithmetic that bounds it: a formal prereg would be testing a mechanism whose **net is capped
below the improvement bar at every point of the continuum, on every declared family, with hindsight,
on two of three datasets** (§5.3, §6.3). The rate finding is real and is a paper-grade correction to
the campaign's stated law; it is **not** a route to the +3-on-2-datasets conjunct, and this record
does not recommend spending GPU on it. **The ruling belongs to the user**, and the two options are
stated plainly:

* **(i) Accept the closure on the net.** Treat §6.3 as the corrected law-I datum, carry the curve
  into the paper as analysis, and close the aggregation axis. *This is what the measured arithmetic
  supports.*
* **(ii) Honour the letter of outcome (a) and prereg.** A formal single-dataset HateMM ceremony on
  `pow` at λ ≈ 3 could be written; §5.3 says it cannot clear the ≥2-dataset conjunct, so it would be
  a ceremony for a single-dataset effect of +0.0255-0.0282 in a raw train arena that F47/F66/F89/F95/
  F97/F98 all say does not convert to the deployed head on test.

---

## §10. ROUTING

1. **Do not spend GPU. Do not promote. Do not ceremony** on the strength of this record. No arm
   reaches the house bar on ≥2 datasets and none can be made to.
2. **Closed by measurement — do not re-propose:** any **λ-interpolated, monotone re-weighting of the
   deployed top-20 by the F95 pair-verifier score**, in any of the three declared families or any
   monotone reparametrisation of them. The continuum from the deployed vote (λ = 0) to
   single-best-verified-neighbour adjudication (λ = ∞) has been measured end to end at 24 points ×
   3 families × 3 datasets, and the net is capped below bar throughout. "A different link function"
   is priced: `pow`, `exp` and `lin` are three different monotone maps of `p` and they agree.
   "A better λ selector" is priced: hindsight λ is measured and still fails on 2/3.
3. **F94's ban text is now measured, not merely generalised.** F94 said *"no truncation **or
   re-weighting** of the retrieved list reaches it"* — a clause `LITSWEEP6_RELGEN` §2 C4 correctly
   flagged as *generalised from F49/F66/F86, not measured for learned weights*. It is now measured
   for learned weights, on the strongest available weight source, and **the clause holds on the net**.
   It does **not** hold on the exchange rate, and F94's ban should be cited with that qualification.
4. **Correct the campaign's exchange-rate law wherever it is cited** (F95 §4, `LITSWEEP6_RELGEN` §1
   wall 2, and any paper draft carrying "best exchange rate anywhere 1.1667"). The rate is not
   bounded; §6.3's rate-volume trade-off replaces it.
5. **Carry §4.3 forward as a provenance rule.** The frozen F95 module's trained-arm numbers are
   session-reproducible only to ~5 items; every closed-form quantity is bit-exact. Any future record
   that needs to gate against F95's MLP arm must gate against a **same-session re-run** (`--stage
   anchor`), not against the recorded JSON. F97's 78/78 parity claim was true when made and would not
   re-assert today.
6. **The relational asset remains analysis-grade only**, as `LITSWEEP6_RELGEN` §5 pre-committed and
   F97 recorded. This record does not reopen it: VSW is under bar on 3/3 and under the F47-gate
   benchmark on 3/3. **C3 (VEA, evidence ranking for the audit pillar) remains the one legal,
   unmeasured use of the verifier, and carries F95's binding "NEVER an accuracy claim" verbatim.**
7. **Litsweep-6 RELGEN is now fully executed**: C1 (VGA) and C2 (VNQ) killed at F97, **RELGEN-C4 (VSW)**
   killed here, C5 recorded as NO-GO in the sweep itself, C3 (VEA) is a writing task. **The RELGEN
   sweep has no untasked arm left.**
   > **NAMING-COLLISION CORRECTION, and it is a routing hazard.** There are **two different candidates
   > called "C4"**: `LITSWEEP6_RELGEN.md` §2 C4 = **VSW**, killed here; `LITSWEEP6_MEMBANK.md` C4 =
   > **aggregate-then-compare subspace residual**, which is **untouched, $0, and still live** — it is
   > nominated as the next candidate by `AGGNET_PREGATE_RECORD.md` §7.2 and `RESTRANS_PREGATE_RECORD.md`
   > §7.2, and F98 §7.1 explicitly places it **outside** its closure because its input is the retrieved
   > **vectors**, not their (cosine, label) profile. A reader who takes "C4 is killed" from this record
   > would skip the one live $0 candidate the family has. **Cite these hereafter as `RELGEN-C4 (VSW)`
   > and `MEMBANK-C4 (subspace residual)`.** Nothing in this record touches MEMBANK-C4 or MEMBANK-C2.

8. **Cross-reference `refine-logs/VSW_ASYMMETRY_RECON.md`** (independent $0 recon, same day, which
   re-verified this record's arms bit-exactly from the banked arena and reproduced every p-value at
   4 dp). It answers *why* HateMM converts and the other two do not, and its answer sharpens this
   record's closure rather than softening it:
   * **The binding constraint is BREAK EXPOSURE, not fix yield.** Fix yield is statistically identical
     across datasets (0.2500 / 0.2273 / 0.2645, **1.16× spread**); break exposure differs **5.33×**
     (0.0127 / 0.0448 / 0.0678). HateMM converts not because its errors are more reachable but because
     its correct set is 3.5-5.3× less fragile. That is a property of the deployed vote's confidence
     profile, not of the verifier — which is why a better verifier cannot fix MHC-ZH or MHC-EN.
   * **The cosine magnitude is decision-inert in this arena.** Replacing `cos_i` by the constant 1 in
     the deployed vote moves accuracy by **−0.0013 / +0.0000 / −0.0018** at **99.60 / 99.65 / 99.82 %**
     decision agreement — the deployed vote is, to within 0-2 items, a pure rank-weighted **sign** vote.
     Any future "restore/de-collapse the magnitude channel" proposal is repairing something worth ≤ 2
     items, and VSW's multiplier is in any case near-orthogonal to the cosine inside the top-20
     (median per-query Spearman 0.0767 / 0.0917 / 0.0962) — it substitutes an orthogonal relational
     order rather than sharpening magnitude.
   * **A $0 pre-check for any future re-weighting proposal**: the joint (flip-cost × verifier-direction)
     supply/exposure ratio reproduces the realised exchange rate to within 0.21 on 3/3 with **zero
     fitting**. Recommend it as the first gate on anything in this family.
   * Its §7.2 proposes an **additive** amendment to F98 §7.1 clause (a) — placing it is the
     orchestrator's call, not this record's; the substance is recorded in this pregate's ban scope.

### ▸ 10.9 ORCHESTRATOR RULING, 2026-07-28 — the §2.3 escalation resolves to **(i) ACCEPT CLOSURE ON THE NET**

*This is an append, not a rewrite. Every frozen bar, every measured number and §9.3's honest
"handed up rather than resolved here" stand exactly as written. What is added is the ruling §9.3
explicitly reserved to the user/orchestrator.*

**Ruling.** The §2.3 escalation clause fired on outcome (a) on 3/3 datasets and is recorded as fired.
It resolves to **option (i): accept the closure on the net.** §6.3 is carried into the paper as
analysis; the aggregation axis is **CLOSED**. **No prereg is written and no ceremony slot is spent.**

**Rationale, entirely from this record's own measurements:**

1. **A single-dataset HateMM prereg cannot serve the goal.** §5.3 measures the whole declared operator
   space **with full hindsight** — 3 families × 48 λ — and gets **+0.0069 on MHC-ZH (23 % of the bar)
   and +0.0164 on MHC-EN (55 %)**. The goal's conjunct requires ≥2 datasets. A ceremony that is
   arithmetically incapable of clearing the conjunct **cannot serve the goal**, and therefore does not
   warrant a ceremony slot regardless of how the HateMM cell reads.
2. **The net is pinned, and the mechanism says why.** K-VSW-2's own measurement holds the net at
   **+11 to +21 across a 16 384× λ range**, because precision decays monotonically with sharpness
   (HateMM **0.8571 at 21 changed → 0.5696 at 79 changed**) and exactly cancels the rise in volume.
   Against the requirement of **22.3 / 17.4 / 16.5 net items** (`LITSWEEP7_LANDING_SITE.md`), the
   ceiling of the entire continuum is **below bar on every dataset at every λ**. There is no point on
   the curve for a prereg to aim at.
3. **Honouring the letter of (a) would test a mechanism this record has already bounded.** §9.3 says
   this in the record's own words; the ruling simply accepts it.

**What is NOT closed by this ruling.** §6.3's rate/volume trade-off is a **paper-grade correction** and
is promoted, not buried — see 10.10. C3 (VEA) remains the one legal, unmeasured use of the verifier,
still carrying F95's "NEVER an accuracy claim" verbatim. MEMBANK-C4 is untouched (see the naming
collision box in item 7).

### ▸ 10.10 SUPERSESSION, 2026-07-28 — the exchange-rate bar is REFUTED **as a screening criterion**

*Propagated to every live record and queued candidate that screens on it.*

**Refuted:** *"exchange rate ≥ 1.2 on the pathology population"* as a **screen**. This record measured
**ER = 6.0000 on HateMM — 5× the bar — and the arm still FAILED.** A rate bar is scale-free and
therefore cannot bound the quantity the goal is denominated in. It survives only as a **diagnostic**
(reporting it remains useful; it is how §6.3 was found).

**The correct law:**

```
net = changed × (2·precision − 1)
```

**The binding screen is NET ITEMS**, against the measured requirement of
**22.3 (HateMM) / 17.4 (MHC-ZH) / 16.5 (MHC-EN)** net items
(`refine-logs/LITSWEEP7_LANDING_SITE.md`). Any candidate whose screen is phrased as an exchange rate
must be re-phrased as a net-item count before it is run; a rate bar passed does **not** license a run.

**Records carrying the superseded screen** (supersession notes added / owners notified):
`LITSWEEP6_MEMBANK.md` §2/§4/§6 frozen bars (four sites) — noted in place;
`MEMBANK_C4_PREGATE_RECORD.md` — **in flight at the time of this ruling, not edited; its owner must be
told** and is the orchestrator's action.

---

## §11. LIMITATIONS

1. **Arena (L1, inherited from F95 §6, restated by both sibling records, and it cuts AGAINST the
   negative verdict).** Banked **raw** encoder key space, **train** split, item-disjoint LOO — not
   the deployed head space and not test. A raw-space, train-side result entails nothing about head
   space or test in either direction. It is mitigated, not removed, by the fact that the bars that
   actually fire here are **relative** comparisons inside one arena under one protocol (VSW vs a
   global threshold shift, vs a fixed k, vs a cosine-only twin, vs its own label-shuffled null, and
   vs a hindsight-chosen λ from its own family), and relative comparisons are far less arena-sensitive
   than an absolute Δacc.
2. **One fold draw.** `FOLD_SEED = 0`, the frozen F95 assignment, no outer resampling. Sign evidence
   is **5 folds**, not 3 seeds — the raw features are seed-independent, as in both sibling records.
3. **One verifier seed and one verifier fit per fold** (`MLP_SEED = 0`), inherited from F95. The
   λ layer adds its own inner CV and a 200-draw permutation null, so the *selector's* variance is
   controlled; the *verifier's* is not. §4.3 shows the verifier's own cross-session variance is worth
   about 5 items on 744 — small, but not zero, and it is not resampled here.
4. **Residual coupling in the nesting (L3, declared in §3.4, inherited verbatim from VGA §2.4).** λ is
   never chosen on the fold it is applied to, but the fitting-pool items' verifier scores were
   produced by F95 verifiers whose fit sets included that fold's items. Removing it needs a
   doubly-nested verifier (25 fits per dataset instead of 5); the bias direction would, if anything,
   **flatter** the treatment arm, which failed anyway.
5. **Three monotone multiplier families, one candidate set, one k.** `pow`, `exp` and `lin` over the
   deployed top-20 at k = 20. A non-monotone multiplier is outside the candidate's own definition
   (§1 requires monotone). A different k is banned by F94. A different candidate set is F95 control
   2b's territory and costs −0.0293 to −0.0437 before any verifier runs.
6. **The exchange rates at small λ rest on small changed-item counts.** HateMM's headline 6.0000 is
   18 fixed / 3 broken on 21 changed decisions; MHC-ZH's 1.3000 is 13/10 on 23 and MHC-EN's 1.5000 is
   18/12 on 30. The ≥20-changed guard was declared in §2.3 *before* the run precisely because of this,
   but it is a floor on the count, not a confidence interval: at 23 changed items a precision of
   0.5652 is not distinguishable from 0.5000, so **MHC-ZH's and MHC-EN's outcome-(a) qualifications
   are statistically weak and should not be leaned on.** HateMM's is not weak — it holds at every one
   of its 23 non-zero λ, including 55 changed decisions at λ = 3.
7. **One degenerate item exists and is faithfully replayed.** HateMM carries one zero-norm key whose
   20 top-20 cosines are all exactly 0.000000 (§4.2). The deployed rule predicts 1 there and so does
   every VSW arm at every λ; it is 1 item in 744.
8. **No test-split file was opened, no test label read, no oracle used to select anything.** The two
   oracle quantities that appear (`ORACLE_lambda_pow`, and the pooled-hindsight ceiling of §5.3) are
   **reported ceilings computed on held-out train items**, never used to choose an arm, a λ or an
   operating point.

---

## §12. FILE MANIFEST

| path | contents |
|---|---|
| `scripts/analysis/vsw_pregate.py` | **FROZEN implementation**, sha256 `ba9982db…932eea`; imports `mechnov_pairverify.py` and `mechfix_ops.py` **unmodified** with both shas asserted at run time; contains the pre-freeze `--selftest`, the `--stage anchor` tier-2 parity reference, `--stage main` and `--stage perm` |
| `scripts/analysis/vsw_pregate_report.py` | reporting only; merges the per-stage JSON, derives the permutation p-values, renders every bar's verdict at 4 dp |
| `scripts/analysis/vsw_drive.sh` | orchestration only; re-invokes the frozen script after a login-node reap (§3.10) |
| `scripts/analysis/vsw_selftest_OUT.json` (+ `.log`) | §3.8 machinery validity: 3 synthetic arms × 60-draw nulls, multiplier monotonicity over 48 cells |
| `scripts/analysis/vsw_selftest_B0_ARCHIVE.json` | the superseded two-arm self-test, retained for the §3.8 disclosure |
| `scripts/analysis/vsw_f95anchor_{hatemm,zh,en}_OUT.json` (+ `.log`) | **tier-2 parity anchors**: the frozen F95 `run_space` re-run unmodified this session |
| `scripts/analysis/vsw_main_{hatemm,zh,en}_OUT.json` (+ `.log`) | main battery: gates, the K-VSW-2 curves (3 families × 24/13/11 λ), the λ-selected arms, DEG-A/B/D, class balance, oracle ceilings |
| `scripts/analysis/vsw_perm_{hatemm,zh,en}_OUT.json` (+ `.log`) | permutation battery: 200 draws × 5 folds of F95 verifier refits on shuffled fitting-pool labels, with the full curve per draw |
| `scripts/analysis/vsw_ckpt/**` | per-fold arena and per-(fold × draw) verifier-score checkpoints (§3.10 reap resilience). **Not committed** — 82 MB / 4 710 files of reproducible intermediate state, `.gitignore`d; every load-bearing number is in the committed `vsw_*_OUT.json` |
| `scripts/analysis/vsw_pregate_OUT.json` | merged result — **the file this record is read from** |

Read-only inputs: `data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_<model>.pt` (**train split only** —
`dev_seen` and `test_seen` were never opened by any script in this record),
`scripts/analysis/mechnov_pairverify.py` and `scripts/analysis/mechfix_ops.py` (imported unmodified,
shas asserted at run time), `scripts/analysis/mechnov_pairverify_{hatemm,zh,en}_OUT.json` (tier-1
parity anchors), `scripts/analysis/vga_pregate_OUT.json` and
`scripts/analysis/aggnet_pregate_OUT.json` (the §7.4 benchmarks, **re-read from source rather than
transcribed from the records**). Read for context, not modified: `LITSWEEP6_RELGEN.md`,
`MECHNOV_PAIRVERIFY_PREGATE.md`, `VGA_PREGATE_RECORD.md`, `AGGNET_PREGATE_RECORD.md`.
No file was deleted or moved outside this pregate's own outputs.
**Zero GPU, zero SLURM submissions, zero Modal calls, zero test contact.**
