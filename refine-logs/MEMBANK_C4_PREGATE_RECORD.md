# MEMBANK-C4 PREGATE — aggregate-then-compare (class-conditional subspace residual)

**Date:** 2026-07-28 NZST · **Agent:** membank-c4-pregate · **Cost: $0** (CPU only, ≤8 threads,
login node, **zero GPU, zero SLURM, zero Modal, zero test-split contact**). Env: conda `HateVideo`,
numpy 1.26.4, scipy 1.17.1, scikit-learn 1.5.2, torch 2.6.0+cu124 (CPU), faiss.

**Test-split contact: NONE.** The only data files any script in this record opens are
`data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_<model>.pt`. No `dev_seen`, no `test_seen`, no
`data/gt/*` (this record declares **no** length covariate, so no transcript file is read either).

**NAMING — read this before anything else.** There are **two** candidates called "C4" in litsweep-6.
`LITSWEEP6_RELGEN.md:256` **RELGEN-C4 = VSW** (verifier-shaped re-weighting; measured 1-for-3 and
dead, `VSW_PREGATE_RECORD.md`). `LITSWEEP6_MEMBANK.md:466` **MEMBANK-C4 = aggregate-then-compare
subspace residual** — *that is this record*, and it is referred to as **MEMBANK-C4** throughout.
The hazard was flagged by `VSW_ASYMMETRY_RECON.md` §7.3; this record adopts its recommended naming.

**Binding design source:** `refine-logs/LITSWEEP6_MEMBANK.md` §4(a)–(f), read in full before any code
was written; its §4(e) frozen bars are quoted **verbatim** in §2.1 below, before any number in this
document was computed.

**Nominated independently by two records, both read in full first:**
`refine-logs/AGGNET_PREGATE_RECORD.md` §7.2 item 2 — *"Next candidate: C4 (aggregate-then-compare
subspace residual) … unaffected by this closure (§7.1). It is $0, CPU, and its degeneracy bar (both
class residuals collapse at every rank `r`) is already written."* — and
`refine-logs/RESTRANS_PREGATE_RECORD.md` §7.2 — *"C1 closed. Route next to C4 (aggregate-then-compare
subspace residual) … C4's own $0 pregate is untouched by anything measured here."*

**Also read in full first:** `refine-logs/MECHNOV_PAIRVERIFY_PREGATE.md` (the frozen F95 harness this
pregate reuses verbatim) and `refine-logs/VSW_ASYMMETRY_RECON.md` (three measured facts of
2026-07-28 that reshape the control set — §0.3 below).

---

## §0. WHAT IS UNDER TEST, AND THE FOUR FACTS THAT SHAPE IT

### 0.1 The deployed decision (the floor)

`src/utils/metrics.py:262-301`, `src/model/evaluate_rac.py:405-465`, replayed by the F89-frozen
`mechfix_ops.deployed_vote`:

```
v = Σ_i (2·lab_i − 1)·cos_i·w_i / Σ_i w_i ,   top-20 own-train neighbours,  w = [20,19,…,1]
predict 1 iff v ≥ 0
```

Write it in **aggregate-then-compare form**, which is an algebraic identity, not a change:

```
s_c = Σ_{i : lab_i = c} cos_i·w_i / Σ_i w_i           (c ∈ {0,1})
v = s_1 − s_0 ,  predict 1 iff s_1 ≥ s_0
```

So the deployed rule *is already* "aggregate each class, then compare", with the per-class aggregator
being a **rank-weighted signed-cosine sum**. This identity is what makes a bit-exact parity assert
possible (§2.4 PARITY-IMPL) and it fixes precisely what MEMBANK-C4 changes: **only the per-class
aggregator**.

### 0.2 MEMBANK-C4 (LITSWEEP6_MEMBANK §4(d), transplanted verbatim)

> **Changes:** the decision rule, replaced by a residual gap. Retrieve the deployed top-20; split by
> gold label; for each class `c` form a rank-`r` basis from that class's retrieved members (`r` small,
> ridge-regularised); predict 1 iff `residual_0 − residual_1 > 0`. **Stays:** encoder, head, keys,
> retrieval, k = 20 — the *candidate set is exactly the deployed one*, which is what keeps this out of
> F95's nomination family.

Concretely, with keys L2-normalised (‖q‖ = 1) and `U_c` an orthonormal rank-`r` basis of class `c`'s
retrieved members,

```
residual_c(q)² = ‖q − U_c U_cᵀ q‖² = 1 − ‖U_cᵀ q‖²
predict 1 iff residual_0 − residual_1 > 0     (⟺ q has more energy in class 1's span)
```

**The mechanical appeal, stated as LITSWEEP6 §4(c) states it:** a single correct analogue can *span*
the query — contributing to its class's representation — without having to *out-vote* nineteen
wrong-class neighbours. This is RelationNet's k-shot composition order (aggregate the support set
first, then compute **one** relation per class) applied to the deployed memory bank.

**Why it is not inside any existing closure, stated before the run so it is falsifiable:**

* **vs F98 / AGGNET (C3).** F98's closure is explicitly limited: *"the closure covers operators whose
  input is the **(cosine, label) profile** of the deployed top-20"* (`AGGNET_PREGATE_RECORD.md:711-713`).
  MEMBANK-C4's input is the retrieved **vectors**. F98's own §7.1 names it as **not closed**
  (`:717-720`), as does the VSW amendment text (`VSW_ASYMMETRY_RECON.md:459-461`).
* **vs F95 / MECHNOV.** F95's ban is on *nomination + per-pair verification*. MEMBANK-C4 does not
  re-nominate (the candidate set is bit-identically the deployed top-20) and has **no pair scorer at
  all** — no function of type (query, bank-item) → score exists anywhere in it.
* **vs F89 re-metrication.** A projection residual onto a **query-dependent** subspace cannot be
  written as `cos(Az, Az′)` for any fixed `A`.
* **vs F94 global-k.** No re-weighting of the rank profile occurs; the operator is count-blind at
  fixed rank (§1.4).
* **vs F99 / VSW.** VSW re-weights the fixed rank profile `[20..1]` by a verifier-driven multiplier;
  the amended clause (a) closes re-weightings of that profile. MEMBANK-C4 is not a re-weighting of it
  — at fixed rank the profile is discarded entirely.

### 0.3 The three measured facts of 2026-07-28 that reshape the control set

From `refine-logs/VSW_ASYMMETRY_RECON.md`, all measured in the *same* raw fused train arena this
record uses:

1. **The cosine magnitude is decision-inert in the deployed vote** (§6.1). Setting `cos_i := 1` moves
   accuracy by **−0.0013 / +0.0000 / −0.0018** with **0.9960 / 0.9965 / 0.9982** decision agreement.
   The deployed vote is, to within 0–2 items of 549–744, a **rank-weighted label count**.
   → **Consequence, binding on this record:** an arm whose effect is mediated by similarity
   *magnitude* rather than by top-k *membership* is worth ≤ 2 items and is dead on arrival. The
   **MEM-MAG control** (§2.3) is installed to detect exactly that, and it is a KILL, not a caveat.
   This matters more for MEMBANK-C4 than for any previous candidate, because at `r = 1` the class
   basis of a cone-concentrated member set is essentially its **mean direction**, so `C4_pca_r1` is
   *a priori* close to a mean-cosine comparison — the very object the recon priced at ≤ 2 items.
2. **The binding constraint is break exposure, not fix yield** (§3.3). Fix yield is statistically
   identical across datasets (**0.2500 / 0.2273 / 0.2645**, 1.16× spread); break exposure differs
   **5.33×** (**0.0127 / 0.0448 / 0.0678**) and traces to the fusion tax (text-stream purity ~0.85 vs
   image ~0.60 on ZH/EN).
   → **Consequence:** the **EXPOSURE PRE-CHECK** (§2.5), the $0 gate proposed at
   `VSW_ASYMMETRY_RECON.md:455-457`, is installed and is **read BEFORE the accuracy read**. If the
   projected net is capped under the bar by the exposure arithmetic at **every** declared cell, the
   candidate dies there.
3. **The head collapses the cone** (§1.4). Raw fused has ~2.4 % cosine dynamic range
   (median top-1 0.9444 / 0.9524 / 0.9407; median c₁−c₂₀ 0.0253 / 0.0221 / 0.0231) against the
   deployed head space's ~1e-4 (top-1 0.999852).
   → **Consequence, stated up front and repeated in §8:** this pregate runs in **raw** space. A
   raw-space result **does not entail** a head-space result (F97 limitation (1)). The asymmetry is
   named in §8: a raw-space *degeneracy* would be **worse** in head space (spans of near-identical
   vectors are more nearly identical), so a degeneracy verdict transfers a fortiori, while a
   *positive* would not transfer at all without a head-space measurement.

### 0.4 The fourth fact: the conditional-signal benchmark already banked in this arena

`VGA_PREGATE_RECORD.md` §4.3: the F47-feature adjudication gate delivers **+0.0269 / +0.0104 /
+0.0182** (p = 0.0050 / 0.0050 / 0.0100) in this exact arena and was ruled **analysis-grade only**.
It is the standing benchmark for what a conditional operator on this neighbourhood is worth. It is
quoted here so that a MEMBANK-C4 result between +0.010 and +0.0269 is visibly *below an already
banked and already-declined number*, rather than being reported as novel headroom.

---

## §1. FROZEN HARNESS

### 1.1 Arena — the F95 harness verbatim

LITSWEEP6_MEMBANK §4(e): *"Same 5-fold item-disjoint harness; **training-free** (a least-squares
projection). $0, minutes. Full version: 0 GPU-h."*

Banked **RAW encoder key spaces** (seed-independent), **train split only**, item-disjoint
`StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` (`mechnov_pairverify.K_FOLDS`,
`.FOLD_SEED`), **PRIMARY space = fused** = `l2n(concat(l2n(img), l2n(txt)))`, 7168-d; `text` and `img`
SECONDARY. This is the arena F95 froze and that RESTRANS, VGA, AGGNET and VSW all used. The trained
RGCL head is **not** the arena: it memorises its own train split (LOO train acc 0.998, F47), so a
train-side screen in head space measures memorisation.

Consequence: this arena has **no seeds** (raw encoder features are seed-independent, and MEMBANK-C4 is
training-free, so it has no initialisation seed either), so sign evidence is **per fold**, not per
seed — as in all four sibling records.

Per-item construction: for each fold, the bank is the fitting pool and the queries are the held-out
items; `mechfix_ops.deployed_vote(X[fit], lab[fit], X[ho], topk=20)` is the identical call that
produces the floor. No held-out item is ever a bank row of its own fold. Fitting-pool quantities
(the DEG-A threshold, the `C4_sel` inner CV) use leave-one-out **inside** the fitting pool
(`exclude_self=True`), so no item ever contributes to a quantity used to decide its own label.

### 1.2 The operator, frozen in full

For held-out query `q` with deployed top-20 `(idx_1..idx_20, cos_1..cos_20)` in deployed rank order,
and `lab` the **fitting-pool gold labels** (the bank labels the deployed vote already uses):

1. Split the 20 by bank label: `Mem_c = [idx_i : lab_{idx_i} = c]`, kept **in deployed rank order**.
   `n_c = |Mem_c|`.
2. **Count matching (PRIMARY, declared).** `m = min(n_0, n_1)`; keep the **first `m`** members of each
   class in deployed rank order. This removes, by construction, the count bias that a subspace
   comparison otherwise carries (a class with more members spans more of the space and therefore has
   a smaller residual for *any* query). The uncorrected variant is run as a SECONDARY arm
   (`*_nomatch`) so the size of the bias is priced rather than assumed.
3. **Rank matching (PRIMARY, declared).** `r_eff = min(r, m)` for **both** classes.
4. `A_c` = the `m × d` matrix of that class's kept member key vectors (L2-normalised, raw space).
   Residual is computed through the Gram matrix `G_c = A_c A_cᵀ` (`m ≤ 20`) and the query row
   `b_c = A_c q`, so no `d`-dimensional decomposition is ever formed:
   * **PCA family** (`C4_pca_r{r}`): eigendecompose `G_c = W Λ Wᵀ`, keep the top `r_eff` eigenpairs
     with `Λ_j > EIG_TOL`; then `‖U_cᵀ q‖² = Σ_{j ≤ r_eff} (W_jᵀ b_c)² / Λ_j`, and
     `residual_c² = 1 − ‖U_cᵀ q‖²` (uncentered span through the origin — the keys live on a cone and
     the span of the members is the object LITSWEEP6 names).
   * **Ridge family** (`C4_ridge_g{γ}`): `α = (G_c + γ·I)⁻¹ b_c`,
     `residual_c² = 1 − 2·αᵀb_c + αᵀG_cα`. Uses all `m` count-matched members; `γ` is the declared
     regulariser LITSWEEP6 §4(d) requires.
5. `d_gap = residual_0 − residual_1`; **predict 1 iff `d_gap > 0`**.

**Declared fallbacks (fixed before the run, not repairs).**
* `n_0 = 0` or `n_1 = 0` (a class-pure top-20): no basis exists for one class, and by construction no
  aggregate-then-compare rule is defined. **Fall back to the deployed prediction.** The fraction of
  held-out items with a class-mixed top-20 is the **hard ceiling on MEMBANK-C4's reach** and is
  reported *before* any Δ (§2.3 coverage).
* `d_gap == 0` exactly: **fall back to the deployed prediction.**
* `EIG_TOL = 1e-12` on eigenvalues; any eigenpair below it is dropped (a numerically rank-deficient
  member set), so `r_eff` can be smaller than `min(r, m)` and the *realised* `r_eff` distribution is
  reported.

### 1.3 Arms — every one declared here, none added later

`r ∈ {1, 2, 3, 5}` and ridge regularisation are **arms, not tuning** — LITSWEEP6 §4(d)'s explicit
instruction ("*must be declared before the run*"). They are all reported; none can be selected on the
held-out fold.

| id | status | what |
|---|---|---|
| `C4_pca_r1`, `C4_pca_r2`, `C4_pca_r3`, `C4_pca_r5` | **declared family** | the LITSWEEP6 grid, count- and rank-matched, raw PRIMARY space |
| `C4_pca_rfull` | declared family | `r_eff = m`, the full count-matched span (the `r → n_c` endpoint) |
| `C4_ridge_g0.001`, `_g0.01`, `_g0.1`, `_g1.0` | declared family | the ridge form over all `m` count-matched members |
| **`C4_sel`** | **PRIMARY (deployable)** | the cell chosen by inner CV on the fitting pool (§1.5). This is the only arm that can carry bar 1 |
| `C4_oracle` | ORACLE ceiling | the best cell chosen on the **held-out** fold — a ceiling only, **never a pass**; it is the object the EXPOSURE PRE-CHECK gate is read on |
| `C4_pca_r{1,2,3,5}_nomatch` | SECONDARY | no count matching (`r_eff = min(r, n_c)` per class) — prices the count bias |
| `C4_pca256_r{1,2,3,5}` | SECONDARY | the same operator inside a fold-fitted **PCA-256** reduction (`mechnov_pairverify.PCA_DIM`), because LITSWEEP6 §4(f) names *"~10 vectors per class in a 128-256-d reduced space"* as the degeneracy risk it is most worried about |
| `DEPLOYED` | floor | `mechfix_ops.deployed_vote`, parity-asserted (§2.4) |

Controls (each defined in §2.3): `THRESH_best` (DEG-A), `FIXK_{1,2,3,5,7,10,15,20}` (DEG-B),
`MAGTWIN_max` / `MAGTWIN_mean` (MEM-MAG), `SIGNVOTE` (the membership channel), `NULL2_geom`
(geometry null).

### 1.4 Why the operator is count-blind at fixed rank, stated as a falsifiable claim

At `r_eff` fixed and equal across classes and with the member counts matched to `m`, the two class
scores are computed from **the same number of vectors with the same basis rank**. The only thing that
can differ is **where those vectors sit relative to `q`**. This is the precise sense in which
MEMBANK-C4 is not a vote: three class-1 members that happen to span `q` beat seventeen class-0
members that do not. `C4_pca_r{r}_nomatch` measures what the uncorrected version does, so the claim
is checkable rather than asserted.

### 1.5 `C4_sel` — the deployable arm, and its mandatory ability to return the floor

Inner `StratifiedKFold(n_splits=5, shuffle=True, random_state=17)` **inside the fitting pool** (the
VGA §2.4 / AGGNET §1.4 nesting precedent). Inner grid = **`{DEPLOYED}` ∪ {the nine declared C4
cells}**; each inner split retrieves its own top-20 from its own inner-fitting bank; the cell with the
highest mean inner accuracy is applied to the held-out fold using the full fitting pool as bank.
**Ties break toward `DEPLOYED`**, then toward the earlier member of the declared order.

Including `DEPLOYED` in the inner grid is not a convenience: it is what gives the harness the property
AGGNET §2.5 arm C had to be redesigned to obtain — **when there is nothing to find, the deployable arm
returns the floor bit-exactly**, so a null is a property of the data rather than of an operator that
was forced to move. The self-test (§2.6 arm C) is the check that it actually does this.

### 1.6 Permutation null (mandatory)

`N_PERM = 200`, `PERM_SEED = 12345` (the VGA / VSW constants; MEMBANK-C4 is training-free so the
draws are cheap and there is no reason to run AGGNET's reduced 100).

**What is shuffled, and why it is the only possible null here.** MEMBANK-C4 has no trained target;
its *only* label input is the **fitting-pool bank labels**, which is what performs the class split.
Each draw permutes `lab` **within the fitting pool of each fold** (held-out labels untouched;
retrieval, geometry and the top-20 candidate sets untouched — they are label-independent) and re-runs
the **full pipeline**: class split, count matching, basis construction, residual comparison, the inner
CV that selects the cell, and the held-out evaluation.

**This is a paired null and the record must read it as one.** Shuffling bank labels also destroys the
deployed floor, so each draw yields `Δ = acc(C4_sel) − acc(DEPLOYED)` on the *same* corrupted bank.
The null therefore tests exactly: *does MEMBANK-C4's advantage over the deployed vote on the same
neighbourhood require the true label assignment?* Reported
`p = (1 + #{null ≥ observed}) / (N_PERM + 1)`, significance at `p < 0.05`, on the PRIMARY arm ×
PRIMARY space for all three datasets.

**Null informativeness is reported, and it is not optional.** F98's null was *significant but
uninformative* — not one of 300 draws reached zero, so an arm that merely fell back to the floor
passed automatically. This record reports **`frac_null_ge_0`**, the fraction of draws reaching
`Δ ≥ 0`, on every dataset, and a null with `frac_null_ge_0` near 0 is declared **UNINFORMATIVE** and
cannot support the arm regardless of its `p`.

**Second null (`NULL2_geom`, declared, not a permutation).** Keep the class **composition** of each
top-20 exactly (same labels, same counts, same `m`) but build each class's basis from `m` **random**
fitting-pool items of that class instead of the retrieved ones (seeded, `GEOMNULL_SEED = 7`,
`N_GEOM = 20` draws, mean reported). This isolates whether the residual gap carries *neighbourhood
geometry* or merely a class-prior/count artefact. If `C4 ≈ NULL2_geom`, the operator is not reading
the retrieved analogues at all.

---

## §2. FROZEN BARS AND CONTROLS

### 2.1 Quoted verbatim from `LITSWEEP6_MEMBANK.md` §4(e)

> **Frozen bars:** (1) Δacc ≥ +0.010, 5/5 fold signs, ≥3/5 positive; (2) exchange rate ≥ 1.2;
> (3) **degeneracy control fired before anything else** — the distribution of `residual_0 − residual_1`
> must be non-degenerate at some declared `r`; if the two residuals are near-identical at every `r`,
> the arm is void and reports nothing about aggregate-then-compare; (4) class-balance sanity.

### 2.2 The pregate bar, quoted verbatim from the tasking

> Bars: pregate promote requires Delta acc >= +0.010 on >= 1 dataset with 5/5 fold signs >= 0 and
> >= 3/5 strictly positive, plus exchange rate >= 1.2 on the pathology population, plus all controls
> clean. The frozen full-version bar is +0.030 acc AND +0.030 mF1 on >= 2 of 3 datasets, 3/3 seeds —
> and per the user's ruling of 2026-07-28 the working protocol is **final-epoch**.

**Read on:** `C4_sel` × PRIMARY space (fused), pooled over all held-out items (each train item held
out exactly once), against the deployed floor on the same fitting-fold bank. Secondary spaces and
secondary arms are reported but **cannot carry a bar**. `C4_oracle` is a ceiling and can never pass.

**Context, not a bar:** the standing conditional-signal benchmark in this arena is **+0.0269**
(§0.4). A result in [+0.010, +0.0269] clears LITSWEEP6's interest threshold while remaining below an
already-declined number; the verdict states that explicitly rather than hiding it.

### 2.3 Mandatory controls — declared before the run, **any firing = KILL**

Gate order is **not** negotiable and follows LITSWEEP6's own instruction that the degeneracy control
*"fires before anything else"*:

**GATE 0 — PARITY (§2.4). A hard assert that aborts the run.** No treatment number exists if it fails.

**GATE 1 — DEGENERACY (MEMBANK-C4's own, LITSWEEP6 bar 3). Fires first among the verdict gates.**
Define the relative residual gap `g_i = |d_gap_i| / (½·(residual_0 + residual_1))_i` on class-mixed
items. A cell is **DEGENERATE** iff `median_i g_i < DEG_TOL` with **`DEG_TOL = 0.01`** (a 1 % relative
gap — the same tolerance constant AGGNET §2.4 used for `MONO_TOL`). Also reported per cell:
`frac(g < DEG_TOL)` and the median of `residual_0`, `residual_1` separately.
**If EVERY declared cell is degenerate, the arm is VOID**: it reports **nothing** about
aggregate-then-compare, the result is recorded as *untested-by-degeneracy*, and — per LITSWEEP6
§4(f) — this is banked as a **harness** null, explicitly **not** a mechanism null.

**GATE 2 — EXPOSURE PRE-CHECK (§2.5). Read BEFORE the accuracy read.** If the projected net is capped
under bar 1 at **every** declared cell on **all three** datasets, MEMBANK-C4 dies at the arithmetic and
the deployable-arm accuracy read is reported as a formality only.

**GATE 3 — DEG-A, threshold-shift twin.** `THRESH_best` = the deployed vote with a **global** decision
threshold `τ` chosen to maximise **fitting-pool LOO** accuracy (exact optimum over all thresholds:
midpoints of consecutive distinct fitting votes, plus `τ = 0` and both open ends — `aggnet_pregate.
best_threshold`, reused verbatim). Pooled agreement between `C4_sel` and `THRESH_best`
**≥ `DEG_KILL` = 0.95 ⇒ KILL** (the RESTRANS §5.3 verdict form; F96 measured 95.03/97.75/99.45 %,
F98 measured 0.9570 / 0.9508).

**GATE 4 — DEG-B, fixed-k twin.** `FIXK_k` = the deployed vote restricted to profile `[k..1, 0×(20−k)]`
over the same top-20, `k ∈ {1,2,3,5,7,10,15,20}` (the F94 grid). `max_k` pooled agreement with
`C4_sel` **≥ 0.95 ⇒ KILL** (MEMBANK-C4 has re-derived a member of the closed global-k family).

**GATE 5 — MEM-MAG, membership-vs-magnitude (NEW, from `VSW_ASYMMETRY_RECON.md` §6.1).** Let
`CH = {i : p_C4(i) ≠ p_dep(i)}` be the changed decisions.
* **magnitude channel twins**, both aggregate-then-compare over the *identical* class split, with the
  subspace replaced by a pure magnitude statistic:
  `MAGTWIN_max` predicts 1 iff `max_{i:lab_i=1} cos_i ≥ max_{i:lab_i=0} cos_i`;
  `MAGTWIN_mean` predicts 1 iff `mean` of the count-matched class-1 cosines `≥` class-0's.
  (`MAGTWIN_max` is F95's own `cos_shape` control restricted to the deployed candidate set.)
* **magnitude-mediated fraction** `= max over the two twins of |{i ∈ CH : p_twin(i) = p_C4(i)}| / |CH|`.
  **≥ 0.95 ⇒ KILL** — the effect is a mean/max-cosine comparison in a subspace costume, and the recon
  measured that channel at ≤ 2 items of 549–744.
* **membership-mediated fraction** `= |{i ∈ CH : p_C4(i) ≠ p_SIGNVOTE(i)}| / |CH|`, where `SIGNVOTE`
  is the deployed vote with `cos_i := 1` — the recon's own object, i.e. the pure rank-weighted label
  count. Reported alongside as the complementary read.
* The kill fires on the dataset that would carry a bar; the fractions are reported for all three. If
  `|CH| = 0` the control is reported as N/A and the arm is a no-op, which is itself a verdict.

**GATE 6 — CLASS BALANCE.** `C4_sel`'s held-out positive rate must sit within **0.10** of the bank
positive rate (**0.4005 / 0.3109 / 0.3060**, re-read this session from the train caches). Outside that
band the nulls are declared **VOID** (a collapsed or inflated positive rate makes a permutation
comparison meaningless).

**GATE 7 — PERMUTATION NULL (§1.6).** `N_PERM = 200`, `PERM_SEED = 12345`, fitting-pool-only shuffle,
full pipeline per draw, `p = (1 + #{null ≥ obs})/(N_PERM + 1)`, **plus the mandatory
`frac_null_ge_0` informativeness read.**

**Coverage (reported before any Δ, not a gate).** `frac_class_mixed` = the fraction of held-out items
whose deployed top-20 contains both classes. On a class-pure neighbourhood MEMBANK-C4 is a declared
no-op (§1.2), so this is the hard ceiling on its reach.

**Exchange rate (bar 2).** F95's definition — `fixed / broken` over all held-out items — reported
together with the count of the **pathology population** fixed, where the pathology population is
F95's: deployed-wrong held-out items whose nearest same-gold-class bank item sits within rank
`PATHOLOGY_RANK = 5`. Directly comparable to F95's 0.53–0.95 (ceiling 1.1667), RESTRANS's
0.2647–0.9474, AGGNET's 1.8333 / 0.8400 / 1.0000 and VSW's 2.1176 / 0.8889 / 1.0476.

### 2.4 PARITY — GATE 0, two independent asserts, both abort on mismatch

**PARITY-ARENA (90 cells).** `sha256(mechfix_ops.py)` must equal
`635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` and
`sha256(mechnov_pairverify.py)` must equal
`77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d`; both are imported **unmodified**.
The deployed floor recomputed inside this harness must reproduce F95's recorded train-side numbers at
**4 dp, per cell**, read from `mechnov_pairverify_{hatemm,zh,en}_OUT.json`: pooled `acc_deployed`,
`mF1_deployed`, `posrate_deployed`, all five per-fold `acc_deployed`, and the integer counts
`n_deployed_wrong`, `n_pathology_pop` — for every dataset × every space =
**3 × 3 × (3 + 5 + 2) = 90 asserted cells.**

**PARITY-IMPL (the degenerate setting, bit-exact).** Per §0.1 the deployed rule *is* an
aggregate-then-compare rule whose per-class aggregator is the rank-weighted signed-cosine sum. The
MEMBANK-C4 engine run in `DEPLOYED_AGG` mode — identical code path, identical class split, identical
count/rank matching **disabled** (the deployed rule uses all members), with the residual aggregator
swapped for that sum — must reproduce `mechfix_ops.deployed_vote` **bit-for-bit in predictions and to
< 1e-12 in the score `s_1 − s_0`**, on **every fold × every dataset × every space** (45 cells). This
proves the treatment differs from the floor **only** in the per-class aggregator.

**PARITY-RECON (3 × 2 cells, cross-check on the fact this record's controls are built from).**
`SIGNVOTE` (the deployed vote with `cos_i := 1`) must reproduce
`VSW_ASYMMETRY_RECON.md` §6.1 at 4 dp: accuracy **0.8427 / 0.8480 / 0.7778** and agreement with the
deployed vote **0.9960 / 0.9965 / 0.9982**. Aborts on mismatch. This gate exists because GATE 5 is
built on that measurement; if it does not reproduce, the control is not trustworthy.

**Soft cross-checks (reported, do NOT abort)** against the same recon, because the construction
details are not fully pinned by its text: median flip cost on deployed-CORRECT / deployed-WRONG
(recon §3.4: 0.3422/0.2521/0.2180 and 0.1583/0.1021/0.0921), median top-20 purity on
CORRECT/WRONG (0.85/0.75/0.70 and 0.325/0.40/0.40), and the fraction of deployed-CORRECT items with
flip cost ≤ 0.10 (recon §3.6 native-bank rows: 0.0653 / 0.1283 / 0.2118).

### 2.5 EXPOSURE PRE-CHECK — GATE 2, the $0 gate installed per `VSW_ASYMMETRY_RECON.md` §7.2

The recon's amendment text (`:455-457`) makes this a standing requirement:

> It can only escape by exhibiting a fix-supply/break-exposure ratio > 1 **at a radius where enough
> items are changed to matter** (§3.2 of `VSW_ASYMMETRY_RECON.md`), which is a $0 pre-check on banked
> geometry that any such proposal must now pass before it is written.

**Adaptation to MEMBANK-C4, stated honestly.** The recon's statistic uses **flip cost** — the minimum
probability mass that must move between the 20 rank weights to cross zero — because VSW is a
*re-weighting* and flip cost is exactly its budget. MEMBANK-C4 is **not** a re-weighting, so flip cost
does not bound it. The transferable half of the statistic is the one that carried the recon's finding:
the **supply/exposure decomposition against the operator's own direction field**, which for
MEMBANK-C4 is simply its predicted class and is computable with **zero fitting** from geometry plus
the fitting-pool labels:

```
n_ERR = #{held-out, deployed wrong}            n_COR = #{held-out, deployed correct}
fix supply     = #{deployed wrong   ∧ p_C4 = gold}     fix yield      = supply  / n_ERR
break exposure = #{deployed correct ∧ p_C4 ≠ gold}     exposure rate  = exposure / n_COR
projected net  = supply − exposure             ER = supply / exposure
```

**Disclosure, because it changes how the gate should be read.** For a re-weighting operator the
recon's pre-check is *predictive* — it estimates an exchange rate before the operator is built. For
MEMBANK-C4 the same decomposition is **exact**: `projected net / n` **is** Δacc. That makes the gate
strictly stronger, not weaker: it cannot be gamed, and taken over the **whole declared cell grid** it
is a **full-hindsight ceiling** — the same argument form as `VSW_ASYMMETRY_RECON.md` §4.1
(*"no selector can beat a hindsight ceiling"*). Read on `C4_oracle`, i.e. the best cell per dataset
chosen with hindsight.

**Required net for the bars** (`ceil(bar × n)`; n = 744 / 579 / 549):

| bar | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| +0.010 (pregate) | **8** | **6** | **6** |
| +0.030 (full version) | **23** | **18** | **17** |

**GATE 2 fires (KILL) iff** `max over all declared cells of (projected net)` is **< the +0.010 row on
all three datasets**. It is reported per cell regardless, together with the recon's own
flip-cost-stratified form at `θ ∈ {0.10, 0.20}` so the numbers are directly comparable to
`VSW_ASYMMETRY_RECON.md` §3.2–3.3.

### 2.6 Machinery validity — synthetic controls, run BEFORE the freeze

`membank_c4_pregate.py --selftest`, **synthetic data only**, at the real problem's scale
(n = 699, 5 folds, class rate ≈ 0.4, d = 256). **No real-dataset number was computed before the sha
in §2.7 was frozen.** Four arms, each with a pre-declared required outcome. All results below are
read from `scripts/analysis/membank_c4_selftest_OUT.json`.

| arm | construction | required outcome | **measured** | |
|---|---|---|---|---|
| **S-A subspace-planted** | items come in *concepts* of 3 that share a 2-d basis and belong to one class; class-0 items carry less off-cone mass so they systematically outrank class-1 items in cosine, dragging the rank-weighted label count toward the majority class while leaving the subspace structure intact | a **clear positive**, Δ ≥ +0.05 | floor **0.6180**, `C4_sel` **+0.2060**, best cell `C4_ridge_g0.1` **+0.2103**, class-mixed 0.7454; the DEG-A twin gets only **+0.0243**, so the arena is not a disguised threshold shift | **PASS** |
| **S-B noise** | labels independent of geometry | Δ ≈ 0 **and** the permutation null non-significant | floor **0.5694**, `C4_sel` **−0.0143**; permutation (100 draws) **p = 0.7327**, null mean −0.007969 ± 0.012839, **`frac_null_ge_0` = 0.39** — i.e. the null machinery is *informative*, unlike F98's | **PASS** |
| **S-C deployed-optimal** | the deployed vote is already optimal; its residual errors are not a function of the retrieved geometry | `C4_sel` returns the floor **bit-exactly** | floor **0.9070**, `C4_sel` **+0.0000**, agreement **1.0000**, `DEPLOYED` selected in **5/5** folds — and **class-mixed coverage 0.9828**, so the arm declined to move on 98 % of items where it *could* have | **PASS** |
| **S-D rank discrimination** (unit test) | `q = (u₁+u₂)/√2`; the gold class has 4 members all at cosine **0.7071** (three near `u₁`, one at `u₂`), the wrong class 4 near-collinear members all at cosine **0.8000** | deployed **and** the magnitude twin must be WRONG, rank 1 WRONG, rank ≥ 2 RIGHT | deployed → 0 ✗, `MAGTWIN_max` → 0 ✗, `C4_pca_r1` → 0 ✗ (gap −0.1071), `C4_pca_r2/r3/r5/rfull` → **1 ✓** (gap +0.5934 … +0.6000), all four ridge cells → 1 ✓ | **PASS** |

**S-D is the control that makes GATE 5 meaningful and it is declared for that reason.** A
membership/magnitude decomposition is worthless if the magnitude twin agrees with the treatment no
matter what; S-D exhibits a configuration where the subspace comparison is decisive and the magnitude
twin is *wrong*, so a high magnitude-mediated fraction on real data will be a fact about the data
rather than an artefact of the control.

> **Disclosed pre-freeze repair (not a silent fix).** The first two drafts of the S-A generator were
> rejected by their own required outcome and replaced **before the freeze**: draft 1 (class-specific
> subspaces plus one shared nuisance direction) put the deployed floor at **1.0000** — no headroom, so
> the control was vacuous; draft 2 (a 40-d nuisance basis at amplitude 0.35 against a 2-d class basis
> at 0.12) put the floor at 0.7371 but the *planted* class directions ranked below the per-item
> nuisance directions in the member scatter, so no rank-`r` basis could recover them and the arm
> returned **−0.0028**. The failure was in the **generator**, not in the operator: the third design
> plants concept structure that is actually recoverable at low rank and the same unmodified operator
> immediately returns +0.2060. **No bar, no control, no operator constant and no threshold was
> changed at any point** — only `_synth`, and `SYNTH_A`/`SYNTH_C_NOISE` are recorded in the frozen
> script. S-C's noise level was likewise raised (1.6 → 6.0) because at 1.6 the floor saturated at
> 1.0000 and the "returns the floor" check was vacuous.

### 2.7 Frozen script sha

| path | sha256 |
|---|---|
| **`scripts/analysis/membank_c4_pregate.py`** | freeze sha **`33367f9a71c5e6203ddf82a27e31e1686c7bdcc552196e5574b7c6b48d47c4a3`**; run sha **`9970876c90f53ef117d707d5131b49135673dcc954cd2b8d379f9a945314ec6c`** after ERRATUM E-1 (§3.2) — the *only* difference is that PARITY-RECON was demoted from a hard abort to a reported cross-check and a second tie-break column was emitted. No bar, arm, control threshold or operator line differs between the two shas. |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` (F89, imported unmodified, sha asserted at run time) |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` (F95, imported unmodified, sha asserted at run time) |

There is no separate report script: §3 onward is filled in directly from
`membank_c4_{hatemm,zh,en}_OUT.json` and `membank_c4_perm_{hatemm,zh,en}_OUT.json`, re-read at 4 dp.

### 2.8 PRE-RUN AMENDMENT — 2026-07-28, after the self-test, before any real-data number

Two results landed from other agents while §0–§2.7 were being written. Both are folded in **here**,
before the first treatment number exists, rather than being retro-fitted to the verdict.

**(a) The exchange-rate bar is refuted as a *screening* criterion; the binding screen is NET ITEMS.**
`VSW_PREGATE_RECORD.md` §9.2 (commit `e9a17fe`, F105) measured exchange rates of **6.0000** on HateMM
— exceeding 1.2 at all 23 non-zero λ, against F95's best-in-36-cells of 1.1667 — and **still failed**,
because *"rate and volume are in a binding trade-off (`net = changed · (2·precision − 1)`), precision
decays monotonically with sharpness on all three datasets, and the product is pinned below the bar at
every point of a 16 000× λ range."* Its instruction is explicit: *"Anyone citing 'the exchange rate
never exceeds ~1.2' as a law of this system must stop."*

**Consequence for this record.** LITSWEEP6's bar 2 (ER ≥ 1.2) is **retained as declared** — it is a
quoted frozen bar and this record does not get to delete one — but it is **demoted to a reported
diagnostic and is explicitly NOT a screen**. The binding screen is **GATE 2, the net-item exposure
pre-check of §2.5**, which was already written in the net-item form and is now promoted to the
decisive arithmetic gate. The required-net table of §2.5 (8/6/6 at +0.010, 23/18/17 at +0.030) is
restated here as the operative screen; F105's own measured requirement is quoted as
**22.3 / 17.4 / 16.5** items, i.e. `0.030 × n` before the ceiling — the same numbers.
**A MEMBANK-C4 cell with a high exchange rate and a net below the table does not pass anything.**

**(b) Determinism tolerance — and why it does not touch this record's parity gates.**
A confirmed defect: re-running frozen modules unmodified on the same node/env/caches/seeds reproduces
**closed-form** quantities bit-exactly but **drifts on trained ones** (44 of 48 cells in the F95
module), residual cause oneDNN/MKL kernel selection.

**Consequence for this record: none of its numbers are exposed, and this is checkable rather than
asserted.** MEMBANK-C4 is **training-free** — the operator is an eigendecomposition of a ≤ 20 × 20
Gram matrix and a comparison of two residuals; the floor is faiss inner-product search plus fixed
arithmetic; DEG-A is an exact threshold enumeration; DEG-B, MEM-MAG, SIGNVOTE and the flip cost are
closed-form. **Every quantity in this record is closed-form, so PARITY-ARENA, PARITY-IMPL and
PARITY-RECON are asserted at full strength and a failure would be a real defect, not kernel drift.**
The one place the defect could enter is the F95 JSON that PARITY-ARENA compares against — but the
cells this record reads from it (`acc_deployed`, `mF1_deployed`, `posrate_deployed`, per-fold
`acc_deployed`, `n_deployed_wrong`, `n_pathology_pop`) are **all** deployed-floor quantities, i.e.
exactly the closed-form subset that was measured to reproduce bit-exactly. No trained F95 quantity
(the MLP/logistic verifier arms) is read anywhere in this record.

Gate order, restated as the execution contract:
**§0–§2 written and shas frozen → synthetic self-test → GATE 0 parity → GATE 1 degeneracy → GATE 2
exposure pre-check → accuracy read → GATE 3-6 → GATE 7 permutation → verdict.**
Machine output: `scripts/analysis/membank_c4_{hatemm,zh,en}_OUT.json`,
`membank_c4_perm_{hatemm,zh,en}_OUT.json`, run logs `…_OUT.log`. Every number in §3 onward is
re-read from those JSONs at report time, 4 dp.

---

<!-- EVERYTHING BELOW THIS LINE WAS WRITTEN AFTER THE RUN -->

*(pre-run freeze ends here; §3 onward is filled in from the machine output)*

## §3. WHAT WAS RUN, AND ONE DISCLOSED ERRATUM

### 3.1 Runs

| artefact | content |
|---|---|
| `scripts/analysis/membank_c4_selftest_OUT.json` | the four pre-freeze synthetic arms of §2.6 (110 s) |
| `scripts/analysis/membank_c4_{hatemm,zh,en}_OUT.json` (+ `.log`) | all arms × all 3 spaces, all gates, all controls |
| `scripts/analysis/membank_c4_perm_{hatemm,zh,en}_OUT.json` (+ `.log`) | GATE 7, `N_PERM = 200`, 223 / 172 / 165 s |

Three datasets × three spaces, CPU only, 4 threads per process, three processes in parallel on the
login node. **Total wall time under 8 minutes. Zero GPU, zero SLURM, zero Modal, zero test contact.**

### 3.2 ERRATUM E-1 — PARITY-RECON demoted from a hard abort to a reported cross-check

**What happened.** The frozen §2.4 PARITY-RECON gate aborted all three datasets on its first run.
It asserts that `SIGNVOTE` (the deployed vote with `cos_i := 1`) reproduces
`VSW_ASYMMETRY_RECON.md` §6.1 at 4 dp. It did not, by **1–2 items** per dataset.

**Diagnosis, done before anything was changed.** The residual is entirely a **tie convention**. With
`cos_i := 1` the score becomes `(Σ ±w_i)/210` with integer weights, so **exact ties at `v = 0` are
reachable** — and they occur on **2 / 2 / 3** items (HateMM / ZH / EN). This record's `SIGNVOTE`
inherits the deployed rule's `v ≥ 0 ⇒ 1`; the recon evidently used strict `v > 0`. Measured both ways
(`controls.signvote` in each `_OUT.json`):

| dataset | declared `≥ 0` acc / agree | recon-convention `> 0` acc / agree | recon §6.1 target | ties |
|---|---|---|---|---|
| HateMM | 0.8414 / 0.9973 | 0.8441 / 0.9973 | 0.8427 / 0.9960 | 2 |
| MHC-ZH | 0.8480 / **1.0000** | 0.8480 / **0.9965** | 0.8480 / **0.9965** | 2 |
| MHC-EN | 0.7760 / 0.9964 | **0.7778 / 0.9982** | **0.7778 / 0.9982** | 3 |

Under the recon's convention **MHC-EN reproduces exactly on both quantities and MHC-ZH reproduces its
agreement exactly.** HateMM remains 1 item off in both directions (the target 0.8427 sits *between*
the two conventions) — and the VSW record itself discloses a HateMM-specific anomaly that predicts
exactly this: **one zero-norm HateMM key**, `vsw_main_hatemm_OUT.json:cos_diagnostic`
(`n_zero_norm_keys: 1`, `n_items_affected: 1`), quoted at `VSW_ASYMMETRY_RECON.md:161-164`. A
zero-norm bank key contributes `cos = 0` to the deployed vote and `cos = 1` after the substitution,
so it is precisely the row on which two implementations of "set the cosine to 1" can differ.

**What was changed, and what was not.** `PARITY-RECON` was demoted from a hard abort to a **reported**
cross-check emitting both conventions and the tie count. **`SIGNVOTE`'s definition was NOT changed to
chase the target** — that would be tuning a control to a number. No bar, no arm, no control threshold
and no operator line differs between the freeze sha and the run sha (§2.7).

**Why the demotion does not weaken GATE 5.** GATE 5 rests on the recon's *qualitative* fact — that
the deployed vote is a rank-weighted label count to within a couple of items. This record measures
that fact **independently and slightly more strongly**: agreement **0.9973 / 1.0000 / 0.9964**
(2 / 0 / 2 items of 744 / 579 / 549) under its own declared convention. And three *other* recon
quantities reproduce **exactly**, which is what makes the tie-break diagnosis credible rather than
convenient (§4.5): median flip cost on correct/wrong items **3/3 exact**, median top-20 purity
**3/3 exact**.

---

## §4. RESULTS — gates in the frozen order

### 4.1 GATE 0 — PARITY: **PASS**

| gate | requirement | measured | verdict |
|---|---|---|---|
| module shas | `mechfix_ops.py` = `635c1312…c83fc8d`, `mechnov_pairverify.py` = `77b0defd…8b7240d` | asserted at run time on every invocation, both matched | **PASS** |
| **PARITY-ARENA** | 90 cells (3 ds × 3 spaces × [3 pooled + 5 per-fold + 2 counts]) at 4 dp vs the frozen F95 JSONs | **90 / 90** (logged `PARITY-ARENA 10/10` on each of the 9 dataset × space cells) | **PASS** |
| **PARITY-IMPL** | the C4 engine in `DEPLOYED_AGG` mode reproduces `mechfix_ops.deployed_vote` bit-for-bit in predictions, < 1e-12 in score, every fold × dataset × space | **45 / 45 folds bit-exact**, max score \|Δ\| **1.11e-16** on all nine cells | **PASS** |
| PARITY-RECON | *(demoted, §3.2)* | reported; EN exact, ZH agreement exact, HateMM 1 item (zero-norm key) | reported |

Per §2.8(b) every asserted quantity here is **closed-form**, so the confirmed oneDNN/MKL
trained-estimator drift does not apply and these asserts stand at full strength.

**Deployed floors, re-derived in this harness** (fused): **0.8441 / 0.8480 / 0.7796** acc,
**0.8419 / 0.8281 / 0.7286** mF1, `n_deployed_wrong` **116 / 88 / 121**, `n_pathology_pop`
**88 / 79 / 109** — identical to F95.

### 4.2 GATE 1 — MEMBANK-C4's own DEGENERACY control: **DOES NOT FIRE**

LITSWEEP6 §4(e) bar 3 and §4(f) both named degeneracy as *"the most likely killer"* and warned that a
degenerate result would be *"a harness null, not a mechanism null"*. **It is not degenerate.** Median
relative residual gap `|d₀−d₁| / (½(d₀+d₁))` over class-mixed held-out items, against
`DEG_TOL = 0.01`:

| cell | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| `C4_pca_r1` | 0.1039 | 0.0889 | 0.0657 |
| `C4_pca_r2` | 0.1065 | 0.0960 | 0.0709 |
| `C4_pca_r3` | 0.1066 | 0.0973 | 0.0748 |
| `C4_pca_r5` | 0.1093 | 0.0987 | 0.0748 |
| `C4_pca_rfull` | 0.1087 | 0.1015 | 0.0760 |
| `C4_ridge_g0.001` | 0.1087 | 0.1015 | 0.0760 |
| `C4_ridge_g0.01` | 0.1089 | 0.1014 | 0.0764 |
| `C4_ridge_g0.1` | 0.1071 | 0.0964 | 0.0739 |
| `C4_ridge_g1.0` | 0.0565 | 0.0606 | 0.0480 |
| **frac below tolerance** | 0.0341–0.0681 | 0.0396–0.0521 | 0.0582–0.0994 |
| **`all_cells_degenerate`** | **False** | **False** | **False** |

The two class residuals differ by a median of **5–11 %**, i.e. **5–11× the declared tolerance**, at
every one of the 27 cells. Median realised `r_eff` is 1–4 (HateMM), 1–6 (ZH), 1–7 (EN), so the ranks
actually bound and the arms are distinct objects rather than one object relabelled.

**This is the load-bearing result of the whole record**, because it removes the escape hatch
LITSWEEP6 wrote for this candidate. The verdict below is a **mechanism kill on a measured,
well-conditioned operator**, not an "untested-by-degeneracy" non-result.

**And the sweep's specific reduced-space worry is measured FALSE.** LITSWEEP6 §4(f) feared that
*"~10 vectors per class in a 128-256-d reduced space"* would make both spans near-universal. In the
declared `C4_pca256_*` SECONDARY arms the median relative gap is **0.1466 / 0.1220 / 0.0996** — larger
than in raw space, not smaller.

### 4.3 GATE 2 — EXPOSURE PRE-CHECK: **FIRES → KILL**, and it is a full-hindsight ceiling

Read **before** the accuracy read, on `C4_oracle` = the best of the nine declared cells chosen **with
hindsight on the evaluation data**. Per `VSW_ASYMMETRY_RECON.md` §4.1, no selector can beat a
hindsight ceiling.

| dataset | n | n_ERR | n_COR | **best cell** | fix supply | break exposure | **net** | **required (+0.010)** | required (+0.030) |
|---|---|---|---|---|---|---|---|---|---|
| HateMM | 744 | 116 | 628 | `C4_ridge_g1.0` | 49 | 69 | **−20** | **+8** | +23 |
| MHC-ZH | 579 | 88 | 491 | `C4_ridge_g1.0` | 38 | 43 | **−5** | **+6** | +18 |
| MHC-EN | 549 | 121 | 428 | `C4_ridge_g0.1` | 44 | 57 | **−13** | **+6** | +17 |

The looser **per-fold** oracle (a different cell allowed in every fold) is **−18 / −1 / −6** — still
negative on all three. **0 of the 27 primary-space cells is positive on any dataset**; the per-cell
range is HateMM −0.0269 … −0.0376, MHC-ZH −0.0086 … −0.0155, MHC-EN −0.0237 … −0.0364.

**GATE 2 fires on all three datasets**: the ceiling is not merely under the bar, it is **below zero**.
Per §2.8(a) this net-item screen — not the exchange rate — is the binding criterion, and it is
arithmetic rather than a matter of effort or budget.

**The mechanism, in the recon's own decomposition, and it is the sharpest instance yet:**

| | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| **MEMBANK-C4 fix yield** (supply / n_ERR) | **0.4224** | **0.4318** | **0.3636** |
| VSW fix yield (`VSW_ASYMMETRY_RECON.md` §3.3) | 0.2500 | 0.2273 | 0.2645 |
| **MEMBANK-C4 break exposure** (exposure / n_COR) | **0.1099** | **0.0876** | **0.1332** |
| VSW break exposure (§3.3) | 0.0127 | 0.0448 | 0.0678 |
| exchange rate | 0.7101 | 0.8837 | 0.7719 |

**MEMBANK-C4 reaches the errors better than anything this campaign has measured** — fix yield
**0.36–0.43** against VSW's 0.23–0.26, i.e. it repairs 36–43 % of every deployed error — **and it
still loses, because its break exposure rises by more: 8.65× / 1.96× / 1.96× VSW's, against a fix
yield only 1.69× / 1.90× / 1.37× VSW's.** This is `VSW_ASYMMETRY_RECON.md`
§9's law restated on an operator that is not a re-weighting at all: *convertibility is set by the
fragility of the correct set, not by the reachability of the errors.*

The recon's flip-cost-stratified form of the same statistic, for the ceiling cell
(`controls.joint_supply_exposure`):

| θ | HateMM supply / exposure / ratio | MHC-ZH | MHC-EN |
|---|---|---|---|
| 0.10 | 26 / 9 / **2.8889** | 20 / 23 / **0.8696** | 27 / 30 / **0.9000** |
| 0.20 | 44 / 28 / **1.5714** | 31 / 36 / **0.8611** | 40 / 47 / **0.8511** |

HateMM's ratio **exceeds 1.2 inside the cheap stratum** (2.8889 at θ = 0.10) and the pooled net is
still **−20**, because the items outside that stratum are broken faster than the cheap ones are
fixed. This is an independent, non-re-weighting confirmation of F105's finding that a high exchange
rate on a favourable sub-population buys nothing (§2.8(a)).

### 4.4 The accuracy read (`C4_sel`, deployable) — bar 1 **FAIL** on all three

| | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| deployed floor | 0.8441 | 0.8480 | 0.7796 |
| `C4_sel` acc | 0.8401 | 0.8307 | 0.7687 |
| **Δacc** | **−0.0040** | **−0.0173** | **−0.0109** |
| ΔmF1 | −0.0054 | −0.0275 | −0.0113 |
| fold Δ | `−0.0201, 0, 0, 0, 0` | `−0.0086, −0.0259, −0.0948, +0.0431, 0` | `0, 0, −0.0545, 0, 0` |
| fold signs ≥ 0 (need **5/5**) | **4/5** | **2/5** | **4/5** |
| fold signs > 0 (need **≥3/5**) | **0/5** | **1/5** | **0/5** |
| fixed / broken / net | 13 / 16 / **−3** | 26 / 36 / **−10** | 6 / 12 / **−6** |
| changed | 29 | 62 | 18 |
| exchange rate | 0.8125 | 0.7222 | 0.5000 |
| pathology fixed (of 88 / 79 / 109) | 13 | 25 | 6 |
| cell chosen per fold | `ridge_g1.0`, DEP, DEP, DEP, DEP | `ridge_g1.0`, `pca_r5`, `ridge_g0.1`, `pca_r2`, DEP | DEP, DEP, `pca_r5`, DEP, DEP |
| class-mixed coverage | 0.8683 | 0.8290 | 0.9709 |

**Bar 1 fails on every clause on every dataset** — the Δ is negative, the 5/5 fold-sign clause fails
2/3 times, and the ≥3/5 strictly-positive clause fails 3/3. **Bar 2 (ER ≥ 1.2)** also fails
(0.8125 / 0.7222 / 0.5000), and per §2.8(a) it is reported as a diagnostic and is not what the
verdict turns on. **Coverage was not the constraint**: MEMBANK-C4 was free to act on 83–97 % of items.

Note the inner CV mostly declined the operator — `DEPLOYED` was selected in 4/5, 1/5 and 4/5 folds —
which is the §1.5 fallback working correctly. On MHC-ZH, where the selector *did* engage in 4/5 folds,
the loss is the largest (−0.0173).

### 4.5 GATES 3–6 — every declared control is **CLEAN** (none fires)

| gate | kill threshold | HateMM | MHC-ZH | MHC-EN | verdict |
|---|---|---|---|---|---|
| **DEG-A** threshold twin | ≥ 0.95 | **0.8965** | **0.8826** | **0.9144** | **clean** |
| **DEG-B** fixed-k twin | ≥ 0.95 | **0.9610** (k = 20) | **0.8946** (k = 15) | **0.9672** (k = 20) | **clean at the declared 0.95 line**, see note |
| **MEM-MAG** magnitude-mediated | ≥ 0.95 | **0.5862** | **0.5161** | **0.7222** | **clean** |
| MEM-MAG membership-mediated | (reported) | **1.0000** | **1.0000** | **1.0000** | — |
| **CLASS BALANCE** | dev > 0.10 | 0.0525 (0.4530 vs 0.4005) | 0.0104 (0.3005 vs 0.3109) | 0.0382 (0.2678 vs 0.3060) | **clean** |

**Note on DEG-B, stated so the boundary is visible rather than hidden.** HateMM 0.9610 and MHC-EN
0.9672 sit *above* 0.95 as raw agreements with `FIXK_20`, but `FIXK_20` **is the deployed rule itself**
— agreement with it is `agree_deployed` (0.9610 / 0.8929 / 0.9672), i.e. the statement that `C4_sel`
changed few decisions, not that it re-derived a *different* member of F94's closed family. The
declared gate compares `C4_sel` to the eight fixed-k profiles and its argmax is `k = 20` on HateMM/EN
and `k = 15` (0.8946) on ZH. Reading it as a kill would be reading "the selector often fell back to
the floor" as "the operator is a disguised fixed-k vote". **It is recorded as clean, and the verdict
does not rest on it either way** — GATE 2 already fired at the arithmetic.

**GATE 5 is the informative one and it does not fire, which matters.** `VSW_ASYMMETRY_RECON.md` §6.1
made magnitude-mediation the disqualifier for any new candidate. MEMBANK-C4 is **not** magnitude-
mediated: only **0.5862 / 0.5161 / 0.7222** of its changed decisions are reproduced by the best of the
two pure-magnitude twins, and **1.0000 / 1.0000 / 1.0000** of them depart from the rank-weighted label
count (`SIGNVOTE`). Corroborating, the magnitude twins are *far worse* arms in their own right
(`MAGTWIN_max` −0.0417 / −0.0293 / −0.0437; `MAGTWIN_mean` −0.0188 / −0.0103 / −0.0073), so
MEMBANK-C4 is not a mean/max-cosine comparison in a subspace costume. **The operator is exactly the
membership-channel object finding 1 said was worth testing, and it loses anyway.**

**`NULL2_geom` (geometry null) confirms the operator reads real neighbourhood geometry.** Replacing
each class's retrieved members by `m` random same-class fitting-pool items (20 draws, seed 7) —
identical class composition, no neighbourhood information — costs **−0.1397 / −0.1785 / −0.1897**
against the floor, i.e. **0.11 to 0.18 accuracy worse than MEMBANK-C4 itself**. The residual gap is
carrying genuine retrieved-analogue signal; it simply does not convert to accuracy.

**Recon cross-checks (soft, §2.4) — 3 of 3 quantities reproduce exactly:**

| quantity | this record | `VSW_ASYMMETRY_RECON.md` | |
|---|---|---|---|
| median flip cost, deployed-CORRECT | **0.3422 / 0.2521 / 0.2180** | §3.4 0.3422 / 0.2521 / 0.2180 | **exact 3/3** |
| median flip cost, deployed-WRONG | **0.1583 / 0.1021 / 0.0921** | §3.4 0.1583 / 0.1021 / 0.0921 | **exact 3/3** |
| median top-20 purity, CORRECT | **0.85 / 0.75 / 0.70** | §3.4 0.85 / 0.75 / 0.70 | **exact 3/3** |
| median top-20 purity, WRONG | 0.675 / 0.60 / 0.60 | §3.4 0.325 / 0.40 / 0.40 | **exact complements** (this record measures purity w.r.t. the *prediction*, the recon w.r.t. *gold*; on wrong items the two are `1 − x`) |
| frac CORRECT with flip cost ≤ 0.10 | 0.0669 / 0.1283 / 0.2126 | §3.6 native-bank rows 0.0653 / 0.1283 / 0.2118 | ZH exact; HateMM/EN within 0.0016 / 0.0008 (the recon's row came from a seeded bank-resampling sweep) |

An independent reimplementation of the recon's flip-cost optimal transport reproduces all six median
values to 4 dp. That is what licenses §3.2's reading of the `SIGNVOTE` residual as a tie convention.

### 4.6 GATE 7 — PERMUTATION NULL: **FAIL**, against an **INFORMATIVE** null

`N_PERM = 200`, `PERM_SEED = 12345`, fitting-pool bank labels shuffled within each fold, full
pipeline (class split, count matching, basis construction, inner-CV cell selection, held-out
evaluation) re-run per draw. From `membank_c4_perm_{ds}_OUT.json`:

| dataset | observed Δacc | **p** | null mean ± sd | null max | **`frac_null_ge_0`** | informative? |
|---|---|---|---|---|---|---|
| HateMM | −0.0040 | **0.6368** | −0.00543 ± 0.013658 | +0.0484 | **0.585** | **YES** |
| MHC-ZH | −0.0173 | **0.9900** | −0.000354 ± 0.002604 | 0.0000 | **0.980** | **YES** |
| MHC-EN | −0.0109 | **0.9950** | −0.000182 ± 0.001363 | 0.0000 | **0.980** | **YES** |

MEMBANK-C4 is **indistinguishable from — in fact worse than — the label-shuffled null on all three**.
The mandatory informativeness read passes comfortably (0.585 / 0.980 / 0.980 of draws reach Δ ≥ 0),
so unlike F98 — where **not one of 300 draws** reached zero and an arm that merely fell back to the
floor passed automatically — this null can distinguish a real effect from a fallback, and it reports
none.

The ZH/EN nulls concentrate at exactly 0.0000 because under shuffled bank labels the inner CV
correctly selects `DEPLOYED`. That makes the observed negatives **worse than chance in a precise
sense**: with *true* labels the inner CV was persuaded to engage the operator (4/5 folds on ZH) and
paid for it, whereas with random labels it declined and paid nothing.

### 4.7 Secondary spaces and secondary arms (cannot carry a bar; reported in full)

| ds / space | floor | `C4_sel` Δ | folds ≥ 0 | hindsight ceiling (net) | required (+0.010) |
|---|---|---|---|---|---|
| HateMM / fused **(PRIMARY)** | 0.8441 | **−0.0040** | 4/5 | **−20** | +8 |
| HateMM / text | 0.8441 | +0.0094 | 3/5 | +6 | +8 |
| HateMM / img | 0.7688 | +0.0000 | 5/5 | −12 | +8 |
| MHC-ZH / fused **(PRIMARY)** | 0.8480 | **−0.0173** | 2/5 | **−5** | +6 |
| MHC-ZH / text | 0.8636 | −0.0000 | 5/5 | −6 | +6 |
| MHC-ZH / img | 0.7012 | +0.0086 | 4/5 | **+16** | +6 |
| MHC-EN / fused **(PRIMARY)** | 0.7796 | **−0.0109** | 4/5 | **−13** | +6 |
| MHC-EN / text | 0.8106 | −0.0000 | 5/5 | −25 | +6 |
| MHC-EN / img | 0.6995 | −0.0110 | 2/5 | −11 | +6 |

**The single cell in the whole record whose hindsight ceiling clears the interest threshold is
MHC-ZH / img** (`C4_pca_r5`, +16 net, +0.0276 with full hindsight). It is reported rather than buried,
and it is not a lever: the img floor is **0.7012**, i.e. **14.68 accuracy points below** the deployed
fused floor of 0.8480 on the same dataset, so the entire ceiling recovers a fifth of the gap the space
itself creates; the *deployable* arm there is **+0.0086**, still under +0.010; and img is a declared
SECONDARY space that cannot carry a bar. `VSW_ASYMMETRY_RECON.md` §3.6 already measured why the img
space behaves differently (purity 0.60 vs fused 0.70), and F44/F50/F85/F86 closed the stream-selection
axis it belongs to.

**Count-matching control (`*_nomatch`, SECONDARY).** Dropping the §1.2 count matching *improves*
HateMM (`C4_pca_r5_nomatch` **+0.0094** vs matched **−0.0336**; `r3` +0.0067 vs −0.0282) and does
nothing elsewhere (ZH 0.0000 … −0.0086; EN −0.0073 … −0.0146). **This is not a positive result and
must not be read as one.** Without count matching the class with more retrieved members gets a
strictly larger span and therefore a strictly smaller residual for *any* query, so the uncorrected
operator silently re-imports the neighbourhood's class-count prior — i.e. it partially reverts to the
very rank-weighted count vote MEMBANK-C4 was proposed to replace. The gain is the count vote leaking
back in, and it still does not reach +0.010 anywhere.

**Reduced-space arms (`C4_pca256_*`, SECONDARY):** −0.0161 / −0.0323 / −0.0296 / −0.0376 (HateMM),
0.0000 / −0.0155 / −0.0069 / −0.0121 (ZH), −0.0383 / −0.0437 / −0.0510 / −0.0401 (EN) for
`r ∈ {1,2,3,5}`. Uniformly negative; the reduction does not rescue anything.

---

## §5. VERDICT

# **KILL — MEMBANK-C4 is measured, non-degenerate, membership-mediated, permutation-controlled, and arithmetically incapable: its full-hindsight ceiling is NEGATIVE on all three datasets.**

| bar / gate | requirement | measured | verdict |
|---|---|---|---|
| **GATE 0 PARITY** | 90 arena cells at 4 dp + 45 folds bit-exact | 90/90; 45/45, max \|Δ\| 1.11e-16 | **PASS** |
| **GATE 1 DEGENERACY** (LITSWEEP6 bar 3, fires first) | void if both residuals near-identical at every `r` | median relative gap **0.0480–0.1093**, 5–11× the 0.01 tolerance, **0 of 27 cells degenerate** | **DOES NOT FIRE — the arm is genuinely tested** |
| **GATE 2 EXPOSURE PRE-CHECK** (the binding screen, §2.8(a)) | hindsight ceiling ≥ +8 / +6 / +6 net items | **−20 / −5 / −13** pooled; **−18 / −1 / −6** per-fold-oracle; **0 of 27 cells positive** | **FIRES → KILL** |
| **bar 1** (LITSWEEP6) | Δacc ≥ +0.010 on ≥1 ds, **5/5** folds ≥ 0, ≥3/5 > 0 | **−0.0040 / −0.0173 / −0.0109**; folds ≥ 0 = 4/5, 2/5, 4/5; folds > 0 = **0/5, 1/5, 0/5** | **FAIL on every clause** |
| **bar 2** exchange rate | ≥ 1.2 on the pathology population | 0.8125 / 0.7222 / 0.5000 (cells 0.6364–0.8837) | **FAIL** — reported as a diagnostic only, per §2.8(a) |
| **bar 4** class balance | within 0.10 of bank rate | 0.0525 / 0.0104 / 0.0382 | **PASS** (nulls valid) |
| **DEG-A** threshold twin | < 0.95 | 0.8965 / 0.8826 / 0.9144 | **clean** |
| **DEG-B** fixed-k twin | < 0.95 | 0.9610 / 0.8946 / 0.9672 — argmax `k=20` = the deployed rule itself (§4.5) | **clean** |
| **MEM-MAG** (new) | magnitude-mediated < 0.95 | **0.5862 / 0.5161 / 0.7222**; membership-mediated **1.0000 / 1.0000 / 1.0000** | **clean — the effect is membership-mediated** |
| **GATE 7 permutation** | must beat the label-shuffled null | **p = 0.6368 / 0.9900 / 0.9950**, nulls **INFORMATIVE** (`frac_null_ge_0` 0.585 / 0.980 / 0.980) | **FAIL** |
| frozen full-version bar | +0.030 acc AND +0.030 mF1 on ≥2/3, 3/3 seeds, final-epoch | not approached by any cell in any space; ceiling negative | **not reached** |

**Cost: $0.** CPU only, ≤ 8 threads, login node, under 8 minutes wall time, zero GPU, zero SLURM,
zero Modal, zero test-split contact. Full version, had it promoted, would have been 0 GPU-h.

### 5.1 Why this kill is worth more than a null

Every previous member of this campaign that died could be dismissed with *"the operator never really
got a different kind of information"*. MEMBANK-C4 cannot be dismissed that way, and the record has
the four measurements to prove it:

1. **It is not degenerate** (GATE 1: 5–11 % residual separation at all 27 cells). LITSWEEP6 wrote the
   degeneracy bar precisely so that a collapse would be recorded as *untested*. It did not collapse.
2. **It is not the inert magnitude channel** (GATE 5: magnitude-mediated only 0.52–0.72;
   membership-mediated 1.0000 on 3/3). It is exactly the membership-channel operator
   `VSW_ASYMMETRY_RECON.md` §6.1 said a candidate must be.
3. **It is not blind** — `NULL2_geom` shows it is 0.14–0.19 accuracy *better* than the same operator
   fed random same-class members, so the residual gap reads genuine retrieved-analogue geometry.
4. **It reaches the errors better than anything measured here** — fix yield **0.4224 / 0.4318 /
   0.3636**, against VSW's 0.2500 / 0.2273 / 0.2645 and against the vote's own error population. It
   repairs 36–43 % of every deployed error.

**And it is still net-negative at every one of 27 cells, on all three datasets, with hindsight.**
The reason is the one the recon isolated the same day. MEMBANK-C4's fix yield exceeds its break
exposure by **3.84× / 4.93× / 2.73×** — and that is *not enough*, because the correct set is
**5.41× / 5.58× / 3.54×** larger than the error set. `net = n_ERR·yield − n_COR·exposure` =
**−20.0 / −5.0 / −13.0** items, negative before any tuning is attempted, and no selection over the
declared space can change the sign of a hindsight ceiling.

The comparison against VSW is the sharp form: raising the reach helped (fix yield **1.69× / 1.90× /
1.37×** VSW's) and cost more than it bought (break exposure **8.65× / 1.96× / 1.96×** VSW's).

**The law this adds, stated for the paper.** *Changing the composition order of a retrieval-memory
decision — aggregating each class's retrieved support into one representation and comparing once,
instead of comparing pairwise and pooling — measurably increases the fraction of errors reached
(0.25 → 0.42 fix yield) and measurably increases the fraction of correct decisions destroyed
(0.013–0.068 → 0.088–0.133 break exposure) by more, on all three datasets. The composition order is
not the binding constraint; the fragility of the correct set is.* This is the **second** independent
operator family (after VSW/F105) to demonstrate that reachability is not the limiting resource, and
the **first** to do so with a fix yield above 0.35.

### 5.2 F95's open question, now answered

`LITSWEEP6_MEMBANK.md` §4(c) framed this candidate as *"the direct answer to the question F95 left
open … F95 measured **compare-then-aggregate** … **Aggregate-then-compare has never been measured
here.**"* It has now.

| composition order | operator | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|---|
| compare-then-aggregate (max) | F95 pair-verification | −0.0040 | −0.0466 | −0.0146 |
| compare-then-aggregate (mean-top-3) | F95 | +0.0054 | −0.0345 | −0.0383 |
| **aggregate-then-compare** | **MEMBANK-C4 (this record)** | **−0.0040** | **−0.0173** | **−0.0109** |

Both orders are negative on 3/3 (5/6 F95 cells, 3/3 here). **The −0.029 to −0.044 "decision-shape
cost" F95 paid is not an artefact of comparing before aggregating** — reversing the order recovers
part of it on MHC-ZH (−0.0466 → −0.0173) and none of it on HateMM (−0.0040 → −0.0040) or MHC-EN
(−0.0146 → −0.0109), and never reaches the floor. **The shape cost is a property of replacing the
vote, not of the order in which the replacement composes.** That closes the question the sweep
opened, in the direction the sweep hoped it would not.

---

## §6. ROUTING CONSEQUENCE

### 6.1 What is now closed

**MEMBANK-C4 is the 5th and last candidate of `LITSWEEP6_MEMBANK`. The membank sweep is EXHAUSTED as
a performance family.**

| # | candidate | status | evidence |
|---|---|---|---|
| C1 | residual-transport vote | **DEAD** | F96 / `RESTRANS_PREGATE_RECORD.md` §7 — a threshold shift in an item-level costume on 95–99 % of items |
| C2 | cell-conditional synthesis into the bank (BSY) | **pre-closed, behind a user ban ruling** — not re-opened by this record; its prereg still carries the `RESTRANS` §6 rewrite requirement (its placement criterion cannot use `p̂`) | `RESTRANS_PREGATE_RECORD.md` §6, `AGGNET_PREGATE_RECORD.md` §7.2 item 3 |
| C3 | learned aggregation profile network | **DEAD** | F98 / `AGGNET_PREGATE_RECORD.md` §7 — 0/45 cells at +0.030, DEG-A and DEG-B both fire |
| **C4** | **aggregate-then-compare subspace residual** | **DEAD (this record)** | 0/27 cells positive, hindsight ceiling **negative** on 3/3, p = 0.6368 / 0.9900 / 0.9950 |
| C5 | per-entry soft reliability weights | **dropped as a performance candidate**; retained for its pillar-④ auditability role only | `AGGNET_PREGATE_RECORD.md` §7.1(d) |

**Answer to the tasking's question: yes — the membank family is exhausted.** No arm of
`LITSWEEP6_MEMBANK` remains untested as a performance candidate. C2 is the only member not killed by
measurement, and it is not available: it sits behind a standing ban ruling and an unrewritten prereg.

### 6.2 What must NOT be re-proposed

Adding to `AGGNET_PREGATE_RECORD.md` §7.1's list, on this neighbourhood object:

* **(e) Any re-ordering of the decision's composition** — aggregate-then-compare, compare-then-
  aggregate, per-class prototype/centroid/mean-representation comparison, RelationNet-style k-shot
  support aggregation, class-conditional subspace or reconstruction residual (ProCon / SubspaceAD
  form), and per-class span/projection scores at **any** rank `r ∈ {1,2,3,5,full}` with **any** ridge
  `γ ∈ [1e-3, 1]`, in **raw fused, text, img or PCA-256** geometry. 27 primary-space cells measured,
  **0 positive**, hindsight ceiling negative on 3/3.
* **(f) "It failed because the residuals collapsed" is not available as a reason to retry.** GATE 1
  measured a 5–11 % median residual separation at every cell, and measured the sweep's specific
  reduced-space fear (`AGGNET`-style 128–256-d) to be **false** (gaps are *larger* there, 0.0996–0.1466).
* **(g) "A better/richer per-class representation" is not the fix.** Rank is measured not to bind
  (r = 1 through full span differ by ≤ 0.011 acc on every dataset), ridge is measured not to bind
  (γ over three decades differ by ≤ 0.011), and the count-matching control shows the only direction
  that *improves* the arm is the one that re-imports the count vote (§4.7).
* **(h) The exchange rate is not a screen.** MEMBANK-C4 reaches ER **2.8889** inside HateMM's cheap
  stratum with a pooled net of **−20**. Second independent confirmation of F105's §9.2 correction.

### 6.3 What is NOT closed by this record

* The **head space**. This is a raw-space, train-side screen (§8). A raw-space negative does not
  entail a head-space negative — though see §8.1 for why the transfer argument is unusually weak in
  *this* direction for a positive and strong for the degeneracy read.
* **LITSWEEP6-MEMBANK C2** (membership creation) — still the only operator anyone has proposed that
  changes *which items are retrievable at all*. Unchanged in status: behind the ban ruling, prereg
  rewrite outstanding.
* Anything outside the deployed top-20 neighbourhood object. This record touched only the per-class
  aggregator.

### 6.4 Routing recommendation

1. **Do not spend GPU. Do not promote. Do not ceremony.** The ceiling is negative; there is nothing
   to validate.
2. **Do not commission a MEMBANK-C4 variant.** §6.2(e)–(g) enumerate the space that was measured, and
   the binding constraint is not in it.
3. **The membank sweep needs no further pregates.** Any next candidate must come from a *new* sweep
   or from a direction outside the deployed-top-20 neighbourhood object.
4. **Carry §5.1's law forward** — it is the second, and stronger, instance of the recon's
   fragility-not-reachability law, and it is the paper-grade output of this pregate.

---

## §7. FILE MANIFEST

| artefact | role |
|---|---|
| `refine-logs/MEMBANK_C4_PREGATE_RECORD.md` | this record |
| `scripts/analysis/membank_c4_pregate.py` | the frozen harness (shas in §2.7) |
| `scripts/analysis/membank_c4_selftest_OUT.json` (+ `.log`) | §2.6 pre-freeze synthetic arms S-A/B/C/D |
| `scripts/analysis/membank_c4_{hatemm,zh,en}_OUT.json` (+ `.log`) | all arms, gates and controls, 3 spaces |
| `scripts/analysis/membank_c4_perm_{hatemm,zh,en}_OUT.json` (+ `.log`) | GATE 7, 200 draws each |

**Read-only inputs:** `scripts/analysis/mechfix_ops.py` and `scripts/analysis/mechnov_pairverify.py`
(both sha-asserted, imported unmodified), `scripts/analysis/mechnov_pairverify_{ds}_OUT.json`
(PARITY-ARENA anchors), `data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_<model>.pt`.
**Not opened:** any `dev_seen` or `test_seen` cache, any `vsw_*` artefact, any `data/gt/*` file.

---

## §8. LIMITATIONS

1. **Arena.** Banked **raw** encoder key space, **train** split — the F95 precedent inherited by all
   four sibling pregates (F47: head LOO train acc 0.998, so a train-side screen in head space measures
   memorisation). **A raw-space, train-side null does not logically entail a head-space or test null.**
   This is stated first because it cuts *against* the verdict.
   **8.1 — but the transfer is asymmetric, and the asymmetry favours the kill.** The head space's cone
   is ~250× more collapsed than the raw fused space (top-1 cosine 0.999852 vs 0.9444;
   `VSW_ASYMMETRY_RECON.md` §1.4). Class spans built from *more nearly identical* vectors are *more
   nearly identical*, so GATE 1 degeneracy can only get **worse** in head space, never better — the
   degeneracy read therefore transfers a fortiori. The *accuracy* result does not transfer, and this
   record does not claim it does.
2. **No seeds.** The raw features are seed-independent and MEMBANK-C4 is training-free, so the sign
   evidence is 5 folds, not 3 seeds. The full-version bar's 3-seed clause is therefore untested by
   construction — moot, since no cell approaches it.
3. **Train-side only.** No test number exists in this document, by design (LITSWEEP6 §1(e)).
4. **The declared operator space is finite.** `r ∈ {1,2,3,5,full}` × ridge `γ ∈ {1e-3,1e-2,1e-1,1}`,
   count- and rank-matched, in four geometries. LITSWEEP6 §4(d) fixed that grid and required it be
   declared before the run; a residual outside it is not measured. The rank- and γ-insensitivity
   documented in §6.2(g) is the evidence that widening the grid would not help.
5. **ERRATUM E-1 (§3.2)** is a real, disclosed departure from the freeze. It changed a gate's
   *severity*, not any number, arm, bar or operator line, and both shas are recorded.
6. **`NULL2_geom` uses `C4_pca_r1` only** (the first declared cell), 20 draws, seed 7 — it prices the
   geometry channel for one cell, not for all nine.
7. **The MHC-ZH / img hindsight ceiling of +16 net (§4.7) is a real number in a space that cannot
   carry a bar**, on a floor 14.68 points below the deployed one. It is recorded so it is not
   re-discovered as news; it is not offered as a result.
