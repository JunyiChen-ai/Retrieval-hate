# C09 Stage-0 (A0) — preregistration **v2** (round-1 repairs)

**Candidate.** C09 · Stable-Inversion Topology Surgery
**Registry claim.** *"OOF-stable high-confidence inversions identify topological
defects that can be corrected at encoder level while explicitly constraining break
exposure."*
**Registry dedup boundary.** *"Encoder-level topology intervention, not
thresholding, local reranking, verifier gating, NCA/SupCon, or hard-example
weighting alone."*
**Authorised by.** `TARGET_STATE.json::gate0_reopen_2026_07_31` —
`next_active_candidate_post_C04`.

> ## STATUS: `V2_REPAIRED_NOT_FROZEN_NOT_SUBMITTED` — awaiting fresh independent review.
>
> v1 (`refine-logs/C09_A0_PREREG_DRAFT.md`) was reviewed once and returned
> **`REVISE (4 Critical / 8 High / 10 Important)`**
> (`refine-logs/C09_A0_PREREG_DRAFT_REVIEW_ROUND1.md`). **All 22 findings are
> repaired here.** v1 is superseded in full and must not be implemented; it is
> retained only as the reviewed object. The repair ledger is §11.
>
> No hash is frozen, no config is written, no job is submitted and no namespace is
> created by this document.

---

## 0. What changed, in one paragraph

The v1 design could not decide its own question. Its identifiability probe was fed
the scored item's own gold label (`C-1`), its discriminator had no outcome for the
realistic case (`C-2`), its rows leaked across seeds (`C-3`) and its conversion
currency left selected items uncosted (`C-4`). v2 replaces the probe with a
**conditional, incremental** statistic — a within-configuration-stratum AUC of a
label-blind feature set, measured **against a configuration-only baseline on the
same folds** — makes the discriminator **three-valued** with the realistic outcome
written in as a KILL, partitions the verdict space so `KILL` and `CONTINUE` are
complements, costs every selected item at every seed, and names the Stage-1
successor it would license so the seam to F75 is visible before the run rather than
after it.

---

## 1. What A0 asks, and why the second question is the real one

`unified_pilot_gate.stage_0_reachability`, verbatim:

> *"Before teacher/GPU spend, the full-bank or representation-level oracle must reach
> at least +0.050 accuracy and +0.050 macro-F1 on at least two datasets, with enough
> net correct-minus-broken items for the +0.030 final bar."*

The campaign has repeatedly measured large oracles in this channel that failed to
convert. AGGNET/F98 held `+0.1492 / +0.1520 / +0.2186` with 96–100 % of every
deployed error inside its function class and delivered `+0.0134 / −0.0069 /
+0.0000`, with the epitaph *"What binds is neither reach nor capacity but that the
local configuration carries no learnable signal about which neighbours to trust at
n = 549–744."* The Gate-0 reopen recorded this as governing: **a large oracle is no
longer evidence for a candidate in this channel — it is the precondition every
failed candidate already met.**

So A0 measures three things, of which only the second and third can promote:

1. **Reach (`O1`)** — is the OOF-stable-inversion population large enough that
   fixing all of it would clear `+0.050 / +0.050`? Expected to pass; a fail kills at
   zero cost and closes every confidence threshold at once by arithmetic (§4.3).
2. **Conditional identifiability (`D-FELDMAN`)** — *within a matched local
   configuration*, is "this item is an OOF-stable inversion" predictable from
   geometry alone, with no access to the item's own label? This is the question the
   F98 epitaph poses, asked of C09's population.
3. **Conversion (`NET`)** — at the frozen operating points, does the population
   yield enough net correct-minus-broken items in the currency
   `banned_constraints[10]` mandates?

**A0 trains no encoder, touches no test split, and establishes that no operator
exists.** A CONTINUE means only that the target population is large enough and
locatable enough to be worth building an operator for — and even that is
conditional on §10's Stage-1 precondition.

---

## 2. Arena, instrument, cost

**Path.** The banked **fold-head / deployed-head arena** only. F113 stands: *a
raw-key arena may KILL but may not PROMOTE*, so a Stage-0 PASS is rendered on the
fold-head path.

**Instrument — verified present, nothing to build.**
`scripts/analysis/headspace_mint.py`, `headspace_arena.py`, `headspace_fidelity.py`,
`headspace_report.py`, plus the six banked
`headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`. **The mint is
`headspace_mint.py` invoked unmodified with its sha256 asserted; the only new code
in this A0 is the analysis script.**

**Fold contract (I-3, pinned).** `StratifiedKFold(n_splits=5, shuffle=True,
random_state=0)` over the train split, stratified on the train label — i.e.
`mechnov_pairverify.K_FOLDS = 5`, `FOLD_SEED = 0`. `headspace_mint.py:203-216`
asserts this assignment against the banked `scripts/analysis/vsw_ckpt/<ds>/f<fold>.npz`
`ho_idx` and refuses to run on mismatch. The assignment is a function of the label
vector alone and is therefore **identical across head seeds** — the property §5.2's
nested split relies on.

**Configuration.** 2 datasets × 3 head seeds × 5 item-disjoint folds = **30
fold-heads in total**, plus 2 × 3 = **6 deployed-configuration (`fold == -1`) heads**
for the real fidelity read = **36 heads total** (I-4: v1's "per sweep" was wrong).
Bank = the fitting pool; queries = the held-out fifth. Query labels are
**train-split** labels held out from the head that judges them — this is what "OOF"
means throughout.

**Datasets.** `HateMM` (n = 744 train items = pooled query count) and `MHC_zh`
(n = 579). Per-fold held-out counts, from the banked arena outputs:
`149/149/149/149/148` and `116/116/116/116/115`. **MHC-EN is OUT OF SCOPE for A0**:
its fold-head arena has never been minted and minting it would introduce a new
instrument requiring its own fidelity check. Two datasets is what the bar requires
— **and with EN out of scope the two-dataset requirement has zero slack: a failure
on either dataset is a failure of the conjunct** (I-9).

**Cost.** `0 GPU-hours`. Per-fold head checkpoints are not persisted
(`headspace_mint.py:274-281` monkeypatches `torch.save` to a no-op because
per-epoch dumps are ~34 MB and nothing downstream reads them), so heads are
re-minted at ~25–60 s CPU each. **Budget: 36 heads; a 60 s/head assumption gives
≈36 CPU-minutes, and the analysis adds minutes.** One CPU-only SLURM job, 8 CPU /
32 GB / no GPU, no `--time`, following C02's A0 (job `13847`: 8 CPU / 0 GPU / 32 G,
`00:29:49` wall). **Resume path (I-4):** `headspace_mint.py:192-194` skips any
`--out` that already exists and the driver is a plain sequential loop, so a
requeued or re-submitted job resumes at the first missing head. No `--time` is set,
so there is no wall-clock kill to resume from; the resume path exists for the
operator-visible failure modes only.

**F88's binding caveat is satisfied by construction.** F88 requires that *"a
CPU-trained arm must be paired against a CPU-TRAINED FLOOR, never against the
banked GPU floor."* Every arm and every floor here is minted inside the same CPU
fold-head arena, so no cross-hardware pairing occurs anywhere.

**Standing clauses adopted (I-3).**
`refine-logs/PREGATE_DETERMINISM_CLAUSE.md` **DET-1 … DET-4** and
`refine-logs/PREGATE_CALIBRATION_CLAUSE.md` **CAL-0 … CAL-5** are adopted by
citation and are binding on this run.

- **DET-1.** `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=8`
  exported before any Python process starts; `headspace_mint.det1_assert` hard-fails
  otherwise. **DET-2.** the full runtime block (env, `threadpoolctl.threadpool_info()`,
  library versions, `torch_num_threads`, node) is recorded in every output JSON.
  **DET-3.** parity is asserted at **Tier A / Tier B, 4 decimal places** — never
  beyond (§7.2 `GATE-FLOOR`). **DET-4.** the primary estimator is
  `sklearn.linear_model.LogisticRegression(solver="lbfgs")`, the convex arm DET-4
  prefers; the gradient-boosting arm is a declared capacity check that **no decision
  rule reads**, and carries no Tier-C band because no verdict rests on it.
- **CAL-0** is restated as written and scoped: *"The raw train-space arena is not
  established as predictive of deployed effects."* C09's decision arena is the
  **fold-head** arena, not the raw arena; the raw arena appears here only as F113's
  KILL-only corroborator (§8). **CAL-1** governs that corroborator: its decisive
  bars are within-arena relative comparisons, which is the property CAL-1 requires
  of a raw-arena negative. **CAL-2**'s hard half is honoured in the form the
  fold-head arena admits: the deployed replay must reproduce the banked pooled
  quantities at 4 dp or the harness is VOID (`GATE-FLOOR`); CAL-2's `FIXK`/Spearman
  provenance leg is a **raw-arena** check against F94's banked k-curve and is not
  run, because this arena is not that arena — the equivalent provenance check here
  is the six-cell 4-dp parity against the banked `headspace_arena_*_OUT.json`, which
  is strictly stronger. **CAL-4** labels every reported quantity closed-form or
  trained (§5.5). **CAL-5** is the one with predictive content and it runs
  **against** C09: the Stage-1 operator this A0 would license changes the map, i.e.
  **channel (a)/(d)**, which *"carries NO transfer warrant and must say so in its
  limitations."* It is said, in §10.

---

## 3. Label-use discipline — LEGAL, on written texts, with the counter-texts carried

Identifying OOF-stable inversions requires reading train labels out of fold. This is
**legal**, and the reopen resolved it on two written texts rather than by inference:

- `autoresearch/goal_mllm_plus3/state/progress.json:25` — the user's own
  oracle-ranked-queue ruling: *"Legal attack on selection-locked pools = trained
  selector/reshaper on train labels only (F66 binds only fixed-map phi0)."*
- `refine-logs/LITSWEEP3_DATA_CENTRIC.md:82` — an on-point in-repo adjudication of
  exactly this shape: *"those select **per test instance**; curation selects
  **train items once, globally, applied identically to every test query** — a
  symmetric operator, so law-III/F66's per-item ban does not apply to the
  mechanism (though Wall-A still caps the achievable magnitude)."* The parenthetical
  is restored per the reopen's R2 I-5, and the same section prices that mechanism at
  *"+3 any dataset: ~1-2%"* (`:95`) and *"at most +0.001-0.006"* (`:91`).

Every text on the other side (F47, F66, EUM precondition 2) bans
**per-test-instance** selection. **Four** boundaries flip C09 to illegal and are
written in as **HALT** conditions:

- **`H-L1`.** Any query-time consultation of the stability statistic or of the
  `D-FELDMAN` classifier. F47 fires directly, and its escape clause is closed: an
  OOF-stability statistic *is* "derivable from banked features/votes", so it is not
  the *"genuinely NEW information source"* the exception requires.
- **`H-L2`.** Any per-item exception that survives to inference as a per-item rule.
- **`H-L3`.** Any read of a dev **label** or any read of a test path, at any stage,
  by any code path. (Dev *features* are read by the six `fold == -1` fidelity heads,
  by design and by `headspace_mint.py`'s own contract; dev **accuracy** is read by
  `headspace_fidelity.py` from banked trainlogs. Both are declared, counted and
  reported by `GATE-LEDGER`; neither reaches any decision rule.)
- **`H-L4` (new, H-7).** Any use of `D-FELDMAN` — the classifier, its score, or any
  monotone function of either — as a **selector, gate, router, abstention rule or
  risk ordering over a deployed decision**. That object is banned by measurement
  twice over and this A0 does not build it; see the adjudication immediately below.

### 3.1 `D-FELDMAN` against F47's ban_scope — adjudicated, not cited (H-7)

F47's `ban_scope`, verbatim (`directions_tried.json`, F47 entry):

> *"Per-item cross-channel selection/routing over banked channels: CLOSED at all
> three supervision sources (unsupervised=K9 zeros; train-supervised=memorization-
> degenerate target, CLIP LOO 0.998; dev-supervised=negative at CV ceiling −0.046 <
> perm-null). Decision-level meta-features (vote margins, purity, sub-votes,
> confidence differential, transcript stats) carry NO per-item routing signal, GBM or
> linear. Do NOT re-propose per-item selectors over frozen channels regardless of
> feature family or nonlinearity unless the selector input is a genuinely NEW
> information source not derivable from banked features/votes."*

`D-FELDMAN`'s feature family is **literally the family that sentence names**, and
its input is **not** a new information source. The ban is engaged, and the honest
statement is:

- **What the ban closes and C09 accepts.** Using these features to **select, route
  or gate a deployed decision** is closed. `H-L4` writes that into this design as a
  HALT boundary. C09 does not propose it, at Stage-0 or Stage-1.
- **What the ban does not reach.** F47's measured object is a **per-item router
  over frozen channels**, and its target is *which channel to trust for this item*.
  `D-FELDMAN`'s target is a different random variable — *is this item a
  three-seed-stable inversion of the deployed vote* — and its output is never
  consulted at prediction time by anything. A probe that measures whether a region
  exists is not a selector over that region. This distinction is the whole of C09's
  claim to be outside the letter, and it is **narrow**: if a future proposal reads
  the probe's score at query time, F47 fires and `H-L1`/`H-L4` HALT the run.
- **The prior F47 sets, and which this design registers as its own (C-2 / H-7).**
  F47 measured a **null** on this feature family at all three supervision sources.
  So does F97 (K-VGA-3: F47-family features beat the trained relation profile as
  gating features and the whole family still failed), and so does F98's epitaph.
  **The registered prior for `D-FELDMAN` is therefore a null, not a coin flip.**
  §6's table is written from that prior.
- **The symmetric correction, which runs in C09's favour and is carried at its true
  weight.** F47's train-supervised leg is *"memorization-degenerate target, CLIP LOO
  0.998"*. F114 rules that exact premise a **CLIP** number, while the deployed Qwen
  heads sit at `0.9406 / 0.8915 / 0.8154`. The saturation objection therefore does
  not transfer to this arena — which is exactly F113's own reason for building it
  (*"That objection does NOT apply to a head trained on 4/5 of the train split and
  queried with the held-out fifth"*). This weakens F47's train-supervised leg **on
  this arena**; it does not touch F47's unsupervised or dev-supervised legs, and it
  does not weaken the decision-level-meta-features sentence, which was measured
  independently of the 0.998 premise.

### 3.2 The counter-text, carried at its adjudicated weight (I-10)

`LITSWEEP5_COMPLETENESS.md` §4(ii), headed *"The contradiction (load-bearing)"*, was
written **after** the oracle-queue ruling and observes that its two blessed classes
— *"Trained SELECTOR on train labels"* and *"Trained symmetric RESHAPER on train
labels"* — are *"both already measured dead"*, and that the ruling *"was written at
lit-round-count 3 — before F75/F77/L1 sharpened the walls."*

**The counter-text is itself DOWNGRADED, NOT VACATED** (reopen R7 I-2): §4(ii)'s
first blessed-class death reads *"Trained SELECTOR on train labels = F47's
train-supervised source. DEAD: the deployed kNN vote memorizes train (CLIP LOO
0.998)"*, and F114 rules that premise a CLIP number against deployed Qwen heads at
`0.9406 / 0.8915 / 0.8154`, leaving train-side headroom 30×–92× larger.
`LITSWEEP5_COMPLETENESS.md` is **not** among the nine records F114 corrected, so the
retraction never reached it. §4(ii)'s **independent** leg — the measured
train-disagreement counts `0/109`, `0/102`, `0/92` — is untouched and stands.

**Net: legality is not in question; viability is. C09 inherits a weakened prior, and
it is weakened by less than §4(ii)'s wording implies.**

### 3.3 `D-FELDMAN` is a probe, not a component

Stated plainly so it is not over-read: **`D-FELDMAN` is a Stage-0 *identifiability
probe*, never a deployable component.** It measures whether a region exists.
Whether a *global operator acting uniformly on that region* is legal and buildable
is a Stage-1 question with its own gate (§10); A0 makes no claim about it.

---

## 4. Population definition — every threshold frozen before any run

### 4.1 The deployed decision and the vote scale (H-3)

`score_i` is defined by **literal reference to `scripts/analysis/mechfix_ops.py:94`**:

```python
votes = ((lab * 2 - 1) * sim * w).sum(1) / w.sum()
```

with `w = _rank_weights(20) = [20, 19, …, 1]`, `Σw = 210`, `sim` the float32 faiss
inner products of L2-normalised keys, `lab` the **bank** labels of the top-20, and
the decision `predict 1 iff score ≥ 0` (`mechfix_ops.py:95`). **The vote is already
divided by `Σw`.** v1's `c_i = |score_i| / Σw` double-normalised, which is why v1's
declared ladder `{0.10, 0.25}` selected the empty set. It is deleted.

`conf_i ≡ |score_i|` on the scale `mechfix_ops.py:94` produces. For orientation, and
declared as a **transferred expectation, not a measurement of this arena** (I-1):
`ERRPAT_HateMM_2026-07-26.md:130` reports median `|vote|` **`0.7267`** for errors
against **`0.9873`** for always-correct items, on the **test split** under a
**CPU-reconstructed proxy** of the deployed head (`ERRPAT_HateMM §0.1`, 52 s/seed).

### 4.2 Inversions and stability

For dataset `D`, seed `s`, fold `f`, query item `i`: item `i` is an **inversion at
seed `s`** iff its deployed prediction disagrees with its gold train label. `i` is an
**OOF-stable inversion** iff it is an inversion in **all three** head seeds;
stability is computed per seed independently and the population is the
**intersection**.

**Provenance of the expectation that this population is large (I-1).** F88 measured
seed-invariance on the **test split** under the **deployed-head proxy**, not in this
arena: *"ZH 22 of the 25-item union wrong 3/3 (88 %) with NOTHING at exactly 2/3 and
ALL 12 false negatives 3/3-stable"*; HateMM *"24-25 of 26-28 errors wrong in 3/3
seeds (89-93 %)"*. Those figures are **transferred expectations** motivating the
design. The population this A0 prices is measured **in-run, in the fold-head train
arena**, and is reported with its own counts; nothing in the decision rule reads an
F88 number.

### 4.3 Confidence thresholds — a primary and a registered co-primary (H-3, H-4)

Two thresholds, both frozen here, both computed **in-run** from the OOF `|vote|`
distribution **among stable inversions**, per dataset:

- **`τ_0 = 0` — PRIMARY.** All OOF-stable inversions, no confidence restriction.
- **`τ_hi = median(|score_i| : i ∈ P_0)` — the REGISTERED "high-confidence"
  co-primary.** The registry claim says *"high-confidence"* inversions; `τ_0` is not
  that population, so a rule evaluated only at `τ_0` cannot KILL the registered
  claim. `τ_hi` is the co-primary that closes that gap.

**Where monotonicity holds, and only there (H-4).** `|P_τ|` is monotone
non-increasing in `τ` by construction, and `Δacc_{O1} = |P_τ| / n`. Therefore
**`K-REACH` firing at `τ_0` closes every `τ ≥ 0` by arithmetic** — that leg of v1's
argument is sound and is retained, scoped to `O1` only. **It is false for `NET` and
for `D-FELDMAN`:** raising `τ` redefines the target, and both AUC and precision can
rise on a purer subpopulation. So neither `K-FELDMAN` nor `K-NET` may be read
beyond the `τ` it was evaluated at, and both are evaluated at **both** `τ_0` and
`τ_hi` (§8).

**Pre-declared arithmetic consequence, so it is not discovered afterwards.**
`τ_hi` is the median, so `|P_{τ_hi}| ≈ |P_0| / 2`. If `|P_{τ_hi}| / n < 0.050`, the
co-primary fails `K-REACH` **by arithmetic** — which is a real result about the
registry claim's own "high-confidence" restriction, not an artefact. The run
therefore also reports **`q_max`**, the largest quantile of the stable-inversion
`|vote|` distribution at which reach still clears `+0.050`, as a frozen, decidable
descriptive quantity that no decision rule reads.

**Declared sensitivity, never a decision:** `τ ∈ {q25, q75}` of the same in-run
distribution, reported in full for mechanism reading only.

### 4.4 Frozen ancillary definitions

- **Right analogue** of `i`: the highest-ranked bank item carrying `i`'s gold label,
  with its rank `r_i` in `i`'s current ordering. **Reported as a mechanism
  diagnostic only; it reads `i`'s gold label and is therefore excluded from every
  feature set by `GATE-BLIND`.** (Motivating transferred measurement, test split,
  proxy head: `ERRPAT_MHC-ZH` reports median rank `1.5` for the 22 ZH core errors,
  11 of 22 at rank 1; `ERRPAT_HateMM:130` reports median rank `3.0` with `6/27`
  errors having no true-label neighbour in the top-20 at all.)
- **`pred_purity_i`**: fraction of `i`'s top-20 whose **bank** label equals `i`'s own
  **predicted** class. Label-blind for the scored item. **This replaces v1's
  gold-purity, which was the C-1 leak.**
- **Configuration stratum (frozen, and now label-blind — I-1, C-2).** The cross of
  - `|score_i|` **tercile**, computed in-run over all `n` query items of the
    (dataset, seed) cell, and
  - `pred_purity_i` bucket `{[0, 0.60), [0.60, 0.80), [0.80, 0.95), [0.95, 1.0]}`.

  12 strata. **Both axes are computed in-run from this arena and are label-blind for
  the scored item**, so v1's defect of freezing buckets derived from a test-split
  gold-purity table is removed rather than merely declared. The bucket edges are
  frozen here and are not tuned.

---

## 5. The three measured quantities

### 5.1 `O1` — reach (necessary; an upper bound, and declared as one)

For each seed `s`, flip the prediction of every item in `P_τ` and recompute accuracy
and macro-F1 against the deployed floor of that (dataset, seed) cell.
`Δacc_{O1} = |P_τ| / n` identically for every seed, because every member of `P_τ` is
wrong at every seed; `ΔmF1` is recomputed from the realised confusion matrix, not
assumed. Primary = mean over the three seeds; per-seed values reported.

**Scope, declared now (I-9).** `O1` is a **label-flip oracle over one nominated
population**, not the *"full-bank or representation-level oracle"*
`stage_0_reachability` names. It is the tightest zero-cost **upper bound** on what
any operator confined to fixing stable inversions could reach, and it is used in
that direction only: a fail is a closure, a pass establishes nothing beyond
"population large enough".

### 5.2 `D-FELDMAN` — conditional, incremental identifiability (C-1, C-2, C-3)

**Question.** *Within a matched local configuration*, can "is this item an OOF-stable
inversion?" be predicted from geometry alone, with no access to the item's own label
at prediction time — **over and above what the configuration itself already says**?

**Why conditional and incremental (C-2).** H-MEMORISATION does **not** predict
unconditional AUC ≈ 0.5. Feldman's long-tail singletons *are* the low-density,
weak-margin, no-analogue items, so a label-blind feature set separates them from
correct items under **either** hypothesis — the separation is already banked
(`ERRPAT_HateMM:130`, median `|vote|` `0.7267` vs `0.9873`). An unconditional AUC
near 0.9 would therefore be uninformative, and v1's `K-FELDMAN` could essentially
never fire. v2 conditions on the configuration stratum and measures the **increment
over a configuration-only baseline**, which is the quantity the two hypotheses
actually disagree about.

**Estimator and split.**
- Rows are `(item, seed)`; the target is **per item** and constant across that
  item's three rows.
- **The nested-CV partition *is* the frozen 5-fold arena partition** (§2). An item's
  score is produced by a model fit on items from the other four arena folds only.
  This simultaneously (a) groups all three seed-rows of an item together, killing
  C-3's cross-seed leak, (b) makes the scored item disjoint from its own arena fold,
  which is what `GATE-NESTED` asserts, and (c) introduces **no new
  hyperparameter and no RNG**.
- **Primary estimator (DET-4):** `LogisticRegression(penalty="l2", C=1.0,
  solver="lbfgs", max_iter=2000, class_weight="balanced", tol=1e-6)` on
  z-scored features, standardisation fit on the training folds only.
  **Capacity check, no decision rule reads it (H-6):**
  `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=3,
  l2_regularization=1.0, random_state=20260801)`. Both parameterisations frozen here
  (I-8); this mirrors F47's own two-family protocol, which found *"NO per-item
  routing signal, GBM or linear"*.

**Two frozen feature sets.**

*`BASE` — configuration-only (7 features).* `|score_i|`; `pred_purity_i`; mean and
standard deviation of the top-20 similarities; the similarity gap between ranks 1
and 20; local bank density (mean similarity to the 50 nearest bank items); the
item's own L2 norm before normalisation.

*`FULL` — `BASE` + structural block (6 further features).* rank of the first
neighbour whose bank label differs from the top-20 majority bank label; the count of
runs (label changes) in the rank-ordered bank-label tuple of the top-20; mean and
standard deviation of the **bank-side degree** of the top-20 members (how often each
appears in other query items' top-20 within the same fold); the signed gap between
the best rank-1 similarity to a class-1 bank item and to a class-0 bank item; the
number of the item's top-20 members that are themselves stable inversions **in the
fitting folds only** — computed from training-fold items exclusively and therefore
disjoint from the scored item's own target.

**Every feature reads only: the query item's own key, the bank keys, and the bank
labels. No feature reads the query-side gold-label array.** This is enforced
structurally and counted (`GATE-BLIND`, §7.2): the feature builder does not receive
the query-label array at all, and the array is wrapped in a read-counting guard whose
count must be exactly `0` across the whole feature-construction phase.

*Declared feature-degeneracy read (I-6).* `ERRPAT_HateMM:130-136` measures cosine
saturated at ~`0.9999` for both errors and correct items — *"Distance-based
abstention/gating has essentially no dynamic range to work with."* Several `BASE`
features are similarity-derived and may therefore be near-constant in this space.
This is **declared in advance**, and the run emits a `FEATURE_DEGENERACY` block —
per-feature standard deviation and distinct-value count, per (dataset, seed) — so a
reader can see which features carried information. It gates nothing.

**Primary statistic.**

> **`ΔAUC = AUC_strat(FULL) − AUC_strat(BASE)`**, where `AUC_strat` is the
> **stratum-conditional AUC**: for each of the 12 configuration strata, the
> Mann-Whitney probability that a random positive outscores a random negative
> **drawn from the same stratum**, pooled across strata weighted by that stratum's
> positive×negative pair count. Both terms are computed on **identical** OOF folds
> and identical rows, so `ΔAUC` is a genuinely **paired** quantity — v1's "paired
> bootstrap on a single AUC" (H-6) is repaired by making the primary a difference.

**Negative class.** `CONFIG-MATCHED-CORRECT`: query items correct at all three
seeds. Matching is achieved **by the stratification itself** — no sampling, no RNG,
no discarded data — which is why `AUC_strat` and not a matched-subsample AUC is the
instrument. Unstable errors (wrong in 1 or 2 seeds) are **excluded from the
`D-FELDMAN` positive and negative classes** and are the subject of `UNSTABLE-POP`
instead; they are still fully costed in `NET` (§5.3).

**Inference (H-6, I-8).** One-sided **item-level** bootstrap: resample **items**
(not `(item, seed)` rows) with replacement, `B = 10000`, `α = 0.05`, RNG
`numpy.random.default_rng(20260801)`, recomputing `AUC_strat(FULL)` and
`AUC_strat(BASE)` on each resample and taking the difference. Reported: point
estimate, one-sided 95 % lower bound, two-sided 95 % interval. Resampling items
rather than rows is C-3's second repair: a row-level bootstrap would be
anti-conservatively narrow by roughly `√3`.

**Multiplicity.** `K-FELDMAN` is the only rule in this design that reads an
inferential quantity. Its family is exactly **2 τ × 2 datasets = 4** hypotheses and
**Holm** is applied across those four at `α = 0.05`. No other family is corrected,
because no other rule performs a test (§8).

**On the "conversion-equivalent AUC" the round-1 review asked for.** It is not
well-defined: AUC does not determine precision at a fixed selected count without the
score distribution, so no AUC threshold is equivalent to a net-item bar. The
conversion leg is therefore adjudicated **where it is exactly decidable — in
precision and net-item space** (`K-NET`), and the run reports, at every operating
point, the **conversion-equivalent precision** `π* = (1 + bar/k)/2` alongside the
realised precision, which is the honest version of the same instrument.

### 5.3 `NET` — conversion, fully costed (C-4)

**Accounting.** At an operating point the classifier selects a set `S` of `k` items
from **all `n` query items**. For each seed `s`:

```
net_s = |{ i ∈ S : deployed prediction wrong at seed s }|
      − |{ i ∈ S : deployed prediction right at seed s }|
      = 2·|{ i ∈ S : wrong at s }| − k
```

Every selected item is costed at every seed — nothing sits in "neither class", which
was C-4's leak. `CONFIG-MATCHED-CORRECT` is retained as **reporting stratification
only**, never as the denominator of the promoting quantity. **Self-test, asserted in
code and HALTing on failure:** `net_s == n · Δacc_s` exactly, for every seed, every
dataset, every operating point.

Primary `net` = mean over the three seeds; the per-seed minimum is also reported.
Exchange rate is reported as a diagnostic and **reads no decision rule**
(`banned_constraints[10]`).

**Item score.** For each item, the mean of its three seed-rows' OOF predicted
probabilities from the `FULL` logistic model. Deterministic; no RNG.

**Frozen operating points (H-6 / review answer 4).** Three points on the
**selected-count** scale: `k ∈ {|P_τ|, round(1.5·|P_τ|), round(2·|P_τ|)}`, top-`k` by
item score over all `n`. v1's six score-percentile points are deleted: four of six
could not clear the bar by arithmetic, and the 95th percentile could at best tie it
at precision `1.000`.

**The currency, adjudicated rather than silently chosen (H-5).** Three surfaces name
a net-item figure and they do not agree. The adjudication:

1. `unified_pilot_gate.stage_0_reachability` — the **governing gate text** — ties
   the net requirement to the **`+0.030` final bar**: *"with enough net
   correct-minus-broken items for the +0.030 final bar."* `banned_constraints[10]`
   supplies that figure: **`22.3` (HateMM) / `17.4` (MHC-ZH)**, which are exactly
   `0.030 × 744` and `0.030 × 579` on the **train arena** at `n = 744 / 579` (the
   R13 instruction to state the arena whenever these figures are quoted).
2. The reopen's C09 `bar` field names **`37.2 / 29.0`** — `0.050 × 744` and
   `0.050 × 579` — as the `+0.050`-equivalent scaling.
3. C02's own A0 ran `net_fix_rate: 0.03`, consistent with (1).

**Binding rule: (1).** The governing gate text is explicit that the *oracle reach*
bar is `+0.050 / +0.050` while the *net* requirement is sized to the `+0.030` final
bar, and this is not a softening — for `O1`, which breaks nothing, `net ≡ n · Δacc`,
so a `+0.050` net screen on `O1` would be the accuracy screen restated, carrying no
independent information. The net screen only bites where breaks are real, i.e. on
the realistic operator, and there the gate text sets it at `+0.030`. **`37.2 / 29.0`
is computed and reported at every operating point as a declared secondary that
*scopes* a CONTINUE (§8) and can never create or block one.**

**The macro-F1 leg (H-6).** The Stage-0 bar has a macro-F1 leg that a net-item count
does not price. At every operating point the run also computes `ΔmF1_s` from the
realised post-flip confusion matrix, and `K-NET` requires
`mean_s ΔmF1_s ≥ +0.030` at the same cell — the same `+0.030` final bar the net-item
figure encodes.

### 5.4 What is *not* an inferential quantity, and why there is no Holm family over `NET`

`stage_0_reachability` is written as a **threshold** rule ("must reach at least
+0.050 … with enough net items"), not as a CI rule; the CI requirement first appears
at `stage_1_signal`. `O1` and `NET` are therefore adjudicated on **point estimates
against frozen thresholds**, and no test is performed, so there is no multiplicity to
correct. Forking-path control is structural instead, and is threefold: the
`2 τ × 3 k` grid is frozen, exhaustive and reported in full; a CONTINUE requires the
**same `(τ, k)` cell to clear on both datasets simultaneously**; and the CONTINUE
names its cell, with §9 scoping the verdict to that cell. One-sided item-level
bootstrap lower bounds (`B = 10000`, `α = 0.05`, same RNG) are computed for every
decision quantity and reported, and are used **only** to tag a CONTINUE
`ROBUST` / `POINT_ESTIMATE_ONLY`; they never create or block a verdict.

### 5.5 CAL-4 declaration

**Closed-form** (reproduces bit-exactly across sessions): every deployed-vote
quantity, every count, `O1`, `net`, `Δacc`, `ΔmF1`, the strata, and every feature.
**Trained**: the head mint itself (30-epoch Adam, DET-3 Tier B — 4-dp parity against
the banked arena is asserted by `GATE-FLOOR`), the logistic estimator (measured
invariant at 4 dp across the whole thread grid, `PREGATE_DETERMINISM_CLAUSE §1.3`)
and the gradient-boosting capacity check (no verdict reads it).

---

## 6. The Feldman discriminator, three-valued and decidable (C-2)

The **numerical** leg of the Feldman objection is already retracted in-repo —
`HEADCOV_PREGATE_RECORD.md:305-310` withdraws *"the Feldman flourish"* because the
deployed heads sit at 0.82–0.94, not 0.998 — while its **substantive** leg is
preserved verbatim there and stands: *"memorising a long-tail singleton does not
transfer to an unseen member of the same one-member sub-population."*

| | **H-TOPOLOGY** (C09's claim) | **H-MEMORISATION** (Feldman) |
|---|---|---|
| what the stable inversions are | a **region** with a shared geometric signature beyond atypicality | **singletons**: each wrong for its own reason, sharing only that no analogue was memorised |
| unconditional AUC | high | **also high** — this is why v1's instrument could not decide |
| **`ΔAUC` (conditional, incremental)** | **> 0**: structure survives conditioning on the configuration | **≈ 0**: the configuration is all there is |
| `NET` at the frozen points | clears `22.3 / 17.4` | ≈ 0 or negative |
| what an operator could do | act uniformly on the region | nothing — the fix requires memorising items never seen |

**The registered prior is a null** (§3.1): F47 measured this feature family carrying
no per-item signal at all three supervision sources; F97's K-VGA-3 measured the same
family beating a trained relation profile and the whole family still failing; F98's
epitaph states it in general terms. **And the closest measured analogue in this very
arena runs the same way:** F113 measured that *"any FITTED relation score over head
keys memorises the bank (in-sample pair AUC 0.9999) and is WORSE than the plain
cosine on held-out pairs (d_AUC +0.1572/+0.2302 raw → −0.0643/−0.1294 head, 30/30
fold cells)"*. That is a **pair**-level score rather than an item-level target, so it
is not a closure of `D-FELDMAN` — but it is the nearest thing to one that exists, it
is in the identical key space, and it is registered here as the prior rather than
discovered afterwards.

**The discriminating observation, pre-declared and three-valued:**

> **Band A — `ΔAUC` Holm-corrected one-sided lower bound `≤ 0`.** No incremental
> structure survives conditioning. **KILL.** When band A holds on **both** datasets,
> the additional statement *"H-MEMORISATION is consistent with this object"* is
> published; when it holds on one dataset only, the KILL stands (§8) but no
> Feldman claim is made.
>
> **Band B — `ΔAUC` lower bound `> 0` but `NET` under bar at every frozen cell.**
> **KILL, under the F98 epitaph** — reach and even conditional identifiability
> without conversion, which is the pattern AGGNET/F98 already measured at the
> largest oracle in campaign history. **This is explicitly NOT a confirmation of
> H-MEMORISATION**; it is a conversion failure, and the record must say so. v1 had
> no row for this outcome, which is the outcome the campaign's own base rate makes
> most likely.
>
> **Band C — `ΔAUC` lower bound `> 0` and `NET` clears at a common cell on both
> datasets.** The defect is a locatable topology defect rather than a
> memorisation-necessary error, and C09 earns Stage-1 — where the separate and
> harder question is whether a *legal global operator* realising it exists (§10).

**Upper-bound caveat, stated in advance.** `AUC_strat` conditions on a stratum
built from the item's **own** `|score|` and `pred_purity`. Those are label-blind, so
this is not a gold-conditioned upper bound as the round-1 review anticipated — but
it *is* an upper bound in a different and important sense: the stratum is computed
from the same OOF arena the target is defined in, and a deployed operator would have
to locate the region without knowing which items are in `P`. `ΔAUC > 0` is therefore
necessary, not sufficient, for a buildable operator.

### 6.1 Controls attached to the discriminator

- **`SHUFFLE-POP` (repaired, H-1).** The permutation is pinned: a **uniform random
  permutation of the per-item target vector over the query items of one dataset**,
  `numpy.random.default_rng(20260801)`, `200` draws, applied identically to both the
  `FULL` and `BASE` pipelines on the same folds. v1's claim that it *"preserves all
  configuration marginals"* was false for a plain permutation and is withdrawn.
  **What it tests:** that the split machinery, the stratified-AUC estimator and the
  bootstrap do not manufacture signal from the target's *marginal* alone.
  **What it cannot test, stated plainly:** it is **blind to feature-side leakage** —
  permuting the target destroys the feature-target association, so an estimator
  leaking the scored item's label through a feature would pass it cleanly. That job
  belongs to `GATE-BLIND`, which is a structural gate, not a statistical one.
  **HALT rule:** the permutation-null mean of `AUC_strat(FULL)` must lie in
  `[0.45, 0.55]`; outside that band the estimator is leaking and no verdict is
  published.
- **`UNSTABLE-POP`.** `D-FELDMAN` re-run with the target redefined as *unstable*
  errors (wrong in exactly 1 or 2 of 3 seeds), negatives unchanged. If stable and
  unstable populations are equally predictable, "stability" carries no information
  and the registry claim's own premise is empty — reported as a mechanism finding
  regardless of the primary verdict. **Data-independent power rule, applied
  identically to both datasets** (review answer 2, so that no test-split measurement
  becomes a frozen scope decision): emit `CONTROL_UNDERPOWERED` iff
  `n_unstable < 20` **or** the two-sided bootstrap CI width on `ΔAUC` exceeds
  `0.30`. Non-gating.
- **`RANDOM-POP`.** A size-matched random sample of query items in place of the
  stable inversions, `default_rng(20260801)`, 200 draws; every reported quantity
  recomputed against it. This prices the headwind the closest prior attempt failed:
  **F88 null (3)** measured HateMM memory-bank LOO curation at `+0.0016` against
  random deletion of the same size at `+0.0031 / +0.0000`, self-labelled
  *"Pregate-grade null (one rule, one proxy head/cell, single draw)"*. **Carried at
  its adjudicated weight (I-10):** the reopen's round 14 records that this is *"a
  val-sel loss and a final-epoch win, all under half a test item per seed, so
  'indistinguishable' is the exact reading"* — the curated rule is **not**
  established as *worse* than random, it is established as **indistinguishable from**
  random on a single draw. It is HateMM-only, on **train-row deletion**, a different
  population and operator from C09's. A headwind to price, not a closure; this
  control prices it. Non-gating.

---

## 7. Controls and validity gates

### 7.1 Scientific controls

`RANDOM-POP`, `CONFIG-MATCHED-CORRECT`, `SHUFFLE-POP`, `UNSTABLE-POP` — §5.2, §6.1.
`CONFIG-MATCHED-CORRECT` additionally supplies the break-exposure stratification, so
that *"constraining break exposure"* is measured rather than asserted.

### 7.2 Validity gates — HALT-only; a failure publishes **no** verdict

- **`GATE-FLOOR`** *(renamed from v1's `GATE-FID`, which collides with the name
  `headspace_fidelity.py` already owns — H-8b)*. The re-minted fold-head arena must
  reproduce the banked per-seed pooled deployed values **at 4 decimal places**, on
  6/6 seeds, in **both** metrics (I-2):

  | | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | HateMM acc | `0.8884` | `0.8858` | `0.8858` |
  | HateMM macro-F1 | `0.8838` | `0.8811` | `0.8812` |
  | MHC-ZH acc | `0.8929` | `0.8895` | `0.8946` |
  | MHC-ZH macro-F1 | `0.8747` | `0.8710` | `0.8765` |

  Source: `scripts/analysis/headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`,
  `result.acc_deployed` / `result.mF1_deployed`, re-read at drafting time.
  **4 dp and not beyond** (review answer 6a): the banked anchors are 4-dp values and
  asserting past them is the engineering-HALT trap that killed three C01 runs.
  DET-3 Tier B entitles this run to gate against banked JSON because DET-1/DET-2
  are honoured and the banked outputs carry their own runtime block.
  **Version/node residual risk (review answer 6b):** the banked arena's
  `meta.runtime` is compared against this run's and any difference in the
  interpreter/numpy/scipy/sklearn/torch quartet or the node is **reported as a
  documented `RUNTIME_DRIFT` flag with the re-run path named**, not silently
  absorbed; a `GATE-FLOOR` failure under `RUNTIME_DRIFT` is an engineering HALT with
  a diagnose-repair-resubmit path, not a scientific result.

- **`GATE-DEVFID`** *(the real one — H-8b)*. `headspace_fidelity.py` is run
  unmodified on the six `fold == -1` heads, which are already inside the 36-head
  budget. Banked reference: `B_fid_abs_3seedmean` `0.0093` (HateMM) / `0.0086`
  (MHC-ZH), `STOP_RULE_TRIGGERED: false` on both
  (`scripts/analysis/headspace_fidelity{,_zh}_OUT.json`, re-read at drafting time).
  **Reported, and NOT a HALT.** Scope, stated honestly: this gate reads the
  CPU-minted proxy against the **banked GPU floor's dev curve**, i.e. it measures
  proxy↔floor fidelity across hardware. C09's entire arena — every arm and every
  floor — is CPU-minted, so F88's binding caveat is satisfied by construction and
  cross-hardware fidelity does not gate the internal comparison. A
  `STOP_RULE_TRIGGERED == true` on either dataset publishes the verdict with a
  `PROXY_FIDELITY_FLAG` and a scope note.

- **`GATE-BLIND`** *(new — C-1)*. A per-feature manifest naming, for every feature in
  `BASE` and `FULL`, the exact arrays it indexes. Enforced structurally: the feature
  builder's signature does not admit the query-side gold-label array, and that array
  is wrapped in a read-counting guard for the whole feature-construction phase.
  **Emitted as integer counts, not booleans**: `query_label_reads_during_features`
  must be exactly `0`; `bank_label_reads` is reported as its (nonzero, legal) integer;
  per-feature array-touch lists are emitted in full. Any nonzero query-label read
  **HALTs**.

- **`GATE-LEDGER`**. A runtime access ledger reporting, as literal integer counts:
  test-split path opens (must be `0`), test-label materialisations (must be `0`),
  dev-split **path** opens (expected nonzero — the six fidelity heads and the
  trainlog reader — reported with its declared expected value), dev **label**
  materialisations into any decision quantity (must be `0`). `headspace_mint.py:106-116`
  installs a global `torch.load` guard that raises on any path containing
  `test_seen` or `/test`; the driver adds an `open()`-level guard with the same
  predicate over the whole job.

- **`GATE-SEED`**. Stability is the 3-seed intersection; the per-seed inversion sets
  are emitted **in full**, as sorted item-index lists, so the intersection is
  independently recomputable from the published artifact without re-running anything.

- **`GATE-NULL`** *(repaired — I-5)*. HateMM train row `355` (`hate_video_95`,
  label `1`) carries an exact-zero vector in **both** streams; MHC-ZH has **no**
  structural-zero row. *(Re-measured this session directly from
  `data/CLIP_Embedding/{HateMM,MHC_zh}/train_*.pt`: HateMM zero-img rows `[355]`,
  zero-txt rows `[355]`; MHC-ZH `[]` and `[]`.)* v1's contract was internally
  contradictory — with-null and remove-null cannot agree on *every metric*, because
  `n` moves from 744 to 743 and every rate changes. The repaired contract:
  1. the **primary** run is **with-null**, on the full `n = 744`, which is the arena
     the banked floors and the `22.3` figure are defined on;
  2. a **remove-null sensitivity** is computed by dropping item 355 from the query
     set **and** from every bank, with its own recomputed floors and its own
     recomputed `22.3 → 0.030 × 743 = 22.29` bar;
  3. the requirement is that the two routes agree on the **verdict and on every
     KILL-rule outcome**, not on the metric values. A disagreement on one item out of
     744 is published as a first-class finding and the verdict is scoped to it.
  4. **In head space the zero row is not zero.** The head applies learned
     projections with biases to the zero input, so the C01/C02 raw-space contract
     *"must remain exact-zero in every derived array"* **does not transfer** and is
     not asserted here. What is asserted is that item 355 is treated identically to
     every other item by every code path, and its per-item fate (bank membership,
     top-20 membership count, inversion status per seed) is reported explicitly.

- **`GATE-ARENA`**. Pooled native accuracy must sit strictly between the majority
  rate and saturation on both datasets: HateMM `0.8858–0.8884` against a majority
  rate of `0.5995` (`posrate_bank = 0.4005`); MHC-ZH `0.8895–0.8946` against
  `0.6891` (`posrate_bank = 0.3109`).

- **`GATE-NESTED`**. The `D-FELDMAN` partition is asserted equal to the frozen arena
  fold partition, and for every scored item the assertion "this item's arena fold was
  excluded from the model that scored it, and all three of its seed-rows were
  excluded together" is checked and emitted as a **per-item check count** that must
  equal the item count.

- **`GATE-SELFTEST`** *(new — C-4)*. `net_s == n · Δacc_s` asserted exactly for every
  seed × dataset × operating point × `τ`. A mismatch HALTs.

- **`GATE-PARITY-λ0`**. The re-minted deployed vote must reproduce the banked
  per-fold `fold_acc_deployed` arrays bit-for-bit at 4 dp for all 30 fold-cells, not
  only the pooled figure — the finer-grained form of `GATE-FLOOR`, free from the same
  emitter.

---

## 8. Decision rule — frozen, two-valued, exhaustive (H-2)

Let `τ ∈ {τ_0, τ_hi}` and `k ∈ {|P_τ|, round(1.5|P_τ|), round(2|P_τ|)}`.

**`CONTINUE`** iff there exists a `τ` such that **all** of the following hold:

1. **`K-REACH` clears at `τ_0`** — `Δacc_{O1} ≥ +0.050` **and** `ΔmF1_{O1} ≥ +0.050`
   on **both** datasets. *(Evaluated at `τ_0` only; monotone in `τ`, §4.3.)*
   Additionally, if `τ = τ_hi`, `K-REACH` must also clear at `τ_hi` on both datasets.
2. **`K-FELDMAN` clears at that `τ`** — the Holm-corrected one-sided 95 % lower bound
   on `ΔAUC` is `> 0` on **both** datasets.
3. **`K-NET` clears at that `τ`** — there exists a **single** `k` such that, on
   **both** datasets simultaneously, `mean_s net_s ≥ 22.3` (HateMM) / `≥ 17.4`
   (MHC-ZH) **and** `mean_s ΔmF1_s ≥ +0.030`.
4. `SHUFFLE-POP`'s permutation-null mean `AUC_strat(FULL) ∈ [0.45, 0.55]` on both
   datasets, and **all ten validity gates pass**.

**`KILL`** in every other case. `KILL` and `CONTINUE` are complements by
construction, so v1's undefined outcome (clear on one dataset, fail on the other)
cannot arise (H-2).

**Which rule fired is recorded, and the KILL is scoped by it:**

- `K-REACH` fired at `τ_0` ⇒ the KILL closes **every** confidence threshold `τ ≥ 0`
  by arithmetic.
- `K-FELDMAN` or `K-NET` fired ⇒ the KILL closes **`τ ∈ {τ_0, τ_hi}` only** — the
  primary population and the registered high-confidence co-primary. It does **not**
  close arbitrary `τ`, because neither AUC nor precision is monotone in `τ` (H-4).

**A `CONTINUE` is tagged with, and scoped to, four things:** the `τ` and `k` it
cleared at; `ROBUST` or `POINT_ESTIMATE_ONLY` from the bootstrap lower bounds
(§5.4); `NET_050_CLEARED` or `NET_050_MISSED` against the reopen's secondary
`37.2 / 29.0`; and `PROXY_FIDELITY_FLAG` if `GATE-DEVFID`'s stop rule fired.

**The raw arena (I-7, specified rather than asserted).** The identical battery is
recomputed on the banked **raw fused key space** — `X = l2n(concat(l2n(img_feats),
l2n(text_feats)))`, 7168-d, seed-free, over the *same* frozen 5-fold assignment and
the same deployed top-20 vote — whose banked pooled deployed accuracies are `0.8441`
(HateMM) and `0.8480` (MHC-ZH) (`headspace_arena_*_OUT.json`,
`membership.raw_deployed_acc`). Because the raw space is seed-free there is exactly
one "seed", so **stability is undefined there**; the raw leg therefore prices the
*single-pass inversion* population, is reported as such, and is **confined to
corroborating a KILL**, which is the only direction F113 permits. Even that carries
F113's own caveat, recorded in the reopen: *"NOT established: that a raw-space
NEGATIVE cannot be a head-space positive."* **No raw-arena number reaches the
decision.**

---

## 9. Scope of any verdict this A0 can produce

- A **KILL** closes the C09 Stage-0 oracle **under the frozen Stage-0 rule, at the
  `τ` values §8 scopes it to**. It is **not** an impossibility proof for
  encoder-level topology intervention: the identifiability probe is one feature set,
  one estimator family and one stratification, and a richer geometry might locate the
  region where this one cannot. This boundary is stated **now**, in advance, because
  C02's A0 had to retract exactly this kind of overclaim once (the v8 erratum) before
  it was re-stated correctly.
- A **CONTINUE** establishes only that the population is large enough and locatable
  enough to justify building an operator. It establishes **nothing** about whether a
  legal global operator exists — Stage-1's question, gated by §10 — and it is scoped
  to its `(τ, k)` cell.
- **`O1` is a label-flip oracle over one nominated population, not the registry's
  "full-bank or representation-level oracle"** (I-9). It is an upper bound and is
  used only in the closing direction.
- **With MHC-EN out of scope the two-dataset requirement has zero slack** (I-9): a
  failure on either dataset fails the conjunct, and there is no third dataset to
  substitute.
- **CAL-5 runs against C09** (§2): the Stage-1 operator this A0 would license is a
  channel-(a)/(d) object — it changes the map — and *"a channel-(a)/(d) arena result
  carries NO transfer warrant."* Any Stage-0 result here is an **arena** result about
  the fold-head train arena, never a prediction of deployed behaviour.
- Neither verdict touches the `+0.030 / +0.030` two-dataset target, which remains
  active and unmet.

---

## 10. The Stage-1 seam — named now, because a CONTINUE with no legal successor is worthless (H-8a)

The reopen's **first** quoted kill-risk is: *"(i) any encoder-level pull of an
inversion toward its right analogue is a label-using metric move ⇒ F75/NCA."* v1
addressed it nowhere. It is a **Stage-0** problem, because `H-L1` forecloses any
query-time locator, so the only legal Stage-1 realisation is a train-time move — and
that is F75's neighbourhood.

**The successor this A0 would license, named concretely.** A **global, symmetric,
train-label-supervised reshaping of the head map** `φ₀ → φ′`, in which the
stable-inversion set is identified **once, offline, on train items only**, and enters
the *training objective* as a region-targeted term. At inference `φ′` is applied
identically to every query; the stability statistic, the probe and the region
membership are never consulted at query time. That is what makes it symmetric under
`LITSWEEP3_DATA_CENTRIC.md:82`, and it is the only shape §3's HALT boundaries leave
standing.

**Is that F75's object? Partly, and the honest accounting is:**

- **F75's `ban_scope`, verbatim:** *"head-loss swaps of the triplet+BCE hybrid toward
  vote-consistent (NCA/soft-kNN), contrastive (SupCon), or mixup-BCE objectives at 7B
  frozen-encoder feature scale; tau/alpha retunes = tactics, banned."* A
  region-targeted term added to the deployed triplet+BCE hybrid is **none of the
  three named objectives**, so it is outside the ban's letter.
- **But F75 is also *"the first measured negative for
  trained-reshaping-unlocks-oracle-headroom"*, and `LITSWEEP5_COMPLETENESS.md §2`
  argues in its own adversarial self-rebuttal that F75's *mechanism* — symmetric
  reshaping does not convert selection-locked headroom — *"generalizes past its
  named-loss letter"*.** That argument is an **argument**, not a measurement, and
  LITSWEEP5 offers it against its own challenge; but it is the correct headwind and
  it is registered here.
- **C09's own dedup boundary independently forbids the cheap version:** *"not …
  hard-example weighting alone."* A region-weighted triplet+BCE **is** hard-example
  weighting, so the successor must be more than that or it fails its own registry
  boundary before it fails F75.
- **The counterweight, at its corrected one-sided-use weight.**
  `NCA_FORENSIC_RECON.md:110` verbatim: *"Ruling: F66 does NOT bind trained-space
  reshaping. The cell is not F66-dead — it is legitimately un-measured."* The same
  record's `:112` prices that cell at *"honest P(≥+3) stays 2-4%"*, and the cell it
  unblocked was subsequently run and killed as F75. Both halves are carried.

**Pre-registered consequence, binding on this A0's own verdict.** A `CONTINUE`
**does not carry a Stage-1 licence.** Stage-1 entry requires, as a precondition
written here rather than negotiated later, that a proponent name an operator that is
(a) global and symmetric at inference, (b) not one of F75's three named objectives,
(c) not hard-example weighting alone, and (d) accompanied by a fresh ban-scope
adjudication against F75, F66 and F98. **If no such operator can be named at Stage-1
entry, the CONTINUE is void and C09 closes with no further spend.** This is
pre-declared so that a CONTINUE cannot be converted into GPU spend by an argument
constructed after seeing the numbers.

---

## 11. Repair ledger — all 22 round-1 findings

| # | finding | repair | where |
|---|---|---|---|
| **C-1** | gold-purity leaks the scored item's own label | gold-purity **deleted**; replaced by `pred_purity` (predicted class); right-analogue rank demoted to a diagnostic and excluded from every feature set; **`GATE-BLIND`** added as a structural read-counting gate emitting per-feature integer counts | §4.4, §5.2, §7.2 |
| **C-2** | `D-FELDMAN` not decidable; H-MEMORISATION does not predict AUC ≈ 0.5; no row for the realistic outcome | statistic is now **conditional and incremental** (`ΔAUC` = stratum-conditional AUC of `FULL` minus `BASE`, paired, same folds); negative class is `CONFIG-MATCHED-CORRECT` **by stratification**; §6 restated **three-valued** with **band B** (identifiable but unconvertible) as an explicit **KILL under the F98 epitaph, explicitly not a Feldman confirmation**; the "conversion-equivalent AUC" is replaced by the **conversion-equivalent precision `π*`**, with the reason given | §5.2, §6 |
| **C-3** | cross-seed leakage; per-item target, per-(item, seed) rows | nested partition **is** the frozen arena fold partition, so all three seed-rows of an item move together and each item is scored disjoint from its own arena fold; **bootstrap resamples items, not rows**; `GATE-NESTED` emits a per-item check count | §5.2, §7.2 |
| **C-4** | `NET` break accounting non-conservative and mis-scoped | `net_s = |S ∩ wrong_s| − |S ∩ right_s|` over **all `n`** query items, **per seed**, so nothing is uncosted; `CONFIG-MATCHED-CORRECT` demoted to reporting stratification; **`GATE-SELFTEST`** asserts `net_s == n·Δacc_s` exactly | §5.3, §7.2 |
| **H-1** | `SHUFFLE-POP` blind to the leaks; self-contradictory marginals claim | permutation **pinned** (uniform over per-item targets, `default_rng(20260801)`, 200 draws); false marginals claim **withdrawn**; what it can and cannot test stated; `GATE-BLIND` named as the actual leak detector | §6.1 |
| **H-2** | undefined outcome in the decision rule | every rule made dataset-conjunctive in the same direction; **`KILL ≡ ¬CONTINUE`** | §8 |
| **H-3** | vote/confidence double-normalised; ladder vacuous | `score` defined by **literal reference to `mechfix_ops.py:94`**; `conf ≡ |score|`; `τ` frozen as **in-run quantiles of the OOF `|vote|` distribution among stable inversions** | §4.1, §4.3 |
| **H-4** | monotonicity over-advertised; `τ=0` is not the registered population | monotonicity restricted to `O1`, where it is arithmetic, and stated as false for `NET`/AUC; **`τ_hi` = median added as a registered co-primary**; KILL scope narrowed **in terms** per firing rule; `q_max` reported | §4.3, §8 |
| **H-5** | three surfaces disagree on the net bar; draft silently picked one | **adjudicated**: the governing `stage_0_reachability` text binds ⇒ `22.3 / 17.4` for `+0.030`, with the reason (`net ≡ n·Δacc` for a break-free oracle makes a `+0.050` net screen non-independent); `37.2 / 29.0` reported as a declared secondary that **scopes** a CONTINUE; **train arena `n = 744 / 579` stated with the figures** (R13) | §5.3 |
| **H-6** | Holm family named with no test; no macro-F1 currency; unpaired "paired bootstrap"; estimator unnamed | testing removed where the gate text is threshold-based, and **Holm retained over exactly the 4-member `K-FELDMAN` family**; **macro-F1 leg added** at `+0.030` per operating point; primary made a **difference**, hence paired; **logistic named primary, GBM named as a capacity check no rule reads**; `B`, `α`, RNG and all hyperparameters frozen | §5.2, §5.3, §5.4 |
| **H-7** | F47's ban_scope covers the feature family; cited as protocol, not adjudicated | **§3.1 added**: F47 quoted verbatim, ban engaged, the narrow distinction stated (different target; never consulted at prediction time), **`H-L4` added as a fourth HALT boundary**, **the registered prior set to F47's null**, and the F114 symmetric correction carried at its true weight | §3, §3.1, §6 |
| **H-8a** | F75/NCA kill-risk addressed nowhere; CONTINUE would have no legal successor | **§10 added**: the successor named concretely, adjudicated against F75's verbatim ban_scope, LITSWEEP5's mechanism-generalisation argument, C09's own dedup boundary and NCA_FORENSIC_RECON's ruling **and** its `:112` correction; **a CONTINUE is pre-declared void if no such operator can be named at Stage-1 entry** | §10 |
| **H-8b** | `GATE-FID` name collides; the real fidelity gate never run | gate **renamed `GATE-FLOOR`**; **`GATE-DEVFID` added**, running `headspace_fidelity.py` unmodified on the six `fold == -1` heads already inside the budget, with banked references and an honest scope note | §7.2 |
| **I-1** | ERRPAT provenance is test-split/deployed-head; strata frozen on test-derived buckets | every ERRPAT/F88 figure relabelled a **transferred expectation** with its provenance; **the configuration stratum is rebuilt on in-run, label-blind axes**, so the test-derived-bucket problem is removed rather than declared | §4.1, §4.2, §4.4 |
| **I-2** | floors pinned on accuracy only | **macro-F1 floors pinned** for all six cells | §7.2 |
| **I-3** | DET/CAL clauses uncited; fold contract unpinned | **DET-1…4 and CAL-0…5 adopted by citation with per-clause application**; fold contract pinned with the code assertion that enforces it | §2 |
| **I-4** | head-count arithmetic wrong; wall clock optimistic; no resume path | **30 + 6 = 36 total** stated correctly; budget restated at 60 s/head; **resume path named** (`headspace_mint.py:192-194` skip-if-exists) | §2 |
| **I-5** | `GATE-NULL` internally contradictory | contract rewritten: **with-null primary, remove-null a sensitivity, agreement required on the verdict not the metrics**, own recomputed bar; and the head-space non-transfer of the raw zero contract stated, with row 355 **re-measured this session** | §7.2 |
| **I-6** | features with no dynamic range | declared in advance with its source, and a **`FEATURE_DEGENERACY`** block emitted (per-feature std, distinct-value count); gates nothing | §5.2 |
| **I-7** | raw arena asserted, never specified | **fully specified**, with its banked pooled accuracies, and with the observation that **stability is undefined in a seed-free space**, so the raw leg prices single-pass inversions and is KILL-only | §8 |
| **I-8** | hyperparameters, `B`, `α`, RNG unfrozen | all frozen: estimator parameters, `B = 10000`, `α = 0.05`, `default_rng(20260801)` | §5.2 |
| **I-9** | §9 missing two scope items | **both added**: `O1` is a label-flip oracle, not a representation-level one; the two-dataset conjunct has **zero slack** with EN out of scope | §5.1, §9 |
| **I-10** | two carried corrections missing from §3 | **both added**: the LITSWEEP5 counter-text's *"downgraded, not vacated"* status with F114's basis; and round 14's *"indistinguishable"* reading of F88 null (3) | §3.2, §6.1 |

---

*v2. No hash frozen, no config written, no code implemented, no namespace created,
no job submitted, no cache or test path opened, no metric or result produced. Zero
GPU, zero SLURM, zero Modal, zero teacher call.*
