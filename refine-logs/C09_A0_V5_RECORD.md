# C09 Stage-0 (A0) — preregistration **v5** (round-4 repairs)

**Candidate.** C09 · Stable-Inversion Topology Surgery
**Registry claim.** *"OOF-stable high-confidence inversions identify topological
defects that can be corrected at encoder level while explicitly constraining break
exposure."*
**Registry dedup boundary.** *"Encoder-level topology intervention, not
thresholding, local reranking, verifier gating, NCA/SupCon, or hard-example
weighting alone."*
**Authorised by.** `TARGET_STATE.json::gate0_reopen_2026_07_31` —
`next_active_candidate_post_C04`.

> ## STATUS: `V5_REPAIRED_NOT_FROZEN_NOT_SUBMITTED` — awaiting fresh independent review.
>
> **Reading order.** v1 `C09_A0_PREREG_DRAFT.md` → R1 `C09_A0_PREREG_DRAFT_REVIEW_ROUND1.md`
> (`REVISE 4C/8H/10I`) → v2 `C09_A0_V2_RECORD.md` → R2 `C09_A0_V2_PREREG_REVIEW.md`
> (`REVISE 1C/8H/10I`) → v3 `C09_A0_V3_RECORD.md` → R3 `C09_A0_V3_PREREG_REVIEW.md`
> (`REVISE 1C/8H/11I`) → v4 `C09_A0_V4_RECORD.md` → R4 `C09_A0_V4_PREREG_REVIEW.md`
> (`REVISE 1C/4H/9I`) → **this file**. v1–v4 are superseded in full and must not be
> implemented. The v5 repair ledger is §12; v4's ledger for the round-3 findings is at
> `C09_A0_V4_RECORD.md §12`. R4 audited R1 as fully discharged, R2's six partials as
> closed except `K-DEG`, and R3 as discharged except H-1, H-2 and H-4; every open item
> is closed here.
>
> **Submission preconditions.** `gate0_reopen_2026_07_31.c09_next_step` and
> `dispositions.promoted.sequencing` both require that the CPU job *"waits for C04's
> tranche to terminate (serial-execution precedent) and for main-dialogue
> authorization"*. Both must be true before `sbatch`, alongside the `squeue -u jehc223`
> check, the sha256 re-verification of the frozen set, and namespace absence.
>
> No hash is frozen, no config is written, no job is submitted and no namespace is
> created by this document.

---

## 0. What v5 changes

Round 4 found four rules still sitting on the wrong *object*, three of them leaning
toward a false CONTINUE.

1. **`K-DEG` carried the campaign's `0.95` constant on a statistic it was never
   calibrated on.** F96 and F98 measure **prediction-vector agreement over all `n`
   items**; v4 measured **selected-set overlap** `|S ∩ twin| / k`. At `k/n ≈ 76/744`
   those differ by `pred_agree = 1 − 2k(1−ov)/n`, so v4's `ov ≥ 0.95` demanded
   `pred_agree ≥ 0.9898` and would have let an `S` that is 76 % a bare threshold shift
   through the campaign's only degeneracy screen. `K-DEG` now fires on the
   prediction-vector agreement F96/F98 actually measured.
2. **`τ_hi` still leaked, one hop further out.** v4 closed the scored item's
   *self*-target path and left the path round 3 actually named: `τ_hi^(g)` for a
   fitting fold `g ≠ f` is a median over `P_0 \ fold g`, which **contains** the scored
   item. v5 labels **every row in a given fit — fitting and scored — with that fit's
   own `τ_hi^(f)`**, which removes the scored item's label from its own model
   completely.
3. **`PERM-STRUCT` permuted rows, not items, and was called "the exact null" for a
   conditional hypothesis.** The row-level unit is the same `√3` anti-conservatism
   round 1 rated Critical; and a marginal block permutation is exact only when the
   blocks are independent, which they are not here. v5 permutes at the **item** level
   within each `(dataset, seed)` cell, states the null honestly as **marginal**, and
   adds a **residualised conditional null** that a CONTINUE must also clear.
4. **`NET`'s new `37.2 / 29.0` currency makes some frozen `(τ, k)` cells arithmetically
   dead**, and the all-single-class stratum case was routed to a no-verdict HALT with
   no repair path. v5 pre-declares the closed-form reachability caps, and demotes the
   stratum degeneracy from a HALT to a **data outcome** that forces
   `IDENTIFIABILITY_UNDERPOWERED` and a KILL scoped *"not identifiable at this power"*.

`STRATUM_OCCUPANCY`'s arbitrary `Σ n_pos·n_neg < 200` floor is replaced by a
power-justified one, a CONTINUE is forbidden from carrying the underpowered tag, and
nine provenance and specification items are corrected.

---

## 1. What A0 asks

`unified_pilot_gate.stage_0_reachability`, verbatim:

> *"Before teacher/GPU spend, the full-bank or representation-level oracle must reach
> at least +0.050 accuracy and +0.050 macro-F1 on at least two datasets, with enough
> net correct-minus-broken items for the +0.030 final bar."*

AGGNET/F98 held an oracle of `+0.1492 / +0.1520 / +0.2186` with 96–100 % of every
deployed error inside its function class and delivered `+0.0134 / −0.0069 / +0.0000`,
with the epitaph *"What binds is neither reach nor capacity but that the local
configuration carries no learnable signal about which neighbours to trust at
n = 549–744."* The Gate-0 reopen recorded this as governing
(`GATE0_REOPEN_2026-07-31.md:1005-1006`): **a large oracle is no longer evidence for a
candidate in this channel — it is the precondition every failed candidate already
met.**

A0 measures three things:

1. **Reach (`O1`)** — is the OOF-stable-inversion population large enough that fixing
   all of it would clear `+0.050 / +0.050`?
2. **Conditional identifiability (`D-FELDMAN`)** — *within a matched local
   configuration*, is "this item is an OOF-stable inversion" predictable from geometry
   alone, without any item's gold label at prediction time, **over and above** what
   the configuration already says?
3. **Conversion (`NET`)** — at the frozen operating points, does the population yield
   enough net correct-minus-broken items in the mandated currency, without collapsing
   into a bare threshold shift (§6.2)?

**A0 trains no encoder, touches no test split, and establishes that no operator
exists.** A CONTINUE means only that the population is large enough and locatable
enough to be worth building an operator for — and is void unless §11's precondition is
met.

---

## 2. Arena, instrument, cost

**Path.** The banked **fold-head / deployed-head arena** only. F113 stands: *a raw-key
arena may KILL but may not PROMOTE*.

**Instrument — verified present, nothing to build.**
`scripts/analysis/headspace_mint.py`, `headspace_arena.py`, `headspace_fidelity.py`,
`headspace_report.py`, plus the six banked
`headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`. **The mint is `headspace_mint.py`
invoked unmodified with its sha256 asserted; the only new code is the analysis script
and the sbatch driver.** `headspace_mint.CLI` admits only `{hatemm, zh}`, an
independent reason MHC-EN is out of scope: adding it would require editing a
sha-pinned artifact.

**Fold contract (pinned).** `StratifiedKFold(n_splits=5, shuffle=True,
random_state=0)` over the train split, stratified on the train label
(`mechnov_pairverify.K_FOLDS = 5`, `FOLD_SEED = 0`). `headspace_mint.py:203-216`
asserts this against the banked `scripts/analysis/vsw_ckpt/<ds>/f<fold>.npz` `ho_idx`
and refuses to run on mismatch. The assignment is a function of the label vector
alone and is therefore **identical across head seeds** — the property §5.2's nested
split relies on.

**Configuration.** 2 datasets × 3 head seeds × 5 folds = **30 fold-heads**, plus
2 × 3 = **6 deployed-configuration (`fold == -1`) heads** = **36 heads total**. Bank =
the fitting pool; queries = the held-out fifth. Query labels are **train-split**
labels held out from the head that judges them.

**Mint layout — following the banked precedent exactly (citations corrected, R4 I-8a).**
`scripts/analysis/headspace_drive.sh:20` assigns `out="$SC/mint_${DS}_s${seed}_f${tagf}.npz"`
and passes it at `:24-25` as `--out "$out" --scratch "$SC"`;
`scripts/slurm/c02_a0_cpu_v9.sbatch:85` assigns the same shape and passes it at `:89`
(to `c02_a0_mint.py`, C02's thin wrapper, not to `headspace_mint.py` directly). Either
way the `.npz` files sit at the **scratch root** and `headspace_mint.py:285-286` writes
each run's `run_rac` output tree under `<scratch>/mint/<tag>/`. C09 uses that layout
unchanged, invoking `headspace_mint.py` directly, with `--mintdir "$SC"` for both
`headspace_fidelity.py` (which hard-codes `mint_{}_s{}_ffull.npz` at `:66`) and the
analysis script. **`<scratch>/mint/` therefore contains `run_rac` output trees, not the
`.npz` files.**

**Datasets.** `HateMM` (n = 744) and `MHC_zh` (n = 579). Per-fold held-out counts from
the banked outputs: `149/149/149/149/148` and `116/116/116/116/115`. **MHC-EN is OUT
OF SCOPE**; **with EN out of scope the two-dataset requirement has zero slack.**

**Cost.** `0 GPU-hours`. Checkpoints are not persisted
(`headspace_mint.py:274-281`), so heads re-mint at ~25–60 s CPU each. Itemised budget:

| item | count | estimate |
|---|---|---|
| head mints | 36 | ≤ 36 CPU-min (C02's 36 banked mints ran 33.2–60.0 s, median 41.85) |
| deployed vote + features | 30 head fold-cells + 10 raw fold-cells | ~2 min |
| `D-FELDMAN` primary (LR: 2 sets × 5 folds × 2 `τ` × 2 ds) | 40 fits | < 1 min |
| GBM capacity check (same shape) | 40 fits | ~3 min |
| **`PERM-STRUCT`** (1000 draws × `FULL` only × 5 folds × 2 `τ` × 2 ds) | 20 000 LR fits | ~12 min |
| **`PERM-STRUCT-COND`** (same shape, residualised) | 20 000 LR fits | ~12 min |
| `SHUFFLE-POP` (200 draws × 2 sets × 5 folds × 2 `τ` × 2 ds) | 8 000 LR fits | ~5 min |
| `RANDOM-POP` (200 draws, same shape) | 8 000 LR fits | ~5 min |
| `UNSTABLE-POP` (2 sets × 5 folds × 2 ds) | 20 fits | < 1 min |
| item bootstrap `B = 10000` (re-scoring only, **no refits**) | — | ~5 min |
| `K-DEG` twins + `FIXK` grid + `GATE-FIXK20` | 8 `k′` × 30 cells | ~2 min |
| **raw-arena leg** (vote, features, `D-FELDMAN`, its permutations and bootstrap, at 1 "seed" instead of 3) | 10 fold-cells | ~8 min |
| `GATE-DEVFID` (`headspace_fidelity.py`, banked trainlogs) | 2 | seconds |

**Total ≈ 90 CPU-minutes budgeted** (R4 I-7: v4's table omitted the raw leg's own
inferential cost, which is scoped `× 2 spaces` and not only `× 2 ds`; v5 adds it as its
own line). The estimates are deliberately generous: the permutation lines assume ~35 ms
per fit-and-score at this scale, and the bootstrap re-scores banked OOF predictions
without refitting. One CPU-only SLURM job, 8 CPU / 32 GB / **no GPU, no `--time`**;
C02's A0 (job `13847`: 8 CPU / 0 GPU / 32 G) ran `00:29:49`. **Resume path:**
`headspace_mint.py:192-194` skips any `--out` that already exists and the driver is a
sequential loop.

**F88's binding caveat is satisfied by construction:** *"a CPU-trained arm must be
paired against a CPU-TRAINED FLOOR, never against the banked GPU floor."* Every arm
and every floor here is minted inside the same CPU fold-head arena.

**Standing clauses.** `PREGATE_DETERMINISM_CLAUSE.md` **DET-1 … DET-4** and
`PREGATE_CALIBRATION_CLAUSE.md` **CAL-0 … CAL-5**, binding.

- **DET-1** `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=8`
  exported before any Python process starts; `headspace_mint.det1_assert` hard-fails
  otherwise. **DET-2** the runtime block in every output JSON. **DET-3** parity at
  Tier A / Tier B, **4 decimal places**, never beyond. **DET-4** the primary estimator
  is `LogisticRegression(solver="lbfgs")`, the convex arm (4-dp invariant across the
  thread grid, `PREGATE_DETERMINISM_CLAUSE §1.3`); the GBM arm is a capacity check
  **no decision rule reads** and carries no Tier-C band because no verdict rests on it.
- **CAL-0** restated and scoped: *"The raw train-space arena is not established as
  predictive of deployed effects."* C09's decision arena is the fold-head arena; the
  raw arena appears only as F113's KILL-only corroborator (§9). **CAL-1** governs that
  corroborator; its decisive bars are within-arena relative comparisons.
  **CAL-2 leg (1) is now RUN as a HALT gate** — `GATE-FIXK20` (§8.1). **CAL-2 leg (2)
  is out of scope, with the reason corrected (R4 I-3).** v4 called its comparator a
  *"raw-arena k-curve"*; `PREGATE_CALIBRATION_CLAUSE.md:80-81` in fact specifies *"the
  Spearman against F94's banked **deployed** k-curve (`scripts/analysis/ksweep_OUT.json`;
  primary arms HateMM final / MHC-ZH final / MHC-EN ARM-V) over `k ∈ {5,7,10,15}`
  ONLY"*, and the deployed curve does not change with the arena. The correct reasons
  for skipping it are two: (i) CAL-2 is scoped by its own header to *"every `$0`
  pregate run in the **raw banked train-space arena**"*, which is not C09's decision
  arena; and (ii) `ksweep_OUT.json`'s per-`k` payload is a **test-split** read
  (`curves.<DS>/seed<n>/<protocol>.test`), and this run materialises no test-derived
  quantity anywhere, not even in a non-gating report. CAL-2 itself declares leg (2)
  *"deliberately no threshold"* and *"not as a validity gate"*, so nothing gating is
  lost. **CAL-3** is discharged in §9. **CAL-4** labels every quantity closed-form or
  trained (§5.6). **CAL-5** runs **against** C09: the Stage-1 operator is a
  channel-(a)/(d) object, which *"carries NO transfer warrant and must say so in its
  limitations."* Said, in §10.

---

## 3. Label-use discipline

Identifying OOF-stable inversions requires reading train labels out of fold. This is
**legal**, resolved on two written texts:

- `autoresearch/goal_mllm_plus3/state/progress.json:25` — *"Legal attack on
  selection-locked pools = trained selector/reshaper on train labels only (F66 binds
  only fixed-map phi0)."*
- `refine-logs/LITSWEEP3_DATA_CENTRIC.md:82` — *"those select **per test instance**;
  curation selects **train items once, globally, applied identically to every test
  query** — a symmetric operator, so law-III/F66's per-item ban does not apply to the
  mechanism (though Wall-A still caps the achievable magnitude)."* The same section
  prices the mechanism at *"+3 any dataset: ~1-2%"* (`:95`) and *"at most
  +0.001-0.006"* (`:91`).

Four boundaries flip C09 to illegal and are **HALT** conditions:

- **`H-L1`.** Any query-time consultation of the stability statistic or of the
  `D-FELDMAN` classifier. F47 fires directly, and its escape clause is closed: an
  OOF-stability statistic *is* "derivable from banked features/votes".
- **`H-L2`.** Any per-item exception that survives to inference as a per-item rule.
- **`H-L3` (restated — R3 I-8).** **Any read of a dev or test label into any decision
  quantity, and any read of a test path, at any stage, by any code path.** The
  materialisation carve-out that v3 wrote as an exception is now inside the boundary
  itself, because the instrument materialises dev labels by its own contract:
  `headspace_mint.py:322-324`'s `np.savez` stores `lab_dev` in **every one of the 36
  mints**, folds `0–4` included and not only the six `fold == -1` heads (R4 I-8b), and
  `run_rac`'s dev evaluation computes the `eval_curve` `GATE-DEVFID` consumes. Those
  are legal and expected; what is banned is any of it reaching a decision quantity.
  `GATE-LEDGER` enforces exactly this predicate, and its **declared expected dev-label
  materialisation count is 36**, one per mint, all outside every decision quantity.
- **`H-L4`.** Any use of `D-FELDMAN` — the classifier, its score, or any monotone
  function of either — as a **selector, gate, router, abstention rule or risk ordering
  over a deployed decision**. Banned by measurement twice over (F47, F97, F98).

### 3.1 `D-FELDMAN` against F47's ban_scope — adjudicated

F47's `ban_scope`, verbatim:

> *"Per-item cross-channel selection/routing over banked channels: CLOSED at all three
> supervision sources (unsupervised=K9 zeros; train-supervised=memorization-degenerate
> target, CLIP LOO 0.998; dev-supervised=negative at CV ceiling −0.046 < perm-null).
> Decision-level meta-features (vote margins, purity, sub-votes, confidence
> differential, transcript stats) carry NO per-item routing signal, GBM or linear. Do
> NOT re-propose per-item selectors over frozen channels regardless of feature family
> or nonlinearity unless the selector input is a genuinely NEW information source not
> derivable from banked features/votes."*

`D-FELDMAN`'s feature family is **literally the family that sentence names**, and its
input is **not** a new information source. The ban is engaged:

- **What the ban closes and C09 accepts.** Using these features to select, route or
  gate a deployed decision is closed. `H-L4` writes that in.
- **What the ban does not reach.** F47's measured object is a per-item router over
  frozen channels, target *which channel to trust for this item*. `D-FELDMAN`'s target
  is a different random variable — *is this item a three-seed-stable inversion* — and
  its output is never consulted at prediction time. This distinction is **narrow**: if
  a future proposal reads the probe's score at query time, F47 fires and `H-L1`/`H-L4`
  HALT.
- **The symmetric correction, at its true weight.** F47's train-supervised leg is
  *"memorization-degenerate target, CLIP LOO 0.998"*; F114 rules that premise a
  **CLIP** number while the deployed Qwen heads sit at `0.9406 / 0.8915 / 0.8154`, so
  the saturation objection does not transfer to this arena — F113's own reason for
  building it (*"That objection does NOT apply to a head trained on 4/5 of the train
  split and queried with the held-out fifth"*). It does not touch F47's unsupervised or
  dev-supervised legs, and it does not weaken the decision-level-meta-features
  sentence, which was measured independently of the 0.998 premise.

### 3.2 `NET` against F98's ban_scope (b) — adjudicated

F98's `ban_scope`, verbatim in the relevant part:

> *"Concretely banned on this neighbourhood object: … (b) ANY per-item selector,
> router or adjudication gate over the same neighbourhood WITH ANY FEATURE FAMILY —
> the verifier features are dead by K-VGA-3 and the F47 features have a measured,
> already-banked ceiling of +0.0269 … (d) re-deriving the HateMM train-arena threshold
> observation — three independent measurements now agree it is ~87% a bare threshold
> move and it is measured DEAD in the deployed head space on test (ERRPAT-HateMM
> sec2.1: +0.0000/+0.0016)."*

**§5.3's `NET` is arithmetically that object.** Reading the ban narrower than its own
text is not available. The adjudication:

1. **The banned object is constructed and measured as a pricing instrument on the train
   arena, and is never deployed** (R4 I-6 — v4's *"the banned object is not built"* was
   too weak a description of what runs). §5.3 does fit the `FULL` logistic probe, rank
   all `n` query items by its OOF probability, take the top `k` and cost the flips —
   that is F98(b)'s object, constructed and measured. What is not done is *deploying*
   it: it is never consulted at query time, and `H-L4` forecloses it as a component, at
   Stage-0 and at Stage-1.
2. **Its measured ceiling is this design's registered prior** (§6.1).
3. **What `NET` does and does not bound (R3 H-5, v3's over-claim withdrawn).**
   `NET` prices **one specific per-item selector**: top-`k` by a 12-feature `lbfgs`
   logistic probe over hand-built geometry features. Its *operator class* dominates
   §11's global symmetric reshaper, but this *particular instrument* may be weaker
   than a reshaper that optimises `φ` end-to-end with the vote in the loop. **`NET`
   therefore bounds the successor's conversion in neither direction.** v3's "optimistic
   upper bound" and its "a `K-NET` failure is a fortiori a KILL of the successor"
   corollary are **withdrawn**. **`O1` is the only upper bound this A0 produces.** The
   correct half stands: **a `K-NET` pass is not evidence the successor converts**, and
   §11's precondition — not `NET` — is what a CONTINUE must satisfy.
4. **Clause (d) is engaged**, and is why §6.2's degeneracy control gates the verdict.

### 3.3 The counter-text, at its adjudicated weight

`LITSWEEP5_COMPLETENESS.md` §4(ii), *"The contradiction (load-bearing)"*, observes
that the ruling's two blessed classes — *"Trained SELECTOR on train labels"* and
*"Trained symmetric RESHAPER on train labels"* — are *"both already measured dead"*,
and that the ruling *"was written at lit-round-count 3 — before F75/F77/L1 sharpened
the walls."*

**DOWNGRADED, NOT VACATED** (reopen R7 I-2): §4(ii)'s first blessed-class death rests
on *"the deployed kNN vote memorizes train (CLIP LOO 0.998)"*, which F114 rules a CLIP
number against deployed Qwen heads at `0.9406 / 0.8915 / 0.8154`, leaving train-side
headroom 30×–92× larger; `LITSWEEP5_COMPLETENESS.md` is not among the nine records
F114 corrected.

**§4(ii)'s independent leg, with its conclusion.** `LITSWEEP5_COMPLETENESS.md:128`:

> *"train-disagreement 'Qwen-correct' = **0/109, 0/102, 0/92**, and that train base
> rate is the *inverse* of the ~0.55 test base rate (L1 §0, F47 §3.2). Training labels
> **cannot** supervise the test-time selection decision in this pipeline — a
> data-generating-process obstacle upstream of any selector capacity."*

**The obstacle it names is an inverse-base-rate obstacle, and it is a property of the
saturated full-train arena, not of this one.** On a full-train-fitted head the train
error rate is ~0 while the test error rate is ~0.11–0.15, so the selector's training
target is near-empty. In the fold-head arena the query items are held out from the
head that judges them, and the banked pooled deployed accuracies `0.8858–0.8946` give
a train error rate of **`0.1054–0.1142`** — comparable to the deployed test error
rate, not its inverse. The `0/109, 0/102, 0/92` counts are therefore not transportable
here; that is what F113 built the arena to fix. **What survives as a headwind:** the
general form of the objection — that a train-supervised target may be a different
object from the test-time one — is not refuted by an error-rate match alone, and §10
scopes every verdict here to the train arena for exactly that reason.

### 3.4 `D-FELDMAN` is a probe, not a component

Whether a *global operator acting uniformly on the region* is legal and buildable is a
Stage-1 question with its own gate (§11); A0 makes no claim about it.

---

## 4. Population definition

### 4.1 The deployed decision and the vote scale

`score_{i,s}` is defined by **literal reference to `scripts/analysis/mechfix_ops.py:94`**:

```python
votes = ((lab * 2 - 1) * sim * w).sum(1) / w.sum()
```

with `w = _rank_weights(20) = [20, 19, …, 1]`, `Σw = 210`, `sim` the float32 faiss
inner products of L2-normalised keys, `lab` the **bank** labels of the top-20, and the
decision `predict 1 iff score ≥ 0` (`mechfix_ops.py:95`). **The vote is already divided
by `Σw`.**

For orientation, and declared as a **transferred expectation, not a measurement of
this arena**: `ERRPAT_HateMM_2026-07-26.md:130` reports *"median |vote| (3-seed mean
vote)"* of **`0.7267`** for errors against **`0.9873`** for always-correct items, on
the **test split** under a **CPU-reconstructed proxy** of the deployed head
(`ERRPAT_HateMM §0.1`, 52 s/seed).

### 4.2 Inversions and stability

Item `i` is an **inversion at seed `s`** iff its deployed prediction disagrees with its
gold train label. `i` is an **OOF-stable inversion** iff it is an inversion in **all
three** head seeds; an **unstable error** iff in exactly 1 or 2; **always correct** iff
in none.

**Provenance of the expectation that this population is large.** Registry text
verbatim (`gate0_reopen_2026_07_31.dispositions.promoted.supporting_evidence_verified`):

> *"F88 FAITHFUL: error sets ~90% seed-invariant - HateMM 24-25 of 26-28 errors wrong
> in 3/3 seeds (89-93%); ZH 22 of the 25-item union wrong 3/3 with NOTHING at exactly
> 2/3 and all 12 false negatives 3/3-stable"*

Measured on the **test split** under the **deployed-head proxy**. These are
**transferred expectations**; nothing in the decision rule reads an F88 number.

### 4.3 Confidence thresholds — a primary and an out-of-fold co-primary

`c_i ≡ mean_s |score_{i,s}|`, the **per-item** confidence scale. **There is exactly one
confidence scale in this design and it is per item** — §5.3's item score, §6.2's
twins and §4.3's thresholds all use `c_i`.

- **`τ_0 = 0` — PRIMARY.** All OOF-stable inversions.
- **`τ_hi` — the REGISTERED "high-confidence" co-primary, computed OUT OF FOLD.** For
  each arena fold `f`:

  > **`τ_hi^(f) = numpy.median( c_i : i ∈ P_0 and fold(i) ≠ f )`**, using
  > `numpy.median`'s default linear interpolation on even counts.

  **Two objects, computed two ways, and the distinction is load-bearing (R4 H-1).**

  > **(a) The POPULATION** — read by `|P_τ|`, `K-REACH`, `O1` and `NET`'s `k`:
  > **`P_{τ_hi} ≡ { i ∈ P_0 : c_i ≥ τ_hi^(fold(i)) }`**. Each item is judged by the
  > threshold computed with its own fold excluded, so no item's own label enters its
  > own membership through the threshold.
  >
  > **(b) The FITTING TARGET** — read only by `D-FELDMAN`'s estimator: for the model
  > that scores fold `f`, **every row in that fit — the fitting rows from folds `≠ f`
  > and the scored rows of fold `f` alike — is labelled with that fit's own
  > `τ_hi^(f)`**, i.e. `y_j^{(f)} = [ j ∈ P_0 ] ∧ [ c_j ≥ τ_hi^{(f)} ]`.

  **Why (b) is not (a).** v3 used a full-sample median, which let item `i`'s gold label
  shift the threshold defining the positive class for the rows fitting the model that
  scores `i`. v4 replaced it with `τ_hi^(fold(i))` per item — which closed the scored
  item's *self*-target path but left the path round 3 actually named: a fitting row
  `j ∈ fold g ≠ f` carried the target `c_j ≥ τ_hi^(g)`, and `τ_hi^(g)` is a median over
  `P_0 \ fold g`, which **contains** item `i`. So `i`'s label still reached the fit that
  scored `i`, at `O(1/|P_0|) ≈ 1.4 %`, in the CONTINUE direction. Under (b) the fit
  that scores fold `f` uses one threshold, `τ_hi^(f)`, computed with fold `f` excluded,
  so **item `i`'s label enters that fit nowhere** — not through any fitting row's
  target, not through the threshold. For the scored item itself (a) and (b) coincide,
  because `fold(i) = f`, so the population and the evaluation target agree by
  construction. `GATE-NESTED` asserts (b) as a per-item check.

  **All five `τ_hi^(f)` values and their spread are emitted per dataset**, together
  with the full-sample median as a descriptive figure that no rule reads.

**Where monotonicity holds.** `|P_τ|` is monotone non-increasing in `τ`; flipping a
strictly larger set of errors weakly raises both per-class F1s, so `ΔmF1_{O1}` is
monotone in the flipped set as well. **`K-REACH` firing at `τ_0` closes every `τ ≥ 0`
on both metrics by arithmetic.** It is **false** for `NET` and `ΔAUC`. Both are
evaluated **and reported** at `τ_0` **and** `τ_hi` unconditionally — including when
`K-REACH` fails at `τ_hi`.

**Pre-declared arithmetic consequence.** `|P_{τ_hi}| ≈ |P_0|/2`, so `K-REACH` at
`τ_hi` is approximately `|P_0|/n ≥ 0.10` — close to the plausible value on the
transferred F88 rates. **The co-primary may therefore be decided by an arithmetic
identity rather than by identifiability or conversion.** Declared now; §9 gives it its
own scope bullet. **`q_max`** — the largest quantile of the `c_i` distribution over
`P_0` at which reach still clears `+0.050` — is reported as a descriptive quantity no
rule reads. *(v3's vestigial `q25`/`q75` sensitivity ladder is deleted — R3 I-11e.)*

### 4.4 Frozen ancillary definitions

- **Right analogue** of `i`: the highest-ranked bank item carrying `i`'s gold label.
  **Mechanism diagnostic only; it reads `i`'s gold label and is excluded from every
  feature set by `GATE-BLIND`.** Transferred motivating measurements, with their
  spaces stated correctly (R3 I-2):
  - **HateMM — test split, proxy head, deployed head space** (`ERRPAT_HateMM §2`,
    *"All from the saved per-item top-20 neighbour lists"*): `:134` median rank of the
    first true-label neighbour `3.0`; `:135` `6 / 27` errors with none in the top-20.
  - **MHC-ZH — test-split errors, `pre-head raw fused space` over the full 579-row
    bank**, *not* the head space: `ERRPAT_MHC-ZH:233-235` — *"In the pre-head raw
    fused space over the full 579-row bank, the first same-gold-class train neighbour
    sits at median rank 1.5 for the core errors (11 of 22 at rank 1; all 22 within
    rank 14). The right analogues are present and top-ranked; they are simply
    out-voted."* **The same record's `:237-240` measures the head space collapsing that
    population:** raw fused purity `0.400` (5 of 22 still majority-correct) →
    *"deployed head space 0.1167 (0 of 22 majority-correct)"*, while correct items
    sharpen `0.85 → 0.9833`. The ZH analogue figure is therefore a **raw-space** fact
    of exactly the kind F113 exists to stop transferring, and is carried as motivation
    only.
- **`pred_purity_{i,s}`**: fraction of `i`'s top-20 whose **bank** label equals `i`'s
  own **predicted** class at seed `s`. Label-blind for the scored item.
- **Configuration stratum (frozen, label-blind).** Computed **per (dataset, seed)
  cell**, as the cross of `|score_{i,s}|` **tercile** (edges from that cell's own `n`
  query items) × `pred_purity_{i,s}` bucket
  `{[0, 0.60), [0.60, 0.80), [0.80, 0.95), [0.95, 1.0]}`. 12 strata per cell. Both
  axes computed in-run and label-blind; edges frozen here and not tuned.

---

## 5. The three measured quantities

### 5.1 `O1` — reach (necessary; an upper bound, and declared as one)

For each seed `s`, flip the prediction of every item in `P_τ` and recompute accuracy
and macro-F1 against that (dataset, seed) cell's deployed floor.
`Δacc_{O1} = |P_τ| / n` identically for every seed; `ΔmF1` is recomputed from the
realised confusion matrix. Primary = mean over three seeds; per-seed reported.

**Scope.** `O1` is a **label-flip oracle over one nominated population**, not the
*"full-bank or representation-level oracle"* the gate text names. It is the tightest
zero-cost **upper bound** on what any operator confined to fixing stable inversions
could reach — and, after §3.2(3), **the only upper bound this A0 produces**.

### 5.2 `D-FELDMAN` — conditional, incremental identifiability

**Why conditional and incremental.** H-MEMORISATION does **not** predict unconditional
AUC ≈ 0.5: Feldman's singletons *are* the low-density, weak-margin items, so a
label-blind feature set separates them under either hypothesis, and the separation is
already banked (`ERRPAT_HateMM:130`). v4 conditions on the configuration stratum and
measures the **increment over a configuration-only baseline**.

**Rows, split, estimators.**
- Rows are `(item, seed)`; the target is **per item**, constant across its three rows.
- **The nested-CV partition *is* the frozen 5-fold arena partition.** An item's score
  comes from a model fit on items from the other four arena folds only — which groups
  all three seed-rows of an item, makes the scored item disjoint from its own arena
  fold, and introduces no new hyperparameter and no RNG. With §4.3's per-fold `τ_hi`,
  the **target definition** is now out-of-fold as well.
- **Primary estimator (DET-4):** `LogisticRegression(penalty="l2", C=1.0,
  solver="lbfgs", max_iter=2000, class_weight="balanced", tol=1e-6)` on z-scored
  features, standardisation fit on the training folds only.
  **Capacity check, read by no decision rule:**
  `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=3,
  l2_regularization=1.0, random_state=20260801)`. This mirrors F47's own two-family
  protocol, which found *"NO per-item routing signal, GBM or linear"*.

**Two frozen feature sets — 7 and 12.**

*`BASE` — configuration-only (7).* `|score_{i,s}|`; `pred_purity_{i,s}`; mean and
standard deviation of the top-20 similarities; the similarity gap between ranks 1 and
20; local bank density (mean similarity to the 50 nearest bank items); the item's own
L2 norm before normalisation.

*`FULL` — `BASE` + structural block (5).*
1. **rank of the first neighbour whose bank label differs from the top-20 majority
   bank label**, with the sentinel **frozen at `21` when all 20 top-20 bank labels are
   identical** (R3 H-7). This case is common: `ERRPAT_HateMM:133` measures median
   rank-weighted top-20 purity toward the true label at `1.0000` for always-correct
   items and `ERRPAT_MHC-ZH:220` measures median top-20 purity `1.00` for correct vs
   `0.15` for errors, and the head space is purer still (`ERRPAT_MHC-ZH:237-240`). The
   sentinel is therefore a near-perfect class indicator by construction, so the
   **fraction of uniform-label neighbourhoods, per class and per cell, is emitted in
   `FEATURE_DEGENERACY`** and must be read alongside any `ΔAUC`.
2. the number of runs (label changes) in the rank-ordered bank-label tuple of the
   top-20;
3. mean and 4. standard deviation of the **bank-side degree** of the top-20 members
   (how often each appears in other query items' top-20 within the same fold);
5. the signed gap between the best rank-1 similarity to a class-1 bank item and to a
   class-0 bank item.

> **v2's sixth structural feature — the count of an item's top-20 members that are
> themselves stable inversions — remains DELETED** (round-2 C-1). It was a function of
> other query items' gold *targets*; its "training-fold items only" exemption was false
> for the model's own fitting rows; and `GATE-BLIND` was structurally blind to a
> derived target array.

**Label-blindness, stated precisely (R3 I-4).**

> **`FULL` contains no feature that reads the *scored item's own* gold label, and no
> feature that reads any *target-derived* array (`is_inversion[seed]`,
> `is_stable_inversion`).** **Bank labels — the gold labels of the other four folds'
> items — are read, legally and by design, exactly as the deployed vote reads them**
> (`mechfix_ops.py:91`), and are counted by `GATE-BLIND` as `bank_label_reads`, a
> nonzero legal integer. v3's blanket phrasing ("no query item's gold label") was
> false in this arena, where every train item is a query in its own fold and a bank
> member in the other four.

**Extra retrieval the features need, declared.** `deployed_vote` returns the top-20
`(I, sim)` only. Local bank density requires a `k = 50` faiss search of each query
against the fold's bank; the per-class rank-1 gap requires a per-class top-1 search.
Both use `mechfix_ops._norm32` + `_flat_ip` — the same engine on the same normalised
keys — so no arm-to-arm delta can be an engine artefact.

*Declared feature-degeneracy read.* `ERRPAT_HateMM:139-141` (R4 I-8c: the *"saturated
at ~0.9999"* characterisation is at `:139-140`; `:131`'s row carries the underlying
`0.999852 / 0.999976`): *"Cosine is saturated at ~0.9999 for both errors and correct
items (the head space is collapsed onto a narrow cone). Distance-based
abstention/gating has essentially no dynamic range to work with."* The run emits
`FEATURE_DEGENERACY` — per-feature standard deviation, distinct-value count, and the
uniform-neighbourhood fraction — per (dataset, seed). It gates nothing.

**Primary statistic, fully specified.**

> For each (dataset, seed) **cell**: partition the cell's rows into its 12 frozen
> strata. For each stratum with ≥ 1 positive and ≥ 1 negative, compute the
> Mann-Whitney probability that a random positive row outscores a random negative row
> **from the same stratum** (ties ½). Pool across strata weighted by `n_pos × n_neg`;
> **strata with zero positives or zero negatives receive weight 0**. That is
> `AUC_strat` for the cell. **Pooling unit is the row** within a cell; the per-dataset
> value is the **mean over the three seed cells**.
>
> **`ΔAUC = AUC_strat(FULL) − AUC_strat(BASE)`**, per dataset, both terms on identical
> OOF folds, identical rows and identical strata — a genuinely **paired** quantity.
>
> **Strata are frozen from the full sample and held fixed across every bootstrap
> resample and every permutation draw.** They are a design object, not an estimate.
>
> **Degenerate case — a DATA outcome, not a HALT (R4 H-4).** If in some cell *every*
> stratum is single-class, `AUC_strat` is undefined there. v4 routed this to a
> no-verdict HALT; but single-class strata are a property of the data under frozen
> strata, not of the harness — the design's own cited marginals (`ERRPAT_HateMM:130`,
> `:133`; `ERRPAT_MHC-ZH:220`, `:237-240`) make it plausible, and there would be
> nothing to repair and nothing gained by resubmitting the identical frozen design.
> So instead: **the affected cell is dropped from that dataset's seed-mean, the drop is
> emitted with the cell named, and `IDENTIFIABILITY_UNDERPOWERED` is forced on that
> dataset.** HALT status is reserved for the nine instrument gates of §8.1 and
> `SHUFFLE-POP`'s band, all of which have a genuine repair path.

**Positives:** `P_τ`. **Negatives:** `CONFIG-MATCHED-CORRECT` = query items correct at
all three seeds; matching is achieved **by the stratification itself** — no sampling,
no RNG, no discarded data. Unstable errors are excluded from both classes
(`UNSTABLE-POP`'s subject) and are still fully costed in `NET`.

**`STRATUM_OCCUPANCY` and the power rule (R3 H-3, frozen now).** `ERRPAT`'s own
marginals predict that both stratification axes separate the classes almost completely
(`|vote|` `0.7267` vs `0.9873` at `:130`; purity `0.1667` vs `1.0000` at `:133`; ZH
`0.15` vs `1.00` at `:220`), so most strata may be single-class and `AUC_strat` may be
carried by two or three — worst at `τ_hi`. A null from **absence of two-class overlap**
would otherwise be published as band A, i.e. as evidence for H-MEMORISATION, and
`SHUFFLE-POP` cannot detect it (permuting the target spreads positives uniformly across
strata, destroying the very concentration that creates the risk). Therefore:

> **Emitted per dataset × seed × `τ`:** the number of strata with weight `> 0`, the
> `(n_pos, n_neg)` of each, `Σ n_pos·n_neg`, and `p_w` ≡ the number of **positives
> lying in weighted strata**.
>
> **Frozen power rule, with its justification (R4 I-9 — v4's `Σ n_pos·n_neg < 200` was
> arbitrary and is replaced).** For a pooled AUC with `p` positives and `q ≫ p`
> negatives, the Hanley–McNeil standard error at `A = 0.5` is
> `sqrt((0.25 + (p+q−2)/12)/(pq)) → sqrt(1/(12p))`, so the *smaller* class count
> governs: `p = 30 ⇒ SE ≈ 0.053`, `p = 10 ⇒ SE ≈ 0.091`. Therefore:
> **a dataset's `ΔAUC` is tagged `IDENTIFIABILITY_UNDERPOWERED` iff its seed-mean `p_w`
> is `< 30`, or its seed-mean number of weighted strata is `< 3`, or any of its cells
> was dropped under the degenerate case above.** `Σ n_pos·n_neg` is retained as a
> descriptive emission only.
>
> **Consequences, both directions.**
> (i) Band A's optional statement *"H-MEMORISATION is consistent with this object"* may
> be published **only if the tag is absent on both datasets**; with the tag present the
> KILL's scope reads *"not identifiable **at this power**"*.
> (ii) **A CONTINUE may not be published while the tag is present on either dataset**
> (R4 I-9). A probe that cannot resolve `ΔAUC` cannot license Stage-1 spend, so
> `K-FELDMAN` is defined to *fail* on a tagged dataset regardless of its `p`. This is
> the conservative direction and is consistent with §9's rule that a CONTINUE needs a
> positive result on **both** datasets.
>
> **Pre-declared expectation.** At `τ_hi` the positive count roughly halves
> (`|P_{τ_hi}| ≈ 38 / 28` on the transferred F88 rates), so `p_w < 30` is **plausible on
> both datasets at the co-primary**, and on MHC-ZH it is not implausible at `τ_0`
> either (`|P_0| ≈ 55`). Declared now, in advance: this A0's identifiability probe has
> limited power on the smaller dataset and at the higher threshold, and a tagged KILL
> there is a statement about power, not about geometry.

**Inference — the null (R3 H-2, repaired onto the right object).** `K-FELDMAN`'s
hypothesis is **incremental**: *the five structural features add nothing beyond
`BASE`*. Two nulls are computed:

- **`PERM-STRUCT` — the primary null, permuted at the ITEM level (R4 H-2i).**
  `D = 1000` draws. In each draw, **within each `(dataset, seed)` cell**, a permutation
  `π` of the cell's **items** is drawn and each item receives the structural feature
  vector of donor item `π(i)`; the **same `π` is applied in all three seed cells**, so
  an item's three seed-rows keep a single donor and the `(item, seed)` structure is
  preserved. `FULL` is then **refit** on the same folds with the same frozen strata;
  `BASE` is untouched, because it is untouched under the incremental null.
  `p = (1 + #{d : ΔAUC_d ≥ ΔAUC_obs}) / (D + 1)`, floor `1/1001 = 9.99e-4`.
  RNG `numpy.random.default_rng(20260801)`.
  v4 permuted **rows**, which gave each item three unrelated structural vectors and let
  the permuted noise average over `3n` quasi-independent rows instead of `n` — the same
  `√3` anti-conservatism round 1 rated Critical for the bootstrap, in the CONTINUE
  direction. The item is this design's resampling unit everywhere and now here too.
  **What this null is, stated honestly (R4 H-2ii).** A naive block permutation is exact
  for the **marginal** null *"the structural block is exchangeable noise, independent of
  the target and of `BASE`"*. It is **not** exact for the **conditional** null
  *"the block adds nothing given `BASE`"* when the blocks are dependent — and they are:
  structural feature 1 (first-differing-label rank, sentinel 21) is nearly a
  deterministic function of `pred_purity`, which is in `BASE`. Under misspecification of
  the linear logit, a conditionally-uninformative but `BASE`-correlated block can raise
  OOF AUC and `PERM-STRUCT` will reject. v4's claim that this is *"the exact null"* is
  **withdrawn**.
- **`PERM-STRUCT-COND` — a residualised conditional null, and a CONTINUE must clear it
  too (R4 H-2, option c).** Same `D = 1000` item-level draws. For each structural
  feature, an OLS regression on the `BASE` block is fitted **on the training folds
  only**; the feature is decomposed into fitted part + residual; the **residuals** are
  permuted at the item level as above and the feature reconstructed as
  `fitted + permuted residual`. This preserves each structural feature's conditional
  mean given `BASE` while destroying its conditional association with the target, which
  is the object `K-FELDMAN`'s hypothesis is about. Its ASL is `p_cond`, computed the
  same way. **`K-FELDMAN` clears at a `τ` only if BOTH `p` and `p_cond` reject after
  Holm.** A rejection on the marginal null alone is reported as
  `MARGINAL_ONLY_NOT_CONDITIONAL` and is a **KILL**, because the design then cannot
  distinguish *"adds conditional information"* from *"re-encodes `BASE` better"*.
- **`SHUFFLE-POP` — retained as the pipeline-leak band check** (§6.3) and additionally
  reported as a **secondary ASL** on the joint null, so all three readings are on the
  record.

**Holm and the dataset conjunction.** Within each dataset, Holm is applied over the
**2 `τ` hypotheses** at `α = 0.05`, **separately to the `PERM-STRUCT` family and to the
`PERM-STRUCT-COND` family**: sort the two `p`, compare the smaller to `α/2 = 0.025`
and, if it rejects, the larger to `α = 0.05`. A `τ` clears only if it rejects in both
families. **The dataset conjunction is an intersection-union test and receives no
correction** — an IUT's level is controlled by the largest component p-value.

**The item bootstrap, and what it is for.** One-sided **item-level** bootstrap:
resample **items** (not rows) with replacement, `B = 10000`, RNG
`default_rng(20260801)`, **re-scoring the banked OOF predictions without refitting**.
It is reported as the interval on `ΔAUC` and is **explicitly conditional on the fitted
model**; it is **not** `K-FELDMAN`'s p-value. Resampling items rather than rows is the
second half of round-1's C-3 repair: a row-level bootstrap would be
anti-conservatively narrow by roughly `√3`.

**On the "conversion-equivalent AUC".** Not well-defined — AUC does not determine
precision at a fixed selected count without the score distribution. The conversion leg
is adjudicated where it *is* exactly decidable, in precision and net-item space
(`K-NET`), and the run reports the **conversion-equivalent precision**
`π* = (1 + bar/k)/2` at every operating point alongside the realised precision.

### 5.3 `NET` — conversion, fully costed

**Accounting.** At an operating point the classifier selects a set `S` of `k` items
from **all `n` query items**. For each seed `s`:

```
net_s = |{ i ∈ S : wrong at s }| − |{ i ∈ S : right at s }| = 2·|{ i ∈ S : wrong at s }| − k
```

Every selected item is costed at every seed. **`GATE-SELFTEST`** asserts
`net_s == n · Δacc_s` exactly, for every seed × dataset × operating point × `τ`.
Primary `net` = mean over three seeds; per-seed minimum also reported. Exchange rate is
a diagnostic and **reads no decision rule** (`banned_constraints[10]`).

**Item score.** For each item, the mean of its three seed-rows' OOF predicted
probabilities from **the `τ`-matched `FULL` logistic fit** (R3 I-11b) — there is one
per `τ`. Deterministic; no RNG.

**Out-of-support declaration.** `D-FELDMAN` is fit on
`P_τ ∪ CONFIG-MATCHED-CORRECT`, so unstable errors are **out of the ranker's support**
while `NET` ranks all `n`. Declared, not repaired: fitting on a three-class target
would change the object `D-FELDMAN` measures. The **composition of `S` by class** —
stable inversion / unstable error / always correct — is reported at every operating
point.

**Frozen operating points.** `k ∈ {|P_τ|, round(1.5·|P_τ|), round(2·|P_τ|)}`, top-`k`
by item score over all `n`.

**Closed-form reachability caps, pre-declared (R4 H-3).** Since
`net_s = 2·|S ∩ wrong_s| − k ≤ 2·|wrong_s| − k` and `net_s ≤ k`, a cell can clear
`K-NET` only if

> **`bar ≤ k ≤ 2·mean_s|wrong_s| − bar`.**

The banked floors fix `mean_s |wrong_s|` exactly: HateMM `84.33` (per-seed errors
`83 / 85 / 85`, from `744 × (1 − 0.8884 / 0.8858 / 0.8858)`) and MHC-ZH `62.33`
(`62 / 64 / 61`, from `579 × (1 − 0.8929 / 0.8895 / 0.8946)`). So under the binding
`37.2 / 29.0` bar, **`K-NET` is arithmetically unreachable for `k > 131.5` (HateMM) or
`k > 95.7` (MHC-ZH), and for `k < 37.2 / 29.0`, whatever the selector does.** On the
transferred F88 rates (`|P_0| ≈ 76 / 55`, `|P_{τ_hi}| ≈ 38 / 28`) that means:
`k = 2|P_0| ≈ 152 / 110` is **dead on both datasets at `τ_0`**; `k = 1.5|P_0| ≈ 114 / 83`
survives only at recall `≥ 0.89 / 0.88` *and* precision `≥ 0.66 / 0.68` together; and
`k = |P_{τ_hi}|` is **strictly impossible on MHC-ZH whenever `|P_{τ_hi}| < 29`**. The
run computes the caps from the realised `|wrong_s|` and `|P_τ|`, marks each of the six
`(τ, k)` cells `LIVE` or `ARITHMETICALLY_DEAD`, and **the KILL record must name the live
cells rather than say "at every `k`"** — otherwise a KILL would read as three tested
operating points when one or two could never have passed.

**The currency, re-adjudicated (R3 H-6).** Three surfaces name a figure:

1. `unified_pilot_gate.stage_0_reachability` ties the net requirement to the `+0.030`
   final bar; `banned_constraints[10]` supplies **`22.3 / 17.4`** = `0.030 × 744 / 579`
   on the **train arena**.
2. **C09's own authorising registry entry**, `gate0_reopen_2026_07_31.dispositions.promoted.bar`,
   verbatim: *"net-items currency **37.2/29.0/27.5** (HateMM/MHC-ZH/MHC-EN) scaled from
   `banned_constraints[10]`'s 22.3/17.4/16.5 for +0.030"*, with
   `strategic_finding.consequence_for_gate_0` repeating *"NET ITEMS against
   22.3/17.4/16.5 for +0.030, **i.e. 37.2/29.0/27.5 for +0.050**"*.
3. C02's own A0 ran `net_fix_rate: 0.03`.

**Binding rule: (2).** v3 bound `K-NET` to `22.3 / 17.4` and justified it with an
argument about `O1` (*"for `O1`, which breaks nothing, `net ≡ n · Δacc`"*) — true, but
`K-NET` is applied to `S`, where breaks are real and a `37.2` screen is fully
independent of the reach screen. The justification did not reach the rule it governed.
v4 therefore adopted, and v5 retains, the figure **C09's own authorising entry names**, which is also the
conservative choice against the campaign's stated failure mode (a false CONTINUE on a
large oracle):

> **`K-NET` binds at `net ≥ 37.2` (HateMM) / `≥ 29.0` (MHC-ZH) and
> `mean_s ΔmF1_s ≥ +0.050`** — the `+0.050`-sized pair, matching the Stage-0 reach bar's
> own two legs.
>
> **The `+0.030`-sized pair (`22.3 / 17.4`, `ΔmF1 ≥ +0.030`) is computed and reported
> at every operating point as a declared secondary.** If a cell clears the secondary
> but not the primary, the verdict is still a **KILL**, and the KILL record must state
> in terms: *"cleared the `+0.030`-sized net figure from `banned_constraints[10]` but
> not the `+0.050`-sized figure the C09 registry entry names."* That leaves the softer
> reading visible on the record for any future ruling without letting it create a
> CONTINUE.

At `k = 80` the two bars differ by **9.3 points** of required precision
(`π* = (1 + 37.2/80)/2 = 0.7325` vs `(1 + 22.3/80)/2 = 0.6394`), which is why the
choice is adjudicated rather than inherited. *(v4 said "~7.5 points", inherited from
the round-3 finding without re-derivation; 7.5 points is the gap at `k = 100`. R4 I-1.)*

### 5.4 What is *not* an inferential quantity

`stage_0_reachability` is a **threshold** rule; the CI requirement first appears at
`stage_1_signal`. `O1` and `NET` are adjudicated on **point estimates against frozen
thresholds**; no test is performed there, so there is no multiplicity to correct.
`K-FELDMAN` is the only rule that tests, and §5.2 specifies its two nulls, families and
correction exactly. Forking paths are controlled structurally: the `2 τ × 3 k` grid is
frozen, exhaustive and reported in full **with each cell marked `LIVE` or
`ARITHMETICALLY_DEAD` by §5.3's closed-form caps**; a CONTINUE requires the **same
`(τ, k)` cell to clear on both datasets simultaneously**; the CONTINUE names its cell;
§10 scopes the verdict to it.

### 5.5 `DATA-DEFECT-OVERLAP` — the third hypothesis, priced

A positive `ΔAUC` is consistent with a third explanation: **clustered annotation or
collection noise**, a data defect no encoder operator can fix. Both flagged populations
are constructed from the `text` field of `data/gt/*/train.jsonl` alone — **the flag
construction reads no label** (the enrichment statistic it feeds does read the
gold-defined `P_τ`, which is the quantity being characterised; R3 I-11f):

- **MHC-ZH `<em class="keyword">` markup.** The reopen records markup-bearing rows
  hating at `5×` the no-markup rate — train hate rate `0.5802` (141/243) with it vs
  `0.1161` (39/336) without, against a `0.3109` base — *"so part of the reported ZH
  0.8537 floor rests on how the corpus was harvested rather than on video content."*
  **Re-measured this session: 243 of 579 MHC-ZH train rows contain `<em`.**
- **HateMM whitespace-only transcripts.** **Re-measured this session: 39 of 744 HateMM
  train rows have a whitespace-only `text` field** (0 on MHC-ZH); independently
  corroborated by C02's `VIEW_SUPPORT.degenerate_causes.EMPTY_TEXT = 39`.

**Reported:** the enrichment of `P_τ` and of `S` in each flagged sub-population against
the arena base rate. **Stated honestly:** *no quantity in this A0 separates
H-DATA-DEFECT from H-TOPOLOGY.* High enrichment would make H-DATA-DEFECT the leading
explanation of any positive `ΔAUC` and would be reported in the verdict's scope. Gates
nothing.

### 5.6 CAL-4 declaration

**Closed-form:** every deployed-vote quantity, every count, `O1`, `net`, `Δacc`,
`ΔmF1`, the strata, every feature, the degeneracy agreements, `GATE-FIXK20`.
**Trained:** the head mint (30-epoch Adam, DET-3 Tier B, 4-dp parity asserted by
`GATE-FLOOR`), the logistic estimator (4-dp invariant across the thread grid), the GBM
capacity check (no verdict reads it).

---

## 6. The discriminator

### 6.1 Hypotheses, prior, bands

The **numerical** leg of the Feldman objection is retracted in-repo —
`HEADCOV_PREGATE_RECORD.md:305-310` withdraws *"the Feldman flourish"* because the
deployed heads sit at 0.82–0.94, not 0.998 — while its **substantive** leg stands
verbatim: *"memorising a long-tail singleton does not transfer to an unseen member of
the same one-member sub-population."*

| | **H-TOPOLOGY** (C09) | **H-MEMORISATION** (Feldman) | **H-DATA-DEFECT** |
|---|---|---|---|
| what the stable inversions are | a **region** with a shared geometric signature beyond atypicality | **singletons**: each wrong for its own reason | items whose **labels** are wrong or shortcut-driven, clustered by collection artefact |
| unconditional AUC | high | **also high** | also high |
| **`ΔAUC`** | **> 0** | **≈ 0** | **> 0** — indistinguishable from H-TOPOLOGY here |
| `NET` at the frozen points | clears `37.2 / 29.0` | ≈ 0 or negative | may clear |
| what an operator could do | act uniformly on the region | nothing | nothing — the fix is annotation, not geometry |
| **separated by this A0?** | — | yes, by `ΔAUC` | **NO** — only *priced*, by `DATA-DEFECT-OVERLAP` |

**The registered prior.** F97's `ban_scope`, verbatim:

> *"HONEST POSITIVE DATUM, RECORDED AND EXPLICITLY NOT PROMOTED: F47-features-as-
> adjudication-gate is REAL and permutation-validated — +0.0269 on HateMM (p=0.0050,
> fold signs +++++), +0.0104 on ZH (p=0.0050), +0.0182 on EN (p=0.0100) — a genuine
> refinement of F47's epitaph (dead as a per-item CHANNEL SELECTOR, not dead as a
> per-item ADJUDICATION GATE). It is nonetheless SUB-BAR on all three …"*

and F98 banks it as a ceiling: *"the F47 features have a measured, already-banked
ceiling of +0.0269."* **So the registered prior for this exact feature family is
`identifiability REAL, conversion SUB-BAR` — band B — with the conversion ceiling
`+0.0269 / +0.0104` far below `K-NET`'s `+0.050`-sized bar on both C09 datasets.** Two
facts sharpen it against C09. First, those are **raw-arena** numbers, and F113 measured
that **9 of 9 raw arms that scored positive in the raw arena fail to transfer to head
space** — `HEADSPACE_TRANSFER_PREGATE.md:858-863` enumerates them across four distinct
operator families (`VSW_pow`, `VSW_exp`, `VSW_lin`, `THRESH_best`, `CTRL_cos_pow`,
`FIXK_15`, `FIXK_10`, F95 `mlp_mean3`, and the λ-oracle ceiling), concluding *"9 of 9
raw positives fail to transfer; the median shrink is >7×; three invert sign."*
**Separately**, F113 records at `:917-919` that as a *campaign-level* result only one of
them was ever a raw-space positive — *"**Established on ONE cell only:** the failure of
a raw-space **positive** to transfer … **F105/VSW is the only raw-space positive the
campaign ever produced.** There is no second one to test. So 'one cell' is the
population, not a sample of it."* v4 conflated the two, describing all nine as
descending from F105/VSW, which they do not (R4 I-2). Second, the closest measured
analogue inside this very arena runs the same way — F113:
*"any FITTED relation score over head keys memorises the bank (in-sample pair AUC
0.9999) and is WORSE than the plain cosine on held-out pairs (d_AUC +0.1572/+0.2302 raw
→ −0.0643/−0.1294 head, 30/30 fold cells)"*.

> **Band B is this design's pre-declared expectation, not its surprise.**

**The bands, as functions of the same rules §9 uses, and exhaustive over completed
runs (R3 I-1):**

> **Band A′ — `K-REACH` fails at every `τ`.** The population is too small regardless of
> identifiability or conversion. **KILL**, and at `τ_0` closing every `τ ≥ 0` by
> arithmetic.
>
> **Band A — `K-REACH` clears at some `τ` but `K-FELDMAN` fires at every such `τ`.**
> No incremental structure survives conditioning. **KILL.** The additional statement
> *"H-MEMORISATION is consistent with this object"* may be published **only** if
> `K-FELDMAN` fired on both datasets at both `τ` **and** `IDENTIFIABILITY_UNDERPOWERED`
> is absent on both datasets (§5.2).
>
> **Band B — some `τ` at which `K-REACH` and `K-FELDMAN` both clear, but `K-NET` or
> `K-DEG` fires there at every `k`.** Identifiability without legal conversion.
> **KILL, under the F98 epitaph and the `+0.0269` ceiling. Explicitly NOT a
> confirmation of H-MEMORISATION** — it is a conversion failure, and the record must
> say so.
>
> **Band C — some `τ` and some `k` at which `K-REACH`, `K-FELDMAN`, `K-NET` and
> `K-DEG` all clear on both datasets.** ⇒ `CONTINUE` (§9), scoped to that `(τ, k)`
> cell and subject to §11's precondition.

**Upper-bound caveat.** `AUC_strat` conditions on a label-blind stratum, so it is not a
gold-conditioned upper bound — but a deployed operator would have to locate the region
without knowing which items are in `P`. `ΔAUC > 0` is **necessary, not sufficient**.

### 6.2 `K-DEG` — the threshold-degeneracy control

F96 makes this a **standing gate**, verbatim: *"ANY VARIANT MUST FIRST PASS THE
DEGENERACY CONTROL, and that is now a standing gate, not a suggestion … Any operator
agreeing with a pure global threshold shift on the bulk of items is DEAD BY THE
EXISTING THRESHOLD BAN regardless of its Dacc — it is a dead lever in an item-level
costume."* F98's DEG-A measured `0.9570` (HateMM) / `0.9508` (EN) against a frozen
`0.95` kill line, with bare `THRESH_best` scoring `+0.0188` — more than the learned
operator; F98's `ban_scope` (d) closes re-deriving that observation; and **C09's own
dedup boundary excludes *"thresholding"***. `BASE` is led by `|score|`, so `S` can be a
threshold move in costume.

**The statistic the `0.95` line was calibrated on — corrected (R4 C-1).** F96's number
is a fraction of **items**: *"C1 agrees with a PURE GLOBAL THRESHOLD SHIFT … on
95.03% / 97.75% / 99.45% of items"*. F98's DEG-A is `(c3 == coll["THRESH_best"]).mean()`
— **prediction agreement over all `n` items** — giving `0.9570 / 0.9508`, and DEG-B is
the same construction. v4 applied the `0.95` line to **selected-set overlap**
`|S ∩ twin| / k` instead. The two are related by

> `pred_agree = 1 − |S △ twin| / n = 1 − 2k(1 − ov)/n` (since `|S| = |twin| = k`),

so at HateMM's `n = 744` with `k ≈ 76`, `pred_agree = 1 − 0.204(1 − ov)`: F98's `0.95`
line corresponds to `ov ≈ 0.755`, while v4's `ov ≥ 0.95` demanded
`pred_agree ≥ 0.9898`. An `S` three-quarters identical to a bare threshold shift — the
exact *"dead lever in an item-level costume"* — would have passed. **`K-DEG` therefore
fires on `pred_agree`, the object F96/F98 measured**, and `|S ∩ twin| / k` is reported
beside it as a descriptive figure that no rule reads.

**Three frozen degenerate twins, each of size exactly `k`, computed on BOTH scales.**
`S` is per-item, so a per-seed twin would depress agreement for reasons unrelated to
degeneracy; but a per-seed twin is what F96/F98 measured. Both are computed and
**`K-DEG` reads the maximum `pred_agree` over the two scales**, the conservative
direction for a gate whose job is to fire.

- **`THRESH-SYM`** — the `k` items with the smallest `c_i` (per-item scale) / smallest
  `|score_{i,s}|` (per-seed scale, then averaged over seeds).
- **`THRESH-BEST`** — the hindsight-best one-sided band of size `k`: of {`k` items with
  the smallest positive signed score}, {`k` items with the largest negative signed
  score}, the one with the higher `net_s`; on the per-item scale the signed score is
  `mean_s score_{i,s}`. Hindsight-optimal by construction, so the control is
  conservative.
- **`FIXK`** — the items whose deployed decision flips under an alternative fixed
  neighbourhood size `k′ ∈ {1, 2, 3, 5, 7, 10, 15}` in the same rank-weighted vote.
  **`Δscore_i(k′) ≡ score_i(k′) − score_i(20)`**, the change in the rank-weighted vote
  of `mechfix_ops.py:94` when the neighbourhood is truncated to `k′` with weights
  `[k′ … 1]` (R4 I-4a — v4 used `|Δscore|` three times without defining it).
  **`best` is pinned as the `k′` maximising `|S ∩ flip(k′)| / k`** (F98's DEG-B
  convention, and the conservative one), **ties broken by the smallest `k′`**
  (R4 I-4b), **reported per cell together with `|flip(k′)|` for all seven `k′`** so a
  reader can see when the twin is padding rather than flipping — the banked degeneracy
  block shows `flip(k′)` can be empty in this space (`B_agree_fixk["15"] = 1.0000` with
  `agree_deployed = 1.0000` on ZH seed 0, i.e. `FIXK_15 ≡ deployed`). **Size contract:**
  if `|flip(k′)| > k`, keep the `k` with the largest `|Δscore_i(k′)|`; if
  `|flip(k′)| < k`, pad with the next-largest `|Δscore_i(k′)|` items so the twin is
  always exactly `k`. **If `|flip(k′)| = 0` for every `k′`, the `FIXK` twin is emitted
  as `DEGENERATE_ALL_EMPTY` and is not read by `K-DEG`**, because a fully padded twin is
  a threshold construct wearing the `FIXK` label.

**`K-DEG` fires — and the verdict is KILL — if the maximum `pred_agree` over the two
scales is `≥ 0.95` with any read twin on either dataset at the cell under
consideration.** The `0.95` line is the campaign's own, now applied to the campaign's
own statistic. Anchor: F113 measured head-space `THRESH_best` at `+0.0041` (from raw
`+0.0148`), so the degenerate lever is worth ~1 item per 244 in this space.

### 6.3 Controls

- **`SHUFFLE-POP`.** A **uniform random permutation of the per-item target vector over
  the `D-FELDMAN` analysis set** — `P_τ ∪ CONFIG-MATCHED-CORRECT`, exactly the rows the
  estimator sees, not all `n` — `default_rng(20260801)`, `200` draws, applied
  identically to `FULL` and `BASE` on the same folds and the same frozen strata,
  **evaluated at both `τ_0` and `τ_hi`**. v2's claim that the permutation *"preserves
  all configuration marginals"* was false and is withdrawn.
  **What it tests:** that the split machinery, the stratified-AUC estimator and the
  bootstrap do not manufacture signal from the target's marginal alone.
  **What it cannot test:** feature-side leakage (that is `GATE-BLIND`'s job), and
  stratum-occupancy failure (that is `STRATUM_OCCUPANCY`'s job, §5.2).
  **HALT rule:** the permutation-null mean of `AUC_strat(FULL)` must lie in
  `[0.45, 0.55]` at **both** `τ` on **both** datasets; outside that band the estimator
  is leaking and **no verdict is published**.
  Its `ΔAUC` draws are additionally reported as a **secondary joint-null ASL** (§5.2).
- **`UNSTABLE-POP`.** `D-FELDMAN` re-run with the target redefined as unstable errors,
  negatives unchanged. **Data-independent power rule applied identically to both
  datasets:** emit `CONTROL_UNDERPOWERED` iff `n_unstable < 20` **or** the two-sided
  bootstrap CI width on `ΔAUC` exceeds `0.30`. Non-gating.
  **Pre-declared expectation (R3 I-9):** the banked floors give per-seed error counts
  of `83–85` (HateMM, from `1 − 0.8884/0.8858/0.8858` on `n = 744`) and `61–64` (ZH,
  from `1 − 0.8929/0.8895/0.8946` on `n = 579`); on the transferred F88 stability rates
  that implies `n_unstable ≈ 7–9` on both datasets, far under the `20` trigger. **So
  `CONTROL_UNDERPOWERED` is the expected outcome on both datasets. That is declared
  now, is not an artefact, and means the registry claim's own stability premise is NOT
  tested by this A0.**
- **`RANDOM-POP`.** A size-matched random sample of query items in place of the stable
  inversions, `default_rng(20260801)`, 200 draws; every reported quantity recomputed
  against it. This prices **F88 null (3)**: HateMM memory-bank LOO curation at
  `+0.0016` against random deletion of the same size at `+0.0031 / +0.0000`,
  self-labelled *"Pregate-grade null (one rule, one proxy head/cell, single draw)"*.
  **At its adjudicated weight:** the reopen's round 14 records this as *"a val-sel loss
  and a final-epoch win, all under half a test item per seed, so 'indistinguishable' is
  the exact reading"* — the curated rule is **indistinguishable from** random on a
  single draw, HateMM-only, on **train-row deletion**, a different population and
  operator from C09's. A headwind to price, not a closure. Non-gating.

---

## 7. Controls summary

`RANDOM-POP`, `CONFIG-MATCHED-CORRECT`, `SHUFFLE-POP`, `UNSTABLE-POP`, `PERM-STRUCT`,
`PERM-STRUCT-COND` (§5.2, §6.3); `K-DEG`'s three twins on two scales (§6.2);
`STRATUM_OCCUPANCY` (§5.2); `DATA-DEFECT-OVERLAP` (§5.5); `FEATURE_DEGENERACY` (§5.2).
`CONFIG-MATCHED-CORRECT` additionally supplies the break-exposure stratification, so
*"constraining break exposure"* is measured rather than asserted.

---

## 8. Gates

### 8.1 HALT gates — a failure publishes **no** verdict

- **`GATE-FLOOR`.** The re-minted fold-head arena must reproduce the banked per-seed
  pooled deployed values **equal at 4 decimal places**, on all **6 (dataset, seed)
  cells** (R4 I-8e: "6/6 seeds" was loose — there are three seeds per dataset), in
  **both** metrics:

  | | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | HateMM acc | `0.8884` | `0.8858` | `0.8858` |
  | HateMM macro-F1 | `0.8838` | `0.8811` | `0.8812` |
  | MHC-ZH acc | `0.8929` | `0.8895` | `0.8946` |
  | MHC-ZH macro-F1 | `0.8747` | `0.8710` | `0.8765` |

  Source: `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`, `result.acc_deployed` /
  `result.mF1_deployed`, re-read at drafting time. **4 dp and not beyond** — the banked
  anchors are 4-dp values. DET-3 Tier B entitles this run to gate against banked JSON
  because DET-1/DET-2 are honoured and the banked outputs carry their own runtime
  block. **Feasibility is demonstrated, not assumed:** C02's independent re-mint
  (`C02_A0_OUT.json`, `ARENA2.pooled_native_acc`) reproduced these values exactly.
  **Version/node residual:** the banked `meta.runtime` is compared against this run's;
  any difference in `python / numpy / scipy / sklearn / torch` (five members — R3
  I-11d) or the node is reported as `RUNTIME_DRIFT` with the re-run path named.
- **`GATE-PARITY-FOLD`.** The re-minted deployed vote must reproduce the banked
  per-fold `result.fold_acc_deployed` arrays, **equal at 4 decimal places**, for all 30
  fold-cells.
- **`GATE-FIXK20`** *(CAL-2 leg (1); named `GATE-FIXK20` everywhere — R4 I-8d)*. In the fold-head arena, the
  fixed-`k` rank-weighted vote at `k′ = 20` **must change 0 items and give
  `Δacc = 0.0000` exactly**, on every dataset × seed × fold — the arena's `k = 20` rule
  *is* the deployed rule. Miss ⇒ **harness VOID**. This is the only gate that checks
  the fixed-`k` vote path, which `K-DEG` reads and which can KILL.
- **`GATE-BLIND`.** A per-feature manifest naming, for every feature in `BASE` and
  `FULL`, the exact arrays it indexes. Enforced structurally: the feature builder's
  signature admits neither the query-side gold-label array nor any target-derived
  array, and **three** arrays are wrapped in read-counting guards for the whole
  feature-construction phase — `lab_query`, `is_inversion[seed]`,
  `is_stable_inversion`. **Emitted as integer counts:** all three must be exactly `0`;
  `bank_label_reads` is reported as its nonzero legal integer; per-feature array-touch
  lists emitted in full. Any nonzero count on the three guarded arrays **HALTs**.
- **`GATE-LEDGER`.** Literal integer counts: test-split path opens (`0`), test-label
  materialisations (`0`), dev-split **path** opens (expected nonzero — 36 mint loads of
  `dev_seen_*.pt` plus 6 banked-trainlog reads — reported with its declared expected
  value), dev-label materialisations **outside** any decision quantity (expected `36`,
  one per mint, §3 `H-L3`), and dev or test **label materialisations into any decision
  quantity** (`0`, the binding `H-L3` predicate). `headspace_mint.py:106-116` installs a global `torch.load` guard raising
  on any path containing `test_seen` or `/test`; the driver adds an `open()`-level
  guard with the same predicate over the whole job.
- **`GATE-NESTED`.** The `D-FELDMAN` partition is asserted equal to the frozen arena
  fold partition; for every scored item, "this item's arena fold was excluded from the
  model that scored it, all three of its seed-rows were excluded together, **and the
  `τ_hi` threshold applied to it was computed with its fold excluded**" is checked and
  emitted as a **per-item check count** that must equal the item count.
- **`GATE-SELFTEST`.** `net_s == n · Δacc_s` exactly, every seed × dataset × operating
  point × `τ`.
- **`GATE-ZEROOP`.** With `S = ∅` (`k = 0`): `Δacc = 0.0000`, `ΔmF1 = 0.0000`,
  `net_s = 0` for every seed and dataset. Checks the *treatment* path, which
  `GATE-FLOOR` does not.
- **`GATE-ARENA`.** Pooled native accuracy must satisfy
  `majority_rate + 0.02 ≤ acc ≤ 0.98` on both datasets (C02 `ARENA2` convention):
  HateMM majority `0.5995` ⇒ band `[0.6195, 0.98]`; MHC-ZH majority `0.6891` ⇒
  `[0.7091, 0.98]`.

**Nine HALT gates**, and exactly one further HALT condition elsewhere: §6.3's
`SHUFFLE-POP` band. **§5.2's all-single-class `AUC_strat` case is NO LONGER a HALT** — it
is a data outcome that drops the cell and forces `IDENTIFIABILITY_UNDERPOWERED` (R4 H-4).
These ten conditions are the complete publication precondition of §9.

### 8.2 Reporting and scoping instruments — these do **not** gate the verdict

- **`GATE-DEVFID`.** `headspace_fidelity.py` run unmodified on the six `fold == -1`
  heads. Banked reference: `B_fid_abs_3seedmean` `0.0093` (HateMM) / `0.0086` (MHC-ZH),
  `STOP_RULE_TRIGGERED: false` on both
  (`headspace_fidelity{,_zh}_OUT.json`, re-read at drafting time). **Reported, not a
  HALT.** It measures proxy↔floor fidelity **across hardware**; C09's entire arena is
  CPU-minted, so F88's binding caveat is satisfied by construction and cross-hardware
  fidelity does not gate the internal comparison. `STOP_RULE_TRIGGERED == true` on
  either dataset publishes the verdict with a `PROXY_FIDELITY_FLAG`.
- **`GATE-SEED`.** The per-seed inversion sets are emitted in full, as sorted item
  indices, so the 3-seed intersection is independently recomputable from the published
  artifact. An emission, not a predicate.
- **`GATE-NULL`.** HateMM train row `355` (`hate_video_95`, label `1`) carries an
  exact-zero vector in **both** streams; MHC-ZH has **no** structural-zero row.
  *(Re-measured this session from `data/CLIP_Embedding/{HateMM,MHC_zh}/train_*.pt`:
  HateMM zero-img `[355]`, zero-txt `[355]`; MHC-ZH `[]` and `[]`.)* Contract:
  1. the **primary** run is **with-null**, on the full `n = 744`, the arena the banked
     floors and the `37.2` figure are defined on;
  2. a **remove-null sensitivity** drops item 355 from the query set **and** every
     bank, with its own recomputed floors and its own recomputed bar
     `0.050 × 743 = 37.15`;
  3. the requirement is agreement on the **verdict and every K-rule outcome**, not on
     metric values — `n` moves and every rate changes. A disagreement on one item out
     of 744 is published as a first-class finding and the verdict is scoped to it;
  4. **In head space the zero row is not zero.** `img_proj`/`text_proj` are
     `nn.Linear` with bias and `mlp[:-2]` applies no final normalisation, so the
     C01/C02 raw-space contract *"must remain exact-zero in every derived array"* does
     **not** transfer and is not asserted. What is asserted is that item 355 is treated
     identically to every other item by every code path, and its per-item fate is
     reported explicitly.

---

## 9. Decision rule — two-valued **on completed runs**

**Publication precondition.** *The run publishes a verdict only if* **all nine HALT
gates of §8.1 pass** *and the* `SHUFFLE-POP` *band holds at both* `τ` *on both
datasets.* **A HALT publishes no verdict**: it is an engineering result with a
diagnose-repair-resubmit path, it consumes no scientific gate, and it is **evidence
neither for nor against C09**. v3 made gate passage a CONTINUE condition, which would
have published a **KILL** on a detected leak, a thread mis-export or a version drift.
v4 fixed that but still routed §5.2's all-single-class stratum case here; **that case is
no longer a HALT** (R4 H-4) — it is a data outcome that drops the cell, forces
`IDENTIFIABILITY_UNDERPOWERED`, and lets the run publish a KILL scoped *"not
identifiable at this power"*. Every remaining HALT condition is an instrument failure
with a real repair path.

**Conditional on a completed run**, let `τ ∈ {τ_0, τ_hi}` and
`k ∈ {|P_τ|, round(1.5|P_τ|), round(2|P_τ|)}`.

**`CONTINUE`** iff there exists a `τ` such that **1–2 hold and there exists a `k` for
which 3–4 both hold at that same `(τ, k)`** (R3 I-11a):

1. **`K-REACH` clears at `τ`** — `Δacc_{O1} ≥ +0.050` **and** `ΔmF1_{O1} ≥ +0.050` on
   **both** datasets.
2. **`K-FELDMAN` clears at `τ`** — on **both** datasets, **both** the marginal null
   `PERM-STRUCT` and the conditional null `PERM-STRUCT-COND` reject at `α = 0.05` after
   Holm over the two `τ` within each dataset and family (IUT across datasets, §5.2),
   **and** `IDENTIFIABILITY_UNDERPOWERED` is absent on both datasets (§5.2).
3. **`K-NET` clears at `(τ, k)`** — on **both** datasets simultaneously,
   `mean_s net_s ≥ 37.2` (HateMM) / `≥ 29.0` (MHC-ZH) **and**
   `mean_s ΔmF1_s ≥ +0.050`.
4. **`K-DEG` does not fire at `(τ, k)`** — the maximum **prediction-vector agreement**
   `1 − |S △ twin| / n` of `S` over both scales with each read twin (`THRESH-SYM`,
   `THRESH-BEST`, and `FIXK` unless emitted `DEGENERATE_ALL_EMPTY`) is `< 0.95` on
   **both** datasets (§6.2).

   *(Cells the §5.3 caps mark `ARITHMETICALLY_DEAD` cannot satisfy 3 and are reported as
   such rather than as tested and failed.)*

**`KILL`** in every other completed case. Conditional on a completed run, `KILL` and
`CONTINUE` are complements.

**Every quantity is computed and reported at both `τ` and all three `k`, on both
datasets, regardless of which rule fires.**

**Which rule fired is recorded, and the KILL is scoped by it:**

- **`K-REACH` fired at `τ_0`** ⇒ closes **every** `τ ≥ 0` on both metrics, by
  arithmetic (§4.3).
- **`K-REACH` cleared at `τ_0` but failed at `τ_hi`** ⇒ the co-primary is closed **on
  reach alone**: the registry's own *"high-confidence"* restriction, taken at the
  out-of-fold median, does not contain enough items to reach the Stage-0 bar, and
  `q_max` quantifies where it stops. **This closes nothing about identifiability or
  conversion at `τ_hi`** — both are still measured and reported there, and must not be
  read as adjudicated.
- **`K-FELDMAN` fired** ⇒ closes `τ ∈ {τ_0, τ_hi}` only, and — if
  `IDENTIFIABILITY_UNDERPOWERED` is present on either dataset — the closure reads *"not
  identifiable at this power"*, not *"not identifiable"* (§5.2).
- **`K-NET` or `K-DEG` fired** ⇒ closes `τ ∈ {τ_0, τ_hi}` only, because neither
  precision nor AUC is monotone in `τ`. If a cell cleared the `+0.030`-sized secondary
  but not the binding `+0.050`-sized pair, the record must say so in terms (§5.3).

**A `CONTINUE` is tagged with, and scoped to:** its `(τ, k)`; `PROXY_FIDELITY_FLAG` if
`GATE-DEVFID`'s stop rule fired; the `DATA-DEFECT-OVERLAP` enrichment; and **`ROBUST` or
`POINT_ESTIMATE_ONLY`, whose predicate is frozen here (R4 I-5): `ROBUST` iff the
one-sided 95 % item-bootstrap lower bound on `ΔAUC` is `> 0` on both datasets at the
CONTINUE's `τ`, else `POINT_ESTIMATE_ONLY`.** That interval is **conditional on the
fitted model** (§5.2) and is a robustness label only; it neither creates nor blocks a
verdict. `IDENTIFIABILITY_UNDERPOWERED` cannot appear on a CONTINUE, because §5.2 makes
`K-FELDMAN` fail on a tagged dataset.

**The raw arena, specified — and what it can and cannot compute (R4 I-7).** The battery
is recomputed on the banked **raw fused key space** — `X = l2n(concat(l2n(img_feats),
l2n(text_feats)))`, 7168-d, seed-free (`headspace_arena.py:7`,
`mechnov_pairverify.py:124`), over the *same* frozen 5-fold assignment and the same
deployed top-20 vote — whose banked pooled deployed accuracies are `0.8441` (HateMM) and
`0.8480` (MHC-ZH) (`headspace_arena_*_OUT.json`, `membership.raw_deployed_acc`).
**Because the raw space is seed-free, three things change and are declared:** (i)
**stability is undefined**, so the raw leg prices the *single-pass inversion* population,
one row per item, not the 3-seed intersection; (ii) `AUC_strat` has **one cell per
dataset**, not three, so there is no seed-mean; (iii) `K-DEG`'s two-scale maximum
**collapses to one scale**, since `c_i` and `|score_{i,s}|` coincide. `O1`, `NET`,
`ΔAUC`, both permutation nulls, the item bootstrap and the twins are all computable
there; `UNSTABLE-POP` is not. The raw leg is **confined to corroborating a KILL** — the
only direction F113 permits. F113's caveat, quoted from its own primary
(`HEADSPACE_TRANSFER_PREGATE.md:920-923`) with all three clauses (R3 I-3):

> *"**NOT established:** that a raw-space **negative** cannot be a head-space positive
> (§8.1); that any of this transfers to the **test** split (all arenas here query
> train-split items held out from their own head, which is closer to deployment than
> raw but is still not deployment); that the CPU-minted proxy head equals the CUDA
> floor to better than **±0.0093** (3-seed) / **±0.0280** (single seed)."*

The third clause bears directly on `GATE-DEVFID`'s own `0.0093 / 0.0086` and is why
that gate reports rather than gates. **No raw-arena number reaches the decision.**

**CAL-3, discharged (R3 I-10, premise corrected).** CAL-3 is mandatory whenever a raw
`Δ ≥ +0.010` is reported and requires the raw `Δ` beside *"the deployed space's own
gold-cheating ceiling for the same operator family"*, labelling the arm
`RAW-ARENA ARTEFACT` if the raw legal number exceeds the deployed oracle. The raw leg
here will report `Δ` above `+0.010`. **The nearest banked deployed-space analogue is
`ERRPAT_HateMM §7` ("SOLUTION MAPPING PER CLUSTER"), which banks per-cluster
flip-every-member ceilings on the deployed proxy head: FN1 `+0.0326` (n=7), FN2
`+0.0140` (n=3), FN3 `+0.0233` (n=5), FP1 `+0.0233` (n=5), FP2 `+0.0233` (n=5), FP3
`+0.0093` (n=2).** These are deployed-space upper bounds for flipping a nominated error
sub-population — `O1`'s object, on the test split — and are reported beside the raw
leg. **They are a different arena and a different population** (test split, `n = 215`;
C09's arena is the train split at `n = 744 / 579`), so they are a **comparator of
record, not a strict CAL-3 ceiling**, and v3's claim that no analogue is banked is
withdrawn. **No arm is escalated in any direction**; the raw leg's only permitted use
remains KILL corroboration. F113's head-space `THRESH_best` `+0.0041` is reported as a
further anchor.

---

## 10. Scope of any verdict

- A **KILL** closes the C09 Stage-0 oracle **under the frozen Stage-0 rule, at the `τ`
  values §9 scopes it to**. It is **not** an impossibility proof for encoder-level
  topology intervention: the probe is one feature set, one estimator family, one
  stratification and one power regime. Stated **now**, because C02's A0 had to retract
  exactly this kind of overclaim once (the v8 erratum) before it was re-stated
  correctly.
- A **CONTINUE** establishes only that the population is large enough and locatable
  enough to justify building an operator, scoped to its `(τ, k)` cell, and is void
  unless §11's precondition is met.
- **`O1` is a label-flip oracle over one nominated population, not the registry's
  "full-bank or representation-level oracle" — and it is the only upper bound this A0
  produces.**
- **`NET` prices one specific per-item selector.** Its operator class dominates §11's
  successor, but this particular instrument may be weaker than the successor, so **`NET`
  bounds the successor's conversion in neither direction** (§3.2(3)).
- **With MHC-EN out of scope the two-dataset requirement has zero slack.**
- **This A0 does not separate H-DATA-DEFECT from H-TOPOLOGY** (§5.5); it only prices
  the overlap. **And it does not test the registry claim's stability premise**, because
  `UNSTABLE-POP` is expected to be underpowered on both datasets (§6.3).
- **CAL-5 runs against C09**: the Stage-1 operator is a channel-(a)/(d) object, which
  *"carries NO transfer warrant."* Any result here is an **arena** result about the
  fold-head **train** arena, never a prediction of deployed behaviour.
- Neither verdict touches the `+0.030 / +0.030` two-dataset target, which remains active
  and unmet.

---

## 11. The Stage-1 seam

The reopen's **first** quoted kill-risk, verbatim and in full:

> *"(i) any encoder-level pull of an inversion toward its right analogue is a
> label-using metric move => F75/NCA and section 1.3's +0.0286"*

with the reopen's binding instruction attached: *"Section 1.3's bound must always be
quoted with R^2=0.027, r=+0.1642, slope CI [-0.0221,+0.1637] straddling zero, MHC-ZH
dev only - F114's standing instruction - and F114 forbids citing F107 as a theory-level
door-closer."* The `+0.0286` is carried **only** with those four qualifiers and is not
used as a bound anywhere in this design.

**The successor this A0 would license, named concretely.** A **global, symmetric,
train-label-supervised reshaping of the head map** `φ₀ → φ′`, in which the
stable-inversion set is identified **once, offline, on train items only**, and enters
the *training objective* as a region-targeted term. At inference `φ′` is applied
identically to every query; the stability statistic, the probe and region membership
are never consulted at query time. That is what makes it symmetric under
`LITSWEEP3_DATA_CENTRIC.md:82`, and it is the only shape §3's HALT boundaries leave
standing.

**Is that F75's object? Partly:**

- **F75's `ban_scope`, verbatim:** *"head-loss swaps of the triplet+BCE hybrid toward
  vote-consistent (NCA/soft-kNN), contrastive (SupCon), or mixup-BCE objectives at 7B
  frozen-encoder feature scale; tau/alpha retunes = tactics, banned."* A region-targeted
  term added to the deployed triplet+BCE hybrid is **none of the three named
  objectives**, so it is outside the ban's letter.
- **But F75's own `ban_scope` also records it as the *"First measured negative for
  trained-reshaping-unlocks-oracle-headroom; F66 selection-locked pools untouched."***
  (the reopen renders this in lower case as *"the first measured negative for
  trained-reshaping-unlocks-oracle-headroom"*, which is the reopen's quotation, not
  F75's own casing — R4 I-8f). **And `LITSWEEP5_COMPLETENESS.md §2` argues in its own
  adversarial self-rebuttal that F75's *mechanism* — symmetric reshaping does not
  convert selection-locked headroom — *"generalizes past its named-loss letter"*.** An
  argument, not a measurement, offered by LITSWEEP5 against its own challenge; it is the
  correct headwind and is registered.
- **C09's own dedup boundary forbids the cheap version:** *"not … hard-example weighting
  alone."* A region-weighted triplet+BCE **is** hard-example weighting, so the successor
  must be more than that or it fails its own registry boundary before it fails F75.
- **The counterweight, at its corrected one-sided-use weight.**
  `NCA_FORENSIC_RECON.md:110`: *"Ruling: F66 does NOT bind trained-space reshaping. The
  cell is not F66-dead — it is legitimately un-measured."* The same record's `:112`
  prices that cell at *"honest P(≥+3) stays 2-4%"*, and the cell it unblocked was
  subsequently run and killed as F75. Both halves carried.
- **And `NET` does not price it** (§3.2(3)): `NET` bounds the successor in neither
  direction, so **a `K-NET` pass is not evidence the successor converts**.

**Pre-registered consequence, binding on this A0's own verdict.** A `CONTINUE` **does
not carry a Stage-1 licence.** Stage-1 entry requires that a proponent name an operator
that is (a) global and symmetric at inference, (b) not one of F75's three named
objectives, (c) not hard-example weighting alone, and (d) accompanied by a fresh
ban-scope adjudication against F75, F66 and F98. **If no such operator can be named at
Stage-1 entry, the CONTINUE is void and C09 closes with no further spend.**

---

## 12. Repair ledger — the 14 round-4 findings

| # | round-4 finding | repair | where |
|---|---|---|---|
| **C-1** | `K-DEG` applied F96/F98's `0.95` constant to **selected-set overlap**, a statistic it was never calibrated on; at `k/n ≈ 76/744` that demanded `pred_agree ≥ 0.9898`, so an `S` three-quarters identical to a bare threshold shift would have passed the campaign's only degeneracy screen | **`K-DEG` now fires on prediction-vector agreement `1 − |S △ twin| / n`**, the object F96 (*"on 95.03% / 97.75% / 99.45% of items"*) and F98's DEG-A (`0.9570 / 0.9508`) actually measure; the mapping `pred_agree = 1 − 2k(1−ov)/n` is written out; `|S ∩ twin| / k` is retained as a descriptive figure no rule reads | §6.2, §9 |
| **H-1** | R3's `τ_hi` finding relocated rather than discharged: a fitting row `j ∈ fold g` carried target `c_j ≥ τ_hi^(g)`, and `τ_hi^(g)`'s median over `P_0 \ fold g` **contains** the scored item | **the population and the fitting target are separated**: `P_{τ_hi}` keeps `τ_hi^(fold(i))`; the model that scores fold `f` labels **every row in that fit** with `τ_hi^(f)`, so the scored item's label enters that fit nowhere; the two objects and the reason are stated; `GATE-NESTED` asserts the fitting-target form as a per-item check | §4.3, §8.1 |
| **H-2** | `PERM-STRUCT` permuted **rows** (the `√3` anti-conservatism of round-1 C-1, in the CONTINUE direction) and was called *"the exact null"* for a conditional hypothesis it is only marginally exact for | permutation moved to the **item** level within each `(dataset, seed)` cell, one donor per item across all three seed-rows; *"the exact null"* **withdrawn** and replaced by an explicit marginal-vs-conditional statement naming the dependence (structural feature 1 ≈ a deterministic function of `BASE`'s `pred_purity`); **`PERM-STRUCT-COND` added** — a residualised conditional null a CONTINUE must **also** clear, with a marginal-only rejection reported as `MARGINAL_ONLY_NOT_CONDITIONAL` and treated as a KILL | §5.2, §2 |
| **H-3** | the `37.2 / 29.0` currency makes some frozen `(τ, k)` cells arithmetically dead, undeclared | **closed-form caps pre-declared** — `bar ≤ k ≤ 2·mean_s|wrong_s| − bar`, with `mean_s|wrong_s|` derived exactly from the banked floors (`84.33` / `62.33`) giving `k ≤ 131.5 / 95.7`; each of the six cells is marked `LIVE` or `ARITHMETICALLY_DEAD` in-run, and **the KILL record must name the live cells** rather than say "at every `k`" | §5.3, §5.4, §9 |
| **H-4** | the all-single-class stratum case — a **data** outcome — was routed to a no-verdict HALT with no repair path | **demoted from HALT to data outcome**: the cell is dropped from the seed-mean with the drop emitted, `IDENTIFIABILITY_UNDERPOWERED` is forced, and the run publishes a KILL scoped *"not identifiable at this power"*; HALT status is reserved for the nine instrument gates plus `SHUFFLE-POP`'s band | §5.2, §8.1, §9 |
| **I-1** | *"~7.5 points"* contradicts the two `π*` values printed beside it | corrected to **9.3 points** with both `π*` re-derived in place, and the inherited error named | §5.3 |
| **I-2** | *"9 of 9 … all nine descending from F105/VSW"* conflates two different F113 statements | **split**: `HEADSPACE_TRANSFER_PREGATE.md:858-863`'s nine arms enumerated across four operator families with its own concluding sentence quoted, and `:917-919`'s *"Established on ONE cell only"* given as the separate campaign-level scoping | §6.1 |
| **I-3** | CAL-2 leg (2)'s comparator called a *"raw-arena"* k-curve | corrected to `PREGATE_CALIBRATION_CLAUSE.md:80-81`'s **deployed** k-curve, and the skip re-justified on two correct grounds: CAL-2's own header scopes it to the raw banked train-space arena, and `ksweep_OUT.json`'s per-`k` payload is a **test-split** read this run does not materialise | §2 |
| **I-4** | `FIXK`'s `Δscore` undefined; `best` had no tie-break; an all-empty `flip(k′)` would make the twin pure padding | **`Δscore_i(k′) ≡ score_i(k′) − score_i(20)`** defined against `mechfix_ops.py:94`; **ties broken by the smallest `k′`**; `|flip(k′)|` emitted for all seven `k′`; and an all-empty case emitted as **`DEGENERATE_ALL_EMPTY`** and **not read by `K-DEG`** | §6.2 |
| **I-5** | `ROBUST` / `POINT_ESTIMATE_ONLY` never defined | **predicate frozen**: `ROBUST` iff the one-sided 95 % item-bootstrap lower bound on `ΔAUC` is `> 0` on both datasets at the CONTINUE's `τ`; restated as fit-conditional and as a label only | §9 |
| **I-6** | *"The banned object is not built"* understates what §5.3 runs | reworded to **"constructed and measured as a pricing instrument on the train arena, and never deployed"**, with `H-L4` named as the foreclosure | §3.2 |
| **I-7** | the raw leg's own inferential cost unbudgeted; its degenerate cases unspecified | **raw-leg budget line added** (≈8 min) and its three declared changes written out: stability undefined, one `AUC_strat` cell instead of three, `K-DEG`'s two scales collapse to one; `UNSTABLE-POP` is not computable there | §2, §9 |
| **I-8** | six provenance and naming nits | (a) `headspace_drive.sh:20` / `c02_a0_cpu_v9.sbatch:85` are **assignments**, the `--out` flags are at `:24-25` / `:89`, and the sbatch invokes `c02_a0_mint.py` — citation corrected; (b) `lab_dev` is stored in **every** mint, not only the six `fold == -1` heads, and `GATE-LEDGER`'s expected count is set to **36**; (c) *"saturated at ~0.9999"* re-cited to `ERRPAT_HateMM:139-141` with `:131` named as the numeric row; (d) `GATE-FIXK20` named consistently and `STRATUM_OCCUPANCY` described as a non-gating emission throughout; (e) `GATE-FLOOR`'s *"6/6 seeds"* → *"all 6 (dataset, seed) cells"*; (f) F75's *"First measured negative…"* quoted at its own casing with the reopen's rendering distinguished | §2, §3, §5.2, §8.1, §11 |
| **I-9** | `STRATUM_OCCUPANCY`'s `Σ n_pos·n_neg < 200` floor unjustified; the tag did not block a CONTINUE | floor replaced by a **power-justified rule on `p_w`, the positives lying in weighted strata** — `< 30` fires, derived from the Hanley–McNeil `SE ≈ sqrt(1/(12p))` at `A = 0.5` (`p = 30 ⇒ SE ≈ 0.053`); `Σ n_pos·n_neg` demoted to a descriptive emission; **a CONTINUE may not carry the tag** — `K-FELDMAN` is defined to fail on a tagged dataset; and the pre-declared expectation that `τ_hi` (and possibly MHC-ZH at `τ_0`) will be underpowered is written in | §5.2, §9 |

---

*v5. No hash frozen, no config written, no code implemented, no namespace created, no
job submitted, no cache or test path opened, no metric or result produced. Zero GPU,
zero SLURM, zero Modal, zero teacher call.*
