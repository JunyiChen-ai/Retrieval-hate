# C09 Stage-0 (A0) — preregistration **v3** (round-2 repairs)

**Candidate.** C09 · Stable-Inversion Topology Surgery
**Registry claim.** *"OOF-stable high-confidence inversions identify topological
defects that can be corrected at encoder level while explicitly constraining break
exposure."*
**Registry dedup boundary.** *"Encoder-level topology intervention, not
thresholding, local reranking, verifier gating, NCA/SupCon, or hard-example
weighting alone."*
**Authorised by.** `TARGET_STATE.json::gate0_reopen_2026_07_31` —
`next_active_candidate_post_C04`.

> ## STATUS: `V3_REPAIRED_NOT_FROZEN_NOT_SUBMITTED` — awaiting fresh independent review.
>
> **Reading order.** v1 `refine-logs/C09_A0_PREREG_DRAFT.md` → review round 1
> `refine-logs/C09_A0_PREREG_DRAFT_REVIEW_ROUND1.md` (`REVISE 4C/8H/10I`) → v2
> `refine-logs/C09_A0_V2_RECORD.md` → review round 2
> `refine-logs/C09_A0_V2_PREREG_REVIEW.md` (`REVISE 1C/8H/10I`) → **this file**.
> v1 and v2 are superseded in full and must not be implemented. The v3 repair ledger
> is §12; v2's ledger for the round-1 findings is retained at `C09_A0_V2_RECORD.md §11`
> and every one of those repairs is carried forward here except where round 2
> overturned it (feature 11, `τ_hi`'s definition, the F97/F98 prior, the Holm
> construction, the gate split, CAL-3).
>
> **Submission preconditions, recorded (I-7).** `gate0_reopen_2026_07_31.c09_next_step`
> and `dispositions.promoted.sequencing` both require that the CPU job *"waits for
> C04's tranche to terminate (serial-execution precedent) and for main-dialogue
> authorization"*. Both must be true before `sbatch`, and both are recorded in the
> freeze record's submission-precondition table alongside the `squeue` check and the
> sha256 re-verification.
>
> No hash is frozen, no config is written, no job is submitted and no namespace is
> created by this document.

---

## 0. What v3 changes

Round 2 found that v2's headline repair did not hold: `FULL` feature 11 (*"the
number of the item's top-20 members that are themselves stable inversions"*) is a
function of **other query items' gold labels**, and `GATE-BLIND` — which counts reads
of the raw query-label array — is structurally blind to a *derived* target array.
Worse, the exemption offered for it was false on the model's own fitting rows.
**Feature 11 is deleted**, which makes §5.2's blanket label-blindness claim true and
`GATE-BLIND` meaningful, and `GATE-BLIND` is additionally re-specified to count reads
of the **target-derived** arrays.

Round 2 also found `K-FELDMAN` not computable (Holm applied to bootstrap quantiles
with no p-value), `τ_hi` not actually a number, the registered prior mis-stated in
C09's favour (F97's **honest positive datum** `+0.0269 / +0.0104 / +0.0182` is not a
null — it is band B, sitting *below* the bar), F98's `ban_scope` (b) unadjudicated
though it names `NET`'s object, no threshold-degeneracy control in a campaign that
has measured this operator class collapsing into one twice, a self-contradictory
gate/decision coupling, and CAL-3 declared binding then omitted. All are repaired.

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

A0 measures three things:

1. **Reach (`O1`)** — is the OOF-stable-inversion population large enough that
   fixing all of it would clear `+0.050 / +0.050`? Expected to pass; a fail kills at
   zero cost and closes every confidence threshold at once by arithmetic (§4.3).
2. **Conditional identifiability (`D-FELDMAN`)** — *within a matched local
   configuration*, is "this item is an OOF-stable inversion" predictable from
   geometry alone, with no access to any item's gold label at prediction time?
3. **Conversion (`NET`)** — at the frozen operating points, does the population yield
   enough net correct-minus-broken items in the currency `banned_constraints[10]`
   mandates, without collapsing into a bare threshold shift (§6.2)?

**A0 trains no encoder, touches no test split, and establishes that no operator
exists.** A CONTINUE means only that the target population is large enough and
locatable enough to be worth building an operator for — and even that is conditional
on §11's Stage-1 precondition.

---

## 2. Arena, instrument, cost

**Path.** The banked **fold-head / deployed-head arena** only. F113 stands: *a
raw-key arena may KILL but may not PROMOTE*, so a Stage-0 PASS is rendered on the
fold-head path.

**Instrument — verified present, nothing to build.**
`scripts/analysis/headspace_mint.py`, `headspace_arena.py`, `headspace_fidelity.py`,
`headspace_report.py`, plus the six banked
`headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`. **The mint is `headspace_mint.py`
invoked unmodified with its sha256 asserted; the only new code in this A0 is the
analysis script and the sbatch driver.** `headspace_mint.CLI` admits only
`{hatemm, zh}`, which is an independent reason MHC-EN is out of scope: adding it
would require editing a sha-pinned artifact.

**Fold contract (pinned).** `StratifiedKFold(n_splits=5, shuffle=True,
random_state=0)` over the train split, stratified on the train label — i.e.
`mechnov_pairverify.K_FOLDS = 5`, `FOLD_SEED = 0`. `headspace_mint.py:203-216`
asserts this assignment against the banked `scripts/analysis/vsw_ckpt/<ds>/f<fold>.npz`
`ho_idx` and refuses to run on mismatch. The assignment is a function of the label
vector alone and is therefore **identical across head seeds** — the property §5.2's
nested split relies on.

**Configuration.** 2 datasets × 3 head seeds × 5 item-disjoint folds = **30
fold-heads**, plus 2 × 3 = **6 deployed-configuration (`fold == -1`) heads** for the
real fidelity read = **36 heads total**. Bank = the fitting pool; queries = the
held-out fifth. Query labels are **train-split** labels held out from the head that
judges them — this is what "OOF" means throughout.

**Mint output naming (I-6a).** All 36 mints are written into **one** directory,
`<scratch>/mint/`, under exactly the names `headspace_arena.load_mint` and
`headspace_fidelity.py:66` expect: `mint_{dataset}_s{seed}_f{fold}.npz` for
`fold ∈ 0..4` and `mint_{dataset}_s{seed}_ffull.npz` for `fold == -1`. Nothing else
is written there.

**Datasets.** `HateMM` (n = 744 train items = pooled query count) and `MHC_zh`
(n = 579). Per-fold held-out counts, from the banked arena outputs:
`149/149/149/149/148` and `116/116/116/116/115`. **MHC-EN is OUT OF SCOPE for A0**;
two datasets is what the bar requires — **and with EN out of scope the two-dataset
requirement has zero slack: a failure on either dataset is a failure of the
conjunct.**

**Cost.** `0 GPU-hours`. Per-fold head checkpoints are not persisted
(`headspace_mint.py:274-281` monkeypatches `torch.save` to a no-op), so heads are
re-minted at ~25–60 s CPU each. **Budget, itemised (I-8):**

| item | count | estimate |
|---|---|---|
| head mints | 36 | ≤ 36 CPU-min (60 s/head ceiling; C02's 36 banked mints ran 33–60 s) |
| deployed vote + features, all cells | 30 fold-cells × 2 spaces | ~2 min |
| `D-FELDMAN` primary (LR, 5 folds × 2 feature sets × 2 `τ` × 2 ds) | 40 fits | < 1 min |
| GBM capacity check | 40 fits | ~3 min |
| `SHUFFLE-POP` (200 draws × 2 sets × 5 folds × 2 `τ` × 2 ds) | 16 000 LR fits | ~10 min |
| `RANDOM-POP` (200 draws, same shape) | 16 000 LR fits | ~10 min |
| `UNSTABLE-POP` | 40 fits | < 1 min |
| item bootstrap `B = 10000` (re-scoring only, **no refits**) | — | ~3 min |
| raw-arena leg | 5 folds × 2 ds | ~2 min |
| `GATE-DEVFID` (`headspace_fidelity.py`, reads banked trainlogs) | 2 | seconds |

**Total ≈ 70 CPU-minutes.** The bootstrap re-scores banked OOF predictions and does
**not** refit, which is what keeps `B = 10000` cheap. One CPU-only SLURM job, 8 CPU /
32 GB / **no GPU, no `--time`**; C02's A0 (job `13847`: 8 CPU / 0 GPU / 32 G) ran
`00:29:49`. **Resume path:** `headspace_mint.py:192-194` skips any `--out` that
already exists and the driver is a sequential loop, so a re-submitted job resumes at
the first missing head.

**F88's binding caveat is satisfied by construction.** F88 requires that *"a
CPU-trained arm must be paired against a CPU-TRAINED FLOOR, never against the banked
GPU floor."* Every arm and every floor here is minted inside the same CPU fold-head
arena.

**Standing clauses adopted.** `PREGATE_DETERMINISM_CLAUSE.md` **DET-1 … DET-4** and
`PREGATE_CALIBRATION_CLAUSE.md` **CAL-0 … CAL-5**, binding on this run.

- **DET-1.** `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=8`
  exported before any Python process starts; `headspace_mint.det1_assert` hard-fails
  otherwise. **DET-2.** the full runtime block recorded in every output JSON.
  **DET-3.** parity asserted at **Tier A / Tier B, 4 decimal places**, never beyond.
  **DET-4.** the primary estimator is `LogisticRegression(solver="lbfgs")`, the convex
  arm DET-4 prefers (measured 4-dp invariant across the whole thread grid,
  `PREGATE_DETERMINISM_CLAUSE §1.3`); the gradient-boosting arm is a declared capacity
  check that **no decision rule reads**, and carries no Tier-C band because no verdict
  rests on it.
- **CAL-0** restated and scoped: *"The raw train-space arena is not established as
  predictive of deployed effects."* C09's decision arena is the **fold-head** arena;
  the raw arena appears only as F113's KILL-only corroborator (§9). **CAL-1** governs
  that corroborator — its decisive bars are within-arena relative comparisons, the
  property CAL-1 requires. **CAL-2**'s hard half is honoured in the form this arena
  admits: `GATE-ZEROOP` (the empty-operator identity) is C09's `FIXK_20` analogue and
  `GATE-FLOOR` checks the floor reproduces. **These are two different objects and
  neither is "strictly stronger" than the other** (I-5); both are run. CAL-2's
  `FIXK`/Spearman provenance leg is a *raw-arena* check against F94's banked k-curve
  and is not run, because this arena is not that arena. **CAL-3** is discharged in
  §9. **CAL-4** labels every quantity closed-form or trained (§5.6). **CAL-5** runs
  **against** C09: the Stage-1 operator this A0 would license changes the map, i.e.
  **channel (a)/(d)**, which *"carries NO transfer warrant and must say so in its
  limitations."* It is said, in §10.

---

## 3. Label-use discipline — LEGAL, on written texts, with the counter-texts carried

Identifying OOF-stable inversions requires reading train labels out of fold. This is
**legal**, resolved on two written texts rather than by inference:

- `autoresearch/goal_mllm_plus3/state/progress.json:25` — *"Legal attack on
  selection-locked pools = trained selector/reshaper on train labels only (F66 binds
  only fixed-map phi0)."*
- `refine-logs/LITSWEEP3_DATA_CENTRIC.md:82` — *"those select **per test instance**;
  curation selects **train items once, globally, applied identically to every test
  query** — a symmetric operator, so law-III/F66's per-item ban does not apply to the
  mechanism (though Wall-A still caps the achievable magnitude)."* The same section
  prices that mechanism at *"+3 any dataset: ~1-2%"* (`:95`) and *"at most
  +0.001-0.006"* (`:91`).

Four boundaries flip C09 to illegal and are written in as **HALT** conditions:

- **`H-L1`.** Any query-time consultation of the stability statistic or of the
  `D-FELDMAN` classifier. F47 fires directly, and its escape clause is closed: an
  OOF-stability statistic *is* "derivable from banked features/votes", so it is not
  the *"genuinely NEW information source"* the exception requires.
- **`H-L2`.** Any per-item exception that survives to inference as a per-item rule.
- **`H-L3`.** Any read of a dev **label** or any read of a test path, at any stage,
  by any code path. (Dev *features* are read by the six `fold == -1` fidelity heads,
  by `headspace_mint.py`'s own contract; dev **accuracy** is read by
  `headspace_fidelity.py` from banked trainlogs. Both are declared, counted and
  reported by `GATE-LEDGER`; neither reaches any decision rule.)
- **`H-L4`.** Any use of `D-FELDMAN` — the classifier, its score, or any monotone
  function of either — as a **selector, gate, router, abstention rule or risk
  ordering over a deployed decision**. Banned by measurement twice over (F47, F97,
  F98); this A0 does not build it. See §3.1 and §3.2.

### 3.1 `D-FELDMAN` against F47's ban_scope — adjudicated

F47's `ban_scope`, verbatim:

> *"Per-item cross-channel selection/routing over banked channels: CLOSED at all
> three supervision sources (unsupervised=K9 zeros; train-supervised=memorization-
> degenerate target, CLIP LOO 0.998; dev-supervised=negative at CV ceiling −0.046 <
> perm-null). Decision-level meta-features (vote margins, purity, sub-votes,
> confidence differential, transcript stats) carry NO per-item routing signal, GBM or
> linear. Do NOT re-propose per-item selectors over frozen channels regardless of
> feature family or nonlinearity unless the selector input is a genuinely NEW
> information source not derivable from banked features/votes."*

`D-FELDMAN`'s feature family is **literally the family that sentence names**, and its
input is **not** a new information source. The ban is engaged:

- **What the ban closes and C09 accepts.** Using these features to **select, route or
  gate a deployed decision** is closed. `H-L4` writes that in. C09 does not propose
  it, at Stage-0 or Stage-1.
- **What the ban does not reach.** F47's measured object is a **per-item router over
  frozen channels**, and its target is *which channel to trust for this item*.
  `D-FELDMAN`'s target is a different random variable — *is this item a
  three-seed-stable inversion of the deployed vote* — and its output is never
  consulted at prediction time. A probe that measures whether a region exists is not
  a selector over that region. This distinction is **narrow**: if a future proposal
  reads the probe's score at query time, F47 fires and `H-L1`/`H-L4` HALT the run.
- **The symmetric correction, at its true weight.** F47's train-supervised leg is
  *"memorization-degenerate target, CLIP LOO 0.998"*. F114 rules that premise a
  **CLIP** number while the deployed Qwen heads sit at `0.9406 / 0.8915 / 0.8154`.
  The saturation objection therefore does not transfer to this arena — F113's own
  reason for building it (*"That objection does NOT apply to a head trained on 4/5 of
  the train split and queried with the held-out fifth"*). This weakens F47's
  train-supervised leg **on this arena**; it does not touch F47's unsupervised or
  dev-supervised legs, and it does not weaken the decision-level-meta-features
  sentence, which was measured independently of the 0.998 premise.

### 3.2 `NET` against F98's ban_scope (b) — adjudicated (round-2 H-4)

F98's `ban_scope`, verbatim in the relevant part:

> *"Concretely banned on this neighbourhood object: … (b) ANY per-item selector,
> router or adjudication gate over the same neighbourhood WITH ANY FEATURE FAMILY —
> the verifier features are dead by K-VGA-3 and the F47 features have a measured,
> already-banked ceiling of +0.0269 … (d) re-deriving the HateMM train-arena
> threshold observation — three independent measurements now agree it is ~87% a bare
> threshold move and it is measured DEAD in the deployed head space on test
> (ERRPAT-HateMM sec2.1: +0.0000/+0.0016)."*

**§5.3's `NET` is arithmetically that object**: top-`k` by a per-item classifier
score over the deployed top-20 neighbourhood, with the selected predictions flipped.
Reading the ban narrower than its own text is not available. The adjudication, in
three parts:

1. **The banned object is not built.** `H-L4` forecloses deploying any such selector,
   and nothing in this A0 or in §11's successor is one. `NET` is an **accounting
   instrument** applied to an *idealised* operator, computed on the train arena, and
   never proposed as a component.
2. **Its measured ceiling is registered as this design's prior** (§6).
3. **The scientific consequence, which is the one that matters.** The per-item
   selector `NET` prices is a **strictly more capable** operator than §11's global
   symmetric reshaper: the selector may act on each item individually, the reshaper
   must apply one map to all. **`NET` is therefore an optimistic upper bound on the
   Stage-1 successor's conversion.** Two corollaries are pre-declared: a `K-NET`
   failure is *a fortiori* a KILL of the successor, and **a `K-NET` pass licenses
   nothing about the successor** — §11's precondition, not `NET`, is what a CONTINUE
   must satisfy.
4. **Clause (d) is engaged too**, and is why §6.2's degeneracy control gates the
   verdict rather than merely being reported.

### 3.3 The counter-text, carried at its adjudicated weight

`LITSWEEP5_COMPLETENESS.md` §4(ii), *"The contradiction (load-bearing)"*, was written
**after** the oracle-queue ruling and observes that its two blessed classes —
*"Trained SELECTOR on train labels"* and *"Trained symmetric RESHAPER on train
labels"* — are *"both already measured dead"*, and that the ruling *"was written at
lit-round-count 3 — before F75/F77/L1 sharpened the walls."*

**The counter-text is itself DOWNGRADED, NOT VACATED** (reopen R7 I-2): §4(ii)'s
first blessed-class death rests on *"the deployed kNN vote memorizes train (CLIP LOO
0.998)"*, and F114 rules that a CLIP number against deployed Qwen heads at
`0.9406 / 0.8915 / 0.8154`, leaving train-side headroom 30×–92× larger.
`LITSWEEP5_COMPLETENESS.md` is **not** among the nine records F114 corrected, so the
retraction never reached it.

**§4(ii)'s independent leg is untouched — and it is carried with its conclusion, not
only its counts (round-2 I-10a).** `LITSWEEP5_COMPLETENESS.md:128` verbatim:

> *"train-disagreement 'Qwen-correct' = **0/109, 0/102, 0/92**, and that train base
> rate is the *inverse* of the ~0.55 test base rate (L1 §0, F47 §3.2). Training
> labels **cannot** supervise the test-time selection decision in this pipeline — a
> data-generating-process obstacle upstream of any selector capacity."*

That sentence bears directly on §11's train-label-supervised successor, so the answer
is given here rather than omitted. **The obstacle it names is an inverse-base-rate
obstacle, and it is a property of the *saturated full-train* arena, not of this
one.** Its mechanism is that on a full-train-fitted head the train error rate is
~0 while the test error rate is ~0.11–0.15, so the training target for a selector is
near-empty. In the fold-head arena the query items are held out from the head that
judges them and the measured pooled deployed accuracy is `0.8858–0.8946` — a train
error rate of `0.105–0.114`, i.e. **comparable to the deployed test error rate, not
its inverse**. The `0/109, 0/102, 0/92` counts are therefore not transportable to
this arena; that is precisely what F113 built the arena to fix. **What survives and
is carried as a headwind:** the *general* form of the objection — that a
train-supervised target may be a different object from the test-time one — is not
refuted by an error-rate match alone, and §10 scopes every verdict here to the train
arena for exactly that reason.

**Net: legality is not in question; viability is. C09 inherits a weakened prior.**

### 3.4 `D-FELDMAN` is a probe, not a component

**`D-FELDMAN` is a Stage-0 *identifiability probe*, never a deployable component.**
Whether a *global operator acting uniformly on that region* is legal and buildable is
a Stage-1 question with its own gate (§11); A0 makes no claim about it.

---

## 4. Population definition — every threshold frozen before any run

### 4.1 The deployed decision and the vote scale

`score_{i,s}` is defined by **literal reference to `scripts/analysis/mechfix_ops.py:94`**:

```python
votes = ((lab * 2 - 1) * sim * w).sum(1) / w.sum()
```

with `w = _rank_weights(20) = [20, 19, …, 1]`, `Σw = 210`, `sim` the float32 faiss
inner products of L2-normalised keys, `lab` the **bank** labels of the top-20, and
the decision `predict 1 iff score ≥ 0` (`mechfix_ops.py:95`). **The vote is already
divided by `Σw`**, which is why v1's `|score|/Σw` was a double normalisation.

`conf_{i,s} ≡ |score_{i,s}|` on that scale. For orientation, and declared as a
**transferred expectation, not a measurement of this arena**:
`ERRPAT_HateMM_2026-07-26.md:130` reports *"median |vote| (**3-seed mean vote**)"* of
**`0.7267`** for errors against **`0.9873`** for always-correct items, on the **test
split** under a **CPU-reconstructed proxy** of the deployed head
(`ERRPAT_HateMM §0.1`, 52 s/seed).

### 4.2 Inversions and stability

Item `i` is an **inversion at seed `s`** iff its deployed prediction disagrees with
its gold train label. `i` is an **OOF-stable inversion** iff it is an inversion in
**all three** head seeds; `i` is an **unstable error** iff it is an inversion in
exactly 1 or 2 seeds; `i` is **always correct** iff in none.

**Provenance of the expectation that this population is large.** F88 measured
seed-invariance on the **test split** under the **deployed-head proxy**, not in this
arena. The registry text verbatim
(`gate0_reopen_2026_07_31.dispositions.promoted.supporting_evidence_verified`):

> *"F88 FAITHFUL: error sets ~90% seed-invariant - HateMM 24-25 of 26-28 errors wrong
> in 3/3 seeds (89-93%); ZH 22 of the 25-item union wrong 3/3 with NOTHING at exactly
> 2/3 and all 12 false negatives 3/3-stable"*

Those figures are **transferred expectations** motivating the design. The population
this A0 prices is measured **in-run, in the fold-head train arena**, and nothing in
the decision rule reads an F88 number.

### 4.3 Confidence thresholds — a primary and a registered co-primary

Two thresholds, both frozen here, both computed **in-run**, per dataset:

- **`τ_0 = 0` — PRIMARY.** All OOF-stable inversions, no confidence restriction.
- **`τ_hi` — the REGISTERED "high-confidence" co-primary, pinned (round-2 H-1):**

  > **`τ_hi = median over items i ∈ P_0 of  c_i`, where `c_i ≡ mean_s |score_{i,s}|`
  > over the three head seeds.**

  `c_i` is a **per-item** quantity, matching §5.3's item score and ERRPAT's own
  3-seed-mean-vote convention. `P_τ ≡ { i ∈ P_0 : c_i ≥ τ }`. The same `c_i`
  convention defines `q25`, `q75` and `q_max` — there is exactly one confidence scale
  in this design and it is per item.

**Where monotonicity holds.** `|P_τ|` is monotone non-increasing in `τ` by
construction and `Δacc_{O1} = |P_τ| / n`; flipping a strictly larger set of errors
also weakly raises both per-class F1s, so `ΔmF1_{O1}` is monotone in the flipped set
as well. **`K-REACH` firing at `τ_0` therefore closes every `τ ≥ 0` on both metrics
by arithmetic.** It is **false** for `NET` and for `ΔAUC`: raising `τ` redefines the
target and both precision and AUC can rise on a purer subpopulation. So neither
`K-FELDMAN` nor `K-NET` may be read beyond the `τ` it was evaluated at, and both are
evaluated **and reported** at `τ_0` **and** `τ_hi` unconditionally — including when
`K-REACH` fails at `τ_hi` (round-2 H-2).

**Pre-declared arithmetic consequence.** `τ_hi` is the median, so
`|P_{τ_hi}| = ⌊|P_0|/2⌋` or `⌈|P_0|/2⌉`, and `K-REACH` at `τ_hi` is approximately the
inequality `|P_0|/n ≥ 0.10`. On the transferred F88 rates that is close to the
plausible value, so **the co-primary may be decided by an arithmetic identity rather
than by identifiability or conversion.** That is declared now, is not an artefact,
and §9's scope bullets give it its own entry. `q_max` — the largest quantile of the
`c_i` distribution at which reach still clears `+0.050` — is reported as a
descriptive quantity that no decision rule reads.

### 4.4 Frozen ancillary definitions

- **Right analogue** of `i`: the highest-ranked bank item carrying `i`'s gold label.
  **Mechanism diagnostic only; it reads `i`'s gold label and is excluded from every
  feature set by `GATE-BLIND`.** (Transferred motivating measurements, test split,
  proxy head: `ERRPAT_MHC-ZH_2026-07-26.md:234-235` — *"first same-gold-class train
  neighbour sits at median rank 1.5 for the core errors (11 of 22 at rank 1; all 22
  within rank 14). The right analogues are present and top-ranked; they are simply
  out-voted."* `ERRPAT_HateMM_2026-07-26.md:134` — median rank `3.0`; `:135` —
  `6 / 27` errors with no true-label neighbour in the top-20 at all.)
- **`pred_purity_{i,s}`**: fraction of `i`'s top-20 whose **bank** label equals `i`'s
  own **predicted** class at seed `s`. Label-blind for every query item.
- **Configuration stratum (frozen, label-blind).** Computed **per (dataset, seed)
  cell** (round-2 I-2), as the cross of
  - `|score_{i,s}|` **tercile**, edges from that cell's own `n` query items, and
  - `pred_purity_{i,s}` bucket `{[0, 0.60), [0.60, 0.80), [0.80, 0.95), [0.95, 1.0]}`.

  12 strata per cell. Both axes are computed in-run from this arena and are
  label-blind, so no test-derived bucket enters. Bucket edges are frozen here and are
  not tuned.

---

## 5. The three measured quantities

### 5.1 `O1` — reach (necessary; an upper bound, and declared as one)

For each seed `s`, flip the prediction of every item in `P_τ` and recompute accuracy
and macro-F1 against the deployed floor of that (dataset, seed) cell.
`Δacc_{O1} = |P_τ| / n` identically for every seed; `ΔmF1` is recomputed from the
realised confusion matrix, not assumed. Primary = mean over the three seeds; per-seed
values reported.

**Scope.** `O1` is a **label-flip oracle over one nominated population**, not the
*"full-bank or representation-level oracle"* `stage_0_reachability` names. It is the
tightest zero-cost **upper bound** on what any operator confined to fixing stable
inversions could reach: a fail is a closure, a pass establishes nothing beyond
"population large enough".

### 5.2 `D-FELDMAN` — conditional, incremental identifiability

**Question.** *Within a matched local configuration*, can "is this item an OOF-stable
inversion?" be predicted from geometry alone — **over and above what the
configuration itself already says**?

**Why conditional and incremental.** H-MEMORISATION does **not** predict
unconditional AUC ≈ 0.5. Feldman's long-tail singletons *are* the low-density,
weak-margin, no-analogue items, so a label-blind feature set separates them under
**either** hypothesis; the separation is already banked
(`ERRPAT_HateMM:130`, median `|vote|` `0.7267` vs `0.9873`). v3 conditions on the
configuration stratum and measures the **increment over a configuration-only
baseline**, which is the quantity the hypotheses actually disagree about.

**Rows, split, estimators.**
- Rows are `(item, seed)`; the target is **per item** and constant across that item's
  three rows.
- **The nested-CV partition *is* the frozen 5-fold arena partition** (§2). An item's
  score comes from a model fit on items from the other four arena folds only. This
  (a) groups all three seed-rows of an item together, (b) makes the scored item
  disjoint from its own arena fold, and (c) introduces **no new hyperparameter and no
  RNG**.
- **Primary estimator (DET-4):** `LogisticRegression(penalty="l2", C=1.0,
  solver="lbfgs", max_iter=2000, class_weight="balanced", tol=1e-6)` on z-scored
  features, standardisation fit on the training folds only.
  **Capacity check, read by no decision rule:**
  `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=3,
  l2_regularization=1.0, random_state=20260801)`. Both frozen here; this mirrors
  F47's own two-family protocol, which found *"NO per-item routing signal, GBM or
  linear"*.

**Two frozen feature sets — 7 and 12 features.**

*`BASE` — configuration-only (7).* `|score_{i,s}|`; `pred_purity_{i,s}`; mean and
standard deviation of the top-20 similarities; the similarity gap between ranks 1 and
20; local bank density (mean similarity to the 50 nearest bank items); the item's own
L2 norm before normalisation.

*`FULL` — `BASE` + structural block (5).* rank of the first neighbour whose bank
label differs from the top-20 majority bank label; the number of runs (label changes)
in the rank-ordered bank-label tuple of the top-20; mean and standard deviation of
the **bank-side degree** of the top-20 members (how often each appears in other query
items' top-20 within the same fold); the signed gap between the best rank-1
similarity to a class-1 bank item and to a class-0 bank item.

> **v2's sixth structural feature — the count of an item's top-20 members that are
> themselves stable inversions — is DELETED (round-2 C-1).** It was a function of
> other query items' gold labels; its claimed exemption ("training-fold items only")
> was false for the model's own fitting rows, whose neighbourhoods include the scored
> item's fold; and `GATE-BLIND`, which counts reads of the *raw* label array, was
> structurally blind to a *derived* target array. Deleting it makes the blanket claim
> below true and `GATE-BLIND` meaningful. **`FULL` now contains no feature that reads
> any query item's gold label, directly or derived.**

**Every feature reads only: the query item's own key, the bank keys, and the bank
labels.** No feature reads any query-side gold label, and none reads any target-derived
array (`is_inversion[seed]`, `is_stable_inversion`). Enforced structurally and
counted — `GATE-BLIND`, §8.1.

**Extra retrieval the features need, declared (I-6d).** `deployed_vote` returns the
top-20 `(I, sim)` only. Two `BASE`/`FULL` features need more: local bank density
requires a `k = 50` faiss search of each query against the fold's bank, and the
per-class rank-1 similarity gap requires a per-class top-1 search. Both are computed
with `mechfix_ops._norm32` + `_flat_ip` — the same engine, on the same normalised
keys, so no arm-to-arm delta can be an engine artefact — and both are inside the
analysis script, which remains the only new analysis code.

*Declared feature-degeneracy read.* `ERRPAT_HateMM:141` measures that with cosine
saturated at ~`0.9999` (`:131`), *"Distance-based abstention/gating has essentially
no dynamic range to work with."* Several `BASE` features are similarity-derived and
may be near-constant here. The run emits a `FEATURE_DEGENERACY` block — per-feature
standard deviation and distinct-value count, per (dataset, seed) — so a reader can
see which features carried information. It gates nothing.

**Primary statistic, fully specified (round-2 I-2).**

> For each (dataset, seed) **cell**: partition the cell's rows into its 12 frozen
> strata. For each stratum containing ≥ 1 positive and ≥ 1 negative, compute the
> Mann-Whitney probability that a random positive row outscores a random negative row
> **from the same stratum** (ties counted as ½). Pool across strata weighted by
> `n_pos × n_neg` in that stratum — **strata with zero positives or zero negatives
> receive weight 0**. That is `AUC_strat` for the cell. **Pooling unit is the row**
> within a cell; the per-dataset value is the **mean over the three seed cells**.
>
> **`ΔAUC = AUC_strat(FULL) − AUC_strat(BASE)`**, per dataset, both terms on
> **identical** OOF folds, identical rows and identical strata — so `ΔAUC` is a
> genuinely **paired** quantity.
>
> **Strata are frozen from the full sample and held fixed across every bootstrap
> resample and every permutation draw.** They are a design object, not an estimate.
>
> **Degenerate case:** if in some cell *every* stratum is single-class, `AUC_strat`
> is undefined there; the run **HALTs** with that cell named, rather than substituting
> a value.

**Positive and negative classes.** Positives: `P_τ`. Negatives:
`CONFIG-MATCHED-CORRECT` = query items correct at all three seeds. Matching is
achieved **by the stratification itself** — no sampling, no RNG, no discarded data.
Unstable errors are **excluded from both classes** and are `UNSTABLE-POP`'s subject;
they are still fully costed in `NET` (§5.3).

**Inference, fully specified (round-2 H-6, I-8).** One-sided **item-level** bootstrap:
resample **items** (not rows) with replacement, `B = 10000`, RNG
`numpy.random.default_rng(20260801)`, re-scoring the banked OOF predictions (no
refits) and recomputing `ΔAUC` on each resample. Define the achieved significance
level

```
p = (1 + #{ b : ΔAUC_b ≤ 0 }) / (B + 1)
```

(the `1/(B+1)` floor is built in, so `p ≥ 9.999e-5`). **Holm is applied to `p`, over
the 2 `τ` hypotheses within each dataset**, at `α = 0.05`: sort the two `p` values,
compare the smaller to `α/2 = 0.025` and, if it rejects, the larger to `α = 0.05`.
**The dataset conjunction is an intersection-union test and receives no correction**
— an IUT's level is controlled by the largest component p-value. The correspondingly
adjusted one-sided lower bounds (the `1 − α/2` = 97.5 % bound for the first
hypothesis, the `1 − α` = 95 % bound for the second) are reported alongside, so the
record carries an interval and not only a decision. Resampling items rather than rows
is the second half of round-1's C-3 repair: a row-level bootstrap would be
anti-conservatively narrow by roughly `√3`.

**On the "conversion-equivalent AUC".** It is not well-defined — AUC does not
determine precision at a fixed selected count without the score distribution — so the
conversion leg is adjudicated where it *is* exactly decidable, in precision and
net-item space (`K-NET`), and the run reports at every operating point the
**conversion-equivalent precision** `π* = (1 + bar/k)/2` alongside the realised
precision.

### 5.3 `NET` — conversion, fully costed

**Accounting.** At an operating point the classifier selects a set `S` of `k` items
from **all `n` query items**. For each seed `s`:

```
net_s = |{ i ∈ S : wrong at s }| − |{ i ∈ S : right at s }| = 2·|{ i ∈ S : wrong at s }| − k
```

Every selected item is costed at every seed. `CONFIG-MATCHED-CORRECT` is retained as
**reporting stratification only**. **`GATE-SELFTEST`** asserts `net_s == n · Δacc_s`
exactly, for every seed × dataset × operating point × `τ`, and HALTs on mismatch.

Primary `net` = mean over the three seeds; the per-seed minimum is also reported.
Exchange rate is reported as a diagnostic and **reads no decision rule**
(`banned_constraints[10]`).

**Item score.** For each item, the mean of its three seed-rows' OOF predicted
probabilities from the `FULL` logistic model. Deterministic; no RNG.

**Out-of-support declaration (round-2 I-9).** `D-FELDMAN` is fit on
`P_τ ∪ CONFIG-MATCHED-CORRECT`, i.e. unstable errors are excluded from the fitting
population — yet `NET` ranks **all `n`** items by that model. Unstable errors are
therefore **out of the ranker's support**, and they are exactly the items whose `NET`
contribution is partial (wrong at some seeds, right at others). This is declared, not
repaired: the alternative (fitting on a three-class target) would change the object
`D-FELDMAN` measures. The run reports the **composition of `S` by class** — stable
inversion / unstable error / always correct — at every operating point, so a reader
can see how much of any net came from out-of-support items.

**Frozen operating points.** Three points on the **selected-count** scale:
`k ∈ {|P_τ|, round(1.5·|P_τ|), round(2·|P_τ|)}`, top-`k` by item score over all `n`.

**The currency, adjudicated.** Three surfaces name a net-item figure:

1. `unified_pilot_gate.stage_0_reachability` — the **governing gate text** — ties the
   net requirement to the **`+0.030` final bar**: *"with enough net
   correct-minus-broken items for the +0.030 final bar."* `banned_constraints[10]`
   supplies the figure: **`22.3` (HateMM) / `17.4` (MHC-ZH)**, exactly `0.030 × 744`
   and `0.030 × 579` on the **train arena** at `n = 744 / 579`.
2. The reopen's C09 `bar` field names **`37.2 / 29.0`** — `0.050 × 744 / 579`.
3. C02's own A0 ran `net_fix_rate: 0.03`, consistent with (1).

**Binding rule: (1).** The gate text is explicit that the *oracle reach* bar is
`+0.050 / +0.050` while the *net* requirement is sized to the `+0.030` final bar, and
this is not a softening — for `O1`, which breaks nothing, `net ≡ n · Δacc`, so a
`+0.050` net screen on `O1` would restate the accuracy screen and carry no
independent information. The net screen only bites where breaks are real. **`37.2 /
29.0` is computed and reported at every operating point as a declared secondary that
*scopes* a CONTINUE and can never create or block one.**

**The macro-F1 leg.** At every operating point the run computes `ΔmF1_s` from the
realised post-flip confusion matrix, and `K-NET` requires `mean_s ΔmF1_s ≥ +0.030`
at the same cell.

### 5.4 What is *not* an inferential quantity

`stage_0_reachability` is written as a **threshold** rule, not a CI rule; the CI
requirement first appears at `stage_1_signal`. `O1` and `NET` are therefore
adjudicated on **point estimates against frozen thresholds**, and no test is
performed, so there is no multiplicity to correct there. `K-FELDMAN` is the only rule
that performs a test, and §5.2 specifies its family and correction exactly. Forking
paths are controlled structurally: the `2 τ × 3 k` grid is frozen, exhaustive and
reported in full; a CONTINUE requires the **same `(τ, k)` cell to clear on both
datasets simultaneously**; and the CONTINUE names its cell, with §10 scoping the
verdict to it. One-sided item-level bootstrap lower bounds are computed and reported
for every decision quantity and used **only** to tag a CONTINUE `ROBUST` /
`POINT_ESTIMATE_ONLY`.

### 5.5 `DATA-DEFECT-OVERLAP` — the third hypothesis, priced (round-2 I-10b)

A positive `ΔAUC` is consistent with a third explanation neither §6's columns
originally named: **clustered annotation or collection noise**, a data defect no
encoder operator can fix. This repository has documented defects on both datasets, so
the run prices the overlap directly, at `$0`, from the `data/gt/*/train.jsonl` text
field only (**no label is read for this diagnostic**):

- **MHC-ZH `<em class="keyword">` markup.** The reopen records markup-bearing rows
  hating at `5×` the no-markup rate — train hate rate `0.5802` (141/243) with it vs
  `0.1161` (39/336) without, against a `0.3109` base — *"so part of the reported ZH
  0.8537 floor rests on how the corpus was harvested rather than on video content."*
  **Re-measured this session: 243 of 579 MHC-ZH train rows contain `<em`.**
- **HateMM whitespace-only transcripts.** **Re-measured this session: 39 of 744
  HateMM train rows have a whitespace-only `text` field** (0 on MHC-ZH).

**Reported:** the enrichment of `P_τ` and of `S` in each flagged sub-population,
against the arena base rate. **Stated honestly:** *no quantity in this A0 separates
H-DATA-DEFECT from H-TOPOLOGY.* A high enrichment would make H-DATA-DEFECT the
leading explanation of any positive `ΔAUC` and would be reported as such in the
verdict's scope; a low enrichment weakens it without excluding it. This diagnostic
gates nothing.

### 5.6 CAL-4 declaration

**Closed-form:** every deployed-vote quantity, every count, `O1`, `net`, `Δacc`,
`ΔmF1`, the strata, every feature, and the degeneracy agreements. **Trained:** the
head mint (30-epoch Adam, DET-3 Tier B, 4-dp parity asserted by `GATE-FLOOR`), the
logistic estimator (measured 4-dp invariant across the thread grid), the GBM capacity
check (no verdict reads it).

---

## 6. The discriminator — three-valued, decidable, with the prior the repository banked

### 6.1 The hypotheses and the bands

The **numerical** leg of the Feldman objection is already retracted in-repo —
`HEADCOV_PREGATE_RECORD.md:305-310` withdraws *"the Feldman flourish"* because the
deployed heads sit at 0.82–0.94, not 0.998 — while its **substantive** leg is
preserved verbatim there and stands: *"memorising a long-tail singleton does not
transfer to an unseen member of the same one-member sub-population."*

| | **H-TOPOLOGY** (C09) | **H-MEMORISATION** (Feldman) | **H-DATA-DEFECT** |
|---|---|---|---|
| what the stable inversions are | a **region** with a shared geometric signature beyond atypicality | **singletons**: each wrong for its own reason | items whose **labels** are wrong or shortcut-driven, clustered by collection artefact |
| unconditional AUC | high | **also high** | also high |
| **`ΔAUC`** | **> 0** | **≈ 0** | **> 0** — indistinguishable from H-TOPOLOGY here |
| `NET` at the frozen points | clears `22.3 / 17.4` | ≈ 0 or negative | may clear |
| what an operator could do | act uniformly on the region | nothing | nothing — the fix is annotation, not geometry |
| **separated by this A0?** | — | yes, by `ΔAUC` | **NO** — only *priced*, by `DATA-DEFECT-OVERLAP` (§5.5) |

**The registered prior, restated correctly (round-2 H-3).** v2 called F47/F97/F98 "a
null". F97's own `ban_scope` says something sharper and it is quoted verbatim:

> *"HONEST POSITIVE DATUM, RECORDED AND EXPLICITLY NOT PROMOTED: F47-features-as-
> adjudication-gate is REAL and permutation-validated — +0.0269 on HateMM (p=0.0050,
> fold signs +++++), +0.0104 on ZH (p=0.0050), +0.0182 on EN (p=0.0100) — a genuine
> refinement of F47's epitaph (dead as a per-item CHANNEL SELECTOR, not dead as a
> per-item ADJUDICATION GATE). It is nonetheless SUB-BAR on all three …"*

and F98 banks it as a ceiling: *"the F47 features have a measured, already-banked
ceiling of +0.0269."* **So the registered prior for this exact feature family is
`identifiability REAL, conversion SUB-BAR` — which is band B — with the conversion
ceiling `+0.0269 / +0.0104` sitting *below* `K-NET`'s `+0.030` bar on both C09
datasets.** Two further facts sharpen it against C09: those are **raw-arena**
numbers, and F113 measured 9 of 9 raw-space positives failing to transfer to head
space, so the head-space expectation is *lower*, not higher; and the closest
measured analogue inside this very arena runs the same way — F113: *"any FITTED
relation score over head keys memorises the bank (in-sample pair AUC 0.9999) and is
WORSE than the plain cosine on held-out pairs (d_AUC +0.1572/+0.2302 raw →
−0.0643/−0.1294 head, 30/30 fold cells)"*.

> **Band B is therefore this design's pre-declared expectation, not its surprise.**

**The three bands, stated as functions of the same rules §9 uses (round-2 I-1):**

> **Band A — `K-FELDMAN` fires** (Holm-adjusted `p ≥ α` on at least one dataset at
> every `τ`). No incremental structure survives conditioning. **KILL.** When it fires
> on **both** datasets at **both** `τ`, the additional statement *"H-MEMORISATION is
> consistent with this object"* is published; otherwise the KILL stands with no
> Feldman claim.
>
> **Band B — `K-FELDMAN` clears at some `τ` but `K-NET` or `K-DEG` fires there.**
> Identifiability without legal conversion. **KILL, under the F98 epitaph and the
> `+0.0269` ceiling.** **Explicitly NOT a confirmation of H-MEMORISATION** — it is a
> conversion failure, and the record must say so.
>
> **Band C — some `τ` at which `K-REACH`, `K-FELDMAN`, `K-NET` and `K-DEG` all
> clear on both datasets, and every HALT gate passes.** ⇒ `CONTINUE` (§9), scoped to
> that `(τ, k)` cell and subject to §11's Stage-1 precondition.

**Upper-bound caveat.** `AUC_strat` conditions on a stratum built from the item's own
label-blind `|score|` and `pred_purity`, so it is not a gold-conditioned upper bound
— but a deployed operator would have to locate the region without knowing which items
are in `P`. `ΔAUC > 0` is **necessary, not sufficient**, for a buildable operator.

### 6.2 `K-DEG` — the threshold-degeneracy control (round-2 H-5)

The campaign has twice measured this operator class collapsing into a bare threshold
move. F96 made it a **standing gate**, verbatim: *"ANY VARIANT MUST FIRST PASS THE
DEGENERACY CONTROL, and that is now a standing gate, not a suggestion … Any operator
agreeing with a pure global threshold shift on the bulk of items is DEAD BY THE
EXISTING THRESHOLD BAN regardless of its Dacc — it is a dead lever in an item-level
costume."* F98's DEG-A measured `0.9570` (HateMM) / `0.9508` (EN) against a frozen
`0.95` kill line, with bare `THRESH_best` scoring `+0.0188` — more than the learned
operator; F98's `ban_scope` (d) closes re-deriving that observation; and **C09's own
registry dedup boundary excludes *"thresholding"***. `D-FELDMAN`'s `BASE` is led by
`|score|`, so `S` can be a threshold move in costume.

**Three frozen degenerate twins, each of size exactly `k`, per (dataset, seed, τ, k):**

- **`THRESH-SYM`** — the `k` items with the smallest `|score_{i,s}|`.
- **`THRESH-BEST`** — the hindsight-best one-sided band: of the two candidate sets
  {`k` items with the smallest positive `score`}, {`k` items with the largest
  negative `score`}, the one with the higher `net_s`. Hindsight-optimal by
  construction, which makes the control conservative.
- **`FIXK`** — the set of items whose deployed decision flips under the best
  alternative fixed neighbourhood size `k' ∈ {1, 2, 3, 5, 7, 10, 15}` in the same
  rank-weighted vote, truncated to the `k` with the largest `|Δscore|`. (F98's DEG-B
  twin.)

**Agreement** = `|S ∩ twin| / k`, averaged over seeds. **`K-DEG` fires — and the
verdict is KILL — if agreement `≥ 0.95` with **any** twin on **either** dataset at
the cell under consideration.** The `0.95` line is the campaign's own, adopted
unchanged. Anchor for the reader: F113 measured head-space `THRESH_best` at
`+0.0041` (from raw `+0.0148`), so the degenerate lever is worth ~1 item per 244 in
this space.

### 6.3 Controls attached to the discriminator

- **`SHUFFLE-POP` (pinned).** A **uniform random permutation of the per-item target
  vector over the `D-FELDMAN` analysis set** — `P_τ ∪ CONFIG-MATCHED-CORRECT`, i.e.
  exactly the rows the estimator sees, **not** all `n` (round-2 I-3) —
  `numpy.random.default_rng(20260801)`, `200` draws, applied identically to `FULL` and
  `BASE` on the same folds and the same frozen strata. **Evaluated at both `τ_0` and
  `τ_hi`.** v2's claim that the permutation *"preserves all configuration marginals"*
  was false and is withdrawn.
  **What it tests:** that the split machinery, the stratified-AUC estimator and the
  bootstrap do not manufacture signal from the target's marginal alone.
  **What it cannot test:** it is **blind to feature-side leakage** — that job belongs
  to `GATE-BLIND`, a structural gate, not a statistical one.
  **HALT rule:** the permutation-null mean of `AUC_strat(FULL)` must lie in
  `[0.45, 0.55]` at **both** `τ` on **both** datasets; outside that band the estimator
  is leaking and no verdict is published.
- **`UNSTABLE-POP`.** `D-FELDMAN` re-run with the target redefined as unstable errors,
  negatives unchanged. If stable and unstable populations are equally predictable,
  "stability" carries no information and the registry claim's own premise is empty —
  reported as a mechanism finding regardless of the verdict. **Data-independent power
  rule applied identically to both datasets:** emit `CONTROL_UNDERPOWERED` iff
  `n_unstable < 20` **or** the two-sided bootstrap CI width on `ΔAUC` exceeds `0.30`.
  Non-gating.
- **`RANDOM-POP`.** A size-matched random sample of query items in place of the stable
  inversions, `default_rng(20260801)`, 200 draws; every reported quantity recomputed
  against it. This prices **F88 null (3)**: HateMM memory-bank LOO curation at
  `+0.0016` against random deletion of the same size at `+0.0031 / +0.0000`,
  self-labelled *"Pregate-grade null (one rule, one proxy head/cell, single draw)"*.
  **Carried at its adjudicated weight:** the reopen's round 14 records that this is
  *"a val-sel loss and a final-epoch win, all under half a test item per seed, so
  'indistinguishable' is the exact reading"* — the curated rule is **not** established
  as worse than random, it is established as **indistinguishable from** random on a
  single draw, HateMM-only, on **train-row deletion**, a different population and
  operator from C09's. A headwind to price, not a closure. Non-gating.

---

## 7. Controls summary

`RANDOM-POP`, `CONFIG-MATCHED-CORRECT`, `SHUFFLE-POP`, `UNSTABLE-POP` (§6.3);
`K-DEG`'s three degenerate twins (§6.2); `DATA-DEFECT-OVERLAP` (§5.5).
`CONFIG-MATCHED-CORRECT` additionally supplies the break-exposure stratification, so
that *"constraining break exposure"* is measured rather than asserted.

---

## 8. Gates — split into HALT gates and reporting instruments (round-2 H-7)

### 8.1 HALT gates — a failure publishes **no** verdict

- **`GATE-FLOOR`.** The re-minted fold-head arena must reproduce the banked per-seed
  pooled deployed values **at 4 decimal places**, on 6/6 seeds, in **both** metrics:

  | | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | HateMM acc | `0.8884` | `0.8858` | `0.8858` |
  | HateMM macro-F1 | `0.8838` | `0.8811` | `0.8812` |
  | MHC-ZH acc | `0.8929` | `0.8895` | `0.8946` |
  | MHC-ZH macro-F1 | `0.8747` | `0.8710` | `0.8765` |

  Source: `scripts/analysis/headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`,
  `result.acc_deployed` / `result.mF1_deployed`, re-read at drafting time.
  **4 dp and not beyond** — the banked anchors are 4-dp values and asserting past them
  is the engineering-HALT trap that killed three C01 runs. DET-3 Tier B entitles this
  run to gate against banked JSON because DET-1/DET-2 are honoured and the banked
  outputs carry their own runtime block.
  **Version/node residual:** the banked arena's `meta.runtime` is compared against
  this run's; any difference in the interpreter/numpy/scipy/sklearn/torch quartet or
  the node is reported as a documented `RUNTIME_DRIFT` flag with the re-run path
  named. A `GATE-FLOOR` failure under `RUNTIME_DRIFT` is an **engineering HALT** with
  a diagnose-repair-resubmit path, not a scientific result.

- **`GATE-PARITY-FOLD`** *(renamed from v2's `GATE-PARITY-λ0`, whose "bit-for-bit at
  4 dp" was the oxymoron round 1 flagged — I-6b)*. The re-minted deployed vote must
  reproduce the banked per-fold `result.fold_acc_deployed` arrays, **equal at 4
  decimal places**, for all 30 fold-cells.

- **`GATE-BLIND`** *(strengthened — round-2 C-1)*. A per-feature manifest naming, for
  every feature in `BASE` and `FULL`, the exact arrays it indexes. Enforced
  structurally: the feature builder's signature admits **neither** the query-side gold
  label array **nor** any target-derived array, and **three** arrays are wrapped in
  read-counting guards for the whole feature-construction phase — `lab_query`,
  `is_inversion[seed]`, `is_stable_inversion`. **Emitted as integer counts, not
  booleans:** all three must be exactly `0`; `bank_label_reads` is reported as its
  (nonzero, legal) integer; per-feature array-touch lists are emitted in full. Any
  nonzero count on the three guarded arrays **HALTs**.

- **`GATE-LEDGER`.** A runtime access ledger reporting, as literal integer counts:
  test-split path opens (must be `0`), test-label materialisations (must be `0`),
  dev-split **path** opens (expected nonzero — the six fidelity heads and the trainlog
  reader — reported with its declared expected value), dev **label** materialisations
  into any decision quantity (must be `0`). `headspace_mint.py:106-116` installs a
  global `torch.load` guard raising on any path containing `test_seen` or `/test`; the
  driver adds an `open()`-level guard with the same predicate over the whole job.

- **`GATE-NESTED`.** The `D-FELDMAN` partition is asserted equal to the frozen arena
  fold partition, and for every scored item the assertion "this item's arena fold was
  excluded from the model that scored it, and all three of its seed-rows were excluded
  together" is checked and emitted as a **per-item check count** that must equal the
  item count.

- **`GATE-SELFTEST`.** `net_s == n · Δacc_s` asserted exactly for every seed ×
  dataset × operating point × `τ`.

- **`GATE-ZEROOP`** *(new — I-5, C09's CAL-2 analogue)*. With `S = ∅` (`k = 0`) the
  treatment harness must return the floor exactly: `Δacc = 0.0000`, `ΔmF1 = 0.0000`,
  `net_s = 0` for every seed and dataset. This checks the *treatment* path, which
  `GATE-FLOOR` does not.

- **`GATE-ARENA`** *(band edges pinned — I-6c, C02 `ARENA2` convention)*. Pooled native
  accuracy must satisfy `majority_rate + 0.02 ≤ acc ≤ 0.98` on both datasets:
  HateMM majority `0.5995` (`posrate_bank = 0.4005`) ⇒ band `[0.6195, 0.98]`;
  MHC-ZH majority `0.6891` (`posrate_bank = 0.3109`) ⇒ band `[0.7091, 0.98]`.

### 8.2 Reporting and scoping instruments — these do **not** gate the verdict

- **`GATE-DEVFID`.** `headspace_fidelity.py` run unmodified on the six `fold == -1`
  heads, already inside the 36-head budget. Banked reference:
  `B_fid_abs_3seedmean` `0.0093` (HateMM) / `0.0086` (MHC-ZH),
  `STOP_RULE_TRIGGERED: false` on both
  (`scripts/analysis/headspace_fidelity{,_zh}_OUT.json`, re-read at drafting time).
  **Reported, not a HALT.** It measures proxy↔floor fidelity **across hardware**;
  C09's entire arena is CPU-minted, so F88's binding caveat is satisfied by
  construction and cross-hardware fidelity does not gate the internal comparison. A
  `STOP_RULE_TRIGGERED == true` on either dataset publishes the verdict with a
  `PROXY_FIDELITY_FLAG` and a scope note.
- **`GATE-SEED`.** The per-seed inversion sets are emitted **in full**, as sorted item
  indices, so the 3-seed intersection is independently recomputable from the published
  artifact without re-running anything. An emission, not a predicate.
- **`GATE-NULL`.** HateMM train row `355` (`hate_video_95`, label `1`) carries an
  exact-zero vector in **both** streams; MHC-ZH has **no** structural-zero row.
  *(Re-measured this session directly from
  `data/CLIP_Embedding/{HateMM,MHC_zh}/train_*.pt`: HateMM zero-img `[355]`, zero-txt
  `[355]`; MHC-ZH `[]` and `[]`.)* The contract:
  1. the **primary** run is **with-null**, on the full `n = 744`, which is the arena
     the banked floors and the `22.3` figure are defined on;
  2. a **remove-null sensitivity** drops item 355 from the query set **and** from every
     bank, with its own recomputed floors and its own recomputed bar
     `0.030 × 743 = 22.29`;
  3. the requirement is agreement on the **verdict and on every K-rule outcome**, not
     on metric values — `n` moves from 744 to 743 and every rate changes, which is why
     v1's "agree on every metric" was incoherent. A disagreement on one item out of
     744 is published as a first-class finding and the verdict is scoped to it;
  4. **In head space the zero row is not zero.** The head applies learned projections
     with biases and `mlp[:-2]` applies no final normalisation, so the C01/C02
     raw-space contract *"must remain exact-zero in every derived array"* **does not
     transfer** and is not asserted. What is asserted is that item 355 is treated
     identically to every other item by every code path, and its per-item fate is
     reported explicitly.

---

## 9. Decision rule — frozen, two-valued, exhaustive

Let `τ ∈ {τ_0, τ_hi}` and `k ∈ {|P_τ|, round(1.5|P_τ|), round(2|P_τ|)}`.

**`CONTINUE`** iff there exists a `τ` such that **all** of the following hold:

1. **`K-REACH` clears at `τ`** — `Δacc_{O1} ≥ +0.050` **and** `ΔmF1_{O1} ≥ +0.050` on
   **both** datasets.
2. **`K-FELDMAN` clears at `τ`** — the Holm-adjusted `p` on `ΔAUC` (§5.2) rejects at
   `α = 0.05` on **both** datasets (IUT across datasets, Holm over the two `τ` within
   each dataset).
3. **`K-NET` clears at `τ`** — there exists a **single** `k` such that, on **both**
   datasets simultaneously, `mean_s net_s ≥ 22.3` (HateMM) / `≥ 17.4` (MHC-ZH)
   **and** `mean_s ΔmF1_s ≥ +0.030`.
4. **`K-DEG` does not fire at that `(τ, k)`** — agreement of `S` with each of
   `THRESH-SYM`, `THRESH-BEST` and `FIXK` is `< 0.95` on **both** datasets (§6.2).
5. **`SHUFFLE-POP`'s permutation-null mean `AUC_strat(FULL) ∈ [0.45, 0.55]`** at that
   `τ` on both datasets, and **all eight HALT gates of §8.1 pass**. *(The §8.2
   instruments are reported and scope the verdict; they do not gate it.)*

**`KILL`** in every other case. `KILL` and `CONTINUE` are complements by construction.

**Every quantity is computed and reported at both `τ` and all three `k`, on both
datasets, regardless of which rule fires**, so the record carries the full grid.

**Which rule fired is recorded, and the KILL is scoped by it:**

- **`K-REACH` fired at `τ_0`** ⇒ the KILL closes **every** confidence threshold
  `τ ≥ 0` on both metrics, by arithmetic (§4.3).
- **`K-REACH` cleared at `τ_0` but failed at `τ_hi`** *(the case v2 had no bullet for
  — round-2 H-2)* ⇒ the co-primary is closed **on reach alone**. The finding is that
  the registry's own *"high-confidence"* restriction, taken at the median, does not
  contain enough items to reach the Stage-0 bar; `q_max` quantifies where it stops.
  **This closes nothing about identifiability or conversion at `τ_hi`** — both are
  still measured and reported there, and the record must not read the `τ_hi` numbers
  as adjudicated.
- **`K-FELDMAN`, `K-NET` or `K-DEG` fired** ⇒ the KILL closes **`τ ∈ {τ_0, τ_hi}`
  only** — the primary population and the registered high-confidence co-primary. It
  does **not** close arbitrary `τ`, because neither AUC nor precision is monotone in
  `τ`.

**A `CONTINUE` is tagged with, and scoped to:** the `τ` and `k` it cleared at;
`ROBUST` or `POINT_ESTIMATE_ONLY` from the bootstrap lower bounds; `NET_050_CLEARED`
or `NET_050_MISSED` against the reopen's secondary `37.2 / 29.0`;
`PROXY_FIDELITY_FLAG` if `GATE-DEVFID`'s stop rule fired; and the
`DATA-DEFECT-OVERLAP` enrichment.

**The raw arena, specified.** The identical battery is recomputed on the banked **raw
fused key space** — `X = l2n(concat(l2n(img_feats), l2n(text_feats)))`, 7168-d,
seed-free (`headspace_arena.py:7`, `mechnov_pairverify.py:124`), over the *same*
frozen 5-fold assignment and the same deployed top-20 vote — whose banked pooled
deployed accuracies are `0.8441` (HateMM) and `0.8480` (MHC-ZH)
(`headspace_arena_*_OUT.json`, `membership.raw_deployed_acc`). Because the raw space
is **seed-free there is exactly one "seed", so stability is undefined there**; the raw
leg prices the *single-pass inversion* population, is reported as such, and is
**confined to corroborating a KILL** — the only direction F113 permits. F113's caveat
is carried in full: *"**NOT established: that a raw-space NEGATIVE cannot be a
head-space positive** (F95's own limitation L1, untouched here); that any of this
transfers to TEST (all arenas query train-split items held out from their own head)."*
**No raw-arena number reaches the decision.**

**CAL-3, discharged (round-2 H-8).** CAL-3 is mandatory whenever a raw `Δ ≥ +0.010`
is reported, and requires the raw `Δ` beside *"the deployed space's own gold-cheating
ceiling for the same operator family"*, labelling the arm `RAW-ARENA ARTEFACT` if the
raw legal number exceeds the deployed oracle. The raw leg here will report `Δ` well
above `+0.010`. **The honest discharge: no deployed-space gold-cheating oracle is
banked for the "flip a nominated stable-inversion population" operator family** — the
ERRPAT oracles cover the threshold, curation, length-de-bias and stream families, none
of which is this one. **The CAL-3 comparison is therefore unavailable, and no raw `Δ`
is escalated in any direction**; the raw leg's only permitted use remains KILL
corroboration. The nearest available anchor, reported for the reader and not used as a
CAL-3 comparator, is F113's head-space `THRESH_best` `+0.0041`.

---

## 10. Scope of any verdict this A0 can produce

- A **KILL** closes the C09 Stage-0 oracle **under the frozen Stage-0 rule, at the
  `τ` values §9 scopes it to**. It is **not** an impossibility proof for
  encoder-level topology intervention: the probe is one feature set, one estimator
  family and one stratification, and a richer geometry might locate the region where
  this one cannot. Stated **now**, because C02's A0 had to retract exactly this kind
  of overclaim once (the v8 erratum) before it was re-stated correctly.
- A **CONTINUE** establishes only that the population is large enough and locatable
  enough to justify building an operator, scoped to its `(τ, k)` cell, and is void
  unless §11's precondition is met.
- **`O1` is a label-flip oracle over one nominated population, not the registry's
  "full-bank or representation-level oracle"** — an upper bound, used only in the
  closing direction.
- **`NET` prices a per-item selector, which is a strictly more capable operator than
  §11's global symmetric reshaper** (§3.2). It is an **optimistic upper bound** on the
  successor's conversion.
- **With MHC-EN out of scope the two-dataset requirement has zero slack.**
- **This A0 does not separate H-DATA-DEFECT from H-TOPOLOGY** (§5.5); it only prices
  the overlap.
- **CAL-5 runs against C09**: the Stage-1 operator this A0 would license is a
  channel-(a)/(d) object — it changes the map — and *"a channel-(a)/(d) arena result
  carries NO transfer warrant."* Any result here is an **arena** result about the
  fold-head **train** arena, never a prediction of deployed behaviour.
- Neither verdict touches the `+0.030 / +0.030` two-dataset target, which remains
  active and unmet.

---

## 11. The Stage-1 seam — named now, because a CONTINUE with no legal successor is worthless

The reopen's **first** quoted kill-risk, verbatim and in full:

> *"(i) any encoder-level pull of an inversion toward its right analogue is a
> label-using metric move => F75/NCA and section 1.3's +0.0286"*

with the reopen's own binding instruction attached: *"Section 1.3's bound must always
be quoted with R^2=0.027, r=+0.1642, slope CI [-0.0221,+0.1637] straddling zero,
MHC-ZH dev only - F114's standing instruction - and F114 forbids citing F107 as a
theory-level door-closer."* The `+0.0286` is carried here **only** with those four
qualifiers and is not used as a bound anywhere in this design.

**The successor this A0 would license, named concretely.** A **global, symmetric,
train-label-supervised reshaping of the head map** `φ₀ → φ′`, in which the
stable-inversion set is identified **once, offline, on train items only**, and enters
the *training objective* as a region-targeted term. At inference `φ′` is applied
identically to every query; the stability statistic, the probe and region membership
are never consulted at query time. That is what makes it symmetric under
`LITSWEEP3_DATA_CENTRIC.md:82`, and it is the only shape §3's HALT boundaries leave
standing.

**Is that F75's object? Partly, and the honest accounting is:**

- **F75's `ban_scope`, verbatim:** *"head-loss swaps of the triplet+BCE hybrid toward
  vote-consistent (NCA/soft-kNN), contrastive (SupCon), or mixup-BCE objectives at 7B
  frozen-encoder feature scale; tau/alpha retunes = tactics, banned."* A
  region-targeted term added to the deployed triplet+BCE hybrid is **none of the three
  named objectives**, so it is outside the ban's letter.
- **But F75 is also *"the first measured negative for
  trained-reshaping-unlocks-oracle-headroom"*, and `LITSWEEP5_COMPLETENESS.md §2`
  argues in its own adversarial self-rebuttal that F75's *mechanism* — symmetric
  reshaping does not convert selection-locked headroom — *"generalizes past its
  named-loss letter"*.** That is an argument, not a measurement, and LITSWEEP5 offers
  it against its own challenge; it is nonetheless the correct headwind and is
  registered.
- **C09's own dedup boundary independently forbids the cheap version:** *"not …
  hard-example weighting alone."* A region-weighted triplet+BCE **is** hard-example
  weighting, so the successor must be more than that or it fails its own registry
  boundary before it fails F75.
- **The counterweight, at its corrected one-sided-use weight.**
  `NCA_FORENSIC_RECON.md:110` verbatim: *"Ruling: F66 does NOT bind trained-space
  reshaping. The cell is not F66-dead — it is legitimately un-measured."* The same
  record's `:112` prices that cell at *"honest P(≥+3) stays 2-4%"*, and the cell it
  unblocked was subsequently run and killed as F75. Both halves are carried.
- **And `NET` does not price it** (§3.2): the successor is strictly less capable than
  the per-item selector `NET` measures, so a `K-NET` pass is not evidence the
  successor converts.

**Pre-registered consequence, binding on this A0's own verdict.** A `CONTINUE`
**does not carry a Stage-1 licence.** Stage-1 entry requires, as a precondition
written here rather than negotiated later, that a proponent name an operator that is
(a) global and symmetric at inference, (b) not one of F75's three named objectives,
(c) not hard-example weighting alone, and (d) accompanied by a fresh ban-scope
adjudication against F75, F66 and F98. **If no such operator can be named at Stage-1
entry, the CONTINUE is void and C09 closes with no further spend.**

---

## 12. Repair ledger — the 19 round-2 findings

| # | round-2 finding | repair | where |
|---|---|---|---|
| **C-1** | feature 11 reads other items' gold labels; `GATE-BLIND` blind to derived target arrays; the "training-fold only" exemption false for fitting rows | **feature 11 DELETED** (`FULL` 13 → 12 features); the blanket label-blindness claim now holds; **`GATE-BLIND` re-specified** to guard three arrays — `lab_query`, `is_inversion[seed]`, `is_stable_inversion` — each with an integer read count that must be `0` | §5.2, §8.1 |
| **H-1** | `τ_hi` not a well-defined number | **pinned**: `τ_hi = median over i ∈ P_0 of c_i`, `c_i ≡ mean_s |score_{i,s}|`, the same per-item scale as §5.3; `q25/q75/q_max` inherit it | §4.3 |
| **H-2** | `τ_hi` can die at `K-REACH`; no scope bullet for it | third scope bullet added stating the reach-only closure and that **nothing about identifiability or conversion at `τ_hi` is adjudicated**; both are still measured and reported there unconditionally | §4.3, §9 |
| **H-3** | F97/F98 mis-registered as a "null"; the real prior is band B below the bar | **F97's HONEST POSITIVE DATUM quoted verbatim** (`+0.0269 / +0.0104 / +0.0182`); F98's `+0.0269` ceiling registered; raw-arena scope and F113's 9/9 non-transfer noted as making it *more* adverse in head space; **band B declared the pre-registered expectation** | §6.1 |
| **H-4** | F98 `ban_scope` (b) unadjudicated though it names `NET`'s object | **§3.2 added**: ban quoted verbatim, object conceded, `H-L4` named as the foreclosure, and the scientific consequence stated — **`NET` is an optimistic upper bound on the successor**, so a failure is a fortiori a KILL and a pass licenses nothing; clause (d) linked to `K-DEG` | §3.2 |
| **H-5** | no threshold-degeneracy control | **`K-DEG` added and made verdict-gating**: three frozen twins (`THRESH-SYM`, `THRESH-BEST`, `FIXK`), agreement `≥ 0.95` fires, F96's standing-gate text and F98's DEG-A precedent quoted, F113's head-space `THRESH_best +0.0041` given as anchor | §6.2, §9 |
| **H-6** | `K-FELDMAN` not computable; Holm on quantiles; family over-large | **bootstrap ASL defined** with the `1/(B+1)` floor; **Holm over the 2 `τ` within each dataset**; **datasets combined as an IUT with no correction**; adjusted one-sided bounds reported alongside | §5.2 |
| **H-7** | §8 clause 4 contradicts §7.2's own gate definitions | gates **split explicitly** into eight HALT gates (§8.1) and three reporting/scoping instruments (§8.2); §9 clause 5 names only the HALT set | §8, §9 |
| **H-8** | CAL-3 declared binding then omitted | **CAL-3 discharged on the record**: no deployed oracle is banked for this operator family, so the comparison is unavailable and no raw `Δ` is escalated; F113's `+0.0041` given as a non-comparator anchor | §9 |
| **I-1** | §6's bands and §9's rule disagree | bands **restated as functions of the same `K-` rules** §9 uses | §6.1 |
| **I-2** | `AUC_strat` under-specified in four ways | all four pinned: **strata per (dataset, seed) cell**, per-dataset value = mean over seed cells; **pooling unit = row within cell**; **strata frozen from the full sample** and held fixed across resamples and permutations; **zero-positive/zero-negative strata get weight 0**, and an all-single-class cell **HALTs** | §4.4, §5.2 |
| **I-3** | `SHUFFLE-POP` domain and evaluation point unstated | domain pinned to the **`D-FELDMAN` analysis set**, not all `n`; **evaluated at both `τ`** and the band checked at both | §6.3, §9 |
| **I-4** | six provenance defects | `(88 %)` interpolation **removed** and the registry line quoted verbatim; `ERRPAT_HateMM` line numbers corrected to `:130 / :131 / :134 / :135 / :141`; `ERRPAT_MHC-ZH:234-235` quoted; F113's caveat carried **in full** including the TEST clause; kill-risk (i) quoted **in full** with F114's four qualifiers; ERRPAT's *"3-seed mean vote"* qualifier restored | §4.1, §4.2, §4.4, §5.2, §9, §11 |
| **I-5** | CAL-2 substitution wrongly called "strictly stronger" | claim **withdrawn** — the two are different objects — and **`GATE-ZEROOP`** added as C09's actual `FIXK_20` analogue (empty operator ⇒ exact zeros) | §2, §8.1 |
| **I-6** | four executability nits | (a) **mint naming pinned** to `mint_{ds}_s{seed}_f{fold\|full}.npz` in one directory, as `headspace_fidelity.py:66` requires; (b) `GATE-PARITY-FOLD` reworded to **"equal at 4 decimal places"**; (c) `GATE-ARENA` band edges pinned to C02's `majority + 0.02` / `0.98`, with both bands computed; (d) the extra `k = 50` and per-class top-1 faiss searches **declared** | §2, §8.1, §5.2 |
| **I-7** | submission preconditions unrecorded | both registry preconditions (C04 tranche terminated; main-dialogue authorization) added to the STATUS block and to the freeze record's precondition table | STATUS |
| **I-8** | budget prices mints only | **itemised budget table** covering the whole battery, ≈70 CPU-minutes, with the note that the bootstrap re-scores rather than refits | §2 |
| **I-9** | ranker fit on a filtered population, applied to all `n` | **declared**, with the reason the alternative was rejected, and the **composition of `S` by class** reported at every operating point | §5.3 |
| **I-10** | LITSWEEP5's conclusion omitted; no third hypothesis | (a) the *"Training labels cannot supervise…"* sentence **quoted** and answered with the arena's own error rates (`0.105–0.114` train vs the deployed test rate), with the residual headwind kept; (b) **H-DATA-DEFECT added as a third column**, `DATA-DEFECT-OVERLAP` added as a `$0` diagnostic with both defect populations **re-measured this session** (`243/579` ZH `<em>` rows; `39/744` HateMM whitespace-only rows), and the honest statement that **this A0 does not separate it** | §3.3, §5.5, §6.1, §10 |

---

*v3. No hash frozen, no config written, no code implemented, no namespace created,
no job submitted, no cache or test path opened, no metric or result produced. Zero
GPU, zero SLURM, zero Modal, zero teacher call.*
