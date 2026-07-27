# MECHNOV PAIR-VERIFY — $0 pregate on replacing the deployed kNN VOTE with a trained PAIR VERIFIER

**Date:** 2026-07-27 NZST · **Agent:** mechnov pair-verify · **Cost: $0** (CPU only, ≤8 threads,
**zero GPU, zero SLURM, zero Modal, zero training of any deployed arm**). Repo sha at freeze time
`074ea00` (working tree dirty). **Test-split contact: NONE** — this pregate loads only the
`train` split of each dataset and never opens `test_seen`.

**What this is.** A $0 pregate on a *mechanism-level* replacement of the deployed decision rule.
**What this is not.** Not a verdict, not a prereg, not a promotion. The arena is the banked **raw**
encoder key space, not the deployed trained-head space (see §2.1 and §6 for why, and what that
costs the reading).

---

## §1. PRINCIPLE AND DISTINCTNESS AUDIT

*(Written in full before any number in §3–§5 existed.)*

### 1.1 The diagnosis being treated

The deployed decision (`src/utils/metrics.py:262-301`, `src/model/evaluate_rac.py:405-465`) is

```
retrieval = faiss.IndexFlatIP over float32 L2-normalised keys, memory = own train split, top-20
vote      = Σ_i (2·lab_i − 1)·cos_i·w_i / Σ_i w_i,   w = [20, 19, …, 1]
decision  = predict 1 iff vote ≥ 0
```

The three ERRPAT reports (2026-07-26, all read in full before this record was written) establish
that this rule fails as a **local-class-prior estimator**, not as a coverage or boundary problem:

* MHC-ZH: median top-20 purity toward the true label **0.15**, median |vote| **0.7137**; the first
  same-gold-class train neighbour sits at **median rank 1.5** in the raw fused space (11 of 22 core
  errors at rank 1). *The right analogue is retrieved and then out-voted* (ERRPAT-ZH §3).
* HateMM: 24-27 of 26-28 errors have top-20 purity < 0.5 toward the truth; the vote inherits the
  bank's length-conditional class base rate, which runs 0.1096 → 0.5538 across word bins
  (ERRPAT-HateMM §2, §4.3).
* F89 (mechfix, 2026-07-27): **all five symmetric vote/geometry repairs are dead** — class-balanced
  quota degenerate (identical predictions on 215/215 and 149/149), CSLS inert, 1-D length excision
  inert, Ledoit-Wolf whitening negative.

### 1.2 The replacement principle

> **Retrieval stops deciding and starts nominating.** The neighbourhood is used only to propose
> candidate analogues. The decision is made by a **trained pair verifier** that scores each
> `(query, candidate)` relation in **difference space** — `[ |e_i − e_j| , e_i ⊙ e_j ]` — as
> "same-class-like" vs "cross-class-like". Supervision becomes **relational**: `n` item labels
> yield `~n²` pair labels.

The mechanistic bet is precise. The deployed vote's only relation function is the cosine, and it
aggregates by *counting* (rank-weighted). Under the ERRPAT diagnosis, counting is exactly the broken
step: the correct analogue is present at rank ~1 and is out-*counted* by a majority of wrong-class
neighbours. Verification replaces counting with per-relation adjudication — one strongly-verified
same-class analogue can outweigh nineteen weakly-verified wrong-class ones.

The function class matters for the audit. With L2-normalised `z`, `Σ_k z_ik·z_jk` **is** the cosine,
so **a linear model on these pair features strictly contains the cosine rule**. Failing to beat the
cosine control is therefore not an expressivity accident — it is the finding that no reweighting of
the coordinate-wise agreement/disagreement profile carries relational information the plain inner
product does not already carry.

### 1.3 Structural-distinctness audit (mandatory; all four arguments must hold)

**vs. F47 — per-item cross-channel selection (dead at all 3 supervision sources).**
F47's object was: *given an item, choose which channel/operator to trust for it.* Its declared
killer was a supervision degeneracy — "the RGCL head memorises train (LOO train acc 0.998) →
train-disagreement routing target degenerate → ANY train-fit selector has no dev-transferable
supervision" (F47 detail).

Three separations, each independent:
1. **We select nothing.** There is one key space, one candidate pool, one scorer. No channel, no
   operator, no branch is chosen per item. The verifier is applied identically to every
   `(query, candidate)` pair.
2. **The supervision target is different in kind and is not degenerate.** F47's target is *"is
   operator A correct on this item?"* — a property of an operator's behaviour, which collapses when
   the operator is ~perfect on the only labelled split. Our target is *"do these two items carry the
   same label?"* — a property of the **label pair itself**, exactly known for every train pair,
   independent of any operator's correctness, and impossible to make degenerate by memorisation.
3. **Same generalisation shape as kNN.** At inference the verifier has never seen the query item —
   precisely the position the deployed kNN vote is already in. It is not asked to predict its own
   reliability.

Also distinct from **P2/P2b** (MLLM neighbour-comparability rerank, epitaph "comparability ⊥
vote-correctness"): that judge was **label-blind** and scored a proxy quantity (are these
comparable?) that was measured orthogonal to whether the vote is right. The verifier is trained
**directly on the label-agreement target**, so it is correctness-aligned by construction rather than
by hope. **HOLDS.**

**vs. F75 — NCA/soft-kNN/SupCon/mixup loss reshaping (7/8 cells KS-arm-dead).**
F75 changed the *training objective of the embedding* so that the learned space would suit a fixed
cosine-kNN vote. Here **the embeddings are frozen and untouched** — no gradient reaches any encoder
or head — and the new object is a *post-hoc inference component* that changes the decision function,
not the representation. The two are also non-nested in function class: F75 searches over
`cos(f_θ(x), f_θ(x'))` for a re-learned `f`; the verifier searches over `g(|z−z'|, z⊙z')` for a
fixed `z`, which includes non-inner-product terms (`|z−z'|`) and, in the MLP arm, nonlinearity.
Honest note: F75 is a *prior* against this family (reshaping to serve retrieval did not convert),
but it is not a measurement of a learned relation scorer. **HOLDS.**

**vs. F89 — symmetric eval-time vote/geometry operators (T1-T4, 0/5).**
T1-T4 are **fixed, label-blind, closed-form** transforms: a per-class quota (no parameters), a CSLS
hubness offset (no parameters), a Ledoit-Wolf whitener and a least-squares length direction (both
fitted, but **label-blind**, and both are *linear maps of the key space* whose composition with the
cosine is still a cosine in a re-metricised space). The verifier is **learned from labels** and
**relational** — its input is a pair, not a key. It cannot be written as `cos(Az, Az')` for any `A`
because of the `|z−z'|` block and (MLP arm) the nonlinearity.
*One honest sub-case, disclosed rather than glossed:* the **product half alone** of the plain
logistic arm is a label-supervised **diagonal re-metric**, which lives in the same shape as F89's
whitening family (though F89 never tested a label-supervised one). That overlap is a reason to keep
the logistic arm — if it lands on the cosine, it extends F89's closure to the supervised diagonal
metric, which was genuinely unmeasured. The MLP arm is outside that family entirely. **HOLDS.**

**vs. LP / F63 — label propagation over the kNN memory graph (KILL, all 3 datasets).**
LP is **transductive**: it diffuses labels multi-hop over a graph whose nodes include the evaluation
items. Here (i) the verifier is fitted on pairs with **both endpoints inside the fitting folds** —
no held-out item participates in any fitted pair; (ii) inference is strictly **one-hop**: the
verifier sees `(query, bank item)` and nothing propagates between bank items or between queries;
(iii) no unlabelled node ever acquires a label. **HOLDS.**

**Two further bans checked.** This is not *"kNN-vote-pool expansion via pseudo-labels"* (banned) —
the bank is unchanged, no pseudo-label is created, no row is added or deleted. It is not the
*"MLLM-scores-as-training-signal"* family — no MLLM is involved, and the training signal is the
gold train labels the pipeline already uses. It uses **no cross-dataset data** (each dataset's own
train split only, per the 2026-07-14 user veto).

**No argument fails. The audit clears; the pregate proceeds.**

### 1.4 What a pass would mean, and what it would not

A pass here licenses **one thing only**: promotion to a formal ceremony in which the verifier is
integrated on the *trained head* space with a pre-registered design and independent review. It would
not be a performance claim. A kill closes the axis at $0.

---

## §2. FROZEN DESIGN AND BARS

Implementation: `scripts/analysis/mechnov_pairverify.py`,
**sha256 `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d`** (455 lines), frozen
before any real-data treatment number was computed. It imports the deployed-vote replay from
`scripts/analysis/mechfix_ops.py`, **sha256 `635c1312…c83fc8d`** — the F89-frozen operator file that
passed 15/15 floor-parity gates at 4 dp — so the comparison floor is machinery already verified
against the recorded run anchors.

### 2.1 Arena: raw banked spaces, and why (a limitation stated up front)

The verifier is fitted and read in the **banked RAW encoder key spaces**, which are
seed-independent:

| space | construction | dim |
|---|---|---|
| **fused** (**PRIMARY**) | `L2norm( concat( L2norm(img_feats), L2norm(text_feats) ) )` — reproduces the errpat raw-encoder control | 7168 |
| text (secondary) | `L2norm(text_feats)` | 3584 |
| img (secondary) | `L2norm(img_feats)` | 3584 |

Feature caches: HateMM `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` (n=744, pos-rate 0.4005), MHC-ZH
`Qwen2.5-VL-7B-Instruct-LoRA_HF` (n=579, 0.3109), MHC-EN `Qwen2.5-VL-7B-Instruct_HF` (n=549,
0.3060) — the deployed encoder per dataset, **train split only**.

**Why not the trained head space.** F47 measured the RGCL head's leave-one-out accuracy on its own
train split at **0.998**. In head space, same-class train pairs have been explicitly pulled together
by the triplet objective, so a verifier fitted there would be fitted on a geometry that has already
memorised the fitting labels, and its pair-AUC would measure memorisation rather than a transferable
relation. The item-disjoint protocol below controls for the *verifier's* overfitting but cannot undo
the *embedding's*. **The raw space is the honest arena for a pregate; the price is that a raw-space
result does not transfer automatically to the deployed head space in either direction.** That is
stated as limitation L1 and is why a pass here buys a ceremony, not a claim.

### 2.2 Item-disjoint protocol (non-negotiable)

`StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over **train items**.
For each fold: *fitting folds* = the other 4; *held-out* = this one.

* **Fit set** = all unordered pairs `(i,j)`, `i<j`, with **both** endpoints in the fitting folds
  (capped at 150 000, seeded subsample `RandomState(0+fold)` — binds on HateMM only).
* **Eval set** = every `(held-out item) × (fitting-fold item)` pair. **No pair with a held-out
  endpoint is ever seen at fit time.** This mimics the deployed situation exactly: an unseen query
  against a bank of seen items.
* PCA (`n_components=256`, `svd_solver='full'`) is fitted **on fitting-fold items only**, then keys
  are L2-renormalised; pair-feature standardisation statistics are computed **on fitted pairs only**.
  Dimension reduction is a tractability measure (7168-d keys → 14336-d pair features × 177k pairs
  does not fit in memory); **the load-bearing cosine control is computed in the FULL raw space**, so
  reduction can only handicap the verifier, never flatter it. The reduced-space cosine is reported
  alongside for completeness.

### 2.3 Arms — all declared here, no post-hoc arms

**Verifier input** `φ(i,j) = [ |z_i − z_j| , z_i ⊙ z_j ]` (512-d), symmetric in `(i,j)`.
**Verifier target** `y = 1[lab_i == lab_j]`.

| axis | levels | status |
|---|---|---|
| model | **MLP** (1 hidden layer, 128 units, ReLU, Adam lr 1e-3, wd 1e-4, batch 1024, **30 epochs fixed, no early stopping**, torch seed 0) · **logistic** (sklearn L2, C=1.0, lbfgs, max_iter 1000) | both **PRIMARY** |
| space | **fused** · text · img | fused **PRIMARY**; text/img **SECONDARY** |
| dataset | **HateMM** · **MHC-ZH** · MHC-EN | HateMM/ZH **PRIMARY**; EN **SECONDARY** |
| aggregation | **max** verified score per class · mean of top-3 | max **PRIMARY**; mean3 **SECONDARY** |

**Primary read = 2 datasets × 1 space (fused) × 2 models × 1 aggregation (max) = 4 cells.**
The full battery is 3 datasets × 3 spaces × 2 models × 2 aggregations = **36 end-to-end cells**
(and **18 pair-AUC cells**, since aggregation does not affect control 1); the **32** non-primary
end-to-end cells are secondary and cannot carry a promotion on their own. Declaring the primary cut
here is the multiplicity control: across 36 cells an isolated `+0.010` is expected by chance, which
is what the fold-sign requirement in bar 2 exists to filter.

**Decision rule under test.** For a held-out query `q`, retrieve the top **`m=10` of each class** from
the fitting-fold bank by full-space cosine (20 candidates, the deployed budget), score each pair with
the verifier to get `p_same(q,·)`, set `s_c = max_{j: lab_j=c} p_same(q,j)` (primary) and predict
`1` iff `s_1 ≥ s_0`.

### 2.4 Controls and kill bars — frozen before any real-data number

1. **Cosine control (load-bearing).** Held-out **pair-AUC** (same-class vs cross-class) of the
   verifier must beat **full-raw-space cosine's pair-AUC on the identical eval pairs** by
   **≥ +0.03 AUC**, with the sign holding in **5/5 folds**. If it cannot beat cosine, it **is**
   cosine re-derived → **KILL**. (Recall §1.2: the linear hypothesis class contains cosine, so this
   is a nested-model test.)
2. **End-to-end read.** Pooled held-out-item accuracy (each train item held out exactly once) of the
   verification rule vs the **deployed top-20 rank-weighted signed-cosine vote computed on the same
   held-out items over the same fitting-fold bank** (LOO form, replayed by the F89-frozen
   `mechfix_ops.deployed_vote`). **Bar: ≥ +0.010 acc on ≥1 dataset, with 5/5 folds Δ ≥ 0 and ≥3/5
   strictly positive.** Else **KILL**.
   * **Control 2b — same-shape cosine control (declared now, not post-hoc).** The identical rule
     (top-10 per class, max per class, `s_1 ≥ s_0`) scored by **cosine instead of the verifier**.
     This isolates the verifier's contribution from the retrieval/aggregation *shape*; F89's T1
     showed a per-class quota can be degenerate on its own. A win over the deployed vote that is not
     also a win over 2b is a win for the aggregation shape, **not** for verification, and is
     reported as such.
3. **Mechanism read.** Among held-out items the deployed vote gets **wrong** whose nearest
   same-gold-class bank item sits within **rank 5** (the ERRPAT pathology population): what fraction
   does verification fix, and how many deployed-correct items does it break? Report the **exchange
   rate** (fixed / broken).
4. **Class-balance sanity.** Verification predictions must not collapse to the majority class:
   report the positive rate of the decision and of the raw pair predictions at 0.5.

### 2.5 Machinery validity (positive control, run before the freeze on synthetic data)

Before freezing, the harness was exercised on a synthetic 200×64 problem whose class signal is a
mean shift that the plain cosine largely misses. It returned pair-AUC cosine 0.5853 → verifier
0.8596 (MLP) / 0.8724 (logistic), 5/5 fold signs positive, end-to-end +0.12 to +0.15 acc over the
deployed vote. **The harness is therefore not structurally incapable of returning a positive**; a
null below is a property of the data, not of the code. No real-dataset number was computed before
the sha above was frozen.

<!-- EVERYTHING BELOW THIS LINE WAS WRITTEN AFTER THE RUN -->

---

## §3. RESULTS PER CONTROL

Every number below is re-read at report time from
`scripts/analysis/mechnov_pairverify_{hatemm,zh,en}_OUT.json` via
`scripts/analysis/mechnov_pairverify_report.py`, 4 dp. All 22 declared cells ran; none was dropped.

**Execution note (provenance).** The first execution of the frozen module's own `main()` was killed
by **SIGTERM (exit 143)** part-way through — the login node reaps long-lived non-SLURM processes —
losing the cells that had not yet been serialised. The remaining cells were re-run by
`scripts/analysis/mechnov_pairverify_runner.py`, which **imports the frozen module unmodified**
(it asserts sha256 `77b0defd…8b7240d` before running) and calls its frozen `run_space` one
`(dataset × space)` cell at a time, serialising each immediately. **No arm, constant, seed, fold or
bar changed**; only the process boundary did. Cells completed in 60-100 s each with no further reaps.

### 3.1 CONTROL 1 — cosine control: **PASS, decisively. This is not cosine re-derived.**

Held-out pair-AUC (same-class vs cross-class) on the identical eval pairs, 5-fold mean.
Bar: **≥ +0.03 over full-raw-space cosine, 5/5 fold signs.**

| dataset | space | cosine (full) | cosine (PCA) | **MLP** | **Δ vs cos** | signs | logistic | Δ vs cos | signs |
|---|---|---|---|---|---|---|---|---|---|
| **HateMM** | **fused** | 0.5843 | 0.6433 | **0.7753** | **+0.1910** | +++++ | 0.7135 | **+0.1292** | +++++ |
| **MHC-ZH** | **fused** | 0.5123 | 0.5738 | **0.7748** | **+0.2625** | +++++ | 0.6810 | **+0.1687** | +++++ |
| MHC-EN | fused | 0.5057 | 0.5375 | 0.7009 | +0.1952 | +++++ | 0.6467 | +0.1410 | +++++ |
| HateMM | text | 0.5732 | 0.7156 | 0.8251 | +0.2519 | +++++ | 0.7699 | +0.1967 | +++++ |
| HateMM | img | 0.5612 | 0.5717 | 0.6741 | +0.1129 | +++++ | 0.6128 | +0.0515 | +++++ |
| MHC-ZH | text | 0.5268 | 0.6288 | 0.7953 | +0.2685 | +++++ | 0.7013 | +0.1746 | +++++ |
| MHC-ZH | img | 0.4974 | 0.5119 | 0.6687 | +0.1713 | +++++ | 0.6170 | +0.1196 | +++++ |
| MHC-EN | text | 0.5124 | 0.5732 | 0.7170 | +0.2046 | +++++ | 0.6594 | +0.1470 | +++++ |
| MHC-EN | img | 0.5000 | 0.5054 | 0.6375 | +0.1375 | +++++ | 0.6144 | +0.1144 | +++++ |

**18/18 arm×space×dataset cells clear the +0.03 bar with 5/5 fold signs**, by margins of **4.3× to
8.8×** on the primary fused space. Per-fold ΔAUC never dips below +0.0431 in any cell.

Two things this settles. First, the load-bearing kill condition **does not fire**: the verifier is
emphatically not the cosine re-derived, even though (§1.2) the cosine lies *inside* its hypothesis
class. Second, it prices the deployed metric: **the raw fused cosine's own pair-AUC is 0.5843 /
0.5123 / 0.5057** — on MHC-ZH and MHC-EN the deployed retrieval metric is within 0.02 of
**chance** at telling a same-class pair from a cross-class pair.

### 3.2 CONTROL 2 — end-to-end: **FAIL on every dataset, every arm, every aggregation.**

Pooled held-out-item accuracy over all train items (each held out exactly once), against the
deployed top-20 rank-weighted signed-cosine vote replayed by the F89-frozen
`mechfix_ops.deployed_vote` over the same fitting-fold bank.
Bar: **≥ +0.010 acc on ≥1 dataset, 5/5 folds Δ ≥ 0 with ≥3/5 strictly positive.**

**PRIMARY cells (fused space × max aggregation):**

| dataset | deployed | ctrl 2b cos-shape | **MLP-max** | **Δ vs deployed** | fold signs | logistic-max | Δ vs deployed | fold signs |
|---|---|---|---|---|---|---|---|---|
| **HateMM** (n=744) | **0.8441** | 0.8024 | 0.8401 | **−0.0040** | −0−+− | 0.7137 | **−0.1304** | −−−−− |
| **MHC-ZH** (n=579) | **0.8480** | 0.8187 | 0.8014 | **−0.0466** | −−−−− | 0.7185 | **−0.1295** | −−−−− |
| MHC-EN (n=549) | 0.7796 | 0.7359 | 0.7650 | −0.0146 | +−+−− | 0.6995 | −0.0801 | −−−−− |

**0 of 4 primary cells clears the bar; all four are negative in the mean.**

Across **all 36 end-to-end cells** (3 datasets × 3 spaces × 2 models × 2 aggregations) exactly
**three** are positive in the 5-fold mean, all of them on HateMM and all with the MLP verifier:

| rank | cell | Δ vs deployed | fold signs |
|---|---|---|---|
| 1 | HateMM × **text** × MLP × **mean-top-3** | **+0.0094** | `++0+−` |
| 2 | HateMM × **text** × MLP × max | **+0.0081** | `−+0+−` |
| 3 | HateMM × fused × MLP × **mean-top-3** | **+0.0054** | `+−+++` |

All three use a secondary space and/or a secondary aggregation, **all three sit under the +0.010
bar**, and **all three fail the 5/5 fold-sign requirement** (each has at least one negative fold).
The other 33 cells are negative in the mean. **`Δ ≥ +0.010` is achieved by 0 of 36 cells.**

### 3.2b CONTROL 2b — the decomposition that explains the failure

The cos-shape control isolates *retrieval/aggregation shape* from *verification*:

| dataset (fused) | deployed vote | shape alone (2b) | **cost of the shape** | MLP-max | **verification's gain over the shape** |
|---|---|---|---|---|---|
| HateMM | 0.8441 | 0.8024 | **−0.0417** | 0.8401 | **+0.0377** |
| MHC-ZH | 0.8480 | 0.8187 | **−0.0293** | 0.8014 | **−0.0173** |
| MHC-EN | 0.7796 | 0.7359 | **−0.0437** | 0.7650 | **+0.0291** |

**This is the crux.** Replacing the deployed rank-weighted top-20 vote with "top-10 per class, take
the best" costs **−0.029 to −0.044 acc before any verifier is involved** — the deployed vote's
*aggregation over twenty neighbours* is itself carrying real signal, and a max over a per-class
shortlist throws it away. On HateMM and MHC-EN the trained verifier then recovers **+0.038 / +0.029**
of that loss over cosine-in-the-same-shape — a genuine, measurable contribution — but recovers
**less than the shape destroyed**. On MHC-ZH it does not even match cosine inside the shape.
So the honest sentence is: **verification is better than cosine at scoring a nominated pair, and the
nomination-plus-verification architecture is still worse than the vote it replaces.**

### 3.3 CONTROL 4 — class-balance sanity: **MLP passes, logistic FAILS (collapse)**

Positive rate of the end-to-end decision vs the bank's own positive rate:

| dataset (fused) | bank pos-rate | deployed | MLP-max | logistic-max | logistic mF1 |
|---|---|---|---|---|---|
| HateMM | 0.4005 | 0.4812 | 0.3911 | 0.2056 | 0.6612 |
| MHC-ZH | 0.3109 | 0.3489 | 0.2919 | **0.0604** | **0.5345** |
| MHC-EN | 0.3060 | 0.2605 | 0.2168 | **0.0237** | **0.4542** |

The **MLP arm does not collapse** in any of the 9 space×dataset cells (positive rate 0.29-0.39
against bank rates 0.31-0.40) — its nulls are honest nulls. The **plain logistic arm collapses
toward the majority class** on MHC-ZH and MHC-EN (positive rate down to 0.0237, and as low as
**0.0036** on MHC-EN × img × mean3), with macro-F1 falling to 0.4084-0.5345. **Control 4 therefore
fires on the logistic arm**, and its large negative end-to-end deltas must be read as a collapse
artefact rather than as a measurement of verification. Its *pair*-AUC is unaffected and healthy
(§3.1), and its pair-level prediction balance is fine (0.50-0.67) — the collapse is created by the
**max-per-class aggregation** of a scorer whose probabilities are not comparable across classes,
not by the verifier's discrimination.

### 3.4 Provenance and fit diagnostics

PCA retains **0.9459-0.9842** of key variance at 256 components in every cell. Fitted pairs:
150 000 (HateMM, capped from 176 715), 106 953 (MHC-ZH), 96 141 (MHC-EN); same-class rate in the
fit set 0.5192 / 0.5705 / 0.5731, matching the eval sets (0.5198 / 0.5715 / 0.5753) — no fit/eval
distribution shift. 1 item = 0.0013 (HateMM) / 0.0017 (ZH) / 0.0018 (EN), so the +0.010 bar is
**7.4 / 5.8 / 5.5 items** — this LOO read is *finer-grained* than the project's usual test reads.

---

## §4. MECHANISM READ — the fix/break exchange rate

Population definitions: "deployed wrong" = held-out items the deployed LOO vote misclassifies;
"pathology population" = the ERRPAT shape, i.e. deployed-wrong items whose **nearest same-gold-class
bank item sits within rank 5** by full-space cosine.

The ERRPAT diagnosis reproduces exactly in this arena: the **median rank of the first same-class
analogue is 1.0 over all items and 2.0-3.0 over the deployed vote's errors**, and **72-92 % of all
deployed errors are in the pathology population** (HateMM 88/116, ZH 79/88, EN 109/121). The right
analogue is retrieved and out-voted, on train items, in raw space, exactly as ERRPAT-ZH §3 reported
for test items in head space.

**Exchange rate, primary cells (fused × MLP × max):**

| dataset | deployed errors | **fixed** | **broken** | **net** | **exchange rate** | pathology fixed |
|---|---|---|---|---|---|---|
| **HateMM** | 116 | 54 | 57 | **−3** | **0.9474** | 48 / 88 = **54.6 %** |
| **MHC-ZH** | 88 | 31 | 58 | **−27** | **0.5345** | 29 / 79 = **36.7 %** |
| MHC-EN | 121 | 49 | 57 | **−8** | **0.8596** | 46 / 109 = **42.2 %** |

Best exchange rate anywhere in the battery: **1.1667** (HateMM × text × MLP × mean3, 49 fixed /
42 broken, net +7). Worst: 0.3750. **No cell reaches an exchange rate of 1.2.**

Three readings.

1. **Verification genuinely reaches the pathology.** It fixes **36.7-54.6 %** of exactly the errors
   the ERRPAT reports diagnosed as unreachable — the confident neighbourhood inversions where the
   correct analogue sits at rank ~1-2 and is out-voted. F89's T1/T2a/T3 fixed **zero** items on
   HateMM and ZH; T2b/T4, the only F89 arms that moved anything, fixed 1-3 per cell. This is the
   first operator measured in this campaign that reaches that population at scale.
2. **And it pays for every fix.** It breaks 42-58 previously-correct items to do it. Net is negative
   in **8 of the 9** primary/secondary MLP-max cells.
3. **The arithmetic is the same one F47, F66, F89 and ERRPAT-HateMM §3.2 all produced.** Image-stream
   substitution: fixes 11-14, breaks 40-43. F89 T2b/T4: fixes 1-5, breaks ≥ as many. Now a *learned
   relational* operator: fixes 31-54, breaks 47-58. The information to fix the core demonstrably
   exists in the space, and **every mechanism that surfaces it symmetrically pays for it at par or
   worse elsewhere.** Raising the fix count by an order of magnitude did not change the exchange rate.

### 4.1 Post-hoc diagnostic: why a much better relation scorer does not decide better

`scripts/analysis/mechnov_pairverify_diag.py` → `mechnov_pairverify_diag_OUT.json`.
**Post-hoc, adds no arm, promotes nothing**; it exists to explain the null. Fused space, 5-fold mean.

The obvious escape — "the pooled pair-AUC gain is an item-level hubness offset that cancels inside a
query" — is **measured false**:

| dataset | within-query AUC cosine | within-query AUC MLP | Δ | pooled Δ |
|---|---|---|---|---|
| HateMM | 0.6067 | 0.7639 | **+0.1572** | +0.1910 |
| MHC-ZH | 0.5363 | 0.7665 | **+0.2302** | +0.2625 |
| MHC-EN | 0.5228 | 0.7013 | **+0.1785** | +0.1952 |

The verifier orders *a single query's own candidates* far better than the cosine does. The gain is
relational, not a main effect. The two-way variance decomposition of the score matrix `S[query, bank]`
makes the same point structurally:

| dataset | scorer | query main | bank main | **interaction** |
|---|---|---|---|---|
| HateMM | cosine | 0.2899 | 0.4442 | **0.2659** |
| HateMM | MLP | 0.0396 | 0.0275 | **0.9329** |
| MHC-ZH | cosine | 0.3510 | 0.3657 | **0.2833** |
| MHC-ZH | MLP | 0.0649 | 0.1456 | **0.7895** |
| MHC-EN | cosine | 0.3101 | 0.3129 | **0.3770** |
| MHC-EN | MLP | 0.0624 | 0.1631 | **0.7745** |

**New structural fact.** In the deployed raw key space, **only 26.6-37.7 % of the cosine's score
variance is query×bank interaction** — 62-73 % is item-level offsets (how hubby the query is, how
hubby the bank row is). The deployed similarity is *mostly not a relation*. The trained verifier
inverts this to **77-93 % interaction**: it is 2.5-3.5× more relational, it is measurably better
within a query, and **it still does not decide better**. That is the sharpest instance of the
project's law-I yet recorded: the improvement is not in some adjacent quantity this time — it is in
*the very quantity the decision rule consumes*, and the conversion is still zero.

---

## §5. VERDICT

# **KILL.**

The pair-verification replacement is **dead as a decision mechanism** on all three datasets. It is
**not** killed for the reason the pregate expected.

**What passed.** Control 1, by 4.3-8.8×, 18/18 cells, 5/5 fold signs. The verifier is a real and
substantially better relation scorer than the deployed cosine — pooled pair-AUC +0.13 to +0.27,
within-query pair-AUC +0.10 to +0.23, interaction share of score variance 0.27→0.93. **It is not
cosine re-derived, and this record should not be cited as if it were.** The relational supervision
route (n labels → n² pair labels) does buy a genuinely better pairwise relation on frozen features.

**What failed.** Control 2, everywhere: **0 of 4 primary cells and 0 of 36 end-to-end cells** clear
+0.010 with consistent fold signs; the primary cells run −0.0040 (HateMM), −0.0466 (MHC-ZH),
−0.0146 (MHC-EN); the best number in the entire battery is **+0.0094** on a secondary space with a
secondary aggregation and a broken sign pattern. Control 4 additionally fires on the logistic arm
(collapse to positive rate 0.0237-0.0604 on ZH/EN). The mechanism read gives an exchange rate of
**0.53-0.95** in the primary cells and never exceeds 1.17 anywhere.

**Why it failed — two measured reasons, not speculation.**
1. **The aggregation the proposal discards was doing the work.** Nomination-plus-max costs
   **−0.0293 to −0.0437** before any verifier runs (control 2b). The deployed rank-weighted vote over
   twenty neighbours is a better aggregator of weak evidence than the best-verified single analogue,
   and verification recovers less than the switch destroys. The ERRPAT framing — "the correct
   analogue sits at rank ~1.5 and is out-voted" — is true, but the same averaging that out-votes the
   correct analogue on ~15 % of items is *protecting* the decision on the other ~85 %.
2. **Better relations do not become better decisions.** This is F47/F66/F89 arithmetic in a new and
   much stronger instance: a 10× increase in the number of core errors reached (31-54 fixed, versus
   0-5 for every F89 arm) produced **no** improvement in the exchange rate.

**Routing.** Do not promote to ceremony. Do not spend GPU. Specifically **do not** re-propose (a)
head-space pair verification as a rescue — the raw-space failure is an aggregation failure, and the
head space (LOO train acc 0.998) would make the pair-AUC read uninterpretable while leaving the
control-2b arithmetic untouched; (b) other pair-scorer architectures (bilinear, cross-attention,
siamese-with-margin) — the binding constraint measured here is not the scorer's quality, which
already beats cosine by 4-9× the bar, but the exchange rate, which the scorer's quality did not move;
(c) verifier-as-reranker-inside-the-vote without first pricing control 2b, since the shape cost is
the dominant term.

**What is worth carrying forward** (paper analysis, not performance): the variance decomposition in
§4.1. "The deployed retrieval similarity carries only 27-38 % of its variance in the query×bank
interaction; a trained relation scorer reaches 77-93 % interaction and +0.16-0.23 within-query AUC,
and the deployed decision does not improve" is a crisp, citable statement of why this family of
repairs is closed, and it is a stronger mechanism claim than F89's.

---

## §6. LIMITATIONS

1. **Raw space, not the deployed head space (L1, the load-bearing one).** Everything here is measured
   on banked raw encoder keys. A raw-space null does not logically entail a head-space null. The
   choice was deliberate (F47's LOO-0.998 memorisation makes head-space pair-AUC uninterpretable),
   and control 2b's shape cost — the dominant failure term — is a property of the aggregation, not of
   the space. But this is a pregate, not a verdict.
2. **Train-split LOO arena, not test.** The end-to-end floor is the deployed vote computed on
   held-out *train* items over a 4/5 train bank (n_bank ≈ 440-595), not the deployed test floor
   (HateMM 0.8760, ZH 0.8456). Absolute numbers here are not comparable to any main-table figure and
   must never be quoted as such; **the paired Δ against the same-bank deployed vote is the claim
   object**, exactly as in F89.
3. **Single verifier seed, single draw.** `MLP_SEED=0`, one fold assignment (`FOLD_SEED=0`), no
   resampling, no confidence intervals. Fold-sign consistency is the only variance control.
4. **Dimensionality reduction.** Pair features live in a 256-d PCA of the raw key space
   (0.9459-0.9842 variance retained). The load-bearing cosine control is computed in the **full** raw
   space, so the reduction can only handicap the verifier — but a verifier on un-reduced 7168-d keys
   was not run (177k × 14336 float pair features do not fit in memory) and is formally unmeasured.
5. **One pair encoding, two model families.** `[|z−z'|, z⊙z']` with an MLP and a logistic model.
   Bilinear, cross-attention and margin-trained siamese encodings are unmeasured; §5 argues they are
   not worth measuring given that scorer quality was never the binding constraint, but that is an
   argument, not a measurement.
6. **The logistic arm's end-to-end numbers are contaminated** by the class collapse control 4
   detected (§3.3) and should be read as "this aggregation is not calibration-safe", not as
   "supervised diagonal metrics are worse than cosine by 0.13 acc".
7. **`m=10` per class and the max/mean-3 aggregations were frozen, not tuned.** A different `m`, or a
   rank-weighted aggregation of verifier scores, is unmeasured. Note however that control 2b prices
   the *shape* independently of the verifier, so any such variant must first beat −0.0293/−0.0437.
8. **No test-fitted quantity appears anywhere.** No oracle, no threshold search, no per-item branch,
   no test label, no test split loaded by any script in this record.

---

## §7. FILE MANIFEST

| path | contents |
|---|---|
| `scripts/analysis/mechnov_pairverify.py` | **FROZEN arms**, sha256 `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d`, 455 lines, frozen before any real-data number |
| `scripts/analysis/mechnov_pairverify_runner.py` | orchestration only; asserts the frozen sha before running one (dataset × space) cell |
| `scripts/analysis/mechnov_drive.sh` | sequential cell driver with reap-retry |
| `scripts/analysis/mechnov_pairverify_diag.py` | §4.1 post-hoc mechanism diagnostics (within-query AUC, two-way variance decomposition) |
| `scripts/analysis/mechnov_pairverify_report.py` | reporting only; re-reads the OUT jsons at 4 dp |
| `scripts/analysis/mechnov_pairverify_{hatemm,zh,en}_OUT.json` | 3 spaces × 5 folds × 2 models × 2 aggregations per dataset |
| `scripts/analysis/mechnov_parts/*.json` | per-cell serialisations (reap-safe) |
| `scripts/analysis/mechnov_pairverify_diag_OUT.json` | §4.1 diagnostics, 3 datasets × 5 folds |
| `refine-logs/MECHNOV_PAIRVERIFY_PREGATE.md` | this record |

Read-only inputs: `data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_*.pt` (**train split only** —
`dev_seen`/`test_seen` were never opened by any script in this record);
`scripts/analysis/mechfix_ops.py` (imported, unmodified). Read for context, not modified: the three
ERRPAT reports, `MECHFIX_PREGATE_2026-07-27.md`, `state/findings.jsonl`, `state/directions_tried.json`.
Nothing under `autoresearch/goal_mllm_plus3/state/` was written. No file deleted or moved.
**Zero GPU, zero SLURM submissions, zero Modal calls, zero training of any deployed arm.**
