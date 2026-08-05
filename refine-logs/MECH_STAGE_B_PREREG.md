# MECH-STAGE-B PREREGISTRATION

Four cheap CPU experiments on the surviving new-mechanism candidates, run in the
F113/C06 fold-head arena. **Every rule below is frozen before any candidate number
exists.** Nothing in this file was written after reading a treatment metric.

Protocol class: CLAUDE.md 实验流程 (2026-08-05). CPU-level, ~2 h total, so at most
one review round; the four hard red lines (zero test contact, rules frozen before
results, blindness during implementation, single submission) are kept in full.

---

## 0. Scope, arena, and the shared decision rule

### 0.1 Arena (identical to F113 / C06 / MECH-PROBES-A probe 3)

For each dataset `ds ∈ {hatemm, zh}` and seed `s ∈ {0,1,2}`:

* 5 folds from `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over the
  **train split only**, asserted against the banked `vsw_ckpt/<ds>/f<f>.npz` `ho_idx`
  by the frozen `headspace_mint.py`.
* For fold `f`: train a head on the fitting pool (`fold_of != f`), then
  `deployed_vote(bank = keys[fold_of != f], bank_lab, query = keys[fold_of == f], topk=20)`
  from the frozen `scripts/analysis/mechfix_ops.py`.
* Concatenating the five folds gives one out-of-fold prediction for every train item.
  Accuracy and macro-F1 (`mechfix_ops.acc`, `mechfix_ops.macro_f1`) over all train items
  are the arena metrics.

### 0.2 Floors (frozen literals, from `scripts/analysis/headspace_arena_<ds>_s<seed>_OUT.json`)

| dataset | acc s0 | acc s1 | acc s2 | mean | mF1 s0 | mF1 s1 | mF1 s2 | mean |
|---|---|---|---|---|---|---|---|---|
| hatemm | 0.8884 | 0.8858 | 0.8858 | 0.886667 | 0.8838 | 0.8811 | 0.8812 | 0.882033 |
| zh | 0.8929 | 0.8895 | 0.8946 | 0.892333 | 0.8747 | 0.8710 | 0.8765 | 0.874067 |

These are the *same* literals `mech_probes_a.py` used. They are the comparison basis for
B1, B2, B3 and B4.

### 0.3 SHARED KILL RULE (B1, B2, B3)

Let `Δacc(ds) = mean_s [ arena_acc(candidate, ds, s) − floor_acc(ds, s) ]` (3-seed mean of
the per-seed paired difference).

> **KILLED** iff `Δacc(ds) < +0.020` for at least one `ds ∈ {hatemm, zh}`,
> **or** `Δacc(ds) < 0` for at least one `ds`.
> **ALIVE** iff `Δacc(ds) ≥ +0.020` for **both** datasets.

Macro-F1 deltas are computed and reported for every cell but **do not enter any verdict**.
B4 has its own rule (§5.4). The +0.020 screening bar (vs the +0.030 goal bar on test)
is inherited from MECH-PROBES-A §1.

### 0.4 Instrument validation (HALT gate, runs first, zero training)

Replay §0.1 directly from the banked C06 mints (`artifacts/c06_falsifier/mints/
mint_<ds>_N_s<seed>_f<f>.npz`, `K_train`) and compare to §0.2.

> **HALT** the whole battery if `|replayed − banked| > 5e-5` for any of the 12 numbers
> (6 acc + 6 mF1). There is **no** fallback floor: a floor mismatch means the arena
> reimplementation is wrong and no candidate number may be read.

Second instrument check: for each of the three training arms, one **"mechanism-off"
parity head** per dataset (`s0 f0`) is trained with the arm's patches installed but the
mechanism bypassed. Its `K_train` must equal the banked mint's `K_train` **bit-exactly**
(`max |Δ| == 0.0`). This proves the harness plumbing (model capture, optimizer injection,
mining interception) is a no-op when the mechanism is off, so any measured delta is the
mechanism and not the harness.

> **HALT** the arm if its parity head is not bit-exact.

### 0.5 Blindness / integrity

* No test split is read anywhere. Three inherited layers: `headspace_mint.py`'s
  `torch.load` guard, its patched `load_feats_from_CLIP` (only `train_*.pt` /
  `dev_seen_*.pt` reachable), and `c09_guard` on `PYTHONPATH`. Every query in every arm
  is a train-split item; `K_dev` is never read by this driver.
* No candidate arena number is computed before this file is committed.
* Single SLURM submission (`scripts/slurm/mech_stage_b_cpu.sbatch`).
* Line-buffered heartbeat to `artifacts/mech_stage_b/progress/MSB_PROGRESS.txt`.

---

## 1. Deployed recipe facts this design depends on (verified in code, not assumed)

1. `src/model/loss.py:compute_loss` — hybrid loss
   `total = triplet_term * (1 − ce_weight) + BCE * ce_weight`, `ce_weight = 0.5`,
   `triplet_term = mean(relu(in_batch_neg − pseudo_gold + hard_neg + 0.1))`.
2. `--no_hard_negatives 1`, `--no_pseudo_gold_positives 1`, `--hard_negatives_multiple 12`
   ⇒ FAISS searches the top 12 of the train bank per query and takes the first
   opposite-label row as the hard negative, the first same-label row as the pseudo-gold
   positive.
3. `Faiss_GPU=False` ⇒ in `utils/retrieval.py:dense_retrieve_hard_negatives_pseudo_positive`
   the bank is a **numpy float32 array** and the retrieved neighbour vectors are copied
   out of it as **detached** tensors. The deployed retrieval terms therefore carry
   gradient **only through the query**. (This is what makes B3 need an explicit
   re-forward — see §4.2.)
4. The query's own row is in the bank and is its own nearest neighbour, so the pseudo-gold
   positive is the item's own epoch-start key. This is the degeneracy B2 and B3 address.
5. The bank is rebuilt once per epoch (at the epoch's first step, when
   `train_feats is None`), by iterating `train_dl` in `model.eval()` mode.
6. `train_dl` is a `shuffle=True` DataLoader ⇒ **creating its iterator draws from the
   global torch RNG**; the per-epoch forwards themselves draw none (eval mode, no
   BatchNorm). Any arm that would skip the epoch bank rebuild must still consume that
   iterator to keep the RNG stream aligned with the floor (§2.4, §3.3).
7. `torch.nn.utils.clip_grad_norm_` is applied to `model.parameters()` only.

---

## 2. B1 — RVS: rank-space deployed-vote surrogate loss

### 2.1 What is added

`L = L_deployed_hybrid + λ_rvs · L_RVS`. The deployed hybrid loss is **not replaced** and
not rescaled.

### 2.2 The surrogate (all constants frozen here)

At each optimizer step, for query `i` in the batch with grad-on fused embedding `f_θ(z_i)`:

1. `q_i = normalize(f_θ(z_i))`, grad-on.
2. Bank `K`: the **epoch-start**, **detached** keys of the head's whole fitting pool,
   L2-normalised, rebuilt at the first step of every epoch by a `model.eval()` +
   `no_grad` forward over the fitting-pool tensors taken **directly from
   `train_dl.dataset`** (no DataLoader iteration ⇒ **zero RNG consumption**).
3. `s_ij = q_i · k_j`. **Self-exclusion is applied here** (the item's own bank row is
   dropped from `i`'s candidate set) — this is the deliberate fix of the degeneracy of
   §1.4 for this term only.
4. Candidate set `C_i` = the top **M = 128** bank rows of `s_i·` by plain cosine
   (selection detached; `M` frozen). Ranks are computed within `C_i`. Rows outside the
   top-128 cannot reach a soft rank below 21 in the collapsed head geometry, so they
   carry weight ≈ 0; the boundary sigmoid mass is reported as a diagnostic
   (`rvs_boundary_sigma`) and **does not gate anything**.
5. Bandwidth, scale-free by construction:
   `β_i = max( IQR(s_i·) / 10 , 1e-8 )`, IQR over the **full** self-excluded bank row.
6. Soft ranks: `r_ij = Σ_{k ∈ C_i} σ( (s_ik − s_ij) / β_i ) − 0.5`
   (the `−0.5` removes the `k = j` self term, so the top-ranked item has `r ≈ 0`).
7. Weights: `w̃_ij = relu(21 − r_ij)` — the continuous analogue of the deployed
   `w = [20, 19, …, 1]` over the top 20.
8. Surrogate vote: `ṽ_i = Σ_j (2 l_j − 1) w̃_ij / ( Σ_j w̃_ij + 1e-8 )`.
   The deployed vote also multiplies by `cos`; in this head space cosines are 0.9998+
   (MECH-PROBES-A probe 1), i.e. an essentially constant factor, so the brief's
   weights-only form is used verbatim.
9. `L_RVS = BCEWithLogits( ṽ_i / temp , y_i )`, **temp = 0.25** frozen (puts `ṽ ∈ [−1,1]`
   on a `[−4,4]` logit range).

### 2.3 λ_rvs grid and its selection rule (frozen)

Grid `Λ = {0.1, 0.3, 1.0}`. All three are run on all 30 (ds, seed, fold) cells.

> `λ* = argmax_{λ ∈ Λ}  mean_{ds ∈ {hatemm, zh}} Δacc(ds, λ)`.
> One shared `λ*` for both datasets (no per-dataset selection). The §0.3 rule is then
> applied to `Δacc(hatemm, λ*)` and `Δacc(zh, λ*)`. All three λ rows are reported.

Selection is on out-of-fold **train** accuracy only; no test split is touched. This is a
max over 3, so it is optimistically biased — which makes a KILL verdict *conservative*
(sound) and would make an ALIVE verdict provisional. Frozen and stated in advance.

### 2.4 RNG

The extra bank forwards use `train_dl.dataset` directly, in eval mode, under `no_grad`;
they draw no RNG. The deployed per-epoch FAISS rebuild is left untouched. Hence the
λ = 0 ("mechanism-off") head must be bit-exact against the banked mint (§0.4).

### 2.5 Kill rule

§0.3, applied at `λ*`.

---

## 3. B2 — XFM: cross-fit memory training

### 3.1 Realization (frozen)

The retrieval terms (hard negative + pseudo-gold positive) of the head being trained for
fold `f` are mined against a **frozen sibling bank**:

* Item `i` in the fitting pool has fold label `k = fold_of[i]` (`k ≠ f`).
* Its bank is the banked C06 head **`mint_<ds>_N_s<seed>_f<k>.npz`** — a head trained
  without fold `k` — restricted to rows with `fold_of ∉ {k, f}`.
  * excluding `k` = the cross-fit requirement (removes the self-positive degeneracy of
    §1.4 by construction: item `i` is not in its own bank);
  * excluding `f` = the arena requirement (the fold-`f` head must never see a fold-`f`
    representation, as a bank row or otherwise).
* Everything else — the loss assembly, the selection loop (first opposite-label row of
  the top-12 as hard negative, first same-label row as pseudo positive), the deployment,
  the arena — is unchanged. The final head with its own full fitting-pool bank is what
  the arena reads.

### 3.2 Known confound, and what it does to each verdict (frozen)

Sibling head `k` was itself trained on folds `≠ k`, which **includes fold `f`**. Its
*weights* therefore carry a trace of the arena's query fold, even though no fold-`f` row
is ever used as a bank row. A leak-free realization would need heads excluding `{k, f}`
(20 siblings per (ds, seed) = 120 extra head trains), which is out of budget.

> Frozen consequence: leakage can only make the arena number **optimistic**.
> A **KILLED** verdict is therefore sound as-is.
> An **ALIVE** verdict is recorded as **ALIVE-CONFOUNDED** and may not be quoted as a
> positive result until replicated with `{k, f}`-excluding siblings.

### 3.3 RNG

The per-epoch bank-rebuild branch of the deployed mining function is bypassed (the
sibling banks are precomputed), so the wrapper explicitly consumes the `train_dl`
iterator once per epoch (`for _ in train_dl: pass`) to keep the RNG stream aligned
(§1.6). The "mechanism-off" parity head delegates to the untouched original function and
must be bit-exact (§0.4).

### 3.4 Kill rule

§0.3, with the §3.2 qualifier on an ALIVE outcome.

---

## 4. B3 — AQM with a trained g_φ

### 4.1 Realization (frozen)

* `g_φ` = a second `classifier_hateClipper`, **`copy.deepcopy(f_θ)`** taken at the first
  line of `model_pass`, i.e. initialized from `f_θ`'s init, same architecture.
* `g_φ`'s parameters are added to the same `AdamW` optimizer through the existing
  `aux_pack["module"]` channel (`src/run_rac.py:676-681`). `lambda_aux = 0`, so
  `compute_aux_loss` is never called and `aux_pack` has no other effect.
  `clip_grad_norm_` covers `model.parameters()` only (§1.7), so `g_φ` is unclipped —
  frozen as-is, recorded, not a design change.
* Bank keys are `g_φ(z)`; query keys and the BCE classification term are `f_θ(z)`.
* Per step: neighbours are **selected** on a detached epoch-start `g_φ` bank (identical
  FAISS engine, identical top-12-then-first-opposite/first-same selection loop), then the
  selected rows' **raw features are re-forwarded through `g_φ` with grad**, so `g_φ`
  actually receives the retrieval gradient (§1.3 makes this necessary — the deployed
  numpy path detaches the bank).
* **No self-exclusion**: the query's own row stays in the bank, exactly as deployed. The
  zero-parameter epoch-snapshot realization (MECH-PROBES-A probe 3, KILLED) also kept it.
* **Arena / deployment for this arm is asymmetric by design**:
  bank = `g_φ(fitting pool)`, query = `f_θ(held-out fold)`, deployed top-20 vote.

### 4.2 Memory-corruption guard (frozen, additional to §0.3)

`self_recall` = over the fitting pool, the fraction of bank rows `j` whose nearest
neighbour among the query-space vectors `{f_θ(z_i)}` is `i = j`.

> If the arm would be ALIVE under §0.3 but `self_recall < 0.8` (3-seed mean, either
> dataset), the verdict is **KILLED — memory-corruption artifact**.
> `self_recall` is reported for every cell regardless of verdict.

### 4.3 RNG

The epoch-start `g_φ` bank replaces the deployed rebuild, so the wrapper consumes the
`train_dl` iterator once per epoch as in §3.3. `copy.deepcopy` draws no RNG. The
"mechanism-off" parity head (patches installed, `g_φ` built and injected into the
optimizer, mining delegated to the original) must still be bit-exact (§0.4) — this is
also the check that adding `g_φ`'s zero-gradient parameters to `AdamW` does not perturb
`f_θ`'s update.

### 4.4 Kill rule

§0.3, plus §4.2.

---

## 5. B4 — TRA: per-bank-item trust radius (no head training)

### 5.1 Operator

Bank row `j` carries a scalar admission radius `ρ_j`. Row `j` may enter query `q`'s top-20
only if `cos(q, k_j) ≥ ρ_j`; rejected rows are replaced by the next-ranked admitted row.
The vote over the admitted 20 is the deployed rank-weighted signed-cosine vote
(`w = 21 − rank`, normalised by the sum of the weights actually used).

Keys come from the **banked C06 fold mints** — no head is trained in this arm.

### 5.2 Fitting protocol (frozen)

For each (ds, seed, fold `f`): bank = `fold_of != f` rows, queries = `fold_of == f` rows.
The queries are split into 5 groups by `position_in_sorted_ho_index % 5` (deterministic,
no RNG). For each group `g`: fit `ρ` on the other four groups, predict group `g`. Rotating
`g` gives one held-out prediction for every train item; accuracy/macro-F1 over all train
items is the B4 arena number, compared to the §0.2 floors under §0.3's shape.

Fitting = coordinate ascent, all constants frozen:

* init `ρ_j = −1` for all `j` (admit always);
* candidate set per row: `{−1} ∪ {quantile(S[:, j], p) : p ∈ {0.5, 0.75, 0.9, 0.95, 0.99}}`
  where `S` is the fit-group query × bank cosine matrix — **per-row quantiles, not a
  global grid**, because the head space is collapsed (cosines 0.9998+) and a fixed grid
  would be degenerate;
* objective `J(ρ) = mean_i [ (2y_i − 1) · v_i(ρ) ] − 0.05 · mean_rejection_rate(ρ)`
  — the mean **signed vote margin**, not 0/1 accuracy (see amendment A1 below); the
  shrinkage coefficient 0.05 is frozen;
* 2 sweeps over bank rows in ascending index order; strict improvement only, so ties keep
  the incumbent, which starts at `−1` (admit always).

**AMENDMENT A1 (2026-08-05, before any real key was read).** The objective was written as
`acc_fit(ρ)` above. The pre-submission synthetic drive (§9) showed that objective is
plateau-locked: in the planted-positive case (12 poisoned bank rows crowding every
query's top-20) rejecting *one* of them flips no prediction, every single-coordinate move
scores exactly 0, and greedy ascent never starts — measured 0.500 → 0.500. Switching to
the signed vote margin, which moves on every admission change and whose sign is the
deployed decision rule, recovers the planted positive 0.500 → 1.000 while the
rejection-rate-matched random control stays at 0.500 and the planted null's held-out
accuracy is unchanged (1.000 → 1.000). The amendment was made **before** the driver was
pointed at any banked mint and is recorded here and in the driver's docstring. Under the
old objective B4 would have been under-powered toward KILL — i.e. this is a
wrong-verdict-class defect, the class that is allowed to block.

**In-run instrument check (added with A1).** For every (ds, seed, fold), `ρ = −1`
(admit always) must reproduce the frozen `mechfix_ops.deployed_vote` exactly: identical
predictions and `max |Δvote| < 1e-9`. Otherwise the TRA operator is not a strict
generalisation of the floor and no B4 delta is interpretable → **HALT**.

**AMENDMENT A2 (2026-08-05, forced by that check firing in job 13998; no B4 candidate
number had been read).** The check above HALTed job 13998 with
`max |Δvote| = 0.00952381` on HateMM s0 f0. Diagnosis: the ordering was built with
`np.argsort(-S, kind="stable")`, while `deployed_vote` takes faiss's own top-20. In the
collapsed head space (cosines 0.9998+) **exact float32 ties occur**, and faiss's heap and
numpy's stable argsort break them differently. On 7 of 149 queries the neighbour **set**
was identical and only the order of one tied pair differed; when that pair carries
opposite labels the swap moves it across the rank-20/rank-19 boundary and changes the vote
by exactly `2·cos/210 = 0.00952`. Fix: `tra_order()` now builds the canonical order as
**faiss's own top-20, in faiss's own order, followed by the remaining bank rows in faiss's
k = n_bank order**, so admit-always is bit-identical to `deployed_vote` by construction;
the same single ordering is used for fitting and for held-out evaluation. Re-verified on
**all 30 (ds, seed, fold) cells: `max |Δvote| = 0`, zero prediction mismatches.** This was
a defect in the B4 instrument only — it could not touch B1/B2/B3, which never enter this
code path.

### 5.3 Mandatory control (frozen)

For each fitted `(ds, seed, fold, group)` cell, 5 random-radius controls matched in
**rejection rate**: draw `ρ^rand` by permuting the fitted `ρ` vector across bank rows
(`np.random.default_rng(1000*seed + 10*fold + g)`), which preserves the rejection-rate
profile exactly while destroying the row↔radius assignment. Evaluate on the same held-out
group.

### 5.4 Kill rule (frozen, B4-specific)

Let `d_s = heldout_acc_fitted(s) − floor_acc(s)` and
`c_s = heldout_acc_fitted(s) − heldout_acc_random(s)` (random = mean over the 5 draws),
per seed `s`.

> **KILLED** iff `mean_s d_s < +0.020` on either dataset,
> **or** `mean_s c_s ≤ SE(c)` on either dataset, where `SE(c) = std(c_s, ddof=1)/√3`.
> **ALIVE** only if both datasets clear both conditions.

---

## 6. Compute projection (measured unit cost × explicit count)

Unit cost is **measured**, not assumed: `artifacts/mech_probes_a/probe3.json` records the
wall time of all 30 fold-head re-mints of this exact recipe on this exact hardware
(8 CPU threads): **hatemm 38.1 s mean (37.6–39.0), zh 30.1 s mean (29.1–30.7)**, i.e.
34.1 s averaged, 1023 s for a full 30-head sweep.

| stage | heads | unit | overhead factor | projected s |
|---|---|---|---|---|
| instrument replay (banked mints, no training) | 0 | — | — | 60 |
| B1 parity + 3 λ × 30 heads | 92 | 34.1 s | 1.15 (bank forward + soft ranks) | 3608 |
| B2 parity + 30 heads | 32 | 34.1 s | 1.05 | 1146 |
| B3 parity + 30 heads | 32 | 34.1 s | 1.50 (g_φ bank + re-forwards + backward) | 1637 |
| B4 (no training) | 0 | — | — | 1000 |
| collect / report | 0 | — | — | 30 |
| **total** | **156** | | | **7481 s ≈ 2.08 h** |

`MSB_PROJECTED_SECONDS = 7481` is the heartbeat denominator.

B4's line is **measured, not estimated**: `_tra_fit` was timed at the real problem sizes
on synthetic matrices before submission — 9.43 s per cell at HateMM's `n_bank = 595,
n_fit_queries = 119` and 2.82 s per cell at MHC-ZH's `463 / 93`. With 5 groups × 5 folds ×
3 seeds = 75 cells per dataset that is 707 s + 212 s = 919 s of fitting, rounded to 1000 s
with the per-fold FAISS/argsort overhead.

Resources: 8 CPU / 32 GB, no GPU, no `--time`, no array, no cloud. Single submission.

---

## 7. Outputs

* `artifacts/mech_stage_b/instrument.json` — §0.4 floor replay.
* `artifacts/mech_stage_b/heads/<arm>_<ds>_s<seed>_f<fold>[_lam<λ>].npz` — per-head
  out-of-fold predictions, votes, held-out indices, diagnostics, parity fields. **No key
  matrices and no per-epoch snapshots are written** (disk quota).
* `artifacts/mech_stage_b/b4.json`
* `artifacts/mech_stage_b/MECH_STAGE_B_RESULT.json` — the four verdicts.
* `artifacts/mech_stage_b/progress/MSB_PROGRESS.txt` — heartbeat.

Nothing under `configs/`, `artifacts/c06_falsifier/` or `artifacts/mech_probes_a/` is
written or modified.

---

## 8. Frozen module SHAs (asserted at runtime; refuse to run on mismatch)

* `scripts/analysis/headspace_mint.py` =
  `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612`
* `scripts/analysis/mechnov_pairverify.py` =
  `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d`
* `scripts/analysis/mechfix_ops.py` — recorded in every output, pinned once measured.

No file under `src/` is edited by this battery. All behaviour changes are monkeypatches
installed by `scripts/analysis/mech_stage_b.py` inside its own process and removed on
exit.

---

## 9. Pre-submission synthetic drive (mandatory, run before the single submission)

`python scripts/analysis/mech_stage_b.py selftest` drives every numeric path on synthetic
data — no cluster artifact is read and no candidate metric is computed. It must show a
**planted positive** and a **planted null** being classified correctly. Recorded outcome:

* `rvs_term`: gradient reaches the query, loss on the true labels 0.6827 < loss on flipped
  labels 0.8369, mean 21.2 non-zero soft-rank weights (the intended top-20 window).
* selection loop reproduces the frozen retrieval semantics (first opposite-label row =
  hard negative, first same-label row = pseudo positive).
* TRA planted positive 0.500 → 1.000; rejection-rate-matched random control 0.500;
  planted null held-out 1.000 → 1.000.
* collect/verdict machinery: planted positive → ALIVE, planted null → KILLED, λ*
  selection exercised, self-recall veto fires when forced below 0.8.
* **Real 2-epoch training smoke through the actual patched paths** for all three arms on
  random features: no non-finite parameters, and `g_φ` is verified to move away from
  `f_θ` (i.e. it really is being trained, not a dead deepcopy).

Amendment A1 (§5.2) is the defect this drive caught.

## 10. Additional frozen implementation decisions

* B3's `g_φ` is forwarded in `eval()` mode when re-encoding the selected neighbours, so
  memory keys are deterministic (no dropout noise in the bank).
* B3's `g_φ` is not covered by `clip_grad_norm_` (§1.7), which is left untouched.
* B2/B3 leave the deployed pseudo-gold-positive **count** and the top-12 search width
  unchanged; only the bank the search runs over is different.
* Head artifacts store predictions and diagnostics only — no key matrices, no per-epoch
  snapshots (disk quota).
