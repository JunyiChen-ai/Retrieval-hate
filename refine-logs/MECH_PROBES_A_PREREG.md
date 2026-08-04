# MECH-PROBES-A — $0 CPU kill probes for three mechanism candidates (RVS / XFM / AQM-zero-param)

Frozen: 2026-08-05, BEFORE any probe metric was computed.
Class: $0 CPU diagnostic. Per CLAUDE.md 实验流程 (2026-08-05 ruling) this gets **no review
round**; author self-test on synthetic data is the release gate. The four hard red lines
hold: zero test-set contact, decision rules frozen before results, blindness during
design/implementation, single submission for the formal run.

Driver: `scripts/analysis/mech_probes_a.py`
Submission: `scripts/slurm/mech_probes_a_cpu.sbatch` (CPU only, 8 cpus, 32 G, no `--time`,
no `--gres`)
Outputs: `artifacts/mech_probes_a/` only. Nothing under `configs/` or
`artifacts/c06_falsifier/` is written or modified.

---

## 0. Code facts verified before freezing (read-only)

**Fact A — CONFIRMED.** The deployed retrieval-contrastive triplet path has no id-based
self-exclusion.
- `src/utils/retrieval.py:421` — `index.add(train_feats_normalized)` indexes **all** train
  rows (the epoch-start bank rebuilt at `retrieval.py:344-386`).
- `src/utils/retrieval.py:426` — `index.search(query_feats_normalized, ...)` where
  `query_feats` is the live grad-on batch embedding `feats` passed at
  `src/model/loss.py:284-296` / `:308-320`.
- `src/utils/retrieval.py:466-509` — the non-TARC mining loop's only filter is the label
  comparison `train_labels[I[i,iter]] != / == query_labels[i]`; `query_ids` is consulted
  **only** inside the `tarc_active` branch (`retrieval.py:313-320`, `:454-462`). No id
  equality test exists on the non-TARC path.
- Deployed configs set `--no_pseudo_gold_positives 1`
  (`scripts/slurm/enc3seed_lora_curric.sbatch:59-63`), so the rank-0 same-label hit — item
  i's own epoch-start row, detached at `src/run_rac.py:752-757` — is taken as item i's
  "pseudo-gold positive".
- Contrast: the NCA arm **does** exclude self by id —
  `src/model/loss.py:655-660` masks `logits[arange(B), own_rows] = -inf` using
  `id_to_row[batch_ids]`; the bank is built once per epoch at `src/run_rac.py:713-715`.

**Fact B — CONFIRMED.** `_nca_head_loss` (`src/model/loss.py:635-668`) is
`L = -mean_i log sum_{j in same-class, j != i} softmax_j(cos(q_i, k_j) / tau)` with
`tau = float(getattr(args, "nca_tau", 0.1))` at `src/model/loss.py:647` (default 0.1,
pinned to 0.1 for the arm per the docstring at `:676`). Anchor grad-on, bank detached
(`:650`). The measured head-space cosine range 0.999852–0.999976 is
`refine-logs/ERRPAT_HateMM_2026-07-26.md:125-140`.

## 0.1 Artifact inventory and one recorded deviation

- Deployed-head train keys: `artifacts/c06_falsifier/mints/mint_{ds}_N_s{seed}_ffull.npz`,
  array `K_train` (float64, HateMM 744x1024, ZH 579x1024), plus `lab`, `fold_of`.
  These are the deployed-configuration (full-train, real dev) heads minted by
  `scripts/analysis/headspace_mint.py` under the byte-verbatim deployed CLI.
- Fold heads: `mint_{ds}_N_s{seed}_f{0..4}.npz` (5 folds x 3 seeds x 2 datasets), fold
  assignment `StratifiedKFold(5, shuffle=True, random_state=0)`, parity-asserted against
  the banked `vsw_ckpt`.
- Fold-arena protocol and floors: `scripts/analysis/c06_falsifier_arena.py:1160-1192`
  (`gate_floor`); banked floors in `scripts/analysis/headspace_arena_{ds}_s{seed}_OUT.json`
  → `acc_deployed` HateMM 0.8884 / 0.8858 / 0.8858, ZH 0.8929 / 0.8895 / 0.8946.
- Vote operator: `scripts/analysis/mechfix_ops.py:74-96` `deployed_vote`, `:56` `macro_f1`.

**DEVIATION (recorded before running):** there is no standalone "banked deployed-head train
key cache" file separate from the C06 mints. The C06 `ffull` mint `K_train` arrays **are**
the deployed-head train keys and are used as-is, read-only. Probes 1 and 2 therefore
require zero new training. Probe 3 re-mints fold heads (the snapshots do not exist).

---

## 1. FROZEN DECISION RULES

### Probe 1 — NCA softmax concentration (candidate: RVS, rank-space surrogate)

Input: `K_train` from `mint_{ds}_N_s{seed}_ffull.npz`, all 6 (dataset, seed) cells.
Computation, per anchor i: L2-normalise keys in float64; `logits_ij = cos(k_i, k_j) / tau`
with `tau = 0.1`; mask `j == i` to `-inf` (matching `loss.py:655-660`); `p_i = softmax_j`.
Reported per (dataset, seed):
- mean over anchors of `exp(H(p_i)) / N`, `H` in nats, `N = n_train` (744 / 579);
- mean over anchors of the probability mass on the top-20 entries of `p_i`.

> **FROZEN RULE:** if mean top-20 mass > 0.5 → F75's NCA genuinely tested a kNN-shaped
> objective → the rank-space-surrogate candidate (RVS) is KILLED. If top-20 mass < 0.1
> (≈ uniform prediction: 20/744 = 2.7 %, 20/579 = 3.5 %) → F75 did not test a kNN objective
> on this geometry → RVS stays ALIVE.

Aggregation, frozen: the verdict is rendered on the **3-seed mean top-20 mass per dataset**.
KILLED requires > 0.5 on **both** datasets; ALIVE requires < 0.1 on **both** datasets. Any
other configuration (including a value inside [0.1, 0.5] on either dataset, or the two
datasets landing on opposite sides) is reported as **INDETERMINATE** — not a kill, not a
pass — with all 6 per-cell numbers printed so the boundary is visible.

### Probe 2 — fold-head vs full-head margin gap (candidate: XFM, cross-fit memory training)

Margin = the deployed kNN vote value `v` returned by `mechfix_ops.deployed_vote`
(top-20 rank-weighted signed cosine, weights 20..1, normalised by sum of weights).
- Config (a) **full head**: keys from the `ffull` mint; bank = all train rows; query = all
  train rows; `exclude_self=True` (LOO row exclusion).
- Config (b) **fold head**: for item i with `f = fold_of[i]`, keys from the `f{f}` mint;
  bank = rows with `fold_of != f` (the fitting pool); query = row i; `exclude_self=False`
  (item i is already out of the bank). This is `gate_floor`'s protocol verbatim.

Per-item shift `d_i = v_i^(b) - v_i^(a)`. Pool over the 3 seeds (ZH: 3 x 579 = 1737 values).

> **FROZEN RULE (decide ZH only — HateMM's gap is established):** two-sample KS test
> between the per-item margin distributions (a) vs (b), pooled over 3 seeds, plus the mean
> absolute per-item margin shift. If KS p > 0.05 AND mean |margin shift| < 0.02 →
> distributions coincide → XFM (cross-fit memory training) is KILLED as a standalone (no
> fuel on ZH, and the goal needs 2 datasets). Otherwise ALIVE.

Test: `scipy.stats.ks_2samp(v_full_pooled, v_fold_pooled)`, two-sided, default method.
HateMM is computed and reported descriptively; it renders no verdict.
Known accuracy facts for context (not re-decided here): HateMM LOO 0.9406 vs fold 0.8867
(5.4 pt gap); ZH LOO 0.8915 vs fold 0.8923 (no accuracy gap).

### Probe 3 — epoch-snapshot asymmetric bank (candidate: AQM zero-parameter realization)

Re-mint fold heads (5 folds x 3 seeds x 2 datasets = 30 head trainings) under the frozen
`headspace_mint.py` recipe, called unmodified, saving full-train key snapshots at the end
of epochs `t* in {10, 15, 20, 25, 29}` (snapshot taken after that epoch's
`eval_and_save_epoch_end`, i.e. after all of that epoch's optimiser steps; no weight update
follows it inside the epoch).

Arena, per (dataset, seed, t*): for each fold f, bank = `K_{t*}[fold_of != f]` with
`lab[fold_of != f]`, query = `K_29[fold_of == f]`; predictions via
`mechfix_ops.deployed_vote(topk=20, exclude_self=False)`. Predictions from the 5 folds are
concatenated into one full-train prediction vector; accuracy and `mechfix_ops.macro_f1` are
computed on it. Queries are out-of-fold train items only. `t* = 29` reproduces the floor
(bank = query = epoch-29) and is run as a self-check.

Floors (bank = query = epoch-29), banked: HateMM 0.8884 / 0.8858 / 0.8858,
ZH 0.8929 / 0.8895 / 0.8946 → 3-seed means 0.88667 and 0.89233.

> **FROZEN RULE:** if NO snapshot epoch t* beats the epoch-29 floor by >= +0.020 accuracy
> (3-seed mean) on BOTH datasets → the zero-parameter AQM realization is KILLED (the
> trained-g_phi version remains a separate later question — note this in the report either
> way). If some t* clears +0.020 on both → ALIVE, report the best t* and its acc/mF1 deltas.

Frozen tie-break on which floor the delta is measured against:
- Primary = the banked floors above, **provided** the re-mint reproduces the banked head:
  `max |K_29(remint) - K_train(banked mint)| <= 1e-6` for every one of the 30 cells.
- If that parity check fails on any cell, the primary floor switches to the **own-re-mint
  epoch-29 arena accuracy** (identical trajectory, apples-to-apples) and both sets of
  numbers are reported with the measured max deviation. Parity failure does not void the
  probe; the within-run t* vs t*=29 comparison is unaffected.
- Additionally, the re-mint's epoch-29 keys must equal the frozen `headspace_mint` output
  of the same process bit-for-bit (in-process assert); a mismatch there is a HALT.

---

## 2. COMPUTE PROJECTION (measured unit cost x explicit count)

Unit costs measured from the `secs` field of the 36 banked Head-N mint metas in
`artifacts/c06_falsifier/mints/` (same recipe, same node class, 8 threads):

| item | measured unit | count | projected |
|---|---|---|---|
| HateMM fold-head mint | 38.6 s median (37.3 min / 54.0 max) | 15 | 579 s |
| ZH fold-head mint | 30.7 s median (29.8 min / 118.6 max) | 15 | 460 s |
| snapshot key forwards (full train split, eval mode, no grad) | <= 0.5 s each (the banked mints price 2 such forwards per cell) | 30 x 5 = 150 | <= 75 s |
| Probe 3 arena votes (faiss IndexFlatIP, N <= 744) | <= 0.5 s per (seed, t*, fold) | 2 x 3 x 5 x 5 = 150 | <= 75 s |
| Probe 1 softmax (N x N float64) | <= 1 s per cell | 6 | <= 6 s |
| Probe 2 votes (1 full + 5 fold per cell) | <= 2 s per cell | 6 | <= 12 s |
| **total** | | | **~1210 s (20 min)** |

Worst case with the observed max unit costs: 15 x 54.0 + 15 x 118.6 + 170 = **~2760 s
(46 min)**. `projected_seconds = 1210` is the heartbeat denominator.

Resources: `--cpus-per-task=8 --mem=32G`, no `--gres`, no `--time`, no `--array`.
Disk: 30 cells x 5 snapshots x N x 1024 x 8 B = ~810 MB under `artifacts/mech_probes_a/`
(1.7 T free at freeze time).

## 3. Test-contact and blindness

- Three guard layers, inherited from the C06 lineage: `headspace_mint.py:106-116`'s
  `torch.load` guard; the driver reads only `train_*.pt` / `dev_seen_*.pt` through the
  frozen `load_split`; `scripts/analysis/c09_guard` on `PYTHONPATH`.
- Every query in every probe is a **train-split** item. No dev metric and no test metric
  enters any decision. `K_dev` is not read.
- No candidate metric was computed before this file was written. The self-test runs on
  synthetic random data only.
- Single submission: one `sbatch`, resumable by artifact presence, no re-submission on a
  clean run.
