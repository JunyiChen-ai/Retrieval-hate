# VGA / VNQ — $0 pregate on VERIFIER-GATED ADJUDICATION (C1) and NEIGHBOURHOOD-QUALITY SELECTIVE PREDICTION (C2)

**Date:** 2026-07-27 NZST · **Agent:** vga-pregate · **Cost: $0** (CPU only, ≤8 threads,
**zero GPU, zero SLURM, zero Modal, zero training of any deployed arm**).
Repo sha at freeze time `49e15ec` (working tree dirty).
**Test-split contact: NONE** — every script in this record loads only the `train` split.

**Spec:** `refine-logs/LITSWEEP6_RELGEN.md` §2, candidates C1 (VGA) and C2 (VNQ), read in full
before any code was written. **Machinery:** `refine-logs/MECHNOV_PAIRVERIFY_PREGATE.md` (F95) and
`refine-logs/MECHFIX_PREGATE_2026-07-27.md` (F89), both read in full.
**Scope:** C1 and C2 only. C3 (VEA) is a writing task, not a measurement; C4 (VSW) was **not**
tasked and is **not** run here (see §7.7); note only that the emitter now exists, so C4's
λ-sweep exchange-rate curve is a much cheaper rider than it was when the sweep record priced it.

**What this is.** A $0 pregate on a *per-item switch* between two complete decision rules.
**What this is not.** Not a verdict, not a prereg for GPU, not a promotion. The arena is the banked
**raw** encoder key space on the **train** split, not the deployed trained-head space and not test.

---

## §0. PROVENANCE — what had to be regenerated, and why that is not a design change

`LITSWEEP6_RELGEN.md` §0(a) is correct and was verified independently before any code was written:
**the F95 pair-score matrices are not banked.** `scripts/analysis/mechnov_pairverify.py` contains
zero `np.save`/`torch.save` calls; only per-cell summary JSON survives
(`mechnov_parts/*.json`, `mechnov_pairverify_{hatemm,zh,en}_OUT.json`). The `S[query, bank]`
matrices and every per-item fixed/broken identity — which is exactly what C1's gate target and C2's
risk ordering are made of — were computed in memory and discarded.

The sweep record's prescribed remedy is followed verbatim: the frozen arms module
(sha256 `77b0defd…b7240d`) is **not edited to add persistence**. A new emitter,
`scripts/analysis/vga_pregate_emit.py`, **imports it unmodified** (asserting both its sha256 and
`mechfix_ops.py`'s `635c1312…c83fc8d` before running) and replays the frozen fused-space primary
cell fold by fold, reusing the frozen module's own `build_space`, `l2n`, `all_unordered_pairs`,
`pair_features`, `fit_mlp`, `predict_mlp` and every frozen constant (`K_FOLDS`, `FOLD_SEED`,
`PCA_DIM`, `PCA_SOLVER`, `PAIR_FIT_CAP`, `PAIR_SUBSAMPLE_SEED`, `MLP_*`, `M_PER_CLASS`,
`TOPK_DEPLOYED`, `MEAN_TOPQ`, `PATHOLOGY_RANK`).

**One efficiency deviation, disclosed.** F95 scored every (held-out × in-fold) pair; the emitter
scores only the **20 nominated candidates** per query. The frozen MLP is a deterministic pointwise
function of its fitted parameters, and the fit set, PCA, standardisation statistics, seeds and fold
assignment are all bit-identical, so the scores on the nominated subset are bit-identical too. This
is asserted rather than argued — see the parity gate in §3, which is a hard `assert` on **26 frozen
F95 quantities per dataset** and aborts the run on any mismatch at 4 dp.

---

## §1. FROZEN KILL BARS — quoted verbatim from `LITSWEEP6_RELGEN.md` §2, before any result

> **Frozen kill bar (declare before any real-data number).**
> - **K-VGA-1 (primary):** nested-CV net ≥ **+0.030 acc on ≥2 of 3 datasets**, fold-sign ≥4/5 on
>   those datasets. Miss ⇒ KILL, axis closed.
> - **K-VGA-2 (permutation null, mandatory):** the gate must beat a label-shuffled null at the same
>   fitting budget. With N = 89-111 gated items, this is the dominant overfitting risk and the null
>   is not optional.
> - **K-VGA-3 (new-signal control, mandatory):** an arm whose gate uses **F47 features only**
>   (vote margin, purity, sub-votes). If it matches the verifier-feature gate, the "new information
>   source" argument that unlocks F47 is **refuted** and the direction dies *regardless of net*.
> - **K-VGA-4 (class balance):** positive rate within 0.10 of the bank rate, per F95 control 4 —
>   the logistic arm collapsed to 0.0237-0.0604 there and its nulls were uninterpretable.

> **Frozen kill bar.** **K-VNQ-1:** AUGRC improvement over the kNN-UE baseline (distance + neighbour
> label ratio) on **≥2 of 3 datasets**, fold-sign ≥4/5. **K-VNQ-2:** must also beat the plain vote
> margin. Both are computed from the same per-item table as C1, so marginal cost is minutes.

The sweep record's §1 arithmetic, which these bars are calibrated against, is also frozen here and
is **recomputed** in §3 rather than transcribed:

| dataset | n | oracle-gate ceiling | gate set N | base rate of "fix" | **break-even precision** |
|---|---|---|---|---|---|
| HateMM | 744 | +0.0726 | 111 | 0.4865 | **0.6005** |
| MHC-ZH | 579 | +0.0535 | 89 | 0.3483 | **0.5976** |
| MHC-EN | 549 | +0.0893 | 106 | 0.4623 | **0.5777** |

---

## §2. FROZEN DESIGN — every operationalisation declared before any real-data number

### 2.1 Arena
Banked **raw fused** key space (`L2norm(concat(L2norm(img), L2norm(text)))`, 7168-d), **train split
only**, item-disjoint `StratifiedKFold(5, shuffle=True, random_state=0)` — the F95 protocol
verbatim. Adjudicator = the F95 **PRIMARY** cell: fused × **MLP** × **max**. The `mean-top-3`
adjudicator is emitted and read as **SECONDARY** and cannot carry a pass. The logistic verifier arm
is **not** used: F95 control 4 fired on it (collapse to positive rate 0.0237-0.0604).

Inherited limitation, stated once and assumed throughout (F95 §6 L1, restated in
`LITSWEEP6_RELGEN` §0(b)): a raw-space, train-split result does not transfer automatically to the
deployed head space or to test, and the campaign's history (F47, F66, F89) is that raw-space oracles
do not survive that trip.

### 2.2 Gate set and gate target
**Gate set** = items where the deployed top-20 rank-weighted signed-cosine vote and the F95
adjudication **disagree**. Disagreement is computable at inference without any label (both decision
rules are label-free given the bank), so restricting the gate to it is deployable, not an oracle.
On agreement items the switch is a no-op *by construction*, so the emitted label is bit-identical to
the deployed vote there and the F95 control-2b shape cost is **priced to exactly zero off the gate
set** — which is the structural discharge of F95 ban clause (c) that C1's distinctness argument
rests on.

**Gate target** `y = 1` if adjudication is right on this item (a **fix**), `0` if the deployed vote
is right (a **break**). Exactly one holds on a disagreement item. **Emission** = adjudicated label
iff the gate fires, deployed vote otherwise.

### 2.3 Feature blocks — declared, budget-matched, no post-hoc additions

**VGA gate features (6)** — the transplant-sketch list from `LITSWEEP6_RELGEN` §2 C1 (ii), verbatim,
computed over the 20-item shortlist; all test-time-computable, none using the query's label:
`v_max` (max verifier score) · `v_top3mean` (top-3 mean) · `v_gap` (best pos-class minus best
neg-class verifier score) · `v_spearman_rho` (Spearman ρ between the cosine ordering and the
verifier ordering over the shortlist) · `v_rank_of_cos_top1` (rank of the cosine top-1 in the
verifier ordering) · `v_disp` (verifier score dispersion).

**K-VGA-3 control features** — the F47 family (vote margin, purity, sub-votes), **no verifier
anywhere**. Two arms, both declared now:
* `f47ctrl` (**6 features, budget-matched to the verifier gate**): `abs_vote`, `purity_pred`,
  `mean_cos20`, `cos_spread`, `label_ratio`, `sub_vote_gap` (|text-channel vote − image-channel
  vote|).
* `f47ctrl_full` (**10 features, a generosity check**): the above plus signed `vote`,
  `sub_vote_text`, `sub_vote_img`, `sub_agree_ft`.
**K-VGA-3 fires if *either* control matches the verifier gate.** Budget-matching is declared because
an over-parameterised control that overfits would flatter the verifier arm, which is the wrong
direction for a bar whose purpose is to *refute* our own unlocking argument.

**kNN-UE features for C2 (5)** — the Hashimoto et al. (Findings of NAACL 2025) ingredients,
distances + neighbour label ratio: `max_cos`, `mean_cos20`, `cos_spread`, `label_ratio`,
`purity_pred`.

### 2.4 Nesting and operating point
For outer fold `f` (the frozen F95 fold assignment): the gate is fitted **only** on gate-set items
whose fold is not `f`; an **inner** `StratifiedKFold(5, shuffle=True, random_state=17)` inside that
pool produces out-of-sample scores; the **operating point** is the threshold maximising net gain
(fired fixes − fired breaks) on those inner out-of-sample scores, with `+inf` (fire on nothing,
net 0) always a candidate and ties resolved to the most conservative threshold; the gate is then
refitted on the whole pool and that inner-chosen threshold is applied to fold `f`. **The gate never
sees the fold it scores, and no threshold is ever chosen on an evaluated fold.**

*Residual coupling, disclosed:* a training item's own fix/break label came from the F95 fold in
which *it* was held out, and that fold's verifier was fitted on a set that included fold-`f` items.
The gate therefore never sees fold `f`'s outcomes, but its training features were produced by
verifiers that saw fold `f`'s items. This is inherent to reusing the frozen LOO protocol rather than
refitting a doubly-nested verifier (which the spec does not ask for and which would cost 25 verifier
fits per dataset instead of 5). It is recorded as limitation L3.

### 2.5 Arms and multiplicity control
Gate models: **logistic** (L2, C=1.0, lbfgs) and **shallow GBM** (100 trees, depth 2, lr 0.05,
seed 0) — the "logistic + shallow GBM" of the transplant sketch. Features are standardised on the
fitting rows only. **PRIMARY gate model = logistic** (lower capacity, appropriate at N = 89-111);
GBM is **SECONDARY**. To block cross-dataset model shopping, **K-VGA-1 must be met by one and the
same gate model on ≥2 of 3 datasets**; a pass assembled from different models on different datasets
does not count. References computed alongside: **`oracle`** (fire iff the item is a fix — the
ceiling) and **`fire_all`** (fire on everything — ungated F95 adjudication, which must reproduce
F95's net of −3 / −27 / −8).

### 2.6 The required-precision arithmetic
Firing on `m` gate-set items with precision `p` gives net `= m(2p − 1)`, so reaching `Δacc = g`
requires `p = 0.5 + g·n/(2m)`. At `g = 0.030` and full coverage (`m = N`) this reproduces the sweep
record's break-even precisions exactly (§1 table). Gate precision achieved is therefore always
reported **against the required precision at the fire count actually achieved**, not against the
full-coverage number, because a gate that fires on fewer items needs a *higher* precision.

### 2.7 Machinery validity (positive control, run BEFORE the freeze, synthetic data only)
`vga_pregate_gate.py --selftest`, two synthetic arms at the real problem's scale (n = 700,
N_gate = 110, base rate 0.47):

| arm | model | Δacc | gate precision | permutation p |
|---|---|---|---|---|
| **A — fix/break IS a function of the features** | logistic | **+0.0571** | 0.8448 | **0.0164** |
| A | gbm | **+0.0543** | 0.8167 | **0.0164** |
| **B — features are pure noise** | logistic | **−0.0129** | 0.4416 | 0.8852 |
| B | gbm | **−0.0043** | 0.4717 | 0.5410 |

**The harness is not structurally incapable of returning a positive**, and it returns an honest null
with a non-significant permutation p when the features carry nothing. A null below is therefore a
property of the data, not of the code. No real-dataset number was computed before the shas below
were frozen.

### 2.8 Frozen script shas
| path | sha256 |
|---|---|
| `scripts/analysis/vga_pregate_emit.py` | `a3a41ae7a15a1ae7796161ad11901d2dc351b3e138cdeebc6430bc7f51b7ce56` |
| `scripts/analysis/vga_pregate_gate.py` | `ea37c57b382b9bb0d1c3a87e9302bac7e52071b8cfe85126e96e26eb524f4e34` |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` (F95, imported unmodified) |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` (F89, imported unmodified) |

Permutation null: `N_PERM = 200`, `PERM_SEED = 12345`, targets shuffled **within the fitting pool
only** (the evaluated fold's true outcomes are untouched), the full nested pipeline including
threshold selection re-run per permutation — i.e. the null is at the *same fitting budget*, as
K-VGA-2 requires. Reported `p = (1 + #{null ≥ observed}) / (N_PERM + 1)`; significance at `p < 0.05`.

<!-- EVERYTHING BELOW THIS LINE WAS WRITTEN AFTER THE RUN -->

---

## §3. PARITY — the regenerated arena is the F95 arena, verified, not asserted

`scripts/analysis/vga_pregate_emit.py` hard-asserts **26 frozen F95 quantities per dataset** at 4 dp
before writing anything. **78/78 gates PASS** (26/26 HateMM, 26/26 MHC-ZH, 26/26 MHC-EN); a single
mismatch would have aborted the run. The gated quantities are, per dataset: `acc/mF1/posrate` of the
deployed vote, of the cosine-shape control (2b), of `mlp_max` and of `mlp_mean3`; `n_deployed_wrong`;
`n_pathology_pop`; `median_sc_rank_all`; `median_sc_rank_deployed_wrong`; and
`fixed / broke / net / exchange_rate / pathology_fixed` for both aggregations.

Beyond the asserted set, the per-fold PCA explained-variance sequences reproduce F95 fold for fold
(HateMM fused `[0.9459, 0.9447, 0.9442, 0.9454, 0.9447]`, ZH `[0.9627, 0.9628, 0.9626, 0.9619,
0.9627]`, EN `[0.9546, 0.9546, 0.9548, 0.9548, 0.9549]`), as do the fitted-pair counts
(150 000 / 106 953 / 96 141). Runtime 67.3 / 47.9 / 42.4 s per dataset on ≤8 threads.

**Minor erratum against F95, recorded for provenance.** F95 §3.4 states "PCA retains
**0.9459**-0.9842 of key variance at 256 components in every cell". The true minimum over the 45
per-fold values in `mechnov_pairverify_*_OUT.json` is **0.9442** (HateMM × fused × fold 2); 0.9459 is
the maximum of that cell, not the minimum over cells. This changes nothing — it is a bound
misstatement in a provenance sentence, not a result — but it is corrected here rather than carried.

**The sweep record's §1 arithmetic reproduces exactly** from the regenerated per-item table, which is
the strongest available check that the C1/C2 design was costed against the right numbers:

| dataset | n | gate set N | fixed | broken | base rate "fix" | oracle-gate ceiling | break-even precision |
|---|---|---|---|---|---|---|---|
| HateMM | 744 | 111 | 54 | 57 | 0.4865 | **+0.0726** | 0.6005 |
| MHC-ZH | 579 | 89 | 31 | 58 | 0.3483 | **+0.0535** | 0.5976 |
| MHC-EN | 549 | 106 | 49 | 57 | 0.4623 | **+0.0893** | 0.5777 |

Every cell matches `LITSWEEP6_RELGEN.md` §1 at the stated precision.

---

## §4. C1 / VGA — RESULTS PER FROZEN BAR

Every number re-read at report time from `scripts/analysis/vga_pregate_OUT.json` via
`scripts/analysis/vga_pregate_report.py`, 4 dp. All declared arms ran; none was dropped.

**Two machinery checks first, both passing.** The `fire_all` reference (gate fires on every gate-set
item = ungated F95 adjudication) returns **−0.0040 / −0.0466 / −0.0146**, reproducing F95 §3.2's
primary-cell deltas exactly; the `oracle` reference returns **+0.0726 / +0.0535 / +0.0893**,
reproducing §1's ceilings exactly. The gating harness is wired to the right two endpoints.

### 4.1 K-VGA-1 (primary bar: net ≥ +0.030 acc on ≥2 of 3 datasets, fold-sign ≥4/5) — **FAIL**

Verifier-feature gate, PRIMARY adjudicator (fused × MLP × max):

| dataset | model | fired / N | gate precision | **required precision at that fire count** | Δacc | ΔmF1 | fold signs |
|---|---|---|---|---|---|---|---|
| HateMM | logistic (**PRIMARY**) | 48 / 111 | 0.4375 | 0.7325 | **−0.0081** | −0.0097 | `00-+-` |
| HateMM | gbm | 88 / 111 | 0.5455 | 0.6268 | **+0.0108** | +0.0056 | `0+0+0` |
| MHC-ZH | logistic (**PRIMARY**) | 0 / 89 | n/a | n/a | **+0.0000** | +0.0000 | `00000` |
| MHC-ZH | gbm | 0 / 89 | n/a | n/a | **+0.0000** | +0.0000 | `00000` |
| MHC-EN | logistic (**PRIMARY**) | 46 / 106 | 0.3696 | 0.6790 | **−0.0219** | −0.0423 | `-----` |
| MHC-EN | gbm | 45 / 106 | 0.5111 | 0.6830 | **+0.0018** | +0.0038 | `+-+-+` |

**0 of 3 datasets reach +0.030 with any verifier gate, under either model, and no verifier arm has
≥4/5 positive fold signs.** The best verifier number anywhere in the primary read is **+0.0108**,
about one third of the bar. On MHC-ZH the nested operating-point selection declined to fire on any
item in any fold — an honest conservative outcome, not a failure: the inner folds offered no
threshold with positive net gain.

The best number in the **entire** battery (36 arm cells across both adjudicators) is **+0.0296**
(HateMM × `f47ctrl:gbm` × the SECONDARY mean-top-3 adjudicator) — still **under bar**, on a secondary
adjudicator, and it is a **control** arm, not a verifier arm.

**An arithmetic point worth recording.** §1's break-even precisions (0.5777-0.6005) assume the gate
fires on the *whole* gate set. Every arm that actually fires does so on 3-79 % of it, and partial
coverage *raises* the required precision to **0.63-0.79** for the arms firing on ≥30 items. The gate's real
target is harder than the headline break-even number suggests, and no arm cleared it: the closest
approach anywhere is `f47ctrl_full:gbm` on HateMM at precision **0.6923** against **0.7146** required.

### 4.2 K-VGA-2 (permutation null, mandatory) — **FAIL for the primary arm**

200 permutations per cell, targets shuffled within the fitting pool only, full nested pipeline
re-run per permutation. PRIMARY adjudicator:

| dataset | arm | observed | null mean ± sd | null q95 | **p** |
|---|---|---|---|---|---|
| HateMM | verifier:logistic | −0.0081 | −0.0024 ± 0.0058 | +0.0067 | **0.8706** |
| HateMM | verifier:gbm | +0.0108 | −0.0026 ± 0.0060 | +0.0067 | **0.0100** |
| MHC-ZH | verifier:logistic | +0.0000 | −0.0024 ± 0.0041 | +0.0017 | **0.5174** |
| MHC-ZH | verifier:gbm | +0.0000 | −0.0027 ± 0.0037 | +0.0017 | **0.3632** |
| MHC-EN | verifier:logistic | −0.0219 | −0.0062 ± 0.0080 | +0.0055 | **0.9751** |
| MHC-EN | verifier:gbm | +0.0018 | −0.0064 ± 0.0081 | +0.0056 | **0.1841** |

The PRIMARY (logistic) verifier gate is **indistinguishable from its own label-shuffled null on all
three datasets**. The GBM verifier arm clears its null on HateMM (p = 0.0100) but at +0.0108 — a
real-but-tiny effect, a third of the bar, and it does not replicate on ZH or EN.

### 4.3 K-VGA-3 (new-signal control, mandatory) — **FIRES. This is the decisive result.**

The frozen bar: *"an arm whose gate uses **F47 features only** … If it matches the verifier-feature
gate, the 'new information source' argument that unlocks F47 is **refuted** and the direction dies
regardless of net."*

**It does not merely match. It beats the verifier gate on all three datasets** (best verifier arm vs
best F47-control arm, PRIMARY adjudicator):

| dataset | best verifier arm | best F47-control arm | control − verifier | control's permutation p |
|---|---|---|---|---|
| HateMM | +0.0108 (gbm) | **+0.0269** (`f47ctrl_full:gbm`, signs `+++++`) | **+0.0161** | **0.0050** |
| MHC-ZH | +0.0000 | **+0.0104** (`f47ctrl_full:logistic`, signs `++0+-`) | **+0.0104** | **0.0050** |
| MHC-EN | +0.0018 (gbm) | **+0.0182** (`f47ctrl_full:logistic`, signs `+-+0+`) | **+0.0164** | **0.0100** |

The inversion is sharpest where it matters most: on the two datasets where the PRIMARY verifier gate
is statistically dead (ZH p = 0.5174, EN p = 0.9751), the F47-only control is **significant against
the same null at the same fitting budget** (p = 0.0050 and 0.0100). The gate signal on the
disagreement set is real and permutation-validated — and it is carried by **vote margin, purity,
neighbour statistics and cross-channel sub-vote disagreement**, i.e. unsupervised functions of the
cosine ordering, *not* by the trained relation profile.

This refutes, by measurement rather than by argument, the load-bearing distinctness claim that
unlocked F47 for this candidate (`LITSWEEP6_RELGEN` §2 C1: *"This argument is the load-bearing one
and it is pre-registered as falsifiable — see control G2 below"*). It was pre-registered as
falsifiable; it has been falsified. **Per the frozen bar, the direction dies regardless of net.**

### 4.4 K-VGA-4 (class balance: positive rate within 0.10 of the bank rate) — **PASS**

| dataset | bank pos-rate | verifier:logistic | verifier:gbm | max deviation |
|---|---|---|---|---|
| HateMM | 0.4005 | 0.4489 | 0.3790 | 0.0484 |
| MHC-ZH | 0.3109 | 0.3489 | 0.3489 | 0.0380 |
| MHC-EN | 0.3060 | 0.2168 | 0.2659 | 0.0892 |

No arm collapses; every emitted decision sits within 0.10 of its bank rate. Unlike F95's logistic
arm, **the nulls here are interpretable** — which is what makes §4.2's non-significance a real null
rather than an artefact. This bar passing is what licenses reading the other three as measurements.

---

## §5. C2 / VNQ — RESULTS PER FROZEN BAR

Selective prediction over the **deployed vote's** own errors. AUGRC (Traub et al., arXiv:2407.01032),
**lower is better**; the implementation was verified against hand-computed values (0.2500 / 0.3333 on
a 4-item case) and orders a perfect detector below random below a perfect anti-detector.

| dataset | arm | **AUGRC** | AURC | AUROC (error detection) |
|---|---|---|---|---|
| HateMM | VNQ (verifier profile, fitted) | 0.0458 | 0.0737 | 0.7451 |
| HateMM | **kNN-UE (distance + label ratio, fitted)** | **0.0429** | 0.0685 | 0.7674 |
| HateMM | plain vote margin | 0.0465 | 0.0766 | 0.7395 |
| HateMM | VNQ raw scalar (\|s₁−s₀\|) | 0.0687 | 0.1253 | 0.5708 |
| MHC-ZH | VNQ | 0.0417 | 0.0687 | 0.7668 |
| MHC-ZH | **kNN-UE** | **0.0393** | 0.0525 | 0.7860 |
| MHC-ZH | **plain vote margin** | **0.0384** | 0.0515 | 0.7927 |
| MHC-ZH | VNQ raw scalar | 0.0553 | 0.0907 | 0.6615 |
| MHC-EN | VNQ | 0.0810 | 0.1307 | 0.6713 |
| MHC-EN | **kNN-UE** | **0.0758** | 0.1135 | 0.7012 |
| MHC-EN | **plain vote margin** | **0.0696** | 0.1053 | 0.7375 |
| MHC-EN | VNQ raw scalar | 0.0891 | 0.1469 | 0.6238 |

**K-VNQ-1 (AUGRC improvement over kNN-UE on ≥2 of 3, fold-sign ≥4/5) — FAIL, 0 of 3.**
ΔAUGRC (positive = VNQ better): **−0.0029** (HateMM, 2/5 folds), **−0.0024** (ZH, 1/5),
**−0.0052** (EN, 1/5). VNQ is **worse** than the kNN-UE baseline on every dataset.

**K-VNQ-2 (must also beat the plain vote margin) — FAIL.**
ΔAUGRC: **+0.0007** (HateMM, 3/5 folds — positive but under the ≥4/5 fold-sign requirement),
**−0.0033** (ZH, 2/5), **−0.0114** (EN, 2/5). 1 of 3 datasets, and that one misses the fold bar.

**This is the sweep record's expectation inverted.** `LITSWEEP6_RELGEN` §2 C2 predicted *"Technically
this probably passes: the verifier's within-query AUC advantage is +0.16 to +0.23 with 5/5 fold signs
on all three datasets, and it would be surprising if none of that showed up in a risk ordering."*
None of it showed up. The **plain deployed vote margin** — a scalar the system already computes for
free — is a **better** error detector than the trained relation profile on all three datasets by
AUROC (0.7395/0.7927/0.7375 vs 0.7451/0.7668/0.6713, i.e. VNQ wins only on HateMM and loses on ZH and
EN), and the raw verifier scalar is far worse than everything (AUROC 0.5708-0.6615).

The mechanism is now legible and is the same one F95 §3.2b found: **separating same-class from
different-class pairs is a different question from knowing whether the vote is about to be wrong.**
The verifier is much better at the first (+0.16 to +0.23 within-query AUC) and no better at the
second. Selective prediction was the axis on which the exchange-rate law was supposed not to apply;
it does not apply, and the result is negative anyway, for an independent reason.

---

## §6. VERDICT

# **KILL — both candidates, on 5 of 6 frozen bars.**

| bar | verdict | margin |
|---|---|---|
| **K-VGA-1** net ≥ +0.030 on ≥2/3, signs ≥4/5 | **FAIL** | best verifier arm +0.0108; 0/3 datasets; best in whole battery +0.0296 (control arm, secondary adjudicator) |
| **K-VGA-2** beat the label-shuffled null | **FAIL** (primary arm) | p = 0.8706 / 0.5174 / 0.9751 for verifier:logistic |
| **K-VGA-3** F47-feature control must not match | **FIRES** | control **beats** verifier on 3/3 (+0.0161 / +0.0104 / +0.0164), and is significant (p ≤ 0.0100) where the verifier is not |
| **K-VGA-4** class balance within 0.10 | **PASS** | max deviation 0.0892 |
| **K-VNQ-1** AUGRC beats kNN-UE on ≥2/3 | **FAIL** | 0/3; ΔAUGRC −0.0029 / −0.0024 / −0.0052 |
| **K-VNQ-2** AUGRC beats plain vote margin | **FAIL** | 1/3 and that one at 3/5 folds |

**C1 dies twice over, and the second death is the informative one.** K-VGA-1 is missed by a factor of
three. But K-VGA-3 is the result that closes the axis rather than merely failing to open it: the
candidate's entire licence to revisit F47 rested on the claim that the trained relation profile is
*"a genuinely NEW information source not derivable from banked features/votes."* Measured head to
head at identical fitting budget, nesting, operating-point rule and permutation null, the **F47
features are strictly better gating features than the verifier profile on all three datasets**. The
verifier profile is not a new information source for this decision; it is a worse one.

**C2 dies on both its bars**, and it dies against the cheapest possible baseline: the deployed vote's
own margin. The one axis in this sweep that did not have to fight the exchange-rate law lost anyway.

**The one genuinely new positive datum, stated without inflation.** There *is* a real, permutation-
validated gating signal on the disagreement set — `f47ctrl_full` reaches **+0.0269** (HateMM,
p = 0.0050, fold signs `+++++`), **+0.0104** (ZH, p = 0.0050) and **+0.0182** (EN, p = 0.0100). This
is a genuine refinement of F47's epitaph: F47's features are dead as a *per-item channel selector*
but are **not** dead as a *per-item adjudication gate*. It is nonetheless **below the +0.030 bar on
all three datasets**, it is measured in raw train-split space where F66's selection lock and the
campaign's F47/F89 history both say conversion is the failing step, and — decisively — it is a gate
on an adjudicator (F95's) that is **net-negative ungated on all three datasets**. It is an analysis
datum, not a lever, and it must not be promoted as one.

**Routing.** Do not spend GPU. Do not promote to ceremony. Specifically **do not** re-propose:
(a) any verifier-profile gate, selector or router over this decision — K-VGA-3 closes it by
measurement, not by argument; (b) verifier-based selective prediction / abstention / risk ordering —
K-VNQ-1/2 close it, and the free vote margin dominates it; (c) "a better gate model" — the binding
constraint is the *feature source*, and the control that beats it is already in the dead F47 family.

The relational asset is now settled as **analysis-grade only**, exactly as `LITSWEEP6_RELGEN` §5
said it would be if C1 failed: *"If C1 fails, the relational asset is settled as analysis-grade only,
and the campaign should stop trying to convert it."* Three independent conversion attempts have now
been made on it — F95 (replace the vote), C1 (gate the replacement), C2 (read it as risk) — and all
three are negative. **The recommended paper framing is unchanged and is now better supported:** C3
(VEA, evidence ranking for the audit pillar) remains the one legal, unmeasured use of the verifier,
and F95's binding *"NEVER an accuracy claim"* should be carried verbatim.

---

## §7. LIMITATIONS

1. **Raw space, train split, not the deployed head space and not test (L1, inherited from F95 §6).**
   A raw-space null does not logically entail a head-space null. This cuts *against* the negative
   verdict and is stated first for that reason. It is mitigated but not removed by the fact that
   K-VGA-3 is a *relative* comparison between two feature families measured in the same arena under
   the same protocol — a comparison far less sensitive to the arena than an absolute Δacc is.
2. **Single verifier seed, single fold draw (F95 L3, inherited).** `MLP_SEED=0`, `FOLD_SEED=0`, no
   resampling of the F95 verifier. The gate layer adds its own inner-CV and a 200-draw permutation
   null, so the *gate's* variance is controlled; the *verifier's* is not.
3. **Residual coupling in the nesting (declared in §2.4).** The gate never sees the fold it scores,
   but its training items' features were produced by F95 verifiers whose fit sets included that fold's
   items. A doubly-nested verifier (25 fits per dataset instead of 5) would remove this; it was not
   run. The direction of this bias would, if anything, *flatter* the verifier arm — which lost anyway.
4. **The gate can only act on the disagreement set.** Items where the deployed vote and the
   adjudication agree are unreachable by construction, so the oracle ceiling (+0.0726/+0.0535/+0.0893)
   is the hard maximum for this whole family. This is a property of the design, correctly priced in
   §1 before the run, not a discovered limitation.
5. **Two gate model families only** (logistic, shallow GBM), one operating-point rule (max net gain on
   inner folds), one gate-target definition. A different threshold rule — e.g. one targeting precision
   rather than net — is unmeasured; note however that §4.1's required-precision arithmetic bounds what
   any such rule could buy.
6. **C2's baselines are fitted, not off-the-shelf kNN-UE.** kNN-UE's published form combines distance
   and label ratio through its own estimator; here both VNQ and kNN-UE are given the *same* nested
   logistic treatment so the comparison isolates the feature family rather than the combiner. The
   plain vote margin is unfitted and still won — which makes the fitting choice non-load-bearing.
7. **C4 (VSW) was not tasked and was not run**; C3 (VEA) is a writing task and is untouched here.
8. **No test-fitted quantity appears anywhere.** No test label, no test split loaded by any script in
   this record; no oracle used for any operating point (the `oracle` arm is a reported ceiling only).

---

## §8. FILE MANIFEST

| path | contents |
|---|---|
| `scripts/analysis/vga_pregate_emit.py` | per-item emitter, sha256 `a3a41ae7…7ce56`; imports the frozen F95 arms module unmodified; 26 parity asserts per dataset |
| `scripts/analysis/vga_pregate_gate.py` | frozen gate + C2 analysis, sha256 `ea37c57b…4f4e34`; includes the pre-freeze `--selftest` |
| `scripts/analysis/vga_pregate_runner.py` | orchestration only; asserts the frozen gate sha, drives one dataset per short-lived process (reap-safe, F95 runner precedent) |
| `scripts/analysis/vga_pregate_report.py` | reporting only; re-reads the OUT json at 4 dp |
| `scripts/analysis/vga_emit_{hatemm,zh,en}_OUT.json` (+ `.log`) | per-item tables: gold, fold, deployed vote/decision, adjudicated decisions, cosine-shape control, `sc_rank`, 3 feature blocks |
| `scripts/analysis/vga_parts/{hatemm,zh,en}.json` (+ `.log`) | per-dataset gate cells (reap-safe serialisation) |
| `scripts/analysis/vga_pregate_OUT.json` | merged result: 36 arm cells, 36 permutation nulls, C2 curves |

Read-only inputs: `data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_*.pt` (**train split only** —
`dev_seen`/`test_seen` were never opened), `scripts/analysis/mechnov_pairverify.py` and
`scripts/analysis/mechfix_ops.py` (both imported, unmodified, shas asserted at run time),
`scripts/analysis/mechnov_pairverify_*_OUT.json` (parity references, read-only).
Read for context, not modified: `LITSWEEP6_RELGEN.md`, `MECHNOV_PAIRVERIFY_PREGATE.md`,
`MECHFIX_PREGATE_2026-07-27.md`. Nothing under `autoresearch/goal_mllm_plus3/state/` was written.
No file deleted or moved. **Zero GPU, zero SLURM submissions, zero Modal calls, zero test contact.**
