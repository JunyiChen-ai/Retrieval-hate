# RESTRANS PREGATE — C1, the residual-transport vote (de-bias the label field)

> **ERRATUM POINTER (2026-08-05, F120):** the "deployed head train LOO" triple `0.9406 / 0.8915 / 0.8154` used in this file (lines 62, 489, 495) is a **protocol-mixed pooled mean** over val-selected **and** final-epoch checkpoints (MHC-EN final-epoch only), not a deployed-protocol LOO triple. Measured: MHC-ZH **0.9303** (not 0.8915), HateMM **0.9404** on the deployed `-LoRA-curric` lineage, MHC-EN 0.8154 unchanged. See `TARGET_FINDINGS.md` F120. This record is left as written.

**Date:** 2026-07-27 NZST · **Agent:** restrans pregate · **Cost: $0** (CPU only, ≤8 threads,
**zero GPU, zero SLURM, zero Modal, zero training**). Repo sha at freeze time `62efd82`
(working tree dirty). Env: conda `HateVideo`, numpy 1.26.4, scipy 1.17.1, scikit-learn 1.5.2,
torch 2.6.0+cu124 (CPU), faiss.

**Test-split contact: NONE.** The only files this pregate opens are
`data/CLIP_Embedding/<DS>/train_<model>.pt`, `data/gt/<DS>/train.jsonl`, and (HateMM only)
`data/gt/HateMM/hate_spans.json` for the B-c duration feature. `dev_seen` and `test_seen` are
never loaded by `restrans_pregate.py`.

**Binding design source:** `refine-logs/LITSWEEP6_MEMBANK.md` §1(a)-(f). Its §1(e) pregate design
and kill bars are quoted **verbatim** in §2 below, before any number in this document was computed.

---

## §0. WHAT IS UNDER TEST

The deployed decision (`src/utils/metrics.py:262-301`, `src/model/evaluate_rac.py:405-465`, replayed
by the F89-frozen `mechfix_ops.deployed_vote`) is

```
v = Σ_i (2·lab_i − 1)·cos_i·w_i / Σ_i w_i ,   top-20 own-train neighbours,  w = [20,19,…,1]
predict 1 iff v ≥ 0
```

C1 replaces the **label summand** with the **nuisance residual**:

```
s_i = 2·lab_i − 1        ⟶        r_i = s_i − (2·p̂_i − 1)
p̂_i = P̂(hate | transcript volume of bank item i),  fitted on the FITTING FOLDS ONLY, leave-one-out
```

Retrieval, `k`, the rank weights, the threshold and the key space are untouched — the identical 20
neighbours in the identical order are retrieved and **only the transported quantity changes**. This
is LITSWEEP6's §0(vi) gap: F89 de-biased the geometry, F94 re-cut the depth, F95 replaced the
decision rule, and nothing in 63 dead entries has touched the label field the vote transports.

---

## §1. FROZEN HARNESS, AND THE THREE PLACES THE TASKING AND THE RECORD DISAGREE

Implementation: `scripts/analysis/restrans_pregate.py`,
**sha256 `99a770cd372148e5e458df51dd86e0e522d0f76084404af4b9de41f9ecfe2531`**, frozen before any
treatment number was computed. It imports, and does not modify, the F89-frozen
`scripts/analysis/mechfix_ops.py` (**sha256 `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d`**,
the file that passed 15/15 floor parity) and the F95-frozen
`scripts/analysis/mechnov_pairverify.py` for the fold protocol and the raw-space constructors.

### 1.1 Arena — the F95 harness verbatim

LITSWEEP6 §1(e): *"Reuse the **F95 harness verbatim**: `StratifiedKFold(5, shuffle=True,
random_state=0)` over train items, item-disjoint, floor = `mechfix_ops.deployed_vote` … on the same
fitting-fold bank. `p̂` is fitted **inside the fitting folds only**; the held-out fold never
contributes to the base model. Train split only; `dev_seen`/`test_seen` unopened."*

F95's arena is the **banked RAW encoder key space** (seed-independent), **PRIMARY = fused**, with
`text` and `img` secondary. That choice is inherited here for the reason F95 stated: the trained
RGCL head memorises its own train split (LOO train acc 0.998, F47), so a train-side screen in head
space measures memorisation. **[ERRATUM — see the appended ERRATUM at the end of this record. 0.998 is
F47's CLIP head; the deployed Qwen heads measure 0.9406 / 0.8915 / 0.8154. The claim is downgraded, not
vacated, and the raw-space justification is superseded by F113's unsaturated fold-head arena.]** A head-space read is **deliberately not run**: the head saw *every*
train item, so even a fold-disjoint bank leaks in head space. Consequence, stated up front: this
arena has **no seeds** (the raw encoder features are seed-independent), so the per-seed sign pattern
the tasking asked for is replaced by the **per-fold** sign pattern the record's bars specify.

### 1.2 Covariate

`transcript volume` = the F89-T3 frozen definition (`mechfix_run.volume_scalar`): whitespace tokens
of `data/gt/<DS>/train.jsonl` `"text"` for HateMM and MHC-EN, **character count** of the composed
text for MHC-ZH. `log1p` transform, as in F89-T3 and as LITSWEEP6 §1(d) specifies. This is the same
scalar whose bin-wise hate rate ERRPAT-HateMM §4.3 tabulates as 0.1096 → 0.5538.

### 1.3 The three base-model arms (LITSWEEP6 §1(d), all three declared)

LITSWEEP6 §1(d) is explicit that **the base model must NOT be the trained head** (F47: head LOO
train acc 0.998 ⇒ residuals ≈ 0 by construction). The base model is a deliberately weak
nuisance-only predictor.

| arm | estimator | parameters | status |
|---|---|---|---|
| **B-a** *(primary)* | univariate logistic on `log(1+volume)`, sklearn L2 `C=1.0`, `max_iter=1000`, genuine per-item leave-one-out over the fitting folds | 2 | run on all 3 |
| **B-b** | ordered equal-count bins (`N_BINS=10`, quantile edges over fitting-fold items) with **FDS-style Gaussian smoothing across neighbouring bins** (arXiv:2102.09554, `ks=5`, `σ=2.0`); numerator and denominator counts each smoothed, rate = their ratio; genuine LOO | ~10 | run on all 3 |
| **B-c** | logistic on `[log(1+volume), log duration]` | 3 | **HateMM only** |

**B-c scope limit, declared before the run.** A duration source exists only for HateMM
(`data/gt/HateMM/hate_spans.json`, verified 744/744 train coverage). `data/_src_Multihateclip/
{Chinese,English}/annotation(new).json` carries no duration field, so **B-c is NOT RUN on MHC-ZH or
MHC-EN**. This is an artifact limit, not a choice.

### 1.4 Mandatory controls

| id | control | why |
|---|---|---|
| **CTRL-W** | reproduction of the LITSWEEP6 §1(c) ERRPAT worked example (rank-1 correct analogue, ranks 2-20 wrong class, `w=[20..1]`): deployed vote **−0.81** → residual vote **−0.10** | the tasking's named worked example |
| **D1** | the **DEAD** score-level length de-bias (ERRPAT-HateMM §4.3/§6.2, train-LOO fit −0.0016 acc): `logistic(deployed vote, log(1+volume)) → gold`, fitted on the fitting folds using the bank items' own LOO votes, applied to held-out items | closest dead relative; **C1 must beat it** |
| **N1** | shuffled-covariate null, *refit* version: permute the covariate across bank items, then refit `p̂` | destroys the nuisance link at its source |
| **N2** | shuffled-`p̂` null, *spread-preserving* version: fit `p̂` on the TRUE covariate, then permute the `p̂` **assignment** across bank items | the sharper null — it preserves the marginal spread of `p̂` and destroys only the item correspondence, so it asks whether the effect needs the **right** nuisance or merely a spread of summand magnitudes |
| **IMPL** | `residual_vote(…, r = 2·lab−1)` must equal `mechfix_ops.deployed_vote` bit-for-bit (votes, predictions and retrieved index matrix), asserted every fold | proves the treatment differs from the floor **only** in the summand |

Both nulls use one frozen RNG, `NULL_SEED = 20260727`.

### 1.5 Parity gate — and why it is train-side

The tasking requires *"k=20 parity must reproduce recorded numbers at 4dp per cell before any
treatment number, abort otherwise"*. The MECHFIX 15/15 floor-parity cells are **test reads**, and
LITSWEEP6 §1(e) binds this pregate to *"Train split only; dev_seen/test_seen unopened."*
**The record wins.** Parity is therefore discharged without touching test, in two parts, both hard
`assert`s that abort the run:

1. `sha256(mechfix_ops.py)` must equal `635c1312…c83fc8d` — i.e. the deployed-vote replay used here
   *is* the object that passed 15/15 floor parity at 4 dp in F89/MECHFIX §2.4.
2. The deployed floor recomputed inside this harness must reproduce **F95's recorded train-side
   numbers at 4 dp, per cell**, read directly from
   `scripts/analysis/mechnov_pairverify_{hatemm,zh,en}_OUT.json`: pooled `acc_deployed` and
   `mF1_deployed`, **all five per-fold `acc_deployed` values**, and the integer counts
   `n_deployed_wrong` and `n_pathology_pop` — for every dataset × every space. That is
   **9 pooled-accuracy + 9 pooled-mF1 + 45 per-fold + 18 count cells = 81 asserted cells.**

### 1.6 Core-error id lists

The tasking asks for the exchange rate *"against the errpat stable-core id lists"*. Those lists
(`errpat_hatemm_forensics_OUT.json` `wrong_3of3_ids`, `errpat_zh_taxonomy_OUT.json`,
`errpat_mhc_en_out.json` `consensus_error_ids`) are **test-split** objects; using them here would
require opening test, which the record forbids. **The record wins.** The substitute is the
record's own bar-2 population, which is the same object measured on train items: the **F95
pathology population** — deployed-wrong held-out items whose nearest same-gold-class bank item sits
within **rank 5** by full-space cosine. F95 §4 established that this population reproduces the
ERRPAT diagnosis exactly in this arena (72-92 % of all deployed errors; median analogue rank 2-3
over errors).

---

## §2. FROZEN BARS — quoted verbatim from LITSWEEP6_MEMBANK.md §1(e)

> **Frozen bars (declare before running, F95 style):**
> 1. **Primary:** pooled held-out-item Δacc vs deployed ≥ **+0.010** on ≥1 dataset, 5/5 folds Δ ≥ 0,
>    ≥3/5 strictly positive. (1 item = 0.0013/0.0017/0.0018, so +0.010 = 7.4/5.8/5.5 items.)
> 2. **Exchange rate ≥ 1.2** on the pathology population (deployed-wrong items whose nearest
>    same-gold-class bank item is within rank 5) — set above F95's ceiling of 1.17 *on purpose*: a
>    candidate that lands inside the band every symmetric operator has already occupied has told us
>    nothing new.
> 3. **Degeneracy control (fires a KILL, not a caveat):** report `sd(p̂)` over the bank and the AUC of
>    `p̂` against the gold train label. If `p̂` is near-constant, C1 is a global threshold shift in
>    disguise and is dead by the existing ban regardless of its Δ.
> 4. **Stratum-honesty control:** report Δacc **separately for the short-transcript and long-transcript
>    halves**. C1 necessarily makes short queries more likely to be called hate. If the gain is
>    short-recall bought at exactly matching long-precision cost, that is the exchange-rate law again
>    and must be reported as such, not netted out.
> 5. **Class-balance sanity** (F95 control 4): decision positive rate vs bank positive rate.

Operationalisation notes, frozen with the bars:

* **Bar 1** is read on the **primary space (fused)**, pooled over all held-out items (each train item
  held out exactly once), against the deployed floor on the same fitting-fold bank. Secondary spaces
  are reported but cannot carry the bar.
* **Bar 2**'s exchange rate is F95's definition — `fixed / broken` over all held-out items, reported
  together with the fraction of the pathology population fixed (F95 §4's table shape), so this
  pregate's number is directly comparable to F95's 0.53-0.95 primary cells and its 1.1667 ceiling.
* **Bar 3** is read per fold on the fitting-fold bank: `sd(p̂)`, min/mean/max, and `AUC(p̂, gold)`.
* **Bar 4**'s short/long split is the **median of `log1p(volume)` over the fitting-fold items** of
  each fold, applied to that fold's held-out items (no held-out quantity enters the split).

Gate order: §0-§2 written and both sha256 frozen → parity gate (81 cells) → `p̂` statistics →
treatments and controls → verdict. Machine output: `scripts/analysis/restrans_pregate_OUT.json`,
run log `scripts/analysis/restrans_pregate_OUT.log`. Every number in §3 onward is re-read from that
JSON at report time, 4 dp.

<!-- ============ EVERYTHING BELOW THIS LINE WAS WRITTEN AFTER THE RUN ============ -->

---

## §3. PARITY GATE — 81/81 CELLS PASS

`sha256(mechfix_ops.py)` asserted equal to `635c1312…c83fc8d` — the F89-frozen file that carries the
15/15 test-side floor parity of MECHFIX §2.4. The deployed floor recomputed inside this harness then
reproduced **every** F95 train-side number, read directly from
`mechnov_pairverify_{hatemm,zh,en}_OUT.json`, at 4 dp. Any mismatch aborts the run.

| dataset × space | pooled acc / mF1 (anchor = recomputed) | per-fold acc (anchor = recomputed) | n wrong | n pathology |
|---|---|---|---|---|
| HateMM × **fused** | 0.8441 / 0.8419 | 0.7987, 0.8322, 0.8926, 0.8255, 0.8716 | 116 | 88 |
| HateMM × text | 0.8441 / 0.8422 | 0.8054, 0.8322, 0.8792, 0.8188, 0.8851 | 116 | 87 |
| HateMM × img | 0.7688 / 0.7561 | 0.7315, 0.7718, 0.8389, 0.7450, 0.7568 | 172 | 136 |
| MHC-ZH × **fused** | 0.8480 / 0.8281 | 0.8534, 0.8534, 0.9224, 0.8017, 0.8087 | 88 | 79 |
| MHC-ZH × text | 0.8636 / 0.8442 | 0.8621, 0.8621, 0.8966, 0.8879, 0.8087 | 79 | 68 |
| MHC-ZH × img | 0.7012 / 0.6083 | 0.6552, 0.7241, 0.7672, 0.6552, 0.7043 | 173 | 158 |
| MHC-EN × **fused** | 0.7796 / 0.7286 | 0.7091, 0.7545, 0.8091, 0.8182, 0.8073 | 121 | 109 |
| MHC-EN × text | 0.8106 / 0.7785 | 0.8273, 0.7636, 0.8364, 0.8273, 0.7982 | 104 | 84 |
| MHC-EN × img | 0.6995 / 0.5561 | 0.7000, 0.7000, 0.6818, 0.7273, 0.6881 | 165 | 150 |

**IMPL gate.** `residual_vote(…, r = 2·lab−1)` was asserted equal to `mechfix_ops.deployed_vote` in
votes, predictions **and** the retrieved index matrix, on every one of the 45 (dataset × space × fold)
cells. The treatment therefore differs from the floor in the summand and in nothing else.

Split sizes and base rates re-read at report time: HateMM 744 items, pos-rate 0.4005, transcript
words min/median/max 0 / 128 / 13677, duration available 744/744; MHC-ZH 579, 0.3109, chars
3 / 106 / 708, **no duration source**; MHC-EN 549, 0.3060, words 2 / 69 / 273, **no duration source**.
Per bar 1's own arithmetic, one item = 0.0013 / 0.0017 / 0.0018.

---

## §4. RESULTS — the primary space (fused), pooled over all held-out train items

`Δ` = arm − deployed floor on the identical fitting-fold bank. Fold signs are per fold (no seeds
exist in this arena; the raw encoder features are seed-independent).

| dataset | arm | Δacc | ΔmF1 | fold signs | folds Δ≥0 | fixed / broken | net | **exchange rate** | pathology fixed | decision pos-rate |
|---|---|---|---|---|---|---|---|---|---|---|
| **HateMM** | *deployed floor* | *0.8441 / 0.8419* | — | — | — | — | — | — | — | 0.4812 (bank 0.4005) |
| | **C1 · B-a** | **−0.0188** | −0.0181 | `+----` | 1/5 | 10 / 24 | −14 | **0.4167** | 10 / 88 | 0.5081 |
| | **C1 · B-b** | **−0.0161** | −0.0154 | `+----` | 1/5 | 6 / 18 | −12 | **0.3333** | 6 / 88 | 0.5081 |
| | **C1 · B-c** | **−0.0188** | −0.0181 | `+----` | 1/5 | 10 / 24 | −14 | **0.4167** | 10 / 88 | 0.5081 |
| | D1 (dead relative) | +0.0215 | +0.0188 | `++++-` | 4/5 | 34 / 18 | +16 | 1.8889 | 33 / 88 | 0.4113 |
| | N1 (perm-cov null) | −0.0282 | −0.0271 | `-----` | 0/5 | 6 / 27 | −21 | 0.2222 | 6 / 88 | 0.5255 |
| | N2 (perm-p̂ null) | −0.0242 | −0.0230 | `-----` | 0/5 | 7 / 25 | −18 | 0.2800 | 7 / 88 | 0.5242 |
| **MHC-ZH** | *deployed floor* | *0.8480 / 0.8281* | — | — | — | — | — | — | — | 0.3489 (bank 0.3109) |
| | **C1 · B-a** | **−0.0863** | −0.0736 | `-----` | 0/5 | 24 / 74 | −50 | **0.3243** | 24 / 79 | 0.5181 |
| | **C1 · B-b** | **−0.0846** | −0.0720 | `-----` | 0/5 | 24 / 73 | −49 | **0.3288** | 24 / 79 | 0.5164 |
| | D1 (dead relative) | −0.0069 | −0.0153 | `-+---` | 1/5 | 12 / 16 | −4 | 0.7500 | 12 / 79 | 0.3005 |
| | N1 (perm-cov null) | −0.0760 | −0.0641 | `-----` | 0/5 | 23 / 67 | −44 | 0.3433 | 23 / 79 | 0.5043 |
| | N2 (perm-p̂ null) | −0.0794 | −0.0674 | `-----` | 0/5 | 23 / 69 | −46 | 0.3333 | 23 / 79 | 0.5078 |
| **MHC-EN** | *deployed floor* | *0.7796 / 0.7286* | — | — | — | — | — | — | — | 0.2605 (bank 0.3060) |
| | **C1 · B-a** | **−0.1002** | −0.0560 | `-----` | 0/5 | 52 / 107 | −55 | **0.4860** | 51 / 109 | 0.5501 |
| | **C1 · B-b** | **−0.1093** | −0.0644 | `-----` | 0/5 | 52 / 112 | −60 | **0.4643** | 50 / 109 | 0.5592 |
| | D1 (dead relative) | −0.0091 | −0.0225 | `-+-+-` | 2/5 | 7 / 12 | −5 | 0.5833 | 7 / 109 | 0.2259 |
| | N1 (perm-cov null) | −0.1038 | −0.0590 | `-----` | 0/5 | 53 / 110 | −57 | 0.4818 | 52 / 109 | 0.5574 |
| | N2 (perm-p̂ null) | −0.0965 | −0.0522 | `-----` | 0/5 | 53 / 106 | −53 | 0.5000 | 52 / 109 | 0.5501 |

Per-fold Δacc for the primary arm C1·B-a: HateMM `[+0.0067, −0.0067, −0.0335, −0.0470, −0.0135]`;
MHC-ZH `[−0.0948, −0.1120, −0.1552, −0.0258, −0.0435]`; MHC-EN `[−0.0727, −0.1636, −0.1091, −0.1182,
−0.0367]`.

**Secondary spaces (`text`, `img`) do not rescue it.** Δacc is negative in **21 of 21**
arm × dataset × space cells. The best C1 Δacc anywhere in the battery is **−0.0013** (HateMM × img ×
B-b, i.e. one item) and the best C1 exchange rate anywhere is **0.9474** (same cell); the range over
all 21 cells is **0.2647 – 0.9474**. **No C1 cell has a positive Δacc and no C1 cell reaches an
exchange rate of 1.0, let alone the 1.2 bar.** One cell has a positive ΔmF1 — the same
HateMM × img × B-b, **+0.0037** — and it is the only positive number C1 produced anywhere. Full
per-space numbers: `restrans_pregate_OUT.json` → `datasets.*.spaces.*.pooled`.

### 4.1 Effect concentration — C1 lands exactly where F94 said the live population is

LITSWEEP6 §0(v): ranks 11-20 flip zero predictions, so a candidate whose effect lives in the tail is
pre-dead. Measured, primary space: of C1·B-a's changed decisions, **34 / 34** (HateMM),
**98 / 98** (MHC-ZH) and **158 / 159** (MHC-EN) fall on items whose nearest same-gold-class bank
item sits within rank 5. **The predicted concentration is confirmed exactly.** C1 reaches the
pathology population and then breaks 2.4× / 3.1× / 2.1× more of it than it fixes. This is
LITSWEEP6 §0(iii) for the ninth time: *reaching the pathology is not the hard part.*

---

## §5. CONTROLS

### 5.1 CTRL-W — the ERRPAT worked example (honest reproduction, with one correction)

Config per LITSWEEP6 §1(c): correct analogue at rank 1, nineteen wrong-class neighbours at ranks
2-20, `w = [20..1]`, uniform cosines (the space is cone-collapsed, deployed top-1 cosine
0.9439-0.9686).

| quantity | value |
|---|---|
| deployed vote | **−0.8095** (record: −0.81 ✔) |
| residual vote at `p̂` = 0.1096 (the record's own cited 0-1-word bin rate, ERRPAT-HateMM §4.3) | **−0.0287** |
| residual vote at `p̂` = 0.2926 (2-50-word bin) | −0.3947 |
| the `p̂` that reproduces the record's stated **−0.10** | **0.1452** |

The mechanism claim reproduces **a fortiori** — the vote is compressed toward the boundary by 8× at
the record's stated figure and by 28× at the record's own cited bin rate — but the specific number
−0.10 corresponds to `p̂ = 0.1452`, not to the 0.1096 the same section cites. Recorded as an erratum,
not a defect in the mechanism.

**What the closed form actually shows, and it is the whole verdict in one line.** With uniform
cosines and a neighbourhood whose members share a `p̂`, the identity is exact:

```
v_res  =  v_dep  −  (2·p̂ − 1)
```

i.e. **the worked example is arithmetically a threshold move.** The item-level content of C1 lives
entirely in the *dispersion* of `p̂` inside a single neighbourhood, and the worked example has none.
§5.3 measures how much there is in the real data.

### 5.2 Shuffled-covariate nulls — the effect does not need the covariate

Primary space, C1·B-a versus its own nulls:

| dataset | C1·B-a Δacc | N1 (permute covariate, then refit) | N2 (fit on true covariate, then permute the assignment) | C1 − N2 |
|---|---|---|---|---|
| HateMM | −0.0188 | −0.0282 | −0.0242 | **+0.0054** (4 items) |
| MHC-ZH | −0.0863 | −0.0760 | −0.0794 | **−0.0069** (C1 *worse* than its null) |
| MHC-EN | −0.1002 | −0.1038 | −0.0965 | **−0.0037** (C1 *worse* than its null) |

N2 is the sharper null by construction: it preserves the marginal spread of `p̂` exactly and destroys
only the item↔`p̂` correspondence. **C1 does not separate from it on any dataset, and is worse than it
on two of three.** The tasking's requirement — *"permuting the covariate must kill the effect, else
it is not transporting the nuisance"* — is inverted here: there is no effect for the permutation to
kill, because what C1 transports is the covariate's **mean**, which permutation leaves untouched.

### 5.3 Degeneracy — bar 3 fires, by direct measurement rather than by inference

`scripts/analysis/restrans_pregate_diag.py`
(sha256 `70c399d9fccf8e769fab679597dffdd3daa4a7467581ac356a02a7a47629ea16`) →
`restrans_pregate_diag_OUT.json`. **Post-hoc; adds no arm, promotes nothing.**

**(a) The constant-shift twin.** Run the identical operator with `p̂_i` replaced by its own bank mean
— a pure global decision-threshold shift, a lever measured dead on all three datasets
(ERRPAT-HateMM §2.1 +0.0000/+0.0016; ERRPAT-ZH §3 test-fitted oracle +0.0201; ERRPAT-EN §6.2
−0.0083 on 0/6 arms). Agreement with C1·B-a's predictions, pooled over all held-out items:

| dataset | agree(C1, pure threshold shift) | acc deployed | acc C1 | acc pure shift |
|---|---|---|---|---|
| HateMM | **707 / 744 = 95.03 %** | 0.8441 | 0.8253 | 0.8159 |
| MHC-ZH | **566 / 579 = 97.75 %** | 0.8480 | 0.7617 | 0.7703 |
| MHC-EN | **546 / 549 = 99.45 %** | 0.7796 | 0.6794 | 0.6812 |

**C1 is a global threshold shift in disguise on 95-99 % of items.** Bar 3 states the consequence:
*"C1 is a global threshold shift in disguise and is dead by the existing ban regardless of its Δ."*
It fires.

**(b) The decomposition that explains it.** Writing `2·p̂_i − 1 = c̄ + δ_i`, the vote shift splits into
a near-constant term `−c̄·(Σcos·w/Σw)` and an item-level term `−Σδ_i cos_i w_i/Σw`:

| dataset | `c̄` | sd(`c̄` term) across queries | sd(item term) across queries |
|---|---|---|---|
| HateMM | −0.197 to −0.200 | 0.0034-0.0154 | 0.203-0.218 |
| MHC-ZH | −0.378 to −0.379 | 0.0085-0.0097 | 0.041-0.079 |
| MHC-EN | −0.385 to −0.390 | 0.0052-0.0071 | **0.0019-0.0162** |

The constant term moves every vote by **+0.18 (HateMM) to +0.36 (ZH/EN)** — enormous against a vote
scale of ±1 and a deployed decision boundary at 0. On MHC-EN the item-level term is **20-200× smaller
than the shift**, which is why 99.45 % of its decisions are the threshold move.

**(c) `sd(p̂)` and `AUC(p̂, gold)`, the bar's own two statistics**, per fold on the fitting-fold bank:

| dataset | sd(`p̂`) | AUC(`p̂`, gold train label) |
|---|---|---|
| HateMM | 0.1337 – 0.1450 | 0.6495 – 0.6703 |
| MHC-ZH | 0.0335 – 0.0624 | 0.5268 – 0.5752 |
| MHC-EN | **0.0024 – 0.0154** | **0.0000, 0.3373, 0.4240, 0.4314, 0.0000** |

MHC-EN's `p̂` is near-constant *and* its AUC is below chance in all five folds, exactly 0 in two.
That is not a bug: the fitted logistic coefficient on `log1p(volume)` is **−0.0003 / +0.0265 /
−0.0638 / −0.0649 / −0.0071**, i.e. ≈ 0, so the leave-one-out `p̂` degenerates to the leave-one-out
base rate `(Σy − y_i)/(n−1)`, which is a strictly *decreasing* function of the item's own label and
therefore has AUC exactly 0. Verified, not asserted: `corr(p̂, pure LOO intercept)` = **+0.8802** and
**+0.7454** in the two folds where AUC = 0.

### 5.4 The dead relative — C1 loses to it in 9 of 9 cells

D1 = `logistic(deployed vote, log(1+volume)) → gold`, fitted on the fitting folds from the bank
items' own LOO votes: the score-level length de-bias that ERRPAT-HateMM §4.3/§6.2 measured dead
(train-LOO fit −0.0016 acc, dev fit +0.0000). C1 must beat it. **C1 loses in 8 of the 9
dataset × space cells**, by +0.0403 (HateMM fused), +0.0794 (ZH fused) and +0.0911 (EN fused) of
accuracy. The single exception is **HateMM × img**, where C1·B-b's −0.0013 edges D1's −0.0027 by
0.0014 — **one item**, in the weakest space, with both arms below the floor. It is noise, not a win,
and it is recorded so the "8 of 9" is not read as "9 of 9".

### 5.5 Stratum honesty (bar 4) and class balance (bar 5)

| dataset | Δacc short half | Δacc long half | deployed pos-rate → C1 pos-rate | bank pos-rate |
|---|---|---|---|---|
| HateMM | −0.0323 | −0.0054 | 0.4812 → 0.5081 | 0.4005 |
| MHC-ZH | −0.0859 | −0.0868 | 0.3489 → **0.5181** | 0.3109 |
| MHC-EN | −0.1011 | −0.0993 | 0.2605 → **0.5501** | 0.3060 |

Bar 4 anticipated a short-recall gain bought with long-precision loss, to be reported rather than
netted out. **There is no trade to report: both halves lose on all three datasets.** Bar 5: no
collapse to one class, but a large prior drift — on ZH and EN C1 pushes the decision positive rate to
0.52-0.55 against bank base rates of 0.31, a 1.7-1.8× over-prediction of hate. That drift is the
constant shift of §5.3(b) arriving at the decision.

---

## §6. THE MECHANISM-LEVEL REASON, WHICH IS THE ONE THING WORTH KEEPING

CP1 — the bank's monotone length-conditional class prior — is a **HateMM-specific** fact. Measured on
the train split of each dataset with the frozen covariate:

| dataset | Spearman ρ(transcript volume, gold label) | p | fitted B-a logistic coefficient, 5 folds |
|---|---|---|---|
| HateMM | **+0.2842** | 2.74e-15 | +0.3192 … +0.3518 |
| MHC-ZH | **−0.1152** | 0.00553 | −0.3600 … −0.1891 (**sign-inverted**) |
| MHC-EN | **−0.0050** | 0.906 | −0.0649 … +0.0265 (**null**) |

The HateMM bin table reproduces ERRPAT-HateMM §4.3 exactly from `data/gt/HateMM/train.jsonl`:
0-1 words n=73, 8 hate, **P=0.1096**; 2-50 n=188, **0.2926**; 51-150 n=136, **0.3824**;
151-400 n=217, **0.5115**; 401+ n=130, **0.5538**. So the diagnosis C1 was built from is real —
**on one of three datasets.** On MHC-ZH the association runs the *other way* (longer transcript →
*less* hate) and on MHC-EN there is none at all. A residual computed against a covariate that carries
no conditional information is, by construction, the leave-one-out base rate, and transporting it is
transporting a constant. §5.3 measured exactly that.

**Consequence for the rest of the LITSWEEP-6 queue, and it is binding.** §6 of LITSWEEP6 recommends
writing C2 *after* C1 "because if C1 shows the label field is correctable then C2's placement
criterion can use `p̂` and becomes much better targeted." **It cannot.** `p̂` is a usable targeting
signal on HateMM only; on MHC-ZH it would target the wrong direction and on MHC-EN it would target
noise. Any C2 prereg must choose a placement criterion that does not rest on the length covariate.

---

## §7. VERDICT — KILL

| bar | requirement | measured | verdict |
|---|---|---|---|
| **1 — primary** | pooled Δacc ≥ +0.010 on ≥1 dataset, 5/5 folds Δ≥0, ≥3/5 strictly positive | best C1 in the primary space **−0.0161** (HateMM B-b); best anywhere **−0.0013**; fold signs 1/5, 0/5, 0/5 ≥0 | **FAIL** |
| **2 — exchange rate** | ≥ 1.2 on the pathology population | C1 range **0.2647 – 0.9474** over all 21 cells; primary space 0.3243 – 0.4167 | **FAIL** |
| **3 — degeneracy (KILL)** | `p̂` near-constant ⇒ dead by the existing ban | C1 agrees with a pure global threshold shift on **95.03 / 97.75 / 99.45 %** of items; MHC-EN sd(`p̂`) 0.0024-0.0154 with AUC 0.0-0.4314 | **FIRES → KILL** |
| **4 — stratum honesty** | report short/long separately, do not net out | **both halves lose** on all three datasets | reported; no trade exists |
| **5 — class balance** | decision pos-rate vs bank pos-rate | no collapse, but 0.5181 / 0.5501 against bank 0.3109 / 0.3060 | drift recorded |
| control — nulls | permuting the covariate must kill the effect | C1 is **indistinguishable from both nulls**, and worse than N2 on 2/3 | **FAIL** |
| control — dead relative | C1 must beat the length-logistic score de-bias | C1 loses in **8/9** dataset × space cells; the 9th is a 1-item tie in the weakest space with both arms below the floor | **FAIL** |

**C1 — the residual-transport vote — is CLOSED.** Cost: **$0**, CPU only, zero GPU, zero SLURM, zero
Modal, zero test-split contact. It is the 64th pre-registered negative and the **tenth** datum for
LITSWEEP6's law (iii): a mechanism that reaches the pathology population cleanly (100 % of its
changed decisions land at ranks 1-5, as F94 predicted) and still pays for every fix at 0.26-0.95.

The kill is *mechanistic*, not budgetary, and it is stronger than a null: C1 did not fail to move
anything (F89-T1's failure mode) — it moved 34-159 decisions per dataset, all in the right
population, and was measured to be a **decision-threshold shift wearing an item-level costume** on
95-99 % of them. That closes the label-field axis of LITSWEEP6 §0(vi) at zero GPU cost.

### 7.1 One observation for routing, deliberately not dressed up

The **dead relative D1** produced exchange rates **above the 1.2 bar** in this arena — HateMM fused
**1.8889** (34 fixed / 18 broken, net +16) and HateMM text **2.2353** (38 / 17, net +21), MHC-ZH text
1.5556, MHC-ZH img 1.2857, MHC-EN img 1.2273 — with pooled Δacc +0.0215 (fused, 4/5 fold signs) and
+0.0282 (text, 5/5 fold signs) on HateMM. F95's ceiling over 36 cells was 1.1667.

**Four reasons this is a note and not a candidate**, all of which must travel with the number:
(i) D1 is a *measured-dead* lever — a monotone recalibration of one scalar, inside the existing ban;
(ii) it was measured dead in the **deployed head space on test** (−0.0016 train-LOO fit, +0.0000 dev
fit), and this is a **raw-space, train-side** screen, so the arenas are not comparable;
(iii) +0.0215/+0.0282 is far below the house promotion bar of +0.030 acc **and** +0.030 mF1;
(iv) it is a **query-side** correction, i.e. the opposite of where C1 and the whole membank lens
looked. If it is worth anything it is as the observation that the fix/break asymmetry this campaign
has never achieved appeared on the query side, not the bank side — and only where the covariate
actually carries information (HateMM). **No GPU should be spent on it without a user ruling on the
ban.**

### 7.2 Routing recommendation

**C1 closed. Route next to C4 (aggregate-then-compare subspace residual)** — LITSWEEP6 §6 nominated
C1 and C4 to run in parallel precisely because they attack different pathologies, and C4's own
$0 pregate is untouched by anything measured here. **C2's prereg must be rewritten**: its placement
criterion cannot use `p̂` (§6 above). C3 and C5 are unaffected.

---

## §8. LIMITATIONS

1. **Arena.** Raw banked encoder space, not the deployed head space, per the F95 precedent (F47: head
   LOO train acc 0.998). The deployed-arena behaviour of C1 is therefore *inferred*, not measured.
   The inference is safe in this instance because §5.3's degeneracy is a property of `p̂` and of the
   cone-collapsed cosine profile, both of which are stronger in head space (top-1 cosine 0.999852,
   ERRPAT-HateMM §2), not weaker — the constant term would dominate *more*.
2. **No seeds.** The raw features are seed-independent, so the sign evidence is 5 folds, not 3 seeds.
3. **Train-side only.** No test number exists in this document, by design (LITSWEEP6 §1(e)).
   The errpat stable-core id lists could not be used for the same reason (§1.6).
4. **B-c ran on HateMM only** — no duration source exists for MHC-ZH or MHC-EN (§1.3). Where it ran
   it was indistinguishable from B-a.
5. **The worked example's −0.10 is an erratum** in LITSWEEP6 §1(c) (§5.1); it does not affect the
   mechanism argument, which reproduces more strongly at the record's own cited bin rate.
6. **D1's positive (§7.1) is a single arena on a dead lever** and is not offered as a result.

---

## ⚠ ERRATUM (appended 2026-07-28, closeout) — the inherited "head memorises train at LOO ≈ 0.998" premise is a **CLIP** number

**No verdict moves.** This is a framing correction to an inherited premise, not to any measurement
taken in this record.

**The error.** This record repeats — from `mechnov_pairverify.py:21-25` / F95 — that the trained RGCL
head *"memorises its own train split (LOO train acc 0.998, F47)"*, and uses it to justify screening in
the **raw** encoder key space rather than the deployed head space.

**0.998 is F47's CLIP head, not the deployed head.** F47's own `ban_scope`
(`directions_tried.json:171`) reads *"train-supervised = memorization-degenerate target, **CLIP LOO
0.998**"*, and the memory index pairs it with *"vs **Qwen 0.800**"*. The deployed system does not use
the CLIP head.

**The deployed Qwen heads, newly computed** (`INSTRUMENT_VALIDATION_RECON.md` §0.2, F111; re-read from
`scripts/analysis/mechfix_{hatemm,zh,en}_OUT.json` → `train_side_sanity.deployed_loo_train_acc`):

| | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| **deployed head train LOO** | **0.9406** | **0.8915** | **0.8154** |
| raw-arena deployed train LOO | 0.8441 | 0.8480 | 0.7796 |
| gap between the two arenas | +0.0965 | +0.0435 | +0.0358 |

**The two arenas differ by 3.6–9.7 accuracy points on the same train items, not by the 0.998-vs-0.84
chasm the premise asserts.** The argument *"a train-side screen in head space measures memorisation"*
is therefore **weaker than stated — downgraded, not vacated**: 0.9406 against a 0.8441 raw floor still
means the head reproduces its own train split far better than its deployed test behaviour.

**CONSEQUENCE 1 — the raw-space screening justification is superseded.** The saturation claim applies
**only to full-train LOO**. `HEADSPACE_TRANSFER_PREGATE.md` (F113) demonstrates the fix nobody used:
**train the head on 4/5 of the train split and query it with the held-out fifth.** That **fold-head
arena is unsaturated**, is a strictly better proxy for deployment than the raw arena, and costs
**~35 s of CPU per fold-head**. The existing `mechfix_ops` / `vsw_pregate` battery runs in it
unmodified. **The head space was available the whole time**, and F113 recommends it become the default
`$0` pregate arena.

**CONSEQUENCE 2 — F107's Q1 argument depended on this figure and has been adjudicated.**
`HEADCOV_PREGATE_RECORD.md` §6.1 claimed *"the objective is already at its optimum on its own training
signal, with ≤0.002 of headroom"*. On the corrected figures the remaining train-side headroom is
**0.0594 / 0.1085 / 0.1846** — 30× / 54× / 92× larger. That step is **RETRACTED**; F107's conclusion
(the metric channel is closed) **survives but is SCOPED and WEAKENED — it is now empirical, not
analytic**, resting on the F75/NCA isomorphism (a measured GPU negative) plus a weak observational
conversion bound (R² = 0.027, MHC-ZH dev only) plus F113's head-space fitting evidence. **F107 must no
longer be cited as a theory-level door-closer.** See `HEADCOV_PREGATE_RECORD.md` §6.1 ERRATUM in full.

**Provenance note.** `scripts/analysis/mechnov_pairverify.py:21-25` still carries the wrong premise and
has been **deliberately left byte-identical**: its sha256 `77b0defd…b7240d` is asserted at run time by
five scripts, so editing even a comment would break the reproducibility of F95, F97, F98, F105, F112
and F113 at once. The correction lives in `MECHNOV_PAIRVERIFY_PREGATE.md` §E.1–E.3.

*Authority: `INSTRUMENT_VALIDATION_RECON.md` §0.2 (F111) · `HEADSPACE_TRANSFER_PREGATE.md` §8 (F113).
Ledger: F114. `$0` — no GPU, no SLURM, no Modal, no training, no test contact.*
