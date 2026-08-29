# VSW ASYMMETRY — FORENSIC RECON (zero-GPU, TRAIN split only)

**Agent:** vsw-asymmetry-recon · **Date:** 2026-07-28 NZST · **Cost $0** (CPU ≤4 threads; zero GPU,
zero SLURM, zero Modal, zero training of any deployed arm).
**Question:** why does HateMM convert under VSW (+0.0255, p = 0.0050) and MHC-ZH not — mechanistically
— and does that reason name anything actionable?

**Test-split contact: NONE.** The only data files this recon opened are
`data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_<model>.pt`.

**Write scope.** This recon wrote **only** this file plus scratch under the session scratchpad. It did
not write, edit, run or delete `refine-logs/VSW_PREGATE_RECORD.md`, `scripts/analysis/vsw_*`, or
`findings.jsonl` — the vsw-pregate agent owns those and they were opened read-only.

---

## 0. VERDICT (up front)

**The asymmetry is fully explained, and it names NOTHING actionable. All four rulings are NO.**

1. **Mechanism (one sentence).** VSW's net on a dataset is `n_ERR·(fix yield) − n_COR·(break exposure)`;
   the **fix yield is statistically identical on all three datasets** (0.250 / 0.227 / 0.264 at a matched
   re-weighting radius, newly computed), and the **break exposure differs by 5.3×** (0.0127 / 0.0448 /
   0.0678) — so HateMM converts **not because its errors are more fixable but because its correct set is
   3.5–5.3× less fragile**, which is a property of the *deployed vote's confidence profile*, not of the
   verifier, not of `n`, not of the cone, and not of the base rate.
2. **(a) Second dataset: NO — HateMM is structurally the only one.** MHC-ZH's entire declared operator
   space with full hindsight tops out at **+0.0069** (23 % of bar) and MHC-EN at **+0.0164** (55 %)
   (`VSW_PREGATE_RECORD.md:596-600`, re-verified this session from `vsw_main_*_OUT.json`). A counterfactual
   that hands ZH **HateMM's entire error/correct structure** still yields only **+0.0240** — ZH has 88
   harvestable errors in 579 items and the arithmetic does not reach +0.030 even then.
3. **(b) Modification to lift HateMM over +0.030: NO.** The whole λ continuum's maximum net is **+21
   items** (`pow`, λ = 3) and the bar needs **≥ 23**; `exp` λ = 4 reaches exactly 23 (+0.0309) but the
   nested selector already picks λ = 4 in 4 of 5 folds and still lands at +0.0255 — the residual is
   **fold heterogeneity, i.e. tuning**, not a mechanism. The only measured objects with net ≥ 23 are
   per-fold-oracle λ (+0.0349) and per-item λ (+0.0685, newly computed) and **both are per-item/per-fold
   selection**, closed by F47/F66/F97-K-VGA-3 and shown by F98 to deliver ~0 regardless of ceiling.
4. **(c) New non-VSW candidate from the mechanism: NO, and the prompt's "restore magnitude to a
   collapsed cone" premise is measured FALSE.** Deleting the cosine magnitude from the deployed vote
   entirely (`cos_i := 1`) changes accuracy by **−0.0013 / +0.0000 / −0.0018** and agrees with the
   deployed vote on **99.60 / 99.65 / 99.82 %** of items (newly computed) — there is no magnitude
   information to restore. VSW's multiplier is **near-orthogonal** to the cosine inside the top-20
   (median per-query Spearman **0.0767 / 0.0917 / 0.0962**, newly computed), so it is supplying an
   orthogonal relational order, not repairing a magnitude channel. F89-T2b's whitening trap is therefore
   not even the binding objection — the object it was meant to repair is decision-inert here.
5. **(d) F98 ban_scope amendment: a small one, and NOT the one the tasking assumed.** F98's re-weighting
   clause (a) carries **no** "with any feature family" qualifier — that phrase belongs to clause (b),
   which covers *per-item selectors/routers/gates* (`AGGNET_PREGATE_RECORD.md:694-698`). F98's own
   scope limiter is explicit: *"the closure covers operators whose input is the (cosine, label) profile
   of the deployed top-20"* (`:711-713`). VSW was **outside** that by information content and is now
   measured. The amendment is **additive, not corrective** — exact wording in §7.

---

## 1. INPUT VERIFICATION — every prompt number re-read from source, plus two premise corrections and one new result

### 1.1 The prompt's numbers check out

Re-read from `scripts/analysis/vsw_main_{hatemm,zh,en}_OUT.json` (`result.arms`) and recomputed from
the 200 raw draws in `scripts/analysis/vsw_perm_{ds}_OUT.json` (`draws[*].arms[*].dacc`,
p = (#{null ≥ obs} + 1)/(200 + 1)):

| dataset | arm | Δacc | **p (recomputed)** | null mean ± sd | null max | frac null ≥ 0 |
|---|---|---|---|---|---|---|
| HateMM | VSW_pow | **+0.0255** | **0.0050** | −0.00051 ± 0.00526 | +0.0134 | 0.570 |
| HateMM | VSW_exp | **+0.0255** | **0.0050** | +0.00045 ± 0.00508 | +0.0148 | 0.590 |
| HateMM | VSW_lin | +0.0188 | 0.0050 | +0.00067 ± 0.00465 | +0.0148 | 0.630 |
| MHC-ZH | VSW_pow | −0.0017 | 0.5522 | −0.00376 ± 0.00523 | +0.0069 | 0.525 |
| MHC-ZH | VSW_exp | −0.0138 | 0.9403 | −0.00378 ± 0.00533 | +0.0086 | 0.445 |
| MHC-ZH | VSW_lin | +0.0000 | 0.3532 | −0.00376 ± 0.00519 | +0.0104 | 0.350 |

Every figure in the tasking reproduces to 4 dp. The null is informative as claimed (57 % of HateMM
draws reach ≥ 0; contrast F98, where **not one** of 300 draws reached zero).

### 1.2 NEW: the MHC-EN battery has since completed, and it is a null

Newly computed from `scripts/analysis/vsw_perm_en_OUT.json` (200 draws, complete, file mtime
2026-07-28 17:33) — **not available when the tasking was written**:

| dataset | arm | Δacc | **p** | null mean ± sd | null max |
|---|---|---|---|---|---|
| MHC-EN | VSW_pow | **+0.0018** | **0.1194** | −0.00357 ± 0.00696 | +0.0164 |
| MHC-EN | VSW_exp | −0.0036 | 0.5423 | −0.00369 ± 0.00636 | +0.0146 |
| MHC-EN | VSW_lin | +0.0073 | 0.1144 | −0.00339 ± 0.00703 | +0.0182 |

**VSW is 1-for-3, not 1-of-2-with-one-pending.** The `§8` placeholder in
`VSW_PREGATE_RECORD.md:896` can be filled with the three tables above.

Note for the record: on MHC-EN the **cosine-only control beats the treatment** — `CTRL_cos_pow`
+0.0200 (ER 3.2000) vs `VSW_pow` +0.0018 (`vsw_main_en_OUT.json:result.arms`, and
`VSW_PREGATE_RECORD.md:833-837`). DEG-D does not fire only because it needs 2 of 3.

### 1.3 PREMISE CORRECTION 1 — the "ZH's analogue is CLOSER" inversion is a cross-arena artefact

The tasking contrasts ERRPAT's HateMM analogue rank ~3.0 with ZH's ~1.5 and calls it counter-intuitive.
Those two numbers are **not measured in the same space or on the same population**:
`ERRPAT_HateMM_2026-07-26.md:134` reports **3.0** for *stable-core TEST errors in the deployed HEAD
space*; `MECHFIX_PREGATE_2026-07-27.md:36-37` reports ZH's **1.5** for *22 core TEST errors in the raw
fused space*.

**In the single arena VSW actually operates in** (raw fused, train, item-disjoint 5-fold), newly
computed over the full-bank neighbour ordering:

| dataset | median analogue rank, all items | **median over deployed-WRONG** | frac of errors with analogue ≤ 5 | frac with analogue in top-20 |
|---|---|---|---|---|
| HateMM | 1.0 | **2.0** | 0.7586 | 0.9569 |
| MHC-ZH | 1.0 | **2.0** | 0.8977 | 1.0000 |
| MHC-EN | 1.0 | **2.0** | 0.9008 | 0.9917 |

All three are **2.0**, inside F95's own "2.0–3.0 over the deployed vote's errors"
(`MECHNOV_PAIRVERIFY_PREGATE.md:372-374`). **There is no inversion to explain**, and analogue rank
does not discriminate the datasets. It is not the mechanism.

### 1.4 PREMISE CORRECTION 2 — the cone-collapse figures are HEAD-space and do not describe VSW's arena

`MECHFIX_PREGATE_2026-07-27.md:38-39` (top-1 cos 0.999852 error vs 0.999976 correct) and `:238`
(train top-1 sim 0.9999) are **deployed HEAD-space** quantities. VSW runs in the **raw fused** space.
Newly computed there:

| dataset | median top-1 cos | median top-20 cos | median (c₁ − c₂₀) | median cos IQR inside a top-20 | top-1 cos ERR / COR |
|---|---|---|---|---|---|
| HateMM | 0.944406 | 0.915703 | 0.025322 | 0.008534 | 0.942857 / 0.944813 |
| MHC-ZH | 0.952434 | 0.928809 | 0.022131 | 0.007275 | 0.948913 / 0.953104 |
| MHC-EN | 0.940715 | 0.915028 | 0.023101 | 0.007244 | 0.941846 / 0.940511 |

The raw fused cone has ~2.4 % relative dynamic range — **~250× more than the head space's 1e-4** —
and the error/correct top-1 separation is ~0.002–0.004, not 1e-4. The tasking's cone premise is
inapplicable as stated. §5 measures what actually matters instead (the magnitude channel is inert
*anyway*, for a different reason).

---

## 2. PARITY — asserted before any new number was read

Two independent gates, both **PASS**, using `scripts/analysis/mechfix_ops.py` **unmodified**
(sha256 re-verified this session = `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d`)
under the frozen VSW protocol (`StratifiedKFold(5, shuffle=True, random_state=0)`, item-disjoint,
fused = `l2n(concat(l2n(img), l2n(txt)))`, 7168-d).

| gate | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| pooled deployed train-LOO acc, recomputed | **0.8441** | **0.8480** | **0.7796** |
| banked anchor (F95 `MECHNOV_PAIRVERIFY_PREGATE.md:298-300`; F98 finding body) | 0.8441 | **0.8480** | 0.7796 |
| **anchor gate** | PASS | **PASS** | PASS |
| bit-exact vs the VSW arena `vsw_ckpt/<ds>/f{0..4}.npz` on `dep_p`, `dep_v`, `nbr_cos`, `nbr_lab` | **5/5** | **5/5** | **5/5** |

**Second gate — the operator itself.** Replaying VSW from the banked arena plus the per-fold λ* in
`vsw_main_*_OUT.json` reproduces the banked arms exactly:

| dataset | arm | Δacc (mine / banked) | fixed | broken | changed | ER |
|---|---|---|---|---|---|---|
| HateMM | VSW_pow | +0.0255 / +0.0255 | 36 | 17 | 53 | 2.1176 |
| MHC-ZH | VSW_pow | −0.0017 / −0.0017 | 8 | 9 | 17 | 0.8889 |
| MHC-EN | VSW_pow | +0.0018 / +0.0018 | 22 | 21 | 43 | 1.0476 |

`exp` and `lin` also reproduce exactly on all three. **Every number below §2 is newly computed this
session** from `scratchpad/vsw_asym_recon.py` → `arena_{ds}.npz`, `vsw_diag{,2,3,4}.py` →
`vsw_diag{,2,3,4}_OUT.json`, with the banked verifier scores `vsw_ckpt/<ds>/f*.npz:nbr_p` read
**read-only**.

*One disclosed 1-item discrepancy:* my any-reweighting family oracle reads **+0.1478 / +0.1520 /
+0.2186**; F98's finding body records **+0.1492 / +0.1520 / +0.2186**. ZH and EN match exactly; HateMM
differs by 1 item of 744 (0.00134), attributable to the single zero-norm HateMM key that
`vsw_main_hatemm_OUT.json:cos_diagnostic` also flags (`n_zero_norm_keys: 1`, `n_items_affected: 1`).

---

## 3. THE MECHANISM — fix yield is constant, break exposure is not

### 3.1 The decisive quantity: **flip cost**

For each held-out item define `v_j = (2·lab_j − 1)·cos_j` over its deployed top-20 and `p = w/Σw` with
`w = [20..1]`. The deployed score is the convex combination `s = Σ p_j v_j`. **Flip cost** = the minimum
probability mass that must be *moved* between the 20 weights to drive `s` across 0 (closed-form optimal
transport: drain the extreme wrong-side entries into the single most extreme right-side entry). It is
the exact, verifier-free, training-free price of a decision under *any* non-negative re-weighting — the
budget VSW is spending.

### 3.2 Joint (cost × verifier-direction) supply and exposure — the table that explains everything

`helps` = the item's push under λ → ∞ (all mass on the verifier-argmax neighbour) points toward gold.

| θ (mass budget) | | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|---|
| 0.10 | fix supply (wrong ∧ cheap ∧ helps) | **29** | 20 | 32 |
| 0.10 | break exposure (correct ∧ cheap ∧ hurts) | **8** | 22 | 29 |
| 0.10 | **ratio** | **3.625** | **0.909** | **1.103** |
| 0.20 | fix supply | 40 | 26 | 42 |
| 0.20 | break exposure | **21** | 36 | 41 |
| 0.20 | **ratio** | **1.905** | **0.722** | **1.024** |
| — | **realised VSW_pow ER** | **2.118** | **0.889** | **1.048** |

The joint statistic — computable with **zero fitting** from geometry plus banked scores — reproduces the
realised exchange-rate ordering and, at θ = 0.20, its magnitude to within 0.21 on all three datasets.

### 3.3 Normalising the two halves isolates the cause

| θ = 0.10 | HateMM | MHC-ZH | MHC-EN | spread |
|---|---|---|---|---|
| **fix yield** = supply / n_ERR | 29/116 = **0.2500** | 20/88 = **0.2273** | 32/121 = **0.2645** | **1.16×** |
| **break exposure** = exposure / n_COR | 8/628 = **0.0127** | 22/491 = **0.0448** | 29/428 = **0.0678** | **5.33×** |

| θ = 0.20 | HateMM | MHC-ZH | MHC-EN | spread |
|---|---|---|---|---|
| fix yield | 0.3448 | 0.2955 | 0.3471 | 1.17× |
| break exposure | **0.0334** | 0.0733 | 0.0958 | **2.87×** |

**This is the whole finding.** The verifier reaches the same fraction of errors on every dataset. What
differs is how much *previously-correct* territory sits inside the same blast radius. HateMM's does not;
ZH's and EN's does.

### 3.4 Where the break exposure comes from: the deployed vote's confidence profile

| quantity (newly computed) | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| median top-20 purity, deployed-CORRECT | **0.85** | 0.75 | 0.70 |
| median top-20 purity, deployed-WRONG | 0.325 | 0.40 | 0.40 |
| **purity gap** | **0.525** | 0.350 | 0.300 |
| median \|s\|, CORRECT | 0.693096 | 0.547154 | 0.434778 |
| median flip cost, CORRECT | **0.3422** | 0.2521 | 0.2180 |
| median flip cost, WRONG | 0.1583 | 0.1021 | 0.0921 |
| frac ERR whose λ→∞ push helps | **0.4310** | 0.3409 | 0.3636 |
| frac COR whose λ→∞ push hurts | 0.1545 | 0.1568 | 0.1425 |

Two readings. (i) `frac COR hurts` is **essentially constant** (0.143–0.157) — the verifier is
equally often wrong-headed on correct items everywhere; what saves HateMM is that those items are
**twice as expensive to break**. (ii) HateMM's error population is the *most* inverted (purity 0.325,
the lowest of the three) — so HateMM does **not** convert because its errors are easier. Its
error/correct populations are simply the most **separated**.

### 3.5 Verifier quality inside the VSW arena — and a third premise correction

The tasking's "ZH has the LARGEST verifier advantage yet converts least" comes from F95's within-query
AUC (`MECHNOV_PAIRVERIFY_PREGATE.md:412-416`: +0.1572 / +0.2302 / +0.1785), which
`scripts/analysis/mechnov_pairverify_diag.py:102-116` computes over each query's row against the
**entire fitting-fold bank** (~440–595 candidates). Measured instead **inside the deployed top-20 —
the exact 20 items VSW re-weights**, which is a far harder pool because they are the 20 nearest
(newly computed; AUC of `nbr_p` for separating neighbours sharing the query's gold class):

| dataset | verifier | cosine | **Δ** | on deployed-WRONG items | frac ERR with AUC > 0.5 | on deployed-CORRECT | median Spearman(verifier, cos) |
|---|---|---|---|---|---|---|---|
| HateMM | **0.7745** | 0.5361 | **+0.2384** | 0.4596 | **0.4775** | 0.8398 | 0.0767 |
| MHC-ZH | 0.7263 | 0.5666 | +0.1597 | 0.4071 | 0.3068 | 0.7980 | 0.0917 |
| MHC-EN | 0.7063 | 0.5492 | +0.1571 | 0.4062 | 0.3667 | 0.7935 | 0.0962 |

**The ordering is not inverted in the arena that matters — HateMM is first (+0.2384), and ZH's
whole-bank advantage of +0.2302 shrinks to +0.1597 once restricted to the 20 items the operator can
act on.** The "law-I-internal-to-the-relational-asset" reading in the tasking is not supported once
the pool is matched to the operator. What *is* a law-I instance, and a sharp one:
**on deployed-wrong items the verifier is below chance on all three datasets (0.4596 / 0.4071 /
0.4062)** while being strongly informative on correct ones (0.79–0.84). The verifier is exactly
un-informative where the decision needs it.

### 3.6 Candidates the tasking listed that are measured NOT to be the mechanism

* **Bank size / density — NO.** Resampling HateMM's fitting bank down to ZH's (463) and EN's (439)
  sizes, 3 seeded repeats, geometry side only:

  | fitting-bank size | HateMM purity_COR / frac_COR cheap(θ≤0.10) | MHC-ZH | MHC-EN |
  |---|---|---|---|
  | 439 | **0.85 / 0.0773** | — | 0.70 / 0.2118 |
  | 463 | **0.85 / 0.0759** | 0.75 / **0.1283** | — |
  | 595 (HateMM native) | 0.85 / 0.0653 | — | — |

  At an **exactly matched bank size** HateMM keeps purity 0.85 and 0.0759 exposure against ZH's 0.75 and
  0.1283. The density effect is real but small (0.0653 → 0.0773 over a 26 % bank reduction) and closes
  **none** of the 1.7×/2.8× gap. `n` is not the mechanism.
* **Class balance / base rate — NO.** Bank rates 0.4005 / 0.3109 / 0.3060; VSW pos-rate deviations
  0.0363 / 0.0121 / 0.0583, all inside the 0.10 tolerance (`vsw_main_*_OUT.json:result.class_balance`).
  ZH has the *smallest* deviation and the worst result; EN the largest and a null. No monotone relation.
* **Degeneracy — CONFIRMED, but it is a consequence, not a cause.** ZH fires DEG-A at **0.9516**
  (threshold twin) and DEG-B at **0.9706** with arg-max **k = 20 = the deployed rule itself**;
  agreement with deployed is the identical 0.9706 (`vsw_main_zh_OUT.json:result.degeneracy`;
  `VSW_PREGATE_RECORD.md:785-796`). ZH's VSW changes **17 items of 579**. But this is *downstream*:
  because the joint ratio is < 1 at every radius (§3.2), the inner CV correctly selects
  λ* ∈ {0.25, 0.25, 0.25, **0**, 0.5} — it *falls back to the deployed rule* because re-weighting genuinely
  does not pay on ZH. The DEG firing is the selector behaving correctly, not an artefact. (F96/RESTRANS's
  95–99 % agreement was a *different* failure — an operator that was a threshold shift in costume; ZH's
  VSW is the deployed rule in costume, which is the F98-C3-on-EN form.)
* **Stream composition — YES, this is the upstream cause, and it re-confirms F44/F86.** Newly computed
  per-stream deployed vote and purity in the same arena:

  | dataset | fused acc | **text-only acc** | img-only acc | purity: fused / text / img |
  |---|---|---|---|---|
  | HateMM | 0.8441 | **0.8441** (Δ +0.0000) | 0.7688 | 0.80 / 0.85 / **0.75** |
  | MHC-ZH | 0.8480 | **0.8636** (Δ **+0.0156**) | 0.7012 | 0.70 / 0.85 / **0.60** |
  | MHC-EN | 0.7796 | **0.8106** (Δ **+0.0310**) | 0.6995 | 0.65 / 0.70 / **0.60** |

  ZH's *text* stream is exactly as pure as HateMM's (0.85 both). The difference is that HateMM's image
  stream is usable (0.75 purity, 0.7688 acc) while ZH's and EN's are not (0.60 / 0.7012, 0.60 / 0.6995),
  and the equal-weight L2 concat drags the fused key down — **so ZH and EN pay a fusion tax that puts a
  large minority of their correct items near the vote boundary**, which is precisely the break exposure
  of §3.3. This is F44's mechanism ("RGCL's equal-weight L2 concat cancels the text gain") and F86's
  `U1 image uniqueness = 0.0000 on 5/6 cells`, independently re-derived in the VSW arena. F58's
  "HateMM's convertible signal is text-carried" is consistent: HateMM's text is the strongest *and* its
  image is the least damaging.

---

## 4. RULING (a) — does the mechanism name a second dataset? **NO. HateMM is structurally the only one.**

Three independent numeric arguments, in increasing strength:

1. **Hindsight ceiling (decisive, needs no model).** Best fixed λ chosen on the evaluation data itself,
   over 3 families × 48 λ (`VSW_PREGATE_RECORD.md:596-600`, re-verified from `result.curve`):
   **HateMM +0.0309 · MHC-ZH +0.0069 (23 % of bar) · MHC-EN +0.0164 (55 %)**. No selector can beat a
   hindsight ceiling. ZH and EN are out of reach as a property of the operator space.
2. **Counterfactual transplant.** Give ZH HateMM's break-exposure rate (θ = 0.20: 0.0334 instead of
   0.0733) *and* HateMM's fix yield (0.3448 instead of 0.2955), keeping ZH's own n: fixes =
   0.3448 × 88 = 30.3, breaks = 0.0334 × 491 = 16.4, **net = +13.9 items = +0.0240** — still under bar.
   ZH's binding constraint is that it has only **88 errors in 579 items**: a +0.030 net requires ≥ 18
   net items, i.e. fix-count ≥ 30 at **≤ 12 breaks**, against VSW's measured best-hindsight ZH cells of
   **11 fixed / 7 broken** (`lin` λ = 0.4, +0.0069) and **13 fixed / 10 broken** (`pow` λ = 0.25,
   +0.0052). It needs 2.3–2.7× the fixes *and* no more breaks.
3. **Bank growth cannot buy it.** ZH's exposure would have to fall from 0.1283 to ≈0.0759. Its own
   2-point slope (0.1315 @ 439 → 0.1283 @ 463) extrapolates to ~394 extra bank items (train ≈ 1 070,
   +85 %); HateMM's shallower and more trustworthy slope (0.0773 @ 439 → 0.0653 @ 595) extrapolates to
   ~681 (train ≈ 1 575, **2.7×** the ZH dataset). Both extrapolations are fragile and both are moot:
   the user's standing ruling forbids cross-dataset mixing, MultiHateClip-ZH has no more train split,
   and §4.2 shows exposure parity alone still lands at +0.0240.

**This closes the direction.** VSW-like re-weighting cannot reach +0.030 on a second dataset, and the
reason is arithmetic, not effort.

---

## 5. RULING (b) — a non-tuning modification to lift HateMM over +0.030? **NO.**

The bar on HateMM is `0.030 × 744 = 22.3` → **net ≥ 23 items**.

| object | net (items) | Δacc | is it a mechanism change? |
|---|---|---|---|
| whole `pow` continuum, 24 λ | max **+21** (λ = 3) | +0.0282 | — |
| whole `exp` continuum | max **+23** (λ = 4) | **+0.0309** | **no — the selector already picks λ = 4 in 4/5 folds** |
| whole `lin` continuum | max +17 (λ = 0.9) | +0.0228 | — |
| deployable nested-CV `exp` | +19 | +0.0255 | (the actual result) |
| per-**fold** oracle λ (`pow`) | +26 | +0.0349 | **selection** |
| per-**item** oracle λ over the 24-point grid (newly computed) | +51 | **+0.0685** | **selection** |
| any non-negative re-weighting (family oracle) | +110 | +0.1478 | **selection** |

The `exp` λ = 4 cell is the tell. It clears the bar by exactly **one item**, and the deployable selector
*already chooses λ = 4 in four of five folds* — the +0.0054 shortfall comes from fold 4 choosing λ = 8
(`vsw_main_hatemm_OUT.json:result.arms.VSW_exp.lambda_per_fold`). Pinning λ = 4 globally is fitting one
scalar on the evaluation data. That is the definition of the tactic the house bans.

Everything else with net ≥ 23 is **per-item or per-fold conditioning**, and three independent findings
price it: F66's β-decomposition (selection-locked headroom), F97's **K-VGA-3** (F47 features *beat* the
verifier profile as gating inputs on 3/3 with significance), and F98's ceiling-vs-delivery law
(*"within this family delivery is uncorrelated with ceiling"* — ceilings +0.1492/+0.1520/+0.2186 →
delivered +0.0134/−0.0069/+0.0000). Note the per-item λ oracle computed here (+0.0685 / +0.0570 /
**+0.0893**) is the same order as VGA's adjudication-gate oracle (+0.0726 / +0.0535 / **+0.0893** —
identical on EN), i.e. it is the *same* per-item selection object wearing a different hat, and it is
already banked as dead.

**I could not name a single modification that is a mechanism change rather than a tuning move.**

---

## 6. RULING (c) — does the mechanism generalise into a new, non-VSW candidate? **NO.**

### 6.1 The tasking's proposed generalisation is falsified at the premise

*"If VSW's gain comes from restoring magnitude information to a collapsed cone…"* — it does not.
Newly computed: replace `cos_i` by the constant 1 in the deployed vote (keep retrieval, k = 20, the
labels, the rank weights and the threshold):

| dataset | deployed acc | **sign-only acc** | Δ | agreement with deployed |
|---|---|---|---|---|
| HateMM | 0.8441 | 0.8427 | **−0.0013** | **0.9960** |
| MHC-ZH | 0.8480 | 0.8480 | **+0.0000** | **0.9965** |
| MHC-EN | 0.7796 | 0.7778 | **−0.0018** | **0.9982** |

**The deployed vote is, to within 0–2 items of 549–744, a pure rank-weighted SIGN vote.** The cosine
magnitude carries essentially no decision information in the raw fused arena either — the head-space
cone collapse (F89) and the raw-space magnitude inertness are two faces of the same fact. There is
nothing to restore. Confirming this from the other side, VSW's multiplier is near-orthogonal to the
cosine inside the top-20 (median per-query Spearman **0.0767 / 0.0917 / 0.0962**), so it is **not**
sharpening the magnitude channel at all — it is substituting an orthogonal relational order for the
fixed rank profile.

**Consequence for F89-T2b:** the whitening trap (cone 0.9999 → 0.5220 but length-nuisance ρ 0.52 → 0.87,
negative on 3/3, `MECHFIX_PREGATE_2026-07-27.md:238,288-290,427`) is not even the binding objection any
more. A de-collapsing operator would be repairing a channel worth **≤ 2 items**. Any future
"fix the collapsed geometry" proposal now inherits *two* refutations: T2b's measured negative **and**
this measurement that the repaired channel is decision-inert.

### 6.2 The one direction the mechanism does point at is already closed

§3.6 shows the upstream cause is the **fusion tax**: ZH's and EN's fused keys are diluted by an image
stream at 0.60 purity, and dropping it would recover +0.0156 / +0.0310 in the raw arena. That is not a
new candidate:

* **F50** killed the fusion/composition axis with an explicit within-Qwen re-weight sweep whose arm A1
  is *exactly* this object: `z = [√w·îmg_Q, √(1−w)·t̂xt_Q]`, `w ∈ {0.00, 0.05, …, 1.00}`, so
  `cos(z_a,z_b) = w·img_cos + (1−w)·txt_cos` and **`w → 0` is Qwen-text-only retrieval**
  (`FA_GATE_RECORD.md:37-40`). Verdict: *"pure rotation at every w … no w converts."* **Disclosed
  limit:** F50's arms ran on **MHC-EN primary + HateMM control only** — ZH was not in that grid, so
  the ZH endpoint is banned by inference from F85/F86, not by direct F50 measurement.
* **F85** killed the fusion-concat family on **ZH and HateMM** (both cells KS-arm-dead: ZH val-sel
  +0.0067 but final −0.0045; HateMM −0.0031 both protocols).
* **F86/LSMI** measured `U1 image uniqueness = 0.0000` on 5/6 cells, i.e. there is no unique image
  information for a smarter fusion to preserve.
* **F44** already recorded the same mechanism ("RGCL's equal-weight L2 concat cancels the text gain",
  MHC-EN image AUC 0.734 → 0.599) as a **dataset property**, not a lever.
* And the arena caveat is fatal on its own: the deployed head fuses by **Hadamard on projected
  streams**, not by raw L2-concat, so this raw-arena number is a *diagnosis of the banked key space*,
  not a measurement of the deployed object.

**No new candidate is nominated.** The mechanism's honest output is a *paper sentence*, not a lever:
*the convertibility of a retrieval-memory vote under any re-weighting operator is set by the fragility
of its correct set, not by the reachability of its errors, and that fragility is inherited from the
fusion of an uninformative stream into the key.*

---

## 7. RULING (d) — must F98/AGGNET's `ban_scope` be amended? **YES, but additively and narrowly.**

### 7.1 First, the tasking's characterisation of the ban is not what F98 says

The tasking quotes the scope as *"ANY learned re-weighting … over the deployed top-20 … WITH ANY
FEATURE FAMILY."* That conflates two separate clauses of `AGGNET_PREGATE_RECORD.md:693-698`:

> * **(a)** any learned re-weighting, soft-mixture-over-k, attention, or gating **over the deployed top-20**
>   — C3 spans that class and DEG-B shows what it converges to;
> * **(b)** any per-item selector, router or adjudication gate over the same neighbourhood, **with any
>   feature family** — the verifier features are dead by K-VGA-3 …

**"With any feature family" attaches to (b) — selectors/routers/gates — not to (a).** And F98 states
its own limiter explicitly at `:711-713`: *"The closure covers operators whose **input is the (cosine,
label) profile of the deployed top-20**."* VSW's input is the **trained relation score**, which is
outside that profile by construction (F98 §1.3 forbade verifier features in C3's profile). **VSW was
never inside F98's ban as written**; the VSW record's own §1 framing — *"inside the functional form of
that closure but outside its information content"* (`VSW_PREGATE_RECORD.md:99-105`) — is the correct
reading. So the amendment is not a correction of an over-broad ban; it is the **completion of a
deliberately incomplete one**.

### 7.2 Proposed amendment text (for the orchestrator to place; not applied here)

> **F98 §7.1 clause (a), AMENDED 2026-07-28 by VSW (F99).** Closed: any learned or hand-designed
> re-weighting of the fixed rank profile `[20..1]` over the deployed top-20 — including
> soft-mixture-over-k, attention, gating, and **λ-interpolated monotone multipliers driven by the F95
> pair-verifier score**. F98 closed this class for inputs that are the **(cosine, label) profile**;
> VSW closes the one input family F98's restriction deliberately excluded, the **trained relation
> score**, and it does so *without* the DEG-A/DEG-B degeneracy that carried F98's verdict — on HateMM
> VSW is a genuinely distinct operator (DEG-A 0.9220, DEG-B 0.9328, beats its threshold twin by +0.0107
> and its cosine twin by +0.0188) and it **still tops out at +0.0255 deployable / +0.0309 with full
> hindsight, on one dataset of three, with 2 of 3 arithmetically unreachable (+0.0069 / +0.0164).**
>
> **Therefore the closure's ground shifts and this must be recorded, because it changes what future
> sweeps should test.** F98's ground was *degeneracy* ("it converges to a threshold shift / fixed k").
> VSW's ground is *the net*: `net = changed · (2·precision − 1)`, precision decays monotonically with
> sharpness on 3/3, and the product is pinned below the bar across a 16 000× λ range
> (`VSW_PREGATE_RECORD.md:748-775`). **A future re-weighting proposal cannot escape this closure by
> demonstrating non-degeneracy** — VSW already is non-degenerate on HateMM and still fails. It can only
> escape by exhibiting a fix-supply/break-exposure ratio > 1 **at a radius where enough items are
> changed to matter** (§3.2 of `VSW_ASYMMETRY_RECON.md`), which is a $0 pre-check on banked geometry
> that any such proposal must now pass before it is written.
>
> **Not closed** (unchanged from F98 `:711-720`): operators whose input is the retrieved **vectors**
> rather than their profile or their pairwise scores — **LITSWEEP6-MEMBANK C4** (aggregate-then-compare
> subspace residual) — and operators that change **membership** — LITSWEEP6-MEMBANK C2.

### 7.3 A routing hazard that will mis-route the next sweep if not fixed now

**There are two different candidates called "C4".** `LITSWEEP6_RELGEN.md:256` C4 = **VSW** (killed
here). `LITSWEEP6_MEMBANK.md:466` C4 = **aggregate-then-compare subspace residual** (untouched,
$0, and nominated as *the* next candidate by both `AGGNET_PREGATE_RECORD.md:728-731` and
`RESTRANS_PREGATE_RECORD.md` §7.2). The VSW record's title says "LITSWEEP-6 C4" and its §10.7 says
*"the sweep has no untasked arm left"* — true of **relgen**, false of **membank**. Any future reader
who reads "C4 is killed" will skip the one live $0 candidate the family has. **Recommend both records
be cited hereafter as `RELGEN-C4 (VSW)` and `MEMBANK-C4 (subspace residual)`.**

---

## 8. WHAT THIS RECON ADDS TO THE RECORD, IN ONE PLACE

1. **MHC-EN's permutation battery completed and is a null** (+0.0018 p = 0.1194 / −0.0036 p = 0.5423 /
   +0.0073 p = 0.1144); VSW is 1-for-3 (§1.2).
2. **The asymmetry is a break-exposure asymmetry, not a fix-supply one** — fix yield 1.16× across
   datasets, break exposure 5.33× (§3.3).
3. **A $0 pre-check that predicts the exchange rate before any operator is written** — the joint
   (flip-cost × verifier-direction) supply/exposure ratio reproduces realised ER to within 0.21 on 3/3
   (§3.2). This should be the first gate on any future re-weighting proposal.
4. **The cosine magnitude is decision-inert in the raw fused arena** (−0.0013/+0.0000/−0.0018,
   99.6–99.8 % agreement) — the raw-space companion to F89's head-space cone collapse, and a second,
   independent refutation of the "de-collapse the geometry" family (§6.1).
5. **The verifier-advantage ordering is NOT inverted** once measured in VSW's own arena: HateMM +0.2384
   > ZH +0.1597 > EN +0.1571 (§3.5). The law-I instance is sharper and different: the verifier is
   **below chance on deployed-wrong items on all three datasets** (0.4596/0.4071/0.4062) while being
   0.79–0.84 on correct ones.
6. **Analogue rank is 2.0 on all three** in this arena; the "ZH's analogue is closer" inversion was a
   head-space-vs-raw-space comparison artefact (§1.3).
7. **Bank size is not the mechanism** — the gap survives exact bank-size matching (§3.6).
8. **The upstream cause is the fusion tax**, independently re-deriving F44's mechanism and F86's
   `U1 = 0.0000` in the VSW arena (§3.6) — and it is closed by F50/F85/F86 (§6.2).

---

## 9. ONE-LINE SUMMARY

HateMM converts and MHC-ZH does not because the verifier reaches the same fraction of errors everywhere
(fix yield 0.250/0.227/0.264) while HateMM's correct set is 3.5–5.3× less fragile (break exposure
0.0127/0.0448/0.0678) — a property of the deployed vote's confidence profile inherited from HateMM
being the one dataset whose image stream does not poison the fused retrieval key; the reason names
**no** second dataset (ZH's full-hindsight ceiling +0.0069; even a transplant of HateMM's entire
structure gives ZH only +0.0240), **no** non-tuning lift for HateMM (the λ continuum's max net is
+21 of the required 23 items, and every object above it is per-item selection), and **no** new
candidate (the "collapsed cone" it was supposed to repair is worth ≤ 2 items) — so the direction is
**closed**, and F98's clause (a) should be amended **additively**, its ground moved from degeneracy to
the capped net, with a $0 supply/exposure pre-check installed as the gate for anything that tries again.

---

## 10. FILE MANIFEST

| artefact | role |
|---|---|
| `refine-logs/VSW_ASYMMETRY_RECON.md` | this record (the only file written under `refine-logs/`) |
| `<scratchpad>/vsw_asym_recon.py` → `arena_{hatemm,zh,en}.npz` | parity-gated arena rebuild via frozen `mechfix_ops.deployed_vote` |
| `<scratchpad>/vsw_diag.py` → `vsw_diag_OUT.json` | cone / analogue rank / purity / flip cost / within-20 verifier AUC / arm replay / oracles |
| `<scratchpad>/vsw_diag2.py` → `vsw_diag2_OUT.json` | joint (cost × direction) supply-and-exposure decomposition |
| `<scratchpad>/vsw_diag3.py` → `vsw_diag3_OUT.json` | bank-size control; cosine-magnitude channel pricing |
| `<scratchpad>/vsw_diag4.py` → `vsw_diag4_OUT.json` | per-stream purity / per-stream deployed vote |

**Read-only inputs:** `scripts/analysis/mechfix_ops.py` (sha256 asserted), `scripts/analysis/vsw_main_*_OUT.json`,
`scripts/analysis/vsw_perm_*_OUT.json`, `scripts/analysis/vsw_ckpt/{hatemm,zh,en}/f{0..4}.npz`,
`scripts/analysis/vsw_pregate.py` (read for the multiplier definitions only),
`data/CLIP_Embedding/{HateMM,MHC_zh,MHC}/train_*.pt`.
**Not committed** — the orchestrator sequences commits, since the vsw-pregate agent is writing adjacent paths.

---

## ⚠ RE-SCOPING (appended 2026-07-28, closeout) — this record's framing question presupposes a raw-arena artefact

**No verdict moves; no measurement in this record is withdrawn.** Every quantity here was computed in
the **raw** train-LOO arena and remains correct *of that arena*.

**The framing correction.** This record opens by asking *"why does HateMM **convert** under VSW
(+0.0255, p = 0.0050) and MHC-ZH not"*. `HEADSPACE_TRANSFER_PREGATE.md` (F113) re-ran that operator on
those items in the **deployed head space** (fold heads, 3 seeds × 5 folds): **+0.0009**, p = **0.0968**,
transfer ratio **0.035**, λ-oracle ceiling **+0.0072**, and **both** degeneracy controls firing (they
did not fire on HateMM in raw). **HateMM does not convert in the deployed space either.** The
asymmetry this record explains is an asymmetry **of the raw arena**, not of the datasets.

**What survives, and it is the useful part.** The *mechanistic* content — break exposure as the binding
constraint, the fold-level structure, the exchange-rate arithmetic — is a within-arena analysis and is
untouched. What must not be re-quoted is any sentence implying HateMM was close to converting: F113
measures a **33× miss** against K-VSW-1 in the arena that matters.

**Wherever `+0.0255` appears in this record it must carry:** *"raw train-LOO arena; head-space transfer
measured at +0.0009, p = 0.0968 (F113)."*

*Authority: `HEADSPACE_TRANSFER_PREGATE.md` §4.4, §4.8, §8.2 row 1 (F113). Ledger: F114. `$0`.*
