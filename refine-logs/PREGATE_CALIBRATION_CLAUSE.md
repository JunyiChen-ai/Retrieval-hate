# PREGATE CALIBRATION CLAUSE (CAL-0 … CAL-5)

**A standing, pre-registerable clause for every `$0` pregate run in the raw banked train-space arena.**

**Status:** proposed by `refine-logs/INSTRUMENT_VALIDATION_RECON.md` (2026-07-28), which is the evidence
base for every number here. Adopt by citing this file in a prereg's bars section.
**Cost of the whole clause: `$0`, minutes of CPU, no test contact.**

**Headline finding it encodes — read this before anything else.**
**The raw train-space arena is NOT ESTABLISHED as predictive of deployed effects.** The single largest
matched paired grid (21 points, 3 datasets, both arenas, same operator) gives pooled Spearman **+0.758**
— but **9 of those 21 points sit in a block where the operator is algebraically 1-NN**
(`KSWEEP_RECORD.md:29-31`) and both arenas agree for a reason that has nothing to do with either arena.
Remove that block and the pooled correlation is **−0.3039** (+0.40 / −0.95 / +0.95 per dataset). The two
matched stream pairs shrink 2.3× and **invert sign** respectively. On the training channel the arena is
valid on the curriculum cell and **anti-correlated across datasets** on encoder adaptation.

**This clause is therefore built to be SAFE under "unvalidated", not to assume an asymmetry.** An
earlier draft of it asserted a validated false-kill rate and set an anchor threshold of `rho_S ≥ 0.60`.
Both were wrong: the first over-read a one-sided sample, and the second would have been **passed by the
very degenerate block that inflated the headline**. They are recorded here as errata so the mistake is
not repeated.

**Companion clause:** `refine-logs/PREGATE_DETERMINISM_CLAUSE.md` (owned separately) governs whether the
arena reproduces *itself*. This clause governs whether it predicts *deployment*. **Keep them separate:**
reproducibility is settled (`PROVENANCE_AUDIT_2026-07-28.md:31` — *"Zero verdicts move"*);
predictiveness is open.

---

## The measured basis

| channel | matched pairs | what they show | source |
|---|---|---|---|
| (b) truncate the retrieved set | **21 points**, complete grid, pre-registered | +0.758 pooled **with** the degenerate block; **−0.3039 without it**. Bound: raw never missed a deployed effect > **0.67 test items**; nothing in either arena reached +0.010 | `INSTRUMENT_VALIDATION_RECON.md` §3.1, §3.1b |
| (b) threshold / calibration | 1 family, both directions banked | Raw *legal* **+0.0188/+0.0242** vs deployed **gold-cheating oracle +0.0124** | §3.2 |
| (a) stream composition, MHC-ZH acc | 1, construction verified | +0.0156 → **+0.0067** — same sign, **2.3× shrink**, 1 item | §3.5b(i) |
| (a) stream composition, HateMM AUC | 1, construction verified | +0.011 train-LOO → **−0.011** dev — **SIGN INVERSION**; 2 of 3 encoders invert | §3.5b(ii) |
| (d) curriculum | 1 cell | Same sign, same ordering, **~2× attenuating** | §3.7c, `PROVENANCE_AUDIT_2026-07-28.md:198-212` |
| (d) encoder adaptation | 3 datasets | Arena ordering EN > ZH > HateMM; conversion ordering **exactly reverse** | §3.7c, `PROVENANCE_AUDIT_2026-07-28.md:219-228` |
| — no-head probe (different cheap arena) | P3 ×3, P8, S2S | Positives failed. **One-sided sample — no information about negatives** | `research-wiki/EXP_p8_semantic_compression.md:132-135` |

**RETRACTED and not to be cited:** the MHC-EN stream "sign inversion" (mismatched baselines — raw arm
vs trained pipeline, §3.6) and the F91/Molmo2 numeric pair (`molmo2_geom_diag.py:71-78` omits
per-stream L2, §3.4).

---

## CAL-0 — The standing statement (mandatory, one sentence)

> *"The raw train-space arena is **not established** as predictive of deployed effects. It is used
> because it is the only `$0` arena available. Results are reported as **arena results**, never as
> predictions of deployed behaviour."*

## CAL-1 — Asymmetric reading (**PRUDENTIAL, explicitly NOT validated**)

> Read a raw-arena **null/negative** as a kill **only when the decisive bar is a within-arena relative
> comparison** — a degeneracy twin, an isomorphism control, a best-fixed-profile control, a
> label-shuffled null — rather than an absolute Δ. That is the property that makes a kill robust to the
> arena question, and it is what actually protects F96's and F98's verdicts.
>
> Read a raw-arena **positive** as a *ticket to a deployed measurement*, **never** as an effect size,
> a "% of the bar", or a ranking key.
>
> **Why this is prudential and not proven.** The evidence that positives fail is real but **one-sided by
> construction**: every instance exists because someone paid to take a raw *positive* to a deployed
> measurement, and nobody has ever taken a raw *negative* to one. There is also a simpler explanation
> that is not a property of the arena at all — **selection / winner's curse**. The campaign has measured
> that directly: one global hyperparameter costs **86-100 % of the found effect on 2 of 3 datasets**
> (`VSW_PREGATE_RECORD.md:609-622`), and F108's stream weight falls from a 2-of-3 conjunct under full
> hindsight to **1-of-3 when made deployable**. **Price your selection; do not blame the arena.**

## CAL-2 — The anchor arm (mandatory, `$0`, ~1 minute) — **provenance check, NOT a validity gate**

> Report the closed-form **`FIXK_k`** grid (`k ∈ {1,2,3,5,7,10,15,20}`, profile `[k..1, 0…]` over the
> deployed top-20) on your own folds, and state:
> 1. **`FIXK_20` must change 0 items and give `d_acc = 0.0000`** — the arena's k=20 rule *is* the
>    deployed rule. Miss ⇒ **harness VOID**, no treatment number is reported. *(This half is a hard
>    gate and it is sound.)*
> 2. the Spearman against F94's banked deployed k-curve (`scripts/analysis/ksweep_OUT.json`; primary
>    arms HateMM final / MHC-ZH final / MHC-EN ARM-V) **over k ∈ {5,7,10,15} ONLY.**
>    **k ≤ 3 must be excluded** — it is algebraically a plain 1-NN classifier, *"verified identical to
>    the top-1 label vector in 19/19 cells"* (`KSWEEP_RECORD.md:29-31`), so both arenas agree there for
>    a reason that carries no information. Reference value: **pooled −0.3039**, per-dataset
>    **+0.40 / −0.95 / +0.95**.
>
> **There is deliberately no threshold on (2).** It is reported so the fold draw and the caches are
> auditable and so drift in the arena instance is visible — **not** as a validity gate, because the
> arena has not been validated and a gate would imply otherwise.

## CAL-3 — The positive-side gate (mandatory whenever a raw Δ ≥ +0.010 is reported)

> Report the raw Δ **together with the deployed space's own gold-cheating ceiling for the same operator
> family**, wherever one is banked (the ERRPAT reports carry test-fitted oracles for the threshold,
> curation, length-de-bias and stream families).
>
> **If the raw *legal* number exceeds the deployed *oracle*, label the arm `RAW-ARENA ARTEFACT` and do
> not escalate** — regardless of fold signs, exchange rate, or permutation p.
>
> *Worked example:* raw legal **+0.0188** (fused) / **+0.0242** (text) against deployed oracle
> **+0.0078** (val-sel) / **+0.0124** (final), `ERRPAT_HateMM_2026-07-26.md:163-164`. CAL-3 would have
> caught both D1 and `THRESH_best` on the day they were measured.

## CAL-4 — Closed-form vs trained declaration

> Label every reported quantity **closed-form** or **trained**.
> * **Closed-form** reproduces bit-exactly across sessions.
> * **Trained** carries the F105 session-dependence caveat (*44 of 48* trained quantities drift;
>   cause is oneDNN/MKL GEMM dispatch compounding through Adam, every random source pinned —
>   `PROVENANCE_AUDIT_2026-07-28.md:396-411`). **Counts over trained quantities are never quoted across
>   sessions.**

## CAL-5 — Channel declaration (**the clause with real predictive content**)

> State which channel the operator acts in, and carry the corresponding warrant:
> * **Channel (b) — re-weight / re-order / truncate the retrieved set.** The two arenas share the entire
>   decision path below retrieval: the same ordered label tuple, the same rank weights, the same
>   threshold. A channel-(b) arena result has a *structural* reason to be about the same object the
>   deployed system computes. **It is still not validated, but it is the only channel with a mechanism
>   for transfer.**
> * **Channels (a)/(d) — change the representation or the map.** The two arenas **do not even share a
>   fusion operator**: raw is `L2norm(concat(L2norm(img), L2norm(txt)))`
>   (`MECHNOV_PAIRVERIFY_PREGATE.md:150`); deployed is **Hadamard on two learned projections**
>   (`src/model/classifier.py:87,140-141`; rendered at `MECHFIX_PREGATE_2026-07-27.md:27`). Elementwise
>   product versus concatenation. **A channel-(a)/(d) arena result carries NO transfer warrant and must
>   say so in its limitations.** The measured encoder-axis inversion is exactly this, and it is
>   predictable in advance rather than mysterious.

---

## What this clause does NOT license

* It does **not** certify the arena in any channel.
* It does **not** license reading an arena null as a deployed null when the decisive bar is an absolute
  Δ rather than a within-arena relative comparison.
* It does **not** replace a deployed-arena verdict. A pregate remains a pregate.

## Cost of actually settling the question

The `$0` route is exhausted: every matched pair that banked artifacts can supply has now been
assembled, and **the informative half of all of them sits inside the deployed arena's noise floor**
(1 test item = 0.0047-0.0067, ±0.014 seed band). The cheapest read that could adjudicate is a
**CPU head re-mint with same-path CPU-trained floors, dev only** — ~52 s/seed/dataset
(`ERRPAT_HateMM_2026-07-26.md:526-529`), ≈15 CPU-minutes for 3 seeds × 3 datasets, **plus a fidelity
gate per dataset**, because the deployed head inventory is gone: 228 `.pt` files in `logging/Retrieval`
all belonging to the F92-dead bidir heads, 97 empty `ckpt/` dirs, and **0 of 9** P2-era deployed
checkpoints surviving (F107). Dev resolution is 0.0093 / 0.0128 / 0.0125 per item, so it can adjudicate
the **+0.030 bar** but not the **+0.010** interest line, and **MHC-ZH dev is anti-correlated with test**
(−0.2402, p = 0.0380, `ERRPAT_MHC-ZH_2026-07-26.md:91-98`).

## Two ban-scope constants this clause retires

Documented in `INSTRUMENT_VALIDATION_RECON.md` §6(d):

1. **`F66 caps it at +0.001-0.006` must not be applied to trained-space reshaping.** F66's arithmetic is
   conditional on a single fixed map φ₀ (`NCA_FORENSIC_RECON.md:104-110`). The contrary reading at
   `LITSWEEP5_COMPLETENESS.md:13,84` is superseded.
2. **`alignment > 0.663` is not a campaign constant.** MHC-EN-dev arithmetic (N = 80, D = 21,
   p_Q = 0.588) against a **+0.020** bar (`MJ_FORENSIC_RECON.md:36-63`), with no HateMM or MHC-ZH
   re-derivation anywhere. Already ruled an *"INDUCTIVE LEAP"* at `REDTEAM_BAN_SCOPE_AUDIT.md:190`,
   with a `$0` remedy at `:204-208`, ranked #2 of 7 at `:365`, and never landed. **Route new router
   inputs through F47's own `$0` banked gate.**
