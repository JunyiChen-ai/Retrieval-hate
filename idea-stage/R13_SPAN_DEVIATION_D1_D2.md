# R13-SPAN — pre-result amendments D1 and D2

**Both issued 2026-08-18 before any arm metric was computed or seen.** Both were triggered by the
round's single external review round (gpt-5.6-sol, xhigh, hostile), which is the review round the
2026-08-05 ceremony ruling allows a CPU-level experiment. Neither amendment loosens a threshold,
neither was informed by a result, and both *narrow* what the probe is allowed to claim.

---

## D1 — the original control is geometrically underpowered; add a matched-length sweep

**Defect.** The frozen primary was Δ₁ = P1 − P2, gold span vs a random interval of *the same
length*. Two intervals of length fraction `c` inside one video must overlap by at least `2c − 1`.
HateMM's gold coverage median is **0.806** → the "random" control is forced to contain at least
**0.612** of the video, and therefore usually contains the evidence it is supposed to exclude.
MHC-EN's 0.937 forces ≥ 0.874. MHC-ZH's 1.000 makes P1 and P2 **identical by construction**.
A null on Δ₁ would therefore have been uninterpretable.

**Amendment.** Added a matched-length coverage sweep at r ∈ {0.10, 0.20, 0.40}: `G_r` keeps a
length-`r·D` interval centred on the longest gold span, `R_r` keeps a random length-`r·D` interval,
negatives byte-identical between the two. At r = 0.10 the forced overlap is **zero**.
Δ_sweep(r) = G_r − R_r becomes the load-bearing quantity; the original Δ₁ is demoted to descriptive.

**Also added: a positive control.** `ORACLE_r` picks, per positive video, the interval that
maximises that video's own score under a P0-trained model. This is deliberately leaky and can never
support a claim; it exists so that a null is interpretable. If even the leaky selector shows no
lift, the measurement lacks power and the probe decides nothing.

---

## D2 — the probe as frozen tests only the standalone-key question

**Defect.** Δ₁ and Δ_sweep ask whether a gold-located crop is a better *standalone* key. That is
the ceiling for **predicted-boundary trimming (A3)** and nothing else. A crop can be useless as a
key and still be useful as training augmentation, or carry soft information a full-video model
lacks. The original freeze's five-way kill rule (A1, A2, A3, A5, B1 all die on a null) was
therefore **invalid**, and is withdrawn.

**Amendment — arm set C, augmentation under unchanged inference.** Train on
{P0 keys} ∪ {G_r keys with the parent label}, **evaluate on P0 keys only**; compare against the same
thing with `R_r` keys and against P0-only. Crops are fold-locked to their parent video.
Δ_aug(r) = AUG_gold(r) − AUG_rand(r). This is the test of candidate A1 at its actual inference
distribution.

**Amendment — arm set D, privileged-information falsification without writing distillation code.**
Cross-fitted two-feature read-outs `M_gold = f(z_P0, z_G_0.2)` vs `M_rand = f(z_P0, z_R_0.2)`,
primary held-out log loss, also restricted to the items P0 alone gets wrong. This hands a
classifier the gold-located view *directly* — strictly more access than any student distilled from
it — so a null here falsifies span-privileged distillation (the one new legal family the review
named) before any distillation code exists.

---

## Revised decision rule (frozen here, still before results)

**Smallest worthwhile gain: δ = +0.015 OOF ROC-AUC**, one-sided. Justification: under an
equal-variance binormal model at AUC 0.70–0.80 this is ≈ +0.010–0.013 balanced accuracy, which is
the project's standing two-dataset bar. With 298 positives / 446 negatives and paired predictions
correlating ≈0.90, the paired SE is ≈0.008–0.009, so a one-sided margin near 0.014 — δ = 0.015 is
about the tightest bound this arena can certify. **If an upper bound lands above 0.015 the result
is INCONCLUSIVE, not a kill.**

| quantity | what a one-sided upper bound < 0.015 closes | what it does **not** close |
|---|---|---|
| Δ_sweep(r), all r | predicted-boundary trimming (A3); the claim that annotated span *location* improves a pooled crop key; cross-dataset boundary transfer's downstream (B1) | anything trained with crops |
| Δ_aug(r) | gold-span crop augmentation (A1) | HateClipSeg |
| M_gold vs M_rand | span-privileged distillation / LUPI | HateClipSeg |
| ORACLE_r shows no lift | **nothing — the probe is void and must be redesigned** | — |

A kill requires the corresponding upper bound to be below δ **and** the ORACLE_r positive control
to have shown a clear lift. Nothing here can close HateClipSeg (B2/B3), which is a separate
substrate pending a user ruling.
