# B3 ZH LoRA DECOMPOSITION — where the +0.0313 lives, is it Pareto or rotation, why val-selection kills it

**Author:** diagnostic analyst (ZERO GPU; banked feature caches + banked trainlogs only; **no
re-extraction** — LoRA-ZH caches already on disk; **no new test-touch** — test numbers are
read from already-logged per-epoch `Test_Retrieval` lines, exactly as `B3_VERDICT_REVIEW.md`
was written). **Date:** 2026-07-17. Does **not** touch `research-wiki/` (paper-integrator live).
**Feeds:** the pending user **D7 LoRA-family novelty ruling** with mechanism evidence.

**Cell under decomposition (B3, `exp-lora-zh-b3.md`, job 13150 vs CLIP 13115):** LoRA-Qwen
encoder vs frozen-CLIP on MHC-ZH, 3-seed paired. Verdict was **final-epoch PASS (MARGINAL)
+0.0313 acc / +0.0453 mF1 (3/3); val-selected FAIL +0.0246 acc**. Memory note: "gain = LoRA
adaptation not encoder identity" (frozen-Qwen ZH = −0.0112, B1). This doc says *why*, with the
same per-modality / Pareto-vs-rotation machinery used in `ENCODER_SWAP_DIAGNOSIS.md` (F44).

**Scripts (committed, reproduce every number):**
`scripts/analysis/encoder_swap_geometry.py` (kNN geometry, reused),
`scripts/analysis/b3_zh_lora_trainlog_parse.py` (→ `..._out.json`; banked-trainlog Test/Val curves).

---

## TOP-LINE (one paragraph)

The LoRA +0.0313 gain lives **entirely in the TEXT stream** — LoRA adaptation lifts the
transcript/language representation (train-LOO text AUC CLIP 0.802 → frozen-Qwen 0.847 →
**LoRA 0.925**) while leaving the **image stream untouched** (0.718 → 0.721 → 0.714). It does
**not** fix an image collapse (MHC-ZH never had one, unlike MHC-EN) and does **not** rebalance
fusion; it sharpens the one stream where ZH hate actually lives. Crucially, this text-stream
sharpening **converts as a Pareto minority-recall gain** — on held-out **test**, LoRA lifts
hate recall **+0.111 at essentially zero non-hate cost (−0.003)**, the *same signature as the
HateMM encoder-swap win* (+0.116 at zero cost). This is mechanistically distinct from
**frozen-Qwen ZH, which is a rotation** (hate +0.074 bought with non-hate −0.048, net −0.011)
— the exact unconvertible easy-example edge B5 proved. **LoRA turns the frozen rotation into a
convertible Pareto gain by moving the decision boundary, not merely re-ranking.** The
val-selection FAIL is a **78-sample-dev selection-noise artifact, not LoRA instability**: LoRA
is the *most* stable of the three arms (lowest val-sel regret, tightest per-seed test spread);
its dev-acc saturates at a 0.8718 plateau by ~epoch 19 while test keeps climbing to epoch 29,
so val-selection picks a plateaued mid-epoch that undershoots. **Verdict for D7: the ZH LoRA
positive is a genuine, interpretable representation-limited conversion — not a selection
artifact or a ranking mirage — but it is single-dataset (ZH), marginal (+0.0313), single-draw,
and protocol-fragile.** It opens no new axis (still Axis B, encoder-adaptation), but it makes
the *evidence for* TERMINUS relaxation-option (c) — "accept LoRA-family as novel enough" —
substantially stronger: the one banked positive gated on that ruling is real.

---

## Q1 — WHERE does the gain live? Entirely the TEXT stream (image untouched)

Per-modality kNN AUC, three encoders on MHC-ZH:

| stream | CLIP | frozen-Qwen | LoRA-Qwen | LoRA − frozen | LoRA − CLIP |
|---|---|---|---|---|---|
| **train-LOO img** | 0.718 | 0.721 | **0.714** | **−0.007** | −0.004 |
| **train-LOO text** | 0.802 | 0.847 | **0.925** | **+0.078** | +0.123 |
| train-LOO concat | 0.764 | 0.840 | 0.913 | +0.073 | +0.149 |
| kNN homog@20 (concat) | 0.648 | 0.669 | **0.723** | +0.054 | +0.075 |
| **dev img** (held-out) | 0.773 | 0.821 | 0.814 | −0.007 | +0.041 |
| **dev text** (held-out) | 0.733 | 0.869 | **0.931** | **+0.062** | +0.198 |

- **The image stream does not move under LoRA** (train-LOO −0.007; held-out dev −0.007). LoRA
  does **not** repair a collapsed vision channel — MHC-ZH's image stream was never collapsed
  (AUC ~0.72 for all three; the MHC-EN image collapse to 0.599 is EN-specific, F44).
- **The entire gain is in the text stream** (train-LOO +0.078 over frozen; held-out dev +0.062).
  The held-out-dev confirmation matters: it rules out the train-LOO number being merely a LoRA
  "saw train" inflation — the localization (text moves, image doesn't) holds on data LoRA never
  adapted on.
- **Neighborhood purity rises the most of any lever measured** (+0.075 over CLIP), i.e. LoRA
  genuinely re-organizes the space, not just re-scales it.
- Interpretation (consistent-with, not proven here): `text_feats` is the LLM final hidden state
  and `img_feats` the pooled vision-token states; LoRA on the LLM backbone would sharpen the
  former far more than the latter — matching the observed text-only movement. **Fusion balance
  is NOT changed by LoRA** — both blocks stay unit-L2-normed 50/50; what changed is the *content*
  of the text block.

**Answer:** the +0.0313 lives in a sharpened **text/transcript representation**, not an image
fix and not a fusion rebalance.

---

## Q2 — Pareto or rotation? **Pareto** (converts) — vs frozen-Qwen's **rotation** (doesn't)

Final-epoch **TEST** per-class recall (3-seed mean, read from banked `Test_Retrieval` lines;
minority = hate, test pos-rate ≈0.30):

| arm | test acc | test mF1 | hate recall | non-hate recall | Δacc | Δhate-rec | Δnonhate-rec | shape |
|---|---|---|---|---|---|---|---|---|
| CLIP | 0.8143 | 0.7720 | 0.6370 | 0.8910 | — | — | — | baseline |
| **frozen-Qwen** | 0.8031 | 0.7712 | 0.7111 | 0.8429 | **−0.0112** | +0.0741 | **−0.0481** | **ROTATION** |
| **LoRA-Qwen** | 0.8456 | 0.8173 | 0.7482 | 0.8878 | **+0.0313** | **+0.1111** | **−0.0032** | **PARETO** |

- **frozen-Qwen ZH = rotation:** +0.074 hate recall bought with −0.048 non-hate recall → net
  acc −0.011, mF1 −0.001. The encoder re-ranks (its text AUC edge is real, +0.045) but only
  swaps errors between classes — the signature of an **unconvertible easy-example edge**,
  exactly what **B5** proved for ZH ("roc +0.050 unconvertible at any threshold incl. label
  oracle") and the same shape as the **MHC-EN** frozen-swap failure (F44).
- **LoRA-Qwen ZH = Pareto:** +0.111 hate recall at **−0.003** non-hate cost → both acc (+0.031)
  and mF1 (+0.045) move up, mF1 more (minority-weighted). This is the **HateMM encoder-swap
  signature** (hate recall +0.116 at zero non-hate cost, F44) — a genuine decision-boundary
  move, not a re-rank.

**Answer:** representation-limited **conversion (real)**, not an unconvertible edge. LoRA's
larger text-stream gain crosses the threshold from "re-rank" (frozen rotation) to "move the
boundary" (Pareto). The bigger mF1-than-acc gain (+0.045 vs +0.031) confirms the improvement is
concentrated on the minority (hate) class the frozen encoders could not decide.

> **Why frozen fails but LoRA converts, in one line:** the ZH hate signal is text/context-borne;
> the frozen-Qwen text edge (+0.045 AUC) is large enough to re-rank but not to re-decide (easy-
> example ordering, B5); LoRA's text edge (+0.078 *on top*, to 0.925 AUC) is large enough to
> re-decide → Pareto minority-recall conversion.

---

## Q3 — Why does val-selection kill it (+0.0246 FAIL)? **78-dev selection noise, not instability**

Per-seed val-selection (argmax dev-acc, ep≥5, roc tie-break) vs final-epoch vs oracle-best test:

| arm | seed | val-sel ep (dev-acc) | test@val-sel | test@final(29) | oracle-best ep | val-sel regret |
|---|---|---|---|---|---|---|
| CLIP | 0/1/2 | 29/28/25 | .8054/.8054/.8121 | .8054/.8054/.8322 | 27/26/23 | +.020/+.020/+.027 |
| frozen-Qwen | 0/1/2 | 22/25/28 | .7919/.8121/.8054 | .8188/.8054/.7852 | **29/17/15** | +.027/**+.047**/+.020 |
| **LoRA-Qwen** | 0/1/2 | 20/26/19 | .8322/.8255/.8389 | .8456/.8389/.8523 | 11/24/27 | **+.013/+.013/+.020** |

Means: CLIP val-sel test 0.8076 (final 0.8143); frozen 0.8031 (=final); **LoRA 0.8322 (final
0.8456)**. LoRA−CLIP shrinks **+0.0313 (final) → +0.0246 (val-sel)**.

Evidence the FAIL is a selection artifact, not LoRA instability:

1. **LoRA is the MOST stable of the three arms** — lowest val-sel regret (+0.013–0.020 vs
   frozen's +0.020–0.047), and the tightest per-seed final-epoch test spread (0.8389–0.8523).
   The **real** instability is frozen-Qwen (oracle-best epochs 15/17/29 — chaotic dev↔test map).
2. **Dev saturates below its ceiling.** LoRA's dev-acc plateaus at **0.8718 across many epochs
   (val-sel picks ep19/20/26, all at 0.8718)** while test keeps climbing to the final epoch — so
   the argmax is a near-coin-flip among tied plateau epochs that all **undershoot** ep29. This is
   a resolution limit of the **78-sample dev**, not an unstable model.
3. **Val-selection undershoots ALL arms** (CLIP −0.0067, LoRA −0.0134 vs their finals); it just
   costs LoRA slightly more because LoRA's test improves more between the plateau and ep29. The
   gap stays **positive under both protocols** (+0.0313 / +0.0246); only the +0.030 bar is the
   line it straddles.

**Answer:** the known "78-dev selection costs ~2 acc pts" problem (novelty-scope memory), acting
on a LoRA arm whose dev-acc saturates early. The **final-epoch PASS is the more reliable read**;
the val-sel FAIL is a low-resolution-dev protocol artifact, not evidence of a fragile mechanism.

---

## Synthesis for the D7 ruling (what the user is deciding)

- **The ZH LoRA positive is a genuine mechanism**, not a selection/ranking artifact: a localized
  (text-stream), interpretable (task-adaptation of the language representation to ZH hate),
  Pareto (minority-recall +0.111 at ~0 cost) **representation-limited conversion** — the same
  conversion signature as the project's most robust positive (HateMM encoder-swap).
- **It is categorically different from the frozen encoder swaps that fail on MHC.** frozen-Qwen
  produces an *unconvertible rotation* (ZH −0.0112, MHC-EN wash); LoRA produces a *convertible
  Pareto gain*. The distinction is not encoder identity but whether the representation gain is
  large enough to move the decision boundary. LoRA's is; frozen's is not.
- **But it is boxed:** single dataset (ZH only), marginal (+0.0313; val-sel +0.0246 < bar),
  single encoder draw, protocol-fragile on a 78-dev. Even accepting LoRA novelty, a ≥2-dataset
  story would be HateMM (frozen-swap) + ZH (LoRA) — **both encoder-class**, one marginal.
- **New axis? No.** This lives inside Axis B (encoder identity/adaptation), D7-novelty-dead by
  user ruling F24. It is *not* generative. Its value is **evidentiary**: it upgrades
  `TERMINUS_round3` relaxation-option (c) ("accept LoRA-family as novel enough") from
  "definitional, one marginal draw" to "definitional, and the draw is a mechanistically-verified
  real conversion" — strengthening the *case for* (c) without lifting the ruling only the user
  can make. It also sharpens option-(c)'s cost: the LoRA gain is a text-representation adaptation
  (a 2024-25-standard technique), so the novelty question is squarely "does SFT-adapting the
  encoder count," and the *performance* half of that question is now answered YES-but-thin.

---

## Provenance / reproduction

- Feature caches (read-only, no re-extract): `data/CLIP_Embedding/MHC_zh/{train,dev_seen}_{openai_clip-vit-large-patch14-336_HF,Qwen2.5-VL-7B-Instruct_HF,Qwen2.5-VL-7B-Instruct-LoRA_HF}.pt`
  (IDs verified identical across all three encoders; LoRA = the Jul-2 cache the enc3seed runner
  consumed for job 13150).
- Banked trainlogs (read-only, per-epoch Val/Test already logged during the completed runs — no
  new test evaluation): `slurm/logs/enc3s_MHC_zh_{tag}_seed{0,1,2}_{13115|13150}.trainlog`.
  Test per-class recall derived as `nonhate = 2*macroR − hate_recall`; cross-checked against
  logged acc (e.g. LoRA 0.30·0.7482+0.70·0.8878 = 0.846 ✓).
- Scripts: `scripts/analysis/encoder_swap_geometry.py`, `scripts/analysis/b3_zh_lora_trainlog_parse.py`
  (+ `_out.json`). conda `HateVideo`, CPU, seconds.
- Anchors: `B3_VERDICT_REVIEW.md` (job 13150, final +0.0313 / val-sel +0.0246), `B5_VERDICT_REVIEW.md`
  (`50f01b9`, frozen-Qwen ZH edge unconvertible), `ENCODER_SWAP_DIAGNOSIS.md` (F44, `8a48938`,
  HateMM Pareto / MHC-EN rotation reference), `TERMINUS_round3_mllm_plus3.md` relaxation-option (c).
- **Required statements:** no NEW held-out test evaluation was run; test metrics are read from
  banked completed-run trainlogs (same provenance discipline as B3_VERDICT_REVIEW). Gold read =
  train + dev `labels` (geometry) and banked logged test metrics (diagnostic only). No `state/`,
  prereg, config, `research-wiki/`, or frozen artifact mutated. Not pushed.
