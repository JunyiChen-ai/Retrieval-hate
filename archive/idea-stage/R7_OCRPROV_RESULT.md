# R7-2 — on-screen-text provenance rule channel: result

Run 2026-08-17. Decision rule frozen and committed at `3ea3741`
(`idea-stage/R7_OCRPROV_FREEZE.md`) **before** `idea-stage/r7_ocrprov/analyze.py` was executed
once. Single submission: the analyzer was run exactly one time, on the complete 30-seed grid, with
no edit between the freeze commit and that run and no re-run afterwards.

- Head runs: `idea-stage/r7_ocrprov/run_heads.sh` → `logging/runs/r7_ocrprov/` (log `run.log`,
  PID `run.pid`, trainlogs `logs/`, per-item dumps `scores/`). **30/30 complete, 0 failures**,
  333 s wall.
- Rules: `idea-stage/r7_ocrprov/rules.py`, SHA-256
  `51031d9c5f1df03a866ba3e1472fcc3c6adabc2af4777f9122e9193fd2dce6e9` (unchanged since the freeze).
- Read-out: `idea-stage/r7_ocrprov/analyze.py` → `idea-stage/r7_ocrprov/results.json`.
- Cost: **¥0**. Zero API calls, zero cloud. Local RTX 5090, shared with another user's job.

# VERDICT: **KILL**

The rule channel does not merely fail to help — it is **worse than a matched set of random boolean
features**, on every seed.

## 1. Headline numbers (P1 primary, HateMM test macro-F1, 30 seeds 100-129)

| arm | mean | std | se |
|---|---|---|---|
| `A0` head alone @0.5 | **0.8730** | 0.0088 | 0.0016 |
| `COMB0` combiner, head logit only | 0.8686 | 0.0062 | 0.0011 |
| `COMBRAND` combiner, head + 6 random bools | 0.8516 | 0.0195 | 0.0036 |
| `COMBR` combiner, head + 6 provenance rules | **0.8267** | 0.0083 | 0.0015 |

Seed-paired deltas, paired bootstrap 95 % CI over seeds (20 000 resamples):

| contrast | mean | 95 % CI | seeds positive |
|---|---|---|---|
| **`COMBR − A0` (gating)** | **−0.0463** | [−0.0509, −0.0415] | **0/30** |
| `COMBRAND − A0` (control) | −0.0213 | [−0.0294, −0.0140] | 4/30 |
| `COMB0 − A0` | −0.0043 | [−0.0079, −0.0006] | 5/30 |
| `COMBR − COMB0` | −0.0419 | [−0.0442, −0.0399] | 0/30 |
| `COMBR − COMBRAND` | −0.0249 | [−0.0317, −0.0176] | 3/30 |

P2 (final epoch) corroborates in sign on every one of these: `COMBR − A0` = **−0.0403**,
`COMBR − COMB0` = −0.0434, `COMBR − COMBRAND` = −0.0311, all 0-2 seeds positive.

Against the frozen rule: condition 1 (`mean ≥ +0.005`) **fails** (−0.0463), condition 4
(`COMBR > COMB0`) **fails** (−0.0419). → **KILL**.

`A0` = 0.8730 sits 0.0017 (≈1 se) below the ledger's HateMM A0 = 0.8747, which was measured on
seeds 30-89; the two are the same quantity on disjoint seed ranges and agree.

## 2. Integrity guard

For every seed and both protocols, the `A0` macro-F1 recomputed from the per-item logit dump was
required to equal the trainlog's confusion-matrix reconstruction to 1e-6, on pain of abort. It
matched on **60/60** (30 seeds × 2 protocols). The dump and the logged metrics describe the same
predictions, so the combiner arms are built on the same object the ledger scores.

## 3. Trigger coverage (label-blind, reported separately as required)

| split | n | any family | stock_watermark | news_chyron | date_stamp | ui_text | handle_watermark | copyright |
|---|---|---|---|---|---|---|---|---|
| train | 744 | 0.353 | 0.024 | 0.192 | 0.177 | 0.152 | 0.203 | 0.027 |
| val | 107 | 0.458 | 0.019 | 0.243 | 0.243 | 0.196 | 0.271 | 0.028 |
| test | 215 | **0.395** | 0.037 | 0.181 | 0.200 | 0.177 | 0.242 | 0.042 |

Coverage is **not** the problem. Test any-family coverage is 0.395, eight times the 0.05
"sub-scale" floor, and four of six families fire on 15-27 % of videos. The channel had ample
surface to work on and still lost 4.6 points.

## 4. Why it lost — the mechanism, not a bug

Mean standardised combiner coefficients (P1, `COMBR`, averaged over 30 seeds):

| feature | coefficient |
|---|---|
| head_logit | +1.9572 |
| copyright | +0.5690 |
| **news_chyron** | **−0.4497** |
| ui_text | +0.3416 |
| date_stamp | +0.1604 |
| handle_watermark | −0.0999 |
| stock_watermark | +0.0244 |

The combiner assigns the rules real weight — `news_chyron` at −0.45 and `copyright` at +0.57 are
roughly a quarter of the head's own coefficient. Those weights are fitted on 107 val items and do
not transfer: on val a broadcast chyron reads as "not hateful", and applying that on test moves
items across the boundary in the wrong direction. The two highest-magnitude rule coefficients are
also on the families the freeze doc named as the plausible positive path (`news_chyron`) and one of
the two rarest (`copyright`, 2.8 % of val = **3 videos**, which is enough for an L2 logistic
regression to hand it the second-largest weight in the model).

The random-feature control separates the two failure sources cleanly:

- **Combiner overfitting alone** costs `COMBRAND − A0` = −0.0213. Six uninformative booleans fitted
  on 107 items already lose 2.1 points. That is the price of the decision-layer combiner as such.
- **The rules cost a further −0.0249 on top of that** (`COMBR − COMBRAND`, 27/30 seeds negative).
  Real provenance indicators are *worse* than noise here, because noise gets shrunk toward zero by
  L2 while a feature with a strong, unstable val-side correlation does not.

`idea-stage/PILOT_C_RESULT.md` §O2 had already measured that on-screen-text *presence* indicators
carry AUROC 0.4927 / 0.4568 against the HateMM label — at or below chance. This pilot tested a
different and richer presence family (lexical source vocabulary rather than geometric typing) at a
different integration point, and lands in the same place, harder: presence-style on-screen-text
indicators do not carry label information in HateMM, and giving them fitted weights actively costs
accuracy.

## 5. Deviation from the brief, and whether it can be blamed

**D1 (declared in the freeze, before any number):** the combiner was fitted on **val** (n=107)
rather than on train, because the head's train logits are in-sample after 30 epochs on those same
744 items and would have driven all weight onto the head logit — making the mechanism untestable
by construction.

Could D1 explain the KILL? Partly, and the size is measured, not guessed: the combiner-as-such cost
is `COMBRAND − A0` = −0.0213, and a larger fit set would shrink it. But it cannot explain the
verdict, because the gating quantity is a *within-combiner* comparison as well: `COMBR − COMBRAND`
= −0.0249 with CI [−0.0317, −0.0176] and `COMBR − COMB0` = −0.0419 with CI [−0.0442, −0.0399].
Both compare the rules against a control fitted on the identical split with the identical
regulariser and the identical number of parameters. A bigger fit set would raise all four arms
together; it would not reverse the ordering `COMB0 > COMBRAND > COMBR`, which is the finding.

## 6. Data discipline

- The rule vocabulary was built label-blind, on `data/OCR/HateMM/ocr_video.jsonl` (train+val only);
  no label file and no test OCR cache was opened during vocabulary design
  (`idea-stage/r7_ocrprov/vocab_recon.json`).
- Test OCR text is used as an **input** only, which the user's 2026-08-09 test-set protocol ruling
  permits. Test labels were read only for the final metric.
- Nothing — no term list, no threshold, no epoch rule, no `C` — was selected on test. Epoch
  selection is on val macro-F1 (P1) or fixed at 29 (P2).
- The head was never retrained: all four arms are read-outs of the same 30 runs.

## 7. Law III

Recorded in the freeze and unchanged by the result: the combiner is a single weight vector applied
identically to every item, not a per-item selection among alternatives. The prohibition is not
engaged. This is noted for completeness; it does not affect the verdict.

## 8. What this closes

The decision-layer provenance-rule channel is closed. Together with `PILOT_C_RESULT.md`
(geometric typing, AMBIGUOUS at +0.0044 against a +0.010 bar) and `OCR_FUSION_PILOT_RESULT.md` /
`A0_OCR_E2E_RESULT.md` (mean-pooled OCR as a feature block), the three integration points a
provenance signal could enter through — feature concatenation, typed feature blocks, and decision-
layer rules — have now all been measured on HateMM, and none clears the current +0.005 bar. The
"on-screen text carries source information the model is missing" hypothesis is not supported at any
of them.

What is **not** closed: the Gate-C finding that motivated unbanning OCR (30.1 % of missed detections
have on-screen-text evidence and no speech evidence, OR 2.29) concerns the *content* of on-screen
text, not its provenance. Nothing here speaks to content-level OCR use.

## 9. Expectation check

The freeze predicted KILL, on the grounds that `PILOT_C_RESULT.md` §O2 put presence-indicator AUROC
at or below chance and that six binary features fitted on 107 items is a noisy instrument. The
prediction was correct in direction. It **understated the magnitude**: the outcome is not a null but
a 4.6-point loss, and the rules being beaten by random booleans of matched marginal rate was not
anticipated.

## 10. Reproduction

```
bash idea-stage/r7_ocrprov/run_heads.sh                 # 30 head runs, ~333 s
python idea-stage/r7_ocrprov/analyze.py                 # CPU, seconds
```
Raw: `idea-stage/r7_ocrprov/results.json` (per-seed values for every arm and protocol),
`logging/runs/r7_ocrprov/logs/*.trainlog`, `logging/runs/r7_ocrprov/scores/*.jsonl`.
