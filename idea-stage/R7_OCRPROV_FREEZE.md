# R7-2 — on-screen-text provenance rule channel at the decision layer: design and decision rule, frozen 2026-08-17

Frozen **before `idea-stage/r7_ocrprov/analyze.py` was executed even once**, i.e. before any
candidate macro-F1 number existed. Zero API cost, zero cloud cost. Local RTX 5090.

## 0. Prior-record check (done before this design was written)

| prior record | what it actually tested | does it kill this pilot? |
|---|---|---|
| `idea-stage/PILOT_C_RESULT.md` — "on-screen-text **provenance** separability", AMBIGUOUS (`arm2 - arm1c` = **+0.0044**, bar +0.010) | Provenance typing by **geometry**: a track is "overlay" iff it appears in >=50 % of text-bearing windows and its centre std <= 0.05 on both axes; "scene" otherwise. Two mean-pooled OCR embedding blocks concatenated into the **feature vector**, 5-fold OOF on HateMM **train only**. `idea-stage/pilot_c_ocr_provenance.py` contains **no lexical rule of any kind** (grep: no watermark, no regex over token content). | **No.** Different signal (lexical source vocabulary vs. on-screen stability geometry), different integration point (decision-layer combiner on top of a fixed head vs. feature-space concatenation), different protocol (real train/val/test with the R6 measurement protocol vs. train-only OOF). Its own caveat §1 says the effect was inside harness noise — which is a reason to test a different mechanism, not a reason to stop. |
| `idea-stage/OCR_FUSION_PILOT_RESULT.md` / `idea-stage/A0_OCR_E2E_RESULT.md` | Mean-pooled OCR embeddings as an extra **feature block**. | No — feature-level fusion, not a rule channel; the head there changes, here it does not. |
| `refine-logs/` OCR veto (RUNBOOK §7) | Blanket ban on OCR as a model input. | **Superseded** by the user's 2026-08-08 ruling (CLAUDE.md §模态通道): OCR is unbanned, on the evidence that 30.1 % of missed detections have on-screen-text evidence with no speech evidence (OR 2.29). |

**Conclusion: not previously tested, not currently vetoed. Proceed.**

## 1. Hypothesis

Some HateMM videos are misclassified because the on-screen text belongs to the *carrier* of the
clip rather than to its content — stock-footage agency watermarks, broadcast chyrons, platform UI
chrome, creator handle watermarks, date burn-ins. A small, fixed set of purely lexical rules that
detect these markers, fed into a **global** combiner alongside the head's own score, recovers some
of that error mass without touching the head.

## 2. The rule channel (frozen)

`idea-stage/r7_ocrprov/rules.py`, SHA-256
`51031d9c5f1df03a866ba3e1472fcc3c6adabc2af4777f9122e9193fd2dce6e9`.
Six families, fixed in this order and not changed after this document:

`["stock_watermark", "news_chyron", "date_stamp", "ui_text", "handle_watermark", "copyright"]`

Each yields one **binary indicator** (count > 0) per video. Deterministic, no I/O, no randomness.

Vocabularies were built **label-blind** on the 851 train+val OCR rows only
(`data/OCR/HateMM/ocr_video.jsonl`, SHA-256 `933a0fcd…2842c70`), by document-frequency and
intra-video repeat statistics with no label file opened and no test cache opened; the evidence
and the rejected terms are recorded in `idea-stage/r7_ocrprov/vocab_recon.json`. The motivating
case (`GLOBALIMAGEWORKS` stock watermark, from the round-5 error dump) fires `stock_watermark`.

**Trigger coverage, computed before this freeze, label-blind** (over all 1066 videos, train+val+test):
`stock_watermark` 0.026, `news_chyron` 0.195, `date_stamp` 0.189, `ui_text` 0.161,
`handle_watermark` 0.218, `copyright` 0.030; **any family 0.372**. This is above the 0.05
"scale" floor, so a positive result will not be disqualified as sub-scale. (73 of 851 train+val
videos have empty OCR text, so the reachable ceiling is below 1.0.)

## 3. Arms

**The head is not touched.** One set of 30 HateMM head runs (arm `A0`, cache `R6RO-A0`,
hyperparameters byte-identical to `idea-stage/r6_confirm/run_confirm.sh`, seeds 100-129) provides
per-item dev and test logits at every epoch via the flag-gated `--dump_head_scores`. All four
read-out arms come from those same 30 runs — no retraining, no second grid.

| arm | read-out |
|---|---|
| `A0` | head alone, test macro-F1 at threshold 0.5 (this is the standard ledger number) |
| `COMB0` | L2 logistic regression fitted on **val** with the single feature `[head_logit]`; test macro-F1 @0.5 |
| `COMBR` | ... with `[head_logit, 6 provenance indicators]` |
| `COMBRAND` | ... with `[head_logit, 6 random Bernoulli indicators]`, per-feature rates matched to the rule indicators' marginal train+val trigger rates, drawn from `default_rng(20260817000 + seed)` |

`COMB0` exists so that the *recalibration* component of any `COMBR` gain can be separated from the
*rule information* component; `COMBRAND` exists so that the combiner's own overfitting can be
separated from both. Features are standardised on the fit split; `LogisticRegression(C=1.0,
solver='lbfgs', max_iter=2000)`. Frozen; nothing tuned.

**Law III check.** The combiner is one weight vector applied identically to every item — a global
fixed function of (head score, rule indicators). It is not a per-item selection among alternatives,
so the per-item-selection prohibition is not engaged. This is stated here, before the run, as
required.

## 4. Deviation D1 from the commissioning brief, decided before any result

The brief specified fitting the combiner on **train** with hard labels and selecting the threshold
on val. That is not sound here: the head is trained on those same 744 train items for 30 epochs, so
its train logits are in-sample and near-separable, and the regression would put essentially all
weight on the head logit and none on the rules — the mechanism would be untestable by construction,
independently of whether it works.

**Adopted instead:** fit the combiner on the **val split** (`dev_seen`, n=107), the only split on
which the head's logits are out of sample, and read out at the natural threshold 0.5 of the
calibrated regression. All four red lines are preserved: zero test contact during fitting, the
decision rule frozen here before any candidate number exists, no candidate metric computed during
design, and one single submission. The cost is a small fit set (107 items, 7 parameters), which is
exactly what `COMBRAND` measures.

## 5. Read-out protocol and data discipline

- **P1 (primary)** = epoch selected on val by validation macro-F1, ties to earliest, epochs >= 5;
  **P2 (corroboration)** = epoch 29. Parser and epoch selector imported verbatim from
  `idea-stage/r6_audit/analyze_audit.py`. HateMM test N=215, P=86.
- **Integrity check, hard halt:** the `A0` macro-F1 recomputed from the per-item dump must equal
  the trainlog's reconstructed macro-F1 to 1e-6 for every seed and protocol. If it does not, the
  run aborts rather than reporting.
- Test labels are read only for the final metric. No rule, no term list, no threshold, no epoch
  rule, no regularisation constant is selected on test. The test OCR text is used as an **input**
  only, which the user's 2026-08-09 test-set protocol ruling permits.

## 6. Frozen decision rule

Paired seed-wise over the 30 seeds; paired bootstrap over seeds (20 000 resamples,
`default_rng(20260817)`); primary protocol **P1**.

- **GO** iff all four hold:
  1. `mean(COMBR - A0) >= +0.005`
  2. that delta's paired-bootstrap 95 % CI excludes 0
  3. `mean(COMBRAND - A0) < +0.005` (the random-feature control does not itself clear the bar)
  4. `mean(COMBR - COMB0) > 0` (the rules beat pure recalibration)
- **GO-BUT-SUBSCALE** — the four conditions hold but test any-feature coverage < 0.05.
- **TRICK** — 1-3 hold but 4 fails: the gain is decision-threshold recalibration, not provenance
  information. Recorded as a trick, not a direction.
- **KILL** — anything else.

Reported alongside, not gating: the mean standardised combiner coefficients (which family, if any,
carries weight), P2 as corroboration, and per-split coverage.

## 7. Expected outcome, stated before the run

KILL. Reasoning: `idea-stage/PILOT_C_RESULT.md` §O2 found that the *presence* of on-screen text of
either provenance has AUROC 0.4927 / 0.4568 against the HateMM label — at or below chance — so a
presence-style indicator has little room; and six binary features fitted on 107 val items is a
noisy instrument. The plausible positive path is `news_chyron` and `handle_watermark`, the two
highest-coverage families, acting as a "this is re-broadcast carrier material" prior. This
expectation is recorded so that a positive result cannot be presented as having been anticipated.

## 8. Ordering note, stated plainly

The 30 head runs were launched before this document was committed. They are the **baseline arm**
(`A0`) with hyperparameters identical to the already-banked `r6_confirm` configuration plus a
score-dump flag verified to be a no-op on training; they are not specific to any candidate arm, and
no candidate quantity existed or was computed while they ran. Every candidate arm is produced by
`idea-stage/r7_ocrprov/analyze.py`, which had not been run when this freeze was committed.

## 9. Code touched

`src/utils/metrics.py` — flag-gated per-item logit dump inside `eval_and_save_epoch_end`;
`src/run_rac.py` — `--dump_head_scores`. Default off. Verified: with the flag on, `MHC_zh / A0 /
seed 30` produces a trainlog identical to the flag-off run on all 60 `dev`/`test` epoch lines.
