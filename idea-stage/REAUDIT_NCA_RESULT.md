# RE-AUDIT — NCA / soft-kNN head loss (τ=0.1) — RESULT

Run 2026-08-17. Design, arms, seed range, read-outs and the decision rule were frozen in
`idea-stage/REAUDIT_NCA_FREEZE.md` and **committed as `5dcfc41` before any run in the frozen seed
range was executed**. One grid, one submission, 60/60 runs complete, 0 failures, no re-run, no tuning
after any number was seen. Local RTX 5090 only; zero API, zero cloud, zero cost.

- Driver `idea-stage/reaudit_nca/run_grid.sh`; log `logging/runs/reaudit_nca/run.log` (elapsed 526 s)
- Analyzer `idea-stage/reaudit_nca/analyze.py`; raw `idea-stage/reaudit_nca/mhczh_results.json`
  (per-seed values for every arm and contrast)
- Instrument replication `idea-stage/reaudit_nca/instrument_s012.json`

---

# VERDICT: **NOT REVIVED.** And this time on evidence, not on a bar.

The primary contrast is +0.0038 with a CI straddling zero. More decisively, the **dev-side contrasts
are negative with CIs excluding zero** (dev macro-F1 −0.0083, dev acc −0.0064) and the **final-epoch
test contrast is negative with its CI excluding zero** (−0.0060). The NCA head loss makes this model
worse. The historical +0.0112 is a property of the epoch-selection rule, not of the loss.

---

## 1. The original verdict, verbatim

`refine-logs/NCA_VERDICT_REVIEW.md` §5 and §6:

> **MARGINAL note (§7.2, B3 §2.2 precedent):** A1a NCA τ=0.1 × ZH val-sel is a clean 3/3-positive acc
> result (mean **+0.0112** acc / **+0.0113** mF1) that survives KS-arm-dead but sits **below** the
> +0.030 FORMAL bar AND **below** the ±0.014 head-seed noise band (§2.3) — a within-noise clean-sign
> positive.

> the sole survivor, A1a NCA τ=0.1 × ZH, survives KS-arm-dead on a within-noise clean-sign val-sel
> positive (+0.0112 acc / +0.0113 mF1) but sits below the FORMAL bar and below the ±0.014 head-seed
> noise band — measured-not-promoted limbo, D7-DEAD. The loss↔inference-mismatch axis is CLOSED

## 2. Instrument checks, all passed before the frozen run

| check | result |
|---|---|
| NCA code path unchanged since the 2026-07-25 freeze | `_nca_head_loss`, `_build_nca_bank`, the `head_loss` dispatch: **zero changed lines** vs the frozen shas. All later edits to `loss.py` / `run_rac.py` are additive, default-off blocks. |
| harness determinism | `MHC_zh_nca01_s0` run twice → **all 60 Val/Test_Retrieval metric lines identical** |
| grid completion | 60/60 runs, rc 0, 30/30 epochs and 60/60 metric lines each |
| historical effect replicates on this hardware | seeds 0/1/2, both arms re-run locally: HIST test-acc **+0.0134** (historical +0.0112), HIST test-macro-F1 **+0.0111** (historical +0.0113) |

Bit-level match to the banked cluster trainlogs (jobs 13150 / 13482) is impossible — those ran on
A100 under the old torch, this machine is RTX 5090 / torch 2.7.1+cu128, and per-epoch values shift by
1–2 points. The design absorbs it: **both arms are re-run here, same hardware, same image, same
seeds**, so the drift is common-mode and cancels inside the seed-paired contrast. No banked cluster
number enters any contrast below.

## 3. The frozen grid — MHC_zh, 30 seeds (41000–41029), both arms

| arm | P1 test mF1 | P2 test mF1 | HIST test mF1 | HIST test acc | dev mF1 (best) | mean P1 epoch |
|---|---|---|---|---|---|---|
| floor (triplet, deployed) | 0.7885 ± 0.0231 | **0.8250** ± 0.0076 | 0.7928 ± 0.0186 | 0.8242 ± 0.0160 | **0.8627** ± 0.0163 | 13.4 ± 6.9 |
| nca01 (`--head_loss nca --nca_tau 0.1`) | 0.7923 ± 0.0159 | 0.8190 ± 0.0098 | 0.8020 ± 0.0200 | 0.8329 ± 0.0177 | 0.8544 ± 0.0060 | 9.0 ± 2.9 |

Seed-paired contrasts `nca01 − floor`, paired bootstrap 95 % CI over 30 seeds, 20 000 resamples:

| read-out | contrast | 3-seed (2026-07-25) | **30-seed mean** | boot SE | **95 % CI** | seeds + |
|---|---|---|---|---|---|---|
| **P1 (gating)** | test macro-F1 @ dev-mF1-argmax | — | **+0.00384** | 0.00535 | **[−0.00640, +0.01450]** | **15/30** |
| P1 | test acc @ dev-mF1-argmax | — | +0.00448 | 0.00478 | [−0.00469, +0.01409] | 14/30 |
| **P2** | test macro-F1 @ final epoch | — | **−0.00597** | 0.00243 | **[−0.01055, −0.00107]** | **8/30** |
| **HIST** | test macro-F1 @ dev-acc-argmax | **+0.0113** | **+0.00924** | 0.00506 | **[−0.00047, +0.01923]** | **18/30** |
| **HIST** | test acc @ dev-acc-argmax | **+0.0112** | **+0.00873** | 0.00446 | **[+0.00022, +0.01767]** | **17/30** |
| dev | dev macro-F1 (best, ≥ warmup) | — | **−0.00826** | 0.00347 | **[−0.01527, −0.00176]** | **9/30** |
| dev | dev acc (best, ≥ warmup) | — | **−0.00640** | 0.00311 | **[−0.01280, −0.00043]** | **6/30** |

### Frozen rule applied verbatim

1. **REVIVED** iff P1 has mean ≥ +0.005 **and** CI excludes zero.
   P1 = **+0.00384** — below the bar — and its CI **contains zero**. **Fails both clauses.**
2. Else **SELECTION-RULE-BOUND** iff HIST passes the same bar on **both** test acc and test macro-F1.
   HIST acc passes (+0.00873, CI [+0.00022, +0.01767]); HIST macro-F1 **fails**, its CI contains zero
   (+0.00924, [−0.00047, +0.01923]). **Not both. Fails.**
3. → **NOT REVIVED.**

The HateMM 15-seed leg was conditional on clause 1 and was therefore **not run**, as frozen.

## 4. What actually happened — the mechanism

The historical number was not inflated. It was attached to the wrong quantity.

**The 3-seed point estimate was roughly right.** HIST acc went +0.0112 → **+0.00873**, HIST macro-F1
+0.0113 → **+0.00924**. Unlike the OCR case in `REAUDIT_RESULT.md` (3-seed read inflated 1.9×), the
old estimate survives powering almost unchanged. What does not survive is the **3/3 sign**: the
per-seed distribution has std 0.025 (acc) and only **17/30 seeds positive**, so P(positive) ≈ 0.57 and
P(3/3 same sign) ≈ **0.18**. The original "clean 3/3-positive" was an ordinary draw from a
near-null distribution, not evidence of consistency. The original verdict's instinct — *within
noise* — was correct; the ±0.014 band that produced it happened to land on the right answer.

**The loss is worse where no epoch selection can rescue it.** Two read-outs involve no winner's-curse
and both are negative with CIs excluding zero: dev macro-F1 **−0.0083** (9/30) and final-epoch test
macro-F1 **−0.0060** (8/30). Swapping the triplet head term for the NCA term costs about half a point
of genuine fit.

**Where the apparent test-side gain comes from.** The NCA arm trains to an **earlier, flatter, lower**
dev optimum: selected epoch 9.0 ± 2.9 vs 13.4 ± 6.9, and dev macro-F1 dispersion across seeds
**0.0060 vs 0.0163** (0.37×). A flatter dev curve means the dev-argmax epoch pick overfits dev less,
so the dev→test generalisation gap shrinks:

| gap (dev − test) at the selected epoch | floor | nca01 | delta | 95 % CI |
|---|---|---|---|---|
| at the P1 (dev-mF1-argmax) epoch | 0.0742 | 0.0621 | −0.0121 | [−0.0267, +0.0016] |
| at the HIST (dev-acc-argmax) epoch | 0.0515 | 0.0364 | **−0.0151** | **[−0.0276, −0.0039]** |

So the candidate buys **−0.012 to −0.015 of epoch-selection winner's curse** while paying **−0.008 of
dev fit**. Under the dev-accuracy argmax the first term is larger, and the difference surfaces as a
+0.009 test-side number. Under the dev-macro-F1 argmax the two nearly cancel (+0.004, CI containing
zero). Under the final epoch, where no selection happens at all, only the cost remains (−0.006, CI
excluding zero). That ordering is the whole result, and it is consistent across all three read-outs.

**Diagnosis:** the NCA / soft-kNN head loss is a regulariser that stabilises the dev curve and lowers
its peak. Its historical +0.0112 was the stabilisation showing through an accuracy-argmax epoch
selector, not a gain in what the head can do. The `loss ↔ inference mismatch` hypothesis — that
aligning the training objective with the deployed kNN vote should help — is measured at
**+0.004 ± 0.005 on the primary read-out and negative wherever selection is removed**, and is closed
on evidence.

**Secondary, real, and reusable:** the NCA arm cuts test macro-F1 seed dispersion at the P1 epoch by
31 % (std 0.0231 → 0.0159) and dev macro-F1 dispersion by 63 %. If a future candidate ever needs
seed-stable head training rather than a better head, this is a working knob — but it costs ~0.6 to
0.8 points of accuracy to buy that stability, so it is not free.

## 5. Protocol note

This re-audit is a counterexample to the assumption that under-powered kills are usually
false negatives. Here the powered measurement **confirms** the original kill and identifies a defect
the original never saw: the claim was stated under a dev-**accuracy** epoch selector while the
project reports under dev-**macro-F1**, and the effect exists only in the former. When a candidate's
headline number is selection-rule-bound, the dev-side contrast is the cheapest tell — it was
−0.0083 with the CI excluding zero, and would have been visible at any seed count with a paired test.

## 6. Ledger

- Total GPU: 60 head-only runs, 526 s wall. Zero API. Zero cloud.
- Zero test-set tuning: test labels were read only to score an epoch chosen by dev.
- Seeds 41000–41029 are consumed and must not be reused for this arm pair.
- HateMM leg not run (conditional clause not triggered).
