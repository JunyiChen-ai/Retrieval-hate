# RE-AUDIT — results

Run 2026-08-17. Design, arms, seed ranges, read-out protocols and the decision rule were frozen in
`idea-stage/REAUDIT_FREEZE.md` and **committed as `b8c680c` before any candidate contrast in the new
seed ranges was computed**. Four grids, one submission each, no re-run, no tuning after any number
was seen. Zero API cost.

- Driver (GPU) `logging/runs/reaudit_gpu/drive.sh`; log `run.log`, PID `run.pid`
- Driver (CPU) `logging/runs/reaudit_cpu/run.sh`; log `run.log`, PID `run.pid`
- Raw: `idea-stage/reaudit/r1a_r2_results.json`, `r1b_results.json`, `r3_results.json`,
  `r4_HateMM_results.json`, `r4_MHC_zh_results.json`

---

# VERDICT: **no revival.** 0 of 5 candidates pass the frozen bar.

The under-powered-kill hypothesis was tested and it does **not** hold for this candidate set. What
the powered re-measurement shows instead is sharper than either the old verdicts or the hypothesis:

1. **The one effect that survives powering is real but sits 0.00015 below the bar.** Frozen-space
   OCR fusion, historically +0.0094 at 3 seeds, measures **+0.00485** at 30 seeds with a
   paired-bootstrap 95 % CI of [+0.00376, +0.00592] and 29/30 seeds positive. It is not noise — but
   the 3-seed read was inflated ~1.9×, and the powered estimate lands just under +0.005.
2. **That same effect reverses sign on the deployed substrate.** Injected into the current A0 text
   stream and read at test level, the identical OCR vector is **−0.0088** against A0 and **−0.0054**
   against a dimension-matched random block, both CIs excluding zero. The gain is a property of the
   CLIP frozen-space train-OOF harness, not of on-screen text.
3. **Two of the small effects were noise and collapsed as predicted.** OCR provenance typing went
   from +0.0044 to **+0.0001** (16/30 seeds), and its information-matched control hardened to
   **−0.0055** with the CI excluding zero.

---

## 1. Instrument checks, all passed before the frozen runs

| check | result |
|---|---|
| `RAOC-A0` seed 30 vs banked `logging/runs/r6_confirm/logs/HateMM_A0_s30.trainlog` | **all 60 dev/test epoch lines identical** |
| `TEXTMERGE-A0` seed 0 vs banked `logging/runs/text_merge/logs/A0_s0.trainlog` | **all 60 dev/test epoch lines identical** |
| `reaudit_ocr.py` synthetic smoke vs `logging/runs/pilot_c/smoke_synthetic.json` | **all four original arms bit-for-bit**, with arm `1r` added |
| `analyze_grid.py` replayed on the banked R6-1C logs (MHC_zh, 60 seeds) | reproduces the published **P1 CAT−A0 = +0.0185**, **P2 = +0.0115 CI [+0.0056, +0.0162]** |
| grid completion | **0 failed runs**, every run 30/30 epochs, every confusion-matrix reconstruction within 1e-4 of the logged macro-F1 |

The +0.0088 OCR-vs-random contrast in R1a and the −0.0088 in R1b are opposite-signed measurements of
related quantities on different substrates, not a contradiction inside one instrument; see §6.

---

## 2. R1a — OCR mean-block fusion, original frozen-space harness, 30 seeds

Out-of-fold macro-F1 over the 744 HateMM **train** videos. `dev_seen` and `test` HALT-guarded, as in
the 2026-08-09 original. Seeds 20260900–20260929, disjoint from the original three.

| arm | dim | 3-seed mean (2026-08-09) | **30-seed mean** | std |
|---|---|---|---|---|
| 0 baseline | 1792 | 0.8104 | **0.8143** | 0.0028 |
| 1 untyped OCR-30 | 2560 | 0.8198 | **0.8191** | 0.0019 |
| 1c untyped ×2 | 3328 | 0.8134 | **0.8135** | 0.0046 |
| **1r random block (new control)** | 2560 | — | **0.8103** | 0.0030 |
| 2 typed overlay ‖ scene | 3328 | 0.8178 | **0.8137** | 0.0028 |

Seed-paired contrasts, paired bootstrap 95 % CI over 30 seeds, 20 000 resamples:

| contrast | 3-seed | **30-seed mean** | boot SE | **95 % CI** | seeds + |
|---|---|---|---|---|---|
| **arm1 − arm0** (gating) | +0.0094 | **+0.00485** | 0.00055 | **[+0.00376, +0.00592]** | **29/30** |
| **arm1 − arm1r** (gating, new control) | — | **+0.00883** | 0.00058 | **[+0.00769, +0.00997]** | **30/30** |
| arm1r − arm0 | — | −0.0040 | — | [−0.0054, −0.0026] | 4/30 |
| arm1c − arm1 | −0.0064 | −0.0056 | — | [−0.0073, −0.0038] | 5/30 |
| arm1c − arm0 | +0.0030 | −0.0007 | — | [−0.0026, +0.0010] | 15/30 |
| arm2 − arm0 | +0.0074 | −0.0006 | — | [−0.0020, +0.0008] | 15/30 |

**Frozen rule**: both `arm1 − arm0` and `arm1 − arm1r` must have mean ≥ +0.005 with CI excluding 0.
`arm1 − arm1r` = +0.00883 **passes**. `arm1 − arm0` = **+0.00485**, which is **0.00015 below the
bar** — the CI excludes zero but the point estimate does not clear +0.005.

### → **NOT REVIVED** (by 0.00015)

The rule is applied verbatim, as frozen. Three things must be said about the margin without
softening the verdict:

- **The effect is real and the sign is not in doubt.** 29/30 seeds positive, boot SE 0.00055, CI
  well clear of zero. This is not a "consistent with zero" result; it is a small positive that the
  bar happens to sit on top of.
- **The 3-seed read was inflated ~1.9×.** +0.0094 → +0.00485. This is the *opposite* of the failure
  mode this round was hunting: the historical under-powered estimate was too generous, not too
  harsh. The 2026-08-09 verdict of AMBIGUOUS was closer to the truth than its number was.
- **The new random control changes the reading of the old one.** `arm1r − arm0` = −0.0040 (4/30):
  adding 768 content-free dimensions *costs* 0.4 points. So the capacity story implied by
  `arm1c − arm0` = +0.0030 at 3 seeds was itself noise (30-seed value −0.0007, CI containing zero),
  and OCR content is worth **+0.0088** over an equally-sized random block. Under a rule written
  around the dimension-matched control alone, this candidate would have passed.

---

## 3. R2 — OCR overlay/scene provenance typing, 30 seeds

Same run, same seeds.

| contrast | 3-seed | **30-seed mean** | boot SE | **95 % CI** | seeds + |
|---|---|---|---|---|---|
| **arm2 − arm1c** (gating) | +0.0044 | **+0.00014** | 0.00088 | [−0.00158, +0.00185] | **16/30** |
| **arm2 − arm1** (gating, information-matched) | −0.0020 | **−0.00546** | 0.00056 | **[−0.00652, −0.00435]** | **1/30** |

### → **NOT REVIVED**

The primary quantity collapses to **+0.0001 with 16/30 seeds positive** — indistinguishable from
zero, exactly as the original result's caveat 1 predicted ("that verdict should be read as
consistent with zero, not as a small positive"). The information-matched contrast hardens in the
negative direction and its CI now excludes zero: splitting the OCR text into overlay and scene
blocks **costs 0.55 macro-F1 points** relative to pooling it into one block, on 29 of 30 seeds.

The 2026-08-09 AMBIGUOUS was carried entirely by `arm1c` being a handicapped control. With 30 seeds
`arm1c − arm0` is −0.0007 (CI containing zero) rather than +0.0030, and the +0.0044 disappears with
it. **Provenance typing is closed, and this time on evidence rather than on a bar.**

---

## 4. R1b — the same OCR vector on the deployed substrate, at test level, 30 seeds

HateMM test macro-F1, seeds 300–329, arms built by `idea-stage/reaudit/build_ocr_arms.py`
(meta + SHA-256 in `build_ocr_meta.json`).

| protocol | contrast | mean | **95 % CI** | seeds + |
|---|---|---|---|---|
| **P1** | **OCR − A0** | **−0.0088** | **[−0.0128, −0.0050]** | 7/30 |
| **P1** | **OCR − RAND** | **−0.0054** | **[−0.0101, −0.0009]** | 12/30 |
| P1 | RAND − A0 | −0.0034 | [−0.0078, +0.0011] | 13/30 |
| P2 | OCR − A0 | −0.0102 | [−0.0151, −0.0056] | 8/30 |
| P2 | OCR − RAND | −0.0080 | [−0.0133, −0.0025] | 5/30 |
| P2 | RAND − A0 | −0.0021 | [−0.0062, +0.0019] | 15/30 |

### → **NOT REVIVED**, and negative with the CI excluding zero on both protocols

This is the informative half of the OCR result. The same 768-d OCR vector that is worth +0.0088 over
a random block in the CLIP frozen-space train-OOF harness is worth **−0.0054 against the same kind
of random block** when concatenated into the deployed LoRA-Qwen text stream and read on test. It is
not a capacity effect — the random block alone is −0.0034 with a CI containing zero, so OCR is
*worse than content-free noise* of the same size.

It agrees in sign and roughly in size with `A0_OCR_E2E_RESULT.md` (−0.0246 through the learned
third-stream path, 3 seeds, val), which was the only prior test of OCR on this substrate. Two
different injections, two substrates, same direction.

---

## 5. R3 — transcript description merge, 30 seeds

HateMM test macro-F1, seeds 300–329, arms from the existing `*_TEXTMERGE-*.pt` caches.

| protocol | contrast | 3-seed (2026-08-13) | **30-seed mean** | **95 % CI** | seeds + |
|---|---|---|---|---|---|
| **P1** | **TMt − A0** (gating) | −0.0105, 0/3 | **+0.0013** | [−0.0042, +0.0069] | 17/30 |
| **P1** | **TMt − TMshuf** (gating) | +0.0017 | **+0.0132** | **[+0.0083, +0.0183]** | 24/30 |
| P1 | TMall − A0 | −0.0161 | −0.0093 | [−0.0154, −0.0035] | 11/30 |
| P2 | TMt − A0 | — | −0.0024 | [−0.0072, +0.0027] | 9/30 |
| P2 | TMt − TMshuf | — | +0.0104 | [+0.0055, +0.0152] | 24/30 |
| P2 | TMall − A0 | — | −0.0157 | [−0.0224, −0.0090] | 9/30 |

### → **NOT REVIVED**

The primary contrast fails on both counts: +0.0013 is below the bar, its CI contains zero, and P2
disagrees in sign.

**But the 3-seed KILL was measuring nothing.** −0.0105 on 0/3 seeds became **+0.0013 on 17/30** —
the sign reversed and the powered estimate is centred on zero. The 2026-08-13 verdict was a
genuine under-powered read; it simply does not revive into a positive, it dissolves into a null.

**What is real here is the decomposition, and it is new.** `TMt − TMshuf` = **+0.0132** with the CI
excluding zero on both protocols, 24/30 seeds: swapping in a *mismatched* description costs 1.3
macro-F1 points relative to the matching one. So the description content genuinely carries
label-relevant information at this injection point. It buys nothing overall because the merge
operation itself costs about as much as the content returns — `TMt − A0` ≈ 0 is the sum of a real
positive content term and a real negative merge-cost term. The original run reported `TMshuf − A0`
= −0.0122 and read it as "content is worth ≈ 0.002"; at 30 seeds the content term is **6× larger**
than that estimate, and the earlier figure was inside the noise.

---

## 6. R4 — concat vs align fusion, 30 seeds, two datasets

Test macro-F1, seeds 300–329, deployed `R6RO-A0` cache on both datasets, `--fusion_mode` the only
difference between arms.

| dataset | arm | P1 mean | std | mean selected epoch |
|---|---|---|---|---|
| HateMM | ALIGN | **0.8762** | 0.0053 | 18.1 |
| HateMM | CAT | 0.8627 | 0.0067 | **9.2** |
| MHC_zh | ALIGN | 0.8040 | 0.0138 | 20.1 |
| MHC_zh | CAT | 0.8052 | 0.0212 | **9.5** |

| dataset | protocol | contrast | 3-seed (2026-08-13) | **30-seed mean** | **95 % CI** | seeds + |
|---|---|---|---|---|---|---|
| HateMM | **P1** | **CAT − ALIGN** | −0.0031 | **−0.0135** | **[−0.0166, −0.0104]** | 2/30 |
| HateMM | P2 | CAT − ALIGN | −0.0031 | −0.0087 | [−0.0151, −0.0021] | 9/30 |
| MHC_zh | **P1** | **CAT − ALIGN** | +0.0067 | **+0.0011** | [−0.0090, +0.0101] | 15/30 |
| MHC_zh | P2 | CAT − ALIGN | −0.0045 | −0.0032 | [−0.0099, +0.0039] | 14/30 |

### → **NOT REVIVED**, on both the pass clause and the no-harm clause

MHC_zh's historical +0.0067 (2/3 seeds) measures **+0.0011 with 15/30 seeds positive** — a coin
flip. HateMM's historical −0.0031 hardens to **−0.0135 with the CI excluding zero on both
protocols**, which independently fails the ≥ −0.002 no-material-harm clause. Concat fusion is worse,
not merely different.

**A mechanism note the 3-seed run could not see.** The concat arm's mean selected epoch is **9.2 /
9.5 against align's 18.1 / 20.1** — the wider first Linear (2 098 176 vs 1 049 600 parameters)
reaches its dev optimum in roughly half the epochs and then degrades. Its seed std is also larger on
both datasets (0.0067 vs 0.0053; 0.0212 vs 0.0138). The extra capacity is being spent on faster
overfitting. This is consistent with the record's binding scope note — the null belongs to the
`concat + 2× first-Linear` bundle — and it now has a measured signature rather than a caveat.

---

## 7. Summary table

| # | candidate | historical effect | seeds | old verdict | **30-seed effect** | **95 % CI** | new verdict |
|---|---|---|---|---|---|---|---|
| R1a | OCR mean-block fusion (frozen space, train OOF) | +0.0094 | 3 | AMBIGUOUS | **+0.00485** | [+0.00376, +0.00592] | **NOT REVIVED** (0.00015 below bar) |
| R1a | — its new dimension-matched control `arm1 − arm1r` | — | — | — | **+0.00883** | [+0.00769, +0.00997] | passes its own clause |
| R2 | OCR provenance typing | +0.0044 | 3 | AMBIGUOUS | **+0.00014** | [−0.00158, +0.00185] | **NOT REVIVED** |
| R2 | — its information-matched control `arm2 − arm1` | −0.0020 | 3 | — | **−0.00546** | [−0.00652, −0.00435] | hardens negative |
| R1b | OCR mean-block fusion (deployed substrate, test) | — | — | untested | **−0.0088** | [−0.0128, −0.0050] | **NOT REVIVED**, negative |
| R3 | transcript description merge | −0.0105 | 3 | KILL | **+0.0013** | [−0.0042, +0.0069] | **NOT REVIVED**, null |
| R3 | — content term `TMt − TMshuf` | +0.0017 | 3 | — | **+0.0132** | [+0.0083, +0.0183] | real, but net-zero |
| R4 | concat vs align fusion (MHC_zh) | +0.0067 | 3 | KILL | **+0.0011** | [−0.0090, +0.0101] | **NOT REVIVED** |
| R4 | concat vs align fusion (HateMM) | −0.0031 | 3 | KILL | **−0.0135** | [−0.0166, −0.0104] | **NOT REVIVED**, harm |

**0 of 5 REVIVED.** No literature novelty check was run, because the frozen rule makes it conditional
on a pass and no candidate passed. Zero API spend.

---

## 8. What this round actually establishes

**The false-kill hypothesis was tested honestly and it did not generalise.** R6-1 remains the only
demonstrated false kill in this project. In the five candidates re-measured here, the 3-seed reads
were wrong in every direction — too generous (R1a +0.0094 → +0.0049; R2 +0.0044 → +0.0001), too
harsh (R3 −0.0105 → +0.0013; R4 HateMM −0.0031 → −0.0135 is harsher still), and simply
uninformative (R4 MHC_zh +0.0067 → +0.0011) — but **none of the errors ran in the direction that
would recover a candidate.**

Four findings are worth carrying forward, none of which is a candidate:

1. **A dimension-matched random control is mandatory and the duplicate-block control is not a
   substitute.** `arm1c − arm0` measured +0.0030 at 3 seeds and **−0.0007** at 30. The whole 2026-08-09
   provenance-typing AMBIGUOUS rested on that +0.0030. Meanwhile the proper random control gives
   `arm1r − arm0` = −0.0040: content-free dimensions *cost* accuracy here, so any candidate measured
   only against a plain baseline is being scored on a mixture of content and a capacity penalty.
2. **On-screen text carries real information in the CLIP frozen space and none on the deployed
   substrate.** +0.0088 over a random block in one harness, −0.0054 against the same kind of block in
   the other. Combined with `R7_OCRPROV_RESULT.md` (rule layer, −0.0463) and `A0_OCR_E2E_RESULT.md`
   (third stream, −0.0246), **all four OCR integration points have now been measured at ≥ 30 seeds
   or with a control, and the OCR direction is closed** — not for lack of signal, but because the
   signal does not survive transfer to the LoRA-Qwen text space.
3. **The description-merge null is a cancellation, not an absence.** `TMt − TMshuf` = +0.0132 with
   the CI excluding zero on both protocols. Description content is worth 1.3 macro-F1 points at the
   text-merge injection point; the merge operation costs about the same. A cheaper injection of the
   same content is the only part of that family still open, and it needs its own pre-registration.
4. **Concat fusion fails through optimisation, not representation.** Mean selected epoch 9.2/9.5 vs
   18.1/20.1, larger seed std, worse mean. The doubled first Linear overfits earlier.

**Process note.** The margin on R1a — 0.00015 — is the sharpest possible illustration of why the
decision rule must be frozen before the numbers exist. Had the bar been written as +0.004, or the
rule keyed on the dimension-matched control alone, R1a would read REVIVED. It was not, so it does
not. The number is reported to five decimals so a future round can re-adjudicate it against a
differently-argued bar without re-running anything.

**Recorded as the standing next candidate, not run:** NCA / soft-kNN head loss at τ = 0.1
(`refine-logs/NCA_VERDICT_REVIEW.md`, ZH +0.0112 / +0.0113 on 3/3 seed signs, killed as "inside the
±0.014 noise band"). It only engages under `--contrast_mode retrieval`, so powering it means running
the deployed retrieval pipeline rather than the bare-head baseline used by all four grids here.
That is a different harness and needs its own freeze.

## 9. Cost

| item | value |
|---|---|
| API | **¥0.00** |
| GPU runs | 330 head runs (90 + 120 + 120), **0 failures**, 3 876 s total on a shared RTX 5090 (another process held ~20 GB / 97 % util throughout) |
| CPU runs | 150 OOF arm-runs (5 arms × 30 seeds), ~14 min, 8 threads |
| wall clock | 18:44 → 19:49 local, both drivers in parallel |
| test-set contact | R1a/R2 none (HALT guard armed and logged); R1b/R3/R4 final metric only, nothing selected on test |


