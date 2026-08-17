# RE-AUDIT — powered re-examination of historically under-powered small-margin verdicts

Frozen 2026-08-17, **before any candidate contrast in the seed ranges below was computed**.
Zero API cost. Local RTX 5090 (shared: another process held 20 GB / 97 % util at freeze time).

## 1. Why this exists

`idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md` and `idea-stage/IDEA_REPORT.md` §10.6 established, by
a 30-seed pre-registered variance diagnostic (`idea-stage/r6_audit/results.json`, 180 runs), that
the project's standard 3-seed / +0.005–+0.015 decision protocol **cannot resolve effects inside the
±0.01 band**: enumerated over all C(30,3) = 4060 three-seed subsets it fires GO 12.9 % of the time
on an effect whose 30-seed truth is +0.0019, and it **misses a genuinely above-bar +0.0145 effect
56.5 % of the time**. The needed seed count for MC SE ≤ 0.0025 is 7 to 71; the protocol used 3.

R6-1 was then re-run properly (60 fresh seeds, disjoint range, two independent random controls) and
came back **CONFIRMED-1DS** where the 3-seed instrument had returned KILL. That is one demonstrated
false kill. **This run asks whether there are others**, by re-measuring, at ≥30 seeds with paired
bootstrap CIs, the historical verdicts that were decided by a small number rather than by an
independent fact.

## 2. Census — every historical verdict that qualifies

Inclusion test, applied to every result document under `idea-stage/`, `refine-logs/`,
`research-wiki/` and `TARGET_FINDINGS.md`: **(a)** seeds ≤ 5 or unreported, **(b)** primary effect
magnitude in [0.000, 0.015], **(c)** verdict is not GO.

Exclusion (per the standing rules and the brief): mechanism independently falsified by a control
arm / disproved premise / inert instrument / absent capability; policy-blocked; needs lost assets;
MLLM-judgement-capability class (a capability gap does not shrink with seeds).

| # | source | mechanism | effect | seeds | verdict | killed by | in scope? |
|---|---|---|---|---|---|---|---|
| 1 | `idea-stage/OCR_FUSION_PILOT_RESULT.md` | mean-pooled OCR-30 block, frozen linear fusion | **+0.0094** OOF macro-F1 (arm2−arm0), 3/3 positive | 3 | AMBIGUOUS | **the +0.015 bar alone**; no control arm ever contradicted it | **YES — R1a / R1b** |
| 2 | `idea-stage/PILOT_C_RESULT.md` | OCR overlay/scene **provenance typing** | **+0.0044** OOF macro-F1 (arm2−arm1c), 3/3 positive | 3 | AMBIGUOUS | the +0.010 bar, **and** a control (`arm1c`) that is handicapped by weight decay; the information-matched contrast `arm2−arm1` is −0.0020 (0/3) | **YES — R2** (the brief's named first candidate) |
| 3 | `idea-stage/TEXT_MERGE_RESULT.md` | MLLM description merged into the transcript before the encoder | **−0.0105** test macro-F1 (TMt−A0), 0/3 | 3 | KILL | mostly the number; the shuffled-description control is *worse* (−0.0122), so content is worth ≈ +0.002 | **YES — R3** |
| 4 | `idea-stage/ARBITER_RESULT.md` | uncertainty-gated MLLM deferral | −0.0135 | 3 | KILL | **mechanism**: in-band the MLLM is never more accurate than the head (21/24 cells worse, 3 ties, 0 better) — the fusion ceiling is set independently of seeds | no |
| 5 | `idea-stage/R6_PILOT_RESULT_2026-08-17.md` (R6-2) | transductive pool refinement | −0.0038 / −0.0009 | 3 | AMBIGUOUS | **mechanism**: frozen operator numerically inert; corrected sweep negative on dev on all 4 datasets; no covariate shift exists | no |
| 6 | `idea-stage/LEG2_KILL_RESULT.md` | human-agreement retrieval, leg (ii) | −0.00506 | 3 | FAMILY-CLOSED | **mechanism**: shuffled-vote placebo reproduces 87 % of the gain | no |
| 7 | `idea-stage/RGCL_ABLATION_RESULT.md` | whole retrieval pipeline vs bare BCE head | −0.0017 (11-cell mean) | 3 | DECORATIVE ×2 | **mechanism**: the one ALIVE component is a fixed-threshold collapse artefact; on the un-collapsed rung the same quantity is −0.0022 | no |
| 8 | `idea-stage/R8_DECOMP_MEMO.md` | ensemble / trajectory-averaging premises | +0.0051 … +0.0001 at the oracle threshold | 3–5 | closed by measurement | **mechanism**: the ROC gain sits away from the decision boundary; conclusion does not rest on these numbers | no |
| 9 | `idea-stage/P2_FORENSIC_MEMO.md` | segment-keyed retrieval localisation | +0.005, CI [−0.024, +0.035] | deterministic | NO-GO | **mechanism**: within-video AUROC 0.511 [0.488, 0.533] — the statistic carries no localisation signal | no |
| 10 | `idea-stage/XBUCKET_RECON.md` | X-bucket error repair | −0.0104 | 3 | SEAL | **mechanism**: five independent axes fail to separate X; the 20 nearest train neighbours carry the opposite label | no |
| 11 | `idea-stage/SCRA_THEORY_MEMO.md` | safe covariate-shift rank adaptation | −0.0072 / −0.0017 | 3 | VACUOUS | **mechanism**: no covariate shift exists (domain AUC 0.42–0.56); certificate floor 19–180× the prize | no |
| 12 | `idea-stage/IDEA_REPORT.md` §8.7 (R4-1 MDL) | monotone disagreement lattice | ΔROC −0.0000, LCB95 −0.00253 | 3 | KILL 0/4 | **mechanism**: positive control recovers a planted interaction (+0.041) — complementarity is additive | no |
| 13 | `idea-stage/IDEA_REPORT.md` §8.8 (R4-2 JLR) | jackknife lower-bound rank head | ΔROC +0.0020; sd-ablation −0.0005…−0.0026 | 3 | KILL 1/4 | **mechanism**: the defining term is a drag in 4/4 against its own sd=0 ablation | no |
| 14 | `idea-stage/CLAUDE_STANCE_GATE_RESULT.md`, `LIKELIHOOD_PROBE_RESULT.md`, `CONTRAST_STANCE_RESULT.md`, `MASK_STANCE_PILOT_RESULT.md`, `PERCEPT_STANCE_RESULT.md`, `STANCE_PILOT_RESULT.md`, `VOICE_FIELD_ANALYSIS.md`, `SYNTH_PAIR_PROBE_RESULT.md`, `CN_VOTE_RECON.md` | MLLM stance / voice judgement family | various, several inside the band | 0 model seeds | FAIL / KILL / BURY | **capability class** — the judgement is degenerate (constant OPPOSE/ENDORSE), or the premise is false; seed count is irrelevant | no (brief's explicit exclusion) |
| 15 | `A0_OCR_E2E_RESULT.md` (−0.0246), `DESC_CHANNEL_RESULT.md` (−0.0371), `CAD_RESULT.md` (−0.0507), `C8_PROSODY_RESULT.md` (−0.0436), `LI_RETRIEVAL_PILOT_RESULT.md` (−0.0431), `R7_*`/`R8_*` (already 30 seeds) | — | outside the band, or already measured at 30 seeds | — | — | out of scope by (b) or by seed count | no |

### 2b. Second sweep — `refine-logs/`, `research-wiki/`, `TARGET_FINDINGS.md`

55 further verdict documents were inventoried under the same test. **Roughly 45 of the 55 are killed
by an independent mechanism-level fact**, not by a small number, and are out of scope: a control arm
matching or beating the candidate (TRA, JLR, VGA's K-VGA-3 no-verifier gate, AGGNET's `THRESH_best`
and `DIRECT_logit`, auto-repair vs random deletion); a degeneracy twin showing the operator *is* an
already-dead lever (RESTRANS 95–99 % agreement with a pure threshold shift, VSW 0.9706, AGGNET
0.996, C09 0.952–0.968, MECHFIX-T1 100 %); an analytically disproved premise (GIR is an exact linear
subset of the baseline; GRADNORM's published sign reverses on our head; KSWEEP's k ≤ 3 is
element-wise identical to 1-NN; MJ fails even with a perfect judge; ISR's β-decomposition shows
91–98 % of the ceiling is selection-locked); or absent conditional information (LAUD, CLAP, APX,
W2-A, CTF, MNTP-S1). Several more are blocked by lost assets — **no LoRA adapter weights survive
anywhere on disk**, and the floor head checkpoints 13150/13241 were pruned — which rules out
vision-unfreeze, MokA, B4-EN and cand-2 curriculum (+0.0093, 3/3, missing its bar by 0.0007).

Rows that reach this round's inclusion test:

| source | mechanism | effect | seeds | verdict | in scope? |
|---|---|---|---|---|---|
| `refine-logs/FUSIONCAT_VERDICT_REVIEW.md` | trained `concat` + MLP fusion instead of `align` (Hadamard) | **ZH +0.0067** val-sel (2/3) / −0.0045 final; HateMM −0.0031 / −0.0031 | 3 | KILL, axis closed | **YES — R4.** The record's own scope caveat forbids upgrading this null to "head capacity doesn't help"; nothing independent falsified it; `--fusion_mode concat` needs zero code change |
| `refine-logs/NCA_VERDICT_REVIEW.md` | NCA / soft-kNN head loss, τ=0.1 | ZH **+0.0112 / +0.0113**, 3/3 sign, declared "inside the ±0.014 noise band" | 3 | KILL 0/8 | **in scope but NOT run this round** — the arm only engages under `--contrast_mode retrieval` (the deployed retrieval pipeline), a different harness and baseline from the four grids below; running it here would change the question, not power it. Recorded as the standing next candidate |
| `refine-logs/HEADRECIPE_VERDICT_REVIEW.md` | SAM flat-minima optimiser on the align head | HateMM **+0.0047 / +0.0046**, not 3/3; ZH −0.0246 / −0.0424 | 3 | KILL 4/4 | no — ZH is 5–9× the bar in the wrong direction, so a two-dataset pass is arithmetically out of reach |
| `refine-logs/SWA_PROBE_RECORD.md` | single-trajectory weight averaging | dev jitter same order as the effect | 3 | KILL | no — **policy-blocked**: the record flags that it needs a user micro-ruling against the cross-seed-ensemble veto before it may enter any claims table |
| `research-wiki/EXP_p3_evidence_pooling.md` | MLLM evidence-density re-weighted pooling | HateMM probe +0.0108 → trained val −0.0041 / final +0.0004 | 3 | FAIL ×3 | no — **mechanism**: the learned align fusion absorbs input-side re-weighting; the signal was redeployed as a localisation asset instead |
| `research-wiki/EXP_p4_schema_distill.md` | archive schema-field distillation | EN −0.001 / ZH +0.008 | sub-threshold | FAIL | no — **mechanism**: the fields are decodable (AUC .62–.93) but redundant with the label already directly supervised |
| `refine-logs/READOUT_SUBMIT_RECORD.md` | intermediate-layer / one-word readout screen | ZH +0.0128, HateMM +0.0093 | $0 CPU screen | KILL | no — **superseded**: the same axis was re-run at 60 seeds as R6-1C and returned CONFIRMED-1DS; this is the false kill that is already corrected |

**Five rows are carried forward**: OCR mean-block fusion (two substrates), OCR provenance typing,
transcript description merge, and concat fusion.
Ranked by revival potential = (closeness of the historical effect to +0.005 from above) ×
(mechanism not independently killed) × (cheapness):

1. **R1** OCR mean-block fusion — +0.0094, i.e. **already 1.9× the new bar**, 3/3 positive, and the
   only row in the whole census whose kill rests on nothing but a bar that has since been retired.
2. **R2** OCR provenance typing — +0.0044, just under the bar, but its control is known-broken in
   the candidate's favour and the clean contrast is negative. Comes free: same run as R1a.
3. **R4** concat fusion — ZH +0.0067 on 2/3 seeds, HateMM flat at −0.0031 (three of its six deltas
   are *exactly* 0.0000, i.e. ≤ 2 flipped test items per seed — the definition of an unresolved
   measurement). The only carried-forward candidate that can be run on **two** datasets, hence the
   only one that could clear a two-dataset bar. Zero code change.
4. **R3** transcript description merge — −0.0105 with a *worse* shuffled control, so the number,
   not a fact, did the killing. Lowest prior; must move +0.0155 to pass.

## 3. What is run

Four grids, each a **single submission**, no re-run, no tuning after any number is seen.

### R1a + R2 — the original frozen-space harness, powered
- Script `idea-stage/reaudit/reaudit_ocr.py`, a copy of `idea-stage/pilot_c_ocr_provenance.py` with
  **one** addition (below). Design, folds, head, optimiser, inner-4-fold lockstep epoch/threshold
  selection, OCR filter and typing rule are unchanged and inherited from
  `idea-stage/OCR_FUSION_PILOT_FREEZE.md` and `idea-stage/PILOT_FREEZE_2026-08-09.md` §P-C.
- Endpoint: **out-of-fold macro-F1 over the 744 HateMM train videos**. Train-only; `dev_seen` and
  `test` are HALT-guarded at process start, as in the original.
- Arms: `0` baseline (1792-d), `1` +untyped OCR-30 (2560-d), `1c` untyped duplicated (3328-d),
  **`1r` NEW** = baseline ‖ l2norm(baseline · R) (2560-d), `2` typed overlay ‖ scene (3328-d).
- **The one addition, and why.** Arm `1r` is the dimension-matched, content-free control the 2026-08-09
  design lacked: R is one fixed Gaussian (1792 × 768), `default_rng(20260817)`, entries N(0, 1/√1792),
  no label involved. It is the RANDCAT construction of `R6_CONFIRM_FREEZE_2026-08-17.md`. Without it,
  `arm1 − arm0` cannot be separated from added head capacity, and the only capacity control on record
  (`arm1c`) is provably handicapped — under weight decay an exactly duplicated block splits its weight
  across two copies, which is why `arm1c − arm1` = −0.0064 on 0/3.
- **Seeds: 30, drawn 20260900…20260929** — disjoint from the original three (20260810–12).
- Verified before freezing: with arm `1r` added, the synthetic smoke reproduces all four original
  arms' per-seed OOF macro-F1 **bit-for-bit** (`idea-stage/reaudit/smoke_synthetic.json` vs
  `logging/runs/pilot_c/smoke_synthetic.json`).

### R1b — the same candidate on the deployed substrate, at test level
The 2026-08-09 result lives on CLIP features and a train-only OOF endpoint. R1b asks the
decision-relevant version of the same question on the current baseline, with the R6 read-out.
- Builder `idea-stage/reaudit/build_ocr_arms.py` → `data/CLIP_Embedding/HateMM/*_RAOC-*.pt`,
  meta + SHA-256 in `idea-stage/reaudit/build_ocr_meta.json`.
- Arms: `A0` = verbatim deployed cache (`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`);
  `OCR` = text stream ← [l2norm(text₃₅₈₄) ‖ l2norm(ocr₇₆₈)];
  `RAND` = text stream ← [l2norm(text₃₅₈₄) ‖ l2norm(text₃₅₈₄·R)], R one fixed Gaussian
  (3584 × 768), `default_rng(20260817)`, sha `5fd429bd66587d1cdeb53ac9790503edd325ca0156555d965689d3db04b09d34`.
  Image stream, ids and labels carried through unchanged in all three arms.
- OCR vector = the same `rac_ocrmean30_*` block as `A0_OCR_E2E_RESULT.md`, whose test split is the
  **real** 215-row file (the placeholder used by that run is preserved separately). Injection is
  **concatenation into the text stream**, not the learned third stream that run measured.
- Runner `idea-stage/reaudit/run_grid.sh`, hyperparameters byte-identical to
  `idea-stage/r6_confirm/run_confirm.sh`. **Seeds 300–329** (30), disjoint from every prior range
  (0–29 audit, 30–89 confirm, 100–129 R7, 200–229 R8).
- Instrument check already run and passed: `RAOC-A0` seed 30 reproduces
  `logging/runs/r6_confirm/logs/HateMM_A0_s30.trainlog` **bit-for-bit** on all 60 dev/test epoch
  lines (`RAOC-A0` is byte-identical to `R6RO-A0`), at 12 s per run.

### R3 — transcript description merge, powered
- Arms `A0`, `TMt`, `TMall`, `TMshuf` from the existing
  `data/CLIP_Embedding/HateMM/*_TEXTMERGE-*.pt` caches; nothing rebuilt.
- Same runner, same hyperparameters (`idea-stage/text_merge/run_arms.sh` is already byte-identical
  to it apart from `--keep_epoch_ckpts`, which no read-out uses). **Seeds 300–329.**

### R4 — concat fusion, powered
- Arms on the **deployed A0 cache** (`RAOC-A0`, byte-identical to `R6RO-A0`), two datasets,
  **HateMM and MHC_zh**: `ALIGN` = `--fusion_mode align` (the deployed baseline, identical to every
  other grid here) and `CAT` = `--fusion_mode concat`. Everything else — features, optimiser, seeds,
  epochs, warmup, read-out — is unchanged. No code change: `--fusion_mode concat` is an existing,
  already-implemented branch of `src/run_rac.py` / `src/model/classifier.py`.
- MHC_zh uses the arm name `R6RO-A0` (its own deployed cache, `-LoRA_HF-ro_L28`); HateMM uses
  `RAOC-A0`. The two are the same construction on their respective datasets.
- **Seeds 300–329.**
- **Declared confound, not patched.** `concat` feeds a 2× wider first Linear (2 098 176 vs
  1 049 600 parameters), so `CAT − ALIGN` bundles the fusion operator with head capacity. That is
  exactly the object the 2026-08-13 record measured, and the record's binding scope note forbids
  reading the null as a statement about capacity alone. Adding a capacity control would test a
  different quantity than the one being re-audited, so none is added; the confound is carried into
  the reading instead.

## 4. Read-out protocols

For R1b and R3, both computed from the **same** runs by `idea-stage/reaudit/analyze_grid.py`, which
imports `parse` / `select_epoch` verbatim from `idea-stage/r6_audit/analyze_audit.py`:
- **P1 (primary)** — epoch = argmax over e ≥ warmup 5 of **dev macro-F1**, ties to the earliest;
  test macro-F1 at threshold 0.5.
- **P2 (corroboration)** — epoch 29 (last of 30); test macro-F1 at threshold 0.5.

Test labels are read only for the final metric. No threshold, no epoch rule and no arm definition is
selected on them.

For R1a/R2 the endpoint is out-of-fold train macro-F1 under the original harness's own inner-4-fold
lockstep epoch+threshold selection; **P1/P2 do not apply and are not reported** — declared here as a
deviation rather than patched, because changing that harness's selection rule would break the
bit-for-bit comparability with the 2026-08-09 numbers that is the entire point of the re-audit.

All contrasts are **seed-paired**, with a **paired bootstrap 95 % CI over seeds**, 20 000 resamples,
`default_rng(20260817)`.

## 5. Decision rule — frozen, applied verbatim

A candidate is **REVIVED** iff **every** contrast listed for it has, on its primary protocol,
`mean ≥ +0.005` **and** its paired-bootstrap 95 % CI **excluding zero**; and, where P2 exists, P2
agrees in sign on each. Anything else is **NOT REVIVED**.

| candidate | contrasts that must all pass | protocol |
|---|---|---|
| **R1a** OCR mean-block fusion, frozen space | `arm1 − arm0` **and** `arm1 − arm1r` | OOF |
| **R1b** OCR mean-block fusion, deployed substrate | `OCR − A0` **and** `OCR − RAND` | P1, P2 sign |
| **R2** OCR provenance typing | `arm2 − arm1c` **and** `arm2 − arm1` | OOF |
| **R3** transcript description merge | `TMt − A0` **and** `TMt − TMshuf` | P1, P2 sign |
| **R4** concat fusion | `CAT − ALIGN` on ≥ 1 dataset, and `≥ −0.002` on the other | P1, P2 sign |

The second contrast in each row is the control that the historical design either lacked (R1) or
whose failure was reported but non-gating (R2, R3). Requiring both is the R6-1C two-condition rule
(`CAT − A0` **and** `CAT − RAND`), transferred unchanged.

**Cross-dataset clause.** R1b and R3 are HateMM-only because HateMM is the only dataset with OCR
coverage on train+dev+test (`data/OCR/MHC_test` and `data/OCR/MHC_zh_test` hold test splits only)
and the only dataset with TEXTMERGE caches. A single-dataset pass is therefore reported as
**REVIVED (1 dataset)** and explicitly does **not** license a method claim; per
`R6_CONFIRM_FREEZE_2026-08-17.md` §"What a pass would and would not mean", a component-level gain on
one dataset is a candidate, not a contribution.

**VOID clause.** If any grid returns a failed run, or if `analyze_grid.py` drops a run on
confusion-matrix reconstruction mismatch, the affected grid is VOID and reported as such, not passed.

## 6. Expected outcome, stated before the run

- **R1a: passes `arm1 − arm0`, fails or barely passes `arm1 − arm1r`.** The historical +0.0094 is
  1.9× the bar with 3/3 positive, so the first contrast should clear. But `arm1c − arm0` = +0.0030
  says roughly a third of it is capacity, and `1r` is a fairer capacity control than `1c`, so
  `arm1 − arm1r` is expected around +0.005 — right on the line. Overall: **coin-flip**.
- **R2: NOT REVIVED.** `arm2 − arm1` = −0.0020 on 0/3 is a clean, information-matched negative; more
  seeds should sharpen it, not flip it.
- **R1b: NOT REVIVED.** Mixing a 768-d CLIP-text OCR block into a 3584-d Qwen text stream is a
  worse-matched injection than the frozen-space version, and `A0_OCR_E2E_RESULT.md` found the same
  vector costs −0.0246 through the third-stream path.
- **R3: NOT REVIVED.** −0.0105 must move +0.0155 to pass; that is ~3σ of the historical paired std.
- **R4: NOT REVIVED, but the least confident of the five.** ZH's +0.0067 was 2/3 and its final-epoch
  read was −0.0045, i.e. the two protocols disagreed in sign at 3 seeds — the exact signature the
  power audit says 3 seeds cannot resolve. HateMM was flat. A pass on ZH alone with HateMM ≥ −0.002
  is a live possibility.

Recording these makes it visible if the reported outcome merely echoes the prior.

## 7. Red lines

1. **Zero test-set contact for R1a/R2** (HALT guard armed and logged, as in the original); for R1b
   and R3 the test split is read only for the final metric, under the user's 2026-08-09 ruling, and
   nothing is selected on it.
2. **This document is committed to git before any candidate contrast in the new seed ranges is
   computed.**
3. **Blind implementation** — the analyzers were written and the smokes run before any real
   candidate number in the new ranges existed.
4. **One submission per candidate.** Three grids, one launch each, no re-runs.

## 8. Cost

Measured 12 s per head run. R1b 3 × 30 = 90 runs ≈ 18 min; R3 4 × 30 = 120 runs ≈ 24 min;
R4 2 arms × 2 datasets × 30 = 120 runs ≈ 24 min; R1a/R2 5 arms × 30 seeds on CPU ≈ 20 min.
**¥0 API. GPU is shared with an unrelated 20 GB process.**
