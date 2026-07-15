# B5 — G-repro Gate Amendment Ruling (post-halt, independent-review adjudication)

**Adjudicator:** fresh zero-context independent pre-registration amendment reviewer (did not design or run
B5). Read-only forensic review; NO GPU, NO SLURM, NO submission. Zero interaction.
**Date:** 2026-07-15.
**Escalation:** B5 probe HALTED at its G-repro gate (roc-4dp mismatch on Qwen slots, both the CPU replay
job 13156 and the one authorized cuda fallback job 13158). Executor escalated a gate-amendment question
above their authority. This document renders the BINDING ruling.
**Files read (primary, verified against raw logs — not the executor's summary):**
`refine-logs/B5_PROBE_RECORD.md`, `refine-logs/B5_PROBE_DESIGN.md`,
`research-wiki/experiments/exp-conv-zh-b5.md`, `refine-logs/B5_PREREG_REVIEW.md`,
`slurm/logs/b5probe_13156.out` (CPU), `slurm/logs/b5probeC_13158.out` (cuda),
`scripts/analysis/b5_conv_probe.py` (on-disk v3), git HEAD `b5_conv_probe.py` (v1), `sacct -j 13158`.

## RULING (one line)

**AMEND-APPROVED.** The pre-registered G-repro clause is amended, via REPLACE-in-place, to:
**acc AND macroF1 exact at 4 dp (test + dev, all 12 slots) AND roc within |Δ| ≤ 1e-3 of anchor.** The
existing job-13158 cuda evidence **satisfies the amended gate as-is** (acc/mF1 12/12 exact; every roc
|Δ| ≤ 0.0007 < 1e-3), so **G-repro = PASS on existing evidence, with NO new GPU/cuda run.** The probe
continues to (b)–(e) under ONE authorized zero-GPU (CPU) submission and the unchanged strict order.

---

## 0. The original clause (quoted verbatim, both loci)

**`refine-logs/B5_PROBE_DESIGN.md` §4** (`:238-243`):

> **G-repro gate (test AND dev; amendment A2):** for all 12 (6 arms × 2 protocols), the probe's
> `deployed_test_acc / deployed_test_mf1 / deployed_test_roc` MUST equal the **test** anchor above to
> 4 dp, **AND** the probe's recomputed **dev** deployed `acc / macroF1 / roc` at each loaded checkpoint
> MUST equal the **DEV** anchor above to 4 dp. Mismatch (test or dev) on CPU ⇒ retry via the §6 GPU
> fallback (bit-exact device match); mismatch on GPU too ⇒ **HALT**, replay machinery invalid, probe
> does not proceed (no calibrated number is trustworthy without this — REFLECTION §4 probe-validity
> mandate).

**`research-wiki/experiments/exp-conv-zh-b5.md` §6.3** (`:232-247`):

> The deployed-arm test acc AND macro-F1 AND roc recomputed from the dumped votes MUST reproduce the
> 13115 banked readings **to 4 decimal places** for all 6 arms × both protocols … Any mismatch ⇒
> **HALT** … **DEV-side anchor (amendment A2, BLOCKING).** … the probe's recomputed **dev** deployed
> acc AND macroF1 AND roc at each loaded checkpoint MUST also match the corresponding trainlog
> `Val_Retrieval Epoch NN` line **to 4 dp** … **Both the test AND dev anchors must pass** for the probe
> to proceed.

The load-bearing defect: this clause demands `roc` reproduce **to 4 dp**, on the same footing as acc/mF1.

---

## 1. Evidence (transcription independently re-verified against the raw logs)

Both device runs: **acc AND macroF1 match the 13115 anchors 12/12 exactly at 4 dp, on BOTH test and
dev.** In both logs the only entries ever flagged `MISMATCH` are `test_roc`/`dev_roc` on Qwen slots.
Every acc/mF1 cell in the executor's `B5_PROBE_RECORD.md` tables was checked against the primary log
lines and against the anchor tables (`B5_PROBE_DESIGN.md` §4 / `B5_PREREG_REVIEW.md` §1.1, §2 Item-5) —
all correct. Splits confirmed in both logs: dev n=78 (28 pos), test n=149 (45 pos); pairing check
(`CLIP vs Qwen ids+labels identical per seed/proto`) prints OK in both.

**roc deltas (recomputed − anchor), the only non-reproducing quantity:**

| slot | CPU (13156) roc Δ | cuda (13158) roc Δ |
|---|---|---|
| Qwen s0 valsel | test +0.0002 | test −0.0005, dev +0.0007 |
| Qwen s1 final | test −0.0002 | dev +0.0007 |
| Qwen s1 valsel | dev −0.0007 | test +0.0004 |
| Qwen s2 final | dev +0.0007 | dev +0.0007 |
| Qwen s2 valsel | test −0.0002 | **exact (PASS)** |
| all 6 CLIP + Qwen s0 final | exact | exact |
| **max \|Δ\|** | **0.0007** | **0.0007** |

CPU fails 5 Qwen slots; cuda fails 4 — a **different** failing-slot set, and even within a shared
slot the failing *metric* differs (e.g. Qwen s1 final: CPU misses on `test_roc`, cuda misses on
`dev_roc`; Qwen s2 valsel: CPU FAIL, cuda PASS). cuda evidence confirmed genuinely on-device:
`sacct -j 13158` AllocTRES `gres/gpu=1`; log header `device=cuda` + `NVIDIA A100-SXM4-80GB`.

---

## A. Is the executor's unsatisfiability argument SOUND? — YES

**A.1 The pattern confirms rank-instability-near-ties, not an implementation bug.**
- **acc/mF1 12/12 exact on two independent compute paths (CPU and cuda)** ⇒ the deployed operating
  point — the vote *sign* at cut = 0 — is fully reproduced everywhere. A genuine implementation fault
  (wrong split / wrong checkpoint / label mismatch) would move acc and mF1, and by a **coarse** amount:
  the acc granularity is 1/149 = 0.0067 (test) and 1/78 = 0.0128 (dev), i.e. ≥ 30× the observed roc
  drift. Zero acc/mF1 movement across both devices **falsifies** all three "wrong-artifact" hypotheses.
  Wrong split is separately excluded (n=78/149, pos=28/45 printed identically in both logs, matching the
  design's verified balances); label/pairing mismatch is separately excluded by the in-run pairing
  assertion printing OK in both logs.
- **roc is the sole residual, it is Qwen-only, and it is rank-quantised.** roc =
  `roc_auc_score(labels, continuous_vote)` (`src/utils/metrics.py:294`, confirmed by the prereg review
  §1.5) is a rank statistic. A single near-tied pos/neg pair swapping in the AUC ordering moves roc by
  exactly one swap-granularity: **1/(45·104) = 0.0002 (test), 1/(28·50) = 0.0007 (dev).** Every observed
  delta is an exact integer multiple of that granularity (test 0.0002 / 0.0004 / 0.0005 = 1/2/2–3 swaps;
  dev 0.0007 = precisely one swap). Qwen-only is mechanistically expected: the 3584-dim Qwen head matmul
  accumulates more float-eps than CLIP's 1024/768, so it is the only head that puts pairs close enough to
  the tie boundary to flip.
- **Device-dependent, non-reproducible failing sets** (CPU ≠ cuda, and metric-within-slot differs) are
  the signature of stochastic float-eps compute-path noise, not a deterministic code error (a bug would
  reproduce the *same* mismatch on both devices and would perturb acc/mF1).

**A.2 The anchor itself is a non-redrawable draw.** 13115 trained with `device='cuda'` and set no
deterministic mode; the anchor roc came from one non-deterministic cuBLAS-kernel draw. A fresh eval
forward — even byte-correct on the same node — can select a different kernel and round differently at
float-eps for the rank statistic. Hence **roc-to-4dp is unsatisfiable by any replay**, including a
perfect one. The prereg's premise that the cuda fallback is "bit-exact to 13115" holds for acc/mF1 (the
deployed sign) but is **false for the rank statistic roc** — a pre-registration specification fact
surfaced by execution, not a machinery defect.

*One honesty caveat, not disqualifying:* the "even cuda-vs-cuda is non-reproducible" leg is an inference
(13115's training-forward was not re-run twice to demonstrate the anchor's own non-determinism directly).
It is the standard, well-founded understanding of cuBLAS non-determinism and is corroborated
circumstantially by the CPU-vs-cuda different-failing-set evidence and the exact swap-granularity match.
For a diagnosis line with the single cuda spend consumed, that circumstantial chain is more than
sufficient; a direct re-draw demonstration is disproportionate and not required.

**Verdict on A:** the unsatisfiability argument is **sound**. No alternative (wrong split / checkpoint /
labels / pairing) survives the acc-mF1-exact-on-both-devices test.

---

## B. RULING — AMEND (option 1)

The G-repro clause is **AMENDED via REPLACE-in-place**. Amendment id **A11** (post-halt,
independent-review adjudicated).

### Amended clause (verbatim — to REPLACE the §4 and §6.3 gate text quoted in §0)

> **G-repro gate (test AND dev; amendment A2 + roc-tolerance amendment A11).** For all 12 (6 arms × 2
> protocols): the probe's recomputed **test** deployed `acc` AND `macroF1` MUST equal the **test** anchor
> to 4 dp (exact), AND the recomputed **dev** deployed `acc` AND `macroF1` at each loaded checkpoint MUST
> equal the **DEV** anchor to 4 dp (exact); AND the recomputed **test** and **dev** deployed `roc` MUST
> each lie within **|Δ| ≤ 1e-3** of the corresponding anchor. **Rationale (A11):** the anchor `roc` is a
> rank statistic produced by a non-deterministic cuBLAS kernel in the 13115 training forward (no
> deterministic mode set), so `roc`-to-4dp is unsatisfiable by *any* replay — even a byte-correct,
> same-hardware one — whereas `acc` and `macroF1` (the deployed vote-sign operating point, and the only
> quantities the calibration consumes) reproduce exactly. Failure of `acc` or `macroF1` (test or dev) at
> 4 dp, OR a `roc` |Δ| > 1e-3, on CPU ⇒ retry via the §6 GPU fallback; the same failure on GPU too ⇒
> **HALT**, replay machinery invalid, probe does not proceed. The 1e-3 bound is ~1.4× the maximum drift
> either device produced (0.0007) and ≈ 5 dev-swaps / 14 test-swaps — tight enough that any *systematic*
> divergence (which would move acc/mF1 first, and roc by far more) still trips it.

### Why this is evidence-driven, NOT outcome-driven (the moral-hazard test)

Post-hoc gate relaxation after a failing run is exactly what preregistration guards against, so this
must be met head-on. It is answered on four independent grounds:

1. **The passing quantities passed BEFORE any amendment was proposed, and they are the ones the probe
   consumes.** acc and macroF1 — the deployed vote-sign, the only inputs the calibration (b)–(e) uses —
   reproduced 12/12 exactly at 4 dp on two devices. The amendment rescues no failing *result*; the
   result the probe exists to produce (b)–(e) is not yet computed, and the gate that protects it (the
   vote signs) already passed clean and unamended.
2. **roc is provably unused downstream — code-verified.** In `scripts/analysis/b5_conv_probe.py`, `roc`
   is referenced *only* inside the (a) gate block (lines 284, 287, 290–291) and in comments. Threshold
   selection (`select_tau`→`mf1_at`), the oracle (`oracle_max` on `acc_at`/`mf1_at`), the honest preview
   (`acc_at`/`mf1_at`), and the D3 bootstrap all operate on the dumped votes via acc/mF1 and **never
   touch roc**. Therefore relaxing the roc clause **cannot change any (b)–(e) verdict** — it can only
   release the strict-order HALT. An outcome-driven relaxation is one that moves a decision; this one is
   mathematically incapable of moving the probe's decision.
3. **The relaxation is principled and bounded to the demonstrated mechanism**, not tuned to scrape a
   pass. The 1e-3 bound is set from the physics of the rank statistic (minimal swap granularity
   0.0002 test / 0.0007 dev), independent of how many slots happen to sit inside it; both devices land at
   max 0.0007 with margin to spare.
4. **The original discipline's purpose is fully served without roc.** The G-repro gate exists to catch
   (i) transcription fabrication (the 0.8732 incident) and (ii) silent code drift. Both are carried
   entirely by the exact-4dp acc/mF1 clause: acc/mF1 pin every vote sign to 4 dp, and drift/fabrication
   would surface there at the 0.0067–0.0128 granularity long before a 1e-3 roc wobble. The discipline
   was never intended to demand reproduction of a metric the original pipeline itself cannot redraw
   deterministically; A11 corrects that specification defect, it does not lower the bar on anything the
   conclusion rests on.

The competing option (UPHOLD) is rejected: it would kill a decision-useful diagnosis on a technicality
(a gate no correct implementation can pass) while **buying zero integrity** — the fraud/drift guard is
already discharged by the acc/mF1 clause. B5 is a diagnosis / performance-clause line, explicitly
**not** novelty-bearing (novelty is a deferred D7-class user ruling, prereg §9), so the cost of a false
kill is a lost measurement for no evidentiary gain. AMEND is the evidence-driven ruling.

---

## C. Does the existing 13158 cuda evidence satisfy the amended gate? — YES

| check | requirement (amended) | 13158 result | pass |
|---|---|---|---|
| test acc, test mF1 (12 slots) | exact 4 dp | 12/12 exact | ✔ |
| dev acc, dev mF1 (12 slots) | exact 4 dp | 12/12 exact | ✔ |
| test roc, dev roc (12 slots) | \|Δ\| ≤ 1e-3 | max \|Δ\| = 0.0007 | ✔ |

**G-repro = PASS on existing evidence. No new GPU/cuda run is authorized or needed for the gate.** (The
already-consumed cuda spend stands; the CPU run 13156 also independently satisfies the amended gate —
its max roc |Δ| is likewise 0.0007 — so the PASS is not device-privileged.)

### Continuation path for (b)–(e)

- **Submissions authorized: ONE.** A single **zero-GPU (CPU) SLURM submission** of the frozen probe to
  compute (b)–(e). Device = **CPU, per the original zero-GPU design** (the cuda budget is consumed and
  CPU also clears the amended gate). No GPU, no second cuda.
- **Prerequisite code edit (implements A11 in the script).** The frozen script hard-codes a single
  `abs(got - exp) < 5e-5` tolerance for **all six** checks including roc (line 293), so a re-run as-is
  would still HALT. The executor MUST widen **only the two roc checks** to the |Δ| ≤ 1e-3 tolerance while
  leaving the four acc/mF1 checks at exact-4dp (5e-5). This is the in-code realisation of the amended
  gate: it must be a REPLACE-in-place, diff-verified edit touching only the gate tolerance for roc — **no
  change to the vote computation, threshold selection, oracle, D3, or the strict order / `sys.exit(2)`
  HALT** — and re-frozen with a new sha256 logged (rides the next B5 commit, §D). Equivalently the
  reviewer's PASS in §C may be taken as the external certification and (b)–(e) computed from the
  already-dumped votes; either way, only ONE CPU submission is authorized and the calibration arithmetic
  is unchanged.
- **Unchanged bindings.** (b)–(e) remain under (i) the **original strict order** (freeze dev-τ → oracle
  kill-switch → honest preview → D3), (ii) the executor applies **NO pass/fail interpretation**, and
  (iii) **independent verdict processing** — including the mandatory A7 hand-recomputation of one honest
  cell from the dumped `votes_*`/`labels_*`, together with the A2 dev anchor — validates the calibration
  machine **before** any formal-stage authorization. The A1 per-protocol oracle kill-switch and the §6.5
  honest-preview gate continue to govern whether any formal GPU is ever spent. **This ruling unblocks the
  HALT only; it authorizes no formal stage.**

---

## D. Sha / hygiene — banner fix confirmed print-only; version chain independently re-derived

The probe script is a three-version chain. I re-derived every hash from first principles (git HEAD for
v1; reconstruction for v2; on-disk for v3) rather than trusting the record:

| ver | sha256 | provenance | verified how |
|---|---|---|---|
| **v1** | `57a774da55b128067d014293347e858de18f1c799cccb8293636350d8bcd02f9` | CPU-only; ran by job **13156**; currently committed at HEAD | `git show HEAD:… \| sha256sum` == this |
| **v2** | `bfa644b20b7738eeb48229dd795e516c250503ae3a0b781b978d36f72432b0ad` | v1 + device-parametric; ran by job **13158** (the cuda evidence adjudicated here) | reconstructed from v3 by reverting the cosmetic HALT block → sha matches record §6 exactly |
| **v3** | `7c88aa03d1241ef50dc29f2d7ae71ad2e7e8654489adc475ecf07b8d80217460` | v2 + cosmetic device-accurate HALT banner; **on-disk, run by no job** | `sha256sum` of working-tree file |

- **v1 → v2 (this one RAN as 13158 — checked for logic contamination):** diff is exactly three changes —
  `make_args(device=None)` reading `B5_PROBE_DEVICE` env; `model_obj.to(args.device)` after `.eval()`;
  the device-aware header `print`. **No change** to the vote (`compute_metrics_retrieval`, use_sim), the
  gate arithmetic, `select_tau`/`grid`/`lower_median_idx`, `oracle_max`, the honest arm, the D3 bootstrap,
  or the strict order. The only functional delta is the head-forward device — exactly the intended
  fallback. **The 13158 evidence is therefore uncontaminated by any logic drift.**
- **v2 → v3 (the "cosmetic banner fix"):** diff is confined to the `if not all_pass:` HALT block — the
  hardcoded "FAILED ON CPU" string becomes device-accurate (`device={dev}`) plus a device-conditioned
  next-step print; the pre-existing `sys.exit(2)` (strict-order HALT) is preserved. **Print-only,
  confirmed** — no logic/threshold/order change. (Minor note: the change is a small if/else print block,
  slightly more than the record's "two print() lines," but materially print-only.)
- **old → new sha256 for the banner fix:** `bfa644b2…` (v2, ran by 13158) → `7c88aa03…` (v3, on-disk).
- **Hygiene flag (must be corrected at re-freeze):** HEAD (commit archiving the B5 batch) committed **v1**
  (the pre-fallback CPU-only script), NOT the v2 that produced the cuda evidence nor the v3 on disk. The
  archived script thus does not itself contain the device plumbing 13158 used. **Required at the next B5
  commit:** commit the actual-run version (v2) and the current (v3), plus the forthcoming A11
  gate-tolerance-amended version, with the full sha chain and a version→job map, so the archived probe
  matches the evidence. **This re-freeze table update must ride the next B5 commit.**

---

## E. Scope guard

**This amendment (A11) applies to the B5 G-repro clause ONLY.** It is **not** a precedent and does
**not** auto-extend to any other line's G-repro gate. The exact-4dp discipline (including for roc where
it is deterministically reproducible) remains the default everywhere else. Any future line that hits an
analogous non-redrawable-metric wall must obtain its **own** independent adjudication on its **own**
evidence — demonstrating, as was demonstrated here, that (i) the quantities the analysis consumes
reproduce exactly, (ii) the drifting quantity is unused downstream, and (iii) the drift is intrinsic
non-determinism rather than a bug. No blanket "roc gets a tolerance" rule is created.

---

## Required actions (document trail — mandatory before continuation)

Carried out under this ruling's authority (this file is the adjudication of record; per my scope I edit
no other file — the mechanical edits below are the executor's to apply):

1. **REPLACE-in-place** the §0 gate text in `refine-logs/B5_PROBE_DESIGN.md` §4 and
   `research-wiki/experiments/exp-conv-zh-b5.md` §6.3 with the **amended clause** (§B verbatim).
2. Add **A11** to the amendment log `research-wiki/experiments/exp-conv-zh-b5.md` §16, marked
   **POST-HALT amendment adjudicated by independent review (2026-07-15)**, pointer to this file.
3. Apply the §C prerequisite roc-tolerance edit to `scripts/analysis/b5_conv_probe.py`, re-freeze
   (new sha256), and update the version→job sha table (§D) — all riding the next B5 commit.
4. Run the ONE authorized zero-GPU CPU submission; proceed through (b)–(e) under the strict order; hand
   the raw numbers to independent verdict processing (A7 + A2 checks) — no formal stage authorized here.

---

## §re-check — verification of the executor's A11 application (commit a08deed) — 2026-07-15

**Scope:** v3 (`7c88aa03`) → v4 only; v1→v2→v3 already cleared in §D. All hashes and diffs re-derived
independently (isolated `diff` of my §D-verified v3 copy against `git show a08deed:…`), not taken from
the executor's report. Commit lineage confirmed: a08deed's parent is this ruling's commit `5295076`;
`--stat` confirms exactly the 5 declared files, nothing else touched.

**1. Script v3 → v4 (`scripts/analysis/b5_conv_probe.py`, committed == on-disk == `3d075345c0425d5ef0a19c87267c6178828c9e72b709798154f370f04147cdb0`): CONFIRMED gate-tolerance-only.**
The isolated v3→v4 diff contains exactly three hunks, all inside the §(a) gate block:
(i) the §(a) header `print` label (cosmetic); (ii) two comment lines + the tolerance branch
`ok = all(abs(got - exp) <= (1e-3 if k.endswith("_roc") else 5e-5) for k, (got, exp) in checks.items())`;
(iii) the matching `mism` line with `> (1e-3 if k.endswith("_roc") else 5e-5)`. The `checks` dict is
unchanged with exactly six keys — `test_mf1, test_acc, test_roc, dev_mf1, dev_acc, dev_roc` — so
`endswith("_roc")` selects exactly {`test_roc`, `dev_roc`}; the four acc/mF1 checks stay at 5e-5.
No touch to the vote path, `select_tau`/`grid`/`lower_median_idx`, `oracle_max`, honest arm, D3
bootstrap, strict order, or `sys.exit(2)`. *Minor note (non-blocking):* the acc/mF1 comparator changed
`< 5e-5` → `<= 5e-5`; behaviorally identical since 4dp-rounded deltas are either 0.0 or ≥ ~1e-4
(verified numerically) — exact-4dp semantics preserved.

**2. Doc REPLACE-in-place: CONFIRMED.** The old "roc to 4 dp" gate text is absent from both
`B5_PROBE_DESIGN.md` §4 and `exp-conv-zh-b5.md` §6.3; both now carry the §B amended clause **verbatim**,
each followed by an italic A11 provenance note (POST-HALT, ruling pointer + commit 5295076, 13158-PASS
statement). `exp-conv-zh-b5.md` §16 has the r2 revision row and an A11 amendment-table row, both marked
POST-HALT with the ruling pointer. `B5_PROBE_RECORD.md` §6 now carries the full v1..v4 sha chain with
the version→job map (v1→13156, v2→13158, v3 no job, v4 = pending CPU run) — every hash matches my
independently derived values in §D, closing the §D hygiene flag (the actual-run v2 and successors are
now archived in-tree at a08deed).

**3. cuda sbatch in the commit: record-keeping no-op, CONFIRMED.** `b5_conv_probe_cuda.sbatch` was the
previously-untracked file that ran job 13158; the committed content's sha256
(`65d1dd05984899a03ad5058a8a4081b77d0c4a81e60fe8dfb4fd6bd98df92a87`) matches both the on-disk file and
the hash frozen in `B5_PROBE_RECORD.md` §6 — i.e. the commit archives the exact artifact of the consumed
cuda spend. Committing it submits nothing and enables no new cuda path; the single-cuda-spend-CONSUMED
status and the ONE-CPU-submission authorization are unchanged.

**Re-check verdict: `CLEARED-FOR-CPU-CONTINUATION`.** The ONE authorized zero-GPU CPU submission of v4
(`3d075345…`) may proceed through (b)–(e) under the unchanged strict order, executor
no-interpretation rule, and independent verdict processing (A7 hand-check + A2 dev anchor). No formal
stage is authorized by this re-check.
