# R12-ANCHOR result — reference-correctness gating does not change which items break

Frozen design: `idea-stage/R12_FREEZE.md` §3, committed at **`a9cd557`** before any pilot code
existed. Deviation: `idea-stage/R12_DEVIATION_D1.md` (demotion-clause key lookup; cannot change
either verdict, and the corrected re-run confirms it did not). Single submission
`idea-stage/r12_anchor/run_all.sh`.

**Cost: ¥0.00 (no API). 315 head-training runs (210 MHC-ZH + 105 HateMM), 0 failures, 41 min wall
on the local RTX 5090 shared with a concurrent extraction. No new extraction and no new teacher —
the R11-UNION out-of-fold teachers are reused unchanged. Zero test-label tuning: λ, α, β, the
correctness masks, the shuffled mask and every arm definition are fixed by the freeze or computed
from train only.**

---

## Headline

1. **Frozen verdict: KILL, both candidates.** `AF_PT − CAT` = **−0.0003** (CI [−0.0048, +0.0041])
   on MHC-ZH and **−0.0002** (CI [−0.0038, +0.0034]) on HateMM. `AF_A0 − CAT` = −0.0007 / −0.0003.
   Clause 1 fails on both datasets for both candidates; the mechanism clauses are never reached as
   a gate, and are reported below because they are the informative part.
2. **The focal gate is not neutral — on HateMM it is actively worse than uniform anchoring.**
   `AF_PT − AU_PT` = +0.0014 (CI straddling zero) on MHC-ZH and **−0.0044 [−0.0090, −0.0006]** on
   HateMM. Up-weighting the anchor on the items the reference gets right *costs* accuracy on the
   dataset where the reference is accurate.
3. **The correctness semantics contribute nothing over an arbitrary class-matched bin.**
   `AF_PT − AF_SHUF` = **+0.0008** / **−0.0009**, both CIs straddling zero. Permuting the
   correctness mask within class — preserving prevalence and per-class rate exactly — changes
   nothing. Even if a gain had appeared, it could not have been attributed to reference correctness.
4. **The decisive secondary number: none of these arms changes which items break.** Error-set
   Jaccard against `CAT`, seed-paired at each arm's own P1 epoch, is **0.84-0.91 on MHC-ZH** and
   **0.91-0.96 on HateMM** — 10-15× the independence null. For comparison, `CAT` vs `LL` — two
   genuinely different read-outs — sits at 0.605 / 0.744 (`R10_COMBO_RESULT.md` §3). A feature
   change moves the error set three to four times as far as any of these objective changes.
5. **The best anchor arm is the uniform one, and the hard-label anchor is again competitive.**
   `AU_PT − CAT` = **+0.0041 [+0.0008, +0.0080]** on HateMM (CI excluding zero) but −0.0017 on
   MHC-ZH; `AU_A0 − CAT` = +0.0019 / +0.0030; `LBL − CAT` = +0.0021 / +0.0007. Nothing is 2/2, so
   nothing is an entry, but the ordering reproduces R11's finding that soft teacher knowledge is
   not worth more than the labels.
6. **`CAT` replicates a fourth time**, on a fourth disjoint seed range: 0.8180 ± 0.0100 (MHC-ZH,
   seeds 900-929) and 0.8774 ± 0.0083 (HateMM, seeds 900-914), against R11's 0.8189 / 0.8783 on
   seeds 700-729.

---

## 1. The table

P1 = test macro-F1 at `argmax_{e≥5}` dev macro-F1. MHC-ZH 30 seeds (900-929), HateMM 15 seeds
(900-914), both fresh ranges. λ = 0.1 in every anchored arm, frozen, **not** dev-selected.
Every arm uses the identical `R10CB-CAT` feature cache; only the loss differs.

| arm | teacher | weighting | MHC-ZH P1 | HateMM P1 |
|---|---|---|---|---|
| **CAT** | — (λ=0) | — | 0.8180 ± 0.0100 | 0.8774 ± 0.0083 |
| **AU_A0** | A0 | uniform | 0.8199 ± 0.0097 | 0.8805 ± 0.0080 |
| **AF_A0** | A0 | focal | 0.8173 ± 0.0101 | 0.8771 ± 0.0088 |
| **AU_PT** | pseudo-teacher | uniform | 0.8162 ± 0.0113 | **0.8815 ± 0.0077** |
| **AF_PT** | pseudo-teacher | focal | 0.8176 ± 0.0097 | 0.8772 ± 0.0075 |
| **AF_SHUF** | pseudo-teacher | focal, shuffled mask | 0.8168 ± 0.0099 | 0.8781 ± 0.0083 |
| **LBL** | hard labels | uniform | **0.8201 ± 0.0091** | 0.8781 ± 0.0065 |

### 1.1 The judgement contrasts (frozen list)

Paired mean ± paired-bootstrap 95 % CI, B = 20000, bootstrap seed 20260817.

| contrast | MHC-ZH P1 | HateMM P1 | clause |
|---|---|---|---|
| **AF_PT − CAT** | **−0.0003 [−0.0048, +0.0041]** 11/30 | **−0.0002 [−0.0038, +0.0034]** 6/15 | 1 — **fails** |
| **AF_PT − AU_PT** | +0.0014 [−0.0017, +0.0044] 10/30 | **−0.0044 [−0.0090, −0.0006]** 2/15 | 2 — **fails** |
| **AF_PT − AF_SHUF** | +0.0008 [−0.0018, +0.0034] 9/30 | −0.0009 [−0.0053, +0.0025] 4/15 | 3 — **fails** |
| **AF_A0 − CAT** | −0.0007 [−0.0051, +0.0039] 11/30 | −0.0003 [−0.0037, +0.0028] 7/15 | 1 — **fails** |
| **AF_A0 − AU_A0** | −0.0026 [−0.0064, +0.0007] 8/30 | −0.0033 [−0.0091, +0.0013] 5/15 | 2 — **fails** |
| **AF_A0 − AF_SHUF** | +0.0005 [−0.0017, +0.0025] 8/30 | −0.0009 [−0.0053, +0.0020] 5/15 | 3 — **fails** |

Reference contrasts (no gate role):

| contrast | MHC-ZH P1 | HateMM P1 |
|---|---|---|
| AU_PT − CAT | −0.0017 [−0.0057, +0.0020] 8/30 | **+0.0041 [+0.0008, +0.0080]** 8/15 |
| AU_A0 − CAT | +0.0019 [−0.0025, +0.0063] 13/30 | +0.0030 [−0.0003, +0.0066] 8/15 |
| LBL − CAT | +0.0021 [−0.0012, +0.0060] 7/30 | +0.0007 [−0.0020, +0.0038] 5/15 |
| AF_PT − LBL | −0.0024 [−0.0067, +0.0018] 9/30 | −0.0009 [−0.0039, +0.0019] 6/15 |

Mechanical application of the frozen rule: `idea-stage/r12_anchor/verdict.py` →
`verdict.json`. Both candidates `KILL`; the demotion clause is `False` on both datasets for both
candidates (it could not have fired: the dev contrasts do not exclude zero and the test contrasts
are negative).

---

## 2. Why it fails, in its own terms

### 2.1 The mechanism does not move the decision surface at all

`idea-stage/r12_anchor/{zh,hm}_errors.json`. Belt: macro-F1 recomputed from the dumped per-item
head logits matches the trainlog with **max abs diff exactly 0.0** on all 315 runs, so these are
the exact prediction sets behind the reported numbers.

Mean error-set Jaccard against `CAT` at each arm's own P1 epoch, seed-paired, against an
independence null over random subsets of the same observed sizes:

| arm vs CAT | MHC-ZH J (null, ratio) | HateMM J (null, ratio) |
|---|---|---|
| AU_A0 | 0.856 (0.085, 10.1×) | 0.914 (0.062, 14.7×) |
| AF_A0 | 0.842 (0.086, 9.8×) | 0.924 (0.063, 14.7×) |
| AU_PT | 0.865 (0.086, 10.1×) | 0.908 (0.062, 14.7×) |
| **AF_PT** | **0.849 (0.085, 9.9×)** | **0.913 (0.063, 14.5×)** |
| AF_SHUF | 0.871 (0.086, 10.2×) | 0.927 (0.063, 14.8×) |
| LBL | 0.907 (0.085, 10.6×) | 0.959 (0.063, 15.2×) |

Mean error counts: MHC-ZH 22.93-23.27 of 149 across all seven arms; HateMM 24.27-25.13 of 215.
The spread across the whole grid is **0.34 items on MHC-ZH and 0.87 on HateMM**.

**The comparison that makes this decisive.** `R10_COMBO_RESULT.md` §3 measured the error-set Jaccard
between `CAT` and `L24⊕L28` — two genuinely different feature read-outs — at **0.605** (MHC-ZH) and
**0.744** (HateMM). Every objective-level intervention in this pilot, including the one specifically
designed to change the breakage set, stays at 0.84-0.96. **A change of input features moves which
items are wrong three to four times as far as any change to the training objective tested here.**
The round's own recommendation — "find a mechanism that changes which items get broken" — was
tested with the family the literature nominates for exactly that job, and the family does not move
the error set enough to matter.

### 2.2 The focal gate is directionally wrong where the reference is good

The gate up-weights the anchor on items the out-of-fold reference already classifies correctly.
Teacher train accuracy, from `build_meta_*.json`:

| dataset | teacher | overall | class 0 | class 1 |
|---|---|---|---|---|
| MHC-ZH | A0 | 0.826 | 0.965 | **0.517** |
| MHC-ZH | PT | 0.838 | 0.957 | **0.572** |
| HateMM | A0 | 0.871 | 0.910 | 0.812 |
| HateMM | PT | 0.879 | 0.910 | 0.832 |

On MHC-ZH the reference is right on 96 % of negatives and barely better than a coin on positives,
so the focal weight is close to a positive-class down-weight — and the arm measures at zero.
On HateMM the reference is accurate on both classes, so the focal weight is close to uniform on
most items — and there the focal arm is **worse than uniform with the CI excluding zero**
(−0.0044). The gate therefore does nothing where it is most differentiated and hurts where it is
least. That is the opposite of the pattern the positive-congruent-training literature reports.

### 2.3 The shuffled control settles attribution before it can be claimed

`AF_PT − AF_SHUF` = +0.0008 / −0.0009, both CIs straddling zero. `AF_SHUF` uses a mask permuted
within class, so it has the same prevalence, the same per-class rate and the same weight histogram
as the real mask, and agrees with it on only 78.6 % (MHC-ZH) / 79.6 % (HateMM) of items. The
control clause was frozen precisely to prevent a "correctness-gated distillation" reading of an
effect that any arbitrary heterogeneous reweighting could produce. It is not needed as a gate here,
because there is no effect to attribute — but it converts "we found nothing" into "there is nothing
specific to reference correctness in this family on this substrate", which is a stronger statement.

### 2.4 What the uniform arms say, and why they are not an entry

`AU_PT − CAT` = +0.0041 with the CI excluding zero on HateMM, and −0.0017 on MHC-ZH; `AU_A0 − CAT`
is +0.0019 / +0.0030 with both CIs touching or containing zero. This is the same shape R11 found
and it fails the same way: positive on one dataset, not on the other, below the bar, and never with
both CIs clear. Fixing λ at 0.1 instead of dev-selecting it (R11 §2.4 showed dev selection is
corrupted by this family) moved the HateMM uniform arm from −0.0041 to +0.0041, which is worth
recording as a protocol observation and nothing more: it is one dataset, and the MHC-ZH sign flips
the other way.

`LBL`, the hard-label anchor, is again the best or joint-best anchor arm on MHC-ZH (+0.0021),
reproducing R11's conclusion that soft out-of-fold teacher knowledge is not worth more than
weighting the labels more heavily.

---

## 3. Scope limits

- Same-machine, head-level only. **No absolute number here is comparable to the project's
  A100-extracted ledger**; only within-table contrasts are results.
- MHC-ZH and HateMM only. One λ (0.1), one (α, β) = (1.0, 3.0), one teacher family (5-fold
  out-of-fold logistic probes on `A0` and `LL` features), one head, one hyper-parameter set.
- The teachers are linear probes, not the MLP head itself. R11 §7 already recorded that a
  non-linear or self-distilled teacher is untested; given that `LBL` matches or beats every soft
  teacher in both rounds, it remains a poor revival route.
- The focal form is the PC-Training `α + β·1(reference correct)` filter. Other members of the
  family — ELODI's top-K logit matching against an ensemble reference, MUSCLE's compatible
  adapters, `2202.02976`'s backward-congruent re-ranking — were not run. What this pilot licenses
  is that the *filter*, which is the family's stated active ingredient, is null-to-negative here.
- +0.005 is ≈ 0.7 test items of 149 (MHC-ZH) and ≈ 1.1 of 215 (HateMM).
- Standing caveat that applies to every number in this document: these test splits have been used
  by roughly 90 prior candidates. Paired-bootstrap intervals are conditional descriptive intervals,
  not post-selection-valid confirmatory ones.

---

## 4. Artefacts

| what | where |
|---|---|
| freeze (`a9cd557`) | `idea-stage/R12_FREEZE.md` §3 |
| deviation D1 | `idea-stage/R12_DEVIATION_D1.md` |
| pseudo-teacher + focal / shuffled weights, sha256 | `idea-stage/r12_anchor/{build_r12a.py,build_meta_*.json,teacher_*_PT.json,w_*.json,wshuf_*.json}` |
| weighted anchor loss | `src/model/loss.py::compute_anchor_loss`; `--anchor_weights` |
| grid runner (fork, 5 changed lines) | `idea-stage/r12_anchor/run_anchor_grid.sh` |
| single submission | `idea-stage/r12_anchor/run_all.sh` |
| judgement read-out | `idea-stage/r12_anchor/{zh,hm}_grid.json` |
| dev panel | `idea-stage/r12_anchor/{zh,hm}_devpanel.json` |
| error-set overlap | `idea-stage/r12_anchor/{zh,hm}_errors.json` |
| mechanical verdict | `idea-stage/r12_anchor/verdict.py` → `verdict.json` |
| logs | `logging/runs/r12_anchor/{run.log,run.pid,zh/,hm/}` |

**Belts that passed.** (i) With `--anchor_weights` absent, the edited loss reproduces the banked
R11 `ANCA` seed-700 trainlog **line for line** (all 30 dev/test metric lines identical), so the code
change is a verified no-op on the default path. (ii) The weighting algebra was unit-tested against
an independent computation before any arm ran, and all-ones weights reproduce the unweighted mean
to 1e-9. (iii) `run_rac.py` HALTs unless the weight file's train mean is 1.0 to within 1e-6, so
uniform and focal arms carry equal expected anchor mass by construction. (iv) macro-F1 recomputed
from the dumped per-item logits matches the trainlog with max abs diff **exactly 0.0** on all 315
runs. (v) 315/315 runs completed, 0 failures, 30/30 epochs each.

Seeds 900-929 / 900-914 are consumed and disjoint from every previously consumed range.
