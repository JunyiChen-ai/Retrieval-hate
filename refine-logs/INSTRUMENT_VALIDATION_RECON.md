# INSTRUMENT_VALIDATION_RECON — is the raw-train-space pregate a valid instrument?

**Agent:** instrument-validation recon · **Date:** 2026-07-28 NZST · **Cost: $0** (CPU, ≤4 threads).
**ZERO GPU / SLURM / Modal / training / test inference.** No new test-split evaluation was performed:
the deployed-side numbers below are **re-reads of already-banked, already-test-consumed artifacts**
produced by F88 (ERRPAT), F89 (MECHFIX) and F94 (KSWEEP), whose test touches were spent by those
findings. No `state/` file other than `findings.jsonl` (one appended row), no prereg, no config and no
frozen artifact was mutated. `refine-logs/{VSW_PREGATE_RECORD,VSW_ASYMMETRY_RECON,LITSWEEP7_LANDING_SITE,
LITSWEEP8_PATHOLOGY_MATCH,MEMBANK_C4_PREGATE_RECORD,STREAMCOMP_FORENSIC_RECON}.md` and
`scripts/analysis/{vsw_*,*membank_c4*}` were **read only**, never edited or executed.

**The question (GAP-A, `LITSWEEP7_LANDING_SITE.md:757-768`).** Every kill rendered by a `$0` pregate
in the raw banked train-space arena carries the same caveat — *"a raw-space null does not logically
entail a head-space null"* — and nobody has ever quantified it. Quantify it, or prove it cannot be
quantified at `$0`, and say which.

**Answer in one line — and it is the unwelcome one.** It **can** be quantified at `$0`, I quantified it
on 21 newly-paired points, and **the instrument is UNVALIDATED.** My own headline correlation
(pooled Spearman **+0.758**) **does not survive its own robustness check**: it is carried almost
entirely by three grid points per dataset where the operator degenerates to 1-NN and both arenas agree
for a trivial algebraic reason. Remove that degenerate block and the pooled correlation **inverts to
−0.3039** (per-dataset **+0.40 / −0.95 / +0.95**). In the regime where the campaign's decisions actually
live, three datasets disagree with each other.

**I also decline to certify the convenient rule.** The proposal *"arena negatives are informative;
arena positives are not"* is operationally prudent and I recommend acting on it — but after the two
retractions in §3.4/§3.6 the surviving positive-side evidence is **two matched pairs and one cell**,
one of the pairs is in a different metric, and the whole positive-side sample is **selected by
construction**. That is not enough to establish an asymmetry, and a rule that gets quoted forward and
is wrong would be worse than no rule. §6(a) states this as UNVALIDATED; §6(c) gives a clause built to
be safe *under* that verdict rather than one that assumes it away.

**What does survive, and it is worth having:** a *bound* rather than a correlation — across 21 matched
points the raw arena never missed a deployed effect larger than **0.67 test items**, and no point in
either arena reaches +0.010 — plus a **mechanical** account (§3.7c) of exactly which channels the two
arenas share and which they do not. Standalone deliverable: `refine-logs/PREGATE_CALIBRATION_CLAUSE.md`.

---

## §0. THREE CORRECTIONS TO THE TASKING'S PREMISE, BEFORE ANY MEASUREMENT

All three are load-bearing. Two shrink the exposure; one is a straight factual correction to a
sentence that four separate records repeat.

### 0.1 F89 and F94 are **not** raw-train-space pregates. They are **deployed head-space, test + dev** records.

The tasking says *"since F89, essentially every kill in this campaign has been rendered by a `$0`
pregate in the raw train-space arena"*. F89 and F94 are the two largest counterexamples, and they are
the reason a pairing is possible at all.

* **F89 / MECHFIX.** `MECHFIX_PREGATE_2026-07-27.md:52-54`: *"All arms operate in the **deployed head
  key space**, at **eval time only**."* Its results are paired same-head Δ **on test**
  (`:295-315`) with a dev corroboration read (`:485-486`), over 15 cells whose deployed floors are
  hard-asserted: `:191` *"Floor parity gate results — **15/15 PASS on test, 15/15 PASS on dev**"*.
* **F94 / KSWEEP.** `KSWEEP_RECORD.md:6-8`: *"Read-only replay of per-item neighbour lists that were
  **already banked and already test-consumed** … Zero GPU, zero SLURM, zero Modal, zero retraining,
  **zero new test inference**."* Its data sources (`:72-80`) are the deployed head-space per-item
  dumps for HateMM/ZH and `p2_out/cache_MHC_s{0,1,2,3}.json` for MHC-EN ARM-V, labelled **EXACT**.
* **F88 / ERRPAT** is likewise a deployed head-space forensic on test + dev.

**The raw-train-space regime begins at F95**, and it began by an explicit, argued choice against the
head space: `MECHNOV_PAIRVERIFY_PREGATE.md:159-166`.

### 0.2 The "head memorises train at LOO ≈ 0.998" premise is a **CLIP** number. The deployed **Qwen** heads measure 0.9406 / 0.8915 / 0.8154. **(NEWLY COMPUTED)**

This premise is the entire justification for abandoning the head space, and it is repeated verbatim in
four records: `MECHNOV_PAIRVERIFY_PREGATE.md:159`, `RESTRANS_PREGATE_RECORD.md:452-453`,
`AGGNET_PREGATE_RECORD.md:80-81`, `VGA_PREGATE_RECORD.md` (inherited), and again by the tasking and by
`LITSWEEP7_LANDING_SITE.md:764-765`.

Its source is F47, and F47 says something narrower. `directions_tried.json:171` (F47 `ban_scope`):
> "train-supervised = memorization-degenerate target, **CLIP LOO 0.998**"

and the memory index records the same entry as *"CLIP LOO train acc 0.998 vs **Qwen 0.800**"*. The
deployed system does **not** use the CLIP head.

**Re-read this session from `scripts/analysis/mechfix_{hatemm,zh,en}_OUT.json` →
`train_side_sanity.deployed_loo_train_acc`** — i.e. the *deployed* Qwen head's own train-split LOO
accuracy in the *deployed head key space*, the exact quantity the premise asserts:

| dataset | deployed head-space train LOO, per cell | mean | raw-arena deployed train LOO | **gap** |
|---|---|---|---|---|
| HateMM | 0.9395 / 0.9153 / 0.9476 / 0.9476 / 0.9462 / 0.9476 | **0.9406** | **0.8441** | **+0.0965** |
| MHC-ZH | 0.9361 / 0.9309 / 0.9240 / 0.8307 / 0.9102 / 0.8169 | **0.8915** | **0.8480** | **+0.0435** |
| MHC-EN | 0.7996 / 0.8142 / 0.8324 | **0.8154** | **0.7796** | **+0.0358** |

(raw-arena column re-read from `scripts/analysis/aggnet_pregate_OUT.json` →
`datasets.<ds>.spaces.fused.pooled.acc_deployed`; independently anchored at
`MECHNOV_PAIRVERIFY_PREGATE.md:298-300` and `VSW_ASYMMETRY_RECON.md:142-143`. Cross-check:
`MECHFIX_PREGATE_2026-07-27.md:234` prints HateMM `0.9476` for the same quantity.)

Correcting for bank size — `BSY_FORENSIC_RECON.md:177-180` measures the **full-bank** raw LOO at
0.8495 / 0.8480 / 0.7687 against F95's 4/5-bank 0.8441 / 0.8480 / 0.7796 — the gap becomes
**+0.0911 / +0.0435 / +0.0467**.

**So the two arenas differ by 3.6 to 9.7 accuracy points on the same train items, not by the
0.998-vs-0.84 chasm the premise asserts.** On MHC-EN — the dataset where the campaign's arithmetic is
tightest — they differ by **3.6 points**. This does not make the arenas identical; it does mean GAP-A's
"not the same object" framing was calibrated against the wrong number, and it materially reduces the
prior that a raw-space null hides a head-space positive.

*Caveat, stated so this is not over-read:* these are the ERRPAT CPU-proxy (HateMM, ZH) and ARM-F
snapshot (EN) heads, not the deleted floor heads (`MECHFIX_PREGATE_2026-07-27.md:460-464`; F78: 6/6
floor ckpts deleted). The proxy reproduces the HateMM val-sel floor exactly at 4 dp
(`ERRPAT_HateMM_2026-07-26.md:43`), so the offset is small, but the numbers are proxy-grade.

### 0.3 Every "parity anchor" in the campaign anchors a **floor to another record's floor inside one arena**. Not one connects a raw *treatment* to a deployed *outcome*.

The tasking asks this precisely, and the answer is uniformly negative:

| claimed anchor | what is actually asserted | does it link arenas? |
|---|---|---|
| F97 "78/78 parity" | `VGA_PREGATE_RECORD.md:196-201`: 26 **frozen F95 quantities per dataset**, hard-asserted at 4 dp — all of them *raw-arena* quantities | **No.** Raw ↔ raw, record ↔ record. |
| F95/F97 `fire_all` | `VGA_PREGATE_RECORD.md:232-235`: reproduces **F95 §3.2's** deltas −0.0040 / −0.0466 / −0.0146 | **No.** Reproduces a raw-arena *treatment* from a sibling raw-arena record. |
| ZH deployed train-LOO **0.8480** | `VSW_ASYMMETRY_RECON.md:142-143`, `BSY_FORENSIC_RECON.md:177-180`, `LITSWEEP8_PATHOLOGY_MATCH.md:123-124`, `MECHNOV_PAIRVERIFY_PREGATE.md:299` | **No.** A raw-arena *floor*, reproduced across four raw-arena records. Impressive harness discipline; zero arena-crossing content. |
| F89 "15/15 floor parity" | `MECHFIX_PREGATE_2026-07-27.md:191-213`: deployed-vote reproduction of the **deployed test and dev floors** | **Half.** Deployed floor ↔ deployed floor. It validates the *harness* in the deployed arena, not any raw treatment. |
| F94 "EN ARM-V k=20 bit-exact" | `KSWEEP_RECORD.md:115-117`: *"the EN ARM-V k=20 vote reproduces every banked `floor_vote` **bit-exactly** (max \|Δ\| = 0.0, all 4 seeds × 161 items)"* | **Half.** Deployed *floor*, bit-exact. Same reading. |

**Parity on the floor is not validation of the instrument for treatments — confirmed on 5 of 5.**

---

## §1. TASK A — the scope of the exposure

### 1.1 Arena of every closure since F88

| id | record | arena | what was measured | head-space confirmation? | test confirmation? |
|---|---|---|---|---|---|
| **F88** ERRPAT | `ERRPAT_{HateMM,MHC-EN,MHC-ZH}_2026-07-26.md` | **DEPLOYED head key space**, test + dev, 3-4 seeds, banked per-item lists | error structure + 6 `$0` repair operators | **is** head space | **YES** (test-consumed) |
| **F89** MECHFIX | `MECHFIX_PREGATE_2026-07-27.md:52-54` | **DEPLOYED head key space**, test + dev, proxy/snapshot heads | 5 eval-time vote operators T1-T4 | **is** head space | **YES** (`:191`, 15/15) |
| **F90** CLAP · **F91** Molmo2 · **F92/F93** MNTP | gate / probe records | mixed: `$0` info gates, raw-kNN reads, dev screens, CPU-head trains | encoder / channel additions | partial | F91 yes (proxy) |
| **F94** KSWEEP | `KSWEEP_RECORD.md:6-8, 72-80` | **DEPLOYED head key space**, test + dev, banked lists | k ∈ {1…60} sweep of the deployed vote | **is** head space | **YES** (already-consumed replay) |
| **F95** MECHNOV | `MECHNOV_PAIRVERIFY_PREGATE.md:144-166` | **RAW** banked encoder keys, **train** split, 5-fold item-disjoint | trained pair verifier replacing the vote | **NO** — declined by design (`:473-474`) | **NO** (`:496-497`) |
| **F96** RESTRANS | `RESTRANS_PREGATE_RECORD.md:51-63` | **RAW**, train | residual-transport vote (C1) | **NO** — *"inferred, not measured"* (`:452-453`) | **NO** (`:458`) |
| **F97** VGA/VNQ | `VGA_PREGATE_RECORD.md:79-88` | **RAW**, train | verifier-gated adjudication + neighbourhood quality | **NO** (`:422-426`) | **NO** (`:16-17`) |
| **F98** AGGNET | `AGGNET_PREGATE_RECORD.md:72-84` | **RAW**, train | learned conditional aggregation profile (C3) | **NO** (`:744-751`) | **NO** |
| **VSW** (F105) | `VSW_PREGATE_RECORD.md` | **RAW**, train | verifier-soft-reweighting | **NO** | **NO** |
| **F99** RDK | `RDK_FORENSIC_RECON.md` | arithmetic on banked raw-arena + banked deployed purity | pre-closure | — | — |
| **F100** EUM | `EUM_FORENSIC_RECON.md:5-6, 89` | train-split caches + cost arithmetic; *"deployed-encoder arena is unreachable at `$0`"* (`:37`) | pre-closure | — | — |
| **F101** BSY | `BSY_FORENSIC_RECON.md:166-180` | **RAW**, train | pre-closure | **NO** — `:161-162` explicit | **NO** |
| **F102** TVB | `TVB_FORENSIC_RECON.md:5-6` | split manifests + banked P8 results | pre-closure | — | — |

### 1.2 The fraction, stated plainly

Of the twelve operator-bearing closures since F88:

* **five rest wholly on the raw train-space arena** — F95, F96, F97, F98, VSW;
* **two more lean on raw-arena arithmetic** — F99 (RDK), F101 (BSY);
* **three are deployed-arena** — F88, F89, F94;
* **two rest on other evidence** — F100 (cost/ban arithmetic), F102 (code inspection + banked P8).

**So the exposure is 5 of 12 wholly, 7 of 12 including leaners — not "roughly ten".** The tasking's
estimate is about 2× high, and the correction matters because F89 and F94 are precisely the two
records that make the instrument testable.

---

## §2. TASK C(pre) — FROZEN DESIGN OF THE `$0` MEASUREMENT (written before any paired number was read)

Frozen at `<scratchpad>/ivr_freeze.md` (sha256 `90e9ea26be56456d153f1d316d8d4e436404f5b39504f2494e5b0e68d0cd7647`),
executed by `<scratchpad>/ivr_pair.py` (sha256 `738a167a545981551748d760a1ac3747370c4638c9f9e8d923e217ad9a9fe14d`).

**The pairing.** One operator family has been measured in **both** arenas on the **same 8-point grid**:
*truncate the deployed rank-weight vector to `[k..1, 0…]` over the deployed top-20*.

* **RAW side:** `FIXK_{1,2,3,5,7,10,15,20}` in `scripts/analysis/aggnet_pregate_OUT.json`, defined at
  `AGGNET_PREGATE_RECORD.md:190` as *"the eight **F94 grid profiles** `[k..1, 0…]`"* — i.e. F98 built
  its degeneracy control to be F94's operator on purpose. Train split, `StratifiedKFold(5,
  shuffle=True, random_state=0)`, item-disjoint, raw 7168-d fused keys, n = 744 / 579 / 549.
* **DEPLOYED side:** F94's k-sweep, `scripts/analysis/ksweep_OUT.json` / `KSWEEP_RECORD.md:100-215`.
  Deployed 1024-d head keys, **test** and **dev**, 3-4 seeds, n = 215 / 149 / 161.

**Declared before any number was read** (verbatim from the freeze): PRIMARY raw space = `fused`;
PRIMARY deployed arm = HateMM final-epoch / MHC-ZH final-epoch (*"(binding arm)"*,
`KSWEEP_RECORD.md:158`) / MHC-EN **ARM-V** val-selected (the deployed headline stack, the only EXACT
non-proxy cell). Grid k ∈ {1,2,3,5,7,10,15}; **k = 20 excluded** because Δ = 0 by construction on both
sides and would be a free anchor inflating any correlation — it is instead a **parity assert**.
Statistics: Spearman `rho_S`, Pearson `r_P`, sign agreement with a one-item tolerance per arena, median
magnitude ratio over |raw Δ| ≥ 0.010, and argmax-k agreement.

**Interpretation rule, fixed in advance.** VALIDATED iff pooled `rho_S` ≥ +0.70 **and** sign agreement
≥ 6/7 on ≥ 2 of 3 datasets **and** argmax-k agrees on ≥ 2 of 3. NOT VALIDATED iff pooled `rho_S` ≤ +0.30
**or** sign agreement ≤ 4/7 on ≥ 2 of 3. ANTI-CORRELATED iff pooled `rho_S` < 0. Between +0.30 and
+0.70 = WEAK / DIRECTIONAL ONLY.

**Scope, declared before the result so it cannot be widened afterwards.** Whatever this returns, it
speaks only to operators that **preserve the retrieved set and re-weight it** (LITSWEEP7 channel (b)).
It says nothing about representation change (a), membership change (c), or map training (d).

**Determinism status of this read — it is on the safe side of the F105 erratum.** The erratum
(`VSW_PREGATE_RECORD.md:514-582`) establishes that the frozen F95 module reproduces **every
closed-form quantity exactly at 4 dp** and drifts on **44 of 48 torch-fitted quantities**. `FIXK_k` is
a **fixed weight profile applied in closed form** to the deployed vote — no fitted parameter anywhere
— and the deployed vote itself is the bit-exact side of the erratum. The parity assert below
(`FIXK_20` changes 0 items in 9/9 cells) confirms it directly. **None of §3.1's numbers is a trained
quantity.**

<!-- EVERY NUMBER BELOW THIS LINE WAS COMPUTED AFTER THE DESIGN ABOVE WAS FROZEN -->

---

## §3. TASK B — EVERY CASE WHERE BOTH A RAW-ARENA AND A DEPLOYED READ EXIST

**Seven candidate pairs were examined. Two were RETRACTED on construction-checking (§3.4 R1, §3.6),
and of the five that survive, none establishes predictiveness.** Read §3.1b before citing §3.1: the
headline correlation there does not survive its own robustness check.

### 3.1 PAIR 1 — the k-grid, 21 paired points, both arenas, same operator **(NEWLY COMPUTED)**

**Parity assert first (mandatory).** `FIXK_20.d_acc = 0.0000` and `FIXK_20.n_changed = 0` in **all 9**
dataset × space cells of `aggnet_pregate_OUT.json`. The raw grid's k = 20 **is** the deployed rule,
bit-identically, so the two grids are the same operator. On the deployed side,
`ksweep_OUT.json.parity_gate[*].bit_exact_4dp` is `true` on **19/19** cells. **Harness valid.**

| | k=1 | k=2 | k=3 | k=5 | k=7 | k=10 | k=15 |
|---|---|---|---|---|---|---|---|
| **HateMM** raw train Δacc (n=744) | −0.0430 | −0.0430 | −0.0430 | −0.0054 | −0.0121 | +0.0027 | +0.0040 |
| **HateMM** deployed **test** Δacc (n=215, final, 3 seeds) | −0.0388 | −0.0388 | −0.0388 | −0.0062 | −0.0000 | −0.0016 | +0.0000 |
| **MHC-ZH** raw train Δacc (n=579) | −0.0293 | −0.0293 | −0.0293 | −0.0224 | −0.0138 | −0.0121 | −0.0121 |
| **MHC-ZH** deployed **test** Δacc (n=149, final, 3 seeds) | −0.0179 | −0.0179 | −0.0179 | +0.0045 | +0.0023 | +0.0000 | +0.0022 |
| **MHC-EN** raw train Δacc (n=549) | −0.0437 | −0.0437 | −0.0437 | −0.0109 | −0.0164 | −0.0036 | −0.0055 |
| **MHC-EN** deployed **test** Δacc (n=161, ARM-V, 4 seeds) | −0.0388 | −0.0388 | −0.0388 | −0.0078 | −0.0078 | −0.0000 | −0.0016 |

(raw rows from `aggnet_pregate_OUT.json`; deployed rows recomputed as seed-mean Δ vs k=20 from
`ksweep_OUT.json` and agreeing with the tables printed at `KSWEEP_RECORD.md:132-197`.)

| statistic | HateMM | MHC-ZH | MHC-EN | **pooled (21 pts)** |
|---|---|---|---|---|
| Spearman `rho_S` (raw ↔ test) | **+0.8846** | **+0.6408** | **+0.9903** | **+0.7580** |
| Pearson `r_P` (raw ↔ test) | +0.9688 | +0.8594 | +0.9955 | **+0.9104** |
| sign agreement (1-item tolerance) | 4/7 | 3/7 | 5/7 | 12/21 |
| median ratio deployed/raw, \|raw\| ≥ 0.010 | **0.9023** | −0.0 | **0.8884** | — |
| argmax k: raw → test | 15 → **15** ✓ | 10 → 5 ✗ | 10 → **10** ✓ | 2/3 |
| Spearman raw ↔ **dev** | +0.6202 | **−0.6765** | *(EN ARM-V has no dev curve)* | — |

**Secondary cells (all computed, none dropped):** secondary raw spaces vs the same deployed arm —
`text` +0.9903 / +0.9903 / +0.9515, `img` +0.9321 / +0.6991 / +0.8544; secondary deployed arms vs the
primary raw space — HateMM val-sel +0.8807, ZH val-sel +0.5049, EN ARM-F final +0.7692. **Every one of
these 12 carries the same degenerate block as the primary and is subject to the same correction in
§3.1b. They are not independent corroboration and must not be read as such.**

**Verdict against the frozen rule.** Pooled `rho_S` = +0.758 clears the +0.70 gate and argmax-k agrees
on 2 of 3 — but sign agreement is ≤ 4/7 on **2 of 3** datasets, so the frozen **NOT VALIDATED**
disjunct **fires**. Under the letter of §2's frozen rule the instrument is **NOT VALIDATED**, and
§3.1b shows the letter was right and my first instinct to explain it away was wrong.

**The explaining-away I attempted, recorded because it was wrong and the reason matters.** My first
reading was that every sign disagreement is a resolution artefact rather than a contradiction — one
item = 0.0013 / 0.0017 / 0.0018 raw versus 0.0047 / 0.0067 / 0.0062 test, so the deployed arena is
**3.5× coarser** and carries a ±0.014 seed band on top. That observation is *true*, and it is the
reason §3.1b concludes the deployed arena **cannot adjudicate at this scale** — but it does not rescue
the correlation, it undermines the whole exercise. Restricting to the 16 points where |raw Δ| ≥ 0.010:

* strict sign agreement **11/16**;
* the 5 remainder are **all** cases where the deployed Δ is smaller than **one test item** (HateMM k=7:
  −0.00003 = −0.01 items; ZH k=5/7/10/15: +0.0045 / +0.0023 / 0.0000 / +0.0022 = 0.67 / 0.34 / 0.00 /
  0.33 items);
* **FALSE KILLS — raw ≤ −0.010 while deployed ≥ +0.010: 0 of 16.** The largest contrary deployed
  excursion anywhere is **+0.0045 = 0.67 test items**;
* magnitude ratio **0.90× (HateMM) and 0.89× (EN)** — **but both medians are taken over sets dominated
  by the degenerate block, so this is not a transfer rate and I withdraw it as one**;
* over all 21 points the maximum Δ is **+0.0040 (raw)** and **+0.0045 (test)** — **neither arena reaches
  +0.010, let alone the +0.030 bar.** Both arenas independently agree the family is dead.

**MHC-ZH's negative dev correlation is not evidence against the raw arena.** It is the known ZH dev
pathology: `ERRPAT_MHC-ZH_2026-07-26.md:91-98` measures the dev↔test Spearman across 25 legal epochs at
**−0.2402 pooled, p = 0.0380** — *"The pooled rank correlation is **significantly negative**"*. On ZH
the raw arena disagreeing with dev is the raw arena agreeing with test.

#### 3.1b THE ROBUSTNESS CHECK THAT BREAKS MY OWN HEADLINE **(NEWLY COMPUTED — read this before citing §3.1)**

`KSWEEP_RECORD.md:29-31` records that *"under rank weights `[k..1]` with descending cosines, **k ≤ 3 is
algebraically a plain 1-NN classifier** (verified identical to the top-1 label vector in 19/19 cells)"*.
So the k ∈ {1,2,3} block is **three identical points per dataset** at which both arenas agree that 1-NN
is worse than 20-NN. **That is a fact about kNN, not evidence that the two arenas track each other.**
It contributes 9 of the 21 points and it sits at the extreme of both axes, where it dominates any
correlation.

Recomputed with that block removed:

| point set | n | pooled `rho_S` | pooled `r_P` | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|---|---|---|
| all k ∈ {1,2,3,5,7,10,15} | 21 | **+0.7580** | +0.9104 | +0.8846 | +0.6408 | +0.9903 |
| **k ∈ {5,7,10,15}** (non-degenerate) | 12 | **−0.3039** | **−0.1522** | **+0.4000** | **−0.9487** | **+0.9487** |
| k ∈ {7,10,15} | 9 | −0.0766 | +0.1407 | +0.5000 | −0.8660 | +1.0000 |

**The headline is an artefact of the degenerate block, and the honest reading is the second row.** On
the 12 non-degenerate points the pooled correlation is **negative**, and the three datasets give
**+0.40 / −0.95 / +0.95** — i.e. they disagree with each other as much as either disagrees with the
deployed arena. The retained points are:

| | raw Δ (k=5,7,10,15) | deployed test Δ |
|---|---|---|
| HateMM | −0.0054, −0.0121, +0.0027, +0.0040 | −0.0062, −0.0000, −0.0016, +0.0000 |
| MHC-ZH | −0.0224, −0.0138, −0.0121, −0.0121 | +0.0045, +0.0023, +0.0000, +0.0022 |
| MHC-EN | −0.0109, −0.0164, −0.0036, −0.0055 | −0.0078, −0.0078, −0.0000, −0.0015 |

**Every one of these 24 numbers is inside its own arena's noise** — the deployed side's 12 values span
−0.0078 to +0.0045, i.e. **−1.7 to +0.7 test items**, against a ±0.014 3-seed band. So the correct
statement is not "the arenas agree" nor "the arenas disagree": **at the scale where the disagreements
live, the deployed test arena has no power to adjudicate.** One test item is 0.0047-0.0067; the
campaign's interest line is +0.010 and its bar is +0.030.

**Consequence, and it is the load-bearing one for §6.** What §3.1 establishes is a **bound, not a
correlation**: over 21 matched points there is **no case where the raw arena said ≤ −0.010 and the
deployed arena said ≥ +0.010**, and the largest contrary deployed excursion is **+0.0045 = 0.67 test
items**. That bound is robust to which points you include, because it is a max over the whole set. The
*correlation* is not, and I withdraw it as a validation.

### 3.2 PAIR 2 — the threshold / calibration family. **The one pair where a raw positive has a deployed twin, and it does not transfer.**

| arena | operator | HateMM Δacc | source |
|---|---|---|---|
| **RAW train, 5-fold, nested-selected (LEGAL)** | `THRESH_best`, bare global threshold on the deployed vote | **+0.0188** (fused), **+0.0242** (text) | `AGGNET_PREGATE_RECORD.md:543-551`; re-read from `aggnet_pregate_OUT.json` |
| **RAW train, 5-fold (LEGAL)** | D1 = threshold **+ length covariate** | **+0.0215** (fused, ER 1.8889, 34 fixed/18 broken, net +16, 4/5 fold signs), **+0.0282** (text, ER 2.2353, 38/17, net +21, 5/5) | `RESTRANS_PREGATE_RECORD.md:425-428` |
| **DEPLOYED head space, TEST (LEGAL)** | dev-fitted threshold by accuracy | **+0.0000** (val-sel) / **+0.0016** (final) | `ERRPAT_HateMM_2026-07-26.md:166` |
| **DEPLOYED head space, TEST (LEGAL)** | dev-fitted threshold by macro-F1 | **+0.0016** acc / +0.0031 mF1 | `ERRPAT_HateMM_2026-07-26.md:167` |
| **DEPLOYED head space, TEST (LEGAL)** | train-LOO-fitted logistic recalibration | **−0.0016** acc / −0.0017 mF1 | `ERRPAT_HateMM_2026-07-26.md:168` |
| **DEPLOYED head space, TEST — GOLD-CHEATING ORACLE** | best **test-fitted** threshold | **+0.0078** (val-sel) / **+0.0124** (final) | `ERRPAT_HateMM_2026-07-26.md:163-164` |

**The decisive line: the raw arena's *legally fitted* number (+0.0188) exceeds the deployed arena's
*gold-cheating ceiling* (+0.0124) by 1.5×, and its text-space number (+0.0242) exceeds it by 2.0×.**
There is no reading under which +0.0188 of that is real in the deployed space; the deployed space
does not contain it even for a cheat.

The two sibling records reached the same conclusion from opposite ends without assembling it:
`RESTRANS_PREGATE_RECORD.md:432-433` — D1 *"was measured dead in the **deployed head space on test**
(−0.0016 train-LOO fit, +0.0000 dev fit), and this is a **raw-space, train-side** screen, so the arenas
are not comparable"*; `AGGNET_PREGATE_RECORD.md:558-562` — *"on HateMM's train arena the deployed vote's
**threshold** is simply mis-set, and any sufficiently flexible operator … converges to the same
correction … the lever remains measured **dead in the deployed head space on test**"*.

**This is the single most important datum in this record.** It is a positive of net +14 to +21 items,
with above-bar exchange rates and 4/5-5/5 fold signs, that the deployed space cannot supply even with
the answer key. Cross-dataset it also shows the effect is HateMM-specific: `THRESH_best` fused is
**−0.0069** on MHC-ZH and **−0.0164** on MHC-EN (`aggnet_pregate_OUT.json`), against deployed-legal
MHC-EN **−0.0083** (`ERRPAT_MHC-EN_2026-07-26.md:496`, 0 of 6 arms improve) — same sign, so the arenas
agree wherever the raw arena is negative and disagree exactly where it is positive.

*Mechanism, reported honestly as suggestive-not-established.* The raw arena's deployed-vote positive
rate drifts from the bank rate by **+0.0807 / +0.0380 / −0.0455** (fused; `aggnet_pregate_OUT.json`),
and HateMM's is by far the largest. **But over all 9 dataset × space cells the correlation between that
drift and `THRESH_best`'s gain is only Pearson +0.1978 / Spearman +0.3167 (newly computed)** — the
fused-only triple lines up (+0.0807 → +0.0188, +0.0380 → −0.0069, −0.0455 → −0.0164) but the full cell
set does not support it. Recorded as a hypothesis, not a finding.

### 3.3 PAIR 3 — stream composition, HateMM, **head space** (the only genuinely like-for-like stream pair)

| arena | text-only vs fused | source |
|---|---|---|
| RAW train LOO, n=744 | text 0.8441 vs fused 0.8441 = **+0.0000** | `aggnet_pregate_OUT.json` `spaces.{text,fused}.pooled.acc_deployed`; independently at `VSW_ASYMMETRY_RECON.md:285` |
| **DEPLOYED head space, TEST**, 3 seeds | `text_proj`-only 0.8822 / 0.8853 vs deployed 0.8775 / 0.8760 = **+0.0047** (val-sel, sign 2/3) / **+0.0093** (final, sign 3/3) | `ERRPAT_HateMM_2026-07-26.md:514-517`; re-read from `errpat_hatemm_ceilings_OUT.json.stream_means` |

Raw **under**-reads by 1-2 test items. Both sides are inside the ±0.014 seed band and both are far
under bar. Caveat recorded by the source itself (`:522-524`): the deployed read is *post-hoc* — the
`text_proj` sub-space of a head trained under Hadamard fusion — so it is not a trained text-only head.

### 3.4 PAIR 4 — the representation channel: **TWO RETRACTIONS. This is where I over-read the evidence.**

Law-I *is* the statement "the raw representation improved and the deployed number did not follow", and
the campaign has nine certified instances (`findings.jsonl:91`; listed at `LITSWEEP7_LANDING_SITE.md:812-815`
as P3, S2S, W2-A, Router, FA, premise-d, LP, vision-unfreeze, Molmo2). **But this set cannot be used the
way I first used it, and I am withdrawing two things.**

**RETRACTION 1 — the F91/Molmo2 numeric pair is UNMATCHED and is withdrawn.** I originally cited
`MOLMO2_PROBE_RECORD.md:87` (raw concat 0.8186, *"the best fused raw read of all three arms"*, `:122`)
against `:64` (arm A − floor B, val-sel **−0.0217**). **Verified in source this session:**
`scripts/analysis/molmo2_geom_diag.py:71-78` builds `"concat": np.concatenate([img, txt], axis=1)` —
**no per-stream L2 normalisation**. The raw arena's fused key and the deployed key both normalise each
stream *before* combining. **Different construction ⇒ not a matched pair.** Withdrawn from every
comparison in this record.
*What survives, because it is construction-independent:* the trained head's HateMM floor is **0.8760**
(`MECHFIX_PREGATE_2026-07-27.md:305`) against a best raw key of **0.8233** — **the trained head is worth
about +0.05 over the best raw key anyone has built.** That is a statement about how far apart the two
arenas are, not about whether one predicts the other.

**RETRACTION 2 — the nine law-I data are a ONE-SIDED SAMPLE and cannot establish an asymmetry.**
Law-I's population is, by construction, *"cases where the raw/probe metric improved and someone
therefore paid to take it to a deployed measurement"*. Nobody ever spends GPU on a representation that
got **worse** in raw space. So the set can only ever exhibit "raw positive → deployed flat"; **it has
no power to test whether a raw negative is trustworthy**, and its unanimity is partly a sampling fact
rather than a property of the arena. It remains strong evidence that **raw positives are unreliable**;
it is **not** evidence that raw negatives are reliable, and I originally over-read it as both.

### 3.5 PAIR 5 — the *other* cheap instrument the campaign already wrote a law about (and it is one-sided too)

The no-head probe is a different surrogate arena, and its calibration is settled:
`research-wiki/EXP_p8_semantic_compression.md:132-135` — *"P8 had the **strongest** no-head probe of
any front … yet the trained retrieval head does WORSE on the compressed text than on the raw
chunk-mean. **A passing no-head probe is necessary but not sufficient**"*; reproduced as a campaign
law at `research-wiki/DRAFT_analysis_chapter.md:115` and `research-wiki/CAMPAIGN_mllm_method_role.md:38`
(*"HateMM had the cleanest probe of the three yet…"*), with P3 measured on all three datasets
(`CAMPAIGN_mllm_method_role.md:55-56`), and `directions_tried.json` recording P3 as *"probe pass, train
flat, 3 datasets"*.

**Same direction again: a cheap arena's positive did not survive.** Note carefully what this does
*not* show: nobody ever took a *failing* probe to a deployed measurement, so this family carries **no
information at all** about whether a cheap-arena negative is trustworthy. It is a one-sided sample.

### 3.5b THE TWO MATCHED PAIRS THAT SURVIVE STRICT CONSTRUCTION-CHECKING — and both fail to predict

After the retractions in §3.4 and §3.6, exactly two stream-composition pairs remain in which the two
sides are the **same operator on the same construction**. I verified the key construction in source for
each, because that is precisely what killed the other two.

**(i) MHC-ZH, accuracy — same sign, 2.3× shrink.** Raw train-LOO text−fused **+0.0156**
(`aggnet_pregate_OUT.json`) → banked test **+0.0067** (text 0.8523 vs fused 0.8456,
`ERRPAT_MHC-ZH_2026-07-26.md:252-253`) = **one item on n=149**. Construction verified matched:
`scripts/analysis/errpat_zh_taxonomy.py:292-301` builds `fused_concat_l2n = np.hstack([l2n(tr_img),
l2n(tr_txt)])`, identical to the raw arena's `L2norm(concat(L2norm(img), L2norm(text)))`. Same encoder
cache (`…LoRA_HF`) both sides, same vote operator. **Differences that must be stated:** split
(train-LOO vs test), bank size, n (579 pooled vs 149).

**(ii) HateMM, AUC — a genuine SIGN INVERSION.** From `HATEMM_LORA_STREAM_DECOMP.md:79-85`, per-stream
kNN AUC under the LoRA encoder: text **0.920** vs concat **0.909** on train-LOO (n=744) = **+0.011**;
text **0.899** vs concat **0.910** on held-out dev (n=107) = **−0.011**. Construction verified matched:
`scripts/analysis/encoder_swap_geometry.py:63-65` builds `concat = np.concatenate([l2n(img), l2n(txt)])`
— per-stream L2, the matched form. **Caveats that must travel with it: this is AUC, not accuracy;
n_dev = 107; and the held-out side is dev, not test.**
*Robustness, computed from the same table:* the other two encoders give train→dev text−concat of
+0.005 → −0.034 (frozen-Qwen, **also inverts**) and −0.020 → −0.026 (CLIP, agrees). **2 of 3 encoders
invert.**

**So on the two datasets where a matched stream pair exists, the raw train-LOO delta fails to predict
the held-out delta on both — one shrinks 2.3×, one inverts sign.** Two pairs, one of them in a
different metric, on a family whose effects are all ≤ 1-2 items. **This is enough to say the arena is
not established as predictive. It is nowhere near enough to establish a systematic asymmetry.**

### 3.6 The claimed MHC-EN "sign inversion" is a **FALSE PAIR** — and saying so is load-bearing

A stream-composition pair was relayed to me mid-task as: ZH +0.0156 → +0.0067 (same sign) and **MHC-EN
+0.0310 → −0.0109 (SIGN INVERSION)**. I verified both sides and **the EN half does not survive**.

* **Raw side, verified independently** from `aggnet_pregate_OUT.json` (`acc_deployed` per space):
  HateMM text 0.8441 vs fused 0.8441 (**+0.0000**); ZH text 0.8636 vs fused 0.8480 (**+0.0156**);
  EN text 0.8106 vs fused 0.7796 (**+0.0310**). These match `VSW_ASYMMETRY_RECON.md:285-287` exactly.
* **MHC-EN "test" side.** `ERRPAT_MHC-EN_2026-07-26.md:238-246` reports Qwen **text-only 0.7826** —
  from `cross_channel_router_gate.raw_modality_vote`, **raw features**, test split — against a row
  labelled *"deployed fused (ARM-V, 4 seeds) 0.7935"*, which is a **trained-head** number.
  **The −0.0109 subtracts a raw-feature arm from a deployed-head baseline.** The record says so itself
  at `:256-258`: *"**The entire trained stack buys ~+0.011 acc / +0.005 mF1 over a raw text 20-NN with
  no head at all**"*. That is the head's value-add with the sign flipped — **not** a text-vs-fused
  contrast, and **not** an arena-prediction failure. No raw-fused-on-test number for EN exists in the
  record, so the like-for-like Δ cannot be formed from banked data.
* **MHC-ZH "test" side.** `ERRPAT_MHC-ZH_2026-07-26.md:244-247` is explicitly headed *"STREAM FORENSICS
  (**pre-head raw banked features; NOT the deployed head space**)"* and states *"The deployed fusion is a
  trained Hadamard `align`, so **no head-space single-stream vote exists**."* Its +0.0067 (text 0.8523
  vs fused 0.8456, `:252-253`) is therefore **raw train-LOO vs raw test** — a *generalisation* pair
  inside one arena, not an arena pair. Same sign, magnitude 0.43×, and the test Δ is **1 item**.

**Ruling: of the three stream cells, only HateMM (§3.3) is a legitimate raw↔deployed pair. The EN
sign inversion is an artefact of mismatched baselines and must not be cited as instrument evidence.**
A false pair would have been worse than none, and this one would have inverted the verdict.

**What is real in it, and it is worth flagging:** on MHC-EN the raw arena reports **+0.0310** for
dropping the image stream — **the single largest raw-arena positive anywhere in the campaign, and above
the +0.030 bar** — and its like-for-like deployed twin had never been measured when I flagged it.

**Update, same day: F108 / STREAMCOMP settled it while this record was being written, and its numbers
match mine exactly.** `findings.jsonl` F108 independently reports the identical three reads — HateMM
head-space +0.0047 / +0.0093, ZH raw-key-on-test +0.0067 (*"= ONE ITEM"*), MHC-EN *"Qwen text-only
0.7826/0.7448 vs **the deployed pipeline** 0.7935±0.0205"* = −0.0109 — and its own wording (*"vs the
deployed pipeline"*) confirms §3.6's diagnosis that the EN leg is not a text-vs-fused contrast. F108
then measures the **deployable** stream weight (`a*` selected per outer fold on the fitting fold's own
LOO): HateMM **−0.0027**, MHC-ZH **+0.0346**, MHC-EN **+0.0200** — *"CONJUNCT ON EXACTLY ONE DATASET"*,
against full-hindsight +0.0040 / +0.0328 / +0.0383 (2 of 3). **So the raw arena's largest positive
degrades from 2-of-3 under hindsight to 1-of-3 when made deployable — the same optimism this record
measures, arriving independently in a fourth family.** The direction is additionally D7-dead by name.
**This item is therefore closed, not open**, and F108 is folded into §6(b) accordingly.

### 3.7 Claimed pairs that do not exist

* **cand-2 curriculum.** Confirmed independently: no train-arena treatment read exists. The only
  curriculum-adjacent train quantity is `CAND2_KC20_HateMM.json` `loo_acc 0.8065` and
  `CAND2_KC20_MHC_zh.json` `loo_acc 0.7927` — **pre-curriculum, frozen-Qwen, mining inputs**, on a
  different feature cache from the raw arena's. Its deployed side is banked (F59 pooled **+0.01317**,
  5/6 signs, `findings.jsonl:59`; ZH TIE, `directions_tried.json` `positives_bank[4-5]`).
  **Superseded during this session:** the MAC/provenance line has now produced the train-arena read —
  see §3.7c. LITSWEEP7 §0.2 was correct that no such read existed in the *records*; the caches did.
* **F56 / F59** are deployed test reads only (`positives_bank[4]`: HateMM-curric val-sel
  0.8775/0.8711, +0.0155 over generic; `positives_bank[5]`: rep2 pooled 6-pt +0.01317). No train-arena
  companion.

### 3.7b PAIR 6 — F107 / HEADCOV: a **structural** raw→head transfer, and its own control says read it narrowly

F107 landed during this session and is the campaign's first *deliberate* raw→head transfer test.
LITSWEEP8 established in the **raw** arena that the deployed decision reads only the retrieved ordered
label tuple (99.6-100 % identity); `HEADCOV_PREGATE_RECORD.md:186` measures the same identity **in the
deployed head space**: *"**K-HC-3 — PASS at 1.0000. Result A transfers to the deployed head space**"*,
0/78 differing items × 3 seeds, and `:383` records 0.9989 over 90 cells, min 0.9872.

**This is a genuine transfer datum and it points the same way as everything else — but HEADCOV's own
degeneracy control failed and the record says so.** Per F107's body: *"DEG-HC CONTROL FAILED AND IS
REPORTED AS FAILED: its premise (epoch 0 = uncollapsed reference) is measured FALSE … the head-space
identity is essentially **FORCED by collapse** … and the mechanism evidence lives in the RAW arena
instead, where LITSWEEP8 measured 99.6-100 % identity **DESPITE** a real 0.021-0.025 spread."*

**Correct reading, and it is a useful one:** on this property the **raw arena is the *stronger* test**,
because it demonstrates the identity where there is dynamic range to violate it, while the head space
demonstrates it where collapse makes it near-tautological. That is a second, independent reason to
keep running pregates in the raw arena rather than the head space — and it is the opposite of the
concern GAP-A raised.

### 3.7c PAIR 7 — the training channel (d), supplied by the MAC provenance audit. **CURDIAG is discharged, with a SPLIT answer.**

`refine-logs/PROVENANCE_AUDIT_2026-07-28.md` (commit `0477d56`) makes the measurement §4.3 listed as
the largest remaining hole. Verified this session against that record:

* **Curriculum cell — arena VALID.** `:198-200`: litsweep-7's outcome 1 obtains — *"HateMM +0.0068
  (2-draw) / +0.0202 (draw-1) train-arena, +0.0132 test; ZH +0.0086 train-arena, TIE test. **Same
  sign, same ordering**, HateMM ≫ ZH on both sides."* `:206-212`: *"the instrument is valid, the
  campaign's `$0`-pregate base rate stands, and **no confidence discount is owed on that ground**"*,
  with the caveat that the arena **attenuates by roughly 2×** (+0.0068 vs +0.0132) — *"valid in sign,
  conservative in magnitude. A kill decided by an arena margin under ~0.007 should not be treated as
  decisive on that ground alone."*
* **Encoder-adaptation axis — arena ordering INVERTED.** `:219-228`, frozen-Qwen vs generic-LoRA train
  arena: HateMM 0.8065 → 0.8293 (**+0.0228**, smallest), MHC-ZH 0.7927 → 0.8480 (+0.0553), MHC-EN
  0.7687 → 0.8415 (**+0.0729**, largest). *"The train arena's cross-dataset ordering … is EN > ZH >
  HateMM. The test-side conversion ordering is the **exact reverse**."*

**So the training channel now has both a confirming and a disconfirming pair, and the disconfirming one
has a named mechanical cause that this record can supply — the two arenas do not share a fusion
operator.** Verified directly in source: the **deployed** key is Hadamard on two learned projections,
`src/model/classifier.py:87,140-141` (`fusion_mode == 'align'`), rendered at
`MECHFIX_PREGATE_2026-07-27.md:27` as `mlp[:-2]( normalize(img_proj(x_img)) * normalize(text_proj(x_txt)) )`;
the **raw arena's** fused key is `L2norm( concat( L2norm(img_feats), L2norm(text_feats) ) )`
(`MECHNOV_PAIRVERIFY_PREGATE.md:150`). **Elementwise product versus concatenation.**

This is decisive for scoping, and it is the most transferable thing in this record:

* On **channel (b)** — re-weight/truncate the retrieved top-20 — the two arenas share *everything below
  retrieval*: the same ordered label tuple, the same rank weights, the same threshold. There is a
  structural reason for the +0.758 correlation in §3.1.
* On **channel (a)/(d)** — anything that changes the representation or the map — the raw arena's
  L2-concat and the deployed Hadamard are **different functions of the same two streams**, and F44's
  measured mechanism (EN's Qwen image stream collapses and the equal-weight concat cancels the text
  gain) says exactly where they diverge. A raw representational gain has **no mechanical reason** to
  survive the deployed fusion. **The encoder-axis inversion is therefore not a mysterious arena
  pathology — it is two different fusion operators, and it is predictable in advance.**

### 3.8 So: does the raw arena predict the deployed outcome — in sign? in magnitude? at all?

| question | answer | evidence |
|---|---|---|
| **At all?** | **NOT ESTABLISHED.** The only large paired grid gives +0.758 with a degenerate block in it and **−0.304 without it**; the two matched stream pairs shrink 2.3× and invert | §3.1b, §3.5b |
| **In sign?** | **No, not reliably.** On the 12 non-degenerate k-points the three datasets give +0.40 / −0.95 / +0.95. Of the two matched stream pairs, one holds sign and one inverts (2 of 3 encoders invert) | §3.1b, §3.5b(ii) |
| **In magnitude?** | **No.** The one place magnitude looked stable (0.89-0.90×) is inside the degenerate block. Elsewhere: ZH stream 2.3× shrink; §3.2's raw legal exceeds the deployed **oracle** by 1.5-2.0× | §3.1b, §3.2, §3.5b(i) |
| **Is anything established?** | **A bound, and a mechanism.** (i) Over 21 matched points the raw arena never missed a deployed effect > **0.67 test items**, and nothing in either arena reached +0.010. (ii) The arenas share the whole decision path below retrieval on channel (b) but use **different fusion operators** on channels (a)/(d) | §3.1b, §3.7c |
| **Is the "negatives informative, positives not" asymmetry established?** | **NO — and I decline to certify it.** The positive-side evidence is real but **one-sided by construction** (§3.4 R2) and has a simpler explanation that is not an arena property at all: **selection / winner's curse** (§6(a)) | §3.4, §3.5, §6(a) |

---

## §4. TASK C — the cheapest possible direct validation, and what it costs

### 4.1 What IS replayable at `$0` in the deployed space

* **Deletions from the memory bank, MHC-EN only, 4 seeds, exact.**
  `ERRPAT_MHC-EN_2026-07-26.md:570-576` (verified verbatim this session): *"`p2_out/cache_MHC_s{0..3}.json`
  banks the top-60 neighbour lists **in the deployed archive-kNN key space for all 4 deployed seeds**,
  which supports **exact, `$0`, multi-seed, deletion-only** bank replay … The limitation is real but
  narrower than F78 states: it applies to bank **additions**, **key-space changes**, and **re-training**,
  not to pruning."* The four files exist (`scripts/analysis/p2_out/cache_MHC_s{0,1,2,3}.json`, 409 KB
  each, 2026-07-06), plus `cache_MHC_zh_s{0,1,2}.json` (388 KB each). **Both claims in the tasking are
  confirmed.**
* **Any re-weighting or truncation of an already-banked top-k list**, all datasets — this is exactly
  what F94 did, and what §3.1 re-mined.
* **Re-reads of banked deployed reads** (`mechfix_*`, `ksweep_*`, `errpat_*` OUT jsons). Free, no new
  test touch, because F88/F89/F94 spent those touches.

### 4.2 What is NOT replayable at `$0` — and its true price

* **Anything that changes the key space or adds bank rows** needs the head — **and the deployed head
  inventory is gone, not merely HateMM's.** F107's instrument inventory extends F78 to the whole
  estate: `find logging/Retrieval -name '*.pt'` returns **228 files in 6 run dirs, ALL of them
  `mntp_s1_cpuhead`** (the F92-dead bidir heads); **97 empty `ckpt/` dirs**; and **0 of the 9** P2-era
  deployed checkpoints named at `p2_rerank_eval.py:55-63` exist (F107 body,
  `HEADCOV_PREGATE_RECORD.md`). This is why HEADCOV could only run on CPU re-mint proxy heads on ZH
  dev, and it is the binding constraint on Task C: **any future head-space validation must budget a
  head re-mint plus its own fidelity gate, on every dataset, before it measures anything.**
* **But the head is now a CPU object, and this is the tasking's `52 s` claim — verified.**
  `ERRPAT_HateMM_2026-07-26.md:526-529` (verbatim): *"The align head **trains and evaluates
  end-to-end in 52 s of wall time on 8 CPUs** (30 epochs, 60 retrieval evals, on banked features) — the
  3-seed proxy family cost ~2.6 CPU-minutes total and zero GPU. F78 priced a faithful curation pregate
  at '~0.3 GPU-h head re-mint'; that estimate can be replaced by **~1 CPU-minute per seed**."*
  It is recorded **for HateMM only**; ZH has a demonstrated CPU re-mint path
  (`scripts/analysis/errpat_remint_dumps/errpat_zh_remint_seed{0,1,2}.pkl`) and EN an ARM-F snapshot
  recompute (`KSWEEP_RECORD.md:78-80`).
  **Binding caveat, from the same passage (`:533-536`):** *"CPU-trained heads are not bit-exact to the
  CUDA floor (−0.0031 final-epoch acc here), so a CPU-trained arm must be paired against a **CPU-trained
  floor**, never against the banked GPU floor."*
* **Test contact.** The campaign's rule is **one budgeted test-touch per pre-registered question**
  (`B2_PREREG_REVIEW.md:306`; `CAND2_REP2_VERDICT_REVIEW.md:157`; `BIDIR_STAGE1_PREREG.md:538`). A new
  operator evaluated on test is a new touch and needs a prereg. **This recon spent none.**
* **Dev is free but under-resolved.** Dev n = 107 / 78 / 80 ⇒ one item = 0.0093 / 0.0128 / 0.0125.
  `ERRPAT_MHC-EN_2026-07-26.md:568` states the wall outright: *"at dev n=80 one item is 0.0125, so
  **dev cannot resolve a +0.009 effect**."* And on ZH, dev is *anti-correlated* with test
  (`ERRPAT_MHC-ZH_2026-07-26.md:98`, −0.2402, p = 0.0380). **A dev-only validation can resolve the
  +0.030 bar (2.4-3.2 items) but not the +0.010 interest line, and on ZH it should not be trusted at
  all.**

### 4.3 The design, and what was actually run

The tasking asked for a mix — one strong positive, one clean null, one negative — measured in the
deployed head space using banked lists. **That design already existed in the repo as banked data and I
ran it rather than re-deriving it:** §3.1's k-grid supplies 21 points spanning a large negative
(k ≤ 3, −0.04), a clean null (k = 15, 0.000) and small positives (k = 10-15 raw, +0.003 to +0.004);
§3.2 supplies the strong raw positive with its deployed twin; §3.3/§3.5b supply the matched stream
pairs; §3.7c supplies the training channel.

**And §3.1b is the reason this was the right thing to run and the wrong thing to trust.** The banked
route is the *only* `$0` route, and it delivers a grid whose informative half is inside the deployed
arena's noise floor. **The honest cost statement is therefore: a validation that could actually
adjudicate the +0.010-to-+0.030 range does not exist at `$0`, and here is what it would cost.**

| what | resolution it buys | cost | test touch |
|---|---|---|---|
| **Banked replay (done here)** | cannot resolve below ~1 test item = 0.0047-0.0067; the disagreements all live there | **`$0`** | none |
| **CPU head re-mint + same-path floor, dev read, 3 seeds × 3 datasets** | dev 1 item = 0.0093 / 0.0128 / 0.0125 ⇒ resolves the **+0.030 bar**, not the +0.010 line; ZH dev is anti-correlated with test and should not be trusted | ~52 s/seed/dataset ⇒ **≈10-15 CPU-minutes**, **plus a fidelity gate per dataset** (the inventory is gone, §4.2) | none |
| **Test read of a new operator** | 1 item = 0.0047-0.0067, 3-4 seeds | CPU-cheap, but **one budgeted test touch and a prereg** | **yes — 1** |

**What is still worth buying, in cost order:**

1. **`$0`, done — CURDIAG.** Discharged during this session by the MAC/provenance line; see §3.7c. It
   need not be run.
2. **~15 CPU-minutes, dev only — a proper matched stream/head pregate on all three datasets**, with
   CPU-trained same-path floors per `ERRPAT_HateMM_2026-07-26.md:533-536`. This is the cheapest thing
   that would turn §3.5b's two pairs into six and give the asymmetry question enough n to be answerable.
   **It is the single highest-value `$0` item left on this line**, and I did not run it because it
   requires training heads, which my brief forbids.
3. **One test touch, prereg-gated** — only if (2) returns something above the dev resolution.

---

## §5. FOLDING IN THE F105 DETERMINISM ERRATUM (kept separate, as instructed)

The determinism defect asks whether the arena reproduces **itself**; this record asks whether the arena
predicts **deployment**. They are orthogonal, they compound, and both bear on how much the recent
closures can carry.

**Verified from the primary source, `VSW_PREGATE_RECORD.md:514-582`:** the frozen
`mechnov_pairverify.py` (sha `77b0defd…b7240d`), re-run unmodified on the same node, env, caches and
seeds, *"reproduces **every closed-form quantity** of its own recorded cell exactly at 4 dp and **fails
to reproduce its torch-fitted MLP arm on 44 of 48 trained quantities**"* (`:522-523`); four diagnostics
exonerate the harness (`:531-540`); residual cause oneDNN/MKL kernel selection (`:546-548`).
Consequence (`:562-565`): F95's *"Δ ≥ +0.010 achieved by 0 of 36 cells"* **does not reproduce** — HateMM
× fused × MLP × mean-top-3 moves **+0.0054 → +0.0107** and clears. And (`:576-577`) *"F97's '78/78
parity' was true when made and **would not re-assert today**"*.

**And it is now settled as NON-DETERMINISM, not a defect — which shrinks the blast radius rather than
enlarging it.** `PROVENANCE_AUDIT_2026-07-28.md:396-411` exonerates every candidate cause: *"every
source is pinned"* (`mechnov_pairverify.py:171` `torch.manual_seed(0)`, `:177`
`np.random.RandomState(0)`, `:210` `StratifiedKFold(random_state=0)`, `:227` `PCA(random_state=0)`,
`:233` per-fold `RandomState(0+fold)`); threads pinned at `:413` `torch.set_num_threads(8)`; library
versions unchanged since 2026-03-27. Residual cause: *"cross-session CPU GEMM kernel dispatch
(oneDNN/MKL) on the 256-core EPYC"* compounding through Adam. `PROVENANCE_AUDIT_2026-07-28.md:31`:
*"**Zero verdicts move**; one headline *count* ('0 of 36') is session-dependent."*

**Interaction with this record — four statements:**

1. **§3.1 is immune.** `FIXK_k` and the deployed vote are closed-form; my own parity assert (`FIXK_20`
   changes 0 items, 9/9 cells) confirms it in-session. §3.1b's collapse is therefore a **real property
   of the data**, not a reproducibility artefact — it cannot be explained away as drift.
2. **§3.2 is largely immune.** `THRESH_best` is a grid search over a scalar, not a torch fit; D1 is a
   closed-form logistic. `C3_net` (used only as context) **is** exposed.
3. **The two problems are now cleanly separated and should stay that way.** Reproducibility: settled,
   zero verdicts move. Predictiveness (this record): **unvalidated**. Only the second is open.
4. **They compound in exactly one place** — F95's and F97's *trained* arms, which are both
   session-dependent and arena-dependent. F96's and F98's decisive bars (RESTRANS's degeneracy at
   95.03/97.75/99.45 %; AGGNET's DEG-A 0.9570 / DEG-B 0.9610) are **within-arena, within-session,
   closed-form agreement counts** and are exposed to neither problem. **That is the property that
   actually protects the recent kills — not any claim that the arena predicts deployment.**

**One correction that changes the arithmetic in this record.** The exchange-rate screen (ER ≥ 1.2) is
refuted — VSW reached ER 6.0000 and still failed (`VSW_PREGATE_RECORD.md:1004, 1037-1040`: *"The exchange rate
reaches **6.0000** on HateMM … Anyone citing 'the exchange rate never exceeds ~1.2' as a law of this
system must stop"*). §3.2's D1 numbers
are quoted with their ERs because the source records them, **but the ER is not the screen**. Under the
correct law `net = changed × (2·precision − 1)` (`VSW_PREGATE_RECORD.md:790`) against the binding requirement of
**22.3 / 17.4 / 16.5 net items**, D1's HateMM net of **+16 (fused) / +21 (text)** is 72 % / 94 % of
requirement on one dataset and unmeasured on the others — so even taken at face value it never was a
pass. **That strengthens §3.2's reading: the instrument's one false positive was not even a false pass
under the corrected law, only a false signal.**

---

## §6. TASK D — VERDICT

### (a) Is the raw-train-space pregate a valid instrument?

**UNVALIDATED.** Not "valid", not "invalid", not "anti-correlated" — **not established**, and the
evidence is not close to sufficient to establish it either way. I reached the opposite conclusion
mid-session and withdrew it; the withdrawal is §3.1b and it was forced by my own frozen robustness
check, not by argument.

**What the measurements actually say:**

* **The one large matched grid does not survive.** Pooled `rho_S` +0.758 over 21 points → **−0.3039**
  over the 12 non-degenerate points, because 9 of the 21 sit in a block where the operator is
  algebraically 1-NN and both arenas agree for a reason that has nothing to do with the arenas
  (§3.1b). Per-dataset on the informative half: **+0.40 / −0.95 / +0.95**.
* **The two matched stream pairs both fail to predict** — ZH shrinks 2.3× (+0.0156 → +0.0067) and
  HateMM AUC **inverts** (+0.011 train-LOO → −0.011 dev), with 2 of 3 encoders inverting (§3.5b).
* **The training channel splits**: valid on the curriculum cell (same sign, same ordering, ~2×
  attenuating) and **anti-correlated across datasets** on encoder adaptation (§3.7c).
* **Everything informative is inside the deployed arena's noise floor.** One test item is
  0.0047-0.0067 against a ±0.014 seed band; every non-degenerate paired point is −1.7 to +0.7 items.
  **The deployed arena cannot adjudicate at the scale where the campaign's decisions are made.**

**Why I decline to certify "arena negatives are informative; arena positives are not", despite
recommending that people act as if it were true.** Three reasons, in increasing order of importance:

1. **n.** After retractions, the positive-side matched evidence is **two pairs and one cell**, one pair
   in a different metric. That is a pattern, not a law.
2. **The sample is one-sided by construction.** Law-I's nine data and the P3/P8/S2S probe precedents
   exist *because someone paid to take a raw positive to a deployed measurement*. Nobody has ever taken
   a raw **negative** to a deployed measurement — there is no reason to. **So the corpus can only ever
   exhibit the failure of positives.** Its unanimity is partly a sampling fact (§3.4 R2).
3. **There is a simpler explanation that is not a property of the arena at all: selection.** Raw
   positives are *found by searching* — over operators, hyperparameters, spaces and datasets — and
   negatives are not. Regression to the mean under selection predicts exactly the observed pattern in
   **any** arena, cheap or expensive. The campaign has already measured this directly: Wall 4
   (`VSW_PREGATE_RECORD.md:609-622`) shows **one global hyperparameter costs 86-100 % of the found
   effect on 2 of 3 datasets**, and F108's stream weight degrades from a 2-of-3 conjunct under full
   hindsight to **1-of-3 when made deployable**. **If selection explains it, then "the arena is
   optimistic" is the wrong lesson and "price your selection" is the right one** — and the wrong lesson
   generalises wrongly, e.g. by licensing raw-arena nulls in channels the arena has never been tested in.

**What IS established, and it is not nothing:**

* **A bound.** Over 21 matched points the raw arena never missed a deployed effect larger than
  **0.67 test items**, and nothing in either arena reached +0.010. This is a max over the full set and
  is robust to point selection — unlike the correlation.
* **A mechanism, verified in source (§3.7c).** On **channel (b)** the arenas share the entire decision
  path below retrieval — same ordered label tuple, same rank weights, same threshold. On **channels
  (a)/(d)** they do not even share a fusion operator: raw = `L2norm(concat(L2norm(img), L2norm(txt)))`
  (`MECHNOV_PAIRVERIFY_PREGATE.md:150`), deployed = Hadamard on learned projections
  (`src/model/classifier.py:87,140-141`). **Elementwise product versus concatenation.** So the encoder-axis
  inversion is predictable *a priori*, and channel-(b) results have a structural reason to travel that
  channel-(a) results do not. **This is the most transferable thing in this record.**
* **A premise correction.** §0.2: the arenas differ by +0.0965 / +0.0435 / +0.0358 on train LOO, not by
  0.998-vs-0.84 — the CLIP-vs-Qwen conflation.

**Is MHC-EN a special failure site?** The relayed claim was that EN fails in 2 of 3 pairs. **My
measurements say no.** The EN "sign inversion" is a false pair (§3.6, independently retracted by F108's
author). In the k-grid, **EN is the *best*-correlated dataset** on both the full grid (+0.9903) and the
non-degenerate half (+0.9487); **MHC-ZH is the worst (−0.9487)**. EN *is* the failure site on the
encoder-adaptation axis specifically — and F44's measured mechanism (EN's Qwen image stream collapses;
the equal-weight concat cancels the text gain) already explains that as a **dataset** fact meeting a
**fusion-operator mismatch**, not as an arena pathology. **"EN is where the arena fails" is not
supported.**

### (b) Which kills are at risk, in which direction, ranked

**Correcting my own earlier framing:** I cannot say the risk is "almost entirely false-PASS", because
that phrasing presupposes the asymmetry I have just declined to certify. What I can say is narrower and
better supported: **the recent kills do not rest on the instrument being predictive in the first
place** — their decisive bars are within-arena, within-session relative comparisons — and the one
quantitative bound available (§3.1b) is consistent with, though far from proof of, a low false-kill
rate. Ranked by how much would change if the instrument were wrong:

| rank | closure | direction of risk | how much would change |
|---|---|---|---|
| 1 | **The raw-arena POSITIVES that have been quoted forward** — F98's HateMM +0.0134, F96's D1 +0.0215/+0.0282, VSW's +0.0255 ("85 % of the bar") | **Over-read — one of them measured so.** §3.2: for the threshold family the raw *legal* number exceeds the deployed *gold-cheating* ceiling by 1.5-2.0× | **Nothing to the verdicts** — all three were KILLs. But the *narrative* "the campaign nearly converted on HateMM" is unsupported and should stop. This is the largest real change |
| 2 | **F95 MECHNOV's trained arms** | Both problems overlap here: session-dependence (`VSW_PREGATE_RECORD.md:562-565`) **and** an untested arena | Verdict stands on closed-form terms (`:568-572`); the *count* "0 of 36" must not be re-quoted |
| 3 | **F96's kill of C1, F98's kill of C3** | **FALSE KILL — low risk, and for a reason that does not need the instrument.** Both decisive bars are **within-arena degeneracy agreement counts** (95.03/97.75/99.45 %, `RESTRANS_PREGATE_RECORD.md:312-314`; DEG-A 0.9570 / DEG-B 0.9610, `AGGNET_PREGATE_RECORD.md:491`) — statements that the treatment *is* a measured-dead twin, not that its Δ is small | Little. `RESTRANS:454-456` additionally argues the degeneracy is **stronger** in head space |
| 4 | **F97 VGA/VNQ** | Its decisive bar K-VGA-3 is a **relative** comparison inside one session and one arena (`VGA_PREGATE_RECORD.md:422-427`) | Little |
| 5 | **F99 RDK, F101 BSY pre-closures** | Inherit the arena; both are *arithmetic* over banked raw quantities | Little; they inherit §3.2's caution wherever they price a positive |
| 6 | **F94's k-axis closure** | **Lowest risk of all** — it is the one axis measured in **both** arenas, and both say dead: max Δ anywhere is +0.0040 (raw) / +0.0045 (test) over 21 points | Nothing |
| 7 | **Anything relying on the arena in a channel it has never been tested in** | **This is the real exposure and it is prospective, not retrospective.** §3.7c shows the arenas do not share a fusion operator on channels (a)/(d) | A future pregate that assumes channel-(b) fidelity carries to a representation- or training-side operator would be unwarranted |

**What I flagged mid-session as the one genuinely open exposure — the MHC-EN text-only +0.0310 (§3.6) —
was closed the same day by F108 / STREAMCOMP, and it closed in the direction this record predicts.**
Its deployable stream weight delivers **−0.0027 / +0.0346 / +0.0200**, the conjunct on exactly one
dataset, against a full-hindsight 2-of-3. **Nothing is left open on the pass side.**

**One cross-record item I am flagging rather than adjudicating, because it is another record's to
rule on.** F107 / HEADCOV's metric-channel argument (Q1) rests in part on *"F47's head train-LOO
**0.998** shows the objective is ALREADY at its optimum on its own training signal with **≤0.002
headroom**"*. Per §0.2, **0.998 is the CLIP head**; the deployed Qwen heads measure **0.9406 / 0.8915 /
0.8154**, i.e. training-signal headroom of **0.06 / 0.11 / 0.18**, not ≤0.002. This does not touch
F107's measured bars (K-HC-3 1.0000, K-HC-1 coverage 0.9829 ⇒ oracle ≤ +0.0171, and the Q3 regression
bound +0.0286 at the upper 95 % CI — all of which stand on their own numbers), but the "already at its
optimum" step of the Q1 *argument* is weaker than stated. **Referred to that record's author.**

### (c) The cheap standing validation every future pregate should carry

**Yes — but it must be a clause that is SAFE under "unvalidated", not one that assumes the asymmetry.**
Written up as `refine-logs/PREGATE_CALIBRATION_CLAUSE.md`. The version I drafted before §3.1b was
computed made two mistakes I am recording so they are not repeated: it asserted a validated false-kill
rate, and its anchor gate (`rho_S` ≥ 0.60) is **gameable by the degenerate block** — the very artefact
that inflated my own headline would let a broken arena instance pass. Both are fixed. Summary:

> **CAL-0 (the standing statement).** *"The raw train-space arena is **not established** as predictive
> of deployed effects. It is used because it is the only `$0` arena available, and its results are
> reported as **arena results**, never as predictions."* Every pregate carries this sentence.
> **CAL-1 (asymmetric reading — a PRUDENTIAL rule, explicitly not a validated one).** Read a raw-arena
> **null/negative** as a kill only when the decisive bar is a **within-arena relative comparison**
> (degeneracy twin, isomorphism control, best-fixed-profile control) rather than an absolute Δ. Read a
> raw-arena **positive** as a *ticket to a deployed measurement*, never as an effect size.
> **CAL-2 (the anchor, `$0`) — with the degenerate block EXCLUDED.** Report the `FIXK_k` grid on your
> own folds. **`FIXK_20` must change 0 items** (else VOID). Then state the Spearman against F94's
> banked deployed curve **over k ∈ {5,7,10,15} only** — k ≤ 3 is algebraically 1-NN
> (`KSWEEP_RECORD.md:29-31`) and must be excluded. **Reference value: pooled −0.3039** (+0.40 / −0.95 /
> +0.95 per dataset). **This anchor is a provenance check on the fold draw and the caches, NOT a
> validity gate — it currently has no threshold to pass, because the arena has not been validated.**
> **CAL-3 (the positive-side gate).** Any raw Δ ≥ +0.010 must be reported with the deployed space's own
> **gold-cheating ceiling** for the same family where one is banked. If the raw *legal* number exceeds
> the deployed *oracle*, label the arm **RAW-ARENA ARTEFACT** and do not escalate.
> **CAL-4 (closed-form vs trained).** Label every quantity; trained quantities carry the F105
> session-dependence caveat and counts over them are never quoted across sessions.
> **CAL-5 (channel declaration — the one with real predictive content).** State which channel the
> operator acts in. **Channel (b)** shares the entire decision path below retrieval with the deployed
> system. **Channels (a)/(d)** do **not share a fusion operator** — raw is L2-concat, deployed is
> Hadamard on learned projections (§3.7c). **A channel-(a)/(d) result carries no transfer warrant at
> all and must say so in its limitations.**

### (d) GAP-C — the two bans applied outside their derivations

**GAP-C1 — F66 applied outside its ruling. CONFIRMED, and the contradiction is verbatim.**

* `LITSWEEP5_COMPLETENESS.md:84`: *"**F66 caps it** — ArcFace is a **symmetric** embedding-geometry
  operator; the convertible ZH/EN headroom is 91-98 % selection-only, so it can recover at most
  **+0.001-0.006**."* The same reading is repeated at `LITSWEEP5_COMPLETENESS.md:13` (*"a trained
  **symmetric reshaper** on train labels = F75's object … and **F66 caps it at +0.001-0.006**"*).
* `NCA_FORENSIC_RECON.md:110`: *"**⇒ Ruling: F66 does NOT bind trained-space reshaping. The cell is not
  F66-dead — it is legitimately un-measured.**"* Its derivation (`:104-109`) is correct and decisive:
  *"**F66's arithmetic is conditional on a single fixed map φ₀** … Every number in F66 (+0.0776,
  +0.0012, +0.0764) is a **property of φ₀'s Gram matrix** … A different `φ′` yields a **different** Gram
  matrix, a **different** oracle headroom … F66 never measured φ′'s decomposition."*
* F99 applies the correct reading (`RDK_FORENSIC_RECON.md:47`, per LITSWEEP7 `:784-785`).

**Ruling: `NCA_FORENSIC_RECON.md:110` is right and `LITSWEEP5_COMPLETENESS.md:84,13` are wrong.** F66's
arithmetic bounds **inference-side symmetric re-weighting of a fixed Gram**; a trained loss produces a
new map, hence a new Gram, hence an object F66 did not evaluate.

**Was anything mis-routed? No candidate was killed *solely* by it — but the ledger is mis-priced.**
ArcFace/angular-margin is the only cell priced by the bad citation, and `LITSWEEP5_COMPLETENESS.md:84`'s
own verdict line rests it on **two independent grounds that survive**: F75's measured 0/8 formal
(*"the first measured negative for trained-reshaping-unlocks-oracle-headroom"*) and D7. So the routing
outcome is right and the arithmetic offered for it is not. **Consequence: any future training-side
proposal must be priced under the F99/NCA reading, and `+0.001-0.006` must not be quoted as a cap on
trained reshaping again.** This is a real defect with a null blast radius — recorded, not inflated.

**GAP-C2 — F49's `q > 0.663`. CONFIRMED, and it was already adjudicated a week ago and never landed.**

* Derivation verified at `MJ_FORENSIC_RECON.md:36-63`: **MHC-EN dev, N = 80**, 3-seed mean disagreement
  **D = 21**, always-Qwen prior **p_Q = 0.588**, `gain(q) = 0.2625·q − 0.15415`, and
  `0.2625·q − 0.15415 ≥ 0.020 ⇒ **q ≥ 0.6634**`. **Note the bar it clears is +0.020, not the campaign's
  +0.030** — so the constant is mis-scoped twice over.
* Promoted unconditionally at `directions_tried.json:179`: *"F47 carve-out now requires demonstrated
  **alignment > 0.663** from banked evidence BEFORE any gate"*, and re-used as a campaign constant at
  `LITSWEEP5_COMPLETENESS.md:13,77,131,133,138`, `ISR_PREGATE_RECORD.md:64,116`,
  `SEG_REENCODE_FORENSIC_RECON.md:106`, `ERRPAT_HateMM_2026-07-26.md:413`.
* **No HateMM or MHC-ZH re-derivation exists anywhere in the repo** — every one of those citations
  traces back to the single MHC-EN-dev arithmetic. Confirmed by grepping `0.663` across
  `refine-logs/`, `research-wiki/`, `autoresearch/`, `scripts/`.
* **This is not a new finding.** `REDTEAM_BAN_SCOPE_AUDIT.md:173-208` recorded it as **GAP-4** with the
  verdict *"**INDUCTIVE LEAP** (dataset-specific number + logical catch-22)"* (`:190`), noted the
  catch-22 that *"a genuinely-new source **cannot** show its alignment from banked evidence — by
  definition — so the F49 bar makes F47's own carve-out **unenterable on arithmetic**"* (`:196-198`),
  prescribed the `$0` remedy (`:204-208`: run F47's own banked router gate instead), and **ranked it
  #2 of 7 gaps, `$0` CPU, in-box** (`:365`). **It was never landed, and the ledger still carries the
  unconditional bar.**

**Ruling: the `q > 0.663` bar is MHC-EN-dev-specific arithmetic against a +0.020 bar and must not be
used as a campaign constant.** Mis-routing check: the only *direction* killed by it is F49/MJ itself,
where the kill is independently safe — the modality-locus alignment ceiling is `a ≤ 0.588`
(`REDTEAM_BAN_SCOPE_AUDIT.md:177-178`), so a **perfect** judge cannot clear even a re-derived bar of any
plausible value. **No live or dead candidate is mis-routed today; the exposure is prospective** — the
bar pre-kills future router inputs on arithmetic that does not apply to them, and the fix is to run
F47's `$0` gate instead, exactly as the redteam audit already prescribed.

*(Recorded, not adjudicated here: LITSWEEP7 `:797-801` also notes F94's upward-k ban is measured on
MHC-EN only. §3.1 partly repairs this — the raw arena now supplies HateMM and ZH k-curves too, both
monotone toward k = 20 — but it does not measure k > 20 on those two datasets either.)*

---

## §7. LIMITATIONS

0. **THE BIGGEST ONE, AND IT IS ABOUT THIS RECORD'S OWN FIRST DRAFT.** I initially reported a
   validated instrument on the strength of §3.1's +0.758. That was wrong, and it was wrong in the
   *convenient* direction — it would have reassured the campaign about ten closures. It was caught only
   because I ran a robustness check I had not pre-registered (§3.1b). **The lesson for the campaign is
   the general one: a correlation over a grid that contains an algebraically degenerate block is not a
   correlation.** Anyone re-using §3.1 must use the k ∈ {5,7,10,15} row.
1. **§3.1 covers ONE operator family**, and after §3.1b what it establishes is a **bound**, not a
   correlation. Free re-weighting, subspace residuals, membership change and map training are not
   covered. The "no missed effect > 0.67 test items" statement is **on that family**, not a
   campaign-wide guarantee — and it is weak, because the deployed arena could not have resolved a
   larger effect at that scale anyway.
2. **The deployed side of §3.1 and §3.3 is proxy-grade on 2 of 3 datasets** (HateMM and ZH CPU proxies,
   EN ARM-V exact). `MECHFIX_PREGATE_2026-07-27.md:460-464` and `KSWEEP_RECORD.md:118-123` both flag it;
   deltas are within-cell so the proxy↔floor offset cancels, but the absolute numbers are not floor-grade.
3. **§3.1 pairs a 5-fold train-LOO Δ against a 3-4-seed test Δ.** Different estimators, different
   variance structures (5 folds vs 3 seeds; n = 744/579/549 vs 215/149/161), plus the ±0.014 seed band.
   Noise attenuates a correlation toward zero — **so the −0.3039 is a lower bound in magnitude on the
   true relationship only if that relationship is monotone, which is exactly what is in question.** I
   claim no direction from it; I claim only that +0.758 is not evidence of validity.
4. **Every point in §3.1 is a negative or a near-zero.** The k family contains no large positive in
   either arena, so §3.1 **cannot** speak to positive transfer at all.
5. **§3.2's mechanism hypothesis is not established** (Pearson +0.198 / Spearman +0.317 over 9 cells).
   The *finding* — raw legal exceeds deployed oracle — does not depend on it.
6. **`THRESH_best` and D1 were not re-run this session.** Both are closed-form and therefore expected to
   be on the deterministic side of F105, but I did not re-execute them to prove it. Their values are
   re-read from `aggnet_pregate_OUT.json` and `RESTRANS_PREGATE_RECORD.md:425-428`.
7. **Two pairs I originally used are RETRACTED** — MHC-EN stream (mismatched baselines, §3.6) and
   F91/Molmo2 (unnormalised concat, §3.4 R1). Both were caught by construction-checking the key. **Any
   future pair must have its key construction verified in source before it is quoted**; two of the four
   candidate pairs offered to this investigation failed that check.
7b. **The channel-(d) evidence is second-hand.** §3.7c relays `PROVENANCE_AUDIT_2026-07-28.md`; I
   verified its cited lines and its arithmetic but did not re-run its computation.
8. **`VSW_PREGATE_RECORD.md`, `VSW_ASYMMETRY_RECON.md` and `STREAMCOMP_FORENSIC_RECON.md` were live at
   read time** (other agents own them). **All line numbers cited into those three files are as-read on
   2026-07-28 and may drift**; the quoted text, not the line number, is the anchor. Every *number* I
   take from them is independently re-derived from `aggnet_pregate_OUT.json` or another banked artifact
   and cited as such, so no result here depends on a live file.
9. **No test-split file was opened and no held-out test metric was produced.** Every deployed number is
   a re-read of an artifact whose test touch was spent by F88/F89/F94.

---

## §8. FILE MANIFEST

| path | contents |
|---|---|
| `refine-logs/INSTRUMENT_VALIDATION_RECON.md` | this record |
| `refine-logs/PREGATE_CALIBRATION_CLAUSE.md` | the reusable CAL-1…CAL-4 clause (Task D(c) deliverable) |
| `<scratchpad>/ivr_freeze.md` | frozen design, sha256 `90e9ea26…0cd7647` |
| `<scratchpad>/ivr_pair.py` | frozen measurement script, sha256 `738a167a…9a9fe14d` |
| `<scratchpad>/ivr_pair_OUT.json` | machine-readable output of §3.1 |

**Read-only inputs** (no file below was modified): `scripts/analysis/{aggnet_pregate_OUT.json,
ksweep_OUT.json, mechfix_{hatemm,zh,en}_OUT.json, errpat_hatemm_ceilings_OUT.json}`;
`refine-logs/{MECHFIX_PREGATE_2026-07-27, KSWEEP_RECORD, MECHNOV_PAIRVERIFY_PREGATE,
RESTRANS_PREGATE_RECORD, VGA_PREGATE_RECORD, AGGNET_PREGATE_RECORD, VSW_PREGATE_RECORD,
VSW_ASYMMETRY_RECON, LITSWEEP7_LANDING_SITE, LITSWEEP5_COMPLETENESS, REDTEAM_BAN_SCOPE_AUDIT,
MJ_FORENSIC_RECON, NCA_FORENSIC_RECON, RDK_FORENSIC_RECON, BSY_FORENSIC_RECON, EUM_FORENSIC_RECON,
TVB_FORENSIC_RECON, MOLMO2_PROBE_RECORD, ERRPAT_{HateMM,MHC-EN,MHC-ZH}_2026-07-26,
CAND2_KC20_{HateMM,MHC_zh}.json}`; `autoresearch/goal_mllm_plus3/state/{findings.jsonl,
directions_tried.json}`; `research-wiki/{EXP_p8_semantic_compression, CAMPAIGN_mllm_method_role,
DRAFT_analysis_chapter, experiments/exp-encoder-3seed}.md`; directory listing of
`scripts/analysis/p2_out/`.

**Required statements.** ZERO GPU / SLURM / Modal / training spent. No held-out test metric read or
produced; no test-split file opened; no new test touch consumed. No prereg, config, or frozen artifact
mutated. `findings.jsonl` received one appended row (this finding) and nothing else.
