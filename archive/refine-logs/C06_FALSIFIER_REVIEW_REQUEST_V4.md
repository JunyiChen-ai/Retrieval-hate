# C06 `$0` falsifier — fresh independent design review request, **ROUND 4**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM
job, no login-node training run, no Modal, no GPU, no cache write, no commit, no edit to
`TARGET_STATE.json`. Read-only numpy/torch-CPU re-derivation on already-banked **train-split**
caches and banked mint checkpoints, plus `sha256sum`, is expected and encouraged.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V4.md` (v4, unfrozen).

You are a **fresh reviewer**, independent of rounds 1–3 and of the designer. You have not seen
the designer's reasoning and must not ask for it.

**Prior rounds and the trajectory.** Round 1: REVISE (3C/6H/10I+4M). Round 2: REVISE
(3C/3H/7I+3M) — and its disposition audit found **three of twenty-three** round-1 findings
recorded as `ADOPTED` were not actually repaired. Round 3: REVISE (2C/1H/6I+4M), with a **clean
disposition sweep** (16/16 round-2 adoptions real, all three reopened items repaired, no
disguised rebuttal). Both round-3 Criticals were calling-convention or single-constant defects
found **by executing C01's frozen code**, and round 3 judged that *"neither Critical requires a
GPU, an extraction, or a redesign … the instrument, once they are fixed, does measure what
`falsifier_spec` asks."*

**So: audit by execution, not by reading.** Treat v4 §14's cumulative disposition table as an
index of things to check, never as evidence. *Any claimed repair the artifact does not contain*
is Critical. Every round so far has found its Criticals in the seam that the previous round's
repair opened — look there first.

**Nothing is implemented yet.** No battery script exists. This reviews the **design** only.
Several v4 claims are about code that does not exist (the shared mint driver, the tie diagnostic,
`GATE-POP`): judge them as *specifications*, say whether they are checkable, and check them
against §13's twelve-item code-lineage handoff.

---

## 0. What C06 is, so you need no prior context

C06 (*Prompt-Orbit Tangent/Curvature*) claims the tangent and curvature of a video's
representation across a fixed prompt orbit encode policy-bound instability no single prompt
captures. It is **not** an active candidate: its registry status is
`gated_on_zero_cost_falsifier`. An earlier candidate, C01, measured the two-point case in a
**raw-key** arena and found the best of six matched-norm orthogonal rotations of the prompt
endpoints **matched or beat** the real prompt displacement on both datasets. Because the registry
says a raw-key arena *"may kill but may not promote"*, the Gate-0 adjudicator gated C06 rather
than striking it: re-run C01's battery in the **fold-head** arena on already-banked caches. If
the rotations again match, C06 closes for `$0` and an authorized `1.7–2.5 GPU-h` extraction is
never spent; if not, C06 has earned that extraction.

---

## 1. Read first, in this order

1. `CLAUDE.md`, `AGENTS.md`.
2. `TARGET_STATE.json`: `gate0_reopen_2026_07_31.dispositions.gated[0]`;
   `iteration_8_queue_state_2026_08_04`;
   `process_rule_compute_projection_and_heartbeat_2026_08_04`;
   `iteration_8_stage0_bounded_extraction_amendment` — read the C06 text **verbatim**.
3. `TARGET_FINDINGS.md` — **F118** in full (including its erratum on boilerplate describing a leg
   that did not run); skim **F88**, **F113**.
4. `refine-logs/GATE0_REOPEN_2026-07-31.md` §4.4;
   `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §3;
   `refine-logs/C09_A0_V17_RECORD.md` §2 and §8.1.
5. All three prior reviews in full: `C06_FALSIFIER_PREREG_REVIEW{,_R2,_R3}.md`.
6. The superseded drafts for context: `C06_FALSIFIER_PREREG_DRAFT{,_V2,_V3}.md`.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V4.md` — **the artifact under review**.

**Primary sources for the code claims:** `scripts/analysis/c01_policy_contrast_a0.py`
(`l2_rows:1183-1205`, its mask assertion `:1193-1194` and its `None`-normalisation `:1187-1188`;
`fuse_modalities:1208-1217`; `paired_key:1220-1239`; `contrast_blocks:1242-1270`;
`orthogonal_blocks:1272`; `prepare_views:1294` and its derived-mask check `:1381-1386`; C01's own
mask construction `:2224` and call site `:2303-2306`; `die:392-393`; the deferred-import site
`:387` / `import_compute_modules:1048`), `src/model/classifier.py` (`:80-81` the biased
projections, `:115-124` the forward), `scripts/analysis/headspace_mint.py` (`:106-116` the guard,
**`:192-194` the resume skip**, `:199` the unconditional dev load, `:203-216` fold parity,
`:219-227` the dummy, `:322` `lab_dev`), `scripts/analysis/c02_a0_mint.py`,
`scripts/analysis/mechfix_ops.py`, `scripts/analysis/mechnov_pairverify.py`,
`configs/c01/c01_a0_v{2,3,4}.json` (including `zero_contract_v2` and
`output.decision_schema.required_halt_only_validity_guards`).

---

## 2. Verify these facts yourself

Report `VERIFIED` / `MISMATCH` with the value you obtained, for **every** item.

| # | claim | where |
|---|---|---|
| **V1** | All 21 sha256 digests in §11 match disk; both provenance chains sha-gated in source. | §11 |
| **V2** | **The C-1 repair.** `prepare_views` **dies** when handed `zero_mask = None` on *any* dataset (`:1381-1386` compares against the raw argument); succeeds with `np.zeros(n, dtype=bool)` on MHC-ZH; and on the **HateMM arena population `n = 743`** the explicit all-False array **succeeds**. Also that `l2_rows` *does* accept `None` (`:1187-1188`), so the two functions genuinely differ. | §3.7, §7.4(i)(j) |
| **V3** | **The C-2 repair.** HateMM arena `n = 743`, `pos = 297`, `neg = 446`, majority `446/743 = 0.600269 → 0.6003`, band `[0.6203, 0.98]`; MHC-ZH `399/579 = 0.689119 → 0.6891`, band `[0.7091, 0.98]`; and the full-population `0.5995` is used by **no gate**. | §3.7, §6.3, §7.1 |
| **V4** | The null-contract defect and its repair: `head_f(0,0)` non-zero at multiple seeds; `common_displacement` unbuildable in head space under either mask; all 13 head-space arms build at `n = 743`. | §7.4(a)–(g) |
| **V5** | **The row-subset identity.** Raw arms at `n = 743` are bit-identical to the `n = 744` one-hot build restricted to the 743 rows — `max\|diff\| = 0.000e+00`, all 13 arms — and every `ρ` unchanged. | §7.4(h) |
| **V6** | `GATE-C01PARITY`: the two-block builder reproduces `prepare_views` bit-exactly, both datasets, using §3.7's mask forms. | §7.6 |
| **V7** | `ρ*` at full precision `0.968176` / `0.977223`, runner-ups `0.964446` / `0.969686`, and all 26 `ρ_raw` values at 6 dp. Confirm the full-precision freeze removes the `endpoint_std` self-exemption round 3 identified. | §6.1 |
| **V8** | **The trained-head `ρ` reference:** over the 36 banked `artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz`, `ρ = ‖mean_i l2n(K_i)‖` gives HateMM min/median/max `0.447803 / 0.562434 / 0.632996` and MHC-ZH `0.340179 / 0.574247 / 0.667326`, with **0/18 cells above `ρ*`** on both. | §6.1 |
| **V9** | `GATE-FLOOR`'s six anchors on both metrics, equal to the banked `acc_deployed` / `mF1_deployed`. | §6 |
| **V10** | The Holm family: `(12+11) × 2 = 46` per `(dataset, lineage)`, `× 2 = 92` per dataset — and that §8 Phase 4's `92` is a **different product** that coincides. | §5.5 |
| **V11** | Every product in §8 and both totals (`2886.3 s`, `3607.9 s`), the mint share `86.9 %`, Phase 3 `9.5 %`, sensitivities `3160.0` / `3981.1 s`; and that the printed column sums to `2886.2` with Phase 7 carried at its `0.1 s` upper bound. | §8 |
| **V12** | **S6's vacuity:** `GATE-SELFTEST`'s identity plus `endpoint_std ∈ C` makes S3 imply `net ≥ 15` (HateMM) and `≥ 12` (MHC-ZH) against frozen minima of `3` and `2`. | §5.8 item 4 |

---

## 3. What you must assess

### A. The two round-3 Criticals — did they land, and did the repairs create new seams?

- **C-1 (the mask convention).** Confirm no `None` survives anywhere a `prepare_views` call is
  specified, and that each of §3.7's **four objects** carries a mask argument correct for its
  population. The load-bearing subtlety: at `n = 743` the all-False mask is correct **because the
  zero row is physically absent** — check that the design says why, rather than relying on it
  coincidentally.
- **C-2 (the arena majority).** Re-derive both constants. Then judge whether `GATE-POP`'s new
  class-count clause `(297,446)` / `(180,399)` genuinely makes the constant *checkable at run
  time* rather than merely asserted — and whether any **other** population-dependent quantity in
  the design still carries a full-population value. Round 3 said the majority rate was the only
  one; verify that independently.
- **Then look for the new seam.** Every round so far found its Criticals where the previous
  round's repair touched. Round 3's repairs touched: the mask convention, the majority constant,
  `GATE-POP`, `GATE-ROWSUBSET`'s name, `ρ*`'s precision, S6's status, the tie diagnostic, and the
  `RuntimeError` wrapper. Look there.

### B. The design's own self-caught defect (§12)

While drafting §15, the designer found that binding `dev_path_opens == 66` would **HALT a
legitimately resumed job**, because `headspace_mint.py:192-194` returns *before* the dev load at
`:199`. §12 now binds against a measured `mints_executed` plus a separate binding assertion that
all 66 `.npz` are present before the arena.

- Is that pair exactly predictable under **every** code path that can legally run — fresh,
  resumed, partially resumed, and a re-run after a HALT?
- Does anything **else** in the design assume a fresh run? Check §8's projection, `GATE-LEDGER`'s
  process count (`66 + 6 + 1`), §5.6's absence rule, and the heartbeat's `units done / total`.
- Is binding these counts at all the right call, given C09 merely reported them?

### C. The decision rule

- **S6 (§5.8 item 4, §15.4).** It is now disclosed as **implied by S3** and therefore unable to
  bind. Rule on whether a condition that cannot fail belongs inside a SURVIVE conjunction, or
  should move out of S1–S6 into the reported quantities. Note the tension: `GATE-SELFTEST` needs
  an object, and the Gate-0 record demands the net-item currency.
- **The tie cap (§6.5, §15.3).** The `1 %` cap (`≤ 7` / `≤ 5` items) is, by the design's own
  admission, **the only invented threshold in the document**. Rule on whether a cap is needed,
  and if so whether `1 %` is defensible or should be derived from something banked.
- Does the sharpened tie criterion (union of the two arms' top-21; max of the two residuals;
  collapse-and-recompute the rank-weighted vote) actually cover the in-set reordering mechanism
  round 3 identified, and can the report-not-HALT branch be reached by anything outside it?
- §5.5's multiplicity, §5.6's finiteness/absence/`RuntimeError` rules: still sound?

### D. Gates and scope

- v4 has **20 gates**, of which `GATE-DOMAIN` and `GATE-DEVFID` do not gate. Are any
  unfalsifiable, redundant, or able to fire on a **warranted CLOSE**? That last question has
  produced a Critical in two of three rounds (`GATE-ARENA` in round 3; `GATE-ORBITSCALE` in round
  1) — test it against every gate, not just the ones previously flagged.
- Is `GATE-ROWSUBSET`'s renaming plus the explicit statement that C01's vote-level property *has
  no object here* an honest resolution of round 3's I-1?
- Does §10.2 say everything a CLOSE is scoped to?
- Hard constraints: anything touched?

### E. The process rules

- **`rule_1_compute_projection`.** v4 claims the round-3 fixes are **compute-neutral**. Verify.
  Then **hunt for an uncounted loop** — rounds 1 and 2 each found some; round 3 found none
  material and noted Phase 7 now names the per-gate arithmetic. Check whether the new
  `GATE-POP` class-count clause, the wrapper, and the sharpened tie diagnostic carry any cost §8
  does not count.
- **`rule_2_heartbeat`.** Round 3 verified no interval exceeds ~15 s. Does the `RuntimeError`
  wrapper change that, and is §13 item 12 complete?
- §7.7 discloses that `U9` was originally a failure path recorded as a measurement. Could any
  remaining unit carry the same defect? Round 3 identified which units are independently
  corroborated and which are not.

### F. Honesty

- Does v4 claim any repair the artifact does not contain?
- Are the initialisation-dependent quantities (M-2, M-3) now stated as ranges rather than as
  digits no reviewer will reproduce?
- Is §7.3's blindness discipline intact — **no arm accuracy anywhere in v1–v4**?

---

## 4. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | The design, if executed as written, could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §2, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or its scope statement without inverting it. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**. Report the tally in that form.

**Do not grade on trajectory.** The finding count has fallen 23 → 16 → 13 across rounds; that is
not a reason to pass a design with a defect, and it is not a reason to manufacture one. C09
needed seventeen design rounds, and a separate code-review lineage then caught two wrong-verdict
paths that seventeen clean design rounds had missed.

---

## 5. What a GO does and does not authorize

A GO authorizes **nothing to run**. Before any job: (1) freeze with hashes; (2) a **separate**
independent code/resource review lineage over the executable reaching its own `0C/0H/0I`;
(3) main-dialogue authorization. A GO is not authority to write `TARGET_STATE.json`.

---

## 6. Deliverable

1. The `0C/0H/0I`-form tally and **GO** or **REVISE**.
2. Every finding with severity, `file:line` citation, and the concrete repair.
3. Your verification result for **all twelve** §2 items, with the values you obtained.
4. **A disposition audit of v4's §14 table, done by execution**, covering all 13 round-3 findings
   and the 4 Minors: `VERIFIED ADOPTED` / `NOT ADOPTED` / `PARTIAL`, with what you checked.
5. An explicit ruling on each of the five open issues in v4 §15.
6. An explicit ruling on §3.D's recurring question: **is there any gate that can fire on a
   warranted CLOSE?**
7. Anything to add to §13's twelve-item code-lineage handoff.
8. If you conclude the falsifier still cannot discharge the written condition at `$0`, say so
   directly and state what would be required instead. If you conclude it can and the design is
   clean, say **GO** plainly.

---

*Read-only. No GPU, SLURM, Modal, model load, head training, arena run, cache write, test-split
access, job submission or commit is authorized by this document, and `TARGET_STATE.json` must not
be modified.*
