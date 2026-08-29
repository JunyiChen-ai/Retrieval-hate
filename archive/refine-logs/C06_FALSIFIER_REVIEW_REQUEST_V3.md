# C06 `$0` falsifier — fresh independent design review request, **ROUND 3**

**Type:** read-only static design review. **No execution of any kind is authorized** — no
SLURM job, no login-node training run, no Modal, no GPU, no cache write, no commit, no edit to
`TARGET_STATE.json`. Read-only numpy/torch-CPU re-derivation on the already-banked
**train-split** caches, plus `sha256sum`, is expected and encouraged.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V3.md` (v3, unfrozen).

You are a **fresh reviewer**, independent of rounds 1 and 2 and of the designer. You have not
seen the designer's reasoning and must not ask for it.

**Prior rounds, and the lesson that should shape your method.** Round 1: **REVISE
(3C/6H/10I + 4M)**. Round 2: **REVISE (3C/3H/7I + 3M)** — and round 2's disposition audit found
that **three of the twenty-three round-1 findings the designer had recorded as `ADOPTED` were
not actually repaired**, one of them a Critical whose repair could not execute at all. Round 2
found all three of its own Criticals **by executing C01's frozen code**, not by reading the
document.

**So: audit by execution, not by reading.** Treat v3 §14's cumulative disposition table as an
index of things to check, never as evidence. The house bar is explicit that *any claimed repair
the artifact does not contain* is Critical.

**Nothing is implemented yet.** No battery script exists. This reviews the **design** only; a
separate independent code/resource review lineage runs later over the executable. Several v3
claims are about code that does not yet exist (notably the single shared mint driver, §3.3) —
judge them as *specifications*, say whether they are sufficiently specified to be checkable,
and name what the code lineage must verify.

---

## 0. What C06 is, so you need no prior context

C06 (*Prompt-Orbit Tangent/Curvature*) claims the tangent and curvature of a video's
representation across a fixed prompt orbit encode policy-bound instability no single prompt
captures. It is **not** an active candidate: its registry status is
`gated_on_zero_cost_falsifier`. An earlier candidate, C01, measured the two-point case in a
**raw-key** arena and found the best of six matched-norm orthogonal rotations of the prompt
endpoints **matched or beat** the real prompt displacement on both datasets — adverse to C06.
Because the registry says a raw-key arena *"may kill but may not promote"*, the Gate-0
adjudicator gated C06 rather than striking it: re-run C01's battery in the **fold-head
(deployed-head)** arena on already-banked caches. If the rotations again match, C06 closes for
`$0` and an authorized `1.7–2.5 GPU-h` extraction is never spent; if not, C06 has earned that
extraction. **The artifact under review is the design of that falsifier.**

---

## 1. Read first, in this order

1. `CLAUDE.md`, `AGENTS.md` — the user's standing instructions; neither may be edited.
2. `TARGET_STATE.json`, four blocks, read **verbatim** rather than through v3's quotation:
   `gate0_reopen_2026_07_31.dispositions.gated[0]`;
   `iteration_8_queue_state_2026_08_04`;
   `process_rule_compute_projection_and_heartbeat_2026_08_04`;
   `iteration_8_stage0_bounded_extraction_amendment`.
3. `TARGET_FINDINGS.md` — **F118** in full (including its erratum about boilerplate describing
   a leg that did not run); skim **F88** and **F113**.
4. `refine-logs/GATE0_REOPEN_2026-07-31.md` §4.4;
   `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §3.
5. `refine-logs/C09_A0_V17_RECORD.md` §2 and §8.1.
6. Both prior reviews in full: `C06_FALSIFIER_PREREG_REVIEW.md` (round 1),
   `C06_FALSIFIER_PREREG_REVIEW_R2.md` (round 2).
7. The superseded drafts, for context on what moved: `C06_FALSIFIER_PREREG_DRAFT.md` (v1),
   `C06_FALSIFIER_PREREG_DRAFT_V2.md` (v2).
8. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V3.md` — **the artifact under review**.

**Primary sources for the code claims:** `scripts/analysis/c01_policy_contrast_a0.py`
(especially `l2_rows:1183-1205` and its mask assertion at `:1193-1194`,
`fuse_modalities:1208-1217`, `paired_key:1220-1239`, `contrast_blocks:1242-1270`,
`orthogonal_blocks:1272`, `prepare_views:1294` and `:1381-1386`, the deferred-import site
`:387` / `import_compute_modules:1048`), `src/model/classifier.py` (`:80-81` the projections
and their **bias**, `:115-124` the forward), `scripts/analysis/headspace_mint.py`,
`scripts/analysis/c02_a0_mint.py`, `scripts/analysis/mechfix_ops.py`,
`scripts/analysis/mechnov_pairverify.py`, `configs/c01/c01_a0_v{2,3,4}.json` (including
`zero_contract_v2` and `output.decision_schema.required_halt_only_validity_guards`).

---

## 2. Verify these facts yourself

Report `VERIFIED` / `MISMATCH` with the value you obtained, for **every** item.

| # | claim | where |
|---|---|---|
| **V1** | All 21 sha256 digests in §11 match disk. | §11 |
| **V2** | The v4 → v3 → v2 config chain and v4 → v3 → base algebra chain, both sha-gated **in source**, not only in config. | §11 |
| **V3** | **The null-contract defect.** `head(0,0)` is non-zero (v3 measures `‖·‖ = 0.634676`); `h_std[355] == h_ow[355]` exactly; and `l2_rows` **dies** on the endpoint and common blocks under `zero_mask={355}` and on the displacement block under `zero_mask=None` — so `common_displacement` is unbuildable in head space under either mask. **Reproduce this yourself.** | §3.7, §7.4(a)–(f) |
| **V4** | **The repair executes.** All 13 head-space arms build at `n = 743, zero_mask = None` through the imported `l2_rows`. | §7.4(g) |
| **V5** | **The bridge.** Raw arms built at `n = 743, None` are **bit-identical** to raw arms built at `n = 744, {355}` restricted to the 743 surviving rows — `max\|diff\| = 0.000e+00`, all 13 arms — and every `ρ` is unchanged. **This is the load-bearing measurement of v3; reproduce it.** | §7.4(h)(i) |
| **V6** | `GATE-C01PARITY`: the two-block builder reproduces `prepare_views` bit-exactly on the raw L24 features. | §7.6 |
| **V7** | Per-dataset `ρ*`: HateMM `0.9681` (max `0.968176`), MHC-ZH `0.9772` (max `0.977223`), runner-ups `0.9644` / `0.9697`; and all 26 `ρ_raw` values. | §6.1 |
| **V8** | Majority rates `0.5995` / `0.6891`; row 355 is `hate_video_95`, label `1`, held out in **fold 4**. | §7.1 |
| **V9** | `GATE-FLOOR`'s six anchors on both metrics, equal to the banked `acc_deployed` and `mF1_deployed`. | §6 |
| **V10** | The Holm family: `(12 + 11) × 2 = 46` per `(dataset, lineage)`, `× 2 lineages = 92` per dataset; and §8 Phase 4's count of 92 comparison-cells. | §5.5, §8 |
| **V11** | Every product and both totals in §8 (`2886.3 s`, `3607.9 s`), the mint share `86.9 %`, Phase 3 `9.5 %`, and the sensitivity figures `3160.0 s` / `3981.1 s`. | §8 |
| **V12** | C09's `GATE-ARENA` (`C09_A0_V17_RECORD.md:1569-1572`) is scoped to **pooled native accuracy** — the floor arm — so v3's restriction of the lower bound to `endpoint_std` matches the precedent it cites. | §6.3 |

---

## 3. What you must assess

### A. The null contract (§3.7) — the substantive change, and the heart of this round

v3 physically removes C01's registered null (HateMM row 355) from the **arm arena**, head leg
and raw leg alike, giving `n = 743`, while head training and `GATE-FLOOR` stay at the full
`n = 744`.

- **Three coexisting populations.** Check that no gate, statistic or comparison silently mixes
  them. Is `GATE-POP` sufficient to detect it if one did? Name any quantity whose population is
  still ambiguous in the text.
- **Is removal verdict-neutral?** §3.7 argues it *removes* a bias rather than creating one,
  because leaving the null in makes it a live top-20 neighbour in eleven control arms and a dead
  key in the one real arm. **Test that argument.** In particular: can removing one bank item
  from a 743-item bank shift a top-20 neighbourhood in a way that systematically favours either
  lane? The claim is symmetry, not harmlessness — check the symmetry.
- **Label-freeness and pre-registration.** Row 355 is C01's pre-existing frozen
  `authorized_null`, selected on a feature property. Confirm no label enters the selection and
  that nothing in the contract depends on a trained-head number.
- **The dataset asymmetry.** HateMM at `743`, MHC-ZH at `579` with no removal. §3.7 argues the
  conjunction-of-independent-verdicts structure contains it. **Rule on this explicitly** — it is
  §15.3.
- **`GATE-DUALPATH`'s new role** (§15.4): it now asserts the row-subset identity rather than
  C01's masked-vs-removed *prediction* equivalence. Is that a faithful use of C01's
  `displacement_registered_null_exclusion`, or a different gate wearing its name? If the latter,
  say what it should be called and whether C01's original property is still needed anywhere.
- **The deleted leg.** v3 states there is **no** head-space null-row sensitivity leg because the
  alternative population is unbuildable, and deletes Phase 5 rather than retaining boilerplate.
  Check that no gate, output field or sentence anywhere still describes it — this is the F118
  erratum lesson applied before the fact, and it is exactly the kind of claim that decays
  silently.

### B. The three round-2 Criticals — did the repairs land, and did they create new defects?

- **C-1 → §3.7.** Verified by V3–V5. Does the repair reach *every* consumer of the old contract?
- **C-2 → §6.3.** `GATE-ARENA`'s lower bound is now restricted to `endpoint_std`. Confirm the
  self-defeat is gone: trace a run where both real arms sit near the majority rate and check
  that it CLOSES rather than HALTs. Then check the converse — is there now any path where a
  genuinely broken real arm escapes every gate?
- **C-3 → §6, §3.7.** `GATE-SHUFFLEFIX` is **deleted** as vacuous and replaced by
  `GATE-NULLREMOVED`; `GATE-ZEROMASK` is restated as feature-space only. Is anything from C01's
  `zero_contract_v2` or its `required_halt_only_validity_guards` now unguarded that should not
  be?

### C. The decision rule and multiplicity

- §5.5 adopts one Holm family of **92** per dataset spanning both lineages. Is that the right
  correction for a SURVIVE-disjunction over lineages, and does it interact correctly with the
  two-dataset conjunction and the two-arm disjunction inside it?
- §5.6 adds an **absence** HALT rule and makes `GATE-LEDGER`'s process count binding. Is the
  absence path now fully closed, given CLOSE is a conjunction over lineages?
- §6.5's `GATE-ZEROOP` **tie diagnostic** introduces a report-not-HALT branch (§15.6). Can it be
  widened to swallow a genuine mismatch? Is the pre-registered boundary sharp enough to be
  checkable by the code lineage?
- §5.8 discloses three things the conservative lean buys. Is the list complete?

### D. Head-R and the shared driver (§3.3)

v3's answer to round 2's H-2 is that **one driver** serves both lineages with `--train-cache`
as its only lineage-varying argument, so `GATE-FLOOR` anchors the driver for both at zero cost.

- Is that a real anchor, or does it only anchor the driver *on the native cache path*? What
  could differ on the ro path that `GATE-FLOOR` would not see?
- §15.5 asks whether round 2's sufficiency ruling survives now that the driver is shared **in
  fact rather than in claim**. Rule on it.
- v3 re-prices Head-R at Head-N's measured units (`+146.9 s`) because the scratchpad harness
  skipped work the real driver does. Is that the right direction, and is the reasoning sound?

### E. The process rules

- **`rule_1_compute_projection`.** §8 has no ratio-derived phase and every unit is attributed
  to a dataset. **Hunt for a loop §8 still does not count** — rounds 1 and 2 each found some.
  Check §7.2's argument that interpreter/import cost is already inside the mint units (the
  `40.39` full-process vs `33.0` internal evidence) and that adding a line would double-count.
- **`rule_2_heartbeat`.** §9 adds the HALT line naming which gate failed. Is any interval still
  longer than the stated ~15 s? Is §9's list of what the code lineage must verify complete?
- §7.7 discloses that one unit (`U9`) was originally a **failure path** recorded as a
  measurement, and corrected. Could any remaining unit carry the same defect?

### F. Scope and honesty

- §10.2 now includes round 2's **post-fusion contrast** bullet. Does §10.2 now say everything a
  CLOSE is actually scoped to?
- Does anything touch a `hard_constraint`?
- Does v3 claim any repair the artifact does not contain? Name it if so.

---

## 4. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | The design, if executed as written, could publish a **wrong verdict** — an unwarranted closure or an unwarranted survival — or **cannot execute** on the path that renders the verdict. Also: any false factual claim in §2, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**. |
| **High (H)** | Materially weakens the verdict's authority or its scope statement without inverting it. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**. Report the tally in that form.

C09 needed seventeen design rounds to reach `0C/0H/0I`, and a separate code-review lineage then
ran seven more and caught two wrong-verdict paths that seventeen clean design rounds had missed.

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
4. **A disposition audit of v3's §14 cumulative table**, done by execution: for each round-2
   finding, `VERIFIED ADOPTED` / `NOT ADOPTED` / `PARTIAL`, with what you checked. Also confirm
   the three reopened round-1 items (C-3, the C-1 companion, I-3) are now genuinely repaired.
5. An explicit ruling on each of the six open issues in v3 §15.
6. An explicit ruling on §3.A's central question: **is the null-removal contract verdict-neutral,
   and is the three-population arrangement free of silent mixing?**
7. A statement of what the **separate code-review lineage** must verify that a design review
   cannot — in particular for the shared mint driver, the tie diagnostic, and `GATE-POP`.
8. If you conclude the falsifier still cannot discharge the written condition at `$0`, say so
   directly and state what would be required instead.

---

*Read-only. No GPU, SLURM, Modal, model load, head training, arena run, cache write,
test-split access, job submission or commit is authorized by this document, and
`TARGET_STATE.json` must not be modified.*
