# C06 `$0` falsifier — fresh independent design review request, **ROUND 10**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM job,
no Modal, no GPU, no cache write, no commit, no edit to `TARGET_STATE.json`. Read-only
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint checkpoints,
plus `sha256sum`, is expected. **Up to four CPU head mints** (~40 s each) are permitted; rounds 8
and 9 both declined them with stated reasons, which is a legitimate choice.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V10.md` (v10, unfrozen).

You are a **fresh reviewer**, independent of rounds 1–9 and of the designer.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M → 3C/3H/6I+5M →
2C/3H/5I+6M → 4C/2H/3I+4M → 2C/2H/4I+6M → **0C**/2H/2I+4M.

---

## 0. Where this stands, and what your round is actually for

**The science layer is closed and three independent reviewers have confirmed it.** Round 9 rebuilt
all thirteen arms from §3.4's prose at the first attempt with nothing silently supplied
(`max|diff| = 0.000e+00`, both datasets), constructed every wrong reading it could and confirmed
each is caught, re-derived the gate analysis and the verdict enumeration, and reported *"no gate can
fire on a warranted CLOSE, all twenty, re-derived."* Round 9 was the first round with **zero
Criticals**.

**The record has been the blocker for four rounds, and its failure mode has evolved.** Rounds 6–8
found *absence* — repairs claimed but not made. Round 9 found **narrowing**: no limb was unlanded,
but two limbs landed a weaker repair than prescribed, and *"the limb table's paraphrase is where the
weakening happens."*

**v10's answer, and the thing you must test.** Every limb in §14 is now a **verbatim quotation** of
round 9's prescription with its location in round 9's review, because a paraphrase is where a clause
can silently disappear. Round 9 was explicit that no machinery can catch this — *"Nothing in the
machinery could have caught this; I caught it by reading round 8's sentence."*

**So your first obligation is a reading obligation, not a script one:** open
`C06_FALSIFIER_PREREG_REVIEW_R9.md`, read each prescription in full, and check the corresponding
quoted limb in v10 §14 is **complete and faithful** — not merely present. A truncated quotation or a
dropped clause is the finding.

---

## 1. What C06 is, so you need no prior context

C06 (*Prompt-Orbit Tangent/Curvature*) claims the tangent and curvature of a video's representation
across a fixed prompt orbit encode policy-bound instability no single prompt captures. It is **not**
an active candidate: its registry status is `gated_on_zero_cost_falsifier`. C01 measured the
two-point case in a **raw-key** arena and found the best of six matched-norm orthogonal rotations of
the prompt endpoints **matched or beat** the real prompt displacement on both datasets. Because the
registry says a raw-key arena *"may kill but may not promote"*, the Gate-0 adjudicator gated C06
rather than striking it: re-run C01's battery in the **fold-head** arena on already-banked caches.
If the rotations again match, C06 closes for `$0` and an authorized `1.7–2.5 GPU-h` extraction is
never spent; if not, C06 has earned that extraction.

---

## 2. Read first, in this order

1. `CLAUDE.md`, `AGENTS.md`.
2. `TARGET_STATE.json`: `gate0_reopen_2026_07_31.dispositions.gated[0]` (**verbatim**);
   `iteration_8_queue_state_2026_08_04`;
   `process_rule_compute_projection_and_heartbeat_2026_08_04`;
   `iteration_8_stage0_bounded_extraction_amendment`.
3. `TARGET_FINDINGS.md` — **F118**; skim **F88**, **F113**.
4. `refine-logs/GATE0_REOPEN_2026-07-31.md` §4.4;
   `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §3;
   `refine-logs/C09_A0_V17_RECORD.md` §2, §8.1.
5. **`refine-logs/C06_FALSIFIER_PREREG_REVIEW_R9.md` in full** — it is the specification v10
   answers, and the source your limb check must be made against. Then the earlier eight as needed.
6. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V9.md` (for the diff) and earlier drafts as context.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V10.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` — `prepare_views:1296-1304`,
`contrast_blocks:1246-1265`, `l2_rows:1183-1205`, `orthogonal_blocks:1272`, the algebra guard
`:1372-1377`, the two `fix_break` sites `:1725` / `:2702-2714`, `select_strongest_ordinary_control`
(guards `:1940-1948`, ranking `:1955-1962`), `:2724`, `displacement_audit:1965-2076` (`small_mask`
`:2036`, `source_rows` `:2049`, `registered_null_rows_excluded` `:2050`, `tiny_ok` `:2054-2057`),
`holm_adjust:1775-1784`; `src/model/classifier.py:81-82`, `:140-141`, `:146`;
`scripts/analysis/headspace_mint.py:192-194`, `:199`, `:209-216`, `:321-325`;
`scripts/analysis/mechfix_ops.py:94`; `configs/c01/c01_a0_v{2,3,4}.json`;
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`.

---

## 3. Verify these facts yourself

| # | claim | where |
|---|---|---|
| **V1** | All **37** sha256 match disk. | §11 |
| **V2** | **Break the corrected self-exclusion, as round 9 broke v9's.** Build a counterfactual whose §14.1 is byte-identical to v9's and run the §14.2 script: it must print `UNCHANGED §14.1` and **fail** the §14.1-citing rows and limbs. v10 claims it does and reports the counterfactual output. | §14.1, §14.2 |
| **V3** | **Re-run the audit** (script in §14.2). Confirm the embedded transcript equals your run byte-for-byte, and that §14.1's size is never printed while its changed/unchanged status is. | §14.1 |
| **V4** | **The limb quotations.** For each of the 20 limbs, open round 9's review and check the quotation is complete and faithful. This is the round's central check and no script performs it. | §14 |
| **V5** | §7.9's headline now reads *"attributable to the v6–v7 rounds"*, the reconciliation sentence is gone, v7 is re-derived at `≈ 4 / ≈ 21`, and the sum `7+1+4+0+0+0 = 12` agrees with §7.8 and the footer. | §7.9 |
| **V6** | **Phase 1f** (`60` cells × 2 matrices, measured `0.0083 s`/cell, timed region stated, carried at `1.0 s`) and the **corrected Phase 7z** (`U_tie = 2.0e-05 s`/item, `cells = 12` per §6.5, row reduced `0.3 → 0.1 s`). Column sums to **`2930.4`**; `× 1.25 = 3663.0`. | §8 |
| **V7** | §1's two previously dashed cells: ZH `common` `0.8718`/`+1`, HateMM `endpoint_concat` `0.8598`/`+2`, against `C01_A0_OUT.json`. | §1 |
| **V8** | Round-7 C-1's measurement still holds and §6's `GATE-C01PARITY` states one predicate. **Rebuild the arms from §3.4 yourself.** | §3.4, §6 |
| **V9** | `ρ*` `0.968176` / `0.977223`; all 26 `ρ_raw` at 6 dp under the frozen float64 reduction; trained-head `0/18`. | §6.1 |
| **V10** | The Holm counterexample table; the S5 feasibility bound `n ≤ 12`. | §5.5, §5.4.1 |
| **V11** | Every constant in §3.7's **two blocks**, each with the right verb (computed vs read-and-asserted). | §3.7 |
| **V12** | §6 has **twenty** gate rows, `12 G / 6 L / 2 R`, matching §5.6; §13 defines **26** contiguous items, and items 10 and 15 still carry their round-7/round-8 repairs even though §14 no longer references them by number. | §6, §13 |

---

## 4. What you must assess

### A. Faithfulness of the limb quotations (the central check)

Read round 9's four findings and four Minors in full. For each prescribed clause, is there a limb?
Is the quotation verbatim? Does any quotation stop before a clause that changes what was asked?
Round 9's I-2 is the worked example of what to look for: round 8 prescribed *"the tie-casualty
evaluation priced as `≤ cap × cells × U_tie` with `U_tie` measured on one synthetic near-tie
group"*, and v9's limb read *"add a Phase 7z row for `GATE-ZEROOP`'s scan and tie-casualty work"* —
faithful to the first clause, silently dropping the second.

### B. The corrected self-exclusion

Break it (V2). Then judge the convention: v10 says self-exclusion covers the **size only**, never
the changed/unchanged fact. Is that the right line? Is there any other place in the audit where a
convention is load-bearing for another row's verification?

### C. Where v10's own repairs could have opened seams

v10 changed: header, §1, §5.2.2, §7.9, §8, §14, §14.1, §14.2, §15. Every round has found its
findings in the previous round's repair. In particular: does Phase 1f's cold-cache convention
(`1.0 s` carried against `0.50 s` measured) conflict with how other measured rows are carried? Does
the Phase 7z reduction leave §6.5's aggregation and §8 consistent?

### D. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** Answered *no for all twenty* by rounds 6,
7, 8 and 9, each re-deriving. Do the same.

### E. The process rules

- **`rule_1_compute_projection`.** Nine rounds, eight uncounted loops found. Hunt again.
- **`rule_2_heartbeat`.** Does anything in v10 change an interval?

### F. Honesty

- Does v10 claim any repair the artifact does not contain, or any repair **narrower** than
  prescribed? Diff v9→v10 and check every row **and every limb against round 9's text**.
- Blindness across v1–v10: grep every decimal in `[0.6, 0.99]` and classify anything new — §1's two
  added cells are C01 dev-arena accuracies already published in that table's other rows.

---

## 5. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | Could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §3, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or scope without inverting it — including a repair landed **narrower** than prescribed. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**.

**Do not grade on trajectory, in either direction.** Ten rounds is evidence of nothing. Rounds 8 and
9 both said so explicitly and both held the line — round 9 graded its own eighth uncounted loop
Important precisely because grading it differently *"would be grading on trajectory, which the brief
forbids in both directions."* Hold the same line: if the design and the record are clean, say **GO**
plainly; if not, name the specific defect.

---

## 6. What a GO does and does not authorize

A GO authorizes **nothing to run**. Before any job: (1) freeze with hashes; (2) a **separate**
independent code/resource review lineage over the executable reaching its own `0C/0H/0I`;
(3) main-dialogue authorization. A GO is not authority to write `TARGET_STATE.json`.

---

## 7. Deliverable

1. The `0C/0H/0I`-form tally and **GO** or **REVISE**.
2. Every finding with severity, `file:line` citation, and the concrete repair.
3. Your verification result for **all twelve** §3 items.
4. **Your limb-faithfulness audit**: for each of the 20 limbs, `FAITHFUL` / `TRUNCATED` /
   `NARROWED`, checked against round 9's review text.
5. An explicit ruling on each of the six open issues in v10 §15.
6. An explicit ruling on §4.B: is the corrected self-exclusion sound, and is the size-only line the
   right one?
7. An explicit ruling on §4.D: any gate that can fire on a warranted CLOSE — all twenty.
8. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly.

---

*Read-only. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit is authorized by this document, and `TARGET_STATE.json` must not be modified.*
