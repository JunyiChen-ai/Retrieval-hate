# C06 `$0` falsifier — fresh independent design review request, **ROUND 9**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM job,
no Modal, no GPU, no cache write, no commit, no edit to `TARGET_STATE.json`. Read-only
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint checkpoints,
plus `sha256sum`, is expected. **Up to four CPU head mints** on the login node (~40 s each) are
permitted if you want to re-derive §7.8; round 8 declined them with a stated reason and that is a
legitimate choice.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V9.md` (v9, unfrozen).

You are a **fresh reviewer**, independent of rounds 1–8 and of the designer.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M → 3C/3H/6I+5M →
2C/3H/5I+6M → 4C/2H/3I+4M → 2C/2H/4I+6M.

---

## 0. Two things that should shape how you read v9

**(a) The science layer has been independently confirmed closed.** Round 8 rebuilt all thirteen arms
from v8's §3.4 prose alone and got `max|diff| = 0.000e+00` against `prepare_views` on both datasets
**at the first attempt with nothing silently supplied**, then constructed every wrong reading it
could and confirmed each is caught by the tightened `GATE-C01PARITY`. It also re-derived the gate
analysis and the verdict-path enumeration independently and found no gate that can fire on a
warranted CLOSE. Round 8's own words: *"the blocker is the record, not the science."*

**(b) The record has failed in the same family for three consecutive versions, and round 8 named the
structural cause.** v6 and v7 claimed repairs never made. v8 added a mechanical audit (§14.1) — and
round 8 showed the audit was **structurally blind to the failure**: it asked *"did the section this
row cites change?"* and could never ask *"did every **limb** land, including limbs in sections no row
cites?"* **Every unlanded limb in v8 was in §13, byte-identical v7→v8 and cited by no row.**

**v9's response, and what you should test.** §13 is edited for the first time since v7. §14 now
carries a **limb table** (26 limbs), and the audit gains **step (5)** (each limb must name a diffed
landing section) and **step (6)** (changed-but-uncited / named-but-unchanged). §14.1 is
**self-excluding** so its transcript is a fixed point, and the transcript was regenerated against
the finished on-disk file and then re-verified byte-identical.

**Round 8's ruling on this protocol is your starting point, not your conclusion:** *"an embedded
self-audit is necessary and not sufficient… the mechanism that actually catches unlanded limbs is an
independent reader diffing against the previous round's prescriptions, not against §14's
self-description."* **That is your obligation this round.** Diff v8→v9 yourself and check v9 against
**round 8's prescriptions**, not against what §14 says it did.

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
5. **`refine-logs/C06_FALSIFIER_PREREG_REVIEW_R8.md` in full** — it is the specification v9 answers.
   Then the earlier seven as needed.
6. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V8.md` (for the diff) and the earlier drafts as context.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V9.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` — `prepare_views:1296-1304`,
`contrast_blocks:1246-1265`, `l2_rows:1183-1205`, `paired_key:1220-1239`, `orthogonal_blocks:1272`,
the algebra guard `:1372-1377`, the two `fix_break` sites `:1725` / `:2702-2714`,
`select_strongest_ordinary_control` (guards `:1940-1948`, ranking `:1955-1962`), `:2724`,
`displacement_audit:1965-2076` (`small_mask` at **`:2036`**, `tiny_ok` at `:2054-2057`),
`paired_bootstrap:1742-1772`, `holm_adjust:1775-1784`; `src/model/classifier.py:81-82`, `:140-141`,
`:146`; `scripts/analysis/headspace_mint.py:192-194`, `:199`, `:209-216`, `:321-325`;
`scripts/analysis/mechfix_ops.py:94`; `configs/c01/c01_a0_v{2,3,4}.json`;
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`.

---

## 3. Verify these facts yourself

| # | claim | where |
|---|---|---|
| **V1** | All **37** sha256 match disk. | §11 |
| **V2** | **Re-run the audit** (script reproduced in full at §14.2, so you need no scratchpad access). Confirm the embedded transcript equals your run byte-for-byte, and that §14.1 never reports its own size. Round-8 C-1 was a stale self-measurement; v9 claims a verified fixed point. | §14.1, §14.2 |
| **V3** | **The limb table.** 26 limbs; each names a section the diff shows as changed. Then do what the audit cannot: check each limb against **round 8's own prescription text**, not against the limb table's paraphrase. | §14 |
| **V4** | **The two §13 limbs round 8 found unlanded.** Item 19 now names endpoint pre-normalisation and the bit-exact predicate; item 15 reads *"never **calls** `displacement_audit`"*. §13.1 diffs `+1904`. | §13.1 |
| **V5** | **`GATE-SHA`'s widened scope** in §6 covers the sixteen banked artifacts, and `U7` is re-described and re-priced (`0.13 s`). | §6, §7.7 |
| **V6** | **§3.7's two blocks** — population-derived (computed) and frozen C01 config (read and asserted) — and that §13 item 5 now uses the right verb for each. `normalization_epsilon = 1e-12` is registered. | §3.7, §13.1 |
| **V7** | §8's product column sums to **`2929.6`**; `× 1.25 = 3662.0`; the new Phase 7z `GATE-ZEROOP` row; the `85.6 %` / `9.3 %` shares and the `3203.3` / `4024.4` sensitivities. | §8 |
| **V8** | §7.9's mint count as a **sum** (`7+1+4+0+0 = 12`) and the reported-versus-spent distinction against §7.8. | §7.9 |
| **V9** | Round-7 C-1's measurement still holds: correct build `0.000e+00`; un-normalised `1.878e-06` / `1.609e-06`, both under `2e-6`; and §6's row states exactly one predicate. **Rebuild the arms from §3.4 yourself.** | §3.4, §6 |
| **V10** | `ρ*` `0.968176` / `0.977223`, all 26 `ρ_raw` at 6 dp under float64; the trained-head `0/18`. | §6.1 |
| **V11** | The Holm counterexample table and the S5 feasibility bound (`n ≤ 12`). | §5.5, §5.4.1 |
| **V12** | §6 has **twenty** gate rows, `12 G / 6 L / 2 R`, matching §5.6; §13 defines **26** contiguous items. | §6, §13 |

---

## 4. What you must assess

### A. The limb protocol — does it close the gap or move it?

Round 8's step (5) and step (6) are implemented. **Test them adversarially.** Can a limb name a
section that diffed for an unrelated reason and pass? (It can — say whether that matters and what
would close it.) Is the limb table's *paraphrase* of round 8's prescriptions faithful, or does a
limb quietly narrow what was asked? Does step (6)'s *changed-but-uncited* output (§14, §15 only)
mean what §14.1 says it means?

**And the framing question:** round 8 ruled that a self-audit is necessary and not sufficient, and
that the real mechanism is an independent reader. Having been that reader, do you agree? What, if
anything, should move from the script into the review request?

### B. §13, edited for the first time since v7

It is the sole input to the mandatory separate code/resource review lineage. Read all 26 items as
someone who must implement from them with no other context. Are items 5, 8, 10, 15 and 19 — the ones
v9 touched — actionable and correct? Is anything a code lineage needs still absent?

### C. Where v9's own repairs could have opened seams

v9 changed: header, §3.7, §5.2.2, §5.2.3, §6, §7.3, §7.7, §7.9, §8, §10.2, §13.1, §14, §14.1, §14.2,
§15. Every previous round found its Criticals in the previous round's repair. Look there first.
Specifically: does §3.7's split leave any consumer reading a constant with the wrong verb? Does
`GATE-SHA`'s widening interact with §12's ledger or §8 Phase 1d?

### D. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** Answered *no for all twenty* by rounds 6,
7 and 8, each re-deriving rather than inheriting. Do the same, with attention to the widened
`GATE-SHA`.

### E. The process rules

- **`rule_1_compute_projection`.** Eight rounds, seven uncounted loops found. Hunt again. And rule
  on the timing spread: three parties measured the same two loops across `8×` and `18×` ranges; v9
  freezes the conservative bound of each and records that the spread is about timing boundaries.
- **`rule_2_heartbeat`.** Does anything in v9 change an interval?

### F. Honesty

- **Does v9 claim any repair the artifact does not contain?** Diff v8→v9 and check every §14 row
  **and every limb**. This method has produced a Critical in each of the last three rounds.
- Blindness across v1–v9: grep every decimal in `[0.6, 0.99]` and classify anything new.
- v9 makes **no blanket adoption claim** anywhere. Confirm that, and say whether the limb table is
  an honest substitute or a longer way of asserting the same thing.

---

## 5. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | Could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §3, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or scope without inverting it. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**.

**Do not grade on trajectory, in either direction.** Nine rounds is evidence of nothing. If the
design and the record are both clean, say **GO** plainly; if they are not, name the specific defect
with the same specificity the previous eight rounds did. Round 8 was explicit that it had not graded
on trajectory and that the findings, not the count, were the argument — hold the same line.

---

## 6. What a GO does and does not authorize

A GO authorizes **nothing to run**. Before any job: (1) freeze with hashes; (2) a **separate**
independent code/resource review lineage over the executable reaching its own `0C/0H/0I` — round 8
noted that lineage has real work waiting in §13; (3) main-dialogue authorization. A GO is not
authority to write `TARGET_STATE.json`.

---

## 7. Deliverable

1. The `0C/0H/0I`-form tally and **GO** or **REVISE**.
2. Every finding with severity, `file:line` citation, and the concrete repair.
3. Your verification result for **all twelve** §3 items.
4. **Your own limb-level audit of round 8's fourteen findings**, by diffing v8→v9 against round 8's
   prescription text: for each limb, `LANDED` / `NOT LANDED` / `NARROWED`, with the diff evidence.
5. An explicit ruling on each of the six open issues in v9 §15.
6. An explicit ruling on §4.A: **does the limb protocol close the gap, or move it?**
7. An explicit ruling on §4.D: any gate that can fire on a warranted CLOSE — all twenty.
8. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly.

---

*Read-only. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit is authorized by this document, and `TARGET_STATE.json` must not be modified.*
