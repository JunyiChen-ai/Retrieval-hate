# C06 `$0` falsifier — fresh independent design review request, **ROUND 11**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM job,
no Modal, no GPU, no cache write, no commit, no edit to `TARGET_STATE.json`. Read-only
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint checkpoints,
plus `sha256sum`, is expected. **Up to four CPU head mints** (~40 s each) are permitted; rounds 8, 9
and 10 all declined them with stated reasons, which is a legitimate choice.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V11.md` (v11, unfrozen, sha256
`94699ac37c4a06b800cde384d67d5b2da6b04ca45f6c9808011f7304aa98fc32`, 157460 bytes, 2136 lines).

You are a **fresh reviewer**, independent of rounds 1–10 and of the designer.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M → 3C/3H/6I+5M →
2C/3H/5I+6M → 4C/2H/3I+4M → 2C/2H/4I+6M → **0C**/2H/2I+4M → **0C/0H**/3I+4M.

---

## 0. Where this stands, and what your round is actually for

**The science layer is closed and four independent reviewers have now confirmed it.** Round 10
rebuilt all thirteen arms from §3.4's prose at the first attempt with nothing silently supplied
(`max|diff| = 0.000e+00`) at `n = 744` one-hot **and** at the arena `n = 743`/`579` all-False,
constructed every wrong reading it could and confirmed each is caught, recomputed all 37 digests and
all 26 `ρ` at 6 dp, re-summed §8's printed column to `2930.4`, and reported *"no gate can fire on a
warranted CLOSE, all twenty, re-derived."* Round 10 was the first round with **zero Criticals and
zero Highs**.

**The record has been the blocker for five rounds, and its failure mode has evolved twice.** Rounds
6–8 found *absence* — repairs claimed but not made. Round 9 found *narrowing* — limbs landing weaker
than prescribed, with the paraphrase as the mechanism. v10 answered with **verbatim quotation**.
Round 10 then found the residual failure that verbatim quotation makes *visible* but does not
prevent: **a quotation that stops before its qualifier**. It found it by extracting each **Repair**
paragraph from round 9's review, **subtracting the quoted limbs**, and reading the residue.

**So your first obligation is that subtraction, and it is a reading obligation, not a script one.**
Open `C06_FALSIFIER_PREREG_REVIEW_R10.md`, extract each Repair paragraph, subtract v11 §14's
thirteen limbs from it, and report what is left. A truncated quotation or a dropped clause **is the
finding**, at the severity round 10 gave the same defect class.

**Four limbs in v11 record a stated DEVIATION**, and judging them is the second half of your reading
obligation:

* round-10 I-2's *"correct the printed scope to describe all three arms"* — v11 has one arm and one
  filter, not three, because it dropped the whitelist;
* round-10 I-2's *"soften the §14.1 parenthetical to say that three unreachable references remain"* —
  v11 reports **zero** remain, having widened the pattern;
* round-10 I-3's *"Cumulative v1–v10"* and M-1's *"`v1–v9` → `v1–v10`"* — landed at **v1–v11**;
* round-10 M-2's *"`round-8` → `round-9`, `v9` → `v10`, `v8->v9` → `v9->v10`"* — landed at
  `round-10`, `v11`, `v10->v11`.

Each is disclosed in the limb table. **Your job is to rule whether each is a disclosed and warranted
reading of a prescription written one version earlier, or narrowing wearing a disclosure.** The
second reading is a finding.

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
5. **`refine-logs/C06_FALSIFIER_PREREG_REVIEW_R10.md` in full** — it is the specification v11
   answers, and the source your limb subtraction must be made against. Then
   `C06_FALSIFIER_PREREG_REVIEW_R9.md` (round-10 I-2 turns on round 9's M-2 sentence), then the
   earlier eight as needed.
6. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V10.md` (for the diff) and earlier drafts as context.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V11.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` — `prepare_views:1296-1304`,
`contrast_blocks:1246-1265`, `l2_rows:1183-1205`, `orthogonal_blocks:1272`, the algebra guard
`:1372-1377`, the two `fix_break` sites `:1725` / `:2702-2714`, `select_strongest_ordinary_control`
(guards `:1940-1948`, ranking `:1955-1962`), `:2724`, `displacement_audit:1965-2076` (`small_mask`
`:2036`, `source_rows` `:2049`, `registered_null_rows_excluded` `:2050`, `tiny_ok` `:2054-2057`),
`holm_adjust:1775-1784`; `src/model/classifier.py:81-82`, `:140-141`, `:146`;
`scripts/analysis/headspace_mint.py:192-194`, `:199`, `:209-216`, `:321-325`;
`scripts/analysis/headspace_arena.py:75-89`; `scripts/analysis/mechfix_ops.py:94`;
`configs/c01/c01_a0_v{2,3,4}.json`;
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`.

---

## 3. Verify these facts yourself

| # | claim | where |
|---|---|---|
| **V1** | All **37** sha256 match disk. Four cache rows elide the model stem with `…`; resolve them. | §11 |
| **V2** | **Break the self-exclusion as round 10 broke v9's.** Build a counterfactual whose §14.1 is byte-identical to v10's and run the §14.2 script: it must print `UNCHANGED §14.1` and **fail** the §14.1-citing rows and limbs. v11's logic is unchanged from v10's, which round 10 verified this way. | §14.1, §14.2 |
| **V3** | **Re-run the audit** (script in §14.2). Confirm the embedded transcript equals your run byte-for-byte, and that §14.1's size is never printed while its changed/unchanged status is. | §14.1 |
| **V4** | **The limb quotations, by subtraction.** For each of the 13 limbs, open round 10's review, check the quotation is complete and faithful, and check the cited `R10:NNN-NNN` line range actually contains it. Then subtract the limbs from each Repair paragraph and report the residue. **This is the round's central check and no script performs it.** | §14 |
| **V5** | **Phase 1f at the extended count**: `60 × 2 + 30 × 1 = 150` materialisations, unit `0.0041 s`/matrix with the timed region stated, carried `1.3 s` against `0.615 s` measured. Column sums to **`2930.7`**; `× 1.25 = 3663.4`; `48.8` / `61.1 min`; mint share `85.6 %`; Phase 3 `9.3 %`; `2×` miss `3204.4 s`; `5×` miss `4025.5 s`. | §8 |
| **V6** | **The placement I-1's second half asked for.** §13.1 item 22 now says all three key forwards happen in the mint and `GATE-FLOOR`'s vote in the arena. **Check that against `headspace_mint.py` and `headspace_arena.py` as they actually are**, and check Phase 2's `GATE-FLOOR` row agrees. | §13.1, §8 |
| **V7** | **The item-scan mechanism.** v11 drops the verb whitelist for round-9 M-2's prescribed prefix exclusion. Run your own exhaustive scan of v11: (a) is any genuine §13-item reference still unreachable? (b) does the pattern now sweep in any non-§13 item reference? (c) is the printed scope line **true of the pattern**? v11 claims nine previously unreachable sites are now reached and zero remain. | §14.1, §14.2 |
| **V8** | §7.9's sum: heading `v1–v11`, `7+1+4+0+0+0+0 = 12`, `22+4+2+1+1+1 = 31`, `89+21+6+3+3+3 = 125`, agreeing with §7.8 and the footer. | §7.9 |
| **V9** | Round-7 C-1's measurement still holds and §6's `GATE-C01PARITY` states one bit-exact predicate. **Rebuild the arms from §3.4 yourself.** | §3.4, §6 |
| **V10** | `ρ*` `0.968176` / `0.977223`; all 26 `ρ_raw` at 6 dp under the frozen float64 reduction; trained-head `0/18`. | §6.1 |
| **V11** | The Holm counterexample table; the S5 feasibility bound `n ≤ 12`; §3.7's **two blocks**, each with the right verb (computed vs read-and-asserted). | §5.5, §5.4.1, §3.7 |
| **V12** | §6 has **twenty** gate rows, `12 G / 6 L / 2 R`, matching §5.6; §13.1 defines **26** contiguous items; items 10, 15, 19 and 22 carry their round-7/round-8/round-10 repairs. | §6, §13.1 |

---

## 4. What you must assess

### A. The limb subtraction (the central check)

Read round 10's three Importants and four Minors in full. For each prescribed clause, is there a
limb? Is the quotation verbatim **and complete**? Round 10's I-2 is the worked example: round 9
prescribed *"or widen the pattern to bare `item N` **with the §5.9/§15 item references excluded by
their own prefixes**"* and v10's limb stopped at *"bare `item N`"*, dropping the qualifier and
landing a different mechanism while recording *"both adopted"*. Then rule the four disclosed
deviations listed in §0.

### B. Round-10 I-2's mechanism, tested rather than read

v11 implements the prescribed prefix exclusion and, to make it exact, gave three bare `§5.9` sibling
references (in §5.9 items 7, 8 and 9) their `§5.9` prefix. Judge both halves: is editing the
document's references to fit the pattern a legitimate repair or a way of making the check easier?
And is the printed list — now the full `1..26`, because §13.1's structural range headings are
expanded — the right output, or should range headings be excluded from it?

### C. Where v11's own repairs could have opened seams

v11 changed: header, §5.9, §7.3, §7.9, §8, §9, §13.1, §14, §14.1, §14.2, §15. Every round has found
its findings in the previous round's repair. In particular: does §9's newly declared output root
`artifacts/c06_falsifier/` conflict with §7's *"zero write into `artifacts/`"* dry-check statement or
with anything in §11/§12? Does Phase 1f's extended count leave §8, §6 and §13.1 mutually consistent?

### D. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** Answered *no for all twenty* by rounds 6,
7, 8, 9 and 10, each re-deriving. Do the same.

### E. The process rules

- **`rule_1_compute_projection`.** Ten rounds, nine uncounted loops found. Hunt the tenth.
- **`rule_2_heartbeat`.** Does anything in v11 change an interval?

### F. Honesty

- Does v11 claim any repair the artifact does not contain, or any repair **narrower** than
  prescribed? Diff v10→v11 and check every row **and every limb against round 10's text**.
- Blindness across v1–v11: grep every decimal in `[0.6, 0.99]` and classify anything new. §7.3 now
  claims the scope `v1–v11` and states v11 adds no accuracy of any kind — verify it rather than
  inherit it.

---

## 5. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | Could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §3, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or scope without inverting it — including a repair landed **narrower** than prescribed. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**.

**Do not grade on trajectory, in either direction.** Eleven rounds is evidence of nothing. Rounds 8,
9 and 10 each said so explicitly and each held the line — round 10 graded its own ninth uncounted
loop Important precisely because grading it differently *"would be grading on trajectory, which the
brief forbids in both directions."* Hold the same line: if the design and the record are clean, say
**GO** plainly; if not, name the specific defect.

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
4. **Your limb-faithfulness audit by subtraction**: for each of the 13 limbs, `FAITHFUL` /
   `TRUNCATED` / `NARROWED`, checked against round 10's review text and its cited line range; plus
   the residue left after subtracting the limbs from each Repair paragraph.
5. **A ruling on each of the four disclosed deviations** listed in §0: warranted or narrowing.
6. An explicit ruling on each of the five open issues in v11 §15.
7. An explicit ruling on §4.D: any gate that can fire on a warranted CLOSE — all twenty.
8. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly.

---

*Read-only. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit is authorized by this document, and `TARGET_STATE.json` must not be modified.*
