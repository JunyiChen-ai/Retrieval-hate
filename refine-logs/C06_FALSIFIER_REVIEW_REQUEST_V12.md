# C06 `$0` falsifier — fresh independent design review request, **ROUND 12**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM job,
no Modal, no GPU, no cache write, no commit, no edit to `TARGET_STATE.json`. Read-only
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint checkpoints,
plus `sha256sum` and process-wall timings, is expected. **Up to four CPU head mints** (~40 s each)
are permitted; rounds 8–11 all declined them with stated reasons, which is a legitimate choice.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V12.md` (v12, unfrozen, sha256
`3504ce8a5850336fb1a4b28c5fefb21e43b52f71c9591100ea260d298a65eb1b`, 159793 bytes, 2177 lines).

You are a **fresh reviewer**, independent of rounds 1–11 and of the designer.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M → 3C/3H/6I+5M →
2C/3H/5I+6M → 4C/2H/3I+4M → 2C/2H/4I+6M → **0C**/2H/2I+4M → **0C/0H**/3I+4M → **0C/0H/1I**+1M.

---

## 0. Where this stands, and what your round is actually for

**The science layer is closed and five independent reviewers have confirmed it.** Round 11 rebuilt
all thirteen arms from §3.4's prose at the first attempt with nothing silently supplied
(`max|diff| = 0.000e+00`) at `n = 744` one-hot and at the arena `n = 743`/`579`, recomputed §1's
table cell for cell from `C01_A0_OUT.json`'s stored confusion matrices (16/16 accuracies, 16/16
net-fix integers), recomputed all 37 digests and all 26 `ρ` at 6 dp, re-derived every §8 count from
the design's own structure, and reported *"no gate can fire on a warranted CLOSE, all twenty,
re-derived."*

**The record is now sound, and round 11 was the first round able to say so without qualification.**
Rounds 6–8 found *absence*; round 9 found *narrowing*; round 10 found a *truncated quotation*. Round
11 executed the subtraction protocol in full — extract each **Repair** paragraph, subtract the
quoted limbs, read the residue — and found **13/13 limbs faithful and complete**, each inside the
line range it cited, with six of seven paragraphs subtracting to bare connective punctuation and all
four disclosed deviations warranted.

**So the blocker moved, and it moved to the place nine earlier rounds also found theirs.** With the
record clean, round 11 hunted §8 and found the **tenth uncounted item** — and the first that is not
a payload loop. The battery runs **73 processes**; §8 priced interpreter+import for **66** of them
(inside the mints' full-process-wall units); the **arena**, the 67th, carried its startup in no unit
and no row, because every arena-side unit is an internal-operation timing two to five orders of
magnitude below a python startup. The sentence that should have caught it — §7.2's *"already inside
every unit"* — was true of the mint units and false of the arena's.

**Your round has two jobs, and the first is short.** v12 has only **four** limbs, so the subtraction
check is cheap: extract round 11's two Repair paragraphs, subtract §14's four limbs, and report the
residue. **One limb lands a scope *wider* than the words prescribed** — round 11 asked §7.2 to name
*"the mint units"*, and v12 names *the 66 mint units **and `U9`***, because the new `U9` measurement
established more than round 11 could assume. Rule it: warranted widening, or a substitution that
should have been recorded as a deviation.

**The second job is a measurement, and it decides a number.** Round-11 I-1 offered a count of `1`
(arena) or `7` (arena + the six fidelity processes) and made the choice turn on `U9`'s unstated
timing boundary. v12 answers `1`, from three timed full-process runs of
`headspace_fidelity.py --seeds 0` at `3.13 / 3.46 / 3.12 s` against `3.06–3.16 s` of
interpreter+import alone — which makes the frozen `U9 = 3.70 s` a full-process wall, since an
internal timing would have to be under `0.4 s`. **Re-measure both yourself and rule on the
inference.** If it is wrong, the count is `7`, the row becomes `7 × 3.2 = 22.4 s` and the total
`≈ 2953` (round 11's own figure for that branch) — a difference in kind rather than in size, since
what would be wrong is a measurement-backed claim, not an estimate.

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
5. **`refine-logs/C06_FALSIFIER_PREREG_REVIEW_R11.md` in full** — it is the specification v12
   answers and the source your subtraction must be made against. Then the earlier ten as needed.
6. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V11.md` (for the diff) and earlier drafts as context.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V12.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` — `prepare_views:1294-1304`,
`contrast_blocks:1242-1265`, `l2_rows:1183-1205`, `orthogonal_blocks:1272`, the algebra guard
`:1372-1377`, the two `fix_break` sites `:1725` / `:2702-2714`, `select_strongest_ordinary_control`
(guards `:1940-1948`, ranking `:1955-1962`), `:2724`, `displacement_audit:1965-2076` (`small_mask`
`:2036`, `source_rows` `:2049`, `registered_null_rows_excluded` `:2050`, `tiny_ok` `:2054-2057`),
`holm_adjust:1775-1784`; `src/model/classifier.py:81-82`, `:140-141`, `:146`;
`scripts/analysis/headspace_mint.py:192-194`, `:199`, `:209-216`, `:306-307`, `:321-325`;
`scripts/analysis/headspace_arena.py:59`, `:75`, `:85`, `:89`, `:92-93`;
**`scripts/analysis/headspace_fidelity.py`** (whole file — `U9`'s boundary turns on it);
`scripts/analysis/mechfix_ops.py:94`; `configs/c01/c01_a0_v{2,3,4}.json`;
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`.

---

## 3. Verify these facts yourself

| # | claim | where |
|---|---|---|
| **V1** | All **37** sha256 match disk. Four cache rows elide the model stem with `…`; resolve them. | §11 |
| **V2** | **`U9` is a full-process wall.** Time `python scripts/analysis/headspace_fidelity.py --dataset hatemm --mintdir artifacts/c09_topo/v1/a0/C09-A0-v1/scratch --out <your scratchpad> --seeds 0` around the invocation, and separately time that module's interpreter+import. v12 measured `3.13 / 3.46 / 3.12 s` and `3.06–3.16 s`. **Rule whether the inference — that a `3.70 s` unit cannot be an internal timing — is sound.** The count in Phase 1g depends on it. | §7.7 |
| **V3** | **Phase 1g and the whole re-multiplied column.** Row `1` × `U11` = `3.2 s`; printed column sums to **`2933.9`**; `× 1.25 = 3667.4`; `48.9` / `61.1 min`; mint share `85.5 %`; Phase 3 `9.3 %`; `2×` miss `3207.6 s = 53.5 min`; `5×` miss `4028.7 s = 67.1 min`. §8 now has **26 rows** and accounts for all **73** processes (`66 + 6 + 1`). Check no stale total survives. | §8 |
| **V4** | **The four limb quotations, by subtraction.** Check each is verbatim and complete, that its `R11:NNN-NNN` range contains it, then subtract the four from round 11's two Repair paragraphs and report the residue. **This is the round's central record check and no script performs it.** | §14 |
| **V5** | **Re-run the audit** (script in §14.2). Confirm the embedded transcript equals your run byte-for-byte and exits `0`. Note the transcript prints `CHANGED §14.2 +0 chars` — v12 explains this as same-length substitutions with `CHANGED` computed from content, not size. **Verify that explanation** rather than accept it. | §14.1 |
| **V6** | **Break the self-exclusion**, as rounds 10 and 11 did: splice v11's §14.1 into a copy of v12 and confirm the script prints `UNCHANGED §14.1`, fails the §14.1-citing rows/limbs, and exits `1`. | §14.1, §14.2 |
| **V7** | **Rebuild the arms from §3.4 yourself**; `GATE-C01PARITY` states one bit-exact predicate; the un-normalised misreading measures `1.878e-06` / `1.609e-06`, both under `2e-6`. | §3.4, §6 |
| **V8** | `ρ*` `0.968176` / `0.977223`; all 26 `ρ_raw` at 6 dp; trained-head `0/18`. | §6.1 |
| **V9** | The Holm counterexample table; `n ≤ 12`; §3.7's **two blocks** with two verbs. | §5.5, §5.4.1, §3.7 |
| **V10** | §7.9's sum: heading `v1–v12`, `7+1+4+0+0+0+0+0 = 12`, `22+4+2+1+1+1+1 = 32`, `89+21+6+3+3+3+3 = 128`, agreeing with §7.8 and the footer. | §7.9 |
| **V11** | §6 has **20** gate rows, `12 G / 6 L / 2 R`, matching §5.6; §13.1 defines **26** contiguous items; items 10, 15, 19 and 22 carry their round-7/8/10 repairs. | §6, §13.1 |
| **V12** | §7.2's re-scoped sentence is true of **every one** of the 72 processes it now covers, and §9's new arena-startup clause conflicts with neither the `~15 s` interval bound nor the driver's echo discipline. | §7.2, §9 |

---

## 4. What you must assess

### A. The subtraction (four limbs, and the one widening)

Read round 11's I-1 and M-1 in full. Is each of the four quotations verbatim **and complete**? Then
rule the widening: round 11's third limb said *"name the scope it is true of — the mint units"*;
v12 names *the 66 mint units and `U9`*. Is that a warranted consequence of v12's own `U9`
measurement, or a substitution that should have carried a deviation label?

### B. The count, and the unit

Two independent judgements. **The count** is `1` and rests entirely on V2's measurement — if `U9`
is an internal timing the count is `7`. **The unit** is `U11`'s upper bound `3.2 s`, carried against
arena-class measurements of `1.82–1.85 s` (v12) and `1.84–1.91 s` (round 11). v12 argues `U11` is a
unit already in §7.7, measured on this node, so `rule_1`'s *"measured unit × explicit count"* form
is satisfied without a new unit of unknown corroboration, and that carrying above the measurement is
what every other carried row does. Rule whether that is right or whether the measured arena figure
should have been used.

### C. Where v12's own repair could have opened a seam

v12 changed: header, §7.2, §7.3, §7.7, §7.9, §8, §9, §14, §14.1, §14.2, §15. Every round has found
its findings in the previous round's repair.

### D. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** Answered *no for all twenty* by rounds
6–11, each re-deriving. Do the same.

### E. The process rules

- **`rule_1_compute_projection`.** Eleven rounds, ten items found — nine payload loops and one
  per-process fixed cost. **Hunt the eleventh, and say which axis you searched**, since the last one
  changed axis.
- **`rule_2_heartbeat`.** Does anything in v12 change an interval?

### F. Honesty

- Does v12 claim any repair the artifact does not contain, or any repair **narrower** than
  prescribed? Diff v11→v12 and check both rows and all four limbs against round 11's text.
- Blindness across v1–v12: grep every decimal in `[0.6, 0.99]` and classify anything new. §7.3 claims
  the scope `v1–v12` and states v12 adds no accuracy — its new measurements are all timings. Verify
  rather than inherit.

---

## 5. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | Could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §3, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or scope without inverting it — including a repair landed **narrower** than prescribed. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**.

**Do not grade on trajectory, in either direction.** Twelve rounds is evidence of nothing. Rounds
8–11 each said so explicitly and each held the line — round 11 graded its own tenth uncounted item
Important and refused to soften it *"because a GO is one finding away"*, which it called grading on
trajectory. Hold the same line: if the design and the record are clean, say **GO** plainly; if not,
name the specific defect.

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
4. **Your limb audit by subtraction**: for each of the four limbs, `FAITHFUL` / `TRUNCATED` /
   `NARROWED`, plus the residue after subtracting them from round 11's two Repair paragraphs.
5. **An explicit ruling on the §7.2 widening** (§4.A) and **on the count and the unit** (§4.B).
6. An explicit ruling on each of the five open issues in v12 §15.
7. An explicit ruling on §4.D: any gate that can fire on a warranted CLOSE — all twenty.
8. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly.

---

*Read-only. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit is authorized by this document, and `TARGET_STATE.json` must not be modified.*
