# C06 `$0` falsifier — fresh independent design review request, **ROUND 13**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM job,
no Modal, no GPU, no cache write, no commit, no edit to `TARGET_STATE.json`. Read-only
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint checkpoints,
plus `sha256sum` and process-wall timings, is expected. **Up to four CPU head mints** (~40 s each)
are permitted; rounds 8–12 all declined them with stated reasons, which is a legitimate choice.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V13.md` (v13, unfrozen, sha256
`308578cc8087f430a8cb0d9a520b67144e272338870fe39514ee86fabcb7db97`, 171270 bytes, 2281 lines).

You are a **fresh reviewer**, independent of rounds 1–12 and of the designer.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M → 3C/3H/6I+5M →
2C/3H/5I+6M → 4C/2H/3I+4M → 2C/2H/4I+6M → **0C**/2H/2I+4M → **0C/0H**/3I+4M → **0C/0H/1I**+1M →
**0C/0H/1I**+3M.

---

## 0. Where this stands, and what your round is actually for

**The science layer is closed and six independent reviewers have confirmed it.** Round 12 rebuilt
all thirteen arms bit-exactly at both population configurations, reproduced 26/26 `ρ` at 6 dp, 16/16
C01 accuracies and 16/16 net-fix integers, 12/12 `GATE-FLOOR` anchors, 37/37 digests, and re-derived
all twenty gates as unable to fire on a warranted CLOSE.

**The record has been clean for two rounds.** Round 12: *"4/4 limbs faithful and complete, both
Repair paragraphs subtracting to bare scaffolding, no undisclosed deviation, no repair claimed that
the artifact does not contain, no repair landed narrower than prescribed, and zero stale totals."*

**Both remaining findings have been inside §8, one round apart, and both were about evidence rather
than arithmetic.** Round 11 found the tenth uncounted item: the arena's interpreter+import was
priced in no row. Round 12 found that the row created to fix it carried a unit measured over an
import set that **omitted `sklearn`** — worth `≈ 1.2 s`, about two-thirds of the number — because
`headspace_mint.py:68` (a §11 sha-frozen import) and `headspace_arena.py:35-36` both require it.
Round 12's framing is the one to carry into this round: *"agreement between parties who made the
same omission is not corroboration."*

**Your round has one central job.** v13 lands round-12 I-1 in full **except for the number**, and
that exception is a **stated deviation** you must rule on. Round 12 prescribed *"Keep the unit at
`U11 = 3.2 s`"* with *"the residual `≤ 0.2 s`"*, which was exactly right on round 12's own maximum
of `3.27 s`. v13 measured the same rungs 14 times and reached **`3.75 s`**, so it carries **`3.8 s`**
instead — above the pooled `3.00–3.75 s` — on the grounds that keeping `3.2 s` would have written a
residual bound its own measurement falsifies and made Phase 1g the second §8 row carried *below* its
measurement.

**Re-measure it yourself** (§7.7's table gives the seven rungs; §13.1 item 27 gives the set) and
rule: **forced by measurement, or a preference dressed as one?** If your maximum lands near round
12's, say so — the honest outcome is then that `3.8 s` is over-conservative by `≈ 0.5 s`, which is
`0.02 %` of the total, but the *basis* would again be the finding, and that is the pattern two
rounds running.

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
5. **`refine-logs/C06_FALSIFIER_PREREG_REVIEW_R12.md` in full** — it is the specification v13
   answers and the source your subtraction must be made against. Then the earlier eleven as needed.
6. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V12.md` (for the diff) and earlier drafts as context.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V13.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` — `prepare_views:1294-1304`,
`contrast_blocks:1242-1265`, `l2_rows:1183-1205`, `orthogonal_blocks:1272`, the algebra guard
`:1372-1377`, the two `fix_break` sites `:1725` / `:2702-2714`, `select_strongest_ordinary_control`
(guards `:1940-1948`, ranking `:1955-1962`), `:2724`, `displacement_audit:1965-2076`,
`holm_adjust:1775-1784`; `src/model/classifier.py:81-82`, `:140-141`, `:146`;
**`scripts/analysis/headspace_arena.py:28-46`** (the import block item 27 pins), `:59`, `:72`, `:75`,
`:85`, `:89`, `:92-93`; **`scripts/analysis/headspace_mint.py:68`** (the `sklearn` entry point),
`:192-194`, `:199`, `:306-307`, `:321-325`; `scripts/analysis/headspace_fidelity.py`;
`scripts/analysis/mechfix_ops.py:94`; `configs/c01/c01_a0_v{2,3,4}.json`;
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`.

---

## 3. Verify these facts yourself

| # | claim | where |
|---|---|---|
| **V1** | **The arena's startup, re-measured by you.** Build the import set up rung by rung as §7.7's table does, and report your range for `headspace_arena.py`'s actual top-level set plus `c01_policy_contrast_a0` plus `runtime_block()`. v13 measured `3.02–3.75 s` over 14 runs; round 12 measured `3.00–3.27 s`. **Then rule the deviation** (§4.A). | §7.7, §8 |
| **V2** | **Phase 1g and the re-multiplied column.** `1 × U11`(arena class) carried at `3.8 s`; printed column sums to **`2934.5`**; `× 1.25 = 3668.1`; `48.9` / `61.1 min`; mint `85.5 %`; Phase 3 `9.3 %`; `2×` miss `3208.2 s = 53.5 min`; `5×` miss `4029.3 s = 67.2 min`. **26 rows**, all **73** processes accounted. Check no stale total survives — every surviving `3.2 s`, `1.7×`, `2933.9` and `3667.4` should be historical, quotational, or the deviation record. | §8 |
| **V3** | **§13.1 item 27** pins the arena's import set. Check it against `headspace_arena.py:28-46` and `headspace_mint.py:68` **as they actually are**, and rule whether it says enough for a code lineage to verify Phase 1g's number without reconstructing the set from source. The list is now `1–27`; confirm contiguity. | §13.1 |
| **V4** | **The five limb quotations, by subtraction.** Verbatim, complete, and inside the `R12:NNN-NNN` range each cites; then subtract them from round 12's Repair paragraphs and report the residue. Note M-3 has no *"Repair:"* sentence — v13 quotes its prescriptive clause and says so; rule that treatment. | §14 |
| **V5** | **Re-run the audit** (script in §14.2); byte-compare against the embedded transcript; confirm exit `0`. `CHANGED §14.2 +0 chars` recurs — verify it by direct diff as round 12 did, rather than accepting the explanation. | §14.1 |
| **V6** | **Break the self-exclusion in the form that bites.** The plain splice is **vacuous** in v13 (no row or limb cites §14.1) — round 12 established this for v12. Do what round 12 did: splice **and** insert a synthetic §14.1-citing row, and report both results. | §14.1, §14.2 |
| **V7** | The three Minors: §7.3's `98` under the **closed** interval (round 12 got `98 / 116 / 118`, and `96 / 114 / 116` excluding the two self-referential endpoint tokens); §6.1's `ρ` reproducing **only on row-renormalised** keys; §6.2's `0.15`–`0.24`. | §7.3, §6.1, §6.2 |
| **V8** | **Rebuild the arms from §3.4 yourself**; `GATE-C01PARITY` one bit-exact predicate; the un-normalised misreading at `1.878e-06` / `1.609e-06`, both under `2e-6`. | §3.4, §6 |
| **V9** | `ρ*` `0.968176` / `0.977223`; 26/26 `ρ_raw` at 6 dp; trained-head `0/18`. | §6.1 |
| **V10** | Holm counterexample; `n ≤ 12`; §3.7's two blocks with two verbs. | §5.5, §5.4.1, §3.7 |
| **V11** | §7.9's sum: heading `v1–v13`, `7+1+4+0+0+0+0+0+0 = 12`, `22+4+2+1+1+1+1+1 = 33`, `89+21+6+3+3+3+3+3 = 131`, agreeing with §7.8 and the footer. | §7.9 |
| **V12** | §6 has **20** gate rows, `12 G / 6 L / 2 R`, matching §5.6; items 10, 15, 19, 22 and 27 carry their repairs; all 37 digests recompute (eight rows carry the `…` ellipsis). | §6, §13.1, §11 |

---

## 4. What you must assess

### A. The deviation (the round's central judgement)

Round 12 prescribed `3.2 s`; v13 carries `3.8 s`. The disclosure is in §14's limb cell, in the
paragraph above the limb table, and in §7.7's parenthetical. Rule it **forced by measurement** or
**a substitution dressed as one**, and say what your own timing range was. Consider both failure
directions: a document that quietly ignores a prescription, and a document that follows a
prescription into a false statement.

### B. Where v13's own repair could have opened a seam

v13 changed: header, §6.1, §6.2, §7.3, §7.7, §7.9, §8, §9, §13.1, §14, §14.1, §14.2, §15. **Every
round for six rounds has found its finding inside the previous round's repair**, and the last two
were both inside Phase 1g. Look there first.

### C. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** Answered *no for all twenty* by rounds
6–12, each re-deriving. Do the same.

### D. The process rules

- **`rule_1_compute_projection`.** Twelve rounds; ten items found; round 12 searched three axes for
  an eleventh and found none, pricing the new heartbeat-line axis at `0.011 s`. **Name the axis you
  search.** The last two findings were a per-process fixed cost and a per-process *unit basis*, not
  payload loops.
- **`rule_2_heartbeat`.** Does anything in v13 change an interval? The arena-startup span is now
  quoted at `3.00–3.75 s` rather than `1.82–1.85 s`.

### E. Honesty

- Does v13 claim any repair the artifact does not contain, or any repair **narrower** than
  prescribed? Diff v12→v13 and check all four rows and all five limbs against round 12's text.
- Blindness across v1–v13: grep every decimal in the **closed** `[0.6, 0.99]` and classify anything
  new. §7.3 claims the scope `v1–v13` and that v13 adds no accuracy — its new measurements are
  timings and arithmetic checks. Verify rather than inherit.
- The footer discloses one machine-generated side effect: a `.pyc` bytecode cache written into
  `scripts/analysis/__pycache__/` by v13's import timings. Confirm no `.py` source moved and that
  all 37 digests still recompute.

---

## 5. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | Could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §3, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or scope without inverting it — including a repair landed **narrower** than prescribed. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**.

**Do not grade on trajectory, in either direction.** Thirteen rounds is evidence of nothing. Round
12 put it best and it is the standard for this round: *"Softening it because a GO is one finding
away would be grading on trajectory, which the brief forbids in both directions; so would inventing
a finding to avoid looking captured."* If the design and the record are clean, say **GO** plainly;
if not, name the specific defect.

---

## 6. What a GO does and does not authorize

A GO authorizes **nothing to run**. Before any job: (1) freeze with hashes; (2) a **separate**
independent code/resource review lineage over the executable reaching its own `0C/0H/0I` — round 12
notes for that lineage that §13 item 23 should be read broadly and that item 27 is addressed to it;
(3) main-dialogue authorization. A GO is not authority to write `TARGET_STATE.json`.

---

## 7. Deliverable

1. The `0C/0H/0I`-form tally and **GO** or **REVISE**.
2. Every finding with severity, `file:line` citation, and the concrete repair.
3. Your verification result for **all twelve** §3 items, including **your own timing range** for the
   arena's import set.
4. **Your limb audit by subtraction**: for each of the five limbs, `FAITHFUL` / `TRUNCATED` /
   `NARROWED`, plus the residue after subtracting them from round 12's Repair paragraphs.
5. **An explicit ruling on the deviation** (§4.A): forced by measurement, or not.
6. An explicit ruling on each of the six open issues in v13 §15.
7. An explicit ruling on §4.C: any gate that can fire on a warranted CLOSE — all twenty.
8. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly.

---

*Read-only. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit is authorized by this document, and `TARGET_STATE.json` must not be modified.*
