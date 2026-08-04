# C06 `$0` falsifier — fresh independent design review request, **ROUND 14**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM job,
no Modal, no GPU, no cache write, no commit, no edit to `TARGET_STATE.json`. Read-only
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint checkpoints,
plus `sha256sum` and process-wall timings, is expected. **Up to four CPU head mints** (~40 s each)
are permitted; rounds 8–13 all declined them with stated reasons, which is a legitimate choice.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V14.md` (v14, unfrozen, sha256
`d80bbb44911daef9e772dfe1246ffa71876147e82d7f8b4bce6d83d5c34b0a46`, 173368 bytes, 2321 lines).

You are a **fresh reviewer**, independent of rounds 1–13 and of the designer.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M → 3C/3H/6I+5M →
2C/3H/5I+6M → 4C/2H/3I+4M → 2C/2H/4I+6M → **0C**/2H/2I+4M → **0C/0H**/3I+4M → **0C/0H/1I**+1M →
**0C/0H/1I**+3M → **0C/0H/1I**+2M.

---

## 0. Where this stands, and what your round is actually for

**The science layer is closed and seven independent reviewers have confirmed it.** Round 13 rebuilt
all thirteen arms from §3.4's prose alone at `0.000e+00`, reproduced 26/26 `ρ` at 6 dp, 16/16 C01
accuracies and 16/16 net-fix integers, 37/37 digests, the Holm counterexample through C01's own
`holm_adjust`, and re-derived all twenty gates as unable to fire on a warranted CLOSE.

**The record has been clean for three rounds.** Round 13: 5/5 limbs verbatim, complete, and inside
their cited line ranges; both Repair paragraphs subtracting to bare scaffolding; zero stale totals.

**The last three findings have all been inside the previous round's repair, and all three were about
evidence rather than arithmetic.** Round 11 found a per-process cost priced in no row. Round 12
found that the row created to fix it carried a unit measured over an import set omitting `sklearn`.
Round 13 found that the **sample size** reported for the corrected measurement did not reconcile —
§7.7's table specified `44` timed starts, §7.9 and the footer said `24` "in total", and the stated
breakdown summed to `40`. Round 13's grading note is the standard: *"When the warrant for overriding
a reviewer is 'I measured more than you did,' the count is load-bearing."*

**Round 13 also ruled for the deviation it inherited.** It timed the arena's import set 24 times at
`3.094–3.717 s`, ten runs above `3.2 s`, and ruled round 12's prescribed `3.2 s` *"falsified by my
own data"* — *"a measurement compelling a departure, not a preference dressed as one."* The carried
`3.8 s` is unchanged in v14.

**Your round has two jobs.** The first is arithmetic and cheap: **grep every run-count and confirm
no figure asserts a total its stated parts cannot produce.** v14 answers round-13 I-1 by
re-measuring §7.7's decomposition as **one uniform sample — `7` rungs × `8` runs = `56` timed
starts, one command** — and restating the arena-class evidence as a **three-party split table**
(this document `8`, round 12 `3`, round 13 `24`, pooled `35`). Every mention should now carry that
one accounting.

**The second is a judgement.** Round 13 prescribed *"arithmetic only, no new measurement"*; **v14
re-measured instead**, and records that as a **widening** with its reason: the prescribed arithmetic
would have produced a *consistent* statement of an *uneven* sample (`4` runs on five rungs, `10` on
two), and the sample size is the entire warrant for overriding round 12. Rule it — warranted, or a
designer preferring its own instrument to a reviewer's instruction? Note the prescribed outcome is
also delivered.

**A third, smaller one v14 raises against itself:** §7.9 now corrects **v13's** recorded spend from
`≈ 1 / ≈ 3` to `≈ 2 / ≈ 4` on a re-counted `52` starts. Correcting a *prior* version's historical
figure is unusual. §15 item 3 asks you to rule whether that is right or whether a recorded figure
should stand with an erratum beside it.

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
5. **`refine-logs/C06_FALSIFIER_PREREG_REVIEW_R13.md` in full** — it is the specification v14
   answers and the source your subtraction must be made against. Then the earlier twelve as needed.
6. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V13.md` (for the diff) and earlier drafts as context.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V14.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` — `prepare_views:1294-1304`,
`contrast_blocks:1242-1265`, `l2_rows:1183-1205`, `orthogonal_blocks:1272`, the algebra guard
`:1372-1377`, the two `fix_break` sites `:1725` / `:2702-2714`, `select_strongest_ordinary_control`
(guards `:1940-1948`, ranking `:1955-1962`), `:2724`, `displacement_audit:1965-2076`,
`holm_adjust:1775-1784`; `src/model/classifier.py:81-82`, `:140-141`, `:146`;
**`scripts/analysis/headspace_arena.py:28-46`** (the import block item 27 pins), `:59`, `:72`,
`:75-89`, `:92-93`; **`scripts/analysis/headspace_mint.py:68`** (the `sklearn` entry point),
`:82-94` (`runtime_block`), `:192-194`, `:199`, `:306-307`, `:321-325`;
`scripts/analysis/headspace_fidelity.py`; `scripts/analysis/mechfix_ops.py:45-49`, `:82`, `:94`;
`configs/c01/c01_a0_v{2,3,4}.json`;
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`;
`artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz`.

---

## 3. Verify these facts yourself

| # | claim | where |
|---|---|---|
| **V1** | **Every run-count reconciles.** `7 × 8 = 56` timed starts here; arena-class pool `8 + 3 + 24 = 35` at `3.094–3.717 s`; v13 re-counted at `52`. Grep §7.7's two tables, its `U11` row, §7.9's v13 and v14 terms and its cumulative sum, §7.3, §8's Phase 1g row, §9 and the footer, and confirm **no figure asserts a total its parts cannot produce**. This is round-13 I-1's exact defect. | §7.7, §7.9, §7.3, §8, §9 |
| **V2** | **Re-measure the arena's import set yourself**, rung by rung as §7.7's table specifies, and report your range. v14 measured rung 7 at `3.10–3.70 s` over 8 runs; round 13 at `3.094–3.717 s` over 24; round 12 at `3.12–3.27 s` over 3. Confirm `3.8 s` still bounds the pool. | §7.7, §8 |
| **V3** | **Phase 1g and the re-multiplied column.** `1 × U11`(arena class) at `3.8 s`; printed column sums to **`2934.5`**; `× 1.25 = 3668.1`; `48.9` / `61.1 min`; mint `85.5 %`; Phase 3 `9.3 %`; `2×` miss `3208.2 s`; `5×` miss `4029.3 s`. **26 rows**, all **73** processes accounted. Check no stale total survives. | §8 |
| **V4** | **The six limb quotations, by subtraction.** Verbatim, complete, inside the `R13:NNN-NNN` range each cites; then subtract them from round 13's Repair paragraphs and report the residue. **Rule the widening** (§4.A). | §14 |
| **V5** | **Re-run the audit** (script in §14.2); byte-compare against the embedded transcript; confirm exit `0`. `CHANGED §14.2 +0 chars` recurs — verify by direct diff that it is **four substitution classes over six lines** (round-13 M-2 corrected v13's *"exactly five"*). | §14.1 |
| **V6** | **Break the self-exclusion in the form that bites.** The plain splice is **vacuous** against v14 — no row or limb cites §14.1. Splice **and** insert a synthetic §14.1-citing row, and report both results. | §14.1, §14.2 |
| **V7** | The two Minors: §6.1's row-norm span `0.027`–`0.56` and gap factor `3.5×`–`7.6×` over all 36 banked `K_train` matrices (round 13 got HateMM `0.0410`–`0.5596`, ZH `0.0271`–`0.2882`); §14.1's four-classes-over-six-lines. | §6.1, §14.1 |
| **V8** | **Rebuild the arms from §3.4 yourself**; `GATE-C01PARITY` one bit-exact predicate; the un-normalised misreading at `1.878e-06` / `1.609e-06`, both under `2e-6`. | §3.4, §6 |
| **V9** | `ρ*` `0.968176` / `0.977223`; 26/26 `ρ_raw` at 6 dp; trained-head `0/18` **on row-renormalised keys**. | §6.1 |
| **V10** | Holm counterexample; `n ≤ 12`; §3.7's two blocks with two verbs. | §5.5, §5.4.1, §3.7 |
| **V11** | §7.9's sum: heading `v1–v14`; mints `7+1+4+0+0+0+0+0+0 = 12`; wall `22+4+2+1+1+1+1+2+2 = 36`; CPU `89+21+6+3+3+3+3+4+4 = 136`. The **v13 terms are `2` and `4`**, carrying round-13 I-1's correction from the `1` and `3` v13 recorded for itself — check the printed terms re-derive and that §7.8 and the footer agree. | §7.9 |
| **V12** | §6 has **20** gate rows, `12 G / 6 L / 2 R`; §13.1 defines **27** contiguous items; items 10, 15, 19, 22, 27 carry their repairs; 37/37 digests recompute (eight rows carry the `…` ellipsis). | §6, §13.1, §11 |

---

## 4. What you must assess

### A. The widening (the round's central judgement)

Round 13 prescribed *"Repair — one line, arithmetic only, no new measurement"* with two disjuncts.
v14 took neither and re-measured. §14's paragraph *"Why I-1 was answered by measurement"* gives the
reason. Rule it warranted or not, and say what you would have done. Consider both failure
directions: a document that ignores a prescription's stated method, and a document that follows the
method into a statement that is consistent but describes an uneven sample.

### B. Where v14's own repair could have opened a seam

v14 changed: header, §6.1, §7.3, §7.7, §7.9, §8, §9, §14, §14.1, §14.2, §15. **Every round for
seven rounds has found its finding inside the previous round's repair**, and the last three were all
in the Phase 1g / §7.7 evidence chain. Look there first.

### C. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** Answered *no for all twenty* by rounds
6–13, each re-deriving. Do the same.

### D. The process rules

- **`rule_1_compute_projection`.** Fourteen rounds, ten items. Round 12 searched three axes, round 13
  two more; none yielded, and round 13's search incidentally corroborated `U2a`–`U2d`. **Name the
  axis you search.**
- **`rule_2_heartbeat`.** Does anything in v14 change an interval? The arena-startup span is quoted
  at `3.094–3.717 s`.

### E. Honesty

- Does v14 claim any repair the artifact does not contain, or any repair **narrower** than
  prescribed? Diff v13→v14 and check all three rows and all six limbs against round 13's text.
- Blindness across v1–v14: grep every decimal in the **closed** `[0.6, 0.99]` and classify anything
  new. §7.3 claims the scope `v1–v14` and that neither v13 nor v14 adds an accuracy. Round 13 warns
  that a regex admitting leading-dot decimals picks up the fragment `.27` from `defined 1..27`;
  the leading-digit convention is the one that reproduces.
- The footer discloses that **v13's** timings wrote one `.pyc` into `scripts/analysis/__pycache__/`
  and that **v14's `56` starts wrote none** (11 files before and after). Confirm, and confirm no
  `.py` source moved.

---

## 5. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | Could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §3, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or scope without inverting it — including a repair landed **narrower** than prescribed. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**.

**Do not grade on trajectory, in either direction.** Fourteen rounds is evidence of nothing. Round 13
held both lines at once and it is the standard for this round: it declined to grant a GO early, and
it declined to invent a second finding, **dropping one candidate at its C.6 after measuring it**. If
the design and the record are clean, say **GO** plainly; if not, name the specific defect.

---

## 6. What a GO does and does not authorize

A GO authorizes **nothing to run**. Before any job: (1) freeze with hashes; (2) a **separate**
independent code/resource review lineage over the executable reaching its own `0C/0H/0I` — rounds 12
and 13 note for that lineage that §13 item 23 should be read broadly, that item 27 binds any
trimming or extension of the arena's import set to a re-measurement, and that item 22's placement of
the `GATE-FLOOR` vote in the arena is what Phase 1f's `150` is priced against; (3) main-dialogue
authorization. A GO is not authority to write `TARGET_STATE.json`.

---

## 7. Deliverable

1. The `0C/0H/0I`-form tally and **GO** or **REVISE**.
2. Every finding with severity, `file:line` citation, and the concrete repair.
3. Your verification result for **all twelve** §3 items, including **your own timing range** for the
   arena's import set and **your own grep of the run-counts**.
4. **Your limb audit by subtraction**: for each of the six limbs, `FAITHFUL` / `TRUNCATED` /
   `NARROWED`, plus the residue after subtracting them from round 13's Repair paragraphs.
5. **An explicit ruling on the widening** (§4.A) and on whether correcting v13's recorded spend term
   is right (§15 item 3).
6. An explicit ruling on each of the six open issues in v14 §15.
7. An explicit ruling on §4.C: any gate that can fire on a warranted CLOSE — all twenty.
8. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly.

---

*Read-only. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit is authorized by this document, and `TARGET_STATE.json` must not be modified.*
