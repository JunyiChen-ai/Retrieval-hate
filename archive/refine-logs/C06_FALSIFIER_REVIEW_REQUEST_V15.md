# C06 `$0` falsifier — fresh independent design review request, **ROUND 15**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM job,
no Modal, no GPU, no cache write, no commit, no edit to `TARGET_STATE.json`. Read-only
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint checkpoints,
plus `sha256sum` and process-wall timings, is expected. **Up to four CPU head mints** (~40 s each)
are permitted; rounds 8–14 all declined them with stated reasons, which is a legitimate choice.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md` (v15, unfrozen, sha256
`75e3aa84a39f3c276be8b5b7fb1271e83c0c2e7e3b0ad0480b47465a81408228`, 176903 bytes, 2338 lines).

You are a **fresh reviewer**, independent of rounds 1–14 and of the designer.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M → 3C/3H/6I+5M →
2C/3H/5I+6M → 4C/2H/3I+4M → 2C/2H/4I+6M → **0C**/2H/2I+4M → **0C/0H**/3I+4M → **0C/0H/1I**+1M →
**0C/0H/1I**+3M → **0C/0H/1I**+2M → **0C/0H/1I**+2M.

---

## 0. Where this stands, and what your round is actually for

**The science layer is closed and eight independent reviewers have confirmed it.** Round 14 rebuilt
all thirteen arms from §3.4's prose alone at `0.000e+00` on both datasets, reproduced 26/26 `ρ` at
6 dp, 16/16 C01 accuracies and 16/16 net-fix integers, 37/37 digests, the Holm counterexample **and
its three-way equality** through C01's own `holm_adjust`, re-multiplied **all 23** §8 unit×count
products with zero mismatches, and re-derived all twenty gates as unable to fire on a warranted
CLOSE.

**The compute projection is settled.** Round 13's count reconciliation and round 14's independent
re-run of the same measurement (56 starts in one command, 95.7 s against the document's "about 96
seconds") closed that thread. Round 14 ruled the widening that closed it **warranted** — *"and I
would have done the same"* — because the re-measurement **pooled with** rounds 12 and 13 rather than
replacing them.

**The last finding was not a number at all.** It was a **false factual claim about the document's own
verification mechanism**: §14.1 said the plain self-exclusion counterfactual was vacuous against v14
and §15 told round 15 the same, while §14.1's own embedded transcript printed
`OK M-2 cites §14.1` two paragraphs away. Round 14 ran the splice: it exits `1`. **The claim was an
unchecked inheritance from v12 and v13 — the precise defect round-13 M-2 had named — sitting in the
paragraph that lands round-13 M-2's repair.**

**Your round's central job follows directly from that.** v15's corrected sentence was derived from
this document as finalized, in a stated order: §14.2 fixed point first, splice second, sentence
third, re-verify fourth. **Do not take it on trust.** §3 V1 asks you to run **both** forms of the
construction yourself and report both. The plain form is **expected to exit `1`**, not `0` — one
failing row, one failing limb, `named by a row but unchanged: ['14.1']`. If it does not, that is
this round's finding, and it is the same finding twice running.

**Two smaller judgements.** v15 records a **non-finding** in §13.1 as item 28, on the reasoning that
§13 is the code lineage's sole input and a note living only in a review file is lost — rule whether
that is right or whether it inflates the handoff list. And §7.9 now supplies the bridge from v13's
printed `44` to its executed `52`, which round 14 made a **condition** of its ruling that correcting
a prior version's spend in place is legitimate — check the bridge against v13's own table.

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
5. **`refine-logs/C06_FALSIFIER_PREREG_REVIEW_R14.md` in full** — it is the specification v15
   answers and the source your subtraction must be made against. Then the earlier thirteen as needed.
6. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V14.md` (for the diff **and** for the splice construction)
   and earlier drafts as context.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` — `prepare_views:1294-1304`,
`contrast_blocks:1242-1265`, `l2_rows:1183-1205`, `orthogonal_blocks:1272`, the algebra guard
`:1372-1377`, the two `fix_break` sites `:1725` / `:2702-2714`, `select_strongest_ordinary_control`
(guards `:1940-1948`, ranking `:1955-1962`), `:2724`, `displacement_audit:1965-2076`,
`holm_adjust:1775-1784`, `id_hash_permutation:1787`; `src/model/classifier.py:81-82`, `:140-141`,
`:146`; `scripts/analysis/headspace_arena.py:28-46`, `:59`, `:72`, `:75-89`, `:92-93`;
`scripts/analysis/headspace_mint.py:68`, `:82-94`, `:106-116`, `:192-194`, `:199`, `:306-307`,
`:321-325`; `scripts/analysis/headspace_fidelity.py`; `scripts/analysis/mechfix_ops.py:45-49`,
`:82`, `:94`; **`c09_guard/sitecustomize.py`** (item 28); `configs/c01/c01_a0_v{2,3,4}.json`;
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`;
`artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz`.

---

## 3. Verify these facts yourself

| # | claim | where |
|---|---|---|
| **V1** | **The self-exclusion, BOTH forms — the round's central check.** Splice **v14's** §14.1 into a copy of v15 with no synthetic row and run the §14.2 script: v15 states this yields `UNCHANGED §14.1`, `FAIL I-1 cites §14.1 -- NOT DIFFED`, `rows failing: 1`, one failing limb, `named by a row but unchanged: ['14.1']`, **exit `1`**. Then add one synthetic §14.1-citing row: v15 states `rows failing: 2`, exit `1`. **Report both.** The plain form is expected to exit `1`; if it exits `0`, that is a finding. | §14.1, §14.2 |
| **V2** | **Re-run the audit** against the finished on-disk v15; byte-compare with the embedded transcript; confirm exit `0`. `CHANGED §14.2 +0 chars` recurs for the fourth version — verify by direct diff that it is **four substitution classes over six lines**. | §14.1 |
| **V3** | **The four limb quotations, by subtraction.** Verbatim, complete, inside the `R14:NNN-NNN` range each cites; then subtract them from round 14's Repair paragraphs and report the residue. **Note limb 1 is landed for v15 rather than transcribed for v14, and says so** — rule that treatment. | §14 |
| **V4** | **The v13 bridge.** §7.9 derives `52` from v13's printed `44`: an eighth unreported rung (`+4`) and rung 5's `(10 runs)` being an increment where rung 7's `(14 runs)` was a total (`+4`). Check against **v13's own §7.7 table**, and rule whether round 14's condition — *"provided the correction states what the earlier version got wrong and how the two reconcile"* — is now met. | §7.9 |
| **V5** | **§13.1 item 28, recorded not prescribed.** Round 14 raised the `PYTHONPATH`/`c09_guard` wiring as *"a code-side wiring item, not a design defect"*. Rule whether recording a non-finding in §13 is right. List is `1–28`; confirm contiguity, and confirm items 10, 15, 19, 22, 27 still carry their repairs. | §13.1 |
| **V6** | **§8 is untouched this round and must still re-multiply.** 26 rows, column sums to **`2934.5`**; `× 1.25 = 3668.1`; `48.9` / `61.1 min`; mint `85.5 %`; Phase 3 `9.3 %`; all **73** processes accounted. Check no stale total survives. | §8 |
| **V7** | **§7.9's sum**: heading `v1–v15`; mints `= 12`; wall `22+4+2+1+1+1+1+2+2+1 = 37`; CPU `89+21+6+3+3+3+3+4+4+1 = 137`; v13's terms carry round-13 I-1's correction. | §7.9 |
| **V8** | **Rebuild the arms from §3.4 yourself**; `GATE-C01PARITY` one bit-exact predicate; the un-normalised misreading at `1.878e-06` / `1.609e-06`, both under `2e-6`; `GATE-ROWSUBSET` at `0.000e+00`. | §3.4, §6 |
| **V9** | `ρ*` `0.968176` / `0.977223`; 26/26 `ρ_raw` at 6 dp; trained-head `0/18` **on row-renormalised keys**. Round 14 cautions that §6.1's gap factor is the ratio of **order statistics** (`3.47×`–`7.64×`); the per-cell ratio is a different quantity (`2.45×`–`9.45×`). | §6.1 |
| **V10** | Holm counterexample and its three-way equality; `n ≤ 12`; §3.7's two blocks with two verbs. | §5.5, §5.4.1, §3.7 |
| **V11** | §6 has **20** gate rows, `12 G / 6 L / 2 R`, matching §5.6; 37/37 digests recompute (eight rows carry the `…` ellipsis); the four new-code paths absent. | §6, §11 |
| **V12** | **The arena's import set**, if you choose to re-measure: §7.7's seven rungs at `8` runs each; arena-class pool `8 + 3 + 24 = 35` at `3.094–3.717 s`; `3.8 s` bounds it. Round 14 got `3.070–3.540 s` on its own 8 runs. | §7.7, §8 |

---

## 4. What you must assess

### A. The corrected sentence, and whether the correction is itself checked

Round-14 I-1 was an unchecked inheritance. **The obvious failure mode for its repair is to be one
too.** v15 states the ordering it used — fixed point, splice, sentence, re-verify — and §14 records
that ordering as the point. Verify the sentence against your own run (V1), and rule whether the
stated ordering is a real safeguard or a claim about process that is itself unverifiable.

### B. Where v15's own repair could have opened a seam

v15 changed: header, §7.3, §7.9, §13.1, §14, §14.1, §14.2, §15. **Every round for fourteen rounds
has found its finding inside the previous round's repair.** §14.1 is where the last one was.

### C. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** Answered *no for all twenty* by rounds
6–14, each re-deriving. Do the same.

### D. The process rules

- **`rule_1_compute_projection`.** Fifteen rounds, ten items. Rounds 12–14 searched six axes between
  them; none yielded, and they incidentally corroborated `U2a`–`U2d`, `U7`, `U8` and `U3`. **`U4` is
  the only substantial unit still uncorroborated end to end** (`273.7 s`, `9.3 %`); round 14 priced
  its one identified companion at `0.13 %` of the total. **Name the axis you search.**
- **`rule_2_heartbeat`.** Does anything in v15 change an interval? v15 touches no §8 row.

### E. Honesty

- Does v15 claim any repair the artifact does not contain, or any repair **narrower** than
  prescribed? Diff v14→v15 and check all three rows and all four limbs against round 14's text.
- Blindness across v1–v15: grep every decimal in the **closed** `[0.6, 0.99]`. §7.3 claims the scope
  `v1–v15` and that v15 measures nothing new at all. Round 14 verified the corpus total is unchanged
  at `118` from v12 through v14. Round 13 warns that a regex admitting leading-dot decimals picks up
  `.27` from `defined 1..27` (now `1..28`); the leading-digit convention is the one that reproduces.
- The footer states **v15 ran no timings** and that `scripts/analysis/__pycache__/` holds **11**
  files. Confirm, and confirm no `.py` source moved.

---

## 5. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | Could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §3, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or scope without inverting it — including a repair landed **narrower** than prescribed. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**.

**Do not grade on trajectory, in either direction.** Fifteen rounds is evidence of nothing. Round 14
held both lines at once and is the standard: it declined to grant a GO early, and it declined to
inflate the finding it had — *"I-1 is graded Important and not High because nothing is narrowed and
no quantity moves, and I raised no second finding I could not measure."* If the design and the record
are clean, say **GO** plainly; if not, name the specific defect.

---

## 6. What a GO does and does not authorize

A GO authorizes **nothing to run**. Before any job: (1) freeze with hashes; (2) a **separate**
independent code/resource review lineage over the executable reaching its own `0C/0H/0I` — rounds
12–14 note for that lineage that §13 item 23 should be read broadly, that item 27 binds any trimming
**or extension** of the arena's import set to a re-measurement of `U11` and a re-carry of Phase 1g,
that item 22's placement of the `GATE-FLOOR` vote in the arena is what Phase 1f's `150` is priced
against, and that item 28's `PYTHONPATH` export is what makes §12's third guard layer active rather
than merely importable; (3) main-dialogue authorization. A GO is not authority to write
`TARGET_STATE.json`.

---

## 7. Deliverable

1. The `0C/0H/0I`-form tally and **GO** or **REVISE**.
2. Every finding with severity, `file:line` citation, and the concrete repair.
3. Your verification result for **all twelve** §3 items, including **both splice forms with their
   exit statuses** (V1).
4. **Your limb audit by subtraction**: for each of the four limbs, `FAITHFUL` / `TRUNCATED` /
   `NARROWED`, plus the residue after subtracting them from round 14's Repair paragraphs.
5. **An explicit ruling on §4.A** (is the correction itself checked?), on the v13 bridge (§3 V4),
   and on recording a non-finding as §13.1 item 28 (§3 V5).
6. An explicit ruling on each of the six open issues in v15 §15.
7. An explicit ruling on §4.C: any gate that can fire on a warranted CLOSE — all twenty.
8. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly.

---

*Read-only. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit is authorized by this document, and `TARGET_STATE.json` must not be modified.*
