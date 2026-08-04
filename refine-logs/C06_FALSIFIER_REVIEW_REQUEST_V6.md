# C06 `$0` falsifier — fresh independent design review request, **ROUND 6**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM
job, no Modal, no GPU, no cache write, no commit, no edit to `TARGET_STATE.json`. Read-only
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint
checkpoints, plus `sha256sum`, is expected and encouraged. A single CPU head mint on the login
node (~40 s) is permitted if you need it to check §7.9; it writes nothing outside a scratchpad.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V6.md` (v6, unfrozen).

You are a **fresh reviewer**, independent of rounds 1–5 and of the designer. You have not seen the
designer's reasoning and must not ask for it.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M → 3C/3H/6I+5M.
Round 5's disposition audit was **18 of 18 VERIFIED ADOPTED** — the first with no failed adoption.
Every round's Criticals have been found **by executing C01's frozen code**, and every round has
found them in the seam the *previous* round's repair opened:

| round | where its Criticals were |
|---|---|
| 1 | the dispersion gate, the OOD transplant, the unanchored arm algebra |
| 2 | the null contract that round 1's builder repair required |
| 3 | the mask convention and the majority constant that round 2's contract required |
| 4 | the viability gate, the S6 axis and the fold count that round 3's repairs required |
| 5 | the dataset axis, the absence rule and S7 — the structures round 4's repairs created |

**v6's repairs create their own seams. Look there first:** the dataset-axis drop rule (§5.6), the
frozen 92-family with `p = 1` padding (§5.5), S7's declared head-space statistic (§5.2.2), the new
global `GATE-RHORAW` (§6), and the reference-arm change that now couples S3, S6, S7 and
`GATE-SELFTEST` to one arm (§5.2.1).

**Audit by execution, not by reading.** Treat §14 as an index of things to check, never as
evidence. Round 2 found three of twenty-three adoptions not repaired; round 4 found one
*textually present but arithmetically false*; round 5 found a claimed repair undone by a rule in
the same subsection.

**One item deserves special scrutiny: D-1.** v6 records a defect **no review round found**, and
which **three rounds affirmatively verified in its wrong form** — S6's reference arm. If three
independent reviewers can converge on a wrong reading of a frozen source, so can a fourth. Check it
from the source, not from v6's account of it.

---

## 0. What C06 is, so you need no prior context

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

## 1. Read first, in this order

1. `CLAUDE.md`, `AGENTS.md`.
2. `TARGET_STATE.json`: `gate0_reopen_2026_07_31.dispositions.gated[0]` (**verbatim**);
   `iteration_8_queue_state_2026_08_04`;
   `process_rule_compute_projection_and_heartbeat_2026_08_04`;
   `iteration_8_stage0_bounded_extraction_amendment`.
3. `TARGET_FINDINGS.md` — **F118** in full; skim **F88**, **F113**.
4. `refine-logs/GATE0_REOPEN_2026-07-31.md` §4.4;
   `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §3;
   `refine-logs/C09_A0_V17_RECORD.md` §2 and §8.1.
5. All five prior reviews: `C06_FALSIFIER_PREREG_REVIEW{,_R2,_R3,_R4,_R5}.md`.
6. The superseded drafts for context: `C06_FALSIFIER_PREREG_DRAFT{,_V2,_V3,_V4,_V5}.md`.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V6.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` — especially the **two**
`fix_break` sites `:1725` and `:2702-2714`, `select_strongest_ordinary_control:1940-1948`, the
consistency `die()` at `:2724`, `displacement_audit:1965-2060` and its zero-fix guard `:1989-1996`,
`paired_bootstrap:1742-1772` with `one_sided_raw_p:1769`, `holm_adjust:1775-1784`,
`id_hash_permutation:1787+`, `l2_rows:1183-1205`, `prepare_views:1294`/`:1381-1386`,
`orthogonal_blocks:1272`, the guard builds `:1357-1370`, `die:392-393`;
`src/model/classifier.py:81-82` and `:140-146`; `scripts/analysis/headspace_mint.py:192-194`,
`:199`, `:203-216`, `:321-325`; `scripts/analysis/headspace_arena.py:75-89`;
`scripts/analysis/mechfix_ops.py:94`; `configs/c01/c01_a0_v{2,3,4}.json`;
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`.

---

## 2. Verify these facts yourself

Report `VERIFIED` / `MISMATCH` with the value you obtained, for **every** item.

| # | claim | where |
|---|---|---|
| **V1** | All 21 sha256 digests in §11 match disk; both provenance chains sha-gated in source. | §11 |
| **V2** | **D-1.** C01 has two `fix_break` sites: `:1725` uses `retrieval.fix_break_reference = endpoint_std` for the **reporting** field, `:2702-2714` uses `select_strongest_ordinary_control` for the **decision** check `net_fixes`. The executed `C01_A0_OUT.json` has `net_fixes.reference = "common"` (HateMM) and `"endpoint_concat"` (MHC-ZH). So v5's S6 reference was wrong. | §5.2.1 |
| **V3** | **S7's missing constants.** `transforms.max_small_displacement_fix_fraction = 0.5`; `small_displacement_train_quantile = 0.1`; C01's statistic is `min(‖d_img‖, ‖d_text‖)` on raw features; the consistency `die()` at `:2724`. | §5.2.2 |
| **V4** | **H-1's corrected floor.** Executing `holm_adjust` over `m = 92` with the witness's 24 (or 22) hypotheses at `1/2001` and the rest at `0.5`: **24/24** (22/22) reject; degrade one to `2/2001` and it becomes 23/24. The floor is the witness's own comparators, **not** 42 of 92. | §5.5 |
| **V5** | **The family invariance.** The witness rejects `24/24` identically under `m = 92` padded `0.5`, `m = 92` padded `1.0`, and `m = 46`. | §5.5 |
| **V6** | **S5 feasibility.** `1/257 = 0.0038911`; a shuffle family of `n` needs `n × 0.0038911 ≤ 0.05`, so `n ≤ 12`; the frozen `n = 4` needs `0.01556`. | §5.4.1 |
| **V7** | **`GATE-FLOOR`'s discharge.** Re-mint HateMM `s0/fold0` through the wrapper path and check the native deployed vote against banked `fold_acc_deployed[0] = 0.8725`. (~40 s; the only item needing a mint.) | §7.9 |
| **V8** | **The trained-head residual.** `θ = 45` residual `2.384e-07` (`8.4×` inside the `2e-6` bar), `θ = 0` `1.490e-08`, median `‖l2(h_ow) − l2(h_std)‖ = 0.2301` on a trained head vs `0.0032` untrained. | §7.9 |
| **V9** | The null-contract defect and repair; the row-subset identity `0.000e+00`; `GATE-C01PARITY` bit-exact both datasets. | §7.4, §7.6 |
| **V10** | `ρ*` `0.968176` / `0.977223`, all 26 `ρ_raw` at 6 dp, the trained-head reference `0/18` above `ρ*`. | §6.1 |
| **V11** | **Every** population-derived constant in §3.7's table, including the **three new rows** (S7's quantile space, the `0.5` threshold, the reference-arm rule). | §3.7 |
| **V12** | §8's product column sums to **`2927.6`**; `× 1.25 = 3659.5`; mint share `85.7 %`; Phase 3 `9.3 %`; and §6's table has **twenty** rows, `12 G / 6 L / 2 R`, matching §5.6's two lists. | §6, §8 |

---

## 3. What you must assess

### A. D-1 — the defect three rounds got wrong

Check it from the source. Then rule: is inheriting C01's **accuracy-based** selector
(`select_strongest_ordinary_control`) pre-registration-safe? v6's argument is that it is a
deterministic *rule* evaluated at run time, so it introduces no researcher degree of freedom even
though the selected arm is unknowable in advance. Is that right, or does a reference arm chosen by
measured accuracy contaminate a preregistration? Note what rides on the answer: S3's
`max_{c∈C}`, S6, S7 and `GATE-SELFTEST` are now all coupled to that one arm.

### B. The three round-5 Criticals, and the seams they open

- **C-1 → the dataset axis (§5.6).** The rule is now *"fails on any dataset ⇒ dropped on both"*.
  Is there any remaining path on which a lineage clean on one dataset only reaches a verdict? Does
  the `(dataset, lineage)` cross appear everywhere it must — §5.2, §5.3, §6's scope column, §10.2,
  §13 item 20? Is the conservative direction the right one here, given that it converts some CLOSEs
  into HALTs and a HALT spends the falsifier without discharging it?
- **C-2 → the absence exemption and the frozen family (§5.5, §5.6).** Check the exemption's
  scoping: *absence by declared drop is lawful; absence by computation failure in a surviving
  lineage still HALTs.* Can an implementer confuse the two? Then check the family freeze: v6 keeps
  `m = 92` on every path with untested hypotheses at `p = 1`, justified by a **measured
  invariance**. Re-derive it. Is padding with `p = 1` statistically coherent, or would recomputing
  at 46 be more honest despite being gameable?
- **C-3 → S7 (§5.2.2).** Five parameters newly frozen. The load-bearing one is the **declared
  departure**: C01's `min(‖d_img‖, ‖d_text‖)` cannot exist in head space, so v6 uses the one-block
  `‖l2(h_ow) − l2(h_std)‖`. Is that the right substitute? Does it keep §3.6 true (the raw leg
  non-decisional)? And is the structural difference — the arena *is* the train split, so the small
  set is the bottom decile of the scored population by construction — adequately handled by a note?

### C. The decision rule end to end

- S1–S7 with the corrected reference: is the conjunction coherent, and is §5.9 item 4's
  mean-versus-minimum argument now exactly right?
- S4 (§5.4) and S5 (§5.4.1): both statistics are now written down. Check each against C01's source.
  Is S5's 4-member family the right scope, and is the `n ≤ 12` bound correctly derived?
- §5.9's seven disclosures: is the list complete? Items 6 and 7 disclose direction changes in
  **opposite** directions; is anything else in v6 a direction change that is not disclosed?

### D. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** Round 5 answered *no, for the first
time*, across nineteen gates. v6 has **twenty** — `GATE-RHORAW` is new and global. Re-test all
twenty, and pay attention to whether the per-lineage drop rule creates a new way for an instrument
condition to reach a verdict.

### E. The process rules

- **`rule_1_compute_projection`.** Re-derive §8. Five rounds have each found exactly one uncounted
  loop; round 5's two are now counted (Phase 1c `66 → 67`, new Phase 1e). Hunt again. Is Phase 7's
  *"sub-`0.1 s` class"* still an honest home for what is now inside it?
- **`rule_2_heartbeat`.** Does anything in v6 change an interval?
- §7.9 is new and involved training a head. Check that it computed no battery-arm accuracy, and
  rule whether one exactly-matching cell is sufficient discharge for a global gate demanding 42
  banked quantities at 4 dp.

### F. Honesty

- Does v6 claim any repair the artifact does not contain?
- Is §7.3's blindness discipline intact across **v1–v6**? Round 5 classified 97 decimals in
  `[0.6, 0.99]`; repeat that and classify anything new, including §7.9's `0.8725` and `0.2301`.
- Are the emitter- and weight-dependent quantities stated as ranges with only invariant claims
  load-bearing — including the new trained-vs-untrained residual comparison?

---

## 4. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | The design, executed as written, could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §2, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or its scope statement without inverting it. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**. Report the tally in that form.

**Do not grade on trajectory, in either direction.** Six rounds is neither evidence of convergence
nor of breakage. C09 needed seventeen design rounds and a separate code-review lineage then caught
two wrong-verdict paths that seventeen clean rounds had missed. If the design is clean, say **GO**
plainly; if it is not, say so with the specificity the previous five rounds did.

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
4. **A disposition audit of §14's round-5 block by execution** — all 12 findings, 5 Minors and D-1:
   `VERIFIED ADOPTED` / `NOT ADOPTED` / `PARTIAL`. Pay particular attention to adoptions that could
   have broken each other; that mechanism produced round 4's C-2 and round 5's C-2.
5. An explicit ruling on each of the six open issues in §15.
6. An explicit ruling on §3.D: **is there any gate that can fire on a warranted CLOSE?** — all
   twenty, gate by gate.
7. An explicit ruling on **D-1**: is C01's accuracy-based reference selector pre-registration-safe?
8. Anything to add to §13's twenty-two-item code-lineage handoff.
9. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly
   and state what would be required instead.

---

*Read-only. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit is authorized by this document, and `TARGET_STATE.json` must not be modified.*
