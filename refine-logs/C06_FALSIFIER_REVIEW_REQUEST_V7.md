# C06 `$0` falsifier — fresh independent design review request, **ROUND 7**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM job,
no Modal, no GPU, no cache write, no commit, no edit to `TARGET_STATE.json`. Read-only
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint checkpoints,
plus `sha256sum`, is expected. **Up to four CPU head mints** on the login node (~40 s each) are
permitted if you need them to check §7.8; they write nothing outside a scratchpad.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V7.md` (v7, unfrozen).

You are a **fresh reviewer**, independent of rounds 1–6 and of the designer. You have not seen the
designer's reasoning and must not ask for it.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M → 3C/3H/6I+5M →
2C/3H/5I+6M.

**Round 6's headline matters for how you should read v7.** It judged **the science layer clean for
the first time**: it enumerated the whole verdict-combination space and *"could not manufacture a
CLOSE anywhere on the new dataset-axis rule, the twenty gates hold under the warranted-CLOSE test,
and the verdict combination is total and mutually exclusive."* Its two Criticals were that **§13 and
§14 had never been edited** — byte-identical to v5 — so a round-5 finding was never adopted at all
and §14 asserted two claims the body refutes. Round 6's own summary: *"the remaining failure surface
lies almost entirely in the record rather than the science."*

**So this round's centre of gravity is the record, and one thing in particular.** §13 is the **sole
input to the mandatory separate code/resource review lineage** — the lineage the campaign's record
says caught two wrong-verdict paths on C09 after seventeen clean design rounds. v7 rebuilds it from
scratch at **26 items**. §14 is rebuilt to cover all six rounds. **Audit both by counting and
resolving, not by reading the claim that they were rebuilt.**

**The one substantive trap, and it has caught two of six reviewers.** §3.4's prose does not, by
itself, determine the arm→formula map. Rounds 5 and 6 both rebuilt the battery from it and both
derived **`common_interaction`** wrongly; round 6 measured the cost at `max|diff| = 9.697e-01`
against `prepare_views`. v7 pins the formula explicitly. **Rebuild the arms from §3.4 as now written
and report whether the pin works** — that is the only evidence that settles it.

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
3. `TARGET_FINDINGS.md` — **F118**; skim **F88**, **F113**.
4. `refine-logs/GATE0_REOPEN_2026-07-31.md` §4.4;
   `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §3;
   `refine-logs/C09_A0_V17_RECORD.md` §2, §8.1.
5. All six prior reviews: `C06_FALSIFIER_PREREG_REVIEW{,_R2,_R3,_R4,_R5,_R6}.md`.
6. The superseded drafts for context: `C06_FALSIFIER_PREREG_DRAFT{,_V2,_V3,_V4,_V5,_V6}.md`.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V7.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` — `contrast_blocks:1246-1265`
(the `common_interaction` definition), `l2_rows:1183-1205`, `fuse_modalities:1208-1217`,
`paired_key:1220-1239`, `orthogonal_blocks:1272`, the guard builds `:1357-1370`, the algebra guard
`:1372-1377`, `prepare_views:1294`/`:1381-1386`, the two `fix_break` sites `:1725` and `:2702-2714`,
`select_strongest_ordinary_control:1940-1948`, the consistency `die()` `:2724`,
`displacement_audit:1965-2060` with its zero-fix guard `:1989-1996` and `tiny_ok` limb `:2068-2076`,
`paired_bootstrap:1742-1772`, `one_sided_raw_p:1769`, `holm_adjust:1775-1784`,
`id_hash_permutation`, `die:392-393`; `src/model/classifier.py:81-82`, `:140-141`, `:146`;
`scripts/analysis/headspace_mint.py:192-194`, `:199`, `:203-216`, `:321-325`;
`scripts/analysis/headspace_arena.py:75-89`; `scripts/analysis/mechfix_ops.py:94`;
`configs/c01/c01_a0_v{2,3,4}.json`;
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`.

---

## 2. Verify these facts yourself

| # | claim | where |
|---|---|---|
| **V1** | All 21 sha256 digests in §11 match disk; both provenance chains sha-gated in source. | §11 |
| **V2** | **§13 has exactly 26 items**, numbered contiguously, and **every** `§13 item N` reference in the body resolves to the right item. | §13 |
| **V3** | **§14's two stale assertions are gone**: no sentence asserts the "42 of 92" floor, and "S6's net-fix reference" no longer appears among rulings carried unchanged. (Both may appear in corrective or historical framing — check the framing, not the string.) | §14 |
| **V4** | **The `common_interaction` pin.** Rebuild the 13 arms from §3.4 as written and compare against `prepare_views`. Report `max\|diff\|` and whether you needed to consult the source. | §3.4 |
| **V5** | **The four-cell tail.** `min d_i` `0.018145`–`0.038435`; fraction `d_i ≤ 1e-3` = `0.0000` in all four cells; `θ = 45` residual `8.848e-08`–`2.682e-07`; headroom `7.5×`–`22.6×`, worst cell HateMM · Head-R. (Four mints if you take it.) | §7.8 |
| **V6** | **`GATE-FLOOR`'s discharge is bit-exact**: a re-minted HateMM `s0/fold0` `K_train` matches the banked mint at `max\|diff\| = 0.000e+00`, and the native deployed vote is `0.8725`. | §7.8 |
| **V7** | **H-1's counterexample table**: executing `holm_adjust`, `m = 92` gives 24/24, 23/24 and 0/24 for the three witness configurations, while `m = 46` gives 24/24 in all three. | §5.5 |
| **V8** | **D-1**: C01's two `fix_break` sites and the executed `net_fixes.reference` = `common` / `endpoint_concat`; plus the frozen `transforms.small_displacement_gate_reference` constant. | §5.2.1 |
| **V9** | **S7's six parameters**: `0.5`, `0.1`, the reference rule, the head-space one-block statistic, `<=`, per-seed `3/3`; and `tiny_ok`'s two constants `0.001` / `0.05`. | §5.2.2, §5.2.3 |
| **V10** | `ρ*` `0.968176` / `0.977223`; all 26 `ρ_raw` at 6 dp — **including `orthrot_83p8` = `0.956894`**, which v6 printed as `0.956893`. | §6.1 |
| **V11** | Every population-derived constant in §3.7's table, including S7's `<=` operator row. | §3.7 |
| **V12** | §8's column sums to `2927.6`; `× 1.25 = 3659.5`; §6's table has **twenty** rows, `12 G / 6 L / 2 R`, matching §5.6's two lists. | §6, §8 |

---

## 3. What you must assess

### A. The two rebuilt sections — by counting, not by reading

- **§13.** Count the items. Does the list cover every limb the six rounds prescribed? Round 6's
  PART E named items 19–26 plus extensions to 5, 10, 15, 16 — check each landed. Is any item
  unactionable as written by someone with no context? Is anything a code lineage must check
  **still** missing?
- **§14.** Is the cumulative record now honest? Check specifically: the round-5 block's PARTIAL
  entries are marked COMPLETE with a pointer; the 46-family is recorded as a **rebuttal** rather
  than an adoption; round-4's H-2 row and measurement ζ are marked superseded; the two struck
  rulings are struck with reasons.

### B. The `common_interaction` pin (§3.4)

The single most consequential item, because the head-space arms have **no anchor other than
`GATE-C01PARITY`**. Rebuild from the prose and report honestly whether you got it right first time.

### C. Where v7's own repairs could have opened seams

Every round has found its Criticals in the previous round's repair. v7 changed: §3.4 (the pin),
§5.2.2/§5.2.3 (S7's operator, dispersion, `tiny_ok`), §5.5 (the H-1 warrant), §5.6 (I-3), §5.9
(items 4 and 8), §6.4/§6.5, §7.3 (the blindness warrant), §7.8 (the four-cell table), §10.2, and
§13/§14 wholesale. Look there.

Specifically: does §5.9 item 8's new disclosure contradict any other item? Does §7.3's two-part
warrant leave any path by which an ro-derived arm could be voted outside the arena phase? Does
§5.2.3's `tiny_ok` non-carriage interact with S7's binding status in a way §5.9 does not disclose?

### D. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** Round 6 answered *no for nineteen, with
`GATE-ALGEBRA` carrying an under-evidenced CLOSE→HALT probability*. v7 answers that with a four-cell
range. Re-test all twenty and rule whether `GATE-ALGEBRA`'s worst measured cell (`7.5×`) is now
adequately evidenced.

### E. The process rules

- **`rule_1_compute_projection`.** Six rounds have found five uncounted loops (Phases 1b, 2b, 2D,
  2z, 1c/1e). Round 6 found none. Hunt again. Does §7.8's four-cell measurement imply any run-time
  loop §8 does not count — in particular §13 item 25's per-cell tail recording?
- **`rule_2_heartbeat`.** Does anything in v7 change an interval?

### F. Honesty

- Does v7 claim any repair the artifact does not contain? Round 6 found two such claims by diffing
  sections against v5 — **diff v7's sections against v6** and do the same.
- Is blindness intact across **v1–v7**? Grep every decimal in `[0.6, 0.99]` and classify anything
  new, including §7.8's four-cell table.

---

## 4. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | The design, executed as written, could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §2, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or its scope statement without inverting it. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**. Report the tally in that form.

**Do not grade on trajectory, in either direction.** Seven rounds is neither evidence of convergence
nor of breakage, and round 6's "science layer is clean" is **not** a reason to wave the record
through — §13 is what a downstream lineage will actually execute against. Equally, if the design is
clean, say **GO** plainly rather than manufacture a finding; round 6 explicitly warned against both
failure modes. C09 needed seventeen design rounds and a separate code-review lineage then caught two
wrong-verdict paths that seventeen clean rounds had missed.

---

## 5. What a GO does and does not authorize

A GO authorizes **nothing to run**. Before any job: (1) freeze with hashes; (2) a **separate**
independent code/resource review lineage over the executable reaching its own `0C/0H/0I`;
(3) main-dialogue authorization. A GO is not authority to write `TARGET_STATE.json`.

---

## 6. Deliverable

1. The `0C/0H/0I`-form tally and **GO** or **REVISE**.
2. Every finding with severity, `file:line` citation, and the concrete repair.
3. Your verification result for **all twelve** §2 items.
4. **A disposition audit of §14's round-6 block by execution** — all 11 findings and 6 Minors:
   `VERIFIED ADOPTED` / `NOT ADOPTED` / `PARTIAL`. Round 6 found v6's §13/§14 unedited by **diffing
   against v5**; diff v7 against v6 the same way.
5. An explicit ruling on each of the six open issues in §15.
6. An explicit ruling on §3.D: **is there any gate that can fire on a warranted CLOSE?** — all
   twenty.
7. An explicit ruling on **§13's completeness**: is a 26-item handoff sufficient for the separate
   code/resource review lineage to check this battery, and what would you add?
8. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly and
   state what would be required instead.

---

*Read-only. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit is authorized by this document, and `TARGET_STATE.json` must not be modified.*
