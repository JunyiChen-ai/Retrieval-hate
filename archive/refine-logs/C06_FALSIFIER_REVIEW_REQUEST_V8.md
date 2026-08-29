# C06 `$0` falsifier — fresh independent design review request, **ROUND 8**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM job,
no Modal, no GPU, no cache write, no commit, no edit to `TARGET_STATE.json`. Read-only
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint checkpoints,
plus `sha256sum`, is expected. **Up to four CPU head mints** on the login node (~40 s each) are
permitted if you need them for §7.8; they write nothing outside a scratchpad.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V8.md` (v8, unfrozen).

You are a **fresh reviewer**, independent of rounds 1–7 and of the designer.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M → 3C/3H/6I+5M →
2C/3H/5I+6M → 4C/2H/3I+4M.

---

## 0. The two things that should shape how you read v8

**(a) The science layer has been clean for two rounds; the *record* has not.** Round 6 judged the
verdict logic sound and could manufacture no CLOSE anywhere in the combination space; round 7 agreed
on the gates and the enumeration and re-verified them independently. But round 6 caught v6 claiming
repairs in §13/§14 that were never made, and **round 7 caught v7 doing it again in three more
places** — §5.2.2, §10.2, and a **§5.2.3 that did not exist while five places referenced it**. Round
7 also recorded that v7's header claim of *"all 17 round-6 findings ADOPTED, 0 rebutted"* was false
(its audit: 11 adopted, 3 partial, 1 not adopted, 2 adopted-with-a-new-defect).

**(b) v8 introduces a protocol in response, and its sufficiency is untested.** §14.1 is a
**mechanical disposition verification**: the drafting now writes after every edit, and a scripted
audit diffs v7→v8 section by section, requires **every** §14 row to cite a section the diff shows as
changed, and resolves every internal reference. Its output is embedded verbatim in §14.1 and reports
**13 of 13 rows verified, 0 failing, no unresolved references**. The script is at
`scratchpad/v8_audit.py` (a drafting instrument; it touches no repository file).

**Do not take §14.1 on trust — re-run it.** The whole point of the protocol is that a disposition
table which cannot cite its diff does not ship, and a self-audit that is never independently
re-executed is exactly the kind of claim the last two rounds were about. Diff v7→v8 yourself.

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
5. All seven prior reviews: `C06_FALSIFIER_PREREG_REVIEW{,_R2,…,_R7}.md`.
6. The superseded drafts: `C06_FALSIFIER_PREREG_DRAFT{,_V2,…,_V7}.md`.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V8.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` — **`prepare_views:1296-1304`**
(the endpoint pre-normalisation at the heart of round-7 C-1), `contrast_blocks:1246-1265`,
`l2_rows:1183-1205`, `fuse_modalities:1208-1217`, `paired_key:1220-1239`, `orthogonal_blocks:1272`,
the algebra guard `:1372-1377`, `:1381-1386`, the two `fix_break` sites `:1725` / `:2702-2714`,
`select_strongest_ordinary_control` (guards `:1940-1948`, ranking `:1955-1962`), the consistency
`die()` `:2724`, `displacement_audit:1965-2076` including the **`tiny_ok` limb `:2047-2076`**,
`paired_bootstrap:1742-1772`, `holm_adjust:1775-1784`, `die:392-393`;
`src/model/classifier.py:81-82`, `:140-141`, `:146`; `scripts/analysis/headspace_mint.py:192-194`,
`:199`, `:209-216`, `:321-325`; `scripts/analysis/headspace_arena.py:75-89`;
`scripts/analysis/mechfix_ops.py:94`; `configs/c01/c01_a0_v{2,3,4}.json`;
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`.

---

## 3. Verify these facts yourself

| # | claim | where |
|---|---|---|
| **V1** | All **37** sha256 in §11 match disk — 21 as before **plus the sixteen** newly digested banked artifacts (six arena OUT JSONs, ten `vsw_ckpt` npz). | §11 |
| **V2** | **Re-run §14.1's audit independently.** Diff v7→v8 by section; check every §14 row cites a changed section; resolve every `§N.N`, every `§13 item N`, and the three new §3.7 rows. Report your own tally against the embedded `13 / 0`. | §14.1 |
| **V3** | **Round-7 C-1's measurement.** Build the 13 arms **with** and **without** endpoint pre-normalisation and compare against `prepare_views`: correct `0.000e+00`; un-normalised `1.878e-06` (HateMM) / `1.609e-06` (MHC-ZH) — **both passing a `2e-6` tolerance**. Then confirm §6's `GATE-C01PARITY` row now states exactly one predicate. | §3.4, §6 |
| **V4** | §5.2.3 **exists** and freezes both `tiny_ok` constants (`0.001`, `0.05`); §3.7 carries all three new rows (`<=`, and the two tiny constants); §10.2 names the per-lineage dataset(s). | §5.2.3, §3.7, §10.2 |
| **V5** | **I-1's reduction order.** HateMM `orthrot_83p8` is `0.9568933249 → 0.956893` under float32 accumulation and `0.9568935731 → 0.956894` under float64; §6.1 freezes float64; all 26 agree at 4 dp. | §6.1 |
| **V6** | **I-2's two loops.** `GATE-ALGEBRA`'s residual comparison (120 reductions) and item 25's tail record (60 cells). Round 7 measured `0.160 s` + `0.122 s`; v8 measured `1.0 s` + `0.034 s` and freezes the larger. §8 totals `2928.7` / `3660.9`. | §8 |
| **V7** | The four-cell tail table: `min d_i` `0.018145`–`0.038435`, `frac ≤ 1e-3` `0.0000` everywhere, `θ=45` residual `8.848e-08`–`2.682e-07`, headroom `7.5×`–`22.6×`. (Four mints if you take it.) | §7.8 |
| **V8** | `GATE-FLOOR`'s bit-exact discharge; `GATE-ROWSUBSET`'s `0.000e+00`; `ρ*` `0.968176` / `0.977223` and the trained-head `0/18`. | §7.8, §6.1 |
| **V9** | D-1: the two `fix_break` sites and the executed `net_fixes.reference` = `common` / `endpoint_concat`. | §5.2.1 |
| **V10** | H-1's Holm counterexample table (`m = 92`: 24/24, 23/24, 0/24; `m = 46`: 24/24 throughout). | §5.5 |
| **V11** | Every population-derived constant in §3.7, now including the three added rows. | §3.7 |
| **V12** | §6's table has **twenty** rows, `12 G / 6 L / 2 R`, matching §5.6's two lists; §13 has **26** contiguous items. | §6, §13 |

---

## 4. What you must assess

### A. §14.1 — the new protocol

Re-run it. Then rule: is an **embedded self-audit** an adequate response to two rounds of false
disposition claims, or does the record need a check the drafting process cannot perform on itself?
Is the audit's own logic sound — could a row cite a section that diffed *for an unrelated reason*
and pass? (It can; say whether that matters and what would close it.)

### B. Round-7 C-1's repair, and whether it opened a third hole

v7's pin closed `common_interaction` and opened endpoint pre-normalisation. **Rebuild the arms from
§3.4 as now written**, under every reading you can construct, and report whether anything is still
unstated. Two successive pins have each closed one hole and left another; that is the pattern to
test, not the individual formula.

Also: `GATE-C01PARITY` is now bit-exact with the `2e-6` clause struck. Confirm no other gate
inherited that tolerance by reference, and that bit-exactness is actually attainable (five reviewers
have measured `0.000e+00`).

### C. Where v8's own repairs could have opened seams

v8 changed: header, §3.4, §3.7, §5.2.1, §5.2.2, **new §5.2.3**, §5.9, §6, §6.1, §7.9, §8, §10.2,
§11, §14, **new §14.1**, §15. Look there. Specifically: does §5.2.3's disclosure item 9 collide with
items 6 and 8? Does the new §3.7 row set interact with §13 item 5's "computed from the arena, not
read" instruction (the two tiny constants are C01 config values, **not** arena-derived — is the
table's semantics now mixed)?

### D. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** Answered *no for all twenty* by rounds 6
and 7. Re-test rather than inherit, with attention to `GATE-C01PARITY`'s tightened predicate.

### E. The process rules

- **`rule_1_compute_projection`.** Seven rounds, six uncounted loops found. Hunt again. Rule on the
  `1.0 s` vs `0.160 s` discrepancy for the same loop.
- **`rule_2_heartbeat`.** Does anything in v8 change an interval?

### F. Honesty

- Does v8 claim any repair the artifact does not contain? **Diff v8 against v7 and check every §14
  row** — that method has produced a Critical in each of the last two rounds.
- Blindness across v1–v8: grep every decimal in `[0.6, 0.99]` and classify anything new.

---

## 5. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | Could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §3, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or scope without inverting it. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**.

**Do not grade on trajectory, in either direction.** Eight rounds is neither convergence nor
breakage. If the design and the record are both clean, say **GO** plainly; if they are not, name the
specific defect. C09 needed seventeen design rounds and a separate code-review lineage then caught
two wrong-verdict paths that seventeen clean rounds had missed.

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
4. **Your own disposition audit of §14's round-7 block**, by diffing v7→v8 — all 13 findings:
   `VERIFIED ADOPTED` / `NOT ADOPTED` / `PARTIAL`, with the diff evidence you used.
5. An explicit ruling on each of the six open issues in v8 §15.
6. An explicit ruling on §4.A: **is the §14.1 protocol adequate**, and what would you add?
7. An explicit ruling on §4.D: any gate that can fire on a warranted CLOSE — all twenty.
8. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly.

---

*Read-only. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit is authorized by this document, and `TARGET_STATE.json` must not be modified.*
