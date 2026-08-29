# C06 `$0` falsifier — fresh independent design review request, **ROUND 5**

**Type:** read-only static design review. **No execution of any kind is authorized** — no SLURM
job, no login-node training run, no Modal, no GPU, no cache write, no commit, no edit to
`TARGET_STATE.json`. Read-only numpy/torch-CPU re-derivation on already-banked **train-split**
caches and banked mint checkpoints, plus `sha256sum`, is expected and encouraged.

**Artifact under review:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V5.md` (v5, unfrozen).

You are a **fresh reviewer**, independent of rounds 1–4 and of the designer. You have not seen the
designer's reasoning and must not ask for it.

**Prior rounds.** REVISE 3C/6H/10I+4M → 3C/3H/7I+3M → 2C/1H/6I+4M → 3C/3H/8I+4M. Every round's
Criticals have been found **by executing C01's frozen code**, and every round has found them in
the seam the *previous* round's repair opened:

| round | where its Criticals were |
|---|---|
| 1 | the dispersion gate, the OOD transplant, the unanchored arm algebra |
| 2 | the null contract that round 1's builder repair required |
| 3 | the mask convention and the majority constant that round 2's null contract required |
| 4 | the viability gate, the S6 axis and the fold count that round 3's repairs required |

Round 2 also found **three of twenty-three** round-1 adoptions not actually repaired, and round 4
found one round-3 adoption *textually present but arithmetically false*. **Audit by execution, not
by reading.** Treat §14's cumulative table as an index of things to check, never as evidence.

**Nothing is implemented yet.** No battery script exists — round 4 confirmed all four new files are
absent, as they must be. Several v5 claims are specifications for code that does not exist: judge
whether they are checkable and whether §13's eighteen-item handoff covers them.

---

## 0. What C06 is, so you need no prior context

C06 (*Prompt-Orbit Tangent/Curvature*) claims the tangent and curvature of a video's
representation across a fixed prompt orbit encode policy-bound instability no single prompt
captures. It is **not** an active candidate: its registry status is
`gated_on_zero_cost_falsifier`. An earlier candidate, C01, measured the two-point case in a
**raw-key** arena and found the best of six matched-norm orthogonal rotations of the prompt
endpoints **matched or beat** the real prompt displacement on both datasets. Because the registry
says a raw-key arena *"may kill but may not promote"*, the Gate-0 adjudicator gated C06 rather than
striking it: re-run C01's battery in the **fold-head** arena on already-banked caches. If the
rotations again match, C06 closes for `$0` and an authorized `1.7–2.5 GPU-h` extraction is never
spent; if not, C06 has earned that extraction.

---

## 1. Read first, in this order

1. `CLAUDE.md`, `AGENTS.md`.
2. `TARGET_STATE.json`: `gate0_reopen_2026_07_31.dispositions.gated[0]` (read the C06 entry
   **verbatim**); `iteration_8_queue_state_2026_08_04`;
   `process_rule_compute_projection_and_heartbeat_2026_08_04`;
   `iteration_8_stage0_bounded_extraction_amendment`.
3. `TARGET_FINDINGS.md` — **F118** in full; skim **F88**, **F113**.
4. `refine-logs/GATE0_REOPEN_2026-07-31.md` §4.4;
   `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §3;
   `refine-logs/C09_A0_V17_RECORD.md` §2 and §8.1.
5. All four prior reviews in full: `C06_FALSIFIER_PREREG_REVIEW{,_R2,_R3,_R4}.md`.
6. The superseded drafts for context: `C06_FALSIFIER_PREREG_DRAFT{,_V2,_V3,_V4}.md`.
7. `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V5.md` — **the artifact under review**.

**Primary sources:** `scripts/analysis/c01_policy_contrast_a0.py` (`l2_rows:1183-1205`, its mask
assertion `:1193-1194`, its `None`-normalisation `:1187-1188`; `fuse_modalities:1208-1217`;
`paired_key:1220-1239`; `contrast_blocks:1242-1270`; `orthogonal_blocks:1272`; the guard builds
`:1357-1370`; the algebra guard `:1372-1377`; `prepare_views:1294` and `:1381-1386`; C01's mask
construction `:2224` and call site `:2303-2306`; `paired_bootstrap:1742-1772` and its
`one_sided_raw_p:1769`; `holm_adjust:1775-1784`; `displacement_audit` and its zero-fix guard
`:1989-1996`; `die:392-393`; the deferred-import site `:387` / `import_compute_modules:1048`),
`src/model/classifier.py` (`:81-82` the biased projections, `:115-124` the forward),
`scripts/analysis/headspace_mint.py` (`:106-116`, **`:192-194`**, `:199`, `:203-216`, `:223-227`,
`:321-325`), `scripts/analysis/headspace_arena.py:75-89`, `scripts/analysis/mechfix_ops.py:94`,
`scripts/analysis/headspace_fidelity.py`, `configs/c01/c01_a0_v{2,3,4}.json`.

---

## 2. Verify these facts yourself

Report `VERIFIED` / `MISMATCH` with the value you obtained, for **every** item.

| # | claim | where |
|---|---|---|
| **V1** | All 21 sha256 digests in §11 match disk; both provenance chains sha-gated in source. | §11 |
| **V2** | **C-1's premise.** C01's measured raw `displacement` (`0.8505` / `0.8846`) and `common_displacement` (`0.8598` / `0.8590`) against the arena bars `0.6203` / `0.7091`. Confirm from `C01_A0_OUT.json` or `GATE0_REOPEN_2026-07-31.md` §4.4 that the table in §1 is faithful — it is the evidence that retires `GATE-ARMVIAB`. | §1, §6.2 |
| **V3** | **C-2's counterexample.** `net = (2, 21, 22)` has mean `15.00 ≥ 0.02 × 743 = 14.86` (S3 satisfiable) while `min = 2 < 3` (S6 fails), and the required spread is `20` net items `= 2.69` accuracy points on `n = 743`. So S3 does **not** imply S6. | §5.9 item 4 |
| **V4** | **C-3's recount.** The head key matrix is per fold: `2 ds × 3 seeds × 5 folds × 2 lineages = 60` cells. Re-derive Phase 2b `60`, Phase 2D `2 + 60 = 62`, Phase 2z's `120` guard votes, and the total `2927.5 s` / `3659.4 s`. Confirm the §8 product column sums to the stated total. | §8 |
| **V5** | **H-2's resolution floor.** `1/2001 = 0.00049975 < α/92 = 0.00054348` clears Holm at rank 1; `2/2001 = 0.00099950` first clears at **rank 43**; therefore **42 of 92** comparators must show zero adverse resamples. | §5.5 |
| **V6** | The null-contract defect and repair: `head_f(0,0)` non-zero at multiple seeds; `common_displacement` unbuildable in head space under either mask; all 13 head-space arms build at `n = 743` with the explicit all-False array, dims `4 × 1024-d` + `9 × 2048-d`. | §7.4 |
| **V7** | **The row-subset identity**: raw arms at `n = 743` bit-identical to the `n = 744` one-hot build restricted to 743 rows, `max\|diff\| = 0.000e+00`, all 13 arms, every `ρ` unchanged. | §7.4(h) |
| **V8** | `GATE-C01PARITY`: the two-block builder reproduces `prepare_views` bit-exactly, both datasets, with §3.7's mask forms. | §7.6 |
| **V9** | `ρ*` at `0.968176` / `0.977223`, runner-ups `0.964446` / `0.969686`, all 26 `ρ_raw` at 6 dp; and the trained-head reference (`0/18` above `ρ*` on both). | §6.1 |
| **V10** | **Every population-derived constant in §3.7's table**: arena size, class counts `(297,446)`/`(180,399)`, majority `0.6003`/`0.6891`, bands, the small-displacement quantile population, `GATE-DOMAIN`'s **two** majorities, the tie cap `7`/`5`. | §3.7 |
| **V11** | `GATE-FLOOR`'s six anchors on both metrics; the Holm family `(12+11) × 2 = 46` per `(dataset, lineage)`, `× 2 = 92` per dataset, distinct from §8 Phase 4's coincident `92`. | §5.5, §6 |
| **V12** | `headspace_fidelity.py` opens **no** `dev_seen_*.pt` (reads `mint_*_ffull.npz` at `:66` and trainlogs), so §12's `dev_path_opens` second term is `0`; and `headspace_mint.py:192-194` returns before both `:199` and `:203-216`, which is why §12 binds `mints_executed` and §3.2 discharges `GATE-FOLD` twice. | §3.2, §12 |

---

## 3. What you must assess

### A. The three round-4 Criticals — did they land, and what did the repairs open?

- **C-1 → `GATE-ARMVIAB` retired (§6.2).** v5 went further than round 4's prescription: rather
  than restricting the gate to `endpoint_std`, it **deletes** it, on the argument that a restricted
  version would be strictly redundant with `GATE-ARENA`'s lower bound. **Check that argument** —
  is the two-case form really a subset of `GATE-ARENA`'s firing set? Then check the consequence:
  the real arms now carry **no lower-bound instrument HALT at all** (§15.1). Is that right, and is
  the residual watch (`≤ 0.98` upper bound, `GATE-ORBITDISP`, the algebra gates) sufficient to
  catch a real arm that is broken rather than merely losing?
- **C-2 → S6 binding again (§5.2, §5.9 item 4).** Verify the counterexample and the seed-mean /
  per-seed distinction. Then rule on §15.3: is requiring `3/3` seeds on an integer minimum frozen
  for an arena `7×` smaller the right transfer, or is it now *too* tight in the other direction?
- **C-3 → the fold axis (§8).** Re-derive the 60-cell counts. Then **hunt for the next uncounted
  loop** — four rounds have found one each. Phase 2z is new; is its `2/13` construction share the
  right accounting, and does anything else in §6 or §5 iterate over folds without appearing in §8?

### B. The two structural changes v5 introduces

- **Per-lineage gate scoping (§5.6, §15.2).** Gates are now **global** (HALT the battery) or
  **per-lineage** (drop that lineage), with SURVIVE on any surviving lineage, CLOSE only if **both**
  lineages passed their gates and neither clears, HALT otherwise. Check the classification gate by
  gate: is anything listed global that should be per-lineage, or vice versa? Is `GATE-FLOOR`
  correctly global because it anchors the shared driver? And does the combination rule preserve
  the conservative lean — can a CLOSE ever now be rendered on one lineage?
- **S7 (§5.2).** `GATE-SMALLDISP` has moved out of §6 into the SURVIVE conjunction, with a
  `common_displacement`-only arm scope and a pre-registered zero-fix convention. Check that
  against C01's `decision` block and its `required_halt_only_validity_guards`, and confirm the
  zero-fix convention matches `c01_policy_contrast_a0.py:1989-1996`.

### C. The decision rule and the statistics

- **S4's statistic (§5.4)** is newly pre-registered. Check it against C01's `paired_bootstrap` and
  `holm_adjust`. Is the seed-averaged per-item correctness delta the right statistic, is the
  one-sided p form correct, and is the Holm step-down specified unambiguously enough for a code
  lineage to check?
- **The resolution floor (§5.5, §15.4).** Rule whether **42 of 92 comparators showing zero adverse
  resamples out of 2000** is an acceptable bar to freeze, or whether `B` should rise. Note that
  raising `B` changes §8 Phase 4's cost.
- Does anything in §5 still contain two decision rules at once — the defect round 4 found in v4's
  §5.2 table versus its footnote?

### D. Gates and scope — the recurring question

**Is there any gate that can fire on a warranted CLOSE?** This has produced a Critical in **three
of four rounds** (`GATE-ORBITSCALE` round 1, `GATE-ARENA` round 3, `GATE-ARMVIAB` round 4, plus
`GATE-SMALLDISP` at High in round 4). Test all **eighteen**, not only the previously flagged ones,
and state your answer gate by gate.

Also: does §10.2 now say everything a CLOSE is scoped to, including round-4 I-8's Givens-family
bullet? Do the hard constraints remain untouched?

### E. The process rules

- **`rule_1_compute_projection`.** Re-derive §8 in full. Its motivating incident is a count *"never
  re-multiplied through"*, and its enumeration list names *"draws × folds × seeds × taus × spaces ×
  datasets"* — round 4 caught the **folds** axis missing. Is the enumeration now literally
  exhaustive?
- **`rule_2_heartbeat`.** §9 now specifies per-`(gate, dataset)` and per-Phase-2D-cell lines. Does
  any interval still exceed the stated ~15 s at the corrected counts?
- §7.7 names `U4`'s space and flags which units remain uncorroborated. Is that disclosure
  sufficient, and is the freeze-time exit-status commitment the right instrument?

### F. Honesty and completeness

- Does v5 claim any repair the artifact does not contain?
- Is §7.3's blindness discipline intact — **no arm accuracy anywhere in v1–v5**? Round 4 checked
  this by grepping every decimal in `[0.6, 0.99]`; repeat that check.
- Are the emitter- and weight-dependent quantities (§7.4(a), §7.5) stated as ranges with only the
  invariant claim load-bearing?

---

## 4. Severity definitions and the bar

| severity | meaning |
|---|---|
| **Critical (C)** | The design, if executed as written, could publish a **wrong verdict**, or **cannot execute** on the verdict path. Also: any false factual claim in §2, any test-split exposure, any un-preregistered threshold touching a decision, any un-counted loop in §8, **any claimed repair the artifact does not contain**, or **any gate that can fire on a warranted CLOSE**. |
| **High (H)** | Materially weakens the verdict's authority or its scope statement without inverting it. |
| **Improvement (I)** | Clarity, completeness, reproducibility, or an argument right for a weaker reason than available. |

**GO requires `0C / 0H / 0I`.** Anything else is **REVISE**. Report the tally in that form.

**Do not grade on trajectory, in either direction.** Five rounds is not evidence the design is
converging, and it is not evidence it is broken. C09 needed seventeen design rounds and a separate
code-review lineage then caught two wrong-verdict paths that seventeen clean rounds had missed. If
the design is clean, say **GO** plainly; if it is not, say so with the same specificity the
previous four rounds did.

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
4. **A disposition audit of §14's round-4 block, done by execution** — all 14 findings and 4
   Minors: `VERIFIED ADOPTED` / `NOT ADOPTED` / `PARTIAL`, with what you checked. Pay particular
   attention to adoptions that could have broken each other, which is how round 4's C-2 arose.
5. An explicit ruling on each of the six open issues in §15.
6. An explicit ruling on §3.D: **is there any gate that can fire on a warranted CLOSE?** — gate by
   gate, all eighteen.
7. Anything to add to §13's eighteen-item code-lineage handoff.
8. If you conclude the falsifier cannot discharge the written condition at `$0`, say so directly
   and state what would be required instead.

---

*Read-only. No GPU, SLURM, Modal, model load, head training, arena run, cache write, test-split
access, job submission or commit is authorized by this document, and `TARGET_STATE.json` must not
be modified.*
