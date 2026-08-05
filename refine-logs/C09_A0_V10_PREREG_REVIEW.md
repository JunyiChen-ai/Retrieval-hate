# C09 Stage-0 (A0) v10 — Independent Design Review, Round 10

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V10_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 2 Important`

---

## Prior-round audit

I diffed v9 → v10 line by line and re-verified R1–R9 against **v10 itself**, re-opening every source rather than trusting §0 or §12.

**The diff confirms the scope claim, with one exception.** v10 changes exactly: line 1 (title), line 17 (STATUS), lines 28–37, lines 55–91 (§0), lines 1161–1170 and 1180–1182 (two loci in §6.1), and §12. `§1`, `§§2–5`, `§7`, `§8`, `§9`, `§10`, `§11` and the rest of `§6` are byte-identical to v9. §12's scope statement is exactly right. §0's is not — see I-1.

**Round 9 (0C/0H/3I) — two of three repairs land cleanly, one re-commits its own defect.**
- **I-2 — DISCHARGED, re-derived at source.** `HEADSPACE_TRANSFER_PREGATE.md:623` is character-exact: *"### 4.10 The transfer ladder (§2.10) — four operator families, one arena swap"*. The rung column at `:630–643` places the nine positives of `:859–864` into exactly four groupings. The "at least six" and the false "not re-derivable" clause are gone, and the "not one lineage" point is retained.
- **I-3 — DISCHARGED.** `:872-873` reads `(d_AUC −0.064 / −0.129), inverting the raw arena's +0.157 / +0.230` — 3 dp confirmed; the `30/30` fold count is at `:847` and `:901`. `dead[F113]` genuinely carries only `name` and `status`.
- **I-1 — NOT DISCHARGED IN §0.** §12's half is repaired correctly; §0's half re-commits the identical defect one revision later.

**Rounds 1–8 — every Critical and every High lands in v10.** Checked against the body, not the ledger: R1 C-1…C-4 and H-1…H-8; R2 C-1 and H-1…H-8; R3 C-1 and H-1…H-8; R4 C-1 and H-1…H-4; R5 H-1/H-2/H-3; R6 H-1/H-2; R7 H-1; R8 H-1.

---

## What I verified as sound

**Quotation fidelity: 36/36 exact.** Every registry field, every `ban_scope` (F47, F51, F66, F75, F96, F97, F98(b)+(d), F99, F112), every `.md` citation. No ellipsis hides material that changes a meaning — several hide clauses that read *more* adversely to C09. The disowned upper-case rendering ("may not PROMOTE") appears in no source outside C09's own earlier drafts, so that parenthetical is accurate.

**Every number re-derived from source.** All twelve floors; 30 `fold_acc_deployed` arrays; fold counts recomputed independently from `StratifiedKFold(5, shuffle=True, random_state=0)`; `raw_deployed_acc`; `B_fid`; per-seed errors exactly `83/85/85` and `62/64/61` from integer correct-counts; caps ⇒ `k ≥ 132/96` dead; `R(1.5×55) = 83` against banker's `82`; `0.896/0.663` and `0.898/0.675`; `π*` gap `9.3125` pt; Hanley–McNeil; `28.95`, `37.15`, the `[28.95, 29.0)` emptiness; `6/215`, `7/149`; `pred_agree` mapping; majority rates and both bands; `n_test` `215/149/161`. Both "re-measured this session" claims reproduce on my own reads, as does `GATE-NULL`.

**Executability is real; nothing would turn this into an engineering HALT.** `sha256(headspace_mint.py)` matches its banked anchor in all six arenas and `C02_A0_OUT.json::frozen_modules`; the real `frozen_artifact_policy` list (`progress.json:30`) genuinely does not contain it. Every cited line range is exact. `lab_dev` is on the unconditional `savez` path. `load_split` is called only with `"train"` and `"dev_seen"`; for `fold ≥ 0` the dev/test splits are a stratified slice of the *fitting pool*, so **no test path is reachable by any code path**. **The mint trains 30 fixed epochs with `torch.save` monkeypatched to a no-op and `best_epoch_path` never re-loaded — no dev-based model selection anywhere**, which is what makes the whole label-use spine hold. The live env is byte-identical to the banked `meta.runtime`. `spawn(6)` works; `int(np.floor(82.5+0.5)) = 83`. `GATE-FIXK20` is corroborated in all six banked arenas, and ZH seed 0's `B_agree_fixk["15"] = 1.0000` is exactly the `DEGENERATE_ALL_EMPTY` anchor §6.2 cites. `squeue` empty; C04's tranche terminated.

**The budget is sound and conservative.** Job `13847` ran 8 CPU / 0 GPU / 32 G in `00:29:49`. The 36 banked mint times are min `33.2`, max `60.0`, median `41.85`, total `24.51 min` — exact. Lines sum to 116 against "≈115", ~7× conservative.

**Statistical soundness and decidability.** `AUC_strat` fully pinned; `ΔAUC` paired. `PERM-STRUCT` exact for the marginal null over `P^{(τ)}`: only `struct` is permuted and `A^{(f)}` is measurable w.r.t. `(target, BASE)`, so the evaluation sets are invariant across draws — precisely the property exactness needs. `PERM-STRUCT-COND` exact for the joint form claimed. Requiring both families is an intersection, hence conservative; Holm within dataset is correct and the `1/1001` floor leaves `α/2` reachable; the dataset conjunction is a genuine IUT; the existential over `τ` is covered because a false CONTINUE requires a false rejection inside one dataset's Holm family. I re-derived the macro-F1 monotonicity — each FN→TP flip raises class-1 F1 iff `2(2TP+FP+FN) > 2TP`, always true, and raises class-0 F1 by shrinking its denominator — so `K-REACH` at `τ_0` genuinely closes every `τ ≥ 0`. `K-DEG` can both fire and clear at every live `k`. **No single rule can carry a CONTINUE.** A KILL is fully available and is the pre-declared expectation.

**On the residual leak path: none beyond the one §4.3 names.** I walked all 13 features. The right analogue does read `lab_query` and is excluded by `GATE-BLIND`'s guard on exactly that array. That last point is stronger than the document claims: because both permutations hold `target`, `BASE` and `A^{(f)}` fixed and permute only `struct`, any target-side contamination is present identically in the observed and permuted draws and is therefore inside the null's own calibration.

**Legality holds, in both directions.** F47 conceded outright; the F114 correction used only where it applies; F98(b) conceded as constructed-and-measured; `[9]` and `[2]`/`[4]` named; F96 treated on the statistic it was calibrated on; F51 now runs against C09's own registered claim.

**Scope honesty.** §10 claims nothing the design does not support and several things that cut against the candidate.

---

## IMPORTANT

**I-1 — §0's own scope sentence is false, is self-contradicted by the clause that follows it, and re-commits the exact defect R9 I-1 named — in the repair that closes R9 I-1.** Line 77 reads *"**Nothing in §§2–9 changed in v10, and nothing in §§1–11 changed in substance.** The edits are three sentences in §0/§6.1, the STATUS block, and §12."* §6.1 is inside §§2–9 (§6 begins at line 1119), and two of its sentences changed — as the very next clause says, as §12's scope statement says, and as the diff shows. R9 I-1's finding was "v9's change summary understates v9's own changes, in both places that state them"; the repair fixed §12 and re-committed the understatement in §0. §12's version is correct and is the one to mirror. **Repair:** "Nothing in §§1–5 or §§7–11 changed in v10; the edits are two sentences of §6.1, the STATUS block, §0 and §12, and no decision rule, threshold, feature set, null, gate or arithmetic has changed since v8."

**I-2 — §7 names `CONFIG-MATCHED-CORRECT` as a live control and rests the registry claim's break-exposure conjunct on it, but v10 nowhere defines the term and §5.3 calls the phrasing that contains it superseded.** The term appears exactly three times in v10: twice in §7, and once at line 980, where §5.3 calls `P_τ ∪ CONFIG-MATCHED-CORRECT` *"the superseded … phrasing"*. §5.2 and §6.3 — the sections §7 points at — do not contain it. The object also changed under the name: in v5 it was the *unrestricted* "query items correct at all three seeds"; after R5's H-3 and R6's H-1 repairs it is confidence-restricted **and** fold-dependent. The phrase "break-exposure stratification" likewise has no referent anywhere else in the document. This changes no rule, threshold or verdict, and break exposure genuinely *is* instrumented (§5.3's per-seed `net_s`, `GATE-SELFTEST`, and the reported class composition of `S`) — but the one conjunct of C09's registered claim that §7 asserts is "measured rather than asserted" is asserted, in a summary section, by reference to a name the frozen specification retired. This is the residue of the R7 I-2 / R6 H-1 repairs, which were scoped to §5.3 and §5.2 and never followed into §7; no round has audited §7. **Repair:** either delete the term and name the object §5.2 actually specifies, or re-introduce it as an explicit alias at its §5.2 definition; and attribute break exposure to what measures it.

---

## Checked and deliberately not counted

- **§6.1's "four operator families".** The source's four at §2.10 are VSW / F95 / fixed-k / **F89**, with `THRESH_best` un-runged; the nine positives occupy VSW / F95 / FIXK / un-runged `THRESH_best`. The two fours differ by one member. But v10's operative enumeration is the second one, stated exactly and labelled "un-runged", the `:623` quote is verbatim, and "the count is the source's four" is true as a count.
- **§2's ksweep parenthetical.** `curves[…]["dev"]` exists on every curve but is `null` for all four `MHC_EN_ARM-V/*/valsel` curves. The concession errs *against* the design, on a parenthetical to a secondary reason for skipping a clause CAL-2 itself declares non-gating.
- **§2's budget.** Lines sum to 116 (or 114 reading "< 1 min" as 0); "≈115" is the midpoint and the table is ~7× conservative.
- **§5.3's caps.** `2 × 84.3333 − 37.2` is literally `131.4666`; the printed `131.4667` is the exact-mean (`253/3`) value. Inert at integer `k`.
- **§5.3's "7.5 points at `k = 100`"** — true value `7.45`.
- **§5.3's "the six `(τ, k)` cells"** — the caps are per-dataset, so there are really 12 markings.
- **The R9 I-2 repair leaves a sentence fragment.** Prose, not science.
- **F66's own `ban_scope` closes *"Re-open requires an operator that converts symmetric structure — the beta decomposition proves none of the tried classes can"*,** a live headwind against §11's symmetric successor that the document never quotes. §11's precondition (d) already requires a fresh F66 adjudication, so this A0 does not turn on it — but the F99 leg is spelled out and the F66 leg is not.

---

## Bottom line

The science is finished and I could not move it, and neither of my two findings touches it. I re-derived every number from the named source on disk, re-opened all 36 quotations, walked all 13 features and all five thresholds for a residual gold-label path, re-checked the exactness argument for both permutation nulls, confirmed the four decision rules are jointly decidable and individually non-vacuous with no single rule able to carry a CONTINUE, confirmed the mint does no dev-based selection and that no test path is reachable, matched the mint's sha256 to its banked anchor, and confirmed the live environment is byte-identical to the banked runtime. The legality spine concedes every adjacent ban, including two — F51 and F99 — that run against C09's own registered claim.

What I cannot pass is two statements about the document's own contents. **I-1 is the one that matters**, because it is R9 I-1 re-committed inside the repair that closed R9 I-1. **I-2** is a stale object name in the controls summary, on the one conjunct of the registry claim whose instrumentation §7 asserts by reference to it.

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not yet — but only barely not.** Both repairs are edits to three sentences in §0 and §7. They need no re-measurement, no GPU, no re-derivation of anything in §§1–11, and they do not touch a single rule, threshold, feature set, null, gate or arithmetic. Fix them and the frozen set is ready: the instrument is on disk and sha-pinned, the queue is empty, C04's tranche has terminated, and the only remaining precondition outside the document is explicit main-dialogue authorization.
