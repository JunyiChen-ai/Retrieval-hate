# C09 Stage-0 (A0) v9 — Independent Design Review, Round 9

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V9_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 3 Important` *(the first round with neither a Critical nor a High)*

---

## Prior-round audit

I diffed v8 → v9 line by line and re-checked every R1–R8 finding against **v9 itself**, re-opening every source rather than trusting §0 or §12.

**The diff is exactly what the ledger claims it is, and nothing else moved.** v9 touches only: the title, the STATUS block, §0, one sentence of §1, one bullet added to §10, one paragraph added to §11, and §12. `§§2–9` and `§11`'s pre-existing text are byte-identical to v8. No rule, threshold, feature set, null, gate or arithmetic changed.

**Round 8 (0C/1H/3I) — all four repairs land, and none of them disturbed the body.**
- **H-1 (§0's "retained unchanged" list) — DISCHARGED.** Rewritten as five history bullets under a heading that says *"§§1–11 govern. Nothing in this subsection is a rule."* I checked each bullet against its governing section: `K-DEG`→§6.2 ✓; `τ_hi`→§4.3(b)+§5.2 ✓; the conditional-null thread now ends *"the OLS variant is withdrawn and the per-`(dataset, seed)`-cell unit is superseded"*, matching §5.2 exactly ✓; caps/stratum-degeneracy→§5.3+§5.2 ✓; `STRATUM_OCCUPANCY`→§5.2 ✓. The affirmative re-assertion R8 rated High is gone.
- **I-1 (stale STATUS block) — DISCHARGED.** Reading order runs through R8; `v1–v8` declared superseded; §12 named the v9 ledger; §0's tail no longer carries R4's Important count.
- **I-2 (§1's negative-existence claim) — DISCHARGED.** *"establishes the existence of no operator"* takes the negative-raising reading, and the inline parenthetical removes residual ambiguity.
- **I-3 (encoder-frozen vs encoder-level, F51) — DISCHARGED, and the new text verifies at source.** `candidate_registry[8].claim` reads *"can be corrected at encoder level"*; `REDTEAM_BAN_SCOPE_AUDIT.md:302-304` and `:303-304` are verbatim; GAP-7's heading at `:293` begins *"F51 two-object closure"*; `dead[F51].ban_scope` is quoted exactly. The declared consequence — a CONTINUE licenses an operator narrower than C09's registered claim — is correct and runs against the candidate.

**Round 7 (1H/4I)** — re-verified independently. The union-pool identity holds at both `τ`; because only `struct` is permuted and the evaluation set is fixed by `target` and `c`, the scheme is exact for the marginal null over the pool. The joint exactness form, the `A^{(f)}` fit set with its `c`-restricted out-of-support population, `n_test` `215 / 149 / 161`, and the `THRESH-SYM` disambiguation all land.

**Rounds 1–6** — spot-verified from source. `aggnet_pregate.py:534/:537` literal; `pred_agree = 1 − 2k(1−ov)/n` re-derives; `6/215 = 0.027907`, `7/149 = 0.046980`; 8/13 everywhere; RNG children `2`/`3`/`4`; no bare `round(` anywhere; caps `131.4667 / 95.6667`; no multiple of `1/3` in `[28.95, 29.0)`.

**One R5 finding did not survive re-checking, and it is the source of I-2 below.** R5's I-5(b) asserted the source's "four" was not re-derivable and "it is at least six." That assertion is wrong at source, and v6–v9 have carried it forward.

---

## Verified as sound (do not re-litigate)

**Registry and ban quotations, re-opened at source.** `unified_pilot_gate.{arena, stage_0_reachability}`; `hard_constraints[4]`; `banned_constraints[2]/[9]/[10]`; `candidate_registry[8].{claim, dedup_boundary}`; the reopen's promoted fields, `what_this_reopen_does_not_do[2]`, all three `c09_next_step` clauses, `strategic_finding`, and kill-risk (i) with all four F114 qualifiers — all verbatim. `ban_scope` for F47, F51, F66, F75, F96, F97, F98(b)+(d), F99, F112 verbatim. **`dead[F113]` genuinely carries only `name` and `status` — no `ban_scope`** — so §2's statement that the binding surface is the registry clause is exactly right. All the `.md` citations verify, including `AGGNET_PREGATE_RECORD.md:690`, which does phrase the epitaph differently as §1 says.

**Every number re-derived.** All twelve floors; 30 `fold_acc_deployed` cells; fold held-out counts recomputed from `StratifiedKFold(5, shuffle=True, random_state=0)` on the real label vectors; `raw_deployed_acc`; `B_fid`; bands; per-seed errors → `84.3333/62.3333`; caps ⇒ `k ≥ 132/96` dead; `k = R(2|P_0|) = 152/110` dead, `R(1.5|P_0|) = 114/83` live; `0.896/0.663` and `0.898/0.675`; `π*` gap `0.093125` at `k=80` and 7.5 at `k=100`; Hanley–McNeil; `0.1054–0.1142`; `0.050 × 743 = 37.15`. Both "re-measured this session" claims reproduce exactly on my own reads, as does `GATE-NULL`.

**Executability is real and I found nothing that would turn this into an engineering HALT.** `sha256(headspace_mint.py) = cefdf8dc…` matches the banked `meta.mint_script_sha256` in all six arena outputs and `C02_A0_OUT.json::frozen_modules`; `headspace_mint.py` is genuinely absent from `frozen_artifact_policy`. Every cited line range is exact. `lab_dev` is written on the unconditional `np.savez` path, hence in all 36 mints. `load_split` is called only with `"train"` and `"dev_seen"`; for `fold ≥ 0` the dev/test splits are a stratified slice of the *fitting pool* (`:221-226`), so **no test path is reachable by any code path**. The live environment is byte-identical to the banked `meta.runtime`. `default_rng(20260801).spawn(6)` works — I ran it. `int(np.floor(82.5+0.5)) = 83` against `round(82.5) = 82`. `GATE-FIXK20` is corroborated: banked `FIXK_20` has `changed = 0`, `dacc = 0.0`, `B_agree_fixk["20"] = agree_deployed = 1.0`; and `B_agree_fixk["15"] = 1.0` with `changed = 0` on ZH seed 0 confirms §6.2's `DEGENERATE_ALL_EMPTY` anchor. `sacct`/`squeue` confirm the job precedent and the empty queue; C04's tranche has terminated.

**The budget is sound and very conservative.** The 36 banked C02 mint wall times: min `33.2 s`, max `60.0 s`, median `41.85`, total `24.51 min` — §2's figures are exact. Every table line reproduces and the lines sum to **116** against ≈115. I timed one `lbfgs` fit-and-score at the design's scale under DET-1 threads: **4.6 ms** against the 35 ms assumed.

**Statistical soundness.** `AUC_strat` fully pinned; `ΔAUC` genuinely paired. `PERM-STRUCT` exact for the marginal null over `P^{(τ)}`; only `struct` is permuted, so the evaluation set is invariant across draws — the property exactness needs. `PERM-STRUCT-COND` exact for the joint form claimed; small item-strata can only push `p_cond` toward 1. Holm within dataset and family with both families required is conservative; the IUT takes no correction; the `1/1001` floor leaves `α/2` reachable. `p_w ≥ 30` neither inert nor fatal. `K-DEG`'s vacuous regime unreachable on the CONTINUE path; the gate can both fire and clear. `net_s ≡ n·Δacc_s`; `Δacc_{O1} = |P_τ|/n` identically per seed; `ΔmF1_{O1}` monotonicity genuine. The four bands partition completed runs, and **no single rule can carry a CONTINUE**.

**On the residual leak path: none beyond the one §4.3 names.** Verified feature by feature, threshold by threshold, stratum by stratum.

**Legality holds in both directions.** F47, F96, F97, F98(b)+(d), `[9]`, `[2]`/`[4]`, F66, F99 and now F51 all conceded rather than narrowed — several read wider than their own text *against* C09.

**Scope honesty.** §10 claims nothing the design does not support, and a KILL is genuinely available.

---

## IMPORTANT

**I-1 — v9's own change summary understates v9's own changes, in both places that state them, and §12 contradicts its own table.** §0 reads *"v9 changes nothing in §§1–11's substance"* and is contradicted two lines later by its own items 3 and 4, which change §1's summary sentence, add a bullet to §10 and add a paragraph to §11. §12 reads *"All four findings are in §0, the STATUS block, one sentence of §1 and one paragraph of §11. **Nothing in §§2–10's substance changed in v9.**"* — but **§10 gained a five-line scope bullet**, and §12's own I-3 row names its `where` as *"§10, §11"*, and §0's item 4 says *"a bullet to §10"*. This is the identical failure class R8 rated High, differing only in direction, which is why it is Important and not High. **Repair:** state v9's true surface (§0, STATUS, §1, §10, §11) in both places, with the invariant spelled out — no rule, threshold, feature set, null, gate or arithmetic changed; §§2–9 byte-identical to v8.

**I-2 — §6.1's "at least six distinct operator families" is contradicted by the cited source's own heading, and the accompanying claim that "four" is not re-derivable is false.** The enumeration is verbatim ✓ — but `HEADSPACE_TRANSFER_PREGATE.md:623` heads the ladder *"### 4.10 The transfer ladder (§2.10) — **four operator families**, one arena swap"*, and that is the only family count anywhere in the file. Its rung column places the nine positives in four groupings: rung 1 = VSW (`VSW_pow/exp/lin`, plus `ORACLE_lambda_pow` *"(hindsight ceiling)"* and `CTRL_cos_pow` *"(DEG-D, no verifier)"* — VSW's own oracle and control arms), rung 2 = F95, rung 3 = FIXK, and un-runged `THRESH_best` *"(global recalibration)"*. Reaching six requires promoting VSW's hindsight ceiling and its degeneracy control to independent families, a regrouping the source declines. R5's I-5(b) introduced the error and v6–v9 have carried it. Nothing gating turns on it and its direction is conservative, but it is a stated count the cited source contradicts, in the section whose only job is to register the prior faithfully. **Repair:** restore the source's taxonomy, and delete the "not re-derivable" clause.

**I-3 — §6.1's decimal-place characterisation of `HEADSPACE_TRANSFER_PREGATE.md:871-873` is wrong.** The two attributions are right — the sentence does live in `dead[F113].status` and `:871-873` does omit the `30/30` fold count (it appears at `:847`/`:901`) — but `:872-873` reads `(d_AUC −0.064 / −0.129), inverting the raw arena's +0.157 / +0.230`, which is **3 decimal places**. **Repair:** "at 3 dp".

---

## Bottom line

The science is finished and I could not move it. R8's four repairs all land and none disturbed the body: the diff confirms §§2–9 are byte-identical to v8, the new §11 paragraph and §10 bullet verify word-for-word against source, and the §1 correction reads as intended. The instrument is on disk, sha-pinned to its banked anchor, and reproducible in a live environment byte-identical to the banked runtime; the budget is ~7× conservative against a measured 4.6 ms fit; the legality spine concedes every adjacent ban including one (F51) that now runs against C09's own registered claim; the nulls are exact for the nulls they name and conservative where they are not; the four decision rules are jointly decidable, individually non-vacuous, and none can carry a CONTINUE alone; and the only route by which a scored item's gold label reaches its own model remains the arena's own bank channel, which §4.3 names, prices and shows cancels.

What I am not able to pass is three statements about the document's and the record's own contents. Two are provenance defects inherited from R5 and R6 that four rounds of review have re-attested without re-opening the source line that contradicts them. The third is the same defect class R8 rated High, one revision later and pointing the other way. These are edits to three sentences. They need no re-measurement, no GPU and no re-derivation of anything in §§2–11. Fix them and freeze.
