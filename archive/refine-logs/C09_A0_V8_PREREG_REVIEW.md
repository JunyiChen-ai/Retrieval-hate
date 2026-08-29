# C09 Stage-0 (A0) v8 — Independent Design Review, Round 8

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V8_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 1 High / 3 Important` *(all four in the front matter; the science passed)*

---

## Prior-round audit

I re-checked every R1–R7 finding against **v8 itself**, not against §12, and re-opened every source. I also diffed v6 → v7 → v8 to see what actually moved.

**Rounds 1–5 — discharged, and they hold in v8.** R1 C-1; C-2 (`π*` re-derived as the exact inversion of `net = k(2π−1)`); C-3 (`headspace_mint.py:204-205` literal, fold map seed-invariant; bootstrap and both permutations item-level); C-4 (`net_s ≡ n·Δacc_s`). R2 C-1, H-3, H-5, H-6, H-7, H-8. R3 C-1, H-1…H-8, I-1…I-11. R4 C-1 (`pred_agree = 1 − |S△twin|/n = 1 − 2k(1−ov)/n`; `aggnet_pregate.py:534/:537`), H-1…H-4, I-1…I-9. R5 H-1/H-2/H-3 and I-1…I-10.

**Round 6 (0C/2H/10I) — all twelve land.** `A^{(f)}` per scoring fold; F99 in §10 and precondition (d) with `6/215 = 0.02791`, `7/149 = 0.04698` and `n_test = 149` independently confirmed; 8/13 everywhere; children `2`/`3` in §6.3; no `round(` anywhere; `56.0/62.33 = 0.898`; the `~24 min` `GATE-NULL` line; all six I-7 provenance items; the authorisation line and all three `c09_next_step` clauses; `[2]`/`[4]`; caps `131.4667 / 95.6667`; the `[28.95, 29.0)` proof.

**Round 7 (0C/1H/4I) — all five land in the body.**
- **H-1 — DISCHARGED and correctly.** §5.2 names `P^{(τ)} ≡ ∪_f A^{(f)}`, one `π` per `(dataset, τ, draw)`, `A^{(f)}`'s rows only inside the fold-`f` refit, donors from `P^{(τ)}`. I verified the set identity at both `τ`. The scheme is exact for the marginal null over the pool and the union pool cannot narrow it.
- **I-1 … I-4 — DISCHARGED.** The joint exactness form; the `A^{(f)}` fit set and the `c`-restricted out-of-support population; `n_test` `215 / 149 / 161` (and `curves` really is keyed by the flat string with `test` and `dev` sub-keys, so §2's notation is literal); `THRESH-SYM` per seed with `pred_agree` averaged.

**What no prior round audited:** the STATUS block and §0's "retained repairs" list. Both are **verbatim v6 text**, carried unchanged through v7 into v8. That is where the one High and the first Important sit.

---

## Verified as sound (do not re-litigate)

**Every quotation I opened is exact at its cited location.** Verbatim-verified this round against source: `unified_pilot_gate.{arena, stage_0_reachability}`; `hard_constraints[4]`; `banned_constraints[2]/[9]/[10]`; `candidate_registry[8].{claim, dedup_boundary}`; the reopen's promoted fields, `what_this_reopen_does_not_do[2]`, `c09_next_step`, `strategic_finding`, kill-risk (i) with all four F114 qualifiers, and F88's CPU-floor caveat; `ban_scope` for F47, F75, F96, F97, F98(b)+(d), F99; `RESTRANS_PREGATE_RECORD.md:409`; F98's DEG-A and `DEG_KILL = 0.95`; `dead[F113].status` and its absent `ban_scope`; `queued.headspace_arena_EN`; `progress.json:25`; `LITSWEEP3_DATA_CENTRIC.md:82/:91/:95`; `LITSWEEP5_COMPLETENESS.md:84/:125/:127/:128`; `HEADCOV_PREGATE_RECORD.md:305-310`; `NCA_FORENSIC_RECON.md:110/:112`; `REDTEAM_BAN_SCOPE_AUDIT.md:303-304` and `:305-308`; `GATE0_REOPEN_2026-07-31.md:243-244` and `:1004-1006`; `HEADSPACE_TRANSFER_PREGATE.md:859-864`, `:871-873`, `:917-919`, `:920-923`; `ERRPAT_HateMM` (nine cited locations plus §0.1 and §7's six ceilings, summing to n=27, with `1/215 = 0.004651`); `ERRPAT_MHC-ZH:220/222-223/233-235/237-240`; CAL-0…CAL-5 and the `:3` header; DET-1…DET-4 including DET-4's *"recommended, not mandatory"*.

**Every number re-derived from source.** All twelve floors; fold sizes; 30 `fold_acc_deployed` cells; bands; `raw_deployed_acc`; `B_fid`; per-seed errors → `84.3333 / 62.3333`; caps ⇒ `k ≥ 132 / 96` dead; `0.8965 / 0.663 / 0.898 / 0.675`; `π*` ⇒ 9.3 points; Hanley–McNeil; the `pred_agree ↔ ov` mapping; `0.1054–0.1142`; F99's `6/215` and `7/149`; `p_w ≥ 30` out of ≈38 = 79 %. Data defects and `GATE-NULL` re-measured independently.

**Executability is real, end to end, and I found nothing that would turn this into an engineering HALT.** All four sha256 match their banked anchors. `mechfix_ops.py:91/:94/:95` literal; `deployed_vote` returns only `(votes, preds, I, sim)`. `headspace_mint.py`'s cited ranges exact; `lab_dev` on the common `np.savez` path, hence in all 36 mints; `det1_assert` at `:75` fires at `:187`; `CLI` exactly `['hatemm','zh']`. `headspace_fidelity.py:57/:66` literal. All ten `vsw_ckpt` files, both `dev_seen` caches and six floor trainlogs present. **No `headspace_arena_en_*` exists anywhere in the repo.** `load_split` is called only with `"train"` and `"dev_seen"`, and `run_rac.load_feats_from_CLIP` is monkeypatched to a closure returning train/dev/dummy-from-fitting-pool — no test path can be reached. The live environment is byte-identical to the banked `meta.runtime`. **`np.random.default_rng(20260801).spawn(6)` works on this numpy — I ran it.** `int(np.floor(82.5+0.5)) = 83` against `round(82.5) = 82`, so the `R(x)` pin is load-bearing and correct. `GATE-FIXK20` is corroborated: banked `FIXK_20` has `changed = 0`, `dacc = 0.0`, `B_agree_fixk["20"] == agree_deployed`. `sacct` confirms `13847` = 8 CPU / 32 G / no GPU / `00:29:49`; `squeue` empty; C04's tranche terminated (`13857`, `13862` COMPLETED).

**The budget is sound and very conservative.** Lines sum to **116** against ≈115. I timed one `lbfgs` fit-and-score at the design's scale under DET-1 threads: **3.6 ms**, against the 35 ms assumed.

**Statistical soundness.** `AUC_strat`'s weighting, zero-weight rule, frozen strata and row-pooling are pinned; `ΔAUC` is genuinely paired. `PERM-STRUCT` is exact for the marginal null over `P^{(τ)}`. `PERM-STRUCT-COND` is exact for the joint form now claimed, and degenerate small strata can only push `p_cond` toward 1 — conservative. Holm within dataset and family, both families required, is conservative; the IUT takes no correction. `SHUFFLE-POP`'s band cannot be silently degenerate. `p_w ≥ 30` is neither inert nor fatal by construction. `K-DEG`'s vacuous regime is unreachable on the CONTINUE path. `ΔmF1_{O1}` monotonicity is genuine arithmetic. The four bands partition completed runs. No single rule can carry a CONTINUE.

**On the residual leak path: none beyond the one §4.3 names.** `i` is never in its own bank; every feature reads only `i`'s key, bank keys and bank labels; `τ_hi^{(f)}` excludes fold `f`; every row in the fold-`f` fit carries `τ_hi^{(f)}`; both stratifications are label-blind; standardisation is fit on training folds only; the `K-DEG` twins feed no feature and no target and can only make the gate fire.

**Legality holds in both directions.** No test path; dev labels materialised exactly 36× and kept outside every decision quantity; the F47, F96, F97, F98(b)+(d), `[9]`, `[2]`/`[4]` and F99 adjudications all concede their bans. `headspace_mint.py` is genuinely not in `frozen_artifact_policy` — §2's statement about what pins the mint is exactly right.

---

## HIGH

**H-1 — §0 asserts, under the heading "retained unchanged in v8", a conditional null that §5.2 explicitly withdraws, and a permutation unit that §5.2 explicitly replaces.** `C09_A0_V8_RECORD.md:69` heads the list *"Rounds 3–6's repairs, retained unchanged in v8:"*. Item 3 (`:84-89`) closes: *"v5 permutes at the **item** level **within each `(dataset, seed)` cell**, states the null honestly as **marginal**, and adds a **residualised conditional null** that a CONTINUE must also clear."* Both italicised clauses are contradicted by the governing body: `:844` reads *"`PERM-STRUCT-COND` … (R5 H-1 — **v5's OLS residualisation is withdrawn**)"* with the reason at `:845-853`; and `:809-823` pins *"**One** permutation `π` of `P^{(τ)}`'s items … per `(dataset, τ, draw)`"*, *"applied identically in all three seed cells"*, so §0's *"within each `(dataset, seed)` cell"* is the v5 phrasing R5 I-1(a) killed and is incompatible with the union pool that is v8's entire reason for existing. The whole four-item list is verbatim v6 text; v7 and v8 re-labelled it without editing its contents, and neither R6 nor R7 audited it. §5.2 governs and is internally consistent, so this is not a wrong-verdict certainty — but a hash-frozen preregistration whose purpose is to remove implementer discretion may not contain, in its own change summary, an affirmative statement that it retains a resampling scheme its body rejects. **Repair:** rewrite item 3 to the v8 object, or relabel the list as history and state that §5.2 governs.

---

## IMPORTANT

**I-1 — the STATUS block is two revisions stale and misdescribes the document's own contents.** `:19-29` is verbatim v6 text. The reading order runs *"… → R5 (0C/3H/10I) → this file"*, omitting v6/R6 and v7/R7; *"v1–v5 are superseded in full"* leaves **v6 and v7 un-superseded** in the only place the document addresses supersession — and `C09_A0_V7_RECORD.md` is on disk carrying the pool specification R7 rated High; *"The v6 repair ledger is §12"* is false in v8, whose §12 heading reads *"the 5 round-7 findings"*. Relatedly §0's tail still ends *"and nine provenance and specification items are corrected"* — R4's Important count.

**I-2 — §1's summary sentence states the strongest possible negative-existence claim, which is the opposite of what A0 does.** `:132-133`: *"A0 trains no encoder, touches no test split, and establishes that no operator exists."* Read as written that asserts a proof of nonexistence — precisely the over-claim §10's first bullet forbids. The intended meaning is *"establishes no operator's existence"*. Present since v2 and never flagged.

**I-3 — §11 names an *encoder-frozen* successor while C09's registry claim and dedup boundary are *encoder-level*; the substitution is unremarked, and F51 appears nowhere.** §11 identifies the successor as GAP-7's cell, *"adapt the retrieval key-map / head recipe only, encoder frozen"*. Those are different objects, and the repository says so at the lines §11 quotes from: GAP-7 is headed *"F51 two-object closure"*, and `REDTEAM_BAN_SCOPE_AUDIT.md:302-304` reads *"F51's 'no third object' is airtight for adapting the MLLM … But the tasking's candidate … is a **different** object F51 does not address."* So F51 does not bind §11's successor — but it does bind the object C09's own claim registers, since an encoder-level region-targeted retrieval term is `dead[F51]`'s *"= P9b's adapted object … do not re-propose."* Nothing here makes the A0 illegal — it trains no encoder — but a CONTINUE would license an operator narrower than C09's registered claim, and the design does not say so.

---

## Bottom line

The science is finished. I could not break the instrument, the legality spine, the executability, the budget, or the inference. Four sha256s match their banked anchors, all twelve floors and thirty fold arrays are exact, the live environment is byte-identical to the banked runtime so `GATE-FLOOR` is achievable rather than aspirational, `spawn(6)` and `R(82.5) = 83` both behave as pinned, the run costs a tenth of what it budgets, and no test path can be opened by any code path I could find. R7's High is genuinely repaired: the union pool `P^{(τ)}` is the right object and one `π` over it is exact for the marginal null. The only route by which a scored item's gold label reaches its own model remains the arena's own bank channel, which §4.3 names, prices and shows cancels.

What blocks the freeze is not science but the document's front matter, which is two revisions behind its body and was never audited by R6 or R7. §0 states in the affirmative that v8 retains the OLS-residualised conditional null — the construction §5.2 withdraws as anti-conservative — and re-imports the per-`(dataset, seed)`-cell permutation unit the union pool replaced; and the STATUS block still reads as v6's. These are edits to four paragraphs, need no re-measurement and no GPU. Fix them and this is ready to freeze and submit as a single CPU-only SLURM job.
