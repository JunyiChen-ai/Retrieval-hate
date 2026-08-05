# C09 Stage-0 (A0) v7 — Independent Design Review, Round 7

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V7_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 1 High / 4 Important`

---

## Prior-round audit

I re-checked every R1–R6 finding against **v7 itself**, and re-opened every source rather than trusting §12. I also diffed v6→v7 line by line to confirm that each repair landed where the ledger says and that nothing else moved.

**Rounds 1–5 — discharged, and they still hold in v7.** R1 C-1 (`pred_purity` against the *predicted* class); C-2 (conditional `AUC_strat` + incremental `ΔAUC`, `π*` re-derived); C-3 (`headspace_mint.py:205` is literally `splits = list(skf.split(np.zeros((n, 1)), lab))`, so the fold map is a function of the label vector alone and is seed-invariant; bootstrap and both permutations item-level); C-4 (`net_s ≡ n·Δacc_s`). R2 C-1, H-3, H-5, H-6, H-7, H-8. R3 C-1, H-1…H-8, I-1…I-11. R4 C-1 (`pred_agree = 1 − |S△twin|/n = 1 − 2k(1−ov)/n`; `aggnet_pregate.py:534/:537`), H-1…H-4, I-1…I-9. R5 H-1/H-2/H-3 and I-1…I-10.

**Round 6 (0C/2H/10I) — all twelve land.**
- **H-1 — DISCHARGED as specified, but the repair opened a new seam.** §5.2 now recomputes `A^{(f)}` per scoring fold for both classes at `τ_hi^{(f)}`; the old text is gone; `P_{τ_hi}` survives only for `|P_τ|`, `K-REACH`, `O1`, `NET`'s `k`; `GATE-NESTED`'s check is extended; per-`(dataset, seed, fold, τ)` counts are emitted. The interaction with I-3's pool pin is HIGH H-1 below.
- **H-2 — DISCHARGED, and both F99 legs are exact.** `dead[F99].ban_scope` reads verbatim as quoted, GAP-7 is at `REDTEAM_BAN_SCOPE_AUDIT.md:293` with its cell text at `:303-304` and the D7 sentence at `:305-308` — both citations correct. The arithmetic checks: `6/215 = 0.02791`, `7/149 = 0.04698`, and `ksweep_OUT.json` independently confirms `n_test = 149` for MHC-ZH.
- **I-1 … I-10 — all DISCHARGED.** 8/13 everywhere; children `2`/`3` in §6.3; `R(x)` in §9 with no `round(` anywhere; `56.0/62.33 = 0.898`; the `~24 min` `GATE-NULL` line and the ≈115 total; all six provenance items; the authorisation line and all three `c09_next_step` clauses; the `banned_constraints[2]` / `hard_constraints[4]` adjudication; and I-10's three precisions including the `[28.95, 29.0)` proof.

---

## Verified as sound (do not re-litigate)

**Every quotation I opened is exact at its cited location.** Verbatim-verified this round: `unified_pilot_gate.{arena, stage_0_reachability}`; `hard_constraints[4]`; `banned_constraints[2]/[9]/[10]`; `candidate_registry[8].{claim, dedup_boundary}`; the reopen's promoted/sequencing/`c09_next_step`/`what_this_reopen_does_not_do[2]`/`strategic_finding` fields; `ban_scope` for F47, F75, F96, F97, F98(b)+(d), F99; `dead[F113].status`'s two measurement sentences; `queued.headspace_arena_EN`; `progress.json:25`; `LITSWEEP3_DATA_CENTRIC.md:82/:91/:95`; `LITSWEEP5_COMPLETENESS.md:84/:125/:127/:128`; `HEADCOV_PREGATE_RECORD.md:305-310`; `NCA_FORENSIC_RECON.md:110/:112`; `RESTRANS_PREGATE_RECORD.md:409`; `GATE0_REOPEN_2026-07-31.md:243-244` and `:1004-1006`; `HEADSPACE_TRANSFER_PREGATE.md:859-864`, `:871-873`, `:917-919`, `:920-923`; `ERRPAT_HateMM:126/130/131/133/134/135/139-141/143-144/395-396` and §7's six cluster ceilings; `ERRPAT_MHC-ZH:220/222-223/233-235/237-240`; `PREGATE_CALIBRATION_CLAUSE` CAL-2(1)/(2)/CAL-3 and the `:3` header; `PREGATE_DETERMINISM_CLAUSE §1.3`, DET-3 Tier B, DET-4's *"recommended, not mandatory"*.

**Every number re-derived from source.** All twelve `GATE-FLOOR` floors; fold sizes; `raw_deployed_acc`; `posrate_bank` → bands; `B_fid`; per-seed errors `83/85/85` and `62/64/61` → `84.3333 / 62.3333`; caps `131.4667 / 95.6667`; `π*` ⇒ 9.3 points; `0.896 / 0.663 / 0.898 / 0.675`; Hanley–McNeil; the `0.95 ↔ ov 0.755` mapping; `0.1054–0.1142`; F99's `6/215` and `7/149`. Data defects and `GATE-NULL` re-measured independently.

**Executability is real, end to end.** All four sha256s match their banked anchors. `mechfix_ops.py:94-95` literal; `deployed_vote` returns only `(votes, preds, I, sim)`, so the extra `k=50` and per-class top-1 searches are genuinely needed and `_norm32`/`_flat_ip` exist to do them on the same engine. `headspace_mint.py`'s six cited line ranges are exact; `lab_dev` in every mint; `CLI` keys exactly `['hatemm','zh']`. All ten `vsw_ckpt` fold files, both `dev_seen_*.pt`, all six floor trainlogs and `headspace_report.py` are present. **No `headspace_arena_en_*` exists anywhere in the repo.** The live environment is byte-identical to the banked `meta.runtime`. `GATE-FIXK20` is corroborated, not merely asserted: banked `FIXK_20` has `changed = 0`, `dacc = 0.0` and all-zero `folddeltas`. `sacct` confirms job `13847` = 8 CPU / 32 G / no GPU / `00:29:49`; `squeue -u jehc223` is empty and C04's tranche has terminated (job `13857` COMPLETED). **I found nothing that would turn this into an engineering HALT.**

**The budget is sound.** Every table line reproduces from its own factors and the lines sum to 114–116 against the declared ≈115, generous by an order of magnitude on the permutation lines.

**Legality holds in both directions.** No test path can be opened; dev labels are materialised exactly 36× and kept outside every decision quantity; the F47, F98(b)+(d), F96, `[9]` and the new `[2]`/`[4]` adjudications all **concede** the ban rather than narrowing it. F99's ban is formally about distilling the F95 verifier into the key space — C09's successor is not that — yet v7 imports F99's cap and novelty leg anyway; that is reading a ban wider than its letter *against* C09, the conservative direction, and is what R6 asked for.

**Decidability.** `K-REACH` forces `2k/n ≥ 0.10`, so `K-DEG`'s vacuous regime is unreachable on the CONTINUE path. `LIVE_ON_NET` leaves two of three `k` live on both datasets at `τ_0`. `p_w ≥ 30` is reachable and can also fail, so it is neither inert nor fatal by construction. The bands partition completed runs; no single rule can produce a CONTINUE. Holm within dataset and family is correct; the IUT takes no correction; `p ∧ p_cond` is conservative. `R(x)` sends `82.5 → 83`.

**On the residual leak path: none beyond the one §4.3 names.** `i`'s own bank never contains `i`; `τ_hi^{(f)}` excludes fold `f`; both stratifications are label-blind; standardisation is fit on training folds only; the feature builder admits neither `lab_query` nor either target-derived array; the `K-DEG` twins feed no feature or target.

**Scope honesty.** §10 claims nothing the design does not support and now carries F99, the headwind that runs hardest against C09. Both "re-measured this session" claims are real `$0` train-side reads and I reproduced both.

---

## HIGH

**H-1 — the R6 I-3 pool pin and the R6 H-1 per-fold analysis set contradict each other at the co-primary, so the null distribution of the primary test is under-determined, and one of the two readings is anti-conservative.** §5.2 pins the permutation pool as *"the `D-FELDMAN` analysis set only … Since the analysis set is now per scoring fold, the pool is `A^{(f)}` for the fold-`f` refit"* and then, in the next sentence, asserts *"**The drawing unit is unambiguous:** per `(dataset, τ, draw)` **one** permutation `π` of that pool's **items** is drawn."* At `τ_0` these agree, because `A^{(f)} = P_0 ∪ {all three-seed-correct}` for every `f`. At `τ_hi` they cannot both hold: `A^{(f)}` is a different set for each of the five folds (§5.2 says so in terms), and a single permutation of items cannot be a permutation of five different pools. An implementer must therefore choose:

- **(a) five `π` per draw, one per scoring fold** — the literal reading. Under (a) each null draw's pooled OOF score vector is built from **five independent** structural re-assignments, so the between-draw variance of `ΔAUC_d` shrinks, the null narrows, and `p` is **too small** — anti-conservative, in the CONTINUE direction. This is the same averaging mechanism R1 C-3 rated Critical for the bootstrap and R4 H-2(i) rated High for the row-level permutation.
- **(b) one `π` per draw over `∪_f A^{(f)}`, restricted to `A^{(f)}` inside each refit** — coherent and conservative, and what the "one `π`" sentence intends — but the document never names the union pool.

The same hole propagates to `PERM-STRUCT-COND` and to `SHUFFLE-POP`. Exposure is confined to `τ_hi` — but `τ_hi` is a registered co-primary that §9's `∃τ` makes CONTINUE-capable, and the design *measures* rather than assumes that `τ_hi` is power-dead. This is exactly the class of defect v7's own §0 rationale for the H-1 repair forbids, applied to the resampling scheme of the only rule that tests. **Repair (one sentence, no re-measurement):** pin the pool as `P^{(τ)} ≡ ∪_f A^{(f)}` (equivalently `{j : c_j ≥ min_f τ_hi^{(f)}}` at `τ_hi`, `P_0 ∪ {3-seed-correct}` at `τ_0`), draw **one** `π` of `P^{(τ)}`'s items per `(dataset, τ, draw)`, and state that inside the fold-`f` refit only the rows of `A^{(f)}` are used, with donors drawn from `P^{(τ)}`; delete "the pool is `A^{(f)}` for the fold-`f` refit"; and apply the identical wording to `PERM-STRUCT-COND` and `SHUFFLE-POP`.

---

## IMPORTANT

**I-1 — `PERM-STRUCT-COND`'s exactness claim omits `BASE` from the conditioning set, and the test statistic reads `BASE`.** §5.2 asserts *"This is an exact permutation null for `struct ⊥ target | ITEM-STRATUM`."* A within-stratum permutation resamples `struct` from its exchangeable conditional law given `ITEM-STRATUM` only; validity for a statistic `T(struct, target, BASE, stratum)` requires `struct ⊥ (target, BASE) | ITEM-STRATUM`. The difference is not idle: the dependence §5.2 itself nominates — structural feature 1's sentinel fires exactly when `pred_purity = 1.0` — is **not** absorbed by the `[0.95, 1.0]` bucket, so the observed `FULL` retains a within-stratum `struct↔BASE` coupling every permuted draw loses. `p_cond` can then reject under the null it was added to protect. §10's scoping is the right honest statement and largely contains the damage; the word "exact" attached to the wrong null is not. **Repair:** state the joint form, and that under the weaker independence a rejection may reflect within-stratum non-linear re-encoding of `BASE`.

**I-2 — §5.3's out-of-support declaration is stale against the R6 H-1 repair and, at `τ_hi`, materially incomplete.** (a) The frozen fit set is now `A^{(f)}`, not `P_τ ∪ CONFIG-MATCHED-CORRECT`. (b) At `τ_hi` **both** classes are `c`-restricted, so the out-of-support population is not just the ~7–9 unstable errors: it is every item with `c_j < τ_hi^{(f)}`, i.e. roughly half of `P_0` plus the whole low-`c` tail of the correct class — while `NET` still ranks all `n`. That is a much larger extrapolation than the one declared, on a rule that can produce a CONTINUE. **Repair:** restate the fit set and add the `c`-restricted items to the named out-of-support population at `τ_hi`.

**I-3 — CAL-2(2)'s secondary skip reason quotes a HateMM-only `n_test` for all three named arms.** In the file, `n_test` is `215` for HateMM, **`149`** for `MHC_zh` and **`161`** for `MHC_EN_ARM-V`. Nothing gating turns on it, but the ZH figure is load-bearing elsewhere in this very document (F99's `+0.0470 = 7/149`).

**I-4 — `THRESH-SYM`'s per-seed scale is grammatically ambiguous in a KILL-capable gate.** Read as "average the scores", the per-seed twin becomes identical to the per-item twin, collapsing the two-scale maximum §6.2 explicitly adopts as *"the conservative direction for a gate whose job is to fire."* **Repair:** "…the `k` items with the smallest `|score_{i,s}|` **within each seed**; `pred_agree` is computed per seed and averaged over the three seeds."

---

## Bottom line

This remains the strongest document in the lineage and I could not break the parts that matter most. The instrument is real, on disk, sha-pinned and demonstrably reproducible in the live environment; the legality spine holds in both directions; the only route by which a scored item's gold label touches its own model is the arena's own bank channel, which §4.3 names and prices; §10 claims nothing the design does not support and now carries F99. R1–R5 are fully discharged and all twelve of R6's findings land.

One thing must change before freeze, and it is a one-sentence fix. R6's H-1 repair made the analysis set fold-dependent; R6's I-3 repair pinned the permutation pool to that same fold-dependent set — and then asserted a single permutation per draw. At `τ_0` the two agree. At `τ_hi` they cannot, and the reading the letter forces narrows the null in the CONTINUE direction. Name the union pool and the contradiction disappears. The four Importants are precision, provenance and completeness.
