# C09 Stage-0 (A0) v6 — Independent Design Review, Round 6

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V6_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 2 High / 10 Important`

---

## Prior-round audit

I re-checked every R1–R5 finding against **v6 itself**, not against §12, and re-opened every source.

**Round 1 (4C/8H/10I) — discharged, and it still holds.** C-1: `pred_purity` is against the *predicted* class (§4.4), label-blind for the scored item. C-2: the discriminator is conditional (`AUC_strat`) and incremental (`ΔAUC`), with `π* = (1+bar/k)/2` re-derived as the exact inversion of `net = k(2π−1)`. C-3: `headspace_mint.py:205` is literally `splits = list(skf.split(np.zeros((n,1)), lab))`, so the fold assignment is a function of the label vector alone and is identical across head seeds; the bootstrap and both permutations are now item-level. C-4: `net_s = 2|S∩wrong_s| − k ≡ n·Δacc_s` re-derived as an identity. H-3: `mechfix_ops.py:93-95` exactly as quoted. H-5/H-8/I-9 all land.

**Round 2 (1C/8H/10I) — closed.** Feature 11 stays deleted; F97's `+0.0269 / +0.0104` is registered as band B and as the design's *expectation*; F98(b) is adjudicated rather than narrowed; `K-FELDMAN` is computable; the gate/decision split holds; CAL-3 is discharged on `ERRPAT_HateMM §7`'s six cluster ceilings (all twelve numbers exact, `:410/:432/:444/:460/:477/:486`). I-6's mint layout is right on disk. Budget arithmetic re-verified line by line.

**Round 3 (1C/8H/11I) — all discharged.** §9's publication precondition, the two-scale twins, the permutation null, `STRATUM_OCCUPANCY`, the per-fold `τ_hi`, the withdrawn `NET` over-claim, `37.2/29.0`, the frozen sentinel `21`, `GATE-FIXK20`, the ZH raw-space relabelling, F113's three-clause caveat verbatim, the runtime quintet, deleted `q25/q75`, pinned `numpy.median`.

**Round 4 (1C/4H/9I) — all discharged.** C-1's `pred_agree = 1 − |S △ twin|/n = 1 − 2k(1−ov)/n` is exact; `aggnet_pregate.py:534` is literally `(c3 == coll["THRESH_best"]).mean()` and `:537` the same for `B_agree_fixk`. H-3's caps, H-4's HALT→data-outcome demotion, I-1's `9.3` points, I-2's "at least six", I-4, I-5, I-7, I-8(a)–(f), I-9's Hanley–McNeil all verified independently.

**Round 5 (0C/3H/10I) — 9 of 13 clean, 4 partial.**
- **H-1 — DISCHARGED as specified.** The OLS residualisation is withdrawn; `PERM-STRUCT-COND` now permutes within `ITEM-STRATUM`, which *is* an exact permutation null for `struct ⊥ target | ITEM-STRATUM`. R5's own primary repair, adopted, with the conditioning set named and `MARGINAL_ONLY_NOT_CONDITIONAL` scoped to it.
- **H-2 — DISCHARGED.** `p_w ≤ |P_τ|` computed in-run, cells marked, the joint ZH requirement written out, §9 adds the liveness conjunct.
- **H-3 — DISCHARGED in substance, but the repair created a new gap** (HIGH H-1 below).
- **I-1 — 2 of 3.** The permutation *pool* is still unstated (IMPORTANT I-3).
- **I-2 — PARTIAL.** The substream policy is not propagated to §6.3 (IMPORTANT I-2).
- **I-3 — PARTIAL.** §9 still writes `round(...)` (IMPORTANT I-4).
- **I-9 — attempted, one figure wrong** (IMPORTANT I-5).
- **I-4, I-5(a)–(f), I-6, I-7, I-8, I-10 — DISCHARGED**, with one line-number drift.

---

## Verified as sound (do not re-litigate)

**Executability is real; I re-opened every cited line.** `headspace_mint.py:106-116`, `:126` (`CLI` keys exactly `['hatemm','zh']`, enforced at `:177`), `:188-189`, `:192-194`, `:203-216`, `:274-281`, `:285-286`, `:322-324` (`lab_dev` in **every** mint). `mechnov_pairverify.py:56-57`, `:124`; `headspace_arena.py:7`; `headspace_fidelity.py:66` and its `--mintdir` at `:57`. The four sha256s match their banked anchors; the live `python 3.11.8 / numpy 1.26.4 / scipy 1.17.1 / sklearn 1.5.2 / torch 2.6.0+cu124` on `foscsmlprd01` is byte-identical to the banked `meta.runtime`. `squeue -u jehc223` is empty and C04's tranche has terminated (F117, job `13857`, COMPLETED), so the first sequencing precondition is satisfied in fact. **I found nothing that would turn this into an engineering HALT.**

**Every floor and count re-derived.** All twelve `GATE-FLOOR` values exact; 30 `fold_acc_deployed` cells present; fold sizes; `raw_deployed_acc 0.8441/0.8480`; `posrate_bank` → bands; `B_fid 0.0093/0.0086`; **`B_agree_fixk["20"] == agree_deployed` in 6/6 cells and `FIXK_20.d_acc = 0.0`, so `GATE-FIXK20` is corroborated rather than merely asserted**; per-seed error counts `83/85/85` and `62/64/61` integer-exact → `84.33 / 62.33`; caps; `DEG_KILL = 0.95`. **No `headspace_arena_en_*` exists anywhere in the repo** — §2's primary EN reason is true. Data defects re-measured independently.

**Decidability holds in both directions.** `K-REACH` at `τ_0` forces `2k/n ≥ 0.10`, so `K-DEG`'s vacuous regime is unreachable on the CONTINUE path; the `0.95` line maps to set-overlap `0.755–0.878` (HateMM) and `0.737–0.868` (ZH) across the live `k`. The four bands partition completed runs. Holm within dataset and family is correct; the across-dataset IUT correctly takes no correction; `p ∧ p_cond` is conservative. `ΔmF1_{O1}` monotonicity is right. `net_s ≡ n·Δacc_s`. `c_i` is label-blind for `i`.

**On the residual leak path, the answer is: none beyond the one §4.3 names.** Features are structurally blocked from `lab_query`, `is_inversion[seed]` and `is_stable_inversion`; standardisation is fit on training folds only; strata are label-blind; `τ_hi^{(f)}` excludes fold `f`. The only surviving channel is the arena's own bank channel, which §4.3 states exactly, prices, and correctly observes cancels in the paired `ΔAUC`.

**On `p_cond`'s exactness.** Within-`ITEM-STRATUM` permutation is exact for the stated null. It is not exact for `struct ⊥ target | BASE` in full, but v6 declares this and scopes it in §10 in the precise terms R5 asked for. Declared limitation, not a finding.

**Scope honesty and legality.** §10 claims nothing the design does not support. No test path can be opened; dev-label materialisation is bounded at 36; `H-L1`…`H-L4` are the right boundaries; §3's new `banned_constraints[9]` adjudication is correct. I found no ban read *narrower* than its own text.

---

## HIGH

**H-1 — §5.2's analysis-set definition and §4.3(b)'s fitting target use two different `τ_hi` conventions, so at the co-primary the design does not determine which rows are positives and which are negatives in the model that scores fold `f`.** §4.3 pins **(b) the FITTING TARGET** as *"for the model that scores fold `f`, every row in that fit … is labelled with that fit's own `τ_hi^(f)`"*. But §5.2 defines the analysis set with the **(a)** convention on both sides: *"Positives: `P_τ`"* — and `P_{τ_hi} ≡ { i ∈ P_0 : c_i ≥ τ_hi^{(fold(i))} }` — and *"Negatives: … `c_i ≥ τ_hi^{(fold(i))}`"*. The v5→v6 H-3 repair was written in the (a) convention while the v4→v5 H-1 repair had already moved the target to (b), and nothing reconciles them. Read literally, a stable inversion `j ∈ fold g` with `τ_hi^{(g)} ≤ c_j < τ_hi^{(f)}` is **in** the analysis set but carries `y_j^{(f)} = 0`, i.e. sits in the *negative* class of the fold-`f` fit, contradicting §5.2's own definition of that class and §6.1's statement of the contrast. Both readings are conservative and the magnitude is a handful of items, so this is not a wrong-verdict risk — but it is an under-determined frozen quantity on a **live** decision path, and a preregistration cannot leave the composition of its estimator's two classes to the implementer. **Repair:** state in §5.2 that the analysis set is **recomputed per scoring fold** under `τ_hi^{(f)}` for both classes, keep `P_{τ_hi}` only for `|P_τ|`, `K-REACH`, `O1` and `NET`'s `k`, extend `GATE-NESTED`'s per-item check to assert it, and note that the emitted negative counts are per-fold.

**H-2 — §11's named successor is the GAP-7 head/key-map class, and the campaign has a banked finding on exactly that class — F99 — carrying both a novelty closure and a closed-form arithmetic cap on the successor's principal channel. Neither appears among §11's headwinds, and F99 is absent from §11's own pre-registered Stage-1 adjudication list.** §11 names the successor as *"a global, symmetric, train-label-supervised reshaping of the head map `φ₀ → φ′`"*, encoder frozen — i.e. `REDTEAM_BAN_SCOPE_AUDIT.md` GAP-7's *"adapt the retrieval key-map / head recipe only, encoder frozen"*. §11 then makes the Stage-1 precondition binding, requiring *"(d) accompanied by a fresh ban-scope adjudication against F75, F66 and F98"* — and F99 is not in that list, nor named anywhere in v6. Two things F99 banks run directly against C09. **(1)** `dead[F99].ban_scope`, verbatim: *"D7 additionally kills the NOVELTY CLAIM regardless of the number (GAP-7 head/key-map class)."* **(2)** The load-bearing half — a closed-form ceiling on the successor's set-preserving channel, computed from the same marginals v6 already cites: from `w = [20…1]`, `Σw = 210` and the measured cone collapse, the best permutation flips a prediction iff top-20 purity `≥ 6/20` (hate) / `≥ 7/20` (non-hate); crossed with the measured error purity (`ERRPAT_HateMM:143-144`, `ERRPAT_MHC-ZH:222-223`), **at most 6 of 27 HateMM errors and 7 of 22 ZH errors are permutation-flippable, capping any set-preserving re-metrication at `≤ +0.0279` / `≤ +0.0470` under a zero-break assumption "the campaign has never met."** That is a statement about **C09's own target population**. §10's *"`O1` … is the only upper bound this A0 produces"* is true of *this A0* but leaves the reader without the far tighter banked bound on the object a CONTINUE would license. This design registers weaker headwinds at length; omitting F99 is the one asymmetry I could find in an otherwise scrupulously two-sided document, and it runs against C09. **Repair.** Add F99 to §11's headwind list and to precondition (d)'s adjudication set, quoting the GAP-7/D7 sentence and the `+0.0279 / +0.0470` permutation cap with its zero-break and test-split-purity provenance; and add one bullet to §10 recording that the set-preserving channel of the successor carries a banked closed-form ceiling far below `O1`.

---

## IMPORTANT

**I-1 — the frozen feature-set cardinalities are stale in three places.** §5.2's header reads *"Two frozen feature sets — 7 and 12"* while its own enumerations give `BASE` = **8** and `FULL` = **13**; §3.2(3) and §10 both say *"a 12-feature `lbfgs` logistic probe"*.

**I-2 — R5 I-2's RNG repair did not propagate to §6.3.** §6.3 still specifies a bare `default_rng(20260801)` for `SHUFFLE-POP` and `RANDOM-POP` — the exact construction R5 I-2 found unacceptable, in the section that defines a HALT gate.

**I-3 — the permutation *pool* is still unstated for both nulls, and R5 I-1(c) named it.** §5.2 never says whether `π` runs over all `n` items or over the `D-FELDMAN` analysis set. At `τ_hi`, where the analysis set is `c`-restricted, permuting over all `n` would import structural vectors from low-`c` items and change the null distribution.

**I-4 — §9 reverts to the word `round`, which §5.3 pinned `R(x)` precisely to avoid.** §9 is the rule that governs at verdict time; it must use `R(·)`.

**I-5 — the R5 I-9 re-derivation carries an arithmetic error, twice.** `56.0 / 62.33 = 0.89844` → **`0.898`**, not `0.899`, in §5.3 and in §12's I-9 row. (The HateMM companion and both precisions are right.)

**I-6 — `GATE-NULL`'s remove-null sensitivity has no line in the budget.** Clause 3 requires agreement on every K-rule outcome, so the sensitivity requires re-running both 1000-draw nulls plus `SHUFFLE-POP`, `RANDOM-POP`, the bootstrap and `K-DEG` on HateMM — ≈24 CPU-min, ~27 % of the declared total. No re-mint is implied and the job carries no `--time`, so this is completeness rather than feasibility.

**I-7 — provenance bundle (six items).** (a) §6.2 presents the `95.03% / 97.75% / 99.45%` sentence as "F96's number" with no citation; F96's `ban_scope` contains no such sentence — the nearest true text is `RESTRANS_PREGATE_RECORD.md:409`. (b) §1's F98 epitaph is exact only against `TARGET_STATE.json`'s rendering. (c) §6.3 calls F88's null (3) *self*-labelled; the quoted string is the reopen's compression at `GATE0_REOPEN:243-244`, and `ERRPAT_HateMM:395-396` phrases it differently. (d) Three sentences attributed to **F113** are not in `HEADSPACE_TRANSFER_PREGATE.md`; they trace to `directions_tried.json:525` and, for the KILL/PROMOTE clause, to `unified_pilot_gate.arena`, where every source writes lower-case "promote". F113's `dead[]` entry has no `ban_scope` and closes *"STANDING RULE PROPOSED (not yet ruled)"*. (e) The *"9 of 9 raw positives…"* clause **begins on `:863`**. (f) `C02_A0_OUT.json`'s real paths are `datasets.<ds>.gates.ARENA2.seed<n>.pooled_native_acc` and `datasets.hatemm.gates.VIEW_SUPPORT.degenerate_causes.EMPTY_TEXT`.

**I-8 — the header's authorisation line contradicts the record it cites, and drops one of that record's own preconditions.** The reopen's `what_this_reopen_does_not_do[2]` reads *"it authorizes no work…"* and `c09_next_step` ends *"NOTHING IS AUTHORIZED BY THE REOPEN"*. Separately, `c09_next_step` opens *"the draft must be repaired against the round-1 review and re-reviewed before anything is frozen"* — the precondition box drops this clause.

**I-9 — `banned_constraints[2]` / `hard_constraints[4]` (cross-seed ensembles) are never named, though the design's three most central objects are cross-seed aggregates.** The defence is clean and short — nothing is deployed, `H-L4` forecloses the selector, and `[4]`'s own qualifier is *"as a final performance method"* — but v6 gives named adjudications to `[9]`, `[10]`, F47, F96, F97 and F98 and is silent here.

**I-10 — three residual scoping precisions.** (a) CAL-2(2)'s sole surviving skip reason is over-broad: `ksweep_OUT.json`'s `curves[...]` has a `dev` per-`k` sub-key beside the `test` one, so *"the per-`k` payload is a TEST-SPLIT read"* is true of the sub-key CAL-2 leaves unnamed, not of the file. (b) §2's EN out-of-scoping does not mention `queued.headspace_arena_EN`'s *"~15 CPU-min"* price. (c) §5.3's caps are stated *looser* than the true values (`131.4667 / 95.6667` written `131.5 / 95.7`), and `28.95` vs `29.0` are used for the same ZH figure in different sections (neither can move an outcome, but they should be reconciled).

---

## Bottom line

This is the strongest document in the lineage and I could not break the parts that matter most. The instrument is real and executable end to end; the legality spine is sound in both directions; the only surviving route by which a scored item's gold label touches its own model is the arena's own bank channel, which §4.3 names, prices, and correctly shows cancels. R1–R4 are fully discharged; nine of R5's thirteen land cleanly and four partially. Band B remains the pre-declared expectation and a KILL scoped *"not identifiable at this power"* is a complete and valuable result.

Two things must change before freeze. The v6 repair of R5's H-3 was written in the `τ_hi` convention that the v5 repair of R4's H-1 had already replaced, so the composition of the estimator's two classes at the co-primary is under-determined. And §11, the clause that gives a CONTINUE its only forward meaning, omits F99 — the one banked finding written on exactly the object it names. The ten Importants are precision and propagation.
