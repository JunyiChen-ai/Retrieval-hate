# C09 Stage-0 (A0) v11 — Independent Design Review, Round 11

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V11_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 2 Important`

---

## Prior-round audit

I diffed v10 → v11 line by line (`diff -u`, 257 lines, 12 hunks) and re-verified R1–R10 against **v11 itself**, re-opening every source on disk rather than trusting §0 or §12.

**The diff, exactly.** v11 changes: line 1 (title); line 17 (STATUS); lines 27–38; lines 54–99 (§0); one sentence in §5.2; three sentences in §5.3; **one** sentence in §6.1; **one** sentence in §7; one sentence in §11; and §12. `§§1, 2, 3, 4, 5.1, 5.4–5.6, 6.2, 6.3, 8, 9, 10` are byte-identical to v10. No rule, threshold, feature set, null, gate or arithmetic moved.

**Round 10 (0C/0H/2I) — both findings discharged.**
- **I-1 — DISCHARGED.** v10's false *"Nothing in §§2–9 changed in v10"* is gone; §0 now mirrors §12's form.
- **I-2 — DISCHARGED, both halves.** §5.2 defines `CONFIG-MATCHED-CORRECT` at the point of definition with both changes of object named; §7 names `net_s` (§5.3, line 978), `GATE-SELFTEST` (§8.1, line 1427) and the reported class composition (§5.3, lines 1003–1005). All three referents check out.

**Round 10's five below-the-line items — all five fixed, all five correct.** The fragment; `7.45`; "twelve `(dataset, τ, k)` cells, six per dataset"; the exact means `253/3` and `187/3` (both re-derived, `k ≥ 132 / 96` dead either way); and F66's re-open clause quoted **verbatim** against `directions_tried.json` `dead[]` index 40 — character-exact, em-dash included.

**Rounds 1–9 — every Critical and every High still lands in v11.** In particular R4 C-1, R4 H-1, R6 H-1 and R7 H-1 are intact and were not disturbed by the §5.2 alias insertion.

---

## What I verified as sound

**Quotations: every one exact at its cited location.** All registry fields, all `ban_scope`s (F47, F51, F66, F75, F96, F97, F98(b)+(d), F99; F113's entry genuinely carries **no** `ban_scope`), all `.md` citations including `HEADSPACE_TRANSFER_PREGATE.md:623/:630-643/:859-864/:863-864/:871-873/:917-919/:920-923`, `ERRPAT_*`, `REDTEAM_BAN_SCOPE_AUDIT.md:293/:302-304/:305-308`, `RESTRANS:409`, both standing clauses, `progress.json:25/:30`. Every ellipsis I checked hides material that is neutral or adverse to C09. The disowned renderings are correctly attributed to the rendering source rather than the primary.

**Every number re-derived from source.** All twelve floors; `raw_deployed_acc`; per-seed errors `83/85/85` and `62/64/61`; `mean_s|wrong_s| = 253/3` and `187/3`; both caps; `R(1.5×55) = 83`; `114/83`, `152/110`, `75.6`, `56.0`, `0.896/0.663`, `0.898/0.675`; `π*` at `k=80` and `k=100`; the `[28.95, 29.0)` emptiness; the Hanley–McNeil SE re-derived from `Q1`/`Q2` at `A=0.5`; the `pred_agree` mapping; `6/215`, `7/149`; `n_test 215/149/161`; the fidelity anchors; F98's DEG-A; majority rates and both bands. Both "re-measured this session" claims reproduce on my own reads, as do the per-fold held-out counts from `StratifiedKFold(5, shuffle=True, random_state=0)`.

**Executability: nothing here becomes an engineering HALT.** `sha256(headspace_mint.py)` matches its banked anchor everywhere; **the mint's other run-time assertion — `sha256(mechnov_pairverify.py) == FROZEN_PAIRVERIFY_SHA` at `:188-189` — also matches on disk**, so the mint will not refuse to run. `--dataset` admits exactly `{hatemm, zh}`. `load_split` is called only with `"train"` and `"dev_seen"`; the `torch.load` guard bars `test_seen` and `/test`; `torch.save` is a no-op and `best_epoch_path` is re-loaded only on an EM branch this recipe never takes, so **there is no dev-based model selection anywhere** — the load-bearing fact behind the whole label-use spine. `GATE-FIXK20` is corroborated in all six banked arenas. The live environment is byte-identical to the banked `meta.runtime`. `squeue -u jehc223` is empty.

**Budget: sound and conservative.** Job `13847` ran 8 CPU / 32 G / no GPU in `00:29:49`. The 36 banked mint times re-derive to min `33.2`, max `60.0`, median `41.85`, total `24.51` min — exact. Every fit-count multiplies out; lines sum to 116 against "≈115".

**Statistics and decidability.** `AUC_strat` fully pinned; `ΔAUC` paired. `PERM-STRUCT` exact for the marginal null over `P^{(τ)}`: only `struct` is permuted and `A^{(f)}`/`P^{(τ)}` are measurable w.r.t. `(target, BASE)`, so the evaluation sets are invariant across draws. `PERM-STRUCT-COND` exact for the joint form claimed. Both families required = intersection = conservative; Holm correct; IUT correct; the existential over `τ` covered. I re-derived the macro-F1 monotonicity independently. All four rules individually non-vacuous and jointly decidable; **no single rule can carry a CONTINUE**; a KILL is fully available and is the pre-declared expectation.

**Residual gold-label paths: none beyond the one §4.3 names.** All 13 features walked. The one remaining channel is named, priced, and identical for `BASE` and `FULL`, so it is inside the paired difference and inside both nulls' own calibration.

**Legality holds in both directions.** F51, F99 and now F66 all run **against** C09's own registered claim or successor and are carried anyway.

---

## IMPORTANT

**I-1 — §6.3's and §5.3's pre-declared `n_unstable ≈ 7–9` is the *per-seed* count of non-stable errors, not the item-level unstable population, and it is arithmetically incompatible with the document's own `|P_0| ≈ 76 / 55`.** §4.2 defines an unstable error at the **item** level and `D-FELDMAN`'s target is per item, so `n_unstable` is an item count. Write `E_s` for the seed-`s` inversion set, `a` = #(wrong in exactly 2), `b` = #(exactly 1). Then `Σ_s |E_s| = 3|P_0| + 2a + b`, with `n_unstable = a + b`. The per-seed counts are **measured**: `83+85+85 = 253` (HateMM), `62+64+61 = 187` (MHC-ZH).

- HateMM at `|P_0| = 76`: `2a + b = 25` ⇒ `n_unstable ∈ [13, 25]`. The declared `7–9` is **unreachable** — the minimum is 13.
- MHC-ZH at `|P_0| = 55`: `2a + b = 22` ⇒ `n_unstable ∈ [11, 22]`; and the very F88 datum §4.2 quotes (*"NOTHING at exactly 2/3"*) forces `a = 0`, i.e. **`n_unstable = 22`, above the `20` trigger**, so `CONTROL_UNDERPOWERED` would *not* fire on that leg.
- Conversely, `n_unstable ≤ 9` forces `|P_0| ≥ 78.3` / `≥ 56.3`, contradicting `≈ 76 / 55` and shifting `|P_{τ_hi}|` off the `38 / 28` on which §5.2's *"the `τ_hi` branch cannot produce a CONTINUE at all"* and §10's ZH bullet both rest.

The origin is visible: `84.33 − 76 = 8.33` and `62.33 − 55 = 7.33` — the per-seed excess. **Why Important and not High:** no decision rule reads it, `UNSTABLE-POP` is explicitly non-gating and its power rule is data-driven, and each of the two incompatible statements is individually conservative in its own context. But the document cannot pre-declare both, and one of the two things §10 will publish with the verdict is arithmetically impossible. **Repair:** keep `|P_0| ≈ 76 / 55` and declare `n_unstable ∈ [13, 25] / [11, 22]`, noting that on the ZH transfer the control may in fact be powered on the count leg, with §10's stability-premise bullet re-worded to depend on the realised tag.

**I-2 — v11's own scope statement, in both places, mis-describes v11's own contents.** (1) **§6.1: one sentence changed, not two** — the preceding sentence is byte-identical; "two" is v10's edit count carried into v11's description of v11. (2) **§7: one of the two `CONFIG-MATCHED-CORRECT` sentences was edited**; the controls-enumeration sentence is byte-identical between v10 and v11. (3) **§0 says "one line of §5.2"; §12 says "one sentence"** — of the same edit, in the two statements §0 says it is mirroring. This changes no rule and the substantive invariant attached to it is **true** — I verified it against the diff — but it is the fourth consecutive round in which the change summary misdescribes its own contents, and it is the one sentence whose entire job is to be checkable.

---

## Checked and deliberately not counted

§5.3's "superseded phrasing" note (still true after the alias insertion, since `P_τ` and `A^{(f)}` remain different objects); §11's F66 paragraph in mild tension with the NCA counterweight (runs **against** the candidate, Stage-1-scoped, already covered by precondition (d)); F75's semicolon truncation (nothing adverse hidden); typographic normalisation inside quotations; §4.3's `≈ 1.4 %` for a withdrawn v4 construction; §2's budget summing to 116 against "≈115"; §2's `ksweep` parenthetical erring against the design; §5.3's R10 attribution slack.

---

## Bottom line

I could not move the science, and neither of my findings is a defect in it. I re-derived every number in §§1–11 from the named source on disk, re-opened every quotation, walked all 13 features and all five thresholds for a residual gold-label path, re-derived both permutation nulls' exactness arguments and the Hanley–McNeil algebra from scratch, confirmed the four decision rules are jointly decidable and individually non-vacuous with none able to carry a CONTINUE alone, confirmed the mint performs no dev-based model selection and that no test path is reachable, matched **both** of the mint's run-time sha256 assertions to their on-disk targets, and confirmed the live environment is byte-identical to the banked runtime. The legality spine concedes every adjacent ban, including three — F51, F99 and now F66 — that run against C09's own registered claim or successor.

**I-1 is the one that matters**, and it is the first arithmetic error found in this document since round 6. **I-2** is the change summary mis-describing its own diff for the fourth round running.

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not yet — but the gap is one paragraph and one sentence.** Neither repair needs re-measurement, GPU, or any change to a rule, threshold, feature set, null, gate or arithmetic. With those made, the frozen set is ready.
