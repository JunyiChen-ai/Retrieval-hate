# C09 Stage-0 (A0) v17 — Independent Design Review, Round 17

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V17_RECORD.md`
**Verdict.** `GO` — **0 Critical / 0 High / 0 Important**

---

## Prior-round audit

I diffed v16 → v17 (6 hunks), section-hashed both, re-opened every source on disk and re-derived the arithmetic myself.

**Round 16's three Importants — all three DISCHARGED at source.**
- **R16 I-1.** The two sentences carrying `§8 (all ten gates)` sat at `C09_A0_V15_RECORD.md:73` and `:1879`; in v16 both were **deleted and rewritten** to describe the hashed span — exactly what round 16 said, and a better repair than v16's ledger claimed. v17's row I-1 now describes that. `all ten gates` occurs **once** in v17, inside row I-1's quotation of the defective string, and nowhere in governing text.
- **R16 I-2.** `C09_A0_V15_PREREG_REVIEW.md:68` ends *"…verdict or scope."*; `scope of any conclusion` occurs **0 times** in round 15's review and **1 time** in round 14's. v17 quotes round 15 exactly and names the splice.
- **R16 I-3.** Now reads *"the ledger of THIS version is always §12"*, with the freeze history named. Version-independent; cannot restale.

**v17's scope claim, verified against the actual diff.** The diff touches exactly: title; STATUS tag; STATUS reading-order + ledger pointer; §0 (including the new *"What is NOT yet done"* paragraph); §12; trailer. **Nothing between v17:114 and v17:1887 changes — §§1–11 are byte-identical to v16.** Five edit sites, exactly as declared.

**The "character-identical apart from the cross-reference" claim, verified mechanically.** Word-diffing both copies: the *only* difference is `copied into §12` vs `copied into §0`.

**§8 + §9 byte-identity, re-verified by my own hash.** `## 8. Gates` through the line before `## 10.`: v8 `a17b56954ee6955013327f82a03904f7`; v15 same; v16 same; **v17 same**. No gate, no publication precondition and no decision rule has moved in nine versions.

**Findings from R1–R16 still open in v17: none.**

---

## What I verified as sound

**Instrument and executability.** All six analysis modules, `headspace_drive.sh`, `c02_a0_cpu_v9.sbatch`, `ksweep_OUT.json`, the six arena outputs, both fidelity outputs and `C02_A0_OUT.json` are present. `sha256(headspace_mint.py) = cefdf8dc…6612`, **identical to `meta.mint_script_sha256` in all six banked arenas**; the mint's own `FROZEN_PAIRVERIFY_SHA = 77b0defd…240d` matches `mechnov_pairverify.py` on disk today, so the mint runs unmodified. `CLI` admits exactly `{hatemm, zh}` — the MHC-EN scope argument is structural. `:288` forces `--device cpu`; `det1_assert` fires at `:187`; `:192-194` gives the resume path; `:203-216` asserts the fold partition against the banked `vsw_ckpt` and refuses on mismatch; `:274-281` no-ops `torch.save`; `:285-286` matches the layout §2 describes. `headspace_fidelity.py:66` hard-codes the `ffull` name; its six FLOOR trainlogs exist; `VAL_RE` drops every non-`Val_Retrieval` line. `mechfix_ops.py:94/95` literal; `Σw = 210`.

**Gate anchors, re-read.** All twelve `GATE-FLOOR` values reproduce exactly, and C02's independent re-mint reproduces them to full precision. Fold counts, `raw_deployed_acc`, `GATE-DEVFID`'s references, `GATE-ARENA`'s bands, `GATE-NULL` (re-measured on both operative caches) and `GATE-FIXK20`'s premise all check out.

**The F88 transfer, re-derived from the ERRPAT tables.** `28/26/26` (Σ 80) with `(25, 2, 1)`; `23/24/22` (Σ 69) with `(22, 0, 3)`; both identities close. Scaling gives `|P_0| ≈ 79 / 60`, `n_unstable ≈ 9 / 8`; closure exact. Val-sel counterfactual `≈ 77`.

**The knife edge, checked past the document.** I reconstructed both confusion matrices exactly from `posrate_deployed`/`posrate_bank`: HateMM `256/41/42/405 ⇒ mF1 0.883777 = 0.8838`; MHC-ZH `148/30/32/369 ⇒ 0.874659 = 0.8747`. **The accuracy leg binds at `τ_hi` on both datasets**, so §5.2's accuracy-only knife edge is complete. All twelve `k`-grid figures, the caps, the required recalls/precisions, the `π*` values, the reach knife edges, the Hanley–McNeil SEs and the `K-DEG` algebra all re-derive.

**Statistical soundness and decidability.** `AUC_strat` fully pinned. `PERM-STRUCT` is an exact marginal permutation null with the item as the unit and the conservative choices in every case. `PERM-STRUCT-COND` is exact for `struct ⊥ (target, BASE) | ITEM-STRATUM`, with the residual coupling declared. Holm over `m = 2` correct, both families required; the dataset conjunction is a genuine IUT; the `p`-floor `≪ α/2`. `IDENTIFIABILITY_UNDERPOWERED` *fails* `K-FELDMAN`. I independently verified §4.3's monotonicity claim (`(A+2)/(A+B+1) > A/(A+B)`), so `K-REACH` at `τ_0` closes every `τ ≥ 0` by arithmetic, and it is correctly declared false for `NET` and `ΔAUC`. `GATE-SELFTEST`'s identity is exact. §9 is two-valued and exhaustive on completed runs; **a KILL is fully reachable at both `K-FELDMAN` and `K-NET`.**

**Legality / ban-scope.** Zero GPU. Train + `dev_seen` only; the `torch.load` guard and `VAL_RE` close every test path; the expected dev-label count of `36` matches the unconditional `np.savez`. Every ban quoted **verbatim** and adjudicated at source. **F113's `dead[]` entry has keys `['name','status']` only.** CAL-0…CAL-5 check out. The three texts that run **against** C09 are carried at their adjudicated weight; F114 is correctly distinguished from its C04-lineage homonym.

**Budget.** `sacct -j 13847` confirms 8 CPU / 32 G / **no GRES** / `00:29:49`. The 36 banked mint durations recompute to min `33.2`, max `60.0`, median `41.85`, total `24.5` CPU-min against a budgeted `≤ 36`. The permutation lines are generous by a factor of ~2–5.

**Scope honesty.** `find` returns **no C09 namespace of any kind**; the analysis script and sbatch driver genuinely do not exist, exactly as §0 says. §10's ten scope bullets are each consistent with the section they cite, and every withdrawal is honoured in the governing text.

---

## Bottom line

**The science is finished and I could not move it — for the ninth round running.** The instrument, the arena and its fold contract, the legality spine, the four label-use adjudications, the two nulls and their honest caveats, the Holm/IUT structure, the power rule, the closed-form caps, the degeneracy control, the executability, the three sha256 pins and the budget all hold under independent re-derivation, and **§8 and §9 are byte-identical to v8 on my own hash**. A KILL is fully available at `K-FELDMAN` and at `K-NET`, and remains the honest expectation given the registered `+0.0269 / +0.0104` prior.

**I raise no finding.** Three things I deliberately did not count, so the record shows they were seen: (i) row I-1's *"a string that appears nowhere in v16"* is self-referentially loose — the intended and only sensible reading ("nowhere as the corrected text") is true, and round 16's own *"zero times"* was the same looseness; (ii) row I-2's *"round 15 ends"* refers to how that sentence ends; (iii) everything on v17's own below-the-line list, none of it load-bearing on any rule.

**The DESIGN is ready to hash-freeze.** What remains is not design work: implement the analysis script and the sbatch driver, hash-freeze the frozen set, and pass a **separate** independent code/resource review. Then the STATUS block's three preconditions plus the immediately-prior `squeue` empty-check, sha256 re-verification and namespace-absence check. For the record, `squeue -u jehc223` is empty as of this review and no C09 namespace exists on disk; neither fact authorizes anything.
