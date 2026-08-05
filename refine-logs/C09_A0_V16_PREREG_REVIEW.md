# C09 Stage-0 (A0) v16 — Independent Design Review, Round 16

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V16_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 3 Important`

---

## Prior-round audit

I diffed v15 → v16 (7 raw hunks), hashed the document section by section, re-opened every source on disk, and re-derived the arithmetic myself.

**Round 15's five findings — four DISCHARGED at source, one repaired-but-misdescribed.**
- **R15 I-1 (`0.9575`) — DISCHARGED.** `aggnet_pregate.py:534` is exactly `"A_agree_threshold_shift": round(float((c3 == coll["THRESH_best"]).mean()), 4),` with `c3 = coll["C3_net"]` (`:533`), the `NET_SEED = 0` arm; `C3_net_s1/s2` are built at `:391` and never enter the `DEG` block, and no per-arm agreement against `THRESH_best` is emitted. **No per-seed or 3-seed DEG-A exists.** `0.9575` now appears only inside the finding's own description. The `0.95` line is correctly sourced.
- **R15 I-2 — DISCHARGED.** `eleven` appears twice in v15, both as the *claim*; v16 restates it accurately.
- **R15 I-3 — DISCHARGED.** `GATE0_REOPEN_2026-07-31.md:1050-1051` reads exactly as v16 quotes it; `0.8537` sits at `:1047-1048` inside a nested quote of `LITSWEEP3_ZH_SPECIFIC.md:36-37`. All three deviations named. `corpus was harvested` first entered at v3, so "carried since v3" is right.
- **R15 I-5 — DISCHARGED, verified by hash.** With the cross-reference normalised, `md5` of the two scope paragraphs is identical (`50fc261144c5f5b2d15af82e63e28836`).
- **R15 I-4 — the *defect* is gone; the *ledger's description of the repair* is false.** See I-1.

**Byte-identity of §8 and §9, re-verified independently.** `## 8. Gates` through the line before `## 10.`: v8 `a17b56954ee6955013327f82a03904f7`; v15 same; v16 same. **§8 and §9 are byte-identical between v8 and v16.**

**v16's scope claim, verified against the diff.** Section-level hashing: §§1, 2, 3, 4, 6, 7, 8, 9, 10, 11 **SAME**; §0, §5, §12 and head/trailer **DIFF**; §5's only hunk is the single §5.5 site. The six enumerated edit sites match exactly.

**Findings from R1–R14 that remain open in v16: none.**

---

## What I verified as sound

**The F88 transfer, re-derived from the primary tables, not from any review.** `ERRPAT_HateMM §1`'s final rows `28 / 26 / 26` (Σ 80, mean 26.6667); `§1.1`'s final row `187/1/2/25` ⇒ `(25, 2, 1)`, identity `= 80` ✓. `ERRPAT_MHC-ZH §2` gives `23/24/22` (Σ 69) and `(22, 0, 3)`, identity `= 69` ✓. The banked accuracies give integer error counts exactly (`744 × 0.8884408602150538 = 661.0`; `579 × 0.8929188255613126 = 517.0`). Scaling gives `|P_0| ≈ 79 / 60`, `n_unstable ≈ 9 / 8`; closure holds on both. Val-sel counterfactual `≈ 77` ✓.

**All twelve `k`-grid figures, re-derived**, together with the knife-edge arithmetic (`40/744 = 0.0538`, `37/744 = 0.0497`, `38/744 = 0.0511`, `30/579 = 0.0518`, `29/579 = 0.05009`, `28/579 = 0.0484`, ZH bar `28.95 ⇒ 29`), Hanley–McNeil, and the π* gaps.

**The knife edge, checked past the document.** I reconstructed the **exact** confusion matrices from `posrate_deployed` / `posrate_bank`: HateMM `(256, 42, 41, 405) ⇒ mF1 0.88378 = banked 0.8838`; MHC-ZH `(148, 32, 30, 369) ⇒ 0.87466 = banked 0.8747`. Flipping `|P_{τ_hi}|` gives `ΔmF1 ≈ +0.056` vs `Δacc 0.0538` and `≈ +0.061` vs `0.0518`. **The accuracy leg binds at `τ_hi` on both datasets** — §5.2's accuracy-only analysis is complete rather than partial.

**Statistical soundness and decidability.** `AUC_strat` fully pinned; both nulls exact for the nulls they name with honest caveats; one shared `π` per `(dataset, τ, draw)` over `P^{(τ)}`, item-level everywhere; Holm over `m = 2` correct; both families required is a pure additional hurdle; the dataset conjunction is a genuine IUT; `p`-floor `≪ α/2`; `IDENTIFIABILITY_UNDERPOWERED` fails `K-FELDMAN`, the conservative direction. §9 is two-valued and complete; **a KILL is fully reachable** and nothing is undecidable by construction.

**Legality / ban-scope.** `H-L1`–`H-L4` enforced structurally by `GATE-BLIND` and `GATE-LEDGER` (expected `36` matches the unconditional `np.savez` at `headspace_mint.py:322-324`); the `torch.load` guard and `VAL_RE` close every test path; CAL-2 leg (2) correctly omitted. Every registry ban quoted verbatim and adjudicated; **F113's `dead[]` entry has no `ban_scope` key** and closes *"STANDING RULE PROPOSED (not yet ruled)"*. The three texts that run **against** C09 are carried.

**Executability and budget.** `sha256(headspace_mint.py)` identical to `meta.mint_script_sha256` in all six banked arenas. `headspace_mint.py:288` forces `--device cpu`; `det1_assert` fires at `:187`; `CLI` admits exactly `{hatemm, zh}`, so the MHC-EN scope argument is structural. `sacct -j 13847` confirms 8 CPU / 32 G / no GPU / `00:29:49`. I recomputed the 36 banked mint durations from their own `meta.secs`: **min 33.2, max 60.0, median 41.85**, total 24.5 CPU-min against a budgeted ≤ 36. Every gate anchor re-read exactly; both data defects re-measured.

**Scope honesty.** **No C09 namespace exists anywhere on disk** — the trailer's claim is literally true. §10's withdrawals are all consistent with §§3.2, 5.1, 6.3 and 11.

---

## IMPORTANT

**I-1 — §12's ledger claims a repaired text that appears nowhere in v16, one row after repairing exactly that defect.** Row **I-4**'s repair column reads *"corrected in **both** copies to `"its nine HALT gates and its three reporting instruments"`."* That string occurs **zero** times in v16. What happened is that both sentences carrying `§8 (all ten gates)` were **deleted and rewritten** to describe the hashed span. The rewrite is a *better* repair than the one claimed and the defect is genuinely gone — but a reader auditing the ledger will search for the quoted corrected text and not find it, which is verbatim the failure round 15 raised as I-2 and which row I-2 immediately above now describes as a lesson.

**I-2 — §12's quotation of round 15's bottom line is not verbatim; the trailing clause is spliced in from round 14's review.** §12 attributes to round 15 *"…verdict or **the scope of any conclusion**."*; `C09_A0_V15_PREREG_REVIEW.md` reads *"…verdict or **scope**."* The phrase `scope of any conclusion` appears **0 times** in round 15's review and **1 time** in round 14's. v15's §12 quoted round 14 accurately; v16 swapped the counts and the round number and carried round 14's trailing words into round 15's mouth. The ellipsis joining round 15's two separated sentences is legitimate; the five-word expansion is not. New in v16, and the same species as R15 I-3.

**I-3 — the STATUS block points a reader at "the v13 ledger" for a §12 that is now v16's ledger.** *"Each carries its own ledger; **the v13 ledger is §12**."* §12's own heading is *"the 5 round-15 findings"*. The sentence names **this** document's ledger and froze at `v13` when v14 forked; it has been false in v14, v15 and v16, and no round has caught it.

---

## Checked and deliberately not counted

Band B's quantifier (§9 is two-valued and complete — I re-derived exhaustiveness myself); §5.6's *"30-epoch Adam"* against `src/run_rac.py:684`'s `torch.optim.AdamW`; §2's *"~25–60 s"* against a measured minimum of `33.2`; the budget summing to ≈116 against ≈115; `ITEM-STRATUM` occupancy not emitted the way the row strata are (conservative); the `(τ, k)`-cell wording; §3's rendering of `LITSWEEP3:91` (inherited verbatim, running against C09); `dead[Fxx]` not being a literally resolvable path; `ksweep_OUT.json`'s `dev` sub-key being `null` on exactly the four `MHC_EN_ARM-V/*/valsel` curves; §6.1's *"`:644-647` … headed"* where the heading is at `:625`; §12's report of round 15's approximate `mF1 0.8831` (my exact reconstruction lands on `0.8838` and confirms the conclusion); typographic drift inside italic quotes.

---

## Bottom line

**The science is finished and I could not move it — for the eighth round running.** The instrument, the arena, the fold contract, the legality spine, the label-use adjudications, the two nulls, the Holm/IUT structure, the power rule, the closed-form caps, the executability, the three sha256 pins and the budget all hold under independent re-derivation, and **§8 and §9 are byte-identical to v8 by hash on my own recomputation**. Round 15's five findings are discharged in substance — including the one that mattered most, the non-existent `0.9575`, which I verified against `aggnet_pregate.py` line by line. **A KILL remains fully available and is still the honest expectation.**

**Does any remaining finding change a verdict, a rule, a threshold, a gate, an operating point, or the scope of any conclusion? No — none of the three does.** All three are statements the document makes about its own repair history: one claims a corrected string that does not exist, one misquotes the review it is answering, one points at the wrong ledger version. §§1–11's governing text is untouched by all three, and for the fourth round running the blockers sit entirely outside it.

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not quite.** Two of the three are *new in v16*, and both reproduce — inside the very ledger that names them as lessons — the two defect species round 15 raised. A preregistration whose audit trail is its ledger should not carry a false description of its own repairs across a hash freeze. They are three one-line text edits, all in §0/§12.

With those made, **v17 is ready to freeze and submit as one 8-CPU / 32-GB / no-GPU / no-`--time` job of ≈116 CPU-minutes** — subject to the three preconditions the STATUS block names, the immediately-prior `squeue` empty-check, sha256 re-verification and namespace-absence check (all four namespaces confirmed absent today), and the separate step this document is honest about but which is not yet done: **the analysis script and the sbatch driver do not exist yet and will need their own implementation and code review before anything is submitted.**
