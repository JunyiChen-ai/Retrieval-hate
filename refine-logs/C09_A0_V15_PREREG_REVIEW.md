# C09 Stage-0 (A0) v15 — Independent Design Review, Round 15

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V15_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 5 Important`

---

## Prior-round audit

I diffed v14 → v15 (7 raw hunks), re-opened every source on disk, and re-derived the arithmetic myself.

**Round 14's three findings — all three DISCHARGED, verified at source.**
- **R14 I-1.** `2×79/744 = 0.2123655914`; `1 − 0.2123655914 × 0.05 = 0.9893817 ⇒ 0.9894`. The retired `76` gives `0.9897849 ⇒ 0.9898`. §0 now reads `0.9894`; §6.2 already carried it and labelled `0.9898` as the retired-`76` value.
- **R14 I-2.** `:786-790` is the ZH degeneracy block; the MECHFIX paragraph begins at **`:791`** and ends at `:796`, carrying `T1 +0.0000, T2a +0.0000, T2b +0.0000, T4 +0.0067`, this arena's `−0.0006 / +0.0000 / −0.0040 / −0.0063`, and *"sign-unstable across the two head-space arenas"* — all quoted exactly. Corrected in all three places; the surviving `:786-790` strings are the description of the defect. The provenance claim checks out: `C09_A0_V13_PREREG_REVIEW.md:52` is where it originated.
- **R14 I-3.** v15 says **six edit sites** and enumerates six, including the trailer. Against the diff, six is correct.

**Round 14's byte-identity claim, re-verified independently.** I hashed `## 8. Gates` through the line before `## 10.` in v8 and in v15: **`md5 a17b56954ee6955013327f82a03904f7` on both.** §8 and §9 are byte-identical between v8 and v15.

**v15's scope claim.** *"§§1–11 are byte-identical to v14 except for that single §6.1 locator"* — **verified true**.

**Findings from R1–R13 that remain open in v15: none.**

---

## What I verified as sound

**The F88 transfer, re-derived from the primary tables.** `ERRPAT_HateMM §1`'s final rows `28 / 26 / 26` (Σ 80, mean 26.6667); §1.1's final row `187/1/2/25` ⇒ `(25, 2, 1)`, identity `= 80` ✓. `ERRPAT_MHC-ZH §2`, headed exactly *"ERROR INVENTORY (Tier 2, final-epoch protocol)"*, gives `23/24/22` (Σ 69) with `(22, 0, 3)`, identity `= 69` ✓. Arena per-seed errors re-derive from the **unrounded** banked accuracies (`744 × 0.8884408602150538 = 661.0`). Scaling gives `|P_0| ≈ 79 / 60`, `n_unstable ≈ 9 / 8`, `|P_{τ_hi}| ≈ 40 / 30`, closing exactly. Val-sel counterfactual `≈ 77` ✓.

**Every k-grid figure.** Caps `131.4667` / `95.6667`; `R(2|P_0|) = 158/120` dead; `R(1.5 × 79) = 119` (and `np.round(118.5) = 118`, so the half-up pin is load-bearing); `78.1 ⇒ 0.9261/0.6563`; `59.5 ⇒ 0.9545/0.6611`; `0.6889/0.7354` and `0.7139/0.7417`; all six `τ_hi` cells inside the caps at `π* = 0.965/0.810/0.7325` and `0.9833/0.8222/0.7417`. **All twelve reproduce.**

**The knife edge, checked beyond the document.** §5.2 analyses the `τ_hi` reach leg on accuracy only, while `K-REACH` is a conjunction with `ΔmF1 ≥ +0.050`. I reconstructed both confusion matrices from the banked floors and the ERRPAT FP/FN splits: HateMM `(251, 47, 36, 410) ⇒ mF1 0.8831` against banked `0.8838`; MHC-ZH `(148, 32, 30, 369) ⇒ 0.8747` exactly. Flipping `|P_{τ_hi}|` gives `ΔmF1 ≈ +0.0565` (vs `Δacc 0.0538`) and `≈ +0.0606` (vs `0.0518`). **The accuracy leg binds on both datasets**, so §5.2's acc-only knife edge is correct rather than incomplete.

**Statistical soundness and decidability.** `AUC_strat` fully pinned; both nulls exact for the nulls they name with the caveats carried; one shared `π` conservative in both places; Holm over `m = 2` correct; requiring both families is a pure additional hurdle; the dataset conjunction is a genuine IUT; `p`-floor `≪ α/2`. **Bands A′/A/B/C exhaustive over completed runs.** At `τ_0`, `k = 2|P_0|` dead on both and the other two live on both; at `τ_hi` all three live — **nothing is undecidable by construction, and a KILL is fully reachable.**

**Legality.** `GATE-BLIND` enforces `H-L1`–`H-L4` structurally; `GATE-LEDGER`'s expected `36` matches the unconditional `np.savez`; the `torch.load` guard and `headspace_fidelity.py`'s `VAL_RE` filter close every test path; CAL-2 leg (2) is correctly omitted. The three texts that run against C09 are carried rather than dropped.

**Executability and budget.** `sha256(headspace_mint.py) = cefdf8dc…` identical to `meta.mint_script_sha256` in all six banked arenas. Every cited line lands. `sacct -j 13847` confirms 8 CPU / 0 GPU / 32 G / `00:29:49`; the 36 mints re-derive to min `33.2`, max `60.0`, median `41.85`. Budget sums to ≈116 against ≈115.

**Every other number** re-measured this session and exact, including `GATE-NULL` on both named caches and the two data-defect counts.

---

## IMPORTANT

**I-1 — §12 asserts a measured quantity (`0.9575`) that exists in no artifact, transcribed from the round-14 review without re-opening the source.** There is no 3-seed DEG-A. `aggnet_pregate.py:534` computes it once — `(c3 == coll["THRESH_best"]).mean()` — from the `NET_SEED = 0` primary arm; the stability arms `C3_net_s1 / C3_net_s2` **never enter the `DEG` block**, and their per-item prediction vectors are not banked, so the figure is not recomputable. Exactly one `A_agree_threshold_shift` exists per `(dataset, space)`; HateMM/fused `= 0.957`. The string `0.9575` appears nowhere in the repository except round 14's review and this document. Two claims are unsupported: that `0.9570` is a *seed-0 cell* of a per-seed family, and that a 3-seed mean of `0.9575` exists. Nothing moves — `K-DEG` reads realised `pred_agree`, and the `0.95` line is separately sourced to `RESTRANS_PREGATE_RECORD.md:409` — but §0 states the repository's rule in this document's own voice (*"re-read the source, never transcribe a locator from a review"*), and v15 then transcribed a **number** from the same review into the same paragraph type. **Strike or re-source.**

**I-2 — the ledger claims a repair whose text does not exist in v15.** §0 and §12 both say v14's scope paragraph was *"corrected to eleven edit sites, in both copies"*. That paragraph does not survive into v15 — it was deleted and replaced by v15's own six-site paragraph. There is no copy to have corrected, and nothing in v15 states eleven. A reader auditing the ledger will search for the repaired text and not find it.

**I-3 — §5.5 presents a composite paraphrase as a verbatim quotation from the reopen.** §5.5 attributes, in quotation marks: *"so part of the reported ZH 0.8537 floor rests on how the corpus was harvested rather than on video content."* `GATE0_REOPEN_2026-07-31.md:1050-1051` reads: *"Part of the reported ZH floor rests on a marker of how the corpus was collected rather than on video content"*. Three deviations: `0.8537` is **spliced in from `:1047-1048`**; `a marker of` is dropped — the source's own hedge; `collected` becomes `harvested`. No locator is given, which is why it has survived since v3. Every surrounding number is exact and the distortion runs *against* C09, but it is a sentence in quotation marks the cited record does not contain. **Note:** this is the only one of the five inside §§1–11, so fixing it retires v15's *"§§1–11 are byte-identical to v14 except…"* claim.

**I-4 — "§8 (all ten gates)" contradicts §8.1's own count, in both new copies.** §8 names **twelve** `GATE-*` objects: nine in §8.1 and three in §8.2. §8.1's own closing sentence is careful about exactly this — *"Nine HALT gates, and exactly one further HALT condition elsewhere… These ten conditions are the complete publication precondition of §9"* — and §9 reads *"all nine HALT gates of §8.1 … and the `SHUFFLE-POP` band."* The underlying claim is **true** (verified by hash), so this is a gloss defect, but it is a statement about the document's own contents that its own §8.1 refutes, and it is new in v15.

**I-5 — the two copies of the scope paragraph are not verbatim identical, against the paragraph's own claim.** v14's two copies differed only by the intentional cross-reference; v15's differ by that **and** by the phrase `this section, `. Semantically inert, but R11 I-2 and R13 I-5 established divergence between these copies as a reviewable convention and R14 verified byte-identity programmatically as the discharge.

---

## Checked and deliberately not counted

Band B's quantifier not formally excluding Band C in one corner (§9 is two-valued and unambiguous); §5.2's acc-only knife edge (verified correct above); §5.6's *"30-epoch Adam"* against `AdamW`; §2's *"~25–60 s"* and the ≈115-vs-116 rounding; `ITEM-STRATUM` occupancy not emitted the way the row strata are (direction conservative — `K-FELDMAN` requires both families, so the conditional null can only remove CONTINUEs); the `(τ, k)`-cell wording; §3's rendering of `LITSWEEP3:91` (inherited from the registry, running against C09); `dead[Fxx]` not being a literally resolvable path (repository convention); `ksweep_OUT.json`'s `dev` sub-key being `null` on the four `MHC_EN_ARM-V/*/valsel` curves; minor typographic drift inside italic quotes.

---

## Bottom line

**The science is finished and I could not move it — for the seventh round running.** The instrument, the arena, the legality spine, the label-use discipline, the executability, the sha256 pins, the budget and the whole inferential apparatus hold under independent re-derivation, and §8 and §9 are byte-identical to v8 by hash. Round 14's three findings are genuinely discharged at source. **A KILL remains fully available and is still the honest expectation.**

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not yet.** All five findings are one-line text edits and none touches a rule, threshold, feature set, null, gate, operating point, verdict or scope. But two are new in v15, and I-1 is a number asserted as measured that exists in no artifact — not a defect a preregistration should carry across a hash freeze, and the exact failure mode §0's own repair note names as forbidden. Four sit in §0/§12; I-3 sits in §5.5 and therefore requires v16's scope statement to be restated rather than carried.

With those five edits made, **v16 is ready to freeze and submit as one 8-CPU / 32-GB / no-GPU / no-`--time` job of ≈116 CPU-minutes**, subject to the three preconditions the STATUS block names plus the immediately-prior `squeue` empty-check, sha256 re-verification and namespace-absence check.
