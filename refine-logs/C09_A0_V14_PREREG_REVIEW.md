# C09 Stage-0 (A0) v14 — Independent Design Review, Round 14

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V14_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 3 Important`

---

## Prior-round audit

I diffed v13 → v14 (21 raw hunks over ten edit sites), re-opened every source on disk, and re-derived §4.2's transfer table and every downstream figure myself.

**v14's scope claim — the load-bearing half is exact.** The enumeration `§§1, 2, 3, 4.1, 4.3, 4.4, 5.1, 5.4–5.6, 6.3, 7, 8, 9, 10 and 11 are byte-identical to v13` is **verified correct and complete**. The *count* is not (I-3).

**"No decision rule … has changed since v8" — verified directly.** I diffed v8 → v14: **§8 (all ten gates) and §9 (the whole decision rule) are byte-identical to v8.** Every v8→v14 hunk elsewhere is prose or a transferred expectation.

**Round 13's five findings.**
- **I-1 — DISCHARGED in full, verified at source.** `ERRPAT_HateMM §1`'s cell table gives final rows `28 / 26 / 26`, Σ = 80, mean 26.6667 ✓; §1.1's final row `187/1/2/25` sums to 215 ✓; identity `3(25)+2(2)+1 = 80` ✓. `ERRPAT_MHC-ZH §2` is headed exactly *"ERROR INVENTORY (Tier 2, final-epoch protocol)"* ✓; consensus `22/0/3/124`, union 25, sum 149 ✓. Val-sel `(24, 2, 3)`, per-seed `26/27/26`, Σ = 79, mean 26.3333, `24/26.3333 × 84.3333 = 76.86 ⇒ ≈ 77` ✓, in the block's own order ✓.
- **I-2 — DISCHARGED for three of four.** §5.2, §5.3 (`np.round(118.5) = 118` vs `R(118.5) = 119` confirmed in the live env) and §6.2 all re-derive. §0's history bullet had its numerator re-based and its derived figure left behind → I-1.
- **I-3 — DISCHARGED on substance, new locator defect.** `:623`, `:625`, `:643`, `:644-647`, `:784`, `:902` all verify. `:786-790` does not → I-2.
- **I-4 — DISCHARGED in full.** `37/744 = 0.049731`, `38/744 = 0.051075`, ZH bar `28.95 ⇒ ≥ 29`, `28/579 = 0.048359`, `29/579 = 0.050086`. All four re-derive; the reach/power attribution is now correct.
- **I-5 — DISCHARGED on convergence, not on count.** The two scope paragraphs are byte-identical apart from the intentional cross-reference (diffed programmatically). The count is wrong → I-3.

**Round 12's H-1 and every Critical and High from R1–R11 still land in v14.**

---

## What I verified as sound

**The transfer, re-derived from scratch.** Rates `0.9375`, `0.95652`, `0.1125`, `0.13043`; arena per-seed errors `83/85/85` and `62/64/61`, means `84.3333` / `62.3333`; scaling `79.0625 / 59.6232` and `9.4875 / 8.1304`; closure exact on both datasets; `|P_{τ_hi}| ≈ 40 / 30`.

**Every downstream figure.** Caps `131.4667` / `95.6667`; `R(2|P_0|) = 158/120` dead; `R(1.5 × 79) = 119`; `78.1 ⇒ 0.9261 / 0.6563`; `59.5 ⇒ 0.9545 / 0.6611`; at `k = |P_0|`, `0.6889/0.7354` and `0.7139/0.7417`; all six τ_hi cells inside the caps at `π* = 0.965/0.810/0.7325` and `0.9833/0.8222/0.7417`. **All twelve reproduce.** π* gaps `9.31` / `7.45`; `[28.95, 29.0)` empty of multiples of 1/3; Hanley–McNeil `0.0527` / `0.0913`; §12's `0.1062 / 0.1036` and `1/79 = 1.27 %`.

**Statistical soundness and decidability.** `AUC_strat` fully pinned; both nulls exact for the nulls they name with the weaker-independence caveat carried; one shared `π` is conservative in both places; Holm + IUT level-controlled; `p`-floor `≪ α/2`. **Bands A′/A/B/C are exhaustive and mutually exclusive on completed runs and map exactly onto §9's CONTINUE.** At least one `LIVE_ON_NET` cell exists per `(dataset, τ)` and no identifiability cell is arithmetically dead at the expected sizes — nothing is undecidable by construction. `IDENTIFIABILITY_UNDERPOWERED` cannot appear on a CONTINUE.

**Legality and ban-scope.** Every `ban_scope` quotation verifies character-exact. **F113's `dead[]` entry carries keys `["name","status"]` only.** The three texts that run **against** C09 are carried anyway.

**Executability, instrument, budget.** All three sha256 match on disk. `frozen_artifact_policy` names four modules and **not** the mint — §2's distinction is right. Every cited line lands, including `VAL_RE` (Test_Retrieval lines cannot be parsed, so `GATE-LEDGER`'s six trainlog reads are safe). `GATE-FLOOR`'s 12 anchors, the fold arrays, `raw_deployed_acc`, the fidelity anchors, `C02_A0_OUT.json`'s `ARENA2` and `EMPTY_TEXT = 39`, `aggnet_pregate.py:534`, DEG-A `0.9570 / 0.9508`, and ZH seed-0's `B_agree_fixk["15"] = 1.0` all re-read exactly. `sacct` confirms job `13847`. `squeue` empty. `default_rng(20260801).spawn(6)` works. Budget sums to ≈116 against ≈115.

**Re-measured this session, independently:** `243/579`, `0.5802 / 0.1161 / 0.3109` (5.0×), `39/744` and `0` on ZH, the majority rates and both `GATE-ARENA` bands, and `GATE-NULL` on both caches.

**Label-use discipline holds.** No test path is opened; no test-derived quantity is materialised anywhere, including CAL-2(2)'s omitted comparator. The one residual bank channel is correctly identified as cancelling in the paired `ΔAUC`. The 36 dev-label materialisations are declared and fenced.

---

## IMPORTANT

**I-1 — §0's `K-DEG` bullet was half-repaired: the numerator was re-based on `79`, the figure it derives was not.** At `k = 79`, `2k/n = 0.212366` and `ov = 0.95` gives `0.98938 ⇒ 0.9894`. `0.9898` is the retired-`76` value. §6.2 states this correctly **and explicitly labels `0.9898` as what "the retired `76`" gives** — so §0 now derives from `79` a number §6.2 attributes to `76`. No rule reads it; §0's own header says *"Nothing in this subsection is a rule."*

**I-2 — the I-3 repair cites the MECHFIX cross-check to lines that do not contain it, in three places.** `:786-790` is the ZH degeneracy block. The MECHFIX paragraph — the `T1/T2a/T2b/T4` test reads, this arena's `−0.0006 / +0.0000 / −0.0040 / −0.0063`, and the *"sign-unstable across the two head-space arenas"* conclusion — is at **`:791-796`**. The quoted text and every number are exact; only the locator is wrong, at §0, §6.1 and §12. **Note the provenance: round 13's own review supplied `:786-790`, and the repair adopted the reviewer's locator without re-opening the file — which is precisely how R13 I-3 arose.** `:784` and `:902` are correct.

**I-3 — the unified scope statement miscounts its own diff.** Both copies read *"**Nine hunks:** …"* and then enumerate **ten** items. It also omits an eleventh edit site: the document trailer (`*v13.` → `*v14.`), the counterpart of "the title". R13 I-5's substantive requirement — one convention, byte-identical in both places — **is met**; what regressed is the arithmetic in the sentence that reports the count.

---

## Checked and deliberately not counted

§4.3's `≈ 1.4 %` (now `1/79 = 1.27 %`, disclosed); §2's `~25–60 s` and the ≈115-vs-116 rounding; §5.6's *"30-epoch Adam"* vs `run_rac.py:684`'s `AdamW`; `ITEM-STRATUM` occupancy not emitted the way the row strata are; `queued.screening_arena_switch`; the `(τ, k)`-cell wording; §6.2's DEG-A `0.9570` being the HateMM **seed-0** cell (3-seed mean `0.9575`, a headwind either way); §3's rendering of `LITSWEEP3:91`, a compression inherited verbatim from the registry and running against C09; §9's raw-arena leg with CAL-3 discharged by a comparator of record, correctly labelled.

---

## Bottom line

**The science is finished and I could not move it — for the sixth round running.** The instrument, the arena, the legality spine, the label-use discipline, the executability, the sha256 pins, the budget and the whole inferential apparatus hold under independent re-derivation from the primary artifacts. Round 12's High and all five of round 13's Importants are discharged on substance. **A KILL remains fully available and is still the honest expectation**, most likely on the τ_hi power leg on MHC-ZH, with τ_0 as the branch where identifiability is actually adjudicated.

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not yet — but only barely not, and for the third round running the blockers are entirely outside §§1–11's governing text.** All three findings are one-line text edits: one stale derived figure in a subsection the document itself declares non-governing (`0.9898 → 0.9894`), one line-range locator wrong by five lines in three places (`:786-790 → :791-796`), and one count wrong by one in a sentence that reports the diff. None touches a rule, threshold, feature set, null, gate, operating point, verdict or scope of any conclusion.

With those three edits made, **v15 is ready to freeze and submit as one 8-CPU / 32-GB / no-GPU / no-`--time` job of ≈116 CPU-minutes**, subject to the three submission preconditions the STATUS block names and the immediately-prior `squeue` empty-check, sha256 re-verification of the frozen set, and namespace-absence check.
