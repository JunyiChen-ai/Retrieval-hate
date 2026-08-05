# C09 Stage-0 (A0) v12 — Independent Design Review, Round 12

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V12_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 1 High / 2 Important`

---

## Prior-round audit

I diffed v11 → v12 (15 hunks) and re-verified R1–R11 against **v12 itself**, re-opening every source on disk.

**v12's scope claim checks out on its load-bearing half.** `§§1, 2, 3, 4, 5.1–5.2, 5.4–5.6, 6.1, 6.2, 7, 8, 9 and 11 are byte-identical to v11` — verified hunk-by-hunk, **exactly right for the first time in five rounds**. §0 and §12 carry the identical scope paragraph. No rule, threshold, feature set, null, gate or arithmetic moved.

**Round 11 (0C/0H/2I).**
- **I-2 — DISCHARGED.** Stated once, in one convention, in both places, with a correct and checkable byte-identical enumeration.
- **I-1 — the repair is internally valid but corrects the wrong term.** The identity, the intervals and the ZH `a = 0` step all re-derive exactly as written. But the input it anchors on (`|P_0| ≈ 76 / 55`) is itself the mis-derived quantity, and R11's `7–9` was approximately right. See HIGH.

**Rounds 1–10 — every Critical and every High still lands in v12**, including R4 C-1, R4 H-1/H-3/H-4, R5 H-1/H-2/H-3, R6 H-1/H-2, R7 H-1, R8/R9/R10's items. R10's five below-the-line fixes survive unchanged.

---

## What I verified as sound

**Every number re-derived from source**, including all twelve floors, per-seed errors `83/85/85` and `62/64/61`, `Σ = 253 / 187`, the caps from the exact means, `R(1.5×55) = 83`, the `π*` figures, the `[28.95, 29.0)` emptiness, Hanley–McNeil re-derived from `Q1`/`Q2`, the `pred_agree` mapping, **F98's DEG-A re-read from `aggnet_main_*_OUT.json` = `0.9570 / 0.9508`**, F113's `THRESH_best +0.0041`, `n_test 215/149/161`, the fidelity anchors, both data-defect measurements, `GATE-NULL`, `sacct` for job `13847`, and the 36 banked mint times.

**Quotations.** Every registry field, every `ban_scope`, every `.md` citation verifies character-exact modulo dash/quote normalisation. **F113's `dead[]` entry is the only one of 76 with keys `["name","status"]` only** — no `ban_scope`. The disowned renderings are all correctly attributed to the rendering source.

**Executability: nothing becomes an engineering HALT.** Both of the mint's run-time sha256 assertions match on disk. `--dataset` choices are exactly `['hatemm','zh']`. `best_epoch_path` is reloaded only inside the EM branch which `--seg_mode full` never enters, so **the keys are final-epoch and there is no dev-based model selection anywhere**. `GATE-FIXK20` corroborated in all six banked arenas and independently in F98's three. Live environment byte-identical to the banked `meta.runtime`. `squeue` empty.

**Statistics and decidability.** `AUC_strat` fully pinned. `PERM-STRUCT` exact for the marginal null (`A^{(f)}`, `P^{(τ)}` and the frozen strata are all measurable w.r.t. `(target, BASE)`, so evaluation sets are draw-invariant). `PERM-STRUCT-COND` exact for the joint form. Both families = intersection = conservative; Holm correct; IUT correct. Macro-F1 monotonicity re-derived independently. **No single rule can carry a CONTINUE**; a KILL is fully available at every branch.

**Residual gold-label paths: none beyond the one §4.3 names.** All 13 features walked. No test-like cache reachable.

**Legality holds in both directions.** F51, F66 and F99 all run against C09's own claim or successor and are carried anyway. I enumerated both constraint arrays independently.

---

## HIGH

**H-1 — v12's headline repair corrects the wrong term: the error was never `n_unstable ≈ 7–9`, it is `|P_0| ≈ 76 / 55`, and the "corrected" figures are now wrong on both datasets.**

`|P_0|` and `n_unstable` are two ratios of the *same* F88 seed-consensus table and must be denominated on the only quantity measured in the C09 arena, the per-seed error count. Transferred that way from the primary tables (both **final-epoch**, the protocol C09's mint produces):

| | `ERRPAT_HateMM §1.1` (final) | `ERRPAT_MHC-ZH §1` |
|---|---|---|
| per-seed errors | `26/27/27` ⇒ `Σ = 80`, mean `26.667` | `23/24/22` ⇒ `Σ = 69`, mean `23` |
| `3/3` (`\|P_0\|`) | `25` | `22` |
| `2/3` (`a`) / `1/3` (`b`) | `2 / 1` ⇒ `n_unstable = 3` | `0 / 3` ⇒ `n_unstable = 3` |
| `\|P_0\|` / per-seed mean | `0.9375` | `0.9565` |
| scale to C09 (`84.333` / `62.333`) | `×3.1625` | `×2.7101` |
| **⇒ `\|P_0\|`** | **`79.1`** | **`59.6`** |
| **⇒ `n_unstable`** | **`9.5`** | **`8.1`** |

Both reproduce `Σ_s|E_s|` exactly: `3(79.06)+15.81 = 253.0` and `3(59.62)+8.13 = 187.0`.

So **`n_unstable ≈ 7–9` was right to within rounding, and `|P_0| ≈ 76 / 55` is the mis-derived term.** The ZH origin is diagnosable to the character: `55 = 0.88 × 62.333`, where `0.88 = 22/25` is F88's **union**-denominated rate applied to a **per-seed** base; the per-seed rate is `22/23 = 0.9565 ⇒ 59.6`. HateMM's `76 ≈ 0.90 × 84.333` takes the low end of a `"≈89-93%"` band against a fuzzy `"~26-28"`; the protocol-matched rate is `25/26.667 = 0.9375 ⇒ 79.1`. The two datasets were computed by *different* conventions, which is exactly why the incompatibility exists.

R11 correctly proved the two could not coexist, then resolved it by keeping the wrong term. v12 has written that into four sections: **§6.3** (`n_unstable ∈ [13,25]/[11,22]`, and *"on MHC-ZH … `n_unstable = 22`, above the `20` trigger"* — the coherent figures are `9.5 / 8.1`, both **under**, i.e. v11's retracted *"expected on both datasets"* was correct); **§5.3**; **§10**; **§12**. The ZH combination implies a 3/3-stability fraction of `55/77 = 71 %` against the `88 %` F88 states; HateMM's implies `75–85 %` against `89–93 %`.

And the same `|P_0|` propagates through `|P_{τ_hi}| ≈ |P_0|/2`: **`38 / 28` becomes `≈ 40 / 30`**, inverting four further pre-declarations v12 left untouched — §5.2's *"On MHC-ZH `28 < 30`, so the tag fires by arithmetic … the `τ_hi` branch cannot produce a CONTINUE at all"* (at `30` the cell is LIVE); §5.2's `K-REACH`-at-`τ_hi` arithmetic; §5.2's *"`|P_0| ≳ 60` against a declared expectation of `≈ 55`"* (the corrected expectation *is* `≈ 60`); §5.3's *"strictly impossible on MHC-ZH whenever `|P_{τ_hi}| < 29`"*; and §10's *"expected to be closed on power on MHC-ZH"*.

**Why High and not Critical.** No decision rule reads an F88 number; every cap, tag and trigger is computed in-run. **Why High and not Important.** These are the design's pre-registered statement of *what it can adjudicate*, on one of exactly two datasets in a design §10 calls "zero slack", and §9 requires the KILL record to be scoped by them. Eleven rounds could not check it because **the document nowhere shows the derivation of `|P_0| ≈ 76 / 55`**.

**Repair (text only).** Transfer both ratios per-seed-denominated from the primary tables, show the arithmetic, and restate: `|P_0| ≈ 79 / 60`, `|P_{τ_hi}| ≈ 40 / 30`, `n_unstable ≈ 9 / 8`. Then re-word §6.3, §5.3's out-of-support figure, §10's bullet, and the four `τ_hi` sentences that turned on `28 < 30`. Keep §10's stronger new framing — *"decided by the realised tag, not in advance"* — which is right under either transfer.

---

## IMPORTANT

**I-1 — §6.1's *"the count is the source's four"* equivocates between two different fours; the source's four does not contain the nine.** `:623`'s rung column assigns **rung 1 = VSW, rung 2 = F95, rung 3 = FIXK, rung 4 = F89** (`:644-647`), with `THRESH_best` deliberately **un-runged** (`:643`). The nine raw positives occupy rungs 1, 2, 3 and the un-runged arm — **no rung-4 arm at all** (F89's RAW column is `—`). So the document's four (`VSW/F95/FIXK/THRESH_best`) and the heading's four (`VSW/F95/FIXK/F89`) are *different sets of the same cardinality*, and the appeal to `:623` does not license the count. The substantive claim (*"not one lineage"*) stands; the sourcing sentence fails. Worth noting alongside: **rung 4 is the four closed-form geometric head-key transforms — the family nearest to C09's own structural block — and it reads `+0.0000 / −0.0004 / +0.0027 / +0.0054`, i.e. null-to-sub-noise with a cross-dataset sign flip.** That is adverse to C09 and is carried nowhere.

**I-2 — `F114` is cited three times with no locator, and the bare number resolves to a different finding in the repository's own findings file.** The finding meant is the CLIP-LOO erratum at `autoresearch/goal_mllm_plus3/state/findings.jsonl:115`. But `TARGET_FINDINGS.md:79` is headed **`### F114 — the v6 teacher producer could never have produced a teacher response`** — an unrelated C04-lineage finding. Every other finding the document leans on is given with a path. The three numbers themselves verify. **Repair:** one locator at first use.

---

## Checked and deliberately not counted

"three sentences of §6.3" understating the *result* by ~3× while being true of the edit's footprint; `GATE-LEDGER`'s trainlog reads being of mixed files whose safety rests on the hard `VAL_RE` filter (never named, but no test label is materialised and `GATE-DEVFID` gates nothing); §5.6's "30-epoch Adam" where `run_rac.py:684` uses `AdamW`; `load_split` calling `_ORIG_TORCH_LOAD` (safe because both call sites are literals); `ITEM-STRATUM` occupancy not emitted the way the row strata are (direction conservative); the ellipsis omissions that all run **against** C09 (F98's head clause and (c), F97's *"DO NOT PROMOTE…"*, `banned_constraints[9]`'s parenthetical, ERRPAT's train-LOO recalibration); `queued.screening_arena_switch`; §2's budget and `ksweep` parenthetical; §4.3's `≈ 1.4 %`; §2's "~25–60 s" against a measured `33.2`.

---

## Bottom line

The instrument, the legality spine, the executability, the budget and the inference are all intact, and I could not move any of them. Every gate is corroborated on disk, both of the mint's run-time hashes match, the runtime is byte-identical, no test path is reachable, and the four decision rules remain jointly decidable with none able to carry a CONTINUE alone. A KILL remains fully available and is still the honest expectation.

But **v12's one substantive repair is aimed one term to the left.** The pre-declared pair `(|P_0|, n_unstable)` has to be transferred jointly, per-seed-denominated; done that way it gives `(79, 9.5)` and `(60, 8.1)` and reproduces `Σ_s|E_s| = 253 / 187` exactly. R11 proved the pair could not coexist and chose to keep `76/55`; the arithmetic says the opposite. v12 propagated that choice into §5.3, §6.3, §10 and §12, and left four further `τ_hi` pre-declarations standing on `≈ 28` when the corrected figure is `≈ 30` — the exact value at which "arithmetically dead" flips.

**Is this ready to hash-freeze and submit? No — not because of the science, which is finished, but because the round-12 repair itself needs re-doing.** The fix is arithmetic on paper: four lines of derivation shown once, `|P_0| ≈ 79 / 60` substituted, and seven sentences restated. It needs no re-measurement, no GPU, and no change to any rule, threshold, feature set, null, gate or operating point. With those made — and with the derivation *shown*, so that a thirteenth reviewer can check in two minutes what eleven could not check at all — the frozen set is ready.
