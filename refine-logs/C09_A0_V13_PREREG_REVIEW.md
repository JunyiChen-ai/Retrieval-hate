# C09 Stage-0 (A0) v13 — Independent Design Review, Round 13

**Reviewer.** Fresh independent worker, no exposure to the author's repair reasoning.
**Target.** `refine-logs/C09_A0_V13_RECORD.md`
**Verdict.** `REVISE — 0 Critical / 0 High / 5 Important`

---

## Prior-round audit

I diffed v12 → v13 (nine content hunks plus title/STATUS) and re-verified R1–R12 against **v13 itself**, re-opening every source on disk.

**v13's scope claim: the load-bearing half is exact.** The enumeration `§§1, 2, 3.2–3.4, 4.1, 4.3, 4.4, 5.1, 5.4–5.6, 6.2, 7, 8, 9 and 11 are byte-identical to v12` is **verified hunk-by-hunk and correct**. No decision rule, threshold, feature set, null, gate or operating-point definition moved. The prose *describing* the edits does not check out; see I-5.

**Round 12 (0C/1H/2I).**
- **H-1 — DISCHARGED on substance, with residue.** The joint per-seed-denominated transfer is derived in §4.2, and I re-derived every step independently. `|P_0| ≈ 79 / 60`, `n_unstable ≈ 9 / 8`, `|P_{τ_hi}| ≈ 40 / 30` are **correct**. §5.2's knife-edge, §5.3's whole `k` grid, §6.3's UNSTABLE-POP expectation and §10's stability bullet all re-derive exactly. Residue: the derivation block mis-transcribes and mis-locates its own primary source (I-1); one knife-edge leg is misattributed (I-4); four statements elsewhere still stand on the retired `76 / 55` (I-2).
- **I-1 — DISCHARGED with a new defect.** The two-fours equivocation is retired and rung 4's adverse numbers are carried; the new paragraph mis-describes its source (I-3).
- **I-2 — DISCHARGED.** `findings.jsonl:115` is the CLIP-LOO erratum; `TARGET_FINDINGS.md:79` is the unrelated C04-lineage `F114`. The locator sits at the genuine first use.

**Rounds 1–11 — every Critical and every High still lands in v13.** One discharged Important has regressed: **R11 I-2** (I-5).

---

## What I verified as sound

**The central derivation — re-derived from scratch, and it is right.** `ERRPAT_HateMM §1.1` final row: `187 / 1 / 2 / 25`, union `28`, sums to `n_test = 215` ✓; identity `3(25)+2(2)+1 = 80` ✓. `ERRPAT_MHC-ZH` consensus: `22 / 0 / 3 / 124`, union `25`, sums to `149` ✓; identity `3(22)+0+3 = 69` ✓. Means `26.6667` / `23` ✓; rates `0.9375`, `0.95652`, `0.1125`, `0.13043` ✓. Arena per-seed errors `83/85/85` and `62/64/61`, means `84.3333` / `62.3333` ✓ — floors re-read from all six arena JSONs and corroborated by `C02_A0_OUT.json`. Scaling: `79.0625 / 59.6232`; `9.4875 / 8.1304`. Closure: `3(79.0625)+15.8125 = 253.0` ✓ and `3(59.6232)+8.1304 = 187.0` ✓.

**The protocol choice is right and consistently applied.** The mint runs `--epochs 30` with checkpoint writes suppressed and `best_epoch_path` reachable only on an EM branch this recipe never takes — **final-epoch, no checkpoint selection**. HateMM's `final` row is the matching row; MHC-ZH's inventory is headed *"(Tier 2, final-epoch protocol)"*.

**Every downstream statement re-derives.** §5.2 reach `0.05376 / 0.05181`; power `30/40 = 75 %` and ZH needing all `30`; the withdrawal of *"cannot produce a CONTINUE at all"* is correct. §5.3 caps `131.4667 / 95.6667`; `R(2|P_0|) = 158/120` dead; `R(118.5) = 119`; `(119+37.2)/2 = 78.1 ⇒ 0.9261 / 0.6563`; `(90+29)/2 = 59.5 ⇒ 0.9545 / 0.6611`; at `k = |P_0|` `0.6889/0.7354` and `0.7139/0.7417`; τ_hi cells all inside the caps at `π* = 0.965/0.810/0.7325` and `0.9833/0.8222/0.7417` — **every one of the twelve figures reproduces.** §5.3's `≈ 9 / 8`, §6.3's ratios and §10's bullet are consistent.

**Everything else numeric** re-derived, including the Hanley–McNeil algebra, the π* gaps, the `[28.95, 29.0)` emptiness, F98's DEG-A from `aggnet_main_*_OUT.json`, `n_test 215/149/161`, both data defects re-measured this session, and `GATE-NULL` re-measured from the two operative caches.

**Quotations.** Every registry field and every `ban_scope` verifies character-exact modulo dash/quote normalisation. **F113's `dead[]` entry carries keys `["name","status"]` only.** All `.md` locators land. Every ellipsis omission still runs **against** C09.

**Executability: nothing becomes an engineering HALT.** All three relevant sha256 match (mint against the banked `meta.mint_script_sha256`; `mechnov_pairverify` against the run-time assertion; `mechfix_ops` against F113's record). Every cited line number is exact. `GATE-FIXK20` corroborated. `sacct` confirms job `13847`. `squeue` empty. Budget sums to ≈116 against ≈115.

**Statistics and decidability.** `AUC_strat` fully pinned; both nulls exact for the nulls they name; both families intersected ⇒ conservative; Holm and the IUT correct; `p`-floor `1/1001 ≪ α/2`. Macro-F1 monotonicity re-derived independently. **No single rule can carry a CONTINUE**; at least one `LIVE_ON_NET` cell exists per (dataset, τ).

**Residual gold-label paths: none beyond the one §4.3 names.** All 13 features, both stratifications, `τ_hi^{(f)}`, the standardisation, the permutation pools and the head-training partition walked.

**Legality holds in both directions.** The three that run **against** C09 (F51, F66's re-open sentence, CAL-5) are carried anyway.

---

## IMPORTANT

**I-1 — §4.2's derivation block, whose whole purpose is checkability, mis-transcribes and mis-locates its primary source in three places.** (a) **The HateMM per-seed triple is wrong.** v13 writes `26 / 27 / 27`; `ERRPAT_HateMM §1`'s cell table gives, for the **final** protocol, `s0 = 28`, `s1 = 26`, `s2 = 26` — i.e. **`28 / 26 / 26`**. The sum `80` and mean `26.6667` are right and the row is pinned by `(25, 2, 1)`, so nothing downstream moves. (b) **The ZH locator is off by a section:** the seed-consensus table is `ERRPAT_MHC-ZH` **§2** (*"ERROR INVENTORY (Tier 2, final-epoch protocol)"*), not §1, which *"avoids per-item claims entirely"*. (c) **The protocol note's counterfactual is wrong:** the val-sel mean is `79/3 = 26.3333`, so `24/26.3333 × 84.3333 = 76.86` ⇒ **`≈ 77`**, not `78`. (d) Minor: the val-sel row is rendered `24 / 3 / 2` in the source's order two rows below a table using `(|P_0|, a, b)`.

**I-2 — four statements still stand on the retired `|P_0| ≈ 76 / 55`.** §5.2's *"the same ≈76 values"*; §6.2's `k ≈ 76` mapping (`1 − 0.204(1−ov)`, `ov ≈ 0.755`, `0.9898` — corrected: `79`, `0.2124`, `0.765`, `0.9894`); §5.3's `R(x)` example on `1.5 × 55 = 82.5` (ZH is now the integer `90`; the live hazard is HateMM's `1.5 × 79 = 118.5`); §0's history bullet `k/n ≈ 76/744`. None is read by a rule, which is why this is Important — but round 12's High was precisely that `|P_0|` was inconsistent, and v13 leaves the inconsistency live in three specification-adjacent places.

**I-3 — §6.1's new rung-4 headwind mis-describes the table it cites.** §4.10 is headed *"3-seed means, head arena. RAW column re-read from `vsw_main_hatemm_OUT.json`"* — a **HateMM-only** ladder whose three-entry column is **per seed**, not per dataset, so *"per-dataset first entries"* is wrong and **no cross-dataset comparison exists at `:644-647`**. A cross-dataset sign flip does exist and is **stronger**: `:784` gives ZH `F89_T4 −0.0063` against HateMM's `+0.0054`, and `:786-790` records the MECHFIX head-space test reads against this arena's, concluding *"T2b/T4 are sub-0.007 and sign-unstable across the two head-space arenas"*; `:902` books it as *"same magnitude, opposite sign."*

**I-4 — §5.2's reach bullet misattributes one knife-edge leg: on MHC-ZH a one-item shortfall closes the branch on *power*, not reach.** `K-REACH` uses the exact `+0.050` rate, so the integer bar is `|P| ≥ 29`; at `29` reach still clears (`0.05009`) and it takes **two** to fail on reach. What a one-item shortfall does is fire `p_w < 30` ⇒ `K-FELDMAN` fails by construction — the **power** leg the next bullet describes.

**I-5 — the scope statement diverges between §0 and §12 again (R11 I-2 regression), and §0's counts understate the diff.** §0 uses "four sentences of §5.2 / four of §5.3 / three of §6.1 / three of §6.3"; §12 uses "one paragraph of §5.2 / one paragraph + one list item of §5.3 / two paragraphs of §6.1 / one block of §6.3". Different units, different counts, different ordering — and both understate the actual hunks. The **byte-identical enumeration is identical in both places and correct**, which is the half that matters.

---

## Checked and deliberately not counted

§10's *"expected to be closed on power on MHC-ZH"*; §4.3's `|P_0|/n ≥ 0.10` remark (now `0.1062 / 0.1036`, **better** than under the retired figures); the `≈ 1.4 %` bank-channel magnitude (`1/79 = 1.27 %`); `GATE-LEDGER`'s trainlog reads of mixed files, safe on the hard `VAL_RE` filter; §5.6's *"30-epoch Adam"* where `src/run_rac.py:684` uses `AdamW`; `ITEM-STRATUM` occupancy not emitted the way the row strata are; `queued.screening_arena_switch`; §2's *"~25–60 s"* against a measured `33.2`; the mint-timing figures; the `(τ, k)`-cell wording.

---

## Bottom line

**The science is finished and I could not move it — for the fifth round running.** The instrument, the legality spine, the label-use discipline, the executability, the budget and the inference all hold under independent re-derivation. A KILL is fully available and remains the honest expectation.

**Round 12's High is discharged.** I re-derived `|P_0| ≈ 79 / 60`, `n_unstable ≈ 9 / 8` and `|P_{τ_hi}| ≈ 40 / 30` myself; the identity closes exactly on both datasets; the final-epoch protocol is right and consistently applied; and every downstream figure reproduces.

What remains is five text-level defects, four of them **inside the material v13 added this round**, none of which touches a rule, threshold, feature set, null, gate, operating point or verdict. Three are provenance failures of exactly the kind the round-12 High was about.

**Is this ready to hash-freeze and submit as a single CPU-only SLURM job? Not yet — but only barely not.** Every repair is a text edit. With those made, v14 is ready to freeze and submit as one 8-CPU / 32-GB / no-GPU / no-`--time` job of ≈115 CPU-minutes, subject to the three submission preconditions the STATUS block names and the immediately-prior `squeue` empty-check, sha256 re-verification and namespace-absence check.
