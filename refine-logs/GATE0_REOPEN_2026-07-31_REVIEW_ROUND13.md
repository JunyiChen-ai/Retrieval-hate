# Gate-0 Reopen 2026-07-31 — Independent Review, Round 13

**Reviewer.** Fresh independent worker, no exposure to the adjudicator's reasoning.
Three parallel fact-check passes plus own recomputation.
**Verdict.** `GO — 0 Critical / 0 High / 1 Important`. **No disposition changed.**
**Disposition.** The Important and all fourteen observations applied. See §20.

---

## What I cleared

**Scope item (a) — the one strike is faithful and in-scope.** C14 carries `eligible_for_primary_target: false` and the quoted `dedup_boundary` verbatim; `hard_constraints[4]` is verbatim. The strike is confined to the performance backlog and preserves the diagnostic role. TVB's "7 of 7 at ~0" support is correctly identified as a **prediction** (`LITSWEEP5_COMPLETENESS.md:120` contains the literal word "predicted") and explicitly not relied on; TVB's own "flagged-not-banned" is verbatim.

**Scope item (b) — statuses are the correct kind of record.** All ten `gate0_reopen_2026_07_31` blocks carry the **byte-identical** string `"registry-level; reversible by a future user ruling. NOT a measured kill."`. `dispositions` sums to exactly 10 (1 strike + 6 downgraded + 1 held + 1 gated + 1 promoted). C01-C04 carry no gate0 key. The historical `ordered_backlog` is intact. `new_jobs`/`new_metrics` empty; the prereg is a DRAFT.

**Scope item (c) — no inference recorded as a measurement.** I recomputed the whole `[M]` layer independently: key-set `['id','label','text']`; whitespace-only 39/9 and 0/0/0/0; ZH tags 243/34 with histogram `em 254 / /em 254` and nothing else; EN entities 64/9 with train-only 43/16/17; ZH strict 1 / hex-inclusive 2; hate rates 0.5802 (141/243) vs 0.1161 (39/336), base 0.3109, val 0.5882/0.1818/0.3590; bare-keyword 10/140 = 0.0714 (train-scoped only); keywords 49/254 train, 50/288 train+val; markup fraction median 0.000000, max 0.862069, 203 rows >10 %, p90 0.505051/0.507133/0.515464 — Note M-1's convention claim is exactly right; medians 106 / 108.5 / 694.5-696 / 369 / 439.5-443 — Note M-2 holds; markup-bearing median 0.2604; `CLIP_Embedding` 100/130/71/6/2; `Archive` = `MHC`, `MHC_zh` only. **100 % reproduction, no fabricated number.** The C01 arm table recomputes to `<1e-12` on all 14 arms x 2 datasets with spreads and the 4-of-6 / 2-of-6 counts exact. §3.7's fold-head identity is identical at 4 dp on 6/6 at exactly the claimed precision. Every inferential step is labelled.

**Scope item (d) + the three headline downgrades.** All eight unblocks are concrete and proponent-actionable; **none is over-cautious**. C10 — EUM's ban is a *conditional* closure supplying three written revival preconditions, with precondition (2) hedged "as of this recon" over a three-item enumeration excluding a rule-based gold-free MLLM-free boundary; BSY's block is procedural and scoped to bank-addition. C11 — the claim is verbatim disjunctive; `ERRPAT:301-306` makes the second disjunct non-empty; `:405` is verbatim and correctly **cluster-scoped** — the record in fact applies it **narrower** than ERRPAT's own broadest claim at `:415`. C12 — F55's ban_scope and detail confine "EN closed at all three levels" to encoder/feature-composition objects. C07, C08, C13 likewise.

**Round-12 application.** I-1 landed on all four places. Eight of nine observations landed.

## Finding

**I-1 · The C12 unblock in `TARGET_STATE.json` still attributes the broad reading of `banned_constraints[5]` to "EUM's gloss", and the round-12 block certifies the repair as applied.** The JSON's C12 `unblock` read `stability-as-weight (lands on [5] under EUM's gloss and is then dead)` — the exact string round 12 charged at I-2 and round 10 charged at I-2 before it. It contradicts the same object's `gap_2` three fields earlier (*"Broad-reading precedent exists but it is a STACK, NOT A GLOSS"*) and misstates the primary source: EUM reaches "MLLM-derived boundaries or **weights**" through four authorities, verbatim *"that is P3 / P11 plus banned_constraints[5] … and [6]"*. `banned_constraints[5]`'s own literal text reaches neither boundaries nor weights. The repair landed on the markdown record but not on the machine-readable surface — while the round-12 block states *"corrected"*. That certification is false as written. Nothing about C12's disposition changes: the error runs conservative, and the record says so. But in a record whose stated methodological finding is that authorities must be read at their own written scope, a live disposition field attributing a four-authority closure to a single ban's gloss is a misstatement of the evidence. **Repair:** correct the JSON string, and do not re-certify round 12's I-2 as applied until it is gone.

## Observations (not findings)

1. §54's "Findings are enumerated in §§8-18"; round 12 is §19.
2. `TARGET_LOOP.md` compresses the same unblock to "stability-as-weight (then `[5]` applies and it is dead)" — asserts `[5]` applies with no stack qualifier.
3. The JSON's `M-1` uses "nearest-rank / linear interpolation / upper-higher" while the other three surfaces use numpy's `lower`/`linear`/`higher`.
4. `banned_constraints[5]` is described as "four words"; hyphen-split it is five tokens. The substantive point is right.
5. `LITSWEEP5_COMPLETENESS.md` §4's table has eight rows (ranks 1-7 plus a parenthesized `(8)`); the record calls it a "seven-row priority table". The rank claim ("7 of 7") is exact.
6. **`banned_constraints[10]`'s `22.3 / 17.4 / 16.5` are train-arena net-item requirements** (`n = 744/579/549`, `LITSWEEP7_LANDING_SITE.md:107-111`). Neither the ban nor the record states the arena, so a proponent applying the recommended Gate-0 currency to a test-sized arena would mis-scale by ~3.5x. Inherited from the ban's own text, not introduced here.
7. `banned_constraints[2]` bans only "cross-seed ensembles" — which is why TVB can call multi-prompt "flagged-not-banned". The literal multi-prompt ban carrying C14 lives in `hard_constraints[4]`. Both ledgers are accurate in their own terms and the strike stands independently on C14's own eligibility flag.
8. C14 is still a member of the `ordered_backlog` array; "struck from the performance backlog" is a status-string change. The record discloses that the historical array is deliberately untouched, so the two are consistent.
9. §4.2 C12 cites archive-as-key `dAcc -0.0014 +/- 0.0313, zero vote flips` without dataset scope — those are MHC-ZH-only (5 ZH seeds); the EN arm is `-0.0062 +/- 0.0051` with 0-2 flips/seed. Inert, because the leg's argument is that the measurement is the wrong *object*.
10. **"C07 is a cone metric — a head-side/representation object" is the record's reading**; C07's registry entry contains no head-side language. Well-founded, and the record explicitly hedges the adjacent graded-auxiliary step, but it is the one unlabelled inferential step inside the C07 downgrade.
11. LBOP-0's gate is reported as ">= +0.050 on both datasets"; `TARGET_GATE0_ITER6_LITERATURE.md:284` also requires macro-F1, per-fold sign agreement and a Farkas/gradient-cone audit. The record **understates** the comparator's bar, against its own argument.
12. Global-R2's quoted epitaph "oracle@coverage 10x under bar" compresses arms of `+0.0044` and `+0.0277` against the `+0.040` bar; the `+0.0277` arm is 1.44x under. The compression predates the record and is quoted, not asserted.
13. The route-scoping sentence quoted for C01 has its exact home in the superseded `c01_a0_v1.json:7` / `v2.json:7`; the executed v4 config has no `negative_scope` key. The substance survives verbatim-in-substance at `TARGET_LOOP.md:649`.
14. `data/CLIP_Embedding` has a sixth directory `MHCsmoke` (0 entries) absent from the `100/130/71/6/2` row.
