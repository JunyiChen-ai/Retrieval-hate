# Gate-0 Reopen 2026-07-31 — Independent Review, Round 10

**Reviewer.** Fresh independent worker, no exposure to the adjudicator's reasoning.
**Verdict.** `REVISE — 0 Critical / 1 High / 3 Important`. **No disposition changed.**
**Disposition.** All four findings applied. See `GATE0_REOPEN_2026-07-31.md` §17.

---

## H-1 · Round 9 does not exist on two of the four surfaces — the third consecutive recurrence of the defect rounds 7-9 each charged

The record and `TARGET_STATE.json` plus `independent_review.round_9` all state **nine** rounds. Both narrative surfaces still stated **eight**: `TARGET_LOOP.md` in four places (*"survives eight rounds"*, *"Eight rounds of fresh independent review"*, *"survives eight rounds"*, Records `ROUND{1..8}.md`) and `TARGET_FINDINGS.md` in four (*"through eight rounds"*, *"survives eight rounds"*, *"survived eight rounds"*, Records `ROUND{1..8}.md`). Grep for `ROUND9|round_9|round 9` returns **0 hits** in both, against 2 in the record and 3 in the JSON. This is the identical defect round 8's H-2 charged for round 7, recurring for round 9, and the review request makes cross-surface agreement on **counts** a GO criterion.

*(Round 9's raw review **is** appended, and round 9's own four findings all landed: H-1's "five of its seven" is gone from `TARGET_LOOP.md`; I-1's D7 narrowing is in `TARGET_STATE.json`; I-2's `LITSWEEP3:80`-as-general-form is on all four; I-3's attestation is restated on all four with `629/629`, `657/657`, `277 / 0` and the no-test-id clause. Only round 9 **itself** is missing.)*

**Repair:** sync both surfaces, add a round-9 paragraph and verdict, extend both Records lists.

## I-1 · The C08 title-median provenance cites a line range that does not contain the figure

Both the record and `TARGET_STATE.json` say the *"title 15 chars, transcript 76, composed 96"* median was inherited *"via F88 ledger correction (c) from `ERRPAT_MHC-ZH_2026-07-26.md:270-271`"*. Read directly: `:271` is the composition line (*"Deployed ZH text = `Title + " . " + Transcript` … (`scripts/prep_mhc.py:73-78`)"*); the medians are at **`:272`** (*"Medians on test: title 15 chars, transcript 76 chars, composed text 96 chars"*). The scope qualifiers the record attaches (Tier-2, test split, markup-stripped) are all correct — only the anchor is wrong. Same class as the pinpoint errors rounds 2 and 3 charged, introduced by round 4's H-2 repair. **Repair:** `:270-271` → `:272` on both surfaces.

## I-2 · EUM's broad reading of `banned_constraints[5]` is attributed to `[5]` alone; EUM stacks four authorities

The record: *"Broad-reading precedent exists (EUM glosses it as covering 'MLLM-derived boundaries or weights'), so if the stability statistic weights or selects training examples the ban applies."* EUM's precondition (2), verbatim: *"WITHOUT MLLM-derived boundaries or weights (that is **P3 / P11 plus** banned_constraints[5] 'MLLM-scores-as-training-signal' **and [6]** 'P1-P5 re-proposals')"*. EUM does not gloss `[5]` as reaching boundaries/weights — it reaches that object through a four-authority stack. In a record whose entire methodological finding is that authorities must be read at their own written scope, and whose C12 unblock turns on *"lands on `[5]` under EUM's gloss, and is then dead"*, the attribution matters. (The error runs in the conservative direction — the stacked authorities make the closure stronger, not weaker — so the C12 disposition is unaffected.) **Repair:** state the stack on both surfaces.

## I-3 · F80 is quoted at partial scope in the C06 warrant

The record: *"F80's object is prompt language … on MHC_zh, and its prohibition is conditional: 'do NOT re-propose prompt-language matching elsewhere without new mechanism' — the recon truncates the qualifier."* F80's ban_scope in full opens with an **unconditional** on-dataset closure the record omits: *"extraction-instruction language variations (**any language, any stream, either encoder arm**) on MHC_zh; prompt-language axis measured null-to-negative; do NOT re-propose prompt-language matching **elsewhere** without new mechanism (HateMM/EN are English-content = no mismatch exists)"*. The conditionality attaches only to *elsewhere*. The C06 warrant survives intact — it rests on **object mismatch** (orbit geometry is not extraction-instruction language), which is unaffected either way — but the record charges the recon with truncating this same entry's qualifier and then truncates its other half. **Repair:** quote F80's opening clause alongside the "elsewhere" clause on both surfaces.

## Checked and cleared (not raised)

- **The C14 strike is faithful and in-scope.** `eligible_for_primary_target: false` and the dedup-boundary sentence are verbatim; `hard_constraints[4]` is verbatim. The strike is scoped to the performance backlog with the diagnostic role preserved, the unblock is a user ruling, and TVB's *"7 of 7 at ~0"* is correctly identified as a **prediction** and explicitly not relied on.
- **All six downgrades justified**, C10/C11/C12 re-derived from primary text. C12's decisive leg holds; `REDTEAM_BAN_SCOPE_AUDIT.md:230` independently rules the broad reading an *"INDUCTIVE LEAP"*. C10's EUM ban does name the object but supplies three revival preconditions — conditional closure implies HOLD. C11's disjunct is genuinely open, with ERRPAT's *"No legal unmeasured lever found"* quoted in full and a written strike path.
- **Kind of record and reversibility:** all ten registry entries carry the reversibility string verbatim; ten `status` strings match their `new_status`; `dispositions` arrays sum to 10 and match the tally; `ordered_backlog`, `hard_constraints`, `unified_pilot_gate`, `serial_execution` unaltered; C01-C04 carry no `gate0_reopen` key; C09's prereg is a DRAFT.
- **Nothing recorded as measured that is not.** I re-derived the entire `[M]` layer independently — **18/18 claims MATCH exactly**, including `891/891`, `897/897`, join-scoped `629/629`, `657/657`, `391`/`0` and `277`/`0`, title medians `51`/`322` and `27`/`13`/`78`, `10/140 = 0.0714`, `49`/`254` and `50`/`288`, the p90 triple, `0.2604`, `203/579`, the entity triples, and both regex conventions. The C01 arm table recomputes cell-for-cell including both rotation spreads and the 4-of-6 / 2-of-6 claim. C02's `ARENA2.pooled_native_acc` equals the six banked `head_deployed_acc` at 4 dp on **6/6**, and `0.8875`/`0.8912` are confirmed to be `summary_3seed.FULL.acc_3seed_mean` — §3.7's correction is exact. The "≈ 36 heads" estimate is right (`K_FOLDS=5` plus the `fold == -1` deployed head). F113's honesty clause is byte-identical and in F113 only. F60's *"un-enumerated generator role is real"* is verbatim. The `3 x 3 x 2 x 2 = 36 cells` decomposition is written at `MECHNOV_PAIRVERIFY_PREGATE.md:202,308`. **No fabricated number found.**
- **No over-cautious hold.** Every hold and the gate names a proponent-actionable unblock; C11 additionally names the condition under which it should be struck without measurement.
