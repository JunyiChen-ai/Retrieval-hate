# Gate-0 Reopen 2026-07-31 — Independent Review, Round 14: **GO (0C/0H/0I)**

**Reviewer.** Fresh independent worker, no exposure to the adjudicator's reasoning.
Four independent fact-check passes plus own recomputation.
**Verdict.** `GO — 0 Critical / 0 High / 0 Important`. **THE CLOSING VERDICT.**
**Disposition.** No finding to apply; all fourteen observations were applied anyway.
See §21.

---

## Verdict

**GO — 0 Critical / 0 High / 0 Important**

No disposition should move; the one strike is faithful; every status is the correct kind of record; nothing measured is an inference; every unblock is usable and none is over-cautious.

## The four bar items, as cleared

**(a) The one strike is faithful and applied within its evidence's own scope.** C14 carries `eligible_for_primary_target: false` — the **only** entry in all fourteen carrying that field — and the `dedup_boundary` quote is byte-exact, as is `hard_constraints[4]`. The strike is confined to the performance backlog and preserves the diagnostic role. TVB's *"7 of 7 at ~0"* is correctly identified as a **prediction** (`LITSWEEP5_COMPLETENESS.md:120` contains the literal word "predicted"), TVB's own *"flagged-not-banned"* is verbatim, and the record explicitly does not rely on it.

**(b) Kind-of-record and reversibility.** All ten `gate0_reopen_2026_07_31` blocks carry the byte-identical string `"registry-level; reversible by a future user ruling. NOT a measured kill."`; `dispositions` sums to exactly 10 (1 strike + 6 downgraded + 1 held + 1 gated + 1 promoted); C01-C04 carry no gate0 key; `new_jobs`/`new_metrics` empty; the C09 prereg is a `DRAFT`. **The JSON's C12 `unblock` no longer says "under EUM's gloss"** — it now reads *"lands on `[5]` under the EUM FOUR-AUTHORITY STACK (P3 / P11 plus [5] and [6])"*, and round 12's certification is amended rather than left standing. Round 12's I-1 landed on all four places.

**(c) Nothing measured that isn't.** Census re-derived from scratch **twice, independently**, with no reference to any prior document: **100 % reproduction on all 18 `[M]` claims** — including the `10/140 = 0.0714` stress test reproducing **only** train-scoped (train+val gives `10/146`), and Note M-1/M-2's percentile and median conventions. The C01 arm table recomputed from the stored confusion matrices at **every one of 28 arm-cells** (`<1e-9`), with spreads `0.8505-0.8692` / `0.8462-0.8974` and the 4-of-6 / 2-of-6 counts exact. I recomputed §3.7 myself: C02's `gates.ARENA2.pooled_native_acc` against the banked `head_deployed_acc` — identical at 4 dp on **6/6**, exactly the precision claimed, with bit-equality correctly disclaimed. §6's AGGNET figures are verbatim, including the **conservative family-scoped** superlative. Every inferential step is labelled.

**(d) Unblocks usable; the three headline downgrades justified.** Re-derived from primary text: **C10** — EUM's precondition (2) enumerates exactly three illegal sources and concludes emptiness *"as of this recon"*; a rule-based gold-free MLLM-free boundary is outside that enumeration; BSY's block is textually scoped to *"bank-ADDITION"* and blocks a **prereg** pending a user ruling — procedural, not a scientific kill. **C11** — the claim is verbatim disjunctive; `ERRPAT_MHC-ZH:301-308` makes the second disjunct measured-positive (`p = 0.0048`, every figure exact); `:405` is genuinely a **cluster-scoped table row**, not a document verdict. **C12** — both `directions_tried.json` and `findings.jsonl` F55 confine *"EN closed at all three levels"* to the **encoder-composition** question. C05's `unwritten_as_posed` is the honest string; C07's registry boundary is conjunctive and its delta genuinely un-attempted; C08's premise 1 is refuted at source; C13's surviving basis is a proponent-satisfiable precondition and C13 carries no `eligible_for_primary_target` field. **No hold should have been a strike, and no unblock is vacuous.**

## Observations (none is a finding)

1. **A direct single-authority gloss of `banned_constraints[5]` exists and is not engaged.** F103/OCR's ban_scope glosses `[5]` directly, on an **archive field**: *"It is Qwen-2.5-VL GENERATED TEXT and falls under banned_constraints[5] (MLLM-scores-as-training-signal / the P4-P11 family boundary)."* It runs **conservative**, F60 conflicts with it head-on, and the C12 downgrade rests on gap_1/gap_3 — both verified — so the disposition is untouched. Worth adding, since it raises the burden on *both* branches of the fork.
2. **F108 is not named in C08's unblock.** F108 bans *"any change to WHICH STREAMS OR IN WHAT PROPORTION enter the RETRIEVAL KEY … CLOSED BY CONSTRUCTION, SO A RENAME CANNOT EVADE IT"*. C08 lands in F108's carve-out (ii) (content, not weight) **as written**, and the record's *"Nothing in hard_constraints or banned_constraints bans a title channel"* is literally true — F108 is a finding-level ban_scope. But a proponent realising unblock (a) as "expose the title as its own key block" walks into it.
3. **C06's six "random rotations" are angles on the same one-parameter family as the primary.** `c01_policy_contrast_a0.py:1272`'s `orthogonal_blocks()` is a Givens mixing of the two endpoint blocks; the code's own guards confirm theta=45deg **is** `common_displacement` (max abs diff `8.9e-08`-`1.2e-07`) and theta=0 **is** `endpoint_concat`. The record's *"a random direction with matched norm"* reads more diffuse than the object is — but this **sharpens** the adverse reading. Relatedly, the arm table omits two arms that also beat the primary — ZH `endpoint_concat` `0.8846`/+2 and HateMM `common` `0.8692`/+3, the decision block's own named strongest controls, with `gain_over_strongest_control` `-0.0256` / `-0.0094`, `pass: false`, `decision.continue = false`. Both omissions **understate** the record's own adverse case.
4. **Two round-13 observations certified applied landed on one surface only** — the "seven-row priority table" and "four words" descriptors. Both are wording/count descriptors on text either explicitly not relied on (TVB) or quoted verbatim adjacent (`[5]`), so neither misstates evidence.
5. **Round-13 observation 12 reached no surface.** Global-R2's quoted epitaph compresses arms of `+0.0044` and `+0.0277` against the `+0.040` bar; the second is 1.44x under. The compression is the source's and is quoted, and C05's leg rests on conditional information `<= 0` with both arms sub-bar regardless.
6. **C07's unblock (a) understates LBOP-0's bar at the point of use.** `:284` also requires macro-F1, per-fold sign agreement and a joint Farkas/gradient-cone audit. Recorded in §20 and the JSON but not where a proponent reads it — and it runs against the record's own argument.
7. **EUM's own status field records that a legal rule-based unit was already built and measured negative** — *"The best LEGAL evidence unit was already built (EXP_mm_segment_keys.md:195, final-epoch dF1 -0.0116, 3/3 seeds negative)"* — uniform K=4 windows with Whisper word-level timestamps, i.e. exactly the rule-based gold-free MLLM-free boundary C10's unblock (2) posits. EN-only and on consensus **vote** keys rather than the retrieval-bank object, so it does not close C10 — but it is an unpriced headwind.
8. **`ERRPAT_MHC-ZH:415` is the stronger headwind and is uncited** — but its section header scopes it to what is open *"in-box, at `$0`"*, and C11 is a training-time representation change. Round 13 already noted the record applies `:405` **narrower** than ERRPAT's broadest claim, i.e. against its own interest.
9. **The `LITSWEEP2` title error has a second uncaught instance** at `LITSWEEP3_ZH_SPECIFIC.md:39-40`; the record names only the LITSWEEP2 instance.
10. Minor: the `ro_*` caches are files, not directories; F88 null (3)'s "does not beat random deletion" is a val-sel loss and a final-epoch win, all under half a test item per seed, so "indistinguishable" is the exact reading; C14 remains a member of the historical `ordered_backlog` array as do C05-C13, disclosed deliberately; the AGGNET epitaph is lightly re-cast inside quotation marks, semantically identical.

---

*Read-only. Zero GPU, SLURM, Modal, model loads or `.pt` opens; no test-split file was opened by me or any worker; nothing was written; the C04 lineage was not touched and nothing in C02 was modified.*
