# Gate-0 Reopen 2026-07-31 — Independent Review, Round 1

**Reviewer.** Fresh independent worker, no exposure to the adjudicator's reasoning.
**Request.** `refine-logs/GATE0_REOPEN_2026-07-31_REVIEW_REQUEST.md`
**Under review.** `refine-logs/GATE0_REOPEN_2026-07-31.md` and how it landed in
`TARGET_STATE.json::gate0_reopen_2026_07_31`, `TARGET_FINDINGS.md`, `TARGET_LOOP.md`.
**Verdict.** `REVISE — 1 Critical / 3 High / 10 Important`
**Disposition.** All fourteen findings applied; the Critical moved C07 from strike
to hold. See `GATE0_REOPEN_2026-07-31.md` §8.

---

## Verdict

**REVISE — 1 Critical / 3 High / 10 Important**

The record's core work is real and largely holds up. I independently re-derived the census from `data/gt/*/{train,val}.jsonl` and every load-bearing `[M]` figure reproduces exactly (key-set `['id','label','text']` on all six files; HateMM whitespace-only 39/9; ZH tags 243/34; `em` 254//`em` 254 and 49 keywords on train; hate rate 141/243 = 0.5802 vs 39/336 = 0.1161, base 0.3109; val 20/34 vs 8/44; markup fraction median 0.0000, max 0.8621, 203 rows >10%, markup-bearing median 0.2604; `<76`-char rows 161/221/92; Archive = MHC + MHC_zh only; CLIP_Embedding 100/130/71/6/2). The C01 arm table verifies cell-for-cell against `C01_A0_OUT.json` with every accuracy recomputable from the stored confusion matrices. F55, F60, F70, F80, F82, F88, F96, F98, F99, F105, F112, F113, F114, EUM, BSY, TVB, LITSWEEP2/3/5/8, HEADCOV, GRADEDLBL, NCA_FORENSIC_RECON and the C05 comparator quotes are all verbatim-faithful. The C12 F55 leg (the decisive downgrade leg) is correct. I also stress-tested the C13 measurement — rows containing a harvested keyword *without* markup hate at only 10/140 = 0.0714 versus 141/243 = 0.5802 with it, so the tag itself carries the signal and the C13 strike's inference survives.

The defects below are scope, kind-of-record, and precision failures, not fabrication.

---

### Critical

**C-1 · C07 is struck on a precondition that was never attempted and on a screen that was never run.**
`refine-logs/GATE0_REOPEN_2026-07-31.md:230` states *"The reachability screen its boundary also demands **has been run and fails on both datasets**"* (mirrored at `TARGET_STATE.json:131`, `TARGET_FINDINGS.md:75`, `TARGET_LOOP.md` disposition table). No reachability screen of C07 has ever been run. What was run is F82's **vote-side** Offensive-reweighting oracle, and F82's own `ban_scope` (`directions_tried.json`, F82 entry) splits the two explicitly: *"vote-side Offensive reweighting closed both datasets … **head-side graded auxiliary = F44-capped + admissibility-gated, only revivable by user ruling WITH a new mechanism argument**."* C07 is a cone metric — a head-side/representation object — so the cited evidence sits on the other side of its own source's written boundary. The record concedes this three paragraphs later: its unblock at `:247` demands *"a **fresh** reachability screen at `+0.050`"*, which is unnecessary if the screen had already run. Leg 1 (`:227-229`) is weaker still: it says only that *"no delta has been written"* — an un-attempted precondition, whereas C05 is given `held_*` at `:445` for a precondition that was attempted **and demonstrated unwritable**. The record's own house rule against this substitution is the one it invokes to downgrade C12 (`LITSWEEP3_DATA_CENTRIC.md:80`, *"a headwind to price, not a coverage of this mechanism"*). C07 is therefore a strike carrying a HOLD's evidence and a HOLD's unblock, and the record's claim at `:222` that each confirmed strike *"survives on ban-free, faithfully-quoted evidence"* is false for it.
**Repair:** re-dispose C07 as `held_lattice_delta_unwritten_reachability_unscreened` with the same reversibility language as C05/C10/C11/C12; delete "has been run and fails"; restate F82 as a headwind priced from the vote-side channel and quote F82's head-side clause verbatim; or, to keep the strike, name a ban that reaches C07's object on its own text.

---

### High

**H-1 · C08 premise 2's categorical "no >=2-dataset substrate" is measured only for HTML *tags*.**
`refine-logs/GATE0_REOPEN_2026-07-31.md:256`: *"`0` on HateMM and `0` on MHC-EN — re-measured exactly. A `>=2`-dataset route has no substrate."* MHC-EN carries **64/549 train and 9/80 val rows with HTML entities** (`&#39;` x51, `&quot;` x22, `&amp;` x18 — recomputed by me) — un-cleaned scrape residue of the same provenance family. The recon's own census table (`C05PLUS_FORENSIC_RECON_2026-07-31.md:436-441`) carries this column; the record's §2 verification table (`:51-67`) silently drops it while certifying the rest "exact".
**Repair:** restate premise 2 as tag-scoped, record the MHC-EN entity counts in §2 and in `TARGET_STATE.json` `premise_2`, and either show entities are not a usable provenance substrate or soften "no substrate" to "no *highlight-marker* substrate".

**H-2 · The C11 downgrade truncates its own headwind citation before the clause that contradicts it.**
`:389` quotes `ERRPAT_MHC-ZH:405` as *"effectively LOCKED … +0.0738 if all 11 flipped, but §7.1 shows no better transcript exists; the deficit is signal absence"* and concludes it is *"a headwind to price rather than a closure."* The source row ends with a further clause the record drops: **"No legal unmeasured lever found."** That is the source's own verdict on the exact cluster the downgrade calls "unscreened", and it is the strongest text against the downgrade. Same truncation at `TARGET_STATE.json:182`.
**Repair:** quote the row in full in both places and re-argue why a missingness-representation lever is not covered by "no legal unmeasured lever found", or convert C11 back to a strike.

**H-3 · The C12 downgrade's narrow-reading precedent is quoted without its own blocking clause.**
`:414` cites *"F60/AUG rules **MLLM-as-data-generator admissible**"* and the unblock at `:435-436` offers stability-as-multi-view-target as *"governed by F60's admissible generator role"*. F60's `ban_scope` ends **"Do not re-propose without D7 generator-role sub-ruling"**, and F60's detail closes *"Revisit only under a user D7 generator-role sub-ruling AND acceptance of a weaker-than-tied prior."* D7 is an open user ruling (`progress.json` handoff: *"the five open USER RULINGS (D7 novelty boundary; …)"*). The record therefore offers C12 an unblock route that is itself blocked — the identical omission it charges the recon with for F80.
**Repair:** add F60's D7 clause verbatim to `gap_2` and to the unblock in both `GATE0_REOPEN_2026-07-31.md:414-418,435` and `TARGET_STATE.json:191,194`.

---

### Important

**I-1 · Both "cosmetic corrections" are median/percentile *convention* differences, not errors — and the correction is applied inconsistently.** `:69` and `:75`. The recon's `0.5155` is exactly numpy's `higher` p90 on MHC-ZH train (I reproduce it to 4 dp); `696` is exactly the upper median of HateMM train. The recon used the upper convention **consistently** — MHC-ZH val `111` and MHC-EN val `443` are also exact upper medians, and the record leaves both unremarked while certifying the layer "exact". Landed as errata in three files (`TARGET_STATE.json:37-47`, `TARGET_FINDINGS.md:75`, `TARGET_LOOP.md`). **Repair:** restate M-1/M-2 as convention notes ("upper vs interpolated"), not corrections, and either check or drop the remaining medians.

**I-2 · The C06 "bans do not reach C06" correction cites carve-outs whose object is C14, not C06.** `:509-515`. F80's *"multi-prompt ensembling remains a SEPARATE user-gated item"* and F70's *"Does NOT price: … multi-prompt ensembling"* both carve out **ensembling** — which C06's own dedup boundary forbids it from becoming (`TARGET_STATE.json:200`). The conclusion (F80's prompt-language ban does not reach a prompt-orbit geometry candidate) is right; the warrant conflates C06 with C14. **Repair:** re-base on object mismatch (C06 is not prompt-language matching, and F70 prices individual readout cells, not orbit geometry).

**I-3 · "A matched-block-L2 random orthogonal rotation matches or beats the real prompt displacement on both datasets" (`:504`) is a best-of-six selection.** Against the **primary** arm `common_displacement`, 4 of 6 HateMM rotations (`orthrot_17p6/29p1/60p4/72p7` at 0.8505/net +1) and 2 of 6 ZH rotations sit *below* it. Full verified spread: HateMM 0.8505–0.8692, ZH 0.8462–0.8974. **Repair:** state the rotation spread, or label the comparison "best of six" in the summary sentence as it is in the table.

**I-4 · "bit-identical … on 6/6 seeds" (`:177`, `TARGET_STATE.json:89`) overstates precision.** F113's banked arena artifacts store 4-dp values (`headspace_arena_hatemm_s0_OUT.json`: `head_deployed_acc: 0.8884`) against C02's `0.8884408602150538`. What is verifiable is identity **at the recorded 4-dp precision**. **Repair:** say "identical at the recorded 4-dp precision on 6/6 seeds".

**I-5 · F88 transcription slip.** `:551` gives HateMM 3/3-seed error invariance as *"(88–93 %)"*; F88 says **"(89-93%)"**. Propagated to `TARGET_STATE.json:222`. **Repair:** correct to 89–93 %.

**I-6 · OBS-1 attributes the validity gates to the wrong artifact.** `:183`: *"`C02_A0_DECISION.json` carries five named validity gates."* `C02_A0_DECISION.json` has no gates key at all (`bars, holm_family, interpretation_boundary, per_dataset, result_exists, run_id, schema_version, target_met, verdict`); the five gates live in `C02_A0_OUT.json` under `datasets.<ds>.gates`. Everything else in OBS-1 verifies exactly (4 with `pass: true` incl. per-seed ARENA2; ZERO_CONTRACT no `pass`; two `DOCUMENTARY_CITATION_NOT_COMPUTED`; ZH `banked_text_zero_rows: []`). **Repair:** change the path in `:183` and `TARGET_STATE.json:119`.

**I-7 · C14's landed status and its recorded prior status are both off-register.** `TARGET_STATE.json:327` sets `struck_gate0_2026_07_31_diagnostic_only_role_preserved`, while the disposition block at `TARGET_STATE.json:158` records `new_status: struck_gate0_2026_07_31` — a machine consumer reading the dispositions gets a different string than the registry. `TARGET_STATE.json:333` records `prior_status: "mechanism_gate_only_low_novelty"`, a string that appears nowhere else in the repository; C14 sat in `ordered_backlog` exactly as C05–C13 did. **Repair:** make the two status strings agree, and set C14's `prior_status` to `ordered_backlog` (its eligibility flag already carries the diagnostic-only fact).

**I-8 · C09's legality citation is one-sided in the way the record itself corrects elsewhere.** `:564` cites `progress.json:25` as affirmative legality. `LITSWEEP5_COMPLETENESS.md` §4(ii) is an on-point, post-ruling in-repo adjudication headed *"The contradiction (load-bearing)"* which states the ruling's two blessed classes — *"Trained SELECTOR on train labels"* and *"Trained symmetric RESHAPER on train labels"* — are *"both already measured dead"*, and that the ruling *"was written at lit-round-count 3 — before F75/F77/L1 sharpened the walls."* This does not defeat the legality verdict, but the record corrects exactly this one-sidedness at `:628` for `NCA_FORENSIC_RECON:110` and does not apply it here. **Repair:** cite LITSWEEP5 §4(ii) alongside `progress.json:25` and state that the legality holds while the ruling's viability premise is flagged stale in-repo.

**I-9 · C08 premise 3 is used past its source's written scope.** `:258-261` calls quoted-hate FPs *"measured at chance."* `ERRPAT_MHC-ZH` §5.4 is (a) TIER-2 CPU-re-mint proxy, (b) on the ZH **test** split, and (c) closes with *"Both are recorded as hypotheses the ZH test split is too small to settle, not as clusters"*; the cluster table at `:409` lists it as "not significant", not measured-null. The record names the MHC-ZH-only residual but not the tier or the source's own refusal to settle. **Repair:** add the tier/scope caveat, or rest the strike on premises 1–2 alone.

**I-10 · The paper-note keyword census is train-only but reads corpus-wide.** `:709` (and `TARGET_STATE.json:292`): *"49 distinct keywords over 254 occurrences (`em` 254 / `/em` 254, no other tag)"* — train-only; train+val is **50 keywords / 288 occurrences** (`em` 288 / `/em` 288). The table immediately above it is train **and** val. Same for "median text length of only 106 characters" (train; val is 108.5). **Repair:** label the census "train split".

---

### Downgrade verification (the record's headline claim)

- **C12 — justified.** F55's ban_scope and detail verify verbatim; "EN closed at all three levels" is unambiguously three levels *of the encoder-composition question* (frozen/F50, collapsed-adapted-deployed/B4-F53, healthy-img+adapted-text/F55). The recon's "MHC-EN is additionally closed at all three levels" was a real misread, and it was the leg that would have made the >=2-dataset arithmetic impossible. Downgrade correct, subject to H-3.
- **C11 — justified in direction, over-stated in support.** The disjunctive claim is verbatim in the registry; the thin-transcript cluster is real (p = 0.0048, robust at 0.0051) and the C02 fallback genuinely targets a different operator. Subject to H-2.
- **C10 — justified, but one clause is too generous.** EUM's ban does contemplate revival on three written preconditions, so `held_*` with those preconditions as the unblock is the faithful record. However `:343`'s *"C10 arguably **replaces** the bank object rather than adds rows"* is undercut by EUM's own measurement, quoted twenty lines later, that a flat unit bank puts only 10.6–11.3 distinct parent videos in a top-20 — i.e. it has more rows than the video bank. The `banned_constraints[3]` conditional in the unblock catches this, but the "arguably replaces" hedge should be withdrawn.

### Kind-of-record and reversibility

Reversibility language is present, correct and consistent on all ten entries (`"registry-level; reversible by a future user ruling. NOT a measured kill."`), the `what_this_reopen_does_not_do` block is accurate, the historical `ordered_backlog` is genuinely untouched, C09's prereg is a draft only, and nothing in C02 or C04 was modified. C05 (hold), C06 (gate), C13/C14 (strikes) are the correct kind of record for their evidence. C07 is not — see C-1.
