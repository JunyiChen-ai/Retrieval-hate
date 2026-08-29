# Gate-0 Reopen 2026-07-31 — Independent Review, Round 2

**Reviewer.** Fresh independent worker, no exposure to the adjudicator's reasoning,
given the request, the revised record and round 1's verdict.
**Verdict.** `REVISE — 0 Critical / 3 High / 7 Important`
**Disposition.** All ten findings applied (H-2 partly refuted on re-check and
corrected in the safer direction). See `GATE0_REOPEN_2026-07-31.md` §9.

---

**Round-1 application audit (my first job).** Eleven of the fourteen findings are genuinely applied across all surfaces: C-1 (C07 is `held_lattice_delta_unwritten_reachability_unscreened` in the registry, the disposition block, `TARGET_FINDINGS.md` and `TARGET_LOOP.md`; "has been run and fails" survives only as an explicit withdrawal at `GATE0_REOPEN_2026-07-31.md:345`), H-1, H-2, H-3, I-1, I-4, I-5, I-6, I-8, I-9 all check out, and all ten registry `status` strings now match their `new_status` counterparts. **I-2, I-3, I-10 and the round-1 C10 hedge withdrawal were applied to the narrative record only** — see H-3. I-7 is half-applied — see I-4.

**Independent re-measurement.** I re-derived the census from the six gt files with no reference to either document. Every load-bearing `[M]` figure reproduces exactly: key-set `['id','label','text']` on all six; ws-only 39/9 and 0x4; ZH tags 243/34 with histogram `em` 254 / `/em` 254 and nothing else; 141/243 = 0.5802 vs 39/336 = 0.1161, base 0.3109; val 20/34 vs 8/44; 49 keywords / 254 occurrences train, 50/288 train+val; markup fraction median 0.0000, max 0.862069, 203 rows >10 %, markup-bearing median 0.2604; `<76`-char rows 161/221/92; HateMM medians 694.5 interp / 696 upper; p90 = 0.5051 (`lower`) / 0.5071 (`linear`) / 0.5155 (`higher`). The C01 arm table recomputes cell-for-cell from the stored confusions (HateMM rotation spread 0.8505-0.8692, 4/6 below primary; ZH 0.8462-0.8974, 2/6 below). §3.7's 6/6 4-dp identity is real. OBS-1 verifies. Asset claims verify: HateMM has only `-LoRA-curric` ro-caches, `MHC_zh` only `-LoRA`; `torch.save = _no_save` is exactly `headspace_mint.py:274-281`; all four `headspace_*.py` and all six arena `_OUT.json` exist. **No fabricated number found.**

**Downgrade verdicts.** All four are justified. C12: F55's ban_scope and detail verify verbatim and "all three levels" is unambiguously the three *encoder-composition* levels — decisive and correct. C10: EUM's `as of this recon` hedge is literally present, and EUM writes three revival preconditions, so HOLD-with-preconditions is the faithful kind of record where C14's ban (no revival path but a user ruling) is not; BSY's `bank-ADDITION` scoping is genuinely in its text (twice). C11: the registry claim is verbatim disjunctive. C07: F82's ban_scope does split vote-side from head-side. The three confirmed strikes (C08, C13, C14) rest on measured premise failures or registry text, not on bans.

## High

**H-1 · `GATE0_REOPEN_2026-07-31.md:417-426` — the C11 downgrade's load-bearing evidence is used past its written scope, in the exact way round-1's I-9 forced the record to fix C08's premise 3 from the same document.** ERRPAT_MHC-ZH §5.2 sits under `## 5. CONTENT COVARIATES (Tier 2)` (`:269`) and is computed on the **ZH test split, n = 149** (`:185`, `:272`) with a **CPU re-mint proxy head** (`:39`). Same tier, document and split as §5.4, which this record demotes to "a non-significant underpowered result". §5.2 is nonetheless recorded as **"measured positive"** with no qualifier — and it is the sole reason C11 is not struck. The record also drops `ERRPAT_MHC-ZH:307-308`: class-stratified, each half is underpowered (negatives p = 0.0506, positives p = 0.0668). Separately, the cross-dataset substrate figures (161/221/92 rows under 76 chars) apply a threshold derived from MHC-ZH's own quartiles to corpora whose medians are 696 and 369 characters. **Repair:** restate the §5.2 leg with tier/split/protocol/pooled-only limitation, note the `test_rule` tension, and replace or drop the 76-char cross-dataset census.

**H-2 · `TARGET_FINDINGS.md:75` and `TARGET_LOOP.md:1653` — "AGGNET held/carried the largest oracle ever measured on this object" is a misquote of F98, broadened in scope and landed as verified fact.** F98's actual text is *"C3 ENTERED THIS PREGATE WITH BY FAR THE LARGEST ORACLE CEILING ANY MEMBER OF THIS FAMILY HAS EVER HAD."* **Repair:** correct both surfaces to F98's own scope, and add the misquote to §3.

> **ADJUDICATION (partly refuted on re-check).** The phrase *"the LARGEST ORACLE
> CEILING EVER MEASURED ON THIS OBJECT"* **is verbatim** in
> `autoresearch/goal_mllm_plus3/state/directions_tried.json`'s F98 entry — round 2
> checked only `findings.jsonl`. So it is a real quote from one of two primary
> records, not a fabrication. However `findings.jsonl` F98 and
> `AGGNET_PREGATE_RECORD.md:678` are both narrower, and the conservative record is
> the family-scoped one. **Applied in the safer direction:** the family-scoped
> phrasing is now used in all landed surfaces and the disagreement between the two
> primary records is recorded in §6.

**H-3 · `TARGET_STATE.json` — four round-1 repairs were applied to the narrative record only; the machine-readable disposition block still carries the defective text, and on C10 it now directly contradicts the record.** C10 `gap_named` still asserts the "arguably REPLACES the bank object" hedge the record says is withdrawn; C06 `load_bearing_evidence` still carries I-3's unqualified best-of-six claim; C06 `why_gated_not_struck` still bases the warrant on the ensembling carve-outs (I-2 unapplied plus a live misstatement); the paper note `extent` still lacks I-10's split label. **Repair:** port the record's corrected text into all four JSON fields verbatim.

## Important

**I-1 · C07's unblock (c) imports F82's head-side clause onto C07's object**, the same over-application the Critical was about. F82's head-side clause governs a *"head-side graded auxiliary"*; C07 is a cone metric over a harm-act partial order, and no text shows those are the same object. **Repair:** state (c) as conditional.

**I-2 · `held_nonisomorphism_gate_unwritable` asserts unwritability from a three-source enumeration the record refuses to treat as exhaustive everywhere else** — the identical enumerative form the record calls "an EXTENSION" at C10 and disavows at C07. The disposition kind (hold) is right; the status string overclaims. **Repair:** rename to `held_nonisomorphism_gate_unwritten_as_posed`.

**I-3 · Two pinpoint citations do not exist, in a record whose subject is citation fidelity.** F99's actual citation for the exchange-rate claim is `:458-459`, not `:463`. `LITSWEEP8:218-222` is the §2.3 hubness table; the reduction argument lives at `:200-201` and `:268-273`. **Repair:** correct both.

**I-4 · `TARGET_STATE.json` — I-7's `prior_status_note` re-asserts as historical fact the very string round-1 found unsourced.** Repo-wide, `mechanism_gate_only_low_novelty` now appears only in round-1's own finding text and in this note; the candidate registry is entirely uncommitted. **Repair:** cite the artifact the string came from, or state it is not recoverable.

**I-5 · The C09 legality quote of `LITSWEEP3_DATA_CENTRIC.md:82` is truncated before its only qualifying clause** — *"…does not apply to the mechanism (though Wall-A still caps the achievable magnitude)"*, and the same section prices it at *"+3 any dataset: ~1-2%"* (`:94`) and *"at most +0.001-0.006"* (`:91`). This is the truncation pattern round-1 charged as High at C11, reproduced at the record's one promotion. Legality is unaffected; the prior is not. **Repair:** quote the parenthetical.

**I-6 · The MHC-EN entity histogram is a train+val occurrence count presented beside per-split row counts** (`&#39;` x51 / `&quot;` x22 / `&amp;` x18 is train+val; train-only is 43 / 17 / 16), inside a table certified "exact" — the mirror image of round-1's I-10. Related: the p90 convention label is self-contradictory across M-1, §7 and `TARGET_FINDINGS.md`. Numerically 0.5051 is numpy's `lower`, 0.5071 `linear`, 0.5155 `higher`. **Repair:** label the histogram and use one convention vocabulary.

**I-7 · The F82 headwind is quoted with the clause that limits its dataset coverage elided, and the GRADEDLBL ceiling's resolution is not stated.** F82's ban_scope ends *"; HateMM out of scope (no Offensive class)"*. And `GRADEDLBL_PREGATE_RECORD.md` states the ceiling is a **dev-label gold cheat on dev splits** (n = 80 EN / n = 78 ZH) whose ZH `+0.0256` is **2 dev items** (`:137`), and pre-declares at `:72-75` that the oracle *"does NOT bound the head's representation-reshaping"*. **Repair:** restore both.

## Not raised, checked and cleared

The three confirmed strikes (C08's premises 1-2 are genuine measured data failures with a named residual and unblock; C13 rests on a direct measurement with a short inference and an explicit "performance route only" scope; C14 rests on registry text alone with TVB correctly disowned); reversibility language, present and uniform on all ten entries; `ordered_backlog` genuinely untouched; `C09_A0_PREREG_DRAFT.md` present and described as a draft; F82/F55/F60/F80/F70/EUM/BSY/TVB/F78/F88/F99/F106/F107/F112/F113/F114/`banned_constraints[1,3,5,6,10]`/`hard_constraints` quotes all verbatim-faithful; LBOP verified as a real, distinct candidate at `research-wiki/TARGET_GATE0_ITER6_LITERATURE.md:246,271,284-288,444` and `TARGET_REVIEW_RAW.md:740-752`, with its order sourced from a label-blind MLLM; `prep_mhc.py:76` correct (the recon's `:72` was wrong); EDCM `+0.0273/+0.0394` and `+0.0380/+0.0444` exact at `TARGET_FINDINGS.md:9`; EUM's 83 %-contiguous-block figure exact; `generate_VideoMLLM_embedding_readout_HF.py` CELLS table confirms the `ow_` prompt/span confound.
