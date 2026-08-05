# Gate-0 Reopen 2026-07-31 — Independent Review, Round 3

**Reviewer.** Fresh independent worker, no exposure to the adjudicator's reasoning.
**Verdict.** `REVISE — 1 Critical / 1 High / 4 Important`
**Disposition.** All findings applied; the Critical moved C08 from strike to hold.
See `GATE0_REOPEN_2026-07-31.md` §10.

---

**Prior-round application audit (my first job).** All fourteen round-1 and all ten round-2 findings are genuinely applied on all four surfaces, with two exceptions charged below (R1 I-9 on `TARGET_LOOP.md`; R2 I-6 on `TARGET_STATE.json` and record §4.1). I re-checked the JSON explicitly: all ten registry `status` strings match their `new_status` counterparts (C14 reconciled to `struck_gate0_2026_07_31` with the diagnostic role in a separate `scope` field); `held_nonisomorphism_gate_unwritten_as_posed` is in all four surfaces; R2 H-3's four JSON ports are all present verbatim; C14's `prior_status_note` now states its provenance; `ordered_backlog` is untouched; reversibility language is present and uniform on all ten entries.

**Round-2 H-2 adjudication — confirmed handled honestly.** `"the LARGEST ORACLE CEILING EVER MEASURED ON THIS OBJECT"` *is* verbatim in `directions_tried.json`'s F98 entry (`dead[65]`), and `findings.jsonl` F98 is narrower. The record adopts the family-scoped phrasing everywhere and records the two-record disagreement in §6. Correct and conservative.

**Independent re-measurement.** Census re-derived from scratch on the six gt files: key-set `['id','label','text']`, no `title` field; ws-only 39/9 and 0x4; ZH tags 243/34, histogram `em` 254 / `/em` 254; 141/243 = 0.5802 vs 39/336 = 0.1161, base 0.3109; val 20/34 vs 8/44; 49/254 train, 50/288 train+val; markup median 0.0000, max 0.862069, 203 rows >10 %, markup-bearing median 0.2604; p90 0.50505 `lower` / 0.50713 `linear` / 0.51546 `higher`; medians 106 / 108.5 / 694.5-interp-696-upper / 369 / 439.5. C01 arm table recomputes cell-for-cell from the stored confusions (all 28 arms, both datasets); rotation spread and 4-of-6 / 2-of-6 counts correct under strict `<`. Arena `head_deployed_acc` 4 dp; C02 `ARENA2.pooled_native_acc` matches on 6/6. Asset claims verify. **No fabricated number found.** All four downgrades are justified — F82's vote/head split, EUM's `"as of this recon"`, BSY's `bank-ADDITION` scoping, C11's verbatim disjunctive claim, and F55's three *encoder-composition* levels all verify verbatim.

## Critical

**C-1 · The C08 strike's premise 1, as restated, is refuted by primary evidence in this repository — and the falsifying evidence is in the same paragraph that states it.**
`GATE0_REOPEN_2026-07-31.md:262-272` (mirrored in `TARGET_STATE.json` `premise_1_corrected`, `TARGET_FINDINGS.md:75`, `TARGET_LOOP.md:1540`) concludes: *"the correct premise is 'no separable title channel without re-deriving source metadata' — a data-collection act declined LOW/~0 in litsweep2."*

The title is already on local disk as a separate field on exactly the two datasets a `>=2`-dataset route needs:

- `/data/jehc223/Multihateclip/English/annotation(new).json` — **891/891 rows carry a non-empty `Title`**; `/data/jehc223/Multihateclip/Chinese/annotation(new).json` — **897/897**. (HateMM: 0/1066, consistent with the record.)
- `scripts/prep_mhc.py:72-85` — the very function the record cites at `:76` — reads `title = (entry.get("Title") or "").strip()` and `transcript` as **separate variables** before concatenating. `scripts/prep_video_dataset.py:126-139` is byte-identical logic for MHC-ZH. Emitting a title-separated gt is a re-run of an existing CPU-only, deterministic script.
- F88 ledger correction (c) — quoted in this same paragraph with an ellipsis exactly where the numbers sit — reports *"medians: title 15 chars, transcript 76, composed 96"*. That measurement is only possible from a separated title.

`LITSWEEP2_INPUT_FIDELITY.md:56`'s `title_present = 0` is true **of the gt-jsonl key schema only**; the record then imports LITSWEEP2 §3.3's inference (*"absent from source … recovering it means re-scraping YouTube metadata"*), which is factually wrong for both MHC datasets. Nothing in `hard_constraints` or `banned_constraints` bans a title channel. So premise 1 — one of the two premises the record says the strike rests on — is false in the direction that matters, and C08's own written unblock (*"exhibit a provenance artifact present on >=2 datasets"*) is arguably already met by the title itself. What remains is premise 2, scoped to the `<em>` **marker** only, plus a premise 3 the record itself demotes to non-significant corroboration.

This is a stretched strike of the same kind round-1's Critical caught at C07.

**Required repair (any one):** (a) re-dispose C08 as `held_*` with the same reversibility language as the other holds and the title half named as the unscreened residual; or (b) retain the strike but restate it as scoped to the **provenance-marker** route only, delete *"without re-deriving source metadata / a data-collection act"*, record that MHC-EN 891/891 and MHC-ZH 897/897 source rows carry a separable `Title` reachable by a CPU re-prep, and name the title-source half as a separate open candidate. Apply on all four surfaces.

## High

**H-1 · `GATE0_REOPEN_2026-07-31.md:63` — the round-2 I-6 repair introduced a transposed number inside the table certified "exact", and was not applied to the other two surfaces.** The record states *"(train-only 43 / 17 / 16)"* in the order `&#39;` / `&quot;` / `&amp;`. Recomputed: **`&#39;` 43, `&quot;` 16, `&amp;` 17** — the last two are swapped. (Train+val 51 / 22 / 18 is correct.) Round 2 asserted the same transposed triple, so it propagated unchecked. Separately, R2 I-6's repair landed only in §2: §4.1 and `TARGET_STATE.json` `premise_2` still present the train+val histogram beside per-split row counts with no label. Also: the `1` MHC-ZH train entity row is regex-convention-dependent — a hex-inclusive `&#?\w+;` gives **2** rows. **Repair:** correct to `43 / 16 / 17`; label the histogram and add the train-only triple on the other surfaces; state the entity regex convention.

## Important

**I-1 · `TARGET_LOOP.md:1540` — round-1's I-9 is unapplied on this surface, and it contradicts the record.** The disposition table records C08's basis as *"three measured premise failures"*; §4.1 demotes premise 3 to non-significant. **Repair:** correct the count.

**I-2 · `GATE0_REOPEN_2026-07-31.md:376-377` and `:584-586` — `banned_constraints[5]`/`[6]` applied as a blanket "MLLM output" ban, unengaged by the counter-precedent this record itself identifies at C12.** `[5]`'s literal text is four words — *"MLLM-scores-as-training-signal"* — and `research-wiki/TARGET_GATE0_ITER6_LITERATURE.md:260` says LBOP's MLLM emits lower/upper policy sets and *"不输出 label、score、memory pair/key 或 rationale"*. `[6]` is *"P1-P5 re-proposals"*. **Repair:** state the attribution at C05 and C07 as construction-dependent under the same EUM-vs-F60 tension recorded at C12.

**I-3 · Wrong pinpoint citation introduced by the round-2 I-5 repair.** *"+3 any dataset: ~1-2 %"* is `LITSWEEP3_DATA_CENTRIC.md:95`, not `:94`; `:69` carries a different *"~2 %"* belonging to §3 (ELR). **Repair:** `:94` → `:95`.

**I-4 · Note M-1 asserts something about §7 that is not true of §7.** M-1 states the three percentile labels are used in §7; §7 gives only `lower` and `linear` — the `higher`/`0.5155` label is absent, and `0.5155` is precisely the recon figure M-1 exists to explain. **Repair:** add it to §7.

## Checked and cleared

The four downgrades (all justified, all quotes verbatim-faithful at their cited scope); the C13 and C14 strikes (measurement-plus-plausibility and registry text respectively, with TVB correctly disowned as a prediction); C06's `gated_on_zero_cost_falsifier` (correct kind of record — C01's evidence is real, its supporting bans genuinely miss C06's object on F80's/F70's own text, and the ro-cache/span-confound design constraints verify); C09's promotion and its legality verdict (`progress.json:25` and `LITSWEEP3:82` verbatim, counter-text carried, three HALT boundaries stated); the strategic finding §6 (every figure traced); the `$0`/zero-touch boundary; and the three-surface agreement on all ten statuses.
