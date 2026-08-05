# Gate-0 Reopen 2026-07-31 — Independent Review, Round 12

**Reviewer.** Fresh independent worker, no exposure to the adjudicator's reasoning.
**Verdict.** `GO — 0 Critical / 0 High / 2 Important`. **No disposition changed.**
**Disposition.** Both Importants and all observations applied. See §19.

---

I verified the record against primary sources with four independent passes (three fact-check workers plus my own recomputation) and found the adjudication sound on all four scope items.

## What I verified independently

**Census / `[M]` layer — re-derived from scratch, 100% reproduction.** All six gt files (key-set `['id','label','text']`, single set); HateMM whitespace-only `39`/`9`, MHC `0/0/0/0`; ZH tags `243`/`34`, tag histogram `em 254 / /em 254` and nothing else; entities `64`/`9` MHC-EN with train-only `&#39;x43, &quot;x16, &amp;x17`; MHC-ZH `1` strict / `2` hex-inclusive; hate rates `0.5802 (141/243)` vs `0.1161 (39/336)` base `0.3109`, val `0.5882 (20/34)` vs `0.1818 (8/44)` base `0.3590`; keywords `49/254` train, `50/288` train+val, top-5 identical; markup fraction median `0.000000`, max `0.862069`, `203` rows >10%, p90 `0.505051` lower / `0.507133` linear / `0.515464` higher — **Note M-1's convention claim is correct**; the `10/140 = 0.0714` bare-keyword stress test reproduces **only** train-scoped (train+val gives `10/146`), as the record states. Medians: HateMM train `694.5`/`696`, MHC-EN train `369`, val `439.5`/`443`, ZH val `108.5`/`111` — **Note M-2's upper-convention claim is correct**.

**Title census.** Whole-file `891/891` and `897/897`; `<em>` in `391` Title / `0` Transcript. Join-scoped: `629/629`, `657/657`, `277 / 0`; **EN title median 51 (transcript 322), ZH 27 raw / 13 markup-stripped (transcript 78)** — exact. `scripts/prep_mhc.py:72-85` and `scripts/prep_video_dataset.py:126-139` do read `title` and `transcript` as separate variables. `LITSWEEP2_INPUT_FIDELITY.md:56` is gt-schema-scoped and §3.3's "re-scraping YouTube metadata" inference is indeed wrong for both MHC datasets. **Round 3's Critical stands.**

**§3.7 fold-head identity — recomputed.** Identical at 4 dp on **6/6**, exactly as claimed and at exactly the precision claimed.

**C01 arm table** recomputed from stored confusions across all 14 arms x 2 datasets (`<1e-12`); `n_dev` 107/78; six rotations; spreads `0.8505-0.8692` / `0.8462-0.8974`; **4 of 6** HateMM and **2 of 6** ZH rotations strictly below the primary arm — all exact.

**Scope item (a) — the one strike is faithful and in-scope.** C14's `eligible_for_primary_target: false`, its `dedup_boundary` and `hard_constraints[4]` are verbatim; the strike is confined to the performance backlog and preserves the diagnostic role; the TVB "7 of 7 at ~0" support is correctly identified as a **prediction** and explicitly not relied on.

**Scope item (b) — statuses are the correct kind.** All ten registry entries carry the byte-exact reversibility string; no candidate was run; `dispositions` sums to 10; historical `ordered_backlog` intact; `new_jobs` and `new_metrics` empty; C09's prereg is a `DRAFT`.

**Scope item (c) — no inference recorded as a measurement.** Every figure traced resolves to a primary source or a stated re-measurement. The inferential steps are all labelled: "markup-stripped" (inferred), TVB (predicted), C13's regression step (plausibility inference), C07's supervision-source enumeration (supporting inference, explicitly non-load-bearing), C10's "space is EMPTY" (extension), EUM's compressed `median 83%` (disclosed as a compression against the true `0.8289` / `0.7174` / `74.2%`), F98's superlative (conservative family-scoped wording with the two-registry disagreement recorded).

**Scope item (d) + the three headline downgrades.** All eight unblocks are concrete and proponent-actionable; **none is over-cautious**. C10 — EUM's ban is a *conditional* closure with three written revival preconditions, hedged "as of this recon" over a three-item enumeration excluding a rule-based gold-free MLLM-free boundary; BSY's block is textually scoped to "bank-ADDITION" and procedural. C11 — the claim is verbatim disjunctive; `ERRPAT §5.2` makes the second disjunct non-empty; the hold carries a written self-destruct clause. C12 — F55's ban_scope and detail confine "EN closed at all three levels" to the encoder-composition question. C07, C08 and C13 likewise.

## Findings

**I-1 · Live text still calls C07 struck.** §3.5 reads *"This strengthens rather than weakens the C07 strike"* and §3.6 *"the strike is recorded with both numbers"*; the JSON mirrors both in `V-5`/`V-6`. C07 is `held_lattice_delta_unwritten_reachability_unscreened` on all four surfaces, and §4.2 states the opposite conclusion explicitly. This is the residue of round 1's Critical never being swept out of §3, and it survives on the machine-readable surface. It changes no disposition, status or asserted fact — the LBOP facts are correct and are used correctly as C07's unblock (a). **Repair:** replace "the C07 strike" with "C07's first unblock condition" / "the hold is recorded with both numbers."

**I-2 · The C12 unblock re-asserts the attribution the record corrects 35 lines earlier.** The unblock reads *"stability-as-weight (lands on `[5]` **under EUM's gloss**, and is then dead)"*, while §4.2 leg 2 says *"it is a **stack, not a gloss**"* — EUM reaches "MLLM-derived boundaries or weights" via `"P3 / P11 plus banned_constraints[5] … and [6]"`, four authorities (verified verbatim). Round 10's I-2 was raised against this exact unblock phrasing and landed only in `gap_2`. The error runs conservative, so C12's disposition is unaffected. **Repair:** "lands on `[5]` under the EUM four-authority stack (P3/P11 + `[5]` + `[6]`)".

## Observations (not findings)

- `§54` — *"Findings are enumerated in §§8-13"*; rounds 7-11 are in §§14-18.
- `TARGET_LOOP.md` — the C12 unblock still ends *"(then F60 governs, subject to D7)"*, ten lines after the same paragraph correctly narrows it to the open **generator-role sub-ruling** (round-8 I-4 lag on one surface only).
- `TARGET_FINDINGS.md` — the attestation gives `629/629` and `657/657` but not `277 / 0`; internally consistent because that surface never makes the whole-file claim the figure join-scopes.
- `§96` labels `0.5051` "nearest-rank" while M-1 and §7 use numpy's `lower` (same value; round 2's I-6 asked for one vocabulary).
- `:417-419` carries a duplicated clause and `:380` begins lowercase — formatting residue from the round-4 and round-11 inserts.
- C07 `gap_2`'s F82 quote elides `(any monotone weighting, any tau)` behind the ellipsis; the elision makes the vote-side ban look *narrower* than written, i.e. runs against the record's own argument.
- On MHC-ZH `orthrot_83p8`'s `0.8974` ties `orthrot_72p7`; the table prints both rows, so "best of six" is transparent.
- C14's `prior_status` field holds a backlog position rather than the historical status string; the substitution is disclosed in `prior_status_note`.
- `generate_VideoMLLM_embedding_readout_HF.py` lives at `src/utils/`, not `scripts/`; the cited line range does contain the `ro_ow_L24` tuple, and the prompt/readout-span confound is real and confirmed at source.
