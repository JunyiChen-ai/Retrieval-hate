# Gate-0 Reopen 2026-07-31 — Independent Review, Round 5

**Reviewer.** Fresh independent worker, no exposure to the adjudicator's reasoning.
**Verdict.** `REVISE — 0 Critical / 2 High / 2 Important`
**Disposition.** All four findings applied. See `GATE0_REOPEN_2026-07-31.md` §12.

---

## What I verified independently (all clean)

**Census, re-derived from scratch** on `data/gt/{HateMM,MHC_zh,MHC}/{train,val}.jsonl` with no reference to either document: key-set `['id','label','text']` on all six, no `title` field; ws-only `39/9` + `0x4`; ZH tags `243/34`, histogram `em` 254 / `/em` 254 only; `141/243 = 0.5802` vs `39/336 = 0.1161`, base `0.3109`; val `20/34` vs `8/44`, base `0.3590`; `49/254` train, `50/288` train+val; markup median `0.000000`, max `0.862069`, `203 > 10 %`, bearing-median `0.2604`; p90 `0.505051` (`lower`) / `0.507133` (`linear`) / `0.515464` (`higher`); medians `106 / 108.5 / 694.5-interp-696-upper / 369 / 439.5-443`; MHC-EN entity rows `64/9`, train-only `&#39;` 43 / `&quot;` 16 / `&amp;` 17, ZH `1` strict / `2` hex-inclusive; R1's stress test `10/140 = 0.0714`. **Every figure reproduces exactly.**

**C01 arm table** recomputed cell-for-cell from the stored confusions: `0.8411/0.8505/0.8598/0.8224/0.8692/0.8505` and `0.8590/0.8846/0.8590/0.8333/0.8974/0.8974`, net-fix all match, spreads `0.8505-0.8692` and `0.8462-0.8974`, 4-of-6 / 2-of-6 correct under strict `<`. `configs/c01/c01_a0_v1.json` confirms `orthogonal_rotation_control.same_block_l2: True` with ex-ante frozen angles.

**C02** `gates.ARENA2.pooled_native_acc` matches the six arena `head_deployed_acc` at 4 dp on 6/6. OBS-1 verifies exactly.

**C08 premise-1 refutation**: `Title` non-empty on `891/891` and `897/897` raw rows; `prep_mhc.py:72-85` and `prep_video_dataset.py:126-139` read title/transcript as separate variables; per-dataset medians recomputed over train+val ids — EN title `51` / transcript `322`, ZH `27` raw / `13` stripped / transcript `78`. All exact.

**Quote fidelity**, checked at cited scope and found verbatim across F82, F80, F70, F60, EUM, BSY, TVB, F55, F99, F113 (**F113 only** — F114 does not contain the honesty clause), F114, F75, F78, F88, HEADCOV, LITSWEEP8, GRADEDLBL, ERRPAT-ZH, LITSWEEP2/3/5, NCA, `progress.json`, LBOP, SSR, EDCM, AGGNET. **No fabricated number found.**

**Kind-of-record**: all ten registry `status` strings equal their `new_status`; reversibility string present and uniform on all ten; historical `ordered_backlog` intact. Both strikes are registry-level, not measured kills; the five downgrades are justified at their sources' written scope; **no hold is over-cautious** — each names a usable unblock.

**Prior-round audit**: 33 of the 34 findings (R1 14, R2 10, R3 6, R4 4) are genuinely applied on all four surfaces. One is not — see H-1.

## High

**H-1 · Round 4's H-2 is unapplied on `TARGET_LOOP.md`, the surface its own repair named — and the stale text contradicts the same file twenty lines later.** `TARGET_LOOP.md` still read: *"price the title channel's Stage-0 oracle knowing its median length is 15 characters against a composed median of 96."* That is exactly the figure round 4 charged as an **MHC-ZH, test-split, markup-stripped** median inherited second-hand via F88 from `ERRPAT_MHC-ZH:270-271`, generalised to a route whose EN leg is `3.4x` larger. R4 H-2's repair was explicit: *"on all three surfaces."* The record, `TARGET_STATE.json` and `TARGET_FINDINGS.md` all carry the corrected per-dataset pricing; `TARGET_LOOP.md` did not, and the same file then described the correction the reader had just been given the superseded version of. **Repair:** port the qualified per-dataset text.

**H-2 · The C13 strike's sole ban-free leg states a categorical no-substrate result that was measured for HTML *tags* only — the identical scoping defect round 1 forced out of C08's premise 2, now carrying a surviving strike alone.** The record read: *"The `<em class="keyword">` markup exists on one of three datasets … the phenomenon C13 acts on has no `>=2`-dataset substrate."* C13's registry claim is not scoped to the `<em>` highlight — it is *"Removing sensitivity to native **HTML/title markup**"* — and **MHC-EN carries HTML markup on `64/549` train and `9/80` val rows**, a fact this record's own §2 table certifies as exact. The step from "`<em>` is on one dataset" to "the phenomenon C13 acts on is on one dataset" is therefore an inference, and it was the *only* basis the strike rested on after round 4 correctly demoted the hate-rate leg to a labelled inference. **Repair (any one):** (a) scope the leg explicitly to the `<em>` harvest-highlight and say why MHC-EN's entity rows are not a substrate for *nuisance invariance*; or (b) re-base on C13's own written self-scoping — its claim says *"a **ZH-specific** extraction nuisance"* — and record that the census confirms rather than establishes; or (c) C13 returns to HOLD.

## Important

**I-1 · The record's provenance attestation "the only files opened for measurement were the six gt files" is false after the round-3 and round-4 repairs, on three surfaces.** Rounds 3 and 4 added measurements taken from `/data/jehc223/Multihateclip/{English,Chinese}/annotation(new).json` — the `891/891` and `897/897` Title counts and the per-dataset medians — and §4.4 recomputes every C01 accuracy from the stored confusion matrices in `C01_A0_OUT.json`. Those are measurements, not reads. Nothing improper happened (both are `$0`, non-test and permitted), but in a record whose subject is provenance fidelity this is a false statement about its own conduct. **Repair:** extend the attestation on all three surfaces.

**I-2 · The machine-readable disposition block files the gated candidate inside an array named `held`, contradicting its own tally.** `dispositions.held` contained C05 **and C06** (`gated_on_zero_cost_falsifier`), while `effective_order_post_c04.tally` reads *"six holds (C05, C07, C08, C10, C11, C12), one gate (C06)"*. A machine consumer reading `dispositions.held` gets two entries, neither set matching the tally. Each entry's own `new_status` is correct, so no status is wrong — but the grouping is, and round 4 treated exactly this class of defect as material because a consumer reads the group. **Repair:** add a `gated` array containing C06.

**One note, not counted as a finding.** The record was internally inconsistent about its own review depth — *"three rounds"* in four places against *"four rounds"* in one. Both statements were true and no disposition depended on it; the three-round phrasing was simply stale.
