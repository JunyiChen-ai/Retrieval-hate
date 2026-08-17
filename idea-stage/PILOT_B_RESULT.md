# P-B result — Near-duplicate and label-conflict census

- **Freeze:** `idea-stage/PILOT_FREEZE_2026-08-09.md`, section P-B (written before any candidate metric was computed).
- **Script:** `idea-stage/pilot_b_dup_conflict_census.py`
- **Raw results:** `idea-stage/pilot_b.json`
- **Log / pid:** `logging/runs/pilot_b/run.log`, `logging/runs/pilot_b/run.pid` (3094564)
- **Run:** 2026-08-09 08:36:53, CPU, conda `HateVideo`, single submission, 0.41 s.
- **Test-set contact:** none. Guard armed and logged (`GUARD ARMED: any path whose name contains
  'test' HALTs; P-B opens TRAIN splits only (no dev_seen, no val, no test)`). P-B's guard is
  stricter than P-A's: it also HALTs on `dev_seen`, `val.jsonl` and `valid.tsv`. The 10 paths the
  real run touched are listed in `pilot_b.json:paths_touched` — all `train`.
  The train↔test contamination audit is **deliberately not run**, per the freeze.

---

## 1. The frozen rule, transcribed unedited

> **P-B — Near-duplicate and label-conflict census**
>
> **What it gates.** The "duplicate-conflict memory" candidate and the "naturally occurring minimal
> pairs" slot; it is also a validity check on every retrieval number this project has produced
> (if reposts are casting repeated votes, neighbourhood purity is inflated).
>
> **Data.** **Train splits only**, all datasets with cached CLIP embeddings: HateMM, MHC-EN,
> MHC-ZH, HateClipSeg, ImpliHateVid. `dev_seen`/`val`/`test` are not opened.
> **The train↔test contamination audit is deliberately NOT run here** — it would require test-set
> contact. It is deferred to a separately authorised, separately pre-registered step.
>
> **Similarity.** Two frozen measures computed on L2-normalised blocks:
> `c_img` (visual cosine) and `c_txt` (transcript/title text cosine). A repost re-encoded or
> re-narrated keeps visual similarity while text may drift, so `c_img` is primary.
>
> **Endpoints.**
> - **N1** — count of pairs at `c_img ≥ {0.85, 0.90, 0.95}`, within-dataset and cross-dataset.
> - **N2** — of those, **conflicting pairs**: near-duplicate pairs whose binary labels differ.
> - **N3** — among MHC conflicting pairs, how many have a `Counter Narrative` vote on one side.
> - **N4 (discriminator against false positives)** — for every flagged pair, token-level Jaccard
>   overlap of the transcripts. A genuine repost shares transcript; two unrelated talking-head
>   videos do not. Report the conflicting-pair counts **additionally filtered** at
>   `Jaccard ≥ 0.5` as the conservative count.
>
> **Frozen decision rule** (on the conservative count: `c_img ≥ 0.90` **and** `Jaccard ≥ 0.5`):
> - **ALIVE** — ≥ 30 conflicting pairs across the train pools.
> - **AMBIGUOUS** — 10–29 conflicting pairs (would need manual verification of a sample).
> - **DEAD** — < 10 conflicting pairs.
>
> **Known limitation, stated in advance.** High cosine on mean-pooled 8-frame CLIP features is
> **not** proof of near-duplication; same-genre talking-head footage can reach 0.9. N4 is the
> mitigation, and any surviving claim requires visual verification of a sample. The census
> measures an upper bound on duplication and a lower bound on verified conflict.

---

## 2. Pool as loaded

| dataset | train n | positives | included |
|---|---|---|---|
| HateMM | 744 | 298 | yes |
| MHC (EN) | 549 | 168 | yes |
| MHC_zh | 579 | 180 | yes |
| ImpliHateVid | 1283 | 649 | yes |
| **HateClipSeg** | — | — | **NO — no train assets exist** |
| **pool total** | **3155** | 1295 | 4 975 435 unordered pairs |

**HateClipSeg was excluded, not fabricated.** `data/CLIP_Embedding/HateClipSeg/` contains only
`test_seen_*` caches and `data/gt/HateClipSeg/` contains only `test.jsonl` — there is no train
embedding file and no train id list. Opening what does exist would have been test-set contact,
so the dataset contributes nothing here. This is recorded in
`pilot_b.json:datasets_excluded_no_train_cache`.

Pairs are **unordered and deduplicated** (upper triangle, `i < j`) and self-pairs are excluded by
construction. A "conflicting pair" requires the two items' **binary** labels to differ.
57 of the 3155 items have an empty transcript; Jaccard for a both-empty pair is frozen to 0
(conservative — it can only *reduce* the conservative count, never inflate it).

SHA-256 of every embedding cache used is recorded in `pilot_b.json:files`.

## 3. N1 / N2 — flagged pairs and conflicts (full, unfiltered)

| `c_img ≥` | N1 total | N1 within-ds | N1 cross-ds | N2 conflicting total | N2 within-ds | N2 cross-ds |
|---|---|---|---|---|---|---|
| 0.85 | 1365 | 1228 | 137 | **194** | 83 | 111 |
| 0.90 | 645 | 630 | 15 | **53** | 43 | 10 |
| 0.95 | 195 | 191 | 4 | **39** | 38 | 1 |

Within-dataset breakdown (N1 / N2-conflicting):

| dataset | ≥0.85 | ≥0.90 | ≥0.95 |
|---|---|---|---|
| HateMM | 224 / 57 | 117 / 37 | 87 / 34 |
| ImpliHateVid | 744 / 6 | 454 / 2 | 89 / 1 |
| MHC (EN) | 17 / 2 | 6 / 1 | 3 / 1 |
| MHC_zh | 243 / 18 | 53 / 3 | 12 / 2 |

Cross-dataset pair types (all of them):

| pair type | ≥0.85 N1 / conflicting | ≥0.90 | ≥0.95 |
|---|---|---|---|
| HateMM ↔ ImpliHateVid | 136 / 111 | 15 / 10 | 4 / 1 |
| HateMM ↔ MHC | 1 / 0 | 0 | 0 |

No MHC-EN ↔ MHC-ZH pair reaches 0.85, and ImpliHateVid never pairs with either MHC.
The HateMM ↔ ImpliHateVid cross-dataset conflict count (111 at 0.85) is high only because the two
corpora have near-opposite base rates (40 % vs 51 % positive) and are both English talking-head
video — every one of those 111 is eliminated by N4 (see below).

## 4. N4 — Jaccard discriminator and the conservative count

| `c_img ≥` | flagged pairs | Jaccard mean / median over flagged | pairs with Jaccard ≥ 0.5 | **conservative conflicting** (conflict ∧ Jaccard ≥ 0.5) | within-ds | cross-ds |
|---|---|---|---|---|---|---|
| 0.85 | 1365 | 0.165 / 0.131 | 58 | **5** | 5 | 0 |
| **0.90 (gate)** | 645 | 0.188 / 0.122 | 49 | **5** | 5 | 0 |
| 0.95 | 195 | 0.273 / 0.134 | 34 | **5** | 5 | 0 |

The Jaccard filter is doing exactly the work the freeze designed it for: median transcript
overlap among visually flagged pairs is ~0.12–0.13, i.e. **the overwhelming majority of high-cosine
pairs are not reposts at all** — they are same-genre footage, precisely the false-positive mode the
freeze warned about. 53 conflicting pairs at `c_img ≥ 0.90` collapse to 5 once transcript overlap
is required. All 5 are within-dataset; **zero** cross-dataset conflicts survive.

The 5 conservative conflicting pairs (identical at all three thresholds), from
`pilot_b.json:census.top_conflicting_pairs_at_c_img_0.90`:

| # | a (label) | b (label) | dataset | c_img | c_txt | Jaccard |
|---|---|---|---|---|---|---|
| 1 | `hate_video_50` (1) | `non_hate_video_338` (0) | HateMM | 1.0000 | 1.0000 | 1.00 |
| 2 | `hate_video_59` (1) | `non_hate_video_338` (0) | HateMM | 0.9995 | 0.8808 | 1.00 |
| 3 | `hate_video_297` (1) | `non_hate_video_338` (0) | HateMM | 0.9995 | 0.8808 | 1.00 |
| 4 | `non_hate_video_338` (0) | `hate_video_63` (1) | HateMM | 1.0000 | 1.0000 | 1.00 |
| 5 | `0q1PET_IDGc` (0) | `KV49pENhk4c` (1) | MHC (EN) | 0.9960 | 0.8726 | 0.70 |

## 5. N3 — Counter Narrative among MHC conflicting pairs

88 of the 1128 pooled MHC items (EN + ZH train) carry at least one raw `Counter Narrative` vote.

| `c_img ≥` | conflicting pairs with ≥1 MHC side | of those, ≥1 side has a CN vote | conservative conflicting w/ MHC side | of those, with a CN vote |
|---|---|---|---|---|
| 0.85 | 20 | **7** | 1 | **0** |
| 0.90 | 4 | **0** | 1 | **0** |
| 0.95 | 3 | **0** | 1 | **0** |

At every threshold, every MHC-involving flagged pair had MHC on *both* sides (no MHC↔non-MHC
flagged pair reaches 0.90). **N3 = 0 at the gate threshold.** The Counter-Narrative hypothesis —
that duplicate-conflict pairs are reposts where one copy is framed as counter-speech — has no
support in the train pools: the only surviving MHC conflict pair (`0q1PET_IDGc` / `KV49pENhk4c`)
has no Counter Narrative vote on either side.

## 6. Null control (label permutation)

Not required by the freeze; run anyway as an honesty check. Binary labels permuted across the
whole 3155-item pool (seed 20260909, 200 permutations), holding the geometry and the Jaccard
filter fixed. There are 49 conservative pairs (`c_img ≥ 0.90` ∧ `Jaccard ≥ 0.5`) in the pool.

| quantity | value |
|---|---|
| conservative conflicting under permuted labels, mean | **24.1** |
| 95 % range over permutations | [15, 30] |
| **observed** | **5** |

The observed count is far *below* chance. Genuine near-duplicates in these corpora are strongly
**label-concordant** — reposts overwhelmingly get the same binary label. This is the opposite of
what the duplicate-conflict candidate needs, and it independently corroborates the DEAD verdict:
the shortage of conflicts is not a shortage of duplicates (there are 49), it is that duplicates
agree. It also bounds the "inflated neighbourhood purity" worry the freeze raised — reposts do
repeat votes, but there are only 49 such pairs among 4.98 M, so the purity inflation is negligible
at the corpus level even though it is real.

## 7. Verdict against the frozen rule

Gate quantity = conservative conflicting pairs at `c_img ≥ 0.90` **and** `Jaccard ≥ 0.5`,
across the train pools = **5**.

| band | range | met |
|---|---|---|
| ALIVE | ≥ 30 | no |
| AMBIGUOUS | 10–29 | no |
| **DEAD** | **< 10** | **yes** |

# VERDICT: **DEAD**

The "duplicate-conflict memory" candidate and the "naturally occurring minimal pairs" slot have
no material supply in the train pools. As a validity check on prior retrieval numbers: near-
duplicates exist (49 conservative pairs) but they are label-concordant, so repeated-vote purity
inflation is real but small and does not invalidate previous retrieval results.

## 8. Post-hoc forensic (does NOT change the verdict, and could only weaken it)

Four of the five surviving pairs looked suspicious — `c_img` of exactly 1.0000000. A follow-up
diagnostic on the raw caches (**not** part of the frozen endpoints, run after the verdict was
fixed, and reported because it argues *against* the surviving count):

| dataset | groups of **bit-identical** `img_feats` | items in them | pairs with cos > 0.9999 |
|---|---|---|---|
| HateMM | 3 | 16 | 60 |
| MHC (EN) | 0 | 0 | 0 |
| MHC_zh | 1 | 2 | 2 |
| ImpliHateVid | 4 | 8 | 4 |

HateMM's largest group has **11 videos sharing one byte-identical mean-pooled image vector**
(`hate_video_{76,109,127,298,308}`, `non_hate_video_{25,90,110,308,395,470}`), spanning both
labels. Inspecting the transcripts of the four HateMM conservative pairs shows all of them carry
the same 11-character placeholder transcript `"🎼  🎼  Yeah."` — i.e. these are **failed
decode / failed-ASR artifacts producing a degenerate feature vector**, not reposts.

**Honest restatement of the gate quantity:** of the 5 conservative conflicting pairs, 4 are
degenerate-feature artifacts. The only pair that survives manual inspection as a genuine
near-duplicate label conflict is the MHC one — `0q1PET_IDGc` ("Faggy mpoarty eating worm . One,
two, do it, do it.", label 0) vs `KV49pENhk4c` ("FAGGY MOGARTY EATING A WORM . One, two, do it,
do it.", label 1). That is a real, textbook minimal pair — and there is exactly **one** of it.
The verified count is therefore **1**, not 5. DEAD either way; the frozen number stands as 5.

This also flags a separate, pre-existing data-integrity issue for the project: at least 16 HateMM
train videos have degenerate CLIP image features. That is out of scope for this pilot but should
not be lost.

## 9. Caveats — stated against the result, not for it

1. **A DEAD verdict from a census can be a detector failure rather than an absence.** `c_img` is a
   cosine over 8-frame mean-pooled CLIP features. A repost that is cropped, re-edited, mirrored,
   re-timed or has a different intro will not reach 0.90 on that representation. The census
   therefore measures duplication *as this specific frozen key sees it*, and its recall is unknown
   and unmeasured. The freeze states this ("an upper bound on duplication"); the DEAD verdict is
   only as strong as that key.
2. **The Jaccard ≥ 0.5 filter is very aggressive.** It removed 48 of 53 conflicting pairs at the
   gate. A genuine repost with a re-recorded voice-over, a translated caption, or ASR failure on
   one side will fail it. In particular, `HateMM ↔ ImpliHateVid` lost all 10 of its conflicting
   pairs to this filter, and none of them were visually inspected.
3. **No visual verification was performed.** The freeze says "any surviving claim requires visual
   verification of a sample." None was done here — the raw videos were not opened. §8 substitutes
   *feature-level* and *transcript-level* forensics, which is weaker evidence: it establishes that
   4 of the 5 pairs are artifacts, but it does **not** establish that the remaining MHC pair is a
   genuine repost, only that it is textually near-identical.
4. **The 5 conservative pairs are not 5 independent items.** Four of them share the single item
   `non_hate_video_338`. The effective count of distinct duplicate *clusters* with a conflict is 2
   (one HateMM artifact cluster, one MHC pair) — even further below the ALIVE bar than the raw 5.
5. **My Jaccard tokenizer is a choice the freeze did not make.** Lowercase, `[a-z0-9]+` runs plus
   each CJK / Kana / Hangul character as its own token. For Chinese this is character-unigram
   Jaccard, which is *more* permissive than word-level (higher overlap), so if anything it
   inflates the conservative count for MHC-ZH — and MHC-ZH still contributed 0.
6. **Cross-dataset comparison mixes incommensurate label definitions.** HateMM "hate" and
   ImpliHateVid "implicit hate" are not the same construct, so a cross-dataset "conflict" may be a
   taxonomy difference rather than an annotation conflict. All cross-dataset conflicts died at N4
   anyway, so this did not affect the verdict.
7. **HateClipSeg is entirely missing from the census** (no train assets). If reposts cluster in
   that corpus, this pilot cannot see them.
8. **The null control permutes labels globally**, so it mixes the four datasets' differing base
   rates; the expected-24 figure is therefore an approximation, not an exact null for a
   within-dataset process. Its qualitative conclusion (observed ≪ chance) is robust to that.
9. **`c_txt` was computed and reported descriptively but is used in no endpoint** — the freeze
   makes `c_img` primary and defines N4 on raw-transcript Jaccard, not on `c_txt`. Mean `c_txt`
   over flagged pairs is 0.636 at the gate, which on its own would look like strong text agreement
   and would have been misleading; the raw-token Jaccard (median 0.122) is the honest number.

## 10. Deviations from the freeze (logged, not hidden)

| # | deviation | why / impact |
|---|---|---|
| B-D1 | **HateClipSeg dropped.** The freeze names it, but no train CLIP cache and no `train.jsonl` exist (test-only assets). | Reported plainly in the JSON and above. Nothing fabricated. Reduces pool coverage. |
| B-D2 | **Jaccard tokenizer unspecified in the freeze.** Frozen in code before the run: lowercase, `[a-z0-9]+` \| single CJK/Kana/Hangul char. Both-empty transcripts → Jaccard 0. | See caveat 5. The both-empty rule is conservative (can only lower the count). |
| B-D3 | **Null control added.** The freeze specifies no null for P-B; a 200-permutation label null (seed 20260909) was run. | Additive honesty check. Does not enter the frozen gate. |
| B-D4 | **N3 "MHC conflicting pairs" is ambiguous** (one side MHC vs both sides MHC). Both are reported. | They coincide at every threshold here: no MHC↔non-MHC pair is flagged. |
| B-D5 | **Post-hoc forensic in §8** run after the verdict was fixed. | It only *reduces* the gate quantity (5 → 1 verified). The frozen number reported as the gate quantity remains 5 and the verdict (DEAD) is unchanged in either reading. |

## 11. Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate HateVideo
python idea-stage/pilot_b_dup_conflict_census.py --smoke synthetic   # random features
python idea-stage/pilot_b_dup_conflict_census.py --smoke planted     # 12 planted dups, 8 conflicting
python idea-stage/pilot_b_dup_conflict_census.py --out idea-stage/pilot_b.json
```

Pre-run smokes (both executed before the real run; neither touches real labels or geometry, so
neither reveals a real endpoint):

- **synthetic** (400 Gaussian items, random labels/transcripts): N1@0.90 = 0, conflicting = 0,
  conservative = 0 → verdict DEAD. No spurious duplicates on noise.
- **planted** (same, with 12 injected near-duplicate pairs of which 8 are label-conflicting and
  all have copied transcripts): N1@0.90 = **12**, conflicting = **8**, conservative = **8**,
  verdict DEAD (8 < 10). Exact recovery of the planted ground truth, and the DEAD/AMBIGUOUS/ALIVE
  banding behaves as written.
