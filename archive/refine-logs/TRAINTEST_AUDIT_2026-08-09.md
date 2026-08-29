# train ↔ test unlabelled audit — HateMM, MHC-EN, MHC-ZH, ImpliHateVid

Date 2026-08-09. Authorised by the user's 2026-08-09 ruling unsealing **test inputs**
(features / pixels / text). **Zero labels were read — not one test label, not one train
label.** No model trained, no classification metric computed.

- Script: `scripts/analysis/traintest_audit.py` (label tensor replaced by a `LabelGuard`
  that raises on access; `label` field of every gt jsonl dropped at parse time)
- Log: `logging/runs/traintest_audit/run.log` — 25.7 s, CPU only
- Raw JSON: `artifacts/traintest_audit/audit.json`
- Near-duplicate rules transcribed from `idea-stage/pilot_b_dup_conflict_census.py`
  (P-B freeze): `c_img ∈ {0.85, 0.90, 0.95}`, gate = `c_img ≥ 0.90 AND token-Jaccard ≥ 0.5`,
  same tokenizer regex. P-B's label-conflict axis is dropped (needs labels); the
  cluster/`md5` axis is added.
- Degenerate exclusion list built from `refine-logs/DEGEN_FEATURE_FIX_2026-08-09.md` and
  `refine-logs/EMPTY_TEXT_AUDIT_2026-08-09.md`, re-derived independently here by
  byte-identical feature-row hashing over the **union** of all three splits (the earlier
  scans hashed each split separately, which is why they missed the cross-split groups
  reported in §1).

**Caveat on identifiers.** HateMM ids (`hate_video_*` / `non_hate_video_*`) and
ImpliHateVid ids (`EX_` / `IM_` / `NH_`) encode the class in the id string itself. Ids are
quoted verbatim below because the finding is unreportable otherwise; no computation in this
audit consumed them, and no label file was opened. That the identifiers leak the label at
all is itself worth a line in any paper that hashes or logs ids.

---

## 0. Headline

| dataset | train/val/test | cross-split near-dup **clusters reaching test** | test items involved | byte-identical raw file across splits | drift (test vs train) | verdict |
|---|---|---|---|---|---|---|
| **HateMM** | 744/107/215 | **7** content clusters + 1 black-video cluster | **11 / 215 (5.1 %)**, of which 7 (3.3 %) are real content leaks | **yes — 3 md5 collisions**, incl. one **val↔test** | in-distribution once degenerates removed; **but test carries 2.3× the train rate of empty transcripts (p = 0.001)** | **CONTAMINATED (mild but real, and the composition asymmetry matters more than the duplicates)** |
| **MHC-EN** | 549/80/161 | **0** | **0** | no | in-distribution (all p > 0.19) | **CLEAN** |
| **MHC-ZH** | 579/78/149 | **4** | **4 / 149 (2.7 %)** | no train↔test / val↔test collision (one md5 pair inside train) | in-distribution (all p > 0.28) | **MILDLY CONTAMINATED** |
| **ImpliHateVid** | 1283/325/401 | **6** | **6 / 401 (1.5 %)** | raw media not on this machine — feature-level only | in-distribution (all p > 0.23) | **MILDLY CONTAMINATED (feature-level evidence only)** |

**No dataset here is a leaderboard-level scandal.** The cross-split near-duplicate *rate* is
statistically indistinguishable from each dataset's own within-train duplicate rate — i.e.
the splits were not made carelessly with respect to duplicates, the corpora simply contain
duplicates and the splitter treated every video as i.i.d.:

| dataset | within-train pair rate @ c_img ≥ 0.90 | train↔test | val↔test | ratio (t↔t / within) |
|---|---|---|---|---|
| HateMM | 0.0423 % | 0.0425 % | 0.0869 % | 1.00 |
| MHC-EN | 0.0040 % | 0.0000 % | 0.0000 % | 0.00 |
| MHC-ZH | 0.0317 % | 0.0441 % | 0.0430 % | 1.39 |
| ImpliHateVid | 0.0552 % | 0.0647 % | 0.0553 % | 1.17 |

The one genuinely load-bearing finding is **not** a duplicate: it is HateMM's
**empty-transcript rate asymmetry** (§2.4).

---

## 1. Cross-split near-duplicate census

Method: cosine on L2-normalised CLIP `img_feats` (1024-d) and `text_feats` (768-d), all
train×test and val×test pairs; P-B thresholds; plus (a) byte-identical feature-row hashing,
(b) `md5` of the raw container, (c) connected components at
`c_img ≥ 0.99 OR (c_img ≥ 0.90 AND Jaccard ≥ 0.5)` over the union of the three splits.

### 1.1 HateMM — 7 content clusters reach test, 3 md5 collisions across splits

Raw media: `~/data/HateMM/video/<id>.mp4` (1083 files). Every id below was md5'd.

**md5-identical files sitting in two different splits** (the hardest form of leakage):

| md5 (first 12) | members | splits |
|---|---|---|
| `027373911a19` | `non_hate_video_519`, `hate_video_277` | **val, test** |
| `709c61f53cf3` | `hate_video_86`, `hate_video_188` | train, val |
| `84f69bdbe438` | `hate_video_50`, `hate_video_63`, `non_hate_video_338`, `hate_video_352` | train ×3, **val** |

`027373911a19` is a single 11 291 903-byte, 283.18 s file that the release ships **twice**,
once in val and once in test, under ids of *opposite* class prefixes. `84f69bdbe438` extends
the known train-internal triple (`DEGEN_FEATURE_FIX` §2b) into val — the earlier per-split
scan could not see this.

**Content clusters that reach test** (`md5` differs, content does not):

| # | test member(s) | train / val members | evidence |
|---|---|---|---|
| C1 | `hate_video_102` | `hate_video_333`, `hate_video_264` (train) | `hate_video_333`↔`hate_video_102` **byte-identical `img_feats` row** (‖v‖ 32.8219) despite different md5 and different durations (145.02 s vs 121.02 s). ffmpeg frame pulls at t = 1/20/60 s are **pixel-identical** on both files → same video track, different audio/length. Transcripts differ (two different songs). |
| C2 | `hate_video_88` | `hate_video_217` (train) | c_img 0.9994, c_txt 0.9212, Jaccard 0.702; durations 126.247 s vs 126.200 s. Same song, different encode. **`hate_video_102` (C1) carries the same lyric as C2** — the racist song appears in train (217, 333, 264) and test (88, 102). |
| C3 | `hate_video_356` | `hate_video_60` (train), `hate_video_163`, `hate_video_281` (val) | 4 copies, one per encode; pairwise c_img 0.9931–0.9991, Jaccard 0.89–0.93; durations 120.95–120.98 s. **Content present in all three splits.** |
| C4 | `non_hate_video_541` | `non_hate_video_535`, `non_hate_video_566` (train) | c_img 0.9917 / 0.9516, Jaccard 0.978 / 0.571; 73.86 s vs 73.77 s. |
| C5 | `hate_video_329` | `hate_video_77` (train) | c_img 0.9937, Jaccard 0.000 (both transcripts are stubs) — visual duplicate only. |
| C6 | `non_hate_video_348` | `non_hate_video_123` (val) | c_img 0.9929, Jaccard 0.667. **val↔test.** |
| C7 | `hate_video_277` | `non_hate_video_519` (val) | md5-identical (above). **val↔test.** |
| B | `non_hate_video_140`, `hate_video_273`, `hate_video_295` (+ `hate_video_89` at c_img 0.9903) | 11 train + 2 val | the known **black-video** constant vector (‖v‖ 31.5444). Not content leakage — a constant. `hate_video_89` is *partially* black (frame at t = 2 s is real, frame at t = 30 s is uniformly 0). |

Distinct **test** videos touched: 7 content (`hate_video_{102,88,356,329,277}`,
`non_hate_video_{541,348}`) + 4 degenerate (`non_hate_video_140`, `hate_video_{273,295,89}`)
= **11 / 215 = 5.1 %**.

P-B-gate counts (`c_img ≥ 0.90 AND Jaccard ≥ 0.5`): train↔test **4 pairs**, val↔test
**4 pairs**, train↔val **16 pairs**. Excluding degenerate rows: 4 / 3 / 4.

**Text-side cross-split duplicates.** Beyond the 74 empty transcripts, HateMM ships literal
stub transcripts that collapse to identical text vectors *across splits*:
`"Yeah."` (5 train + 1 val + 1 test = `non_hate_video_63`), `"Oh."` (2 train + 1 test =
`non_hate_video_115`), `"🎼 ."` (1 train + 1 test = `non_hate_video_221`), `"🎼  🎼"`
(2 train + 1 val). None of these carry information; they matter only because any retrieval
memory keyed on CLIP text will return them as cosine-1.0 neighbours.

### 1.2 MHC-EN — clean

Raw media: `~/data/Multihateclip/English/video_mp4` (792 files).
**Zero** train↔test and zero val↔test pairs at `c_img ≥ 0.90`. Maximum cross-split image
cosine: **0.8988** (train↔test), **0.8718** (val↔test) — below the P-B flag threshold
entirely. Zero byte-identical feature rows anywhere, zero empty transcripts in any split.
Four near-dup clusters exist inside the corpus, but only one crosses a split boundary and it
is train↔val (`1PEr_STq3jk` ↔ `c_Ewd_d6mho`, c_img 0.9789, Jaccard 0.778), never touching test.

### 1.3 MHC-ZH — 4 test items in cross-split clusters

Raw media: `~/data/Multihateclip/Chinese/video` (814 files). No md5 collision crosses a
split boundary (the one md5 collision, `e8e9e9370645` = `BV1ka4y1m7Ti` / `BV1UT4y1p7WS`, is
train-internal and already known).

| # | test member | train / val members | evidence |
|---|---|---|---|
| Z1 | `BV1Pe411Y7c5` | `BV17w411g7iA`, `BV1Jp4y1R7Au`, `BV1A94y1M7bQ`, `BV1UT4y1p7WS`, `BV1dC4y1N73S`, `BV1ka4y1m7Ti` (train ×6) | 7-member cluster; test↔train c_img 0.907–0.908, Jaccard 0.849. Two md5-identical members inside train. |
| Z2 | `BV15E411a7Jd` | `BV1US4y1k7tq` (train), `BV16E411b7B6` (val) | c_img 0.9162 / 0.9672, Jaccard 0.778 / 0.478. Spans all three splits. |
| Z3 | `BV1va4y1m72C` | `BV1hC4y1Z7tH` (train) | c_img 0.9919, c_txt 0.9843, Jaccard 0.904. Different md5, same content. |
| Z4 | `BV12C4y1m7ic` | `BV12u4y1c7B8` (val) | c_img 0.9936 and **byte-identical `text_feats` row** (c_txt = 1.0000, Jaccard 1.0). **val↔test.** |

P-B-gate counts: train↔test **4**, val↔test **1**, train↔val **2**.
Test items touched: **4 / 149 = 2.7 %**.

### 1.4 ImpliHateVid — 6 test items, feature-level evidence only

**Raw videos are not on this machine and not in the B2 backup** (`DEGEN_FEATURE_FIX` §5
item 5), so no md5 can be produced. All statements below are feature-level.

| # | test member | train member | evidence |
|---|---|---|---|
| I1 | `NH_180` | `EX_83` | **byte-identical `img_feats` row**, c_txt 0.9787, Jaccard 0.926. Same video under two ids in train and test. |
| I2 | `IM_164` | `IM_397` | c_img 0.9882, c_txt 0.9743, Jaccard 0.959. |
| I3 | `IM_243` | `EX_233` | c_img 0.9657, Jaccard 0.730. |
| I4 | `EX_258` | `EX_136` | c_img 0.9326, Jaccard 0.535. |
| I5 | `IM_285` | `IM_409` | c_img 0.9300, Jaccard 0.543. |
| I6 | `NH_83` | `NH_666` | c_img 0.9917, Jaccard 0.240 (visual duplicate only). |

Plus one train↔val byte-identical pair (`NH_727` / `NH_400`, c_img = c_txt = 1.0).
P-B-gate counts: train↔test **5**, val↔test **0**, train↔val **2**.
Test items touched: **6 / 401 = 1.5 %**.

The raw count of pairs at `c_img ≥ 0.90` is large (train↔test 333, val↔test 72) but that is
**global geometry, not leakage**: the within-train rate at the same threshold is 0.0552 %
against a cross-split 0.0647 %, and only 5 of the 333 survive the Jaccard gate.

---

## 2. Distribution drift (zero labels)

Keys: `img` = L2(CLIP image), `txt` = L2(CLIP text), `concat` = `[L2(img), L2(txt)]/√2`.
Two statistics, both with a 200-draw split-membership permutation null:
**energy distance** (Euclidean, unbiased) and **unbiased MMD²** with an RBF kernel at the
median heuristic. Plus a **size-matched nearest-neighbour control**: for each of 20 random
partitions, `n_test` train items are held out, the remaining `n_train − n_test` form the
reference bank, and the NN cosine distance distribution of the **test** queries is compared
(KS) against that of the **held-out train** queries. Both variants reported: with and
without the degenerate rows named in `DEGEN_FEATURE_FIX` / `EMPTY_TEXT_AUDIT`
(black-video constant image rows + whitespace-only transcripts + their byte-identical
feature-row groups).

| dataset | key | variant | n_tr/n_te | energy | p | MMD² | p | NN d(test) | NN d(ctrl) | KS |
|---|---|---|---|---|---|---|---|---|---|---|
| HateMM | img | with | 744/215 | 0.00049 | 0.224 | 0.00030 | 0.244 | 0.2625 | 0.2659 | 0.069 |
| HateMM | **txt** | **with** | 744/215 | **0.00479** | **0.005** | **0.00291** | **0.005** | 0.1969 | 0.2140 | 0.084 |
| HateMM | concat | with | 744/215 | 0.00176 | **0.005** | 0.00126 | **0.005** | 0.2775 | 0.2857 | 0.084 |
| HateMM | img | excl | 675/181 | 0.00065 | 0.189 | 0.00043 | 0.194 | 0.2635 | 0.2725 | 0.089 |
| HateMM | **txt** | **excl** | 675/181 | **0.00019** | **0.348** | 0.00017 | 0.323 | 0.2289 | 0.2400 | 0.094 |
| HateMM | concat | excl | 675/181 | 0.00040 | 0.239 | 0.00032 | 0.219 | 0.2938 | 0.3025 | 0.094 |
| MHC-EN | img | with = excl | 549/161 | 0.00054 | 0.199 | 0.00038 | 0.194 | 0.3404 | 0.3266 | 0.098 |
| MHC-EN | txt | with = excl | 549/161 | −0.00014 | 0.493 | −0.00009 | 0.502 | 0.4017 | 0.3906 | 0.101 |
| MHC-EN | concat | with = excl | 549/161 | 0.00014 | 0.333 | 0.00008 | 0.363 | 0.4155 | 0.4024 | 0.106 |
| MHC-ZH | img | with | 579/149 | −0.00060 | 0.746 | −0.00044 | 0.736 | 0.2105 | 0.2152 | 0.093 |
| MHC-ZH | txt | with | 579/149 | 0.00022 | 0.318 | 0.00007 | 0.343 | 0.0526 | 0.0614 | 0.090 |
| MHC-ZH | concat | with | 579/149 | −0.00017 | 0.562 | −0.00010 | 0.493 | 0.1527 | 0.1605 | 0.113 |
| MHC-ZH | img | excl | 577/149 | −0.00048 | 0.692 | −0.00034 | 0.672 | 0.2105 | 0.2171 | 0.088 |
| MHC-ZH | txt | excl | 577/149 | 0.00028 | 0.289 | 0.00016 | 0.343 | 0.0525 | 0.0626 | 0.091 |
| MHC-ZH | concat | excl | 577/149 | −0.00007 | 0.488 | 0.00001 | 0.433 | 0.1525 | 0.1625 | 0.122 |
| ImpliHateVid | img | with | 1283/401 | 0.00008 | 0.318 | 0.00002 | 0.348 | 0.2507 | 0.2547 | 0.060 |
| ImpliHateVid | txt | with | 1283/401 | −0.00046 | 0.970 | −0.00035 | 0.965 | 0.2379 | 0.2354 | 0.064 |
| ImpliHateVid | concat | with | 1283/401 | −0.00016 | 0.657 | −0.00012 | 0.657 | 0.2871 | 0.2865 | 0.069 |
| ImpliHateVid | img | excl | 1268/400 | 0.00010 | 0.239 | 0.00004 | 0.294 | 0.2514 | 0.2573 | 0.061 |
| ImpliHateVid | txt | excl | 1268/400 | −0.00048 | 0.955 | −0.00037 | 0.950 | 0.2386 | 0.2367 | 0.069 |
| ImpliHateVid | concat | excl | 1268/400 | −0.00016 | 0.667 | −0.00012 | 0.672 | 0.2879 | 0.2880 | 0.061 |

### 2.1 Answer to "is test an i.i.d. draw from the train distribution?" — yes, everywhere

Two-sample KS 5 % critical values for the matched design are 0.131 (HateMM, n = m = 215),
0.152 (MHC-EN), 0.158 (MHC-ZH), 0.096 (ImpliHateVid). **Every observed KS is below its
critical value** (max 0.122, MHC-ZH concat). Test queries are neither farther from the train
bank than held-out train queries are, nor closer: the mean NN-distance gap is between
−0.017 and +0.014 in all 24 cells, and it is *negative* (test slightly closer) in 20 of 24 —
which is the expected sign given the handful of duplicates in §1, not a distribution shift.

### 2.2 The one significant drift signal is entirely a degeneracy artefact

HateMM `txt` is the only cell that rejects the permutation null (energy 0.00479, p = 0.005,
z = +6.0 σ; MMD² 0.00291, p = 0.005). Removing the degenerate rows collapses it to
0.00019 / p = 0.348 — a **25× reduction**. The `concat` signal (p = 0.005 → 0.239) is the
same effect diluted. Nothing about HateMM's *content* distribution differs between train and
test; what differs is how much of each split is a constant vector.

### 2.3 Image side is clean everywhere

No `img` cell is significant in any dataset, with or without exclusions (min p = 0.189). The
black-video rate is statistically identical across HateMM splits (train 11/744 = 1.48 %,
test 3/215 = 1.40 %, Fisher p = 1.00).

### 2.4 The load-bearing number: HateMM test has 2.3× the train rate of empty transcripts

| split | whitespace-only transcripts | rate |
|---|---|---|
| train | 39 / 744 | 5.24 % |
| val | 9 / 107 | 8.41 % |
| test | **26 / 215** | **12.09 %** |

Fisher exact, train vs test: **OR = 2.49, p = 0.0010** (train vs val p = 0.18, val vs test
p = 0.35). This is a real, significant compositional asymmetry in the *official* HateMM
split, and it is the whole of the drift signal in §2.2.

Why it matters, combining with `EMPTY_TEXT_AUDIT_2026-08-09.md` §2d (which established, on
train, that this cluster is 93 % one class): **12.1 % of the HateMM test set is a single
point in CLIP text space, and that point sits in a high-purity stratum.** Any read-out with a
CLIP-text component gets those 26 items nearly free, and it gets *proportionally more* of
them on test than the train distribution would predict. Every HateMM test number in this
project that includes a CLIP text channel is inflated by an amount this audit cannot
quantify without touching labels — but the direction is unambiguous, and it is the opposite
of the image-side conclusion in `DEGEN_FEATURE_FIX` §4.

This supersedes nothing in the earlier audits; it adds the split-composition dimension they
did not measure. It also **strengthens the two RECHECK items** already open in
`EMPTY_TEXT_AUDIT` §3c: the "HateMM is text-carried" 0.847-vs-0.826 margin and the
CLIP-vs-Qwen +4.2 encoder delta both live on a test set with double the empty-transcript
density of the training pool.

---

## 3. Test-side degeneracy census (completing the train/val-only scans)

Byte-identical feature-row hashing over the union of all splits, per dataset. "Constant
image rows" counts every row sharing its exact `img_feats` bytes with at least one other row
anywhere in the dataset; "constant text rows" likewise for `text_feats`.

| dataset | split | n | constant `img` rows | whitespace-only transcript | constant `text` rows | excluded from §2 "excl" |
|---|---|---|---|---|---|---|
| HateMM | train | 744 | **18** | 39 | 61 | 69 |
| HateMM | val | 107 | **5** | 9 | 16 | 16 |
| HateMM | **test** | 215 | **5** | **26** | **31** | **34** |
| MHC-EN | train | 549 | 0 | 0 | 0 | 0 |
| MHC-EN | val | 80 | 0 | 0 | 0 | 0 |
| MHC-EN | **test** | 161 | **0** | **0** | **0** | **0** |
| MHC-ZH | train | 579 | 2 | 0 | 4 | 2 |
| MHC-ZH | val | 78 | 0 | 0 | 1 | 0 |
| MHC-ZH | **test** | 149 | **0** | **0** | **1** | **0** |
| ImpliHateVid | train | 1283 | 10 | 5 | 12 | 15 |
| ImpliHateVid | val | 325 | 1 | 0 | 1 | 1 |
| ImpliHateVid | **test** | 401 | **1** | **0** | **0** | **1** |

### 3.1 Corrections and additions to the earlier census

`DEGEN_FEATURE_FIX_2026-08-09.md` §1 hashed each split **separately**, so it reported
HateMM train 16 / val 2 / test 3 constant image rows. Hashing across the union finds
**more**, because several duplicate pairs straddle a split boundary:

- HateMM train **18** (not 16): adds `hate_video_333` (pairs with test `hate_video_102`) and
  `hate_video_86` (pairs with val `hate_video_188`).
- HateMM val **5** (not 2): adds `hate_video_352` (`84f69bdbe438`, pairs with the train
  triple), `hate_video_188`, `non_hate_video_519`.
- HateMM test **5** (not 3): adds `hate_video_102` and `hate_video_277`.
- ImpliHateVid val **1** and test **1** (not 0/0): `NH_400` and `NH_180`.

The earlier verdict is unaffected — none of these are broken vectors, they are faithful
encodings of duplicated media — but the *duplicate* half of the census was undercounted by
construction. Anything that consumes `degen_flags` should be regenerated with union hashing.

### 3.2 Whitespace-only vs "no alphanumeric token"

The whitespace-only counts above reproduce `EMPTY_TEXT_AUDIT` exactly (39/9/26 HateMM;
ImpliHateVid train 5). A slightly wider definition — transcripts with no
alphanumeric/CJK token at all, i.e. emoji- or punctuation-only, such as `"🎼  🎼"` and
`"🎼 ."` — gives HateMM 51/11/29 and ImpliHateVid train 6. Those extra rows do **not** collapse
to the empty-text constant (CLIP BPE tokenises emoji), but they do form their own small
cosine-1.0 groups that cross splits (§1.1).

### 3.3 HateClipSeg

Out of scope for this audit (it has no train/test split of its own — all 395 rows are one
partition), and its text channel is already known to be 395/395 constant
(`EMPTY_TEXT_AUDIT` §1). Not re-measured here.

---

## 4. Verdicts

**HateMM — CONTAMINATED (mild duplicate leakage; significant composition asymmetry).**
Three md5-identical files span split boundaries, one of them **val↔test** under opposite
class-prefix ids; seven content clusters put near-identical video in both train (or val) and
test, covering 3.3 % of the test set (5.1 % including the black-video constant). None of
this is unusual for the corpus — the cross-split duplicate rate exactly equals the
within-train rate — so HateMM numbers are not "virtual". The material finding is §2.4: the
official test split carries 12.1 % whitespace-only transcripts against 5.2 % in train
(p = 0.001), and that stratum is a single constant vector in a high-purity class region.
Any HateMM test figure with a CLIP text channel is inflated by an unquantified amount.

**MHC-EN — CLEAN.** Zero cross-split near-duplicates at any P-B threshold (max cross-split
image cosine 0.8988), zero degenerate rows in any split, no drift on any key. This is the
only one of the four that would survive a hostile reviewer's duplicate audit unamended.

**MHC-ZH — MILDLY CONTAMINATED.** Four test videos (2.7 %) sit in clusters that also contain
train or val members, including one val↔test pair with a **byte-identical text vector**
(`BV12u4y1c7B8` / `BV12C4y1m7ic`). No raw-file md5 collision crosses a split. Distribution
is in-distribution on every key. Effect size is too small to move a corpus-level metric, but
the four ids should be named in any per-item error analysis.

**ImpliHateVid — MILDLY CONTAMINATED, evidence feature-level only.** Six test items (1.5 %)
duplicate train content, one of them (`EX_83` / `NH_180`) with a **byte-identical image
feature row** and 0.926 transcript Jaccard — i.e. the same video shipped in train and test
under ids whose prefixes assert different classes. This cannot be confirmed at the source
because the raw corpus exists neither locally nor in the B2 backup; re-staging the corpus is
the only way to close it. No drift on any key.

## 5. Handed forward

1. Regenerate `degen_flags` with **union-of-splits** row hashing (§3.1); the per-split scan
   structurally cannot see cross-split duplicate groups.
2. HateMM: decide whether the 26 empty-transcript test items are reported separately. The
   cheapest honest fix is a stratified test table (with / without the constant-text stratum),
   which needs labels and therefore a separate authorisation.
3. HateMM `84f69bdbe438` now has **4** copies (3 train + 1 val) and `hate_video_105`/
   `hate_video_52` join the same visual cluster from val — the DUP_SOURCE table in
   `DEGEN_FEATURE_FIX` §2b should be extended.
4. Any retrieval/memory method evaluated on HateMM must exclude, or at minimum report, the
   7 content-duplicate test items in §1.1 — a kNN memory retrieves its own near-copy first.
5. ImpliHateVid raw media re-staging remains the blocker for source-level confirmation of I1.
6. Note in the paper that HateMM and ImpliHateVid ids encode the label; never expose ids to
   a model, a hash bucket, or a sort key.
