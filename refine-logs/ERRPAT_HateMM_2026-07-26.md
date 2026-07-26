# ERRPAT — HateMM forensic error-pattern analysis (2026-07-26)

**Status:** forensic diagnostics only. Zero GPU, zero training of any deployed arm, no prereg, no
promotion, no selection or tuning decision derived from anything below. Repo sha at analysis time
`172371e` (working tree dirty; nothing committed by this analysis).

---

## 0. PROVENANCE HEADER

### 0.1 Artifact status: PROXY, not the bit-exact floor

The deployed HateMM floor is job **13241** (per-dataset LoRA-**curric** Qwen2.5-VL-7B encoder →
dual-stream 3584-d mean-pooled → RGCL align head, Hadamard fuse, triplet+BCE hybrid, FAISS hard
negatives → top-20 rank-weighted signed-cosine kNN vote over own-train memory, V=744).

Per-item predictions for job 13241 are **UNRECOVERABLE**:

* the floor run was launched with `--save_embed` at its default `False`
  (`src/model/evaluate_rac.py:196-198` and `:481-484` gate the per-item pickle on `args.save_embed`;
  the banked `Namespace(...)` line in every 13241 trainlog records `save_embed=False`), so no
  `*_retrieval_logging_dict.pkl` was ever written; a filesystem-wide search finds none;
* all 6 floor head checkpoints are deleted (**F78**).

So this report analyses a **CPU-reconstructed proxy**: the byte-identical `run_rac.py` command from
`scripts/slurm/enc3seed_lora_curric.sbatch` re-run with only `--device cpu --save_embed True
--group_name RAC_errpat_proxy --output_path <scratch>` changed, on the same banked LoRA-curric
feature caches. Nothing in `logging/` or `data/` was written or modified. **The proxy is never
presented as the floor.** Cost: 52 s wall per seed on 8 CPUs.

### 0.2 Proxy ↔ floor parity (re-read from primary trainlogs at report time)

| protocol | seed | floor 13241 acc / mF1 (ep) | proxy acc / mF1 (ep) |
|---|---|---|---|
| val-sel | 0 | 0.8791 / 0.8730 (29) | 0.8791 / 0.8730 (25) |
| val-sel | 1 | 0.8744 / 0.8678 (14) | 0.8744 / 0.8684 (15) |
| val-sel | 2 | 0.8791 / 0.8724 (10) | 0.8791 / 0.8730 (29) |
| final | 0 | 0.8791 / 0.8730 (29) | 0.8698 / 0.8632 (29) |
| final | 1 | 0.8791 / 0.8724 (29) | 0.8791 / 0.8735 (29) |
| final | 2 | 0.8791 / 0.8724 (29) | 0.8791 / 0.8730 (29) |

3-seed means — floor **val-sel 0.8775 / 0.8711**, **final 0.8791 / 0.8726**; proxy val-sel
0.8775 / 0.8715, final 0.8760 / 0.8699. **proxy − floor = +0.0000 / +0.0004 (val-sel),
−0.0031 / −0.0027 (final)** — i.e. the val-sel protocol is exact at 4 dp and the final-epoch
protocol differs by 0.67 test items per seed (1 item = 0.00465 on n=215). Residual = CUDA-vs-CPU
dropout RNG streams during training (weight init and dataloader shuffling both use the CPU
generator and are therefore shared). Selected epochs differ (proxy 25/15/29 vs floor 29/14/10) as
expected under a different training trajectory.

**Machinery parity (independent of the proxy question):** the vote replay in
`errpat_hatemm_forensics.py` reproduces every proxy trainlog acc **and** mF1 **bit-exact at 4 dp**
in 6/6 seed×protocol cells (`vote_replay_parity` in the OUT json), and the head-space
recomputation of the fused vote from the saved checkpoint reproduces the saved logging-dict vote
exactly (`stream_error_set_overlap[*].acc.fused == .acc.deployed`, 6/6). The kNN operator, the
weight vector `[20..1]`, the signed-cosine map and the `sigmoid(v) ≥ 0.5 ⟺ v ≥ 0` decision are all
replayed faithfully.

### 0.3 Inputs (all read-only)

| file | nature |
|---|---|
| `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog` | EXACT — banked floor primary logs |
| `data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` | EXACT — banked pre-head features, md5 `5c6b0edab4eda147fa36853cb9c167ef` |
| `data/CLIP_Embedding/HateMM/dev_seen_…-LoRA-curric_HF.pt` | EXACT, md5 `cb3deb76e792b7733c05d61cf60b028d` |
| `data/CLIP_Embedding/HateMM/test_seen_…-LoRA-curric_HF.pt` | EXACT, md5 `ba12ceba4e416bd21b2eef8a4c6c1f66` |
| `data/gt/HateMM/{train,val,test}.jsonl` | EXACT — labels + the transcript string actually fed to the text stream (`generate_VideoMLLM_embedding_lora_HF.py:432` reads `item["text"]`) |
| `data/gt/HateMM/hate_spans.json` | EXACT — gold duration + hate spans (1083 entries) |
| `data/gt/HateMM/HateMM_annotation.csv` | EXACT — target-group column |
| `<scratch>/errpat/proxy_s{0,1,2}.trainlog` | PROXY |
| `<scratch>/errpat/Retrieval/HateMM/RAC_errpat_proxy/*/testepoch_{25,15,29}_retrieval_logging_dict.pkl` | PROXY — per-item top-20 neighbours, ids, sims, labels |
| `<scratch>/errpat/…/ckpt/epoch_model_{25,15,29}_*.pt` | PROXY — head weights for stream forensics |

Scratch root: `/data/jehc223/home/tmp/claude-135258174/-data-jehc223-RGCL/e8f03e41-3e21-4cea-b12c-29207373bfca/scratchpad/errpat`.

### 0.4 Scripts and machine-readable outputs (written by this analysis)

| script | output |
|---|---|
| `scripts/analysis/errpat_hatemm_forensics.py` | `scripts/analysis/errpat_hatemm_forensics_OUT.json`, `scripts/analysis/errpat_hatemm_peritem.csv` |
| `scripts/analysis/errpat_hatemm_ceilings.py` | `scripts/analysis/errpat_hatemm_ceilings_OUT.json` |
| `scripts/analysis/errpat_hatemm_clusters.py` | `scripts/analysis/errpat_hatemm_clusters_OUT.json` |

Split sizes re-read at report time: train 744, val 107, test 215 (86 hate / 129 non-hate).

---

## 1. ERROR COUNTS

Deployed (fused) kNN vote, proxy heads.

| cell | epoch | n_err | FP | FN | acc |
|---|---|---|---|---|---|
| seed0 val-sel | 25 | 26 | 11 | 15 | 0.8791 |
| seed1 val-sel | 15 | 27 | 12 | 15 | 0.8744 |
| seed2 val-sel | 29 | 26 | 11 | 15 | 0.8791 |
| seed0 final | 29 | 28 | 12 | 16 | 0.8698 |
| seed1 final | 29 | 26 | 12 | 14 | 0.8791 |
| seed2 final | 29 | 26 | 11 | 15 | 0.8791 |

**FN-heavy by ~25%** in every cell (14-16 FN vs 11-12 FP). 3-seed mean recall — hate **0.8256**
(both protocols), non-hate **0.9121** (val-sel) / **0.9096** (final). The deployed vote's test ROC
AUC is **0.9331** (final-epoch, 3-seed-mean vote), i.e. the ranking is far better than the
thresholded decision, which §2.1 shows is nonetheless not exploitable.

### 1.1 Seed stability — the error set is essentially fixed

| protocol | wrong 0/3 | 1/3 | 2/3 | 3/3 | union ever wrong |
|---|---|---|---|---|---|
| val-sel | 186 | 3 | 2 | **24** | 29 |
| final | 187 | 1 | 2 | **25** | 28 |

**24-25 of ~26-28 errors (≈89-93%) are wrong in all three seeds.** Only 3 (val-sel) / 1 (final)
items are single-seed flips. Head-seed variance touches ≤5 items out of 215. Consequently every
cluster below is a *representation/geometry* property, not a training-noise property.

### 1.2 val-sel vs final-epoch flips

Predictions changed between the two protocols on **2 items (seed0), 5 items (seed1), 0 items
(seed2)**. Seed2's val-sel epoch *is* 29 in the proxy, hence 0. The protocol choice moves ≤5
items; it does not reshape the error set.

---

## 2. VOTE STRUCTURE OF THE ERRORS

All from the saved per-item top-20 neighbour lists (`errpat_hatemm_peritem.csv`).

| statistic (seed0, final ep29) | errors (≥2/3 seeds, n=27) | always correct (n=187) |
|---|---|---|
| median \|vote\| (3-seed mean vote) | 0.7267 | 0.9873 |
| median top-1 neighbour cosine | 0.999852 | 0.999976 |
| top-1 neighbour carries the TRUE label | **7.4 %** | 95.2 % |
| median rank-weighted top-20 purity toward TRUE label | **0.1667** | 1.0000 |
| median rank of first TRUE-label neighbour | 3.0 | 1.0 |
| errors with **zero** TRUE-label neighbour in the top-20 | **6 / 27** | 0 |

Two structural facts:

1. **Nearest-neighbour distance is not a usable signal.** Cosine is saturated at ~0.9999 for both
   errors and correct items (the head space is collapsed onto a narrow cone). Distance-based
   abstention/gating has essentially no dynamic range to work with.
2. **The retrieved neighbourhood is majority-wrong for almost every error.** Unweighted top-20
   purity toward the true label is <0.5 for **24-27 of 26-28** errors in every cell, mean purity
   **0.1590 (val-sel) / 0.1599 (final)**, with purity <0.25 for 21/27. Six errors have *no*
   correct-label neighbour at all in the top-20 — no vote rule, no re-weighting, no calibration
   over the retrieved set can reach them.

### 2.1 Threshold-reachable vs vote-locked (F66 currency)

An error is **threshold-reachable** iff some global threshold τ on the vote classifies it
correctly *without* dropping overall test accuracy below the deployed value; otherwise
**vote-locked**.

| cell | n_err | threshold-reachable | vote-locked | best test-fitted τ (ORACLE) | acc at that τ |
|---|---|---|---|---|---|
| s0 val-sel | 26 | 2 | 24 | +0.0956 | 0.8837 |
| s1 val-sel | 27 | 9 | 18 | +0.0478 | 0.8884 |
| s2 val-sel | 26 | 7 | 19 | −0.5020 | 0.8837 |
| s0 final | 28 | 14 | 14 | +0.3364 | 0.8930 |
| s1 final | 26 | 6 | 20 | −0.5990 | 0.8884 |
| s2 final | 26 | 7 | 19 | −0.5020 | 0.8837 |

Even a **test-fitted** threshold (forensic oracle — not legally available) recovers a 3-seed mean
of only **+0.0078 acc (val-sel) / +0.0124 (final)**. Legally fitted thresholds recover nothing:

* dev-fitted by accuracy: **+0.0000 (val-sel) / +0.0016 (final)** acc;
* dev-fitted by macro-F1 (final-epoch head): **+0.0016 acc / +0.0031 mF1**;
* train-LOO-fitted logistic recalibration on the vote: **−0.0016 acc / −0.0017 mF1**.

**Calibration is measured dead as a lever.** This is the per-item confirmation of F66's arithmetic
(91-98 % of oracle headroom reachable only by banned per-item selection).

---

## 3. STREAM FORENSICS

Two lenses, both CPU-cheap on the banked dual streams.

**(a) Untrained raw-encoder key space** (L2-normalised 3584-d, same top-20 rank-weighted
signed-cosine vote, own-train memory): image-only **acc 0.7209 / mF1 0.7069**, text-only
**0.8233 / 0.8208**, L2-concat of both **0.8140 / 0.8118**.

**(b) Trained head space** (proxy head, kNN over the normalised `img_proj` / `text_proj` /
Hadamard-fused MLP embedding), 3-seed means:

| key space | val-sel acc / mF1 | final acc / mF1 |
|---|---|---|
| deployed (= recomputed fused) | 0.8775 / 0.8715 | 0.8760 / 0.8699 |
| text stream only | **0.8822 / 0.8759** | **0.8853 / 0.8797** |
| image stream only | 0.7426 / 0.7341 | 0.7411 / 0.7318 |

Text-only − deployed = **+0.0047 acc (val-sel, sign 2/3) / +0.0093 acc (final, sign 3/3)**. Both
are inside the project's ±0.014 seed-noise band and far under the +0.030 house bar; see §5 for how
this should and should not be used.

### 3.1 Cross-tab of the deployed errors by stream correctness (head space)

| cell | both streams wrong | text right, image wrong | image right, text wrong | fusion lost it (both right) |
|---|---|---|---|---|
| s0 val-sel | 13 | 1 | 11 | 1 |
| s1 val-sel | 12 | 1 | 12 | 2 |
| s2 val-sel | 12 | 1 | 12 | 1 |
| s0 final | 14 | 3 | 11 | 0 |
| s1 final | 12 | 0 | 13 | 1 |
| s2 final | 12 | 1 | 12 | 1 |

Raw-encoder-space cross-tab on the consensus (≥2/3 seeds wrong) set, final: both wrong 13, text
right 0, image right 9, fusion lost it 5 (n=27).

### 3.2 Error-SET arithmetic — the apparent image complementarity is pure selection

| cell | n_err fused | n_err text | n_err image | fused errors text fixes | new text errors | fused errors image fixes | new image errors | oracle pick {fused,text} | oracle pick {fused,text,image} |
|---|---|---|---|---|---|---|---|---|---|
| s0 final | 28 | 25 | 57 | 3 | 0 | 11 | 40 | 0.8837 | 0.9349 |
| s1 final | 26 | 25 | 55 | 1 | 0 | 14 | 43 | 0.8837 | 0.9442 |
| s2 final | 26 | 24 | 55 | 2 | 0 | 13 | 42 | 0.8884 | 0.9442 |

The image stream *does* hold per-item information the fusion misses (it fixes 11-14 of the
deployed errors) but it breaks 40-43 items that the fusion gets right. Net is deeply negative
(−0.13 acc). Realising the 11-14 requires an oracle per-item channel choice, which is exactly the
object F47 killed (per-item cross-channel routing dead at read *and* realisable ceiling) and F66
priced as selection-locked. The oracle 3-way pick reaches **0.9411 mean (final)** — i.e.
**+0.065 of selection-locked headroom**, in the same currency as F66's +0.0776. This is the
per-item confirmation of F86's PID reading (no synergy term; U1 image uniqueness 0.0000 on 5/6
cells) and of F85's concat null: fusion can only recombine, and the recombination that would pay
is per-item selection.

---

## 4. THE DOMINANT COVARIATE: TRANSCRIPT VOLUME

This is the load-bearing new finding of the analysis.

### 4.1 Covariates split by error type (consensus ≥2/3 seeds, final protocol)

| group | n | median transcript words | median duration s | median gold-span fraction | empty-transcript rate |
|---|---|---|---|---|---|
| **FN — hate missed** | 15 | **85** | 94.4 | 0.6316 | 0.1333 |
| hate caught | 70 | **227** | 145.0 | 0.7322 | 0.0000 |
| **FP — non-hate flagged** | 12 | **171.5** | 154.2 | 0 | 0.0000 |
| non-hate correct | 117 | **47** | 85.6 | 0 | 0.2393 |

Missed hate has ~2.7× less speech than caught hate; flagged non-hate has ~3.6× more speech than
correctly-passed non-hate. (The first pass pooled classes and looked flat on `n_words`; splitting
by class is what exposes the effect — the two classes move in opposite directions and cancel.)

### 4.2 Accuracy stratified by transcript volume × class (3-seed mean, final)

| transcript words | hate: n / acc | non-hate: n / acc |
|---|---|---|
| 0-1 | 2 / **0.0000** | 28 / **1.0000** |
| 2-50 | 13 / 0.6154 | 34 / 0.9706 |
| 51-150 | 22 / 0.7576 | 15 / 0.8667 |
| 151-400 | 34 / 0.9216 | 33 / 0.8283 |
| 401+ | 15 / **1.0000** | 19 / 0.8421 |

Monotone increasing for hate, monotone decreasing for non-hate. Per-seed values are identical or
near-identical in every bin (see `stratified_by_transcript_volume` in the clusters OUT json), so
this is not seed noise.

**Empty-transcript behaviour is absolute:** of the 30 test items with ≤1 transcript word, the model
predicts non-hate for **all 30 in all 3 seeds, under both protocols** (`pred_hate_count_over_3seeds
= 0`). The 2 empty-transcript hate videos (`hate_video_1`, `hate_video_329`) are therefore missed
6/6 times; the 28 empty-transcript non-hate videos are correct 84/84 times.

### 4.3 Mechanism — the memory bank's length-conditional class prior

`data/gt/HateMM/train.jsonl` (V=744, overall P(hate)=0.4005):

| transcript words | n train rows | hate rows | P(hate \| bin) |
|---|---|---|---|
| 0-1 | 73 | 8 | **0.1096** |
| 2-50 | 188 | 55 | 0.2926 |
| 51-150 | 136 | 52 | 0.3824 |
| 151-400 | 217 | 111 | 0.5115 |
| 401+ | 130 | 72 | **0.5538** |

Retrieval is strongly length-organised: Spearman ρ between a query's word count and the median
word count of its top-20 retrieved train rows = **0.5817 (p = 7.4e-21, n=215)**. The vote therefore
inherits the bank's length-conditional prior — a speech-poor hate video lands in a region whose
base rate of hate is 11-29 %, and the rank-weighted signed-cosine sum goes negative before any
content evidence is consulted.

**Calibrated statement of scope (important):** transcript length is a *bias direction in the
score*, not the main signal. Length alone as a hate score gives test AUC **0.6570** (dev 0.6891,
train 0.6674) and a train-fitted length threshold (`n_words ≥ 182.5`) gives test acc **0.6279 /
mF1 0.6139** — far below the deployed AUC **0.9331** and acc 0.8760. But within each class the
deployed vote is still significantly length-correlated (Spearman ρ = **+0.3106, p = 0.0036** within
hate; **+0.2003, p = 0.0228** within non-hate; +0.3507 overall), and that residual correlation is
what produces the two monotone curves in §4.2.

**And it is not post-hoc correctable.** A logistic model on `{vote, log(1+n_words)}` fitted on the
train LOO votes gives a correctly-signed but tiny length coefficient (−0.0848 / −0.0830 / −0.0785
across seeds) and moves test accuracy by **−0.0016 acc / −0.0017 mF1**; fitted on dev the length
coefficient flips sign (+0.24, dev noise) and the effect is **+0.0000 / −0.0002**. The bias lives
in the retrieval geometry, not in a monotone miscalibration of the score.

### 4.4 Target-group distribution (descriptive)

`Blacks` 14 of 27 consensus errors on 87 test items; `Others` 6/65; the remaining 7 spread over
`Jews`, `Muslims`, `Whites` and the bracketed-list variants of those labels. No group is
disproportionately hard once the small counts are accounted for; the annotation column mixes two
formats (bare string vs Python-list repr), which is why the table in the OUT json has duplicate
keys.

---

## 5. NAMED ERROR CLUSTERS

Definitions are **descriptive and post-hoc** (chosen after reading exemplars), not pre-registered.
Consensus error set = wrong in ≥2 of 3 seeds. Counts below are the **final-epoch** protocol
(n=27); val-sel (n=26) is given in the last column and differs only in FP2 (5→4).

| # | cluster | rule | n (final) | % of errors | 3/3 seeds | median words | median dur s | median span frac | top-20 purity toward true | mean \|vote\| | n (val-sel) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FN1 | **speech-poor visual hate** | y=1, ≤25 words | **7** | 25.9 % | 7 | 6 | 24.4 | 0.632 | 0.112 | 0.757 | 7 |
| FN2 | **needle hate, diluted pool** | y=1, >25 words, span<25 % of runtime or ≤1 of 8 frames inside a span | **3** | 11.1 % | 3 | 131 | 101.2 | 0.085 | 0.083 | 0.867 | 3 |
| FN3 | **talky hate, text-stream miss** | y=1, >25 words, span-covered, frames land in span | **5** | 18.5 % | 4 | 145 | 98.7 | 0.763 | 0.197 | 0.611 | 5 |
| FP1 | **slur-bearing non-hate** | y=0, transcript matches the slur lexicon | **5** | 18.5 % | 5 | 158 | 86.5 | 0 | 0.100 | 0.780 | 5 |
| FP2 | **long-form talky non-hate** | y=0, no slur, ≥100 words | **5** | 18.5 % | 4 | 232 | 263.4 | 0 | 0.263 | **0.510** | 4 |
| FP3 | residual non-hate | y=0, no slur, <100 words | **2** | 7.4 % | 2 | 39 | 86.8 | 0 | 0.242 | 0.606 | 2 |

Slur lexicon used (documented for reproducibility, substring/word-boundary match on the lowercased
transcript): `nigger|nigga|nicker|niger|coon|jigaboo|spic|wetback|kike|jewburg|heeb|yid|raghead|
towelhead|paki|chink|gook|tranny|faggot|fag|dyke|retard|subhuman|gorilla|monkey|ape`.

### 5.1 Members and exemplars

**FN1 — speech-poor visual hate** (`hate_video_1, 10, 36, 71, 276, 279, 329`).
Image stream alone is correct on **5 of 7** in at least one seed; the text stream is correct on
**0 of 7**.
* `hate_video_329` — **0 transcript words**, 7.1 s clip, gold span covers 70 % of runtime. Nothing
  reaches the text stream; 3-seed mean vote −0.7491.
* `hate_video_10` — 4 words, transcript is `"🎼 In and and.🎼And."` (music, no speech), 38 s,
  span 98.8 %, 7 of 8 sampled frames inside the span, image stream correct, vote −0.6873.
* `hate_video_36` — 20 words, the "what is it / it's a gorilla" dehumanising taunt; vote pinned at
  **−1.0000** (all 20 retrieved neighbours non-hate) — both streams wrong.

**FN2 — needle hate, diluted pool** (`hate_video_304, 365, 427`).
* `hate_video_365` — 131 words, 94 s, gold span **8.5 %** of runtime, **1 of 8** sampled frames
  inside the span; vote −1.0000.
* `hate_video_304` — 119 words, 239 s, span 13.0 %, **0 of 8** frames inside the span; vote
  −0.9523.
* `hate_video_427` — explicit antisemitic content ("Jewburg…"), 214 words, but the **only** test
  hate video with no gold span recorded in `hate_spans.json`; image stream correct, text wrong.

**FN3 — talky hate, text-stream miss** (`hate_video_12, 143, 218, 295, 317`). Plenty of speech,
span covers most of the runtime, sampled frames land inside it — and both streams still fail.
* `hate_video_143` — 144 words, explicitly hateful speech ("I don't fucking support the Black Lives
  Matter movement… black people voted…"), span 82.6 %, 6 of 8 frames in span; vote −0.7229, top-20
  purity 0.20. This is a pure representation failure, not a coverage failure.
* `hate_video_218` — 85 words, span 88.9 %; vote −0.5486, the softest FN.

**FP1 — slur-bearing non-hate** (`non_hate_video_16, 32, 149, 314, 400`). All five carry explicit
slurs in the transcript and are gold-labelled **non-hate** — quotation, archival footage, music and
documentary framing.
* `non_hate_video_32` — "Man is a nigger… he's a nigger of the world" — a song lyric.
* `non_hate_video_314` — a reading of neo-Nazi text ("Siege by James Mason…").
* `non_hate_video_16` — archival segregationist committee footage.
A text-dominant model fires on the token; the label turns on speaker stance and framing. All five
wrong in 3/3 seeds, purity 0.100 — the memory bank has no non-hate-quoting-a-slur region to retrieve.

**FP2 — long-form talky non-hate** (`non_hate_video_121, 134, 140, 312, 642`). Median 232 words,
median 263 s. Lowest mean \|vote\| of any cluster (0.510) and the highest purity (0.263) — this is
the cluster sitting on the boundary, and it supplies most of the threshold-reachable items in §2.1.

**FP3 — residual** (`non_hate_video_167` 7 words, `non_hate_video_528` 72 words of
multilingual/garbled ASR).

---

## 6. TWO ADDITIONAL $0 CEILINGS MEASURED HERE

Both are legal (no test fitting) and both are **new measurements**, not literature claims.

### 6.1 Memory-bank curation — measured null, fails its own control (bears on F78)

F78 parked memory-bank curation partly because a faithful pregate needed "~0.3 GPU-h head
re-mint". The CPU proxy supplies the trained head for free, so the pregate is now runnable at
literally zero GPU. Curation rule (train-only, no dev/test involvement): drop every train row whose
own leave-one-out top-20 kNN vote in the trained head space disagrees with its label; re-run the
test vote over the pruned bank.

| cell | n_train | n LOO-disagree | rate | Δacc (curated drop) | Δacc (random drop, same n) |
|---|---|---|---|---|---|
| s0 val-sel | 744 | 45 | 0.0605 | +0.0000 | +0.0000 |
| s1 val-sel | 744 | 63 | 0.0847 | +0.0047 | **+0.0093** |
| s2 val-sel | 744 | 39 | 0.0524 | +0.0000 | +0.0000 |
| s0 final | 744 | 39 | 0.0524 | +0.0047 | +0.0000 |
| s1 final | 744 | 40 | 0.0538 | +0.0000 | +0.0000 |
| s2 final | 744 | 39 | 0.0524 | +0.0000 | +0.0000 |

3-seed means: **curated +0.0016 (both protocols)** vs **random-deletion control +0.0031 (val-sel)
/ +0.0000 (final)**. The curated rule does **not** beat deleting the same number of rows at random.
Scope: one curation rule, one proxy head per cell, single draw per seed — this is a pregate-grade
null, not a formal verdict. It does convert F78's "unpriced because it needs GPU" into
"measured ≈0 at $0", and it prices the memory-bank label/geometry noise at **39-63 of 744 rows
(5.2-8.5 %)**.

### 6.2 Length de-bias — measured null

See §4.3. Δ = −0.0016 acc / −0.0017 mF1 (train-LOO fit), +0.0000 / −0.0002 (dev fit).

---

## 7. SOLUTION MAPPING PER CLUSTER

Ceiling = flip *every* member of the cluster and break nothing else. 1 test item = 0.00465 acc.

### FN1 — speech-poor visual hate (n=7, ceiling **+0.0326** acc)

* **LOCKED**: per-item image/fusion routing — F47 (routing dead at read *and* realisable ceiling)
  + F66 (selection-locked arithmetic) + F49 (alignment ceiling q>0.663); vision-side adaptation —
  F65 (image stream MOVED, K-V2 TIE everywhere, 8th law-I); fusion operator — F83/F85 (concat
  measured null, axis closed); frame budget — F67 (8f saturates, 16f dead); resolution — F76
  (headroom anti-correlated, parked at recon); segment pooling — P3; MLLM-scores-as-signal — P11.
* **VETOED**: OCR channel (standing user veto). This is the single most on-mechanism carrier for
  speech-poor hate videos, where the hateful content is frequently on-screen text/meme overlay —
  and it is the ablation-load-bearing channel in MM-HSD's 0.878 (F81 S2). It stays vetoed.
* **OPEN behind user gate**: **CLAP general-audio** (F81, download-gated, HateMM-only). This is the
  only channel that could carry non-speech audio (music, chanting, screams, tone) for videos where
  Whisper produced nothing. Ceiling **+0.0326**, and it is the tightest-targeted gate in the queue —
  but the prior is low: F41 killed whole-video eGeMAPS prosody at a $0 conditional-info gate and
  F64 killed the learned Whisper-encoder audio axis on all 3 datasets, so an audio channel has
  already come back with zero conditional label information over `Z_best` twice. Realistic
  expectation ≪ ceiling.
* **OPEN behind user gate (secondary)**: Molmo2-8B (SigLIP2 tower) — F65-weakened, since the
  vision-adaptation axis is closed; would target the same 5/7 items the current image stream
  already gets right in at least one seed, which is precisely the selection-locked slice.
* **GENUINELY OPEN in-box at $0**: none.

### FN2 — needle hate, diluted pool (n=3, ceiling **+0.0140**)

* **LOCKED — completely.** F66 (ISR NO-GO; the only ban-surviving per-segment operator is flat
  +0.0012 on HateMM, and burstiness is selection-locked), P3 (evidence-density pooling dead on all
  3 datasets, including the clean-probe HateMM cell), P11 (weak-sup localisation dead at zero
  training cost), F67 (denser sampling adds nothing), F81 S1 (temporal axis closed at 4 levels;
  gold-timestamp trimming is the only demonstrated temporal lever and it is banned as gold-label
  use).
* Nothing open, at any price, inside the constraint set. `hate_video_365` and `hate_video_304` have
  **0-1 of 8** sampled frames inside the gold hate span; only a legal localiser could reach them
  and every localiser route is closed.

### FN3 — talky hate, text-stream miss (n=5, ceiling **+0.0233**)

* **LOCKED**: head-side reshaping — F75 (NCA/soft-kNN/SupCon/mixup, 7/8 cells KS-arm-dead), F73
  (SAM + mod-dropout), F70 (readout layer/token/prompt), F69 (grad-norm selection), F62b (SWA),
  F63 (label propagation); adapter structure — F87 (MokA routed-LoRA measured null on ZH, HateMM
  stage-2 auto-defunded, and the undiluted-text-target bet **refuted**); encoder identity/scale —
  D7 + F44/B2 (32B *regresses* on the HateMM anchor); fusion — F85; direction-flip — F72 (bidir
  crater −10 to −14 pt).
* **OPEN behind user gate**: **MNTP stage-2** (F81 EV ~8-12 %). It is the only remaining lever that
  changes the *text representation itself* rather than the head or the routing, and FN3 is by
  construction the cluster where the text stream fails on content it fully observes. Ceiling
  **+0.0233** for FN3 alone; if it also converted the 5 text-stream failures inside FN1/FN2 the
  arithmetic ceiling rises to ~+0.0465, but F72's Llama-pattern crater is a direct warning on this
  family and the user gate stands.
* **GENUINELY OPEN in-box at $0**: none.

### FP1 — slur-bearing non-hate (n=5, ceiling **+0.0233**)

* This is an **annotation-scheme boundary**, not a representation gap: the transcript literally
  contains the slur and the gold label is non-hate because of quotation/archival/musical framing.
  Separating them needs speaker-stance and pragmatics, i.e. exactly the MLLM-reasoning role that
  the campaign closed 13 ways (P1-P5 negative, P2b 32B, P10-b 72B A-fuse promoted only for
  localisation, P10-c generation-jump under gate).
* **LOCKED**: consensus/auto memory repair NEGATIVE (2026-07-06, AND-rule C−A=0); curation
  measured null here (§6.1); F63; F82 (graded 3-class soft-label parked at the $0 pregate — its
  oracle ceiling did not clear); relabelling test items is not available at any price.
* **VETOED**: closed-model API stance judging (needs data-export ruling; also the "new signal, no
  conversion" law has fired 8-9 times).
* **GENUINELY OPEN in-box at $0**: none. **Report value: high** — this cluster is the concrete,
  citable statement of where the ceiling on HateMM actually sits. Five of 27 errors (18.5 %) are
  cases where a text-dominant detector is behaving *reasonably* and the label encodes pragmatics the
  representation was never given. It belongs in the paper's limitations/analysis chapter.

### FP2 — long-form talky non-hate (n=5, ceiling **+0.0233**)

* Softest cluster (mean \|vote\| 0.510, purity 0.263) and the source of most threshold-reachable
  items — but every legal calibration lever is measured null: dev-fitted threshold by accuracy
  +0.0000/+0.0016, by macro-F1 +0.0016 acc/+0.0031 mF1, train-LOO logistic −0.0016, length de-bias
  −0.0016/+0.0000, test-fitted oracle threshold only +0.0078/+0.0124.
* **LOCKED**: F66; §2.1 and §6.2 of this document.
* **GENUINELY OPEN in-box at $0**: none that survives its own control.

### FP3 — residual (n=2, ceiling **+0.0093**) — no identified mechanism.

---

## 8. TOP QUANTIFIED FIX OPPORTUNITIES

Honest summary first: **the genuinely-open-in-box-at-$0 set is EMPTY.** Three candidates that the
error structure suggested were measured *in this analysis* and all three came back null — global
threshold recalibration (§2.1), transcript-length de-bias (§4.3), and memory-bank curation with a
random-deletion control (§6.1). What follows is ranked by ceiling, with the gate that owns it.

**1. CLAP general-audio channel — USER-GATED (download). Ceiling +0.0326 acc (7 items).**
The only fix whose target cluster (FN1) is defined by the *absence* of the signal every existing
channel carries: 7 of 27 errors have ≤25 transcript words, 2 have zero, and the model predicts
non-hate on 30/30 empty-transcript test items in 6/6 cells. The image stream already recovers 5 of
the 7 in at least one seed but only via the banned per-item selection. Downweight heavily: F41
(eGeMAPS prosody) and F64 (Whisper-encoder audio) both returned zero conditional information over
`Z_best`, so this is the third bite at the audio axis with the first two dead. Recommend the same
$0 conditional-info pregate on CLAP embeddings *before* any GPU, if the download gate opens.

**2. MNTP stage-2 — USER-GATED. Ceiling +0.0233 (FN3 alone) to +0.0465 (FN3 + the text-stream
failures inside FN1/FN2).**
FN3 is 5 fully-observed, span-covered, speech-rich hate videos where the text stream — the stream
that carries *both* of the project's measured passes (F45 ZH, F58 HateMM) — is simply wrong, at
top-20 purity 0.197. MNTP is the last unmeasured lever that changes the text representation itself.
F81 priced it at ~8-12 %; F72's bidir crater is the standing warning; F87 just refuted the
adjacent text-side bet at the *routing* level (which does not touch pretraining).

**3. Text-only arm — PAPER VALUE, not a performance bet. Measured here: +0.0047 acc (val-sel,
sign 2/3) / +0.0093 acc (final, sign 3/3), 3-seed.**
In the trained head space, kNN over the text projection alone scores 0.8822/0.8853 vs the deployed
0.8775/0.8760, while the image projection alone scores 0.7411-0.7426. Both deltas are inside the
±0.014 seed band and far under the +0.030 bar, so this is **measured-not-promoted** — and it is
exactly what F86 predicts (U1 image uniqueness 0.0000, no synergy term) and what F85's concat null
is consistent with. Its value is as the honest **ablation** for the paper: on HateMM the image
stream contributes no net accuracy and costs ~0.5-0.9 pt in the deployed Hadamard fusion, while
supplying 11-14 per-item fixes that are only realisable by F66-banned selection. Note the read here
is post-hoc (the `text_proj` sub-space of a head trained under Hadamard fusion), so a paper-grade
text-only ablation needs a properly trained text-only arm.

**Infra finding worth carrying forward.** The align head trains and evaluates end-to-end in
**52 s of wall time on 8 CPUs** (30 epochs, 60 retrieval evals, on banked features) — the 3-seed
proxy family cost ~2.6 CPU-minutes total and zero GPU. F78 priced a faithful curation pregate at
"~0.3 GPU-h head re-mint"; that estimate can be replaced by ~1 CPU-minute per seed. The head-side
base rate is ~0-for-21, but the marginal cost of the 22nd head-side bite — and of every
head-space diagnostic, ablation and control the paper might want — is now a CPU minute, not a
queue slot. Any head-only cell (text-only arm, curation variants, calibration variants, purity
diagnostics) can be run under full 3-seed discipline without touching the GPU allocation. Caveat:
CPU-trained heads are not bit-exact to the CUDA floor (−0.0031 final-epoch acc here), so a
CPU-trained arm must be paired against a **CPU-trained floor**, never against the banked GPU floor
— the same same-path-floors discipline F87 established after the merge-drift incident.

---

## 9. LIMITATIONS

1. **Proxy, not the floor.** Per-item predictions are from CPU-retrained heads (§0.1-0.2). Val-sel
   3-seed means match the floor exactly at 4 dp; final-epoch is 0.67 items/seed off. Cluster
   membership at the level of individual borderline items could differ by a few items on the true
   floor heads; the cluster *structure*, the covariate contrasts, the bank-prior arithmetic and the
   ceilings are all robust to that (24-25/27 errors are wrong in 3/3 seeds and every stratification
   bin reproduces across seeds).
2. **Cluster definitions are post-hoc**, chosen after reading exemplars. They are descriptive
   labels for reporting, not a pre-registered taxonomy, and the boundaries (25 words, 100 words,
   25 % span) are round numbers, not fitted.
3. **The slur lexicon is a hand-written forensic probe** (§5), not a validated resource. It is used
   only to separate FP1 from FP2; the FP1 exemplars were read individually and all five confirm the
   quotation/archival/music framing.
4. **Head-space single-stream reads are post-hoc sub-spaces** of a head trained under Hadamard
   fusion; they answer "would this stream's neighbourhood have voted correctly" and are not
   deployable configurations.
5. **All test-set quantities are single-draw reads** used descriptively. The two oracle numbers
   (best global threshold; oracle channel pick) are explicitly test-fitted and are labelled
   FORENSIC/BANNED wherever they appear. No threshold, curation rule, bin boundary or cluster
   definition was selected by test accuracy.
6. **No HateMM noisy-label id list exists** in the repo. The consensus-denoising work (including
   the human-in-the-loop 2-entry deletion that helped on EN) was run on MHC EN/ZH archive space,
   and the AUTO two-vote repair was negative. §6.1's LOO-disagreement set (39-63 train rows) is the
   closest available label/geometry-noise proxy for HateMM and is a *derived* proxy, not an
   annotation.
7. `data/audio/HateMM` holds only eGeMAPS-v02 and Whisper-large-v3 derived caches (89 MB) — the two
   axes already killed by F41 and F64. No general-audio (CLAP-class) representation exists locally,
   which is why opportunity #1 remains download-gated.
