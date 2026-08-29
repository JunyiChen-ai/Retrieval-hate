# ERROR-PATTERN FORENSICS — MultiHateClip-ZH (Bilibili), deployed LoRA-Qwen floor

**Agent:** errpat-zh · **Date:** 2026-07-26 NZST · **GPU spent: 0** (CPU only, ≤6 threads, ~2m10s total compute)
**Test-touch:** READ-ONLY diagnostics. No tuning, no selection, no training decision derived from test.
**Scope:** binding ZH floor = **job 13150**, 3 head-seeds, `Qwen2.5-VL-7B-Instruct-LoRA_HF`, group `RAC_video_b3_lora`.

---

## 0. PROVENANCE HEADER

### 0.1 Primary (bit-exact) sources

| object | path | sha256 |
|---|---|---|
| 13150 seed0 trainlog | `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed0_13150.trainlog` | `62f10a1e7c6186b78e9266736de434fdd291f59ec27cdcdf17ada49bf569785e` |
| 13150 seed1 trainlog | `…_seed1_13150.trainlog` | `72fa1f138348dba4feb3f2c9baecadb3291e6d1f246b948157094745c99829e6` |
| 13150 seed2 trainlog | `…_seed2_13150.trainlog` | `516cb495084b42884f06bd0843634920e1ba05cf59f70f8ed71b7009069e1a30` |
| pre-head feature caches | `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | inventoried in `CURATION_FORENSIC_RECON.md` §2.1 (`7025391`) |
| ZH labels + deployed text | `data/gt/MHC_zh/{train,val,test}.jsonl` | 579 / 78 / 149 rows |
| MHC 3-class labels | `data/_src_Multihateclip/Chinese/annotation(new).json` | 897 rows, `Label` ∈ {Normal, Offensive, Hateful} |
| ZH Whisper ASR | `data/ASR/MHC_zh/{train,dev_seen,test_seen}_asrK4_whisper-large-v3.jsonl` | 149 test rows, carries `duration` |

Every number below was re-read from these files at report time. Load-bearing readouts were additionally
spot-checked by grepping the raw trainlogs directly (§1.1).

### 0.2 Artifact status — per-item floor predictions are NOT recoverable bit-exactly

Confirmed against F78 / `CURATION_FORENSIC_RECON.md` §2.2: **all 6 deployed floor head ckpts are deleted**
(`logging/Retrieval/MHC_zh/RAC_video_b3_lora/**/ckpt/` empty), no `*_retrieval_logging_dict.pkl` survives,
and head embeddings were never persisted. The pre-head feature caches survive (seed-independent).

Therefore this report uses **two clearly-separated evidence tiers**:

- **TIER 1 — BIT-EXACT.** Per-epoch dev/test accuracy, macro-F1, ROC, precision, recall for all 30 epochs
  × 3 seeds, parsed from the 13150 trainlogs. **The entire val-selected↔final-epoch protocol analysis
  (§1) rests on Tier 1 alone.** Script: `scripts/analysis/errpat_zh_trainlog_curves.py`
  → `errpat_zh_curves_OUT.json`. Reproduces the recorded floor exactly:
  val-sel **0.8322 / 0.8015**, final-epoch **0.8456 / 0.8173** acc/mF1.
- **TIER 2 — PROXY (labelled, priced).** A **same-recipe CPU re-mint** of the head, used only where
  per-item data is needed (§2–§5). Byte-identical CLI to `scripts/slurm/enc3seed_zh_b3.sbatch` except
  `--device cpu`, `--num_workers 0`, and a fresh `--group_name errpat_zh_remint_v2` (no banked dir touched).
  Script: `scripts/analysis/errpat_zh_remint.py` → `errpat_remint_dumps/errpat_zh_remint_seed{0,1,2}.pkl`.
  **NOT bit-exact to 13150** — model init draws from the CPU RNG in both runs so initialisation is shared,
  but dropout masks come from the CUDA RNG on GPU and matmul reduction order differs.

#### Proxy fidelity — measured, not assumed (`errpat_zh_remint_fidelity.py`)

| seed | ep29 test acc re-mint / banked | ep29 mF1 re-mint / banked | ep29 confusion re-mint / banked-implied |
|---|---|---|---|
| 0 | **0.8456 / 0.8456** | 0.8158 / 0.8181 | TP33 FP11 FN12 TN93 / TP34 FP12 FN11 TN92 |
| 1 | **0.8389 / 0.8389** | 0.8090 / 0.8113 | TP33 FP12 FN12 TN92 / TP34 FP13 FN11 TN91 |
| 2 | **0.8523 / 0.8523** | 0.8226 / 0.8226 | TP33 FP10 FN12 TN94 / TP33 FP10 FN12 TN94 (**identical**) |

At the final epoch the proxy reproduces banked test accuracy **to 4 dp on all three seeds**; error *count*
matches exactly (23/24/22); the error *set* differs by at most **one item** on seeds 0–1 and is identical on
seed 2. Mean |difference| across the whole curve, by window (`errpat_zh_clusters.py` §1):

| window | mean abs diff (test items) | max |
|---|---|---|
| ep5–19 | 2.47 | 9.00 |
| ep20–24 | 2.60 | 5.99 |
| ep25–29 | 1.74 | 4.00 |
| **ep29** | **0.01** | **0.01** |

**The final epoch is the only device-reproducible readout in the run.** This is itself a §1 finding, not
merely a fidelity note. Consequence for this report: the §2–§5 per-item taxonomy is trustworthy at the
final-epoch protocol (≤1 item of slack) and is **not** trustworthy for reconstructing which specific items
the *val-selected* epoch got wrong in the banked run. §1 therefore avoids per-item claims entirely.

---

## 1. THE VAL-SELECTED ↔ FINAL-EPOCH QUESTION (Tier 1, bit-exact)

### 1.1 The gap is exactly two test items, in every seed

Deployed selection rule (from the sbatch, `max(warm, key=(val_acc, val_roc))` → earliest on tie, warmup ≥ 5):

| seed | val-sel epoch | dev acc there | test acc / mF1 | ep29 dev acc | ep29 test acc / mF1 | Δ acc (items) |
|---|---|---|---|---|---|---|
| 0 | 20 | 0.8718 (68/78) | 0.8322 / 0.8023 | 0.8462 (66/78) | 0.8456 / 0.8181 | **+0.0134 (+2)** |
| 1 | 26 | 0.8718 (68/78) | 0.8255 / 0.7956 | 0.8590 (67/78) | 0.8389 / 0.8113 | **+0.0134 (+2)** |
| 2 | 19 | 0.8718 (68/78) | 0.8389 / 0.8065 | 0.8462 (66/78) | 0.8523 / 0.8226 | **+0.0134 (+2)** |
| **mean** | | | **0.8322 / 0.8015** | | **0.8456 / 0.8173** | **+0.0134 / +0.0159** |

All values verified by direct grep of the raw trainlogs. The dev argmax is **0.8718 = 68/78 in all three
seeds**, and epoch 29 sits **1–2 dev items** below it. **The protocol discards the final epoch on a deficit
of one or two items out of 78, and pays exactly two test items out of 149 for doing so, three times over.**

### 1.2 The dev signal is not weak — it is anti-correlated with the objective

Spearman correlation between dev accuracy and test accuracy across the 25 legal epochs (≥ warmup):

| seed | Spearman | p | Pearson | p |
|---|---|---|---|---|
| 0 | **−0.3457** | 0.0905 | −0.1623 | 0.4383 |
| 1 | +0.0419 | 0.8425 | +0.0614 | 0.7706 |
| 2 | −0.1531 | 0.4649 | −0.1848 | 0.3765 |
| **pooled (n=75)** | **−0.2402** | **0.0380** | −0.1520 | 0.1929 |

The pooled rank correlation is **significantly negative**. This is the quantified form of F45's observation
(dev saturates ~ep19 while test climbs to ep29): in the late-training regime the 78-item dev set moves
*against* test accuracy. A selector reading it is not noisy — it is pointed the wrong way.

### 1.3 Val-selection buys almost nothing over a coin flip; the last epoch buys 2.7 items

3-seed mean test accuracy under each rule (Tier 1):

| rule | test acc | vs random-legal-epoch |
|---|---|---|
| worst legal epoch per seed | 0.7852 | −6.3 items |
| **uniformly random legal epoch** (mean over ep5–29) | **0.8278** | — |
| **val-selected (deployed)** | **0.8322** | **+0.65 items** |
| pooled-dev argmax (shared epoch 19, see §1.5) | 0.8210 | −1.0 items |
| **final epoch 29** | **0.8456** | **+2.7 items** |
| best legal epoch per seed (test-oracle) | 0.8479 | +3.0 items |

Two readings matter. First, **val-selection recovers 0.65 test items over choosing an epoch at random** —
its information content is essentially nil. Second, **the final epoch lands within 0.0023 acc (0.34 items)
of the per-seed test-oracle epoch**: there is almost nothing left for *any* selection rule to win.

### 1.4 The selection is a lottery over a wide test range

Number of epochs whose dev accuracy sits within k dev items (k/78) of the argmax, and the span of their
test accuracies:

| seed | k=0 | k=1 | k=2 |
|---|---|---|---|
| 0 | 2 epochs, test 0.8255–0.8322 (1 item) | **7 epochs, 0.7919–0.8322 (6 items)** | 20 epochs, 0.7584–0.8456 (13 items) |
| 1 | 4 epochs, 0.8121–0.8322 (3 items) | **11 epochs, 0.7785–0.8389 (9 items)** | 19 epochs, 0.7785–0.8389 (9 items) |
| 2 | 1 epoch, 0.8389 | **9 epochs, 0.8188–0.8523 (5 items)** | 14 epochs, 0.8188–0.8523 (5 items)|

**Flipping a single dev item — one video out of 78 — moves the protocol's output by up to 6–9 test items.**

### 1.5 Enlarging the dev signal 3× does not rescue the protocol — it makes it worse

A natural repair keeps a selection rule but pools the dev evidence: choose the single epoch maximising
**mean** dev accuracy across the three seeds (234 dev decisions instead of 78), then read test per seed.
This is not a cross-seed ensemble (no predictions are averaged; three independent heads remain).

Result: pooled-dev argmax = **epoch 19** (pooled dev acc 0.8590) → 3-seed test **0.8210 / mF1 0.7823**,
which ranks **19th of the 25 legal shared epochs**. Meanwhile:

- ep29 ranks **1st of 25** legal fixed shared epochs (0.8456).
- The top five are ep29, 27, 28, 26, 25 (0.8456 / 0.8434 / 0.8434 / 0.8411 / 0.8367) — the entire late-training
  region dominates, so "final epoch" is not a lucky point but the top of a broad, monotone plateau.

**The problem is not dev-set size.** Tripling the selection signal moved the pick from rank ~10–19 to rank 19.
The dev criterion is mis-aligned in this regime, not merely under-sampled.

### 1.6 Protocol-labile item pool (Tier 2, for characterisation only)

Using the banked val-sel epoch indices (20/26/19) against ep29 in the re-minted per-item predictions:
**14 distinct test items (9.4%)** ever change prediction between the two protocols across the three seeds;
per seed 10 / 1 / 12 items change. Only **5 of those 14** are in the stable error core (§2) — the other 9 are
items that the final-epoch readout gets right in all seeds. Over the whole legal epoch window
(ep5–29) **35 / 38 / 42 items (23.5% / 25.5% / 28.2%)** are not constant, against 98–104 always-correct and
9–10 always-wrong. So roughly a quarter of the ZH test set is epoch-labile, and both protocols are drawing
from that pool; the final-epoch rule simply draws from the top of it.

### 1.7 VERDICT ON PROTOCOL RETIREMENT

**The evidence supports retiring the 78-item val-selected protocol on methodological grounds, and it is
strong.** Five independent Tier-1 facts point the same way: (i) the dev↔test rank correlation over legal
epochs is *negative*, significantly so when pooled (−0.2402, p = 0.0380); (ii) the rule recovers only 0.65
test items over a uniformly random legal epoch, i.e. it carries essentially no information; (iii) a
one-dev-item perturbation relocates the pick across a 6–9-test-item range; (iv) tripling the dev signal by
pooling across seeds makes the pick *worse* (rank 19/25), which rules out "the dev set is just small" as the
explanation; and (v) decisively, the val-selected readout is **not reproducible across a device change** —
a same-recipe CPU re-mint moved it by up to −0.0335 acc and relocated the argmax from epoch 20 to epoch 5 —
whereas the final-epoch readout reproduced banked test accuracy **to four decimal places on all three
seeds**. A protocol whose output is not stable under floating-point reduction order is not measuring the
model. **The honest caveat the user must weigh:** every one of these facts is a property of the 78-item dev
set and holds without reference to any ZH outcome, but retirement nonetheless converts exactly one verdict
in our own favour (ZH val-sel FAIL → not applicable), and that is textbook rule-shopping exposure. It is
mitigated but not eliminated by the fact that retirement costs nothing elsewhere: HateMM already passes
**both** protocols (F53) and EN fails both (F55), so no other verdict moves. The defensible framing is not
"ZH passes if we drop val-sel" but "the ZH dev split is too small and too anti-aligned to function as a
model-selection instrument, as shown by (i)–(v), and single-protocol final-epoch reporting is the
methodologically sound reading" — with the protocol decision itself remaining the user's ruling.

---

## 2. ERROR INVENTORY (Tier 2, final-epoch protocol)

n_test = 149 (104 Normal / 45 positive: 28 Offensive + 17 Hateful).

| seed | acc | errors | FP | FN |
|---|---|---|---|---|
| 0 | 0.8456 | 23 | 11 | 12 |
| 1 | 0.8389 | 24 | 12 | 12 |
| 2 | 0.8523 | 22 | 10 | 12 |

Seed consensus:

| | count |
|---|---|
| wrong in **3/3** seeds ("**core**") | **22** |
| wrong in exactly 2/3 | **0** |
| wrong in exactly 1/3 | 3 |
| never wrong | 124 |
| union (any seed) | 25 |

**The error set is almost entirely seed-invariant: 22 of the 25-item union (88%) fail in all three seeds, and
nothing at all fails in exactly two.** Broken down by direction, **all 12 false negatives are 3/3 stable**;
the entire seed-to-seed variance of the floor lives on the false-positive side (10 stable + 3 flippy).
Head-seed variation therefore is not exploring the FN problem at all.

---

## 3. VOTE STRUCTURE — the errors are confident, not marginal (Tier 2)

Deployed decision: `vote = Σ (2·lab−1)·cos·w / Σw` over top-20, `w = [20…1]`, predict 1 iff vote ≥ 0
(`src/utils/metrics.py:262-301`; every dumped vote was re-derived from the dumped neighbour lists and
matched to < 1e-9, and re-derived accuracy matched `metrics.py` exactly).

| quantity | errors | correct |
|---|---|---|
| median \|vote\| (margin) | 0.7137 | 0.9999 |
| mean \|vote\| | 0.6367 | 0.8867 |
| **median top-20 purity** (share of neighbours whose label = query's gold) | **0.15** | **1.00** |

Core-error purity distribution (22 items, seed-averaged): **8 at ≤ 0.10, 7 in (0.10, 0.25], 7 in (0.25, 0.45],
and zero above 0.45** — median 0.1167. **Not one of the 22 stable errors has a majority-correct
neighbourhood.** 14 of 22 are simultaneously high-margin (≥ 0.5) and inverted (purity ≤ 0.25); only 2 of 22
are genuinely low-margin (≤ 0.3).

**These are not boundary cases.** The vote is confidently wrong because the retrieved memory neighbourhood
is confidently the wrong class. Two consequences follow directly:

- **Calibration / threshold work cannot reach them.** The test-fitted **global-threshold oracle** buys
  +0.0134 / +0.0336 / +0.0134 acc (**mean +0.0201**), below the +0.030 bar even as a gold-cheat upper bound,
  and only 4 / 8 / 3 errors per seed can be flipped with collateral damage ≤ 1 item.
- **It is not a memory-coverage problem.** In the pre-head raw fused space over the full 579-row bank, the
  **first same-gold-class train neighbour sits at median rank 1.5** for the core errors (11 of 22 at rank 1;
  all 22 within rank 14). The right analogues are present and top-ranked; they are simply out-voted.

Pre-head vs deployed head-space purity on the core errors: raw image-only 0.500, raw text-only 0.375, raw
fused 0.400 (5 of 22 still majority-correct) → **deployed head space 0.1167 (0 of 22 majority-correct)**,
while correct items sharpen 0.85 → 0.9833. The trained head **sharpens an inversion that mostly already
exists in the encoder features**, converting 5 marginally-recoverable neighbourhoods into inverted ones.

---

## 4. STREAM FORENSICS (pre-head raw banked features; NOT the deployed head space)

The deployed fusion is a trained Hadamard `align`, so no head-space single-stream vote exists. Same vote
operator, three raw key spaces, single draw (features are seed-independent):

| key space | acc | AUC | FP | FN |
|---|---|---|---|---|
| image only | 0.7047 | 0.6981 | 17 | 27 |
| **text only** | **0.8523** | 0.9171 | 16 | 6 |
| fused (L2-norm concat) | 0.8456 | 0.9194 | 15 | 8 |

Consistent with F86 (image uniqueness U1 = 0.0000 on ZH) and F45 (the LoRA gain lives entirely in the text
stream): **the image stream is near-useless on ZH and fusion adds no accuracy over text alone** (text-only
0.8523 vs fused 0.8456 = 1 item, and the deployed 3-seed mean is also 0.8456).

Cross-tab of the raw streams on the 22 core errors:

| | count | acc if all flipped | status |
|---|---|---|---|
| **neither stream right** | **8** | +0.0537 | irreducible under any reweighting of the two banked streams |
| exactly one stream right | 12 | +0.0805 | per-item channel selection — **F47-closed** at all three supervision sources; **F66** arithmetic applies |
| both streams right, deployed wrong | 2 | +0.0134 | head/fusion discards information both raw streams have; fusion-operator axis **F85-closed**, fixed composition **F50-closed** |

---

## 5. CONTENT COVARIATES (Tier 2)

Deployed ZH text = `Title + " . " + Transcript` from `annotation(new).json` (`scripts/prep_mhc.py:73-78`).
Medians on test: title **15** chars, transcript **76** chars, composed text **96** chars (title = 15.6% of it).

**Correction to a load-bearing ledger item.** F77 L2 records the deployed ZH text as "the Bilibili
search-result description metadata, NOT the Whisper ASR". More precisely: it is the harvested **title**
(which carries the `<em class="keyword">` search-term markup) concatenated with MultiHateClip's own
**speech transcript**, which is itself ASR-derived and visibly noisy — several core errors carry
wrong-language output or repetition loops (§5.2). So the ZH text channel *does* contain a speech transcript;
F77's "not Whisper ASR" is right only in the narrow sense that it is not *our* `data/ASR/` run.

### 5.1 What does and does not predict error

Error rate per seed (= Σ seeds-wrong / 3n), with permutation tests where a cluster was claimed:

| covariate | groups | err rate/seed | 3/3 errors | verdict |
|---|---|---|---|---|
| **transcript length** | Q2 band 31–76 chars (n=37) vs rest | **0.2973** vs 0.0631 / 0.1532 / 0.1053 (Q1/Q3/Q4) | **11 of 22** | **REAL**, perm p = **0.0048** |
| deployed-text length | Q2 49–96 chars (n=37) vs rest | 0.3243 vs ~0.10 | 12 | same effect, p = 0.001 |
| 3-class label | Normal / Offensive / Hateful | 0.1058 / 0.2500 / **0.2941** | 10 / 7 / 5 | see §5.3 |
| `<em>` keyword markup | present (63) vs absent (86) | 0.1799 vs 0.1357 | 11 / 11 | not significant (§5.4) |
| ASR-corrupted transcript | garbage ≥ 0.15 (12) vs rest | 0.2500 vs 0.1460 | 3 | **not significant**, p = 0.2552 |
| duration | quartiles (16 / 31 / 41 s) | 0.135 / 0.189 / 0.153 / 0.140 | 5/7/5/5 | flat, no effect |
| Whisper ASR empty | — | — | — | **0 of 149 empty**; ZH audio always has speech |

**The "empty or off-topic description" hypothesis is refuted.** No ZH test item has an empty text channel
(composed text ranges 8–336 chars, zero empties), and error rate is *non-monotone* in text length: the two
extreme quartiles are the two best groups (longest ≥ 183 chars: 0.0789; shortest < 49 chars: 0.0901) while the
mass sits in the middle. Only 7 of 149 items have an empty transcript (text = title only) and they error at
0.1429, *below* the transcript-present rate of 0.1549.

### 5.2 The one real covariate cluster: thin transcript

Errors concentrate where the transcript is present but **thin — roughly one short utterance**. The empirical
transcript-length quartiles are 31 / 76 / 162 chars, and the Q2 band [31, 76) holds **11 of the 22 core errors
in 37 items** (err rate/seed 0.2973 vs 0.0631 / 0.1532 / 0.1053 in Q1 / Q3 / Q4) — 2.0× enrichment,
permutation p = **0.0048** (50 000 perms; robust to integer vs exact-quantile cut, p = 0.0051). Class-stratified
the effect is consistent but each half is underpowered (negatives p = 0.0506, positives p = 0.0668), so the
honest claim is a pooled effect present in both directions rather than a class-specific one. Mechanism:
enough text to dominate a text-dominant fused key (§4), too little to individuate — the retrieval lands on
generic same-topic neighbours. Exemplars (`errpat_zh_clusters.py` §6):

- `BV1ia411m7Yy` gold=1 Hateful, transcript 35 chars, purity 0.150, margin 0.742 —
  title "珍爱生命，远离公主病", transcript is a repetition loop ("我心到…我心到…").
- `BV1Km4y1u7ri` gold=0 Normal, transcript 49 chars, purity 0.117 — title "妈宝男~", i.e. a clip
  *about* the slur it was harvested by.
- `BV1qZ4y1T71a` gold=1 Offensive, transcript 33 chars, purity 0.067, margin 0.927.

### 5.3 Within-positive wall: ZH inverts the EN pattern

F44/F82 established the within-positive wall via Offensive being the majority of positives. On ZH test the
errors are **not** concentrated in Offensive: Offensive n=28 err 0.2500 (7 core), **Hateful n=17 err 0.2941
(5 core)** — the graded class the deployed binary merge treats as *most* positive is the one it misses most
often. This is consistent with F82's conclusion (ZH graded oracle only +0.0256, and down-weighting Offensive
is monotonically harmful) and adds a reason: there is no Offensive-specific error mass to reallocate.

### 5.4 Topic-vs-stance, and keyword-absent positives — suggestive, both underpowered

The `<em class="keyword">` markup is the Bilibili search term the clip was harvested by, usually the slur
itself, and it appears verbatim in the deployed text. Two readings were tested and **neither reaches
significance**:

- **FP side (topic-vs-stance).** Negatives carrying the markup false-positive at 0.172 vs 0.0776 without —
  a 2.2× raw ratio — but at the core-error level 5 observed vs 4.57 expected, **p = 0.5022**. The five stable
  FPs that do carry it are all clips *about* the slur (`公主病`, `妈宝男`, `花痴`, `流氓`, `绿帽`), which is
  the expected counter-speech confusion, but the count is exactly chance.
- **FN side (lexical dependence).** Positives *lacking* the harvested keyword are missed far more often —
  0.4615 vs 0.1875 err rate/seed — but n = 13 gives 6 observed vs 3.47 expected, **p = 0.0681**. Suggestive
  of a text channel that works largely by lexical slur matching (consistent with F86's text-uniqueness
  dominance), not established.

Both are recorded as hypotheses the ZH test split is too small to settle, not as clusters.

---

## 6. NAMED CLUSTERS

Counts are over the 22-item stable core (3/3 at final epoch); clusters overlap by construction.

| # | cluster | n | % of core | fix status |
|---|---|---|---|---|
| C1 | **Inverted-neighbourhood confident error** — top-20 purity ≤ 0.45 with the vote far from the boundary | **22** | 100% | structural; see C1a–C1c |
| C1a | ↳ **no channel knows** — neither raw stream gets it right | 8 | 36% | **LOCKED** (irreducible in the banked representation) |
| C1b | ↳ **one channel knows** — exactly one raw stream right | 12 | 55% | **LOCKED** — F47 (selection dead at all 3 supervision sources) + F66 (arithmetic lock) |
| C1c | ↳ **fusion loses it** — both raw streams right, deployed wrong | 2 | 9% | **LOCKED** — F85 (concat null), F50 (fixed composition) |
| C2 | **Thin transcript** — transcript in the Q2 band [31, 76) chars; 2.0× enrichment, p = 0.0048 | 11 | 50% | see §7.1/§7.2 — priced dead: the deficit is speech absence, not transcription quality |
| C3 | **FN-locked** — false negatives, 12/12 wrong in all 3 seeds, none seed-flippy | 12 | 55% | **LOCKED** at seed level; head-seed variance never touches them |
| C4 | **Topic-vs-stance FP** — Normal clip discussing its own harvest keyword | 5 | 23% | **not significant** (p = 0.5022); OCR-free stance signal would be needed, and no legal channel exists |
| C5 | **Hateful-class miss** — gold Hateful, err rate 0.2941 > Offensive 0.2500 | 5 | 23% | **LOCKED** — F82 graded oracle ZH +0.0256 < +0.030 |
| C6 | **ASR-corrupted transcript** — wrong-language / repetition-loop transcript | 3 | 14% | **not significant** (p = 0.2552); ceiling priced in §7.1 |
| — | **Protocol-labile pool** (distinct from the core: 14 items, only 5 in the core) | 14 | 9.4% of test | **OPEN, user-gated** — §1 |

---

## 7. SOLUTION MAPPING

### 7.1 Alternative ZH text channel (Whisper ASR) — NEWLY MEASURED, priced DEAD at $0

The task flagged that Whisper ASR text was never measured on ZH (F64/LAUD tested Whisper *encoder hidden
states*, a different object, and killed it on ZH at −0.0052 / −0.0082). Confirmed unmeasured as a *text*
channel. It is now priced without spending GPU (`errpat_zh_asr_ceiling.py`):

- ~~The deployed transcript is already ASR-derived, so~~ our `whisper-large-v3` K4 run is a **re-run of the same
  channel, not a new modality**. Median character-bigram overlap between the two = **0.5227**; 45 of 149 items
  ≥ 0.80. The residual disagreement is homophone-level (e.g. deployed 妈宝男 vs ours 妈保男).
- **Neither transcript is systematically more complete:** counting punctuation-stripped content characters,
  our run is *shorter* on 66 of 149 items although its median is marginally longer (70 vs 63 chars). There is
  no direction in which a swap is an upgrade.
- Decisive: on the 22 core errors, our independent run recovers materially more speech (≥ 50 chars) on
  **2 items only** — `BV1Lj411D76g` and `BV1jk4y1Q7JZ`, the same two under either normalisation.
  **Gold-cheat oracle ceiling if both flipped = +0.0134** — under half the bar, before any conversion loss.
- The thin-transcript cluster C2 is therefore a **speech-absence fact, not a transcription failure**: across
  its 11 core errors Whisper's length delta runs −45 to +12 chars on **10 of 11** (median −2), with
  `BV1jk4y1Q7JZ` the lone outlier at +296. The videos are quiet; no transcript can fill them.

> **ERRATUM 2026-07-28 (propagated from `LITSWEEP3_ZH_SPECIFIC.md:29-37`, F77 / commit `d4af64b`).** The
> struck premise above is wrong: the **deployed ZH text is the Bilibili DESCRIPTION/metadata, not the
> Whisper ASR** (`gt["text"]`, median **106 Chinese characters**, 42 % of train rows carrying literal
> `<em class="keyword">…</em>` markup). The Whisper ASR is a **separate, non-deployed** file
> (`data/ASR/MHC_zh/*_asrK4_whisper-large-v3.jsonl`). **Every number in §7.1 stands** — the bigram overlap
> 0.5227, the 66-of-149 length comparison, the 2-item recovery set and the +0.0134 ceiling were all
> *measured* by comparing the two strings directly, never inferred from the premise. What changes is the
> framing: the comparison is deployed-**description** vs Whisper-**ASR** (two different channels that happen
> to overlap at 0.52), which if anything makes the +0.0134 ceiling *more* surprising, not less. The ruling
> below is unaffected.

**Ruling: ZH ASR re-channelling is arithmetically capped below the bar at $0. Do not spend GPU.**

### 7.2 Cluster-by-cluster

| cluster | status | ceiling / citation |
|---|---|---|
| C1a no-channel-knows (8) | **LOCKED** | irreducible in the banked dual-stream; would need a new input channel, and every one is closed or vetoed: OCR **VETOED**, audio F41/F64, segments F37/F66/ISR, resolution F76 (park, anti-correlated), frames F67 |
| C1b one-channel-knows (12) | **LOCKED** | +0.0805 in principle, but reachable **only** by per-item selection: F47 (dead at unsupervised, train-supervised, dev-supervised) and F66 (91–98% of oracle headroom formally selection-locked). This report adds the ZH per-item instance of that arithmetic |
| C1c fusion-loses-it (2) | **LOCKED** | +0.0134 = 2 items; F85 measured concat null, F50 fixed composition |
| C2 thin transcript (11) | **effectively LOCKED** | +0.0738 if all 11 flipped, but §7.1 shows no better transcript exists; the deficit is signal absence. No legal unmeasured lever found |
| C3 FN-locked (12) | **LOCKED at seed level** | head-recipe F73, loss family F75, ELR F79, curation F78 all closed/parked; FNs never flip across seeds so nothing in the head-seed family reaches them |
| C4 topic-vs-stance FP (5) | not significant | a stance signal is what would separate them; text-side re-extraction closed (F80 language, F70 readout), OCR **VETOED** |
| C5 Hateful-class miss (5) | **LOCKED** | F82: ZH graded oracle +0.0256 < +0.030, honest proxy monotone-negative |
| C6 ASR-corrupted (3) | not significant | §7.1 |
| global recalibration | **LOCKED** | test-fitted oracle mean **+0.0201** < +0.030; B5 per-encoder threshold calibration already closed this class |
| **protocol retirement** | **OPEN — user-gated** | **+0.0134 acc / +0.0159 mF1, 3/3 seeds, measured not oracle** (§1) |

### 7.3 Genuinely open, in-box, at $0

**Honest answer: nothing on the performance axis.** Every cluster in §6 maps to a closed axis, a vetoed
channel, or an arithmetic cap below the bar. The largest measured, legally-available move on ZH remains the
protocol question in §1, and that is a user ruling rather than an experiment. Two by-products are worth
banking as paper analysis rather than as bets:

1. **The trained RGCL head buys ≈ 0 accuracy over a raw-key kNN vote on ZH.** Same vote operator on the
   pre-head L2-concat key gives 0.8456 — exactly the deployed 3-seed mean — and text-only raw gives 0.8523.
   What the head does is sharpen margins in both directions (correct purity 0.85 → 0.9833, error purity
   0.40 → 0.1167). Single-draw, pre-head, diagnostic only; it should not be quoted as a floor.
2. **A mechanistic upgrade for the analysis chapter:** ZH errors are *neighbourhood inversions with the right
   analogue at rank ~1*, not coverage gaps and not boundary cases. That is the per-item face of law-I and it
   explains why every reranking, reweighting and calibration lever measured on ZH returned null.

### 7.4 Two quantified opportunities, ranked

| | opportunity | measured / oracle | ceiling | gate |
|---|---|---|---|---|
| 1 | **Retire the 78-item val-selected protocol** | **measured**, 3/3 seeds | **+0.0134 acc / +0.0159 mF1** — converts ZH to a clean single-protocol PASS (final-epoch already +0.0313/+0.0453 vs CLIP, 3/3) | **user ruling** (rule-shopping exposure, §1.7) |
| 2 | ZH Whisper-ASR text re-channelling | gold-cheat oracle | **+0.0134** (2 items) | **CLOSED at $0** by §7.1 — recommend no GPU |
| 3 | Global threshold recalibration | test-fitted oracle | +0.0201 | **CLOSED** — oracle already under bar; B5 precedent |

---

## 8. SCRIPTS AND OUTPUTS

All under `scripts/analysis/`, prefix `errpat_`; no `src/` file modified, nothing deleted or moved, no commit.

| script | output |
|---|---|
| `errpat_zh_trainlog_curves.py` | `errpat_zh_curves_OUT.json` — Tier-1 per-epoch curves, protocol arithmetic |
| `errpat_zh_remint.py` | `errpat_remint_dumps/errpat_zh_remint_seed{0,1,2}.pkl` — per-item, per-epoch votes + top-20 |
| `errpat_zh_remint_fidelity.py` | `errpat_zh_fidelity_OUT.json` — proxy pricing |
| `errpat_zh_taxonomy.py` | `errpat_zh_taxonomy_OUT.json` — inventory, vote structure, flips, streams, covariates |
| `errpat_zh_clusters.py` | `errpat_zh_clusters_OUT.json` — purity bands, coverage-vs-ranking, exemplars |
| `errpat_zh_opps.py` | `errpat_zh_opps_OUT.json` — pre-head vs head purity, ASR redundancy, ceilings |
| `errpat_zh_midlen.py` | stdout — thin-transcript cluster diagnosis |
| `errpat_zh_mech.py` | `errpat_zh_mech_OUT.json` — ASR-corruption and topic-stance scoring |
| `errpat_zh_asr_ceiling.py` | `errpat_zh_asr_ceiling_OUT.json` — §7.1 pricing |
| `errpat_zh_perms.py` | `errpat_zh_perms_OUT.json` — permutation tests |
| `errpat_zh_c2_settle.py` | `errpat_zh_c2_settle_OUT.json` — canonical C2 band definition + robustness |

Re-mint training artefacts landed in the fresh group `logging/Retrieval/MHC_zh/errpat_zh_remint_v2/`
(plus one aborted `errpat_zh_remint/` dir from the first attempt); no banked group was written to.
