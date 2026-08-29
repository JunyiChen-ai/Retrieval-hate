# CAND-2 FORENSIC RECON — retrieval-mined hard-negative contrastive SFT *curriculum*

**Agent:** wave-5 candidate-2 forensic recon (read-only; **ZERO GPU / ZERO test-touch / ZERO Modal spend /
ZERO user interaction**). Reading + forensic reasoning + prereg-shaped design only. Deliverable = this
committed doc. **Date:** 2026-07-17.

**Cell under recon (cand-2, `refine-logs/WAVE5_CANDIDATES.md` §cand-2, 7166232):** a generative yes/no
LoRA-SFT of Qwen2.5-VL-7B in the **same encoder-level regime as B3 / LoRA-HateMM** (r16/α32, `stage: sft`
CAUSAL_LM, vision+projector frozen), but whose **training set is constructed from the RGCL archive's
confusable neighbours** — per anchor, mine its top-k cross-label neighbours from the own-train-split memory
under the frozen encoder, and let that confusion structure shape the SFT curriculum. Features → the standard
fresh RGCL align-fusion head + top-20 kNN (archive OFF, B3 protocol), 3-seed paired vs frozen-CLIP,
dual-protocol +0.03/+0.03. Own-train gold labels only (allowed in training); single-dataset per arm; no OCR;
no cross-dataset mixing.

**Governing docs read this recon (verbatim):** `WAVE5_CANDIDATES.md` §cand-2/§2/§5 (the outline this doc
deepens), `C3GEO_FORENSIC_RECON.md` (**the load-bearing prior kill** — retrieval-mined hard negatives already
adjudicated D7-dead once), `B3_PREREG_REVIEW.md` (ZH per-seed numbers), `B3_ZH_LORA_DECOMPOSITION.md` (F45),
`LORA_HATEMM_FORENSIC_RECON.md` (regime disambiguation + HateMM prior + branch structure), `ENCODER_SWAP_DIAGNOSIS.md`
(F44), `src/utils/build_lora_sft_data.py`, `src/utils/generate_VideoMLLM_embedding_lora_HF.py` (extraction
prompt — decisive for the leakage audit), `scripts/analysis/cross_channel_router_gate.py` (the mining
machinery to reuse), `src/utils/retrieval.py`/`src/model/loss.py` (the head's *existing* mining loop),
`state/directions_tried.json` (bans + `positives_bank`), novelty-scope memory.

---

## BOTTOM LINE UP FRONT

1. **RULING = GO-IF (conditional), not a clean GO — and it is ONE mechanism-distinction away from the C3GEO
   pre-kill.** Cand-2 is the strongest *surviving* novelty-bearing adaptation member, but the closest prior
   (`C3GEO_FORENSIC_RECON.md`, 2026-07-15) already KILLED "retrieval-mined hard negatives" **before ceremony**
   as a D7-dead textbook trick. Cand-2 survives that kill on exactly **one** load-bearing difference (§2):
   C3-geo re-sourced hard negatives for the **head's triplet loss** — which *already* mines global
   hardest-opposite-label pairs per-epoch in its own evolving space (`retrieval.py:347-353,480`), making the
   re-source redundant (the P3 "probe passes, train flat" pattern). Cand-2 instead shapes the **encoder's
   generative-SFT capacity allocation** — a channel the head's frozen-feature mining structurally *cannot*
   substitute for. That distinction is real and keeps cand-2 legal and non-redundant, but it is thin, and
   whether it clears D7 is a **user sub-ruling**, not a recon call.

2. **The performance ceiling is "strengthen what generic LoRA already delivers," NOT "reach a dataset generic
   LoRA can't."** By the F44/F45 modality-locus arithmetic (§3), a text-stream curriculum can (a) hold +
   possibly *solidify* the marginal ZH pass, and (b) inherit — not newly convert — HateMM's image-borne pass.
   It opens **no new dataset** (EN label-limited/dead; HateMM image-borne). So cand-2's realistic best case is
   a **cleaner, coupling-novel, protocol-robust 2-dataset story** on the datasets generic LoRA already passes —
   a novelty + robustness upgrade, not a new performance route. Performance prior on a *new* ≥2-dataset
   conjunct generic LoRA doesn't already deliver: **~5%.**

3. **Only ONE of the three curriculum constructions is leakage-clean AND cost-neutral: confusion-weighted
   sampling of the standard single-video yes/no prompts (design (i)).** The decisive fact (§2.2): the
   feature-extraction path (`generate_VideoMLLM_embedding_lora_HF.py`) deploys **FIXED single-video neutral
   instructions**, never the SFT prompt and never neighbour context. So paired-prompt / neighbour-context
   curricula (designs (ii)/(iii)) train the adapter on a prompt *shape* it never sees at extraction — a
   train/deploy mismatch that is a performance *cost*, not a leak. Design (i) changes only *which/how-often*
   the identical single-video records appear; mining gold labels select the multiset, the deployed encoder
   input stays label-free and single-video. **Recommend design (i) only.**

4. **Sequencing = queue NOTHING now; per-branch on the LoRA-HateMM verdict (task chain 13233→13234→13235,
   confirmed PENDING JobHeldUser this recon):**
   - **BRANCH A (LoRA-HateMM PASSES, ~75-85%):** performance conjunct met by generic LoRA (ZH+HateMM). Cand-2
     runs **only if the user opens a D7 sub-ruling** and wants a coupling-novelty + ZH-robustness upgrade — as
     a **novelty upgrade on ZH+HateMM** (hold both passes, earn the add-over-generic margin), **never** as a
     new-dataset bet. Prereg skeleton (Appendix) is the ready reserve.
   - **BRANCH B (LoRA-HateMM FAILS, ~15-25%):** generic LoRA is ZH-specific; a text-stream curriculum **cannot
     supply the missing second dataset** (HateMM is image-borne, EN label-limited — §3.2). Cand-2's only
     residual value is ZH marginal→solid = **single dataset**, which does not meet the ≥2-dataset goal.
     **NO-GO; escalate to the user** (adaptation family exhausted on performance). A single ZH curriculum run
     for a stronger paper ZH row is a user call, not autonomous.

5. **Cost = ~7-8 A100-hours for the two-dataset arm** (ZH ~3-3.5h SFT + HateMM ~3.5h SFT + 2×~0.4h extract +
   ~2×2min head), **local SLURM only** (SFT is a training run; Modal is features-only). Curriculum is built to
   hold total SFT step-count ≈ generic (duplication with a size cap — §2.3), so per-run cost ≈ B3/LoRA-HateMM.
   The mining is **~$0 CPU** over the already-banked frozen-Qwen train cache (no new GPU forward).

---

## 1. MACHINERY INVENTORY — what exists, what must be authored

### 1.1 The proven pipeline cand-2 rides (the 13233-chain, verified in queue this recon)

The LoRA-SFT → extraction → 3-seed head pipeline is proven and currently queued for HateMM:

```
13233 lora_sft  PD (JobHeldUser)   sbatch scripts/slurm/lora_sft.sbatch HateMM
13234 lora_emb  PD (JobHeldUser)   gen_embed_lora.sbatch HateMM logging/lora/HateMM
13235 enc3seed  PD (JobHeldUser)   3-seed align-fusion head over the LoRA cache
```

Cand-2 reuses **every stage unchanged** except the *content* of the SFT `train.json` at stage 0:

| stage | artifact | reuse for cand-2 |
|---|---|---|
| 0 build SFT data | `src/utils/build_lora_sft_data.py` (`--dataset {MHC,MHC_zh,HateMM}`, `--answer yesno`) | **the ONLY code that changes** — a curriculum builder replaces the uniform `build_split` (§1.3) |
| 1 LoRA-SFT | `scripts/slurm/lora_sft.sbatch` + `<ds>_qwen25vl_lora_sft.yaml` (r16/α32, stage sft, vision+proj frozen) | config clone: `dataset:` → the curriculum key; **recipe byte-identical** |
| 2 extraction | `scripts/slurm/gen_embed_lora.sbatch` + `generate_VideoMLLM_embedding_lora_HF.py` (dataset-generic) | **unchanged** — reads the adapter, emits `..._LoRA_HF.pt` |
| 3 head | `enc3seed.sbatch` CONFIGS + `run_rac.py` (align, topk20, hybrid triplet+BCE, archive OFF) | **unchanged** — same fresh-head protocol as B3 |

### 1.2 Mining machinery — already written, needs no GPU

The confusable-neighbour mining is a **CPU load over banked frozen features** — no new forward:

- **Banked features (verified on disk):** `data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct_HF.pt` and
  `data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct_HF.pt` (keys `ids/img_feats/text_feats/labels`,
  3584-d frozen-Qwen pooled). Mining runs over these directly.
- **Vote/kNN machinery to lift verbatim:** `cross_channel_router_gate.py:73-131` — `_weighted_signed_vote()`
  (arithmetic rank-weighted signed-cosine top-20), `raw_modality_vote(..., exclude_self=True)` (LOO kNN over
  raw frozen single-modality features), `knn_channel(..., exclude_self=True)` (per-item vote + `topsim`,
  `simmargin`, `purity` neighbour stats). These give, per anchor, exactly the two mining signals cand-2 needs:
  (a) **closeness-to-boundary** = |LOO signed vote| (small ⇒ confusable), and (b) **hardest opposite-label
  neighbour** = walk the FAISS-ranked list to the first cross-label id.
- **Mining is over FROZEN features with a fresh LOO kNN — NOT the trained head.** This matters: the *trained
  head's* train-LOO is degenerate (memorizes, ~0.998 — `cross_channel_router_gate.py:344-345`), but the
  **frozen-encoder** LOO kNN is 0.72-0.81 (F47; router-gate frozen anchors), so the confusable subset is
  **non-degenerate** — there is a real error/boundary mass to target, not a memorized 1.0.

### 1.3 The exact code to author — enumerated, ~80-120 new LOC, NO overwrite

Per the refine-loop no-overwrite discipline, author a **new** file (do not edit `build_lora_sft_data.py`):
`src/utils/build_curriculum_sft_data.py`.

| function | source | est. LOC | what it does |
|---|---|---|---|
| `read_gt`, `ensure_frames`, `register_dataset_info` | **import verbatim** from `build_lora_sft_data.py` | 0 | reuse |
| `mine_confusion(ds)` | new; lift faiss+vote from `cross_channel_router_gate.py:73-131` | ~40-55 | load frozen-Qwen train cache → LOO kNN (concat or text stream) → per-anchor confusion score `c_i ∈ [0,1]` (design (i-a)) or misclassified flag (design (i-b)) |
| `build_curriculum_split(items, c, cap)` | fork of `build_lora_sft_data.build_split` | ~25-35 | emit the **standard single-video yn record** but with integer duplication `dup_i = 1 + round(λ·c_i)`, then subsample-to-cap so total ≈ generic N (§2.3) |
| `main` | fork | ~15-20 | `--dataset`, `--lambda`, `--mode {softconf,error}`, `--cap`; writes `data/lora_sft/<DS>/train_curric_yn.json`, registers `<prefix>_lora_curric_yn_train` |

Config clone: `<ds>_qwen25vl_lora_curric_sft.yaml` = verbatim copy of the existing SFT yaml with `dataset:`
→ `<prefix>_lora_curric_yn_train` (and val unchanged = `<prefix>_lora_yn_val`). sbatch: add a `*_curric`
case or a `CONFIG` override arg to `lora_sft.sbatch`. **Total new/changed: 1 new py file (~100 LOC) + 1 config
clone + 1 tiny sbatch case.** Extraction, head, and the frozen-CLIP control are all untouched.

### 1.4 Data reality (verified)

Train splits / balance: **MHC_zh 579 (31.1% hateful)**, **HateMM 744 (40.1% hateful)**. Both large enough
that a size-capped curriculum (§2.3) does not shrink below the B3 (579) / LoRA-HateMM (744) SFT footprint,
so no new overfit risk beyond the generic arm. `data/lora_sft/{MHC_zh,HateMM}/train_yn.json` already exist as
the uniform baseline the curriculum builder forks.

---

## 2. CURRICULUM DESIGN — 3 constructions, leakage-audited

### 2.1 The construction menu

- **(i) confusion-weighted sampling of the standard yes/no prompts** *(RECOMMENDED)* — identical single-video
  `<image>×8 + INSTR_YN → Yes/No` records as B3, but each anchor is duplicated `dup_i = 1 + round(λ·c_i)`,
  where `c_i` is its frozen-encoder confusability. Two mining modes:
  - **(i-a) soft-confusion:** `c_i = exp(-|vote_i|/τ)` from the frozen LOO signed-cosine vote (peaks at the
    decision boundary; the "hardest same-community opposite-intent" videos F45 says carry the ZH gain).
  - **(i-b) memory-error-focus** *(= cand-3 folded in, per WAVE5 §1):* `c_i = 1{frozen LOO kNN misclassifies i}`
    — upweight exactly the videos the current memory vote gets wrong. Simplest binary curriculum.
- **(ii) contrastive paired prompts** "video A vs its confusable neighbour B" — one SFT record presents the
  anchor and its hardest opposite-label neighbour and asks to distinguish. **NOT RECOMMENDED (§2.2).**
- **(iii) neighbour-context prompts** — append the neighbour's transcript (and/or gold label) as context to
  the anchor's prompt. **NOT RECOMMENDED (§2.2).**

### 2.2 Leakage audit — the decisive fact is the extraction prompt

**`generate_VideoMLLM_embedding_lora_HF.py` deploys FIXED single-video neutral instructions** (`IMG_INSTRUCTION`,
`TEXT_INSTRUCTION`, `:59-64`), one video at a time, **no neighbour, no label** — the SFT prompt is *never*
used at extraction (the SFT only shapes the adapter weights). This governs the whole audit:

| design | training-time gold use | deployed encoder input | leakage verdict | other cost |
|---|---|---|---|---|
| **(i)** | mining uses own-train gold labels to compute `c_i` and select/weight the multiset | **fixed single-video, label-free** (unchanged from B3) | **CLEAN** — gold never enters the deployed path; only the *multiset of training videos* changes | none: SFT prompt shape identical to B3 |
| (ii) | neighbour B's identity (and implicitly its label) inside the SFT prompt | fixed single-video, label-free | **bounded** — deployed path is still label-free single-video, so no test/dev/neighbour-label smuggling | **train/deploy prompt-SHAPE mismatch**: adapter tuned on 2-video prompts, extracted on 1-video prompts ⇒ likely *hurts* |
| (iii) | neighbour transcript/label as context | fixed single-video, label-free | **bounded** (same as (ii)) | same prompt-shape mismatch + risk of the adapter learning to *depend* on context it never gets at extraction |

**Conclusion:** designs (ii)/(iii) are not *leaky* (the deployed encoder never sees a neighbour or a label),
but they are *mechanically self-defeating* because extraction reverts to the fixed single-video prompt. Only
**design (i)** keeps the SFT prompt shape byte-identical to the deployed/extracted shape, so the adapter is
tuned for the exact conditions it is read under. **Lead with (i-a); (i-b) as the cheap binary variant.**

One residual leakage guard for (i), pre-declared: the mining LOO kNN must run with `exclude_self=True` over
**train only** (never dev/test), and `c_i` must be a function of *train* neighbours only — no dev/test video
ever enters the mining index. (Enforced by loading only `train_*.pt`.)

### 2.3 Cost-neutrality construction (hold SFT steps ≈ generic)

Naive duplication inflates the epoch and the SFT wall-time. To keep per-run cost ≈ B3/LoRA-HateMM, cap the
curriculum multiset at `|curric| ≤ 1.0×N_train` by **subsampling the low-`c_i` (easy) tail** to make room for
the duplicated high-`c_i` (hard) head — i.e. re-weight, don't inflate. Net effect: the 3-epoch SFT sees the
same number of examples, but the *mass* is shifted onto the confusable boundary. This makes the run a true
apples-to-apples "same budget, different curriculum" comparison against generic LoRA (strengthens the
add-over-generic attribution) and holds cost at ~3-3.5h/dataset.

### 2.4 Mechanism prior, grounded in F44/F45/C3GEO

- **The channel cand-2 uses that the head's mining cannot:** the downstream RGCL head already mines the
  hardest opposite-label pairs per-epoch in its evolving space (`retrieval.py:347-353,480`; `loss.py:453-455`).
  But that mining reads **frozen extracted features** — it can only *exploit* whatever separation the encoder
  already put on the boundary; it cannot make the encoder *allocate representation capacity* there. Cand-2's
  curriculum is the only lever that spends the r16/3-epoch encoder-adaptation budget preferentially on the
  confusable region. **This is the exact non-redundancy that C3-geo lacked** (C3-geo changed *which* hard
  negatives a saturated head loss saw; cand-2 changes *where the encoder spends capacity*).
- **Why the boundary is the right place (F45):** the ZH gain lives entirely in the text stream, and it is a
  Pareto minority-recall conversion — "the frozen-Qwen text edge is large enough to re-rank but not re-decide;
  LoRA's is large enough to re-decide" (`B3_ZH_LORA_DECOMPOSITION.md:105-108`). Concentrating adaptation on
  the same-community opposite-intent boundary is a mechanism-aligned way to push *more* of the marginal
  re-rankable mass across the decision boundary — plausibly deepening the +0.111 hate-recall Pareto move that
  is currently marginal (+0.0313).
- **Why HateMM inherits, doesn't convert (F44):** HateMM decides on the image stream (train-LOO img AUC 0.826,
  highest of the three); a text-curriculum sharpens the secondary modality, so it rides the frozen/generic-LoRA
  image-borne pass without adding on top (`LORA_HATEMM_FORENSIC_RECON.md:164-180`).

---

## 3. THE ≥2-DATASET ARITHMETIC & THE ZH-MARGIN ANGLE (quantified)

### 3.1 Structural cap (F44/F45, inherited from WAVE5 §0.2)

A text-stream adaptation lever **holds ZH's pass and can add only HateMM or EN**. HateMM is image-borne ⇒
inherited-not-converted. EN is label-limited (image collapsed 0.734→0.599; even the best-ever fusion AUC 0.898
is unconvertible, F50) ⇒ dead to the whole representation family. **⇒ cand-2 opens no new dataset;** its 2nd
leg is whatever generic LoRA already carries.

### 3.2 The ZH marginal→solid angle — the concrete value, with numbers

This is cand-2's most defensible *quantitative* target (WAVE5 under-weighted it). B3 ZH final-epoch
(`B3_PREREG_REVIEW.md:36-40`), LoRA vs CLIP:

| seed | Δacc | ΔmF1 | note |
|---|---|---|---|
| 0 | **+0.0402** | +0.0475 | clears |
| 1 | **+0.0335** | +0.0571 | clears |
| 2 | **+0.0201** | +0.0313 | **below the +0.030 per-seed acc bar** |
| **mean** | **+0.0313** | **+0.0453** | acc +0.0013 above bar; **val-selected +0.0246 acc = FAIL** |

The pass is marginal on **two** counts: (a) mean Δacc only +0.0013 over the bar, carried by seeds 0/1 with
seed2 under the per-seed bar; (b) the val-selected protocol FAILs on acc (+0.0246), so the 2-dataset claim
currently **depends on protocol choice**. F45 attributes the val-sel FAIL to 78-dev selection noise (LoRA dev
plateaus at 0.8718 while test climbs), not instability.

**What margin cand-2 must earn to remove the protocol-dependence:**

- **To make final-epoch a *non-marginal* pass:** lift the weakest seed (seed2 Δacc +0.0201 → ≥ +0.030), i.e.
  ~**+0.010 acc on seed2** while holding 0/1, OR a uniform ~**+0.007 acc** across seeds → mean ≈ +0.040,
  comfortably clear of the between-seed spread. This is the K-C2-2 operational target.
- **To make val-selected *also* pass:** val-sel mean Δacc +0.0246 → ≥ +0.030 = **+0.0054 more acc under
  val-selection**. Mechanism route: a curriculum that drives the LoRA encoder to a *higher, earlier* test
  plateau (so val-selection's plateaued mid-epoch pick lands nearer the final-epoch level) would lift both
  protocols at once — the F45 diagnosis says the gap *is* the plateau-vs-final divergence.
- **Net:** if cand-2 delivers ~+0.007-0.010 acc over generic LoRA on ZH and pushes val-selected across +0.030
  while holding HateMM's inherited pass, the **2-dataset performance claim stops depending on protocol
  choice** — a genuine strengthening of the goal-relevant result even though it adds no dataset. That is the
  strongest *performance* case for cand-2, and it is modest and bounded.

---

## 4. KILL-SWITCHES + BARS (house style, adversarial)

All paired, both protocols judged independently, 3/3 sign, pre-declared. The **novelty-earning** bar is
K-C2-2 (add-over-generic) — without it, cand-2 collapses to "generic LoRA with reshuffled data."

- **K-C2-0 (mining-validity, $0 CPU pre-GPU gate).** The curriculum must be a *distinct* method, not generic
  LoRA in disguise. Compute over the frozen-Qwen train cache: (a) frozen LOO kNN train error rate must be a
  **meaningful minority** (~15-35%, non-degenerate; if ≈0 the encoder memorizes and every `c_i≈0` ⇒ curriculum
  ≡ uniform ⇒ **auto-KILL**; if ≈50% the signal is noise); (b) the confusion weights must concentrate on a
  distinct subset (weight-distribution Gini ≥ some non-trivial floor, i.e. the hard head is not the whole set);
  (c) the resulting curriculum multiset must differ from uniform by a real margin (Jaccard of duplicated-set vs
  full-set < 0.9). **If any fails → KILL pre-GPU** (the C3-geo overlap-screen precedent: a >90%-overlap number
  is a $0 epitaph).
- **K-C2-1 (performance, primary — must HOLD the inherited passes).** ZH LoRA-curriculum − CLIP: mean Δacc ≥
  +0.030 **AND** ΔmF1 ≥ +0.030, 3/3, **AND** ≥ generic-B3-LoRA − 0.014 (must not regress the pass it
  inherits). If LoRA-HateMM passed, the HateMM curriculum arm must likewise hold ≥ generic-LoRA-HateMM − 0.014.
  Below → **KILL**.
- **K-C2-2 (add-over-generic — THE NOVELTY-EARNING BAR).** The paired comparison is **against the GENERIC LoRA
  arm, not just CLIP.** ZH curriculum-LoRA must **beat generic B3 LoRA** by a real margin on ≥1 protocol
  (operationally: final-epoch mean Δacc-vs-CLIP ≥ ~+0.040 with 3/3 per-seed ≥ +0.030, **or** val-selected
  crosses +0.030 — i.e. it actually solidifies §3.2). **If it merely ties generic LoRA within head-seed noise
  (±0.014), the coupling earns NO novelty** and the route reduces to "generic LoRA with extra machinery" →
  **report as no-value**, bank the negative, do not claim the coupling.
- **KS-regression (below-generic kill).** If curriculum-LoRA lands **below the generic LoRA arm − 0.014** on
  its held leg, the curriculum *degraded* adaptation (overfit the tiny confusable subset, or the size-cap
  starved easy-example coverage) → **KILL**, bank as "confusion-curriculum hurts."
- **KS-below-floor (regime sanity).** If curriculum-LoRA lands below the **CLIP floor** on ZH — the one leg it
  was built to strengthen — bank the strong negative (curriculum broke the mechanism).

**Pre-declared collapse pattern (adversarial, the honest failure mode):** the downstream RGCL head re-mines
the confusable structure from the extracted frozen features regardless of how the encoder was SFT'd. If the
curriculum-adapted encoder produces feature geometry ≈ the generic-adapted encoder on the confusable boundary
(K-C2-2 ties), then **the head's own mining has already extracted everything the curriculum tried to inject**
— the exact P3/C3-geo "objective already sees the hard structure, curriculum is redundant, train flat" pattern,
applied one level up (encoder SFT instead of head loss). This is the **most likely** outcome (~50-60%) and it
is *not* a bug — it would be the empirical finding that even encoder-capacity-allocation is redundant with the
head's frozen-feature mining, closing the last adaptation-family cell. K-C2-2 is designed to detect exactly
this and force the honest "it's just LoRA" verdict.

---

## 5. NOVELTY ARGUMENT — strongest and weakest point (honest)

**Strongest point.** The **coupling object** is genuinely new *and* mechanistically non-redundant in a way
C3-geo was not: the retrieval memory (the archive) *constructs the encoder-adaptation curriculum*, allocating
scarce LoRA capacity to the same-community opposite-intent boundary where F45 says the convertible ZH gain
lives — a channel the head's existing hard-negative mining **cannot** reach (the head reads frozen features;
it cannot make the encoder spend capacity). This is precisely the "OBJECTIVE/COUPLING specific to the
retrieval-contrastive architecture, not a generic encoder-class lever" the wave sought, and it is
novel-in-hateful-video under the project's in-field definition (no hateful-video method SFT-adapts an encoder
on retrieval-mined confusable pairs for a kNN-vote memory).

**Weakest point (and it is serious).** `C3GEO_FORENSIC_RECON.md:125-136` already adjudicated the *sibling*
idea — retrieval-mined hard negatives — as **D7-dead**: "hard-negative mining from a foundation-model
embedding space is precisely the class of generic, textbook trick that D7 says does not count merely for
being first-in-hateful-video" (with a lit wall: NV-Retriever, CoRNStack, MoCHi, etc.). Cand-2 differs on
*injection point* (encoder-SFT-curriculum vs head-triplet-loss) and on *coupling framing* (memory constructs
adaptation), but **curriculum learning / hard-example mining for SFT is equally textbook outside hateful
video.** So the novelty rests **entirely** on (a) in-field-first coupling and (b) the injection-point
distinction being judged *meaningful* rather than cosmetic. If the user's D7 applies the C3-geo operative
principle strictly — "generic techniques don't become novel by being pointed at this domain" — **cand-2 is
covered and dies on novelty regardless of performance.** The adapted object is still the encoder (Axis-B,
D7-encoder-class), so this is a **narrower, stronger D7 *sub*-ruling** ("does a retrieval-coupled SFT
curriculum count as distinct from generic LoRA?"), **not a route that escapes D7 by construction.** That
sub-ruling is the user's, not the recon's.

---

## 6. GPU COST + SEQUENCING — both LoRA-HateMM branches spelled out

**Cost (two-dataset arm, local SLURM only — SFT cannot run on Modal features-only):**

| stage | ZH | HateMM | note |
|---|---|---|---|
| mining (CPU, banked frozen cache) | ~min | ~min | ~$0, no GPU |
| LoRA-SFT (size-capped curriculum) | ~2.8-3.3 h | ~3.1-3.5 h | ≈ B3 (579) / LoRA-HateMM (744) footprint |
| extraction | ~0.35 h | ~0.4 h | dataset-generic runner unchanged |
| 3-seed head | ~2 min | ~2 min | cached feats, fresh RGCL head |
| **total NEW GPU** | **~3.2-3.7 h** | **~3.6-4.0 h** | **~7-8 A100-h combined** |

Chainable as {SFT→extract} then {head}, per dataset. Wall time longer under `PENDING (JobHeldUser)` —
auto-release, never force (CLAUDE.md).

**BRANCH A — LoRA-HateMM PASSES (~75-85%).** Performance conjunct met by generic LoRA (ZH+HateMM); only D7
remains. **Cand-2 is NOT queued speculatively** — it adds no dataset (§3.1). Run cand-2 **iff the user opens a
D7 sub-ruling** and asks for a coupling-novelty variant: pre-registered as a **novelty + robustness upgrade on
ZH+HateMM** — ZH is the primary leg (must earn K-C2-2 add-over-generic and, ideally, cross val-selected +0.030
per §3.2); HateMM is the "hold the inherited pass" leg (K-C2-1 hold ≥ generic − 0.014). Value = converts the
2-dataset claim from "generic encoder LoRA (D7-weak, protocol-dependent on ZH)" to "memory-coupled adaptation
curriculum (D7-sub-ruling, protocol-robust on ZH)."

**BRANCH B — LoRA-HateMM FAILS (~15-25%).** Generic LoRA is ZH-specific; the second dataset is structurally
unavailable to any text-stream lever (§3.2). **Cand-2 cannot rescue HateMM** (image-borne; if generic LoRA
failed it, a text-curriculum won't convert it) and **cannot convert EN** (label-limited). So cand-2's only
residual is ZH marginal→solid = **one dataset**, which does **not** meet the ≥2-dataset goal. **NO-GO for the
goal; escalate to the user** (adaptation family exhausted on performance, round-3/4 terminus stands). A single
ZH curriculum run to harden the paper's ZH row is a legitimate but *user-directed* action, not autonomous
queue-on-fail.

**Either branch:** cand-2 is **never** queued before the LoRA-HateMM verdict lands (its calculus depends on
the branch), and **never** run as a new-dataset bet.

---

## APPENDIX — PREREG SKELETON (ready to formalize on a GO-IF)

**H-C2 (per dataset D ∈ {MHC_zh, HateMM}, performance clause).** Replacing the generic-LoRA encoder with a
**retrieval-confusion-curriculum LoRA-SFT** encoder (design (i-a) soft-confusion, size-capped to ≈ N_train;
tag `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`; trained on D's own train split only; mining = frozen-Qwen LOO
kNN over D-train), every other component identical (same RGCL align-fusion head, topk=20, `lambda_seg=0`,
archive OFF, same split, lr=1e-4/ep30/bz64/proj=map=1024/dropout/hard-neg/hybrid-loss/warmup=5) — yields, on
D, mean paired Δacc ≥ +0.030 AND mean paired ΔmF1 ≥ +0.030 with 3/3 sign vs the frozen-CLIP control, judged
independently under each protocol; **AND** beats the generic-LoRA arm by ≥ the K-C2-2 margin on ≥1 protocol.

- **Arms:** (1) curriculum-LoRA (new); (2) generic-LoRA (B3 for ZH / LoRA-HateMM for HateMM — already banked);
  (3) frozen-CLIP control (12850/13115 — banked). Comparisons: (1)−(3) = K-C2-1; (1)−(2) = K-C2-2.
- **Protocols:** (A) val-selected (epoch ≥ warmup 5 max Val acc, roc tie-break); (B) final-epoch (ep29). Both
  reported, judged independently; fixed write-up "final-epoch: pass/fail; val-selected: pass/fail".
- **Gates, in order:** K-C2-0 ($0 mining-validity, pre-GPU) → G-repro-adapted (SFT smoke: loss sane/decreasing,
  no NaN, ckpt saves; head run = same-code Namespace diff vs 12850/13115 except `--model` + inert group/path)
  → K-C2-1 → K-C2-2 → KS-regression → KS-below-floor. Single test-touch per dataset (the fresh head read; note
  ZH/HateMM test are not virgin — prior enc3s arms read them; this is a re-measure under the same protocol).
- **Artifacts to author (3, diff-verified):** (i) `src/utils/build_curriculum_sft_data.py` (§1.3); (ii)
  `<ds>_qwen25vl_lora_curric_sft.yaml` (dataset-key swap only); (iii) a `*_curric` case in `lora_sft.sbatch`.
  Head runner = `enc3seed.sbatch` + curriculum-LoRA rows + fresh GROUP.
- **Honesty clauses (mandatory, mirror B3):** novelty = PENDING USER D7 SUB-RULING (not decided by this
  experiment); single-encoder-draw limitation (3 head-seeds read ONE curriculum-SFT draw ⇒ head-seed variance,
  not SFT-draw variance — symmetric with the single-draw controls); marginal-pass language carries if ZH lands
  marginal again; the add-over-generic (K-C2-2) result, not the vs-CLIP result, is what any novelty claim rests
  on.
- **Framing sentence (verbatim):** *this measurement tests whether a retrieval-coupled adaptation curriculum
  adds over generic encoder LoRA; a PASS strengthens the case for a D7 sub-ruling that memory→adaptation
  coupling is novel-in-field, but the ruling remains the user's.*

---

## PROVENANCE
- Prior kill (load-bearing): `refine-logs/C3GEO_FORENSIC_RECON.md` (retrieval-mined hard-neg = D7-dead
  textbook trick; head already mines global hardest-opposite-label online — `:8,12-59,125-136`); dead entry
  `state/directions_tried.json` #19 `R3-C3geo`.
- Head's existing mining: `src/utils/retrieval.py:347-353` (head-space index rebuild), `:480` (opposite-label
  selection), `:497` (same-label pseudo-positive); `src/model/loss.py:453-455` (triplet+BCE hybrid);
  `src/run_rac.py:202-240` (hard-neg config), `:581-589` (per-epoch/step re-index).
- Mining machinery to reuse ($0 CPU): `scripts/analysis/cross_channel_router_gate.py:73-131`
  (`_weighted_signed_vote`, `raw_modality_vote`, `knn_channel`, exclude_self LOO); frozen train caches
  `data/CLIP_Embedding/{MHC_zh,HateMM}/train_Qwen2.5-VL-7B-Instruct_HF.pt`.
- Extraction prompt (leakage audit): `src/utils/generate_VideoMLLM_embedding_lora_HF.py:29-46,58-64,264-380`
  (FIXED single-video `IMG_INSTRUCTION`/`TEXT_INSTRUCTION`; SFT prompt never used at extraction).
- SFT machinery: `src/utils/build_lora_sft_data.py` (yn schema, `--dataset`/`--answer`); `scripts/slurm/lora_sft.sbatch`
  (MHC/MHC_zh/HateMM cases); `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/mhc_zh_qwen25vl_lora_sft.yaml`
  (r16/α32, stage sft, vision+proj frozen); 13233→13234→13235 chain PENDING JobHeldUser (verified `squeue`).
- ZH numbers: `refine-logs/B3_PREREG_REVIEW.md:36-60` (final-ep +0.0313/+0.0453 3/3, seed2 +0.0201 sub-bar;
  val-sel +0.0246 FAIL); mechanism `refine-logs/B3_ZH_LORA_DECOMPOSITION.md:23-108` (F45, text-stream Pareto).
- Arithmetic / branches: `refine-logs/LORA_HATEMM_FORENSIC_RECON.md:33-49,146-194,283-294`;
  `refine-logs/ENCODER_SWAP_DIAGNOSIS.md` (F44); `refine-logs/WAVE5_CANDIDATES.md` §0.2/§2/§4/§5.
- D7 / bans / novelty: `state/directions_tried.json` `banned_constraints[]`, `positives_bank[B3-lora-zh]`;
  novelty-scope memory (in-field novelty definition); train balance MHC_zh 579/31.1%, HateMM 744/40.1%.
- **Required statements:** ZERO GPU / SLURM / Modal spent by this recon; no held-out test metric read or
  produced; no `state/`, prereg, config, `research-wiki/`, or frozen LoRA-HateMM artifact mutated. Committed
  on `main`, not pushed.
