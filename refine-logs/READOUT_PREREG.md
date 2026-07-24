# READOUT-GRID Pre-Registration — layer / token / prompt readout (R0–R3) on the deployed merged encoder, ZH + HateMM

**Author:** readout-grid prereg author (CPU-only; no GPU/SLURM/Modal spent; NO job submitted).
**Date:** 2026-07-25 NZST.
**Status:** `DRAFT — AWAITING INDEPENDENT 0-CONTEXT REVIEW + HASH-FREEZE.` No test metric produced; no job submitted; zero GPU spent.
**Implements:** `refine-logs/READOUT_FORENSIC_RECON.md` (commit `61a9f4a`, the GO-IF recon) — its §§4–8 design (grid R0–R3,
amortization, cache tags, $0 screen, kill-bars) transcribed and re-verified below. Deviations from the recon are flagged
**loudly** in §11.
**House-style precedent:** `refine-logs/FRAME16_PREREG.md` (closest shape: extraction-then-head, determinism honesty,
freeze block, single-submit plan), `refine-logs/CAND2_CURRICULUM_PREREG.md` (grid/screen discipline, floors, freeze),
`research-wiki/experiments/exp-encoder-3seed.md:73-85` (decision rule verbatim). $0-screen machinery lifted from
`refine-logs/ISR_PREGATE_RECORD.md` + `refine-logs/LP_GATE_RECORD.md` + `scripts/analysis/cross_channel_router_gate.py:73-79`.

## Title + claim scope (verbatim)

> This measurement tests **the one axis our extraction pipeline never varied — the READOUT (which transformer LAYER, which
> TOKEN/pooling span, and which PROMPT the video-level embedding is read from)** — on the **deployed merged encoder**
> (frozen base + already-trained LoRA adapter), on **ZH (`MHC_zh`, primary — harden the marginal val-selected leg, F45)**
> and **HateMM (curric, hold check)**. It changes **NOTHING about the encoder, the training, or the loader contract**: the
> merged frozen forward is byte-identical to the deployed one — only which layer/token/prompt the same output is read from
> changes. A **$0 CPU dev screen** (raw fused-key kNN vote, ISR/LP machinery) is the KILL-or-promote gate; **at most one
> cell** advances to a **single test-touch** verdict. It makes **NO novelty claim standing alone — a layer/token/prompt
> choice is extraction engineering (D7-thin, F0.3)**; even a formal PASS is a performance/robustness row whose novelty
> weight (as a component of the MLLM-embedding-paradigm story) is the **USER's** D7 call. This prereg decides the
> **performance clause only.** The one-word readout is a **SINGLE readout, NOT multi-prompt ensembling** (F0.6) — the
> ensembling micro-ruling stays unconsumed.

The grid = deployed baseline `R0` + **3** new cells per dataset (≤4-cell cap honored). EN is **excluded** (label-capped at
five levels, F44/F55 — F0.4). `state/` is untouched; the bidirectional-encoder chain (parallel prereg) submits **after**
this chain clears (§6).

---

## 0. Binding facts / honesty clauses (all present; pre-declared)

**F0.1 — Test is NOT virgin (declared).** ZH and HateMM test were already read under the identical `enc3s`/12850 protocol by:
frozen-CLIP (ZH 13115, HateMM 12850), frozen-Qwen (12850), generic-LoRA (ZH 13150 = B3), curric-LoRA (ZH+HateMM 13241),
and the LoRA-HateMM / cand-2 verdicts. This prereg's verdict head reads are **re-measurements under the identical
protocol**, not first exposures. Each promoted cell consumes exactly ONE budgeted **readout-cell** test evaluation (the 3
head-seed reads on the winner cache). **Zero test-touch before the independent verdict.** The $0 dev screen (§1.2) and the
optional tie-break (§3.6) open **only** `train_*.pt` / `dev_seen_*.pt` (`readout_screen.py:load_cache` hard-asserts
`split ∈ {train, dev_seen}`; the tie-break uses a throwaway `test_seen := dev_seen` copy so the REAL test cache is never
opened — §3.6).

**F0.2 — Extraction is fully deterministic; there is NO stochastic encoder draw (STRONGER than the LoRA F0.2, mirrors
FRAME16 F0.2).** The four grid caches are produced by ONE merged-frozen forward per (prompt, stream); the forward is
deterministic given (base weights, merged adapter, sampled frame indices, max_pixels): frame sampling is `np.linspace`
(no RNG), `attn=sdpa`, `bf16`, `no_grad`, `output_hidden_states=True`, single forward, adapter merged via
`merge_and_unload` (deterministic given the fixed adapter). The R0 recompute is therefore expected to reproduce the banked
deployed cache **BIT-EXACT** (the clobber-guard / G-repro anchor, §4.1a). The reported head ±band at verdict is **purely
head-seed variance**, symmetric with the banked R0 floor (also 3 head-seeds over one deterministic cache). **No
single-SFT-draw caveat applies** — no SFT happens; the adapter is the already-banked deployed one.

**F0.3 — Novelty = D7-THIN standing alone; the D7 weight is the USER's (not decided here).** A layer/token/prompt choice
is **extraction engineering**, not a method mechanism. Standing alone it is **novelty-thin** (recon §8). Its value is (a)
hardening the marginal ZH val-selected leg (performance/robustness, F45) and (b) as a **component of the MLLM-embedding-
paradigm story** ("principled readout for hate-video MLLM embeddings", F68-P1). Even a formal PASS needs the paradigm
framing to carry D7 weight, and **that framing call is the USER's** — same as every encoder-adjacent cell. (Unlike the
litsurvey B-axis candidates, this cell does NOT need the encoder-class D7 sub-ruling — F0.6/recon §6: nothing about the
encoder changes.) **This prereg decides the performance clause only.**

**F0.4 — Structural ceiling: this grid opens NO new dataset; EN excluded honestly (pre-declared, material).** A readout
change is a **representation-READ** change; the F44/F55 arithmetic caps representation-side gains on EN independent of how
the vector is read (EN proven label-limited at five levels: F44 frozen, F50 collapsed-adapted, F55 healthy-image, F58, F65
vision-unfreeze). EN prior ≈ near-0 ⇒ **EN is SKIPPED** (a frozen-EN hold could be added at ~1 GPU-h if a reviewer insists;
not recommended). ZH is image-text with a text-borne marginal leg (F45) — the primary target. HateMM already passes both
protocols (F53/cand-2) — the **hold** check (must not regress). Realistic best case = a cleaner, protocol-robust ZH leg on
a dataset already passing; **not** a new performance route (recon §8, prior LOW–MODEST ~15–20% ZH).

**F0.5 — This is NOT a $0 gate; ~2 GPU-h of mandatory local re-extraction is the entry price (pre-declared, material).**
Unlike ISR (F66) and LP, which screened on already-banked caches, the readout axis **cannot** be screened on any banked
artifact — the deployed extractor discards intermediate hidden states and per-token positions after pooling (only the
final-layer mean-pool is banked). So ~2 GPU-h of local re-extraction (ZH+HateMM, videos never leave the node) is required
to even reach the $0 CPU screen. The screen then decides whether any verdict GPU is spent. Most-likely outcome (consistent
with terminus): flat screen ⇒ **KS-readout-dead** at ~2 GPU-h, no test-touch.

**F0.6 — Ban / isomorphism check: NOT ensembling, NOT encoder-class, NOT temporal, NOT schema-distill (verified vs
recon §6).**
- **NOT multi-prompt ENSEMBLING.** Every cell is a **single** readout (one layer, one span, one prompt). The grid
  **compares** single readouts and picks ONE; it **never averages** across prompts/layers. The MetaEOL multi-prompt
  *ensemble* (2402.18458) is **explicitly excluded**; the core grid needs **no** user micro-ruling because it does no
  ensembling. The one-word prompt is a single PromptEOL-style readout.
- **NOT F24 / encoder-class.** No new encoder, no retraining, no LoRA re-fit, no mask surgery. Base + adapter are
  byte-identical to the deployed encoder; the merged frozen forward is identical (`_encode`/pooling math bit-verified,
  §4.1b). Only the read changes. Sidesteps the F24 encoder-class veto and does NOT need the B-axis D7 sub-ruling.
- **NOT F35–F39 / F67 (temporal / frame axis).** Frames held at the deployed **8** (`--num_frames 8`); no frame-group
  object, no per-segment re-encode (that is ISR/F66), no denser sampling (F67 killed 8→16). The layer/token/prompt axis is
  orthogonal to frame handling.
- **NOT P4 (schema-distill).** P4 changed the SFT training target; this grid changes nothing about training — pure
  extraction-time read of the SAME merged forward.

**F0.7 — Single-dataset own-train-split VETO compliance (hard user veto; trivially clean — no training).** No SFT: the
readout extractor reads each video + the fixed instructions only; no labels enter the encoder path. The $0 screen and the
verdict head each use **own-dataset own-split** data only (ZH head trains on ZH train; HateMM on HateMM train). NO
cross-dataset mixing, NO gold spans/attributes, NO OCR, raw videos never leave the node (the extractor extracts locally,
B2-pushes derived `.pt` only). All standing vetoes cleared.

---

## 1. Pipeline spec — fully pinned (3 stages; NO SFT; nothing left to interpretation)

**Stage 0 — none.** No SFT, no data build. The readout extractor consumes `data/gt/{MHC_zh,HateMM}/{train,val,test}.jsonl`
+ `data/video/<DS>/All/<id>.mp4` and the two DEPLOYED adapters directly (verified present §5.2).

### 1.1 Stage 1 — readout-grid feature extraction (ONE combined SLURM job; NO edit to the banked extractor)

- **Submit:** `sbatch scripts/slurm/gen_embed_readout.sbatch` (artifact B, §5 — hardcodes both dataset configs).
- **New code (codex-review-gated — reads model internals):** `src/utils/generate_VideoMLLM_embedding_readout_HF.py`
  (artifact A) — a **clone** of the banked LoRA extractor `generate_VideoMLLM_embedding_lora_HF.py`. The model-load path
  (frozen base + `PeftModel.from_pretrained` + `merge_and_unload`), the frame sampler, the cache contract, and the pooling
  **math** are byte-identical. The **only** changes: (i) read `out.hidden_states[L]` for `L ∈ {28, 24}` instead of the
  hardcoded `[-1]`; (ii) a `last_token` span (`last_hidden[-1]`) in addition to the deployed prefix/response means; (iii)
  two module-level one-word prompt constants. The banked extractor file is **NOT edited** (codex gate rationale: any change
  reading model internals — hidden-state layer index, generation-position token — is codex-review-gated before GPU, per
  CLAUDE.md / precedent).
- **The FROZEN grid (4 cells/dataset — pinned in code, NO sweep):**

  | cell | layer | token/span | prompt | recipe / cite | forward source |
  |---|---|---|---|---|---|
  | **R0** (deployed) | 28 (final) | mean (img=prefix / text=response) | baseline descriptive/analytic | **deployed** `_encode` as-is | Pass A, free — **re-computes the banked R0 → BIT-EXACT clobber-guard** |
  | **R1** | **24** | mean (same spans) | baseline | intermediate-layer (**VidVec**, 2602.08099) | Pass A, free |
  | **R2** | 28 (final) | **last token** @ gen-position | **one-word** | PromptEOL/E5-V (2307.16645 / 2407.12580) | Pass B, free |
  | **R3** | **24** | **last token** @ gen-position | **one-word** | **VidVec full recipe** (intermediate + one-word) | Pass B, free |

  **Layer indexing (reviewer: re-verify the off-by-one).** Qwen2.5-VL-7B LLM `num_hidden_layers=28`, `hidden_size=3584`
  (verified this prereg from the HF config). `out.hidden_states` is a tuple of length **29**: index 0 = embedding output,
  indices 1..28 = decoder-layer outputs, `[-1]` = index 28 = final. So `hidden_states[28]` **==** the deployed
  `hidden_states[-1]` (R0 recompute), and `hidden_states[24]` = the intermediate layer (VidVec depth 24/28 ≈ 0.857). Both
  indices literature-pinned; **no layer sweep** (the GPU-free availability of all 29 layers is a forking-path trap, resisted
  by pinning {28,24} a priori in `CELLS`).

- **The PINNED one-word prompts (frozen text — reviewer: re-verify these tokenize as expected):**
  - **img (R2/R3), verbatim:** `Describe this video in one word:` — a standalone PromptEOL prompt that **replaces** the
    baseline img instruction; the last token (generation position) is read.
  - **text (R2/R3), verbatim:** the baseline analytic instruction + the per-item title/transcript + a one-word tail at the
    **very end** (so the last token compresses the whole input):
    `You are analysing a short video for potentially hateful or offensive content. Considering the frames together with the provided title and transcript, summarise the targets, symbols, tone, and any harmful intent conveyed.\nTitle: {title}\nTranscript: {transcript}\nSummarise the above in one word:`
  - **Baseline prompts (R0/R1) unchanged, verbatim:** img =
    `Describe the people, symbols, gestures, and on-screen text in this video.`; text = the analytic instruction above +
    `\nTitle: {title}\nTranscript: {transcript}` (no one-word tail). These are byte-identical to the deployed extractor
    (the R0 cell MUST reproduce the banked cache bit-exact).

- **Amortization (recon §2):** `output_hidden_states=True` is already on, so ONE forward yields all layers; reading L24 in
  addition to L28 and re-pooling a last-token span are **GPU-free**. Only the prompt change forces a new forward ⇒
  **4 forwards/item** (Pass A baseline img+text; Pass B one-word img+text) = 2× the deployed per-item cost. Frames held at
  **8**.
- **Output (per dataset, DISTINCT `-ro_*` tags — the banked R0 is NEVER clobbered, §4.3):**
  `data/CLIP_Embedding/<DS>/{train,dev_seen,test_seen}_<BASE>-ro_{L28,L24,ow_L28,ow_L24}.pt`, loader contract
  `{ids, img_feats, text_feats, labels}`, Dv=Dt=3584. Base tags: ZH `Qwen2.5-VL-7B-Instruct-LoRA_HF`;
  HateMM `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`.
- **Cost:** ~1 GPU-h/dataset (4 forwards/item, 8 frames), ~**2 GPU-h** total to reach the $0 screen. Then B2-push derived
  `.pt` only (videos never leave — CLAUDE.md data boundary).

### 1.2 Stage 2 — $0 CPU dev screen (KILL-or-promote; the decision gate)

- **Run:** `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python scripts/analysis/readout_screen.py` (artifact C) — CPU-only,
  <1 min, writes `refine-logs/READOUT_SCREEN_OUT.json`. **Test split never opened** (hard assert `split ∈ {train, dev_seen}`).
- **Object (identical for every cell — the only shared admissible $0 object, LP §1):** per video, `img_feats` (3584) +
  `text_feats` (3584) → **L2-norm each stream, concat → 7168, L2-renorm** = the raw fused key. There are NO trained head
  ckpts for the readout key-spaces (same situation LP §1 documents), so the raw fused key is the uniform screen object.
- **Vote (deployed, verbatim):** rank-weighted signed-cosine top-20 `_weighted_signed_vote`
  (`cross_channel_router_gate.py:73-79`), two arms, both reading only train + dev_seen: **(i)** video-level LOO over
  train∪dev (diagonal self-exclusion), scored on dev; **(ii)** strict dev-query → train-memory, scored on dev
  (ISR_PREGATE_RECORD §0.2 verbatim). Fused key is the **decision object**; img-only / text-only are reported as
  **diagnostics**.
- **Screen statistic:** per cell, dev **Δacc = (cell fused-key vote acc) − (R0 fused-key vote acc)**, on the same dev videos
  (n = ZH **78** / HateMM **107**), both arms. Report exact **items fixed / broken / net** (count-level).
- **Machinery-validity guards (reported, mirror ISR/LP — validity checks, not extra bars):** (a) permutation-null ≥200
  perms (shuffle memory labels, recompute best-cell−R0 Δ, real Δ vs null p95); (b) bootstrap-1000 5th-pct of Δ; (c)
  **degenerate-recovery / R0 bit-exact:** the R0 cache (`-ro_L28`) reproduces the banked deployed cache **bit-exact**
  (`img/text max|Δ| == 0` on train+dev — the G-repro anchor), and R0-vs-R0 Δacc == 0.

### 1.3 Stage 3 — 3-seed RGCL align-fusion head + kNN verdict (ONLY if the screen promotes)

- Authored at promote-time (winner unknown until the screen runs) as a **clone of `enc3seed_lora_curric.sbatch`** with
  `LORA=<BASE>-<winner_suffix>`, `GROUP_NAME=RAC_video_readout`, `CONFIGS` = 3 head-seeds of the promoted dataset(s). The
  `run_one` python block MUST stay **byte-identical** to `enc3seed_lora_curric.sbatch` (the runner that produced 13150 ZH /
  13241 HateMM); the ONLY manipulated variables vs the banked R0 arm are `--model` and `--group_name`. Config verbatim:
  `--batch_size 64 --lr 0.0001 --epochs 30 --topk 20 --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode
  align --hard_negatives_loss True --no_hard_negatives 1 --metric cos --loss triplet --hybrid_loss True --warmup 5
  --lambda_seg 0 --archive OFF`.
- **Pairing:** per head-seed (winner seed s − banked-R0 seed s), `s ∈ {0,1,2}`. Control = the banked deployed R0 head runs
  (**ZH 13150**, **HateMM 13241**, re-derived §2), NOT re-run. Because R0-`ro_L28` is bit-exact to the deployed cache
  (F0.2), a fresh head on `ro_L28` would reproduce 13150/13241 seed-for-seed; pairing vs the banked runs is therefore valid
  and avoids a redundant R0 head job. **Single test-touch per promoted dataset** = the 3 head reads on the winner cache.

---

## 2. Comparison floors — INDEPENDENTLY RE-DERIVED from raw trainlogs (numeric-provenance discipline)

Every number below was re-parsed **this prereg** from the raw trainlogs with the EXACT `enc3seed.sbatch` embedded parser
(val-sel = epoch ≥ warmup 5 max `Val_Retrieval` acc, roc tie-break; final = max epoch 29). All bit-match
`CAND2_CURRICULUM_PREREG.md` §2 to 4dp.

### 2.1 ZH R0 anchor — generic-LoRA (B3, job 13150) = the DEPLOYED ZH encoder read

| seed | val-sel Test F1 / acc (sel ep, line) | final-ep Test F1 / acc (ep29, line) |
|---|---|---|
| 0 | 0.8023 / 0.8322 (ep20, L199) | 0.8181 / 0.8456 (L272) |
| 1 | 0.7956 / 0.8255 (ep26, L248) | 0.8113 / 0.8389 (L273) |
| 2 | 0.8065 / 0.8389 (ep19, L187) | 0.8226 / 0.8523 (L268) |
| **mean** | **0.8015 / 0.8322** | **0.8173 / 0.8456** |

Provenance: `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog` (lines cited). Bit-matches
CAND2 §2.1 ZH generic-LoRA. **B3 leg is marginal / protocol-dependent** (generic−CLIP final +0.0313 acc 3/3 but seed2
+0.0201 < the +0.030 per-seed bar ⇒ MARGINAL; val-sel +0.0246 acc FAIL) — the leg a readout gain must strengthen.

### 2.2 HateMM R0 anchor — curric-LoRA (job 13241) = the DEPLOYED HateMM encoder read (project best, hold)

| seed | val-sel Test F1 / acc (sel ep, line) | final-ep Test F1 / acc (ep29, line) |
|---|---|---|
| 0 | 0.8730 / 0.8791 (ep29, L301) | 0.8730 / 0.8791 (L301) |
| 1 | 0.8678 / 0.8744 (ep14, L163) | 0.8724 / 0.8791 (L299) |
| 2 | 0.8724 / 0.8791 (ep10, L129) | 0.8724 / 0.8791 (L301) |
| **mean** | **0.8711 / 0.8775** | **0.8726 / 0.8791** |

Provenance: `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog` (lines cited).
val-sel mean acc **0.8775** = the project-best HateMM number. HateMM already PASSES both protocols; the readout's job here
is to **HOLD**, not add.

### 2.3 Context floors — frozen-CLIP (NOT paired anchors; re-derived for orientation)

- **ZH CLIP (13115):** val-sel **0.8076 / 0.7676**, final **0.8143 / 0.7720** (`enc3s_MHC_zh_openai_clip-vit-large-patch14-336_HF_seed{0,1,2}_13115.trainlog`).
- **HateMM CLIP (12850):** val-sel **0.8202 / 0.8085**, final **0.8124 / 0.7936** (`enc3s_HateMM_openai_clip-...-336_HF_seed{0,1,2}_12850.trainlog`).

Both bit-match CAND2 §2.1/§2.2. The verdict does NOT re-pair vs CLIP — the R0 anchors §2.1/§2.2 are the deployed reads.

---

## 3. Decision rule + kill-bars (screen bar + verdict bar, both pre-declared)

### 3.1 Screen advance bar (permissive, to catch signal — recon §7)

The winner cell's dev **Δacc ≥ +0.020** over R0 on the **fused key**, on **≥1 dataset** (ZH or HateMM), in **either** arm
(LOO or dev-query), with the machinery-validity guards sane (R0 bit-exact PASS; perm-null/bootstrap reported). The +0.020
screen bar is deliberately **below** the verdict bar (a screen should not miss a candidate; the verdict is the strict
judgment). **Single winner only** — the highest fused-key dev Δacc clearing +0.020 advances; the whole {R1,R2,R3} family is
**one pre-registered bite, one winner, one test measurement**. `readout_screen.py` computes `advance = (winner Δacc ≥ 0.02)
AND R0-bit-exact PASS`.

### 3.2 KS-readout-dead — the KILL bar (recon §7)

**KILL iff ALL grid cells are ≤ +0.020 over R0** on the $0 dev screen (no cell clears +0.020 on either dataset in either
arm) → **cell DEAD at ~2 GPU-h total** (extraction only; zero verdict GPU, zero test-touch). Mirrors ISR NO-GO / FRAME16
KS-16f-dead: a flat $0 screen closes the axis without any test-touch. The negative is a candidate F-finding (orchestrator's
call — this prereg modifies no `state/`).

### 3.3 Verdict bar (strict, house standard — B3 / frame16 / encoder-3seed) — verbatim from `exp-encoder-3seed.md:73-85`

> For each dataset × protocol: (1) per-seed paired difference δ = (treatment − control) for acc and macro-F1 at seeds
> 0/1/2; (2) 3-seed mean ± std + sign consistency; (3) n=3 too small for a bootstrap — report the paired-t as an
> **effect-size descriptor only**, no significance claim; (4) **pass = mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1
> ≥ +0.030 AND sign 3/3 positive**; (5) headline claim requires pass on ≥ 2 datasets under a stated protocol; both
> protocols judged separately; verdict written exactly "final-epoch: pass/fail; val-selected: pass/fail".

Treatment = the promoted winner cell; control = the banked deployed R0 arm (ZH 13150 §2.1 / HateMM 13241 §2.2). Both
protocols judged **independently** (no protocol-shopping, no metric-shopping). **Single test-touch, verdict stage only.**
ZH val-selected is the target (F45 78-dev selection tax); HateMM is the hold check (F53 — must not regress).

### 3.4 KS-regression — BELOW-R0 KILL (hold-leg protection)

If the promoted winner − R0 **mean Δacc ≤ −0.014** on a held leg (below the largest banked head-seed spread, ±0.014 per
CAND2 §2.3), the readout **degraded** the deployed representation → bank "readout hurts", KILL. (HateMM is the hold leg
whose inherited pass must not regress.)

### 3.5 Ladder + gate order

`KS-readout-dead (all cells ≤ +0.02 dev)` ⊂ `screen-promote (winner ≥ +0.02 dev, ≥1 ds, either arm)` → `FORMAL verdict
(winner 3-seed vs banked R0: ≥ +0.030/+0.030, 3/3, per protocol)`; `KS-regression (≤ −0.014 vs R0)` is the hold-leg floor.
Gate order: G-repro (extractor sha + R0 bit-exact + Namespace-diff) → $0 screen → [KS-readout-dead | promote] → [if
promote] single test-touch → verdict bar (both protocols) → KS-regression check. The verdict is rendered by an
**independent 0-context reviewer against this prereg VERBATIM**; the executor transcribes raw both-protocol per-seed numbers
(line-numbered) and applies NO gates/interpretation.

### 3.6 Optional tie-break head-retrain (PINNED, NOT auto-run — recon §5)

**Trigger (ambiguity only):** the fused-key winner is within **±1 dev item** of R0 acc, **OR** the LOO and dev-query arms
name **different** winners. Then, to break the tie before advancing, run a ~25 s/cell head-retrain **selected on DEV only**.
**Test-safe construction (DEV-1 deviation, §11):** `run_rac.py` evaluates `Test_Retrieval` every epoch (L745, no skip
flag), so the tie-break MUST NOT load the real test cache. For each candidate suffix `S` (throwaway `-tb` tag with
`test_seen := dev_seen` copy, so the REAL test cache is never opened):

```
DS=<MHC_zh|HateMM>; BASE=<...>; S=<ro_L24|ro_ow_L28|ro_ow_L24>
cp data/CLIP_Embedding/$DS/train_${BASE}-${S}.pt     data/CLIP_Embedding/$DS/train_${BASE}-${S}-tb.pt
cp data/CLIP_Embedding/$DS/dev_seen_${BASE}-${S}.pt  data/CLIP_Embedding/$DS/dev_seen_${BASE}-${S}-tb.pt
cp data/CLIP_Embedding/$DS/dev_seen_${BASE}-${S}.pt  data/CLIP_Embedding/$DS/test_seen_${BASE}-${S}-tb.pt   # test := dev copy
python ./src/run_rac.py --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 --dataset $DS \
    --model ${BASE}-${S}-tb --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align \
    --hard_negatives_loss True --no_hard_negatives 1 --final_eval False --seed 0 \
    --group_name RAC_video_readout_tiebreak --metric cos --loss triplet --batch_norm False --hybrid_loss True \
    --warmup 5 --majority_voting arithmetic --no_pseudo_gold_positives 1 --lambda_seg 0 --seg_mode full \
    --num_subclips 4 --em_rounds 2 --consensus_topk 10 --consensus_margin 0.2 --exp_comment _readout_tb \
    --Faiss_GPU False --force False
# select the winner by the Val_Retrieval (dev) acc ONLY (warmup>=5, roc tie-break); the "Test_Retrieval" lines are
# dev-on-dev (throwaway) and are DISCARDED UNREAD; then rm the *-tb.pt throwaways.
```

The single budgeted real test-touch remains the verdict head. This step is **not auto-run**; the executor invokes it only
on ambiguity.

---

## 4. G-repro + smoke plan + collision safety

### 4.1 G-repro discipline

- **(a) R0 bit-exact clobber-guard = the determinism anchor.** The R0 recompute (`-ro_L28`, baseline prompt, current span,
  layer 28) MUST reproduce the banked deployed cache **bit-exact** (`img/text max|Δ| == 0` on train+dev; asserted by the
  screen guard §1.2c and by the smoke §4.4). This proves the new extractor's default path is byte-identical to the banked
  `_encode` — the whole basis for R0 being a valid paired anchor. A mismatch = **VOID / investigate** (do NOT silently
  proceed): a non-zero Δ means the merged forward is not reproducing (GPU-arch drift, library change, or a code bug), which
  would also invalidate R1/R2/R3.
- **(b) Extractor same-code + pooling bit-verified.** `generate_VideoMLLM_embedding_readout_HF.py` clones the banked LoRA
  extractor (`_encode` byte-identical between the frozen and LoRA extractors — verified this prereg). The new
  `_pool_span(prefix|response, layer 28)` was unit-tested against a verbatim reimplementation of the deployed `_encode`
  pooling on synthetic tensors: **max|Δ| = 0.0** for prefix, response, and last-token; L2-norm = 1.0 (§9). The banked
  extractor file is **NOT edited** (sha re-verified at submit, §5.2).
- **(c) Head same-code as 13150/13241.** At promote-time the verdict head's `run_one` block MUST be byte-identical to
  `enc3seed_lora_curric.sbatch`; the Namespace diff vs the banked R0 control MUST be `--model` + derived-inert fields only
  (proven inert by the 12850 bit-exact seed0 reproductions, `exp-encoder-3seed.md:126-146`).
- **(d) Extraction shape sanity ($0 CPU, post-extraction):** each `{split}_<BASE>-ro_*.pt` loads with
  `img_feats`/`text_feats` shape `(N, 3584)`, N = split size (ZH 579/78/149; HateMM 744/107/215), labels present, finite.

### 4.2 Same-code + syntax verification (run this prereg — PASS)

- `python -m py_compile` on the extractor + screen: **COMPILE_OK**. `bash -n gen_embed_readout.sbatch`: **SYNTAX_OK**.
- `_encode` byte-identical between `generate_VideoMLLM_embedding_HF.py` and `..._lora_HF.py` (`diff` empty).
- `_pool_span` vs deployed `_encode` pooling: **max|Δ| = 0.0** (prefix/response/last-token, synthetic tensors, CPU).
- Screen synthetic end-to-end self-test (fake caches, CPU): R0 bit-exact gate PASS; identical caches ⇒ all cell Δacc =
  **0.0** (degenerate recovery); a planted-stronger cell ⇒ promote fires with correct fix/broken/net; perm-null + bootstrap
  compute without error.

### 4.3 Collision safety (verified this prereg — ABSENT; re-check at submit)

- `data/CLIP_Embedding/{MHC_zh,HateMM}/*-ro_*.pt` — do NOT exist (0 found) ⇒ fresh extraction; banked deployed caches
  (`*-LoRA_HF.pt` ZH, `*-LoRA-curric_HF.pt` HateMM — sha16 recorded §5.2) **untouched** (distinct `-ro_*` suffix; the
  extractor NEVER writes `{split}_<BASE>.pt`).
- `logging/Retrieval/*/RAC_video_readout*` — do NOT exist (0) ⇒ fresh verdict group; `RAC_video_readout_tiebreak` fresh.
- `slurm/logs/*-ro_*` trainlogs — do NOT exist (0; the lone `*ro_*` match was `p9b_c3rep`ro`_12491.out`, unrelated).

### 4.4 Smoke plan (executor runs BEFORE the real extraction; leave no artifact that trips §4.3)

1. **Readout extraction smoke (GPU, ~1–2 min):** run the readout extractor with `--dataset HateMM --lora_dir
   logging/lora/HateMM_curric --out_model_base_tag Qwen2.5-VL-7B-Instruct-LoRA-curric_HF --splits test --limit 3
   --EXP_FOLDER logging/_smoke_ro --device cuda`. Confirm: it writes the 4 caches
   `logging/_smoke_ro/HateMM/test_seen_...-ro_{L28,L24,ow_L28,ow_L24}.pt`, each `img/text` shape `(3, 3584)`, 0 NaN, no OOM,
   the masked-scatter assert holds; the one-word prompt tokenizes as expected; and the `-ro_L28` 3-row slice reproduces the
   **banked** deployed HateMM cache rows for those 3 videos **bit-exact** (`max|Δ| == 0`). Then `rm -rf logging/_smoke_ro`.
2. **$0 screen dry sanity (CPU, optional):** the synthetic self-test in §4.2 already validates the screen wiring;
   post-extraction, `readout_screen.py`'s R0 bit-exact guard re-asserts the clobber-guard on the full caches before any Δ is
   trusted.

---

## 5. Artifacts authored this prereg + hash-freeze block

### 5.1 New artifacts (candidates for the reviewer's hash-freeze)

| # | path | change | sha256 (current) |
|---|---|---|---|
| P | `refine-logs/READOUT_PREREG.md` | **NEW** — this file | *(reviewer fills at freeze)* |
| A | `src/utils/generate_VideoMLLM_embedding_readout_HF.py` | **NEW** — readout-grid extractor (clone of the LoRA extractor; hidden-state layer read + last-token span + one-word prompts; NO edit to the banked extractor) | `ef05f3d45a3e8c31f8dc198ba41e18c2e525cd29e9ba0ed539dfd9b4c6d869c3` |
| B | `scripts/slurm/gen_embed_readout.sbatch` | **NEW** — combined ZH+HateMM readout extraction (hardcoded tags, 8 CPU / 64 G / 1×A100, no `--time`) | `948db8514c9e4b02d6d20ceed3e6a63104893c8a6e623def75e4c22bc9419e29` |
| C | `scripts/analysis/readout_screen.py` | **NEW** — $0 CPU dev screen (fused-key vote, 2 arms, perm-null/bootstrap, R0 bit-exact guard; test hard-blocked) | `f56badb64b9dc8a4d18fbbcbbff99994234df3812dccd7334f8827e100d35547` |

### 5.2 Reused-unchanged machinery (verify sha at submit time; do NOT edit)

| path | role | sha256 (or sha16) |
|---|---|---|
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | fork source for A (`_encode` bit-verified; NOT edited) | `b6b61a3fa4214f28d2098ca305ca9a981934445afc81ccc5ac3939f8c1fb0ec6` |
| `scripts/analysis/cross_channel_router_gate.py` | vote-machinery source (`_weighted_signed_vote` L73-79) | `d4adf545125a5a08d78ec9198947dc44f6c6abeec158ed308e138fc9d3d96a5d` |
| `scripts/slurm/enc3seed_lora_curric.sbatch` | verdict-head clone source (§1.3) | `00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02` |
| `logging/lora/MHC_zh/adapter_model.safetensors` | ZH deployed adapter | *(present, verified)* |
| `logging/lora/HateMM_curric/adapter_model.safetensors` | HateMM deployed adapter | *(present, verified)* |
| `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | banked ZH R0 (paired floor; NOT clobbered) | train `b2e8e78d19c71d2c` / dev `4c07af75098391c9` / test `4e107bf65f58745a` |
| `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` | banked HateMM R0 (paired floor; NOT clobbered) | train `5e80f39327a74314` / dev `46ee4fd9fcaec80b` / test `b50ae4ecb077a833` |

### 5.3 Hash-freeze (to be filled by the independent reviewer at freeze time)

```
FROZEN <sha256 of this file READOUT_PREREG.md, after review>
A ef05f3d45a3e8c31f8dc198ba41e18c2e525cd29e9ba0ed539dfd9b4c6d869c3  generate_VideoMLLM_embedding_readout_HF.py
B 948db8514c9e4b02d6d20ceed3e6a63104893c8a6e623def75e4c22bc9419e29  gen_embed_readout.sbatch
C f56badb64b9dc8a4d18fbbcbbff99994234df3812dccd7334f8827e100d35547  readout_screen.py
```
Executor re-runs `sha256sum` on A/B/C (and this file) + confirms the fork-source extractor sha `b6b61a3f…` unchanged at
submit; any mismatch = authorization VOID. **Reviewer to independently re-verify: (i) the pinned one-word prompts
(§1.1); (ii) the L24/L28 layer indexing off-by-one vs `num_hidden_layers=28`; (iii) the R0 bit-exact gate wiring.**

---

## 6. Single-submit / execution plan + resource plan + ordering

**Order (extraction → $0 screen → [if promote] verdict; codex-gate the extractor first):**

0. **Codex-review the extractor** (artifact A reads model internals) BEFORE any GPU (CLAUDE.md gate).
1. **Extraction smoke** (§4.4.1, GPU ~1–2 min) → then `sbatch scripts/slurm/gen_embed_readout.sbatch` (ONE combined job,
   ZH then HateMM sequential, ~2 GPU-h). On COMPLETE apply §4.1d shape sanity.
2. **$0 CPU screen:** `CUDA_VISIBLE_DEVICES="" python scripts/analysis/readout_screen.py` → `READOUT_SCREEN_OUT.json`. This
   is the KILL-or-promote decision. If flat → **KS-readout-dead**, STOP (bank the negative; no test-touch).
3. **[If promote]** escalate to the normal ceremony: author the verdict head sbatch (§1.3, clone of
   `enc3seed_lora_curric.sbatch` with the winner tag) → 0-context review → freeze → 3-seed test verdict on the winner
   cache, paired vs banked R0 (§2). Single test-touch per promoted dataset.

**Resource plan (STANDING INFRA RULE compliant):** the extraction sbatch requests **8 CPU / 64 G / 1×A100**, runs both
datasets **sequentially in ONE job** (peak 8 CPU / 64 G / 1 GPU — well within the 16 CPU / 128 G / 2 GPU cap, and NEVER two
16-CPU jobs in flight). `conda activate HateVideo`; `sbatch` with **NO `--time`**; initial `PENDING (JobHeldUser)` =
**WAIT for auto-release, never force** (CLAUDE.md). The verdict head (if promoted) is a separate ~few-min 8-CPU job.

**Ordering vs the bidirectional-encoder chain (pre-declared):** the bidir chain (parallel prereg) submits **AFTER** this
readout chain clears its extraction+screen (respecting the submit-time aggregate cap; never two 16-CPU jobs — both chains
are 8-CPU, but sequence them to avoid concurrent GPU contention per the orchestrator).

**Test-touch:** the Stage-3 head reads are the ONLY budgeted readout-cell test evaluations; zero test-touch before the
independent verdict. **No job is submitted by this prereg author.**

---

## 7. Outcome tables (filled ONLY from raw output at the relevant stage)

### 7.1 $0 screen (fill from `READOUT_SCREEN_OUT.json`)

| dataset | cell | LOO Δacc (fix/brk/net) | dev-query Δacc (fix/brk/net) | best Δacc | perm-null p95 · obs>p95 | boot Δ p5 |
|---|---|---|---|---|---|---|
| ZH | R1 `ro_L24` | ___ | ___ | ___ | ___ | ___ |
| ZH | R2 `ro_ow_L28` | ___ | ___ | ___ | ___ | ___ |
| ZH | R3 `ro_ow_L24` | ___ | ___ | ___ | ___ | ___ |
| HateMM | R1 `ro_L24` | ___ | ___ | ___ | ___ | ___ |
| HateMM | R2 `ro_ow_L28` | ___ | ___ | ___ | ___ | ___ |
| HateMM | R3 `ro_ow_L24` | ___ | ___ | ___ | ___ | ___ |

`R0 bit-exact: <PASS/FAIL both splits both datasets>. Winner: <ds/cell/arm, Δacc>. Verdict: <KS-readout-dead | PROMOTE>.`

### 7.2 Verdict (ONLY if promoted; fill from `enc3s_<DS>_<BASE>-<winner>_seed{0,1,2}_<JID>.trainlog`)

| seed | protocol | winner acc/F1 | R0 floor acc/F1 (§2.1/§2.2) | Δ(winner−R0) acc/F1 |
|---|---|---|---|---|
| 0 | val-sel | ___ | ___ | ___ |
| 1 | val-sel | ___ | ___ | ___ |
| 2 | val-sel | ___ | ___ | ___ |
| **mean** | **val-sel** | ___ | ___ | **___** |
| 0 | final-ep | ___ | ___ | ___ |
| 1 | final-ep | ___ | ___ | ___ |
| 2 | final-ep | ___ | ___ | ___ |
| **mean** | **final-ep** | ___ | ___ | **___** |

`<DS> (readout <winner> vs deployed R0): final-epoch: <pass/fail>; val-selected: <pass/fail> [verdict bar §3.3].`
`KS-regression (§3.4): <held | regressed>.` (+ MARGINAL note if a within-noise pass per B3 §2.2 precedent.)

---

## 8. What a PASS / FAIL means for the goal (D7 boundary — readout is engineering; the D7 weight is the USER's)

- **KS-readout-dead (recon prior — the honest expected result):** no readout cell clears +0.020 on dev ⇒ the axis is
  **CLOSED** at ~2 GPU-h (extraction only). The single un-enumerated in-paradigm axis (litsurvey verdict) is measured and
  closed at $0-screen cost; no test-touch. Strengthens the terminus.
- **PROMOTE + FORMAL PASS (≥ +0.030/+0.030, 3/3, a protocol):** a principled-readout **performance/robustness** result. On
  ZH, the target pattern is the val-selected leg crossing the bar (hardening F45's marginal pass). **Still D7-thin standing
  alone (F0.3)** — its novelty weight depends on the MLLM-embedding-paradigm framing, which is the **USER's** call. Not a
  new-dataset route (F0.4).
- **KS-regression:** the readout degraded the deployed representation on a held leg → bank "readout hurts", KILL.

**Framing sentence (verbatim):** *this measurement tests the one axis extraction never varied — the readout (layer, token,
prompt) of the deployed merged encoder — via a ~2 GPU-h re-extraction, a $0 dev screen that kills cheaply if flat, and (only
if it promotes) a single-test-touch 3-seed verdict; a pass is a performance/robustness row whose novelty weight is the
user's D7 call, and the one-word readout is a single readout, not ensembling.*

---

## 9. Provenance index

- Recon (GO-IF; grid, amortization, cache tags, $0 screen, kill-bars): `refine-logs/READOUT_FORENSIC_RECON.md` (`61a9f4a`).
- Extractor / pooling / layer read: `src/utils/generate_VideoMLLM_embedding_lora_HF.py:277-346` (`_encode`, layer read
  L302, spans L313-341, prompts L58-66); Qwen config `num_hidden_layers=28`, `hidden_size=3584` (HF cache, verified).
- $0-screen machinery: `scripts/analysis/cross_channel_router_gate.py:73-79` (`_weighted_signed_vote`); fused-key
  construction `LP_GATE_RECORD.md §1`; two-arm protocol `ISR_PREGATE_RECORD.md §0.2`.
- Floors (re-derived §2, line-cited): ZH R0 `enc3s_MHC_zh_..-LoRA_HF_seed{0,1,2}_13150.trainlog`; HateMM R0
  `enc3s_HateMM_..-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`; CLIP context 13115 / 12850. Decision rule
  `exp-encoder-3seed.md:73-85`. Bit-matches `CAND2_CURRICULUM_PREREG.md §2`.
- Verdict/kill-bar templates: `FRAME16_PREREG.md`, `CAND2_CURRICULUM_PREREG.md`.
- Literature: VidVec 2602.08099 (intermediate layer 24 on a ~28-layer 7B), PromptEOL 2307.16645, E5-V 2407.12580, MetaEOL
  2402.18458 (ensemble — excluded), Echo 2402.15449 / 2502.20726 (R4, deferred out of grid). F68 ledger candidate P1.

**Required statements:** ZERO GPU/SLURM/Modal spent by this prereg author (only pure-CPU login-node floor re-parsing,
sha/syntax/collision verification, and synthetic screen self-tests, seconds). No held-out test metric produced. All floor
numbers re-parsed from banked completed-run trainlogs (numeric-provenance discipline). No `state/` mutated. No
`research-wiki/` mutated. NO job submitted. Not pushed. Cloud/external numbers are triage context only, never mixed with
local G-repro numbers.

---

## 10. (reserved)

---

## 11. DEVIATIONS FROM THE RECON — flagged loudly

1. **DEV-1 (tie-break made TEST-SAFE via a `test_seen := dev_seen` throwaway copy). MATERIAL / discipline-favorable.**
   The recon §5 pins the tie-break as "enc3s head on the cell's train cache, **eval dev**" — but `run_rac.py` evaluates
   `Test_Retrieval` EVERY epoch (L745, no skip flag), so a naive head-retrain WOULD read the real test cache before promote,
   violating single-test-touch. Resolution (§3.6): the tie-break runs on a throwaway `-tb` cache whose `test_seen` is a COPY
   of `dev_seen`, so the REAL test cache is never opened; the winner is chosen by `Val_Retrieval` (dev) acc only, and the
   dev-on-dev "Test" lines are discarded unread. The single budgeted real test-touch remains the verdict. This closes a gap
   the recon left under-specified.

2. **DEV-2 (ONE combined extraction job for both datasets, not one-job-per-dataset). MATERIAL / operationally simpler.**
   Recon §9 step 3 says "one job/dataset"; the task permits "one combined job — recon's call". I authored a single
   `gen_embed_readout.sbatch` with a hardcoded 2-row CONFIGS array (ZH then HateMM, sequential, 8 CPU / 64 G / 1 GPU
   throughout). Rationale: one hash-frozen artifact, one submit, hardcoded per-dataset tags (frame16 DEV-2 anti-clobber
   discipline — no submit-time arg typo can point at the wrong adapter), never two concurrent CPU jobs. Partial progress is
   recoverable (ZH caches persist if HateMM fails). The per-dataset screen is unaffected.

3. **DEV-3 (grid hardcoded in the extractor `CELLS` table, NOT exposed as sweepable `--readout_layer`/`--readout_span`
   args). Favorable / forking-path-safe.** The recon §9 sketched `--readout_layer`/`--readout_span`/`--readout_prompt`
   argparse flags. I instead **pin the 4 cells in a frozen `CELLS` constant** (layers {28,24}, the two prompt-passes, the
   spans), so no submit-time flag can silently widen the grid into a layer/prompt sweep (the multiplicity trap the recon §5
   warns against). The one-word prompts are frozen module constants. This makes the frozen grid a code-level invariant, not
   a runtime choice.

4. **DEV-4 (screen advance bar = the recon §7 +0.020, perm-null/bootstrap REPORTED as validity guards not folded into the
   hard advance boolean). Documented / faithful.** `readout_screen.py` sets `advance = (winner fused Δacc ≥ +0.020) AND
   R0-bit-exact PASS`; perm-null (≥200) and bootstrap (1000) are computed and written to `READOUT_SCREEN_OUT.json` as the
   validity guards the recon §5 calls "validity guards, not extra bars" (the reviewer/verdict weighs them; a winner whose Δ
   sits inside the perm-null band is flagged for the tie-break/skepticism, not auto-killed). Matches ISR/LP treatment.

5. **DEV-5 (R4 Echo excluded from the grid; verdict pairs vs banked deployed R0, no R0 head re-run). Documented.** Per
   recon §3, R4/Echo is held OUT of the core grid (weakest prior, +cost). The verdict pairs the winner against the banked
   deployed R0 head runs (13150 ZH / 13241 HateMM) rather than a fresh R0-`ro_L28` head — valid because `ro_L28` is
   bit-exact to the deployed cache (F0.2), so a fresh R0 head would reproduce the banked runs seed-for-seed. EN excluded per
   F0.4 (recon §4).
