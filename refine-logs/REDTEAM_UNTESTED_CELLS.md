# RED-TEAM AUDIT — UNTESTED CELLS INSIDE THE DEPLOYED PIPELINE

**Author:** red-team recon subagent (adversarial). **Date:** 2026-07-20 NZST.
**Discipline:** CPU-only; ZERO SLURM / GPU / Modal / training / test-touch. No `state/` mutation.
One cheap CPU probe was run on banked train/dev caches only (no new feature extraction) — see §0.
**Mission:** adversarially enumerate cells INSIDE the current pipeline that carry **no binding
verdict and no covering ban**, to test the `TERMINUS_round3` claim that "every injection point
inside the frozen constraint box is closed." **Finding: the strong form of that claim is false.**
The box is closed for the *operator families that were probed*, but at least **two representation /
new-channel cells were never measured at all**, only argued-down in prose — and the pipeline's own
config/code makes them legally reachable on the local 7B.

**Framing (honest, up front).** This document REFUTES "we measured everything," not "the goal is
reachable." Most gaps below are LOW-prior and several are D7-novelty-dead even if they gained. The
two load-bearing refutations are **C1 (vision-tower/projector unfreeze)** and **C3 (learned-audio
third stream)** — both are *representation-level* or *new-input-channel* levers, the only two gain
classes the campaign's own diagnosis frame (`state/directions_tried.json` `diagnosis_frame`; F1/F2)
ever blesses, and **neither was ever run**. The terminus §2 relaxation table lists a 32B/72B
download (a) and a Qwen-Omni download (b) but **never lists the two in-box, zero-download cells this
audit surfaces**.

---

## 0. CHEAP CPU PROBE (banked caches only, prior-mover, non-binding)

Script: `scratchpad/redteam_stream_topk_probe.py` (not committed; scratch). Per-stream top-20
rank-weighted signed-cosine LOO vote — the SAME operator as `build_curriculum_sft_data.py:95-103`
and the deployed mining — over `data/CLIP_Embedding/**/{train,dev_seen}_*.pt`, three encoders.

**Machinery validity:** reproduces F58's banked HateMM text-stream train-LOO AUC ladder to ~0.01
(mine: CLIP 0.847 / frozen-Qwen 0.884 / LoRA 0.920; F58 `51eb95b`: 0.847 / 0.888 / 0.920). Sound
for prior-strengthening (NOT a verdict).

**Per-stream train-LOO AUC (image-only / text-only / fused):**

| dataset | encoder | image AUC | text AUC | fused AUC |
|---|---|---|---|---|
| HateMM | CLIP | 0.836 | 0.847 | 0.871 |
| HateMM | frozen-Qwen | 0.820 | 0.884 | 0.886 |
| HateMM | LoRA | 0.823 | **0.920** | 0.906 |
| **MHC-EN** | CLIP | **0.745** | 0.780 | 0.811 |
| **MHC-EN** | frozen-Qwen | **0.653** | 0.854 | 0.832 |
| **MHC-EN** | LoRA | **0.659** | 0.925 | 0.910 |
| MHC-ZH | CLIP | 0.724 | 0.804 | 0.775 |
| MHC-ZH | frozen-Qwen | 0.734 | 0.856 | 0.853 |
| MHC-ZH | LoRA | 0.733 | 0.928 | 0.918 |

**Two measured facts that drive the ranking below:**

1. **The MHC-EN image stream is collapsed AND upstream-of-the-LLM.** CLIP 0.745 → frozen-Qwen 0.653
   (train-LOO AUC; dev 0.743 → 0.695) confirms F44's collapse. **LLM-only LoRA does NOT repair it:**
   0.653 → 0.659 train, 0.695 → 0.695 dev — flat, matching F58's "image stream flat under LoRA"
   *extended to EN*. Because the deployed LoRA adapts only the LLM decoder
   (`lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, vision tower + projector
   frozen), the collapse lives in a part of the network **no lever tried so far can reach**. The
   only lever that structurally can is C1.

2. **topk is near-flat and direction-inconsistent** — fused train-LOO acc: HateMM LoRA best at k=10
   (0.8414 vs 0.8293 @ k=20), MHC-EN LoRA best at k=80 (0.8452), ZH LoRA best at k=40 (0.8497). The
   swings are ~1pt and point different ways per dataset ⇒ classic forking-path; the topk knob is not
   a live lever (prices down part of C4/C8).

---

## 1. THE CELLS (each: (a) current-state evidence · (b) covering ban? · (c) in-box · (d) GPU-h ·
## (e) cheapest kill gate · (f) honest prior of ≥+1pt on any dataset · (g) rank)

### C1 — Vision-tower / projector UNFREEZE during LoRA-SFT  ·  RANK 1

**(a) Current state.** Every LoRA-SFT config freezes the visual path and adapts only the LLM decoder:
`RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_sft.yaml:13-15`
— `lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, `freeze_vision_tower: true`,
`freeze_multi_modal_projector: true` (identical in the ZH/EN/curric configs). So **every "adaptation"
in the campaign adapted the LLM only**; the pooled `img_feats` are LLM-contextualised vision tokens,
but the vision *encoder + projector* that produce those tokens were never touched.
**On-disk support is present:** `RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/hparams/finetuning_args.py:539,543`
expose `freeze_vision_tower` / `freeze_multi_modal_projector` as togglable, and L99/103/107 add
`use_rslora` / `use_dora` / `pissa_init`. The cell is a one-line flip (`freeze_vision_tower: false`,
optionally `lora_target: all`).

**(b) Covering ban? NO.** The closest banked evidence is *informational, not a kill*:
- F58 (`51eb95b`, `HATEMM_LORA_STREAM_DECOMP.md`) measured image ΔAUC +0.0045/+0.0062 **under the
  LLM-only recipe** and its "premise-(a) vision-SFT prior stays low" note is explicitly labelled
  *"a diagnostic note for the orchestrator, not a proposed cell."* It measured what the frozen-vision
  recipe does — it did **not** run a vision-adapted encoder.
- F55 (`6e6061b`) priced *CLIP's healthy* EN image × LoRA-EN-text at oracle +0.025 < +0.03. But that
  is a **cross-encoder fixed composition** (CLIP image glued to Qwen text), not a *same-encoder
  Qwen* image healed by vision adaptation and co-trained with the adapted Qwen text through the
  align-Hadamard head. F55's ban scope reads "Cross-encoder composition with ADAPTED text on EN" —
  it does not name same-encoder vision adaptation.
- F44 attributes EN failure to "label-limited errors" + "image collapse," but its own verdict says
  "NEW AXIS: honest NO — finding lives inside Axis B (encoder identity)"; it never tested a
  vision-adapted encoder either.

  There is **no ban whose scope covers same-encoder vision-tower/projector adaptation.** It is a
  genuine untested cell, and §0 fact-1 shows it targets exactly the un-repaired EN image collapse.

**(c) In-box.** LEGAL — Qwen2.5-VL-7B is local, single-dataset own-train, no gold, no OCR, no
external API; LoRA weights stay on disk. No constraint lifted.

**(d) GPU-h & overfit.** Two variants:
- *LoRA-on-vision* (`freeze_vision_tower: false` + rank-16 on visual blocks): adds only a few M
  trainable params on top of the ~20M LLM-LoRA; overfit risk moderate on 743/549/579 videos but
  bounded by rank-16. Est. **~4-6 A100-h/dataset** (generic LoRA-SFT was ~3.1-3.5 h; vision tokens
  lengthen the backward graph) + ~1 h re-extraction + ~20-25 s head.
- *Full vision-tower unfreeze* (~675M trainable): **overfit-doomed** on <750 videos — do NOT propose
  this variant; the LoRA-on-vision variant is the only defensible one.

**(e) Cheapest kill gate.** No fully-$0 pre-GPU gate exists (the adapted features don't exist yet).
The strongest cheap *prior-mover already in hand* is §0 fact-1 (LLM-only LoRA leaves EN image flat ⇒
the collapse is upstream ⇒ vision adaptation is the mechanistically-unique lever). The pre-declared
GPU gate would be: extract vision-adapted EN features, then the F55 oracle-threshold screen
(`d_oracle >= +0.03`) as the kill bar, plus the KS-2 image-moved check
(`img ΔAUC(vis-LoRA − LLM-LoRA) >= +0.010`).

**(f) Prior of ≥+1pt.** **LOW-MODEST (~12-15%)**, and it is the *highest* prior in the untested set
because (i) representation-level is the only class that EVER cleared +3 (encoder-swap, F44);
(ii) §0 fact-1 makes it the only lever that can touch EN's collapsed image stream, i.e. the only
untested path to the missing 2nd/3rd dataset; against it, F55's EN-oracle ceiling (+0.025) is real
but was measured on a *different* (cross-encoder) image and so does not fully subsume the
same-encoder co-trained case. On HateMM/ZH it would only sharpen already-passing legs (adds no
dataset) — the EN payoff is the whole ballgame and it is capped-but-not-closed.

**(g) RANK 1** — the single cleanest refutation of "every injection point is closed": a
representation-class lever, one config flip away, aimed at the exact stream the terminus calls
"label-limited," never run.

---

### C3 — Learned-audio representation (Whisper-encoder / wav2vec2 hidden states) as a third stream  ·  RANK 2

*(listed before C2 because it is a "new input channel" — one of only two blessed gain sources — and
the terminus explicitly PARKED-not-killed it.)*

**(a) Current state.** The pipeline has **no audio channel at all**; `img_feats`/`text_feats` are
both Qwen forwards over 8 frames + transcript. The ONLY audio work done is:
- **APX (F41, `9c54faf`)** killed **classical eGeMAPS-88-d whole-video prosody** at a $0
  conditional-info gate (raw-88d Δacc **+0.0005**, calibration accZA 1.0 ⇒ genuine null).
- **W2-D (F31, `b143834`)** — its "prior near-zero" was an **INFERENCE, never a measurement**
  ("plausibly under +0.040"); no learned audio representation was ever extracted or screened.

  Environment check (CPU import only): `transformers 4.49.0` exposes `Wav2Vec2Model`, `WhisperModel`,
  `HubertModel`, `ASTModel`; **`openai/whisper-base` + `openai/whisper-large-v3` weights are already
  on disk** (`~/.cache/huggingface/hub/`, from the ASR transcript step). `torchaudio` is **MISSING**
  and only `.mp4` exists (0 `.wav` cached, 1066 HateMM mp4s) ⇒ audio decode needs ffmpeg/librosa.

**(b) Covering ban? NO — scoped to classical prosody only.** APX ban scope, verbatim
(`state/directions_tried.json`): *"classical whole-video prosody as auxiliary channel on HateMM;
… audio axis PARKED — any future audio proposal (incl. W2-D Qwen-Omni download) must first explain
how it beats a zero-information classical baseline through the same conditional screen."* That is a
**bar, not a kill** — the ban explicitly contemplates future audio proposals. A **Whisper-*encoder*
hidden-state** vector is a *learned* acoustic/paralinguistic representation, categorically distinct
from (i) eGeMAPS-88d hand-crafted prosody (what APX killed) and (ii) the Whisper-*decoder* transcript
(what `text_feats` already banks): it carries non-lexical acoustic events — music, tone, laughter,
non-speech aggression, slurs-in-song — that eGeMAPS compresses lossily and the transcript misses
entirely. F1/F2's `diagnosis_frame` blesses exactly two gain sources: "representation-level OR
new-input-channel bandwidth." **Audio is the one physically-unused input channel**, and only its
lossy hand-crafted proxy was ever screened.

**(c) In-box.** LEGAL — Whisper is already local (zero download); a wav2vec2/HuBERT variant would be
a *weights* download (in-box per the constraint box: local weights allowed, raw video never leaves).
Frictions: `torchaudio` missing (use ffmpeg→librosa), no `.wav` cache.

**(d) GPU-h.** Whisper-base encoder is light: ~1-2 GPU-h/dataset to encode ~1066/790/806 videos'
audio (CPU-feasible but slow), + head re-train ~20-25 s (features cache). Est. **~2-4 A100-h/dataset**
incl. extraction — the cheapest of the three representation/channel cells.

**(e) Cheapest kill gate.** After a cheap Whisper-base encode, the **identical $0 conditional-info
screen that killed APX** (`c3_fusion_probe.py` machinery, label-oracle calibration arm mandatory):
kill iff pooled-audio Δacc over `Z_best` < +0.040 with calibration accZA≈1.0. So the gate is
"cheap-CPU-after-a-cheap-extraction," not fully $0 — but far below a full run.

**(f) Prior of ≥+1pt.** **LOW-MODEST (~10-14%)**. Against: the transcript already banks lexical hate
(the F41/F31 hazard is real); D1 conditional-redundancy applies. For: it is a genuinely new,
non-lexical channel that no learned model has screened, and it is a blessed gain source. D7 caveat:
"add audio" is HateMM's 2023 founding contribution and all three baselines already fuse audio ⇒
novelty-thin (catch-up), so even a gain is a *performance/ablation* row, not a novelty win.

**(g) RANK 2** — untested + new-channel + blessed source + cheapest extraction; the terminus's
"audio PARKED" is honest that this was never measured.

---

### C2 — Frame budget (8 → 16 / 32; denser or adaptive sampling)  ·  RANK 3

**(a) Current state.** 8 frames, hard-coded everywhere: extraction
`src/utils/generate_VideoMLLM_embedding_lora_HF.py:114-118` (`--num_frames` default 8) and the SFT
builder `src/utils/build_lora_sft_data.py:39` (`NUM_FRAMES = 8`, `IMG_TOKENS = "<image>"*8`). Budget
constraints: `cutoff_len: 4096` (config L21), `video_max_pixels: 16384` (L4), `MAX_TRANSCRIPT_CHARS
1500` (builder L40). 8 frames already sit well inside 4096 tokens, but 16-32 would pressure it.

**(b) Covering ban? NO.** S2S/W2-B/W2-C/CTF (F35/F37/F39) killed *temporal SET-matching and
order-kernels over the 8 frame groups* — that is the retrieval **operator over a fixed 8-frame
sampling**, not the **sampling density**. F35's cumulative-causal finding concerns how the 8 groups
relate, not whether 16/32 frames give better visual coverage of the video (a coverage axis, not an
order axis). No banked finding varies frame count.

**(c) In-box.** LEGAL.

**(d) GPU-h.** 16 frames ≈ 2× vision tokens ⇒ longer sequences (cutoff_len likely must rise ⇒ memory
pressure), full re-extraction of every cache + re-SFT (~1.5-2×) + head. Est. **~6-8 A100-h/dataset**
for the full chain. 32 frames heavier still.

**(e) Cheapest kill gate.** **None on cached features** — this cell can only be tested by spending GPU
on re-extraction; there is no $0 pre-screen. That absence of a cheap gate is itself the argument for
its middling rank.

**(f) Prior of ≥+1pt.** **LOW-MODEST (~8-12%)**. More frames = more visual coverage, could help the
visually-grounded HateMM leg — but HateMM already passes, and §0 fact-1 says the EN bottleneck is the
image *encoder quality* (collapse), not the frame *count*; adding frames won't heal a collapsed
tower. Unlikely to add a dataset.

**(g) RANK 3.**

---

### C4 — Head / objective engineering: fusion architecture · loss family · proj/map dims · topk  ·  RANK 4

**(a) Current state.** Deployed head (`scripts/slurm/enc3seed_lora_hatemm.sbatch:53-66`):
`--fusion_mode align` (Hadamard product of L2-normed projections, `src/model/classifier.py:119-120`),
`--loss triplet --hybrid_loss True` (triplet-margin + BCE), `--proj_dim 1024 --map_dim 1024`,
`num_layers 3` (default), `--topk 20 --majority_voting arithmetic`, `--metric cos`,
`--no_hard_negatives 1 --no_pseudo_gold_positives 1`. **The head code already supports untested
alternatives:** `classifier.py:80-122` implements `fusion_mode ∈ {concat, align, cross}` (cross = bmm
outer product, `map_dim²`-d); `run_rac.py:141` `--loss {naive, triplet, contrastive}`. No InfoNCE/
SupCon path exists (would need ~30 LOC). **No banked sweep** of fusion-architecture, loss-family,
dims, or topk on the video datasets appears anywhere in F1-F60.

**(b) Covering ban? PARTIAL — and the precise answer to "does FA/F50 ban learned fusion heads" is
NO.** F50 ban scope, verbatim: *"do not re-propose **fixed compositions, reweights, or per-modality
temperatures** over banked frozen features; conversion requires adaptation (F45) or a new
information source."* A **trained** fusion head (concat / cross / a small cross-attention block
optimised end-to-end with triplet+BCE) is **none of those three** — it is a nonlinear trained
operator, not a fixed composition. FA (F50) measured only (A1) a scalar within-Qwen reweight (pure
rotation at every w) and (A2) a fixed cross-encoder CLIP-img+Qwen-text concat scored by kNN/oracle —
neither is a trained fusion-mode swap. **However**, two real dampers: (i) F50's oracle argument caps
the *linear-composition family* on EN (best AUC 0.898 → oracle +0.025) — a trained nonlinear head
can in principle exceed a linear composition, but the deployed align-MLP is *already* a nonlinear
head and defines the floor, so headroom is thin; (ii) topk is near-flat and direction-inconsistent
(§0 fact-2). P4 (schema-distill) kill is unrelated — it measured that generated schema *text* is
redundant (AAAI25 ensembling result), not head fusion architecture.

**(c) In-box.** LEGAL. But **D7-DEAD** — head engineering is not an MLLM-novelty mechanism; a gain
here is a performance/ablation row only.

**(d) GPU-h.** **~$0-cheap** — head-only over cached features, ~20-25 s/run per the deployed sbatch.
A whole fusion×loss×dims family test = minutes-to-~1 h total.

**(e) Cheapest kill gate.** The axis itself is the gate — a single **pre-registered** family test
(concat vs align vs cross vs one cross-attention head; triplet+BCE vs InfoNCE) with one held bar and
a multiplicity correction (sweeping = forking paths otherwise). Door-closing at worst.

**(f) Prior of ≥+1pt.** **LOW (~5-8%)** for a real ≥2-dataset gain (D7-dead; EN linear-composition
family oracle-capped). But because cost≈0, **prior×(1/cost) is high** — highest door-closing EV of
any cell.

**(g) RANK 4** (on prior×cost) — cheap, genuinely-unswept, but D7-dead.

---

### C5 — LoRA recipe axes: rank · epochs · lr · target-module set · DoRA / rsLoRA / PiSSA / LoRA+  ·  RANK 5

**(a) Current state.** `hatemm_qwen25vl_lora_sft.yaml:11-13,38-39`: rank 16 / alpha 32 / 3 epochs /
lr 1e-4 / target = 7 LLM modules; **no DoRA/rsLoRA/PiSSA/LoRA+ flags set**, all defaulting off. The
on-disk LLaMA-Factory supports every one of them (`finetuning_args.py:91-117`).

**(b) Covering ban? NO — but adjacent.** F51's "adaptation two-object closure" is about the *adapted
object* (encoder vs joint encoder+decision), NOT the *recipe knobs* — it does not measure rank/epoch/
DoRA. No banked sweep exists. Multiplicity caveat: recipe sweeps are forking paths unless a single
preregistered family test with a held bar.

**(c) In-box.** LEGAL.

**(d) GPU-h.** Each SFT ~3-3.5 h/dataset + re-extraction + head; a rank×epoch×variant grid is many
GPU-h. **Expensive per informative cell.**

**(e) Cheapest kill gate.** None pre-GPU (adapted features don't exist). A DoRA/rsLoRA single-variant
head-readout could be gated by the same KS-2/oracle bars as C1.

**(f) Prior of ≥+1pt.** **LOW (~5-8%)** — recipe-tuning of a D7-dead lever whose EN bottleneck is the
frozen vision tower (§0 fact-1), which no LoRA recipe touches. rsLoRA/DoRA might sharpen the *text*
stream further, but text is not EN's bottleneck (EN text AUC is already 0.925 under generic LoRA).

**(g) RANK 5.**

---

### C6 — Extraction-prompt axis: prompt variants / multi-prompt averaged embeddings  ·  RANK 6

**(a) Current state.** Embeddings come from ONE fixed instruction pair:
`generate_VideoMLLM_embedding_lora_HF.py:59-66` (`IMG_INSTRUCTION`, `TEXT_INSTRUCTION`). No prompt
variation or multi-prompt averaging is implemented.

**(b) Covering ban? NO literally — grazes the ensemble-veto spirit.** The banned constraint is
"**cross-seed** ensembles" (`state/directions_tried.json` banned_constraints). Multi-*prompt*
single-model averaging is NOT cross-seed, so the veto does not literally cover it. P4 (schema-distill,
dead) killed generated schema *text* as redundant — not extraction-prompt variation of the encoder
pooling span. So neither literally covers it, but both point the same way (a within-model average is
ensemble-flavoured; D1 conditional-redundancy applies to reshuffling the same frozen representation).

**(c) In-box.** LEGAL-but-grazes-veto-spirit.

**(d) GPU-h.** Re-extraction per prompt (~GPU-h/dataset/prompt) + head. Moderate; scales with #prompts.

**(e) Cheapest kill gate.** None on cached features. Would need per-prompt extraction, then a $0
conditional-info screen of the averaged vs single-prompt embedding.

**(f) Prior of ≥+1pt.** **LOW (~4-6%)** — prompt variation reshuffles the same frozen encoder; D1.

**(g) RANK 6.**

---

### C7 — Curriculum error-variant (i-b), unrun  ·  RANK 7 (covered-by-scope)

**(a) Current state.** `src/utils/build_curriculum_sft_data.py:106,137,266-267` implements
`--mode error` (i-b: `c_i = 1{LOO vote misclassifies i}`) alongside the registered `softconf` (i-a).
Cand-2 ran **only softconf** (F52/F56); i-b is code-present, never run.

**(b) Covering ban? YES (by scope).** Cand-2 queue note (`state/directions_tried.json`): "do NOT
re-run curriculum variants (tactics) without new structural premise"; F52 "Only design (i)
confusion-weighted single-video sampling is viable." Plus multiplicity: the cand-2 single-draw
binding is consumed (F0.2; rep2 F59 = the one extra draw, "no further draws ever"). i-b is a tactic
swap of a lever that already tied (ZH) / marginally-passed (HateMM val-sel).

**(c) In-box.** LEGAL. **(d)** ~7-8 A100-h (ZH+HateMM 3-arm). **(e)** K-C2-0 mining-validity runs $0;
full verdict needs GPU. **(f)** **VERY LOW (~3-5%)** — banned-by-scope tactic. **(g) RANK 7.**

---

### C8 — kNN-vote / protocol / augmentation micro-variants  ·  (folded, LOW)

Bundled because each is thin and mostly covered:
- **topk / adaptive-k / metric(l2 vs cos) / majority(mean vs arithmetic):** decision-side (Axis A/H).
  §0 fact-2 shows topk near-flat + direction-inconsistent; B5 (F34) closed operating-point conversion
  on ZH. LOW, largely covered.
- **dev-set protocol (val-sel vs final-epoch):** this is the ZH-marginality question, not a method
  lever; F45 attributes ZH val-sel FAIL to 78-dev selection noise (dev plateaus ep19, test climbs
  ep29). A protocol *choice* for the user, not an untested cell.
- **train-time frame augmentation (temporal jitter / crop):** genuinely un-implemented, distinct from
  AUG (F60 killed *MLLM-generated* augmentation; frame-jitter is different) — but decision/data-side
  thin, and shares AUG's "adds no dev items, cannot move dev-argmax" logic. LOW (~4%).
- **longer head training / warmup / lr schedule:** head already 30 epochs, warmup 5; forking-path.

None of these rises above LOW; listed for completeness so the audit is not accused of cherry-picking
only the favourable gaps.

---

## 2. TOP-5 RANKED (prior × cost) — the concrete refutation

| # | Cell | Untested? (covering ban) | In-box | GPU-h/ds | Prior ≥+1pt | Cheapest gate | Verdict |
|---|---|---|---|---|---|---|---|
| **1** | **Vision-tower/projector unfreeze** (LoRA-on-vision) | **YES** — no ban; F58/F55/F44 are LLM-only or cross-encoder, never same-encoder vision-adapted | ✔ (1 config flip) | ~4-6 | **~12-15%** | §0 fact-1 prior-mover; then F55 oracle + KS-2 image-moved bars | **Representation-class, targets the un-repaired EN image collapse — the strongest gap** |
| **2** | **Learned-audio 3rd stream** (Whisper-encoder, on disk) | **YES** — APX ban is scoped to *classical* prosody; W2-D was inferred not measured | ✔ (Whisper local; wav2vec2 = in-box download) | ~2-4 | **~10-14%** | cheap Whisper-base encode → APX's $0 conditional-info screen | **The one unused input channel; blessed gain source; terminus PARKED-not-killed** |
| **3** | **Frame budget 8→16/32** | **YES** — S2S/CTF killed the operator over 8 frames, not the density | ✔ | ~6-8 | **~8-12%** | none $0 (re-extract required) | Untested coverage axis; but EN bottleneck is encoder not count |
| **4** | **Head fusion-arch + loss family** (concat/cross/x-attn; InfoNCE/SupCon) | **PARTIAL** — F50 bans *fixed* compositions, NOT *trained* fusion heads | ✔ (D7-dead) | **~$0** (head-only, cached) | ~5-8% | the axis is the gate; 1 preregistered family test | Cheapest door-closer; highest prior×(1/cost) |
| **5** | **LoRA recipe: DoRA/rsLoRA/rank/epochs** | **YES** — F51 closes the object, not the recipe knobs | ✔ (D7-dead) | ~3-3.5×grid | ~5-8% | none $0 | Code-supported, unswept; but EN bottleneck untouched by recipe |

**Bottom line for the terminus claim.** `TERMINUS_round3` §4's binding sentence — "every injection
point is closed by a binding verdict or a calibrated-zero G0-cond gate" — is **overstated**. It is
true for the *operator families that were probed* (decision-side, retrieval-object, temporal-order,
cross-modal-grounding, threshold, scale-in-tier). It is **false** for at least two cells that were
**argued-down in prose but never measured**: the vision-tower/projector unfreeze (C1) and the
learned-audio channel (C3). Both are in-box, zero-download or one-download, and both live in the only
two gain classes the campaign's own diagnosis frame blesses. They do not promise the goal — priors
are LOW-MODEST and C3 is novelty-thin — but they refute "we exhausted the method space" with
concrete, config-line-level gaps. The honest revision: *"every **probed operator family** is closed;
two representation/new-channel cells (vision-adapted encoder, learned audio) remain unmeasured, at
LOW-MODEST prior, and were deferred as prose priors rather than run."*

---

## PROVENANCE
- Configs: `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/hatemm_qwen25vl_lora_sft.yaml:4,11-15,21,38-39`;
  `finetuning_args.py:91-117,539,543`.
- Extraction/SFT: `src/utils/generate_VideoMLLM_embedding_lora_HF.py:59-66,114-118`;
  `src/utils/build_lora_sft_data.py:39-40,48-76`; curriculum `build_curriculum_sft_data.py:106,137,266-267`.
- Head/loss/knn: `src/model/classifier.py:80-122`; `src/model/loss.py:12,85-113`;
  `src/run_rac.py:118,132,141-158,162-200,223`; deployed `scripts/slurm/enc3seed_lora_hatemm.sbatch:53-66`.
- Kill ledger read: `state/findings.jsonl` F1-F60; `state/directions_tried.json`;
  `TERMINUS_round3_mllm_plus3.md`; `D7_RULING_DOSSIER.md`; `HATEMM_LORA_STREAM_DECOMP.md` (F58).
- Probe (scratch, uncommitted): `scratchpad/redteam_stream_topk_probe.py` — banked train/dev caches only,
  machinery-validated vs F58 to ~0.01 AUC. Env check: `transformers 4.49.0` audio classes present;
  `whisper-base` + `whisper-large-v3` on disk; `torchaudio` missing; only `.mp4` (0 `.wav`).
- CPU-only, zero GPU/Modal/SLURM/test-touch; no `state/` mutation. Committed on `main`, not pushed.
