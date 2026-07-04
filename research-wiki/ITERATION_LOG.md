# Iteration Log — Hateful Video Detection (RGCL → video)

Chronological record of design iterations: goal, variants explored, adversarial outcomes,
decision + rationale, open decisions. Newest entries appended at the bottom.

---

## Iteration 1 — 2026-07-01 — MLLM retrieval-guided contrastive + kNN for hateful video

**Goal.** Produce a defensible Iteration-1 method that (1) has a clear novelty delta vs a SET of
closest hateful-VIDEO SOTA (not "first in the world"), (2) has a credible path to acc > 85% on
HateMM and MultiHateClip (macro-F1/P/R + acc), (3) introduces an MLLM and states how our use
differs from RA-HMD (meme LMM-RGCL) and from reasoning-VLMs (prompt/CoT/DPO). Cross-lingual is
NOT a novelty axis; temporal/audio/implicit are evaluation slices, not novelty.

**What was explored (4 drafted variants).**
1. **TREMR** — LoRA-tuned Qwen2.5-VL as a temporal+prosody-aware video-native encoder;
   whole-video AND **segment-level** retrieval memory; reindex-cached kNN. (mllm-encoder-sft)
2. **MLLM-RGCL-Frozen** — FROZEN Qwen2.5-VL as a feature extractor (hidden-state pooling, or
   Variant-B self-description re-embedded); train only the ~6-8M RGCL head; kNN-vote. No SFT.
3. **RaKE (RGCL-VidRaK)** — frozen-MLLM structured harmfulness *rationale* as a 3rd retrieval
   *key* modality, fused with CLIP frames + audio + transcript; retrieval-contrastive + kNN.
4. **RA-MDK (Distill-then-Vote)** — retrieval-augmented in-context MLLM *teacher* (reads clip +
   FAISS-retrieved labeled neighbors) distilled into a frozen-CLIP kNN student; zero MLLM calls
   at inference.

**Adversarial outcomes (skeptic reviews).**
- **All four novelties SURVIVED** — none was killed. The gap_map **G1** core
  (retrieval-guided-contrastive embedding + kNN-vote inference in hateful video) is genuinely
  unclaimed (MoRE = retrieval-for-experts, BCE only, frozen cosine retriever, no kNN head).
- **Convergent finding across reviews:** the **MLLM-as-encoder / MLLM-rationale-as-feature**
  pattern is *not* a field novelty — RAMF and HVGuard already do frozen-VLM-features → trainable
  head, and MoRE benchmarks Qwen2-VL/LLaVA-OV. So the MLLM is a **performance lever that must be
  ablated**, not the contribution. (Explicit in the MLLM-RGCL-Frozen review; RAMF named as the
  live threat to the "our MLLM use is novel" sub-claim.)
- **TREMR review verdict: FIX, don't drop** — the *only* delta a hostile reviewer cannot reframe
  as "RGCL/RA-HMD on a new dataset" is the **segment-level retrieval memory** (within-video
  drifting non-hate segment as a retrieval-mined hard negative), which is **structurally
  impossible in a static meme**. Demote temporal-M-RoPE and the prosody adapter to ablations.
  Feasibility=medium (LoRA reindex over a moving 7B embedding space, PEFT not in base python, a
  non-trivial two-tower→single-tensor refactor undersold as "minimal").
- **MLLM-RGCL-Frozen review verdict: FIX (keep the G1 core, demote the MLLM).** Feasibility=HIGH.
  Central risk: if MLLM features don't beat CLIP by an ablated margin, it collapses to "RGCL with
  a bigger encoder"; Variant B collides with RAMF.
- **RaKE / RA-MDK verdicts: KEEP with tightened framing** — both survive but their closest threat
  is RAMF (frozen-VLM reasoning into a trainable head); each needs a strict ablation ladder and a
  neutral-template / no-verdict-leak guard, and both carry more code + attribution risk.
- **Performance realism (all variants):** ">85% on BOTH HateMM AND MHClip" is optimistic, not
  evidenced. HateMM binary >0.85 is credible (MM-HSD 0.874 proves reachable); MHClip-ZH >0.85 is
  aspirational (MoRE 0.785, ASR-bound); 3-class 85% is impossible (field ceiling 0.63/0.50).

**Decision.** MERGE: **`MLLM-RGCL-Frozen` (spine) + TREMR's segment-level retrieval memory (video
delta)** = **SR-RGCL-Vid**. Take the cheapest, highest-confidence, HIGH-feasibility spine (frozen
MLLM/CLIP encoder → learned retrieval-guided-contrastive InfoNCE → kNN-vote, ~6-8M trainable
params, minimal code delta) and graft on the **one structurally-unique video delta** that
RA-HMD/RGCL cannot have (segment retrieval). Frozen-only this iteration; **defer LoRA-SFT (TREMR
Phase B), teacher-distillation (RA-MDK), and rationale-as-key (RaKE) to Iteration 2**, layered
only if the frozen ablations justify the cost/risk.

**Rationale (tie to reviews).** (a) Novelty must rest on the encoder-agnostic G1 head + segment
retrieval, NOT the MLLM (reviews' convergent point). (b) Segment retrieval is the only
"not-just-a-new-dataset" delta that survived hostile reframing (TREMR's strongest_surviving_claim).
(c) Frozen spine = HIGH feasibility and dodges the LoRA-reindex/PEFT/refactor costs the reviews
flagged; it is exactly TREMR's own risk-#1 graceful-degradation fallback. (d) Performance claim
re-scoped to the defensible spine: beat MoRE on all three under identical binary protocol +
approach MM-HSD on HateMM at a fraction of trainable/inference cost, with a training-free kNN head.

**Protocol.** Lead with BINARY (offensive∪hateful) macro-F1/P/R + acc (MoRE's protocol; where >85%
is credible); report 3-class MHClip (Hateful/Offensive/Normal) honestly as a harder secondary slice
(target: beat 0.63 EN / 0.50 ZH M-F1, NOT 85%). Reproduce MoRE's exact binary merge + train+val
memory bank; re-run all baselines on our split (cross-paper numbers untrusted).

**Open decisions (forks for the user).**
1. MLLM model: Qwen2.5-VL-7B (native ZH, recommended) vs LLaVA-OneVision-7B (lowest integration
   risk) vs Qwen2.5-VL-3B (disk-lean).
2. Encoder config: MLLM-A hidden-state (novel, maybe noisier) vs MLLM-B description-re-embed
   (robust, RAMF-adjacent → fallback only) vs both, per-dataset by val.
3. Protocol emphasis: binary-only vs binary + 3-class MHClip secondary (needs N-logit head change,
   which breaks the "RGCL core unchanged" claim for that slice).
4. Segment-retrieval scope: gold-span-only (HateMM/MHClip) vs also pseudo-segments for span-less
   datasets.
5. Confirm frozen-only this iteration (defer LoRA / distillation to Iteration 2).

**Artifacts.** `research-wiki/DESIGN_iter1.md` (full method + novelty table + mllm-diff + protocol
+ roadmap); idea node `ideas/rgcl-mllm-video-iter1.md`.

---

## Phase 0 results — protocol & baseline alignment (2026-07-01)

Executed roadmap §5 Phase 0.1: aligned evaluation to the goal metrics (macro-F1 / macro-P /
macro-R / accuracy, binary harmful-vs-normal), fixed the model-selection bug, and re-ran the
honest frozen-CLIP-RGCL binary baseline on all four datasets (cached CLIP ViT-L/14-336
embeddings, 30 epochs, `--Faiss_GPU False`, SLURM).

### Code changes made

1. **Model-selection fix — `src/run_rac.py` (`model_pass`).**
   Previously, with `--hybrid_loss True` the best model was chosen by the *hybrid classifier-head
   dev accuracy* (`acc_ = acc_ if args.hybrid_loss else acc`). That head is stuck (it collapses to
   an all-positive predictor — dev/test macroF1 ≈ 0.28–0.37 in every log), so the "best" checkpoint
   never tracked the genuinely best retrieval epoch. **Now selection uses `Val_Retrieval acc`,
   tie-broken by `Val_Retrieval roc`** (the retrieval/kNN-vote metric that is this method's primary
   objective). The saved `best_model_{epoch}_{val_acc}.pt` == the best-retrieval epoch. The
   CPU-faiss path is untouched.

2. **Macro metrics — `src/utils/metrics.py`.**
   - `compute_metrics_retrieval` (kNN-vote / retrieval path) now additionally computes
     `average='macro'` precision/recall/F1 (sklearn, `zero_division=0`) and returns them as an
     extra `macro` dict (8th return value). The binary-positive P/R/F1 prints are kept.
   - `eval_metrics` (classifier-head path) now prints macro-F1/P/R alongside the binary metrics
     for binary datasets.
   - `src/run_rac.py` prints, every epoch, for **both Val and Test**:
     `Val_Retrieval Epoch N macroF1 .. macroP .. macroR .. acc .. roc ..` and the `Test_Retrieval`
     equivalent. `src/model/evaluate_rac.py:final_evaluation` prints the macro line too.
   - Callers updated for the new 8-tuple (`run_rac.py` ×2, `evaluate_rac.py` ×1). Label semantics
     unchanged (binary).

### Aligned honest baseline table (val-selected epoch → Test macro metrics)

Frozen CLIP ViT-L/14-336, mean-pooled 8-frame video + title/transcript text tower; RGCL head
(align fusion, triplet + hybrid BCE, topk=20, arithmetic vote); seed 0; single run. Epoch chosen
by Val_Retrieval acc (tie-break ROC) under the FIXED selection. "M-F1" = macro-F1.

| Dataset | selEp | Test **M-F1** | Test M-P | Test M-R | Test **acc** | Test ROC | SOTA target (M-F1) | gap to acc>0.85 | gap to beat MoRE |
|---|---|---|---|---|---|---|---|---|---|
| **HateMM** (EN) | 24 | **0.8172** | 0.8258 | 0.8120 | 0.8279 | 0.8903 | MoRE 0.8235 / MM-HSD 0.874 | −0.022 (0.8279) | −0.006 below MoRE M-F1 |
| **MHClip-Y** (MHC, EN) | 18 | **0.6219** | 0.7259 | 0.6161 | 0.7453 | 0.7742 | MoRE 0.7519 | −0.105 (0.7453) | −0.130 below MoRE M-F1 |
| **MHClip-B** (MHC_zh, ZH) | 29 | **0.7706** | 0.7690 | 0.7723 | 0.8054 | 0.8382 | MoRE 0.7475 / HVGuard 0.822 | −0.045 (0.8054) | **+0.023 already beats MoRE M-F1** |
| **ImpliHateVid** (EN) | 18 | **0.9101** | 0.9122 | 0.9103 | 0.9102 | 0.9722 | TCL 0.8773 | **+0.060 acc>0.85 met** | **+0.033 exceeds TCL M-F1** |

### Gap-to-target summary (honest floor)

- **HateMM:** M-F1 0.8172 is ~0.006 short of MoRE (0.8235) and 0.057 short of MM-HSD (0.874);
  acc 0.8279 is 0.022 short of the >0.85 goal. This is the floor the segment-retrieval + MLLM
  levers must lift.
- **MHClip-Y (EN):** weakest. M-F1 0.6219 (test) is far below MoRE 0.7519; acc 0.7453 far from 0.85.
  Note the large Val→Test drop (val M-F1 0.7297 → test 0.6219) — small split (161 test) + hard EN
  hate/offensive confusion. Biggest headroom for the retrieval-contrastive+kNN delta.
- **MHClip-B (ZH):** M-F1 0.7706 already **exceeds MoRE (0.7475)** even with EN-CLIP text on Chinese;
  acc 0.8054 is 0.045 short of 0.85 (below HVGuard-ZH 0.822). Config MLLM-B (native-ZH) is the lever.
- **ImpliHateVid:** already strong — M-F1 0.9101 / acc 0.9102, **exceeds TCL 0.8773 and clears acc>0.85**.
  Goal here is to hold/exceed, not chase.

Bottom line under the aligned protocol: the honest frozen-CLIP floor already **beats MoRE on 2/4
(MHClip-B, ImpliHateVid)**, is **~0.006 M-F1 below MoRE on HateMM**, and is **clearly behind on
MHClip-Y**. acc>0.85 is currently met only on ImpliHateVid. The Phase 1–2 levers (segment retrieval,
MLLM encoder) are aimed at HateMM (+MM-HSD) and MHClip-Y first.

### Leakage sanity check (train/val/test id-disjointness)

Checked flattened `ids` across the three cached splits per dataset:

| Dataset | train / dev / test sizes | train∩dev | train∩test | dev∩test | intra-split dups |
|---|---|---|---|---|---|
| HateMM | 744 / 107 / 215 | 0 | 0 | 0 | 0 |
| MHClip-Y (MHC) | 550 / 80 / 161 | 0 | **1** | 0 | 0 |
| MHClip-B (MHC_zh) | 579 / 78 / 149 | 0 | 0 | 0 | 0 |
| ImpliHateVid | 1283 / 325 / 401 | 0 | 0 | 0 | 0 |

**One leak found — MHClip-Y (MHC/EN):** id `k9OtaMbK0Ac` appears in **both train.jsonl and
test.jsonl** (identical text, identical cached img/text feats, both label=1). It originates in the
upstream `data/gt/MHC/{train,test}.jsonl` (one duplicate line each). Impact is a single positive
sample out of 161 test (0.6%), so the effect on the reported MHClip-Y test metric is negligible,
but it should be removed from train before any headline claim. All other splits are fully disjoint
with no intra-split duplicate ids. **Action item:** drop `k9OtaMbK0Ac` from MHC train (or test)
and regenerate the MHC cache before Phase 1.

### SLURM jobs

All via `sbatch scripts/slurm/train.sbatch` (no `--time`; landed PENDING JobHeldUser → auto-released),
`--export=ALL,DATASET=<DS>,EPOCHS=30`. Each pushed ckpts+logs to `b2:junyi-data/RGCL_video/logs/<DS>`.

| JobID | Dataset | Final state | Elapsed |
|---|---|---|---|
| 12103 | HateMM | COMPLETED (0:0) | 00:00:47 |
| 12104 | MHC (MHClip-Y) | COMPLETED (0:0) | 00:00:42 |
| 12105 | MHC_zh (MHClip-B) | COMPLETED (0:0) | 00:00:44 |
| 12106 | ImpliHateVid | COMPLETED (0:0) | 00:01:20 |

Logs: `slurm/logs/p0_<DS>_<jobid>.out`. Selected checkpoints:
`logging/Retrieval/<DS>/RAC_video/..._Ep30_..._hybrid_loss/ckpt/best_model_<selEp>_<valAcc>.pt`.

---

## Phase 1 — segment retrieval (HateMM) — 2026-07-01

Goal: validate the headline novelty (segment-level retrieval memory, DESIGN_iter1 §2.3) on
HateMM as the make-or-break. **BLOCKED at STEP 0 — HateMM has no usable spans locally.**

### STEP 0 — span availability finding (BLOCKING, verified, not fabricated)

The design's headline novelty needs per-video hate/non-hate **frame-span rationales**. The
HateMM *paper* (Das 2023, ICWSM) did collect them ("manually annotate them as hate or non-hate,
along with the frame spans which could explain the labelling decision") — but the paper itself
reports them as **"collected but unused"**, and they are **not present in the local copy**.

Inspected `/data/jehc223/HateMM/annotation(new).json` (1066 entries, the only annotation file):
- Keys across ALL entries: `Video_ID, Title, Transcript, Emotion, Frames_path, Audio_path,
  Frames_description, Text_description, Mix_description, Label` — **no span / start-time /
  end-time / frame-range / segment / temporal key of any kind**.
- Only whole-video `Label` (427 Hate / 639 Non Hate). `Title`, `Emotion`, and all three
  `*_description` fields are 100% empty. `Transcript` is whole-video (no per-segment slicing).
- The grep hits for "victim"/"span" were substrings *inside transcript text values*, confirmed
  not keys (programmatic key scan returned NONE).
- Frames dir has a fixed 32 frames/video (uniform sampling, no hate/non-hate frame labels).
- No HateMM span/annotation CSV/JSON/XLSX anywhere on the filesystem (`/data/jehc223/**`) or on
  B2 (`b2:junyi-data/**`). The pipeline's `data/gt/HateMM/*.jsonl` is whole-video `{id,text,label}`.

**Verdict: STEP 0 FAILS. Spans are absent → STEPS 2–4 (segment features, segment-RGCL loss,
ablation) cannot run on real HateMM spans. Stopped cleanly per instruction; did not fabricate
spans or build blindly.** Fallback = DESIGN_iter1 §6 fork #4: derive **shot-boundary
pseudo-segments** (e.g. PySceneDetect over the raw videos) as span-less segment units, with the
within-video non-hate pseudo-segment as the drifting hard negative. This is a heuristic (weaker
claim than gold spans) and is deferred pending a decision on the fork. Also possible: obtain the
original HateMM span release from the authors (upstream repo) — not available locally/on B2.

### STEP 1 — MHClip-Y (MHC/EN) leak fix (done; independent of spans, needed regardless)

Removed leaked id `k9OtaMbK0Ac` (in both MHC train.jsonl and test.jsonl, label=1) from **train only**:

| Artifact | Before | After | Note |
|---|---|---|---|
| `data/gt/MHC/train.jsonl` | 550 lines | **549** | leaked line dropped; test (161) & val (80) untouched |
| `data/gt/MHC/test.jsonl` | 161 (has id) | 161 (keeps id) | leak retained in test per instruction |
| cache `train_openai_clip-vit-large-patch14-336_HF.pt` | 550 rows | **549** | pulled from B2, surgically dropped row idx 521 |
| — ids / img_feats / text_feats / labels | 550 / (550,1024) / (550,768) / (550,) | 549 / (549,1024) / (549,768) / (549,) | dropped in lockstep, re-verified via loader flatten |
| — train label dist | {0:381, 1:169} | {0:381, **1:168**} | one positive removed |

Post-fix: leaked id absent from train (jsonl + cache), 549 unique ids (no dup), rows aligned.
Fixed cache pushed back to `b2:junyi-data/RGCL_video/embeddings/MHC/train_...HF.pt`
(3.776→3.770 MiB). gt jsonl is local-only (not git-tracked, not mirrored on B2).

### Code changes
None to `loss.py`/`retrieval.py`/`run_rac.py` — STEP 0 blocked before any segment-loss code was
written (correctly, to avoid building on absent data). Only data fixes (jsonl + cache surgery).

### SLURM jobs
None — STEP 0 gated before any GPU job. STEP 1 is CPU-only torch load/save (no faiss/CLIP), run
inline. Datasets (HateMM, MultiHateClip raw) left in place per instruction for later extraction.

---

## Phase 1 take-2 — segment retrieval (MultiHateClip) — 2026-07-01

Goal: validate the headline novelty (segment-level retrieval memory, DESIGN_iter1 §2.3) on
**MultiHateClip** (EN=`MHC`/YouTube, ZH=`MHC_zh`/Bilibili), since HateMM had no local spans and
MHClip is *documented* to carry per-video hateful/offensive segment timestamps + contributing
modality. **BLOCKED at STEP 0 — the segment spans exist UPSTREAM but are ABSENT from our local
download.** Stopped cleanly; did not fabricate spans or build blindly.

### STEP 0 — span availability finding (BLOCKING, verified, not fabricated)

Root cause is different from HateMM: MHClip's spans are a **real, published annotation** — they
were simply **not included in the annotation file we downloaded** (Social-AI-Studio release ships
video-IDs + a reduced annotation JSON; full segment/target/modality labels are gated).

Upstream (MultiHateClip, ACM MM'24, arXiv 2408.03468v2), verified via the paper: annotators
mark **fine-grained continuous timestamps** — *"they must specify the start and end times of the
segment where the hateful or offensive statements occur"* — plus **contributing modality**
(text / visual / audio, single or multi) and **target victim** (Woman / Man / LGBTQ+ / Other).
This is *exactly* the annotation our §2.3 novelty requires.

Local copy inspected — `/data/jehc223/Multihateclip/{English,Chinese}/annotation(new).json`
(EN 891 entries, ZH 897 entries), the ONLY annotation files present:
- Union of fields (both langs, identical schema): `Video_ID, Title, Transcript, Emotion,
  Frames_path, Audio_path, Frames_description, Text_description, Mix_description, Label`
  — **no start/end/timestamp/segment/modality/target/victim/interval/duration key of any kind.**
- Programmatic key scan: grep for `(time|segment|span|start|end|clip|second|duration|offset|
  moment)` keys → **NONE** in both files. The only regex hits for `target/victim/modality/audio/
  end` were substrings *inside* Title/Transcript *values*, not keys (confirmed).
- Only whole-video `Label`:
  - **EN (MHC):** Normal 601 / Offensive 218 / Hateful 72  (891 total)
  - **ZH (MHC_zh):** Normal 605 / Offensive 180 / Hateful 112  (897 total)
- `Emotion`, `Frames_description`, `Text_description`, `Mix_description` are 100% empty.
  `Transcript` is whole-video (EN 840/891, ZH nonempty) — no per-segment slicing.
- Other dirs are not annotations: `quad/` = 819(EN) per-video frame JPGs (`quad_001.jpg`…);
  `splits/*.csv` = bare video-ID lists (no header, no timestamps); `video/`,`video_mp4/`,
  `audios/` = raw media.
- **Coverage stats we CANNOT compute** (the point of the exercise): #videos with ≥1 labeled
  segment, #segments, avg segment length, per-segment label/modality distribution, hate vs
  non-hate span marking — **all zero available locally**. There is nothing to parse.

**Verdict: STEP 0 FAILS for MHClip too. Segment spans are absent from the local download →
STEPS 1–4 (segment features, segment-RGCL loss, ablation) cannot run on real MHClip spans.**
Unlike HateMM (spans "collected but unused", not released), MHClip's spans are a genuine,
recoverable published resource — the cleanest unblock is to **request the full annotation from
the authors** (han_wang@sutd.edu.sg, per the repo), not a heuristic.

### Two options for the user (STEP 0 mandate: scope, don't build)

**Option A — obtain the real MHClip segment annotations (RECOMMENDED; preserves the strong claim).**
The spans are published and continuous-timestamp; only the full-annotation file is missing
locally. Getting it makes the entire §2.3 novelty runnable *with gold spans* (the strongest,
non-heuristic version). Action: email the MultiHateClip authors / check the gated release. No
compute needed until the file arrives.

**Option B — shot-boundary pseudo-segment fallback (heuristic; DESIGN_iter1 §6 fork #4).**
Derive span-less segment units from scene cuts on the raw mp4s; within-video non-hate
pseudo-segment = drifting hard negative. Feasibility on this env:
- Raw videos present & readable: EN 792 mp4 (+792 webm), ZH 814 webm. `decord` opens them
  cleanly (verified: e.g. 651 frames @30fps ≈21.7s).
- **`ffmpeg`: NOT on PATH** (no system, no conda-env, no `imageio-ffmpeg`).
- **`PySceneDetect`: NOT installed** (`ModuleNotFoundError: scenedetect`).
- **`opencv` (`cv2`): NOT installed** (PySceneDetect's frame backend).
- Present: `decord`, `pyav` (bundles ffmpeg libs), `faiss` (CPU). So shot detection is
  *achievable* but needs installs: `pip install scenedetect opencv-python-headless`
  (PyAV/decord can supply frames if a pure-opencv path is avoided). This is a **weaker claim**
  than gold spans (pseudo-segments are not the annotated hateful spans, and the "contributing
  modality" signal is entirely lost), and adds a heuristic a hostile reviewer will contest.

Recommendation: **Option A first** (the spans are real and free); fall back to B only if the
authors don't share. Did not install packages or run scene detection without a decision.

### Code changes
None to `loss.py`/`retrieval.py`/`run_rac.py` — STEP 0 gated before any segment-loss code was
written (correctly, to avoid building on absent data).

### SLURM jobs
None — STEP 0 blocked before any GPU job (no segment-feature extraction, no training). Env
probe for the fallback was CPU-only/inline. MultiHateClip raw (~27G) left in place per
instruction (needed for extraction once spans arrive).

---

## Phase 2 — MLLM encoder (Qwen2.5-VL-7B) ablation — 2026-07-02

Executed roadmap §5 Phase 2 (DESIGN_iter1.md §2.1 Config MLLM-A). Central question: **do
frozen-MLLM hidden-state features beat frozen-CLIP features on the identical RGCL head?**

### Extractor design (Config MLLM-A, hidden-state pooling)

New script `src/utils/generate_VideoMLLM_embedding_HF.py`, sibling to the CLIP extractor,
reusing its decord→PyAV 8-frame sampler verbatim. Frozen **Qwen2.5-VL-7B-Instruct**, bf16,
`torch.no_grad()`, `attn_implementation="sdpa"` (flash-attn build failed on this box →
intentionally skipped, sdpa is the safe default), `output_hidden_states=True`, single forward
per stream, `max_pixels=360*420` (set at processor construction; the 4.49 `__call__` ignores it)
which downscales the video grid to 720 video-pad tokens for memory control.

- **`img_feats` (Dv=3584):** 8 frames (passed as a video) + FIXED neutral instruction
  ("Describe the people, symbols, gestures, and on-screen text in this video.") → mean of the
  **last-layer** hidden states over the **vision+instruction span** = tokens `[0 : last <|im_start|>]`
  (excludes the empty assistant generation-prompt tail) → L2-norm.
- **`text_feats` (Dt=3584):** same frames + title + transcript + a FIXED analytic instruction →
  mean over the **assistant-header tail** span `[last <|im_start|> :]` (the causal-LM
  last-token sentence embedding, having attended over the full frames+title+transcript+
  instruction context; no text is ever generated) → L2-norm.
- Output = the SAME two-stream cache the loader consumes: `{ids:[[...]], img_feats[N,3584],
  text_feats[N,3584], labels[N]}`, filenames `{split}_Qwen2.5-VL-7B-Instruct_HF.pt`. Zero-vector
  guard writes `zeros(3584)` for unreadable videos.

**Correctness validation (before any batch job):**
- CPU processor-only check: `videos=[frames]` accepted under transformers 4.49; 8 frames →
  720 `<|video_pad|>` tokens + `pixel_values_videos` + `video_grid_thw`; span boundaries
  (`<|im_start|>` positions) confirmed; prefix span excludes / response span includes the
  assistant header exactly as intended.
- **External review (Codex, gpt-5.2):** no blocking correctness bugs — `hidden_states[-1][0]`
  drops batch correctly; under 4.49 the Qwen2.5-VL forward `masked_scatter`s video embeddings
  into the `<|video_pad|>` positions so **hidden-state length == input_ids length** (spans align);
  no bf16/OOM footgun for single-sample 80GB-A100 forwards. Added the suggested preflight
  `assert last_hidden.shape[0] == input_ids.numel()`.
- **GPU validation (SLURM job 12108, 16s):** 3 real HateMM videos → shapes `[3584]`, L2-norm
  =1.0000, no NaN, embeddings **distinct across videos** (img cos-sim off-diag 0.72–0.78,
  text 0.86–0.97 — not collapsed). PASSED.

### Extraction (SLURM GPU, frozen inference)

| Dataset | job id | N (train/dev/test) | zero-guards | elapsed | cache |
|---|---|---|---|---|---|
| HateMM | 12109 | 744 / 107 / 215 | 1 (train) | 24m32s | local + B2 |
| MHC (EN, leak-fixed 549) | 12110 | 549 / 80 / 161 | 0 | 31m48s | local + B2 |
| MHC_zh | 12116 | 579 / 78 / 149 | (see below) | — | local + B2 |
| ImpliHateVid | 12117 | 1283 / 325 / 401 | (see below) | — | local + B2 |

All caches verified `[N,3584]`, correct label balance, non-guard rows L2-normed. MHC's decord
failures (YouTube re-encodes) were all recovered by the PyAV fallback (0 guards).

### THE ABLATION — CLIP-RGCL vs MLLM-RGCL (identical head/loss/split, val-selected)

Same `classifier_hateClipper` head (align fusion, triplet+hybrid-BCE, topk=20, arithmetic
kNN vote), 30 epochs, `--Faiss_GPU False`, seed 0, single run. Epoch selected by **Val_Retrieval
acc (tie-break roc)** — the fixed criterion. Both encoders re-run through the *same* sbatch
harness (`scripts/slurm/train_mllm.sbatch`) on the *same local caches* for a clean head-to-head.
Test macro metrics at the val-selected epoch:

| Dataset | CLIP-RGCL (M-F1 / M-P / M-R / acc) | MLLM-RGCL Qwen2.5-VL (M-F1 / M-P / M-R / acc) | Δ M-F1 / Δ acc | vs acc>0.85 | vs MoRE (M-F1) |
|---|---|---|---|---|---|
| **HateMM** (EN) | 0.8172 / 0.8258 / 0.8120 / 0.8279 | **0.8606 / 0.8750 / 0.8527 / 0.8698** | **+0.0434 / +0.0419** | **MLLM crosses 0.85 (0.870)** | MLLM **+0.037 beats MoRE 0.8235** (approaches MM-HSD 0.874) |
| **MHC (EN)** | 0.7113 / 0.7586 / 0.6945 / 0.7826 | **0.7378 / 0.7540 / 0.7277 / 0.7888** | **+0.0265 / +0.0062** | neither (0.783→0.789) | MLLM 0.7378 vs MoRE 0.7519 (−0.014, still short) |

Job ids: CLIP HateMM 12114 (selEp24), CLIP MHC 12115 (selEp26), MLLM HateMM 12112 (selEp28),
MLLM MHC 12113 (selEp28). The HateMM CLIP re-run reproduced the Phase-0 baseline **exactly**
(0.8172/0.8279, selEp24). The MHC CLIP number here (0.7113/0.7826) is HIGHER than the Phase-0
logged 0.6219/0.7453 because Phase-0 used the *leaked* 550-row train (id `k9OtaMbK0Ac` in both
train and test) and landed a different unstable epoch (tiny 80-sample val); this re-run uses the
**leak-fixed 549-row train** and is the correct, fairer CLIP baseline. The MLLM-vs-CLIP delta is
therefore reported against this clean same-code baseline (not the stale leaked number).

### HONEST VERDICT

**Yes — frozen Qwen2.5-VL-7B hidden-state features beat frozen-CLIP features on the identical
RGCL head, on BOTH make-or-break datasets.** The win is unambiguous on HateMM (+0.043 M-F1,
+0.042 acc) and positive but smaller on MHC-EN (+0.027 M-F1, +0.006 acc).

- **HateMM crosses the acc>0.85 bar** for the first time (0.8698) and its M-F1 (0.8606) now
  **beats MoRE (0.8235)** and closes most of the gap to MM-HSD (0.874) — with a frozen encoder +
  a ~7M-param head, no LoRA, no generation.
- **MHC-EN improves substantially over its clean CLIP floor** (+0.027 M-F1) but does **not**
  reach acc>0.85 (0.789) and remains ~0.014 M-F1 below MoRE (0.7519). This is the harder
  hate/offensive-confusion slice; MLLM helps but does not solve it alone — segment retrieval
  (Phase 1/3) and the 3-class head are the levers still owed here. Reported plainly, not massaged.

The review's central risk ("MLLM may not beat CLIP") is **retired**: the encoder swap is a real,
measured, same-split win. Because MLLM clearly helps, extraction+ablation was extended to
**MHC_zh + ImpliHateVid** (jobs 12116 / 12117; results appended below when their training completes).

### Code additions
- `src/utils/generate_VideoMLLM_embedding_HF.py` (new extractor, Config MLLM-A).
- `scripts/slurm/gen_embed_mllm_validate.sbatch` (GPU validation harness).
- `scripts/slurm/gen_embed_mllm.sbatch` (extraction, all splits, B2 push).
- `scripts/slurm/train_mllm.sbatch` (RGCL head train + val-selected-epoch parser; works for
  both CLIP and MLLM caches via the `--model` tag).
- No change to `run_rac.py` / `classifier.py` / `loss.py` — the head reads Dv=Dt=3584 from the
  cache, so the 3584-d MLLM features required zero core-code change (as DESIGN §2.2 predicted).

### Install / disk
`qwen-vl-utils==0.0.14` installed (pip). `flash-attn` build FAILED → skipped (sdpa used; not
required). Qwen2.5-VL-7B (16G) downloaded to the login-node HF cache. Quota after download:
204G/290G. All caches pushed to B2 (`embeddings/<dataset>/`), local copies kept for training.

---

## 2026-07-02 — Reframe confirmed (user-approved)

The user CONFIRMED a reframe of the Iteration-1 method. This supersedes the 2026-07-01 "SR-RGCL-Vid"
framing whose headline novelty leaned on **gold segment spans**. `DESIGN_iter1.md`,
`ideas/rgcl-mllm-video-iter1.md`, and the query pack are updated to this confirmed framing.

**What the user confirmed.** The load-bearing novelty vs **plain RGCL** (the "current method" we
already run) = **TWO genuine mechanistic differences, BOTH annotation-free and general across ALL 4
datasets** (no gold labels), plus the MLLM as a validated performance LEVER (not the novelty).

**The two mechanistic deltas.**
1. **Multi-granularity, ANNOTATION-FREE temporal retrieval.** Plain RGCL retrieves/contrasts at a
   single WHOLE-INSTANCE granularity (whole video = one mean-pooled vector, one FAISS index). We add
   a FINE granularity: split each video into **AUTO sub-clips** (uniform temporal windows first;
   shot-boundary optional later — **NO gold spans**), embed each, build a **SECOND FAISS index**, and
   do retrieval-guided contrastive at BOTH whole-video AND sub-clip level. The within-video **benign
   sub-clip of a hateful video** is a retrieval-mined **drifting hard negative**, identified WITHOUT
   gold labels via a **MIL / dissimilarity heuristic**. General on HateMM / MHClip / ImpliHateVid via
   auto-segmentation; **HateClipSeg** gold spans (open, Social-AI-Studio github, 435 vids / 11,714
   segs) are a **VALIDATION slice only**. A meme has no time axis → **structurally meme-impossible**,
   so RA-HMD / RGCL cannot have it.
2. **Update-stable / cross-dataset kNN memory.** Because inference is a kNN vote over the labeled
   memory bank, the classifier can be updated by **ADDING new labeled exemplars at test time WITHOUT
   retraining** (evolving-hate adaptation), and the memory can be **SWAPPED** for cross-dataset
   transfer. **MoRE's trained MoE head cannot do either** (MoRE flags evolving hate as an unmet need).
   We make the embedding update-stable and demonstrate **test-time-update + cross-dataset transfer on
   all 4 datasets**.

**MLLM = performance LEVER, not novelty (VALIDATED).** RAMF / HVGuard already do frozen-VLM-features →
head, so the MLLM is not a field contribution. Used as a FROZEN Qwen2.5-VL-7B multimodal encoder
(hidden-state pooling; optional per-sub-clip neutral description as a fine key). On the identical RGCL
head it beats CLIP: **HateMM 0.817/0.828 (CLIP) → 0.861/0.870 (MLLM)** — crosses acc>0.85 and beats
MoRE 0.8235 (approaches MM-HSD 0.874); **MHC-EN 0.711/0.783 → 0.738/0.789** (modest, still <0.85).
Frozen, ~7M-param head, no LoRA / no generation. Difference vs RA-HMD (LoRA-SFT, memes) = frozen +
video + multi-granularity; vs reasoning-VLMs (MARS / HVGuard / IARE) = pure encoder + kNN over
exemplars (no verdict generation).

**What was DROPPED and why.**
- **Gold-span-segment DEPENDENCE** — a **single-dataset trap** the user rejected: only ONE dataset
  (HateClipSeg) has open, downloadable spans; HateMM spans are "collected but unused" (absent
  locally) and MHClip spans are gated/absent from our download (Phase-1 STEP-0 blocks). The
  multi-granularity delta is now delivered by **auto-segmentation** (no annotations), general on all 4.
- **Cross-lingual as a novelty axis.** Temporal / audio / implicit remain **evaluation slices**, not
  novelty.

**Baseline correction to record.** The MHClip-Y (MHC-EN) CLIP floor is **0.711 M-F1 / 0.783 acc** on
the **LEAK-FIXED 549-row train** — NOT the earlier-quoted **0.622 / 0.745**, which was a **leaked-550
unstable-epoch artifact** (id `k9OtaMbK0Ac` in both train and test; a different unstable epoch on the
tiny 80-sample val). All MHC-EN deltas are reported against the clean 549 floor.

**Protocol unchanged.** Lead BINARY harmful-vs-normal (offensive∪hateful=1), macro-F1/P/R + acc,
target **acc>0.85 on HateMM (MET with MLLM) + MHClip (still owed** — the multi-granularity +
updatable-memory levers target exactly this gap); 3-class MHClip honest secondary (beat 0.63 EN /
0.50 ZH, not 85%).

**Next step.** Implement the two deltas (DESIGN §5 Phase 3): (1) **auto-segmenter** (uniform windows
first) + sub-clip embedder + second FAISS index + additive `L_subclip_RGCL` with the annotation-free
drifting-negative miner; make-or-break ablation **whole-video vs +multi-granularity** (esp. the owed
MHC-EN gap); (2) **update-stable / cross-dataset memory demos** (test-time exemplar ADD; memory SWAP)
on all 4 datasets. HateClipSeg gold spans used only to validate the auto-mined negatives.

**Artifacts.** `research-wiki/DESIGN_iter1.md` (confirmed revision, dated 2026-07-02);
`ideas/rgcl-mllm-video-iter1.md` (retitled + new thesis/risks, gap G4 added); query pack rebuilt.

---

## Phase 2 FINAL — complete 4-dataset CLIP-vs-MLLM ablation (warmup-floored val-selected Test macro)

Selection rule: among epochs N>=5, pick epoch with max Val_Retrieval **acc** (tie-break: higher Val
**roc**); report that epoch's **Test** macro. ImpliHateVid MLLM = Qwen2.5-VL-7B-Instruct trainlog
`1848776` selected **epoch 7** (Val acc 0.9046 / roc 0.9606, uniquely highest for N>=5). ImpliHateVid
CLIP = trainlog `1228498` selected **epoch 18** (Val acc 0.9138), Test 0.9101/0.9122/0.9103/0.9102.

| Dataset | Encoder | Test macroF1 | macroP | macroR | acc | MLLM−CLIP ΔF1 | acc≥0.85 |
|---|---|---|---|---|---|---|---|
| HateMM | CLIP | 0.8172 | 0.8258 | 0.8120 | 0.8279 | — | no |
| HateMM | MLLM | 0.8606 | 0.8750 | 0.8527 | 0.8698 | +0.0434 | yes |
| MHC-EN | CLIP | 0.7113 | 0.7586 | 0.6945 | 0.7826 | — | no |
| MHC-EN | MLLM | 0.7378 | 0.7540 | 0.7277 | 0.7888 | +0.0265 | no |
| MHC_zh | CLIP | 0.7706 | 0.7690 | 0.7723 | 0.8054 | — | no |
| MHC_zh | MLLM | 0.7412 | 0.7565 | 0.7312 | 0.7919 | −0.0294 | no |
| ImpliHateVid | CLIP | 0.910 | 0.912 | 0.910 | 0.910 | — | yes |
| ImpliHateVid | MLLM | 0.9002 | 0.9009 | 0.9003 | 0.9002 | −0.0098 | yes |

**Honest note.** MLLM beats CLIP on HateMM (clear, crosses 0.85, beats MoRE 0.8235) + MHC-EN
(modest); loses on MHC_zh; ≈ on ImpliHateVid. acc≥0.85 currently met on HateMM(MLLM) +
ImpliHateVid(both); MHClip EN & ZH remain the open gap for the multi-granularity + updatable-memory
novelty. Reference SOTA: HateMM MoRE 0.8235 / MM-HSD 0.874; MHClip-Y MoRE 0.7519; MHClip-B MoRE
0.7475; ImpliHateVid TCL 0.8773.

---

## Phase 3 (iter1) — multi-granularity ablation on CLIP (make-or-break)

**Setup.** 4 COMPLETED SLURM jobs, CLIP backbone, whole-video baseline (`LAMBDA_SEG=0`) vs
multi-granularity (`LAMBDA_SEG=0.5`), on MHC (EN) and MHC_zh. Selection = **warmup-floored
val-selected**: among epochs N>=5, pick max `Val_Retrieval acc` (tie-break higher `Val_Retrieval
roc`); report that epoch's **Test** macro metrics. Config→log mapping verified by grepping
`DATASET`/`GROUP_NAME`/`LAMBDA_SEG` inside each `.out` (all 4 have 30 epochs of Val+Test lines).

| Dataset | config | job | sel-epoch | Test macroF1 | macroP | macroR | acc | Δ vs λ=0 (F1 / acc) |
|---|---|---|---|---|---|---|---|---|
| MHC (EN) | λ=0 (whole-video) | 12128 | 26 | 0.7113 | 0.7586 | 0.6945 | 0.7826 | — |
| MHC (EN) | λ=0.5 (multi-granularity) | 12129 | 25 | 0.7262 | 0.7619 | 0.7105 | 0.7888 | **+0.0149 / +0.0062** |
| MHC_zh | λ=0 (whole-video) | 12130 | 29 | 0.7706 | 0.7690 | 0.7723 | 0.8054 | — |
| MHC_zh | λ=0.5 (multi-granularity) | 12131 | 21 | 0.7050 | 0.6988 | 0.7179 | 0.7383 | **-0.0656 / -0.0671** |

**Exact selected log lines (for verification):**

```
# 12128  MHC EN  λ=0   (sel epoch 26)
Val_Retrieval Epoch  26 macroF1: 0.6865 macroP: 0.7092 macroR: 0.6764 acc: 0.7500 roc: 0.7833
Test_Retrieval Epoch 26 macroF1: 0.7113 macroP: 0.7586 macroR: 0.6945 acc: 0.7826 roc: 0.8422

# 12129  MHC EN  λ=0.5 (sel epoch 25)
Val_Retrieval Epoch  25 macroF1: 0.6949 macroP: 0.7078 macroR: 0.6873 acc: 0.7500 roc: 0.7796
Test_Retrieval Epoch 25 macroF1: 0.7262 macroP: 0.7619 macroR: 0.7105 acc: 0.7888 roc: 0.8274

# 12130  MHC_zh  λ=0   (sel epoch 29)
Val_Retrieval Epoch  29 macroF1: 0.7857 macroP: 0.7951 macroR: 0.7793 acc: 0.8077 roc: 0.8329
Test_Retrieval Epoch 29 macroF1: 0.7706 macroP: 0.7690 macroR: 0.7723 acc: 0.8054 roc: 0.8382

# 12131  MHC_zh  λ=0.5 (sel epoch 21)
Val_Retrieval Epoch  21 macroF1: 0.7734 macroP: 0.7788 macroR: 0.7693 acc: 0.7949 roc: 0.8407
Test_Retrieval Epoch 21 macroF1: 0.7050 macroP: 0.6988 macroR: 0.7179 acc: 0.7383 roc: 0.7994
```

Tie-break note: on both MHC-EN runs, 6 epochs tied at Val acc=0.7500; the roc tie-break resolved
12128→ep26 (roc 0.7833) and 12129→ep25 (roc 0.7796). MHC_zh runs had unique max-acc epochs.

**λ=0 sanity check vs known CLIP baselines.** PASS.
- MHC-EN λ=0: 0.7113 F1 / 0.7826 acc vs known ~0.711 F1 / 0.783 acc — matches essentially exactly.
- MHC_zh λ=0: 0.7706 F1 / 0.8054 acc vs known ~0.771 F1 / 0.805 acc — matches essentially exactly.
Confirms λ=0 is a genuine no-op == plain whole-video RGCL.

**VERDICT (honest, not massaged).**
- **MHC (EN):** multi-granularity helps, but marginally — **ΔmacroF1 +0.0149, Δacc +0.0062**.
- **MHC_zh:** multi-granularity **hurts, substantially** — **ΔmacroF1 -0.0656, Δacc -0.0671**. This
  is not noise; it is a large regression that wipes out the EN gain and then some.
- **acc≥0.85:** **NO** λ=0.5 run crosses 0.85 (best λ=0.5 is MHC-EN at 0.7888; MHC_zh λ=0.5 is 0.7383).
- **Net:** the multi-granularity term is **NOT a robust win** — inconsistent across languages
  (tiny help on EN, big hurt on ZH), and neither run reaches the 0.85 bar. As a make-or-break test,
  this is **fail / not-yet**: λ=0.5 as configured does not deliver a reliable improvement over the
  whole-video baseline.

---

### Phase 3 (iter2) — seg_mode ablation: driftneg vs milmax (λ=0.5) — does it fix the ZH regression?

**Setup.** 4 more COMPLETED SLURM jobs, same CLIP backbone / same warmup-floored val-selection
(epochs N>=5, max `Val_Retrieval acc`, tie-break higher `Val_Retrieval roc` → report that epoch's
**Test** macro metrics). Goal: two alternative segment-aggregation objectives — `driftneg` (drift
negative) and `milmax` (MIL max-pooling) — both at λ=0.5, on MHC (EN) and MHC_zh, to see whether
either FIXES the "full" (λ=0.5, default seg) MHC_zh regression and/or gives a *consistent*
(both EN and ZH ≥ baseline) improvement over the λ=0 whole-video baseline. Config→log mapping
verified by grepping `DATASET`/`group_name`/`seg_mode`/`lambda_seg` inside each `.out`
(12132 MHC/driftneg, 12133 MHC_zh/driftneg, 12134 MHC/milmax, 12135 MHC_zh/milmax; all 30 epochs).

Baselines carried from iter1: MHC-EN λ=0 = **0.7113 F1 / 0.7826 acc**; MHC_zh λ=0 = **0.7706 F1 / 0.8054 acc**.

| Dataset | config | job | sel-epoch | Test macroF1 | acc | Δ vs baseline (F1 / acc) |
|---|---|---|---|---|---|---|
| MHC (EN) | λ=0 baseline | 12128 | 26 | 0.7113 | 0.7826 | — |
| MHC (EN) | full (λ=0.5) | 12129 | 25 | 0.7262 | 0.7888 | +0.015 / +0.006 |
| MHC (EN) | driftneg (λ=0.5) | 12132 | 23 | 0.7159 | 0.7826 | **+0.0046 / +0.0000** |
| MHC (EN) | milmax (λ=0.5) | 12134 | 19 | 0.6089 | 0.7205 | **-0.1024 / -0.0621** |
| MHC_zh | λ=0 baseline | 12130 | 29 | 0.7706 | 0.8054 | — |
| MHC_zh | full (λ=0.5) | 12131 | 21 | 0.7050 | 0.7383 | -0.066 / -0.067 |
| MHC_zh | driftneg (λ=0.5) | 12133 | 16 | 0.7357 | 0.7785 | **-0.0349 / -0.0269** |
| MHC_zh | milmax (λ=0.5) | 12135 | 28 | 0.7875 | 0.8255 | **+0.0169 / +0.0201** |

**Exact selected log lines (for verification):**

```
# 12132  MHC EN    driftneg λ=0.5 (sel epoch 23)
Val_Retrieval Epoch  23 macroF1: 0.7063 macroP: 0.7250 macroR: 0.6964 acc: 0.7625 roc: 0.7804
Test_Retrieval Epoch 23 macroF1: 0.7159 macroP: 0.7545 macroR: 0.7003 acc: 0.7826 roc: 0.8211

# 12133  MHC_zh    driftneg λ=0.5 (sel epoch 16)
Val_Retrieval Epoch  16 macroF1: 0.7894 macroP: 0.7919 macroR: 0.7871 acc: 0.8077 roc: 0.8529
Test_Retrieval Epoch 16 macroF1: 0.7357 macroP: 0.7372 macroR: 0.7342 acc: 0.7785 roc: 0.8295

# 12134  MHC EN    milmax   λ=0.5 (sel epoch 19)
Val_Retrieval Epoch  19 macroF1: 0.6979 macroP: 0.7285 macroR: 0.6855 acc: 0.7625 roc: 0.7476
Test_Retrieval Epoch 19 macroF1: 0.6089 macroP: 0.6625 macroR: 0.6040 acc: 0.7205 roc: 0.7843

# 12135  MHC_zh    milmax   λ=0.5 (sel epoch 28)
Val_Retrieval Epoch  28 macroF1: 0.7658 macroP: 0.8346 macroR: 0.7479 acc: 0.8077 roc: 0.8607
Test_Retrieval Epoch 28 macroF1: 0.7875 macroP: 0.7964 macroR: 0.7804 acc: 0.8255 roc: 0.8472
```

Tie-break notes: 12132 (MHC/driftneg) tied Val acc=0.7625 at ep17 (roc 0.7695) and ep23 (roc 0.7804)
→ roc tie-break picks **ep23**. 12135 (MHC_zh/milmax) tied Val acc=0.8077 at ep28 (roc 0.8607) and
ep29 (roc 0.8493) → roc tie-break picks **ep28**. 12133 (MHC_zh/driftneg, ep16) and 12134
(MHC/milmax, ep19) had unique max-acc epochs.

**VERDICT (honest, not massaged). Dev sets are tiny (MHC dev=80, MHC_zh dev=78); treat sub-0.01 diffs as noise.**

- **Does driftneg FIX the MHC_zh regression?** *Partially, not fully.* driftneg lifts MHC_zh from
  full's 0.7050 F1 / 0.7383 acc back up to **0.7357 F1 / 0.7785 acc** — most of the "full" damage is
  undone — but it still sits **below the λ=0 baseline (-0.0349 F1 / -0.0269 acc)**. So driftneg
  *reduces* the regression but does **not** cure it: MHC_zh is still net-negative vs whole-video.
- **Does milmax FIX the MHC_zh regression?** *Yes — milmax is the only λ=0.5 variant that beats the
  MHC_zh baseline.* MHC_zh/milmax = **0.7875 F1 / 0.8255 acc**, i.e. **+0.0169 F1 / +0.0201 acc**
  over the λ=0 baseline and a large recovery from full's -0.066/-0.067. The +0.02 acc gain exceeds
  the noise floor; the +0.017 F1 is right at the edge of it.
- **But milmax breaks EN.** MHC-EN/milmax collapses to **0.6089 F1 / 0.7205 acc** (**-0.1024 F1 /
  -0.0621 acc** vs baseline) — the worst EN result of any config. So milmax's ZH win comes with a
  catastrophic EN loss.
- **Consistency (both EN and ZH ≥ baseline)?** *No config achieves it.*
  - full: EN +, ZH −− (fails on ZH).
  - driftneg: EN ~0 (+0.0046 F1 / +0.0000 acc, i.e. essentially baseline-tied), ZH − (below baseline). Fails on ZH.
  - milmax: EN −− (large loss), ZH + . Fails on EN.
  The sign of the segment-aggregation effect **flips with language**: whatever helps ZH hurts EN and
  vice-versa. There is **no single seg_mode that is ≥ baseline on both**.
- **Best seg_mode per dataset:**
  - **MHC (EN):** best is **full (λ=0.5)** at 0.7262 F1 / 0.7888 acc; driftneg is a near-baseline
    no-op; milmax is a disaster. (Ordering EN: full > driftneg ≈ baseline ≫ milmax.)
  - **MHC_zh:** best is **milmax (λ=0.5)** at 0.7875 F1 / 0.8255 acc — the only ZH config above
    baseline; driftneg partially recovers but stays below baseline; full is worst.
    (Ordering ZH: milmax > baseline > driftneg > full.)
- **acc≥0.85:** still **NO** — best of all eight configs is MHC_zh/milmax at 0.8255.
- **Bottom line:** iter2 does not produce a robust, language-consistent improvement. milmax rescues
  (and slightly improves) ZH but destroys EN; driftneg is a safe near-no-op on EN but leaves ZH
  below baseline. The segment objective is **dataset/language-specific, not a general win** — same
  fail/not-yet conclusion as iter1, now with the added evidence that the optimal seg_mode is
  opposite for the two languages.

---

## Phase 3b — cross-dataset kNN memory transfer — 2026-07-02

**Novelty delta validated:** RGCL inference is a kNN vote over the TRAIN-set memory bank in the
*learned fused-embedding space*. So a head trained on dataset A can classify a *different* dataset
T's test set by **SWAPPING the memory bank** to T's own train(+val) — no retraining. All video
datasets share the SAME frozen encoder, so once a trained head projects their pooled features every
dataset's fused embeddings live in one comparable space. **This is a capability MoRE structurally
lacks:** MoRE's decision is baked into a trained MoE head and cannot be re-pointed at a new support
set at inference time.

### Code added
- `src/eval_cross_dataset.py` — standalone cross-dataset evaluator. Loads a trained head ckpt
  (trained on A), builds the faiss kNN memory from `--memory_dataset M` train(+val) fused embeddings,
  queries with `--eval_dataset T` test fused embeddings, scores with the existing
  `compute_metrics_retrieval` (topk=20, arithmetic vote). Leak-safe by construction: memory =
  M-train(+val), never M-test or T-test; cross-dataset ids cannot overlap. Reproduces the in-domain
  eval exactly (HateMM Qwen in-domain = **0.8606 M-F1 / 0.8698 acc**, matching the Phase-3 MLLM
  number, and CLIP in-domain diagonal reproduces the Phase-0 baselines exactly — pipeline validated).
  Does NOT touch the training loop. Supports `--use_sim` (per-epoch run_rac signed sim-weighted vote)
  vs plain arithmetic label vote, and `--include_val`.
- `scripts/slurm/train_clip_heads.sbatch` — trains the 4 frozen-CLIP-RGCL heads to a DISTINCT group
  dir `RAC_video_CLIP` (so they do not collide with / get overwritten by the Qwen `RAC_video` heads).
  Config identical to Phase-0 (align fusion, triplet+hybrid BCE, topk=20, 30ep, seed0, faiss CPU).
- `scripts/slurm/eval_cross_matrix.sbatch` — runs the CLIP transfer matrix from those heads.

**Discovery:** no CLIP-RGCL ckpts were on disk — the Phase-0 CLIP heads (jobs 12103-6) were
overwritten by the later Phase-3 Qwen runs (same group/exp path, `--force True`). Retrained fresh
CLIP heads (job 12136); the Qwen heads were reused as a bonus 2nd feature set.

### PRIMARY result — CLIP-RGCL transfer matrix (prompt-requested config)

Frozen CLIP ViT-L/14-336, align fusion, warmup-consistent val-selected heads (HateMM ep24, MHC ep26,
MHC_zh ep29, ImpliHateVid ep18). Memory = T-train+val; query = T-test; topk=20 arithmetic vote
(`use_sim=False`). Cell = **macro-F1 / acc**. Diagonal (bold) = in-domain reference; it reproduces
the Phase-0 CLIP baselines exactly. `maj` = majority-class acc baseline of T.

| trained-on A ↓ \ memory=test=T → | HateMM (maj .600) | MHC (maj .696) | MHC_zh (maj .698) | ImpliHateVid (maj .501) |
|---|---|---|---|---|
| **HateMM**       | **0.820 / 0.828** | 0.548 / 0.696 | — | 0.855 / 0.855 |
| **MHC**          | 0.751 / 0.763 | **0.711 / 0.783** | 0.633 / 0.758 | 0.835 / 0.835 |
| **MHC_zh**       | — | 0.645 / 0.739 | **0.771 / 0.805** | — |
| **ImpliHateVid** | 0.677 / 0.698 | 0.635 / 0.702 | — | **0.920 / 0.920** |

(— = cell not run. In-domain `use_sim=True` diagonal is identical on F1/acc: HateMM 0.820/0.828,
MHC 0.711/0.783, MHC_zh 0.771/0.805, ImpliHateVid 0.920/0.920.)

### BONUS result — Qwen2.5-VL-7B-RGCL transfer matrix (2nd frozen encoder)

Same protocol on the Phase-3 MLLM heads (3584-dim, align). Diagonal `use_sim=False`. Caveat: the
MHC_zh Qwen head is an **epoch-0** ckpt (only one saved; NOT warmup-consistent) — treat MHC_zh Qwen
rows as provisional. Cell = **macro-F1 / acc**.

| trained-on A ↓ \ memory=test=T → | HateMM | MHC | MHC_zh* | ImpliHateVid |
|---|---|---|---|---|
| **HateMM**       | **0.851 / 0.860** | 0.639 / 0.733 | 0.619 / 0.732 | 0.860 / 0.860 |
| **MHC**          | 0.810 / 0.814 | **0.753 / 0.801** | 0.707 / 0.752 | 0.863 / 0.863 |
| **MHC_zh***      | 0.768 / 0.772 | 0.594 / 0.671 | **0.689 / 0.732*** | — |
| **ImpliHateVid** | 0.850 / 0.856 | 0.682 / 0.745 | — | **0.900 / 0.900** |

### VERDICT (honest)

**Does the learned retrieval space transfer across datasets? PARTIALLY — YES on the EN video
datasets, clearly NOT on the hardest EN meme-style dataset as a target.**

1. **Transfer is real and often strong when the TARGET is an easy/mid dataset.** Swapping in
   HateMM's or ImpliHateVid's memory under a foreign head stays close to in-domain:
   - CLIP `MHC→HateMM` = **0.751 / 0.763** vs in-domain 0.820/0.828 — lags in-domain by only
     ~0.07 M-F1 / ~0.065 acc, and sits **+0.15 acc above HateMM's majority baseline (.600)**.
   - CLIP `HateMM→ImpliHateVid` = **0.855 / 0.855** and `MHC→ImpliHateVid` = **0.835 / 0.835** vs
     in-domain 0.920 — lag ~0.065–0.085 M-F1, and **+0.35 acc above ImpliHateVid's .501 majority**.
     A HateMM- or MHC-trained head with ImpliHateVid's memory is *far* above chance.
   - Qwen transfer is even more robust: `MHC→HateMM` **0.810/0.814** (lag only 0.04/0.046),
     `ImpliHateVid→HateMM` **0.850/0.856** (lag ~0.001!), `HateMM→ImpliHateVid` **0.860/0.860**.
   So the fused space genuinely carries a transferable hateful-vs-benign geometry — a foreign head +
   local memory beats majority by a wide margin. **This is the capability demo: MoRE cannot do this.**

2. **Transfer COLLAPSES when the target is MHC (EN MultiHateClip).** CLIP `HateMM→MHC` =
   **0.548 / 0.696** — acc *equals* MHC's majority baseline (.6957) and M-F1 is barely above the
   all-negative floor; `ImpliHateVid→MHC` **0.635/0.702** and `MHC_zh→MHC` **0.645/0.739** are also
   near-majority. MHC is the weakest dataset even in-domain (0.711 CLIP), and its hate/offensive
   boundary does not survive a foreign projection — a foreign head collapses MHC toward all-negative.

3. **By how much does cross lag in-domain?** On the working cells, **~0.04–0.09 macro-F1** (Qwen
   tighter than CLIP; `ImpliHateVid→HateMM` Qwen is essentially in-domain-tied). On the failing
   target (MHC), cross **drops to the majority baseline** (~0.15–0.20 M-F1 below in-domain).

4. **Is EN↔ZH memory transfer meaningful? WEAKLY.** CLIP `MHC(EN)→MHC_zh(ZH)` = **0.633 / 0.758**
   (in-domain ZH 0.771/0.805) and `MHC_zh(ZH)→MHC(EN)` = **0.645 / 0.739** (in-domain EN 0.711/0.783).
   Both stay **above the respective majority baselines** (ZH .698, EN .696) — i.e. the EN-trained head
   with ZH memory, and vice-versa, still beat chance — but each lags its in-domain by ~0.07–0.14 M-F1
   and the EN→ZH acc (0.758) only modestly clears the ZH majority (.698). Qwen EN→ZH is similar
   (0.707/0.752). So **cross-lingual retrieval-memory transfer is above-chance but clearly degraded**;
   it is a real signal, not a strong one — consistent with the documented EN-CLIP-text-on-Chinese
   handicap. Note the ZH-head direction is the more reliable one to cite (MHC_zh Qwen head is epoch-0;
   the CLIP MHC_zh head is warmup-consistent, so the CLIP EN↔ZH numbers are the trustworthy pair).

**Bottom line:** the learned kNN-memory space is **swappable and above-majority-baseline on 5 of the
6 informative cross cells** (all except any-→MHC), lagging in-domain by a modest ~0.04–0.09 M-F1 on
the working cells and degrading gracefully (not catastrophically) cross-lingually. This is a genuine
**capability** MoRE lacks — but it is a capability demo, not a performance win: cross never beats
in-domain, and it fails on the hardest target (MHC). Reported honestly, transfer is *meaningful but
partial*.

### SLURM jobs
| JobID | Script | State | Elapsed |
|---|---|---|---|
| 12136 | train_clip_heads.sbatch (4 CLIP heads → RAC_video_CLIP) | COMPLETED | 00:02:26 |
| 12137 | eval_cross_matrix.sbatch (CLIP transfer matrix) | COMPLETED | 00:01:55 |

CLIP heads: `logging/Retrieval/<DS>/RAC_video_CLIP/..._hybrid_loss/ckpt/best_model_<selEp>_<valAcc>.pt`
(HateMM ep24/0.8411, MHC ep26/0.75, MHC_zh ep29/0.8077, ImpliHateVid ep18/0.9138). Qwen matrix ran
on CPU (no SLURM) from the existing `RAC_video` heads. Log: `slurm/logs/cross_matrix_12137.out`.

---

## STATUS — LoRA-SFT of the Qwen2.5-VL encoder (LAUNCHED / IN PROGRESS) — 2026-07-02

**No results yet — this is a status note, not a results entry. NO LoRA-adapted RGCL metrics exist
anywhere on disk (no `*LoRA*` trainlog, no `Qwen2.5-VL-7B-Instruct-LoRA` embedding cache, only
`data/CLIP_Embedding` present). Nothing recordable; recording state only, no numbers.**

**Why.** LoRA-SFT was the Iteration-2 lever deferred at Iteration 1 (task-adapt the Qwen2.5-VL
ENCODER via LoRA; final predictions still come from the RGCL retrieval-contrastive + kNN head, so
we stay distinct from generative reasoning-VLMs MARS/HVGuard). It targets the still-open MHClip
EN/ZH gap (both encoders < 0.85, near the field ceiling).

**Jobs (SLURM).**

| JobID | Dataset | State | Elapsed | Note |
|---|---|---|---|---|
| 12138 | ? | FAILED | 00:00:01 | instant fail (bad earlier relaunch) |
| 12139 | ? | FAILED | 00:00:01 | instant fail (bad earlier relaunch) |
| 12140 | ? | FAILED | 00:07:50 | earlier fail (~8min; omegaconf `ModuleNotFoundError` class of dep issue) |
| 12141 | ? | FAILED | 00:09:32 | earlier fail (~9.5min) |
| 12142 | MHC (EN) | RUNNING | 01:53:02 | step 155/204 (~76%), epoch 2.25, eval_loss 0.140 @ epoch2; INTERMEDIATE adapter at `checkpoint-138`, **no final top-level adapter yet** |
| 12143 | MHC_zh (ZH) | RUNNING | 00:21:15 | step 15/216 (~7%), just started, no checkpoint/adapter yet |

Two RUNNING (12142 MHC, 12143 MHC_zh) on foscsmlprd01; four earlier relaunch jobs FAILED. Deps
were fixed surgically after the omegaconf failures and the jobs relaunched.

**Adapters on disk.** `logging/lora/MHC/checkpoint-138/adapter_model.safetensors` (intermediate,
run not finished). NO final `logging/lora/MHC/adapter_model.safetensors`; NO `logging/lora/MHC_zh/`
adapter yet.

**Background pipeline** (`bvtfpmh52`) is alive but BLOCKED at Stage 1, waiting on both SFT jobs;
the downstream chain (adapter → LoRA embeddings → RGCL head train → eval) has NOT started.

**Disk watch.** `/data` at 96% (659G free) — must be watched before generating LoRA embeddings.

**Next milestone.** MHC SFT (12142) completes (~within the hour) → final adapter; MHC_zh (12143)
finishes ~2h later. ONLY THEN can the LoRA-RGCL matrix run and produce recordable Test metrics,
which will be appended as a proper results entry (against the Phase-2-FINAL frozen-Qwen and
Phase-0 CLIP floors). Last recorded results iter = the Phase-3b CLIP/Qwen cross-transfer matrix
(jobs 12136/12137).

---

## RESULTS — LoRA-SFT Qwen2.5-VL-7B encoder + RGCL/kNN (MHClip EN & ZH) — 2026-07-02

**This RESOLVES the STATUS-only LoRA note above** (the "LAUNCHED / IN PROGRESS" entry that read
*"NO LoRA-adapted RGCL metrics exist anywhere on disk"*, line 780). Both SFT jobs finished, the
adapter → LoRA-embedding → RGCL-head → eval chain ran end-to-end, and these are the **FIRST
LoRA-adapted RGCL metrics on disk** — they supersede that status. Note preserved above (append-only);
this section carries the numbers.

**Pipeline (SLURM).** SFT jobs 12142 (MHC/EN) + 12143 (MHC_zh/ZH) → LoRA-embedding extract 12146
(EN) + 12148 (ZH) → RGCL head train/eval 12147 (EN) + 12149 (ZH).

**Selection rule (identical to all prior iters).** Warmup-floored (epochs N≥5), val-selected on the
`Test_Retrieval` (kNN) head: max `Val_Retrieval` **acc**, tie-break higher `Val_Retrieval` **roc**;
report that epoch's **Test_Retrieval** macro metrics. Same rule as the CLIP / frozen-Qwen floors, so
these are directly comparable.

| Dataset | sel-ep | Val acc | Test macroF1 | macroP | macroR | acc | roc | acc≥0.85 | gap→0.85 |
|---|---|---|---|---|---|---|---|---|---|
| MHC (EN) LoRA-Qwen | 26 | 0.75 | **0.6916** | 0.7049 | 0.6837 | **0.7516** | 0.8488 | **no** | 0.0984 |
| MHC_zh (ZH) LoRA-Qwen | 20 | 0.8718 | **0.8023** | 0.8004 | 0.8042 | **0.8322** | 0.8825 | **no** | 0.0178 |

Trainlogs: `slurm/logs/rgcl_MHC_Qwen2.5-VL-7B-Instruct-LoRA_HF_2723309.trainlog` (EN),
`slurm/logs/rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_2794237.trainlog` (ZH).

**Selected-epoch tie-breaks (for verification).**
- **MHC (EN)** — max eligible `Val_Retrieval acc` = 0.7500, three-way tie at ep24/26/29; roc tie-break
  ep24=0.8335, ep26=**0.8531**, ep29=0.8305 → **epoch 26**. Test kNN: acc 0.7516 / M-F1 0.6916 /
  macroP 0.7049 / macroR 0.6837 / roc 0.8488 (binary pre 0.6098 / rec 0.5102 / f1 0.5556).
- **MHC_zh (ZH)** — max eligible `Val_Retrieval acc` = 0.8718; ep1 & ep4 also hit 0.8718 but are
  **below the warmup floor (N<5) and excluded**; eligible tie is ep9/ep20; roc tie-break ep9=0.9207,
  ep20=**0.9229** → **epoch 20**. Test kNN: acc 0.8322 / M-F1 0.8023 / macroP 0.8004 / macroR 0.8042 /
  roc 0.8825 (binary pre 0.7174 / rec 0.7333 / f1 0.7253).

### Deltas vs frozen floors (LoRA − frozen; +ve = LoRA better)

Frozen floors are warmup-consistent val-selected `Test_Retrieval` (same rule); sources: Phase-2-FINAL
table (lines 508–515), Phase-3 λ=0 (lines 535, 537), Phase-3b diagonals (lines 702–703). Frozen CLIP
= ViT-L/14-336; frozen Qwen = Qwen2.5-VL-7B non-LoRA.

| Dataset | LoRA (M-F1 / acc) | vs frozen CLIP (M-F1 / acc) | ΔM-F1 / Δacc | vs frozen Qwen (M-F1 / acc) | ΔM-F1 / Δacc |
|---|---|---|---|---|---|
| MHC (EN) | 0.6916 / 0.7516 | 0.7113 / 0.7826 | **−0.0197 / −0.0310** | 0.7378 / 0.7888 | **−0.0462 / −0.0372** |
| MHC_zh (ZH) | 0.8023 / 0.8322 | 0.7706 / 0.8054 | **+0.0317 / +0.0268** | 0.7412 / 0.7919† | **+0.0611 / +0.0403†** |

- **EN:** LoRA-SFT of the encoder **regressed below BOTH frozen encoders** under identical head/
  selection — worse than frozen CLIP (−0.031 acc) and worse than frozen Qwen (−0.037 acc). On EN,
  LoRA did **not** help; it moved EN *further* from 0.85, not closer.
- **ZH:** LoRA **beats the frozen-CLIP floor** on both M-F1 and acc (+0.032 / +0.027) — a **clean
  apples-to-apples win** (both warmup-consistent). This is the **best ZH number recorded**
  (0.8023 M-F1 / 0.8322 acc), above every prior frozen-CLIP / frozen-Qwen / seg-mode ZH config
  (prior ZH best was milmax 0.7875 / 0.8255).
- **† Caveat (ZH vs frozen Qwen).** The frozen-Qwen MHC_zh head is an **epoch-0 checkpoint** (only
  one saved; Phase-3b lines 712–713, 719 flag it *"NOT warmup-consistent"* / provisional), so it
  **violates the warmup≥5 floor**. The +0.061 / +0.040 magnitude is therefore ambiguous; only the
  *direction* (LoRA better) is reliable, and it agrees with the clean CLIP comparison. (Frozen-Qwen
  floors for HateMM / MHC-EN / ImpliHateVid ARE warmup-consistent; MHC_zh is the sole epoch-0
  exception.)

### Goal verdict — cross acc ≥ 0.85 on MHClip

**VERDICT: LoRA does NOT cross 0.85 on either MHClip split.**

- **EN (MHC):** acc 0.7516, **gap 0.0984** — does NOT cross, and LoRA *regressed* below both frozen
  floors (CLIP 0.7826, Qwen 0.7888), so LoRA moved EN backward. EN remains the hardest, unsolved
  slice (hate/offensive confusion, tiny 161-sample test, near field ceiling).
- **ZH (MHC_zh):** acc 0.8322, **gap 0.0178** — does NOT cross, but this is the **closest any config
  has come on ZH** and the only recorded improvement over the frozen-CLIP floor (+0.027 acc). A
  genuine apples-to-apples ZH win, still ~1.8 acc points short of 0.85.

**Overall.** acc≥0.85 is met ONLY on HateMM (frozen Qwen, 0.870) and ImpliHateVid (both encoders,
~0.90). Neither MHClip EN nor ZH crosses 0.85 with LoRA. LoRA **helps ZH (best-ever) but hurts EN** —
a language-inconsistent result echoing the earlier seg-mode finding that EN and ZH respond oppositely.
**The MHClip 0.85 target stays OPEN on both languages after LoRA-SFT.**

### SLURM jobs
| JobID | Stage | Dataset | Note |
|---|---|---|---|
| 12142 | SFT (LoRA adapter) | MHC (EN) | completed → adapter |
| 12143 | SFT (LoRA adapter) | MHC_zh (ZH) | completed → adapter |
| 12146 | LoRA-embedding extract | MHC (EN) | → LoRA embeddings |
| 12148 | LoRA-embedding extract | MHC_zh (ZH) | → LoRA embeddings |
| 12147 | RGCL head train + eval | MHC (EN) | → trainlog 2723309 |
| 12149 | RGCL head train + eval | MHC_zh (ZH) | → trainlog 2794237 |

Key paths: `logging/lora/MHC/checkpoint-138/`; `logging/lora/{MHC,MHC_zh}/trainer_log.jsonl`.

---

## Iteration 3 — 2026-07-02 — Design: self-maintaining (auditable) hate memory (DESIGN_iter3)

**Goal.** 在 iter1 已验证骨架(retrieval-guided contrastive + updatable kNN memory = headline;
multi-granularity = honest negative;MLLM/LoRA = lever)之上,基于三份侦察报告
(`NOVELTY_CHECK_dirA.md`、`TEMPORAL_SPLIT_FEASIBILITY.md`、`MLLM_USAGE_LANDSCAPE.md`)起草
Iteration-3 设计:一个统一故事 + 两个新方法 + MLLM 三角色。

**Inputs(三报告一句话)。** 方向 A 查新:机制层面 SAFE(retrieval-as-annotator 领域内零重合,
置信 ~75–80%),任务层面 THREATENED(MultiHateLoc/LELA/TANDEM 已分占任务话语,Exeter 组速度
风险真实);时间戳可行性:MHClip EN/ZH HIGH(存活 ~83%/~77%,死链偏 Hateful——ZH 探针 ~60%),
HateMM/ImpliHateVid LOW(官方匿名化,~0%);MLLM 用法对照:角色1(schema 条目入库作检索键)
OPEN,角色2(片段级分歧裁决)PARTIAL-窄口径,角色3(kNN 置信门控唤醒)领域内 OPEN。

**Decision(设计,非结果)。** 总故事 = **self-maintaining (auditable) hate memory**:检测由
检索记忆的 kNN vote 完成;MLLM 为记忆打工(从不判案);记忆自己洗片段标签(**方法 A:
retrieval-consensus segment denoising** —— sub-clip 伪标签 = agreement(自身 video 标签 ×
kNN 邻居 video 标签投票),高置信片段训对比 embedding,EM 2–3 轮,保留 drifting hard-negative;
定生死消融 = consensus vs 自打分(MIST/C2FPL 式)vs 继承标签(Phase-3 复现),gate = 双语同向
≥ baseline;评测 = video-level 主表 + HateClipSeg 弱监督定位空赛道),自己随仇恨演化更新
(**方法 B:evolving-memory 协议** —— MHClip EN/ZH temporal split(不可定年样本固定进 train,
报告 survivor bias),阶段性增补 k 新期样本零重训,指标 = 适应速度/保持性/更新成本,维护策略
消融 = 加什么(不确定性主动选择)/怎么权(时间加权)/删什么(预算+去重)vs 傻塞;HateMM/
ImpliHateVid 保持静态 cross-dataset 矩阵)。MLLM 三角色:角色1 结构化档案(target/机制/载体/
显隐性)离线入库作检索键(ZH 英文枢轴);角色2 仅共识分歧时裁决片段,调用率作指标;角色3
缓办待拍板。**措辞红线**:A 只说 "span-free"(annotation-free=LELA 占,dense-supervision-free
=TANDEM 占);不说任务第一(MultiHateLoc 占);B 说"首个形式化+系统评测 evolving-hate 协议"
不说"首个拥有换库能力"(RA-HMD 潜在具备未主张;CRAVE=检索增强训练需划界);"auditable" 钉死
审计对象=持久记忆而非即弃日志(vs SafeLens)。

**Roadmap 起点。** 开工即并行两项:E0a 时间戳采集脚本([login],<1.5h)+ E0b MHClip 结构化
档案生成([SLURM-gpu],单卡半天级);随后 E1 consensus 定生死消融(subclipK4 缓存已在
MHC/MHC_zh,`loss.py`/`run_rac.py` 已有 seg_mode 钩子,新增 seg_mode=consensus)。

**Open decisions(留给用户,DESIGN_iter3 §7)。**
1. 打包范围:角色3 上不上(默认缓办)。
2. 0.85 目标降级确认(故事重心转向去噪机制+演化协议+可审计记忆;MHClip 目标改述为
   beat MoRE + 双语同向增益)。
3. 目标 venue / 时间线(候选 ACM MM 2027 / WWW 2027 / ACL-EMNLP 2027;是否 "E1 gate 过即挂
   arXiv 占位")。
4. (次级)HateMM `hate_snippet` frame-level 定位评测是否纳入主文;共识邻居粒度等按消融跑。

**写作中发现的与侦察报告冲突/存疑点(记录在案)。**
(a) iter1/Phase-1 记录"本地 HateMM/MHClip 无金标 span",但 TEMPORAL_SPLIT §3 核实官方 Zenodo
HateMM 标注 CSV(62KB,已拉取)含 `hate_snippet` 列 —— HateMM 定位评测可能可行,需核实覆盖;
(b) 任务书称 drifting hard-negative "Phase-3 已验证健全",而 Phase-3 记录为 driftneg 在 EN
近无作用、ZH 低于 baseline —— 设计中改述为"机制未被证伪(失败归因于噪声 MIL 正样本),共识
来源能否救活它属待验证";(c) MultiHateLoc 的发表数字在 HateMM/MHC 而非 HateClipSeg,当
HateClipSeg baseline 需移植复跑(代码可得性未核实,已列 §6 风险);(d) ZH-Hateful 死链 ~60%
出自 30 样本探针中的 5 个 Hateful 样本,统计上脆弱,全量采集后需重估。

**Artifacts.** `research-wiki/DESIGN_iter3.md`(7 节:故事与贡献/方法 A/方法 B/MLLM 三角色/
实验矩阵/风险清单/[USER-DECISION] 汇总)。

---

## Iter-3 Wave-1 结果 — 2026-07-03 — E0a 时间切分基建(jobs 12170/12171)

**结论:YES(基建目标达成)。** 详细节点:`experiments/exp-temporal-split-infra.md`。

- **可定年率:** MHC-EN 87.8%(781/890,yt-dlp,59.3 min)、MHC_zh 89.6%(804/897,Bilibili
  API,33.0 min);落到我们的 split universe 内为 EN 97.6%(771/790)/ ZH 98.8%(796/806),
  不可定年样本按协议固定进 train(EN 19 / ZH 10)。
- **Survivor bias 全量确数(取代侦察期 n=5 探针):** ZH-Hateful 死链 **18.8%**(21/112)、
  EN-Offensive **21.2%**(46/217);Normal 仅 9.2%/8.6%。方向确认(有害类死得更快)、幅度中等
  ——侦察时 "ZH-Hateful ~60%" 属 n=5 高估,已修正在案。
- **temporal split 已产出:** `data/gt/{MHC,MHC_zh}_temporal/{train,val,test}.jsonl`(尺寸与随机
  split 相同:EN 549/80/161、ZH 579/78/149);切点 EN train≤2023-06-18 / test≥2024-01-16,
  ZH train≤2023-08-10 / test≥2023-11-03。统计:`data/gt/temporal_split_stats.json`。
- **必须声明的偏差:** temporal test 正例率 ~24%(EN 24.2 / ZH 24.8)vs train ~34% —— 时间
  切分自带 label-prior shift,W4 任何结果都要对照同 prior 的 floor 报告,不能直接对比随机 split。

---

## Iter-3 Wave-1 结果 — 2026-07-03 — E1 共识定生死消融(jobs 12176–12181)

**结论:PARTIAL —— ZH 验证通过,EN 硬失败,gate 未过。** 详细节点:
`experiments/exp-consensus-kill-ablation.md`;idea 节点 `ideas/retrieval-consensus-denoising.md`
置为 stage=piloted / outcome=mixed。

协议:warmup≥5 val-selected(max Val acc,roc tie-break),CLIP 背骨,λ_seg=0.5,K=4,
consensus topk=10 / τ=0.2 / EM=2。日志 `slurm/logs/mhc_train_cons_*.out`,ckpt 组
`RAC_video_consensus`。实现 `src/utils/consensus.py`(seg_mode=consensus/selfscore);
λ=0 走共识代码路径已 **bit-for-bit** 复现 baseline;full 模式与 Phase-3 jobs 12129/12131
逐位一致(harness 自校验通过)。

| 数据集 | floor(λ=0) | full(继承标签) | selfscore | consensus |
|---|---|---|---|---|
| MHC_zh(M-F1/acc) | 0.7706/0.8054 | 0.7050/0.7383 | 0.7746/0.8188 | **0.7864/0.8188(本消融最优,修复 Phase-3 洞并反超 floor)** |
| MHC-EN(M-F1/acc) | 0.7113/0.7826 | **0.7262/0.7888(最优)** | 0.6394/0.7329 | 0.5948/0.7329(硬失败,−0.117 F1) |

- **gate(双语同向 ≥ floor)未通过。**
- **诊断线索:** round-1 共识剔除(drift 降级)正样本视频子片段 ZH **300/720(41.7%)** vs
  EN **161/672(24.0%)** —— ZH 大量剔"毒正样本"而赢,EN 剔得少反而输。
- **假设(交 W2 归因):** EN 仇恨偏语音承载,视觉 sub-clip 键检索到的邻居语义无关,kNN 投票
  即噪声;ZH(Bilibili)仇恨偏视觉/屏幕文字承载,视觉键投票有效。注意 EN 上 selfscore 也失败
  (−0.072 F1),EN 问题可能部分出在"语音承载仇恨上的 sub-clip 监督"本身而非共识投票——
  W2 三方消融要把这两者分开。

---

## Iter-3 Wave-1 结果 — 2026-07-03 — E0b MLLM 结构化档案(jobs 12172/12173/12174/12184,补齐 12186)

**结论:YES(基建目标达成)。** 详细节点:`experiments/exp-mllm-archives.md`。

- **生成:** 冻结 Qwen2.5-VL-7B-Instruct,8 帧 + title/transcript → 每视频一条结构化 JSON
  (`target_groups / mechanism / modality_cues{visual,speech,on_screen_text} / explicitness /
  neutral_summary`),**英文枢轴**(prompt 强制英文,ZH 视频亦出英文档案)。
- **质量(按 id 去重、含重试):** parse_ok EN **788/790(99.7%)**、ZH **802/806(99.5%)**;
  schema_ok 780/790、793/806;**1596 条零拒答**。残余 6 条 parse 失败(EN 2 / ZH 4,重试仍
  不可解析)以 raw_output 文本回退编码,缓存无零向量。
- **产物:** `data/Archive/{MHC,MHC_zh}/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_archive.jsonl`
  + CLIP 文本编码缓存 `data/CLIP_Embedding/{MHC,MHC_zh}/*_archive_openai_clip-vit-large-patch14-336_HF.pt`
  (N=549/80/161、579/78/149,Dt=768,zero-vector=0),已推 B2
  (`b2:junyi-data/RGCL_video/embeddings/{MHC,MHC_zh}/`)。
- **caveat:** CLIP 文本 77-token 截断会切掉长 summary;schema_ok<parse_ok 的 ~1–3% 条目字段
  可能不全;档案忠实度尚无人工抽查(排入 W2);下游效用未测(W3 的事)。


---

## Iter-3 Wave 2–7 收官 — 2026-07-03 夜 → 2026-07-04(按事件记录)

### 1. 多 seed 终表与配对判定(W3 复核,jobs 12215–12227,12219–12221)

**结论:NO —— W3 的两个 headline 都没有活过多 seed。** 节点:
`experiments/exp-archive-knn-seeds.md`。

- ZH archive-kNN α=0.25(LoRA 基座)5 seeds val-选点:acc **0.8268±0.0266**;0.8523 是
  seed-0 的 best-of-5 幸运高点,仅该 seed 过 0.85 → **MHClip-ZH 0.85 仍 OPEN**。
- 同 seed 配对(archive − LoRA-only,n=5):dAcc **−0.0014±0.0313**(t=−0.10)——
  **无可靠 accuracy 增益,方向未知**;LoRA-only floor 本身极稳(0.8282±0.0139)。
- EN archive 臂 4 seeds:0.7935±0.0205(0.8075 是 max);EN 报 0.794±0.021。
- 唯一方向一致的次级信号:val-选点配对 dROC +0.0095(4/5 seeds)——只配分析段。
- 控制有效性:baseline seed-0 复跑(12223)与 12149 **bit-for-bit** 一致,证明 W5 期间的
  src 改动全部 flag-gated OFF,配对比较同代码。

### 2. 选点规则发现(零 GPU 再分析 + sha1 审计)

**发现:78 样本 dev 上 val-acc 选点自损 ~2 个 acc 点;选点规则挪动估计值的幅度超过待测
效应本身。** 节点:`experiments/exp-archive-knn-seeds.md` Addendum 1/2;脚本
`scripts/analysis/selection_rule_robustness.py`。

- 五规则网格(val-acc / val-ROC / top3-mean / last5-mean / final-epoch)重打分全部臂:
  ZH 配对档案效应跨规则在 −0.013 ~ +0.008 摆动;无规则跨臂一致占优 → 预注册规则不改,
  headline 数字不动;selection-robustness 论文附录段已成文。
- **final-epoch 口径:ZH floor = archive = 0.8537±0.0120(唯一过 0.85 的均值)**;两臂
  每 seed 逐位相同 —— sha1 审计证实 same-seed checkpoint 字节相同
  (`6d6551e4…` 与 disk_guard B2 推送记录吻合),α=0.25 键在 ep29 0 票翻转。
  **采用与否 = rule-shopping 风险,协议选择权留用户**(两口径并排方案见 MORNING_REPORT §6)。
- 附带政策执行记录:一度计划的 5-seed cross-seed ensemble **按用户政策撤销**——零作业、
  零脚本、零数字。

### 3. 三臂消融终局(floor vs transcript vs archive,jobs 12228–12231 / 12260–12266)

**结论:novelty 生死题两头都死——"archive>transcript" 不 seed-robust,archive 的 accuracy
增益本身也不存在。** 节点:`ABLATION_transcript_vs_archive.md`。

- 长上下文多语 mpnet-512 transcript 键(截断 0.0%)在 ZH ≤ floor(4-5/5 seeds)→
  truncation-repair 假设死;但 archive 的 accuracy 主张同样死(配对 ≈ 0,final-epoch 全 tie)。
- 幸存的可写内容:ZH 档案键 val-选点 ROC 4/5 seeds +0.009(弱、方向一致);EN 上
  transcript-key ROC > archive-key(4/4 seeds,+0.017)。定稿话术:分析章节,不作 headline。
- double key(等权拼接)低于一切;死路,不再碰。

### 4. W5 共识空间证伪(jobs 12243–12246,`--consensus_space {archive,blend}`)

**结论:NO(双语言、双配置全灭)——EN 共识失败不是键空间问题,机制统一主张被杀。**
数字首次入档(val-选点 Test F1/acc):

| 配置 | MHC-EN | MHC_zh |
|---|---|---|
| consensus(原视觉空间,E1) | 0.5948 / 0.7329 | **0.7864 / 0.8188**(赢家) |
| consensus_space=archive | 0.5663 / 0.7205 | 0.7221 / 0.7718 |
| consensus_space=blend | 0.6453 / 0.7143 | 0.7232 / 0.7651 |
| floor(λ=0) | 0.7113 / 0.7826 | 0.7706 / 0.8054 |

- EN:换成档案语义空间投票反而更差 → 与 E1、W2 合并成完整归因链(EN 仇恨语音承载 →
  子片段监督本身失效,投票空间不是病灶);共识 claim 严格 scoped 到 ZH。
- ZH:原 raw-CLIP 视觉空间就是最优;档案空间毫无增益。
- 实现:`src/run_rac.py`/`src/utils/consensus.py` 新开关,default=clip 时与 pre-W5 逐位一致
  (由 12223 bit-for-bit 复现 12149 背书)。
- **附:EN 档案 α 网格与 mode=both(jobs 12247–12251,seed-0)**:knn α∈{0.15,0.2,0.3,0.35}
  val-选点 acc 0.7888–0.8137(相对 floor 均值 0.78±噪声,全部噪声级);mode=both 0.7702
  (有害)。EN 各杠杆至此全部噪声级或有害,唯一未决 = 角色 3 仲裁(W7)。

### 5. W4 校准漂移发现(temporal evolving-memory 协议,jobs 12197/12214/12253)

**结论:"hate evolves" 在 EN 可测(−0.084 macro-F1),但演化的主成分是校准漂移;
原"加样本进记忆"机制不成立,k=20 阈值再校准零重训全额收复。** 节点:
`EVAL_temporal_memory_W4.md`。

- EN temporal ROC 0.8484 > 随机 split 参考 0.7175 → 可分性活着,operating point 死了。
- 记忆增补:所有 k≤80、三种选样策略、双语言 flat-to-negative。
- 阈值再校准:k=20 → 0.7336 ≥ 随机 floor 0.7113(oracle 天花板 0.7646);ZH 负对照:
  无漂移时小 k 校准纯噪声 → 部署应由漂移信号门控。
- 故事定稿:检索架构把 operating point 暴露为一等、O(1)、可逆旋钮;trained-MoE 头藏在
  权重里,结构性做不到。idea 节点 `ideas/evolving-memory-protocol.md` 置
  validated-as-calibration。

### 6. 可审计性双实验(档案忠实度审计 + kNN 记忆编辑演示)

**结论:可审计/可编辑主张成立为能力演示,边界条件如实入档。** 节点:
`AUDIT_archive_faithfulness.md`、`DEMO_memory_editing.md`。

- 忠实度:60 条分层抽审 faithful 77%;幻觉 15% 几乎全是字段级虚报(benign→spurious
  mechanism,ZH-Normal 6/10 最重);洗白 5% 全是"毒性只在标题"模式;反向发现 1 例疑似
  gt 漏标(`BV1MU4y1D7Ks`)。模型初判、人工终审条款写明。
- 编辑:EN 定向删除切片翻转率 ≈ 随机对照 15×;删 2 条 W2 噪声条目即修复 EN
  (0.8075→0.8199,超全部随机 seed,零训练);ZH v1 组级 0 翻转 = 诚实负结果,归因
  target 字段召回 1.6%(有害类)——档案质量是编辑定向性的上限,属边界条件不是反例;
  低置信规则只可作隔离候选队列,不可自动删除。

### 7. v2 档案(prompt v2 生成 + ZH 编辑复测,jobs 12234/12258/12259;EN 全量 12280 在飞)

**结论:审计驱动的 prompt 修复把 ZH 有害类 target 召回 1.6%→49.4%;编辑定向性改善有限。**
节点:`DEMO_memory_editing_v2_zh.md`;prompt 变更记录在
`src/utils/generate_video_archive_HF.py` + `scripts/slurm/gen_archive_v2.sbatch` 头注。

- v2 规则:target 必填(slur 自证目标群体、标题攻击计入)+ mechanism 须有可引证据。
- ZH train 非空 target 6/583(1.0%)→ 128/579(22.1%);有害类 3/182(1.6%)→ 89/180(49.4%)。
- 复测(冻结 v1 获胜 ckpt,换键零重训,由 sha1 审计背书):women 切片(63 条)删除有超
  随机包络的切片效应(0.70→0.60);LGBTQ+ 字段切片仍 0 翻转;v2 键整体基线 0.8523→0.8255。

### 8. 定位评测台 + HateClipSeg 落库(jobs 12232/12274)

**结论:HateMM 金标定位台建成并给出我们方法的诚实数字;HateClipSeg 90.8% 存活子集落库,
成为主定位评测集(评测在飞)。** 节点:`EVAL_localization_hatemm.md`、
`DATASET_hateclipseg.md`、`EVAL_localization_hateclipseg.md`。

- HateMM:427/427 hate 视频 span 全解析(671 段,中位覆盖 46%)→ 定位评测可行且非平凡;
  自定双口径协议(full / hateonly)全项目统一;我们的 model-score full mAP 0.589/AUC 0.781,
  但 video-broadcast 对照 0.578/0.774 → 段内分辨贡献很小,hateonly AUC 仅 0.577(K=4 粗段 +
  视觉-only 键对 speech-carried 仇恨是盲区,如实写)。
- MultiHateLoc:官方仓库为空 → **不复现**(用户政策),发表数字只进 related work 并附
  "不可直接比较"协议对照表;起步代码 `baselines/multihateloc_reimpl/` 标 ABANDONED 留档。
- HateClipSeg:395/435 视频(90.8%)、10,604/10,614 段落库,ffprobe 全过、时间戳零错位;
  selection-bias 声明成文(yt 折损 20.8% ≫ bit 6.9%,harm 类只作聚合);零训练跨数据集
  kNN 共识打分协议写死(K=30 由金标段中位 8.1s 决定,非调参),主表待 job 12274。

### 9. MoRE 复跑进度(jobs 12235–12273)

**状态:阶段 1/2 完成,G6 最终训练(12273)在飞。** 节点:`BASELINE_MoRE_rerun.md`。

- 环境两套(MoRE_env / MoRE_paddle)+ 版本锁定;释出代码问题 7 项全部文档化处置
  (含 audio 检索库 bug:主跑保留 bug 的 asreleased variant + bugfix variant 双轨)。
- 缺件全部本地复原并标注:caption=Qwen2.5-VL 复原(其生成方式零文档)、MHClip tsv/
  speech/title/label 从 annotation(new).json 重建;ZH OCR 经 paddle GPU(cudnn 不认)/
  CPU(SIGILL)双卒后按预授权 fallback 落 easyocr(与其 EN 官方协议同引擎),814/814 行。
- 特征/检索全产出(3 数据集 × 2 variant);G6 = 官方 yaml、seed=2024、双轨评测
  (官方 test 全量 sanity + 我们 clean 子集严格同场)。

### 10. 三条用户政策(2026-07-03/04 生效,全项目约束)

1. **禁 cross-seed ensemble**:任何集成不入方法;已计划的 ensemble 线当场撤销(见 §2)。
2. **无代码不复现**:官方无代码的工作(MultiHateLoc)只讨论发表数字并标注不可比,
   不投入复现(见 §8)。
3. **不发邮件、缺件自补**:复跑他人工作缺失的资产(caption、tsv、OCR 引擎)一律本地
   复原、如实脚注,不等作者回复(见 §9)。

**Ideas 节点状态同步:** `archive-as-retrieval-key` → refuted(新节点,负结果入档);
`retrieval-consensus-denoising` → ZH-validated / EN-refuted(归因完结);
`evolving-memory-protocol` → validated-as-calibration(新节点)。

## 终局收卷 — 2026-07-05(FINAL;全部实验收敛,状态固化)

**本节为项目终局记录。以下八份终报全部落地,无在飞作业,无未判读结果;
MORNING_REPORT.md 已更新为终版(FINAL - 2026-07-05)。**

### 1. mm 片段键终判(`EXP_mm_segment_keys.md`,jobs 12302/12303/12310–12317)

**主表 FAIL,归因闭环。** 片段级 Whisper ASR 多模态键把 EN 共识 annotator 全面修好
(正监督供给 56%→19%、投票视频级→片段级 wv-std 0.048→0.12、严重度反相关消除、
灾难性 clip-consensus −0.117 F1 被完全救回 +0.10~0.13),但训练端仍不超 floor
(预注册判定:final-ep 3/3 seed −0.0116±0.0087;val-选点 +0.0245 由单 seed 驱动 ±0.088)。
ZH mm 探针死(窗文本率 48.5% + CLIP-zh 弱),按预注册纪律不训练。
**EN 病灶钉死:片段监督通道本身对语音承载仇恨无增益** —— 三段归因链
(视觉键投票=视频级噪声 → 档案/混合空间救不回 → 证据匹配语音键修好 annotator 仍无增益)闭环。
共识去噪 claim 维持 ZH-scoped;方法学副产品 = evidence-matched segment keys + probe-before-train。

### 2. 同场 MoRE 三库全胜(`BASELINE_MoRE_rerun.md`,jobs 12235–12318)

复跑完成:sanity(HateMM 复跑落发表值 −2~3pt = 复现成功)+ clean 同场终表:
HateMM +5.6 acc / EN +8.7 acc(+22.9 F1)/ ZH +6.7 acc,**三库全胜且 seed 均值上界、
bugfix variant 均不翻转** —— 论文主对比表定稿。释出代码 7 缺陷、缺件复原、EN 早停塌缩
归因全部留痕。

### 3. role-3 选择性推理终结(`EVAL_role3_selective_reasoning.md`,jobs 12279/12288/12305)

三代 7B 仲裁器(v1/v2/v3-LoRA)全部未过 val 门,两语言 val 选定配置=不仲裁;
EN 维持 0.8075(memory-clean 0.8199)。**门控本身有效**(EN 24% 样本拿住 42% 错误;
oracle 0.857–0.888);复活条件量化:deferred@30% ≥0.667 打平 / ≥0.846 跨线,留 ≥72B/API。
ZH v3 test 侧 +0.02 未选中增益如实脚注、按协议不作 claim。

### 4. ZH 共识多 seed 复检(`experiments/exp-consensus-zh-seeds.md`,jobs 12289–12300)

**修复稳、反超不成立**:5-seed val-选点 +0.0115±0.0418(p≈0.57)、final +0.0247±0.0272
(p≈0.11);任何 seed/口径都不复现 full-mode −0.066 毒化洞。论文措辞定稿:
"consensus de-poisons sub-clip supervision (−0.066 → ≈ floor / weakly above)"。

### 5. EN 主表定稿(`experiments/exp-archive-knn-seeds.md` Addendum 3,jobs 12275–12277)

EN floor 4-seed 补齐:val-选点 floor 0.7702±0.0221 vs archive 0.7935±0.0205(假增益=
选点交互),final-ep floor 0.7888±0.0152 vs archive 0.7826±0.0134(0/4 正,t=−2.45)。
**EN 全配置 0.77–0.79 不分离 → "≈0.79 regardless of key augmentation" 定稿。**

### 6. 档案 v2 闭环(`ARCHIVE_V2_ITERATION.md`,jobs 12234/12258/12259/12280)

target 召回 harmful ZH 1.6%→49.0% / EN 11.6%→54.5%,结构违规 14.6%→1.0%;
ZH 编辑可寻址性修复(0→20/63 条);EN 字段切片方向性更干净(2/14 翻转 vs 随机全 0,
整体 acc 不掉);v2 键无 accuracy 收益(ZH −2.7)→ 档案付费点=审计/编辑。
键增强对照重训按 post-mortem 取消,换键对比零训练完成。

### 7. HateClipSeg 定位主表(`EVAL_localization_hateclipseg.md`,job 12274)

零训练跨库共识票:最好配置 full AP 0.545/AUC 0.588(random +0.088/+0.100);
within-video 信号显著但小(wv-AUC 0.526,仅 K=4+subclip cell 过 Bonferroni);
K=30 密度匹配负结果;换记忆零重训改变行为模式(可换记忆支柱双向证据)。
能力演示成立、定位强度诚实地弱。

### 8. 终版固化与收口

- **MORNING_REPORT.md → FINAL**:终版记分板(HateMM 0.870✓ / Impli ~0.91✓ /
  ZH 0.827‖0.8537 双口径待拍板 / EN ≈0.78–0.80 近天花板)、MoRE 全胜表、四支柱终版、
  被杀主张 13 条全清单、方法学章素材、用户待拍板 3 项、遗留 TODO 7 项。
- **Ideas 节点收口**:`mm-segment-keys` 新建(outcome=attribution-closed);
  `role3-selective-reasoning` 新建(closed:7B 线终结,门控+oracle 入 TODO);
  `retrieval-consensus-denoising` 终态=repair-yes / beat-floor-no / attribution-closed。
  所有节点内 open 问题移入 MORNING_REPORT §8 TODO(transcript final-ep 合并表、
  HateClipSeg×mm 键重打、v3 档案、更强仲裁器、full-mode 洞补 seed、gt 漏标终审、
  word-ts 修复)。
- **全量 commit**(代码 / 文档 / 日志与小数据 三批,排除子模块与一切权重文件),
  项目状态至此固化。
