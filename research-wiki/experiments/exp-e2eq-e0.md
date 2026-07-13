---
type: experiment
node_id: exp:exp-e2eq-e0
title: "E2EQ (C1): RA-HMD-style two-stage QLoRA of Qwen2.5-VL-7B under the retrieval-contrastive objective (HateMM + MHC-EN + MHC-ZH)"
idea_id: ""
verdict: KILLED-AT-E-G0 (evidence triage, zero-GPU) — KILL_CONFIRMED by independent 0-context review (refine-logs/C1_KILL_REVIEW.md)
confidence: n/a (design; killed at G0 on expected value)
date: "2026-07-13"
status: >
  KILLED-AT-E-G0 by evidence triage (zero GPU); KILL_CONFIRMED by independent fresh 0-context review
  (refine-logs/C1_KILL_REVIEW.md, Task A). +3 bar structurally unreachable (ceiling ≈ +1.6 over the
  frozen-Qwen floor, inside the ±1-2pt noise floor). NO GPU campaign will be spent on C1. An optional
  near-free MEASURED settling run (sequential Stage-2 DEV read-out on already-cached *_p9c3_* features,
  NO test touch) is being executed SEPARATELY to convert the inferred negative into a measured one.
  No C1 campaign job submitted, no code changed for it, nothing committed.
hardware: >
  planned: 1x A100-SXM4-80GB (SLURM node foscsmlprd01, gpu:a100:8, per-user cap 2 GPU /
  16 CPU / 128 GB). Stage-1 LoRA-SFT ~30-46 min/cell (P9 precedent); LoRA feature
  extraction ~16-18 min/cell (P9b precedent); Stage-2 frozen-feature RGCL ~1 min/cell
  (cached features, enc3seed precedent). No --time.
provenance: >
  RA-HMD numbers read from arXiv 2502.13061v1 (HTML full text) via WebFetch 2026-07-13,
  Tables 1 / 3(a) / 3(b), cross-checked against the repo's own released QLoRA config
  (RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/memes/hm/qwen2vl-7B_qlora_classifier.yaml).
  Frozen-Qwen floors cite exp-encoder-3seed.md file:line. Prior video LMM-RGCL results
  cite research-wiki/EXP_p9_lmm_rgcl_video.md file:line. Asset/VRAM facts verified live
  on the login node 2026-07-13.
added: 2026-07-13T00:00:00Z
tags: ["hateful-video", "MLLM", "QLoRA", "two-stage", "retrieval-contrastive", "RA-HMD", "LMM-RGCL", "pre-registered", "KILLED-AT-E-G0", "KILL_CONFIRMED", "HateMM", "MHC", "C-line", "E2EQ"]
---

# E2EQ — RA-HMD-style two-stage QLoRA under the retrieval-contrastive objective (C1)

> ## STATUS: C1 KILLED AT E-G0 — KILL_CONFIRMED (2026-07-13)
>
> **C1 is KILLED at gate E-G0 by evidence triage (zero GPU), and the kill is KILL_CONFIRMED by an
> independent fresh 0-context review** (`refine-logs/C1_KILL_REVIEW.md`, Task A). **No GPU campaign
> will be spent on C1.** The pre-registration below is retained as the audit record of the killed route.
>
> **Rationale (why the +3 bar is structurally unreachable):**
> - **RA-HMD Table 3a prices the untested sequential cell at ~+0.7 acc in-domain.** "w/o Stage 2" = 81.4
>   acc provably retains a trained head (only 0.7 below the full 82.1), so full − w/o-Stage2 = **+0.7
>   in-domain** genuinely prices C1's marginal Stage-2 RGCL-contrastive increment — on ~8,500 memes
>   (10-15× our video data). Stage-2's real value is OOD (−7.6 acc cross-dataset if removed), not the
>   in-domain accuracy our goal requires.
> - **P9 already measured Stage-1 on video: C3-mlp ≈ frozen-Qwen floor** (EN +0.6 / ZH +1.0 / HateMM
>   +0.9, all within the ±1-2pt noise floor); **C3-knn lands below floor** (EN −2.7 / ZH −2.2 / HateMM
>   −4.7). The meme "+7.9" Stage-1 gain does NOT transfer — our frozen-Qwen features are already strong.
> - **P9b ran RGCL jointly inside the LMM loop and FAILED: 0/12 wave cells beat the protocol-matched
>   floor**; the rgcl term only **redistributes** ±1.8pt head↔memory with no net system gain.
> - **Sum ceiling ≈ +1.6 acc** over the frozen-Qwen floor (P9 Stage-1 flat + RA-HMD Stage-2 +0.7) —
>   **below the +3 bar and inside the ±1-2pt noise floor.** The one faithful untested cell (sequential
>   frozen-feature Stage-2 on LoRA-adapted features) cannot exceed RA-HMD's own confound-free +0.7.
>
> **Optional near-free MEASURED settling run (executed SEPARATELY; converts inferred → measured negative).**
> A single sequential Stage-2 **DEV** read-out on the **already-cached** `data/CLIP_Embedding/*/*_p9c3_*`
> LoRA-adapted features (~seconds/run, enc3seed precedent) + one loader edit + `codex-code-review`,
> **NO test touch**. A DEV result within ±0.010 of the frozen-Qwen DEV floor closes the cell as a
> *measured* negative — "RA-HMD's exact sequential two-stage, run faithfully on video, still lands within
> noise of the frozen-encoder RGCL floor at 7B." This changes only the paper's *framing* (measured vs
> inferred), **not the decision**, and **no GPU campaign is spent** (`C1_KILL_REVIEW.md` §5).
>
> The **DRAFT-UNREVIEWED** pre-registration banner immediately below is **SUPERSEDED** by this kill;
> it is retained unaltered for provenance.

> **PRE-REGISTRATION STATUS: DRAFT-UNREVIEWED.** This file pre-registers hypotheses,
> mechanisms, gates, kill numbers, seeds, and protocol for the top-ranked C-line literature
> candidate (C1). **No experiment is run, no code is changed, nothing is committed.** It must
> pass a **fresh 0-context reviewer** and a **user-visible report** before any `sbatch`.
> **The GPU queue is shared with the running A-line (lb_scgp_global M2/M3); C-line queues
> behind it** (reflection §5 ruling: A-line gets its M3 decision first, C-line runs in the gaps).

**One-line mechanism.** Instead of feeding a *frozen* Qwen2.5-VL-7B feature into the RGCL head
(the encoder-swap positive, HateMM +5.3 acc), **LoRA-adapt** the Qwen backbone to the task
(Stage 1: LoRA-SFT with LM + binary-classifier cross-entropy), then **freeze it, extract the
adapted last-token features, and train the RGCL alignment-fusion head + triplet-contrastive +
kNN memory on those adapted features** (Stage 2, RA-HMD's exact recipe: `RA-HMD/Stage2/src/run_rac_lmm.py`).
The bet is that task-adaptation of the *representation manifold itself* (the project's only proven
+3 lever, D2) transfers the HateMM encoder-swap win to a second and third dataset.

---

## 1. Hypothesis

**H (representation-level task adaptation).** RA-HMD's headline in-domain gain on hateful memes
is a **representation** gain from Stage-1 LoRA task-adaptation (§3), and the project's one robust
+3 lever is likewise representation-level (frozen encoder swap, HateMM +5.3 acc / +6 F1, 3/3 seeds,
both protocols; `exp-encoder-3seed.md:184-198`). Composing the two — LoRA-adapting the Qwen encoder
*and* keeping the RGCL retrieval head — should lift the RGCL head **≥ +0.030 acc AND ≥ +0.030 F1
over the frozen-Qwen floor** on ≥ 1 dataset, 3-seed paired, under a stated protocol.

**H0 (the null this line most likely dies on).** The frozen-Qwen encoder-swap already banked the
representation gain; LoRA-SFT on the tiny video train sets (HateMM 744 / MHC-EN 549 / MHC-ZH 579,
vs RA-HMD's ~8,500-meme HatefulMemes) adds nothing over the frozen floor and/or overfits, so the
adapted-feature RGCL head lands within seed noise of the frozen-Qwen floor — **exactly what the
project's own P9/P9b video LMM-RGCL runs already found** (§4). If E-G1/E-G2 reproduce P9/P9b's
flat-to-negative result, the line dies before the test touch.

---

## 2. Mechanism — the exact RA-HMD sequential two-stage (with asset map)

RA-HMD (LMM-RGCL) is a **sequential** two-stage pipeline, NOT joint training (§3, resolved from
the paper). C1 runs it faithfully on video, reusing assets that already exist in the repo:

**Stage 1 — LoRA-SFT of Qwen2.5-VL-7B (representation adaptation).**
- Trainer: `RA-HMD/LLAMA-FACTORY-Ver202512`, `stage: sft_classifier` (joint LM + binary
  classifier head; loss = L_LM + L_cls, `loss_ratio [1,1]`). Config precedent already written:
  `my_configs/hatevideo/p9_hatemm_c3_s1.yaml` (and `p9_mhc_*`, `p9_mhc_zh_*`).
- QLoRA: `quantization_bit: 4` (bitsandbytes), `lora_target: all`, `lora_rank 128`,
  `lora_alpha 256`, `lora_dropout 0.05` — the RA-HMD meme default
  (`my_configs/memes/hm/qwen2vl-7B_qlora_classifier.yaml:2-14`). **Anti-overfit amendment (§8):
  this draft pre-registers a lower-rank sweep** (r ∈ {8, 16, 32}) because RA-HMD tuned r128 on
  8,500 memes, not 744 videos.
- Vision tower + mm-projector frozen (`p9_hatemm_c3_s1.yaml:19-20`). 8 frames/video, low
  per-frame pixels (`image_max_pixels 65536`) — the documented video deviation that lets 8 frames
  fit one A100-80GB (`p9_hatemm_c3_s1.yaml:5-8`).

**Stage 2 — freeze the LMM, extract adapted features, train the RGCL head + kNN.**
- Extraction: `src/utils/generate_VideoMLLM_embedding_lora_HF.py` + `scripts/slurm/gen_embed_lora.sbatch`
  (already produce `*-LoRA_HF.pt` caches — e.g. `data/CLIP_Embedding/MHC/train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`).
- RGCL Stage-2 trainer: `RA-HMD/Stage2/src/run_rac_lmm.py` (loss options `naive|triplet|contrastive`,
  `run_rac_lmm.py:54-55`; margin 0.1, `:57`; hybrid BCE `--hybrid_loss`, `:69`). The FB meme recipe
  `RA-HMD/Stage2/scripts/FB/qwen2vl7b-contrastive.sh:7-17` trains **only the CLS head on the
  extracted feature** ("only the CLS model is trained ... to save time and GPU memory") — i.e.
  Stage-2 is frozen-feature metric learning, identical in kind to our main video RGCL head.
- Decision: top-20 similarity-weighted kNN vote over the train memory (`--topk 20 --majority_voting arithmetic`).

**The C1 novelty vs everything already run:** this is the **only** combination that composes both
levers *sequentially in RA-HMD's order* — Stage-1 LoRA-adapt → extract adapted features → **train
the RGCL contrastive head + kNN on the adapted features**. §4 shows why P9 (raw kNN on adapted
features, no Stage-2 head training) and P9b (RGCL trained *jointly* inside the LMM loop) are both
distinct from, and do not subsume, this recipe.

---

## 3. Evidence — RA-HMD discrepancy RESOLVED (arXiv 2502.13061v1, read 2026-07-13)

**Task-1 question:** is +3.0 acc / +4.1 AUC the *in-domain full-pipeline* delta, and where does
the retrieval-contrastive part's benefit actually sit? **Two scouts disagreed** (LITERATURE
doc §C1, `LITERATURE_mllm_integration_2026-07-13.md:12`): methods-scout said "whole QLoRA pipeline
+3 acc vs frozen RGCL"; cross-domain-scout said "Stage-2 contrastive increment is OOD-weighted,
in-domain ≈ frozen." **Reading the paper resolves it: both are true.**

### 3.1 The in-domain headline IS the full-pipeline delta vs frozen RGCL (methods-scout correct)

Table 1 (Qwen2VL-7B, HatefulMemes, in-domain), read from arXiv 2502.13061v1 HTML:

| system | AUC | Acc | source |
|---|---|---|---|
| original **frozen RGCL** (CLIP/ALIGN, non-LMM), Table 1 row 4 | **87.04** | **78.82** | 2502.13061v1 Table 1 |
| Qwen2VL-7B **SFT** | 86.33 | 78.60 | 2502.13061v1 Table 1 |
| Qwen2VL-7B **LMM-RGCL** (full 2-stage) | **91.1** | **82.1** | 2502.13061v1 Table 1 |

→ **+3.0 acc = 82.1 − 78.82 ≈ +3.28**; **+4.1 AUC = 91.1 − 87.04 ≈ +4.06** — the full-pipeline
in-domain delta **vs the frozen-feature RGCL baseline** (exactly the "82.1 vs 78.82" the LITERATURE
doc cited, `LITERATURE_mllm_integration_2026-07-13.md:11`). Note the two ~78.x rows are distinct
systems (frozen RGCL 78.82 acc; Qwen SFT 78.60 acc) — the headline is measured against the frozen
RGCL row, i.e. the meme analogue of our pipeline.

### 3.2 The in-domain gain is STAGE-1-driven; Stage-2 contrastive pays off OOD (cross-domain-scout correct)

Table 3(a) ablation (Qwen2VL-7B, HatefulMemes, **in-domain**):

| config | AUC | Acc | in-domain increment |
|---|---|---|---|
| full LMM-RGCL | 91.1 | 82.1 | — |
| **w/o Stage 1** (drop LoRA-SFT) | 84.4 | 74.2 | Stage 1 buys **+6.7 AUC / +7.9 Acc** |
| **w/o Stage 2** (drop RGCL contrastive) | 90.2 | 81.4 | Stage 2 buys only **+0.9 AUC / +0.7 Acc** |

Table 3(b) (**cross-domain**): dropping Stage 2 costs **−3.7 AUC / −7.6 Acc** cross-dataset;
cross-domain HM→HarMeme LMM-RGCL 88.8 AUC vs SFT 62.99 AUC. **So the retrieval-contrastive Stage-2
term is an OOD/robustness lever; in-domain its increment is ~+0.7 acc.** The in-domain +3.28 acc is
carried by **Stage-1 LoRA representation adaptation**.

### 3.3 Verdict on the discrepancy + the load-bearing implication for C1

**Resolution: both scouts are right and can be simultaneously true.** The +3.0 acc / +4.1 AUC
in-domain headline is the full two-stage pipeline vs frozen RGCL, but it is a **Stage-1-LoRA
representation gain**, not a retrieval-contrastive gain (Stage-2 adds ~+0.7 acc in-domain, and its
real value is cross-domain, −7.6 acc if removed).

**Implication (this is the crux of the C1 bet and its biggest risk):** on our *in-domain* video
targets, any C1 accuracy gain must come from **Stage-1 LoRA task-adaptation of the representation**,
because the retrieval-contrastive objective alone is worth ~+0.7 in-domain. But the project already
banked the *frozen* Qwen representation gain (encoder-swap, HateMM +5.3, `exp-encoder-3seed.md:184`),
and P9's Stage-1-LoRA read-out did **not** beat that frozen floor on HateMM (§4). C1's entire
marginal value therefore rests on the question P9 left open: does re-training the RGCL head on the
*LoRA-adapted* features recover/exceed the frozen-Qwen floor, or does LoRA-SFT reshape the manifold
against the retrieval memory (as P9's raw-kNN read-out suggested)?

### 3.4 Hyperparameters / data scale (Task-1 d)

Paper (Appendix B.2, via WebFetch): QLoRA (Dettmers et al. 2023), ~5M trainable params (MLP+LoRA),
proj_dim 1024, AdamW, lr 1e-4, batch 64, 30 epochs (Stage-2). **Rank/alpha not stated in the fetched
text**, but the released config uses `quantization_bit 4`, `lora_rank 128`, `lora_alpha 256`,
`lora_dropout 0.05`, lr 4e-5, 3 epochs (`my_configs/memes/hm/qwen2vl-7B_qlora_classifier.yaml:2-14,37`).
**Training data scale: RA-HMD trains on six meme datasets** (HatefulMemes ~8,500; MAMI ~9,000;
HarMeme ~3,000; Harm-P ~3,000; PrideMM ~5,000; MultiOFF ~450). **Our video train sets are 7-19×
smaller** (HateMM 744 / MHC-EN 549 / MHC-ZH 579) — the central over-fitting risk (§8).

> ⚠️ **Provenance caveat for the reviewer.** RA-HMD numbers were read by a small model over the
> arXiv HTML; a fresh reviewer should re-verify Table 1 / 3(a) / 3(b) decimals against the PDF
> before any number is transcribed into the paper. The qualitative resolution (§3.3) is robust to
> ±0.x decimals.

---

## 4. Differentiation vs the project's own settled video LMM-RGCL negatives (hard constraint)

**The project already ran video LMM-RGCL twice and killed it** (`EXP_p9_lmm_rgcl_video.md`). C1 is
NOT a re-run only if it occupies the one cell P9/P9b left open. The reviewer MUST scrutinise this.

| prior | what it did | result (protocol-matched floors) | why C1 ≠ it |
|---|---|---|---|
| **P9 C3-mlp** | Stage-1 LoRA-SFT, decide by the in-LMM MLP head | ≈ floor: EN +0.6 / ZH +1.0 / HateMM +0.9 pt (noise) `EXP_p9:134,150,191` | C1 decides by the **RGCL kNN memory**, not the LMM's own head |
| **P9 C3-knn** | Stage-1 LoRA-SFT, then **raw kNN** over adapted features (no Stage-2 head training) | **below floor**: EN −2.7 / ZH −2.2 / HateMM −4.7 `EXP_p9:135,192` | C1 **trains an RGCL contrastive head on the adapted features** (Stage 2), which C3-knn omitted |
| **P9b D3** | RGCL contrastive trained **jointly inside** the LMM LoRA loop (`loss_ratio [1,1,1]`) | **0/12 cells beat floor**; bs=1 in-batch degeneracy; ±1.8pt head↔memory swap, no net gain `EXP_p9:335-350,370` | C1 runs RGCL **sequentially on frozen adapted features** (RA-HMD's actual order), not jointly; sidesteps the bs=1 in-batch-negative degeneracy P9b hit (`EXP_p9:259-272`) |

**The open cell C1 occupies:** *Stage-1 LoRA-SFT → extract adapted features → Stage-2 frozen-feature
RGCL head + triplet-contrastive + kNN* (RA-HMD's exact sequential recipe via `run_rac_lmm.py`). P9
did Stage-1 then raw kNN (no Stage-2 head). P9b did RGCL jointly (not sequentially, and degenerate at
bs=1). **Neither trained the RGCL contrastive head on the frozen LoRA-adapted features.** That is the
only untested composition — and §3.3 explains why it is the theoretically-motivated one (Stage-2
frozen-feature metric learning is what RA-HMD actually ships; `qwen2vl7b-contrastive.sh`).

**Honest framing for the paper regardless of outcome:** if C1 also lands within noise of the
frozen-Qwen floor, it *completes* the P9/P9b negative — "the RA-HMD two-stage, run in its exact
released sequential form on video, still does not beat the frozen-encoder RGCL floor at 7B" — a
stronger, cleaner negative than P9/P9b's (which had the jointly-trained-RGCL and raw-kNN confounds).

---

## 5. Asset & VRAM recon (verified live 2026-07-13)

| asset | status | note |
|---|---|---|
| `RA-HMD/LLAMA-FACTORY-Ver202512` | full LLaMA-Factory checkout (submodule `../../.git/modules/...`), git submodule | has `sft_classifier` custom stage; **its RGCL path is unwired/rgcl-OFF as shipped** (P9 finding, `EXP_p9:11-17`) — C1 uses it ONLY for Stage-1 LoRA-SFT; Stage-2 RGCL runs in `RA-HMD/Stage2/run_rac_lmm.py` |
| Stage-1 video configs | **exist**: `my_configs/hatevideo/p9_{hatemm,mhc,mhc_zh}_c3_s{0,1,2}.yaml` (r128/α256/4-bit-capable) | reusable; add low-rank variants (§8) |
| Stage-2 RGCL trainer | **exists**: `RA-HMD/Stage2/src/run_rac_lmm.py` + `scripts/FB/qwen2vl7b-contrastive.sh` | meme-tested; needs a video-feature entry (LMM loader present: `load_feats_from_LMM`) |
| LoRA feature extractor | **exists**: `src/utils/generate_VideoMLLM_embedding_lora_HF.py` + `scripts/slurm/gen_embed_lora.sbatch` | already emits `*-LoRA_HF.pt` (MHC caches present) |
| LoRA-SFT data (Yes/No) | **registered all 3 datasets**: `{hatemm,mhc,mhc_zh}_lora_yn{,4}_{train,val,test}` (`data/dataset_info.json`) | yn = 8-frame, yn4 = 4-frame/bs4 (P9b) |
| LoRA frames | **present**: HateMM 1065 / MHC 790 / MHC_zh 806 dirs (`data/lora_frames/`) | |
| ASR transcripts | **present** all 3 (`data/ASR/{HateMM,MHC,MHC_zh}/*whisper-large-v3*`) | text stream ready |
| Frozen-Qwen features (floor) | **cached** HateMM/MHC/MHC_zh train/dev/test (`data/CLIP_Embedding/*/‌*Qwen2.5-VL-7B-Instruct_HF.pt`) | the E-G2 control floor |

**Local model checkpoints (verified — do NOT download):**

| model | local status | usable for C1? |
|---|---|---|
| **Qwen2.5-VL-7B-Instruct** | **COMPLETE** (16 GB, 5 safetensors shards) at `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct` | **YES — the only fully-downloaded VL model; C1 is a 7B-only bet** |
| Qwen2.5-VL-32B-Instruct | **NOT present** (only `.locks`, no `blobs/`, no snapshots) despite a completed-looking `dl_qwen25vl_32b.log` | no |
| Qwen2.5-VL-72B-Instruct | **NOT present** (only `.locks`) | no |
| Qwen3-VL-8B-Instruct | metadata stub only (8 KB, 0 safetensors) | no |
| Qwen3-VL-235B-A22B-Instruct | metadata stub only (8 KB) | no |

**VRAM feasibility verdict (7B QLoRA):** node = 8× **A100-SXM4-80GB** (`scontrol show node foscsmlprd01`,
`Gres=gpu:a100:8`, per-user cap 2 GPU). Stage-1 8-frame Qwen2.5-VL-7B LoRA fits **one** 80 GB A100 in
**bf16** (not even 4-bit) at bs1 with `image_max_pixels 65536`, per the P9 live run
(`p9_hatemm_c3_s1.yaml:5-8`; `EXP_p9:52,116`); 4-bit QLoRA is strictly lighter. 4-frame/bs4 also fits
(`EXP_p9:288`). Stage-2 runs on cached features in ~seconds (enc3seed ~25 s/run). **7B QLoRA on the
available GPUs is comfortably feasible on 1 of the 2 permitted GPUs.**

---

## 6. Data plumbing & gold-annotation isolation

**Gold-annotation rule (user hard constraint, upheld):** the method uses **no gold annotations**.
Stage-1 LoRA-SFT trains on TRAIN labels only (the binary hateful/normal label = standard supervised
training data, allowed — same status as any classifier's training labels; NOT a gold *annotation*
channel like target/time-spans). Stage-2 RGCL uses TRAIN labels for the memory + contrastive pairs;
DEV labels for selection only; TEST labels touch only the final metric (one touch, §7 E-G3). **No OCR
channel** (user veto 2026-07-13). **No cross-seed ensembles.** **Local open weights only (7B).**

---

## 7. Gate sequence (numeric kill numbers + cost + test-touch discipline)

**Floors (frozen-Qwen RGCL, 3-seed, the C1 control) — pulled with provenance from `exp-encoder-3seed.md`:**

- **HateMM** frozen-Qwen: **val-sel Test** acc/F1 s0 0.8698/0.8606 (`exp-encoder-3seed.md:251`),
  s1 0.8651/0.8586 (`:252`), s2 0.8837/0.8753 (`:253`) → mean **0.8729 acc / 0.8648 F1**;
  **final-ep** s0 0.8605/0.8507, s1 0.8605/0.8514, s2 0.8837/0.8753 → mean **0.8682 acc / 0.8591 F1**
  (per-seed table `exp-encoder-3seed.md:154-159`).
- **MHC-EN** frozen-Qwen: **val-sel** s0 0.7888/0.7378 (`:257`), s1 0.7826/0.7283 (`:258`),
  s2 0.7702/0.6997 (`:259`) → mean **0.7805 acc / 0.7219 F1**; **final-ep** s0 0.8012/0.7596,
  s1 0.7702/0.7203, s2 0.7826/0.7475 → mean **0.7847 acc / 0.7425 F1** (table `:164-170`).
- **MHC-ZH** (no encoder-swap 3-seed exists; floor from the P9 reconciliation): frozen-Qwen RGCL
  test **0.8188** (`EXP_p9_lmm_rgcl_video.md:125`); protocol-matched LoRA final-epoch multi-seed
  **0.8537 ± 0.012** (`EXP_p9_lmm_rgcl_video.md:150,245`). **ZH already sits at/above published
  frontier and crosses 0.85** — treat ZH as secondary/completeness, judged at G3 with the same rule.

**Primary bets: HateMM + MHC-EN** (clean 3-seed frozen-Qwen floors). MHC-ZH carried for completeness.

### E-G0 — static feasibility (cost: 0 GPU)
- **Checks:** (a) 7B checkpoint complete (§5 ✓); (b) VRAM fits (§5 ✓); (c) all Stage-1 data + frames +
  ASR present for the target dataset (§5 ✓); (d) `run_rac_lmm.py` accepts the video LoRA-feature cache
  contract (verify the `load_feats_from_LMM` loader shape matches `generate_VideoMLLM_embedding_lora_HF.py`
  output — the one plumbing edit needed); (e) a **λ/no-op discipline plan** (Stage-2 with the RGCL term
  off must reproduce a raw-kNN read-out to 4 dp, mirroring enc3seed's code-version audit).
- **Kill:** any of (a)-(d) fails and cannot be fixed with a localized, reviewed edit → line does not
  reach GPU. (Deliverable: a static-feasibility note; this gate is the zero-GPU analogue of TARC G0.)

### E-G1 — single-seed single-dataset smoke (cost: ~1 hr, 1 GPU; HateMM, seed 0)
- **Run:** one full C1 cell on **HateMM seed 0** at the RA-HMD default rank AND one low-rank arm
  (r ∈ {8,16}) — Stage-1 QLoRA → extract adapted features → Stage-2 RGCL head+kNN.
- **Minimum-viable-signal (pre-declared):** the C1 **DEV** (val) acc must be **≥ frozen-Qwen DEV floor
  − 0.010** (i.e. not a regression beyond one noise band) on **either** rank arm, AND the Stage-2 RGCL
  loss must train non-degenerately (positive/negative gap does NOT collapse to ln2 — the explicit
  P9b bs-degeneracy check, `EXP_p9:259-266`; here Stage-2 is frozen-feature so bs is unconstrained,
  but the guard is retained).
- **Kill:** C1 DEV acc < frozen DEV floor − 0.010 on both rank arms, OR the RGCL loss is degenerate →
  **kill the line** (this is P9-C3-knn reproducing below floor; no reason to spend E-G2 seeds).

### E-G2 — 3-seed paired train/val, BOTH protocols (cost: ~6 hr, 1 GPU; HateMM + MHC-EN)
- **Run:** C1 vs frozen-Qwen floor, **seeds 0/1/2, paired within seed**, best rank arm from E-G1,
  on **HateMM and MHC-EN**, judged on **DEV/val** under **both** protocols (A = val-selected warmup≥5;
  B = final-epoch 29) — identical parser to enc3seed.
- **Pass (pre-declared, per dataset × protocol):** 3-seed mean paired **Δacc ≥ +0.015 AND ΔF1 ≥ +0.015
  vs the frozen-Qwen floor, sign ≥ 2/3 positive**, on **≥ 1 primary dataset** under a stated protocol.
  (+0.015 on val is the minimum that could plausibly survive to a +0.030 test effect; it is judged on
  **val** to preserve the single test-touch for E-G3.)
- **Kill:** no primary dataset clears +0.015 val under either protocol → **kill the line** (C1 joins the
  P9/P9b completion negative). Report the honest verdict; do not proceed to test.

### E-G3 — single test touch, both primary datasets, +0.030 bar (the ONE sanctioned read)
- **Run:** the E-G2-surviving config, **HateMM and MHC-EN** (+ MHC-ZH for completeness), seeds 0/1/2,
  C1 vs frozen-Qwen floor, both protocols, judged by the **exact enc3seed decision rule**
  (`exp-encoder-3seed.md:73-85`).
- **Pass (pre-registered campaign bar):** mean paired **Δacc ≥ +0.030 AND Δmacro-F1 ≥ +0.030 AND sign
  3/3 positive**, on **≥ 1 dataset**, under a stated protocol (each protocol judged separately).
- **Test-touch budget:** this is the **only** time the C1 test set is read (val used for all E-G0/1/2
  selection). One serial sbatch. After E-G3, no further C1 test read.
- **Kill/verdict:** neither dataset passes → C1 killed, pre-registered negative, archived like TARC.

**Single-submit-per-lineage ceremony note.** Each gate = one serial sbatch, pre-registered before
submit; no adaptive re-submission on the same lineage after seeing test. Code touching the LMM
forward / Stage-2 loss routes through `codex-code-review` before any GPU submit (project convention,
same as TARC G1). **C-line queues behind A-line M2/M3** — do not contend for the 2-GPU cap while
A-line's decision run is live.

---

## 8. Anti-overfitting clauses (from LITERATURE §4 / SAV warning)

The SAV paper warns tiny-data LoRA overfits (`LITERATURE_mllm_integration_2026-07-13.md:69`).
With 744/549/579 train videos vs RA-HMD's ~8,500 memes, C1 pre-commits:
1. **Low rank primary.** Sweep r ∈ {8, 16, 32} in E-G1; carry the *lowest* rank that meets the
   minimum-viable-signal, NOT r128 (which RA-HMD tuned on 10× the data). r128 is an upper-bound arm only.
2. **Stage-2 backbone freeze (RA-HMD-faithful).** The RGCL contrastive term trains only the MLP+head on
   **frozen** adapted features (`run_rac_lmm.py` / `qwen2vl7b-contrastive.sh`), never end-to-end — this is
   also why C1 avoids P9b's joint-training bs-degeneracy.
3. **No-selection protocol primary.** Final-epoch (protocol B) is the primary judged protocol at every
   gate; val-selected reported alongside but the 78-item MHC dev is known to inject ~2 acc pts of
   selection noise (TARC §12.3), so a val-selected-only pass is treated as a selection artifact.
4. **3 epochs max Stage-1** (RA-HMD default), early-checkpoint monitoring for the LM/cls loss divergence
   that flags backbone overfit.

---

## 9. Where this line most likely dies (honest prior)

**Most likely: E-G1 or E-G2, via H0 + §3.3 + §4.** The frozen-Qwen encoder swap already banked the
representation gain on HateMM; §3.2 shows RA-HMD's retrieval-contrastive Stage-2 adds only ~+0.7 acc
in-domain; and P9's Stage-1-LoRA MLP read-out did not beat the frozen floor on HateMM (+0.9, noise),
while its raw-kNN read-out went *below* floor (−4.7). C1's one hope is that **training** the RGCL head
on adapted features (which P9 omitted) recovers the memory read-out above the frozen floor. Point
estimate: **HateMM lands within ±0.01 of the frozen-Qwen floor (flat), MHC-EN flat-to-negative, MHC-ZH
already-saturated.** The campaign's +0.030 test bar has been met **once** (frozen encoder swap, HateMM
only, `exp-encoder-3seed.md:228`); C1's realistic ceiling on HateMM is that same frozen result, not
beyond it. **The value of running C1 is a clean, complete negative** (or, on the tail, the first video
result that beats frozen — which would satisfy the goal). Either way it costs ~1 day of 1-GPU time,
gated cheaply at E-G1.

---

## 10. Status / next step

**DRAFT-UNREVIEWED.** Route to a fresh 0-context reviewer (ceremony), then a user-visible report.
No submission until the user rules, and only in the gaps behind A-line M2/M3. The single most
important thing the reviewer must decide: **is the "open cell" in §4 a real methodological distinction
from P9/P9b, or a distinction without a difference?** If the latter, C1 should not spend GPU and the
P9/P9b negative stands as the final word on video LMM-RGCL at 7B.
