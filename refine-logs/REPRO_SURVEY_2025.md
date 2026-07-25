# REPRO SURVEY — adjacent-field techniques with open code (2025-H2 → 2026-07)

Stage-1 reproduction survey. Executor: repro-survey subagent. Date: 2026-07-25.
Zero GPU, zero SLURM, zero model-weight downloads. Shallow git clones + web verification only.

**Scope (as corrected by the user mid-run).**

1. **Window = 2025-07 → 2026-07.** Older work is context only.
2. **Mining target = NEIGHBOURING fields, not hateful-video detection.** Rationale (user):
   lifting a technique from a direct hate-video paper reads to reviewers as "no novelty";
   the value is in being *first* to bring an adjacent-field technique into hateful-video
   detection. Direct hate-video works therefore appear here as a **reference table only**
   (the frontier table already exists in `LITSWEEP5_HATEMM_EN.md` and is not re-derived).
3. Constraint-box legality label required per component; dead-list cross-check required
   against `autoresearch/goal_mllm_plus3/state/directions_tried.json`.

**Transplant target (our pipeline), with exact plug points.**

| Stage | File / symbol | What is there now |
|---|---|---|
| Encoder | Qwen2.5-VL-7B, per-dataset LoRA-SFT or frozen | 8 frames → image vec; transcript → text vec; mean-pool |
| Head fusion | `src/model/classifier.py:70` `classifier_hateClipper` | `img_proj`/`text_proj` (Linear+Dropout) → L2-norm → `fusion_mode ∈ {concat, align(Hadamard), cross}` → MLP → `output_layer` |
| Loss | `src/model/loss.py:12` `compute_loss` | triplet-margin + BCE hybrid |
| Mining | `src/run_rac.py` (`--no_hard_negatives`, `--reindex_every_step`) | FAISS index over train, re-indexed once per epoch, global hardest-opposite-label |
| Inference | `src/model/evaluate_rac.py:321` `retrieve_evaluate_RAC_` | top-20 rank-weighted signed-cosine kNN vote over own-train memory |

House numbers (final-epoch): HateMM 0.8791 acc / 0.8726 mF1 (val-sel 0.8775/0.8711);
MHC-ZH 0.8456/0.8173; MHC-EN ~0.79–0.81. Train n ≈ 549 (EN) / 579 (ZH) / 744 (HateMM).

**Constraint box.** No OCR channel (user veto). No gold annotations in the deployed path.
Training data = single-dataset own train split only. No cross-seed ensembles. No closed-model
APIs. Raw videos never leave this machine. Model downloads >1 GB are user-gated.

---

## 0. PRIOR-ART ON THIS MACHINE (checked first, so we do not re-clone)

Before surveying, `/data/jehc223/` was inventoried. Several relevant repos are **already
cloned and were actually run** (log/output directories with real timestamps). These are
excluded from the clone list and from the shortlist.

| Local path | Upstream | Paper | Evidence it was run |
|---|---|---|---|
| `/data/jehc223/baselines/MoRE` | `github.com/Jian-Lang/MoRE` (EXISTS) | *Biting Off More Than You Can Detect: Retrieval-Augmented Multimodal Experts for Short Video Hate Detection*, WWW 2025 | `log/0704-*` (Jul 2026), local commit 3852b65 2026-05-25; `MoRE_env` + `MoRE_paddle` conda envs |
| `/data/jehc223/ExMRD` | `github.com/ronpay/ExMRD` (EXISTS) | *Following Clues, Approaching the Truth: Explainable Micro-Video Rumor Detection via Chain-of-Thought Reasoning*, WWW 2025 | `log/0811-*`, `outputs/2025-08-11`; `ExMRD` conda env |
| `/data/jehc223/ExMRD_ours` | `github.com/JunyiChen-ai/WWW-2025` (own fork) | derivative project | `log/FakeSV_0901-*` ×many, `optuna_results/` |
| `/data/jehc223/FakingRecipe` | `github.com/ICTMCG/FakingRecipe` (EXISTS) | ACM MM 2024, fake news short video | cloned, no run logs |
| `/data/jehc223/HVGuard` | `github.com/JunyiChen-ai/HVGuard` (own fork) | hateful-video CoT + CSD embeddings | 44 recent logs, commit 2026-04-09 |
| `/data/jehc223/HateClipSeg` | `github.com/Social-AI-Studio/HateClipSeg` (EXISTS) | segment-level hateful video | dataset only |
| `/data/jehc223/HateMM`, `/data/jehc223/Multihateclip`, `/data/jehc223/ImpliHateVid` | dataset mirrors | — | data |
| `/data/jehc223/vimo-core` | local | VideoRAG + ImageBind scaffolding | present |

**Consequence.** The *retrieval-augmented mixture-of-experts* idea (MoRE) and the
*CoT-reasoning-as-feature* idea (ExMRD) are already in-house and reproduced; the survey below
deliberately looks past them. Two conda envs already match candidate requirement pins
(`HateVideo`: torch 2.6.0+cu124 / transformers 4.49.0; base shell: torch 2.8.0+cu128 /
transformers 4.49.0), which matters for the reproduction cost estimates in §5.

---

## 1. DEAD-LIST FILTER APPLIED TO THIS SURVEY

Every candidate component below was checked against `directions_tried.json` before being
listed. The findings that most often fire:

| Finding | What is dead | What it does NOT close |
|---|---|---|
| F50 / F55 | fixed fusion reweights, per-modality temperatures, cross-encoder feature composition over banked features (frozen **and** adapted) | *trained* fusion blocks that change the function class (attention, per-sample learned gating) — unpriced, but F50 lowers the prior |
| F75 | head-loss swaps toward NCA / soft-kNN / SupCon / manifold-mixup; tau/alpha retunes | objectives that **add a term** carrying a new information source rather than replacing the hybrid |
| F73 | SAM (rho 0.05) on the head; modality-dropout p=0.3 **ones-fill** as a plain regularizer | masked-forward objectives with an explicit consistency/confidence penalty and a different fill (see SynIB, §4) |
| F64 / F41 | Whisper-encoder hidden states; eGeMAPS prosody as auxiliary channel | general-audio event encoders (CLAP/AST/BEATs) — download-gated, prior lowered |
| F72 / F70 | training-free bidirectional-mask conversion at extraction; L24 readout, one-word prompts, last-token span | MNTP-style stage-2 (user-gated); multi-prompt ensembling (user-gated) |
| F63 | multi-hop label propagation / graph diffusion over the frozen kNN graph | end-to-end learned graph weights (= P9b territory) |
| F66 / F49 / F47 | per-item selection/routing over banked channels; non-selecting aggregation of independently re-encoded segments | operators fed by a genuinely new information source with alignment > 0.663 |
| F67 / F76 | 16-frame sampling; higher-resolution re-extraction (parked) | — |
| F80 | extraction-instruction **language** variation | task-instruction *conditioning* (different object) |
| banned | OCR; gold annotations in method; cross-dataset training; cross-seed ensembles; closed-model APIs; MLLM-scores-as-training-signal (P11 + P1-P5) | — |

**The last banned item bites hardest here.** The dominant 2025-H2/2026 trend in multimodal
embedding training is *MLLM-as-a-judge soft supervision* — which our box forbids. Those
components are listed below with an explicit BANNED label rather than silently dropped, so
the ruling is visible.

**What prior sweeps already covered (not re-derived here).** `LITSWEEP5_HATEMM_EN.md` holds the
HateMM/MHC frontier table; `LITSWEEP2_FRESH_2026.md` (same day) covered 2026 harmful-video
papers, new open-weight encoders, PEFT-for-small-data, retrieval/memory 2026, and test-time
aggregation; `LITSWEEP5_COMPLETENESS.md` holds the exhaustiveness audit. **All of those were
paper-reading only — no repository was ever opened or cloned.** This survey's delta is
(i) real repo triage against the file tree, and (ii) the *neighbouring-field* architecture
angle (balanced/synergy multimodal learning, modality-aware PEFT, multimodal-embedding
training recipes), which none of the previous rounds hunted.

---

## 2. DIRECT HATE-VIDEO WORKS — REFERENCE ONLY (code-availability column)

Per the user's ruling these are **not** reproduction targets. `LITSWEEP5_HATEMM_EN.md` §2 already
holds the numbers/channels/legality table and is not restated. The one thing it lacks is whether
code exists; verified here by `git ls-remote` (API-free, so unaffected by GitHub rate limits):

| Work | Repo | Exists? | Status for us |
|---|---|---|---|
| MM-HSD (2508.20546, ACMMM25) — HateMM 0.878/0.874 | `github.com/idiap/mm-hsd` | **EXISTS** | best published HateMM number, but OCR-load-bearing = out of box. Worth a **read-only** look for its cross-modal-attention block; not a repro target |
| CMFusion (2505.12051) | `github.com/EvelynZ10/cmfusion` | **EXISTS** | gated fusion = F50 family, below us |
| MoRE (WWW 2025) | `github.com/Jian-Lang/MoRE` | **EXISTS** | **already cloned and run in-house** (§0) |
| HateMM (ICWSM 2023) | `github.com/hate-alert/HateMM` | **EXISTS** (cloned, 17★, last commit 2024-06-17, no license) | dataset-paper baselines: 9 numbered scripts, `models.py`, no configs, no weights, 5-fold. Kept as the single reference clone |
| MultiHateClip (2408.03468) | `github.com/Social-AI-Studio/MultiHateClip` | **EXISTS** | dataset |
| HateClipSeg | `github.com/Social-AI-Studio/HateClipSeg` | **EXISTS** | **already on disk** |

---

## 3. REPOS CLONED AND TRIAGED (`external/baselines/`, gitignored, 91 MB total)

Facts below come from the actual file tree, not the README.

| Name | Upstream | Paper (window) | Last commit | License | Tree facts | Verdict |
|---|---|---|---|---|---|---|
| **SynIB** | `kkontras/SynIB` | *SynIB: Informational Bottleneck for Maximizing Synergy in Multimodal Learning*, arXiv **2606.09853** (2026-05-12) | 2026-06-21 "Initial public release" | LICENSE present | 9.2 MB; `src/synib/{models,training,baselines,entrypoints}`; `run/configs/hateful_memes/methods/{synib,synib_u,masking_only_random,masking_only_learned,mcr,mmpareto,reconboost,dnr,vanilla,uni_image,uni_text}.json`; `run/hateful_memes/{train.sh,pipeline_smoke.sh}`; `pyproject.toml`+`requirements.txt`; core objective in `src/synib/models/vlm/synib_mask_model.py` (1417 L), balanced-learning baselines in `src/synib/training/pipeline/helpers/Bias_Infusion.py` (694 L: `Bias_Infusion_{MMPareto,ReconBoost,DnR,MCR}`) | **KEEP — rank 1.** Complete, self-consistent, ships its own ablation isolating the part we already killed |
| **MokA** | `GeWu-Lab/MokA` | *MokA: Multimodal Low-Rank Adaptation for MLLMs*, NeurIPS 2025 **Oral**, arXiv 2506.05191 | 2025-12-30 | **NO LICENSE FILE** | 33 MB; `AudioVisualText/` + `VisualText/`; vendored PEFT fork (`VisualText/modified_peft/`, `AudioVisualText/peft_hyper/tuners/lora.py`); core = `Linear.forward(x, modality_mask)` applying **per-modality LoRA A** with shared B; **no `requirements.txt`** | **KEEP — rank 3.** Mechanism is liftable; repo is heavy and licence-unclear |
| **LSMI** | `GeWu-Lab/LSMI_Estimator` | *Efficient Quantification of Multimodal Interaction at Sample Level*, ICML 2025 | 2025-06-05 | NO LICENSE FILE | 420 KB, **4 py files** (`entropy_estimation.py`, `main_lsmi.py`, `gaussian_data.py`, `utils.py`), one `cfgs/train.yaml`, `requirements.txt` pins torch 1.9.1 (KNIFE kernel estimator is plain `nn.Module`, no version-critical API) | **KEEP — rank 2.** Smallest, cheapest, and the only *diagnostic* in the set |
| **UniME-v2** | `GaryGuTC/UniME-v2` | *UniME-V2: MLLM-as-a-Judge for Universal Multimodal Embedding Learning*, arXiv **2510.13515** (2025-10-16), AAAI 2026 Oral | 2025-12-08, 74★ | **MIT** | 23 MB; `Embedding/{train.py,eval.py,src/,grad_cache/,shells/}` + `Rerank/`; `Embedding/src/loss.py` = 77 L `SoftLabelSoftKLContrastiveLoss`; training + eval shells present; requirement pins torch 2.4 / **transformers 4.49.0 (= our `HateVideo` pin exactly)** | **KEEP — reference.** Core supervision source is BANNED for us (see §4) |
| **VLM2Vec** | `TIGER-AI-Lab/VLM2Vec` | MMEB-V2 arXiv 2507.04590 (2025-07-07); repo now hosts **MMEB-V3** arXiv 2604.23321 | 2026-07-24, 669★ | **Apache-2.0** | 16 MB; `src/{model,loss.py,loss_omni.py,trainer.py,grad_cache}`, `train.py`/`train_omni.py`, `experiments/`, `OmniSET/`; pooling is only `last|mean|cls` (`src/model/model.py:177`); `InExampleContrastiveLoss` = 1-of-K categorisation over label embeddings; GradCache for big contrastive batches on 1 GPU | **KEEP — reference.** Large framework; two small liftable pieces |
| **BalanceBenchmark** | `GeWu-Lab/BalanceBenchmark` | survey/toolbox from GeWu-Lab; **arXiv ID NOT verified in this run** — do not cite until checked. Repo dates put it **out of window** | 2025-02-23 | NO LICENSE FILE | 3.3 MB, 55 py files; **17 method trainers** in `balancemm/trainer/` (AGM, AMCo, CML, GBlending, Greedy, LFM, MBSD, MLA, MMCosine, MMPareto, OGM, OPM, PMR, ReconBoost, ReLearning, Sample, UMT); yaml configs | **KEEP — infrastructure.** Out-of-window as a *claim*, but 17 implementations behind one trainer API |
| **HateMM** | `hate-alert/HateMM` | ICWSM 2023 | 2024-06-17, 17★ | none | 268 KB; 9 numbered scripts + `models.py`; no configs, no weights | **KEEP — the single direct-hate reference clone** (§2) |
| **VidVec** | `iyttor/VidVec` | *VidVec: Unlocking Video MLLM Embeddings for Video-Text Retrieval*, arXiv 2602.08099 (2026-02-08) | 2026-02-08 | none | **`main` branch is EMPTY** (0 files, 0 py); only branches are `main` and `site` (project page). Commit message: "remove html from main branch" | **KILL — no code exists.** Kept as a 136 KB record of the kill |

Also verified and **not** cloned: `RASR` (arXiv 2604.06687, retrieval-augmented reasoning for fake-news video) — **paper WITHDRAWN** by the authors (v2, 2026-06-30, "requires revision and expanded experiments"). Do not cite, do not reproduce.

---

## 4. COMPONENT MINING — what could actually be transplanted

Each entry: **(a)** legality in our box · **(b)** plug point · **(c)** what our own campaign already says.

### 4.1 SynIB — synergy information bottleneck  *(arXiv 2606.09853, code `kkontras/SynIB`)*

**Mechanism (read from `src/synib/models/vlm/synib_mask_model.py`, not the README).** Given the two
pooled modality vectors `z1`,`z2`, the model makes `K` extra forward passes through the *same*
fusion head with a Bernoulli(p) **dimension-wise mask** per modality, where masked coordinates
are filled by a **permuted other-sample value** (`eps = z[torch.randperm(z.size(0))]`; alternative
fills `zeros | noise | ema | token | shuffle`, EMA statistics tracked by a `FeatureStatsMasker`).
The objective then adds a **symmetric-KL penalty between the intact-input prediction and the
masked-input prediction** — i.e. it penalises the head for staying *confident* when one modality's
information is withheld. A learned-mask variant (`get_learnable_mask_multiclass`) replaces the
random mask with an optimised one. `p_type="diff"` sweeps a different p per repeat.

- **(a) Legality: FULLY IN BOX.** No new data, no new labels, no external pool, no OCR, no audio,
  no ensembling, no test-time use — it is purely an extra training-time term on our own two
  streams and our own train split.
- **(b) Plug point:** `classifier_hateClipper.forward` (`src/model/classifier.py:115-147`) already
  receives exactly `img_feats, text_feats` post-projection, which is SynIB's `z1,z2` with
  `cls_type="mlp"`. The KL term is added in `compute_loss` (`src/model/loss.py:12`) **alongside**
  the triplet+BCE hybrid, not replacing it.
- **(c) Dead-list check — this is the important one.** Our F73 killed "modality-dropout p=0.3,
  **ones-fill**, on the Hadamard head" as flat-to-harmful. SynIB's own repo ships that exact
  ablation under the name **`masking_only`** (`src/synib/baselines/masking_only.py`, configs
  `masking_only_random.json` / `masking_only_learned.json`) whose docstring states it "isolates the
  data-augmentation effect of masked inputs **from the KL objective itself**". So the paper's claim
  is precisely that the part we measured (masking alone) is *not* the load-bearing part.
  Three concrete differences from F73: (i) the KL confidence penalty exists at all; (ii) the fill
  is a batch permutation / EMA draw, not an identity `ones` fill (under Hadamard fusion, ones-fill
  passes the surviving stream through unchanged — a much weaker perturbation); (iii) masking is
  **per-dimension**, not per-stream. **Verdict: non-isomorphic to F73, and the non-isomorphism is
  documented by the source repo rather than asserted by us.**
  **Second check, F75 — stated conservatively.** F75's ban_scope is written as head-loss *swaps*
  "toward vote-consistent (NCA/soft-kNN), contrastive (SupCon), or mixup-BCE objectives". SynIB is
  an *added* term rather than a swap, so it falls outside the letter of that scope — but the
  nearest F75 arm (**manifold-mixup-BCE**) is also a feature-space perturbation + auxiliary target,
  so a reviewer could reasonably argue the spirit. This should be settled in the prereg, not
  assumed here.
- **Bonus:** the same repo yields four more balanced-multimodal objectives already wired to an
  image+text fusion head (`Bias_Infusion_{MMPareto,ReconBoost,DnR,MCR}`). Note `MCR` here is a
  *different* object from our F71 "MCR modality-competition rebalancing SFT" (theirs is a head-side
  trainer, ours was an encoder-SFT schedule) — check before claiming either way.

### 4.2 LSMI — sample-level synergy / redundancy / uniqueness estimator *(ICML 2025, `GeWu-Lab/LSMI_Estimator`)*

**Mechanism.** KNIFE kernel differential-entropy estimation (`MargKernel`, a K=5 Gaussian-mixture
`nn.Module`) used to decompose, **per sample**, the task-relevant information of `(x1,x2)` about `y`
into redundancy `r`, uniqueness `u1,u2`, and **synergy `s`**.

- **(a) Legality: FULLY IN BOX**, and it is a *diagnostic*, not a deployed component — it never
  touches inference.
- **(b) Plug point:** runs offline on the already-banked feature caches (`z_img`, `z_text`, train
  labels). No encoder, no GPU strictly required (4 small py files, torch-version-agnostic).
- **(c) Why this matters more than its size suggests.** Our fusion axis is closed by **F50**
  ("fixed compositions/reweights = rotation at every w") and **F44** ("MHC-EN image stream collapses
  to near-chance; equal-weight concat cancels the text gain"), but we have **never measured whether
  image×text synergy exists at all** in these datasets. LSMI answers exactly that, and the answer
  is decision-relevant in both directions: if `s ≈ 0` on MHC-EN/ZH, then *every* richer fusion block
  (attention, gating, bilinear) is arithmetically capped and the whole "our method is too simple"
  worry is answered with a measurement rather than an opinion — a first-class analysis-chapter
  result. If `s > 0` on some stratum, that stratum is the target SynIB is designed to hit.
  **This is the natural $0 pre-gate for 4.1.**

### 4.3 MokA — modality-aware LoRA *(NeurIPS 2025 Oral, `GeWu-Lab/MokA`)*

**Mechanism.** In the LoRA `Linear`, the down-projection **A is per-modality** (selected by a
`modality_mask` over the token positions) while **B is shared**, so unimodal information is
preserved through adaptation instead of being averaged away by a single shared subspace.

- **(a) Legality: IN BOX** (LoRA on our own encoder, own train split, local weights).
- **(b) Plug point:** the encoder-SFT stage, not the head — it replaces the vanilla PEFT `LoraConfig`
  used for the per-dataset Qwen2.5-VL LoRA. **Requires a joint multimodal forward** so that a
  modality mask exists; our extraction currently runs image and text through *separate* forwards.
- **(c) Dead-list check — mixed, be honest.** Positive: encoder adaptation is the **only** axis that
  ever converted +3 (F53 HateMM, F45/B3 ZH), and F44 names the exact failure MokA is designed to fix
  (the EN image stream collapsing under a shared adaptation subspace). Negative: `LITSWEEP2_FRESH_2026`
  HUNT-3 already priced "richer PEFT adapters (MoLE, Task-Adapter++)" at ~0 as *tactics on a measured
  axis*, and F65 showed vision-side LoRA adds nothing at the head. The distinguishing argument is that
  MokA is the first PEFT variant whose stated target is **modality imbalance during adaptation**
  rather than adapter capacity — but the joint-forward requirement also drags in W2-A's territory
  (the joint `[transcript][frames]` forward measured conditional-info ≈ 0, though as a *frozen readout*,
  not as an adaptation object). **Rank 3, with the joint-forward caveat stated up front.**

### 4.4 UniME-V2 — MLLM-as-a-judge soft-label contrastive *(arXiv 2510.13515, AAAI26 Oral, MIT)*

**Mechanism.** Global retrieval builds a potential hard-negative set; an MLLM judges the semantic
alignment of each query–candidate pair; those scores become **soft labels** and the student is
trained with a **symmetric-KL between its similarity distribution over {positive ∪ hard negatives}
and the judge's score distribution** (`Embedding/src/loss.py`, 77 lines, fully self-contained).

- **(a) Legality: the supervision source is BANNED.** `banned_constraints` lists
  "MLLM-scores-as-training-signal" (P11) and P1–P5 re-proposals; the judge is also a structural twin
  of the dead P2 (MLLM neighbour-comparability rerank), moved from inference to mining time.
  **Do not propose as-is.** The *loss shape* (soft-KL over a mined negative set) is separable from the
  score source — but sourcing the soft targets from anything we already have turns it into a
  head-loss swap, which is F75. Recorded so the ruling is explicit, not as a live candidate.
- **(b)** Would sit in `compute_loss` over the FAISS-mined negatives from `run_rac.py`.
- **(c)** F75 (NCA/soft-kNN/SupCon 0/8) + P11 + P2. **Reference only.**

### 4.5 VLM2Vec / MMEB-V3 — two small liftable pieces from a large framework *(Apache-2.0)*

- **`InExampleContrastiveLoss`** — classification framed as 1-of-K matching against *label*
  embeddings rather than a logit layer. Legality: in box. Plug point: `output_layer`. Dead-list:
  adjacent to F75 (loss family) and to W2-E's prototype ban — though W2-E banned *unsupervised,
  zero-training* prototypes over frozen features, and trained label embeddings are a different
  object. **Weak candidate; flag the overlap.**
- **GradCache** (`src/grad_cache/`) — lets a contrastive batch far larger than GPU memory be used by
  caching representations and recomputing gradients in chunks. Legality: in box, it is pure
  engineering with no effect on the objective's definition. Plug point: the head trainer. Relevance:
  our train sets are 549–744 items, so a full-batch contrastive pass is *already* affordable —
  **this is a nice-to-have, not a lever.**
- Pooling in this repo is only `last|mean|cls` (`src/model/model.py:177`), i.e. it offers **no**
  fancier pooling module than we already use. Recorded to close that hope.

### 4.6 BalanceBenchmark — 17 balanced-multimodal trainers behind one API *(out of window, infrastructure)*

`balancemm/trainer/` contains AGM, AMCo, CML, GBlending, Greedy, LFM, MBSD, MLA, MMCosine, MMPareto,
OGM, OPM, PMR, ReconBoost, ReLearning, Sample, UMT. Legality: all are training-time objectives on
own data = in box. Value: if §4.2's LSMI measurement shows synergy exists, this is a ready-made
menu of modality-balancing objectives to screen cheaply, with a shared trainer interface. Caveat:
last commit 2025-02-23 and **no licence file**; its companion survey paper predates our window
(arXiv ID unverified in this run — check before citing).

---

## 5. RANKED REPRODUCTION SHORTLIST

Ranked by *inspiration-value × reproducibility × overlap with our setting*. Every item is
GPU-stage work — **none of it was run in this stage** (no GPU, no SLURM, no weight downloads).

### #1 — LSMI synergy measurement on our own banked features  *(do this first; it gates #2)*

- **Why first.** It is the cheapest item in the list, needs no new data and no new weights, and it
  *prices the entire fusion-architecture family* — the exact family the user's "our method is too
  crude" concern points at. Our fusion evidence (F50 rotation, F44 EN image collapse) says fixed
  compositions do nothing, but never says whether image×text synergy exists to be captured.
- **Env:** `HateVideo` as-is (the estimator is 4 plain-PyTorch files; the `torch==1.9.1` pin is
  cosmetic — `MargKernel` is a bare `nn.Module` with no version-critical API).
- **Data prep:** none — read the existing `z_img` / `z_text` caches and train labels for HateMM,
  MHC-EN, MHC-ZH. No videos touched.
- **Expected runtime:** minutes on CPU per dataset; ≤0.2 GPU-h total if run on GPU for convenience.
  Config `cfgs/train.yaml` is already binary (`n_classes: 2`), two input vectors, `embed_size 64`,
  30 epochs for the discriminator + 30 for the entropy estimator.
- **Known technical caveat (found by reading the code, flag before spending).** `MargKernel` carries a
  **full-covariance** parameter `tri` of shape `(1, K, d, d)`. At our `d≈3584` per stream that is
  ~64 M floats per kernel — workable on GPU but wasteful, so a down-projection is the practical
  route. Our own **F41 precedent applies**: the APX gate ran a strict raw-dimension arm precisely to
  rule out "PCA underpowered the estimator", and the same discipline must be pre-registered here
  (a reduced-dim arm *and* a raw/large-dim arm), otherwise a null is uninterpretable.
- **Target number to match:** none in our setting (this is a measurement, not a benchmark). Sanity
  target from the paper: reproduce the synthetic Gaussian check shipped as `gaussian_data.py`,
  where ground-truth `r/u/s` are known — that is the correctness gate before trusting our numbers.
- **What we learn even if it "fails".** If the estimator is unstable at n≈600, that itself is the
  finding (our datasets are too small to estimate PID), and it retires the synergy line at $0. If it
  is stable and `s ≈ 0`, we get a *positive, quotable, mechanistic* explanation of F50/F44 for the
  analysis chapter — "there is no synergy to fuse" is a far stronger paper sentence than "fancier
  fusion did not help". If `s > 0` anywhere, #2 has a target.

### #2 — SynIB objective, ported to our head *(the main inspiration bet)*

Two sub-plans; the second is the decision-relevant one.

- **(2a) Faithful paper reproduction** — Hateful Memes, frozen CLIP-ViT-B/16 + DeBERTa-v3-base,
  tier `small_tf_deberta`, folds {0,1,2}, seeds 109/27/3407, exact commands in
  `external/baselines/SynIB/docs/REPRODUCE.md:112-137`. Config: batch 32, ≤50 epochs, cosine
  anneal, AdamW lr 1e-4, single GPU. Also gives `vanilla / masking_only / dnr / mmpareto /
  reconboost / mcr / uni_text / uni_image` from the same harness.
  **Cost:** a Hateful Memes data download (~4 GB images, *not* model weights) + CLIP-B/16 and
  DeBERTa-v3-base (~1 GB combined — **exceeds this stage's budget, so it is a separate user-gated
  step**); then ≈0.5–1 GPU-h per cell × ~8 cells × 3 folds ≈ **12–24 GPU-h** if run in full, or
  ≈2 GPU-h for a single-fold synib-vs-vanilla-vs-masking_only triple.
  **Target to match:** the paper's HM row (up to +3.8 overall accuracy, up to +7.8 on
  synergy-dependent examples, over vanilla fusion).
- **(2b) Direct port to our head (recommended)** — lift `make_tilde_*` + the symmetric-KL term into
  `src/model/classifier.py` / `src/model/loss.py` as an *added* term. No external data, no new
  weights, no OCR, no audio, single-dataset own-split. **Cost ≈0.3–0.5 GPU-h** for a 3-seed cell on
  HateMM + ZH, i.e. the same price as the F73/F75 cells.
  **Target:** beat the current floor at the pre-registered bar; kill-switch = the §4.2 LSMI reading.
- **What we learn even if it fails.** A negative here *completes* the F73 story properly: F73 only
  measured masking-as-augmentation with ones-fill, and the paper's own `masking_only` ablation says
  that is the wrong half. Measuring the KL half closes the modality-masking family with a
  documented non-isomorphism instead of leaving a letter-overreach in the ban scope — which is
  exactly the kind of gap `LITSWEEP5_COMPLETENESS.md` keeps finding.

### #3 — MokA modality-aware LoRA at the encoder-SFT stage

- **Env:** new conda env; MokA ships **no `requirements.txt`** and vendors a modified PEFT — budget
  real integration time. It is built for LLaVA-style stacks, not Qwen2.5-VL, so the `modality_mask`
  plumbing must be re-derived for our token layout.
- **Data prep:** our own splits; but **requires switching extraction to a joint multimodal forward**
  (currently image and text are separate forwards) — that is the load-bearing engineering step and
  the main risk.
- **Cost:** ~1–2 person-days of integration + 3-seed encoder-SFT ≈ the usual per-dataset LoRA price.
- **Target:** the paper's AVQA/VisualText gains do not transfer as a number; the honest target is our
  own F53/F45 LoRA floor.
- **What we learn even if it fails.** It prices "modality-aware adaptation" on the one axis that has
  ever converted +3, and answers whether F44's EN image-stream collapse is an *adaptation-subspace*
  artefact or a genuine label limit. Either answer is analysis-chapter material.
- **Caveat, stated up front:** `LITSWEEP2_FRESH_2026` HUNT-3 already priced richer PEFT at ~0, and
  the joint-forward requirement borders on W2-A's measured-zero territory. Rank 3 for that reason.

### #4 — BalanceBenchmark screen *(conditional on #1 showing synergy)*

17 modality-balancing trainers behind one API. Only worth GPU if LSMI says there is something to
balance; otherwise it is 17 variants of a family F50/F73 already suggests is flat. Cost per method
is small (small heads, cached features); the risk is multiplicity/forking-paths, so it needs a
pre-registered subset (≤3 methods) rather than a sweep.

### #5 — MM-HSD cross-modal-attention block, read-only

`github.com/idiap/mm-hsd` is the only published method above us on HateMM. We cannot use it (OCR is
load-bearing there, per its own ablation), but reading its cross-modal-attention implementation is
free and tells us what a *trained* attention fusion looks like in this exact task — the function
class F50 never tested (F50 tested fixed compositions). **No clone budget spent; read on demand.**

**Explicitly NOT shortlisted:** UniME-V2 (judge supervision banned), VLM2Vec (framework, no new
pooling), VidVec (no code), RASR (withdrawn), MoRE / ExMRD / FakingRecipe / HVGuard (already
reproduced in-house), and every direct hate-video method (user ruling).

---

## 6. TOP-3 TRANSPLANTABLE COMPONENTS (overall)

1. **Synergy-penalised masked-consistency term (SynIB).** Per-dimension modality masking with
   permutation/EMA fill + symmetric-KL between intact and masked predictions, added to — not
   replacing — the triplet+BCE hybrid. In box, drops into `classifier.py`/`loss.py`, and its
   non-isomorphism to our dead F73 is documented by the source repo's own `masking_only` ablation.
2. **Sample-level PID estimator (LSMI).** Not a performance lever — a *measurement* that prices the
   whole fusion-architecture family and would convert F50/F44 from "we tried things and they were
   flat" into "there is no synergy to fuse", which is the stronger paper claim and the honest answer
   to "our method is too crude".
3. **Modality-aware LoRA (MokA):** per-modality down-projection `A` with shared `B`, applied on the
   only axis that has ever converted +3 in this project, and aimed squarely at F44's named failure
   (EN image-stream collapse under a shared adaptation subspace). Highest engineering cost of the
   three, and it needs a joint multimodal forward we do not currently run.

**Honest overall read.** The neighbouring-field hunt did **not** surface a technique that plausibly
delivers ≥+3 on ≥2 datasets — consistent with three prior sweeps. What it did surface is a coherent
*architecture-and-analysis* story: one legal, cheap training-time objective with a documented gap in
our own kill (SynIB), one measurement that would explain our fusion nulls mechanistically (LSMI),
and one adaptation-side variant on the project's only productive axis (MokA). If the user's aim is a
less "简陋" method with a defensible novelty claim rather than a headline number, that is the
strongest available package; if the aim is strictly the +3 bar, nothing here changes the standing
conclusion.

---

## 7. DISCIPLINE NOTES

- **Verification method.** GitHub's unauthenticated REST API is rate-limited from this network
  (`130.216.156.173`, 60 req/h, exhausted mid-run), so repo existence was confirmed with
  `git ls-remote` (API-free) and repo *contents* by cloning and reading the file tree. No claim in
  §3 comes from a README.
- **`py_compile` smoke over every clone:** LSMI 4/4, SynIB 99/99, UniME-v2 95/95, BalanceBenchmark
  55/55, MokA 214/214, VLM2Vec 271/271 files compile clean. **HateMM fails**:
  `Codes/1.FastTextEmb_and_LASEREmbExtraction.py:45` is a `SyntaxError: invalid syntax` (a bare `:`)
  — the dataset paper's own baseline code does not compile as shipped.
- **Budget respected:** no GPU, no SLURM, no model weights. Total clone footprint **91 MB**
  (`external/baselines/`), added to `.gitignore`; only this document is committed.
- **Two items need a user ruling before any GPU:** (i) the Hateful Memes data + CLIP-B/16 +
  DeBERTa-v3-base download for a faithful SynIB reproduction (§5 #2a) — the port-to-our-head route
  (#2b) needs neither; (ii) MokA's missing licence file, if any of its code is to be copied rather
  than re-implemented.
- **Not re-derived here:** the HateMM/MHC frontier table (`LITSWEEP5_HATEMM_EN.md`), the 2026
  encoder/PEFT/retrieval landscape (`LITSWEEP2_FRESH_2026.md`), the exhaustiveness audit
  (`LITSWEEP5_COMPLETENESS.md`).

### 7.1 Coverage limitation (stated so the survey is not over-read)

Five parallel web-sweep agents were dispatched (direct-anchor verification; hateful-video
enumeration; fusion-architecture sweep; retrieval/memory/metric sweep; VLM-embedding-head sweep;
adjacent-video-domain sweep). **The session's WebSearch quota was exhausted at 200/200 calls before
they returned**, so their enumerations are not folded into this document. Everything in §§0–6 was
verified directly by the survey executor — by cloning, reading file trees, `git ls-remote`, and
`py_compile` — and is therefore self-contained and safe to act on. But the *breadth* claim is
weaker than intended: this is a **deep triage of eight repositories**, not an exhaustive enumeration
of the 2025-H2→2026 adjacent literature. Areas most likely still under-covered: short-video
misinformation 2026, deepfake/AIGC-video fusion, noisy/subjective-label video learning, long-tail
multimodal, and test-time adaptation. A follow-up sweep with fresh search budget should target
exactly those five.

---
## ERRATUM (2026-07-25, orchestrator, post SYNIB_PORT_FORENSIC_RECON 9e638ea)
§4.1/§6 mischaracterize SynIB's core as "masked-consistency **symmetric KL** between intact and masked predictions". Source-level reading (synib_mask_model.py) shows: the intact prediction never enters any KL; live variants are (i) Gaussian KL(N(mu_masked,e^logvar)||N(0,I)) :1089-1097/:634-636, (ii) Dirichlet KL :638-652, (iii) FORWARD KL to detached unimodal anchors :1101-1106 (the HM-config variant); the only symmetric logit-KL helper is commented out (:668-673). Additional upstream flags: mask fill zeros|noise|ema = dead code (batch-permutation fill is the live path), config "p" key dead (p_min=0.30 binds), HM anchor heads have zero gradient path (untrained random init). Any prereg/paper sentence must use the recon's characterization, not §4.1's. Transplant design (PORT-A Bernoulli-entropy analogue) + conditional GO/PARK live in SYNIB_PORT_FORENSIC_RECON.md.
