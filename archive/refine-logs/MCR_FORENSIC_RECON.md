# MCR / MODALITY-COMPETITION REBALANCING — FORENSIC RECON (F68-P4)

**Author:** modality-competition-rebalancing forensic-recon subagent (CPU-only; **ZERO GPU / SLURM / Modal /
download / test-touch**; no `state/` mutation; **NO prereg authored, NO job submitted**). **Date:** 2026-07-25 NZST.
**Mission:** zero-GPU GO/NO-GO + execution skeleton for the **modality-competition rebalancing** cell —
transplant **MCR** (Kontras et al., NeurIPS 2025, arXiv 2411.07335, `github.com/kkontras/MCR`) and/or
**Data Remixing** (Ma/Chen/Deng 2025, arXiv 2506.11550) into our LoRA-SFT encoder adaptation, to force the
collapsed EN image stream to carry predictive load *during adaptation* (a loss/schedule object, not a capacity
object). Source: `refine-logs/LITSURVEY_NOVEL_MECHANISMS.md` §C1 (top-5 #1).
**Reads (verified this recon):** `LITSURVEY_NOVEL_MECHANISMS.md` §C1/§C5/§4; findings F24/F44/F45/F51/F55/F58/F64/F65/F66/F67
(`state/findings.jsonl`); `state/directions_tried.json` (`dead[]` 0–43, `banned_constraints[]` 0–8, `positives_bank[]`);
`VISION_UNFREEZE_FORENSIC_RECON.md` (F65, the load-bearing prior cell); `CAND2_CURRICULUM_RECON.md` (the curriculum-ban
reference); `LORA_HATEMM_PREREG.md` §2.2 (EN floors, F0.x house clauses); on-disk
`src/utils/build_lora_sft_data.py`, `src/utils/generate_VideoMLLM_embedding_lora_HF.py` (extraction prompt = decisive
for the transplant design), `RA-HMD/LLAMA-FACTORY-Ver202512/src/llamafactory/{data,train}`.

---

## 0. VERDICT (one line)

**GO-IF (conditional; user-gated) — NOT an autonomous perf bet, NOT a clean GO, NOT a hard NO-GO.** As a
*performance* bet the cell is **NO-GO**: its mechanism ("force the EN image stream to carry load during
adaptation") is the exact axis **F65 already exercised** — vision-LoRA *moved* the EN image stream (+0.032
trLOO) and converted **zero** head accuracy (K-V2 TIE everywhere) — and F55's oracle ceiling (+0.025 < +0.03
on EN) caps any stream-rebalancing; honest perf prior **~5–8%**, *below* the litsurvey's 10–20% precisely
because F65 is a same-axis null. It becomes **GO-IF** only if the user (a) opens a **D7 sub-ruling** (a
modality-balance *objective* is encoder-class, same D7 collision as generic LoRA — a pass is not automatically
novel), and (b) explicitly wants the *last mechanistic escape hatch on EN* closed for the paper — the
publishable-either-way null ("we rebalanced the training **objective**, not just the capacity, and EN still
did not convert" = label-limit confirmed at the strongest level). If GO-IF fires: run **ONE** EN-only arm,
**design (b) Data-Remixing-style modality-masked SFT schedule**, **~4–6 GPU-h**, gated by a cheap-after-extract
image-rebalance diagnostic. **HateMM and ZH are NOT run** (no mechanism fit — §4).

---

## 1. MECHANISM FIT — the honest transplant (the SFT is generative, not a joint classifier)

### 1.1 The load-bearing mismatch (why this is not a drop-in)

MCR's loss operates on **per-modality logits/gradients feeding a JOINT CLASSIFIER**: two encoders produce two
feature vectors, concatenated into a supervised head; MCR adds (i) a *perturbed-difference* term (latent
permutations + JS divergence measuring each modality's marginal effect on the output), (ii) a supervised-contrastive
alignment term, and (iii) a conditional-entropy-bottleneck term — driving each modality to maximize its
**conditional** predictive role `I(Xₘ;Y|X¬ₘ)`. **Data Remixing** is the cheaper cousin: decouple samples by
unimodal separability (KL), mask the **stronger** modality on the decoupled subset, re-assemble batches so
per-modality gradients align — forcing the weak modality to be *sufficient on its own subset*.

**Our stack does NOT match MCR's native home in two ways that determine the design:**

1. **The SFT objective is GENERATIVE word-label, not a joint classifier.** `build_lora_sft_data.py` emits a
   **single fused prompt** — `<image>×8 + instruction-with-transcript → "Yes"/"No"` — and LLaMA-Factory trains
   next-token CE on the label token. There is **no per-modality classifier logit** to apply MCR's
   perturbed-difference/JS term to. To obtain "image-only" and "text-only" label predictions you must run the
   forward with the *other* modality masked out of the prompt — **two extra forwards per step** (image-only,
   text-only) on top of the joint forward.

2. **MCR's *natural* transplant is our HEAD, and the head form is already DEAD.** The RGCL align-fusion head
   over cached `img_feats`/`text_feats` + kNN **is exactly** a two-tower-encoder + joint-classifier — MCR's
   home turf. But an MCR loss over **frozen** `{img_feats, text_feats}` is a reweight/rebalance of frozen
   streams, which is **subsumed by F55's oracle ceiling** (best per-item fusion of a *healthy* CLIP image +
   best text on EN = **+0.0250 < +0.030**) and **grazes F50** (fusion door closed) / **F49** (modality-reliability
   router). So the head-level MCR is dead on arrival. The litsurvey's routing to the **encoder-SFT** level is
   correct precisely because that is the *only* level not already closed — but at that level the transplant is
   generative and unnatural, per (1).

**Decisive extraction fact (governs design choice, verified in `generate_VideoMLLM_embedding_lora_HF.py:29–64,264–409`):**
the deployed encoder produces the two streams with **modality-isolated-ish** forwards — `img_feats` = 8 frames
+ a **neutral describe instruction, NO transcript** (pooled over the visual+instruction span); `text_feats` =
8 frames + title + transcript + an analytic instruction (pooled over the trailing text span). So the **image
stream is already extracted from an image-only-context forward**. This is the opposite of cand-2's problem: a
modality-masked (transcript-removed) SFT example has the **same prompt shape** the `img_feats` path is read
under — the masking is *train/deploy-consistent*, not a mismatch cost.

### 1.2 The three transplants, ranked by implementation risk on OUR stack

| arm | what it is on our stack | machinery delta | fwd cost/step | impl. risk | mechanism fidelity |
|---|---|---|---|---|---|
| **(c) modality-dropout SFT** | randomly drop the transcript from the joint prompt with prob *p* (crudest) | data-level pre-gen mix **OR** collator token-mask | 1× (data-level) / ~1× (collator) | **LOWEST (data-level)** | **LOW** — random drop hits examples where text is the *only* signal → can just hurt; no targeting |
| **(b) Data-Remixing schedule** *(RECOMMENDED)* | mine unimodal separability over cached feats ($0 CPU); on the subset where **text** is separable/dominant, add **text-masked (image-only)** SFT records; assemble a size-capped schedule | **new `build_*_data.py`** (cand-2 pattern) + config clone + sbatch case; **NO loss/collator/Trainer surgery** | 1× | **LOW–MEDIUM** | **MEDIUM–HIGH** — faithfully forces weak-modality sufficiency on the decoupled subset; masked shape **matches `img_feats` extraction** |
| **(a) MCR per-modality loss** | override `Trainer.compute_loss`; per step run joint + image-only + text-only forwards; add perturbed-diff/JS + sup-con + CEB terms | **`compute_loss` override + custom collator + Qwen2.5-VL vision-token masking under mRoPE** | **~3×** | **HIGHEST** | HIGH (most faithful) **but** native form (head-level) is F55-dead; SFT form is expensive + bug-prone |

**Risk order (lowest→highest): (c) < (b) < (a).** **Recommended registered arm = (b).** Rationale in one line:
(b) is the cheapest arm that still faithfully implements the MCR/Remixing mechanism (weak-modality sufficiency),
reuses the *proven* cand-2 build→SFT→extract→head pipeline with **no model-internal surgery**, and its
text-masked example shape is **extraction-aligned** (unlike cand-2's paired/context prompts). (a) is rejected
for a first draw — its 3×-forward Trainer surgery on Qwen2.5-VL vision-token masking is exactly the
model-internals class that mandates a Codex review and carries real correctness risk, for a cell whose honest
prior is single-digit; and its *natural* head-level realization is already dead (F55). (c) is retained only as
a cheaper-but-cruder fallback if (b)'s separability mining proves noisy on EN.

---

## 2. BAN CHECK — precise, per the team-lead checklist

**vs F65 (vision-LoRA) — NON-ISOMORPHIC (different manipulated variable).** F65's ban scope is **capacity/reach**:
`VISION_UNFREEZE_FORENSIC_RECON.md` §6 rules it *"encoder-class, **SAME D7 collision as generic LoRA** … LoRA-on-ViT
is a 2024-25-standard technique; a pass is a **performance/ablation row**"* — it changed **which linear modules
LoRA reaches** (ViT blocks) with the **joint word-label objective UNCHANGED**. Its `dead[]` entry (#39) records
"image moved, zero head conversion (K-V2 TIE everywhere)." MCR/Remixing change the **objective/schedule**
(modality-balance loss / masked-data schedule), **not** which modules are reachable. F65's ban explicitly does
**NOT price** an objective/schedule change (it prices *capacity*, calling ViT-LoRA a standard technique). **Non-isomorphic
on the manipulated variable.** (The *damper* F65 provides is empirical, not a ban — §6.)

**vs P9b (retrieval-loss-coupled LoRA) — NON-ISOMORPHIC (different loss family).** P9b (`dead[]` #6; F51) couples
the **RGCL retrieval loss** into the LoRA. MCR couples a **modality-balance MI loss**; Remixing couples **no loss
at all** (data schedule). No retrieval-head coupling. Different loss family. Non-isomorphic (litsurvey §C1c concurs).

**vs F51's two-object closure — REFUTED AT THE OBJECTIVE LEVEL (as F65 refuted it at the capacity level).** F51
enumerated exactly two *adapted objects* — encoder (generic word-label LoRA) and joint encoder+decision
(retrieval-loss LoRA = P9b) — under a **fixed objective family**. MCR/Remixing introduce a **third structural
object: the training objective/schedule itself** (modality-balance loss / masked-data schedule), which is neither
generic word-label nor retrieval-loss. F65 already showed F51's "two-object closure" was factually incomplete
(vision *reach* was an un-enumerated cell); this is the **objective-level** analogue. Litsurvey §C1d states this
verbatim: *"neither MCR-loss nor Remixing-schedule was measured."* Non-isomorphic.

**vs curriculum ban (cand-2) — NON-COLLISION (different manipulated variable AND different mining signal).**
This is the sharp ruling the team lead asked for. Cand-2 (`CAND2_CURRICULUM_RECON.md` §2.1, design (i); banked
F56/F59, `positives_bank[]`) manipulates **EXAMPLE FREQUENCY**: it duplicates/subsamples the **identical,
full-modality** single-video records by their **cross-label confusability** score (`dupᵢ = 1 + round(λ·cᵢ)`,
size-capped) — "which / how-often the identical records appear." Design (b) here manipulates **MODALITY CONTENT
WITHIN examples**: it **removes the transcript** (masks the stronger modality) on the decoupled subset, mined by
**unimodal separability**. Two independent axes of difference:
- **Manipulated variable:** example *multiset/frequency* (cand-2) vs *modality composition inside each example* (b).
- **Mining signal:** *cross-label confusability / boundary hardness* (cand-2) vs *unimodal separability / which
  modality is individually sufficient* (Data-Remixing).
Neither is the other's lever. **Non-collision.** **Honest flag (the one way it *could* graze):** if arm (b) were
additionally to **upweight example frequency** by separability (duplicate the separable-modality examples), that
frequency lever would collide with cand-2. The registered arm must therefore manipulate **modality masking ONLY**
(content, not frequency); keep the masked schedule at a fixed per-example rate / size-cap, not a
separability-weighted duplication. Under that constraint the cell is clean of the curriculum ban.

**Standing bans — all CLEAR:** own-train-split only (no cross-dataset mixing, `banned_constraints[8]`); no gold
in inference (masking uses gold labels only to *select* the training schedule, deployed path stays label-free —
same clearance as cand-2 §2.2); no OCR; local Qwen2.5-VL-7B single model; no external API; no MLLM-scores-as-training-signal
(the masking signal is *label separability*, not an MLLM score). **In-box on training grounds; D7 = adjacent/sub-ruling.**

---

## 3. $0 / CHEAP PRE-GATE — honest: NO genuine $0 pre-GPU gate; a cheap-after-extract diagnostic exists

**There is no $0-pre-GPU kill that prices the encoder cell** — identical honesty to F65 (`VISION_UNFREEZE_FORENSIC_RECON.md`
§5). The rebalanced-adapted features **do not exist until the modality-rebalanced SFT runs**; nothing cached can
screen them. **This cell is irreducibly a training experiment.**

**The head-level $0 sanity the team lead flagged does NOT price the cell — and is likely already dead.** Training
the align-fusion **head** on modality-dropout-augmented **cached** features tests **head-level** rebalancing
(reweighting frozen `{img,text}` streams), not **encoder-level** adaptation. Worse, that head-level rebalance is
**bounded by F55's oracle ceiling** (frozen-stream fusion on EN ≤ **+0.0250 < +0.030**) and grazes **F50/F49** —
i.e. it is *already answered dead*. Do **not** dress it up as a gate; at best it is a null confirmation of F55,
not a screen for the encoder cell.

**What DOES exist — a cheap-after-extract kill (spend the ~0.5 GPU-h extract, gate before the head budget), using
the F58 image-MOVED / stream-decomposition operator (`scripts/analysis/encoder_swap_geometry.py`):**
- **Rebalance-fired check:** on the rebalanced-adapted EN cache, the **img↔text train-LOO AUC gap must CLOSE**
  (image climbs toward text) by a pre-declared margin **beyond F65's generic vision-LoRA** (F65 moved EN image
  +0.032 trLOO; the rebalancing must move it *further and toward closing the gap*, not merely replicate F65). If
  the gap does not close beyond F65 → the balance objective did nothing capacity did not → **cheap kill, no head
  spend** (rearranging a dead cell, F65's §4 logic).
- Only if the gap closes do you spend the (trivial) 3-seed head budget.

---

## 4. DESIGN — EN primary and ONLY; HateMM & ZH not run

| dataset | run? | why |
|---|---|---|
| **MHC-EN** (549 train) | **YES — primary & only** | The sole target: image collapses upstream (frozen-Qwen img trLOO 0.599 vs CLIP 0.734; F44); the *whole* imbalance-vs-label-limit question lives here. |
| **HateMM** (743) | **NO** | **Text-carried & frozen-swap-sufficient** (F58); image is already the *strong* modality (img trLOO 0.826, highest of three). There is no weak modality to rebalance *toward* — no mechanism fit. Running it wastes ~4–6 GPU-h to (at best) sharpen an already-passing leg. |
| **MHC-ZH** (579) | **NO** | **Text-borne** (F45; gain lives entirely in the text stream, image flat 0.718→0.714). Forcing image load is *off-mechanism*; ZH's marginal pass is a text-adaptation story. |

**Arms (EN):** (1) **rebalanced-LoRA** (new, design (b) — tag e.g. `Qwen2.5-VL-7B-Instruct-LoRA-mcr_HF`); comparators
all **banked, NO re-run**: (2) **frozen-CLIP floor** (job 12850) — val-sel **0.7619/0.6715**, final **0.7785/0.7202**
(`LORA_HATEMM_PREREG.md` §2.2); (3) **frozen-Qwen floor** (12850) — val-sel **0.7805/0.7219**, final **0.7847/0.7425**
(the **EN honesty bar**); (4) **generic-LoRA EN** (B4-EN, `dead[]` #18 — banked *below both frozen floors*, the
"generic adaptation fails EN" anchor); (5) **F65 vision-LoRA EN** (`dead[]` #39, VISION_UNFREEZE — the image-MOVED-but-TIE
context arm, the sharpest "capacity moved the image, converted nothing" comparator).

**Success bars (house style, 3 head-seeds, dual protocol judged independently, pre-declared):**
- **K-MCR-1 (house conjunct, vs frozen-CLIP):** mean Δacc ≥ **+0.030** AND ΔmF1 ≥ **+0.030**, sign **3/3**, per protocol.
- **K-MCR-2 (EN honesty flag, MANDATORY — vs frozen-Qwen):** must beat the frozen-Qwen floor (val-sel 0.7805/0.7219,
  final 0.7847/0.7425). Generic EN LoRA (B4) could **not** clear this; if the rebalanced arm cannot either, it did
  **not** repair the collapse decision-relevantly ⇒ rearranging a dead cell.
- **K-MCR-3 (add-over-capacity — the decisive novelty-relevant stat):** paired vs the **banked F65 vision-LoRA** arm.
  If the rebalanced objective merely ties F65's capacity arm (which already moved the image and converted nothing),
  the *objective* added nothing over *capacity* → bank the null. This is the bar that makes the cell non-redundant
  with F65.
- **Mechanism check (F58 operator, pre-head):** the §3 image-rebalance / img↔text AUC-gap-closure diagnostic must fire.
- **KS-regression / KS-below-floor:** below B4/generic − 0.014, or below CLIP floor on EN → strong negative banked.

**Honesty clauses (carry verbatim, mirror F65/cand-2):** novelty = **PENDING USER D7 SUB-RULING**, not decided by
this experiment; **single-encoder-draw** limitation (3 head-seeds read ONE rebalanced-SFT draw ⇒ head-seed variance,
not SFT-draw variance); the K-MCR-3 (add-over-capacity) result, not the vs-CLIP result, is what any novelty claim
rests on; EN test is **not virgin** (prior enc3s arms read it; this is a re-measure under the same protocol).

---

## 5. COST

**EN only, one arm, design (b):**

| stage | GPU-h | note |
|---|---|---|
| separability mining | ~0 (CPU) | $0 over banked frozen-Qwen EN train cache (cand-2 mining machinery, `cross_channel_router_gate.py:73–131`) |
| rebalanced LoRA-SFT | ~3–4 | ≈ generic EN SFT footprint (549 videos); +~50% if a two-stage remix (complete → masked) is used |
| extraction | ~0.5 | adapter-generic runner, unchanged |
| **cheap-after-extract diagnostic** | ~0 (CPU) | F58 image-MOVED / AUC-gap gate **before** head spend |
| 3-seed head | ~minutes | cached feats, fresh RGCL head |
| **TOTAL NEW GPU** | **~4–6 GPU-h** | local SLURM only (SFT is a training run; Modal is features-only) |

Design (a) MCR-loss would be **~8–12 GPU-h** (3× forwards) — a further reason it is not the first draw. HateMM/ZH
**not run** saves ~8–12 GPU-h vs a naive three-dataset sweep.

---

## 6. PRIORS — honest, F65-damped

**Performance prior on EN clearing K-MCR-1 (≥+0.03 conjunct): ~5–8%** — *below* the litsurvey's stated 10–20%,
and I will say plainly why: **F65 is a same-axis null.** F65 *moved* the collapsed EN image stream (+0.032 trLOO,
first lever ever to) and converted **zero** head accuracy (K-V2 TIE both protocols; cleared frozen-Qwen val-sel
only, not final). MCR/Remixing's *entire* mechanism is "make the image stream carry load" — but F65 already gave
the image more capacity, moved it, and the head converted nothing, while **F55's oracle** shows even an oracle
per-item fusion of a *healthy* image + best text on EN is +0.025 < +0.03. The **only** residual over F65 is that
MCR forces the image to be **conditionally** predictive (`I(img;Y|text)`), a subtly different target than F65's
"generally better image rep" — and that a *non-frozen* rebalanced image was not the object F55's oracle measured.
That residual is real but thin; hence ~5–8%.

**Publishable-either-way value: LOW–MODERATE (not high).** A clean null *does* close "the last mechanistic escape
hatch on EN" — "we rebalanced the **objective**, not merely the capacity, and EN still did not convert" is the
strongest form of the F44/F55 label-limited claim. **But** it is **largely redundant with F65**, which already
banked "image moved, zero conversion." The MCR-null adds "…and even a modality-balance *objective* converts
nothing" — a *marginal* hardening of an already-owned chain (F44→F55→F65), not a new pillar. So the incremental
paper value does not, by itself, justify autonomous spend; it justifies spend **iff** the user judges the
escape-hatch worth ~4–6 GPU-h AND opens the D7 sub-ruling.

---

## 7. VERDICT & STAGE PLAN

**GO-IF (conditional, user-gated).** **NO-GO as an autonomous performance bet** (~5–8% prior, F65-subsumed at the
mechanism level, F55-capped). **GO-IF** the user (a) opens a **D7 sub-ruling** (is a modality-balance *objective*
a contribution distinct from generic encoder LoRA, or a textbook trick? — same class of ruling as cand-2), and
(b) wants the EN escape-hatch **null** on record for the paper.

**Recommended arm:** **(b) Data-Remixing-style modality-masked SFT schedule** — mine unimodal separability over
the banked frozen-Qwen EN train cache ($0 CPU), add text-masked (image-only) SFT records on the text-separable
subset, size-capped, **modality-content masking only (no example-frequency reweighting** — keeps it clean of the
cand-2 curriculum ban). Reuses the proven cand-2 build→SFT→extract→head pipeline; masked shape is
extraction-aligned with `img_feats`; **no model-internal surgery.**

**Stage plan (only on GO-IF):**
1. **$0 CPU** — author `src/utils/build_remix_sft_data.py` (new file, cand-2 pattern); mine EN separability;
   build the size-capped masked schedule; author config clone + sbatch case; diff-verify.
2. **Prereg** K-MCR-{1,2,3} + F58 mechanism gate + honesty clauses (single submit; independent review; freeze hash) —
   only after the D7 sub-ruling opens.
3. **~3–4 GPU-h** — rebalanced EN LoRA-SFT (local SLURM; JobHeldUser auto-release, never force).
4. **~0.5 GPU-h** — extraction → **cheap-after-extract image-rebalance diagnostic (F58 operator); KILL before head
   if the img↔text AUC gap does not close beyond F65.**
5. **~minutes** — 3-seed head; adjudicate K-MCR-1/2/3 vs banked floors; bank verdict (pass = D7-sub-ruling-gated
   novelty-relevant; null = label-limit hardened, escape-hatch closed).

**Total on GO-IF: ~4–6 GPU-h, EN only. Ranking by implementation risk: (c) < (b) < (a); recommended = (b).
Curriculum-ban (cand-2) collision: NON-COLLISION (different manipulated variable — modality content vs example
frequency — and different mining signal — separability vs confusability), clean provided the arm masks modality
content and does NOT reweight example frequency.**

---

## PROVENANCE
- Source cell: `refine-logs/LITSURVEY_NOVEL_MECHANISMS.md` §C1 (MCR arXiv 2411.07335 `github.com/kkontras/MCR`
  NeurIPS 2025; Data Remixing arXiv 2506.11550), §4 top-5 #1, §C5 modality-dropout.
- Dampers: F65 (`state/findings.jsonl`; `VISION_UNFREEZE_FORENSIC_RECON.md`; `dead[]` #39) — image MOVED +0.032
  trLOO, K-V2 TIE, zero head conversion; F55 (`dead[]` #34; `PREMISE_D_GATE_RECORD.md`) — EN oracle ceiling +0.0250;
  F44 (encoder-swap diagnosis) — EN image collapse 0.734→0.599, label-limited; F58 (`HATEMM_LORA_STREAM_DECOMP.md`)
  — HateMM text-carried; F45 (`B3_ZH_LORA_DECOMPOSITION.md`) — ZH text-borne.
- Ban rulings: F65 §6 (capacity ≠ objective); P9b `dead[]` #6 / F51; F51 two-object closure (refuted at objective
  level); cand-2 curriculum (`CAND2_CURRICULUM_RECON.md` §2.1, `positives_bank[]` F56/F59) — example frequency vs
  modality content, confusability vs separability; `banned_constraints[]` 0–8.
- Machinery: `src/utils/build_lora_sft_data.py` (fused single-prompt SFT schema); `src/utils/generate_VideoMLLM_embedding_lora_HF.py:29–64,264–409`
  (`img_feats` = frames+neutral-instr no-transcript; `text_feats` = frames+transcript, text-pooled — masking is
  extraction-aligned); `scripts/analysis/cross_channel_router_gate.py:73–131` (mining, $0 CPU);
  `scripts/analysis/encoder_swap_geometry.py` (F58 image-MOVED operator); EN floors `LORA_HATEMM_PREREG.md` §2.2.
- **Required statements:** ZERO GPU / SLURM / Modal / download / test-touch spent by this recon; no held-out metric
  produced; no `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated; NO job submitted. Committed
  on `main`, not pushed.
