# LIT SURVEY — Novel multimodal-classification mechanisms (WIDE-NET sweep, 2025–2026)

**Author:** lit-novel-mechanisms (ZERO GPU / CPU-only; WebSearch/WebFetch + kill-ledger cross-check).
**Date:** 2026-07-24 NZST.
**Mission (user-ordered):** the WIDE-NET sweep — novel mechanisms in **multimodal classification at large** (NOT restricted to video/harmful-content) that could inspire our stack, including ideas that strengthen the **novelty/paper** story even if performance-neutral.
**Scope discipline:** recon only. No prereg, no promotion, no GPU authorization. Every surviving candidate carries an explicit isomorphism check against `state/directions_tried.json` (dead[]/banned_constraints[]), `state/findings.jsonl` (F1–F67), and `refine-logs/REDTEAM_EXTERNAL_FAMILIES.md` (Families A–E + rejected table). Priors are on **P(≥+1pt on ≥1 dataset)** given the regime (test n≈215/161/149, seed noise ≈1.4pt). This sweep deliberately looks *outside* video: memes, sarcasm, implicit hate, misinformation, sentiment, medical/e-commerce multimodal, multimodal ICL, general representation-learning + information theory.

**One-line stack recap:** LoRA-adapted Qwen2.5-VL-7B dual-stream encoder (8 frames + ASR text) → Hadamard-fusion contrastive head → top-20 kNN vote over an editable own-train memory; ~600 train/dataset; 67-entry kill ledger.

---

## 0. Verdict up front

The performance box is genuinely well-exhausted: the strongest *new* operator classes I found either (i) rebalance a collapsed modality during **encoder adaptation** — a real, un-tried object that F65 (vision-LoRA = added *capacity*, no loss change) does **not** cover, but whose payoff is capped by the F44/F55 finding that MHC-EN is **label-limited, not representation-limited**; or (ii) are cheap training-dynamics operators (validation-free selection, SAM) aimed at the one *measured* failure mode — the F45 78-dev selection tax that costs ZH its val-selected pass.

**The higher-value payload of this sweep is on the paper/novelty side, not the leaderboard.** Two finds materially upgrade the write-up at zero experimental risk:

1. **V-usable information (Xu et al. ICLR 2020; Ethayarajh et al. ICML 2022, outstanding-paper, code released).** This is the *exact* formalization of our Law-I ("oracle headroom exists but no tried operator converts it," 8 instances). It distinguishes **information present** (Shannon) from **information usable by a model family** (V-information). Our "oracle-survives-but-unconvertible" wall is a V-usable-information gap. This elevates Law-I from a campaign observation to a **literature-connected, named phenomenon** — and it is *also* a $0 CPU measurement (pointwise-V-information / PVI on our cached features quantifies the gap per dataset).
2. **The learning-with-disagreement literature crystallized in 2025 (LeWiDi-2025, third edition).** Soft-label/distributional training that models annotator disagreement beats majority-vote gold on subjective tasks *including toxicity/hate*. This is the field-level anchor our **ZH consensus-denoising pillar** currently lacks a citation for.

Everything genuinely novel-and-in-box is below; the isomorphic-dead lookalikes are flagged so the team does not re-derive them.

---

## 1. SURVIVING CANDIDATES (in-box, non-isomorphic)

### C1 — Modality-competition / collapsed-modality rebalancing *during encoder adaptation* (MCR, Data Remixing) — *targets the F44 EN-collapse mechanism with a NEW object*

**(a) Mechanism.** Both are **training-time** rebalancers that force the weaker modality to carry predictive load, but via the *loss / data schedule* rather than added capacity.
- **MCR (Multimodal Competition Regularizer):** decomposes joint MI `I(X₁;X₂;Y)=I(X₁;Y|X₂)+I(X₂;Y|X₁)+I(X₁;X₂)−I(X₁;X₂|Y)` and adds three synchronized losses — a perturbed-difference term (latent permutations + JS divergence) that measures each modality's marginal effect on the output, a supervised-contrastive alignment term, and a conditional-entropy-bottleneck term that suppresses task-irrelevant shared info. Frames the two encoders as a game (collaborate / independent / compete) and drives each to maximize its *conditional* predictive role.
- **Data Remixing:** two-stage — first train on complete data; then decouple samples by unimodal separability (KL-divergence), mask the stronger modality, and re-assemble batches so per-modality gradients align and cross-modal interference is removed. Forces the weak modality to be *sufficient* on its own subset.

**(b) Citation + code.** MCR — Kontras et al., "Balancing Multimodal Training Through Game-Theoretic Regularization," **NeurIPS 2025 poster** (arXiv 2411.07335v3), code `github.com/kkontras/MCR`; reports e.g. CREMA-D 76.1 vs 71.9 (MLB) across 6 AV/text datasets. Data Remixing — Ma, Chen & Deng, "Improving Multimodal Learning Balance and Sufficiency through Data Remixing," **2025** (arXiv 2506.11550). Broader landscape: BalanceBenchmark survey (2502.10816); QQR prototype rebalancing (2508.11159); adaptive-classifier-assignment (2502.20120).

**(c) Isomorphism check.**
- **vs F65 vision-LoRA (image MOVED, K-V2 TIE everywhere):** F65 added *capacity* to the vision path with the standard word-label SFT and the fusion/head loss unchanged. MCR/Remixing keep the encoder objective and instead **change the training signal** (MI-decomposition loss / modality-masked data schedule) to force the collapsed EN image stream to contribute. **Different adapted object** (loss/data-schedule, not adapter weights). Non-isomorphic.
- **vs F50/FA (fusion levers over frozen features) and F49 (modality-reliability router):** those are *inference-time* re-weightings over **frozen** features (killed as rotation / alignment-ceiling). MCR/Remixing operate **during training** and shape the encoder, not a fixed post-hoc combiner. Non-isomorphic.
- **vs P9b (retrieval-loss-coupled LoRA):** P9b couples the *RGCL retrieval loss* into the LoRA; MCR couples a *modality-balance* MI loss. Different loss family. Non-isomorphic (though it lives in the same "adapt the encoder" neighborhood — see legality).

**(d) In-box legality.** **IN-BOX** on training grounds (own train split, no gold-in-inference, no OCR, local Qwen-7B, single model). **D7-novelty: adjacent** — it is a *new adaptation object* (F51's two-object closure is again factually incomplete: neither MCR-loss nor Remixing-schedule was measured), which helps it clear the F24 encoder-*identity* veto, but "add a known modality-balance regularizer" needs a user D7 sub-ruling on whether that's a contribution or a trick.

**(e) Cost.** **MEDIUM–HIGH** — new loss/data wiring into the LoRA trainer + GPU training (~existing LoRA budget/dataset/seed). No cache shortcut. A **$0 pre-check exists**: does MCR/Remixing move EN *image-only* train-LOO AUC off the F65/F44 0.599 collapse floor? If not, do not spend GPU.

**(f) Prior + paper value.** **Perf prior LOW (~10–20%)**, hard-capped on EN by F44/F55 (EN errors are label-limited; F65 already showed *moving* the image stream converts nothing at the head) — so best case is a Pareto conversion on a *representation-limited* dataset, i.e. hardening HateMM or the marginal ZH, not opening EN. **Novelty/paper value MODERATE–HIGH:** it is the cleanest empirical test of "is EN collapse curable by *rebalancing* rather than *capacity*?", and a null further hardens the label-limited claim (a *publishable* negative that closes the last mechanistic escape hatch on EN).

**(g) Rank: #1 among perf-oriented candidates** (highest structural interest; sharpest attack on the one *named* mechanism), but honest perf prior is low and EN-capped.

---

### C2 — Validation-free checkpoint/epoch selection via head-gradient norm — *directly attacks the F45 selection tax; escapes the SWA kill*

**(a) Mechanism.** Select the epoch/checkpoint by a **single forward-backward pass**: detach features Z, run cross-entropy through the linear head only, backprop through head weights W, record `‖∇_W ℒ‖_F`. Lower gradient norm ⇒ flatter minimum ⇒ better generalization; pick the checkpoint minimizing the (scale-normalized) norm. Replaces the noisy 78-item dev argmax with a **data-free geometric criterion**.

**(b) Citation + code.** "No Validation, No Problem: Predicting Model Performance from a Single Gradient," **arXiv 2601.16874 (Jan 2026)**; validated on ImageNet (Pearson r≈−0.85 vs Top-1), COCO det/seg (ρ up to −0.98), near-oracle diffusion checkpoint selection; open-source with auto-tuning. Lineage: flat-minima↔generalization (SAM, Foret et al. ICLR 2021; "worst-case m-sharpness" selection line 2508.00522/2511.03548).

**(c) Isomorphism check.**
- **vs F62/F62b SWA (Family C, KILLED):** SWA **averages** per-epoch head weights and *lost* dev points on HateMM's mid-peak-dev seeds. This candidate does **not average** — it **selects** one existing checkpoint by a validation-free score. Different object; the SWA failure mode (averaging drags a mid-peak optimum) does not apply. Non-isomorphic.
- **vs F45 itself:** F45 *names* the pathology (dev saturates ~ep19 while test climbs to ep29 → argmax undershoots; 78-dev selection noise). This is a **direct** selection-free remedy for exactly that, not a new signal. In-scope, non-isomorphic.

**(d) In-box legality.** **IN-BOX, $0 CPU** — runs on banked per-epoch head checkpoints (`src/run_rac.py:764` saves every epoch) + cached features; no gold, no ensemble, no test-touch (it selects using train-side geometry, then the *selected* model is scored once). D7: performance-only *and* a **protocol contribution** (selection-free evaluation is itself a paper asset — see §3).

**(e) Cost.** **VERY LOW** — one backward pass per banked checkpoint, minutes on CPU/1-GPU-inference.

**(f) Prior + paper value.** **Perf prior LOW–MODEST (~15%)**, *targeted*: most likely payoff is recovering the **ZH val-selected leg** that F45/F53 lose to the selection tax, hardening the 2-dataset story under *both* protocols. Caveat: the authors flag small-head instability (temperature/scale-norm mitigations offered) — our head is tiny, so a $0 sanity check (does the score rank our banked epochs sensibly on the *train* LOO?) gates it. **Paper value HIGH:** "selection-free protocol" is precisely the F45 pain point and a clean methodological contribution independent of whether it moves a number.

**(g) Rank: #2** (best cost×paper-value; $0; attacks a documented failure SWA could not fix).

---

### C3 — Sharpness-Aware Minimization (SAM) on the head — *flat-minima optimizer, distinct from the averaged SWA that died*

**(a) Mechanism.** Replace/augment the head optimizer with SAM: minimize loss *and* worst-case sharpness in an ε-ball, steering the tiny head to a wider optimum → better generalization on n≈150 test with a 78-dev-noise selection regime.

**(b) Citation + code.** Foret et al., SAM, **ICLR 2021** (widely implemented). 2025 refinements: GLAD for VLMs (2507.13089), Focal-SAM long-tailed (2505.01660), Bi-LoRA efficient SAM-for-fine-tuning (OpenReview 2025), Sparse-Layer SAM (2602.09395). Few-shot evidence: "Flatness Improves Backbone Generalisation in Few-shot Classification" (2404.07696).

**(c) Isomorphism check.** **Non-isomorphic** to F62 SWA: SAM is an **optimizer** that finds a flat solution *during* training (one model, one trajectory); SWA *averages weights post-hoc*. It is also not manifold-mixup (Family D) nor a decision-side operator. It varies the **head-training optimizer**, an object never touched (head was fixed Adam-ish + triplet+BCE throughout).

**(d) In-box legality.** **IN-BOX** (head training over cached embeddings, no gold/ensemble/encoder-touch). D7: generic optimizer → **D7-dead / performance-only**.

**(e) Cost.** **LOW** — swap optimizer, re-run cheap head training. Minutes–hour.

**(f) Prior + paper value.** **Perf prior LOW–MODEST (~10–15%)** — SAM's gains on tiny already-margin-regularized heads at n≈150 sit near seed noise; best as a **margin-hardener** for the marginal ZH pass, companion to C2. Paper value LOW (generic).

**(g) Rank: #3.**

---

### C4 — Learning-from-disagreement / soft-label training for the ZH consensus pillar

**(a) Mechanism.** Instead of training on collapsed majority-vote gold, train on the **annotator label distribution** (soft labels) or model per-annotator perspectives, then evaluate with soft/distributional metrics. On subjective tasks (hate/toxicity), disagreement-aware training measurably beats majority-vote gold. Our ZH consensus-denoising pillar is a *special case* of this family (it denoises toward a consensus); the 2025 literature offers principled soft-label objectives and evaluation.

**(b) Citation + code.** **LeWiDi-2025**, third Learning-With-Disagreements shared task (arXiv 2510.08460) — soft-label/distributional eval, includes toxicity/hate datasets, finding: disagreement-as-signal beats collapsed gold on subjective phenomena. Supporting: MO-WEL weighted-ensemble over label projections (Springer *Datenbank-Spektrum* 2026); agreement-based clustering of annotators (2605.09955); demographic-aware experts + synthetic perspectives (2508.02853); SeedBERT recovering rating distributions (2211.13196).

**(c) Isomorphism check.**
- **vs archive-auto-repair (DEAD, AND-rule C−A=0):** that DELETED noisy memory *entries* (a data-cleaning op with a two-vote AND rule). Soft-label training changes the **head loss** to consume label *distributions* — a different object entirely. Non-isomorphic.
- **vs MLLM-scores-as-training-signal (banned):** soft labels come from **human annotators**, not an MLLM. Not covered by that ban. Non-isomorphic.
- **vs cross-seed ensembles (banned):** MO-WEL is an *ensemble* over label projections → that specific realization **grazes the ban** (flag). The *single-model soft-label-loss* realization does not.

**(d) In-box legality.** **CONDITIONAL — needs per-annotator labels.** MHC/HateMM ship majority-vote gold; if annotator-level annotations are unavailable this is **paper-value-only** (frame the ZH pillar within the LeWiDi paradigm). If per-annotator votes exist for ZH (plausible for a consensus-labeled set — needs a $0 data check), a single-model soft-label-loss run is in-box (own train, no gold-in-inference beyond the training labels, no ensemble). Flag: **user/data-availability check required**.

**(e) Cost.** **LOW–MEDIUM** (head/encoder retrain on soft targets) *if* labels exist; **$0** for the paper-framing use.

**(f) Prior + paper value.** **Perf prior UNCERTAIN** (gated on label availability; genuine but not large on n≈150). **Paper value HIGH:** it gives our consensus-denoising pillar its missing field-level anchor and reframes it from an ad-hoc trick to a member of a named, growing research program — a novelty-story upgrade even at +0.

**(g) Rank: #4** (paper-value-forward; perf gated on data).

---

### C5 — Modality dropout as a training regularizer against text-stream dominance

**(a) Mechanism.** Randomly mask a whole modality branch per-sample/mini-batch during head/encoder training — macro-structure dropout that penalizes over-reliance on the dominant (text) stream, the documented root of modality imbalance. Adaptive/entropy-gated variants drive the mask from training-time entropy.

**(b) Citation + code.** Survey of modality dropout (EmergentMind topic; "Multimodal Learning Under Imperfect Data" survey 2025); MMP Masked-Modality-Projection (2410.03010); game-theoretic MCR (2411.07335, shared with C1). Root-cause: text-dominance in MLLMs (2508.10552).

**(c) Isomorphism check.** **Non-isomorphic** to any kill — it is a **training regularizer** on the fusion/head, an object the campaign never varied (fusion was fixed Hadamard). Not F50 (inference-time frozen reweight), not F65 (adapter capacity), not manifold-mixup. Related to C1 Data-Remixing (masking) but simpler / cheaper.

**(d) In-box legality.** **IN-BOX** (own train, no gold/ensemble). D7: generic → **D7-dead / perf-only**.

**(e) Cost.** **LOW** — a few lines in the trainer over cached streams (or cheap re-encode for encoder-side).

**(f) Prior + paper value.** **Perf prior LOW (~10–15%)** — could rebalance the EN text-dominant fusion, but F44/F55 EN label-limit caps it; best framed as a companion to C1. Paper value LOW.

**(g) Rank: #5.**

---

## 2. ISOMORPHIC-DEAD lookalikes (do NOT re-derive) — one line each

| Candidate (2025–2026) | Anchor | Why dead / isomorphic |
|---|---|---|
| **GatedCLIP** dynamic gated fusion for hateful memes | arXiv 2602.20818 | Per-sample learned gate = per-modality reweight over (near-)frozen features → **F50** ("do not re-propose per-modality temperatures/reweights over banked frozen features"). Our head already learns a Hadamard-align fusion. |
| **Adaptive Evidential (Dempster–Shafer) fusion** | ETASR 2025 (15931) | Per-modality uncertainty/reliability weighting = modality-reliability combiner → **F49** (alignment-ceiling) + **F50** (fusion door closed). *Calibrated-uncertainty output has minor guard-rail/audit paper-value only.* |
| **ExPO-HM explain-then-detect** for hateful memes | arXiv 2510.08630 | MLLM reasoning → decision = **P4/P5** territory (schema-distill / counterfactual, both dead) and MLLM-decision roles measured dead at 7B–32B. *Framing value only.* |
| **Multimodal ICL with retrieved demonstrations** | 2503.04839 / 2510.04560 / 2410.20482 | Memory-as-ICL-demos feeding an MLLM to produce a **decision** = **P1/P2/P5** dead. The "retrieval-fed ICL producing a FEATURE not a decision" escape is real *in principle* (representation-engineering / attention-editing variants) but untried, speculative, and D7-encoder-adjacent — **not** promoted; recorded as a distant option. |
| **MM-HSD cross-modal-attention fusion** | ACM MM 2025 | Its only novel operator is OCR-queried CMA → **OCR vetoed**; non-OCR CMA ≈ trained fusion over frozen {text,audio(banked-zero F41/F64),video} ≈ F50/P9b. (Confirmed in REDTEAM rejected table.) |
| **Balanced-softmax / LDAM / logit-adjust** class-balance | Menon ICLR'21 / LDAM | ~40% positive rate is not severe imbalance; generic training trick (D7-dead); honorable-mention-only per REDTEAM. |
| **wav2vec2 / AST / BEATs learned audio** | MM-HSD / DeToxy | **F64** killed Whisper-encoder audio at $0 (zero conditional info all 3 datasets); general-audio encoders are download-gated relaxation with a now-lowered prior. |
| **VideoMAE / self-sup continued pretraining on own train** | NeurIPS'22 | Encoder-class → **D7-dead (F24)**; and 549–743 videos is ~5× below its data-efficient floor. |
| **Manifold/embedding mixup (REDTEAM Family D)** | ICML'19 / S2M2 | *Not killed — unmeasured*; but D7-dead / perf-only and near-seed-noise at n≈150. Left as a cheap companion, not a lead. |
| **Multi-view test-time frame augmentation (REDTEAM Family E)** | 2406.08593 | *Not killed — unmeasured*; per-item legal but variance-reduction on n≈150 buys fractions of a point. Left as-is. |

---

## 3. PAPER-VALUE-ONLY finds (citations/framing that strengthen the story with NO new experiment required)

These are the highest-leverage results of the sweep. None needs a GPU; the first is *also* a $0 CPU measurement.

1. **V-usable information / predictive V-information — the formalization of Law-I.**
   Xu et al., "A Theory of Usable Information Under Computational Constraints," **ICLR 2020**; Ethayarajh, Choi & Swayamdipta, "Understanding Dataset Difficulty with V-Usable Information," **ICML 2022 (outstanding paper)**, code `github.com/kawine/dataset_difficulty`; pointwise-V-information (PVI) for per-instance difficulty. **Why it matters:** our "oracle headroom exists but no tried operator converts it" (8 instances: P3, S2S, W2-A, GIR, FA, ISR…) is *exactly* a gap between **Shannon information present** and **information usable by our model family** — the definitional content of V-information. Citing this converts Law-I from a campaign idiosyncrasy into a **named, theoretically-grounded phenomenon**, and lets us say the frozen-feature ceiling is a *V-usable-information* ceiling, curable only by expanding the model family (= adaptation, per F45) — which is precisely our empirical finding. **Bonus $0 experiment:** compute PVI/V-information on our cached features per dataset to *quantify* the info-vs-usability gap (turns Law-I from qualitative to measured). Supporting cites: minimal-sufficient-representation-is-not-optimal-downstream (2203.07004); "recovering collapsed info costs downstream accuracy" (2601.11334). **This is the single most valuable find for the novelty story.**

2. **Learning-with-disagreement paradigm (LeWiDi-2025) as the anchor for the consensus-denoising pillar.**
   arXiv 2510.08460 (+ MO-WEL 2026, agreement-clustering 2605.09955). **Why it matters:** our ZH consensus-denoising pillar currently reads as a bespoke trick; framing it as an instance of the *learning-with-disagreements* program (soft-label/distributional supervision beating majority-vote gold on subjective hate/toxicity) gives it a citation lineage and a principled evaluation vocabulary. Upgrades the pillar's framing even with zero new runs.

3. **Editable/auditable-model-component governance for the archive pillar.**
   DMM-Gov auditable closed-loop editing (admission thresholds → progressive rollout → reversible rollback → change-audit certificates; from the 2509.18868 LLM-memory survey); unlearning-based conflict-free model editing (**NAACL 2025**, 2025.naacl-long.325); Editable-XAI co-editable explanations (2602.12569); Sharpness-Aware Machine Unlearning (2506.13715). **Why it matters:** our "auditable/editable archive memory (human-in-the-loop)" pillar can now cite a 2025–2026 governance literature (model editing, memory unlearning, reversible/audited edits) instead of standing alone — turns a design choice into a positioned contribution amid the 2026 HITL-governance wave.

4. **Modality-dominance / imbalance diagnosis literature to license the EN-collapse analysis chapter.**
   Text-dominance in MLLMs (2508.10552); BalanceBenchmark survey (2502.10816); classification-ability-disproportion (2502.20120). **Why it matters:** our §3.9 EN-image-collapse mechanistic story (F44/F58/F65) is currently self-contained; these give it an external, named diagnosis ("modality imbalance / text dominance") and let us say F65's "image moved, converted nothing" is the *expected* signature of a **label-limited** rather than **imbalance-curable** collapse — a sharper, citable distinction.

---

## 4. TOP-5 (perf-oriented, ranked prior × cost; paper value noted separately)

| # | Candidate | In-box? | Cost | Perf prior (≥+1pt/≥1 ds) | Non-isomorphism | Paper value | Cheapest next step |
|---|---|---|---|---|---|---|---|
| **1** | **Modality-competition / collapsed-modality rebalancing during adaptation** (MCR, Data Remixing) | YES-train / **D7-adjacent** | MED–HIGH (GPU) | **~10–20%** (EN-capped by F44/F55) | **HIGH** — new adaptation object (loss/data-schedule); F65 covered only *capacity* | MOD–HIGH (tests if EN collapse is imbalance- vs label-limited; publishable either way) | **$0 pre-check:** does it move EN image-only train-LOO AUC off 0.599? |
| **2** | **Validation-free head-gradient checkpoint selection** (2601.16874) | YES (**$0 CPU**, banked ckpts) | **VERY LOW** | **~15%**, targeted at F45 ZH val-sel tax | HIGH — *selects* not *averages* → escapes F62 SWA kill | **HIGH** — selection-free protocol = the F45 pain point | Score banked per-epoch heads, sanity-check on train LOO, then one selected read |
| **3** | **SAM flat-minima head optimizer** (Foret ICLR'21; GLAD/Bi-LoRA 2025) | YES (perf-only) | **LOW** | **~10–15%** | MED — optimizer object, distinct from SWA averaging | LOW (generic) | Swap head optimizer to SAM, re-train cheap head |
| **4** | **Learning-from-disagreement soft-label training** (LeWiDi-2025) | **CONDITIONAL** (needs per-annotator labels) | LOW–MED / $0 framing | UNCERTAIN (gated on data) | MED — head-loss consumes label *distributions*, not gold | **HIGH** — anchors the consensus pillar | $0 data check: do MHC-ZH/HateMM ship annotator-level votes? |
| **5** | **Modality-dropout regularizer** vs text dominance | YES (perf-only) | **LOW** | **~10–15%** | MED — training regularizer on fusion never varied | LOW | Add per-sample modality mask to head trainer |

**Bottom line.** On performance, the box holds: every new lead is either EN-capped by the label-limited finding (C1/C5) or aimed at the marginal ZH selection tax (C2/C3), with honest priors ≤~20%. The **decisive value of this wide-net sweep is in the paper**: V-usable information formalizes Law-I into a named phenomenon (and a $0 measurement), and the learning-with-disagreement + model-editing-governance literatures give the consensus-denoising and editable-archive pillars the citation lineage they currently lack. Highest-EV concrete move = the **$0 C2 validation-free-selection probe** (cheap, targets a documented failure SWA could not fix, doubles as a protocol contribution), with the **$0 PVI/V-information measurement** as the paper-side companion.

---

## Provenance
- Kill ledger: `autoresearch/goal_mllm_plus3/state/findings.jsonl` (F1–F67), `state/directions_tried.json` (dead[]/banned_constraints[]/positives_bank[]), `refine-logs/REDTEAM_EXTERNAL_FAMILIES.md` (Families A–E + rejected table), `refine-logs/TERMINUS_round3_mllm_plus3.md`.
- Feasibility: `src/run_rac.py:764` (per-epoch head checkpoints saved) → enables $0 C2/C3.
- External anchors (2020–2026, code-released unless noted): MCR NeurIPS 2025 (arXiv 2411.07335, `github.com/kkontras/MCR`); Data Remixing 2506.11550; val-free-selection 2601.16874; SAM Foret ICLR 2021 / GLAD 2507.13089 / Bi-LoRA (OpenReview 2025); LeWiDi-2025 2510.08460 / MO-WEL (Datenbank-Spektrum 2026) / agreement-clustering 2605.09955; V-usable-info Xu ICLR 2020 + Ethayarajh ICML 2022 (`github.com/kawine/dataset_difficulty`); minimal-sufficient-rep 2203.07004; info-theoretic-rep 2601.11334; GatedCLIP 2602.20818; evidential-fusion ETASR 15931; ExPO-HM 2510.08630; MM-HSD ACM MM 2025; text-dominance 2508.10552 / BalanceBenchmark 2502.10816; DMM-Gov (2509.18868) / conflict-free editing NAACL 2025 (2025.naacl-long.325) / Editable-XAI 2602.12569 / SAM-unlearning 2506.13715.
- **Discipline:** CPU-only, zero GPU/SLURM/Modal/downloads; `state/` unmodified. Cloud/external numbers are triage context only, never mixed with local G-repro numbers.
