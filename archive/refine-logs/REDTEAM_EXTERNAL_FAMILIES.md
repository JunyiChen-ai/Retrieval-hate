# RED-TEAM: External operator-family sweep — attacking the "method-space exhausted" claim from outside

**Author:** redteam-external-families (ZERO GPU / CPU-only; web literature + kill-ledger cross-check only).
**Date:** 2026-07-20 NZST.
**Mission:** Sweep external literature (2023–2026) for transferable operator families the project has **not** tried and that are **not isomorphic** to anything in the kill ledger, in order to test — from the outside — the `TERMINUS_round3_mllm_plus3.md` claim that *every injection point inside the frozen constraint box is closed*.
**Scope discipline:** This is recon. Nothing here is a prereg, a promotion, or a GPU authorization. Every candidate carries an explicit isomorphism check against `state/findings.jsonl` (F1–F60) and `state/directions_tried.json`, an in-box legality verdict, and an honest prior. Priors are on **P(≥+1pt on ≥1 dataset)** given the regime (test n = 215/161/149, seed noise ≈ 1.4pt).

---

## 0. Verdict up front

The exhaustion claim is **strong but not airtight**. The campaign closed the injection points it *enumerated* — decision-side auxiliary signals (Axis A), encoder identity/adaptation on the **language** pathway (Axis B / Law IV two-object closure F51), retrieval-object/temporal operators (Axes C/D, Laws II), per-item cross-channel selection (Axis A/Law III F47), fusion/threshold/residual operators (F50/B5/GIR). Three **genuinely un-enumerated** operator classes survive the isomorphism check:

1. **The decision-aggregation *topology* itself** — every decision-side kill injected a *signal* into, or *selected between arms* of, the top-20 one-hop kNN vote. **No finding ever changed the vote from a one-hop read to a multi-hop graph-diffusion / label-propagation read over the same frozen keys.** F46 explicitly precisifies the banked nulls as "LINEAR/logistic over feature space" plus a *specific enumerated list* of killed nonlinear operators (set-matching, pooling, thresholds, residuals). Graph label-propagation is a nonlinear decision operator **not in that list**.
2. **The vision-tower adapted object** — F51's "adaptation family has exactly two adapted objects" closure covers the LLM-backbone LoRA and the joint decision-level fine-tune. F54/F58 record that the **vision tower + multimodal projector were frozen in every run**, and the image stream "stayed flat only because every SFT target is text-decodable." **No experiment ever adapted the vision pathway directly.**
3. **Single-trajectory training-dynamics operators** (weight averaging, embedding mixup, TTA) that target the *named, measured* failure mode F45 identified — the 78-dev validation-selection tax that costs ZH its val-selected pass — rather than injecting a new signal.

None of these is a slam-dunk (all carry guarded priors, and two are D7-novelty-adjacent). But the factual claim "no untried injection point remains inside the box" is **falsified as stated**: the decision-aggregation topology and the vision adapted-object are untried, in-box, and cheap-to-cheap-ish. Refutation ranking and the honest performance priors are in §7.

---

## 1. FAMILY A — Global label propagation / graph diffusion over the kNN memory graph *(lead refutation)*

**(a) Mechanism.** Replace/augment the deployed top-20 rank-weighted signed-cosine kNN vote with a **multi-hop label-propagation** decision operator over a graph built from the *same* frozen fused keys: `Y^(t+1) = α·W̃·Y^t + (1−α)·Y^0`, W̃ a row-normalized kNN affinity, Y^0 the train-label one-hots (query rows zero), iterated ~3 steps, then read the query rows. Labels diffuse through the manifold instead of being read only from immediate neighbours — over-smoothing is controlled by α and hop count.

**(b) Citation + code.** ECALP, *Efficient and Context-Aware Label Propagation*, ICLR 2025 — arXiv 2412.18303, code `github.com/Yushu-Li/ECALP` (training-free, edge-reweighting, both transductive and an inductive graph-expansion variant). Lineage: Iscen et al., *Label Propagation for Deep SSL*, CVPR 2019; "Combining Label Propagation and Simple Models out-performs GNNs" (C&S). Semi-supervised-GNN cousins (GraFN, SIGIR 2022; diffusion-augmented GCN 2503.12563) exist but need a trained GNN (see legality).

**(c) Isomorphism check.** This is the load-bearing candidate, so the check is careful.
- **vs F47 / Law III (per-item cross-channel selection, `30d0ee1`):** F47 kills a router that *selects between the CLIP-arm and the Qwen-arm* using decision-level meta-features. LP does **not** select between channels and uses **no** meta-features — it operates *within one representation's* similarity graph and changes the vote's *topology*, not the choice of channel. **Non-isomorphic.**
- **vs the K9 linear-zero gates (W2-A F42, CTF F39, GIR F43) and F46's precisification:** F46 states the banked nulls are "LINEAR/logistic over feature space" and that the campaign *additionally* killed a **named list** of nonlinear operators — set-matching (S2S F37), pooling (CTF F39), thresholds (B5 F34), residuals (GIR F43). Multi-hop graph diffusion is a nonlinear operator **absent from that list**; it re-aggregates the label field, it does not add an auxiliary feature, so the "conditionally redundant given Z_best" law (D1) does not directly reach it — D1 is about *new signals*, LP injects *no new signal*.
- **vs W2-E / Axis I (memory reorganization, F28):** W2-E reorganizes the *keys* (k-means/prototypes) — a lossy function of the pooled vector. LP keeps keys **fixed** and changes the *propagation dynamics*. Different object. **Non-isomorphic.**
- **The real hazard — pseudo-label-pool ban + test-touch:** the **transductive** ECALP form adds test↔test edges, i.e. it soft-pseudo-labels unlabeled *test* nodes and lets them vote. That **grazes** `banned_constraints: "kNN-vote-pool expansion via pseudo-labels"` **and** violates the single-test-touch discipline. → The in-box form must be **inductive, train-graph-only** (query→train edges + train↔train edges; propagate; read query rows; no test↔test edges). That form uses exactly the train labels the current kNN vote already uses, adds no pseudo-labels, and is per-item.

**(d) In-box legality.** **IN-BOX** in the inductive train-graph-only form (no gold in the inference path beyond the train labels the vote already consumes; no OCR; no new model; no cross-seed ensemble). **FLAG OUT-OF-BOX / protocol-violating** for the transductive test↔test variant (grazes pseudo-label ban + test-touch). D7-novelty: a decision-operator swap is *not* an encoder-class lever, so it sidesteps the F24 encoder-novelty veto — but whether "propagate instead of vote" clears the D7 *substantive-novelty* bar is a user sub-ruling (it is a known trick from CLIP-adaptation, applied to a new pipeline).

**(e) Implementation cost.** **VERY LOW.** Runs on existing `.pt` key caches; a sparse kNN affinity + ~3 power-iterations on CPU, a few dozen LOC. Deployable exactly as a G0-cond-style **$0 dev probe** (label-oracle calibration arm mandatory per the C3/CTF precedent) before any commitment.

**(f) Honest prior.** **LOW–MODEST (~15–25%)** for ≥+1pt on ≥1 dataset. *For:* it is the one decision operator that attacks the vote *topology*, an axis the campaign never touched; the "oracle-exists-unconvertible" wall (P3/S2S/W2-A) was about auxiliary signals, not about re-aggregating the existing graph, so it is silent here. *Against:* the head is trained (triplet+BCE) to make the fused key **1-hop-separable**, so the manifold is already shaped for the current vote and multi-hop may add little (over-smoothing on a shaped, tiny, label-noisy graph — the archive has known label noise per the memory-editing findings). Best-case target: harden the marginal ZH pass, not open EN.

**(g) Rank: #1** (best prior × cost; strongest non-isomorphism; the sharpest single refutation of "every injection point is closed").

---

## 2. FAMILY B — Vision-tower / projector PEFT (VPT · AdaptFormer · SSF) *(strongest structural refutation)*

**(a) Mechanism.** Every adaptation in the campaign froze the vision tower + multimodal projector and LoRA'd only the LLM backbone (text-decodable targets). Insert **visual prompt tokens (VPT)**, **parallel ViT adapters (AdaptFormer)**, or **scale-shift affine (SSF)** into the Qwen2.5-VL vision encoder — or unfreeze the projector — with the dataset's own word-label SFT, **targeting the F44 failure locus**: MHC-EN's Qwen image stream collapses to near-chance (image-only AUC 0.734→0.599), which is *the* mechanistic reason EN never converts.

**(b) Citation + code.** VPT (Jia et al., ECCV 2022, code released); AdaptFormer (Chen et al., NeurIPS 2022, code); SSF (Lian et al., NeurIPS 2022, code); VIPAMIN (2510.16446, 2025) and AdapterTune zero-init LoRA-for-frozen-ViT (2603.14706) are current refinements. Strong small-data / few-shot evidence (VPT reduces error vs LoRA at <1.5% params).

**(c) Isomorphism check.**
- **vs Law IV two-object closure (F51, `7166232`):** F51 asserts "adaptation has exactly two adapted objects — encoder (generic LoRA) and joint encoder+decision (P9b) — no third object exists." This is the closure this candidate **directly falsifies**: F54 (`6b9985a`) and F58 (`51eb95b`) both record that the vision tower + projector were **frozen in every run**, and the image stream stayed flat *only because SFT targets were text-decodable*. Adapting the vision pathway is a **third adapted object** F51 never measured. **Non-isomorphic** to any LoRA cell (B1/B2/B3/B4/F53), to P9/P9b (decision-level, whole-VLM), and to C3geo/C5 (frozen-geometry mining).
- **Heavy counter-evidence to weigh honestly:** F50 (`e0877c9`) already showed a **healthy** image stream (CLIP-img ⊕ Qwen-text, AUC 0.898 — campaign max on EN) is **unconvertible** (oracle +0.025 < +0.03), and F44/F55 argue MHC-EN is **label-limited, not representation-limited**. So un-collapsing the Qwen image stream may just reproduce the F50 healthy-image ceiling → rotation, not conversion. This does not make the cell isomorphic (the *object* is new), but it caps the EN prior hard.

**(d) In-box legality.** **IN-BOX** on training grounds (own train split, no gold-in-inference, local Qwen-7B). **D7-novelty: likely fails** as encoder-class unless the user rules the vision-adapted object distinct (F24 veto is on encoder-class levers broadly). Needs **GPU training + new adapter wiring** — not a $0 cache operator.

**(e) Implementation cost.** **MEDIUM–HIGH.** New adapter modules into the Qwen vision tower, GPU training (~8 A100-h/dataset/seed on the existing LoRA infra), no cache shortcut.

**(f) Honest prior.** **LOW (~10–20%)** for a *new* dataset (EN), because F50 prices the healthy-EN-image below the bar; somewhat higher for retaining/strengthening HateMM. Its value is **refutation, not performance**: it is the cleanest empirical falsification of the "two-object adaptation closure," worth a $0 pre-check (does a vision-adapter even move EN image-only train-LOO AUC off 0.599?) before any GPU.

**(g) Rank: #4 on prior × cost, but #1 on structural-refutation value** — flagged separately because it is the single finding that most directly contradicts a *stated* closure (F51).

---

## 3. FAMILY C — Single-trajectory weight averaging (SWA) of head checkpoints *(cheap; targets a named failure mode)*

**(a) Mechanism.** Average the head weights across the last-K epochs of **one** training run (one seed) → flatter/wider optimum, better generalization. Directly attacks the F45-measured pathology: "dev saturates ~ep19 while test climbs to ep29 → argmax undershoots; 78-dev selection noise" — SWA over the ep19–29 window recovers the epochs val-selection discards.

**(b) Citation + code.** SWA (Izmailov et al., UAI 2018, in PyTorch core `torch.optim.swa_utils`); Model Soups (Wortsman et al., ICML 2022) — but *soup = multi-run*; SWA = single-run. Generalization-bound refresh 2406.19092 (2024).

**(c) Isomorphism check.** Not isomorphic to any kill. **The one live boundary is the `cross-seed ensembles` ban** — argued precisely: SWA averages weights along **one seed's** trajectory, yielding **one** model at **one** inference cost, whereas the ban targets averaging predictions/weights across **independent seeds**. This is an input-to-the-ban edge case → **flag for a one-line user micro-ruling**, but on the plain text of the ban ("cross-seed") it is in-box. **Feasibility confirmed:** `src/run_rac.py:764` already saves a per-epoch head checkpoint ("Save a checkpoint for every epoch so the fallback path can load any epoch") → SWA runs on **banked artifacts, $0**.

**(d) In-box legality.** **IN-BOX** if single-trajectory averaging is accepted as not-an-ensemble (flag). No new data, no gold, no encoder change. D7: a training-dynamics trick, performance-only (no novelty claim).

**(e) Implementation cost.** **VERY LOW** — average existing per-epoch head checkpoints, re-run the (cheap) head inference. Minutes, CPU/1-GPU-inference.

**(f) Honest prior.** **LOW–MODEST (~15%)**, *targeted*: it is the one operator whose mechanism (F45's dev/test epoch mismatch) is already measured to exist. Most likely payoff = converting the **ZH val-selected** leg that F53/F45 lose to the selection tax, hardening the 2-dataset story under both protocols; low odds of opening a new dataset.

**(g) Rank: #2** (prior × cost — $0 on banked checkpoints, and it aims at a documented failure rather than a generic hope).

---

## 4. FAMILY D — Manifold / embedding mixup + consistency regularization on the head *(cheap small-data regularizer)*

**(a) Mechanism.** During head training, interpolate hidden **embeddings** and their labels (manifold mixup) and/or add a consistency loss across augmented views; smooths the decision boundary — the S2M2 recipe reports +3–8% in the <1k-sample few-shot regime, exactly our scale (train 549–743).

**(b) Citation + code.** Manifold Mixup (Verma et al., ICML 2019, code); S2M2 (Mangla et al., WACV 2020, code, self-sup + manifold-mixup); EMMeT embedding-mixup meta-training (Springer 2023).

**(c) Isomorphism check.** Not isomorphic to any decision-side or encoder kill — it is a **training regularizer on the head over cached embeddings**, an object the campaign never varied (the head loss was fixed triplet+BCE throughout). It is *not* P9b (that couples retrieval loss into the encoder LoRA), *not* pooling/set-matching, *not* threshold. **Non-isomorphic**, but it is a **generic training trick → D7-novelty-dead**, performance-only.

**(d) In-box legality.** **IN-BOX** (own-train cached embeddings, no gold, no encoder touch, no ensemble). D7 fails (generic). Cheap.

**(e) Implementation cost.** **LOW** — head-training-only over cached `.pt` embeddings; ~50 LOC into the head trainer; minutes-to-an-hour on the tiny head.

**(f) Honest prior.** **LOW–MODEST (~10–20%)** on ≥1 dataset; mixup genuinely helps small noisy data, but the head is already margin-regularized and 150-item test gains sit near seed-noise. Best framed as a **margin-hardener** for the marginal ZH pass, companion to Family C.

**(g) Rank: #3.**

---

## 5. FAMILY E — Test-time frame augmentation (multi-view TTA) *(per-item legal variance reduction)*

**(a) Mechanism.** The pipeline reads **fixed 8 frames / fixed neutral prompts**. At inference, sample several frame subsets/augmentations of the *same* video, encode each, average the votes — per-item variance reduction, no labels, no test-set-wide fitting.

**(b) Citation + code.** Intelligent Multi-View TTA (2406.08593, 2024); TTA-meets-Variational-Bayes (2409.12587); CLIP-TTA line. Standard, widely implemented.

**(c) Isomorphism check.** **Non-isomorphic.** Distinct from S2S/CTF (F35/F37/F39): those matched frame-**group causal-prefix structure** and died on cumulative-causal grounds; TTA averages *independent stochastic whole-video reads* to cut variance — no temporal-structure claim, so Law II does not reach it. Distinct from the `cross-seed ensemble` ban — it is an **input-space** augmentation ensemble of **one** model/seed (argue: the ban is model/seed-space). Per-item, so no test-touch violation.

**(d) In-box legality.** **IN-BOX and per-item legal.** Needs re-encoding videos with alternate frame samples (some Qwen encode GPU, cheap; aggregation trivial). D7: performance-only.

**(e) Implementation cost.** **LOW–MEDIUM** — new frame extraction + encode passes (GPU), then trivial mean-vote.

**(f) Honest prior.** **LOW (~10–15%)** — the fixed-8-frame read is already a reasonable video summary; variance reduction on n≈150 buys fractions of a point. Could stabilize the marginal ZH val-sel.

**(g) Rank: #5.**

---

## 6. Families checked and REJECTED (isomorphic, out-of-box, or dominated) — with the reason

| Family | External anchor | Reject reason (ledger cite) |
|---|---|---|
| **Test-time adaptation (TENT/EATA entropy-min)** | TENT ICLR'21, EATA ICML'22 | **OUT-OF-BOX:** adapts model params on test batches → violates single-test-touch; and the head has no BatchNorm (TENT's knob). Reject. |
| **Self-supervised continued pretraining on own train (VideoMAE / contrastive)** | VideoMAE NeurIPS'22 (code) | Arguably in-box (self-sup on own train — **flag for ruling**) but (i) still **encoder-class → D7-dead (F24)**; (ii) train 549–743 videos is ~5× below VideoMAE's data-efficient floor (HMDB51 3.5k). Prior very low. |
| **Conformal / dev threshold co-tuning** | conformal-for-accuracy | **Subsumed by B5/F34 (`50f01b9`):** the label-oracle threshold ceiling is itself <+0.03 both protocols; any operating-point method inherits that null. |
| **Class-balanced loss (balanced-softmax / LDAM / logit-adjust)** | Menon ICLR'21; LDAM NeurIPS'19 (code) | **Not literally subsumed** by B5 (it reshapes the boundary during *training*, not post-hoc), but ~40% positive rate is not a severe imbalance, and it is a generic training trick (D7-dead). Honorable mention, low prior. |
| **Learned audio reps (wav2vec2 / AST / HuBERT) beyond transcript** | MM-HSD uses wav2vec2-xlsr; DeToxy | **APX/F41 (`9c54faf`) ban scope** explicitly requires any audio proposal to first beat a zero-information classical baseline through the same conditional-info screen; transcript already banks spoken content (F31); "add audio" = HateMM 2023 founding contribution (D7-thin); **needs a model download (flag)**. wav2vec2 was not *literally* screened (it is learned, eGeMAPS is not), so a $0 conditional-info gate is the only admissible next step, prior LOW. |
| **MM-HSD cross-modal-attention fusion** | MM-HSD, ACM MM 2025 (`github.com/idiap/mm-hsd`), HateMM 0.878/0.874 — the project's SOTA anchor | Its **only novel operator is OCR-queried CMA** → OCR **vetoed**. Without OCR, CMA over {transcript, audio, video} is a trained fusion module ≈ head-architecture change over frozen features (audio banked-zero F41, text banked); grazes F50/P9b. **No non-OCR operator we lack.** |
| **Tip-Adapter / Proto-Adapter cache-model, ProtoNet head** | Tip-Adapter ECCV'22 (code); Proto-Adapter 2024 | Our RGCL memory **is** a trained key-value cache with a kNN read; Tip-Adapter's extra term is a **zero-shot CLIP-text-prototype** blend — grazes the killed decision-side fusion (F50) and needs CLIP text prototypes we don't build. Prototype memory specifically **killed as W2-E (F28)**. |
| **Trained GNN over the similarity graph** | GraFN SIGIR'22; diffusion-GCN 2503.12563 | The *training-free* LP version is Family A (kept). A *trained* GNN needs the **transductive** graph (test nodes) → test-touch + pseudo-label graze, and a trained message-passing head ≈ P9b/head-training territory. Reject the trained/transductive form; keep the inductive training-free LP. |

---

## 7. TOP-5 — ranked by prior × cost (refutation value noted separately)

| # | Family | In-box? | Cost | Prior (≥+1pt/≥1 dataset) | Non-isomorphism strength | Cheapest next step |
|---|---|---|---|---|---|---|
| **1** | **Label propagation / graph diffusion over the kNN memory graph** (ECALP) | YES (inductive train-graph-only; **transductive = OUT-OF-BOX**) | **$0 CPU** on cached keys | **~15–25%** | **HIGH** — attacks the vote *topology*; outside F46's linear-zero + named-nonlinear-operator list; not F47 selection | $0 dev probe, label-oracle calibration arm mandatory |
| **2** | **Single-trajectory SWA of per-epoch head checkpoints** | YES (flag cross-seed micro-ruling) | **$0** on banked ckpts (`run_rac.py:764`) | **~15%**, *targeted at F45 ZH val-sel tax* | MED — new object (training dynamics), grazes ensemble-ban edge | Average ep-window ckpts, re-infer |
| **3** | **Manifold/embedding mixup + consistency reg on head** | YES (D7-dead / perf-only) | **LOW** on cached embeddings | **~10–20%** | MED — head-loss object never varied | ~50 LOC into head trainer |
| **4** | **Vision-tower/projector PEFT** (VPT/AdaptFormer/SSF) | YES-train / **D7-adjacent** | **MED–HIGH** (GPU) | **~10–20%** (EN-capped by F50) | **HIGHEST** — falsifies F51 "two-object closure"; vision path never adapted | $0 pre-check: does a vision adapter move EN image-only AUC off 0.599? |
| **5** | **Test-time frame augmentation (multi-view TTA)** | YES (per-item legal) | LOW–MED (re-encode) | **~10–15%** | MED — variance reduction, not temporal structure (escapes Law II) | Re-sample frames, mean-vote |

**Bottom line for the exhaustion claim.** Two of these (Family A decision-topology, Family B vision adapted-object) are un-enumerated *injection points*, not just un-tried tactics — so `TERMINUS_round3`'s "no untried injection point remains inside the box" is **too strong as literally worded**. The honest correction is narrower and survives: *every injection point the campaign enumerated is closed, and the strongest remaining openings carry guarded performance priors (≤~25%) and are D7-novelty-adjacent, but they are real openings and at least two are $0 to falsify.* The highest-EV move is the **Family A $0 label-propagation dev probe** (cheap, most non-isomorphic), with **Family C SWA** as a $0 companion aimed at the one measured failure mode (the ZH val-selection tax).

---

## Provenance

- Kill ledger: `autoresearch/goal_mllm_plus3/state/findings.jsonl` (F1–F60), `state/directions_tried.json`, `refine-logs/TERMINUS_round3_mllm_plus3.md`, `research-wiki/DRAFT_analysis_chapter.md` (Laws I–IV, §3.6–3.9).
- Feasibility check: `src/run_rac.py:764` (per-epoch head checkpoints saved) — enables $0 SWA.
- External anchors (2023–2026, code-released unless noted): ECALP ICLR 2025 (arXiv 2412.18303, `github.com/Yushu-Li/ECALP`); Iscen CVPR 2019 LP; VPT ECCV 2022; AdaptFormer NeurIPS 2022; SSF NeurIPS 2022; VIPAMIN 2510.16446; SWA UAI 2018 (`torch.optim.swa_utils`); Model Soups ICML 2022; Manifold Mixup ICML 2019; S2M2 WACV 2020; Multi-View TTA 2406.08593; TENT ICLR 2021 / EATA ICML 2022 (rejected); VideoMAE NeurIPS 2022 (flagged); MM-HSD ACM MM 2025 (`github.com/idiap/mm-hsd`, HateMM 0.878/0.874, OCR-based); Tip-Adapter ECCV 2022.
- **Discipline:** CPU-only, zero GPU/SLURM/Modal/downloads; `state/` unmodified. Cloud/external numbers are triage context only, never mixed with local G-repro numbers.
