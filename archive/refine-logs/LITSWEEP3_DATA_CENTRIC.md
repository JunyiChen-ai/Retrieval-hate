# LITSWEEP3 — Training-Data-Centric Levers Inside the Own-Split Box

**Agent:** litsweep-3 #L3 (literature sweep ROUND-4 of 5) · **Date:** 2026-07-25 · **ZERO GPU / ZERO SLURM / ZERO Modal / ZERO test-touch.** Reading + web-verified citations + prereg-shaped design only. `autoresearch/state/` untouched. Deliverable = this doc + one local commit (no push).

**Lens (unique to this agent):** *training-data-centric* levers inside the single-dataset own-train-split box — the family never systematically enumerated. Five sub-axes: (1) feature-space augmentation for small-n heads/memories; (2) label-noise handling for the head; (3) memory-bank curation learned from train labels; (4) class-imbalance / macro-F1-aware losses over a kNN vote; (5) SFT-example selection/weighting from own split.

**All citations below are web-verified** (arXiv ID and/or venue+year confirmed 2026-07-25). PAPER-VALUE list and user-ruling flags at the end.

---

## 0. Grounding — the data path, the box, and the three walls every data-centric operator must clear

**Pipeline (code-confirmed, carried from `LITSWEEP2_HEAD_OBJECTIVES.md` §0, `src/model/{loss,classifier}.py`):** frozen/LoRA Qwen2.5-VL-7B dual-stream features → per-stream `Linear→Dropout`, L2-norm, **Hadamard fuse**, MLP → embedding. **Training loss = triplet-margin (cosine, m=0.1) over FAISS-mined pseudo-gold-positive + hard-negative pairs, + ~0.5·BCE.** **Inference = top-20 similarity-weighted kNN vote over the banked OWN-TRAIN memory** (not the logit). Head is tiny: ZH bank n=579; per-dataset train n≈0.6–4k; test ~150–500; head-seed noise ±0.014.

**Box (binding):** no gold annotations in deployed path (train labels OK); no OCR; **single-dataset own-train-split only — no external data, augmentation must derive from own train**; no cross-seed ensembles; no closed-model APIs; raw video never leaves the machine; only local models (Qwen2.5-VL-7B, CLIP); no reimplementing codeless baselines.

**The three walls (all banked, all brutal) that pre-price every candidate in this lens:**

- **Wall-A — F66 arithmetic selection-lock (`a6e41f8`).** The ISR β-decomposition proved **91–98 % of the ZH/EN oracle headroom is reachable ONLY by per-item (per-test-instance) selection**, which law-III bans; the *symmetric/legal* slice is **+0.001–0.006**. Every training-side data operator (better loss, cleaner memory, denoised training, re-weighted examples) is a **symmetric** operator. So the +3-via-headroom-conversion story is **arithmetically capped near zero on ZH/EN**.
- **Wall-B — EN is label-limited, not representation-limited** (F44/F50/F55, 5 proven levels; best-ever fusion AUC 0.898 unconvertible). *Input/training* interventions cannot manufacture *label* signal → **EN stays dead for the whole lens**.
- **Wall-C — HateMM already PASSES both protocols (F53); the only live perf target is ZH, and the ZH miss is a *dev-SELECTION* failure, not a representation failure** (F45: 78-item dev plateaus ~ep19 while **test keeps climbing to ep29**; LoRA is already the *most stable* arm). Two consequences that gut most of this lens:
  1. **Train-side operators add zero dev items and cannot change the dev-argmax** that selects the checkpoint (the AUG kill's decisive point, F60). Only a *flatter-generalizing embedding that makes dev-argmax and test-optimal coincide better* (a variance-reduction story) can touch val-sel indirectly — and that exact bet was measured **dead** for manifold-mixup (**F75-A3**).
  2. **ZH test-accuracy climbs late.** Any noise-robust / early-stop / trajectory-shrinking mechanism (co-teaching's small-loss drop, late-stopping's example removal, SWA's averaging) is **anti-aligned** with ZH dynamics — the identical shape that killed SWA on HateMM's mid-peak (F62) and is present on ZH's late-climb. **Price this against every "stop learning the noisy tail" method below.**

**Consequence that governs every honest prior in this doc:** the realistic deliverables for a data-centric operator are **(i) ZH val-sel *stabilisation* (variance reduction), (ii) a macro-F1 move on the mF1 half of the bar, or (iii) PAPER-VALUE (pillar-3 denoising / pillar-4 auditable memory)** — **not a new +3 dataset**. Three prior independent sweeps (F68/F74) converged on "in-box ≥+3-on-2-datasets unreachable"; this sweep confirms it *at the data-centric enumeration level* and surfaces the 3 cells that best survive as diagnostic/paper probes.

---

## 1. What is already DEAD or DOMINATED in this exact lens (so nothing below re-proposes it)

| Prior result | Finding | What it forecloses in *this* lens |
|---|---|---|
| **AUG** MLLM data-generation | F60 (`f1abd28`) | Example-*synthesis* for encoder-SFT: dominated by cand-2 ZH TIE; val-sel = dev-selection noise untouchable from train side; no cheap gate. Any *generation* variant re-enters this. |
| **cand-2** hard-neg SFT curriculum | F56/F59 | Example-*distribution/pairing/weighting* for the ZH encoder-SFT: measured **TIE both protocols**, "ZH-robustness NOT strengthened." `queued`: "do NOT re-run curriculum variants (tactics) without new structural premise." |
| **NCA/SupCon/manifold-mixup** head-loss family | F75 (`f03cae0`) | **Naive/manifold mixup (A3, α=2.0) MEASURED DEAD.** Loss-swaps toward vote-consistent/contrastive/mixup banned. **First measured negative for "trained reshaping unlocks oracle headroom."** |
| **archive-auto-repair** MLLM two-vote deletion | dead-list | *MLLM-driven* memory deletion = guard-rail role only; AND-rule C−A=0; embedding-only over-deletes. (Train-label-only deletion is a *different signal source* — see §4.) |
| **kNN-vote-pool expansion via pseudo-labels** | banned_constraints | Adding pseudo-labelled synthetic entries to the memory bank: banned ("representation-training expansion only"). Gates feature-space *memory-populating* augmentation (§2). |
| **P11 / MLLM-scores-as-training-signal** | banned_constraints | MLLM scores as weak-sup labels. (Gold-label-preserving operators are outside this.) |
| **LP graph-diffusion** over the kNN memory graph | F63 (`7be6e3f`) | Not this lens, but its finding is the key headwind for §4: **the 1-hop top-20 vote already reads the extractable signal; perm-null center is POSITIVE (diffusion helps random labels more than real).** |

**Human 2-entry EN memory deletion helped EN** (banked positive, MEMORY.md, human-in-the-loop only). That is the *only* banked positive anywhere in this lens, and it is the precedent §4 tries to automate with a **train-label-only** (not MLLM, not human) signal.

---

## 2. Feature-space augmentation for small-n heads/memories

**Family & verified cites.** mixup (Zhang et al., ICLR 2018, arXiv:1710.09412); Manifold Mixup (Verma et al., ICML 2019, arXiv:1806.05236); **ISDA — Implicit Semantic Data Augmentation** (Wang et al., NeurIPS 2019, arXiv:1909.12220), class-conditional feature-space augmentation via per-class covariance with a closed-form upper-bound loss (i.e. *not* naive interpolation).

**Honest transplant & three sub-cells:**
- **(a) mixup / manifold-mixup at the proj head.** **DEAD** — F75-A3 measured α=2.0 manifold-mixup-BCE inside the KS-dead family (7/8 cells). Retunes = banned tactics.
- **(b) ISDA / difficulty-aware / feature-adversarial augmentation of the head training set.** *More sophisticated than mixup* (semantic covariance directions, not convex mixes), so **not literally F75-A3**. **But dominated:** it is still a **symmetric variance-reduction regularizer of the head/BCE path** whose entire achievable target is the same ZH val-sel slice mixup just missed; the covariance is estimated from ~600 binary samples (ISDA's own failure regime is small-n/binary where the per-class covariance is rank-deficient and noisy); and, decisively, **the kNN vote reads the REAL banked memory** — head-side augmentation regularizes the embedding map but the neighbours the vote returns are unchanged, so the effect on the *deployed decision* is second-order (the same friction LITSWEEP2 flagged for mixup). **Prior ≥+1 ~5 %, +3 ~1 %.** No promotion.
- **(c) within-class feature interpolation to POPULATE the memory bank** (densify the kNN pool with same-class synthetic keys). This is the *only* feature-aug variant that would change what the vote reads — **but it collides with `banned_constraints`: "kNN-vote-pool expansion via pseudo-labels."** A within-class interpolation carries a *certain* (not pseudo) label, so the literal wording ("pseudo-labels") is arguable, but the *mechanism* ("vote-pool expansion") is squarely the banned object, refuted-as-posed by 3 lit scouts. **FLAG: USER-RULING** required before any spend; default = banned. If ever legalised it is the highest-novelty feature-aug cell, but the F66 wall still caps it (synthetic keys add no headroom the real keys don't span).

**Verdict for the lens:** feature-space augmentation contributes **no promotable cell**; (a) dead, (b) dominated, (c) banned pending user ruling.

---

## 3. Label-noise handling for the head (the one mechanism genuinely distinct from F75)

**Family & verified cites.** Co-teaching (Han et al., NeurIPS 2018, arXiv:1804.06872 — dual-net small-loss selection); **ELR — Early-Learning Regularization** (Liu et al., NeurIPS 2020, arXiv:2007.00151 — an *additive regularizer* pulling predictions toward their own early-epoch moving-average target, prevents memorization of noisy labels); **SOP — Robust Training by Over-parameterization** (Liu et al., ICML 2022, arXiv:2202.14026 — sparse over-parameterization models & subtracts the label-noise term); **Late Stopping** (Yuan, Feng, Liu, ICCV 2023, arXiv:2308.13862 — prolonged training that *removes high-confidence-mislabeled* examples while *retaining clean-hard* ones, explicitly criticising small-loss selection for discarding clean-hard examples).

**Why it is genuinely un-enumerated (isomorphism check, passed).** F75 swapped the loss *family* (triplet→NCA/SupCon/mixup). **Noise handling is orthogonal to the loss family:** ELR *adds a term* to the existing BCE; co-teaching changes the *training procedure* (two heads, peer small-loss selection); late-stopping changes the *effective training set over time*. None is a "loss-swap toward vote-consistent/contrastive/mixup," so none is inside F75's `ban_scope`; none is a channel (C3), a score-label (P11), a pseudo-label pool expansion, or the cand-2 curriculum (which *re-weighted* mined pairs but never modelled/removed *label noise*). **The target is real and named in the tasking:** (i) the head's **FAISS-mined pseudo-gold positives are self-admitted label noise** in the triplet term; (ii) both benchmarks carry **documented human-annotation noise** (the reason pillar-3 consensus-denoising was ZH-validated in this project). Nobody has applied a *noise-robust training* operator to the head.

**Why it is nonetheless NOT a +3 lever (be brutal, price all three walls):**
- **Wall-A caps headroom conversion.** Noise-robust training is a *symmetric* operator; F75 already showed trained reshaping does not unlock the selection-locked headroom. So the only live story is ZH val-sel *variance reduction* — the same slice mixup (F75-A3) missed.
- **Wall-C anti-alignment (the decisive headwind).** ZH **test climbs to ep29** while dev plateaus; co-teaching (drop large-loss late) and late-stopping (remove "mislabeled" late) both *shrink late-epoch learning* — exactly what SWA's kill (F62) and the "clean-hard has large loss" caveat of the Late-Stopping paper itself warn against. On a dataset whose test *keeps improving late*, "stop trusting the hard tail" is the wrong prescription. **ELR is the safest member** (it does not drop examples, only regularizes toward the *early-epoch* target — which, on ZH where the early-epoch model is *worse* than late, is again anti-aligned; ELR's implicit early-stopping bias fights the late climb).
- **n≈600 binary, ~40 % positive:** co-teaching/DivideMix-style small-loss partition is unstable at this scale; the mislabeled/clean loss distributions overlap heavily with only ~250 positives.

**Honest prior.** ZH val-sel hardening ≥+1 (stable, 3/3): **~6–10 %** (ELR the only arm not anti-aligned, and even it is weakly anti-aligned); +3 any dataset: **~2 %**. **Cost:** head-only re-train, **~0.3 GPU-h for a 3-arm × 3-seed family** (= F75's footprint). **DIAGNOSTIC value if it fails:** converts "the head trains on noisy mined pairs" from a hypothesis into a measured non-factor — a clean paper sentence and the natural companion to the ZH-validated consensus-denoising (pillar-3).

**This is the lens's technically-cleanest new axis, but Wall-C makes it a variance bet against the grain of ZH's own dynamics.** Recommend **one** small pre-registered probe (ELR-additive as the lead arm, co-teaching as a contrast arm, both vs the F75 floor), *only if* the orchestrator wants the noisy-mined-pair hypothesis closed on the record.

---

## 4. Memory-bank curation learned from TRAIN LABELS (the lens's best cost/value — $0, paper-value, precedent)

**Family & verified cites.** Prototype selection taxonomy (Garcia, Derrac, Cano, Herrera, IEEE TPAMI 2012); Prototype selection for interpretable classification (Bien & Tibshirani, Ann. Appl. Stat. 2011, arXiv:1202.5933); Condensed Nearest Neighbour (Hart, IEEE Trans. IT, 1968); Coresets for the Nearest-Neighbor Rule (Flores-Velazco & Mount, ESA 2020, arXiv:2002.06650); **Data-OOB** out-of-bag data valuation, *state-of-the-art at identifying mislabeled / harmful points, <2.25 h CPU at 10⁶ pts* (Kwon & Zou, ICML 2023, arXiv:2304.07718); Data Shapley (Ghorbani & Zou, ICML 2019, arXiv:1904.02868).

**The question the tasking poses:** *which own-train items should even BE in the kNN memory?* This is a **symmetric, train-label-only, global** edit of the deployed bank — categorically different from the three dead siblings:
- ≠ **archive-auto-repair** (F-guard-rail): that used an **MLLM two-vote** signal; here the deletion signal is **gold train labels + kNN geometry only**. Different information source. The AND-rule C−A=0 finding (MLLM re-finds the human ids but embedding-only over-deletes) is a *headwind to price*, not a coverage of this mechanism.
- ≠ **pseudo-label pool expansion** (banned): that *adds* pseudo-labelled items; curation *removes* gold-labelled items. Opposite direction, gold labels.
- ≠ **F47 per-item routing / F66 per-item selection** (law-III): those select **per test instance**; curation selects **train items once, globally, applied identically to every test query** — a symmetric operator, so law-III/F66's per-item ban does not apply to the *mechanism* (though Wall-A still caps the achievable magnitude).

**Why it is the best cost/value here:**
1. **It changes the actual deployed vote** (unlike §2/§3 training-regularizers whose effect on the real-neighbour vote is second-order): remove a harmful/mislabeled train memory and every test query's neighbour set changes symmetrically. So it can move **final-epoch test** directly, and it is the one lens member not fully gated by Wall-C's "can't touch dev-argmax."
2. **It is ~$0 on the already-banked keys** — the F63/W2B machinery (cached fused keys + deployed top-20 signed-cosine vote) is exactly the substrate. **LOO-kNN-influence pruning** (for each memory item, Δ in train-LOO or dev kNN accuracy when it is dropped) is pure kNN arithmetic; **Condensed-NN** and **coreset** condensation are greedy passes on cached distances; **Data-OOB** on n≈600 is minutes of CPU. **No GPU, no re-extraction, no test-touch to *find* the pruned set** (test only spent once at the pre-registered verdict).
3. **It has a banked positive precedent** (human 2-entry EN deletion helped EN) and it is the **principled auto-version** of it — direct pillar-4 (auditable/editable archive) content.
4. **It carries the mF1 arm for free** (§5): a **class-balanced bank** (prune/undersample the majority class, or prototype-select to equal per-class counts) directly rebalances the similarity-weighted vote's neighbour-class frequency — the one operator that targets the *macro-F1* half of the bar via the memory rather than a (B5-dead) threshold.

**Why it is nonetheless low-prior (be brutal):**
- **Wall-A** still caps the magnitude: curation is symmetric; the convertible ZH/EN headroom is 91–98 % per-item-selection-only, so bank-cleaning can recover at most the +0.001–0.006 legal slice on the *headroom* story. Its live target is the same ZH val-sel stabilisation + the (thin) mF1 rebalance.
- **F63's decisive headwind:** the 1-hop top-20 vote **already reads the extractable signal**, and its perm-null center is **positive** (operations on this graph help *random* labels more than real). Pruning a graph that is already 1-hop-separable risks removing the **clean-hard** memories (Late-Stopping's exact warning) and *lowering* the vote on a tiny bank.
- **n≈600 makes every valuation estimate noisy** — Data-OOB/Shapley variance is large at this scale, and the human precedent was **2 entries**, i.e. a hand-audited micro-edit, not a learned mass prune.

**Honest prior.** ZH/EN val-sel hardening or mF1 move ≥+1 (stable): **~8–12 %**; +3 any dataset: **~1–2 %**. **Cost:** **~$0-CPU** to select the pruned bank (reuse F63 keys) + **one 3-seed head-vote re-eval** to verdict. **Highest PAPER-VALUE in the lens** (pillar-4 auditable memory + automates the banked human-deletion positive). **Recommend as the round LEAD $0 probe** on cost/value grounds, pre-registered against ZH val-sel + a macro-F1 secondary, with F63's perm-null as the honest floor and F66's legal slice as the oracle ceiling.

---

## 5. Class-imbalance / macro-F1-aware operators over the kNN vote

**Family & verified cites.** Balanced Meta-Softmax (Ren et al., NeurIPS 2020) and Long-tail Learning via Logit Adjustment (Menon et al., ICLR 2021) — the two are the same balanced-softmax object (label-frequency logit shift). Influence-Balanced Loss (Park et al., ICCV 2021, arXiv:2110.02444).

**Honest transplant & the B5 collision.** The bar includes **macro-F1**, and the deployed vote is class-imbalance-sensitive via **neighbour-class frequency in the bank**. Two injection points:
- **(a) logit-adjusted / class-prior-corrected vote** (shift the vote by log class base-rate). This is **decision-side threshold territory** — and **B5 (`50f01b9`) killed per-encoder operating-point calibration by oracle**: the ZH ranking edge is *easy-example ordering*, unconvertible at *any* threshold **including the label-oracle**. A class-prior correction is a monotone threshold move → the **B5 oracle logic covers it → DEAD**. Also inert if applied to the BCE head (the vote, not the logit, decides). **No promotion.**
- **(b) class-balanced MEMORY** (equal per-class bank via undersample/prototype-select). This changes *which neighbours are available*, not a threshold, so it is **NOT covered by B5's threshold-oracle** — it is a **special case of §4's curation** and inherits §4's $0 cost and low-but-nonzero prior. **Fold into §4 as the mF1 arm**, do not run standalone.

**Verdict:** the only non-dead class-imbalance operator is the balanced-bank arm, which lives inside §4. Loss-level balancing (a) is B5-oracle-dead on the vote.

---

## 6. SFT-example selection / weighting from own split

**Family & verified cites.** **LESS** — influence-based instruction-data selection, "5 % selected can outperform full data" (Xia, Malladi, Gururangan, Arora, Chen, ICML 2024, arXiv:2402.04333); gradient/influence data-selection lineage generally.

**Honest transplant & domination.** LESS-style influence selection of the LoRA-SFT training examples from the own train split is the *encoder-adaptation* analogue of §4. **Dominated on two counts:** (i) the adapted object is still the **encoder** → F51 two-object closure re-entered, D7-encoder-class-dead, and Wall-B/C bar any new dataset (HateMM inherits, EN label-limited, ZH = the only target); (ii) **cand-2 already ran a data-selection/curriculum intervention on the identical ZH SFT and TIED** (F56), and **F60 (AUG) already priced the whole "change which/what SFT examples" family as dominated-with-no-cheap-gate**. Influence-selection is a *different selection criterion* but the *same object/leg/split/failure-mode*, with real GPU cost (~7–9 A100-h/dataset like AUG) and **no $0 screen**. **Prior ≥+1 ~4 %, +3 ~1 %.** **No promotion**; bank as pre-priced so wave-N+ never re-spends. Revisit only under a user D7 sub-ruling (same gate as F60).

---

## 7. Isomorphism + ban ledger (each promotable/flagged cell vs the dead list)

| Cell | C3/P4/P11 | cand-2 curric | F75 loss-swap | F60 AUG | archive-repair | F47/F66 per-item | pool-expansion | Verdict |
|---|---|---|---|---|---|---|---|---|
| §4 train-label bank curation | clean (removes gold items, not a feature/score) | clean (edits memory, not SFT set) | clean (no loss change) | clean (no generation) | **distinct signal** (gold labels, not MLLM) | **symmetric** (global, not per-item) | **opposite** (removes, not pseudo-adds) | **ADMISSIBLE — LEAD ($0)** |
| §3 noise-robust head (ELR/co-teach) | clean | clean (models noise, not re-weights mined curriculum) | **outside letter** (additive reg / procedure, not family-swap) | clean | n/a | symmetric | clean | **ADMISSIBLE — probe** |
| §5b balanced bank (mF1) | clean | clean | clean | clean | distinct | symmetric | opposite | **= §4 arm** |
| §2c memory-populating feat-interp | clean | clean | ≠ mixup-BCE (adds keys) | clean | n/a | symmetric | **HITS pool-expansion** | **USER-RULING (default banned)** |
| §2b ISDA/adversarial feat-aug | clean | clean | ≠ A3 literally | clean | n/a | symmetric | clean | dominated, no promo |
| §5a logit-adjusted vote | clean | clean | clean | clean | n/a | symmetric | clean | **B5-oracle DEAD** |
| §6 LESS SFT selection | clean | **same object/leg/split as cand-2** | clean | **F60-dominated** | n/a | symmetric | clean | dominated, no promo |

---

## 8. RANKED SHORTLIST (max 3) — prior × cost, honest

| # | Candidate | Verified cite | P(≥+1 stable; mostly ZH val-sel / mF1) | P(≥+3 any ds) | Cost | Why not already dead |
|---|---|---|---|---|---|---|
| **1** | **Train-label memory-bank curation** (LOO-kNN-influence / Data-OOB / condensed-NN / prototype-select prune of the deployed bank; **balanced-bank arm for mF1**) | Data-OOB ICML23 (2304.07718); CNN Hart'68; Bien-Tibshirani AoAS'11 (1202.5933); Coresets-NN ESA20 (2002.06650); Garcia TPAMI'12 | **~8–12 %** | ~1–2 % | **~$0-CPU on banked keys.** Symmetric global edit (≠ F47/F66 per-item), gold-label signal (≠ MLLM archive-repair), removal (≠ pool-expansion); changes the *deployed vote directly*; automates the banked human-2-entry-EN positive; **highest paper-value (pillar-4).** |
| **2** | **Noise-robust head training** (ELR-additive lead arm; co-teaching contrast arm; on the FAISS-mined-pair noise) | ELR NeurIPS20 (2007.00151); Co-teaching NeurIPS18 (1804.06872); SOP ICML22 (2202.14026); Late-Stopping ICCV23 (2308.13862) | **~6–10 %** | ~2 % | **Orthogonal to F75** (regularizer/procedure, not loss-family-swap); the one operator that models the head's *self-admitted mined-pair label noise*; **~0.3 GPU-h**. Anti-aligned with ZH late-climb → variance bet only. |
| **3** | **Memory-populating within-class feature interpolation** *(FLAG — user-ruling)* | Manifold-Mixup ICML19 (1806.05236); ISDA NeurIPS19 (1909.12220) | ~5 % *(if legalised)* | ~1 % | Highest-novelty feature-aug, only variant that changes what the vote reads — **but hits the vote-pool-expansion ban; default banned, needs a user ruling.** F66 caps it even if legal. |

**Bottom line for the orchestrator.** The data-centric family — swept here for the first time end-to-end — contains **exactly one clean, cheap, non-dead, paper-valuable cell: train-label memory-bank curation (#1)**, worth **one LEAD $0-CPU probe** because it (a) reuses banked keys, (b) is the principled auto-version of the only banked positive in the lens, (c) targets both the ZH val-sel and the mF1 halves of the bar, and (d) is pillar-4 paper material *whether or not* it moves the number. **#2 (noise-robust head)** is the technically-cleanest *new* axis but Wall-C's ZH late-climb makes it a variance bet against the grain — run it only to close the noisy-mined-pair hypothesis on the record (~0.3 GPU-h). **#3** is flagged for a user ruling (default banned). Everything else in the lens is **dead (mixup F75, logit-vote B5), dominated (ISDA, LESS/cand-2/F60), or banned (pool-expansion, MLLM-gen F60)**. **No cell carries a defensible ≥+3-on-a-new-dataset prior** — Walls A/B/C are the binding constraints, exactly as F68/F74 concluded; this sweep confirms it at the data-centric enumeration level.

---

## 9. Minimal-decisive-cell sketches + kill-switches

**CELL-1 (LEAD) — memory-bank curation, $0-CPU selection + one 3-seed head-vote verdict.**
- *Selection (train-label-only, no test-touch, no GPU):* on the banked ZH (and EN as a paper-value companion) fused keys, compute a per-memory value with **(a) LOO-kNN influence** (Δ train-LOO top-20 vote accuracy on removal) and **(b) Data-OOB**; prune the lowest-value ~5–15 % and, as the mF1 arm, **class-balance** the retained bank. Freeze the pruned index list *before* any test read.
- *Verdict:* re-run the deployed top-20 signed-cosine vote over the pruned bank, 3 seeds, both protocols, vs the intact-bank floor (4dp, G-repro).
- **Kill-switches (house style):**
  - **K-CUR-0 ($0, kill-only):** the pruned set must (i) *reproduce the deployed decisions bit-exact when prune-fraction→0* (machinery parity, like F63's α→0 check), and (ii) beat the **perm-null** built by pruning *random* equal-size sets — if the learned prune is inside the random-prune null (F63's positive-center warning), **auto-KILL** (curation reading no more than random removal).
  - **K-CUR-1 (primary):** ZH pruned-bank − intact-bank ≥ +0.014 *and* clears val-sel per-seed 3/3 (strengthen, don't tie); anything ≤ intact = no value.
  - **K-CUR-2 (mF1 arm):** balanced-bank macro-F1 gain ≥ +0.030 3/3 (the bar's mF1 half) with no acc regression.
  - **K-CUR-3 (sanity):** HateMM pruned bank must not regress below its inherited pass; over-deletion tripwire (if best cell needs >25 % prune, flag as the archive-repair over-deletion pattern C−A).

**CELL-2 — noise-robust head, ~0.3 GPU-h.**
- *Arms:* **A-ELR** (BCE + λ·ELR moving-average regularizer, λ∈{1,3} declared once), **A-COT** (dual-head co-teaching small-loss keep-rate = 1−est.noise), both on the existing triplet+BCE head, ZH + HateMM × 3 seeds, vs the **F75 floor** (not just CLIP).
- **Kill-switches:**
  - **K-NR-0 ($0):** estimated mined-pair noise-rate must be >5 % (else there is no noise to be robust to ⇒ auto-KILL "no target").
  - **K-NR-1 (primary):** ZH val-sel strengthened 3/3 over F75 floor by ≥+0.014; **KS-arm-dead** if any arm ≤ +0.020 (F75/F70 precedent).
  - **K-NR-2 (Wall-C tripwire):** if the winning arm's dev-argmax epoch *moves earlier* than the intact head's while ZH test is still climbing, flag anti-alignment (the SWA/F62 failure) even on a nominal pass.

---

## 10. PAPER-VALUE list (quantifiable/citable regardless of perf outcome)

- **PV-1 (top find):** §4's **LOO-influence / Data-OOB valuation of the memory bank** produces, at **$0**, a *quantified* "how much of the vote is carried by how few memories" curve — the **automated, principled companion to the banked human-2-entry-EN deletion (pillar-4 auditable/editable archive)**, and the natural quantitative partner to the **ZH-validated consensus-denoising (pillar-3)**. Cite Data-OOB (2304.07718), Bien-Tibshirani (1202.5933).
- **PV-2:** §3's noise-robust probe, even if null, **measures the head's mined-pair label-noise contribution** — closes a hypothesis the campaign has only asserted; cite ELR (2007.00151), Late-Stopping (2308.13862).
- **PV-3:** §5's B5-oracle collision is a **clean statement that macro-F1 on this vote is threshold-unreachable** (strengthens the B5/Wall-A story); cite Logit-Adjustment ICLR21, Balanced-Softmax NeurIPS20.
- **PV-4 (meta):** this doc is the **first end-to-end enumeration of the data-centric family** showing every sub-axis dies on Walls A/B/C — a defensible "we exhausted the training-data-centric axis" paragraph for the limitations/analysis chapter, corroborating F68/F74's convergent verdict.

---

## 11. USER-RULING flags

- **§2c memory-populating within-class feature interpolation** — default **BANNED** (vote-pool-expansion, `banned_constraints`); needs an explicit user relaxation to run. Highest-novelty feature-aug if legalised, but F66-capped.
- **§6 LESS/influence SFT-example selection** — needs a **D7 generator/selection-role sub-ruling** (same gate as F60 AUG) *and* acceptance of a prior strictly weaker than cand-2's measured TIE.
- **Cross-seed-ensemble adjacency:** §2/§3 are single-trajectory head trainings — no ensemble; §4 curation is a single pruned bank — no ensemble. **None trips the cross-seed veto** (noted for completeness; §4/§3 are clean, unlike SWA/EMA which needed the F62 micro-ruling).

---

## PROVENANCE & required statements
- Pipeline/loss/inference grounding: `LITSWEEP2_HEAD_OBJECTIVES.md` §0 (`3e89c9b`), `src/model/{loss,classifier}.py`.
- Walls: F66 `ISR_PREGATE_RECORD.md` (`a6e41f8`); F44/F50/F55 EN label-limit; F45 `B3_ZH_LORA_DECOMPOSITION.md` (`d76e407`, ZH late-climb + 78-dev selection noise); F53 HateMM pass; F62/F62b SWA; F63 `LP_GATE_RECORD.md` (`7be6e3f`, 1-hop-reads-signal + positive perm-null); F75 `NCA_VERDICT_REVIEW.md` (`f03cae0`, mixup A3 dead); F60 `AUG_FORENSIC_RECON.md` (`f1abd28`); F56/F59 cand-2 ZH TIE; convergent-unreachability F68/F74.
- Bans: `state/directions_tried.json` dead[] + banned_constraints[]; human-2-entry-EN deletion positive: `MEMORY.md`.
- Verified citations (web, 2026-07-25): ELR arXiv:2007.00151 (NeurIPS 2020); Co-teaching arXiv:1804.06872 (NeurIPS 2018); SOP arXiv:2202.14026 (ICML 2022); Late-Stopping arXiv:2308.13862 (ICCV 2023); Data-OOB arXiv:2304.07718 (ICML 2023); Data-Shapley arXiv:1904.02868 (ICML 2019); Prototype-selection Bien-Tibshirani arXiv:1202.5933 (AoAS 2011) + Garcia TPAMI 2012 + Hart 1968; Coresets-NN arXiv:2002.06650 (ESA 2020); ISDA arXiv:1909.12220 (NeurIPS 2019); Manifold-Mixup arXiv:1806.05236 (ICML 2019); mixup arXiv:1710.09412 (ICLR 2018); LESS arXiv:2402.04333 (ICML 2024); Balanced-Softmax (Ren et al., NeurIPS 2020); Logit-Adjustment (Menon et al., ICLR 2021); Influence-Balanced-Loss arXiv:2110.02444 (ICCV 2021).
- **ZERO GPU / SLURM / Modal spent. No held-out test metric read or produced. No `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated. Committed on `main`, not pushed.**
