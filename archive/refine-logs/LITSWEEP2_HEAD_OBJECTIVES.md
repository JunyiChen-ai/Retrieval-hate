# LITSWEEP2 — Training Objectives / Losses / Regularization for the tiny retrieval head

**Agent:** literature-sweep ROUND-2 #1 · **Date:** 2026-07-25 · CPU-only, no GPU/SLURM/Modal, autoresearch state untouched.
**Scope:** operators that change the *training objective* of the ~600-sample, 1024-d projection head (frozen features in), read by a top-20 cosine kNN vote out. NOT encoders, NOT channels, NOT selection/averaging.

---

## 0. What we currently do (grounded in code, not prose)

- `src/model/loss.py::compute_loss`: **triplet-margin contrastive** on the fused embedding `feats` (metric = cosine, `--triplet_margin 0.1`), with a per-anchor label-coincidence matrix giving **in-batch positives + in-batch negatives**, plus **FAISS-mined pseudo-gold positive + hard negative** (`utils/retrieval.dense_retrieve_hard_negatives_pseudo_positive`), **plus `--hybrid_loss` BCE** at weight ~0.5 (`--cross_entropy_weight`, `--positive_weight`).
- `src/model/classifier.py::forward`: img/text → `Linear→Dropout` proj, **L2-normalize each stream**, fuse by **Hadamard product** (`fusion_mode='align'`, `x = img*text`), MLP, `output_layer→1`. Embedding = `mlp[:-2](x)`.
- **Inference = top-20 similarity-weighted kNN vote over the banked memory** (NOT the logit).
- AdamW lr 1e-4 wd 1e-4, 30ep, batch 64.

**The load-bearing observation the mission names is real and code-confirmed:** the loss shapes pairwise margins (triplet) + a per-sample logit (BCE); it *never* optimizes the top-20 vote that inference actually reads. That mismatch is the one genuinely un-enumerated axis in the objective space. Everything below is judged against it.

**The two ceilings every candidate must clear (both banked, both brutal):**
1. **Law-I (8+ instances, F37/F42/F50/F63/F65/F66…):** oracle headroom exists on ZH/EN but no *symmetric* operator converts it.
2. **F66 arithmetic wall (a6e41f8):** the ISR β-decomposition proved **91–98% of ZH/EN oracle headroom is reachable ONLY by per-item selection (law-III banned)**; the legal symmetric component is +0.001–0.006. **A better training loss is a symmetric operator.** So *any* objective swap on ZH/EN is arithmetically capped near zero on the headroom-conversion story. HateMM already PASSES (F53) and needs nothing.

**Consequence that governs every honest prior below:** the only realistic target for a new head objective is **ZH val-sel *stability*** (turn the marginal final-epoch ZH pass into a val-sel pass by producing a flatter/better-generalizing embedding — a *variance-reduction* story, not a headroom-conversion story). That is the same target F62 (SWA), F69 (grad-norm select), F70 (readout) all attacked and missed. Nobody has attacked it from the **loss functional** side. That is the entire opening.

---

## 1. Deep dive — NCA / soft-kNN-aware objectives (the mission's flagged axis)

**Citations:** Goldberger, Roweis, Salakhutdinov, Hinton, *Neighbourhood Components Analysis*, NIPS 2004; Frosst, Papernot, Hinton, *Analyzing and Improving Representations with the Soft Nearest Neighbor Loss*, ICML 2019; *ASK: Adversarial Soft k-NN*, arXiv:2106.14300; *Revisiting Nearest Neighbor for Tabular Data*, ICLR 2025 (NCA revival on TabR); ProxyNCA++ arXiv:2004.01113.

**Mechanism.** NCA defines a stochastic 1-NN classifier: `p_ij ∝ exp(−d(x_i,x_j))` (softmax over negative distances in the *learned* space, diagonal excluded), and maximizes the expected leave-one-out accuracy `Σ_i Σ_{j: y_j=y_i} p_ij`. The loss **is** a differentiable surrogate for exactly the LOO neighbour-vote our inference performs. Soft-NN loss (ICML19) is the same object used as an auxiliary. This is the *only* loss family in the entire sweep whose training objective is isomorphic to the deployed decision rule.

**Why it is genuinely un-enumerated here (isomorphism check, passed):** the campaign has swapped the *mining source* (C3geo, R3, dead), *reweighted the SFT curriculum* (cand-2, tie), moved the retrieval loss *into the encoder* (P9b, dead), averaged/selected checkpoints (F62/F69), and swapped readouts (F70) — but **never replaced the head's training functional** `triplet(m=0.1)+0.5·BCE` with a vote-optimizing objective. Not covered by any `ban_scope`. Not a pseudo-label pool expansion (banned_constraints) — it reshapes the embedding, not the memory contents.

**Why it is nonetheless NOT a +3 lever (be brutal):**
- **F66 caps it.** NCA optimizes a *symmetric* embedding geometry. F66 proved the convertible ZH/EN headroom is 91–98% selection-only. A vote-matched symmetric loss can, at most, recover the +0.001–0.006 legal slice. That is a rounding error, not +3.
- **NCA's documented failure regime is exactly ours.** Every source (statwiki, Zakka's reference impl, Regularized-NCA Springer, Bayesian-NCA arXiv:1604.02354) reports NCA **overfits and collapses on small-N / high-dimension / binary / label-noise** data without strong regularization — the matrix "blows up," points collapse onto a hyperplane, train-error falls while test-error rises. Our head is 1024-d projections, ~600 samples, binary, with **FAISS-mined pseudo-gold positives = label noise**. This is the textbook NCA-overfits cell.
- **HateMM doesn't need it** (already PASS), **EN is label-limited at 5 proven levels** (F44/F55), so the only live target is **ZH val-sel stability**, where NCA's contribution would be *variance reduction*, and NCA is *more* variance-prone, not less, at this scale.

**Verdict on the deep-dive axis:** the loss↔inference mismatch is a *real, un-enumerated, non-isomorphic* axis and NCA/soft-kNN is its cleanest instantiation — **worth exactly one $0/minutes probe as the round's lead**, because it is the only operator that could test whether law-I is "wrong loss shaping the wrong thing" vs. a hard information ceiling. But the honest prior that it clears **+3** is **~2–4%** (F66 wall), and the prior it delivers a defensible **ZH val-sel hardening (≥+1 stable)** is **~12–18%** (NCA's own small-N instability fights you). The value is **diagnostic-per-dollar**, not expected +3. If it also fails, it converts law-I from "operators mismatch" to "even the vote-matched objective can't," which is a strong paper sentence. **Recommend: run the soft-kNN/NCA head loss as the lead $0 probe, pre-registered against ZH val-sel with F66's legal-slice as the oracle ceiling.**

---

## 2. The other five spaces (compressed, each with transplant + ban check + prior)

**(2) Supervised contrastive (SupCon) & binary-imbalance variant.** Khosla NeurIPS20 (arXiv:2004.11362); *A Tale of Two Classes*, arXiv:2503.17024 (2025). SupCon = softmax over **all** same-class positives per anchor (we already have in-batch positives via `label_matrix`, but as triplet-margin, not log-softmax). The 2025 variance analysis (arXiv:2510.02161) finds **triplet ≥ SupCon on small/mid datasets** (CUB/Cars/CIFAR) — so a naive SupCon swap likely *loses*. BUT the binary-imbalanced paper is our **exact regime** (binary, ~40% positive, small) and reports **up to +35% over standard SupCon** by re-structuring the **local neighbourhood class distribution** — i.e. it is *kNN-neighborhood-aware*, the same property that makes NCA attractive. Caveats: the +35% is *over SupCon*, not over our triplet+BCE; SupCon needs larger batch and **collapses on imbalanced binary as batch grows** (search-confirmed) — our batch 64 is borderline. Ban check: clean (not P9b, not a channel, not selection). Prior ≥+1: low-moderate (~10%); ≥+3: ~2%.

**(3) Manifold / feature mixup at the projection head — round-1 Family D, still UNMEASURED.** Verma et al. ICML19 (arXiv:1806.05236). On *frozen cached features* "manifold mixup" reduces to interpolating `(x_i,x_j)`→`(λ·mix, λ·label)` at the img/text-proj layer inside the head. Documented to **reduce overfitting and flatten representations in low-data classification** — which is *precisely the diagnosed ZH failure mode* (F45: 78-dev selection noise, dev saturates ep19 while test climbs to ep29). Ban check: **not** modality-dropout (that's masking, just measured flat/negative — interpolation ≠ dropout), **not** averaging (F62), **not** selection (F47/F69). Genuinely un-enumerated regularizer that attacks *variance*, not *headroom*, so **F66 does not gate it** (it is not trying to convert oracle headroom; it is trying to stabilize the argmax). Friction: the kNN vote reads *real* banked neighbours, so mixup only regularizes the head/BCE path and the embedding it produces — the memory bank stays real. Cost: ~10 LOC, minutes. Prior ≥+1 (ZH val-sel): low-moderate (~12–15%, the cleanest variance-reduction bet in the sweep); ≥+3: ~1%.

**(4) Proxy-anchor / ProxyNCA++.** Kim CVPR20; Teh ECCV20 (arXiv:2004.01113). Replace noisy mined pairs with **learned class proxies**: robust to noisy labels/outliers, faster convergence, works at **lower embedding dim**. Directly targets a real weakness — our pseudo-gold positives are **FAISS-mined = noisy**, and proxy losses are the literature's answer to exactly that. ProxyNCA++ is *also* NCA-family (kNN-aware) so it doubles as a more-stable NCA. Caveat: with **2 classes** the proxy benefit (largest with many classes / open-set) is thin, and F66 still caps headroom. Ban check: clean. Prior ≥+1: low (~8%); ≥+3: ~2%. **Best framed as the stability-hardened variant to fold into the §1 probe's grid, not a separate ceremony.**

**(5) ArcFace / CosFace angular margin.** Deng CVPR19 (arXiv:1801.07698); Wang CVPR18. Our vote is **cosine-weighted** — angular-margin losses sharpen exactly the angular class separation a cos-kNN reads, a *better geometric match to the vote* than the current pairwise cosine-margin triplet. Enforces a **global** angular margin vs. class centers rather than a per-pair margin on mined hardest. Caveats: designed for **many-class open-set** face ID; the margin gain is documented to shrink toward binary; still a symmetric operator (F66 caps ZH/EN). Ban check: clean, non-isomorphic to triplet (proxy-vs-pair, global-angular-vs-hardest-pair). Prior ≥+1: low (~7%); ≥+3: ~1.5%.

**(6) Two-stage (contrastive-then-classify) + longer/cosine-restart schedule.** Gunel arXiv:2011.01403. Our loss is already a *joint* triplet+BCE. Decoupling to stage-1 metric → stage-2 frozen-embedding classifier **collapses to "change the contrastive loss"** for our purposes, because inference reads the **kNN vote (embedding)**, not the stage-2 classifier — so stage-2 is inert to the deployed decision. Longer training + cosine restarts is cheap and F45 shows dev *undershoots* test on ZH, but that is a **selection-protocol** issue already worked by F62/F69/F70. Low marginal novelty, **not** promoted to top-5.

---

## 3. Precise ruling requested: training-time EMA of head weights vs. F62 (SWA)

**They are distinct objects — F62 does NOT literally ban EMA.** The literature (arXiv:2411.18704, 2024; Mosaic/Composer EMA card) draws the line explicitly: **SWA** = equal-weight average of **saved per-epoch checkpoints**, computed **after** training, not in the loop; **EMA** = in-loop exponential-decay running average of weights, used as the eval model, emphasizing **recent** states. F62's `ban_scope` is verbatim *"SWA weight-averaging of per-epoch head checkpoints"* — that is the post-hoc equal-weight object. Training-time EMA is a different estimator (recency-weighted, in-loop). **So EMA is not covered by the letter of F62.**

**BUT the F62 kill *mechanism* transfers, so its prior is low and it inherits F62's governance flag:**
- On **HateMM** F62 killed because the **dev curve has a mid-training peak + lower late plateau** → any trajectory average (SWA *or* EMA) is pulled toward the late plateau and cannot recover the peak. EMA with any non-trivial decay still averages toward late weights → **same failure, retro-predicted.**
- On **ZH** F62b killed as **dev-underpowered** (78-item dev, 3–4 item spread = the size of the effect) → EMA faces the identical inability to license the test-touch.
- **Governance:** single-trajectory weight-averaging (SWA *and* EMA) needs the user's **micro-ruling vs. the cross-seed-ensemble veto** before any claims-table entry.

**Net ruling:** EMA-of-head-weights is *technically outside* F62's ban but *mechanistically pre-priced near-zero* by F62/F62b's dev-shape evidence, and blocked by the same governance gate. **Do not spend a slot on it** above the top-5; if ever run, it must be pre-declared against the same dev-shape and carry the F62 micro-ruling. (This is the honest "distinct object, dead-anyway" call the mission asked for.)

---

## 4. TOP-5 (prior × cost). All are head-only = minutes of GPU; ranking is on prior × non-deadness.

| # | Candidate | Cite | P(≥+1 stable, mostly ZH val-sel) | P(≥+3, any dataset) | Cost | One-line why-it-isn't-already-dead |
|---|-----------|------|-------------------------------|-------------------|------|-------------------------------------|
| 1 | **Soft-kNN / NCA head loss** (replace triplet+BCE with LOO-vote surrogate) | NCA NIPS04; SoftNN ICML19; TabR-NCA ICLR25 | ~12–18% | ~2–4% | Only objective in the whole campaign that optimizes the **deployed kNN vote itself**; every prior swap changed mining/curriculum/where-the-loss-goes, never the head functional → not in any ban_scope. |
| 2 | **Manifold/feature mixup at proj head** (Family D, unmeasured) | Verma ICML19 | ~12–15% | ~1% | A **variance-reduction** regularizer, not a headroom-conversion operator → **F66 wall doesn't apply**; attacks the *actual* ZH failure (78-dev selection noise, F45); ≠ modality-dropout (masking, just-measured-flat) and ≠ averaging/selection (F62/F69). |
| 3 | **Binary-imbalance neighborhood-SupCon** | Khosla NeurIPS20; arXiv:2503.17024 (2025) | ~10% | ~2% | Co-designs the loss around the **local-neighbourhood class distribution the kNN vote reads**, for **binary + mild-imbalance + small-N = our exact cell**; never tried (we use triplet-in-batch, not log-softmax-all-positives). |
| 4 | **Proxy-anchor / ProxyNCA++** | Kim CVPR20; Teh ECCV20 (2004.01113) | ~8% | ~2% | Directly targets **noisy FAISS-mined pseudo-gold pairs** (proxies are the lit's noise-robust answer); doubles as a *stability-hardened* NCA — **fold into #1's grid**, don't run standalone. |
| 5 | **ArcFace/CosFace angular margin** | Deng CVPR19 (1801.07698); Wang CVPR18 | ~7% | ~1.5% | Sharpens **angular** class separation that the **cosine**-weighted vote reads → better geometric match to inference than the current pairwise-cosine triplet; global-angular ≠ hardest-pair-margin. |

**Bottom line for the orchestrator.** The sweep found exactly **one** structurally new axis — the **loss↔inference mismatch** — and NCA/soft-kNN is its head. It is worth **one lead $0/minutes probe (candidates #1 with #3/#4 as arms in the same grid, pre-registered against ZH val-sel with F66's +0.001–0.006 legal slice as the honest oracle ceiling)**. But be clear-eyed: **F66 arithmetically caps the +3 story** for every symmetric objective here, so the realistic deliverable is **ZH val-sel *hardening*, not a new +3 dataset** — and even that is a variance bet against NCA's own small-N instability. Manifold mixup (#2) is the cleanest *independent* second arm because it is the one candidate the F66 wall does **not** gate (variance, not headroom). Everything else (two-stage, longer schedule, focal/logit-adjust — the latter is decision-side threshold territory already killed by B5/F34 for the vote, and inert on the vote if applied to the BCE head, EMA) is dominated or pre-priced.
