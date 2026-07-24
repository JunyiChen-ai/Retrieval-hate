# NCA / Soft-kNN HEAD-LOSS family — FORENSIC RECON (zero-GPU)

**Agent:** forensic-recon (litsweep2 wave-3) · **Date:** 2026-07-25 NZST · CPU-only, no job submission, no prereg, autoresearch state untouched.
**Source:** `refine-logs/LITSWEEP2_HEAD_OBJECTIVES.md` (lit2-objectives ROUND-2 #1). This recon converts that sweep's #1 axis into a GO/NO-GO + one pre-registerable family.
**Mission axis:** replace/augment the head's `triplet(m=0.1)+0.5·BCE` objective with a loss that DIRECTLY optimizes the deployed top-20 signed-cosine kNN vote (NCA NIPS'04 / Soft-NN ICML'19 / TabR-NCA ICLR'25), plus the two sweep-recommended fold-ins (neighborhood-SupCon; manifold mixup as the independent variance arm).

---

## 0. VERDICT (up front)

**GO — as a diagnostic-per-dollar lead $0 probe, ONE pre-registered family of 4 arms, NOT as an expected-+3 bet.**

- **F66 ruling (centerpiece, ≤3 sentences):** F66's β-decomposition is *conditional on a fixed embedding map φ₀* — it measures, on the deployed/frozen key space, how much a **symmetric inference-side operator** (vote re-weighting / re-aggregation over a *fixed* query–key Gram matrix) can convert φ₀'s oracle headroom, and finds only +0.001–0.006 is legal. An NCA/soft-kNN **training** loss does not act on φ₀'s Gram matrix; it produces a **different** map φ′, hence a different Gram matrix, a different oracle, and a different decomposition — objects F66 never measured — so **F66 does NOT arithmetically bind trained-space reshaping and the cell is not F66-dead.** The honest counter-pressure is law-I's 8-instance empirical pattern ("better representation ⇒ zero vote conversion"), which warns the ceiling may live in the *frozen features*, not the objective; NCA is the one operator that discriminates "wrong objective" from "feature-information ceiling," which is exactly its diagnostic value and exactly why the honest +3 prior stays at ~2–4%.
- **Pinned family (≤4 arms × 2 datasets × 3 seeds = 24 runs):** A1a NCA τ=0.1 · A1b NCA τ=0.2 (pre-declared 2-value tau grid, NCA is tau-sensitive) · A2 neighborhood-SupCon (binary-imbalance, τ=0.1) · A3 manifold mixup (α=2.0, independent variance arm, F66-immune). ProxyNCA++ dropped (dominated by the tau grid; litsweep already ruled "fold-or-drop").
- **Total GPU-h:** ~0.33 (24 head-only runs × ≤50 s on cached features), ONE sbatch. Well under the 0.5 ceiling.
- **Honest prior (per litsweep, restated per arm below):** P(≥+1 stable ZH val-sel) ≈ 12–18% (NCA), 12–15% (mixup), 10% (SupCon); P(≥+3 any dataset) ≈ 1–4%. Realistic deliverable = **ZH val-sel hardening**, not a new +3 dataset. If it fails, law-I upgrades from "operators mismatch the vote" to "even the vote-matched objective can't" — a strong paper sentence.

---

## 1. Grounded picture of the deployed object (code-confirmed, not prose)

### 1.1 The head and the loss (what we would replace)
- `src/model/classifier.py::classifier_hateClipper.forward`: img/text → `Linear→Dropout` proj, **L2-normalize each stream**, fuse by **Hadamard** (`fusion_mode='align'`, `x = img*text`), MLP; `embed = mlp[:-2](x)` (the retrieval embedding), `output = output_layer(mlp(x))` (the BCE logit).
- `src/model/loss.py::compute_loss`: with the deployed config the contrastive term is **triplet-margin** (`args.loss='triplet'`, `args.metric='cos'`, `args.triplet_margin=0.1`): `mean(relu(in_batch_loss + hard_loss − pseudo_gold_loss + margin))`, where the positive is a **FAISS-mined pseudo-gold** (`no_pseudo_gold_positives=1`, in-batch positives OFF) and the negative is a **FAISS-mined hard negative** (`no_hard_negatives=1`). Plus **hybrid BCE** at `ce_weight=0.5`: `total = 0.5·triplet + 0.5·BCE` (`loss.py:545-554`).
- Per-epoch bank: `run_rac.py:631-639` sets `train_feats=None` at the start of each epoch → the mining path (`retrieval.py:341-385`) rebuilds the **full train bank** by a `model.eval()` forward over `train_dl`, reused within the epoch (`reindex_every_step` default **False**). CPU-FAISS path (`--Faiss_GPU False`, deployed) builds the bank as a **detached numpy** array (`retrieval.py:363,377`).

### 1.2 The deployed decision rule (what the loss must match)
`run_rac.py:746-780` → `evaluate_rac.retrieve_evaluate_RAC_` (top-`args.topk=20` cosine retrieval over the train bank, `IndexFlatIP` on L2-normed feats, `similarity_threshold=-1` ⇒ keep all 20) → `metrics.compute_metrics_retrieval(..., majority_voting="arithmetic", topk=20, use_sim=True)`.

With `use_sim=True, majority_voting="arithmetic"` (`metrics.py:262-284`) the per-query score is
```
s(q) = Σ_{j∈top20(q)} w_rank(j) · (2·y_j − 1) · cos(φ(q), φ(k_j))  /  Σ w_rank(j)
```
`w_rank = [20,19,…,1]` (nearest gets 20×), `(2y_j−1)∈{−1,+1}`, decision `= [sigmoid(s)≥0.5] = [s≥0]`, ROC on raw `s`.
**⇒ the deployed decision is a rank-decay AND cosine-magnitude weighted SIGNED top-20 kNN vote over the train bank in the learned embedding space.** No objective in the 74-finding campaign ever optimized this object; the loss shapes pairwise margins + a per-sample logit, never the vote. **This is the one genuinely un-enumerated axis, and it is code-confirmed.**

### 1.3 Deployed recipe knobs (pinned from `scripts/slurm/enc3seed_lora_curric.sbatch`, byte-identical across enc3seed floors)
`batch 64 · lr 1e-4 · epochs 30 · topk 20 · proj_dim 1024 · map_dim 1024 · dropout 0.2 0.4 0.1 · fusion align · loss triplet · metric cos · hard_negatives_loss True · no_hard_negatives 1 · no_pseudo_gold_positives 1 · hybrid_loss True (ce_weight 0.5) · majority_voting arithmetic · warmup 5 · Faiss_GPU False · reindex per-epoch.` Model selection = Val_Retrieval acc (tie ROC), warmup≥5.

### 1.4 Banked floors (the bars)
| Floor | job | val-sel (acc/mF1) | final (acc/mF1) | source |
|---|---|---|---|---|
| **ZH generic-LoRA** | 13150 | 0.8322 / 0.8015 | 0.8456 / 0.8173 | `B3_VERDICT_REVIEW.md`, re-cited in `HEADRECIPE_*` (mean-over-3-seed) |
| **HateMM curric** | 13241 | 0.8775 / 0.8711 | 0.8791 / 0.8726 | `CAND2_*`, `HEADRECIPE_*` (0.8775 = project best) |

Feature caches: ZH `logging/lora/MHC_zh` (`Qwen2.5-VL-7B-Instruct-LoRA_HF`), HateMM `logging/lora/HateMM_curric` (`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`). Both exist, head-only. **The prereg stage must re-derive the per-seed 4dp floors** (the table above is the mean; the 3/3 test needs per-seed) from the banked trainlogs/review docs.

---

## 2. LOSS DESIGN (one primary + 2 fold-ins; knobs frozen)

### 2.0 The surrogate-fidelity argument (soft-vs-hard top-20, rank weights, signed)
Deployed correct-classification for query `q` of label `y_q` ⟺ same-label neighbors dominate the rank/cos-weighted signed vote, i.e. `s(q)` has the sign of `(2y_q−1)`. NCA maximizes the softmax-weighted probability that a *drawn neighbor shares the anchor's label*:
```
p_{ib} = exp(cos(φ(x_i), k_b)/τ) / Σ_{c≠i} exp(cos(φ(x_i), k_c)/τ)          (softmax over the bank, LOO j≠i)
P_i    = Σ_{b : y_b = y_i} p_{ib}                                            (same-class mass)
L_NCA  = − (1/B) Σ_i log P_i                                                 (stabilized NCA / soft-NN cross-entropy)
```
Fidelity, four gaps, each a *standard soft-surrogate relaxation*, none a defect:
1. **Decision object.** Driving `P_i → 1` forces same-class neighbors to dominate the anchor's neighborhood in cosine similarity — exactly the condition that lands the signed vote on the correct side. `∇(−log P_i)` pulls `φ(x_i)` toward same-class bank points and pushes it from opposite-class points, weighted by softmax responsibility. This is the smooth relaxation of the 0/1 vote-correctness indicator.
2. **Soft-vs-hard top-20.** The softmax over the full bank (no hard truncation → differentiable, keeps gradient to all points) concentrates responsibility on the nearest neighbors; as `τ→0` it → hard 1-NN. The vote's decision margin is dominated by the highest-similarity neighbors, which softmax up-weights; the rank-15–20 tail carries `≤5/210` arithmetic weight AND negligible `exp(cos/τ)` mass — so **soft-full-bank ≈ hard-top20 at the margin.** Hard top-20 truncation is rejected (non-differentiable, zeros gradient to non-retrieved points).
3. **Rank weights.** The deployed vote's arithmetic `[20..1]` decay says "trust nearer neighbors more"; the softmax `exp(cos/τ)` is itself a (steeper, similarity-driven) rank-decay with the same intent. We do **not** replicate the exact `[20..1]` profile — the softmax rank-weighting is the canonical NCA form and a defensible (arguably sharper) surrogate. Declared as a deliberate surrogate choice.
4. **Signed ±1 vs same-class mass.** For binary labels the objects coincide: same-class mass `> 0.5` ⟺ signed vote `> 0` (monotone under similarity weighting). NCA's argmax-improving direction aligns with signed-vote-correctness.

### 2.1 PRIMARY — A1a/A1b: soft-kNN / NCA head loss (2-value tau grid)
- **Object:** `L_NCA` above, replacing the triplet term. `total = (1−ce)·L_NCA + ce·BCE`, `ce=0.5` (deployed).
- **Bank (pinned):** the **current-epoch train-bank embeddings** rebuilt at start-of-epoch by a `model.eval()` forward over `train_dl` (mirrors the FAISS reindex cadence and the vote's per-epoch key semantics), materialized as a **detached torch tensor** `[N,1024]` (+ labels, + ids). Memory trivial (ZH ~744×1024 fp32 ≈ 3 MB; HateMM similar).
- **Gradient flow (pinned):** **anchor grad-on, full bank detached (stop-grad).** = the memory-bank / ProxyNCA-style soft-NN variant that the small-N-NCA literature (statwiki, Regularized-NCA Springer, Bayesian-NCA 1604.02354) recommends over full-grad NCA to avoid the documented matrix-blow-up/collapse at small-N·high-dim·binary·label-noise (our exact cell). Every train point is an anchor across an epoch, so all points get shaped despite bank-detach.
  - *Alternative considered and rejected:* in-batch-grad + external-bank-detach ("NCA-standard grad through both within batch"). Rejected for a $0 pinned probe: the 64/744 in-batch neighbor subsample makes the softmax noisier and doubles anchor↔neighbor coupling — worse for NCA's known small-N variance. Bank-detach is the conservative pin.
- **LOO self-exclusion (pinned, load-bearing):** at TRAINING time the anchor *is* in the train bank; its own bank row must be masked out of its softmax (classic NCA `j≠i`), matched by batch-id vs bank-id. (Deployment never self-matches because dev/test queries are disjoint from the train bank; the surrogate replicates this via LOO.)
- **BCE ruling:** **KEEP** at 0.5. It isolates the clean triplet→NCA swap (dropping BCE would confound "NCA vs triplet" with "with/without BCE") and is cheap insurance against NCA small-N collapse (the logit anchors the embedding). NCA-only is a parked follow-up, not spent here.
- **τ grid (pinned, NO tuning):** `τ ∈ {0.1, 0.2}` — off-the-shelf cosine-softmax temperatures (SimCLR τ=0.1 canonical; 0.2 the milder/more-stable value preferred at small-N). NCA/soft-NN is documented tau-sensitive (Frosst ICML'19 learns T); rather than tune, we **pre-declare a 2-point grid** that brackets the peaked↔soft tradeoff. Both values are reported; the per-arm pass rule applies to each — we do NOT pick the winner and re-run (that would be selection-laundering).

### 2.2 FOLD-IN 1 — A2: neighborhood-SupCon (binary-imbalance variant)
- **Cite:** Khosla NeurIPS'20 (2004.11362); *A Tale of Two Classes* (2503.17024, 2025 — binary+mild-imbalance+small-N = our exact cell).
- **Object:** replace triplet with SupCon (log-softmax over **all** same-class in-batch positives), binary-imbalance neighborhood weighting; `total = 0.5·L_SupCon + 0.5·BCE`. Distinct from A1: in-batch same-class positives + log-softmax, vs A1's full-bank LOO same-class mass. **τ = 0.1** (Khosla canonical, single value, no grid). **BCE KEEP 0.5.**
- Honest caveat (litsweep): triplet ≥ SupCon on small/mid datasets (2510.02161); SupCon needs larger batch and can collapse on imbalanced binary as batch grows — batch 64 is borderline. Prior ≥+1 ~10%, ≥+3 ~2%.

### 2.3 FOLD-IN 2 — A3: manifold / feature mixup at the projection head (independent variance arm)
- **Cite:** Verma ICML'19 (1806.05236). On cached features, manifold mixup = interpolate `(x_i,x_j)→(λ·mix,λ·label)` at the projection layer, feed the mixed sample through the rest of the head to the BCE logit.
- **Object (pinned):** `total = triplet + BCE_mixup`. Mixup interpolates the **fused post-projection representation** and the label; the **BCE term is computed on the mixed sample**; the **triplet term is UNCHANGED on the un-mixed feats** (the kNN vote reads REAL banked neighbors, so mixup can only regularize the classifier/embedding path — the memory bank stays real). **α = 2.0** (Verma's low-data generalization default, `Beta(2,2)`; single value, no grid). **BCE KEEP** (it is mixup's target term).
- **Why it is the cleanest *independent* second bet:** it is a **variance-reduction** regularizer, not a headroom-conversion operator, so **F66 does not gate it at all** (it does not try to convert φ₀'s oracle headroom; it stabilizes the argmax). It attacks the *actual* diagnosed ZH failure (F45: 78-dev selection noise, dev saturates ep19 while test climbs to ep29). ≠ modality-dropout (masking, F73 just-killed) — interpolation ≠ dropout; ≠ averaging (F62); ≠ selection (F47/F69/F70). Prior ≥+1 ~12–15%, ≥+3 ~1%.

### 2.4 Family table
| Arm | Object | τ / α | BCE | F66 gates it? | P(≥+1 ZH val-sel) | P(≥+3 any) |
|---|---|---|---|---|---|---|
| A1a | NCA LOO-vote surrogate | τ=0.1 | keep 0.5 | No (φ-reshape, §3) | 12–18% | 2–4% |
| A1b | NCA LOO-vote surrogate | τ=0.2 | keep 0.5 | No (§3) | 12–18% | 2–4% |
| A2 | neighborhood-SupCon | τ=0.1 | keep 0.5 | No (φ-reshape) | ~10% | ~2% |
| A3 | manifold mixup | α=2.0 | keep (target) | **No — variance arm, not headroom** | 12–15% | ~1% |

---

## 3. BAN CHECK (scopes quoted; F66 is the centerpiece)

- **Not F73** (`Head-recipe family (SAM + mod-dropout)`, F73 body: "SAM x … mod x …" — SAM is an **optimizer** perturbation, mod-dropout is **input masking**). NCA/SupCon change the **training objective (loss functional)**; mixup is an **input-interpolation regularizer on the BCE path**. None is an optimizer or a masking operator. **Distinct — clean.**
- **Not P9b** ("no encoder in the loop"): every arm is **head-only on cached deployed features** (ZH LoRA_HF, HateMM curric_HF). No encoder gradients. **Distinct.**
- **Not cand-2** (data reweighting / curriculum): the arms change the **loss functional / a feature-mixup regularizer**, not the data, its weights, or its curriculum. **Distinct.**
- **Not F50** (`FA fusion/composition gate`: modality-fusion / cross-encoder composition, an **inference-time** channel operation): NCA/SupCon/mixup are **training-time objectives**; the align-Hadamard fusion and the eval path are untouched. **Distinct.**
- **Not F62/F69/F70** (SWA weight-averaging / grad-norm selection / readout) — none is a loss functional.

### 3.1 F66 binding ruling (the recon lives or dies here)
**F66 body (a6e41f8):** on **banked CLIP subclip caches** (a *fixed* key space) the decomposition is `HateMM oracle +0.0776 = symmetric +0.0012 (legal) + selection +0.0764 (banned)`, `EN +0.0700 = +0.0064 + +0.0636`; "the convertible headroom and the legal operator are formally disjoint." Restated in litsweep as: "**A better training loss is a symmetric operator.** So *any* objective swap on ZH/EN is arithmetically capped near zero on the headroom-conversion story."

**Why that restatement over-reaches, precisely:**
1. **F66's arithmetic is conditional on a single fixed map φ₀.** The decomposition holds the embedding geometry FIXED and asks: *given these keys and this query–key Gram matrix, how much can a symmetric (per-query-identical, non-selective) re-weighting of the vote improve it?* Answer ≈ 0; the headroom is per-item-selection-locked. Every number in F66 (+0.0776, +0.0012, +0.0764) is a **property of φ₀'s Gram matrix.** It bounds **inference-side** operators acting on a fixed similarity structure — which is exactly what ISR (per-segment re-agg), re-agg, and vote-reweighting are, and why they are F66-dead.
2. **An NCA loss is not an inference-side operator on φ₀.** It changes the map: `φ₀ → φ′`. A different `φ′` yields a **different** Gram matrix, a **different** oracle headroom, and a **different** symmetric/selection split. F66 never measured φ′'s decomposition; its arithmetic simply does not evaluate on an object it did not measure. The set of achievable vote-accuracies over `{φ}` is not bounded by the symmetric slice of any one φ₀'s headroom.
3. **The "symmetric operator" label is true but not binding.** Yes, at *inference* the NCA-trained head applies the same `φ′` to every query (permutation-equivariant, no per-item selection), so the resulting vote is "symmetric" in F66's sense. But the operative question is whether *reshaping φ to make the vote itself the training target* can move symmetric-reachable vote-accuracy where prior symmetric operators could not. Crucially, **φ₀ was trained by triplet+BCE — an objective misaligned with the vote.** F66's "symmetric slice ≈ 0" is a statement about φ₀'s headroom being selection-locked; it is silent on whether a *vote-aligned* φ′ has a *smaller* oracle (vote already near-optimal) or a *larger* symmetric-reachable accuracy. NCA is precisely the operator that produces and tests φ′.

**⇒ Ruling: F66 does NOT bind trained-space reshaping. The cell is not F66-dead — it is legitimately un-measured.**

**Honest counter-pressure (kept in the open, drives the low prior):** law-I is an 8-instance empirical pattern — better image stream (F65), better fusion (F50), better readout (F70), denser frames (F67), per-segment re-encode (F66) all improved the *representation* and gained *zero* on the vote. If the ceiling is a **feature-information ceiling** (in the frozen LoRA-Qwen/CLIP embeddings fed to the head), NCA cannot beat it either, because law-I is agnostic about *why* the ceiling exists. NCA is the single operator that discriminates "wrong objective" (beatable) from "feature ceiling" (not) — which is why the honest **P(≥+3) stays 2–4%** and the value is **diagnostic-per-dollar**, not expected lift. This is the same call the litsweep made, and I concur.

---

## 4. HYPERPARAM FREEZING (no tuning, one bite)
All values are literature off-the-shelf; the ONLY multiplicity is the pre-declared 2-point τ grid (justified by documented NCA tau-sensitivity, counted against the 4-arm budget), reported per-arm with no winner-reselect-and-rerun.
- `τ_NCA ∈ {0.1, 0.2}` (SimCLR/ProxyNCA++ cosine-softmax canonical bracket).
- `τ_SupCon = 0.1` (Khosla canonical, single value).
- `α_mixup = 2.0` (`Beta(2,2)`, Verma low-data default, single value).
- `ce_weight = 0.5` for all arms (deployed default). Bank-detach, LOO self-exclusion, per-epoch bank rebuild — all fixed as pinned in §2.1.

---

## 5. COST
Head-only on existing caches; 4 arms × 2 datasets × 3 seeds = **24 runs**. Deployed enc3seed run ~20–25 s (curric) to ~50 s; NCA adds one detached bank forward/epoch (~+1–2 s/epoch × 30 = negligible on ~12 train batches); mixup ~floor cost. **24 × ≤50 s ≈ 20 min ≈ ~0.33 GPU-h < 0.5. ONE sbatch** (8 CPU / 1 GPU:a100 / 64 G, mirroring `enc3seed_lora_curric.sbatch`).

---

## 6. BARS (formal-house discipline, unchanged)
- **PASS:** house **+0.030 / +0.030** (acc / macro-F1) **3/3 seeds**, **DUAL protocol** (val-sel AND final-epoch), per-seed paired vs the banked floors **13150 (ZH)** and **13241 (HateMM)** — prereg re-derives per-seed 4dp floors.
- **KS-arm-dead:** an arm whose val-sel AND final are **≤ floor on both datasets** is KILLED (fold to dead), matching F70/F73 KS-arm-dead discipline.
- **Family one-bite:** single sbatch, single test-touch per arm, no re-run-the-winner; independent review + freeze-hash ceremony at prereg time.
- **Honest prior restated per arm** (litsweep §4): §2.4 table. Realistic deliverable = ZH val-sel *hardening* (variance-reduction), with F66's **+0.001–0.006 legal slice** as the honest oracle-conversion ceiling on the headroom story — NOT a new +3 dataset.

---

## 7. IMPLEMENTATION RISK (pinned)
1. **Grad through the bank:** bank **detached**, anchor **grad-on** (§2.1). No O(N²) grad; no bank drift within an epoch.
2. **Per-epoch re-mine interaction / triplet machinery going inert:** in A1/A2 the triplet term is dropped, so the FAISS-mined `hard_negative_features` / `pseudo_positive_features` are **no longer consumed by the loss.** Verified nothing else consumes them: `loss.py` routes both ONLY into `hard_loss`/`pseudo_gold_loss` → the triplet/contrastive/naive `total_loss`; the eval path and `run_rac` do not read them. **Pinned implementation:** add the NCA/SupCon branch behind a flag; build the NCA bank **independently** as a detached torch tensor (start-of-epoch `model.eval()` forward, cached like `train_feats`, rebuilt on the `train_feats=None` reset) rather than reusing the CPU-FAISS **numpy** `train_feats` (which is detached-numpy, not a device torch tensor). The mining call may stay (its returned pairs are simply ignored by the NCA loss) or be skipped — either keeps `run_rac`'s `train_feats.detach()` bookkeeping intact. A3 (mixup) keeps triplet + mining exactly, so its machinery is untouched.
3. **RNG / G-repro:** all new branches gated behind new args (`--head_loss {triplet,nca,supcon}`, `--head_loss_tau`, `--mixup_alpha`), default = **byte-identical floor** (mirrors the `lambda_seg` / `lambda_aux` / `cf_negs` / `mod_dropout` additive-flag discipline). Mixup's `Beta`/permutation draws are confined to the mixup arm; the flag-off path draws nothing (exactly the classifier.py:129 ARM-B pattern).
4. **Eval path untouched:** `retrieve_evaluate_RAC_` + `compute_metrics_retrieval` unchanged; the top-20 signed-cos arithmetic vote reads the trained `φ′` exactly as the floor does. Confirmed no edit needed outside `loss.py` (+ small `run_rac.py` bank plumbing + argparse).

---

## 8. EXECUTION SKELETON (for the prereg/submit stage — NOT run here)
- **New args** (`run_rac.py` argparse): `--head_loss {triplet,nca,supcon}` (default `triplet`), `--head_loss_tau FLOAT`, `--mixup_alpha FLOAT` (default `0.0`). Default set ⇒ floor byte-identical.
- **`loss.py::compute_loss`:** when `head_loss=='nca'`, compute `L_NCA` (§2.1) over a detached bank tensor (anchor grad-on, LOO self-mask by id), set the contrastive term to `L_NCA`, keep `0.5·BCE`; `head_loss=='supcon'` analogous with in-batch same-class log-softmax; both bypass the triplet assembly (mined pairs ignored). `mixup_alpha>0` ⇒ interpolate fused feats + label, add `BCE_mixup`, leave triplet on un-mixed feats.
- **`run_rac.py`:** add a detached per-epoch NCA-bank builder alongside the existing reindex reset (rebuild on `train_feats=None`); pass bank+labels+ids into `compute_loss`. No change to the eval/selection block.
- **sbatch:** clone `enc3seed_lora_curric.sbatch`; `CONFIGS` = 4 arms × {MHC_zh LoRA_HF, HateMM curric_HF} × seeds {0,1,2}; each arm flips only `--head_loss/--head_loss_tau/--mixup_alpha` vs the floor command; fresh `GROUP_NAME` (e.g. `RAC_video_nca`), `--force False` (never overwrite banked arms). Inline val-sel+final readout parser reused verbatim.
- **Codex code-review gate** before submit (touches `loss.py` internals): mandatory per project ceremony.

---

## 9. ONE-LINE SUMMARY
The loss↔inference mismatch is real, code-confirmed, and un-enumerated; NCA/soft-kNN is its only vote-isomorphic instantiation; F66's φ₀-conditional decomposition does **not** bind a loss that reshapes φ; **GO** for one $0 4-arm family (NCA τ∈{0.1,0.2} + SupCon + mixup, ~0.33 GPU-h) with an honest ~2–4% +3 prior and ZH val-sel hardening as the realistic target.
