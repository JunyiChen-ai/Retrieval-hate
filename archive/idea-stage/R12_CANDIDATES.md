# R12 candidate slate — 12 generated, 2 piloted

Round 10 of idea discovery, 2026-08-18. Generated from three literature reconnaissance sweeps run
in parallel, then scored by **gpt-5.6-sol at xhigh reasoning**, conversation only (the Codex sandbox
is broken on this machine: `bwrap: loopback: Failed RTM_NEWADDR`), instructed to be hostile and
given the full constraint map, the R11 union result, the effective-rank diagnostic, the three
in-house dev-positive/test-negative artefacts and the test-set-reuse problem.

API cost: **¥0.00**. Round budget ¥15; cumulative ¥0 of ¥60.

---

## 1. The three reconnaissance angles

### 1.1 Angle A — mechanisms that change *which* items break (beyond prediction churn)

The round's brief was set by `R11_UNION_RESULT.md` §3: `CAT` already retains 0.650 (MHC-ZH) /
0.822 (HateMM) of the `CAT ∪ LL` fix pool, and the unbought headroom is entirely in the **breakage**
column. The sweep was told to skip the churn-reduction family the project had already reasoned
about and to look for objectives whose stated effect is on per-example error *composition*.

**The one live family: positive-congruent training / negative-flip reduction.**
PC-Training `2011.09161` (CVPR 2021 oral), ELODI `2205.06265` (TPAMI 2024), MPT `2511.08322`,
`2105.03048` (ACL 2021), `2202.02976` (NeurIPS 2022), MUSCLE `2407.09435`, BCT `2003.11942`.
Mechanism: `L_FD = Σ_i [α + β·1(reference_i correct)]·d(φ, φ*)` — the distillation term is
up-weighted on exactly the items the reference model gets right. The relevant number is **MPT
Table 4**, a frozen ViT-B/32 with only the classification layer trained on CIFAR-100: negative-flip
rate 7.08 → 3.26 and old-class error 19.44 → 15.16 **simultaneously**, which is a counterexample to
the one-for-one retention/breakage exchange R11 measured. R11's `ANCA`/`ANCL` are the β = 0,
single-source degenerate member of this family, i.e. the member the literature says does not work.
BCT is a documented negative (it *increases* negative flips, 14.88 vs 14.04).
**Occupancy in hate / harm / toxicity / meme detection: zero papers.**

**Second live item: stability-scored hard block selection.** `2606.30791` (2026-06) is an existence
proof that hard block-*dropping* beats fusing everything on a frozen backbone: frozen XLS-R-300M,
per-layer probes ranked by **cross-domain** power rather than in-distribution accuracy, fuse only 4
of 25 layers, 28 % relative improvement over fusing all 25. Combined with stability selection
(Meinshausen & Bühlmann `0809.2932`), whose defining property is that the selected set is a
*non-smooth* function of subsample draws — "different items break" is its native failure mode.
Occupancy: `"stability selection"` × `"deep learning"` and × `"embeddings"` both return **zero**
arXiv hits. Framing collision: `2601.13288` (2026-01) already frames classification as selection
over the full **token × layer** hidden-state tensor, on safety benchmarks, with probes in this
project's parameter range — a *learned soft* aggregator, so a hard-selection variant is still open,
but "the token-layer tensor is the right object" is an occupied claim.

**Structurally killed before scoring:** residualise-then-concatenate. `[CAT ‖ resid(LL|CAT)]` and
`[CAT ‖ LL]` span the **same column space**, so for a linear read-out the two are mathematically
identical; only a loss-level variant escapes the equivalence. No published evidence at any `n`
supports feature-level orthogonalisation beating raw concatenation.

**Checked and judged irrelevant:** anti-churn mean distillation (`2102.05140`, `2106.02654` — no
fix/break decomposition), AMC `2305.04135` (isomorphic to the failed `AVG`), `2405.02581`
(retrieval), ROSE `2411.10896` (selects annotations; the train set is fixed), SAM / flatness,
LDAM `1906.07413` and logit adjustment `2007.07314` (class-level; at 2 balanced classes there is
nothing to reshape), PCGrad / CAGrad / MGDA (on the project's anti-repeat list), JTT `2107.09044` /
GroupDRO / EIIL / LfF / difficulty-bin reweighting (banned), DFR `2204.02937` (same banned family),
Dataset Cartography (diagnostic), selective ensembling and abstention (per-item selection banned),
Deep CCA / sparse mCCA / MOFA (validated at n = 53), FactorCL `2306.05268` as an objective (its
own §3 shows it degenerates to ordinary contrastive learning when unique information is zero),
MISA / Self-MM / ConFEDE / MulT / tensor fusion / UniMSE (all require training the backbone),
deep knockoffs (`1811.06687`, `2402.17176`, `2510.01418`, `2602.00218` — tabular/genomics only),
group / sparse-group / exclusive lasso over `CAT ⊕ LL` (a smooth re-weighting strictly inside the
already-killed concat/PCA/blend hull), classical information-theoretic feature selection
(mRMR / CMIM / HSIC-Lasso — marginal-axis objectives whose estimators fail at d ≈ 4000, n ≈ 1000).

### 1.2 Angle B — the image stream's position axis

Two independent sweeps (general read-out literature; hate-domain occupancy) reached the same
verdict.

**The read-out primitive is occupied, not merely crowded.** DINOv2 `2304.07193`'s frozen linear-eval
feature *is* class token ⊕ mean-pooled patch tokens, so the image-side two-position read-out is its
literal recipe rather than a transplant of it. On top of that: `2506.10178` (ICLR 2026) is a
dedicated benchmark of thirteen structured read-outs against global average pooling with a **+7.9**
headline (MAE 67.7 → 75.6) and the stated reason that "valuable information is distributed across
patch tokens"; `2509.24901` (ICLR 2026) opens on "global pooling creates an information
bottleneck"; `2608.00726` (2026-08) opens on "information lost at readout time". In the hate domain
the primitive is occupied twice — **HateSieve** `2408.05794` (NAACL 2025 Findings) reads patch
tokens alongside the pooled vector on frozen CLIP, and **xDORA** `2602.19212` replaces the mean with
learned soft attention pooling over the token sequence.

**The published causal chain for *why* a flat mean over ~1000 positions loses variance already
exists**, which is the other half of any mechanism story: massive activations `2402.17762`
(COLM 2024, ~10⁴× the median on 2-4 fixed dimensions, and setting them to their corpus mean costs
nothing — i.e. they are a constant offset, which is corpus mean-centring, already dead in-house);
visual attention sink `2503.03321`; `2603.00510` (CVPR 2026, only ~60 % of visual tokens are
"alive"); `2410.07149` (ICLR 2025, removing object-specific visual tokens costs > 70 %);
`2603.17228` (causal attention specifically penalises early image-token positions). And the
training-free fix is published at spotlight level: test-time registers `2506.08010` (NeurIPS 2025
Spotlight), whose own **linear-probe** numbers are Δ 0.0 / +0.1 — all its gain is dense prediction.
Counter-evidence in the same direction: `2310.17715` (EMNLP 2023) finds ablating outlier dimensions
**hurts** downstream classification.

**What is genuinely unclaimed** is narrow and all analysis: nobody has reported what the ~1000 MLLM
visual token positions contain *for hateful content*; nobody has ablated frame-grouped pooling
against a flat token mean for a frozen encoder plus a light supervised head; and **no video paper
anywhere reports a rank / variance / anisotropy diagnostic of mean pooling**. All three are
analysis results, which the method-paper-only constraint bans.

Counter-evidence carried into the pilot design: CLIP-Hitchhiker `2205.08508` — "there has been
limited success in learning temporal aggregation that outperforms mean-pooling"; `2406.01604` beats
mean pooling by only +0.5 to +2.7 R@1.

### 1.3 Angle C — label-free structural / spectral / frequency-domain transforms

This sweep produced one structural argument that retrodicts most of the project's dead list and is
worth recording independently of any candidate.

**If the head's first operation is a dense linear layer, replacing `x` by `Ax + c` for invertible
`A` is an exact reparameterisation** (`W → WA⁻¹`, `b → b − WA⁻¹c`): the function class is identical,
and any measured difference can only come from optimiser implicit bias, basis-dependent weight
decay, or early stopping — all conditioning effects at seed-noise scale. Non-invertible linear maps
(PCA-k, low-rank projection, non-square random projection) can only lose information and can only
help as regularisers, which produces exactly a dev-positive / test-negative signature. Published
form of the theoretical half: `2605.17180` (ICML 2026), "linear heads perform implicit subspace
whitening".

Checked against this project's ledger, six of eight dead feature-transform entries are predicted:
corpus mean-centring (affine, inert) dead; PCA-512 (lossy regulariser) dev-positive/test-negative;
low-rank projection of concatenated blocks actively harmful; random projections as extra width
negative. The two that work are the two that are **not** in the inert class: row L2 normalisation is
per-sample nonlinear, and `CAT` adds a genuinely new block rather than transforming an old one.

**Family verdicts.** Whitening / isotropisation is not merely unsupported but **refuted for
supervised classification**: `2402.03191` (ACL 2024) proves isotropy is incompatible with cluster
structure "which also negatively impacts linear classification objectives"; `2511.11041` reports
full PCA whitening "hurts every model" across 38 models; every reported whitening gain in the
literature is measured on cosine similarity or retrieval, where the read-out is fixed and untrained
and therefore cannot absorb `A`. This project has already measured it twice independently —
`DRAFT_analysis_chapter.md` §3.13 (Ledoit–Wolf whitening on the retrieval-vote stage, 0 of 5
operators promotable, with the diagnosis that shrinkage ≈ 0 at d > n so near-null eigendirections
are amplified ~1000×) and `PCD_SPEC.md` (whitened clause scores worse than un-whitened).
Massive-activation clipping is dead by the same argument as mean-centring. Fourier / wavelet /
covariance pooling is blocked by a repo fact rather than by the literature: the token sequences do
not exist on disk (pooling happens inside the extractors), and the only cached sub-clip windows are
K = 4 for the MultiHateClip splits, which makes a DCT a fixed 4×4 orthogonal matrix and a covariance
pooling rank-3. Random Fourier features are strictly dominated by a head that already learns its own
nonlinear first layer. Hyperbolic / product-manifold heads are refuted by `2607.05268` (seven
released checkpoints, three matched seeds, "the geometry is decorative"). SAE / dictionary codes
need ~9M-token dictionaries and win only at matched sparsity, a regime this project is never in.
LDA / Fisher / Mahalanobis / NCM / RanPAC / analytic-ridge heads are degenerate or straw-man-baselined
at C = 2 (LDA yields exactly C − 1 = 1 discriminant direction, which is the ridge solution on ±1
targets). TabPFN is a calibration win, with its accuracy advantage confined to d ≤ 32.

**The small-n decay curve, which is the decision-relevant object.** Three independent papers, three
mechanisms, one shape:

| method | 1-shot | 5-shot | 10-shot | 25-shot |
|---|---|---|---|---|
| SimpleShot CL2N vs raw | +6.79 | +1.39 | — | — |
| "Free Lunch" Tukey transform alone | +7.93 | +2.30 | — | — |
| FroFA best variant | +6.1 | +1.6 | +0.9 | **+0.3** |

This project sits at ~275-650 examples per class, an order of magnitude past the right-hand column.
Every headline in this family is inflated 5-30× relative to the operating regime.

**One item flagged and resolved without a pilot.** The sweep flagged the macro-F1 operating point
(a fixed 0.5 threshold is provably not macro-F1-optimal) as a possible 1-3 point bug class. Already
priced: `R8_DECOMP_MEMO.md` §3 caps every decision-rule / calibration mechanism at **+0.25 to +1.2
points** with a train+val-fitted global threshold oracle, and a dev-fitted threshold is **negative
on 3 of 4 datasets**.

---

## 2. The slate and the hostile scoring

Composite 0-10, with the reviewer's stated components: prior of clearing +0.005 with the CI
excluding zero on **both** datasets; whether the candidate is outside the already-measured hull;
method novelty; operational cost/risk.

| rank | candidate | prior | outside hull | novelty | ops | composite |
|---|---|---|---|---|---|---|
| 1 | **B2 IMGSPLIT** — img = `[n(vision mean) ‖ n(instruction mean)]` | ~12 % | 9 | **1** | 8 | **4.8** |
| 2 | **B1 IMG2M** — img = `[n(mean) ‖ n(std)]` over the same positions | ~10 % | 9 | **1** | 8 | **4.4** |
| 3 | A5 STABSEL — stability-scored hard block selection over the 14 banked blocks | ~5 % | 6 | 3 | 6 | 3.4 |
| 4 | C3 DMD-SEP — loss-level shared/unique separation of the A0 and TXT blocks | ~4 % | 7 | 2 | 5 | 3.1 |
| 5 | **A1 FOCAL-ANCHOR** — reference-correctness-gated distillation | ~8 % | 3 | 1 | 9 | **3.0** |
| 6 | A4 SPECDEC — spectral decoupling (L2 on unnormalised logits) | ~4 % | 6 | 0.5 | 9 | 2.9 |
| 7 | B3 IMGSINK — drop the top-k highest-norm positions before averaging | ~3 % | 8 | 1 | 5 | 2.7 |
| 8 | A2 ELODI-ENS — distil a CAT-head ensemble into one CAT head | ~2 % | 2 | 0.5 | 8 | 1.7 |
| 9 | C2 DBAT — agree on train, disagree on unlabelled test inputs | ~1 % | 6 | 0.5 | 5 | 1.5 |
| 10 | A3 RECONCILE — patch the loser in the CAT/LL disagreement region | < 1 % | 2 | 1 | 2 | 1.0 |
| 11 | B4 IMGFRAME — img = `[n(mean) ‖ 4 frame-group means]` | < 1 % | 1 | 0 | 6 | 0.7 |
| 12 | C1 PIDU — partial-information-decomposition uniqueness measurement | 0 % | — | 0 | 3 | 0.1 |

### 2.1 The reviewer's hard kills, in his own terms

- **A2**: distillation cannot be expected to outperform a teacher ensemble that itself does not
  beat `CAT`. It compresses variance reduction; it does not recover the union.
- **A3**: the disagreement region contains very few held-out examples. A learned patch at that
  sample size is a higher-dimensional version of the scalar blend that already landed on opposite
  corners of its grid on the two datasets.
- **C2**: inducing diversity has no route to accuracy without subsequently selecting (banned),
  routing (banned), or profitably averaging (already lost).
- **A5**: stability identifies reproducibility, not usefulness; correlated nuisance blocks can be
  extremely stable.
- **B3**: high norm is not synonymous with irrelevant; it risks deleting exactly the global carrier
  dimensions probes use, and it introduces a free choice of `k`.
- **C1**: analysis, out of scope under the method-paper-only rule, and the estimators are
  variance-dominated at n ≈ 1000.

### 2.2 The reviewer's adjudication of A1

Two objections, both accepted into the freeze:

1. **The ban.** `1(reference correct)` is a reference-correctness bin, and the proposal reweights a
   loss term by that bin; reweighting KL rather than BCE does not make it stop being bin
   reweighting. A1 is ineligible unless the campaign ban is explicitly narrowed. *Ruling recorded
   in `R12_FREEZE.md` §0.1a: the ban is scoped to iteration 6 of the RGCL full-bank campaign and is
   narrowed to it, with the shuffled-correctness-mask control adopted as the price.*
2. **The double-source term is algebra, not a mechanism.** `Σ_s w_s KL(p‖q_s)` equals, up to a
   constant, anchoring to a single weighted geometric-mean pseudo-teacher of total strength
   `Σ_s w_s`; in binary classification that pseudo-teacher carries one scalar logit. *Accepted:
   the pilot tests only the focal filter, and the "double source" is collapsed into a single
   explicitly-constructed pseudo-teacher arm.*

On transfer: **"No — not as a quantitative prior."** CIFAR-100 supplies ~50 000 training examples
and 100 logits whose inter-class structure carries the transferable dark knowledge; here there are
579 examples and one binary logit, where KL is mostly margin matching. MPT shows the mechanism can
work somewhere; it does not materially raise the prior that it breaks this project's one-for-one
wall.

### 2.3 The reviewer's adjudication of family B

**"There is no honest method-contribution framing for B1-B4 as currently defined."** DINOv2 owns
the topology; the ICLR 2026 work owns structured-read-out-versus-global-pooling and the
information-bottleneck motivation; HateSieve and xDORA own token-level and learned pooling inside
hate detection. "We found that Qwen visual-token pooling matters for hateful video" is a domain
analysis result.

It is nevertheless worth running, as a terminal feature-engineering check, for four stated reasons:
the read-out axis is genuinely untouched; B2 is the exact causal analogue of `CAT` (later
instruction states can attend to the visual prefix and may act as image-conditioned summary
states); the new information comes from the **image** stream, so `CAT`-image additivity is
structurally more plausible than the `CAT`-`LL` additivity that failed; and the extraction cost is
negligible. With the caution: **low effective rank is consistent with destructive pooling and with
beneficial denoising**, so the rank diagnostic is motivation, not prediction.

### 2.4 The reviewer's answer on missing families

Asked to name any legal mechanism family the slate is missing with a non-trivial prior of ≥ +0.005
on ≥ 2 datasets: **"There is none"** with a defensible combination of legality, probability and
novelty. The nearest omissions he named are learned token pooling (occupied by xDORA and the
structured-read-out literature) and training-time token/block dropout with fixed inference; block
dropout, SAM/SWA, mixup, shrinkage classifiers, random-subspace ensembles and feature-space
augmentation are "legal engineering baselines but not new methods, and their two-dataset prior after
this campaign is not defensibly above a few percent". Error-residual boosting was named and rejected
as collapsing into the banned difficulty reweighting plus the already-failed held-out gating regime.

### 2.5 The reviewer's answer on closure

Asked directly whether the honest conclusion is that the substrate contains no further reachable
method contribution: **"Yes."** Reasons given: the remaining plausible tweaks are already-occupied
methods, absorbed by previous failures, forbidden by the campaign rules, analysis rather than
method, or statistically unconfirmable on the remaining data. On `CAT` specifically: *"Survival does
not create novelty."* His minimum close-out list is transcribed into `IDEA_REPORT.md` §13.

---

## 3. What was piloted

`idea-stage/R12_FREEZE.md`, committed at `a9cd557` **before** any pilot code existed.

| pilot | candidates | controls | seeds | cost |
|---|---|---|---|---|
| **R12-IMG** | ISPLIT (B2), I2M (B1) | I0 (deployed), IRW (matched width), IRSPLIT (random positional split), + 3 non-selectable diagnostics | 800-829 / 800-814 | 1 extraction pass + 360 head runs |
| **R12-ANCHOR** | AF_PT (focal, pseudo-teacher), AF_A0 (focal, A0 teacher) | CAT (λ=0), AU_PT and AU_A0 (uniform, matched anchor mass), AF_SHUF (shuffled correctness mask), LBL (hard-label anchor) | 900-929 / 900-914 | 315 head runs |

Results: `idea-stage/R12_IMG_RESULT.md`, `idea-stage/R12_ANCHOR_RESULT.md`, and
`IDEA_REPORT.md` §13.
