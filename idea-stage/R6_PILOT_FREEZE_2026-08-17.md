# Round-6 pilot freeze — 2026-08-17

Frozen **before** any arm cache is built, any head is trained and any metric is computed.
Two pilots. Both are zero-API-cost and run entirely on assets already on disk.
API spend this round so far: **¥0.00**. Neither pilot may spend any.

Protocol is inherited unchanged from the 2026-08-13/14 pilot series so the numbers are
comparable: `src/run_rac.py`, `--batch_size 64 --lr 0.0001 --epochs 30 --topk 20 --proj_dim 1024
--map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align --hard_negatives_loss True
--no_hard_negatives 1 --final_eval False --metric cos --loss triplet --batch_norm False
--hybrid_loss True --warmup 5 --majority_voting arithmetic --no_pseudo_gold_positives 1
--lambda_seg 0 --contrast_mode none --Faiss_GPU False`, seeds 0/1/2, epoch selected on `val` by
validation macro-F1, **test macro-F1 reported**.

---

## Pilot R6-1 — Multi-layer readout fusion at head level

### Why this is not already closed

`refine-logs/READOUT_SUBMIT_RECORD.md` closed the readout axis as `KS-readout-dead`. Three facts
make that closure incomplete rather than wrong:

1. It was a **retrieval-arena kNN screen on the dev split** (n_dev = 78 ZH / 107 HateMM),
   scored in fix/break item counts. **No head was ever trained on the L24 features** — the record
   says so explicitly: *"NO verdict GPU, ZERO test-touch, NO head job."*
2. Its own permutation null had p95 = **+0.0769 (ZH) / +0.0939 (HateMM)**. A screen whose null band
   is 8-9 accuracy points wide cannot detect a 1-2 point effect. The observed winners (+0.0128,
   +0.0093) were declared "inside the null", which is true and also uninformative at that power.
3. **F111** later ruled the raw-key arena **UNVALIDATED as a predictor of deployed effects**
   (pooled Spearman −0.3039, and explicitly *not* certified as "valid for kills"). **F113** made the
   standing rule that a raw arena may KILL but may not promote; F111 undercuts the KILL direction
   too.
4. Every cell was tested **alone** (`ro_L24` vs `ro_L28`). The **combination** of two layers — the
   actual mechanism in the 2025-26 multi-layer-probing literature (`2605.10494` ICASSP 2026,
   `2601.09322`, HiProbe-VAD `2507.17394` ACM MM 2025) — has never been evaluated in any arena.

So the frozen claim under test is narrow and honest: *does a fixed global mixture of two frozen
decoder layers beat the final layer alone, once a head is actually trained on it?*

### Position in the operator pincer
**A-side (fixed symmetric).** The layer weights are global and shared across every item; there is no
per-item layer choice. A-side has a poor track record in this project, which is priced into the
expectation below, not hidden.

### Assets (all present, verified 2026-08-17)
`data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF-ro_{L24,L28}.pt`
`data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF-ro_{L24,L28}.pt`
24 files, 3584-d, produced by `src/utils/generate_VideoMLLM_embedding_readout_HF.py` in one forward
pass each, so L24 and L28 are **bit-matched on prompt, frame sampling and pooling span** — the only
difference between them is the decoder layer index. That is the correct controlled comparison and it
is why `ro_L28`, not the deployed cache, is the baseline.

### Datasets
HateMM (LoRA-curric encoder) and MHC_zh (LoRA encoder). These are the only two with `ro_` caches.
Two datasets, which is the project's standing minimum for a mechanism claim.

### Arms (4 arms × 3 seeds × 2 datasets = 24 head runs, ~5 minutes total)
| arm | img_feats | text_feats | dim |
|---|---|---|---|
| **A0** | `ro_L28` | `ro_L28` | 3584 |
| **L24** | `ro_L24` | `ro_L24` | 3584 |
| **CAT** | `[ro_L28 ‖ ro_L24]` | `[ro_L28 ‖ ro_L24]` | 7168 |
| **RANDCAT** (control) | `[ro_L28 ‖ R·ro_L28]` | `[ro_L28 ‖ R·ro_L28]` | 7168 |

`R` is a fixed random Gaussian projection 3584→3584, drawn once with `numpy.random.default_rng(20260817)`
and shared across splits, seeds and datasets. Each half of every concatenated vector is L2-normalised
before concatenation, so CAT and RANDCAT are matched on dimensionality, norm and parameter count.
RANDCAT exists because §4.10 of the constraint map records that random projections of the frozen
feature are a strong baseline; without it, any CAT gain is unattributable.

### Frozen decision rule
Primary quantity: mean over seeds of (CAT − A0) in **test macro-F1**, per dataset.

- **GO** — `mean(CAT − A0) ≥ +0.005` **and** 3/3 seeds positive **and**
  `mean(CAT − RANDCAT) ≥ +0.005`, on **both** datasets.
- **AMBIGUOUS** — the GO conditions hold on exactly one dataset.
- **KILL** — anything else.

`L24 − A0` is recorded for the record (it is the quantity the old screen tried to measure with a
trained head for the first time) but **is not part of the decision rule**.

### Expected gain, stated before the run
+0.000 to +0.006, most likely inside seed noise. Reasons it should be small: F92 (stream collapse is
monotone in pooled span and all layers see all vision tokens under this attention pattern), F70 (L24
alone was inside its null even at low power), and Law I (nine certified instances of a richer signal
converting to nothing). The reason to run it anyway is that it costs five minutes and it converts an
argument into a measurement on an axis the project's own records leave formally open.

### Most likely cause of death
L24 and L28 carry nearly the same information because the decoder is causal and every text token
already attends the full vision span, so the concatenation is a dimensionality increase rather than
an information increase — and RANDCAT will match it.

---

## Pilot R6-2 — Transductive pool refinement over the unlabelled test set

### Mechanism
Take the trained head's inductive test probabilities and its fused test embeddings. Fit a
two-component class-conditional Gaussian (shared spherical covariance) over the **test pool's**
embeddings by block-MM EM, initialised from the inductive probabilities, with (i) a KL term
anchoring the refined posteriors to the inductive ones and (ii) a class-balance term. Emit refined
posteriors and threshold them at the same validation-selected threshold. Only test **inputs** are
used; **no test label is read at any point**.

This is the TransCLIP / UNEM / StatA operator (`2406.01837`, `2412.16739`, `2501.03729`) ported to a
trained binary moderation head. It is legal under the user's 2026-08-09 ruling unsealing test inputs.

### Why this is not F63
**F63** killed multi-hop **label propagation / graph diffusion over the frozen kNN graph** on all
three datasets, monotone-negative in the diffusion coefficient (HateMM −0.0187, ZH −0.0385,
α=0.9 catastrophic at −0.19/−0.22). That operator moves label mass **along edges of a neighbour
graph**. This operator estimates **two global class centroids and a shared scale from the pool
density** and never consults a neighbour list. The distinction is mechanical, not nominal. It is
nonetheless the same broad transductive family, and F63 is priced as a headwind below, not ignored.

### Position in the operator pincer
**A-side (fixed symmetric).** One global density model applied identically to every test item. No
per-item selection, so Law III / F47 is not engaged.

### Hyperparameter selection — the leakage guard
λ (KL anchor weight) ∈ {0.25, 0.5, 1.0, 2.0, 4.0} and the class-balance strength
ρ ∈ {0.0, 0.5, 1.0} are selected **per dataset × seed on the validation split treated as its own
pool** (fit the same EM over val embeddings, score against val labels, pick the best macro-F1).
The selected (λ, ρ) is then applied once to the test pool. Test labels are read only at the final
scoring step, after the configuration is fixed.

### Arms (all four datasets, 3 head seeds)
| arm | description |
|---|---|
| **IND** | the inductive head, unchanged — baseline |
| **TRANS** | transductive refinement as above |
| **SHUF** (control) | identical EM, but the test embeddings are randomly permuted across items relative to their inductive probabilities |

SHUF is the load-bearing control. It preserves the class-balance term and the KL anchor while
destroying the pool geometry. If TRANS ≈ SHUF, then any movement came from the balance prior — which
is threshold tuning under another name and is capped by F34 — and not from transduction.

Encoder per dataset is the contrast-line encoder: HateMM LoRA-Qwen, MHC-EN frozen Qwen,
MHC-ZH LoRA-Qwen, ImpliHateVid CLIP.

### Frozen decision rule
Primary quantity: mean over seeds of (TRANS − IND) in **test macro-F1**, per dataset.

- **GO** — `mean(TRANS − IND) ≥ +0.005` **and** 3/3 seeds positive **and**
  `mean(TRANS − SHUF) ≥ +0.005`, on **at least two** of the four datasets.
- **AMBIGUOUS** — those conditions hold on exactly one dataset.
- **KILL** — anything else.

### Expected gain, stated before the run
+0.000 to +0.010 on the two small datasets, ~0 on ImpliHateVid. Headwinds already on record and
priced: F63 (the neighbouring transductive operator is negative here); the measured **absence of any
train/test covariate shift** on all four datasets (domain-classifier AUC 0.42-0.56, MMD p 0.17-0.96),
which removes the usual reason transduction helps; TransCLIP's own limitations section showing gains
decaying to +0.8/+0.9/−1.0 at 16 shots while our head sees 549-1283 labels; `2204.11181` showing
transductive methods fall **below** inductive under class imbalance, and our pools are 25-40 %
positive; and framing killer 3, under which a positive result would still be hard to claim as novel
because retrieval ≡ TTA is formally absorbed.

### Most likely cause of death
With hundreds of in-domain training labels and no covariate shift, the inductive decision boundary
is already better than anything two pool-estimated centroids can express, so the KL anchor either
dominates (Δ ≈ 0) or the density term drags accuracy down.

---

## What this round will NOT run, and why

- **Taxonomy-preserving auxiliary head** (fine MHC/ImpliHateVid subclass labels as auxiliary
  targets). Blocked, not free: **F82** places "head-side graded auxiliary" under an
  admissibility gate that is *"only revivable by user ruling WITH a new mechanism argument"*, and
  prices the headwind at EN +0.0250 / ZH +0.0256. Recorded as requiring a user ruling.
- **Joint multi-dataset training.** Blocked by the standing registry constraint *"training data =
  single-dataset train split only / no cross-dataset mixing"*, which has not been lifted. Requires a
  user ruling before it can be run.
- **Metadata / title channel.** Closed by measurement this round, no pilot needed — see §10.
- **Noise-robust objectives (ELR/GCE/SCE/JAL).** Pre-priced by **F79** (boundary-dominated error is a
  13-17 % upper bound, fixable part single-digit) and damaged by **F75** (head-loss swaps 0/8 FORMAL,
  7/8 arm-dead). A proposal must beat F79's arithmetic first; none does.
- **Any paid-API pilot.** Nothing this round needs one. Budget stays at ¥0.00 of ¥60.
