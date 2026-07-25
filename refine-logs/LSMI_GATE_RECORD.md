# LSMI GATE — sample-level synergy / redundancy / uniqueness on the deployed (img, text) streams

Date: 2026-07-25
Executor = **Claude Fable 5** (`claude-opus-5[1m]`), LSMI gate executor, conda `HateVideo`,
**CPU-only, zero SLURM, zero GPU, zero Modal** (the whole gate fits on CPU after the design below).
Read-only on banked caches. No `state/` mutation, no push, no test-split touch.

**Candidate provenance:** `refine-logs/REPRO_SURVEY_2025.md` §4.2 + §5 #1 (rank-2 clone,
`GeWu-Lab/LSMI_Estimator`, ICML 2025, HEAD `13e4db2e033a3721d5ea7c0e31c540b3445a5532`,
`external/baselines/LSMI/`, NO LICENSE FILE — code is **read and re-implemented/imported for
measurement only**, nothing is redistributed).

**What this gate is.** A **measurement**, not a performance lever. It prices the entire
fusion-architecture family mechanistically: if the two deployed streams carry no *synergistic*
task-relevant information, then every richer fusion block (attention, gating, bilinear, concat vs
Hadamard) is arithmetically capped, and F50 ("fixed compositions = rotation at every w") / F44
("EN image stream collapses") / F76 become a *mechanism* rather than a list of nulls.

**What this gate can NOT do** (scoped up front, F66-style): see §5.

---

## 1. METHOD RECON — what LSMI actually estimates (read from the 4 py files, not the README)

### 1.1 The decomposition

LSMI is a **pointwise (per-sample) partial-information decomposition** of the task-relevant
information of a modality pair `(x1,x2)` about a label `y`, into redundancy `r`, uniqueness
`u1,u2`, synergy `s`, obeying (README + `main_lsmi.py:114-121`):

```
i(x1;y)      = r + u1
i(x2;y)      = r + u2
i(x1,x2;y)   = r + u1 + u2 + s
```

Three of the four are pinned by the three pointwise mutual informations; the **fourth degree of
freedom is fixed by LSMI's redundancy rule** (`main_lsmi.py:114-118`):

```
r_plus  = min( h(x1),            h(x2) )
r_minus = min( h(x1) - i(x1;y),  h(x2) - i(x2;y) )
r       = r_plus - r_minus
```

i.e. redundancy is defined through the **pointwise differential entropies (surprisals)** of the two
modalities, via a min-based information-lattice flow.

### 1.2 Two estimators, both trained on the train split

- **Pointwise MI** (`get_mutual_info`, `main_lsmi.py:84-104`): three MLP discriminators
  (`cls_network`, `utils.py:43` — `Linear(d,64) ReLU Linear(64,64) ReLU Linear(64,2)`) trained
  jointly with cross-entropy: one on `x1`, one on `x2`, one on `concat(x1,x2)`. Then
  `i(x;y) := log(n_classes) + log softmax(out)[y]`.
  **Assumption disclosed:** this is `log p̂(y|x) − log p(y)` **with a uniform class prior**
  `p(y)=1/K`. Our splits are imbalanced (ZH 180/579, HateMM 298/744, EN 168/549).
  *Derived invariance (proved in §1.4): `S`, `U1`, `U2` are EXACTLY invariant to this prior
  convention; only `R` absorbs it.* Both conventions are reported.
- **Pointwise differential entropy** (`MargKernel`, `entropy_estimation.py`): the **KNIFE**
  kernel estimator — a `K=5` Gaussian-mixture density model with learned means, a
  **tanh-bounded** per-dim log-variance (`var = exp(tanh(logvar)) ∈ [0.368, 2.718]`) and a
  **full `(1,K,d,d)` strictly-lower-triangular** mixing parameter `tri`; `h(x) := −log p̂(x)`
  per sample. Trained by minimising mean NLL on the train split (no labels used).
  **Assumptions:** (a) `x` has a density w.r.t. Lebesgue measure on `R^d` (our L2-normalised
  features live on a sphere — measure-zero — and, at raw `d=3584 > n`, inside an ≤(n−1)-dim
  affine subspace: the density does **not** exist there, see §5); (b) the mixture is
  expressive enough at the given `d`; (c) inputs are **O(1)-scaled** — the tanh bound means the
  model cannot rescale any coordinate by more than 2.72×, so it is calibrated for the
  demo regime (`gaussian_data.py`: unit one-hot + 0.5 noise).

### 1.3 Compute path

`estimation_main` → fit 3 discriminators (30 ep, Adam 1e-3, StepLR(15,0.1), bs 32) → fit 2 KNIFE
kernels (30 ep, Adam 1e-3, StepLR(20,0.1), bs 32) → `LSMI_estimation` on **train** and again on
**val** with the same train-fitted models → `RUS_adjustment` (`main_lsmi.py:9-59`) shifts
`(r,u1,u2,s) → (r+a, u1−a, u2−a, s+a)` to make the means non-negative.

**Cost driver = `tri` of shape `(1,5,d,d)`.** Measured on this box (CPU, 32 threads, bs 32/64):
`d=64` and `d=256` are seconds-per-fit; **`d=3584` is ≈64.3 M parameters per kernel and ≈2500 s
per kernel fit** — tractable but statistically hopeless (§5).

### 1.4 Derived properties (proved here, used by the readout)

Let `δ(y)` be any per-sample constant added to all three pointwise MIs (this is exactly what a
change of class-prior convention does: `i_emp = i_unif + δ`, `δ = −log K − log p̂(y)`).

1. **`R → R + δ`, `U1, U2, S` unchanged.** (`r_minus` shifts by `−δ` since `min(a−δ,b−δ)=min(a,b)−δ`;
   `u_k = i_k − r` cancels; `s = i12 − i1 − i2 + r` cancels.) ⇒ **synergy is prior-convention-free.**
2. **`S − R = i(x1,x2;y) − i(x1;y) − i(x2;y)` exactly** — the classical *interaction information*,
   and it uses **only the three discriminators, no KNIFE at all.** ⇒ an estimator-free cross-check.
3. **`RUS_adjustment` preserves `S − R`, `R+U1`, `R+U2`, and the total.** (All four shift-invariants
   verified algebraically from `main_lsmi.py:54-57`.)
4. **`R` (hence `S`) is invariant to a *common* additive shift of `h(x1)` and `h(x2)`** — so a
   *common* rescaling of both streams (e.g. one shared scalar σ) leaves the decomposition alone,
   but a *per-stream* rescaling does **not**. This dictates the projection conventions in §2.2.
5. In the regime `|h(x1) − h(x2)| ≫ max(i1,i2)` (guaranteed once `d` is large, since `h` scales
   like `d` while `i ≤ log(1/p(y)) ≈ 1.2` nats), the min-rule **degenerates**: `r → i_{argmin h}`,
   one uniqueness → 0. This is the raw-dim failure mode predicted before running; it is measured.

### 1.5 Defects found in the released code (disclosed; fix declared BEFORE running)

- **`obtain_entropy_estimator` never calls `optimizer.zero_grad()`** (`main_lsmi.py:174-187`):
  KNIFE gradients accumulate across every step of the whole run. The discriminator loop *does*
  call it (`main_lsmi.py:152`). **Primary arms use the `zero_grad`-fixed loop; the as-shipped loop
  is run as a declared fidelity arm** on the primary cell and on the synthetic gates.
- `get_loader` hardcodes `num_workers=16` for ≤744×3584 in-memory tensors → replaced with
  `num_workers=0` (determinism + speed only; no numerical effect).
- `MargKernel.logC` is `−d/2·log(2π)`; it is **common to both streams only because our two streams
  have equal `d`** — that is why it cancels in `r` here (property 4). Recorded, not relied upon.

---

## 2. PRE-DECLARED DESIGN (frozen BEFORE any number on our data is computed)

### 2.1 Data — banked caches only, train + dev, NO test touch

Exactly the deployed two streams `img_feats`, `text_feats` (3584-d, L2-normed) that
`src/model/classifier.py:115-147` consumes, from the deployed encoder lineage per dataset:

| dataset | encoder lineage | cache stem | n_train (pos) | n_dev (pos) |
|---|---|---|---|---|
| **MHC-ZH** | generic-LoRA, floor job **13150** (B3) | `data/CLIP_Embedding/MHC_zh/{train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | 579 (180) | 78 (28) |
| **HateMM** | curric-LoRA, floor job **13241** (project best) | `data/CLIP_Embedding/HateMM/{train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` | 744 (298) | 107 (43) |
| **MHC-EN** | frozen Qwen (EN closed, no LoRA deployed) | `data/CLIP_Embedding/MHC/{train,dev_seen}_Qwen2.5-VL-7B-Instruct_HF.pt` | 549 (168) | 80 (25) |

sha256 (read-only, verified at load — see §3):
```
b2e8e78d19c71d2ca674903586d53ca171c33a539956ee37c1c61f44a5e01f1d  MHC_zh/train_...-LoRA_HF.pt
4c07af75098391c999013e1cf6fb7ffe8fac29546d9ce329d51004a37e4f5d3c  MHC_zh/dev_seen_...-LoRA_HF.pt
5e80f39327a743144067857e6f8c9f0c909e3131bdc13bcb063be6abc333e7cf  HateMM/train_...-LoRA-curric_HF.pt
46ee4fd9fcaec80b7859a5e4c18b76e84b4020fa242ced802f289f790e4d7cb0  HateMM/dev_seen_...-LoRA-curric_HF.pt
05a9b2def29230259d7dca0e75b096edcbb483db98ea1c21d89d25c8a4940409  MHC/train_...-7B-Instruct_HF.pt
cd5d4c7dc08311f84df89c2841aa89ce4b6a0a4be4860422fbd4cc06974cd50c  MHC/dev_seen_...-7B-Instruct_HF.pt
```

All fitting (PCA, discriminators, KNIFE) is **train-only**; dev is a **read-only replication**
consumed with train-fitted models, exactly as the reference `estimation_main` does. Test splits are
not opened by this gate.

### 2.2 Projection arms (F41 precedent: a reduced arm AND a raw arm, so a null is interpretable)

Two conventions, both pre-declared, because §1.4 property 4 says the `R/S` split depends on the
*relative* scale convention of the two streams:

- **P-B "estimator-native" (whitened):** per-stream `PCA(n_components=d', whiten=True)` fit on
  **train only**. Puts both streams at unit isotropic scale = the regime `MargKernel`'s tanh-bounded
  variance was demo'd in. Destroys the streams' intrinsic relative scale by construction.
- **P-A "common-scale" (physical):** per-stream `PCA(n_components=d', whiten=False)` fit on train
  only, then **both** streams divided by **one shared scalar** `σ = sqrt(mean train per-dim variance
  pooled over both streams)`. Preserves relative scale (property 4 ⇒ `R`, `S` unchanged by σ).

| arm | dim | convention | role |
|---|---|---|---|
| **A1** | d'=64 | P-B whitened | **PRIMARY DECISION ARM** |
| **A2** | d'=256 | P-B whitened | dimension replication |
| **A3** | d'=64 | P-A common-scale | scale-convention sensitivity |
| **A4** | **d=3584 (RAW)** | P-A common-scale (train-mean centred, one shared σ) | **F41 raw-dim arm** — rules out "the projection destroyed the signal". Whitening is impossible here (`n < d`), so P-A only |
| **A5** | d'=64 | P-B whitened, **as-shipped (no `zero_grad`)** | code-fidelity arm |

**The verdict is read off A1 and must be *replicated* on A2 and *not contradicted in sign* by A3/A4.**

### 2.3 Reported quantities

Per (dataset × arm × split): `I1, I2, I12` (both prior conventions), `H1, H2`,
`R, U1, U2, S` (post-`RUS_adjustment`, as the reference does) and pre-adjustment raws,
per-sample distributional summaries of `s`, the **estimator-free** `S−R = I12−I1−I2`, the
context quantity `I12 − max(I1,I2)`, the three discriminator accuracies (overfit diagnostic), and
the **synergy share `S_share = S / I12`** (dimensionless, cross-dataset comparable).

### 2.4 Controls and nulls (thresholds come from THESE, not from taste)

- **N1 — label-permutation null (the primary threshold source).** Permute train labels, **refit the
  three discriminators only** (the KNIFE kernels are label-free, so they are correctly reused —
  this is the exact null "features unchanged, label association destroyed"), recompute the full
  decomposition. **B = 50** draws per dataset on A1; dev read uses an independently permuted dev
  label vector. Gives `q95(S)` and `q95(S_share)` under the null. This null is what absorbs the
  **joint-discriminator overfitting confound** (the joint head sees 2·d inputs and memorises more
  than the unimodal heads, which mechanically manufactures apparent synergy at n≈600).
- **C1 — duplicate-stream control (ground-truth S = 0 on OUR OWN features).** Set `x2 := x1 = img`.
  Then truly `u1=u2=0, r=i(img;y), s=0`. Full refit (two *independently initialised* kernels, as the
  reference does, so the control also exposes the estimator's own noise inside `min(h1,h2)`).
  This is the **false-synergy floor on real features at our n**.
- **C2 — within-stream split-half control (redundancy calibration).** `x1 = img[:, :1792]`,
  `x2 = img[:, 1792:]` as two pseudo-modalities, each projected by the arm's rule. Two halves of one
  encoder's hidden state should read redundancy-dominated.
- **G1 — XOR positive control AT OUR SAMPLE SIZE (the power gate).** `y ~ Bern(0.5)`,
  `b1 ~ Bern(0.5)`, `b2 = b1 XOR y`; `x_k = 3.0·(2b_k−1)·w_k + N(0,I_{d'})` with random unit `w_k`,
  `n = n_train` of each dataset. Ground truth: `I1 = I2 = 0`, `I12 = log 2 = 0.6931 nats`,
  `S = 0.6931`, `R = U1 = U2 = 0`, `S_share = 1.0`.
- **G2 — shipped synthetic sanity.** `gaussian_data.generate_gaussian_data` demo run verbatim
  (n=1600/400, ρ=0.5, noise 0.5, K=2), both as-shipped and `zero_grad`-fixed — confirms the port is
  faithful to the released entry point.

### 2.5 PRE-DECLARED DECISION RULE (frozen before running)

**Machinery gate (must pass first, else `LSMI_MEASUREMENT_INVALID`):**
- **(M1) Power:** G1 XOR-at-our-n must recover `S ≥ 0.30` nats **and** `S_share ≥ 0.50` on A1
  (truth 0.6931 / 1.00) for **all three** sample sizes. If the estimator cannot see a *pure*
  synergy at n≈600, a null on our data means nothing and the gate returns INVALID.
- **(M2) Specificity:** C1 duplicate-stream must return `S_share ≤ 0.20`. If a provably
  synergy-free input pair reads as materially synergistic, the estimator is not specific at our n.

**Given M1 ∧ M2, on each dataset (A1 primary, A2 replication, train read primary / dev replication):**

Let `S_floor := max( q95(S under N1), S from C1 duplicate-stream )` — the false-synergy floor.

- **(i) `FUSION_CAPPED` (the PAPER-VALUE outcome):** `S ≤ S_floor` **and** `S_share ≤ 0.10`
  **and** `R > U1 + U2` (redundancy-dominated). Interpretation: the two deployed streams carry
  essentially **no synergistic** task-relevant information; the fusion-architecture family
  (attention / gating / bilinear / concat-vs-Hadamard) is **mechanistically capped**, and the
  Hadamard head loses nothing that a richer block could recover. Supports F50 / F44 / F76.
- **(ii) `SYNERGY_PRESENT` on dataset D:** `S > S_floor` **and** `S_share ≥ 0.20`, replicated on
  dev and on A2. Interpretation: **D is named** as the dataset where a fusion upgrade could matter
  → it becomes the SynIB port target (`REPRO_SURVEY_2025.md` §5 #2b), and the survey's
  conditional item #4 (BalanceBenchmark) unlocks for D only.
- **(iii) `INDETERMINATE`:** anything between (`S_floor < S` but `S_share < 0.20`), or A1/A2
  disagree, or the train and dev reads disagree in verdict.

Sign-contradiction from A3 (scale convention) or A4 (raw dim) does not overturn (i)/(ii) by itself
but **must be reported in the verdict sentence**, per F41.

**Honest prior before running:** the project's fusion evidence (F50 rotation-at-every-w, F44 EN
image collapse, F76 anti-correlation, the in-flight fusion-concat family) points at (i). Run
exactly and let the numbers decide.

Script: `scripts/analysis/lsmi_gate.py` (CPU-only, per-cell checkpointed against the login-node
reaper). Results: `refine-logs/LSMI_GATE_OUT.json` + `refine-logs/LSMI_GATE_run.log`.

### 2.6 AMENDMENT AMD-1 / AMD-2 — declared after the MACHINERY GATES fired, before ANY real-data cell

**Trigger (a control, not a decision number).** The §2.4 gates were run first, on synthetic data
only. `G1` (XOR at our n, d'=64) returned, on the **released in-sample protocol**:

```
train acc  img 1.000 / text 1.000 / joint 1.000     <-- on a construction where img and text
dev   acc  img 0.477 / text 0.533 / joint 0.505         provably carry ZERO information about y
train I1 0.653  I2 0.656  I12 0.690  (log 2 = 0.6931)
```

i.e. at n≈600 the `cls_network` discriminators **memorise the train split completely**, so all
three pointwise MIs saturate at `log 2` and the in-sample decomposition is arithmetically
determined by memorisation rather than by information. This is a demonstrated pathology of the
*released reading protocol* at our sample size, found on a synthetic control **before a single
number was computed on any RGCL cache**. Amending now (rather than after seeing our data) is what
keeps the readout pre-registered.

- **AMD-1 (protocol).** Add a **K=5 stratified cross-fitted read**: for each fold the three
  discriminators *and* the two KNIFE kernels are fitted on the 4/5 in-fold part and the pointwise
  `i` / `h` are read on the held-out 1/5; the per-sample vectors are reassembled in original order.
  **`train_crossfit` becomes the PRIMARY read.** The released in-sample read (`train_insample`)
  and the held-out `dev` read are still computed and reported for every cell. For the raw arm A4
  only the discriminators are cross-fitted (a 5× KNIFE refit at d=3584 is ~7 h of CPU); the
  full-train KNIFE is reused there and this is flagged in the output (`crossfit_knife=false`).
  The **permutation null N1 is computed on the cross-fitted read** with the *same* per-fold KNIFE
  kernels reused (they are label-free, so this is the correct null), plus a full-train-fitted
  null for the dev read.
- **AMD-2 (power gate).** `G1` is extended to localise whether the power wall is **n** or **d**:
  XOR cells at `(n=579,d'=64)`, `(n=744,d'=64)`, `(n=549,d'=64)`, `(n=2000,d'=64)`,
  `(n=8000,d'=64)`, `(n=579,d'=8)`, `(n=8000,d'=8)`. **M1 is re-evaluated on the cross-fitted
  read.** If XOR is recovered at large n but not at n≈600, the honest conclusion is *"our datasets
  are too small to estimate PID"* — an outcome `REPRO_SURVEY_2025.md` §5 #1 explicitly
  pre-accepted ("If the estimator is unstable at n≈600, that itself is the finding … and it
  retires the synergy line at $0").
- **AMD-3 (guard).** `S_share = S/I12` is reported as **undefined** whenever `I12 < 0.05` nats
  (the ratio is meaningless when there is no total task-relevant information to take a share of).
  In that case the decision rule falls back to `S` vs `S_floor` alone.

- **AMD-4 (power gate: separate `n` from optimisation budget).** The released recipe trains the
  discriminators for a **fixed 30 epochs**, so the number of gradient steps is proportional to `n`
  (≈570 steps at n≈600 vs ≈7500 at n=8000). AMD-2's `n`-ladder therefore confounds *sample size*
  with *training budget*, and a null at n≈600 could be either. AMD-4 adds **matched-budget XOR
  arms at our own n** (`--epochs 400`, ≈7400 steps, i.e. G1b's step count at n≈600) plus a
  matched-budget `n=600` reference. **If and only if the matched budget restores XOR detection**,
  the three real datasets are re-read on the primary arm at the same budget (**arm A6**) with a
  fresh 50-draw permutation null and a fresh duplicate-stream control, and **A6 replaces A1 as the
  primary decision arm**; otherwise A6 is not run and A1 stands. Runner:
  `scripts/analysis/lsmi_gate_power.py` (imports `lsmi_gate.py`, edits nothing in it).

No other element of §2.1–§2.5 is changed: same caches, same arms, same controls, same thresholds,
same verdict labels.

### 2.7 Where it ran (infrastructure note, no GPU)

The gate is CPU-only. It was first attempted on the login node (the precedent of the prior $0
gates), but the login-node reaper **SIGTERMed every process past ~2 min of CPU** — 12 × `exit=143`
across three concurrent stages, and no per-cell checkpoint can survive a cell that never finishes.
The whole gate was therefore moved to a **CPU-only SLURM job**
(`scripts/slurm/lsmi_gate_cpu.sbatch`, job **13522**): `--cpus-per-task=16 --mem=64G`,
**no `#SBATCH --gres` line at all** (`scontrol show job` reports no `Gres`; the job log records
`nvidia-smi -L` → `No devices found`). **Zero GPU requested, zero GPU consumed, no Modal.** All
partial state from the login-node attempts was deleted before submission so every number below
comes from a single job under identical thread settings.

---

## 3. RESULTS
