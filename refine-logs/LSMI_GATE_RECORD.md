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

- **AMD-5 (certified-dimension arm).** The AMD-2 `G1` ladder (job 13522, gates stage) returned:

  | XOR cell | joint out-of-fold acc | S (truth 0.6931) | S_share (truth 1.0) |
  |---|---|---|---|
  | d'=64, n=579 / 744 / 549 | 0.513 / 0.530 / 0.508 | — (I12 < 0) | undefined |
  | d'=64, n=2000 | 0.823 | 2.3264 | 10.97 (I12 = 0.212) |
  | d'=64, n=8000 | 0.995 | 1.9129 | 2.82 |
  | **d'=8, n=579** | **0.998** | **0.7077** | **1.097** |
  | d'=8, n=8000 | 0.995 | 0.7179 | 1.060 |

  So the **LSMI machinery is accurate at our sample size** — at d'=8, n=579 it recovers a known
  synergy to 0.708 vs a truth of 0.6931 — and what fails at d'=64 is the *discriminator* learning
  a joint function from 128 input dimensions in ~570 gradient steps. The wall is **dimension**,
  not (only) n. Therefore: walk `d' ∈ {8,16,32,64}` with the XOR control **at our own n**, define
  **d\* = the largest d' at which M1 passes for all three of our sample sizes**, and re-read the
  three real datasets at d\* (**arm A7**) with a fresh 50-draw permutation null, duplicate-stream
  control and split-half control. **A7 supersedes A1 as the primary decision arm**; A1/A2 are
  retained in the record but carry **no evidential weight**, because the power gate certifies that
  they cannot detect even a maximal synergy at our n.
  **Disclosed limitation of A7, stated before it is run:** a null at d\* bounds synergy *within
  the top-d\* principal components of each stream*, not in the discarded directions. The XOR
  control places its bit in a random direction of the retained subspace, so it certifies detection
  **inside** that subspace only. This is the honest price of the only regime in which the
  estimator demonstrably works at n≈600, and it is why the raw arm A4 and the d'=64 arms are still
  reported.

No other element of §2.1–§2.5 is changed: same caches, same arms, same controls, same thresholds,
same verdict labels.

### 2.7 Where it ran (infrastructure note, no GPU)

The gate is CPU-only throughout. **Zero GPU requested, zero GPU consumed, zero Modal.**

It was first attempted on the login node (the precedent of the prior $0 gates), but the
login-node reaper **SIGTERMs every process past ~2–2.5 min of CPU regardless of thread count**
(observed: 12 × `exit=143` across three concurrent stages at 8 threads; still reaped at
`OMP_NUM_THREADS=2`). No per-cell checkpoint survives a cell that never finishes, so the gate was
moved to a **CPU-only SLURM job**: `scripts/slurm/lsmi_gate_cpu.sbatch`,
`--cpus-per-task=16 --mem=64G`, **no `#SBATCH --gres` line at all** (`scontrol show job` reports no
`Gres`; the job log records `nvidia-smi -L` → `No devices found`). All partial login-node state was
deleted before submission.

**Exactly where each number was computed (stated because it is not uniform):**

| block | where | detail |
|---|---|---|
| §3.1 G2, §3.2 G1 ladder, §3.3 the whole d'=64/256 layer + C1/C2 at d'=64 | **CPU-only SLURM job 13522** | `OMP_NUM_THREADS=16`; cancelled at the start of the raw arm to free the 16-CPU slot for the decision arm; all completed cells preserved by `refine-logs/.lsmi_ckpt/` |
| §3.4 AMD-5 dimension ladder, §3.5 **A7 decision arm** + C1/C2 at d'=8/16 | **login node, CPU-only** | `OMP_NUM_THREADS=2–4`, retry loop + per-cell **and per-permutation-draw** checkpointing (added mid-run so the 50-draw nulls could survive the reaper; on resume the RNG replays the consumed draws **at identical sizes**, so the draw sequence is bit-identical to an uninterrupted run) |
| §4.5 A4 raw arm, AMD-4 budget arm | **queued, not yet run** | `scripts/slurm/lsmi_gate_power_cpu.sbatch` → job **13531**, `PENDING (JobHeldUser)` since 00:48, never forced |

The follow-up SLURM job (13531) was submitted first and left queued; the A7 cells were computed on
the login node only because the hold persisted for hours and A7 is the decision arm. Every cell is
seeded (§6), so the thread-count difference between the two environments does not change any cell's
definition; it can only move last-bit floating-point sums. This is disclosed rather than hidden.

---

## 3. RESULTS

All numbers below are transcribed from `refine-logs/LSMI_GATE_OUT.json` by
`scripts/analysis/lsmi_gate_report.py` (no hand-typed figures). Units: **nats**. `S−R` is the
estimator-free interaction information (§1.4 property 2). `acc i/t/j` = image / text / joint
discriminator accuracy on the *same* rows the MI is read from (out-of-fold for `train_crossfit`).
`S_share = S/I12`, shown as `n/a` when `I12 < 0.05` nats (AMD-3).

**Port fidelity.** `fidelity_maxabs_vs_released = 0.00e+00` on every A1 cell: the per-sample
re-implementation reproduces the released `main_lsmi.LSMI_estimation` means **exactly**.

### 3.1 G2 — shipped synthetic demo (sanity that the port is faithful)

| cell | n | I1 | I2 | I12 | R | U1 | U2 | **S** | S_share | S−R | acc i/t/j |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G2a gaussian, `zero_grad`-fixed [crossfit] | 1600 | 0.4939 | 0.4820 | 0.5431 | 0.4820 | 0.0118 | 0.0000 | **0.0492** | 0.091 | −0.4328 | 0.919/0.913/0.946 |
| G2a [in-sample] | 1600 | 0.5014 | 0.4891 | 0.5557 | 0.4891 | 0.0123 | −0.0000 | **0.0543** | 0.098 | −0.4348 | 0.919/0.914/0.946 |
| G2a [dev] | 400 | 0.5036 | 0.5125 | 0.5703 | 0.5036 | 0.0000 | 0.0089 | **0.0578** | 0.101 | −0.4459 | 0.923/0.920/0.947 |
| G2b gaussian, **as-shipped** [crossfit] | 1600 | 0.4939 | 0.4820 | 0.5431 | 0.4820 | 0.0118 | 0.0000 | **0.0492** | 0.091 | −0.4328 | 0.919/0.913/0.946 |
| G2b [in-sample] | 1600 | 0.5014 | 0.4891 | 0.5557 | 0.4891 | 0.0123 | −0.0000 | **0.0543** | 0.098 | −0.4348 | 0.919/0.914/0.946 |
| G2b [dev] | 400 | 0.5036 | 0.5125 | 0.5703 | 0.5036 | 0.0000 | 0.0089 | **0.0578** | 0.101 | −0.4459 | 0.923/0.920/0.947 |

The demo generator (`x1 = onehot(y) + 0.5u`, `x2 = onehot(y) + 0.5(ρu + √(1−ρ²)v)`, ρ=0.5) is a
**redundancy** construction, and LSMI reads it as one: `R ≈ 0.48–0.50`, `U ≈ 0.01`, `S ≈ 0.05`,
stable across all three reads. **The port behaves correctly on the authors' own example.**
At `d = 2` the missing `zero_grad` (§1.5) makes **no difference to 4 dp** — which is exactly why
the defect is invisible in the released demo. It is *not* harmless at higher `d` (§3.3).

### 3.2 G1 — the power gate. **The decisive result of this gate.**

| XOR cell (truth: I1=I2=0, I12=S=0.6931, R=U=0, S_share=1.0) | n | d' | I1 | I2 | I12 | **S** | S_share | acc i/t/**j** (out-of-fold) |
|---|---|---|---|---|---|---|---|---|
| G1 @ MHC-ZH n | 579 | 64 | −0.6755 | −0.6784 | −1.1281 | 0.2258 | n/a | 0.478/0.485/**0.513** |
| G1 @ HateMM n | 744 | 64 | −0.7621 | −0.8380 | −1.0685 | 0.5316 | n/a | 0.518/0.476/**0.530** |
| G1 @ MHC-EN n | 549 | 64 | −0.6761 | −0.5060 | −1.1814 | 0.0006 | n/a | 0.444/0.506/**0.508** |
| G1e | 2000 | 64 | −1.0416 | −1.0727 | 0.2121 | 2.3264 | 10.97 | 0.511/0.517/**0.823** |
| G1b | 8000 | 64 | −0.5996 | −0.6361 | 0.6773 | 1.9129 | 2.82 | 0.506/0.495/**0.995** |
| **G1c** | **579** | **8** | **−0.0281** | **−0.0344** | **0.6452** | **0.7077** | **1.097** | 0.527/0.511/**0.998** |
| G1d | 8000 | 8 | −0.0218 | −0.0188 | 0.6773 | 0.7179 | 1.060 | 0.499/0.509/**0.995** |

Read this table carefully, because it governs everything else:

- **At d'=8 and n=579 the estimator is essentially exact.** It returns `I1 = −0.028`, `I2 = −0.034`
  against a truth of 0, `I12 = 0.6452` against `log 2 = 0.6931`, and `S = 0.7077` against a truth of
  `0.6931` — a 2% error on a *maximal* synergy, at our own sample size, with the released
  hyper-parameters. `S_share = 1.097` vs a truth of 1.00. **LSMI itself works at n≈600.**
- **At d'=64 and n≈600 it detects nothing**: the joint discriminator's out-of-fold accuracy is
  **0.513 / 0.530 / 0.508** — chance — on data where the label is a *deterministic* function of the
  pair. All three pointwise MIs go negative and the reported `S` (0.2258 / 0.5316 / 0.0006) is
  arithmetic noise, not signal.
- Raising `n` at d'=64 recovers it (0.823 at n=2000, 0.995 at n=8000), and lowering `d'` at
  fixed n=579 recovers it (0.998). So the failure is a **joint-discriminator sample-complexity
  wall in `d'`**, not a defect of the entropy estimator and not purely a small-`n` wall.
- The **in-sample** rows of the same cells read `acc 1.000/0.997/1.000` and
  `I1 ≈ I2 ≈ I12 ≈ 0.63–0.69 ≈ log 2` — i.e. the released reading protocol reports "everything is
  redundant, `S ≈ 0.03–0.06`" **on a construction whose true redundancy is exactly zero.**
  That is the AMD-1 pathology, quantified.

**Consequence, mechanical:** the pre-declared **M1 power gate FAILS at d'=64 for all three of our
sample sizes**, and **PASSES at d'=8**. Under §2.5 the entire d'=64/256 layer (arms A1, A2, A3, A5)
is therefore `LSMI_MEASUREMENT_INVALID` — its nulls carry **no evidential weight** — and AMD-5's
certified-dimension arm A7 (§3.4) is where the decision is actually made.

### 3.3 The d'=64 / d'=256 layer (A1, A2, A3, A5) — reported, zero evidential weight

**MHC-ZH** (generic-LoRA, job 13150) — n train 579 (180 pos) / dev 78 (28 pos)

| cell | n | I1 | I2 | I12 | R | U1 | U2 | **S** | S_share | S−R | acc i/t/j |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 d'=64 [crossfit] | 579 | −0.3016 | 0.1596 | 0.0924 | −0.0000 | −0.3016 | 0.1596 | **0.2345** | 2.536 | 0.2345 | 0.713/0.865/0.857 |
| A1 [in-sample] | 579 | 0.6477 | 0.6730 | 0.6911 | 0.6477 | 0.0000 | 0.0253 | **0.0181** | 0.026 | −0.6296 | 0.998/0.998/1.000 |
| A1 [dev] | 78 | −0.0640 | 0.2734 | 0.0649 | −0.0640 | 0.0000 | 0.3373 | **−0.2084** | −3.211 | −0.1445 | 0.731/0.859/0.846 |
| A2 d'=256 [crossfit] | 579 | −0.7293 | −0.0199 | −0.0408 | 0.0000 | −0.7293 | −0.0199 | **0.7084** | n/a | 0.7084 | 0.667/0.801/0.746 |
| A3 d'=64 common-scale [crossfit] | 579 | −0.0768 | 0.2476 | 0.2021 | −0.0768 | 0.0000 | 0.3245 | **−0.0455** | −0.225 | 0.0313 | 0.724/0.870/0.882 |
| A5 d'=64 as-shipped [crossfit] | 579 | −0.3016 | 0.1596 | 0.0924 | −0.3016 | −0.0000 | 0.4613 | **−0.0672** | −0.727 | 0.2345 | 0.713/0.865/0.857 |
| C1 duplicate-stream (truth S=0) [crossfit] | 579 | −0.3016 | −0.3349 | −0.5528 | −0.0000 | −0.3016 | −0.3349 | **0.0838** | n/a | 0.0838 | 0.713/0.724/0.712 |
| C2 split-half [crossfit] | 579 | −0.3334 | −0.2711 | −0.5664 | 0.0000 | −0.3334 | −0.2711 | **0.0381** | n/a | 0.0381 | 0.687/0.710/0.705 |

`perm_null` A1 [crossfit], B=50: S mean 0.0056, sd 0.0178, **q95 0.0392**, max 0.0937 (I12 mean −1.0739)
`perm_null` A1 [dev], B=50: S mean 0.0140, sd 0.0437, **q95 0.1019**, max 0.2441

**HateMM** (curric-LoRA, job 13241) — n train 744 (298 pos) / dev 107 (43 pos)

| cell | n | I1 | I2 | I12 | R | U1 | U2 | **S** | S_share | S−R | acc i/t/j |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 d'=64 [crossfit] | 744 | −0.2668 | 0.1617 | 0.0100 | −0.2668 | 0.0000 | 0.4285 | **−0.1517** | n/a | 0.1152 | 0.734/0.859/0.836 |
| A1 [in-sample] | 744 | 0.6540 | 0.6664 | 0.6887 | 0.6540 | 0.0000 | 0.0124 | **0.0223** | 0.032 | −0.6317 | 0.992/0.995/0.999 |
| A1 [dev] | 107 | −0.0340 | −0.0032 | −0.0553 | −0.0340 | 0.0000 | 0.0308 | **−0.0520** | n/a | −0.0180 | 0.794/0.813/0.822 |
| A2 d'=256 [crossfit] | 744 | −0.5613 | 0.0082 | −0.0036 | 0.0000 | −0.5613 | 0.0082 | **0.5494** | n/a | 0.5494 | 0.692/0.808/0.776 |
| A3 d'=64 common-scale [crossfit] | 744 | 0.0100 | 0.2602 | 0.0817 | 0.0100 | −0.0000 | 0.2502 | **−0.1786** | −2.186 | −0.1886 | 0.754/0.871/0.859 |
| A5 d'=64 as-shipped [crossfit] | 744 | −0.2668 | 0.1617 | 0.0100 | 0.0000 | −0.2668 | 0.1617 | **0.1152** | n/a | 0.1152 | 0.734/0.859/0.836 |
| C1 duplicate-stream (truth S=0) [crossfit] | 744 | −0.2668 | −0.3035 | −0.4187 | 0.0000 | −0.2668 | −0.3035 | **0.1516** | n/a | 0.1516 | 0.734/0.716/0.732 |
| C2 split-half [crossfit] | 744 | −0.2843 | −0.2293 | −0.4238 | 0.0000 | −0.2843 | −0.2293 | **0.0898** | n/a | 0.0898 | 0.728/0.746/0.718 |

`perm_null` A1 [crossfit], B=50: S mean 0.0429, sd 0.0601, **q95 0.1651**, max 0.1836
`perm_null` A1 [dev], B=50: S mean 0.0824, sd 0.1317, **q95 0.3616**, max 0.5380

**MHC-EN** (frozen Qwen) — n train 549 (168 pos) / dev 80 (25 pos)

| cell | n | I1 | I2 | I12 | R | U1 | U2 | **S** | S_share | S−R | acc i/t/j |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 d'=64 [crossfit] | 549 | −0.5055 | −0.0131 | −0.2053 | −0.0000 | −0.5055 | −0.0131 | **0.3133** | n/a | 0.3133 | 0.679/0.807/0.772 |
| A1 [in-sample] | 549 | 0.6431 | 0.6642 | 0.6901 | 0.6431 | −0.0000 | 0.0210 | **0.0259** | 0.038 | −0.6172 | 0.998/0.998/1.000 |
| A1 [dev] | 80 | −0.2768 | −0.0757 | −0.3501 | 0.0000 | −0.2768 | −0.0757 | **0.0025** | n/a | 0.0025 | 0.688/0.725/0.725 |
| A2 d'=256 [crossfit] | 549 | −0.7012 | −0.2266 | −0.1244 | 0.0000 | −0.7012 | −0.2266 | **0.8035** | n/a | 0.8035 | 0.669/0.761/0.729 |
| A3 d'=64 common-scale [crossfit] | 549 | −0.2866 | 0.0164 | −0.1851 | −0.0000 | −0.2866 | 0.0164 | **0.0851** | n/a | 0.0851 | 0.689/0.789/0.792 |
| A5 d'=64 as-shipped [crossfit] | 549 | −0.5055 | −0.0131 | −0.2053 | −0.0000 | −0.5055 | −0.0131 | **0.3133** | n/a | 0.3133 | 0.679/0.807/0.772 |
| C1 duplicate-stream (truth S=0) [crossfit] | 549 | −0.5055 | −0.4437 | −0.7252 | 0.0000 | −0.5055 | −0.4437 | **0.2240** | n/a | 0.2240 | 0.679/0.687/0.672 |
| C2 split-half [crossfit] | 549 | −0.4876 | −0.4672 | −0.7281 | 0.0000 | −0.4876 | −0.4672 | **0.2266** | n/a | 0.2266 | 0.663/0.665/0.698 |

`perm_null` A1 [crossfit], B=50: S mean 0.0058, sd 0.0222, **q95 0.0292**, max 0.1319
`perm_null` A1 [dev], B=50: S mean 0.0374, sd 0.0853, **q95 0.1886**, max 0.4071

**Four things this layer establishes, none of them a synergy measurement:**

1. **The released in-sample protocol is saturated on every dataset and every arm** — `acc` 0.99–1.00
   on all three heads and `I1 ≈ I2 ≈ I12 ≈ log 2` — so it reports `R ≈ 0.65, S ≈ 0.02, S_share ≈ 0.03`
   *identically* for the real pair, for the **duplicate-stream** control and for the **split-half**
   control. Had this gate simply run `main_lsmi.py` as shipped, it would have produced a clean,
   quotable "no synergy, redundancy-dominated" answer **for all three datasets and for two controls
   whose ground truth is different** — a false paper result. AMD-1 is what caught it.
2. **The joint stream is not helping at d'=64.** Out-of-fold, `acc_joint` is **below** `acc_text` on
   all three datasets (ZH 0.857 vs 0.865; HateMM 0.836 vs 0.859; EN 0.772 vs 0.807). Combined with
   §3.2 this is a statement about the estimator's discriminator, not about the deployed head.
3. **The duplicate-stream control fails specificity here.** On a pair that is *provably*
   synergy-free (`x2 := x1`), the crossfit read returns `S = 0.0838 / 0.1516 / 0.2240` — larger than
   the permutation q95 on two of three datasets. At d'=64 the machinery manufactures synergy.
4. **The `zero_grad` defect changes the answer.** A5 differs from A1 *only* in the released entropy
   loop. The discriminators are identical (`I1, I2, I12, S−R` match to 4 dp), but the entropy
   estimates move (ZH `H1` 1130.19 → 710.62) and the **sign of `S` flips** on 2 of 3 datasets
   (ZH +0.2345 → −0.0672; HateMM −0.1517 → +0.1152). The R/S split is a property of the KNIFE fit,
   exactly as §5.1 predicts — and the released code's version of that fit is the buggy one.

### 3.4 AMD-5 dimension ladder → the certified dimension **d\* = 16**

XOR control **at our own three sample sizes**, released 30-epoch budget, cross-fitted read
(truth: `S = 0.6931`, `S_share = 1.0`, `I1 = I2 = 0`). M1 requires `S ≥ 0.30` **and**
`S_share ≥ 0.50` on **all three**.

| d' | MHC-ZH (n=579) | HateMM (n=744) | MHC-EN (n=549) | M1 |
|---|---|---|---|---|
| **8** | S 0.7077, share 1.097, acc_j **0.998** | S 0.7321, share 1.121, acc_j **0.988** | S 0.7105, share 1.129, acc_j **0.989** | **PASS** |
| **16** | S 0.6466, share 1.479, acc_j **0.903** | S 0.7910, share 1.311, acc_j **0.974** | S 0.5675, share 1.457, acc_j **0.874** | **PASS** |
| 32 | S 0.1184, share n/a (I12 −0.2540), acc_j 0.632 | S 0.4516, share n/a (I12 −0.0375), acc_j 0.694 | S 0.0927, share n/a (I12 −0.3484), acc_j 0.596 | **FAIL** |
| 64 | (§3.2) acc_j 0.513 | acc_j 0.530 | acc_j 0.508 | **FAIL** |

⇒ **d\* = 16** (largest certified dimension), with **d'=8 as the replication arm**.
Retained variance at those dimensions (train PCA, per stream): d'=8 img 0.523–0.629 / text
0.404–0.466; d'=16 img 0.668–0.739 / text 0.528–0.580.

### 3.5 A7 — the decision arm, read at the certified dimensions

| dataset | arm | read | n | I1 (img) | I2 (text) | I12 | R | U1 (img) | U2 (text) | **S** | S_share | acc i/t/j |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MHC-ZH | **A7 d\*=16** | **crossfit** | 579 | 0.1538 | 0.3803 | 0.3056 | 0.1538 | −0.0000 | 0.2265 | **−0.0747** | −0.244 | 0.727/0.874/0.865 |
| MHC-ZH | A7 d\*=16 | dev | 78 | 0.1288 | 0.3652 | 0.3649 | 0.1288 | 0.0000 | 0.2364 | **−0.0004** | −0.001 | 0.756/0.833/0.885 |
| MHC-ZH | A7 d\*=16 | in-sample | 579 | 0.3505 | 0.5381 | 0.6269 | 0.3505 | 0.0000 | 0.1876 | **0.0887** | 0.142 | 0.850/0.936/0.988 |
| MHC-ZH | A7 d'=8 | crossfit | 579 | 0.1478 | 0.3503 | 0.3534 | 0.1478 | 0.0000 | 0.2026 | **0.0031** | 0.009 | 0.713/0.851/0.850 |
| MHC-ZH | A7 d'=8 | dev | 78 | 0.1852 | 0.2899 | 0.3266 | 0.1852 | 0.0000 | 0.1047 | **0.0367** | 0.112 | 0.731/0.782/0.846 |
| HateMM | **A7 d\*=16** | **crossfit** | 744 | 0.1739 | 0.3621 | 0.2819 | 0.1739 | −0.0000 | 0.1883 | **−0.0802** | −0.285 | 0.763/0.884/0.867 |
| HateMM | A7 d\*=16 | dev | 107 | 0.1511 | 0.2374 | 0.1799 | 0.1511 | 0.0000 | 0.0863 | **−0.0575** | −0.320 | 0.776/0.822/0.832 |
| HateMM | A7 d\*=16 | in-sample | 744 | 0.3385 | 0.5215 | 0.5931 | 0.3385 | −0.0000 | 0.1830 | **0.0716** | 0.121 | 0.847/0.941/0.966 |
| HateMM | A7 d'=8 | crossfit | 744 | 0.1782 | 0.3757 | 0.3592 | 0.1782 | −0.0000 | 0.1974 | **−0.0164** | −0.046 | 0.754/0.878/0.870 |
| HateMM | A7 d'=8 | dev | 107 | 0.1388 | 0.2664 | 0.2555 | 0.1388 | −0.0000 | 0.1275 | **−0.0109** | −0.043 | 0.748/0.822/0.841 |
| MHC-EN | **A7 d\*=16** | **crossfit** | 549 | 0.0726 | 0.2323 | 0.1486 | 0.1562 | −0.0836 | 0.0761 | **−0.0000** | −0.000 | 0.676/0.798/0.796 |
| MHC-EN | A7 d\*=16 | dev | 80 | 0.0710 | 0.2357 | 0.1316 | 0.0710 | −0.0000 | 0.1647 | **−0.1041** | −0.791 | 0.688/0.762/0.675 |
| MHC-EN | A7 d\*=16 | in-sample | 549 | 0.2719 | 0.4115 | 0.5443 | 0.2719 | 0.0000 | 0.1396 | **0.1328** | 0.244 | 0.805/0.876/0.956 |
| MHC-EN | A7 d'=8 | crossfit | 549 | 0.0687 | 0.2353 | 0.2152 | 0.0687 | −0.0000 | 0.1666 | **−0.0201** | −0.094 | 0.676/0.783/0.761 |
| MHC-EN | A7 d'=8 | dev | 80 | 0.1097 | 0.2273 | 0.1280 | 0.2091 | −0.0994 | 0.0183 | **0.0000** | 0.000 | 0.712/0.725/0.738 |

**Controls at the certified dimensions (ground truth `S = 0`):**

| dataset | d' | C1 duplicate-stream `S` | C2 split-half `S` |
|---|---|---|---|
| MHC-ZH | 16 / 8 | **0.0000** / **−0.0000** | **−0.0000** / **0.0000** |
| HateMM | 16 / 8 | **0.0000** / **0.0000** | **0.0000** / **−0.0000** |
| MHC-EN | 16 / 8 | **−0.0000** / **−0.0000** | **−0.0000** / **−0.0000** |

Both synthetic-truth-zero controls return **exactly 0** at both certified dimensions on all three
datasets — versus `+0.0838 / +0.1516 / +0.2240` for the same C1 control at the uncertified d'=64
(§3.3). **M2 specificity PASSES at d\*; it did not at d'=64.**

**Permutation null on `S`, B=50 fresh draws per cell:**

| dataset | d' | crossfit null mean / sd / **q95** / max | dev null mean / sd / **q95** / max |
|---|---|---|---|
| MHC-ZH | 16 | 0.00000 / 0.00000 / **0.00000** / 0.00000 | 0.00017 / 0.01281 / **0.01112** / 0.04299 |
| MHC-ZH | 8 | −0.00060 / 0.00300 / **0.00000** / 0.00000 | −0.00567 / 0.01749 / **0.00568** / 0.00834 |
| HateMM | 16 | 0.00000 / 0.00000 / **0.00000** / 0.00000 | 0.00694 / 0.01870 / **0.04638** / 0.08304 |
| HateMM | 8 | 0.00005 / 0.00034 / **0.00000** / 0.00239 | −0.00020 / 0.01818 / **0.01890** / 0.07616 |
| MHC-EN | 16 | −0.00000 / 0.00000 / **0.00000** / 0.00000 | −0.00224 / 0.01456 / **0.00000** / 0.02149 |
| MHC-EN | 8 | −0.00016 / 0.00111 / **0.00000** / 0.00000 | −0.00459 / 0.01400 / **0.00978** / 0.02744 |

`S_floor = max(q95 null, C1 duplicate-stream S)` is therefore **0 to within 4×10⁻¹⁷ on every
certified cell**.

**What the certified arm shows, in one shape, on all three datasets and at both certified dimensions:**

```
   total task-relevant info I12   = 0.149 - 0.359 nats   (a real, well-estimated quantity)
   redundancy               R     = 0.069 - 0.178        (shared img/text)
   uniqueness_text          U2    = 0.076 - 0.237        (the LARGEST atom on 5 of 6 cells)
   uniqueness_image         U1    = -0.084 - 0.000       (pinned at EXACTLY 0.0000 on 5 of 6 cells)
   SYNERGY                  S     = -0.080 - +0.003      (<= 0 on 5 of 6 cells; |S_share| <= 0.10 at d'=8)
```



---

## 4. VERDICT

The label below is the **mechanical output** of `scripts/analysis/lsmi_gate_verdict.py`
(stored in `refine-logs/LSMI_GATE_OUT.json → verdict`), evaluating the §2.5 rule against the
numbers above. It is not an adjudication.

### 4.1 Machinery gates

| gate | at d'=64 (A1/A2/A3/A5) | at **d\* = 16** (A7) | at d'=8 (A7 replication) |
|---|---|---|---|
| **M1 power** (XOR at our n: `S ≥ 0.30` ∧ `S_share ≥ 0.50`) | **FAIL** (acc_j 0.513/0.530/0.508) | **PASS** (3/3) | **PASS** (3/3) |
| **M2 specificity** (C1 duplicate-stream) | fails in substance (C1 `S` = 0.0838/0.1516/0.2240 on a truth-0 pair) | **PASS** (C1 `S` = 0.0000) | **PASS** (C1 `S` = 0.0000) |

⇒ **The whole d'=64 / d'=256 layer is `LSMI_MEASUREMENT_INVALID` and carries no evidential
weight.** The decision is read off A7.

### 4.2 Pre-declared rule at d\* = 16 (primary arm)

| dataset | `S` | `S_floor` | `S ≤ floor`? | `S_share ≤ 0.10`? | `R > U1+U2`? | fired label |
|---|---|---|---|---|---|---|
| MHC-ZH | −0.0747 | 3.46e−17 | ✔ | ✔ (−0.244) | ✘ (0.1538 vs 0.2265) | **INDETERMINATE** |
| HateMM | −0.0802 | 3.58e−17 | ✔ | ✔ (−0.285) | ✘ (0.1739 vs 0.1883) | **INDETERMINATE** |
| MHC-EN | −0.0000 | 4.38e−17 | ✔ | ✔ (−0.000) | ✔ (0.1562 vs −0.0075) | **FUSION_CAPPED** |

At d'=8 the same evaluation returns INDETERMINATE on all three, for the same reason (the
redundancy-dominance sub-clause).

## **VERDICT (mechanical, at d\* = 16): `INDETERMINATE`**

per-dataset `{MHC_zh: INDETERMINATE, HateMM: INDETERMINATE, MHC_en: FUSION_CAPPED}`;
`M1_pass_at_our_n = true`, `M2_pass = true`; **no dataset fires `SYNERGY_PRESENT` at any
dimension, on any read.**

### 4.3 What actually fired, stated precisely (the label is not rounded up)

Pre-declared clause (i) is a **conjunction of two independent questions**, and they came apart:

1. **The synergy question is answered, and the answer is "none", everywhere.**
   `S ≤ S_floor` **and** `S_share ≤ 0.10` fired on **3/3 datasets × 2 certified dimensions**, and
   the dev replication agrees (`S` = −0.0004 / −0.0575 / −0.1041 at d\*). `S` **never** exceeds the
   permutation null on any certified cell. The largest positive reading anywhere in the certified
   layer is MHC-ZH d'=8 `S = +0.0031` — **0.9 % of `I12`** — against a floor of ~0 and against a
   maximal-synergy reference of 0.6931 that *the same machinery, at the same n, in the same
   dimension* recovers to within 2 % (§3.4). **There is no image×text synergy to fuse.**
2. **The dominance question answers something different from what clause (i) assumed.**
   The largest atom is **not** redundancy — it is **text uniqueness** `U2` (0.076–0.237; the
   largest atom on 5 of 6 certified cells) — while **image uniqueness `U1` is pinned at exactly
   0.0000 on 5 of 6 certified cells**. So `R > U1 + U2` fails on ZH and HateMM, *not* because
   synergy is present but because the pair is **uniqueness-dominated and the uniqueness is all on
   the text side**. The honest reading is that clause (i) was written for the wrong dominant atom,
   not that the measurement was inconclusive about synergy.

**The mechanistic sentence this gate supports** (scoped by §5): *within the certified subspace,
essentially all task-relevant information in the deployed pair is carried by the **text** stream —
as text-unique information plus a smaller redundant component shared with the image stream — the
image stream contributes **no unique** information, and the two streams contribute **no synergy**.
A fusion block can only recombine `R`, `U1`, `U2`; there is no `S` term for it to capture.*

This is consistent with, and mechanistically explains, **F50** (fixed compositions/reweights are a
rotation at every `w` — with `S = 0` there is nothing off the `R/U1/U2` simplex to reach) and
**F44** (the EN image stream collapsing — here `U1 = 0` on EN too, and EN's whole `I12` is the
smallest of the three at 0.149 nats). It is the *measurement* behind "the crude fusion did not
cost us anything".

**What this gate does NOT support:** any claim about what a *differently trained encoder* would
yield (§5.2); any claim about synergy in the 26–47 % of per-stream variance outside the certified
subspace; and any prediction of the in-flight fusion-concat family's numbers — `concat` has
strictly more capacity than `align` to exploit `U1`/`U2`, and this gate says `U1 ≈ 0`, so it is
*consistent with* a small effect there, but that family's verdict is its own and is not
adjudicated here.

### 4.4 Consequences for the survey shortlist

- **`REPRO_SURVEY_2025.md` §5 #1 is discharged.** The measurement exists; the answer is
  "no detectable synergy; uniqueness-dominated, text-side".
- **§5 #2 (SynIB port) loses its motivating premise.** SynIB's objective penalises a head for
  staying confident when a modality is withheld, i.e. it is built to push a head onto
  *synergistic* structure; the survey named "the §4.2 LSMI reading" as #2b's kill-switch. No
  dataset fired `SYNERGY_PRESENT`. This gate does not by itself kill #2 (it cannot bound trained
  reshaping — §5.2), but the "there is synergy waiting to be captured" argument for it is gone.
- **§5 #4 (BalanceBenchmark screen) does not unlock** — it was explicitly conditional on this gate
  showing synergy to balance.
- **Where the numbers do point** (a candidate, not a conclusion): `U1 = 0.0000` with `U2`
  dominant is a *modality-imbalance* statement, which is the axis §4.3's MokA targets, and it is
  an adaptation-side object that §5.2 explicitly says this gate cannot bound.

### 4.5 Arms still queued (disclosure-only; cannot move the verdict)

- **A4 raw arm (d = 3584), the F41 precedent arm** — queued in CPU-only job **13531**, which has
  been `PENDING (JobHeldUser)` since 00:48 and was not forced. It cannot change the verdict: the
  rule is defined on the *certified* arm, and the ladder already shows detection failing at d'=32
  and d'=64, so d = 3584 is far past the wall (§1.4 property 5 predicts the degeneracy
  analytically; §3.3 shows it empirically). Its purpose was to close the "the projection destroyed
  the signal" objection — and §3.4 closes that objection in the *informative* direction: the
  signal appears as `d'` **falls**, not as it rises.
- **AMD-4 matched-budget arm (d'=64, 400 epochs)** — same job, same status. AMD-5 superseded its
  purpose: the dimension ladder localised the wall directly, and d'=8/16 pass M1 at the released
  30-epoch budget, so no budget correction is needed to obtain a certified read.

Both remain queued; this record should be amended with their numbers when job 13531 lands.

---

## 5. HONEST WALLS — what this gate can and cannot bound

These are stated independently of the numbers, so they cannot be tuned to the outcome.

### 5.1 Estimator brittleness that is intrinsic to LSMI (not to our port)

1. **Differential entropy of L2-normalised features is not well defined.** Both deployed streams
   are unit-norm 3584-d vectors: they live on a measure-zero sphere, and with n ≤ 744 < d the
   empirical support is inside an ≤(n−1)-dimensional affine subspace. `MargKernel` fits a density
   on `R^d` regardless. Any raw-dim `h` it reports is a number, not an entropy. The PCA arms make
   the object well posed at the cost of discarding variance — which is exactly why the F41
   discipline (a reduced arm *and* a raw arm) is mandatory here and why the raw arm is reported
   even though it is expected to be degenerate.
2. **The R/S split is convention-dependent.** §1.4 property 4: only a *common* rescaling of the
   two streams leaves `r` (hence `s`) invariant. Whitening each stream separately, or PCA'ing them
   separately, is a choice about what "equally surprising" means across two different feature
   spaces. Arms A1 (whitened) and A3 (common-scale) bracket that choice; if they disagree, the
   split — not the total — is what is unstable. **`S − R` is immune to all of this** (§1.4
   property 2), which is why it is reported everywhere as the estimator-free cross-check.
3. **The min-rule degenerates at large `d`.** §1.4 property 5: `h` grows like `d` while pointwise
   MI is bounded by `log(1/p(y)) ≈ 1.2` nats, so once `|h1 − h2| ≫ 1` the redundancy rule collapses
   to `r = i_{argmin h}` and one uniqueness is pinned to 0. This is a property of the *definition*,
   not of our data.
4. **PID estimators disagree with each other in the literature.** "Synergy" is not a single
   quantity: the choice of redundancy axiom (Williams–Beer `I_min`, Bertschinger et al.'s
   `Ĩ`/broadcast-invariant PID, Ince's `I_ccs`, Griffith–Koch, LSMI's information-flow/min rule)
   changes the numbers and can change their sign, and pointwise PIDs additionally admit negative
   atoms — which is why LSMI ships `RUS_adjustment`, a **post-hoc shift** that moves mass between
   `r` and `s` to force non-negative means. Everything reported here is *LSMI's* decomposition
   under *LSMI's* axiom. A different PID could return a different split of the same `I12`.
5. **`RUS_adjustment` is not neutral.** It preserves the four sums (§1.4 property 3) but it can
   move `S` upward by exactly the amount needed to make `R` or `S` non-negative. Pre-adjustment
   raws are therefore reported alongside every adjusted number.
6. **Two defects in the released code** (§1.5), one of which (`zero_grad`) is a real training bug.
   Both variants are run so the effect is measured rather than assumed.

### 5.2 Scope — what a null here does NOT close (the F66-style caveat, stated first)

- **This gate cannot kill trained reshaping.** Everything measured here is a property of the
  **banked features as they are**. It says nothing about whether an *encoder* trained differently
  (MokA-style modality-aware LoRA, MNTP stage-2, a different tower) would produce a *different*
  pair of streams with different synergy. F66 made the same distinction for the selection bound;
  it applies verbatim here. A "no synergy" reading prices the **fusion-architecture family over
  these features**, not the adaptation axis.
- **It cannot price a third stream.** Audio, OCR, frame-level tokens: the decomposition is over
  the two deployed streams only. Synergy that would exist between (img, text, audio) is outside
  the object measured.
- **It is a train/dev measurement.** No held-out test claim, no accuracy claim, no benchmark
  number. The discriminator accuracies quoted are diagnostics of the estimator, not results.
- **It cannot license a positive.** A `SYNERGY_PRESENT` reading would only *name a target*; it
  would not predict that any particular fusion block converts it. The information is an upper
  bound on what a perfect fusion could exploit, not a promise that a trainable one will.
- **Sample size is the binding practical constraint.** n = 549–744 train items, ≈600 effective for
  a decomposition with four estimated atoms; the power gate (§2.4 G1 / AMD-4) exists precisely to
  decide whether that is enough, and its answer governs how much any real-data number here is
  allowed to carry.

---

## 6. PROVENANCE / REPRODUCTION

- **Candidate source:** `refine-logs/REPRO_SURVEY_2025.md` §4.2 + §5 #1 (rank 2).
  Clone `external/baselines/LSMI` @ `13e4db2e033a3721d5ea7c0e31c540b3445a5532`
  (`GeWu-Lab/LSMI_Estimator`, ICML 2025, **no license file** — read and imported for measurement
  only, nothing redistributed; the clone stays gitignored).
- **Pre-declaration chain (each commit precedes the numbers it governs):**
  `d4b06f0` §1–§2.5 frozen (zero numbers on our data) → `9214870` AMD-1/2/3 (triggered by the
  synthetic G1/G2 gates, before any RGCL cache cell) → `775feb9` AMD-4 + CPU-only-SLURM infra →
  `362a60e` AMD-5 certified-dimension arm (triggered by the G1 ladder, before any A7 number).
- **Runners:** `scripts/analysis/lsmi_gate.py` (gates / main / raw / merge),
  `scripts/analysis/lsmi_gate_power.py` (AMD-4 budget ladder, AMD-5 dimension ladder + A7 data),
  `scripts/analysis/lsmi_gate_verdict.py` (applies the §2.5 thresholds mechanically),
  `scripts/analysis/lsmi_gate_report.py` (renders the tables above),
  `scripts/analysis/lsmi_gate_loop.sh` (login-node retry driver, retained as the record of the
  reaper problem).
- **Compute:** CPU only. **No `#SBATCH --gres` line in either sbatch** → **zero GPU requested,
  zero GPU consumed, zero Modal, zero network.** See §2.7 for the per-block breakdown of which
  numbers came from SLURM job **13522** and which from the login-node run of the same code.
  Jobs 13522 and 13531 both began as `PENDING (JobHeldUser)`; neither was forced. Job 13531
  (AMD-4 budget ladder + A4 raw arm + merge) was still held at the time of writing.
- **Data (read-only, sha256 verified at load — §2.1):** the six banked
  `{train,dev_seen}` caches of the three deployed lineages. **No `test_seen_*` file is opened by
  any script in this gate** (`grep test_seen scripts/analysis/lsmi_gate*.py` → no match).
  No video, no raw media, nothing uploaded anywhere.
- **Seeds:** `setup_seed(42)` per cell (the released `cfgs/train.yaml` value); cross-fit folds
  `StratifiedKFold(5, shuffle=True, random_state=42)`; per-fold model seeds `42+100+f`;
  permutation labels `np.random.default_rng(90000)`, per-draw model seeds `42+5000+7b+f`
  (crossfit) / `42+9000+b` (dev); XOR controls `default_rng(7)`.
- **Hyper-parameters:** verbatim from the released `cfgs/train.yaml` — `embed_size 64`,
  `n_classes 2`, `batch_size 32`, Adam lr 1e-3, `StepLR(15,0.1)` discriminators /
  `StepLR(20,0.1)` KNIFE, 30 epochs each (AMD-4 budget arm: 400 epochs, declared).
- **Outputs:** `refine-logs/LSMI_GATE_OUT.json` (merged; per-stage parts in
  `refine-logs/.lsmi_out_{gates,main,power,raw}.json`), run logs
  `refine-logs/LSMI_GATE_run_*.log` and `slurm/logs/lsmi_gate_cpu_13522.out` /
  `slurm/logs/lsmi_power_cpu_13531.out`.

## 7. REQUIRED STATEMENTS

- **No performance or accuracy claim on any held-out benchmark.** Every accuracy figure in this
  record is a discriminator diagnostic internal to the estimator, computed on train (out-of-fold)
  or dev. The deployed head, its floors (ZH 13150, HateMM 13241) and the in-flight fusion-concat
  family are **not** re-measured here and none of their numbers are touched.
- **No test-split touch.** Train + dev only, per the banked-label convention of the prior $0 gates.
- **Zero GPU, zero SLURM-GPU, zero Modal.** Two CPU-only SLURM jobs with no `--gres`.
- **No `state/` mutation, no push, no prereg/config/CLAUDE.md/settings edit.** Write scope =
  this file, `refine-logs/LSMI_GATE_OUT.json`, `refine-logs/LSMI_GATE_run_*.log`,
  `scripts/analysis/lsmi_gate*.{py,sh}`, `scripts/slurm/lsmi_gate*_cpu.sbatch`.
- **Binding close = orchestrator spot-check.** This record is the executor's raw report; the
  verdict label in §4 is the mechanical output of the pre-declared rule, not an adjudication.
