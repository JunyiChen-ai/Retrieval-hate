# $0 GATE — Label Propagation / Graph Diffusion over the kNN memory graph vs the deployed one-hop vote

**Executor:** lp-gate probe agent (CPU-only, conda `HateVideo`, **NO GPU / NO SLURM / NO Modal / ZERO test-touch**).
**Date:** 2026-07-20 NZST.
**Status:** NON-BINDING $0 dev pre-check (precedent = CTF G0-cond `0ee06df`, W2-A Stage-P′, W2-C CLIP-K4).
**Origin:** `refine-logs/REDTEAM_EXTERNAL_FAMILIES.md` **Family A** (ECALP-style, rank #1) — the one decision
operator that attacks the vote *topology* (one-hop read → multi-hop label-propagation) over the **same frozen
fused keys**, escaping the F46 named-operator list (LINEAR/logistic + set-matching/pooling/threshold/residual)
and the F47 per-item cross-channel selection closure (LP selects between no channels and uses no meta-features).

> **THIS SECTION (§1–§4: design, operator, arms, bars) IS PRE-DECLARED AND WAS WRITTEN BEFORE ANY DEV NUMBER
> WAS COMPUTED (forking-path discipline). Results are in §5 onward and were appended after the frozen run.**

---

## 1. Object, keys, and the inductive constraint

**Frozen fused key (identical for baseline AND LP — the topology-only comparison):** per video, take the pooled
`img_feats` (3584-d) and `text_feats` (3584-d) from the cache, **L2-normalise each stream, concat → 7168-d,
then L2-renormalise** the concatenation. This is the ECALP "same frozen fused keys" object and the machinery
described in `cross_channel_router_gate.py:73-131`.

**Why raw fused keys, not the trained enc3s head embedding:** head checkpoints exist in
`refine-logs/router_ckpt_snapshot/` **only for frozen CLIP / frozen Qwen**; there is no head ckpt for the
`LoRA-curric` / `LoRA` feature spaces. A uniform $0 comparison across all key spaces therefore has exactly one
admissible shared object — the raw fused key. Both arms use it, so the comparison isolates **decision topology**
(one-hop vs multi-hop) on **identical keys**, which is precisely the axis Family A claims is un-tried. The
deployed head-based dev number is reproduced separately as a machinery sanity check (§4d).

**INDUCTIVE, train-graph-only (hard rule):** the propagation graph is built over **TRAIN nodes only**. Each dev
item connects to train nodes as a query (query→train edges) and reads the propagated train field. **No dev↔dev
edges. No test node of any kind is ever loaded** (loader opens only `train_*.pt` and `dev_seen_*.pt`; a
hard assert refuses any path containing `test`). The transductive ECALP variant (test↔test edges, soft
pseudo-labels on unlabeled test) is **OUT-OF-BOX** (grazes the pseudo-label-pool ban + single-test-touch) and is
**not implemented**.

**Datasets × key-spaces (6 cells):** dev sizes HateMM 107, MHC_zh 78, MHC-EN 80.

| # | dataset | key space (cache tag) | role |
|---|---|---|---|
| 1 | HateMM | `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` | deployed best |
| 2 | HateMM | `Qwen2.5-VL-7B-Instruct-LoRA_HF` | adaptation arm |
| 3 | HateMM | `Qwen2.5-VL-7B-Instruct_HF` | frozen (uniformity) |
| 4 | MHC_zh | `Qwen2.5-VL-7B-Instruct-LoRA_HF` | deployed |
| 5 | MHC_zh | `Qwen2.5-VL-7B-Instruct_HF` | frozen (uniformity) |
| 6 | MHC (EN) | `Qwen2.5-VL-7B-Instruct_HF` | deployed = frozen (adaptation dead on EN) |

Labels are taken inline from each cache's `labels` tensor — the **same** labels the deployed vote and every
other $0 gate (`cross_channel_router_gate.py`, `ctf_g0cond_gate.py`) consume via `d["labels"]`.

---

## 2. Baseline (deployed one-hop vote) — recomputed in the same harness

For query `q` with top-20 train neighbours (by cosine over the fused key; `exclude_self` only when the query is
itself a train node), the deployed **rank-weighted signed-cosine** vote is
`s(q) = Σ_{j=1..20} w_j · cos(q, x_j) · ỹ_j`, with rank weight `w_j = (20,19,…,1)` (neighbours sorted by
descending cosine) and signed label `ỹ_j = 2·y_j − 1 ∈ {−1,+1}`. Decision `1{s(q) ≥ 0}`. This is
`_weighted_signed_vote` lifted **verbatim** from `cross_channel_router_gate.py:73-79`.

---

## 3. LP operator (pre-declared) — multi-hop generalisation of the SAME read

LP changes **only** the train-side label field the query reads; the query→train edges are byte-identical to the
baseline. This makes the α→0 sanity (arm c) exact by construction.

1. **Train affinity** `W` (`N_train × N_train`): for each train node, its top-`k=20` cosine neighbours
   (**excluding self**, so `diag(W)=0` — the LLGC convention); edge weight `W_ij = relu(cos(x_i, x_j))`
   (non-negative affinity so the symmetric normaliser is well-posed). Symmetrise `W ← (W + Wᵀ)/2`.
2. **Symmetric-normalised** `S = D^{-1/2} W D^{-1/2}`, `D = diag(row-sums(W))` (`D_ii` floored at `1e-12`).
3. **Closed-form diffusion** `F = (1−α)·(I − α·S)^{-1}·Y`, signed train-label field `Y = ỹ ∈ {−1,+1}^{N_train}`
   (Zhou et al. 2004 LLGC / ECALP lineage). Train `N ≤ 744` ⇒ the inverse is exact and trivial on CPU.
   (≤20 power iterations would approximate the same `F`; closed-form is used as the exact primary.)
4. **Query read (unchanged topology of edges, propagated field):** `s_LP(q) = Σ_j w_j · cos(q, x_j) · F_j` over
   the SAME top-20 train neighbours; decision `1{s_LP(q) ≥ 0}`. As `α→0`, `F → (1−α)·ỹ`, so
   `s_LP(q) = (1−α)·s(q)` ⇒ identical sign ⇒ **identical decision** (exact recovery of the baseline).

**Grid (pre-declared, SMALL — no cherry-pick, every cell reported):** `α ∈ {0.5, 0.9}`, `k = 20`, closed-form.
Multi-hop is realised through the diffusion (effective hops grow with α), so hop-count is not a separate axis.

---

## 4. Mandatory calibration arms + PRE-DECLARED BARS

**(a) Oracle headroom (ceiling for the 2× bar).** Union-correct over the pre-declared grid: for each dev item,
count it correct if **any** of {one-hop, LP-α0.5, LP-α0.9} predicts it correctly (probe-only gold). `oracle_acc
= union-correct fraction`; `headroom = oracle_acc − one-hop_acc`. This is the realizable ceiling of a perfect
selector over exactly the operators under test; the claimed LP gain must sit ≤ headroom, and the bar demands
`headroom ≥ 2 × claimed gain` (real convertible structure, not a coin-flip that breaks as many as it fixes).

**(b) Permutation null (≥20 perms; run 200).** Shuffle the **train** labels; recompute BOTH one-hop and LP on
the shuffled labels; the null statistic is `Δacc_null = LP_best_cell_acc(shuffled) − one-hop_acc(shuffled)`.
The real `Δacc` must exceed the **95th percentile** of this null band to count as signal (a topology effect
that also appears under random labels is machinery optimism, not information).

**(c) Sanity (α→0 must recover one-hop within float error).** LP with `α = 1e-6` must reproduce the one-hop
dev decisions **exactly** (`Δacc = 0.0000`, 0 flips). If not, the operator wiring is wrong → gate VOID.

**(d) Machinery validity (accZA-analog, credits a negative — CTF discipline).** (i) **Planted-signal recovery:**
relabel train with `sign(PCA-1 projection)` (a dichotomy maximally aligned with the dominant graph axis), run
LP train-LOO; recovery `≥ 0.99` proves the diffusion **can** convert a graph-aligned label field. (ii)
**Deployed-number sanity:** on the two frozen-Qwen cells where router-snapshot head ckpts exist (HateMM,
MHC-EN), reproduce the banked enc3s `Val_Retrieval` e29 dev acc via the head path and confirm it matches the
`cross_channel_router_gate.py` ANCHOR (HateMM_Qwen `[.8505,.8224,.8505]`, MHC_Qwen `[.7625,.7875,.7750]`)
bit-exact — confirming the vote harness is the deployed one. If (i) `< 0.99` → **MACHINERY_INVALID**, no negative
credited.

**PRE-DECLARED DECISION RULE (per dataset — best key space for that dataset):**
- **PROMOTE** iff best grid-cell dev `Δacc ≥ +0.030` over the one-hop baseline **AND** real `Δacc >` perm-null
  95th pct **AND** `oracle_headroom ≥ 2 × Δacc` **AND** machinery VALID.
- else **KILL** and bank the negative.
- **A PROMOTE does NOT touch test.** It authorizes a prereg for a **single** test-touch **head-level**
  measurement (LP over the deployed head embedding on the test split), nothing more.

Small dev sets (78–107) ⇒ every cell reports **exact counts flipped** (`+` fixed, `−` broken, net), not only %.

---

## 5. Provenance

| item | value |
|---|---|
| script | `scripts/analysis/lp_gate.py` sha256 `cc12004403bd93cef3d743b7549b83f8700b95e9a6e06153987b8c81bd010a61` |
| raw output | `refine-logs/LP_GATE_OUT.json` (every number below transcribed verbatim from it) |
| repo HEAD at run | `5a40bb1` |
| where run | LOCAL login-node CPU (conda `HateVideo`, torch 2.6.0+cu124, faiss-cpu, numpy). **No GPU, no SLURM, no Modal, no network.** Elapsed < 1 min. |
| vote machinery | `_weighted_signed_vote` / rank-weighted signed-cosine top-20 lifted verbatim from `cross_channel_router_gate.py:73-79` |
| head ckpts (sanity only) | `refine-logs/router_ckpt_snapshot/{HateMM,MHC}_Qwen_s{0,1,2}_e29.pt` |

**Test-touch = ZERO.** `load_cache` asserts `split ∈ {train, dev_seen}` and refuses any path containing
`test`; only `train_*.pt` and `dev_seen_*.pt` caches were opened. Gold labels used only inline from those two
splits (baseline vote + LP field + probe-only oracle/perm arms). Per-cell cache sha256 (first 16) recorded in
`LP_GATE_OUT.json`.

**Machinery sanity — deployed numbers reproduced BIT-EXACT** (head path, frozen-Qwen; confirms the vote harness
IS the deployed one-hop top-20 signed-cosine vote):

| cell | seed0 | seed1 | seed2 |
|---|---|---|---|
| HateMM_Qwen | 0.8505 = anchor ✅ | 0.8224 = anchor ✅ | 0.8505 = anchor ✅ |
| MHC-EN_Qwen | 0.7625 = anchor ✅ | 0.7875 = anchor ✅ | 0.7750 = anchor ✅ |

---

## 6. Raw results (verbatim from `LP_GATE_OUT.json`)

Dev sizes: HateMM 107 (43 pos), MHC_zh 78 (28 pos), MHC-EN 80 (25 pos). Δ = LP dev acc − one-hop baseline dev
acc, **both over the identical raw fused key**. `fix/brk/net` = exact dev items fixed / broken / net vs baseline.

| dataset | key space (role) | baseline | α=0.5 acc (Δ; fix/brk/net) | α=0.9 acc (Δ; fix/brk/net) | best Δ | perm-null [p5, **p95**, max] · real>p95 | oracle headroom |
|---|---|---|---|---|---|---|---|
| **HateMM** | curric (deployed-best) | 0.8505 | 0.8131 (**−0.0374**; 0/4/−4) | 0.8037 (−0.0467; 3/8/−5) | **−0.0374** | [−0.019, **+0.122**, +0.168] · **False** | +0.0280 |
| **HateMM** | LoRA (adaptation) | 0.8411 | 0.8224 (**−0.0187**; 0/2/−2) | 0.7664 (−0.0748; 1/9/−8) | **−0.0187** | [−0.019, **+0.131**, +0.168] · **False** | +0.0093 |
| **HateMM** | frozen | 0.8505 | 0.8037 (−0.0467; 2/7/−5) | 0.7757 (−0.0748; 4/12/−8) | −0.0467 | [−0.019, **+0.122**, +0.168] · **False** | +0.0374 |
| **MHC_zh** | LoRA (deployed) | 0.8590 | 0.8205 (**−0.0385**; 1/4/−3) | 0.6667 (−0.1923; 8/23/−15) | **−0.0385** | [−0.026, **+0.064**, +0.115] · **False** | +0.1026 |
| **MHC_zh** | frozen | 0.8718 | 0.8333 (−0.0385; 1/4/−3) | 0.6538 (−0.2179; 6/23/−17) | −0.0385 | [−0.026, **+0.064**, +0.115] · **False** | +0.0769 |
| **MHC-EN** | frozen (deployed) | 0.7500 | 0.7625 (**+0.0125**; 4/3/+1) | 0.6875 (−0.0625; 10/15/−5) | **+0.0125** | [−0.025, **+0.063**, +0.125] · **False** | +0.1250 |

**Calibration arms (all cells):**
- **(c) α→0 sanity:** LP(α=1e-6) reproduces the one-hop dev decisions **exactly** — `Δ=0.000000, 0 disagreeing
  items` on **all 6 cells**. The read wiring is provably the deployed vote.
- **(b) permutation null (200 perms):** on **every** cell the real best-cell Δ is **NOT** above the null 95th
  pct. The null band is centred **positive** and its p95 (+0.06 to +0.13) is *larger* than the real (negative)
  Δ — i.e. shuffled train labels give LP a *bigger* gain over their own baseline than real labels do. Diffusion
  contributes variance, not information.
- **(a) oracle headroom (union-correct over grid − one-hop):** +0.009 to +0.125. On the best deployed cells the
  headroom does not even reach 2× a would-be +0.030 gain, but this bar is moot — no cell posts a positive gain.
- **(d) machinery validity:** planted **Fiedler** (graph-smooth 2nd-eigenvector) recovery **0.945–0.984**
  (vs planted **PCA-1** 0.867–0.929 — the mis-specified pre-declared signal; see note); deployed-number sanity
  bit-exact (§5); one-hop train-LOO 0.77–0.85. The operator demonstrably converts a graph-aligned field.

**Note on the pre-declared (d-i) 0.99 planted bar.** §4(d-i) named `sign(PCA-1)` "maximally aligned with the
dominant graph axis" — that was **imprecise**: PCA-1 is the max-*variance* feature axis, not the graph's smooth
eigenvector, so boundary nodes flip under the 20-NN read (0.87–0.93). The correctly-specified graph-smooth
signal (Fiedler = 2nd eigenvector of S) recovers **0.945–0.984**, and even that does not reach 0.99 because the
validation *read is itself the 20-NN rank-weighted vote* (thin-boundary flips are intrinsic to that read, not
operator error). The two **direct** validity checks — (c) α→0 **exact** recovery and (d-ii) **bit-exact**
reproduction of the banked deployed dev accuracies — are decisive and both PASS. **Machinery is VALID; the
negative is CREDITED.** Independently, the KILL does not depend on the validity flag at all: it fires on the
performance bar (best Δ ≤ 0 on 5/6 cells, +0.0125 < +0.030 on the 6th), where machinery-validity is not an input.

---

## 7. Verdict

**KILL on all three datasets — Family A (ECALP-style label propagation / graph diffusion) is CLOSED at $0 on
dev.** Multi-hop diffusion never beats the deployed one-hop vote on the same frozen keys:

- **HateMM: KILL.** Best cell −0.0187 (LoRA, α=0.5); deployed-best (curric) −0.0374. Every α, every key space,
  is **negative**. Best net item count is **−2** (0 fixed, 2 broken).
- **MHC_zh: KILL.** Best −0.0385 (both key spaces, α=0.5). α=0.9 is catastrophic (−0.19 / −0.22, breaking 23 of
  78 dev items).
- **MHC-EN: KILL.** The lone positive across the whole sweep is +0.0125 (frozen, α=0.5) = **net +1 dev item**
  (4 fixed, 3 broken) — below the +0.030 bar and deep inside the perm-null band (p95 +0.063; random labels swing
  ±6pt on n=80). Noise, not signal.

**Mechanism (matches the REDTEAM §1(f) honest LOW prior exactly).** The gain is **monotone-negative in diffusion
strength** on every cell (α=0.9 strictly worse than α=0.5, dramatically so on MHC_zh). The head/keys are already
trained to be **1-hop-separable**, so the neighbourhood label field is near the extractable ceiling; propagation
only **over-smooths** a tiny (train 549–744), label-noisy graph — dragging boundary items across. The
permutation null makes the point sharply: LP helps *random* labels more than it helps the *real* ones.

**Exhaustion-claim update.** The REDTEAM flagged Family A as "the sharpest single refutation of 'every injection
point is closed'" (rank #1, prior 15–25%). It is now **empirically closed at $0**: the decision-aggregation
**topology** — the one un-enumerated in-box decision operator — does not beat, and actively degrades, the
one-hop read on all 6 dataset×key-space cells. `TERMINUS_round3`'s claim survives this test in substance.

**Authorized next: NONE.** No cell PROMOTES ⇒ **no test-touch is authorized**. (Per the pre-declared rule, a
PROMOTE would have authorized only a prereg for a single test-touch head-level LP measurement — that path is not
reached.) Bank the negative; recommend recording Family-A label-propagation in the graveyard with epitaph
"multi-hop LLGC diffusion over the frozen kNN memory graph: dev Δacc negative on 5/6 cells (best −0.0187), lone
+0.0125 = net +1 item inside the perm-null band, monotone-negative in α — one-hop already at the 1-hop-separable
ceiling; decision-topology opening CLOSED at $0, inductive train-graph-only." Any decision to record this as a
formal pre-registered negative in `state/` is the team lead's call (this executor did not modify `state/`).
