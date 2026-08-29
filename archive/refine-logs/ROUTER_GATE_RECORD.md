# ROUTER GATE RECORD — mechanism-guided per-item cross-channel routing ($0 screening gate)

**Executor:** router-gate executor (ZERO GPU, ZERO test-touch, banked artifacts only; no Modal).
**Date:** 2026-07-17. **Repo HEAD at design time:** (recorded in provenance §7).
**Raw-only record.** The executor applies the pre-declared mechanical kill/pass rules and attaches a
**NON-binding** label; the binding verdict is a separate reviewer's job.

---

## 0. Direction & why it is not already dead (non-isomorphism — verified against banked records)

**Direction (round-4, user-directed continuation):** *mechanism-guided per-item cross-channel routing.*
Per video, predict which prediction **channel** to trust — the CLIP-encoder RGCL arm vs the
Qwen2.5-VL-7B-encoder RGCL arm — from **decision-level meta-features** (per-item kNN vote margins,
neighbour label agreement/purity, rank/similarity components, per-modality sub-votes, channel
disagreement indicators), converting the F44-documented MHC-EN encoder rotation into a Pareto gain.

Non-isomorphism to the closed axes (each verified by reading the cited record):

- **K9 conditional-info zeros (W2-A F42, CTF F39, GIR F43)** are **linear-probe zeros over FEATURE
  space** (`Z_best`). The router consumes **outputs of the vote** — per-item vote margins, neighbour
  label purity, rank-weighted vote components, per-modality sub-votes — which are **not present in
  `Z_best`**, plus **nonlinearity**, and only on the **disagreement subset**. Different feature family,
  different function class, different support.
- **B5 (`B5_VERDICT_REVIEW.md`)** kills **global per-encoder decision-threshold / operating-point
  calibration** ("ZH per-encoder decision-threshold calibration as a conversion lever … closed";
  the oracle is a single global τ per arm). **Per-item routing between two arms is a different family**
  (a per-item selector, not a global threshold on one arm's score).
- **P1** (scalar prior recalibration) and **P2** (neighbour rerank) are different mechanisms (single
  channel; no cross-channel selection).
- **F44 (`ENCODER_SWAP_DIAGNOSIS.md`, 8a48938)** proves the disagreements are **modality-structured at
  the POPULATION level** (Qwen text stream +0.054 AUC; Qwen image stream collapses 0.734→0.599 on
  MHC-EN; on the concat/align fuse the swap is a **rotation**: dev hate-recall +0.040 bought with
  non-hate −0.036; error-set overlap **fixes 11 / breaks 12 = net −1** on dev, "no coherent subgroup").
  The **open question this gate measures = per-item PREDICTABILITY**: can a per-item meta-feature model
  pick the winning channel on the disagreement subset better than committing to the globally-better
  single channel? F44 is a population statement and is a **strong prior AGAINST** (no coherent
  subgroup), but it does not measure per-item routability — that is exactly this gate's judged read.

**Honest prior (recorded, does not change the bar):** F44's "fixes 11 / breaks 12, no coherent
subgroup" predicts the MHC-EN disagreement base rate for "Qwen correct" ≈ 0.48, i.e. near-coin-flip,
so the pre-registered expectation is a KILL. The gate is run to convert that prior into a measured,
pre-registered number.

---

## 1. Banked inputs (provenance — all verified present, read-only)

**Feature caches** (`data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen}_{MODEL}.pt`; per-video pooled
`img_feats`/`text_feats`/`labels`/`ids`), MODEL ∈ {`openai_clip-vit-large-patch14-336_HF`,
`Qwen2.5-VL-7B-Instruct_HF`}. Dims: CLIP img 1024 / txt 768; Qwen img 3584 / txt 3584.
Sizes: **HateMM** train 744 / dev 107 (43 hate); **MHC-EN** train 549 / dev 80 (25 hate).

**Head checkpoints** — the enc3s 3-seed encoder-swap arms (job 12850; `exp-encoder-3seed.md`), final
epoch e29, `fusion_mode='align'`, `majority_voting='arithmetic'`, `topk=20`, num_layers=3,
map_dim=proj_dim=1024, dropout=[0.2,0.4,0.1], batch_norm=False. 12 heads snapshotted to
`refine-logs/router_ckpt_snapshot/{DATASET}_{ENC}_s{SEED}_e29.pt`:
- HateMM {CLIP,Qwen} s0/s1/s2 and MHC {CLIP} s0/s1/s2 and MHC {Qwen} s0 pulled from
  `b2:junyi-data/RGCL_video/manual_backup_2026-07-14/Retrieval/…/RAC_video_archive_seeds/…/ckpt/epoch_model_29_*.pt`.
- MHC {Qwen} s1/s2 (reused arcbase 12275/12276; pruned from `logging/Retrieval` before the 07-14 backup)
  pulled from the disk_guard mirror
  `b2:junyi-data/RGCL_video/logs/MHC/Retrieval/MHC/RAC_video_archive_seeds/…/ckpt/epoch_model_29_*.pt`.

**Vote machinery**: `src/model/classifier.py` (`classifier_hateClipper`, align fusion,
`embed = mlp[:-2](x)`), and the deployed vote reproduced from `src/utils/metrics.py`
`compute_metrics_retrieval(majority_voting='arithmetic', topk=20, use_sim=True)` — top-20 rank-weighted
signed-cosine kNN vote, decision = `sigmoid(vote)≥0.5 ⇔ vote≥0` (memory=train). This is exactly the
per-epoch dev/test vote in `src/run_rac.py:673-674,691-692`.

**MACHINERY VALIDATION (pre-computed, mandatory):** the reloaded e29 head + faiss-CPU vote reproduces
the deployed **dev (Val_Retrieval) accuracy** of ALL 12 arms **bit-exact to 4 dp** (the ckpt filename
suffix = the trainlog Val_Retrieval acc): HateMM CLIP 0.7944/0.8131/0.8224, Qwen 0.8505/0.8224/0.8505;
MHC CLIP 0.7375/0.7500/0.7000, Qwen 0.7625/0.7875/0.7750. **12/12 MATCH** ⇒ the regenerated channels
are the faithful deployed channels.

---

## 2. Pre-declared gate design (locked BEFORE the routed read; quoted bars are verbatim from the mandate)

**Channels** (per dataset, per seed): CLIP-arm and Qwen-arm = e29 head → align embed → top-20
rank-weighted signed-cosine kNN vote (memory=train). Channel prediction = `1{vote≥0}`.

**Splits / anti-leak:** router **fit** on the **train** split (LOO kNN votes: memory = train minus
self) disagreement subset, using train GOLD labels (permitted). Router **judged read** on the **dev**
split (memory = full train); dev labels used **only** to score routed vs single, **never** to fit.
**Test rows are never read.** All meta-features are inference-time computable and label-free w.r.t. the
query (neighbour labels are train gold).

**Router task (on the CHANNEL-DISAGREEMENT subset):** predict `y = 1{Qwen arm correct}` (on the
disagreement subset exactly one arm is correct). **Meta-features (per item):** per channel c — signed
vote `vote_c`, margin `|vote_c|`, neighbour hate-fraction `phate_c`, purity `max(phate,1-phate)`,
label entropy, nearest-neighbour cosine `topsim_c`, mean top-20 cosine `meansim_c`, similarity margin
(mean sim of sign-agreeing neighbours − disagreeing), raw image-only sub-vote `vimg_c`, raw text-only
sub-vote `vtxt_c`, fuse-vs-modality agreements, raw text-feature L2 norm, empty-transcript indicator;
cross-channel — `vote_CLIP−vote_Qwen`, `|vote_CLIP|−|vote_Qwen|` (confidence differential), agreement
indicator. **Models:** nonlinear `HistGradientBoostingClassifier` (fixed seed) **and** a linear
`LogisticRegression` baseline (standardized), to isolate the nonlinear/meta increment. Fit on all
train-disagreement items; k=5 CV + permutation for internal checks.

**Routing → dev prediction:** agreement items keep the shared prediction; disagreement items take the
router-selected channel's prediction. (routed and best-single differ ONLY on disagreement items.)

**PRIMARY metric:** routed accuracy vs **best-single-channel** accuracy on the FULL dev split, per seed,
**3-seed mean, paired per seed**. `best_single_acc` (per seed) = `max(dev_acc_CLIP, dev_acc_Qwen)`
(the dev-optimal single channel = hardest baseline).

**Pre-declared BAR (verbatim):** *"routed − best-single ≥ +0.020 on MHC-EN dev (… +0.020 = credible
signal above the ~1.4pt noise) AND routed must not lose on HateMM sanity cell."*

**KILL-SWITCHES (verbatim, pre-declared):**
- **K-R1** = routed−best-single **< +0.020 on MHC-EN dev** (3-seed mean) **OR** CI-low ≤ 0 via
  **bootstrap 1000** (95% percentile CI of the 3-seed-mean gain over dev items).
- **K-R2** = **label-oracle calibration**: plant the true disagreement-resolution `y` as a feature; the
  router must recover it, **accZA ≥ 0.99** — if calibration fails, **MACHINERY_INVALID** (not a
  credited kill).
- **K-R3** = the router's dev gain must survive the **permutation null** (labels of
  disagreement-resolution shuffled on the fit set, **100 seeds**, observed gain **> null-95th**).
- **HateMM sanity:** routed − best-single must **not be negative** (router should learn ~always-Qwen).

---

## 3. RAW RESULTS (script `cross_channel_router_gate.py` sha256 `d4adf545…`; `ROUTER_GATE_OUT.json`)

**Machinery validation:** `MACHINERY_ALL_MATCH = True` — all 12 arms' regenerated dev vote acc match the
deployed anchors bit-exact (§1). **K-R2 oracle calibration (plant true `y`, 5-fold OOF on the
dev-disagreement subset):** MHC (primary) **accZA = 1.000** (GBM) ⇒ the router *does* recover the
routing target when the signal is present ⇒ **machinery VALID, not MACHINERY_INVALID**. (HateMM accZA
0.68 is a tiny-n artifact — the HateMM disagreement subset is only 13–15 items/seed, too few for the
planted-feature recovery to be a meaningful check; MHC governs. Linear accZA MHC 0.907.)

### 3.1 Label-oracle routing headroom — the CEILING (ruled first, B5-style)

A *perfect* per-item router (routes every dev-disagreement item to the channel that IS correct):

| dataset | s0 | s1 | s2 | **3-seed mean gain** |
|---|---|---|---|---|
| **MHC-EN** | +0.1125 | +0.1250 | +0.0875 | **+0.1083** |
| HateMM | +0.0374 | +0.0654 | +0.0467 | +0.0498 |

**Headroom EXISTS and far exceeds the +0.020 bar** (MHC +0.108). The disagreements are resolvable *in
principle*; the question is whether they are *predictable per item*.

### 3.2 PRIMARY pre-registered read — train→dev nonlinear router (GBM)

| dataset | s0 | s1 | s2 | **3-seed mean** | boot 95% CI | best-chan |
|---|---|---|---|---|---|---|
| **MHC-EN** | +0.0000 | +0.0000 | +0.0000 | **+0.0000** | [0.0, 0.0] | Qwen |
| HateMM | +0.0000 | +0.0000 | +0.0000 | +0.0000 | [0.0, 0.0] | Qwen |

Per-seed dev disagreement sizes: MHC 20/23/20 (train-disagree 109/102/92); HateMM 14/15/13.
The router **collapses to the majority/best channel** (routed acc = best-single acc exactly).
**Mechanistic cause (measured, load-bearing):** the CLIP head **memorises train** — LOO train acc
**0.998** vs Qwen **0.800** — so on the **train**-disagreement subset "Qwen correct" is
**0/109, 0/102, 0/92** (degenerate, always-CLIP), the exact **inverse** of the **dev** base rate
(Qwen correct 0.55 / 0.565 / 0.65). The deployable train→dev router therefore has **no dev-transferable
supervision** for the routing decision and degenerates to the prior (majority channel).

### 3.3 Maximally-FAVORABLE realizable ceiling — dev-CV router (in-distribution supervision; supplementary, post-hoc)

Fits the router *within* dev via stratified k-fold OOF on the dev-disagreement subset (peeks dev labels
via CV — an optimistic realizable upper bound; NO test-touch):

| router | dataset | s0 | s1 | s2 | **3-seed mean** | boot 95% CI |
|---|---|---|---|---|---|---|
| GBM | **MHC-EN** | −0.0500 | −0.0750 | −0.0125 | **−0.0458** | [−0.0875, 0.0] |
| GBM | HateMM | −0.0374 | +0.0093 | 0.0000 | −0.0094 | [−0.0312, +0.0125] |
| Linear | **MHC-EN** | −0.0625 | −0.0375 | 0.0000 | **−0.0333** | [−0.0750, +0.0125] |
| Linear | HateMM | −0.0093 | 0.0000 | −0.0187 | −0.0093 | [−0.0312, +0.0093] |

Even with in-distribution supervision, **both** router classes are **NEGATIVE** on MHC — the fitted
router does *worse* than committing to the best single channel; the decision-level meta-features carry
**no per-item routing signal** (they overfit noise). Nonlinearity does not help (GBM ≈ Linear, both < 0).

### 3.4 K-R3 permutation null (on the favorable dev-CV router, MHC)

Shuffle the dev-disagreement resolution labels (100 seeds), redo OOF routing: null p95 = **+0.0042**,
null mean −0.015; **observed −0.0458, p = 0.97 ⇒ does NOT exceed the null** (there is no positive
signal to survive). Linear: observed −0.0333, p = 0.83. **K-R3 not survived.**

## 4. MECHANICAL KILL/PASS (pre-declared rules applied verbatim; PRIMARY router = nonlinear GBM)

| switch | rule | value | fires? |
|---|---|---|---|
| **K-R2** | oracle-calib accZA(MHC) ≥ 0.99 ⇒ machinery valid | **1.000** | machinery **VALID** |
| **K-R1** | routed−best < +0.020 on MHC dev (3-seed) **OR** boot CI-low ≤ 0 | gain **+0.0000**, CI-low **0.0** | **KILL fires** |
| dev-CV ceiling | realizable dev-CV gain < +0.020 OR CI-low ≤ 0 | **−0.0458**, CI-low −0.0875 | **KILL confirmed at ceiling** |
| **K-R3** | dev-CV gain > perm-null p95 | −0.0458 vs +0.0042 | **not survived** |
| HateMM sanity | routed−best not negative | **+0.0000** | **OK** |

## 5. NON-BINDING EXECUTOR LABEL: **KILL** (dead at the read AND the realizable ceiling; machinery valid)

The direction is dead on a pre-registered, machinery-valid basis: the per-item router yields **+0.0000**
at the deployable read and **−0.046** at the maximally-favorable realizable ceiling — below the +0.020
bar and below the permutation null — **despite a genuine +0.108 label-oracle headroom.** The headroom
is real but **per-item unpredictable** from decision-level meta-features (vote margins, neighbour
purity/agreement, per-modality sub-votes, confidence differential, transcript indicators), with or
without nonlinearity, with or without in-distribution supervision. **This is the binding review's call;
the executor asserts no scientific verdict beyond the mechanical rule outputs above.**

**Scientific reading (for the reviewer, non-binding):** this measures F44's population-level
"fixes 11 / breaks 12, no coherent subgroup" **at the per-item predictability level** and confirms it —
the MHC-EN encoder rotation is **not** convertible to a Pareto gain by per-item channel routing. It also
surfaces a second structural obstacle specific to the frozen-artifact setting: the retrieval head's
**train-set memorisation** (CLIP LOO 0.998) makes the routing target **non-transferable** train→dev, so
the deployable router has no valid supervision even before the predictability question. Non-isomorphic
to K9/B5/P1/P2 as argued in §0 (verified); adds **one new measured negative** — per-item cross-channel
routability is absent — and re-confirms the terminus. No headline/family claim created.

## 6. Provenance / hygiene

- **Scripts (committed):** `scripts/analysis/cross_channel_router_gate.py` (sha256 `d4adf545…`,
  recorded in `ROUTER_GATE_OUT.json`). Deterministic (RNG=20260717); CPU-only (`OMP_NUM_THREADS=4`,
  faiss-cpu); ~3 min total. Vote machinery reused faithfully from `src/model/classifier.py` +
  `src/utils/metrics.py` semantics (validated bit-exact §1).
- **Outputs:** `refine-logs/ROUTER_GATE_OUT.json` (full raw numbers + per-ckpt sha256).
- **Head snapshot (NOT committed — 335 MB of `.pt`):** `refine-logs/router_ckpt_snapshot/` (12 e29
  heads). Per-file sha256 recorded in `ROUTER_GATE_OUT.json:ckpt_sha`. Source: enc3s job-12850
  `RAC_video_archive_seeds` e29 heads, pulled from B2 `manual_backup_2026-07-14/Retrieval` (10) and
  the disk_guard mirror `RGCL_video/logs/…` (MHC Qwen s1/s2, reused arcbase 12275/12276, pruned before
  the 07-14 backup). All reproduce their deployed dev (Val_Retrieval) acc bit-exact.
- **ZERO GPU, ZERO test-touch** (train + dev features/labels only; test caches never opened), **no
  Modal**. Gold read = train + dev labels only. No `state/`, prereg, config, or frozen artifact mutated.
  Not pushed.
- **Repo HEAD at run:** recorded via the commit that lands this record.

