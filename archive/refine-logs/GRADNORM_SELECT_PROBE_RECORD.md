# $0 PROBE — Validation-free checkpoint selection by head-gradient norm (Stage-A, dev-side)

**Author:** gradnorm-select probe agent (ZERO GPU / ZERO SLURM / ZERO Modal; CPU
forward-backward on the tiny head only). **Date:** 2026-07-25 NZST.
**Candidate:** F68 ledger P2 / `refine-logs/LITSURVEY_NOVEL_MECHANISMS.md` §C2 —
"No Validation, No Problem: Predicting Model Performance from a Single Gradient"
(**arXiv 2601.16874**, Jan 2026). Select the inference epoch by the head-gradient
Frobenius norm computed with **no dev/test data**, as a data-free remedy for the F45
val-selection tax (dev saturates ~ep19 while test climbs to ep29 → argmax-dev
undershoots; the 78-item ZH dev costs ~2 acc pts).
**Machinery template:** `scripts/analysis/swa_probe.py` (SWA probe, F62/F62b) — its
loader / head-forward / dev-eval / checkpoint census are reused verbatim; the operator
is swapped from *weight-averaging* to *argmin-gradient-norm SELECTION*.
**Script:** `scripts/analysis/gradnorm_select_probe.py`
**Machine output:** `refine-logs/GRADNORM_SELECT_PROBE_OUT.json` (fb16 machinery leg),
`..._HATEMM_OUT.json`, `..._ZH_OUT.json` (both BLOCKED-census stubs).

---

## 0. VERDICT UP FRONT

| Dataset / arm | Stage-A role | Census | Verdict |
|---|---|---|---|
| **MHC_zh** swaregen (job 13294) — F45 **PRIMARY** target | promotable | **0/0/0 ckpts** (pruned to B2 2026-07-20) | **BLOCKED** |
| **HateMM** curric-LoRA rep2 (job 13246) — F45 secondary target | promotable | **0/0/0 ckpts** (pruned to B2 2026-07-20) | **BLOCKED** |
| **HateMM** frame-16 (`RAC_video_fb16`, F67 KILLED arm) — only live ckpts | machinery only (NOT promotable) | 30/30/30 live | **MECHANISM REFUTED** (degenerate pass) |

**Bottom line.** Both promotable F45 targets are **BLOCKED**: their per-epoch head
checkpoints were pruned off local disk by `disk_guard` on 2026-07-20 (after sha1-verified
B2 backup), so the STEP-0 "if ZH ckpts are missing, STOP and report BLOCKED" rule fires.
The **only** group with live per-epoch head checkpoints is the killed frame-16 arm
(`RAC_video_fb16`); a $0 dev-only **machinery / mechanism-transfer smoke test** was run
there to answer the launch brief's core question — *is this selection rule well-behaved
on our banked per-epoch head checkpoints?* **It is not.** The paper's whole premise —
that the head-gradient norm is **strongly NEGATIVELY** correlated with accuracy
(paper: Spearman ρ ≈ −0.85..−0.98, so argmin(‖g‖) picks a high-accuracy epoch) —
**inverts on our tiny head**: our Spearman(‖g‖-norm, dev-acc) is **+0.61 / +0.72 / +0.62**
(wrong sign, 3/3 seeds). The head-scale-normalized gradient rises **monotonically** across
training (0.003 → 0.010) while accuracy also rises, so an unrestricted argmin(‖g‖) lands
at the **earliest, lowest-accuracy** epoch (ep5, dev acc 0.757–0.776). The pre-declared
tail-window rule "passes" the three Stage-A conditions **only degenerately** — argmin over
a flat ep20-29 tail always lands at the window's left edge (ep20) whose dev acc happens to
sit within 0.02 of val-sel. This is a **boundary artifact, not the mechanism working.**

**Consequence.** Even after a (cheap, CPU-only) B2 restore of the ZH/HateMM-curric
checkpoints, the honest prior is that grad-norm selection would exhibit the same wrong-sign
correlation and fail to recover the F45 tax. The recommendation is therefore **do NOT spend
a GPU regen or a test-touch** on this candidate on the strength of the machinery evidence;
if the team lead still wants the promotable measurement, the **$0-CPU B2 restore path**
(below) is the way to get it without new GPU.

---

## 1. STEP-0 CHECKPOINT CENSUS (done before any design lock)

`src/run_rac.py:764` saves `{output_path}/ckpt/epoch_model_{epoch}_{select_acc}.pt` every
epoch (`select_acc` = the Val_Retrieval dev acc used for val-selection). Census of the
mission's two named groups **and** a global `find . -name 'epoch_model_*.pt'`:

| Run group | dataset | job | seeds live (want 30 each) | on disk |
|---|---|---|---|---|
| `RAC_video_lora_swaregen` (**ZH PRIMARY**) | MHC_zh | 13294 | **0 / 0 / 0** | **empty — pruned to B2** |
| `RAC_video_lora_curric_rep2` (HateMM secondary) | HateMM | 13246 | **0 / 0 / 0** | **empty — pruned to B2** |
| `RAC_video_fb16` (frame-16, F67 KILLED) | HateMM | 13352/13353 | **30 / 30 / 30** | **1.2 G/seed — LIVE** |

**What changed since the SWA probe (2026-07-20).** SWA_PROBE_RECORD §1 recorded the
curric-rep2 and (post-regen) swaregen ckpts as LIVE. Between then and now, `disk_guard`
pushed both groups to Backblaze B2, sha1-verified the copies, and **pruned local**:
- ZH swaregen: `[2026-07-20 05:39–05:40] [disk_guard] PRUNED local (verified on B2): …/MHC_zh/RAC_video_lora_swaregen/…/epoch_model_*.pt (freed 40.03MB)` — 108 prune lines.
- HateMM curric-rep2: `[2026-07-20 04:14] … PRUNED local (verified on B2): …/RAC_video_lora_curric_rep2/…` — 181 prune lines.

**BLOCKED is liftable at $0-CPU (no GPU).** Both groups are on `b2:junyi-data/RGCL_video/logs/…`
with matching sha1; `rclone` is installed. A `rclone copy` of the two groups' `ckpt/`
dirs (~3.6 GB ZH + ~3.6 GB HateMM) restores every per-epoch checkpoint **without any GPU
regen** — this is the cheap path to the promotable measurement if the team lead authorizes
it (the probe did **not** do it: the STEP-0 rule says STOP, and disk_guard would re-prune
under quota pressure without a guard pause). This is strictly cheaper than the SWA leg's
GPU regen (job 13294).

**Features present for all three arms** (train + dev_seen only were opened here):
`train/dev_seen_Qwen2.5-VL-7B-Instruct_HF-16f.pt` (fb16), `…-LoRA-curric-rep2_HF.pt`,
`MHC_zh/…-LoRA_HF.pt`. HateMM train n=744 (298 pos), dev n=107 (43 pos); ZH dev n=78.

**Head (clean-gradient check).** `classifier_hateClipper`, `batch_norm=False`,
`fusion_mode=align`, `num_layers=3` → state_dict = 12 float32 tensors (img_proj.0,
text_proj.0, mlp.1/4/7, output_layer — weight+bias each); **no BatchNorm buffers, no
integer buffers**; forward is Linear/ReLU/Dropout + parameter-free L2-normalize. Under
`model.eval()` dropout is inactive ⇒ the gradient of the loss w.r.t. the weights is a
**deterministic** function of (weights, batch) — required for the batch-swap stability test.

---

## 2. PRE-DECLARED DESIGN (written before computing selection results)

**Paper method (as characterised in litsurvey §C2 + arXiv 2601.16874v1 abstract).** Per
banked per-epoch head checkpoint θ_e: one forward-backward of the **classification** loss
on a fixed batch of detached train features through the head; record ‖g‖_F = ‖dL/dW‖_F,
scale-normalize, and **select the checkpoint with the minimum head gradient in a short
TAIL window** (v1 abstract, verbatim: *"Selecting the checkpoint with the minimum head
gradient in a short tail window closes most of the gap to the oracle."*). Paper offers
head-scale (CNN) or feature-scale (Transformer) normalization; the **public arXiv version
DEFERS the exact algorithm** (*"full algorithmic details … will appear in a forthcoming
paper"*), so the rule below is pre-declared from that characterisation. Paper's reported
correlation: ‖g‖ **strongly negative** with Top-1 (ρ ≈ −0.85..−0.98) and positive with loss.

**Locked design:**
1. **Selection statistic** `S(e) = ‖∇_W L_BCE(B; θ_e)‖_F / (‖W_e‖_F + 1e-12)`, where
   - `L_BCE` = plain `nn.BCEWithLogitsLoss` (pos_weight=None) on the head's classification
     logit vs the binary label — i.e. the **classification (BCE) term of the training
     hybrid loss, alone** (`src/model/loss.py:545–554`). **Why BCE-only, not the full
     hybrid** (pre-declared): (a) the paper computes the *classification* gradient through
     the head; (b) all three arms trained the BCE term with `pos_weight=None`, so plain BCE
     is faithful; (c) the hybrid's triplet term needs memory-bank retrieval + pseudo-gold /
     hard-negative **mining**, which is bank- and batch-composition-dependent — not "a
     single gradient through the head" — and would inject non-classification structure;
     (d) F45's tax is a classification-accuracy phenomenon that BCE directly targets. The
     triplet term is excluded **by design**.
   - `W_e` = concatenation of ALL 12 trainable head tensors; ‖·‖_F over the concatenation.
     **Head-scale (relative-gradient) normalization** is the ONE scale-norm the paper offers
     for small-head instability; scale-invariant across epochs (weights grow during training).
   - `model.eval()` (dropout OFF) ⇒ S(e) deterministic.
2. **Selection rule (ONE rule, no menu):** `argmin_e S(e)` over the **tail window ep20-29**
   (the paper's "short tail window"; 30-epoch run → last-10 = short tail). **This deviates
   from the launch brief's tentative "argmin over epochs ≥ warmup 5"** and follows the
   paper's tail-window rule instead; warmup=5 is subsumed (tail starts at 20). The full-range
   ep5-29 argmin is reported as a **diagnostic only** (to expose whether the tail restriction
   is load-bearing), never as the rule.
3. **Two fixed probe batches** (stability, bar i): one seed-0 permutation of the train
   indices; **batch A = perm[0:64]** (the PRIMARY, used for the reported selection),
   **batch B = perm[64:128]** (disjoint; the pre-declared SWAP). Both fixed across all
   epochs and seeds (features are seed-independent, so the batch is identical across seeds).
4. **Reported** per seed: the full 30-epoch S(e) curve; grad-sel epoch (+dev acc/mF1) vs
   val-sel epoch (post-warmup argmax dev acc, tie-break dev roc; +dev metrics) vs final
   epoch (+dev metrics); Spearman(S, dev_acc) over all 30 and over the tail.
5. **Stage-A promote bar (dev-only; a probe cannot prove a test gain).** PROMOTABLE on a
   dataset iff, on ≥1 dataset: **(i)** STABLE — grad-sel epoch shifts ≤2 on the A→B batch
   swap; **(ii)** NON-VACUOUS — grad-sel ≠ val-sel on ≥2/3 seeds; **(iii)** NOT-BROKEN —
   grad-sel dev acc ≥ val-sel dev acc − 0.02. **Mechanism gate (pre-declared amendment,
   §4):** a PROMOTE additionally requires the paper's mechanism to hold, i.e. median
   Spearman(S, dev_acc) **< 0**; otherwise any "pass" is a flat-tail boundary artifact and
   the verdict is KILL. PROMOTE would authorise a prereg whose **single** test-touch compares
   {grad-sel} vs {val-sel} vs {final} per seed — that prereg is **not** written or run here.

---

## 3. RESULTS — fb16 machinery / mechanism-transfer leg (HateMM dev n=107, CPU $0)

**Reproduction fidelity.** Recomputed per-epoch dev acc vs the filename `select_acc`:
max |diff| = **0.0000** (seed0), **0.0093** (seeds 1,2) = ≤1 item/107 (GPU-vs-CPU float
drift on borderline signed-sim votes, as in SWA §3). Inference is faithful; every arm uses
the identical CPU path so the within-probe comparison is exact.

| seed | val-sel ep (dev acc / mF1) | grad-sel ep A (dev acc / mF1) | grad-sel ep B (swap) | final ep29 (dev acc / mF1) | Spearman all30 | Spearman tail | (i) stable | (ii) ≠val-sel | (iii) not-broken |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **23** (0.8411 / 0.8341) | 20 (0.8318 / 0.8250) | 20 | 29 (0.8131 / 0.8040) | **+0.606** | −0.548 | ✓ | ✓ | ✓ |
| 1 | **28** (0.8318 / 0.8236) | 20 (0.8131 / 0.8040) | 21 | 29 (0.8131 / 0.8056) | **+0.723** | +0.548 | ✓ | ✓ | ✓ (margin +0.0013) |
| 2 | **20** (0.8224 / 0.8146) | 20 (0.8224 / 0.8146) | 20 | 29 (0.8131 / 0.8040) | **+0.623** | +0.101 | ✓ | ✗ (=val-sel) | ✓ |

**Stage-A conditions:** (i) stable **True** (grad-sel shifts 0/1/0 ≤2), (ii) non-vacuous
**2/3 ≥ 2 True**, (iii) not-broken **True** → the three literal conditions **PASS**.

**But the S(e) curve is monotone-increasing, positively tracking accuracy** (representative
seed 0; full curves in OUT.json — all three seeds identical in shape):

```
ep :  S_A     dev_acc            ep :  S_A     dev_acc
 0 : 0.0033  0.7850              15 : 0.0050  0.8037
 5 : 0.0034  0.7757  <-fullrange 20*: 0.0061  0.8318  <-grad-sel A (tail left edge)
10 : 0.0042  0.7944     argmin   23*: 0.0071  0.8411  <-val-sel (dev peak)
                                 29*: 0.0098  0.8131  <-final   (* = tail window ep20-29)
```

S rises ~3× over training (0.003 → 0.010) and so does accuracy — hence the **positive**
Spearman. The min of S is always at the **earliest** epochs; the full-range (no tail) argmin
lands at **ep5 on all 3 seeds, dev acc 0.7757 / 0.7570 / 0.7570** (near-worst). The
tail-window argmin lands at **ep20 = the window's left edge on all 3 seeds** simply because
S is monotone within the tail — not because ep20 is a flat minimum.

---

## 4. MECHANISM REFUTATION (the decisive finding)

The paper's method rests on **‖g‖ being NEGATIVELY correlated with accuracy** (ρ ≈
−0.85..−0.98), so that `argmin(‖g‖)` selects a high-accuracy checkpoint. On our tiny
head this **inverts**:

- **Spearman(S, dev_acc) over all 30 epochs = +0.606 / +0.723 / +0.623** (median **+0.623**),
  wrong sign, 3/3 seeds. Over the tail the correlation is weak and sign-unstable
  (−0.548 / +0.548 / +0.101). This is exactly the **small-head instability the paper flags**,
  manifesting here as a full sign flip: for a tiny head on frozen features, the normalized
  BCE gradient grows as the head specialises and does **not** dip at the best-generalising
  epoch.
- Consequently `argmin(‖g‖)` is an **anti-selector**: unrestricted it picks the worst
  (earliest) epochs; the pre-declared tail window merely bounds the damage by forcing the
  pick to the window edge (ep20), whose dev acc coincidentally sits within 0.02 of val-sel
  **because the ep20-29 tail is flat (0.79–0.84 jitter)**, not because the rule found a good
  checkpoint.

**Therefore the three Stage-A conditions pass DEGENERATELY.** Per the pre-declared mechanism
gate (§2.5), a positive median Spearman ⇒ the "pass" is a boundary artifact ⇒ the machinery
verdict is **MECHANISM REFUTED**, and on a promotable dataset this would be a **KILL**, not a
PROMOTE, regardless of the flat-tail (iii) pass. The candidate does not transfer to our
architecture.

---

## 5. DECISION

- **MHC_zh (F45 PRIMARY): BLOCKED.** No live per-epoch checkpoints (pruned to B2). Per
  STEP-0, STOP. No test-touch, no prereg. Liftable via the **$0-CPU B2 restore** (§1), but
  see the mechanism caveat below before spending anything.
- **HateMM curric-rep2 (F45 secondary): BLOCKED.** Same reason.
- **HateMM fb16 (machinery, only live group): MECHANISM REFUTED / degenerate pass.** Not a
  promote (wrong arm — killed frame-16 — and no F45 tax measured here); its DEV was read,
  its TEST never. This leg exists to answer *is the rule well-behaved on our banked head
  checkpoints* — the answer is **no** (wrong-sign correlation).

**Recommendation to the team lead.** The machinery evidence is a strong (not dispositive)
prior that grad-norm selection will **not** recover the F45 tax on ZH/HateMM-curric either —
the failure is a property of the tiny head's gradient geometry, which the promotable arms
share. If a promotable measurement is still wanted, prefer the **$0-CPU B2 restore** of the
checkpoints over any GPU regen, then re-run this exact script on the `zh` / `hatemm_curric`
legs (they will execute the full pipeline the moment the ckpts are present). Do **not**
authorise a test-touch on this candidate on the current evidence.

---

## 6. GOVERNANCE NOTE (carried per the launch brief)

This is a **SELECTION** rule (one existing checkpoint at inference), **not** an ensemble or
a weight-average — so it does **not** collide with the standing cross-seed-ensemble veto, and
it is **not subsumed by the F62/F62b SWA kill** (different operator class). Litsurvey §C2(c),
verbatim: *"SWA **averages** per-epoch head weights and lost dev points on HateMM's
mid-peak-dev seeds. This candidate does **not average** — it **selects** one existing
checkpoint by a validation-free score. Different object; the SWA failure mode (averaging
drags a mid-peak optimum) does not apply. Non-isomorphic."* The F62 ban-scope is
weight-averaging; grad-norm selection is a distinct object and was measured on its own merits
here. No number in this record enters any claims table, paper draft, or `state/` artifact.

---

## 7. PROVENANCE / DISCIPLINE

- Zero GPU, zero SLURM, zero Modal, zero downloads. CPU forward-backward over the 12-tensor
  head + cached `.pt` features only. Runtime ≈ 32 s wall for the full fb16 leg (90 ckpts ×
  [dev retrieval-eval + 2 grad passes]).
- **No test reads.** `load_feats_from_CLIP` is deliberately **bypassed** (it also loads
  `test_seen`); this script `torch.load`s **only** `train_*.pt` and `dev_seen_*.pt`. No
  `test_seen*.pt` cache and no `Test_*` trainlog line was opened for selection or evaluation.
- `autoresearch/goal_mllm_plus3/state/` **not modified**.
- Dev metrics via the project's own `retrieve_evaluate_RAC_` + `compute_metrics_retrieval`
  (`use_sim=True`, arithmetic top-20 vote), mirroring `run_rac.py:659–676` and the fb16 sbatch
  (`fusion_mode=align`, `topk=20`, `majority_voting=arithmetic`, `metric=cos`,
  `batch_norm=False`, `warmup=5`, `hybrid_loss=True`, `ce_weight=0.5` default,
  `pos_weight_value=None` default).
- Deliverables: `scripts/analysis/gradnorm_select_probe.py`,
  `refine-logs/GRADNORM_SELECT_PROBE_RECORD.md`, `refine-logs/GRADNORM_SELECT_PROBE_OUT.json`
  (+ `_HATEMM_OUT.json`, `_ZH_OUT.json` BLOCKED stubs). Local commit only; never pushed.
