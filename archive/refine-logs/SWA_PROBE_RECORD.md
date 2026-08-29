# $0 PROBE — Single-trajectory SWA of RGCL head checkpoints (Family C)

**Author:** swa-probe agent (ZERO GPU / ZERO SLURM / ZERO Modal; CPU head-forward only).
**Date:** 2026-07-20 NZST.
**Target:** the F45 val-selection tax — "dev saturates while test keeps climbing → argmax-dev
undershoots; the small dev split costs ~2 acc pts." Attack it, at $0, with single-trajectory
Stochastic Weight Averaging (SWA, Izmailov et al. UAI 2018): uniform-average the head weights
over a window of one seed's per-epoch checkpoints → one model, one inference cost.
**Scope anchor:** `refine-logs/REDTEAM_EXTERNAL_FAMILIES.md` §3 (Family C), which cites
`src/run_rac.py:764` (per-epoch head checkpoint saved every epoch) as the feasibility hook.
**Script:** `scripts/analysis/swa_probe.py`   **Machine output:** `refine-logs/SWA_PROBE_OUT.json`

> Note on the tasking label: the launch brief called this "Family B". In
> `REDTEAM_EXTERNAL_FAMILIES.md` the SWA family is **Family C (§3)**; Family B (§2) is
> vision-tower PEFT. The mission text unambiguously describes SWA of head checkpoints, so this
> record executes Family C. No vision-tower work was done.

---

## 0. VERDICT UP FRONT

| Dataset | Status | Verdict |
|---|---|---|
| **MHC_zh (ZH)** — the *primary* F45 target | **regenerated** (job 13294, G-repro **bit-exact** vs banked 13150) — see §7 | **KILL** (0/3 seeds; dev is underpowered) |
| **HateMM** (curriculum-LoRA rep2, the ONLY group with live checkpoints) | measured | **KILL** (1/3 seeds promote, and that seed is degenerate) |

> **§7 update (ZH leg completed).** The ZH BLOCKED status was lifted under a one-job authorization:
> the B3/13150 generic-LoRA per-epoch checkpoints were regenerated (job 13294, group
> `RAC_video_lora_swaregen`, run_one byte-identical) and reproduce the banked 13150 dev curve
> **bit-exact** (all 3 seeds × 30 epochs, 4dp diff = 0.0000). SWA on the ZH checkpoints is **KILL
> (0/3 seeds)** — full ZH section in §7. Both datasets now KILL.

**Bottom line: KILL on both datasets.** On **HateMM** (curriculum-LoRA rep2, job 13246, 3 seeds),
every SWA window lands **0.9–6.6 dev-acc points BELOW** the val-selected single-epoch max on the two
seeds with a real selection gap — SWA fails precisely where it is needed and only "passes" on the one
degenerate seed where val-selection already picked the final epoch. On **ZH** (job 13294, the true
F45 target), the dev curve is so **flat and jittery** (all epochs 64–68/78, the "dev saturates"
regime) that SWA windows cluster **at or 1–3 items below** the flat dev ceiling: the best window
matches the ceiling on 2/3 seeds but **no single pre-declared window is uniformly at-ceiling across
seeds**, so cond_A (every window within 0.005 of max) fails 0/3. The dev jitter (2–4 items) is the
same size as the effect being measured, so **dev has no power to promote SWA** — and a test-touch is
not authorized by the dev-only gate.

---

## 1. STEP-0 INVENTORY (checkpoint census — done before any design lock)

`src/run_rac.py:764` saves `{output_path}/ckpt/epoch_model_{epoch}_{select_acc}.pt` every epoch,
where `select_acc` is the **Val_Retrieval dev acc** (the val-selection criterion; `select_acc = acc`
from `compute_metrics_retrieval(..., use_sim=True)` at `run_rac.py:673`). Census of every
`logging/Retrieval/*/*/ckpt/` dir:

| Run group | dataset | seeds | epoch_model files | on disk |
|---|---|---|---|---|
| `RAC_video_lora_curric_rep2` | HateMM | 0,1,2 | **30 each (ep0–29)** | **1.5 G/seed — LIVE** |
| `RAC_video_lora_curric` | HateMM | 0,1,2 | 0 | empty (cleaned) |
| `RAC_video_lora_hm` (generic LoRA, job 13235) | HateMM | 0,1,2 | 0 | empty (cleaned) |
| `RAC_video_lora_hm` (generic LoRA) | MHC (EN) | 0,1,2 | 0 | empty (cleaned) |
| `RAC_video_lora_curric` | **MHC_zh** | 0,1,2 | **0** | **empty (cleaned)** |

A global `find . -name 'epoch_model_*.pt'` returns files under **only** `RAC_video_lora_curric_rep2`.
The originally-named banked runs (ZH job 13150, HateMM generic 13235, HateMM curric 13241, CLIP
floors 13115/12850) all have **empty** ckpt dirs — their per-epoch checkpoints were cleaned, and
their CLIP-floor arms never used this per-epoch-save path. **The only live checkpoints belong to a
different, later run — the curriculum-LoRA DRAW-2 replication (`rep2`, job 13246, HateMM only).**

Consequence for the mission's primary aim:
- **ZH is BLOCKED.** F45's tax was measured on ZH; ZH has zero checkpoints. Regenerating them is
  NOT a $0 op — it needs a GPU head-retrain via SLURM (~20–25 s/run on cached features, but
  submission is the user's, not the probe's). Per the STEP-0 rule this is reported, not run.
- **HateMM is probeable** via `rep2`. HateMM is *not* the dataset where F45 was measured, but it is
  the only place SWA can be tested at $0, and the pre-declared design evaluates "on ≥1 dataset."

`rep2` provenance (`scripts/slurm/enc3seed_lora_curric_rep2.sbatch`): a 2nd independent SFT draw of
the HateMM curriculum head, byte-identical run_one to the banked arms; it is an ACTIVE replication
(rep2-* agents). This probe only **reads** its checkpoints — it does not touch `state/`, the
verdict, or any Test_* line.

**Head architecture (clean-averaging check).** `classifier_hateClipper`, `batch_norm=False`,
`fusion_mode=align`, num_layers=3. State_dict = 12 float32 tensors (img_proj, text_proj, mlp.1/4/7,
output_layer); **NO BatchNorm buffers, no integer buffers.** Dropout is inactive under
`model.eval()`, and the L2-normalize in the forward is parameter-free. ⇒ **uniform weight averaging
is clean** (no batch-statistic recalibration needed). train n=744 (298 pos), dev n=107 (43 pos).

---

## 2. PRE-DECLARED DESIGN (written before computing)

- **SWA arms** (uniform weight-average of per-epoch head state_dicts, 3 windows only, all reported):
  `{post-warmup ep5–29 (n=25), last-10 ep20–29 (n=10), last-5 ep25–29 (n=5)}`.
- **Evaluation** (DEV acc + macro-F1, via the project's own retrieval path
  `retrieve_evaluate_RAC_` + `compute_metrics_retrieval(use_sim=True, majority_voting=arithmetic,
  topk=20)` — the top-20 signed-similarity rank-weighted vote): each SWA arm vs (a) the val-selected
  single-epoch max (post-warmup argmax dev acc, tie-break dev roc) and (b) the final-epoch (ep29)
  checkpoint, on the SAME cached dev features. **Train + dev_seen only; test never read.**
- **BN/normalization caveat:** none — head is pure Linear/ReLU/Dropout + parameter-free normalize
  (verified from the state_dict), so averaging is exact.
- **Decision (pre-declared):** SWA cannot prove a TEST gain from dev alone. **PROMOTE iff, on ≥1
  dataset, EVERY SWA window's dev acc ≥ (max single-epoch post-warmup dev acc − 0.005) AND the SWA
  dev spread across windows < the single-epoch post-warmup dev spread** (SWA does not pay the
  selection tax on dev). We require the criterion to hold on **all 3 seeds** to call the dataset a
  promote (a single-seed pass is seed-noise). Promotion authorizes drafting a prereg whose **single**
  test-touch compares {SWA-last10} vs {val-sel} vs {final} on 3 seeds. Otherwise **KILL**.
- **Free train-side diagnostics:** L2(val-sel, final) per seed, and last-10 dev-acc jitter (max−min).

---

## 3. RESULTS — HateMM curriculum-LoRA rep2 (dev n=107, CPU $0)

Reproduction fidelity: recomputed per-epoch dev acc vs the `select_acc` in each filename differs by
at most **1–2 items/107** (`0.0093`/`0.0187`). This is GPU-vs-CPU float drift on borderline
signed-sim votes (the banked run's default `--device cuda`; this probe forces CPU). It only affects
absolute anchoring to the banked GPU numbers — **every arm here (single-epoch AND SWA) uses the
identical CPU path, so the within-probe comparison is exact and apples-to-apples.**

| seed | val-sel ep | val-sel dev acc (=max single) | final ep29 dev acc | SWA ep5–29 | SWA ep20–29 | SWA ep25–29 | cond_A | cond_B | seed verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 18 | **0.8598** | 0.8318 | 0.8505 | 0.8411 | 0.8318 | ✗ | ✓ | **KILL** |
| 1 | 16 | **0.8692** | 0.8318 | 0.8037 | 0.8318 | 0.8131 | ✗ | ✓ | **KILL** |
| 2 | 29 | 0.8411 (=final) | 0.8411 | 0.8505 | 0.8411 | 0.8411 | ✓ | ✓ | PROMOTE* |

`cond_A` = all SWA windows ≥ (max single − 0.005). `cond_B` = SWA spread < single-epoch spread.
macro-F1 tracks acc throughout (e.g. seed0 val-sel mF1 0.8566 vs SWA-ep5–29 0.8466; seed1 val-sel
0.8658 vs SWA-ep5–29 0.7951). Full per-epoch curves in `SWA_PROBE_OUT.json`.

**\*seed2 is degenerate:** its val-selected epoch IS the final epoch (dev argmax landed at ep29), so
there is **no selection tax to pay** and L2(val-sel, final)=0.0. Its "PROMOTE" is vacuous.

**Dev-curve shape (why SWA loses):**
- seed0: dev peaks ep15 (0.8598), late plateau ~0.832; peak→final drop **+0.028**.
- seed1: dev peaks ep16 (0.8692), late plateau ~0.832; peak→final drop **+0.037**.
- seed2: dev rises to a ~0.83–0.84 plateau by ~ep14 and stays there; val-sel=final.

The HateMM dev optimum is a **mid-training peak**, not a late/flat plateau. Averaging weights across
ep20–29 lands on the lower late plateau; averaging across ep5–29 additionally drags in the weak
early epochs (0.76–0.79) and is worst of all (seed1 → 0.8037, *below even the final epoch*). SWA
therefore **cannot recover the mid-training dev peak** that val-selection captures.

**Free diagnostics (dev-selection IS noisy — but SWA is the wrong fix):**
| seed | L2(val-sel, final) | last-10 dev-acc jitter |
|---|---|---|
| 0 | 4.99 | 0.028 (3/107) |
| 1 | 5.91 | 0.028 (3/107) |
| 2 | 0.00 | 0.019 (2/107) |

Jitter of 2–3 items across the last 10 epochs confirms genuine dev-selection noise on n=107, and on
seeds 0/1 the val-selected and final checkpoints are ~5–6 apart in weight space (different models).
So the *premise* (noisy val-selection) holds; the *remedy* (SWA) does not, because the noise sits on
top of a mid-peaked dev surface rather than a flat one.

---

## 4. DECISION

**KILL.** Per the pre-declared criterion: `cond_A` fails on 2/3 seeds (SWA dev < max single − 0.005),
and the only passing seed is the degenerate no-tax case. **1/3 (and the passing 1 vacuous) → the
dataset does not promote.** No test-touch is authorized; do not draft the SWA prereg on HateMM.

**Honesty caveats (do not over-read this KILL):**
1. **Dev-only.** By the no-test-read rule this probe never observed HateMM *test*. It is possible SWA
   helps HateMM test if HateMM test (like ZH in F45) climbs late — but the pre-declared gate is that
   SWA must first not lose *dev* vs the val-sel max, as the precondition to spend the single
   test-touch. It loses dev, so the disciplined action is to not spend it.
2. **Wrong dataset for the mechanism.** F45's tax is a **ZH** phenomenon (dev saturates ~ep19 while
   *test* climbs to ep29). HateMM's dev curve has the *opposite* shape (mid-peak, late-decline),
   which is structurally adverse to trajectory averaging. This probe is therefore **not** a test of
   SWA against F45's actual mechanism — it is a test against the only checkpoints that exist.
3. The clean way to actually test Family C against F45 is on **ZH** — this was **completed in §7**
   under a one-job authorization (regen job 13294, G-repro bit-exact). ZH SWA is **KILL (0/3)**, but
   as a *dev-underpowered* KILL: the 78-item dev is too noisy to discriminate SWA from val-selection.
   Family C is now measured and killed on both datasets. *(This §4 caveat, written for the
   HateMM-only leg, is superseded by §7 on the ZH point.)*

---

## 5. GOVERNANCE FLAG (carried verbatim, per the launch brief)

> The standing veto bans CROSS-SEED ensembles; single-trajectory weight averaging is one model at
> inference from one seed — plain-text reading says NOT covered, but the user must micro-rule before
> any SWA number enters a claims table. The probe itself is measurement, not a claim.

This record reports measurement only. No SWA number here is entered into any claims table, paper
draft, or `state/` artifact. Both datasets KILL (HateMM §3, ZH §7), so the question is moot for now,
but the flag stands verbatim for any future SWA re-run.

---

## 6. PROVENANCE / DISCIPLINE

- Zero GPU, zero SLURM, zero Modal, zero downloads. CPU head-forward over cached `.pt` features only.
- No test-set reads: only `data/CLIP_Embedding/HateMM/{train,dev_seen}_...rep2_HF.pt`;
  `test_seen_*.pt` and all Test_* trainlog lines were never opened for selection or evaluation.
- `autoresearch/goal_mllm_plus3/state/` NOT modified.
- Reproduction path = the project's own `retrieve_evaluate_RAC_` + `compute_metrics_retrieval`
  (`use_sim=True`, arithmetic top-20 vote), mirroring `run_rac.py:659–676` and the rep2 sbatch args
  (`fusion_mode=align`, `topk=20`, `majority_voting=arithmetic`, `metric=cos`, `batch_norm=False`,
  `warmup=5`, `Faiss_GPU=False`).
- Deliverables (HateMM leg): `scripts/analysis/swa_probe.py`, `refine-logs/SWA_PROBE_RECORD.md`,
  `refine-logs/SWA_PROBE_OUT.json`. Local commit only; never pushed.
- Deliverables (ZH leg, §7): `scripts/slurm/swaregen_zh.sbatch`, `refine-logs/SWA_PROBE_ZH_OUT.json`,
  and the `zh` config path of `scripts/analysis/swa_probe.py`. Local commit only; never pushed.
- ZH no-test-read: only `data/CLIP_Embedding/MHC_zh/{train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`
  and the **Val_** lines of the banked 13150 trainlogs (for G-repro) were read; the new job-13294
  trainlogs' Test_* lines and every `test_seen_*.pt` were never opened.

---

## 7. ZH LEG — checkpoint regeneration + SWA on the TRUE F45 target (job 13294)

**Authorization.** A one-job override of the STEP-0 no-submission rule was granted specifically to
complete Family C on ZH, where F45's val-selection tax was actually measured. Scope: regenerate the
per-epoch head checkpoints only; the no-test-read rule stands absolutely.

**Regeneration.** `scripts/slurm/swaregen_zh.sbatch` (job **13294**, COMPLETED, exit 0) clones the
B3/13150 arm — model `Qwen2.5-VL-7B-Instruct-LoRA_HF`, MHC_zh, seeds 0/1/2, `run_one` byte-identical,
`--force False` — changing **only** `--group_name` to a fresh `RAC_video_lora_swaregen` (collision
pre-checked: neither the output group nor matching trainlogs existed). The training code path is
unchanged since before 13150 (last touch of `run_rac.py`/loss/eval/metrics/consensus/dataloader =
commit `5fa01ab`, 2026-07-13, predating 13150's 2026-07-14 run), so the run is deterministic.
Result: 3 seeds × 30 per-epoch checkpoints (90 files) on disk. ZH dev n = **78** (F45's "78-item dev").

**G-repro sanity — PASS (bit-exact).** Regenerated per-epoch DEV curve (ckpt filename `select_acc`)
vs the banked 13150 `Val_Retrieval` dev acc (parsed from the trainlog **Val_** lines only; no Test_
line read): **max |regen − banked| dev acc (4dp) = 0.0000 on all 3 seeds, all 30 epochs** (0/30
epochs differ per seed). The regeneration is a perfect reproduction of the banked run; the
checkpoints are trustworthy.

**SWA results (§2 design UNCHANGED; dev n=78, CPU $0).** Recomputed-vs-filename dev-acc drift is
≤1 item/78 (seeds 0/2) and 3 items/78 (seed 1) — GPU-vs-CPU float drift on borderline signed-sim
votes (note this is *only* the CPU recompute; the GPU-vs-GPU G-repro above is bit-exact). Every SWA
arm uses the identical CPU path, so the SWA-vs-single comparison is internally exact.

| seed | val-sel ep | val-sel dev acc (=max single) | final ep29 | SWA ep5–29 | SWA ep20–29 | SWA ep25–29 | best SWA | cond_A | cond_B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 20 | **0.8718** (68/78) | 0.8462 | **0.8718** | 0.8590 | 0.8462 | 0.8718 (=ceiling) | ✗ | ✓ | **KILL** |
| 1 | 26 | **0.8718** | 0.8590 | 0.8590 | **0.8718** | 0.8590 | 0.8718 (=ceiling) | ✗ | ✓ | **KILL** |
| 2 | 19 | **0.8718** | 0.8462 | 0.8590 | 0.8333 | 0.8333 | 0.8590 (<ceiling) | ✗ | ✓ | **KILL** |

**DATASET VERDICT (ZH): KILL (0/3 seeds promote).** cond_A (every SWA window ≥ max single − 0.005)
fails on all three seeds; cond_B (SWA spread < single-epoch spread) passes on all three.

**Why it fails — and why this is a dev-underpowered KILL, not a "SWA hurts" KILL.** The ZH dev curve
is **flat and jittery**: every post-warmup epoch sits in 0.821–0.872 (64–68/78), with the ceiling
0.8718 touched at scattered epochs and no trend — the textbook F45 "dev saturates" regime. SWA
windows therefore cluster *at or 1–3 items below* the flat ceiling: on 2/3 seeds the **best** SWA
window exactly **matches** the dev ceiling (seed0 ep5–29, seed1 ep20–29), and cond_B passes (SWA is
flatter than the single-epoch curve). What kills it is that **no single pre-declared window is
uniformly at-ceiling across seeds** (ep5–29 wins seed0, ep20–29 wins seed1, neither wins seed2), so
some window always drops 1–3 items and cond_A fails. Critically, the post-warmup dev **spread is
0.038–0.051 (3–4 items) and the last-10 jitter is 2–4 items — the same magnitude as the SWA-vs-max
differences.** The 78-item dev simply **has no power to discriminate** SWA from val-selection: this is
the very uninformativeness F45 blames for ZH losing its val-selected pass. Free diagnostics confirm a
real (non-degenerate) selection gap on all three seeds — val-sel epoch ≠ final on every seed, and
L2(val-sel, final) = 4.93 / 1.77 / 4.37 — so unlike HateMM seed2 there is always a tax being paid;
SWA just cannot be *shown* on dev to fix it.

**Decision.** **KILL.** The pre-declared dev-only gate is not met (cond_A 0/3), so **no test-touch is
authorized** and no ZH SWA prereg is drafted. The honest characterization is *dev-underpowered*: on
its true target, SWA neither clearly helps nor clearly hurts dev, and the dev split is too small/noisy
to license spending the single ZH test-touch on it. Family C is now **measured and killed on both
datasets** (HateMM decisively; ZH on an underpowered dev), closing the Family-C opening that
`REDTEAM_EXTERNAL_FAMILIES.md` §3 left as the one $0-testable shot at the F45 tax.

**Governance (carried verbatim, still binding).** Single-trajectory weight averaging is one model at
inference from one seed → plain-text NOT the cross-seed-ensemble ban, but a user micro-ruling is
required before any SWA number enters a claims table. Moot given the KILL, but it stands. No SWA
number here enters any claims table, paper draft, or `state/` artifact; `state/` was not modified.
