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
| **MHC_zh (ZH)** — the *primary* F45 target | **BLOCKED** — no per-epoch checkpoints on disk (ckpt dir empty) | cannot probe |
| **HateMM** (curriculum-LoRA rep2, the ONLY group with live checkpoints) | measured | **KILL** (1/3 seeds promote, and that seed is degenerate) |

**Bottom line: KILL on the available evidence; the primary target (ZH) is BLOCKED.**
On the one dataset whose checkpoints survive (HateMM curriculum-LoRA rep2, job 13246, 3 seeds),
every SWA window lands **0.9–6.6 dev-acc points BELOW** the val-selected single-epoch max on the
two seeds that actually have a selection gap. SWA fails *precisely where it is needed* and only
"passes" on the one seed where val-selection already picked the final epoch (nothing to fix). The
mechanistic reason: HateMM's dev curve **peaks mid-training (ep14–18) and settles onto a lower
late plateau**, so trajectory weight-averaging pulls the model *below* the dev peak. SWA would only
pay off if the dev optimum were a late/flat plateau (the ZH F45 shape) — and **ZH is exactly the
dataset with no checkpoints on disk.**

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
3. The clean way to actually test Family C against F45 is on **ZH**, which is **BLOCKED** pending a
   (user-submitted, GPU/SLURM) regeneration of the ZH curriculum-LoRA per-epoch checkpoints
   (~20–25 s/run on cached features). Until then Family C is **unfalsified on its true target** and
   **killed on the one dataset it could be measured on.**

---

## 5. GOVERNANCE FLAG (carried verbatim, per the launch brief)

> The standing veto bans CROSS-SEED ensembles; single-trajectory weight averaging is one model at
> inference from one seed — plain-text reading says NOT covered, but the user must micro-rule before
> any SWA number enters a claims table. The probe itself is measurement, not a claim.

This record reports measurement only. No SWA number here is entered into any claims table, paper
draft, or `state/` artifact. Even though the HateMM verdict is KILL (so the question is moot for
now), the flag stands for any future ZH-unblocked re-run.

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
- Deliverables: `scripts/analysis/swa_probe.py`, `refine-logs/SWA_PROBE_RECORD.md`,
  `refine-logs/SWA_PROBE_OUT.json`. Local commit only; never pushed.
