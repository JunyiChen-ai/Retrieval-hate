# RE-AUDIT — NCA / soft-kNN head loss (τ=0.1) — FREEZE

**Date:** 2026-08-17 NZST. **Cost:** local GPU only, zero API, zero cloud.
**Status at freeze:** design, arms, seed range, read-outs and decision rule below are final.
No run in the frozen seed range has been executed. This file is committed **before** the first
frozen-range run.

---

## 1. What is being re-audited

`refine-logs/REAUDIT_RESULT.md` (commit `c021009`) lists the NCA / soft-kNN head loss as the
**standing next candidate, not run** — the one member of the 2026-07-25 NCA family that survived its
own kill switch and was then closed on a noise-band argument rather than on a powered measurement.

**Original verdict, verbatim** (`refine-logs/NCA_VERDICT_REVIEW.md` §5 MARGINAL note and §6 family
verdict line):

> **MARGINAL note (§7.2, B3 §2.2 precedent):** A1a NCA τ=0.1 × ZH val-sel is a clean 3/3-positive acc
> result (mean **+0.0112** acc / **+0.0113** mF1) that survives KS-arm-dead but sits **below** the
> +0.030 FORMAL bar AND **below** the ±0.014 head-seed noise band (§2.3) — a within-noise clean-sign
> positive.

> the sole survivor, A1a NCA τ=0.1 × ZH, survives KS-arm-dead on a within-noise clean-sign val-sel
> positive (+0.0112 acc / +0.0113 mF1) but sits below the FORMAL bar and below the ±0.014 head-seed
> noise band — measured-not-promoted limbo, D7-DEAD. The loss↔inference-mismatch axis is CLOSED

The kill is of exactly the type `REAUDIT_RESULT.md` showed to be unreliable: a 3-seed read judged
against a ±0.014 band, when a 3-seed test has 56.5 % miss rate against a true +0.0145 effect.

## 2. The candidate, exactly

**Loss.** `src/model/loss.py::_nca_head_loss` (function unchanged since the 2026-07-25 freeze; see
§6). With `--head_loss nca`, the head's contrastive term is replaced by the deployed-vote-aligned
soft-kNN surrogate

```
L_NCA = - mean_i  log P_i ,
P_i = sum_{j in bank, y_j = y_i, id_j != id_i} exp(cos(f_i, b_j)/tau)
      / sum_{j in bank,            id_j != id_i} exp(cos(f_i, b_j)/tau)
```

over a **detached, per-epoch, whole-train memory bank** rebuilt in the current embedding space by
`src/run_rac.py::_build_nca_bank` (model.eval() + no_grad forward over the train split), with
leave-one-out self-exclusion **by video id**. `tau = --nca_tau = 0.1`. The BCE half of the hybrid
loss, `--contrast_mode retrieval` (default), `--loss triplet`, FAISS mining, EM rounds, and every
other knob are untouched — the candidate swaps the head contrastive term and nothing else.

**Base.** MHC_zh cached LoRA features
`data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`
(the historical "ZH floor", job 13150). HateMM (conditional leg) uses
`data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt`
(job 13241). These are the exact caches the 2026-07-25 family used, so the re-audit measures the same
object rather than a re-based one.

## 3. Arms

Two arms, identical python command, one flag pair apart. The command is byte-identical to
`scripts/slurm/ncafam_family.sbatch` (the frozen 2026-07-25 runner) except `--group_name` /
`--exp_comment` / no `tee` / per-run checkpoint dir removed.

| arm | flags added | meaning |
|---|---|---|
| `floor` | *(none)* | deployed head: triplet contrastive + FAISS mining + BCE, `--contrast_mode retrieval` |
| `nca01` | `--head_loss nca --nca_tau 0.1` | candidate: NCA/soft-kNN contrastive term, everything else identical |

The pairing is exact: same cache, same architecture (identical parameter count), same 30 epochs,
same batch size, lr, dropout, warmup, EM rounds, seeds. No dimension mismatch and no training-volume
mismatch exists between the arms, so the REAUDIT §2 lesson (dimension-matched / recipe-matched
control needed) does **not** apply here; the floor already *is* the recipe-matched control. The one
asymmetry — the NCA arm builds an extra inference-only forward pass per epoch to refresh the bank —
adds no gradient steps and no parameters.

Runner: `idea-stage/reaudit_nca/run_grid.sh`
sha256 `820e78943635ed32d2bf33f2e4b0760f3f60b546a137ca746b4ac0d32f99b17a`
Analyzer: `idea-stage/reaudit_nca/analyze.py`
sha256 `272865db21bdef240731efa6c2de6d285bb8262f1dfeef8f61c06a5906fdc5f1`
Code: `src/model/loss.py` sha256 `89c130d38e528f6f3f39ec6f62510558ccc68bb6422296b41ab7dbc22427b898`,
`src/run_rac.py` sha256 `50321cb1f1ce120b7229765415457e1a7c20b5ce5cadf6232d65156a1f9519ab`.

## 4. Seeds

**MHC_zh: seeds 41000–41029 (30 seeds), both arms, seed-paired.**
Disjoint from every seed range used before: the original NCA family used 0/1/2; `REAUDIT` used
20260900–20260929 and 300–329; no artefact anywhere in the repo carries a seed in 41000–41029
(checked by grep over scripts/logs).

**HateMM (conditional): seeds 41000–41014 (15 seeds), both arms.** Run **only if** the MHC_zh primary
read-out passes §5. Not run otherwise.

## 5. Read-outs and the decision rule — frozen

All read-outs use the retrieval-vote lines the deployed recipe emits (`Val_Retrieval` /
`Test_Retrieval`), warmup 5, i.e. epoch selection restricted to epochs ≥ 5. Test labels are used
**only** to compute the reported metric at an epoch chosen by dev; no test-driven selection, no
tuning after any frozen-range number is seen.

| tag | epoch selection | metric read | role |
|---|---|---|---|
| **P1** | argmax over epochs ≥ 5 of **dev macro-F1** | test macro-F1 | **PRIMARY, gating** |
| **P2** | final epoch (29) | test macro-F1 | corroborating |
| **HIST** | argmax over epochs ≥ 5 of (**dev acc**, tie-break dev roc) | test acc **and** test macro-F1 | historical-claim replication |
| dev-side | — | dev macro-F1 / dev acc at the selected epoch, and the best dev value | descriptive only |

`HIST` is the *exact* selection rule the 2026-07-25 family used, and it is the rule under which the
+0.0112 / +0.0113 was produced. It is registered here because the claim under audit is stated in that
frame; it is **not** the gating read-out, because the project's current standard and this re-audit's
mandate is dev-macro-F1 selection.

Statistics: seed-paired differences `nca01 − floor`, mean, sample std, paired bootstrap over seeds
(20 000 resamples, fixed rng seed 12345) giving SE and a percentile 95 % CI, plus the count of
positive seeds.

### Decision rule (frozen, applied verbatim)

1. **REVIVED** iff **P1** has `mean ≥ +0.005` **and** its paired-bootstrap 95 % CI excludes zero.
   → then run the HateMM 15-seed leg and report whether the effect crosses datasets (same bar,
   reported, not gating the MHC_zh verdict).
2. Else, if **P1** fails but **HIST** passes the same bar on **both** test acc and test macro-F1
   → verdict **SELECTION-RULE-BOUND**: the historical effect is real but is a property of
   accuracy-argmax epoch selection, not of the loss. This is **not** a revival, no HateMM leg is run,
   and the direction stays closed; the finding is recorded as a protocol lesson.
3. Else → **NOT REVIVED**.

Non-negotiables: single submission of the frozen grid, no re-run, no seed added or dropped after any
frozen-range number is seen, no post-hoc change of read-out or bar.

## 6. Instrument checks — all run **before** this freeze, all passed

| check | result |
|---|---|
| NCA code path unchanged since the 2026-07-25 freeze | `src/model/loss.py` and `src/run_rac.py` differ from frozen shas `2ae7a73f…` / `b85eb72a…` **only** by additive, default-off blocks added later (`--contrast_mode none`/`random`, `--soft_target_json`, `--label_smoothing`). `git diff` over `_nca_head_loss`, `_supcon_head_loss`, `_build_nca_bank` and the `head_loss` dispatch: **zero changed lines**. Diff verified line by line. |
| harness determinism on this machine | `MHC_zh_nca01_s0` run twice in separate dirs → **all 60 Val/Test_Retrieval metric lines identical**. |
| grid completion | every instrument run finished 30/30 epochs, 60/60 metric lines, rc 0. |
| historical effect replicates on this hardware | seeds 0/1/2 re-run for both arms locally. `HIST` test-acc contrast **+0.0134** (historical **+0.0112**), `HIST` test-macro-F1 contrast **+0.0111** (historical **+0.0113**). Effect size reproduced to within seed noise. |

**Bit-level match to the banked 2026-07-25 trainlogs is impossible and was not attempted**: jobs
13150/13482 ran on the cluster's A100 under the old torch build, this machine is an RTX 5090 on
torch 2.7.1+cu128. Per-epoch values shift by up to ~1–2 macro-F1 points, the documented cross-hardware
drift. The design absorbs this completely: **both arms are re-run here on the same hardware, same
image, same seeds**, so the drift is common to both and cancels inside the seed-paired contrast. No
banked cluster number enters any contrast in this re-audit.

**Declared, pre-freeze, non-blinding-violating:** the seeds 0/1/2 replication above computes the
candidate contrast on the *historical* seeds. Those numbers are a replication of an already-published
result and are disjoint from the frozen 41000–41029 range. The bar (+0.005, CI excluding zero) comes
from the re-audit mandate, not from these numbers, and was fixed before they were computed.

**Known before the frozen run, and stated here so it cannot be claimed as a post-hoc discovery:** on
the historical seeds 0/1/2 the P1 (dev-macro-F1-selected) contrast is negative (−0.0064, 0/3), and
the dev-side contrasts are ≈0 (dev macro-F1 −0.0023, dev acc 0.0000). This is *why* HIST is
registered separately as clause 2 above. It does not alter the bar or the read-outs, which are the
ones the mandate specified.

## 7. Execution

```
bash idea-stage/reaudit_nca/run_grid.sh logging/runs/reaudit_nca \
     floor,nca01 MHC_zh Qwen2.5-VL-7B-Instruct-LoRA_HF <41000..41029> RAC_RNCA
python idea-stage/reaudit_nca/analyze.py logging/runs/reaudit_nca/logs \
     --dataset MHC_zh --seeds <41000..41029> --out idea-stage/reaudit_nca/mhczh_results.json
```

Detached from SSH (`setsid nohup`), log `logging/runs/reaudit_nca/run.log`, PID
`logging/runs/reaudit_nca/run.pid`. ~8 s per run, 60 runs ≈ 8 min. Results and verdict go to
`idea-stage/REAUDIT_NCA_RESULT.md`.
