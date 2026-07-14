# B5 PROBE DESIGN — executable G0-cond spec for the frozen-Qwen MHC-ZH operating-point conversion

**Author:** B5 pre-registration designer (read-only recon; ZERO GPU; no submissions; no commits except
this doc + the prereg). **Date:** 2026-07-14.
**Companion prereg:** `research-wiki/experiments/exp-conv-zh-b5.md` (design, decision rules, ledgers).
**Purpose of this file:** the *executable* probe — exact on-disk paths (verified), reload mechanics,
commands, expected outputs, the G-repro anchor table (numbers re-read from primary logs, provenance-
lined), the oracle kill-switch computation, and the D3 bootstrap. **Do NOT run anything from a design
agent; this is the spec the executor follows after review sign-off.**

---

## 1. What 13115 was (config the probe must reproduce exactly)

Parsed from `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed0_13115.trainlog` Namespace (line 1):

```
dataset='MHC_zh'  model=<CLIP|Qwen>  topk=20  majority_voting='arithmetic'  metric='cos'
similarity_threshold=-1.0  fusion_mode='align'  num_layers=3  proj_dim=1024  map_dim=1024
dropout=[0.2, 0.4, 0.1]  batch_norm=False  batch_size=64  loss='triplet'  hybrid_loss=True
warmup=5  lambda_seg=0.0  archive_feats=None  Faiss_GPU=False  device='cuda'
```

Consequences for the probe:
- **Vote is CONTINUOUS, not 21-level.** `majority_voting='arithmetic'` + `use_sim=True` ⇒ per-video
  vote = `sum_k [ (2·label_k − 1)·sim_k · w_k ] / sum_k w_k`, `w = [20,19,…,1]`
  (`src/utils/metrics.py:274-284`). Threshold-search grid = empirical unique-vote midpoints (§5), NOT
  a fixed 21-level grid.
- **Deployed cut = vote ≥ 0** (`sigmoid(vote) ≥ 0.5`, `metrics.py:300`); macro-F1 at the same cut
  (`:307-309`).
- **CPU-faiss** (`Faiss_GPU=False`, `evaluate_rac.py:423-430`): `IndexFlatIP` over L2-normalized
  float32, exact search ⇒ **faiss retrieval deterministic on CPU conditional on identical head-forward
  embeddings** (amendment A9: end-to-end CPU-vs-13115(GPU) reproduction is NOT guaranteed — it is
  exactly what the G-repro gate verifies). The head forward ran on `device='cuda'` in 13115, so a
  pure-CPU replay may differ at float epsilon; the G-repro gate (§4) to 4 dp is the arbiter, with a
  1-min GPU fallback (§6) for bit-exact match if needed.
- **Model:** `classifier_hateClipper(image_dim, text_dim, num_layers=3, proj_dim=1024, map_dim=1024,
  fusion_mode='align', dropout=[0.2,0.4,0.1], batch_norm=False, args=args)`
  (`src/run_rac.py:1117-1120`; class `src/model/classifier.py`). Dims: **CLIP img 1024 / text 768**,
  **Qwen img 3584 / text 3584**.

---

## 2. Checkpoint-recoverability finding — RECOVERABLE (all 12 heads on disk, verified 2026-07-14)

**Finding: YES — the probe is zero-GPU.** run_rac.py saves a head checkpoint for **every** epoch
(`src/run_rac.py:764-767`: `ckpt/epoch_model_{epoch}_{val_acc}.pt`). All 6 ZH runs from 13115 retain
`epoch_model_0..29_*.pt`. The per-epoch retrieval pickles were **NOT** written (`save_embed` off ⇒ 0
`*.pkl` on disk), so votes must be **recomputed** from the checkpoints — which exist.

### 2.1 The 12 required checkpoints (final-epoch e29 + val-selected epoch), exact paths

Base dir: `logging/Retrieval/MHC_zh/RAC_video_archive_seeds/`
Per-run dir: `RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20__PseudoGold_positive_1_hard_negative_1_seed{S}_hybrid_loss_{MODEL}/ckpt/`
(`{MODEL}` ∈ `openai_clip-vit-large-patch14-336_HF`, `Qwen2.5-VL-7B-Instruct_HF`)

| arm | seed | final-epoch ckpt (protocol B) | val-sel epoch | val-sel ckpt (protocol A) |
|---|---|---|---|---|
| CLIP | 0 | `epoch_model_29_0.8076923076923077.pt` | 29 | *(same as final)* |
| CLIP | 1 | `epoch_model_29_0.7692307692307693.pt` | 28 | `epoch_model_28_0.782051282051282.pt` |
| CLIP | 2 | `epoch_model_29_0.7948717948717948.pt` | 25 | `epoch_model_25_0.8205128205128205.pt` |
| Qwen | 0 | `epoch_model_29_0.782051282051282.pt` | 22 | `epoch_model_22_0.8205128205128205.pt` |
| Qwen | 1 | `epoch_model_29_0.8205128205128205.pt` | 25 | `epoch_model_25_0.8717948717948718.pt` |
| Qwen | 2 | `epoch_model_29_0.782051282051282.pt` | 28 | `epoch_model_28_0.8461538461538461.pt` |

Robust selector (do NOT hardcode the float suffix): `epoch_model_29_*.pt` (final) and
`epoch_model_{SEL}_*.pt` (val-sel) with `SEL` from the table. The float suffix = the run's dev acc at
that epoch and is deterministic; the glob avoids transcription error. Val-sel epochs cross-checked
against the checkpoint dev-acc suffixes and the B1 verdict (`B1_VERDICT_REVIEW.md:43-44`): Qwen s0
e22 tie {22,26,28}@0.8205 → roc tie-break e22; CLIP s1 e28 tie {18,27,28}@0.7821 → roc tie-break e28.

### 2.2 Cached feature inputs (present; verified in B1 prereg asset check)

`data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_{openai_clip-vit-large-patch14-336_HF,Qwen2.5-VL-7B-Instruct_HF}.pt`
(loader: `src/data_loader/dataset.py:499,587-589`; MHC_zh supported; Qwen 3584/3584, CLIP 1024/768).

### 2.3 Disk-guard risk + mitigation (concurrent cleanup ALERT)

`scripts/disk_guard.sh` "push-verify-prune oldest logging/ checkpoints (mirrored to B2 under logs/)"
(`slurm/logs/disk_guard.log:550373`). It prunes **oldest-first** and **mirrors to B2 before** deleting.
The 13115 ZH ckpts are **today's (newest)**, so immediate deletion risk is **LOW but non-zero** (total
footprint ~6.7 GB: Qwen 1.4–1.6 GB/seed, CLIP 0.69–0.78 GB/seed). **Mitigation (executor step 0, cheap
~372 MB):** copy only the 12 needed heads to a guard-excluded path, e.g.

```
DEST=/data/jehc223/RGCL/refine-logs/b5_ckpt_snapshot   # NOT under logging/ (guard scope)
mkdir -p "$DEST"
# for each of the 12 (arm,seed,epoch): cp the resolved epoch_model_{E}_*.pt into $DEST/{model}_s{S}_e{E}.pt
```

If the heads are pruned before the probe runs, they are recoverable from B2
(`b2:junyi-data/RGCL_video/logs/...`), or regenerate via the §6 GPU fallback.

---

## 3. The probe script (spec) — `scripts/analysis/b5_calibration_probe.py` (to be authored by executor)

Reuses repo modules; **does not reimplement the vote** — it calls the real
`compute_metrics_retrieval`, whose 6th return value IS the per-video vote `list_majority_voted`.

### 3.1 Per (encoder, seed, protocol): dump votes

```python
# pseudo-spec; executor authors the runnable version, argv-driven
import torch, numpy as np, faiss
from model.classifier import classifier_hateClipper
from utils.metrics import compute_metrics_retrieval
from model.evaluate_rac import retrieve_evaluate_RAC_
# ... build args Namespace with the §1 fields; args.device='cpu'; args.Faiss_GPU=False
# ... build train/dev/test dataloaders for (dataset='MHC_zh', model=MODEL), shuffle=False
#     (identical loader construction to run_rac.py; eval loaders are not shuffled)
img_dim, txt_dim = (3584,3584) if 'Qwen' in MODEL else (1024,768)
model = classifier_hateClipper(img_dim, txt_dim, 3, 1024, 1024, 'align',
                               dropout=[0.2,0.4,0.1], batch_norm=False, args=args)
model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu')); model.eval()

ld_dev,  y_dev  = retrieve_evaluate_RAC_(train_dl, dev_dl,  model, largest_retrieval=20,
                                         threshold=-1.0, args=args, eval_name='dev',  epoch=EP,
                                         archive_bank=None, target_pack=None)
ld_test, y_test = retrieve_evaluate_RAC_(train_dl, test_dl, model, largest_retrieval=20,
                                         threshold=-1.0, args=args, eval_name='test', epoch=EP,
                                         archive_bank=None, target_pack=None)

# 6th return = list_majority_voted = per-video CONTINUOUS vote; 8th = macro dict (deployed metrics)
_, _, _, _, _, votes_dev,  labels_dev,  macro_dev  = compute_metrics_retrieval(
        ld_dev,  y_dev,  majority_voting='arithmetic', topk=20, use_sim=True)
_, _, _, _, _, votes_test, labels_test, macro_test = compute_metrics_retrieval(
        ld_test, y_test, majority_voting='arithmetic', topk=20, use_sim=True)

np.savez(f'{OUT}/{MODEL}_s{S}_{PROTO}.npz',
         votes_dev=np.asarray(votes_dev),  labels_dev=np.asarray(labels_dev),
         votes_test=np.asarray(votes_test), labels_test=np.asarray(labels_test),
         deployed_test_acc=macro_test['acc'], deployed_test_mf1=macro_test['macro_f1'],
         deployed_test_roc=macro_test['roc'])
```

`EP` = the checkpoint's epoch (29 for final; the SEL epoch for val-sel). Labels are `{0,1}` with
positive = 1 (harmful); the deployed prediction is `vote ≥ 0`.

### 3.2 Threshold arms (pure numpy on the dumped votes)

```python
def metrics_at(votes, labels, tau):
    pred = (np.asarray(votes) >= tau).astype(int)
    acc  = (pred == labels).mean()
    mf1  = macro_f1(labels, pred)   # sklearn f1_score(..., average='macro', zero_division=0)
    return acc, mf1

def grid(votes):                                   # §5 empirical unique-vote midpoints + sentinels
    u = np.unique(np.asarray(votes)); 
    mids = (u[:-1]+u[1:])/2.0
    return np.concatenate([[u.min()-1e-6], mids, [u.max()+1e-6]])

# (a) deployed
acc_dep, mf1_dep = metrics_at(votes_test, labels_test, 0.0)   # MUST == anchor (G-repro, §4)
# (b) honest / method  — τ maximizes dev macro-F1, mid-plateau tie-break (§5)
G = grid(votes_dev); mf1_dev = np.array([metrics_at(votes_dev, labels_dev, t)[1] for t in G])
plateau = np.flatnonzero(np.isclose(mf1_dev, mf1_dev.max()))
tau_star = G[plateau[len(plateau)//2 - (1 if len(plateau)%2==0 else 0)]]   # lower-median of plateau
acc_hon, mf1_hon = metrics_at(votes_test, labels_test, tau_star)
# secondary sensitivity: τ maximizes dev BALANCED-acc (report only, NOT decision)
# (c) oracle — τ maximizes test acc (and separately test macro-F1); UPPER BOUND, never a result
Gt = grid(votes_test)
acc_orc = max(metrics_at(votes_test, labels_test, t)[0] for t in Gt)
mf1_orc = max(metrics_at(votes_test, labels_test, t)[1] for t in Gt)
```

Positive-class F1 sanity: `macro_f1` here must reproduce the pipeline's `macro_f1` at τ=0 to 4 dp
(same sklearn call, `metrics.py:309`).

### 3.3 Outputs (one CSV/JSON table)

Per (encoder, seed, protocol): `deployed_acc, deployed_mf1` (+ G-repro pass/fail), `honest_tau,
honest_acc, honest_mf1`, `balacc_tau, balacc_acc, balacc_mf1` (sensitivity), `oracle_acc_tau,
oracle_acc, oracle_mf1_tau, oracle_mf1`, and `calib_tax_acc = oracle_acc − honest_acc`,
`calib_tax_mf1 = oracle_mf1 − honest_mf1`. Plus the §7 paired-Δ summary and the §8 bootstrap
distributions.

---

## 4. G-repro anchor table (deployed arm MUST match to 4 dp) — re-read from primary logs

Re-read directly from the six `slurm/logs/enc3s_MHC_zh_*_13115.trainlog` `Test_Retrieval Epoch NN
macroF1:` print lines (`\r`/`\n`-split parser, as B1's verified re-parse; values match
`B1_VERDICT_REVIEW.md:29-40` exactly). `~ln` = grep -n newline-line of the print segment.

### Protocol B (final-epoch, e29) — deployed-arm anchors

| arm | seed | ~ln | macroF1 | acc | roc |
|---|---|---|---|---|---|
| CLIP | 0 | 275 | 0.7706 | 0.8054 | 0.8382 |
| CLIP | 1 | 274 | 0.7542 | 0.8054 | 0.8342 |
| CLIP | 2 | 271 | 0.7913 | 0.8322 | 0.8444 |
| Qwen | 0 | 275 | 0.7864 | 0.8188 | 0.8906 |
| Qwen | 1 | 272 | 0.7759 | 0.8054 | 0.8951 |
| Qwen | 2 | 269 | 0.7514 | 0.7852 | 0.8806 |

### Protocol A (val-selected epoch) — deployed-arm anchors

| arm | seed | selEp | ~ln | macroF1 | acc | roc |
|---|---|---|---|---|---|---|
| CLIP | 0 | 29 | 275 | 0.7706 | 0.8054 | 0.8382 |
| CLIP | 1 | 28 | 265 | 0.7579 | 0.8054 | 0.8346 |
| CLIP | 2 | 25 | 238 | 0.7742 | 0.8121 | 0.8419 |
| Qwen | 0 | 22 | 218 | 0.7412 | 0.7919 | 0.8838 |
| Qwen | 1 | 25 | 239 | 0.7871 | 0.8121 | 0.8874 |
| Qwen | 2 | 28 | 260 | 0.7759 | 0.8054 | 0.8940 |

### DEV anchors (amendment A2, BLOCKING) — `Val_Retrieval Epoch NN` lines, re-read from primary logs

Because the calibration selects τ on the **dev** vote ordering, the dev deployed metrics are anchored
co-equally with the test metrics. Re-read from the same six `enc3s_MHC_zh_*_13115.trainlog`
`Val_Retrieval Epoch NN macroF1:` print lines (independently re-verified 2026-07-14; matches
`B5_PREREG_REVIEW.md` §2 Item-5).

#### Protocol B (final-epoch, e29) — DEV deployed-arm anchors

| arm | seed | macroF1 | acc | roc |
|---|---|---|---|---|
| CLIP | 0 | 0.7857 | 0.8077 | 0.8329 |
| CLIP | 1 | 0.7225 | 0.7692 | 0.8879 |
| CLIP | 2 | 0.7645 | 0.7949 | 0.8764 |
| Qwen | 0 | 0.7650 | 0.7821 | 0.8579 |
| Qwen | 1 | 0.8050 | 0.8205 | 0.8864 |
| Qwen | 2 | 0.7613 | 0.7821 | 0.8436 |

#### Protocol A (val-selected epoch) — DEV deployed-arm anchors

| arm | seed | selEp | macroF1 | acc | roc |
|---|---|---|---|---|---|
| CLIP | 0 | 29 | 0.7857 | 0.8077 | 0.8329 |
| CLIP | 1 | 28 | 0.7471 | 0.7821 | 0.8836 |
| CLIP | 2 | 25 | 0.7894 | 0.8205 | 0.8343 |
| Qwen | 0 | 22 | 0.7940 | 0.8205 | 0.8693 |
| Qwen | 1 | 25 | 0.8628 | 0.8718 | 0.9307 |
| Qwen | 2 | 28 | 0.8301 | 0.8462 | 0.8514 |

**G-repro gate (test AND dev; amendment A2):** for all 12 (6 arms × 2 protocols), the probe's
`deployed_test_acc / deployed_test_mf1 / deployed_test_roc` MUST equal the **test** anchor above to
4 dp, **AND** the probe's recomputed **dev** deployed `acc / macroF1 / roc` at each loaded checkpoint
MUST equal the **DEV** anchor above to 4 dp. Mismatch (test or dev) on CPU ⇒ retry via the §6 GPU
fallback (bit-exact device match); mismatch on GPU too ⇒ **HALT**, replay machinery invalid, probe does
not proceed (no calibrated number is trustworthy without this — REFLECTION §4 probe-validity mandate).

Deployed-arm sanity means (from the anchors): final-epoch acc CLIP 0.8143 / Qwen 0.8031 (Δ −0.0112);
roc CLIP 0.8389 / Qwen 0.8888 (Δ +0.0499). These are the numbers the conversion probe starts from.

---

## 5. Primary calibration statistic + grid + tie-break (mirrors prereg §5)

- **PRIMARY (decision) statistic:** τ = **argmax dev macro-F1** (a binding goal metric; harder,
  class-aware clause; self-consistent, no metric-shopping — full rationale in prereg §5.1).
- **Grid:** empirical unique-vote midpoints on dev + two sentinels (vote is continuous; §1).
- **Tie-break:** lower-median threshold of the maximal-macro-F1 plateau; secondary = nearest to the
  deployed cut (vote = 0). **Amendment A3:** "plateau" = the full set of argmax grid indices
  (`np.flatnonzero`, contiguity NOT assumed); the tie-break is the lower-median of that index array;
  the §3.2 code (index-median of the flatnonzero array) is authoritative.
- **Secondary (sensitivity only, NOT decision):** τ = argmax dev balanced-acc — computed in the same
  pass, reported for transparency, never swapped in after results.

---

## 6. GPU fallback (only if CPU G-repro fails, or if the 12 heads get pruned)

- **G-repro-fail fallback (bit-exact device):** re-run the §3.1 dump with `args.device='cuda'`,
  `Faiss_GPU=False` in a ~1-min eval-only sbatch (1x A100). This matches 13115's GPU head forward +
  CPU faiss exactly.
- **Heads-pruned fallback:** re-run the 13115 configs from the B1 runner
  (`scripts/slurm/enc3seed_zh_b1.sbatch`, 6 rows, seconds/run) with the §3.1 vote-dump instrumentation
  added; anchor every deployed-arm number to the §4 table (this IS the G-repro). One serial sbatch,
  `FORCE=False`, fresh group, no `--time`, `PENDING (JobHeldUser)` = wait.

Both are single-submit, ceremony per prereg §14. Neither is needed if CPU G-repro passes.

---

## 7. Oracle kill-switch computation (binding; decides any formal GPU) — mirrors prereg §6.4

Per seed s, paired on the **oracle** arm (each encoder uses its own test-optimal threshold):
`ΔAcc_oracle(s) = oracle_acc(Qwen,s) − oracle_acc(CLIP,s)`, likewise `ΔmF1_oracle(s)`. Then:

- **KILL-SWITCH (binding, per-protocol; amendment A1 — mirrors prereg §6.4):** per seed *s* and
  protocol *P* ∈ {final-epoch, val-selected}, pair the oracle arms (each encoder its own test-optimal
  threshold): `ΔAcc_oracle(s,P) = oracle_acc(Qwen,s,P) − oracle_acc(CLIP,s,P)`, likewise
  `ΔmF1_oracle(s,P)`. Protocol *P* is **ELIGIBLE** for the formal stage iff, under *P*,
  `mean_s ΔAcc_oracle(P) ≥ +0.03` **AND** `mean_s ΔmF1_oracle(P) ≥ +0.03` (3 seeds). **B5 is DEAD iff
  NEITHER protocol is eligible** — advantage non-convertible even with a perfect cut; no formal run.
  final-epoch is the reporting-emphasis reference but holds **no veto** over an independently eligible
  val-selected protocol; an eligible protocol authorizes the formal stage **only under that same
  protocol** (no cross-protocol claim). Oracle numbers are an upper bound, **NEVER** a result.
- Report per protocol; final-epoch is the primary reporting-emphasis reference; val-selected judged in
  parallel (no post-hoc protocol-shopping).
- **Honest preview gate (prereg §6.5):** even if the oracle passes, the formal single-submit runs only
  if the honest arm already clears mean Δacc ≥ +0.030 AND ΔmF1 ≥ +0.030, 3/3 sign, under an eligible
  protocol. Oracle-pass + honest-miss ⇒ banked quantified **near-miss (dies on D3)**, no GPU.

---

## 8. D3 bootstrap (val-threshold selection noise as a distribution) — mirrors prereg §8

Per (encoder, seed, protocol), on the dumped `votes_dev/labels_dev` and fixed `votes_test/labels_test`:

```python
rng = np.random.default_rng(1234)                 # fixed seed; report it
boot_acc, boot_mf1 = [], []
for b in range(1000):                             # ≥1000 resamples
    idx = rng.integers(0, len(votes_dev), len(votes_dev))     # resample 78 dev w/ replacement
    Gb  = grid(votes_dev[idx])
    mf1b = np.array([metrics_at(votes_dev[idx], labels_dev[idx], t)[1] for t in Gb])
    plb  = np.flatnonzero(np.isclose(mf1b, mf1b.max()))
    tb   = Gb[plb[len(plb)//2 - (1 if len(plb)%2==0 else 0)]]
    a, f = metrics_at(votes_test, labels_test, tb) # apply to FIXED test
    boot_acc.append(a); boot_mf1.append(f)
# report 5th/50th/95th pct of boot_acc, boot_mf1, and of the paired (Qwen−CLIP) at matched b
```

Report: per-arm test-acc/test-mF1 5/50/95 percentiles; the **paired** Δ = (Qwen − CLIP) bootstrap
distribution (pair by resample index b using a **common** dev-resample index across the two encoders'
seed-matched runs); flag the honest pass **D3-fragile** if the paired-Δ 5th percentile crosses 0. Also
report the 3 selected τ per encoder (cross-seed threshold stability) and the calibration tax
(oracle→honest) per arm.

**Amendment A6 (paired-index construction, binding).** Precompute the 1000 dev-resample index arrays
**once** (`np.random.default_rng(1234)`, each `size=78`) and reuse them **identically** across every
(encoder, seed, protocol). The `default_rng(1234)` in the loop above must therefore be lifted to a
single pre-loop draw of `boot_idx = [rng.integers(0, 78, 78) for _ in range(1000)]`; every arm indexes
`boot_idx[b]`, so the seed-matched Qwen and CLIP runs resample the **same** dev videos at each b and the
paired Δ_b cannot silently desynchronize.

---

## 9. Executor checklist (after prereg review sign-off; NOT for the design agent)

1. Step 0: snapshot the 12 heads (§2.3) to guard against the disk cleanup.
2. Author `scripts/analysis/b5_calibration_probe.py` (§3); construct args/loaders identical to
   run_rac.py eval (shuffle=False); device='cpu', Faiss_GPU=False.
3. Dump votes for all 6 arms × 2 protocols (§3.1). Run the **G-repro gate FIRST** (§4). HALT on
   mismatch → §6 fallback.
4. Compute the three threshold arms (§3.2) + secondary balanced-acc + calibration tax.
5. Compute the **oracle kill-switch** (§7). If DEAD → write the negative verdict, stop, no GPU.
6. If an eligible protocol survives + honest preview clears (§7) → run the D3 bootstrap (§8), then
   escalate to the formal single-submit ceremony (prereg §14) or accept the CPU replay as the record
   (verdict-processing decision).
7. Report raw numbers (line-numbered) to the orchestrator; apply NO pass/fail interpretation in the
   executor record — verdict processing is independent (project rule).

**Verdict-stage check (amendment A7 — for the independent verdict reviewer, NOT the executor).** The
G-repro gate exercises only the deployed τ=0 votes, not the calibration arithmetic (grid / tie-break /
macro-F1-at-τ), and this probe script is not independently code-reviewed before it runs. At verdict
processing an **independent hand-recomputation** of **one** (arm, seed, protocol) honest cell (τ_star,
honest_acc, honest_mF1) directly from the dumped `votes_dev/labels_dev`,`votes_test/labels_test` must
be performed (with the A2 dev anchor, this fully validates the machine). The executor **dumps the raw
`votes_*`/`labels_*` arrays** so this hand-check is possible, but does **not** perform it.

---

## 10. Provenance (all re-read this session)

- Fixed-threshold code + vote formula: `src/utils/metrics.py:274-309` (arithmetic+use_sim continuous
  vote; deployed cut vote≥0 at :300; macro-F1 at :307-309).
- Per-epoch eval + fixed-cut call + checkpoint save: `src/run_rac.py:659-694,764-767`.
- Retrieval eval (deterministic; CPU-faiss branch): `src/model/evaluate_rac.py:321-430`.
- Model constructor: `src/model/classifier.py` (class `classifier_hateClipper`);
  build call `src/run_rac.py:1117-1120`.
- 13115 Namespace (config): `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed0_13115.trainlog:1`.
- G-repro anchors (§4): the six `enc3s_MHC_zh_*_13115.trainlog` `Test_Retrieval` macro lines
  (final e29 + val-sel epoch), cross-checked to `refine-logs/B1_VERDICT_REVIEW.md:29-40`.
- Checkpoints present + val-sel epochs: this session's `ls` of the six 13115 `ckpt/` dirs (§2.1),
  cross-checked to `B1_VERDICT_REVIEW.md:43-44`.
- Disk-guard behavior: `slurm/logs/disk_guard.log:550373`; `scripts/disk_guard.sh`.
- G0-cond probe framing + oracle upper-bound + calibration mandate: `refine-logs/EXHAUSTION_AUDIT_2026-07-14.md`
  §7; `research-wiki/REFLECTION_mllm_integration_failures.md` §4.
