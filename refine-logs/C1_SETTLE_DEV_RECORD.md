# C1 SETTLE — measured DEV read-out of the faithful sequential Stage-2 cell

**Purpose.** Convert the C1 (E2EQ) kill from an *inferred* negative into a *measured* one, per
`refine-logs/C1_KILL_REVIEW.md` §5 ("Minimal cheap evidence"). The kill review confirmed KILL on
expected value but flagged that the one faithful configuration — Stage-1 LoRA-adapt the features,
then **sequentially** train the project's standard RGCL Stage-2 head (align-fusion + triplet-margin
cos 0.1 + BCE hybrid, top-20 kNN vote) on the frozen adapted features — was never run on video
(P9 ran a *raw* kNN with no trained Stage-2 head; P9b ran RGCL *jointly*, degenerate at bs=1).
This run measures exactly that cell on HateMM seed-0, DEV read-out only.

**Verdict: `KILL_STANDS_MEASURED`.** The sequential Stage-2 head on the Stage-1-adapted features
lands **below** the frozen-Qwen RGCL floor on DEV (−0.019 val-sel / −0.028 final acc), never above
it. No positive signal, no unexpected jump. The inferred within-noise negative is now measured.

---

## The run

| field | value |
|---|---|
| Job id | **13039** (SLURM), `COMPLETED`, ExitCode `0:0` |
| Submission | `sbatch scripts/analysis/c1settle_hatemm_s0.sbatch` (single attempt, no resubmit) |
| Partition / GPU | `slurmpartition`, 1× A100, 8 CPU / 64 GB (mirrors `enc3seed.sbatch`); no `--time` |
| Env | `conda activate HateVideo`; `HF_HUB_OFFLINE=1`, `WANDB_MODE=disabled` |
| Walltime | seconds (frozen-feature head; enc3seed precedent ~20–25 s/run) |
| Trainlog | `slurm/logs/c1settle_HateMM_c1settle_hatemm_s0_seed0_13039.trainlog` |
| Batch out | `slurm/logs/c1settle_13039.out` |

**Invocation** — byte-identical to the frozen-Qwen floor command
(`scripts/slurm/enc3seed.sbatch` / `train_archive_baseline.sbatch`) **except `--model`**:

```
python ./src/run_rac.py --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
  --dataset HateMM --model c1settle_hatemm_s0 \
  --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align \
  --hard_negatives_loss True --no_hard_negatives 1 --final_eval False --seed 0 \
  --group_name c1settle --metric cos --loss triplet --batch_norm False \
  --hybrid_loss True --warmup 5 --majority_voting arithmetic --no_pseudo_gold_positives 1 \
  --lambda_seg 0 --seg_mode full --num_subclips 4 --em_rounds 2 \
  --consensus_topk 10 --consensus_margin 0.2 --exp_comment _c1settle_hatemm_s0 \
  --Faiss_GPU False --force False
```

Runtime `Namespace` confirms the head config matches the floor: `fusion_mode='align'`, `topk=20`,
`loss='triplet'`, `triplet_margin=0.1`, `hybrid_loss=True`, `proj_dim=map_dim=1024`, `warmup=5`
(trainlog `:1`). The **only** manipulated variable vs the floor is `--model`
(`Qwen2.5-VL-7B-Instruct_HF` → `c1settle_hatemm_s0`), so any DEV delta is attributable solely to the
Stage-1 LoRA adaptation of the input features.

### Inputs (the P9 Stage-1 LoRA-adapted feature caches)

`--model c1settle_hatemm_s0` resolves through symlinks in `data/CLIP_Embedding/HateMM/`:

| split slot | symlink | target | rows / dim |
|---|---|---|---|
| train | `train_c1settle_hatemm_s0.pt` | `train_p9c3_hatemm_s0.pt` (P9 adapted) | 744 × 3584 img+text |
| dev | `dev_seen_c1settle_hatemm_s0.pt` | `dev_seen_p9c3_hatemm_s0.pt` (P9 adapted) | 107 × 3584 |
| "test" | `test_seen_c1settle_hatemm_s0.pt` | **`dev_seen_p9c3_hatemm_s0.pt` (DEV cache)** | 107 × 3584 |

Cache-integrity checks (CPU, pre-run): the P9 adapted caches are structurally identical to the
frozen-Qwen caches — same `{ids,img_feats,text_feats,labels}` keys, same 744/107/215 splits, same id
order, same labels — and the LoRA adaptation genuinely reshaped the features (train text row-cosine
vs frozen = 0.936, img = 0.996; not a copy). So the caches are usable for a faithful sequential
Stage-2 run; **no re-extraction was needed or performed.**

### TEST WAS NEVER TOUCHED

The `"test"` split slot is deliberately symlinked to the **DEV** cache, so `run_rac.py`'s per-epoch
`Test_Retrieval` block loads a duplicate of DEV and the real held-out test cache
`test_seen_p9c3_hatemm_s0.pt` (215 rows, distinct inode, mtime Jul 8, untouched) is **never opened,
loaded, or evaluated**. `Test_Retrieval == Val_Retrieval` by construction and is ignored; only
`Val_Retrieval` (DEV) and the hybrid classifier DEV acc are reported. No existing file was modified;
`run_rac.py` was run unmodified. Model selection uses DEV (`Val_Retrieval` acc, warmup≥5) as in the
floor.

---

## VAL results table (HateMM seed-0, identical 107-sample DEV set, same head, seed 0)

Two read-outs of the trained RGCL head, both protocols (val-selected = epoch ≥5 max `Val_Retrieval`
acc, roc tie-break; final = epoch 29):

| read-out · protocol | **this run** (Stage-1-adapted + sequential Stage-2) | frozen-Qwen floor (same head, frozen feats) | Δ (this − floor) |
|---|---|---|---|
| **kNN vote · val-sel** | **0.8411** (e12) `:147` | 0.8598 (e28) `¹:292` | **−0.0187** |
| **kNN vote · final-ep** | **0.8224** (e29) `:301` | 0.8505 (e29) `¹:302` | **−0.0281** |
| MLP head · val-sel | 0.5981 (e12) | 0.5981 (e28) `¹:288` | 0.0000 |
| MLP head · final-ep | 0.5981 (e29) | 0.5981 (e29) `¹:298` | 0.0000 |

`:NNN` = line in this run's trainlog (`c1settle_..._13039.trainlog`).
`¹` = frozen-Qwen floor raw log `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog`.

- The settle head's kNN DEV acc **never exceeds 0.8411 at any of the 30 epochs** (max 0.8411 at
  e0/e12/e25); the frozen floor reaches 0.8598 at e28. The below-floor result is genuine, not a
  val-selection artifact.
- The **MLP (hybrid classifier) head collapses to majority-class 0.5981 on HateMM under BOTH feature
  sets** (64/107 negatives) — a known property of this head on HateMM; the kNN-vote head is the
  method's operative read-out. The MLP column carries no signal and is reported only for completeness.

### P9 comparators (context; TEST-only, not DEV — cannot be differenced against the DEV table above)

The P9 record committed HateMM read-outs on **TEST only** (no HateMM DEV was committed; the P9 dev
numbers on record are EN/ZH only, `EXP_p9_lmm_rgcl_video.md:137`). For orientation:

| P9 read-out (HateMM s0, TEST) | value | vs trained-RGCL TEST floor 0.8605 | file:line |
|---|---|---|---|
| C3-mlp (in-LMM MLP head) | 0.8698 | +0.9 (≈ floor) | `EXP_p9_lmm_rgcl_video.md:191` |
| C3-knn (**raw** kNN, no trained Stage-2 head) | 0.814 (s0 0.823) | **−4.7 (below)** | `EXP_p9_lmm_rgcl_video.md:192` |

Frozen-Qwen TEST floor for reference: val-sel 0.8698 / final 0.8605
(`experiments/exp-encoder-3seed.md:155`).

---

## Interpretation

**Does the sequential Stage-2 head repair P9's kNN-below-floor?** Partially, and only up to — in fact
slightly under — the floor. P9's *raw* C3-knn (no trained Stage-2 head) sat ~4.7 pts below the floor
on TEST. Training the actual RGCL Stage-2 head on the same adapted features recovers most of that
collapse: the kNN read-out climbs back to ≈ floor − 0.019 (val-sel) / − 0.028 (final) on DEV. So the
contrastive Stage-2 objective does re-shape the adapted embedding space back toward the retrieval
memory, as designed — but it climbs only to *meet* a flat ceiling, never past it. This is the exact
mechanism the kill review predicted (§4b: "a kNN that climbs to meet a flat head is still a flat
system") and that P9b independently showed jointly (head↔memory redistribution, net-zero).

**Does the system beat the frozen floor by more than noise?** No. On the clean apples-to-apples DEV
comparison (identical head, identical 107-sample DEV set, seed 0; only the input features differ
frozen→adapted), the sequential Stage-2 head lands **−0.019 (val-sel) to −0.028 (final) below** the
frozen-Qwen floor — i.e. within/around the project's repeatedly-observed ±1–2 pt seed-noise band, on
the negative side. There is **no** positive increment, let alone the +3 the goal requires. The
measured outcome matches the kill review's prediction ("≈ floor, +~0.7 at best") and if anything sits
marginally worse than the neutral prediction — never better. **No `UNEXPECTED_SIGNAL`; no >+3 jump; no
leakage check triggered.** Stage-1 feature adaptation neither helps the trained Stage-2 kNN head nor
harms it beyond noise; the faithful sequential cell reproduces the frozen floor.

**Conclusion.** RA-HMD's exact sequential two-stage recipe, run faithfully on video (Stage-1 adapted
features → sequentially-trained RGCL Stage-2 head + kNN vote), still lands within noise of — and here
marginally below — the frozen-encoder RGCL floor at 7B on HateMM. This closes C1 as a **measured**
negative, a cleaner completion of the P9 (raw-kNN, no Stage-2 head) / P9b (joint, bs=1-degenerate)
negatives. The kill stands.

---

## Artifacts

- Runner (new, not a modification): `scripts/analysis/c1settle_hatemm_s0.sbatch`
- Input symlinks (new): `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_c1settle_hatemm_s0.pt`
  (`test_seen_*` → the DEV cache, by design — see "TEST WAS NEVER TOUCHED")
- Trainlog: `slurm/logs/c1settle_HateMM_c1settle_hatemm_s0_seed0_13039.trainlog`; batch out
  `slurm/logs/c1settle_13039.out`
- Head checkpoints: `logging/Retrieval/HateMM/c1settle/RAC_lr0.0001_..._c1settle_hatemm_s0/ckpt/`

### Provenance index

- This run's DEV read-out: trainlog `:147` (val-sel e12), `:301` (final e29), `:1` (Namespace).
- Frozen-Qwen HateMM s0 DEV floor: `enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog:292`
  (kNN val-sel e28 0.8598), `:302` (kNN final e29 0.8505), `:288`/`:298` (MLP 0.5981).
  *(Note: the kill review §Provenance cited "val-sel 0.8729 / final 0.8682" for this floor; those do
  not reproduce in the raw log — the directly-read DEV values are 0.8598 / 0.8505 and are used here.)*
- Frozen-Qwen HateMM s0 TEST floor: `experiments/exp-encoder-3seed.md:155`.
- P9 HateMM TEST comparators: `research-wiki/EXP_p9_lmm_rgcl_video.md:191` (C3-mlp), `:192` (C3-knn).
- Kill review + settling prescription: `refine-logs/C1_KILL_REVIEW.md` §5, §4b.
