# Scale-aware cross-modal attention, commit e4e8d67 (dense-product table gradient): rule-6 code review

Date: 2026-10-31. Reviewer: independent agent (Claude Fable 5.1). Scope per rule 6: only bugs that would change
experimental observations or conclusions. The change under review is commit e4e8d67 in
`experiments/20261031_scale_aware_cma/`: the bias lookup `scale_bias[:, buckets]` in `ScaleBiasAttention.forward`
now goes through `_BucketBias` (custom `torch.autograd.Function`, same forward, table gradient computed as
`grad.reshape(h, T*T) @ one_hot(buckets)`), plus the new `bench_bias.py`, `launch/sbatch_bench.sh` and README
section 6. Purpose stated by the author: speed only; seed 234 ran with the old backward, seeds 2025 and 3407 will run
with the new one.

Read: `git show e4e8d67`; `backbone.py`, `bench_bias.py`, `launch/sbatch_bench.sh`, `train.py`, README (section 6);
`src/qtl/model.py` (`MultiHeadAttention`, `CrossModalLayer`, `PriorNet`), `src/qtl/train.py` (optimizer groups,
training loop, `LengthBatches` use, lines 255-375), `src/qtl/data.py` (`collate`, `LengthBatches`);
`scripts/reproduction_baselines/macilsd/align.py` (feature roots) and `hate_common/data.py` (split files).
CPU checks ran with `~/miniconda3/envs/HateVideo/bin/python` (torch 2.7.1+cu128), script given inline in the
session, not kept in the repository; no training, no GPU. Video lengths: `data/gt/HateMM/hate_spans.json` durations
plus `artifacts/cvoi_acq/premetric-v2/durations/hatemm_train_val.json` split roles locally; the split membership and
the VGGish row counts of the two longest videos were read on `uoa-lab3` (`results/reproduction/splits/hatemm_train.txt`,
`results/reproduction/features/vggish_1s/hatemm/*.npy`), read-only.

## Verdict: PASS

No bug that changes a number. The forward is bitwise the same as before the commit; the table gradient equals the
old one exactly in float64 and to float32 rounding in float32; every other parameter's gradient is bitwise the same;
the frozen-table ablation path, the padding path and the bucket cache behave as before. The benchmark's monkeypatch
does exercise the old path and its comparison is valid. GPU memory at the longest HateMM training video is well
within the card.

## Checks and results

1. **Forward identity.** `_BucketBias.forward` is the same indexing op `table[:, buckets]`; `apply` only wraps it
   in a custom node. Checked bitwise (`torch.equal`): the attention module alone (float32, B 1/3/6, T 37/120, with
   and without key mask, dropout 0.0 and 0.1 with the same RNG seed) and the full `ScaleAwarePrior` (padded batch of
   lengths 50/31/7, all six outputs). Old path = the same module with `_BucketBias.apply` monkeypatched to the plain
   lookup, exactly as `bench_bias.LookupAttention` does.
2. **Gradient correctness.** The Function's output is `(h, T, T)`; the `[None]` and the broadcast add over `B` are
   outside it, so autograd reduces the batch and unsqueeze before the custom backward sees `grad` of shape
   `(h, T, T)`. `backward` returns `(h, K)` of `grad.dtype` for the table and `None` for the int64 buckets, which is
   what autograd requires for a non-differentiable index input. Results against the old lookup backward:
   float64 (B 3 T 37 masked; B 2 T 600 masked, dropout 0.1): max abs diff 0.0, i.e. exactly equal; float32: max abs
   diff 6e-8 to 7e-6 on gradients of magnitude 3-6 (about 1e-6 relative), which is float summation order. A
   non-contiguous `grad_output` (transposed) also matches (1.3e-5 on magnitude 37). All other parameters' gradients
   (`lin_q/k/v/o`, and in the full model `fc_v`, `fc_a`, `cma.*`, `fc`, `att`, `vid`, `node`) are bitwise identical,
   as expected: the gradient into `scores` does not pass through the Function.
3. **`scale_bias=false` (rule-14(g) ablation).** `requires_grad=False` on the table: `_BucketBias.apply` with no
   differentiable input produces an output without `grad_fn`, backward runs without error, `scale_bias.grad` stays
   `None`, the other parameters get gradients, `small_parameters()` is empty, and the optimizer groups in
   `src/qtl/train.py` 266-271 are therefore unchanged from the previous review. `torch.no_grad()` forward (the
   evaluator's per-video calls) works.
4. **Padding and bucket cache.** The key mask is applied after the bias as before (`masked_fill` to -1e9), so padded
   keys are still excluded; verified by the bitwise-equal outputs with mask in check 1. `buckets()` is untouched
   (same `(T, str(device))` key, clear at 64 entries, returns the same tensor object); `ctx.save_for_backward` holds a
   reference, so a cache clear between forward and backward cannot free it.
5. **Precision settings.** No `autocast`, `GradScaler`, `allow_tf32`, `set_float32_matmul_precision` or
   `torch.compile` anywhere in `src/qtl/` or the experiment, so the dense product runs in full float32 (PyTorch's
   default matmul precision). The only difference between the old and the new table gradient is summation order.
6. **Seed mixing (234 old backward, 2025 and 3407 new).** Given checks 1, 2 and 5, the training trajectories differ
   only by float32 rounding of 44 gradient entries per step, far below the seed-to-seed spread the 3-seed protocol
   averages over. The README statement in section 6 is accurate. `uoa-lab3` is at commit 6063408 (before e4e8d67)
   with jobs 1301 (HateMM) and 1302 (HateClipSeg) running, consistent with the commit message.
7. **Benchmark (`bench_bias.py`).** `LookupAttention.forward` sets `backbone._BucketBias.apply` to the plain lookup
   for the duration of `super().forward`, which reads the module-global `_BucketBias`, so the old graph is built
   (grad_fn `IndexBackward0`; the new path gives `_BucketBiasBackward`). Backward uses the recorded graph, so
   patching only during forward is sufficient; `run()` calls backward outside the patch and still exercises the old
   scatter-add. Restoring `orig` (the bound classmethod) as a class attribute works: `types.MethodType` has no
   `__get__`, so class access returns it unchanged and the call binds `cls`; after restore the new path's grad_fn is
   `_BucketBiasBackward` again. Weights are copied between the two modules before the comparison; forward equality
   via `torch.equal` and the relative table-gradient difference are the right quantities; warm-up before timing;
   dropout 0 removes the RNG from the timed path. Valid as a timing and equality check.
8. **GPU memory at the largest training T.** `LengthBatches` shrinks the batch until `bs * T_max**2 <= 32 * 512**2`
   (8.39M) or `bs == 1`; T itself is never capped. HateMM train split (744 videos) contains `non_hate_video_533` with
   5809 VGGish rows (T = 5809) and `non_hate_video_318` with 2384; 26 train videos exceed 512 s, 3 exceed 1024,
   2 exceed 2048. Both long videos train alone (`bs == 1`). Per backward call `_BucketBias` builds the int64
   one-hot `(T*T, 11)` and its float32 copy: at T = 5809, 2.97 GB + 1.48 GB = about 4.5 GB transient, twice per step
   (two attention directions, sequential, freed after the matmul). The attention's own tensors at that T are
   `(1, 4, 5809, 5809)` float32 = 0.54 GB each (a handful of copies across forward and backward). The step peak
   stays well under the 32 GB of the RTX 5090 on the lab partitions, and `bench_bias.py`'s largest shape (B 4,
   T 2000) needs 0.5 GB transient under `--mem=16G`. No OOM expected; not a numbers issue in any case.
9. **`launch/sbatch_bench.sh`.** 1 GPU, 2 CPUs, 16G, 20 min; runs the bench from the repository root with the
   `HateVideo` python; stdout goes to `runs/20261031_scale_aware_cma/bench_bias_<job>.out` (the directory exists on
   the lab hosts since the searches started). Nothing in it touches training.

## Notes (not bugs, no number changes)

- README section 6 says HateMM batches reach "T of several hundred seconds"; the training set actually reaches
  T = 5809 (one video) and 2384 (one video), both trained at batch size 1, which is where the lookup's sort-based
  scatter-add is most expensive. Worth stating the real maximum there.
- Speed only: `F.one_hot` materialises an int64 tensor before the float cast. Building the float one-hot directly
  (`torch.zeros(T*T, K, dtype=grad.dtype).scatter_(1, buckets.view(-1, 1), 1.0)`) would halve the transient memory
  and traffic. Same mathematics; not needed for correctness.
