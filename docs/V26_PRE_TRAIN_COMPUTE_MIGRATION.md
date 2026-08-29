# V26 PRE-TRAIN COMPUTE MIGRATION: exact local-recompute CTW

Status: **frozen design; unexecuted**. This migration is recorded before any
formal V26 `F` training, validation temporal access, or test access. The frozen
negative-only decoder/reference artifacts remain reusable. It changes only the
implementation and architecture of the still-untrained relational classifier
`F` (and the matched independent probe `H`). The definition of a CTW score,
full-`T` interventions, weak-label losses, arms, seeds, selection rule, and all
gates remain unchanged.

## Why migration is necessary

The draft two-layer global Transformer evaluates `F(X_cf_t)` from scratch for
every second. A full attention forward is quadratic in `T`, so all `T`
interventions are cubic in video length. This is computationally mismatched to
hour-scale videos and is not needed for an exact causal replacement effect.

## Frozen replacement: finite-field relational convolution

The input construction is frozen exactly. Visual, audio, and text dimensions
are respectively `(512,128,768)`. Each modality has its own affine projection
`Linear(d_m,128,bias=True)` followed by GELU. Before projection, an unavailable
modality is represented internally by zero; after projection its 128-vector is
multiplied by its binary modality mask, so projection bias cannot leak through.
Concatenate the three masked 128-vectors and the three binary masks, giving
dimension `384+3=387`. Apply `P=Linear(387,256,bias=True)`. With
`q_i=available_any_i`, the initial state is exactly

`h_i^0 = q_i * (P(concat(proj_m(x_i^m)*mask_i^m, mask_i^1..3)) + pos_i)`,

where `pos_i` is the fixed width-256 sinusoidal position vector already frozen
by V26. No sequence-, batch-, or video-global normalization is legal.

`F` then uses exactly four deterministic residual blocks, with dilations
`d_l=[1,2,4,8]`. For each block and token, exactly

`u = DWConv1D(h, channels=256, kernel=3, dilation=d_l, zero_padding=d_l, bias=False)`

`v = h + u`

`h = q * LayerNorm(v + W2 GELU(W1 v + b1) + b2)`.

Here `W1` is `Linear(256,512,bias=True)`, `W2` is
`Linear(512,256,bias=True)`, and LayerNorm has normalized shape 256,
`eps=1e-5`, and affine weight/bias enabled. Dropout is zero. Multiplication by
`q` occurs after every block: an all-missing second is hard zero at every layer
and can neither retain nor propagate a hidden state. Padding outside the video
is fixed zero padding. The resulting symmetric receptive-field radius is

`R = 1 + 2 + 4 + 8 = 15 seconds`.

Thus each token representation is relational—it is a nonlinear function of up
to a 31-second multimodal neighbourhood—not an independent snippet head.

The scalar contribution and residual are exactly

`a_i = q_i * (w_a^T h_i + b_a)`

`R_theta(X) = (1/T_eff) * sum_i a_i`, where `T_eff=sum_i q_i`,

`F_theta(X,G) = G + R_theta(X)`.

Only `w_a` and `b_a` are initialized to exact zero. All projections,
convolutions, MLPs and LayerNorm affine parameters use the following unique
explicit initialization. Modules are constructed and named in this canonical
order:

`proj.visual, proj.audio, proj.text, input_projection, block0.dw, block0.w1, block0.w2, block0.ln, ..., block3.dw, block3.w1, block3.w2, block3.ln, contribution_head`.

Create a CPU `torch.Generator` and call `manual_seed(model_seed)`. Iterate the
canonical names above, never Python dictionary or device enumeration order. For
every Linear or depthwise-convolution weight before `contribution_head`, call
`torch.nn.init.xavier_uniform_(weight_cpu, gain=1.0, generator=generator)`;
initialize every such bias to exact zero (the depthwise convolutions have no
bias). Initialize every LayerNorm weight to exact one and bias to exact zero.
After all preceding modules, initialize contribution `w_a` and `b_a` to exact
zero without consuming generator state. Initialization occurs on CPU and the
completed state is then copied to the target device. Consequently epoch 0 is
bit-exact `F(X,G)=G` and every epoch-0
effect is exact zero, without requiring internal representations to be zero.
The legal fallback remains the frozen raw signed `G` repeated over the video.
For a model seed, `real`, `permuted`, and `negative_mean` are deep copies of one
canonical initialized state, and their epoch-0 state hashes must be exactly
equal before any optimizer is created. Probe `H` is independently initialized
with model seed `26027` by the same rule. Every process sets
`torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`,
`torch.backends.cudnn.benchmark=False`, and the frozen CUDA deterministic
workspace configuration before importing/initializing CUDA state.

## Exact full-T intervention without approximation

For target second `t`, construct the same counterfactual as the frozen design:
replace every available `x_t^m` by its decoder background `b_t^m`, leave masks,
positions, all other seconds, and `G` unchanged. Because the receptive field is
finite, only final token contributions with

`i in A_t = [max(0,t-R), min(T-1,t+R)]`

can change. Cache all base block activations and base `a_i`. For each `t`,
recompute the exact layerwise causal cone: at layer `l`, recompute precisely the
indices reachable from `t` through dilations `d_1..d_l`; untouched cached
activations remain graph-connected shared tensors (they are never detached).
Clone the base contribution vector, replace the entries in `A_t`, and run the
same full-vector masked sum. The following delta form is its algebraic
specification:

`R_cf_t = R + (1/T_eff) * sum_{i in A_t} (a_i_cf_t - a_i)`

and the unchanged frozen score

`e_t = T_eff * [(G + R_theta(X)) - (G + R_theta(X_cf_t))]`.

This is algebraically identical to a complete forward on `X_cf_t`; it is not a
linearization, influence-function estimate, sampled intervention, or detached
surrogate. Replacement backgrounds remain detached exactly as before. Every
available second is evaluated; unavailable seconds retain the existing mask
semantics.

One base forward costs `O(T * L * C^2)` because of pointwise channel mixing.
All exact interventions cost `O(T * R * L * C^2)` with fixed `R=15`, rather
than `O(T^3)`. In the sequence-length dimension this is linear, and memory is
`O(T * L * C)`. A batched-cone implementation may evaluate several targets at
once but may not alter arithmetic, padding, or target coverage.

## Objective and controls are unchanged

The total weak-label objective remains exactly:

- video BCE on the finite signed logit `F(X,G)`;
- for negative videos, mean Huber-to-zero over all clipped effects;
- for positive videos, coefficient `0.25` times softplus of the negative fixed
  fractional top-20% LME over all clipped effects;
- effects clipped once to `[-12,12]` after exact computation.

Real, permuted, and negative-mean arms retain the same data mappings,
backgrounds, masks, optimizer, steps, seed, and epoch selector. Permuted moves
the donor raw sequence/masks/OOF backgrounds together; negative-mean changes
only the replacement. The independent `H` probe uses the same migrated
convolution/pooling architecture, a distinct seed and parameters, BCE only,
and no CTW gradient. Therefore all matched-control and faithfulness definitions
are unchanged.

## Rejected alternatives

| Candidate | Length complexity for all effects | Main issue |
|---|---:|---|
| Global Transformer recomputation | `O(T^3)` | Not viable for hour-scale videos |
| Linear SSM with analytic deletion | `O(T*C)` or `O(T*C^2)` | Lowest cost, but bidirectional state/deletion algebra and nonlinear gating make exact replacement auditing substantially harder; higher implementation-risk before the kill pilot |
| Independent snippet head + pooling | `O(T*C)` | Cheap but removes the load-bearing relational mechanism |
| **Dilated finite-field convolution** | **`O(T*R*L*C^2)`, fixed `R=15`** | Chosen: exact, bounded, easily audited; possible risk from context limited to 31 seconds |

The unique priority is the dilated finite-field model. No architecture sweep,
receptive-field sweep, or post-result switch to an SSM is permitted in the THVL
kill pilot.

## Required pre-training verification and tiny benchmark

Before any formal training, implement one synthetic harness with seed `26026`,
float32, lengths `T={17,61,301}`, and real availability patterns. Run it once
on CPU and once on the pinned RTX 5090. For every target it must compare local
recomputation against a deliberately slow complete counterfactual forward:

1. base-path epoch-0 `G` and zero effects are bit-exact;
2. every counterfactual logit and clipped/unclipped `e_t` satisfies
   `atol=1e-6, rtol=1e-5` between fast and slow implementations;
3. for every element of every trainable parameter gradient, the single gate is
   `abs(fast_grad-slow_grad) <= 1e-6 + 1e-5*abs(slow_grad)`;
4. changes outside `A_t` exactly zero in value and gradient;
5. missing-modality and boundary targets included;
6. repeated executions of the fast path on the same runtime are bit-exact.

The fast implementation uses fixed ascending target order and fixed cone chunk
size 64. Within a target, contribution pooling, effect clipping, fractional LME,
per-video loss and batch loss retain their frozen reduction order. The benchmark
uses RTX 5090, float32, seed `26026`, deterministic algorithms, one process,
five warmup passes and ten measured passes. It records wall time, peak resident
memory, peak GPU memory, and maximum value/gradient errors, and additionally
runs fast-only at `T={900,3600}`.

The formal time projection includes reference loading plus exactly three arms
(`real`, `permuted`, `negative_mean`) times eight trained epochs; epoch 0 is a
frozen evaluation state and is not counted as a training epoch. Validation
prediction/evaluation time is excluded. Measure one complete epoch on the
frozen first 20 THVL train IDs (canonical train-manifest order), including
forward, all full-`T` effects, backward, optimizer step and data transfer.
Live-recompute `T_eff` for every video from the frozen availability masks in the
authoritative feature manifest, and freeze both canonical artifacts:
`sum_Teff_first20` and `sum_Teff_all314`, including the ordered per-video
`(opaque_id,T_eff,feature_record_sha256)` rows and their canonical SHA-256 root.
Missing/extra IDs, stale feature hashes, or a root mismatch is fatal. The
reference-load measurement is performed once against the frozen reference
manifest and records its path, bytes SHA, nested root/state hashes, wall seconds,
runtime identity and measurement-source SHA. The only
legal projection is

`hours = [measured_first20_full_epoch_wall_seconds * (sum_Teff_all314 / sum_Teff_first20) * 3 * 8 + measured_frozen_reference_load_seconds] / 3600`.

Video-count scaling such as `314/20` is forbidden. The projection must be `<=12` RTX
5090 GPU-hours. Runtime growth from `T=900` to `T=3600` must be no worse than
`5x`, and peak memory growth no worse than `5x`. Any numerical or compute gate
failure is a compute-design BLOCK, not permission to sample seconds or
approximate effects.

## Performance risk and falsification

The only substantive risk is that a 31-second neighbourhood misses discourse
dependencies spanning longer intervals. Global `G` still carries video-level
context, while the local effect intentionally asks whether a bounded temporal
neighbourhood contributes beyond that prior. The pre-registered THVL pilot
gates remain the falsification test: video-label gains over both controls,
within-video ROC/AP and paired V25 gains, shuffle contrast, duration robustness,
variance coverage, and independent-probe deletion faithfulness. A compute pass
does not count as a method pass.
