# V26 finite-RF pre-training fundamental audit

Status: **PASS for training launch; formal training unexecuted**. This audit used
only frozen THVL train features, train weak video labels for the one-epoch
compute path, and the frozen train-negative reference. Validation labels,
temporal GT, and test data were not accessed.

The migrated implementation preserves the load-bearing claims: every available
second receives an exact replacement counterfactual; all `T` targets are used;
the classifier remains `F=G+R`; the finite receptive field is relational; the
background decoder/reference is unchanged; and real/permuted/negative-mean arms
share a canonical epoch-0 state. Tests cover receptive-field indexing,
boundaries, missing modalities, hard-zero all-missing states, epoch-0 raw `G`
and zero effects, fast-path repeat determinism, gradient connectivity, absence
of Transformer state, and fail-closed migrated checkpoint identity.

Fast versus slow full-forward discrepancies are diagnostic, not a method gate:
the largest observed float32 effect difference was `8.91e-6` and largest
parameter-gradient absolute difference was `1.25e-5`. These are normal
operation-shape-dependent CPU/CUDA convolution roundoff; fast execution repeats
bit-exactly, dependency-cone values and gradients are structurally zero outside
the reachable field, and no mathematical effect definition was changed.

On RTX 5090 with cone chunk 64, `T=900` took `0.10542 s` and `T=3600` took
`0.40963 s`; time scaling was `3.886x` and peak-memory scaling `1.215x`.
All-314 preload used `1,305,114,112` of `33,670,758,400` bytes. The frozen
workload sums were `9,345` available seconds for the first 20 canonical videos
and `115,088` for all 314. One complete first-20, one-arm epoch took `1.98487 s`;
reference live verification/load took `244.90884 s`; the frozen formula projects
three arms times eight epochs plus one preload/reference load to `0.231049`
GPU-hours, below the 12-hour gate.

Authoritative artifacts:

- workload SHA-256 `89d6fef59937be3fd2f470282bf2702560c0f6ce6d4a5414cfae236befd8c406`;
- compute audit SHA-256 `075468eab6449f3d90e9d1a5d04fe9ac3746383dd4369a6c2d5af1a1455a0364`;
- first-20 benchmark SHA-256 `0738e0c7f26fb8137f57f95c95fc0e6e1c15b4577d61ae3c3f40abc4148d3306`;
- frozen compute protocol SHA-256 `ab247e3cba047ba951f7710421679b340d1c06d08eabea7697560d2485b248dc`.
