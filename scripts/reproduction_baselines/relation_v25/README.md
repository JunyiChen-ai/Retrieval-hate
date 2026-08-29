# Relation V25 implementation

This directory implements the frozen executable addendum in
`relation_v24/docs/V24_FAILURE_ANALYSIS_AND_V25.md` without modifying V24 artifacts.

- `core.py`: ECDF logits, exactly replication-stable fractional top-20% LME,
  V25 model/loss, and deterministic 1 Hz reduction.
- `reference_builder.py`: canonical full and five-fold cross-fit negative references
  with fail-closed source/producer/file/aggregate hashes.
- `train.py`: four matched arms, three seeds and epochs 0..5; exact-length local
  permutation with moved-video/instance accounting.
- `selector.py`: shared real-selected epoch, controls at the same epoch, paired
  bootstrap, activation/permutation/temporal/shuffle gates and exact fallback.
- `inference.py`: label-independent local posterior or exact V16-global constant
  fallback, with full-coverage 1 Hz reducer.
- `seal.py`: explicit signed transition from validation pass to test-enabled state.

No real training or validation/test evaluation has been run. The only real-data
operation performed during implementation was a read-only train-input/reference
preflight; temporary reference files were deleted afterward.

The authoritative V24 producer defines `global_causal_score` as the arithmetic
mean of V16 packed `causal_continuous` raw Yes-vs-No margins (not a probability).
Therefore epoch-0/failure inference copies that stored IEEE-754 value directly to
every covered 1 Hz position; it must never pass through sigmoid or calibration.
