# C01 retrieval equivalence diagnostic preregistration

**Status:** `DEBUG_READY_NOT_RUN`. This is a read-only diagnostic for job `13730`; it does not modify A0 and does not authorize submission, retry, CONTINUE/KILL, C02, or any scientific claim.

## Trigger and static audit

Job `13730` halted at `real/endpoint_std with-null/remove-null retrieval mismatch` before result/decision publication. The stderr does not identify whether neighbors, similarities, or scores first diverged.

Static inspection of `retrieval_without_registered_null` found no evident index-construction bug:

- the full and reduced searches each construct a fresh `faiss.IndexFlatIP` and add their memory once;
- the removal mask excludes original train index 355;
- `original_indices = flatnonzero(keep)` is the correct local-to-original map;
- `original_indices[reduced_neighbors]` correctly maps local FAISS indices back to original train indices;
- reduced labels use the same removal mask.

Therefore mapping bug, raw FAISS tie order, candidate-set/boundary ties, float variation after changing matrix shape, and unexpected null selection remain competing explanations. No explanation is preselected.

## Frozen identity and access

- Run/schema: `C01-RETRIEVAL-EQUIV-PROBE-v1` / `c01_retrieval_equivalence_probe_v1`
- Config: `configs/c01/c01_retrieval_equivalence_probe_v1.json`
- Analysis: `scripts/analysis/c01_retrieval_equivalence_probe.py`
- Wrapper: `scripts/slurm/c01_retrieval_equivalence_probe.sbatch`
- Exclusive namespace: `artifacts/c01_policy_contrastive/v2/retrieval_equivalence_probe/C01-RETRIEVAL-EQUIV-PROBE-v1/`
- Artifact: `retrieval_equivalence_probe.json`
- Manifest SHA256: `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305`
- Zero-probe SHA256: `bee4964ce7e4ca81cfdb72c3859f78196568badf982aef587bc14ee6dbe63526`
- Registered tuple: HateMM/train, `hate_video_95`, row 355, expected-label-integrity-only 1, standard image/text exact-zero.

The only cache inputs are exact approved-manifest HateMM standard-L24 train and dev_seen files. The probe constructs the A0 `endpoint_std` fused view from image/text. Test-like paths are rejected before opening, cache SHA256 is checked before `torch.load`, and no feature vector, transcript, full neighbor matrix, or full ID list is serialized.

## Frozen comparisons

Both searches use a fresh CPU `IndexFlatIP`, A0-equivalent float32 normalization, top-20 signed-cosine rank weighting and cutoff zero. The probe reports:

- with/remove train sizes and the complete mapping hash, formula check, and limited boundary examples;
- every occurrence of the registered null in top-20, with limited query IDs and ranks;
- raw FAISS neighbor element differences, per-query set/order differences, hashes, and limited first examples;
- float32 similarity dtype/shape/byte hashes, element byte-diff count, maximum absolute and ULP differences, and limited first examples with float32 hex;
- score/prediction/metric differences and hashes;
- adjacent ties, one-ULP ties and rank-20/rank-21 gaps;
- a second comparison after deterministic sorting by `(-FAISS float32 similarity, original train index)`.

The diagnostic classification is frozen:

1. invalid local-to-original formula → `MAPPING_BUG`;
2. null in full top-20 → `REGISTERED_NULL_SELECTED_IN_TOP20`;
3. raw order differs but stable order agrees → `RAW_FAISS_TIE_ORDER`;
4. stable identities agree but similarity bytes differ → `FLOAT_VARIATION_WITH_STABLE_NEIGHBOR_IDENTITY`;
5. any remaining raw neighbor disagreement → `MIXED_OR_FLOAT_VARIATION`;
6. only similarity bytes differ → `FLOAT_VARIATION`;
7. otherwise → `NO_MISMATCH_REPRODUCED`.

This classification is diagnostic only. Any future A0 repair requires separate review and authorization.

## NO-GO repair

The initial probe draft simplified endpoint construction after the first modality normalization and was not admissible. The repaired probe imports the current A0 v2 analysis module as a pure-function reference and constructs `endpoint_std` exactly as A0:

1. raw image/text row L2;
2. `fuse_modalities` re-L2 of each already-normalized modality block;
3. concatenate image/text blocks;
4. final fused row L2.

An independent local implementation repeats those exact four steps. Dtype, shape and C-order bytes must match the imported A0 result for both train and dev_seen; source/key hashes and construction steps are recorded. A mismatch halts.

The ordered float32 ULP mapping is now monotone across signs: negative encodings use `bitwise_not(raw_uint32)`, nonnegative encodings use `raw_uint32 xor 0x80000000`. Thus `-0.0` and `+0.0` are distinct adjacent codes and cross-sign distance passes through both. NaN and either infinity are forbidden. Before cache access, runtime self-checks validate known bit patterns for `-1`, negative minimum subnormal, `-0`, `+0`, positive minimum subnormal and `+1`, including signed-zero distance 1 and cross-zero minimum-subnormal distance 3.

`RAW_FAISS_TIE_ORDER` is allowed only when the deterministic stable neighbor arrays **and** stable similarity bytes both match. Stable identity with unequal similarity bytes is `FLOAT_VARIATION_WITH_STABLE_NEIGHBOR_IDENTITY`; any remaining raw neighbor disagreement is `MIXED_OR_FLOAT_VARIATION`.

## Execution/resource boundary

The registered wrapper exactly matches failed A0 job 13730's compute/thread shape: CPU-only, `8 CPU / 32 GB`, `OMP/MKL/OPENBLAS/NUMEXPR=8`, conda `HateVideo`, no `--time`, GPU, force, disk guard, or TARGET mutation. It records the thread environment, exclusive-creates its namespace and refuses reruns. Current state is `DEBUG_READY_NOT_RUN`; no Python execution or SLURM submission occurred while repairing this probe.
