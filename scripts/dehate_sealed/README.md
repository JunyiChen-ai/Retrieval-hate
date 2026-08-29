# DeHate self-sealed tooling

This directory is data-blind by default. It performs no downloads and the
synthetic test does not locate, read, or parse any real annotation workbook.

Real annotations may only be handed to a steward-owned process after calling
`guarded_annotation_path(..., steward_mode=True)`. The isolated evaluator
accepts a steward-supplied decryptor, emits aggregate metrics only, and writes
a `TEST_OPEN` provenance event. The public manifest contains HMAC-SHA256 IDs;
the HMAC key and raw IDs remain private.

Duplicate groups are assigned from the exact preregistered bytes
`SHA256(UTF8(salt + "\\n" + "\\n".join(sorted(canonical_ids))))`, using the
first eight digest bytes as a big-endian uniform value and the frozen thresholds
`[0,.7)`, `[.7,.8)`, and `[.8,1)`; observed counts are archived and are never
manually rebalanced. Internal group hashes are derived from the sorted frozen
canonical IDs, so external row/group renaming and input order cannot change a split.
The test service requires a signed freeze manifest, writes an append-only
`TEST_OPEN` ledger, and defaults to 10,000 duplicate-cluster bootstrap draws.

Run the offline fixture with:

```bash
python scripts/dehate_sealed/test_synthetic.py
```
