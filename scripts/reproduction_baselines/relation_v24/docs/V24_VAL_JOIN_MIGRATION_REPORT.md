# V24 validation join migration (conclusion path)

Status: implemented and synthetic-tested; no test labels/data opened and no model retraining performed.

The immutable approved train join remains `steward_join.py` at SHA-256
`d96b1a5f1306e1a5f48596e22e2ba6c31a91d33dd81f6b68e55dcc7053c57c62`.
Validation now uses a separate `steward_val_join.py`. It accepts only the exact
`v24_video_labels_v1` schema containing `(video_id, any_target_label)` and rejects
all extra fields, including temporal annotations, timestamps, intervals and frame
labels. Its v2 manifest separately binds evidence producer, join producer,
evidence manifest/config, label manifest, bags, and frozen global source hashes.

`selector.py` requires an explicit `v24_protocol_addendum_v1`. The addendum binds
the immutable old train protocol/artifacts to the new selector and validation join
source hashes. It compares train and validation **evidence** producer identity,
while validating the train and validation join producers independently. It does
not rewrite or impersonate the old training protocol.

The selector's epoch-0 exact global fallback, three-arm comparison, gamma/family
weight gates, and video-validation gates are unchanged. The output remains pending
the separately signed temporal steward gate before any test inference.

The evidence preflight additionally enforces the exact evidence-manifest schema,
exact disk/manifest/frozen-input ID sets, per-record file SHA, config/producer/
model-revision/prompt identity, frozen-window identity and order, contiguous bounds,
finite local/global scores, and aggregate video/window counts.

Synthetic fail-closed coverage includes wrong split, swapped evidence producer,
temporal/extra label fields, config/manifest/bags hash tampering, score tampering
without updating the record manifest, same-cardinality ID swapping, and rejection
of the legacy ID schema on the new validation path. All 19 Relation-V24 synthetic
tests pass (5 evidence + 4 join-v2 + 1 pipeline + 9 model/training tests).

The real no-retrain migration artifact is
`results/steward_private/thvl_bench/train314/v24_protocol_migration_v2/addendum.json`.
No real validation video labels were joined as part of this change.
