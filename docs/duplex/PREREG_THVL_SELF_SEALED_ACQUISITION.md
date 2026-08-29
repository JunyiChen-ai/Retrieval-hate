# THVL-Bench Self-Sealed Acquisition Preregistration

Status: acquisition activated by an independent data steward; test GT remains sealed.  
Frozen: 2026-08-29.

## Frozen source

- Hugging Face dataset: `THVL/THVL-Bench`
- Revision: `5ea20ec4074dea9d3419e88fea944313ab25818d`
- Annotation: `THVL-Bench.csv`, 71,604 bytes
- Annotation SHA-256: `2ba5127eb05bee6e614ff4e6da511422eb4d2bac830281f55edb860daeedf5a7`
- Resolve ETag: `ffb9cf062d86a3eaf001fd0f70d0e742f4bcce38`
- README SHA-256: `858b271afbf95abf99d214102cf7c87a1409eac40665bbfc0fb7682290faee19`

License metadata is inconsistent at this revision: YAML/API tags say CC BY 4.0, while the README body twice says CC BY-NC 4.0. This project adopts the stricter non-commercial interpretation unless the publisher clarifies it. Raw source-platform content remains subject to its platform terms.

## Target and split

The preregistered target is the union of `Verbal Abuse`, `Hate`, and `Bias`. Other harmful categories are negative for this narrow target, not missing labels. This mapping was fixed from the published 11-class schema before model development.

Canonical IDs are `youtube:<source-id>` or `bilibili:<source-id>`. Mechanically numbered excerpts of one source are placed into a common label-free source group. For each group:

```text
group_key = newline-joined, bytewise-sorted canonical IDs
u = uint64_be(SHA256("THVL-Bench-selfsealed-v1-2026-08-29\n" + group_key)[0:8]) / 2^64
train if u < .70; validation if .70 <= u < .80; test otherwise
```

There is no label balancing, cohort repair, or resampling. Exact split counts are archived as outcomes rather than targets.

## Isolation

- The raw CSV, ID/media map, HMAC key, encryption key, and temporal test GT live only under steward-private storage with mode 0600.
- Development manifests contain HMAC-SHA256 IDs, group hashes, split membership, and media availability. Only train exposes the narrow-target weak video label.
- Validation temporal GT is accessible only through an aggregate validation evaluator with an immutable query ledger.
- Test temporal GT is encrypted. Test evaluation requires the same signed freeze and atomic `OPEN_STARTED/COMPLETED/FAILED` ledger used by the audited DeHate sealed evaluator.
- The evaluator additionally requires a steward-frozen, exact-cohort media-duration manifest. Until media durations are available from label-independent media QC, frame rasterization and all formal validation/test evaluation remain blocked.
- Test labels cannot enter training, normalization, calibration, feature extraction, threshold/configuration selection, cohort selection, or media recovery.

## Media acquisition

The initial audit uses Hugging Face repository metadata and HTTP HEAD only. It does not bulk-download long videos. Every annotation ID is assigned `unique_path`, `missing_path`, or `ambiguous_path` before labels are used for model selection. Missing media are never negative. Any later media acquisition records revision/path, bytes, ETag, content SHA-256, decoder status, duration, streams, and UTC in the steward ledger. The eligible cohort is frozen from label-independent media QC.

## Metrics

Primary metrics follow the current project table: pooled 1-Hz Frame AP (`Frame mAP` for this one-class target) and pooled Frame ROC-AUC. Secondary metrics are within-video macro AP/ROC with eligibility counts, video/source-group cluster bootstrap with 10,000 draws and 95% percentile intervals, and temporal AP at IoU 0.1/0.3/0.5/0.7 for interval outputs. Scores must have exact opaque-ID coverage, exact timeline length, and finite values.

## Amendments

Before test opening, changes require a timestamped amendment listing old/new protocol text and artifact hashes. After test opening, any changed model, cohort, mapping, metric, or selector is `TEST-INFORMED`; it cannot replace the original sealed result. No per-video/test-label diagnostics are returned to development.
