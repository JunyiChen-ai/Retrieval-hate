# DeHate Self-Sealed Acquisition and Evaluation Preregistration

Status: **DRAFT FOR USER/PI CONFIRMATION — NOT ACTIVATED**  
Prepared: 2026-08-29  
Scope: acquisition and evaluation protocol only. No dataset application has been submitted and `DeHate.xlsx` has not been opened or parsed.

## 1. Objective and claims boundary

DeHate will be treated as a fifth, same-corpus weakly supervised hateful-video-localization benchmark. Each model may use only that corpus's training split and weak video-level training labels. Validation temporal labels may select all model and evaluation hyperparameters. Test temporal labels remain behind a sealed evaluator until a configuration is frozen.

This is a **self-sealed** cohort, not an author-hosted blind benchmark: the annotations are publicly reachable in the official repository. Any result may be called self-sealed only if the access controls and ledger below are followed. A later test-informed development run must be labeled `TEST-INFORMED` and cannot be presented as sealed confirmation.

## 2. Frozen upstream identity

Official repository: `https://github.com/Multimodal-Intelligence-Lab-MIL/DeHate`  
Frozen branch: `main`  
Remote HEAD on 2026-08-29: `8b3ecac98223ef953ad657b319cf90ffcff9ada1`

| Object | Git type | Bytes | Git object SHA-1 |
|---|---:|---:|---|
| `DeHate.xlsx` | ordinary blob, not an observed LFS pointer | 1,267,040 | `1f959059c6f96d1d46e580743f8a1b7ac02ec0e3` |
| `README.md` | blob | 9,816 | `2811a809aa90eefa32472f09b8eca43146ad34c6` |

The tree contains only `README.md`, `DeHate.xlsx`, and statistical images under `Images/`. No official split, feature archive, media URL manifest, evaluation server, `LICENSE`, or `.gitattributes` was observed.

The Git SHA-1 above is an object identity, not a content SHA-256. At the first authorized acquisition, the data steward must record content SHA-256, byte count, acquisition time, and upstream commit before any parsing. A mismatch from the frozen object requires a preregistered amendment; it must not be silently accepted.

## 3. Terms-of-use gate

The Microsoft application form states, in summary:

- access is exclusively for non-commercial research;
- users must comply with TikTok and BitChute terms and do not acquire rights to the original platform content;
- redistribution, publication, sharing, sublicensing, and exploitation of the dataset, subsets, or derived data are prohibited;
- collaborators must apply independently;
- access may be revoked;
- applications are processed weekly and require complete applicant, affiliation, PI/supervisor, institutional-email, Gmail, position, and purpose information.

This is a custom restricted-use agreement, not a standard open-data license. Before applying, the user/PI must confirm that the intended workflow is permitted, especially internal derived features, checkpoint collaboration, aggregate localization metrics, and publication of code that contains no dataset-derived artifacts. No agent may accept these terms on the user's behalf.

Official contact listed in the repository: `yuchen.zhang@essex.ac.uk`.

## 4. Roles and access ledger

Two logical roles are required:

1. **Data steward/evaluator:** may acquire and parse the annotation file, construct the split, and operate the sealed evaluator.
2. **Development team:** receives permitted media, train weak labels, validation temporal labels, and opaque test IDs; it does not receive test video labels, spans, rasterized GT, per-video test metrics, or error traces derived from test GT.

The steward maintains an append-only ledger with:

`UTC timestamp, person/service, action, purpose, upstream commit, file SHA-256, output artifact SHA-256, authorization reference`.

Opening the annotation workbook, exporting any labels, running the test evaluator, or changing evaluator code is a ledger event. Access by an unrecorded identity invalidates the sealed claim.

## 5. Acquisition and media QC before labels

After author approval, acquisition proceeds in this order:

1. Verify delivered files and record SHA-256/bytes without inspecting annotation values.
2. Establish `canonical_id = lower(platform) + ":" + dataset_source_id`, preserving the dataset-provided source ID as a string. Whitespace is stripped; no numeric coercion is allowed.
3. For every delivered media object, record media SHA-256, bytes, container, video/audio stream presence, ffprobe duration, average frame rate, decoder-open result, and acquisition UTC.
4. Assign exactly one predeclared status: `available_decodable`, `missing_from_delivery`, `corrupt`, `no_video_stream`, or `duplicate_exact`.
5. Exact byte-identical media form a group by media SHA-256. Author-provided aliases also form one group. Grouping uses no labels. Suspected perceptual duplicates are reported separately and are not moved between splits after labels are exposed.

Missing, corrupt, or streamless media are never converted to negative examples. The full delivered-ID cohort and every exclusion reason are reported. The primary localization cohort is the label-independent `available_decodable` cohort frozen before temporal labels are exposed.

## 6. Deterministic 70/10/20 group split

Public split salt:

`DeHate-selfsealed-v1-2026-08-29`

For each exact-duplicate/alias group, sort its canonical IDs bytewise and define:

```text
group_key = canonical_id_1 + "\n" + ... + canonical_id_n
digest = SHA256(UTF8(salt + "\n" + group_key))
u = big_endian_uint64(digest[0:8]) / 2^64
```

Assignment is:

- `train` if `0 <= u < 0.70`;
- `validation` if `0.70 <= u < 0.80`;
- `test` if `0.80 <= u < 1.00`.

The split is not balanced or repaired using platform, duration, hate type, target group, modality, video label, or temporal GT. If the authors provide an official split before any annotation is opened, that official split takes precedence and this fact is recorded as a pre-access amendment. An official split disclosed after labels are opened does not retroactively replace the frozen self-sealed split.

The steward publishes an artifact containing salted hashes of IDs, group membership hashes, split, media-QC status, and artifact SHA-256. Restricted source IDs and media are not published.

## 7. Supervision and isolation rules

- Training uses DeHate train media/features and train video-level weak labels only.
- Train temporal spans, frame labels, modality-at-segment labels, rationales, or target-at-segment labels may not enter loss, teacher construction, sampling, normalization, prompt selection, or diagnostics.
- Validation temporal GT may select epoch, regularization, fusion weights, calibration, thresholds, prompts, and all other hyperparameters.
- Feature extractors or external teachers must not be fitted using DeHate validation/test labels. Label-free transforms fitted on corpus data must state their split scope; the default is train-only.
- Test video or temporal labels cannot enter gradients, fitting, ECDF/calibration, configuration selection, threshold selection, cohort selection, or media repair.
- The final source commit, environment, checkpoint, configuration, validation decision, evaluator commit, and their hashes are frozen in `TEST_OPEN.json` before the sealed evaluator is invoked.

The evaluator accepts opaque test IDs and one finite score per timeline unit. It fails closed on missing IDs, extra IDs, duplicates, non-finite scores, or length mismatch. It returns only the preregistered aggregate table and confidence intervals, never per-video or per-frame GT-derived feedback.

## 8. Timeline and metrics

### Frame construction

The primary frame grid is 1 Hz. Index `t` represents `[t, t+1)` seconds. Ground-truth spans are clipped to `[0, media_duration)` and unioned. A frame is positive when its interval has non-zero overlap with a hateful span. Videos annotated non-hateful contain all-negative frames. Original native timestamp spans are retained by the steward for segment evaluation.

Any alternative boundary convention or duration source must be an amendment made before test opening and must be used identically on validation and test.

### Primary metrics

1. **Pooled Frame AP** over all eligible test frames. For this binary task, this is the quantity labeled `Frame mAP` in the project table; the report must state that it is one-class AP.
2. **Pooled Frame ROC-AUC** over the same frames and cohort.

Both metrics must always be reported together. Scores must retain their ranking values; no test-selected threshold is involved.

### Secondary metrics

- Within-video macro AP and ROC-AUC, with eligibility rules and undefined single-class videos reported explicitly.
- Video-cluster bootstrap confidence intervals: 10,000 resamples of videos with replacement, fixed seed recorded in the evaluator manifest, percentile 95% interval.
- For methods that emit intervals: temporal AP at IoU `0.1, 0.3, 0.5, 0.7` and their arithmetic mean. Segment metrics supplement rather than replace the two primary frame metrics.

Every report includes numbers of delivered/eligible/excluded videos, split counts, frame counts, positive-frame prevalence, undefined-macro coverage, and score coverage assertions.

## 9. Amendments and invalidation

Before test opening, a timestamped amendment may change this protocol only if it states the reason, exact old/new text, affected artifacts, and whether any annotation or test GT had already been accessed. It receives a new immutable hash.

After the first sealed test evaluation:

- no cohort, split, rasterization, metric, model, checkpoint, calibration, or selector change can be included in the same sealed result;
- further iterations form a new `TEST-INFORMED` track;
- the original sealed result and access ledger remain archived;
- a new sealed claim requires a genuinely untouched external cohort or author-operated blind evaluation, not merely a renamed split.

## 10. Activation checklist

- [ ] User and PI approve the purpose and restricted terms.
- [ ] Derived-feature, aggregate-result, and collaborator permissions are clarified if needed.
- [ ] Named steward and storage/access controls are assigned.
- [ ] This document is committed and its SHA-256 recorded before acquisition.
- [ ] Upstream HEAD/blob identity is rechecked.
- [ ] Application is submitted by the authorized human, not an agent.
- [ ] Media-QC and split manifests are frozen before temporal-label exposure.
- [ ] Sealed evaluator passes synthetic alignment/leakage tests.
- [ ] `TEST_OPEN.json` is frozen before the first test call.

