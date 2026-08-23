# CVoI acquisition implementation appendix v1

> **Status: DESIGN ONLY / NOT EXECUTABLE (2026-08-08).** Companion to
> `research-wiki/EXP_cvoi_acquisition_prereg.md`. This appendix freezes the
> implementation contract, not any result. No candidate metric may be computed until the
> completeness ledger in §18 is entirely `PASS`, an independent reviewer approves it, and
> the payload is frozen. Placeholder hashes below are deliberately unresolved; replacing a
> placeholder with the digest of a newly generated pre-metric asset is an asset-registration
> act, not permission to alter an algorithm.

## 1. Version, authority and immutable scope

- Study ID: `CVOI-ACQ-v1`.
- Primary dataset: HateMM train, `n=744`; confirmation: HateMM val, `n=107`.
- Test is absent from every manifest, glob and loader allow-list.
- Temporal grid: K=30 windows, `I_k=[kD/30,(k+1)D/30)`, final interval right-closed.
- Action types: one midpoint OCR action and one four-frame dense-visual action per window.
- Primary budget: 10% of predicted per-video joint always-acquire incremental cost.
- Split seeds: `{20260811, 20260812, 20260813}`.
- Refit seeds: `{0,1,2}`.
- Inner split seed for outer split seed `s` and outer fold `f`:
  `1000003 + 1009*s + f`, evaluated in unsigned 64-bit integer arithmetic.
- Bootstrap: 10,000 replicates, seed `20260819`.

The preregistration controls scientific endpoints and gates. This appendix controls bytes,
schemas and algorithms. A conflict is resolved in favour of the preregistration and triggers a
pre-metric amendment; it is never silently resolved in code.

## 2. Known source assets and registration table

The freeze manifest contains one record per file with canonical repo-relative path, byte length,
SHA256, schema version and role. Current known inputs are:

| role | path/pattern | known contract | freeze state |
|---|---|---|---|
| labels/text | `data/gt/HateMM/{train,val}.jsonl` | `id,text,label`; exact 744/107 IDs | register digest |
| duration only | `data/gt/HateMM/hate_spans.json` | only `duration` may enter action building | register digest; span keys denied |
| OCR windows | `data/OCR/HateMM/ocr_windows_K30.jsonl` | 30 rows/video, midpoint action output | register digest |
| OCR video | `data/OCR/HateMM/ocr_video.jsonl` | QC only, never an action input | register digest |
| OCR metadata | `data/OCR/HateMM/meta.json` | run provenance; run-level cost only | register digest |
| old dense train | `data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt` | flat `[744*30,1024]`, four-frame mean, parent index | diagnostic/parity only |
| cheap window visual | newly generated train/val midpoint-CLIP cache | one independently encoded midpoint frame/window; legal pre-action input | generate, audit, register digest |
| base features | selected HateMM train/val whole-video Qwen feature cache | `ids,img_feats,text_feats,labels` | exact path chosen by §8 rule, then register |
| strict-22 IDs | steward-created `strict_ocr_no_speech.ids` | sorted IDs + trailing newline | digest fixed below |
| broad-28 IDs | steward-created `ocr_no_speech_flag.ids` | sorted IDs + trailing newline | digest fixed below |

The two required lineage digests are:

```text
strict-22  058aac5fa3bc8360429fe99331fdbfbb4dc3c025de14740827c86aaeddf1f317
broad-28   9d268f469bb0e8731d2681b9ceaab5fa3db6db14cc03ba4773f608d690cafe74
```

No unresolved digest is guessed. `asset_registry.json` is generated only after all assets exist,
then independently rehashed. The metric executable refuses `PENDING`, missing or extra inputs.

## 3. Canonical identifiers and row ordering

`video_id` is the exact string in the HateMM split JSONL. `window_id` is integer 0--29.
`action_id` is ASCII `video_id + ":" + action_type + ":" + zero-padded-window`, where action type
is `ocr` or `dense4` and window is `%02d`. `frame_id` for dense is
`action_id + ":" + frame_slot`, slot 0--3.

Every JSONL is UTF-8, compact canonical JSON (`sort_keys=True`, separators `(',',':')`), one record
per line, LF endings and a final LF. Ordering is `(video_id bytewise UTF-8, action_type order
ocr<dense4, window_id, frame_slot)`. Float artifacts use little-endian float32 and include shape and
dtype in their sidecar. NaN/Inf is forbidden.

## 4. Group construction

### 4.1 Sources

Grouping is built on HateMM train only. Val is never used to define train folds. The generator may
read raw train videos, train transcript/text, and trusted creator metadata if a creator field is
actually present in an audited source. It emits `group_sources.jsonl` with:

```text
video_id, video_sha256, duration_ms,
phash30[30:uint64], chromaprint|null,
normalized_transcript_sha256, transcript_minhash128[128:uint64],
creator_namespace|null, creator_id|null,
decode_status, source_paths_sha256
```

Transcript normalization is Unicode NFKC, lowercase, whitespace collapse, punctuation removal;
MinHash uses character 5-grams, 128 permutations seeded `20260810`. `phash30[k]` is 64-bit DCT pHash
of the RGB midpoint frame of window k after letterboxed 256x256 conversion. Chromaprint is generated
from mono 16-kHz PCM and compared using its library's normalized fingerprint similarity. Creator
edges are allowed only when namespace and ID are direct source metadata, never inferred by an LLM.

The audited HateMM train JSONL has exactly `id,label,text` and no trusted creator namespace/ID;
creator fields are therefore deterministically null in v1 and creator edges are unavailable. The
frozen HateVideo environment also has no `fpcalc`; Chromaprint is deterministically null in v1 and
rule 4 creates no edge. Record both absences in `group_capabilities.json`. Installing a tool,
inferring a creator, or enabling either edge source requires a pre-metric versioned rebuild; the
implementation may not silently change capability mid-run.

### 4.2 Frozen edges and components

An undirected edge exists iff at least one condition holds:

1. identical non-empty `video_sha256`;
2. identical trusted `(creator_namespace,creator_id)`;
3. at least 24 of 30 same-index pHash pairs have Hamming distance `<=6` **and** estimated transcript
   5-gram Jaccard from MinHash is `>=0.80`;
4. Chromaprint similarity `>=0.95` and duration ratio `min(D1,D2)/max(D1,D2) >=0.90`.

Missing features never create an edge. Components are union-find transitive closures; root/group ID
is the lexicographically smallest member ID. Emit sorted `group_edges.jsonl`, `group_components.json`
and an audit table with group-size/class counts and the triggering rule. No manual merge/split is
permitted after labels or metrics are inspected. A pre-metric blinded integrity review may reject
false linkage and must then version the algorithm and rebuild everything.

### 4.3 Group-aware folds and fallback

For each outer seed, sort groups by SHA256(ASCII decimal `seed` + literal `||` + UTF-8 `group_id`),
then assign them greedily to the fold
minimizing, lexicographically: the sum **over all folds** of normalized absolute positive- and
negative-count deviations after the hypothetical assignment, the corresponding all-fold absolute
total-count deviation, current target-fold size, fold index. The
normalizers are the requested per-fold positive and negative targets, each floored at 1. This joint
objective is required: the earlier draft's positive-deviation-first lexicographic objective sends
negative-only groups to the fold whose positive count is already closest to target and fails even
on perfectly balanceable synthetic data. Entire groups move together. Request 5 folds. If any query
fold lacks a class, repeat the same algorithm at 4 then 3 folds. Failure at 3 is
`HALT_GROUP_FOLDS`.

Inner folds use the same algorithm on the outer-train groups with the derived seed, requesting
4 then 3 then 2 folds. Failure at 2 is `HALT_INNER_FOLDS`. The algorithm never redraws at the same
fold count. Save all assignments before model fitting. Assertions cover group disjointness, exact
query coverage and both-class presence.

## 5. Action schemas

### 5.1 OCR(k)

The existing midpoint OCR observation is canonicalized into `ocr_actions.jsonl`:

```json
{"schema":"cvoi-ocr-action/1","video_id":"...","action_id":"...:ocr:00",
 "window_id":0,"window_start_s":0.0,"window_end_s":1.2,"sample_t_s":0.6,
 "texts":[{"text":"...","conf":0.93,"bbox":[[0,0],[1,0],[1,1],[0,1]]}],
 "engine_status":"ok","output_sha256":"..."}
```

OCR filtering is confidence `>=0.50` and stripped text length `>=2`. Sort detections by bbox top
coordinate then left coordinate, stable on original order. Normalize NFKC, collapse whitespace,
deduplicate exact normalized strings within a window, concatenate with `" [SEP] "`. Empty output
maps to a learned `EMPTY_OCR` token. Long text is tokenized by the frozen text encoder without
discarding a prefix: chunks of 75 content tokens plus boundary tokens, non-overlapping; encode all
chunks and attention-mask mean-pool, then L2-normalize.

The OCR engine/model/language and encoder are copied from the existing cache metadata and pinned in
the asset registry. If metadata cannot uniquely identify their bytes, OCR outcomes must be rebuilt
under a newly pinned engine before G0. Existing run-level timing is never expanded into fake
per-action timing.

### 5.2 DENSE(k)

For window `[l,r)`, sample four timestamps
`t_m = l + (m+0.5)*(r-l)/4`, `m=0..3`, clipped to the last decodable timestamp. Decode each frame
independently, record requested and actual timestamp, and encode it with frozen
`openai/clip-vit-large-patch14-336`, vision `pooler_output`, float32. Store:

```json
{"schema":"cvoi-dense-frame/1","video_id":"...","action_id":"...:dense4:00",
 "window_id":0,"frame_slot":0,"requested_t_s":0.1,"actual_t_s":0.101,
 "decode_status":"ok","width":1280,"height":720,"feature_row":0,
 "frame_rgb_sha256":"...","feature_sha256":"..."}
```

`dense_frames.f32` is `[N_actions,4,1024]`. The action outcome is the eps-safe L2-normalized mean of
the four stored vectors. A failed slot repeats the nearest successfully decoded slot within that
action and records `fallback`; if all four fail, return learned `EMPTY_DENSE`. The old K30 mean must
**not** be expected to match the new action mean: the old producer sampled 120 frame indices by
global `linspace` including the video endpoints, whereas the new action uses four interior
timestamps per duration-defined window. For provenance only, a separate parity routine must replay
the old producer's exact frame-index schedule and match the old cache within max absolute error
`<=5e-5` on all non-guard rows. Failure blocks an old-cache comparability claim but does not alter or
replace the registered new action. Val dense actions must be generated under the identical new
action contract.

### 5.3 Joint universe

Each complete video has exactly 60 candidate actions in fixed order: all 30 OCR then all 30 dense4.
An unavailable action remains in the universe, consumes its measured attempted cost, and yields the
empty token. Action availability cannot be inferred from its cached output before purchase.

## 6. Synchronized per-action cost protocol

Cost calibration and outcome generation are separate artifacts. Hardware, driver, CUDA/cuDNN,
engine, container/conda lock, clocks/power mode and batch size 1 are recorded. Both candidate and
baselines use the same node/GPU model; historical timings are context only.

For each extractor:

1. execute 100 unrecorded warm-up actions drawn from train only;
2. measure each train and val action five times in a seed-`20260814` randomized order;
3. immediately before timing call `torch.cuda.synchronize()` (if GPU), start both
   `time.perf_counter_ns()` and CUDA events, perform decode/preprocess/inference/postprocess, then
   synchronize and stop;
4. record wall ns, CUDA ms, decode ns, preprocess ns, inference ns, postprocess ns, allocated bytes,
   retries and status for each repetition;
5. the action cost is the median of repetitions 2--5; repetition 1 is a cache warm-up diagnostic.

Policy scoring, acquired-output encoding, retrieval and DP overhead are separately timed by the
same rule per acquisition step and charged to that policy. The common base-model pass is reported
but excluded from incremental action cost because every arm pays it. `cost_actions.jsonl` contains
one row per action and raw repetitions; `cost_summary.json` contains distributions, never substituted
values.

The deployed pre-action cost estimate is a gradient-boosted regressor limited to cheap covariates:
duration, source resolution from container headers, action type and window index. Per-window decode
status, retry count, actual timestamp, pixels returned and any metadata learned by seeking/decoding
the candidate window are forbidden cost features because they execute or reveal the action.
Hyperparameters are fixed: squared error, 100 trees, depth 3, learning rate
0.05, min leaf 10, seed 20260815. It is fitted fold-internally. Missing estimate uses the outer-train
median for that type.

For video i and budget fraction b, the deployable budget is
`B_i(b)=b*sum_a c_hat_i(a)`, using predictions available before acquisition. It is not defined using
the query's realized all-action cost. Feasibility uses `ceil(c_hat/0.1ms)` and reserves the predicted
policy overhead for one more decision. Figures and Pareto inference use realized costs.

## 7. Base representation and acquired-evidence classifier

### 7.1 Base selection without candidate metrics

The base is the latest already-frozen HateMM Qwen whole-video feature family that has both train and
val caches, exact split-ID agreement, and the highest **pre-existing, pre-CVoI** train-OOF macro-F1
recorded before 2026-08-08. Selection is made by a provenance steward from dated campaign records,
not by rerunning or evaluating a CVoI arm. The exact record, feature paths, dimensions and hashes are
inserted into `base_selection.json` and independently reviewed before metrics. If the ordering is
ambiguous, use frozen Qwen2.5-VL-7B (non-LoRA) rather than make a new comparison.

Let base vector `z=[img_feats;text_feats]`, separately L2-normalized and concatenated. Base features
are frozen; encoder fine-tuning is out of scope.

The legal cheap window feature is a separate CLIP embedding of the same midpoint RGB frame used to
define `OCR(k)`, generated without OCR and stored with requested/actual timestamp and frame hash.
It is available to every arm before acquisition, and its extraction/encoding cost is part of the
common base pass reported for all arms. The old four-frame K30 mean is **not** a cheap feature: it is
equivalent to observing the registered dense action and is denied until `DENSE(k)` is purchased.
No window-ASR extractor is assumed. Whole-video transcript/title features may be combined with the
cheap midpoint feature, but action-specific OCR, dense or decode-result fields remain forbidden.

### 7.2 Action encoders

- OCR: frozen, registry-pinned multilingual text encoder; output dimension from its config, projected
  by `Linear(d_text,256)`.
- dense4: 1024-d outcome, projected by `Linear(1024,256)`.
- add learned 256-d action-type, sinusoidal window-position and acquired-order embeddings.

All projected tokens use LayerNorm, GELU and dropout. No gold span/Claude feature exists.

### 7.3 Shared state classifier

`z` is projected by `Linear(d_z,256)+LayerNorm+GELU`. Acquired tokens plus a learned CLS token pass
through a 2-layer pre-norm Transformer encoder: width 256, 4 heads, FFN 512, GELU, dropout selected
below. Concatenate state CLS and projected z (512-d), then
`Linear(512,256)-GELU-Dropout-Linear(256,1)`. Empty `S` contains CLS only.

The state classifier is trained on the frozen state distribution (§9) with video BCE. One shared
classifier is used to evaluate all selectors within a refit. B0 and B1 also receive separately
trained specialist heads of identical architecture as strong no/all controls; the policy-comparison
table uses the shared classifier so selection, not head specialization, is isolated.

Every binding G1--G4 delta, gain-recovery denominator and oracle comparison uses the shared state
classifier evaluated at the corresponding empty, acquired or all-action state. The specialist B0
and B1-O/B1-D/B1-J heads are reported as descriptive strong controls only and cannot substitute
into a binding gate after results are observed.

## 8. Models, grids and training

All learned modules use AdamW, gradient norm clip 1.0, batch size 32 videos, deterministic PyTorch
algorithms, mixed precision disabled for head fitting, maximum 60 epochs. Grid:

```text
learning_rate   {1e-4, 3e-4}
weight_decay    {0, 1e-4}
dropout         {0.1, 0.3}
utility_loss_w  {0.5, 1.0}
ranking_loss_w  {0, 0.2}
```

Classifier configs ignore the last two fields. Utility regression uses Huber loss (`delta=1`) on
clipped targets `[-5,5]` plus optional pairwise logistic ranking loss within state. Standardize
utility targets using inner-train mean/std only.

Checkpoint/epoch selection is pooled inner-OOF macro-F1 at the primary budget, with lower realized
cost then lower epoch then lexicographic config ID as ties. Within a fit, early stopping monitors
the fit's legal validation fold macro-F1 with patience 8, minimum delta `1e-4`; the selected epoch for
outer refit is the rounded median best epoch across inner folds. Outer refit trains exactly that many
epochs and has no validation callback.

Threshold candidates are every distinct pooled inner-OOF score, `0.5`, and
`nextafter(max_score,+inf)` so the all-negative operating point is representable. Apply prereg
§3.3's macro-F1, distance-to-0.5 and smaller-threshold tie order exactly.

Arm definitions:

- B2 random: Philox keyed by `(split_seed,refit_seed,video_id,budget,draw_id)`, 20 draws. Each draw
  is a complete independent policy execution with its own prediction and realized cost. Compute a
  complete-run metric for each draw, then average the 20 metrics/costs to define B2; never average
  per-video probabilities across counterfactual acquired sets. Prediction/traces include `draw_id`,
  and bootstrap resampling preserves each draw's complete paired rows.
- B3 uniform: alternates types and selects window indices by farthest-point distance from already
  selected timestamps, initial index 15; deterministic.
- B4 salience: cosine distance of the legal cheap midpoint visual feature from the mean of the 30
  legal cheap midpoint features, standardized on outer-train and ranked descending. It uses no
  window ASR/OCR/dense outcome. Ties use lower window index, then action-type order.
- B5 uncertainty: expected absolute change proxy from the classifier gradient norm with respect to
  the type-specific empty token, computed without action output.
- B6 coarse per-video router: a logistic gate on z predicts whether the action-universe-specific
  B3 uniform feasible package has positive inner-OOF utility. If positive, execute that complete
  precomputed B3 package up to the video's budget; otherwise execute no action. It never attempts
  the 30- or 60-action full-information bundle at a fractional per-video budget.
- B7 MultiHateLoc-style: one `Linear(cheap_window_dim,128)-tanh-Linear(128,2)` score for window/type;
  hard top-k/feasible selection, video BCE only.
- B8 relevance: same B7 architecture; privileged version targets train-fold Gate-C required-type
  labels where available, cheap version targets OCR-presence from training outcomes. Neither can
  access query OCR presence.
- B9: same policy architecture as B10, targets singleton `u(a|empty)` and sums no interactions.
- B10: state/action scorer `MLP([stateCLS;cheap_action;type;position;c_hat;remaining],
  widths 1024->512->128->1, GELU, dropout)` trained on `u(a|S)`.
- B11: fit a ridge regressor (`alpha` in `{0.1,1,10}` inner-selected) to singleton
  `u(a|empty)` targets using the same legal pre-action features. On an outer query, predict every
  singleton utility once at the empty state, freeze those values for the whole trajectory and order
  feasible actions by predicted utility/estimated cost. It never consumes a realized utility or
  query label and is non-oracle.
- B12: B10 scores followed by the exact cost-aware solver in §10.
- O1/O2: explicit oracle namespace and label-access guard; no weights or thresholds flow back.

Width-matched control replaces the B10 final score with a shuffled-within-training-fold utility
target, frozen shuffle seed 20260816, retaining identical parameter count and optimization.

## 9. State distribution and target generation

Targets are generated only for an inner-held-out video by a classifier fitted without its group.
For every held-out video create states from four generators:

1. empty state: one state;
2. random prefixes: 8 trajectories, action order Philox-keyed by video and seed 20260817;
3. uniform/salience/B5/B7 prefixes: one trajectory per policy;
4. on-policy DAGGER: three rounds. Round 0 fits for 10 epochs using generators 1--3 only. Rounds
   1--3 each generate exactly one trajectory per eligible video from the preceding round's frozen
   policy, append its deduplicated states, then fit a fresh policy from the same initialization for
   10, 20 and 30 epochs respectively. Each trajectory for inner fold `h` is generated by a policy
   fitted without any target row or group from `h`; targets use that fold's already frozen
   cross-fitted state classifier. There is no checkpoint-dependent or metric-dependent extra round.

For every state retain prefix lengths `{0,1,2,4,8}` that fit the 50% budget and the terminal prefix.
At each retained state evaluate at most 12 candidates: all feasible selected next actions from the
four generators, plus seed-20260818 uniform samples without replacement to reach 12. If fewer exist,
use all. Compute before/after log-loss for each candidate. No candidate is selected using its utility
before inclusion in this target set.

Deduplicate identical `(video_id,S,a)` rows by summing the contributing generators' mass and retain
the complete generator-provenance list. Each generator first receives equal total mass; after
deduplication, normalize the aggregated row weights to sum to one within each video. Thus a state
reached by multiple generators retains their combined mass while every video has equal total mass,
so long videos or many states cannot dominate. Save `utility_targets_inner_<...>.parquet` with fold,
model and source hashes. Outer-query labels are never used to generate a training target.

## 10. B12 heterogeneity and solver

On cost-calibration records within each outer train partition, costs are heterogeneous only if,
for at least one live action type, both hold: coefficient of variation `>=0.10` and percentile ratio
`p90/p10 >=1.25`. Evaluate this rule before any performance metric and require it in at least 4/5
outer folds for all three split seeds. Otherwise B12 is automatically `NOT_APPLICABLE`, B10 uses
sequential feasible top-k, and the paper is forbidden to say knapsack.

When applicable, B12 maximizes the sum of frozen B10 scores for the current decision batch subject
to predicted cost. Convert costs and budget to integer ticks by ceiling at 0.1 ms. Use exact 0/1
dynamic programming with action-index lexicographic tie-breaking; with at most 60 actions, return the
highest score, then lowest cost, then lexicographically smallest selected action list. Sequentially
recompute scores after executing the chosen first action and solve again. Compare with greedy
score/cost and cost-blind top-k. Any DP state count above 5,000,000 halts B12 rather than silently
approximating it.

## 11. Nested selection and confirmation execution

The binding outer procedure is grouped **5 outer x 4 inner** nested OOF, as materialized in the
registered `outer_folds.json` and `inner_folds.json`; execution reads and asserts each artifact's
`n_folds` rather than hard-coding a different count. The later all-train confirmation refit uses a
fresh grouped **5-fold inner OOF** construction. These are distinct stages and the confirmation
five-fold construction must not be substituted into outer model selection.

For each outer split/fold and refit:

1. build inner OOF classifier/state predictions for every config;
2. derive CVoI targets for every inner-held-out group. For policy/config evaluation on inner fold
   `h`, fit the policy only on target rows whose videos are outside `h`, then execute it on `h`;
   after selection, refit the chosen policy on the union of all cross-fitted target rows. Thus a
   policy never evaluates the same group's utility rows that fitted that policy;
3. obtain exactly one pooled inner-OOF probability, realized cost and label per video for every
   arm/budget/config; duplicate state/trajectory rows never become duplicate metric rows;
4. choose config/epoch under §8 and threshold under prereg §3.3;
5. refit classifier and policy on full outer train for the fixed epoch;
6. execute every frozen arm once on outer-query cached actions and emit predictions/cost traces.

Arm selection never pools the same video's predictions from multiple inner models as independent.
After all train OOF artifacts, hashes, selected B10 config, primary budget and verdict code are
immutable, refit the frozen procedure on all train using a fresh 5-fold grouped inner OOF target
construction. Derive the confirmation threshold from train inner OOF. Emit HateMM-val once. No val
epoch/config/threshold is computed.

## 12. Metrics and exact bootstrap

Binary macro-F1 uses sklearn semantics with labels `[0,1]` and `zero_division=0`. AUROC is score-based.
Log-loss clips probabilities to `[1e-7,1-1e-7]`. Cost is the per-video sum of realized action,
output-encoding, policy, retrieval and solver times. AUPAC trapezoid-integrates macro-F1 over realized
mean incremental-cost fraction, sorts by cost, collapses equal-cost points by keeping the lower F1,
and divides by the observed 0--1 cost span.

Primary point estimate:

1. concatenate five outer-query folds within a `(split_seed,refit_seed)` run;
2. compute the complete-run metric/delta;
3. arithmetic-mean the nine complete-run values.

Primary bootstrap replicate `r`:

1. within each split seed, resample group IDs with replacement, stratified by group label profile
   `(contains_positive,contains_negative)`; a sampled group carries all member videos;
2. apply identical multiplicities to every arm, budget and refit;
3. compute each full-run metric using the already frozen fold-specific thresholds;
4. average nine run metrics;
5. for B10 identify anew the non-CVoI baseline with highest macro-F1 whose replicate mean realized
   cost is `<=` B10's; ties use lower cost then arm ID;
6. compute Delta*, gates, Pareto and secondary deltas.

The percentile 2.5/97.5 endpoints form the 95% CI. A lower-bound gate reads percentile 2.5. If a
bootstrap sample lacks a class, macro-F1 remains defined by fixed labels `[0,1]`, but AUROC is null;
report the count. Training-uncertainty sensitivity resamples the nine complete run IDs with
replacement outside group sampling, seed `20260820`; it is secondary.

Budget/Pareto: a baseline is cost-admissible in a replicate only when its mean actual cost is no
greater than B10's. Performance dominance requires Delta 2.5th percentile `>0`. Cost dominance at
matched performance requires the 2.5th percentile of cost saving `>0`. No interpolation is used in
binding gates; linear interpolation is descriptive only.

## 13. Artifact contract

Run root: `artifacts/cvoi_acq/<run_id>/`, where `run_id = cvoi-acq-v1-<UTC>-<payload_sha256[:8]>`.
Required files:

```text
manifest.json                     payload, environment, contact ledger, status
asset_registry.json               every input path/hash/schema
groups/{sources,edges,components,folds}.*
actions/{ocr_actions,dense_frames,dense_sidecar,cost_actions,cost_summary}.*
selection/inner_selection.jsonl   cfg, epoch, threshold and provenance
targets/*.parquet                 inner-OOF utility targets only
predictions/train_oof.jsonl       one row/video/arm/budget/split/refit
predictions/val_confirmation.jsonl
traces/acquisitions.jsonl         ordered S, scores, estimated/actual cost, status
metrics/metrics.json              points, CIs, gates, verdict
metrics/bootstrap.npz             indices/deltas, not labels copied from test
fixtures/report.json
resources.jsonl                   wall/RSS/GPU/runtime guard events
RESULTS.md                         immutable design digest plus rendered outcome
```

Prediction row schema:

```text
schema, run_id, dataset, split_role, video_id, group_id,
outer_split_seed, outer_fold, refit_seed, arm_id, budget_fraction,
score, prediction, threshold, threshold_source,
estimated_budget_ms, realized_cost_ms, action_trace_sha256,
config_id, epoch, model_sha256, payload_sha256, draw_id|null
```

Every video has exactly one row per arm/budget/split/refit in train OOF. Val has one row per frozen
arm/budget/refit and no outer fold. Artifact writes are atomic temp-file + fsync + rename; an existing
run root is never overwritten.

## 14. Fixtures

All fixtures use synthetic labels/features and must pass before real metric imports are enabled:

| ID | assertion |
|---|---|
| F1 | canonical action IDs/order and K=30 boundaries |
| F2 | groups never cross outer/inner folds; exact query coverage |
| F3 | duplicate transitive closure and no missing-feature edge |
| F4 | inner-held-out group absent from classifier, target and memory fit rows |
| F5 | query OCR/dense bytes inaccessible until action purchase |
| F6 | empty/failed action consumes cost and returns frozen token |
| F7 | threshold uses pooled inner OOF only and tie rule is exact |
| F8 | set-conditioned utility has correct sign and differs from singleton on planted interaction |
| F9 | B9 cannot represent the planted interaction; B10 can overfit synthetic train only |
| F10 | policy inference is label-, gold-span-, Claude- and oracle-free |
| F11 | per-video budget cannot use realized full-query cost |
| F12 | exact DP matches brute force on <=18 actions and tie rule |
| F13 | uniform-cost case makes B12 NOT_APPLICABLE and bans knapsack label |
| F14 | measured cost includes policy/encoding/retry and catches an overshoot |
| F15 | bootstrap keeps groups/arms/runs paired and does not expand repeated videos |
| F16 | strongest admissible baseline is reselected inside each synthetic replicate |
| F17 | oracle outputs cannot reach selection/refit/threshold namespaces |
| F18 | val/test path denial; contact counter remains zero |
| F19 | checkpoint/refit has no outer-query callback and exact selected epoch |
| F20 | exact replay of the old global-linspace producer matches its planted reference, while the new interior-timestamp action remains a distinct schema |
| F21 | artifact cardinality, hashes and atomic collision refusal |
| F22 | planted G1--G5 pass/fail cases render exact verdict labels |

Fixture expected hashes are recorded only after an independent implementation review confirms the
synthetic constructions. A fixture that merely snapshots buggy output is invalid; F8, F12, F16 and
F22 include analytically stated expected values in the implementation source.

## 15. Payload and freeze mechanism

`freeze_manifest.json` contains:

```text
study_id, prereg_sha256, appendix_sha256, source_file_sha256 map,
asset_registry_sha256, groups_sha256, folds_sha256,
action_registry_sha256, cost_registry_sha256,
fixture_report_sha256, environment_lock_sha256, created_utc,
payload_sha256
```

Compute `payload_sha256` over canonical JSON of every preceding field except itself. The execution
CLI takes `--frozen-config`; before importing labels or features it recomputes all hashes and exits
`HALT_CONFIG_HASH_MISMATCH` on any difference. New assets are registered in this order:

1. generate action/cost/group artifacts without importing the candidate metric module;
2. QC schemas/counts and independently verify hashes;
3. fill the previously `PENDING` registry entries;
4. run F1--F22 on synthetic data;
5. independently review the appendix-to-code mapping;
6. hash prereg, appendix, code, assets, folds, costs and fixtures into one payload;
7. enable the metric executable and submit the formal run once.

Steps 1--6 cannot print labels by arm, predictions, utility distributions, macro-F1, loss deltas,
ranking metrics or any candidate-sensitive statistic. Shape, count, finite, hash, timing and failure
rate QC are allowed. Any code/asset/fold change after step 6 creates a new payload and requires a
timestamped prereg deviation.

## 16. Resource and runtime guards

- Environment: `conda activate HateVideo` inside jobs.
- All GPU/computational jobs use SLURM, no `--time`, at most 2 GPU / 16 CPU / 128 GB.
- Initial `PENDING (JobHeldUser)` is left for automatic release.
- Separate asset jobs from the single formal metric job. Asset jobs may resume by action ID and
  write append-only shards; their deterministic merge is hashed before freeze.
- Formal job is one `sbatch` submission. It may checkpoint operational state but cannot alter a
  seed, fold, arm or selection after resume.
- Preflight estimates target counts: train 44,640 actions (744*30*2), val 6,420 actions; dense has
  four frames/action. A mismatch halts unless explained by an explicitly registered unavailable
  action row.
- RSS guard: soft warning at 96 GB, hard halt before 120 GB. CPU threads <=16.
- GPU guard: record model and peak memory; OOM causes `HALT_RESOURCE`, not smaller batch or arm-specific
  truncation. A batch-size amendment must precede metrics and apply to all cost-comparable arms.
- Runtime watchdog writes a progress row every 100 videos or 10 minutes, whichever occurs first.
- Disk preflight requires 2x projected new-asset bytes plus 20 GB free. No existing cache is deleted
  or overwritten by this study.
- Cost trials share one hardware model and software image. Cross-hardware numbers never enter one
  Pareto table.

## 17. Gold/Claude/test process isolation

Use separate modules and OS paths:

- `deployable/`: labels for training loss, legal train/val action outputs, cheap features;
- `diagnostics/`: strict-22/broad-28 and Gate-C annotations, imported only after global metrics freeze;
- `oracle/`: query label access for O1/O2, writes only `oracle_*` fields;
- `sealed/`: test deny-list, never mounted.

The deployable CLI rejects keys containing `span`, `required_modalities`, `primary_cause`,
`secondary_causes`, `claude`, `oracle` or `test_seen`. Labels are passed to loss/metric functions,
not included in policy feature dictionaries. Diagnostics execute in a later process against frozen
prediction hashes and cannot rewrite `metrics.json` or `selection/`.

## 18. Pre-execution completeness ledger

All rows begin `PENDING`; only evidence paths and hashes turn them to `PASS`.

| gate | required evidence | initial status |
|---|---|---|
| C1 | source asset schemas/counts/hashes, including legal cheap midpoint cache | PENDING |
| C2 | strict-22/broad-28 files match fixed hashes | PENDING |
| C3 | group sources, edge audit, components and folds frozen | PENDING |
| C4 | OCR actions canonicalized and engine bytes identified | PENDING |
| C5 | dense train+val per-frame/timestamp assets and separate old-schedule parity replay | PENDING |
| C6 | synchronized per-action cost registry and hardware lock | PENDING |
| C7 | base selection record and feature ID alignment | PENDING |
| C8 | exact model/arm/config implementation review | PENDING |
| C9 | inner-OOF target/state generator leakage review | PENDING |
| C10 | nested selection/threshold dry run on synthetic data | PENDING |
| C11 | B12 heterogeneity rule and exact-solver fixtures | PENDING |
| C12 | metrics/bootstrap/Pareto envelope fixtures | PENDING |
| C13 | artifact cardinality, atomic writes and contact ledger | PENDING |
| C14 | resource/runtime guards and SLURM script review | PENDING |
| C15 | F1--F22, independent review, complete payload hash | PENDING |

Formal execution is mechanically impossible while any row is not `PASS`. The completeness ledger
may report only non-candidate QC. Its final signed copy is stored beside `freeze_manifest.json`.

### 18.1 Responsibility map (append-only clarification, 2026-08-08)

This mapping narrows ownership without promoting any gate. C1 owns source-level schema, count,
digest and train/val coverage, including the frozen 768-d OCR action embedding bank and the
unified video-by-K30 action join. C4 owns canonical OCR outcome bytes and OCR engine identity; it
does not certify the downstream text encoder. C5 owns dense4 frame/action bytes, timestamps,
status and old-schedule parity. C6 alone owns measured per-action costs and hardware lock; a
unified action registry must mark cost joins pending until C6 passes. C7 owns the already selected
whole-video base-feature assets and their ID alignment; C1 references that record and never copies
or reselects base features. C8 owns executable model/arm/config semantics, including consumption
of the registered OCR 768-d outcomes, but does not own source-asset completeness. Thus an OCR
embedding-bank or unified-registry audit may support C1 while C4, C5, C6, C7 and C8 retain their
separate evidence and status.

### 18.2 Registered deviation D1: retire old K30 comparability (2026-08-08T10:55:00+12:00)

The pre-registered train replay was executed at the unchanged `5e-5` tolerance over all 744
videos and failed (`max_abs=1.0418891906738281e-4`; 24 videos exceeded tolerance). This remains a
FAIL and is not reinterpreted or repaired by widening tolerance. The old train K30 cache and its
dev-seen counterpart are permanently retired from every formal consumer; their paths and SHA256
digests are a denylist, and val old-cache parity is not run because train failed first.

Accordingly, `old_cache_comparability=FAIL`. The replay artifact, log and non-zero exit are required
negative C5 evidence. C5's positive asset evidence is now limited to the newly generated
interior-timestamp dense4 train and val assets: exact video-by-30-by-4 schema and action IDs,
requested timestamp formula, frame slots, action-level feature-row joins, float32 feature hashes,
decode/fallback/`EMPTY_DENSE` behavior, pinned model provenance, source/start-manifest integrity and
zero test contact. This deviation does not alter the new dense action definition, any endpoint,
budget, model, threshold or candidate analysis.

## 19. Deliberately unresolved values

The following depend on assets/code that do not yet exist and are not fabricated here: exact source
file hashes other than the two lineage lists; group/component counts; fold counts after deterministic
fallback; dense parity result; OCR engine weight digest if absent from metadata; per-action cost
distribution; B12 heterogeneity applicability; exact base-cache path; fixture and payload hashes;
runtime and storage totals. Each is resolved only through §§15/18 before candidate metrics.

## 20. Results

Not run. No candidate metric is authorized or recorded in this appendix.
