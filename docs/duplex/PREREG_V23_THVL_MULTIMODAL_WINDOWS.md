# V23 THVL Validation-Informed Multimodal Window Protocol

Status: **DRAFT PRE-REVIEW — MUST FREEZE BEFORE V23 RAW FORWARD**  
Disclosure: V23 is `VAL-INFORMED`; THVL validation aggregate results from V21/V22 are known. THVL test GT remains sealed.

## Fixed question

V23 tests whether a synchronized visual frame plus local speech can supply temporal evidence that the V22 speech-only locator lacked. It is not a new taxonomy, prompt, split, or test cohort. The global comparator remains the frozen V22 causal-ASR global stream.

## Label-free 30-second windows

For a media duration `D`, construct non-overlapping half-open windows
`[30k, min(30(k+1), D))` for every integer `k` with `30k < D`. The union must equal `[0,D)` with no gaps or overlaps. Window construction uses only the frozen ffprobe duration.

Each window receives:

- one RGB frame decoded at `(start+end)/2` seconds;
- every sanitized ASR chunk with positive temporal overlap with the window, ordered by `(start,end,source_chunk_index)`;
- ASR text clipped only by a fixed character/token cap chosen before forward.

ASR chunks crossing a boundary contribute to both adjacent windows; their text is not duration-weighted or truncated differently by label. A window with no overlapping speech uses the fixed literal marker `[NO SPEECH]`. Empty speech is not a negative target and does not change cohort eligibility.

Center-frame decode must use one frozen fallback sequence independent of labels, for example exact center, then `center-0.5s`, then `center+0.5s`, clipped to the window. A window failing all attempts is recorded unavailable before GT access. Formal raw requires 100% window coverage; no window may be dropped, duplicated, or replaced after labels are consulted.

## Frozen judge

Use the same pinned Qwen3-VL checkpoint for every window. Run each window sequentially; multimodal packed inference is forbidden unless exact token/mRoPE equivalence was separately proven before raw generation.

Prompt text is fixed before raw and contains no THVL taxonomy leaves, examples, targets, dataset name, platform cues, or validation-derived lexicon. Proposed policy:

> Judge whether this local video moment contains hateful or severely abusive content. Use the synchronized frame and speech jointly. Answer Yes or No.

The raw score is the frozen next-token `logsumexp(Yes IDs)-logsumexp(No IDs)` margin. Raw artifacts contain opaque video ID, window index/start/end, input hashes, decode provenance, model/prompt/source hashes, and finite score, but no GT or per-video labels.

## Score construction

- Global `G_v` is exactly the frozen V22 duration-weighted causal-ASR video mean transformed by its frozen validation mid-ECDF state. It is not refitted using V23 labels.
- Window scores are copied to every 1-Hz frame whose interval midpoint belongs to that window. Center the resulting local timeline within each video and divide by one validation RMS frozen before the grid. Because windows cover the complete media duration, there is no missing local support.
- Final score is `S_v(t)=alpha G_v + beta L_v(t)`.

Grid: `alpha,beta ∈ {0,.25,.5,1,2}`, excluding `(0,0)`. Mandatory arms are global-only `(1,0)`, local-only `(0,1)`, and equal `(1,1)`. `beta=0` must be exact global-only ranking.

## Validation evaluation

Use the existing sealed validation evaluator and the frozen scoped mask:

- `Hate`, `Bias`, or `Verbal Abuse` frames positive;
- other-harm-only frames ignored;
- outside all annotated harm segments negative.

Select by pooled Frame AP, then pooled ROC, within-video macro AP, smaller `alpha+beta`, then ascending lexicographic `(alpha,beta)`. Report all 24 grid cells, mixed-video eligibility, full media/window/decode coverage, and finite/exact-length assertions.

Controls:

- deterministic within-video window-score shuffle, `B=200`, preserving window lengths and video membership;
- paired source-group cluster bootstrap, `B=2000`, seed `23023`, full minus global-only, reporting AP/ROC and within macro AP/ROC 95% percentile intervals.

No per-video validation error analysis is returned to development.

## Activation and test discipline

V23 activates only if all conditions hold:

1. all validation videos and every 30-second window have frozen finite raw scores and valid frame provenance;
2. selected `beta>0`;
3. selected within-video macro ROC is at least `.55`;
4. selected pooled Frame AP and ROC are each strictly greater than global-only;
5. selected within macro AP and ROC are each no lower than global-only;
6. selected within macro ROC strictly exceeds the `97.5%` shuffle quantile;
7. source-group bootstrap lower 95% bounds for pooled AP and ROC deltas are nonnegative.

Failure freezes a negative `VAL-INFORMED` result and leaves THVL test sealed. Passing validation only authorizes label-free test media preprocessing and raw forward. The chosen config, prompt, model, window rules, preprocessing hashes, and test raw must then be frozen before a separate one-shot steward evaluator may open test GT.

