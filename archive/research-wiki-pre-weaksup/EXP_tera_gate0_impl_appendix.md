# TERA Gate-0 — implementation appendix (v4 — FROZEN)

> **Status: v4 — RE-FROZEN 2026-08-07.** All `TO-FILL-AT-ASSET-AUDIT` placeholders are resolved
> (§2.1, §2.7, §2.9, §13.2 of the prereg), the §9 fixture battery passed **16/16**, and the
> companion config is `research-wiki/tera_gate0_frozen_config.json` (no longer `.draft.json`).
> v4 differs from the frozen v3 only by registered deviation **D-3**
> (`refine-logs/TERA_GATE0_DEVIATION_D3_2026-08-07.md`): a harness defect that narrowed the §6.7
> msc subset was fixed before any affected metric was computed, and the package/battery digests
> and this document's own digest were re-embedded. **No definition, threshold, seed, or decision
> rule in this document changed between v3 and v4** — the change log entry in §14 is the full
> difference. Any change after this point is governed by prereg §12 (documentary correction vs
> material deviation) and produces a new payload hash and a new `run_id`.
>
> Independent review of v1-draft returned **APPROVE-WITH-FIXES**
> (`research-wiki/EXP_tera_gate0_impl_appendix_review.md`, 2026-08-07, review round 1 of 1).
> All five BLOCKING-FIX items (B-1 … B-5), all four factual corrections (F-1 … F-4), and the
> accepted NOTEs are applied here. Per the project's proportional-ceremony rule this was the only
> review round; author self-test evidence (a full §9 fixture-battery pass) releases the freeze.
>
> This document instantiates `EXP_tera_gate0_prereg.md` §11.1. It **may not** add or remove an
> arm, change an endpoint, threshold, split, or decision rule, and it contains no candidate
> result. §13 records the implementation readings adopted while building the harness; every one of
> them is a fixture-construction-layer or bookkeeping reading and none changes an arm, endpoint,
> threshold, split, or decision rule.
>
> **Blinding statement.** While drafting v1 and v2 no file under `data/` was loaded,
> deserialized, or reduced to a number. Only directory listings (file names and byte sizes) and
> the repository's own extraction/probe source code were read. Every asset-derived constant was
> marked `TO-FILL-AT-ASSET-AUDIT` or derived from source code, not from data. The reviewer's
> independently verified structural facts (record counts of `hate_spans.json` = 1083,
> `gold_segments.json` = 395, `p11_split.json` = 237/39/119) were cited from the review record and
> not re-derived at drafting time. The v3 asset audit (2026-08-07) resolved the placeholders under
> the §2.8 restricted reader and the §10.4 test-seal guard, reading **only** id lists, labels,
> tensor shapes and raw bytes; **no metric of any kind was computed** and
> `data/gt/HateMM/test.jsonl` was opened **zero** times (`test_contact_count = 0`,
> `opened_test_paths = []`).
>
> **Freeze procedure (executed).** (1) Every `TO-FILL-AT-ASSET-AUDIT` field was resolved in the
> read-only asset audit (prereg §13.2), including the two formerly pending caches of §2.4;
> (2) the §9 fixture battery (F1–F15, 16 cases including F7b) passed 16/16
> (`artifacts/tera_gate0/_fixtures/fix-20260806T231531Z/fixtures_report.json`); (3) the completed
> payload's canonical hash (§10.2) was recomputed and written into
> `research-wiki/tera_gate0_frozen_config.json`; (4) `appendix_sha256` was recomputed over this
> file byte-for-byte and embedded in that payload; (5) at run start the harness re-verifies both
> digests and copies the config to `artifacts/tera_gate0/<run_id>/frozen_config.json`. Executing
> with a mismatch is `HALT_CONFIG_HASH_MISMATCH` and a §12 violation.

---

## 0. Scope, environment, and execution discipline

### 0.1 Frozen upstream

The scientific design is `research-wiki/EXP_tera_gate0_prereg.md` (registered 2026-08-07).
Arm list (A0, A1, A2, A3, A4, O1, O2, B0–B5), endpoints, decision thresholds, split roles, fold
seeds, bootstrap protocol, taxonomy, and stopping rules are taken verbatim from it.

### 0.2 Execution environment (registered)

| item | value |
|---|---|
| host class | single-GPU workstation, NVIDIA GeForce RTX 5090 (32 GiB), **no SLURM** |
| scheduler | none — direct local execution, per `CLAUDE.md` single-GPU exemption |
| conda env | `HateVideo` |
| device for all A/B/O heads | **`cpu`** (see §7.9) |
| device for feature extraction (§2.4) | `cuda:0` |
| `torch.set_num_threads` | `8` (pinned; CPU reduction order depends on thread count) |
| cloud | **not used.** Gate-C needs raw video; Gate-A/B heads are minutes of CPU. |

`nvidia-smi` reports one GPU and `sbatch`/`squeue` are absent, so prereg §13.4's "one-GPU machine
may run directly" branch applies. Recorded in `manifest.json` as `scheduler: "none"` with the
`nvidia-smi` device string.

### 0.3 Execution discipline — detached background runs (CLAUDE.md 2026-08-07)

Every command in this plan that is not instantaneous — feature extraction (§2.4), the Gate-A OOF
sweep, the Gate-B sweep, the fixture battery — **must be detached from the SSH session** and must
write a log and a PID file at the registered locations:

```bash
TASK=tera_gate0_<stage>            # e.g. tera_gate0_extract, tera_gate0_stageA, tera_gate0_fixtures
mkdir -p logging/runs/$TASK
nohup <command> > logging/runs/$TASK/run.log 2>&1 &
echo $! > logging/runs/$TASK/run.pid
# progress:  tail -f logging/runs/$TASK/run.log
# liveness:  ps -p $(cat logging/runs/$TASK/run.pid)
```

Long-running stages must emit a parseable progress line to `run.log` at least every outer fold
(format `[tera-gate0] stage=<A|B> arm=<id> outer=<k> cfg=<i>/<n> epoch=<e> elapsed=<s>s`).
`manifest.json` records `log_path` and `pid_file` for every stage.

### 0.4 Fully registered before execution

Everything the prereg left open is pinned here. Nothing in this appendix may be chosen after a
number is seen. In particular the five items the review identified as previously open — the
confirmation-set protocol (§7.10), the cross-fold aggregation that selects `D` (§5.3), the epoch
selection rule (§7.2/§7.4), the sealed-id reader (§2.8), and the Gate-B `msc`-subset conventions
(§6.7) — are now single-valued.

---

## 1. Notation

- `V` — number of videos in a dataset partition. `K = 30` — registered window count.
- `k ∈ {0,…,29}` — window index, temporally ordered. `D_v` — probed media duration in seconds.
- window `k` of video `v` covers `[k·D_v/K, (k+1)·D_v/K)`, last window closed at `D_v`.
- `s_{v,k} ∈ R^{d}` — segment representation (§2.3). `x_v ∈ R^{d}` — whole-video representation.
- `l2n(z) = z / max(‖z‖₂, 1e-12)`; `l2n(0) = 0`. `σ` — logistic sigmoid.
- macro-F1 — `sklearn.metrics.f1_score(y, ŷ, average="macro", zero_division=0)`.

---

## 2. Feature family, assets, and the timestamp-boundary argument

### 2.1 Encoder (frozen, shared by every arm)

`openai/clip-vit-large-patch14-336`, HuggingFace `transformers` 4.49.0, frozen, no fine-tuning
(prereg §7 forbids encoder fine-tuning in Gate-0).

- **Visual stream**: `CLIPVisionModel.pooler_output` per frame, mean-pooled.
  `Dv = vision_model.config.hidden_size` — the vision tower hidden size, **not** the 768-d joint
  projection (`src/utils/generate_subclip_embedding_HF.py:316`). Expected `Dv = 1024`;
  **asset audit 2026-08-07: `Dv_observed = 1024`** (identical in all six caches).
- **Text stream**: `CLIPTextModel.pooler_output` over the transcript, chunked and mean-pooled
  (`src/utils/generate_VideoCLIP_embedding_HF.py`). Expected `Dt = 768`;
  **asset audit 2026-08-07: `Dt_observed = 768`**, hence `d = Dv + Dt = 1792` as registered.
- Empty transcript placeholder: `"(none)"` (repository parity constant, RUNBOOK §5.3).

### 2.2 Caches (canonical inputs)

| role | path | status | sha256 (asset audit 2026-08-07) |
|---|---|---|---|
| HateMM train, K=30 segments | `data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt` | present (91,800,918 B) | `8b4a706cec51d106151e57109b24850232239168d5e0ca363341ee76493d7fb7` |
| HateMM train, whole video | `data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt` | present (5,359,881 B) | `0802b6ba00669ec546e63f36dca1772cb2d7806b969de307235af3450a8176c1` |
| HateMM val (`dev_seen`), K=30 segments | `data/CLIP_Embedding/HateMM/dev_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt` | **extracted 2026-08-07** (§2.4), 13,204,954 B | `a2ae105e61478b86193267fe67263d1c26436f0881620222f0aa1544fa380778` |
| HateMM val, whole video | `data/CLIP_Embedding/HateMM/dev_seen_openai_clip-vit-large-patch14-336_HF.pt` | present (772,382 B) | `ab9cd8a070b93afbf994ed876e3adfd9c2a139e82d801af21346c29f17c1888d` |
| HateClipSeg, K=30 segments | `data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt` | present (48,739,122 B) | `df6e1c0434ba4b0fb210c3470b3407e05e041f718834d70ad3bc20bcde34d89e` |
| HateClipSeg, whole video | `data/CLIP_Embedding/HateClipSeg/test_seen_openai_clip-vit-large-patch14-336_HF.pt` | **extracted 2026-08-07** (§2.4), 2,846,592 B | `43227d527d402e1707f770386667cb39114c861f01345c0ab3b9087abedf6f30` |
| HateMM gold spans | `data/gt/HateMM/hate_spans.json` | present (corpus-spanning — §2.8) | `f8f2be10856a40c0ef5763b9211ecbed506743792ccddfb3adc92bed460c1846` |
| HateClipSeg gold segments | `data/gt/HateClipSeg/gold_segments.json` | present (corpus-spanning — §2.8) | `a1dad37e686a5106a1392e8151f0946858bc6086cdfd05efc9457b1d7c634a36` |
| HateClipSeg durations | `data/gt/HateClipSeg/video_durations.jsonl` | present (corpus-spanning — §2.8) | `d8bd334b90f270a703c8419717c979fddd13da537486bb5699d12400f1c1e292` |
| HateClipSeg split | `data/gt/HateClipSeg/p11_split.json` | present (ids only, not corpus-restricted) | `a279431137feeaf72241e1ca4a7ef76d1e86c8381d08aac47123b4b287db98b1` |

The three gold artifacts and the two HateClipSeg caches were hashed **inside the §2.8 reader
scope** (`data.py:hash_under_guard`): provenance hashing reads bytes only and never deserializes
an id or a label, but it does open the file, so it runs under the restricted-reader guard.
Every segment cache reports `num_subclips = 30`, `num_frames = 120`; every whole-video cache
stores no `num_frames` key and its `8` stays `provenance_only` (review F-2).

**Segment-cache schema** (`generate_subclip_embedding_HF.py:27-40`) — note `video_ids` is a
**flat** list:

```text
{"video_ids": [V] str, "subclip_img_feats": [V*K, Dv] float32,
 "subclip_parent": [V*K] long, "labels": [V*K] long,
 "num_subclips": K, "num_frames": M}
```

**Whole-video-cache schema** (`generate_VideoCLIP_embedding_HF.py:14-15, 395-397`) — `ids` is a
**nested list containing exactly one sublist**, per the extractor's explicit contract comment
(`# CONTRACT: ids is a list containing ONE sublist of all string ids.`):

```text
{"ids": [[V] str], "img_feats": [V, Dv] float32, "text_feats": [V, Dt] float32, "labels": [V] long}
```

**The registered read convention is `d["ids"][0]`**, matching every reader in the repository
(`c3_nontarget_probe.py:73`, `apx_g0cond_gate.py:64,77`, `s2s_probe.py:95,102`,
`eval_localization_ours.py:82`, `clap_cache_verify.py:55,68`, `w2b_probe.py:128`). Treating
`d["ids"]` as flat raises `TypeError: unhashable type: 'list'`. *(review F-1)*

Every cache used gets a SHA256 in `feature_manifest.json`, together with its `num_subclips` /
`num_frames` where the file stores them. The whole-video cache stores **no** `num_frames` key, so
its frame budget (`8`, the extractor default) is recorded as `provenance_only` — it is asserted
against the extraction command line, never against the artifact. *(review F-2)*

### 2.3 Registered representations

The segment cache stores **visual features only**; sub-clips share the parent's video-level text
embedding by construction (`generate_subclip_embedding_HF.py:22-25`), so there is no per-segment
text even in principle. The registered representations are:

```text
s_{v,k} = concat( l2n(subclip_img_feats[parent==v][k]),  l2n(text_feats[v]) )     ∈ R^{Dv+Dt}
x_v     = concat( l2n(img_feats[v]),                     l2n(text_feats[v]) )     ∈ R^{Dv+Dt}
d = Dv + Dt   (expected 1792)
```

Decisions:

1. **Text is included, identically, in every arm** (review OP-2 ratified). It is a video-level
   input, carries no segment supervision, and excluding it would make A0 weaker than the
   repository's standard HateMM whole-video baseline and would make the Gate-C `cross_modal`
   category untestable by any Gate-A arm. **Disclosure (kept verbatim from v1):** because the text
   half is constant across `k`, all within-video discrimination comes from the visual half; this
   is not a temporal-text capability. As the review verified, a per-video additive constant cannot
   change a within-video ranking, so including text cannot inflate the temporal criterion
   (prereg §5.2 item 5) — it only strengthens the video-level baselines, which is the conservative
   direction for criteria 1–3.
2. **Disclosure — A0 and A1 do not share a frame budget** *(review N-2)*. A0's `img_feats` come
   from the whole-video cache built at `--num_frames 8`; A1–A4 consume the segment cache built at
   `M = 120`. The encoder is identical, the sampling budget is not, and A0 L2-normalizes a plain
   8-frame mean whereas A1 averages 30 per-window-normalized vectors. This cannot inflate any
   criterion because criteria 1–3 compare against `max(A0, A1)` and A1 is the 120-frame
   comparator, so `max()` absorbs it. `num_frames` for every cache is recorded in
   `feature_manifest.json`.
3. **Per-stream L2 normalization, then concatenation.**
4. **No fitted preprocessing.** `fitted_preprocessing: "none"`; prereg §7 step 1 is satisfied
   vacuously and no fold-fitted object exists to leak.
5. **No retrieval memory.** `retrieval_memory: "none"` for every arm, the single registered
   exception being B5's donor draw (§6.6), which is a lesion, not a retrieval component.
6. `l2n(0) = 0`, so a decode-failed video keeps its zero-vector guard through normalization and is
   countable (§2.7).

### 2.4 Pending-extraction assets (environment change, 2026-08-07) — **completed**

> **v3 status.** Both extractions ran and completed on 2026-08-07 under the pinned parity
> constants below; logs at `logging/runs/tera_cache_extract/{hatemm_val_subclipK30.log,
> hateclipseg_wholevideo.log}` (PID files alongside, per §0.3). Realized outputs:
> HateMM val K=30 — `V = 107`, `TotalSub = 3210`, `Dv = 1024`, zero-vector videos `0`;
> HateClipSeg whole-video — `N = 395`, `Dv = 1024`, `Dt = 768`, zero-vector videos `1`
> (`yt_NzvfkIYS5Yg`, an undecodable container; it lies in `p11_split["test"]`, so it is dropped
> by the §2.8 restriction and never enters any evaluated Gate-0 set — see §2.7).
> `HALT_MISSING_ASSET` therefore does not fire. SHA256s are in the §2.2 table.


The raw corpora have been restored from B2 to `/home/jehc223/data/{HateMM,HateClipSeg,Multihateclip}`
(HateMM 6.2 G, HateClipSeg 4.2 G) and are exposed to the extractors through
`data/video/<DATASET>/All/`, which is the layout
`generate_subclip_embedding_HF.py:294` (`video_root = join(video_dir, dataset, "All")`) expects.
The two caches that were `ABSENT` in v1 are therefore **pending extraction**, not blocked:

| cache | extraction command (pinned parity constants, RUNBOOK §5.3) |
|---|---|
| HateMM val, K=30 | `python src/utils/generate_subclip_embedding_HF.py --dataset HateMM --splits val --num_subclips 30 --num_frames 120 --model openai/clip-vit-large-patch14-336 --batch_size 32 --device cuda` |
| HateClipSeg, whole video | `python src/utils/generate_VideoCLIP_embedding_HF.py --dataset HateClipSeg --splits test --num_frames 8 --model openai/clip-vit-large-patch14-336` |

Both run detached per §0.3 (`TASK=tera_gate0_extract`). The exact command line, git commit,
`HF_HUB_OFFLINE=1`, encoder id, and output SHA256 are recorded in `feature_manifest.json`;
prereg §13.2 forbids *silent* regeneration, not recorded regeneration. The extraction is a
**pre-execution asset step**: it produces no metric, touches no test id (HateMM `--splits val`
reads only `val.jsonl`; HateClipSeg's extraction covers the whole 395-video corpus and its output
is id-restricted at load time by §2.8).

**If extraction fails or produces a cache that fails its asset-audit assertions**, the affected
confirmation cannot be evaluated and the run stops with `verdict.status = "HALT_MISSING_ASSET"`
(prereg §12: a missing asset is a HALT, never a performance negative, and never a substituted
endpoint).

### 2.5 Raw media (now present)

`/home/jehc223/data/HateMM/video` and `/home/jehc223/data/HateClipSeg/videos` are restored, so
Gate-C's requirement that annotators watch the video (prereg §4.1) is satisfiable on this host and
§2.4's extraction is executable. Transcript-only auditing and MLLM-generated descriptions remain
**forbidden** as annotation substitutes — either would change the registered protocol and is a
§12 material deviation. Raw video is never uploaded to any cloud (prereg §13.1, `CLAUDE.md`).

### 2.6 Duration, window boundaries, and the sampling-alignment argument

**Registered duration source.** `D_v` is the `duration` field of `data/gt/HateMM/hate_spans.json`
(ffprobe container duration, produced by `scripts/analysis/hatemm_spans.py`). Window `k` is
`[k·D_v/30, (k+1)·D_v/30)`, last closed at `D_v`. HateClipSeg uses
`data/gt/HateClipSeg/video_durations.jsonl`.

**Frame-time convention (registered)** *(review N-1)*: the alignment argument uses the
**endpoint-index convention** — the extractor's sampled frame `j ∈ {0,…,M−1}` is taken to sit at
normalized time `j/(M−1)`, which is exactly what `_sample_frame_indices` implements
(`np.round(np.linspace(0, num_total−1, num_frames))`,
`generate_subclip_embedding_HF.py:135-147`). This convention, not the frame-midpoint convention
`(m+0.5)/N`, is what the asset audit re-asserts.

**Does the existing K=30 cache respect the registered boundaries?** `M = 120` frames are sampled
and `_window_bounds(120, 30)` (`generate_subclip_embedding_HF.py:236-259`) gives `base = 4,
rem = 0`, so window `k` receives sampled frames `j = 4k … 4k+3`, at normalized times
`[4k/119, (4k+3)/119]`. The nominal window is `[k/30, (k+1)/30] = [4k/120, (4k+4)/120]`. Then

```text
4k/119     ≥ 4k/120 = k/30                       (left edge, all k ≥ 0)
(4k+3)/119 ≤ (k+1)/30  ⟺  120k+90 ≤ 119k+119  ⟺  k ≤ 29   (right edge, all k ≤ 29)
```

so **every sampled frame of window `k` lies strictly inside nominal window `k`**, for all 30
windows; the `k = 29` right edge is exact (`119/119 = 1.0`) and the last window is closed at `D`.
`scripts/slurm/hateclipseg_subclip.sbatch` runs the same `K=30 / M=120` configuration, so the
HateClipSeg K=30 cache is covered by the identical argument. Recorded as
`boundary_rule.sampling_alignment_proof` and re-asserted numerically at asset audit against each
cache's stored `num_frames` / `num_subclips`.

### 2.7 Failure accounting

*(review N-6)* — `zero_vector_videos` is the **union across both caches**: a video counts if its 30
segment vectors are all exactly zero **or** its whole-video `img_feats` row is exactly zero. The
union is what enters the HALT rule, and such videos are **kept** (with their zero vectors) in every
arm so the evaluated video set is bit-identical across arms, as prereg §5.1's matching requires.

- `zero_vector_videos` — union as above.
- `missing_duration_videos` — `duration` null or ≤ 0 in the gold duration source.
- **HALT if** `|zero_vector_videos ∪ missing_duration_videos| / V > 0.01` on HateMM-train
  (prereg §12).
- Missing-duration videos are excluded from the temporal-metric eligible set (§8.1) but still
  receive video-level predictions.

**Observed counts (asset audit 2026-08-07, read-only).**

| partition | V | `zero_vector_videos` | `missing_duration_videos` | union | rate | HALT? |
|---|---|---|---|---|---|---|
| HateMM train (development) | 744 | 1 (`hate_video_95`) | 0 | 1 | 0.001344 | no (≤ 0.01) |
| HateMM val (confirmation) | 107 | 0 | 0 | 0 | 0.000000 | no |
| HateClipSeg, restricted to `p11_split["train"]` | 237 | 0 | 0 | 0 | 0.000000 | no |
| HateClipSeg, restricted to `train ∪ val` | 276 | 0 | 0 | 0 | 0.000000 | no |

The binding HALT rule is the HateMM-train row: `1/744 = 0.13 % ≤ 1 %`.
The HateClipSeg corpus does contain exactly **one** zero-vector video, `yt_NzvfkIYS5Yg`, reported
by both extractors (segment cache and whole-video cache agree, so the union adds nothing). It is a
`p11_split["test"]` id, so under §2.8's whitelist restriction it is discarded before any load
returns and the *restricted* counts above are `0` in both phases. That corpus-level fact is
recorded here as **extraction provenance** (`logging/runs/tera_cache_extract/*.log`), not as a
restricted read.

### 2.8 Sealed-id restriction on corpus-spanning gold artifacts (BLOCKING-FIX B-4)

`data/gt/HateMM/hate_spans.json` holds **1083** records — the full HateMM corpus, i.e. train + val
+ the sealed official test set — and every record carries a `label` field. Prereg §2.3 forbids
loading "any official test labels, predictions, spans, or per-example artifacts". A path-based
guard cannot seal this file, because Gate-C and O1 legitimately need it. The same applies to
HateClipSeg's `gold_segments.json` (395), `video_durations.jsonl`, `test.jsonl`, and the
whitelisted K=30 / whole-video caches, all of which span the 119 `p11_split["test"]` videos.

**Registered rule — the single admissible reader.**

> Every corpus-spanning artifact listed below is loaded **only** through
> `load_corpus_spanning(path)`, which immediately restricts the returned object to the currently
> authorized id set and **discards every other entry before returning**. No code path may hold an
> unrestricted handle to these files; doing so is a HALT (`HALT_UNRESTRICTED_GOLD_HANDLE`).

Registered artifact list:

```text
data/gt/HateMM/hate_spans.json
data/gt/HateClipSeg/gold_segments.json
data/gt/HateClipSeg/video_durations.jsonl
data/gt/HateClipSeg/test.jsonl
data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt
data/CLIP_Embedding/HateClipSeg/test_seen_openai_clip-vit-large-patch14-336_HF.pt
```

Authorized id set, switched **exactly once** in the whole run:

| phase | HateMM | HateClipSeg |
|---|---|---|
| development (Gate-C, Gate-A OOF, Gate-B OOF) | `train.jsonl` ids | `p11_split["train"]` |
| confirmation (§7.10, one-time) | `train ∪ val` ids | `p11_split["train"] ∪ ["val"]` |

The switch is a single call `unlock_confirmation()` that writes `confirmation_unlock_utc` into
`manifest.json` and can be invoked at most once; a second call is a HALT. Test ids are never in the
authorized set in either phase.

Reader assertions (all mandatory, any failure is a HALT):

1. **only authorized ids survive**: `set(restricted_ids) ⊆ authorized_ids`. This is **whitelist**
   semantics — the reader keeps `authorized_ids ∩ file_ids` and drops everything else — so no
   sealed id can survive *by construction*, whatever the sealed splits contain;
2. `len(restricted) == len(authorized_ids ∩ file_ids)`;
3. `sealed_ids_dropped = len(file_ids) − len(restricted) > 0` for every corpus-spanning file
   (a zero drop count means the restriction did not run);
4. no `p11_split["test"]` id survives the restriction (free — `p11_split.json` is already read to
   build the HateClipSeg authorized set, and it contains ids only, no labels or spans).

**Note on the review's assertion 1.** The review asked additionally for "no HateMM `test.jsonl` id
survives". That check is **redundant under whitelist semantics and is deliberately not
implemented**: assertion 1 above is strictly stronger (it excludes *every* unauthorized id, not
just the test ones), and performing the weaker check would require opening
`data/gt/HateMM/test.jsonl` — a sealed per-example artifact — for no additional guarantee.
Prereg §2.3's prospective rule is therefore satisfied with **zero** contact with that file:
`test_contact_count` stays `0` and `opened_test_paths` stays `[]` with no exception carve-out.

`manifest.json` gains `sealed_ids_dropped{hatemm, hateclipseg}`, `authorized_id_hash` (sha256 of
the sorted authorized id list, per phase), and `confirmation_unlock_utc`. Fixture **F15** (§9)
verifies the whole mechanism on synthetic data, including that an unauthorized id planted in a
corpus-spanning file cannot survive.

### 2.9 Canonical split source (prereg §2.1)

`data/gt/HateMM/{train,val,test}.jsonl` **exist** (dated 2026-07-01; the review confirmed all
three), so resolution branch 1 below is the live branch. *(review N-13; v1's present-tense "absent"
text was stale.)*

```text
1. data/gt/HateMM/train.jsonl exists:
     load ids+labels; assert set(jsonl_ids) == set(cache["ids"][0])   # NOTE the [0] — §2.2, F-1
     and, for every id, jsonl.label == cache.label.
     On success: split_source = "gt_jsonl", record its SHA256.
     On any mismatch -> HALT_SPLIT_MISMATCH. Never silently prefer one source.
2. Else: split_source = "feature_cache_embedded", ids/labels from cache["ids"][0] and
     cache["labels"].
3. Record split_source, the source file SHA256, and split_id_hash = sha256(sorted id list)
     in split_manifest.json.
```

The identical rule applies to `val.jsonl`, loaded exactly once at confirmation time (§7.10).
`data/gt/HateMM/test.jsonl` is on `forbidden_paths` and is **never opened by any code path**.

**Asset-audit result (2026-08-07).** Branch 1 resolved for both partitions; every id and every
label matched the whole-video cache exactly, so `HALT_SPLIT_MISMATCH` did not fire. The id hash
below is `sha256(utf8("\n".join(sorted(ids))))` (`common.py:sha256_ids`), the same function the
run uses for `split_id_hash`, `authorized_id_hash` and `hateclipseg_surviving_id_hash`.

| partition | V | pos / neg | `split_source` | `split_source_sha256` | `split_id_hash` |
|---|---|---|---|---|---|
| HateMM train | 744 | 298 / 446 | `gt_jsonl` (`data/gt/HateMM/train.jsonl`) | `73295d4b96d9937dca7787fc59a13561ac15020c1608e196c5685f2a055d7741` | `54e1e9beb97c3e76fcd5c8f664d9b948dcb368e202c6e686f346e0f8a5e1273c` |
| HateMM val | 107 | 43 / 64 | `gt_jsonl` (`data/gt/HateMM/val.jsonl`) | `33a3768976a68db4fe3da39cacafa6beac11b04f83e5740fef0e4f91b391e2b3` | `9cee85f3db92e816c8e867743d7a87ad6d4043eb7d4ab732f95ce8f11d9fb7b3` |

The audit read `val.jsonl` for exactly this integrity assertion (ids, labels, id set equality with
the `dev_seen` caches). It computed no score, fitted nothing, and selected nothing; the run's own
one-time confirmation load of `val.jsonl` (§7.10.1, after `unlock_confirmation()`) is unaffected.

HateClipSeg: surviving 395-video corpus per `DATASET_hateclipseg.md`; split from `p11_split.json`
(237 / 39 / 119, re-verified at audit); the binding endpoint (`has ≥1 segment labelled hateful`) is
recomputed from the **id-restricted** `gold_segments.json` at run time and its per-split class
counts are reported; if either class has `< 10` videos in the P11 validation split the HateClipSeg
confirmation is reported as **underpowered** and is not used to satisfy any criterion (prereg §2.2).

- `hateclipseg_surviving_id_hash` (395 ids = `train ∪ val ∪ test` of `p11_split.json`) =
  `37d852b7f72cc87465bbcc293bb345e46617ac61a105be7a70ad3ff24640ba19`.
- `p11_split.json` sha256 = `a279431137feeaf72241e1ca4a7ef76d1e86c8381d08aac47123b4b287db98b1`
  (the file also carries a `meta` key of 8 entries alongside the three id lists).
- Binding-endpoint class counts under the **restricted** views (recomputed at audit from the
  id-restricted `gold_segments.json`): development `train` = 237 videos, 109 positive / 128
  negative; confirmation `train ∪ val` = 276 videos, 127 positive / 149 negative. The P11
  validation split therefore contributes 18 positive / 21 negative — **both classes ≥ 10**, so the
  underpower rule is not triggered by the asset audit's counts. (These are class counts, not
  metrics; the confirmation itself is still a single pass at §7.10.2.)
- Restriction evidence: `sealed_ids_dropped` = 119 for each of the four corpus-spanning HateClipSeg
  artifacts (`gold_segments.json`, `video_durations.jsonl`, and both `test_seen_*` caches) in the
  confirmation phase, and 232 for `hate_spans.json` — consistent with 1083 HateMM records minus the
  851 authorized `train ∪ val` ids, which independently reproduces the review's 1083 count without
  ever holding an unrestricted handle.

**Observed gold-span schema** *(review F-4)*: `hate_spans.json` records carry
`duration / spans / label`, plus `clipped` and `anomaly` on a small number of records. The
registered schema assertion checks exactly this field set; `parse_error` (emitted by
`hatemm_spans.py` only on a hard parse failure) does **not** occur in the current file and is not
asserted. **Observed at asset audit** on the 744 restricted HateMM-train records: `duration` 744,
`spans` 744, `label` 744, `clipped` 2, `anomaly` 1, `parse_error` 0.

---

## 3. Uniform definitions used by every arm

### 3.1 The registered per-segment score

**Decision (single rule for all arms).** For any arm `f` mapping a bag of segment representations
to a video logit, the registered per-segment score of window `k` is the arm's own scoring function
evaluated on the singleton bag containing only that segment:

```text
seg_score_arm(v, k) = σ( f_arm( { s_{v,k} } ) )
```

- A2, A4: reduces to the linear segment logit `z_{v,k} = w·s_{v,k} + b`.
- A1: `head(s_{v,k})`.
- A3: on a singleton `α = 1`, so `p = s_{v,k}` and the score is `head(s_{v,k})`. Because
  `p_v = Σ_k α_k s_{v,k}` and the head is affine, `logit_v = Σ_k α_k · head(s_{v,k})` exactly, so
  `head(s_{v,k})` *is* segment `k`'s own contribution to A3's video logit.
- A0: no segment model; per prereg §5.2 item 5 the video score is broadcast to every second,
  giving within-video AUROC exactly 0.5.

**Implementation note** *(review N-4)*: the singleton reduction must be implemented directly, not
by calling the generic pooling function — `torch.topk(z, k=2)` on a one-element tensor raises. The
registered reduction for each arm is the closed form listed above.

**A3 α-based diagnostic (pre-registered, non-binding)** *(review N-3)*: in addition to the binding
score above, the run computes and records the within-video second-level AUROC using A3's attention
weights `α_k` as the segment score. It is a **diagnostic**: it may never rescue a failed
criterion 5 and never enters any decision. Its purpose is that, if `D = A3` fails criterion 5 while
the α-based AUROC is high, the record contains the evidence that the failure is
metric-definitional rather than substantive. `attention_weights` are already stored in
`segment_scores.jsonl`.

This rule defines uniformly: the §8 temporal metrics, gold-span recall@{1,2,4}, the
selected-vs-unselected separation, and Gate-B's top-two selection.

### 3.2 Seconds → windows

Second `t` (integer, `t = 0 … floor(D_v)−1`) is **positive** iff `t + 0.5` falls inside a gold span
(prereg §8.2). Its score is the score of window `w(t) = min(29, floor((t+0.5)·30/D_v))`.

### 3.3 Threshold selection (prereg §8.1)

Given pooled inner-OOF scores `S` and labels `y`:

```text
u = sorted(unique(S))
candidates = { (u_i + u_{i+1})/2 : i = 1..|u|-1 } ∪ { u_1 - 1e-6, u_{|u|} + 1e-6 }
prediction rule:  ŷ_i = 1 iff s_i >= θ            # ">=" is pinned
θ* = argmax_θ macroF1(y, ŷ(θ))
ties -> smallest |θ - 0.5| ; still tied -> smallest θ
```

`θ*` is carried unchanged to the outer-query fold. Every prediction row records `threshold` and
`threshold_source`. No re-thresholding after results are seen, and **no threshold is ever derived
on a confirmation set** (§7.10).

### 3.4 Deterministic seed derivation

Prereg-fixed: outer folds `20260807`, inner folds `20260808`, bootstrap `20260809`, B4/B5
`20260807`, Gate-C sampling `20260807`. Model-initialization seeds are not fixed by the prereg, so
this appendix registers a derivation (base `20260810`) that is order-independent and
collision-free:

```python
def derive_seed(scope: dict) -> int:
    payload = canonical_json({"base": 20260810, **scope})   # §10.2 canonicalization
    return int(sha256(payload.encode()).hexdigest()[:8], 16) % (2**31 - 1)
```

`scope` keys, always all present: `{"stage", "arm", "dataset", "outer", "inner", "config"}`, with
`inner = -1` for an outer refit and `outer = -1` for a confirmation refit (§7.10).
Epoch shuffling uses `torch.Generator().manual_seed(derive_seed(scope) + epoch)`.
Every derived seed is written to `folds/fold_<k>/selected_hparams.json`.

Determinism switches, pinned: `torch.use_deterministic_algorithms(True)`,
`torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`,
`PYTHONHASHSEED=0`, `torch.set_num_threads(8)`, device `cpu`.

### 3.5 Config enumeration and the single tie-break rule

Every arm's configuration list is enumerated in a **registered fixed order**; selection is `argmax`
of the relevant pooled inner-OOF macro-F1 with one global tie rule:

```text
ties -> smaller epoch ; still tied -> smaller index in the registered config list
```

Registered iteration order (outermost first): `arm_local` → `lr` → `weight_decay`, with

- `arm_local`: A2 `k ∈ (1, 2, 4)`; A4 `τ ∈ (0.1, 0.3, 1.0)`; all other arms `(none,)`;
- `lr ∈ (1e-2, 3e-3, 1e-3, 3e-4)` — **`1e-2` added post-review** (review OP-6 free
  recommendation, accepted): it costs a few CPU-minutes and permanently removes the "you under-fit
  A3's 231k-parameter attention MLP" objection from any future `NO-GO-A-SELECTOR` writeup;
- `weight_decay ∈ (1e-4, 1e-2)`.

So A2/A4 have 24 configs, every other arm has 8. Because ties resolve to the smaller config index,
A2 ties favour smaller `k`, A4 ties favour smaller `τ`, and lr ties favour the larger lr. These are
consequences of the one global rule, not separate rules.

---

## 4. Gate-A arms — exact instantiation

All heads take `d = Dv + Dt` inputs and emit one logit; loss is `torch.nn.BCEWithLogitsLoss()` with
**no** `pos_weight`, no class-balanced sampler, and no label smoothing, identically for every arm.

Linear-head init convention (matching `scripts/analysis/p11_probe_hatemm.py:181-183`):
`torch.nn.init.normal_(weight, 0.0, 0.01)`, `torch.nn.init.zeros_(bias)`. Hidden `nn.Linear` layers
(A3's attention MLP, B0–B3's trunks, B2's projection) use PyTorch's **default** `nn.Linear` init
(Kaiming-uniform `a=√5`, bias `U(±1/√fan_in)`), deterministic under the derived seed. Only the
final scalar output layer uses the `normal_(0,0.01)/zeros_` convention. Bias present on every layer
unless stated.

**L2 regularization** is applied exclusively through AdamW's `weight_decay` (grid in §3.5), on all
parameters including biases. No separate explicit L2 term. Uniform across arms, so prereg §5.1's
"matched optimizer budget" holds.

### 4.1 A0 — whole-video baseline

```text
logit_v = w · x_v + b            (nn.Linear(d, 1));   score_v = σ(logit_v);   params: d + 1
```

### 4.2 A1 — mean of K=30 segment representations, same head

```text
m_v = (1/K) Σ_k s_{v,k} ;  logit_v = w · m_v + b   (nn.Linear(d,1), separate instance from A0)
```

"Same head" means same head *form*, not shared weights. Because the head is affine,
`logit(m_v) = mean_k logit(s_{v,k})` exactly; §5 uses this identity to make O1 well-defined under
either reading of "pool the segment scores".

### 4.3 A2 — max / top-k MIL

**Operates on SCALAR SEGMENT LOGITS, in both training and evaluation** (prereg §11.1).

```text
z_{v,k} = w · s_{v,k} + b
T_v(k)  = indices of the k largest z_{v,·}, ties -> smaller window index
logit_v = (1/k) Σ_{j ∈ T_v(k)} z_{v,j}
```

`k = 1` is max-MIL. `k ∈ {1,2,4}` selected only from pooled inner-OOF macro-F1 under §3.5. Hard
top-k in train and eval; no soft relaxation, no straight-through estimator; gradient flows only
through the selected `k` segments.

Rationale for scalars: a top-k over vectors is undefined without a scalar ranker (which would
silently reintroduce one); it matches `p11_probe_hatemm.py:186-195`; and it makes the §8 temporal
scores the same objects the arm optimizes.

### 4.4 A3 — learned attention pooling

Non-gated, single-hidden-layer, tanh attention.

```text
h_k = tanh( W s_{v,k} + c )      W ∈ R^{H×d}, c ∈ R^H,  H = 128
a_k = u · h_k                    u ∈ R^H (nn.Linear(H,1,bias=False))
α   = softmax(a)                 over k = 0..29
p_v = Σ_k α_k s_{v,k} ;  logit_v = w · p_v + b   (nn.Linear(d,1))
```

Pinned: `gated = false` (the gated variant is a different architecture; exactly one instantiation
is registered); `H = 128`; one hidden layer; `tanh`; no dropout, no layer norm, no softmax
temperature; **video-level BCE only** — no entropy regularizer, no sparsity penalty, no
segment-level target. Params `H·d + H + H + d + 1` (expected **231,425** at `d=1792`).

**Recorded limitation — Gate-A capacity asymmetry (review OP-3, confirmed as a limitation, not to
be repaired).** A3 has ~231k parameters against ~1.8k for A0/A1/A2/A4, and the pre-registration
installs a capacity control only in Gate-B (B3). Adding one to Gate-A would be a new arm, which
prereg §11.1 forbids the appendix from doing. Two facts bound the confound and are registered here:

1. prereg §5.2 **criterion 5 is itself the capacity/selection discriminator** — a pure capacity
   effect raises video-level macro-F1 without localizing, and Gate-A requires **all six** criteria,
   so a capacity-only pass on criterion 3 cannot promote the route;
2. `D` is chosen among A2/A3/A4, and A2/A4 are ~1.8k-parameter arms — if `D` is A2 or A4 the
   confound is moot for the promoted arm.

**Required record:** any Gate-A pass with `D = A3` must carry the sentence *"the A3 advantage over
A0/A1 is not capacity-controlled at Gate-A"* in `verdict.json`.

### 4.5 A4 — log-sum-exp pooling

**Same scalar segment logits as A2** (prereg §11.1 consistency requirement).

```text
logit_v = τ · ( logsumexp_k(z_{v,k} / τ) − log K )
```

The `− log K` term makes the operator interpolate mean (`τ → ∞`) and max (`τ → 0⁺`) on the same
scale as the logits, so the shared threshold rule and lr grid stay comparable to A1/A2.
`torch.logsumexp`; `τ ∈ {0.1, 0.3, 1.0}` selected only from pooled inner-OOF macro-F1 under §3.5.

### 4.6 Summary

| arm | pools | operand | params (expected, `d=1792`) | arm-local grid | configs |
|---|---|---|---:|---|---:|
| A0 | — | whole-video vector | 1,793 | — | 8 |
| A1 | mean | segment vectors | 1,793 | — | 8 |
| A2 | top-k mean | **scalar logits** | 1,793 | `k ∈ {1,2,4}` | 24 |
| A3 | attention | segment vectors | 231,425 | — | 8 |
| A4 | LSE | **scalar logits** | 1,793 | `τ ∈ {0.1,0.3,1.0}` | 24 |

---

## 5. O1 and O2 — deterministic oracle rules

Both oracles reuse the **A1 fold-trained linear head** as the "fixed fold-trained segment scorer"
of prereg §5.1 — the tightest matched choice, making A1 / O1 / O2 an exact nested family differing
only in the selected window set. No new parameters are trained. Both are marked
`oracle_or_eval_only: true` on every row they touch, and neither may select a deployable arm.

**Recorded reading (ratified by review §1.9, ex ante):** O1 and O2 receive their own thresholds
from their own pooled inner-OOF scores, i.e. gold-span-derived information reaches an *oracle*
threshold. Prereg §3's "must not affect ... thresholds" governs *deployable* arms and is
immediately followed by "Gold spans may be read only by ... the explicitly named non-deployable
oracle/evaluation routines". Giving O1 a gold-blind threshold would make the oracle bound
meaningless, since O1's scores live on a different scale.

### 5.1 O1 — gold-span pooling (executable pseudocode)

```python
# INPUTS
#   z[v, k]  : A1 fold-trained segment logits, k = 0..29
#   spans[v] : list of [start_s, end_s] from the ID-RESTRICTED hate_spans.json (§2.8)
#   D[v]     : duration from the same restricted object
# The routine may inspect span PRESENCE but must never branch on the video label.

def o1_video_logit(v, z, spans, D, K=30):
    W = []
    if D[v] is not None and D[v] > 0:
        for k in range(K):
            lo, hi = k * D[v] / K, (k + 1) * D[v] / K
            for (a, b) in spans.get(v, []):
                if min(hi, b) - max(lo, a) > 0.0:      # strict POSITIVE-duration overlap
                    W.append(k); break
    if len(W) == 0:
        sel, fallback = list(range(K)), True           # registered A1 mean-pooling fallback
    else:
        sel, fallback = W, False
    return sum(z[v, k] for k in sel) / len(sel), sel, fallback
```

Notes: overlap is strict positive duration; because the head is affine, pooling logits over `sel`
equals applying the head to the mean of `s_{v,k}` over `sel`, so both readings of prereg §5.1
coincide and the fallback is exactly registered A1 mean pooling. A video with `D` missing takes the
fallback and is flagged `missing_duration: true`. Negatives typically have no spans and take the
fallback — the registered behaviour, and precisely why O1 is non-deployable; the `o1_fallback` rate
is reported. O1's threshold comes from pooled inner-OOF O1 scores (§3.3), computed inside the same
nested loop.

### 5.2 O2 — true-label-aware best candidate subset

```python
def o2_video_logit(v, z, y_true, K=30):
    # The label-aware optimum of mean_{k in S} z over non-empty subsets S is attained at a
    # singleton, so no subset search is needed.
    k_star = argmax_k(z[v, :]) if y_true[v] == 1 else argmin_k(z[v, :])   # ties -> smaller k
    return z[v, k_star], [k_star]
```

Exactly the rule prereg §5.1 names. `label_leaking: true` in addition to
`oracle_or_eval_only: true`. Threshold from pooled inner-OOF O2 scores (§3.3).

**Guaranteed inequality (fixture F3):** per video, `O2_logit ≥ O1_logit` when `y=1` and `≤` when
`y=0`, because a max (min) over `k` dominates (is dominated by) any mean over a non-empty subset.
Asserted at run time on synthetic and real data inside the `oracle_or_eval_only` routine.

### 5.3 D — the promoted deployable arm, and the cross-fold aggregation (BLOCKING-FIX B-2)

```text
D = argmax_{arm ∈ {A2, A3, A4}} ( POOLED-INNER-OOF-MACRO-F1(arm) )
ties -> arm-id order A2 < A3 < A4
```

**`POOLED-INNER-OOF-MACRO-F1(arm)` is defined as follows and only as follows.** Concatenate the
inner-held-out predictions of **all 5 outer folds**, each fold contributing its own selected
`(cfg*, epoch*)` at its own `θ*`, and compute **one** macro-F1 on that single concatenation. It is
never a mean of per-fold macro-F1 values — the same convention as prereg §7's primary metric.

Two consequences are recorded so they are not discovered later:

- the quantity is selection-optimistic (each fold's `cfg*/epoch*/θ*` were chosen to maximize that
  fold's own inner-OOF macro-F1), but it is computed identically for A2, A3 and A4 and is used
  only to *rank* them, never as a reported performance number;
- it uses no outer-query prediction, no val, no test, and no oracle, so prereg §5.2's "selected
  solely by pooled inner-OOF macro-F1" is satisfied literally.

`D`'s identity, its per-fold `cfg*`, `epoch*` and `θ*`, and the three arms' aggregated numbers are
written to `metrics.json` **before** any Gate-B computation and are frozen for Gate-B.

---

## 6. Gate-B arms — exact instantiation

Gate-B runs only after a full A pass, freezes `D` as the segment-scoring/selection basis, and does
not reopen A hyperparameters (prereg §6). B arms are new heads trained on the frozen pair
selection, using the same folds, grid, budget, and threshold rule as Gate-A.

### 6.1 Where D's segment scores come from (no in-sample leakage)

```text
for outer fold f:
    v in outer-query(f) : D's outer-fold-f model  (cfg*, epoch* of fold f)
    v in outer-train(f) : D's INNER-OOF predictions within outer-train(f), produced by D's
                          inner-fold models AT THAT OUTER FOLD'S SELECTED cfg* AND epoch*
```

*(The second clause's config source was previously unstated — review B-2, second half.)* Recorded
per row as `d_score_source ∈ {"outer_fold_model", "inner_oof"}` plus `d_config_id` and `d_epoch`.

### 6.2 Top-two selection (minimum separation 2 windows)

```python
def select_pair(seg_score_row, K=30, min_sep=2):
    i1   = argmax_k(seg_score_row)                     # ties -> smaller k
    cand = [k for k in range(K) if abs(k - i1) >= min_sep]
    i2   = argmax over cand of seg_score_row           # ties -> smaller k
    a, b = min(i1, i2), max(i1, i2)                    # TEMPORAL order: earlier slot first
    return i1, (a, b)
```

`min_sep = 2` means `|i1 − i2| ≥ 2`; adjacent windows are ineligible, exactly as prereg §6 states.
With `K = 30` the candidate set is never empty. `(e_first, e_second) = (s_{v,a}, s_{v,b})`,
`e_top = s_{v,i1}`. Selection is a pure function of D, which never saw a span or segment label.

### 6.3 Shared trunk

```text
P : nn.Linear(d, r),  r = 128, default init            # each arm has its OWN P instance
g : nn.Linear(in, H) -> ReLU -> nn.Linear(H, 1)        # H = 64 unless stated (B3: H3)
p = P(e_first),  q = P(e_second),  p_top = P(e_top)
```

Final scalar layer `normal_(0,0.01)/zeros_`; `P` and the hidden layer default init.

### 6.4 Relative-time encoding and the pair interaction (B2)

With **presented** slots `(A, B)` at window indices `(iA, iB)`:

```text
tA = iA/(K-1)   tB = iB/(K-1)   δ = tB - tA          (δ ∈ [-1, 1])
φ  = [ tA, tB, δ, |δ|, sin(π δ), cos(π δ) ]         ∈ R^6
in_B2   = concat( p, q, p ⊙ q, φ )                  ∈ R^{3r+6} = R^{390}
logit_v = g_B2(in_B2)                                # 390 -> 64 -> 1
```

`sin(πδ)` and the `[tA,tB]` order are odd in the swap and carry temporal order; `|δ|` and `cos(πδ)`
are even and carry only separation. `p ⊙ q` is the registered interaction term; a full bilinear
form is `d²` parameters and is not registered. `P` is shared between slots, so order information
lives only in the concatenation order and `φ` — which is what makes B4 a clean lesion.

### 6.5 B0, B1, B3, B4

```text
B0  in = p_top            (R^128)  -> g(128 -> 64  -> 1)
B1  in = (p + q)/2        (R^128)  -> g(128 -> 64  -> 1)     # non-interactive, no φ
B3  in = p_top            (R^128)  -> g(128 -> H3  -> 1)     # capacity control
B4  = B2 with the presented slot order permuted per video (below)
```

B1 is the mean of the top-two segment representations: `P` is linear, so
`P((e_first+e_second)/2) = (p+q)/2`.

**B3 capacity match (within-5% construction rule).**

```text
params(P)  = d·r + r
params(B2) = params(P) + (390·H + H) + (H + 1),            H = 64
params(B3) = params(P) + (r·H3 + H3) + (H3 + 1)
H3* = argmin_{H3 ∈ N, H3 ≥ 1} | params(B3) − params(B2) | ;  ties -> smaller H3
assert |params(B3) − params(B2)| / params(B2) ≤ 0.05        # else HALT
```

At `d = 1792, r = 128, H = 64`: `params(P) = 229,504`, `params(B2) = 254,593`, `H3* = 193`,
`params(B3) = 254,595`, relative difference `7.86e-6`. B0/B1 sit at `237,825` (6.59% below B2),
which is exactly why prereg §6 requires B3 separately. The *head* capacities are matched too —
B2's `390→64→1` is 25,024 parameters against B3's `128→193→1` at 25,090 (0.26% apart) — so the
match is not trivially satisfied by the shared projection. `H3` is computed by the rule at run time
from the observed `d`, not hard-coded, and asserted.

**B4 — temporal-order lesion** (review OP-1: **RATIFIED**, adjudicated 2026-08-07 before any
candidate metric).

```python
rng4 = numpy.random.default_rng(20260807)
swap = {vid: bool(rng4.integers(0, 2)) for vid in sorted(all_video_ids)}   # id-sorted
(iA, iB) = (b, a) if swap[vid] else (a, b)
# φ is recomputed FROM THE PRESENTED SLOTS, so δ flips sign exactly on swapped videos
```

The same per-video draw is used in training **and** evaluation, so B4 is a fixed dataset, not a
stochastic augmentation. Mandatory assertions (prereg §11.1):

1. realized swap fraction ∈ **`[0.40, 0.60]`**, else HALT. *(Widened from v1's `[0.45,0.55]` per
   review N-5: at `V ≈ 744` the fraction has SE ≈ 0.018, so `[0.45,0.55]` is ±2.7σ and would trip
   ~0.7% of the time with **no legal recovery**, the seed being prereg-pinned. `[0.40,0.60]` is
   ±5.5σ, which still catches a broken or degenerate draw — the only thing the assertion is for.
   The exact realized fraction is recorded in `metrics.json` regardless.)*
2. for every swapped video, `in_B4 ≠ in_B2` and `sign(δ_B4) = −sign(δ_B2)`, else HALT;
3. no downstream code re-sorts slots by time (asserted by `iA > iB` on swapped videos);
4. the realized swap set is **identical between the training pass and the evaluation pass**
   (asserted, per review N-11).

**Why an independent per-video coin flip rather than "always swap":** an unconditional swap is an
invertible relabelling — `φ_B4 = [t_b, t_a, −δ, |δ|, −sin πδ, cos πδ]` carries exactly the same
information as `φ_B2`, so a B4-trained model simply relearns the reversed convention, `B2 − B4 ≈ 0`
by construction, and Gate-B would return a near-certain **false** `NO-GO-B` regardless of whether
temporal order matters. The Bernoulli(0.5) draw destroys the dataset-level correspondence between
presented order and true temporal order — the quantity H-B is about — while retaining both
segments' content exactly as §11.1 requires. Assertions 1–4 close the degenerate-permutation
failure modes §11.1 warns about. The alternative (zeroing `φ`'s order-carrying components) is a
*different* lesion and would be a §12 material deviation; the review explicitly did not request it.

### 6.6 B5 — second-evidence lesion

```python
rng5 = numpy.random.default_rng(20260807)      # SEPARATE generator instance from rng4

# Dpred(u): D's frozen BINARY prediction at D's frozen inner-OOF threshold.
#   u in outer-train(f) -> D's inner-OOF prediction inside outer-train(f)
#   u in outer-query(f) -> D's outer-fold-f prediction
# The query's TRUE label is never consulted.

def legal_support(v, f):
    return sorted(set(outer_train_ids[f]) - {v})     # for BOTH training and query rows

def b5_replacement(v, f):
    pool     = legal_support(v, f)
    stratum  = [u for u in pool if Dpred[u] == Dpred[v]]
    if stratum: return rng5.choice(stratum), False
    else:       return rng5.choice(pool),    True    # record the fallback

q_lesion = P( e_second[donor] )                      # donor's own second-slot segment
in_B5    = concat( p, q_lesion, p ⊙ q_lesion, φ_query )   # query's OWN φ retained
```

Pinned: the support partition is `outer_train(f)` for both training and held-out-query rows, so no
draw crosses an outer fold and a query video never donates to another query video (prereg §6,
§11.1); only the second slot's **content** is replaced, with the query's own `φ` retained, so B5
(identity of the second evidence unit) stays orthogonal to B4 (temporal relation); the donor
contributes its own `e_second` under D's top-two selection, not a random window; `rng5` is consumed
in ascending sorted video-id order; an empty stratum falls back to a uniform draw from the full
legal pool with `b5_fallback: true` per row and an aggregate `b5_fallback_count`; `Dpred` is a
*predicted* label and the routine's closure is asserted to hold no reference to the label array.

### 6.7 Gate-B decision inputs, and the `msc`-subset conventions (BLOCKING-FIX B-5)

From outer-OOF predictions (prereg §6): `B2 − max(B0,B1,B3) ≥ +0.020` with paired bootstrap CI
excluding zero; `B2 − B4 ≥ +0.015`; `B2 − B5 ≥ +0.015`; the rescue criterion below; and positive B2
deltas on HateMM-val and HateClipSeg-val (§7.10).

**Registered definition of the frozen Gate-C `multi_segment_complementary` (msc) subset:**

> The msc subset is the set of **audited videos of any category** — audited false negatives **and**
> the 30 TP + 30 FP controls — carrying `multi_segment_complementary` as **primary or secondary**
> cause, per prereg §4.3's presence rule. It is frozen when Gate-C's adjudicated audit is written
> and is stored as `msc_subset.json` with a SHA256 in `manifest.json`.

**Rescue rate** (label-1 members only):

```text
rescue = |{ v ∈ msc, label(v)=1 : B0 predicts 0 and B2 predicts 1 }| / |{ v ∈ msc, label(v)=1 : B0 predicts 0 }|
criterion: rescue ≥ 0.20 ; report exact numerator/denominator and a Wilson 95% interval
if the denominator is 0 -> record "not_evaluable" and treat the criterion as NOT satisfied
   (a rescue criterion with nothing to rescue cannot license a pass)
```

**False-positive side condition** (label-0 members only), expressed in **counts** so `0/0` cannot
arise as a ratio:

```text
FP_arm = |{ v ∈ msc, label(v)=0 : arm predicts 1 }|
criterion: FP_B2 ≤ FP_B0 + max(1, ceil(0.10 · FP_B0))
if the msc subset contains no label-0 member -> record "not_evaluable" and treat the side
   condition as SATISFIED (it is a do-no-harm guard, not a positive requirement)
```

The asymmetry between the two `not_evaluable` conventions is deliberate and registered: the rescue
rate is a *positive* requirement (unmet ⇒ fail), the FP condition is a *do-no-harm guard* (vacuous
⇒ pass).

*(review N-12, recorded)* With ≤120 audited FNs and prereg §4.3's msc ≥ 15% pass floor, the msc
label-1 subset could be ~18 videos. The prereg sets no minimum; the exact numerator/denominator
plus Wilson interval are the registered treatment, and the writeup must not over-read the rate.

---

## 7. Nested protocol, optimization, budget, and confirmation

### 7.1 Folds

- Outer: 5 video-stratified folds, seed `20260807`. Inner: 4 video-stratified folds within every
  outer-training partition, seed `20260808`.
- `sklearn.model_selection.StratifiedKFold(n_splits, shuffle=True, random_state=seed)` applied to
  the **sorted** video-id list with the video-level binary label as target, making fold membership
  independent of cache row order. Assignments written to `folds/fold_<k>/{train_ids,query_ids}.json`
  with SHA256 in `split_manifest.json`.
- All 30 segments of a video travel with the video (prereg §3). Asserted.
- Either class below the fold count → `HALT_FOLD_INFEASIBLE` (prereg §7).

### 7.2 Per outer fold — the single registered selection loop (BLOCKING-FIX B-3)

v1 contained two conflicting epoch rules (§7.2's "argmax over all cfg/epoch" vs §7.4's
patience-truncated best). They are unified here into **one** rule, with the four inner folds
advanced in lockstep so the patience break is genuinely both a compute saver and the selection
rule *(review B-3 + N-10)*:

```text
for f in 0..4:
  for cfg in registered_configs(arm):            # §3.5 order
      init 4 inner models, model_j seeded by derive_seed({stage,arm,dataset,outer=f,inner=j,config=cfg})
      best, best_epoch, best_theta, stall = -inf, 1, None, 0
      for epoch in 1..E_max:                     # E_max = 200
          for j in 0..3:
              train model_j for ONE epoch on inner-train(f,j)
              score inner-held-out(f,j)
          pool the 4 inner-held-out score sets FOR THIS EPOCH
          theta = threshold rule §3.3 on the pooled scores
          m     = pooled inner-OOF macro-F1 at theta
          if m > best + 1e-4:  best, best_epoch, best_theta, stall = m, epoch, theta, 0
          else:                stall += 1
          if stall >= 40: break                  # patience = 40
      candidate[cfg] = (best, best_epoch, best_theta)

  (cfg*, epoch*, theta*) = argmax over candidate[·] of best
                           ties -> smaller epoch ; then smaller cfg index   (§3.5)
  refit ONCE on the full outer-train with cfg*, exactly epoch* epochs, inner = -1 seed
  score outer-query(f) ONCE ; apply theta*
  assert: every video in exactly one outer-query fold; no video/segment/derived id appears in
          both outer-train(f) and outer-query(f); inner folds nest inside outer-train(f)
```

The primary metric concatenates all outer-query predictions and is computed once on the
concatenation (prereg §7), never as a mean of fold metrics.

### 7.3 Optimizer

| field | value |
|---|---|
| optimizer | `torch.optim.AdamW`, betas `(0.9, 0.999)`, eps `1e-8`, `amsgrad=False` |
| lr grid | `{1e-2, 3e-3, 1e-3, 3e-4}` |
| weight_decay grid | `{1e-4, 1e-2}` |
| lr schedule | constant (no warmup, no decay) |
| batch size | `64` videos (a sample is always a video, never a segment) |
| shuffling | per-epoch permutation from `torch.Generator().manual_seed(derive_seed(scope)+epoch)` |
| last partial batch | kept (`drop_last=false`) |
| gradient clipping | none |
| loss | `BCEWithLogitsLoss()`, unweighted |
| `E_max` / patience / `min_delta` | `200` / `40` / `1e-4` |
| dtype / device | `float32` / `cpu` |

### 7.4 Epoch selection

Epoch is a selected hyperparameter chosen from pooled inner-OOF macro-F1 (prereg §7 step 2), by the
**single** rule embedded in §7.2's loop: the patience-truncated best-so-far epoch per config, then
`argmax` across configs with §3.5's tie rule. The truncation *is* part of the registered rule and
is applied identically to every arm, config and fold. *(v1's claim that early stopping "must not
change the selection" was false and is deleted — review B-3.)*

### 7.5 Budget alignment across arms

Identical optimizer, `lr × weight_decay` grid, `E_max`, `patience`, `min_delta`, `batch_size`,
loss, dtype, device and loop structure for every arm. The arm-local grids (`k`, `τ`) are registered
by the prereg itself and are additional, so A2/A4 evaluate 24 configs and the others 8. This
asymmetry is contained by proper nesting (extra configs are selected on inner-OOF and *evaluated*
on outer-OOF) and is made visible, not hidden, by `metrics.json.budget_report`.

Expected totals with the 4×2 grid:
A-stage `5·(8·4+1)·3 + 5·(24·4+1)·2 = 495 + 970 = 1465` head trainings;
B-stage `5·(8·4+1)·6 = 990`. Each training is seconds of CPU at these sizes.

*(review N-8, recorded)* `batch_size = 64` with `drop_last = false` means the outer refit (on a
~33% larger partition) takes more gradient steps at `epoch*` than an inner fit did. This is
standard nested practice, prereg §7 step 3 says "using the selected fixed epoch/budget", and it is
applied identically to every arm. Recorded in the limitations section.

### 7.6 Complete seed register

| purpose | seed | source |
|---|---|---|
| outer folds | `20260807` | prereg §7 |
| inner folds | `20260808` | prereg §7 |
| video bootstrap (primary metrics) | `20260809` | prereg §8.3 |
| temporal-metric bootstrap (eligible set) | `20260809`, separate draw | this appendix, §8.2 |
| Gate-C sampling / double-coding selection | `20260807` | prereg §4.1 |
| Gate-C coverage bootstrap | `20260809` | this appendix, §11.3 |
| B4 order permutation | `20260807` (`rng4`) | prereg §6 |
| B5 donor draw | `20260807` (`rng5`, separate instance) | prereg §6 |
| model init (base) | `20260810` | this appendix, §3.4 |
| synthetic fixtures (base) | `424242` | this appendix, §9 |

`rng4` and `rng5` are separate `default_rng` instances so B5's draw cannot be perturbed by B4's
consumption; both are consumed in ascending sorted video-id order.

### 7.7 Bootstrap

10,000 paired resamples of videos, seed `20260809`, stratified by video label: within each label
stratum draw `n_label` indices with replacement. Indices generated once with
`numpy.random.default_rng(20260809)` over the **sorted** outer-OOF video-id list, saved to
`bootstrap_indices.npz` (`int32[10000, N]`), and **reused identically** by every arm and every
paired delta — that is what makes the deltas paired. All segments/seconds of a sampled video travel
with it. Macro-F1 recomputed inside each resample at the frozen threshold. CIs are percentile
`[2.5, 97.5]`. No multiple-testing adjustment (prereg §8.3).

### 7.8 Metrics computed

Primary: binary macro-F1. Secondary: balanced accuracy, accuracy, positive-class F1, AUROC.
Diagnostic: predicted-positive rate, confusion matrix. Temporal: §8.

### 7.9 Why CPU

Heads are ≤ 255k parameters on ≤ 744 × 30 × 1792 float32 inputs. CPU gives bitwise reproducibility
under a pinned thread count and removes GPU kernel nondeterminism. The only GPU step in the plan is
the §2.4 feature extraction. `manifest.json` records `gpu_model` regardless (prereg §11).

### 7.10 Confirmation-set protocol, registered (BLOCKING-FIX B-1)

Prereg §5.2 item 6 and §6 bullet 4 are **binding pass conditions**, so the procedure that produces
a confirmation number is registered here in full. It runs **once**, after every outer-OOF number is
computed and every choice is frozen, and immediately after the single `unlock_confirmation()` call
of §2.8. **A confirmation set may not select an arm, threshold, epoch, granularity, config, or
decision rule** (prereg §2.1, §8.1). No quantity below is derived on val.

#### 7.10.1 HateMM-val

**Scoring model — one refit on the whole of HateMM-train.** For each arm needed
(A0, A1, D for Gate-A; B0, B1, B2, B3 for Gate-B), the confirmation model is a **single refit on
the entire HateMM-train partition**, using hyperparameters transferred deterministically from the
five outer folds:

```text
cfg_val   = the modal cfg* across the 5 outer folds ; ties -> smaller registered config index
epoch_val = median of the 5 outer folds' epoch*      (5 values, odd count -> exact median)
theta_val = median of the 5 outer folds' theta*      (5 values, odd count -> exact median)
seed      = derive_seed({stage, arm, dataset:"HateMM", outer:-1, inner:-1, config:cfg_val})
```

All three quantities are functions of inner-OOF-derived selections only. The refit runs exactly
`epoch_val` epochs on all of HateMM-train with `cfg_val`.

**Scoring and thresholding.** Score HateMM-val **once** with that single model and apply
`theta_val` unchanged. The threshold is **never** re-derived, re-optimized, or tie-broken on val.

**Segment scores on val.** `D`'s per-segment scores on HateMM-val come from `D`'s val-refit model,
under the §3.1 singleton rule. Gate-B's pair selection on val uses those scores through the
unchanged §6.2 procedure. (B4 and B5 are not needed on val: prereg §6's val condition is on the B2
delta only. If they were, B5's donor pool would be all of HateMM-train.)

**Criteria evaluated.**
Gate-A: `macroF1(D_val) − max(macroF1(A0_val), macroF1(A1_val)) > 0`.
Gate-B: `macroF1(B2_val) − max(macroF1(B0_val), macroF1(B1_val), macroF1(B3_val)) > 0`.
Neither requires significance (prereg §5.2 item 6).

#### 7.10.2 HateClipSeg-val

Prereg §2.2 makes HateClipSeg-**train** the development partition. The registered fitting procedure
is: **fit every arm on HateClipSeg-train with the same architectures and the same registered grid,
selecting by the same rules, then score HateClipSeg-val once.**

```text
partition:   p11_split["train"]  (237 videos, development)
selection:   4 inner video-stratified folds on HateClipSeg-train, seed 20260808,
             the SAME §7.2 loop, the SAME §3.5 config list and tie rule, the SAME §3.3
             threshold rule -> (cfg*, epoch*, theta*) from pooled inner-OOF only.
             No outer loop is used: HateClipSeg-val is the evaluation set, not an OOF construction.
refit:       ONCE on all of HateClipSeg-train, cfg*, exactly epoch* epochs,
             seed = derive_seed({stage, arm, dataset:"HateClipSeg", outer:-1, inner:-1, config:cfg*})
score:       HateClipSeg-val ONCE ; apply theta* unchanged.
arms fitted: A0, A1, D (Gate-A) and B0, B1, B2, B3 (Gate-B).
```

**What transfers and what does not.** Only `D`'s **arm identity** (A2, A3 or A4) transfers from
HateMM. `D`'s arm-local value (`k` or `τ`), `lr`, `weight_decay`, `epoch` and `θ` are **re-selected
on HateClipSeg-train inner-OOF** under the identical registered grid. Rationale: prereg §2.2 admits
only "matched method comparisons on the identical surviving IDs"; fitting every arm identically on
HateClipSeg-train satisfies that, whereas transferring a HateMM-fitted head would be a
domain-transfer experiment that is not the registered comparison. The feature family and dimensions
are identical across the two datasets, so the architectures transfer unchanged.

**Weak-supervision boundary statement (explicit, per review B-1 item 4).** The HateClipSeg binding
endpoint — `has at least one segment labelled hateful` — is *derived* from gold segment labels, but
it is consumed **only as a video-level binary label**: as the training target, the selection target
and the evaluation target, exactly as prereg §3 permits ("allowed supervision: video-level binary
label"). **No per-segment gold label, no gold segment boundary, and no gold-derived segment mask,
weight, or pooling set ever enters any HateClipSeg deployable path.** No HateClipSeg temporal
metric is computed in Gate-0 (prereg §8.2's temporal criterion is HateMM-only), so gold segment
boundaries are never read at all on the HateClipSeg side beyond the endpoint derivation. This is
asserted in code: the endpoint derivation returns a single `int` per video and the segment list is
discarded before returning.

**Underpower.** Per prereg §2.2, the binding-endpoint class counts in the P11 validation split are
reported, and if either class has `< 10` videos the confirmation is recorded as **underpowered**
and cannot satisfy any criterion. `any-toxic` is descriptive only and can never replace a failed
binding confirmation.

#### 7.10.3 Single-pass discipline

Each confirmation dataset is scored exactly once per stage. `manifest.json` records
`confirmation_passes{hatemm_val, hateclipseg_val}` which must each be `1`; a second pass is a HALT.

---

## 8. Temporal evaluation (prereg §5.2 item 5, §8.2)

### 8.1 Eligible video set (frozen once, shared across arms)

```text
eligible(v) := label(v) == 1
           AND v has >= 1 parsed gold span
           AND D_v is not null and D_v > 0
           AND the video contains at least one positive AND at least one negative second
               under the §3.2 midpoint rule
```

Computed once from the id-restricted gold spans before any arm's temporal metric, written to
`eligible_videos.json` with a SHA256 in `manifest.json`, reused unchanged by every arm (prereg
§5.2 item 5). Its size is reported.

### 8.2 Metrics

- **Primary temporal metric**: mean over eligible videos of the within-video second-level AUROC,
  each second scored by its window's registered per-segment score (§3.1, §3.2). Reported with the
  eligible-video count and a video-level bootstrap CI.
- **Temporal bootstrap index set** *(review N-7, registered)*: a **separate** draw from
  `numpy.random.default_rng(20260809)` over the **sorted eligible video list**, 10,000 resamples,
  stratified trivially (all eligible videos are label-1), saved alongside the master indices as
  `bootstrap_indices.npz:temporal`. It is not a filtering of the master indices. Criterion 5 is a
  point-estimate threshold; the CI is reported, not binding.
- **A0 comparator**: A0's video score broadcast to every second gives within-video AUROC exactly
  `0.5` per eligible video. Criterion 5 therefore requires D's mean `≥ 0.60` **and** `≥ 0.53`; both
  are checked explicitly.
- **A3 α-based diagnostic**: §3.1, non-binding.
- Secondary: pooled full second-level AP and AUROC over all evaluated seconds (documented as
  containing between-video separability).
- `gold-span recall@n`, `n ∈ {1,2,4}`: fraction of eligible videos for which at least one of the
  `n` highest-scoring windows overlaps a gold span (positive duration).
- Selected-vs-unselected separation: mean score of gold-overlapping windows minus mean score of the
  rest, per eligible video, averaged; and within-video score standard deviation.
- Per-video AP is **not** averaged (prereg §8.2).

---

## 9. Deterministic synthetic fixtures

Fixtures are the pre-execution correctness evidence required by prereg §11.1. **They use only
synthetic data.** No fixture reads `data/`, `/home/jehc223/data/`, or any real label or span.

### 9.1 Harness

- `scripts/tera_gate0/fixtures.py` *(v3 path correction — see §13 reading R-1; v2 wrote
  `scripts/analysis/tera_gate0_fixtures.py`, which is not where the harness was built)*; run
  detached per §0.3 (`TASK=tera_gate0_fixtures`).
- Output `artifacts/tera_gate0/_fixtures/<fixture_run_id>/fixtures_report.json` — outside the
  registered run namespace, referenced by SHA256 from `frozen_config.json`.
- Reduced dims for speed: `Dv_fix = 32`, `Dt_fix = 16`, `d_fix = 48`, `K = 30` (unchanged — the
  registered grid must be exercised), `V = 240`, positives 40%.
- Generated objects use **exactly** the real schemas of §2.2 (including the nested `"ids": [[...]]`
  whole-video contract) and §2.9, so the production code path runs end to end (folds → inner OOF →
  threshold → refit → outer OOF → bootstrap → verdict), not a mock.
- Fixture seed base `424242`; fixture `i` uses `default_rng(424242 + i)`.
- `H3` recomputed by §6.5 at `d_fix` and separately asserted at `d = 1792`.
- Bootstrap reduced to 1,000 resamples inside fixtures (`fixture_bootstrap_n`); the registered run
  uses 10,000.
- Every fixture additionally asserts the run terminates with a verdict and **no HALT**, unless it
  is explicitly a HALT fixture.

### 9.2 The battery

| id | construction | assertions |
|---|---|---|
| **F1** span-localized signal | each positive video: one synthetic span covering exactly windows `{c, c+1}`, `c ~ U{2,…,26}`, those windows get `+1.2·e₁`; everything else `N(0,1)`; text half noise for all | `O1 − max(A0,A1) ≥ +0.050`; `max(A2,A3,A4) − max(A0,A1) ≥ +0.020`; A2's selected `k ∈ {1,2}`; D's mean within-video second-level AUROC `≥ 0.60`; gold-span recall@1 `≥ 0.80`; no HALT |
| **F2** global signal only | positives get `+0.20·e₁` on **every** window | `O1 − max(A0,A1) < +0.050`; `A1 ≥ A0 − 0.02`; verdict `NO-GO-A-NO-HEADROOM`; no HALT |
| **F3** oracle ordering | F1 plus sign-flipped noise spikes on a random 20% of windows | per-video `O2 ≥ O1` for `y=1` and `≤` for `y=0`, 100% of videos; `O2 − max(A0,A1) ≥ +0.050` |
| **F4** ordered pair interaction | `pA = +1.5·e₁`, `pB = +1.5·e₂`; label 1 iff `pA` at `i` and `pB` at `j` with `j − i ≥ 2`; label 0 videos have `j < i` or one pattern only | `B2 − max(B0,B1,B3) ≥ +0.020`; `B2 − B4 ≥ +0.015`; `B2 − B5 ≥ +0.015` — **load-bearing for B4 genuineness** |
| **F5** single segment sufficient | label 1 iff `pA` occurs anywhere | `B2 − max(B0,B1,B3) < +0.020`; verdict `NO-GO-B`; no HALT |
| **F6** B5 fallback | force `Dpred` constant `1` for all but 2 outer-train videos in one fold | `b5_fallback_count > 0`; every fallback row carries `b5_fallback: true`; run completes |
| **F7** degenerate assets | 3% all-zero segment videos, 2 zero-only in the whole-video cache (union coverage), 2 videos `duration = null`, 5 positives with empty span list | `zero_vector_videos` equals the **union** count; `o1_fallback` count matches the span-less set; missing-duration videos excluded from `eligible_videos` yet still predicted; zero videos retained in every arm; run completes |
| **F7b** decode-failure HALT | 5% zero videos (union) | HALT on the >1% rule |
| **F8** threshold rule | hand-built vectors with (a) two equidistant tied thresholds, (b) two tied at different distances from 0.5 | (a) smaller θ selected; (b) the one closer to 0.5 selected; rule is `>=` |
| **F9** fold and leakage integrity | any fixture data | one outer-query fold per video; `outer_train ∩ outer_query = ∅` at video/segment/derived-id level; all 30 segments share a fold; inner nested in outer-train |
| **F10** capacity match | pure arithmetic | `|params(B3)−params(B2)|/params(B2) ≤ 0.05` at `d_fix` **and** `d=1792`; `H3(d=1792) = 193`; head-capacity check `|25,090−25,024|/25,024 ≤ 0.01` |
| **F11** Gate-C weighting | synthetic FN population of 400, tercile sizes `(150,130,120)`, known mechanism flags, forced subsample to 120 | weighted coverage equals the analytic value to `1e-9`; tercile-stratified bootstrap CI covers it; controls excluded from the FN denominator; deficit redistribution reproduces the registered allocation on an undersized tercile |
| **F12** B4 genuineness | F4 data | swap fraction ∈ `[0.40,0.60]`; `sign(δ_B4) = −sign(δ_B2)` and `iA > iB` on every swapped video; `in_B4 ≠ in_B2` for swapped; `in_B4` bitwise `== in_B2` for unswapped; **the realized swap set is identical between the training pass and the evaluation pass** |
| **F13** determinism | rerun F1 end to end in two separate processes | byte-identical `oof_predictions.jsonl`; identical `metrics.json` numbers |
| **F14** test-seal guard | attempt to open a registered forbidden path | guard raises; `test_contact_count` increments; run HALTs |
| **F15** sealed-id restriction | synthetic corpus-spanning gold file (JSON and `.pt` variants) whose ids are `authorized ∪ sealed`, plus a synthetic `p11_split` with a non-empty test list | `load_corpus_spanning` returns an object containing **zero** sealed ids; `sealed_ids_dropped > 0` and equals the planted sealed count; `len(restricted) == len(authorized ∩ file_ids)`; a zero drop count raises; holding an unrestricted handle raises `HALT_UNRESTRICTED_GOLD_HANDLE`; a second `unlock_confirmation()` call raises; `authorized_id_hash` changes exactly once, at the unlock |

Any failed assertion blocks the freeze. Results are summarized in `fixtures_report.json` with the
fixture code's SHA256; the freeze payload embeds both.

**Battery result (v4 re-freeze run, 2026-08-07).** `fixture_run_id = fix-20260807T083546Z`,
report `artifacts/tera_gate0/_fixtures/fix-20260807T083546Z/fixtures_report.json`
(sha256 `b9161be50cd33227eb1c158e378f32ff0f3624e5f903ad1267a83fca137021e0`),
**16 cases requested, 16 PASS, 0 FAIL** (F1–F15 plus F7b), wall clock 965.2 s,
`fixture_bootstrap_n = 1000`, `seed_base = 424242`, log
`logging/runs/tera_gate0_fixtures_v2/run.log`. F11 carries three additional assertions
covering §6.7 msc-subset membership, added with the D-3 fix.

*(v3 release run, superseded, retained for the record: `fixture_run_id = fix-20260806T231531Z`,
report sha256 `f21b465e69ac11dc620dfdf9bc66e676cd6749bde65344f1e33762ed979a1fb5`, 16/16 PASS,
wall clock 1343.2 s, log `logging/runs/tera_gate0_fixtures/run.log`. That run was launched
before the `.draft.json` → `.json` rename, i.e. against the byte-identical config at the
harness's unpatched argparse default path; the v4 re-run reproduced that condition with a
byte-identical temporary copy at the draft path, removed immediately afterwards.)*

**Fixture / harness code hash.** `fixtures.py` alone:
`d967f78e87fe31e4275ca163834bc304f6314a36f4e031b0de90825f0d282f7c`. The registered *stable
aggregate* over the whole harness package is

```python
per_file  = {p.name: sha256(open(p,'rb').read()) for p in sorted(Path('scripts/tera_gate0').glob('*.py'))}
aggregate = sha256(canonical_json(per_file).encode('utf-8'))     # canonical_json as in §10.2
```

i.e. `sha256-canonical-json-v1` applied to the `{filename: file-sha256}` map of the sorted
`*.py` files of `scripts/tera_gate0/` (14 files; `__pycache__` and non-`.py` files excluded).
Observed aggregate at v4:
`cb619464b0223ed551f6078d31a67a4a9f832bb42f59d540136fe8d7dd7463aa`
(v3, superseded by D-3:
`7e20884b6272bc98a94a367dc2823ac06c772c16d54a5f1bd415993c11f8e9f2`). The per-file map is
byte-identical to the `package_sha256` block the battery itself wrote into
`fixtures_report.json`, so the report and the frozen config describe the same code.

---

## 10. `frozen_config.json` — mechanism and layout

### 10.1 The frozen file

`research-wiki/tera_gate0_frozen_config.json` is the structured form of everything above.
**v3 (2026-08-07): the file is frozen** — every `TO-FILL-AT-ASSET-AUDIT` and `TO-FILL-AT-FREEZE`
field is resolved, the `.draft.json` suffix is dropped, `status` is `FROZEN`, and `payload_sha256`
carries the canonical hash (§10.2) of the completed payload. Any later payload change produces a
new hash and a new `run_id`; prereg §12 decides whether it is a documentary correction or a
material deviation.

**v4 (2026-08-07): re-frozen under registered deviation D-3.** The payload's
`study.appendix_sha256`/`appendix_version`, `fixtures.package_sha256`,
`fixtures.package_aggregate_sha256`, `fixtures.script_sha256`, `fixtures.report_*`,
`fixtures.run_mode`, `fixtures.fixture_run_id`, `fixtures.battery_result` and `change_log` were
updated; `status` stays `FROZEN` and `payload_sha256` was recomputed. The `run_id` prefix changes
accordingly, so Run 2 executes in a new namespace and cannot collide with Run 1's
`…-7ba80eaf` directory. Full old→new hash chain:
`refine-logs/TERA_GATE0_REFREEZE_2026-08-07.md`.

**Registered launch requirement.** `run_gate0.py`'s `--config` *argparse default* still names the
pre-freeze path `research-wiki/tera_gate0_frozen_config.draft.json`. That default is **not**
corrected in code, deliberately: `scripts/tera_gate0/*.py` is hash-frozen as of the fixture-battery
release run (§9.1), and editing one byte would invalidate both the battery's `package_sha256`
evidence and the `fixtures.package_aggregate_sha256` recorded in the payload. Every registered
execution therefore passes the path **explicitly**:

```bash
python -m tera_gate0.run_gate0 --config research-wiki/tera_gate0_frozen_config.json ...
```

Launching without an explicit `--config` points at a path that no longer exists and fails
immediately; it can never silently run an unfrozen config. `manifest.json` records the full
`command_line`, so the config actually used is auditable, and `load_frozen_config` re-verifies
`payload_sha256` at run start (`HALT_CONFIG_HASH_MISMATCH` otherwise).

### 10.2 Canonical payload hash

Matching `scripts/analysis/edcm_a0.py:47-58`:

```python
def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)

payload_sha256 = hashlib.sha256(canonical_json(cfg["payload"]).encode("utf-8")).hexdigest()
```

**The hash covers `cfg["payload"]` and nothing else.** Every sibling key of `payload` — currently
`schema_version`, `status`, `hash_algorithm`, `payload_sha256`, and any key beginning with `_` — is
outside the hashed object by construction, so the file can carry its own digest. *(review N-9:
the JSON note and this section now state the same operative rule.)*

- `hash_algorithm = "sha256-canonical-json-v1"`.
- Verified at run start; mismatch → `HALT_CONFIG_HASH_MISMATCH`.
- The payload embeds `appendix_sha256` (SHA256 of this file, byte-exact) so prose and config cannot
  drift apart.
- Any post-freeze payload change produces a new hash and a new `run_id`; prereg §12 governs whether
  it is a documentary correction or a material deviation.

### 10.3 Run namespace

`artifacts/tera_gate0/<run_id>/` exactly as listed in prereg §11.1, with
`run_id = "tera-gate0-" + UTC `YYYYMMDDTHHMMSSZ` + "-" + first 8 hex of payload_sha256`.
Non-overwriting: `os.makedirs(..., exist_ok=False)` and every writer opens with mode `"x"`; an
existing path is a HALT. Fixture artifacts live outside the namespace at
`artifacts/tera_gate0/_fixtures/<fixture_run_id>/`.

### 10.4 Provenance, guards, and `manifest.json`

Required fields (prereg §11 plus this appendix's additions), all mandatory, no defaults:

```text
git_commit, git_dirty, git_diff_sha256, host, platform, cpu_model, nvidia_smi_device,
gpu_used, python_version, conda_env, package_versions{torch,numpy,sklearn,transformers},
torch_num_threads, deterministic_flags, command_line, scheduler ("none"), job_id (null),
log_path, pid_file, start_utc, end_utc, wall_clock_s,
seeds{...},                                   # §7.6 register, verbatim
inputs[{path,sha256,bytes}], outputs[{path,sha256,bytes}],
split_source, split_id_hash, split_manifest_sha256,
hateclipseg_surviving_id_hash,
authorized_id_hash{development, confirmation}, sealed_ids_dropped{hatemm, hateclipseg},
confirmation_unlock_utc, confirmation_passes{hatemm_val, hateclipseg_val},
encoder_id, encoder_revision, encoder_config_sha256,
duration_rule, boundary_rule, frame_time_convention, sampling_alignment_proof,
zero_vector_videos, missing_duration_videos, failure_rate,
test_contact_count (must be 0), opened_test_paths (must be []), forbidden_paths[],
overlap_assertions{outer_disjoint, segment_disjoint, inner_nested, one_query_fold_per_video},
msc_subset_sha256, eligible_videos_sha256,
frozen_config_sha256, appendix_sha256, fixtures_report_sha256, fixture_code_sha256, verdict_sha256
```

**Test-seal guard.** A process-wide file-open wrapper checks each resolved absolute path against
`forbidden_paths` (`data/gt/HateMM/test.jsonl`, `data/CLIP_Embedding/HateMM/test_seen_*`). A match
increments `test_contact_count`, appends to `opened_test_paths`, and raises. HateClipSeg's
`test_*`-named files are whitelisted **by exact path** (never by glob) because its whole 395-video
corpus was extracted under `--splits test` and its sealed split lives inside `p11_split.json`, not
in the filename; the whitelist is inside the hashed payload so it cannot be widened silently, and
the sealed ids in those files are removed at load time by §2.8. There is **no** permitted contact
with `data/gt/HateMM/test.jsonl`: §2.8's whitelist semantics make an id-level check of that file
redundant, so the run opens it zero times.

### 10.5 Prediction row schema

```json
{"video_id": "...", "dataset": "HateMM", "outer_fold": 0, "gold_label": 0,
 "arm": "A3", "score": 0.0, "prediction": 0, "threshold": 0.0,
 "threshold_source": "inner_oof:fold0:A3:cfg_lr1e-3_wd1e-4:epoch37",
 "config_id": "cfg_lr1e-3_wd1e-4", "epoch": 37, "seed": 0,
 "selected_segment_ids": [], "selected_second_intervals": [[0.0, 0.0]],
 "d_score_source": null, "d_config_id": null, "d_epoch": null,
 "b5_donor_id": null, "b5_fallback": null, "b4_swapped": null, "o1_fallback": null,
 "oracle_or_eval_only": false, "label_leaking": false,
 "gold_overlap_windows": null, "gold_span_ratio": null}
```

`gold_overlap_windows` / `gold_span_ratio` are populated only by evaluation or oracle routines; any
file containing a non-null gold-derived field carries a top-level `{"oracle_or_eval_only": true}`
sidecar marker (prereg §11.1). Confirmation rows additionally carry
`"split": "val"` and `"confirmation": true`.

`folds/fold_<k>/segment_scores.jsonl`: `{video_id, arm, outer_fold, scores[30],
attention_weights[30]|null, second_boundaries[30][2]}`.
`folds/fold_<k>/selected_evidence.jsonl`: `{video_id, arm, i_top, pair[2], presented_slots[2],
phi[6], b4_swapped, b5_donor_id, b5_fallback}`.

---

## 11. Gate-C implementation details

### 11.1 OOF baseline predictions

Gate-C's prediction source is the **A0 whole-video baseline** run through the §7 protocol on
HateMM-**train** only, with the same 5 outer folds, the same inner-OOF threshold rule, and the same
seeds. `oof_predictions.jsonl` for `arm == "A0"` is the Gate-C input. Val and test are untouched
(prereg §4.1). The A0 OOF run must complete and be hashed before any sampling.

### 11.2 Populations

`FN = {y=1, ŷ=0}`, `TP = {y=1, ŷ=1}`, `FP = {y=0, ŷ=1}`.

### 11.3 Stratified sampling and weights

```python
rngC = numpy.random.default_rng(20260807)

q33, q67 = numpy.quantile(scores_FN, [1/3, 2/3], method="linear")   # WITHIN the FN population
tercile(v) = 0 if s < q33 else (1 if s < q67 else 2)                # half-open, deterministic

if len(FN) <= 120:
    audit_FN = sorted(FN)                      # audit all; weights all 1.0
else:
    target = {0: 40, 1: 40, 2: 40}
    for t in (0, 1, 2):                        # deterministic deficit redistribution
        if N_t < target[t]:
            deficit = target[t] - N_t; target[t] = N_t
            for u in (0, 1, 2):
                if u == t: continue
                take = min(deficit, N_u - target[u]); target[u] += take; deficit -= take
    audit_FN = concat over t of rngC.choice(sorted(FN_t), size=target[t], replace=False)

w[v] = N_{tercile(v)} / n_{tercile(v)}         # population weight, FROZEN at sampling time
```

Controls: 30 TP and 30 FP, 10 per score tercile computed within each control population, same
redistribution rule, drawn from the same `rngC` **after** the FN draw so consumption order is
fixed. Controls never enter the FN coverage denominator (prereg §4.1).

Weighted coverage of a mechanism set `M` (primary **or** secondary presence, prereg §4.3):

```text
coverage(M) = ( Σ_{v ∈ audit_FN, mech(v) ∩ M ≠ ∅} w[v] ) / ( Σ_{v ∈ audit_FN} w[v] )
```

Bootstrap: 10,000 resamples, seed `20260809`; within each tercile draw `n_t` audited items with
replacement; reapply the **frozen** `w[v]` (not recomputed from the resample); percentile 95% CI.
Unweighted audit-sample proportions are reported separately, labelled `diagnostic`.

### 11.4 Annotation form (frozen field schema)

`annotation_protocol.json` contains the instruction text, the taxonomy, the blank form schema, and
the double-coding assignment, and is hashed **before any label is entered** (prereg §4.2).

| field | type | notes |
|---|---|---|
| `video_id` | str | |
| `coder_id` | str | |
| `primary_cause` | enum | exactly one prereg §4.2 value: `short_localized`, `multi_segment_complementary`, `cross_modal`, `quotation_or_counterstance`, `external_knowledge`, `global_evidence`, `annotation_ambiguity_or_noise`, `representation_failure_other` |
| `secondary_causes` | list[enum] | zero or more; must not contain `primary_cause` |
| `minimal_sufficient_intervals` | list[[start_s, end_s]] | may be empty |
| `required_modalities` | list[enum(`visual`,`speech`,`on_screen_text`,`audio_nonspeech`,`transcript`)] | ≥1 |
| `single_interval_sufficient` | bool | |
| `span_video_duration_ratio` | float | computed as `union(minimal intervals)/D_v`, not typed |
| `confidence` | enum(`high`,`medium`,`low`) | |
| `notes` | str | free text, used by no criterion |
| `protocol_sha256` | str | must equal the hashed protocol |
| `form_version` | str | |

Blinding (prereg §4.1): the interface shows video, transcript (only because it is an ordinary model
input), and the official span overlay; it must **not** show the model score, correctness category,
retrieval output, or TERA output. The item list handed to annotators is shuffled with `rngC` after
the draws and carries no category field; the category mapping is kept in a separate file not given
to annotators.

Double coding: `ceil(0.20 × |audit_FN ∪ controls|)` items chosen by a seeded permutation
(`20260807`) of the sorted audited-id list, assigned to a second coder. Both coders' raw
pre-adjudication labels are retained in `gate_c_audit.jsonl`; adjudication appends a third row with
`adjudicated: true`. Reported: raw agreement and Cohen's kappa on `primary_cause`
(`sklearn.metrics.cohen_kappa_score`) over pre-adjudication double-coded pairs.

### 11.5 Hashing order (strict)

```text
1. write annotation_protocol.json  -> sha256 -> record in manifest.json
2. write the blinded item list     -> sha256 -> record
3. ONLY THEN collect labels into gate_c_audit.jsonl; every row carries protocol_sha256
4. adjudicate; append adjudication rows (append-only, never rewrite)
5. freeze msc_subset.json (§6.7) -> sha256 -> record
6. compute coverage, kappa, bootstrap; write metrics.json; write verdict.json
```

Any edit to the protocol after step 3 invalidates the audit (prereg §12).

### 11.6 Media dependency (resolved)

Gate-C requires annotators to watch raw HateMM video. The corpus is now present at
`/home/jehc223/data/HateMM/video` (§2.5), so the dependency is satisfied on this host. The
remaining dependency is **human annotator time**, including the ≥20% double-coding, which is a
scheduling matter, not a technical blocker. Transcript-only or MLLM-described auditing remains a
§12 material deviation and is not an option.

---

## 12. Open points — status after review

The independent review adjudicated OP-1 … OP-7 on 2026-08-07, **before any candidate metric was
computed**. This section is the ex-ante record.

| OP | topic | status |
|---|---|---|
| **OP-1** | B4 lesion semantics | **RATIFIED** — per-video Bernoulli(0.5) swap retained; always-swap rejected as an invertible relabelling that would force a near-certain false `NO-GO-B`. Swap-fraction bound widened to `[0.40,0.60]` (N-5). Recorded in §6.5. |
| **OP-2** | text stream in the feature family | **RATIFIED** — `concat(l2n(img), l2n(text))` at both levels, disclosure sentence retained verbatim, plus the new A0/A1 frame-budget disclosure (N-2). §2.3. |
| **OP-3** | Gate-A capacity asymmetry | **CONFIRMED as a recorded limitation** — not repaired (would be a new arm). Criterion 5 bounds the confound; a `D = A3` pass must carry the stated sentence in `verdict.json`. §4.4. |
| **OP-4** | previously-missing caches | **RESOLVED to pending extraction** — raw corpora restored 2026-08-07; both caches are extractable locally under pinned parity constants (§2.4). `HALT_MISSING_ASSET` remains the rule if extraction fails or its assertions fail. |
| **OP-5** | raw video for Gate-C | **RESOLVED** — raw HateMM/HateClipSeg video present (§2.5, §11.6). Remaining dependency is annotator time. |
| **OP-6** | grid width, `E_max`, patience | **RATIFIED**, and the review's free recommendation **accepted**: `lr` grid extended to `{1e-2, 3e-3, 1e-3, 3e-4}` (A-stage 1465 head trainings, B-stage 990). §3.5, §7.3, §7.5. |
| **OP-7** | `r=128`, `H=64`, `H_att=128` → `H3=193` | **RATIFIED** — arithmetic independently reproduced by the review; head capacities are matched to 0.26%, so the 5% match is not an artefact of the shared projection. §6.5. |

**No open point remains that requires a user or reviewer decision before freezing.** Two matters
remain user-facing but are scheduling, not science: (i) running the §2.4 extraction, and (ii)
securing annotator time for Gate-C. Neither changes a registered rule.

### Recorded limitations to carry into any writeup

1. Gate-A has no capacity control (OP-3); a `D = A3` pass is not capacity-controlled.
2. A0 and A1 do not share a frame budget (8 vs 120 frames); `max(A0,A1)` absorbs it (§2.3).
3. The outer refit takes more gradient steps than the inner fits at the same `epoch*` (§7.5).
4. `POOLED-INNER-OOF-MACRO-F1` is selection-optimistic; it ranks arms and is never reported as
   performance (§5.3).
5. The Gate-B msc label-1 subset may be small (~18 videos); the rate is reported with exact
   counts and a Wilson interval and must not be over-read (§6.7).
6. A negative Gate-A result falsifies only the registered granularity, representation, weak/no-span
   learners, data, thresholds — and the registered `lr ∈ {1e-2 … 3e-4}` grid span, which is now
   wide enough that the under-fitting objection does not apply.

---

## 13. Registered implementation readings (pre-freeze)

While building the harness (`scripts/tera_gate0/`) against §9's fixture specification the
implementer had to read a number of under-specified fixture details one particular way. Those
readings are registered here, **before execution**, so that the fixture evidence and the frozen
document describe the same artifact.

**Classification, applying to every item below.** These are **fixture-construction-layer
readings** (plus one file-path correction and one block of pre-execution observations). **None of
them adds or removes an arm, and none changes an endpoint, a threshold, a split, a fold seed, or a
decision rule.** The registered scientific content of §§3–8, §10.5, §11 and the prereg is
untouched. The one item that introduces new code surface — R-5's `d_segment_scores_file` hook — is
**available only under `--fixture-mode`**; outside `--fixture-mode` the runner refuses it, and its
use is recorded in `metrics.json`.

| id | reading | layer |
|---|---|---|
| **R-1** | fixture script path: the appendix/payload wrote `scripts/analysis/tera_gate0_fixtures.py`; the actual file is `scripts/tera_gate0/fixtures.py` — the payload field is updated. | path correction (§9.1, config `fixtures.script_path`) |
| **R-2** | the fixture spike amplitude is defined **relative to the L2-normalized window vector's scale**; read literally as raw `N(0,1)` units, F1 could never pass. | fixture construction (§9.2 F1) |
| **R-3** | F1's text half is a **fixed vector shared by all videos** (per-video text noise would drown A1's visual block). | fixture construction (§9.2 F1) |
| **R-4** | F2 is constructed as **within-video same-window** signal with `amp = 0.35`; at the registered `0.20` with independent windows, F2's own `A1 ≥ A0 − 0.02` assertion is unreachable. The property being tested (`O1 ≡ A1`) is unchanged. | fixture construction (§9.2 F2) |
| **R-5** | F4's negative-class mixture (20 % inverted / 80 % single-pattern) is an item the appendix left unspecified; the D-stage segment scores used for pair selection are injected through a **fixture-only hook** (`d_segment_scores_file`), refused outside `--fixture-mode` and recorded in `metrics.json`. | fixture construction + fixture-only hook (§9.2 F4) |
| **R-6** | F6's empty-stratum construction: inside the chosen fold, force `Dpred = 1` for **every** video except the 2 designated query videos, which are forced to `Dpred = 0`. | fixture construction (§9.2 F6) |
| **R-7** | F7 keeps the registered categories and counts but uses `V = 800`, so the union is `0.875 % ≤ 1 %`; the v2 text's "3 %" contradicted F7b, which tests the `> 1 %` HALT. | fixture construction (§9.2 F7) |
| **R-8** | F11's `(150, 130, 120)` tercile split is unreachable under §11.3's quantile rule, so the assertion is split into three sub-assertions. | fixture construction (§9.2 F11) |
| **R-9** | F5 needs `--fixture-mode`'s `forced_stage_b` to reach `NO-GO-B`. | fixture harness (§9.2 F5) |
| **R-10** | O1/O2 fixture scores are taken as `σ(pooled logit)`, so they live on the `[0,1]` scale the shared threshold rule expects. | scale convention (§5.1/§5.2 as exercised by fixtures) |
| **R-11** | B5's `rng5` is **re-instantiated per outer fold** and consumed in ascending video-id order; the Gate-B *train-side* pair construction uses D's **outer-OOF** segment scores. | implementation reading of §6.6 / §6.1 |
| **R-12** | three protocol-fragility points observed **on synthetic data**: (a) an A2-refit sign-inversion basin at roughly 20 % of draws, (b) `θ*` transfer collapse when the epoch loop saturates early, (c) A1's `1/√K` norm disadvantage. Registered as **pre-execution observations only — they change no rule.** | observation (no rule change) |

R-11's two clauses are readings of text already registered in §6.6 (a per-outer-fold donor draw
consumed in sorted id order) and §6.1 (train-side D scores must never be in-sample); they pin the
implementation to the reading that satisfies both, and add nothing.

R-12 is deliberately **not** repaired. Each point is a property of the registered protocol under
synthetic data, and repairing any of them would change a registered rule (the refit procedure, the
epoch-selection rule, or the representation) after the design was fixed. They are recorded so that
a later negative result is read against the protocol's known fragilities rather than
re-interpreted after the fact.

**Source list, as submitted (verbatim, Chinese original).**

```text
(1) fixture 脚本路径:附录/payload 写 scripts/analysis/tera_gate0_fixtures.py,实际是 scripts/tera_gate0/fixtures.py —— payload 字段需更新;
(2) fixture 尖峰幅度定义为相对 L2 归一化后窗口向量的尺度(字面 raw N(0,1) 下 F1 永不可过);
(3) F1 text 半部为全体视频共享的固定向量(逐视频噪声会淹没 A1 的视觉块);
(4) F2 构造改为视频内同窗 + amp 0.35(注册的 0.20/独立窗下 F2 自身的 A1≥A0−0.02 不可达;O1≡A1 性质不变);
(5) F4 负类混合(20% 反转/80% 单模式)为附录未指定项;pair 选择的 D 段分通过 fixture-only hook 注入(d_segment_scores_file,非 --fixture-mode 拒绝,记录于 metrics.json);
(6) F6 的空 stratum 构造:fold 内除 2 个 query 视频强制 Dpred=0 外全部强制 1;
(7) F7 保留类别与计数但 V=800 使 union 0.875%≤1%(原文 3% 与 F7b 测试的 >1% HALT 自相矛盾);
(8) F11 的 (150,130,120) tercile 在 §11.3 分位规则下不可达,拆为三个子断言;
(9) F5 需 --fixture-mode 的 forced_stage_b 才能到 NO-GO-B;
(10) O1/O2 分数取 σ(pooled logit) 以共享阈值规则的 [0,1] 尺度;
(11) B5 的 rng5 每 outer fold 重实例化,按视频 id 升序消费;Gate-B 确认的 train 侧 pair 构造用 D 的 outer-OOF 段分;
(12) 另有三条合成数据上观察到的协议脆弱点(A2 refit 反转盆地 ~20%、epoch 早饱和时 θ* 迁移塌陷、A1 的 1/√K 范数劣势)——作为 pre-execution 观察登记,不改任何规则。
```

---

## 14. Change log

- **v4 (2026-08-07) — RE-FROZEN under registered deviation D-3**
  (`refine-logs/TERA_GATE0_DEVIATION_D3_2026-08-07.md`; re-freeze record
  `refine-logs/TERA_GATE0_REFREEZE_2026-08-07.md`). A harness defect was found after Gate-C
  annotation was complete and **before** Run 2, i.e. before any affected metric existed: the
  msc-subset construction at `run_gate0.py:817-819` admitted a row only if it was an adjudication
  row or its video carried exactly one row, so every double-coded video whose two coders **agreed**
  (two rows, no adjudication row) was dropped from the subset. That contradicts the §6.7 registered
  definition ("audited videos of any category … carrying `multi_segment_complementary` as primary
  or secondary"), which admits every audited video under the adjudicated-else-first resolution the
  coverage path already uses. On the submitted audit this narrowed the candidate pool from 133
  audited videos to 111. Fixed minimally: the adjudicated-else-first resolution is lifted into
  `gate_c.resolve_audit_rows()`, shared by the coverage/kappa loop (verified bit-identical) and by
  `gate_c.msc_subset()`, which now receives raw audit rows. Fixture F11 gained three assertions
  pinning §6.7 membership, including the dropped case; they fail on the v3 bytes and pass on v4.
  **No definition, threshold, seed, taxonomy entry, decision rule or HALT condition in this
  document changed**; no Gate-C quantity is affected (the coverage inputs are unchanged); the fix
  reaches only `msc_subset.json` and the two Gate-B criteria that read it. Battery re-run
  `fix-20260807T083546Z` 16/16 PASS. Package digests, this file's digest and `payload_sha256`
  re-embedded.
- **v3 (2026-08-07) — FROZEN.** Asset audit executed read-only (`Dv = 1024`, `Dt = 768`,
  `d = 1792`; six cache SHA256s; HateMM `train` 744 / `val` 107 with exact id+label agreement
  against the caches, `split_source = gt_jsonl`; HateClipSeg `p11_split` 237/39/119 and the
  395-id surviving hash; `zero_vector_videos` = 1 on HateMM-train, 0 elsewhere in the restricted
  views; `missing_duration_videos` = 0 everywhere; `test_contact_count = 0`). §9 fixture battery
  16/16 PASS (`fix-20260806T231531Z`). The two §2.4 extractions completed. Added §13, the
  registered implementation readings. `tera_gate0_frozen_config.draft.json` renamed to
  `tera_gate0_frozen_config.json`, `status = FROZEN`, all placeholders resolved, `appendix_sha256`
  re-embedded and `payload_sha256` recomputed.
- **v2 (2026-08-07)** — post-review. Applied BLOCKING-FIX B-1 (§7.10 confirmation protocol),
  B-2 (§5.3 cross-fold aggregation + §6.1 config source), B-3 (§7.2/§7.4 unified epoch rule),
  B-4 (§2.8 sealed-id reader, manifest fields, fixture F15), B-5 (§6.7 msc-subset membership and
  0/0 conventions); factual corrections F-1 (nested `ids` contract), F-2 (`num_frames`
  provenance-only for the whole-video cache), F-3 (source line ranges), F-4 (observed gold-span
  field set); accepted NOTEs N-1 … N-9, N-11, N-12, N-13; accepted the OP-6 recommendation
  (`lr` grid extended by `1e-2`); recorded the OP-1 … OP-7 adjudications; updated the environment
  section for the restored raw corpora and the new detached-run discipline. Ready to freeze.
- **v1-draft (2026-08-07)** — first draft; reviewed, `APPROVE-WITH-FIXES`.
