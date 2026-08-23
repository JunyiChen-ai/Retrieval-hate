# TERA Gate-0 — freeze record (2026-08-07)

Closing record for the Gate-0 pre-registration package: implementation appendix **v3 (FROZEN)** +
`tera_gate0_frozen_config.json`. Nothing below is a candidate result; no metric of any arm exists
yet.

---

## 1. Frozen artifacts and their digests

| artifact | path | sha256 |
|---|---|---|
| implementation appendix v3 (FROZEN) | `research-wiki/EXP_tera_gate0_impl_appendix.md` | `ea158b2c23bd0a9ed8cecdbaccdecd21e97621f9a88b3db8a7c2dcbba2c42ffc` |
| frozen config (whole file) | `research-wiki/tera_gate0_frozen_config.json` | `fdebff8bd72b704f0a5da8e007145bdb06a1f365c6ed2ab4e38507bf92541bdc` |
| **canonical payload hash** | `cfg["payload"]`, `sha256-canonical-json-v1` | **`7ba80eaf697ac46bb90b30161b1726aba7ee238e73001dd832ce30dba8a1dabe`** |
| pre-registration (upstream, unchanged) | `research-wiki/EXP_tera_gate0_prereg.md` | `f6c1ce6c652bcedd18451d4ee3a490ca2c72c603489e89c6a161855537ed6e98` |
| independent review record (unchanged) | `research-wiki/EXP_tera_gate0_impl_appendix_review.md` | `9147ad4c1adacf1160566f0503937d45e0f5205b43549359d3eefe68263637c5` |

- Run namespace prefix that follows from the payload hash:
  `artifacts/tera_gate0/tera-gate0-<UTC YYYYMMDDTHHMMSSZ>-7ba80eaf/`.
- `payload_sha256` is written in **two places**: `cfg["payload_sha256"]` inside the frozen config
  (re-verified at run start by `load_frozen_config`, mismatch → `HALT_CONFIG_HASH_MISMATCH`) and in
  this record.
- The payload embeds `appendix_sha256`, so prose and config cannot drift apart. Verified after
  writing: payload hash recomputes from disk, and the embedded appendix digest equals the appendix
  file byte-for-byte.
- `research-wiki/tera_gate0_frozen_config.draft.json` no longer exists (plain rename; the file was
  never git-tracked). Nothing is committed — the main conversation decides that.

**Registered launch requirement (appendix §10.1).** `run_gate0.py`'s `--config` argparse *default*
still names the pre-freeze `.draft.json` path. It is deliberately **not** patched: the harness
package is hash-frozen as of the fixture-battery release run, and editing one byte would invalidate
both the battery's `package_sha256` evidence and `fixtures.package_aggregate_sha256` inside the
frozen payload. Every registered execution passes the path explicitly:

```bash
python -m tera_gate0.run_gate0 --config research-wiki/tera_gate0_frozen_config.json ...
```

A default launch points at a path that no longer exists and fails immediately, so an unfrozen
config can never run silently.

---

## 2. Fixture battery (pre-execution correctness evidence)

- Report: `artifacts/tera_gate0/_fixtures/fix-20260806T231531Z/fixtures_report.json`
  (sha256 `f21b465e69ac11dc620dfdf9bc66e676cd6749bde65344f1e33762ed979a1fb5`).
- **16 requested / 16 PASS / 0 FAIL** — F1, F2, F3, F4, F5, F6, F7, F7b, F8, F9, F10, F11, F12,
  F13, F14, F15. Wall clock 1343.2 s, `fixture_bootstrap_n = 1000`, `seed_base = 424242`.
- Log: `logging/runs/tera_gate0_fixtures/run.log`.
- `fixtures.py` sha256 `1cd6c48226345c91c7423c7c61e805f94c8d05c36492af814a1bc266491dfd36`.
- **Stable aggregate over the harness package** — algorithm registered in appendix §9.1 and in the
  payload (`fixtures.package_aggregate_algorithm`):

  ```python
  per_file  = {p.name: sha256(bytes(p)) for p in sorted(Path("scripts/tera_gate0").glob("*.py"))}
  aggregate = sha256(json.dumps(per_file, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8"))
  ```

  i.e. `sha256-canonical-json-v1` over the `{filename: file-sha256}` map of the 14 sorted `*.py`
  files (`__pycache__` and non-`.py` excluded). Observed aggregate
  **`7e20884b6272bc98a94a367dc2823ac06c772c16d54a5f1bd415993c11f8e9f2`**. The per-file map stored in
  the frozen payload is byte-identical to the `package_sha256` block the battery itself wrote, so
  the pre-freeze evidence and the frozen config describe the same code.

---

## 3. Asset audit (read-only) — full table

Performed 2026-08-07 in a single detached process with the §10.4 `SealGuard` installed and every
corpus-spanning artifact read only through the §2.8 `load_corpus_spanning` reader. **Discipline:**
id lists, labels, tensor shapes and raw bytes only; **no metric of any kind was computed**;
`data/gt/HateMM/test.jsonl` was opened **zero** times (`test_contact_count = 0`,
`opened_test_paths = []`).

### 3.1 Dimensions

`Dv_observed = 1024`, `Dt_observed = 768`, `d = 1792` — identical across all six caches, matching
the registered expectations.

### 3.2 Feature caches

| key | path | bytes | sha256 |
|---|---|---|---|
| `hatemm_train_segments` | `data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt` | 91,800,918 | `8b4a706cec51d106151e57109b24850232239168d5e0ca363341ee76493d7fb7` |
| `hatemm_train_wholevideo` | `data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt` | 5,359,881 | `0802b6ba00669ec546e63f36dca1772cb2d7806b969de307235af3450a8176c1` |
| `hatemm_val_segments` **(new)** | `data/CLIP_Embedding/HateMM/dev_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt` | 13,204,954 | `a2ae105e61478b86193267fe67263d1c26436f0881620222f0aa1544fa380778` |
| `hatemm_val_wholevideo` | `data/CLIP_Embedding/HateMM/dev_seen_openai_clip-vit-large-patch14-336_HF.pt` | 772,382 | `ab9cd8a070b93afbf994ed876e3adfd9c2a139e82d801af21346c29f17c1888d` |
| `hateclipseg_segments` | `data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt` | 48,739,122 | `df6e1c0434ba4b0fb210c3470b3407e05e041f718834d70ad3bc20bcde34d89e` |
| `hateclipseg_wholevideo` **(new)** | `data/CLIP_Embedding/HateClipSeg/test_seen_openai_clip-vit-large-patch14-336_HF.pt` | 2,846,592 | `43227d527d402e1707f770386667cb39114c861f01345c0ab3b9087abedf6f30` |

Both **new** digests match the values handed over with the extraction task. Every segment cache
reports `num_subclips = 30`, `num_frames = 120`; no whole-video cache stores a `num_frames` key, so
its `8` stays `provenance_only` (review F-2).

*Bookkeeping note:* the task brief said "5 caches"; the registered payload has **six** cache
entries with a `sha256` placeholder. All six are filled. No cache is missing and none is extra.

### 3.3 Gold / split artifacts

| artifact | bytes | sha256 | read mode |
|---|---|---|---|
| `data/gt/HateMM/hate_spans.json` | 108,875 | `f8f2be10856a40c0ef5763b9211ecbed506743792ccddfb3adc92bed460c1846` | hash-only, inside reader scope |
| `data/gt/HateClipSeg/gold_segments.json` | 425,726 | `a1dad37e686a5106a1392e8151f0946858bc6086cdfd05efc9457b1d7c634a36` | hash-only, inside reader scope |
| `data/gt/HateClipSeg/video_durations.jsonl` | 18,806 | `d8bd334b90f270a703c8419717c979fddd13da537486bb5699d12400f1c1e292` | hash-only, inside reader scope |
| `data/gt/HateClipSeg/p11_split.json` | — | `a279431137feeaf72241e1ca4a7ef76d1e86c8381d08aac47123b4b287db98b1` | ids only (not corpus-restricted) |
| `data/gt/HateMM/train.jsonl` | — | `73295d4b96d9937dca7787fc59a13561ac15020c1608e196c5685f2a055d7741` | ids + labels |
| `data/gt/HateMM/val.jsonl` | — | `33a3768976a68db4fe3da39cacafa6beac11b04f83e5740fef0e4f91b391e2b3` | ids + labels |
| `data/gt/HateMM/test.jsonl` | — | **not opened** | forbidden path, zero contact |

### 3.4 Split resolution (§2.9) — branch 1, both partitions

`split_source = "gt_jsonl"`. `set(jsonl_ids) == set(cache["ids"][0])` and per-id label equality held
for **both** partitions; `HALT_SPLIT_MISMATCH` did not fire.
`split_id_hash = sha256(utf8("\n".join(sorted(ids))))` (`common.py:sha256_ids`).

| partition | V | pos / neg | `split_id_hash` |
|---|---|---|---|
| HateMM train | **744** ✓ | 298 / 446 | `54e1e9beb97c3e76fcd5c8f664d9b948dcb368e202c6e686f346e0f8a5e1273c` |
| HateMM val | **107** ✓ | 43 / 64 | `9cee85f3db92e816c8e867743d7a87ad6d4043eb7d4ab732f95ce8f11d9fb7b3` |

`val.jsonl` was read for exactly this integrity assertion. Nothing was fitted, scored or selected;
the run's own one-time confirmation load (§7.10.1, after `unlock_confirmation()`) is unaffected.

### 3.5 HateClipSeg

- `p11_split.json` counts re-verified: train 237 / val 39 / test 119 (plus a `meta` key of 8
  entries). Surviving corpus 395.
- `hateclipseg_surviving_id_hash` (395 ids) =
  `37d852b7f72cc87465bbcc293bb345e46617ac61a105be7a70ad3ff24640ba19`.
- Binding-endpoint class counts recomputed from the **id-restricted** `gold_segments.json`:
  development (`p11 train`) 237 videos = 109 pos / 128 neg; confirmation (`train ∪ val`) 276
  videos = 127 pos / 149 neg ⇒ P11 val contributes 18 pos / 21 neg, **both classes ≥ 10**, so the
  §2.2 underpower rule is not triggered by these counts.
- Authorized-id hashes: development `HateMM` `54e1e9be…`, `HateClipSeg`
  `06949b8bcf6a6376009fe387d2aa79dd22bbee0541510f52938fb321c32f25fa`; confirmation `HateMM`
  `4a38fb4a534c512cb071de34198b42f70ed86f1f7ce2dd76cc505e2d47857c0b`, `HateClipSeg`
  `28a9b48cd0acaec28a66f7348a10ea77f4fb3eb4f667ad457c1ad4d37219a90c`. (Computed in the audit
  process to record the constants; the run builds its own `Authorization` and performs its own
  single `unlock_confirmation()`.)

### 3.6 Sealed-id restriction evidence

`sealed_ids_dropped` (value recorded per path is the last phase in which the artifact was read,
i.e. confirmation): `hate_spans.json` **232**; `gold_segments.json`, `video_durations.jsonl` and
both `test_seen_*` caches **119** each.

- 1083 HateMM records − 851 authorized (`train ∪ val`) = 232 → independently reproduces the
  review's `hate_spans.json` record count of **1083** without ever holding an unrestricted handle.
- 395 HateClipSeg ids − 276 authorized = 119 → reproduces `gold_segments.json` = **395** and
  `p11_split["test"]` = **119**.

### 3.7 Failure accounting (§2.7)

| partition | V | `zero_vector_videos` | `missing_duration_videos` | union | rate | HALT? |
|---|---|---|---|---|---|---|
| HateMM train (binding) | 744 | 1 (`hate_video_95`) | 0 | 1 | **0.001344** | no (≤ 0.01) |
| HateMM val | 107 | 0 | 0 | 0 | 0 | no |
| HateClipSeg ∩ `p11 train` | 237 | 0 | 0 | 0 | 0 | no |
| HateClipSeg ∩ `p11 train ∪ val` | 276 | 0 | 0 | 0 | 0 | no |

The HateClipSeg corpus does contain exactly one zero-vector video, `yt_NzvfkIYS5Yg` (undecodable
container; both extractors agree, so the union adds nothing). It is a `p11_split["test"]` id, so it
is discarded by the whitelist restriction before any load returns — hence the restricted counts of
0. That corpus-level fact is recorded from the extraction log as **provenance**, not from a
restricted read.

### 3.8 Observed gold-span schema (review F-4)

On the 744 restricted HateMM-train records: `duration` 744, `spans` 744, `label` 744, `clipped` 2,
`anomaly` 1, `parse_error` **0** — exactly the registered field set.

### 3.9 Audit outcome

**No inconsistency found. No HALT condition met.** Every registered assertion held: id/label
equality both partitions, dimension equality across caches, `num_subclips`/`num_frames` equality,
restriction assertions 1–4 on every corpus-spanning artifact, failure rate under the 1 % rule,
`test_contact_count = 0`.

---

## 4. Extraction record (referenced, not re-run)

`logging/runs/tera_cache_extract/`:

| cache | log | realized |
|---|---|---|
| HateMM val K=30 | `hatemm_val_subclipK30.log` (+ `.pid`) | `V = 107`, `TotalSub = 3210`, `Dv = 1024`, zero-vector videos **0** |
| HateClipSeg whole-video | `hateclipseg_wholevideo.log` (+ `.pid`) | `N = 395`, `Dv = 1024`, `Dt = 768`, zero-vector videos **1** (`yt_NzvfkIYS5Yg`) |

Both ran under the pinned parity constants of appendix §2.4 (`--num_subclips 30 --num_frames 120`
and `--num_frames 8` respectively, `openai/clip-vit-large-patch14-336`), detached per §0.3.
`HALT_MISSING_ASSET` does not fire.

---

## 5. Environment fingerprint

| item | value |
|---|---|
| host | single-GPU workstation, **NVIDIA GeForce RTX 5090** (32,607 MiB), driver 595.71.05 |
| scheduler | none (`sbatch`/`squeue` absent) — CLAUDE.md single-GPU exemption, prereg §13.4 |
| conda env | `HateVideo` (`/home/jehc223/miniconda3/envs/HateVideo`) |
| python | 3.11.8 |
| torch | **2.7.1+cu128** (CUDA 12.8) |
| numpy / scikit-learn / transformers | 1.26.4 / 1.5.2 / 4.49.0 |
| platform | `Linux-6.17.0-35-generic-x86_64-with-glibc2.39` |
| device for A/B/O heads | `cpu`, `torch.set_num_threads(8)` |
| device for feature extraction | `cuda:0` |
| git commit at freeze | `16ebf90647f02917b10065931f98bc7195be08c4` (working tree carries the
  untracked Gate-0 files; **nothing committed** — the main conversation decides) |

---

## 6. Difference list vs appendix v2

### 6.1 Registered implementation readings (new appendix §13)

Twelve items registered before execution, all classified as **fixture-construction-layer readings**
(plus one path correction and one block of pre-execution observations). **None adds or removes an
arm, and none changes an endpoint, threshold, split, fold seed or decision rule.** Full text —
including the verbatim Chinese source list — is appendix §13; the same items are in the payload at
`payload.registered_implementation_readings`.

| id | summary |
|---|---|
| R-1 | fixture script path corrected: `scripts/analysis/tera_gate0_fixtures.py` → `scripts/tera_gate0/fixtures.py`; payload field updated |
| R-2 | spike amplitude is relative to the L2-normalized window vector's scale (literal raw `N(0,1)` makes F1 unpassable) |
| R-3 | F1's text half is one fixed vector shared by all videos |
| R-4 | F2 built as within-video same-window with `amp 0.35`; `O1 ≡ A1` unchanged |
| R-5 | F4 negative mixture 20 % inverted / 80 % single-pattern (unspecified in v2); D segment scores injected via the **fixture-only** `d_segment_scores_file` hook |
| R-6 | F6 empty stratum: `Dpred = 1` for all but the 2 designated query videos, which are forced to 0 |
| R-7 | F7 uses `V = 800` so the union is 0.875 % ≤ 1 % (v2's "3 %" contradicted F7b) |
| R-8 | F11's `(150,130,120)` terciles unreachable under §11.3's quantile rule → three sub-assertions |
| R-9 | F5 needs `--fixture-mode`'s `forced_stage_b` to reach `NO-GO-B` |
| R-10 | O1/O2 scores taken as `σ(pooled logit)` for the `[0,1]` shared-threshold scale |
| R-11 | B5's `rng5` re-instantiated per outer fold, consumed in ascending video-id order; Gate-B train-side pair construction uses D's outer-OOF segment scores |
| R-12 | three pre-execution observations on synthetic data (A2 refit sign-inversion basin ~20 %, `θ*` transfer collapse on early epoch saturation, A1's `1/√K` norm disadvantage) — **no rule changed** |

The R-5 hook is refused outside `--fixture-mode` (`HALT_D_SCORE_OVERRIDE`, verified in
`run_gate0.py`) and its use is recorded in `metrics.json` as `d_score_override`. R-9's
`forced_stage_b` is likewise gated on `--fixture-mode`. R-11 was verified against the code: `rng5`
is constructed inside the per-outer-fold builder, while B4's global swap draw is taken once.

### 6.2 Placeholder resolution and status changes

- All six `caches[*].sha256`, `Dv_observed`, `Dt_observed`, `split_source`, `split_source_sha256`,
  `split_id_hash`, `gold_spans_sha256`, `surviving_id_hash`, `split_sha256`, `zero_vector_videos`,
  `missing_duration_videos` resolved; `prereg_sha256`, `fixtures.script_sha256`,
  `fixtures.report_sha256` (the `TO-FILL-AT-FREEZE` set) resolved.
- New payload keys carrying audit evidence: `features.d_observed`,
  `features.caches[*].{bytes_observed, present_at_v3, num_subclips_observed, num_frames_observed}`,
  `features.extraction_commands.{executed_utc, logs, realized}`,
  `data.hatemm.{split_id_hash_algorithm, train_counts_observed, val_split_source,
  val_split_source_sha256, val_split_id_hash, val_counts_observed, id_label_match_assertion,
  gold_spans_observed_field_counts_on_744_restricted_records}`,
  `data.hateclipseg.{surviving_n_observed, split_counts_observed, gold_segments_sha256,
  video_durations_sha256, binding_endpoint_counts_observed}`,
  `data.failure_accounting.{observed_failure_rate_hatemm_train, halt_evaluation}`,
  `fixtures.{package_dir, package_sha256, package_aggregate_algorithm, package_aggregate_sha256,
  fixture_run_id, battery_result, script_path_correction}`,
  `study.{frozen_utc, config_path, launch_requirement}`, plus the new blocks
  `registered_implementation_readings` and `asset_audit`.
- Status flips: top-level `status` `POST_REVIEW_READY_TO_FREEZE` → **`FROZEN`** (+ `frozen_utc`);
  `study.appendix_version` `v2` → **`v3`**; the two `pending_extraction` caches →
  `extracted_2026-08-07`; `_hash_note` tail rewritten to say the digest is final.
- Appendix v2 → v3: status/blinding/freeze-procedure block rewritten; §2.1 dims filled; §2.2 cache
  table gains an sha256 column and the gold artifacts; §2.4 marked completed with realized outputs;
  §2.7 gains the observed-count table; §2.9 gains the split-resolution table, the HateClipSeg
  hashes/counts and the observed span field counts; §9.1 path corrected + battery result and the
  package-hash algorithm added; §10.1 rewritten (frozen file + launch requirement); new §13; change
  log renumbered to §14 with a v3 entry.

### 6.3 Not changed

Arms (A0–A4, O1, O2, B0–B5), endpoints, all thresholds and decision rules, splits and fold seeds,
the seed register, the optimizer/grid/epoch rule, the confirmation protocol, the sealed-id
restriction, the Gate-C protocol, and every HALT condition are **identical to v2**. The review
adjudications (B-1…B-5, F-1…F-4, N-1…N-13, OP-1…OP-7) are untouched.

---

## 7. Standing discipline for execution

1. Zero test-set contact — `data/gt/HateMM/test.jsonl` is never opened; `test_contact_count` must
   end at 0 and `opened_test_paths` at `[]`.
2. Decision rules frozen before any result is seen — this record is the timestamp.
3. Blinding — no candidate metric was computed during design, implementation or this audit.
4. Single-submission execution — one registered run per stage, non-overwriting namespace.
5. Launch with an explicit `--config research-wiki/tera_gate0_frozen_config.json`.
