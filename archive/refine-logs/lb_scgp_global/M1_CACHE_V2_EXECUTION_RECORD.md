# M1 CACHE v2 Execution Record — dual-cache verification + seal single-submit

Date: 2026-07-13

## Role declaration

Author = **Claude Opus 4.8** (`claude-opus-4-8`, 1M context), fresh independent **M1 v2
cache-verification + seal-executor** role for the `lb_scgp_global_r2` M1 block. This role is
**separate** from m1-prep (v2 author/implementer/freezer, `M1_CACHE_V2_FREEZE.md`), the v2 amendment
reviewer, the fresh 0C/0H v2 code reviewer (`M1_CACHE_CODE_REVIEW.md` DELTA-2), and the v2
execution-authorizer (`M1_CACHE_V2_EXECUTION_AUTHORIZATION.md`). I performed only the executor
obligations: verify the two terminal cache jobs' artifacts strictly, confirm the seal submission is
inside the granted authorization envelope, single-submit the seal **exactly once**, poll to a
handoff state, and write this record.

**Model-binding note** (precedent: v4 execution record): `AGENTS.md` binds the main-dialogue
subagent to "GPT-5.5 xhigh"; that backend is unavailable, so this runs on the `CLAUDE.md`-bound
**Opus 4.8**. Process fact, not a defect.

**Write scope:** this file only. No edit to any plan / machine.json / `_HASHES.sha256` / config /
schema / script / wrapper / sbatch / authorization / review / freeze / artifact. The only mutation
was the single authorized `sbatch` of the seal. Verification was read-only
(`sacct`, `squeue`, `ls`, `wc`, `grep`, `jq`, `sha256sum`); no GPU/MLLM/OCR/network/model run and no
validation/test or train-label read.

---

## 0. Verdict

**CACHES_VERIFIED_SEAL_SUBMITTED → SEAL COMPLETED = GO (CACHE_SEALED).** Both M1 v2 cache jobs
COMPLETED cleanly; every strict artifact check PASSES against the frozen contract; the seal
submission is explicitly within `M1_CACHE_V2_EXECUTION_AUTHORIZATION.md` §7 item 3 and all its
preconditions are met. The seal (`LBSCGP-GLOBAL-M1-CACHE-SEAL-v1`, job name `…_seal_v2`) was
single-submitted → **job 13035**, submit `2026-07-13T09:11:02Z` UTC, held `PENDING (JobHeldUser)`
briefly, auto-released (**not** force-released), and reached **`COMPLETED` (ExitCode 0:0,
Elapsed 00:00:15)** within ~1 min. It published `cache_seal_decision.json` with
**`decision="GO"`, `terminal_state="CACHE_SEALED"`**; the seal independently recomputed both dataset
Merkle roots (match), and stamped `verified:true` for BOTH datasets. **M1 is SEALED.** Terminal
evidence in §7. Single-submit budget spent (1/1); no resubmission.

**One non-gating scientific flag (surfaced, does NOT block the seal):** the caches are dominated by
canonical **all-unresolved** fallback records — MHC `parse_rate=0.0874` (192/2196 parse-ok),
MHC_zh `parse_rate=0.0691` (160/2316) — driven by `transport_fallback_records` 2000/2196 (MHC) and
2104/2316 (MHC_zh). Per `M1_CACHE_V2_EXECUTION_AUTHORIZATION.md` §6 and the machine plan
`runs[4]/[5].failure_transition` ("STOP or canonical unresolved fallback; no prompt rescue"), a low
parse rate is a **designed, allowed** outcome with **no parse-rate floor gate** (DELTA-1 D4). It is
flagged here for the science owner because it materially affects how much signal the sealed cache
carries downstream, but it is not an M1 gate failure.

---

## 1. Terminal states — cache jobs 13012 / 13013 (`sacct`)

`sacct -j 13012,13013 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,Start,End`:

| JobID | JobName | State | ExitCode | Elapsed | MaxRSS | Start → End |
|---|---|---|---|---|---|---|
| 13012 | `lbscgp_global_r2_m1_cache_mhc_v2` | **COMPLETED** | 0:0 | 03:30:21 | — | 16:46:19 → 20:16:40 |
| 13012.batch | batch | COMPLETED | 0:0 | 03:30:21 | 8829496K (~8.42 GiB) | — |
| 13013 | `lbscgp_global_r2_m1_cache_mhc_zh_v2` | **COMPLETED** | 0:0 | 04:12:14 | — | 16:46:36 → 20:58:50 |
| 13013.batch | batch | COMPLETED | 0:0 | 04:12:14 | 8700432K (~8.30 GiB) | — |

Both COMPLETED, ExitCode `0:0`. Wrappers use `set -euo pipefail` and end with a schema/QC gate, so
exit 0 is a structural proof the producer + in-job validation passed.

## 2. Log-tail verdicts (clean; no traceback)

- `…_mhc_v2_13012.out` (143 lines) / `…_mhc_zh_v2_13013.out` (39 lines): **only** `[WARN] decord
  failed … trying PyAV` fallback lines — the DESIGNED fail-open decord→PyAV path, not errors. Zero
  non-WARN lines; `grep -vE '^\[WARN\]'` returns empty for both.
- `…_13012.err` / `…_13013.err`: only the checkpoint-shard load progress bar and the two
  known-benign warnings (`use_fast` slow-processor deprecation; `do_sample=False` with
  `temperature=1e-06` under greedy — cosmetic, determinism holds). **No traceback, no exception.**

## 3. Strict artifact verification table

Artifacts under `artifacts/lb_scgp_global/v1/m1/cache/{MHC,MHC_zh}/`. Every number below is read
directly from the primary source this session (jq / wc / sha256sum), not transcribed.

| # | Check | Expected | Observed MHC | Observed MHC_zh | Provenance |
|---|---|---|---|---|---|
| 1 | `cache.jsonl` line count | 4·549=2196 / 4·579=2316 | **2196** | **2316** | `wc -l cache.jsonl` |
| 2 | every row parses as JSON | all | 2196/2196 (`jq -c .`) | 2316/2316 | `jq -c . | wc -l` == line count, exit 0 |
| 3 | rows conform to replica cert schema (6 top keys; 8 tri-state {state,conf} + modality_binding; parse_flags[]; enums; conf∈[0,4]; replica_index∈[0,3]; `evidence_pack_sha256` `^[0-9a-f]{64}$`; `schema_version=="scgp_global_cert_v2"`; no extra keys) | all valid | **2196 valid / 0 invalid / 0 jq-err** | **2316 valid / 0 invalid / 0 jq-err** | jq validator `scratchpad/validate.jq` vs `scgp_global_cache_replica_v2.schema.json` |
| 4 | `network_model_api_call_count` (manifest + ledger) | 0 | **0 / 0** | **0 / 0** | `jq .zero_counters` on manifest & ledger |
| 5 | ALL 29 `zero_counters` == 0 (manifest) | all 0 | **all 0** | **all 0** | `jq .zero_counters` manifest |
| 6 | ALL 29 `zero_counters` == 0 (ledger) | all 0 | **all 0** | **all 0** | `jq .zero_counters` ledger |
| 7 | `ocr_call_count` | 0 | **0** | **0** | manifest/ledger zero_counters |
| 8 | manifest + ledger present | both | present | present | `find artifacts/.../m1` |
| 9 | `record_count`=`call_count`=`merkle_leaves`=4·`unique_pack_count` | consistent | 2196=2196=2196=4·549 | 2316=2316=2316=4·579 | `jq` manifest scalars |
| 10 | `unique_pack_count`==`video_count` (full dedup) | equal | 549==549 | 579==579 | `jq` manifest |
| 11 | distinct `video_id` | 549 / 579 | **549** | **579** | `jq -r .video_id | sort -u | wc -l` |
| 12 | `replica_index` distinct set | {0,1,2,3} | **0 1 2 3** | **0 1 2 3** | `jq -r .replica_index | sort -u` |
| 13 | (video_id, replica) unique pairs == rows | 2196 / 2316 | **2196** | **2316** | `jq '"\(.video_id)|\(.replica_index)"' | sort -u | wc -l` |
| 14 | every video has exactly 4 replicas | 0 violations | **0** | **0** | pair-count `uniq -c | awk '$1!=4'` empty |
| 15 | cache id-set ⊆⊇ frozen train id-set | both diffs 0 | in-cache∖train=0, train∖cache=0 | 0 / 0 | `comm` vs `jq -r .id data/gt/<ds>/train.jsonl` |
| 16 | label-family keys anywhere in rows (label/split/seed/neighbor/prediction/margin/gold/y_true/is_hate/target) | none | **none** | **none** | `grep -oE '"(…)"\s*:'` empty |
| 17 | `gold_isolation.train_labels_opened` | false | **false** | **false** | `jq .gold_isolation` manifest |
| 18 | ledger read-kinds ⊆ {title_source, asr_source, train_video_read} | only these | 1+1+1098 | 1+1+1158 | `jq .access_ledger[].kind | uniq -c` |
| 19 | `authorized_train_evidence_read_count` = 2 + 2·N | 1100 / 1160 | **1100** (=2+2·549) | **1160** (=2+2·579) | manifest + ledger + `access_ledger|length` |
| 20 | video reads: all out-of-repo symlink, none in-repo | is_symlink=true, in_repo=false | 1098 symlink, 0 in-repo | 1158 symlink, 0 in-repo | `jq` ledger audit fields |

Manifest scalars (provenance for the flag in §0): MHC `parse_ok_records=192`, `parse_rate=0.08743`,
`unresolved_records=2004`, `transport_fallback_records=2000`, `retry_used=3440`/`retry_cap=4392`,
`total_invocations=5636`, `missing_video_count=0`, `terminal_state="CACHE_PRODUCED_PENDING_SEAL"`,
`model_id="Qwen/Qwen2.5-VL-7B-Instruct"`, `num_frames=16`. MHC_zh `parse_ok_records=160`,
`parse_rate=0.06908`, `unresolved_records=2156`, `transport_fallback_records=2104`,
`retry_used=3625`/`retry_cap=4632`, `total_invocations=5941`, `missing_video_count=0`. (`jq` manifest)

### 3.1 Frozen hash bindings re-verified (seal-chain + producer-recorded)

- Producer-recorded manifest `hashes` match `M1_CACHE_V2_FREEZE.md` §1: `common_sha256`
  `56fbb403…` (#1), `evidence_pack_builder_sha256` `54a0a97e…` (#2), `replica_schema_sha256`
  `4bfcfea2…` (#13); config `4e42013d…` (MHC, #5) / `ce9bcab2…` (MHC_zh, #6).
- Seal-chain on-disk `sha256sum` this session (pre-submit), all == freeze §1: common_v2
  `56fbb403…`, seal_v2.py `62d1d100…` (#4), seal config `6d6b4faf…` (#7), seal wrapper `17f47f60…`
  (#9), seal sbatch `e8f8e706…` (#12), replica schema `4bfcfea2…` (#13), seal schema `f4605bb7…`
  (#14), cert_v2 schema `4d3f1663…` (seal-config `run1_frozen` binding).
- `EXPERIMENT_PLAN.machine.json` = **`ab0a06fb…`** == freeze cascade target == seal-config
  `hash_bindings.authoritative_inputs`.

*(Cryptographic Merkle-root recompute and `payload_sha256` canonicalization are the seal tool's own
hard gates — `lb_scgp_global_r2_m1_cache_seal_v2.py:88-111`; my checks confirm every input the seal
needs is structurally present and internally consistent, so the seal is warranted.)*

## 4. Authorization citation for the seal submission

`M1_CACHE_V2_EXECUTION_AUTHORIZATION.md` **§7 item 3** (binding scope): *"`runs[6]` CACHE-SEAL-v1
(`sbatch scripts/slurm/lb_scgp_global_r2_m1_cache_seal_v2.sbatch`) — exactly one submission,
permitted only after both runs[4] and runs[5] reach `COMPLETED` and their artifacts are in place
(`cache.jsonl` line count `== 4·U_D`, `cache_manifest.json` + `access_ledger.json` present,
`zero_counters` all 0). CPU-only."* Every precondition is satisfied (§1 both COMPLETED; §3 rows 1,
5, 6, 8). This is **not** an authorization gap — the seal is inside the already-granted envelope.
Pre-submit ceremony (all PASS): single-submit ledger `sacct --name=…_seal_v2` = **zero prior rows**;
no-clobber `artifacts/lb_scgp_global/v1/m1/cache_seal_decision.json(.publish.lock)` **absent**;
`squeue -u $USER` empty; seal sbatch is CPU-only (`--cpus-per-task=4 --mem=32G`, no `--gres`, no
`--time`, `HF_HUB_OFFLINE=1`).

## 5. Seal submission + monitoring

- Command (executed **exactly once**): `sbatch scripts/slurm/lb_scgp_global_r2_m1_cache_seal_v2.sbatch`
- Returned: **`Submitted batch job 13035`**, `sbatch_exit=0`, submit **`2026-07-13T09:11:02Z`** UTC.
- Run id `LBSCGP-GLOBAL-M1-CACHE-SEAL-v1`; job name `lbscgp_global_r2_m1_cache_seal_v2`.
- First poll (09:11:40Z): `PENDING / JobHeldUser`. Second poll (09:12:40Z): **COMPLETED** — the
  hold auto-released and the CPU-only seal ran in 15 s. **Not** force-released, cancelled, or
  resubmitted (CLAUDE.md "等自动放行即可,不要强行释放"; authorization §7 out-of-scope list).
- **State at handoff: `COMPLETED`, decision `GO`.** Terminal evidence in §7.

## 6. Single-submit ledger — SPENT

- Prior seal_v2 submissions before this session: **0** (`sacct --name=…_seal_v2` zero rows).
- This session: **1** (job 13035). Authorized seal submissions remaining: **0**.
- **No resubmission** regardless of the eventual terminal outcome. On seal STOP or FAILED: halt,
  collect evidence, report `main`, route to a fresh result-to-claim (do not resubmit). On seal GO
  (`cache_seal_decision.json.decision=="GO"`): M1 sealed; hand off to the fresh independent
  post-seal review (separate role) — this executor does not self-certify.

## 7. TERMINAL EVIDENCE — seal COMPLETED, decision = GO (added on the terminal notification)

### 7.1 SLURM terminal (`sacct -j 13035`)

| field | value |
|---|---|
| State | **COMPLETED** |
| ExitCode | **0:0** |
| Elapsed | **00:00:15** |
| Start → End | `2026-07-13T21:11:50` → `2026-07-13T21:12:05` (NZ local; = `09:11:50Z` → `09:12:05Z` UTC) |
| MaxRSS | `3348K` (CPU-only) |
| hold | brief `PENDING (JobHeldUser)` then auto-released (never force-released) |

Seal logs `…_seal_v2_13035.{out,err}` are **0 bytes** — the seal is raise-or-succeed by design; the
wrapper gate `jq -e '.decision=="GO"'` (`…_m1_cache_seal_v2.sh:53`) passed → `COMPLETE=1` → exit 0.
Therefore COMPLETED ⟹ a published **GO** decision (the wrapper's `cleanup_on_exit` trap would have
removed the artifact on any non-GO/abort).

### 7.2 Seal decision (`artifacts/lb_scgp_global/v1/m1/cache_seal_decision.json`)

- `decision="GO"`, `terminal_state="CACHE_SEALED"`, `run_id="LBSCGP-GLOBAL-M1-CACHE-SEAL-v1"`,
  `no_success_claim=true`, `labels_enter_after_this_seal_only=true`, `slurm_job_id="13035"`,
  `payload_sha256=eb1d40d3…e348534`.
- `hashes`: `config_sha256=6d6b4faf…` (freeze #7), `replica_schema_sha256=4bfcfea2…` (#13),
  `seal_schema_sha256=f4605bb7…` (#14) — all match `M1_CACHE_V2_FREEZE.md` §1.
- `gold_isolation.train_labels_opened_by_seal=false`.
- Published exactly the two allowlisted new files (`cache_seal_decision.json` +`.publish.lock`), no
  clobber. Validated against `scgp_global_cache_seal_v1.schema.json`.

### 7.3 Per-dataset independent re-verification (`.per_dataset[].seal_checks`) — both `verified:true`

| check | MHC | MHC_zh |
|---|---|---|
| `merkle_root_recomputed_match` | true | true |
| `recomputed_merkle_root` == manifest root | `ad98d8e8…` ✓ | `563bcefb…` ✓ |
| `call_count_equals_4_unique` (`unique_pack_count`) | true (549) | true (579) |
| `record_count` / `video_count` | 2196 / 549 | 2316 / 579 |
| `all_videos_have_R_replicas` | true | true |
| `zero_counters_all_zero` | true | true |
| `provenance_hashes_present` | true | true |
| `manifest_payload_ok` | true | true |
| **`verified`** | **true** | **true** |

The seal independently recomputed the Merkle roots and manifest payload hashes (the cryptographic
gates my read-only jq pass could not recompute) and both matched — closing the full artifact
verification chain. Sealed `cache_jsonl_sha256`: MHC `c7e2bf8a…`, MHC_zh `df22ce48…`.

### 7.4 Handoff routing

M1 seal = **GO / CACHE_SEALED**. Next per the run order: `LBSCGP-GLOBAL-M2-COMPARATOR-FREEZE-v1`
(comparator freeze), which carries its own ceremony and is where labels/validation first enter — it
is **not** unlocked by this executor. Route the sealed M1 to a fresh independent post-seal review
(separate role); this executor records the terminal fact only and does not self-certify. **Reminder
for M2 and downstream:** the sealed certificate is ~91-93% canonical all-unresolved (§0 flag) — the
science owner should weigh how much signal the certificate carries before resting a claim on it.

## Required statements

- No performance evidence exists or is claimed; the caches are the train-only label-blind MLLM
  certificate; no accuracy / macro-F1, no training, no kNN.
- The only project gold is `parent_video_binary_label`. The M1 chain opened **no** train label and
  **no** validation/test content or label (labels enter only after the seal decision). The mp4 read
  is authorized train evidence; the followed external symlink target is recorded for audit only
  (`followed_target_in_repo=false`); rows carry no label/split/seed/neighbor/prediction/margin field.
- M2 (comparator freeze), validation/test, and training remain **locked**; this record and the seal
  submission unlock nothing downstream of a GO seal.
- Executor = Claude Opus 4.8, fresh, execution role only. Write scope = this file. Only mutation =
  the single authorized `sbatch` of job 13035. Not committed to git (archiver handles commits).
