# C04-A0T-SMALL-v1 impl-v8 — Independent Code/Resource Review (round 1)

Date: 2026-08-01
Reviewer: fresh independent Opus reviewer, zero exposure to the implementation
reasoning; inputs were the frozen bytes plus
`refine-logs/C04_A0T_SMALL_V1_V8_CODE_RESOURCE_REVIEW_REQUEST.md`.
Stage authorized: `CPU_PREFLIGHT` only.

---

# VERDICT: `GO (0C/0H/0I)`

Authorizes the CPU preflight only. All 15 SHA-256s match the review request byte-for-byte, and the live config's `implementation_hashes` is identical to that same 15-row set; all 15 `frozen_design_hashes` also verify on disk.

No finding at any severity. I hunted the stated failure family specifically and could not construct an instance in the v8 tree.

---

## What I RECOMPUTED

| Quantity | Config claim | My recomputation | |
|---|---|---|---|
| `sacct -X -n -P -j 13852` | FAILED / 1:0 / 1978 / `billing=8,cpu=8,gres/gpu=1,mem=64G,node=1` | `13852\|c04_a0t_small_v1_v7\|FAILED\|1:0\|1978\|billing=8,cpu=8,gres/gpu=1,mem=64G,node=1` | OK |
| v7 prompt-record count | 400 | 400 files, 400 `elapsed_seconds` | OK |
| `hatemm_forward_seconds_sum` | 1769.9 | 1769.9382669688 -> 1769.9 | OK |
| `hatemm_forward_seconds_mean` | 4.4248 | 4.42484566742 -> 4.4248 | OK |
| median / max | 4.457 / 10.15 | 4.456705 / 10.149983 | OK |
| `non_forward_overhead_seconds` | 208.1 (1978 - 1769.9) | 208.0617 -> 208.1 | OK |
| Campaign ledger payload digest | `b19629de...` | recomputed identical | OK |
| Hash chain | GENESIS -> `e8db5f88...` = head | prev-link OK, row digest OK, head OK | OK |
| Aggregate | 1978 | sum rows = 1978 = field | OK |
| Effective cap / headroom | 7200 | min(7200, 28800)=7200; 1978+5222=7200 <= cap; 1978+5223=7201 > cap | OK |
| OOM closed form 43056^2*16*4 B | 110.50 GiB | 118,644,424,704 B = 110.49623 GiB -> 110.50 | OK |
| `project_gpu_window` | window 4022, 800 fwd, 3807.9 s, 5.3 % | 4022 / 800 / **3807.9** / margin **214.1** / **0.0532** | OK |
| secondary capped projection | — | 2588.1 s (36 % margin) | OK |
| max affordable mean/forward | — | 4.6924 s (basis 4.4248, capped-regime measurement 2.9) | OK |
| Max frame-pack load seconds that still fits | — | 274.1 s (builder's basis: 60 s) | — |

Also recomputed: `guard_item_margin` 300, `guard_seal_reserve` 600, `watchdog_reserve` 300 all present in config **and** asserted in all three of preflight/gpu_ledger/producer; `TERMINAL_SECONDS_HARD_MAX` = 5822 (5222+600), coherent — strictly tighter than the schema's 7800 ceiling, so no unpublishable state; wrapper `timeout` ceiling = 4922.

Independently re-derived facts the request did not ask for but that the argument rests on: v7's `elapsed_seconds` timer starts *after* `load_or_create_frame_pack` (`v7_producer.py:1759-1768`), so the 201 packs v7 built really are inside the 208.1 s residual — the over-count claim is true, worth ~111 s.

MHC_zh frame-pack footprint sampled on 8 real selected videos (incl. one 3840x2160): mean 8.48 MB -> ~1.70 GB; HateMM measured from v7 = 0.32 GB. Total ~2.0 GB against ~10 GB of soft-quota headroom (280 G used / 290 G soft / 3000 G hard). Fits.

---

## Negative fixtures I EXECUTED

All under `.../scratchpad/reviewer_r1`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=""`, no GPU, no SLURM, no v8 namespace, no write outside scratch (verified after: `artifacts/c04/a0t_small_v1_impl_v8` still absent, v6/v7 untouched, campaign ledger untouched, all 15 frozen hashes unchanged).

| # | Fixture | Result |
|---|---|---|
| 1 | `self_test_fixtures()` | **65 fixtures, 0 failed** — matches the claim exactly |
| 2 | `run_self_tests(cfg)` on the live config | **74 checks, all pass** (65 + 4 role-map + 4 AST + `no_test_paths`) |
| 3 | Preflight vs producer visual-geometry paths, real processor, 7 geometries (1920x1080, 1080x1920, 640x360, 224x224, **3840x2160**, 333x247, 720x1280) | **grids identical on all 7**; `max_pixels=151200` propagates; max 2880 pre-merge tokens, all < 4096 |
| 4 | Config-contract neutrality of the authority flip (`preflight_materialization_authorized`->true, verdict->GO, pin filled, prompt hashes frozen) | contract hash **unmoved** |
| 5 | Contract tamper probes: `projection_basis.mean`, `teacher_contract.max_pixels`, `resources.small_cap_gpu_seconds` | contract hash **moves** in all three — the basis and the cap are inside the pinned contract, not decoration |
| 6 | Projection gate at 4096 tokens | `geometry_fits=False`, `fits=False` — HALT |
| 7 | Projection gate at 600 s frame-pack load | `projected=4347.9`, `time_fits=False` — HALT |
| 8 | `assert_campaign_aggregate_headroom` vs **live** ledger | 5222 accepted, 5223 refused — non-vacuous |
| 9 | `write_frame_pack` -> `strict_validate_frame_pack` round trip, real PNG bytes | manifests identical; key set = `{schema_version} u FRAME_PACK_BINDING_KEYS u FRAME_PACK_METADATA_KEYS u {frames, payload_sha256}` |
| 10 | Reader vs wrong `teacher_max_pixels` / wrong `code_resource_authorization_sha256` / 1 flipped PNG byte / stray file in pack dir / re-write of an existing pack | **all 5 rejected**, distinct HALT messages; clean pack re-validates after cleanup |
| 11 | Zero-frame path (`total=0` -> `requested_indices=[]`) written, validated, and pushed through the **prompt-record schema** as the real producer would emit it | **accepted** (schema `minItems: 0`) — the fixture's `[0]*8` does not cover this, so I covered it; not empty |
| 12 | `visual_patch_tokens` = 4096 / 4097 vs schema | 4096 accepted, 4097 rejected; `assert_visual_token_ceiling` agrees; preflight gate is strictly tighter (`<`) |
| 13 | `resource_final_state` built as the code builds it, validated in **4 regimes**: sealed, no-seal (`NO_SEAL_PUBLISHED`), at `TERMINAL_SECONDS_HARD_MAX`, recovery publication | **all publishable**; required-set == builder key-set exactly |
| 14 | Same with a stale `cap_gpu_seconds: 7200` | **rejected** by the `const: 5222` — CHANGE-1 reached the schema |
| 15 | AST decode-guard on a mutated producer (`import decord`, `x.save()`, `from ...common import write_frame_pack`) | 3 of 4 sub-checks go **red** |
| 16 | Same with `.asnumpy()` injected | `producer_calls_no_decoder_attribute` goes **red** — all 4 sub-checks proven non-vacuous |
| 17 | Template-equality clause deleted from `assert_teacher_visible_containment` | `teacher_visible_benign_template_tamper_rejected` goes **red**; the plain tamper stays red via the ban scan. The carried-forward round-5 observation is real, not cosmetic |
| 18 | `NO_SEAL_SENTINEL` | no assignment in `gpu_ledger.py`; `G.NO_SEAL_SENTINEL is C.NO_SEAL_SENTINEL` -> **True** |
| 19 | **Selection reproduction** from the live ASR | 200+200; reproduces the **v6 and v7 frozen allowlists exactly on both datasets**; `label_value_materialized = 0`, `label_field_syntactically_skipped = 744 + 579` |
| 20 | **800 real containment renderings** on the real selected transcripts | **800/800 pass, 0 halts**, 402 banned tokens (400 ids + 2 HateMM label-bearing prefixes) |
| 21 | **v8 decode+PNG-encode vs v7's frozen packs**, 10 random HateMM items | **10/10 byte-identical**, incl. backend / total / requested_indices / decode-failure vector |
| 22 | **Two-pass vs one-pass pyav on the real fallback item** `MHC_zh/BV18N4y1B7qA` | decord genuinely fails (`DECORDError`); 2879 frames @1920x842; **48.5 s -> 7.0 s**; **PNG bytes identical**, `n` identical |
| 23 | Path containment | every staged path + frame-pack manifest path is inside the no-clobber namespace; the campaign ledger is outside it |

---

## Answers to (a)-(d)

**(a) `max_pixels = 151200` — legitimate at code/resource level; no design re-review needed to authorize a CPU preflight.**

I verified the authority claim rather than accepting it: all seven cited entrypoints do default to `360*420` at the exact cited lines (`generate_c02_density_view_text_embedding_HF.py:157`, `predict_target_qwen.py:299`, `generate_vision_summary.py:69`, `p10_score_segments.py:147`, `p10c_score_segments.py:137`, `score_segments_mllm.py:87`, `role3/arbitrate_qwen.py:363`), and `generate_VideoMLLM_embedding_bidir_textpool_HF.py:29` does call 151200 "the deployed max_pixels". Two of the config's paths are under `scripts/analysis/` rather than `src/utils/`, but the line numbers are all correct.

The decisive point is that the frozen design is *silent* on visual resolution — it fixes the frame count, the index rule, the black-frame rule, the transcript rule and the decoding parameters, none of which move. v7's "no `max_pixels`" was therefore not a reviewed choice either; it was an unreviewed default that made this teacher's visual input unlike every other Qwen2.5-VL call in the project. Choosing the deployed cap moves *toward* the reviewed protocol, not away. And it is not optional: at 43,056 pre-merge tokens the run is physically impossible on an A100.

It is also fully auditable: declared as a teacher-input change in the config, inside `config_contract_sha256`, inside all 400 frame-pack bindings, and `const: 151200` in the prompt-record schema's `provenance.teacher_max_pixels`, so every sealed record states what the teacher saw. Uniformity is preserved by refusing to salvage any v7 record. This is a no-performance-claim survival screen, and the amendment's non-waived gate binds the full bank, not this tranche.

Forward note, not a finding: if this tranche passes and a full-bank tranche is requested, the cap is by then part of the frozen teacher protocol and should be named explicitly in the proposal text at that review.

The fail-closed ceiling is correct on all three counts you asked about — `assert_visual_token_ceiling` runs at `producer.py:1536`, `model.generate` at 1545; the count is `visual_patch_tokens(prepared["video_grid_thw"][0])`, i.e. **pre-merge**, the quantity vision SDPA is quadratic in (the merged count would have been wrong by 4x); and it runs before `prepared.to(model.device)`, so nothing oversized ever reaches the card.

**(b) 5.3 % is acceptable — but the more important resource fact is that this is the last first-tranche allocation.**

Four independent reasons the margin is larger than 5.3 %: (i) the per-forward basis is v7's **native-resolution** mean, and the config's own least-squares fit (`1.9913 + 3.567e-4*tokens`) prices a capped v8 forward at ~3.0 s against 4.4248 budgeted, i.e. ~1140 s of unbudgeted slack; (ii) `non_forward_overhead_seconds` double-counts ~111 s of v7 frame-pack work that v8 provably does not do (I confirmed against v7's own timer placement); (iii) the corroborating capped-regime projection is 2588 s, a 36 % margin, and the 46 forwards it rests on are a *direct* measurement in the capped regime, correctly reported as corroboration only; (iv) crucially, **3807.9 is a prediction, not the gate** — the gate runs on measured inputs at preflight time, so a worse-than-predicted measurement HALTs before the namespace exists at zero GPU cost. The breach path is clean: exit 40, guard stops before an item, no output truncated, every completed checkpoint intact, accounting-only breach record, and the wrapper jq-verifies that record.

What the authorizer should know regardless: `1978 + 5222 = 7200` **exactly**. This reservation consumes the FIRST_TRANCHE phase ceiling to the second. If it breaches, there is no headroom for a v9 under the current phase — only `PASS_C04_SMALL_V2` plus a fresh result-to-claim GO plus a new code/resource review could raise it. That is a consequence of the amendment, not a defect, and the code enforces it correctly (5223 is refused against the live ledger).

The one arithmetic gap I found is immaterial and I state it for completeness: the gate compares against 4022 measured from job start, whereas the true bound is `4022 - claim_elapsed` (the guard deadline sits `300 + claim_elapsed` before the wrapper kill, because `BudgetGuard.at_job_start` receives the claim-time *remainder* and subtracts elapsed-since-entry again). `claim_elapsed` is a few seconds of small-file hashing — 412 staged files, of which 400 are ~1 KB manifests — against 111 s of double-counted slack. The double-subtraction errs *safe* on the guard side.

MHC_zh priced at the HateMM mean is the one unmeasured input, and declaring it is enough: the geometry it depends on is measured per item at preflight, both datasets land at the same capped grid (I measured 1080x1920 -> 2880 tokens, identical to HateMM's capped median), MHC-ZH transcripts are far shorter than HateMM's, and the basis it borrows is a native-resolution upper bound. The failure mode is a clean halt, not a corrupted bank.

**(c) Nothing is unbound. I walked the chain end to end and executed it.**

The chain: each pack's 8 PNG digests are pinned in its own `manifest.json`, and `strict_validate_frame_pack` re-hashes all 8 every time (I proved it catches a single flipped byte and a stray file); all 400 manifests enter `staged_output_hashes`; `verify_preflight_manifest` **re-hashes every staged entry at every stage** (claim, producer, reconciliation); `verify_payload_review` asserts `staged_output_hashes` and `preflight_manifest_sha256` equality; `verify_gpu_execution_authorization` pins `payload_review_sha256`. The three dropped fields are dropped because they *cannot exist* at preflight time, and the allocation binding is not lost — it lives in `provenance.allocation_claim_sha256`, which is in the prompt-record schema's `required` list, so every record still ties allocation -> pack -> frames.

Key-set drift is structurally impossible: `write_frame_pack` and `strict_validate_frame_pack` both derive their key set from the single `FRAME_PACK_BINDING_KEYS` / `FRAME_PACK_METADATA_KEYS` constants, and the only producer of a binding dict is `frame_pack_binding`, which `require_exact_keys`-checks it. Round-tripped for real. (`write_frame_pack` will accept a hand-built binding carrying an extra key, but the reader then rejects the pack and no such call site exists — not exploitable.)

The two-pass pyav is byte-identical, independently reproduced on the real fallback item; the second pass cannot select a different frame because it enumerates the same decode order and raises if any requested index is missing (which degrades to the frozen black-frame rule, now visible in the preflight manifest *before* the payload review instead of inside the GPU job — an improvement).

A failed gate leaves nothing: the namespace is created only at `os.rename(temp_namespace, namespace)` at `preflight.py:951`, everything before it lives in a `tempfile.mkdtemp` sibling that the `except` clause `rmtree`s, and `namespace.parent.mkdir` only touches the already-existing `artifacts/c04`. The projection HALT at line 903 raises inside that `try`.

No remaining path lets the producer decode a video: the AST guard parses the producer's frozen bytes for decoder imports, decoder attributes, PIL `save`, and the five frame-writing symbols from the common module (closing the transitive route) — and I proved all four sub-checks go red under mutation.

**(d) Strict equality is the right trade; warn-and-continue would be wrong.**

The equality cannot false-positive within a fixed environment: I measured that in transformers 4.49 `Qwen2_5_VLProcessor.__call__` routes videos to `self.image_processor(images=None, videos=videos, ...)` — literally the call the preflight makes — and the two paths produced identical grids on all 7 geometries I tested. The frames are byte-identical by hash, so the computation is deterministic end to end.

Its real value is therefore not memory safety (that is the separate `assert_visual_token_ceiling`, which runs regardless) but **desync detection**: it fires only if the processor tree, the pack bytes, or the transformers version drifted between the preflight and the GPU job. In exactly that case the teacher's visual input is no longer what the payload review approved, and continuing would silently seal a bank whose teacher input differs from the reviewed one. Losing the tranche is the cheaper failure. A warn-and-continue would convert an auditable halt into an unauditable bank.

---

## Non-blocking observations (no action required to proceed)

1. `preflight.py:848` — the freeze-stage `access_ledger.json` snapshots the audit *before* `materialize_frame_packs` runs, so its `guarded_runtime_evidence` holds 402 events while the same file claims `frame_packs_frozen_by_this_program: 400`. The preflight *manifest*'s `guarded_access_audit` (line 998) is complete (802 events) and is the artifact the payload review pins, so the trail exists; nothing compares the two. Cosmetic.
2. `resource_final_state.schema.json:111` — `terminal_sacct_gpu_seconds.maximum` is still 7800 (the v7-era 7200+600) while the code's `TERMINAL_SECONDS_HARD_MAX` is 5822. The code is strictly tighter, so the intersection is non-empty and every reachable state is publishable (verified at 5822). A stale bound, not a defect.
3. `assert_teacher_visible_precondition` runs in the producer, after the allocation and the single-use ticket are consumed, and has no preflight counterpart — structurally the shape of the hunted family. It cannot fire here: its inputs are the transcripts the preflight freezes and hash-pins, and I executed the check on exactly those frozen inputs (800/800 pass, item 20 above). Carried over from v7 unchanged. Worth moving into the preflight in any future version, purely as belt-and-braces.
4. Disk: the preflight will add ~2.0 GB of native-resolution frame packs and ~3,600 inodes. Usage is 280 G against a 290 G **soft** quota (3000 G hard), so it fits with ~8 G to spare, but the soft quota is close enough to be worth knowing before a job that writes for ~20-30 minutes.

---

## Builder's disposition of the four non-blocking observations

All four are **accepted as recorded, not repaired**, because repairing any of them
would move the reviewed bytes and void this `GO` — the same disposition v7 used
for its round-5 observations, one of which (`NO_SEAL_SENTINEL`) v8 then closed.

1. Accepted. The complete 802-event audit is in the preflight manifest, which is
   the artifact the payload review pins. Cosmetic.
2. Accepted. Verified independently: the code bound (5822) is strictly tighter
   than the schema bound (7800), so every reachable terminal state is publishable
   and the round-4 High shape (an empty code-schema intersection) does not recur.
3. Accepted and **carried forward as a named instruction for any v9**: move
   `assert_teacher_visible_precondition` into the CPU preflight. It cannot fire
   in v8 (its inputs are the preflight-frozen, hash-pinned transcripts, and it was
   executed on exactly those, 800/800), but it is the right structural fix.
4. Verified independently by the builder before submission: `quota -s` reports
   280 G used against a 290 G soft quota and a 3000 G hard limit, and the v7 tree's
   201 frame packs occupy 328 MB, consistent with the reviewer's ~2.0 GB estimate
   for 400 packs. Proceeding.
