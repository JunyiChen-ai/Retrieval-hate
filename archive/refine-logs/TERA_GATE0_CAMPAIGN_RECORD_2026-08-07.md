# TERA Gate-0 — CAMPAIGN RECORD (2026-08-07)

Closing record for the TERA Gate-0 campaign, from cold start on a fresh machine to the terminal
verdict. This file is the index: it carries the timeline, the complete pointer/hash chain to every
freeze and deviation record, the final verdict, and the sealing declaration. It does not restate
what those records already say.

- **Study**: TERA-GATE0
- **Pre-registration**: `research-wiki/EXP_tera_gate0_prereg.md`
  (registration text sha256 `f6c1ce6c652bcedd18451d4ee3a490ca2c72c603489e89c6a161855537ed6e98`
  through both runs; results and deviations back-filled at close-out — see §6)
- **Terminal verdict**: **`NO-GO-C`** — Gate-C failed its `multi_segment_complementary >= 0.15`
  criterion at **6/73 = 0.0822**. The registered route is closed.
- **Finding**: `TARGET_FINDINGS.md` → **F122**
- **Cost**: two CPU-only harness runs, **48m 36s** total wall clock, **0 GPU-hours** for the gates
  themselves, **$0 cloud**, plus the Claude-agent annotation pass over 133 items. GPU was used
  only for the cold-start feature extraction.
- **Environment**: single-GPU workstation `sc474399`, NVIDIA GeForce RTX 5090, **no SLURM**
  (`sbatch`/`squeue` absent — CLAUDE.md single-GPU exemption, prereg §13.4); conda `HateVideo`,
  Python 3.11.8, torch 2.7.1+cu128, numpy 1.26.4, sklearn 1.5.2, transformers 4.49.0. Gate heads
  run on **CPU** with `torch.set_num_threads(8)` and deterministic algorithms on.
- **Git**: `git_commit = 16ebf90647f02917b10065931f98bc7195be08c4`, `git_dirty = true` for both
  runs — the Gate-0 files were untracked working-tree files throughout. **Nothing was committed by
  this campaign**; the main conversation decides.

---

## 1. Timeline

All times UTC, taken from artefact `start_utc`/`end_utc` fields and from file mtimes.

| # | UTC | event | evidence |
|---|---|---|---|
| 1 | 2026-08-06 ~20:08 → 20:15 | **Cold start on a fresh machine.** Raw video corpora pulled (HateMM, HateClipSeg, Multihateclip) and symlink layout replayed; CLIP `openai/clip-vit-large-patch14-336` downloaded. | `logging/runs/raw_video_pull/` (`pull.sh`, `replay_symlinks.py`, per-corpus logs), `logging/runs/hf_download_clip/` |
| 2 | 2026-08-06 ~20:36 | **Feature-cache recovery.** The two caches Gate-0 needs and did not have were extracted on GPU under the pinned §2.4 parity constants: HateMM val K=30 subclips (`V=107`, `TotalSub=3210`, `Dv=1024`, 0 zero-vector) and HateClipSeg whole-video (`N=395`, `Dv=1024`, `Dt=768`, 1 zero-vector `yt_NzvfkIYS5Yg`, an undecodable container that the whitelist discards as a test id). | `logging/runs/tera_cache_extract/`, FREEZE §4 |
| 3 | 2026-08-06 21:01 → 23:15 | **Harness smoke, structural scans, and the fixture battery brought to green.** Eight fixture-battery attempts (`fix-20260806T205904Z` … `fix-20260806T224350Z`) preceded the release run. | `logging/runs/tera_gate0_smoke/`, `scan_11`, `scan_21`, `scan_31`, `scan_F4b`, `tera_gate0_fix2`, `tera_gate0_fix25`, `tera_gate0_fix2b`, `tera_gate0_fixtures/` |
| 4 | 2026-08-06 23:15 | **Fixture battery v1 release run: 16 requested / 16 PASS / 0 FAIL** (F1–F15 incl. F7b), 1343.2 s, `fixture_bootstrap_n=1000`, `seed_base=424242`. | `artifacts/tera_gate0/_fixtures/fix-20260806T231531Z/fixtures_report.json` |
| 5 | 2026-08-06 23:44 | **Read-only asset audit.** No inconsistency, no HALT condition: id/label equality on both partitions, dimension and `num_subclips`/`num_frames` equality across caches, all four restriction assertions on every corpus-spanning artefact, `test_contact_count = 0`. Binding HateMM-train partition `V = 744`, failure rate **0.001344** against the 1% HALT bar. | `logging/runs/tera_gate0_asset_audit/run.log`, FREEZE §3 |
| 6 | 2026-08-06 23:55:58 | **FREEZE (appendix v3).** Canonical payload hash `7ba80eaf…`; `run_id` prefix `tera-gate0-<UTC>-7ba80eaf`. | `refine-logs/TERA_GATE0_FREEZE_2026-08-07.md` |
| 7 | 2026-08-07 00:06:00 | **D-1 registered** — stage A necessarily executes twice; written before Run 1 was launched and before any Gate-A/B/C/temporal number existed (`artifacts/tera_gate0/` held only `_fixtures/`). | `refine-logs/TERA_GATE0_DEVIATION_D1_2026-08-07.md` |
| 8 | 2026-08-07 00:06:25 → 00:30:04 | **Run 1 — the prediction-source run.** `--stages A,C --confirmation none`, 1418.2 s, CPU. Produces the A0 OOF predictions, the Gate-C stratified sample, the frozen tercile weights and the blank annotation package. `msc_subset_sha256 = null` (submitted without `--gate-c-audit`). Its `verdict.json` is void *ab initio* per D-1 §2. | `artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/` |
| 9 | 2026-08-07 ~07:04 | Second-coder plan revised: a Qwen download was started and **abandoned**; user adjudication changed the second coder to a second Claude instance (recorded in D-2's authority line, together with the widening of the DUA frame-exposure exemption). | `logging/runs/qwen_download/` |
| 10 | 2026-08-07 07:08:37 | **D-2 registered** — Gate-C annotators are Claude Opus 5 agents, not humans; written before any Gate-C label was produced (`gate_c_audit.jsonl` did not exist; no coverage/kappa/msc quantity computed). | `refine-logs/TERA_GATE0_DEVIATION_D2_2026-08-07.md` |
| 11 | 2026-08-07 → 08:08:43 | **Primary coding.** `claude-opus-5-c1` labels all **133** audited items (73 FN + 30 TP + 30 FP controls), one separate agent instance per item, blinded to score/category/retrieval/TERA output. | `logging/runs/gate_c_annotation/claude_c1_rows.jsonl` (133 rows) |
| 12 | 2026-08-07 → 08:21:16 | **Independent double coding.** `claude-opus-5-c2` re-labels the **27** items registered in `gate_c_sample.json["double_coded"]`, with no shared context and no sight of any c1 row or of the fact that a c1 label exists. | `logging/runs/gate_c_annotation/claude_c2_rows.jsonl` (27 rows) |
| 13 | 2026-08-07 08:26:30 | **Adjudication and audit assembly.** `claude-opus-5-adj` resolves the **5** double-coded items whose two `primary_cause` values disagree. Rows are concatenated c1 → c2 → adjudicated, never rewritten or deleted; that ordering is load-bearing for the frozen harness's kappa pairing. **165 rows** total. | `logging/runs/gate_c_annotation/claude_adj_rows.jsonl` (5 rows), `…-7ba80eaf/gate_c_audit.jsonl` (165 rows) |
| 14 | 2026-08-07 08:32:26 | **D-3 registered** — a defect in the **frozen bytes**: `msc_subset` silently dropped double-coded videos whose two coders agreed. Found by code reading during a Gate-C hand-off review, **after** the audit was assembled and **before** Run 2 was submitted; at that moment `msc_subset.json` did not exist in any run directory and no msc subset, rescue rate, FP count, Gate-B decision or Gate-C coverage/kappa quantity had ever been computed. §12 "stop before computing the affected candidate metric" path — no verdict invalidated. | `refine-logs/TERA_GATE0_DEVIATION_D3_2026-08-07.md` |
| 15 | 2026-08-07 08:34:05 | Fixture battery v2, first launch: died in 2 s with `FileNotFoundError` on the deliberately-unpatched `.draft.json` argparse default (appendix §10.1 working as designed). Directory retained as the record of the failure; **not** the release report. | `artifacts/tera_gate0/_fixtures/fix-20260807T083405Z/` |
| 16 | 2026-08-07 08:35:46 | **Fixture battery v2 release run: 16 / 16 PASS / 0 FAIL**, 965.2 s, 75 assertions (72 at v3 + 3 new F11). **Directed regression evidence**: the three new F11 assertions were run against the v3 bytes in a scratch copy and **FAIL** there, so they test the defect and not merely the code. | `artifacts/tera_gate0/_fixtures/fix-20260807T083546Z/fixtures_report.json`, `logging/runs/tera_gate0_fixtures_v2/run.log` |
| 17 | 2026-08-07 08:57:16 | **RE-FREEZE (appendix v4).** Payload `7ba80eaf…` → **`f2caade9…`**; `run_id` prefix becomes `tera-gate0-<UTC>-f2caade9`. Completeness proof: reverting exactly and only the rewritten payload fields to their v3 values reproduces `7ba80eaf…` bit-for-bit — no threshold, seed, arm, split, cache digest, taxonomy entry, decision rule, HALT condition, whitelist or asset-audit value was touched. | `refine-logs/TERA_GATE0_REFREEZE_2026-08-07.md` |
| 18 | 2026-08-07 09:01:11 → 09:26:09 | **Run 2 — the registered decision run.** `--stages A,C,B --gate-c-audit …-7ba80eaf/gate_c_audit.jsonl --confirmation all`, 1497.9 s, CPU, `status = COMPLETE`, `halt = null`. One submission. **Verdict `NO-GO-C`.** (Confirmation unlocked at 09:25:22Z — see §5 erratum 1.) | `artifacts/tera_gate0/tera-gate0-20260807T090111Z-f2caade9/` |

---

## 2. Pointer and hash chain

### 2.1 Campaign records

| record | path | sha256 (at close-out) |
|---|---|---|
| freeze (appendix v3) | `refine-logs/TERA_GATE0_FREEZE_2026-08-07.md` | `29f19495562b13bcf20e1c913691264b813e106a12031941a5f5e8eb4bf56ad0` |
| deviation D-1 | `refine-logs/TERA_GATE0_DEVIATION_D1_2026-08-07.md` | `0eb9c2c7344a426bc6a6a8a791762e9e244fa85e543c3c1333dfd974c7826255` |
| deviation D-2 | `refine-logs/TERA_GATE0_DEVIATION_D2_2026-08-07.md` | `0c21e04d15c921937351bebdcd993fc28b5914243d0511d7351e68b28968bd25` |
| deviation D-3 | `refine-logs/TERA_GATE0_DEVIATION_D3_2026-08-07.md` | `ae252f569e7dc0b6d7a9179b5f948e20d222db4a287bf4bfd14cc29ccb008033` |
| re-freeze (appendix v4) | `refine-logs/TERA_GATE0_REFREEZE_2026-08-07.md` | `842df40ebd08bc63edc7cdbc5dc82ecbff473038d2a3aa2dfd8e8b53816d044f` |
| this record | `refine-logs/TERA_GATE0_CAMPAIGN_RECORD_2026-08-07.md` | — |

### 2.2 Frozen-artefact hash chain, v3 → v4

| item | v3 (freeze) | v4 (re-freeze, D-3) |
|---|---|---|
| **canonical payload hash** | `7ba80eaf697ac46bb90b30161b1726aba7ee238e73001dd832ce30dba8a1dabe` | **`f2caade97712f8421232dee0a9c6b02545e3ac9ce95357e82e664802316a81e0`** |
| `run_id` prefix | `tera-gate0-<UTC>-7ba80eaf` | `tera-gate0-<UTC>-f2caade9` |
| implementation appendix | `ea158b2c23bd0a9ed8cecdbaccdecd21e97621f9a88b3db8a7c2dcbba2c42ffc` | `06808e12d737bd5b43cb8b0cd4779428c443adfa064e3c7cf750216aa356231e` |
| frozen config, whole file | `fdebff8bd72b704f0a5da8e007145bdb06a1f365c6ed2ab4e38507bf92541bdc` | `e45abb3749130e43b2135016f92047e068605aa18616d69dffbdc887267fcb82` |
| harness `package_aggregate_sha256` | `7e20884b6272bc98a94a367dc2823ac06c772c16d54a5f1bd415993c11f8e9f2` | `cb619464b0223ed551f6078d31a67a4a9f832bb42f59d540136fe8d7dd7463aa` |
| `gate_c.py` | `27dc026d1f0fea882ee71e007660a6efe5d06cac4531dbe39d07c5e37c05bd6b` | `811b292dd0aa5831c1b7b2ffd8ea5eeda5b1d755f10c29637cbdad779a42e6f9` |
| `run_gate0.py` | `a947af0cf05548c802b4f30cccdc3eb7e4d32c533606a9dd08fee162a122d81d` | `12a5f10d513983f843e95aefe13bc06f370043b4cd4ccbabe59cc736e5799e9a` |
| `fixtures.py` | `1cd6c48226345c91c7423c7c61e805f94c8d05c36492af814a1bc266491dfd36` | `d967f78e87fe31e4275ca163834bc304f6314a36f4e031b0de90825f0d282f7c` |
| fixtures report | `f21b465e69ac11dc620dfdf9bc66e676cd6749bde65344f1e33762ed979a1fb5` (`fix-20260806T231531Z`) | `b9161be50cd33227eb1c158e378f32ff0f3624e5f903ad1267a83fca137021e0` (`fix-20260807T083546Z`) |
| pre-registration | `f6c1ce6c652bcedd18451d4ee3a490ca2c72c603489e89c6a161855537ed6e98` | **unchanged** during execution (back-filled at close-out, §6) |
| independent review record | `9147ad4c1adacf1160566f0503937d45e0f5205b43549359d3eefe68263637c5` | **unchanged** |

The other 11 `*.py` files of `scripts/tera_gate0/` are byte-identical across the re-freeze.

### 2.3 Run artefacts

| item | value |
|---|---|
| Run 1 directory | `artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/` |
| Run 1 command | `run_gate0.py --config research-wiki/tera_gate0_frozen_config.json --stages A,C --confirmation none` |
| Run 2 directory | `artifacts/tera_gate0/tera-gate0-20260807T090111Z-f2caade9/` |
| Run 2 command | `run_gate0.py --config research-wiki/tera_gate0_frozen_config.json --stages A,C,B --gate-c-audit artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/gate_c_audit.jsonl --confirmation all` |
| Run 2 `verdict.json` | sha256 `7c97e16c664c281433d0a6a92b8a9543737155605e0834c56c5b01c4287339eb` |
| Run 2 `metrics.json` | sha256 `ccc5d51d474c174c46fdcef6f2cc4b22833cf7457675a66506901aac47027bf5` |
| Gate-C audit input | `…-7ba80eaf/gate_c_audit.jsonl`, sha256 `491a2fbaa1bc15c41e960f39c2b54e8ccd4ecdf5694c6be01ffa418a88fc071d`, 165 rows |
| `gate_c_sample.json` | sha256 `d43a22975f1f485ab420dcc1e1cf798baca9aa2bcdbe2415c0dc426f40703bed` |
| `msc_subset.json` (Run 2) | sha256 `bf5655072d16a0d74b73361982c8fe0f84c38ef62e21439c8d02fe8eef7c6fe6` |
| annotation rows, c1 | `logging/runs/gate_c_annotation/claude_c1_rows.jsonl`, sha256 `776563c60dd336ea4d5f61850c646ef448bb87523896c11ac92fe3480a19ec26` |
| annotation rows, c2 | `logging/runs/gate_c_annotation/claude_c2_rows.jsonl`, sha256 `7efc41e40d5de79c0d288da0b897344b00d28d96e8079fa7c4e22cef82879477` |
| annotation rows, adjudicator | `logging/runs/gate_c_annotation/claude_adj_rows.jsonl`, sha256 `396abc5a508d313f049fe4d3cf642f5bd1305add7ef9b30c6cec93b9ed97f198` |
| primary feature cache | `data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt`, sha256 `8b4a706cec51d106151e57109b24850232239168d5e0ca363341ee76493d7fb7` |
| whole-video cache | `data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt`, sha256 `0802b6ba00669ec546e63f36dca1772cb2d7806b969de307235af3450a8176c1` |
| gold spans (hash-only read) | `data/gt/HateMM/hate_spans.json`, sha256 `f8f2be10856a40c0ef5763b9211ecbed506743792ccddfb3adc92bed460c1846` |

---

## 3. Final verdict

`artifacts/tera_gate0/tera-gate0-20260807T090111Z-f2caade9/verdict.json` → **`verdict: "NO-GO-C"`**,
`status: "COMPLETE"`, `halt: null`.

Gate-C sample (`metrics.json → gate_c_sampling`): OOF false-negative population **73** on
HateMM-train — at or below the §4.1 cap of 120, so `audited_all = true`, every false negative was
audited, tercile weights are all 1, and weighted coverage equals unweighted coverage exactly.
Tercile population sizes 24 / 24 / 25 equal sampled sizes; cuts `q33 = 0.20950712263584137`,
`q67 = 0.3440778851509094`. Controls: 30 TP + 30 FP, never in the FN denominator. 27 of 133
audited videos double-coded (20.3%, above the 20% floor). Sampling seed `20260807`; video
bootstrap 10,000 resamples, seed `20260809`.

| §4.3 criterion | bar | observed | count | decision |
|---|---|---|---|---|
| union{`short_localized`, `multi_segment_complementary`, `cross_modal`} | `>= 0.30` | 0.8356164383561644 | 61/73 | PASS |
| union bootstrap 95% CI lower bound | `>= 0.20` | 0.7534246575342466 | 55/73 (upper 0.9178082191780822 = 67/73) | PASS |
| **`multi_segment_complementary`** | **`>= 0.15`** | **0.0821917808219178** | **6/73** | **FAIL** |
| `annotation_ambiguity_or_noise` | `<= 0.25` | 0.1643835616438356 | 12/73 | PASS |
| double-coded primary-cause Cohen's kappa | `>= 0.60` | 0.7326732673267327 | raw agreement 0.8148148148148148 = 22/27 | PASS |

`gate_c.pass = false`; one binding failure, by roughly a factor of two (6 of 73 where 11 were
needed). Stopping rule applied: **prereg §9 bullet 1** — *"C fails → stop, `NO-GO-C`"* — reinforced
by §4.3 — *"do not run A or B, and do not claim temporal evidence is a large enough performance
lever."*

This is **not** the §4.3 reliability escape hatch: kappa passed at 0.733, so the result is
substantive evidence against the registered compositional hypothesis, not a measurement failure.
It is also not a power problem: the audit is a census of the entire false-negative population.

Integrity of the decision run: `test_contact_count = 0`, `opened_test_paths = []`,
`failure_rate = 0.0013440860215053765` against the 1% HALT bar, `zero_vector_videos = 1`
(`hate_video_95`), `missing_duration_videos = 0`, and all four overlap assertions
(`outer_disjoint`, `inner_nested`, `segment_disjoint`, `one_query_fold_per_video`) true.

---

## 4. Sealing declaration — Gate-A and Gate-B

Under prereg §9 the decision path **stops at C**, so Gate-A and Gate-B carry no registered
decision in this campaign. Under D-1 §2 clause 4 — *"if Gate-C returns NO-GO, Run 1's
`metrics.json` and `verdict.json` remain sealed and this record stands as the note that they were
void from the moment they were written"* — the seal is **permanent**, and it is applied here to
**both** runs:

1. No Gate-A quantity from either run is read, quoted, summarised, transcribed or acted on in
   this record, in `research-wiki/EXP_tera_gate0_prereg.md`, or in `TARGET_FINDINGS.md` — this
   covers the A0/A1/A2/A3/A4 and O1/O2 arm metrics and their deltas, the arm-`D` identity and its
   selection statistics, the paired bootstrap CI, the temporal within-video AUROC, and the
   confirmation deltas.
2. No Gate-B quantity is read or reported. Run 2's `gate_b` is `null` on disk in any case: stage B
   was requested (`stages_run = ["A","C","B"]`) and skipped, `forced_stage_b = false` — see §5
   erratum 2.
3. Run 1's `verdict.json` is void *ab initio* and constitutes no decision of any kind, per D-1 §2
   clauses 1–2. The A0 confusion matrix is the one exempt object: it is the Gate-C sampling frame,
   not a TERA candidate metric.
4. Both runs' `metrics.json` and `verdict.json` are **retained unmodified** for audit
   completeness. Confirming their existence and hashing them is permitted; opening their Gate-A/B
   sections is not.

Every number in this record and in the two documents it indexes comes from `gate_c`,
`gate_c_sampling`, `manifest.json`, or the sampling frame.

**Incidental-exposure note, for completeness.** Run 2's `verdict.json` is a single small file whose
top level contains both the sealed `gate_a` block and the `gate_c` block that had to be read to
produce this record, so the sealed block was visible on that read. Nothing from it has been
transcribed, summarised or used anywhere, and the seal above stands unchanged. Any future reader
should extract `gate_c` by key rather than opening the file whole.

---

## 5. Post-run errata

Both were discovered **after** Run 2 completed and after the verdict was read, so neither can have
influenced any registered decision. Both are registered in the prereg's
`REGISTERED DEVIATIONS / ERRATA` section.

1. **The confirmation set was consumed by a run that stopped at C.** `run_gate0.py` executes
   `run_confirmation()` under `if self.args.confirmation != "none":` **unconditionally** — the
   §9 "C fails → stop" rule is enforced in the reported verdict, not in the harness control flow.
   Run 2 was launched with `--confirmation all`, so the passes were spent:
   `confirmation_unlock_utc = 2026-08-07T09:25:22Z`,
   `confirmation_passes = {"hateclipseg_val": 1, "hatemm_val": 1}`, with
   `confirmation_predictions.jsonl` and `confirmation_summary.json` written into the Run 2
   directory. **No test data was touched** (`test_contact_count = 0`, `opened_test_paths = []`);
   what was spent is the §7.10 val-side confirmation allowance. Because the run stopped at C, the
   consumed confirmation supports no registered claim, and its outputs fall under the §4 seal. A
   future re-execution must either re-register the confirmation budget or run the decision run
   with `--confirmation none` until Gate-C has passed.
2. **Stage B is structurally unreachable in a non-fixture run.** The branch reads
   `if not self.gate_a["pass"] and not self.args.fixture_mode: note("stage B skipped …")`, and the
   `else` branch — the only assignment site of `self.forced_stage_b` — is entered only when
   `self.gate_a["pass"]` is true, so `forced_stage_b` can never be true outside fixture mode; it
   is dead code. `"B"` in `--stages` therefore runs stage B only if Gate-A passed. **No effect on
   this verdict**: §9 stops at a Gate-C failure and forbids Gate-B from binding, so a Gate-B
   number would have been inadmissible regardless, and none was produced. If this design is ever
   re-run, this is a **D-4-class material deviation** that must be registered, fixed and
   re-frozen before any run in which Gate-B could bind.

---

## 6. Close-out documentation

- **Prereg back-fill.** `research-wiki/EXP_tera_gate0_prereg.md` received, at close-out and after
  the terminal verdict, an appended `REGISTERED DEVIATIONS / ERRATA` subsection (D-1/D-2/D-3
  summaries with file pointers and hashes, plus the two post-run errata) and a `RESULTS` section
  (Run 2 identifiers, the full Gate-C criteria table, the verdict and its §9 stopping rule, and
  the Gate-A/B sealing declaration). **The registration text above the `REGISTERED DEVIATIONS /
  ERRATA` heading is unchanged byte for byte**; the registration-time `RESULTS` line ("Not run…")
  is retained verbatim and marked superseded rather than deleted. This is precisely the back-fill
  D-1 §3 deferred to campaign close-out. It changes the file's digest away from
  `f6c1ce6c652bcedd18451d4ee3a490ca2c72c603489e89c6a161855537ed6e98`; no registered execution
  remains to be launched under either frozen payload, and any future re-execution requires a fresh
  freeze under §12 regardless.
- **Findings ledger.** `TARGET_FINDINGS.md` → **F122**.
- **Nothing was committed.** All Gate-0 artefacts, records and edits are working-tree state; the
  commit decision belongs to the main conversation.

---

## 7. What the campaign leaves on the board

The registered TERA route is **closed**, at the cheapest possible point: a CPU-only audit falsified
the method's founding assumption before a single line of TERA training code was written. Only
**8.2%** of the baseline's errors require two separated evidence units to interact, so even a
perfect compositional method could touch at most 6 of 73 false negatives.

The same audit records one large descriptive fact that is **not** a licensed claim and **not** a
promotion path: **83.6% (61/73)** of false negatives are attributable to the union of
short-localized, multi-segment-complementary and cross-modal evidence, bootstrap lower bound
**75.3%**. Localized and cross-modal error mass is abundant; only the compositional part is thin.
A single-segment or cross-modal selection route is a plausible next candidate but **requires its
own preregistration** — prereg §12 forbids reading a failed criterion as a near-pass on a
neighbouring quantity, and §10 authorises a performance claim only on a full Gate-0 pass.

Per the fast-kill discipline, the direction is closed now and the next candidate is drawn from the
queue rather than after a rescue attempt on TERA.
