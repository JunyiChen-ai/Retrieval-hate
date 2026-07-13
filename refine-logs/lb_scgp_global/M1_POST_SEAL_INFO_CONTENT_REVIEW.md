# M1 POST-SEAL INFORMATION-CONTENT REVIEW — lb_scgp_global M1 cache

Date: 2026-07-13
Author = **Claude Opus 4.8** (`claude-opus-4-8`, 1M ctx), fresh zero-prior-context **post-seal
information-content reviewer** for the `lb_scgp_global_r2` M1 gate. This role is separate from
m1-prep (v2 author/freezer), the v2 code reviewer, the v2 execution-authorizer, and the seal
executor (`M1_CACHE_V2_EXECUTION_RECORD.md`). I did NOT re-seal, submit SLURM, or use a GPU. All
work was read-only over the sealed artifacts + CPU/jq/python under conda `HateVideo`. Write scope =
this file only. Not committed (archiver handles commits).

**Scope note:** the seal (`M1_CACHE_V2_EXECUTION_RECORD.md`, job 13035, decision GO) is
**procedurally valid** — I confirm that independently below. This review answers a *different*
question the seal does not: **how much usable signal the sealed cache actually carries**, and
whether the downstream GPU spend (M2 48 GPU-h + M3 216 GPU-h = **264 GPU-h**) is justified under the
project's new mandatory zero-cost conditional-information institution
(`research-wiki/REFLECTION_mllm_integration_failures.md` §4).

---

## 0. VERDICT

**PAUSE_A_LINE.** Do **not** spend the 264 M2+M3 GPU-hours on the current sealed cache, and do
**not** blindly launch a v3 repair either. The sealed cache is *not* a literal all-constant blob
(its ~8.7%/6.9% parse-ok subset carries genuine cross-video variance), but its effective signal is
**structurally too sparse and too low-entropy** to plausibly clear the C2 bar (+0.030 acc AND
macro-F1 on BOTH datasets × 3 seeds, bootstrap lower > 0, Holm ×4) against a ±1–2 pt test noise
floor on 161/149-sample test sets. Before any further GPU (M2 **or** a v3 repair), run the
**mandatory zero-cost G0-cond oracle probe** (§4-reflection) on the sealed parse-ok subset; the
whole campaign's evidence (D1 redundancy) predicts it fails, in which case redirect GPU to C-line and
**archive M0 + M1-sealed as the clean stopping point**. A v3 infra-repair is technically plausible
(the failure mechanism is a fixable OOM/sequence-length exception, §2d) but is **conditional** on the
probe passing — see §3.

---

## 1. Independent seal-evidence spot check (Task 1) — CONFIRMED, no discrepancy

Second independent pair of eyes over the sealed artifacts. Every execution-record key claim holds.

| Check | MHC | MHC_zh | Source (this session) |
|---|---|---|---|
| `cache.jsonl` row count | **2196** (=4·549) | **2316** (=4·579) | `len(rows)` |
| distinct `video_id` | **549** | **579** | `set(video_id)` |
| `replica_index` set | **{0,1,2,3}** | **{0,1,2,3}** | `set(replica_index)` |
| unique (video_id, replica) pairs | **2196** | **2316** | `set(pairs)` |
| videos without exactly 4 replicas | **0** | **0** | `Counter(video_id)!=4` |
| label-family keys anywhere (recursive: label/split/seed/neighbor/prediction/margin/gold/y_true/is_hate/target/parent_video_binary_label) | **0 rows** | **0 rows** | recursive key scan |
| manifest `record_count` / `parse_ok_records` / `parse_rate` | 2196 / 192 / **0.0874** | 2316 / 160 / **0.0691** | `cache_manifest.json` |
| `zero_counters` (30 keys) all 0 | **all 0** | **all 0** | `jq .zero_counters` |
| `cache_merkle_root` | `ad98d8e8…` | `563bcefb…` | manifest |
| seal decision | **GO / CACHE_SEALED** | — | `cache_seal_decision.json` |
| per-dataset `verified` + `merkle_root_recomputed_match` | **true / true** (recomputed `ad98d8e8…`) | **true / true** (recomputed `563bcefb…`) | seal decision `.per_dataset[].seal_checks` |

**No discrepancy** with `M1_CACHE_V2_EXECUTION_RECORD.md`. The seal is valid. The `parse_rate`
flag the executor surfaced (§0 of that record) is confirmed and is the subject of the rest of this
review.

---

## 2. Information-content analysis (Task 2)

### 2a. Row-value diversity — the fallback rows are ONE literal constant

Semantic payload hashed over the 9 observables' `{state,confidence}` + `parse_flags` (excluding
`video_id`, `replica_index`, `evidence_pack_sha256`, `schema_version`). Command: python `sha1` over
that tuple, `Counter` of hashes.

| | MHC | MHC_zh |
|---|---|---|
| rows | 2196 | 2316 |
| **distinct semantic payloads** | **28** | **22** |
| top payload multiplicity | **2000** | **2104** |
| transport_fallback rows | 2000 | 2104 |
| **distinct fallback payloads** | **1** (literal constant) | **1** (literal constant) |

The ~91–93% transport_fallback rows are **literally one identical constant**: every observable
`state=unresolved`/`confidence=0` (modality_binding `state=unresolved`), `parse_flags=["transport_failure"]`.
**This kills the "fallback rows retain modality_binding or confidence variance" defense** — the
binary `parse_rate` does *not* undercount partial signal; the fallback carries exactly **0 bits**.

### 2b. Parse-ok subset — real variance, but tiny coverage, and R=4 is fully redundant

Parse-ok = rows with empty `parse_flags`.

| | MHC | MHC_zh |
|---|---|---|
| parse-ok rows | 192 | 160 |
| **distinct videos covered** | **48 / 549 (8.74%)** | **40 / 579 (6.91%)** |
| per-video #parse-ok-replicas distribution | `{0: 501, 4: 48}` | `{0: 539, 4: 40}` |
| videos with ≥1 non-`transport_failure` replica | 49 (8.93%) | 53 (9.15%) |
| videos with ≥2 parse-ok replicas whose R=4 state-vectors are **identical** | **48 / 48** | **40 / 40** |

Two structural facts:
1. **Perfectly bimodal**: a video's 4 replicas either ALL parse or ALL fail. Greedy decoding on a
   fixed per-video input is deterministic, so R=4 replication bought **zero** extra information (the
   consensus/robust-vote machinery is decorative here; the K_r replica-stability gate will pass
   trivially with `sigma_cache=0`). This is D1-redundancy at the cache level.
2. **Two observables are dead constants everywhere**: `context_shift_observable` and
   `counter_context_observable` are `unresolved` in **all 192/160 parse-ok rows** and `0` in the
   consensus of **all 549/579 videos** (H = 0 bits). The model *never* resolves them — a
   prompt/task-design failure, independent of coverage; even a 100%-parsed cache would carry 0 bits
   from 2 of 9 observables.

Cross-video variance *does* exist in the parse-ok subset (this is why the cache is not a pure blob):
e.g. MHC `modality_binding` ∈ {single_modal 96, visual_text 52, multi_modal 20, visual_audio 12,
text_audio 12}; `text_audio_reference` ∈ {supported 120, unresolved 64, contradicted 8};
`source_alignment` ∈ {unresolved 116, supported 56, contradicted 20}. But it is confined to <9% of
the bank.

### 2c. Effective-signal estimate — the actual M2 compiler input (per-video `consensus`)

The M2 compiler consumes the manifest `consensus` field (the R=4 reduction to +1/−1/0 / modality).
Analyzed over ALL videos. `H` = Shannon entropy in bits.

| | MHC | MHC_zh |
|---|---|---|
| videos | 549 | 579 |
| **distinct consensus vectors** | **20** | **14** |
| videos in the constant all-unresolved class | **501 (91.26%)** | **539 (93.09%)** |
| **informative (non-constant) videos** | **48 (8.74%)** | **40 (6.91%)** |
| distinct informative consensus vectors | 19 | 13 |
| dead observables (H=0 over all videos) | context_shift, counter_context | context_shift, counter_context |
| top informative observable | modality_binding H=0.590 b | modality_binding H=0.477 b |
| naive Σ marginal entropies (loose upper bnd) | 1.807 b/video | 1.391 b/video |
| **JOINT entropy of full consensus vector** | **0.743 b/video** | **0.580 b/video** |
| total info in consensus layer | **407.9 bits / whole train set** | **335.5 bits / whole train set** |

Effective bits-per-video for the whole cache = **0.743** (MHC) / **0.580** (MHC_zh) — and that is the
*unconditional* certificate entropy. The label-relevant *conditional* information beyond the frozen
encoder features, I(C;Y|Z), is ≤ this and, per the campaign's repeated D1 finding, empirically ≈ 0.
Best-case ceiling (assume 100% of the cert entropy is novel AND label-aligned, and it exists only for
covered videos): average **cov·H_joint = 0.065 b/video (MHC) / 0.040 b/video (MHC_zh)**.

### 2d. Mechanism sanity-read — WHY 91% failed (raw outputs NOT stored → inferred from code + logs + correlation)

**Raw model outputs are NOT stored.** The producer (`lb_scgp_global_r2_m1_cache_producer_v2.py:190-208`)
discards `raw` after parsing and keeps only observables + flags. So a per-example raw read is
impossible; I characterize the mechanism from the code path, the deterministic per-video pattern, and
an input-size correlation.

Decisive code fact: `transport_failure` is **not a parse/format/refusal failure**. It is set only
when `one_call()` (which runs `model.generate`) **raises an exception**, caught by a bare
`except Exception:` that is **swallowed with no logging and no raw capture** (lines 194-208). Genuine
*parse* failures produce different, rare flags — MHC: `bad_state:cross_modal_binding` (4 rows);
MHC_zh: `missing_or_malformed:visual_reference` (44), `extra_keys:confidence` (4), `bad_state` (4).
So the 91–93% is a **transport/inference-layer throw**, categorically distinct from format drift.

Evidence for the cause (OOM / sequence-length), all this session:
- **Decode is not the primary cause.** MHC `.out` has 143 decord-fallback WARN lines
  (`DECORDError … cannot find video stream`), all "trying PyAV". Of the 500 transport-fail videos,
  **only 141 were decord-warned; 359 decoded cleanly and still threw.** So the dominant failure is on
  videos with good frames.
- **Strong input-size correlation** (proxy = longest string field in `data/gt/MHC/train.jsonl`, the
  transcript/title source):

  | group | n | median chars | mean chars | max |
  |---|---|---|---|---|
  | parse-ok videos | 48 | **94** | 169 | 1016 |
  | transport-fail videos | 500 | **429** | 476 | 1367 |

  The parse-ok videos are systematically the **short-input** ones. Combined with 16 frames of vision
  tokens (`NUM_FRAMES=16`) + `MAX_NEW_TOKENS=320` on a single GPU in bf16, this is the classic
  signature of an **OOM / KV-cache / sequence-length exception** that fires above an input-size
  threshold and is deterministic per input (hence all-4-replicas identical).

**Conclusion:** the failure is a **fixable infrastructure bug (OOM/seqlen), not a prompt/format
problem.** A "prompt/format repair" — the example fix named in the REPAIR option — would target the
wrong failure mode and would **not** raise `parse_rate`. A real fix (reduce frames, truncate
transcript, memory management, larger GPU) is plausible but **unconfirmed** (errors were swallowed)
and would require a **new cache lineage v3**.

### 2e. Downstream ceiling argument (grounded in the compile mechanics)

How the certificate enters M2 (`FINAL_PROPOSAL.md` §"Certificate encoding"/"Structural operator",
lines 189-291): the consensus/replica encodings `Phi` are centered `B0 = H_N Phi`, reduced to a
common basis `Q = orth_cap(B0, r_max=8)` (**rank capped at ≤ 8**), and the *only* signal injected is
the target moment `b_struct = vech(Qᵀ(K_C − I)Q / N)` inside a strongly-convex proximal projection
of the Gram `G` toward baseline `G0` under tight caps (`rho_row=0.05·√(N−1)`, `|G_ij−G0_ij|≤rho_coord`).
Final inference is ordinary top-20 kNN over the train memory. So the certificate's entire job is a
**rank-≤8, small-magnitude warp of the train-train Gram**.

Now the arithmetic against the C2 bar (`statistics_protocol.joint_success_rule`):

- The certificate partitions the train bank into **≤20 (MHC) / ≤14 (MHC_zh)** equivalence classes,
  with **≥91.3% / ≥93.1% of the bank in ONE class.** The proximal projection therefore imposes
  **zero differential geometry** among the 501/539 constant-class videos — they are geometrically
  interchangeable to the constraint. Only the **48/40 informative videos** can be moved relative to
  baseline, and only within the tight proximal caps.
- To *pass*, FULL must beat the frozen comparator by **+0.030 acc AND +0.030 macro-F1** on **both**
  datasets, with **all 3 seed deltas positive**, bootstrap lower > 0, Holm ×4. On the test sets that
  is **≈4.83 net correct flips / 161 (MHC)** and **≈4.47 / 149 (MHC_zh)** — *simultaneously* for acc
  and F1, on both datasets, all seeds — driven by a Gram warp that touches ≤8.7%/6.9% of memory
  embeddings by a bounded amount, in a direction governed by I(C;Y|Z) ≈ 0.
- Information ceiling: best-case effective **0.065 / 0.040 bits/video** (§2c, already granting the
  absurd 100%-novel-and-aligned assumption). At the method's ~0.80-acc operating point a per-video
  conditional-entropy reduction of this size projects to an accuracy change **well inside the
  ±1–2 pt noise floor** (`REFLECTION §D3`), i.e. **structurally below the +0.030 measurement line**,
  before even accounting for offsetting wrong-direction flips. The test-set noise band alone can
  produce ±2 pt swings that dominate any real effect and that the 3-seed × both-dataset × Holm
  conjunction is designed to (and will) reject.

This is the exact anti-pattern the reflection institutionalized: a low-bandwidth auxiliary signal
whose *marginal* quality gates (parse schema, merkle) all pass, but whose *conditional* information —
never yet measured for A-line — is the thing that decides the outcome, and it is near-zero.

---

## 3. RECOMMENDATION (Task 3)

**Primary: PAUSE_A_LINE.** Halt the M2/M3 path on the current sealed cache. Rationale is quantitative
(§2c/§2e): ≤8.7%/6.9% coverage, ONE class covering ≥91%/93% of the memory bank, joint cert entropy
0.743/0.580 b/video, 2 of 9 observables structurally dead, R=4 fully redundant, best-case effective
0.065/0.040 b/video — against a +0.030-both-metrics-both-datasets-3-seeds-Holm bar on 161/149-sample
tests with a ±1–2 pt floor. Spending 264 GPU-h to measure an effect this far below the noise line is
exactly what `REFLECTION §4` forbids.

**Mandatory gate before ANY further GPU (this is the actionable next step, zero GPU):** run the
`REFLECTION §4` **G0-cond oracle conditional-information probe** on the *sealed* parse-ok subset
(48/40 informative videos): compare codelength/accuracy of g(Z) vs g'([Z, C]) with a probe capacity
matched to the ordinary-kNN head, using train gold only for probing (compliant post-seal). Multi-seed
bootstrap CI. Decision rule:
- **Probe FAIL** (oracle conditional gain < +3 acc, CI includes 0 — *predicted* by D1 and the 14-route
  campaign): the whole certificate signal family is dead **at full coverage too**. Redirect the GPU to
  the C-line (literature candidates) and **archive M0 + M1-sealed as the clean, procedurally-valid
  stopping point**. Do not build v3.
- **Probe PASS** (real conditional gain on the parse-ok subset): only then is a **v3 infra-repair**
  warranted — because §2d shows the coverage loss is a fixable OOM/seqlen bug, not fundamental, and
  raising coverage would then be worth the cost.

**On REPAIR_LINEAGE (documented, NOT recommended as the immediate action):** a v3 is *technically
plausible* — the mechanism (§2d) is a fixable inference-layer OOM/seqlen throw (reduce frames /
truncate transcript / memory mgmt), so an infra fix could recover much of the 500/539 failed videos,
including longer-transcript (possibly more informative) ones. **GPU cost ≈ 8 h/dataset** (matching the
v2 jobs' 3.5–4.2 h × recovered fraction) **plus a full heavy-ceremony v3 lineage** (freeze → code
review → authorization → execution → seal → post-seal review). But three reasons make a *blind* v3
the wishful-thinking trap:
1. The named "prompt/format repair" targets the **wrong** failure mode (the 91% is transport, not
   parse) — it would not move `parse_rate`.
2. The cause is **unconfirmed** (errors swallowed); v3 would first need a diagnostic run.
3. **Even at 100% coverage the ceiling is bounded** — 2 dead observables, low per-video entropy, and
   the never-run G0-cond gate that D1 predicts fails. Raising coverage of a zero-conditional-info
   signal yields more zero.

Hence the correct ordering is **PAUSE → zero-cost G0-cond probe → (only on pass) v3 repair**, never
**M2/M3 or v3 first**.

**Adversarial check, both directions (as required):**
- *Not over-killing a live cache:* the parse-ok subset is genuinely non-constant (§2b) with real
  cross-video variance, and the failure is a fixable bug not a dead idea — so I do **not** declare the
  signal "zero" and I preserve the repair path behind the probe.
- *Not wishfully proceeding:* the fallback rows are a literal single constant (§2a, 1 distinct
  payload), the binary parse_rate does **not** undercount hidden variance, R=4 is redundant, and the
  quantitative ceiling (§2e) sits below the measurement floor — so PROCEED_TO_M2 on this cache is
  unjustified.

---

## 4. Numbers → source index (provenance)

- Row/replica/label-key/merkle spot checks (§1): python over
  `artifacts/lb_scgp_global/v1/m1/cache/{MHC,MHC_zh}/cache.jsonl` + `cache_manifest.json` +
  `cache_seal_decision.json`, this session.
- Payload/consensus diversity, entropies, coverage, replica agreement (§2a-c): python
  (`Counter`, `sha1`, Shannon `H`) over the two `cache.jsonl` and the manifest `consensus` field.
- parse_flags distributions: MHC `{transport_failure:2000, []:192, bad_state:cross_modal_binding:4}`;
  MHC_zh `{transport_failure:2104, []:160, missing_or_malformed:visual_reference:44, extra_keys:confidence:4, bad_state:cross_modal_binding:4}`.
- Mechanism (§2d): `lb_scgp_global_r2_m1_cache_producer_v2.py:154-216` (bare `except`→transport_failure,
  raw discarded), `lb_scgp_global_r2_m1_cache_v2_common.py:54,59` (NUM_FRAMES=16, MAX_NEW_TOKENS=320);
  `slurm/logs/lbscgp_global_r2_m1_cache_mhc_v2_13012.out` (143 decord WARN); decord-vs-outcome and
  transcript-length correlation computed this session over `data/gt/MHC/train.jsonl`.
- Compile mechanics + ceiling (§2e): `FINAL_PROPOSAL.md` lines 189-297 (common basis r_max≤8, target
  moment, proximal caps); `EXPERIMENT_PLAN.machine.json` `runs[7-25]` (M2/M3 specs), `claim_map.C2`,
  `statistics_protocol` (+0.030 both-metrics/both-datasets/3-seeds/Holm), `immutable_contract`
  (test_n MHC 161 / MHC_zh 149, ordinary top-20 kNN), M2 GPU=48 h + M3 GPU=216 h = 264 GPU-h.
- Institution: `research-wiki/REFLECTION_mllm_integration_failures.md` §4 (G0-cond gate), §5 (A-line
  "one M3 chance" — this review supplies the §4 zero-cost check §5 presumed passed).
