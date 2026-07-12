# SQ-RGCL Experiment Tracker

**Status date:** 2026-07-11  
**Execution status:** S0 was implemented, independently reviewed to `0 CRITICAL / 0 HIGH`, and executed through the formal fail-closed decision. `SQ-S0-DECISION-v1=STOP`; S1 was not submitted.  
**Hard lock:** no new teacher call is authorized. `S1_unlocked=false`, `S2_unlocked=false`, and S2--S4 remain locked. Only parent-video binary labels are gold; `segment_gold_exists=false`, `segment_gold_used=false`.

| Run ID / namespace | Stage | Purpose | Dataset / fold / arm | Binding checks | Resources / dependency | Priority | Status | Expected artifact / note |
|---|---|---|---|---|---|---|---|---|
| `SQ-S0-FREEZE-v1` | S0 | Freeze config, code/data inventory and supervision contract | both datasets | canonical config/hash; SSR fold/data hashes; zero calls; no segment gold | CPU 4 / 16 GB; first | MUST | COMPLETED_PASS job 12724 | `artifacts/sq/v1/CONFIG_FREEZE.json`; config `8645c9...c2149` |
| `SQ-S0-PROVENANCE-MHC-v1` | S0 | Recover original archive provenance and audit reader | MHC train | prompt/model/input/code original linkage; ID set; forbidden-key and label-poison audit | CPU 4 / 16 GB; after freeze | MUST | COMPLETED_PASS_PROXY_ONLY job 12726 | 549/549 IDs; forbidden access 0; original cryptographic linkage absent |
| `SQ-S0-PROVENANCE-MHC_zh-v1` | S0 | Same provenance audit | MHC_zh train | same | CPU 4 / 16 GB; after freeze; may parallel MHC | MUST | COMPLETED_PASS_PROXY_ONLY job 12725 | 579/579 IDs; forbidden access 0; original cryptographic linkage absent |
| `SQ-S0-QPROXY-MHC-v1` | S0 | Build six-way posterior from neutral summary only | MHC train | fixed prototypes/temp/confidence; reader access count; coverage/ESS; no field-concat cache | 1 A100 / 4 CPU / 32 GB; after provenance | MUST | COMPLETED_PASS job 12728 | 549 rows, coverage 1.0; remains `q_proxy`; zero teacher calls |
| `SQ-S0-QPROXY-MHC_zh-v1` | S0 | Same posterior construction | MHC_zh train | same | 1 A100 / 4 CPU / 32 GB; after provenance; <=2 GPU jobs | MUST | COMPLETED_PASS job 12727 | 579 rows, coverage 1.0; remains `q_proxy`; zero teacher calls |
| `SQ-S0-AUDIT-FREEZE-v1` | S0 | Freeze 64-ID/dataset label-blinded presentation QC sample | both train sets | posterior-argmax allocation; salted hash; rater sheet excludes labels/semantics | CPU 4 / 16 GB; after both qproxy jobs | MUST | COMPLETED_ARTIFACT_PREPARED job 12729 | 64 whole videos/dataset; no label column; awaiting human QC; never gold |
| `SQ-S0-AUDIT-MHC-v1` | S0 | Ingest two-rater + adjudicator QC | MHC, 64 videos | contamination <=3/64; appropriateness Wilson LB >=.90 | CPU 4 / 16 GB; after completed human sheet | MUST | NOT_RUN_BINDING_ABSENT | No human judgments were fabricated; absence is a formal S0 STOP reason |
| `SQ-S0-AUDIT-MHC_zh-v1` | S0 | Same blind QC | MHC_zh, 64 videos | same | CPU 4 / 16 GB | MUST | NOT_RUN_BINDING_ABSENT | No human judgments were fabricated; absence is a formal S0 STOP reason |
| `SQ-S0-PARITY-POWER-P0-MHC-v1` | S0 | SSR reuse, exact vote/exposure, P0 relevance, S2 power freeze | MHC five-fold train OOF | exact top20 parity; positivity; +.03 edge/error AUC; +.10 alignment; 10k anchor bootstrap; variance/FPC | CPU 4 / 16 GB; after qproxy | MUST | NOT_RUN_AFTER_BINDING_FAST_FAIL | Provenance/audit failure stopped before outcome/power computation |
| `SQ-S0-PARITY-POWER-P0-MHC_zh-v1` | S0 | Same | MHC_zh five-fold train OOF | same | CPU 4 / 16 GB; may parallel MHC | MUST | NOT_RUN_AFTER_BINDING_FAST_FAIL | No edge-level claim or power claim was made |
| `SQ-S0-MICRO-MHC-S0-v1` | S0 | Numerics, lambda freeze and six-arm GPU cost | MHC | scalar/vector <=2e-5; grad <=1e-3; <=24 GiB; FULL <=2x REMOVE | 1 A100 / 4 CPU / 32 GB; after qproxy | MUST | NOT_RUN_AFTER_BINDING_FAST_FAIL | `lambda_Q=null`; no resource/performance inference |
| `SQ-S0-MICRO-MHC_zh-S0-v1` | S0 | Same | MHC_zh | same | 1 A100 / 4 CPU / 32 GB | MUST | NOT_RUN_AFTER_BINDING_FAST_FAIL | `lambda_Q=null`; no resource/performance inference |
| `SQ-S0-DECISION-v1` | S0 | Independent joint fast-fail decision | both datasets | rehash/recompute provenance/audit fast-fail; explicit signal status; zero calls | CPU 4 / 16 GB | MUST | COMPLETED_STOP job 12730 | `PROXY_ONLY_CHEAP_FORMAT`; `S1_unlocked=false`; `S2_unlocked=false` |
| `SQ-S1-OOF-MHC-F{0..4}-S0-v1` | S1 | Learned actual ordinary-kNN SQ-0 folds | MHC, each fold; six arms inside job | identical init/order/epochs/bank; outer-held q read=0; predictions/neighbors for all arms | 1 A100 / 8 CPU / 64 GB each; after S0 GO; <=2 concurrent | MUST | NOT_RUN_LOCKED_S0_STOP | Entry requires verified S0 GO and therefore rejects this predecessor |
| `SQ-S1-OOF-MHC_zh-F{0..4}-S0-v1` | S1 | Same learned screen | MHC_zh, each fold; six arms | same | 1 A100 / 8 CPU / 64 GB each; after S0 GO | MUST | NOT_RUN_LOCKED_S0_STOP | No learned SQ accuracy/macro-F1 experiment was launched |
| `SQ-S1-DECISION-v1` | S1 | Recompute full two-dataset SQ-0 gate | 10 folds x six arms | +.05 acc/+ .05 mF1 vs moving non-MLLM; +.01 vs label/shuffle/random; all fold signs; 10k bootstrap/Holm; signed-mass and reach-beyond; zero calls | CPU 4 / 16 GB; after all 10 folds | MUST | NOT_RUN_LOCKED_S0_STOP | `S2_unlocked=false` already established by S0 |
| `SQ-S2-POWERED-CLOSURE-{MHC,MHC_zh}-v1` | S2 | Representative graph closure before calls | frozen powered anchors | <=128 unique/dataset; achieved power >=.80; fixed vertices/edges/IPW | LOCKED; only after S1 GO | CONDITIONAL | NOT_RUN_LOCKED | No shrinking or hub selection |
| `SQ-S2-TEACHER-{MHC,MHC_zh}-v1` | S2 | Four-invocation whole-video presentation pilot | <=128 unique/dataset | <=512 invocations/dataset; strict presentation-only schema; common-edge teacher advantage | LOCKED; only after closure GO | CONDITIONAL | NOT_RUN_LOCKED | No val/test call; no segment fields |
| `SQ-S3-{MHC,MHC_zh}-S0-{CONTROL}-v1` | S3 | Seed-0 causal controls | two datasets | dev ordinary kNN +.01 vs every binding control; corruption monotone | LOCKED; only after S2 GO | CONDITIONAL | NOT_RUN_LOCKED | Exact controls frozen in plan |
| `SQ-S4-{MHC,MHC_zh}-S{0,1,2}-{FULL,REMOVE,SHUFFLE}-v1` | S4 | Final endpoint/removability | 2 datasets x 3 seeds | moving-bar +.03/+ .03; all signs; bootstrap/Holm; FULL beats remove/shuffle | LOCKED; only after S3 GO | CONDITIONAL | NOT_RUN_LOCKED | Ordinary full-video kNN, no teacher at test |
| `SQ-S4-FINAL-STATS-v1` | S4 | Completion audit | all final ledgers | every project target and supervision item verified | CPU; locked | CONDITIONAL | NOT_RUN_LOCKED | Only this stage can prove completion |

## Submission DAG

```text
S0-FREEZE
   +--> PROVENANCE x2 --> QPROXY x2 --> AUDIT-FREEZE --> AUDIT-INGEST x2 --+
   +--------------------> QPROXY x2 --> PARITY/POWER/P0 x2 ---------------+--> S0-DECISION
   +--------------------> QPROXY x2 --> MICRO x2 --------------------------+

verified S0 GO --> S1 OOF 5 folds x 2 datasets --> S1-DECISION

verified S1 GO --> S2 powered graph closure (still no calls)
closure GO      --> S2 teacher pilot
S2 GO           --> S3 seed 0
S3 GO           --> S4 final 2 datasets x 3 seeds
```

## Current Evidence State

- SQ method specification remains READY (9.12), but the frozen execution route stopped at S0 before any learned SQ performance run.
- Jobs `12722--12730` (with the expected unused ID gap) completed the config hash, independently reviewed static sanity, formal freeze, two provenance audits, two local-CLIP proxy builds, blind whole-video audit-sheet freeze, and independent decision.
- Both archives have complete train ID coverage and reader forbidden-key access count zero, but neither embeds the original prompt/model-revision/input/code cryptographic linkage. Both signals therefore remain `PROXY_ONLY_CHEAP_FORMAT`.
- Two label-blinded 64-video whole-video QC sheets were prepared. Human ratings were not fabricated or treated as gold; their absence is binding. Decision job `12730` wrote `STOP`, `lambda_Q=null`, `S1_unlocked=false`, `S2_unlocked=false`.
- P0/power/micro and S1 were intentionally not run after the binding fast-fail. Thus no accuracy/macro-F1 evidence—positive or negative—exists for learned SQ.
- New teacher/MLLM/OCR calls and teacher-cache reads/writes were all zero. Only parent-video binary labels are gold; no segment gold exists or was used. The global two-dataset, three-seed final +.030/+ .030 objective remains unmet.
