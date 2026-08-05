# C04 Existing-Bank Stage-0 Asset Audit

**Status:** `FROZEN FOR INDEPENDENT DESIGN REVIEW`  
**Date:** 2026-07-29 (Pacific/Auckland)  
**Scope:** filename, schema, code and prior-record inspection only; no Python,
cache loading, test access or computation

## Binding question

Can C04 run the registry's pre-teacher/pre-GPU reachability gate using existing
full-bank assets on two datasets, without pretending that an old field is a
four-factor label?

An eligible Stage-0 must use the exact proposed tensor interface, actual
fold/deployed-head path, train-memory to untouched-dev evaluation, and the
frozen `+0.050` accuracy and macro-F1 threshold on both HateMM and MHC-ZH.

## Asset inventory

| Asset | Coverage / fields | Admissible role | Disqualifying limitation |
|---|---|---|---|
| `data/Summaries/{HateMM,MHC_zh}/{train,val}.jsonl` and `train/dev_seen_p8sum_HF.pt` | full train/dev; label-free Qwen text summary encoded by CLIP | proposition proxy/input | unstructured compression, not source or stance; P8 was negative end to end |
| `data/MLLM_scores/{HateMM,MHC_zh}/{train,dev_seen}_segscoreK4_qwen.jsonl` | full train/dev; four label-blind density scores | coarse harm-surface proxy | scalar/localization signal, not harm-act binding; P3/P11 family is negative |
| `data/Archive/MHC_zh/v2/*.jsonl` | target groups, mechanism, modality cues, explicitness, neutral summary | secondary harm/proposition ordinary control | no source-agent or stance; records carry labels outside the generated payload; target/mechanism are known weak fields |
| `artifacts/lb_scgp_global/v1/m1/cache/{MHC,MHC_zh}` | source alignment, counter context, harmful surface and six other structural observables | historical kill-only control | `source_alignment` is cross-modal alignment, not proposition source; proposition/stance were forbidden by schema; 8.74%/6.91% parse coverage and measured zero conditional information; no HateMM |
| `artifacts/c3_nontarget/{MHC,HateMM}` | 300 examples per dataset of unstructured dense reasoning text | historical rationale control | not full bank, no MHC-ZH, no factor schema, and DEAD_AT_FUSION on the strongest bank |
| `data/gt/HateMM/target_pred_qwen7b.jsonl` | predicted target codes, mixed split | target-only forensic input | no split field, no ZH counterpart, no other factors |
| SSR/P2/Role3 pair and rationale files | pair relations or final verdict rationales | none | one-dataset, label-bearing, mixed-split or test-tainted paths; excluded |

No existing file is a matched, explicit
`source_agent + proposition + presenter_stance + harm_act` bank on both
HateMM and MHC-ZH.

## Legal existing-bank Stage-0

The only non-fabricated path under the current registry is a conservative proxy
constructed exclusively from already banked label-blind payloads:

1. **P (proposition proxy):** the existing P8 summary embedding, using only the
   summary `text_feats`; copied image features and outer labels are unread by the
   factor reader.
2. **H (harm proxy):** the four existing K4 scores, normalized by the frozen
   `0..3` scale and augmented with explicit missingness.
3. **S and T (source/stance proxy):** a new *deterministic* bilingual cue compiler
   over the already banked `orig_text` and `summary`. Its source, quote/report/
   lyric/archive, endorse, reject/counter and uncertain cue lists are frozen
   before any label or metric read. It makes no MLLM call and has an explicit
   `uncertain` output.
4. All four proxies are encoded by the same role slots and tensor operator that
   Stage-1 would use. The compiler produces no prediction, weight, selection,
   pair, target name or sample exclusion.

This is a deliberately weak *mechanism reachability* screen. It does not turn
the cue states into gold and cannot support a source/stance quality claim.

## Two Stage-0 arms are both mandatory

- `S0-DIRECT-TENSOR`: privileged upper-bound arm in which the sealed proxy tensor
  is present for train and dev. It asks whether the tensor contains enough
  conditional information under the actual fold/deployed-head evaluator.
- `S0-STUDENT-TENSOR`: train proxy tensors supervise the tensor student; dev
  receives native video input only. It asks whether the signal survives the
  teacher-removal path C04 actually claims.

Both must achieve at least `+0.050` accuracy and `+0.050` macro-F1 against the
paired strongest baseline on both datasets, with enough net fixes for the final
`+0.030` bar. FULL must also beat the strongest ADDITIVE/LOWER-ORDER ordinary
control by `+0.020/+0.020`. A raw-key or direct-only pass cannot promote C04.

## Contract interpretation and decision fork

The registry text says the reachability gate occurs **before teacher/GPU spend**.
It does not authorize building the missing explicit four-factor bank first.
Therefore:

- if the deterministic existing-bank tensor passes both mandatory arms, C04 may
  proceed to a new train-only local-teacher Stage-1 after independent
  result-to-claim review;
- if it fails, C04 is killed under the current registry even though a stronger
  teacher might have helped;
- if the independent reviewer rules that this proxy may KILL but can never PASS
  because S/T are not representation-matched, the current contract has no
  executable C04 path.

The smallest possible escape from the third branch is a **user-approved registry
amendment**, not an agent assumption: authorize one train-only, label-blind local
Qwen cache on at most 200 frozen-ID videos per dataset solely to instantiate the
four-factor Stage-0 signal gate. No dev/test teacher, training or final claim
would be authorized by that amendment. Estimated ceiling: one sequential GPU,
at most 2 GPU-hours total plus a CPU seal/probe, all via SLURM. The exact
resource estimate must be rechecked before any authorization.

## Exclusions

No test file, `target_map.json` gold oracle, mixed-split Role3/P2 artifact,
teacher verdict, archive outer label, factor-quality manual label or
error-derived per-item rule may enter Stage-0.

