# Phase 0 Anchor: LB-SCGP Global Pivot

## Immutable Problem Anchor

hateful video detection adapting RGCL/RA-HMD to video; MLLM meaningful+novel; final MHC-EN/MHC-ZH seeds0/1/2 vs strongest same-protocol non-MLLM, acc and macro-F1 each ≥+0.030, all paired seed deltas positive, hierarchical paired bootstrap lower>0, Holm; only parent-video binary gold, no segment/timestamp/span/localization/stance/target/mechanism/rationale gold; train-only label-blind MLLM cache; test ordinary full-video train-memory top20 kNN, no teacher/head/rerank; SLURM; no sample weighting/key selection/pair-triplet/SupCon/segment route; REMOVE/SHUFFLE/NOISE/direct attribution. Local rank-cell v7 formally retired, no v8.

Absolutely do not assume any fragment/segment has gold annotation. The only gold is parent_video_binary_label. Any segment/timestamp/span/localization/stance/target/mechanism/rationale output is not gold and may not be treated as supervision, pseudo-groups, selection, or evaluation gold. Preserve this literally and operationally.

## Bottom-line Problem

Build a meaningful and novel MLLM integration for adapting RGCL/RA-HMD to hateful-video detection, while preserving the final endpoint as ordinary full-video train-memory top20 kNN. The target is not a diagnostic, a teacher score, a head improvement, or a local solver certificate; it is final paired performance on MHC-EN and MHC-ZH.

## Must-solve Bottleneck

Previous MLLM routes either became verdict/rationale features, selector/reranker/key mechanisms, sample weighting, sparse local pair constraints, segment-local routes, or solver/certificate chasing. The global pivot must let label-blind train-only whole-video structural certificates alter a full-bank representation geometry without treating MLLM fields as gold, without selecting samples or pairs, and without relying on local rank-cell stationarity.

## Non-goals

No segment, timestamp, span, localization, stance, target, mechanism, or rationale supervision or evaluation claim. No test-time MLLM, teacher, head, reranker, router, score fusion, key selection, sample weighting, pair/triplet/SupCon addition, segment route, or v8 local rank-cell repair. No performance claim before a new global G0 and later staged validation pass.

## Constraints

All compute must use SLURM with the `HateVideo` conda environment and no `--time`. The only gold is `parent_video_binary_label`. MLLM cache construction is train-only and label-blind. Labels first enter only after the cache is sealed, inside a deterministic compiler for train-only parent-label vote preservation and final metric evaluation. Validation/test never load certificates, target banks, compiler artifacts, teacher outputs, heads, or rerankers.

## Success Condition

Final success requires MHC-EN and MHC-ZH, seeds 0/1/2, strongest same-protocol non-MLLM comparator, accuracy and macro-F1 each at least +0.030, every paired seed delta positive, hierarchical paired bootstrap lower bound >0, and Holm correction. FULL must also survive REMOVE, SHUFFLE, NOISE, and direct-attribution controls; otherwise the MLLM/global-proximal mechanism is not supported.
