# SQ-RGCL Iteration 4 — Frozen Problem Anchor

This block is immutable and must be copied verbatim into every SQ proposal and refinement.

- **Bottom-line problem:** Integrate an MLLM into hateful-video RGCL as a meaningful, novel, causally removable method component that produces a substantial improvement in the final ordinary full-video train-memory kNN classifier—not merely in localization, audit quality, an auxiliary/native head, or qualitative explanations.
- **Must-solve bottleneck:** Existing RGCL memory geometry confounds the binary hate decision with whole-video presentation/context nuisance. Prior MLLM routes were label-redundant, nearly orthogonal to vote correctness, too sparse in their correctable unit, or competed with the memory readout. The new route must use train-only MLLM information to reshape the exact embedding consumed by final kNN, prove dense conditional value beyond video labels and cheap controls, and avoid merely moving accuracy between heads.
- **Non-goals:** No segment/timestamp/span/localization task; no segment weighting; no stance/harm/evidence/target/mechanism field as a nuisance variable; no final-verdict teacher; no archive/schema prediction as the proposed mechanism; no concat/fusion/reranking/router/GroupDRO/test-time MLLM; no larger-model/data/epoch/ensemble story; and no claim that a cost/numerics preflight is a theoretical performance upper bound.
- **Constraints:** The only gold supervision is the parent video's binary label. No segment-level gold exists or may be assumed. Every MLLM/archive quantity is a weak or privileged train-only pseudo-signal with an explicit confidence, deterministic missing/low-confidence fallback, remove/shuffle/noise controls, and no teacher/environment artifact at validation or test inference. All eventual compute must run through SLURM in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no code and launches no job.
- **Success condition:** On at least two datasets, paired seeds 0/1/2, the final ordinary full-video kNN must improve both accuracy and macro-F1 by at least +0.030 absolute over the moving strongest same-protocol non-MLLM comparator; every paired delta must be positive, hierarchical paired bootstrap lower bounds must exceed zero, the four dataset×metric tests must survive Holm correction, and FULL must beat remove-MLLM plus a within-split permutation of the MLLM information with a significant same-direction removal cost. The method may stop only after these final gates—not after proposal readiness, preflight, or seed-0 evidence.

## Supervision audit

- `only_gold_supervision = video_level_binary_label`
- `segment_gold_exists = false`
- `segment_gold_used = false`
- Uniform frames, automatic clips, ASR, OCR, archive summaries, environment categories, and MLLM confidence are inputs or weak/privileged pseudo-signals, never gold annotations.
