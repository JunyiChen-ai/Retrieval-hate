# CTE-RGCL grounding evidence

## Authoritative target and supervision

- `TARGET_LOOP.md`, `TARGET_STATE.json`, and `TARGET_FINDINGS.md` freeze the real endpoint: ordinary full-video train-memory kNN, at least two datasets and paired seeds 0/1/2, and at least +0.030 absolute in both accuracy and macro-F1 over the moving strongest non-MLLM comparator.
- The same files state that the only available gold is the parent-video binary label. No segment, timestamp, span, localization, stance, target, mechanism, or rationale gold exists.
- `research-wiki/TARGET_GATE0_ITER3_LITERATURE.md` and the verbatim Iteration-3 review in `TARGET_REVIEW_RAW.md` select CTE as the only first-run route and prohibit concat, score fusion, segment weighting, teacher keys, test-time MLLM/reranking, routers/MoE, scaling, and SSR/EDCM retuning.

## Negative evidence that CTE must not misstate

- SSR's all-candidate optimistic train-OOF screen touched only 2/7 MHC and 3/15 MHC-ZH MI/SC errors and failed all +0.05/+0.05 cells. It rules out SSR's sparse one-neighbour candidate universe, not learned representation motion.
- EDCM's exact top-64, at-most-two-swap screen reached 15 MHC and 22 MHC-ZH errors, with +0.0273/+0.0394 and +0.0380/+0.0444 accuracy/macro-F1. It rules out that frozen action space, not shared query/key motion outside the old top-64.
- `refine-logs/EXPERIMENT_RESULTS.md` and `refine-logs/edcm/EXPERIMENT_RESULTS.md` are terminal negative evidence. CTE must not reuse their relation universe, keys, swap operation, or loss.
- P3/P11 show that segment salience and density distillation can be absorbed or label-redundant. CTE therefore has no segment field, loss, weight, endpoint, or pseudo-label.

## Code facts that constrain the design

- `src/model/classifier.py` projects visual and language features, L2-normalizes both, combines them by the configured fusion (the strongest route uses multiplicative `align`), and returns the pre-output embedding used for retrieval.
- `src/model/evaluate_rac.py` rebuilds the train bank from the selected model and evaluates full-video queries with the repository's ordinary FAISS/kNN vote.
- `src/run_rac.py` and `src/utils/retrieval.py` already refresh the detached train embedding bank at epoch boundaries (or every step when requested) using the same model as the query encoder. CTE must retain one shared encoder and full-video keys; it may add a loss and a neutral-path calculation but no teacher key or second retrieval encoder.
- A literal absent modality is unsafe in `align`: a zero projected modality makes the multiplicative fused representation degenerate. CTE must never pass a zero vector, blank string, padding-only text, black frame, or mean-pixel image as the student's neutralized modality.

## Closest prior and narrow defensible delta

- TextTeacher covers train-time semantic teachers removed at inference.
- EmbedDistill and geometric KD cover retrieval/representation geometry distillation.
- CGO covers harmful-video modality intervention and gradient control.
- RAMF and other recent hateful-video work cover counter-reasoning and semantic fusion.

The defensible delta is therefore only: **a label-blind train-only MLLM supplies ordinal whole-modality counterfactual relations, and those relations supervise the local response of the exact full-bank true-class RGCL retrieval margin under a shared query/key encoder, while final inference remains unchanged full-video kNN.**
