# Research Proposal: ECM-RGCL — Label-Blind Failure-Mode Feasible Descent

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causally removable part of hateful-video RGCL and do not stop until the unchanged ordinary full-video train-memory kNN endpoint improves by at least `+0.030` absolute in both accuracy and macro-F1 on MHC-EN and MHC-ZH, paired seeds `0/1/2`, under the complete statistical and mechanism-attribution protocol.
- **Must-solve bottleneck:** The MLLM must diagnose dense whole-video failure mechanisms from a strict-OOF, label-blind prediction trace for every training video, and those modes must directly alter the optimizer of the shared final RGCL embedding. The route must reach errors outside SSR's sparse edge universe and EDCM's frozen top-64/two-swap universe, without becoming sample reweighting, a router, a renamed GroupDRO/JTT/EIIL method, or generic gradient surgery.
- **Non-goals:** No localization endpoint; no segment weighting or segment loss; no rationale/schema/summary concatenation; no teacher-selected key, score fusion, reranking, veto, auxiliary/native-head claim, test-time MLLM/mode/teacher, MoE/router, model/data/epoch/ensemble scaling, or rescue of frozen SSR/EDCM/CTE routes. SQ-RGCL is still at formal S0/S1 plan status and is not declared a performance failure here.
- **Constraints:** The only gold supervision that exists is the parent video's binary label. There is no segment, timestamp, span, stance, target, mechanism, rationale, or localization gold. MLLM modes are confidence-bearing train-only weak/privileged pseudo-signals. The teacher sees neither gold label nor any correctness/error/loss/true-class-margin indicator. Every train video is processed under the same strict-OOF rule; validation/test have no teacher, trace, mode, or extra head. All eventual computation must use SLURM in `HateVideo`, with at most 2 GPU / 16 CPU / 128 GB and no `--time`.
- **Success condition:** Relative to `max(historical strongest non-MLLM point, paired same-seed strongest non-MLLM mean)`, FULL gains `>=+0.030` accuracy and `>=+0.030` macro-F1 on both datasets; all three paired deltas are positive; hierarchical paired-bootstrap lower bounds exceed zero and four dataset-by-metric tests survive Holm correction. FULL significantly beats REMOVE and within-train MODE-SHUFFLE, and must beat margin-bin, standard GroupDRO, JTT, EIIL and generic gradient-surgery controls under matched capacity. Only final ordinary full-video kNN counts.

## Technical Gap

SSR and EDCM failed because their frozen edge/swap units could not touch enough errors; CTE stopped on a registered numerical implementation gate, not a representation-performance bound. Prior semantic filtering, fields, scores, summaries and competing heads were sparse, redundant, orthogonal to vote correctness, or displaced the memory endpoint. SQ remains an unexecuted presentation-invariance plan and is not evidence against semantic modes.

The remaining gap is not another sample score. It is a legal interface that turns a frozen teacher's model-specific, whole-video failure diagnosis into **separate retrieval-geometry objectives**, while preventing the diagnosis from seeing whether the model was right. GroupDRO/PG-DRO already optimize hard/soft groups; JTT already exploits errors; EIIL already infers environments; PCGrad/MGDA/CAGrad already manipulate conflicting gradients. Thus neither a mode posterior, a min-max loss, nor projection alone is novel.

Two routes were compared. Route A assigns teacher modes and uses PG-DRO/minimax weights; it is simple but collapses to probabilistic GroupDRO. Route B uses the same modes only to form balanced per-mode exact-vote-margin gradients, then projects the ordinary RGCL update into a base-preserving common-descent region. Route B is chosen because the MLLM changes the optimizer's feasible direction rather than weights, routes, samples or inference, and generic gradient-surgery baselines can isolate its narrow delta.

## Method Thesis and Contribution Focus

**Thesis:** A strict-OOF, label-blind MLLM failure-mode posterior can improve final RGCL only if its semantic modes expose reproducible conflicts among class-balanced exact-vote-margin gradients; an EMA-stabilized, base-preserving feasible-descent step can resolve those conflicts in the shared final embedding while requiring no mode at inference.

- **Dominant contribution:** MLLM-defined whole-video failure modes as executable constraints on the optimizer of final RGCL retrieval geometry.
- **Supporting contribution:** a correctness-firewalled strict-OOF trace/cache protocol that distinguishes semantic failure diagnosis from JTT/error upweighting.
- **Explicit non-contributions:** group discovery, DRO, gradient surgery, causal deconfounding, localization, a new classifier, or test-time reasoning.
- **New trainable components:** zero; only the existing `img_proj`, `text_proj`, fusion and MLP parameters move.

## Proposed Method

### System overview

```text
five strict train OOF models
  -> every held-out train video: whole video + label-blind OOF decision trace
  -> frozen MLLM: six-way failure-mode posterior q_i and confidence r_i
  -> freeze train-ID-only cache; join video label only after cache closure
  -> current full-bank exact-vote-margin risk per semantic mode
  -> base-preserving mode common-descent projection of the RGCL gradient
  -> same shared embedding and refreshed train bank

validation/test full video -> shared embedding -> ordinary repository kNN
                                      no trace / mode / teacher / extra head
```

### Correctness-firewalled strict-OOF teacher input

Use the frozen five folds already established for SSR. Model `f^{-k}` is trained without fold `k`; it produces for each held-out train video a trace consisting only of:

1. deterministic uniform whole-video frames plus full-video transcript/OCR with timestamps, segment IDs and spans stripped;
2. the OOF model's predicted decision and calibrated confidence band;
3. its visual-only and language-only predicted decisions/confidence bands, obtained from the same frozen OOF checkpoint;
4. label-blind geometric summaries: prediction entropy and unlabeled neighbour-similarity dispersion.

The teacher never receives the video gold label, correctness bit, supervised loss, true-class margin, neighbour label, error rank, fold/seed, selection flag or validation/test record. Crucially, **all train videos**, correct or incorrect, are sent through the same payload template; presence in the cache cannot reveal error status. The OOF error bit is joined only in offline relevance evaluation and is never a teacher input, mode gate, curriculum weight or training sample selector.

### Mode ontology and weak record

The ontology is fixed before calls:

1. `presentation_context_inversion`: quotation/reportage/counter-speech/satire or implicit endorsement can invert a surface reading;
2. `target_binding`: the model may bind an utterance/action to the wrong speaker, target or affected entity;
3. `modality_conflict`: visual and language evidence support incompatible moderation interpretations;
4. `surface_shortcut`: an isolated lexical/visual cue appears to dominate integrated context;
5. `evidence_dilution`: decision-relevant evidence is weak relative to whole-video irrelevant content, without locating it;
6. `undiagnosed`: none is supported or the trace is ambiguous.

The strict output is only six probabilities, confidence in `{0,.25,.5,.75,1}`, parse status and provenance hashes. It contains no verdict, rationale, identity, span, timestamp, segment, target, stance or mechanism description. Two prompts × two presentation orders produce four records. Canonical posterior `q_i` is their mean; reliability `r_i` is minimum confidence times one minus normalized mean pairwise JS. Parse failure, modal-category agreement `<.75`, JS `>.10`, or `r_i<.5` maps to `undiagnosed` and exact REMOVE fallback.

Dense coverage is binding: valid active coverage `>=90%`; each actionable mode has effective mass `>=30` overall and `>=8` in each video class per dataset; class-conditional Kish ESS is `>=8`; no actionable mode exceeds 60% of reliable mass. `undiagnosed` is monitored but never optimized as a failure mode. These are weak pseudo-signals, not annotations.

### Exact-vote mode risks

At each epoch-bank refresh, freeze current top-20 ranks for differentiation. For query `i`, let `s_ij` be cosine similarity, `t_ij=+1` for equal video labels and `-1` otherwise, and `a_r=21-r`. Define the repository-facing differentiable deficit

`ell_i^mem = softplus(sum_{j in N20(i)} a_rank(j) max(0,-t_ij s_ij) - sum_{j in N20(i)} a_rank(j) max(0,t_ij s_ij))`.

This uses only parent-video labels. Rank is detached for the step; the query and every key still co-move over training because the shared encoder is used for all videos and the bank refreshes each epoch. No old top-64/action universe is fixed.

For mode `m`, form a class-balanced risk rather than a sample-weighted scalar loss:

`R_m = .5 * [sum_{i:y=0} r_i q_im ell_i / sum r_i q_im + sum_{i:y=1} r_i q_im ell_i / sum r_i q_im]`.

`q` is used only to estimate five separate semantic objective gradients, never to multiply the ordinary RGCL loss, choose a sample, or route an example. Define `g_0=grad L_RGCL` and EMA-normalized `g_m=EMA(grad R_m)`.

### Executable constraint optimizer

On a deterministic mode macro-batch, solve one small dual quadratic program over five mode coefficients, equivalent to the primal update

`d* = argmin_d 0.5||d-g_0||^2`

subject to `g_0^T d >= beta ||g_0||^2` and `g_m^T d >= 0` for every supported actionable mode. Use one frozen `beta`; norm clipping and AdamW moments remain the baseline's. If finite precision, missing support or infeasibility prevents certification, use `d=g_0` exactly and log the fallback. No slack is outcome-tuned. EMA is frozen because instantaneous stochastic projection has known convergence problems.

This edits gradients immediately between the existing `backward()` and `optimizer.step()` in `src/run_rac.py`; it adds no forward module. The scientific claim is not that the projection is new, but that correctness-firewalled MLLM failure mechanisms produce final-retrieval constraints that generic groups/gradients do not.

## Training and Inference

The teacher cache is created once from seed-0 strict-OOF traces and frozen across student seeds. It contains train IDs only. Student training preserves the dataset-specific strongest RGCL recipe, data order, epochs, bank refresh, checkpoint rule, top-20 vote, optimizer hyperparameters and parameter count. One global `beta` is selected before teacher calls by the zero-call screen and shared across datasets.

Validation/test load neither OOF trace nor mode cache; they run full videos through the shared encoder and the original ordinary kNN. The output layer/native head is not a claimed endpoint.

## Prior-Art Separation and Binding Controls

- `REMOVE`: unmodified strongest non-MLLM RGCL.
- `MARGIN-BIN-PROJECT`: same projection with deterministic label×OOF-margin bins.
- `RANDOM-MODE-PROJECT`: matched posterior mass/entropy/confidence/ESS.
- `MODE-SHUFFLE`: derange complete `q/r` records within train-fold `video-label × OOF-prediction × correctness × confidence-bin` cells; deterministic adjacent-bin merging preserves JTT/difficulty information while destroying semantics.
- `GROUPDRO-MODE` and `PGDRO-MODE`: standard hard-argmax GroupDRO and soft probabilistic GroupDRO on the same mode records.
- `JTT`: standard two-stage OOF-error upweighting with matched steps.
- `EIIL+GROUPDRO`: reference-model environment inference followed by GroupDRO.
- `PCGRAD-MODE` and `CAGRAD-MODE`: generic gradient manipulation on the identical `R_m` objectives.

All arms share initialization, examples, steps, bank refresh, parameter count and gradient clipping; auxiliary-gradient norm and backward-pass budget are reported. FULL must beat these rather than claim their primitives.

## Three Claim-Driven Validation Blocks

### Block 1 — ECM-0 zero-new-call capacity and legality screen

Use only existing strict-OOF predictions/embeddings and fixed deterministic residual-signature clusters as proxy modes. Before any teacher call, verify evaluator parity, QP scalar/vector parity, mode support, finite gradients, fallback rate, memory/time, and train five-fold OOF `REMOVE`, proxy-ECM, margin-bin, random, GroupDRO, JTT and EIIL controls.

The frozen cost gate requires proxy-ECM to improve concatenated actual OOF accuracy and macro-F1 by `>=.050` on **both** datasets, with all fold signs positive, while not losing to the strongest robust-learning control; constraint fallback must be `<=5%` and each proxy mode must meet the same support gates. This is a bounded empirical capacity/cost screen, not an upper bound and not MLLM evidence. Failure stops ECM without teacher spend.

### Block 2 — ECM-1 teacher legality, density and conditional value

After ECM-0 GO, freeze a class×prediction×confidence-stratified, label-blinded 128-video/dataset pilot before calls. If four-call graph closure or power is infeasible, stop. Then require schema/provenance and dense-coverage gates plus, on outer-held folds, adding `q/r` to label, predicted class, confidence/margin, modality decisions, embedding cluster and loss/difficulty controls improves OOF-error AUC by `>=.030` and NLL, every fold delta positive, anchor-bootstrap lower bound `>0`. MODE-SHUFFLE must be null. A label-prediction audit (`q -> y`) is reported; label predictiveness does not prove value.

Only pilot GO permits identical extraction for every remaining train video. No validation/test call exists. A frozen one-step gradient gate then requires FULL's worst-mode predicted margin improvement to exceed MODE-SHUFFLE, MARGIN-BIN, PGDRO, PCGrad and CAGrad on both datasets with adjusted lower bounds above zero.

### Block 3 — ECM-2 seed-0 and final ordinary-kNN proof

Seed 0 must improve validation ordinary-kNN accuracy and macro-F1 by `>=.010` over every binding control on both datasets, reduce worst actionable-mode exact-vote deficit, correct errors outside SSR/EDCM unions, and degrade monotonically under posterior corruption `{.25,.50,.75,1}`. Any dataset/metric/control failure is terminal; do not expand modes, teacher or epochs.

Final runs use MHC-EN/ZH × seeds 0/1/2. Per metric, FULL must exceed the moving comparator by `>=.030`; all paired signs are positive; 10,000 hierarchical paired bootstraps resample seeds then shared video IDs, lower bounds exceed zero, and four dataset×metric p-values pass Holm FWER `.05`. FULL-minus-REMOVE and FULL-minus-MODE-SHUFFLE use the same inference. Only this block can satisfy the project objective.

## Failure Modes and Diagnostics

- Teacher modes predict only video label or OOF difficulty -> conditional gate/control failure; STOP.
- Modes collapse to presentation/topic environments -> merge conceptually with SQ; ECM novelty fails.
- Projection gain is matched by GroupDRO/JTT/EIIL/PCGrad/CAGrad -> ECM is prior-art repackaging; STOP.
- Sparse/label-pure mode, low confidence, parse failure -> exact REMOVE fallback and route-level density failure if above limit.
- Native head improves but ordinary kNN does not -> failure.
- No segment/timestamp/span/localization gold exists or is used; evidence dilution is a whole-video mode and never identifies a segment.
- SQ remains formal S0/S1 until its own execution decides otherwise; ECM documents do not pre-empt that result.

## Novelty and Elegance Argument

GroupDRO/PG-DRO optimize groups, JTT exploits errors, EIIL infers environments, DISC discovers concepts, and PCGrad/MGDA/CAGrad manipulate gradients. ECM's narrow candidate delta is the end-to-end combination of: uniform all-train strict-OOF decision traces; a teacher correctness firewall; bounded whole-video failure-mechanism posteriors; class-balanced exact-vote-margin mode objectives; and a base-preserving feasible-descent edit of the same embedding optimizer consumed by unchanged kNN. Remove any element and the method becomes an existing baseline or loses endpoint relevance. No new trainable part or inference path is added.

## Experiment Handoff and Compute

- **Must prove:** legality/no correctness leak; dense semantic coverage; conditional error information beyond label/difficulty; conflict not matched by prior-art controls; final two-dataset +3/+3.
- **Highest risks:** semantics collapse to OOF error/margin; mode cells are sparse; QP is generic surgery; final average metrics trade off against worst-mode improvement.
- **Estimate:** ECM-0 about 20--60 A100-hours for ten OOF folds with serial arms; teacher pilot <=1,024 invocations, full cache <=4,512 invocations; seed-0 20--50 GPU-hours; final 80--180 GPU-hours depending on measured arm cost. Human/gold annotation cost is zero. A vectorized SLURM microbenchmark must replace these estimates before execution.
- **Timeline:** 1 day implementation/audit; 1--3 days ECM-0; 1 day pilot/extraction if unlocked; 1--3 days seed-0; 3--7 days final. No code or job is authorized by this refinement document.
