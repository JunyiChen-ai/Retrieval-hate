# TARGET LOOP — MLLM × RGCL substantial final-performance loop

## Fixed objective and non-negotiable stop condition

The loop exists to make an MLLM a **meaningful and novel causal component** of the hateful-video RGCL method and to obtain a **substantial final classification improvement**, not merely a localization, audit, encoder, or qualitative contribution.

Success is proved only when one frozen candidate satisfies all of the following at once:

1. Under the same split, label space, preprocessing, checkpoint selection, retrieval rule, and test protocol as the strongest non-MLLM RGCL comparator, **accuracy and macro-F1 each improve by at least +0.030 absolute**.
2. The joint result holds on **at least two datasets** and **at least three paired seeds** (initial fixed seeds 0/1/2).
3. For each claimed dataset and both primary metrics: all three paired seed deltas are positive; mean±std is reported; a hierarchical paired bootstrap over seeds and test examples has a 95% lower bound above zero; the four primary dataset×metric tests are Holm-corrected at familywise α=0.05.
4. Mechanism attribution is mandatory: the full method must beat both a `remove-MLLM` control and a `shuffle/permutation-of-MLLM-information` control. The intended interface must lose performance when removed, with same-direction paired effect and 95% CI excluding zero; both accuracy and macro-F1 removal costs are reported.
5. Novelty must survive literature and external-review gates. A renamed version of an existing route is not a new hypothesis.
6. The gain may not primarily come from a larger model, materially more data, extra epochs/steps, heavy/cross-seed ensemble, post-processing stack, test-time tuning, protocol changes, leakage, or a changed label space.
7. The supervision contract below is a hard invariant. Violating it invalidates a result even if the metric target is reached.

If a future same-seed baseline rerun exceeds the historical reference, the target moves upward: per metric the binding bar is `max(historical strongest point, paired baseline mean) + 0.030`. Existing standards requiring significance, removability, both protocols, or guard-backed reproduction are retained whenever they are stricter.

## Hard supervision contract

- The **only gold supervision that exists or may be assumed is the video-level binary label**. There is no segment-level gold annotation in this project. No design, loss, gate, calibration, evaluation, or written claim may silently assume segment boundaries, segment labels, stance labels, target labels, mechanism labels, rationales, or localization spans are gold.
- Every MLLM-produced segment score, stance, target, mechanism, rationale, localization, or structured semantic field is a **weak/privileged pseudo-signal**. It must never be called ground truth, dense supervision, annotation, or oracle evidence.
- At test time, an MLLM pseudo-signal cannot be used, tuned, or evaluated **as an annotation**. If a frozen candidate generates one at inference as an ordinary label-blind model feature, it remains a weak pseudo-signal, must follow the identical frozen protocol for every sample, and may not access test labels or test-derived thresholds.
- Every use of an MLLM pseudo-signal must include: (i) an explicit confidence or reliability variable; (ii) a deterministic missing/parse-failure/low-confidence fallback that reduces to an available non-MLLM path; (iii) `remove-MLLM`; (iv) within-split `shuffle/permutation`; and (v) calibrated noise/corruption controls. Report pseudo-signal coverage, confidence distribution, missing/fallback rate, corruption sensitivity, and final accuracy/macro-F1 for every control.
- A method is not mechanistically validated merely because pseudo-signals correlate with the video label. It must outperform video-label-only supervision and its remove/shuffle/noise controls under the same protocol; otherwise the signal is classified as redundant or non-causal.

## Iteration 0 — initialization and recovery audit (2026-07-10)

### Environment routing

- `whoami = jehc223`; `$USER = jehc223`.
- No prior `TARGET_STATE.json`, `TARGET_LOOP.md`, `TARGET_FINDINGS.md`, or `TARGET_REVIEW_RAW.md` existed: this is a fresh persistent-loop initialization, not a resume.
- Execution policy is permanently recorded as **`slurm_only`**. Any training, evaluation, embedding extraction, GPU inference, or other compute must be prepared/submitted with existing project Slurm style and monitored through `squeue`/`sacct`; never run such work directly on the login node. This initialization performed only read-only audit plus these four document writes; no job was submitted.
- Repository HEAD at initialization: `a1b1922bc970bb831526b4d21c911380ec871248`.

### Exact baseline registry

The comparator is **the strongest documented non-MLLM RGCL configuration per dataset**, not a conveniently weak floor. All candidates must also rerun that exact comparator on the same three seeds. Historical points below are hard lower bounds.

| dataset | exact non-MLLM RGCL comparator | acc / macro-F1 | binding historical +3pt target | evidence / caveat |
|---|---|---:|---:|---|
| HateMM | frozen-CLIP RGCL (`RAC_video_CLIP`), warmup≥5 val-selected | **0.8279 / 0.8172** | **0.8579 / 0.8472** | val-selected epoch-24 Test_Retrieval test acc / macro-F1 at `slurm/logs/rgcl_HateMM_openai_clip-vit-large-patch14-336_HF_1035814.trainlog:257,259` (n=215). (corrected in place 2026-07-12, erratum 66012e9; original mis-transcription 0.8732 = Val_Retrieval ROC-AUC / 0.8686 no provenance; the superseded refs were `research-wiki/EVAL_localization_hatemm.md` §3 and `PAPER_MASTER_TABLES.md` T1.1.) |
| MHC-EN | CLIP-RGCL `full`, λseg=0.5, warmup≥5 val-selected | **0.7888 / 0.7262** | **0.8188 / 0.7562** | `ITERATION_LOG.md`, Phase 3 iter1, job 12129 ep25. Stronger than plain λ=0 0.7826/0.7113. The 0.8199/0.7748 human memory-edit demonstration is not the standard comparator because its intervention was MLLM-assisted/manual, but remains an additional guardrail. |
| MHC-ZH | CLIP-RGCL `milmax`, λseg=0.5, warmup≥5 val-selected | **0.8255 / 0.7875** | **0.8555 / 0.8175** | `ITERATION_LOG.md`, Phase 3 iter2, job 12135 ep28; stronger on both metrics than the plain floor 0.8054/0.7706 and seed-0 consensus 0.8188/0.7864. |
| ImpliHateVid | frozen-CLIP RGCL, warmup≥5 val-selected | **0.9102 / 0.9101** | **0.9402 / 0.9401** | `ITERATION_LOG.md` encoder ablation / selected-run record. |

The first two-dataset claim should preferentially use MHC-EN and MHC-ZH because they expose complementary speech-heavy English and visual/on-screen-text-heavy Chinese failure modes. HateMM and ImpliHateVid remain valid confirmation datasets, but their higher ceilings make +3 points harder. This preference does not relax the target.

### Target status

- Baseline: exact registry above.
- Current classification bests with an MLLM are insufficient: frozen Qwen+RGCL HateMM 0.8698/0.8606; MHC-EN 0.7888/0.7378; historical LoRA-Qwen+RGCL MHC-ZH 0.8322/0.8023. None forms a two-dataset, three-seed, +3/+3, statistically supported result with causal ablations.
- Current gap: at least two datasets still need **both** +0.030 acc and +0.030 macro-F1 above the moving strongest-non-MLLM bar.
- Decision: **advance to Gate 0 literature/novelty; do not stop**.

## Anti-repeat registry — all prior MLLM routes and mechanism findings

The authoritative synthesis is `research-wiki/CAMPAIGN_mllm_method_role.md`, checked against `TERMINUS_mllm_campaign_DRAFT.md`, `PAPER_MASTER_TABLES.md` T4, the individual EXP documents, `src/run_rac.py`, `src/model/loss.py`, `src/model/evaluate_rac.py`, `src/utils/consensus.py`, and `scripts/role3/`. The route-family count below follows the 13-row campaign table; role-3 is logged separately as a pre-campaign ancestor.

| route | MLLM integration already tried | result and measured failure mechanism | anti-repeat rule |
|---|---|---|---|
| auto-repair | MLLM semantic verdict + embedding-outlier AND rule deletes suspected memory noise | **FAIL raw accuracy**: C−A=0 over 4 EN seeds. Semantic vote can veto harmful over-deletion but the AND geometry cannot select semantic contradictions that are not embedding outliers. | Do not re-propose threshold retuning of the same two-vote deletion rule. A new memory-edit hypothesis must alter the causal observability or learning objective, not its cutoff. |
| P1 | label-free MLLM harmfulness count estimates a temporal test prior and recalibrates kNN threshold | **FAIL**: prior error 0.22 EN/0.18 ZH; corrected EN 0.48 < static 0.63. MLLM FPR itself drifts across eras, so the estimator is biased precisely at deployment shift. | No new classify-and-count prior variant unless it identifies and corrects class-conditional era drift without test labels. |
| P2 | 7B pairwise comparable/incomparable neighbor filtering before re-vote | **FAIL**: Δacc −0.002 EN/−0.020 ZH; 83%/70% over-drop. Topical comparability is almost independent of vote correctness. | Do not repackage semantic neighbor filtering/reranking without a train-only demonstration that the score predicts **label-match conditional on query**. |
| P2b/P2c | 7B/32B/72B judge × archive/transcript × prompt flip, train-side selectivity calibration | **FAIL before test**: best EN selectivity lift +2.7pt vs +10 bar; all ZH configs negative. Scale improved calibration/drop rate, not selectivity. | Scaling or prompt variants of comparability judging are exhausted; prohibited as primary strategy. |
| P3 | per-segment MLLM hate-density soft weighting of visual pooling (EN/ZH/HateMM) | **FAIL classification**: EN probe −0.0055; ZH final +0.0088; HateMM clean probe +0.0108 but trained ΔF1 +0.0004. Learned align-fusion absorbs the input reweighting; localized evidence is not equivalent to better global separation. | No more static scalar segment reweighting into the same align-fusion head. A successor must change information flow/credit assignment and distinguish causal evidence from correlated salience. |
| P4 | auxiliary heads distil MLLM schema fields (explicitness, modality, mechanism, target) | **within noise** despite fields being decodable and label-informative. The fields are redundant with direct video-label supervision. | No auxiliary-task reprise using label-subset semantic fields. New supervision must carry information not recoverable from the binary label and prove conditional information gain. |
| P5 | sanitized transcript counterfactual twin as an extra hard negative | **FAIL premise**: flip only 0.503 EN/0.337 ZH; diagnostic hurts EN −0.027. Same real visuals keep the twin too close (cos≈0.73), so repulsion fights visual evidence. | Do not generate same-visual text-only twins or rely on self-verdict quality. Counterfactuals require independently verified causal validity and modality-consistent intervention. |
| P6 | MLLM window scorer for span-free localization | **PASS localization only**: wv-AUC 0.5435 vs memory 0.5140; later amplified by P10-b. It does not improve final acc/macro-F1. | Preserve as evidence asset, but it cannot satisfy the classification target by itself. Any reuse must explain a new train-time causal bridge and beat P3/P11 controls. |
| P7 | rank/veto score-level fusion of kNN vote with MLLM binary/density channel | **FAIL before test**: corr +0.21…+0.51; MLLM channel AUC 0.54–0.69 vs floor 0.81–0.86; net error correction −0.10…−0.38. The channels are correlated and the MLLM one is weaker. | No linear/rank/veto late fusion unless train-only evidence shows conditional error complementarity and positive net correction. |
| P8/P8b/P8c | MLLM semantic compression / text or visual summaries, including Chinese | **FAIL all datasets**: strongest EN probe passed but training lost; ZH/HateMM probes closed; Chinese summary was worst. The frozen English-centric CLIP text tower truncates Chinese byte fragments and bottlenecks summary content. | Do not feed more generated summaries through the same frozen CLIP text tower. A new route must change representation/interface, not prompt wording. |
| P9 | LoRA-SFT whole Qwen LMM + native MLP head, plus kNN readout of SFT embeddings | **FAIL substantial target**: native head only +0.6 EN/+1.0 ZH vs protocol-matched floors; kNN readout −2.7/−2.2/−4.7 EN/ZH/HateMM. The head displaces rather than enhances the memory pillar. | Do not retry rgcl-OFF LMM SFT or claim gain vs weaker frozen floor. Same-seed LoRA-RGCL is the comparator. |
| P9b | rgcl-ON while LoRA-SFTing LMM; native head and kNN readouts | **FAIL**: D3-knn −1.0 EN/−1.5 ZH vs floor, 0/12 cells win. RGCL moves accuracy from head to memory (+knn/−head) without net gain. | No loss-weight sweep of the same head↔memory objective. A new method must remove the competition for representational capacity, with a falsifiable prediction beyond redistribution. |
| P10/P10-b/P10-c | stronger scorer, coarse×fine A-fuse, 7B→32B→72B, Qwen3 generation | **localization modest-plus only**: HateClipSeg wv-AUC 0.5755 < 0.60; open-source calibration ceilings 0.5932/0.5913/0.5866 < 0.616. Scale is the only lever here and is explicitly forbidden as the primary classification strategy. | Do not reaggregate existing scores, scale the scorer, or re-spend HateClipSeg test. A-fuse can be a fixed diagnostic/control, not the new scientific hypothesis. |
| P11 | distil 72B A-fuse segment scores into weakly supervised segment head | **probe-killed**: same-operator A-fuse teacher−MIL +0.0359 CI includes zero; raw gaps nonsignificant. Coarse×fine aggregation, not better per-segment labels, creates most of the apparent edge; video-label MIL already contains the teachable signal. | No teacher-student localization training from the same density scores unless a train-only probe first beats same-operator video-label MIL significantly and by the required effect. |
| pre-route role-3 | kNN margin gates deferred examples to an MLLM arbiter using query/archive/neighbors | **FAIL**: one-way over-flag ratchet; on deferred examples the generic moderator prior is worse than calibrated kNN. Frames add ≈0. | No confidence-gated absolute verdict arbiter. A successor must not ask the MLLM to replace the decision at the break-even boundary. |

### Cross-route mechanism constraints

1. Semantic competence is repeatedly **orthogonal to the required decision variable** (comparability ≠ vote correctness; salience ≠ separability) or **redundant with direct labels** (schema distillation).
2. A frozen probe pass is necessary but not sufficient; P3-HateMM and P8-EN show that the learned head can wash out an apparently useful input transform.
3. The native LMM head and retrieval memory compete for capacity. The next hypothesis must predict and measure synergy, not merely shift accuracy between readouts.
4. Any proposed signal must show **conditional error complementarity on train/validation before test**. Marginal semantic plausibility is not evidence.
5. The MLLM must affect the final decision through a causal, removable interface that cannot be replicated by label-only supervision, a shuffled MLLM channel, or the non-MLLM backbone alone.
6. No hypothesis may rely on nonexistent segment-level gold. Segment/stance/target/mechanism outputs from an MLLM are weak or privileged pseudo-signals and must have confidence-aware routing, deterministic missing fallback, and remove/shuffle/noise controls before promotion.

## Iteration 0 cards

### Hypothesis card

- Active slot: **A**, definition pending Gate 0 literature + novelty search.
- Claim: not yet registered.
- Mechanism: not yet registered.
- Falsifiable prediction: must include train-only conditional information gain and a non-redistribution readout before any full experiment.
- Minimal intervention: must be smaller than a full-scale sweep and must include confidence/missing fallback plus remove/shuffle/noise controls; video-level binary labels are the only gold supervision.

### Experiment card

- Intervention: none in iteration 0; state and evidence were initialized.
- Controls: exact baseline registry, anti-repeat registry, and hard supervision contract frozen above.
- Metrics: no new measurements.
- Uncertainty: baseline point-estimate tensions are explicitly recorded; paired three-seed reruns are mandatory before promotion.

### Decision

**Advance to Gate 0.** Generate exactly three genuinely distinct scientific hypotheses only after literature/novelty review; reject any that maps to an anti-repeat row or depends primarily on forbidden scaling.

### Why scientific

The loop's target is a causal method contribution: the MLLM must add conditionally useful information to RGCL, its removal/permutation must destroy the gain, and the effect must replicate across datasets and seeds. Engineering-only ways of increasing capacity, data, runtime, ensembling, or favorable evaluation are excluded by construction.

## Iteration 1 — Gate 1 method refinement (2026-07-10)

- **Selected route:** SSR-MemRGCL, the sole Gate-0 first-run candidate.
- **Independent reviewer:** `/root/ssr_method_refine/ssr_reviewer`, one continuous reviewer thread, four review/revision rounds.
- **Score path:** `6.6 REVISE → 7.4 REVISE → 8.1 REVISE → 9.03 READY`.
- **Gate-1 verdict:** **READY for experiment handoff**, with preserved anchor and no drift. This is not target completion and contains no new experimental result.
- **Frozen mechanism:** paired-seed directed real hard pairs; frozen label-blind MLLM supplies only reliability-filtered stance–target–mechanism weak pseudo-relations; video-level binary labels alone sign MI+/SC− constraints; one parameter-free ranking term shapes the same embedding used by final train-memory kNN; no MLLM/relation artifact at validation/test inference.
- **Hard supervision invariant:** no segment-level gold exists. All semantic fields are weak train-only pseudo-signals; low confidence/missing/parse failure maps to no edge/non-MLLM fallback.
- **Next gate:** implement and run B0/B1 only. Strict five-fold train OOF diagnostics must prove a common EN/ZH family, ≥80 accepted records per dataset×family with 95% Wilson precision lower bound ≥0.80, significant conditional information, ≥+0.05 oracle headroom in both accuracy and macro-F1, and exact canonical semantic-null feasibility. Any failure stops SSR without model/prompt/architecture scaling.
- **Canonical documents:** `refine-logs/FINAL_PROPOSAL.md`, `refine-logs/REVIEW_SUMMARY.md`, `refine-logs/REFINEMENT_REPORT.md`; every raw review is in `refine-logs/round-N-review.md`.

## Iteration 1 — B0 implementation integrity (2026-07-10)

- Two independent read-only code reviews were completed before execution; leakage, strict parsing, resume provenance, fold-local integrity, and missing/no-edge issues were repaired.
- SLURM jobs `12686` (config-hash bootstrap), `12687` (frozen folds/config), and `12688` (static sanity) completed successfully.
- Static sanity is `GO`: only video-level binary labels are gold; no segment schema/gold exists; MLLM payload forbidden-key count is zero; strict JSON and BA canonicalization pass.
- This does not unlock B2/B3. Ten strict train-only OOF heads, relation smoke/full extraction, human Wilson audit, conditional information, dual-metric oracle headroom, and exact shuffle remain pending.

## Iteration 1 — SSR-MemRGCL terminal preflight (2026-07-10)

- **Execution:** strict OOF jobs `12691–12700`, mining `12701–12702`, formal oracle upper bound `12704`, and verifier `12705` all completed successfully through SLURM.
- **Decision:** `B1_DECISION=STOP`; B2/B3 remain locked. The target is still not met.
- **Why this is conclusive:** job 12704 optimistically treats every selected pre-MLLM candidate as an accepted reliable MLLM relation. Any real accepted set is a subset, so it cannot touch more exact event-positive OOF errors or produce larger oracle gains.
- **Upper bounds:** MHC MI `+0.0036 acc / +0.0048 mF1` (2 touched), MHC SC `+0.0128 / +0.0176` (7); MHC_zh MI `+0.0052 / +0.0065` (3), MHC_zh SC `+0.0259 / +0.0307` (15). Every cell is below `+0.05/+0.05`; the common-family set is empty.
- **Skipped by gate:** 7B smoke/full extraction, 2+1 human audit, conditional permutations, exact shuffle, B2, and B3. This saves approximately 8–20 A100 GPU-hours for relation extraction, 12–20 person-hours, and downstream CPU/GPU campaigns.
- **Anti-repeat:** do not revisit prompt/model/reliability/loss tuning on the same one-neighbour `C_SC/C_MI` and `Y_SC/Y_MI` universe. A successor must first pass a strict train-only, video-label-only two-dataset dual-metric oracle ceiling before any MLLM call, by changing the causal correctable unit rather than semantic assignment quality. Segment gold remains forbidden.

## Iteration 2 — EDCM-RGCL Gate-1 refinement (2026-07-10)

- **Pivot evidence:** SSR stopped because its optimistic selected-arc universe touched only 2/7 EN and 3/15 ZH unique MI/SC OOF errors. EDCM changes the correctable unit from sparse directed arcs to a per-training-video coalition signature that controls the whole full-video memory-list gradient.
- **Selected route:** `EDCM-RGCL — Dense Interventional Coalition Control of Retrieval Memory Geometry`.
- **Independent reviewer:** `/root/edcm_pivot_refine/edcm_reviewer`, one continuous reviewer thread, three rounds.
- **Score path:** `6.93 REVISE → 8.28 REVISE → 9.11 READY`; anchor preserved, drift warning NONE, no-segment-gold audit PASS in all rounds.
- **Frozen mechanism:** a label-blind frozen Qwen2.5-VL-7B teacher compares deterministic `V/S/O` coalitions for train videos only. Reliable ordinal preservation distributions form a six-dimensional necessity/synergy weak pseudo-signal. One zero-new-parameter listwise Memory-NCA term uses it to change the gradient of the same ordinary full-video embeddings/keys consumed by final kNN. Validation/test load no MLLM, OCR, coalition, signature or confidence artifact.
- **Dense-support correction:** cache coverage is insufficient. A1 requires broad all-OOF teacher-specific TV/`R` activity and positive equal-step `DeltaD` over A0-reachable errors relative to Label-only ListNCA. One teacher-semantic-free strength-matched modality/content proxy, plus remove/shuffle/noise, is binding.
- **Pre-MLLM A0:** before any teacher call, strict fold-local video-label-only top-64/two-swap reachability must show ≥80% all-video candidate support, at least `ceil(.05N)` reachable errors and ≥+.050 accuracy/+.050 macro-F1 frozen-geometry reachability on both MHC-EN and MHC-ZH. It is a cost screen, not a learned-method upper bound.
- **Supervision invariant:** the binary video label is the only gold. Uniform frames are whole-video inputs, not annotated segments; transcript/OCR are inputs; every coalition output is a weak train-only pseudo-signal. No segment-level gold exists or is assumed.
- **Status at refinement handoff:** Gate-1 method specification was READY for experiment planning, but the global target remained unmet and no EDCM result yet existed. The authorized next action was planning and A0 only; no MLLM call was authorized unless A0 passed. The execution outcome is recorded below.
- **Canonical documents:** `refine-logs/edcm/FINAL_PROPOSAL.md`, `REVIEW_SUMMARY.md`, `REFINEMENT_REPORT.md`; raw reviews and full refinements remain isolated under `refine-logs/edcm/` and do not overwrite SSR logs.

## Iteration 2 — EDCM A0 execution (2026-07-10)

- **Implementation integrity:** independent review closed three HIGH issues across two repair rounds: atomic non-overwrite, complete current-lineage/numeric decision verification, and authoritative per-query witness reconstruction. Final review was 0 HIGH / 0 CRITICAL. SLURM sanity jobs 12708--12709 froze and verified config/code hashes.
- **Reuse audit:** job 12710 returned `GO`. The SSR OOF candidates match the frozen comparator, train-only folds, source/data/output hashes, exact top-20 arithmetic cosine vote, and repository retrieval vote. OOF reuse was not the blocker.
- **MHC reachability:** job 12711 found support `202/549=0.3679`, 15 unique reachable errors (required 28), oracle `+0.0273 accuracy / +0.0394 macro-F1`; all geometry/headroom gates failed, provenance passed.
- **MHC-ZH reachability:** job 12712 found support `364/579=0.6287`, 22 unique reachable errors (required 29), oracle `+0.0380 accuracy / +0.0444 macro-F1`; all geometry/headroom gates failed, provenance passed.
- **Joint decision:** job 12713 independently rebuilt authoritative rankings and canonical witnesses and wrote `STOP`, `A1_unlocked=false`, `A2_A3_locked=true`. It records zero EDCM MLLM/OCR/teacher calls and no validation/test teacher artifact.
- **Supervision audit:** only video-level binary labels were used. K4 labels remain inherited parent-video labels, not segment labels; segment gold does not exist or enter any calculation. Validation/test source files were not read.
- **Route status:** EDCM-RGCL is stopped at its preregistered pre-MLLM cost screen. Do not tune prompts, teacher scale, EDCM loss, top-64 depth, two-swap budget, or thresholds on this route. A successor must change the video-level correctable unit and first pass a new two-dataset dual-metric oracle screen before any MLLM call.
- **Global status:** target still unmet and remains active. Canonical results are `refine-logs/edcm/EXPERIMENT_RESULTS.md`, `research-wiki/experiments/exp-edcm-a0.md`, and `artifacts/edcm/v1/A0_DECISION.json`.

## Iteration 3 — CTE-RGCL Gate-1 refinement (2026-07-11)

- **Selected route:** `CTE-RGCL — Withholding-Informed Tangent Supervision of Full-Bank Retrieval`, the sole Gate-0 first-run candidate.
- **Independent reviewer:** `/root/cte_method_refine/cte_reviewer`, one continuous reviewer over three review/revision rounds.
- **Score path:** `6.65 REVISE → 8.75 REVISE → 9.20 READY`; immutable anchor preserved, drift NONE, video-label-only supervision audit PASS, remaining blockers NONE.
- **Frozen mechanism:** a label-blind train-only MLLM compares the same whole train video under `full`, whole `visual-withheld`, and whole `language-withheld` conditions. It emits only confidence-bearing `preserve/weaken/reverse/unclear` weak relations. After a class-conditional two-radius transfer gate, one parameter-free interval cost supervises the supported response of the exact epoch-refreshed full-bank true-class retrieval margin. Query and keys use one shared encoder; validation/test use only full videos and the unchanged ordinary train-memory kNN.
- **Neutralization/OOD rule:** teacher uses typed whole-channel withholding, not blank/black/generated content. Student uses local paths toward frozen modality-specific train-video medoid IDs. Both anchor IDs and adjacent radii are frozen before teacher calls; joint projected/fused support or direction drift can only STOP, never trigger reselection. Zero vectors, blank strings, black frames, segment selection and localization metadata are forbidden.
- **Orientation rule:** teacher relation does not logically identify the gold-class sign. CTE-1 must prove reliability-weighted `preserve < weaken < reverse` degradation separately for `y=0` and `y=1` at both frozen radii with positive lower bounds. No pooled result or absolute teacher verdict may rescue a failed class.
- **Staged authorization:** method refinement only; no code or job was produced here. Next handoff is experiment planning, then a vectorized full-bank SLURM microbenchmark and independently audited **CTE-0 only**. CTE-0 is a strict nested train-OOF, zero-teacher bounded continuous cost/capacity screen, never a theoretical upper bound or MLLM evidence. If passed, its label-only arm raises the moving comparator.
- **Teacher lock:** CTE-1 remains locked unless CTE-0 passes both MHC-EN and MHC-ZH in accuracy and macro-F1 under its registered gates. CTE-1 is capped at 128 strict train videos per dataset (maximum 2,048 calls); extraction beyond the pilot, CTE-2, test and final seeds remain locked behind successive gates.
- **Controls:** exact REMOVE, assignment-free/teacher-mask-free multiview, cached strict-OOF label-only, modality-energy heuristic, strength-matched random, feasible whole-record SHUFFLE and two calibrated NOISE rates. All share the same anchor IDs, radii, support mask rule, encoder, bank refresh, optimizer steps and checkpoint budget.
- **Supervision invariant:** the parent-video binary label is the only gold. There is no segment, timestamp, span, localization, stance, target, mechanism or rationale gold; uniform frames and full-video ASR/OCR are only inputs. Teacher cache is train-ID-only and contains relation+confidence, never an annotation.
- **Global status:** **target still unmet and active.** READY means the method specification can be handed to planning; no CTE performance result exists. The loop cannot stop until final full-video kNN proves the two-dataset, seeds 0/1/2, `+0.030 accuracy/+0.030 macro-F1`, bootstrap/Holm and mechanism-removability gates.
- **Canonical files:** `refine-logs/cte/FINAL_PROPOSAL.md`, `REVIEW_SUMMARY.md`, `REFINEMENT_REPORT.md`, `score-history.md`; verbatim reviews are `round-1-review.md` through `round-3-review.md`.

## Iteration 3 — CTE C0 execution (2026-07-11)

- **Implementation integrity:** the experiment bridge implemented the registered C0/C1 interfaces, but execution advanced only through C0. An independent review initially found four HIGH issues (full numerical coverage, aggregate control matching, post-C1 freeze identity and fail-closed partition/provenance verification). All were fixed; re-review ended at `0 CRITICAL / 0 HIGH`. SLURM sanity job 12715 passed and freeze job 12716 bound config, implementation, review, folds, train caches, inherited-parent-label K4 caches and ten checkpoint fixtures.
- **C0 execution:** GPU jobs 12717 (MHC, 5m36s) and 12718 (MHC-ZH, 6m08s) completed with no NaN/OOM. Both were initially `JobHeldUser`, cleared automatically and were never manually released. CPU job 12719 independently required five folds and exactly 32 unique numerical cases per fold before recomputing every gate.
- **Supported geometry:** all folds selected adjacent pair `(a1,a2)=(0.20,0.30)`. Minimum joint video support was 0.9681 on MHC and 0.9438 on MHC-ZH; peak allocated CTE memory was only 0.0958/0.0963 GiB. Support, completeness, finite, stable-LSE, margin, gradient, norm and resource gates passed.
- **Binding numerical failure:** frozen T/cost tolerance was `2e-5`. MHC had max T error `8.0526e-5` and cost error `2.2231e-5`; MHC-ZH had `1.0099e-4` and `2.1911e-5`. Margin errors (`2.6889e-7`/`1.5718e-7`) and relative gradient errors (`7.1451e-7`/`3.7093e-7`) passed, but the independent dual-dataset verdict is `C0_DECISION=STOP`, `C1_unlocked=false`.
- **Stopped work:** no C1 nested probe/training, C2 pilot, MLLM, OCR or teacher-cache call was launched. There was no val/test endpoint and no teacher/view artifact. Only parent-video binary gold was used; K4 labels remained mechanical parent inheritance, not segment gold.
- **Interpretation:** this is a numerical/cost STOP for the exact preregistered FP32 tangent kernel, not an MLLM negative and not a theoretical impossibility result for shared representation learning. Do not relax tolerance, change kernel precision, anchor/radius selection or grid after observing it.
- **Global status:** the two-dataset, three-seed `+0.030 accuracy/+0.030 macro-F1` target remains unmet and active, but this frozen CTE route is terminal. Canonical records are `refine-logs/cte/EXPERIMENT_RESULTS.md`, `research-wiki/experiments/exp-cte-c0.md` and `artifacts/cte/v1/C0_DECISION.json`.

## Iteration 4 — SQ-RGCL Gate-1 refinement (2026-07-11)

- **Selected route:** `SQ-RGCL — Presentation-Crossed Exact-Vote-Exposed Ranking`, the Iteration-3 second reserve activated only after CTE's frozen C0 numerics STOP.
- **Independent reviewer:** canonical continuous thread `/root/sq_reviewer_replacement`, four rounds. The initially spawned `/root/sq_method_refine/sq_reviewer` repeatedly interrupted without output and was discarded; it contributes no score or raw review.
- **Score path:** `6.88 REVISE → 7.90 REVISE → 8.46 REVISE → 9.12 READY`. The immutable anchor was preserved verbatim; drift NONE; no-segment-gold and CTE-interpretation audits PASS; final scientific blockers NONE.
- **Frozen mechanism:** a label-blind train-only MLLM emits only a confidence-bearing six-way whole-video presentation posterior (`news/reportage`, `satire/skit`, `educational/explanatory`, `personal narrative/discussion`, `gaming/music/entertainment`, `other/unclear`). Stance, endorsement, harm, target, mechanism, explicitness, evidence, label/prediction and all segment/timestamp/span fields are forbidden nuisance inputs/outputs. One scalar crossed ranking loss makes a full-bank same-label/different-presentation memory outrank a different-label/same-presentation memory only when the latter has harmful signed contribution in the current exact repository top-20 vote. Rank>20 exposure is exactly zero; shared query/keys and the bank co-move across refreshes. Validation/test use unchanged ordinary full-video kNN and no posterior/teacher artifact.
- **P2/P4/prior-art separation:** P0 must show the posterior conditionally enriches actual wrong-class top-20 attraction, rather than generic comparability. FULL must beat LABEL_ONLY, within-class SHUFFLE, strength-matched RANDOM, BASE-CLUSTER, CHEAP-FORMAT, same-posterior ENV-SUPCON, Yang-style decorrelation and P4-style posterior prediction. The claim remains narrowly `MLLM-defined presentation crossing × exact-vote-exposed RGCL ranking`, not causal deconfounding, first invariance or general quotient theory.
- **Staged authorization:** refinement made no code change, launched no job and made zero new teacher calls. Next is an independent experiment plan/implementation audit, then P0 and learned strict-OOF **SQ-0 only**. SQ-0 must achieve `>=+0.050 accuracy` and `>=+0.050 macro-F1` on both MHC-EN and MHC-ZH and beat label-only/shuffle/random controls before any new teacher call. Archive summary provenance failure or any SQ-0 dataset failure is terminal for this frozen route.
- **Teacher lock:** SQ-1 is forbidden until SQ-0 GO. Its pilot is representative, graph-closed and anchor-power-valid under `<=128` unique videos/dataset and `<=1024` invocations total; an underpowered or oversized closure stops before calls. No validation/test teacher calls are ever allowed.
- **Supervision invariant:** the parent-video binary label remains the only gold. Blind whole-video presentation audit is signal QC, never training supervision. `segment_gold_exists=false`, `segment_gold_used=false`.
- **Global status:** **target remains active and unmet.** READY is method-specification readiness only; no SQ accuracy/macro-F1 result exists. Final success still requires two datasets, paired seeds 0/1/2, ordinary full-video kNN `+0.030/+0.030`, bootstrap/Holm and removal/permutation gates.
- **Canonical files:** `refine-logs/sq/FINAL_PROPOSAL.md`, `REVIEW_SUMMARY.md`, `REFINEMENT_REPORT.md`, `score-history.md`; full raw reviews are `round-1-review.md` through `round-4-review.md` and full anchor/simplicity revisions are `round-1-refinement.md` through `round-3-refinement.md`.

## Iteration 4 — SQ-RGCL S0 terminal provenance/audit fast-fail (2026-07-11)

- **Implementation integrity:** the first independent review found `4 CRITICAL / 9 HIGH`. The reachable formal-STOP path was repaired, including self-exclusion before rank assignment and fixed exact top-20 `20..1` signed-cosine arithmetic, then re-reviewed at `0 CRITICAL / 0 HIGH`.
- **SLURM execution:** jobs `12722` config hash, `12723` static sanity, `12724` freeze, `12725--12726` provenance, `12727--12728` local-CLIP q-proxy, `12729` blind whole-video audit-sheet freeze, and `12730` decision all completed. No `JobHeldUser` job was manually released.
- **Provenance result:** both archives match frozen hashes and train IDs with forbidden-key access count zero, but neither carries original-run cryptographic linkage for prompt, exact model revision, generator code, and input manifest. Both are therefore `PROXY_ONLY_CHEAP_FORMAT`, not promoted MLLM signals.
- **Blind QC:** 64 whole train videos per dataset were sampled without a dataset-label column. These sheets are QC only and never training gold. Human ratings were not fabricated; their absence is binding.
- **Decision:** `SQ-S0-DECISION-v1=STOP`, `lambda_Q=null`, `S1_unlocked=false`, `S2_unlocked=false`; S2--S4 remain locked. P0/power/micro and learned S1 were not run after the binding fast-fail.
- **Interpretation:** this is a provenance/governance failure, not evidence that the learned SQ mechanism succeeds or fails accuracy/macro-F1, and not a theoretical upper bound. It does not satisfy the global objective.
- **Supervision/call audit:** the parent-video binary label is the only gold. `segment_gold_exists=false`, `segment_gold_used=false`; new teacher/MLLM/OCR calls and teacher-cache reads/writes were zero.
- **Evidence:** `artifacts/sq/v1/S0_DECISION.json`, `refine-logs/sq/EXPERIMENT_TRACKER.md`, and `research-wiki/experiments/exp-sq-s0.md`.

## Iteration 5 — ECM-RGCL Gate-1 method review (2026-07-11)

- **Candidate:** `ECM-RGCL — Executable Constraint Modes`, activated only after SQ's formal S0 STOP. The proposed MLLM would process every train video under a strict-OOF, no-direct-outcome-field whole-video trace and emit weak posteriors over presentation/context inversion, target binding, modality conflict, surface shortcut, evidence dilution and undiagnosed. It would never see gold/correctness/error/loss/true-class margin; validation/test would have no mode or teacher.
- **Independent review:** canonical reviewer `/root/ecm_method_refine/ecm_reviewer`, one round, `4.98/10`, verdict **RETHINK**. No-segment-gold and test-clean audits passed; method identity failed.
- **Core rejection:** each soft mode risk is a weighted sum of per-example gradients, and the QP solution is another weighted sum. Thus the route is dynamic sample reweighting plus generic gradient surgery rather than a novel optimizer mechanism. It overlaps PG-DRO, JTT/EIIL and PCGrad/MGDA/CAGrad. Moreover, applying the constrained raw gradient through AdamW momentum/preconditioning/weight decay invalidates the claimed executable descent constraints.
- **Teacher-legality boundary:** no literal outcome field is useful hygiene, but not a correctness firewall. The teacher can infer its own hate judgment from the whole video and compare it with the OOF prediction, reconstructing an error propensity. A future semantic claim must beat a cross-fitted scalar ERROR-PROPENSITY arm with identical downstream capacity and a fine semantic shuffle preserving label/prediction/margin/error propensity.
- **Decision:** **ECM ABANDONED before implementation.** No code, SLURM job, MLLM/OCR/teacher call, cache, validation/test read or performance result was produced. The archival proximal-bank sketch in `round-1-refinement.md` is explicitly non-canonical and would require a fresh hypothesis/novelty review.
- **Anti-repeat:** do not rename pseudo-groups plus standard/soft GroupDRO, JTT, EIIL, PCGrad, MGDA, CAGrad or the frozen raw-gradient QP. Do not claim raw-gradient guarantees after AdamW or semantic necessity from error AUC. A genuinely new route must operate on realized final-bank geometry or actual adaptive-optimizer updates, prove numerical non-reducibility to scalar weighting, and keep the only-gold/video-label/no-segment/test-clean contract.
- **Global status:** target remains active and unmet. A new Gate-0 hypothesis is required; nothing in ECM establishes accuracy or macro-F1 improvement.
- **Canonical files:** `refine-logs/ecm/FINAL_PROPOSAL.md`, `REVIEW_SUMMARY.md`, `REFINEMENT_REPORT.md`; the complete raw response is `round-1-review.md`.

## Iteration 6 — LB-SCGP Gate-1 method refinement (2026-07-11)

- **Selected route/reviewer:** `LB-SCGP — Exact-Vote-Safe Structural-Reflection Gram Projection`; continuous reviewer `/root/iter6_architect`, scores `6.74 -> 7.86 -> 8.69 -> 9.15 READY`, anchor preserved, drift NONE, no-segment-gold PASS, no method blocker.
- **Frozen mechanism:** label-blind whole-video proposition/stance/quotation/condemnation/reportage/cross-modal-binding certificate, no verdict/score/rationale/key/outcome trace. Cache closes before labels enter a deterministic one-family structural-reflection compiler. Product-space Dykstra solves a self-excluded canonical-tie PSD/unit-diagonal exact-top20-vote-safe local Gram target; factor/Procrustes gives stopped `Z*`; the shared encoder fits it uniformly; test is unchanged ordinary full-video kNN.
- **Attribution:** FULL must beat DIRECT-AEXC, STATE-MOMENT, scalar label/error propensity, P4/TextTeacher and legal relation controls. Abstract and realized Farkas audits cover only registered cones. Final FULL-minus-strongest-direct must also pass 3/3 positive paired-seed and corrected-inference gates.
- **Supervision:** only parent-video binary gold. No segment/timestamp/span/localization gold, loss, weight or endpoint. All certificate fields remain weak/privileged pseudo-signals; validation/test load no certificate/compiler/target.
- **Authorization:** specification READY only. No code/job/cache/teacher call/result was produced. Next is independent implementation audit plus sealed synthetic/one-real-fold microbenchmark; projected ten-fold cost must be `<160 GPU-hours`. Only then may zero-teacher SCGP-0 run; teacher stays locked.
- **SCGP-0:** strict five-fold actual OOF on both datasets must gain `>=+0.050` accuracy and `>=+0.050` macro-F1, every fold positive, with all numerical/realized-fit/Farkas gates. Failure is terminal before teacher; a successful label-only arm raises the moving comparator.
- **Global status:** target remains active and unmet. Final success still requires both datasets × seeds 0/1/2 moving-baseline `+0.030/+0.030` ordinary-kNN gains and all mechanism/statistical gates.
- **Canonical files:** `refine-logs/lb_scgp/FINAL_PROPOSAL.md`, `REVIEW_SUMMARY.md`, `REFINEMENT_REPORT.md`, `score-history.md`; raw reviews `round-1-review.md` through `round-4-review.md`.

## Iteration 6 — LB-SCGP G0 implementation handoff audit (2026-07-11)

- **Launcher metadata:** implementation worker recorded supplied config `gpt-5.5`, reasoning `xhigh`, `--strict-config`.
- **Implementation status:** real-fold G0 producer/verifier were patched for fail-closed data isolation and richer real-core verification, but no independent 0C/0H audit exists. Do not submit SLURM from this handoff alone.
- **Data-access correction:** formal G0 no longer hashes/opens combined train/subclip caches, whole mixed `embeddings.npz`, or fold JSON because those package or contain held labels/content. This Round2 statement is superseded by Round3: formal G0 now requires only the train-only whole-video feature artifact with a frozen hash; subclips are not G0/G1 inputs.
- **Blocker:** the required train-only whole-video feature artifact is absent and hash-null, so formal freeze/realfold must fail closed until an independent reviewer approves safe artifacts.
- **Counts:** teacher/MLLM/OCR calls `0`; SLURM jobs `0`; no G0 artifact created.
- **Canonical handoff:** `refine-logs/lb_scgp/G0_IMPLEMENTATION_HANDOFF.md`.

## Iteration 6 — LB-SCGP G0 Round2 repair handoff (2026-07-11)

- **Launcher metadata:** Round2 worker recorded supplied config `gpt-5.5`, reasoning `xhigh`, `--strict-config`.
- **Status:** repair code/docs prepared for independent review only. No sanitizer build, sanitizer verification, G0 freeze, synthetic, realfold, replay, or decision job was submitted. No Python experiment computation was run on login. No performance result exists.
- **Data-isolation decision:** metadata/path inspection found no physically separated fold4 train-only feature source. Before formal freeze, the byte-level non-opening contract is revised to an explicit quarantine sanitizer plus independent sanitizer verifier. This Round2 statement is superseded by Round3: source locators/hashes are confined to quarantine source config/manifest; formal G0 freezes sanitized provenance/decision and the train-only whole-video feature output only, with no subclip artifact.
- **Critical repair surfaces:** real Dykstra/rank-cell now requires independently replayed compatible-cell objectives; fit/rollback requires a separate GPU replay artifact using the actual RA-HMD/RGCL checkpoint and live AdamW/scheduler/scaler/RNG/cursor state; Farkas covers registered singleton/pair/triplet/SupCon cones with separation oracles; H10 uses the exact registered formula with final bank outside the refresh term; resource gate verifies one-GPU SLURM/runtime metadata.
- **Governance:** protected-path dirty hashing excludes mixed/protected data and quarantine source config. Formal mixed-cache locator/hash count is zero. Supervision remains parent-video binary label only; no segment/timestamp/span/localization/stance/target/mechanism/rationale gold exists or is used.
- **Counts:** teacher/MLLM/OCR calls `0`; SLURM jobs `0`; no sanitizer/G0 artifact created; no 0C/0H closure is self-certified.
- **Global status:** target remains active and unmet. This Round2 next-action line is superseded by Round3; the next required action is independent review of the Round3 repairs, not G1 or teacher execution.

## Iteration 6 — LB-SCGP G0 Round3 repairs (2026-07-11)

- **Status:** Round3 repairs prepared for independent review, not run. No sanitizer build, sanitizer verification, G0 freeze, synthetic, realfold, replay, decision, G1, teacher, MLLM or OCR job was submitted or run.
- **Round2 review basis:** canonical `refine-logs/lb_scgp/G0_INDEPENDENT_REVIEW_ROUND2.md` was produced by a sole explicit GPT-5.5 xhigh reviewer with no delegation and reported FAIL: 3 Critical / 3 High. A first aborted Round2 review attempt was rejected only as a process issue because it spawned sidecars without xhigh proof; it is not a scientific finding.
- **No-segment repair:** G0/G1 subclips are no longer inputs. The sanitizer produces no subclip artifact; formal config/freeze no longer names one; producer and replay use whole-video memory only, enforce `lambda_seg=0`, pass `segment_cache=None`, and fail closed if a segment cache/objective appears. Inherited parent labels are not segment gold and are not used in segment supervision.
- **Isolation and verification repair:** the sanitizer formal decision carries no quarantine manifest locator/hash or access ledger naming source artifacts. Formal scanners reject quarantine/mixed/protected locators and source/mixed/legacy hash surfaces. The real verifier has a populated denylist and recursively scans manifest input/access records. Real Dykstra/rank-cell evidence now binds selected and adjacent per-projector transition/correction-state hashes; Farkas uses a pinned registered cone definition.
- **Global status:** no 0C/0H closure is self-certified. G1 and all teacher stages remain locked. The two-dataset, three-seed `+0.030 accuracy/+0.030 macro-F1` target remains active and unmet.

## Iteration 6 — LB-SCGP pre-G0 sanitizer build attempt (2026-07-11)

- **Review state before execution:** canonical Round3 independent review reported FAIL only because the physical sanitizer artifacts were absent. Code review had 0 High findings and the sole Critical was physical C1.
- **Authorized scope:** sanitizer build under SLURM, then verifier only if the build completed and produced `outer_train_features.pt`, `sanitized_provenance.json`, and quarantine `sanitizer_manifest.json`. No freeze, synthetic, realfold, replay, decision, G1, teacher, MLLM or OCR work was authorized or run.
- **Preflight:** `jq empty` passed for `configs/lb_scgp/lb_scgp_v1.json`, `configs/lb_scgp/lb_scgp_sanitizer_sources.json`, and `TARGET_STATE.json`; `bash -n scripts/slurm/lb_scgp_sanitize_inputs.sbatch` passed; metadata check found `artifacts/lb_scgp` absent.
- **SLURM execution:** build command `sbatch --export=ALL,TASK=build scripts/slurm/lb_scgp_sanitize_inputs.sbatch` submitted job `12737` for run ID `LBSCGP-G0-SANITIZE-MHC_zh-F4-v1`. It failed with state `FAILED`, exit code `1:0`, elapsed `00:00:04`, allocation `4 CPU / 32G`. No manual release/cancel/requeue was used.
- **Failure:** the job failed before sanitizer artifact creation during config loading. The log (`slurm/logs/lbscgp_sanitize_12737.out`, SHA256 `80d071490eb9f799ed119dce04d9004aad30a5fd2189ccc6b2c09d7d4fa4b4d7`) records `ValueError: 'configs/lb_scgp/lb_scgp_v1.json' is not in the subpath of '/data/jehc223/RGCL' OR one path is relative and the other is absolute` from `scripts/analysis/lb_scgp_common.py:73`.
- **Artifacts and counters at that point:** `artifacts/lb_scgp` was absent; sanitizer provenance, decision, quarantine manifest and feature artifact were absent, so no feature hash or decision fields existed. `segment_artifact_created=false`, `segment_objective_allowed=false`, `mllm_call_count=0`, `ocr_call_count=0`, teacher cache read/write counts `0`, and no performance result existed.
- **Historical status:** C1 remained open after job `12737`. This is superseded only by the later physical-verification entry below. G0 was not PASS, G1 and teacher stages remained locked, and the final two-dataset, three-seed target remained active and unmet. Canonical execution record: `refine-logs/lb_scgp/G0_SANITIZER_EXECUTION.md`.

## Iteration 6 — LB-SCGP sanitizer path-normalization repair (2026-07-11)

- **Status at repair handoff:** path-normalization repair prepared for independent review. No sanitizer build, sanitizer verification, G0 freeze, synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, network, artifact/cache write, Python/import execution, data/model execution, or performance work was run during the repair step. The later physical-verification entry below records the successful `12738/12739` rerun.
- **Root cause repaired:** `AccessLedger.hash_file` attempted `Path(path).relative_to(ROOT)` on the wrapper's relative `CONFIG=configs/lb_scgp/lb_scgp_v1.json`. A shared fail-closed canonical helper now resolves relative inputs under `/data/jehc223/RGCL`, rejects absolute/symlink escapes outside ROOT, and records stable ROOT-relative path strings.
- **Static hardening:** the same rule was applied to AccessLedger hash/read/record surfaces and reachable analogous config/source-config/artifact path handling in the sanitizer, sanitizer verifier, G0 producer, independent verifier, and real replay scripts. The sanitizer wrapper may continue using relative `CONFIG` and `SOURCE_CONFIG` defaults.
- **Static scan:** no direct `Path(...).relative_to(ROOT)` callsite remains outside canonical helper internals. Remaining `.resolve()` sites are helper internals or the independent verifier's formal-surface scanner fallback and are classified in `refine-logs/lb_scgp/G0_SANITIZER_PATH_FIX.md`.
- **Invariants:** no segment path was reintroduced; parent-video binary labels remain the only gold. `segment_gold_exists=false`, `segment_gold_used=false`, `segment_artifact_created=false`, `segment_objective_allowed=false`, `lambda_seg=0`, and `segment_cache=None` remain binding. Protected-path, formal-surface, no-clobber, source-isolation, and allowed-member checks were not loosened.
- **Historical status:** no 0C/0H closure was self-certified by the repair handoff itself. The subsequent path-fix review and physical-verification entries below supersede the resubmission blocker only. G1 and teacher remain locked, and the final two-dataset, three-seed target remains active and unmet.

## Iteration 6 — LB-SCGP sanitizer physical verification (2026-07-11)

- **Path-fix review:** `refine-logs/lb_scgp/G0_SANITIZER_PATH_FIX_REVIEW.md` reported PASS with 0 Critical / 0 High, authorizing a fresh sanitizer build under SLURM.
- **Job history:** `12737` remains preserved as failed-before-artifacts history. Job `12738` completed `TASK=build` in 5 seconds and job `12739` completed `TASK=verify` in 5 seconds. Both used `scripts/slurm/lb_scgp_sanitize_inputs.sbatch` with 4 CPU / 32G. At the end of this sanitizer-verification step, no G0 freeze, synthetic, realfold, replay, decision, G1, teacher, MLLM or OCR work had run.
- **Physical hashes:** feature `ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496`; sanitized provenance file `b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`; quarantine manifest file `055dffed9b61053293741ec5ba0ce3577daf458ceef4a3f81143a81d937c684b`; verifier log `40f333bb82884e9ad412fc87a20165e1640238815c3c6b6951d003e1cb0ec247`; decision file `172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`; decision payload `8685c805995f97ad0513658c1345b6320dc794e2bad02b907d9a2d07ff16cb1f`.
- **Verifier decision:** status `PASS`; `memory_id_count=464`; `query_id_sentinel_count=115`; all gates true; no segment artifact; no segment objective; no teacher/MLLM/OCR/network calls; no formal query reads.
- **Physical review:** `refine-logs/lb_scgp/G0_SANITIZER_PHYSICAL_REVIEW.md` reports 0 Critical / 0 High and closes the remaining C1 physical artifact blocker at artifact level only.
- **Global status at sanitizer close:** target remained active and unmet. G0 PASS was not claimed; G0 freeze and formal audit had not run. G1 and teacher remained locked. The next registered gate was G0 freeze/formal audit preparation, not G1 or teacher.

## Iteration 6 — LB-SCGP G0 freeze execution (2026-07-11)

- **Eligibility:** the canonical tracker registered `LBSCGP-G0-FREEZE-v1` as next. Sanitizer build `12738` and verifier `12739` were complete; physical review was 0 Critical / 0 High and closed C1 at artifact level only; `artifacts/lb_scgp/v1` was absent; no segment/subclip artifact existed under `artifacts/lb_scgp`.
- **SLURM execution:** exact command `TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v1 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch` submitted job `12742`. The job completed with state `COMPLETED`, exit `0:0`, elapsed `00:00:03`, allocation `8 CPU / 64G`, and no `--time`. No manual release, cancel or requeue was used.
- **Freeze artifact:** `artifacts/lb_scgp/v1/CONFIG_FREEZE.json` was created with file SHA256 `b6697472b61a61706c694a67b21618d618fcad6e7f59265d8696aee79dc46889`, payload SHA256 `b3b33090b39b3b975c2cf213aab669041b345c6ef3a3f7c200366a506bcebfd5`, and publish-lock SHA256 `34a05bde46775bcb75384c1c853cef7985f5c05efcdf3afec5fad6d85ef57d8d`. The job log `slurm/logs/lbscgp_g0_cpu_12742.out` has SHA256 `70cd194ed6a811be6956644e832055907f94750ccec6876352a8c1d6b5e98628`.
- **Bound contract:** the freeze records `status=FROZEN`, `stage=G0_FREEZE`, `conda_env=HateVideo`, only parent-video binary gold, `segment_gold_exists=false`, `segment_gold_used=false`, all MLLM/OCR/teacher/cache/held-label/held-content/val/test counters at zero, `G1_G4_locked=true`, and held IDs as exclusion sentinels only. The sanitizer provenance and decision hashes remain bound.
- **Non-claims:** no synthetic, realfold, replay, decision, G1, teacher, MLLM or OCR job was submitted. This is not G0 PASS, not a code audit, not a numerical result, not a performance result, and not a G1 or teacher unlock.
- **Audit note:** `CONFIG_FREEZE.json` includes pre-update hashes of mutable audit-trail records, including `refine-logs/lb_scgp/EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, and `TARGET_STATE.json`. The required post-run record updates therefore create a frozen-input drift hazard for later numerical predecessor checks; the independent formal code audit must assess this before any synthetic/realfold execution. This worker did not patch code or self-certify the audit.
- **Next gate:** `LBSCGP-G0-CODE-AUDIT-v1`, an independent formal code audit artifact with 0 Critical / 0 High. G1 and teacher remain locked. The global two-dataset, three-seed `+0.030 accuracy/+0.030 macro-F1` target remains active and unmet.

## Iteration 6 — LB-SCGP G0 v2 freeze repair (2026-07-11)

- **Why v2 exists:** `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW.md` failed v1 on C1 and H1. C1 was that v1 freeze hash-bound mutable progress records (`EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, `TARGET_STATE.json`) that must change after freeze. H1 was that the registered plan did not distinguish the narrower pre-freeze sanitizer schema from the full formal manifest/decision schema.
- **Repair:** v1 artifacts and locks remain untouched. `configs/lb_scgp/lb_scgp_v2.json` creates namespace `artifacts/lb_scgp/v2` and exact run identity `LBSCGP-G0-FREEZE-v2`. Producer and independent verifier now read run identities and freeze input keys from config while preserving v1 defaults. The v2 formal input set binds stable scientific protocol docs, immutable code/config/data artifacts, and `refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json`; it excludes mutable tracker/target/findings/handoff/execution logs from formal `input_files` and dirty-state predecessor checks.
- **Sanitizer schema clarification:** `refine-logs/lb_scgp/EXPERIMENT_PLAN.md` now prospectively registers the dedicated pre-freeze sanitizer schema, including mandatory fields, payload/hash bindings, no-clobber locks, counters, no-segment guarantees and no-held-access requirements. It states that the full generic manifest/decision schema applies from `G0_FREEZE` onward. Existing sanitizer artifacts were not edited.
- **Static checks:** shell-only `jq empty`, `bash -n`, `git diff --check`, `rg`, payload-hash checks, no-segment/subclip `find`, sanitizer schema `jq -e`, `sha256sum`, `squeue`, and `sacct` checks were used. No login-node Python/import/data/model execution occurred.
- **SLURM execution:** exact command `CONFIG=configs/lb_scgp/lb_scgp_v2.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v2 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch` submitted job `12746`. It began as `PENDING (JobHeldUser)`, released automatically, and completed `0:0` in `00:00:03` on `8 CPU / 64G`; no `--time`, manual release, requeue or cancel.
- **Artifacts:** `artifacts/lb_scgp/v2/CONFIG_FREEZE.json` SHA256 `4c6f0199a429bbf766cb284b9ecaedd9dcaf38733834dd54574245ab86b633ae`, payload `bda18d7dfda00ab5808595afec04e5b925d3a4920f55fe963dec5e9d99795c0b`, lock `22426f693ba3f72d52928b0a86d08fe439d7e7fd43f9c74c61aedb710443e211`; log `slurm/logs/lbscgp_g0_cpu_12746.out` SHA256 `e95d167af054b91582e1d1f8fbf66fb57a3b5cd67e298883940d385a65ccf563`.
- **Frozen-input proof:** immediately after freeze, every v2 file input and every allowed NPZ member matched the hash stored in `CONFIG_FREEZE.json`. The formal input list does not contain `EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, `TARGET_REVIEW_RAW.md`, `G0_V2_REPAIR_HANDOFF.md`, or `G0_FREEZE_EXECUTION_V2.md`.
- **Supervision and calls:** only parent-video binary labels are gold; `segment_gold_exists=false`; `segment_gold_used=false`; no segment/subclip artifact exists; MLLM/OCR/teacher/cache/held-label/held-content/val/test/formal-model-held counters are all zero.
- **Non-claims:** v2 freeze is not G0 PASS, not a formal code audit, not synthetic/realfold/replay/decision, not a performance result, and not a G1/teacher unlock. No PASS audit artifact was created. G1 and teacher remain locked.
- **Next gate:** fresh independent `LBSCGP-G0-CODE-AUDIT-v2`. The global final target remains active and unmet.

## Iteration 6 — LB-SCGP G0 v3 dirty-state repair (2026-07-11)

- **Why v3 exists:** `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V2.md` failed v2 with exactly one Critical. The v2 repair excluded mutable tracker/state/freeze records, but not the mandatory formal review record itself, so writing a required post-freeze review could change the dirty-state predecessor hash that the downstream decision verifier compares against `CONFIG_FREEZE.json`.
- **Repair:** v1/v2 freeze artifacts and locks remain untouched. `configs/lb_scgp/lb_scgp_v3.json` creates namespace `artifacts/lb_scgp/v3` and exact v3 run identities. Producer/common and the independent verifier now use the same config-explicit dirty-state policy for v3, with v1/v2 default compatibility retained. The v3 formal artifact exclusions are exactly `artifacts/lb_scgp/v1/`, `artifacts/lb_scgp/v2/`, and `artifacts/lb_scgp/v3/`; the only dirty prefix is `refine-logs/lb_scgp/runtime/`.
- **Narrow mutable audit paths:** v3 excludes only records expected to change after freeze: `refine-logs/lb_scgp/EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, `TARGET_REVIEW_RAW.md`, `refine-logs/lb_scgp/G0_V3_REPAIR_HANDOFF.md`, `refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V3.md`, and `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V3.md`. Source, config, experimental result and unrelated documentation paths are not broadly excluded.
- **Replay wrapper:** `scripts/slurm/lb_scgp_g0_gpu.sbatch` now reads `lineage.run_ids.replay` from `CONFIG` and fails closed if absent or mismatched. No replay job was submitted.
- **Static checks:** shell-only `jq empty`, `bash -n`, and `git diff --check` passed before the v3 freeze. No local Python/import/data/model execution was used.
- **SLURM execution:** exact command `CONFIG=configs/lb_scgp/lb_scgp_v3.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v3 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch` submitted job `12748`. It completed `0:0` in `00:00:03` on `8 CPU / 64G`; no `--time`, manual release, requeue or cancel.
- **Artifacts:** `artifacts/lb_scgp/v3/CONFIG_FREEZE.json` SHA256 `9fba7f1649dd67d4bb0fcc193e555d8246d7a4966307732e87d5e9fca7346dd9`, payload `352ec2215e2225b1768a13f39f96ef935b91966606d8db874d6de4410b1a9f3d`, lock `9c32d07c524e466ad06c06fc2a472829764cd1facff22390eac8b2879d329b8f`; log `slurm/logs/lbscgp_g0_cpu_12748.out` SHA256 `f60d2301e7460ab91b25f6c323c578e49a00001b46c67a590f9f3c3d58abf545`.
- **Freeze-bound hashes:** `dirty_diff_sha256=91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`; `config_canonical_sha256=84227b68eaa496da6e307ce5c5ef3469e1b7c68e350f0d62d1677d01f07645bf`; `implementation_sha256=b8759436a6c5e2a67bf7125cbd1ab57cb05187e764e837373abfdf1a92916e75`; `independent_verifier_sha256=d1e50057b4c166a71426f89474b6526e3eab11da5547e15368112b6620dbf5ce`; `access_ledger_sha256=3db4b94900a9d9b807ab495be869a5ef87a3894f987eef03ea1e948030abdc72`.
- **Frozen inputs:** v3 freezes the config, sanitizer provenance/decision, v2 pre-freeze sanitizer contract snapshot, checkpoint, train ledger, allowed bank members, train-only feature cache, and stable protocol documents. Mutable tracker/target/handoff/freeze/review records are not formal inputs.
- **Supervision and calls:** only parent-video binary labels are gold; `segment_gold_exists=false`; `segment_gold_used=false`; no segment/subclip artifact exists; MLLM/OCR/teacher/cache/held-label/held-content/val/test/formal-model-held counters are all zero.
- **Non-claims:** v3 freeze is not G0 PASS, not a formal code audit PASS, not synthetic/realfold/replay/decision, not a performance result, and not a G1/teacher unlock. No PASS audit artifact was created.
- **Next gate:** fresh independent `LBSCGP-G0-CODE-AUDIT-v3`. The global final target remains active and unmet.

## Iteration 6 — LB-SCGP G0 v3 formal code audit review (2026-07-11)

- **Review record:** `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V3.md`.
- **Verdict:** `PASS_REVIEW_ONLY`, with 0 Critical / 0 High / 2 Important. The no-segment-gold audit passed, and the exact v2 Critical is closed for v3.
- **Dirty-state proof:** the frozen v3 dirty hash and independent SLURM recomputes before report creation, after report creation, after report update, and after required documentation updates all equal `91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`. The mandatory v3 report is excluded by exact path, not by a broad report-prefix or root-level exclusion.
- **Artifact boundary:** no formal PASS artifact was created. The repository contains downstream consumers for `artifacts/lb_scgp/v3/g0/code_audit/{review.md,audit.json}`, but no existing protocol-authorized producer task was found; hand-fabricating PASS JSON is forbidden.
- **Supervision and calls:** parent-video binary label remains the only gold; `segment_gold_exists=false`, `segment_gold_used=false`; MLLM/OCR/teacher/cache/held-label/held-content/val/test counters remain zero.
- **Non-claims:** no synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, held/val/test, or performance stage ran. G0 is not passed, and G1/teacher remain locked.
- **Next boundary:** provide or explicitly authorize a real no-clobber formal code-audit artifact producer, then re-verify artifact schema/hash locks before any synthetic stage. The global two-dataset, three-seed `+0.030 accuracy/+0.030 macro-F1` target remains active and unmet.

## Iteration 6 — LB-SCGP G0 v4 tooling-lineage repair and freeze (2026-07-11)

- **Why v4 exists:** v3 closed the dirty-state predecessor defect but exposed a tooling boundary: a 0 Critical / 0 High free-form review existed without an authorized no-clobber formal PASS producer. v4 repairs only that tooling-lineage gap.
- **Config and namespace:** `configs/lb_scgp/lb_scgp_v4.json` registers namespace `artifacts/lb_scgp/v4`, exact v4 run IDs, future independent review path `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md`, and strict machine-record path `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json`.
- **Publisher architecture:** `scripts/analysis/lb_scgp_independent_verify.py --task audit-publish` validates a strict machine-verifiable independent review record rather than parsing free-form report text. It recomputes freeze/config/implementation/verifier/dirty hashes, rehashes all frozen inputs and allowed NPZ members without opening forbidden query members, checks v1-v3 freeze/lock hashes, verifies zero forbidden counters and no-segment-gold, and publishes formal artifacts transactionally only after all checks pass.
- **Wrapper:** `scripts/slurm/lb_scgp_g0_audit_publish.sbatch` is CPU-only 2 CPU / 4G, has no `--time`, activates `HateVideo`, uses offline flags, and rejects wrong `TASK`, `RUN_ID`, `CONFIG`, `REVIEW`, or `REVIEW_RECORD`.
- **Consumer synchronization:** the producer predecessor loader and final decision verifier both consume the same exact v4 audit schema and reject missing/additional/drifted fields.
- **SLURM freeze:** exact command `CONFIG=configs/lb_scgp/lb_scgp_v4.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v4 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch` submitted job `12759`. It initially entered `PENDING (JobHeldUser)`, released automatically, and completed with state `COMPLETED`, exit `0:0`, elapsed `00:00:03`, allocation `8 CPU / 64G`, and no `--time`.
- **Freeze artifact:** `artifacts/lb_scgp/v4/CONFIG_FREEZE.json` SHA256 `dcf65eceba04e7c4f08145b2012653705f7347c6e96ebc8b2b769280dff48fd0`; payload `92301ad95870b8a1af41e7e69e45054a2e63fe80a021cf4c1c22906aea0872bf`; lock `09003ce9e741d7c0310045f854479deb8fecff74bfddd33f6b9d80dc6df9572a`; log `6203cab3eded38f22638980c4828020a10ddbd8421819a0d5f5059eca6faa6da`.
- **Freeze-bound hashes:** dirty `8ca10aec315f800959e7869beb200f4bbc5f5d27841d8c307d896f1644803e7a`; config canonical `9e99cba37486e2511b0e37fb7d2c3b59053fbac8aca577ba05b36c138aa67c56`; implementation `c7e9371494f991d88a7ab93cc64769fa1e6a92913df3afd2f647201d0eef1bf1`; independent verifier `03a78a89867d3cea468b5319463ccabcefa4b4a589a61863bffd3e14c9df5402`; access ledger `ef67ad3b6521a9b8e9b73dd27260917c531e4ec72a84e04c52afed0c34ba72a7`.
- **Non-claims at v4 freeze:** this repair did not create `G0_FORMAL_CODE_AUDIT_REVIEW_V4.md`, did not create a review machine record, did not create formal PASS artifacts, and did not run audit-publish. No synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, held/val/test, or performance stage ran.

## Iteration 6 — LB-SCGP G0 v4 operational failure and v5 NameError repair (2026-07-11)

- **v4 operational status:** v4 formal code-audit files later existed and were internally consistent, but publication verification recorded `all_ok=false` because producer `_load_freeze_and_audit` raised `NameError: name 'git_state' is not defined` while consuming the strict audit schema. v4 is therefore operationally `FAIL/BLOCKED`; its formal outputs are failed-lineage evidence only and never valid downstream unlock evidence.
- **Repair scope:** v5 repairs only the missing producer import plus v4/v5 lineage/path/schema/no-clobber support across publisher, producer, and decision consumer. It does not change scientific method, thresholds, supervision, data protocol, numerical logic, or evaluation behavior.
- **v5 config and namespace:** `configs/lb_scgp/lb_scgp_v5.json` registers namespace `artifacts/lb_scgp/v5`, freeze run ID `LBSCGP-G0-FREEZE-v5`, code-audit run ID `LBSCGP-G0-CODE-AUDIT-v5`, future review path `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.md`, and future record path `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json`.
- **Focused regression:** audit-only SLURM job `12820` completed under conda `HateVideo` with result `PASS`, proving valid v5 producer consumption and fail-closed wrong-hash, dirty-hash, run-ID, and path cases using temporary runtime fixtures. The fixture root was removed and no formal v5 namespace was created by the regression.
- **Formal freeze:** the only authorized formal v5 workload was `CONFIG=configs/lb_scgp/lb_scgp_v5.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v5 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch`. Job `12823` initially entered `PENDING (JobHeldUser)`, released automatically, and completed `0:0` in `00:00:02` on `8 CPU / 64G`. It produced `artifacts/lb_scgp/v5/CONFIG_FREEZE.json` SHA256 `254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931`, payload `d89f7cd4ad43c8ef83a04b7530ec19186ec1cb0e96958d239f1ce5c2b146bb4d`, lock `54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9`, and log `c43a2b16fc8c95bdfafb5c48c674fa4b778dd9c3d3c3764e24fe50ed038a0526`. No v5 review, record, audit-publish, synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, held/val/test, or performance workload is authorized in this repair.
- **Supervision and counters:** the only gold remains `parent_video_binary_label`; `segment_gold_exists=false` and `segment_gold_used=false`; MLLM/OCR/teacher/cache/held-label/held-content/val/test/formal-held counters remain zero.
- **Next boundary:** a fresh independent GPT-5.5 xhigh v5 audit only. The global two-dataset, three-seed `+0.030 accuracy/+0.030 macro-F1` target remains active and unmet.

## Iteration 6 — LB-SCGP G0 v5 formal audit publication and post-verification (2026-07-11)

- **Independent audit:** `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.md` is finalized with PASS, Critical `0`, High `0`, Important `3`, no-segment-gold PASS, and SHA256 `495b5f3bc453034ae5f9830a77bc9b4a2b04af181b0d4365e95bdbaf450bd36b`. The strict record `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json` has SHA256 `4cb399a0209025581cef094f9f339b6617d9a0ad1d22d5925c131107118a3770` and payload SHA256 `b11589d4a892afa3f05982b29dd93276bbde20969804e4c853734f76e6be63c0`.
- **SLURM audit chain:** negative audit job `12825`, record validation job `12829`, formal publisher job `12830`, and post-publication verifier job `12831` all completed `0:0` under conda `HateVideo`. The earlier negative-harness attempt `12824` failed before a valid matrix and created no formal output.
- **Publication:** job `12830` produced the formal v5 code-audit bundle transactionally under `artifacts/lb_scgp/v5/g0/code_audit/`; `audit.json` payload SHA256 is `46c3eece9f51b285749d7b70cab863be44eaf3acb930ca7c8c91a47353997016`. No temporary publication residue remains.
- **Post-verification:** job `12831` really called producer `_load_freeze_and_audit` and the strict decision-consumer verifier against the actual published v5 bundle. It reports `producer_consumer_ok=true`, `decision_consumer_ok=true`, `all_ok=true`, and `dirty_equal_frozen=true` for frozen/current dirty hash `1c8284781fb57e90714b390fdbef362e978b70789632b19df3d8161dfe8827b7`.
- **Supervision and locks before numerical execution:** only `parent_video_binary_label` is gold; `segment_gold_exists=false`, `segment_gold_used=false`; forbidden MLLM/OCR/teacher/cache/held-label/held-content/val/test/formal-held counters remain zero. At that boundary only `LBSCGP-G0-SYNTH-v5` was unlocked. Realfold, replay, decision, G1, teacher, MLLM, OCR, held, val, test, and performance work remained locked.

## Iteration 6 — LB-SCGP G0 v5 synthetic numerical STOP (2026-07-11)

- **Execution:** exact command `CONFIG=configs/lb_scgp/lb_scgp_v5.json TASK=synthetic RUN_ID=LBSCGP-G0-SYNTH-v5 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch` submitted exactly one synthetic job, `12833`. It ran under SLURM with conda `HateVideo`, no `--time`, no manual release/requeue/cancel, and completed terminal state `FAILED`, exit `2:0`, elapsed `00:00:44`, allocation `8 CPU / 64G`, MaxRSS `161708K`.
- **Failure artifact:** log `slurm/logs/lbscgp_g0_cpu_12833.out` SHA256 `a8a249101ebf8ebe3ab56d5b152b8df35a8f593c2271e26e111b04364726ce49`; manifest `artifacts/lb_scgp/v5/g0/synthetic/manifest.json` SHA256 `07dc7d5d17194cd7a2b5d42d539adb9e8248e78b4dc629bbcdaf9d4f64719242`, payload SHA256 `751b5ede4cdd6f05032768b4c9295b56ba62fbe370be11436f7ca3f7dbec3fc5`.
- **Gate result:** manifest records `status=FAIL`, `thresholds_ok=true`, `expected_statuses_ok=false`, `dykstra_gate=false`, `rank_gate=true`, `farkas_gate=true`, `factor_gate=true`, `rollback_gate=true`, and `overflow_nan_inf_count=0`. The concrete mismatch is Dykstra expected-status parity: `feasible_interior`, `feasible_boundary`, and `feasible_oriented_boundary` returned `BOUNDED_SEARCH_FEASIBLE` while the frozen v5 ledger expected `LOCAL_STATIONARY_CERTIFIED`.
- **Stopped work:** no replacement synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, held/validation/test evaluation, or final performance training job was submitted. This is a numerical/executability STOP, not a final accuracy or macro-F1 experiment and not evidence against or for the final LB-SCGP performance claim.
- **Supervision/access:** only `parent_video_binary_label` remains gold; `segment_gold_exists=false`, `segment_gold_used=false`; MLLM/OCR/teacher/cache/held-label/held-content/val/test counters remain zero. No `query_z`/`query_labels`, held content/labels, validation/test content, segment objective/cache, MLLM, OCR, or teacher artifact was opened or generated.
- **Record:** `refine-logs/lb_scgp/G0_V5_NUMERICAL_EXECUTION.md`. The next boundary is fresh repair/review authorization before any further LB-SCGP G0 work. The global two-dataset, three-seed `+0.030 accuracy/+0.030 macro-F1` target remains active and unmet, but this v5 G0 branch is stopped at synthetic.

## Iteration 6 — LB-SCGP G0 v6 result-to-claim KILL (2026-07-12)

- **Recording note:** this section and every section below were recorded post-hoc by assistant on 2026-07-12 at user request; the loop's own workers did not write them. Each cited number was re-read from the referenced source file before transcription.
- **Verdict:** fresh independent result-to-claim review returned `claim_supported=partial`, `route=supplement`. The authorized single v6 repair did not certify the actual oriented fixture and did not reach G0 PASS.
- **Cell evidence:** Cell 0 Phase-I had a replayed full-rank feasible witness, but Cell 0 Phase-II returned `BOUNDED_REMOVE` (not `LOCAL_STATIONARY_CERTIFIED`) with original-objective stationarity infinity-norm `0.0022782803493536386` and PSD minimum eigenvalue `-5.8836793827797916e-09`. Cell 1 was `NO_WITNESS`: candidates satisfied all 589 residuals with positive eig margin but realized a non-target top20 hash across starts.
- **Root causes:** strict rank-cell encoding ambiguity plus absent original-objective local stationarity. No conic Farkas incompatibility certificate was produced, so this is explicitly not an infeasibility proof of the target cells.
- **Authorization:** the review authorized at most one further v7 repair (explicit signed tie-gap halfspaces + canonical compatible-cell definition + solver-independent primal-dual certificate) and pre-registered a single global PSD/unit-diagonal Gram-target projection as the pivot fallback if v7 failed.
- **Supervision/counters:** only `parent_video_binary_label`; `segment_gold_exists=false`, `segment_gold_used=false`; MLLM/OCR/teacher/held/val/test counters zero.
- **Canonical record:** `refine-logs/lb_scgp/G0_V6_ACTUAL_RESULT_TO_CLAIM_REVIEW.md` (also `G0_V6_NUMERICAL_CERTIFICATE_REVIEW.md`, `refine-logs/lb_scgp/v6/`). Global target remains active and unmet.

## Iteration 6 — LB-SCGP G0 v7 result-to-claim KILL + PIVOT (2026-07-12)

- **Verdict:** fresh independent result-to-claim review returned `claim_supported=no`, `route=pivot`. The final authorized local rank-cell repair failed and the route is retired as the main G0 gate.
- **Execution:** certificate job `12896` and independent replay job `12898`. Canonical-cell enumeration was complete (one boundary descriptor `p00: p15 vs p18`, compatible assignments `[-1]` and `[1]`, 528 additive signed-gap rows per cell). Both cells kept the target canonical top20 with small original residuals and positive PSD margins.
- **Decisive failure:** the preregistered strict signed-gap `G[q,a]-G[q,b] >= tau+eta` (`tau=1e-7`, `eta=1e-12`, RHS `1.00001e-7`) failed by cell 0 `min_margin=-6.736095449260811e-15` and cell 1 `min_margin=-1.7186069668545097e-14`, both `pass=false`. There is no replay tolerance for negative margins; `top20_equal_cell=true` is insufficient because the gate was strict signed-cell compatibility.
- **Non-infeasibility:** Phase II was empty (no strict Phase-I witness), and replay preserved `NO_COMPATIBILITY_WITNESS_NO_FARKAS` — neither compatibility nor a Farkas incompatibility certificate. The route is retired because this was the final authorized repair, not because the cell was proven empty. v8 / further local rank-cell solver tuning is forbidden.
- **Pivot skeleton (mechanism-preserving):** train-only, label-blind structural certificates -> one global full-bank PSD/unit-diagonal Gram target -> uniform encoder fit -> ordinary test kNN. Still MHC-EN and MHC-ZH, seeds 0/1/2, `+0.030/+0.030`, with REMOVE/SHUFFLE/NOISE/direct controls; no sample weighting/reranking/key selection/pair-triplet/SupCon/segment route.
- **Reusable assets:** parent-video-label-only supervision contract, no-held/val/test discipline, immutable/hash-bound artifacts + independent replay, PSD/unit-diagonal full-bank Gram representation, exact ordinary top20 kNN endpoint.
- **Canonical record:** `refine-logs/lb_scgp/G0_V7_RESULT_TO_CLAIM_PIVOT_REVIEW.md` (design `refine-logs/lb_scgp/v7/G0_V7_PREREGISTERED_DESIGN.json`; results `v7/results/v7_actual_certificate_12896.json`, `v7_independent_replay_12898.json`). Global target remains active and unmet.

## Iteration 7 — LB-SCGP Global-R2 Gate-0 method refinement (2026-07-12)

- **New method family:** LB-SCGP Global-R2 (`refine-logs/lb_scgp_global/`), reached via the v7-authorized pivot. The iteration is incremented 6 -> 7 following the loop convention that each new Gate-0 method family opens a new iteration (SSR=1, EDCM=2, CTE=3, SQ=4, ECM=5, LB-SCGP local-rank-cell=6); v6 and v7 stay under iteration 6 because they are intra-family LB-SCGP local-rank-cell repairs.
- **Mechanism:** a closed train-only, label-blind structural-certificate cache defines `A_eq`/`A_band`/`A_reg` and robust-edge geometry for a single replayable full-bank PSD/unit-diagonal proximal Gram target; the shared encoder fits it uniformly; test is unchanged ordinary full-video train-memory top20 kNN. No test-time MLLM/teacher/head/rerank; certificates are not weights/selectors/keys/losses.
- **Review:** independent GPT-5.5 xhigh research-refine scored `7.1 REVISE -> 8.3 REVISE -> 9.1 READY` across three rounds (`refine-logs/lb_scgp_global/score-history.md`, `REFINE_STATE.json` `last_verdict=READY`).
- **Success condition (unchanged):** MHC-EN and MHC-ZH, seeds 0/1/2, accuracy and macro-F1 each `+0.030` over the strongest same-protocol non-MLLM comparator, all paired seed deltas positive, hierarchical paired-bootstrap lower bound >0, Holm; FULL must beat REMOVE/SHUFFLE/NOISE and the strongest direct/scalar control.
- **Status:** READY is method-specification readiness only; no performance result exists. Canonical files: `refine-logs/lb_scgp_global/FINAL_PROPOSAL.md`, `REVIEW_SUMMARY.md`, `REFINEMENT_REPORT.md`, `EXPERIMENT_PLAN.md`. Global target remains active and unmet.

## Iteration 7 — LB-SCGP Global-R2 M0 contract freeze, Run2 infra crashes, and Welch-bound repair (2026-07-12)

- **M0 contract freeze (Run1):** job `12901` completed `0:0`, state FROZEN; `artifacts/lb_scgp_global/v1/m0/contract_freeze.json` SHA256 `09b78682389f1c9774c9dffc43c759bceeec9d7f44eca1ce4cd626d0cd6d12da`, payload `57f935cfa6ff22f81ec726eba9e0000d76f95bf93575b7539b78ba4d7c5bde53`. All access counters zero; validation/test/held/cache/MLLM/OCR/query content not opened; protected old LB-SCGP scope unchanged (278 files, manifest `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`). Not M0 success. Record: `refine-logs/lb_scgp_global/M0_CONTRACT_FREEZE_EXECUTION.md`.
- **Run2-v1 fail-closed crashes:** the v1 synthetic-KKT was attempted twice under the same run ID and both failed before any published artifact. Job `12902` (older `run2_synth_kkt` path) raised `KeyError: 'finite_vi_diagnostic'` in `verify_manifest` (exit `1:0`, `00:00:04`). Job `12904` (newer `run2` validator path) raised `KeyError: 'payload_schema'` in preflight (exit `1:0`, `00:00:01`). Both are infrastructure/interface failures, not scientific/KKT/numerical/rank/factor/mechanism results; total spend ~40 CPU-seconds, 0 GPU. The v1 Run2 lineage is spent and must not be retried. Records: `refine-logs/lb_scgp_global/M0_SYNTH_KKT_EXECUTION.md`, `M0_RUN2_RESULT_TO_CLAIM_REVIEW_FRESH.md` (`claim_supported=no`, `route=infrastructure_repair`).
- **Run2-v2 (prospective, locked):** a non-clobbering v2 lineage (`LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2`, namespace `artifacts/lb_scgp_global/v2/...`) was authorized in principle. The plan amendment passed 0C/0H; the first implementation fix/freeze was reviewed 0C/2H/1M/1L; fix2 is now complete and statically frozen but `ready_for_execution=false`. Execution remains locked pending a fresh independent 0C/0H implementation/code review, exact-hash no-clobber review, and separate execution authorization. Record: `refine-logs/lb_scgp_global/M0_RUN2_V2_IMPLEMENTATION_FIX2_FREEZE.md`, tracker Run2-v2 notes.
- **Welch-bound feasibility finding:** the failed implementation tried to keep `G0=I_N` while requiring `G_star` PSD, unit-diagonal, rank `<=d=3`, nontrivially moved, within off-diagonal coordinate-trust `<=0.02`. For `N>d` this is impossible: any rank-`d` correlation matrix obeys `mean_{i<j} G_ij^2 >= (N-d)/(d(N-1))`, so for the FULL fixture `N=10,d=3` some `|G_ij| >= sqrt(7/27) ~= 0.509`, far above `0.02`. Fix2 preserves `d`, thresholds, fixture identities/counts, and the rank/movement gates by using a unit-diagonal local-projection baseline `G0` instead of the identity baseline. Record: `refine-logs/lb_scgp_global/M0_RUN2_V2_IMPLEMENTATION_FIX2_FREEZE.md` (lines 11-15).
- **Pending semantic adjudication:** if a future reviewer decides `G0=I_N` was a frozen scientific semantic (rather than a flawed implementation choice), H5 cannot be closed under `N=10,d=3,trust=0.02`, and the minimal amendment would be either to authorize the unit-diagonal local-projection baseline used by fix2 or to change one of those frozen constraints. This adjudication is deferred to the user / a fresh reviewer.
- **Supervision/counters:** only `parent_video_binary_label`; `segment_gold_exists=false`, `segment_gold_used=false`; MLLM/OCR/teacher/held/val/test counters zero throughout M0. Global target remains active and unmet.

## Registry erratum — HateMM strongest-non-MLLM baseline (2026-07-12)

- **Correction:** the Iteration-0 "Exact baseline registry" HateMM row value `0.8732 / 0.8686` is superseded by the verified `0.8279 / 0.8172` (binding `+0.030` target `0.8579 / 0.8472`). `0.8732` was a Val_Retrieval ROC-AUC mis-transcribed as a test accuracy; `0.8686` was a phantom with no source reading. Correct value = val-selected epoch-24 Test_Retrieval test acc / macro-F1 at `slurm/logs/rgcl_HateMM_openai_clip-vit-large-patch14-336_HF_1035814.trainlog:257,259` (n=215); erratum commit `66012e9`.
- **Scope of edit:** `TARGET_STATE.json` `exact_baselines.HateMM` was corrected in place (accuracy/macro_f1/hard_target/source/audit_note). The historical Iteration-0 registry table above was also corrected in place on 2026-07-12 (HateMM row: value `0.8279 / 0.8172`, hard target `0.8579 / 0.8472`, and provenance caveat), with an inline correction marker; treat this erratum as authoritative. (recorded post-hoc by assistant, 2026-07-12, per user request)

## Iteration 8 — target-loop recovery and candidate registry (2026-07-28)

### Environment routing and resume decision

- `whoami=jehc223`, `$USER=jehc223`; execution policy remains `slurm_only`.
- Any later GPU or compute workload must use project-local `scripts/slurm/*.sbatch`, `conda activate HateVideo`, no `--time`, and must wait for `PENDING (JobHeldUser)` to release automatically. This registry update is documentation-only: no computation, teacher call, GPU process or SLURM job was launched.
- `TARGET_STATE.json` existed, so this is a resumed run. Iterations 0–7, their jobs, frozen artifacts, failures and reviewer outputs remain historical evidence. Iteration 8 reopens Gate 0; it does not convert the unfinished Iteration-7 LB-SCGP Global-R2 lineage into a performance result.

### Binding target and constraints

- Target: on at least **two datasets**, relative to the strongest strict same-protocol paired baseline, both accuracy and macro-F1 must improve by at least `+0.030`, using seeds `0/1/2`; all paired seed deltas must be positive and the hierarchical paired-bootstrap/Holm 95% lower bound must exceed zero.
- The no-selection/final-epoch protocol is primary and validation-selected is corroborative; split, evaluator, label space, head/retrieval path and checkpoint rule must match the paired baseline.
- Forbidden: model-size scaling, substantial extra data, brute-force epochs, ensemble stacking, cross-seed or final multi-prompt ensembles, leakage/test-driven selection, OCR, external APIs, cross-dataset mixing/training, external-pool training, segment/span/target gold, or treating MLLM pseudo-signals as gold.

### Evidence inherited into Gate 0

- Stable errors, not seed noise, dominate the remaining pool: HateMM has 25 errors stable across 3/3 seeds; MHC-EN has 22 hard 4/4 errors; MHC-ZH has 22 stable 3/3 errors. See `refine-logs/ERRPAT_HateMM_2026-07-26.md`, `ERRPAT_MHC-EN_2026-07-26.md`, and `ERRPAT_MHC-ZH_2026-07-26.md`.
- Discourse/source/stance errors recur: HateMM contains quote/archive/song and long-talk false positives; MHC-EN contains lexical-surface and counterspeech/meta false positives. Evidence density also recurs but with dataset-specific shape: short/speech-poor hate in HateMM, long-transcript dilution in EN, and a middle-thin interval in ZH. This motivates C01/C02/C04/C08 rather than a global length or lexical rule.
- Archive inference keys are null-to-negative and Archive-v2 can harm; C12 is therefore allowed only as a training-time stability curriculum. Sources: `research-wiki/experiments/exp-archive-knn-seeds.md` and `research-wiki/ARCHIVE_V2_ITERATION.md`.
- F113 (`refine-logs/HEADSPACE_TRANSFER_PREGATE.md`) establishes that raw-key-space positives are optimistic: the only raw lead shrank about 28x in fold-head space. Future raw-space screens may kill a candidate, but only fold-head/deployed-head measurements may promote one.
- Existing literature maps are `refine-logs/LITSURVEY_MLLM_EMBEDDING.md`, `LITSURVEY_NOVEL_MECHANISMS.md`, and `LITSURVEY_RETRIEVAL_MEMORY.md`. They cover LLM2Vec/NV-Embed-style MNTP, prompt/readout methods, representation/geometry distillation, DkNN/NCP conformal uncertainty, and memory-editing/modular-memory lineages. C03 must be native policy-anchored MNTP, not the killed readout-only or incompatible LoRA-transplant variants documented in `MNTP_FORENSIC_RECON.md` and `MNTP_S1_RECORD.md`.

### Active hypothesis cards

| Slot | Candidate | Claim | Falsifiable prediction | Non-isomorphism boundary |
|---|---|---|---|---|
| A | **C01 Policy-Contrastive Discourse Transport** | Policy-conditioned contrasts isolate endorsement/quotation/reporting and move that residual into the native representation. | Source/stance hard-error subsets gain conditional information and actual fold-head net fixes without an equal rise in break exposure on at least two datasets. | Representation transport, not archive lookup, scalar MLLM scores, vote replacement or routing. |
| B | **C02 Evidence-Density Quotient Geometry** | Controlled views can quotient semantic content from evidence quantity and remove length/density nuisance. | Within-video view agreement rises while length predictability falls, with `+0.040/+0.040` projected signal on at least two datasets. | Not global length correction, P3 pooling, summary replacement, modality dropout or stream weighting. |
| C | **C03 Policy-Anchored Native MNTP** | Native in-domain MNTP anchored to moderation-policy relations changes semantic geometry beyond readout changes. | Native objective improves discourse-relation geometry and fold-head metrics; readout-only and shuffled-policy controls do not. | F92/F93 remain dead: no training-free mask/readout rerun and no published-LoRA transplant shortcut. |

### Candidate registry and ordered backlog

1. C01 Policy-Contrastive Discourse Transport — active A.
2. C02 Evidence-Density Quotient Geometry — active B.
3. C03 Policy-Anchored Native MNTP — active C.
4. C04 Source-Proposition-Stance-Harm Tensor.
5. C05 Full-Bank Signed Discourse Manifold.
6. C06 Prompt-Orbit Tangent/Curvature.
7. C07 Harm-Lattice Cone Metric.
8. C08 Provenance-Antisymmetric/Title-source Encoder.
9. C09 Stable-Inversion Topology Surgery.
10. C10 Gold-free Reasoning-Boundary Structured Memory.
11. C11 Null-aware Evidence Representation.
12. C12 Archive-version Stability Curriculum.
13. C13 ZH HTML Markup Invariance.
14. C14 Multi-prompt Representation Ensemble — mechanism/upper-bound gate only, low novelty, never eligible as the final ensemble method.

The machine-readable registry, status, claims and exact dedup boundaries are in `TARGET_STATE.json` under `registry_update_2026_07_28`.

### Unified minimal-pilot gate

1. **Reachability before teacher/GPU:** strict train-OOF or untouched-dev, actual fold-head/deployed-head path. Full-bank/representation oracle must reach `+0.050 Acc` and `+0.050 macro-F1` on at least two datasets and supply enough net correct-minus-broken items for the final `+0.030` bar.
2. **Signal gate:** any teacher/auxiliary signal must produce capacity-matched projected `ΔAcc >= +0.040` and `Δmacro-F1 >= +0.040` on at least two datasets, 95% CI lower bound `>0`, with valid oracle/calibration sensitivity.
3. **Minimal end-to-end:** seed-0 same-protocol dev pilot must obtain at least `+0.020/+0.020` on at least two datasets and no claimed-dataset primary harm below `-0.005`.
4. **Promotion:** only then expand to seeds 0/1/2. Final promotion needs mean `+0.030/+0.030` on at least two datasets, 3/3 paired signs, corrected CI lower bounds above zero, and FULL beating REMOVE, SHUFFLE/PERMUTE and NOISE controls.

### Serial execution and pivot rule

- Execute exactly one candidate at a time: `C01 -> C02 -> C03`. No parallel GPU/teacher pilots.
- Give each candidate one minimal falsifiable round. A binding failure is logged and immediately advances to the next candidate.
- After two consecutive active-candidate failures, reopen literature/novelty Gate 0 before consuming the ordered backlog. Backlog order is C04 through C14 as listed above.
- No experiment is authorized by this registry entry itself. Before any implementation or submission, the selected candidate still needs its prospective preregistration, novelty/non-isomorphism review, experiment plan and code review.

### Explicitly eliminated scope

- The authoritative ledger is `autoresearch/goal_mllm_plus3/state/directions_tried.json` (`dead` currently contains 76 entries), with detailed findings in `findings.jsonl`. Its bans remain binding.
- Closed families include P1–P5/P9–P11/TARC/archive-auto-repair; old LB-SCGP local lineages; plain scale/encoder/LoRA novelty; global threshold and vote/top-k changes; routers and stream reweight/fusion; W2/S2S/CTF/GIR temporal/set/grounded retrieval; audio/prosody/CLAP; generic head losses; top-20 verifier/reranker/residual/aggregation and the exhausted MEMBANK sweep; evidence-unit/bank-synthesis/verbalization as posed; OCR; readout-only/bidirectional-mask/MNTP transplant; vision-only swaps/resolution/frame budget; generated-data augmentation and cross-dataset/external-pool expansion.
- Candidate names do not override these closures. C03, C05, C07, C10, C12 and C14 have explicit collision boundaries in the registry and must fail closed if the claimed delta cannot be demonstrated.

### Out-of-target alternatives

- **Conformal three-way output** `{hate, non-hate, refer}` is retained only as a selective-risk/coverage contribution. It changes coverage/output space and therefore is not part of the full-coverage `+0.030/+0.030` loop.
- **Cross-bank drift certificate** is retained only as a robustness/audit contribution, not as a primary accuracy mechanism.

### Target status / decision

- Baseline registry remains `TARGET_STATE.json.exact_baselines`; current proven target status remains unmet on `0` qualifying datasets.
- Decision: **Gate 0 remains open and unpassed.** Active A/C01 is permitted only a pre-Stage0 kill-only diagnostic; it has not advanced Gate 0 -> Gate 1. B/C remain serial fallbacks. No job or new metric exists for Iteration 8.
- Why scientific: every candidate must state a representation-level causal mechanism, a measurable intermediate prediction, a smallest falsifying intervention, and a non-isomorphism boundary against the 76-entry eliminated ledger. Scale, data, ensemble and protocol changes cannot carry a result.

### C01 A0 experiment card — revised implementation ready, not run (2026-07-28)

- **Frozen scope:** parameter-free block-normalized contrast of the existing standard/one-word L24 endpoints on MHC-ZH and HateMM, strict train-memory -> dev-query only. Because prompt and pooling both differ, this is a readout-policy endpoint audit, not a prompt-only or safety-disentanglement claim.
- **Strong controls:** both endpoints, score average, equally normalized endpoint concat, common/displacement, secondary `common_interaction`, 256 label-blind ID-hash shuffled pairs, and the complete ex-ante orthogonal-rotation angle set with identical block L2 and Holm correction. Displacement-norm/tiny-row and train-defined small-displacement gain-concentration audits are binding.
- **Static NO-GO repair:** code now exact-compares a complete internal canonical binding, rejects empty check families, freezes the decision schema, removes force/overwrite behavior, and records actual cache access. Existing cache provenance is only size+sha16, so A0 is locked behind the independent full-SHA256 manifest produced by `scripts/slurm/c01_hash_inputs.sbatch`; each exact 64-hex value is rechecked before `torch.load`.
- **Decision boundary:** A0 is pre-Stage0 and kill-only. Survival can only authorize the next same-pooling preparation step; it is not Stage0 success, Gate0→Gate1 advancement, safety disentanglement, or a performance claim. A negative result kills only the current endpoint route.
- **Execution state:** `READY_NOT_RUN`; `next=hash_preflight_then_runtime_smoke`; no job submitted, no manifest/result/decision artifact, and no new metric. The external static reviewer raw conclusion is pending reviewer-owned persistence to `TARGET_REVIEW_RAW.md`; this implementation does not fabricate or overwrite it.

### C01 A0 external Gate1 re-evaluation — REVISE, not run (2026-07-28)

| Requirement | Static finding | Evidence / consequence |
|---|---|---|
| Claim narrowed to readout-policy contrast | PASS | Config, canonical binding and record forbid prompt-only, safety/stance/discourse and end-to-end interpretations. |
| Random orthogonal control with identical block L2 | PASS | Six ex-ante non-45° rotations, full-set upper bound, per-angle paired bootstrap and rotation-family Holm are binding; theta-0/theta-45 algebra guards are explicit. |
| Tiny/small displacement audit | PARTIAL / BLOCKING | Norm distributions, epsilon, tiny fraction and train-defined small subset exist, but fix concentration is measured only versus `endpoint_concat`; final net-fix and gain gates use the dynamically strongest ordinary control. |
| Same-pooling authorization semantics | PASS | `CONTINUE_SAME_POOLING_CACHE_ONLY` authorizes only matched cache preparation; `KILL_CURRENT_ENDPOINT_ROUTE_ONLY` does not falsify same-pooling policy contrast. |

- **Reviewer:** `/root/idea_reviewer`, continuous C01 A0 method review Round 1 plus Gate1 re-evaluation Round 2; full raw text is appended in `TARGET_REVIEW_RAW.md`.
- **Gate1 scores:** Problem Fidelity `9.2`, Method Specificity `9.0`, Contribution Quality `6.5`, Frontier Leverage `8.0`, Feasibility `9.3`, Validation Focus `8.0`, Venue Readiness `5.8`, overall `8.1`.
- **Verdict:** **REVISE.** The sole method blocker is reference consistency: retain the endpoint-concat mechanism diagnostic, but also compute the small-displacement gain-concentration gate against the exact `strongest_control_name` used by the decision.
- **Authorization boundary:** no hash preflight/runtime submission is authorized by this review. After the one-reference repair and fresh static check, the code may become `READY_TO_RUN_A0`; only an actual all-gates A0 pass can authorize same-pooling neutral/policy cache extraction.
- **Execution evidence:** no script was run, no job submitted, no cache/test path opened, and no manifest/result/decision/new metric was created during this external review.

### C01 A0 strongest-control reference repair — implemented, pending reviewer confirmation (2026-07-28)

- **Sole blocker repaired:** one shared selector chooses the strongest frozen ordinary control by accuracy, then macro-F1, then frozen `gain_controls` order; both net fixes and the small-displacement concentration gate consume that exact name.
- **Diagnostic retained:** the parallel `endpoint_concat` concentration remains `diagnostic_only`; no gate reads it.
- **No circular dependency:** selection reads only frozen ordinary-control metrics, excludes primary `common_displacement`, and does not read the small-displacement outcome.
- **Binding/schema:** config, canonical binding, result checks, and decision artifact schema bind the rule, per-dataset selected reference, and endpoint-concat role.
- **State:** `READY_NOT_RUN_PENDING_REVIEW_CONFIRMATION`; `next=fresh_external_static_review_then_hash_preflight_then_runtime_smoke`. No reviewer approval is fabricated.
- **Execution evidence:** static checks only; no Python/pytest, SLURM job, cache/test access, manifest, result, decision, or metric.

### C01 full-SHA256 hash preflight — COMPLETED and validated, A0 not run (2026-07-28)

- **Pre-submit gates:** the exclusive namespace was absent, `squeue -u jehc223` had no active job, and `scripts/slurm/c01_hash_inputs.sbatch` contained no `--time`, GPU request, disk guard, or force path.
- **Execution:** exact command `sbatch scripts/slurm/c01_hash_inputs.sbatch` submitted job `13710` once. It remained `PENDING (JobHeldUser)` until automatic release; no manual release or resubmission was performed. Accounting is `COMPLETED`, exit `0:0`, elapsed `00:00:01`, start/end `2026-07-28T22:03:29+12:00` / `2026-07-28T22:03:30+12:00`.
- **Logs:** `/data/jehc223/RGCL/slurm/logs/c01_hash_13710.out` is 239 bytes; `/data/jehc223/RGCL/slurm/logs/c01_hash_13710.err` is empty. No NaN, traceback, error, exception, or missing-file pattern was found.
- **Manifest validation:** `artifacts/c01_policy_contrastive/v1/hash_preflight/C01-HASH-v1/full_sha256_manifest.json` exists, is 6051 bytes, and has SHA256 `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305`. Read-only shell/jq checks verified the exact eight `MHC_zh`/`HateMM` × `train`/`dev_seen` × `standard`/`oneword` records, eight 64-hex full hashes matching the registered sha16 prefixes, eight successful ledger entries, and `test_like_attempt_count=0`, `test_like_open_count=0`.
- **Boundary:** this is provenance-only preflight evidence. No A0 analysis job, result, decision, metric, test access, Stage0 success, or Gate0→Gate1 advancement is claimed; the existing A0 scientific verdict is unchanged.

### C01 A0 runtime — HALT_FAIL_CLOSED_NO_DECISION (2026-07-28)

- **Environment routing:** resumed target loop as `whoami=jehc223`, `USER=jehc223`, therefore `execution_policy=slurm_only`; the existing project wrapper is mandatory.
- **Pre-submit validation:** the full-SHA manifest exists with exact SHA256 `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305`; A0 namespace `artifacts/c01_policy_contrastive/v1/a0/C01-A0-v1` and its `C01_A0_OUT.json` / `C01_A0_DECISION.json` are absent; `squeue -u jehc223` is empty; `scripts/slurm/c01_a0_cpu.sbatch` has no time/GPU/disk-guard/force path.
- **Authorization boundary:** run exactly `sbatch scripts/slurm/c01_a0_cpu.sbatch` once and monitor it to terminal. No test access, manual release, resubmission, or next-candidate job is authorized.
- **Execution:** exact command `sbatch scripts/slurm/c01_a0_cpu.sbatch` submitted job `13712` once. It ran on CPU only and terminated `FAILED`, exit `1:0`, elapsed `00:00:53`, start/end `2026-07-28T22:07:46+12:00` / `2026-07-28T22:08:39+12:00`; no release, resubmission, repair, or other job occurred.
- **Fail-closed guard:** stderr reports `HateMM/train/standard/img has 1 rows at/below epsilon 1e-12; first=355`. The row-norm guard first checks finiteness, so this is a finite zero/tiny-row integrity failure rather than NaN/Inf divergence. The exception occurred while preparing HateMM views, before HateMM historical parity, global bootstrap/Holm, decision construction, or publication.
- **Logs/output scale:** stdout is empty; stderr is 1218 bytes with SHA256 `355464ba3cdf965e697c4c48050c306c5eaf3695a092ba2540629abee49860ee`. The run namespace exists but is empty (directory size 10 bytes); `C01_A0_OUT.json` and `C01_A0_DECISION.json` do not exist.
- **Raw metrics:** unavailable for both datasets. No durable R0, endpoint, average, concat, common, displacement, `common_displacement`, `common_interaction`, rotation, primary delta, bootstrap/Holm, fix/break/net, shuffle-p95, small-displacement, or fired-rule values were published. MHC_zh must also remain unreported: control flow reached HateMM only after its in-memory dataset analysis returned, but the fail-closed exception discarded the uncommitted aggregate.
- **Parity/test audit:** startup canonical historical binding passed; MHC_zh per-dataset history/algebra checks are only control-flow-inferred and not persisted; HateMM history parity was not reached. No test-like path appears in the logs, the registered manifest contains only train/dev_seen, and the canonical loader rejects test-like paths before `torch.load`; the final runtime access ledger was not persisted because publication was never reached.
- **Decision:** no formal CONTINUE/KILL artifact exists. Operational state is **`HALT_FAIL_CLOSED_NO_DECISION`**, not `KILL_CURRENT_ENDPOINT_ROUTE_ONLY`. A red Feishu hard-blocker notification was delivered successfully (HTTP 200). Human adjudication is required before any prospective repair; the target remains unmet, Gate 0 is unpassed, and no next candidate was submitted.

### C01 zero-contract probe — COMPLETED_VALIDATED (2026-07-28)

- **Review and pre-submit gates:** the final scoped re-review was `GO (0 Critical / 0 High / 0 Important)`: `structural_zero` was absent, the neutral enum was `normal_nonzero|exact_zero|tiny_nonzero|nonfinite`, and `allow_zero_block_in_a0=false` remained frozen. Immediately before submission, the exclusive artifact namespace was absent, `squeue -u jehc223` was empty, the approved whole-manifest SHA256 exact-matched `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305`, and the wrapper contained no time/GPU/disk-guard/force path.
- **Execution:** exact command `sbatch scripts/slurm/c01_zero_contract_probe.sbatch` was submitted once as job `13717`. The initial `PENDING (JobHeldUser)` state cleared automatically; no release or resubmission occurred. The job completed `0:0`, start/end `2026-07-28T22:46:56+12:00` / `2026-07-28T22:46:59+12:00`, elapsed `00:00:03`, with `2` allocated CPUs and `4G` requested memory.
- **Logs/artifact:** stdout `/data/jehc223/RGCL/slurm/logs/c01_zero_probe_13717.out` is 322 bytes, SHA256 `b8faeebbd9001fdcdc170fa65b0c0e44bb235e2c059ab83d9cd777938b346770`; stderr `/data/jehc223/RGCL/slurm/logs/c01_zero_probe_13717.err` is empty, SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. The valid exclusive artifact `artifacts/c01_policy_contrastive/v2/zero_contract_probe/C01-ZERO-PROBE-v1/zero_contract_probe.json` is 18,090 bytes, SHA256 `bee4964ce7e4ca81cfdb72c3859f78196568badf982aef587bc14ee6dbe63526`.
- **Input/access contract:** exactly eight registered MHC_zh/HateMM train/dev_seen standard/oneword caches were hashed and exact-matched to the approved manifest before `torch.load`; all eight ledger entries have `sha256_matched_before_torch_load=true`, `torch_loaded=true`, and `test_like=false`. Test-like attempts/opens are `0/0`, no feature vectors are serialized, raw IDs remain strings, and `exact_zero` remains observation-only.
- **Per-cell observations:** every MHC_zh train/dev_seen × standard/oneword × img/text cell has `zero=0, tiny_nonzero=0, nonfinite=0`. HateMM dev_seen is also all-zero across those three anomaly counts. HateMM train has exactly one exact-zero row in each of its four policy/modality cells: row `355`, ID `hate_video_95`, label `1`; both image and text are exact-zero under both standard and one-word policies, and each zero-row record reports the other modality as `exact_zero`. All HateMM train tiny-nonzero and non-finite counts are `0`.
- **Endpoint masks:** standard versus one-word exact-zero masks match for every dataset/split/modality. The matched ID set is empty for all MHC_zh cells and HateMM dev_seen; for HateMM train it is exactly `["hate_video_95"]` in both img and text. Every standard-only and one-word-only set is empty.
- **v2 necessary conditions and boundary:** `endpoint_zero_masks_exact_match=true`, `non_structural_tiny_absent=true`, and `nonfinite_rows_absent=true`. These are necessary, not sufficient: historical-baseline same-null consumption remains `REPORTED_EXTERNAL_NOT_VERIFIED_BY_PROBE`, with its frozen path/line evidence outside this eight-cache probe. Therefore `allow_zero_block_in_a0=false` remains binding; no A0 science/config change, A0 retry, next-candidate submission, or scientific CONTINUE/KILL decision is authorized by this diagnostic.

### C01 A0 v2 aligned structural-null adjudication — GO_CONTRACT_ONLY / NO_GO_EXECUTION (2026-07-28)

- **Problem Anchor:** A0 remains a pre-Stage0 kill-only paired readout-policy endpoint audit on frozen train-memory → dev-query caches. It still cannot support prompt-only causality, safety/discourse disentanglement, Stage0 success, Gate0→Gate1 advancement, or a performance claim.
- **Evidence closure:** independent read-only review exact-matched the probe artifact identity/SHA256 `bee4964ce7e4ca81cfdb72c3859f78196568badf982aef587bc14ee6dbe63526` and job `13717` log, confirmed the unique four-way aligned HateMM/train row 355 `hate_video_95` exact-zero, traced it to Decord+PyAV decode failure and the extractor's all-cell zero guard, and confirmed historical deployed R0 consumption from the full-cache bit-exact record in `refine-logs/READOUT_SUBMIT_RECORD.md:166-185`.
- **Verdict:** a prospective v2 may preserve exact zero only for the evidence-bound tuple `(HateMM, train, hate_video_95, row 355, standard/oneword × img/text)`. This is a no-information historical-parity sentinel, never a general `allow_zero=true`; label 1 is integrity-only and cannot construct the permission mask.
- **Binding v2 guards:** all other exact-zero/tiny/nonfinite/mismatched rows fail closed; the authorized mask must remain exact through every raw/fused/derived/rotation arm; 256 shuffles must fix the sole null and permute only non-null IDs; the null must never enter any top20; with-null versus remove-null scores, predictions and gate booleans must be exact-equal; hash/identity drift and every negative fixture stop before decision publication.
- **Comparability/gates:** primary R0 and every arm retain the same row/order/exact-zero, so historical R0 comparability is preserved iff the full contract passes. All existing `+0.020/+0.020`, harm, bootstrap/Holm, net-fix, fixed-rotation, shuffle-p95 and strongest-control small-displacement thresholds remain unchanged; new validity guards are HALT-only.
- **Authorization boundary:** `GO_CONTRACT_ONLY / NO_GO_EXECUTION`. Existing v1 `allow_zero_block_in_a0=false` remains active. Only prospective v2 config/code/schema/preregistration implementation followed by fresh independent static review is next; no v2 run, retry, result, metric, CONTINUE/KILL, test access, or next-candidate job is authorized or prewritten. Full verbatim review is appended in `TARGET_REVIEW_RAW.md`.

### C01 A0 v2 contract implementation — V2_READY_NOT_RUN_PENDING_REVIEW (2026-07-28)

- **Implemented lineage:** new `C01-A0-v2` config and CPU wrapper target the fresh exclusive namespace `artifacts/c01_policy_contrastive/v2/a0/C01-A0-v2/`; the canonical analysis accepts only v2. The v1 config/wrapper, failed job `13712`, and empty v1 namespace remain historical and are not reused or overwritten.
- **Exact evidence binding:** v2 requires the frozen eight-cache manifest SHA256 `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305` and zero-probe SHA256 `bee4964ce7e4ca81cfdb72c3859f78196568badf982aef587bc14ee6dbe63526`. The only accepted exact-zero tuple is HateMM/train `hate_video_95`, row 355, integrity-label 1, standard/oneword × img/text.
- **HALT-only validity:** zero-preserving normalization must keep the exact registered mask through every raw/derived/fused/rotation representation; all other zero/tiny/nonfinite/mismatch cases halt. Every label-blind shuffle fixes index 355 and bijects the remaining indices. Every real and shuffled retrieval audits FAISS top-20 indices; the null must never appear. Rebuilt retrieval requires identical dtype/shape/C-order bytes plus per-side SHA256 for indices, similarities, scores and predictions; metrics require identical canonical typed IEEE-754 serialization/hash, with signed zero distinct and NaN/Inf forbidden.
- **Scientific gates unchanged:** strict R0 parity, frozen rotations, 256 shuffle draws, 2,000 paired bootstraps, Holm correction, small/tiny-displacement gates, `+0.020/+0.020`, and net fixes `+2/+3` are unchanged. New guards may only HALT and can never produce CONTINUE evidence.
- **State/boundary:** `V2_READY_NOT_RUN_PENDING_REVIEW`; next is a fresh independent static review. No Python, SLURM submission, cache/test access, v2 namespace creation, result, decision, or new metric occurred, and execution remains unauthorized.

### C01 A0 v2 review repair — 1 High + 1 Important closed statically (2026-07-28)

- The HateMM registered train null is excluded from the displacement train quantile, every displacement distribution/tiny denominator, and the final small-row gate. A retained-array masked route and a physically deleted-row route independently recompute threshold, dev small mask, tiny fractions/counts, `small_rows_dominate_fixes`, and the final boolean; byte/typed exact disagreement HALTs.
- Retrieval exactness now means dtype + shape + C-order bytes, with SHA256 for both sides. Metric exactness uses canonical sorted typed IEEE-754 binary64-hex JSON and payload hashes. `+0.0/-0.0` differ; NaN/Inf is forbidden. Raw/derived nulls remain numeric exact-zero row-mask checks and are not described as byte comparisons.
- Score/prediction/metric invariance plus displacement dual-path invariance prevents the null from influencing bootstrap/Holm inputs, net fixes, quantiles, or decisions. Scientific thresholds remain unchanged. State remains `V2_READY_NOT_RUN_PENDING_REVIEW`; no Python or job ran.

### C01 A0 v2 runtime — HALT_FAIL_CLOSED_NO_DECISION (2026-07-29)

- **Review/authorization:** the scoped re-review closed the earlier 1 High + 1 Important with `GO (0 Critical / 0 High / 0 Important)`. The main dialogue then explicitly authorized only `scripts/slurm/c01_a0_cpu_v2.sbatch`; no C02 or later job was authorized.
- **Pre-submit gates:** `squeue -u jehc223` was empty; `artifacts/c01_policy_contrastive/v2/a0/C01-A0-v2`, result and decision were absent; the manifest and zero-probe exact-matched SHA256 `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305` and `bee4964ce7e4ca81cfdb72c3859f78196568badf982aef587bc14ee6dbe63526`; the CPU wrapper used HateVideo, `8 CPU / 32G`, no GPU, no `--time`, no dependency/array/singleton/chain, and no force/release path.
- **Execution:** exact command `sbatch scripts/slurm/c01_a0_cpu_v2.sbatch` was submitted once as job `13730`. `scontrol` showed `Dependency=(null)`, `SubmitLine=sbatch scripts/slurm/c01_a0_cpu_v2.sbatch`, `AdminComment=PENDING_APPROVAL`, and CPU/memory-only `ReqTRES`; the initial `JobHeldUser` cleared automatically after 32 minutes 16 seconds. No manual release, resubmission, array, chained job, or later candidate occurred.
- **Terminal state:** job `13730` ran `2026-07-29T00:01:09+12:00` to `00:02:20+12:00` and terminated `FAILED`, exit `1:0`, elapsed `00:01:11`. The exact fail-closed message is `real/endpoint_std with-null/remove-null retrieval mismatch`, raised by `retrieval_without_registered_null` before publication. The log does not persist whether neighbor IDs, similarities, or scores were the first unequal byte-level component, so no more specific cause is claimed.
- **Logs/artifacts:** stdout `/data/jehc223/RGCL/slurm/logs/c01_a0_v2_13730.out` is empty with SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; stderr `/data/jehc223/RGCL/slurm/logs/c01_a0_v2_13730.err` is 1,245 bytes with SHA256 `fd74537da9cdd2cb435bf160328b65289d4d9a8e2ea2ef5c654f764f22eaf5b4`. The v2 namespace exists but is empty (10 directory bytes); `C01_A0_OUT.json` and `C01_A0_DECISION.json` do not exist.
- **Unavailable scientific outputs:** no durable raw metric exists for MHC_zh or HateMM, including R0/endpoint standard, endpoint one-word, average-score, endpoint-concat, common, displacement, common-displacement, common-interaction, orthogonal rotations, or shuffled-pair controls. Primary/control deltas, paired bootstrap, Holm families, fix/break/net, rotation bounds, shuffle p95/p-values, displacement threshold/tiny/small-row audits, aggregate null guards, fired scientific decision rules, dataset pass values, and CONTINUE/KILL verdict are all unavailable. MHC_zh in-memory work, if completed before the later HateMM-only registered-null comparison, was not published and must not be reported.
- **Boundary:** this is an operational **HALT_FAIL_CLOSED_NO_DECISION**, not a scientific KILL. No test-like path appears in the logs, and only canonical train/dev caches are reachable, but a final runtime ledger was not published. A red Feishu runtime-error notification succeeded with HTTP 200. No retry, repair, C02 submission, or other follow-up job is authorized.

### C01 job 13730 retrieval-equivalence debug probe — DEBUG_READY_NOT_RUN (2026-07-29)

- **Static diagnosis:** A0 creates a fresh `IndexFlatIP` per full/reduced search, adds memory once, removes original index 355, and maps with `flatnonzero(keep)[reduced_neighbors]`; no evident mapping bug was found. The failed component was not persisted, so tie order, boundary candidates and matrix-shape-dependent float variation remain unresolved.
- **Prepared probe:** a read-only 8CPU/32G probe with `OMP/MKL/OPENBLAS/NUMEXPR=8` exactly matches failed A0 job 13730's compute/thread shape, exact-binds the manifest, zero-probe and unique null tuple, then compares only HateMM standard endpoint_std train→dev_seen. The repaired key path imports A0's `l2_rows`/`fuse_modalities`, independently reconstructs the exact block-normalize → concatenate → final-normalize path and requires byte parity. Its ordered-float32 ULP map is sign-aware with known-bit runtime self-checks, NaN/Inf forbidden, and `RAW_FAISS_TIE_ORDER` requires both stable-neighbor and stable-similarity-byte identity; other raw-neighbor disagreements are mixed/float variation. It reports only limited mapping/null/top20/set/order/ULP/score/prediction/metric/tie evidence.
- **Boundary:** `DEBUG_READY_NOT_RUN`. Test paths are hard-blocked; features, text and full arrays are not serialized. A0 was not changed, Python was not run, no job/result exists, and execution requires fresh review plus explicit authorization.

### C01 job 13730 retrieval-equivalence debug probe — COMPLETED_DIAGNOSED (2026-07-29)

- **Scoped re-review:** the four requested repairs passed static review with `GO (0 Critical / 0 High / 0 Important)`. The runtime imports A0 `l2_rows`/`fuse_modalities`, independently reconstructs the same four endpoint-standard steps, and requires dtype/shape/C-order-byte parity for both train and dev_seen. The sign-aware ordered-float32 mapping passed its known-bit, monotonicity, signed-zero, cross-zero and nonfinite self-checks. `RAW_FAISS_TIE_ORDER` requires both stable-neighbor and stable-similarity-byte identity. The wrapper exactly matches job 13730 at `8 CPU / 32G` and `OMP/MKL/OPENBLAS/NUMEXPR=8`.
- **Pre-submit and execution:** immediately before submission the exclusive namespace/artifact were absent, `squeue -u jehc223` was empty, manifest and zero-probe SHA256 exact-matched `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305` and `bee4964ce7e4ca81cfdb72c3859f78196568badf982aef587bc14ee6dbe63526`, both standard HateMM caches exact-matched the approved manifest, and the wrapper had no GPU, `--time`, dependency, array, singleton, chain, force or release path. Exact command `sbatch scripts/slurm/c01_retrieval_equivalence_probe.sbatch` submitted job `13732` once. `Dependency=(null)` and CPU/memory-only resources were confirmed. `JobHeldUser` cleared automatically after `00:17:36`; no manual release or resubmission occurred.
- **Terminal/log/artifact:** job `13732` completed `0:0`, start/end `2026-07-29T00:46:32+12:00` / `00:46:35+12:00`, elapsed `00:00:03`. Stdout is 362 bytes, SHA256 `7b4f221273916a9f50d32947263242bebdb33c8e6bf2da39be13cdda673523d7`; stderr is empty, SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. The exclusive 29,347-byte artifact `artifacts/c01_policy_contrastive/v2/retrieval_equivalence_probe/C01-RETRIEVAL-EQUIV-PROBE-v1/retrieval_equivalence_probe.json` has SHA256 `724c87cd2fbdb763180b663bc6492322887bc2077f378c5b21c4184c4ba80e6f`.
- **Key/ULP/mapping/null validity:** imported-A0 and independent-local train keys are byte-identical `<f4 [744,7168]>`, SHA256 `d14fedf9523dcd59acbcede6e6caba47177ecd5feb9d3d19fc39785baf294712`; dev_seen keys are byte-identical `<f4 [107,7168]>`, SHA256 `e3e99a77578525e654cd69d78077d3efa5d7eada327440c3ec0fa1aeb3c15ff2`. The local-to-original formula is exact for all 743 retained rows. The registered null occurs in zero top-20 positions across zero queries.
- **Concrete root cause:** full-memory `744×7168` and physically removed `743×7168` CPU `IndexFlatIP` searches return exactly the same raw top-20 neighbor array and, after deterministic sorting, exactly the same stable neighbor array: zero element, set or order differences. Similarity dtype/shape match but C-order bytes do not: 22 float32 elements differ, maximum absolute difference `2.980232238769531e-7`, maximum ordered distance `5 ULP`. Raw and stable comparisons show the same similarity discrepancy. Therefore the frozen diagnosis is **`FLOAT_VARIATION_WITH_STABLE_NEIGHBOR_IDENTITY`**: matrix-shape-dependent FAISS floating-point variation, not mapping, null selection, candidate-set change, or raw tie order.
- **Downstream invariance:** raw and stable with/remove scores each differ numerically in 20 of 107 rows, maximum absolute difference `2.2706531321858847e-8`. Predictions are byte-identical with zero differences. Canonical metrics are byte-identical on both routes: accuracy `0.8411214953271028`, macro-F1 `0.8390977443609022`, ROC-AUC `0.9164244186046512`. Both searches have 16 exact adjacent ties and 17 adjacent pairs within one ULP over top-21; both have one exact rank-20/rank-21 boundary tie (`non_hate_video_190`) and minimum boundary gap zero, but no neighbor set/order change.
- **Boundary:** this artifact diagnoses why job 13730's exact-byte HALT guard fired; it is diagnostic-only and does not retroactively create an A0 result or CONTINUE/KILL verdict. The A0 strict byte-equivalence guard is incompatible with these observed finite shape-dependent similarity/score differences even though neighbor identity, predictions and metrics are invariant. Any prospective guard repair and A0 retry require separate review and explicit authorization. No A0 source/config was modified, no retry/C02/follow-up job was submitted, test-like attempts/opens are `0/0`, and no feature vector was serialized. The configured green Feishu completion notification returned success with HTTP 200.

### C01 retrieval numerical-equivalence guard adjudication — GO_CONTRACT_ONLY / NO_GO_EXECUTION (2026-07-29)

- **Problem Anchor/evidence:** A0 remains the same pre-Stage0 kill-only endpoint audit. Independent review accepted artifact SHA256 `724c87cd2fbdb763180b663bc6492322887bc2077f378c5b21c4184c4ba80e6f` as evidence that 744→743 physical removal preserves A0 key bytes, mapping, null top20=0, raw/stable neighbor identity/order, prediction bytes and canonical metrics, while only 22 finite float32 similarities (`max_abs=2.980232238769531e-7`, `max_ulp=5`) and 20 finite scores (`max_abs=2.2706531321858847e-8`) vary.
- **Guard verdict:** replace prospective v2 similarity/score byte equality with a formula-derived numerical-equivalence guard, never an observed-value `allclose`. Retained normalized operands, mapping, raw/stable neighbors/order, labels, predictions, canonical accuracy/macro-F1/ROC-AUC, scientific gate booleans and no-NaN/Inf remain exact; null top20 remains exactly zero.
- **Derived bounds:** with `u32=2^-24`, actual arm dimension `d`, audited operand scale `rho`, use `gamma32(d)=d*u32/(1-d*u32)` and power-of-two `B_sim=2^ceil(log2(2*gamma32(d)*rho))`; ordered-ULP allowance is the finite code-count induced by outward-rounded `[a-B_sim,a+B_sim]`. With weights 20→1 and sum 210, require per query `B_score=sum(w*abs(delta_similarity))/210 + 2^-45`, arm maximum `<=B_sim+2^-45`, and a cutoff interval that cannot cross zero. Exceeding any numeric or exact invariant HALTs.
- **Necessary ablation/objection:** deterministic binary64 re-scoring over exact agreed top20 IDs and byte-identical normalized operands is binding. This addresses the strongest objection that a post-hoc tolerance fitted to `endpoint_std` could hide arm/rotation/shuffle or cutoff instability; every arm/control must independently pass the formula-derived envelope and exact discrete invariants.
- **Versioning/boundary:** because job `13730` executed frozen v2 semantics, the repair requires both new config and full v3 identity: `C01-A0-v3`, `configs/c01/c01_a0_v3.json`, v3 result/decision schemas, wrapper and `artifacts/c01_policy_contrastive/v3/a0/C01-A0-v3`. V1/v2 remain immutable. Scientific CONTINUE/KILL thresholds are unchanged. `GO_CONTRACT_ONLY / NO_GO_EXECUTION`: no implementation/static approval, v2 retry, v3 submission, C02, test access, result or decision is authorized or prewritten. Full raw review is appended in `TARGET_REVIEW_RAW.md`.

### C01 A0 v3 numerical-equivalence implementation — V3_READY_NOT_RUN_PENDING_REVIEW (2026-07-29)

- **Frozen scientific base:** the new v3 config SHA-binds the complete v2 config (`f3997bdd...63f5`) and v2 source (`d2b9c2...b855`). Runtime validates v2's full canonical binding first and permits only v3 run/schema/namespace/output identity plus the HALT-only numerical-equivalence definition. R0/history parity, all real/control arms, six rotations, 256 fixed-null shuffles, 2000 bootstraps, Holm families, shuffle p95, small-displacement, `+0.020/+0.020`, deployed-R0 and net-fix gates remain exact.
- **Implemented guard:** retained raw/normalized operands, mapping, null top20=0, raw/stable neighbors and per-query set/order, neighbor labels, predictions, canonical metrics and deterministic scientific-boolean basis remain exact. Similarities use actual `d`, `u32=2^-24`, `gamma32(d)`, upward-safe binary64 `rho`, next-power-of-two `B_sim`, and outward-rounded exponent-aware ordered-ULP bounds. Scores use the frozen 20→1/210 propagation plus `2^-45` and strict cutoff-interval stability. Every audit reports derivation, observed maxima and bound ratios; any failure raises `HALT_NUMERICAL_EQUIVALENCE`.
- **Independent reference:** every real arm, rotation and fixed-null shuffle re-scores only the exact agreed float32 top-20 IDs using a frozen chunked C-order binary64 multiply/add-reduce algorithm. It does not mine or replace neighbors; reference predictions/canonical metrics stay exact and FAISS/reference residuals must remain inside the prospective float32 envelope.
- **Boundary:** files are `scripts/analysis/c01_policy_contrast_a0_v3.py`, `configs/c01/c01_a0_v3.json`, `scripts/slurm/c01_a0_cpu_v3.sbatch`, and `refine-logs/C01_A0_V3_RECORD.md`; namespace is new and exclusive. No Python or SLURM execution occurred, no result/decision/metric exists, and execution remains unauthorized pending fresh independent static review.
- **Static NO-GO repair:** the v2→v3 whitelist now covers both numerical schema strings without exposing scientific fields. The acyclic frozen chain is source `40b35eee...4dfb` → config/v2/review/diagnostic, wrapper `e61b9962...af99` → source+config, record `3af07f73...38fb` → wrapper, and TARGET → record. Binary64/FAISS similarity and score zero masks plus zero signbits are exact-or-HALT; the authorized tuple is semantically equal to frozen v2 and every diagnostic field; decision validation now covers the complete v2/v3 provenance/type/schema/small-displacement/continue/result contract; and every public guard scalar is recomputed from runtime caches/audits/counters/ledger rather than overwritten with success. Formulas and scientific arms remain unchanged. Status remains `V3_READY_NOT_RUN_PENDING_REVIEW`; no execution is authorized.

### C01 A0 v3 job 13735 — HALT_FAIL_CLOSED_NO_DECISION (2026-07-29)

- **Scoped re-review and authorization:** the requested provenance, signed-zero, authorized-tuple, decision-schema, runtime-derived-public-guard and frozen-science checks were statically accepted as `GO (0 Critical / 0 High / 0 Important)`. In hindsight that verdict was incomplete: it missed the cross-function aggregate schema bug below and is invalidated by the deterministic runtime failure. Immediately before the one authorized submission, the v3 namespace/result/decision were absent, `squeue` was empty, all pinned source/config/wrapper/record/v2/review/diagnostic/manifest/probe hashes matched, and the 8CPU/32G wrapper had no GPU, time limit, dependency, array, singleton, chain, force or release path.
- **Execution:** exact command `sbatch scripts/slurm/c01_a0_cpu_v3.sbatch` submitted job `13735` once. It waited `03:20:18` in `JobHeldUser` and was released automatically; no manual release or resubmission occurred. Start/end were `2026-07-29T05:07:21+12:00` / `05:12:31+12:00`, elapsed `00:05:10`, state `FAILED`, exit `1:0`, total CPU `06:21.604`, batch MaxRSS `1594260K`, and MaxVMSize `5278300K`. No NaN, Inf, OOM or resource-pressure message appeared.
- **Logs and fail-closed point:** stdout is 81 bytes, SHA256 `cf2a95043ca98139756f42a93693869184c111c577c58168ba5c7987435c9124`, and records successful source/config SHA checks. Stderr is 974 bytes, SHA256 `9271e642fb6f0fd85265cf9fd4633432647c5ca49659a4cb2eb431f950c92cf6`, ending at `HALT_NUMERICAL_EQUIVALENCE: HateMM: derived public contract summary failed`.
- **Exact root cause:** HateMM's `retrieval_null_influence` contains `avg_score` with `status=PASS`, but `avg_score_equivalence` returns no `registered_null_top20_count` field (`c01_policy_contrast_a0_v3.py:1512-1539`). The runtime-derived aggregate nevertheless applies `status == NO_REGISTERED_NULL or registered_null_top20_count == 0` to every retrieval audit (`:1998-2002`). Thus the aggregate treats the non-retrieval average-score audit's absent field as failure, derives `registered_null_absent_from_all_top20=false`, and halts at `:2118`. MHC_zh uses `NO_REGISTERED_NULL`; HateMM deterministically exposes the mismatch. This is a guard-schema implementation blocker, not evidence that a numerical/scientific threshold failed.
- **Outputs and boundary:** the exclusive namespace exists but is empty (10 directory bytes); no result or decision exists. Because publication is atomic after both datasets pass the public guard, no durable raw MHC_zh/HateMM metric, public guard, fired-rule list, dataset pass or CONTINUE/KILL verdict is available and none is inferred from in-memory work. The outcome is operational **HALT_FAIL_CLOSED_NO_DECISION**, not scientific KILL. Source/config/wrapper/record and v1/v2 remain frozen; no repair, retry or C02/follow-up job was submitted or authorized.

### C01 A0 v4 typed-audit schema repair — V4_READY_NOT_RUN_PENDING_REVIEW (2026-07-29)

- **Scope and predecessor freeze:** v4 exact-binds the frozen v3 config/source/wrapper/record hashes `4ddb0f6f...cda5`, `40b35eee...4dfb`, `e61b9962...af99`, and `3af07f73...38fb`; job `13735` stdout/stderr and `TARGET_REVIEW_RAW.md` correction lines 1340–1363 are also exact-bound. The v3 namespace remains empty with no result/decision. V3 bytes and its failed execution history were not modified.
- **Typed union repair:** every audit now carries `audit_kind`, explicit `registered_null_top20_count`, and applicability semantics. A real registered-null retrieval requires exact integer-not-bool count `0`. The non-retrieval `avg_score` requires count `null` and a source proof linking exactly `endpoint_std` and `endpoint_ow`, each a direct retrieval with integer count `0`; no false direct count is fabricated. `NO_REGISTERED_NULL` requires explicit occurrence count `0` with not-applicable semantics. Exact top-level keys, critical nested evaluation/reference keys, literal boolean pass values, discriminator, types, and cross-arm linkage are fail-closed.
- **Aggregation/self-test:** public and final HALT-only guards consume typed summaries and do not use permissive missing-field defaults. The frozen shuffle producer is wrapped so each emitted audit is schema-validated before its legacy counter and the returned aggregate is exact-key/type checked. A required pure schema self-test runs after SLURM enforcement but before namespace creation and covers valid HateMM registered-null/direct+derived and valid MHC-ZH no-null paths, plus rejection of missing/wrong average count, nonzero source count, and missing no-null count. The self-test is embedded but was not run on the login node.
- **Frozen v4 chain:** config `configs/c01/c01_a0_v4.json` SHA256 `2d9488e6f9af6be00d500d1c2f13912fd4be0ab9439608d33b0857178efe7ca6`; analysis `scripts/analysis/c01_policy_contrast_a0_v4.py` SHA256 `3c545eed876f97aa05f3e85375430bedf8e63226c70f3ee8ea12da02e9bf5514`; wrapper `scripts/slurm/c01_a0_cpu_v4.sbatch` SHA256 `9ae7f10370114647063f5ab18d97e6080ff8f90543a6e74c7b4fc24494bbf107`; record `refine-logs/C01_A0_V4_RECORD.md` SHA256 `916074b875ff3ea5b5bbe920da31006c35b582892f617ef65de2942fa403d136`. The prospective namespace `artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4` is absent.
- **Unchanged science/boundary:** v4 exact-compares the complete v3 numerical contract and changes no representation, retrieval operation, arm, rotation, shuffle, bootstrap, Holm family, displacement rule, metric, threshold, gain/net-fix decision, finite bound, signed-zero rule, or binary64 reference. Static preparation used only JSON/Bash/hash/text/diff checks. No Python, job, test/cache access, result, decision, metric, retry, or C02 action occurred. State is `V4_READY_NOT_RUN_PENDING_REVIEW`; execution requires fresh independent review and separate authorization.

### C01 A0 v4 job 13738 — COMPLETED / KILL_CURRENT_ENDPOINT_ROUTE_ONLY (2026-07-29)

- **Review, preflight and execution:** after independent static `GO (0 Critical / 0 High / 0 Important)`, the actual v4 config/source/wrapper/record hashes exactly matched `2d9488e6...7ca6`, `3c545eed...5514`, `9ae7f103...f107`, and `916074b8...d136`; the exclusive namespace/result/decision and prior v4 job record were absent; `squeue` was empty; the 8CPU/32G CPU-only wrapper had no time, GPU, dependency, array, singleton, chain, release or force path; and frozen v3 files, logs, empty namespace and job `13735` history were unchanged. Exact command `sbatch scripts/slurm/c01_a0_cpu_v4.sbatch` submitted job `13738` once. `JobHeldUser` cleared automatically after `01:35:08`; there was no release, resubmission or chain.
- **Terminal and artifacts:** job `13738` completed `0:0`, start/end `2026-07-29T08:32:40+12:00` / `08:38:45+12:00`, elapsed `00:06:05`, CPUTime `00:48:40`, TotalCPU `07:16.331`, MaxRSS `1638056K`, MaxVMSize `5318304K`. Stdout is 499 bytes, SHA256 `b7f8b481e34f877a57e89e792f18e4c1a354767f05e53fabe2340432e6cc3aef`; stderr is empty. Result `C01_A0_OUT.json` is 847,309 bytes, SHA256 `b45adf18...65f53`; decision `C01_A0_DECISION.json` is 15,922 bytes, SHA256 `670c768f...20fa`; the decision exact-pins the result hash.
- **Validity:** this is not an engineering HALT. The v4 fail-closed schema self-test passed 6/6; source/config/v3 lineage, scientific base, software/CPU/thread binding, 8/8 cache hashes and zero test-like opens passed. All 11 aggregate halt-only guards and all per-dataset guards are true. HateMM's registered null appears zero times in every real top-20 audit and across 512 shuffle-arm checks; both datasets pass 256 train/dev bijections and fixed-point draws with zero numerical, binary64-reference or scientific-boolean mismatches.
- **Raw primary evidence:** on MHC-ZH the primary `common_displacement` is accuracy `0.8589743589743589`, macro-F1 `0.8479532163742689`, ROC-AUC `0.9428571428571428`; the strongest ordinary control `endpoint_concat` is `0.8846153846153846 / 0.8773370609820024 / 0.9328571428571428`, so binding accuracy/F1 gains are `-0.02564102564102566 / -0.029383844607733467`. On HateMM the primary is `0.8598130841121495 / 0.8573713676352972 / 0.90625`; strongest ordinary control `common` is `0.8691588785046729 / 0.8671985815602836 / 0.9135174418604651`, giving `-0.009345794392523366 / -0.009827213924986422`.
- **Binding decision failures:** neither dataset has a single Holm rejection among the 12 primary-versus-ordinary-control accuracy/F1 hypotheses or the 12 primary-versus-rotation hypotheses; adjusted p-values are all `1.0`. MHC-ZH's best rotations reach accuracy/F1 `0.8974358974358975 / 0.8902181562280085`, and HateMM's best rotation reaches `0.8691588785046729 / 0.8671985815602836`, both above the primary. Net fixes versus the selected strongest controls are `-2` for MHC-ZH (required `+2`) and `-1` for HateMM (required `+3`). Gain, bootstrap, rotation and net-fix gates therefore fail on both datasets.
- **Evidence that passes but is insufficient:** the real-versus-shuffled-pair Holm family rejects all 8 accuracy/F1 hypotheses, both real arms are above the required shuffle p95 values, history/contract and displacement stability pass, maximum tiny fraction is zero, and small-displacement rows do not dominate fixes. This shows the pair structure is non-random, but it does not establish that the primary contrast is better than ordinary or equally normalized rotation controls.
- **Frozen conclusion/boundary:** formal decision is **`KILL_CURRENT_ENDPOINT_ROUTE_ONLY`**, `continue=false`, dataset passes are `MHC_zh=false`, `HateMM=false`. This retires only the current standard-L24 versus one-word-L24 endpoint-contrast route and does not falsify policy contrast under same-pooling caches. C01 is now frozen at this scientific KILL. C02 may enter design and independent review only; no C02 job, resource expansion or chained execution is authorized.

### C02 design and independent review — KILL_C02_DESIGN_COLLISION_OR_INFEASIBILITY (2026-07-29)

- **Problem anchor:** internal error evidence still supports a transcript/evidence-density nuisance: HateMM has opposite within-class transcript-volume effects and length-organized retrieval, MHC-EN degrades most in its longest quartile, and MHC-ZH has a significant non-monotone mid-length error concentration. P3, SAV, MECHFIX, and global length correction nevertheless already rule out simple pooling, one-direction excision, or scalar calibration.
- **Historical proposal:** EDQ-Orbit would contract same-video density views during training and use only the native full view at inference. The draft limited itself to two claims, a kill-only A0, and staged `+0.050/+0.050`, `+0.040/+0.040`, `+0.020/+0.020`, and final `+0.030/+0.030` gates.
- **First independent review:** `REVISE_DESIGN`. Existing P3 mean/soft/mild caches change image pooling while the EDQ method changes transcript density, so the proposed P3 `max_{a,b}` oracle was a proxy-target mismatch. Evidence-core deletion was also rejected as a label-preserving view because omitted context can reverse quotation, counterspeech, archive, lyric, or reportage meaning.
- **Read-only asset audit:** no HateMM+MHC-ZH train/dev representation bank exists for native versus exact full-transcript repeat, localized repeat, prefix/suffix repeat, echo, or another full-transcript-preserving density orbit. P3 is image-only, bidirectional caches are readout/attention variants, `nullop2merge` is a merge-path probe, and `curric-rep2` is an independent SFT draw rather than an input-repeat view.
- **Terminal adjudication:** the reviewer returned **`KILL_C02_DESIGN_COLLISION_OR_INFEASIBILITY`**. The frozen Stage-0 contract demands a representation-matched existing-bank oracle on two datasets before new extraction, but no such bank exists; creating it would violate that same gate. This kills C02 under the current registry contract without scientifically falsifying abstract EDQ.
- **Boundary:** no implementation/config/schema/wrapper, Python, cache opening, feature extraction, teacher call, GPU/test access, SLURM job, artifact, or metric occurred. C02 is frozen; only C03 design and independent review may proceed, with no C03 execution authorized.

### C03 design and independent review — KILL_C03_DESIGN_INFEASIBILITY (2026-07-29)

- **Problem anchor:** F92 closed training-free mask/readout repair; F93's published MNTP transplant produced the first same-sign bidirectional text recovery but damaged image geometry, collapsed stream diversity, inverted fusion from additive to destructive, and remained below the causal floor. The only scientifically live MNTP form was native training at the deployed task-LoRA weight point.
- **Prospective method boundary:** the historical design specified single-dataset train-only, label-blind policy presence/masks/loss weights, native policy-free inference, a plain-bidirectional image-preservation anchor, and matched `CAUSAL_MATCHED_COMPUTE`, `NATIVE_MNTP_ONLY`, REMOVE, SHUFFLE, NOISE and REMOVE_IMAGE_ANCHOR controls. The fixed policy prefix was explicitly limited to generic conditioning, not an identified per-example relation mechanism.
- **Asset audit:** no HateMM+MHC-ZH train/dev bank isolates native policy-conditioned MNTP under matched prompt, pooling, readout, fusion and actual fold/deployed-head path. F72 is mask-only, F92 is readout, F93 is an incompatible external transplant, F70/C01 confound prompt/pooling/readout, and `nullop2merge` is a numerical control. Old banks can strengthen a KILL but cannot PASS the missing policy-native axis.
- **Binding contradiction:** the registry requires a two-dataset representation-matched `+0.050/+0.050` Stage-0 oracle before teacher/GPU/extraction. Creating the missing bank would itself require native MNTP training and re-extraction, violating that ordering. The candidate is therefore design-infeasible under the current registry.
- **Independent review:** concept review was `REVISE (3 Critical / 5 High / 4 Important)`; after the formal proposal/plan/tracker/audit, the reviewer exact-matched the submitted hashes and returned **`CONFIRM_KILL (0 Critical / 0 High)`**. Two non-blocking wording points were incorporated into the final proposal.
- **Boundary:** this is not an experimental refutation of native policy-MNTP. No Python, cache opening, performance metric, teacher, GPU, test access, implementation or SLURM job occurred. All three active slots are now closed; reopen literature/novelty Gate 0 before C04. No C04 action is authorized here.

### C04 design and independent review — REVISE_USER_AMENDMENT_REQUIRED (2026-07-29)

- **Anchor and method:** C04 narrows the discourse problem to an ordered
  source→proposition→presenter-stance→protected-target-harm interaction. The
  proposed `SPaSH-Tensor` uses a train-only label-blind dense four-way tensor as
  privileged supervision for a native tensor student; final inference remains one
  native full-video embedding and ordinary train-memory top-20 kNN. It makes no
  scalar verdict, routing or test-time-teacher claim.
- **Read-only evidence:** HateMM has five stable quotation/archive/lyric/documentary
  slur false positives, but their break-free ceiling is only `+0.0233`. EN has five
  lexical-surface plus two counterspeech/meta false positives; ZH has five
  topic-versus-stance false positives but nonsignificant enrichment
  (`p=0.5022`). Existing Archive target/mechanism fields, LB-SCGP observables and
  C3 dense rationales are incomplete or historically negative.
- **Exact reviewed files:** reviewer exact-matched the five SHA256 values recorded
  in `refine-logs/C04_DESIGN_REVIEW.md` and returned
  **`REVISE_USER_AMENDMENT_REQUIRED (2 Critical / 5 High / 4 Important)`**.
- **Critical 1 — proxy-target mismatch:** P8 summary + K4 evidence density +
  deterministic S/T cues do not instantiate the proposed neutral proposition,
  protected-target harm act, source and stance factors. This proxy may be a
  nonbinding diagnostic, but it cannot PASS or scientifically KILL SPaSH.
  Therefore the frozen pre-teacher Stage-0 order currently has no executable C04
  route.
- **Critical 2 — hard contract:** the draft lacks explicit per-slot reliability,
  conflict handling and deterministic fallback. Revision must freeze
  `stable/single_valid/conflict/missing`, report coverage/fallback/corruption, and
  prevent confidence from becoming an unreviewed selector or weight.
- **Required method repairs:** add capacity-matched `CONCAT_ALL4_MLP`, retained
  independent-four-target/P4-strong, slotwise shuffle and role permutation; close
  P4 plus LEAF/C5-style KD collisions; separate train-only DIRECT OOF from STUDENT
  OOF/native-only dev; freeze nested folds and full adaptation+head paired seeds;
  and extend novelty comparison to RAMF, LEAF, TFN/LMF and DR-HM/Intent-style
  decomposition.
- **Minimal user decision:** authorize, for C04 only, a bounded matched-teacher
  pre-gate before the current full-bank Stage-0. First tranche = exactly 200
  label-blind SHA256-selected train IDs per dataset, train-only local open weights,
  two fixed prompts/eight frames/capped transcript, one GPU at a time,
  `8 CPU / 64 GB`, aggregate cap 2 GPU-hours across HateMM+MHC-ZH. It may test only
  reliability/conditional information/calibration/permutation. On PASS plus fresh
  independent GO, complete the 744/579 train-only banks under a total cap of
  8 GPU-hours including the first tranche; no dev/test teacher. The original
  full-bank `+0.050/+0.050` gate remains mandatory.
- **Boundary:** this is a decision request, not authorization. Even user approval
  requires a revised proposal and fresh independent design/code/resource GO.
  No Python, test, cache opening, teacher generation, GPU, test access,
  implementation or SLURM submission occurred.

### C04 V2 bounded-teacher design freeze — PENDING FRESH REVIEW (2026-07-29)

- User approval is limited to the fixed 200+200 matched-teacher pre-gate and
  conditional full train bank under the `<=2`/`<=8 GPU-hour` serial caps; the
  full-bank `+.050 accuracy / +.050 macro-F1` two-dataset gate is unchanged.
- V2 closes the prior 2C/5H/4I with explicit reliability/fallback, exact safe
  tensor/lower-order operators, retained-independent/flexible-concat controls,
  exact perturbations, same-arena nested OOF, native-only dev, complete paired
  seeds and RAMF/LEAF/TFN-LMF/DR-HM/Intent novelty boundaries.
- Exact files/hashes are registered in `TARGET_STATE.json`. No implementation,
  Python/test, cache/teacher, GPU, test or SLURM action occurred. Fresh
  independent design review is the only next legal step.

### C04 V2 review and minimal V3 repair — PENDING FRESH REVIEW (2026-07-30)

- V2 exact-hash review returned `REVISE (0C/4H/3I)` with no new user-contract
  request. V3 freezes exact fallback control semantics/gates, semantic
  unreliability as scientific KILL, unique fair nested tuning and seed 0,
  block-covered dense lower-order compression, zero-frame handling, full
  perturbation margins/CI and an internal resource watchdog.
- V2 accepted clauses and the user amendment remain binding. No implementation,
  Python/test, cache/teacher, GPU, test or SLURM action occurred.

### C04 V3 review and minimal V4 repair — PENDING FRESH REVIEW (2026-07-30)

- V3 review returned `REVISE (0C/2H/1I)`. V4 makes reliability-state evidence
  anti-artifact rather than gain-seeking, gives every omitted retained control
  the full equal search, and reconciles malformed/missing outputs with the five
  frozen rate thresholds.
- No user-contract change, implementation, Python/test, cache/teacher, GPU, test
  or SLURM action occurred.

### C04 V4 fresh design review — GO (0C/0H/0I) (2026-07-30)

- The reviewer exact-matched all V4 hashes, confirmed unchanged V3 bases and
  accepted the fallback anti-artifact, equal-tuning and quantified reliability
  taxonomy repairs with no remaining design finding.
- This is design-only GO. The next boundary is prospective implementation plus
  fresh code/resource review. It does not authorize implementation, teacher,
  Python/test, GPU, SLURM, labels/test or experiment execution.

### C04 implementation-v5 and CPU-preflight authority — PENDING UNLOCK REVIEW (2026-07-30)

- Five fail-closed implementation revisions converged to an independent
  `GO (0C/0H/0I)`. The exact reviewed prospective config and implementation
  record hashes are frozen in
  `C04_A0T_SMALL_V1_V5_CODE_RESOURCE_REVIEW.md`.
- A strict `CPU_PREFLIGHT` authority manifest and authorized config snapshot
  are prepared. Only implementation and CPU-preflight materialization are true;
  teacher/GPU/Slurm-GPU/small/reconciliation/dev/test/OCR/API/network/cross/
  label/chain/release/resubmit remain false.
- Prompt/map payload hashes and payload/GPU/reconciliation reviews remain
  pending, and the v5 artifact namespace is absent. No Python, data/model
  access, or SLURM submission occurred. Fresh independent unlock review is the
  only next legal step; CPU preflight has not been submitted.

### C04 v5 CPU-preflight unlock review — GO (0C/0H/0I), READY NOT SUBMITTED

- The reviewer independently recomputed the authority manifest/file/closure,
  normalized config contract, reviewed predecessor, all implementation/design/
  source/model hashes, pending semantics and absent namespace.
- GO authorizes exactly one fixed CPU-only entrypoint:
  `scripts/slurm/c04_a0t_small_v1_v5_preflight.sbatch`.
- No submission occurred in this update. Teacher, GPU, Slurm-GPU, small,
  reconciliation, labels and every forbidden surface remain blocked.

### C04 v5 CPU preflight job 13805 — ENGINEERING HALT, no scientific verdict (2026-07-30)

- The one authorized submission was spent. `sbatch scripts/slurm/c04_a0t_small_v1_v5_preflight.sbatch`
  produced job `13805`, submitted `2026-07-30T08:48:50`, held as `JobHeldUser`
  for ~7h52m, auto-released and started `16:40:57`, and ended `16:40:57`
  `FAILED` `1:0` with `00:00:00` elapsed on `foscsmlprd01` (8 CPU / 64 GB, no
  GPU). The hold was normal and was never forced.
- Stdout is exactly 0 bytes; stderr is 1191 bytes. The wrapper's first call is
  `--mode self-test`, whose result is printed to stdout, so the program died in
  `verify_static_config` before any mode dispatch.
- The halt is a self-contradictory static contract, not a data, model, resource,
  environment or SLURM fault. `c04_a0t_small_v1_v5_preflight.py:152` asserts
  `prompt_hashes() == cfg["prompt_hashes"]`, but
  `configs/c04/c04_a0t_small_v1_v5.json:115-120` holds
  `PENDING_CPU_PREFLIGHT_HASH_FREEZE` for all four keys — and computing those
  hashes is precisely what this preflight exists to do. The freeze run could
  never pass its own gate. The v5 producer asserts the same equality at line
  177, so the defect is in the shared v5 contract.
- Fail-closed evidence: `artifacts/c04/` does not exist at all, so the
  no-clobber namespace check at line 277 was never reached, nothing was staged
  or partially published, and no temp namespace remains. No train ASR was
  opened, no video or model file was hashed, no label value was materialized,
  no teacher and no GPU ran.
- **No metric, result, decision, CONTINUE or KILL verdict was published.** This
  event carries no evidence for or against C04 and consumes no scientific gate.
- This termination was not recorded when it happened; `TARGET_STATE.json`,
  `TARGET_LOOP.md` and `TARGET_FINDINGS.md` were last written `08:44` and kept
  claiming `READY_NOT_SUBMITTED` for ~2h24m after the job died. The delayed
  primary record is `iteration_8_c04_impl_v5_cpu_preflight_engineering_halt` in
  `TARGET_STATE.json`.
- The v5 config/source pair is unrunnable by construction and may not be
  resubmitted. The only legal next step is a fresh versioned v6 implementation
  closure plus fresh independent static review; teacher, GPU, small tranche,
  reconciliation, dev/test, OCR, API/network and label access all remain false.

### Stage-0 bounded-extraction adjudication amendment (2026-07-30)

- **User ruling.** The user authorised extraction (`可以去抽`), generalising the
  one-off bounded-teacher exception previously granted to C04 into a standing
  registry rule.
- **The deadlock it resolves.** The frozen `stage_0_reachability` rule demanded an
  *existing-bank* representation oracle at `+0.050` accuracy and `+0.050` macro-F1
  on two datasets before any new extraction. A candidate whose mechanism lives on a
  representation view that no banked cache expresses therefore had to extract to
  pass the gate and pass the gate to extract. 99 top-level `train_*.pt` caches
  exist (HateMM 30, MHC 26, MHC-ZH 41, ImpliHateVid 2, counted 2026-07-30) and all
  of them vary image pooling, attention/readout span, LoRA weight point or PEFT
  merge path — never the text channel. C02 and C03 were both killed on exactly
  this, and both kills said in writing that they were not scientific refutations.
- **Amended rule.** Stage-0 reachability may be adjudicated on an existing bank
  **or** on a bank produced under a bounded extraction budget of **≤ 4.0 GPU-hours
  per candidate**, subject to all of:
  - (a) design hash-frozen and independently `GO (0C/0H/0I)`-reviewed **before any
    extraction job is submitted** — this is the whole point; without it the
    extraction becomes test-driven selection;
  - (b) extraction covers **train + dev_seen only**; test is never touched and the
    loader's test-like path rejection stays asserted in code;
  - (c) the `+0.050` / `+0.050` two-dataset bar is **unchanged**, net-fix clause
    included;
  - (d) **no other `hard_constraint` is relaxed** — no OCR, no cross-dataset
    mixing or cross-dataset training, no external model API, single-dataset train
    split only, parent-video binary label as the only gold, no ensembles, no
    model-size scaling, Slurm-only with `conda HateVideo` and no `--time`;
  - (e) `one_candidate_at_a_time` and `parallel_gpu_or_teacher_pilots_forbidden`
    stand, and a bounded extraction counts as a GPU pilot for both;
  - (f) actual GPU wall time is measured with `sacct` and reported; an over-budget
    run is a protocol violation and its result is void;
  - (g) **F113 stands** — a raw-key arena may KILL but may not promote, so a
    Stage-0 PASS must be rendered on the fold-head / deployed-head path.
- **Budget basis (sacct-measured, 8 CPU / 1 GPU / 48-64 G).** `13295` 00:24:23,
  `13329` 00:27:18, `13302` 00:34:37, `13648` 00:46:27, `13352` 00:59:57,
  `13470` 01:01:44, and the largest single extraction job ever observed,
  `13468 gen_embed_readout` 02:00:08.
- **Unblocked in principle.** C02 now; C05, C06, C09, C10, C11 and C12 are the
  candidates most likely to hit the identical wall. Unblocking is procedural: each
  still needs its own prereg, its own independent GO and its own verdict.
- **Not covered: C03.** C03 must *train* native policy-anchored MNTP before it
  could extract anything — a different order of commitment that a 4-GPU-hour
  extraction budget does not price. C03 stays `killed_design_infeasibility_frozen`
  pending a separate explicit user decision. This amendment is not authority for
  it.
- No extraction, GPU, teacher, Python, cache open, label read or test access
  occurred in this update, and no scientific claim was created or altered.

### C02 revived under the amendment — preregistration reopened (2026-07-30)

- `C02 Evidence-Density Quotient Geometry` moves from
  `killed_design_infeasibility_frozen` to
  `revived_under_stage0_bounded_extraction_amendment_a0_prereg`.
- The 2026-07-29 `KILL_C02_DESIGN_COLLISION_OR_INFEASIBILITY` is **retained
  verbatim in the registry as historical evidence and is not overturned on its own
  terms**. `refine-logs/C02_DESIGN_REVIEW.md` and
  `refine-logs/C02_EXPERIMENT_PLAN.md` are left as written.
- Its two genuine scientific objections were already repaired before it was
  issued, and the repairs are what the new A0 implements: the orbit is defined on
  the **native text channel** with representation-matched extracted views (no P3
  image-pooling proxy), and every view **retains the complete native transcript as
  an ordered subsequence**, adding only controlled repetition (no deletion).
- Every requirement the reviewer left standing is carried into the new A0:
  `RANDOM_WINDOW_REPEAT`, `MIN_WINDOW_REPEAT`, `REPEAT_ONLY`,
  `LOCALIZED_REPEAT_ONLY`, frozen orbit radius, frozen KRR length metric,
  retrieval-length correlation, frozen confidence/control thresholds, declared
  lambda-selection status, declared Holm family, full self-orbit exclusion, and
  explicit fail-closed handling of examples whose transcript orbit is the identity
  (including HateMM `hate_video_95`, the known structural all-zero null).
- Revival authorises **pre-registration and independent review only**. It is no
  evidence for the evidence-density hypothesis and no job was submitted by this
  update.

### C04 implementation-v6 — engineering repair of the v5 ordering defect, GO (0C/0H/0I), job 13840 submitted (2026-07-30)

- v6 exists to repair one thing: the v5 CPU preflight could not pass its own
  first static gate. `verify_static_config` demanded
  `prompt_hashes() == cfg["prompt_hashes"]` before the freeze that materializes
  those four values. `resolve_prompt_hashes(cfg, freeze_stage)` now permits the
  `PENDING_CPU_PREFLIGHT_HASH_FREEZE` sentinel only on the freeze run, only for
  those four keys, and only when `preflight_materialization_authorized` is
  exactly `True`. Any other value, a mixed pending/frozen state, a foreign key
  set, or the sentinel on any non-freeze path HALTs.
- The computed hashes are written to a literal freeze artifact
  (`freeze/prompt_hashes.json`); `assert_literal_prompt_hashes` guards the
  preflight manifest and that artifact, and the producer plus both GPU-ledger
  validators require `LITERAL_BOUND`, so no downstream stage can ever read the
  sentinel. 13 new self-test fixtures, 25 checks total.
- Two review findings mattered more than the original bug, because the first
  repair had *displaced* the impossibility rather than removed it. (a)
  `config_contract_sha256` hashed `prompt_hashes` verbatim, so the contract
  baked into the authorization manifest, genesis ledger and resource ticket was
  computed over the pre-freeze config while downstream stages need the
  post-freeze one — leaving the config unfrozen would have burned the single GPU
  allocation before the producer's HALT, and amending it would have invalidated
  the pinned manifest permanently. (b) A claim-time HALT ran the GPU wrapper's
  `EXIT` trap, and `mark_exit` bumped the genesis ledger even with an empty job
  list, breaking the ticket's `genesis_gpu_ledger_sha256` pin inside a
  no-clobber namespace. Both are now closed, and the config contract is provably
  invariant across the freeze.
- Everything else is byte-identical to v5 modulo the version rename: all three
  wrappers, all three sbatch files and all five schemas differ by 0 lines. No
  scientific constant, authorization flag or guard changed in meaning.
- Independent static review: five code/resource rounds
  (`REVISE 0C/2H/2I` → `REVISE 0C/1H/2I` → `REVISE 0C/1H/2I` → `GO 0C/0H/1I` →
  `GO 0C/0H/1I`) then four unlock rounds over the complete authority snapshot
  (`GO 0C/0H/4I` → `REVISE 0C/1H/2I` → `GO 0C/0H/3I` → **`GO 0C/0H/0I`**), fresh
  reviewer each round, every hash independently recomputed.
- Unlike v1-v5, every CPU-preflight gate was executed read-only on the login
  node against the exact authorized bytes before submission — that check's
  absence is what let 13805 reach an 8-hour queue and die on its own first gate.
  All passed: static config, code/resource authorization, 25 self-tests, model
  snapshot (both tree hashes), both dataset evidence loads (200 of 744 and 200
  of 579, `label_value_materialized=0`), all 400 selected videos resolving, the
  freeze payload, and staged-path containment.
- `sbatch scripts/slurm/c04_a0t_small_v1_v6_preflight.sbatch` → job `13840`,
  submitted `2026-07-30T21:41:35`, `PENDING (JobHeldUser)`, awaiting automatic
  release. Submitted exactly once; no force release. CPU-only, 8 CPU / 64 GB, no
  GPU, no `--time`, no array, no dependency.
- **No metric, result, decision or CONTINUE/KILL verdict is published by this
  update.** Teacher, GPU, small tranche, reconciliation, dev/test, OCR,
  API/network, labels, chain, release and resubmit all remain false. After 13840
  terminates an independent reviewer must issue a fresh payload-hash verdict; a
  successful preflight authorizes nothing further by itself.

### C02 A0 preregistration — frozen and independently reviewed to GO (0C/0H/0I) (2026-07-30)

- **Amendment condition (a) satisfied.** The design was hash-frozen and independently
  reviewed **before any extraction job was submitted**. Without that ordering the
  extraction would be test-driven selection, which is the whole reason the amendment
  exists.
- **The object.** Each video carries a discrete orbit of controlled evidence-density views
  of its own text channel — `NAT`, `RFULL = T + " " + T`, and `RW1..RW4` (window `k`
  duplicated in place at the frozen quarter cuts `c_k = (k*len(T))//4`, one integer
  expression, no snapping heuristic, dataset-symmetric across English and Chinese). The A0
  asks whether the induced quotient similarity `s_Q(i,j) = max_{a,b} cos(z_i^a, z_j^b)`
  clears `+0.050` accuracy **and** `+0.050` macro-F1 over the paired native floor on
  **both** HateMM and MHC-ZH.
- **The 2026-07-29 reviewer's two scientific repairs are what it implements.** The orbit is
  defined on the native text channel with representation-matched extracted views, so there
  is no P3 image-pooling proxy for a text-density target; and every view **retains the
  complete native text as an ordered subsequence**, adding only controlled repetition —
  proved per item per view before any forward pass, with the checker itself tested against
  a deletion.
- **Arena.** Fold-head / deployed-head: 5 frozen folds asserted item-for-item against the
  banked `vsw_ckpt`, head trained on the fitting pool, queries the held-out fifth, 3 head
  seeds, 3-seed mean primary. Bank and query index sets are disjoint per fold, so a query's
  own orbit can never be retrieved. The raw fused arena is computed but is **secondary**
  and may only corroborate a KILL (F113).
- **Controls.** Every arm is a sub-orbit of the single extraction: `REPEAT_ONLY`,
  `LOCALIZED_REPEAT_ONLY`, `RANDOM_WINDOW_REPEAT`, `MIN_WINDOW_REPEAT` (the four the
  reviewer named), `MAX_WINDOW_REPEAT` secondary, plus `SHUFFLE` and `NOISE`. The primary
  treatment uses **no P3 at all**; P3 enters only MIN/MAX, whose window correspondence is a
  declared positional approximation, which is why they are controls and not the treatment.
- **Seven freezes, six fresh independent reviews**, each by a reviewer that had not seen
  the implementation reasoning: `REVISE (0C/4H/20I)` → `REVISE (0C/2H/19I)` →
  `REVISE (0C/4H/23I)` → `GO (0C/0H/23I)` → `GO (0C/0H/4I)` → `GO (0C/0H/3I)` →
  **`GO (0C/0H/0I)`**.
- **Three findings changed the science, not the prose:**
  1. the `SHUFFLE` control was measuring leakage rather than orbit membership — twice. A
     global derangement handed held-out queries the views of bank items; fixing the
     partition boundary did not fix it, because donating the donor's *absolute* view keys
     still made bank row `j` a near-duplicate of bank row `pi(j)`. It is now
     **displacement donation**: `z_i^v := NAT_i + (view_v(pi(i)) - NAT_pi(i))`;
  2. `net_fix_rate` is **algebraically identical** to `delta_acc`
     (`fixed - broken = n * delta_acc`), so a `+0.030` net-fix conjunct can never bind
     under a `+0.050` bar. The registry's net-fix clause is discharged **by the accuracy
     bar**, and the design now says so instead of restating the bar as a second gate;
  3. the claim that `s_Q` **upper-bounds** what any orbit-contracting representation could
     buy is **retracted**. It is one particular orbit-invariant similarity — the canonical
     max-matching quotient pseudo-metric — so a **KILL is a gate verdict under the frozen
     Stage-0 rule**, not a proof that no such representation could help.
- **Two engineering defects caught before any GPU spend:** an exhaustive `k = n_bank` faiss
  search is **not** bit-equal to the deployed `k = 20` call (measured max
  `|delta sim| = 1.5e-07`, enough to break `PARITY-NAT`), so the oracle searches `k = 20`
  per view pair — exact for the top-20 and literally the deployed call for a singleton
  orbit; and `mechfix_ops._norm32` can **alias its input** while `faiss.normalize_L2` works
  in place, so the arena's own `_norm32` always copies.
- **Nothing has run.** No extraction, GPU, teacher, cache, label or test access has
  occurred, and no result, decision or verdict exists. The extraction is queued behind
  C04's preflight under `one_candidate_at_a_time`.

### C02 A0 erratum — a false empirical claim, self-detected after GO (2026-07-30)

- Every freeze v1-v7, and the v7 review that returned `GO (0C/0H/0I)`, carried the claim
  that **an exhaustive `k = n_bank` faiss search is NOT bit-equal to the deployed `k = 20`
  call**, with a measured `|delta sim| = 1.5e-07`.
- **It is false.** Re-measured on synthetic arrays with private, singly-normalised
  operands:

  | comparison | sims bit-equal | ids equal | max \|Δsim\| |
  |---|---|---|---|
  | exhaustive `k = n_bank` vs deployed `k = 20` | **True** | **True** | **0.0** |
  | `k = 20` per pair (the frozen path) vs deployed | True | True | 0.0 |
  | operands normalised **twice** (aliasing) vs once | False | — | **1.4901161193847656e-07** |

- The whole discrepancy was the `mechfix_ops._norm32` **aliasing** defect — a real and
  separately recorded finding — and I attributed it to the search width.
- **Cause.** Both defects were live in the same first dry run. I changed the search width,
  saw the discrepancy persist, then found and fixed the aliasing bug, after which parity
  was exact — and never re-tested the exhaustive path with the aliasing fix in place. That
  is a **fabricated companion measurement**, precisely what the project's
  numeric-provenance rule forbids.
- **Why six reviews missed it.** Each review request asked whether the `k = topk` exactness
  *argument* holds. It does, and each reviewer verified it independently. The false
  statement was an empirical claim standing beside a correct argument and presented as
  already measured — which a reviewer who is not re-running the measurement cannot detect.
  The lesson generalises: **a review request should name the measurements a design claims
  to have made, not only the arguments it makes.**
- **Nothing numerical changes.** `k = topk` per view pair is retained, on the true reasons:
  provably sufficient for the top-topk, `O(topk)` rather than `O(n_bank)` per view pair,
  and **literally** the deployed call for a singleton orbit rather than merely equal to it
  — which is the cleanest possible basis for `PARITY-NAT`.
- **No measurement is affected.** No extraction, GPU run, result, decision or verdict
  existed at any point in this episode. What was wrong was a frozen justification, and it
  is now retracted in the frozen config, the frozen arena docstring and
  `refine-logs/C02_A0_V8_RECORD.md`. Records v1-v7 are left as written, as the historical
  evidence of what was believed when.

### C02 A0 v8 erratum re-check — GO (0C/0H/1I), and a new review instrument (2026-07-30)

- The confined re-check confirmed the eight v8 hashes, that the retracted search-width
  claim survives nowhere but inside its own retraction, that the exactness argument is
  unchanged and **never depended on it**, and that `PARITY-NAT` still binds.
- **One Info is declared open rather than fixed**, on the reviewer's explicit advice
  ("do not cut a v9 for this"): "`O(topk)` rather than `O(n_bank)` per view pair" is false
  under the natural reading, because a flat inner-product index computes all `n_bank`
  similarities regardless of `k` — what `k` bounds is the selection heap and the
  `(nq x topk)` result, not the scan. No number the run produces depends on it, and the
  other two reasons for `k = topk` over-determine the choice.
- **A new instrument came out of the erratum: the unverifiable-claim register.** At my
  request the reviewer enumerated every statement in the frozen set asserted as *measured*
  that a static reviewer cannot re-derive. Building it took one pass, converted three
  previously-trusted claims into verified ones (the MHC-ZH whitespace counts, the fourth
  `max_chars`, and the two banked `B_fid` figures read straight out of
  `headspace_fidelity{,_zh}_OUT.json`), and found that **the largest residual risk in the
  set is not a science claim at all** — it is the GPU-hour projection.
- **The lesson, worth generalising beyond C02:** a review request should name the
  *measurements* a design claims to have made, not only the *arguments* it makes. Six
  rounds verified a correct argument and never questioned the false empirical claim sitting
  beside it. The cheap companion habit: when two defects are live in one dry run,
  **re-measure every claim attributed to the first after fixing the second**.
- **Carried into the run:** the 4.0 GPU-hour cap is a post-hoc `sacct` check with no in-job
  enforcement, and an overrun voids the result. It is now a monitored quantity.

### C02 A0 v9 — GO (0C/0H/0I), the GPU cap enforced in-job, the claims register adopted (2026-07-30)

- **Three changes, nothing else.**
  1. **The last Info closed.** "`O(topk)` rather than `O(n_bank)` per view pair" was false
     of the scan: a flat inner-product index computes every inner product regardless of
     `k`, which bounds only the selection heap and the `(nq x topk)` result width. The two
     load-bearing reasons for `k = topk` — provable sufficiency, and *literally* the
     deployed call for a singleton orbit — are untouched.
  2. **The 4.0 GPU-hour cap is enforced in-job.** The v8 register had named this the
     largest downside in the set: condition (f) makes an over-budget run a protocol
     violation whose result is **void**, and nothing enforced it. One absolute deadline
     (`14400 s` cap minus a `600 s` margin) is computed at job start and passed to both
     dataset invocations; `budget_check` refuses to **start** work needing more than
     `max(2 x slowest item so far, 60 s)` of headroom; a breach publishes an
     accounting-only `BUDGET_BREACH_<dataset>.json` and exits **5**.
     **Inertness is the load-bearing property and was verified, not asserted:** every write
     — view caches, manifest, breach record — is outside the guard window, so a breach
     leaves the in-progress dataset entirely unwritten and any completed dataset intact,
     and the A0 then fails closed on the missing manifest rather than reading a truncated
     bank. The guard touches no scientific semantic, threshold, arm, metric or decision
     rule.
  3. **The measured-claims register is adopted** — claim / how obtained / what depends on
     it / re-derivable by a static reviewer without execution, with `[V]`/`[D]`/`[U]`
     classification. This is the instrument the v8 erratum produced.
- **The register worked on its first outing.** The reviewer independently re-derived every
  `[V]` row, including the forward count at **exactly 8758**; found no `[U]` that was
  statically re-derivable, no `[V]` that failed, and nothing asserted-as-measured missing.
  Row 15 — "the wall clock is still `[U]` but is now bounded by the in-job guard" — was
  tested hardest and held without being overstated.
- **Standing habit, recorded for the next agent:** when two defects are live in one dry
  run, **re-measure every claim attributed to the first after fixing the second**. And a
  review request should name the **measurements** a design claims to have made, not only
  the **arguments** it makes.
- **Two disclosed observations, deliberately not fixed** (both verified, neither false,
  neither able to affect the run): a superseded record pointer in the config's
  condition-(e) evidence line, whose operative counterpart is correct; and a true, ungated
  `raw_effect_under_test 0.0255` assertion missing a register row. Recorded rather than
  spun into a v10.
- Nine freezes, eight independent review rounds. Still no extraction, GPU, result,
  decision or verdict.

### C04 v6 CPU preflight job 13840 — COMPLETED, payload frozen and independently verified (2026-07-31)

- `sacct -j 13840`: `COMPLETED`, exit `0:0`, submitted `2026-07-30T21:41:35`,
  eligible `23:32:20`, start `2026-07-31T05:11:31`, end `05:11:50`, elapsed
  `00:00:19`, CPU time `00:02:32`, 8 CPU / 64 GB on `foscsmlprd01`. `ReqTRES`
  is `billing=8,cpu=8,mem=64G,node=1` — **no GPU was ever requested**. The
  queue split was `JobHeldUser` for 1h50m45s then `Priority` for 5h39m11s,
  7h29m56s total for 19 seconds of work; the hold auto-released and was never
  forced.
- stderr is **0 bytes**; stdout is **1399 bytes**
  (`c7e3f85255234c4fc5fb6e949e371d2572a5a6a3466907af6a029c003639cdeb`),
  reporting `"all_passed": true` over 25 checks with
  `config_prompt_hash_binding: SENTINEL_PENDING_CPU_PREFLIGHT_FREEZE`.
- **The 13805 defect is closed in both directions**, exercised in the real
  SLURM environment: sentinel accepted on the authorized freeze run; rejected on
  the non-freeze path; rejected without materialization authorization; wrong
  value rejected on every path; mixed pending/frozen rejected; foreign key set
  rejected; the post-freeze manifest guard accepting the literal while rejecting
  both sentinel and wrong value; the frozen payload carrying literal hashes; a
  sentinel-bearing payload rejected even when internally hash-consistent; and
  the config contract invariant across the freeze while still moving under
  tampering.
- The frozen payload was then verified independently, recomputing rather than
  trusting `all_passed`: 15 files / 5 directories / 5,178,606 bytes under
  `artifacts/c04/a0t_small_v1_impl_v6/`, no temp or lock residue.
  `prompt_hashes.json` carries the four literal hashes — byte-identical to
  those printed in 13805's stderr — with no key holding the sentinel or any
  `PENDING_*` string. All 14 `staged_output_hashes` match the files on disk.
  The 15th file is `preflight_manifest.json` itself, which cannot list its own
  hash (`preflight.py:471` builds the list, `:476` hashes the manifest, `:477`
  stages it); its integrity rests on its own `payload_sha256`, verified, and it
  is pinned at the next stage.
- Access ledger: exactly two operation kinds, 2 train-ASR opens and 400
  train-video hashes, both ASR reads resolving to the `train_asrK4` files,
  `label_value_materialized = 0` with 1323 label fields syntactically skipped
  (744 + 579), no dev/test/validation token in any event, events merkle root
  recomputing, and no teacher or frame code invoked.
- Maps match the config's declarations exactly: `le3` is 3,684,352 bytes =
  256x3598x4 and `additive` is 1,048,576 bytes = 256x1024x4, every float32 entry
  is ±0.0625 per `maps.scale`, and both byte streams plus all four role maps
  reproduce exactly from the frozen generators. Allowlists are 200 rows per
  dataset with recomputing selection digests and merkle roots, ascending order,
  unique IDs, train-only paths and no label field in any row.
- Resource state untouched: ledger `GENESIS_UNCLAIMED`, revision 0, zero jobs,
  zero accounted seconds, cap 7200; ticket `single_use` true, `consumed` false,
  0 completed GPU seconds, genesis pin matching the ledger on disk. The
  consumption record, allocation claim, entry marker, `seal/` and
  `checkpoints/` are all absent.
- **No metric, result, decision or CONTINUE/KILL verdict is published.** This is
  an engineering milestone; it is no evidence for or against SPaSH-Tensor and
  consumes no scientific gate. The unified pilot gate and the +0.030/+0.030
  two-dataset target are untouched.
- **Nothing downstream is authorized.** The next legal step is an independent
  collector/reviewer issuing a fresh payload-hash review verdict pinning
  `preflight_manifest_sha256`
  `06bf6b38f424dd53d142367abd029dfa1f485380fb1482d72beabb7f5943ad1a`; only that,
  followed by a separate GPU-execution authorization, could unlock the teacher
  stage. No further job was submitted; the queue holds only `13843
  c02_density_extract`, so under `one_candidate_at_a_time` C04 submits nothing.

### C02 Stage-0 extraction job 13843 — COMPLETED, verified against the frozen contract (2026-07-31)

- `sacct -j 13843`: `c02_density_extract`, **`COMPLETED`, exit `0:0`**, submitted
  `2026-07-31T05:13:25`, started `07:31:04`, ended `09:59:19`, **elapsed `02:28:15`**,
  `billing=8,cpu=8,gres/gpu=1,mem=64G,node=1` on `foscsmlprd01`, an
  `NVIDIA A100-SXM4-80GB`. Queue wait `02:17:39`; the `JobHeldUser` hold auto-released
  and was never forced.
- **Amendment condition (f) HOLDS — the result is not void.** `02:28:15` is **8895 s =
  2.4708 GPU-h** against the **4.0 GPU-h** cap: **61.8 % of budget, 1.5292 GPU-h of
  headroom**. The wrapper's own closing line reads `elapsed 8894s of the 14400s cap`,
  1 s under `sacct Elapsed` because its timer starts after the preamble; `sacct` is the
  authority and is the number reported.
- **The in-job budget guard never fired**, and no `BUDGET_BREACH_*.json` exists anywhere
  under `artifacts/c02_edq`. It used the intended anchor rather than the fallback: the
  wrapper printed `job start 1785439864`, which is exactly `sacct`'s `Start 07:31:04`,
  and its deadline `1785453664` is `11:21:04 NZST` — the job finished **1 h 21 m 45 s
  early**. Budget remaining at each split end was `8678.5 / 8311.0 / 5286.0 / 4906.9 s`
  against slowest items of `6.55 / 5.35 / 19.38 / 9.22 s`, so the guard's largest
  requirement all run was `max(2 x 19.38, 60) = 60 s` against a minimum remaining of
  `4906.9 s`. It was never within two orders of magnitude of firing.
- **24/24 view caches** and **2/2 manifests** exist. All **24 manifest `sha256` values
  were recomputed against the files on disk this session: 24 OK, 0 mismatch.** Both
  manifests parse, carry `schema_version c02_density_extract_manifest_v1`, record
  `this_script bb698ab9…` and `view_module 44fbb00b…` equal to the v9 frozen hashes, and
  hold one `per_item` row per item (744 + 107 + 579 + 78).
- Per split: `HateMM/train n=744 forwards=4197 copied=261 degenerate=48 full-identity=48
  view_support=0.935484 zero_guard=1` · `HateMM/dev_seen n=107 forwards=589 copied=53
  degenerate=10 full-identity=10 view_support=0.906542 zero_guard=0` · `MHC_zh/train
  n=579 forwards=3473 copied=1 degenerate=0 view_support=1.0 zero_guard=0` ·
  `MHC_zh/dev_seen n=78 forwards=468 copied=0 degenerate=0 view_support=1.0 zero_guard=0`.
  Degeneracy causes are `EMPTY_TEXT 39 + LENGTH_GUARD 9` on HateMM train and
  `9 + 1` on HateMM val — exactly the counts register rows 2 and 3 predicted.
- **The decode failure is the SAME item as the known structural null, not a new one.**
  Exactly one video failed both decoders — `hate_video_95.mp4`, decord
  `av_read_frame failed with 1094995529` then PyAV `Error splitting the input into NAL
  units` — and it took the frozen zero-vector guard path (`n_forward=0`,
  `video_ok=false`, `zero.clone()` into all six view slots). It is **HateMM train row
  355, `hate_video_95`, label 1**, the null cited in `C01_ZERO_CONTRACT_PROBE.md` and
  `PROVENANCE_AUDIT_2026-07-28.md:187-193`. On MHC-ZH there were **40 decord failures,
  all recovered by the PyAV fallback**: zero PyAV failures, zero `no decodable frames`,
  `zero_guard_videos=0` on both splits.
- **Zero contract pre-verified by direct measurement** (read-only load of the two banked
  train caches and the twelve train view caches; no `dev_seen`, no test path). HateMM
  train: banked text zero rows `[355]`, banked img zero rows `[355]`, and all six
  `c02den` views zero exactly at `[355]`; MHC-ZH train: empty on all eight arrays. No
  non-structural tiny row (`0 < norm <= 1e-12`) anywhere. Criteria 2 and 3 will hold.
  **This upgrades measured-claims register row 12 — the structural-zero census,
  previously `[U]` "needs a `.pt` load" — to MEASURED, confirming the prior records
  exactly: 1 on HateMM train, 0 on MHC-ZH.**

**Register reconciliations — annotations, not silent corrections.**

- **Forwards `8727` observed vs the register's re-derived `8758`.** Shortfall **31**,
  itemised: HateMM train **-27** = 6 (the zero-guard item contributes 0 forwards, not 6)
  + 17 (14 items hit the declared `EMPTY_WINDOW` view-level identity — 3 on two windows,
  11 on one) + 4 (exact view-string collisions between two `RW` views); HateMM dev_seen
  **-3** = 1 + 2; MHC-ZH train **-1** = 1 `EMPTY_WINDOW`; MHC-ZH dev_seen **0**.
  `8758` was an upper bound that did not model two mechanisms the design itself declares
  (the zero-guard path, and the one-forward-per-distinct-string rule interacting with
  `EMPTY_WINDOW`). The observed number is **lower** — the run was cheaper than projected
  — and row 15's only dependent was the budget headroom, which held. The six colliding
  items are `non_hate_video_99` (RW4==RW3), `non_hate_video_319` (RW3==RW1),
  `non_hate_video_13`, `non_hate_video_621` (RW3==RW2) on train and `non_hate_video_263`
  (RW4==RW3), `hate_video_52` (RW3==RW2) on val; all six are music-note-only ASR
  transcripts of 4-10 characters, and **every collision is RW-vs-RW — none collides with
  `NAT`**.
- **`view_support` observed vs predicted: NO MISMATCH.** The two numbers are different
  quantities and the register already separates them. Row 5's **text-only** prediction
  `0.9355 / 0.9065 / 1.0 / 1.0` is **exact** against the extractor's manifest
  (`0.935484 / 0.906542 / 1.0 / 1.0`). Row 6's **runtime-gate** prediction
  `0.9341` / `1.0000` is **also exact**: the arena computes support from
  `degen_mask = degen_text | degen_zero` (`arena_v9.py:823`), which additionally counts
  the zero-guard row, and the mask was checked directly rather than inferred —
  `degen_text=48, degen_zero=1, overlap=0` on HateMM train and `0, 0` on MHC-ZH, giving
  `1 - 49/744 = 0.934140` and `1.000000`. Both are far above `VIEW_SUPPORT_MIN = 0.60`.
- **`MHC_zh/train copied=1` is NOT a decode failure — the N2 hypothesis is refuted.**
  MHC-ZH recorded `zero_guard_videos=0` and no PyAV failure at all. The single copy is
  item index 183, `BV1oK41127n4`, native text `基佬紫` (3 characters): with `K=4`
  character quarters window 1 is empty, so the declared `EMPTY_WINDOW` rule sets
  `RW1 := T` and the string dedup copies the `NAT` vector into the `RW1` slot
  (`identity_views=['RW1']`, `empty_windows=[1]`, `degenerate=NONE`, `n_forward=5`).
  **The N2 condition is unreachable in the measured data**: `degen_mask` is all-False on
  MHC-ZH, so the degenerate donor class is *empty* rather than a singleton and the
  `g.size >= 2` drop rule stays inert there. `shuffle_singletons_dropped` will be read
  from the artifact, not predicted.
- Still carried forward: the register **row 19** the v9 reviewer disclosed
  (`configs/c02/c02_a0_v9.json:204`, `raw_effect_under_test 0.0255` /
  `STOP_RULE_TRIGGERED`, `[V]`, load-bearing for nothing) is to be added when the
  post-run record is written.
- **No A0 had been submitted when this section was written, and no metric, delta,
  decision or CONTINUE/KILL verdict exists.** The extraction is an input, and is evidence
  for or against C02 in neither direction.

### C02 A0 job 13847 — KILL_C02_DENSITY_ORBIT_UNREACHABLE (2026-07-31)

- `sacct -j 13847`: `c02_a0_v9`, **`COMPLETED`, exit `0:0`**, elapsed `00:29:49`, 8 CPU /
  0 GPU / 32 G. Submitted `19:16:59`, released from `JobHeldUser` after 13 s. **Exit `0:0`
  and not `3`, and the result carries no `halt` key with `result_exists: true` — this is a
  scientific verdict, not a validity-guard HALT.** Second and final authorized submission;
  the C02-A0-v9 execution budget is now spent.
- Submit-time preconditions, run in the same step as `sbatch`: `sha256sum -c` **8/8 OK**
  (the seven v9 executables plus `C02_A0_V9_RECORD.md`, which the v9 review §0 also pins)
  and `squeue -u jehc223` **empty**. The v9 record was deliberately **not** edited to fill
  its §6 pending fields, because it is hash-pinned at `20da7533…`; the evidence is in
  `TARGET_STATE.json::…a0_submission_record` instead, following the extraction's precedent.

**The measured result — quoted from `artifacts/c02_edq/v1/a0/C02-A0-v9/C02_A0_DECISION.json`.**

| dataset | `delta_acc` | `delta_mf1` | bar | acc 95% CI | Holm `p` |
|---|---|---|---|---|---|
| HateMM | **+0.0008960573476701761** | **+0.0005790634953705132** | +0.050 / +0.050 | `[-0.008961, +0.010753]` | 0.4425 / 0.4509 |
| MHC-ZH | **-0.0011514104778355128** | **-0.0027456813390588364** | +0.050 / +0.050 | `[-0.013241, +0.011514]` | 0.5802 / 0.6414 |

- **0 of 4 Holm nulls rejected** at `alpha = 0.05`. Both bootstrap lower bounds are
  negative on both datasets. On MHC-ZH `FULL` does **not** beat `SHUFFLE` or `NOISE` in
  either metric (all four false); on HateMM it beats both, by an amount indistinguishable
  from zero. Per-seed `delta_acc` straddles zero on HateMM
  (`[-0.006720, +0.009409, 0.000000]`) and is non-positive on MHC-ZH
  (`[-0.001727, 0.000000, -0.001727]`). Native floors are `0.886649 acc / 0.882044 mF1`
  (HateMM) and `0.892343 / 0.874052` (MHC-ZH). **HateMM reached 1.8 % of the accuracy
  bar and MHC-ZH is on the wrong side of zero — a null result, not a marginal miss.**
- **All five validity gates passed on both datasets**, so the KILL rests on a measurement
  the design itself certifies. `GATE-FID` `B_fid` **0.0093 / 0.0086** against the 0.050
  stop rule, identical to the banked register-row-7 values, `STOP_RULE_TRIGGERED false`.
  `GATE-EXT` `median_cos 1.0`, `max_abs_diff` **0.0** on both datasets over 743 and 579
  rows — the re-extracted `NAT` is **bit-identical** to the banked native text features,
  so the paired floor is exact. `PARITY-NAT` bit-equal on all 15 seed x fold cells per
  dataset (407 / 472 rows exempted from the neighbour-ID assert for exact float32 ties).
  `ARENA-2` pooled native accuracy `0.8858-0.8884` (majority `0.5995`) and
  `0.8895-0.8946` (majority `0.6891`) — neither saturated nor collapsed. `VIEW_SUPPORT`
  **0.93414 / 1.0**, exactly the values precomputed from the banked masks before
  submission. `ZERO_CONTRACT` criteria 2 and 3 held, `banked_text_zero_rows [355]`
  (`hate_video_95`, label 1) on HateMM and empty on MHC-ZH; the sensitivity read excluding
  the structural null moves HateMM `delta_acc` from `0.00089606` to `0.00089726`.
- **The secondary raw arena corroborates the KILL, which is the only direction F113
  permits.** `raw_preds` never reached `dec`. On HateMM `FULL` is `+0.006720` acc and the
  largest raw delta in *any* arm is `+0.012097` — a quarter of the bar. On MHC-ZH every
  arm is at or below native (`FULL -0.003454`, worst `REPEAT_ONLY -0.017271`).

**Why it failed — the diagnostics are unusually clear.**

- **There is barely an orbit to contract.** `orbit_radius_median_oof` is **0.000466 /
  0.000383 / 0.000415** across seeds 0-2 on HateMM and **0.000217 / 0.000203 / 0.000176**
  on MHC-ZH, in the deployed head key space. The largest per-view radius (seed 0) is
  `RFULL` at `0.001727 / 0.001289`, with the `RW` views at `0.00008-0.00061`. Duplicating
  a transcript, or a quarter of one, essentially does not move the deployed head's key.
- **What movement there is, is label-uninformative.** `FULL` changes only **22-29**
  predictions of 744 / 579; `net_fix` per seed is `-5 / +7 / 0` on HateMM and
  `-1 / 0 / -1` on MHC-ZH, against the roughly `+37` HateMM would need. Precision on the
  changed items is **0.5040 / 0.4881** — chance. Meanwhile the mean top-20 overlap with
  native is only **5.32-5.47 / 20** and **6.41-6.59 / 20**, so the orbit max reshuffles
  about three quarters of the retrieved neighbourhood and still buys nothing.
- **What the orbit max actually adds is a length artifact.** Retrieval-length Spearman
  rises from NATIVE `0.4808 / 0.5316 / 0.4909` to FULL `0.7784 / 0.7869 / 0.7962` across
  seeds 0-2 on HateMM, and from `0.3797 / 0.3821 / 0.3584` to `0.7356 / 0.7280 / 0.7377`
  on MHC-ZH, with `SHUFFLE` also lifted (`0.6584 / 0.6457` at seed 0). The native keys
  carry little length information to begin with — the OOF KRR length probe gives
  `R^2 = 0.0880 / 0.1071 / 0.0879` on HateMM and `-0.0034 / +0.0140 / +0.0072` on MHC-ZH,
  i.e. near zero on every seed — so the correlation is **introduced** by maximising over
  the orbit, not inherited from the representation. (Secondary and ungated: the KRR probe
  is reported, never read by the decision.)
- **`NOISE` collapses onto `NATIVE`**, bit-identically on both datasets (same accuracy,
  same macro-F1, Spearman `0.48086` vs `0.48082` on HateMM and identical on MHC-ZH):
  norm-matched random displacements never win the argmax, which is what displacements this
  small imply.
- **Control-arm caveat, recorded rather than spun.** `MIN_WINDOW_REPEAT` (`+0.004928`)
  outscoring `MAX_WINDOW_REPEAT` (`-0.002240`) on HateMM is **not** evidence that the
  anti-core beats the evidence-core: the two arms pick largely the *same* window
  (`min_window_hist RW1 = 634/744`, `max_window_hist RW1 = 572/744` on HateMM;
  `571/579` and `541/579` on MHC-ZH). The declared P3-to-text-window positional
  approximation degenerates onto window 1, so this contrast carries little information in
  either direction. `n_p3_missing = 0` on both datasets.
- `n_singleton_class_groups_dropped_head_arena` is **0 on both datasets**, so the
  v5→v9 **N2 singleton-DROP path never fired anywhere** — the open question the extraction
  record raised is settled empirically, not by prediction.
- **Register row 19 is discharged by the run itself:** both fidelity JSONs emit
  `raw_effect_under_test 0.0255` and `STOP_RULE_TRIGGERED false`, confirming the claim the
  v9 reviewer disclosed at `config:204` without a register row.
- Provenance in the artifact: arena `8cdaf1d3…`, config `62b38cd9…`, mint `e6430b76…`,
  view module `44fbb00b…` — all equal to the v9 frozen hashes; `frozen_modules` records
  `headspace_mint cefdf8dc…`, `mechfix_ops 635c1312…`, `mechnov_pairverify 77b0defd…`;
  `extract_manifest_sha256` `35d533a0…` / `1363a826…`.

**Scope.** This kills the **C02 Stage-0 oracle under the registry's frozen Stage-0 rule**
and closes the candidate. It is **not** an impossibility proof for density invariance:
`s_Q` is one particular orbit-invariant similarity, **not** a proven supremum over all
orbit-contracting representations — the arena's own `interpretation_boundary` says exactly
this. That stronger reading was overclaimed once and retracted in the v8 erratum; it is
not reintroduced here. Total C02 spend: **2.4708 GPU-h** of the 4.0 authorized, plus a
29-minute CPU job. The two authorized submissions are used and the +0.030/+0.030
two-dataset target remains unmet.

### C04 v6 payload review — GO (0C/0H/3I), eligible for a teacher tranche, no GPU work authorized (2026-07-31)

The frozen contract required that "after the preflight terminates, an
independent collector/reviewer must inspect the frozen artifacts and issue a
fresh payload-review verdict; no GPU or downstream stage becomes authorized
merely because the CPU preflight succeeds." That review is now done, by a
reviewer with no exposure to the v6 implementation reasoning, and it verified by
**recomputation** rather than by reading the job's own `all_passed`. It ran from
a scratchpad outside the repository with `PYTHONDONTWRITEBYTECODE=1` throughout
— so no `.pyc` was written anywhere, the defect the round-3 unlock reviewer
filed — and decoded only `id`/`window_text`/`language` from any ASR file, so the
reviewer held the same label-blindness the pipeline claims. The frozen common
module was **not** imported from its repository path: the prompt constants were
extracted by static `ast` parse of the frozen bytes.

**Verdict: `GO (0 Critical / 0 High / 3 Important)`.**
`refine-logs/C04_A0T_SMALL_V1_V6_PAYLOAD_REVIEW.md`.

- **Prompt-hash freeze — the 13805 failure mode is closed as a contract, not
  bypassed.** All four literal hashes reproduce from the frozen prompt sources:
  `system 1ffc0675…`, `A cecb3555…`, `B 9521bee7…`, `combined a42268e4…`.
  `payload_sha256 eb485b9a…` is self-consistent, the key set is exactly
  `{A,B,combined,system}`, `downstream_binding` is `LITERAL_BOUND`, and across
  the whole 15-file namespace the string `PENDING_CPU_PREFLIGHT_HASH_FREEZE`
  occurs **only** in `prompt_hashes.json` and there only as the value of
  `pending_sentinel_token` — a token *name*, never a key's value.
- **Allowlists are exact and clean.** The 200+200 ID sequences reproduce from
  the frozen selection rule with contiguous ranks 0..199, every
  `selection_sha256` recomputed, strictly ascending `(digest, id)`, merkle roots
  `5897b44c…` / `24d40b0e…` recomputed, and `selection_contract` equal to the
  config block. **Zero dev/test contamination**: selected ∩ dev = 0 and
  selected ∩ test = 0 on both datasets (HateMM dev 107 / test 215; MHC-ZH dev 78
  / test 149), and every selected ID is in train.
- **Source manifests re-hashed against disk, 400/400.** 3,430,759,978 bytes
  read, **0 mismatches** on `video_sha256`, size, device, inode,
  `resolved_train_relative` and `video_path`; all 400 `transcript_sha256`,
  `transcript_scalar_count` and `language` values re-derived through the frozen
  normalization rule and matched; merkle roots `a8eab8ad…` / `af2f8d7a…`
  recomputed. All 15 payload files also match the per-file `bytes`/`sha256`
  table already recorded in `TARGET_STATE.json`: 15/15 exact, no post-job drift.
- **Access ledger.** 402 events, exactly two kinds — 2 train-ASR opens and 400
  video hashes; `label_field_syntactically_skipped 1323` (= 744 + 579) with
  `label_value_materialized 0`; events merkle `60f22f38…` recomputed; no
  `dev`/`test`/`val` path component anywhere; the video events match the
  manifest rows 1:1 **in order** on id-hash, relative path, device and inode.
- **Maps rebuild byte-exactly.** `le3` is exactly `256×3598×4 = 3684352` bytes
  and `additive` exactly `256×1024×4 = 1048576`, both reproduced byte-for-byte
  from `dense_rademacher_payload`, with every float32 at ±0.0625 equal to
  `maps.scale`. All four role maps also rebuild byte-exactly, with 256 unique
  indices in `[0, 3584)`, signs ⊆ {−1, +1}, self-consistent `payload_sha256`,
  and pairwise-distinct index lists. This empirically closes the implementation
  record's own caveat that `maps.role_input_dim`, `role_output_dim`,
  `le3_shape`, `additive_shape` and `scale` are "read by no module": each was
  checked against the module constants **and** the materialized bytes and agrees
  on every count.
- **Zero GPU, confirmed independently of the payload.** Ledger
  `GENESIS_UNCLAIMED`, revision 0, zero jobs, zero accounted and reconciled
  seconds, cap `7200 s` = the amendment's 2 GPU-hour first-tranche ceiling;
  ticket unconsumed, single-use, one authorized allocation, 0 completed seconds,
  watchdog `7080 = 7200 − 120`, genesis pin equal to the ledger file's own
  sha256 on disk, `issued_by_slurm_job_id 13840`. `sacct` shows the only two C04
  jobs that have ever existed are `13805` (v5, FAILED `1:0`) and `13840` (v6,
  COMPLETED `0:0`, `00:00:19`), and **neither carries `gres/gpu` in AllocTRES** —
  both are `billing=8,cpu=8,mem=64G,node=1`. Consumption, claim, entry-marker,
  lock, `seal/` and `checkpoints/` are all absent; the queue is empty.
- **Authority chain and contract neutrality.** 15/15 implementation hashes;
  config `40ec6d97…`; the normalized contract `2b66775c…` recomputed
  independently from the normalization rule and equal in all four in-payload
  copies; manifest `5e56041a…` with self-consistent closure `5375c393…`; both
  records, both review transcriptions and both job logs match their pins. The
  neutrality claim was *tested*: filling the four sentinels in with the literals
  leaves the contract hash unmoved, while tampering with
  `small_cap_gpu_seconds` or `teacher_contract.num_frames` moves it.
- **Self-test honesty.** 13 prompt-hash fixtures + 7 legacy fixtures + 5 added
  by `run_self_tests` (`role_{S,P,T,H}_shape`, `no_test_paths`) = exactly the 25
  the job reported, with the fixture name set a strict subset of the manifest's
  check set. Four checks were probed for vacuity and found real, including the
  sharpest one: a freeze payload whose `payload_sha256` was **recomputed to be
  valid** but whose four keys hold the sentinel is still rejected, and
  `_raises_runtime_error` returns `False` for a no-op and for a `ValueError`,
  `True` only for a `RuntimeError`, so no negative fixture can pass by raising
  the wrong exception.

**The three Importants.**

1. **The HateMM identifier is itself label-bearing.** IDs are `hate_video_*` /
   `non_hate_video_*`, and the sealed allowlist and source manifest store plain
   IDs, so any reader of the frozen payload holds the HateMM label of all 200
   selected videos — while the access ledger stores only `video_id_sha256`,
   which then buys nothing. This does **not** break the selection: the exact
   200-ID sequence reproduces from tag/dataset/id/suffix alone, and the draw is
   demonstrably unengineered — 78 hate / 122 non-hate (0.390) against a train
   prior of 298/446 (0.401), which is what an unbiased hash draw looks like. It
   does **not** reach the teacher either: `producer.py:768` interpolates only
   `{transcript}`. It is filed because the seal must not be relied on downstream
   as label containment for HateMM, and "the teacher sees no label" is currently
   a property of one line of prompt assembly rather than a checked precondition.
2. **`selection_deterministic` is a tautology** —
   `selection_digest("HateMM","x") == selection_digest("HateMM","x")`, a pure
   function against itself, unfailable under any mutation of the rule, tag,
   suffix or digest payload. "25 checks, all_passed" is quoted as evidence in
   three documents; 24 of the 25 carry information. Non-blocking, since the
   property is independently proven by the exact allowlist reproduction above.
3. **The amendment's 8 GPU-hour aggregate C04 ceiling is encoded nowhere.** The
   payload correctly encodes the 2 GPU-hour first-tranche cap; the conditional
   full-bank tranche's 8 GPU-hour aggregate — explicitly inclusive of every
   first-tranche GPU-second — has no accumulator in any artifact and is enforced
   by prose only. Acceptable now, but the conditional tranche's code/resource
   review must add one and carry the first tranche's actual spend into it.

One observation recorded rather than filed: `frozen_payload.total_bytes`
`5,178,606` in the preceding entry is `du -sb artifacts/c04`, not the sum of the
15 file sizes (`5,174,184`). Every per-file `bytes` and `sha256` in that table is
exact against disk, so this is field-labelling ambiguity with zero integrity
impact.

**Boundary — what this GO does and does not do.** It means exactly one thing:
the payload frozen by 13840 is well-formed and faithful to the frozen contract,
so C04 is **eligible** to be considered for a separately-authorized teacher small
tranche. **It authorizes no work.** All GPU work remains blocked until explicit
main-dialogue execution authorization; `teacher_authorized`, `gpu_authorized`,
`slurm_authorized`, `small_tranche_execution_authorized` and
`post_job_reconciliation_authorized` are all still `false` in the frozen config
and this review does not change them. **Dev/test teacher remains forbidden
forever under the amendment**, and the frozen payload contains zero dev/test IDs.
OCR, external API, network, cross-dataset data, pre-seal label access, chained
submission, release, resubmission and namespace reuse all remain false.

And this markdown verdict is **not** the machine-checked authorization.
`verify_payload_review` still requires `review.payload_hash_verdict == "GO"` in
the config, a `refine-logs/C04_A0T_SMALL_V1_V6_PAYLOAD_HASH_REVIEW.json` under
schema `c04_payload_review_v6`, a 64-hex `payload_review_sha256` replacing the
`PENDING_CPU_PREFLIGHT_AND_PAYLOAD_REVIEW` sentinel that `_verified_review_file`
rejects with `HALT_REVIEW_LINEAGE`, a self-consistent `closure_sha256`, and
`attested_closure_sha256 = sha256("C04-PAYLOAD-REVIEW-GO-v6\n" +
reviewed_payload_sha256)`. None of it exists. Whoever builds it must pin
`preflight_manifest_sha256`
`06bf6b38f424dd53d142367abd029dfa1f485380fb1482d72beabb7f5943ad1a`,
`config_contract_sha256` `2b66775c…`, `code_resource_authorization_sha256`
`5e56041a…`, and `prompt_hashes`/`map_hashes`/`staged_output_hashes` equal to the
preflight manifest's. Two further items still gate the GPU stage: the
post-freeze config amendment to literal prompt hashes (the producer requires
`LITERAL_BOUND`), and the `allocation_entry_marker` non-re-runnability property
of the GPU wrapper.

No metric, result or CONTINUE/KILL verdict is published by this review. The
`+0.030 / +0.030` two-dataset target and the amendment's full-bank
`+0.050 / +0.050` DIRECT-OOF and STUDENT-OOF gates are untouched and unwaived.

---

## Gate-0 reopen — 2026-07-31 (registry adjudication, `$0`, no computation)

**Why now.** `registry_update_2026_07_28.serial_execution.fast_fail`: *"After two
consecutive active-candidate failures, reopen Gate 0 before continuing the ordered
backlog."* C01 (`KILL_CURRENT_ENDPOINT_ROUTE_ONLY`, job `13738`) and C02
(`KILL_C02_DENSITY_ORBIT_UNREACHABLE`, job `13847`) are those two measured
failures. This reopen governs the **post-C04** backlog only; the C04 lineage was
not touched and nothing in C02 was modified.

**What was adjudicated.** `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md`, a
zero-compute advisory recon recommending strikes on seven of ten backlog
candidates. It creates no registry status by itself, so it was verified rather
than adopted: four independent passes — a from-scratch re-measurement of every
`[M]` claim on the six gt label files, plus three adversarial quote/scope audits
by workers with no exposure to this adjudication's reasoning — with the C01 arm
table re-derived directly from `C01_A0_OUT.json`. The resulting record then went
to a fresh independent reviewer, which returned `REVISE (1C/3H/10I)` and moved a
further candidate out of the strike column.

**Headline.** The recon's measurement layer is sound: every `[M]` figure
reproduces, no fabricated number was found, and the two apparent numeric
discrepancies turned out to be percentile/median **convention** differences the
recon applied consistently — not errors. **But six of its seven recommended strikes
rest on a ban, a screen, a premise or a boundary clause read wider than its own
text, and are downgraded to HOLD. Exactly one survives fourteen rounds of adversarial
review — C14, and only because that candidate was already registry-ineligible
before the recon was written.**

### Dispositions

| candidate | disposition | basis |
|---|---|---|
| C05 | `held_nonisomorphism_gate_unwritten_as_posed` | registry precondition attempted; not dischargeable from any enumerated source |
| C06 | `gated_on_zero_cost_falsifier` | C01 evidence real; its bans don't reach C06's object |
| C07 | `held_lattice_delta_unwritten_reachability_unscreened` | delta un-attempted; F82 is vote-side, C07 is head-side |
| C08 | `held_title_channel_separable_route_unscreened` | premise 1 refuted: `Title` is separable on both MHC datasets |
| C09 | **`next_active_candidate_post_C04`** | zero-GPU Stage-0, legality affirmatively established |
| C10 | `held_eum_preconditions_unmet` | ban covers it by name; "space is EMPTY" leg is an extension |
| C11 | `held_thin_evidence_disjunct_unscreened` | claim is disjunctive; 2nd disjunct measured *positive* |
| C12 | `held_ban_scope_ambiguous_construction_unnamed` | F55 misread; `[5]` construction-dependent |
| C13 | `held_zh_scoped_no_cross_dataset_pairing_named` | self-scoped ZH-specific; the unmet pairing is a proponent task |
| C14 | **struck** from the performance backlog | `eligible_for_primary_target: false` |

One strike, seven holds, one gate, one promotion. A Gate-0 strike is
**registry-level and reversible by a future user ruling** — recorded as
`struck_gate0_2026_07_31`, and **not** a measured kill.

### The six downgrades, and the gap in each

**C08 — round 3's Critical.** Revision 2 restated its premise 1 as *"no **separable**
title channel without re-deriving source metadata — a data-collection act."* That is
refuted by files on local disk: `Multihateclip/English/annotation(new).json` carries
a non-empty `Title` on **`891/891`** rows and the Chinese file on **`897/897`**;
`scripts/prep_mhc.py:72-85` — the very function cited for the folding — reads title
and transcript as **separate variables**; and F88's *"medians: title 15 chars,
transcript 76, composed 96"* is only measurable from a separated title. Exposing it
is a re-run of a deterministic CPU script, so **a `≥2`-dataset title route exists on
MHC-EN + MHC-ZH** — the same shape as C12, where a leg that appeared to make the
two-dataset arithmetic impossible turned out not to. What survives: premise 2 at
*marker* scope only (`<em>` is MHC-ZH-only; MHC-EN's `64`+`9` entity rows are
ordinary apostrophe/quote/ampersand escaping carrying no source identity), and
premise 3 as corroboration only (a TIER-2 proxy read the source declines to settle).
**Unblock:** re-pose around the half that has a substrate, and price the title
channel's Stage-0 oracle **per dataset** — the *"15 characters"* figure the earlier
draft used is an MHC-ZH **test-split** median inherited second-hand — *likely*
markup-stripped, an inference rather than a reading, since neither `ERRPAT:272` nor
F88 states the convention —
and measured directly the title median is **51 characters on MHC-EN** (transcript
322) against **27 raw / 13 stripped on MHC-ZH** (transcript 78), leaving the EN
half — the one that makes the two-dataset route possible — `3.4×` larger and
**unpriced**.

**C07 — the review's Critical.** Its boundary demands two things: a written
mathematical delta against prior lattice work, and a reachability screen. The
delta is **un-attempted**, not unwritable — a different kind of fact from C05's,
where the gate was attempted and *demonstrated* unwritable, and not one that
supports a strike. And **no reachability screen of C07's object has ever been
run**: F82's own ban_scope splits the sides — *"vote-side Offensive reweighting
closed both datasets … **head-side graded auxiliary = F44-capped +
admissibility-gated, only revivable by user ruling WITH a new mechanism
argument**"* — and a cone metric is head-side. F82's `EN +0.0250 / ZH +0.0256` is a
**headwind to price**, not a screen of C07, and an earlier draft's "has been run
and fails" is withdrawn.

**C11.** Its claim is disjunctive — *"speech-poor **or thin-evidence** videos"*.
The recon's census kills only the literal-null disjunct, correctly: whitespace-only
`text` is `39/744` and `9/107` on HateMM and `0` on all four MHC splits. But the
second disjunct is **not empty on the dataset the recon says has zero instances**
— `ERRPAT_MHC-ZH_2026-07-26.md:301-306`, the `[31,76)`-char band holds *"11 of the
22 core errors in 37 items … 2.0× enrichment, permutation p = **0.0048**"*. That
leg is recorded at the same standard applied to C08's premise 3: a **TIER-2
CPU-re-mint proxy read on the MHC-ZH test split, `n = 149`, pooled effect only**
(class halves `p = 0.0506` / `0.0668`). It establishes that the disjunct cannot be
dismissed by a whitespace census — not that it is a live lever. The hold nonetheless carries a heavy burden after review restored the
clause the earlier draft truncated: that ERRPAT row ends *"**No legal unmeasured
lever found.**"* The hold survives only on the narrow distinction that ERRPAT
searched for levers supplying *better transcript signal* while C11 proposes
**representing the absence** — and if a proponent cannot articulate that
difference, C11 should be struck without further measurement.

**C12 — the deepest stretch.** Three of its four legs read past their text. (i)
Two **inference-time** nulls (archive-as-key `ΔAcc −0.0014 ± 0.0313`; AUTO two-vote
`C−A = 0`) are used against a **training-time** boundary that already excludes
them — and neither measurement ever computed a cross-version disagreement, which is
C12's entire quantity. *(An on-point precedent, cited as a general form only:
`LITSWEEP3_DATA_CENTRIC.md:80` calls the `C−A = 0` null "a headwind to price, **not
a coverage** of this mechanism." Its own subject is memory-bank curation and its
ground is that the null used an **MLLM two-vote** signal — the side C12's
archive-derived statistic sits on — so the downgrade rests on the
inference-time/training-time leg alone.)* (ii)
`banned_constraints[5]` — literally four words, *"MLLM-scores-as-training-signal"* —
is treated as settled, with the narrow-reading counter-precedent unengaged: F60/AUG
rules **MLLM-as-data-generator admissible** and killed AUG on domination, not on
`[5]` — though F60 is itself gated on the open **D7 generator-role sub-ruling**
— D7 proper is `RESOLVED 2026-07-14 (RESOLVED-NEGATIVE)`; it is the *sub-ruling* on
the generator role that is open — so that route is not a free pass either. (iii) Decisively, **F55 is misread**: its
ban_scope is *"Cross-encoder composition with ADAPTED text on EN: dead … EN closed
at all three levels"*, and F55's own detail names those levels as frozen (F50),
collapsed-adapted-deployed (B4/F53) and healthy-img+adapted-text composition (F55)
— three levels **of the encoder-composition question**, not a closure of MHC-EN
for all method families. C12 is not an encoder-composition candidate, so its
MHC-EN + MHC-ZH route is arithmetically available and the "≥2 datasets impossible"
argument fails. **Unblock:** C12 must name its construction — stability-as-weight
(then `[5]` applies **under EUM's four-authority stack, `P3 / P11` plus `[5]` and
`[6]`**, and it is dead) versus stability-as-multi-view-target (then F60
governs, subject to the open **D7 generator-role sub-ruling** — D7 proper is
`RESOLVED 2026-07-14 (RESOLVED-NEGATIVE)`).

**C13.** Its basis moved three times under review and each move narrowed it. The
recon struck it on a plausibility argument — invariance would delete a predictive
feature — which round 4 established is an **inference about** C13, not a
measurement **of** it. The replacement substrate arithmetic was measured for HTML
**tags** only, while C13's claim says *"HTML/title markup"* and MHC-EN carries
entities on `64`+`9` rows (round 5). What finally remained was registry text: the
claim self-scopes to *"a **ZH-specific** extraction nuisance"*, and the dedup
boundary **conditions** — it does not prohibit — a two-dataset route on *"a
genuinely cross-dataset mechanism"* the entry never names. Round 6 established that
an un-named pairing is a **precondition a proponent can satisfy**, which this
reopen's own C07 ruling holds cannot carry a strike. **Unblock:** name the
cross-dataset mechanism, and show the invariance does not delete predictive signal
against a headwind of `5×` the no-markup rate and `8×` the bare-keyword rate. The
measurement itself survives the disposition and is the more valuable output — it is
recorded as paper material.

**C10.** EUM's ban covers C10 by name on C10's own language. But the *"legal
unit-definition space is EMPTY"* leg is an extension: EUM concludes emptiness
*"as of this recon"* over a three-item enumeration, and a rule-based, gold-free,
MLLM-free boundary is not in it — which is precisely what "gold-free" claims. The
BSY block is scoped in its own text to *"bank-**ADDITION**"* candidates and is a
procedural block pending a user ruling. HOLD is the right disposition because the
burden already sits on C10 via EUM's three written preconditions.

### Corrections that changed nothing but should not propagate

- **LBOP exists.** The recon's `[M]` grep (zero hits) is true as scoped but its
  inference — that C07's dedup boundary points at nonexistent work — is refuted:
  `research-wiki/TARGET_GATE0_ITER6_LITERATURE.md:246-288` carries a full LBOP
  spec with its own gates, and `:444` retains LB-SCGP, **LBOP** and RHT as three
  distinct candidates.
- **F82's graded structure is richest on EN, not ZH** (Offensive 73.2 % of
  positives vs 62.8 %), and the record gives **both** oracles: EN `+0.0250`, ZH
  `+0.0256`.
- **C02's fold-head match was paired against the wrong arm.** `0.8875 / 0.8912` is
  C02's FULL *treatment* arm; the correct statement is that C02's
  `gates.ARENA2.pooled_native_acc` matches F113's fold-head floor on **6/6 seeds**
  at the 4-dp precision F113's banked artifacts record.
- **The banked ro-cache inventory is misstated:** HateMM has only `-LoRA-curric`
  and MHC_zh only `-LoRA` — one adapter lineage each, not a matched pair — and the
  `ow_` cells change the readout span as well as the prompt. Both constrain the
  C06 falsifier's design.
- **Filed, not acted on:** C02's five named validity gates live in
  `C02_A0_OUT.json` (not `C02_A0_DECISION.json`), and **four** of them have a
  machine `pass: true`; `ZERO_CONTRACT` has no `pass` field, two of its four
  criteria are `DOCUMENTARY_CITATION_NOT_COMPUTED`, and on MHC-ZH its population is
  empty. The phrase "all five validity gates passed" originates in the existing
  C02 finding, not in the recon. C02 is out of scope here and nothing about it was
  edited.

### The reopen's main output — conversion, not reach

The recon's §9 claimed **three independent ceilings**, each under the Stage-0 bar.
Verification withdraws both the independence and one of the scopes:

- The coverage bound (`≤ +0.0171` head-space) bounds **pool expansion only**.
  F114 names this exact misuse in advance: *"K-HC-1's coverage bound bounds POOL
  EXPANSION not purity-within-a-fixed-pool, as F107 sec6.1 itself states."* The
  rank-constrained **re-ordering** oracle is `+0.0780 / +0.1123 / +0.1876` at pool
  20 — 2.6× to 6.3× over the final bar — and the re-ranking family is closed by
  **reduction** to the measured-dead AGGNET/VSW family, not by an under-bar oracle.
- The ceilings are not independent: the head-space coverage leg and the
  purity-conversion leg are the same 90-cell, `n = 78`, MHC-ZH-dev proxy-head
  table; the permutation cap is a strictly dominated sub-case of the first's
  function class by `LITSWEEP8:200-201`'s own words.
- *"Exchange rate never above 1.17 in 36 cells"* is true only of F95's
  pair-verifier battery and is stale campaign-wide (F96 `1.8889`/`2.2353`, F98
  `1.8333`, F105 `6.0000`, F112 `2.8889`) — which the recon itself correctly quotes
  fourteen lines later. F99's zero-break *upper* bounds are unaffected.

**What survives is sharper than what was claimed.** AGGNET carried an oracle of
`+0.1492 / +0.1520 / +0.2186` — in F98's own words *"by far the largest oracle
ceiling any member of this family has ever had"*, with 96-100 % of every deployed
error inside its function class — and delivered
`+0.0134 / −0.0069 / +0.0000`. Its own epitaph is the finding: *"What binds is
neither reach nor capacity but that the local configuration carries no learnable
signal about which neighbours to trust at n = 549-744."* So **a large oracle is no
longer evidence *for* a candidate in this channel — it is the precondition every
failed candidate already met.** Gate 0 must now ask a candidate why its mechanism
*converts*, in the currency `banned_constraints[10]` already names: NET ITEMS
against `22.3 / 17.4 / 16.5` for `+0.030`, i.e. `37.2 / 29.0 / 27.5` for `+0.050`,
with exchange rate explicitly **not** a screening criterion.

**And the backlog needs replenishment, not re-ordering.** After C04 resolves it
holds one candidate with a live mechanism and a zero-cost Stage-0 (C09), one `$0`
falsifier (C06), and seven holds each blocked on a condition someone must
affirmatively satisfy.

### Effective post-C04 order

`C09` → (`C06` `$0` falsifier) → `C05`/`C07`/`C08`/`C10`/`C11`/`C12`/`C13` only on
their named unblock conditions. The historical `ordered_backlog` array is left untouched as
the record of what was ordered on 2026-07-28.

**C09 goes first and alone.** It is the only candidate aimed at the population the
error forensics actually found (F88: `89-93 %` seed-invariant on HateMM; ZH `22`
of the 25-item union wrong 3/3 with **nothing at exactly 2/3**), the only one whose
Stage-0 needs no extraction, and the only one adjudicable at zero GPU while C04's
tranche holds the serial-execution lock. Its label-use legality is **affirmatively
established**, not merely un-banned — `progress.json:25` and
`LITSWEEP3_DATA_CENTRIC.md:82` — though under a ruling whose viability premise
`LITSWEEP5_COMPLETENESS.md §4(ii)` flags as stale — **and that counter-text is
itself downgraded, not vacated**: §4(ii)'s selector leg rests on the F114-retracted
*"CLIP LOO 0.998"* premise (deployed Qwen heads are `0.9406 / 0.8915 / 0.8154`) and
`LITSWEEP5` is not among the nine records F114 corrected, while its independent
train-disagreement leg (`0/109`, `0/102`, `0/92`) is untouched. So C09 inherits a
weakened prior rather than a formality, but by less than the counter-text's wording
implies. Its sharpest objection is the Feldman one, whose
numerical leg is already retracted in-repo while its substantive leg stands; the
Stage-0 design makes it an explicit **measured discriminator**, not a caveat.

### Review

**Fourteen rounds** of fresh independent review, closing at **`GO (0C/0H/0I)`**, scoped to strike fidelity, status
scoping and measured-vs-inferred.

**Round 1: `REVISE (1 Critical / 3 High / 10 Important)`.** The Critical moved C07
from strike to hold; the three Highs restored a truncated clause in each of C08,
C11 and C12. The reviewer independently re-derived the census, re-verified the C01
arm table cell-for-cell, confirmed the C12/F55 leg, and added a stress test this
record had not run: rows carrying a harvested keyword **without** the `<em>` markup
hate at only `10/140 = 0.0714` against `141/243 = 0.5802` with it — so the tag
itself carries the signal and C13's regression inference survives.

**Round 2: `REVISE (0 Critical / 3 High / 7 Important)`.** No Critical; the C07
downgrade, all four downgrades, the three strikes, the reversibility language and
the untouched `ordered_backlog` were checked and cleared. Its most useful finding
was procedural: **four round-1 repairs had landed in the narrative record but not
in the machine-readable disposition block**, which a machine consumer reads. It
also scoped C11's `ERRPAT §5.2` leg to its tier and split, restored F82's *"HateMM
out of scope (no Offensive class)"* clause and the dev-split resolution behind
`ZH +0.0256` (**2 dev items**), restored the *"(though Wall-A still caps the
achievable magnitude)"* parenthetical to C09's legality quote, corrected two
pinpoint citations, and renamed C05's status to `unwritten_as_posed`. All applied.
One round-2 finding was **partly refuted on re-check**: the superlative *"largest
oracle ceiling ever measured on this object"* **is** verbatim in
`directions_tried.json`'s F98 entry (round 2 checked only `findings.jsonl`), but
`findings.jsonl` is narrower, so the family-scoped phrasing is now used everywhere
and the disagreement between the two primary records is recorded.

**Round 3: `REVISE (1 Critical / 1 High / 4 Important)`.** Its Critical moved C08
out of the strike column after locating the source annotation files that refute its
premise 1. It also caught a transposed entity triple introduced by a round-2 repair
(`43 / 16 / 17`, not `43 / 17 / 16`), required `banned_constraints[5]`/`[6]` to be
stated as **construction-dependent** at C05 and C07 under the same EUM-vs-F60
tension the record already sets out at C12, and fixed two more pinpoint citations.
It confirmed both prior rounds' findings as applied, verified round 2's H-2
adjudication as honest, and cleared all downgrades, both surviving strikes, C06's
gate, C09's promotion and the strategic finding.

**The pattern, which is the reopen's methodological finding.** Every Critical across
three rounds was the same defect in a different place: a candidate recorded as
struck on evidence that, read at its own written scope, does not close it — C07 on a
screen never run for its object, C08 on a premise the source files refute. Both were
caught only because each round was handed the primary sources and told to be
adversarial. **Of the recon's seven recommended strikes, one survives.** That is not a criticism
of the recon, which was explicitly advisory and whose measurement layer proved sound
throughout; it is the reason the registry requires adjudication before a recon moves
a status.

**Round 4: `REVISE (0 Critical / 2 High / 2 Important)`.** It audited **all thirty**
prior findings against all four surfaces and confirmed every one genuinely applied,
including in the JSON that rounds 2 and 3 had both caught lagging. Its two Highs
were a grouping error (C08 filed under the "Strikes CONFIRMED" heading in the record
only — every other surface had it right) and a pricing error: C08's unblock costed
the title channel from *"title 15 chars / composed 96"*, an **MHC-ZH test-split,
markup-stripped** median inherited second-hand, when the title median measured per
dataset is **51 characters on MHC-EN** against **27 raw / 13 stripped on MHC-ZH** —
so the EN half, which is what makes the two-dataset route possible, is `3.4×` larger
than the figure the "thin channel" verdict rested on and is now recorded as
unpriced. It also withdrew a quotation that had no source, corrected an F113/F114
attribution, and **re-based C13's strike on ban-free arithmetic**. It cleared all
five downgrades, both strikes, C06's gate, C09's promotion and legality, the
strategic finding and the zero-touch boundary, and confirmed — asked explicitly —
that **no hold is over-cautious**: each names a usable unblock.

**Round 5: `REVISE (0 Critical / 2 High / 2 Important)`.** Thirty-three of the
thirty-four prior findings were confirmed applied on all four surfaces; the
exception was round 4's own repair lagging on *this* file. Its second High mattered
more: **C13's remaining ban-free leg was measured for HTML *tags* only**, while
C13's claim says *"HTML/title markup"* and MHC-EN carries entities on `64`+`9` rows
— the identical scoping defect round 1 forced out of C08, this time carrying a
surviving strike alone. C13 is therefore re-based on **its own registry text**: its
claim declares the target *"a ZH-specific extraction nuisance"* and its boundary
conditions any two-dataset route on a cross-dataset mechanism it never names, so
the census now corroborates rather than establishes. It also caught that the
record's provenance attestation had gone stale, and that the JSON filed the *gate*
inside the `held` array.

**Round 6: `REVISE (0 Critical / 1 High / 2 Important)`.** Its High closed the C13
question: after round 5's re-basing, the surviving basis was an **un-named
cross-dataset pairing** — a precondition a proponent can satisfy, which this
reopen's own C07 ruling holds cannot carry a strike. **C13 moved to hold**, and the
paper finding was preserved. It also completed the provenance attestation and
softened an unranked superlative ("the strongest lexical shortcut") to what the
measurement supports: `5×` the no-markup rate and `8×` the bare-keyword rate.

**Exactly one of the recon's seven recommended strikes survives fourteen rounds** —
C14, and only because it was already registry-ineligible before the recon was
written.
Every other collapsed the same way: the evidence offered closed something narrower
than the candidate, or named a precondition a proponent could still satisfy. **That
is the reopen's methodological output**: an advisory recon is a hypothesis-generator,
and the gap between "this candidate looks unpromising" and "this candidate is
closed" turned out to be seven candidates wide.

**Round 7: `REVISE (0 Critical / 2 High / 3 Important)`.** Asked explicitly, now
that only one strike remained, whether any hold was **over-cautious**, it confirmed
none is — at C10 in particular, EUM's ban does name the object but supplies its own
three revival preconditions, so a strike would over-read a conditional closure as
an absolute one. Both its Highs were round-6 repairs lagging on a surface. Its most
useful Important cut *against* the candidate this reopen promotes: the C09
counter-text (`LITSWEEP5 §4(ii)`) rests its selector leg on the **F114-retracted**
*"CLIP LOO 0.998"* premise, and `LITSWEEP5` is not among F114's nine corrected
records — so that headwind is **downgraded, not vacated**, while §4(ii)'s
independent train-disagreement leg (`0/109`, `0/102`, `0/92`) is untouched and
stands.

**Round 8: `REVISE (0 Critical / 2 High / 4 Important)`.** Tasked to make the
**cross-surface sweep** its first and most exhaustive job, since every round from 2
to 7 had caught a repair lagging on exactly one surface. No disposition changed. It
found that **round 7 did not exist on three of the four surfaces**, that a round-7
count repair had left a self-contradiction in this file's own headline, and that
round 7's C09 downgrade had not reached the two narrative surfaces. It also
narrowed a claim the record had over-stated — D7 itself is `RESOLVED 2026-07-14
(RESOLVED-NEGATIVE)`; what is open is the **D7 generator-role sub-ruling** — and
re-scoped a precedent (`LITSWEEP3:80`) whose own subject is memory-bank curation.
Along the way it produced a **new corroboration** of the paper note: the ZH `<em>`
markup lives in **391 `Title` fields and 0 `Transcript` fields** of the source
annotation, independently confirming the marker rides on the harvested title.

**Round 9: `REVISE (0 Critical / 1 High / 3 Important)`.** No disposition changed.
Its findings were that three of round 8's six repairs had not actually landed —
including, again, on this file. Its most substantive was an **attestation** defect:
the provenance line claimed the source annotation files were *"joined only to
train+val ids"*, but the `Title`-presence counts `891/891` and `897/897` (and round
8's own `391 Title / 0 Transcript` corroboration) are **whole-file**, spanning test
ids. Nothing improper followed — only `Title`/`Transcript` were read, no label was
consumed, and the join-scoped counts (`629/629`, `657/657`, `277 / 0`) make the
identical point — but the attestation is now stated exactly, with an explicit "no
label or prediction was read for any test id".

**Round 10: `REVISE (0 Critical / 1 High / 3 Important)`.** No disposition changed.
It re-derived the `[M]` layer independently and reported **18/18 claims matching
exactly**, recomputed the C01 arm table cell-for-cell, and confirmed the 6/6
fold-head identity — **no fabricated number**. Its High was, for the third
consecutive round, that the previous round did not exist on the two narrative
surfaces. Its Importants were citation precision: the C08 title-median anchor
(`ERRPAT_MHC-ZH:272`, not `:270-271`); that EUM reaches "MLLM-derived boundaries or
weights" through a **four-authority stack** rather than by glossing
`banned_constraints[5]` alone; and that F80 was quoted at partial scope — its
ban_scope opens with an **unconditional** on-dataset closure and the "without new
mechanism" conditionality attaches only to *elsewhere*. None changed a warrant.

**Round 11: `REVISE (0 Critical / 1 High / 2 Important)`.** No disposition changed.
It re-derived all 18 census claims and added a check the record had not made: round
1's stress test reproduces **only** under the train-only keyword set
(`10/140 = 0.0714`; train+val gives `10/146`), confirming the record's scoping. It
recomputed the C01 arm table across all 14 arms × 2 datasets to `<1e-9`. Its High
was the F80 scope repair lagging on `TARGET_FINDINGS.md`; its Importants marked
*"markup-stripped"* as an **inference** rather than a source reading (neither
`ERRPAT:272` nor F88 states the convention, and F88 describes the title as
*carrying* the markup), and narrowed a claim that F112 corroborated F113's
TEST-transfer caveat — it corroborates only the raw-vs-head half.

**Round 12: `GO (0 Critical / 0 High / 2 Important)`.** The first GO. It cleared all
four scope items explicitly — the one strike faithful and in-scope; all ten statuses
the correct kind of record with byte-exact reversibility language; **no inference
recorded as a measurement**, every inferential step labelled as such; and all eight
unblocks concrete and proponent-actionable with **none over-cautious**. It verified
through four independent passes, recomputing the C01 arm table across all 14 arms ×
2 datasets to `<1e-12` and the fold-head identity at exactly the precision claimed.
Its two Importants were bookkeeping residue — §§3.5/3.6 still calling C07 "struck"
five rounds after it was downgraded, and the C12 unblock re-asserting "EUM's gloss"
35 lines after the record corrected it to a four-authority stack — both applied
along with its observations.

**Round 13: `GO (0 Critical / 0 High / 1 Important)`.** A second GO, again clearing
all four scope items with the `[M]` layer recomputed at 100 % reproduction and the
C01 arm table exact to `<1e-12`. Its single Important was one JSON string —
the C12 unblock's *"under EUM's gloss"*, which rounds 10 and 12 had both charged and
which survived because the repair matched a variant with a comma. Its observations
added a caveat worth carrying: `banned_constraints[10]`'s net-item figures are
**train-arena** requirements at `n = 744 / 579 / 549`, so a proponent applying the
Gate-0 currency to a test-sized arena would mis-scale by roughly `3.5×`. It also
noted, against the record's own interest, that on C11 the record applies ERRPAT's
*"No legal unmeasured lever found"* **more narrowly** than ERRPAT's own broadest claim.

**Round 14: `GO (0 Critical / 0 High / 0 Important)` — the closing verdict.** Four
independent fact-check passes plus the reviewer's own recomputation cleared all four
bar items with **zero findings at any severity**: the census re-derived from scratch
twice at 100 % reproduction, the C01 arm table recomputed at every one of 28
arm-cells, §3.7 recomputed as identical at 4 dp on 6/6, and — in the reviewer's own
words — **"No hold should have been a strike, and no unblock is vacuous."** Its
observations were applied anyway, and five tighten the record *against* its own
arguments: a **direct** single-authority gloss of `[5]` does exist (F103, on an
archive field), so C12's safer-looking branch is not free; F108 is now named in
C08's unblock; C06's six "random rotations" are **one Givens parameter family** that
contains the primary, a sharper and more adverse reading, with two omitted arms
restored because leaving them out understated the adverse case; LBOP-0's fuller bar
is stated at the point of use; and C10 gains an unpriced headwind — EUM records that
a legal rule-based unit was **already built and measured negative**.

**Where fourteen rounds leave it.** Two Criticals, both moving a candidate out of
the strike column. No disposition has moved since round 6, no Critical since round
3, and the last three rounds all returned GO. **The adjudication is closed.**

Records: `refine-logs/GATE0_REOPEN_2026-07-31_REVIEW_ROUND{1,...,14}.md`,
request `..._REVIEW_REQUEST.md`, raw appended to `TARGET_REVIEW_RAW.md`.

### Boundary

This reopen publishes no metric, delta, result or CONTINUE/KILL verdict for any
candidate; consumes no scientific gate; and authorizes no work. No job was
submitted, no hash frozen, no preregistration frozen. `refine-logs/C09_A0_PREREG_DRAFT.md`
is a **draft only** — its CPU job waits for C04's tranche to terminate
(serial-execution precedent) and for main-dialogue authorization. Cost `$0`: zero
GPU, SLURM, Modal, teacher call, model load, cache write and test-split contact.
Measurement provenance in full: the gt census files
`data/gt/{HateMM,MHC_zh,MHC}/{train,val}.jsonl`;
`/data/jehc223/Multihateclip/{English,Chinese}/annotation(new).json`, from which the
per-dataset title/transcript medians are **join-scoped to train+val ids** while the
`Title`-presence counts and the `<em>`-location corroboration were taken over **all
rows**, reading only the `Title` and `Transcript` fields and consuming **no label**
— join-scoped those counts are `629/629` and `657/657`, and `277 / 0`;
recomputation from the already-banked `C01_A0_OUT.json` and
`C02_A0_{OUT,DECISION}.json`; and the six banked `headspace_arena_*_OUT.json`. All
`$0` and permitted; none is a new experiment; **no label or prediction was read for
any test id.**

Full adjudication: `refine-logs/GATE0_REOPEN_2026-07-31.md`; state block
`gate0_reopen_2026_07_31`.

---

## 2026-07-31 — C04 implementation-v7: five review rounds, one Critical caught, tranche submitted

The v6 payload review returned `GO (0C/0H/3I)`. Closing those three Importants
required editing hash-pinned implementation files, and `config_contract_sha256`
does not normalize `implementation_hashes`, so v7 is a full namespace rebuild in
the v5→v6 discipline. **No v6 byte was edited** (all 15 v6 hashes verified).

**The headline is not the closures.** While closing I-1 (teacher-visible
containment) I had to render a prompt outside `build_messages`, and it raised.
`c04_a0t_small_v1_v6_producer.py:768` — the only render call site in the v6 tree
— used `PROMPTS[form].format(transcript=...)`, and both templates embed
`_SCHEMA_TEXT` whose literal JSON braces `str.format` reads as replacement
fields. Verified by static `ast` parse of the frozen v6 bytes: both forms raise
`KeyError: '"source_relation"'` unconditionally. **No v6 fixture ever rendered a
prompt**, so all 25 v6 checks passed with the defect live, and the v6 payload
reviewer read line 768 correctly without executing it. Submitting v6 as
authorized would have consumed the single-use ticket, entered the no-clobber
namespace, loaded 7B weights onto the A100, and died on the *first* forward.

That is the third instance of one family — v5's static gate (13805), v6's
`mark_exit` genesis bump, and this. Five independent code/resource rounds then
found six more of the same shape, including a GPU wrapper that `mkdir`-ed the
no-clobber namespace *before any code read an authorization flag*, and a
campaign ledger that bricked itself by writing an over-cap row and then raising.
**The lesson is now explicit: the CPU preflight must round-trip every record
against the contract that will read it, and no stage may take an irreversible
step before the check that would reject the run.**

Verdicts: `REVISE 2C/2H/3I` → `REVISE 0C/2H/3I` → `REVISE 0C/1H/3I` →
`REVISE 0C/1H/0I` → **`GO 0C/0H/0I`**; payload review **`GO 0C/0H/8I`**.
CPU preflight job `13850` COMPLETED 0:0 in 19 s, 57 checks, zero GPU.

**No scientific semantic moved**: the v7 selection recomputed from the frozen
rule reproduces the v6 frozen allowlists exactly on both datasets, and the four
prompt hashes still equal the v6 freeze. Two reviewers re-derived this
independently.

Two rulings the lead should note: the campaign accumulator is a new
campaign-scoped file (the namespace ledger is hash-pinned and cannot be extended
in place), and its **effective ceiling today is 7200 s, not 28800 s** — the
amendment binds the approved first tranche at 2 GPU-hours across all C04 jobs,
and only the conditional full-bank tranche raises it to 8.

GPU job `13852` submitted — the one authorized teacher small-tranche submission.

---

## C04 impl-v8 — rebuilt after the v7 vision OOM, submitted 2026-08-01

v7's allocation (job `13852`, 1978 GPU-seconds) was lost to a 110.50 GiB
allocation failure in the Qwen2.5-VL **vision** attention: the producer passed no
`max_pixels`, so a 1080x1920 MHC-ZH item entered the vision tower as 43,056
pre-merge patch tokens. No v7 record was salvaged — the cap also changes what the
teacher sees for 178/200 HateMM items, and uniform teacher input across one
sealed tranche is a scientific bar.

v8 is a full namespace rebuild with four flagged changes: the tranche reservation
7200 → **5222 s** (the phase ceiling is unchanged; 1978 + 5222 = 7200 exactly);
`max_pixels = 151200`, the cap seven other Qwen2.5-VL entrypoints in this
repository already use, plus a fail-closed 4096-token ceiling checked before
`model.generate`; a **measured** pre-submit projection gate with two states and no
third; and the 400 frame packs moved from the GPU job to the CPU preflight.

That last change is the one that mattered. Measured: it removed **834.7 s** of
decode and PNG-encode from the allocation. The conservative projection is 3817.6 s
against a 4022 s window — with the frame work still in-job it would have been
4652.3 s and **the gate would have failed**.

Two fresh independent reviewers, each with zero exposure to the build reasoning
and the payload reviewer additionally blind to the code/resource review, returned
`GO 0C/0H/0I` and `GO 0C/0H/5I`. Between them they recomputed sacct, the v7
timing basis (refitting the regression themselves), the campaign hash chain, all
414 staged hashes, all 3200 PNGs, all 400 video SHA-256 over 3.43 GB, the
selection on both datasets from their own implementations, the role maps and JL
payloads byte-for-byte, and the visual geometry for all 400 items through **both**
processor paths, and executed 23 negative fixtures between them.

CPU preflight job `13855` COMPLETED 0:0 in 976 s, no GPU, 74 self-test checks,
414 staged outputs, 802 audit events. Selection reproduces the v6/v7 frozen
allowlists exactly on both datasets and the four prompt hashes still equal the v6
freeze. All 200 HateMM frame packs are byte-identical to the ones job 13852 wrote.

GPU job `13857` submitted — the one authorized teacher small-tranche submission.
Two caveats travel with whatever it returns: a breach after a complete HateMM half
still yields no seal and no verdict, and **24/200 MHC-ZH items carry fewer than 10
transcript scalars**, so an MHC-ZH kill would not be separable from input poverty.

**VERDICT (2026-08-01): `KILL_C04_TEACHER_SEMANTIC_RELIABILITY`.** GPU job `13857`
COMPLETED 0:0 in 2668 s — 800/800 prompt records, seal published, no breach. All
eight slot cells and both joint gates fail the taxonomy frozen since v6; the
nearest miss is 0.26 absolute short. A fresh reviewer re-implemented the parse and
the slot logic from the frozen design and reproduced every rate bit-for-bit (0
disagreements over 7,200 fields). The failure is the teacher, not the harness: a
*lenient* parser makes the joint rate **worse**, and the perfect-parser ceiling is
structurally 0.36-0.65 against a 0.85 bar, because the two prompt forms genuinely
disagree where both parse. The MHC-ZH transcript-poverty caveat is discharged —
the kill survives deleting every short-transcript item at every cut.

Campaign accumulator: revision 2, 4646 s of the 7200 s first-tranche ceiling
(actual sacct seconds, not the 5222 s reservation), leaving 2554 s. Labels remain
closed. **Stopped here**: no full-bank submission under any outcome; phase advance
needs PASS + a fresh result-to-claim GO + main-dialogue authorization. C04's
scientific gates are untouched and unwaived.

---

## C09 Stage-0 preregistration — design converged to GO, nothing frozen, nothing run (2026-08-01)

**Status: `DESIGN_GO_0C_0H_0I_AT_ROUND_17 — NOT FROZEN, NOT IMPLEMENTED, NOT SUBMITTED`.**
Cost `$0`. The single CPU submission the brief authorized is **unspent**.

**Design of record: `refine-logs/C09_A0_V17_RECORD.md`.** v1 (`C09_A0_PREREG_DRAFT.md`)
and v2–v16 are superseded in full and must not be implemented. Sixteen independent
review records sit beside them (`C09_A0_PREREG_DRAFT_REVIEW_ROUND1.md`,
`C09_A0_V{2..17}_PREREG_REVIEW.md`), each by a fresh reviewer with no exposure to the
repair reasoning.

**Trajectory.** `4C/8H/10I → 1C/8H/10I → 1C/8H/11I → 1C/4H/9I → 0C/3H/10I → 0C/2H/10I
→ 0C/1H/4I → 0C/1H/3I → 0C/0H/3I → 0C/0H/2I → 0C/0H/2I → 0C/1H/2I → 0C/0H/5I →
0C/0H/3I → 0C/0H/5I → 0C/0H/3I → GO`. No Critical after round 4; no High after round 12;
the last five rounds found only statements the document made about its own repair
history or provenance locators. Round 17's bottom line: *"The science is finished and I
could not move it — for the ninth round running… The DESIGN is ready to hash-freeze."*

**The four repairs that changed what the A0 can decide.**

1. **The identifiability probe was being fed the answer.** v1's feature set opened with
   top-20 purity, which §4.4 defined against the item's own gold label — and
   `ERRPAT_HateMM:143-144` measures that below 0.5 for 24–27 of 26–28 errors, so the
   feature nearly *was* the target. Worse, v1's discriminator had **no row for the
   realistic outcome**: H-MEMORISATION does not predict AUC ≈ 0.5, because Feldman's
   singletons *are* the low-density weak-margin items a label-blind feature set
   separates under either hypothesis. The repaired probe is **conditional and
   incremental** — `ΔAUC = AUC_strat(FULL) − AUC_strat(BASE)` within frozen label-blind
   configuration strata — with an item-level **marginal** permutation null and an exact
   **within-`ITEM-STRATUM` conditional** null, **both** required for a CONTINUE, on an
   analysis set recomputed per scoring fold so that the threshold, the class composition
   and the fitting target all use the same out-of-fold `τ_hi^{(f)}`.
2. **`K-DEG` was disabled at the operating point that matters.** F96 makes the
   degeneracy control a standing gate and F98's DEG-A measures **prediction-vector
   agreement over all `n` items** (`aggnet_pregate.py:534`); v4 applied their `0.95`
   line to **selected-set overlap**. At `k/n ≈ 79/744` those differ by
   `pred_agree = 1 − 2k(1−ov)/n`, so the gate demanded `pred_agree ≥ 0.9894` and would
   have let an `S` three-quarters identical to a bare threshold shift carry a CONTINUE.
3. **The conversion currency was softened against C09's own authorising entry.**
   `K-NET` is now bound to the reopen's `dispositions.promoted.bar` figure —
   **`37.2 / 29.0`** — with a `+0.050` macro-F1 leg, and the `+0.030`-sized pair reported
   as a declared secondary that can never create a CONTINUE.
4. **The Stage-1 seam is now named, and it runs against C09.** §11 names the successor
   concretely (a global, symmetric, train-label-supervised reshaping of the head map),
   shows it is the **encoder-frozen** GAP-7 object rather than the **encoder-level**
   object C09's registry claim registers, and records that **F51 reaches the latter**
   — so a CONTINUE would license an operator *narrower* than the registered claim. F99's
   closed-form cap on the successor's set-preserving channel (`≤ +0.0279 / +0.0470`,
   zero-break) and F66's re-open clause are both added to the binding Stage-1
   adjudication list. **A CONTINUE is pre-declared void if no such operator can be
   named at Stage-1 entry.**

**Arithmetic established this session, derived rather than asserted.** Transferring
**both** F88 ratios per-seed-denominated from `ERRPAT_HateMM §1/§1.1` (final rows
`28/26/26`; consensus `25 / 2 / 1`) and `ERRPAT_MHC-ZH §2` (`23/24/22`; `22 / 0 / 3`)
onto this arena's **measured** per-seed error means `253/3 = 84.3333` and
`187/3 = 62.3333` gives **`|P_0| ≈ 79 / 60`**, **`n_unstable ≈ 9 / 8`** and
**`|P_{τ_hi}| ≈ 40 / 30`**, closing `Σ_s|E_s| = 3|P_0| + 2a + b = 253 / 187` exactly.
The long-carried `76 / 55` was mis-derived — the ZH figure applied F88's **union**-
denominated rate (*"22 of the 25-item union"*) to a **per-seed** base — and is retired.
Consequences: `K-NET` is **arithmetically unreachable** for `k ≥ 132` (HateMM) or
`k ≥ 96` (MHC-ZH); and the `τ_hi` co-primary is **live but marginal on both legs on both
datasets** — reach `40/744 = 0.0538` and `30/579 = 0.0518` against `+0.050`, and the
`p_w ≥ 30` power rule needs `≥ 75 %` of HateMM's positives and **all** of MHC-ZH's in
two-class strata.

**Stability of the decision machinery.** `§8` (nine HALT gates plus three reporting
instruments) and `§9` (the whole decision rule) are **byte-identical from v8 through
v17** — `md5 a17b56954ee6955013327f82a03904f7` over `## 8. Gates` to the line before
`## 10.` — independently re-verified by three separate reviewers.

**Pre-declared expectation: BAND B.** F97's own `ban_scope` carries the
*"HONEST POSITIVE DATUM"* that F47-features-as-adjudication-gate is real and
permutation-validated at `+0.0269 / +0.0104 / +0.0182` and *"nonetheless SUB-BAR on all
three"*, and F98 banks `+0.0269` as this feature family's ceiling — both far below the
`+0.050`-sized bar, and both **raw-arena** numbers that F113's non-transfer makes *more*
adverse in head space. Identifiability without conversion is the expected outcome, is
written in as a KILL before the run, and is explicitly **not** a Feldman confirmation.

**What does NOT exist.** The analysis script and the sbatch driver were **not written**;
no config, no frozen sha256 set, no `C09_A0_RECORD.md`, and no C09 namespace anywhere
(`artifacts/c09*`, `configs/c09`, `scripts/analysis/c09*`, `scripts/slurm/c09*` — all
verified absent). **No metric, result, decision or verdict exists. C09 is neither killed
nor continued, and the `+0.030 / +0.030` two-dataset target remains active and unmet.**

**Next step, in order:** implement the analysis script, the config and the CPU sbatch
exactly as v17 specifies → a **separate** independent code/resource review to
`GO (0C/0H/0I)` → hash-freeze into `refine-logs/C09_A0_RECORD.md` with the
measured-claims register (C02 v9 house style) → the three submission preconditions
(re-reviewed ✓; C04's tranche terminated ✓, jobs `13857`/`13862` COMPLETED; explicit
main-dialogue authorization) plus the immediately-prior `squeue -u jehc223` empty-check,
sha256 re-verification and namespace-absence check → **one** 8-CPU / 32-GB / no-GPU /
no-`--time` submission of ≈115 CPU-minutes.
