# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable part of hateful-video RGCL, and do not stop until one frozen method improves **final test accuracy and macro-F1 by at least +0.030 absolute each** over the moving strongest same-protocol non-MLLM RGCL comparator on **at least two datasets**, using paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied sparse neighbour events, absolute verdicts, static segment salience, extra embeddings, auxiliary semantic fields or a competing native head. They were sparse, redundant with the video label, absorbed by the fusion head, or merely redistributed accuracy between head and memory. SSR now adds decisive evidence: even its optimistic all-candidate OOF oracle touched only 2/7 EN and 3/15 ZH unique MI/SC error queries and could not reach its dual-metric gate. The successor must therefore provide a **reliable, dense, per-training-video causal signal** that directly changes the listwise gradient of the same full-video embedding geometry used by the final kNN memory.
- **Non-goals:** Localization-only, explanation-only, audit/guard-rail-only or native-head-only success; test-time MLLM annotation, judging, score fusion, reranking or veto; simple MLLM score/embedding/rationale concatenation; static segment weighting or segment-weighted memory; generated counterfactual content; a second parallel method stacked with SSR; gains primarily from a larger model, more data, more epochs/steps, ensembling, altered preprocessing, altered checkpoint selection, changed retrieval/voting, changed labels or any protocol relaxation.
- **Constraints:** The **only gold supervision is the video-level binary label**. No segment-level gold annotation exists or may be assumed. Every MLLM modality, coalition, necessity, sufficiency, preservation, stance, target, mechanism, rationale, localization or segment output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold, dense annotation or oracle evidence. Validation/test receive no such annotation or pseudo-signal; low-confidence, missing or invalid train pseudo-signals deterministically reduce to the exact non-MLLM path. Use the exact strongest per-dataset RGCL comparator, fixed splits/preprocessing/labels/epochs/checkpoint rule/retrieval/seeds. Required controls are remove-MLLM, within-split signature shuffle and calibrated noise/corruption, with coverage/confidence/fallback reporting. All later compute is SLURM-only in `HateVideo`, without `--time`, within 2 GPUs/16 CPUs/128 GB.
- **Success condition:** On at least two datasets and paired seeds 0/1/2, both final accuracy and macro-F1 gain at least +0.030 over `max(historical strongest point, paired baseline mean)`; every seed delta is positive; mean±std and hierarchical paired-bootstrap uncertainty are reported; the four dataset×metric primary tests pass Holm-corrected familywise α=0.05 with 95% lower bounds above zero. The full method must beat remove-MLLM and shuffled-MLLM controls with same-direction paired effects and 95% CIs excluding zero in both metrics, survive calibrated corruption, improve the **kNN readout itself** without head↔memory redistribution, and retain a defensible retrieval-specific novelty claim. Any missing metric/dataset/seed/statistical/mechanism/supervision/protocol item is `not_working`, not success.

## Anchor Check

- The end point, datasets/seeds/statistics, final kNN locus, train-only MLLM role and no-segment-gold contract are unchanged.
- No reviewer request creates drift. Strength matching and fold-local formulas only make the same mechanism identifiable.
- We reject any additional proxy, learned calibration, teacher, module, test-time signal or segment objective.

## Simplicity Check

- **Dominant contribution:** privileged MLLM coalition semantics define a broadly active conditional neighborhood gradient for final memory.
- **Method modules:** still zero new trainable components, one teacher cache and one loss.
- **Controls:** one low-level proxy, now strength-matched; Label-only, shuffle and noise remain necessary controls, not contributions.
- **No new metrics:** TV is descriptive; `R` and `Delta D` are the only binding teacher-influence statistics.

## Changes Made

### 1. Strength-matched teacher-semantic-free proxy

- Fit raw low-level visual/text-content quantities with fold-bank-only robust signed scaling to `[-1,1]`.
- Reuse the exact EDCM acceptance mask and reliability.
- Select one **shared EN+ZH** proxy temperature by deterministic OOF bisection so pooled median proxy `R` matches pooled median EDCM `R`; no accuracy/F1, validation or per-video MLLM alignment enters calibration.
- Freeze/hash it before A2 and report TV/R/distance/active distributions per dataset.

### 2. Fully specified directional gate

- Freeze normalized descent directions, same 8+8 lists and temperature for `mu`, exact `Delta D`, epsilon and query bootstrap.
- Make all embeddings/banks/signatures/list IDs fold-local and freeze them before gradients.

### 3. Exact training-bank contract

- EDCM reuses the comparator's detached bank object and refresh schedule.
- All EDCM/ListNCA arms explicitly ID-exclude the query from their auxiliary 8+8 list; this does not alter base RGCL or final retrieval and is identical across controls.

## Revised Proposal

# Research Proposal: EDCM-RGCL — Dense Interventional Coalition Control of Retrieval Memory Geometry

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable part of hateful-video RGCL, and do not stop until one frozen method improves **final test accuracy and macro-F1 by at least +0.030 absolute each** over the moving strongest same-protocol non-MLLM RGCL comparator on **at least two datasets**, using paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied sparse neighbour events, absolute verdicts, static segment salience, extra embeddings, auxiliary semantic fields or a competing native head. They were sparse, redundant with the video label, absorbed by the fusion head, or merely redistributed accuracy between head and memory. SSR now adds decisive evidence: even its optimistic all-candidate OOF oracle touched only 2/7 EN and 3/15 ZH unique MI/SC error queries and could not reach its dual-metric gate. The successor must therefore provide a **reliable, dense, per-training-video causal signal** that directly changes the listwise gradient of the same full-video embedding geometry used by the final kNN memory.
- **Non-goals:** Localization-only, explanation-only, audit/guard-rail-only or native-head-only success; test-time MLLM annotation, judging, score fusion, reranking or veto; simple MLLM score/embedding/rationale concatenation; static segment weighting or segment-weighted memory; generated counterfactual content; a second parallel method stacked with SSR; gains primarily from a larger model, more data, more epochs/steps, ensembling, altered preprocessing, altered checkpoint selection, changed retrieval/voting, changed labels or any protocol relaxation.
- **Constraints:** The **only gold supervision is the video-level binary label**. No segment-level gold annotation exists or may be assumed. Every MLLM modality, coalition, necessity, sufficiency, preservation, stance, target, mechanism, rationale, localization or segment output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold, dense annotation or oracle evidence. Validation/test receive no such annotation or pseudo-signal; low-confidence, missing or invalid train pseudo-signals deterministically reduce to the exact non-MLLM path. Use the exact strongest per-dataset RGCL comparator, fixed splits/preprocessing/labels/epochs/checkpoint rule/retrieval/seeds. Required controls are remove-MLLM, within-split signature shuffle and calibrated noise/corruption, with coverage/confidence/fallback reporting. All later compute is SLURM-only in `HateVideo`, without `--time`, within 2 GPUs/16 CPUs/128 GB.
- **Success condition:** On at least two datasets and paired seeds 0/1/2, both final accuracy and macro-F1 gain at least +0.030 over `max(historical strongest point, paired baseline mean)`; every seed delta is positive; mean±std and hierarchical paired-bootstrap uncertainty are reported; the four dataset×metric primary tests pass Holm-corrected familywise α=0.05 with 95% lower bounds above zero. The full method must beat remove-MLLM and shuffled-MLLM controls with same-direction paired effects and 95% CIs excluding zero in both metrics, survive calibrated corruption, improve the **kNN readout itself** without head↔memory redistribution, and retain a defensible retrieval-specific novelty claim. Any missing metric/dataset/seed/statistical/mechanism/supervision/protocol item is `not_working`, not success.

## Technical Gap, Thesis and Focus

Absolute MLLM decisions, extra semantic inputs, segment weighting and sparse relation edges have not moved final memory enough. SSR's pre-extraction STOP proves that its selected edge universe was too sparse. EDCM instead produces one whole-video interventional coalition weak pseudo-signal for every reliable train example and uses it to change the entire normalized full-video memory-list gradient.

> **Thesis:** Relative label-blind MLLM judgments over deterministic `V/S/O` coalitions define a privileged conditional neighborhood measure whose teacher-specific gradient is internalized into the exact full-video geometry read by final kNN.

This is the sole contribution. It adds no backbone, head, router, memory key, segment model, generated content or inference component.

## System and Complexity

```text
train video -> fixed V frames + S dataset transcript + O title/fixed OCR
            -> {V,S,O,VS,VO,SO,VSO}
            -> frozen 7B MLLM, 2 prompts x 2 orders
            -> ordinal distributions -> s_i,rho_i (weak train-only pseudo-signal)

full-video z_i + current detached full-video bank {z_j,y_j}
            -> signature distance -> one listwise L_EDCM
            -> exact base L_RGCL and unchanged final full-video kNN geometry

validation/test -> ordinary full-video encoder + ordinary train-memory FAISS/vote;
                   no MLLM/OCR/signature/confidence artifact loaded
```

- **Frozen/reused:** exact strongest per-dataset RGCL config, encoders, frame sampler, projection/align fusion, bank/miner, optimizer, epochs, checkpoint, final retrieval/vote and seeds.
- **New trainable components:** zero.
- **New artifacts:** one hashed teacher cache, one scalar loss.
- **Excluded:** teacher keys, coalition student views, relation graph, adapters, extra heads, rationale/score concat, segment loss, SSR stacking.

## A0: Mandatory Pre-MLLM Reachability/Cost Screen

Five-fold train OOF only; video binary labels are the only gold. Each fold model encodes held-out queries and its own training-partition bank. Search top 64. A wrong query is structurally reachable if at most two opposite-label entries inside the exact comparator top-`k` can be replaced by same-label entries at ranks `k+1:64` to flip the exact vote.

Before any MLLM call, both EN and ZH must have: at least 80% of all OOF videos with four same/four opposite candidates; at least `ceil(.05N)` reachable unique errors; correcting only those errors gives at least +.050 OOF accuracy and +.050 macro-F1; all folds/lists/predictions/votes/hashes verify. This is a conservative fixed-geometry reachability and cost screen, not an upper bound on learned EDCM and not evidence of MLLM usefulness.

## Frozen Teacher Interface

Pre-extraction manifest hashes:

- local `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, greedy (`do_sample=false`, `temperature=0`, `top_p=1`, `max_new_tokens=384`);
- four uniform timestamp-ordered frames at frozen 336-pixel processing;
- `S`: dataset-provided transcript as raw input, with no temporal/segment gold;
- `O`: dataset title plus deterministic PP-OCRv4 Chinese detector/recognizer + `ch_ppocr_mobile_v2.0` angle classifier on the same frames; exact local checkpoints/versions/hash, empty remains empty;
- Unicode NFC, whitespace collapse, fixed channel tags, 1,024-codepoint head/tail cap per text channel;
- two exact prompt files, forward/reverse coalition order, strict schema/parser/canonicalization/fallback.

The prompts ask how faithfully each coalition preserves the full `VSO` interpretation needed to distinguish targeted harmful endorsement from quotation, condemnation, reportage, satire or unrelated/offensive context; they explicitly forbid a hateful/non-hateful label.

Four calls emit only seven records:

```text
coalition: V|S|O|VS|VO|SO|VSO
preservation: 0|1|2|3
confidence: 0|1|2|3
```

No rationale, target/stance/mechanism field, timestamp, span, segment or hate score exists. Keep each coalition's four-call ordinal distribution and expected rank `rbar(C)`. Accept only if every coalition has modal rank agreement ≥3/4 and modal confidence ≥2. `rho_i` is minimum agreement. Any failure makes the whole signature missing and `L_EDCM(i)=0`.

```text
nV=rbar(VSO)-rbar(SO)       hVS=rbar(VS)-max(rbar(V),rbar(S))
nS=rbar(VSO)-rbar(VO)       hVO=rbar(VO)-max(rbar(V),rbar(O))
nO=rbar(VSO)-rbar(VS)       hSO=rbar(SO)-max(rbar(S),rbar(O))
s_i=[nV,nS,nO,hVS,hVO,hSO]/3 in [-1,1]^6
```

Negative interference is retained. These are interventional weak pseudo-signals, never gold or causal annotations.

## EDCM Listwise Memory NCA

For reliable full-video query `z_i`, take top 8 same-label and top 8 opposite-label full-video keys from the current detached train bank, explicitly excluding ID `i` for this auxiliary list. All EDCM/ListNCA controls use identical lists.

```text
d_ij = mean(abs(s_i-s_j))
q±_ij = exp(-d_ij/tau_s) / sum_{k in L_i±} exp(-d_ik/tau_s)
tau_s = 1.0 shared/frozen
a_ij = cos(z_i,z_j)
A+_i = sum_{j in L_i+} q+_ij exp(a_ij/tau_rgcl)
A-_i = sum_{j in L_i-} q-_ij exp((a_ij+m_rgcl)/tau_rgcl)
L_EDCM(i) = -log[A+_i/(A+_i+A-_i)]
L_total = L_exact_RGCL + .2 mean_i rho_i L_EDCM(i)
```

`tau_rgcl`, `m_rgcl` and `.2` are one route-wide frozen registration, never dataset/test tuned. The auxiliary term consumes the comparator's existing detached bank object and exact refresh/reindex schedule. The base RGCL loss and final memory construction are unchanged. A compatible positive is attracted most; a compatible opposite-label confound is repelled most. This is one normalized list gradient, not isolated weighted edges.

## Fold-Local Teacher-Active Gate

For fold `f`, the exact comparator is trained only on `T\F`; the same fold model encodes both held-out query `F` and bank `T\F`. Bank IDs are disjoint from queries; no held-out query/self appears in the bank. Candidate keys/signatures come only from `T\F`. The fixed 8+8 list and signatures are frozen before gradients. No full-data-trained embedding, list or scaling statistic is substituted. Hash model/checkpoint, IDs, embeddings, lists and signature rows.

Uniform Label-only ListNCA uses `u±=1/8`. Define standard total variation and teacher-specific gradient:

```text
TV(q,u) = 0.5 sum_j |q_j-u_j|
TV_i = [TV(q_i+,u_i+) + TV(q_i-,u_i-)] / 2
Delta g_i = grad_z L_EDCM(i) - grad_z L_uniform(i)
R_i = ||Delta g_i|| / (||grad_z L_uniform(i)|| + eps), eps=1e-12
active(i) iff TV_i>=.10 and R_i>=.10
```

On the same frozen 8+8 list and `tau_rgcl`, define

```text
mu_i = logsumexp(a_same/tau_rgcl) - logsumexp(a_opp/tau_rgcl)
vE_i = -grad L_EDCM / (||grad L_EDCM||+eps)
vU_i = -grad L_uniform / (||grad L_uniform||+eps)
DE_i = grad(mu_i)^T vE_i
DU_i = grad(mu_i)^T vU_i
DeltaD_i = DE_i-DU_i
```

Use 10,000 query bootstrap replicates for mean `DeltaD` CI. Per dataset pass requires reliable signature coverage ≥85% with Wilson lower ≥.80; reliable 8+8 lists ≥80% all OOF videos; teacher-active ≥70% all OOF; active ≥60% A0-reachable errors; mean `DeltaD>0` with 95% lower bound >0 and individual `DeltaD>0` on ≥60% reachable errors. TV is descriptive support; `R` and `DeltaD` are binding mechanism statistics. No segment field or segment metric exists.

## One Strength-Matched Teacher-Free Proxy

Raw proxy quantities contain no semantic model output:

```text
xV = decoded-frame fraction * median adjacent-frame (1-SSIM)
xS = log1p(Unicode transcript characters)
xO = log1p(Unicode title+OCR characters)
raw = [xV,xS,xO,xV*xS,xV*xO,xS*xO]
```

Within each OOF fold, fit median/IQR on bank `T\F` only; transform query and bank by `clip((x-median)/(2*IQR+eps),-1,1)`. Reuse the exact EDCM accepted mask and `rho_i`. Use the identical 8+8 list, loss, lambda and workload.

The proxy kernel has one shared temperature `tau_proxy`. After teacher extraction but before any development training, select it once across pooled EN+ZH OOF queries by deterministic log-grid/bisection to minimize absolute log difference between proxy and EDCM **median `R`**. Use no accuracy, F1, `DeltaD`, validation result or per-video teacher alignment; ties select the larger/weaker temperature. Freeze/hash the single value for both datasets and every seed/control. Report per-dataset pairwise-distance, TV, `R` and active-fraction distributions for proxy and EDCM. This strength matching asks whether MLLM semantic alignment matters beyond equal perturbation magnitude.

## Exact Controls and Gates

- **Remove-MLLM:** `.2 -> 0`, exact RGCL.
- **Label-only:** uniform weights, identical list/loss/workload.
- **Strength-matched proxy:** exactly above; the only teacher-free proxy.
- **Shuffle:** derange complete `(s_i,rho_i)` within dataset × video label × confidence × raw availability, preserve degrees/coverage, no fixed points.
- **Noise:** corrupt complete ordinal-distribution records at empirical four-call disagreement rate within availability/confidence strata; empirical rate all seeds, 2× seed 0.

A1 stops unless all teacher-active gates pass. A2 seed 0 runs paired baseline/remove, Label-only, proxy, full, shuffle and noise from identical initialization/schedule. Full must beat baseline, Label-only, proxy and shuffle by ≥+.010 dev accuracy and macro-F1, improve actual kNN purity/wrong-neighbour rate and avoid head↔memory redistribution. Failure is final for the route, not tuning permission.

A3 alone runs EN/ZH seeds 0/1/2 and the immutable final target/statistics. Validation/test get no teacher artifact. Bank refresh, final memory, FAISS and voting remain exact comparator behavior.

## Diagnostics, Failure and No-Segment-Gold Contract

- A0 sparse -> no MLLM call. Reliable but inactive teacher -> A1 stop. Proxy matches full -> MLLM unnecessary. Shuffle/noise preserves gain -> no causal attribution. Head improves but kNN does not -> failure.
- Report coverage/confidence/fallback; TV/R/DeltaD; gradient cosine with base RGCL; proxy strength match; neighbour purity/wrong-neighbour rate; class recall; embedding variance; head-vs-kNN.
- Fixed frames are whole-video input samples, not annotated segments. Dataset transcripts contain no temporal gold. OCR is deterministic input extraction. The teacher schema has no timestamp/span/segment/localization field. The only gold is the video binary label.
- Reserve “causal role” for the controlled final effect of remove/shuffle/noise; call the teacher outputs privileged interventional weak pseudo-signals.

## Novelty and Claim-Driven Validation

CGO uses student perturbation/convergence to control harmful-video gradients; general work uses intervention consistency; RAMF/IARE use reasoning; RGCL uses retrieval hard pairs. EDCM's narrow claim is the composition of relative label-blind MLLM coalition judgments, broad measured teacher-specific query gradients, exact final-memory listwise geometry and complete teacher removal at inference.

1. **Dense/non-redundant support (A0/A1):** both datasets pass reachability, reliability, all-video `R` activity, reachable-error `DeltaD` and strength-matched proxy separation without segment annotation.
2. **Causal geometry (A2):** full beats remove, Label-only, proxy and shuffle by ≥1 point in both dev metrics; noise degrades; actual kNN topology improves.
3. **Actual endpoint (A3):** EN/ZH × seeds 0/1/2 pass +3 accuracy/+3 macro-F1, 3/3 signs, hierarchical paired bootstrap, Holm correction and significant remove/shuffle costs.

## Compute and Handoff

- A0 reuses OOF artifacts and precedes teacher cost.
- A1 ~4 deterministic 7B calls/train video (~4,500 EN+ZH), estimated 10–30 GPU-hours after smoke.
- A2/A3 estimated 30–60 GPU-hours with cache reuse.
- No new gold, segment annotation or human dense supervision. All compute is SLURM-only in `HateVideo`, no `--time`, within 2 GPUs/16 CPUs/128 GB.
- Highest risks: A0 locality is too narrow; teacher is reliable but gradient-inert; strength-matched proxy explains the effect; `DeltaD` fails; full cannot beat all controls.
