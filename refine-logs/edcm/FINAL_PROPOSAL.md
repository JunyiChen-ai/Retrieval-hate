# Research Proposal: EDCM-RGCL — Dense Interventional Coalition Control of Retrieval Memory Geometry

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable part of hateful-video RGCL, and do not stop until one frozen method improves **final test accuracy and macro-F1 by at least +0.030 absolute each** over the moving strongest same-protocol non-MLLM RGCL comparator on **at least two datasets**, using paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied sparse neighbour events, absolute verdicts, static segment salience, extra embeddings, auxiliary semantic fields or a competing native head. They were sparse, redundant with the video label, absorbed by the fusion head, or merely redistributed accuracy between head and memory. SSR now adds decisive evidence: even its optimistic all-candidate OOF oracle touched only 2/7 EN and 3/15 ZH unique MI/SC error queries and could not reach its dual-metric gate. The successor must therefore provide a **reliable, dense, per-training-video causal signal** that directly changes the listwise gradient of the same full-video embedding geometry used by the final kNN memory.
- **Non-goals:** Localization-only, explanation-only, audit/guard-rail-only or native-head-only success; test-time MLLM annotation, judging, score fusion, reranking or veto; simple MLLM score/embedding/rationale concatenation; static segment weighting or segment-weighted memory; generated counterfactual content; a second parallel method stacked with SSR; gains primarily from a larger model, more data, more epochs/steps, ensembling, altered preprocessing, altered checkpoint selection, changed retrieval/voting, changed labels or any protocol relaxation.
- **Constraints:** The **only gold supervision is the video-level binary label**. No segment-level gold annotation exists or may be assumed. Every MLLM modality, coalition, necessity, sufficiency, preservation, stance, target, mechanism, rationale, localization or segment output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold, dense annotation or oracle evidence. Validation/test receive no such annotation or pseudo-signal; low-confidence, missing or invalid train pseudo-signals deterministically reduce to the exact non-MLLM path. Use the exact strongest per-dataset RGCL comparator, fixed splits/preprocessing/labels/epochs/checkpoint rule/retrieval/seeds. Required controls are remove-MLLM, within-split signature shuffle and calibrated noise/corruption, with coverage/confidence/fallback reporting. All later compute is SLURM-only in `HateVideo`, without `--time`, within 2 GPUs/16 CPUs/128 GB.
- **Success condition:** On at least two datasets and paired seeds 0/1/2, both final accuracy and macro-F1 gain at least +0.030 over `max(historical strongest point, paired baseline mean)`; every seed delta is positive; mean±std and hierarchical paired-bootstrap uncertainty are reported; the four dataset×metric primary tests pass Holm-corrected familywise α=0.05 with 95% lower bounds above zero. The full method must beat remove-MLLM and shuffled-MLLM controls with same-direction paired effects and 95% CIs excluding zero in both metrics, survive calibrated corruption, improve the **kNN readout itself** without head↔memory redistribution, and retain a defensible retrieval-specific novelty claim. Any missing metric/dataset/seed/statistical/mechanism/supervision/protocol item is `not_working`, not success.

The anchor's phrase “causal signal” names the desired removable causal role. EDCM's teacher outputs are only **interventional weak pseudo-signals**, never causal ground truth.

## Technical Gap and Thesis

Prior MLLM routes did not move the final memory boundary: absolute verdicts were weak, semantic fields were video-label redundant, segment weighting was absorbed, and SSR's selected hard-event universe was too sparse even under an impossible all-accepted oracle. The missing intervention is a broad, per-video MLLM-specific gradient over the exact full-video memory list.

> **Thesis:** Relative label-blind MLLM judgments over deterministic visual/speech/on-screen-text coalitions define a privileged conditional-neighborhood measure whose broadly active teacher-specific gradient is internalized into the ordinary full-video kNN geometry and then discarded.

This is the sole contribution. It is not a feature, score, segment weight, selected key, generated view, relation graph or test-time reasoner.

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

- **Frozen/reused:** exact strongest per-dataset RGCL configuration, CLIP towers, frame sampler, projection/align fusion, bank/miner, optimizer, epochs, checkpoint selection, final retrieval/vote and seeds.
- **New trainable components:** zero.
- **New artifacts:** one hashed teacher cache and one scalar listwise loss.
- **Explicit exclusions:** teacher keys, student coalition branches, relation graphs, adapters, extra heads, rationale/score concat, segment objectives, test-time pseudo-signals and SSR stacking.

## A0: Mandatory Pre-MLLM Reachability/Cost Screen

Use five-fold train OOF geometry and video binary labels only. For fold `f`, train the exact comparator on `T\F`; the same fold model encodes held-out queries `F` and bank `T\F`. Search each query's top 64 fold-local keys. A wrong query is structurally reachable if replacing at most two opposite-label keys in the exact comparator top-`k` with same-label keys at ranks `k+1:64` flips the exact vote.

Before **any** MLLM call, both MHC-EN and MHC-ZH must satisfy:

1. at least 80% of all OOF train videos have at least four same-label and four opposite-label candidates in top 64;
2. at least `ceil(0.05*N)` unique errors are structurally reachable;
3. correcting all and only reachable errors yields at least +0.050 OOF accuracy and +0.050 OOF macro-F1;
4. fold disjointness, candidate lists, predictions, vote and metrics verify by hash.

A0 is a conservative fixed-geometry reachability and cost screen, not an upper bound on learned EDCM and not evidence of MLLM usefulness. Failure stops before teacher cost.

## Frozen Teacher Interface

Before extraction, hash a manifest containing:

- local `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, greedy decoding (`do_sample=false`, `temperature=0`, `top_p=1`, `max_new_tokens=384`);
- four uniform timestamp-ordered frames at frozen 336-pixel processing;
- `S`: dataset-provided transcript as a raw input modality, containing no temporal or segment gold;
- `O`: dataset title plus PP-OCRv4 Chinese detector/recognizer and `ch_ppocr_mobile_v2.0` angle classifier output from the same frames; exact local checkpoint/version hashes; empty remains empty;
- Unicode NFC, whitespace collapse, fixed channel tags and 1,024-codepoint head/tail cap per text channel;
- two exact prompt files, forward/reverse coalition order, strict JSON schema/parser/canonicalization/fallback.

Prompts ask only how faithfully each coalition preserves the full `VSO` interpretation needed to distinguish targeted harmful endorsement from quotation, condemnation, reportage, satire or unrelated/offensive context. They explicitly forbid a hateful/non-hateful label.

Four calls output only:

```text
coalition: V|S|O|VS|VO|SO|VSO
preservation: 0|1|2|3
confidence: 0|1|2|3
```

No rationale, target/stance/mechanism field, timestamp, span, segment or hate score exists. Retain each coalition's four-call ordinal distribution and expected rank `rbar(C)`. Accept a video only if every coalition has modal-rank agreement ≥3/4 and modal confidence ≥2. `rho_i` is the minimum agreement. Any failure makes the complete signature missing and sets `L_EDCM(i)=0`.

```text
nV=rbar(VSO)-rbar(SO)       hVS=rbar(VS)-max(rbar(V),rbar(S))
nS=rbar(VSO)-rbar(VO)       hVO=rbar(VO)-max(rbar(V),rbar(O))
nO=rbar(VSO)-rbar(VS)       hSO=rbar(SO)-max(rbar(S),rbar(O))
s_i=[nV,nS,nO,hVS,hVO,hSO]/3 in [-1,1]^6
```

Negative interference is retained. `s_i` is a privileged interventional weak pseudo-signal, not an annotation.

## EDCM Listwise Memory NCA

For reliable full-video query `z_i`, take the top eight same-label and top eight opposite-label full-video keys from the comparator's current detached train bank, explicitly excluding ID `i` from this auxiliary list. All EDCM/ListNCA controls share the exact list.

```text
d_ij = mean(abs(s_i-s_j))
q±_ij = exp(-d_ij/tau_s) / sum_{k in L_i±} exp(-d_ik/tau_s)
tau_s = 1.0 shared/frozen
a_ij = cos(z_i,z_j)
A+_i = sum_{j in L_i+} q+_ij exp(a_ij/tau_rgcl)
A-_i = sum_{j in L_i-} q-_ij exp((a_ij+m_rgcl)/tau_rgcl)
L_EDCM(i) = -log[A+_i/(A+_i+A-_i)]
L_total = L_exact_RGCL + 0.2 * mean_i rho_i L_EDCM(i)
```

`tau_rgcl`, `m_rgcl` and `0.2` are frozen once route-wide and never dataset/test tuned. The auxiliary term consumes the exact comparator detached-bank object and refresh/reindex schedule. Base RGCL and final memory construction do not change. A signature-compatible positive receives the strongest attraction; a signature-compatible opposite-label key is the hardest confound and receives the strongest repulsion.

## Fold-Local Teacher-Active Gate

For every fold, query and bank embeddings come from the same fold model; bank IDs are `T\F`, disjoint from query IDs. Bank and held-out query signature rows, candidate IDs, robust-scaling statistics, embeddings and lists are frozen and hashed before gradients. No full-data-trained embedding/statistic is substituted.

Uniform Label-only ListNCA uses `u±=1/8`. Define:

```text
TV(q,u) = 0.5 * sum_j |q_j-u_j|
TV_i = [TV(q_i+,u_i+) + TV(q_i-,u_i-)] / 2
Delta g_i = grad_z L_EDCM(i) - grad_z L_uniform(i)
R_i = ||Delta g_i|| / (||grad_z L_uniform(i)|| + eps)
eps = 1e-12
active(i) iff TV_i>=0.10 and R_i>=0.10
```

On the same frozen 8+8 list and `tau_rgcl`:

```text
mu_i = logsumexp(a_same/tau_rgcl) - logsumexp(a_opp/tau_rgcl)
vE_i = -grad L_EDCM / (||grad L_EDCM||+eps)
vU_i = -grad L_uniform / (||grad L_uniform||+eps)
DE_i = grad(mu_i)^T vE_i
DU_i = grad(mu_i)^T vU_i
DeltaD_i = DE_i-DU_i
```

Use 10,000 query-bootstrap replicates for the mean `DeltaD` CI. Per dataset, A1 requires:

1. reliable complete-signature coverage ≥85%, with Wilson 95% lower bound ≥0.80;
2. reliable 8+8 lists for ≥80% of all OOF videos;
3. teacher-active ≥70% of all OOF videos;
4. teacher-active ≥60% of A0-reachable errors;
5. mean `DeltaD>0`, bootstrap 95% lower bound >0, and individual `DeltaD>0` on ≥60% of reachable errors.

TV is support description; `R` and `DeltaD` are the binding MLLM-gradient statistics. Cache coverage alone cannot pass.

## Teacher-Semantic-Free Strength-Matched Proxy

The one low-level proxy contains no semantic model output:

```text
xV = decoded-frame fraction * median adjacent-frame (1-SSIM)
xS = log1p(Unicode transcript characters)
xO = log1p(Unicode title+OCR characters)
raw = [xV,xS,xO,xV*xS,xV*xO,xS*xO]
```

Within each OOF fold, fit median/IQR on bank `T\F` only; transform query/bank by `clip((x-median)/(2*IQR+eps),-1,1)`. Fewer than two decoded frames or undefined SSIM gives the visual-variation factor zero; zero IQR maps that centered dimension to zero. OCR/decode failure remains empty/missing. Reuse the exact EDCM accepted mask, `rho_i`, 8+8 list, loss, coefficient and workload.

The proxy uses one shared EN+ZH `tau_proxy`. The authoritative grid is `{2^(-8+0.25t): t=0,...,64}`. After extraction but before A2, choose the grid point minimizing absolute log difference between pooled OOF proxy and EDCM median `R`; ties choose the larger/weaker temperature. A match must be within 5% relative median `R`, otherwise A1 stops. No bisection, accuracy, F1, `DeltaD`, development result or per-video MLLM alignment selects it. Freeze one value for both datasets and all seeds. Report pairwise-distance, TV, `R` and active-fraction distributions per dataset.

This is intentionally teacher-**semantic**-free, not fully teacher-independent: it shares teacher acceptance/reliability and aggregate strength so that only sample-specific semantic alignment differs.

## Exact Controls and Staged Gates

- **Remove-MLLM:** coefficient `0.2 -> 0`, exact RGCL.
- **Label-only:** uniform weights, same list/loss/workload.
- **Teacher-semantic-free strength-matched proxy:** exactly above; the only low-level proxy.
- **Shuffle:** derange complete `(s_i,rho_i)` records within dataset × video label × confidence × raw availability; preserve degrees/coverage; no fixed points.
- **Calibrated noise:** corrupt complete ordinal-distribution records at the empirical four-call disagreement rate within availability/confidence strata; empirical rate on all final seeds and twice-rate at seed 0.

### A1 stop

Every reliability, teacher-active, directional and proxy-strength gate must pass on both datasets. Failure is route failure, not tuning permission.

### A2 seed-0 mechanism gate

From identical initialization/schedule, run remove, Label-only, proxy, full, shuffle, empirical-noise and 2×noise. Full must beat remove, Label-only, proxy and shuffle by at least +0.010 dev accuracy and macro-F1 and improve actual kNN purity/wrong-neighbour rate without head↔memory redistribution. For calibrated corruption, both dev metrics must satisfy `clean > empirical-noise > 2x-noise`, and empirical-noise must remain above remove in both metrics.

### A3 final target

Run MHC-EN/ZH seeds 0/1/2 only after A2. The immutable +3/+3/statistical/removability target remains binding. Empirical-noise must have positive per-seed deltas over remove in both metrics, retain at least 50% of the clean-minus-remove mean gain in both metrics per dataset, and remain below clean in mean performance; seed-0 2×noise must not exceed empirical-noise in either metric. These rules operationalize “survive calibrated corruption” before results.

Validation/test receive no teacher artifact. Final bank, FAISS, checkpoint selection and voting remain exact comparator behavior.

## Diagnostics, Failure and Supervision Contract

- A0 sparse -> no teacher call. Reliable but inactive teacher -> A1 stop. Proxy matches full -> MLLM unnecessary. Shuffle/noise preserves clean gain -> no causal attribution. Native head improves but kNN is flat/down -> failure.
- Report coverage/confidence/fallback; TV/R/DeltaD; proxy strength; gradient cosine with base RGCL; neighbour purity/wrong-neighbour rate; class recall; embedding variance; head-vs-kNN.
- The only gold is the binary video label. Uniform frames are whole-video input samples, not annotated segments. Dataset transcripts have no temporal gold. OCR is deterministic input extraction. The teacher schema contains no timestamp/span/segment/localization field. “Dense” means broad training influence, never dense human annotation.
- Reserve “causal role” for controlled final effects of remove/shuffle/noise.

## Novelty and Claim-Driven Validation

CGO controls harmful-video gradients using student perturbation/convergence; general modality-interference work uses intervention consistency; RAMF/IARE use reasoning; RGCL uses retrieval hard pairs. EDCM's narrow difference is the composition of relative label-blind MLLM coalition judgments, broad measured teacher-specific list gradients, exact final-memory geometry and complete teacher removal at inference.

1. **Dense and non-redundant support (A0/A1):** both datasets pass reachability, reliability, all-video `R` activity, reachable-error `DeltaD` and proxy strength/semantic separation without segment annotation.
2. **Causal memory repair (A2):** full beats remove, Label-only, proxy and shuffle by ≥1 point in both dev metrics; corruption degrades predictably; actual kNN topology improves.
3. **Actual endpoint (A3):** EN/ZH × seeds 0/1/2 pass +3 accuracy/+3 macro-F1, 3/3 positive signs, hierarchical paired bootstrap, Holm correction, significant remove/shuffle costs and calibrated-noise survival.

## Compute and Handoff

- A0 reuses OOF artifacts and precedes all MLLM cost.
- A1 requires about four deterministic 7B calls per train video (~4,500 EN+ZH), estimated 10–30 GPU-hours after smoke.
- A2/A3 are estimated at 30–60 GPU-hours with baseline/cache reuse.
- No new gold, segment annotation or human dense supervision is created.
- All compute is SLURM-only in `HateVideo`, no `--time`, within 2 GPUs/16 CPUs/128 GB.
- Highest risks are explicit: A0 locality too narrow; teacher reliable but gradient-inert; proxy explains the effect; `DeltaD` fails; full cannot beat all controls.
