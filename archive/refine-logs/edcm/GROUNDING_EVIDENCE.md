# EDCM Gate-1 Grounding Evidence

**Read date:** 2026-07-10 (Pacific/Auckland)  
**Scope:** method refinement only; no source-code edit, no MLLM call, no SLURM submission.

## Authoritative target and negative evidence

- `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, `TARGET_REVIEW_RAW.md` freeze the +3 accuracy/+3 macro-F1, two-dataset, three-seed, statistics, removability and protocol contract.
- `TARGET_LOOP.md` makes the supervision boundary explicit: the video-level binary label is the only gold; no segment-level gold exists. Every MLLM-produced field is weak/privileged pseudo-signal.
- `research-wiki/CAMPAIGN_mllm_method_role.md` closes eleven main-table routes. The binding failures are: static evidence weighting is absorbed (P3), schema distillation is label-redundant (P4), score fusion is correlated and weaker (P7), generated views are unreliable (P5), and RGCL-on LMM training redistributes head/memory accuracy (P9/P9b).
- `refine-logs/FINAL_PROPOSAL.md` and `refine-logs/EXPERIMENT_PLAN.md` specify SSR's sparse directed-neighbour mechanism; they remain untouched.
- `artifacts/ssr/v1/B1_DECISION.json` and `artifacts/ssr/v1/b1/preflight_oracle_upper_bound.json` are decisive updated evidence. Under the optimistic assumption that every selected arc would be a reliable MLLM relation, the correctable universe was only 2 EN MI, 7 EN SC, 3 ZH MI and 15 ZH SC unique OOF errors. All four cells failed the frozen dual-metric gate; relation extraction could not change this bound. EDCM must therefore establish dense video-level correctability **before any MLLM call**.

## Gate-0 literature and novelty

- `research-wiki/TARGET_GATE0_LITERATURE.md` permits CCGC/EDCM only if relative deterministic modality interventions directly control RGCL listwise/memory geometry. Simple fusion weights, segment scores, embeddings and memory-entry weighting are automatic novelty failures.
- `research-wiki/TARGET_GATE0_NOVELTY_REVIEW.md` rates the route 5.5/10 before refinement. The closest threats are CGO (harmful-video gradient control), modality-interference intervention/consistency, BridgeVLM/CFPO causal internalization, and RAMF/IARE reasoning supervision. The narrow defensible delta is an MLLM relative coalition signature controlling the final kNN memory's listwise training geometry.
- The strongest initial EDCM variant in Gate 0 selected sufficient teacher keys, but risks full-query/clean-key mismatch and overlaps P3/P11 if implemented as view weighting. This refinement therefore retains full-video keys at both train and test and lets the signature change only a list-normalized memory objective.

## Code feasibility

- `src/model/classifier.py`: the current head maps frozen image/text features into the full-video embedding used by kNN. The strongest `align` fusion is multiplicative, so forcing missing-modality student views would create degenerate zero representations and require architecture changes.
- `src/model/loss.py`, `src/utils/retrieval.py`, `src/run_rac.py`: the baseline already rebuilds a full-train feature bank, mines same/opposite-label neighbours and returns the exact embedding used by retrieval evaluation. A list-normalized auxiliary objective can therefore reuse full-video query/key embeddings and introduce no new trainable module.
- Raw MHC/MHC-ZH annotations retain separate `Title` and `Transcript`; videos provide the visual channel. A fixed OCR pass may recover on-screen text from the same frames. These are teacher-only deterministic evidence packets, not new labels. If a channel is absent it is explicitly represented as absent, never hallucinated or treated as gold.

## Route comparison

| route | attraction | fatal risk | decision |
|---|---|---|---|
| Sufficient-view memory keys | Very direct memory-writing story | Full test query vs selected train key mismatch; close to P3/P11 or segment-weighted memory | reject |
| Student coalition-view consistency | Dense per video | Current `align` head cannot represent unimodal omissions without changing architecture; encourages module growth | reject |
| **Full-view coalition-signature listwise memory control** | Dense per train video; exact final embedding/memory; no new trainable component or test signal | Must prove signature adds more than label-only listwise weighting and is not CGO-style generic reweighting | **select** |

The selected route is the smallest buildable mechanism consistent with the anchor and updated SSR failure.
