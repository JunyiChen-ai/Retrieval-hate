# Round 3 Review

**Reviewer agent id:** `/root/sq_reviewer_replacement` (continuous from Round 1)

<details open>
<summary>Full raw reviewer response</summary>

# Round 3 Re-evaluation — SQ-RGCL

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.8 |
| Method Specificity | 8.6 |
| Contribution Quality | 8.0 |
| Frontier Leverage | 8.3 |
| Feasibility | 8.1 |
| Validation Focus | 8.2 |
| Venue Readiness | 7.5 |

**Weighted overall: 8.46/10**  
**Verdict: REVISE**

## Core findings

- **Problem Anchor:** PRESERVED verbatim. Method remains train-only, whole-video, final ordinary kNN facing.
- **Nuisance validity:** Substantially closed. Label-blind provenance, forbidden-semantic audit, environment×class mass/ESS and two-sided relation ESS are scientifically better than the deleted universal `q→y` ceiling.
- **Graph closure:** Operationally closed—both pair endpoints receive posteriors, comparison universes match, calls are explicitly bounded.
- **Contribution:** Now a precise, narrow method-level delta rather than merely a control-level distinction: crossed presentation relations and current-vote exposure jointly define one ranking constraint. It is no longer ordinary weighted SupCon, although it remains a composition of standard triplet/ranking primitives rather than a broad new quotient-learning theory.
- **Simplicity:** PASS. One posterior, one scalar loss, no new FULL head or inference path; the extensive controls do not make the method itself overbuilt.
- **No-segment-gold audit:** PASS.
- **CTE-interpretation audit:** PASS.

## Remaining scientific blockers

### 1. Delete the harmonic tail

For `rank>20`, the repository vote exposure is exactly zero. The harmonic continuation is therefore an invented optimization prior, not repository-vote exposure, and can make gains attributable to generic far-negative metric learning.

Use exact top-20 exposure only:

`E_i(j)=0` for `rank>20`.

This does not return to the frozen SSR/EDCM universe: the bank and shared encoder co-move at every refresh, outsiders can enter the top 20, and full-bank positive sampling already moves previously distant memories. No tunable replacement is needed.

### 2. SQ-1 has closure but insufficient inferential support

`16` anchors per class and `200` directed edges do not justify class-specific AUC, 10-point alignment and bootstrap-lower-bound claims. Edges sharing an anchor are correlated; the effective sample size is roughly the anchor count, not 200.

Replace the fixed minima with a preregistered **anchor-cluster power gate** derived before teacher calls. It must show adequate power for the registered `+0.02 AUC` and `+10pp` effects separately by class. If the `≤128`-vertex closure cannot supply that many independent anchors, SQ-1 must stop as infeasible rather than treating edge count as replication.

### 3. Greedy overlap creates selection bias, not label leakage

Using train labels solely to stratify class coverage is permitted and is not validation/test leakage; the teacher remains label-blind. However, greedily maximizing endpoint overlap preferentially selects high-degree/hub regions, so pilot enrichment may not generalize to the full train graph.

Freeze anchors by representative sampling within class×margin strata first, then close their endpoints. Report selected-versus-full differences in degree, margin and base cluster; any remaining unequal selection probability must be reflected in the pilot estimand or weighting. The pilot cannot claim dense population value from a hub-only subgraph.

### 4. Freeze the four-call aggregation

The proposal specifies two prompts × two orders but does not define the single `q^T,r` used by the loss. Before calls, freeze a deterministic aggregation, e.g. arithmetic-mean posterior and minimum reported confidence, with the existing agreement gate. This is small but scientifically necessary because it defines the teacher artifact.

## Documentation polish

- Define whether all six ontology entries are “active”; otherwise rare categories could be silently excluded from environment×class gates.
- Freeze blind-audit sample size, raters and contamination decision rule.
- Confirm that the quoted repository rank weights exactly match evaluator arithmetic.
- State that the `1024` cap counts model invocations, while `256` is the maximum number of unique videos across both datasets.

## Prior-art judgment

The novelty boundary is sufficiently precise for a method proposal: train-only MLLM presentation assignments define a coupled positive fiber crossing and negative class transversal inside a vote-exposed RGCL ranking loss. It must remain a narrow claim. ENV-SUPCON/Yang/P4 controls can establish mechanism attribution, but cannot by themselves prove bibliographic novelty.

Once the harmonic tail is removed and SQ-1 receives a power-valid, representative graph-closed design with frozen posterior aggregation, no major architectural blocker remains. It is not READY yet because those issues affect the actual scientific estimand, not merely wording.

</details>
