# R9-1 ANCHOR-INT — frozen design, arms and decision rule

**Frozen 2026-08-18, before any pilot code was written and before any arm metric exists.**
Committed as its own commit; nothing below is edited after the first candidate number is produced.

## 1. Question

Round-9 diagnostic D1 (`idea-stage/R9_CANDIDATES.md` §0) measured that adapting the encoder
(LoRA on the dataset's own train split) both **repairs** 9-12 test items and **breaks** 4-10 items
that the frozen encoder answered correctly, on every dataset where both encoders exist. The
adaptation branch is the only axis in this project where an intervention moves ≥9 test items.

R9-1 asks the cheapest falsifiable question about that branch:

> Is there a point **between** the frozen and the adapted representation that keeps the repairs
> without the breaks?

A pass motivates the expensive weight-space version (re-extraction under `W_α = W_0 + α·s·B·A`,
plus anchored re-training). A failure closes the adaptation branch at zero GPU-hours beyond head
training, and round 9 ends with a documented closure.

## 2. Arms

Both pooled streams of the same model family, caches that already exist:

- `z0` = `Qwen2.5-VL-7B-Instruct_HF` (frozen)
- `z1` = `Qwen2.5-VL-7B-Instruct-LoRA_HF` (adapted, per-dataset LoRA)
- arm(α): `z_α = L2normalize((1−α)·z0 + α·z1)`, applied **independently and identically** to the
  `img_feats` and the `text_feats` stream. Both caches are already L2-normed per vector at
  extraction; re-normalising after the mix is declared here as part of the arm definition.

α grid, frozen: **{0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}** (7 values). A single global α per dataset.
No per-item, per-stream, per-class or per-dimension α — those would be per-item selection
(Law III / F47) or a learned mixer (closed in R8 D1).

## 3. Datasets, seeds, protocol

| dataset | seeds | n seeds |
|---|---|---|
| HateMM | 400-414 | 15 |
| MHC-EN | 400-429 | 30 |
| MHC-ZH | 400-429 | 30 |

ImpliHateVid is **excluded and declared**: it has no LoRA-adapted cache and its raw video is not
on this machine, so the arm cannot be constructed.

Seeds 400-429 are disjoint from every previously consumed range (0-29 protocol audit, 30-89 R6-1C,
100-129 R7, 200-229 R8, 300-314 R9 diagnostics).

Protocol is `idea-stage/r4_harness.py` unchanged: train on train, select the epoch on **validation
macro-F1**, report test. Every arm is trained from the same seed list so all comparisons are
seed-paired.

## 4. Comparator

The strongest endpoint **measured inside this harness and this seed set**:
`ENDPOINT = argmax over {α=0, α=1} of mean test macro-F1`. Per dataset, chosen from the arms of
this run (not from the historical contrast-line table, which uses a different protocol).

## 5. α selection

α* = the α maximising **mean validation macro-F1 across the dataset's seeds**. Validation only.
Test labels are never consulted for selection. If two α tie on val, the smaller α wins.

## 6. Decision rule (frozen)

**PASS** requires all four clauses on **≥ 2 of the 3 datasets**:

1. **Interior**: α* ∉ {0.0, 1.0}.
2. **Size**: Δ = mean test macro-F1(α*) − mean test macro-F1(ENDPOINT) ≥ **+0.005**.
3. **Certainty**: the 95 % paired bootstrap CI of Δ excludes 0. Bootstrap resamples **test items**
   (2000 draws), and for each draw averages the per-seed paired difference — so both test-item and
   seed variance enter.
4. **Mechanism**: relative to the *adapted* endpoint (α=1), arm α* reduces the break rate
   P(arm wrong | frozen correct) by ≥ 25 % **and** retains ≥ 80 % of the repair rate
   P(arm correct | frozen wrong), both computed on per-item across-seed majority errors.

Anything else is **KILL**: the adaptation branch is closed for round 9 and no GPU is spent on the
weight-space version.

Clause 4 is adopted verbatim from the external reviewer (gpt-5.6-sol, xhigh, 2026-08-18). Its
purpose is stated by that reviewer: passing clauses 1-3 alone could merely show that regularisation
makes the model more frozen.

## 7. Declared limitations, written before the run

1. **Feature-space interpolation is not weight-space interpolation.**
   `f_{(1−α)θ0+αθ1}(x) ≠ (1−α)f_{θ0}(x) + α f_{θ1}(x)` for a transformer. This arm is a *screen*:
   it is informative when it fails (no interior point helps even in the easier, more expressive
   feature-space version) and only suggestive when it passes.
2. **At deployment the arm needs two encoder passes**, so even a pass is not a deployable method —
   it is a premise gate for the weight-space version, which needs one.
3. The mechanism, if it passed, is occupied: WiSE-FT (CVPR 2022), model soups (ICML 2022),
   L2-SP (ICML 2018), LwF (ECCV 2016), DELTA (ICLR 2019), LP-FT (ICLR 2022), PromptSRC (ICCV 2023),
   "LoRA Learns Less and Forgets Less" (TMLR 2024), Model Tailor (CVPR 2025). A pass buys a
   direction to investigate, not a contribution.
4. **Test reuse**: this is the ~90th candidate evaluated against these official test splits.
   Paired bootstrap CIs do not correct for adaptive reuse of a fixed evaluation set. Any pass must
   be treated as a hypothesis for a confirmatory design, not as a result.

## 8. Cost and submission

Head training only, on caches that exist: 7 α × (15+30+30) seeds = 525 runs, ~11 s each ≈ 1.6 h
wall on the local RTX 5090. **¥0 API.** The analyzer is run exactly **once**, on the complete grid.
