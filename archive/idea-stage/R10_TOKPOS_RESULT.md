# R10 result — spectral saturation diagnostic (Task A) + token-position readout pilot (Task B)

Frozen design: `idea-stage/R10_TOKPOS_FREEZE.md`, committed at `f182877` before any number existed.
Deviations: `idea-stage/R10_TOKPOS_DEVIATION_D1.md` (`dfeb01e`), `idea-stage/R10_TOKPOS_DEVIATION_D2.md`
(`2e38db9`) — both filed before any arm metric was computed.

**Cost: ¥0.00 (no API). ~65 minutes on a shared local RTX 5090. 285 head-training runs
(150 + 60 + 75), 0 failures. 1872 Qwen forwards for extraction, 1 undecodable video.
Zero test-label tuning: every epoch rule, arm definition, threshold and winner selection used
train or dev only.**

---

## Headline

- **Task B is a GO on both datasets it was run on.** Concatenating a readout over the
  transcript-content token positions with the deployed assistant-header readout beats the deployed
  readout by **+0.0076 test macro-F1 on MHC-ZH (30 seeds)** and **+0.0101 on HateMM (15 seeds)**,
  both with paired-bootstrap 95 % CIs excluding zero, both with P2 agreeing in sign, and both
  beating a matched-width random-projection control by more than they beat the narrow control.
- **The mechanism is the paper's, not a width artefact.** The transcript-position readout on its
  own is *much worse* than the deployed one (−0.019 on MHC-ZH, −0.105 on HateMM). Doubling the
  width with a random projection of the deployed vector does nothing (−0.0018 / −0.0036). Only the
  concatenation of the two genuine readouts helps.
- **It does not stack on the layer axis.** Adding the same token-position block on top of the
  L24⊕L28 configuration costs −0.0097 on MHC-ZH under P1. The token axis and the layer axis are
  reading the same limited pool of extra signal, not two independent ones.
- **Task A says all four datasets are far from label saturation** — but the statistic degenerates
  on this substrate and the reading carries almost no dataset-specific information (§1.3).

---

## 1. Task A — spectral saturation diagnostic (arXiv 2606.24903)

`idea-stage/r10_sat/sat.py`, raw `idea-stage/r10_sat/sat.json`, log
`logging/runs/r10_sat/run.log`. Train + dev_seen only; `test_seen` is never opened by the loader.

### 1.1 The four readings (PCA-50 arm at K = K_max, the paper's calibrated regime, τ = 0.02)

`concat` = `[img_feats ‖ text_feats]`, the pair the head consumes. 50 trials per cell.

| dataset | K_max (per class) | erank(Σ̂_W) | **S(K_max)** | verdict at τ = 0.02 |
|---|---|---|---|---|
| HateMM | 341 | 33.65 | **0.0987** | **CONTINUE** |
| MHC-EN | 193 | 38.50 | **0.1995** | **CONTINUE** |
| MHC-ZH | 208 | 35.48 | **0.1706** | **CONTINUE** |
| ImpliHateVid | 799 | 35.52 | **0.0445** | **CONTINUE** |

Per-stream (same arm): HateMM img 0.0889 / text 0.0873 · MHC-EN 0.1885 / 0.1806 ·
MHC-ZH 0.1614 / 0.1516 · ImpliHateVid 0.0424 / 0.0379. Every one of the twelve cells reads
CONTINUE, ImpliHateVid closest to the line at 2.2× the threshold.

Native-dimension arm (no PCA, ℓ2-normed features, the paper's "foundation model" regime, monitored
against its 0.3 → 0.05 band):

| dataset | concat (d=7168) | img (d=3584) | text (d=3584) |
|---|---|---|---|
| HateMM | erank 89.2, S 0.262 | 43.5, 0.128 | 98.1, 0.288 |
| MHC-EN | 97.2, 0.504 | 49.3, 0.256 | 102.0, 0.529 |
| MHC-ZH | 73.0, 0.351 | 34.1, 0.164 | 78.6, 0.378 |
| ImpliHateVid | 127.3, 0.159 | 56.3, 0.070 | 132.3, 0.166 |

### 1.2 What the numbers say, taken at face value

The paper's saturation scale is `K_sat ≈ erank(Σ_W)/τ`. With erank ≈ 34–39 in the calibrated
regime that is ≈ 1700–1900 labelled examples per class:

| dataset | K_max | K_sat | K_sat / K_max |
|---|---|---|---|
| HateMM | 341 | 1682 | 4.9× |
| MHC-EN | 193 | 1925 | 10.0× |
| MHC-ZH | 208 | 1774 | 8.5× |
| ImpliHateVid | 799 | 1776 | 2.2× |

Read literally: none of the four datasets is anywhere near the point where additional labels stop
helping; MHC-EN and MHC-ZH would need roughly an order of magnitude more labelled video per class.

### 1.3 Why that reading is weak here — the statistic degenerates on this substrate

This is the important part and it was written into the freeze as scope limit §1.4.3 before the
sweep ran.

The PCA-50 within-class effective rank does not vary meaningfully across our four datasets. The
K-sweep shows it peaking around K = 32 and then *declining* to a common value:

| dataset | erank at K = 2 / 8 / 32 / 128 / K_max |
|---|---|
| HateMM | 2.0 / 12.8 / 38.5 / 35.5 / 33.6 |
| MHC-EN | 2.0 / 13.1 / 41.3 / 39.3 / 38.5 |
| MHC-ZH | 2.0 / 12.7 / 38.4 / 36.5 / 35.5 |
| ImpliHateVid | 2.0 / 13.0 / 40.5 / 38.0 / 35.5 |

All four land at 33.6–38.5 out of a possible 50. The within-class covariance is close to isotropic
in the top-50 principal subspace regardless of dataset, encoder, language or class balance. Since
`S(K) = erank/K` and erank is effectively a constant here, **the cross-dataset ordering of S(K_max)
is just the ordering of 1/K_max** — it restates per-class sample size and adds nothing. The paper
itself reports that controlling for log K drops the correlation from ρ = 0.637 to ρ = 0.324; on our
substrate the shared-erank effect is stronger still, because every dataset sits at the same erank.

The native-dim arm is the more informative of the two, and it says something the PCA arm cannot:
**the text stream carries roughly twice the within-class effective rank of the img stream on every
dataset** (98 vs 43, 102 vs 49, 79 vs 34, 132 vs 56). The img stream is a mean over ~1000 prefix
positions and that averaging collapses its within-class variance; the text stream is a 3-token
readout and retains far more. This is a geometric fact about the two deployed readouts, and it
points in the same direction as Task B.

### 1.4 Standing of this diagnostic

It is a diagnostic, not a verdict, exactly as frozen. It did not gate Task B and could not have.
Three further limits, all declared in advance:

1. It was validated on image classification with unregularised logistic-regression probes. Our head
   is a 3-layer HateClipper-align MLP with dropout and a triplet+BCE hybrid objective.
2. τ = 0.02 was calibrated at d = 50; only the PCA-50 arm is on-protocol.
3. **It is about labels, not representations.** "More labels would help" is not "a better method
   would help", and round 9 banned new annotation. The honest use of this reading is as a price on
   annotation, not as evidence that a method exists.

---

## 2. Task B — token-position readout

### 2.0 What the deployed readout actually is

Established from the extractor source before any run (freeze §0), and re-confirmed by the
extraction's own span statistics:

| dataset | median sequence | video block ends | assistant header starts | **deployed readout width** | **transcript-content span** |
|---|---|---|---|---|---|
| MHC-ZH train | 1137 tokens | 1023 | 1134 | **3 tokens** | **124–128 tokens** |
| HateMM train | 1173 tokens | 939 | 1170 | **3 tokens** | **222–229 tokens** |

The deployed `text_feats` is the mean of the layer-28 hidden states over the trailing
`<|im_start|>assistant\n` header — three format tokens. The title and transcript occupy 124–229
positions in the middle of the same forward and are pooled by **neither** deployed stream (the img
stream comes from a different forward with a different prompt that contains no transcript).
This is precisely the readout geometry arXiv 2605.12726 criticises.

Prior project work on alternative spans (`refine-logs/MNTP_S1_RECORD.md`, arms S1 and S1b) was
confounded: it flipped the attention mask to bidirectional, which collapses the two streams onto
each other at cosine 0.76–0.93. **Under the deployed causal mask no alternative span had ever been
measured.** Under causal attention the streams stay separated — measured here:
cos(A0, TXT) = 0.452, cos(A0, img) = 0.307, cos(TXT, img) = 0.605 on MHC-ZH train.

### 2.1 Extraction

`idea-stage/r10_tokpos/extract_tokpos.py`, a thin fork of the frozen readout extractor: one
text-stream forward per video, all pooling spans read GPU-free from that single forward at layers
28 and 24; `img_feats` carried over verbatim from the banked caches and therefore identical across
every arm.

- MHC-ZH: 806 videos, 3 splits, 0 failures, 23 min.
- HateMM: 1066 videos, 3 splits, 1 undecodable video (zero-vector guard, same item the banked
  caches also fail on), 11 min.
- LoRA adapters recovered from B2 and **sha256-verified against the record**:
  MHC-ZH `35a510f4…dd8` and HateMM `6571d132…efa`, both exactly the hashes in
  `refine-logs/MNTP_S1_RECORD.md` §1.1.

**Belt (deviation D2): A0 is bit-identical to the frozen deployed `_pool_span(span="response")`
run on the same forward — 12/12, max abs diff exactly 0.0, at both layers.** The cross-hardware
cosine against the A100-extracted banked caches (mean 0.996, min 0.962 on MHC-ZH; mean 0.995,
min 0.916 on HateMM) is recorded as descriptive platform drift, not as a gate; D1 removed those
caches from the comparison table and every arm here comes from one pass on one machine.

### 2.2 Leg 1 — MHC-ZH, 5 arms × 30 seeds (500–529), P1

All arms share the same `img_feats`; only `text_feats` differ. `n(·)` = row L2-norm.

| arm | text_feats | dim | **P1 test macro-F1** | P2 |
|---|---|---|---|---|
| **A0** | `n(A0_28)` — deployed 3-token header readout | 3584 | **0.8075 ± 0.0119** | 0.8128 |
| **TXT** | `n(TXT_28)` — transcript-content positions | 3584 | 0.7885 ± 0.0190 | 0.7919 |
| **CAT** | `[n(A0_28) ‖ n(TXT_28)]` | 7168 | **0.8151 ± 0.0141** | 0.8160 |
| **RAND** | `[n(A0_28) ‖ n(A0_28·R)]` — width control | 7168 | 0.8057 ± 0.0132 | 0.8042 |
| *SEG* | `[n(A0_28) ‖ 4 segment means]` — exploratory | 17920 | 0.8135 ± 0.0091 | 0.8191 |

Frozen contrasts (`idea-stage/reaudit/analyze_grid.py`, bar 0.005, B = 20000, seed 20260817):

| contrast | P1 mean | P1 95 % CI | seeds > 0 | P2 mean |
|---|---|---|---|---|
| TXT − A0 | −0.0190 | [−0.0275, −0.0108] | 8/30 | −0.0210 |
| **CAT − A0** | **+0.0076** | **[+0.0008, +0.0140]** | 19/30 | +0.0031 |
| **CAT − RAND** | **+0.0094** | **[+0.0029, +0.0157]** | 23/30 | +0.0118 |
| RAND − A0 | −0.0018 | [−0.0089, +0.0054] | 14/30 | −0.0087 |
| *SEG − A0* | +0.0060 | [+0.0007, +0.0113] | 19/30 | +0.0063 |

**Verdict under the frozen rule (freeze §2.5): GO.** CAT clears both required clauses — mean
≥ +0.005 with CI excluding zero against A0, P2 agreeing in sign, *and* the matched-width control
clause (CAT − RAND ≥ +0.005, CI excluding zero). TXT alone fails, as its own arm.

Note: `analyze_grid.py` prints its own aggregate verdict "NOT REVIVED". That is the re-audit
script's built-in rule — a conjunction over *every* listed contrast, which here includes the
deliberately-negative TXT − A0. This pilot's rule is the disjunction-with-width-clause frozen in
§2.5. The per-contrast numbers above are the analyzer's; only the aggregation differs.

### 2.3 Leg 3 — HateMM confirmation, 5 arms × 15 seeds (500–514), P1

| arm | P1 test macro-F1 | P2 |
|---|---|---|
| **A0** | 0.8660 ± 0.0113 | 0.8495 |
| TXT | 0.7612 ± 0.0089 | 0.7633 |
| **CAT** | **0.8761 ± 0.0093** | 0.8597 |
| RAND | 0.8624 ± 0.0064 | 0.8482 |
| *SEG* | 0.8727 ± 0.0126 | 0.8585 |

| contrast | P1 mean | P1 95 % CI | seeds > 0 | P2 mean |
|---|---|---|---|---|
| **CAT − A0** | **+0.0101** | **[+0.0037, +0.0161]** | 13/15 | +0.0102 |
| **CAT − RAND** | **+0.0137** | **[+0.0079, +0.0191]** | 13/15 | +0.0115 |
| TXT − A0 | −0.1048 | [−0.1119, −0.0966] | 0/15 | −0.0862 |
| RAND − A0 | −0.0036 | [−0.0084, +0.0022] | 3/15 | −0.0013 |
| *SEG − A0* | +0.0068 | [−0.0021, +0.0136] | 12/15 | +0.0091 |

**Second dataset confirms, and by a larger margin than the first.** Same sign, same structure,
same controls.

### 2.4 Leg 2 — does it stack on L24⊕L28? **No.** MHC-ZH, 2 arms × 30 seeds

W = CAT, selected on **dev** macro-F1 (CAT 0.8499 vs TXT 0.7925), never on test. Both arms rebuilt
from this extraction pass per D1 clause 4, so the A100-extracted `R6RO-CAT` is not used as `C0`.

| arm | img | text | P1 | P2 |
|---|---|---|---|---|
| **C0** | `[n(img28)‖n(img24)]` | `[n(A0_28)‖n(A0_24)]` | **0.8130 ± 0.0204** | 0.8197 |
| **C1** | same | `C0 text ++ TXT at both layers` | **0.8033 ± 0.0179** | 0.8244 |

`C1 − C0`: P1 **−0.0097** [−0.0186, −0.0007], 8/30 seeds positive; P2 +0.0047 [−0.0021, +0.0126].
Fails the stacking bar, with the P1 CI on the wrong side of zero.

Reading: CAT alone (0.8151) already matches or slightly exceeds the two-layer configuration C0
(0.8130). Putting both axes together costs a point of macro-F1 under P1 while doubling the text
width to 14336 on 579 training items. **The layer axis and the token-position axis are not
independent sources of signal on this substrate** — whatever extra discriminative content the
transcript positions carry is largely the same content L24 was already contributing.

### 2.5 Mechanism

The pattern is consistent across both datasets and is the paper's own story:

1. **The transcript-position readout is a worse summary than the header readout.** −0.019 on
   MHC-ZH, −0.105 on HateMM as a standalone. Under causal attention the header tokens sit at the
   end and have attended over everything; the transcript tokens have only seen the video block and
   the transcript prefix. Replacing the readout is not the fix — exactly what 2605.12726 says about
   naive pooling, and consistent with this project's own S1/S1b result that no single alternative
   span beats the deployed one.
2. **It nonetheless carries information the header readout does not.** cos(A0, TXT) = 0.45: these
   are not near-copies. Concatenating them gains +0.008 to +0.010.
3. **The gain is not width.** A matched-width random projection of the deployed vector gains
   nothing (−0.002 / −0.004), and CAT beats RAND by more than it beats A0 on both datasets. The
   head is not simply benefiting from more parameters.
4. **The gain is small and not additive with the layer axis.** It is roughly the size of the
   L24⊕L28 effect and overlaps with it.

### 2.6 What this does and does not license

**Licensed:** on MHC-ZH and HateMM, pooling the transcript-content token positions of the same
frozen causal Qwen2.5-VL forward and concatenating that with the deployed assistant-header readout
beats the deployed readout by +0.0076 and +0.0101 test macro-F1, above a matched-width control,
under the project's standard P1 protocol with paired-bootstrap CIs.

**Not licensed:** (a) any claim about MHC-EN or ImpliHateVid — ImpliHateVid has no raw video left
and cannot be tested at all on this axis; (b) any claim that this composes with the strongest
current configuration — leg 2 says it does not; (c) any absolute-number comparison to the ledger
(0.8014 MHC-ZH, 0.8747 HateMM), because those were extracted on A100 and this table is a same-pass
5090 table (D1); the within-table contrasts are the result, not the levels; (d) any claim about
layers other than 28 and 24, or about the img stream, which was held constant throughout.

**Caveat worth stating plainly:** +0.008 on MHC-ZH is about 1.2 test items out of 149, and +0.010
on HateMM is about 2.2 items out of 215. This is a real, replicated, control-passing effect at the
scale the principal has ruled acceptable ("真实可叠加的涨点就值得尝试"), but leg 2 shows it is
**not** stackable on the best current configuration, so it is a small alternative to the layer
trick rather than an addition to it.

---

## 3. Artefacts

| what | where |
|---|---|
| freeze | `idea-stage/R10_TOKPOS_FREEZE.md` |
| deviations | `idea-stage/R10_TOKPOS_DEVIATION_D1.md`, `…_D2.md` |
| Task A code / raw | `idea-stage/r10_sat/sat.py`, `idea-stage/r10_sat/sat.json` |
| extractor fork | `idea-stage/r10_tokpos/extract_tokpos.py` |
| arm builders | `idea-stage/r10_tokpos/build_arms.py`, `build_leg2.py` |
| runners | `idea-stage/r10_tokpos/run_leg1.sh`, `run_leg2.sh`, `run_leg3.sh` |
| raw results | `idea-stage/r10_tokpos/leg1.json`, `leg2.json`, `leg3.json` |
| build metadata | `idea-stage/r10_tokpos/build_meta_MHC_zh.json`, `build_meta_HateMM.json`, `leg2_meta.json` |
| logs | `logging/runs/r10_sat/`, `r10_extract/`, `r10_extract_hm/`, `r10_leg1/`, `r10_leg2/`, `r10_leg3/` |
| new caches | `data/CLIP_Embedding/{MHC_zh,HateMM}/{train,dev_seen,test_seen}_*-tp.pt`, `*_R10TP-*.pt`, `*_R10L2-*.pt` |

Seeds 500–529 (MHC-ZH) and 500–514 (HateMM) are disjoint from every previously consumed range and
from the `REAUD_*` grid (300–329) that ran concurrently on the same machine.

## 4. Next step

The open question this hands back is why the token axis and the layer axis are not additive. Both
give ~+0.01 alone and less than either together. A cheap follow-up on already-extracted data —
the `-tp` caches hold all six spans at both layers for both datasets, so it costs head-training
time only — would be to check whether a fixed low-rank projection of the concatenated readout
(rather than raw concatenation into a 14336-wide linear layer) recovers the sum. That is a new
pilot with a new freeze, not an extension of this one.
