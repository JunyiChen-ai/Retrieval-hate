# R10 novelty check — token-position readout on a frozen causal MLLM

Subject: the Task-B candidate frozen at `f182877` and resulted at `d811796`
(`idea-stage/R10_TOKPOS_FREEZE.md`, `idea-stage/R10_TOKPOS_RESULT.md`).

Date: 2026-08-17. Method: arXiv API (~35 queries, 2026-08 index), Semantic Scholar graph API
(citation walk on the nearest neighbour), OpenAlex full-text search, targeted PDF/HTML fetches, plus
one independent cross-check by GPT-5.6-Sol at xhigh reasoning (conversation only, no tool use).
WebSearch quota for the session was exhausted before this task, so Google Scholar was not queried
directly; OpenAlex and Semantic Scholar were used in its place.

Every arXiv id below was resolved against the live arXiv API in this session and the title/abstract
read. Ids attributed to the cross-check model and not independently resolved are marked *unverified*.

---

## 0. The candidate, restated for the check

Frozen Qwen2.5-VL-7B with a fixed LoRA adapter, one causal forward per video. The deployed text
readout is the mean of layer-28 hidden states over the trailing `<|im_start|>assistant\n` chat
template — 3 format tokens out of a ~1130-token sequence. The title/transcript content positions
(124–229 tokens, mid-sequence) are pooled by neither deployed stream. The candidate mean-pools those
content positions from the *same* forward and concatenates with the deployed readout.

Result: +0.0076 test macro-F1 on MHC-ZH (30 seeds) and +0.0101 on HateMM (15 seeds), paired-bootstrap
95 % CIs excluding zero, P2 agreeing in sign, above a matched-width random-projection control.
Standalone the content readout is much worse (−0.019 / −0.105); cos(header, content) = 0.45.
Secondary finding: the gain does **not** stack on the L24⊕L28 layer configuration (−0.0097).

---

## 1. Question 1 — token/span-position pooling for frozen LLM/MLLM classification

### 1.1 The stated nearest neighbour, and its follow-up field

`2605.12726` **"Before the Last Token: Diagnosing Final-Token Safety Probe Failures"** (Doda, 2026-05,
single author, ICML 2026 MI workshop). Confirmed via Semantic Scholar: **2 citations**, both of them
safety-probe works (`2607.14147` prefill-jailbreak mechanistics, `2608.08029` a reproducibility study
of latent-space safety probes). **Neither follow-up moves the finding into downstream supervised
classification, into multimodality, or into a concatenated-readout method.** The citation front on
this axis is empty in our direction.

Our differences from it are real and easy to state: it is an unsupervised/linear safety probe on
text-only LLMs, its remedy is a PCA-HMM trajectory model over the prefill, and its framing is a
*diagnosis of probe failure*, not a representation that improves a supervised head. We are a
supervised multimodal classifier, our remedy is a static concatenation, and we report a gain.

### 1.2 The genuinely dangerous neighbour, found in this sweep

`2604.18901` **"Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams"**
(2026-04, v2). Verbatim from the abstract:

> "two pooling choices applied to the same chat-templated activations at the same residual-stream
> layer (max-pool over content tokens versus last-token at the post-instruction position) recover
> harm directions 73° apart, and projecting one out leaves detection under either max-pool extraction
> essentially intact."

This is the same comparison we make — content-token pool vs. last-token-after-template pool, same
layer, same chat template, harm domain — and it already establishes that the two readouts are far
from copies (73°, i.e. cos ≈ 0.29; ours is cos 0.45). What it does **not** do: concatenate them,
report any downstream gain, touch a second modality, or look at the layer axis. Its framing is a
warning that probing recovers a protocol-specific direction. So the *observation* half of our
mechanism story is partly occupied; the *constructive* half is not.

Also in this family: `2605.02958` **"Tracing the Dynamics of Refusal"** (SALO, 2026-05) explicitly
opens on "static directions extracted from terminal or pooled representations … misses how refusal is
constructed across **layer-token positions**" and operates on raw hidden-state volumes over a layer
window. Joint layer×token, but white-box jailbreak detection, sparse localisation, no redundancy
analysis.

### 1.3 The crowding risk: the readout primitive itself is old

The cross-check model named the prior art here and every id resolved:

| id | work | what it occupies |
|---|---|---|
| `1905.08284` | R-BERT, CIKM 2019 | **the exact concatenation topology**: [CLS] summary vector ‖ mean-pooled mid-sequence entity spans → classifier |
| `2304.07193` | DINOv2, TMLR 2024 | frozen-feature linear eval that concatenates class-token features from several late layers **with the mean-pooled patch tokens** — summary-token axis + content-token axis + layer axis, in one recipe |
| `2202.08904` | SGPT, 2022 | position-weighted mean pooling, justified by exactly our causal-attention argument (later positions have seen more) |
| `2402.15449` | Echo Embeddings, 2024 | repairs the causal-attention information imbalance by repeating the input and pooling the second copy |
| `2404.05961` | LLM2Vec, COLM 2024 | the bidirectional-mask alternative (which this project already tried and which confounded MNTP S1/S1b) |
| `2405.17428` | NV-Embed, 2024 | replaces fixed pooling with learned latent-attention pooling |
| `2002.06652` | SBERT-WK, TASLP 2020 | **layer-axis fusion combined with token weighting**, in one sentence-embedding method |
| `2409.02727` | "Pooling And Attention: What Are Effective Designs For LLM-Based Embedding Models?" | proposes **Multi-Layers Trainable Pooling** — cross-attention over *all* hidden layers' token outputs. Reports that trainable pooling and bidirectional attention **do not significantly beat EOS-last-token pooling on clustering and classification** |
| `2603.03389` | GLOT, 2026-03 | pooling over a frozen LLM's token outputs as relational learning on a token graph; opens on mean/max pooling causing "signal dilution"; GLUE + MTEB |

Sentence-Transformers has shipped concatenable CLS/mean/max/weighted-mean/last-token pooling modes
for years, so "last-token pool concatenated with mean pool" is not even a new software primitive.

**Reading:** at the level of "concatenate two pooled spans of a frozen transformer", the slot is
occupied several times over. What is not occupied is the specific claim on the *causal chat-template*
readout: that the deployed 3-token template readout is a strictly better standalone summary yet still
misses content-position information, in a decoder-only MLLM, with a measured supervised gain.
No paper found makes that claim.

---

## 2. Question 2 — MLLM hidden-state features × hateful/harmful video (6-month recheck)

**Still zero.** Rechecked over the 2026-02 → 2026-08 window.

Every hateful-video paper indexed on arXiv from 2026 is output-level or fusion-level, none reads
internal hidden states:

- `2606.11953` "Decoding Multimodal Cues: Unveiling the Implicit Meaning Behind Hateful Videos" —
  new Ex-HateMM / Ex-ImpliHateVid rationale datasets, multimodal CoT + DPO. Generation-level.
- `2602.09637` LELA — training-free hate **localisation**, LLM over five caption-derived modalities.
- `2602.00132` SCANNER — test-time adaptation for hate video, centroid alignment on output space.
- `2601.15115` MARS — training-free multi-stage adversarial reasoning. Prompting only.
- `2512.02743` RAMF — reasoning-aware multimodal fusion, VLM-generated text as extra views.

Queries `abs:"hidden states" AND abs:"hateful" AND abs:"video"`, `abs:"hidden state" AND
abs:"features" AND abs:"hate"`, `abs:"video" AND abs:"MLLM" AND abs:"hidden states" AND
abs:"classifier"` all return **0 hits**. OpenAlex full-text search over 2025-01→now surfaced only
survey/meme/text work plus the already-known RAMF/retrieval-experts line.

Nearest occupants stay where they were:
- `2507.17394` HiProbe-VAD (ACM MM 2025) — frozen MLLM intermediate hidden states, but video **anomaly**
  detection, tuning-free, dynamic layer saliency.
- `2512.17601` HeadHunt-VAD (2025-12) — attention-head selection in an MLLM, again VAD.
- `2604.18519` SIREN (ACL 2026) — internal-representation harmfulness detection, **text only**.
- `2606.22864` (2026-06) — hidden-state probes on **Qwen2.5-VL-7B** for indirect prompt injection in
  computer-use agents. Same backbone, same "linear head on frozen internals" recipe, different harm
  and no video. This is the closest thing to a same-substrate occupant and it is an evaluation-protocol
  paper, not a method.

**Grade for this axis alone: (a) blank.** But it is an *application* blank. Moving a known probing
recipe to a new dataset family is not, by itself, a method contribution, and the project's standing
constraint is method-paper-only.

---

## 3. Question 3 — layer axis vs token axis redundancy

No paper found whose result is that multi-layer concatenation and multi-token-position concatenation
of frozen-LM features carry overlapping signal so their gains do not add.

Works that *use* both axes without asking the question:
- `2002.06652` SBERT-WK — inter-layer geometry drives token weights.
- `2304.07193` DINOv2 — last-4-layer class tokens ‖ mean patch tokens in linear eval.
- `2409.02727` — cross-attention over all layers' token outputs (and reports the negative result that
  it does not beat EOS-last-token on classification).
- `2601.09322` ALF (2026-01) — attentive fusion over all ViT layers; attentive probing pools over
  patch tokens by construction, so the two axes are jointly present.
- `2605.10494` (2026-05, ICASSP-track bioacoustics) — the closest structural parallel: it crosses
  *last- vs multi-layer* probing with *linear vs attention (time-aware)* probing, and reports both
  help ("gains … around 0.08 accuracy for BEANS classification" from multi-layer; attention probes
  beat linear on all transformer models). Read as complementary, not redundant. **No factorial
  additivity analysis.**
- `2603.05280` "Layer by layer, module by module" (2026-03) — a genuine two-axis probing study, but
  the second axis is the *module* (MHSA output vs FFN activation), not token position.

**So the question is open.** It is also, however, the axis on which our own evidence is weakest, and
the cross-check was blunt about why:

1. cos = 0.45 between two readouts says little about *label-relevant conditional* information in an
   anisotropic residual space.
2. C1 − C0 = −0.0097 with a 14336-wide text block on 579 training rows is as consistent with
   multicollinearity / capacity / sample-size limits as with "same signal pool".
3. The claim as written ("the two axes read the same limited pool") is stronger than the measurement
   licenses. Supporting it would need cross-fitted residualisation or conditional probes, centred
   CKA on residualised features, a full layer × span grid, and parameter-matched fusion heads.

A separate control caveat surfaced by the cross-check and worth recording: the RAND arm is
`[n(h) ‖ n(hR)]` with a fixed Gaussian `R`. Since `‖hR‖` concentrates tightly around `‖h‖`, the first
linear layer of the head sees `W₁n(h) + W₂n(hR)`, which is close to a reparametrisation of a linear
map of `n(h)`. RAND therefore controls **parameter count / width**, which is what the freeze claimed,
but it is *not* a control for "any second view of the same forward helps". Stronger controls would be
a shuffled-transcript span and a matched-length random contiguous span. This does not invalidate the
GO (the frozen rule asked only for the width control) but it caps how far the mechanism claim can be
pushed in a paper.

---

## 4. Question 4 — placeholder risk in the SIREN-family layer-fusion wave

The seven parallel works recorded in `IDEA_REPORT.md` §10.8 were re-read. Do any of them sweep the
token dimension?

| work | layer axis | token/position axis | pre-empts us? |
|---|---|---|---|
| `2604.18519` SIREN (ACL 2026) | yes, adaptive layer weighting over safety neurons | streaming per-token scoring, but no span-pooling comparison | no |
| `2512.21863` Frozen LVLMs for micro-video rec | yes, DFF multi-layer fusion | not examined | no |
| `2502.02013` Layer by Layer (ICML 2025) | yes | no | no |
| `2412.09563` Does Representation Matter? | yes | no | no |
| `2507.17394` HiProbe-VAD (ACM MM 2025) | yes, dynamic layer saliency | no | no |
| `2601.09322` ALF (2026-01) | yes, attention over all layers | **implicitly yes** — attentive probing pools patch tokens | partial (bidirectional ViT, no template/last-token issue, no additivity result) |
| `2605.10494` multi-layer attentive probing, bioacoustics (2026-05) | yes | **yes, explicit** — "larger probe heads that leverage time information have superior performance" | **partial and the closest** — crosses both axes, but audio, attention probes, no redundancy finding |

Add `2409.02727` (2024) as the pre-existing joint-axis occupant in the LLM-embedding literature.

**Verdict:** no exact pre-emption of the layer×token *additivity* result, but two 2026 works already
publish in the layer×token cross-product, and the direction of travel is toward learned joint
readouts. A fixed hand-picked span-pair concatenation reads as a simplified ablation of that family —
which is precisely the sentence §10.8 wrote to kill the layer axis.

---

## 5. Rating

**Overall: (c) — crowded.**

Per axis:

| axis | grade | reason |
|---|---|---|
| the readout method itself (concat two pooled spans of a frozen transformer) | **(c)/(d)** | R-BERT owns the topology, DINOv2 owns the frozen two-axis recipe, SGPT/Echo own the causal-attention rationale, Sentence-Transformers ships it as a config flag |
| the specific claim on a causal **chat-template** readout in an MLLM | **(b)** | `2605.12726` and `2604.18901` are close but neither concatenates nor reports a supervised gain; the citation front is empty |
| hateful-video application | **(a)** | genuinely zero papers, rechecked to 2026-08 — but this is application novelty, not method novelty |
| layer × token non-additivity | **(a) as a question, (c) as evidence** | unoccupied, and the most interesting thing R10 found — but the current measurement does not license the claim, and the paper it would produce is an analysis paper, which the project's method-paper-only constraint bans |

Consistency check with the project's own precedent: §10.8 of `IDEA_REPORT.md` graded the **layer**
axis **(d) with (c) fallback** and banked it as "a better feature default and an ablation row, not a
direction". The token axis is one notch better — its nearest neighbour is a single-author workshop
paper with two citations, and the hateful-video domain slot is empty where the layer axis had SIREN
sitting in it. But the two axes are, by R10's own leg-2 measurement, **not additive**, so the token
axis is a substitute for a mechanism already graded (d), not an addition to it. It cannot inherit a
better grade than a mechanism it replaces at equal effect size.

### Can this be a method paper?

**No, not as it stands.** Three independent reasons, any one of which is sufficient:

1. **The gain does not stack.** +0.008/+0.010 is roughly the size of the L24⊕L28 effect and overlaps
   with it (C1−C0 = −0.0097). The project cannot claim both, so the candidate buys no net movement
   over what is already banked.
2. **Effect size vs. the method-paper bar.** +0.0076 on MHC-ZH is ≈ 1.2 of 149 test items; +0.0101 on
   HateMM is ≈ 2.2 of 215. Real and replicated, and acceptable under the standing "incremental gains
   are acceptable" ruling — but a method paper whose method is "also pool the middle of the sequence"
   at this magnitude will be read as a pooling ablation.
3. **The method primitive is occupied.** The reviewer's one-line objection ("this is R-BERT's readout,
   or DINOv2's linear-eval recipe, applied to a video MLLM") has no answer that survives the fact that
   the transcript readout alone is worse and the combination gains 1 point.

The only framing with a path to (b)-grade contribution is the **conditional-redundancy** one: a
causal, same-forward readout method that *selects* which (layer, span) pairs carry conditionally
non-redundant label information rather than concatenating everything, validated by the factorial
layer × span grid the `-tp` caches already support. That is a new pilot with a new freeze, it needs
the residualisation/CKA machinery listed in §3, and its headline would be an analysis result with a
small method attached — i.e. it inherits the same tension with the method-paper-only constraint that
killed the layer axis.

### Disposition recommended

Same as the layer axis: **bank it as a feature default and an ablation row, do not open it as a
direction.** Record the RAND-control caveat (§3) and the over-strong "same signal pool" wording in
`R10_TOKPOS_RESULT.md` §2.4/§2.5 as claims to soften if either is ever written up.

---

## 6. Cross-check record

GPT-5.6-Sol (xhigh, conversation only, no shell) was given the candidate description and the four
questions independently. Its verdict: **(b) overall**, with the same per-axis split — application
blank on hateful video, redundancy question open, readout primitive crowded. It named R-BERT,
DINOv2, SGPT, Echo, LLM2Vec, NV-Embed, SBERT-WK, ELMo, Tuned Lens and DoLa as prior art (all
resolved and confirmed here except ELMo/Tuned Lens/DoLa, which were not chased because they read a
single position and only sweep layers), and named `2605.12726` as the single most dangerous prior,
agreeing with the freeze's own reading. It flagged two things this sweep had not: the RAND-control
weakness (§3) and the "frozen Qwen + LoRA" wording ambiguity — the LoRA adapter is fixed, not
jointly trained, and any write-up must say "frozen base and fixed LoRA adapter".

The one place this check departs from the cross-check is the overall grade: it said (b), this
document says (c). The reason is project-specific and not visible to an outside model — the layer
axis was already graded (d)/(c) here on the same substrate for the same reason, and leg 2 shows the
token axis is a substitute for it rather than an addition, so grading the substitute higher than the
thing it substitutes for would be inconsistent.

## 7. Sources

Resolved in this session against the live arXiv API: 2605.12726, 2604.18901, 2605.02958, 2606.24903,
2604.18519, 2512.21863, 2507.17394, 2601.09322, 2605.10494, 2502.02013, 2412.09563, 2603.05280,
2603.03389, 2606.11953, 2602.09637, 2602.00132, 2601.15115, 2512.02743, 2606.22864, 2512.17601,
2409.02727, 2002.06652, 2202.08904, 2402.15449, 2404.05961, 2405.17428, 1905.08284, 2304.07193.
Citation walk on 2605.12726 via Semantic Scholar graph API (2 citing works: 2607.14147, 2608.08029).
