# Scale-aware cross-modal attention: independent rule-4 proposal review

Date: 2026-10-31. Reviewer: independent agent (Claude Fable 5.1), the one rule-4 review of
`experiments/20261031_scale_aware_cma/README.md`. Read: that README, `RESEARCH_ITERATION_RULES.md` rule 4,
`src/qtl/model.py`, `src/qtl/data.py`, `docs/20260903_three_module_program.md`, the backbone-gap JSON the
proposal cites, and the papers and code listed in section 3. Rule 4 allows a STOP in four cases only; it forbids
stopping on pre-implementation reasoning (degeneration, identifiability, shortcut). This review follows that.

**Verdict: GO.** The proposed mechanism (a learned additive bias on the cross-modal attention logits indexed by a
log2 bucket of the temporal distance |t - t'|, shared by both attention directions, zero-initialised) was not found
in any hateful video detection or localisation paper or released code I checked. The other three STOP conditions do
not apply. GO means implement and code-review; it does not assert a performance gain or a first-ever claim.

## 1. Object under review

One change to r5's content prior network (`PriorNet.cma`, MACIL-SD's cross-modal layer): attention score
`q_t . k_t' / sqrt(d)` becomes `q_t . k_t' / sqrt(d) + b[h, k(|t - t'|)]`, with `k(0) = 0`,
`k(d) = min(1 + floor(log2 d), 10)`, 4 heads x 11 buckets = 44 learned scalars, init 0. Everything else
(inputs, answers, answer model, chain, node prior, losses, EIG policy, evaluation, search space) is r5 unchanged.
Ablation: bias frozen at 0, which is r5 exactly.

## 2. Code check: is r5's cross-modal attention order-blind in time?

Yes. Checked in `src/qtl/model.py` and `src/qtl/data.py`:

- `MultiHeadAttention.forward`: `scores = q k^T / sqrt(d_k)`, then `masked_fill` with the key padding mask,
  softmax, dropout, `p v`. No positional term of any kind.
- `CrossModalLayer`: pre-norm, one `attn` module used for both directions (`one(v, a)` and `one(a, v)`), FFN. No
  positional embedding. The proposal's statement that the bias table being shared across directions matches the
  existing sharing of the attention module is correct.
- `PriorNet.forward`: `fc_v(f_v)` and `fc_a(f_a)` go straight into `cma`. `f_v` is the I3D crop on the 1 s grid;
  `f_a` is the aligned audio row plus one 768-d text row per text source (`data.py` `Store.__init__`, lines 53-76).
  Revision 4's position channel is not present in r5's input.

So a permutation of the second indices permutes the per-second outputs identically; temporal order enters only
downstream (chain prior, and the node prior's interval pooling along the tree). The proposal's sentence "the only
temporal structure is added afterwards by the chain prior" should also name the node prior, which pools over tree
intervals; neither is inside the encoder, so the point stands.

Original MACIL-SD (github.com/JustinYuu/MACIL_SD, `Transformer.py`): same scores line
`torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)`, no positional encoding. Its `attention()` has a
`masksize` argument that, when not 1, zeroes attention outside a hard window of `masksize/2` seconds. The repo's
own port (`scripts/reproduction_baselines/macilsd/Transformer.py`, patch M1 comment) records that upstream never
sets it to anything but the default 1. So "no positional term" is correct for MACIL-SD as published and as ported;
the proposal should say that MACIL-SD's code contains an unused hard local-window option, which is a different
mechanism (hard cut-off, one scale, not learned) from the proposed learned multi-scale soft bias.

## 3. Literature check

### 3.1 Searches run (WebSearch, 2026-10-31)

1. `"hateful video" "relative position" attention localization`
2. `HateMM multimodal hateful video detection cross-modal attention positional encoding`
3. `MultiHateLoc hateful video temporal localization architecture transformer positional encoding`
4. `HateClipSeg VERA hateful video segment localization method cross-modal attention`
5. `"hateful video" "relative positional" OR "relative position bias" OR "distance-aware attention"`
6. `MACIL-SD modality-aware contrastive instance learning self-distillation cross-modal attention positional encoding github`
7. `audio-visual event localization cross-modal attention relative position bias temporal distance`
8. `RAMF OR HVGuard OR MoRE OR CLARA OR MultiHateGNN OR ImpliHateVid hateful video detection temporal position attention`
9. `HVGuard hateful video detection multimodal large language model paper`
10. `MultiHateGNN graph neural network hateful video detection paper arxiv`
11. `MoRE hateful video detection mixture of experts multimodal arxiv`
12. `VERA hateful video localization OR "hateful video" "VERA" arxiv 2025`
13. `weakly supervised video anomaly detection audio-visual cross-modal attention "relative position" bias temporal transformer`
14. `ImpliHateVid implicit hateful video dataset method cross-modal attention architecture ACL 2025`
15. `CMFusion contrastive multimodal fusion hateful video detection Zhang 2024 architecture`
16. `"hate" video detection transformer "relative position encoding" OR "relative positional encoding" OR "learned relative bias" multimodal`
17. `DeHate hateful video dataset temporal localization method 2025`
18. `LELA training-free hateful video localization arxiv 2602.09637 method`
19. `"hateful" OR "hate speech" video "temporal distance" OR "distance bucket" OR "log-scale" relative bias cross-modal attention localization`

None of the 19 queries returned a hateful-video paper using a relative-position or distance-bucketed bias in
attention.

### 3.2 Primary sources read (full text or code, via WebFetch)

| Source | Task | Attention and positional mechanism | Match to candidate |
|---|---|---|---|
| MultiHateLoc, arXiv 2512.10408v3, Sec. 3.2 and 3.4 | weakly supervised temporal localisation (HateMM, MHC) | Per-modality encoders Eq. 1 `FFN(softmax(Q K^T / sqrt(d_k)) V) + F`; DCM-Fusion weights per time step from that step's features, then one attention over concatenated features. No positional or distance term described. Smoothness loss on adjacent frames is in the loss, not the attention. | No |
| MultiHateLoc code, github.com/Multimodal-Intelligence-Lab-MIL/MultiHateLoc, `src/models/fusion_models.py`, `attentional_fusion_models.py`, `basic_attentional_fusion_models.py`, `cbranch.py` | same | `nn.MultiheadAttention` / `nn.TransformerEncoder` with key padding mask only. The `TransformerFusion` variant adds a learned absolute position parameter `pos_encoder = nn.Parameter(torch.zeros(1, 1000, hidden))` to the inputs. No relative, distance or bucketed term in any file. | No. Absolute learned PE on inputs, inside self-attention, is a different mechanism from a learned distance-bucketed bias inside cross-modal attention. |
| HateClipSeg, arXiv 2508.01712v2 | dataset; localisation benchmark | Localisation baseline is ActionFormer on unimodal visual features with late fusion by majority voting. No cross-modal attention. "VERA" does not appear in the paper and was not found by search 12. | No |
| ActionFormer code (`third_party/actionformer/libs/modeling/blocks.py`, `backbones.py`) | temporal action localisation | Local-window self-attention; optional `use_rel_pe` learned relative bias added to the local attention map (`att += self.rel_pe`), default `False`; no released config enables it; `use_abs_pe` on for some configs. Unimodal self-attention. | No. Generic TAL transformer baseline; mechanism is within-modality, windowed, and off by default. |
| RAMF, arXiv 2512.02743v2 | video-level classification | LGCF: Conv1D (k=3) local channel + global average pool, gated. SCA: `softmax(Conv2D_heads(Q K^T / sqrt(d)) + M)` with causal mask M and HeadMix. No positional encoding or distance-indexed bias. | No. The 3x3 Conv2D on the logit map is a local smoothing of attention logits, not a distance-bucketed learned bias, and the model does not localise. |
| CLARA, arXiv 2608.15905v1 | video-level classification | Gated MoE clip encoder over pooled embeddings; Transformer encoder "with positional encoding" (form unspecified) over rationale + clip tokens. No cross-modal attention; no distance term. | No |
| MultiHateGNN, arXiv 2509.13515v1 / BMVC 2025 | video-level classification | Graphs with edges from temporal adjacency, epsilon cosine similarity, and same-timestamp cross-modal links; GNN details in supplement. No positional attention bias. | No. Adjacency edges are a graph construction, not an attention bias; classification only. |
| CMFusion, arXiv 2505.12051v1 (ICDMW 2024) | video-level classification (HateMM) | "Temporal cross-attention" is Conv1D over time per modality with multiplicative interaction (Eq. 3-4); no Q/K/V, no positional term. | No |
| MM-HSD, arXiv 2508.20546v1 (ACM MM 2025) | video-level classification | `CMA(K,Q,V) = softmax(Q K^T / sqrt(d_k)) V`, Q = on-screen text, K = V = concatenated other modalities; no positional term; frame-level temporal CMA named as future work. | No |
| HVGuard, EMNLP 2025 (anthology abstract; PDF text is rasterised) | video-level classification | MLLM chain-of-thought + MoE fusion. No temporal attention described. | No |
| MoRE, WWW 2025 (chatpaper mirror of OpenReview) | video-level classification | Retrieval-augmented per-modality experts, sample-sensitive integration. No temporal attention described. | No |
| ImpliHateVid, ACL 2025, arXiv 2508.06570 | video-level classification | Two-stage contrastive encoders, no cross-modal attention. | No |
| TANDEM, arXiv 2601.11178 | localisation as text time spans | VLM/ALM with RL, 30 s chunks, structured text; no trained attention of its own. | No |
| LELA, arXiv 2602.09637 | training-free localisation | LLM prompting over captions; no trained attention. | No |
| Temporal label noise, arXiv 2508.04900v1 (MUWS 2025) | analysis | Uses HateMM / MHC baselines; proposes no model. | No |
| DeHate (ACM MM 2025) | dataset | No localisation method found (search 17). | No |

### 3.3 Nearest precedents outside hateful video (context only; not a rule-4 STOP condition)

- Relative position bias in NLP: Shaw et al. 2018 (clipped relative embeddings); T5, Raffel et al. 2020
  (bucketed relative bias, log-spaced for larger distances). The proposal cites both correctly and claims no
  invention of the component.
- Weakly supervised video anomaly detection: DAKD, arXiv 2406.02831v2, uses DeBERTa-style disentangled
  content-to-position / position-to-content attention with the signed distance clipped to a window (k = 25) to
  aggregate several visual backbones; not log-scale buckets, not audio-visual, no hate. LSTC (ICME 2023,
  arXiv 2303.18044) is reported by search 13 to add 3D relative position bias in a video-only spatio-temporal
  transformer; the abstract does not state it and I did not read the full text.
- Audio-visual event localisation: Ran et al., Applied Sciences 2022 (DOAJ listing; MDPI and DOAJ pages returned
  403), abstract describes a "temporally adaptive cross-modal attention" weighted by time differences between
  audio and visual features. Mechanism details not read.

These show that distance-aware attention is an established idea in neighbouring tasks. Rule 4's first condition is
about hateful video detection / localisation, where it was not found.

### 3.4 Fetches that failed

MM-HSD PDF (compressed streams; read the arXiv HTML instead), HVGuard PDF (rasterised; used the anthology
abstract), MDPI and DOAJ pages for Ran et al. 2022 (HTTP 403), Semantic Scholar API (HTTP 429), label-noise PDF
(read the arXiv HTML instead). None of these gaps concerns a hateful-video method with temporal attention.

## 4. The four STOP conditions

| Rule 4 condition | Triggered | Reason |
|---|---|---|
| Source method already used for hateful video detection / localisation | No | 19 searches and 15 primary sources (section 3). The only positional mechanism found in hateful-video work is MultiHateLoc's learned absolute input PE in one self-attention variant; no relative, distance-bucketed, or cross-modal positional bias anywhere. |
| Pure training / test ensemble | No | One network, one forward pass, 44 added parameters inside the existing attention. |
| Pure calibration / post-processing / smoothing | No | The bias changes the attention logits during training and inference and so the learned representation; it is not applied to output scores. |
| Pure engineering trick (hyperparameters, feature swap, augmentation, training config only) | No | It is an architectural change to the encoder with a stated hypothesis (locality in cross-modal retrieval improves within-video shape) and a one-switch ablation (table frozen at 0 = r5). It is small and it transfers a known component; rule 4 does not stop on size or on transfer. |

Whether order-blind attention is "clearly bad for this task" (the 2026-09-03 ruling's wording for counting a
MACIL-SD fix as backbone novelty) is decided by the experiment, not by this review.

## 5. Factual checks on the proposal

Verified correct:

- Table in section 1 against `runs/20261031_two_goal_diagnostics/backbone_gap/{hatemm,hateclipseg}.json`
  (3-seed means recomputed): HateMM r5 .6909/.8849/.7729 video ROC .9276; IO .6765/.8792/.7378, .9207;
  r5-shape-on-IO-level .6852/.8834/.7729, .9216. HateClipSeg r5 .6865/.6737/.5602, .8647; IO
  .6684/.6426/.5603, .7954; IO-shape-on-r5-level .6767/.6651/.5603, .8382. "Recovers .009 of the .015 AP gap"
  = .6852 - .6765 over .6909 - .6765. All match.
- r5 seed-234 reference numbers (HateMM .6944/.8867/.7887, HateClipSeg .6885/.6748/.5561) match the JSON
  `fixed8.official` of trial 9 and trial 13.
- Bucket arithmetic: K = 11 gives {0}, {1}, {2-3}, {4-7}, {8-15}, {16-31}, {32-63}, {64-127}, {128-255},
  {256-511}, {>= 512}; 4 heads x 11 = 44 scalars.
- "Within-video localisation comes mainly from the content model" is stated in
  `experiments/20260925_query_paradigm/README.md` (score decomposition, line 173).

Precision notes (not errors, no change to the verdict):

- "Buckets equal to the question tree's scales": the tree splits [0, T) at midpoints down to single seconds, so
  node lengths at depth d are floor or ceil of T / 2^d. The dyadic distance buckets equal the node scales only when
  T is a power of two; otherwise they approximate them. The proposal's "about" covers this; the rule-14 writing
  should keep that qualifier.
- Training uses the five I3D spatial crops (`crop_repeat`) and shrinks the batch for long videos (`long_T`); there
  is no temporal cropping, so every distance up to the full video length occurs in training. The >= 512 bucket is
  trained only on videos longer than 512 s (HateMM has some, per the r5 README).
- Section 1 should name the node prior next to the chain prior as the other downstream temporal structure, and
  section 2 or 3 should mention MACIL-SD's unused `masksize` hard-window option (section 2 of this review).

## 6. Scope of the novelty claim

Supportable, if the experiment shows the gain: a learned log2-bucketed relative-distance bias inside the shared
bidirectional cross-modal attention of a weakly supervised hateful-video localiser, with buckets set to the
question tree's dyadic scales and a zero-initialised table whose freezing recovers r5 exactly as the ablation.

Not supportable: inventing relative position bias or log buckets (Shaw 2018, T5), first distance-aware
cross-modal attention in general (Ran et al. 2022 in AVE), or first relative-position attention in weakly
supervised localisation (DAKD in VAD).
