# Deep Post-Hoc Novelty Check — C5 Effective Mechanisms (iteration step 12)

**Date:** 2026-08-30. **Scope:** per-mechanism novelty of what the passed pilot
actually showed (README.md final table), independent of NOVELTY_C5.md's
pre-pilot sweep. Method: WebSearch/WebFetch, arXiv + CVPR/ECCV/ICCV/AAAI/MM/ACL
2020–2026, adversarial stance.

---

## Item 1 — Zero-shot LOO cross-corpus span transfer beats all target weak supervision

**Verdict: open-with-differentiation (phenomenon crowded, carrier open).**

Closest prior art:

| Paper | What it does | Gap vs. ours |
|---|---|---|
| **LaGoVAD / PreVAD** (ICLR 2026, arXiv:2503.13160) | Pretrains on PreVAD (35,279 videos, **video-level weak labels only**, MIL + language alignment); zero-shot transfer **beats target-trained weak methods**: UCF-Crime 81.12 AUC vs VadCLIP-on-UCF 80.16; XD-Violence 74.25 AP vs VadCLIP-on-XD 67.43. | The generic phenomenon "zero-shot cross-dataset transfer > target-trained weak supervision" is **published**. But their carrier is *scale* (35k weakly labeled videos + language); ours is *span positions* from 2–3 tiny corpora (shuf_span ablation: .53–.55 ≈ chance proves position, not corpus statistics, carries transfer). No span supervision anywhere in their training. |
| **RefineVAD** (arXiv:2511.13204) | Weak-source zero-shot UCF→XD transfer (77.6 AP). | Weak source; does not beat XD-trained weak SOTA; no adaptation. |
| **UBnormal** (CVPR 2022, arXiv:2111.08644) | Supervised (frame/pixel) synthetic source used to help real targets (ShanghaiTech/Avenue via CycleGAN). | Span-supervised source exists in VAD, but as *augmentation* to target training, not zero-shot-beats-target-weak; synthetic→real, single-direction. |
| **AherNet** (ECCV 2020, arXiv:2008.13705) | Span-annotated source categories → weakly supervised target categories (TAL). | Cross-category within one data universe; transfer helps but no claim/evidence that pure zero-shot span transfer beats target-trained weak methods. |
| **VT-TWINS** (CVPR 2022, arXiv:2203.16784) | HowTo100M-pretrained step localization surpasses models trained on target CrossTask (40.7 vs 35.5–37.1 recall). | Transfer-beats-target-trained precedent in localization, but self-supervised source, no spans. |
| **TANDEM** (arXiv:2601.11178) | MLLM SFT on 100 HateMM videos **with span/IoU rewards**, tested on ImpliHateVid for generalization. | Span-supervised source transfer *inside the hate domain* exists — but transfer eval is **video-level classification only**, no temporal localization on the target, no LOCO, 1 source. Must be cited. |

Not found anywhere: a controlled demonstration that **span-supervised** source
training on **small** auxiliary corpora, transferred zero-shot, outperforms
*every* weakly-supervised and training-free method trained/run on the target —
in VAD, TAL, or hate. Also not found: any MIL-vs-full-supervision *transfer*
comparison study (the closest, Chéron et al. NeurIPS 2018 "flexible levels of
supervision", is within-dataset).

**Do not claim:** "first to show zero-shot cross-dataset transfer beats target
weak supervision" — LaGoVAD owns that sentence in VAD. **Claimable:** span
positions as the transfer carrier (shuf_span attribution), at 3-corpora scale,
in the hate domain, on within-video ordering.

## Item 2 — MIL adaptation destroys transferred ordering; rank-preserving distillation repairs it

**Verdict: open-with-differentiation (finding open, tool taken).**

Closest prior art:

| Paper | What it does | Gap vs. ours |
|---|---|---|
| **AMR** (arXiv:2510.19622, moment retrieval) | Two-stage: augmented-data pretrain → real-data adaptation; direct fine-tuning "forgets" boundary/semantic knowledge; fixed by cross-stage distillation from frozen stage-1 queries. | Closest structural match to "adaptation forgets localization knowledge + distill-from-pre-adaptation fix". BUT: same corpus (augmented→real), forgetting driver is data scarcity not the *weak MIL objective*, distillation is query/feature consistency not rank-preserving, and no ordering metric is tracked. |
| **LwF / self-distillation fine-tuning line**; **Ranking Distillation** (KDD 2018), **RankDistil** (AISTATS 2021) | Distill from the pre-adaptation model to prevent forgetting; pairwise/top-k order-preserving distillation objectives. | Tools, fully established. The loss cannot be claimed. |
| **OnPoint** (arXiv:2607.00289) | Offline→online multi-level distillation for point-supervised TAL. | Teacher-student across inference regimes, not forgetting-during-weak-adaptation. |
| WSTAL MIL pathology literature (e.g. CO2-Net, cross-video context arXiv:2308.12609) | MIL over-focuses on discriminative snippets → incomplete localization. | Known as a *training-from-scratch* pathology; nobody measures MIL adaptation **destroying a transferred within-video ordering** (our .63→.58, .73→.66) or repairs it with an ordering constraint. |

**Claimable:** the *diagnosis with measurement* (ordering metric before/after
naive MIL fine-tune, across 4 corpora) + the demonstration that pairwise
margin-ranking self-distillation is the minimal repair (+.09 over naive on
HateMM/EN). Cite Ranking Distillation/SDFT/AMR openly as the tool lineage;
claim the finding, not the loss. A reviewer can still say "AMR + rank loss";
the defense is the ablation table, which the pilot already has.

## Item 3 — Val-selected adaptation depth per corpus (epoch 0 in candidate set)

**Verdict: taken (as mechanism) / trivial engineering.**

This is standard model selection / early stopping (instance- and
dataset-dependent early stopping are studied, e.g. arXiv:2502.07547; "how much
to fine-tune per target" guidance is folklore across transfer learning). No
paper claims it as a contribution and neither should ours. The only defensible
sentence is protocol-level: putting **zero-shot in the candidate set** unifies
"adapt or don't" into one val-driven decision and is what resolved the EN/ZH
regression (gate note in README). Present as one line of protocol + one
sentence of analysis ("selected depth correlates with source-target
definition alignment"), never as a contribution bullet. If the depth-vs-corpus
pattern were shown to be *predictable* from a measurable corpus property, that
would upgrade it — the pilot has no such evidence (selected depths are also
seed-unstable: 4/1/15, 15/0/0).

## Item 4 — Negative-result package (VLM window scoring / ASR hate classifier / appearance kNN all fail at within-video ordering)

**Verdict: crowded, and carries a direct contradiction risk.**

- Generic MLLM temporal-grounding failure is well documented (ToG-Bench
  arXiv:2512.03666, Know-Show arXiv:2512.05513, SVAG-Bench: models "correct
  semantically but fail to ground temporally", <half human performance).
  The generic sentence is taken.
- **LELA** (arXiv:2602.09637, "Towards Training-free Multimodal Hate
  Localisation with LLMs"): training-free GPT-4o-mini frame-level hate scoring
  on **HateMM + MultiHateClip** — the same corpora — reporting ROC-AUC ~67.5
  and "outperforms all training-free baselines, approaching supervised". This
  *appears* to contradict "VLM scoring fails (.41–.59)". The reconciliation is
  the metric: LELA's ROC is pooled across hateful and non-hateful videos
  (between-video separation dominates); ours is within-hate-video ordering.
  This distinction MUST be made explicit, and ideally a LELA-style multi-stage
  pipeline should be evaluated under the within-video metric in the scale-up
  (currently our MLLM repro is Qwen window scoring, a weaker representative).
  Without that, a reviewer holding LELA kills the negative-result section.
- ASR-hate-classifier and appearance-kNN failures at *within-video ordering*
  specifically: not documented anywhere (HateMM/MHC modality ablations are
  video-level). Open, but minor — worth a paragraph + table, not a headline.

**Claimable:** the unified within-video-ordering evaluation showing all three
training-free/proxy families fail where span transfer succeeds — as *analysis
supporting the main claim*, explicitly reconciled with LELA's pooled numbers.

## Pairwise cross-corpus occupancy check (HateMM / MHC-EN / MHC-ZH / HateClipSeg)

No 2025–2026 paper found doing corpus→corpus transfer among these for temporal
localization, on any pair:

- **HateMM↔MultiHateClip:** MultiHateLoc (arXiv:2512.10408) uses both but
  trains per-corpus, video-level weak supervision, no cross-corpus transfer.
  LELA uses both training-free (no transfer learned). Memes→Videos
  (arXiv:2501.15438) is cross-*modality* augmentation, video-level
  classification only. Temporal-label-noise paper (arXiv:2508.04900) uses both
  corpora's spans for analysis, proposes no method.
- **HateMM→ImpliHateVid** (not our set): TANDEM — classification-only transfer
  eval. Closest existing hate cross-corpus result; cite.
- **HateClipSeg↔anything:** nothing found; the dataset (MM 2025,
  arXiv:2508.01712) is only benchmarked in-corpus (its own baselines,
  SafeLens-style LLM pipelines). No inbound or outbound transfer paper.

The four-corpus LOCO protocol remains unoccupied.

---

## Overall claimability

Safe contribution statement (in this order):

1. **First cross-corpus (LOCO) study of temporal hate localization** across
   HateMM/MHC-EN/MHC-ZH/HateClipSeg; target is span-free (video labels only),
   auxiliary corpora supply span supervision — stated verbatim in every table.
2. **Finding: foreign span positions transfer.** A frozen-feature temporal
   conv trained on other corpora's spans, zero-shot, beats every reproduced
   weakly-supervised and training-free method on within-video ordering on 3/4
   corpora; shuf_span shows positions, not corpus statistics, carry it. Frame
   as a *hate-domain finding with attribution*, positioned against LaGoVAD
   (weak-source, scale-carried analogue in VAD) — not as a new phenomenon.
3. **Finding: MIL adaptation destroys transferred ordering; pairwise
   rank-preserving self-distillation restores it** (measured before/after,
   4 corpora). Tool credited to ranking-distillation/SDFT lineage; the
   diagnosis and ablation are the contribution.
4. Supporting analysis: within-video ordering as the metric that exposes the
   failure of training-free MLLM/ASR-text/kNN proxies (reconciled with LELA's
   pooled-metric success), and the pooled-AP cost of uncalibrated zero-shot.

Do NOT say: "weakly supervised" unqualified; "first zero-shot transfer to beat
weak supervision"; "novel rank-distillation loss"; anything claiming item 3.

Scale-up obligations created by this check (feeds step 14): cite and position
LaGoVAD, AMR, TANDEM, LELA, MultiHateLoc; add a LELA-style training-free
pipeline under the within-video metric (or at minimum re-score LELA's protocol
distinction explicitly); keep OSAD-style joint multitask + CDL-style variant
baselines already planned.

## Single biggest novelty risk

**LaGoVAD (ICLR 2026).** It already prints the headline-shaped sentence
"zero-shot cross-dataset transfer outperforms methods trained on the target"
in the adjacent VAD field, at a top venue, with much bigger numbers. If our
paper leads with the phenomenon, it reads as a small-domain replay. The paper
survives only if the lead claim is the *carrier and the domain* — span
positions from tiny foreign corpora (proved by shuf_span), the hate
definition-shift setting, and the ordering-destruction/rank-repair finding —
with LaGoVAD cited in the second paragraph, not discovered by reviewer 2.
Secondary risk: LELA's pooled ROC ~67 being waved against our ".41–.59 MLLM
failure" if the within-video vs pooled metric distinction is not nailed down.

## Sources (this check)

- https://arxiv.org/abs/2503.13160 (LaGoVAD/PreVAD, ICLR 2026)
- https://arxiv.org/abs/2511.13204 (RefineVAD)
- https://arxiv.org/abs/2111.08644 (UBnormal, CVPR 2022)
- https://arxiv.org/abs/2008.13705 (AherNet, ECCV 2020)
- https://arxiv.org/abs/2203.16784 (VT-TWINS, CVPR 2022)
- https://arxiv.org/abs/2601.11178 (TANDEM)
- https://arxiv.org/abs/2510.19622 (AMR, moment retrieval two-stage distillation)
- https://arxiv.org/abs/2607.00289 (OnPoint)
- https://arxiv.org/abs/2502.07547 (instance-dependent early stopping)
- https://arxiv.org/abs/2602.09637 (LELA, training-free hate localisation)
- https://arxiv.org/abs/2512.10408 (MultiHateLoc)
- https://arxiv.org/abs/2501.15438 (Memes→Videos)
- https://arxiv.org/abs/2508.04900 (temporal label noise)
- https://arxiv.org/abs/2508.01712 (HateClipSeg)
- https://arxiv.org/abs/2512.03666, https://arxiv.org/abs/2512.05513 (MLLM temporal grounding failure benchmarks)
- https://arxiv.org/abs/2408.05191 (CDL, ECCV 2024)
