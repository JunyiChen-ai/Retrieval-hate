# Analysis: When Does an Open-Weights MLLM Earn a Method Role?

*Draft chapter — negative-results / mechanism analysis. All quantities are transcribed from
committed campaign documents; internal provenance is given as [DOC:file §section] and external
methods as \cite{} placeholders. This chapter reports project-closed results and is independent of
any pending headline-protocol decision.*

---

## 1. Research question

Our detector uses a multimodal large language model (MLLM) as a **frozen encoder**: Qwen2.5-VL
features \cite{qwen25vl} feed a retrieval-contrastive memory with a k-nearest-neighbour vote, the
core we port from RGCL / RA-HMD \cite{rgcl,rahmd}. A frozen encoder is a passive component, however.
The question this chapter answers is stronger and was fixed as a user mandate before any experiment:
**beyond encoding, can an open-weights 7B–72B MLLM earn a *removable, accuracy-bearing* method
role** — a component whose deletion measurably costs main-table accuracy? We treat a role as earned
only if removing the MLLM costs more than the noise floor of our ~150-video test sets (1 accuracy
point ≈ 1.6 videos; MHClip-EN n=161, MHClip-ZH n=149, HateMM clean n=215). To answer it we ran a
**thirteen-route pre-registered campaign** [DOC:CAMPAIGN_mllm_method_role.md], later extended by three
further pre-registered sprints (rounds 2–4, §3.6–3.9) that closed every remaining injection point in
the constraint box. Eleven routes give the
MLLM a distinct non-encoder job aimed at main-table accuracy (label-noise repair, prior
recalibration, neighbour reranking, evidence-density pooling, schema distillation, counterfactual
mining, score-level fusion, semantic speech compression, and decision-level LMM fine-tuning); two
localization threads (the P6 scorer and its P10/P11 escalation thread) — three routes at per-route
granularity (P6, P10, P11; cf. master-tables tension #7). The verdict is uniform and, we will argue,
mechanistically legible:
**the main-table-accuracy role is refuted across all eleven routes**, while the MLLM earns three
genuinely removable roles — encoder, localization scorer, and guard-rail/audit — none of which is a
main-table-accuracy role. This chapter is the analysis: the discipline that makes the negative
result trustworthy (§2), the five mechanisms that explain it and the four structural laws that rounds
2–4 crystallised them into (§3), what survives with ablation evidence (§4), and the implications (§5).

## 2. Methodology: how a negative result earns trust

A negative result is only as credible as the protocol that produced it. Five practices, applied
uniformly across the campaign, distinguish "we did not find a gain" from "there is no gain to find"
[DOC:TERMINUS_mllm_campaign_DRAFT.md §2].

**Pre-registration with frozen bars.** Every route fixes its success criterion, its candidate set,
and its accounting *before* touching held-out data. The localization amplifier P10 froze five
candidates and a +0.04 promotion line across two rounds and eleven comparisons, with the bar
explicitly not loosened for round two (pre-reg commit `3d641f4`); the generational probe P10-c fixed
a wv-AUC ≥ 0.616 gate (`8810c11`); the weak-supervision route P11 committed its probe gate before
the K30 cache existed (`eaf72db`). Bars set after seeing results are bar-shopping; ours were not.

**Probe-before-train.** For routes that would otherwise cost a GPU training run, we first ask a
cheap, no-head **probe** on the training benchmark whether the proposed signal even helps a
retrieval read-out. A failed probe kills the route with zero training (P3-EN, P7). A passed probe
buys a training run — but, crucially, does not license a claim (§3.3).

**Single test touch.** The HateClipSeg \cite{hateclipseg} localization test split was contacted
**exactly once**, by the single promoted amplifier (P10-b); P10-c and P11 never consumed it, and
P11's split (`p11_split.json`) remains frozen. Sequential comparisons accumulate against a fixed
anchor, so a lucky variant cannot be laundered into the headline.

**Protocol-matched floors, read two ways.** Classification numbers are reported under both the
pre-registered protocol (warmup ≥ 5, validation-selected) and a selection-free final-epoch protocol,
side by side [DOC:PAPER_MASTER_TABLES.md T1]. Every MLLM route is compared to the floor **under its
own protocol** — e.g. the decision-level fine-tune is judged against the protocol-matched LoRA
final-epoch floor of 0.8537 ± 0.0120 on ZH, not against a weaker frozen-CLIP baseline that would
manufacture an illusory gain.

**Honest-kill discipline.** Sub-1-point effects are recorded as *within-noise, no claim*; the
reported signal is the paired-delta sign pattern across seeds, not a p-value mined from a single
run. There is **no cross-seed ensembling** anywhere. Every verdict is guard-backed by a reproduction,
bit-for-bit sanity check, or probe, so a null is attributable to mechanism rather than a harness
artefact. When a *confounded* literal gate passed but the *matched* gate did not, we killed the
route (P11, §3.5) — the conservative direction.

## 3. Five mechanisms — and four structural laws (rounds 2–4)

### 3.1 Semantic competence is orthogonal to the decision variable

This is the most unifying failure shape. In every main-table route the MLLM demonstrably has real
semantic competence, and that competence is nonetheless **orthogonal to, or redundant with, the
quantity a main-table lift would need moved** [DOC:TERMINUS_mllm_campaign_DRAFT.md §3.1].

The sharpest case is neighbour reranking. Given a boundary query, a 7B judge labels each retrieved
neighbour COMPARABLE / INCOMPARABLE (label-blind) and we drop the incomparable ones before the vote
(P2). The intervention neither helps nor is neutral: B−A = −0.002 EN / −0.020 ZH, because the judge
over-flags INCOMPARABLE (83% EN / 70% ZH) and the drops are **indiscriminate** — the selectivity
lift, i.e. whether it preferentially removes vote-flipping neighbours, is +1.1% EN / −3.2% ZH
[DOC:EXP_p2_neighbor_rerank.md]. Topical comparability, which the MLLM can genuinely judge, simply
does not track label-match. The same shape recurs elsewhere: the verdict rate the MLLM would supply
as a prior drifts across the very temporal boundary it is applied to (P1: false-positive rate
.372 → .238 EN, so the train-calibrated count is biased exactly where it is used, p̂ error 0.22 EN /
0.18 ZH against a ≤ 0.07 criterion) [DOC:EXP_p1_zerolabel_recal.md]; the semantic channel it would
contribute to late fusion is **positively** correlated with the visual vote (corr +0.21…+0.51) and
is the weaker classifier (channel AUC 0.54–0.69 vs floor LOO accuracy 0.81–0.86), so fusion damages
more errors than it corrects (net −0.10…−0.38), falsifying the "decorrelated error channels"
intuition by measurement, not assumption (P7) [DOC:EXP_p7_score_fusion.md]; and the structured
archive fields it produces are linearly decodable (AUC .62–.93) *and* label-informative (AUC
.74–.78), yet distilling them adds nothing (train EN −0.001 / ZH +0.008) because they are **redundant
with the hateful label the embedding is already supervised on** (P4) [DOC:EXP_p4_schema_distill.md].
Semantic *aboutness* is not the same quantity as *which side of the hate/offensive/benign boundary* —
and it is that boundary, already directly supervised by the retrieval head, that a main-table gain
would have to move.

### 3.2 Scale improves calibration, not selectivity

If a bigger judge could rescue the rerank line, comparability would track label-match better at 32B
or 72B. It does not. Sweeping the P2b comparability judge across 7B → 32B → 72B (× two evidence sets
× two prompts, a train-side selectivity leaderboard) shows the judge becoming **better-behaved** —
the original-prompt drop-rate falls monotonically 72.5% → 64.6% → 30.9% on EN (58.2% → 50.7% → 14.9%
ZH), i.e. it stops being trigger-happy — while **selectivity stays pinned near zero**: the best EN
selectivity lift is +2.7 points anywhere against a +10-point promotion bar, and ZH is negative for
all eight configurations [DOC:EXP_p2b_stronger_judge.md]. A larger judge is more disciplined, not
more discriminating; comparability ⊥ vote-correctness is a **mechanism**, not an execution deficit,
so a still-larger (e.g. closed) judge is unlikely to reverse it. This is the single most direct piece
of evidence against "a bigger model will fix it," and its one exception (localization, §3.5) proves
the rule: there, the MLLM's semantic quantity *is* the evaluated target, so scale converts directly.

### 3.3 A passing no-head probe is necessary but not sufficient

The probe-before-train gate (§2) is a filter, not a predictor. Two of the campaign's cleanest probe
passes trained to within-noise, and the mechanism is the same both times. Evidence-density pooling on
HateMM produced the **cleanest probe of the three datasets** — +0.0108, k-consistent, on the densest
evidence (within-video score variance 1.28/0.71) — yet training the reweighted encoder against the
floor gave val-selected ΔF1 −0.0041 and final-epoch +0.0004, both under a point (P3-HateMM)
[DOC:EXP_p3_evidence_pooling.md]. Semantic speech compression on EN produced the **strongest probe of
the entire campaign** — a ≤60-word evidence-dense summary scoring 0.7523, beating both the floor
(0.7359, +1.6) and naive truncation (0.7067, +4.6) — yet trained to −0.023 / −0.079 and *below* naive
truncation (P8) [DOC:EXP_p8_semantic_compression.md]. The common cause: the learned alignment-fusion head
(elementwise image × text) **absorbs the input-space advantage** — it re-exploits the original, even
diluted, text — washing out the reweight the probe measured. The practical consequence is our
dual-protocol rule: a probe pass must be confirmed under both validation-selected and final-epoch
training before any claim.

### 3.4 Contrastive re-wiring redistributes accuracy; it does not add it

The last architectural locus is decision-level: LoRA-SFT the *whole* Qwen2.5-VL as the classifier and
read it out two ways — its own in-LMM MLP head, and our kNN memory over the fine-tuned embeddings
(P9). The MLP head only *matches* what we already had — ZH 0.8635 is +1.0 vs the protocol-matched
LoRA floor 0.8537 (within noise) and EN 0.7909 is +0.6 (noise) — while reading the same fine-tuned
space through our memory *loses* (EN −2.7 / ZH −2.2 / HateMM −4.7 below floor)
[DOC:EXP_p9_lmm_rgcl_video.md]. Turning our retrieval-contrastive (rgcl) loss **on** during the
fine-tune (P9b) is the decisive control: it trains the embedding space toward the memory vote
(D3−C3′ on the kNN read-out +1.8pt ZH / +0.2pt EN) but the in-LMM head mirrors **down** by the same
amount (−1.8 / −1.2 mlp) — a **head↔memory redistribution of roughly ±1.8 points, not a net gain**.
Across the full wave, 0/12 cells beat the floor (D3-knn ZH 0.8389, −1.5, 0/3 seeds; EN 0.7743, −1.0,
0/3). The MLLM's own head and the memory pillar contend for the same capacity: the head *displaces*
memory rather than *enhancing* it, and pushing accuracy from one to the other is not the same as
creating it.

### 3.5 Localization is the one scale-responsive lane

Exactly one lane behaves differently, and §3.1–3.2 explain why: in localization the MLLM's semantic
quantity — per-window hate saliency — *is* the evaluated target, so a better scorer converts
directly. The per-window evidence scorer ranks HateClipSeg windows at within-video AUC **0.5435** vs
memory 0.5140 and random 0.5088 (paired over memory Δ+0.0296, CI [+.009, +.050], p=0.007; vs-null
p=5.4e-8) — a removable role, modest in magnitude, solid in statistics (P6)
[DOC:EXP_p6_mllm_localization.md]. Amplifying it, a coarse×fine aggregation (A-fuse) grows
**monotonically with scorer size** on the HateMM calibration set: +0.0305 (7B) → +0.0437 (32B) →
+0.0526 (72B), whereas raw scale alone does not clear the bar (anchor-agg 0.5387 → 0.5512 → 0.5593)
[DOC:PAPER_MASTER_TABLES.md T2.2]. The promoted 72B A-fuse, on the single permitted HateClipSeg test,
reaches wv-AUC **0.5755** (bootstrap CI [0.5581, 0.5933], sign-p 1.4e-9, n=329; paired vs memory
Δ+0.0615 CI [+.0359, +.0869]; vs P6-7B Δ+0.0319 CI [+.0170, +.0474], p=0.0024) — **modest
amplification**, not substantial: 0.5755 < 0.60 [DOC:TERMINUS_mllm_campaign_DRAFT.md §6].

Three walls close the open-source ceiling. **Re-aggregation**: the strongest legal recombination of
existing 7B/32B/72B scores tops out at calibration wv-AUC 0.5932. **Scale**: the 72B champion is
0.5913. **Generation**: Qwen3-VL-32B A-fuse (0.5866) lands inside two-generations-earlier
Qwen2.5-VL-32B noise (0.5825), and the 30B-A3B model with 3B active parameters is weakest — so
localization is governed by *active* parameter count, not generation, and A-fuse significance
replicates across five scorers [DOC:PAPER_MASTER_TABLES.md T2.2]. All three sit below the 0.616
calibration line that extrapolates to a substantial test result; substantial localization is
unreachable in the open-source domain on this cluster. Finally, distilling the 72B teacher into a
*trained* segment head buys nothing over cheap supervision (P11): the teacher's edge over a
same-operator video-label MIL proxy is +0.0359 (CI [−0.0009, +0.0730], sign-p 0.13, not significant),
because the 72B advantage is a coarse×fine **aggregation trick**, not a better per-segment labeller —
a 5-fold linear MIL head already reaches ~0.55 wv-AUC, so **video labels alone already contain most of
what the MLLM weak label would teach** [DOC:EXP_p11_weaksup_localization.md]. The localizer's edge is
real and large **versus memory** (A-fuse − memory +0.0996, CI [+0.0635, +0.1366]) but not versus a
trivially-supervised MIL head — which sharpens, rather than removes, its role (§4).

### 3.6 Structural law I — better signal without conversion (eight instances, now arithmetic; F44 the mechanism)

Beyond the thirteen-route campaign, three further pre-registered sprints (rounds 2–4,
[DOC:TERMINUS_round2_mllm_plus3.md, DOC:TERMINUS_round3_mllm_plus3.md, DOC:ROUTER_GATE_RECORD.md,
DOC:FA_GATE_RECORD.md, DOC:PREMISE_D_GATE_RECORD.md]) hardened §3.1 and §3.3 into a
single law with **six independent instances**, and a post-terminus red-team audit (round 5, §3.10)
added **two more** — bringing the total to **eight** — and, decisively, made the law *arithmetic*
rather than merely repeated (the F66 decomposition below). In each instance a candidate signal is
demonstrably *richer* than the pipeline already has, and yet the best in-constraint operator converts
**none** of it into main-table accuracy. Each shares a sharp form — a **gold/label oracle proves the
convertible headroom is present**, but no unsupervised, frozen, or even supervised operator inside the
constraint box recovers it:

- **P3** (evidence-density pooling, §3.3): the no-head probe passes on all three datasets — HateMM the
  cleanest at +0.0108 — yet training is flat (val −0.0041 / final +0.0004) because the learned
  alignment-fusion head absorbs the input-space reweight [DOC:EXP_p3_evidence_pooling.md].
- **S2S** (Qwen frame-group set-matching): a gold membership oracle shows **+0.0917 acc (HateMM) /
  +0.1399 (MHC)** of headroom, but the realizable MeanMaxSim operator delivers **+0.0035 acc / +0.0003
  macro-F1 on HateMM and −0.0397 acc on MHC-EN** — inside the permutation null on every sub-condition —
  so the retrieval-object / "don't-pool" family is dead across both encoders
  [DOC:S2S_PROBE_VERDICT_REVIEW.md, commit `2c96ab6`].
- **W2-A** (transcript-grounded vision key): the same-shaped oracle survives at **+0.0635 (HateMM) /
  +0.0970 (MHC)**, yet the sole binding conditional-info gate finds **Δacc −0.0000 (HateMM, CI
  [−0.0052, +0.0049]) / −0.0038 (MHC, CI [−0.0099, +0.0019])** over the 8960-dimensional best
  representation — "a clean CLIP-redundancy null," the reviewer's phrase, because the joint
  frames+transcript forward already banks the interaction the grounded key claims to add
  [DOC:W2A_PROBE_VERDICT_REVIEW.md, commit `7228373`].
- **Router** (per-item cross-channel selection, round 4): a *perfect* per-item router that sends every
  channel-disagreement video to the arm that is actually correct would gain **+0.1083 (MHC-EN) /
  +0.0498 (HateMM)**, yet the realizable router converts **+0.0000** at the deployable read and
  **−0.0458** at the maximally-favorable dev-CV ceiling (below the permutation null p95 of +0.0042) —
  the decision-level meta-features carry no per-item routing signal (developed as its own closure in
  §3.8) [DOC:ROUTER_GATE_RECORD.md, commit `30d0ee1`].
- **FA** (modality-fusion / cross-encoder composition, round 4) — **the sharpest statement of the
  law.** A cross-encoder key that composes CLIP's image stream with Qwen's better text stream
  (`CLIP-imĝ ⊕ Qwen-text̂`) lifts the MHC-EN dev AUC to **0.898 — the highest value measured anywhere
  in the campaign** — i.e. it improves the *exact quantity* (ranking / AUC) that B5 proved
  unconvertible. And it still buys **no accuracy**: the only grid point that even *looks* Pareto
  (Δacc +0.050) fails the bootstrap CI ([−0.0625, +0.150]), fails the selection-null (p = 0.766, below
  the noise median), and its **label-oracle-threshold** edge is only **+0.025 < +0.03** (the ported B5
  kill-switch fires), while the identical test passes on HateMM's genuine win (+0.0467). A within-Qwen
  reweight is a pure **rotation** at every mixing weight (F44-exact +0.040 hate / −0.036 non-hate at
  50/50). The best possible ranking, and zero accuracy — better-signal-without-conversion made
  literal [DOC:FA_GATE_RECORD.md, commit `e0877c9`].
- **Premise-(d)** (healthy-image ⊕ *adapted*-text composition, round-4 closing) — **the law tested
  against its own escape clause.** FA closed EN's *frozen* composition, but its ban carved out one
  untested cell: "conversion requires adaptation," i.e. compose CLIP's healthy image stream with the
  LoRA-*adapted* Qwen text stream instead of the frozen one — the exact lever that converts on ZH (F45,
  §3.9). A $0 gate reusing the FA machinery bit-exact (the frozen control reproduces FA-A2 to
  **0.000000** absolute error, peak AUC 0.8982) measures it: the adapted text stream does **not** close
  the +0.005 oracle gap (the max label-oracle `d_oracle` anywhere on the grid stays **+0.0250 < +0.03**,
  the ported B5 kill-switch fires) and in fact **degrades** the composite — peak dev AUC **0.8982 →
  0.8698 (−0.0284)**, the mirror image of the ZH conversion. So even the adaptation the ban itself names
  as the conversion mechanism converts none of it; this **sixth** instance completes the F50 story by
  closing EN at *every* composition level — frozen, collapsed-adapted, and healthy-image ⊕ adapted-text
  [DOC:PREMISE_D_GATE_RECORD.md, commit `6e6061b`].
- **LP** (label propagation / graph diffusion over the kNN memory graph, round-5 audit) — the
  decision-*aggregation topology* opening, the one un-enumerated in-box decision operator (multi-hop LLGC
  over the *same* frozen fused keys, escaping the F46 named-operator list and the F47 per-item selection
  closure). It converts nothing and actively degrades: on dev the gain is **monotone-negative** in
  diffusion strength — HateMM best **−0.0187**, MHC-ZH **−0.0385** (α = 0.9 catastrophic, −0.19 / −0.22
  breaking 23 of 78 items), MHC-EN **+0.0125** = net +1 item on n = 80, deep inside a permutation null
  (p95 **+0.063**) whose centre is *positive* (diffusion helps random labels *more* than real ones). The
  one-hop head already sits at the 1-hop-separable ceiling, so the MHC-ZH oracle headroom of **+0.1026**
  stays entirely unconverted — the **seventh** instance, closed at $0 with zero test-touch
  [DOC:LP_GATE_RECORD.md, commit `7be6e3f`].
- **Vision-unfreeze LoRA** (round-5 audit) — the **representation-level** instance, and the one that
  refutes law I's *own* escape wording. Unfreezing the ViT tower and projector inside the LoRA-SFT
  (320 ViT-LoRA tensors, census-verified) is the **first lever ever to *move* the collapsed MHC-EN image
  stream** — image-only train-LOO AUC **+0.0320**, dev **+0.0065**, reviewer-reproduced bit-for-bit —
  refuting the F51 / GAP-5b "no vision lever was ever tried / EN is closed to the entire representation
  family" wording at the mechanism level (§3.9). Yet the decisive add-over-generic bar (K-V2) is a **TIE
  on both datasets and both protocols** (HateMM val-sel −0.0016 acc 0/3, final +0.0000 1/3; MHC-EN val-sel
  +0.0269 acc but sign only 2/3, a wide-between-seed-spread artefact, final −0.0062 1/3): the upstream image
  representation genuinely improved and the head converted **zero** of it. The **eighth** instance,
  ~15 GPU-h [DOC:VISION_UNFREEZE_VERDICT_REVIEW.md, commit `09d02f8`].

*(The round-5 learned-audio gate is deliberately **not** an instance: the Whisper-encoder stream added no
conditional information on any dataset [DOC:LAUD_GATE_RECORD.md, commit `3573f82`], but with **no oracle
surplus** — the signal itself is absent, exhausted by the ASR transcript — so it is a redundancy null, not a
better-signal-without-conversion datum.)*

These eight instances are unified by a single **mechanism**, surfaced by the encoder swap itself — the
result that turns the campaign's central positive from an anomaly into a law. A zero-GPU geometry
diagnosis on banked train/dev caches shows Qwen's representation upgrade is **real and roughly equal on
all three datasets** — top-20 neighbourhood purity rises **+0.023 / +0.023 / +0.021** and the
text-stream AUC rises **+0.041 / +0.054 / +0.045** on HateMM / MHC-EN / MHC-ZH — and yet it converts to
accuracy on **HateMM only** [DOC:ENCODER_SWAP_DIAGNOSIS.md, commit `8a48938`]. The mechanism has two
legs, both *dataset* properties rather than method-fixable ones:

1. **Modality-locus × multiplicative fusion.** HateMM's hate is visually grounded (image-only train-LOO
   AUC 0.826), so Qwen's uniformly better text stream rides on a neutral-strong image stream and the
   fused gain is a clean Pareto move (hate-recall +0.116 (dev) at **zero** non-hate cost). On MHC-EN the Qwen
   **image stream collapses to near-chance (0.734 → 0.599)**, and because the deployed head fuses the two
   L2-normed projections by an **element-wise (Hadamard) product** (`fusion_mode='align'`,
   `src/model/classifier.py:110–122`), that collapse **corrupts the fused key multiplicatively** and
   cancels the +0.054 text gain (net dev −0.012). *(Erratum, F48/F50, commits `6032d32` / `e0877c9`:
   F44's earlier prose described this fusion as an equal-weight L2-normed **concat**. The diagnosis
   **numbers stand** — F44's concat-kNN read-out is a sign-faithful proxy, and the FA gate reproduces
   the deployed dev sign to −0.0125 vs F44's −0.012 — but the deployed fusion is **align/Hadamard**, so
   the corruption is **multiplicative, not a cancelling 50/50 block**, and the head has* less *attenuation
   capacity than a concat, because a linear projection cannot zero a modality inside a Hadamard product;
   FA then measures the cell F44 had only asserted and confirms no reweight converts.)* The collapse
   persists at 32B (image AUC 0.608), which is exactly why *scale regresses* rather than
   rescues — the diagnosis retro-predicts B2 (§3.2).
2. **Representation-limited vs label-limited errors.** HateMM's residual errors are
   representation-limited, so a better encoder Pareto-fixes them; MHC's are a hard/label-limited core,
   so the same encoder only **rotates** the ranking (hate-recall +0.040 / non-hate −0.036 (dev);
   net +5 videos fixed on HateMM vs −1 on MHC-EN) — an AUC gain that B5 already proved unconvertible to
   accuracy at any operating point, including the label-oracle cut [DOC:B5_VERDICT_REVIEW.md, commit
   `50f01b9`]. The companion figure (fig_pareto_rotation) shows the same phenomenon on the binding test
   footing (final-epoch, 3-seed: HateMM +0.128 / +0.008; MHC-EN +0.095 / −0.033) — the MHC-EN minority
   gain is larger on test but remains a rotation.

This account **unifies three prior verdicts** the paper previously left disconnected — SAV (MHC-EN is
data/label-limited; the dilution hypothesis is falsified), B5 (the ZH/MHC ranking edge is
easy-example ordering), and B2 (scale regresses on MHC) — and, read alongside P3 / S2S / W2-A / router /
FA / premise-(d), states the law in its general form: **a signal being measurably better is necessary but not
sufficient for a main-table gain; what decides conversion is where the gain lands — which modality,
which error type — and whether the decision metric can absorb it, not how much better the signal is.**
FA is the clean edge case: it drives the ranking to its campaign maximum and converts nothing, because
the MHC-EN core is label-limited and the AUC it improves is exactly the quantity the accuracy metric
cannot absorb; premise-(d) then shows that *adapting* the text stream of that same composition does not
rescue it either — it lowers the ranking rather than converting it (§3.9). The design-time corollary sharpens §3.3's dual-protocol rule into a question to ask of
any auxiliary-signal proposal: not "is the signal richer?" (it usually is) but "is its advantage in the
modality and the error type the decision boundary is actually limited by?"

**Law I is now arithmetic, not merely eight-times-repeated (F66).** The round-5 ISR pre-gate closes the
last aggregation object — an independent per-segment re-encode read by a *uniform* per-segment-kNN
vote-mean, the sole operator that survives both the pooling ban and the per-item-selection ban — and in
doing so it *proves* the law rather than adding a ninth anecdote. On the banked CLIP sub-clip caches the
legal uniform operator is flat (HateMM **+0.0012** / MHC-EN **+0.0032**, both under the permutation null,
bootstrap 5th-pct < 0, ΔmF1 negative), while the vote machinery is bit-exact to the deployed one (Fano
= 1.0). The decisive step is the decomposition of the oracle headroom into a **legal symmetric slice**
(reachable by any non-selecting operator) and a **banned selection slice** (reachable only by the
law-III-forbidden per-item selector): HateMM's **+0.0776** of oracle headroom splits into **+0.0012**
legal + **+0.0764** banned, and MHC-EN's **+0.0700** splits into **+0.0064** legal + **+0.0636** banned —
so **91–98 % of the convertible headroom is formally disjoint from every legal operator**. The convertible
slice and the reachable slice do not intersect. Law I therefore stops being an observation repeated across
eight cells and becomes an **arithmetic statement** about where the headroom lives: the frozen-feature
operator can access only the symmetric slice, which every legal operator measures at ≈ 0
[DOC:ISR_PREGATE_RECORD.md, commit `a6e41f8`].

This has a name in the literature, which sharpens the claim from a campaign idiosyncrasy to a recognised
phenomenon. Law I is a **usable-information** ceiling in the sense of *V-usable information* (Xu et al.
\cite{xu2020vinfo}; Ethayarajh et al. \cite{ethayarajh2022vinfo}, whose pointwise-V-information formalises
exactly the gap between *information present* in a representation and *information a fixed model family can
extract from it*): the oracle proves the bits are present — so this is **not** a Shannon-information gap — but
they are not usable by any operator in the constraint box. In that framing the frozen-feature ceiling is a
V-usable-information ceiling, closable only by **expanding the model family**, which is precisely what
adaptation does and precisely where the campaign's one reliable conversion lives (§3.9); the same cached
features admit a $0 pointwise-V-information measurement that would quantify the gap per dataset, should the
paper want Law I stated quantitatively rather than by its eight instances [DOC:LITSURVEY_NOVEL_MECHANISMS.md
§3.1].

### 3.7 Structural law II — the cumulative-causal three-level closure

The second law is narrower but methodologically transferable, and it explains why every temporal /
"don't-pool" / per-frame route failed at once. Qwen2.5-VL's language backbone is **fully causal**
(`is_causal=True`, verified at the transformers-4.49 source level), so a per-frame-group vector g_t is
a **cumulative causal prefix summary, not a frame-local state**. Round 3 closed this at all three
operator levels simultaneously — the most complete closure in the campaign:

- **Structural (F35, postmortem commit `4358ca1`).** The groups *are* prefixes: a permutation-based
  temporal control is unsatisfiable by construction (a diff-colour-same-position stimulus pair scores
  cos 0.939 against a same-colour-diff-position pair at 0.674 — position dominates content), so
  frame-local order semantics are simply unavailable in these representations. The gate-0a control was
  accordingly replaced with a causal-consistent onset-invariance control [DOC:S2S_GATE0A_AMENDMENT_RULING.md,
  commit `20c0bf2`].
- **Unsupervised operator (F37).** Set-to-set MeanMaxSim over the groups adds nothing the pooled key
  does not already carry (§3.6): pooling is effectively lossless on cumulative-causal vectors.
- **Supervised operator (F39, CTF gate commit `0eb6d33`).** Even a **label-supervised** conditional-info
  probe over the flat [g_1 … g_T] tensor and the arc increment g_T − g_1 finds **exactly zero** beyond
  the pooled key (HateMM +0.0000, CI [−0.0031, +0.0031]; MHC −0.0029; arc −0.0049 / −0.0010), with
  label-oracle calibration accZA = 1.0 crediting the null as genuine rather than machinery-dead
  [DOC:CTF_GATE_RECORD.md].
- **Direct causal-mask attack (F72, round-6 audit).** The sharpest confirmation is adversarial rather than
  observational. Flip the LoRA-Qwen decoder's attention from causal to **bidirectional** at inference — the
  LLM2Vec / NV-Embed recipe \cite{llm2vec}, on the *same* banked adapters, same prefix-mean readout, same
  8 frames — and the head does not merely fail to gain, it **craters**: paired within head-seed, MHC-ZH mean
  **−0.1163** (val-sel) / **−0.1409** (final) acc and up to **−0.2802** macro-F1; HateMM **−0.1210** /
  **−0.1256** acc; **0/12 per-seed deltas positive**, a maximally concordant regression. This is *not* the
  flat Law-I null — it is the pre-declared DEGRADE / "Llama-pattern" branch, every mean ≈ 7–10× past the
  −0.014 line, exactly the failure LLM2Vec documents for causally-trained Llama weights under a naked mask
  flip. Removing the causal structure destroys the representation the deployed head reads, which is direct
  causal evidence that the pipeline's signal *lives in* the cumulative causal-prefix summary of this section
  rather than in any recoverable frame-local state [DOC:BIDIR_STAGE1_VERDICT_REVIEW.md, commit `f733bbe`].
  (Whether a Stage-2 MNTP repair recovers the loss is a user-gated funding question the DEGRADE branch routes
  to the user, not a closed result.)

The contribution generalises beyond hateful video: **anyone building set-matching or temporal-order
retrieval over decoder-VLM token summaries is operating on prefix summaries, not frame states, so the
"extra" temporal structure they hope to exploit is already integrated into each token** — a concrete,
transferable caution, established structurally and then confirmed at both the unsupervised and the
supervised operator level. (The same causal cumulation is why W2-A's grounded key was architecturally
real yet redundant in §3.6: the joint forward integrates the transcript into every vision token.)

**The temporal closure is now externally named and in-domain grounded (F81).** A final literature sweep
re-audited the temporal axis and confirmed the opposite of the "never varied" framing: temporal structure is
among the *most* heavily attacked axes, closed at four independent operator levels — an order-constrained
soft-DTW kernel (W2-C: observed Δ = the order-shuffle null's 95th percentile exactly), a set-to-set retrieval
object over the frame groups (S2S/F37/F38), the causal-prefix tensor (CTF/F39), and independent-segment
aggregation (ISR/F66) — plus frame count (F67), all unified by F35 (cumulative-causal prefixes) and F66's
symmetric-vs-selection arithmetic. Two external anchors sharpen this into a stated negative result. First, the
in-domain temporal-label-noise study (yang et al., arXiv 2508.04900) reports that HateMM / MHClip-EN hateful
videos contain **33 % / 35 % non-hateful segments** ("systematic, not random contamination") and that trimming
to the **gold** hateful spans reaches ~98 % macro-F1 — i.e. temporal burstiness converts to accuracy *only*
through timestamp selection, exactly the law-III-banned per-item gold selection: the in-domain confirmation that
F66's selection-lock is temporal as well as segmental. Second, the extraction path is verified **pacing-blind**
— Qwen2.5-VL receives monotonic frame *order* via mRoPE over T = 4 temporal-patch groups but no fps/timestamp,
and CTF measured that group tensor at exactly +0.0000 / −0.0029 conditional information — pinning that the
pipeline exploits the cumulative-causal *semantics*, not a recoverable temporal-dynamics signal (consistent with
the F72 bidirectional crater, §3.7 above). The genuinely virgin temporal sub-cells that remain (raw
optical-flow as a new modality; mRoPE absolute-time injection) each require raw-video re-extraction and inherit
the conditional-redundancy null that already zeroed audio (F64) and prosody (F41); a W2-C forensic even finds
the hateful class is *more static* (within-class cosine 0.899 > 0.874), inverting the reveal/escalation premise
[DOC:LITSWEEP5_TEMPORAL.md, commit `ad81ffb`].

### 3.8 Structural law III — per-item selection is closed at all three supervision sources

Rounds 2–3 closed *global* levers (a single operating point, a fixed fusion, a pooled key). Round 4
asked the last structurally-distinct question. The encoder swap only **rotates** the MHC-EN ranking
(§3.6), and that rotation has real per-item content — Qwen fixes some videos while breaking others. So
can a **per-item** selector route each video to whichever channel — the CLIP-encoder arm or the
Qwen-encoder arm — is right for *it*, converting the rotation into a Pareto gain? The answer is a clean
**no**, and it is a no at *all three* places supervision could come from — the most complete closure of
a selection family in the campaign [DOC:ROUTER_GATE_RECORD.md, commit `30d0ee1`].

The oracle headroom is real and large (the §3.6 instance): a perfect per-item router — send every
channel-disagreement video to the arm that is actually correct — would gain **+0.1083 on MHC-EN /
+0.0498 on HateMM**. Yet no realizable router recovers any of it, and the three failures are
mechanistically distinct:

- **Unsupervised / feature-conditional.** The conditional-information probes already zeroed the
  frozen feature space (K9: W2-A, CTF, GIR, §3.7) — there is no linear signal in `Z_best` to select on.
- **Train-supervised.** Fitting the selector on the training disagreement subset **degenerates**,
  because the retrieval head **memorises its own training bank**: CLIP leave-one-out train accuracy is
  **0.998** (vs Qwen 0.800), so on the *train* disagreement subset "Qwen is the correct arm" holds for
  **0 / 109, 0 / 102, 0 / 92** items across seeds — the exact inverse of the *dev* base rate
  (0.55–0.65). A train-fit selector therefore has **no dev-transferable supervision** and collapses to
  always-pick-the-majority-channel (routed − best-single = **+0.0000** on every seed, both datasets).
  This is a new obstacle specific to the frozen-artifact setting: the memorised bank makes the routing
  *target* non-transferable before per-item predictability is even in question.
- **Dev-supervised.** Even fitting the router *within* dev by cross-validation — an optimistic
  realizable ceiling that peeks dev labels — is **negative**: MHC-EN gradient-boosted −0.0458
  (CI [−0.0875, 0]), linear −0.0333, both below the permutation-null p95 of +0.0042 (observed p = 0.97).
  The decision-level meta-features (vote margins, neighbour purity, per-modality sub-votes, confidence
  differential, transcript indicators) carry **no per-item routing signal**, with or without
  nonlinearity, with or without in-distribution supervision.

A companion arithmetic recon fixes the **quantitative bar** any future router input must clear before it
earns a gate [DOC:MJ_FORENSIC_RECON.md, commit `d57d05d`]. On the 80-item MHC-EN dev split (disagreement
sizes 20 / 23 / 20, always-Qwen prior 0.588), clearing the +0.020 gate requires the selector to pick the
winning arm on a fraction **q ≥ 0.663** of disagreement items. But the *alignment ceiling* — how well the
true modality locus predicts which arm wins — is at most **0.588** (the global prior itself), and F44's
"no coherent subgroup" / F47's realizable read place it nearer **0.50–0.41**; a **perfect** modality
judge (which the archive already banks as a per-video `modality_cues` field, so no MLLM generation is
even owed) therefore yields a gain of ≈ 0 to −0.046 and **cannot** reach +0.020. This is the same
comparability ⊥ vote-correctness orthogonality that killed P2 (§3.1), now hardened into a
pre-measurement bar: **a per-item selector is admissible only if its input can be shown, from banked
evidence, to align with which-arm-wins above q = 0.663** — a threshold no signal in the constraint box
meets. The transferable caution generalises §3.1 to the selection setting: *richer per-item
side-information does not imply per-item routability; the binding quantity is the alignment between the
side-signal and the decision the router must make, and that alignment is measurable in advance.*

### 3.9 Structural law IV — the convertibility line runs through adaptation, not encoder identity

The first three laws concern *frozen* representations: a signal is richer but does not convert (§3.6), a
pooled causal key already carries the temporal structure (§3.7), a per-item selector has no signal to
route on (§3.8). The fourth law is the positive counterpart, and it is the one place a signal reliably
*does* convert. It concerns what changes when the encoder is **adapted** rather than swapped. Across the
three classification datasets the encoder *identity* swap (frozen CLIP → frozen Qwen) converts to
accuracy on HateMM only (§3.6), but *adapting* that same encoder with a small encoder-level LoRA-SFT
(r16/α32, generative word-label supervision on the dataset's own train split, vision tower and projector
frozen so only the language backbone moves — distinct from the §3.4 decision-level fine-tune, where the
whole VLM becomes its own classifier; here the adapted encoder feeds the *unchanged* retrieval head)
converts where the frozen swap could not.

The decisive contrast is ZH. The frozen Qwen swap **fails** on ZH (−0.0112 acc, 1/3 seeds — the round-2
B1 negative), merely rotating the ranking; the *same* encoder under LoRA **passes** the final-epoch
conjunct (+0.0313 acc / +0.0453 mF1, 3/3, marginal) [DOC:B3_VERDICT_REVIEW.md, job 13150]. A zero-GPU
decomposition locates the whole gain in the text stream (train-LOO text AUC 0.802 → 0.847 → 0.925 for
CLIP → frozen-Qwen → LoRA, image stream flat) and shows it converts as a genuine Pareto minority-recall
move (hate-recall +0.1111 at −0.0032 non-hate) rather than the frozen swap's rotation (+0.0741 hate
bought with −0.0481 non-hate) — LoRA crosses from re-rank to re-decide exactly where ZH hate lives, in
the language representation [DOC:B3_ZH_LORA_DECOMPOSITION.md, commit `d76e407`]. The convertibility line
therefore runs through *adaptation*, not encoder identity.

The round-4 LoRA-HateMM measurement is this law's **strongest confirmation** and completes the
three-dataset adaptation map [DOC:LORA_HATEMM_VERDICT_REVIEW.md, commit `6b8f634`, job 13235].
Encoder-level LoRA **passes HateMM under both protocols, solidly** — val-selected +0.0419 acc / +0.0460
mF1, final-epoch +0.0573 / +0.0682, 3/3 sign each, the val-sel acc cushion ≈ 9× B3's — while EN stays
**closed**: the bundled EN LoRA-encoder cell FAILs both protocols (val-selected −0.0021 acc, final-epoch
+0.0000), because EN is label-limited with a collapsed image stream (§3.6, F44) that no encoder move
converts. The three-dataset map is thus **HateMM solid-pass / ZH marginal-pass / EN closed**, and the
**performance-conjunct ledger now reads, with its protocol qualifier: under the final-epoch protocol one
lever — encoder-level LoRA — clears +0.03/+0.03 on two datasets (HateMM and ZH); under val-selection,
HateMM only** (ZH's val-selected pass is lost to the 78-dev selection tax, §2). This is the first single
lever to clear ≥ 2 datasets in the campaign — but it is one lever with two mechanisms, not one mechanism.

That last distinction is where the LoRA-HateMM family-coherence flag (KS-2) earns a sentence. Despite
adapting **only** the text-generative pathway — the vision tower and projector are frozen, so LoRA never
touches the image stream — LoRA on HateMM **matches** the frozen-Qwen encoder (final-epoch LoRA 0.8698 ≥
frozen-Qwen 0.8682; val-selected within the 0.014 seed band), so the honesty flag does not trip. Read
through §3.6's modality mechanics this is exactly what F44/F45 predict, not a coincidence, and a zero-GPU
per-stream decomposition of the passing cell now measures it directly [DOC:HATEMM_LORA_STREAM_DECOMP.md,
commit `51eb95b`]: HateMM's image stream is strong and swap-neutral (image-only train-LOO AUC in the 0.82
band, uncollapsed unlike MHC-EN's 0.599), and LoRA leaves it flat (ΔAUC +0.0045 train-LOO / +0.0062 dev);
the decisive single stream is actually **text** (text-only ≥ image-only for CLIP, frozen-Qwen, and LoRA
on both footings). LoRA does sharpen that text stream (train-LOO 0.888 → 0.920, the ZH signature), but it
adds ≈ 0 downstream (+0.0015 acc final / −0.0108 val-selected) **because the frozen swap already
converted HateMM's text signal to a Pareto** (frozen-Qwen − CLIP +0.0558 acc) — there is no further
boundary for the sharpening to move. The two passes of the one lever therefore convert through the
**same** decisive modality — text — but by different levers: ZH's is text-borne and LoRA-specific
(frozen-Qwen fails there, so adaptation is the necessary lever), HateMM's is text-carried on a
swap-neutral image base and frozen-swap-sufficient, inherited by LoRA (LoRA ≈ frozen-Qwen there). Whether
an encoder-class adaptation lever, however it performs, satisfies the goal's *novelty* clause is the
standing D7 user ruling; the mechanism analysis fixes only what the lever does, not whether it counts.

The round-4 **curriculum coupling probe** completes the map. One question the frozen-vs-adapted contrast
leaves open is whether *coupling the retrieval memory into the adaptation objective* — rather than a
generic word-label SFT — converts where generic LoRA ties or fails. The cand-2 probe answers it: a
confusion-weighted SFT curriculum, whose only manipulated variable is how often each train video appears
(weighted by the memory's leave-one-out confusability, cost-neutral to generic), **ties** generic LoRA
on the primary ZH leg under both protocols — the pre-declared most-likely outcome, "generic LoRA with
reshuffled data" — and **adds** over generic on exactly one cell, HateMM val-selected
(+0.0155 acc / +0.0166 mF1, 3/3 on the draw-1 curriculum; pooled weakly-hardened across two draws,
5/6 sign, per-draw 3/3 gate not met), tying on HateMM final-epoch by 0.0007; it
does **not** strengthen the marginal ZH leg [DOC:CAND2_VERDICT_REVIEW.md, commit `546acc5`, job 13241;
DOC:CAND2_REP2_VERDICT_REVIEW.md, commit `aa48275`, job 13246].
Read honestly, the memory→adaptation coupling's measurable effect over generic LoRA is **dataset- and
protocol-local**: the RGCL head already re-mines the confusable boundary per-epoch from the frozen
extracted features, so on the primary leg it has already extracted what the curriculum tried to inject
into the encoder — the §3.3 "the objective already sees the hard structure, curriculum is redundant"
pattern, one level up on the encoder.

With cand-2 the adaptation family is **completely mapped**, and the map reads as a phase diagram of where
adaptation converts:

- **generic encoder-level LoRA** — HateMM PASS both protocols (solid), ZH PASS final-epoch only
  (marginal), EN FAIL both;
- **memory-coupled curriculum LoRA (cand-2)** — ties generic on ZH (both protocols), adds over generic
  on HateMM val-selected only (pooled weakly-hardened across two draws, 5/6 sign; per-draw 3/3 gate
  not met), and structurally opens no new dataset (a text/curriculum
  lever can only hold ZH and add HateMM-or-EN; HateMM is inherited (frozen-swap-sufficient, its
  convertible signal text-carried), EN is label-limited);
- **retrieval-loss-coupled decision-level fine-tune (P9b)** — dead: a head↔memory redistribution of
  ≈ ±1.8 points, not a net gain, 0/12 cells above floor (§3.4);
- **the EN composition family** — closed at every level: frozen swap (B1/F44), generic LoRA (B4/F53),
  curriculum (opens no new dataset), and even the healthy-CLIP-image ⊕ *adapted*-Qwen-text composition
  (premise-(d), §3.6).

Read across the diagram, **adaptation converts exactly where the dataset's decisive modality is the one
the adaptation reaches and the residual error core is representation-limited — with the qualifier that on
a dataset where the frozen *identity* swap already converts that same modality (HateMM), adaptation only
inherits the conversion rather than being the necessary lever; it ties or fails everywhere else.** ZH's
text-borne LoRA gain is the one place adaptation *itself* is the converting lever (the language pathway
is where ZH hate lives, and frozen-Qwen fails there); HateMM's pass is inherited from the frozen swap
(LoRA ≈ frozen-Qwen), its convertible signal likewise **text-carried** (text is the decisive single
stream) but fused with a strong swap-neutral image stream that the frozen swap already converts, so
LoRA's further text-sharpening adds ≈ 0 [DOC:HATEMM_LORA_STREAM_DECOMP.md, commit `51eb95b`]; and EN,
which is label-limited with a collapsed image stream, is unreachable by every adaptation the box
permits — generic, memory-coupled, or composed with a healthy foreign image stream.

A **scoping correction** travels with this map, for honesty about the mechanism prose. The decompositions
above describe the ZH LoRA gain as living "in the text stream, image stream flat," and read HateMM's KS-2
result as LoRA "never touching the image stream." A source-level recon (F54) sharpens these into an
empirical, not architectural, claim: the vision tower and multimodal projector are indeed frozen, but the
LLM backbone that re-contextualises the vision-pad tokens *is* LoRA-adapted (`lora_target: all`), and the
banked `img_feats` are pooled from the vision-token span of a forward that passes **through** that adapted
backbone — so the image stream is architecturally *movable* by an SFT target that routes gradient through
the vision tokens. It stays flat only because every SFT target in this campaign is a text-decodable
yes/no with the transcript present, which routes gradient into the language pathway. Nothing in the phase
diagram changes — F50/premise-(d) already price EN's *healthy* image stream out below the oracle bar, and
HateMM/ZH already pass — but the "text-only" phrasing should be read as "text-only *for these targets*,"
not as a claim that the vision path is unadaptable [DOC:TIE_BRANCH_RECON.md, commit `6b9985a`].

The round-5 **vision-unfreeze** measurement (F65, §3.6) then closes this correction empirically. Where F54
argued the image stream was *architecturally* movable, F65 unfroze the ViT tower and projector inside the
LoRA-SFT and *measured* the movement: the MHC-EN image stream did move (image-only train-LOO AUC **+0.0320**,
dev +0.0065, reviewer-reproduced bit-for-bit — the first lever ever to move it). This refutes the F51 /
GAP-5b "two-object closure" *wording* ("EN is closed to the entire representation family / no vision lever
was ever tried") at the mechanism level — the vision object was un-enumerated, and it is reachable. But the
movement converted **zero** head accuracy (K-V2 tie on both datasets and both protocols), so the phase
diagram is unchanged in substance: EN's image stream is now shown to be *movable and still unconvertible*,
exactly the F50/premise-(d) label-limit reading, confirmed at the representation level rather than assumed.
The F51 closure is thus correct in its *phase-diagram conclusion* (no adaptation the box permits converts EN)
and wrong only in its enumeration ("two adapted objects"): capacity/reach was a third object, and it too
fails — the eighth better-signal-without-conversion instance [DOC:VISION_UNFREEZE_VERDICT_REVIEW.md,
commit `09d02f8`].

### 3.10 The post-terminus audit — the mechanism picture holds, with two small-head optimization notes

The four laws were crystallised on rounds 2–4; a post-terminus red-team (round 5) and a literature-driven
sweep (round 6) then re-opened every prose-argued gap the laws rested on and measured each one dead, at ~16 +
~3.5 GPU-h, with the project's best numbers unchanged (the experiments chapter §8 is the ledger). None
reopened a law: the decision-aggregation *topology* (label propagation, F63) over-smooths and is closed at
$0; the last aggregation object (ISR, F66) is arithmetically selection-locked (§3.6); the representation cell
(vision-unfreeze, F65) moves the image and converts nothing (§3.6, §3.9); the causal-mask attack (F72)
craters (§3.7); and the input-fidelity levers — denser frames (F67), readout-layer/prompt variants (F70),
learned audio (F64) — each add nothing. **Two** of the round-6 nulls, however, sharpen the
*optimization-landscape* picture of the tiny retrieval head itself and are worth recording as mechanism, not
merely as kills.

**The head's gradient geometry is non-standard (F69).** A validation-free checkpoint selector that picks the
epoch minimising the head-gradient norm (arXiv 2601.16874) rests on that norm being *negatively* correlated
with accuracy — the paper reports Spearman ρ ≈ −0.85…−0.98, so argmin(‖g‖) lands on a high-accuracy epoch. On
our head the correlation **inverts**: Spearman(‖g‖, dev-acc) = **+0.61 / +0.72 / +0.62** across seeds, the
scale-normalised gradient rising *monotonically with* accuracy (≈ 0.003 → 0.010) as the tiny head specialises,
so an unrestricted argmin lands at the worst (earliest) epoch and the tail-window "pass" is a left-edge
boundary artefact. The generalisation mechanism the method assumes for large vision heads simply does not hold
for a 12-tensor head over frozen features — a concrete caution against importing flat-minima *selection* to a
small retrieval head [DOC:GRADNORM_SELECT_PROBE_RECORD.md, commit `ada5849`].

**Flat-minima *training* does not help the head either (F73).** Consistent with F69, a SAM optimiser (ρ = 0.05)
on the same head is a KS-arm-dead null on both datasets and both protocols: a material regression on the
marginal MHC-ZH leg (val-sel −0.0246 / final −0.0424 acc, 0/3 sign) and only a within-noise +0.0047 nudge on
the near-ceiling HateMM leg (not 3/3-signed — its best single seed, 0.8884 val-sel, is the highest single
HateMM value observed anywhere, but the mean sits far below the bar and is **never** claimed). Modality-dropout
on the head regresses text-carried ZH and HateMM exactly as F45/F58 predict. The two remaining
head-training-dynamics escape hatches are closed at < 0.15 GPU-h, both disclosed headwinds borne out
[DOC:HEADRECIPE_VERDICT_REVIEW.md, commit `8e60f42`].

The audit then continued through three further round-6 waves (litsweep2 batch-3, litsweep-3 batch-4,
litsweep-5), each re-opening a prose-argued gap and measuring or $0-recon-pricing it dead; none moved a
law or the project's best numbers, and one of them closed the last hatch the arithmetic of §3.6 had left
formally open.

**The trained-reshaping escape hatch is now measured shut (F75).** §3.6's arithmetic proved the frozen-feature
*symmetric* operator can reach only the tiny legal slice, but it left one hatch open — a loss that *reshapes
the embedding by training* rather than operating over fixed features, and specifically one that directly
optimises the deployed top-20 kNN vote (F66 measured non-selecting operators over *frozen* features, not a
trained reshaper). A one-bite family that swaps the head's triplet+BCE for a vote-consistent soft-kNN (NCA,
τ ∈ {0.1, 0.2}), a neighbourhood-SupCon, or a manifold-mixup-BCE objective — the four operators that most
directly train *toward* the deployed vote — went **0/8 on the FORMAL bar and 7/8 KS-arm-dead** at ~0.33 GPU-h
(job 13482): no arm×dataset cell cleared +0.030/+0.030 3/3 both protocols, the family-max mean was A3-mixup ZH
final +0.0134 (2/3 sign), and the sole KS survivor (NCA τ=0.1 × ZH val-sel, +0.0112 acc / +0.0113 mF1, clean
3/3 sign) sits *below* the ±0.014 head-seed band — a measured-not-promoted within-noise hardening, still
D7-dead. This is the **first measured negative for "trained reshaping unlocks the oracle headroom"**: the one
training objective built to convert the selection-locked pool converts none of it on either dataset, so Law I
now holds against a *trained* operator, not only frozen ones [DOC:NCA_VERDICT_REVIEW.md, commit `f03cae0`]. (A
process note worth the record: the mandatory external code gate caught, and a surgical re-freeze fixed, an
A3-only confound — the manifold-mixup BCE forward inheriting the mining call's leaked eval-mode, which would
have disabled the head's dropout and conflated the mixup delta with a dropout-off delta — *before* any GPU
spend; the fix is an 18-line, Dropout-submodule-scoped restore, floor-path bit-exact
[DOC:NCA_REFREEZE_FIX.md, commit `8f08e9f`; DOC:NCA_REFREEZE_REVIEW.md, commit `467a6f4`].)

This closure was reached under an independent enumeration of the *entire training-data-centric family*
(litsweep-3), which found every sub-axis — feature-space augmentation, noise-robust head training, memory-bank
curation, class-imbalance/mF1 operators, SFT-example selection — pre-priced dead, dominated, or banned by the
same three walls: F66's arithmetic (91–98 % of the ZH/EN oracle headroom is per-item-selection-only, legal
slice +0.001–0.006), the EN label-limit, and the fact that the ZH miss is a *dev-selection* failure a train-side
operator cannot touch. The deepest obstacle is a data-generating-process one, not a capacity one: the retrieval
head **memorises its own training bank** (CLIP leave-one-out train accuracy 0.998), so any selector or reshaper
trained on the train split sees a degenerate, base-rate-inverted target (train-disagreement "Qwen correct" =
0/109, 0/102, 0/92) — the mechanism that killed the F47 router (§3.8) generalises to the whole
train-supervised-conversion family [DOC:LITSWEEP3_DATA_CENTRIC.md, commit `8629188`;
DOC:LITSWEEP3_SELECTOR_CONVERSION.md, commit `e103d54`].

**Wall-C is a protocol-shape, not a representation gap (F79 quantified).** The data-centric door-closers that
were run to recon rather than to verdict all PARK, and one measured the shape of the ZH/HateMM ceiling
precisely enough to record as mechanism. On both deployed floors, **test accuracy peaks late**: HateMM
test-optimal epochs are **18 / 21 / 24** (0.8884 / 0.8930 / 0.8884), +4 / +7 / +14 epochs *after* the
78/107-item dev saturates, so the val-selected and final checkpoints sit below a late test peak the dev set
cannot see; on ZH the **final epoch beats the val-selected checkpoint by exactly +0.0134 on all three seeds**.
This is the same late-climb shape that killed single-trajectory SWA (F62), and it makes every "stop trusting the
noisy tail" operator — noise-robust regularisation, early-target pulls, weight-averaging — *anti-aligned* with
the data by construction. It is also why the ZH headline is a protocol decision, not an operator problem: the
residual ZH gap is 78-dev selection variance over a late-climbing test curve, not an encoder or head deficiency
[DOC:ELR_FORENSIC_RECON.md, commit `9e41447`; DOC:CURATION_FORENSIC_RECON.md, commit `7025391`]. The ELR
noise-robust probe is further undercut at the mechanism level — the head's FAISS-mined pairs are
*gold-label-filtered*, so "mined-pair noise" is definitionally the gold-video-label noise the ZH-validated
consensus-denoising pillar already addresses, and the regulariser attaches to the BCE leg the deployed kNN vote
does not read. (The curation recon adds a machinery caveat that also matters for pillar-4: the deployed vote
indexes the *trained head embedding*, whose six floor checkpoints are disk-deleted, so a faithful multi-seed
bank-curation pregate is not the $0 operation it was costed as — it needs a ~0.3 GPU-h head re-mint, and the
only $0 object, the raw fused key, is seed-independent and thus single-draw, the withdrawn archive-as-key
failure class.)

**The extraction-instruction language is not the ZH lever either (F80).** A faithful Chinese re-extraction of
the deployed English instruction/scaffolding — the one un-varied axis on the ZH path, and the natural test of
the hypothesis that the ZH-LoRA (SFT-trained under a Chinese instruction) suffers a train/inference language
mismatch when read with English prompts — is **KS-dead on both arms and both protocols** at ~1.1 GPU-h
(job 13487, KS-parity bit-exact): the LoRA arm regresses −0.0358 val-sel / −0.0112 final acc and the frozen arm
−0.0336 / −0.0045, both val-sel legs past the −0.014 degrade line (the Chinese prompt *hurts* under
val-selection). The mismatch hypothesis is thus *refuted*, not merely unconfirmed — the LoRA and frozen arms
regress by near-identical margins, so English prompting is not a ZH liability; a native-bilingual encoder reads
the Chinese body regardless of instruction language [DOC:ZHPROMPT_VERDICT_REVIEW.md, commit `1a8c5fe`].

**The discarded label granularity does not sharpen the deployed boundary (F82, wave 5).** MultiHateClip ships a
3-class {Normal, Offensive, Hateful} annotation that the deployed task collapses to {Offensive, Hateful} = 1; a
natural EN-revival longshot gives the merged-in Offensive class a *softer* positive target in head training. A
$0 pre-gate closes it before any GPU: because retrieval on the fused key is label-independent, the vote is
exactly *linear* in the Offensive weight, so the τ grid and the gold-cheat oracle are exact, not sampled.
Offensive is the *majority* of the positive class (EN 73 % / ZH 63 %), so down-weighting it drags true positives
toward Normal — the honest proxy is monotone-negative on both datasets at every τ (ZH loo τ=0.25 = −0.1538), and
the fully gold-cheating oracle ceiling, with dev labels choosing *both* the weight and the threshold, reaches
only **EN +0.0250 / ZH +0.0256, both below the +0.030 bar**, with no arm's observed Δ exceeding its F63
permutation null (down-weighting the *true* Offensive set is no better than a random equal-size positive subset).
This is the §3.6 within-positive label-limit made arithmetic on the label axis: the 3-class split refines the
Hateful-vs-Offensive structure the binary task merges, not the harmful-vs-Normal boundary it is scored on
[DOC:GRADEDLBL_PREGATE_RECORD.md, commit `c4333ce`]. (This is also the *only* released finer label granularity:
MultiHateClip publishes an aggregated majority-vote label with no per-annotator votes, so the
learning-with-disagreement / soft-label-from-annotators lineage of the related work is foreclosed at the data
level — see limitations [DOC:LITSWEEP5_HATEMM_EN.md, commit `36d833e`].)

**The fusion operator itself is a measured null (F83/F85, round 7).** §3.6's mechanism turns on the deployed head
fusing the two L2-normed projections by an element-wise (Hadamard) product — yet `fusion_mode` has always been a
first-class constructor argument with `concat` and `cross` branches already wired
(`src/model/classifier.py:85–90, 138–143`), and **`align`/Hadamard is the only fusion this project ever ran on
video**. A zero-GPU recon confirmed that both standing bans over-reach on the letter: F50 names *fixed*
compositions, reweights and per-modality temperatures, and F75 names head-*loss* swaps, so a **trained** fusion
operator — optimised end-to-end under the unchanged triplet+BCE hybrid, with the deployed top-20 kNN read-out
untouched — is banned by neither, even though F50's conversion thesis, F75's trained-symmetric-reshaper mechanism
and F66's arithmetic all predict ≈ 0 and the head-side base rate stood at 0-for-~20
[DOC:FUSIONSWAP_FORENSIC_RECON.md, commit `934bc9a`]. Because the `concat` arm is a one-token, **zero-code-diff**
swap at ~0.1 GPU-h, the gap was closed by measurement rather than by argument. A one-bite six-run family
(job 13514; branch-assert 6/6, three independent parsers agreeing on all 48 values) leaves **both dataset cells
KS-arm-dead**: MHClip-ZH is +0.0067 acc val-selected (2/3 sign) but **−0.0045 final-epoch** (1/3), HateMM is
**−0.0031 on both protocols** (0/3 sign on every leg), no cell is anywhere near the FORMAL +0.030/+0.030
conjunct, and no KS-regression fires (worst mean −0.0045, above the −0.014 line). The effect sizes read most
honestly in test items: HateMM's entire effect is **≤ 2 flipped predictions on any seed** (n = 215, one flip =
0.00465; three of its six Δacc values are exactly 0.0000, the other three being −1, −1 and −2 items) and the ZH
val-selected mean is **+1 item per seed**
(n = 149, one flip = 0.00671). The fusion-operator axis is therefore closed as a measured null, and the live
reviewer question — *why Hadamard and not concat?* — is answered with a number instead of a preference
[DOC:FUSIONCAT_VERDICT_REVIEW.md, commit `129fe2e`]. Two disciplines travel with it. First, the arm that failed
was `concat` **bundled with a 2.0× first-Linear capacity bump** (2,098,176 vs 1,049,600 parameters), so the null
is evidence about the *bundle* and may never be upgraded into "extra head capacity cannot help" — the
param-matched control that would separate them was placed outside the family by the frozen scope. Second, the
val-selected legs were again decided by `roc` tie-breaks over up to six Val-tied epochs (selected epochs ranged
5 → 26): the F45/F63 78-item dev wall, visible directly in the selection trace.

**A measurement-discipline lesson: the merged and unmerged adapter paths are not a null channel (F87).** The
round-7 transplant cell — modality-routed LoRA (a per-modality down-projection `A` with a shared `B`, `r_v =
r_t = 16`) inside the deployed ZH encoder-SFT, i.e. the **PEFT-adapter-structure axis no banked adapter in this
campaign had ever varied** — is a measured null: **final-epoch: fail; val-selected: fail** against both floors,
landing at **+0.0000 acc** on both protocols against the banked generic-LoRA floor 13150, at 5.573 GPU-h. What
makes it worth a mechanism paragraph rather than a table row is *how* it failed. A pre-registered drift gate
fired on all six (split × stream) cells of the banked cache (worst mean per-item cosine **0.99954879** against a
0.9999 bar), which made a **same-path unmerged floor mandatory** and switched the binding pairing to it. Against
that same-path floor the arm reads **+0.0268 val-selected acc, 3/3 seeds** — the family's only above-noise
number. It is not a result: with routing **entirely absent**, the unmerged path loses **−0.0268 acc / −0.0340
macro-F1 (0/3)** against the merged path, and the three seed-paired comparisons are one identity,
**+0.0000 = +0.0268 + (−0.0268)**. The manipulation that produced the −0.0268 is the *same* banked adapter, the
same frames, prompts and head code, differing only in `merge_and_unload()`'s folded `W+BA` single matmul versus
the unmerged `Wx + B(Ax)` — a **bf16 accumulation-order** difference, method-null by construction — and it is
carried almost entirely by one seed at **−0.0604 = −9 of 149 test items**, whose val-selection collapsed to
epoch 5 (two epochs tied at Val acc 0.8718; the `roc` tie-break took the earlier). The no-selection final-epoch
protocol shows the same manipulation at only **−0.0067**, one test item. The transferable lesson is a rule about
channels, not about routing: **a measurement channel whose sensitivity to a demonstrably method-null manipulation
exceeds both the ±0.014 house seed band and the effect it is being asked to certify carries no discriminating
power about the manipulated variable** — so a same-path floor is the *default cost* of any adapter-structure
comparison, not a contingency, and the drift is not even symmetric across streams (the text stream drifts ≈ 3×
further than the image, means ≈ 0.99955 vs ≈ 0.99985 — the more drifted stream is the one both measured passes
ride on) [DOC:MOKA_VERDICT_REVIEW.md, commit `91f64a6`; DOC:MOKA_SUBMIT_RECORD.md, commit `ed609eb`]. Three
further readings are fixed by the frozen clauses and must travel with any citation of this cell. (i) The
stream-level decomposition is a **null-op**: the text stream is FLAT under both floors (Δ train-LOO AUC −0.0007 /
+0.0018), which **refutes the prereg's own text-side bet** that an undiluted `A_t` would sharpen the dominant
stream. (ii) The image stream is **AMBIGUOUS, not MOVED** (train-LOO +0.0137 but dev −0.0121 under the binding
floor; +0.0120 / +0.0043 under the merged floor, missing the +0.005 dev leg by 0.0007), so although the head is
flat, **this is not the ninth instance of law I — the count in §3.6 stays at eight** and the cell is recorded as
law-I-*shaped* but not law-I-*certified*; MokA's advertised visual-modality-protection narration is barred
outright. (iii) The most economical explanation of the null is a **regime inversion**, priced before the run: our
SFT records are **94.6 % vision tokens** (median 2,688 vision + 153 text) against MokA's own shipped regime of
**98.4 % text** (16,128 vs 256), so routing gives the text stream its own undiluted down-projection while
*starving* it — from 100 % of positions to ≈ 5.4 %, ≈ 18× fewer token-gradients — and `A_v`'s gradient norm ran
25–40× *below* `A_t`'s despite vision dominating the token count. A standing caveat binds all of it: there was
**one** SFT draw and `--seed` varied only the head, so encoder-draw noise is not separable from the routing
effect within this budget (limitations §3). (Process value, alongside wave 3's dropout-mode catch: the mandatory
external code gate blocked this family pre-spend on two P1 defects — a modality-mask hook registered on
`get_base_model()` that **never fires** on the production `PeftModelForCausalLM`, because that call chain reaches
the base model through a direct `.forward()` while `nn.Module` hooks fire only in `__call__`, and a `median`
computed as `vals[len(vals)//2]` over 196 layers, an upper-neighbour order statistic. After the fix and a
re-freeze, routing was runtime-verified live — `hook_calls` 314, `routed_calls` 77,224, **`fallback_calls` 0** —
so the null is functional, not mechanical [DOC:MOKA_REFREEZE_FIX.md, commit `72a947b`].)

### 3.11 The information structure of the deployed pair — no synergy to fuse, and no unique image information (F86)

The audit's last mechanism result is a **measurement rather than a lever**, and it is the one that converts a
list of fusion nulls into an arithmetic statement. Using a sample-level **partial-information decomposition**
(PID) as implemented by the **LSMI** estimator \cite{lsmi} (ICML 2025), the
task-relevant information the two deployed streams carry about the label is decomposed, per sample, into
**redundancy `R`**, **per-stream uniqueness `U1` (image) / `U2` (text)** and **synergy `S`**, on the banked
train/dev caches of all three deployed lineages (MHClip-ZH generic-LoRA 13150, HateMM curric-LoRA 13241,
MHClip-EN frozen Qwen). The gate is CPU-only, reads no test split, and produced **zero GPU-hours**. Its stake is
stated before its numbers: a fusion block can only recombine `R`, `U1` and `U2`, so if `S ≈ 0` then *every*
richer fusion operator — attention, gating, bilinear, concat-versus-Hadamard — is mechanistically capped, and the
"the method is too crude" objection is answered by a measurement instead of an opinion.

**The machinery had to be certified before it could be believed, and at the released settings it fails.** A
pre-declared XOR positive control at our own sample sizes — a label that is a *deterministic* function of the
pair, with zero unimodal information — is read at **chance** by the joint discriminator at the released
projection dimension (out-of-fold accuracy **0.513 / 0.530 / 0.508** at `d' = 64`, n = 579 / 744 / 549), so the
pre-declared power gate **fails** and the whole `d' = 64/256` layer is declared measurement-invalid and carries
no evidential weight. Walking the dimension down localises the wall: joint out-of-fold accuracy on the same known
synergy runs **0.998 → 0.903 → 0.632 → 0.513** at `d' = 8 → 16 → 32 → 64`, so power is *monotone decreasing in
dimension* and **`d* = 16`** is the largest certified dimension (replicated at `d' = 8`, where the estimator
recovers the maximal synergy of `log 2 = 0.6931` nats as **0.7077 / 0.7321 / 0.7105** — a ≈ 2 % error at
n ≈ 600). Specificity certifies with it: a duplicate-stream control (`x2 := x1`, ground-truth `S = 0`) returns
**exactly 0.0000** at both certified dimensions against **+0.0838 / +0.1516 / +0.2240** for the same control at
the uncertified `d' = 64`.

**Read at the certified dimensions, the shape is the same on all three datasets.** Total task-relevant
information `I12` is a real, well-estimated **0.149–0.359 nats**; redundancy `R` is **0.069–0.178**; **text
uniqueness `U2` is the largest atom on 5 of 6 certified cells (0.076–0.237)**; **image uniqueness `U1` is pinned
at exactly 0.0000 on 5 of 6 cells** (range −0.084–0.000); and **synergy `S` is −0.0747 (ZH) / −0.0802 (HateMM) /
−0.0000 (EN)** at `d* = 16`, ≤ 0 on 5 of 6 certified cells, with the largest positive reading anywhere in the
certified layer being **+0.0031 on ZH at `d' = 8` — 0.9 % of that cell's `I12`**. A fresh 50-draw permutation
null gives `q95 = 0` on every certified cross-fitted cell, so the false-synergy floor is zero to within 4 × 10⁻¹⁷,
and the held-out dev read replicates (−0.0004 / −0.0575 / −0.1041). The mechanical verdict is reported
unrounded: **INDETERMINATE** at `d* = 16` (per-dataset ZH / HateMM INDETERMINATE, MHClip-EN FUSION_CAPPED) — but
only because the pre-declared clause *conjoined* "no synergy" with "redundancy-dominated". The synergy half fired
on **3/3 datasets × 2 certified dimensions, plus dev**; the dominance half failed because the pair turns out to
be **uniqueness-dominated on the text side**, not redundancy-dominated — the clause was written for the wrong
dominant atom, not left inconclusive about synergy [DOC:LSMI_GATE_RECORD.md, commit `a8905ac`; pre-declaration
chain `d4b06f0` → `362a60e`, every threshold committed before the numbers it governs].

**The supported sentence, and what it explains.** Within the certified subspace, essentially all task-relevant
information in the deployed pair is carried by the **text** stream — as text-unique information plus a smaller
redundant component shared with the image — the **image stream contributes no unique information**, and the two
streams contribute **no synergy**. That is the mechanism under §3.6's F44 (the MHClip-EN image stream collapsing:
here `U1 = 0` on EN, whose `I12` is also the smallest of the three at 0.149 nats) and under F50 (a fixed
composition is "a rotation at every mixing weight" — with `S = 0` there is nothing off the `R`/`U1`/`U2` simplex
for any operator to reach), and it is what the F85 concat null looks like from the information side: `concat` has
strictly more capacity than `align` to exploit `U1` and `U2`, and `U1` is measured at zero. The two results are
*consistent*, not derived from one another — the gate explicitly refuses to predict the fusion family's numbers,
and F85's verdict stands on its own measurement.

**Walls, stated as the gate stated them — before its numbers.** (i) It **cannot bound trained reshaping**:
everything here is a property of the banked features *as they are*, and says nothing about whether a differently
trained encoder would produce a differently structured pair — the same F66-style distinction that keeps the
adaptation axis (§3.9) alive. (ii) It **cannot price a third stream**: audio, OCR and frame-level tokens are
outside the decomposed object. (iii) It is a **train/dev measurement** with no held-out claim; the discriminator
accuracies quoted are estimator diagnostics, not results. (iv) A null at `d*` bounds synergy **inside the
retained principal subspace** (per-stream retained variance: `d' = 16` image 0.668–0.739 / text 0.528–0.580;
`d' = 8` image 0.523–0.629 / text 0.404–0.466), which is the honest price of the only regime where the estimator
demonstrably works at n ≈ 600. (v) **PID is axiom-dependent**: redundancy is defined differently across the
Williams–Beer, Bertschinger et al., Ince and Griffith–Koch families, and pointwise decompositions additionally
admit negative atoms, so everything reported is *LSMI's* decomposition under *LSMI's* min-rule.

**A methods note the paper should carry, because it nearly produced a false result.** Two properties of the
released estimator matter to anyone reproducing this. Its entropy-estimator loop **never calls
`optimizer.zero_grad()`**, so kernel gradients accumulate across the whole run; that defect is invisible in the
authors' own 2-dimensional demo (identical to 4 dp) but not at our dimensions — holding the discriminators
bit-identical, the fixed and as-shipped loops move the entropy estimates (ZH `H1` 1130.19 → 710.62) and **flip
the sign of `S` on 2 of 3 datasets** (ZH +0.2345 → −0.0672; HateMM −0.1517 → +0.1152). More consequential, the
**released in-sample reading protocol saturates at our sample size** — all three discriminators reach 0.99–1.00
accuracy and all three pointwise mutual informations collapse to ≈ `log 2` — and then reports "redundancy-
dominated, `S ≈ 0.02`, share ≈ 0.03" *identically* for the real pair **and** for the duplicate-stream and
split-half controls whose ground truths are different. Run as shipped, the code would have handed us a clean,
quotable "no synergy, redundancy-dominated" paper sentence that its own truth-known controls contradict; the
cross-fitted read that caught it was declared *after the synthetic controls fired but before any cell on our
data*, which is the only reason the readout stayed pre-registered. The conclusion we do bank is the *narrower*
one the certified arm supports — no synergy, uniqueness-dominated, text-side — and the discarded one is recorded
as the near-miss.

**Consequence for the transplant queue.** The survey's first *executed* item is discharged by this measurement,
and the second dies by its own hand: the SynIB port had pre-declared "the LSMI reading" as its kill-switch, and the
branch that fired (`s ≈ 0` on all datasets) prescribes **PARK at $0** — an objective built to push a head onto
*synergistic* structure has no structure to push onto here — while the conditional BalanceBenchmark screen never
unlocks, being conditional on synergy existing to balance [DOC:SYNIB_PORT_FORENSIC_RECON.md, commit `9e638ea`;
DOC:REPRO_SURVEY_2025.md, commit `9367338`]. Where the numbers *do* point is the adaptation side — `U1 = 0.0000`
with `U2` dominant is a modality-imbalance statement — which is exactly the object the round-7 routed-LoRA cell
went on to measure, and to null (§3.10).

## 4. What survives

Three MLLM roles survive with removable-ablation evidence; none is a main-table-accuracy role.

**Encoder.** Qwen2.5-VL features beat CLIP \cite{clip} on HateMM by +4.2 accuracy (+4.4 macro-F1) and
cross the 0.85 threshold (frozen-Qwen 0.870 / 0.861 vs the CLIP floor) [DOC:PAPER_MASTER_TABLES.md T1.1]. Removing
the MLLM here means reverting to CLIP and losing the crossing — a genuine cost — but this is the
frozen-encoder identity, not the new method role the mandate sought. The swap's HateMM-specificity is
**no longer an unexplained anomaly**: §3.6 shows it converts precisely when hate is visually grounded
*and* the residual errors are representation-limited, and merely rotates the ranking otherwise. The
role is therefore earned *and* mechanistically bounded — it will not generalise to MHC, whose visual
channel is uninformative to the VLM (image AUC collapses 0.734 → 0.599) and whose error core is
label-limited [DOC:ENCODER_SWAP_DIAGNOSIS.md, commit `8a48938`].

**Localization scorer.** The per-window scorer is a removable component: delete it and localization
falls to the memory read-out (0.5140) or random (0.5088); keep it and P6/P10-b rank hate windows
significantly better (0.5435 → 0.5755, paired-significant vs both) [DOC:EXP_p6_mllm_localization.md,
DOC:PAPER_MASTER_TABLES.md T2.1]. Two honest caveats travel with it: the magnitude is modest-plus
(≈7.5 AUC points over chance) and the MLLM's dominant capability is actually video-level toxicity
density (broadcast AP 0.62), the within-window increment being smaller though statistically stable;
and its advantage is established **against the retrieval memory**, not against a video-label MIL head
(§3.5), so the paper's contrast for this role is memory, not MIL — a distinction that leaves the
zero-shot memory-swap capability untouched, since MIL needs target-domain video labels the swap does
not.

**Guard-rail / audit.** The auditable archive memory supports a *veto*. The two-vote AND rule for
automatic noise repair does **not** reproduce the human 2-entry gain (C−A = +0.0000, 0/4 EN seeds),
because the AND rule structurally cannot reach memories that are semantically contradictory yet not
embedding outliers; but the semantic vote **vetoes** the embedding-only rule's over-deletion of
genuinely-hateful-but-embedding-hard entries (abuse testimony, assault reporting, slur-bearing text),
worth C−D = +0.47pt EN / +0.40pt ZH [DOC:EXP_auto_memory_repair.md]. Separately, a human deleting two
flagged noisy entries lifts EN from 0.8075 to 0.8199 with **zero retraining**
[DOC:DEMO_memory_editing.md], and the label-blind archive audit independently re-finds those same
human-flagged ids with correct reasons. Here removal cost surfaces as **integrity and
controllability**, not raw accuracy — a defensible framing precisely because we do not dress it as an
accuracy claim.

**Explicit non-role.** For completeness and falsifiability: the MLLM earns **no main-table-accuracy
role** in this retrieval-memory pipeline. Eleven pre-registered routes at 7B–72B scale are all honest
kills or within-noise, each guard-backed [DOC:PAPER_MASTER_TABLES.md T4]. We also record the
quantified headroom that this rules out approaches to, not the prize itself: an oracle membership
editor (delete by *true* label) lifts the gated slice to 100% and overall accuracy +7.5pt EN /
+10.6pt ZH, both across 0.85 — the prize is real, and P2b shows a stronger comparability judge is not
the key that unlocks it [DOC:EXP_p2_neighbor_rerank.md, DOC:EXP_p2b_stronger_judge.md].

## 5. Implications

**For practitioners.** Reach for an MLLM where its semantic output *is* the evaluated quantity, and
be sceptical where it is merely correlated with it. As an encoder it pays off precisely — and only —
where hate is visually grounded *and* the residual errors are representation-limited (the HateMM
crossing); the same swap merely rotates the ranking on MHC, whose visual channel is uninformative to
the VLM and whose error core is label-limited, so no encoder upgrade converts there and scale regresses
rather than rescues (§3.6) [DOC:ENCODER_SWAP_DIAGNOSIS.md]. As a localization scorer it pays off because
per-window saliency is the target itself, and there — and only there — **scale converts** (monotone
A-fuse 7B → 72B). It does **not**
pay off as a main-table component in a retrieval-memory detector whose decision boundary is already
directly supervised: comparability, priors, distilled schema fields, sanitized counterfactuals, and a
late-fusion semantic channel each turn out orthogonal to or redundant with that boundary. The
scale-versus-calibration finding (§3.2) is the most actionable: a bigger judge buys
well-behavedness, not selectivity, so budgeting a larger (or closed) model to rescue a reranking or
prior-estimation line is, on this evidence, a poor bet — whereas budgeting it for the localization
lane has a monotone gradient to extrapolate from.

**For the field.** The credibility of a negative result is manufactured by protocol, and three of our
practices generalize. First, **a passing no-head probe is necessary but not sufficient** (§3.3): the
cleanest and the strongest probes of the campaign both trained flat because a learned fusion head
absorbs input-space advantages — probe results must be confirmed under a final-epoch protocol, not
just validation-selected. Second, on ~150-video test sets **noise-floor honesty is non-negotiable**:
1 accuracy point ≈ 1.6 videos, sub-point effects are within-noise, and a 78-sample dev set makes
validation selection itself cost ≈2 accuracy points, so single-seed, single-protocol "gains" of a
point or two are not gains. Third, **pre-registration with a single test touch** (§2) is what lets a
sequence of eleven comparisons close a question rather than reopen it: the bar was fixed before the
data, the localization test was spent once, and the confounded-but-passing literal gate in P11 was
killed rather than promoted. Whether the localization lane can be pushed from modest (0.5755) to
substantial (≥0.60) with a closed-weights scorer, without violating reproducibility or data-egress
constraints, is the one open question this campaign leaves precisely posed.
