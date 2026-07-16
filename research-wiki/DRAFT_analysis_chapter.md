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
**thirteen-route pre-registered campaign** [DOC:CAMPAIGN_mllm_method_role.md], later extended by two
further pre-registered sprints (rounds 2–3, §3.6–3.7) that closed every remaining injection point in
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
result trustworthy (§2), the five mechanisms that explain it and the two structural laws that rounds
2–3 crystallised them into (§3), what survives with ablation evidence (§4), and the implications (§5).

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

## 3. Five mechanisms — and two structural laws (rounds 2–3)

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

### 3.6 Structural law I — better signal without conversion (four instances; F44 the mechanism)

Beyond the thirteen-route campaign, two further pre-registered sprints (rounds 2 and 3,
[DOC:TERMINUS_round2_mllm_plus3.md, DOC:TERMINUS_round3_mllm_plus3.md]) hardened §3.1 and §3.3 into a
single law that now has **four independent instances**: a candidate signal is demonstrably *richer*
than the pipeline already has, and yet the best in-constraint operator converts **none** of it into
main-table accuracy. The first three share a sharp form — a **gold/label oracle proves the convertible
headroom is present**, but no unsupervised, frozen, or even supervised operator inside the constraint
box recovers it:

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

The **fourth instance is the encoder swap itself**, and it is the one that turns the campaign's central
positive result from an anomaly into a mechanism. A zero-GPU geometry diagnosis on banked train/dev
caches shows Qwen's representation upgrade is **real and roughly equal on all three datasets** — top-20
neighbourhood purity rises **+0.023 / +0.023 / +0.021** and the text-stream AUC rises **+0.041 / +0.054
/ +0.045** on HateMM / MHC-EN / MHC-ZH — and yet it converts to accuracy on **HateMM only**
[DOC:ENCODER_SWAP_DIAGNOSIS.md, commit `8a48938`]. The mechanism has two legs, both *dataset*
properties rather than method-fixable ones:

1. **Modality-locus × equal-weight fusion.** HateMM's hate is visually grounded (image-only train-LOO
   AUC 0.826), so Qwen's uniformly better text stream rides on a neutral-strong image stream and the
   fused gain is a clean Pareto move (hate-recall +0.116 (dev) at **zero** non-hate cost). On MHC-EN the Qwen
   **image stream collapses to near-chance (0.734 → 0.599)**, and because the head fuses image and text
   as equal-weight L2-normed blocks, that collapse **cancels** the +0.054 text gain (net dev −0.012).
   The collapse persists at 32B (image AUC 0.608), which is exactly why *scale regresses* rather than
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
easy-example ordering), and B2 (scale regresses on MHC) — and, read alongside P3 / S2S / W2-A, states
the law in its general form: **a signal being measurably better is necessary but not sufficient for a
main-table gain; what decides conversion is where the gain lands — which modality, which error type —
and whether the decision metric can absorb it, not how much better the signal is.** The design-time
corollary sharpens §3.3's dual-protocol rule into a question to ask of any auxiliary-signal proposal:
not "is the signal richer?" (it usually is) but "is its advantage in the modality and the error type
the decision boundary is actually limited by?"

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

The contribution generalises beyond hateful video: **anyone building set-matching or temporal-order
retrieval over decoder-VLM token summaries is operating on prefix summaries, not frame states, so the
"extra" temporal structure they hope to exploit is already integrated into each token** — a concrete,
transferable caution, established structurally and then confirmed at both the unsupervised and the
supervised operator level. (The same causal cumulation is why W2-A's grounded key was architecturally
real yet redundant in §3.6: the joint forward integrates the transcript into every vision token.)

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
