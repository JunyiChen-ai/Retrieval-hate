# WAVE-4 CANDIDATE RECON — adversarial re-enumeration after round-4 line-B KILL (F47)

**Agent:** wave-4 ideation/recon. **Date:** 2026-07-17. **ZERO GPU / ZERO test-touch / ZERO user interaction.**
Reading + forensic reasoning + prereg-shaped design only. Deliverable = this committed doc.
**Frame at time of writing:** round-4 line B (per-item cross-channel router) landed **KILL** (F47, `30d0ee1`) — per-item
channel-selection now closed at all three supervision sources; round-4 line A (encoder-level LoRA-HateMM) is **frozen and
in single-submit ceremony** (`LORA_HATEMM_PREREG`, `8de0991`; encoder-class ⇒ D7-dead for novelty regardless of outcome).
**Governing docs read this recon (verbatim):** `state/directions_tried.json` (22 dead + 9 bans + positives bank),
`findings.jsonl` F35–F47, `TERMINUS_round3_mllm_plus3.md` (9-axis map), `ROUTER_GATE_RECORD.md` (F47),
`ENCODER_SWAP_DIAGNOSIS.md` (F44), `B3_ZH_LORA_DECOMPOSITION.md` (F45), `LORA_HATEMM_FORENSIC_RECON.md` (line A),
`WAVE3_CANDIDATES.md` (non-isomorphism target + house style), and **`src/model/classifier.py`** (the deployed head — see §0.3).

---

## BOTTOM LINE UP FRONT

1. **Inside the frozen constraint box the pool is empty of goal-hitting candidates.** Angles (a)–(e) were worked honestly
   (§1). Every genuinely-untested lever is either (i) encoder-class (Axis B, D7-novelty-dead), (ii) decision-side
   (Axis A / D1, 20+ dead + F47 per-item closure), or (iii) blocked by the **MHC-EN label-limited wall** (F44/SAV: MHC-EN
   errors are intrinsic, *no* representation lever converts them) — which alone forecloses the only realistic ≥2-dataset
   route that a novel non-encoder mechanism could still need.
2. **One banked finding is materially imprecise, and correcting it reopens exactly one $0-measurable cell.** The deployed
   head uses **`fusion_mode='align'` — an element-wise Hadamard product of two L2-normed projections**
   (`classifier.py:119-120`), **not** the "equal-weight L2-normed *concat* blocks" that F44's mechanism prose describes
   (§0.3). F44's *numbers* survive (its concat-kNN diagnostic is validated as a sign-faithful proxy), but its *dismissal*
   of a modality-reweighting/fusion lever — "the trained head already has attenuation capacity and still failed on test"
   (F44 §7.3) — rests on a concat premise the align head does **not** satisfy. That cell is therefore **unmeasured, not
   closed.** It is the single positive-EV $0 move left, and it is door-closing (house style), **not** a likely goal-hit.
3. **Three candidates survive to the ranked list — all LOW prior:** **FA** (fusion/composition $0 probe, closes the F44
   cell), **MJ** (MLLM modality-reliability judgment as a *new* router input — the one door F47 explicitly left ajar), and
   **CC** (cross-encoder composition as a deployed arch — a D7-*composition* ruling downstream of FA). **Recommended single
   next gate = FA** (§3): $0, decisive, converts F44's assertion into a measurement.
4. **If the loop wants a *goal-hitting* opening it must relax a user-set constraint.** §4 lists three *new* relaxations
   beyond the five terminus options, the strongest being a **D7 sub-ruling for cross-encoder composition** (narrower than
   the LoRA option-c).

---

## 0. The structural picture wave-4 must respect (and the one correction it adds)

### 0.1 Two walls account for the entire graveyard (unchanged, re-verified)
- **Oracle-exists-but-unconvertible — now FOUR instances (P3 · S2S F37 · W2-A F42 · router F47).** A gold/label oracle
  proves convertible headroom (S2S +0.09/+0.14; W2-A +0.0635/+0.0970; **router MHC-EN +0.1083 / HateMM +0.0498**) that the
  best in-constraint operator recovers ≈none of. F47 is the sharpest yet: it closed per-item channel-selection at
  **unsupervised** (K9 zeros), **train-supervised** (degenerate target — CLIP LOO train acc 0.998), **and dev-supervised**
  (dev-CV realizable ceiling −0.046, below perm-null) simultaneously.
- **Cumulative-causal closure (F35/F37/F39).** Qwen2.5-VL's causal LLM makes per-frame-group vectors prefix summaries;
  temporal/set/order operators are redundant with the pooled key at structural, unsupervised, and supervised levels.

### 0.2 The two decisive priors every wave-4 candidate inherits
- **MHC-EN is label-limited, not representation-limited** (F44 §4/§5, SAV #18). Qwen fixes 11 / breaks 12 dev videos =
  net −1; the swap is a **rotation** (+0.040 hate recall bought with −0.036 non-hate), the B5 unconvertible-edge signature.
  ⇒ **no representation lever clears +0.03 acc on MHC-EN in-constraint**, and F47 proved the rotation is not per-item
  routable either. The honest ≥2-dataset arithmetic is **HateMM (representation levers work) + {ZH via adaptation}** — and
  both legs are encoder-class (D7-dead). A *novel* mechanism therefore has to convert HateMM or ZH by a **non-encoder,
  non-decision-side** route, or it is off-goal.
- **Adaptation converts; identity re-ranks (F45).** Frozen swaps re-rank (AUC up, acc flat = rotation); LoRA re-decides
  (Pareto). The convertibility line runs through **adaptation of the representation** — and every adaptation of the
  *encoder* is Axis-B/D7-dead, while every adaptation of the *decision* is Axis-A/D1-dead. Wave-4's job was to find an
  adaptation target that is neither. It could not (§1a).

### 0.3 CORRECTION to F44 (adversarial-to-ledger; load-bearing for candidate FA)
`src/model/classifier.py:70-129` (`classifier_hateClipper`) with the deployed `enc3s` config (`fusion_mode='align'`,
map_dim=proj_dim=1024, num_layers=3; `router_ckpt` §1) fuses the two modalities as:

```
img = normalize(img_proj(img_feats));  text = normalize(text_proj(text_feats))   # each L2-unit, map_dim=1024
x   = img ⊙ text                                                                  # align → ELEMENT-WISE product
```

This is a **parameter-free bilinear Hadamard interaction**, not a 50/50 concat. Consequences F44's prose gets subtly wrong:
- **A collapsed modality corrupts *multiplicatively*, not additively.** When Qwen's MHC-EN image stream collapses to
  near-chance (F44: AUC 0.599), a near-random unit vector multiplies *every* fused dimension — it does not merely occupy an
  ignorable 50% block. This makes F44's "cancels in the 50/50 concat" mechanism *directionally right but mis-stated*.
- **The head has *less* attenuation capacity than F44 claims.** In align mode the head cannot down-weight or zero a
  modality: a linear `img_proj` cannot map varying inputs to a constant (zero weights ⇒ `normalize(0)` NaN), so it cannot
  route around the collapsed factor. F44 §7.3's dismissal — "the trained head already has attenuation capacity and still
  failed on test" — therefore rests on a premise the **deployed align head does not satisfy.**

**What survives / what reopens.** F44's *numbers* stand: its diagnostic used a **concat-kNN** read-out that §1 validated as
reproducing the align-head's downstream sign+size on HateMM/MHC-EN, so the diagnosis is sound as a proxy. But F44's
*architectural dismissal* of a fusion/reweighting lever is **not** a measurement of that lever — it is an assertion against a
mis-described head. The cell "does a modality-reweighted or concat fusion of Qwen features convert on MHC-EN?" is **genuinely
unmeasured.** That is the entire basis for FA (§2.1). (This does not touch the terminus's *other* eight axes.)

---

## 1. Angles (a)–(e) — adjudicated honestly

| angle | question | verdict | why (specific ban / finding) |
|---|---|---|---|
| **(a)** | non-encoder adaptation target (adapt the *fusion* so text gains aren't cancelled)? | **near-dead → FA ($0 probe only)** | fusion-reweighting is Axis-A/B textbook (D7-dead) **and not MLLM-specific** (applies to any encoder). BUT F44's dismissal is on a false concat premise (§0.3) ⇒ one $0 measurement is owed. Encoder-adaptation = D7-dead; decision-adaptation = D1-dead; no third target exists. |
| **(b)** | mixed-encoder composition (CLIP-image ⊕ Qwen-text), fixed, no routing? | **→ CC (LOW, D7-composition ruling)** | F44 §7.2 pre-dismisses it 3 ways; two are contestable (its "zero-training reorg" tag mis-fits a *trained-head* composition; its "AUC-only, B5-unconvertible" is a prior not a measurement of *this* cell). One is solid: still encoder-class ⇒ **D7 ruling**, like LoRA option-c. Measurable for $0 via FA; ranked as FA's downstream. |
| **(c)** | cross-encoder memory (query one space, keys another, trained alignment)? | **DEAD** | collapses to Axis-B (a trained CLIP→Qwen alignment *is* a representation lever) OR to per-item cross-channel selection (F47-closed) OR to fixed vote-averaging (Axis-A fusion). No door: the neighbours' label information is encoder-invariant; only the retrieval geometry changes, which is Axis-B. |
| **(d)** | training-dynamics: does head memorization (CLIP LOO 0.998, F47) hurt the vote? early-stop / de-memorized memory? | **DEAD on goal** | (i) **not an MLLM lever** — cross-fit / out-of-fold memory embeddings apply identically to CLIP ⇒ fails the goal's "MLLM integrated" clause; (ii) early-stop = a *tactic* not structure (stall rule); (iii) memorization degrades the *train-side routing target* (F47) but the *deployed* out-of-sample vote is the validated SOTA-competitive floor — no evidence de-memorizing lifts it. |
| **(e)** | a genuinely NEW MLLM information source (not banked features/votes/frame-groups, not P1–P11/OCR/gold)? | **empty inside the box** | attention/head-mining = SAV #18 dead; reasoning-text = C3 dead ("info banked in Qwen pathway"); logits = P10 dead; predicted-attribute = C3-target dead; localization = P6 (earned role, no accuracy role); intermediate-layer/multi-prompt features = Axis-B; RAG-into-MLLM = decision-side D1-dead. The one residual is **audio via Qwen-Omni (E, parked, needs download)** and the one *meta-cognitive* judgment not yet asked (→ MJ, §2.2). |

**Angle (a) is the crux and it is negative for the goal.** F45 says adaptation converts; the only adaptation target that is
neither encoder (D7) nor decision (D1) would be the *fusion*, and (1) the align head can't reweight modalities so it is a
genuine architecture change, but (2) that change is generic (encoder-agnostic) fusion engineering, D7-dead, and does not
"integrate the MLLM novelly." It can only *measure* whether the cancelled Qwen-text gain is recoverable — a base-pipeline /
paper question, not a goal candidate. Hence FA is scoped as door-closing, not goal-hitting.

---

## 2. Surviving candidates (all LOW prior; full treatment)

### CANDIDATE FA — Fusion/composition $0 probe: is the cancelled Qwen-text gain convertible on MHC-EN? **[recommended gate]**

**(a) Mechanism / bandwidth / injection point.** The deployed align head multiplicatively corrupts Qwen's real
MHC-EN text gain (+0.054 AUC, F44 T3) with its collapsed image factor (AUC 0.599). FA measures, at $0 on banked caches,
whether **recovering that text gain via a different modality composition** converts to *accuracy* (Pareto) or only re-ranks
(rotation, B5-dead). Three arms over the RGCL kNN geometry (train-LOO, k=20 signed-cosine, the §0.3-validated proxy):
(1) **weighted fusion** — `z = normalize([√w·imĝ , √(1−w)·text̂])` swept over w (incl. w→0 = Qwen-text-only); (2) **cross-encoder
composition** — `concat(CLIP-imĝ , Qwen-text̂)` (the angle-(b) object); (3) **align vs concat** control on Qwen. Injection
point = fusion/key-construction; bandwidth = representation-level (D2), zero new features.

**(b) Non-isomorphism (verified against the cited records).**
- **vs F44 §7.2/§7.3 (the pre-dismissal):** F44 dismissed fusion-reweighting on a **concat + attenuation-capacity** premise
  that the deployed **align/Hadamard** head does not satisfy (§0.3), and priced the composition's outcome from B5 rather
  than measuring *this* cell (Qwen-text-heavy per-class recall at the label-oracle threshold). FA measures exactly the
  unmeasured quantity. **This is not re-litigation: it converts an assertion into a G0-cond measurement, house-style.**
- **vs W2-E / C3geo (frozen-reorg ban):** those ban **zero-training unsupervised reorganisation of frozen pooled features**;
  FA's composition feeds a **freshly-trained RGCL head** (the standard trained head), so it is outside the unsupervised-reorg
  meta-family. (The kNN read-out is a diagnostic proxy, not the deliverable operator.)
- **vs router F47 / B5:** FA is **not** a per-item selector (F47) and **not** a global operating-point threshold on one arm's
  score (B5) — it is a modality-composition change *inside* one arm, upstream of the vote.

**(c) Expected-effect prior — LOW, grounded in F44/F45 numbers.** On MHC-EN, w→0 approaches Qwen-text-only (AUC 0.851 >
CLIP-concat 0.801) so **AUC almost certainly lifts** — but B5 proved MHC/ZH AUC edges are easy-example ordering, unconvertible
at any threshold incl. the label-oracle cut, and F44 rules the MHC-EN core label-limited (rotation net −1). The genuinely
*unmeasured* sliver is whether Qwen-text-*heavy* fusion has a *different* per-class profile than the collapsed-concat rotation
F44 measured. Prior it is Pareto: **~10–20%** (against: B5 + label-limited + F47-rotation-not-routable; for: the specific cell
is unmeasured and align-corruption may have masked a text-only Pareto). On HateMM the composition ≈ Qwen-swap (CLIP-img 0.826 ≈
Qwen-img 0.817) ⇒ no new win. On ZH it echoes B5 (frozen text edge unconvertible; LoRA is the converter, F45).

**(d) $0 gate design (house-style bars).** Reuse `encoder_swap_geometry.py` + `encoder_swap_diagnosis_tables.py` machinery
(banked `data/CLIP_Embedding/MHC/{train,dev_seen}_{CLIP,Qwen}.pt`; CPU; zero test-touch; train+dev only, N≈549/80).
- **K-FA-1 (Pareto gate, binding):** on MHC-EN dev, the best-w composition's **per-class recall** must be a Pareto move over
  CLIP-concat (Δhate-recall ≥ +0.03 AND Δnon-hate-recall ≥ −0.01) **AND** Δacc ≥ +0.02 with bootstrap CI-low > 0. A
  *rotation* (Δacc ≤ 0 with the classic +hate/−non-hate trade) = **KILL** (B5/F44 confirmation, no GPU).
- **K-FA-2 (label-oracle-threshold calibration, à la B5, binding):** report acc/mF1 at the *label-oracle* operating point of
  the composition. If the oracle-threshold acc itself is < +0.03 over CLIP-concat, the edge is easy-example ordering ⇒ **KILL**
  (this is the B5 kill-switch, ported).
- **K-FA-3 (machinery validity):** the concat-kNN proxy must reproduce the banked align-head dev sign on MHC-EN (−0.012,
  F44 §1) before any composition claim; else MACHINERY_INVALID.
- **Survival → CC (§2.3):** only a *Pareto* pass (K-FA-1 & K-FA-2 clear) promotes to CC and a D7-composition ruling; there is
  no GPU inside FA.

**(e) Cost.** **$0**, CPU, minutes, banked caches only. Reuses two committed scripts. No Modal, no download.

---

### CANDIDATE MJ — MLLM modality-reliability judgment as a NEW router input (F47's explicit carve-out)

**(a) Mechanism / bandwidth / injection point.** F47 closed per-item routing over **decision-level meta-features** but wrote
its own escape clause: *"unless the selector input is a genuinely NEW information source not derivable from banked
features/votes."* MJ supplies exactly that: a per-video **MLLM meta-cognitive judgment** — Qwen2.5-VL-7B asked *"is the
hateful content here carried mainly by the visuals or by the speech/text?"* (a low-bandwidth reliability signal), used as an
*additional* feature in the F47 router to pick CLIP-arm vs Qwen-arm on the MHC-EN disagreement subset, targeting the
F44 rotation (route text-borne items to Qwen, visual-borne to CLIP). Injection = decision/vote (per-item selection);
bandwidth = a few bits of a *new* signal class.

**(b) Non-isomorphism.**
- **vs F47 (the router kill):** F47's selector saw *only* vote margins / purity / sub-votes / confidence-differential /
  transcript stats — all derivable from banked features/votes. An MLLM modality-reliability judgment is a **generative
  meta-cognitive output** not linearly/GBM-derivable from those, so it satisfies F47's literal carve-out.
- **vs C3-target (predicted-attribute, dead):** C3 asked *"who is the target"* as a *content* channel fused into the key; MJ
  asks *"which modality is reliable"* as a *routing* signal on the disagreement subset. Different question type, different
  injection point.

**(c) Expected-effect prior — LOW.** Strongly grazed by three banked negatives: **C3-target** measured *real Qwen-7B
predicted attributes* at best +0.0094, **MHC anti-informative**; **D1** (decision-side MLLM outputs conditionally redundant);
**F44** "no coherent subgroup" + **F47** realizable dev-CV ceiling −0.046. If the MLLM reliably knew which modality carried
the hate, its own encoding would already reflect it (and F44 shows it does not resolve the rotation). Prior ~**5–12%**. The
one thing keeping it alive is that it is the *only* proposal satisfying F47's literal new-source carve-out.

**(d) Gate design.** Generate the modality judgment for MHC-EN videos on **Modal cloud** (features/labels export OK per
policy; raw videos never leave — the judgment is produced from already-uploaded frame/transcript features or re-derived
in-container), add it as one feature to `cross_channel_router_gate.py`, re-run F47's gate verbatim.
- **K-MJ-1:** routed − best-single ≥ +0.020 MHC-EN dev (3-seed mean) AND boot CI-low > 0 (F47's K-R1).
- **K-MJ-2:** label-oracle calibration accZA ≥ 0.99 (F47's K-R2).
- **K-MJ-3:** dev-CV realizable ceiling with the MJ feature > perm-null p95 (F47's K-R3) — the decisive one: F47's ceiling was
  −0.046, so MJ must move a *negative* ceiling above the null, a steep ask.
- Cloud drift is triage-only (never mixed with local numbers); a pass promotes to a local pre-registered router, a fail closes
  the F47 carve-out.

**(e) Cost.** Modal Qwen generation for ~629 MHC videos (cloud, ~1–2 h) + $0 router re-run. Higher than FA; lower than any
training route.

---

### CANDIDATE CC — Cross-encoder composition as a deployed architecture (D7-*composition* ruling) **[downstream of FA]**

**(a) Mechanism.** If FA's K-FA-1/2 show a *Pareto* composition on MHC-EN, deploy `concat(CLIP-imĝ , Qwen-text̂)` (or the
best-w weighted fusion) as a fixed architectural choice → freshly-trained RGCL head → 3-seed paired vs frozen-CLIP, both
protocols, the standard `enc3s` +0.03/+0.03 rule. **A fixed composition, no per-item routing, no encoder-identity swap.**

**(b) Non-isomorphism.** Distinct from a plain encoder swap (it is a *cross-encoder modality decomposition*, a claim that
"MLLM language + vision-encoder grounding" beats either encoder whole) and from F47 (no selector). Its live grazes are (i)
still uses frozen-encoder streams ⇒ **D7 ruling** and (ii) subsumed by FA's prior on performance.

**(c) Prior — LOW and gated:** only reachable if FA passes (prior ~10–20% that FA passes); conditional on FA-pass its own
3-seed pass is plausible but **D7-novelty-unresolved** (is a cross-encoder composition a "novel MLLM mechanism" or encoder
plumbing?). This is a *narrower* D7 question than the LoRA option-c and is worth surfacing to the user as such.

**(d) Gate.** Standard `enc3seed.sbatch` pre-register → independent review → freeze → single-submit, KS = +0.03/+0.03 3/3
both protocols vs frozen-CLIP (12850 control on disk). ~2 min GPU (caches exist) once FA clears.

**(e) Cost.** ~2 min GPU **only if FA passes**; else never runs.

---

## 3. RANKING (prior × cheapness × non-isomorphism robustness) + recommended gate

| # | candidate | one-line mechanism | prior | cheapness | non-iso robustness | overall |
|---|---|---|---|---|---|---|
| **1** | **FA** | does a modality-reweighted / cross-encoder fusion convert the cancelled Qwen-text gain on MHC-EN (Pareto) or only re-rank (rotation)? | LOW (~10–20% Pareto) | **$0 CPU, banked, minutes** | **HIGH** — F44's dismissal is on a mis-described (align) head; this cell is genuinely unmeasured | **LEAD** |
| **2** | **MJ** | MLLM modality-reliability judgment as the one NEW router input F47's carve-out allows | LOW (~5–12%) | MODERATE (cloud Qwen gen) | MODERATE-HIGH (literal F47 carve-out) but grazes C3-target hard | 2nd |
| **3** | **CC** | fixed cross-encoder composition (CLIP-img ⊕ Qwen-text) → trained head, D7-composition ruling | LOW, gated on FA | ~2 min GPU *iff* FA passes | MODERATE (D7-composition, narrower than LoRA-c) | 3rd (downstream) |

**RECOMMENDED SINGLE NEXT GATE → FA ($0 fusion/composition probe).** It is the only positive-EV move left inside the box:
zero cost, decisive either way, and it **converts F44's asserted fusion-dismissal (built on the §0.3 align/concat
mis-description) into a G0-cond measurement.** Its most likely outcome is a *rotation* result that **cleanly closes the last
modality-fusion door and strengthens the terminus** (a fifth "better-signal-no-conversion" datum, paper-grade); its unlikely
Pareto outcome promotes to CC and a narrow D7 ruling. Either way it is honest, cheap, and in the house style of the CTF / APX /
GIR $0 gates. **It is not represented as a likely goal-hit** — the MHC-EN label-limited wall makes the performance prior LOW.

---

## 4. IF the loop wants a GOAL-HITTING opening — minimal relaxations BEYOND the five terminus options

The five terminus options are (a) 32B/72B download, (b) Omni download, (c) D7-LoRA ruling, (d) goal renegotiation, (e)
closed-API. Wave-4 adds three *new* minimal relaxations, each strictly smaller than "renegotiate the goal":

| # | new relaxation | exact constraint it lifts | why it is minimal / distinct from the five | honest prior it helps |
|---|---|---|---|---|
| **f** | **D7 sub-ruling for cross-encoder *composition***| "encoder-class levers do not satisfy novelty" (F24) — *narrowly*, for a fixed multi-encoder modality composition (not a swap, not LoRA) | strictly narrower than option-(c): asks only whether *composing streams from two frozen encoders* is a novel architectural contribution, a different question than "does SFT-adapting one encoder count" | LOW — gated on FA passing first (measure before ruling) |
| **g** | **shared multi-task head across the 3 own-train splits** (one head, dataset-conditioned; each item still from its own dataset) | "single-dataset own-train-split only" (F24) — *without* the vetoed split-mixing | the veto's stated rationale was "cross-dataset split *mixing* = trivial trick"; a shared representation trained multi-task (no example crosses datasets) is a *different object* the veto may not intend to cover — a user clarification, not a re-open | LOW-UNKNOWN — untested; but not MLLM-specific, so weak on the goal's "MLLM" clause |
| **h** | **transductive / test-time adaptation** (adapt the head on *unlabeled* test features at inference) | no explicit ban, but risks the standing test-touch discipline | a genuinely untested method *class* (not in the ledger); would need a leakage-safe protocol ruling and is **not MLLM-specific** ⇒ weak on the goal's "MLLM integrated" clause | LOW — off the MLLM axis |

Relaxation **(f)** is the recommended one to hold in reserve: it is the natural landing spot **if and only if FA returns a
Pareto pass**, and it is the smallest constraint-lift that could turn "HateMM(swap) + MHC-EN(composition)" into an arguable
≥2-dataset story — still with both legs encoder-adjacent, so the user should weigh it against option-(d) honestly.

---

## 5. DOCUMENTED PRE-KILLS (so wave-5 does not re-spend)

- **Cross-encoder memory** (query-space ≠ key-space, trained alignment). **KILL (angle c):** collapses to Axis-B (trained
  alignment = representation lever) or F47 (per-item selection) or Axis-A (fixed vote-averaging). Label info is
  encoder-invariant; only geometry changes. No probe.
- **De-memorized / out-of-fold memory bank** (cross-fit each memory item's embedding). **KILL (angle d):** not an MLLM lever
  (applies to CLIP identically ⇒ fails the goal's "MLLM integrated" clause); early-stop variant = tactic not structure; no
  evidence the deployed out-of-sample vote is memorization-degraded. No probe.
- **RAG-into-MLLM** (retrieve neighbours, feed into MLLM context for the decision). **KILL (angle e):** decision-side
  MLLM-as-reasoner = D1-dead + P2/P4 territory; in-domain reasoning-fusion wall (2512.02743 / 2601.15115). No probe.
- **Intermediate-layer / multi-prompt MLLM features.** **KILL:** Axis-B encoder-representation choice (which layer / which
  prompt to read), D7-dead, and redundant with the pooled last-token key. No probe.
- **MLLM-difficulty-weighted head training / MLLM-similarity contrastive target.** **KILL:** MLLM-scores-as-training-signal
  (banned) + C5/CRD (banned). No probe.
- **Generic fusion-mode swap (align→concat) as a GOAL candidate.** **KILL-as-goal:** F44 §1 shows concat-kNN Qwen−CLIP dev is
  *also* −0.012 on MHC-EN, so align→concat does not rescue MHC-EN; and it is not MLLM-specific. It survives *only inside FA*
  as a control arm, never as a standalone goal candidate.

---

## PROVENANCE
- Bans / axes / positives: `state/directions_tried.json`; findings `state/findings.jsonl` F35–F47.
- Deployed head architecture (the §0.3 correction): `src/model/classifier.py:70-129` (`classifier_hateClipper`,
  `fusion_mode='align'` ⇒ `torch.mul(imĝ, text̂)`); config anchor `ROUTER_GATE_RECORD.md` §1 (align, map_dim 1024, 3-layer).
- F44 numbers reused (concat-kNN proxy, validated §1): `ENCODER_SWAP_DIAGNOSIS.md` (`8a48938`) — MHC-EN image 0.734→0.599,
  text 0.797→0.851, concat 0.801→0.825, dev Qwen−CLIP −0.012, per-class rotation +0.040 hate / −0.036 non-hate.
- F45 (adaptation converts): `B3_ZH_LORA_DECOMPOSITION.md` (`d76e407`). F47 (router + carve-out): `ROUTER_GATE_RECORD.md`
  (`30d0ee1`) — MHC-EN oracle +0.1083, dev-CV −0.046, carve-out text §5/directions_tried.json F47 entry.
- Line A (do not duplicate): `LORA_HATEMM_FORENSIC_RECON.md`, `LORA_HATEMM_PREREG` frozen `8de0991`.
- Machinery for FA: `scripts/analysis/encoder_swap_geometry.py`, `encoder_swap_diagnosis_tables.py`; banked
  `data/CLIP_Embedding/MHC/{train,dev_seen}_{openai_clip-...,Qwen2.5-VL-7B-Instruct}_HF.pt`.
- House style / non-isomorphism target: `WAVE3_CANDIDATES.md`; terminus `TERMINUS_round3_mllm_plus3.md`.
- **Required statements:** ZERO GPU / SLURM / Modal spent by this recon; no held-out test metric read or produced; no
  `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated. Not pushed.
