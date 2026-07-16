# TERMINUS ASSESSMENT — round 3 (DRAFT, for the user) — constraint-space exhaustion + relaxation options

**Author:** wave3-recon (ZERO GPU; reading + forensic synthesis only). **Date:** 2026-07-16 (drafted in parallel with the in-flight GIR gate).
**Status:** **DRAFT with one PENDING slot (GIR).** GIR — the last candidate in the wave-3 pool — is running under `gir-gate-executor` (task #44) in its correctly-fired live branch (W2-A raw-FAIL + oracle-SURVIVE). Its verdict folds into §1 Axis-6 and §3 when it lands; **this document is not final until that slot is filled.**
**This is not a decision.** Every option below is framed as a **user ruling with evidence attached** (§4). The loop cannot self-authorize any of the relaxations; they lift constraints only the user set.

**The goal being assessed (frozen, user-set).** An MLLM **meaningfully AND novelly** (D7) integrated into the retrieval-contrastive method, delivering **substantial performance improvement** — the user's loop framing: **+0.03 acc AND +0.03 macro-F1 on ≥2 of the 3 datasets** (HateMM, MHC-EN, MHC-ZH). Pipeline core (unchanged all campaign): frozen encoder → triplet+BCE alignment head → top-20 kNN rank-weighted signed-cosine vote.

**Terminus condition.** After S2S (F37), CTF (F39), APX+AVC (F41), and W2-A (F42), the wave-2/wave-3 candidate pool is consumed except GIR-in-flight. If GIR kills, **every axis reachable inside the frozen constraint box is closed by a binding verdict or a calibrated-zero G0-cond gate** — the goal is unmet not for want of ideas but because the constraint box has no remaining opening. That is the claim this document supports and hands to the user.

---

## 0. The two structural findings that explain every closure (read these first)

Two mechanisms, each independently established at multiple levels, account for the entire graveyard. They are findings, not line items.

### 0.1 The oracle-exists-but-unconvertible wall — now THREE independent instances (P3 · S2S · W2-A)

In each case a **gold/label oracle proves the convertible information is present**, yet the **best achievable in-constraint operator recovers essentially none of it**:

| instance | oracle headroom (upper bound, gold-guided) | achievable operator's raw effect | verdict |
|---|---|---|---|
| **P3** (MLLM segment-density pooling) | probe passes on all 3 datasets | training flat, 3 datasets | dead (CAMPAIGN_mllm_method_role) |
| **S2S** (frame-group set-matching, F37) | **+0.0917 HateMM / +0.1399 MHC** | +0.0035 / −0.0397 (MeanMaxSim) | KILL `2c96ab6` |
| **W2-A** (transcript-grounded key, F42) | **+0.0635 HateMM / +0.0970 MHC** | K9 −0.0000 / −0.0038 (conditional-info); advisory kNN −0.0259 / −0.0509 | KILL `7228373` |

**The wall:** the information the goal needs is demonstrably *in the signal*; no unsupervised, frozen, or (CTF) even supervised generalizing operator inside the constraint box converts it. The prereg discipline anticipated exactly this — every prereg states the oracle ceiling "can **NEVER** be claimed as a result." Three consecutive independent confirmations make this the campaign's deepest empirical law, not an accident of one route.

### 0.2 The three-level cumulative-causal closure of temporal/per-frame structure (F35 · F37 · F39)

Qwen2.5-VL's LLM backbone is fully causal (`is_causal=True`, verified at transformers-source level), so the per-frame-group vectors are **cumulative causal prefix summaries, not frame-local states**. This is closed at all three operator levels:

- **F35 (structural / source-code):** the groups *are* prefixes — a permutation-based temporal control is unsatisfiable by construction; frame-local order semantics are unavailable.
- **F37 (unsupervised operator):** set-to-set MeanMaxSim cannot recover per-segment information; pooling is effectively lossless on these representations.
- **F39 (supervised operator, CTF):** even a **label-supervised conditional-info probe** finds **exactly zero** beyond the pooled key (HateMM +0.0000, calibration accZA=1.0 → credited genuine null).

**⇒ The temporal / don't-pool / per-frame axis is closed at structural, unsupervised, AND supervised levels simultaneously** — the most complete closure in the campaign.

---

## 1. CONSTRAINT-SPACE EXHAUSTION MAP (by axis)

Every dead direction from `state/directions_tried.json` (rounds 1–3) + `findings.jsonl` F27–F42, organized by axis. "Closure evidence" cites the binding verdict or the calibrated-zero gate; "meta-pattern" names what killed the whole axis, not just the instance.

### Axis A — Decision-side low-bandwidth signals
**Members:** P1 (prior recal), P2/P2b (neighbor rerank, incl. 32B), P10/P10b/P10c (logit A-fuse 7B→72B, gen-jump), P11 (segment-scores-as-weak-labels), TARC V1/V2/V3 (predicted-target graph conditioning), archive-auto-repair (two-vote deletion), W2-K1 (disagreement-as-signal, documented-kill).
**Strongest closure:** REFLECTION **D1** + the calibrated conditional-info gates that all return genuine nulls with accZA=1.0 (C3, CTF `0eb6d33`, APX `9c54faf`). "Probe passes, training flat" recurs ≥4× (P3/P11/TARC/P4).
**Meta-pattern:** **conditional redundancy.** These inject a few bits→few-dozen bits on the *decision side* of a frozen-feature pipeline; the gates that ever measured *conditional* info (not marginal quality) find the signal accurate but already banked in the representation. 20+ instances. **CLOSED.**

### Axis B — Representation-level encoder levers (identity / adaptation)
**Members:** encoder-swap (**the one +3 positive**, HateMM-only), P9 (LoRA-SFT features), P9b (joint LMM-RGCL), B1 (frozen-Qwen ZH), B2 (32B frozen), B3 (LoRA-Qwen ZH, **marginal positive**), B4 (LoRA-Qwen EN), C1 (QLoRA sequential), C5/old-C4 (7B CRD-KD), R3-C3geo (frozen-Qwen hard-neg mining).
**Strongest closure:** **F24 USER RULING** — encoder-class levers do **not** satisfy the novelty clause (**D7 resolved-negative**). Performance map: HateMM encoder-swap +5.3–5.6 acc 3/3 seeds both protocols (`040adb8`) but **HateMM-only**; MHC-EN fails every encoder lever (P9/B1/B2/B4); MHC-ZH passes only via LoRA-B3 (marginal, §3).
**Meta-pattern:** representation gains are **real but boxed** — (i) they are the *only* +3 class ever seen, yet (ii) encoder-class = **D7-novelty-dead** by user ruling, and (iii) **MHC-EN is data/label-limited** (SAV #18: dilution hypothesis FALSIFIED), so *no* representation lever converts it. **CLOSED for novelty; open only as a D7 relaxation (§2c).**

### Axis C — Retrieval-object / don't-pool family
**Members:** W2-B (frozen-CLIP subclip set-matching), S2S (Qwen frame-group set-matching), C2 multi-view memory (folded into S2S), W2-E (prototype memory).
**Strongest closure:** **S2S F37 outcome (d)**, binding verdict `2c96ab6` (+ W2-B F27 `0f43bdd`); W2-E killed pre-ceremony F28.
**Meta-pattern:** **§0.1 oracle-unconvertible + §0.2 pooling-lossless.** Oracle headroom real (+0.09/+0.14) but MeanMaxSim recovers ~none; set-matching over cumulative-causal reps adds no frame-local information. **CLOSED across both encoders.**

### Axis D — Temporal structure (order / development)
**Members:** W2-C (order-kernel / soft-DTW / transition-set), CTF (supervised temporal-pool + arc-increment).
**Strongest closure:** **§0.2 three-level closure (F35/F37/F39)**; CTF gate `0eb6d33` (HateMM +0.0000, MHC −0.0029, arc −0.0049; calibration valid → credited null). W2-C's sole vehicle (S2S order arm) extinguished with S2S.
**Meta-pattern:** cumulative-causal redundancy at every operator level. **CLOSED.**

### Axis E — Auxiliary channel: audio / prosody
**Members:** APX (whole-video eGeMAPS prosody), AVC (audio-visual correspondence, never started), W2-D (acoustic, queued-conditional).
**Strongest closure:** **APX F41 gate `9c54faf`** — strictest arm (raw 88-d eGeMAPS, no PCA) conditional Δacc **+0.0005 CI[−0.0031,+0.0042] = exactly zero** over Z_best; calibration accZA=1.0 → credited genuine null. (Extraction verified: job 13203 COMPLETED, N=851, 0 NaN.)
**Meta-pattern:** the **ASR transcript channel already banks the spoken-hate content**; classical prosody is conditionally redundant given it. **PARKED at $0** — the one genuinely unused modality returns zero conditional information through the same screen that killed everything else. Reopenable only via an audio-MLLM download (§2b), which would have to beat a *zero-information* classical baseline.

### Axis F — Cross-modal grounding / interaction
**Members:** W2-A (transcript-first grounded vision key), C3-nontarget (dense reasoning-text fusion), C3-target (predicted-target one-hot), P4 (schema-field distillation), P5 (counterfactual twins). **[GIR — PENDING, see slot below.]**
**Strongest closure:** **W2-A F42 K9 binding**, verdict `7228373` — grounded key adds **zero** conditional info over Z_best (HateMM −0.0000 CI[−0.0052,+0.0049]; MHC −0.0038 CI[−0.0099,+0.0019]; covered-rows HateMM −0.0032; advisory kNN grounded *worse* than concat). Reviewer's classification: **"clean C3-style CLIP-redundancy null."** C3-nontarget epitaph: "info banked in Qwen pathway."
**Meta-pattern:** the cross-modal **interaction term is already present** — `text_feats` is a *joint* frames+transcript forward (verified source-level), so re-pooling (W2-A) or generating text (C3) only **reshuffles information the pooled channels already carry**. **CLOSED (pending GIR).**

> **⏳ PENDING SLOT — GIR (grounded-incongruity residual `grd − ungrd_vis`), task #44, running.**
> Live only because W2-A landed in the raw-FAIL + oracle-SURVIVE branch (its pre-registered live condition, `WAVE3_CANDIDATES.md` §2b). Prior **very low**: W2-A's K9 linear-zero is expected to subsume the linear residual arm; the $0 gate runs (rather than armchair-subsuming) only because the kNN metric is nonlinear. **Kill-credit requires K-GIR-2 calibration (accZA≈1.0) to pass.** When the verdict lands: if DEAD, Axis F is fully closed and the terminus claim is complete; if it somehow SURVIVES, it re-opens the isolated-interaction cell for a head-training stage (would then need its own prereg + D7 ruling). **`<FILL: GIR verdict, commit, numbers>`**

### Axis G — Scale
**Members:** B2 (32B encoder), P2b (32B rerank), P10b (72B A-fuse), P10c (Qwen3-VL-32B gen-jump).
**Strongest closure:** **B2 (21st)** — on the HateMM anchor, **32B sits *between* CLIP and 7B: scale REGRESSES**; 32B-vs-7B fails on every dataset; P10c shows generation ≠ scale (in-tier).
**Meta-pattern:** the head+kNN conversion is **alignment/active-param dominated, not scale-dominated**; raw scale never promotes. **CLOSED as an in-constraint axis; reopenable only via 32B/72B download for the *encoder* slot (§2a), against measured counter-evidence.**

### Axis H — Calibration / operating-point conversion
**Members:** B5 (per-encoder threshold calibration), P1 (prior recal).
**Strongest closure:** **B5 F34 K1**, verdict `50f01b9` — the label-oracle ceiling **itself** is under +0.03 on both protocols; the frozen-Qwen ZH roc +0.050 ranking edge is **easy-example ordering**, unconvertible at *any* threshold including the gold-optimal cut.
**Meta-pattern:** AUC/ranking-edge gains are easy-example reordering, **not convertible to acc/mF1** at any operating point. **CLOSED.**

### Axis I — Memory organization (unsupervised)
**Members:** W2-E (prototype/mode-local memory), archive-auto-repair (two-vote deletion).
**Strongest closure:** W2-E F28 (deterministic lossy function of the same pooled vector → zero new signal); archive AUTO-repair AND-rule C−A=0 (guard-rail role only).
**Meta-pattern:** unsupervised reorganization of frozen features adds nothing the flat top-20 vote does not already use. **CLOSED** (human-in-the-loop audit is the only surviving memory role — a guard-rail, not an accuracy lever).

**Coverage check.** Injection points in the pipeline = {encoder/key-construction, memory-organisation, retrieval-metric, decision/vote, training-signal, new-signal-source}. Axes A–I cover all six: decision/vote (A), encoder identity (B), retrieval-metric/object (C), temporal key-construction (D), new-signal-source (E), cross-modal key-construction (F), scale of encoder (G), operating-point (H), memory-organisation (I). **No untried injection point remains inside the frozen constraint box.**

---

## 2. RELAXATION OPTIONS (each lifts a specific user-set constraint — evidence for/against + honest prior)

None of these is recommended; each is the user's to rule. Priors are on **P(clears the ≥2-dataset goal)**, on the banked evidence.

| # | Option | Exact constraint it lifts | Banked evidence FOR | Banked evidence AGAINST | Honest prior |
|---|---|---|---|---|---|
| **a** | **32B / 72B encoder download** | "only Qwen2.5-VL-7B available locally" (32B/72B = user rulings) | encoder-swap is the *one* +3 lever (representation class); larger models are stronger reasoners | **B2 measured scale REGRESSES**: HateMM 32B *between* CLIP and 7B; 32B-vs-7B fails every dataset; P10b 72B A-fuse = localization-only modest, no accuracy role | **LOW** — direct measured counter-evidence; scale is not the conversion lever |
| **b** | **Qwen2.5-Omni audio-MLLM download** | "no MLLM audio encoder locally (only Whisper ASR)"; audio-axis promotion | audio is the one physically-unused modality; HateMM audio-only mF1 0.669 | **F41: classical prosody = calibrated ZERO** conditional info over Z_best (ASR transcript already banks spoken-hate); an Omni encoder must beat a *zero-information* baseline through the same screen; D7-thin ("add audio" = HateMM's 2023 founding contribution) | **LOW** — the cheap prosody proxy already returned zero; D7-marginal |
| **c** | **D7 ruling: accept LoRA-family as "novel enough"** | "encoder-class levers do not satisfy novelty" (D7, F24 user ruling) | **B3 = the one banked positive gated on this**: MHC-ZH LoRA-Qwen 3-seed final-epoch **+0.0313 acc / +0.0453 mF1, 3/3 PASS (MARGINAL)** | val-selected FAIL (+0.0246 acc); margin +0.0013 structural; **single encoder draw**; gain = LoRA adaptation not encoder identity; **only ONE dataset** (ZH). HateMM's pass is encoder-swap (also encoder-class). Even accepting D7, the ≥2-dataset story = HateMM(swap) + ZH(LoRA), **both encoder-class** | **DEFINITIONAL, not empirical** — a ruling, not a run. If accepted, a 2-dataset story becomes *arguable* but both legs are encoder-class and one (ZH) is marginal/protocol-fragile |
| **d** | **Goal renegotiation** | "≥+0.03/+0.03 on ≥2 datasets via a *novel* MLLM mechanism" | What the evidence **does** support (see §3): encoder-swap HateMM +5.3–5.6; MLLM localization (P6/P10b); guard-rail/audit role; B3 ZH marginal | the *substantial-improvement + novel-mechanism + ≥2-dataset* conjunction is unmet on frozen terms and every in-box axis is closed | **N/A (scope choice)** — the honest achievable claim is "MLLM's earned roles = encoder + localizer + guard-rail, not a main-table accuracy lever at 7B" |
| **e** | **Closed-model API route** | "no external model APIs" (data-export not ruled) | frontier closed models are stronger reasoners; an untested scorer increment | decision-side reasoning-fusion is **D1-dead** (in-domain 2512.02743 / 2601.15115 confirm the wall); C3-nontarget shows generated reasoning text redundant; needs a **data-export governance ruling** (features/videos leaving local); P10c: generation ≠ scale in-tier | **LOW** for accuracy; additionally gated on a data-governance ruling |

---

## 3. WHAT THE CAMPAIGN PROVED (the positive ledger — publishable material)

The goal is unmet, but a rigorous campaign of **robust negatives with mechanism-level attribution** is itself a contribution. The strongest five, each with its evidence chain:

1. **Decision-side MLLM signals are conditionally redundant given a frozen retrieval representation (D1).** Not "the MLLM is inaccurate" — the signals pass every *marginal*-quality gate; they fail the *conditional*-info gate. Evidence: 20+ instances (P1–P11, TARC), formalized by three round-3 calibrated-zero gates with label-oracle accZA=1.0 proving genuine nulls (C3, CTF `0eb6d33`, APX `9c54faf`). *Mechanism:* the top-20 kNN vote over the frozen key already contains the label information the decision-side signal carries.

2. **The oracle-exists-but-unconvertible wall (P3 · S2S · W2-A).** Three independent routes where a gold oracle proves convertible headroom (+0.09/+0.14; +0.0635/+0.0970) that the best in-constraint operator cannot recover (§0.1). *Mechanism:* representation-level information is present but not extractable by unsupervised/frozen/supervised operators inside the box — a precise, reusable characterization of *why* auxiliary MLLM signals fail on strong frozen features.

3. **Three-level cumulative-causal closure of temporal structure (F35/F37/F39).** Qwen2.5-VL's causal LLM makes per-frame-group vectors cumulative prefix summaries; temporal/set/order operators are redundant with the pooled key at the structural, unsupervised, and supervised levels simultaneously (§0.2). *Mechanism:* source-level causal masking, empirically confirmed by onset-invariance controls and a calibrated conditional-info null. A concrete, transferable caution for anyone building set/temporal retrieval over decoder-VLM token summaries.

4. **Cross-modal "grounding" is a CLIP-redundancy null, because the joint forward already banks the interaction (W2-A F42, `7228373`).** The transcript-conditioned vision key adds zero conditional info because `text_feats` is *already* a joint frames+transcript pool (verified source-level). *Mechanism:* the interaction term the mechanism claims to add is not missing from the pipeline.

5. **Scale regresses, and AUC-edges don't convert (B2 + B5).** For head+kNN conversion, 32B sits between CLIP and 7B on the HateMM anchor (scale is not the lever, B2); and a real frozen-Qwen ZH ranking edge (roc +0.050) is easy-example ordering, unconvertible at any operating point including the label-oracle cut (B5 `50f01b9`). *Mechanism:* conversion is alignment/active-param dominated; ranking gains ≠ decision gains.

**Banked positives (the earned MLLM roles).** encoder-swap HateMM +5.3–5.6 acc, 3/3 seeds, both protocols (`040adb8`) — the project's most robust effect; MLLM localization scorer (P6 HateClipSeg wv-AUC 0.5435 vs memory 0.5140, paired p=0.007; P10b 72B A-fuse calib); MLLM guard-rail/audit in memory curation (human-in-the-loop). **The defensible thesis: at 7B, the MLLM's earned roles in hateful-video retrieval are encoder, localizer, and guard-rail — not a main-table accuracy lever.**
**`<FILL when GIR lands: add as a 6th ledger item if it confirms the isolated-interaction residual is also a redundancy null.>`**

---

## 4. FRAMING (binding)

**No recommendation is a decision.** Sections 2 and 3 attach evidence to options; they do not choose. Specifically: the loop does **not** recommend a download (a/b), does **not** rule on LoRA novelty (c), does **not** renegotiate the goal (d), and does **not** authorize data export (e). Each lifts a constraint only the user set, and each carries banked counter-evidence the user should weigh. The loop's factual claim is narrow and defensible: **inside the frozen constraint box (7B-only, no gold-in-method, no OCR, single-dataset own-train, no cross-seed ensembles, no external APIs, D7-tightened novelty), every injection point is closed by a binding verdict or a calibrated-zero G0-cond gate, and the ≥2-dataset novel-mechanism goal is unmet.** What happens next is a user ruling, not a loop action.

---

## PROVENANCE

- Graveyard + bans + positives bank + diagnosis frame: `autoresearch/goal_mllm_plus3/state/directions_tried.json`; findings `state/findings.jsonl` F27–F42.
- Round-3 binding verdicts / gates (all LOCAL — SLURM or local CPU; cloud numbers are triage-only and never mixed): W2-A `7228373`/`688ef87` (F42), CTF `0eb6d33` (F39), APX `9c54faf` (F41), S2S `2c96ab6` (F37, Modal triage), W2-B `0f43bdd` (F27, Modal triage), B5 `50f01b9` (F34).
- Structural findings: F35 `4358ca1` (causal-prefix postmortem), F36 `20c0bf2` (onset-invariance amendment ruling), F37 (S2S), F39 (CTF), F42 (W2-A).
- Positives: encoder-swap `040adb8` + erratum `66012e9`; B3 ZH LoRA (positives bank + MEMORY numeric-provenance); P6/P10b (CAMPAIGN_mllm_method_role, novelty-scope memory); D7 user ruling F24.
- Wave-3 candidate designs + kill-switch definitions: `refine-logs/WAVE3_CANDIDATES.md` `0ee06df`.
- REFLECTION D1/D2/D3 + G0-cond mandate: `research-wiki/REFLECTION_mllm_integration_failures.md`.
- **PENDING:** GIR verdict (task #44, `gir-gate-executor`) — folds into Axis-F slot (§1), the §3 ledger, and finalizes §0/§4.
