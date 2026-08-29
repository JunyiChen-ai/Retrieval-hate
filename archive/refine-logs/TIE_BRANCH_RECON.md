# TIE-BRANCH FORENSIC RECON — what remains open inside the box if cand-2 TIES (K-C2-2)

**Agent:** TIE-branch forensic recon. **Date:** 2026-07-18. **ZERO GPU / ZERO Modal / ZERO
test-touch / ZERO user interaction.** Reading + forensic reasoning + prereg-shaped design only.
Deliverable = this committed doc. Did **not** touch the live cand-2 ceremony artifacts or `state/`.

**Situation.** cand-2 (retrieval-confusion curriculum SFT, `CAND2_CURRICULUM_PREREG.md` 76ef0e2) is
mid-measurement (chain 13237–13241). Its pre-declared most-likely outcome (~50–60%, F52) is a
**K-C2-2 TIE** — "generic LoRA with reshuffled data," curriculum adds nothing over generic. This
recon assumes the TIE fires and asks: **with curriculum-shaping tied, is any untested cell left in
the constraint box** {no gold in-method, no OCR, single-dataset own-train, 7B-local, no cross-seed
ensembles, no closed APIs, D7 encoder-exclusion}? It attacks the ledger's *prose* for a third
unexamined premise (the F48-Hadamard and F46-linear-zero finds were both premise attacks — that is
the winning pattern).

**Docs read verbatim:** `state/directions_tried.json` (23 dead + bans + positives), `findings.jsonl`
F44–F53, `TERMINUS_round3_mllm_plus3.md`, `WAVE5_CANDIDATES.md`, `CAND2_CURRICULUM_RECON.md`
(esp. §2.2 leakage/extraction-prompt audit), `B3_ZH_LORA_DECOMPOSITION.md` (F45),
`ENCODER_SWAP_DIAGNOSIS.md` (F44), `FA_GATE_RECORD.md` (F50), plus the **source-level extraction
and LoRA configs** (`src/utils/generate_VideoMLLM_embedding_lora_HF.py`,
`RA-HMD/.../my_configs/hatevideo/*.yaml`, `scripts/analysis/w2a_extract.py`).

---

## BOTTOM LINE UP FRONT

1. **The third unexamined premise exists and is load-bearing: "LoRA moves the TEXT stream only"
   (F45 / TERMINUS §0.2) is an EMPIRICAL fact about the generic *yes/no target*, NOT an
   architectural constraint.** I verified at source level that `img_feats` is the pooled hidden
   states of the **vision-token span from a forward that passes through the LoRA-adapted LLM**
   (vision tower + multimodal projector are frozen; `lora_target: all` adapts the LLM backbone,
   which re-processes the vision tokens). So the image stream is architecturally **movable by
   adaptation** — the campaign never moved it only because every SFT target so far (yes/no with the
   transcript present) is text-decodable and routes gradient into the language pathway. §0.2's
   second-leg-death ("no text-stream lever can convert image-borne HateMM or image-collapsed EN")
   rests entirely on this premise.

2. **BUT the premise attack does not, by itself, open a new dataset — because F50 already priced the
   image stream OUT as EN's binding constraint.** F50 composed the *healthy* CLIP image stream
   (EN AUC 0.734) with frozen Qwen-text (AUC 0.851) → composite AUC **0.898 (best ever on EN)** yet
   `d_oracle +0.025 < +0.03` = unconvertible. A healthy image stream on EN does **not** convert at
   the oracle threshold; therefore *making Qwen's image stream healthy via vision-adaptation*
   (premise-(a)'s only goal-relevant payoff) is largely pre-priced below the bar. HateMM/ZH already
   pass, so image-stream adaptation adds no new dataset there either.

3. **The ONE genuinely-uncovered, $0, goal-relevant cell is the F50 carve-out the ban names itself:
   CLIP-img ⊕ LoRA-Qwen-text.** F50's ban language is explicit — *"do not re-propose fixed
   compositions … over banked FROZEN features; **conversion requires adaptation (F45)** or a new
   information source with alignment>0.663."* The FA gate's A2 arm used **frozen** Qwen-text and
   missed by +0.005 (`d_oracle +0.025` vs the `+0.03` bar). Swapping in the **LoRA-adapted**
   Qwen-text — the adaptation the ban itself names as the carve-out — is a genuinely untested
   composition, it is **$0** (all three EN caches are banked on disk), and it needs only to close a
   +0.005 oracle gap that a real text-adaptation (F45: ZH text AUC 0.847→0.925) could plausibly
   supply. This is the single recommended gate for **both** branches.

4. **Everything else in the box is dominated or metric-incoherent** (multi-round iteration:
   text-stream + encoder-class + downstream of a tied cand-2; asymmetric archive/query encoder:
   cross-space cosine is ill-defined; rationale-generation SFT target: grazes P11/gold). Honest
   ranking + kill-switches in §2–3. If the §3 gate dies, EN is closed at the *adaptation* level too
   and the round-3 terminus is complete for the adaptation family → revert to the D7 ruling on
   generic LoRA + the 5 relaxations (§4).

---

## 0. The architecture facts this recon establishes (source-verified, load-bearing)

These were NOT previously pinned in the ledger and they decide which premise attacks are real.

**F-arch-1 — `img_feats` passes through the LoRA-adapted LLM.**
`generate_VideoMLLM_embedding_lora_HF.py:29,264-330`: `img_feats` (Dv=3584) is the **mean over the
vision + instruction span** of a forward whose content is **`[8 frames (as video)] + FIXED
IMG_INSTRUCTION`** ("Describe the people, symbols, gestures, and on-screen text in this video.") —
**no transcript, no title**. `text_feats` (Dt=3584) is the **response-span** pool of a **joint**
forward `[frames] + title + transcript + FIXED TEXT_INSTRUCTION`. Both forwards run the vision
tokens through the LLM. The hatevideo LoRA configs
(`my_configs/hatevideo/p9_mhc_zh_d3lam0_s0.yaml:15-20`) set **`freeze_vision_tower: true`,
`freeze_multi_modal_projector: true`, `lora_target: all`** — the ViT and the projector are frozen,
but the LLM backbone (which re-contextualizes the vision-pad tokens) is adapted. ⇒ **The image
stream is architecturally movable by an SFT target that routes gradient through the vision tokens.**
F45's "image untouched −0.007" is therefore a property of the *generic yes/no, transcript-present*
target, not a wall.

**F-arch-2 — the `img_feats` extraction forward is already transcript-free.** Because IMG_INSTRUCTION
carries no transcript, a **transcript-dropout SFT curriculum** (train video-only yes/no) is
*better matched* to the `img_feats` extraction condition than generic SFT is (generic SFT includes
the transcript). This dodges the cand-2 §2.2 trap: the self-defeating designs there were
**prompt-SHAPE** changes (2-video neighbour/paired prompts the extraction never provides);
transcript-dropout keeps the **single-video shape** and only removes a field the img_feats forward
already omits. So there is no train/deploy shape mismatch for the vision path.

**F-arch-3 — F50 already measured the healthy-image EN ceiling.** `FA_GATE_RECORD.md` A2:
`z = [√w·imghat_CLIP , √(1-w)·texthat_Qwen(frozen)]`; peak dev AUC **0.8982 at w=0.15**, `d_oracle
+0.025`, KILL. CLIP-img EN 0.734 ⊕ frozen-Qwen-text 0.851. So the *healthiest available image
stream* composed with the *frozen* text stream is unconvertible on EN at the oracle cut. The only
untested lever left on that composition is the **text side becoming adapted** (LoRA), which the ban
explicitly carves out.

---

## 1. The four candidate premises the task flagged — adjudicated honestly

| # | premise attacked | verdict | why |
|---|---|---|---|
| **(a)** | "LoRA moves the TEXT stream only" (F45/§0.2) — via a vision-obligatory / **transcript-dropout SFT curriculum** that routes gradient into the vision tokens | **STRUCTURALLY REAL, but EN-payoff pre-priced by F50** → survives only to a $0 forensic that likely kills it; keep as the *image-side* companion to (d) | F-arch-1/2 make it architecturally viable and non-isomorphic to generic LoRA (different modality-locus of adaptation). But its sole goal-relevant target is EN, and F-arch-3 shows a *healthy* image stream ⊕ text is already unconvertible on EN. Adapting Qwen-img toward CLIP-img's 0.734 improves the component F50 shows is **not binding**. HateMM/ZH already pass. Prior ~5–8%. |
| **(b)** | "no ban covers iteration" — multi-round: re-mine confusion under the *adapted* encoder, 2nd LoRA round | **OPEN but DOMINATED** | No ban names iteration. But it is still a **text-stream** lever (§0.2), still **encoder-class** (D7-dead), and it is **downstream of cand-2**: if cand-2 TIES, a 2nd round on a tied lever will not un-tie the goal; if cand-2 CLEARS, iteration is a marginal ZH/HateMM refinement, not a new dataset. No new-conjunct path. |
| **(c)** | "memory-rebuild policy (which encoder embeds archive vs query) is untested" | **NEAR-NON-STARTER** | Cosine between a LoRA-space archive and a frozen-space query is geometrically ill-defined (different bases); the coherent version ("which single encoder embeds a shared archive") **collapses to the encoder-swap question** (Axis-B, fully mapped). W2-A's ban is about grounded-key *identity*, not archive/query policy — so it is technically untested, but it carries no new conditional information the top-20 vote does not already use (Axis-I closure). |
| **(d)** | F50's own ban: "over banked **FROZEN** features; conversion requires **adaptation**" — **CLIP-img ⊕ LoRA-Qwen-text** | **LEAD — genuinely uncovered, $0, goal-relevant** | The literal adaptation carve-out. FA-A2 with frozen→LoRA text swapped. Non-isomorphic to F50 (frozen text), to B4/F53 (deployed LoRA pipeline = **collapsed** LoRA-img ⊕ LoRA-text, never the healthy CLIP-img), and to cand-2 (Qwen-only curriculum). All EN caches banked → $0. Full treatment §2. |

**On the task's P4 sub-question** ("is generation-target-as-SFT-supervision the same cell as P4
schema-distill?"): **No** — P4 appended MLLM-generated schema *fields as decision-side FEATURES*
(a decision-side channel); an SFT *generation target* shapes the *encoder*. Different injection
point, non-isomorphic. **However**, the gold-free realization matters: training the encoder to
*generate a hate rationale* needs either gold rationales/spans (**banned**: gold in-method) or
MLLM-self-generated targets (**grazes** P11 "MLLM-scores-as-training-signal" and adds no external
information — self-distillation is conditional-redundancy-shaped). The **clean, gold-free,
non-isomorphic** version of "change what the SFT conditions on" is therefore the **input-modality
ablation** (transcript-dropout, premise (a)), **not** rationale generation. Record this so wave-6
does not re-propose rationale-target SFT as if it were novel and cheap.

---

## 2. LEAD candidate — full treatment: CLIP-img ⊕ LoRA-Qwen-text composition (premise (d))

**(a) Mechanism / injection point / bandwidth.** The RGCL head fuses two L2-normed blocks
`[img_feats, text_feats]`. F50 proved the best *frozen* cross-encoder block-pair on EN
(CLIP-img 0.734 ⊕ frozen-Qwen-text 0.851) reaches composite AUC 0.898 but sits `+0.025` at the
oracle threshold — a **rotation that misses convertibility by +0.005**. F45 proved that on ZH the
*adaptation* of the Qwen text stream (frozen 0.847 → **LoRA 0.925**, +0.078) is precisely what turns
a frozen **rotation** into a convertible **Pareto** move (+0.111 hate-recall at −0.003 non-hate).
Premise (d) applies that exact adaptation to F50's best composition: **replace the frozen Qwen-text
block with the banked LoRA Qwen-text block, keep the healthy CLIP image block**, and re-run the FA
oracle machinery. Injection = decision-side fixed composition over one *adapted* + one *frozen*
block; bandwidth = the LoRA text-stream AUC gain on EN (unmeasured in isolation).

**(b) Non-isomorphism (scope-quoted).**
- **vs F50 (`directions_tried` FA ban):** *"…over banked **frozen** features; conversion requires
  **adaptation (F45)**…"* — premise (d)'s text block is LoRA-adapted, i.e. the carve-out the ban
  names. Not "banked frozen features."
- **vs B4 / F53 (deployed LoRA-EN pipeline, FAIL both protocols):** B4 fused **LoRA-Qwen-img
  (collapsed, F45 leaves image flat) ⊕ LoRA-Qwen-text**. Premise (d) fuses **frozen CLIP-img
  (healthy 0.734) ⊕ LoRA-Qwen-text** — the healthy image block B4 never used. Different composition.
- **vs cand-2 / WAVE5:** those are **Qwen-only encoder curricula**; premise (d) is a **cross-encoder
  decision-side composition**, no training. Orthogonal axis.
- **vs the training-data veto / OCR / gold bans:** no training, no OCR, no gold; within EN only.

**(c) Expected effect — LOW-MODEST (~10–20%), grounded.** The gap to close is small (`+0.005`
d_oracle). LoRA text-adaptation is a real Pareto lever on ZH (F45). AGAINST: F44 rules EN
**label-limited** (rotation net −1), B5 rules EN AUC-edges **easy-example ordering**, and B4 shows
generic LoRA-EN **fails** — three converging negatives, all on **frozen or collapsed-img**
compositions the F45 escape hatch does not strictly bind. Honest read: this is the **best-priced
cheap shot at EN** the box has left, precisely because it needs a *small* convertibility gain from
the *one* adaptation the ban itself admits can convert. Only EN is in play (HateMM/ZH pass already);
clearing it would give a **val-selected 2-dataset story** the current ledger lacks (F53: val-sel =
HateMM only).

**(d) Novelty vs D7 (honest).** If it clears, the deployable method is "RGCL head over
[CLIP-img, LoRA-Qwen-text]" = a **fixed cross-encoder composition** — the **relaxation-(f)
D7-composition sub-ruling** F50 declared MOOT *only because FA failed*. A pass **reactivates** that
sub-ruling; novelty remains a user call (composition-novelty, distinct from generic-LoRA D7).

**(e) Gate design — $0 CPU, oracle-gated, FA machinery reused verbatim.**
- Inputs (all **banked on disk**, verified): `data/CLIP_Embedding/MHC/{train,dev_seen}_openai_clip-vit-large-patch14-336_HF.pt`
  (CLIP-img), `…/{train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` (LoRA text). Ids align by
  common key (FA already verified CLIP↔Qwen order match).
- Probe = FA's A2 arm with `texthat_Qwen(frozen) → texthat_Qwen(LoRA)`, w-grid {0.00..1.00},
  `d_oracle` = candidate@oracle-τ − CLIP-concat@oracle-τ (both arms get their own dev-oracle
  threshold; test never read). Reuse `scripts/analysis/` FA composition script (`9e2fcbf3`).
- **Cost:** minutes, CPU, $0. No Modal, no SLURM, no test-touch.

**(f) Kill-switches (pre-declared, FA-ported).**
- **K-D-0 (machinery):** planted Pareto/rotation detectors fire + HateMM positive control would-pass
  (same calibration as FA). Fail → VOID.
- **K-D-1 (oracle, binding):** `d_oracle < +0.03` on EN dev = KILL (B5 port — AUC edge is
  easy-example ordering, unconvertible).
- **K-D-2 (Pareto-not-rotation):** the point-bar config must be Pareto-shaped (both classes' recall
  ≥ baseline, or minority-up at ~0 majority cost); a rotation that only trades errors → report
  no-value even if AUC rises (B5/F44 discipline).
- **K-D-3 (deployable-w sanity):** train-LOO-selected w must not regress below CLIP-concat floor.

---

## 3. RANKING + single recommended gate per branch

| # | candidate | one-line | perf prior (new conjunct) | novelty | cost | overall |
|---|---|---|---|---|---|---|
| **1** | **CLIP-img ⊕ LoRA-text** (premise d) | FA-A2 with the adaptation carve-out; closes a +0.005 EN oracle gap | LOW-MODEST (~10–20%) | D7-composition sub-ruling (relaxation-f, reactivated on pass) | **$0 CPU, banked caches** | **LEAD** |
| **2** | **transcript-dropout / vision-obligatory SFT** (premise a) | routes LoRA gradient into the movable image stream | LOW (~5–8%, EN pre-priced by F50) | encoder-class + composition; user D7 | local SLURM SFT (~4 h/ds), behind queue | **companion, conditional on #1** |
| **3** | multi-round iteration (premise b) | 2nd LoRA round under adapted encoder | ~nil new dataset | encoder-class (D7-dead) | 1 SFT/ds | dominated |
| — | asymmetric archive/query encoder (premise c) | mix encoders for memory vs query | nil (metric-incoherent) | — | — | non-starter |

**Recommended next gate — SAME for both branches: run the premise-(d) $0 CPU composition gate
(CLIP-img ⊕ LoRA-Qwen-text on MHC-EN, FA oracle machinery, K-D-1 binding).** It is the cheapest
possible move, it is genuinely uncovered by the literal F50 ban, it targets EN (the one dataset
§0.2 declares structurally unreachable), and its outcome is decisive either way:

- **TIE branch (cand-2 K-C2-2 tie).** The Qwen-only adaptation family is confirmed text-stream and
  fully mapped. Premise (d) is the *only* remaining $0 attack on EN. Run it **now, in parallel with
  the cand-2 verdict** (it touches nothing cand-2 touches). If it **clears** K-D-1 as a Pareto move,
  escalate to full ceremony (prereg→0-context review→freeze→single-submit) and **then** premise (a)
  vision-obligatory SFT becomes worth a local-SLURM run (adapt the image side to lift Qwen-img toward
  CLIP-img and drop the CLIP dependency). If it **dies**, EN is closed at the adaptation level too;
  the round-3 terminus is complete for the entire adaptation family → §4.
- **CLEAR branch (cand-2 clears K-C2-2, coupling adds over generic).** Curriculum-coupling is
  novelty-bearing on ZH+HateMM but adds no dataset (§0.2). The live "what's the third dataset"
  question is still EN, and premise (d) is still the cheapest EN probe; additionally, premise (a)'s
  vision-obligatory idea **composes with** the now-validated cand-2 confusion-mining (a
  *confusion-mined video-only* curriculum) as the natural EN follow-on. Same recommended first gate.

---

## 4. IF NOTHING SURVIVES — the honest fallback (and the minimal reopeners)

If the premise-(d) gate dies at K-D-1 (the honest modal outcome given F44 label-limited + B5
AUC-edge + B4 LoRA-EN-fail), then **EN is closed at the frozen, collapsed-adapted, AND
healthy-img-adapted composition levels simultaneously**, the "LoRA moves text only" premise is
attacked-but-inert (image movable, but the image stream is not EN's binding constraint), and **no
in-box cell remains**. The result is not new: it reverts to `TERMINUS_round3` + the F53 blockers:

1. **D7 ruling on generic LoRA** (the actual live decision). F53 already meets the performance
   conjunct on **2 datasets under final-epoch by one lever** (HateMM +0.0573/+0.0682 SOLID, ZH
   +0.0313/+0.0453 MARGINAL). The blocker is purely whether encoder-level LoRA counts as the
   *novel* MLLM integration (D7 / relaxation-c). This is a **user ruling, not a run** — and it is
   the highest-EV move on the board, requiring zero further GPU.
2. **The 5 TERMINUS relaxations** (a 32B/72B encoder [B2 counter: scale regresses], b Qwen2.5-Omni
   [F41 counter: classical prosody zero], c D7-LoRA [F45 evidence upgrade], d goal renegotiation,
   e closed-API [data-export ruling]) — each a user ruling with banked evidence attached.
3. **The moot relaxation-(f) D7-composition sub-ruling** reactivates *only if* premise (d) passes;
   otherwise it stays moot.

**Minimal reopeners beyond the 5, with evidence-based EV (honest):**
- **Reopen-(g): allow a fixed cross-encoder composition (CLIP-img ⊕ LoRA-Qwen-text) as the deployed
  key** — this is relaxation-(f) generalized; EV = exactly the premise-(d) gate result. **Run the $0
  gate first; do not ask for the ruling until it passes.** (~10–20% it is even on the table.)
- **Reopen-(h): allow vision-obligatory SFT (transcript-dropout curriculum) as a novel encoder
  coupling** — EV low (~5–8%, F50 pre-prices the image ceiling), but it is the only lever that would
  make the D7 story a *non-generic* coupling AND is the natural composition partner if (d) passes.
  Gate it behind (d).

No reopener beyond these has positive evidence; the axis map (TERMINUS §1, A–I) remains complete and
every closure is a binding verdict or a calibrated-zero $0 gate. The honest campaign claim is
unchanged: **inside the frozen box, the goal is unmet; the live decision is D7 on generic LoRA, and
the one cheap probe still worth spending is the $0 premise-(d) EN composition.**

---

## PROVENANCE
- Architecture (source-verified this recon): `src/utils/generate_VideoMLLM_embedding_lora_HF.py:29,264-330`
  (img_feats video-only vision+instruction span; text_feats joint response span);
  `RA-HMD/LLAMA-FACTORY-Ver202512/my_configs/hatevideo/p9_mhc_zh_d3lam0_s0.yaml:15-20`
  (`freeze_vision_tower`/`freeze_multi_modal_projector: true`, `lora_target: all`);
  `scripts/analysis/w2a_extract.py:15-20` (banked img_feats = L2 mean of vision-pad hidden states).
- Premise-(d) inputs banked (verified `ls`): `data/CLIP_Embedding/MHC/{train,dev_seen}_openai_clip-vit-large-patch14-336_HF.pt`,
  `…/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt`.
- Findings cited: F44 `ENCODER_SWAP_DIAGNOSIS.md` 8a48938 (EN image CLIP 0.734→Qwen 0.599, label-limited);
  F45 `B3_ZH_LORA_DECOMPOSITION.md` d76e407 (ZH text 0.847→0.925, image −0.007, Pareto +0.111/−0.003);
  F50 `FA_GATE_RECORD.md` e0877c9 (A2 CLIP-img⊕frozen-Qwen-text AUC 0.898, d_oracle +0.025 KILL; ban language);
  F52 `CAND2_CURRICULUM_RECON.md` 7087b5a (§2.2 extraction-prompt-fixed / prompt-shape-mismatch);
  F53 `LORA_HATEMM_VERDICT_REVIEW.md` 6b8f634 (HateMM PASS both, KS-2 clean, EN LoRA FAIL both).
- Terminus map + relaxations: `TERMINUS_round3_mllm_plus3.md`; adaptation two-object closure:
  `WAVE5_CANDIDATES.md` 7166232.
- **Required statements:** ZERO GPU / SLURM / Modal spent; no held-out test metric read or produced;
  no `state/`, prereg, config, `research-wiki/`, or live cand-2 ceremony artifact mutated. Committed
  on `main`, not pushed.
