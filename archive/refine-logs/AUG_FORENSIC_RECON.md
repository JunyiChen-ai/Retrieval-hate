# AUG FORENSIC RECON — MLLM-generated training-data augmentation as a ZH-robustness lever

**Agent:** AUG forensic recon (goal_mllm_plus3). **Date:** 2026-07-18. **ZERO GPU / ZERO Modal / ZERO
test-touch / ZERO user interaction.** Reading + forensic reasoning + prereg-shaped design only. Deliverable =
this committed doc.

**Candidate (as posed).** Qwen2.5-VL-7B (local) is used as a **train-time DATA GENERATOR**: it
paraphrases/perturbs the **train-split transcripts** (optionally conditioned on frames) to synthesize extra
LoRA-SFT training examples for the encoder-SFT stage (B3/LoRA-HateMM machinery), and/or extra records for the
RGCL-head stage. The MLLM's product is *input text carrying the gold train label* (label-preserving), never a
score, never a feature at inference. Deployed inference path is byte-identical to B3
(`transcript → LoRA-adapted-Qwen encoder → align-fusion head → top-20 kNN vote`); the generator is **not** in
the deployed path (memory-free at inference). Target gap named by the tasking: the **ZH leg's fragility** —
B3/cand-2 pass ZH final-epoch only (+0.0313 marginal, seed2 sub-per-seed-bar; val-sel FAIL; F45 diagnoses the
val-sel miss as ~2pt 78-dev selection noise), the open half of the D7 dossier's branch-B (F59: "ZH-robustness
half remains unmet").

---

## BOTTOM LINE UP FRONT

**RULING: NO-GO — documented pre-kill. Do not queue GPU.** Augmentation is *not* a literal dead-list entry
(data-**generation** is a genuinely un-enumerated lever — wave-5's taxonomy enumerated example-*distribution*
and *loss* levers, not example-*synthesis*), so it clears the isomorphism bans (C3/P4/P11/TARC) honestly.
**But it inherits every structural wall of the adaptation family and is DOMINATED by an already-measured
negative (cand-2, F56):**

1. **Same adapted object → D7-encoder-class-dead.** Augmentation changes only *which/what* text the encoder
   LoRA-SFT trains on. The adapted object is still the **encoder** (Axis-B); F51's two-object closure is not
   escaped, only re-entered. Novelty is at best a narrow D7 sub-ruling, **weaker** than cand-2's coupling
   novelty (LLM-based data augmentation is a well-established 2023–25 genre → high D7 risk).
2. **≥2-dataset arithmetic (F44/F45) bars any new dataset.** Augmenting transcripts sharpens the **text
   stream only**. HateMM decides on the **image stream** (F44) → augmentation *inherits*, never *newly
   converts* it. **MHC-EN is label-limited, not representation-limited (F44)** → *input* augmentation cannot
   fix *labels*; EN stays dead. The only reachable target is **ZH-hardening** — exactly the gap posed, and it
   adds **no** dataset the generic/cand-2 lever lacks.
3. **The one live leg (ZH-hardening) is un-addressable on both halves.**
   - *val-sel half:* F45 diagnosed the ZH val-sel FAIL as **78-dev SELECTION noise** (dev plateaus ~ep19,
     test climbs to ep29; argmax undershoots). Train augmentation adds **zero** dev items and cannot change
     dev-argmax selection — it attacks the wrong failure mode. **Not addressable even in principle.**
   - *final-epoch half:* to strengthen requires lifting seed2 over the per-seed bar (B3 seed2 sub-bar;
     cand-2 seed2 +0.0134 sub-bar) — tiny-split (n=579) variance. **The nearest measured sibling already
     failed here:** cand-2 (retrieval-mined hard-neg SFT *curriculum*, a data-distribution intervention on
     the identical ZH encoder-SFT aimed at the identical ZH-hardening) returned **K-C2-2 TIE both protocols,
     "ZH-robustness NOT strengthened"** (F56, `546acc5`).
4. **No cheap $0 screen exists.** Augmentation's mechanism is SFT **regularization/invariance** (variance
   reduction in the *learned* encoder), which by construction adds **no** conditional label information — so
   the campaign's signature G0-cond conditional-info gate (which is exactly what cheaply killed CTF/APX/GIR/
   MJ) **cannot screen it**; it would require real GPU to measure. Worst cost/prior profile in the family:
   LOW prior (~5%), NO cheap gate, real SLURM cost — negative EV.

**Recommendation.** Bank as a documented pre-kill so wave-7 does not re-spend. Revisit **only** if the user
opens a D7 sub-ruling AND specifically requests a *generator-role* novelty upgrade AND accepts a prior
strictly weaker than cand-2's already-tied ZH result — i.e. a slot cand-2 already occupies with a measured
TIE. Never queue speculatively.

---

## 1. Ledger ruling — the eight bans, quoted verbatim and adjudicated

| # | ban / prior cell (scope quoted) | augmentation | verdict |
|---|---|---|---|
| 1 | **C3 gated dense-text + C3-target** — MLLM text as a **FEATURE channel** at inference (C3-nontarget "DEAD_AT_FUSION"; C3-target "real predictor ~0 … MHC anti-informative", `directions_tried`) | augmentation puts MLLM text into the **training set only**, label-preserving; the **deployed path is unchanged** (no MLLM text feature at inference). Different injection point, different object. | **NOT isomorphic** (clean) |
| 2 | **P4 schema-distill** ("redundant; formalized by AAAI25 2412.11917 ensembling") | again a *feature/distillation* cell (MLLM schema fields as inference features). Augmentation is train data, not a feature. | **NOT isomorphic** (clean) |
| 3 | **P11 / "MLLM-scores-as-training-signal"** (banned_constraints) — dead entry: "MLLM segment scores as weak-sup **training labels**" | augmentation uses MLLM as an **input-text generator**; the **label is always the gold train label** (label-preserving), never an MLLM score/regression target. A paraphrase is not a score. | **NOT covered** (verify: label wiring must stay gold — see §5 risk on frame-fill) |
| 4 | **TARC** ("regularization-only; val moves, test flat; test-touch spent") — an *additive head aux loss* on a frozen encoder | augmentation is **not a loss** and touches the **encoder-SFT data**, not the head regularizer. | **NOT isomorphic** (clean) |
| 5 | **cand-2 / F51 two-object closure** — "adaptation has exactly two adapted objects … encoder … and joint {encoder+decision}"; cand-2 = *data-**curriculum*** ("which examples, in what pairing/weighting") | augmentation **ADDS synthesized examples** — genuinely outside the enumerated *distribution/loss* levers, so **not a literal dead-list hit**. BUT the **adapted object is still the encoder** → F51's closure is **re-entered, not escaped**; F51 §0.2 arithmetic binds it identically (text-stream only). And cand-2 is the **nearest measured sibling** (§3). | **NOT a new object; DOMINATED by cand-2** |
| 6 | **Single-dataset veto** ("TRAINING DATA = single-dataset train split ONLY … no cross-dataset mixing … conservatively also bans external unlabeled-pool training") | paraphrases derive from the **same dataset's own train transcripts** via the **same base Qwen** already used as the encoder — **no external corpus, no cross-dataset mixing, no unlabeled pool**. Clears the veto (identical discipline to B3). *Double-edged:* because the source is the same train text + same base weights, augmentation injects **no new information** (§4). | **COMPLIANT** (and a low-prior tell) |
| 7 | **No-gold-in-deployed-path** | gold train labels condition the label-preserving paraphrase and are the SFT target — **training-time only**, exactly as B3. Deployed path never sees the generator or a gold annotation. | **COMPLIANT** |
| 8 | **Data boundary** (CLAUDE.md: features-only to Modal; raw video/frames never leave) | text-only paraphrase over 579 ZH transcripts and any **frame-conditioned** generation touch near-raw frames → **must run LOCALLY via SLURM**, not Modal (Modal is features-only; frames banned). Priced in §6. | **CONSTRAINT: local SLURM only** |

**Net of §1:** augmentation is *admissible* — it is not any banned feature-channel, loss, or pseudo-label
cell, and it clears the single-dataset veto and the data boundary. The kill is **not** an isomorphism kill; it
is a **structural-wall + dominated-by-cand-2** kill (§2–§4).

---

## 2. Wall A — the adapted object is still the encoder → D7-dead; novelty weaker than cand-2

Augmentation does not create a third adapted object. The pipeline still adapts the **encoder** via generative
LoRA-SFT; augmentation only changes the SFT training *set* (adds paraphrase rows). F51's finding is object-level
("**no third adapted object exists**"), and it holds: a bigger/varied SFT set is still encoder LoRA. Therefore
the D7 ruling F24 ("encoder-class levers do not satisfy the novelty clause") **covers it by default**, exactly
as it covers cand-2; the most augmentation can claim is a *narrower* D7 sub-ruling.

And that sub-ruling is **weaker than cand-2's**. cand-2's novelty hook was a *coupling* to the retrieval
memory (retrieval-mined confusable pairs — architecture-specific). Augmentation's generic form (label-preserving
paraphrase) has **no** architecture-specific hook and is a **well-established genre** — LLM/MLLM data
augmentation for text classification (paraphrase / self-instruct-style / hateful-text augmentation, 2023–25).
The only way to re-introduce an architecture hook is to *mine/condition the paraphrases on the retrieval
geometry* — but that just re-imports cand-2's coupling with a generation twist, and **cand-2 tied** (§3). So on
the novelty axis augmentation is **dominated**: same object, strictly weaker or identical hook, higher D7 risk.

---

## 3. Wall B + the dominating datum — modality-locus arithmetic and the cand-2 TIE

**Modality-locus arithmetic (F44/F45), re-verified this recon against the decomposition:**
- LoRA (and hence any SFT-data intervention) moves the **text stream only** — F45: ZH text-LOO AUC
  0.802(CLIP)→0.847(frozen)→0.925(LoRA), **image untouched −0.007**.
- **HateMM decides on the image stream** (F44 image train-LOO AUC 0.826, highest of three; frozen-swap Pareto
  is image-grounded; F58 confirms the HateMM pass is text-carried *but frozen-sufficient* — LoRA inherits).
  A text-augmentation lever **inherits** HateMM's pass, adds ~0 → **no new dataset here**.
- **MHC-EN is label-limited** (F44: net −1 rotation; image stream collapsed 0.734→0.599; F50 even the best-ever
  fusion AUC 0.898 is unconvertible). *Input* augmentation cannot manufacture *label* signal → **EN dead**
  (stated plainly per tasking; augmentation must not promise EN).

⇒ The **only** reachable target is **ZH-hardening**, and it adds no dataset generic/cand-2 LoRA lacks.

**The dominating datum — cand-2 (F56, `546acc5`) already measured this exact slot:**

> ZH-curric val-sel 0.8255/0.7947 (**K-C2-1 FAIL** vs generic 0.8322), final 0.8523/0.8249 (HOLD marginal,
> **seed2 +0.0134 sub-bar**); **K-C2-2 TIE both protocols ⇒ "generic LoRA with reshuffled data" on the primary
> leg; ZH-robustness NOT strengthened (both clause tests fail).**

cand-2 is a **data-distribution intervention on the identical ZH encoder-SFT (n=579), aimed at the identical
ZH-hardening target**, using a *stronger* (architecture-coupled) hook than generic augmentation. It **tied**.
Augmentation is a *sibling* data intervention on the same object / same leg / same 579-item split / same ~2pt
val-sel noise wall, with a *weaker* novelty hook and **no cheaper screen**. The honest prior that a second,
weaker-novelty data trick breaks the wall cand-2 hit is **LOW (~5%)**.

**Why neither half of the ZH gap is reachable:**
- **val-sel half is the wrong failure mode.** F45 attributes the ZH val-sel FAIL to **78-dev selection noise**
  (dev plateaus ~ep19 while test climbs to ep29; LoRA is already the *most stable* arm). The failure lives in
  **dev-set argmax selection**, not in representation quality. Augmenting the **train** split adds no dev
  items and does not change which epoch dev-argmax picks. There is a thin second-order steelman (a flatter,
  regularized trajectory could make dev-argmax and test-optimal coincide better) — but it is speculative,
  and the direct measured intervention (cand-2) did **not** move val-sel (K-C2-1 FAIL vs generic). So the
  "val-sel conjunct passes" half is **not addressable in principle by train augmentation**.
- **final-epoch half is bounded below by the cand-2 TIE.** Strengthening final-epoch means lifting the
  sub-bar seed to a solid 3/3; cand-2's seed2 stayed sub-bar (+0.0134). Same variance source, already tied.

---

## 4. Wall C — no $0 pre-GPU gate exists (worst cost/prior profile in the family)

The campaign's cheap kills (CTF F39, APX F41, GIR F43, MJ F49) all fired a **G0-cond conditional-info gate**:
they measured `I(label; candidate-feature | Z_best)` on banked caches for ~$0 and killed at calibrated-zero.
**That gate does not apply to augmentation.** Augmentation is *label-preserving* and adds **no** conditional
label information by design (§1 row 6 — its content is a function of the same train text + same base model);
its entire mechanism is **training-dynamics regularization / input-invariance in the learned encoder**, which
is unobservable without actually running the SFT. A tempting cheap check — "does the frozen encoder's feature
of a paraphrase move?" — is the **wrong** probe: a good encoder maps a faithful paraphrase to a near-duplicate
vector, so frozen-feature movement is near-zero *even when* the augmentation would help the *trained* model
(and large movement would signal an *unfaithful/label-drifting* paraphrase — a defect, not a signal). Hence:
- The only $0 checks available are **negative screens** (auto-KILL if the augmentation is a near-no-op):
  paraphrase-diversity / label-fidelity (see K-AUG-0, §5). They can **only** kill, never promote.
- Any actual *effect* measurement requires **real GPU** (SFT + head + 3-seed), at ~cand-2 cost.

Net: **LOW prior, NO promoting cheap gate, real SLURM cost** = the worst cost/prior profile of the whole
adaptation family. This is the decisive practical reason it should not be queued.

---

## 5. Design (recorded for completeness / reserve only) + kill-switches

Three label-preserving constructions, all **train-only, single-dataset, gold-labelled, local SLURM**:
- **A1 · label-preserving transcript paraphrase ×k.** For each ZH train video, generate k∈{1..4} paraphrases
  of its transcript (text-only Qwen; keep gold label). Adds input variance; regularizer.
- **A2 · confusable-adjacent paraphrase.** Paraphrase toward the mined same-community opposite-label boundary
  (retrieval-aware) — re-imports cand-2's coupling; dominated by the cand-2 TIE, no independent value.
- **A3 · empty-transcript fill from frames.** Frame-conditioned caption for videos with empty/near-empty ASR.
  **Highest-risk:** it manufactures *content*; the label must remain the **gold** train label, never
  MLLM-inferred, or it crosses into P11/pseudo-label territory. Also the priciest (frame loading, local only).

**Leakage discipline (mandatory if ever run):** the generator sees **train transcripts/frames only** —
**never** any dev or test text/frame, and the mined pairs (A2) come only from the own-train kNN. Dev/test
stay untouched; deployed path unchanged.

**Kill-switches (house style):**
- **K-AUG-0 ($0 CPU pre-GPU, kill-only):** the augmentation must be non-trivial *and* label-faithful — (a)
  paraphrase-diversity: mean pairwise cosine(orig, paraphrase) in the **frozen** encoder text space must be
  **< ~0.98** (else near-duplicates ⇒ curriculum ≡ B3 ⇒ auto-KILL "not a distinct method"); (b) label-
  fidelity guard: a paraphrase whose frozen text-feature crosses the LOO decision boundary is a label-drift
  defect ⇒ discard. Note: passing K-AUG-0 is **necessary not sufficient** — it cannot show the *trained*
  effect (§4).
- **K-AUG-1 (primary, ZH-robustness clause):** ZH LoRA-aug − CLIP paired **AND** ≥ generic-B3 − 0.014, judged
  per protocol; the earning bar is the **val-sel conjunct crossing** OR **final-epoch 3/3 above per-seed bar**
  — i.e. strictly *strengthen* what cand-2 tied. Anything ≤ cand-2 (TIE) ⇒ no value.
- **K-AUG-2 (hold-HateMM sanity):** HateMM must not regress below its inherited pass.
- **K-AUG-3 (add-over-cand-2):** must beat **cand-2** (not just generic) on ≥1 protocol, else it is dominated
  and banked negative.

---

## 6. Cost / sequencing

- **Generation:** text-only paraphrase of 579 ZH transcripts × k≤4 ≈ ~0.5–1 A100-h (short generations);
  frame-conditioned (A3) more, still modest. **Local SLURM only** (Modal features-only; frames banned).
- **SFT + extract + head:** ~1 LoRA-SFT run/dataset (~3.5–4 h) + ~0.4 h extract + ~2 min head × 3 head-seeds,
  matching one cand-2/LoRA-HateMM run. **Total ≈ 7–9 A100-h per dataset**, queue-blocked behind any live
  adaptation chain.
- **Sequencing:** do **not** queue. It cannot add a dataset (§3), cannot be cheaply screened (§4), and is
  dominated by cand-2 (F56). It belongs only in a **user-opened D7 generator-role sub-ruling** as a reserve
  outline, never speculatively — and even then it is strictly weaker than the already-run cand-2.

---

## 7. Novelty note (no claim)

MLLM-in-the-loop data augmentation is a **known genre** (LLM paraphrase / self-instruct-style / hateful-text
augmentation). Its only project-distinct object would be the **generator role + memory-free train-time
integration** (MLLM never in the deployed path). That framing is honest but thin, and **higher D7 risk** than
cand-2's retrieval-coupling. No novelty claim is made or should be made without a user D7 sub-ruling; and even
with one, the performance ceiling is the cand-2 TIE.

---

## PROVENANCE
- Adapted-object two-object closure + data-curriculum lever taxonomy: `WAVE5_CANDIDATES.md` (`7166232`, F51);
  cand-2 recon `CAND2_CURRICULUM_RECON.md` (F52).
- cand-2 ZH TIE / "ZH-robustness NOT strengthened": `CAND2_VERDICT_REVIEW.md` (`546acc5`, F56) §2.1/§7; rep2
  `CAND2_REP2_VERDICT_REVIEW.md` (F59, ZH-robustness half unmet).
- Modality-locus ≥2-dataset arithmetic: F44 `ENCODER_SWAP_DIAGNOSIS.md` (`8a48938`), F45
  `B3_ZH_LORA_DECOMPOSITION.md` (`d76e407`; text 0.847→0.925, image −0.007, val-sel = 78-dev selection noise),
  F58 `HATEMM_LORA_STREAM_DECOMP.md` (HateMM text-carried, frozen-sufficient), F50 `FA_GATE_RECORD.md` (EN
  fusion AUC 0.898 unconvertible / label-limited).
- Ban scope quotes: P11 / MLLM-scores-as-training-signal / pseudo-label ("representation-training expansion
  only") / single-dataset veto / TARC / C3 — `state/directions_tried.json` (dead[] + banned_constraints[]).
- $0 G0-cond gate precedent (why it does NOT apply here): CTF F39, APX F41, GIR F43, MJ F49.
- ZH train split n=579 verified: `data/lora_sft/MHC_zh/train.json` (+ `train_yn.json`, `train_curric.json`).
- **Required statements:** ZERO GPU / SLURM / Modal spent by this recon; no held-out test metric read or
  produced; no `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated. Committed on `main`,
  not pushed.
