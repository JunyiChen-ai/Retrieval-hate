# LITSWEEP-5 / S2 — Push the datasets themselves (HateMM > 0.88, EN revival)

FINAL literature sweep (round 5 of 5). Agent S2. Lens: can we push HateMM beyond
0.88 or revive MHC-EN, treating the datasets — not the pipeline — as the object.
ZERO GPU. Verified-only citations. Constraint box binding (no OCR / no gold in
deployed path / single-dataset own-split / no cross-seed ensembles / no closed APIs /
raw video never leaves machine / only local models, Molmo & general-audio downloads
USER-GATED).

Bottom line up front: **on legal channels we are already AT the 2025–2026 published
frontier on HateMM AND at/above it on MHC-EN.** The only published method that beats
our HateMM number uses OCR, and its own ablation shows OCR is load-bearing. That is
the load-bearing paper-value finding of this sweep. Two low-prior levers survive; no
lever priced above ~5% for the goal bar.

---

## 0. In-repo ground truth (verified; corrects task-prompt premises)

| Dataset | train | val | test | total src | positives | binary scheme |
|---|---|---|---|---|---|---|
| HateMM | 744 | 107 | 215 | 1083 (431 Hate) | Hate | Hate vs Non-Hate (csv) |
| MHC-EN | 549 | 80 | 161 | 891 src | 168/549 | `harmful_vs_normal` = {Hateful,Offensive}=1, Normal=0 |
| MHC-ZH | 579 | 78 | 149 | 897 src | 180/579 | same |

- **Task-prompt sizes were wrong.** EN is train 549 / test 161 (NOT ~2k/500); HateMM
  test is 215 clean (NOT ~430; the 427-row `gt_p10hate` is the localization subset, not
  the classification test). All house numbers are on HateMM n=215 clean.
- **Current bests (provenance `BIDIR_STAGE1_VERDICT_REVIEW.md`, job 13241 curric-LoRA):**
  HateMM val-sel **0.8775/0.8711**, final **0.8791/0.8726**. MHC-EN ~0.79–0.81 frozen.
- **EN `text` already contains the title** (`prep_mhc.py:72 build_text` = `Title + " . " +
  Transcript`); the litsweep2 "titles absent" note is about the *separate* `title` field
  the Qwen extractor reads (`title_present=0` in gt jsons) — the title *content* is folded
  into `text`, so title-scrape is a non-lever (already declined LOW/~0, litsweep2). ZH
  `text` is genuinely near-empty (median 106 chars).
- **Label granularity is richer than the deployed binary and is in-repo:** the source
  `annotation(new).json` carries a 3-class `Label` (EN: Normal 601 / Offensive 218 /
  Hateful 72; ZH: 605 / 180 / 112). The deployed task collapses Offensive+Hateful→1.

---

## 1. ANNOTATION-FILE FINDING (mandated)

**Per-annotator votes are NOT available — not in-repo, not in the public release.**

- In-repo `data/_src_Multihateclip/{English,Chinese}/annotation(new).json` keys =
  `Video_ID, Title, Transcript, Emotion, Frames_path, Audio_path, *_description, Label`.
  Only the **aggregated 3-class `Label`**. No annotator IDs, no vote counts, no
  disagreement field.
- Public MultiHateClip (arXiv 2408.03468) confirms the design: 2 annotators → 3rd on
  disagreement → expert escalation → **majority-vote final label only**. Raw per-annotator
  labels are **not released** (unlike HateXplain, which does release them). Verified via the
  paper and a targeted disagreement/soft-label search.

Consequence: the **LeWiDi / learning-with-disagreement / soft-label-from-annotators**
lineage is **blocked at the data level** — there are no per-annotator votes to train on,
in-box or out. This kills the "soft-label from annotator distribution" longshot outright.

What DOES survive is the **3-class granularity** (own-split, already-released, no new
gold): the discarded "Offensive" middle class is a legal graded-label signal. See §3-cand-1.

HateMM gold time-spans (`hate_snippet` column) ARE in-repo but are **gold annotations
inside the pipeline = vetoed** (F44), and using them changes the eval task (see §4 on the
temporal-label-noise paper). Not a legal lever.

---

## 2. HateMM SOTA TABLE — method / channels / number / legality (2023–2026, verified)

Numbers are each paper's own protocol (mostly 5-fold CV or a custom split); **not
cross-comparable at 4dp** with our n=215 clean, but the ordering and channel-legality
are what matter.

| Method (year) | Channels | HateMM acc / mF1 | In our box? | Note |
|---|---|---|---|---|
| Das 2023 (orig) | text+audio(MFCC)+video | 0.798 / 0.790 | audio dl-gated | dataset paper |
| Wang 2025b Vid+RM-FT | text+audio+video | 0.820 / 0.820 | audio dl-gated | via MM-HSD tbl |
| CMFusion (2505.12051) | Whisper-text+MFCC+ViT, gated add-fusion | 0.823 / 0.860 | audio=MFCC(=our F41) | **no OCR**; audio = MFCC-40 = eGeMAPS class we killed; gain is fusion (+1.4) not audio |
| Xiong 2024 TCE-DBF | text+audio+video | 0.849 / 0.840 | audio dl-gated | via MM-HSD tbl |
| **Koushik HCC1 (2502.07138)** | HateXplain-text + CLIP + **CLAP-audio**, late concat | **0.854 / 0.848** | **legal iff CLAP download user-gated**; **no OCR** | the one method whose audio is a *general-audio* encoder; ablation +2.9 mF1 (0.819→0.848) — but from a base **below** us |
| RAMF (2512.02743, Dec-2025) | text+audio+video + **Qwen2.5-VL-32B counterfactual reasoning** | 0.856 / 0.851 | 32B local = B2-dead; method = P5/P10-dead | **no OCR**; below us; see §4 |
| **MM-HSD (2508.20546, ACMMM25)** | transcript + audio(wav2vec2-xlsr) + video + **OCR** + Cross-Modal Attn | **0.878 / 0.874** | **OUT — uses OCR** | ablation: drop any one modality → mF1 0.815–0.845, so **OCR is load-bearing** |
| MultiHateGNN (BMVC25) | multimodal GNN | – / 0.771 | – | below field |
| **HOUSE (curric-LoRA, ours)** | LoRA-Qwen dual-stream + RGCL + kNN, **no OCR/no audio** | **0.879 / 0.873** | in-box | at/above every legal-channel method |

**Reading:** every published method that is legal-in-box (Koushik 0.848, CMFusion 0.860,
RAMF 0.851, Xiong 0.840, Wang 0.820) sits **at or below our 0.873 mF1**. The single method
above us — MM-HSD 0.874 — buys its edge with **OCR** (the exact veto), and its ablation
proves the OCR channel is load-bearing (removing a modality costs 0.03–0.06 mF1). **There
is no legal, published route to HateMM > 0.88.** HateMM is genuinely near-ceiling for our
channel set; this is the mechanistic proof that the OCR veto — not method weakness — is what
separates us from the 0.874 headline.

---

## 3. RANKED SHORTLIST (2 above threshold; no third clears prior)

### Cand-1 — 3-class "Offensive" graded soft-label (EN + ZH) — EN-revival longshot
**Prior ≥+1 either dataset ~15%; goal-bar (+3 acc & +3 mF1, 3/3, both protocols, ≥2
datasets) ~1–2%.** Rank #1 only because it is the **sole legal, non-isomorphic, multi-
dataset** lever left and is near-free to try.

- **Transplant.** Deployed target is binary {Hateful,Offensive}=1. Re-derive training
  targets from the in-repo 3-class `Label`: give "Offensive" a softer positive target
  (e.g. 0.6–0.8) than "Hateful" (1.0), Normal=0.0 — an ordinal/soft-label BCE. Uses ONLY
  own-split, already-released annotation; **eval stays binary**; no new gold channel, no
  OCR, no cross-dataset. Applies to BOTH EN and ZH (both have 3-class).
- **Why it could work.** Directly attacks the documented "systematic contamination" of the
  positive class (§4: 35% of EN hate-labeled videos are mostly non-hateful) by refusing to
  train the many borderline Offensive clips as hard 1.0s. This is a *legal, input-side*
  instantiation of the noise-robust idea F79 could only reach via head-loss.
- **Strongest failure reason.** The deployed boundary is harmful-vs-Normal; the 3-class info
  refines the *within-positive* split (Hateful vs Offensive) that we already merge — it does
  not sharpen the boundary we are scored on. F44's label-limit ("rotation not Pareto") says
  EN errors are not representation-fixable; soft-labeling relabels, it does not separate.
  Expect the same arithmetic ceiling as F79 (~1–2% goal-level).
- **Non-isomorphism.** No prior finding used the Offensive middle class as a graded target
  (F79 = mined-pair head noise; F75 = head-loss family; generic label-smoothing untried
  with the *Offensive-specific* target). Genuinely new axis = input-label granularity.
- **Minimal decisive cell.** $0 to build the soft-target files (deterministic from
  `annotation(new).json`); ~0.3 GPU-h to train the existing head on EN (and ZH) with the
  graded target, 3 seeds, both protocols, vs the current binary-target floor. Kill-switch:
  a $0 CPU pre-gate first — label-oracle upper bound of *any* monotone reweighting of
  Offensive rows on the dev fold; if the oracle is < +0.03 (B5-style), do not spend GPU.
- **USER-RULING FLAG (protocol, borderline).** Is finer-grained own-split label granularity
  in the *training loss* admissible, or does "no gold annotations in deployed path" extend
  to using extra annotation *dimensions*? My reading: legal — the class label is the
  supervised target (same as the binary), used only in the loss, never at inference; the
  veto is about time-span/target-group gold injected into the method (F44). But it is
  borderline and should be user-ratified before the GPU spend.

### Cand-2 — CLAP / general-audio-event channel on HateMM — HateMM-beyond-0.88 longshot
**Prior for a HateMM effect ~8–12%; goal-bar contribution ~2% (HateMM-only ⇒ cannot
clear the ≥2-dataset user-loop even on success).** Rank #2: strongest *literature*
evidence, but box-constrained and prior already lowered by F64.

- **The mechanism difference the lens asked for.** Our audio kills tested (a) F41 eGeMAPS
  = whole-video prosody, (b) F64 Whisper-encoder = ASR-oriented hidden states. Both are
  conditionally redundant over the ASR transcript (F41: transcript already banks spoken
  hate). Neither tests **general-audio-event** representation. CLAP / AST / BEATs are
  AudioSet-pretrained on non-speech events (gunshots, sirens, music genre, screams,
  slur-as-sound) that the ASR transcript literally cannot carry, and Whisper's ASR encoder
  actively *down-weights*. This is exactly the axis-closer our own `AUDIO_AXIS_FORENSIC_RECON`
  named ("a Whisper null must NOT close the learned-audio axis; a general-audio encoder is
  the proper closer") and that F64 left download-gated with prior lowered.
- **External evidence (verified).** Koushik HCC1 (2502.07138) Table 2b: text+visual 0.819
  → +CLAP 0.848 mF1 = **+2.9 from a general-audio encoder**, no OCR. This is the only
  published datapoint isolating a general-audio (not MFCC/prosody) contribution on HateMM.
- **Strongest failure reason.** Koushik's *post-CLAP* number (0.848) is still **below our
  current 0.873**. The +2.9 is measured over a text+visual base weaker than our LoRA-Qwen
  representation; the K-LAUD conditional-info screen (which zeroed Whisper over Z_best on
  HateMM, F64) is the exact instrument that would catch "CLAP info is already implied by our
  stronger representation." High chance of a calibrated-zero, same as Whisper. And it is
  **HateMM-only** — audio did not convert on EN/ZH (F64 all 3), so even a win is a single-
  dataset paper number, not a goal-bar move.
- **Minimal decisive cell.** Two-stage, cheapest-first: (1) **$0-GPU** — download CLAP
  (user-gated), extract embeddings on CPU/Modal-features-only, run the **K-LAUD conditional-
  info gate** vs Z_best (deployed-7168 AND strict-8960 arms) exactly as F64. Kill-switch:
  if ΔD_acc ≤ ~0 like Whisper, dead at $0, no head GPU, prior slashed → **closes the audio
  axis for good** (the "proper closer" the recon demanded). (2) only if it survives, one
  3-seed head cell on HateMM.
- **USER-RULING FLAG.** Requires a model download (CLAP/AST/BEATs) = the same user-gated
  relaxation as Molmo2-8B. Priced here, **not** proposed for spend without the gate.

### No Cand-3 above threshold.
Everything else the datasets offer is either dead or sub-threshold: per-annotator
soft-labels (no data, §1), gold time-spans / segment-trimming (vetoed + changes eval task,
§4), title-scrape (folded already / declined), resolution (F76 parked, HateMM 2.71× only),
RAMF-style 32B counterfactual reasoning (B2 + P5/P10 dead, §4). Given the FINAL-round
box-empty state, the honest output is 2 low-prior levers plus the frontier synthesis below.

---

## 4. PAPER-VALUE (verified external corroboration; 6 items)

1. **We are at legal-channel HateMM SOTA.** 0.879/0.873 ≥ Koushik 0.854/0.848, CMFusion
   0.823/0.860, RAMF 0.856/0.851, Xiong 0.849/0.840, Wang 0.820/0.820; only MM-HSD
   0.878/0.874 ties/edges us. Frame the results table as "at the frontier on legal channels."
2. **The one method above us (MM-HSD) buys it with OCR, provably.** Its ablation: drop any
   single modality → mF1 0.815–0.845 ⇒ OCR is load-bearing. This is the mechanistic
   external proof that the OCR veto — not our method — is the 0.874 gap. Strongest possible
   support for "the box is real" (F74) in the discussion/limitations.
3. **MHC-EN: we are at/above the published frontier, not lagging.** RAMF (Dec-2025) MHC-EN
   0.740/0.717; coarse video-level 0.684/0.644 (2508.04900); GPT-4V multiclass 0.63 mF1.
   Our ~0.79–0.81 acc sits above them. Independent confirmation that EN is **label-limited
   at the field ceiling** (F44/F55), not method-limited.
4. **Temporal-label-noise datapoint (2508.04900, Aug-2025):** HateMM hate videos contain
   33% non-hateful segments; MHC-EN 35% — "systematic, not random contamination." Segment-
   trimmed clips hit 98% mF1 but require gold spans (vetoed) and change the eval task —
   **not a legal method**, but a strong external quantification of the irreducible video-
   level ceiling and independent corroboration of our EN label-limit / F79 noise diagnosis.
5. **Koushik CLAP ablation (+2.9 mF1 audio over text+visual, base 0.819 < our 0.873):**
   the audio axis exists in the literature but its increment lands *below our current floor*
   — external support for F64's conditional-redundancy prediction, and the empirical basis
   for the Cand-2 K-LAUD screen.
6. **RAMF (2512.02743, Dec-2025) = concurrent work reaching a sub-frontier via the route we
   killed.** Its "hate-assumed / non-hate-assumed inference" fusion at Qwen2.5-VL-32B is our
   dead P5 (counterfactual twins) + P10 (MLLM logit fuse) family; it lands HateMM 0.851,
   EN 0.717, ZH 0.709 — below us on all three. Cite as independent evidence that VLM-
   counterfactual-reasoning fusion is a sub-frontier route (validates our P5/P10 kills, and
   that 32B does not rescue it — echoes B2).

---

## 5. Discipline notes

- All external numbers are each paper's own protocol; kept strictly out of any 4dp house
  comparison table (they are frontier context, not paired measurements). Cross-hardware /
  cross-split ⇒ ordering + channel-legality only.
- Two USER-RULING flags: (a) Cand-1 = admissibility of 3-class own-split label granularity
  in the training loss (borderline-legal, my read = legal); (b) Cand-2 = CLAP/AST/BEATs
  model download (user-gated relaxation, priced not proposed).
- Both candidates carry a $0 kill-switch that fires *before* any GPU (Cand-1 oracle pre-gate;
  Cand-2 K-LAUD conditional screen), consistent with G0-cond doctrine. Neither is priced
  above ~2% for the goal bar; both agents' independent conclusion of round 5 stands: in-box
  ≥+3-on-2-datasets is unreachable, and the datasets themselves confirm it — we already sit
  at the legal-channel frontier on HateMM and above it on EN.
