# HATEMM LoRA STREAM DECOMPOSITION — is the passing cell image-inherited, image-MOVED, or text-driven?

**Author:** per-stream decomposition analyst (**ZERO GPU / ZERO Modal / ZERO test-touch**; banked
train/dev feature caches + banked completed-run trainlogs only — **no re-extraction**, **no new test
evaluation**; test numbers read from already-logged per-epoch `Test_Retrieval` lines, exactly as
`B3_VERDICT_REVIEW.md` / `LORA_HATEMM_VERDICT_REVIEW.md` were written). **Date:** 2026-07-18. Does
**not** touch `research-wiki/` (paper-integrator live), `state/`, prereg, or any frozen artifact.
**Feeds:** analysis-chapter §3.9 (firm-or-errata) and premise-(a)'s vision-SFT prior (informational).

**Design source:** `WAVE6_PREMISE_HUNT.md` surface-4 (`c53cfe1`). **Method template:**
`B3_ZH_LORA_DECOMPOSITION.md` (F45, `d76e407`) — this doc replicates F45's ZH machinery on HateMM.
**Anchors:** `ENCODER_SWAP_DIAGNOSIS.md` (F44, `8a48938`, HateMM Pareto / MHC collapse reference),
`LORA_HATEMM_VERDICT_REVIEW.md` (F53, `6b8f634`, HateMM PASS both protocols), `TIE_BRANCH_RECON.md`
(F54, `6b9985a`, "image stream architecturally movable").

**Cell under decomposition:** encoder-level LoRA-Qwen vs frozen-Qwen vs frozen-CLIP on **HateMM**,
the round-4 PASS (F53: val-sel +0.0419 acc / +0.0460 mF1, final-epoch +0.0573 / +0.0682, 3/3 sign
each). F45 decomposed only ZH; §3.9 asserts HateMM's mechanism *by inference from F44/F45*. This
measures it.

**Scripts (committed, reproduce every number):** `scripts/analysis/hatemm_lora_stream_decomp.py`
(reuses `scripts/analysis/encoder_swap_geometry.py` verbatim for the kNN geometry; embeds the F45
banked-trainlog parser for Pareto-vs-rotation) → `..._out.json`.

---

## PRE-DECLARED CLASSIFICATION RULE (design-locked BEFORE any LoRA delta was seen)

Locked in the script docstring before running; at lock time only F44's *frozen*-Qwen numbers were
known — the LoRA image ΔAUC (the crux) was **not**. Let `dAUC_s = AUC_s(LoRA) − AUC_s(frozen-Qwen)`
per stream `s ∈ {img, text}`, on two footings: train-LOO kNN AUC and held-out dev kNN AUC.

- **Stream MOVED** iff `dAUC_s ≥ +0.010` train-LOO **and** `≥ +0.005` dev (same + sign); **FLAT**
  iff `|dAUC_s| < 0.010` train-LOO; **DEGRADED** iff `dAUC_s ≤ −0.010`. (+0.010 is the resolution
  floor: ZH's image moved −0.007/−0.007 ⇒ FLAT under this same rule, matching F45 — ZH-calibrated.)
- **Decisive modality** = the stream with the higher **standalone** kNN AUC, required to agree on
  both footings.
- **Top-level label** (decision tree, top-down): **(B) IMAGE-MOVED** iff image MOVED both footings;
  **(C) TEXT-DRIVEN** iff image FLAT/DEGRADED, text MOVED, and LoRA beats frozen-Qwen downstream by
  `≥ +0.010` acc (banked); **(A) IMAGE-INHERITED** iff image FLAT, decisive-modality == image, and
  LoRA ≈ frozen-Qwen downstream (`|Δacc| < 0.010`); **else MIXED / REPORT-RAW** (no forced label).
- **Pareto** (convertible) iff `Δhate_recall > 0` and `Δnonhate_recall ≥ −0.010`; **rotation**
  otherwise-with-hate-gain.
- **Machinery-validity gate (kill bar i):** the concat dev read-out must reproduce the banked
  downstream **sign** — `dev concat AUC(frozen−CLIP) > 0`. **Result: +0.0458 > 0 → PASS** (matches
  F44's dev +0.047; the geometry substrate is valid).

---

## TOP-LINE (one paragraph)

**The HateMM LoRA pass is neither image-MOVED nor image-inherited-as-§3.9-states-it; it is
TEXT-CARRIED, frozen-swap-SUFFICIENT, and LoRA-INHERITED.** Three measured facts: **(1) the image
stream did NOT move under LoRA** — `dAUC_img(LoRA−frozen) = +0.0045` train-LOO / `+0.0062` dev, both
sub-threshold FLAT — so **F54's "image-MOVED" possibility does not materialize** on the passing cell
(the faint +0.6pt dev nudge is a whisper of the `lora_target: all` backbone re-processing the
vision-pad tokens, but it is nowhere near material). **(2) The decisive modality on HateMM is TEXT,
not image** — text-only kNN AUC ≥ image-only for **all three encoders on both footings** (CLIP
0.847/0.837 ≥ 0.826/0.806; frozen 0.888/0.875 ≥ 0.817/0.807; LoRA 0.920/0.899 ≥ 0.821/0.814), so
"HateMM *decides* on the image stream (image-only train-LOO AUC 0.826)" is a **misreading**: 0.826
is CLIP's *image* AUC and it is *below* CLIP's *text* AUC 0.847. **(3) The text stream DID move under
LoRA** (`dAUC_text = +0.0317` train / `+0.0236` dev, the ZH signature) — but that sharpening **adds
≈ 0 downstream** (`LoRA − frozen-Qwen = +0.0015` acc final-epoch, **−0.0108** val-selected) **because
the frozen swap already converted HateMM** (`frozen − CLIP = +0.0558` acc, a clean Pareto). So HateMM
is the mirror image of ZH: **in ZH the frozen swap FAILS and LoRA is the converting lever (text-borne,
LoRA-specific); in HateMM the frozen swap already SUCCEEDS and LoRA's further text-sharpening is
redundant (text-borne, frozen-sufficient, inherited).** The pre-declared tree correctly refused to
label this "image-inherited" (its decisive-modality premise is false) and returned **MIXED /
REPORT-RAW**. **Verdict for the paper: §3.9's "inherited from the frozen swap (LoRA ≈ frozen-Qwen)"
half is CONFIRMED; its "image-borne / decides on the image stream / text is HateMM's secondary
modality" half is REFUTED and needs errata** — and the errata brings §3.9 back into agreement with
§3.6's own already-correct phrasing ("Qwen's uniformly better text stream rides on a neutral-strong
image stream").

---

## Q1 — WHERE does the pass live? TEXT stream (on a strong, swap-neutral image base) — NOT image

Per-stream kNN AUC, three encoders on HateMM (train-LOO, n=744; held-out dev, n=107):

| stream | CLIP tr | frozen tr | LoRA tr | LoRA−froz (tr) | CLIP dv | frozen dv | LoRA dv | LoRA−froz (dv) |
|---|---|---|---|---|---|---|---|---|
| **image** | 0.826 | 0.817 | 0.821 | **+0.0045** | 0.806 | 0.807 | 0.814 | **+0.0062** |
| **text** | 0.847 | 0.888 | **0.920** | **+0.0317** | 0.837 | 0.875 | **0.899** | **+0.0236** |
| concat | 0.867 | 0.883 | 0.909 | +0.0257 | 0.863 | 0.909 | 0.910 | +0.0010 |

- **Image stream = FLAT under LoRA** (train-LOO +0.0045, dev +0.0062; both `< 0.010`). LoRA does
  **not** materially move the vision-token-derived stream. This directly **answers F54's open
  question**: the HateMM SFT did *not* move the image stream — the architectural movability F54
  identified stays latent under a text-decodable yes/no target. (The faint *positive* nudge, unlike
  ZH's −0.007, is the only trace of the movable backbone; sub-threshold, downstream-immaterial.)
- **Text stream = MOVED under LoRA** (train-LOO **+0.0317**, dev **+0.0236**) — the **same
  mechanism as ZH** (ZH text moved +0.078 train). LoRA sharpens the language representation on
  HateMM too. The held-out-dev confirmation (+0.0236) rules out a train-only "LoRA saw train"
  inflation: the text move holds on data LoRA never adapted on.
- **The decisive single modality is TEXT for every encoder, both footings** (text ≥ image in all six
  cells above). HateMM's classification does **not** ride primarily on the image stream. What F44
  actually established — and what is true here — is that image is **strong and swap-neutral** (0.82
  band for all three; *not* collapsed like MHC-EN's 0.599), which is the **enabling** condition that
  lets the fused decision convert; the **driving** stream is text. §3.6 already states this correctly
  ("Qwen's uniformly better text stream rides on a neutral-strong image stream"); §3.9 compresses it
  into the false "decides on the image stream."
- **The pass over CLIP is carried in the text stream's AUC ladder:** text-only train-LOO
  0.847 → 0.888 → 0.920 (CLIP → frozen → LoRA), a monotone +0.073 climb; image-only barely moves
  across the same ladder (0.826 → 0.817 → 0.821). The separation that the fusion converts to accuracy
  accumulates in **text**, on top of a flat-strong image base.

**Answer:** the HateMM pass lives in a **sharpened text/transcript representation fused with a
strong, swap-neutral image stream** — **text-carried, not image-borne**, and the decisive single
modality is text, not image.

---

## Q2 — Pareto or rotation? The *pass* (LoRA−CLIP) is Pareto; LoRA's *increment over frozen* is nil

Final-epoch **TEST** per-class recall (3-seed mean, banked `Test_Retrieval` lines; minority = hate,
test pos-rate ≈ 0.30):

| arm | test acc | test mF1 | hate recall | non-hate recall | vs → | Δacc | Δhate-rec | Δnonhate-rec | shape |
|---|---|---|---|---|---|---|---|---|---|
| CLIP | 0.8124 | 0.7936 | 0.6395 | 0.9277 | — | — | — | — | baseline |
| **frozen-Qwen** | 0.8682 | 0.8591 | 0.7674 | 0.9354 | −CLIP | **+0.0558** | +0.1279 | +0.0078 | **PARETO** |
| **LoRA-Qwen** | 0.8698 | 0.8618 | 0.7868 | 0.9251 | −CLIP | **+0.0573** | +0.1473 | −0.0026 | **PARETO** |
| " | " | " | " | " | −frozen | **+0.0015** | +0.0194 | −0.0103 | ROTATION (nil) |

- **frozen-Qwen already converts HateMM Pareto** (+0.0558 acc; hate recall +0.128 at +0.008 non-hate
  *cost-free*). This is F44's finding, reproduced bit-exact. **The conversion is done at the frozen
  level.**
- **LoRA-Qwen over CLIP is Pareto** (+0.0573 acc; hate +0.147 at −0.003 non-hate) — the banked pass,
  convertible, 3/3 seeds hate-recall up (+0.163/+0.174/+0.105), non-hate cost 2/3 non-negative
  (−0.031/0.000/+0.023, mean −0.003).
- **LoRA-Qwen over frozen-Qwen is a nil rotation** (+0.0015 acc; hate +0.019 bought with −0.010
  non-hate). The extra text-sharpening LoRA does *on top of frozen* re-ranks a few boundary items
  but nets ≈ 0 — **val-selected it is even slightly negative (−0.0108 acc)**. Downstream, LoRA's text
  move is past the point of diminishing returns: frozen already moved the decision boundary HateMM
  needed.

**Answer:** the HateMM *pass* (vs CLIP) is a genuine Pareto minority-recall conversion — but it is
**inherited from the frozen swap**, not manufactured by LoRA. LoRA's marginal contribution over
frozen is a nil rotation. This is the crux distinction from ZH, where the frozen swap is itself a
*rotation* (−0.0112, unconvertible) and **LoRA is what crosses it to Pareto**. **Same lever, opposite
role:** ZH's conversion is LoRA-specific; HateMM's is frozen-sufficient and LoRA-inherited.

> **One line:** HateMM's text signal is already convertible at frozen-Qwen (fused with strong-neutral
> image ⇒ Pareto +0.056); LoRA sharpens text further (0.888 → 0.920 AUC) but there is no further
> boundary to move, so it adds ≈ 0. ZH's text signal is *not* convertible at frozen (rotation
> −0.011); only LoRA's larger text edge (to 0.925) re-decides it.

---

## Q3 — F54 image-movability + empty-transcript control

- **F54 ("image architecturally movable") — RESOLVED, does not materialize downstream.** The image
  stream nudged **+0.0045 train / +0.0062 dev** under LoRA — a *positive* drift (ZH's was −0.007),
  the faint signature of the `lora_target: all` backbone re-contextualising the vision-pad tokens
  the pooled `img_feats` are read from. But it is **sub-threshold (< 0.010) and downstream-null**:
  the image AUC ladder (0.826/0.817/0.821 train) shows no material LoRA movement, and the pass is
  text-carried regardless. **F54's scoping correction is CONFIRMED exactly as written** — the vision
  path *is* movable, but "stays flat only because every SFT target is a text-decodable yes/no with
  the transcript present." No errata to the §3.9 scoping paragraph (lines 439–450); it stands.
- **Empty-transcript subgroup — non-factor on HateMM.** F44 flagged ~5.6% degenerate *CLIP* text
  embeddings; the **Qwen** encoder emits a (near-)zero text vector for only **1/744 train, 0/107
  dev** (near-zero-norm), so HateMM dev has effectively no empty-transcript queries. Restricting dev
  to full-transcript queries leaves the per-encoder text/concat AUC ordering unchanged (text: CLIP
  0.838 / frozen 0.875 / LoRA 0.899; concat: 0.863 / 0.909 / 0.910). The text-stream localization is
  not an empty-transcript artifact.

---

## SYNTHESIS — §3.9 errata (exact sentences)

**CONFIRMED (no change):** the *inheritance* claim — "LoRA inherits and preserves frozen-Qwen's …
conversion," "LoRA ≈ frozen-Qwen there," "adds ≈ 0 on top" — is **measured true** (LoRA−frozen
downstream +0.0015 final / −0.0108 val-sel; text sharpening redundant because frozen already
converted). The KS-2 family-coherence read (LoRA matches frozen-Qwen, honesty flag does not trip) is
correct. The §3.9 scoping-correction paragraph (F54) is correct and stands.

**REFUTED (errata needed):** the *locus* claim — that the pass is **image-borne** and that HateMM
**decides on the image stream** with text as its **secondary** modality. Measured: text is the
**decisive** single modality (text-only AUC ≥ image-only for all three encoders, both footings) and
the pass is **text-carried** on a strong swap-neutral image base. The exact draft sentences to amend
(`research-wiki/DRAFT_analysis_chapter.md`, §3.9 and cross-refs):

1. **L393–396** — "HateMM *decides* on the image stream (image-only train-LOO AUC 0.826), which
   LoRA leaves intact, so LoRA inherits and preserves frozen-Qwen's **image-borne** Pareto
   conversion; the text stream it does sharpen is HateMM's **secondary** modality, so it adds ≈ 0 on
   top." → **image AUC 0.826 is CLIP's and is *below* CLIP's text AUC 0.847; text ≥ image for all
   encoders.** Recommended: *"HateMM's image stream is strong and swap-neutral (image-only train-LOO
   AUC 0.82 band, uncollapsed unlike MHC-EN), and LoRA leaves it flat (ΔAUC +0.005/+0.006); the
   decisive single stream is actually text (text-only ≥ image-only for CLIP/frozen/LoRA). LoRA
   sharpens that text stream (train-LOO 0.888 → 0.920) but adds ≈ 0 downstream (+0.0015 final /
   −0.0108 val-sel) **because the frozen swap already converted HateMM's text signal to Pareto** —
   there is no further boundary to move."*
2. **L397–398** — "HateMM's **image-borne** and inherited from the frozen swap (LoRA ≈ frozen-Qwen
   there)." → keep "inherited from the frozen swap (LoRA ≈ frozen-Qwen)"; change **"image-borne"** to
   **"text-carried on a swap-neutral image base, and frozen-swap-sufficient."**
3. **L424** — "HateMM is inherited **image-borne**" → "HateMM is inherited (frozen-swap-sufficient),
   its convertible signal **text-carried**."
4. **L434–435** — "HateMM's pass is inherited **image-borne** (LoRA ≈ frozen-Qwen, the adapted text
   stream is HateMM's **secondary** modality)" → "HateMM's pass is inherited from the frozen swap
   (LoRA ≈ frozen-Qwen); its convertible signal is **text-carried** (text is the decisive single
   stream), fused with a strong swap-neutral image stream — the frozen swap already converts it, so
   LoRA's further text-sharpening adds ≈ 0."

**Net effect on the paper's argument:** *strengthens* it. The errata makes §3.9 **consistent with
§3.6** (which already says text rides on a neutral-strong image stream) and **sharpens the ZH↔HateMM
contrast** into a cleaner law: *both passes are text-carried; the difference is whether the frozen
swap already converts the text signal (HateMM: yes ⇒ LoRA inherits) or not (ZH: no ⇒ LoRA is the
converting lever).* Structural-law IV's headline — "convertibility runs through adaptation, not
encoder identity" — is **unaffected for ZH** (where it is exactly right) and needs a **one-clause
qualifier for HateMM** (there the *identity* swap already converts; adaptation only inherits). The
three-dataset phase-diagram entries are otherwise unchanged: HateMM PASS (now correctly "frozen-swap
Pareto, text-carried, LoRA-inherited"), ZH PASS (text-borne, LoRA-specific), EN FAIL (label-limited,
collapsed image).

**Implication for premise-(a) (vision-SFT prior) — informational only, no new direction proposed.**
The passing cell provides **no encouragement** for a vision-targeted SFT: on the one cell where the
`lora_target: all` backbone *could* have moved the image stream, it moved it **+0.6pt (sub-threshold,
downstream-null)** — the image path stays inert under a text-decodable objective even when
architecturally reachable. Combined with F50/F55 already pricing EN's *healthy* image ceiling below
the oracle bar, premise-(a)'s prior stays **low** (unchanged from its armchair ~5–8%; if anything the
observed +0.6pt inertia nudges it down, not up). This is a diagnostic note for the orchestrator, not
a proposed cell.

---

## Provenance / reproduction

- **Feature caches (read-only, no re-extract):** `data/CLIP_Embedding/HateMM/{train,dev_seen}_{openai_clip-vit-large-patch14-336_HF, Qwen2.5-VL-7B-Instruct_HF, Qwen2.5-VL-7B-Instruct-LoRA_HF}.pt`.
  IDs + labels verified **identical** across all three encoders (train n=744 pos 0.4005, dev n=107 pos
  0.4019). LoRA = Jul-18 job-13234 extraction (the enc3seed runner consumed it for job 13235).
  sha256 (train/dev per encoder): CLIP `0802b6ba…`/`ab9cd8a0…`; frozen-Qwen `ba52bc0d…`/`1b219e12…`;
  LoRA `31bcc402…`/`5147120d…`.
- **Banked trainlogs (read-only, per-epoch Val/Test already logged during completed runs — NO new
  test evaluation):** `slurm/logs/enc3s_HateMM_{tag}_seed{0,1,2}_{12850|13235}.trainlog` (CLIP +
  frozen-Qwen job 12850; LoRA job 13235). Non-hate recall derived `= 2·macroR − hate_recall`
  (cross-checked vs logged acc). Parse reproduces the banked verdict **bit-exact**: LoRA−CLIP
  final-epoch +0.0573 acc / +0.0682 mF1, val-sel +0.0419 / +0.0460; LoRA 0.8698 ≥ frozen-Qwen 0.8682
  final-epoch (F53 / §3.9). sha256 e.g. LoRA seed0 `f1a43bcc…`, CLIP seed0 `dd9ad86e…`.
- **Machinery-validity gate PASSED:** dev concat AUC(frozen−CLIP) = +0.0458 > 0 (matches F44's dev
  +0.047 → the geometry substrate reproduces the banked downstream sign).
- **Scripts:** `scripts/analysis/hatemm_lora_stream_decomp.py` (+ `..._out.json`),
  reusing `scripts/analysis/encoder_swap_geometry.py`. conda `HateVideo`, CPU, seconds.
- **Required statements:** no NEW held-out test evaluation was run; test metrics read from banked
  completed-run trainlogs (same provenance discipline as F45/`B3_VERDICT_REVIEW` /
  `LORA_HATEMM_VERDICT_REVIEW`). Gold read = train + dev `labels` (geometry) and banked logged test
  metrics (diagnostic). No `state/`, prereg, config, `research-wiki/`, or frozen artifact mutated.
  Committed on `main`, not pushed.
