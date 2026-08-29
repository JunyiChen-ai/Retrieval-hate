# LITSWEEP-2 — the INPUT-FIDELITY axis (what raw information reaches the encoder)

**Agent:** literature-sweep ROUND-2 #2. **Date:** 2026-07-25. **Mode:** CPU-only, zero GPU/SLURM/Modal;
WebSearch/WebFetch + code inspection + `$0` CPU data audits on banked caches/jsons. `autoresearch/…/state/`
untouched. Nothing pushed.

**Charge.** The goal is a *substantial* gain (≥+3). Input fidelity — the raw information that actually reaches
the frozen Qwen2.5-VL encoder — has been varied on **exactly one** dimension so far: frame COUNT (8→16),
killed at **F67 / FRAME16** because the pooled representation saturates over near-duplicate frames. This note
enumerates the OTHER input dimensions, audits our real settings/data, and grades each against verified
literature and our text-carried regime.

---

## 0. Verified current settings (code citations)

Workhorse extractor `src/utils/generate_VideoMLLM_embedding_HF.py` (the `…_HF` caches every formal head-run
consumes; the `lora/readout/bidir` variants share the sampler + prompt verbatim):

| dimension | current value | code |
|---|---|---|
| **Spatial resolution cap** | `max_pixels = 360*420 = 151200` px/frame (~389² ; set at **processor construction**, `__call__` ignores it under transformers 4.49) | `:97-101`, `:403-404`, comment `:268-269` |
| **Frame count** | `num_frames = 8` | `:90-95` |
| **Frame selection** | uniform `np.linspace(0, N-1, 8)`, rounded — **no RNG, no content signal** | `:146-152` |
| **Transcript** | full `text` field concatenated after `"\nTranscript: "` — **no `truncation=`, no `max_length=`, no token cap** in the `processor(...)` call | `:270-275`, `:349-355` |
| **Title** | `"\nTitle: " + (title or "(none)")` — read via `obj.get("title","")` | `:136-137`, `:349-354` |
| **Readout** | img = **mean-pool** over visual+instruction span; text = **last-token** of assistant header | `:290-318` |

**Native encoder capability (on-disk config, per `FRAME_BUDGET_FORENSIC_RECON.md:56,65-67`):**
`patch_size=14`, `spatial_merge_size=2`, `temporal_patch_size=2`; 1 visual token = a 28×28-px merged patch;
~720–768 visual tokens total @8f; the forward has **no `cutoff_len` / no truncation — only VRAM bounds it.**
Qwen2.5-VL's ViT is *native-dynamic-resolution* and supports millions of px/frame; **our 151200 cap is a
memory knob, ~1/6 of a 720p frame.**

---

## 1. `$0` DATA AUDITS (banked caches + gt jsons; primary evidence, no cloud)

**(A) The resolution cap is BINDING — median video is aggressively downscaled.**
From `data/CLIP_Embedding/{HateMM,MHC}/frameset_qwen7b_8f/*_frameset.pt` (`grid_thw`, the real per-video vision
grid the ViT used, 14-px patch units):

| ds / split | median grid h×w (patches) | median processed px-area | cap | % frames ≥95% of cap |
|---|---|---|---|---|
| HateMM test | 20×36 → **280×504 px** | 141 120 | 151 200 | 11% |
| HateMM dev | 20×34 | 141 120 | 151 200 | 13% |
| MHC test | 36×20 → **504×280 px** (portrait) | 141 120 | 151 200 | 3% |
| MHC dev | 36×20 | 141 120 | 151 200 | 2% |

The median video is resized to **~141 K px (≈93% of the cap)** — i.e. the cap is doing real work: a native
≈720p frame (≈920 K px) is downscaled **~6.5× in area (~2.5× per side)** to fit. Raising `max_pixels` puts
genuinely new pixels in front of the encoder (on-screen text, small symbols, facial detail) — this is **not**
adding redundant frames.

**(B) Transcript coverage is COMPLETE; title is ABSENT from source; ZH text is near-empty.**
gt jsons carry only `{id, text, label}` — **`title_present = 0` in every split of HateMM/MHC/MHC_zh** (so the
extractor's `Title:` line is *always* `(none)` — no title is being lost, there simply is none), and
`text_empty = 0` (no missing transcripts). Text length (words):

| ds | median | p90 | max | note |
|---|---|---|---|---|
| HateMM train | 128 | 580 | **13 677** | heavy long tail |
| MHC (EN) train | 69 | 184 | 273 | moderate |
| MHC_zh train | **4** | 5 | 133 | ~~**near-empty**~~ — **WITHDRAWN, see erratum below** |

> **ERRATUM 2026-07-28 (propagated from `LITSWEEP3_ZH_SPECIFIC.md:18-26,29-34`, F77 / commit `d4af64b`).**
> The MHC_zh row of this table is a **whitespace-split artefact and is withdrawn.** Chinese has no
> inter-word spaces, so `text.split()` returns ~1 token per punctuation-delimited run; the *character*
> median of the deployed ZH text stream is **106 Chinese characters** (train 106 / val 108.5 / test 105).
> Second correction in the same erratum: the deployed ZH "transcript" is **not** the Whisper ASR — it is the
> **Bilibili description/metadata** field, and **42 %** of ZH train rows carry literal `<em class="keyword">…</em>`
> HTML markup baked into the key. The Whisper ASR lives in a separate, **non-deployed** file
> (`data/ASR/MHC_zh/*_asrK4_whisper-large-v3.jsonl`). Everything below that reasons from "ZH text ≈ 4 words /
> near-empty" — items (ii) here and §5 — is superseded: the ZH stream is **content-rich**, and ZH's binding
> wall is 78-dev val-selection noise plus representation saturation (LoRA-Qwen ZH text-AUC 0.925), not a
> degenerate transcript. The HateMM and MHC(EN) rows are unaffected (space-delimited languages).

No transcript hits any truncation limit (there is none in code; the longest ≈13.7 K-word HateMM item is
≈18 K tokens < Qwen's 32 K context, so even it is encoded whole). **Transcript quality/coverage is therefore
NOT a live input-fidelity gap** — ASR is Whisper-large-v3 (`data/ASR/*/*_whisper-large-v3.jsonl`, ceiling),
coverage is full, nothing is cut. Two real observations fall out, but neither is an input-fidelity fix:
(i) HateMM's 13.7 K-word transcripts are compressed into a single **last-token** embedding — a *readout*
bottleneck (belongs to the readout/head-recipe agents, not here); (ii) ~~ZH text ≈4 words means the ZH image
stream carries *more relative weight* than the "text-carried" label suggests (see §5)~~ — **WITHDRAWN by the
2026-07-28 erratum above**: ZH text is a median **106 Chinese characters** of Bilibili description metadata,
so ZH's image stream carries **no** extra relative weight on this argument.

**(C) Frames are near-redundant at the pooled-feature level (corroborates F67; weakens frame-selection).**
Mean pairwise cosine across the 4 temporal frame-groups per video (from the banked `g` tensor):

| ds | median inter-frame cos | p10 | p90 | % videos > 0.95 |
|---|---|---|---|---|
| HateMM test | **0.911** | 0.841 | 0.969 | 22% |
| MHC test | 0.905 | 0.852 | 0.946 | 7% |

Frames are similar-but-not-identical (0.91, not 0.99). This is exactly the redundancy F67 exploited to kill
*more frames*; it also means content-aware *selection* has little slack (only 7–22% of videos have truly
near-duplicate frames to prune), consistent with S2S's near-dup-excluded arm being ≈null (§2).

**(D) Decode health is clean.** Extraction zero-vector guards: HateMM = **1** undecodable video (known),
MHC = 0, MHC_zh = 0. No evidence of black/letterbox/watermark pathology at the aggregate level; the frames
are stored only as *features* (not pixels), so a per-frame luminance audit is not possible `$0`, but the
zero-guard counts and the clean grid_thw distribution show no gross corruption. **Dimension 6 (preprocessing
pathologies) = no actionable gap found.**

---

## 2. BAN-SCOPE CHECK (F67 / S2S) — does any prior kill cover these dimensions?

**F67 / FRAME16 scope (quoted, `FRAME16_VERDICT_REVIEW.md:234-241`):**
> "Doubling visual sampling **density from 8 to 16 frames** through the frozen Qwen2.5-VL-7B encoder +
> mean-pool … produces no head-level gain … the **frame-budget door** is CLOSED … the prereg's pre-declared
> honest most-likely outcome (F0.5 **dilution/redundancy**)."

The kill mechanism is **pooled-rep saturation over near-duplicate FRAMES** (more frames → more redundancy
averaged into the mean). It closes the **frame-COUNT** door only. **Spatial resolution and frame-selection
POLICY are explicitly NOT in scope** — resolution changes *what is inside each frame's tokens* (new detail),
not *how many* frames; F67's dilution argument does not apply.

**S2S scope (quoted, `S2S_PROBE_VERDICT_REVIEW.md:234-237`):**
> "the **retrieval-object ('don't-pool') family is closed** … set-to-set / late-interaction retrieval over
> frozen Qwen **frame-group tokens** does not beat pooled-cosine retrieval."

S2S killed an **operator over already-extracted features** (pool vs. late-interaction). It does **not** touch
the *sampling policy at extraction time* or the *pixel budget*. Crucially, `S2S_PREREG_REVIEW.md:81` records
that S2S **held resolution FIXED as a control** ("`max_pixels` varies `grid_h,grid_w` but never `grid_t`"),
and its sensitivity arms were {Chamfer, WITH-TEXT, 16-frame, near-dup-excluded} — **no resolution arm.**

**Conclusion: spatial resolution is a genuinely VIRGIN axis** — a repo-wide grep finds `max_pixels` only ever
as a fixed *parity constant* (S2S/W2A/B2/SAV code reviews), **never as a treatment**. No prior negative binds
it.

---

## 3. DIMENSION-BY-DIMENSION

### 3.1 SPATIAL RESOLUTION — the spatial analogue of F67, but NEW information per frame ★ lead
**Mechanism.** Higher `max_pixels` gives the ViT more patches per frame → the encoder can resolve on-screen
text (memes-in-video, captions, chyrons), small hate symbols/flags/gestures, and facial expression — the
exact fine-grained content this task turns on. Unlike F67, this adds information rather than diluting a pool.

**Literature (verified).**
- **S2 / "When Do We Not Need Larger Vision Models?"** (Shi et al., ECCV 2024, arXiv 2403.13043): multi-scale
  features from a **FROZEN** vision backbone (parameter-free `S2-Wrapper`) give **+2.8 on TextVQA** (58.2→61.0),
  **+8 on V*-attention** (43.5→51.3), **+5 on V*-spatial** (56.6→61.8). Headline finding: *"larger image
  scales consistently improve performance while bigger models sometimes fail to improve or even hurt."*
  → **image-scale is a DIFFERENT lever than the model-scale we already exhausted** (our F-line: CLIP<32B<7B,
  scale regresses). We killed *model* scale; we never touched *image* scale.
- **Qwen2.5-VL Technical Report** (arXiv 2502.13923): the ViT is native-dynamic-resolution *specifically to
  preserve fine detail for document/OCR/chart understanding*; resolution is the intended axis for text-in-image.
- **Hateful-meme evidence** (arXiv 2505.00150): visual signal is decisive (image-inclusive 69.5% vs text-only
  62.8%); reading in-image text lifts most VLMs +0.7–2.35 acc. **Higher resolution lets Qwen read on-screen
  text NATIVELY** — this is *not* the vetoed separate-OCR channel (§4), it is the encoder seeing its own pixels
  better.

**Expected effect size (honest, our regime).** Literature gains (+2.8 TextVQA, +5–8 V*) are on models that
feed **all** visual tokens to the LLM. Our readout **mean-pools ~720 tokens into one vector** (or takes a
single last token), which will **attenuate** fine-detail gains — the dominant risk. Net: a **real ≥+1 candidate
on the image-converting dataset (HateMM)**; **+3 across ≥2 datasets is unlikely** given EN's label cap and ZH's
marginality (§5). Best paired with a detail-preserving readout (→ readout/head-recipe agents).

**Cost.** Extraction-only; visual tokens scale ~linearly with pixels, so 2×–4× `max_pixels` ≈ 2×–4× extraction
time (still cheap, single frozen forward) + more VRAM. **Features-only → Modal-eligible as a triage probe.**
**Ban check: CLEAR** (virgin axis, §2). **Prior ≥+1: MODERATE (HateMM); ≥+3: LOW.**

### 3.2 FRAME-SELECTION POLICY — uniform vs content-aware, at fixed 8-frame budget
**Mechanism.** Replace `linspace` with scene-change / motion / CLIP-diversity keyframes at the same budget.
**Literature (verified).** AKS (arXiv 2502.21271): **+3.8–5.0 on LongVideoBench, +0.9–2.3 on VideoMME** over
uniform — **but** all runs ≥16 frames, on **long-form** video, and **query-aware** (frames scored vs the
question). AdaRD-Key (2510.02778) similarly targets long videos at 32/64-frame budgets.
**Why it transfers POORLY to us:** (i) our videos are **short** (MHC short-form/portrait; HateMM ~1 min; the
long-form regime these methods target is minutes-to-hour), (ii) we use **8 frames** (below every reported
budget — at 8f on short video uniform is near-optimal), (iii) our encoder is **query-agnostic** (fixed
instruction), so the query-aware relevance that drives most of AKS's gain is unavailable — only diversity/
scene-change selection applies, and our own data says the slack is tiny (§1C: only 7–22% near-dup videos;
S2S near-dup-excluded HateMM Δ=+0.0082 ≈ null). **Ban check: CLEAR** (F67 killed count, S2S killed operators
over features — neither touches the sampling policy). **Prior ≥+1: LOW; ≥+3: negligible.**

### 3.3 TRANSCRIPT QUALITY / COVERAGE / TITLE — audited CLOSED
Full coverage, no truncation, Whisper-large-v3 ceiling, no in-code token cap (§1B). **Not a gap.** Title is
absent from *source* (not truncated) — recovering it means re-scraping YouTube metadata (a data-collection
task, veto-adjacent, and hate-video titles are frequently on deleted/removed channels → low yield). ZH's
4-word text is data poverty (likely music/no-speech clips), **not** a fidelity defect ASR can fix.
**Prior ≥+1: LOW (title-scrape, uncertain); ≥+3: negligible.**

### 3.4 AUDIO — CLOSED, not reopened. F64 (Whisper-encoder) null, F41 (eGeMAPS) null. Noted only.

### 3.5 OCR — USER-VETOED. Not proposed. (Note: §3.1 native high-res text-reading is *distinct* — it adds no
channel; it is the encoder resolving pixels it already receives.)

### 3.6 PREPROCESSING PATHOLOGIES — no actionable gap (§1D: 1 undecodable HateMM video, clean grids).

---

## 4. TOP-3

| # | move | mechanism | verified lit / effect | our-regime expectation | cost | ban | ≥+1 / ≥+3 prior |
|---|---|---|---|---|---|---|---|
| **1** | **Raise `max_pixels` (e.g. 2–4×, to ~300–600 K px/frame)** | new per-frame detail: on-screen text, small symbols, faces — resolves what a 280×504 frame loses | S2 (ECCV'24, **frozen** backbone): +2.8 TextVQA, +5–8 V*; *image-scale > model-scale* | **≥+1 plausible on HateMM** (image-converting); attenuated by our mean-pool readout; **+3 unlikely** across EN(label-cap)/ZH(marginal) | extraction-only, ~2–4× time, features-only → **Modal-triageable** | **CLEAR — virgin axis** | MOD / LOW |
| **2** | Content-aware keyframe selection @8f (diversity/scene-change, query-agnostic) | dedupe the near-dup frames uniform can hit | AKS +3.8–5.0 (LongVideoBench) — **but ≥16f, long-form, query-aware** | LOW: short videos, 8f, query-agnostic; S2S near-dup arm ≈null; only 7–22% videos have slack | extraction-only, cheap | CLEAR | LOW / ~0 |
| **3** | Recover video TITLE (+ description) text | adds a new, often-decisive text field (RA-HMD meme heritage used titles) | title/OCR text lifts VLMs +0.7–2.35 (meme lit) | uncertain: absent from source, needs metadata re-scrape, hate-video titles often gone | data-collection (network); veto-adjacent | n/a | LOW / ~0 |

---

## 5. HONEST ASSESSMENT — can any input-fidelity move reach +3?

**Probably not on its own — but resolution is the one axis worth a cheap probe, and it is the only virgin one.**

The binding constraint is the campaign's **text-carried finding** (F45/F58): the text stream dominates all
three datasets, so resolution and frame-selection — which improve the **image** stream — start behind. Mapping
onto our own dataset roles: **EN is label-capped** (fidelity can't break a label ceiling), **HateMM is the one
dataset whose image stream converts** (it already carries the project's cleanest encoder-level PASS), and **ZH
is marginal** — ~~though the `$0` audit adds a real nuance: **ZH text is only ~4 words median, so ZH's image
stream carries far more relative weight than "text-carried" implies**, making ZH a *secondary* place resolution
could move a needle (from a low base)~~ **[WITHDRAWN 2026-07-28 — see the erratum at §(B): the 4-word figure is
a whitespace-split artefact; deployed ZH text is a median 106 Chinese characters of Bilibili description
metadata, so this "nuance" does not exist and ZH gets no relative-image-weight credit here]**. A +3 needs a
gain on ≥2 datasets; the only two image-responsive
candidates are HateMM (already passing — headroom to +3 more is the open question) and ZH (marginal, low base).
That conjunction is a **long shot**.

Two structural discounts cap the upside honestly: (1) **our readout mean-pools ~720 visual tokens into a single
vector** (or a single last token), and every literature resolution gain comes from architectures that keep all
visual tokens — so fine-detail information may be washed out before it reaches the head; resolution's payoff is
**conditional on a detail-preserving readout** (a natural conjunction with the readout/head-recipe agents, and
a reason not to probe resolution in complete isolation). (2) The team-lead's own hook — better image fidelity
could also sharpen the **fused retrieval keys** — is real but second-order (the keys are dominated by the same
text stream).

**Bottom line.** Frame-selection (#2) and title-scrape (#3) are **low-value** — decline both. **Resolution
(#1) is the single input-fidelity move that is (a) never tested, (b) cap-bound with the median video downscaled
~6.5× in area, (c) supported by frozen-backbone literature that specifically flips our exhausted model-scale
result (image-scale > model-scale), and (d) uncovered by any F67/S2S ban.** It is a legitimate **≥+1 triage
probe on HateMM** (cheap, features-only, Modal-eligible), *not* a credible standalone route to +3. If it is run,
run it **jointly with a detail-preserving readout** so the mean-pool does not silently absorb the added
fidelity — resolution alone, through the current mean-pool, is the most likely way to reproduce F67's
"probe-passes-training-goes-flat" pattern a spatial time.
