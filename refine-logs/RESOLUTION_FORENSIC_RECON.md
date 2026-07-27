# SPATIAL-RESOLUTION CELL — FORENSIC RECON (litsweep2 batch-3, last candidate)

**Author:** resolution forensic-recon subagent (zero-GPU). **Date:** 2026-07-25 NZST.
**Discipline:** CPU-only. ZERO SLURM / GPU / Modal / job-submission / prereg / test-touch.
No `autoresearch/goal_mllm_plus3/state/` mutation. Nothing pushed. All resolution numbers below are
`$0`-computed on-disk (ffprobe on raw videos + `grid_thw` in the banked framesets); all accuracy/timing
anchors copied verbatim from banked logs (provenance §6).

**VERDICT: PARK.** The cap is genuinely binding and the axis is genuinely virgin, but the litsweep2 "~6.5×"
premise is **wrong in a way that guts the prior**: the one dataset whose image stream converts (HateMM) has
only **2.71×** area headroom (its sources are 480p, not 720p), while the datasets with real headroom (EN 10.5×,
ZH 13.7× — 1080p sources) are exactly the ones whose image streams do **not** convert (EN collapsed F65, ZH
marginal). Headroom is **anti-correlated with conversion.** Combined with the mean-pool attenuation caveat, the
F65 law-I no-conversion precedent, and the fact that there is **no cheap cloud-triage path** (extraction needs
the raw videos, which never leave the box → Modal is blocked), this prices below the +3 bar. A fully-specified
~1 GPU-h door-closer is provided (§4) if the campaign wants the last input-fidelity axis *measured* rather than
prose-argued, but it is not a credible route to the goal.

---

## §0. CLAIM VERIFICATION — the "~6.5×" is WRONG (premise error, not arithmetic error)

litsweep2 (`LITSWEEP2_INPUT_FIDELITY.md:50-53`) claims: *"a native ≈720p frame (≈920 K px) is downscaled
**~6.5× in area (~2.5× per side)** to fit."* The **920 K-px / 720p "native" figure was assumed, never
measured.** I ffprobed the raw sources (`data/video/<ds>/All/*.mp4`, CPU, ~200-video samples per dataset):

| dataset | source median area (px) | typical source | source/cap area | per-side | % videos above cap |
|---|---|---|---|---|---|
| **HateMM** | **409 920** (854×480 class) | **480p** | **2.71×** | **1.65×** | 89.3% |
| MHC (EN) | 1 595 720 | 720p–1080p+ | 10.55× | 3.25× | 98.5% |
| **MHC_zh** | **2 073 600** (1920×1080) | **1080p** | **13.71×** | **3.70×** | 98.8% |
| *litsweep2 assumed* | *920 000* | *720p* | *6.08×* | *2.47×* | — |

**Findings.**
1. **The cap IS binding** (89–99% of videos in every dataset are downscaled) — that half of litsweep2 is TRUE.
2. **But no dataset sits at 6.5×.** HateMM — the dataset litsweep2 recommended and priced its "≥+1" on — is at
   **2.71×** (480p source), *less than half* the claimed multiplier. The 6.5× is a fictional average that
   describes neither the converting dataset (2.7×) nor the high-res ones (10–14×).
3. **The true multiplier is anti-correlated with conversion** (the load-bearing correction): the biggest
   real-pixel headroom is on EN (10.5×) and ZH (13.7×), the two datasets whose image streams the campaign has
   already shown do **not** convert to head signal (EN image stream collapses upstream of the LLM, `FRAME_BUDGET_FORENSIC_RECON.md:102-104`;
   ZH marginal, never a clean PASS). HateMM converts but is nearly at native already.

**Corrected statement:** the cap discards real source pixels (~63% of HateMM's, ~90% of ZH's), so it is a real
constraint — but on HateMM the recoverable detail is a modest **1.65×/side** bump toward its own 480p ceiling,
not the "2.5×/side toward 720p" litsweep2 implied.

Processed-area anchor confirmed independently from the banked `grid_thw` (`data/CLIP_Embedding/{HateMM,MHC}/frameset_qwen7b_8f/*_frameset.pt`):
median grid **20×36 patches → 280×504 px → 141 120 px processed (≈93% of the 151 200 cap), 720 visual tokens/video @8f** —
matches litsweep2 §1A exactly. The cap side is verified; only the "native" side was fabricated.

---

## §1. MECHANISM + EXACT CURRENT SETTINGS (file:line)

Deployed extractor = `src/utils/generate_VideoMLLM_embedding_HF.py` (git: last touched `ece6a3b`; the `_HF.pt`
caches every formal head-run consumes; `lora/readout/bidir` variants share sampler+prompt verbatim). Banked
16f-analogue run (job 13352, `mllm_embed_16f_13352.out`) confirms the arg wiring end-to-end.

| control | value | code |
|---|---|---|
| **`--max_pixels`** (the resolution knob) | `360*420 = 151200` px/frame | `:96-101` (argdef), `:404` (applied) |
| **How it is applied** | `AutoProcessor.from_pretrained(model, max_pixels=...)` at **construction**; the `processor(...)` **`__call__` ignores it** under transformers 4.49 | `:403-404`, comment `:268-269` |
| **`min_pixels` / `resized_h` / `resized_w`** | **NOT SET** — only `max_pixels` is passed; Qwen `smart_resize` uses its default `min_pixels` floor and factor-28 rounding | (absent) |
| **`--num_frames`** | `8`, uniform `np.linspace(0,N-1,8)` rounded (no RNG, no content signal) | `:90-95`, `:146-152` |
| **Frames→pixels interaction** | 8 frames enter as ONE `{"type":"video"}` turn, temporal-merged in pairs (`temporal_patch_size=2` → `grid_t=4`); `max_pixels` caps the **spatial** grid per frame-slot, independent of frame count | `:241-251`, `:270-275` |
| **Readout (the attenuator)** | img = **mean-pool** of last-layer hidden states over the whole visual+instruction span → L2; text = mean over trailing assistant-header tokens (≈last-token) → L2 | `:290-322` |
| **Truncation ceiling** | NONE — single frozen forward, no `cutoff_len`/`max_length`; only VRAM bounds it | `:270-278` |

**Native encoder ceiling** (on-disk config, per `FRAME_BUDGET_FORENSIC_RECON.md:56-58`): `patch_size=14`,
`spatial_merge_size=2` (1 visual token = a 28×28-px merged block), native-dynamic-resolution ViT — supports
millions of px/frame; our 151 200 cap is a pure **VRAM knob** (~1/6 of a 720p frame in the abstract, but as
§0 shows, only ~0.37× of a real HateMM 480p frame). Raising `max_pixels` puts more of each frame's **real**
source pixels in front of the ViT — but only up to that frame's native area (beyond native, `smart_resize`
upscales → interpolated, no new information). **This caps HateMM's usable treatment at its 2.71× native.**

---

## §2. FEASIBILITY — resolution ladder × tokens × GPU-h (A100-SXM4-80GB)

**Hardware anchor:** past extractions ran on **NVIDIA A100-SXM4-80GB** (`mllm_embed_16f_13352.out`: `NVIDIA
A100-SXM4-80GB, 81920 MiB`), 1 GPU / 8 CPU / 64 GB. **Timing anchors (banked, verbatim):** 8f/1× HateMM
3-split (1066 vids) via S2S extractor = **949.1 s** (`FRAME_BUDGET_FORENSIC_RECON.md:184`); the plain 16f
extraction (job 13352, HateMM 3-split, 2× visual tokens **and** 2× frame-decode) = **elapsed 00:59:57**
(`sacct` 13352). Extraction cost ≈ frame-decode (∝ frame count, FIXED at 8 here) + ViT+LLM forward (∝ visual
tokens ∝ `max_pixels`). Because **frame count stays 8**, a resolution bump adds forward cost but **not** decode
cost — so per-token it is *cheaper* than the 16f job (which doubled both).

| setting (8 frames) | max_pixels | ~visual tok/vid | source recovered (HateMM/ZH) | HateMM 3-split ext | MHC_zh 3-split ext |
|---|---|---|---|---|---|
| **current 1×** | 151 200 | 720 | 37% / 7% of source | ~16 min (banked) | ~25 min (banked 8f) |
| **2×** | 302 400 | ~1440 | 74% / 15% | **~0.4–0.6 GPU-h** | ~0.6–0.9 GPU-h |
| **native HateMM (2.71×)** | 409 920 | ~1950 | ~100% / 20% | **~0.6–0.9 GPU-h** | — |
| **4×** | 604 800 | ~2880 | HateMM UPSCALES / 29% | ~1.0–1.3 GPU-h | ~1.0–1.4 GPU-h |

Head: `enc3seed` head ≈ **29 s/run** (trainlog tqdm), 3 seeds × dual protocol ≈ ~3 min ≈ **~0.03 GPU-h**
(negligible, `$0`-class after extraction). All rows fit **1 GPU / ≤16 CPU / 64 GB** — trivially within the
2-GPU/16-CPU user cap (single-GPU job).

**Where extraction can run — LOCAL SLURM ONLY.** The expensive step is turning **raw videos → features** with
the local Qwen2.5-VL-7B weights. Raw videos are **hard-blocked from the cloud** (`modal_probe_runner.py`,
CLAUDE.md data boundary). Modal can only ever see derived `.pt` features — but those *are the output* we would
be producing. **⇒ There is NO Modal-triage path for this axis** (litsweep2's "features-only → Modal-eligible
triage probe", `:149,:185`, is FALSE — it conflates the cheap head with the expensive extraction). The minimal
spend is a real local GPU extraction, not a cheap cloud probe.

---

## §3. BANKED EVIDENCE — for / against

**Repo-wide `max_pixels` grep:** appears ONLY as a fixed parity constant in S2S/W2A/B2/SAV/readout code
reviews and prereg controls — **never once as a treatment.** S2S explicitly held it fixed as a control
(`S2S_PREREG_REVIEW.md`, quoted in `LITSWEEP2_INPUT_FIDELITY.md:111-113`). **Spatial resolution is a genuinely
VIRGIN axis; no prior negative binds it.** (Confirmed, litsweep2 §2.)

**Strongest argument FOR:**
- **Virgin, unbanned axis.** F67 killed frame **COUNT** (8→16, mean paired Δacc **−0.0077** val-sel /
  **+0.0031** final, both inside noise → FAIL both protocols, `FRAME16_VERDICT_REVIEW.md` §3); S2S killed
  **operators over fixed features** (pool vs late-interaction). Neither touches within-frame pixel budget.
- **S2 / "When Do We Not Need Larger Vision Models?"** (Shi et al., ECCV'24, arXiv 2403.13043): multi-scale
  features from a **FROZEN** backbone give +2.8 TextVQA, +5–8 V*, and — the pointed part — *image-scale keeps
  helping where model-scale plateaus/hurts.* Our F-line exhausted **model** scale (CLIP<32B<7B, scale
  regresses); we never touched **image** scale. Different lever.
- **ZH has headroom** ~~AND relative image weight~~**.** ZH sources are 1080p (13.7× headroom); ~~and ZH transcripts
  are ~4 words median (`LITSWEEP2_INPUT_FIDELITY.md:64`), so the ZH **image** stream carries more relative
  weight than the "text-carried" label implies — the one place added pixels feed a stream that isn't drowned
  by text.~~
  > **ERRATUM 2026-07-28 (propagated from `LITSWEEP3_ZH_SPECIFIC.md:18-26,29-34`, F77 / commit `d4af64b`).**
  > The struck half of this bullet is **withdrawn.** The "ZH transcripts ~4 words median" figure it rests on
  > (`LITSWEEP2_INPUT_FIDELITY.md:64`) is a **whitespace-split artefact** — Chinese has no inter-word spaces.
  > The deployed ZH text stream is a median **106 Chinese characters** (train 106 / val 108.5 / test 105) of
  > **Bilibili description metadata** — *not* Whisper ASR, and **42 %** of ZH train rows carry literal
  > `<em class="keyword">…</em>` markup inside the key. ZH's image stream therefore gets **no** extra relative
  > weight from a thin text stream, and the "one place added pixels feed a stream that isn't drowned by text"
  > argument does not hold. The 13.7× source-resolution headroom is measured separately (§ffprobe) and stands.
- **Not the OCR channel (boundary, flagged honestly).** The user veto forbids a *separate OCR channel*. Higher
  input resolution adds **no channel** — it lets the *existing* frozen ViT resolve pixels it already receives
  (on-screen text, small symbols, faces). It is mechanistically distinct from OCR. **But flag the adjacency:**
  the *purpose* being litsweep2-argued ("let Qwen read on-screen text natively") is text-recovery-shaped; if a
  reviewer reads "resolution to read in-image text" as an OCR-by-another-name move, it is user-ruling-adjacent.
  It does not cross the technical veto, but it is the nearest neighbor to it.

**Strongest argument AGAINST (this is where it prices out):**
- **Headroom ⟂ conversion (§0, the new load-bearing fact).** HateMM (the ONLY dataset with a clean
  encoder-level image PASS) is at 2.71× — nearly native; the recoverable detail is small. EN (10.5×) and ZH
  (13.7×) have the pixels, but EN's image stream is collapsed upstream (`FRAME_BUDGET_FORENSIC_RECON.md:102-104`,
  feeding a broken vision path more pixels heals nothing) and ZH is marginal. **The lever is strongest exactly
  where the head can't use it.**
- **Mean-pool attenuation (the campaign's recurring killer).** Every S2 gain comes from architectures that
  feed **all** visual tokens to the LLM. Our readout mean-pools ~720–1950 tokens into **one** vector
  (`:303`) — a single hateful sub-region's added detail is averaged toward zero before the head sees it. This
  is the exact "probe-passes / training-goes-flat" mechanism, now spatial.
- **F65 law-I precedent (direct).** Vision-LoRA already **moved** the image representation and produced
  **zero head conversion** (the 8th law-I; MEMORY: "image MOVED, zero conversion"). Higher resolution is
  another way to move the image representation; law-I predicts the same non-conversion.
- **F70 readout perm-null.** The whole readout grid (layer/prompt/pooling) sits inside the permutation-null —
  the head extracts no more from the frozen representation than it already does. Changing the representation's
  detail has no extraction path if the readout is the binding constraint.
- **F67 shared mechanism.** The direct temporal twin of this spatial move died by pooled-rep saturation; the
  saturation mechanism (mean-pool over near-redundant content) is shared.
- **Text-carried finding (F45/F58).** Text dominates all three datasets; an image-stream lever starts behind.

**Net:** the FOR case is real but every arm of it lands on a stream the head can't convert (EN/ZH) or a
dataset with little headroom (HateMM), and passes through the same mean-pool that has flattened every prior
image-side move. This is not MCR-F71 "arithmetic proves impossible" — real headroom exists — but it is
"banked evidence + mechanism price the prior LOW and off the +3 path."

---

## §4. MINIMAL DECISIVE CELL (specified, ready to fire on user ruling) + KILL-SWITCHES

Mirror frame16's two-stage logic exactly (frozen gate first; expensive LoRA composition only if it moves).

**Stage-1 (the only cell worth defining) — frozen-Qwen-8f @ HateMM native resolution, extraction + head.**
- **One changed variable:** `--max_pixels 151200 → 409920` (HateMM native; 2× is the alternative but 2.71×
  gives the axis its *best shot* on the converting dataset without upscaling). No code edit — arg exists (`:96-101`).
- **Why HateMM despite low headroom:** it is the ONLY dataset whose frozen image stream converts to head
  signal. If near-native resolution on the converting dataset does not move the frozen head, the axis is dead
  (the high-headroom datasets EN/ZH don't convert, so they can't rescue it). If it *does* move, ZH (headroom +
  relative image weight) becomes the stage-1.5 follow-up.
- **Collision-safe naming:** `--out_model_tag Qwen2.5-VL-7B-Instruct_HF-hires410k`
  → `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF-hires410k.pt` (banked
  1× `..._HF.pt` untouched). Head reads `--model Qwen2.5-VL-7B-Instruct_HF-hires410k`, group `RAC_video_hires`.
- **Comparison = paired within head-seed vs the FROZEN-8f-1× floor (job 12850 / F53):** final-ep
  **0.8682/0.8591**, val-sel **0.8729/0.8648** (`FRAME_BUDGET_FORENSIC_RECON.md:145-148`; re-read verbatim at
  prereg). This is the clean single-variable floor (same encoder family, resolution the only change) — NOT the
  LoRA-curric floors, which use a different (adapted) encoder.
- **Cost:** ~0.6–0.9 GPU-h extraction + ~3 min head ≈ **~1 GPU-h total**, 1 GPU. Local SLURM (no Modal path).

**Draft kill-switches (pre-declarable):**
- **KS (auto-kill, kills ZH stage-1.5 too):** frozen-hires **ties or regresses** the 1× floor — mean paired
  Δacc ≤ 0 OR bootstrap CI straddles 0 — on **both** protocols ⇒ axis KILLED; ZH follow-up auto-dead (added
  pixels on the converting dataset carry no head signal ⇒ they carry none on the non-converting ones either).
- **CONTINUE-to-ZH gate (spend gate, not a claim):** mean paired Δacc ≥ **+0.010**, 3/3 sign-consistent, on
  ≥1 protocol ⇒ justifies the ~2 GPU-h MHC_zh @4× stage-1.5.
- **FORMAL (house, paper-worthy):** mean **Δacc ≥ +0.030 AND ΔmF1 ≥ +0.030, 3/3 seeds, BOTH protocols**, vs
  the frozen-8f floor. Single **PRIMARY arm = HateMM @410k**; ZH @4× is a pre-declared **secondary** arm gated
  on the CONTINUE gate (multiplicity note; never a free {2×,4×} sweep). To count as a *deployed-system* gain it
  must additionally survive composition with LoRA (hi-res LoRA re-extract vs LoRA-curric floors **13241**
  HateMM / **13150** ZH) — an expensive stage-2, NO-GO unless stage-1 clears the CONTINUE gate.

---

## §5. RECOMMENDATION — **PARK** (prior below bar)

**PARK for the +3 goal.** After the §0 correction the litsweep2 "≥+1 on HateMM MODERATE" prior does not
survive: it rested on a fictional 6.5× (720p) headroom; HateMM is actually 2.71× (480p), so the recoverable
detail on the converting dataset is a modest 1.65×/side bump, and the datasets that *do* have headroom (EN/ZH)
are the ones whose image streams don't convert. Layer on mean-pool attenuation, the F65 law-I no-conversion
precedent, the F70 readout perm-null, and F67's already-realized shared-mechanism death, and this is the same
LOW-MODEST door-closer frame16 was — but with a *worse* prior (frame16 at least targeted its full headroom;
resolution's headroom is off-target) and **no cheap cloud-triage escape** (extraction is local-GPU-bound). It
does not manufacture a launch.

**Prior estimate:** ≥+1 acc on HateMM (frozen probe): **LOW, ~5–10%** (below frame16's already-failed ~8–12%,
because HateMM's headroom is smaller than frame16's frame-doubling and passes the same mean-pool). ≥+3 on ≥2
datasets: **NEGLIGIBLE, <3%** (requires HateMM + one of EN/ZH; EN collapsed, ZH marginal, both attenuated).

**Not a hard kill — a park.** Unlike MCR-F71 the arithmetic does not *prove* impossibility (real source
headroom exists, ZH genuinely so). So the honest disposition is PARK with a **ready ~1 GPU-h minimal cell
(§4)**: if the campaign later wants the *last* virgin input-fidelity axis converted from "prose-argued" to
"measured-and-closed" (the value frame16 was run for), fire Stage-1 HateMM @410k as specified — single
variable, clean floor, ~1 GPU-h, KS pre-declared. Absent that closure goal, it stays parked below C-line
candidates; it is not a substantial-gain route.

---

## §6. PROVENANCE (all `$0` / banked; zero GPU/Modal/SLURM/test-touch this session)

- **Extractor / resolution knob:** `src/utils/generate_VideoMLLM_embedding_HF.py:96-101` (`max_pixels=360*420`
  argdef), `:403-404` (applied at processor construction), `:268-269` (`__call__` ignores it), `:146-152`
  (uniform sampler), `:290-322` (mean-pool readout). Arg wiring confirmed live in `mllm_embed_16f_13352.out`
  (`max_pixels=151200, num_frames=16`).
- **Source resolutions (`$0` ffprobe, ~200-vid samples):** HateMM median 409 920 px (89.3% > cap); MHC(EN)
  median 1 595 720 (98.5%); MHC_zh median 2 073 600 = 1080p (98.8%). Cap = 151 200.
- **Processed area / tokens (`$0` from `grid_thw`):** `data/CLIP_Embedding/{HateMM,MHC}/frameset_qwen7b_8f/*_frameset.pt`
  — median grid 20×36 patches → 141 120 px processed → 720 visual tokens/video @8f (matches
  `LITSWEEP2_INPUT_FIDELITY.md:43-48`).
- **GPU + timing:** `NVIDIA A100-SXM4-80GB` (`mllm_embed_16f_13352.out`); 16f HateMM 3-split elapsed 00:59:57
  (`sacct -j 13352`); 8f HateMM 949.1 s + head ~29 s/run (`FRAME_BUDGET_FORENSIC_RECON.md:184`, trainlog tqdm).
- **Frozen-8f floor:** job 12850 / F53 — final 0.8682/0.8591, val-sel 0.8729/0.8648
  (`FRAME_BUDGET_FORENSIC_RECON.md:145-148`; re-read verbatim at prereg).
- **F67 verdict:** `FRAME16_VERDICT_REVIEW.md` §3 (Δ −0.0077 val-sel / +0.0031 final, FAIL both protocols).
- **F65 / EN-collapse:** `VISION_UNFREEZE_FORENSIC_RECON.md` (image-MOVED / zero-conversion);
  `FRAME_BUDGET_FORENSIC_RECON.md:102-104` (EN image stream collapsed upstream).
- **LoRA-curric floors:** HateMM job 13241, ZH job 13150 (`slurm/logs/enc3s_*_13241/13150.trainlog`).
- **Modal data boundary:** CLAUDE.md + `scripts/cloud/modal_probe_runner.py` (raw videos hard-blocked from cloud).
- **Litsweep2 input:** `refine-logs/LITSWEEP2_INPUT_FIDELITY.md` (commit 7dc3b4c).
- Split sizes: HateMM 744/107/215, MHC 549/80/161, MHC_zh 579/78/149.
