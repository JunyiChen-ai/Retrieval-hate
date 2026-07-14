# W2-B Forensic Recon — Banked multimodal sub-clip caches (retrieval-object axis)

**Owner:** W2-B probe chain. **Mode:** ZERO local GPU; CPU metadata inspection only (`torch.load` +
shape/norm/id reads — no matrix compute, no training). **Companion:** `refine-logs/W2B_PROBE_DESIGN.md`
(the executable design), `research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md §W2-B` (the candidate),
`refine-logs/S2S_PROBE_DESIGN.md` (the machinery adapted). **Status:** RECON COMPLETE — awaiting fresh
review of the design; NOTHING is authorized to run.

**Bottom line (read this first).** The banked sub-clip caches are real, exactly K-per-video,
parent-indexed, id-aligned to the pooled caches, and **cloud-eligible** (derived CLIP floats). BUT the
scout's W2-B writeup under-cited the closest prior: these exact caches were **already consumed by an
in-project route — "Delta 1 / segment-mode multi-granularity temporal retrieval"** — which was run as a
**trained** retrieval-contrastive head and **killed with high confidence** (`exp-seg-mode-ablation.md`:
"sign-flips by language, no consistent gain … do NOT re-attempt segment-level temporal retrieval on these
datasets without gold spans"). W2-B is **non-isomorphic** to that route (it is a *zero-training frozen
paired kNN* measurement, not a trained head, and it removes the exact confound Delta-1's failure was
attributed to), but the in-project negative is a strong **prior-lowering / anti-repeat** flag that must be
foregrounded. W2-B's honest value is unchanged from the scout's read — **a ~$0 family de-risker for S2S,
not a novelty bet** — but the prior on a *positive* is now weaker than the scout's "MODEST," because the
trained version of essentially this idea already sign-flipped on the same data.

---

## 1. Cache reality — exact paths, shapes, indexing (verified 2026-07-15)

All caches: `openai/clip-vit-large-patch14-336` (**CLIP, not Qwen** — the pre-declared encoder asymmetry
below). Files under `data/CLIP_Embedding/<DS>/`.

### 1.1 Contract (per file)
Distinct from the pooled / S2S frame-set contract. Keys:
```
video_ids          : list[str]  length V            # parent video ids, IN THE POOLED-CACHE ORDER
subclip_img_feats  : float32 [V*K, 1024]            # CLIP vision pooler_output, MEAN-POOLED per window
subclip_parent     : int64   [V*K]                  # contiguous K-blocks: [0,0,0,0,1,1,1,1,...]
labels             : int64   [V*K]                  # per-subclip, INHERITED from parent (MIL)
num_subclips       : int  = K   (4 or 30)
num_frames         : int  = M   (16 for K4, 120 for K30 → 4 frames/sub-clip in both)
# _mm variant additionally:
subclip_txt_feats  : float32 [V*K, 768]             # per-segment CLIP-text of that sub-clip's ASR
subclip_txt_has_text : bool  [V*K]                  # whether that segment had ASR text
asr_source         : str  = ./data/ASR/<DS>/<split>_asrK4_whisper-large-v3.jsonl
```
**Key differences from `s2s_probe.py`'s loader** (which the design must adapt): (a) ids under `video_ids`
(flat), **not** `ids[0]`; (b) features are a **flat `[V*K, D]`** block, not `[N,T,D]` — reshape to
`[V,K,D]` via the verified contiguous parent blocks; (c) `labels` are **per-subclip** (dedup to per-video
via the first subclip of each parent); (d) features are **NOT L2-normalized at storage** (norms ≈ 21–35,
mean ≈ 30) → the probe must normalize at score time (S2S does the same).

### 1.2 Per-dataset / per-split availability and shapes

| dataset | split | K4 | rows (=V·4) | K30 | K4_mm | V |
|---|---|---|---|---|---|---|
| HateMM | train | ✓ | 2976 | ✓ (22320) | — | 744 |
| HateMM | dev_seen | ✓ | 428 | **✗** | — | 107 |
| HateMM | test_seen | ✓ | 860 | **✗** | — | 215 |
| MHC (EN) | train | ✓ | 2196 | — | ✓ | 549 |
| MHC (EN) | dev_seen | ✓ | 320 | — | **✗** | 80 |
| MHC (EN) | test_seen | ✓ | 644 | — | **✗** | 161 |
| MHC_zh | train | ✓ | 2316 | — | ✓ | 579 |
| MHC_zh | dev_seen | ✓ | 312 | — | ✗ | 78 |
| MHC_zh | test_seen | ✓ | 596 | — | ✗ | 149 |
| HateClipSeg | test_seen only | ✓ (6.3M) | — | ✓ (47M) | — | (loc set, no train) |

**Load-bearing availability facts:**
- **K4 is the only granularity with all three splits on both anchor datasets.** So the **primary** probe
  (memory = train ∪ dev_seen, LOO) is K4-only. Its memory sizes are **HateMM 744+107 = 851** and
  **MHC-EN 549+80 = 629 — identical to S2S's `EXPECTED_MEM` (851 / 629)**, because they are the same
  videos. The S2S N4 size-guard transfers verbatim.
- **K30 exists for HateMM TRAIN ONLY** (no dev/test K30). So the K30 granularity arm cannot use the
  train∪val memory; it is a **within-train LOO sensitivity** (744 videos) and must be paired against a
  **K4-train-only** arm on the identical 744 videos, never mixed with the 851 primary. Pre-declared.
- **`_mm` exists for MHC/MHC_zh TRAIN ONLY** (no dev/test _mm). So the multimodal arm is likewise a
  **within-train LOO sensitivity** (MHC-EN 549), paired against K4-visual-train-only on the same 549.
- **HateClipSeg has no train split** → unusable for the W2-B kNN probe (needs a train∪val memory). It is
  the P6 localization set; noted here only to explain why its subclips exist.
- **MHC_zh** has full K4 + train `_mm`, but ZH is out of the goal's binding gap (MHC-EN) and is a
  LoRA-lever dataset per the graveyard; carried as an OPTIONAL third-dataset arm, not primary.

### 1.3 Representation — exactly what a sub-clip vector IS
From `src/utils/generate_subclip_embedding_HF.py` (header + code, read 2026-07-15) and
`scripts/slurm/gen_subclip.sbatch`:
1. Uniformly sample **M frames** (16 for K4, 120 for K30) with the **same decord/PyAV sampler** as the
   whole-video CLIP cache.
2. Split the M frames into **K contiguous temporal windows** (4 frames each).
3. Frozen `CLIPVisionModel` per frame → `pooler_output` (1024-d) → **mean-pool within each window** → K
   sub-clip visual vectors `[K, 1024]`. Unnormalized at storage.
4. Sub-clips **inherit the parent video label** (MIL); **NO gold spans**.

So a sub-clip vector is a **CLIP vision-pooler mean over 4 contiguous frames** — a coarse temporal
segment. This is the **same 1024-d vision-pooler space** as the pooled parent `img_feats` (which is why
their dims match); the pooled parent is a mean over the whole video, a sub-clip is a mean over one window.

The **`_mm` text stream** (`gen_segment_asr.sbatch` → `generate_segment_asr_HF.py`, then the mm build)
re-extracts **segment-aligned Whisper-large-v3 ASR** per window and encodes **CLIP-text** per segment →
`subclip_txt_feats [V*K, 768]`. This is genuinely per-segment multimodal (the plain K4 cache instead
shares the single video-level text). **ASR = native transcript, NOT OCR** → veto-compliant.

### 1.4 Integrity checks (all PASS)
- **Parent indexing:** `subclip_parent` is exactly the contiguous K-block pattern `[0,0,0,0,1,1,…]`;
  every parent has exactly K subclips (min=max=K) on **all** splits/datasets. Flat→`[V,K,D]` reshape is
  therefore a trivial `.view(V,K,D)`.
- **Id-alignment:** `video_ids == pooled_cache ids[0]` in the **same order, all V**, on every
  split/dataset; and **per-video labels == pooled labels** on every split/dataset. Alignment to labels
  and to the pooled caches is trivial (index-identity).
- **Label consistency:** per-subclip labels are constant within a video (MIL inheritance verified).
- **Guard rows (zero vectors):** HateMM **train K4 = 4 zero rows** and **train K30 = 30 zero rows** — the
  same **1 undecodable video** × K (matches the pooled cache's known single HateMM-train zero-guard row).
  All other splits: **0** zero rows. The probe must (i) L2-normalize with an eps floor, (ii) derive a
  per-video `zero_guard = all-K-subclips-zero`, (iii) exclude guard rows from the near-dup audit — exactly
  as S2S handles its 1 guard video.
- **`_mm` text coverage (MHC-EN train):** 1562/2196 subclips (71.1%) have ASR text; 634 are zero-text
  vectors (no ASR in that window); **all 549 videos have ≥1 ASR subclip** (no fully-text-empty video). The
  probe's `_mm` arm must zero-mask empty-text subclips (a `has_text` gate is banked).

---

## 2. Prior consumption of these caches — what was measured, and non-isomorphism

The task asks precisely what prior route consumed these caches and why W2-B's set-**retrieval** question is
non-isomorphic. Three distinct prior uses exist; W2-B is non-isomorphic to all three, but the FIRST is the
close precedent the scout omitted.

### 2.1 Delta-1 / segment-mode multi-granularity temporal retrieval — the CLOSE PRECEDENT (KILLED)
`research-wiki/experiments/exp-seg-mode-ablation.md` (verdict **NO**, confidence **high**, 2026-07-01/02);
`research-wiki/DESIGN_iter1.md` §Delta-1; `ITERATION_LOG.md:528-570`.
- **What it did:** built a **second FAISS index over the auto sub-clips** and ran **retrieval-guided
  contrastive TRAINING** (InfoNCE/triplet, `LAMBDA_SEG=0.5`) at both whole-video AND sub-clip granularity,
  with **MIL pseudo-positives** and a **within-video drifting-benign-subclip hard negative**. It then
  measured the **trained head's** downstream acc / macro-F1 vs the whole-video baseline.
- **What it found:** MHC-EN **+0.0149 F1 / +0.0062 acc** (marginal); MHC_zh **−0.0656 F1 / −0.0671 acc**
  (hurts). **Sign-flips by language; no config beats whole-video on BOTH languages; none crosses acc
  0.85.** Diagnosed as **noisy MIL pseudo-positives without gold segment labels.** Explicit anti-repeat
  note: *"do not re-attempt segment-level temporal retrieval on these datasets without gold spans."*
  Note: this ablation was run on the **MHClip** datasets (the "owed" gap), **not on HateMM**.
- **Why W2-B is non-isomorphic (and why it is still worth a probe despite the anti-repeat flag):**
  Delta-1's mechanism was a **trained multi-granularity contrastive head** whose failure was attributed to
  the **training signal** (noisy MIL sub-clip pseudo-positives / drifting negatives). W2-B trains
  **nothing**: it is a **zero-training paired LOO kNN** on the **frozen** sub-clip vectors, asking only
  whether **MeanMaxSim set-matching** over the K frozen sub-clips retrieves better label-neighbours than
  the **pooled mean** of the same K vectors. It has **no second FAISS training index, no pseudo-positive
  mining, no drifting-negative, no InfoNCE.** It therefore *isolates the representation-geometry question
  from the training-recipe confound that Delta-1's negative was blamed on* — the same isolation logic by
  which S2S is non-isomorphic to pooled retrieval. AND the K4 set-vs-pool question was **never measured on
  HateMM at all** (Delta-1 only touched MHClip). **Honest consequence:** the anti-repeat flag lowers the
  prior on a *positive* below the scout's "MODEST"; the most likely and most useful outcome is a **clean
  kill that corroborates Delta-1 at zero cost**.

### 2.2 P11 — weak-supervision localization labels (HateMM train subclipK30) — KILLED, different question
`scripts/slurm/p11_hatemm_subclipK30.sbatch`, `scripts/analysis/p11_probe_hatemm.py`; graveyard id **P11**
("probe fail; MIL already carries it"). P11 built the **K30** HateMM-train cache to test **MLLM segment
scores as weak-sup TRAINING labels** — a *training-signal* question, not a retrieval-metric question. W2-B
uses the same cache for **frozen set-vs-pool kNN**, no MLLM scores, no training. Non-isomorphic.

### 2.3 P6 — MLLM localization scorer (HateClipSeg subclips) — POSITIVE, different question
Graveyard `P6+P10b` ("MLLM localization scorer wv-AUC modest"); `ITERATION_LOG.md:1222` ("within-video
signal significant but small; only K=4+subclip cell passed Bonferroni"). P6 **scored** sub-clips for a
**localization read-out** (wv-AUC on HateClipSeg). W2-B **matches** sub-clip **sets** for a
**classification retrieval vote**. Different object (scoring vs set-matching), different task
(localization vs classification), different dataset (HateClipSeg has no train). Non-isomorphic — this is
the P6 distinction the scout correctly drew.

### 2.4 Verdict on non-isomorphism
W2-B's *specific measurement* — a **zero-training paired LOO kNN vote comparing MeanMaxSim(SET) vs
pooled-mean(POOLED) on frozen sub-clip CLIP features** — has **NOT been run before** on any of these
caches. It is non-isomorphic to Delta-1 (trained head + mined pseudo-labels), P11 (weak-sup training
labels), and P6 (localization scoring). The novelty of the *mechanism*, however, is **near-isomorphic to
S2S** (set-matching over temporal units), so — as the scout said — W2-B is **not a standalone D7-novel
contribution**; its only novel sliver is the `_mm` multimodal sub-clip unit, and its real job is
de-risking the don't-pool family cheaply.

---

## 3. Cloud-readiness (Modal, features-only)

- **Volume:** live name is **`rgcl-features`** (`scripts/cloud/modal_probe_runner.py:VOLUME_NAME`).
  (The earlier `CLOUD_GPU_FEASIBILITY` doc's tentative `rgcl-feats` is superseded — the task's
  `rgcl-features` is correct.)
- **Are the subclip caches on the volume?** The `sync` entrypoint uploads the **entire**
  `data/CLIP_Embedding/<dataset>/` dir via `rglob("*")`, gated by `assert_uploadable` (allowlist
  `.pt/.jsonl/.json/.csv/.npy/.txt`; refuses media extensions and any `video/` dir). The subclip caches
  are `.pt` files not under a `video/` dir → **they pass the guard and are included automatically**. So a
  single `modal run scripts/cloud/modal_probe_runner.py::sync --dataset HateMM` (and `--dataset MHC`)
  uploads pooled + **subclip** + gt in one shot (`batch_upload(force=True)`). **They are cloud-eligible
  derived floats; whether a prior sync already pushed them is immaterial — the design's execution step-0
  re-runs sync, ~$0, a few MB.**
- **Mount + run:** volume mounts at `/root/data`; probe runs via
  `modal run scripts/cloud/modal_probe_runner.py::run --script scripts/analysis/<probe>.py --args "..."`
  on **CPU by default** (the image auto-mounts `scripts/analysis/` and `src/`). The probe must therefore
  (a) live in `scripts/analysis/`, and (b) take a `--data_root` arg (default local repo path; `/root/data`
  on Modal) so the same file runs locally and on cloud.
- **Cost:** CPU-only, minutes → **~$0** within Modal's free credits. No GPU. (W2-B is lighter than S2S:
  CLIP 1024-d, K=4 → the all-pairs frame-frame tensor is tiny.)

---

## 4. Constraints / vetoes (all satisfied)
Banked own-train sub-clips ✓ (single-dataset own-train memory); `_mm` uses **native Whisper ASR, not OCR**
✓; **no gold** (labels are video-level, used only for Fano + oracle ceiling) ✓; single-dataset ✓; no
cross-seed ensemble / no external API ✓; not a P1–P5 re-proposal ✓; **CLIP encoder only** — the
pre-declared **encoder asymmetry** (B-line: CLIP < Qwen on HateMM) means a CLIP-null does **not** fully
close the Qwen-token S2S version; this asymmetry is stated in the design and bounds what a W2-B negative
licenses.
