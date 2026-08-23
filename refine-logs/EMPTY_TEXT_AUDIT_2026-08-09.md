# Empty-transcript audit — HateClipSeg test (395/395) and HateMM (74/1066)

Date 2026-08-09. **Pure audit: zero GPU, zero code change, zero test-set metric computed.**
Caches and gt files were opened read-only (shapes, row hashes, label counts). No model was
trained, no candidate metric was produced.

Trigger: `refine-logs/DEGEN_FEATURE_FIX_2026-08-09.md` §2d + §5 item 4 — an empty transcript
makes `generate_VideoCLIP_embedding_HF.py:244-257` build `windows = [[]]`, run the CLIP text
tower on `[BOS, EOS]`, and emit one identical 768-d vector for every such video.

Verified row-hash census (`torch.load`, CPU, read-only):

| cache | rows | unique `text_feats` rows |
|---|---|---|
| `data/CLIP_Embedding/HateClipSeg/test_seen_openai_clip-vit-large-patch14-336_HF.pt` | 395 | **1** |
| `data/CLIP_Embedding/HateMM/train_..._HF.pt` | 744 | 695 (39 empty collapse to 1) |
| `data/CLIP_Embedding/HateMM/dev_seen_..._HF.pt` | 107 | 98 (9 empty collapse to 1) |
| `data/CLIP_Embedding/HateMM/test_seen_..._HF.pt` | 215 | 190 (26 empty collapse to 1) |
| `data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_..._HF.pt` | 11850 | **no text tensor at all** (`subclip_img_feats` only) |

---

## 1. HateClipSeg — upstream never shipped transcripts; our prep never filled them in

### 1a. The official release has no transcript field. Not a loss on our side.

Upstream checkout: `~/data/HateClipSeg`, git remote
`https://github.com/Social-AI-Studio/HateClipSeg.git` @ `0095f18`. Complete file inventory:

```
Dataset/segment_level_annotation.csv   Video Id, Segment-Level Label, Segment Timestamp
Dataset/video_level_annotation.csv     Video Id, Video-Level Label, Target Victim
lexicons.json                          hate lexicons used for video search
Images/, pilot/, README.md, videos/    figures, our download scripts, README, media
```

Three columns in one CSV, three in the other. **There is no transcript, subtitle, caption or ASR
column anywhere in the release**, and the README's "Dataset File Structure" section documents
exactly these six columns and nothing else. A `grep -ril "transcript|subtitle|caption|asr|whisper"`
over the whole repo (excluding `.git` and `videos/`) returns exactly one hit — a *video title*
string inside our own `pilot/probe_results.tsv`.

The release also ships **annotations only, no videos** (we re-fetched 395/435 from the platforms;
`research-wiki/DATASET_hateclipseg.md` §3). So there is no upstream media-plus-text bundle we
could have mis-parsed.

**Verdict: upstream. HateClipSeg has no transcript channel to lose.** Not a DUA restriction and
not an intentional blank — the dataset simply is a label-only release.

### 1b. There is no "train/val side" for HateClipSeg either

`research-wiki/DATASET_hateclipseg.md:17` — "**No official train/val/test split is released**".
`data/gt/HateClipSeg/p11_split.json` is *our* stratified 60/10/30 split (seed 0, n=395,
`stratify=has_toxic_second`) over the same 395 videos, made for P11. All 395 rows come from the
same `test.jsonl`, so **all three of our HateClipSeg partitions have empty text**, not just "test".

### 1c. Where our pipeline stops short — and it is a non-wiring, not a drop

`scripts/analysis/hateclipseg_prep.py:104-107`:

```python
with open(JSONL_OUT, "w") as f:
    for vid in sorted(gold):
        f.write(json.dumps({"id": vid, "text": "", "label": 0}) + "\n")
```

`text: ""` is **hardcoded and deliberate**, consistent with the file's docstring
(`hateclipseg_prep.py:11-14`): `test.jsonl` was written purely as an *id list* to drive the
subclip extractor, with a dummy `label: 0` so that "no HateClipSeg label enters the pipeline"
stays auditable. At the time it was written (2026-07-04) there was no transcript to put there.

**But a transcript source has existed locally since the segment-ASR work and was never wired
back.** `data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl`, produced by
`src/utils/generate_segment_asr_HF.py` (Whisper large-v3, PyAV audio decode):

- 395/395 rows present, `audio_ok` true for **394**, non-empty `chunks` for **394**
- mean **2507** characters of transcript per video, `language: en` for all 395
- the single failure is `yt_NzvfkIYS5Yg` — the same truncated 138 KB file already flagged
  `DECODE_FAIL_UNREPAIRABLE` in `DEGEN_FEATURE_FIX_2026-08-09.md` §2c

So the honest characterisation is: **not "official had it and we dropped it"**, but **"we
generated it ourselves and never connected it"**. Two specific non-wirings:

| # | location | what it does | what it could do |
|---|---|---|---|
| L1 | `scripts/analysis/hateclipseg_prep.py:106` | writes `"text": ""` for all 395 rows | populate from `data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl` (join key `id`, concatenate `chunks[i][2]`) |
| L2 | `src/utils/generate_segment_asr_HF.py` (output side) | writes per-window `window_text` to `data/ASR/`, never to any gt jsonl or CLIP text cache | a video-level text cache, and/or per-window `subclip_txt_feats` (which the HateClipSeg subclip cache also lacks) |

Cost of remediation is CPU-cheap for L1 and one CLIP-text-tower pass for the re-extraction. **Not
done here — audit only.**

---

## 2. HateMM's 74 empty texts — upstream annotation blanks, and the sources really are speechless

### 2a. Provenance: upstream, and it is `Title` + `Transcript` both empty

`data/gt/HateMM/*.jsonl` is written by `scripts/prep_video_dataset.py`, whose `build_text()`
(lines 126-139) takes `Title` and `Transcript` straight from
`~/data/HateMM/annotation(new).json` and falls back to a **single space** `" "` when both are
empty. All 74 affected rows contain exactly `" "` (verified: `{"' '": 39}` / `{"' '": 9}` /
`{"' '": 26}`), which the tokenizer reduces to zero content ids → the constant vector.

Checking the 1066 upstream entries directly: **all 74 have `Title == "" and Transcript == ""` in
`annotation(new).json`**. Not one is a parse loss on our side, and HateMM's own
`HateMM_annotation.csv` (`video_file_name,label,hate_snippet,target`) has no text column at all.
**Verdict: upstream blanks, faithfully copied.**

### 2b. Are the sources silent, or did upstream ASR fail? — Mostly silent.

Independent evidence: our own Whisper large-v3 cache `data/ASR/HateMM/*_asrK30_*.jsonl` covers
all 1066 videos, including these 74. Result: **`audio_ok` is true for all 74** — every one has a
decodable audio track, so this is not a decode failure. What Whisper actually hears:

| bucket (unique lowercase word types in the full Whisper transcript) | n | reading |
|---|---|---|
| ≤ 6 types — silence-hallucination filler only (`"Thank you."` ×N, `"you you you"`, `"¶¶"`, `"."`, `"Outro Music"`) | **60** | music / ambient / no intelligible speech |
| 7-20 types — one or two short real utterances | **12** | sparse speech |
| > 20 types — real connected speech upstream failed to transcribe | **2** | `hate_video_157`, `non_hate_video_320` |

So ≈81 % (60/74) carry **no speech at all** and the flat filler is Whisper's well-known
silence artefact; ≈16 % carry a single line; and exactly **2 of 74** are genuine upstream ASR
misses. This is a *source property*, not an ASR bug — ours or theirs.

### 2c. Disjoint from the 16 black videos

`black_video_ids ∩ empty_text_ids = ∅` (n = 0). The picture-side and text-side degeneracies hit
**different** videos, so 16 + 74 = 90 distinct HateMM rows have one dead modality each. No video
in HateMM is dead on both channels.

### 2d. The part that matters for conclusions: the constant cluster is label-skewed

| split | empty-text rows | of which hate | P(hate \| empty text) | split base rate |
|---|---|---|---|---|
| train | 39 | 3 | **0.077** | 0.401 |
| val | 9 | 0 | **0.000** | 0.402 |
| test | 26 | 2 | **0.077** | 0.400 |
| **all** | **74** | **5** | **0.068** | 0.400 |

This is the load-bearing fact of the whole audit. The 74 rows are not merely uninformative — on
any CLIP-text-containing key they are **one single point at cosine 1.0 from each other**, and
that point is **93 % non-hate against a 40 % base rate**. A kNN / retrieval read-out over such a
key gets a near-free correct answer on ~7 % of the corpus from a constant. Independently
corroborated by `refine-logs/CLAP_GATE_SPEC_2026-07-27.md` (≤1-word stratum, P(hate)=0.0920) and
`refine-logs/ERRPAT_HateMM_2026-07-26.md` (0-for-30 empty-transcript test predictions, all
non-hate, all 3 seeds).

Note this points the **opposite way** from the image-side conclusion in
`DEGEN_FEATURE_FIX_2026-08-09.md` §4: the 16 black-video rows score at chance and *depress* the
metric, whereas the 74 empty-text rows sit in a high-purity cluster and can *inflate* a
text-inclusive read-out. The §4 "historical numbers are pessimistic" verdict was scoped to the
image side and **does not transfer** to the text side.

---

## 3. Blast radius — which historical conclusions need a footnote

Triage rule used below (judgement only; nothing was modified):

- **CLEAR** — the result never touched a CLIP `text_feats` block.
- **FOOTNOTE** — text block present, but the degeneracy cannot change the sign or the decision;
  a caveat line is enough.
- **RECHECK** — the degeneracy is plausibly load-bearing for the stated conclusion; the number or
  the mechanism story should be re-derived before it goes in a paper.

### 3a. HateClipSeg-side — no reported number is affected

Every HateClipSeg *result* in the corpus traces to one of two scoring paths, and **neither reads
the degenerate video-level `text_feats`**:

1. `scripts/analysis/eval_localization_hateclipseg.py` — visual-only keys. Its module docstring
   (lines 15-18) already states "HateClipSeg has no transcripts extracted, and a
   per-video-constant text half contributes ZERO within-video temporal signal by construction",
   and the code never references `text_feats`. This is a **pre-existing, correct disclosure**.
2. `scripts/analysis/p6_eval_localization.py` — reads `data/MLLM_scores/HateClipSeg/*.jsonl`.
   The MLLM's text channel is **Whisper ASR fed into the prompt**, not the CLIP text tower.

| file | verdict |
|---|---|
| `research-wiki/EVAL_localization_hateclipseg.md` (AP 0.545 / AUC 0.588 / wv-AUC 0.526) | **CLEAR** — visual-only key, deviation declared in §2 |
| `research-wiki/EXP_p6_mllm_localization.md` (wv-AUC 0.5435 vs memory 0.5140) | **CLEAR** |
| `research-wiki/EXP_p10_loc_amplify.md` (P10-b 0.5755) | **CLEAR** |
| `research-wiki/EXP_p11_weaksup_localization.md` (uses the 1024-d subclip cache) | **CLEAR** |
| `research-wiki/{PAPER_MASTER_TABLES,DRAFT_abstract,DRAFT_experiments_chapter,DRAFT_analysis_chapter,ITERATION_LOG,TERMINUS_mllm_campaign_DRAFT,CAMPAIGN_mllm_method_role}.md`; `refine-logs/{D7_RULING_DOSSIER,TERMINUS_round3_mllm_plus3,LITSWEEP5_TEMPORAL,LITSWEEP6_PARADIGM}.md` | **CLEAR** — all re-quote the two paths above |
| `research-wiki/EXP_cvoi_acquisition_KILL_2026-08-09.md`, `HEADTOHEAD_FEASIBILITY.md`, `EXP_p3_evidence_pooling.md` | **CLEAR** — no HateClipSeg result number of ours |

**One design-level exposure, with no number attached.** `scripts/tera_gate0/data.py:123, 193-201`
builds HateClipSeg features as `x_whole = concat(l2n(img), l2n(txt))` and
`x_seg = concat(l2n(seg_img), l2n(whole_txt) broadcast over k)`, i.e. **768 of 1792 dimensions
are the same constant for all 395 videos**. Registered in
`research-wiki/EXP_tera_gate0_impl_appendix.md` §2.3-2.4 and ratified as OP-2 in
`..._impl_appendix_review.md`.

- **Disclosure gap (real, but harmless so far):** both documents justify keeping the text half on
  the grounds that it is *constant across `k`* ("a per-video additive constant cannot change a
  within-video ranking") and that "it only strengthens the video-level baselines". Neither notices
  that on HateClipSeg it is *also constant across videos*, so it strengthens nothing — it is pure
  zero-information padding on every arm including A0. The stated justification is wrong for this
  dataset even though its conclusion (does not change within-video ranking) still holds.
- **Why no number is at risk:** the Gate-0 run stopped at `NO-GO-C` on **HateMM** Gate-C; the
  consumed `hateclipseg_val` confirmation supports no registered claim and is under the §4 seal
  (`refine-logs/TERA_GATE0_CAMPAIGN_RECORD_2026-08-07.md` §5 erratum 1).
- **Verdict: FOOTNOTE on the two Gate-0 appendix documents** (correct the OP-2 rationale), and a
  standing note that any *future* HateClipSeg arm under `tera_gate0/data.py` is running on a
  1792-d vector that is 43 % constant. Since the whole-corpus text half is identical, appending it
  is a monotone transform of cosine similarity *within* HateClipSeg (all vectors have norm √2, so
  `cos = (⟨î_i,î_j⟩+1)/2`) — but that protection **disappears the moment the memory bank is
  cross-dataset**, which is exactly the swappable-memory design.

### 3b. HateMM-side — this is where the real exposure is

Two distinct text channels exist and only one is degenerate. **CLIP `text_feats` (768-d)** is
transcript-only → fully constant on the 74. **Qwen2.5-VL `text_feats` (3584-d)** is built from
frames + title + transcript, so those rows stay distinct — transcript *content* is missing, the
vector is not. Below, "CLIP-text-inclusive" means the first channel.

| conclusion / document | verdict | reasoning |
|---|---|---|
| **"HateMM is TEXT-carried": CLIP text-only kNN AUC 0.847 ≥ image-only 0.826** — `refine-logs/HATEMM_LORA_STREAM_DECOMP.md` §Q1, propagated into `DRAFT_analysis_chapter.md` and `DRAFT_experiments_chapter.md` | **RECHECK (highest priority)** | Text-only key, 0.021 margin, and 39/744 memory rows are one point in a 92 %-non-hate stratum. The doc's own §Q3 "empty-transcript control" tested the *Qwen* encoder's near-zero-norm count (1/744) and a dev-query restriction — it never removed the 39 degenerate rows from the **CLIP** memory bank and never re-ran 0.847. This is the one conclusion whose sign could plausibly move. |
| **"Frozen Qwen2.5-VL beats CLIP on HateMM by +4.2 acc / +4.4 mF1, crossing 0.85"** — `PAPER_MASTER_TABLES.md` T1.1 (0.8279/0.8172), `DRAFT_experiments_chapter.md`, `DRAFT_analysis_chapter.md`, `DESIGN_iter1.md`, `ITERATION_LOG.md`, `experiments/exp-encoder-3seed.md`, `gap_map.md`, `query_pack.md` | **RECHECK** | A between-encoder delta where **only one side is degraded**: the CLIP arm loses the transcript entirely on 74 videos, the Qwen arm keeps frames+title. `refine-logs/ENCODER_SWAP_DIAGNOSIS.md` T6 names this ("5.6 % degenerate CLIP text embeddings … slightly handicaps CLIP's HateMM text stream") but demotes it to "secondary" using an argument about *MHC*, not a measurement on HateMM. Direction here is *deflating* for CLIP, so the gain is an upper bound — the claim likely survives, but the magnitude is not clean. |
| **Every `Z_best` 8960-d conditional-information KILL** — `W2A_PROBE_RECORD.md` (−0.0000), `APX_GATE_RECORD.md` (−0.0038), `CTF_GATE_RECORD.md` (+0.0000), `GIR_GATE_RECORD.md` (+0.0012), `LAUD_GATE_RECORD.md` (+0.0014), `CLAP_GATE_RECORD.md`, `C3_FUSION_PROBE_RECORD.md`, plus `W2A_PROBE_VERDICT_REVIEW.md`, `W2A_CODE_REVIEW.md`, `REDTEAM_BAN_SCOPE_AUDIT.md`, `EUM_FORENSIC_RECON.md`, `OCR_FORENSIC_RECON.md`, `AUDIO_AXIS_FORENSIC_RECON.md` | **FOOTNOTE** | `Z_best` = CLIP img 1024 ⊕ **CLIP text 768** ⊕ Qwen img 3584 ⊕ Qwen text 3584. The degenerate block gives the *conditioning* model a free high-purity cluster, i.e. it **raises the bar** a candidate must clear ⇒ these kills are systematically **conservative**, never inflated. A kill that is too strict cannot manufacture a false positive. Footnote only; but the three whose CI straddles zero (`+0.0014 [−0.0073,+0.0106]`, `+0.0012`, `+0.0000 [−0.0031,+0.0031]`) should not be quoted as *exact* zeros. |
| **FA / premise-(d) "calibrated kill" positive control: HateMM Qwen-concat vs CLIP-concat Δacc +0.0467 / Δhate +0.1163** — `refine-logs/FA_GATE_RECORD.md`, `PREMISE_D_GATE_RECORD.md`, echoed in `PAPER_MASTER_TABLES.md` T-round5 | **FOOTNOTE** | Same asymmetry as the encoder claim: the CLIP-concat comparator is depressed, so +0.0467 is an upper bound. The gate's *logic* (detector fires on a known-good win) is unaffected — an inflated positive control still demonstrates sensitivity. The MHC kills it calibrates stand. |
| **`refine-logs/ERRPAT_HateMM_2026-07-26.md` §4-§5 — the silent-video failure story (FN1)**: 0-for-30 empty-transcript test items predicted non-hate in all 3 seeds, attributed to "the memory bank's length-conditional class prior" / "retrieval geometry" | **RECHECK (mechanism, not number)** | The number is right and reproduces here (P(hate\|empty)=0.068). The *explanation* is more complicated than it needs to be: on any CLIP-text-containing key those rows are literally the same point, so they must retrieve each other first, and the 0.068 hate rate of that cluster **is** the prediction. This reframes a "bias in retrieval geometry" finding as a feature-extraction defect. Rewrite, don't re-run. |
| `refine-logs/C3_NONTARGET_PILOT_RECORD.md` — the CLIP `text_pca_k{8,16,32,64}` ladder | **FOOTNOTE** | PCA over a matrix with a 39-fold repeated row; components are tilted toward that cluster. Direction unknown, magnitude small. |
| `refine-logs/ROUTER_GATE_RECORD.md` — oracle headroom +0.0498, deployable +0.0000 | **FOOTNOTE** | Already carries an `empty-transcript indicator` meta-feature, uncommented. The deployable +0.0000 is unaffected; the *oracle* headroom is partly the router learning to detect these 74. |
| `research-wiki/EVAL_localization_hatemm.md`, `experiments/exp-tarc-t0.md`, `experiments/TARC_VERDICT_AUDIT.md`, `refine-logs/BIDIR_STAGE1_VERDICT_REVIEW.md`, `HEADTOHEAD_FEASIBILITY.md`, and the cite-only long tail (`PAPER_CONSISTENCY_AUDIT_*`, `LITSWEEP*`, `WAVE*_CANDIDATES`, `B1/B2/B3/B4`, `REDTEAM_*`, `EXHAUSTION_AUDIT`, `research-wiki/{log,index,gap_map,DECISION_MEMO_pending}.md` …) | **FOOTNOTE (inherited)** | All quote the frozen-CLIP HateMM floor 0.8279 / 0.8172. One caveat attached to that floor covers the whole tail. |
| `refine-logs/DEGEN_FEATURE_FIX_2026-08-09.md` §4 "historical A0 numbers are safe" | **FOOTNOTE** | True and well-evidenced **for the image side**, which is what it measured (arm 0 PRE/POST differs in exactly one row). It should say so explicitly: the text-side degeneracy was §2d/§5-item-4, was never instrumented, and points the other way (§2d above). |
| All frozen-Qwen / LoRA-Qwen HateMM arms (`FRAME16_*`, `FUSIONCAT_*`, `KSWEEP_RECORD`, `VISION_UNFREEZE_*`, `CAND2_*`, `LORA_HATEMM_*`, `SWA_PROBE_RECORD`, `GRADNORM_SELECT_PROBE_RECORD`, `MNTP_S1_RECORD`, `VSW_ASYMMETRY_RECON`, `LSMI_GATE_RECORD`, `ELR_FORENSIC_RECON`, `INSTRUMENT_VALIDATION_RECON`, `READOUT_*`, `HEADRECIPE_*`) | **CLEAR of the constant-vector defect** | Qwen text vectors stay distinct. They are still *transcript-blind* on the same 74 videos, which is worth one sentence wherever a text-uniqueness claim is made (e.g. `LSMI_GATE_RECORD.md`'s "text uniqueness U2 is the largest atom"), but no vector collapses. |
| `refine-logs/W2B_*` (per-segment CLIP text, MHC only), `TVB_FORENSIC_RECON.md`, `research-wiki/ABLATION_transcript_vs_archive.md` | **CLEAR** | No HateMM CLIP-text instance involved. |
| `research-wiki/ideas/silence-route-crosstower-imputation.md`, `refine-logs/LITSWEEP2_INPUT_FIDELITY.md` | **CLEAR** | Already describe the phenomenon correctly; no numbers. |

### 3c. Summary count

- **RECHECK: 3** — the "HateMM is text-carried" 0.847-vs-0.826 margin; the CLIP-vs-Qwen +4.2
  encoder delta (magnitude only, direction is safe); the ERRPAT FN1 mechanism story (rewrite).
- **FOOTNOTE: ~8 clusters** plus a large inherited cite-only tail sharing one caveat on the
  0.8279 / 0.8172 CLIP floor.
- **CLEAR: everything HateClipSeg-numbered**, and every Qwen-encoder HateMM arm.

### 3d. Cheapest way to close the two RECHECKs (not executed here)

Both are CPU-scale and touch **no test set**: recompute the CLIP text-only and img-only train-LOO
AUCs on HateMM train after deleting the 39 empty-text rows from *both* query and memory side, and
report the 0.847 / 0.826 pair on the clean 705. If the ordering survives, the text-carried claim
is confirmed and the encoder delta gets a magnitude band. Same 705-row restriction applied to the
frozen-CLIP floor gives the clean comparator for the +4.2 claim.

---

## 4. Open items handed forward (nothing fixed in this note)

1. `hateclipseg_prep.py:106` → populate `text` from
   `data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl` (394/395 available, mean 2507
   chars) and re-extract the HateClipSeg video-level CLIP text cache. Supersedes
   `DEGEN_FEATURE_FIX_2026-08-09.md` §5 item 4 for the HateClipSeg half.
2. `generate_VideoCLIP_embedding_HF.py:244-257` → an empty/whitespace transcript should be
   flagged (`EMPTY_TEXT`) rather than silently encoded as `[BOS, EOS]`, mirroring the
   `degen_flags` mechanism already built for the image side.
3. HateMM's 74: 12 sparse + 2 real-speech rows are recoverable from the existing Whisper cache;
   the other 60 are genuinely speechless and should be *flagged*, not imputed.
4. `EXP_tera_gate0_impl_appendix.md` §2.3 / `..._review.md` OP-2 → the "constant across `k`"
   rationale needs the "and constant across videos on HateClipSeg" correction.
