# DESC_CHANNEL — freeze

**Frozen 2026-08-13 (Pacific/Auckland), before any downstream metric was computed.**
Repo HEAD at freeze time: `6e8f7c63b087fc18a689e1adc53a25f0d2144bb9`.
Nothing in §§2–8 is edited after results exist. Results go to
`idea-stage/DESC_CHANNEL_RESULT.md`.

---

## 1. Question, and what the method claim is

The measured motivation, all pre-existing:

- `idea-stage/IDEA_REPORT.md` §9.2 — of the 26 HateMM test errors of the round-4 comparator,
  the **M** bucket (transcript empty or music-only) is worth **+0.9** macro-F1 under an oracle
  fix, and the **O** bucket (decisive evidence burned into on-screen text) another +1.9.
- `refine-logs/EMPTY_TEXT_AUDIT_2026-08-09.md` — **74/1066** HateMM videos carry a literally
  empty upstream `Title`+`Transcript` (train 39 / val 9 / test 26, i.e. **12.1 %** of test vs
  **5.2 %** of train). Whisper confirms 60 of the 74 are music/ambient with no intelligible
  speech, so this is a real input hole, not a decode bug.
- `idea-stage/STANCE_PILOT_RESULT.md` — a frontier VL model asked to make a **judgement**
  about a video scores 0.257 against a 0.70 bar and is only accurate on items the detector
  already gets right. Its **perceptual** outputs, by contrast, were sound (it correctly typed
  archival newsreel, on-screen speaker, caption overlay). This experiment therefore buys
  **perception only** and forbids judgement.

**Method claim under test (headline):** *measurement-guided targeted repair*. A frozen,
label-free input-defect detector (§3) selects the ~20 % of videos whose ASR text channel is
empty or garbled; only those videos get an MLLM perceptual description substituted into the
text channel. The claim is **not** "adding captions helps"; it is "repairing the specific
inputs a measured error taxonomy says are broken, under a gate, helps — and does so beyond what
undifferentiated captioning of every video buys."

**Undifferentiated captioning of all videos (arm B, §5) is run as an analysis control, not as
the claim.** If arm B's gain is ≥ arm G's gain, the gate has no incremental value and this is
reported as such (§8, clause 5). See `DESC_CHANNEL_RESULT.md` §"与现有工作的区分" for the
per-paper comparison against the caption-then-classify family
(`research-wiki/MLLM_USAGE_LANDSCAPE.md` red line).

Prior negative result this design must respect: `idea-stage/A0_OCR_E2E_RESULT.md` — a new
768-d channel routed through the same learnable third stream cost **−0.0246** val macro-F1.
A new channel entering a learnable fusion is *not* free. Hence the mismatch and
parameter-matched-noise controls in §5 are mandatory, not optional.

---

## 2. Step 1 — description generation (frozen spec)

### 2.1 Inputs per video

- **8 frames**, the pre-existing uniform-in-time 8-frame extraction in
  `data/lora_frames/HateMM/<id>/frame_{0..7}.jpg` (produced by
  `src/utils/build_lora_sft_data.py::ensure_frames`, decord with PyAV fallback, 8 evenly
  spaced frames, short side 480). Re-encoded to JPEG q80 with **max side 512** before upload,
  identical to the tier used in `idea-stage/stance_pilot/run_pilot.py`.
  - `hate_video_95` had no frame directory (the file is truncated; decord and PyAV both fail).
    Its 8 frames were extracted with `ffmpeg -err_detect ignore_err` at
    `t = (i+0.5)/8 × 448.749 s`; the 8th seek landed past the truncation, so frame 7 is a copy
    of frame 6 — the same padding policy `ensure_frames` itself uses. Recorded here as the one
    deviation in frame provenance; 1065/1066 videos use the untouched cache.
- **OCR text** for the video from the existing cache
  `data/OCR/HateMM/ocr_video.jsonl` (train+val, 851 rows) and `ocr_video_test.jsonl`
  (test, 215 rows) — all 1066 covered. Lines deduplicated preserving order, joined with
  `" | "`, truncated to 4000 characters. **The OCR cache is read-only; it is not modified.**
- **No transcript, no title, no label, no dataset name, no split name.** The description
  channel must be independent of ASR — that independence is the whole point, since the channel
  exists to repair videos whose ASR is empty.

### 2.2 Output schema (frozen, purely perceptual)

A single JSON object, exactly these six string keys, English:

| key | content |
|---|---|
| `scene` | 场景与环境 — setting, location, indoor/outdoor, lighting,背景 |
| `people` | 出场人物 — how many, apparent appearance/dress, whether anyone addresses the camera |
| `actions` | 动作与事件 — what physically happens across the frames, in temporal order |
| `on_screen_text` | 画面文字逐字 — on-screen text transcribed verbatim (from the frames and the supplied OCR), no paraphrase |
| `production_format` | 制作形态 — one or more of: selfie/talking-head, news segment, animation, video-game capture, archival footage, text-card/slideshow, music video, screen recording, compilation, other (free text allowed after the type) |
| `audio_visible_cues` | 音频可见线索 — only what is *visible* about audio: burned-in subtitles, karaoke/lyric captions, waveform overlays, cut rhythm suggesting music, visible instruments, mouth movement vs. no speaker |

### 2.3 Forbidden-word rule (frozen)

The prompt forbids any judgement of harmfulness. After generation, every field **except
`on_screen_text`** is scanned, case-insensitively, for the frozen banned list:

```
hate, hateful, hatred, offensive, offend, racist, racism, racial slur, slur,
derogatory, bigot, bigoted, bigotry, supremacist, supremacy, extremist, extremism,
propaganda, harmful, harm, toxic, abusive, abuse, discriminat, antisemit, anti-semit,
xenophob, homophob, transphob, misogyn, sexist, sexism, nazi, kkk, radicalis, radicaliz,
仇恨, 攻击, 冒犯, 歧视, 种族主义, 极端, 有害, 辱骂
```

`on_screen_text` is exempt because a verbatim transcription of on-screen text must be allowed
to contain whatever is on screen; exempting it is the only way the field can do its job.
(`nazi`/`kkk` are on the list for the other five fields: naming a symbol there would be an
inference about content rather than a transcription.)

Procedure: an item with ≥1 violating field is **regenerated once** with the identical prompt.
If it still violates, **each still-violating field is set to the empty string** and counted.
Counts of (items flagged round 1, items still violating after regeneration, fields blanked) are
reported.

### 2.4 Model, endpoint, budget

- Model `qwen3-vl-plus` (the strongest VL tier this account exposes; the Batch API refuses
  pinned snapshots — see `STANCE_PILOT_RESULT.md` §6 — so the moving alias is used and the run
  date is recorded instead).
- **Smoke**: 8 hand-checked videos via the realtime endpoint, **at least 2 with an empty
  transcript**. Prompt may be iterated during smoke; every iteration is logged in
  `idea-stage/desc_channel/PROMPT_LOG.md` and the final prompt text is stored verbatim in
  `idea-stage/desc_channel/prompts.py`.
- **Main run**: all 1066 videos via the **Batch API**, single submission, idempotent resume
  (an id already present in the output file is never re-requested).
- **Budget for step 1: ≤ ¥15.** Estimated from `STANCE_PILOT_RESULT.md` §7 measured rates:
  ≈2.5 K input + ≈0.4 K output tokens per item × 1066 items at batch (50 %) rates ≈ **¥4.4**.
- Output `idea-stage/desc_channel/descriptions_hatemm.jsonl`, one row per video:
  `{"id", "fields": {...6 keys...}, "n_violations", "regenerated", "raw_len", "model", "ts"}`.
- Logs `logging/runs/desc_channel/run.log`, PID `logging/runs/desc_channel/run.pid`.
- Items the vendor refuses (DashScope input moderation rejected 1/99 items in the stance pilot)
  are recorded with `fields = null` and counted; their downstream vector is the same
  empty-string encoding any empty description gets. No re-prompting to dodge moderation.

---

## 3. Input-defect detector (frozen, label-free, input-only)

Computed from `data/gt/HateMM/{train,val,test}.jsonl` `text` field only. No labels, no model
output, no test metric involved.

- Lexicon: `/usr/share/dict/american-english` (102,485 entries), lowercased.
- Tokens: `re.findall(r"[a-zA-Z']+", text)`, lowercased, stripped of leading/trailing `'`,
  empties dropped. `T` = token count, `U` = number of tokens present in the lexicon.
- `nwr` (non-word rate) `= 1 − U/T`, and `nwr := 1.0` when `T = 0`.

> **DEFECT(i) ⟺ U_i < 10 OR nwr_i ≥ 0.30**

Frozen counts on HateMM (computed before any arm was trained, from inputs only):

| | train (744) | val (107) | test (215) | total (1066) |
|---|---|---|---|---|
| literally empty text | 39 | 9 | 26 | 74 |
| **DEFECT** | **140** | **26** | **52** | **218 (20.5 %)** |

The `U < 10` clause catches empty and near-empty ASR (`"Yeah."`, `"🎼 Thank."`); the
`nwr ≥ 0.30` clause catches long-but-garbled ASR (`hate_video_383`: 61 tokens, nwr 0.311).
Thresholds were chosen from the input distribution alone (U percentiles 0/0/1/3 at the
2/5/10/15th percentile; nwr 99th percentile 0.50) and are frozen here before any training run.

---

## 4. Step 2 — encoding the description into a vector

**Encoder (frozen):** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`,
`max_seq_length = 512`, mean pooling, 768-d — i.e. **the project's existing long-text→vector
path for transcripts**, `scripts/generate_transcript_embedding.py` (written precisely because
the CLIP text tower truncates transcripts at 77 tokens). It emits the archive-cache contract
`{ids, text_feats [N,768], labels}` that `--archive_feats` consumes with **zero `src/` change**.

Why not the LoRA-Qwen text tower (the encoder that produced the main `text_feats`): the LoRA
adapter directory does not exist on this workstation (`logging/lora/` is absent — the adapter
lived on the retired machine), so re-encoding in the main text space is impossible. This is
stated rather than worked around; the consequence is that the description channel enters as a
**third stream** rather than by literally overwriting `text_feats` rows, and the "targeted
replacement" is realised inside that third stream (§5, arm G) instead.

A local copy of `scripts/generate_transcript_embedding.py` adapted to this repo's paths is
written as `idea-stage/desc_channel/build_desc_feats.py`; the encoder id, `max_seq_length`,
pooling and output contract are identical.

Text fed to the encoder per video:
- **description text** = the six fields joined as
  `"Scene: {scene}\nPeople: {people}\nActions: {actions}\nOn-screen text: {on_screen_text}\nFormat: {production_format}\nAudio cues: {audio_visible_cues}"`;
- **transcript text** = the `text` field of `data/gt/HateMM/<split>.jsonl` verbatim.

Both are encoded by the same call with the same settings. An empty string encodes to whatever
the encoder returns for an empty string — deliberately not special-cased, because that constant
vector *is* the "no usable text" state the repair is supposed to remove.

---

## 5. Arms (7 arms × 3 seeds = 21 head-level runs, single submission)

Backbone for every arm = the **HateMM best cell of the component ablation**
(`idea-stage/RGCL_ABLATION_RESULT.md` §2/§3): encoder `LORA`
(`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`), loss rung **L1** (`--contrast_mode none`, BCE only),
readout **I1** (classifier head). That cell's published test macro-F1 is **0.8774 ± 0.0041**.

Exact command (identical to `scripts/rgcl_ablation_grid.sh`, LORA/HateMM/L1, plus the arm flags):

```
python ./src/run_rac.py \
  --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
  --dataset HateMM --model Qwen2.5-VL-7B-Instruct-LoRA-curric_HF \
  --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align \
  --hard_negatives_loss True --no_hard_negatives 1 --final_eval False \
  --seed {SEED} --group_name DESC_CHANNEL_20260813 \
  --metric cos --loss triplet --batch_norm False --hybrid_loss True --warmup 5 \
  --majority_voting arithmetic --no_pseudo_gold_positives 1 --lambda_seg 0 \
  --contrast_mode none --exp_comment "_LORA_{ARM}" \
  --Faiss_GPU False --force False \
  [--archive_feats idea-stage/desc_channel/feats/{split}_{ARM}.pt --archive_mode stream]
```

| arm | third-stream vector for video *i* | role |
|---|---|---|
| **A0** | *(no `--archive_feats`; `classifier_hateClipper`)* | baseline |
| **T** | `mpnet(transcript_i)` for all *i* | channel-only control: the same extra stream and the same +0.79 M params, carrying **no MLLM output at all** |
| **B** | `mpnet(desc_i)` for all *i* | 接法① — undifferentiated caption-for-everything (analysis control) |
| **G** | `mpnet(desc_i)` if `DEFECT(i)` else `mpnet(transcript_i)` | 接法② — **headline arm**: measurement-gated targeted repair |
| **Bmis** | `mpnet(desc_π(i))` for all *i* | mismatch control for B |
| **Gmis** | `mpnet(desc_π(i))` if `DEFECT(i)` else `mpnet(transcript_i)` | mismatch control for G |
| **N** | fixed random vector, `N(0,1)` 768-d, L2-normalised, RNG seed 20260813 | parameter-matched noise control |

`π` = one fixed derangement of the 1066 ids, `numpy.random.default_rng(20260813).permutation`,
rejected and redrawn if any fixed point remains; **the same π for Bmis and Gmis**, and π maps
within the full 1066-id pool (so a test video can receive a train video's description — this is
a control arm, and it consumes no label).

All seven arms have identical optimiser, loss, epoch budget and selection rule. Arms T/B/G/
Bmis/Gmis/N are byte-identical to each other in every respect except the contents of the
`.pt` files, so they are exactly parameter-matched (`classifier_hateClipperArchive`,
`archive_proj: Linear(768,1024)+Dropout(0.2)`, fusion MLP input 1024→2048).

---

## 6. Protocol

- **train** trains, **val** (`dev_seen`, 107) selects the epoch, **test** (`test_seen`, 215) is
  reported. Same three-way protocol as `RGCL_ABLATION_RESULT.md`, so the numbers are in the
  same frame as the project's published 0.8774.
- Epoch selection (unchanged, reused verbatim from `scripts/rgcl_ablation_analyze.py::parse_run`,
  head readout I1): `argmax` over epochs `≥ warmup(5)` of (dev head acc, dev head roc); report
  test macro-F1 and test ROC at that epoch.
- Seeds **0, 1, 2**, paired by seed.
- **Single submission**: all 21 runs launched in one background process
  (`setsid nohup`, log `logging/runs/desc_channel/run.log`, PID file `run.pid`). No re-run and
  no tuning after any number is seen. A crashed run is reported as a failure, not silently
  re-run with different settings.
- Implementation is validated only on **synthetic random-feature caches** before the real
  submission; no real candidate metric is computed before this file is committed.
- Test labels are used **only** to compute the reported test metrics after the epoch was
  already selected on val. No threshold, hyperparameter or arm is chosen using test.

---

## 7. Reported quantities

1. Test macro-F1 and test ROC-AUC per arm, per seed, mean ± std (3 seeds).
2. Paired-by-seed deltas vs **A0** for every arm.
3. Paired-by-seed deltas **G − T** and **B − T** (isolates the MLLM content from the extra
   stream itself).
4. **Defect-subset readout**: on the 52 DEFECT test videos, the number of correct predictions
   per arm per seed at the selected epoch, mean over seeds, and the paired change vs A0. Also
   reported for the 26 literally-empty-transcript test videos as a nested subset. This readout
   is descriptive; it does not gate the verdict.
5. Description-quality readout: 8 smoke items human-checked, forbidden-word counts, and 2 full
   descriptions of empty-transcript videos quoted verbatim.
6. Measured token counts and the resulting cost.

---

## 8. Decision rule (frozen; primary = seed-paired mean Δ test macro-F1 of arm **G** vs **A0**)

| # | clause | requirement for **GO** |
|---|---|---|
| 1 | `mean(G − A0)` on test macro-F1 | `≥ +0.005` |
| 2 | sign agreement | positive on **3/3** seeds |
| 3 | mismatch control | `mean(Gmis − A0) < 0.5 × mean(G − A0)` **and** `mean(Gmis − A0) < +0.005` |
| 4 | noise control | `mean(N − A0) < +0.005` |

**GO** iff clauses 1–4 all hold. **KILL** otherwise. There is no AMBIGUOUS band: this is a
cheap, CPU-level experiment and a borderline result is a kill.

Clause 5 (**reported, not gating**): if `mean(B − A0) ≥ mean(G − A0)`, the gate carries no
incremental value over undifferentiated captioning, and the result document must say so in
those words, regardless of clauses 1–4.

Clause 6 (**reported, not gating**): `mean(G − T)` is the size of the effect attributable to
MLLM description content rather than to the existence of a third stream. If `mean(G − A0) ≥
+0.005` but `mean(G − T) ≤ 0`, the gain is a third-stream artefact and must be reported as
such.

---

## 9. What this experiment cannot show

- It is one dataset (HateMM) and one backbone cell (LORA/L1/I1). No claim of generality.
- The description channel enters as a third stream, not as a literal `text_feats` overwrite
  (§4). "Targeted replacement" here means "the gated stream carries the description instead of
  the transcript embedding for defective videos".
- 3 seeds cannot separate a +0.005 effect from seed noise with any confidence; the frozen bar
  is a screening bar, not a significance test. Cross-hardware drift (~1.4 pt) is why every arm,
  including the baseline, is re-run here rather than compared against the published 0.8774.
