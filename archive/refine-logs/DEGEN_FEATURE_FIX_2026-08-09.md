# Degenerate CLIP feature rows — root cause, fix, blast radius

Date 2026-08-09. Trigger: `idea-stage/PILOT_B_RESULT.md` §8 flagged 16 HateMM / 2 MHC_zh /
8 ImpliHateVid feature rows that share byte-identical mean-pooled CLIP image vectors, and
attributed them to "failed decode / failed-ASR artifacts". This note establishes what they
actually are, repairs what can be repaired, marks the rest explicitly, and measures whether
any historical A0 number is at risk.

- Scanner: `scripts/analysis/degen_feat_scan.py` (read-only census of byte-identical rows)
- Fixer: `scripts/analysis/degen_feat_fix.py` — log `logging/runs/degen_feat_fix/run.log`,
  pid file `run.pid`, ledger `artifacts/degen_feat_fix/ledger.json`
- Audit: `scripts/analysis/degen_feat_oof_audit.py` — log
  `logging/runs/degen_feat_oof_audit/run.log`, results `artifacts/degen_feat_fix/oof_audit.json`
- Extractor under investigation: `src/utils/generate_VideoCLIP_embedding_HF.py`
- **Test-set contact:** none at the metric level. Test caches were *read* for the integrity
  census and *rewritten* with flags, but no test label was ever scored: the whole audit is the
  frozen HateMM 5-fold **train** OOF.

---

## 1. Complete census (all splits, `openai_clip-vit-large-patch14-336`)

`degen_feat_scan.py` hashes every `img_feats` row and groups exact byte matches. This is a
complete detector for the failure mode: any two videos that produce the same constant frame
grid land in the same group.

| dataset | split | n | groups | items | zero rows |
|---|---|---|---|---|---|
| HateMM | train | 744 | 3 | 16 | 1 (`hate_video_95`) |
| HateMM | dev_seen | 107 | 1 | 2 | 0 |
| HateMM | test_seen | 215 | 1 | 3 | 0 |
| MHC (EN) | train/dev/test | 549/80/161 | 0 | 0 | 0 |
| MHC_zh | train | 579 | 1 | 2 | 0 |
| ImpliHateVid | train | 1283 | 4 | 8 | 0 |
| ImpliHateVid | dev/test | 325/401 | 0 | 0 | 0 |
| HateClipSeg | test_seen | 395 | 6 | 12 | 1 (`yt_NzvfkIYS5Yg`) |

Crucially, the largest HateMM group (`‖v‖ = 31.5444`) **spans all three splits**: 11 train +
2 dev_seen + 3 test_seen = **16 videos on one vector**. P-B saw only its train slice.

## 2. Root cause — three distinct causes, not one

### 2a. `BLACK_VIDEO` — 16 HateMM items. Source has no picture.

The 16 videos of the `‖v‖ = 31.5444` group have distinct md5s, distinct durations, distinct
resolutions, valid h264 + aac streams, and long real transcripts. They decode **successfully**
and every pixel of every frame is 0. Verified independently with decord (64-frame sweeps over
the full clip: 0 non-black frames) and with the ffmpeg CLI (single frames pulled at 1 s / 25 % /
50 % / 75 %: mean 0.0, std 0.0). These are audio-only uploads with a black video track.
`CLIP(black frame)` is a constant, so all 16 collapse onto one vector spanning both labels.

This is **not** a decode failure and **not** a script fallback. The extractor faithfully encoded
what the file contains. The defect is that the pipeline has **no constant/blank-frame guard**:
`load_video_frames` (`generate_VideoCLIP_embedding_HF.py:185-216`) only tests `if not frames`,
never frame content or variance, so a black video is indistinguishable from a good one.

Ids — train: `hate_video_{76,109,127,298,308}`, `non_hate_video_{25,90,110,308,395,470}`;
dev_seen: `non_hate_video_101`, `hate_video_34`; test_seen: `non_hate_video_140`,
`hate_video_{273,295}`.

### 2b. `DUP_SOURCE` — 27 items. The datasets ship the same video under several ids.

| dataset | ids | evidence |
|---|---|---|
| HateMM train | `hate_video_50`, `non_hate_video_338`, `hate_video_63` | mp4s **byte-identical**, md5 `84f69bdbe438` |
| HateMM train | `hate_video_59`, `hate_video_297` | mp4s byte-identical, md5 `e718f5935977` (same 103.103 s content as the triple, different encode) |
| MHC_zh train | `BV1ka4y1m7Ti`, `BV1UT4y1p7WS` | mp4s byte-identical, md5 `e8e9e9370645` |
| HateClipSeg test | 5 pairs | mp4s byte-identical |
| HateClipSeg test | `bit_T8WbPXzZTebJ`, `bit_jubQcJCZQ3dQ` | different container md5, **byte-identical decoded frames** |
| ImpliHateVid train | `IM_476`/`NH_933`, `EX_80`/`EX_81`, `EX_172`/`EX_224`, `EX_74`/`EX_297` | raw videos not on this machine and not in B2 (`raw/` has no ImpliHateVid); identical `img_feats` and, for 3 of the 4 pairs, identical `text_feats` too. Marked *unverified at source*. |

Identical features here are **correct**. Nothing to fix. `hate_video_50/63` vs
`non_hate_video_338` (labels 1,1,0) and `IM_476`/`NH_933` (labels 1,0) are genuine
**annotation conflicts on identical content**.

**Correction to P-B §8.** P-B diagnosed its 4 surviving HateMM conservative pairs as "failed
decode / failed-ASR artifacts producing a degenerate feature vector". They are not. The videos
decode to real, non-black content (`hate_video_50` frame mean 81.2); the shared 11-character
transcript `"🎼  🎼  Yeah."` is the *upstream* HateMM annotation, copied verbatim by
`scripts/prep_video_dataset.py:126-139` — it is what the dataset ships for that clip, not an ASR
failure in this repo. So all 5 of P-B's conservative pairs are genuine near-duplicate label
conflicts, from 3 distinct duplicate clusters, and the "verified count = 1" restatement in §8 is
too harsh. **The P-B verdict is unaffected**: 5 pairs (or 3 clusters) is still far below the
ALIVE bar of 30, so `duplicate-conflict memory` stays DEAD either way.

### 2c. `DECODE_*` — 2 zero rows. Truncated source files, silently zero-filled.

`generate_VideoCLIP_embedding_HF.py:298-335`: when `load_video_frames` returns `ok=False` the
caller substitutes `torch.zeros(dv)` and only increments a counter. This is the silent-fallback
path P-B suspected, and it fired exactly twice in the whole project.

- **`hate_video_95`** (HateMM train). 37.5 MB, 448.7 s, 13 461 declared frames. decord raises at
  open; PyAV decodes 12 346 frames then raises `InvalidDataError(1094995529)` inside
  `avcodec_send_packet` — the file is truncated. The extractor's PyAV fallback lets that
  exception propagate before it reaches its target index 13 460, so the whole row was zeroed
  **even though 91.7 % of the video is perfectly decodable**. → **repaired**.
- **`yt_NzvfkIYS5Yg`** (HateClipSeg test_seen). 138 358 bytes for a 274 s video; ffmpeg cannot
  determine the stream format, PyAV decodes 0 frames. The B2 copy is the same 138 358 bytes, so
  the upstream artifact itself is truncated. → **unrepairable**, flagged.

### 2d. Related, out of the requested scope but not to be lost: the text stream

The same census run over `text_feats` shows an analogous constant-vector mode from
`generate_VideoCLIP_embedding_HF.py:244-257`: an empty transcript yields `windows = [[]]`, the
text tower is run on `[BOS, EOS]`, and every empty-transcript video receives the identical
vector. Counts: HateMM train 39, dev_seen 9, test_seen 26; ImpliHateVid train 5; MHC_zh 2×2;
MHC 0. **HateClipSeg test_seen: all 395 rows share one text vector** — `data/gt/HateClipSeg/test.jsonl`
has `text: ""` for all 395 rows, i.e. HateClipSeg has no transcript channel at all and its "text"
feature carries zero information. Not fixed here; logged as an open item.

## 3. Fix

`scripts/analysis/degen_feat_fix.py`. Old caches are **never** overwritten; every output is a
new file with suffix `-degenfix1`. Each output carries two new keys:

- `degen_flags`: `{video_id: CODE}` with `BLACK_VIDEO`, `DECODE_TRUNCATED_REPAIRED`,
  `DECODE_FAIL_UNREPAIRABLE`, `DUP_SOURCE_VERIFIED:<md5>`, `DUP_SOURCE_UNVERIFIED:<group>`.
- `degen_meta`: producer, source path, source SHA-256, repair record, code glossary.

Repair recipe for `hate_video_95`: prefix-tolerant PyAV decode (keep everything decodable,
swallow the terminal abort), uniform 8-frame grid over the 12 346 decodable frames, then the
**unchanged** encode path from the extractor (bicubic resize to 336, `CLIPImageProcessor`,
`CLIPVisionModel.pooler_output`, mean over frames). Result `‖v‖` 0 → 30.4326, frame coverage
0.9172 of the declared length. Run took 17 s on the local GPU.

**Repaired 1 of 46 flagged rows. Abandoned 17 as irreparable by re-extraction** (16 black-video
+ 1 truncated-beyond-recovery); the remaining 28 were never broken (duplicate sources).

### SHA-256 ledger

| new cache | rows | flagged | SHA-256 (new) | source SHA-256 |
|---|---|---|---|---|
| `data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF-degenfix1.pt` | 744 | 17 | `ae0269d5077ac72b4bd0f2a87b9677d2fce78c3b91827a98e10c27352830bd28` | `0802b6ba00669ec546e63f36dca1772cb2d7806b969de307235af3450a8176c1` |
| `data/CLIP_Embedding/HateMM/dev_seen_..._HF-degenfix1.pt` | 107 | 2 | `a24dadb4589dd72b3477eb3879feb5b39c7456ad26ca99f82676eb4df184b6d4` | `ab9cd8a070b93afbf994ed876e3adfd9c2a139e82d801af21346c29f17c1888d` |
| `data/CLIP_Embedding/HateMM/test_seen_..._HF-degenfix1.pt` | 215 | 3 | `a3213ff769a267e4596d05370334b538dfb4c5b67163925390fa8f5190dea312` | `e0ee5f74ff13c5ada568dae465cf19883ce78be3bf698d00d1fc03612d17f012` |
| `data/CLIP_Embedding/MHC_zh/train_..._HF-degenfix1.pt` | 579 | 2 | `cfeecf57e8bb1ba742a905008ff07f55135ebf404fd18bf343850247d224913d` | `929571f81a5bf4a7f306eaaeceecafd62c85b73cf86ce8177ab175dbcfff8f17` |
| `data/CLIP_Embedding/ImpliHateVid/train_..._HF-degenfix1.pt` | 1283 | 8 | `07cda2ace111e633966c62809921e7e733844b83cbfdc9427a7efdbb6bd953e0` | `90b4700a2d687862135564580dce10a2b4c7278ff84b7ff4c72d69e1272fb685` |
| `data/CLIP_Embedding/HateClipSeg/test_seen_..._HF-degenfix1.pt` | 395 | 13 | `8f124d933cfcdd8609bf93bf5812373f1cf5fffe89e696230f55ffbb796079fe` | `43227d527d402e1707f770386667cb39114c861f01345c0ab3b9087abedf6f30` |

The three source SHA-256 values that P-B recorded in `idea-stage/pilot_b.json:files` (HateMM
train, MHC_zh train, ImpliHateVid train) match the sources read here — same bytes, same census.

## 4. Blast radius

### Where the degenerate rows land

HateMM's 16 black-video rows split **11 train / 2 val (dev_seen) / 3 test**, so every HateMM A0
number ever produced — train, val and test — contains some of them. Inside the frozen HateMM
5-fold train OOF split (`artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf`), the 17
flagged train rows sit in folds 0/1/2/3/4 = 5/4/1/4/3. No fold is disproportionately loaded.

### Head-level OOF, before vs after

Instrument: the project's existing frozen-space A0 harness (`ocr_fusion_pilot.py` arm 0 =
`l2(img) ⊕ l2(txt)`, linear head, inner-4-fold epoch + threshold selection, the frozen 5-fold
train OOF). Arm id pinned to 0 for both arms so head init and batch shuffling are byte-identical
and only the feature matrix differs. 3 seeds (20260810/11/12). Exactly one row changes between
PRE and POST (`hate_video_95`).

| arm | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| PRE (original cache), OOF macro-F1 over all 744 | 0.8077 | 0.8143 | 0.8092 | **0.8104 ± 0.0035** |
| POST (`-degenfix1`), all 744 | 0.8026 | 0.8064 | 0.8122 | **0.8071 ± 0.0049** |
| paired POST − PRE | −0.0052 | −0.0079 | +0.0031 | **−0.0033** |
| PRE, restricted to the 727 non-degenerate rows | 0.8138 | 0.8205 | 0.8152 | **0.8165 ± 0.0036** |
| POST, restricted to the 727 | 0.8100 | 0.8140 | 0.8186 | **0.8142 ± 0.0043** |

Prediction disagreements between PRE and POST: 10 / 19 / 10 of 744.

### Verdict

**Historical A0 numbers are safe. Nothing needs re-running.**

1. The paired PRE→POST delta is **mixed-sign** (2 negative, 1 positive) and its mean −0.0033 is
   *smaller than* the PRE seed std (0.0035). Repairing the single genuinely-wrong row moves the
   OOF number by less than seed noise. There is no direction to it.
2. Only **1 of the 46** flagged rows across the whole project was actually a wrong vector. The
   other 45 are faithful: black videos really are black, duplicate ids really are the same file.
   A number computed on them is not corrupted, it is merely uninformative on those rows.
3. The degenerate rows **cost** accuracy, they do not inflate it. Their OOF accuracy is 52.9 %
   (9/17, chance level, exactly what a constant image vector plus a text-only signal predicts),
   and deleting them from the evaluation *raises* OOF macro-F1 by +0.006 (0.8104 → 0.8165). So
   every historical HateMM figure is slightly **pessimistic** because of these rows. No published
   conclusion was ever propped up by them.
4. Consequently no result can have been "flipped by these 16 vectors" in the favourable
   direction. The exposure that remains is the reverse one: a *future* method that happens to
   exploit the black-video constant (e.g. a retrieval memory that clusters them) would earn a
   spurious gain. That is what the new `degen_flags` are for.

## 5. Open items (not fixed here)

1. **No blank/constant-frame guard in the extractor.** `load_video_frames` should reject a frame
   grid whose pixel variance is ~0 and mark the row, instead of returning it as a normal vector.
   Same hole in the three sibling extractors that copy the decode block verbatim
   (`generate_subclip_embedding_HF.py`, `generate_VideoMLLM_embedding_HF.py`,
   `generate_video_archive_HF.py`).
2. **Prefix-tolerant decoding should move into the extractor.** `_decode_with_pyav` throws away a
   12 346-frame decodable prefix because of a terminal abort. The recipe in
   `degen_feat_fix.decode_prefix` is the fix.
3. **`generate_VideoCLIP_embedding_HF.py` has no id-subsetting flag** (`--ids/--only/--resume`),
   and `main()` unconditionally rewrites the whole split file — a partial re-extraction destroys
   the rest of the split. This is why the repair here was done as a separate merge-and-save
   rather than by re-running the extractor.
4. **The text-side constant vector (§2d)**, including HateClipSeg's 395/395 empty transcripts.
5. **ImpliHateVid raw videos exist neither locally nor in B2**, so its 8 duplicate rows are
   marked `DUP_SOURCE_UNVERIFIED` and cannot be confirmed at the source until the corpus is
   re-staged.
6. Consumers still read the **original** cache names. The `-degenfix1` files are opt-in on
   purpose; adopting them is a separate decision, and on this evidence it is not urgent.
