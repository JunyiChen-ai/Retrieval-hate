# DESC_CHANNEL — prompt iteration log (smoke phase)

Smoke set (frozen in `gen_desc.py::SMOKE_IDS`, 8 videos): 2 strictly empty-transcript
(`hate_video_1`, `non_hate_video_119`), 2 further DEFECT videos (`non_hate_video_318` garbled
music-only, `hate_video_383` garbled long ASR), 4 ordinary videos with usable transcripts
(`hate_video_104`, `non_hate_video_30`, `hate_video_100`, `non_hate_video_590`).
Endpoint: realtime, `qwen3-vl-plus`, `temperature=0.0`, `seed=20260813`, `max_tokens=700`.

## V1 (2026-08-13 07:07) — `idea-stage/desc_channel/smoke_v1.jsonl`

8/8 parsed as valid JSON with all six keys. **0/8 forbidden-word violations.**
Mean tokens 2064 in / 329 out. ~8.6 s per item.

Human check (all 8 read in full):

- `hate_video_1` (transcript literally empty, label 1): the model transcribed the full
  slur-bearing text card verbatim into `on_screen_text` and kept every other field purely
  descriptive ("text-card/slideshow", "none visible"). This is exactly the evidence the ASR
  channel cannot see, recovered without any judgement leaking out.
- `non_hate_video_119` (transcript literally empty, label 0): church interior, ~30–40 people
  ducking, "ACTIVE SELF PROTECTION" watermark, typed as archival footage. Accurate.
- `non_hate_video_318` (music-only garbled ASR): correctly typed `video-game capture`,
  scoreboard text transcribed.
- `hate_video_383` (garbled long ASR): panel discussion, burned-in subtitles read verbatim,
  and `audio_visible_cues` correctly noted subtitles synchronised with mouth movement.
- `hate_video_100`: OCR cache for this video is garbage (`"S \nS \nS …"`) yet the model read
  "THE COMPLETE | JOHNNY REBEL" off the frames — it is not merely parroting the OCR input.
- Weakness: `audio_visible_cues` collapses to "no visible audio cues" on 6/8 items;
  `production_format` is terse.

## V2 (2026-08-13 07:08) — `idea-stage/desc_channel/smoke_v2.jsonl`

Changes vs V1: (a) `actions` asked to be concrete about who moves and what changes;
(b) `production_format` asked to give visual evidence when footage looks like an earlier era
(film grain, 4:3, monochrome, period clothing/vehicles); (c) hard rule 1 extended — describe a
symbol/gesture/flag/uniform by visual appearance only; (d) hard rule 5 — 1–3 sentences per
field, never leave a field empty.

8/8 parsed. **0/8 forbidden-word violations.** Mean tokens 2168 in / 291 out.

Difference that matters: on `non_hate_video_119` V2 returns
"archival footage, likely shot on consumer-grade video camera (low resolution, slight motion
blur, 4:3 aspect ratio, no modern production elements)" against V1's bare "archival footage".
`IDEA_REPORT.md` §9.2 lists 1950s archival segregationist footage among the HateMM false
positives, so era evidence is directly on the failure mode this channel is aimed at.
`audio_visible_cues` did not improve.

## Decision

**V2 is frozen as the production prompt** (stored verbatim in `prompts.py`), on the era-evidence
difference. Neither prompt produced a single forbidden-word violation on the smoke set, so the
violation machinery is exercised in code but was not triggered here.

Smoke cost: 16 realtime calls, 34.6 K input + 4.9 K output tokens ≈ **¥0.11**.
