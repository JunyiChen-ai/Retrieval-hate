# Frame-level evaluation protocol: DeHate amendment

**Added 2026-09-26.** This amendment is additive. It adds DeHate as an external-validation corpus, on the user's
request of 2026-09-26. The protocol in `FRAME_EVAL_PROTOCOL.md` is unchanged: the 1 fps grid, half-open containment
and the degenerate-span rule all stay the same, and no existing gold array moves. The builder is
`scripts/dehate/prepare_dehate.py` (stage `gt`).

## Source and split

- **Annotations:** `~/data/DeHate/DeHate_labels.csv`, the local export of the released `DeHate.xlsx`. It has 6689
  rows, one per delivered video, and every row has a local `.mp4`.
- **Video label:** the released `Hate` column (0/1).
- **Split:** the released `Split` column (train 4680, val 668, test 1341), preserved as released. Media sit in
  `~/data/DeHate/<split>/<id>.mp4`.
- **Other columns:** the modality flags (`Textual/Visual/Audio Content`), target flags, title and description are not
  used by any method or by the gold.

## Spans

- **Format:** `Hate Segment` holds second pairs, almost always as `[(a, b), ...]`.
- **Parsing:** every `(a, b)` pair in the string is a span. Four rows are malformed but unambiguous: one is
  written `[a, b]`, and three lack the opening bracket. They are parsed as the same pairs.
- **Degenerate spans:** `(0, 0)` and any `end <= start` are dropped and counted, as in the general protocol.
- **Past the end:** a span reaching past the media end is truncated by the grid.

## Duration

The duration is the wav duration of the 16 kHz mono audio. It is taken from the chunk manifest
(`results/reproduction/asr/dehate_all`) first, then from the wav header, and from the container for a video with no
audio stream. The resolver is `scripts/duplex/extract_clip_features.find_duration`, the same one the features use.

## Rule (b) applies to 46% of hateful videos

983 of the 2120 hateful videos (46%) carry no usable span. Almost all of them are text-only hate:

- mean `Textual Content` flag .98, `Visual Content` .04, `Audio Content` .02;
- hateful videos that do have spans: .45 / .66 / .95.

The hate is most likely in the title or description, not on the timeline. These videos fall under rule (b) of the
general protocol, as the MultiHateClip positives without spans do:

- they are excluded from localization evaluation, and are never scored as negatives or positives;
- they stay in training with their video label, because every method is trained from video labels only.

The builder records the exclusions in `results/reproduction/gt/dehate_{val,test}.json`
(`excluded_positive_without_span`).

## Cohort

| split | videos scored | hateful scored | non-hateful | hateful excluded (rule b) |
|---|---:|---:|---:|---:|
| val | 559 | 103 | 456 | 109 |
| test | 1151 | 234 | 917 | 190 |

These are the counts expected from the label file. The builder's sidecar gives the final numbers after the duration
check.
