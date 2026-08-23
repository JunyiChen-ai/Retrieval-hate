---
type: idea
node_id: idea:ocr-provenance-typing
title: "Provenance typing of on-screen text (overlay vs scene)"
stage: archived
outcome: negative
added: 2026-08-08T20:44:09Z
based_on: []
target_gaps: ["gap:G4"]
tags: ["ocr", "fusion", "hatemm", "negative-result"]
---

# Provenance typing of on-screen text (overlay vs scene)

**stage:** `archived`  ·  **outcome:** `negative`

## Thesis
Separate uploader-overlaid text (subtitles, title bars, meme captions = the uploader's own speech act) from in-scene filmed text (signs, clothing) using OCR box persistence and position stability, then fuse the two through separate channels.

## Key risks
KILLED by pilot P-C (2026-08-09): typed OCR loses to plain untyped OCR on 3/3 seeds (-0.0020); the positive gating contrast (+0.0044 vs a duplicated-block control) is 90% reproduced by the label-permuted null (-0.0039); the control was not neutral under weight decay. Typing works as a MEASUREMENT (401 overlay / 448 scene / 255 both of 744) but does not pay through this fusion.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

