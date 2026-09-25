# data/frames_1fps/DeHate — provenance

- Generated 2026-09-26 on uoa-lab2 (sc474399) by `scripts/dehate/prepare_dehate.py media`.
- Command per video: `ffmpeg -vf fps=1 -q:v 2 -start_number 0 -frames:v 6000`. This is the same as `scripts/repro_campaign/blip2_caption.extract_1fps`.
- Output: `<id>/%06d.jpg` holds the content at t seconds.
- Coverage: 6688 of 6689 videos. `8bAfN6vXoIZp` (train, non-hateful) has a 0.14 s video stream with 134 s of audio, and ffmpeg writes no frame for it (`data/AV2A_wav/DeHate/media_report.json`).
- Copies on uoa-lab1 and uoa-lab3 (rsync 2026-09-26 08:35) serve the VLM questions (`data/vlm_tree/DeHate`).
- No labels are read.
