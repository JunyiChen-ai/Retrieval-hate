# data/AV2A_wav/DeHate — provenance

- Generated 2026-09-26 on uoa-lab2 (sc474399) by `scripts/dehate/prepare_dehate.py media`, code version "DeHate external validation: data preparation" (2026-09-26).
- Upstream: `~/data/DeHate/<split>/<id>.mp4` (6689 videos, released split folders).
- Command per video: `ffmpeg -vn -map 0:a:0 -ac 1 -ar 16000 -acodec pcm_s16le -f wav`. This is the same as `scripts/repro_campaign/run_av2a.py demux_wav`.
- Outcome (`media_report.json`): 6689 demuxed, 0 without audio.
- Copies on uoa-lab1 and uoa-lab3 (rsync 2026-09-26 08:12) are used for the sharded ASR only.
- No labels are read.
