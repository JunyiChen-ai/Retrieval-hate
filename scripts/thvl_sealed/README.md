# THVL label-blind media preflight

This tool accepts only a steward-produced JSONL manifest with the exact fields
`platform,id,url,split`. Label fields and CSV inputs are rejected. It selects
the SHA-256-smallest ten opaque IDs per split and asks `yt-dlp` for metadata
with `--skip-download`; media, subtitles, captions, and thumbnails are never
written. The workflow is restricted to non-commercial research under CC
BY-NC 4.0 and does not authorize redistribution.
