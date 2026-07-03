#!/usr/bin/env python
"""Collect video upload dates for MultiHateClip EN (YouTube) and ZH (Bilibili).

DESIGN_iter3 E0a. Method validated in research-wiki/TEMPORAL_SPLIT_FEASIBILITY.md:
  - EN : Video_ID is a native 11-char YouTube ID ->
         `yt-dlp --skip-download --print upload_date` (metadata only, no download).
  - ZH : Video_ID is a native BV id -> Bilibili web API
         `api.bilibili.com/x/web-interface/view?bvid=` field `data.pubdate`
         (unix seconds). Offline BV->date decoding is NOT reliable and is not used.

Checkpoint / resume:
  Every single query result is appended immediately to a per-dataset JSONL
  checkpoint (data/gt/<DS>_upload_dates.jsonl). On restart, ids already present
  in the checkpoint are skipped (last record per id wins). Failed ids are also
  skipped on resume unless --retry-failed is given (a retry appends a new
  record; readers must take the LAST record per id).

Record schema (one JSON object per line):
  {"dataset": "MHC"|"MHC_zh", "video_id": str,
   "status": "ok"|"fail",
   "upload_date": "YYYYMMDD" | null,      # ZH date rendered in UTC+8
   "pubdate_ts": int | null,              # ZH only: raw unix seconds
   "error": str | null,                   # failure reason (truncated)
   "queried_at": "...Z"}

CPU + network only. No GPU. Intended to run as a SLURM CPU job (login-node
processes get reaped when SLURM jobs start). Random 1-3 s sleep between
queries to avoid rate limiting.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

# ----------------------------------------------------------------------------
# Fixed paths
# ----------------------------------------------------------------------------
REPO_ROOT = "/data/jehc223/RGCL"
OUT_DIR = os.path.join(REPO_ROOT, "data", "gt")

DATASETS = {
    "en": {
        "name": "MHC",
        "annotation": "/data/jehc223/Multihateclip/English/annotation(new).json",
        "checkpoint": os.path.join(OUT_DIR, "MHC_upload_dates.jsonl"),
    },
    "zh": {
        "name": "MHC_zh",
        "annotation": "/data/jehc223/Multihateclip/Chinese/annotation(new).json",
        "checkpoint": os.path.join(OUT_DIR, "MHC_zh_upload_dates.jsonl"),
    },
}

DEFAULT_YTDLP = "/data/jehc223/home/.local/bin/yt-dlp"
BILI_API = "https://api.bilibili.com/x/web-interface/view"
BILI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
    ),
    "Referer": "https://www.bilibili.com/",
}
CST = timezone(timedelta(hours=8))  # Bilibili pubdate rendered in China time


def log(msg):
    print(msg, flush=True)


def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def truncate(s, n=300):
    s = " ".join(str(s).split())
    return s[:n]


# ----------------------------------------------------------------------------
# Inputs / checkpoint
# ----------------------------------------------------------------------------
def load_video_ids(annotation_path):
    """Ordered unique Video_IDs from a MultiHateClip annotation json."""
    with open(annotation_path, "r") as f:
        data = json.load(f)
    ids, seen = [], set()
    for entry in data:
        vid = entry.get("Video_ID")
        if vid and vid not in seen:
            seen.add(vid)
            ids.append(vid)
    return ids


def load_checkpoint(path):
    """id -> last record. Tolerates a truncated trailing line."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # interrupted write; the id will simply be re-queried
            done[rec["video_id"]] = rec
    return done


def append_record(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------------
# Per-platform queries
# ----------------------------------------------------------------------------
def query_youtube(video_id, ytdlp_bin, timeout=90):
    """Return (upload_date 'YYYYMMDD' or None, pubdate_ts=None, error or None)."""
    url = "https://www.youtube.com/watch?v=" + video_id
    cmd = [ytdlp_bin, "--skip-download", "--no-warnings", "--no-playlist",
           "--print", "upload_date", url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None, "yt-dlp timeout after %ds" % timeout
    except OSError as e:
        return None, None, "yt-dlp exec error: %s" % truncate(e)
    out = proc.stdout.strip().splitlines()
    date = out[-1].strip() if out else ""
    if proc.returncode == 0 and re.fullmatch(r"\d{8}", date):
        return date, None, None
    err_lines = [l for l in proc.stderr.strip().splitlines() if l.strip()]
    err = err_lines[-1] if err_lines else "no upload_date in output (rc=%d)" % proc.returncode
    return None, None, truncate(err)


def query_bilibili(bvid, session, timeout=30, retries=2):
    """Return (upload_date 'YYYYMMDD' or None, pubdate_ts or None, error or None)."""
    err = "unknown"
    for attempt in range(retries + 1):
        try:
            r = session.get(BILI_API, params={"bvid": bvid},
                            headers=BILI_HEADERS, timeout=timeout)
            j = r.json()
        except Exception as e:  # network / json errors -> transient, retry
            err = "request error: %s" % truncate(e)
            time.sleep(5 * (attempt + 1))
            continue
        code = j.get("code")
        if code == 0:
            ts = j.get("data", {}).get("pubdate")
            if isinstance(ts, int) and ts > 0:
                date = datetime.fromtimestamp(ts, tz=CST).strftime("%Y%m%d")
                return date, ts, None
            return None, None, "code=0 but no pubdate"
        if code == -412:  # request blocked (rate limit) -> back off and retry
            err = "code=-412 request blocked (rate limited)"
            log("    [zh] -412 rate limited on %s; backing off 30s" % bvid)
            time.sleep(30 * (attempt + 1))
            continue
        # permanent failures: -404 deleted, 62002 invisible, 62012 private, ...
        return None, None, "code=%s msg=%s" % (code, truncate(j.get("message", "")))
    return None, None, err


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run_dataset(key, args):
    cfg = DATASETS[key]
    name, ckpt = cfg["name"], cfg["checkpoint"]
    ids = load_video_ids(cfg["annotation"])
    done = load_checkpoint(ckpt)

    todo = []
    for vid in ids:
        rec = done.get(vid)
        if rec is None:
            todo.append(vid)
        elif rec.get("status") != "ok" and args.retry_failed:
            todo.append(vid)
    if args.limit > 0:
        todo = todo[: args.limit]

    n_ok = sum(1 for r in done.values() if r.get("status") == "ok")
    log("[%s] annotation ids=%d  checkpointed=%d (ok=%d fail=%d)  querying now=%d"
        % (name, len(ids), len(done), n_ok, len(done) - n_ok, len(todo)))
    if not todo:
        return

    session = None
    if key == "zh":
        import requests
        session = requests.Session()

    t0 = time.time()
    for i, vid in enumerate(todo, 1):
        if key == "en":
            date, ts, err = query_youtube(vid, args.ytdlp)
        else:
            date, ts, err = query_bilibili(vid, session)
        rec = {
            "dataset": name,
            "video_id": vid,
            "status": "ok" if date else "fail",
            "upload_date": date,
            "pubdate_ts": ts,
            "error": err,
            "queried_at": utcnow_iso(),
        }
        append_record(ckpt, rec)
        log("  [%s %d/%d] %s -> %s%s"
            % (name, i, len(todo), vid, date or "FAIL",
               ("  (%s)" % err) if err else ""))
        if i < len(todo):
            time.sleep(random.uniform(args.min_sleep, args.max_sleep))

    # summary over the full checkpoint
    done = load_checkpoint(ckpt)
    n_ok = sum(1 for r in done.values() if r.get("status") == "ok")
    log("[%s] DONE in %.1f min: checkpointed=%d/%d  ok=%d (%.1f%%)  fail=%d"
        % (name, (time.time() - t0) / 60.0, len(done), len(ids), n_ok,
           100.0 * n_ok / max(len(done), 1), len(done) - n_ok))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="en,zh",
                    help="comma-separated subset of {en,zh}")
    ap.add_argument("--limit", type=int, default=0,
                    help="max NEW queries per dataset (0 = all); for smoke tests")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-query ids whose last checkpointed status is fail")
    ap.add_argument("--min-sleep", type=float, default=1.0)
    ap.add_argument("--max-sleep", type=float, default=3.0)
    ap.add_argument("--ytdlp", default=DEFAULT_YTDLP)
    args = ap.parse_args()

    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    for k in keys:
        if k not in DATASETS:
            sys.exit("unknown dataset key: %r (use en,zh)" % k)
    os.makedirs(OUT_DIR, exist_ok=True)
    for k in keys:
        run_dataset(k, args)


if __name__ == "__main__":
    main()
