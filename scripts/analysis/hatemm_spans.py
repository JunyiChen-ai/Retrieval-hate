#!/usr/bin/env python3
"""HateMM gold time-span verification.

Parses the official Zenodo `hate_snippet` column, aligns each span against the
true video duration (ffprobe), and emits a machine-readable gold file plus a
statistics report.

Read-only w.r.t. raw data. Reusable: has a main() and caches ffprobe durations
to an intermediate jsonl so it can resume if the login node reaps the process.

Outputs:
  data/gt/HateMM/hate_spans.json   -- gold: {video_id: {duration, spans, ...}}
Intermediate (cache, resumable):
  <scratch>/hatemm_durations.jsonl -- {"id":.., "duration":..} one per line
"""
import argparse
import ast
import csv
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(REPO, "data/gt/HateMM/HateMM_annotation.csv")
VIDEO_DIR = "/data/jehc223/HateMM/video"
SPLIT_DIR = os.path.join(REPO, "data/gt/HateMM")
OUT_JSON = os.path.join(REPO, "data/gt/HateMM/hate_spans.json")
FFPROBE = "/data/jehc223/miniconda3/envs/ExMRD/bin/ffprobe"
DEFAULT_CACHE = (
    "/data/jehc223/home/tmp/claude-135258174/-data-jehc223-RGCL/"
    "e8f03e41-3e21-4cea-b12c-29207373bfca/scratchpad/hatemm_durations.jsonl"
)

TIME_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")


def vid_id(fname):
    return fname[:-4] if fname.lower().endswith(".mp4") else fname


def hms_to_sec(t):
    """Parse 'HH:MM:SS' -> seconds (int). Returns None if malformed."""
    m = TIME_RE.match(str(t).strip())
    if not m:
        return None
    h, mi, s = (int(x) for x in m.groups())
    if mi >= 60 or s >= 60:
        return None
    return h * 3600 + mi * 60 + s


def parse_snippet(raw):
    """Parse a hate_snippet cell.

    Returns (spans, errors) where spans is a list of [start_s, end_s] and errors
    is a list of human-readable strings describing anomalies (empty means clean).
    On a hard parse failure, spans is None and errors carries the reason.
    """
    s = (raw or "").strip()
    if s == "" or s == "[]":
        return [], []
    try:
        parsed = ast.literal_eval(s)
    except Exception as e:  # noqa: BLE001
        return None, [f"literal_eval failed: {e}"]
    if not isinstance(parsed, (list, tuple)):
        return None, [f"top-level not a list: {type(parsed).__name__}"]
    spans, errs = [], []
    for iv in parsed:
        if not isinstance(iv, (list, tuple)) or len(iv) != 2:
            errs.append(f"interval not a 2-list: {iv!r}")
            continue
        a, b = hms_to_sec(iv[0]), hms_to_sec(iv[1])
        if a is None or b is None:
            errs.append(f"bad time in interval {iv!r}")
            continue
        if b <= a:
            errs.append(f"end<=start in interval {iv!r} ({a}s,{b}s)")
        spans.append([float(a), float(b)])
    return spans, errs


def load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_splits():
    out = {}
    for sp in ("train", "val", "test"):
        p = os.path.join(SPLIT_DIR, f"{sp}.jsonl")
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        out[sp] = {r["id"]: r["label"] for r in rows}
    return out


def ffprobe_duration(vid):
    path = os.path.join(VIDEO_DIR, vid + ".mp4")
    if not os.path.exists(path):
        return None
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=60,
        )
        val = r.stdout.strip()
        return float(val) if val and val != "N/A" else None
    except Exception:  # noqa: BLE001
        return None


def build_duration_cache(ids, cache_path, hate_only_ids=None):
    """Fill cache_path (jsonl) with durations for `ids`, resuming if present."""
    have = {}
    if os.path.exists(cache_path):
        for l in open(cache_path, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                d = json.loads(l)
                have[d["id"]] = d["duration"]
            except Exception:  # noqa: BLE001
                continue
    todo = [i for i in ids if i not in have]
    print(f"[dur] cached={len(have)} todo={len(todo)}", flush=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        for n, i in enumerate(todo, 1):
            dur = ffprobe_duration(i)
            have[i] = dur
            f.write(json.dumps({"id": i, "duration": dur}) + "\n")
            f.flush()
            if n % 100 == 0:
                print(f"[dur] {n}/{len(todo)}", flush=True)
    return have


def pct(x, n):
    return f"{100.0*x/n:.1f}%" if n else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--no-write-json", action="store_true",
                    help="skip writing hate_spans.json (stats only)")
    args = ap.parse_args()

    rows = load_rows()
    splits = load_splits()
    id2split = {}
    for sp, d in splits.items():
        for i in d:
            id2split[i] = sp

    # ---- parse all snippets ----
    parsed = {}          # id -> (spans_or_None, errs, raw, label)
    hate_ids = []
    for r in rows:
        vid = vid_id(r["video_file_name"])
        spans, errs = parse_snippet(r["hate_snippet"])
        parsed[vid] = (spans, errs, r["hate_snippet"], r["label"])
        if r["label"] == "Hate":
            hate_ids.append(vid)

    n_total = len(rows)
    n_hate = sum(1 for r in rows if r["label"] == "Hate")
    n_nonhate = n_total - n_hate

    # ---- durations (all videos, resumable) ----
    all_ids = [vid_id(r["video_file_name"]) for r in rows]
    dur = build_duration_cache(all_ids, args.cache)

    # ================= STATISTICS =================
    print("\n" + "=" * 70)
    print("HATEMM GOLD SPAN VERIFICATION")
    print("=" * 70)
    print(f"CSV rows: {n_total} | Hate: {n_hate} | Non Hate: {n_nonhate}")

    # parse outcomes among hate
    hard_fail = [i for i in hate_ids if parsed[i][0] is None]
    with_errs = [i for i in hate_ids if parsed[i][0] is not None and parsed[i][1]]
    clean = [i for i in hate_ids if parsed[i][0] is not None and not parsed[i][1]
             and len(parsed[i][0]) > 0]
    empty_span_hate = [i for i in hate_ids
                       if parsed[i][0] is not None and len(parsed[i][0]) == 0]
    print(f"\n-- Parse outcomes (Hate videos, n={n_hate}) --")
    print(f"  clean (>=1 span, no anomaly): {len(clean)} ({pct(len(clean),n_hate)})")
    print(f"  parsed with anomalies       : {len(with_errs)}")
    print(f"  empty span list             : {len(empty_span_hate)}")
    print(f"  hard parse failure          : {len(hard_fail)}")

    # anomaly breakdown
    anomaly_rows = []
    for i in hate_ids:
        spans, errs, raw, _ = parsed[i]
        if errs:
            anomaly_rows.append((i, raw, errs))
    print(f"\n-- Anomaly detail ({len(anomaly_rows)} videos) --")
    for i, raw, errs in anomaly_rows:
        print(f"  {i}: {errs} | raw={raw}")
    if not anomaly_rows:
        print("  (none)")

    # non-hate with unexpected non-empty snippet
    nh_nonempty = [i for i in parsed
                   if parsed[i][3] == "Non Hate" and parsed[i][0]]
    print(f"\n-- Non Hate rows with non-empty span: {len(nh_nonempty)} --")
    for i in nh_nonempty[:20]:
        print(f"  {i}: {parsed[i][2]}")

    # span-count / span-duration / coverage distributions (hate w/ spans)
    span_counts = []
    span_durs = []
    cover_ratios = []       # sum(span)/duration, spans clamped, over hate w/ dur
    ge95 = []               # coverage >=95%
    ge99 = []
    missing_dur = []
    overrun_rows = []       # end > duration
    for i in hate_ids:
        spans = parsed[i][0]
        if not spans:
            continue
        span_counts.append(len(spans))
        d = dur.get(i)
        # raw span durations (pre-clamp)
        for a, b in spans:
            span_durs.append(max(0.0, b - a))
        if d is None or d <= 0:
            missing_dur.append(i)
            continue
        # overrun
        max_end = max(b for _, b in spans)
        if max_end > d + 0.5:
            overrun_rows.append((i, max_end, d, max_end - d))
        # clamped coverage (union, clamped to [0,d])
        clamped = [[max(0.0, a), min(d, b)] for a, b in spans]
        clamped = [(a, b) for a, b in clamped if b > a]
        # union length
        union = 0.0
        for a, b in sorted(clamped):
            union = union  # placeholder
        merged = []
        for a, b in sorted(clamped):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        union = sum(b - a for a, b in merged)
        ratio = union / d
        cover_ratios.append(ratio)
        if ratio >= 0.95:
            ge95.append(i)
        if ratio >= 0.99:
            ge99.append(i)

    def dist(xs):
        if not xs:
            return "n/a"
        xs = sorted(xs)
        n = len(xs)
        q = lambda p: xs[min(n - 1, int(p * n))]
        return (f"min={xs[0]:.1f} p25={q(.25):.1f} median={q(.5):.1f} "
                f"p75={q(.75):.1f} p90={q(.9):.1f} max={xs[-1]:.1f} mean={sum(xs)/n:.1f}")

    print(f"\n-- Span-count per hate video (n={len(span_counts)}) --")
    print("  ", Counter(span_counts))
    print(f"\n-- Span duration seconds (n={len(span_durs)} spans) --")
    print("  ", dist(span_durs))
    print(f"\n-- Coverage ratio union(span)/duration (n={len(cover_ratios)}) --")
    print("  ", dist(cover_ratios))
    print(f"  >=95% coverage: {len(ge95)} ({pct(len(ge95),len(cover_ratios))})")
    print(f"  >=99% coverage: {len(ge99)} ({pct(len(ge99),len(cover_ratios))})")

    print(f"\n-- Duration alignment --")
    print(f"  hate videos missing/zero duration: {len(missing_dur)} {missing_dur[:10]}")
    print(f"  hate videos with span end > duration (>0.5s): {len(overrun_rows)}")
    for i, me, d, over in sorted(overrun_rows, key=lambda x: -x[3])[:15]:
        print(f"    {i}: max_end={me:.1f}s dur={d:.1f}s overrun={over:.1f}s")
    all_missing = [i for i in all_ids if dur.get(i) is None]
    print(f"  ALL videos missing duration: {len(all_missing)} {all_missing[:10]}")

    # splits intersection
    print(f"\n-- Split intersection (hateful videos with >=1 span) --")
    for sp in ("train", "val", "test"):
        ids = list(splits[sp])
        hate_in = [i for i in ids if parsed.get(i, (None,))[0]
                   and parsed[i][3] == "Hate"]
        withspan = [i for i in hate_in if parsed[i][0]]
        lab1 = [i for i in ids if splits[sp][i] == 1]
        print(f"  {sp}: total={len(ids)} label1={len(lab1)} "
              f"hate_with_span={len(withspan)}")

    # ---- random sample 10 (seed 42) manual check ----
    print(f"\n-- Random sample 10 (seed 42), hate videos with spans --")
    pool = [i for i in hate_ids if parsed[i][0] and dur.get(i)]
    rng = random.Random(42)
    sample = rng.sample(pool, 10)
    for i in sample:
        spans = parsed[i][0]
        d = dur[i]
        span_total = sum(b - a for a, b in spans)
        merged = []
        for a, b in sorted([[max(0, a), min(d, b)] for a, b in spans]):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        union = sum(b - a for a, b in merged)
        ratio = union / d if d else 0
        over = max(b for _, b in spans) - d
        verdict = []
        if ratio >= 0.95:
            verdict.append("COVERS>=95% (near whole video)")
        if over > 0.5:
            verdict.append(f"OVERRUN {over:.1f}s")
        if not verdict:
            verdict.append("localized (partial)")
        print(f"  {i} [{id2split.get(i,'-')}]: dur={d:.1f}s spans={spans} "
              f"union={union:.1f}s ratio={ratio:.2f} -> {'; '.join(verdict)}")

    # ================= WRITE GOLD JSON =================
    if not args.no_write_json:
        gold = {}
        for r in rows:
            vid = vid_id(r["video_file_name"])
            spans, errs, raw, label = parsed[vid]
            d = dur.get(vid)
            entry = {"duration": d, "spans": []}
            if spans is None:
                entry["parse_error"] = errs[0] if errs else "parse_error"
                entry["raw"] = raw
            else:
                clipped = False
                out_spans = []
                for a, b in spans:
                    if d is not None and d > 0:
                        nb = min(b, d)
                        na = min(a, d)
                        if nb != b or na != a:
                            clipped = True
                        if nb > na:
                            out_spans.append([na, nb])
                        elif b > a:
                            # span entirely beyond duration -> record clip
                            clipped = True
                    else:
                        out_spans.append([a, b])
                entry["spans"] = out_spans
                if clipped:
                    entry["clipped"] = True
                if errs:
                    entry["anomaly"] = errs
            entry["label"] = 1 if label == "Hate" else 0
            gold[vid] = entry
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(gold, f, indent=1, ensure_ascii=False)
        print(f"\n[write] {OUT_JSON}  ({len(gold)} videos)")


if __name__ == "__main__":
    main()
