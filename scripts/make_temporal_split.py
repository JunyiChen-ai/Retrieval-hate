#!/usr/bin/env python
"""Build temporal train/val/test splits for MultiHateClip EN/ZH (DESIGN_iter3 E0a).

Inputs
  - upload-date checkpoints from scripts/collect_upload_dates.py:
      data/gt/MHC_upload_dates.jsonl, data/gt/MHC_zh_upload_dates.jsonl
    (JSONL, last record per video_id wins, status ok => upload_date YYYYMMDD)
  - the EXISTING random splits data/gt/{MHC,MHC_zh}/{train,val,test}.jsonl,
    whose union defines the sample universe (annotation + local video present)
    and whose sizes define the target train/val/test ratios (DESIGN_iter3 S3:
    ratios not specified explicitly -> align with the existing random split).
  - the raw MultiHateClip annotations (for the 3-class Label survivor-bias
    crosstab over the FULL annotation, incl. samples without local video).

Split rule (DESIGN_iter3 S3.1)
  - datable samples are sorted ascending by (upload_date, id);
    test = the LATEST n_test, val = the n_val just before them,
    train = all earlier datable samples;
  - undatable samples (dead links etc.) ALL go to train / memory bank,
    NEVER to val/test (no contamination of the "future" sets).

Outputs (never touches the existing random splits)
  - data/gt/MHC_temporal/{train,val,test}.jsonl
  - data/gt/MHC_zh_temporal/{train,val,test}.jsonl
    records: {"id", "text", "label", "upload_date"}  (upload_date null in
    train for undatable samples; extra column is ignored by the loader)
  - data/video/{MHC,MHC_zh}_temporal -> symlink to the base video dir
  - data/gt/temporal_split_stats.json (machine-readable stats)

CPU only, seconds. Deterministic across reruns.
"""
import json
import os
from collections import Counter, OrderedDict

REPO_ROOT = "/data/jehc223/RGCL"
GT_ROOT = os.path.join(REPO_ROOT, "data", "gt")
VIDEO_ROOT = os.path.join(REPO_ROOT, "data", "video")

CONFIGS = OrderedDict(
    [
        (
            "MHC",
            {
                "dates": os.path.join(GT_ROOT, "MHC_upload_dates.jsonl"),
                "annotation": "/data/jehc223/Multihateclip/English/annotation(new).json",
            },
        ),
        (
            "MHC_zh",
            {
                "dates": os.path.join(GT_ROOT, "MHC_zh_upload_dates.jsonl"),
                "annotation": "/data/jehc223/Multihateclip/Chinese/annotation(new).json",
            },
        ),
    ]
)
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------
def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_dates(path):
    """video_id -> upload_date 'YYYYMMDD' (ok records only; last record wins).
    Also returns video_id -> last record (for failure analysis)."""
    last = {}
    for rec in read_jsonl(path):
        last[rec["video_id"]] = rec
    dates = {
        vid: r["upload_date"] for vid, r in last.items() if r.get("status") == "ok"
    }
    return dates, last


def label_pct(records):
    c = Counter(r["label"] for r in records)
    n = max(len(records), 1)
    return {str(k): {"n": v, "pct": round(100.0 * v / n, 1)} for k, v in sorted(c.items())}


def date_span(records):
    ds = [r["upload_date"] for r in records if r.get("upload_date")]
    return (min(ds), max(ds)) if ds else (None, None)


def year_hist(dates):
    return dict(sorted(Counter(d[:4] for d in dates).items()))


def fail_reason(err):
    """Bucket an error string into a coarse failure category."""
    e = (err or "").lower()
    if "sign in to confirm your age" in e:
        return "age_gated"
    if "video unavailable" in e or "code=-404" in e:
        return "unavailable/deleted"
    if "private" in e or "code=62012" in e:
        return "private (62012)"
    if "code=62002" in e:
        return "invisible (62002)"
    if "terminated" in e or "removed" in e or "violat" in e:
        return "removed/tos"
    if "timeout" in e or "request error" in e or "-412" in e:
        return "transient (retryable)"
    return "other"


# ----------------------------------------------------------------------------
def build_dataset(name, cfg, stats):
    print("=" * 74)
    print("DATASET %s" % name)
    print("=" * 74)

    dates, last = load_dates(cfg["dates"])

    # ----- universe = union of the existing random splits ------------------
    base = {}
    ratio_counts = {}
    for sp in SPLITS:
        recs = read_jsonl(os.path.join(GT_ROOT, name, sp + ".jsonl"))
        ratio_counts[sp] = len(recs)
        for r in recs:
            base[r["id"]] = r
    n_total = sum(ratio_counts.values())
    assert len(base) == n_total, "duplicate ids across existing splits?"
    universe = [dict(base[i], upload_date=dates.get(i)) for i in sorted(base)]

    datable = sorted(
        (r for r in universe if r["upload_date"]),
        key=lambda r: (r["upload_date"], r["id"]),
    )
    undatable = sorted(
        (r for r in universe if not r["upload_date"]), key=lambda r: r["id"]
    )

    # ----- target sizes: align with the existing random-split ratios -------
    # (universe == union of the existing splits, so matching the ratios over
    #  the same total means matching the existing sizes exactly)
    n_test = ratio_counts["test"]
    n_val = ratio_counts["val"]
    if n_val + n_test > len(datable):  # safety: shrink to the datable pool
        scale = len(datable) / float(n_val + n_test)
        n_test, n_val = int(n_test * scale), int(n_val * scale)
        print("WARNING: datable pool too small; scaled val/test down")

    test = datable[len(datable) - n_test:]
    val = datable[len(datable) - n_test - n_val: len(datable) - n_test]
    train = datable[: len(datable) - n_test - n_val] + undatable

    out = {"train": train, "val": val, "test": test}

    # ----- write (new files only; never the existing random splits) --------
    out_name = name + "_temporal"
    out_dir = os.path.join(GT_ROOT, out_name)
    os.makedirs(out_dir, exist_ok=True)
    for sp, recs in out.items():
        path = os.path.join(out_dir, sp + ".jsonl")
        with open(path, "w") as f:
            for r in recs:
                f.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "text": r["text"],
                            "label": r["label"],
                            "upload_date": r["upload_date"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print("wrote %4d -> %s" % (len(recs), path))

    # video dir: same ids as the base dataset -> one dataset-level symlink
    link = os.path.join(VIDEO_ROOT, out_name)
    target = os.path.join(VIDEO_ROOT, name)
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink(target, link)
        print("symlinked %s -> %s" % (link, target))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    ann = read_jsonl_annotation(cfg["annotation"])
    cross, dead_rate = crosstab(ann, dates, last)
    fails = Counter(
        fail_reason(r.get("error"))
        for r in last.values()
        if r.get("status") != "ok"
    )

    cov = {
        "universe": n_total,
        "datable": len(datable),
        "coverage_pct": round(100.0 * len(datable) / n_total, 1),
        "undatable_to_train": len(undatable),
    }
    ds = {
        "coverage_universe": cov,
        "annotation_crosstab_label_x_datable": cross,
        "dead_link_rate_by_label_pct": dead_rate,
        "failure_reasons": dict(fails.most_common()),
        "year_histogram_datable_universe": year_hist(
            [r["upload_date"] for r in datable]
        ),
        "existing_random_split_sizes": ratio_counts,
        "temporal_split": {},
    }
    for sp in SPLITS:
        recs = out[sp]
        lo, hi = date_span(recs)
        n_undated = sum(1 for r in recs if not r["upload_date"])
        ds["temporal_split"][sp] = {
            "n": len(recs),
            "labels": label_pct(recs),
            "date_min": lo,
            "date_max": hi,
            "n_undatable": n_undated,
        }
    stats[name] = ds

    # ----- console report ---------------------------------------------------
    print("\ncoverage: %(datable)d/%(universe)d datable (%(coverage_pct).1f%%), "
          "%(undatable_to_train)d undatable -> train" % cov)
    print("year histogram (datable universe):",
          ds["year_histogram_datable_universe"])
    print("\nannotation-level crosstab (FULL annotation, unique ids):")
    print("  %-10s %8s %8s %8s %10s" % ("Label", "queried", "datable", "dead", "dead%"))
    for lab, row in cross.items():
        print("  %-10s %8d %8d %8d %9.1f%%"
              % (lab, row["queried"], row["datable"], row["dead"],
                 dead_rate.get(lab, 0.0)))
    print("\nfailure reasons:", dict(fails.most_common()))
    for sp in SPLITS:
        d = ds["temporal_split"][sp]
        print("temporal %-5s n=%4d  labels=%s  dates=[%s..%s]  undatable=%d"
              % (sp, d["n"], d["labels"], d["date_min"], d["date_max"],
                 d["n_undatable"]))
    print()


def read_jsonl_annotation(path):
    """Full annotation -> unique id -> 3-class Label."""
    with open(path) as f:
        data = json.load(f)
    out = OrderedDict()
    for e in data:
        vid = e.get("Video_ID")
        if vid and vid not in out:
            out[vid] = e["Label"]
    return out


def crosstab(ann_labels, dates, last):
    """label x datable over ALL annotation ids that have been queried."""
    cross = OrderedDict()
    for lab in ("Hateful", "Offensive", "Normal"):
        ids = [v for v, l in ann_labels.items() if l == lab and v in last]
        ok = sum(1 for v in ids if v in dates)
        cross[lab] = {"queried": len(ids), "datable": ok, "dead": len(ids) - ok}
    dead_rate = {
        lab: round(100.0 * row["dead"] / row["queried"], 1) if row["queried"] else None
        for lab, row in cross.items()
    }
    return cross, dead_rate


def main():
    stats = OrderedDict()
    for name, cfg in CONFIGS.items():
        build_dataset(name, cfg, stats)
    stats_path = os.path.join(GT_ROOT, "temporal_split_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("stats written to", stats_path)


if __name__ == "__main__":
    main()
