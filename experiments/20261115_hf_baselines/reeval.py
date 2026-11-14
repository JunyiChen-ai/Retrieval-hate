"""Re-evaluate the baselines of the Hate-follow-up project on this repository's test cohort and evaluator
(README.md in this directory).

Input: each baseline's per-video predictions.jsonl in ~/Hate-follow-up/runs/ (schema_version 1, `score_curve` on the
4 fps grid of that project). Nothing is trained or re-run here.

Conversion to this repository's 1 fps grid: the value of second t is the mean of 4 fps frames 4t .. 4t+3 of
`score_curve` (for 1 fps sources, which that project repeats 4 times, this returns the original value); the result is
cut or padded with its last value to the length of this repository's per-second GT array. Every cohort video must
have a prediction; any missing video or non-finite value stops the run. Scores are written with
hier_evidence_common.write_scores (6 decimals, as for this repository's own runs) and evaluated by
scripts/reproduction_baselines/eval_baseline_scores.py --require-full-coverage.

Output: runs/20261115_hf_baselines/<row>/<corpus>/<seed>/{scores.jsonl, metrics.json, convert.json} and
runs/20261115_hf_baselines/table.json (seed mean, sd n-1).
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hier_evidence_common as hc                                       # noqa: E402

HF = Path.home() / "Hate-follow-up" / "runs"
DS = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg", "dehate": "DeHate"}
SEEDS = ["seed2025", "seed234", "seed3407"]
B = "20261008_baselines"
L = "20261113_lf_baselines"
# row: (group, source directory pattern under HF with {D} = corpus name and {S} = seed, seeded)
ROWS = {
    "BERT+MIL": ("weak", B + "/mil_bert/{D}/{S}", True),
    "wav2vec2+MIL": ("weak", B + "/mil_wav2vec2/{D}/{S}", True),
    "CLIP+MIL": ("weak", B + "/mil_clip/{D}/{S}", True),
    "VadCLIP (fixed preset)": ("weak", B + "/vadclip/{D}/{S}", True),
    "AVadCLIP": ("weak", B + "/avadclip/{D}/{S}", True),
    "SAGE": ("weak", B + "/sage/{D}/{S}", True),
    "CLARA": ("weak", B + "/clara/{D}/{S}", True),
    "ZS-CLIP": ("free", B + "/zs_clip_av/{D}", False),
    "ZS-ImageBind": ("free", B + "/zs_imagebind_av/{D}", False),
    "LAVAD": ("free", B + "/lavad_av/{D}", False),
    "EventVAD": ("free", B + "/eventvad_av/{D}", False),
    "PRISM": ("free", B + "/prism_av/{D}", False),
    "VideoMind": ("free", B + "/videomind_av/{D}", False),
    "Qwen3-8B (transcript)": ("free", L + "/naive/llm/{D}", False),
    "Qwen3-VL-8B": ("free", L + "/naive/vlm/{D}", False),
    "MARS (Qwen3-VL-8B)": ("free", L + "/mars/{D}/prob", False),
    "EventVAD (Qwen3-VL-8B)": ("free", L + "/eventvad/{D}/prob", False),
}


def to_1fps(curve, n):
    c = np.asarray(curve, dtype=np.float64)
    m = len(c) // 4
    s = c[:4 * m].reshape(m, 4).mean(1) if m else c[:0]
    if len(c) > 4 * m:
        s = np.append(s, c[4 * m:].mean())
    pad = max(0, n - len(s))
    if pad:
        s = np.append(s, np.full(pad, s[-1]))
    return s[:n], pad, max(0, len(s) - n - pad)


def one(row, src, corpus, seed, ids, gt, out_root):
    preds = {}
    for line in open(src / "predictions.jsonl"):
        r = json.loads(line)
        preds[r["video_id"]] = r["score_curve"]
    missing = [v for v in ids if v not in preds]
    if missing:
        raise SystemExit(f"ABORT {row} {corpus} {seed}: {len(missing)} cohort videos missing, e.g. {missing[:3]}")
    scores, pads, cuts = {}, {}, {}
    for v in ids:
        s, p, c = to_1fps(preds[v], len(gt[v]))
        if not np.all(np.isfinite(s)):
            raise SystemExit(f"ABORT {row} {corpus} {seed}: non-finite score in {v}")
        scores[v] = s
        if p:
            pads[v] = p
        if c:
            cuts[v] = c
    od = out_root / row.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p") / corpus / seed
    od.mkdir(parents=True, exist_ok=True)
    hc.write_scores(od / "scores.jsonl", scores)
    (od / "convert.json").write_text(json.dumps({
        "host": socket.gethostname(), "source": str(src / "predictions.jsonl"), "n_videos": len(ids),
        "n_extra_source_videos_dropped": len(set(preds) - set(ids)),
        "padded_seconds": {"videos": len(pads), "total": int(sum(pads.values())), "max": int(max(pads.values(), default=0))},
        "cut_seconds": {"videos": len(cuts), "total": int(sum(cuts.values())), "max": int(max(cuts.values(), default=0))},
    }, indent=1) + "\n")
    subprocess.run([sys.executable, hc.EVALUATOR, "--corpus", corpus, "--split", "test",
                    "--scores", str(od / "scores.jsonl"), "--branch", "score_av", "--require-full-coverage",
                    "--json-out", str(od / "metrics.json")], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    r = json.load(open(od / "metrics.json"))["results"]["score_av"]
    return {"ap": r["pr_auc"], "roc": r["roc_auc"], "within": r["per_video"]["macro_auc"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "dehate"])
    ap.add_argument("--rows", nargs="+", default=list(ROWS))
    ap.add_argument("--out", default="runs/20261115_hf_baselines")
    a = ap.parse_args()
    out_root = ROOT / a.out
    table = json.load(open(out_root / "table.json")) if (out_root / "table.json").exists() else {}
    for corpus in a.corpora:
        _, ids, gt, _ = hc.load_fixed_cohort(corpus)
        tid, tgt = ids["test"], gt["test"]
        for row in a.rows:
            group, pat, seeded = ROWS[row]
            res = {}
            for seed in (SEEDS if seeded else ["single"]):
                src = HF / pat.format(D=DS[corpus], S=seed)
                if not (src / "predictions.jsonl").exists():
                    print(f"skip {row} {corpus} {seed}: no {src}/predictions.jsonl", flush=True)
                    continue
                res[seed] = one(row, src, corpus, seed, tid, tgt, out_root)
            if not res:
                continue
            arr = {k: np.array([r[k] for r in res.values()]) for k in ("ap", "roc", "within")}
            table.setdefault(row, {"group": group})[corpus] = {
                "seeds": res, **{k: float(v.mean()) for k, v in arr.items()},
                **{k + "_sd": (float(v.std(ddof=1)) if len(v) > 1 else None) for k, v in arr.items()}}
            t = table[row][corpus]
            print(f"{row:26s} {corpus:8s} n={len(res)} AP {t['ap']:.3f} ROC {t['roc']:.3f} within {t['within']:.3f}",
                  flush=True)
            (out_root / "table.json").write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
