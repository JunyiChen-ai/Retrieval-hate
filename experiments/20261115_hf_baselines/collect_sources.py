"""Collect the baseline predictions produced in ~/Hate-follow-up on this repository's cohorts (README.md, section
"HateClipSeg reruns on this repository's split") into runs/20261115_hf_baselines/sources/.

Input: ~/Hate-follow-up/runs/20261115_rh_hcs_split/<run>/.../predictions.jsonl (that project's schema: one row per
video and method, `score_curve` on its 4 fps grid of length ceil(4 * duration)).
Output: runs/20261115_hf_baselines/sources/<method_key>/<Dataset>/<seed<k> | single>/predictions.jsonl with exactly
one row per video of this repository's test cohort (hier_evidence_common.load_fixed_cohort), the source row unchanged.
A source is written only if every cohort video has a row of the wanted method with a finite curve of length
ceil(4 * duration); otherwise it is reported as missing or bad and nothing is written for it. Nothing is evaluated.

    python experiments/20261115_hf_baselines/collect_sources.py        # on the machine holding both repositories
"""
import json
import math
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hier_evidence_common as hc                                       # noqa: E402

SRC = Path.home() / "Hate-follow-up" / "runs" / "20261115_rh_hcs_split"
OUT = ROOT / "runs" / "20261115_hf_baselines" / "sources"
CORPUS = {"HateMM": "hatemm", "HateClipSeg": "hateclipseg", "DeHate": "dehate"}
SEEDS = ["seed2025", "seed234", "seed3407"]
HCS = ["HateClipSeg"]
# method_key: (source directory pattern, method name in the rows, datasets, seeded)
SOURCES = {
    "BERTpMIL": ("mil_bert/{D}/{S}", "mil_bert", HCS, True),
    "CLIPpMIL": ("mil_clip/{D}/{S}", "mil_clip", HCS, True),
    "wav2vec2pMIL": ("mil_wav2vec2/{D}/{S}", "mil_wav2vec2", HCS, True),
    # the AVadCLIP file holds two branches per video: score_align (that project's headline) and score_mlp
    "AVadCLIP_align": ("avadclip/{D}/{S}", "avadclip", HCS, True),
    "AVadCLIP_mlp": ("avadclip/{D}/{S}", "avadclip_mlp", HCS, True),
    "SAGE": ("sage/{D}/{S}", "SAGE_win8", HCS, True),
    "CLARA": ("clara/{D}/{S}", "CLARA_win8", HCS, True),
    "Qwen3-VL-8B": ("lf/naive/vlm/{D}", "naive_vlm", HCS, False),
    "MARS_Qwen3-VL-8B": ("lf/mars/{D}/prob", "mars_q3vl_prob", HCS, False),
    "EventVAD_Qwen3-VL-8B": ("lf/eventvad/{D}/prob", "eventvad_q3vl_prob", HCS, False),
    "LAVAD": ("lavad_av/{D}", "lavad_av_base", HCS, False),
    "PRISM": ("prism_av/{D}", "prism_av", HCS, False),
    "VideoMind": ("videomind_av/{D}", "videomind_av", HCS, False),
    "InternVL3.5-8B": ("lf/naive/vlm_internvl/{D}", "naive_vlm_internvl", ["HateMM", "HateClipSeg", "DeHate"], False),
}


def main():
    cohort = {}
    report = {"host": socket.gethostname(), "written": [], "missing": [], "bad": []}
    for key, (pat, method, datasets, seeded) in SOURCES.items():
        for ds in datasets:
            if ds not in cohort:
                cohort[ds] = list(hc.load_fixed_cohort(CORPUS[ds])[1]["test"])
            for seed in (SEEDS if seeded else ["single"]):
                src = SRC / pat.format(D=ds, S=seed) / "predictions.jsonl"
                name = f"{key}/{ds}/{seed}"
                if not src.is_file():
                    report["missing"].append(name)
                    continue
                rows, methods = {}, set()
                for line in open(src):
                    r = json.loads(line)
                    methods.add(r["method"])
                    if r["method"] == method and r["dataset"] == ds:
                        if r["video_id"] in rows:
                            raise SystemExit(f"{src}: duplicate row {r['video_id']}")
                        rows[r["video_id"]] = line if line.endswith("\n") else line + "\n"
                problems = []
                for v in cohort[ds]:
                    if v not in rows:
                        problems.append((v, "no row"))
                        continue
                    r = json.loads(rows[v])
                    c = np.asarray(r["score_curve"], dtype=np.float64)
                    n = max(1, math.ceil(4 * float(r["duration"]) - 1e-9))
                    if r.get("error") or len(c) not in (n, math.ceil(4 * float(r["duration"]))) or not np.all(np.isfinite(c)):
                        problems.append((v, f"error {r.get('error')}, {len(c)} values for {n} frames"))
                if problems:
                    report["bad"].append({"source": name, "methods_in_file": sorted(methods), "n": len(problems),
                                          "first": problems[:3]})
                    continue
                od = OUT / key / ds / seed
                od.mkdir(parents=True, exist_ok=True)
                with open(od / "predictions.jsonl", "w") as fh:
                    for v in cohort[ds]:
                        fh.write(rows[v])
                report["written"].append({"source": name, "from": str(src), "method": method,
                                          "videos": len(cohort[ds]), "source_rows_not_in_cohort": len(rows) - len(cohort[ds])})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "collect_report.json").write_text(json.dumps(report, indent=1) + "\n")
    for w in report["written"]:
        print("written", w["source"], w["videos"], "videos;", w["source_rows_not_in_cohort"], "other rows dropped")
    for m in report["missing"]:
        print("MISSING", m)
    for b in report["bad"]:
        print("BAD", b)


if __name__ == "__main__":
    main()
