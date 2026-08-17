"""Round-5 Phase A2b: dump per-error evidence for manual bucket attribution."""
import ast, csv, json, os, re
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(os.path.join(ROOT, "idea-stage", "r5_phase_a.json")))["A2_error_attribution"]

GT = {"HateMM": "HateMM", "MHC": "MHC", "MHC_zh": "MHC_zh", "ImpliHateVid": "ImpliHateVid"}
VOTE = {"MHC": "English", "MHC_zh": "Chinese"}


def texts(ds):
    o = {}
    for line in open(os.path.join(ROOT, "data", "gt", GT[ds], "test.jsonl"), encoding="utf-8"):
        r = json.loads(line); o[r["id"]] = r.get("text", "")
    return o


def ocr(ds):
    p = os.path.join(ROOT, "data", "OCR", ds, "ocr_video_test.jsonl")
    if not os.path.exists(p):
        return {}
    o = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        t = r.get("text", "")
        # OCR windows repeat the running text; keep unique lines in order
        seen, keep = set(), []
        for ln in t.split("\n"):
            ln = ln.strip()
            if ln and ln not in seen:
                seen.add(ln); keep.append(ln)
        o[r["video_id"]] = " | ".join(keep)
    return o


def votes(ds):
    if ds not in VOTE:
        return {}
    o = {}
    for sp in ("train", "valid", "test"):
        p = os.path.join(ROOT, "data", "gt", "mhc_votes", f"mhc_{VOTE[ds]}_{sp}.tsv")
        for r in csv.DictReader(open(p, newline="", encoding="utf-8"), delimiter="\t"):
            o[r["Video_ID"].strip()] = {"votes": ast.literal_eval(r["Label"]),
                                        "maj": r["Majority_Voting"].strip(),
                                        "target": r["Target_Victim"],
                                        "component": r["Component"]}
    return o


def clean(s, n=900):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:n] + ("…" if len(s) > n else "")


out = {}
for ds, b in A.items():
    T, O, V = texts(ds), ocr(ds), votes(ds)
    rows = []
    for i in b["err_ids"]:
        r = {"id": i, "kind": "FP" if i in b["fp_ids"] else "FN",
             "score": round(b["err_scores"][i], 3),
             "stable": i in b["stable_err_ids"],
             "transcript": clean(T.get(i, ""))}
        if i in O:
            r["ocr"] = clean(O[i], 700)
        if i in V:
            r["votes"] = V[i]["votes"]; r["maj"] = V[i]["maj"]
            r["target"] = V[i]["target"]; r["component"] = V[i]["component"]
        rows.append(r)
    out[ds] = rows

with open(os.path.join(ROOT, "idea-stage", "r5_error_dump.json"), "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
for ds, rows in out.items():
    print(ds, len(rows))
