"""S_PRIZE_DECOMP -- decompose the 49-item S bucket into recoverable / contested / split.

Rules frozen in idea-stage/S_PRIZE_DECOMP.md section 0 (commit 8b4a792) before this
script was written. Read-only over existing annotation + prediction files; the only
computation is a re-run of the frozen r5_bucket_value.py oracle arithmetic with a
different `sel` subset.
"""
import json, os, collections
import numpy as np, torch
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATE = os.path.join(HERE, "claude_stance_gate")
RATERS = ["r7k", "m3q", "z9x"]
GOLD = {"S_FP": "DISTANCED", "S_FN": "ENDORSE"}


def load_rater(tag):
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(GATE, f"annot_{tag}", "*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            b = (r.get("binary") or "").strip().upper()
            out[r["item"]] = {"binary": b if b in ("ENDORSE", "DISTANCED") else None,
                              "voice": (r.get("voice") or "").strip().lower(),
                              "why": r.get("why", "")}
    return out


def binarise_5way(s):
    if s is None:
        return None
    return "ENDORSE" if s == "endorses" else "DISTANCED"


def read_5way(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[(r["dataset"], r["id"])] = (r.get("parsed") or {}).get("stance")
    return out


man = {x["item"]: x for x in json.load(open(os.path.join(GATE, "manifest.json")))["items"]}
raters = {t: load_rater(t) for t in RATERS}
qwen_r2 = read_5way(os.path.join(HERE, "mask_stance_pilot", "pred_m1.jsonl"))
qwen_r1 = read_5way(os.path.join(HERE, "stance_pilot", "pred_strong.jsonl"))

rows = []
for item, meta in sorted(man.items()):
    if meta["group"] not in GOLD:
        continue
    key = (meta["dataset"], meta["id"])
    gold = GOLD[meta["group"]]
    votes = {t: raters[t].get(item, {}).get("binary") for t in RATERS}
    votes["qwen_r2"] = binarise_5way(qwen_r2.get(key))
    valid = [v for v in votes.values() if v]
    k = sum(1 for v in valid if v == gold)
    n = len(valid)
    if n == 4:
        cls = "RECOVERABLE" if k >= 3 else "CONTESTED" if k <= 1 else "SPLIT"
    elif n == 3:
        cls = "RECOVERABLE" if k >= 2 else "CONTESTED"
    else:
        cls = "INSUFFICIENT"
    rows.append({"item": item, "dataset": meta["dataset"], "id": meta["id"],
                 "group": meta["group"], "gold": gold, "n_frames": meta["n_frames"],
                 "votes": votes, "n_valid": n, "k_for_gold": k, "cls": cls,
                 "qwen_r1_5way": qwen_r1.get(key), "qwen_r2_5way": qwen_r2.get(key),
                 "voices": {t: raters[t].get(item, {}).get("voice") for t in RATERS},
                 "why": {t: raters[t].get(item, {}).get("why", "") for t in RATERS}})

out = {"rows": rows}
out["counts"] = dict(collections.Counter(r["cls"] for r in rows))
out["by_dataset"] = {}
for ds in sorted({r["dataset"] for r in rows}):
    sub = [r for r in rows if r["dataset"] == ds]
    out["by_dataset"][ds] = {"n": len(sub),
                             **{c: sum(1 for r in sub if r["cls"] == c)
                                for c in ("RECOVERABLE", "CONTESTED", "SPLIT")}}
out["by_group"] = {}
for g in ("S_FP", "S_FN"):
    sub = [r for r in rows if r["group"] == g]
    out["by_group"][g] = {"n": len(sub),
                          **{c: sum(1 for r in sub if r["cls"] == c)
                             for c in ("RECOVERABLE", "CONTESTED", "SPLIT")}}
out["by_dataset_group"] = {}
for ds in sorted({r["dataset"] for r in rows}):
    for g in ("S_FP", "S_FN"):
        sub = [r for r in rows if r["dataset"] == ds and r["group"] == g]
        if sub:
            out["by_dataset_group"][f"{ds}/{g}"] = {
                "n": len(sub), **{c: sum(1 for r in sub if r["cls"] == c)
                                  for c in ("RECOVERABLE", "CONTESTED", "SPLIT")}}
# vote-margin histogram
out["k_hist"] = dict(collections.Counter(f"{r['k_for_gold']}/{r['n_valid']}" for r in rows))
# unanimity of the contested items
out["contested_unanimous_4_0"] = sum(1 for r in rows
                                     if r["cls"] == "CONTESTED" and r["k_for_gold"] == 0)

# ---------------------------------------------------------------- oracle recompute
BEST = {"HateMM": "mlp", "MHC": "mean_logit", "MHC_zh": "logistic",
        "ImpliHateVid": "logistic"}
B = json.load(open(os.path.join(HERE, "r5_buckets.json")))
pil = json.load(open(os.path.join(HERE, "r4_pilot1.json")))
rec_ids = {(r["dataset"], r["id"]) for r in rows if r["cls"] == "RECOVERABLE"}
recsplit_ids = rec_ids | {(r["dataset"], r["id"]) for r in rows if r["cls"] == "SPLIT"}
con_ids = {(r["dataset"], r["id"]) for r in rows if r["cls"] == "CONTESTED"}

oracle = {}
for ds, key in BEST.items():
    d = torch.load(os.path.join(ROOT, "data", "CLIP_Embedding", ds,
                                "test_seen_openai_clip-vit-large-patch14-336_HF.pt"),
                   map_location="cpu", weights_only=False)
    ids = list(d["ids"][0]) if isinstance(d["ids"][0], list) else list(d["ids"])
    y = torch.as_tensor(d["labels"]).view(-1).numpy().astype(int)
    preds = []
    for s in pil["datasets"][ds]["per_seed"]:
        sv = np.array(s["scores"][key])
        c = np.unique(sv)
        grid = np.concatenate([[c[0] - 1e-9], (c[:-1] + c[1:]) / 2, [c[-1] + 1e-9]])
        t = s["methods"][key]["test_macro_f1"]
        h = [g for g in grid
             if abs(f1_score(y, (sv >= g).astype(int), average="macro") - t) < 1e-9]
        preds.append((sv >= h[len(h) // 2]).astype(int))
    base = float(np.mean([f1_score(y, q, average="macro") for q in preds]))
    idx = {i: k for k, i in enumerate(ids)}
    S_all = [v for v in B[ds] if B[ds][v] == "S"]

    def gain(sel_vids):
        sel = [idx[v] for v in sel_vids]
        if not sel:
            return 0.0
        vals = []
        for q in preds:
            z = q.copy()
            z[sel] = y[sel]
            vals.append(f1_score(y, z, average="macro"))
        return round(float(np.mean(vals)) - base, 4)

    oracle[ds] = {
        "base": round(base, 4),
        "n_S": len(S_all),
        "n_rec": sum(1 for v in S_all if (ds, v) in rec_ids),
        "n_split": sum(1 for v in S_all if (ds, v) in recsplit_ids and (ds, v) not in rec_ids),
        "n_con": sum(1 for v in S_all if (ds, v) in con_ids),
        "oracle_S_all": gain(S_all),
        "oracle_recoverable": gain([v for v in S_all if (ds, v) in rec_ids]),
        "oracle_rec_plus_split": gain([v for v in S_all if (ds, v) in recsplit_ids]),
        "oracle_contested": gain([v for v in S_all if (ds, v) in con_ids]),
    }
out["oracle"] = oracle
m = lambda k: round(float(np.mean([oracle[d][k] for d in BEST])), 4)
out["oracle_mean"] = {k: m(k) for k in ("oracle_S_all", "oracle_recoverable",
                                        "oracle_rec_plus_split", "oracle_contested")}
out["ratio_recoverable_over_S"] = round(out["oracle_mean"]["oracle_recoverable"]
                                        / out["oracle_mean"]["oracle_S_all"], 4)
out["ratio_rec_plus_split_over_S"] = round(out["oracle_mean"]["oracle_rec_plus_split"]
                                           / out["oracle_mean"]["oracle_S_all"], 4)

# ------------------------------------------------- cross-check: A bucket, dataset conc.
A_ids = {(ds, v) for ds, dd in B.items() if not ds.startswith("_")
         for v, b in dd.items() if b == "A"}
out["A_bucket"] = {"n": len(A_ids), "ids": sorted(f"{a}/{b}" for a, b in A_ids),
                   "overlap_with_S": sorted(f"{a}/{b}" for a, b in (A_ids & (rec_ids | con_ids)))}
json.dump(out, open(os.path.join(HERE, "s_prize_decomp.json"), "w"),
          indent=1, ensure_ascii=False)

P = print
P("counts:", out["counts"])
P("k_hist:", out["k_hist"])
P("by_dataset:", json.dumps(out["by_dataset"]))
P("by_group:", json.dumps(out["by_group"]))
P("by_dataset_group:", json.dumps(out["by_dataset_group"]))
P("contested 4-0 unanimous:", out["contested_unanimous_4_0"])
P("oracle:", json.dumps(oracle, indent=1))
P("mean:", out["oracle_mean"])
P("R (recoverable / S):", out["ratio_recoverable_over_S"],
  " R (+split):", out["ratio_rec_plus_split_over_S"])
P("A bucket:", out["A_bucket"]["n"], out["A_bucket"]["ids"])
