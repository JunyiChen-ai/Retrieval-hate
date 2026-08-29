"""Round-5 Phase A2c: what is each error bucket worth in macro-F1 (oracle repair)."""
import json, os, collections
import numpy as np, torch
from sklearn.metrics import f1_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = json.load(open(os.path.join(ROOT, "idea-stage", "r5_buckets.json")))
BEST = {"HateMM": "mlp", "MHC": "mean_logit", "MHC_zh": "logistic", "ImpliHateVid": "logistic"}
pil = json.load(open(os.path.join(ROOT, "idea-stage", "r4_pilot1.json")))
tot = collections.Counter(); out = {}
for ds, key in BEST.items():
    d = torch.load(os.path.join(ROOT, "data", "CLIP_Embedding", ds,
                   "test_seen_openai_clip-vit-large-patch14-336_HF.pt"),
                   map_location="cpu", weights_only=False)
    ids = list(d["ids"][0]) if isinstance(d["ids"][0], list) else list(d["ids"])
    y = torch.as_tensor(d["labels"]).view(-1).numpy().astype(int)
    preds = []
    for s in pil["datasets"][ds]["per_seed"]:
        sv = np.array(s["scores"][key]); c = np.unique(sv)
        grid = np.concatenate([[c[0]-1e-9], (c[:-1]+c[1:])/2, [c[-1]+1e-9]])
        t = s["methods"][key]["test_macro_f1"]
        h = [g for g in grid if abs(f1_score(y,(sv>=g).astype(int),average="macro")-t) < 1e-9]
        preds.append((sv >= h[len(h)//2]).astype(int))
    base = float(np.mean([f1_score(y,q,average="macro") for q in preds]))
    idx = {i:k for k,i in enumerate(ids)}
    cnt = collections.Counter(B[ds].values()); tot.update(cnt)
    row = {"n_err": len(B[ds]), "base": round(base,4), "counts": dict(cnt), "value": {}}
    for b in sorted(cnt):
        sel = [idx[i] for i,v in B[ds].items() if v == b]
        vals = []
        for q in preds:
            z = q.copy(); z[sel] = y[sel]
            vals.append(f1_score(y, z, average="macro"))
        row["value"][b] = round(float(np.mean(vals)) - base, 4)
    # S+O+M combined (the three design targets)
    sel = [idx[i] for i,v in B[ds].items() if v in ("S","O","M")]
    vals = []
    for q in preds:
        z = q.copy(); z[sel] = y[sel]; vals.append(f1_score(y, z, average="macro"))
    row["value_S_O_M_combined"] = round(float(np.mean(vals)) - base, 4)
    out[ds] = row
out["_TOTAL_counts"] = dict(tot)
n = sum(tot.values())
out["_TOTAL_share"] = {k: round(v/n, 3) for k, v in tot.items()}
json.dump(out, open(os.path.join(ROOT,"idea-stage","r5_bucket_value.json"),"w"), indent=1)
print(json.dumps(out, indent=1))
