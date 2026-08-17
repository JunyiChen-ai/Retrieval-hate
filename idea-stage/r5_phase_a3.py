"""Round-5 Phase A3: prize pool per dataset.

Combines the A1 ceiling and the A2 error set into "how many macro-F1 points are
actually purchasable", and splits the MHC error budget by annotator agreement.
"""
import ast, csv, json, os
import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PA = json.load(open(os.path.join(ROOT, "idea-stage", "r5_phase_a.json")))
A2 = PA["A2_error_attribution"]
VOTE = {"MHC": "English", "MHC_zh": "Chinese"}
POS = {"Hateful", "Offensive"}
RNG = np.random.default_rng(20260810)
BEST = {"HateMM": "mlp", "MHC": "mean_logit", "MHC_zh": "logistic",
        "ImpliHateVid": "logistic"}


def load_votes(lang):
    V = {}
    for sp in ("train", "valid", "test"):
        p = os.path.join(ROOT, "data", "gt", "mhc_votes", f"mhc_{lang}_{sp}.tsv")
        for r in csv.DictReader(open(p, newline="", encoding="utf-8"), delimiter="\t"):
            v = [x for x in ast.literal_eval(r["Label"])
                 if x in ("Hateful", "Offensive", "Normal", "Counter Narrative")]
            V[r["Video_ID"].strip()] = [1 if x in POS else 0 for x in v]
    return V


def ids_labels(ds):
    d = torch.load(os.path.join(ROOT, "data", "CLIP_Embedding", ds,
                                "test_seen_openai_clip-vit-large-patch14-336_HF.pt"),
                   map_location="cpu", weights_only=False)
    ids = list(d["ids"][0]) if isinstance(d["ids"][0], list) else list(d["ids"])
    return ids, torch.as_tensor(d["labels"]).view(-1).numpy().astype(int)


def recover_pred(ds):
    pil = json.load(open(os.path.join(ROOT, "idea-stage", "r4_pilot1.json")))
    key = BEST[ds]
    ids, y = ids_labels(ds)
    preds = []
    for s in pil["datasets"][ds]["per_seed"]:
        sv = np.array(s["scores"][key])
        cands = np.unique(sv)
        grid = np.concatenate([[cands[0] - 1e-9], (cands[:-1] + cands[1:]) / 2,
                               [cands[-1] + 1e-9]])
        tgt = s["methods"][key]["test_macro_f1"]
        hits = [t for t in grid
                if abs(f1_score(y, (sv >= t).astype(int), average="macro") - tgt) < 1e-9]
        preds.append((sv >= hits[len(hits) // 2]).astype(int))
    return ids, y, preds


out = {}
for ds in BEST:
    ids, y, preds = recover_pred(ds)
    base = float(np.mean([f1_score(y, q, average="macro") for q in preds]))
    idx = {i: k for k, i in enumerate(ids)}

    def repair_mean(sel):
        """mean over seeds of macro-F1 after oracle-fixing the selected indices"""
        vals = []
        for q in preds:
            z = q.copy(); z[list(sel)] = y[list(sel)]
            vals.append(f1_score(y, z, average="macro"))
        return float(np.mean(vals))

    row = {"base_macro_f1": base, "n_test": len(ids)}
    if ds in VOTE:
        V = load_votes(VOTE[ds])
        split = [k for k, i in enumerate(ids) if 0 < sum(V[i]) < len(V[i])]
        uni = [k for k in range(len(ids)) if k not in set(split)]
        row["n_split_vote"] = len(split)
        row["repair_split_vote_items"] = repair_mean(split)
        row["repair_unanimous_items"] = repair_mean(uni)
        # ceiling restricted to this test split: panel resample
        f1s = []
        for _ in range(4000):
            la, lb = [], []
            for i in ids:
                b = np.array(V[i]); m = len(b)
                sa = b[RNG.integers(0, m, m)].mean(); sb = b[RNG.integers(0, m, m)].mean()
                la.append(int(sa > 0.5) if sa != 0.5 else int(RNG.random() < 0.5))
                lb.append(int(sb > 0.5) if sb != 0.5 else int(RNG.random() < 0.5))
            f1s.append(f1_score(la, lb, average="macro"))
        row["testsplit_panel_resample_ceiling"] = float(np.mean(f1s))
        row["testsplit_panel_resample_p05"] = float(np.percentile(f1s, 5))
        row["prize_to_ceiling"] = row["testsplit_panel_resample_ceiling"] - base
    row["repair_all"] = 1.0
    out[ds] = row

# HateMM defect accounting
ids, y, preds = recover_pred("HateMM")
idx = {i: k for k, i in enumerate(ids)}
DEFECT = ["non_hate_video_140", "hate_video_273", "hate_video_295", "hate_video_89",
          "hate_video_102", "hate_video_88", "hate_video_356", "hate_video_329",
          "hate_video_277", "non_hate_video_541", "non_hate_video_348"]
sel = [idx[i] for i in DEFECT if i in idx]
errs = set(A2["HateMM"]["err_ids"])
out["HateMM"]["n_audit_flagged_test_items"] = len(sel)
out["HateMM"]["audit_flagged_in_error_set"] = sorted(errs & set(DEFECT))
IMPL_DEF = ["NH_180", "IM_164", "IM_243", "EX_258", "IM_285", "NH_83"]
out["ImpliHateVid"]["audit_flagged_in_error_set"] = sorted(
    set(A2["ImpliHateVid"]["err_ids"]) & set(IMPL_DEF))
MZ_DEF = ["BV1Pe411Y7c5", "BV15E411a7Jd", "BV1va4y1m72C", "BV12C4y1m7ic"]
out["MHC_zh"]["audit_flagged_in_error_set"] = sorted(
    set(A2["MHC_zh"]["err_ids"]) & set(MZ_DEF))

with open(os.path.join(ROOT, "idea-stage", "r5_phase_a3.json"), "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
