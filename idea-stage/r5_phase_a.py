"""Round-5 Phase A: prize-pool measurement.

A1. Annotation noise ceiling from MHC per-annotator votes (EN + ZH).
A2. Error attribution of the round-4 best ensemble on the four test sets.

Zero GPU. Reads frozen artefacts only (idea-stage/r4_pilot1.json test scores,
data/gt/mhc_votes/*.tsv, data/gt/*/test.jsonl, CLIP caches for id order).
Test labels ARE read -- authorised by the user's 2026-08-09 test-protocol ruling;
this is a disclosed diagnostic, not a tuned selection.
"""
import ast
import csv
import json
import os
import re
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = {}
RNG = np.random.default_rng(20260810)

POS_MAJ = {"Hateful", "Offensive"}          # binary positive classes for MHC
VOTE_POS = {"Hateful", "Offensive"}


def macro_f1(y, p, thr=0.5):
    return f1_score(y, (np.asarray(p) >= thr).astype(int), average="macro")


# --------------------------------------------------------------------------
# A1  MHC annotation noise ceiling
# --------------------------------------------------------------------------
def load_votes(lang, split):
    p = os.path.join(ROOT, "data", "gt", "mhc_votes", f"mhc_{lang}_{split}.tsv")
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8"), delimiter="\t"))
    out = {}
    for r in rows:
        v = ast.literal_eval(r["Label"])
        v = [x for x in v if x in ("Hateful", "Offensive", "Normal", "Counter Narrative")]
        out[r["Video_ID"].strip()] = {
            "votes": v,
            "bin": [1 if x in VOTE_POS else 0 for x in v],
            "maj": r["Majority_Voting"].strip(),
        }
    return out


def project_labels(ds, split="test_seen"):
    tag = "openai_clip-vit-large-patch14-336_HF"
    d = torch.load(os.path.join(ROOT, "data", "CLIP_Embedding", ds, f"{split}_{tag}.pt"),
                   map_location="cpu", weights_only=False)
    ids = list(d["ids"][0]) if isinstance(d["ids"][0], list) else list(d["ids"])
    y = torch.as_tensor(d["labels"]).view(-1).numpy().astype(int)
    return ids, y


def a1():
    res = {}
    for lang, ds in (("English", "MHC"), ("Chinese", "MHC_zh")):
        allv = {}
        for sp in ("train", "valid", "test"):
            allv.update(load_votes(lang, sp))
        # --- consistency check against the project's binary labels (test split)
        ids, y = project_labels(ds)
        matched = [(i, yy) for i, yy in zip(ids, y) if i in allv]
        agree = sum(1 for i, yy in matched
                    if int(allv[i]["maj"] in POS_MAJ) == yy)
        cov = len(matched) / len(ids)

        # --- pooled over every split (labels used only as annotation statistics)
        items = list(allv.values())
        n_ann = np.array([len(it["bin"]) for it in items])
        # pairwise raw agreement + Cohen-style chance correction on the binary collapse
        pa_num = pa_den = 0
        b_all = []
        for it in items:
            b = it["bin"]
            for a in range(len(b)):
                for c in range(a + 1, len(b)):
                    pa_num += int(b[a] == b[c]); pa_den += 1
            b_all += b
        p_obs = pa_num / pa_den
        p1 = float(np.mean(b_all))
        p_chance = p1 ** 2 + (1 - p1) ** 2
        kappa = (p_obs - p_chance) / (1 - p_chance)
        # Krippendorff alpha (nominal, binary) via the standard coincidence formula
        n_units = sum(len(it["bin"]) * (len(it["bin"]) - 1) for it in items
                      if len(it["bin"]) > 1)
        Do = 0.0
        for it in items:
            b = it["bin"]; m = len(b)
            if m < 2:
                continue
            k1 = sum(b); k0 = m - k1
            Do += 2 * k1 * k0 / (m - 1)
        Do /= n_units
        N = len(b_all)
        n1 = sum(b_all); n0 = N - n1
        De = 2 * n1 * n0 / (N - 1) / N
        alpha = 1 - Do / De

        # --- split-vote census
        split_votes = sum(1 for it in items if 0 < sum(it["bin"]) < len(it["bin"]))
        tie = sum(1 for it in items
                  if len(it["bin"]) % 2 == 0 and sum(it["bin"]) * 2 == len(it["bin"]))

        # --- ceiling 1: single annotator vs majority-of-the-rest (leave-one-out)
        yl, yp = [], []
        for it in items:
            b = it["bin"]
            if len(b) < 2:
                continue
            for a in range(len(b)):
                rest = b[:a] + b[a + 1:]
                s = sum(rest) / len(rest)
                if s == 0.5:
                    continue                       # rest is tied: undefined reference
                yl.append(int(s > 0.5)); yp.append(b[a])
        loo_f1 = f1_score(yl, yp, average="macro")
        loo_acc = float(np.mean(np.array(yl) == np.array(yp)))

        # --- ceiling 2: panel-resample reproducibility.
        # Draw two disjoint-as-possible panels by bootstrapping the observed votes;
        # a deterministic model can at best predict one panel's label, and is scored
        # against the other.  Reported as macro-F1 of panel A labels vs panel B labels.
        reps = 2000
        f1s, accs = [], []
        for _ in range(reps):
            la, lb = [], []
            for it in items:
                b = np.array(it["bin"])
                m = len(b)
                ia = RNG.integers(0, m, size=m)
                ib = RNG.integers(0, m, size=m)
                sa, sb = b[ia].mean(), b[ib].mean()
                la.append(int(sa > 0.5) if sa != 0.5 else int(RNG.random() < 0.5))
                lb.append(int(sb > 0.5) if sb != 0.5 else int(RNG.random() < 0.5))
            f1s.append(f1_score(la, lb, average="macro"))
            accs.append(float(np.mean(np.array(la) == np.array(lb))))
        res[ds] = {
            "lang": lang, "n_items_with_votes": len(items),
            "mean_annotators": float(n_ann.mean()),
            "ann_hist": {int(k): int(v) for k, v in zip(*np.unique(n_ann, return_counts=True))},
            "test_coverage_of_project_ids": cov,
            "test_majority_matches_project_label": agree / max(1, len(matched)),
            "pairwise_raw_agreement": p_obs,
            "cohen_kappa_binary": kappa,
            "krippendorff_alpha_binary": alpha,
            "positive_vote_rate": p1,
            "split_vote_items": split_votes, "split_vote_rate": split_votes / len(items),
            "even_panel_tie_items": tie,
            "ceiling_single_annotator_vs_rest_macroF1": loo_f1,
            "ceiling_single_annotator_vs_rest_acc": loo_acc,
            "ceiling_panel_resample_macroF1_mean": float(np.mean(f1s)),
            "ceiling_panel_resample_macroF1_p05": float(np.percentile(f1s, 5)),
            "ceiling_panel_resample_macroF1_p95": float(np.percentile(f1s, 95)),
            "ceiling_panel_resample_acc": float(np.mean(accs)),
        }
    return res


# --------------------------------------------------------------------------
# A2  error attribution of the round-4 best ensemble
# --------------------------------------------------------------------------
BEST = {  # §8.10 / §8.7 best-ensemble comparator per dataset (macro-F1 column)
    "HateMM": "mlp", "MHC": "mean_logit", "MHC_zh": "logistic", "ImpliHateVid": "logistic",
}
DS_TAG = {"HateMM": "HateMM", "MHC": "MHC-EN", "MHC_zh": "MHC-ZH",
          "ImpliHateVid": "ImpliHateVid"}


def load_texts(ds):
    name = {"HateMM": "HateMM", "MHC": "MHC", "MHC_zh": "MHC_zh",
            "ImpliHateVid": "ImpliHateVid"}[ds]
    out = {}
    with open(os.path.join(ROOT, "data", "gt", name, "test.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r.get("text", "")
    return out


def ocr_windows_hatemm():
    """HateMM test OCR text, if the cache carries strings."""
    cands = ["artifacts/ocr_cache", "data/ocr_cache", "artifacts/ocr"]
    for c in cands:
        p = os.path.join(ROOT, c)
        if os.path.isdir(p):
            return p
    return None


def a2():
    pil = json.load(open(os.path.join(ROOT, "idea-stage", "r4_pilot1.json")))
    res = {}
    for ds, key in BEST.items():
        ids, y = project_labels(ds)
        blk = pil["datasets"][ds]
        seeds = blk["per_seed"]
        assert len(seeds[0]["scores"]["y"]) == len(ids), (ds, len(ids))
        assert np.allclose(np.array(seeds[0]["scores"]["y"]), y), ds
        # R4-1 picked each method's threshold on VALIDATION and only stored the resulting
        # test macro-F1, not the threshold.  Recover the threshold by inverting the stored
        # test_macro_f1 over the test-score grid; assert the recovery is exact.
        def recover(sv, target):
            cands = np.unique(sv)
            grid = np.concatenate([[cands[0] - 1e-9],
                                   (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-9]])
            hits = [t for t in grid
                    if abs(f1_score(y, (sv >= t).astype(int), average="macro") - target) < 1e-9]
            assert hits, "threshold recovery failed"
            return hits[len(hits) // 2]

        preds = []
        for s in seeds:
            sv = np.array(s["scores"][key])
            preds.append((sv >= recover(sv, s["methods"][key]["test_macro_f1"])).astype(int))
        P = np.mean([np.array(s["scores"][key]) for s in seeds], axis=0)
        pred = (np.mean(preds, axis=0) >= 0.5).astype(int)
        f1_per_seed = [f1_score(y, q, average="macro") for q in preds]
        f1 = float(np.mean(f1_per_seed))          # matches the §8.7 reporting convention
        roc = float(np.mean([roc_auc_score(y, np.array(s["scores"][key])) for s in seeds]))
        per_seed_err = [set(np.where(q != y)[0].tolist()) for q in preds]
        stable = set.intersection(*per_seed_err)
        union = set.union(*per_seed_err)
        err = np.where(pred != y)[0]
        fp = [i for i in err if y[i] == 0]
        fn = [i for i in err if y[i] == 1]
        texts = load_texts(ds)
        empty = [i for i in range(len(ids))
                 if len(re.sub(r"[\s　]", "", texts.get(ids[i], ""))) < 3]
        res[ds] = {
            "n_test": len(ids), "macro_f1_mean_of_seeds": f1,
            "macro_f1_per_seed": f1_per_seed,
            "macro_f1_of_majority_pred": f1_score(y, pred, average="macro"), "roc": roc,
            "n_err": int(len(err)), "n_fp": len(fp), "n_fn": len(fn),
            "err_ids": [ids[i] for i in err],
            "fp_ids": [ids[i] for i in fp], "fn_ids": [ids[i] for i in fn],
            "stable_err_ids": [ids[i] for i in sorted(stable)],
            "union_err_n": len(union),
            "empty_transcript_test_n": len(empty),
            "empty_transcript_err_n": len([i for i in err if i in set(empty)]),
            "empty_transcript_ids_err": [ids[i] for i in err if i in set(empty)],
            "err_scores": {ids[i]: float(P[i]) for i in err},
            "comparator": key,
        }
        # macro-F1 value of fixing each bucket (oracle repair)
        def repair(idxs):
            q = pred.copy()
            for i in idxs:
                q[i] = y[i]
            return f1_score(y, q, average="macro")
        res[ds]["repair_all_err"] = repair(err)
        res[ds]["repair_empty_transcript"] = repair([i for i in err if i in set(empty)])
        res[ds]["repair_fp"] = repair(fp)
        res[ds]["repair_fn"] = repair(fn)
        # ImpliHateVid subtype split from the id prefix
        if ds == "ImpliHateVid":
            sub = {}
            for i in err:
                sub.setdefault(ids[i].split("_")[0], []).append(ids[i])
            res[ds]["err_by_subtype"] = {k: len(v) for k, v in sub.items()}
            res[ds]["nh_fp_ids"] = sub.get("NH", [])
            res[ds]["repair_nh_fp"] = repair([i for i in err if ids[i].startswith("NH")])
    return res


if __name__ == "__main__":
    OUT["A1_noise_ceiling"] = a1()
    OUT["A2_error_attribution"] = a2()
    with open(os.path.join(ROOT, "idea-stage", "r5_phase_a.json"), "w") as f:
        json.dump(OUT, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: (v if k == "A1_noise_ceiling" else
                          {d: {kk: vv for kk, vv in b.items()
                               if not kk.endswith("_ids") and kk != "err_scores"}
                           for d, b in v.items()})
                      for k, v in OUT.items()}, indent=1, ensure_ascii=False))
