"""Review control E1 (README, round 1): the frozen strongest baseline plus a fusion head LEARNED on all answers.

GPT 6 Astra's round-1 point 1: the rule-14(f) control (experiments/20261101_r5_completion/baseline_plus_vlm.py) keeps
only the shortest asked node per second and mixes logits with one alpha, so it shows only that our method beats one
lossy fusion. This control gives the frozen baseline every answer our method used, through a head trained on the
training videos with video labels (the same weak supervision as our method), selected on validation.

Per second t of a video with asked set A (intervals [a, b) with soft answer p = first-token P(Yes), soft_both):
  baseline logit z_t (the frozen official checkpoint, baseline_infer.py; MACIL-SD HateMM from its official
    train_infer / val_infer / test scores);
  covering-node tokens: every answered n in A with a <= t < b gives u = [logit p / 5, p, log(b - a) / 5,
    (t - a) / (b - a)]; e = MLP_tok(u) (4 -> 32 -> 32); per second the sum and the max of e over covering nodes and
    the number of covering nodes / 8;
  video context: mean and max of logit p / 5 over the answered nodes of A, their number / 8 (so seconds no asked
    node covers still see the answers);
  variant "feat" only: the per-second input features of our own backbone (I3D crop 0, VGGish, BERT rows of
    qtl.data.Store) through a linear map to 32 dimensions.
f_t = alpha z_t + h_t, h_t = MLP([sum e, max e, n / 8, context (, features)]) (-> 64 -> 1). The baseline logit is
NOT an input of h: on its own training videos the frozen baseline is far more confident than on held-out videos
(MACIL-SD HateMM video AP .98 train vs .71-.77 test), so a head trained on it would learn to over-trust it (rule-6
review, REVIEW_RULE6_R1.md F1). h is trained on the training videos alone: binary cross-entropy of the video label
against the mean of the top-k per-second logits of h (k = T // 16 + 1, the baselines' MIL readout), AdamW, weight
decay 1e-4, 30 epochs, batches of 16 videos; alpha in ALPHAS is chosen on validation with the learning rate and the
epoch. Training answer sets: per video and epoch, with probability .5
the first 8 nodes our policy asked on that training video (policy_replay.py, the same checkpoint and seed) and
otherwise 8 random answered nodes of the full cached tree. Validation and test: the first 8 nodes our policy asked
(exactly the answers our method used at 8 calls). Selection: learning rate in {1e-3, 3e-3}, epoch and alpha by validation
(pooled AP + pooled ROC) / 2 (validation labels, no gradient). Test scored once per variant by the unified evaluator.
Rule-10 developmental evidence; the baseline is not retrained.

Variant "hid" (review round 2, E1b; GPT 6 Astra: the fusion must learn from the baseline and the answers jointly):
the head's input adds the frozen baseline's per-second hidden features (the input of its final per-second
classification layer, baseline_infer.py --hidden; standardized per dimension with the training-split mean and standard
deviation, then a linear map to 32 dimensions) and the baseline logit z_t / 5; the output IS the fused score
(f_t = h_t, alpha fixed at 0) and the MIL loss is on it. Same answers, node sampling, learning rates, epochs and
validation selection as the other variants. Written to its own --out-root (runs/20261108_astra_review/e1b).
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import pickle
import socket
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata, qtree                                    # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

SEEDS = (234, 2025, 3407)
LRS = (1e-3, 3e-3)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
EPOCHS, BATCH, K_ASK = 30, 16, 8
METS = ("pooled_ap", "pooled_roc", "within_roc")


def logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(x) - np.log1p(-x)


def load_baseline(root, method, corpus, seed, split):
    out = {}
    for line in open(root / method / corpus / f"seed_{seed}" / f"{split}.jsonl"):
        r = json.loads(line)
        out[r["video_id"]] = np.asarray(r["score"], dtype=np.float64)
    return out


def fit_len(x, T):
    """Baseline scores to the T seconds of our grid: cut, or pad with the last value (as baseline_plus_vlm.py)."""
    n = min(len(x), T)
    return np.concatenate([x[:n], np.full(T - n, x[n - 1] if n else 0.5)])


def load_hidden(root, method, corpus, seed, ids, T):
    """Variant "hid": per-second baseline hidden features on our grid (rows cut or the last row repeated, as fit_len;
    a video the baseline did not score gets zeros), standardized with the training split's mean and standard
    deviation. Returns {video id: tensor (T, D)} over all splits and the number of missing videos per split."""
    raw, missing = {}, {}
    for s in ("train", "val", "test"):
        z = np.load(root / method / corpus / f"seed_{seed}" / f"{s}_hidden.npz")
        missing[s] = sum(v not in z.files for v in ids[s])
        for v in ids[s]:
            if v in z.files:
                x = z[v].astype(np.float32)
                n = min(len(x), T[v])
                raw[v] = np.concatenate([x[:n], np.repeat(x[n - 1:n], T[v] - n, 0)]) if n else None
            else:
                raw[v] = None
    tr = np.concatenate([raw[v] for v in ids["train"] if raw[v] is not None])
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    D = tr.shape[1]
    out = {v: torch.as_tensor((x - mu) / sd if x is not None else np.zeros((T[v], D), np.float32),
                              dtype=torch.float32) for v, x in raw.items()}
    return out, missing


class Head(nn.Module):
    def __init__(self, feat_dim, use_z=False):
        super().__init__()
        self.use_z = use_z
        self.tok = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.proj = nn.Linear(feat_dim, 32) if feat_dim else None
        self.out = nn.Sequential(nn.Linear(32 + 32 + 1 + 3 + (32 if feat_dim else 0) + int(use_z), 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, item):
        """h_t. item: z (T,) (an input only if use_z, variant "hid"), pairs (P, 2) [second, token index], tok (N, 4),
        ctx (3,), feat."""
        z, pairs, tok, ctx, feat = item
        T = z.shape[0]
        s = torch.zeros(T, 32)
        m = torch.zeros(T, 32)
        cnt = torch.zeros(T, 1)
        if len(pairs):
            e = self.tok(tok)[pairs[:, 1]]                                  # P, 32
            s = s.index_add(0, pairs[:, 0], e)
            m = m.scatter_reduce(0, pairs[:, 0, None].expand(-1, 32), e, reduce="amax", include_self=True)
            cnt = cnt.index_add(0, pairs[:, 0], torch.ones(len(pairs), 1))
        x = [s, m, cnt / K_ASK, ctx[None].expand(T, 3)]
        if self.proj is not None:
            x.append(self.proj(feat))
        if self.use_z:
            x.append(z[:, None] / 5.0)
        return self.out(torch.cat(x, 1)).squeeze(1)


def build_item(T, z, nodes, feat):
    """nodes: list of (a, b, p) answered intervals; returns the tensors Head.forward reads."""
    toks, secs = [], []
    for a, b, p in nodes:
        t = np.arange(a, b)
        secs.append(t)
        toks.append(np.stack([np.full(len(t), float(logit(p)) / 5.0), np.full(len(t), p),
                              np.full(len(t), math.log(b - a) / 5.0), (t - a) / (b - a)], 1))
    if nodes:
        secs = np.concatenate(secs)
        pairs = np.stack([secs, np.arange(len(secs))], 1)
        toks = np.concatenate(toks).astype(np.float32)
        lps = [float(logit(p)) / 5.0 for _, _, p in nodes]
        ctx = [float(np.mean(lps)), float(np.max(lps)), len(nodes) / K_ASK]
    else:
        pairs, toks, ctx = np.zeros((0, 2)), np.zeros((0, 4), np.float32), [0.0, 0.0, 0.0]
    return (torch.as_tensor(z, dtype=torch.float32), torch.as_tensor(pairs, dtype=torch.int64),
            torch.as_tensor(toks), torch.as_tensor(ctx, dtype=torch.float32), feat)


def asked_nodes(run, T, P_v):
    tr = qtree.tree(T)
    out = []
    for node in run["asked"][:K_ASK]:
        a, b = int(tr["a"][node]), int(tr["b"][node])
        p = P_v.get((a, b))
        if p is not None:
            out.append((a, b, float(p)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--replay-dirs", nargs=3, required=True, help="policy_replay.py trial dirs of seeds 234 2025 3407")
    ap.add_argument("--variants", nargs="+", default=["nofeat", "feat"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="seeds this process trains")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--baseline-root", default="runs/20261108_astra_review/baseline_scores",
                    help="scores <split>.jsonl and, for variant hid, <split>_hidden.npz")
    ap.add_argument("--ours", default="runs/20261101_r6_length_table/final_curves/test_selected")
    ap.add_argument("--out-root", default="runs/20261108_astra_review/e1")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    hate = {s: {v for v in ids[s] if labels[v] == 1} for s in ("val", "test")}
    P, _, _ = qdata.load_soft_p(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, ids["train"] + ids["val"] + ids["test"], ["bert"])
    feats = {}
    if "feat" in a.variants:
        for v in store.T:
            feats[v] = torch.as_tensor(np.concatenate([store.visual(v, 0), store.at[v]], 1), dtype=torch.float32)
    ours = json.loads((ROOT / a.ours / a.corpus / "summary.json").read_text())["trials"]
    out_dir = (ROOT / a.out_root / a.corpus / a.method).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed, rd in zip(SEEDS, a.replay_dirs):
        if seed not in a.seeds or (out_dir / f"seed_{seed}.json").exists():
            continue
        rp = pickle.load(open(Path(rd) / "runs_L0.pkl", "rb"))
        assert int(rp["seed"]) == seed, (rd, rp["seed"])
        runs = rp["runs"]
        base = {s: load_baseline(ROOT / a.baseline_root, a.method, a.corpus, seed, s) for s in ("train", "val", "test")}
        z = {s: {v: logit(fit_len(base[s].get(v, np.full(store.T[v], 0.5)), store.T[v])) for v in ids[s]}
             for s in base}
        missing = {s: sum(v not in base[s] for v in ids[s]) for s in base}
        hfeat, hmiss = load_hidden(ROOT / a.baseline_root, a.method, a.corpus, seed, ids, store.T) \
            if "hid" in a.variants else ({}, None)
        fixed = {s: {v: asked_nodes(runs[s][v], store.T[v], P[v]) for v in ids[s]} for s in ("val", "test")}
        pol_train = {v: asked_nodes(runs["train"][v], store.T[v], P[v]) for v in ids["train"]}
        pool = {v: [(x, y, float(p)) for (x, y), p in P[v].items() if p is not None] for v in ids["train"]}
        name = next(n for n, r in ours.items() if int(r["seed"]) == seed)
        assert Path(rp["trial"]).resolve() == Path(ours[name]["trial"]).resolve(), (rp["trial"], ours[name]["trial"])
        row = {"replay": str(rd), "baseline_missing": missing, "hidden_missing": hmiss, "ours_trial": name,
               "ours_fixed8": {m: ours[name]["fixed"]["8"][m] for m in METS}, "variants": {}}
        for var in a.variants:
            vf = feats if var == "feat" else hfeat if var == "hid" else {}
            fd = next(iter(vf.values())).shape[1] if vf else 0
            alphas = (0.0,) if var == "hid" else ALPHAS
            ev = {s: {v: build_item(store.T[v], z[s][v], fixed[s][v], vf.get(v)) for v in ids[s]}
                  for s in ("val", "test")}
            best = None
            for lr in LRS:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                head = Head(fd, use_z=var == "hid")
                opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
                tr_ids = list(ids["train"])
                for ep in range(1, EPOCHS + 1):
                    head.train()
                    rng.shuffle(tr_ids)
                    for i in range(0, len(tr_ids), BATCH):
                        loss = 0.0
                        for v in tr_ids[i:i + BATCH]:
                            if rng.random() < 0.5 or not pool[v]:
                                nodes = pol_train[v]
                            else:
                                pick = rng.choice(len(pool[v]), size=min(K_ASK, len(pool[v])), replace=False)
                                nodes = [pool[v][j] for j in pick]
                            f = head(build_item(store.T[v], z["train"][v], nodes, vf.get(v)))
                            k = store.T[v] // 16 + 1
                            vs = f.topk(min(k, len(f))).values.mean()
                            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                                vs, torch.tensor(float(labels[v])))
                        opt.zero_grad()
                        (loss / len(tr_ids[i:i + BATCH])).backward()
                        opt.step()
                    head.eval()
                    with torch.no_grad():
                        hv = {v: head(ev["val"][v]).numpy() for v in ids["val"]}
                    for al in alphas:
                        sc = {v: 1.0 / (1.0 + np.exp(-(al * z["val"][v] + hv[v]))) for v in ids["val"]}
                        m = hc.frame_metrics(sc, gt["val"], hate["val"])
                        obj = (m["pooled_ap"] + m["pooled_roc"]) / 2
                        if best is None or obj > best["val_obj"]:
                            best = {"val_obj": float(obj), "lr": lr, "epoch": ep, "alpha": al,
                                    "val": {k_: float(m[k_]) for k_ in METS},
                                    "state": {k_: t.clone() for k_, t in head.state_dict().items()}}
                print(datetime.now().astimezone().isoformat(), a.corpus, a.method, seed, var, "lr", lr,
                      "best so far", best["lr"], best["epoch"], best["alpha"], round(best["val_obj"], 4), flush=True)
            head = Head(fd, use_z=var == "hid")
            head.load_state_dict(best.pop("state"))
            head.eval()
            with torch.no_grad():
                sc = {v: 1.0 / (1.0 + np.exp(-(best["alpha"] * z["test"][v] + head(ev["test"][v]).numpy())))
                      for v in ids["test"]}
            sp = out_dir / f"seed_{seed}_{var}_scores_test.jsonl"
            hc.write_scores(str(sp), sc)
            res = hc.run_evaluator(a.corpus, "test", str(sp), str(out_dir / f"seed_{seed}_{var}_metrics_test.json"))
            mt = res["results"]["score_av"]
            best["test"] = {"pooled_ap": mt["pr_auc"], "pooled_roc": mt["roc_auc"],
                            "within_roc": mt["per_video"]["macro_auc"]}
            row["variants"][var] = best
            print(a.corpus, a.method, seed, var, "test", {k_: round(x, 4) for k_, x in best["test"].items()},
                  "ours", {k_: round(x, 4) for k_, x in row["ours_fixed8"].items()}, flush=True)
        row.update(host=socket.gethostname(), created=datetime.now().astimezone().isoformat())
        (out_dir / f"seed_{seed}.json").write_text(json.dumps(row, indent=1) + "\n")
    if not all((out_dir / f"seed_{s_}.json").exists() for s_ in SEEDS):
        print("DONE seeds", a.seeds, "(summary when all three seeds are done)", flush=True)
        return
    report = {"corpus": a.corpus, "method": a.method, "created": datetime.now().astimezone().isoformat(),
              "per_seed": {str(s_): json.loads((out_dir / f"seed_{s_}.json").read_text()) for s_ in SEEDS}}
    rows = list(report["per_seed"].values())
    report["mean"] = {"ours_fixed8": {m: float(np.mean([r["ours_fixed8"][m] for r in rows])) for m in METS}}
    for var in a.variants:
        report["mean"][var] = {m: float(np.mean([r["variants"][var]["test"][m] for r in rows])) for m in METS}
        report["mean"][var + "_sd"] = {m: float(np.std([r["variants"][var]["test"][m] for r in rows])) for m in METS}
        report["mean"][var + "_ours_minus"] = {
            m: float(np.mean([r["ours_fixed8"][m] - r["variants"][var]["test"][m] for r in rows])) for m in METS}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    print("DONE", a.corpus, a.method, json.dumps(report["mean"]), flush=True)


if __name__ == "__main__":
    main()
