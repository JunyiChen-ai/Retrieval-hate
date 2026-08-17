"""
Round-4 pilot harness: frozen features + bare BCE head (the new baseline established by
idea-stage/RGCL_ABLATION_RESULT.md, cell L1+I1).

Protocol (user standing instruction, 2026-08-09): train on train, select epoch on val,
REPORT TEST. Multi-seed. This module only provides the machinery; every pilot supplies
its own frozen decision rule.

Architecture mirrors src/model/classifier.py::classifier_hateClipper with the deployed
hyper-parameters from scripts/rgcl_ablation_grid.sh:
  map_dim 1024, proj_dim 1024, num_layers 3, fusion align (Hadamard), dropout (0.2,0.4,0.1),
  AdamW lr 1e-4, batch 64, 30 epochs, warmup 5.
Difference from run_rac.py (declared, not hidden): epoch selection here is on val macro-F1
of the head, because the bare-BCE arm has no kNN read-out to select on. Absolute numbers are
therefore near, not byte-identical to, the ablation table; every pilot verdict is a
seed-paired delta computed INSIDE this harness, so the comparison is internally same-frame.
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP = "openai_clip-vit-large-patch14-336_HF"
QWEN = "Qwen2.5-VL-7B-Instruct_HF"
SPLIT_FILE = {"train": "train", "val": "dev_seen", "test": "test_seen"}


def feat_path(dataset, model_tag, split):
    return os.path.join(ROOT, "data", "CLIP_Embedding", dataset,
                        f"{SPLIT_FILE[split]}_{model_tag}.pt")


def load_split(dataset, model_tag, split):
    d = torch.load(feat_path(dataset, model_tag, split), map_location="cpu",
                   weights_only=False)
    return {
        "ids": list(d["ids"][0]) if isinstance(d["ids"][0], list) else list(d["ids"]),
        "img": d["img_feats"].float(),
        "txt": d["text_feats"].float(),
        "y": torch.as_tensor(d["labels"]).float().view(-1),
    }


class Head(nn.Module):
    """classifier_hateClipper, align fusion, with an optional extra input block."""

    def __init__(self, img_dim, txt_dim, extra_dim=0, map_dim=1024, proj_dim=1024,
                 num_layers=3, dropout=(0.2, 0.4, 0.1)):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, map_dim), nn.Dropout(dropout[0]))
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, map_dim), nn.Dropout(dropout[0]))
        in_shape = map_dim + extra_dim
        layers = [nn.Dropout(dropout[1])]
        for _ in range(num_layers):
            layers += [nn.Linear(in_shape, proj_dim), nn.ReLU(), nn.Dropout(dropout[2])]
            in_shape = proj_dim
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(proj_dim, 1)

    def forward(self, img, txt, extra=None):
        i = nn.functional.normalize(self.img_proj(img), p=2, dim=1)
        t = nn.functional.normalize(self.txt_proj(txt), p=2, dim=1)
        x = torch.mul(i, t)
        if extra is not None:
            x = torch.cat([x, extra], dim=1)
        return self.out(self.mlp(x))


def macro_f1(y, p):
    return f1_score(y, (p >= 0.5).astype(int), average="macro")


def train_head(tr, va, te, seed, epochs=30, warmup=5, lr=1e-4, bs=64,
               device="cuda", extra=None, num_layers=3, sample_w=None,
               select="val_macro_f1"):
    """Returns dict with test/val probabilities at the val-selected epoch.

    extra: optional dict split-> tensor of extra input features concatenated post-fusion.
    sample_w: optional per-train-item BCE weight vector (torch tensor, len == n_train).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    ex = extra or {}
    ex_dim = ex["train"].shape[1] if "train" in ex else 0
    model = Head(tr["img"].shape[1], tr["txt"].shape[1], extra_dim=ex_dim,
                 num_layers=num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = tr["y"].shape[0]
    Xi, Xt, Y = tr["img"].to(device), tr["txt"].to(device), tr["y"].to(device)
    Xe = ex["train"].to(device) if ex_dim else None
    W = sample_w.to(device) if sample_w is not None else None
    packs = {}
    for name, s in (("val", va), ("test", te)):
        packs[name] = (s["img"].to(device), s["txt"].to(device),
                       ex[name].to(device) if ex_dim else None, s["y"].numpy())

    best = (-1.0, None)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            logit = model(Xi[idx], Xt[idx], Xe[idx] if Xe is not None else None).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, Y[idx], weight=W[idx] if W is not None else None)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            probs = {}
            for name, (a, b, c, _) in packs.items():
                probs[name] = torch.sigmoid(model(a, b, c).squeeze(1)).cpu().numpy()
        if ep >= warmup:
            score = macro_f1(packs["val"][3], probs["val"])
            if score > best[0]:
                best = (score, {"ep": ep, "val": probs["val"], "test": probs["test"]})
    sel = best[1]
    yte = packs["test"][3]
    return {
        "epoch": sel["ep"],
        "val_macro_f1": float(best[0]),
        "test_macro_f1": float(macro_f1(yte, sel["test"])),
        "test_acc": float(((sel["test"] >= 0.5).astype(int) == yte).mean()),
        "test_roc": float(roc_auc_score(yte, sel["test"])),
        "test_prob": sel["test"],
        "val_prob": sel["val"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ImpliHateVid")
    ap.add_argument("--model", default=CLIP)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tr = load_split(a.dataset, a.model, "train")
    va = load_split(a.dataset, a.model, "val")
    te = load_split(a.dataset, a.model, "test")
    print(f"{a.dataset}/{a.model}: {len(tr['y'])}/{len(va['y'])}/{len(te['y'])}"
          f" img{tr['img'].shape[1]} txt{tr['txt'].shape[1]}")
    rows = []
    for s in a.seeds:
        r = train_head(tr, va, te, s)
        rows.append(r)
        print(f"seed {s}: ep={r['epoch']} val={r['val_macro_f1']:.4f} "
              f"test_macroF1={r['test_macro_f1']:.4f} acc={r['test_acc']:.4f} "
              f"roc={r['test_roc']:.4f}")
    m = np.mean([r["test_macro_f1"] for r in rows])
    sd = np.std([r["test_macro_f1"] for r in rows])
    print(f"TEST macro-F1 {m:.4f} +/- {sd:.4f}")
    if a.out:
        json.dump({"dataset": a.dataset, "model": a.model,
                   "rows": [{k: v for k, v in r.items() if not k.endswith("prob")}
                            for r in rows],
                   "mean": float(m), "std": float(sd)}, open(a.out, "w"), indent=2)
