"""Crash diagnosis for the revision-2 HateMM failure (val scores became NaN -> evaluator
ZeroDivisionError). Trains a few epochs with a failing trial's hyper-parameters and, after
each epoch, scans every validation video for non-finite model outputs. Output goes to
stdout only (no runs/ directory)."""
import json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
import train as T, dataset as ds, model as M
from torch.utils.data import DataLoader
import torch.nn.functional as F

corpus, ablation = sys.argv[1], sys.argv[2]
cfg = dict(T.DEFAULTS); cfg.update(json.loads(sys.argv[3])); n_epoch = int(sys.argv[4])
device = "cuda"
torch.manual_seed(234); np.random.seed(234)
labels = T.hdata.load_labels(corpus)
val_gt = T.hdata.gt_arrays(corpus, "val")
train_ids = T.usable(corpus, T.hdata.load_split(corpus, "train"))
val_ids = [v for v in T.usable(corpus, T.hdata.load_split(corpus, "val")) if v in val_gt]
V = {k: T.vlm_verdict.load_verdicts(corpus, k=k, tag="qwen") for k in (T.K_FINE, T.J_COARSE)}
binary = {v: (T.verdict_hmm.binarize(V[T.K_FINE][v]), T.verdict_hmm.binarize(V[T.J_COARSE][v]))
          for v in V[T.K_FINE] if v in V[T.J_COARSE]}
hmm, _, _ = T.fit_hmm(train_ids, labels, binary); pot = ds.Potentials(hmm)
caches = {n: ds.VideoCache(corpus, ids, binary, pot) for n, ids in (("train", train_ids), ("val", val_ids))}
a = T.Args(cfg)
train_loader = DataLoader(ds.TrainDataset(corpus, train_ids, labels, caches["train"], a.max_seqlen, a.crop_repeat),
                          batch_size=a.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(ds.EvalDataset(corpus, val_ids, caches["val"]), batch_size=1, shuffle=False, num_workers=4)
model_abl = ablation if ablation in M.MODEL_ABLATIONS else "full"
model = M.EvidenceChainNet(a, pot, model_abl).to(device)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
KEYS = ("u", "d_v", "gf", "gc", "log_Z", "log_Z0", "log_rho", "chain_logodds", "score")

def scan(tag):
    model.eval(); bad = 0
    with torch.no_grad():
        for item in val_loader:
            b = T.eval_batch(item, device); o = model(b)
            T_ = int(b["mask"].shape[1])
            nf = {k: int((~torch.isfinite(o[k])).sum()) for k in KEYS if k in o}
            if any(nf.values()):
                bad += 1
                if bad <= 3:
                    print("%s | %s T=%d nonfinite=%s | u [%.2f,%.2f] d_v %s a_step %s log_Z %s log_Z0 %s"
                          % (tag, item["vid"][0], T_, {k: v for k, v in nf.items() if v},
                             o["u"].min().item(), o["u"].max().item(),
                             np.round(o["d_v"].cpu().numpy(), 4), np.round(o["a_step"].cpu().numpy(), 6),
                             np.round(o["log_Z"].cpu().numpy(), 2), np.round(o["log_Z0"].cpu().numpy(), 2)), flush=True)
                    ph = b["phi_f"][0]; print("   phi_f [%.3f,%.3f] n_w [%.0f,%.0f] phi_c [%.3f,%.3f] gf [%.3f,%.3f] gc [%.3f,%.3f] post_s1 nonfinite %d"
                          % (ph.min(), ph.max(), b["n_w"][0].min(), b["n_w"][0].max(), b["phi_c"][0].min(), b["phi_c"][0].max(),
                             o["gf"].min(), o["gf"].max(), o["gc"].min(), o["gc"].max(), int((~torch.isfinite(o["post"])).sum())), flush=True)
    print("%s | %d/%d val videos with non-finite outputs" % (tag, bad, len(val_ids)), flush=True)
    model.train(); return bad

w_nonfinite = lambda: sum(int((~torch.isfinite(p)).sum()) for p in model.parameters())
scan("epoch 0")
for epoch in range(n_epoch):
    t0 = time.time(); step = 0
    for batch in train_loader:
        batch = T.to_device(batch, device); out = model(batch); y = batch["label"]
        lv, _, _ = T.video_loss(out, y)
        ld = F.binary_cross_entropy_with_logits(out["d_logit"], y)
        lb = T.block_mil_loss(out, batch, a.topk_div)
        lc = T.contrast_loss(out, batch, "posterior", a.contrast_tau, a.topk_div, a.contrast_max_normal) if epoch > 0 else torch.zeros((), device=device)
        lds = T.distill_loss(out, batch, True) if ablation not in ("chain_output", "no_distill") else torch.zeros((), device=device)
        total = lv + lb + lc + ld + lds
        if not torch.isfinite(total):
            print("epoch %d step %d NON-FINITE LOSS video %.3f block %.3f contrast %.3f density %.3f distill %.3f | u [%.1f,%.1f] log_rho min %.3f"
                  % (epoch + 1, step, lv, lb, lc, ld, lds, out["u"].min(), out["u"].max(), out["log_rho"].min()), flush=True)
            sys.exit(0)
        opt.zero_grad(); total.backward(retain_graph=True)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(gn):
            print("epoch %d step %d NON-FINITE GRAD NORM %s | losses video %.3f block %.3f contrast %.3f density %.3f distill %.3f | log_rho min %.4g max %.4g | y %s"
                  % (epoch + 1, step, gn.item(), lv, lb, lc, ld, lds, out["log_rho"].min(), out["log_rho"].max(), y.int().tolist()), flush=True)
            for name, l in (("video", lv), ("block", lb), ("contrast", lc), ("density", ld), ("distill", lds)):
                if not l.requires_grad: continue
                opt.zero_grad(); l.backward(retain_graph=True)
                g = [p.grad for p in model.parameters() if p.grad is not None]
                mx = max(float(x.abs().max()) for x in g) if g else 0.0
                print("   %s: grad max %.4g finite %s" % (name, mx, all(torch.isfinite(x).all() for x in g)), flush=True)
            per = -(y * out["log_p_video"] + (1 - y) * out["log_rho"])
            print("   per-video loss", np.round(per.detach().cpu().numpy(), 3).tolist(), flush=True)
            print("   log_rho", np.round(out["log_rho"].detach().cpu().numpy(), 6).tolist(), flush=True)
            sys.exit(0)
        opt.step(); step += 1
        if w_nonfinite():
            print("epoch %d step %d NON-FINITE WEIGHTS after step" % (epoch + 1, step), flush=True); sys.exit(0)
    print("epoch %d done %.0fs | last losses video %.3f block %.3f contrast %.3f density %.3f distill %.3f | u [%.1f,%.1f]"
          % (epoch + 1, time.time() - t0, lv, lb, lc, ld, lds, out["u"].min(), out["u"].max()), flush=True)
    if scan("epoch %d" % (epoch + 1)) and epoch >= 1:
        break
