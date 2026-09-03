"""Mechanical shape/gradient checks for evidence_chain_net on random tensors (no data, no training)."""
import sys, math, torch, numpy as np
sys.path.insert(0, "experiments/20260904_evidence_chain_net"); sys.path.insert(0, "src")
import dataset as ds, model as M, train as T
torch.manual_seed(0)
class Pot:  # mimic Potentials
    a = 0.118; p0_hate = 0.35
B, L = 3, 40
def make_batch():
    b = {}
    b["f_v"] = torch.randn(B, L, ds.V_DIM); b["f_a"] = torch.randn(B, L, ds.F_A_DIM)
    mask = torch.ones(B, L, dtype=torch.bool); mask[1, 30:] = False; mask[2, 12:] = False
    b["mask"] = mask
    w = torch.zeros(B, L, dtype=torch.long); j = torch.zeros(B, L, dtype=torch.long)
    for i in range(B):
        n = int(mask[i].sum())
        wi = np.clip(((np.arange(n) + 0.5) * ds.K / n).astype(int), 0, ds.K - 1)
        w[i, :n] = torch.tensor(wi); w[i, n:] = ds.K
        j[i, :n] = torch.tensor(ds.BLOCK_OF_WINDOW[wi]); j[i, n:] = ds.J
    b["w"], b["j"] = w, j
    for k in ("phi_f", "phi_c", "bf", "bc", "bfp", "bfn", "ph"):
        b[k] = (torch.rand(B, L) > 0.5).float() if k in ("bf", "bc", "bfp", "bfn") else torch.randn(B, L) * 0.5
    b["ph"] = torch.rand(B, L)
    b["n_w"] = torch.ones(B, L) * 2; b["n_j"] = torch.ones(B, L) * 10
    for k in ("phi_f", "phi_c", "bf", "bc", "bfp", "bfn", "ph", "n_w", "n_j"):
        b[k] = b[k] * mask.float()
    b["profile"] = torch.rand(B, ds.PROFILE_DIM)
    b["label"] = torch.tensor([1.0, 0.0, 1.0])
    return b
cfg = T.Args(dict(T.DEFAULTS)); cfg["dropout"] = 0.1
for abl in M.MODEL_ABLATIONS:
    m = M.EvidenceChainNet(cfg, Pot, abl)
    b = make_batch()
    out = m(b)
    lv, _, _ = T.video_loss(out, b["label"]); lb = T.block_mil_loss(out, b, 16)
    lc = T.contrast_loss(out, b, "posterior", 0.1, 16, 256)
    if abl == "topk_head":   # no chain in this arm: distill weight is 0 in train.py
        ldc = ldu = torch.zeros(())
    else:
        ldc = T.distill_loss(out, b, True); ldu = T.distill_loss(out, b, False)
    assert torch.isfinite(ldc) and torch.isfinite(ldu), abl
    tot = lv + lb + lc + ldc + ldu
    tot.backward()
    g = [p.grad for p in m.parameters() if p.grad is not None]
    fin = all(torch.isfinite(x).all() for x in g)
    assert out["score"].shape == (B, L) and torch.isfinite(out["score"][b["mask"]]).all(), abl
    print("%-18s video %.3f block %.3f contrast %.3f distill %.3f/%.3f | grads finite %s | n_grad %d | d_v %s | a_step %s | score is u %s"
          % (abl, lv.item(), lb.item(), lc.item(), ldc.item(), ldu.item(), fin, len(g), np.round(out["d_v"].detach().numpy(), 3), np.round(out["a_step"].detach().numpy(), 3), bool(torch.equal(out["score"], out["u"]))))
m = M.EvidenceChainNet(cfg, Pot, "full"); b = make_batch(); out = m(b)
for mode in ("self_topk", "vlm_thresh"):
    print(mode, T.contrast_loss(out, b, mode, 0.1, 16, 256).item())
# all-negative batch and T<2 positive
b2 = make_batch(); b2["label"] = torch.zeros(B); print("all-negative contrast", T.contrast_loss(m(b2), b2, "posterior", 0.1, 16, 256).item(), "block", T.block_mil_loss(m(b2), b2, 16).item())
b3 = make_batch(); b3["mask"][:] = False; b3["mask"][:, :1] = True; b3["w"][:, 1:] = ds.K; b3["j"][:, 1:] = ds.J
o3 = m(b3); print("T=1 rows: contrast", T.contrast_loss(o3, b3, "posterior", 0.1, 16, 256).item(), "block", T.block_mil_loss(o3, b3, 16).item(), "score", o3["score"][:, 0])
# fit_length check: phi_f sum preserved, padding indices
vt = {k: np.random.rand(97).astype(np.float32) for k in ds.ROW_KEYS}; vt["w"] = np.clip(((np.arange(97)+.5)*30/97).astype(int),0,29); vt["j"] = ds.BLOCK_OF_WINDOW[vt["w"]]
fv, fa, o, mk = ds.fit_length(np.random.rand(97, ds.V_DIM).astype(np.float32), np.random.rand(97, ds.F_A_DIM).astype(np.float32), vt, 40)
print("fit_length long: phi_f sum before %.4f after %.4f | j monotone %s | mask all %s" % (vt["phi_f"].sum(), o["phi_f"].sum(), bool((np.diff(o["j"])>=0).all()), mk.all()))
fv, fa, o, mk = ds.fit_length(np.random.rand(20, ds.V_DIM).astype(np.float32), np.random.rand(20, ds.F_A_DIM).astype(np.float32), {k: v[:20] for k, v in vt.items()}, 40)
print("fit_length short: pad j value %d (J=%d), w %d (K=%d), mask sum %d" % (o["j"][-1], ds.J, o["w"][-1], ds.K, mk.sum()))
m = M.EvidenceChainNet(cfg, Pot, "full"); b = make_batch(); out = m(b)
lp = torch.sigmoid(out["chain_logodds"]).detach(); rho = torch.exp(out["log_rho"]).detach()
q = (lp / (1 - rho)[:, None]).clamp(max=1)
print("distill target: pos rows q>=post %s | q<=1 %s | rho %s" % (bool((q[[0, 2]] >= lp[[0, 2]] - 1e-6).all()), bool((q <= 1).all()), np.round(rho.numpy(), 4)))
print("OK")
