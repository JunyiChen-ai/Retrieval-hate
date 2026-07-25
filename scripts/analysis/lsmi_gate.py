#!/usr/bin/env python
"""
LSMI GATE — sample-level synergy / redundancy / uniqueness on the deployed (img, text) streams.

Design frozen in refine-logs/LSMI_GATE_RECORD.md §2 (commit d4b06f0) BEFORE this runner touched
any cache; AMENDMENTS AMD-1/AMD-2 (§2.6) were declared after the G1/G2 machinery gates fired and
BEFORE any real-data cell was computed. CPU-only, zero SLURM, zero GPU, zero Modal.
Read-only on banked caches; train+dev only, no test split is opened.

Fidelity: estimator functions are IMPORTED VERBATIM from external/baselines/LSMI @ 13e4db2
(hydra is stubbed only so main_lsmi.py imports at all; no released line is edited).
Declared deviations:
  * obtain_entropy_estimator_fixed = released loop + the missing optimizer.zero_grad()
    (the as-shipped loop still runs as arm A5 and gate G2b);
  * num_workers=0 in-memory loaders instead of the released num_workers=16 DataLoader;
  * lsmi_full() reproduces main_lsmi.LSMI_estimation line-for-line but keeps per-sample vectors
    (asserted equal to the released function's means to 1e-6 -> reported as maxabs);
  * AMD-1 K=5 stratified cross-fitted read (the released in-sample read is ALSO reported).
"""
import os, sys, json, time, math, hashlib, argparse, types, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "external", "baselines", "LSMI"))

if "hydra" not in sys.modules:                      # stub so released main_lsmi.py imports as-is
    _h = types.ModuleType("hydra")
    _h.main = lambda *a, **k: (lambda f: f)
    sys.modules["hydra"] = _h

from entropy_estimation import MargKernel                     # noqa: E402  (verbatim)
from utils import cls_network, setup_seed                     # noqa: E402  (verbatim)
import main_lsmi as ML                                        # noqa: E402  (verbatim)
import gaussian_data                                          # noqa: E402  (verbatim)
from sklearn.decomposition import PCA                         # noqa: E402
from sklearn.model_selection import StratifiedKFold           # noqa: E402

DEV, KFOLD = "cpu", 5
CKPT = os.path.join(ROOT, "refine-logs", ".lsmi_ckpt")
OUTP = os.path.join(ROOT, "refine-logs", "LSMI_GATE_OUT.json")   # overridden per stage in main()
os.makedirs(CKPT, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------- loaders (num_workers=0)
class _DS:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n


class SimpleLoader:
    def __init__(self, x1, x2, y, batch_size=32, shuffle=False, seed=0):
        self.x1, self.x2, self.y = x1, x2, y
        self.bs, self.shuffle, self.seed, self._ep = batch_size, shuffle, seed, 0
        self.dataset = _DS(x1.shape[0])

    def __len__(self): return int(math.ceil(self.x1.shape[0] / self.bs))

    def __iter__(self):
        n = self.x1.shape[0]
        if self.shuffle:
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed + self._ep))
            self._ep += 1
        else:
            idx = torch.arange(n)
        for i in range(0, n, self.bs):
            j = idx[i:i + self.bs]
            yield (self.x1[j], self.x2[j], self.y[j])


def make_cfg(d1, d2, n_classes=2, ep=30, bs=32):
    c = types.SimpleNamespace()
    c.device, c.batch_size = DEV, bs
    c.input_size_1, c.input_size_2 = int(d1), int(d2)
    c.embed_size, c.n_classes = 64, n_classes
    c.num_epochs_discriminator = c.num_epochs_entropy_estimator = ep
    return c


def obtain_entropy_estimator_fixed(cfg, train_loader):
    """main_lsmi.obtain_entropy_estimator + the missing optimizer.zero_grad()."""
    ms = [MargKernel(dim=cfg.input_size_1).to(cfg.device), MargKernel(dim=cfg.input_size_2).to(cfg.device)]
    opt = torch.optim.Adam([p for m in ms for p in m.parameters()], lr=1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    for _ in range(cfg.num_epochs_entropy_estimator):
        for m in ms: m.train()
        for batch in train_loader:
            m1, m2, _ = ML.obtain_feature_input(batch, device=cfg.device)
            opt.zero_grad()                                   # <-- THE FIX (absent upstream)
            (ms[0](m1) + ms[1](m2)).backward()
            opt.step()
        sch.step()
    return ms


# ------------------------------------------------------------------------- core estimation
def fit_models(P, y, cfg, seed, shipped, fit_knife=True):
    setup_seed(seed)
    disc = ML.obtain_discriminator(cfg, train_loader=SimpleLoader(P[0], P[1], y, cfg.batch_size, True, seed))
    ent = None
    if fit_knife:
        ld = SimpleLoader(P[0], P[1], y, cfg.batch_size, True, seed + 1)
        ent = ML.obtain_entropy_estimator(cfg, ld) if shipped else obtain_entropy_estimator_fixed(cfg, ld)
    return disc, ent


def read_pointwise(P, y, disc, ent, cfg):
    ld = SimpleLoader(P[0], P[1], y, cfg.batch_size, False)
    return dict(I1=ML.get_mutual_info(ld, disc[0], "modality_1", cfg),
                I2=ML.get_mutual_info(ld, disc[1], "modality_2", cfg),
                I12=ML.get_mutual_info(ld, disc[2], "modality_12", cfg),
                H1=ML.get_entropy(ld, ent[0], "modality_1", cfg),
                H2=ML.get_entropy(ld, ent[1], "modality_2", cfg))


def decompose(pw):
    """main_lsmi.LSMI_estimation lines 114-122, per-sample."""
    I1, I2, I12, H1, H2 = pw["I1"], pw["I2"], pw["I12"], pw["H1"], pw["H2"]
    r = torch.minimum(H1, H2) - torch.minimum(H1 - I1, H2 - I2)
    u1, u2 = I1 - r, I2 - r
    s = I12 - r - u1 - u2
    ra, u1a, u2a, sa = ML.RUS_adjustment([r, u1, u2, s])
    return dict(pw, r_raw=r, u1_raw=u1, u2_raw=u2, s_raw=s, r=ra, u1=u1a, u2=u2a, s=sa)


def qs(v):
    v = v.detach().cpu().numpy().astype(np.float64)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                q05=float(np.quantile(v, .05)), q50=float(np.quantile(v, .50)),
                q95=float(np.quantile(v, .95)), frac_pos=float((v > 0).mean()))


def acc(model, X, y):
    model.eval()
    with torch.no_grad():
        return float((model(X).argmax(1) == y).float().mean())


def summarize(res, y, K=2, accs=None):
    d = {k: float(res[k].mean()) for k in ("I1", "I2", "I12", "H1", "H2",
                                           "r_raw", "u1_raw", "u2_raw", "s_raw", "r", "u1", "u2", "s")}
    d["R"], d["U1"], d["U2"], d["S"] = d.pop("r"), d.pop("u1"), d.pop("u2"), d.pop("s")
    # S_share guarded: undefined when the total task-relevant info is at/below 0.05 nats
    d["S_share"] = d["S"] / d["I12"] if d["I12"] >= 0.05 else None
    d["SmR_estimator_free"] = d["I12"] - d["I1"] - d["I2"]
    d["I12_minus_maxI"] = d["I12"] - max(d["I1"], d["I2"])
    d["R_gt_U1U2"] = bool(d["R"] > d["U1"] + d["U2"])
    yy = y.numpy(); p = np.array([(yy == k).mean() for k in range(K)])
    delta = torch.tensor(-math.log(K) - np.log(p[yy]), dtype=res["I1"].dtype)
    r_e = torch.minimum(res["H1"], res["H2"]) - torch.minimum(res["H1"] - res["I1"] - delta,
                                                              res["H2"] - res["I2"] - delta)
    d["R_emp_prior_preadj"] = float(r_e.mean())
    d["S_emp_prior_preadj"] = float((res["I12"] + delta - r_e - (res["I1"] + delta - r_e)
                                     - (res["I2"] + delta - r_e)).mean())
    d["s_dist"], d["r_dist"] = qs(res["s"]), qs(res["r"])
    d["n"] = int(y.shape[0])
    if accs: d.update(accs)
    return d


def crossfit(P, y, cfg, seed, shipped, fit_knife=True, full_ent=None, K=KFOLD):
    """AMD-1: K-fold stratified cross-fitted pointwise read. Returns (pw, per-fold models, accs)."""
    n = y.shape[0]
    parts = {k: torch.zeros(n, dtype=torch.float64) for k in ("I1", "I2", "I12", "H1", "H2")}
    models, a = [], {"acc_img": [], "acc_text": [], "acc_joint": []}
    for f, (tri, tei) in enumerate(StratifiedKFold(K, shuffle=True, random_state=seed).split(np.zeros(n), y.numpy())):
        Pin, Pout = (P[0][tri], P[1][tri]), (P[0][tei], P[1][tei])
        disc, ent = fit_models(Pin, y[tri], cfg, seed + 100 + f, shipped, fit_knife=fit_knife)
        if ent is None: ent = full_ent
        pw = read_pointwise(Pout, y[tei], disc, ent, cfg)
        for k in parts: parts[k][torch.from_numpy(tei)] = pw[k].double()
        models.append((disc, ent, tri, tei))
        a["acc_img"].append(acc(disc[0], Pout[0], y[tei]))
        a["acc_text"].append(acc(disc[1], Pout[1], y[tei]))
        a["acc_joint"].append(acc(disc[2], torch.cat(Pout, 1), y[tei]))
    return parts, models, {k: float(np.mean(v)) for k, v in a.items()}


# ------------------------------------------------------------------------------ one cell
def run_cell(key, Ptr, Pdv, ytr, ydv, shipped=False, seed=42, nperm=0, cf_knife=True, fidelity=False):
    ck = os.path.join(CKPT, key.replace("/", "__") + ".json")
    if os.path.exists(ck):
        log(f"  [ckpt] {key}"); return json.load(open(ck))
    t0 = time.time()
    cfg = make_cfg(Ptr[0].shape[1], Ptr[1].shape[1])
    out = {"arm_dims": [int(Ptr[0].shape[1]), int(Ptr[1].shape[1])], "shipped_entropy_loop": bool(shipped),
           "crossfit_K": KFOLD, "crossfit_knife": bool(cf_knife)}

    disc_f, ent_f = fit_models(Ptr, ytr, cfg, seed, shipped)           # full-train (released protocol)
    ins = decompose(read_pointwise(Ptr, ytr, disc_f, ent_f, cfg))
    out["train_insample"] = summarize(ins, ytr, accs=dict(
        acc_img=acc(disc_f[0], Ptr[0], ytr), acc_text=acc(disc_f[1], Ptr[1], ytr),
        acc_joint=acc(disc_f[2], torch.cat(Ptr, 1), ytr)))
    dv = decompose(read_pointwise(Pdv, ydv, disc_f, ent_f, cfg))
    out["dev"] = summarize(dv, ydv, accs=dict(
        acc_img=acc(disc_f[0], Pdv[0], ydv), acc_text=acc(disc_f[1], Pdv[1], ydv),
        acc_joint=acc(disc_f[2], torch.cat(Pdv, 1), ydv)))
    if fidelity:
        Rr, U1r, U2r, Sr = ML.LSMI_estimation(SimpleLoader(Ptr[0], Ptr[1], ytr, 32, False), disc_f, ent_f, cfg)
        t = out["train_insample"]
        out["fidelity_maxabs_vs_released"] = float(max(abs(Rr.item() - t["R"]), abs(U1r.item() - t["U1"]),
                                                       abs(U2r.item() - t["U2"]), abs(Sr.item() - t["S"])))
    pw, fmods, accs = crossfit(Ptr, ytr, cfg, seed, shipped, fit_knife=cf_knife, full_ent=ent_f)
    out["train_crossfit"] = summarize(decompose(pw), ytr, accs=accs)

    if nperm > 0:
        rng = np.random.default_rng(90000)
        pck = ck + ".perm.json"
        nul = json.load(open(pck)) if os.path.exists(pck) else {"train_crossfit": [], "dev": []}
        start = len(nul["train_crossfit"])
        for _ in range(start):            # replay consumed draws AT THE SAME SIZES -> bit-identical
            rng.permutation(len(ytr)); rng.permutation(len(ydv))
        if start:
            log(f"    [perm ckpt] resuming at {start}/{nperm}")
        for b in range(start, nperm):
            ytr_p = ytr[torch.from_numpy(rng.permutation(len(ytr)))]
            ydv_p = ydv[torch.from_numpy(rng.permutation(len(ydv)))]
            pk = {k: torch.zeros(len(ytr), dtype=torch.float64) for k in ("I1", "I2", "I12", "H1", "H2")}
            for f, (dsc, ent, tri, tei) in enumerate(fmods):        # KNIFE reused: it is LABEL-FREE
                cfgf = cfg
                dp, _ = fit_models((Ptr[0][tri], Ptr[1][tri]), ytr_p[tri], cfgf, seed + 5000 + 7 * b + f,
                                   shipped, fit_knife=False)
                q = read_pointwise((Ptr[0][tei], Ptr[1][tei]), ytr_p[tei], dp, ent, cfgf)
                for k in pk: pk[k][torch.from_numpy(tei)] = q[k].double()
            for nm, res, yy in (("train_crossfit", decompose(pk), ytr_p),):
                nul[nm].append(dict(S=float(res["s"].mean()), R=float(res["r"].mean()),
                                    I12=float(res["I12"].mean())))
            dp, _ = fit_models(Ptr, ytr_p, cfg, seed + 9000 + b, shipped, fit_knife=False)
            rd = decompose(read_pointwise(Pdv, ydv_p, dp, ent_f, cfg))
            nul["dev"].append(dict(S=float(rd["s"].mean()), R=float(rd["r"].mean()),
                                   I12=float(rd["I12"].mean())))
            json.dump(nul, open(pck, "w"))
            if (b + 1) % 10 == 0: log(f"    perm {b+1}/{nperm}")
        for nm, lst in nul.items():
            arr = {k: np.array([x[k] for x in lst]) for k in ("S", "R", "I12")}
            arr["S_share"] = np.where(arr["I12"] >= 0.05, arr["S"] / np.maximum(arr["I12"], 1e-9), np.nan)
            out["perm_null_" + nm] = {k: dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)),
                                              q95=float(np.nanquantile(v, .95)), max=float(np.nanmax(v)),
                                              n=int(np.sum(~np.isnan(v)))) for k, v in arr.items()}
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(ck, "w"), indent=1)
    cf = out["train_crossfit"]
    log(f"  [done] {key} {out['seconds']}s  CF: S={cf['S']:.4f} I12={cf['I12']:.4f} "
        f"share={cf['S_share']} accs={cf['acc_img']:.3f}/{cf['acc_text']:.3f}/{cf['acc_joint']:.3f}")
    return out


# ------------------------------------------------------------------------- data / arms
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


DATASETS = {
    "MHC_zh": dict(dir="MHC_zh", stem="Qwen2.5-VL-7B-Instruct-LoRA_HF", lineage="generic-LoRA job 13150 (ZH floor)"),
    "HateMM": dict(dir="HateMM", stem="Qwen2.5-VL-7B-Instruct-LoRA-curric_HF", lineage="curric-LoRA job 13241 (project best)"),
    "MHC_en": dict(dir="MHC", stem="Qwen2.5-VL-7B-Instruct_HF", lineage="frozen Qwen (EN closed)"),
}
ARMS = {
    "A1": dict(kind="pca", dim=64, whiten=True, shipped=False, label="d'=64 PCA-whitened (PRIMARY)"),
    "A2": dict(kind="pca", dim=256, whiten=True, shipped=False, label="d'=256 PCA-whitened (dim replication)"),
    "A3": dict(kind="pca", dim=64, whiten=False, shipped=False, label="d'=64 PCA common-scale (scale sensitivity)"),
    "A4": dict(kind="raw", dim=3584, whiten=False, shipped=False, label="RAW d=3584 common-scale (F41 raw arm)"),
    "A5": dict(kind="pca", dim=64, whiten=True, shipped=True, label="d'=64 PCA-whitened AS-SHIPPED (no zero_grad)"),
}


def load_ds(name):
    m = DATASETS[name]; out, shas = {}, {}
    for split, pre in (("train", "train"), ("dev", "dev_seen")):
        p = os.path.join(ROOT, "data", "CLIP_Embedding", m["dir"], f"{pre}_{m['stem']}.pt")
        shas[split] = sha256(p)
        d = torch.load(p, map_location="cpu", weights_only=False)
        out[split] = (d["img_feats"].float(), d["text_feats"].float(), d["labels"].long())
    return out, shas


def project(tr, dv, kind, dim, whiten):
    A1, A2 = tr[0].numpy().astype(np.float64), tr[1].numpy().astype(np.float64)
    B1, B2 = dv[0].numpy().astype(np.float64), dv[1].numpy().astype(np.float64)
    if kind == "raw":
        m1, m2 = A1.mean(0, keepdims=True), A2.mean(0, keepdims=True)
        P1, P2, Q1, Q2 = A1 - m1, A2 - m2, B1 - m1, B2 - m2
    else:
        p1 = PCA(n_components=dim, whiten=whiten, random_state=0, svd_solver="full").fit(A1)
        p2 = PCA(n_components=dim, whiten=whiten, random_state=0, svd_solver="full").fit(A2)
        P1, P2, Q1, Q2 = p1.transform(A1), p2.transform(A2), p1.transform(B1), p2.transform(B2)
    if not whiten:                       # ONE shared scalar -> preserves relative stream scale
        sig = math.sqrt((P1.var(0, ddof=1).sum() + P2.var(0, ddof=1).sum()) / (P1.shape[1] + P2.shape[1]))
        P1, P2, Q1, Q2 = P1 / sig, P2 / sig, Q1 / sig, Q2 / sig
    f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    return (f(P1), f(P2)), (f(Q1), f(Q2))


def xor_control(n_tr, n_dv, dim=64, margin=3.0, seed=7):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=dim); w1 /= np.linalg.norm(w1)
    w2 = rng.normal(size=dim); w2 /= np.linalg.norm(w2)      # shared by train and dev
    def gen(n):
        y = rng.integers(0, 2, n); b1 = rng.integers(0, 2, n); b2 = np.bitwise_xor(b1, y)
        x1 = margin * (2 * b1 - 1)[:, None] * w1[None] + rng.normal(size=(n, dim))
        x2 = margin * (2 * b2 - 1)[:, None] * w2[None] + rng.normal(size=(n, dim))
        return torch.from_numpy(x1).float(), torch.from_numpy(x2).float(), torch.from_numpy(y).long()
    return gen(n_tr), gen(n_dv)


# -------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main", choices=["main", "raw", "gates", "merge"])
    ap.add_argument("--nperm", type=int, default=50)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    global OUTP
    if a.stage == "merge":
        out = {}
        for st in ("gates", "main", "power", "raw"):
            f = os.path.join(ROOT, "refine-logs", f".lsmi_out_{st}.json")
            if not os.path.exists(f): continue
            part = json.load(open(f))
            out["meta"] = part.get("meta", out.get("meta", {}))
            for sec in ("controls", "datasets"):
                if sec in part:
                    for k, v in part[sec].items():
                        out.setdefault(sec, {}).setdefault(k, {}).update(v) if isinstance(v, dict) else None
        if os.path.exists(OUTP):        # never clobber a computed verdict block
            prev = json.load(open(OUTP))
            if "verdict" in prev:
                out["verdict"] = prev["verdict"]
                out["verdict"]["STALE_AFTER_MERGE"] = ("re-run scripts/analysis/lsmi_gate_verdict.py "
                                                       "if new arms were merged")
        json.dump(out, open(OUTP, "w"), indent=1); log("MERGED -> " + OUTP); return
    OUTP = os.path.join(ROOT, "refine-logs", f".lsmi_out_{a.stage}.json")
    R = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
    R["meta"] = dict(lsmi_repo_head="13e4db2e033a3721d5ea7c0e31c540b3445a5532", torch=torch.__version__,
                     numpy=np.__version__, device=DEV, pregate_commit="d4b06f0", crossfit_K=KFOLD,
                     generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    save = lambda: json.dump(R, open(OUTP, "w"), indent=1)

    if a.stage == "gates":
        R.setdefault("controls", {})
        log("=== G2 shipped gaussian demo (verbatim generator, n=1600/400, rho=.5) ===")
        o = gaussian_data.generate_gaussian_data(1600, 400, 0.5, 0.5, 2)
        t = lambda z: torch.from_numpy(z).float()
        g1 = (t(o["train_data"][0]), t(o["train_data"][1]), torch.from_numpy(o["train_data"][2]).long())
        g2 = (t(o["test_data"][0]), t(o["test_data"][1]), torch.from_numpy(o["test_data"][2]).long())
        for sh, nm in ((False, "G2a_gaussian_zerograd_fixed"), (True, "G2b_gaussian_as_shipped")):
            R["controls"][nm] = run_cell(f"v2_gauss_{nm}", (g1[0], g1[1]), (g2[0], g2[1]), g1[2], g2[2],
                                         shipped=sh, fidelity=True); save()
        log("=== G1 XOR power gates (AMD-2: localize the power wall in n and d) ===")
        cells = []
        for ds in DATASETS:
            d, _ = load_ds(ds)
            cells.append((f"G1_xor_{ds}_n{d['train'][0].shape[0]}_d64",
                          d["train"][0].shape[0], d["dev"][0].shape[0], 64))
        cells += [("G1b_xor_n8000_d64", 8000, 2000, 64), ("G1c_xor_n579_d8", 579, 78, 8),
                  ("G1d_xor_n8000_d8", 8000, 2000, 8), ("G1e_xor_n2000_d64", 2000, 500, 64)]
        for nm, ntr, ndv, dim in cells:
            (x1, x2, y), (v1, v2, vy) = xor_control(ntr, ndv, dim=dim)
            P, Q = project((x1, x2), (v1, v2), "pca", dim, True)
            R["controls"][nm] = run_cell("v2_" + nm, P, Q, y, vy); save()
        log("WROTE " + OUTP); return

    arms = ["A4"] if a.stage == "raw" else ["A1", "A2", "A3", "A5"]
    R.setdefault("datasets", {})
    for ds in DATASETS:
        d, shas = load_ds(ds)
        e = R["datasets"].setdefault(ds, {})
        e["sha256"], e["lineage"] = shas, DATASETS[ds]["lineage"]
        e["n"] = dict(train=int(d["train"][2].shape[0]), dev=int(d["dev"][2].shape[0]),
                      pos_train=int((d["train"][2] == 1).sum()), pos_dev=int((d["dev"][2] == 1).sum()))
        tr, dv = (d["train"][0], d["train"][1]), (d["dev"][0], d["dev"][1])
        ytr, ydv = d["train"][2], d["dev"][2]
        for arm in arms:
            A = ARMS[arm]; log(f"=== {ds} / {arm}: {A['label']} ===")
            P, Q = project(tr, dv, A["kind"], A["dim"], A["whiten"])
            e[arm] = run_cell(f"v2_{ds}_{arm}", P, Q, ytr, ydv, shipped=A["shipped"],
                              nperm=(a.nperm if arm == "A1" else 0),
                              cf_knife=(A["kind"] != "raw"), fidelity=(arm == "A1"))
            e[arm]["arm_label"] = A["label"]; save()
        if a.stage == "main":
            log(f"=== {ds} / C1 duplicate-stream control (ground truth S=0) ===")
            P, Q = project((tr[0], tr[0]), (dv[0], dv[0]), "pca", 64, True)
            e["C1_dup_img"] = run_cell(f"v2_{ds}_C1dup", P, Q, ytr, ydv); save()
            log(f"=== {ds} / C2 within-stream split-half control ===")
            h = tr[0].shape[1] // 2
            P, Q = project((tr[0][:, :h], tr[0][:, h:]), (dv[0][:, :h], dv[0][:, h:]), "pca", 64, True)
            e["C2_splithalf_img"] = run_cell(f"v2_{ds}_C2half", P, Q, ytr, ydv); save()
    save(); log("WROTE " + OUTP)


if __name__ == "__main__":
    main()
