#!/usr/bin/env python
"""B-SRTD pilot — Balanced Semantic Response-Tensor Distillation (C4 revived, R1).

Decision rules frozen in research-wiki/EXP_bsrtd_prereg.md BEFORE this file existed.
Nothing here selects, tunes or thresholds on test.  Test probabilities are computed inside
the training loop (to avoid a second pass) but are withheld from every selection path and
printed only in the final block.

MECHANISM (frozen).  Each 2x2 lattice gives four cells (orig / targetsub / stancerev /
both).  The teacher scores all four; three finite differences form the response tensor
    dA = T1 - T0 , dB = T2 - T0 , dAB = (T3 - T2) - (T1 - T0).
The student is the deployed frozen-feature head; its per-cell probabilities give the same
three differences.  The auxiliary loss matches the student's response tensor to the
teacher's, in probability space (both in [0,1], so no temperature hyper-parameter).

ARMS
    bce    bare BCE head (project baseline)
    pair   pairwise-AUC + 0.1 BCE, single head (the sec 8.10(2b) baseline)
    cad    pair + BCE on the four cells against their GOLD cell_expected_labels
           (counterfactually augmented data; Kaushik et al. 2020) -- NO teacher
    kdabs  pair + per-cell Huber(p_c, T_c): teacher LEVELS, no response structure
    jac    pair + first-order response terms only (Jacobian matching, 1803.00443)
    bsrtd  pair + first-order + mixed partial                      <- CANDIDATE
    null   bsrtd with the teacher response tensors permuted across lattices within seed
           hard label (coordinate-permutation null)

GATES (all train/val only, all before the single test submission)
    G0  asset + encoder responsivity + teacher informativeness beyond the gold pattern
    G1  teacher qualification exam, stratified by cell type (>= 0.90 each)
    G2  disagreement-cell filter (whole-lattice drop) + post-filter floors
Any gate failure => HALT (no test run, no verdict).  HALT is not KILL.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from r4_harness import Head, load_split, train_head  # noqa: E402
from r4_pilot1_mdl import macro_f1, pick_threshold  # noqa: E402
import bsrtd_lattice as BL  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ frozen hyper-parameters
SEEDS = [0, 1, 2]
EPOCHS, LR, BS, WARMUP, NPAIR = 30, 1e-4, 64, 5, 2048
LAT_BS = 64                      # lattices per auxiliary step
HUBER_DELTA = 0.1                # both sides live in [0, 1]
LAMBDAS = (0.3, 1.0, 3.0)        # frozen grid, selected on validation macro-F1
NULL_REPS = 20
BOOT_REPS = 10000
BOOT_RNG = 20260810
PRIMARY = [("MHC", "en"), ("MHC_zh", "zh")]          # rule-bearing
SECONDARY = ["HateMM", "ImpliHateVid"]               # exploratory transfer, NOT rule-bearing
AUX_ARMS = ("cad", "kdabs", "jac", "bsrtd")
COMPARATOR_POOL = ("bce", "pair", "cad", "kdabs")
TIE_ORDER = ["kdabs", "cad", "pair", "bce"]          # strongest-looking control wins ties

# thresholds referenced by the verdict block (frozen; mirrored in the prereg)
R1_MEAN_ROC, R1_MIN_ROC = 0.005, -0.005
R2_MIN_POS_CELLS = 5                                  # of 6 (dataset x seed)
R3_MARGIN = 0.005
R4_NULL_MARGIN = 0.005


def log(s):
    print(s, flush=True)


# --------------------------------------------------------------------------- data loading
def cell_cache_path(emb_dir, ds, split, tag):
    return os.path.join(emb_dir, ds, f"{BL.SPLIT2CACHE[split]}_{tag}.pt")


def load_bundle(lang, emb_dir, lattice_root, teacher_path, cell_tag):
    """Assemble everything the gates and the aux losses need for ONE language."""
    ds = BL.LANG2DS[lang]
    teacher = BL.load_teacher_scores(teacher_path)
    out = {"dataset": ds, "engines": set(), "models": set()}
    for split in ("train", "val"):
        rows = [BL.normalise_row(r, split, lang)
                for r in BL.load_lattices(split, lang, root=lattice_root)]
        cache = torch.load(cell_cache_path(emb_dir, ds, split, cell_tag),
                           map_location="cpu", weights_only=False)
        cids = list(cache["ids"][0])
        assert list(cache["cells"]) == list(BL.CELLS), "cell order in cache is not frozen order"
        pos = {c: i for i, c in enumerate(cids)}
        base = load_split(ds, BL.STUDENT_TAG, split)
        bpos = {c: i for i, c in enumerate(base["ids"])}

        keep, T, E = [], [], []
        for r in rows:
            sid = r["seed_id"]
            if sid not in pos or sid not in bpos:
                continue
            ts = []
            miss = False
            for c in BL.CELLS:
                row = teacher.get(BL.teacher_key(lang, split, sid, c))
                if row is None:
                    miss = True
                    break
                ts.append(float(row["score"]))
                out["engines"].add(row.get("engine", "?"))
                out["models"].add(row.get("model", "?"))
            if miss:
                continue
            keep.append(r); T.append(ts); E.append(r["expected"])
        idx = [pos[r["seed_id"]] for r in keep]
        bidx = [bpos[r["seed_id"]] for r in keep]
        out[split] = {
            "rows": keep,
            "txt": cache["text_feats"][idx].float().numpy(),
            "img": base["img"][bidx].float().numpy(),
            "y": np.array([r["seed_label"] for r in keep], dtype=np.float32),
            "T": np.array(T, dtype=np.float64),
            "expected": np.array(E, dtype=np.int64),
            "n_lattice_rows": len(rows),
        }
    out["probe_acc"] = (BL.axis_probe(out["train"]["txt"], out["val"]["txt"])
                        if len(out["train"]["rows"]) and len(out["val"]["rows"]) else 0.0)
    out["engines"] = sorted(out["engines"]); out["models"] = sorted(out["models"])
    return out


def lat_tensors(b, mask=None, device=DEV):
    m = np.ones(len(b["y"]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return {
        "img": torch.tensor(b["img"][m], device=device),
        "txt": torch.tensor(b["txt"][m], device=device),
        "T": torch.tensor(b["T"][m], dtype=torch.float32, device=device),
        "E": torch.tensor(b["expected"][m], device=device),
        "y": torch.tensor(b["y"][m], device=device),
        "n": int(m.sum()),
    }


# ------------------------------------------------------------------------------ training
def _fd(x):
    return BL.finite_differences(x)


def aux_loss(arm, p, logits, T, E):
    """p, logits, T: (b, 4); E: (b, 4) ints (-1 = unspecified).  All frozen forms."""
    hub = lambda a, t: nn.functional.huber_loss(a, t, delta=HUBER_DELTA)  # noqa: E731
    if arm == "cad":
        m = (E >= 0)
        if m.sum() == 0:
            return logits.sum() * 0.0
        return nn.functional.binary_cross_entropy_with_logits(
            logits[m], E[m].float())
    if arm == "kdabs":
        return hub(p, T)
    pA, pB, pAB = _fd(p)
    tA, tB, tAB = _fd(T)
    first = 0.5 * (hub(pA, tA) + hub(pB, tB))
    if arm == "jac":
        return first
    if arm == "bsrtd":
        return 0.5 * (first + hub(pAB, tAB))
    raise ValueError(arm)


def train_arm(tr, va, te, seed, arm, lam=0.0, lat=None):
    """One head.  arm='bce' uses the deployed BCE trainer; everything else is pairwise+aux."""
    if arm == "bce":
        r = train_head(tr, va, te, seed, device=DEV)
        return {"val": r["val_prob"], "test": r["test_prob"], "ep": r["epoch"]}

    torch.manual_seed(seed); np.random.seed(seed)
    n = tr["y"].shape[0]
    Xi, Xt, Y = tr["img"].to(DEV), tr["txt"].to(DEV), tr["y"].to(DEV)
    h = Head(tr["img"].shape[1], tr["txt"].shape[1]).to(DEV)
    opt = torch.optim.AdamW(h.parameters(), lr=LR)
    pos = torch.nonzero(Y > 0.5).squeeze(1); neg = torch.nonzero(Y < 0.5).squeeze(1)
    g = torch.Generator().manual_seed(seed)
    packs = {nm: (s["img"].to(DEV), s["txt"].to(DEV), s["y"].numpy())
             for nm, s in (("val", va), ("test", te))}
    use_aux = arm != "pair" and lat is not None and lat["n"] > 0
    best = (-1.0, None)
    for ep in range(EPOCHS):
        h.train()
        for _ in range(max(1, n // BS)):
            pi = pos[torch.randint(len(pos), (NPAIR,), generator=g)].to(DEV)
            ni = neg[torch.randint(len(neg), (NPAIR,), generator=g)].to(DEV)
            sp = h(Xi[pi], Xt[pi]).squeeze(1); sn = h(Xi[ni], Xt[ni]).squeeze(1)
            loss = nn.functional.softplus(-(sp - sn)).mean() + 0.1 * 0.5 * (
                nn.functional.binary_cross_entropy_with_logits(sp, torch.ones_like(sp))
                + nn.functional.binary_cross_entropy_with_logits(sn, torch.zeros_like(sn)))
            if use_aux:
                b = min(LAT_BS, lat["n"])
                li = torch.randint(lat["n"], (b,), generator=g).to(DEV)
                im = lat["img"][li].unsqueeze(1).expand(-1, 4, -1).reshape(b * 4, -1)
                tx = lat["txt"][li].reshape(b * 4, -1)
                lg = h(im, tx).squeeze(1).view(b, 4)
                loss = loss + lam * aux_loss(arm, torch.sigmoid(lg), lg,
                                             lat["T"][li], lat["E"][li])
            opt.zero_grad(); loss.backward(); opt.step()
        h.eval()
        with torch.no_grad():
            pr = {nm: torch.sigmoid(h(a, b_).squeeze(1)).cpu().numpy()
                  for nm, (a, b_, _) in packs.items()}
        if ep >= WARMUP:
            f = macro_f1(packs["val"][2], pr["val"], pick_threshold(packs["val"][2], pr["val"]))
            if f > best[0]:
                best = (f, {"val": pr["val"], "test": pr["test"], "ep": ep})
    return best[1]


def perm_tensor(lat, rng):
    """Coordinate-permutation null: reattach teacher tensors to other lattices of the same
    seed hard label.  Preserves the marginal distribution of response tensors exactly and
    destroys only the item-specific pairing."""
    y = lat["y"].cpu().numpy()
    idx = np.arange(lat["n"])
    for cls in (0.0, 1.0):
        m = np.nonzero(y == cls)[0]
        if len(m) > 1:
            idx[m] = m[rng.permutation(len(m))]
    out = dict(lat)
    out["T"] = lat["T"][torch.tensor(idx, device=lat["T"].device)]
    return out


# ------------------------------------------------------------------------------- scoring
def score(pv, pt, yv, yt):
    th = pick_threshold(yv, pv)
    return {"val_roc": float(roc_auc_score(yv, pv)),
            "val_macro_f1": float(macro_f1(yv, pv, th)),
            "test_roc": float(roc_auc_score(yt, pt)),
            "test_macro_f1": float(macro_f1(yt, pt, th))}


HOLDOUT = "test"   # set to "val" by --holdout val so a smoke never opens the test cache


def run_dataset(ds, lat_tr, seeds, arms, lambdas, secondary_lambda=None):
    """Train every arm on one dataset.  Returns per-arm per-seed scores + chosen lambdas."""
    tr, va = (load_split(ds, BL.STUDENT_TAG, s) for s in ("train", "val"))
    te = va if HOLDOUT == "val" else load_split(ds, BL.STUDENT_TAG, "test")
    yv, yt = va["y"].numpy(), te["y"].numpy()
    res = {"n": {"train": len(tr["y"]), "val": len(yv), "test": len(yt)},
           "arms": {}, "lambda": {}, "probs": {}}
    for arm in arms:
        if arm in AUX_ARMS:
            grid = [secondary_lambda[arm]] if secondary_lambda else list(lambdas)
            per_lam = {}
            for lam in grid:
                rows = []
                for sd in seeds:
                    t0 = time.time()
                    p = train_arm(tr, va, te, sd, arm, lam, lat_tr)
                    rows.append((sd, p, score(p["val"], p["test"], yv, yt)))
                    log(f"    {ds} {arm} lam={lam} seed={sd} "
                        f"val_roc={rows[-1][2]['val_roc']:.4f} "
                        f"val_f1={rows[-1][2]['val_macro_f1']:.4f} ({time.time()-t0:.0f}s)")
                per_lam[lam] = rows
            best_lam = max(grid, key=lambda L: float(
                np.mean([r[2]["val_macro_f1"] for r in per_lam[L]])))
            res["lambda"][arm] = best_lam
            rows = per_lam[best_lam]
            res["arms"][arm] = {"lambda_val_macro_f1": {
                str(L): float(np.mean([r[2]["val_macro_f1"] for r in per_lam[L]]))
                for L in grid}}
        else:
            rows = []
            for sd in seeds:
                t0 = time.time()
                p = train_arm(tr, va, te, sd, arm, 0.0, None)
                rows.append((sd, p, score(p["val"], p["test"], yv, yt)))
                log(f"    {ds} {arm} seed={sd} val_roc={rows[-1][2]['val_roc']:.4f} "
                    f"val_f1={rows[-1][2]['val_macro_f1']:.4f} ({time.time()-t0:.0f}s)")
            res["arms"][arm] = {}
        res["arms"][arm]["per_seed"] = [{"seed": sd, **sc} for sd, _, sc in rows]
        res["arms"][arm]["mean_val_roc"] = float(np.mean([s["val_roc"] for _, _, s in rows]))
        res["arms"][arm]["mean_val_macro_f1"] = float(
            np.mean([s["val_macro_f1"] for _, _, s in rows]))
        res["probs"][arm] = {sd: np.asarray(p["test"]) for sd, p, _ in rows}
    res["_y_test"] = yt
    res["_y_val"] = yv
    res["_packs"] = (tr, va, te)
    return res


def bootstrap_lcb(per_ds, rng_seed=BOOT_RNG, reps=BOOT_REPS):
    """Paired stratified bootstrap over test items, independently per dataset.

    Statistic: mean over datasets of [mean over seeds of (ROC_bsrtd - ROC_comparator)].
    Returns (lcb95, mean).
    """
    rng = np.random.default_rng(rng_seed)
    pre = []
    for ds, R in per_ds.items():
        y = R["_y_test"]
        comp = R["comparator"]
        pre.append((y, [R["probs"]["bsrtd"][s] for s in sorted(R["probs"]["bsrtd"])],
                    [R["probs"][comp][s] for s in sorted(R["probs"][comp])],
                    np.nonzero(y == 0)[0], np.nonzero(y == 1)[0]))
    stats = np.zeros(reps)
    for r in range(reps):
        vals = []
        for y, pb, pc, i0, i1 in pre:
            idx = np.concatenate([rng.choice(i0, len(i0), replace=True),
                                  rng.choice(i1, len(i1), replace=True)])
            yy = y[idx]
            if yy.min() == yy.max():
                vals.append(0.0); continue
            vals.append(float(np.mean([roc_auc_score(yy, b[idx]) - roc_auc_score(yy, c[idx])
                                       for b, c in zip(pb, pc)])))
        stats[r] = float(np.mean(vals))
    return float(np.percentile(stats, 5)), float(stats.mean())


# ---------------------------------------------------------------------------------- main
def main():
    global EPOCHS, HOLDOUT, NULL_REPS, BOOT_REPS
    ap = argparse.ArgumentParser(description="B-SRTD pilot")
    ap.add_argument("--mode", choices=["primary", "gates", "smoke-planted", "smoke-nosignal"],
                    default="primary")
    ap.add_argument("--emb-dir", default=os.path.join(BL.ROOT, "data", "CLIP_Embedding"))
    ap.add_argument("--lattice-root", default=BL.BSRTD_DIR)
    ap.add_argument("--teacher", default=BL.TEACHER_PATH)
    ap.add_argument("--cell-tag", default=BL.CELL_TAG)
    ap.add_argument("--out", default=os.path.join(BL.ROOT, "idea-stage", "bsrtd_pilot.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--allow-synthetic-teacher", action="store_true",
                    help="rehearsal only; a primary verdict is refused without it")
    ap.add_argument("--skip-secondary", action="store_true")
    ap.add_argument("--holdout", choices=["test", "val"], default="test",
                    help="'val' keeps the test cache closed (smoke / instrument checks)")
    ap.add_argument("--null-reps", type=int, default=NULL_REPS)
    ap.add_argument("--boot-reps", type=int, default=BOOT_REPS)
    ap.add_argument("--epochs", type=int, default=None, help="rehearsal speed knob only")
    a = ap.parse_args()
    if a.epochs:
        EPOCHS = a.epochs
    HOLDOUT = a.holdout
    NULL_REPS, BOOT_REPS = a.null_reps, a.boot_reps
    assert a.mode == "primary" or a.holdout == "val" or a.mode == "gates", \
        "smoke modes must run with --holdout val (the test cache stays closed)"

    if a.emb_dir != os.path.join(BL.ROOT, "data", "CLIP_Embedding"):
        import r4_harness
        r4_harness.feat_path = lambda dataset, model_tag, split, _d=a.emb_dir: os.path.join(
            _d, dataset, f"{r4_harness.SPLIT_FILE[split]}_{model_tag}.pt")

    log("=" * 78)
    log(f"B-SRTD pilot  mode={a.mode}  start {time.strftime('%Y-%m-%dT%H:%M:%S')}  device={DEV}")
    log("Rules: research-wiki/EXP_bsrtd_prereg.md (frozen before this file existed)")
    out = {"pilot": "B-SRTD", "mode": a.mode,
           "freeze": "research-wiki/EXP_bsrtd_prereg.md",
           "seeds": a.seeds, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---------------------------------------------------------------- load + gates
    bundles = {lang: load_bundle(lang, a.emb_dir, a.lattice_root, a.teacher, a.cell_tag)
               for _, lang in PRIMARY}
    engines = sorted({e for b in bundles.values() for e in b["engines"]})
    models = sorted({m for b in bundles.values() for m in b["models"]})
    out["teacher"] = {"engines": engines, "models": models}
    log(f"teacher models={models} engines={engines}")
    synthetic = any(e == "synthetic" for e in engines)
    if synthetic and a.mode == "primary" and not a.allow_synthetic_teacher:
        log("HALT: teacher_scores.jsonl contains synthetic rows; a primary verdict is refused.")
        json.dump({**out, "halt": "synthetic_teacher"}, open(a.out, "w"), indent=2)
        return

    if a.mode in ("smoke-planted", "smoke-nosignal"):
        rng = np.random.default_rng(777)
        for lang, b in bundles.items():
            if a.mode == "smoke-nosignal":
                for split in ("train", "val"):
                    b[split]["T"] = rng.random(b[split]["T"].shape)
                continue
            # planted: a smooth monotone read of the TRAIN class-mean-difference direction
            # in cell-text space.  This is a response structure the student demonstrably CAN
            # learn and which is relevant to the primary task, so a sound null instrument
            # must show real >> permuted here.  Estimated on train labels only.
            tr_ = b["train"]
            w = (tr_["txt"][tr_["y"] > 0.5, 0].mean(0) - tr_["txt"][tr_["y"] < 0.5, 0].mean(0))
            w = w / (np.linalg.norm(w) + 1e-12)
            mu = float((tr_["txt"][:, 0] @ w).mean())
            sd = float((tr_["txt"][:, 0] @ w).std()) + 1e-12
            for split in ("train", "val"):
                z = b[split]["txt"] @ w
                b[split]["T"] = 1.0 / (1.0 + np.exp(-2.5 * (z - mu) / sd))
        log(f"SMOKE {a.mode}: teacher scores replaced in-memory (cache untouched)")

    g0, g0_ok = BL.g0_report(bundles)
    out["G0"] = {"report": g0, "pass": g0_ok}
    log(f"G0 (assets / encoder responsivity / beyond-gold information): {'PASS' if g0_ok else 'FAIL'}")
    log(json.dumps(g0, indent=2, default=float))

    g1 = {}
    g1_ok = True
    for lang, b in bundles.items():
        r, ok = BL.g1_exam(b)
        g1[lang] = r; g1_ok &= ok
        log(f"G1 teacher exam [{lang}]: overall={r['overall_acc']:.4f} "
            f"worst_cell={r['worst_cell_acc']:.4f} -> {'PASS' if ok else 'FAIL'}")
        for c, v in r["by_cell"].items():
            log(f"    {c:<10} n={v['n']:<5} acc={v['acc']:.4f} fp={v['false_pos']:.4f} "
                f"fn={v['false_neg']:.4f} {'ok' if v['ok'] else 'FAIL'}")
    out["G1"] = {"report": g1, "pass": bool(g1_ok)}

    g2, masks = {}, {}
    g2_ok = True
    for lang, b in bundles.items():
        r, m = BL.g2_filter(b)
        g2[lang] = r; masks[lang] = m; g2_ok &= r["ok"]
        log(f"G2 disagreement filter [{lang}]: "
            f"train {r['train']['n_before']}->{r['train']['n_after']} "
            f"(this-lang floor {r['train']['floor_this_lang']}), "
            f"val {r['val']['n_before']}->{r['val']['n_after']} "
            f"(this-lang floor {r['val']['floor_this_lang']}) -> {'PASS' if r['ok'] else 'FAIL'}")
        log(f"    dropped by cell: {r['train']['dropped_by_cell']}")
    pooled = {s: int(sum(g2[l][s]["n_after"] for l in g2)) for s in ("train", "val")}
    g2["_pooled"] = {
        "train": {"n": pooled["train"], "need": BL.G2_MIN_TRAIN,
                  "ok": bool(pooled["train"] >= BL.G2_MIN_TRAIN)},
        "val": {"n": pooled["val"], "need": BL.G2_MIN_VAL,
                "ok": bool(pooled["val"] >= BL.G2_MIN_VAL)}}
    g2_ok = bool(g2_ok and g2["_pooled"]["train"]["ok"] and g2["_pooled"]["val"]["ok"])
    log(f"G2 pooled: train {pooled['train']} (floor {BL.G2_MIN_TRAIN}), "
        f"val {pooled['val']} (floor {BL.G2_MIN_VAL}) -> {'PASS' if g2_ok else 'FAIL'}")
    out["G2"] = {"report": g2, "pass": bool(g2_ok)}
    if a.mode in ("smoke-planted", "smoke-nosignal"):
        # The smokes check the NULL INSTRUMENT, not the teacher; their fabricated scores are
        # not expected to pass G1/G2, so the filter is disabled and every lattice is used.
        for lang, b in bundles.items():
            for split in ("train", "val"):
                masks[lang][split] = np.ones(len(b[split]["y"]), dtype=bool)
        log("SMOKE: G1/G2 reported but not enforced; all lattices enter the aux loss")

    gates_ok = bool(g0_ok and g1_ok and g2_ok)
    out["gates_pass"] = gates_ok
    if a.mode == "gates" or (not gates_ok and a.mode == "primary"):
        out["halt"] = None if gates_ok else "gate_failure"
        json.dump(out, open(a.out, "w"), indent=2, default=float)
        log(f"gates -> {'PASS' if gates_ok else 'HALT (no test run, no verdict)'}")
        if a.mode == "gates":
            return
        return

    # ---------------------------------------------------------------- training
    lat = {lang: lat_tensors(bundles[lang]["train"], masks[lang]["train"])
           for _, lang in PRIMARY}
    arms = ["bce", "pair", "cad", "kdabs", "jac", "bsrtd"]
    per_ds = {}
    for ds, lang in PRIMARY:
        log(f"-- dataset {ds} (lang {lang}), {lat[lang]['n']} lattices in the aux loss")
        per_ds[ds] = run_dataset(ds, lat[lang], a.seeds, arms, LAMBDAS)

    for ds, _ in PRIMARY:
        R = per_ds[ds]
        mv = {k: R["arms"][k]["mean_val_roc"] for k in COMPARATOR_POOL}
        top = max(mv.values())
        R["comparator"] = sorted([k for k in COMPARATOR_POOL if abs(mv[k] - top) < 1e-12],
                                 key=TIE_ORDER.index)[0]
        log(f"  [{ds}] FROZEN COMPARATOR = {R['comparator']}  (mean val ROC {mv})")

    # ---------------------------------------------------------------- null
    ds0, lang0 = PRIMARY[0]
    R0 = per_ds[ds0]
    tr, va, te = R0["_packs"]
    yv, yt = R0["_y_val"], R0["_y_test"]
    base = R0["probs"][R0["comparator"]]
    lam0 = per_ds[ds0]["lambda"]["bsrtd"]
    nd = []
    log(f"PROGRESS phase=null dataset={ds0} lambda={lam0} reps={NULL_REPS}")
    for rep in range(NULL_REPS):
        sd = a.seeds[rep % len(a.seeds)]
        p = train_arm(tr, va, te, sd, "bsrtd", lam0,
                      perm_tensor(lat[lang0], np.random.default_rng(40_000 + rep)))
        nd.append(float(roc_auc_score(yt, p["test"]) - roc_auc_score(yt, base[sd])))
        if (rep + 1) % 5 == 0:
            log(f"  null {rep+1}/{NULL_REPS} running mean {np.mean(nd):+.4f}")
    null95 = float(np.percentile(np.maximum(0.0, nd), 95))
    out["null"] = {"dataset": ds0, "lambda": lam0, "deltas": nd, "Null95": null95}

    # ---------------------------------------------------------------- deltas + verdict
    cells, dr, df, d_cad, d_kd, d_jac = [], {}, {}, {}, {}, {}
    for ds, _ in PRIMARY:
        R = per_ds[ds]
        comp = R["comparator"]
        get = lambda arm, k: [s[k] for s in R["arms"][arm]["per_seed"]]  # noqa: E731
        dr[ds] = float(np.mean(np.array(get("bsrtd", "test_roc")) - np.array(get(comp, "test_roc"))))
        df[ds] = float(np.mean(np.array(get("bsrtd", "test_macro_f1"))
                               - np.array(get(comp, "test_macro_f1"))))
        d_cad[ds] = float(np.mean(np.array(get("bsrtd", "test_roc")) - np.array(get("cad", "test_roc"))))
        d_kd[ds] = float(np.mean(np.array(get("bsrtd", "test_roc")) - np.array(get("kdabs", "test_roc"))))
        d_jac[ds] = float(np.mean(np.array(get("bsrtd", "test_roc")) - np.array(get("jac", "test_roc"))))
        cells += list(np.array(get("bsrtd", "test_roc")) - np.array(get(comp, "test_roc")))

    lcb, bmean = bootstrap_lcb({d: per_ds[d] for d, _ in PRIMARY}, reps=BOOT_REPS)
    mean_roc = float(np.mean([dr[d] for d, _ in PRIMARY]))
    mean_f1 = float(np.mean([df[d] for d, _ in PRIMARY]))
    r1 = (mean_roc >= R1_MEAN_ROC and min(dr.values()) >= R1_MIN_ROC and mean_f1 >= 0.0)
    r2 = int(sum(1 for c in cells if c > 0)) >= R2_MIN_POS_CELLS
    r3a = (float(np.mean(list(d_cad.values()))) >= R3_MARGIN and min(d_cad.values()) > 0)
    r3b = (float(np.mean(list(d_kd.values()))) >= R3_MARGIN and min(d_kd.values()) > 0)
    r3 = bool(r3a and r3b)
    r4 = (mean_roc >= null95 + R4_NULL_MARGIN) and (lcb > 0)
    out["verdict"] = {
        "DeltaROC_by_dataset": dr, "DeltaF1_by_dataset": df,
        "MeanDeltaROC": mean_roc, "MeanDeltaF1": mean_f1,
        "per_seed_cells": [float(c) for c in cells],
        "positive_cells": int(sum(1 for c in cells if c > 0)), "n_cells": len(cells),
        "Delta_vs_CAD": d_cad, "Delta_vs_KDABS": d_kd, "Delta_vs_JAC": d_jac,
        "Null95": null95, "boot_LCB95": lcb, "boot_mean": bmean,
        "R1": bool(r1), "R2": bool(r2), "R3a_vs_CAD": bool(r3a), "R3b_vs_KDABS": bool(r3b),
        "R3": r3, "R4": bool(r4),
        "GO": bool(r1 and r2 and r3 and r4)}

    # ---------------------------------------------------------------- null-instrument smoke
    # D1 lesson (idea-stage/R4_DEVIATION_D1_2026-08-10.md): a permutation null must be
    # checked against a planted-signal control AND a no-signal control before it gates
    # anything.  Frozen pass criteria, evaluated on the val holdout only.
    if a.mode in ("smoke-planted", "smoke-nosignal"):
        d_pair = float(np.mean([
            np.mean(np.array([s["test_roc"] for s in per_ds[d]["arms"]["bsrtd"]["per_seed"]])
                    - np.array([s["test_roc"] for s in per_ds[d]["arms"]["pair"]["per_seed"]]))
            for d, _ in PRIMARY]))
        null_mean = float(np.mean(nd))
        if a.mode == "smoke-planted":
            ok = bool(d_pair - null_mean >= 0.005 and d_pair > 0)
            crit = "Delta(bsrtd-pair) - mean(null) >= +0.005 AND Delta(bsrtd-pair) > 0"
        else:
            ok = bool(abs(d_pair) <= 0.01 and (null_mean - d_pair) <= 0.01)
            crit = "|Delta(bsrtd-pair)| <= 0.01 AND mean(null) - Delta(bsrtd-pair) <= 0.01"
        out["smoke"] = {"criterion": crit, "delta_bsrtd_minus_pair": d_pair,
                        "null_mean": null_mean, "Null95": null95, "pass": ok}
        log("=" * 78)
        log(f"SMOKE {a.mode}: Delta(bsrtd-pair)={d_pair:+.4f} mean(null)={null_mean:+.4f} "
            f"Null95={null95:.4f}")
        log(f"  criterion: {crit}  ->  {'PASS' if ok else 'FAIL'}")
        json.dump(out, open(a.out, "w"), indent=2, default=float)
        return

    # ---------------------------------------------------------------- secondary transfer
    if not a.skip_secondary:
        out["secondary"] = {}
        for ds in SECONDARY:
            log(f"-- SECONDARY (exploratory, not rule-bearing) {ds}")
            fixed = {arm: per_ds["MHC"]["lambda"].get(arm, 1.0) for arm in AUX_ARMS}
            r = run_dataset(ds, lat["en"], a.seeds, ["pair", "cad", "kdabs", "bsrtd"],
                            LAMBDAS, secondary_lambda=fixed)
            b_ = np.array([s["test_roc"] for s in r["arms"]["bsrtd"]["per_seed"]])
            out["secondary"][ds] = {
                "lambda": fixed, "arms": r["arms"],
                "Delta_vs_pair": float(np.mean(
                    b_ - np.array([s["test_roc"] for s in r["arms"]["pair"]["per_seed"]]))),
                "Delta_vs_cad": float(np.mean(
                    b_ - np.array([s["test_roc"] for s in r["arms"]["cad"]["per_seed"]]))),
                "Delta_vs_kdabs": float(np.mean(
                    b_ - np.array([s["test_roc"] for s in r["arms"]["kdabs"]["per_seed"]])))}
            log(f"  [{ds}] (exploratory) bsrtd - pair = "
                f"{out['secondary'][ds]['Delta_vs_pair']:+.4f}, "
                f"- cad = {out['secondary'][ds]['Delta_vs_cad']:+.4f}, "
                f"- kdabs = {out['secondary'][ds]['Delta_vs_kdabs']:+.4f}")

    for ds, _ in PRIMARY:
        R = per_ds[ds]
        out.setdefault("cells", {})[ds] = {
            "comparator": R["comparator"], "lambda": R["lambda"], "n": R["n"],
            "arms": {k: {kk: vv for kk, vv in v.items()} for k, v in R["arms"].items()}}
    json.dump(out, open(a.out, "w"), indent=2, default=float)

    # ---------------------------------------------------------------- report
    log("=" * 78)
    log("FULL TEST TABLE (withheld until now)")
    for ds, _ in PRIMARY:
        R = per_ds[ds]
        log(f"-- {ds}, frozen comparator = {R['comparator']}, lambda = {R['lambda']}")
        for arm in R["arms"]:
            rr = [s["test_roc"] for s in R["arms"][arm]["per_seed"]]
            ff = [s["test_macro_f1"] for s in R["arms"][arm]["per_seed"]]
            log(f"     {arm:<7} test ROC {np.mean(rr):.4f}+/-{np.std(rr):.4f}   "
                f"macroF1 {np.mean(ff):.4f}+/-{np.std(ff):.4f}")
        log(f"     => DeltaROC={dr[ds]:+.4f}  DeltaF1={df[ds]:+.4f}  "
            f"vsCAD={d_cad[ds]:+.4f}  vsKDABS={d_kd[ds]:+.4f}  vsJAC={d_jac[ds]:+.4f}")
    v = out["verdict"]
    log(f"MeanDeltaROC={mean_roc:+.4f} MeanDeltaF1={mean_f1:+.4f} "
        f"Null95={null95:.4f} bootLCB95={lcb:+.4f} positive_cells={v['positive_cells']}/{v['n_cells']}")
    log(f"R1={r1} R2={r2} R3a(vs CAD)={r3a} R3b(vs KD-ABS)={r3b} R4={r4}")
    log(f"VERDICT: {'GO' if v['GO'] else 'KILL'}")
    log(f"raw -> {a.out}")


if __name__ == "__main__":
    main()
