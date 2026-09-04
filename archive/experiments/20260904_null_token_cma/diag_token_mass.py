"""Diagnostic (no training): how much attention a trained model puts on the
null token at test time, per cross-modal direction, averaged over valid
queries of every test video (single full-sequence pass, no crops).

    python experiments/20260904_null_token_cma/diag_token_mass.py \
        --setting c4  --corpus hateclipseg --run runs/20260904_null_token_cma/hateclipseg/seed234/trial15
    python experiments/20260904_null_token_cma/diag_token_mass.py \
        --setting rev1 --corpus hateclipseg --run runs/20260904_null_token_cma/rev1/hateclipseg/seed234/trial8

`--run` holds config.json (corpus, seed, ablation, hparams) and model.pth.
c4   = this candidate's own training (model.py NTCA; attention in cma.attn.attn)
rev1 = candidate 1's training with src/null_token_cma.NullTokenKeys on the
       MACIL-SD layer (state keys prefixed `av.`; attention in
       cma.layer.self_attn.attn). The rev1 input preprocessing (bookkeeping
       columns zeroed, ell scaled, context c) is repeated here in four lines
       so this file does not import candidate 1's experiment code.
Writes <run>/diag_token_mass.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "duplex"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, HERE)

from hate_common import data as hdata                  # noqa: E402
from macilsd import align                              # noqa: E402
from macilsd.avce_network import AVCE_Model            # noqa: E402
import hier_evidence_common as hc                      # noqa: E402
import vlm_verdict                                     # noqa: E402
import verdict_hmm                                     # noqa: E402
from null_token_cma import NullTokenKeys               # noqa: E402
from model import NTCA, STRUCT_ARMS, N_EVID            # noqa: E402

K_FINE, J_COARSE = vlm_verdict.GRANULARITIES


class Args(dict):
    __getattr__ = dict.__getitem__


def test_loader(corpus, ablation, w_fine):
    labels = hdata.load_labels(corpus)
    train_ids = hc.usable(corpus, hdata.load_split(corpus, "train"))
    test_gt = hdata.gt_arrays(corpus, "test")
    test_ids = [v for v in hc.usable(corpus, hdata.load_split(corpus, "test")) if v in test_gt]
    V = {k: vlm_verdict.load_verdicts(corpus, k=k, tag="qwen") for k in (K_FINE, J_COARSE)}
    binary = {v: (verdict_hmm.binarize(V[K_FINE][v]), verdict_hmm.binarize(V[J_COARSE][v]))
              for v in V[K_FINE] if v in V[J_COARSE]}
    hmm, _, _ = hc.fit_hmm(corpus, train_ids, labels, binary)
    cache = hc.ScaffoldCache(corpus, test_ids, hc.make_scaffold_fn(hmm, binary, ablation, w_fine))
    return DataLoader(hc.EvalDataset(corpus, test_ids, cache), batch_size=1, shuffle=False, num_workers=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=("c4", "rev1"))
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    a = Args(cfg["hparams"])
    ablation = cfg["ablation"]
    device = torch.device(args.device)
    state = torch.load(os.path.join(args.run, "model.pth"), map_location=device)
    loader = test_loader(args.corpus, ablation, a.w_fine)

    if args.setting == "c4":
        arm = ablation if ablation in STRUCT_ARMS else "full"
        model = NTCA(a, a.prior_scale, arm=arm).to(device)
        model.load_state_dict(state)
        has_token = model.cma.base is not None
        get_attn = lambda: model.cma.attn.attn                    # noqa: E731
    else:
        a["a_feature_size"] = hc.A_EXT_DIM
        a["v_feature_size"] = align.V_DIM
        av = AVCE_Model(a).to(device)
        token = {"null_token": "evidence", "null_token_const": "const",
                 "masked_no_token": "none"}.get(ablation)
        if token is not None:
            av.cma = NullTokenKeys(av.cma.layer, a.hid_dim, token=token).to(device)
        av.load_state_dict({k[3:]: v for k, v in state.items() if k.startswith("av.")})
        model = av
        has_token = token in ("evidence", "const")
        get_attn = lambda: av.cma.layer.self_attn.attn            # noqa: E731

    model.eval()
    rows = []
    with torch.no_grad():
        for f_v, f_a, index_map, n_seconds, vid in loader:
            f_v = f_v[0][:1].to(device)          # crop 0 only, full sequence
            f_a = f_a[0][:1].to(device)
            T = f_a.shape[1]
            if args.setting == "c4":
                _ = model(f_a, f_v, seq_len=None)
                # NTCA.cma.forward runs the visual-query direction first, then audio;
                # the stored attention is from the LAST call (audio queries, visual keys)
                a_dir = get_attn()
                # rerun the first direction alone to read its attention
                mask = torch.ones(1, T, dtype=torch.bool, device=device)
                evid = f_a[..., hc.SCAF_OFFSET:hc.SCAF_OFFSET + N_EVID].clone()
                evid[..., hc.COL_ELL] = evid[..., hc.COL_ELL] / hc.ELL_SCALE
                c = evid.mean(1)
                h_a = model.fc_a(torch.cat([f_a[..., :hc.SCAF_OFFSET], evid], dim=-1))
                h_v = model.fc_v(f_v)
                model.cma.one(h_v, h_a, 1, c, mask)
                v_dir = get_attn()
            else:
                f_a_in = f_a.clone()
                f_a_in[..., hc.SCAF_OFFSET + hc.N_INPUT_SCAF:] = 0.0
                f_a_in[..., hc.SCAF_OFFSET + hc.COL_ELL] /= hc.ELL_SCALE
                if token is not None:
                    mask = torch.ones(1, T, dtype=torch.bool, device=device)
                    c = f_a_in[..., hc.SCAF_OFFSET:hc.SCAF_OFFSET + hc.N_INPUT_SCAF].mean(1)
                    av.cma.set_context(c, mask)
                _ = av(f_a_in, f_v, None)
                a_dir = get_attn()                                    # last call: audio queries
                if token is not None:
                    av.cma.set_context(c, mask)
                h_v, h_a = av.fc_v(f_v), av.fc_a(f_a_in)
                if token is not None:
                    n = av.cma.null_token(1)
                    km = torch.cat([torch.ones(1, 1, dtype=torch.bool, device=device), mask], dim=1)
                    av.cma._one(h_v, h_a, n, km)
                    av.cma.context, av.cma.mask = None, None
                else:
                    av.cma.layer(h_v, h_a, h_a)
                v_dir = get_attn()
            def mass(p):
                if not has_token:
                    return float("nan"), float("nan")
                tok = p[0, :, :, 0]                         # (heads, Tq)
                return float(tok.mean()), float((tok.mean(0) > 0.5).float().mean())
            (vm, vf), (am, af) = mass(v_dir), mass(a_dir)
            rows.append({"vid": vid[0], "T": T, "vq_token_mass": vm, "vq_frac_over_half": vf,
                         "aq_token_mass": am, "aq_frac_over_half": af})
    out = {"setting": args.setting, "corpus": args.corpus, "run": args.run, "ablation": ablation,
           "n_videos": len(rows),
           "mean_vq_token_mass": float(np.nanmean([r["vq_token_mass"] for r in rows])),
           "mean_aq_token_mass": float(np.nanmean([r["aq_token_mass"] for r in rows])),
           "mean_vq_frac_over_half": float(np.nanmean([r["vq_frac_over_half"] for r in rows])),
           "mean_aq_frac_over_half": float(np.nanmean([r["aq_frac_over_half"] for r in rows])),
           "per_video": rows}
    with open(os.path.join(args.run, "diag_token_mass.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_video"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
