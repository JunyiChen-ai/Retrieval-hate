"""README section 9 analyses 1 and 3 (revision 4 best trials): top-k set overlap between
backbone logit and combined logit on train hateful videos, and trained prior weights vs
init. Run from repo root:
    python experiments/20260902_verdict_boundary_contrast_mil/analysis_topk_prior.py <corpus> <trial_dir>
Output is printed; runs/<exp>/analysis/topk_prior_<corpus>_<seed>.txt keeps the record."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, "experiments/20260902_verdict_boundary_contrast_mil")
sys.path.insert(0, "scripts/reproduction_baselines"); sys.path.insert(0, "src")
import train as T, dataset as ds, vlm_verdict
from hate_common import data as hdata
from macilsd import align
corpus, tdir = sys.argv[1], sys.argv[2]
cfg = dict(T.DEFAULTS); cfg.update(json.load(open(os.path.join(tdir, "hparams.json"))))
a = T.Args(cfg); a["a_feature_size"] = ds.A_EXT_DIM; a["v_feature_size"] = align.V_DIM
model = T.Candidate(a); model.load_state_dict(torch.load(os.path.join(tdir, "model.pth"), map_location="cpu")); model.eval()
w = model.prior.weight.detach().numpy()[0]; b = float(model.prior.bias)
init = np.zeros_like(w); 
for c in T.LEVEL_COLS: init[c] = cfg["prior_scale"] / 2
print("prior_dims", cfg["prior_dims"], "prior_scale %.3f" % cfg["prior_scale"])
print("prior weight trained:", np.round(w, 3).tolist(), "bias %.3f (init %.3f)" % (b, -cfg["prior_scale"] / 2))
print("prior weight init:   ", np.round(init, 3).tolist())
labels = hdata.load_labels(corpus)
ids = [v for v in T.usable(corpus, hdata.load_split(corpus, "train")) if labels[v] == 1]
try: gt = hdata.gt_arrays(corpus, "train")
except Exception as e: gt = {}; print("no train gt:", e)
verdicts = [vlm_verdict.load_verdicts(corpus, k=k) for k in vlm_verdict.GRANULARITIES]
cache = ds.ScaffoldCache(corpus, ids, verdicts)
jac, pos_in, pos_out, pos_all, n_sw = [], [], [], [], 0
with torch.no_grad():
    for vid in ids:
        f_a, n_sec, snip = cache[vid]
        f_v = align.aligned_visual_crop(corpus, vid, 0, "snippet", n_sec, snip)
        f_v = torch.from_numpy(np.ascontiguousarray(f_v, dtype=np.float32))[None]
        f_at = torch.from_numpy(np.ascontiguousarray(f_a, dtype=np.float32))[None]
        t = f_v.shape[1]; seq = torch.tensor([t])
        _, _, _, av_log, *_ = model(f_at, f_v, seq)
        comb = av_log[0, :, 0].numpy(); cont = model.last_content_logit[0, :, 0].numpy()
        k = max(1, int(-(-t // 16)))
        s_c = set(np.argsort(-cont)[:k]); s_m = set(np.argsort(-comb)[:k])
        jac.append(len(s_c & s_m) / len(s_c | s_m))
        if vid in gt:
            idx = align.snippet_index_for_seconds(snip, n_sec)  # second -> snippet
            g = gt[vid][:len(idx)]; gs = np.zeros(t); cnt = np.zeros(t)
            for sec, si in enumerate(idx):
                if si < t: gs[si] += g[sec]; cnt[si] += 1
            gs = gs / np.maximum(cnt, 1)
            pos_all.append(gs.mean())
            if s_m - s_c:
                pos_in.append(np.mean([gs[i] for i in s_m - s_c])); pos_out.append(np.mean([gs[i] for i in s_c - s_m])); n_sw += 1
print("%s %s: %d hateful train videos; top-k Jaccard(backbone vs combined) mean %.3f median %.3f; videos with any swap %d"
      % (corpus, tdir, len(ids), np.mean(jac), np.median(jac), n_sw))
if pos_all:
    print("GT positive rate: all rows %.3f | swapped-in by prior %.3f | swapped-out %.3f (n=%d videos with GT)" % (np.mean(pos_all), np.mean(pos_in), np.mean(pos_out), len(pos_all)))
