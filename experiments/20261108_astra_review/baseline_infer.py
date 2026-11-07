"""Review control E1, step 1: per-second scores of the frozen strongest baselines on train / validation / test.

The learned fusion control (e1_fusion.py) needs the baseline's scores on the training and validation videos, which
the official final runs did not write for Fed-WSVAD (HateClipSeg, DeHate) and DSANet (HateClipSeg). This script loads
each official checkpoint unchanged (model.pth + its train_meta.json / frozen_config.json, the same adapters that
trained them) and scores the requested videos of the fixed cohort (hc.load_fixed_cohort). Fed-WSVAD: score_align of
fed_wsvad_adapter.score_ids; DSANet: score_mlp (sigmoid(logits1), the branch of the official table,
scripts/reproduction_baselines/aggregate_official_val.py; note experiments/20261101_r5_completion/baseline_plus_vlm.py
read score_align for DSANet, REVIEW_RULE6_R1.md F2). Check: the test scores are recomputed and compared with the
official scores.jsonl (--ref); the largest absolute difference is written to check.json.

Writes <out-root>/<method>/<corpus>/seed_<s>/<split>.jsonl ({"video_id", "score"}) and check.json.

--hidden (review round 2, E1b): instead writes <split>_hidden.npz (video id -> per-second hidden features, float16)
next to the scores: the input of each baseline's final per-second classification layer, from the same forward pass
that gives the score: Fed-WSVAD the temporal encoder output (encode_video, compared with the text prompts);
DSANet the input of `classifier` (visual_features + mlp2(visual_features)); MACIL-SD the audio and visual
cross-attention outputs [a_out, v_out] (Att_MMIL.fc reads each), mean over the five crops, mapped to seconds by the
test dataset's index map. Check: the score recomputed in that pass against the <split>.jsonl written before (largest
absolute difference, hidden_check.json).
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
import hier_evidence_common as hc                                       # noqa: E402
from hate_common import data as hdata, runtime                          # noqa: E402


def fed_wsvad_scorer(model_dir, corpus, device):
    # hier_evidence_common puts src/ first on sys.path, so `utils` would resolve to src/utils; bind the name to the
    # pinned upstream package (a namespace package, no __init__.py) before the adapter imports utils.model
    up = ROOT / "third_party" / "Fed-WSVAD"
    up = up if up.is_dir() else ROOT / "third_party" / "fed_wsvad"
    for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[k]
    pkg = types.ModuleType("utils")
    pkg.__path__ = [str(up / "utils")]
    sys.modules["utils"] = pkg
    import fed_wsvad_adapter as fa
    meta = json.loads((model_dir / "train_meta.json").read_text())
    args, vl, aw = meta["args"], int(meta["visual_length"]), int(meta["attn_window"])
    model = fa.Model(512, vl, args["prompt_prefix"], args["prompt_postfix"], 512, args["visual_layers"], 1, aw,
                     str(device)).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location=device))
    model.eval()

    feats = []
    # encode_video returns temporal(x) permuted to (blocks, visual_length, width); fa.score_ids makes one forward per
    # video, in the order of ids
    model.temporal.register_forward_hook(lambda _m, _i, out: feats.append(out.permute(1, 0, 2).reshape(-1, out.shape[2])))

    def score(ids):
        feats.clear()
        sc = fa.score_ids(model, corpus, ids, vl, device)[0]
        assert len(feats) == len(ids), (len(feats), len(ids))
        return sc, {v: f[:len(sc[v])].float().cpu().numpy() for v, f in zip(ids, feats)}
    return score


def dsanet_scorer(model_dir, corpus, device):
    from dsanet import option
    from dsanet.model import DSANet
    p = json.loads((model_dir / "frozen_config.json").read_text())["params"]
    argv = ["--corpus", corpus, "--device", str(device), "--visual-length", str(p["visual_length"]),
            "--attn-window", str(p["attn_window"]), "--num_prototypes", str(p["num_prototypes"]),
            "--normal_selection_ratio", str(p["normal_selection_ratio"]), "--t_w", str(p["t_w"]),
            "--temp", str(p["temp"]), "--loss2_weight", str(p["loss2_weight"])]
    args = option.resolve(option.build_parser().parse_args(argv))
    model = DSANet(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
                   args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args, device,
                   clip_download_root=args.clip_download_root)
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location=device))
    model.to(device)
    model.eval()
    labels = hdata.load_labels(corpus)
    cin = []
    model.classifier.register_forward_hook(lambda _m, inp, _o: cin.append(inp[0]))

    def score(ids):
        out, hid = {}, {}
        ds = hdata.HateVideoDataset(corpus, ids, args.visual_length, True, labels)
        with torch.no_grad():
            for i in range(len(ds)):
                visual, _label, length, vid = ds[i]
                n = int(length)
                visual, lengths = runtime.prepare_test_item(torch.as_tensor(visual)[None], n, args.visual_length)
                cin.clear()
                o = model(visual.to(device), None, list(hdata.PROMPT_TEXT), lengths.to(device), bool(args.DNP_use))
                logits1 = o[1].reshape(-1, o[1].shape[2])
                out[vid] = torch.sigmoid(logits1[0:n].squeeze(-1)).float().cpu().numpy()
                hid[vid] = cin[0].reshape(-1, cin[0].shape[-1])[0:n].float().cpu().numpy()
        return out, hid
    return score


def macilsd_scorer(model_dir, corpus, device):
    from macilsd.avce_network import AVCE_Model
    from macilsd.dataset import MacilTestDataset
    margs = argparse.Namespace(**json.loads((model_dir / "train_meta.json").read_text())["args"])
    assert margs.modality == "av" and margs.grid == "snippet", (margs.modality, margs.grid)
    model = AVCE_Model(margs)
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location=device))
    model.to(device)
    model.eval()

    def score(ids):
        out, hid = {}, {}
        ds = MacilTestDataset(corpus, ids, margs.max_seqlen, margs.grid, margs.modality)
        with torch.no_grad():
            for i in range(len(ds)):
                f_v, f_a, index_map, _n, vid = ds[i]
                # macilsd/infer.py: av = mean over crops of sigmoid(av_logits), rows -> seconds by index_map; the
                # model returns (..., v_out, a_out) in that order (avce_network.py)
                _mm, _al, _vl, av_logits, v_out, a_out = model(f_a.to(device), f_v.to(device), seq_len=None)
                im = index_map.numpy()
                out[vid] = torch.sigmoid(av_logits.squeeze(-1)).mean(0).cpu().numpy().astype(np.float64)[im]
                hid[vid] = torch.cat([a_out, v_out], -1).mean(0).cpu().numpy()[im]
        return out, hid
    return score


def have_features(method, corpus, v):
    if method == "macilsd":
        from macilsd import align
        return align.has_features(corpus, v)
    return Path(hdata.feature_path(corpus, v)).exists()


def write_hidden(a, ids, od, score):
    check = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
             "model_dir": a.model_dir, "splits": {}}
    for split in a.splits:
        prev = {}
        for line in open(od / f"{split}.jsonl"):
            r = json.loads(line)
            prev[r["video_id"]] = np.asarray(r["score"], dtype=np.float64)
        have = [v for v in ids[split] if have_features(a.method, a.corpus, v)]
        sc, hid = score(have)
        np.savez(od / f"{split}_hidden.npz", **{v: hid[v].astype(np.float16) for v in have})
        diffs = [float(np.max(np.abs(sc[v][:len(prev[v])] - prev[v][:len(sc[v])]))) for v in have if v in prev]
        check["splits"][split] = {"videos": len(ids[split]), "scored": len(have), "dim": int(hid[have[0]].shape[1]),
                                  "rows_equal_scores": all(len(hid[v]) == len(sc[v]) for v in have),
                                  "score_vs_written_max_abs_diff": max(diffs) if diffs else None,
                                  "compared": len(diffs)}
        print(datetime.now().astimezone().isoformat(), a.method, a.corpus, a.seed, split, check["splits"][split],
              flush=True)
    (od / "hidden_check.json").write_text(json.dumps(check, indent=1) + "\n")
    print("DONE", flush=True)


SCORERS = {"fed_wsvad_3client": fed_wsvad_scorer, "dsanet": dsanet_scorer, "macilsd": macilsd_scorer}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=tuple(SCORERS), required=True)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ref", help="the official test scores.jsonl of this checkpoint (score mode)")
    ap.add_argument("--hidden", action="store_true", help="write the hidden features (E1b) instead of the scores")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-root", default="runs/20261108_astra_review/baseline_scores")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    _, ids, _, _ = hc.load_fixed_cohort(a.corpus)
    od = (ROOT / a.out_root / a.method / a.corpus / f"seed_{a.seed}").resolve()
    od.mkdir(parents=True, exist_ok=True)
    if (od / ("hidden_check.json" if a.hidden else "check.json")).exists():
        print("skip (done)", od, flush=True)
        return
    score = SCORERS[a.method](Path(a.model_dir).resolve(), a.corpus, device)
    if a.hidden:
        write_hidden(a, ids, od, score)
        return
    assert a.ref, "--ref is required when writing scores"
    check = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
             "model_dir": a.model_dir, "ref": a.ref, "splits": {}}
    for split in a.splits:
        have = [v for v in ids[split] if Path(hdata.feature_path(a.corpus, v)).exists()]
        sc = score(have)[0]
        with (od / f"{split}.jsonl").open("w") as fh:
            for v in have:
                fh.write(json.dumps({"video_id": v, "score": [round(float(x), 6) for x in sc[v]]}) + "\n")
        check["splits"][split] = {"videos": len(ids[split]), "scored": len(have)}
        if split == "test":
            ref = {}
            for line in open(a.ref):
                r = json.loads(line)
                ref[r["video_id"]] = np.asarray(r["score_mlp"] if a.method == "dsanet" else r["score_align"],
                                                dtype=np.float64)
            diffs = [float(np.max(np.abs(sc[v][:len(ref[v])] - ref[v][:len(sc[v])]))) for v in have if v in ref]
            check["test_vs_official_max_abs_diff"] = max(diffs) if diffs else None
            check["test_compared"] = len(diffs)
        print(datetime.now().astimezone().isoformat(), a.method, a.corpus, a.seed, split, check["splits"][split],
              check.get("test_vs_official_max_abs_diff", ""), flush=True)
    (od / "check.json").write_text(json.dumps(check, indent=1) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
