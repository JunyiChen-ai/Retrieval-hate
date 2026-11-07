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
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
import hier_evidence_common as hc                                       # noqa: E402
from hate_common import data as hdata, runtime                          # noqa: E402


def fed_wsvad_scorer(model_dir, corpus, device):
    import fed_wsvad_adapter as fa
    meta = json.loads((model_dir / "train_meta.json").read_text())
    args, vl, aw = meta["args"], int(meta["visual_length"]), int(meta["attn_window"])
    model = fa.Model(512, vl, args["prompt_prefix"], args["prompt_postfix"], 512, args["visual_layers"], 1, aw,
                     str(device)).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location=device))
    model.eval()

    def score(ids):
        return fa.score_ids(model, corpus, ids, vl, device)[0]
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

    def score(ids):
        out = {}
        ds = hdata.HateVideoDataset(corpus, ids, args.visual_length, True, labels)
        with torch.no_grad():
            for i in range(len(ds)):
                visual, _label, length, vid = ds[i]
                n = int(length)
                visual, lengths = runtime.prepare_test_item(torch.as_tensor(visual)[None], n, args.visual_length)
                o = model(visual.to(device), None, list(hdata.PROMPT_TEXT), lengths.to(device), bool(args.DNP_use))
                logits1 = o[1].reshape(-1, o[1].shape[2])
                out[vid] = torch.sigmoid(logits1[0:n].squeeze(-1)).float().cpu().numpy()
        return out
    return score


SCORERS = {"fed_wsvad_3client": fed_wsvad_scorer, "dsanet": dsanet_scorer}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=tuple(SCORERS), required=True)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ref", required=True, help="the official test scores.jsonl of this checkpoint")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-root", default="runs/20261108_astra_review/baseline_scores")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    _, ids, _, _ = hc.load_fixed_cohort(a.corpus)
    od = (ROOT / a.out_root / a.method / a.corpus / f"seed_{a.seed}").resolve()
    od.mkdir(parents=True, exist_ok=True)
    score = SCORERS[a.method](Path(a.model_dir).resolve(), a.corpus, device)
    check = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
             "model_dir": a.model_dir, "ref": a.ref, "splits": {}}
    for split in a.splits:
        have = [v for v in ids[split] if Path(hdata.feature_path(a.corpus, v)).exists()]
        sc = score(have)
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
