#!/usr/bin/env python
"""REPRO campaign Wave 2 — T3AL (Liberatori et al., CVPR 2024), test-time adaptation
for zero-shot temporal action localisation, on the four hate corpora.

Supervision class: **label-free**.  T3AL adapts a frozen CoCa checkpoint on each
*unlabelled* test video, which is the method's own published mechanism.  No target
label of any split is read at any point: `get_segments_gt` below is overridden to
return nothing, so the one place upstream touches annotations is severed.

What is upstream and unchanged
------------------------------
`T3ALNet.forward` — pseudo-label inference, the BYOL-style test-time adaptation
loop, `select_segments`, the moving average, the caption-refinement filter and the
final segment classification — is imported from `third_party/T3AL` and executed
verbatim.  This driver supplies the corpus loop, the features, the class list, the
per-video seeding and the campaign's output format, and overrides exactly four
hooks (`__init__`, `get_video_fps`, `get_segments_gt`, `plot_visualize`).

Outputs (freeze §14 / the campaign's curve interface)
  idea-stage/repro_t3al/curves[_s<SEED>]/<DS>/<vid>.npz   arrays keyed by variant,
                                                          plus scalar `rate`
  idea-stage/repro_t3al/curves[_s<SEED>]/<DS>_intervals_<variant>.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
T3AL_DIR = ROOT / "third_party/T3AL"
FEAT_DIR = lambda ds: ROOT / f"data/CLIP_Embedding/{ds}/coca_vitL14_4fps"
CAP_DIR = lambda ds: ROOT / "idea-stage/repro_t3al/captions" / ds
WORK_DIR = ROOT / "idea-stage/repro_t3al/work"

FPS = 4.0  # our feature rate; T3AL's own is one vector per native video frame

# --------------------------------------------------------------------------- #
# FROZEN CLASS LIST — written and committed before any T3AL number was computed.
#
# T3AL takes a closed list of bare class names and builds its own prompt from it
# ("a video of action" + " " + name), so the campaign's six hate categories are
# given here as bare names.  The wording is the wording of `HCS_CLASS_QUERIES` in
# scripts/repro_campaign/run_unitime.py, with that file's fixed sentence frame
# "the moment containing ..." removed, and the list is in the freeze §4
# HateClipSeg class order (0 normal .. 5 harm).
#
#   run_unitime.HCS_CLASS_QUERIES              -> T3AL class name
#   "the moment containing normal, non-toxic content" -> "normal, non-toxic content"
#   "the moment containing hateful content"           -> "hateful content"
#   "the moment containing insulting content"         -> "insulting content"
#   "the moment containing sexual content"            -> "sexual content"
#   "the moment containing violent content"           -> "violent content"
#   "the moment containing harmful content"           -> "harmful content"
CLASS_NAMES = [
    "normal, non-toxic content",
    "hateful content",
    "insulting content",
    "sexual content",
    "violent content",
    "harmful content",
]
CLASS_KEYS = ["c0_normal", "c1_hateful", "c2_insulting", "c3_sexual",
              "c4_violence", "c5_harm"]

# The Wave 0/1 main query, same source file, reduced the same way.  Used as the
# adaptation target of the `mainq_sim` variant so that T3AL answers the exact
# question Qwen2.5-VL and UniTime were asked.
#   run_unitime.MAIN_QUERY
#   "the moment containing hateful content targeting a person or group"
MAIN_LABEL = "hateful content targeting a person or group"
# --------------------------------------------------------------------------- #

# The four knob presets the val sweep chooses between.  All values other than
# `steps`, `kernel_size`, `stride` and the four preset switches are upstream's and
# are never varied: lr 1e-5, weight_decay 1e-4, ltype BYOLfeat, randper 10,
# text_projection/image_projection/logit_scale on, text_encoder off,
# refine_with_captions on, topk 3, m 0.7.
PRESETS = {
    # upstream configs/model/tt_thumos.yaml, verbatim
    "A_thumos": dict(steps=60, kernel_size=20, stride=20, normalize=True,
                     remove_background=True, p=0.75, n=4, style="thumos"),
    # same, with the two index-unit knobs rescaled from THUMOS' 30 fps native
    # feature rate to our 4 fps grid (20 frames at 30 fps = 0.67 s = 3 at 4 fps)
    "B_thumos_rescaled": dict(steps=60, kernel_size=3, stride=3, normalize=True,
                              remove_background=True, p=0.75, n=4, style="thumos"),
    # THUMOS preset with ActivityNet's adaptation budget
    "C_thumos_15steps": dict(steps=15, kernel_size=20, stride=20, normalize=True,
                             remove_background=True, p=0.75, n=4, style="thumos"),
    # upstream configs/model/tt_anet.yaml, verbatim
    "D_anet": dict(steps=15, kernel_size=50, stride=200, normalize=False,
                   remove_background=False, p=0.8, n=20, style="anet"),
}

VIDEO_SPLIT_META = {}


def gt_meta(ds):
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): (float(z["duration"][i]), str(z["split"][i]))
            for i, v in enumerate(z["video_ids"])}


# ------------------------------------------------------------------ the net ---
def build_net(preset: dict, device):
    sys.path.insert(0, str(T3AL_DIR))
    from src.models.components.tt_method import T3ALNet
    from src.models.components.loss import ByolLoss

    class CampaignT3AL(T3ALNet):
        """T3ALNet with the dataset-specific plumbing replaced, mechanism untouched."""

        def __init__(self, cfg):
            torch.nn.Module.__init__(self)
            import open_clip
            self.stride = cfg["stride"]
            self.randper = 10
            self.p = cfg["p"]
            self.n = cfg["n"]
            self.normalize = cfg["normalize"]
            self.text_projection = True
            self.text_encoder = False
            self.image_projection = True
            self.logit_scale = True
            self.remove_background = cfg["remove_background"]
            self.ltype = "BYOLfeat"
            self.steps = cfg["steps"]
            self.refine_with_captions = True
            self.split = 0
            self.setting = 50
            # `dataset` is not a corpus name inside T3ALNet, it is a two-way switch
            # over (moving average on/off, segment threshold rule).  We keep the two
            # published settings as they are and let the val sweep pick one.
            self.dataset = cfg["style"]
            self.visualize = True           # routed into `plot_visualize` below
            self.kernel_size = cfg["kernel_size"]
            self.video_path = ""
            self.topk = 3
            self.m = 0.7

            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name="coca_ViT-L-14",
                pretrained="mscoco_finetuned_laion2B-s13B-b90k")
            self.model = self.model.float()

            self.cls_names = {c: i for i, c in enumerate(CLASS_NAMES)}
            self.num_classes = len(self.cls_names)
            self.inverted_cls = {v: k for k, v in self.cls_names.items()}
            self.inverted_cls[len(CLASS_NAMES)] = MAIN_LABEL  # `mainq_sim` target
            self.text_features = self.get_text_features(self.model)
            self.annotations = {}
            self.tta_loss = ByolLoss()

            self.force_index = None
            self.last_similarity = None

        # our features are on the campaign's 4 fps grid, not the video's own fps
        def get_video_fps(self, video_name):
            return FPS

        # severed on purpose: no label of any split reaches the method
        def get_segments_gt(self, video_name, fps):
            return [], set()

        def infer_pseudo_labels(self, image_features):
            idx, scores = super().infer_pseudo_labels(image_features)
            if self.force_index is None:
                return idx, scores
            return torch.tensor(self.force_index, device=idx.device), scores

        # `visualize=True` is how upstream exposes the final similarity signal; we
        # capture it instead of plotting.  No value is altered.
        def plot_visualize(self, video_name, similarity, indexes, segments_gt,
                           segment, unique_labels):
            self.last_similarity = similarity.detach().float().cpu().numpy().reshape(-1)
            return None

    net = CampaignT3AL(preset).to(device)
    return net


def make_optimizer(net, lr=1e-5, weight_decay=1e-4, scaling_factor=0.001):
    """The parameter groups of `T3ALModule.configure_optimizers`, verbatim.

    Rebuilt per video (see the section's deviation on optimiser scope) so that the
    run is order-independent and a resume reproduces an uninterrupted run.
    """
    for p in net.parameters():
        p.requires_grad = False
    groups = []
    net.model.text.text_projection.requires_grad = True
    groups.append({"params": net.model.text.text_projection, "lr": lr})
    net.model.visual.proj.requires_grad = True
    groups.append({"params": net.model.visual.proj, "lr": lr * scaling_factor})
    net.model.logit_scale.requires_grad = True
    groups.append({"params": net.model.logit_scale, "lr": lr})
    return torch.optim.Adam(groups, lr=lr, weight_decay=weight_decay)


# ----------------------------------------------------------------- the loop ---
def run_video(net, feats, vid, force_index, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    net.force_index = force_index
    net.last_similarity = None
    # Per-sample adaptation: the checkpoint is restored to its pre-trained state
    # before every video.  See the section's deviation on adaptation scope --
    # upstream's own restore point is one optimiser step stale, which lets state
    # leak from one test video into the next.
    # `strict=False`: upstream's caption-refinement branch contains the typo
    # `self.model.locit_scale = ...`, which registers an extra, unused parameter of
    # that name on the module the first time refinement fires.  It is left in place
    # (removing it would change the published code path); it is simply not restored.
    net.model.load_state_dict(net.pristine, strict=False)
    opt = make_optimizer(net)
    x = (0, [vid], feats.clone().to(device))
    _, output, pred_mask, _, _, _ = net(x, opt)
    return output, pred_mask.detach().float().cpu().numpy(), net.last_similarity


def curves_from_output(output, T, toxic_only):
    """Piecewise-constant curve + intervals from T3AL's segment list.

    `output[i]['segment']` is a pair of feature indices; `output[i]['score']` is the
    softmax over the six frozen classes for that segment.
    """
    c = np.zeros(T, dtype=np.float32)
    ivs = []
    for o in output:
        if o["label"] == -1 or not len(o["segment"]):
            continue
        s = o["score"]
        s = s.detach().float().cpu().numpy().reshape(-1) if torch.is_tensor(s) else np.asarray(s).reshape(-1)
        score = float(s[1:].sum()) if toxic_only is None else float(s[toxic_only])
        i0, i1 = int(o["segment"][0]), int(o["segment"][1])
        i0, i1 = max(0, min(i0, T)), max(0, min(i1, T))
        if i1 <= i0:
            continue
        c[i0:i1] = np.maximum(c[i0:i1], score)
        ivs.append([i0 / FPS, i1 / FPS, score])
    return c, ivs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HateMM,MHC,MHC_zh,HateClipSeg")
    ap.add_argument("--splits", default="test")
    ap.add_argument("--preset", default="A_thumos", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=20250819)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--hcs-classes", action="store_true",
                    help="also run the six forced-class variants on HateClipSeg")
    ap.add_argument("--variants", default="main,mainq_sim,c1_hateful")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-frames", type=int, default=16)
    ap.add_argument("--mem-frac", type=float, default=0.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mem_frac > 0 and device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    out_root = Path(args.out_dir) if args.out_dir else (
        ROOT / f"idea-stage/repro_t3al/curves_s{args.seed}")
    out_root.mkdir(parents=True, exist_ok=True)

    net = build_net(PRESETS[args.preset], device)
    net.eval()
    net.pristine = {k: v.detach().clone() for k, v in net.model.state_dict().items()}

    base_variants = [v for v in args.variants.split(",") if v]
    splits = set(args.splits.split(","))
    t_all = time.time()
    for ds in args.datasets.split(","):
        meta = gt_meta(ds)
        fdir = FEAT_DIR(ds)
        ids = sorted(v for v, (_, sp) in meta.items()
                     if sp in splits and (fdir / f"{v}.npy").exists())
        if args.limit:
            ids = ids[: args.limit]
        variants = list(base_variants)
        if args.hcs_classes and ds == "HateClipSeg":
            variants += [k for k in CLASS_KEYS if k not in variants]
        odir = out_root / ds
        odir.mkdir(parents=True, exist_ok=True)
        # `./captions/<vid>.txt` is hardcoded in upstream's refinement step, so the
        # process runs from a per-dataset work directory holding that symlink.
        wdir = WORK_DIR / ds
        wdir.mkdir(parents=True, exist_ok=True)
        link = wdir / "captions"
        if not link.exists():
            link.symlink_to(CAP_DIR(ds))
        os.chdir(wdir)

        iv_all = {v: {} for v in variants}
        for v in variants:
            f = out_root / f"{ds}_intervals_{v}.json"
            if f.exists():
                try:
                    iv_all[v] = json.loads(f.read_text())
                except Exception:
                    iv_all[v] = {}

        print(f"[plan] {ds} preset={args.preset} seed={args.seed} "
              f"n={len(ids)} variants={variants}", flush=True)
        t0, ndone = time.time(), 0
        for i, vid in enumerate(ids):
            p = odir / f"{vid}.npz"
            if p.exists() and all(vid in iv_all[v] for v in variants):
                continue
            feats = np.load(fdir / f"{vid}.npy")
            T = feats.shape[0]
            if T < args.min_frames:
                print(f"[SKIP-SHORT] {ds} {vid} T={T}", flush=True)
                continue
            ft = torch.from_numpy(feats).float().unsqueeze(0)
            # per-video seed, so a resumed run reproduces an uninterrupted one
            vseed = (args.seed ^ zlib.crc32(vid.encode())) % (2 ** 31 - 1)
            store = {"rate": np.float32(FPS)}
            try:
                for v in variants:
                    if v == "main":
                        fi, toxic = None, None
                    elif v == "mainq_sim":
                        fi, toxic = len(CLASS_NAMES), None
                    else:
                        k = CLASS_KEYS.index(v)
                        fi, toxic = k, k
                    output, pred_mask, sim = run_video(net, ft, vid, fi, device, vseed)
                    seg_c, ivs = curves_from_output(output, T, toxic)
                    if v == "main":
                        curve = seg_c
                    else:
                        # the adapted model's continuous localisation signal
                        curve = np.zeros(T, dtype=np.float32) if sim is None else sim[:T]
                        if curve.shape[0] < T:
                            curve = np.concatenate(
                                [curve, np.full(T - curve.shape[0], curve[-1] if curve.size else 0.0,
                                                dtype=np.float32)])
                    store[v] = curve.astype(np.float32)
                    iv_all[v][vid] = ivs
            except Exception as e:
                print(f"[FAIL] {ds} {vid}: {type(e).__name__}: {e}", flush=True)
                continue
            np.savez(p, **store)
            ndone += 1
            if ndone % 10 == 0 or i + 1 == len(ids):
                el = time.time() - t0
                rate = ndone / max(el, 1e-9)
                for v in variants:
                    (out_root / f"{ds}_intervals_{v}.json").write_text(json.dumps(iv_all[v]))
                print(f"[prog] {ds} {i+1}/{len(ids)} done={ndone} {el:.0f}s "
                      f"{rate:.3f} vid/s eta={(len(ids)-i-1)/max(rate,1e-9)/60:.1f}min",
                      flush=True)
        for v in variants:
            (out_root / f"{ds}_intervals_{v}.json").write_text(json.dumps(iv_all[v]))
        print(f"[ds-done] {ds} {time.time()-t0:.0f}s", flush=True)
    print(f"[done] {time.time()-t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
