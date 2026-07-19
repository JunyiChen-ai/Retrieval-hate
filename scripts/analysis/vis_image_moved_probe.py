#!/usr/bin/env python
"""VISION-UNFREEZE — EN image-MOVED early-kill gate (prereg refine-logs/VISION_UNFREEZE_PREREG.md §3.4).

Pre-declared $0-after-extract diagnostic. Reuses the COMMITTED F58 machinery
(scripts/analysis/encoder_swap_geometry.py — the same functions hatemm_lora_stream_decomp.py
imports) to ask ONE question on the newly-extracted vision-adapted EN features:

    Did LoRA reaching the Qwen2.5-VL vision tower actually MOVE the EN image stream vs the
    banked LLM-only generic-LoRA image stream?

Gate (F58's MOVED rule VERBATIM; encoder_swap_geometry has no threshold constants, F58's
hatemm_lora_stream_decomp.py:86-89 pins MOVE_TR=0.010 / MOVE_DV=0.005):
    dAUC_img = AUC_img(vis-LoRA) - AUC_img(generic-LoRA), per footing:
      * train-LOO kNN AUC (G.loo_knn -> G.auc)
      * held-out dev kNN AUC (G.knn_vote memory=train -> G.auc)
    MOVED    iff dAUC_img >= +0.010 (train-LOO) AND >= +0.005 (dev)   [same + sign]
    DEGRADED iff dAUC_img <= -0.010 (train-LOO)
    FLAT     otherwise.

Verdict routing (pre-registered):
    MOVED    -> EN head budget PROCEEDS (the vision LoRA is live on the EN image stream).
    FLAT/DEGRADED -> EN head budget CANCELLED, bank the "vision LoRA inert on EN image" kill;
                    the combined head job runs the HateMM leg ONLY (submit enc3seed_lora_vis.sbatch "HateMM").

Anchor (re-derived by the prereg AUTHOR with THIS operator, banked generic-LoRA EN cache):
    generic-LoRA EN image train-LOO AUC 0.6236 / dev 0.6756 (context: CLIP healthy 0.7338/0.7367,
    frozen-Qwen collapsed 0.5992/0.6865). NB: these differ from the red-team §0 scratch-probe
    numbers (gen .659/.695) by OPERATOR (~0.01-0.04) — the DELTA gate is operator-robust because
    both arms are read by the SAME committed operator; the threshold is F58's operator-independent
    resolution floor. See prereg §11 DEV-4.

ZERO GPU / ZERO Modal / ZERO test-touch. Banked train + dev_seen .pt caches only; test is never read.
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import encoder_swap_geometry as G  # committed F58 machinery: load, build_modality, loo_knn, knn_vote, auc

K = 20
MOVE_TR = 0.010   # F58 train-LOO threshold for MOVED (hatemm_lora_stream_decomp.py:86)
MOVE_DV = 0.005   # F58 dev threshold for MOVED (hatemm_lora_stream_decomp.py:87)


def stream_auc(ds_dir, tag):
    """Return per-stream (img/text/concat) train-LOO + dev kNN AUC for one encoder cache."""
    tr_img, tr_txt, tr_y, tr_ids = G.load(ds_dir, tag, "train")
    dv_img, dv_txt, dv_y, dv_ids = G.load(ds_dir, tag, "dev_seen")
    rec = {"n_train": int(len(tr_y)), "n_dev": int(len(dv_y)),
           "train_pos_frac": float(tr_y.mean()), "dev_pos_frac": float(dv_y.mean()),
           "tr_ids": tr_ids, "dv_ids": dv_ids, "tr_y": tr_y, "dv_y": dv_y}
    for mode in ("img", "text", "concat"):
        Xtr = G.build_modality(tr_img, tr_txt, mode)
        Xdv = G.build_modality(dv_img, dv_txt, mode)
        _, ls = G.loo_knn(Xtr, tr_y, K)
        _, score, _ = G.knn_vote(Xtr, tr_y, Xdv, K)
        rec[mode] = dict(train_loo_auc=float(G.auc(tr_y, ls)), dev_auc=float(G.auc(dv_y, score)))
    return rec


def move_class(dtr, ddv):
    if dtr >= MOVE_TR and ddv >= MOVE_DV:
        return "MOVED"
    if dtr <= -MOVE_TR:
        return "DEGRADED"
    return "FLAT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHC", help="cache dir under data/CLIP_Embedding/ (default MHC = EN)")
    ap.add_argument("--generic_tag", default="Qwen2.5-VL-7B-Instruct-LoRA_HF",
                    help="banked LLM-only generic-LoRA cache tag (the anchor)")
    ap.add_argument("--vis_tag", default="Qwen2.5-VL-7B-Instruct-LoRA-vis_HF",
                    help="newly-extracted vision-unfrozen LoRA cache tag (the treatment)")
    ap.add_argument("--context", action="store_true",
                    help="also print CLIP + frozen-Qwen image AUC (collapse / healthy ceiling context)")
    args = ap.parse_args()

    gen = stream_auc(args.dataset, args.generic_tag)
    vis = stream_auc(args.dataset, args.vis_tag)

    # id / label alignment (must be identical across the two LoRA caches; F58-style assert)
    assert (gen["tr_ids"] == vis["tr_ids"]).all(), "train id mismatch generic vs vis"
    assert (gen["dv_ids"] == vis["dv_ids"]).all(), "dev id mismatch generic vs vis"
    assert (gen["tr_y"] == vis["tr_y"]).all() and (gen["dv_y"] == vis["dv_y"]).all(), "label mismatch"

    d_img_tr = vis["img"]["train_loo_auc"] - gen["img"]["train_loo_auc"]
    d_img_dv = vis["img"]["dev_auc"] - gen["img"]["dev_auc"]
    d_txt_tr = vis["text"]["train_loo_auc"] - gen["text"]["train_loo_auc"]
    d_txt_dv = vis["text"]["dev_auc"] - gen["text"]["dev_auc"]
    img_move = move_class(d_img_tr, d_img_dv)

    ctx = {}
    if args.context:
        for name, tag in (("CLIP", "openai_clip-vit-large-patch14-336_HF"),
                          ("frozen-Qwen", "Qwen2.5-VL-7B-Instruct_HF")):
            try:
                r = stream_auc(args.dataset, tag)
                ctx[name] = dict(img_train_loo=r["img"]["train_loo_auc"], img_dev=r["img"]["dev_auc"])
            except Exception as e:
                ctx[name] = f"(unavailable: {e})"

    proceed = (img_move == "MOVED")
    verdict = ("EN-HEAD-PROCEEDS (image stream MOVED)" if proceed
               else "EN-HEAD-CANCELLED (image stream %s -> vision LoRA inert on EN; bank kill; run HateMM only)" % img_move)

    print("=" * 78)
    print("VISION-UNFREEZE EN IMAGE-MOVED GATE  (dataset=%s, K=%d)" % (args.dataset, K))
    print("  thresholds: MOVED iff dAUC_img >= +%.3f train-LOO AND >= +%.3f dev" % (MOVE_TR, MOVE_DV))
    print("=" * 78)
    print("%-14s | %-12s %-12s" % ("arm", "img trLOO", "img dev"))
    print("%-14s | %-12.4f %-12.4f" % ("generic-LoRA", gen["img"]["train_loo_auc"], gen["img"]["dev_auc"]))
    print("%-14s | %-12.4f %-12.4f" % ("vis-LoRA", vis["img"]["train_loo_auc"], vis["img"]["dev_auc"]))
    print("%-14s | %+12.4f %+12.4f  -> IMG %s" % ("dAUC(vis-gen)", d_img_tr, d_img_dv, img_move))
    print("  (context) text dAUC(vis-gen): trLOO %+.4f  dev %+.4f" % (d_txt_tr, d_txt_dv))
    if ctx:
        for name, r in ctx.items():
            if isinstance(r, dict):
                print("  (context) %-11s img trLOO %.4f  dev %.4f" % (name, r["img_train_loo"], r["img_dev"]))
    print("-" * 78)
    print("GATE VERDICT: %s" % verdict)
    print("=" * 78)

    out = dict(dataset=args.dataset, K=K, thresholds=dict(MOVE_TR=MOVE_TR, MOVE_DV=MOVE_DV),
               generic=gen, vis=vis, context=ctx,
               d_img_train_loo=float(d_img_tr), d_img_dev=float(d_img_dv),
               d_text_train_loo=float(d_txt_tr), d_text_dev=float(d_txt_dv),
               img_move=img_move, en_head_proceeds=bool(proceed), verdict=verdict)
    # drop non-serialisable arrays before dump
    for r in (out["generic"], out["vis"]):
        for k in ("tr_ids", "dv_ids", "tr_y", "dv_y"):
            r.pop(k, None)
    outp = os.path.join(HERE, "vis_image_moved_%s_out.json" % args.dataset)
    with open(outp, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", outp)
    # exit code encodes the gate for --dependency-free scripting (0 = PROCEED, 10 = CANCELLED)
    sys.exit(0 if proceed else 10)


if __name__ == "__main__":
    main()
