#!/usr/bin/env python
"""
HateMM LoRA per-stream decomposition (F45's ZH machinery, replicated on HateMM).

QUESTION (orchestrator, surface-4 of WAVE6_PREMISE_HUNT.md). The analysis chapter
§3.9 currently ASSERTS by inference (from F44/F45) that HateMM's LoRA gain is
"image-inherited": HateMM decides on the image stream (image-only train-LOO AUC
0.826), LoRA leaves it intact, so LoRA inherits and preserves frozen-Qwen's
image-borne Pareto conversion; the text stream LoRA sharpens is HateMM's secondary
modality and adds ~0 on top. F45 only ever decomposed ZH. F54 corrected the
"LoRA moves text only" premise into an EMPIRICAL (not architectural) claim: the
LLM backbone that re-contextualises the vision-pad tokens IS LoRA-adapted
(lora_target: all), so the image stream is architecturally MOVABLE. Did the
HateMM SFT actually move it? Measure it.

Classification is one of three (orchestrator's menu):
  (A) IMAGE-INHERITED  — image stream ~ frozen under LoRA, decisive modality is
      image, LoRA ~ frozen-Qwen downstream (the pass is inherited; §3.9's claim).
  (B) IMAGE-MOVED      — F54's open possibility: vision-pad tokens re-processed by
      the LoRA'd backbone shifted the image stream materially. New finding => errata.
  (C) TEXT-DRIVEN      — like ZH: gain lives in the text stream, image flat, and
      LoRA adds accuracy on top of frozen-Qwen via the text move.

==============================================================================
PRE-DECLARED CLASSIFICATION RULE (design-locked BEFORE any LoRA delta was seen;
only F44's frozen-Qwen numbers were known — the LoRA image ΔAUC, the crux, was NOT):
==============================================================================
Let dAUC_s = AUC_s(LoRA) - AUC_s(frozen-Qwen), per stream s in {img, text},
measured on TWO footings: train-LOO kNN AUC and held-out dev kNN AUC.

STREAM-MOVEMENT (which stream moved under SFT):
  * MOVED     iff dAUC_s >= +0.010 on train-LOO AND >= +0.005 on dev (same + sign).
  * FLAT      iff |dAUC_s| < 0.010 on train-LOO (dev corroborating |.|<0.010).
  * DEGRADED  iff dAUC_s <= -0.010 on train-LOO.
  (+0.010 is the resolution floor: ZH's image moved -0.007/-0.007 => FLAT under
   this same rule, matching F45's "image untouched" verdict — rule is ZH-calibrated.)

DECISIVE MODALITY: the stream (img vs text) with the higher STANDALONE kNN AUC,
required to agree between train-LOO and dev. (F44 gave HateMM image-only 0.826.)

TOP-LEVEL CLASSIFICATION (decision tree, evaluated top-down):
  (B) IMAGE-MOVED   iff image stream is MOVED (dAUC_img >= +0.010 both footings).
  (C) TEXT-DRIVEN   iff image FLAT/DEGRADED AND text MOVED AND LoRA beats
                    frozen-Qwen downstream by >= +0.010 acc (banked test/dev acc).
  (A) IMAGE-INHERITED iff image FLAT AND decisive-modality == image AND
                    LoRA ~ frozen-Qwen downstream (|Δacc| < 0.010, "adds ~0 on top").
                    (text MAY be MOVED — §3.9 already says the text stream is
                     sharpened — but it adds ~0 because image is decisive & unmoved.)
  else -> REPORT-AS-MIXED with the raw deltas and no forced label.

PARETO vs ROTATION (banked test per-class recall, minority=hate):
  * Pareto (convertible) iff Δhate_recall > 0 AND Δnonhate_recall >= -0.010.
  * Rotation (unconvertible) iff Δhate_recall > 0 AND Δnonhate_recall < -0.010.
  Applied to LoRA-CLIP, LoRA-frozen, frozen-CLIP.

MACHINERY-VALIDITY GATE (kill bar i): the concat dev read-out must reproduce the
banked HateMM downstream SIGN — dev concat AUC(frozen-Qwen) - AUC(CLIP) > 0, the
geometry gap that F44 tied to the PASS 3/3. If it does not, the read-out is void.
==============================================================================

ZERO GPU / ZERO Modal / ZERO test-touch. Banked train/dev .pt caches (geometry)
and banked completed-run trainlogs (per-class recall, already logged — NO new test
evaluation) only, same provenance discipline as F45/B3_VERDICT_REVIEW.

Reuses scripts/analysis/encoder_swap_geometry.py verbatim (imported).
"""
import os, sys, json, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import encoder_swap_geometry as G  # load, l2n, build_modality, auc, knn_vote, loo_knn, knn_homogeneity, centroid_sep, macro_f1, bal_acc, linear_probe

DS_DIR = "HateMM"
ENC = {
    "CLIP": "openai_clip-vit-large-patch14-336_HF",
    "frozenQwen": "Qwen2.5-VL-7B-Instruct_HF",
    "LoRAQwen": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
}
K = 20
LOG = "/data/jehc223/RGCL/slurm/logs"
JOBS = {"CLIP": "12850", "frozenQwen": "12850", "LoRAQwen": "13235"}
SEEDS = (0, 1, 2)
WARMUP = 5

MOVE_TR = 0.010   # train-LOO threshold for MOVED
MOVE_DV = 0.005   # dev threshold for MOVED
DOWNSTREAM_ACC = 0.010  # "adds on top" acc threshold
PARETO_COST = 0.010     # non-hate recall cost that flips Pareto->rotation


# ----------------------------- geometry -----------------------------
def geometry():
    caches = {}
    for name, tag in ENC.items():
        tr_img, tr_txt, tr_y, tr_ids = G.load(DS_DIR, tag, "train")
        dv_img, dv_txt, dv_y, dv_ids = G.load(DS_DIR, tag, "dev_seen")
        caches[name] = dict(tr_img=tr_img, tr_txt=tr_txt, tr_y=tr_y, tr_ids=tr_ids,
                            dv_img=dv_img, dv_txt=dv_txt, dv_y=dv_y, dv_ids=dv_ids)
    # id / label alignment (must be identical across encoders)
    ref = caches["CLIP"]
    for name in ENC:
        assert (caches[name]["tr_ids"] == ref["tr_ids"]).all(), f"train id mismatch {name}"
        assert (caches[name]["dv_ids"] == ref["dv_ids"]).all(), f"dev id mismatch {name}"
        assert (caches[name]["tr_y"] == ref["tr_y"]).all(), f"train label mismatch {name}"
        assert (caches[name]["dv_y"] == ref["dv_y"]).all(), f"dev label mismatch {name}"

    out = {}
    for name in ENC:
        c = caches[name]
        rec = {"n_train": int(len(c["tr_y"])), "n_dev": int(len(c["dv_y"])),
               "train_pos_frac": float(c["tr_y"].mean()), "dev_pos_frac": float(c["dv_y"].mean())}
        for mode in ("img", "text", "concat"):
            Xtr = G.build_modality(c["tr_img"], c["tr_txt"], mode)
            Xdv = G.build_modality(c["dv_img"], c["dv_txt"], mode)
            pred, score, _ = G.knn_vote(Xtr, c["tr_y"], Xdv, K)
            d2d = dict(acc=float(np.mean(pred == c["dv_y"])), balacc=G.bal_acc(c["dv_y"], pred),
                       macrof1=G.macro_f1(c["dv_y"], pred), auc=G.auc(c["dv_y"], score))
            lp, ls = G.loo_knn(Xtr, c["tr_y"], K)
            loo = dict(acc=float(np.mean(lp == c["tr_y"])), balacc=G.bal_acc(c["tr_y"], lp),
                       macrof1=G.macro_f1(c["tr_y"], lp), auc=G.auc(c["tr_y"], ls))
            homog = G.knn_homogeneity(Xtr, c["tr_y"], K, loo=True)
            b, w, r = G.centroid_sep(Xtr, c["tr_y"])
            rec[mode] = dict(dev_knn=d2d, train_loo_knn=loo, knn_homog20=homog,
                             centroid_between=b, centroid_within=w, centroid_ratio=r)
        out[name] = rec
    return caches, out


def empty_transcript_subgroup(caches):
    """F44's HateMM had ~5.6% degenerate (empty/near-empty) CLIP transcripts.
    Identify the empty-transcript subgroup by near-zero Qwen text-norm (the Qwen
    encoder emits a (near-)zero text vector when the transcript is empty), and
    report per-stream dev-kNN AUC restricted to full-transcript vs the whole set."""
    sub = {}
    for split, prefix in (("train", "tr"), ("dev_seen", "dv")):
        # use frozenQwen text norms to flag empties (identical id order across encoders)
        txt = caches["frozenQwen"][f"{prefix}_txt"]
        norms = np.linalg.norm(txt, axis=1)
        empty = norms < 1e-3
        sub[split] = dict(n=int(len(norms)), n_empty=int(empty.sum()),
                          frac_empty=float(empty.mean()))
    # dev AUC on full-transcript-only subset, per encoder, text & concat streams
    dv_txt_norm = np.linalg.norm(caches["frozenQwen"]["dv_txt"], axis=1)
    dv_full = dv_txt_norm >= 1e-3
    tr_txt_norm = np.linalg.norm(caches["frozenQwen"]["tr_txt"], axis=1)
    tr_full = tr_txt_norm >= 1e-3
    perenc = {}
    for name in ENC:
        c = caches[name]
        r = {}
        for mode in ("text", "concat"):
            Xtr = G.build_modality(c["tr_img"], c["tr_txt"], mode)
            Xdv = G.build_modality(c["dv_img"], c["dv_txt"], mode)
            # full-transcript dev queries only, memory = full train bank
            pred, score, _ = G.knn_vote(Xtr[tr_full], c["tr_y"][tr_full], Xdv[dv_full], K)
            r[mode + "_devAUC_fulltxt"] = G.auc(c["dv_y"][dv_full], score)
        perenc[name] = r
    sub["fulltxt_dev"] = dict(n_train_full=int(tr_full.sum()), n_dev_full=int(dv_full.sum()),
                              per_encoder=perenc)
    return sub


# ----------------------------- trainlog parse (banked, no new test) -----------------------------
reA = re.compile(r"(Val|Test)_Retrieval Epoch\s+(\d+) acc: ([\d.]+) roc: ([\d.]+) pre: ([\d.]+) recall: ([\d.]+) f1: ([\d.]+)\s*$")
reB = re.compile(r"(Val|Test)_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")


def parse_trainlog(path):
    ep = {}
    with open(path) as f:
        for ln in f:
            m = reA.match(ln.strip())
            if m:
                split, e = m.group(1), int(m.group(2))
                acc, roc, pre, rec, f1 = map(float, m.groups()[2:])
                d = ep.setdefault(e, {})
                d[f"{split}_acc"] = acc; d[f"{split}_roc"] = roc
                d[f"{split}_hate_recall"] = rec; d[f"{split}_hate_pre"] = pre; d[f"{split}_hate_f1"] = f1
                continue
            m = reB.match(ln.strip())
            if m:
                split, e = m.group(1), int(m.group(2))
                mf1, mp, mr, acc, roc = map(float, m.groups()[2:])
                d = ep.setdefault(e, {})
                d[f"{split}_mF1"] = mf1; d[f"{split}_macroR"] = mr; d[f"{split}_acc"] = acc
    for e, d in ep.items():
        for sp in ("Val", "Test"):
            if f"{sp}_macroR" in d and f"{sp}_hate_recall" in d:
                d[f"{sp}_nonhate_recall"] = 2 * d[f"{sp}_macroR"] - d[f"{sp}_hate_recall"]
    return ep


def valselect(ep):
    cand = [e for e in ep if e >= WARMUP and "Val_acc" in ep[e]]
    return max(cand, key=lambda e: (ep[e]["Val_acc"], ep[e].get("Val_roc", 0)))


def trainlogs():
    OUT = {}
    for arm, tag in ENC.items():
        job = JOBS[arm]
        OUT[arm] = {"seeds": {}}
        for s in SEEDS:
            p = f"{LOG}/enc3s_HateMM_{tag}_seed{s}_{job}.trainlog"
            ep = parse_trainlog(p)
            fin = max(ep)
            vs = valselect(ep)
            best_test = max((e for e in ep if e >= WARMUP), key=lambda e: ep[e]["Test_acc"])
            OUT[arm]["seeds"][s] = dict(final_ep=fin, valsel_ep=vs, best_test_ep=best_test,
                                        final=ep[fin], valsel=ep[vs],
                                        best_test_acc=ep[best_test]["Test_acc"])
        for proto, key in [("final", "final"), ("valsel", "valsel")]:
            for met in ("Test_acc", "Test_mF1", "Test_hate_recall", "Test_nonhate_recall",
                        "Test_roc", "Val_acc"):
                vals = [OUT[arm]["seeds"][s][key].get(met, float("nan")) for s in SEEDS]
                OUT[arm].setdefault(proto + "_mean", {})[met] = float(np.nanmean(vals))
    return OUT


def pareto_or_rotation(dhate, dnon):
    if dhate > 0 and dnon >= -PARETO_COST:
        return "PARETO"
    if dhate > 0 and dnon < -PARETO_COST:
        return "ROTATION"
    if dhate <= 0:
        return "NO-HATE-GAIN"
    return "MIXED"


# ----------------------------- classification -----------------------------
def stream_move(dtr, ddv):
    if dtr >= MOVE_TR and ddv >= MOVE_DV:
        return "MOVED"
    if dtr <= -MOVE_TR:
        return "DEGRADED"
    return "FLAT"


def classify(geo, tl):
    L, F, C = geo["LoRAQwen"], geo["frozenQwen"], geo["CLIP"]
    # per-stream dAUC LoRA - frozen, on train-LOO and dev
    d = {}
    for s in ("img", "text"):
        d[s] = dict(
            dtr_LmF=L[s]["train_loo_knn"]["auc"] - F[s]["train_loo_knn"]["auc"],
            ddv_LmF=L[s]["dev_knn"]["auc"] - F[s]["dev_knn"]["auc"],
            dtr_LmC=L[s]["train_loo_knn"]["auc"] - C[s]["train_loo_knn"]["auc"],
            ddv_LmC=L[s]["dev_knn"]["auc"] - C[s]["dev_knn"]["auc"],
            dtr_FmC=F[s]["train_loo_knn"]["auc"] - C[s]["train_loo_knn"]["auc"],
        )
        d[s]["move"] = stream_move(d[s]["dtr_LmF"], d[s]["ddv_LmF"])
    # decisive modality: higher standalone kNN AUC, agreeing on both footings, LoRA arm
    def decisive(rec):
        tr = "img" if rec["img"]["train_loo_knn"]["auc"] >= rec["text"]["train_loo_knn"]["auc"] else "text"
        dv = "img" if rec["img"]["dev_knn"]["auc"] >= rec["text"]["dev_knn"]["auc"] else "text"
        return tr, dv
    dec_L = decisive(L); dec_C = decisive(C); dec_F = decisive(F)
    decisive_mod = dec_L[0] if dec_L[0] == dec_L[1] else "SPLIT(%s/%s)" % dec_L

    # downstream LoRA vs frozen (banked, val-selected AND final-epoch acc)
    dacc_LmF_final = tl["LoRAQwen"]["final_mean"]["Test_acc"] - tl["frozenQwen"]["final_mean"]["Test_acc"]
    dacc_LmF_val = tl["LoRAQwen"]["valsel_mean"]["Test_acc"] - tl["frozenQwen"]["valsel_mean"]["Test_acc"]

    img_move = d["img"]["move"]; txt_move = d["text"]["move"]
    # decision tree
    if img_move == "MOVED":
        label = "IMAGE-MOVED"
    elif img_move in ("FLAT", "DEGRADED") and txt_move == "MOVED" and dacc_LmF_final >= DOWNSTREAM_ACC:
        label = "TEXT-DRIVEN"
    elif img_move in ("FLAT", "DEGRADED") and decisive_mod == "img" and abs(dacc_LmF_final) < DOWNSTREAM_ACC:
        label = "IMAGE-INHERITED"
    else:
        label = "MIXED / REPORT-RAW"

    return dict(per_stream=d, decisive_modality=decisive_mod,
                decisive_LoRA=dec_L, decisive_CLIP=dec_C, decisive_frozen=dec_F,
                dacc_LoRA_minus_frozen_final=dacc_LmF_final,
                dacc_LoRA_minus_frozen_valsel=dacc_LmF_val,
                img_move=img_move, txt_move=txt_move, label=label)


def main():
    caches, geo = geometry()
    sub = empty_transcript_subgroup(caches)
    tl = trainlogs()
    cls = classify(geo, tl)

    # machinery-validity gate: dev concat AUC frozen-Qwen - CLIP must be > 0
    dev_concat_FmC = geo["frozenQwen"]["concat"]["dev_knn"]["auc"] - geo["CLIP"]["concat"]["dev_knn"]["auc"]
    gate_ok = dev_concat_FmC > 0

    OUT = dict(geometry=geo, empty_transcript=sub, trainlogs=tl, classification=cls,
               machinery_gate=dict(dev_concat_AUC_frozen_minus_CLIP=dev_concat_FmC, passes=bool(gate_ok)),
               thresholds=dict(MOVE_TR=MOVE_TR, MOVE_DV=MOVE_DV,
                               DOWNSTREAM_ACC=DOWNSTREAM_ACC, PARETO_COST=PARETO_COST))

    # console report
    print("=" * 78)
    print("MACHINERY GATE: dev concat AUC frozenQwen-CLIP = %+.4f  -> %s" %
          (dev_concat_FmC, "PASS" if gate_ok else "FAIL(void)"))
    print("=" * 78)
    print("\n=== per-stream kNN AUC (train-LOO / dev), 3 encoders ===")
    print("%-6s | %-22s | %-22s" % ("stream", "train-LOO AUC", "dev AUC"))
    print("%-6s | %6s %6s %6s | %6s %6s %6s" % ("", "CLIP", "froz", "LoRA", "CLIP", "froz", "LoRA"))
    for s in ("img", "text", "concat"):
        tr = [geo[e][s]["train_loo_knn"]["auc"] for e in ("CLIP", "frozenQwen", "LoRAQwen")]
        dv = [geo[e][s]["dev_knn"]["auc"] for e in ("CLIP", "frozenQwen", "LoRAQwen")]
        print("%-6s | %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f" % (s, tr[0], tr[1], tr[2], dv[0], dv[1], dv[2]))
    print("\n=== per-stream Δ (LoRA - frozen) and movement class ===")
    for s in ("img", "text"):
        ps = cls["per_stream"][s]
        print("  %-5s dAUC train-LOO=%+.4f  dev=%+.4f  -> %s   (LoRA-CLIP: tr%+.4f dv%+.4f)" %
              (s, ps["dtr_LmF"], ps["ddv_LmF"], ps["move"], ps["dtr_LmC"], ps["ddv_LmC"]))
    print("  decisive modality (LoRA arm):", cls["decisive_modality"],
          " | CLIP:", cls["decisive_CLIP"], " frozen:", cls["decisive_frozen"])

    print("\n=== banked TEST per-class recall (final-epoch mean, minority=hate) ===")
    print("%-11s | %7s %7s %9s %11s" % ("arm", "acc", "mF1", "hate_rec", "nonhate_rec"))
    for arm in ("CLIP", "frozenQwen", "LoRAQwen"):
        m = tl[arm]["final_mean"]
        print("%-11s | %7.4f %7.4f %9.4f %11.4f" %
              (arm, m["Test_acc"], m["Test_mF1"], m["Test_hate_recall"], m["Test_nonhate_recall"]))
    print("\n=== Pareto vs rotation (final-epoch TEST) ===")
    cM = tl["CLIP"]["final_mean"]; fM = tl["frozenQwen"]["final_mean"]; lM = tl["LoRAQwen"]["final_mean"]
    for lab, a, b in (("frozen - CLIP", fM, cM), ("LoRA - CLIP", lM, cM), ("LoRA - frozen", lM, fM)):
        dh = a["Test_hate_recall"] - b["Test_hate_recall"]
        dn = a["Test_nonhate_recall"] - b["Test_nonhate_recall"]
        da = a["Test_acc"] - b["Test_acc"]
        print("  %-14s dAcc=%+.4f  dHate=%+.4f  dNonhate=%+.4f  -> %s" %
              (lab, da, dh, dn, pareto_or_rotation(dh, dn)))

    print("\n=== downstream LoRA vs frozen-Qwen (does text sharpening add on top?) ===")
    print("  final-epoch dAcc(LoRA-frozen)=%+.4f  val-sel dAcc=%+.4f" %
          (cls["dacc_LoRA_minus_frozen_final"], cls["dacc_LoRA_minus_frozen_valsel"]))

    print("\n=== empty-transcript subgroup ===")
    for split in ("train", "dev_seen"):
        s = sub[split]
        print("  %-9s n=%d empty=%d (%.1f%%)" % (split, s["n"], s["n_empty"], 100 * s["frac_empty"]))
    ft = sub["fulltxt_dev"]
    print("  full-transcript dev (n_dev_full=%d): text/concat dev-AUC per encoder" % ft["n_dev_full"])
    for e in ("CLIP", "frozenQwen", "LoRAQwen"):
        pe = ft["per_encoder"][e]
        print("     %-11s text=%.4f concat=%.4f" % (e, pe["text_devAUC_fulltxt"], pe["concat_devAUC_fulltxt"]))

    print("\n" + "=" * 78)
    print("CLASSIFICATION: %s" % cls["label"])
    print("  img stream: %s | text stream: %s | decisive: %s" %
          (cls["img_move"], cls["txt_move"], cls["decisive_modality"]))
    print("=" * 78)

    outp = os.path.join(HERE, "hatemm_lora_stream_decomp_out.json")
    with open(outp, "w") as f:
        json.dump(OUT, f, indent=1, default=float)
    print("\nwrote", outp)


if __name__ == "__main__":
    main()
