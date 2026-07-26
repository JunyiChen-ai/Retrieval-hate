"""KS-MNTP-1: raw-key dev screen for the S1 bidir+meanpool arm.

Extends scripts/analysis/mntp_rawkey_devscreen.py (frozen sha
8bc009e68833d8bad3aecb531c7c8b9879e05a2e00430465e0b2b4f05f9dede0) to a THIRD arm
without editing it: `load_cache` / `l2` / `knn_vote` are IMPORTED from it, so the
vote operator is byte-identical to the one that produced the recon §1.2 numbers and
that file's sha is unchanged.

Three arms per dataset, all DEV, all raw untrained key space, NO head, NO test read:
  causal          — banked, deployed EOS-class text readout
  bidir-lasttoken — banked F72 arm (mask flipped, readout unchanged)
  bidir-meanpool  — S1 (mask flipped AND text readout = LLM2Vec mean over all positions)

Gate bars are FROZEN in MNTP_FORENSIC_RECON.md §5.2 and are hard-coded below; they are
quoted, never recomputed from the new data.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Verbatim operators from the frozen recon screen (sha 8bc009e6...).
from mntp_rawkey_devscreen import ROOT, TOPK, knn_vote, l2, load_cache  # noqa: E402

# FROZEN bars — MNTP_FORENSIC_RECON.md §5.2, text stream, dev, raw key space.
BARS = {
    "HateMM": {"causal": 0.8037, "bidir": 0.7570, "bar50": 0.7804, "floor25": 0.7687},
    "MHC_zh": {"causal": 0.8462, "bidir": 0.6282, "bar50": 0.7372, "floor25": 0.6827},
}

CELLS = [
    # dataset, causal tag, bidir-lasttoken tag, S1 bidir-meanpool tag
    ("HateMM",
     "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
     "Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF",
     "Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-meanpool_HF"),
    ("MHC_zh",
     "Qwen2.5-VL-7B-Instruct-LoRA_HF",
     "Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF",
     "Qwen2.5-VL-7B-Instruct-LoRA-bidir-meanpool_HF"),
]

ARMS = ("causal", "bidir", "meanpool")


def streams_of(tr_i, tr_t, dv_i, dv_t):
    return {
        "img": (l2(tr_i), l2(dv_i)),
        "text": (l2(tr_t), l2(dv_t)),
        "concat": (l2(torch.cat([l2(tr_i), l2(tr_t)], 1)),
                   l2(torch.cat([l2(dv_i), l2(dv_t)], 1))),
    }


def run(ds, tags):
    feats, out = {}, {}
    ids_ref = None
    for arm in ARMS:
        tr_ids, tr_i, tr_t, tr_y = load_cache(ds, "train", tags[arm])
        dv_ids, dv_i, dv_t, dv_y = load_cache(ds, "dev_seen", tags[arm])
        if ids_ref is None:
            ids_ref = (tr_ids, dv_ids)
        else:
            assert list(tr_ids) == list(ids_ref[0]), "train id order differs in arm " + arm
            assert list(dv_ids) == list(ids_ref[1]), "dev id order differs in arm " + arm
        feats[arm] = (tr_i, tr_t, dv_i, dv_t, tr_y, dv_y)
        for name, (M, Q) in streams_of(tr_i, tr_t, dv_i, dv_t).items():
            out["{}|{}".format(arm, name)] = knn_vote(Q, M, tr_y, dv_y)

    def cos(a, b):
        """Mean per-item cosine over DECODABLE items only.

        The zero-vector guard writes an all-zeros row for any undecodable video (HateMM
        train `hate_video_95`, idx 355 — present identically in every arm). A zeros-vs-zeros
        pair is a DEGENERATE comparison: the two arms agree perfectly, but cosine is
        undefined and torch returns 0.0, which drags the mean down and can masquerade as a
        real feature difference. Excluding those rows measures what the belt is actually for.
        """
        keep = (a.norm(dim=1) > 0) & (b.norm(dim=1) > 0)
        if int(keep.sum()) == 0:
            return float("nan")
        return float((l2(a[keep]) * l2(b[keep])).sum(1).mean())

    c = feats["causal"]
    b = feats["bidir"]
    m = feats["meanpool"]
    out["drift"] = {
        # S1 vs the deployed causal arm
        "meanpool_vs_causal": {"train_img": cos(m[0], c[0]), "train_text": cos(m[1], c[1]),
                               "dev_img": cos(m[2], c[2]), "dev_text": cos(m[3], c[3])},
        # S1 vs the F72 bidir arm (same mask, only the text readout differs)
        "meanpool_vs_bidir": {"train_img": cos(m[0], b[0]), "train_text": cos(m[1], b[1]),
                              "dev_img": cos(m[2], b[2]), "dev_text": cos(m[3], b[3])},
    }
    # STREAM-COLLAPSE diagnostic. Mean-over-all-tokens makes the text span nearly equal
    # to the img span (prefix = all but the ~3-4 trailing header tokens), and 82.5 % of
    # BOTH sequences are the SAME 8 frames' vision tokens (recon §1.5). The two streams
    # then differ only in the prompt text. If they collapse toward each other the concat
    # /align head loses the diversity it relies on, even if the text row recovers. This
    # is the cheapest way to see that happening, and it is $0.
    out["stream_collapse"] = {
        arm: {"train_img_text_cos": cos(f[0], f[1]), "dev_img_text_cos": cos(f[2], f[3])}
        for arm, f in (("causal", c), ("bidir", b), ("meanpool", m))
    }
    # BELT: the S1 img readout DELEGATES to the frozen causal `_encode` and the mask is
    # the same as the F72 arm's, so S1 img must reproduce banked bidir img (bf16 GPU
    # nondeterminism only). A low cosine here means the fork changed the img path.
    out["img_nullop_belt"] = {
        "train_cos": out["drift"]["meanpool_vs_bidir"]["train_img"],
        "dev_cos": out["drift"]["meanpool_vs_bidir"]["dev_img"],
        "bar": 0.9999,
        "pass": (out["drift"]["meanpool_vs_bidir"]["train_img"] >= 0.9999
                 and out["drift"]["meanpool_vs_bidir"]["dev_img"] >= 0.9999),
    }
    return out


if __name__ == "__main__":
    blob = {}
    for ds, ct, bt, mt in CELLS:
        tags = {"causal": ct, "bidir": bt, "meanpool": mt}
        r = run(ds, tags)
        blob[ds] = {"tags": tags, **{k: v for k, v in r.items()}}

        print("\n=== {} (DEV only, raw untrained key space, no head, no test) ===".format(ds))
        print("{:8s} {:16s} {:>8s} {:>8s} {:>8s}".format("stream", "arm", "acc", "mF1", "roc"))
        for name in ("img", "text", "concat"):
            for arm, lab in (("causal", "causal"), ("bidir", "bidir-lasttoken"),
                             ("meanpool", "bidir-MEANPOOL")):
                a, f, ro = r["{}|{}".format(arm, name)]
                print("{:8s} {:16s} {:8.4f} {:8.4f} {:8.4f}".format(name, lab, a, f, ro))
            dm = r["meanpool|{}".format(name)][0] - r["bidir|{}".format(name)][0]
            dc = r["meanpool|{}".format(name)][0] - r["causal|{}".format(name)][0]
            print("{:8s} {:16s} {:+8.4f}   (vs causal {:+8.4f})".format(
                name, "D vs bidir", dm, dc))

        # --- KS-MNTP-1 verdict on the TEXT stream, against the FROZEN bars ---
        bar = BARS[ds]
        acc = r["meanpool|text"][0]
        gap = bar["causal"] - bar["bidir"]
        rec = (acc - bar["bidir"]) / gap if gap else float("nan")
        if acc >= bar["bar50"]:
            verdict = "CONTINUE (>=50% recovery)"
        elif acc < bar["floor25"]:
            verdict = "KILL-side (<25% recovery)"
        else:
            verdict = "PARTIAL (25-50%)"
        print("KS-MNTP-1 text: acc {:.4f} | frozen bars causal {:.4f} bidir {:.4f} "
              "bar50 {:.4f} floor25 {:.4f}".format(
                  acc, bar["causal"], bar["bidir"], bar["bar50"], bar["floor25"]))
        print("KS-MNTP-1 recovery fraction = {:+.4f}  -> {}".format(rec, verdict))
        print("img null-op belt (S1 img vs banked bidir img): train {:.6f} dev {:.6f} "
              "bar 0.9999 -> {}".format(r["img_nullop_belt"]["train_cos"],
                                        r["img_nullop_belt"]["dev_cos"],
                                        "PASS" if r["img_nullop_belt"]["pass"] else "FAIL"))
        print("stream collapse cos(img,text) per arm:")
        for arm in ARMS:
            sc = r["stream_collapse"][arm]
            print("   {:16s} train {:.4f}  dev {:.4f}".format(
                arm, sc["train_img_text_cos"], sc["dev_img_text_cos"]))
        print("drift meanpool-vs-causal :", r["drift"]["meanpool_vs_causal"])
        print("drift meanpool-vs-bidir  :", r["drift"]["meanpool_vs_bidir"])
        blob[ds]["ks_mntp_1"] = {"text_acc": acc, "bars": bar,
                                 "recovery_fraction": rec, "verdict": verdict}

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mntp_s1_devscreen_OUT.json")
    with open(outp, "w") as fh:
        json.dump(blob, fh, indent=2)
    print("\nwrote {}".format(outp))
