"""KS-MNTP-1 for the S1b bidir+TEXT-POSITIONS-ONLY arm, plus the declared collapse belt.

Four arms side by side, all DEV, raw untrained key space, no head, no test read:
  causal    — banked, deployed EOS-class text readout
  bidir     — banked F72 arm (mask flipped, readout unchanged)
  meanpool  — S1  (mask flipped, text readout = mean over ALL positions)  [collapsed, cos 0.93]
  textpool  — S1b (mask flipped, text readout = mean over TEXT positions only)

`load_cache` / `l2` / `knn_vote` are IMPORTED from the frozen recon screen
(sha 8bc009e68833d8bad3aecb531c7c8b9879e05a2e00430465e0b2b4f05f9dede0) so the vote operator
is byte-identical to the one that produced every prior number, and that file's sha is unchanged.

Bars are FROZEN (MNTP_FORENSIC_RECON.md §5.2) and the collapse bar was declared in
MNTP_S1_RECORD.md §6b.3 BEFORE this arm was built. Neither is recomputed from the new data.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mntp_rawkey_devscreen import knn_vote, l2, load_cache  # noqa: E402

# FROZEN KS-MNTP-1 bars — recon §5.2, text stream, dev, raw key space.
BARS = {
    "HateMM": {"causal": 0.8037, "bidir": 0.7570, "bar50": 0.7804, "floor25": 0.7687},
    "MHC_zh": {"causal": 0.8462, "bidir": 0.6282, "bar50": 0.7372, "floor25": 0.6827},
}
# DECLARED IN ADVANCE — MNTP_S1_RECORD.md §6b.3 item 2. cos(text, img) must stay in the causal
# regime. S1 self-refuted at 0.9273/0.9320; causal is 0.3105-0.3523.
COLLAPSE_BAR = 0.60

CELLS = [
    ("HateMM", {
        "causal": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
        "bidir": "Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF",
        "meanpool": "Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-meanpool_HF",
        "textpool": "Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-textpool_HF",
    }),
    ("MHC_zh", {
        "causal": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
        "bidir": "Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF",
        "meanpool": "Qwen2.5-VL-7B-Instruct-LoRA-bidir-meanpool_HF",
        "textpool": "Qwen2.5-VL-7B-Instruct-LoRA-bidir-textpool_HF",
    }),
]
ARMS = ("causal", "bidir", "meanpool", "textpool")
LABEL = {"causal": "causal", "bidir": "bidir-lasttoken",
         "meanpool": "S1 bidir-MEANPOOL", "textpool": "S1b bidir-TEXTPOOL"}


def cos(a, b):
    """Mean per-item cosine over DECODABLE rows only.

    The zero-vector guard writes an all-zeros row for any undecodable video (HateMM train
    `hate_video_95`). Zeros-vs-zeros is a DEGENERATE pair: the arms agree perfectly but cosine
    is undefined and torch returns 0.0, which drags the mean and can masquerade as a real
    difference. This exclusion is the S1 belt-design erratum fix, carried over as declared.
    """
    keep = (a.norm(dim=1) > 0) & (b.norm(dim=1) > 0)
    if int(keep.sum()) == 0:
        return float("nan")
    return float((l2(a[keep]) * l2(b[keep])).sum(1).mean())


def run(ds, tags):
    feats, out = {}, {}
    ref = None
    for arm in ARMS:
        tr_ids, tr_i, tr_t, tr_y = load_cache(ds, "train", tags[arm])
        dv_ids, dv_i, dv_t, dv_y = load_cache(ds, "dev_seen", tags[arm])
        if ref is None:
            ref = (list(tr_ids), list(dv_ids))
        else:
            assert list(tr_ids) == ref[0], "train id order differs in arm " + arm
            assert list(dv_ids) == ref[1], "dev id order differs in arm " + arm
        feats[arm] = (tr_i, tr_t, dv_i, dv_t)
        streams = {
            "img": (l2(tr_i), l2(dv_i)),
            "text": (l2(tr_t), l2(dv_t)),
            "concat": (l2(torch.cat([l2(tr_i), l2(tr_t)], 1)),
                       l2(torch.cat([l2(dv_i), l2(dv_t)], 1))),
        }
        for name, (M, Q) in streams.items():
            out["{}|{}".format(arm, name)] = knn_vote(Q, M, tr_y, dv_y)

    out["stream_collapse"] = {
        a: {"train": cos(f[0], f[1]), "dev": cos(f[2], f[3])} for a, f in feats.items()
    }
    b, tp = feats["bidir"], feats["textpool"]
    out["drift_textpool_vs_bidir"] = {
        "train_img": cos(tp[0], b[0]), "train_text": cos(tp[1], b[1]),
        "dev_img": cos(tp[2], b[2]), "dev_text": cos(tp[3], b[3]),
    }
    c = feats["causal"]
    out["drift_textpool_vs_causal"] = {
        "train_img": cos(tp[0], c[0]), "train_text": cos(tp[1], c[1]),
        "dev_img": cos(tp[2], c[2]), "dev_text": cos(tp[3], c[3]),
    }
    out["img_nullop_belt"] = {
        "train_cos": out["drift_textpool_vs_bidir"]["train_img"],
        "dev_cos": out["drift_textpool_vs_bidir"]["dev_img"],
        "bar": 0.9999,
        "pass": (out["drift_textpool_vs_bidir"]["train_img"] >= 0.9999
                 and out["drift_textpool_vs_bidir"]["dev_img"] >= 0.9999),
    }
    return out


if __name__ == "__main__":
    blob = {}
    for ds, tags in CELLS:
        r = run(ds, tags)
        blob[ds] = {"tags": tags, **r}
        print("\n=== {} (DEV only, raw untrained key space, no head, no test) ===".format(ds))
        print("{:8s} {:20s} {:>8s} {:>8s} {:>8s}".format("stream", "arm", "acc", "mF1", "roc"))
        for name in ("img", "text", "concat"):
            for arm in ARMS:
                a, f, ro = r["{}|{}".format(arm, name)]
                print("{:8s} {:20s} {:8.4f} {:8.4f} {:8.4f}".format(name, LABEL[arm], a, f, ro))
            d_b = r["textpool|{}".format(name)][0] - r["bidir|{}".format(name)][0]
            d_c = r["textpool|{}".format(name)][0] - r["causal|{}".format(name)][0]
            d_s1 = r["textpool|{}".format(name)][0] - r["meanpool|{}".format(name)][0]
            print("{:8s} {:20s} vs bidir {:+.4f} | vs causal {:+.4f} | vs S1 {:+.4f}".format(
                name, "S1b DELTAS", d_b, d_c, d_s1))

        # --- BELT 1 (declared in advance): stream collapse ---
        sc = r["stream_collapse"]
        tp_cos = max(sc["textpool"]["train"], sc["textpool"]["dev"])
        collapse_pass = tp_cos < COLLAPSE_BAR
        print("\ncollapse belt cos(text,img)  bar < {:.2f} (declared §6b.3)".format(COLLAPSE_BAR))
        for arm in ARMS:
            print("   {:20s} train {:.4f}  dev {:.4f}".format(
                LABEL[arm], sc[arm]["train"], sc[arm]["dev"]))
        print("   => S1b worst {:.4f} -> {}".format(
            tp_cos, "PASS (stayed in the causal regime)" if collapse_pass
            else "FAIL — ARM SELF-REFUTES regardless of accuracy"))

        # --- BELT 2: img null-op ---
        nb = r["img_nullop_belt"]
        print("img null-op belt (S1b img vs banked bidir img): train {:.6f} dev {:.6f} "
              "bar 0.9999 -> {}".format(nb["train_cos"], nb["dev_cos"],
                                        "PASS" if nb["pass"] else "FAIL"))

        # --- KS-MNTP-1 against the FROZEN bars ---
        bar = BARS[ds]
        acc = r["textpool|text"][0]
        gap = bar["causal"] - bar["bidir"]
        rec = (acc - bar["bidir"]) / gap if gap else float("nan")
        if acc >= bar["bar50"]:
            verdict = "CONTINUE (>=50% recovery)"
        elif acc < bar["floor25"]:
            verdict = "KILL-side (<25% recovery)"
        else:
            verdict = "PARTIAL (25-50%)"
        print("KS-MNTP-1 text: acc {:.4f} | frozen bars causal {:.4f} bidir {:.4f} "
              "bar50 {:.4f} floor25 {:.4f}".format(acc, bar["causal"], bar["bidir"],
                                                   bar["bar50"], bar["floor25"]))
        print("KS-MNTP-1 recovery fraction = {:+.4f}  -> {}".format(rec, verdict))
        print("drift S1b-vs-causal:", r["drift_textpool_vs_causal"])
        print("drift S1b-vs-bidir :", r["drift_textpool_vs_bidir"])
        blob[ds]["ks_mntp_1"] = {"text_acc": acc, "bars": bar, "recovery_fraction": rec,
                                 "verdict": verdict}
        blob[ds]["collapse_belt"] = {"bar": COLLAPSE_BAR, "s1b_worst": tp_cos,
                                     "pass": collapse_pass}

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mntp_s1b_devscreen_OUT.json")
    with open(p, "w") as fh:
        json.dump(blob, fh, indent=2)
    print("\nwrote {}".format(p))
