#!/usr/bin/env python
"""
Paper figures for the encoder-swap / B3-LoRA diagnosis (F44 + F45).

Outputs (research-wiki/figures/):
  fig_modality_auc.{pdf,png}     — per-modality kNN AUC (train-LOO) across the three
                                    datasets x arms {CLIP, frozen-Qwen-7B, Qwen-32B,
                                    LoRA-Qwen-7B(ZH only)}; foregrounds the MHC-EN Qwen
                                    image-stream collapse and that 32B does not fix it.
  fig_pareto_rotation.{pdf,png}  — Delta(non-hate recall) vs Delta(hate recall), TEST
                                    final-epoch 3-seed mean, for four cells; the Pareto
                                    (converts) vs rotation (doesn't) distinction.

Data provenance (no hand-typed numbers; everything re-read from banked artifacts):
  * AUC (CLIP/frozen-7B/32B, all datasets) <- committed encoder_swap_diagnosis_tables_out.json (T2)
  * ZH LoRA per-modality AUC              <- computed deterministically via encoder_swap_geometry
  * ZH frozen/LoRA/CLIP TEST recall       <- committed b3_zh_lora_trainlog_parse_out.json
  * HateMM / MHC-EN TEST recall           <- banked enc3s/arcbase trainlogs (final ep, 3-seed mean)
A fig_data.json snapshot is written next to the figures for audit.

Palette: Okabe-Ito (colourblind-safe, the scientific-figure standard).
Zero GPU, no new test evaluation (test read from already-logged trainlog lines).
"""
import os, sys, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from encoder_swap_geometry import load, build_modality, loo_knn, auc  # noqa: E402

ROOT = "/data/jehc223/RGCL"
FIGDIR = f"{ROOT}/research-wiki/figures"
LOG = f"{ROOT}/slurm/logs"
os.makedirs(FIGDIR, exist_ok=True)

# Okabe-Ito, assigned by ENTITY (arm), fixed order, never cycled.
COL = {
    "CLIP":       "#0072B2",  # blue
    "frozen-7B":  "#E69F00",  # orange
    "Qwen-32B":   "#CC79A7",  # reddish purple (well-separated from orange for CVD)
    "LoRA-7B":    "#009E73",  # bluish green
}
CONVERTS = "#009E73"   # Pareto / converts
ROTATION = "#D55E00"   # rotation / doesn't convert (vermillion)
INK = "#222222"; MUTED = "#666666"; GRID = "#DDDDDD"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
    "figure.dpi": 120,
})

# ---------------------------------------------------------------- data assembly
with open(f"{HERE}/encoder_swap_diagnosis_tables_out.json") as f:
    T2 = json.load(f)["T2"]
with open(f"{HERE}/b3_zh_lora_trainlog_parse_out.json") as f:
    B3 = json.load(f)

DS_KEYS = {"HateMM": "HateMM", "MHC-EN": "MHC-EN", "MHC-ZH": "MHC-ZH"}  # T2 keys

# ZH LoRA per-modality train-LOO AUC (deterministic recompute)
zh_lora = {}
ti, tt, ty, _ = load("MHC_zh", "Qwen2.5-VL-7B-Instruct-LoRA_HF", "train")
for m in ("img", "text", "concat"):
    X = build_modality(ti, tt, m); p, s = loo_knn(X, ty, 20); zh_lora[m] = float(auc(ty, s))

# assemble AUC table: ds -> arm -> {img,text,concat}
AUC = {}
for ds in ("HateMM", "MHC-EN", "MHC-ZH"):
    AUC[ds] = {
        "CLIP":      {m: T2[ds]["CLIP"][m] for m in ("img", "text", "concat")},
        "frozen-7B": {m: T2[ds]["Qwen7B"][m] for m in ("img", "text", "concat")},
        "Qwen-32B":  {m: T2[ds]["Qwen32B"][m] for m in ("img", "text", "concat")},
    }
AUC["MHC-ZH"]["LoRA-7B"] = zh_lora

# ---- TEST final-epoch 3-seed-mean per-class recall for HateMM / MHC-EN cells ----
reA = re.compile(r"Test_Retrieval Epoch\s+(\d+) acc: ([\d.]+) roc: ([\d.]+) pre: ([\d.]+) recall: ([\d.]+) f1: ([\d.]+)\s*$")
reB = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")


def final_recall(path):
    fin = -1; hate = macroR = acc = None
    for ln in open(path):
        m = reA.match(ln.strip())
        if m and int(m.group(1)) >= fin:
            fin = int(m.group(1)); hate = float(m.group(5))
        m = reB.match(ln.strip())
        if m and int(m.group(1)) >= fin:
            macroR = float(m.group(4)); acc = float(m.group(5))
    return acc, hate, 2 * macroR - hate  # acc, hate_recall, nonhate_recall


def mean_recall(paths):
    return np.mean([final_recall(p) for p in paths], axis=0)


arms_hm_clip = [f"{LOG}/enc3s_HateMM_openai_clip-vit-large-patch14-336_HF_seed{s}_12850.trainlog" for s in (0, 1, 2)]
arms_hm_qw = [f"{LOG}/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed{s}_12850.trainlog" for s in (0, 1, 2)]
arms_en_clip = [f"{LOG}/enc3s_MHC_openai_clip-vit-large-patch14-336_HF_seed{s}_12850.trainlog" for s in (0, 1, 2)]
arms_en_qw = [f"{LOG}/enc3s_MHC_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog",
              f"{LOG}/arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed1_12275.trainlog",
              f"{LOG}/arcbase_MHC_Qwen2.5-VL-7B-Instruct_HF_seed2_12276.trainlog"]

hm_c, en_c = mean_recall(arms_hm_clip), mean_recall(arms_en_clip)
hm_q, en_q = mean_recall(arms_hm_qw), mean_recall(arms_en_qw)

# ZH cells from committed B3 json (final_mean)
zc = B3["CLIP"]["final_mean"]; zf = B3["frozenQwen"]["final_mean"]; zl = B3["LoRAQwen"]["final_mean"]

CELLS = [  # (label, d_hate, d_nonhate, d_acc, shape, (dx,dy,ha,va))
    ("HateMM: frozen-Qwen vs CLIP", hm_q[1]-hm_c[1], hm_q[2]-hm_c[2], hm_q[0]-hm_c[0], "Pareto",
     (-0.004, 0.010, "right", "bottom")),
    ("MHC-ZH: LoRA-Qwen vs CLIP",  zl["Test_hate_recall"]-zc["Test_hate_recall"],
                                    zl["Test_nonhate_recall"]-zc["Test_nonhate_recall"],
                                    zl["Test_acc"]-zc["Test_acc"], "Pareto",
     (-0.004, 0.011, "right", "bottom")),
    ("MHC-EN: frozen-Qwen vs CLIP", en_q[1]-en_c[1], en_q[2]-en_c[2], en_q[0]-en_c[0], "rotation",
     (-0.004, -0.011, "right", "top")),
    ("MHC-ZH: frozen-Qwen vs CLIP", zf["Test_hate_recall"]-zc["Test_hate_recall"],
                                    zf["Test_nonhate_recall"]-zc["Test_nonhate_recall"],
                                    zf["Test_acc"]-zc["Test_acc"], "rotation",
     (0.004, -0.011, "left", "top")),
]

with open(f"{FIGDIR}/fig_data.json", "w") as f:
    json.dump({"AUC": AUC,
               "pareto_rotation_cells": [c[:5] for c in CELLS],
               "note": "AUC=train-LOO kNN; recall deltas=TEST final-epoch 3-seed mean"}, f, indent=1)

# ============================== FIGURE (a) ==============================
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
mods = ["img", "text", "concat"]
arm_order = ["CLIP", "frozen-7B", "Qwen-32B", "LoRA-7B"]
for ax, ds in zip(axes, ("HateMM", "MHC-EN", "MHC-ZH")):
    present = [a for a in arm_order if a in AUC[ds]]
    n = len(present); w = 0.8 / n
    for j, arm in enumerate(present):
        vals = [AUC[ds][arm][m] for m in mods]
        xs = np.arange(len(mods)) + (j - (n - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, color=COL[arm], label=arm,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(2.42, 0.505, "chance", fontsize=6.5, color=MUTED, va="bottom", ha="right")
    ax.set_title(ds, fontsize=10, color=INK)
    ax.set_xticks(range(len(mods))); ax.set_xticklabels(["image", "text", "concat"])
    ax.set_ylim(0.45, 0.96)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("kNN AUC  (train-LOO, k=20)")
# collapse callout on MHC-EN image group
axc = axes[1]
axc.annotate("Qwen image stream\ncollapses to chance;\n32B does not fix it",
             xy=(0.0, 0.60), xytext=(0.30, 0.905), fontsize=6.8, color=ROTATION,
             ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color=ROTATION, lw=1.0))
handles = [Patch(facecolor=COL[a], label={"CLIP": "frozen CLIP", "frozen-7B": "frozen Qwen-7B",
          "Qwen-32B": "frozen Qwen-32B", "LoRA-7B": "LoRA Qwen-7B (ZH)"}[a]) for a in arm_order]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.06), fontsize=8.5)
fig.suptitle("Per-modality frozen-feature separability: the text stream lifts uniformly, "
             "the image stream is dataset-specific", y=1.13, fontsize=9.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 1.0))
for ext in ("pdf", "png"):
    fig.savefig(f"{FIGDIR}/fig_modality_auc.{ext}", bbox_inches="tight", dpi=200)
plt.close(fig)

# ============================== FIGURE (b) ==============================
fig, ax = plt.subplots(figsize=(6.6, 5.4))
xmax = max(c[1] for c in CELLS) * 1.20
ymin = min(c[2] for c in CELLS) - 0.02; ymax = max(c[2] for c in CELLS) + 0.02
# majority-recall PRESERVED (y>=0) vs SACRIFICED (y<=-0.02) regions; the
# empirical gap between the two clusters (-0.003 vs -0.033) is left neutral so
# the ~zero LoRA point (-0.0032) is not mis-shaded as "sacrificed".
GAP = -0.02
ax.axhspan(0.0, ymax + 0.06, color=CONVERTS, alpha=0.07, zorder=0)
ax.axhspan(ymin - 0.06, GAP, color=ROTATION, alpha=0.07, zorder=0)
ax.axhline(0, color=INK, lw=1.1, ls=(0, (5, 2)), zorder=2)
ax.text(0.001, 0.0016, "strict Pareto frontier (majority recall unchanged)",
        fontsize=6.6, color=MUTED, ha="left", va="bottom")
ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
ax.text(xmax * 0.985, ymax + 0.006, "majority (non-hate) recall preserved  →  Pareto, converts",
        fontsize=7.6, color=CONVERTS, ha="right", va="bottom", fontweight="bold")
ax.text(xmax * 0.985, ymin - 0.006, "majority recall sacrificed  →  rotation, no accuracy gain",
        fontsize=7.6, color=ROTATION, ha="right", va="top", fontweight="bold")
for label, dh, dn, da, shape, (dx, dy, ha, va) in CELLS:
    c = CONVERTS if shape == "Pareto" else ROTATION
    ax.scatter([dh], [dn], s=130, color=c, edgecolor="white", linewidth=1.3, zorder=5)
    ax.annotate(f"{label}\n$\\Delta$acc = {da:+.4f}", xy=(dh, dn),
                xytext=(dh + dx, dn + dy), fontsize=7.7, color=INK, va=va, ha=ha, zorder=6)
ax.set_xlim(-0.006, xmax); ax.set_ylim(ymin - 0.055, ymax + 0.06)
ax.set_xlabel(r"$\Delta$ minority (hate) recall   vs frozen CLIP")
ax.set_ylabel(r"$\Delta$ majority (non-hate) recall   vs frozen CLIP")
ax.grid(color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
legend = [Patch(facecolor=CONVERTS, label="Pareto  (converts to +acc/+mF1)"),
          Patch(facecolor=ROTATION, label="rotation (AUC up, acc flat)")]
ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=8)
ax.set_title("Every encoder change lifts minority recall — only the ones that\n"
             "preserve majority recall convert to accuracy  (TEST, final-epoch, 3-seed mean)",
             fontsize=9.2, color=INK)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{FIGDIR}/fig_pareto_rotation.{ext}", bbox_inches="tight", dpi=200)
plt.close(fig)

print("wrote figures to", FIGDIR)
for fn in sorted(os.listdir(FIGDIR)):
    print("  ", fn, os.path.getsize(f"{FIGDIR}/{fn}"), "bytes")
print("\nPareto/rotation cells (TEST final-ep 3-seed mean):")
for label, dh, dn, da, shape, _off in CELLS:
    print(f"  {shape:8s} {label:32s} dHate={dh:+.4f} dNonhate={dn:+.4f} dAcc={da:+.4f}")
print("\nMHC-EN image AUC collapse: CLIP %.3f -> 7B %.3f -> 32B %.3f"
      % (AUC["MHC-EN"]["CLIP"]["img"], AUC["MHC-EN"]["frozen-7B"]["img"], AUC["MHC-EN"]["Qwen-32B"]["img"]))
print("ZH LoRA text AUC: CLIP %.3f -> 7B %.3f -> LoRA %.3f"
      % (AUC["MHC-ZH"]["CLIP"]["text"], AUC["MHC-ZH"]["frozen-7B"]["text"], AUC["MHC-ZH"]["LoRA-7B"]["text"]))
