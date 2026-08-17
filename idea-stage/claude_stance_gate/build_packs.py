"""CLAUDE_STANCE_GATE -- build one anonymised data pack per eval item.

Pack: pack/item_XXX/transcript.txt  + frame_1.jpg .. frame_8.jpg (frame-bearing items only).
Frames match run_pilot.frame_urls exactly: all 8 existing frames in temporal order,
max side 512, JPEG quality 80.

The item_XXX <-> (dataset, video_id) mapping goes to manifest.json, which annotators
never see. Order is shuffled with a fixed seed so pack index leaks nothing.
"""
import json
import os
import random
import re
import shutil
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SP = os.path.join(ROOT, "idea-stage", "stance_pilot")
sys.path.insert(0, SP)
from run_pilot import load_texts  # noqa: E402

MAX_SIDE, JPEG_Q, SEED = 512, 80, 20260817
TITLE_DS = {"MHC", "MHC_zh"}


def main():
    sample = json.load(open(os.path.join(SP, "sample.json")))["eval"]
    order = list(range(len(sample)))
    random.Random(SEED).shuffle(order)

    texts = {ds: load_texts(ds) for ds in sorted({e["dataset"] for e in sample})}
    packdir = os.path.join(HERE, "pack")
    if os.path.isdir(packdir):
        shutil.rmtree(packdir)
    os.makedirs(packdir)

    manifest = []
    for slot, idx in enumerate(order, start=1):
        e = sample[idx]
        ds, vid = e["dataset"], e["id"]
        name = f"item_{slot:03d}"
        d = os.path.join(packdir, name)
        os.makedirs(d)

        t = (texts[ds].get(vid) or "").strip() or "(empty)"
        open(os.path.join(d, "transcript.txt"), "w", encoding="utf-8").write(t + "\n")

        src = os.path.join(ROOT, "data", "lora_frames", ds, vid)
        nf = 0
        if os.path.isdir(src):
            fs = sorted(os.listdir(src), key=lambda x: int(re.findall(r"\d+", x)[0]))
            for i, f in enumerate(fs, start=1):
                im = Image.open(os.path.join(src, f)).convert("RGB")
                if max(im.size) > MAX_SIDE:
                    s = MAX_SIDE / max(im.size)
                    im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))))
                im.save(os.path.join(d, f"frame_{i}.jpg"), "JPEG", quality=JPEG_Q)
                nf += 1
        manifest.append({"item": name, "dataset": ds, "id": vid, "group": e["group"],
                         "n_frames": nf, "has_title": ds in TITLE_DS,
                         "transcript_chars": len(t)})

    json.dump({"seed": SEED, "items": manifest}, open(os.path.join(HERE, "manifest.json"), "w"),
              indent=1, ensure_ascii=False)
    print("packs:", len(manifest), "with frames:", sum(1 for m in manifest if m["n_frames"]))


if __name__ == "__main__":
    main()
