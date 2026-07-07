#!/usr/bin/env python3
"""P9b (option a): derive 4-frame yn data from the existing 8-frame yn json by subsampling
frames [0,2,4,6] (the 8 were uniformly sampled, so this is 4 uniformly-spaced frames — no
re-extraction). Fixes the <image> count (8->4) and the '8 frames' wording, writes <split>_yn4.json,
and registers <prefix>_lora_yn4_<split> in the fork's dataset_info.json.

Motivation: at 8 frames the vision tower OOMs above bs=1, which zeroes RGCL's in-batch contrastive
term (see EXP_p9b). 4 frames lets physical bs>=4 fit, restoring the in-batch negatives.
"""
import json, os, sys, copy

RGCL = "/data/jehc223/RGCL"
LF = os.path.join(RGCL, "RA-HMD/LLAMA-FACTORY-Ver202512")
DINFO = os.path.join(LF, "data/dataset_info.json")
KEEP = [0, 2, 4, 6]  # subsample indices out of 0..7
PREFIX = {"MHC": "mhc", "MHC_zh": "mhc_zh", "HateMM": "hatemm"}


def convert(ds, split):
    src = os.path.join(RGCL, f"data/lora_sft/{ds}/{split}_yn.json")
    if not os.path.exists(src):
        return None
    data = json.load(open(src))
    out = []
    for rec in data:
        r = copy.deepcopy(rec)
        imgs = r.get("images", [])
        if len(imgs) >= 8:
            r["images"] = [imgs[i] for i in KEEP]
        for m in r["messages"]:
            if m["role"] == "user":
                # collapse <image>*8 -> <image>*4 (rebuild from the trailing text)
                c = m["content"]
                n_img = c.count("<image>")
                text = c.replace("<image>", "")
                m["content"] = "<image>" * len(KEEP) + text
                m["content"] = m["content"].replace("8 frames", f"{len(KEEP)} frames")
        out.append(r)
    dst = os.path.join(RGCL, f"data/lora_sft/{ds}/{split}_yn4.json")
    json.dump(out, open(dst, "w"), ensure_ascii=False)
    return dst, len(out)


def register(ds, split, path):
    info = json.load(open(DINFO))
    key = f"{PREFIX[ds]}_lora_yn4_{split}"
    info[key] = {
        "file_name": path,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant"},
    }
    json.dump(info, open(DINFO, "w"), ensure_ascii=False, indent=2)
    return key


if __name__ == "__main__":
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["MHC", "MHC_zh"]
    for ds in datasets:
        for split in ("train", "val", "test"):
            r = convert(ds, split)
            if r is None:
                print(f"  [{ds}] {split}: no source yn json, skip")
                continue
            dst, n = r
            key = register(ds, split, dst)
            print(f"  [{ds}] {split}: n={n} -> {os.path.basename(dst)} (registered {key})")
