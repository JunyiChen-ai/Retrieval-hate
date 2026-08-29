"""Sensitivity arm for the hate_video_95 all-zero row.

Builds R6RO-A0ZI = R6RO-A0 (== ro_L28 == deployed HateMM LoRA-curric cache) with the
single all-zero HateMM *train* row (hate_video_95) replaced by the l2-normalised mean
of the other 743 train rows. This is NOT the repair (the repair needs the LoRA adapter
and the raw video); it bounds how much ANY non-degenerate value in that slot can move
the head. dev/test caches are copied verbatim.
"""
import os, torch
E = "/home/jehc223/Retrieval-hate/data/CLIP_Embedding/HateMM"
def flat(ids):
    out = []
    for x in ids:
        out.extend(x) if isinstance(x, (list, tuple)) else out.append(x)
    return out
for split in ["train", "dev_seen", "test_seen"]:
    d = torch.load(os.path.join(E, "%s_R6RO-A0.pt" % split), map_location="cpu")
    if split == "train":
        ids = flat(d["ids"])
        i = ids.index("hate_video_95")
        for k in ["img_feats", "text_feats"]:
            v = d[k].float()
            assert float(v[i].norm()) == 0.0
            keep = torch.cat([v[:i], v[i+1:]], 0)
            m = keep.mean(0)
            v[i] = m / m.norm().clamp_min(1e-12)
            d[k] = v.contiguous()
        print("imputed row %d (%s)" % (i, ids[i]))
    torch.save(d, os.path.join(E, "%s_R6RO-A0ZI.pt" % split))
    print("wrote %s_R6RO-A0ZI.pt" % split)
