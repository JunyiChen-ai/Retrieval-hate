"""PCD feasibility probe: is the frozen CLIP joint space usable for policy-clause directions?

TRAIN + VAL ONLY. No test file is opened (asserted below). Zero-shot, no training.
Purpose: (a) verify that projecting the cached pre-projection poolers through CLIP's
visual_projection / text_projection reproduces the CLIP joint space; (b) measure whether
a single hate anchor and a handful of paired policy clauses carry ANY zero-shot signal
in that space, before a pilot is specified.
"""
import os
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_ID = "openai/clip-vit-large-patch14-336"
TAG = "openai_clip-vit-large-patch14-336_HF"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(ds, split):
    assert "test" not in split, "probe is train/val only"
    p = os.path.join(ROOT, "data", "CLIP_Embedding", ds, f"{split}_{TAG}.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    ids = list(d["ids"][0]) if isinstance(d["ids"][0], list) else list(d["ids"])
    return ids, d["img_feats"].float(), d["text_feats"].float(), \
        torch.as_tensor(d["labels"]).float().view(-1).numpy()


@torch.no_grad()
def enc_sentences(model, tok, sents):
    """Same path as the cache: CLIPTextModel pooler_output -> text_projection."""
    out = []
    for s in sents:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        ids = [tok.bos_token_id] + ids[:75] + [tok.eos_token_id]
        t = torch.tensor([ids], device=DEV)
        h = model.text_model(input_ids=t, attention_mask=torch.ones_like(t)).pooler_output
        out.append(model.text_projection(h)[0].float().cpu())
    return torch.stack(out)


PAIRS = [
    ("a video that attacks a protected group with a slur",
     "a video that quotes a slur in order to condemn it"),
    ("a video that dehumanises an ethnic group as animals or vermin",
     "a video that gives a neutral factual description of an ethnic group"),
    ("a video that calls for violence against a religious group",
     "a video that condemns violence against a religious group"),
    ("a video that mocks people for their disability",
     "a video that documents discrimination against disabled people"),
    ("a video that praises a racist ideology",
     "a video that reports on a racist ideology as news"),
    ("a video that says women are inferior and should be controlled",
     "a video that criticises sexism"),
]


def main():
    model = CLIPModel.from_pretrained(CLIP_ID).to(DEV).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_ID)
    Wv, Wt = model.visual_projection, model.text_projection

    hate = enc_sentences(model, tok, ["a hateful video", "a video that attacks a group of people"])
    safe = enc_sentences(model, tok, ["a harmless video", "an ordinary video about everyday life"])
    anchor = torch.nn.functional.normalize(hate.mean(0) - safe.mean(0), dim=0)

    vio = enc_sentences(model, tok, [p[0] for p in PAIRS])
    exm = enc_sentences(model, tok, [p[1] for p in PAIRS])
    vio_n = torch.nn.functional.normalize(vio, dim=1)
    exm_n = torch.nn.functional.normalize(exm, dim=1)
    D = torch.nn.functional.normalize(vio_n - exm_n, dim=1)   # paired difference dirs
    print("pair-direction cosines (off-diag mean): %.3f" %
          float((D @ D.T - torch.eye(len(D))).abs().sum() / (len(D) * (len(D) - 1))))
    print("violation-vs-exemption cosine per pair:",
          [round(float(a @ b), 3) for a, b in zip(vio_n, exm_n)])

    for ds in ["ImpliHateVid", "HateMM", "MHC", "MHC_zh"]:
        rows = []
        for split in ["train", "dev_seen"]:
            _, img, txt, y = load(ds, split)
            with torch.no_grad():
                uv = torch.nn.functional.normalize(Wv(img.to(DEV)).cpu().float(), dim=1)
                ut = torch.nn.functional.normalize(Wt(txt.to(DEV)).cpu().float(), dim=1)
            u = torch.nn.functional.normalize(uv + ut, dim=1)
            res = {}
            for nm, rep in (("img", uv), ("txt", ut), ("sum", u)):
                res[f"anchor_{nm}"] = roc_auc_score(y, (rep @ anchor).numpy())
                P = (rep @ D.T).numpy()                       # [N, K] pair projections
                res[f"cone_max_{nm}"] = roc_auc_score(y, P.max(1))
                res[f"cone_mean_{nm}"] = roc_auc_score(y, P.mean(1))
                Pv = (rep @ vio_n.T).numpy()
                res[f"vio_only_max_{nm}"] = roc_auc_score(y, Pv.max(1))
            rows.append((split, res))
        print(f"\n=== {ds} (n_train/val) ===")
        keys = list(rows[0][1].keys())
        print("  " + "".join(f"{k:>18}" for k in keys))
        for split, res in rows:
            print(f"{split:>10}" + "".join(f"{res[k]:>18.4f}" for k in keys))


if __name__ == "__main__":
    main()
