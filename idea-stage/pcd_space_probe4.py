"""PCD feasibility probe 4: (a) clause-geometry statistics in CLIP space;
(b) the same clause-vs-random test in a genuine multilingual SENTENCE-embedding space
(paraphrase-multilingual-mpnet-base-v2), which is the cached transcript encoder for
MHC-EN and MHC-ZH. Forecloses "CLIP is simply a bad text encoder for policy clauses".
TRAIN + VAL ONLY. No test file is opened.
"""
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_ID = "openai/clip-vit-large-patch14-336"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from pcd_space_probe import PAIRS, enc_sentences  # noqa: E402

ZH_PAIRS = [
    ("一个用侮辱性称呼攻击特定族群的视频", "一个引用侮辱性称呼并加以谴责的视频"),
    ("一个把某个民族比作动物或害虫的视频", "一个中立客观描述某个民族的视频"),
    ("一个煽动对某宗教群体施加暴力的视频", "一个谴责针对宗教群体的暴力的视频"),
    ("一个嘲笑残障人士的视频", "一个记录残障人士遭受歧视的视频"),
    ("一个赞美种族主义意识形态的视频", "一个以新闻方式报道种族主义意识形态的视频"),
    ("一个宣称女性低人一等应被管束的视频", "一个批评性别歧视的视频"),
]


def nz(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.nan_to_num(X / np.maximum(n, 1e-12))


def geometry_stats(vio, exm, name):
    v, e = nz(vio), nz(exm)
    d = nz(v - e)
    K = len(v)
    off = lambda M: (M.sum() - np.trace(M)) / (K * (K - 1))
    print(f"[{name}] mean cos(violation_k, exemption_k) = {np.mean((v * e).sum(1)):.3f}")
    print(f"[{name}] mean off-diag cos among violations   = {off(v @ v.T):.3f}")
    print(f"[{name}] mean off-diag cos among exemptions   = {off(e @ e.T):.3f}")
    print(f"[{name}] mean off-diag cos among differences  = {off(d @ d.T):.3f}")
    print(f"[{name}] mean ||v-e|| (unit v,e)              = "
          f"{np.mean(np.linalg.norm(v - e, axis=1)):.3f}")


def mpnet_arm():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    m.max_seq_length = 512
    rng = np.random.default_rng(20260810)
    for ds, pairs in (("MHC", PAIRS), ("MHC_zh", ZH_PAIRS)):
        vio = m.encode([p[0] for p in pairs], convert_to_numpy=True).astype(np.float64)
        exm = m.encode([p[1] for p in pairs], convert_to_numpy=True).astype(np.float64)
        geometry_stats(vio, exm, f"mpnet/{ds}")
        packs = {}
        for split in ["train", "dev_seen"]:
            p = os.path.join(ROOT, "data", "CLIP_Embedding", ds,
                             f"{split}_transcript_mpnet512_HF.pt")
            d = torch.load(p, map_location="cpu", weights_only=False)
            packs[split] = (nz(d["text_feats"].numpy().astype(np.float64)),
                            torch.as_tensor(d["labels"]).float().view(-1).numpy())

        def run(dv, de):
            out = []
            for split in ["train", "dev_seen"]:
                u, y = packs[split]
                out.append((np.concatenate([u @ nz(dv).T, u @ nz(de).T], 1), y))
            lr = LogisticRegression(max_iter=5000, C=1.0).fit(*out[0])
            return roc_auc_score(out[1][1], lr.decision_function(out[1][0]))

        K, D = len(pairs), vio.shape[1]
        rs = [run(rng.normal(size=(K, D)), rng.normal(size=(K, D))) for _ in range(5)]
        u, y = packs["train"]
        uv, yv = packs["dev_seen"]
        full = roc_auc_score(yv, LogisticRegression(max_iter=5000, C=0.1)
                             .fit(u, y).decision_function(uv))
        print(f"  {ds}: clause_2K={run(vio, exm):.4f}  vio_only={run(vio, vio):.4f}  "
              f"random_2K={np.mean(rs):.4f} [{min(rs):.4f},{max(rs):.4f}]  "
              f"FULL_mpnet_LR={full:.4f}\n")


def main():
    model = CLIPModel.from_pretrained(CLIP_ID).to(DEV).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_ID)
    geometry_stats(enc_sentences(model, tok, [p[0] for p in PAIRS]).numpy(),
                   enc_sentences(model, tok, [p[1] for p in PAIRS]).numpy(), "CLIP joint")
    print()
    del model
    torch.cuda.empty_cache()
    mpnet_arm()


if __name__ == "__main__":
    main()
