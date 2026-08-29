"""DESC_CHANNEL step 2a -- encode descriptions / transcripts into archive-contract caches.

Frozen in idea-stage/DESC_CHANNEL_FREEZE.md sections 4 and 5.

Encoder, max_seq_length, pooling and output contract are identical to the project's existing
long-text transcript encoder scripts/generate_transcript_embedding.py:
    sentence-transformers/paraphrase-multilingual-mpnet-base-v2, max_seq_length=512,
    mean pooling, 768-d, cache = {"ids": [[id, ...]], "text_feats": FloatTensor[N,768],
    "labels": [...]}.

Writes idea-stage/desc_channel/feats/{split}_{ARM}.pt for ARM in
    T, B, G, Bmis, Gmis, N
with split in {train, dev_seen, test_seen}.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from defect import is_defect, load_gt  # noqa: E402
from gen_desc import FIELDS, OUT  # noqa: E402

ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MAX_SEQ_LEN = 512
PERM_SEED = 20260813
NOISE_SEED = 20260813
FEATS = os.path.join(HERE, "feats")
SPLIT_OUT = {"train": "train", "val": "dev_seen", "test": "test_seen"}
ARMS = ["T", "B", "G", "Bmis", "Gmis", "N"]

DESC_TMPL = ("Scene: {scene}\nPeople: {people}\nActions: {actions}\n"
             "On-screen text: {on_screen_text}\nFormat: {production_format}\n"
             "Audio cues: {audio_visible_cues}")


def desc_text(fields):
    if not fields:
        return ""
    return DESC_TMPL.format(**{k: (fields.get(k) or "") for k in FIELDS})


def load_desc():
    d = {}
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[r["id"]] = desc_text(r.get("fields"))
    return d


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="build caches from random vectors only (implementation smoke test)")
    ap.add_argument("--outdir", default=FEATS)
    a = ap.parse_args()

    gt = load_gt(ROOT)
    ids = sorted(gt)
    n = len(ids)
    row = {v: i for i, v in enumerate(ids)}

    desc = {v: "" for v in ids} if a.synthetic else load_desc()
    missing = [v for v in ids if v not in desc]
    if missing:
        raise SystemExit("descriptions missing for %d ids (first: %s)"
                         % (len(missing), missing[:5]))

    trans = {v: gt[v]["text"] for v in ids}
    defect = {v: is_defect(gt[v]["text"]) for v in ids}
    print("[defect] %d / %d videos flagged" % (sum(defect.values()), n))

    if a.synthetic:
        rng = np.random.default_rng(7)
        E_desc = torch.tensor(rng.normal(size=(n, 768)), dtype=torch.float)
        E_tran = torch.tensor(rng.normal(size=(n, 768)), dtype=torch.float)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(ENCODER)
        model.max_seq_length = MAX_SEQ_LEN
        print("[encoder] %s max_seq_length=%d" % (ENCODER, model.max_seq_length))
        E_desc = torch.tensor(model.encode([desc[v] for v in ids], batch_size=32,
                                           show_progress_bar=True, convert_to_numpy=True),
                              dtype=torch.float)
        E_tran = torch.tensor(model.encode([trans[v] for v in ids], batch_size=32,
                                           show_progress_bar=True, convert_to_numpy=True),
                              dtype=torch.float)
    print("[encode] desc %s  transcript %s" % (tuple(E_desc.shape), tuple(E_tran.shape)))

    rng = np.random.default_rng(PERM_SEED)
    perm = derangement(n, rng)                       # video i receives description of perm[i]
    E_desc_mis = E_desc[torch.tensor(perm, dtype=torch.long)]

    nrng = np.random.default_rng(NOISE_SEED)
    E_noise = torch.tensor(nrng.normal(size=(n, 768)), dtype=torch.float)
    E_noise = torch.nn.functional.normalize(E_noise, p=2, dim=1)

    dmask = torch.tensor([defect[v] for v in ids], dtype=torch.bool).unsqueeze(1)
    bank = {
        "T": E_tran,
        "B": E_desc,
        "G": torch.where(dmask, E_desc, E_tran),
        "Bmis": E_desc_mis,
        "Gmis": torch.where(dmask, E_desc_mis, E_tran),
        "N": E_noise,
    }
    # integrity: G must equal T off the defect set and B on it
    assert torch.equal(bank["G"][~dmask.squeeze(1)], E_tran[~dmask.squeeze(1)])
    assert torch.equal(bank["G"][dmask.squeeze(1)], E_desc[dmask.squeeze(1)])

    os.makedirs(a.outdir, exist_ok=True)
    for arm in ARMS:
        for sp, outname in SPLIT_OUT.items():
            sids = [v for v in ids if gt[v]["split"] == sp]
            rows = torch.tensor([row[v] for v in sids], dtype=torch.long)
            obj = {"ids": [sids],
                   "text_feats": bank[arm].index_select(0, rows).contiguous(),
                   "labels": [gt[v]["label"] for v in sids]}
            p = os.path.join(a.outdir, "%s_%s.pt" % (outname, arm))
            torch.save(obj, p)
        print("[write] arm %-5s -> %s/{train,dev_seen,test_seen}_%s.pt  dim=%d"
              % (arm, a.outdir, arm, bank[arm].shape[1]))


if __name__ == "__main__":
    main()
