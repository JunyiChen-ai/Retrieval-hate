"""Parse and validate the exact cached inputs used by QTL; no content fingerprints."""
from datetime import datetime
from pathlib import Path
import socket

import numpy as np

from . import data as qdata, qtree
from macilsd import align
import hier_evidence_common as hc


def check_inputs(corpus):
    labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
    sets = {s: set(vs) for s, vs in ids.items()}
    assert all(len(sets[s]) == len(ids[s]) for s in ids)
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    qdata.configure_source("soft_both", 8)
    answers, answer_T = qdata.load_answers(corpus, "soft_both")
    _, _, raw_split = qdata.load_soft_p(corpus, "soft_both")
    video_ids = [v for split in ids.values() for v in split]
    assert all(v in answers and v in labels for v in video_ids)
    for split in ids:
        assert all(raw_split[v] == split for v in ids[split]), (corpus, split, "answer split mismatch")
    store = qdata.Store(corpus, video_ids, ["bert"])
    assert store.n_missing_text == 0, "Missing/malformed text cache"
    no_queryable = []
    for i, v in enumerate(video_ids):
        assert store.T[v] == answer_T[v]
        assert store.at[v].shape == (store.T[v], qdata.A_IN) and np.isfinite(store.at[v]).all()
        visual = align.load_visual(corpus, v)
        assert visual.ndim == 3 and visual.shape[-1] == align.V_DIM
        assert visual.shape[1] >= align.N_CROPS and np.isfinite(visual).all()
        assert store.W[v].shape[1] == visual.shape[0]
        tree = qtree.tree(store.T[v])
        assert all(pair in tree["index"] for pair in answers[v])
        observed = qdata.observed(answers[v], store.T[v])
        if np.any(tree["queryable"]):
            assert len(observed[0]) > 0, (v, "queryable video has no parsed observations")
        else:
            # Existing <4-second videos legally have no queryable nodes and use the prior at zero calls.
            no_queryable.append(v)
        for split in ("val", "test"):
            if v in sets[split]:
                assert v in gt[split] and len(gt[split][v]) == store.T[v]
        if (i + 1) % 200 == 0:
            print(f"inputs {corpus} {i+1}/{len(video_ids)}", flush=True)
    return {"host": socket.gethostname(), "corpus": corpus, "status": "parsed_and_covered",
            "checked_at": datetime.now().astimezone().isoformat(), "videos": len(video_ids),
            "splits": {s: len(v) for s, v in ids.items()}, "source": "soft_both", "soft_levels": 8,
            "soft_edges": qdata.SOFT_EDGES[(corpus, "soft_both")].tolist(), "missing_text": 0,
            "videos_without_queryable_nodes": no_queryable,
            "features": "I3D crops / VGGish / BERT through shared qtl.data.Store",
            "answers": str(Path(qdata.ROOT) / "data/vlm_tree" / qdata.CORPUS_DIR[corpus]),
            "checks": ["parsed shapes and finite arrays", "answer/feature/GT time lengths", "full cohort coverage",
                       "train/validation/test isolation", "answer split metadata", "question-tree node alignment"]}
