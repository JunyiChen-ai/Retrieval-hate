"""Review round 1 (README): replay the revision-6 policy of given trials on the cached answers and save the runs.

For each trial directory: cpolicy.run_batch with the checkpoint's own answer table (no copy likelihood, the method)
up to --max-calls calls on the requested splits, once per --levels entry: 0 = every node may be asked (the method);
L > 0 = only the nodes of one tree depth whose windows are L to 2L seconds (qtree.level_nodes; control E3). Writes
<out-root>/<corpus>/<trial name>/runs_L<level>.pkl = {"runs": {split: {video id: run}}, "trial": ..., "seed": ...}.
Nothing is trained or scored here; E1 / E3 / E4 read the pickles.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import pickle
import socket
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata                                           # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
from qtl.replay import run_source                                       # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--trial-dirs", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=("train", "val", "test"))
    ap.add_argument("--levels", nargs="+", type=int, default=[0])
    ap.add_argument("--max-calls", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-root", default="runs/20261108_astra_review/replay")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    qdata.configure_source("soft_both", 8)
    answers, _ = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, [v for s in a.splits for v in ids[s]], ["bert"])
    for td in a.trial_dirs:
        trial = Path(td).resolve()
        summary, cfg, model, am, chain = load_trial(trial, "cpu", answers, ids, model_factory=make_model)
        assert cfg["answer_source"] == "soft_both" and int(cfg["soft_levels"]) == 8, cfg["answer_source"]
        assert cfg.get("soft_fixed_edges") is None, "replay assumes the method's train-quantile answer levels"
        name = f'seed{summary["seed"]}_{trial.name}'
        od = (ROOT / a.out_root / a.corpus / name).resolve()
        od.mkdir(parents=True, exist_ok=True)
        (od / "run.pid").write_text(str(os.getpid()))
        for L in a.levels:
            path = od / f"runs_L{L}.pkl"
            prev = pickle.load(path.open("rb"))["runs"] if path.exists() else {}
            todo = [s for s in a.splits if s not in prev]
            if not todo:
                print("skip", name, "L%d" % L, "(done)", flush=True)
                continue
            runs = run_source(model, store, ids, am, chain, answers, cfg, a.max_calls, "cpu", None, splits=todo,
                              level=L)
            prev.update(runs)
            with path.open("wb") as fh:
                pickle.dump({"runs": prev, "trial": str(trial), "seed": summary["seed"], "level": L,
                             "max_calls": a.max_calls, "host": socket.gethostname(),
                             "created": datetime.now().astimezone().isoformat()}, fh)
            print(datetime.now().astimezone().isoformat(), a.corpus, name, "L%d" % L, "splits", todo, flush=True)
    print("DONE", a.corpus, datetime.now().astimezone().isoformat(), flush=True)


if __name__ == "__main__":
    main()
