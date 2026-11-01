"""Test-time check of the per-video answer offset (README section 3.1) on the r5 checkpoints, no retraining.

For each r5 test-selected trial: the offset answer model (qtl.answer_offset.fit_offset, label-free, training answers)
replaces the anchored table at inference; the batched EIG policy with the offset summed out (qtl.opolicy) runs up to
--max-calls on validation and test with the cached answers (off_real), with the copy likelihood as well
(off_real_copy), and with the shuffled answers of sim_monotone.py (off_shuf, the answers' dependence within a video
removed). Curves, qmixE1 budgets and group statistics as in sim_monotone.py; the r5 rows to compare with are in the
same summary directory (real_copy / real_indep). Rule-10 developmental evidence.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qtl import data as qdata, opolicy
from qtl.answer_offset import OffsetAnswerModel, fit_offset
from qtl.backbones import make_model
from qtl.checkpoint import load_trial
from qtl.copy_noise import bucket_of, estimate_pi_neg
import hier_evidence_common as hc
from sim_monotone import evaluate, mean_rows, shuffle                     # noqa: E402

SOURCES = ("off_real", "off_real_copy", "off_shuf")


def run_source(model, store, ids, oam, chain, answers, cfg, max_calls, device, copy_fn):
    runs = {}
    for split in ("val", "test"):
        vids = sorted(ids[split], key=lambda v: store.T[v])
        runs[split] = {}
        for i in range(0, len(vids), int(cfg["eval_chunk"])):
            with torch.no_grad():
                runs[split].update(opolicy.run_batch(model, store, vids[i:i + int(cfg["eval_chunk"])], oam, chain,
                                                     answers, max_calls, device, copy_pi=copy_fn))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--trial-dir", required=True)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--max-calls", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--budgets-cap", nargs="+", type=float, default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--budgets-max", nargs="+", type=float, default=[1, 2, 4, 6, 8, 12, 16, 24])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-root", default="runs/20261101_r5_completion/offset_check")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    hate = {v for v in ids["test"] if labels[v] == 1}
    gt_all = {**gt["val"], **gt["test"]}
    qdata.configure_source("soft_both", 8)
    answers, answer_T = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, ids["val"] + ids["test"], ["bert"])
    trial = Path(a.trial_dir).resolve()
    summary, cfg, model, am, chain = load_trial(trial, a.device, answers, ids, model_factory=make_model)
    assert cfg["answer_source"] == "soft_both" and not cfg["length_term"]
    ta = [(labels[v], [(x, y, answer_T[v], o) for (x, y), o in answers[v].items()]) for v in ids["train"]]
    theta, support, logw, info = fit_offset(ta)
    oam = OffsetAnswerModel(theta, support, logw, cfg["categories"])
    pi, *_ = estimate_pi_neg(answers, labels, ids["train"], answer_T, am, cfg["categories"])
    copy_fn = lambda length, p=pi: float(p[bucket_of(length)])  # noqa: E731
    name = f'seed{summary["seed"]}_{trial.name}'
    out = (ROOT / a.out_root / a.corpus / name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.pid").write_text(str(__import__("os").getpid()))
    print("host", socket.gethostname(), name, "offset fit", json.dumps(info), flush=True)
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "trial": str(trial), "offset_fit": info, "theta": theta.tolist(), "max_calls": a.max_calls,
              "sources": {}}
    for src in a.sources:
        rows = []
        for rep in range(1 if src != "off_shuf" else a.reps):
            rng = np.random.default_rng(1000 * int(summary["seed"]) + rep)
            ans = answers
            if src == "off_shuf":
                ans = {}
                for split in ("val", "test"):
                    ans.update(shuffle(ids[split], store, answers, labels, gt_all, rng))
            runs = run_source(model, store, ids, oam, chain, ans, cfg, a.max_calls, a.device,
                              copy_fn if src.endswith("copy") else None)
            rows.append(evaluate(runs, ids, labels, gt, hate, a.max_calls, a.budgets_cap, a.budgets_max))
            f = rows[-1]["fixed"]
            print(name, src, rep, " ".join("k%s %.3f/%.3f/%.3f" % (k, f[k]["pooled_ap"], f[k]["pooled_roc"],
                                                                  f[k]["within_roc"]) for k in ("0", "4", "8", "16", "32")
                                           if k in f), flush=True)
        report["sources"][src] = {"mean": mean_rows(rows)}
        (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    print("DONE", name, datetime.now().astimezone().isoformat(), flush=True)


if __name__ == "__main__":
    main()
