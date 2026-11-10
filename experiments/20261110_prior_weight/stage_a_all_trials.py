"""Stage A over ALL revision-6 trials (developmental, README section 5): selection-fair comparison of inference
variants. The revision-6 test-selected trials were chosen for revision 6's own inference, which biases any comparison
on those three checkpoints towards revision 6. Here every variant is replayed on all 20 trials of a seed (the
checkpoints of the revision-6 search, nothing retrained) and picks its own best trial per seed by (test AP + test
ROC) / 2, as rule 7 does; reported are that and the mean over all trials.

A variant is table|weights|policy[|budget]:
  table    r6 = the checkpoint's anchored table; val3 = three states with a length slope each, fitted on the
           validation per-second labels (answer_calib.py; no gradient);
  weights  none, or a mode of prior_weight.fit (label-free, validation answers): loc, joint, root_all;
  policy   eig (greedy, revision 6), plan_eer / plan_eig (budget-aware disjoint plan), level (equal windows);
  budget   calls per video (default 8).
Test AP / ROC / within, scores at 6 decimals, unified core evaluator (hc.frame_metrics).
Output runs/20261110_prior_weight/all_trials/<corpus>/seed<seed>/trial<k>.json (one file per trial; finished trials
are skipped on restart). Summary: all_trials_table.py.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, data as qdata, policy, prior_weight, qtree      # noqa: E402
from qtl.answer_calib import fit_validation_answer_model                # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

METS = ("pooled_ap", "pooled_roc", "within_roc")
POLICIES = {"eig": dict(order="eig"), "eer": dict(order="eer"), "level": dict(order="eig", level=True),
            "plan_eig": dict(order="eig", plan="disjoint"), "plan_eer": dict(order="eer", plan="disjoint"),
            "planfill_eig": dict(order="eig", plan="disjoint"), "planfill_eer": dict(order="eer", plan="disjoint")}
# plan_eig / plan_eer results in the trial files were computed before the plan fell back to nested nodes when no
# disjoint node is left (fewer calls than the budget on short videos); planfill_* = the current plan, all calls used.
DEFAULT = ["r6|none|eig", "r6|loc|eig", "r6|loc|level", "val3|none|eig", "val3|loc|eig", "val3|loc|level",
           "r6|loc|planfill_eig", "r6|loc|planfill_eer", "r6|none|planfill_eig", "val3|loc|planfill_eer"]
SRC = "runs/20261101_r6_length_table"


def replay(model, store, vids, am, chain, answers, cfg, budget, w, kw, device):
    runs = {}
    kw = dict(kw)
    level = kw.pop("level", False)
    depth = int(np.floor(np.log2(budget)))
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), int(cfg["eval_chunk"])):
        chunk = vids[i:i + int(cfg["eval_chunk"])]
        allowed = [set(qtree.depth_nodes(store.T[v], depth).tolist()) for v in chunk] if level else None
        with torch.no_grad():
            runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"], budget, device,
                                          allowed=allowed, prior_weight=None if w is None else
                                          prior_weight.as_tuple(w), **kw))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variants", nargs="+", default=DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-root", default="runs/20261110_prior_weight/all_trials")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    od = (ROOT / a.out_root / a.corpus / f"seed{a.seed}").resolve()
    od.mkdir(parents=True, exist_ok=True)
    (od / "run.pid").write_text(str(os.getpid()))
    print("host", socket.gethostname(), datetime.now().astimezone().isoformat(), "variants", a.variants, flush=True)
    trials = sorted((ROOT / SRC / a.corpus / f"seed{a.seed}").glob("trial*/model.pth"),
                    key=lambda p: int(p.parent.name[5:]))
    assert len(trials) == 20, "expected the 20 trials of the revision-6 search, found %d" % len(trials)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    val, test = list(ids["val"]), list(ids["test"])
    hate = {v for v in test if labels[v] == 1}
    qdata.configure_source("soft_both", 8)
    answers, T_all = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, val + test, ["bert"])

    def mets(sc):
        sc = {v: np.round(np.asarray(s, dtype=np.float64), 6) for v, s in sc.items()}
        m = hc.frame_metrics(sc, gt["test"], hate)
        return {k: float(m[k]) for k in METS}

    pri = {}
    plain_prior = policy.video_prior_nodes
    policy.video_prior_nodes = lambda model, store_, v, device: pri[v]   # the backbone runs once per video and trial
    for mp in trials:
        trial = mp.parent
        out = od / f"{trial.name}.json"
        done = json.loads(out.read_text()) if out.exists() else {"variants": {}}
        todo = [v for v in a.variants if v not in done["variants"]]
        if not todo:
            continue
        t0 = time.time()
        summary, cfg, model, am6, chain = load_trial(trial, a.device, answers, ids, model_factory=make_model)
        assert int(summary["seed"]) == a.seed and cfg["answer_source"] == "soft_both"
        pri.clear()
        with torch.no_grad():
            for v in val + test:
                pri[v] = plain_prior(model, store, v, a.device)
        tables, weights = {"r6": am6}, {}
        for var in todo:
            parts = var.split("|")
            tab, mode, pol = parts[:3]
            B = int(parts[3]) if len(parts) > 3 else 8
            if tab not in tables:
                assert tab == "val3"
                tables[tab], _counts = fit_validation_answer_model(answers, labels, gt["val"], val, T_all, am6.len_mu,
                                                                   am6.len_sd, cfg["categories"])
                tables[tab] = tables[tab].to(a.device)
            if (tab, mode) not in weights:
                weights[(tab, mode)] = None if mode == "none" else prior_weight.fit(
                    pri, store, val, tables[tab], chain, answers, a.device, mode=mode)
            w = weights[(tab, mode)]
            runs = replay(model, store, test, tables[tab], chain, answers, cfg, B, w, POLICIES[pol], a.device)
            res = mets({v: runs[v]["scores"][-1] for v in test})
            res["mean_calls"] = float(np.mean([len(runs[v]["asked"]) for v in test]))
            res["weights"] = w
            done["variants"][var] = res
        done.update(trial=str(trial), seed=a.seed, corpus=a.corpus, host=socket.gethostname(),
                    created=datetime.now().astimezone().isoformat(),
                    r6_summary_test={k: float(summary["test"][k]) for k in METS},
                    r6_val_criterion=float(summary["val_criterion"]))
        out.write_text(json.dumps(done, indent=1) + "\n")
        print(a.corpus, a.seed, trial.name, "%.0fs" % (time.time() - t0),
              json.dumps({v: [round(done["variants"][v][m], 4) for m in METS] for v in todo}), flush=True)
    print("DONE", a.corpus, a.seed, flush=True)


if __name__ == "__main__":
    main()
