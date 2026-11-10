"""Stage A over ALL revision-6 trials (developmental, README section 5): selection-fair comparison of inference
variants. The revision-6 test-selected trials were chosen for revision 6's own inference, which biases any comparison
on those three checkpoints towards revision 6. Here every variant is replayed on all 20 trials of a seed (the
checkpoints of the revision-6 search, nothing retrained) and picks its own best trial per seed by (test AP + test
ROC) / 2, as rule 7 does; reported are that and the mean over all trials.

A variant is table|weights|policy[|budget]:
  table    r6 = the checkpoint's anchored table; val3 = three states with a length slope each, fitted on the
           validation per-second labels (answer_calib.py; no gradient); val3h = val3 whose state 0 (no hate in the
           video) is the anchored table of the training videos (the validation set has few negative videos on HCS);
  weights  none, or a mode of prior_weight.fit (label-free, validation answers): loc, joint, root_all, trust
           (location weight plus the answer leak / miss scalars), trust_len;
  policy   eig (greedy, revision 6), plan_eer / plan_eig (budget-aware disjoint plan), level (equal windows);
           stopE1c<cap> = greedy EIG with the qmixE1 stopping rule (stop_rules.py; threshold set on the validation
           videos, no labels, so that their mean number of calls is the budget; at most <cap> calls per video);
  budget   calls per video (default 8); for stopE1c<cap> the MEAN number of calls per video.
A variant ending in "@val" is replayed on the validation videos instead of the test videos (metrics of the validation
split; used by policy_select.py to choose the question policy on validation). Not available for stopE1c<cap>.
AP / ROC / within, scores at 6 decimals, unified core evaluator (hc.frame_metrics).
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
from qtl import cpolicy, data as qdata, policy, prior_weight, qtree, stop_rules      # noqa: E402
from qtl.answer_calib import fit_validation_answer_model                # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

METS = ("pooled_ap", "pooled_roc", "within_roc")
POLICIES = {"eig": dict(order="eig"), "eer": dict(order="eer"), "level": dict(order="eig", level=True),
            "plan_eig": dict(order="eig", plan="disjoint"), "plan_eer": dict(order="eer", plan="disjoint"),
            "planfill_eig": dict(order="eig", plan="disjoint"), "planfill_eer": dict(order="eer", plan="disjoint"),
            "planfill_loc": dict(order="eig_loc", plan="disjoint")}
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
    vids_of = {"val": val, "test": test}
    hate = {sp: {v for v in vs if labels[v] == 1} for sp, vs in vids_of.items()}
    qdata.configure_source("soft_both", 8)
    answers, T_all = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, val + test, ["bert"])

    def mets(sc, split="test"):
        sc = {v: np.round(np.asarray(s, dtype=np.float64), 6) for v, s in sc.items()}
        m = hc.frame_metrics(sc, gt[split], hate[split])
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
        tables, weights, capped = {"r6": am6}, {}, {}
        for var in todo:
            split = "val" if var.endswith("@val") else "test"
            parts = var[:-len("@val")].split("|") if split == "val" else var.split("|")
            tab, mode, pol = parts[:3]
            B = int(parts[3]) if len(parts) > 3 else 8
            if tab not in tables:
                assert tab in ("val3", "val3h")
                tables[tab], _counts = fit_validation_answer_model(answers, labels, gt["val"], val, T_all, am6.len_mu,
                                                                   am6.len_sd, cfg["categories"])
                if tab == "val3h":               # state 0 (no hate in the video) keeps the anchored training table
                    with torch.no_grad():
                        tables[tab].theta[:, 0] = am6.theta[:, 0].cpu()
                        tables[tab].omega[:, 0] = am6.omega[:, 0].cpu()
                tables[tab] = tables[tab].to(a.device)
            if (tab, mode) not in weights:
                weights[(tab, mode)] = None if mode == "none" else prior_weight.fit(
                    pri, store, val, tables[tab], chain, answers, a.device, mode=mode)
            w = weights[(tab, mode)]
            if pol.startswith("stopE1c"):
                assert split == "test", "the stopping-rule variants are test only"
                cap = int(pol[len("stopE1c"):])
                if (tab, mode, cap) not in capped:
                    kw = dict(order="eig", record_voi=True)
                    capped[(tab, mode, cap)] = {
                        sp: replay(model, store, vs, prior_weight.table(tables[tab], w), chain, answers, cfg, cap, w,
                                   kw, a.device) for sp, vs in (("val", val), ("test", test))}
                    vals = {sp: {v: stop_rules.rule_values(r, len(r["scores"][0])) for v, r in rr.items()}
                            for sp, rr in capped[(tab, mode, cap)].items()}
                    stop_rules.add_quantile_rules(vals)
                    capped[(tab, mode, cap)]["vals"] = vals
                cr = capped[(tab, mode, cap)]
                thr, vm = policy.calibrate([cr["vals"]["val"][v]["qmixE1"] for v in val], float(B))
                calls = {v: policy.stop_calls(cr["vals"]["test"][v]["qmixE1"], thr) for v in test}
                res = mets({v: cr["test"][v]["scores"][calls[v]] for v in test})
                res.update(mean_calls=float(np.mean(list(calls.values()))), validation_mean_calls=float(vm),
                           mean_calls_positive=float(np.mean([calls[v] for v in test if labels[v] == 1])),
                           mean_calls_negative=float(np.mean([calls[v] for v in test if labels[v] == 0])))
                res["weights"] = w
                done["variants"][var] = res
                continue
            vids = vids_of[split]
            runs = replay(model, store, vids, prior_weight.table(tables[tab], w), chain, answers, cfg, B, w,
                          POLICIES[pol], a.device)
            res = mets({v: runs[v]["scores"][-1] for v in vids}, split)
            res["mean_calls"] = float(np.mean([len(runs[v]["asked"]) for v in vids]))
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
