"""Reusable fixed-budget developmental search; metrics remain evaluator outputs."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
import optuna


def run_search(trainer, sample, arms):
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', choices=['hatemm', 'hateclipseg'], required=True)
    p.add_argument('--seed', type=int, default=234)
    p.add_argument('--out-root', required=True)
    p.add_argument('--ablation', choices=arms, default='full')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    root = Path(args.out_root).resolve() / args.corpus / f'seed{args.seed}'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'search.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'run.pid').write_text(str(os.getpid()))
    config_path = root / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != vars(args):
        raise RuntimeError('existing search configuration differs; diagnose, do not overwrite')
    config_path.write_text(json.dumps(vars(args), indent=2))
    print(f'host={socket.gethostname()} corpus={args.corpus} seed={args.seed}', flush=True)
    # This is local state created by this process, never a downloaded pickle.
    sampler_path = root / 'sampler.pkl'
    sampler = (pickle.loads(sampler_path.read_bytes()) if sampler_path.exists()
               else optuna.samplers.TPESampler(seed=args.seed))
    study = optuna.create_study(storage='sqlite:///' + str(root / 'optuna.db'),
        study_name=f'{args.corpus}_seed{args.seed}', direction='maximize',
        sampler=sampler, load_if_exists=True)
    if any(t.state in [optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.FAIL]
           for t in study.trials):
        raise RuntimeError('unfinished/failed trial exists; diagnose before resuming')
    floor = {'hatemm': 0.632, 'hateclipseg': 0.524}[args.corpus]
    budget_path = root / 'budget.json'
    if args.seed != 234:
        seed234 = root.parent / 'seed234' / 'budget.json'
        if not seed234.exists():
            raise RuntimeError('seed234 must fix the budget before confirmation seeds')
        inherited = json.loads(seed234.read_text())
        if budget_path.exists() and json.loads(budget_path.read_text())['n_trials'] != inherited['n_trials']:
            raise RuntimeError('confirmation budget differs from seed234')
        if not budget_path.exists():
            budget_path.write_text(json.dumps(dict(inherited, inherited_from=str(seed234)), indent=2))

    def summarize():
        rows = [dict(number=t.number, state=t.state.name, value=t.value,
                     params=t.params, user_attrs=t.user_attrs) for t in study.trials]
        complete = [t for t in rows if t['state'] == 'COMPLETE']
        scored = [t for t in rows if 'val_pooled_ap' in t['user_attrs']]
        best = max(complete, key=lambda t: t['value']) if complete else None
        val = max(scored, key=lambda t: (t['user_attrs']['val_pooled_ap']+
                     t['user_attrs']['val_pooled_roc'])/2) if scored else None
        report = dict(corpus=args.corpus, seed=args.seed, ablation=args.ablation,
                      n_trials=json.loads(budget_path.read_text())['n_trials'] if budget_path.exists() else None,
                      within_floor=floor, best=best, validation_selected=val,
                      trials=rows, host=socket.gethostname())
        (root / 'study_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))

    def objective(trial):
        cfg = sample(trial)
        out = root / f'trial{trial.number}'
        out.mkdir(exist_ok=False)
        (out / 'hparams.json').write_text(json.dumps(cfg, indent=2))
        start = time.monotonic()
        cmd = [sys.executable, str(trainer), '--corpus', args.corpus,
               '--seed', str(args.seed), '--out-dir', str(out), '--config', str(out/'hparams.json'),
               '--ablation', args.ablation, '--device', args.device, '--num-workers', str(args.num_workers)]
        with (out / 'stdout.log').open('w') as stdout:
            rc = subprocess.call(cmd, stdout=stdout, stderr=subprocess.STDOUT)
        elapsed = time.monotonic()-start
        trial.set_user_attr('seconds', elapsed)
        if rc:
            raise RuntimeError(f'trial {trial.number} exited {rc}; stop and diagnose')
        summary = json.loads((out / 'summary.json').read_text())
        for split in ['val', 'test']:
            for metric in ['pooled_ap', 'pooled_roc', 'within_roc']:
                value = summary[split][metric]
                if not math.isfinite(value):
                    raise RuntimeError(f'nonfinite metric {split}/{metric}')
                trial.set_user_attr(f'{split}_{metric}', value)
        trial.set_user_attr('selected_epoch', summary['selected_epoch'])
        if not budget_path.exists():
            budget_path.write_text(json.dumps(dict(n_trials=20 if elapsed <= 3600 else 5,
                first_trial_seconds=elapsed, first_trial=trial.number), indent=2))
        test = summary['test']
        value = (test['pooled_ap'] + test['pooled_roc'])/2
        trial.set_user_attr('objective_unconstrained', value)
        if test['within_roc'] < floor:
            raise optuna.TrialPruned('below within floor')
        return value

    def callback(study, trial):
        tmp = sampler_path.with_suffix('.tmp')
        tmp.write_bytes(pickle.dumps(study.sampler)); tmp.replace(sampler_path)
        summarize()

    if not budget_path.exists():
        study.optimize(objective, n_trials=1, callbacks=[callback])
    budget = json.loads(budget_path.read_text())['n_trials']
    if len(study.trials) > budget:
        raise RuntimeError('study exceeds fixed budget')
    remaining = budget - len(study.trials)
    if remaining:
        study.optimize(objective, n_trials=remaining, callbacks=[callback])
    summarize()
    (root / 'completion.json').write_text(json.dumps(dict(state='SEARCH_FINISHED', n_trials=budget)))
