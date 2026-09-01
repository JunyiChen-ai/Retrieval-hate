#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/reproduction_baselines'));from eval_baseline_scores import main as evaluate
p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--run-dir',required=True);a=p.parse_args();run=Path(a.run_dir).resolve();evaluate(['--corpus',a.corpus,'--scores',str(run/'scores.jsonl'),'--split','test','--branch','score_final','--json-out',str(run/'metrics.json'),'--require-full-coverage'])
