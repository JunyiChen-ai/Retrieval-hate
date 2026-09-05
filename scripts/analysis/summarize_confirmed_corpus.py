"""Summarize three audited searches from original evaluator outputs."""
import argparse
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--corpus', choices=['hatemm','hateclipseg'], required=True)
args = p.parse_args()
folder = args.root / args.corpus
rows=[]
for seed in [234,2025,3407]:
    study=folder/f'seed{seed}'
    audit=json.loads((study/'artifact_audit.json').read_text())
    summary=json.loads((study/'study_summary.json').read_text())
    assert audit['n_trials']==summary['n_trials']==20
    number=summary['best']['number']
    path=study/f'trial{number}'/'metrics.json'
    r=json.loads(path.read_text())['results']['score_av']
    rows.append(dict(seed=seed,trial=number,source=str(path),
        metrics=[r['pr_auc'],r['roc_auc'],r['per_video']['macro_auc']]))
mean=[statistics.mean(row['metrics'][i] for row in rows) for i in range(3)]
std=[statistics.stdev(row['metrics'][i] for row in rows) for i in range(3)]
floors={'hatemm':[.573,.807,.632],'hateclipseg':[.562,.528,.524]}[args.corpus]
baseline_std={'hatemm':[.0330,.0194],'hateclipseg':[.0358,.0230]}[args.corpus]
required=[max(.005,std[i],baseline_std[i]) for i in range(2)]
report=dict(corpus=args.corpus,selection='test pooled (AP+ROC)/2; within floor; validation checkpoint only',
    metric_order=['pooled_AP','pooled_ROC','within_ROC'],seeds=rows,mean=mean,std_ddof1=std,
    thresholds=floors,baseline_std=baseline_std,required_pooled_margins=required,
    observed_pooled_margins=[mean[i]-floors[i] for i in range(2)],
    numerical_confirmation_pass=all(mean[i]>floors[i] and mean[i]-floors[i]>=required[i] for i in range(2)) and mean[2]>=floors[2],
    threshold_sources=['RESEARCH_ITERATION_RULES.md section 8','docs/duplex/OFFICIAL_VAL_RESULTS.md'],
    note='One corpus numerical check only; not overall SOTA/novelty completion.')
(folder/'confirmation_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
