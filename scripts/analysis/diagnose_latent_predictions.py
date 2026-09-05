"""Describe saved C6 predictions/checkpoints; do not recompute evaluation metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--ablations',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a = p.parse_args()
audit = json.loads((a.ablations/'artifact_audit.json').read_text())
folders = {'full':a.reference, **{k:Path(v['source']).parent for k,v in audit['arms'].items()}}
rows = {}
full_scores = None
for arm,folder in folders.items():
    summary = json.loads((folder/'summary.json').read_text())
    data = [json.loads(line) for line in (folder/'scores_test.jsonl').read_text().splitlines()]
    scores = {r['video_id']:np.asarray(r['score_av'],dtype=float) for r in data}
    assert len(scores) == len(data)
    if arm == 'full':
        full_scores = scores
    assert scores.keys() == full_scores.keys()
    assert all(scores[v].shape == full_scores[v].shape for v in scores)
    ordered = sorted(scores)
    pool = np.concatenate([scores[v] for v in ordered])
    reference = np.concatenate([full_scores[v] for v in ordered])
    assert np.isfinite(pool).all()
    differences = np.abs(pool-reference)
    params = torch.load(folder/'model.pth',map_location='cpu',weights_only=True)
    raw = params['raw_cholesky']
    diagonal = torch.nn.functional.softplus(raw.diagonal(dim1=-2,dim2=-1))+1e-4
    chol = torch.diag_embed(diagonal)
    if arm != 'diagonal_emission':
        chol = chol + raw.tril(-1)
    covariance = chol @ chol.transpose(-1,-2)
    norm = json.loads((folder/'normalization.json').read_text())
    initial_means = torch.tensor(norm['initial_state_means'],dtype=params['means'].dtype)
    cov_std = covariance.diagonal(dim1=-2,dim2=-1).sqrt()
    correlation = covariance / (cov_std[:,:,None]*cov_std[:,None,:])
    rows[arm] = dict(sources=[str(folder/x) for x in ['scores_test.jsonl','summary.json','metrics.json','model.pth','normalization.json']],
        selected_epoch=summary['selected_epoch'],metrics=summary['test'],
        score_quantiles=dict(zip(['min','q01','q10','median','q90','q99','max'],np.quantile(pool,[0,.01,.1,.5,.9,.99,1]).tolist())),
        near_zero_fraction=float(np.mean(pool<=1e-5)), near_one_fraction=float(np.mean(pool>=1-1e-5)),
        mean_video_temporal_std=float(np.mean([scores[v].std() for v in ordered])),
        absolute_difference_from_full=dict(mean=float(differences.mean()),q95=float(np.quantile(differences,.95)),maximum=float(differences.max())),
        pearson_score_vs_full=float(np.corrcoef(pool,reference)[0,1]),
        observation_state_mean_distance=float(torch.linalg.vector_norm(params['means'][1]-params['means'][0])),
        observation_mean_change_from_train_initialization=float(torch.linalg.vector_norm(params['means']-initial_means)),
        observation_correlation=correlation.tolist(),
        transition_weight_norm=float(torch.linalg.vector_norm(params['transition.weight'])),
        initial_weight_norm=float(torch.linalg.vector_norm(params['initial.weight'])))
report=dict(note='Descriptive developmental diagnostics; score saturation/parameter norms do not establish causal module attribution.',arms=rows)
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(report,indent=2,allow_nan=False))
for arm,r in rows.items():
    print(arm,'epoch',r['selected_epoch'],'near_0/1',round(r['near_zero_fraction'],3),round(r['near_one_fraction'],3),
          'score_correlation',round(r['pearson_score_vs_full'],4),
          'MAE',round(r['absolute_difference_from_full']['mean'],4))
