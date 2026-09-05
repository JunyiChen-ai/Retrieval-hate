"""Developmental error analysis of saved scores; does not recompute AP/AUC."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/reproduction_baselines'))
sys.path.insert(0, str(ROOT))
from hate_common import data as hdata
from src.saved_prediction_diagnostics import describe_run

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
report = dict(purpose='Developmental test error analysis; no training or checkpoint selection', corpora={})
for corpus, seeds in [('hatemm', [234]), ('hateclipseg', [234, 2025, 3407])]:
    gt = hdata.gt_arrays(corpus, 'test')
    labels = hdata.load_labels(corpus)
    split_counts = {}
    for split in ['train', 'val', 'test']:
        ids = hdata.load_split(corpus, split)
        split_counts[split] = dict(n=len(ids), positive=sum(labels[v] for v in ids))
    result = dict(split_video_counts=split_counts, gt_source=str(Path(hdata.GT_ROOT) / f'{corpus}_test.npz'), seeds={})
    for seed in seeds:
        study = a.root / corpus / f'seed{seed}'
        number = json.loads((study / 'study_summary.json').read_text())['best']['number']
        full = study / f'trial{number}'
        abl = a.root / 'ablations' / corpus / f'seed{seed}'
        audit = json.loads((abl / 'artifact_audit.json').read_text())
        arms = {'full': full, **{arm: Path(x['source']).parent for arm, x in audit['arms'].items()}}
        seed_rows = {}
        for arm, folder in arms.items():
            seed_rows[arm] = describe_run(folder, gt)
        result['seeds'][seed] = seed_rows
    report['corpora'][corpus] = result
out = a.root / 'error_analysis'
out.mkdir(exist_ok=True)
(out / 'saved_prediction_diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
for corpus, result in report['corpora'].items():
    print(corpus, result['split_video_counts'])
    for seed, arms in result['seeds'].items():
        for arm in ['full','raw_verdict','ordinary_attention','additive_fusion','no_block']:
            r = arms[arm]
            print(seed, arm, 'epoch', r['selected_epoch'], 'scores',
                  {k: round(v['mean'], 4) if v['mean'] is not None else None for k,v in r['mean_frame_scores'].items()},
                  'temporal_std', round(r['mean_mixed_video_temporal_std'], 4),
                  'inverted_videos', r['n_mixed_video_auc_below_half'])
