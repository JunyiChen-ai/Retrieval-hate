"""CPU-only developmental diagnostics of explicitly named completed runs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/reproduction_baselines'))
from hate_common import data as hdata
from src.saved_prediction_diagnostics import describe_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', choices=['hatemm', 'hateclipseg'], required=True)
    parser.add_argument('--run', action='append', required=True, help='Unique label=run directory')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'runs'):
        parser.error('--out must be under this repository runs/')
    gt = hdata.gt_arrays(args.corpus, 'test')
    report = dict(purpose='Developmental test error analysis; no training, checkpoint selection, or metric recomputation',
                  created_at=datetime.now().astimezone().isoformat(), command=[sys.executable, *sys.argv],
                  code_description='2026-09-06 shared saved-prediction descriptive analysis; evaluator unchanged',
                  corpus=args.corpus, gt_source=str(Path(hdata.GT_ROOT) / f'{args.corpus}_test.npz'), runs={})
    for spec in args.run:
        label, folder = spec.split('=', 1)
        if label in report['runs']:
            parser.error(f'duplicate run label: {label}')
        result = describe_run(folder, gt)
        report['runs'][label] = result
        print(label, json.dumps({k: v for k, v in result.items() if k not in ['per_video', 'sources', 'hparams', 'val_first', 'val_last']}))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
