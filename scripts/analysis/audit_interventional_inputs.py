#!/usr/bin/env python3
"""Parse candidate-5 input coverage; never infer completion from a PID/marker."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = '2026-09-05 four-input evidence v2 raw logits'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', choices=['hatemm', 'hateclipseg'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    splits = {s: (ROOT / 'results/reproduction/splits' / f'{args.corpus}_{s}.txt').read_text().split()
              for s in ['train', 'val', 'test']}
    ids = sum(splits.values(), [])
    if len(ids) != len(set(ids)):
        raise ValueError('split overlap or duplicate IDs')
    cache = ROOT / 'data/interventional_evidence' / args.corpus
    good, missing, invalid = {}, {}, {}
    for k in [30, 4]:
        good[k], missing[k], invalid[k] = [], [], {}
        for vid in ids:
            path = cache / f'K{k}' / f'{vid}.json'
            if not path.exists():
                missing[k].append(vid)
                continue
            try:
                obj = json.loads(path.read_text())
                assert obj['version'] == VERSION, 'wrong version'
                assert obj['model'] == 'Qwen/Qwen2.5-VL-7B-Instruct', 'wrong model'
                assert obj['id'] == vid, 'wrong ID'
                assert obj['order'] == ['av', 'v', 'a', 'empty'], 'wrong intervention order'
                assert isinstance(obj['asr_missing'], bool), 'missing ASR availability flag'
                assert len(obj['windows']) == k, 'wrong window count'
                for i, w in enumerate(obj['windows']):
                    assert w['index'] == i, 'wrong window order'
                    assert w['relative_bounds'] == [i/k, (i+1)/k], 'wrong window bounds'
                    for name in ['log_odds', 'entropy']:
                        assert len(w[name]) == 4, 'wrong input shape'
                        assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in w[name]), 'nonfinite input'
                    assert all(-1e-6 <= v <= math.log(2)+1e-6 for v in w['entropy']), 'invalid binary entropy'
                good[k].append(vid)
            except (AssertionError, KeyError, ValueError, TypeError) as exc:
                invalid[k][vid] = str(exc)
    complete_ids = set(good[30]) & set(good[4])
    provenance = cache / 'PROVENANCE.md'
    report = dict(corpus=args.corpus, version=VERSION, cache=str(cache), expected=len(ids),
                  complete_videos=len(complete_ids), valid_files={k:len(v) for k,v in good.items()},
                  split_complete={s:sum(v in complete_ids for v in vs) for s,vs in splits.items()},
                  split_expected={s:len(vs) for s,vs in splits.items()}, missing=missing, invalid=invalid,
                  provenance_present=provenance.is_file())
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in ['corpus','expected','complete_videos','valid_files','split_complete','split_expected']}))
    if any(invalid.values()) or not provenance.is_file():
        raise SystemExit('invalid input or missing provenance: inspect audit')
    if args.require_complete and len(complete_ids) != len(ids):
        raise SystemExit('input coverage incomplete')


if __name__ == '__main__':
    main()
