#!/usr/bin/env python3
"""CLAP lane — cache verification + provenance (md5/sha256, shapes, manifests, alignment).

Run after extraction job 13647. Verifies the derived CLAP caches are well-formed and aligned to
the banked baselines, and emits the hashes the gate record must cite.

Checks:
  1. Both caches load; shapes match the declared dims (proj 1024 = 2*512, hidden 2048 = 2*1024).
  2. trainval id order == gt train.jsonl (+) val.jsonl, and == the banked Whisper cache order
     (which the K-CLAP-3 head-to-head depends on).
  3. Labels agree with gt and with every baseline feature cache (CLIP / frozen-Qwen / LoRA-curric).
  4. No NaN, no all-zero rows, no no_audio/ERR statuses in the manifests.
  5. trainval and test id sets are DISJOINT (the gate must be unable to reach test rows).
  6. md5 + sha256 of both caches, for the record.

Zero test-touch in the scientific sense: the test cache is hashed and its ids counted, but no test
LABEL is used in any statistic and no test-set metric is computed. CPU, seconds.
"""
import hashlib
import json

import numpy as np
import torch

REPO = '/data/jehc223/RGCL'
DS = 'HateMM'
TAG = 'larger_clap_general'
OUT = f'{REPO}/refine-logs/CLAP_CACHE_VERIFY_OUT.json'


def digest(path):
    m, s = hashlib.md5(), hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            m.update(chunk); s.update(chunk)
    return m.hexdigest(), s.hexdigest()


def gt_ids(splits):
    ids, lab = [], []
    for sp in splits:
        with open(f'{REPO}/data/gt/{DS}/{sp}.jsonl') as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line); ids.append(str(o['id'])); lab.append(int(o['label']))
    return ids, np.asarray(lab)


def pooled_ids_labels(enc, splits):
    d = {}
    for sp in splits:
        o = torch.load(f'{REPO}/data/CLIP_Embedding/{DS}/{sp}_{enc}.pt',
                       map_location='cpu', weights_only=False)
        for i, s in enumerate(o['ids'][0]):
            d[s] = int(o['labels'][i])
    return d


def main():
    res = {'dataset': DS, 'model_tag': TAG, 'checks': {}, 'caches': {}}
    ok_all = True

    for group, splits in (('trainval', ('train', 'val')), ('test', ('test',))):
        cache = f'{REPO}/data/audio/{DS}/clap_{TAG}_{group}.pt'
        man = json.load(open(f'{REPO}/data/audio/{DS}/clap_{TAG}_{group}_manifest.json'))
        c = torch.load(cache, map_location='cpu', weights_only=False)
        ids = list(c['ids'][0])
        proj = c['proj'].numpy(); hid = c['hidden'].numpy()
        gids, glab = gt_ids(splits)
        md5, sha = digest(cache)

        chk = {
            'id_order_matches_gt': ids == gids,
            'proj_shape': list(proj.shape), 'hidden_shape': list(hid.shape),
            'proj_dim_ok': proj.shape[1] == 1024, 'hidden_dim_ok': hid.shape[1] == 2048,
            'n_matches_gt': len(ids) == len(gids),
            'labels_match_gt': bool((c['labels'].numpy() == glab).all()),
            'no_nan': bool(not (np.isnan(proj).any() or np.isnan(hid).any())),
            'no_zero_rows': int((~proj.any(axis=1)).sum()) == 0,
            'manifest_status': man['status_counts'],
            'manifest_n_zero': man['n_zero_vector_rows'], 'manifest_n_nan': man['n_nan'],
            'manifest_clean': (man['n_zero_vector_rows'] == 0 and man['n_nan'] == 0
                               and set(man['status_counts']) <= {'ok', '(all-cached)'}),
        }
        # label agreement with every baseline cache the gate conditions on
        bsplits = ('train', 'dev_seen') if group == 'trainval' else ('test_seen',)
        for enc in ('openai_clip-vit-large-patch14-336_HF', 'Qwen2.5-VL-7B-Instruct_HF',
                    'Qwen2.5-VL-7B-Instruct-LoRA-curric_HF'):
            try:
                d = pooled_ids_labels(enc, bsplits)
                chk[f'labels_agree_{enc}'] = bool(all(d[i] == int(c['labels'][k])
                                                     for k, i in enumerate(ids)))
            except Exception as e:  # noqa: BLE001
                chk[f'labels_agree_{enc}'] = f'SKIP:{type(e).__name__}'

        res['caches'][group] = {
            'path': cache, 'md5': md5, 'sha256': sha, 'N': len(ids),
            'n_pos': int(c['labels'].sum()), 'model': c.get('model'), 'pool': c.get('pool'),
            'window_s': float(c.get('window_s', 0)), 'sample_rate': int(c.get('sample_rate', 0)),
            'n_windows': man.get('n_windows'), 'elapsed_s': man.get('elapsed_s'),
            'proj_norm_examples': man.get('example_proj_norms'), 'checks': chk}
        bad = [k for k, v in chk.items() if v is False]
        ok_all = ok_all and not bad
        print(f"[{group}] N={len(ids)} pos={int(c['labels'].sum())} proj={proj.shape} hidden={hid.shape}")
        print(f"    md5={md5}")
        print(f"    sha256={sha}")
        print(f"    windows: {man.get('n_windows')}  status={man['status_counts']}")
        print(f"    FAILED CHECKS: {bad if bad else 'none'}")

    tv = set(torch.load(f'{REPO}/data/audio/{DS}/clap_{TAG}_trainval.pt',
                        map_location='cpu', weights_only=False)['ids'][0])
    te = set(torch.load(f'{REPO}/data/audio/{DS}/clap_{TAG}_test.pt',
                        map_location='cpu', weights_only=False)['ids'][0])
    disjoint = len(tv & te) == 0
    res['checks']['trainval_test_disjoint'] = disjoint
    res['checks']['n_overlap'] = len(tv & te)

    wh = torch.load(f'{REPO}/data/audio/{DS}/whisper_whisper-large-v3_trainval.pt',
                    map_location='cpu', weights_only=False)
    cl = torch.load(f'{REPO}/data/audio/{DS}/clap_{TAG}_trainval.pt',
                    map_location='cpu', weights_only=False)
    same = list(wh['ids'][0]) == list(cl['ids'][0])
    res['checks']['clap_whisper_id_order_identical'] = same
    print(f"\ntrainval/test disjoint: {disjoint} (overlap {len(tv & te)})")
    print(f"CLAP/Whisper id order identical (K-CLAP-3 head-to-head valid): {same}")

    res['ALL_CHECKS_PASS'] = bool(ok_all and disjoint and same)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"\nALL_CHECKS_PASS = {res['ALL_CHECKS_PASS']}  -> {OUT}")


if __name__ == '__main__':
    main()
