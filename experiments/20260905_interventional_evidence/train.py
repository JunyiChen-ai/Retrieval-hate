"""Full candidate-5 training; all scoring uses the shared evaluator."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import hier_evidence_common as common
import vlm_verdict
import verdict_hmm
from interventional_observations import load_observations
from fixed_training_protocol import fit_and_evaluate
from hate_common import data as hdata
from macilsd.train import _seq_len_of
from model import Candidate

ARMS = ['full', 'raw_verdict', 'full_input_only', 'four_logits', 'no_interaction',
        'ordinary_attention', 'additive_fusion', 'dempster_fusion', 'no_block']
DEFAULTS = dict(lr=0.0003, dropout=0.2, max_seqlen=200)


def evidence_rows(path, k, vid, arm, raw):
    if arm == 'raw_verdict':
        out = np.zeros((k, 8), dtype=np.float32)
        out[:, 0] = np.asarray(raw, dtype=np.float32) / 3
        return out
    logits, entropy = load_observations(path, k, vid)
    av, v, a, empty = logits.T
    features = np.stack([empty, v-empty, a-empty, av-v-a+empty], -1)
    out = np.concatenate([features, entropy], -1)
    if arm == 'four_logits':
        out[:, :4] = logits
    elif arm == 'full_input_only':
        out.fill(0)
        out[:, 0], out[:, 4] = av, entropy[:, 0]
    elif arm == 'no_interaction':
        out[:, 3:5] = 0
    if not np.isfinite(out).all():
        raise ValueError(f'nonfinite inputs: {path}')
    return out


def train(args, cfg):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'run.log').open('a')

    def say(message):
        print(message, flush=True)
        log.write(message + '\n'); log.flush()

    say(f'host={socket.gethostname()} corpus={args.corpus} seed={args.seed} arm={args.ablation} code={common._git_describe()}')
    (out / 'run.pid').write_text(str(os.getpid()))
    (out / 'config.json').write_text(json.dumps(dict(vars(args), hparams=cfg,
        fixed=dict(epochs=50, batch_size=32, hidden=128, heads=4, topk_divisor=16,
                   crop_repeat=5, block_weight=1, hmm_w_fine=1)), indent=2))
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    corpus = args.corpus
    labels = hdata.load_labels(corpus)
    ids = {s: hdata.load_split(corpus, s) for s in ['train', 'val', 'test']}
    gt = {s: hdata.gt_arrays(corpus, s) for s in ['val', 'test']}
    excluded_no_gt = {}
    for s in ['val', 'test']:
        excluded_no_gt[s] = sorted(set(ids[s]) - gt[s].keys())
        expected_exclusion = ['hate_video_427'] if corpus == 'hatemm' and s == 'test' else []
        if excluded_no_gt[s] != expected_exclusion:
            raise ValueError(f'unexpected fixed GT coverage: {s} {excluded_no_gt[s]}')
        ids[s] = [v for v in ids[s] if v in gt[s]]
        if set(ids[s]) != set(gt[s]):
            raise ValueError(f'evaluation IDs do not cover the fixed GT: {s}')
    for s in ids:
        if set(common.usable(corpus, ids[s])) != set(ids[s]):
            raise ValueError(f'missing baseline features in {s}; cannot evaluate a subset')
    flattened = sum(ids.values(), [])
    if len(flattened) != len(set(flattened)):
        raise ValueError('duplicate video or split overlap')
    raw = {k: vlm_verdict.load_verdicts(corpus, k=k, tag='qwen') for k in [30, 4]}
    if any(v not in raw[k] for v in flattened for k in [30, 4]):
        raise ValueError('missing original verdict')
    binary = {v: tuple(verdict_hmm.binarize(raw[k][v]) for k in [30, 4]) for v in ids['train']}
    hmm, _, _ = common.fit_hmm(corpus, ids['train'], labels, binary)
    hmm.save(str(out / 'hmm_train_targets.json'))
    scaffold = common.make_scaffold_fn(hmm, binary, 'full', 1.0)
    train_set = set(ids['train'])

    def train_targets(vid, snip, duration):
        if vid in train_set:
            return scaffold(vid, snip, duration)
        # No HMM posterior is computed for validation/test or fed to inference.
        return np.zeros((len(snip), common.SCAF_DIM), dtype=np.float32)

    cache = common.ScaffoldCache(corpus, flattened, train_targets)
    for vid in flattened:
        audio, duration, snip = cache.items[vid]
        extra = [vlm_verdict.verdict_rows(evidence_rows(
            ROOT / 'data/interventional_evidence' / corpus / f'K{k}' / f'{vid}.json',
            k, vid, args.ablation, raw[k][vid]), snip, duration) for k in [30, 4]]
        cache.items[vid] = (np.ascontiguousarray(np.concatenate([audio] + extra, -1)), duration, snip)
    (out / 'coverage.json').write_text(json.dumps(dict(splits=ids, excluded_no_gt=excluded_no_gt,
                                                      missing_text=cache.n_missing_text), indent=2))
    loader = DataLoader(common.TrainDataset(corpus, ids['train'], labels, cache, cfg['max_seqlen']),
                        batch_size=32, shuffle=True, num_workers=args.num_workers)
    eval_loaders = {s: DataLoader(common.EvalDataset(corpus, ids[s], cache), batch_size=1,
                                  num_workers=args.num_workers) for s in ['val', 'test']}
    model = Candidate(cfg['dropout'], args.ablation).to(args.device)
    def loss_fn(model, output, audio, lengths, label):
        loss = F.binary_cross_entropy(output[0], label)
        if args.ablation != 'no_block':
            loss = loss + common.block_bag_loss(model.last_content_logit, audio, lengths, label, 16)
        return loss

    fit_and_evaluate(model, loader, eval_loaders, gt, labels, args, cfg, say, loss_fn)
    log.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', choices=['hatemm', 'hateclipseg'], required=True)
    p.add_argument('--seed', type=int, default=234)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--config')
    p.add_argument('--ablation', choices=ARMS, default='full')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-workers', type=int, default=4)
    args = p.parse_args()
    cfg = DEFAULTS.copy()
    if args.config:
        config = json.loads(Path(args.config).read_text())
        if set(config) - cfg.keys():
            raise ValueError('undeclared hyperparameters')
        cfg.update(config)
    train(args, cfg)
