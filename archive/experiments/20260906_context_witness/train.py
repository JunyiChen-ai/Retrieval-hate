"""Full fixed-protocol training; frozen inputs and train-only normalization."""
import argparse
import json
import os
from pathlib import Path
import random
import socket
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/hier_evidence_common.py').is_file())
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader
import hier_evidence_common as common
from context_witness import VERSION, read_measurement, feature_rows
from fixed_training_protocol import fit_and_evaluate
import vlm_verdict
from model import Candidate

ARMS = ['full', 'target_only', 'raw_four', 'no_residual', 'visible_reconstruction', 'no_deletion', 'no_sparsity']
DEFAULTS = dict(lr=.0003, dropout=.2, max_seqlen=200)


def train(args, cfg):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (out/'run.log').open('a')
    def say(message):
        print(message, flush=True)
        log.write(message+'\n'); log.flush()
    say(f'host={socket.gethostname()} corpus={args.corpus} seed={args.seed} arm={args.ablation} code={common._git_describe()}')
    (out/'run.pid').write_text(str(os.getpid())+'\n')
    (out/'config.json').write_text(json.dumps(dict(vars(args), hparams=cfg,
        fixed=dict(epochs=50, batch_size=32, hidden=128, crop_repeat=5, K=30,
                   content_dim=1920, observation_dim=30, normalization_floor=1e-4,
                   pooling_floor=1e-6, loss_weights=[1, 1, 1, 1, 1])), indent=2))
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); random.seed(args.seed)
    labels, ids, gt, excluded = common.load_fixed_cohort(args.corpus)
    all_ids = sum(ids.values(), [])
    cache = common.ScaffoldCache(args.corpus, all_ids,
        lambda vid, snip, duration: np.zeros((len(snip), common.SCAF_DIM), dtype=np.float32))
    for vid in all_ids:
        audio, duration, snip = cache.items[vid]
        logits, entropy = read_measurement(ROOT/'data/context_witness'/args.corpus/'K30'/f'{vid}.json', vid)
        extra = vlm_verdict.verdict_rows(feature_rows(logits, entropy, args.ablation), snip, duration)
        cache.items[vid] = (np.ascontiguousarray(np.concatenate([audio, extra], -1)), duration, snip)
    total = np.zeros(1950, dtype=np.float64)
    squares = total.copy()
    count = 0
    for vid in ids['train']:
        audio, duration, snip = cache.items[vid]
        visual = common.align.aligned_visual_crop(args.corpus, vid, 0, 'snippet', duration, snip)
        rows = np.concatenate([visual, audio[:, :common.SCAF_OFFSET], audio[:, common.A_EXT_DIM:]], -1).astype(np.float64)
        if rows.shape[1] != 1950 or not np.isfinite(rows).all():
            raise ValueError(f'invalid training inputs: {vid}')
        total += rows.sum(0); squares += np.square(rows).sum(0); count += len(rows)
    mean = total/count
    std = np.sqrt(np.maximum(squares/count-mean**2, 0)).clip(1e-4)
    (out/'normalization.json').write_text(json.dumps(dict(source_split='train', visual_crop=0,
        video_ids=ids['train'], mean=mean.tolist(), std=std.tolist(), input_version=VERSION), indent=2))
    (out/'coverage.json').write_text(json.dumps(dict(splits=ids, excluded_no_gt=excluded,
        missing_text=cache.n_missing_text), indent=2))
    loader = DataLoader(common.TrainDataset(args.corpus, ids['train'], labels, cache, cfg['max_seqlen']),
                        batch_size=32, shuffle=True, num_workers=args.num_workers)
    eval_loaders = {s: DataLoader(common.EvalDataset(args.corpus, ids[s], cache), batch_size=1,
                                 num_workers=args.num_workers) for s in ['val', 'test']}
    model = Candidate(mean, std, cfg['dropout'], args.ablation).to(args.device)
    fit_and_evaluate(model, loader, eval_loaders, gt, labels, args, cfg, say,
                     lambda model, output, audio, lengths, label: model.loss(label, lengths))
    log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', choices=['hatemm', 'hateclipseg'], required=True)
    parser.add_argument('--seed', type=int, default=234)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--config')
    parser.add_argument('--ablation', choices=ARMS, default='full')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    cfg = DEFAULTS.copy()
    if args.config:
        supplied = json.loads(Path(args.config).read_text())
        if set(supplied)-cfg.keys():
            raise ValueError('undeclared hyperparameters')
        cfg.update(supplied)
    train(args, cfg)
