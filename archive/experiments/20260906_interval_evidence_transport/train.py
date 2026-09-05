"""Full fixed-protocol training with one frozen VLM's interval observations."""
import argparse
import json
import os
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader
import hier_evidence_common as common
import vlm_verdict
from fixed_training_protocol import fit_and_evaluate
from interval_observation_data import TrainDataset, EvalDataset, content_normalization
from model import Candidate

ARMS = ['full', 'hard_observation', 'uniform_assignment', 'additive_readout',
        'categorical_noise', 'no_vlm', 'no_observation_loss']
DEFAULTS = dict(lr=.0003, dropout=.2, max_seqlen=200)


def train(args, cfg):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.log').open('a') as log:
        def say(message):
            print(message, flush=True)
            log.write(message + '\n'); log.flush()
        say(f'host={socket.gethostname()} corpus={args.corpus} seed={args.seed} arm={args.ablation} code={common._git_describe()}')
        (out / 'run.pid').write_text(str(os.getpid()) + '\n')
        (out / 'config.json').write_text(json.dumps(dict(vars(args), hparams=cfg, fixed=dict(
            epochs=50, batch_size=32, hidden=128, crop_repeat=5, exchange_rounds=1, topk_div=16,
            loss_weights=[1, 1], observation_scale_weights=[.5, .5], initial_noise_precision=1,
            probability_floor=1e-6, normalization_floor=1e-4, denominator_floor=1e-12,
            frozen_vlm='Qwen/Qwen2.5-VL-7B-Instruct original single-pass K30/K4',
            new_vlm_calls=0, deployment_vlm_calls=0 if args.ablation == 'no_vlm' else 34)), indent=2))
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed); random.seed(args.seed)
        labels, ids, gt, excluded = common.load_fixed_cohort(args.corpus)
        all_ids = sum(ids.values(), [])
        cache = common.ScaffoldCache(args.corpus, all_ids,
            lambda vid, snip, duration: np.zeros((len(snip), common.SCAF_DIM), dtype=np.float32))
        observations = None
        if args.ablation != 'no_vlm':
            raw = {k: vlm_verdict.load_verdicts(args.corpus, k=k, tag='qwen', video_ids=all_ids, strict=True)
                   for k in [30, 4]}
            for k in [30, 4]:
                if set(raw[k]) != set(all_ids):
                    raise ValueError(f'incomplete VLM input coverage: K{k}')
            observations = {v: np.concatenate([raw[k][v] for k in [30, 4]]) for v in all_ids}
        mean, std = content_normalization(cache, ids['train'])
        (out / 'normalization.json').write_text(json.dumps(dict(source_split='train', visual_crop=0,
            video_ids=ids['train'], mean=mean.tolist(), std=std.tolist()), indent=2))
        (out / 'coverage.json').write_text(json.dumps(dict(splits=ids, excluded_no_gt=excluded,
            missing_text=cache.n_missing_text, vlm_input_ids=all_ids if observations is not None else [],
            observation_loss_split='train', vlm_provenance='data/MLLM_scores/PROVENANCE.md'), indent=2))
        loader = DataLoader(TrainDataset(args.corpus, ids['train'], labels, cache, cfg['max_seqlen'],
            verdicts=observations), batch_size=32, shuffle=True, num_workers=args.num_workers)
        eval_loaders = {s: DataLoader(EvalDataset(args.corpus, ids[s], cache, verdicts=observations),
            batch_size=1, num_workers=args.num_workers) for s in ['val', 'test']}
        model = Candidate(mean, std, cfg['dropout'], args.ablation).to(args.device)
        fit_and_evaluate(model, loader, eval_loaders, gt, labels, args, cfg, say,
                         lambda model, output, audio, lengths, label: model.loss(output, label))
        (out / 'observation_channel.json').write_text(json.dumps(dict(scales=[30, 4],
            emission=model.emission().detach().exp().cpu().tolist(),
            note='selected-checkpoint ordinal observation channel; not identified true error rates'), indent=2))


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
        if set(supplied) - cfg.keys():
            raise ValueError('undeclared hyperparameters')
        cfg.update(supplied)
    train(args, cfg)
