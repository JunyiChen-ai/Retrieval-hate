"""Full 50-epoch fixed search trial; train-only single-VLM window observations."""
import argparse
import json
import os
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader
import hier_evidence_common as common
import vlm_verdict
from fixed_training_protocol import fit_and_evaluate
from interval_observation_data import content_normalization
from dataset import TrainDataset, EvalDataset
from model import Candidate

ARMS = ['full', 'hard_observation', 'unfactorized', 'topk_event', 'no_vlm', 'fine_only']
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
        fixed=dict(epochs=50, batch_size=32, hidden=128, crop_repeat=5,
            kernel=3, dilations=[1, 2], loss_weights=[1, 1, 1],
            normalization_floor=1e-4, intensity_floor=1e-8, observation_floor=1e-6,
            teacher='frozen Qwen K30/K4 original verdicts, train-only',
            new_vlm_extraction=0, deployment_vlm_calls=0)), indent=2))
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); random.seed(args.seed)
    labels, ids, gt, excluded = common.load_fixed_cohort(args.corpus)
    cache = common.ScaffoldCache(args.corpus, sum(ids.values(), []),
        lambda vid, snip, duration: np.zeros((len(snip), common.SCAF_DIM), dtype=np.float32))
    observations = None
    false_positive = [.1, .1]
    if args.ablation != 'no_vlm':
        raw = {k: vlm_verdict.load_verdicts(args.corpus, k=k, tag='qwen', video_ids=ids['train'], strict=True) for k in [30, 4]}
        for k in [30, 4]:
            if set(raw[k]) != set(ids['train']):
                raise ValueError(f'incomplete train VLM observations: K{k}')
        observations = {v: np.concatenate([(raw[k][v] >= 2).astype(np.float32) for k in [30, 4]]) for v in ids['train']}
        negative = [v for v in ids['train'] if labels[v] == 0]
        if not negative:
            raise ValueError('negative train videos required for noise initialization')
        false_positive = [(sum(float((raw[k][v] >= 2).sum()) for v in negative)+1)/(len(negative)*k+2) for k in [30, 4]]
    # Streaming statistics use train content only, no VLM or held-out features.
    mean, std = content_normalization(cache, ids['train'])
    (out/'normalization.json').write_text(json.dumps(dict(source_split='train', visual_crop=0,
        video_ids=ids['train'], mean=mean.tolist(), std=std.tolist(),
        initial_false_positive=false_positive, initial_sensitivity_fraction=.9,
        beta_initialization=[1, 1]), indent=2))
    (out/'coverage.json').write_text(json.dumps(dict(splits=ids, excluded_no_gt=excluded,
        missing_text=cache.n_missing_text, teacher_video_ids=ids['train'] if observations is not None else [],
        val_test_teacher_used=False), indent=2))
    loader = DataLoader(TrainDataset(args.corpus, ids['train'], labels, cache,
        cfg['max_seqlen'], verdicts=observations), batch_size=32, shuffle=True, num_workers=args.num_workers)
    eval_loaders = {s: DataLoader(EvalDataset(args.corpus, ids[s], cache), batch_size=1,
                                 num_workers=args.num_workers) for s in ['val', 'test']}
    model = Candidate(mean, std, false_positive, cfg['dropout'], args.ablation).to(args.device)
    fit_and_evaluate(model, loader, eval_loaders, gt, labels, args, cfg, say,
                     lambda model, output, audio, lengths, label: model.loss(label, audio))
    if args.ablation not in ['no_vlm', 'hard_observation']:
        r = model.noise_logit.detach().sigmoid()
        q = r+(1-r)*model.sensitivity_gap.detach().sigmoid()
        (out/'observation_channel.json').write_text(json.dumps(dict(
            scales=[30, 4], false_positive=r.cpu().tolist(), sensitivity=q.cpu().tolist(),
            note='learned observation parameters, not identified true error rates'), indent=2))
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
        supplied = json.loads(Path(args.config).read_text())
        if set(supplied)-cfg.keys():
            raise ValueError('undeclared hyperparameters')
        cfg.update(supplied)
    train(args, cfg)
