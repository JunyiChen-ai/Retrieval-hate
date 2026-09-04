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
    obj = json.loads(path.read_text())
    if obj.get('version') != '2026-09-05 four-input evidence v2 raw logits':
        raise ValueError(f'input version is not raw logits v2: {path}')
    if obj['id'] != vid or obj['order'] != ['av', 'v', 'a', 'empty']:
        raise ValueError(f'input identity/order: {path}')
    windows = obj['windows']
    if len(windows) != k or [w['index'] for w in windows] != list(range(k)):
        raise ValueError(f'window alignment: {path}')
    logits = np.asarray([w['log_odds'] for w in windows], dtype=np.float32)
    entropy = np.asarray([w['entropy'] for w in windows], dtype=np.float32)
    if logits.shape != (k, 4) or entropy.shape != (k, 4):
        raise ValueError(f'input shape: {path}')
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    hate_ids = {v for v, label in labels.items() if label == 1}
    best, best_state, best_epoch = -math.inf, None, None
    history = []
    for epoch in range(50):
        start = time.monotonic()
        model.train()
        losses = []
        for visual, audio, _, label in loader:
            lengths = _seq_len_of(visual)
            keep = int(lengths.max())
            visual, audio = visual[:, :keep].to(args.device), audio[:, :keep].to(args.device)
            label = label.float().to(args.device)
            bags, *_ = model(audio, visual, lengths)
            loss = F.binary_cross_entropy(bags, label)
            if args.ablation != 'no_block':
                loss = loss + common.block_bag_loss(model.last_content_logit, audio, lengths, label, 16)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        val = common.frame_metrics(common.score_split(model, eval_loaders['val'], args.device), gt['val'], hate_ids)
        criterion = (val['pooled_ap'] + val['pooled_roc']) / 2
        if not math.isfinite(criterion):
            raise RuntimeError('nonfinite validation criterion')
        history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), val=val,
                            seconds=time.monotonic()-start))
        say(json.dumps(history[-1]))
        if criterion > best:
            best, best_epoch = criterion, epoch+1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(best_state, out / 'model.pth')
    summary = dict(corpus=corpus, seed=args.seed, ablation=args.ablation, hparams=cfg,
                   selected_epoch=best_epoch, val_criterion=best, history=history,
                   host=socket.gethostname())
    for split in ['val', 'test']:
        scores = out / f'scores_{split}.jsonl'
        common.write_scores(str(scores), common.score_split(model, eval_loaders[split], args.device))
        metrics = common.run_evaluator(corpus, split, str(scores),
            str(out / ('metrics.json' if split == 'test' else 'metrics_val.json')))
        r = metrics['results']['score_av']
        if r['n_videos'] != len(ids[split]):
            raise RuntimeError(f'evaluator coverage differs from fixed {split} set')
        summary[split] = dict(pooled_ap=r['pr_auc'], pooled_roc=r['roc_auc'],
                              within_roc=r['per_video']['macro_auc'], n_videos=r['n_videos'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    say('TEST ' + json.dumps(summary['test']))
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
