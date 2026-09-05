"""Full 50-epoch loop shared by evidence candidates; single evaluator calls."""
import copy
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
import torch
import hier_evidence_common as common
from macilsd.train import _seq_len_of


def fit_and_evaluate(model, loader, eval_loaders, gt, labels, args, cfg, say, loss_fn):
    out = Path(args.out_dir)
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
            output = model(audio, visual, lengths)
            loss = loss_fn(model, output, audio, lengths, label)
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
    summary = dict(corpus=args.corpus, seed=args.seed, ablation=args.ablation, hparams=cfg,
                   selected_epoch=best_epoch, val_criterion=best, history=history,
                   host=socket.gethostname())
    for split in ['val', 'test']:
        scores = out / f'scores_{split}.jsonl'
        common.write_scores(str(scores), common.score_split(model, eval_loaders[split], args.device))
        metrics = common.run_evaluator(args.corpus, split, str(scores),
            str(out / ('metrics.json' if split == 'test' else 'metrics_val.json')))
        r = metrics['results']['score_av']
        if r['n_videos'] != len(eval_loaders[split].dataset):
            raise RuntimeError(f'evaluator coverage differs from fixed {split} set')
        summary[split] = dict(pooled_ap=r['pr_auc'], pooled_roc=r['roc_auc'],
                              within_roc=r['per_video']['macro_auc'], n_videos=r['n_videos'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    say('TEST ' + json.dumps(summary['test']))
    return summary
