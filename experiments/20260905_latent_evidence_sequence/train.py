"""Full weakly supervised training; all GT handling and metrics use shared code."""
import argparse
import json
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
from hate_common import data as hdata
from interventional_observations import load_observations, VERSION
from fixed_training_protocol import fit_and_evaluate
from model import Candidate

ARMS = ['full','diagonal_emission','full_input_emission','raw_verdict',
        'static_transition','no_temporal_content','independent_state',
        'event_to_topk','no_observation_likelihood']
DEFAULTS = dict(lr=.0003, dropout=.2, max_seqlen=200)


def train(args, cfg):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'run.log').open('a')
    def say(message):
        print(message, flush=True); log.write(message + '\n'); log.flush()
    say(f'host={socket.gethostname()} corpus={args.corpus} seed={args.seed} arm={args.ablation} code={common._git_describe()}')
    import os
    (out / 'run.pid').write_text(str(os.getpid()))
    (out / 'config.json').write_text(json.dumps(dict(vars(args), hparams=cfg,
        fixed=dict(epochs=50,batch_size=32,hidden=128,kernel=3,states=2,crop_repeat=5,
                   numeric_floor=1e-4,likelihood_per_dimension=True,loss_weights=[1,1])), indent=2))
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); random.seed(args.seed)
    corpus = args.corpus
    labels = hdata.load_labels(corpus)
    ids = {s: hdata.load_split(corpus,s) for s in ['train','val','test']}
    gt = {s: hdata.gt_arrays(corpus,s) for s in ['val','test']}
    excluded = {}
    for split in ['val','test']:
        excluded[split] = sorted(set(ids[split]) - gt[split].keys())
        allowed = ['hate_video_427'] if corpus == 'hatemm' and split == 'test' else []
        if excluded[split] != allowed:
            raise ValueError(f'unexpected GT exclusion {split}: {excluded[split]}')
        ids[split] = [v for v in ids[split] if v in gt[split]]
        if set(ids[split]) != set(gt[split]):
            raise ValueError('fixed GT coverage mismatch')
    all_ids = sum(ids.values(), [])
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('duplicate video or split overlap')
    for split, videos in ids.items():
        if set(common.usable(corpus,videos)) != set(videos):
            raise ValueError(f'missing baseline features in {split}')
    cache = common.ScaffoldCache(corpus, all_ids,
        lambda vid,snip,duration: np.zeros((len(snip),common.SCAF_DIM),dtype=np.float32))
    raw = {k: vlm_verdict.load_verdicts(corpus,k=k,tag='qwen') for k in [30,4]} if args.ablation == 'raw_verdict' else None
    for vid in all_ids:
        audio, duration, snip = cache.items[vid]
        columns = []
        for k in [30,4]:
            logits, _ = load_observations(ROOT/'data/interventional_evidence'/corpus/f'K{k}'/f'{vid}.json',k,vid)
            if args.ablation == 'raw_verdict':
                logits = np.asarray(raw[k][vid], dtype=np.float32)[:,None] / 3
            elif args.ablation == 'full_input_emission':
                logits = logits[:, :1]
            columns.append(vlm_verdict.verdict_rows(logits,snip,duration))
        extra = np.concatenate(columns,-1).astype(np.float32)
        cache.items[vid] = (np.ascontiguousarray(np.concatenate([audio,extra],-1)), duration, snip)
    # Only train inputs/video labels initialize density statistics and normalization.
    train_observations = [cache.items[v][0][:,common.A_EXT_DIM:].astype(np.float64) for v in ids['train']]
    joined = np.concatenate(train_observations)
    mean, std = joined.mean(0), joined.std(0).clip(1e-4)
    state_means = np.stack([np.concatenate([o for v,o in zip(ids['train'],train_observations) if labels[v] == s]).mean(0)
                            for s in [0,1]])
    state_means = (state_means-mean)/std
    (out/'normalization.json').write_text(json.dumps(dict(source_split='train', video_ids=ids['train'],
        mean=mean.tolist(),std=std.tolist(),initial_state_means=state_means.tolist(),input_version=VERSION),indent=2))
    (out/'coverage.json').write_text(json.dumps(dict(splits=ids,excluded_no_gt=excluded,
        missing_text=cache.n_missing_text),indent=2))
    loader = DataLoader(common.TrainDataset(corpus,ids['train'],labels,cache,cfg['max_seqlen']),
                        batch_size=32,shuffle=True,num_workers=args.num_workers)
    eval_loaders = {s: DataLoader(common.EvalDataset(corpus,ids[s],cache),batch_size=1,
                                 num_workers=args.num_workers) for s in ['val','test']}
    model = Candidate(mean,std,state_means,cfg['dropout'],args.ablation,cfg['max_seqlen']).to(args.device)
    fit_and_evaluate(model,loader,eval_loaders,gt,labels,args,cfg,say,
        lambda model,output,audio,lengths,label: model.loss(label,lengths))
    log.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corpus',choices=['hatemm','hateclipseg'],required=True)
    p.add_argument('--seed',type=int,default=234)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--config')
    p.add_argument('--ablation',choices=ARMS,default='full')
    p.add_argument('--device',default='cuda')
    p.add_argument('--num-workers',type=int,default=4)
    args = p.parse_args()
    cfg = DEFAULTS.copy()
    if args.config:
        supplied = json.loads(Path(args.config).read_text())
        if set(supplied)-cfg.keys():
            raise ValueError('undeclared hyperparameters')
        cfg.update(supplied)
    train(args,cfg)
