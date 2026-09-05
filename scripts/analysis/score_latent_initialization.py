"""Full fixed-test initialization reference, not a trained candidate or smoke run."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments/20260905_latent_evidence_sequence'))
import numpy as np
import torch
import hier_evidence_common as common
import vlm_verdict
from hate_common import data as hdata
from interventional_observations import load_observations
from model import Candidate

p = argparse.ArgumentParser()
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--out-dir',type=Path,required=True)
p.add_argument('--device',default='cuda')
a = p.parse_args()
a.out_dir.mkdir(parents=True,exist_ok=True)
log = (a.out_dir/'run.log').open('x')
def say(message):
    print(message,flush=True); log.write(message+'\n'); log.flush()
say(f'host={socket.gethostname()} code={common._git_describe()} diagnostic=initialization_only_full_test')
(a.out_dir/'run.pid').write_text(str(os.getpid()))
summary = json.loads((a.reference/'summary.json').read_text())
assert summary['ablation']=='full'
corpus, seed, cfg = summary['corpus'],summary['seed'],summary['hparams']
norm = json.loads((a.reference/'normalization.json').read_text())
assert norm['source_split']=='train' and norm['video_ids']==hdata.load_split(corpus,'train')
ids = json.loads((a.reference/'coverage.json').read_text())['splits']['test']
expected = hdata.load_split(corpus,'test')
expected = [v for v in expected if not (corpus=='hatemm' and v=='hate_video_427')]
assert ids == expected
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
model = Candidate(norm['mean'],norm['std'],norm['initial_state_means'],cfg['dropout'],'full',cfg['max_seqlen']).to(a.device).eval()
# At initialization both these weights are zero, so all content/crop inputs are
# exactly irrelevant to rates/initial probabilities. Do not make this shortcut
# for any trained checkpoint. No optimized weights are loaded here.
assert torch.count_nonzero(model.transition.weight)==0
assert torch.count_nonzero(model.initial.weight)==0
(a.out_dir/'config.json').write_text(json.dumps(dict(reference=str(a.reference),corpus=corpus,seed=seed,
    hparams=cfg,normalization_source=str(a.reference/'normalization.json'),
    optimized_weights_loaded=False,training_steps=0,split='test',n_videos=len(ids),
    note='One crop equals all five exactly because initialized content-to-transition/initial weights are zero.'),indent=2))
scores = {}
with torch.no_grad():
    for i,vid in enumerate(ids):
        audio = common.align.load_audio(corpus,vid)
        visual = common.align.load_visual_crop(corpus,vid,0)
        duration = audio.shape[0]
        snip = common.align.snippet_bounds(corpus,vid,visual.shape[0])
        extra=[]
        for k in [30,4]:
            logits,_ = load_observations(ROOT/'data/interventional_evidence'/corpus/f'K{k}'/f'{vid}.json',k,vid)
            extra.append(vlm_verdict.verdict_rows(logits,snip,duration))
        observations=np.concatenate(extra,-1)
        fa=torch.zeros((1,len(snip),common.A_EXT_DIM+8),device=a.device)
        fa[...,common.A_EXT_DIM:]=torch.as_tensor(observations,device=a.device)
        fv=torch.zeros((1,len(snip),1024),device=a.device)
        logits=model(fa,fv)[3]
        posterior=logits.sigmoid()[0,:,0].cpu().numpy()
        scores[vid]=posterior[common.align.snippet_index_for_seconds(snip,duration)]
        if (i+1)%20==0:
            say(f'completed={i+1}/{len(ids)}')
common.write_scores(str(a.out_dir/'scores_test.jsonl'),scores)
m=common.run_evaluator(corpus,'test',str(a.out_dir/'scores_test.jsonl'),str(a.out_dir/'metrics.json'))
assert m['results']['score_av']['n_videos']==len(ids)
say('METRICS '+json.dumps(m['results']['score_av']))
(a.out_dir/'completion.json').write_text(json.dumps(dict(state='INITIALIZATION_REFERENCE_FINISHED',n_videos=len(ids))))
log.close()
