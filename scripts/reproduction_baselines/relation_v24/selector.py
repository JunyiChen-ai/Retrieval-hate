#!/usr/bin/env python3
"""Video-label-only validation selector; freezes config before test inference."""
import argparse,json,sys
from pathlib import Path
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import V24
from train import sha,load_bags,load_global_source,corpus_sha
from steward_val_join import load_val_manifest
from protocol_migration import load as load_addendum
def load_epoch(path,epoch):
 x=torch.load(path,weights_only=False);m=V24();m.load_state_dict(x['history'][epoch]['state']);return m
def measure(model,bags):
 y=[];s=[]
 with torch.no_grad():
  for _,b in sorted(bags.items()):y.append(b['label']);s.append(float(model(b['global'],b['families'])[0]))
 if len(set(y))!=2:raise RuntimeError('validation needs both video classes')
 return {'video_ap':float(average_precision_score(y,s)),'video_roc':float(roc_auc_score(y,s))}
def evaluate_surfaces(paths,bags):
 """All arms are measured on the identical original frozen validation bags."""
 return {arm:[measure(load_epoch(path,e),bags) for e in range(6)] for arm,path in paths.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--train-dir',required=True);p.add_argument('--protocol-addendum',required=True);p.add_argument('--val-bags',required=True);p.add_argument('--val-id-manifest',required=True);p.add_argument('--v23-global-source',required=True);p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args();td=Path(a.train_dir);tp=json.load(open(td/'train_protocol.json'));ad=load_addendum(a.protocol_addendum)
 if tp['corpus_sha256']!=corpus_sha(a.corpus):raise RuntimeError('train protocol corpus mismatch')
 if ad['corpus']!=a.corpus or ad['train_protocol_sha256']!=sha(td/'train_protocol.json'):raise RuntimeError('migration/train protocol mismatch')
 if ad['selector_source_sha256']!=sha(Path(__file__).resolve()):raise RuntimeError('selector source not bound by migration')
 im=load_val_manifest(a.val_id_manifest,a.corpus)
 if im['evidence_producer_sha256']!=ad['train_evidence_producer_sha256']:raise RuntimeError('train/val evidence producer identity differs')
 if im['join_producer_sha256']!=ad['val_join_source_sha256']:raise RuntimeError('val join producer differs from addendum')
 if tp['producer_sha256']!=ad['train_join_producer_sha256']:raise RuntimeError('immutable train join differs from addendum')
 if sha(a.val_bags)!=im['bags_sha256']:raise RuntimeError('val bags hash mismatch')
 bags=load_bags(a.val_bags,im['ids'],a.corpus,'val',im['v23_global_source_sha256'],True);gs=load_global_source(a.v23_global_source,im['ids'],im['v23_global_source_sha256'])
 if any(bags[v]['global']!=gs[v] for v in gs):raise RuntimeError('val epoch0 global source mismatch')
 paths={'real':td/'real_local.pt','permuted':td/'permuted_local_negative_control.pt','global_only':td/'matched_global_only.pt'};surf=evaluate_surfaces(paths,bags)
 best={arm:max(range(6),key=lambda e:(surf[arm][e]['video_ap'],surf[arm][e]['video_roc'],-e)) for arm in paths};r=surf['real'][best['real']];g=surf['global_only'][best['global_only']];n=surf['permuted'][best['permuted']];sm=load_epoch(paths['real'],best['real'])
 gamma=max(0.,float(sm.gamma.detach()));weights=torch.softmax(sm.family_logits.detach(),0).tolist();diagnostics={'raw_gamma':float(sm.gamma.detach()),'effective_gamma':gamma,'family_names':list(sm.families),'family_softmax_weights':[float(x) for x in weights],'max_family_weight':float(max(weights))}
 gates={'ap_noninferior_epoch0':r['video_ap']>=surf['real'][0]['video_ap']-.002,'roc_noninferior_epoch0':r['video_roc']>=surf['real'][0]['video_roc']-.005,'gain_over_global':r['video_ap']-g['video_ap']>=.005,'gain_over_permuted':r['video_ap']-n['video_ap']>=.005,'effective_gamma_min_0.01':gamma>=.01,'max_family_weight_le_0.95':max(weights)<=.95};passed=all(gates.values());chosen=best['real'] if passed else 0;status='VIDEO_VAL_PASS_PENDING_TEMPORAL' if passed else 'VIDEO_VAL_FAIL_FALLBACK_EPOCH0';selected=load_epoch(paths['real'],chosen)
 frozen={'status':status,'corpus':a.corpus,'selected_arm':'real' if chosen else 'epoch0_v23_global','selected_epoch':chosen,'gates':gates,'all_video_gates_pass':passed,'selected_real_diagnostics':diagnostics,'best_epochs':best,'surface':surf,'train_protocol_sha256':sha(td/'train_protocol.json'),'protocol_addendum_sha256':sha(a.protocol_addendum),'val_bags_sha256':sha(a.val_bags),'val_id_manifest_sha256':sha(a.val_id_manifest),'val_v23_global_source_sha256':im['v23_global_source_sha256'],'producer_sha256':im['evidence_producer_sha256'],'evidence_producer_sha256':im['evidence_producer_sha256'],'train_join_producer_sha256':ad['train_join_producer_sha256'],'val_join_producer_sha256':im['join_producer_sha256'],'selected_state':{k:v.tolist() for k,v in selected.state_dict().items()},'test_labels_opened':False,'temporal_steward_gate_signed':False};Path(a.out).write_text(json.dumps(frozen,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
