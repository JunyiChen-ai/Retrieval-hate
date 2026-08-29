#!/usr/bin/env python3
"""Verify the honest post-access identity-only correction."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def correction_verify_declaration(correction_path=None,declaration_path=None):
 """Strict historical declaration plus the two correction-authorized drifts."""
 c=verify(correction_path);p=Path(declaration_path or HERE/'FINAL_PROTOCOL_DECLARATION.json');d=json.load(open(p))
 from core import canon_hash
 if set(d)!={'schema','status','supersedes','canonical_derivation','identities','identity_set_sha256','labels_or_gt_read','test_read'} or d['identity_set_sha256']!=canon_hash(d['identities']):raise RuntimeError('historical declaration schema/hash')
 allowed={'selector':('2c6267c239faf0ea4921c4ad4a8fe006c21992a60bcb53035d2b1b0e59e09b66',c['new_identities']['selector']['sha256']),'post_training_migration_verifier':('2a8f49c165c1e02d07b697bff33295420426ea8676ebeec5dfbed5918b1a8182',c['new_identities']['historical_migration_verifier']['sha256'])}
 for name,x in d['identities'].items():
  actual=sha(x['path'])
  if actual==x['sha256']:continue
  if name not in allowed or (x['sha256'],actual)!=allowed[name]:raise RuntimeError('unauthorized declaration identity drift '+name)
 return d
def correction_verify_outputs(root,ids_sha,epoch,checkpoints,states,reference_sha,correction_path=None):
 """Correction-aware equivalent of frozen val_predict.verify_outputs."""
 c=verify(correction_path);d=correction_verify_declaration(correction_path);root=Path(root);m=json.load(open(root/'manifest.json'));keys={'schema','status','epoch','ids_sha256','evidence_manifest_sha256','evidence_config_sha256','reference_manifest_sha256','checkpoint_sha256_by_seed','state_sha256_by_seed','files','shuffle_rule','producer_sha256','reducer_sha256','labels_read','test_read'};ve=d['identities']['authoritative_val_evidence_manifest'];vc=d['identities']['authoritative_val_evidence_config']
 if set(m)!=keys or m['schema']!='v25_val_predictions_v1' or m['status']!='VAL_LABEL_GT_BLIND' or m['labels_read'] is not False or m['test_read'] is not False or m['epoch']!=epoch or m['ids_sha256']!=ids_sha or m['checkpoint_sha256_by_seed']!=checkpoints or m['state_sha256_by_seed']!=states or m['reference_manifest_sha256']!=reference_sha or m['evidence_manifest_sha256']!=ve['sha256'] or m['evidence_config_sha256']!=vc['sha256'] or m['producer_sha256']!='986f6a6af7956bae144fbd726472bd4933b53df4eae1161933c9d7c745cf2deb' or m['reducer_sha256']!=c['unchanged_evaluation_bindings']['prediction_reducer_sha256']:raise RuntimeError('val prediction identity')
 from core import SEEDS
 expected={f'seed{s}_{k}.jsonl' for s in SEEDS for k in ('raw','shuffle')}
 if set(m['files'])!=expected or any(sha(root/n)!=m['files'][n] for n in expected):raise RuntimeError('val prediction raw tamper')
 return m
def verify(path=None):
 path=Path(path or HERE/'POST_ACCESS_PROVENANCE_CORRECTION_V2.json');raw=json.load(open(path));v2=raw.get('schema')=='v25_post_access_provenance_correction_v2'
 if v2:
  if set(raw)!={'schema','status','supersedes','new_identities'} or raw['status']!='POST_ACCESS_IDENTITY_REPAIR_ONLY':raise RuntimeError('v2 correction schema')
  s=raw['supersedes']
  if set(s)!={'path','sha256','reason'} or sha(s['path'])!=s['sha256'] or s['sha256']!='90e1bce94f56a9c958ccc4a75a28ca4c7376d61888047fff00a06e5b10f6193c':raise RuntimeError('correction chain')
  c=json.load(open(s['path']));c['new_identities']=raw['new_identities']
 else:c=raw
 base={'schema','status','aggregate_seen','observed_aggregate','change_scope','semantic_scoring_changed','gt_arrays_changed','predictions_changed','metrics_or_gates_changed','old_identities','new_identities','frozen_artifacts','unchanged_evaluation_bindings'}
 if set(c)!=base or c['schema']!='v25_post_access_provenance_correction_v1' or c['status']!='POST_ACCESS_IDENTITY_REPAIR_ONLY' or c['aggregate_seen'] is not True or c['observed_aggregate']!={'within_macro_roc':0.5667818116520346,'paired_ci_gate_pass':False} or c['change_scope']!=['generator_identity_repair'] or any(c[k] is not False for k in ('semantic_scoring_changed','gt_arrays_changed','predictions_changed','metrics_or_gates_changed')):raise RuntimeError('correction schema/honesty')
 if c['old_identities']!={'pre_addendum_sha256':'9f71355569dc2a9f31d7de3668888cb897381bf6608068e3816545904d853dc8','pre_addendum_generator_sha256':'cbf26c832453ed17c7986d7ef5a7dcdaa4c8f2bb09cad1cdbf851f78a8a3062f','migration_sha256':'518c83c2aa393c010127c7fdf763e1547d9811de2657ac50f08d6ac4969a61c8','selector_sha256':'2c6267c239faf0ea4921c4ad4a8fe006c21992a60bcb53035d2b1b0e59e09b66'}:raise RuntimeError('wrong old identities')
 for sec in ('new_identities','frozen_artifacts'):
  for name,x in c[sec].items():
   if set(x)!={'path','sha256'} or sha(x['path'])!=x['sha256']:raise RuntimeError('correction tamper '+name)
 b=c['unchanged_evaluation_bindings']
 if set(b)!={'taxonomy_sha256','prediction_manifest_sha256','prediction_reducer_sha256','temporal_report_sha256','reference_manifest_sha256','overlap_rule_sha256'}:raise RuntimeError('evaluation binding schema')
 man=json.load(open(c['frozen_artifacts']['encrypted_manifest']['path']));pred=json.load(open(c['frozen_artifacts']['prediction_manifest']['path']))
 if man['bindings']['taxonomy_sha256']!=b['taxonomy_sha256'] or pred['reducer_sha256']!=b['prediction_reducer_sha256'] or sha(c['frozen_artifacts']['prediction_manifest']['path'])!=b['prediction_manifest_sha256'] or sha(c['frozen_artifacts']['stale_temporal_report']['path'])!=b['temporal_report_sha256'] or sha(HERE/'temporal_eval_rule.py')!=b['overlap_rule_sha256']:raise RuntimeError('evaluation identities changed')
 return c
