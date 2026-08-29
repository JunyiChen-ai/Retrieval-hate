#!/usr/bin/env python3
"""Fail-closed verifier for the post-training/pre-temporal migration."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(path=HERE/'POST_TRAINING_PRE_TEMPORAL_MIGRATION.json',allow_post_access=False):
 m=json.load(open(path));keys={'schema','status','temporal_metrics_seen','test_read','change_scope','old_training_identities','immutable_artifacts','new_evaluation_identities'}
 if set(m)!=keys or m['schema']!='v25_post_training_pre_temporal_migration_v1' or m['status']!='FROZEN_AFTER_TRAINING_BEFORE_TEMPORAL_METRICS' or m['temporal_metrics_seen'] is not False or m['test_read'] is not False or m['change_scope']!=['provenance_validation','gt_discretization_center_to_any_overlap'] :raise RuntimeError('migration schema/scope')
 old=m['old_training_identities']
 if old!={'selector_sha256':'0a6ce65fac40c1362c2e64208201c8028b2139a1c5bd3842798f8e8a1b484c91','protocol_declaration_sha256':'7d1dd8f2ba3129bdd3ab3f237a4d6e3049614be7504a190428245f7a7fd297f8'}:raise RuntimeError('old training identities')
 for section in ('immutable_artifacts','new_evaluation_identities'):
  if not isinstance(m[section],dict) or not m[section]:raise RuntimeError('migration identities')
  for name,x in m[section].items():
   if set(x)!={'path','sha256'} or not Path(x['path']).is_absolute():raise RuntimeError('migration identity '+name)
   if sha(x['path'])!=x['sha256'] and not (allow_post_access and section=='new_evaluation_identities' and name=='selector'):raise RuntimeError('migration tamper '+name)
 return m
