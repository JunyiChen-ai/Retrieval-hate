#!/usr/bin/env python3
import json,torch
from pathlib import Path
from artifacts import sha
from core import CTW,DESIGN_SHA
def fallback(g,T):return [g for _ in range(T)]
def load_authorized(selection,seal,test_features):
 s=json.load(open(selection));z=json.load(open(seal))
 if s.get('design_sha256')!=DESIGN_SHA or s.get('status')!='FINAL_PASS' or s.get('test_authorized') is not True:raise RuntimeError('test not authorized')
 if set(z)!={'schema','design_sha256','selection_sha256','test_feature_path','test_feature_sha256','status'} or z['schema']!='v26_test_seal_v1' or z['design_sha256']!=DESIGN_SHA or z['status']!='FINAL_PASS' or z['selection_sha256']!=sha(selection) or z['test_feature_path']!=str(Path(test_features).resolve()) or z['test_feature_sha256']!=sha(test_features):raise RuntimeError('test seal/input')
 return s
def frame_scores(model,xs,masks,b,g):return torch.sigmoid(torch.clamp(model.effects(xs,masks,b,g),-12,12))
