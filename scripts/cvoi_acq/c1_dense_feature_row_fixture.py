"""Regression fixture: dense feature_row is action-level and must not be divided."""
from __future__ import annotations
import argparse
from pathlib import Path
from .common import atomic_json
from .ocr_embedding_bank import dense_outcome_ref

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    frames=[{"feature_row":1,"frame_slot":i} for i in range(4)]
    ref=dense_outcome_ref("train",frames)
    if ref!="artifacts/cvoi_acq/premetric-v2/visual-v10/train_dense4.f32:1":raise RuntimeError("HALT_DENSE_FEATURE_ROW_REGRESSION")
    failed=False
    try:dense_outcome_ref("train",[{"feature_row":0,"frame_slot":0},{"feature_row":1,"frame_slot":1},{"feature_row":1,"frame_slot":2},{"feature_row":1,"frame_slot":3}])
    except RuntimeError:failed=True
    if not failed:raise RuntimeError("HALT_DENSE_MIXED_ROW_ACCEPTED")
    atomic_json(a.out,{"schema":"cvoi-c1-dense-feature-row-fixture/1","passed":True,"input_action_row":1,"expected_ref":ref,"division_by_four_forbidden":True,"mixed_rows_rejected":True,"candidate_metric_computed":False})
if __name__=="__main__":main()
