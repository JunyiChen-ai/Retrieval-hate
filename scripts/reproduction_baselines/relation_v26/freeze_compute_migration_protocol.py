#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from artifacts import atomic,sha
from core import MIGRATION_SHA,ARCH
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--workload',required=True);p.add_argument('--audit',required=True);p.add_argument('--benchmark',required=True);p.add_argument('--out',required=True);a=p.parse_args();here=Path(__file__).resolve().parent;doc=here.parents[2]/'docs'/'V26_PRE_TRAIN_COMPUTE_MIGRATION.md'
 x={'schema':'v26_finite_rf_compute_protocol_v1','status':'PRETRAIN_FROZEN','migration_sha256':MIGRATION_SHA,'architecture':ARCH,'sources':{n:bind(here/n) for n in ('core.py','train.py','val_predict.py','reference.py','compute_migration_audit.py','benchmark_first20.py')},'design':bind(doc),'workload':bind(a.workload),'numerical_audit':bind(a.audit),'benchmark':bind(a.benchmark),'numerical_cuda_roundoff_nonblocking':True,'fundamental_gates':['finite_rf_index_boundary_mask','fast_slow_same_order_of_magnitude','all_parameter_gradients_finite_connected','full_T_no_sampling','epoch0_G_exact_effect_zero','three_arm_epoch0_state_match','train_only_inputs','projected_hours_le_12'],'formal_training_run':False,'validation_labels_or_gt_read':False,'test_read':False,'producer_sha256':sha(__file__)};atomic(a.out,x)
if __name__=='__main__':main()
