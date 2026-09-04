#!/bin/bash
# candidate 3 (evidence_guided_attention) full chain for hateclipseg: seed 234/2025/3407 Optuna search, then ablation arms with each seed's best trial hparams
cd ~/Retrieval-hate
PY=~/miniconda3/envs/HateVideo/bin/python
EXP=experiments/20260904_evidence_guided_attention
RR=runs/20260904_evidence_guided_attention_rev2

git pull -q origin main
mkdir -p $RR
echo "start hateclipseg $(hostname) $(date)" >> $RR/chain_hateclipseg.log
for SEED in 234 2025 3407; do
  $PY $EXP/search.py --corpus hateclipseg --seed $SEED --out-root $RR > $RR/search_hateclipseg_seed$SEED.stdout 2>&1
  echo "search done hateclipseg $SEED $(date)" >> $RR/chain_hateclipseg.log
  SEARCH=$RR/hateclipseg/seed$SEED
  BEST=$($PY -c "import json;print(json.load(open('$SEARCH/study_summary.json'))['best']['number'])")
  CFG=$SEARCH/trial$BEST/hparams.json
  ROOT=$RR/ablations/hateclipseg/seed$SEED
  for ABL in avce stream_enc no_qk_enc no_cell no_bias scalar_bias no_context no_block no_prior no_cmal mean_prior no_verdict; do
    OUT=$ROOT/$ABL
    if [ -f $OUT/metrics.json ]; then continue; fi
    mkdir -p $OUT
    $PY $EXP/train.py --corpus hateclipseg --seed $SEED --config $CFG --ablation $ABL --out-dir $OUT > $OUT/stdout.log 2>&1
    echo "done hateclipseg $SEED $ABL $(date)" >> $RR/chain_hateclipseg.log
  done
  touch $RR/C3R2_hateclipseg_seed${SEED}_DONE
done
touch $RR/C3R2_hateclipseg_DONE
