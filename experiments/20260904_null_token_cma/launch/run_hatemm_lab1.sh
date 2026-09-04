#!/bin/bash
# candidate 4 (null_token_cma) full chain for hatemm: seed 234/2025/3407 Optuna search, then ablation arms with each seed's best trial hparams
cd ~/Retrieval-hate
PY=~/miniconda3/envs/HateVideo/bin/python
EXP=experiments/20260904_null_token_cma
RR=runs/20260904_null_token_cma

git pull -q origin main
mkdir -p $RR
echo "start hatemm $(hostname) $(date)" >> $RR/chain_hatemm.log
for SEED in 234 2025 3407; do
  $PY $EXP/search.py --corpus hatemm --seed $SEED --out-root $RR > $RR/search_hatemm_seed$SEED.stdout 2>&1
  echo "search done hatemm $SEED $(date)" >> $RR/chain_hatemm.log
  SEARCH=$RR/hatemm/seed$SEED
  BEST=$($PY -c "import json;b=json.load(open('$SEARCH/study_summary.json'))['best'];print(b['number'] if b else '')")
  if [ -z "$BEST" ]; then echo "no valid best hatemm $SEED, arms skipped $(date)" >> $RR/chain_hatemm.log; touch $RR/C4_hatemm_seed${SEED}_DONE; continue; fi
  CFG=$SEARCH/trial$BEST/hparams.json
  ROOT=$RR/ablations/hatemm/seed$SEED
  for ABL in no_token_unmasked no_token_masked const_token shared_token zero_value_sink gated_cma no_input no_block no_prior no_cmal mean_prior no_verdict; do
    OUT=$ROOT/$ABL
    if [ -f $OUT/metrics.json ]; then continue; fi
    mkdir -p $OUT
    $PY $EXP/train.py --corpus hatemm --seed $SEED --config $CFG --ablation $ABL --out-dir $OUT > $OUT/stdout.log 2>&1
    echo "done hatemm $SEED $ABL $(date)" >> $RR/chain_hatemm.log
  done
  touch $RR/C4_hatemm_seed${SEED}_DONE
done
touch $RR/C4_hatemm_DONE
