#!/bin/bash
# candidate 4 (null_token_cma) full chain for hateclipseg: seed 234/2025/3407 Optuna search, then ablation arms with each seed's best trial hparams
cd ~/Retrieval-hate
PY=~/miniconda3/envs/HateVideo/bin/python
EXP=experiments/20260904_null_token_cma
RR=runs/20260904_null_token_cma
while [ ! -f runs/20260904_evidence_guided_attention_rev2/C3R2_hateclipseg_DONE ]; do sleep 120; done
git pull -q origin main
mkdir -p $RR
echo "start hateclipseg $(hostname) $(date)" >> $RR/chain_hateclipseg.log
for SEED in 234 2025 3407; do
  $PY $EXP/search.py --corpus hateclipseg --seed $SEED --out-root $RR > $RR/search_hateclipseg_seed$SEED.stdout 2>&1
  echo "search done hateclipseg $SEED $(date)" >> $RR/chain_hateclipseg.log
  SEARCH=$RR/hateclipseg/seed$SEED
  BEST=$($PY -c "import json;b=json.load(open('$SEARCH/study_summary.json'))['best'];print(b['number'] if b else '')")
  if [ -z "$BEST" ]; then echo "no valid best hateclipseg $SEED, arms skipped $(date)" >> $RR/chain_hateclipseg.log; touch $RR/C4_hateclipseg_seed${SEED}_DONE; continue; fi
  CFG=$SEARCH/trial$BEST/hparams.json
  ROOT=$RR/ablations/hateclipseg/seed$SEED
  for ABL in no_token_unmasked no_token_masked const_token shared_token zero_value_sink gated_cma no_input no_block no_prior no_cmal mean_prior no_verdict; do
    OUT=$ROOT/$ABL
    if [ -f $OUT/metrics.json ]; then continue; fi
    mkdir -p $OUT
    $PY $EXP/train.py --corpus hateclipseg --seed $SEED --config $CFG --ablation $ABL --out-dir $OUT > $OUT/stdout.log 2>&1
    echo "done hateclipseg $SEED $ABL $(date)" >> $RR/chain_hateclipseg.log
  done
  touch $RR/C4_hateclipseg_seed${SEED}_DONE
done
touch $RR/C4_hateclipseg_DONE
