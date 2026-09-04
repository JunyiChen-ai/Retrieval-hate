#!/bin/bash
# candidate 4 revision 1 (null token inside candidate 1's exact training): Optuna search
# (candidate 1 search.py, --ablation null_token) for seeds 234/2025/3407, then the arms
# null_token_const / masked_no_token / full at each seed's best-trial hparams.
cd ~/Retrieval-hate
PY=~/miniconda3/envs/HateVideo/bin/python
EXP=experiments/20260903_hier_evidence_mil
RR=runs/20260904_null_token_cma/rev1
CORPUS=hateclipseg

git pull -q origin main
mkdir -p $RR
echo "start rev1 $CORPUS $(hostname) $(date)" >> $RR/chain_$CORPUS.log
for SEED in 234 2025 3407; do
  $PY $EXP/search.py --corpus $CORPUS --seed $SEED --ablation null_token --out-root $RR > $RR/search_${CORPUS}_seed$SEED.stdout 2>&1
  echo "search done $CORPUS $SEED $(date)" >> $RR/chain_$CORPUS.log
  BEST=$($PY -c "import json,sys; b=json.load(open('$RR/$CORPUS/seed$SEED/study_summary.json'))['best']; print('' if b is None else b['number'])")
  if [ -z "$BEST" ]; then echo "no valid trial $CORPUS $SEED" >> $RR/chain_$CORPUS.log; continue; fi
  CFG=$RR/$CORPUS/seed$SEED/trial$BEST/hparams.json
  for ARM in null_token_const masked_no_token full; do
    OUT=$RR/ablations/$CORPUS/seed$SEED/$ARM
    [ -f $OUT/metrics.json ] && continue
    $PY $EXP/train.py --corpus $CORPUS --seed $SEED --config $CFG --ablation $ARM --out-dir $OUT > /dev/null 2>&1
    echo "done $CORPUS $SEED $ARM $(date)" >> $RR/chain_$CORPUS.log
  done
  touch $RR/REV1_${CORPUS}_seed${SEED}_DONE
done
touch $RR/REV1_${CORPUS}_DONE
