#!/usr/bin/env bash
# R12-ANCHOR -- the single frozen run.  idea-stage/R12_FREEZE.md section 3.
# One submission: build (PT teacher + focal/shuffled weight files), then MHC_zh
# (30 seeds, 900-929) and HateMM (15 seeds, 900-914), then the judgement read-out,
# the dev-side panel, the union accounting and the frozen verdict.
# Nothing here selects anything on test.
set -uo pipefail
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

R11=idea-stage/r11_union
R12=idea-stage/r12_anchor
LAM=0.1   # frozen, R12_FREEZE.md 3.2 -- NOT dev-selected

# TAG:MODEL:FUSION:TEACHER:LAMBDA:WEIGHTS      ("-" = absent)
build_spec () {  # $1 = dataset
  local D="$1"
  local S="CAT:R10CB-CAT:align:-:0:-"
  S="$S,AU_A0:R10CB-CAT:align:${R11}/teacher_${D}_A0.json:${LAM}:-"
  S="$S,AF_A0:R10CB-CAT:align:${R11}/teacher_${D}_A0.json:${LAM}:${R12}/w_${D}_A0.json"
  S="$S,AU_PT:R10CB-CAT:align:${R12}/teacher_${D}_PT.json:${LAM}:-"
  S="$S,AF_PT:R10CB-CAT:align:${R12}/teacher_${D}_PT.json:${LAM}:${R12}/w_${D}_PT.json"
  S="$S,AF_SHUF:R10CB-CAT:align:${R12}/teacher_${D}_PT.json:${LAM}:${R12}/wshuf_${D}_PT.json"
  S="$S,LBL:R10CB-CAT:align:${R11}/teacher_${D}_LBL.json:${LAM}:-"
  echo "$S"
}

ARMS="CAT,AU_A0,AF_A0,AU_PT,AF_PT,AF_SHUF,LBL"
CONTRASTS="AF_PT-CAT,AF_PT-AU_PT,AF_PT-AF_SHUF,AF_A0-CAT,AF_A0-AU_A0,AF_A0-AF_SHUF,AU_PT-CAT,AU_A0-CAT,LBL-CAT,AF_PT-LBL"

ZH_SEEDS=$(seq -s, 900 929)
HM_SEEDS=$(seq -s, 900 914)

echo "=== R12-ANCHOR start $(date -Is) ==="

echo "=== build $(date -Is) ==="
python ${R12}/build_r12a.py --dataset MHC_zh || exit 1
python ${R12}/build_r12a.py --dataset HateMM || exit 1

echo "=== MHC_zh grid start $(date -Is) seeds=$ZH_SEEDS ==="
bash ${R12}/run_anchor_grid.sh logging/runs/r12_anchor/zh "$(build_spec MHC_zh)" \
  MHC_zh "$ZH_SEEDS" R12AN

echo "=== HateMM grid start $(date -Is) seeds=$HM_SEEDS ==="
bash ${R12}/run_anchor_grid.sh logging/runs/r12_anchor/hm "$(build_spec HateMM)" \
  HateMM "$HM_SEEDS" R12AN

echo "=== judgement read-out $(date -Is) ==="
python idea-stage/reaudit/analyze_grid.py \
  --logdir logging/runs/r12_anchor/zh/logs --dataset MHC_zh \
  --arms "$ARMS" --seeds "$ZH_SEEDS" --contrasts "$CONTRASTS" \
  --out ${R12}/zh_grid.json || exit 1
python idea-stage/reaudit/analyze_grid.py \
  --logdir logging/runs/r12_anchor/hm/logs --dataset HateMM \
  --arms "$ARMS" --seeds "$HM_SEEDS" --contrasts "$CONTRASTS" \
  --out ${R12}/hm_grid.json || exit 1

echo "=== dev panel (demotion clause) $(date -Is) ==="
python idea-stage/r10_combo/analyze_dev_panel.py \
  --logdir logging/runs/r12_anchor/zh/logs --dataset MHC_zh \
  --arms "$ARMS" --seeds "$ZH_SEEDS" --contrasts "$CONTRASTS" \
  --out ${R12}/zh_devpanel.json || exit 1
python idea-stage/r10_combo/analyze_dev_panel.py \
  --logdir logging/runs/r12_anchor/hm/logs --dataset HateMM \
  --arms "$ARMS" --seeds "$HM_SEEDS" --contrasts "$CONTRASTS" \
  --out ${R12}/hm_devpanel.json || exit 1

echo "=== union accounting (secondary, no verdict power) $(date -Is) ==="
python idea-stage/r10_combo/diag_errors.py \
  --logdir logging/runs/r12_anchor/zh/logs --dataset MHC_zh \
  --arms "$ARMS" --seeds "$ZH_SEEDS" --out ${R12}/zh_errors.json || exit 1
python idea-stage/r10_combo/diag_errors.py \
  --logdir logging/runs/r12_anchor/hm/logs --dataset HateMM \
  --arms "$ARMS" --seeds "$HM_SEEDS" --out ${R12}/hm_errors.json || exit 1

echo "=== frozen verdict $(date -Is) ==="
python ${R12}/verdict.py --zh ${R12}/zh_grid.json --hm ${R12}/hm_grid.json \
  --zhdev ${R12}/zh_devpanel.json --hmdev ${R12}/hm_devpanel.json \
  --out ${R12}/verdict.json

echo "=== R12-ANCHOR ALLDONE $(date -Is) ==="
