#!/usr/bin/env bash
# R12-IMG -- the single frozen run.  idea-stage/R12_FREEZE.md section 2.
# One submission: build the 8 arm caches from the -tp (text) and -ip (img) caches,
# then MHC_zh (30 seeds, 800-829) and HateMM (15 seeds, 800-814), then the judgement
# read-out, the dev-side panel and the frozen verdict.
# The extraction pass (idea-stage/r12_img/extract_img.py) runs BEFORE this script and
# is driven by logging/runs/r12_extract/drive.sh.  Nothing here selects on test.
set -uo pipefail
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

R12=idea-stage/r12_img
ARMS="I0,ISPLIT,I2M,IRSPLIT,IRW,IVIS,IINS,ISTD"
CONTRASTS="ISPLIT-I0,ISPLIT-IRW,ISPLIT-IRSPLIT,I2M-I0,I2M-IRW,I2M-IRSPLIT,IRW-I0,IRSPLIT-I0,IVIS-I0,IINS-I0,ISTD-I0,ISPLIT-I2M"

# TAG:MODEL:FUSION
SPEC="I0:R12IM-I0,ISPLIT:R12IM-ISPLIT,I2M:R12IM-I2M,IRSPLIT:R12IM-IRSPLIT,IRW:R12IM-IRW,IVIS:R12IM-IVIS,IINS:R12IM-IINS,ISTD:R12IM-ISTD"

ZH_SEEDS=$(seq -s, 800 829)
HM_SEEDS=$(seq -s, 800 814)

echo "=== R12-IMG start $(date -Is) ==="

echo "=== build $(date -Is) ==="
python ${R12}/build_img.py --dataset MHC_zh \
  --base "Qwen2.5-VL-7B-Instruct-LoRA_HF" || exit 1
python ${R12}/build_img.py --dataset HateMM \
  --base "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF" || exit 1

echo "=== MHC_zh grid start $(date -Is) seeds=$ZH_SEEDS ==="
bash idea-stage/reaudit/run_grid.sh logging/runs/r12_img/zh "$SPEC" MHC_zh "$ZH_SEEDS" R12IM

echo "=== HateMM grid start $(date -Is) seeds=$HM_SEEDS ==="
bash idea-stage/reaudit/run_grid.sh logging/runs/r12_img/hm "$SPEC" HateMM "$HM_SEEDS" R12IM

echo "=== judgement read-out $(date -Is) ==="
python idea-stage/reaudit/analyze_grid.py \
  --logdir logging/runs/r12_img/zh/logs --dataset MHC_zh \
  --arms "$ARMS" --seeds "$ZH_SEEDS" --contrasts "$CONTRASTS" \
  --out ${R12}/zh_grid.json || exit 1
python idea-stage/reaudit/analyze_grid.py \
  --logdir logging/runs/r12_img/hm/logs --dataset HateMM \
  --arms "$ARMS" --seeds "$HM_SEEDS" --contrasts "$CONTRASTS" \
  --out ${R12}/hm_grid.json || exit 1

echo "=== dev panel (demotion clause) $(date -Is) ==="
python idea-stage/r10_combo/analyze_dev_panel.py \
  --logdir logging/runs/r12_img/zh/logs --dataset MHC_zh \
  --arms "$ARMS" --seeds "$ZH_SEEDS" --contrasts "$CONTRASTS" \
  --out ${R12}/zh_devpanel.json || exit 1
python idea-stage/r10_combo/analyze_dev_panel.py \
  --logdir logging/runs/r12_img/hm/logs --dataset HateMM \
  --arms "$ARMS" --seeds "$HM_SEEDS" --contrasts "$CONTRASTS" \
  --out ${R12}/hm_devpanel.json || exit 1

echo "=== frozen verdict $(date -Is) ==="
python ${R12}/verdict.py --zh ${R12}/zh_grid.json --hm ${R12}/hm_grid.json \
  --zhdev ${R12}/zh_devpanel.json --hmdev ${R12}/hm_devpanel.json \
  --out ${R12}/verdict.json

echo "=== R12-IMG ALLDONE $(date -Is) ==="
