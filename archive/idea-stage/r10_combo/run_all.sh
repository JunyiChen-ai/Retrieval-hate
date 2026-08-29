#!/usr/bin/env bash
# R10-COMBO -- the single frozen run.  idea-stage/R10_COMBO_FREEZE.md.
# Arms already built by build_combo.py (no metric computed there).
# One submission: MHC_zh (30 seeds) then HateMM (15 seeds), then the two
# diagnostics.  Nothing here selects anything on test.
set -uo pipefail
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

ARMS="A0,LL,CAT,PC0,K1,K2,K3,K4,K5,K6"
SPEC="A0:R10CB-A0,LL:R10CB-LL,CAT:R10CB-CAT,PC0:R10CB-PC0,K1:R10CB-K1,K2:R10CB-K2,K3:R10CB-K3,K4:R10CB-K4,K5:R10CB-K5,K6:R10CB-K6"
CONTRASTS="K1-LL,K1-CAT,K2-LL,K2-CAT,K3-LL,K3-CAT,K4-LL,K4-CAT,K5-LL,K5-CAT,K6-LL,K6-CAT,K1-PC0,K2-PC0,K6-PC0,CAT-A0,LL-A0,CAT-LL,PC0-A0"

ZH_SEEDS=$(seq -s, 600 629)
HM_SEEDS=$(seq -s, 600 614)

echo "=== R10-COMBO start $(date -Is) ==="

# ---------------- MHC_zh, primary ----------------
echo "=== MHC_zh grid start $(date -Is) seeds=$ZH_SEEDS ==="
bash idea-stage/r10_combo/run_combo_grid.sh logging/runs/r10_combo/zh "$SPEC" MHC_zh "$ZH_SEEDS" R10CB

echo "=== MHC_zh analyse $(date -Is) ==="
python idea-stage/reaudit/analyze_grid.py \
  --logdir logging/runs/r10_combo/zh/logs --dataset MHC_zh \
  --arms "$ARMS" --seeds "$ZH_SEEDS" --contrasts "$CONTRASTS" \
  --out idea-stage/r10_combo/zh_grid.json
python idea-stage/r10_combo/analyze_dev_panel.py \
  --logdir logging/runs/r10_combo/zh/logs --dataset MHC_zh \
  --arms "$ARMS" --seeds "$ZH_SEEDS" --contrasts "$CONTRASTS" \
  --out idea-stage/r10_combo/zh_devpanel.json
python idea-stage/r10_combo/diag_errors.py \
  --logdir logging/runs/r10_combo/zh/logs --dataset MHC_zh \
  --arms A0,LL,CAT,K3,K5 --seeds "$ZH_SEEDS" \
  --out idea-stage/r10_combo/zh_errors.json

# ---------------- HateMM, second dataset ----------------
echo "=== HateMM grid start $(date -Is) seeds=$HM_SEEDS ==="
bash idea-stage/r10_combo/run_combo_grid.sh logging/runs/r10_combo/hm "$SPEC" HateMM "$HM_SEEDS" R10CB

echo "=== HateMM analyse $(date -Is) ==="
python idea-stage/reaudit/analyze_grid.py \
  --logdir logging/runs/r10_combo/hm/logs --dataset HateMM \
  --arms "$ARMS" --seeds "$HM_SEEDS" --contrasts "$CONTRASTS" \
  --out idea-stage/r10_combo/hm_grid.json
python idea-stage/r10_combo/analyze_dev_panel.py \
  --logdir logging/runs/r10_combo/hm/logs --dataset HateMM \
  --arms "$ARMS" --seeds "$HM_SEEDS" --contrasts "$CONTRASTS" \
  --out idea-stage/r10_combo/hm_devpanel.json
python idea-stage/r10_combo/diag_errors.py \
  --logdir logging/runs/r10_combo/hm/logs --dataset HateMM \
  --arms A0,LL,CAT,K3,K5 --seeds "$HM_SEEDS" \
  --out idea-stage/r10_combo/hm_errors.json

# ---------------- representation diagnostic (train only, no labels) ----------------
echo "=== repr diagnostic $(date -Is) ==="
python idea-stage/r10_combo/diag_repr.py --dataset MHC_zh \
  --base "Qwen2.5-VL-7B-Instruct-LoRA_HF" --out idea-stage/r10_combo/zh_repr.json
python idea-stage/r10_combo/diag_repr.py --dataset HateMM \
  --base "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF" --out idea-stage/r10_combo/hm_repr.json

echo "=== R10-COMBO ALLDONE $(date -Is) ==="
