#!/usr/bin/env bash
# R11-UNION -- the single frozen run.  idea-stage/R11_UNION_FREEZE.md.
# One submission: build (MC cache + the three anchor teachers), then MHC_zh
# (30 seeds) and HateMM (15 seeds), then the read-out and the frozen verdict.
# Nothing here selects anything on test.
set -uo pipefail
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

T_ZH=idea-stage/r11_union/teacher_MHC_zh
T_HM=idea-stage/r11_union/teacher_HateMM

# TAG:MODEL:FUSION:TEACHER:LAMBDA   ("-" = no anchor term)
SPEC_ZH="A0:R10CB-A0:align:-:0,LL:R10CB-LL:align:-:0,CAT:R10CB-CAT:align:-:0,MC:R11UN-MC:align:-:0"
SPEC_ZH="$SPEC_ZH,ANCA_l01:R10CB-CAT:align:${T_ZH}_A0.json:0.1"
SPEC_ZH="$SPEC_ZH,ANCA_l03:R10CB-CAT:align:${T_ZH}_A0.json:0.3"
SPEC_ZH="$SPEC_ZH,ANCA_l10:R10CB-CAT:align:${T_ZH}_A0.json:1.0"
SPEC_ZH="$SPEC_ZH,ANCL_l01:R10CB-CAT:align:${T_ZH}_LL.json:0.1"
SPEC_ZH="$SPEC_ZH,ANCL_l03:R10CB-CAT:align:${T_ZH}_LL.json:0.3"
SPEC_ZH="$SPEC_ZH,ANCL_l10:R10CB-CAT:align:${T_ZH}_LL.json:1.0"
SPEC_ZH="$SPEC_ZH,LBL_l01:R10CB-CAT:align:${T_ZH}_LBL.json:0.1"
SPEC_ZH="$SPEC_ZH,LBL_l03:R10CB-CAT:align:${T_ZH}_LBL.json:0.3"
SPEC_ZH="$SPEC_ZH,LBL_l10:R10CB-CAT:align:${T_ZH}_LBL.json:1.0"

SPEC_HM="${SPEC_ZH//${T_ZH}/${T_HM}}"

SPEC_B="CATB:R10CB-CAT:align:-:0"

ZH_SEEDS=$(seq -s, 700 729)
HM_SEEDS=$(seq -s, 700 714)
ZH_SEEDS_B=$(seq -s, 50700 50729)
HM_SEEDS_B=$(seq -s, 50700 50714)

echo "=== R11-UNION start $(date -Is) ==="

echo "=== build $(date -Is) ==="
python idea-stage/r11_union/build_r11.py --dataset MHC_zh \
  --base "Qwen2.5-VL-7B-Instruct-LoRA_HF" || exit 1
python idea-stage/r11_union/build_r11.py --dataset HateMM \
  --base "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF" || exit 1

# ---------------- MHC_zh ----------------
echo "=== MHC_zh grid start $(date -Is) seeds=$ZH_SEEDS ==="
bash idea-stage/r11_union/run_union_grid.sh logging/runs/r11_union/zh "$SPEC_ZH" MHC_zh "$ZH_SEEDS" R11UN
echo "=== MHC_zh CATB grid start $(date -Is) seeds=$ZH_SEEDS_B ==="
bash idea-stage/r11_union/run_union_grid.sh logging/runs/r11_union/zh "$SPEC_B" MHC_zh "$ZH_SEEDS_B" R11UN

# ---------------- HateMM ----------------
echo "=== HateMM grid start $(date -Is) seeds=$HM_SEEDS ==="
bash idea-stage/r11_union/run_union_grid.sh logging/runs/r11_union/hm "$SPEC_HM" HateMM "$HM_SEEDS" R11UN
echo "=== HateMM CATB grid start $(date -Is) seeds=$HM_SEEDS_B ==="
bash idea-stage/r11_union/run_union_grid.sh logging/runs/r11_union/hm "$SPEC_B" HateMM "$HM_SEEDS_B" R11UN

# ---------------- read-out ----------------
echo "=== analyse $(date -Is) ==="
python idea-stage/r11_union/analyze_union.py \
  --logdir logging/runs/r11_union/zh/logs --dataset MHC_zh \
  --seeds "$ZH_SEEDS" --seeds_b "$ZH_SEEDS_B" \
  --out idea-stage/r11_union/zh_union.json || exit 1
python idea-stage/r11_union/analyze_union.py \
  --logdir logging/runs/r11_union/hm/logs --dataset HateMM \
  --seeds "$HM_SEEDS" --seeds_b "$HM_SEEDS_B" \
  --out idea-stage/r11_union/hm_union.json || exit 1

echo "=== frozen verdict $(date -Is) ==="
python idea-stage/r11_union/verdict.py \
  --zh idea-stage/r11_union/zh_union.json \
  --hm idea-stage/r11_union/hm_union.json \
  --out idea-stage/r11_union/verdict.json

echo "=== R11-UNION ALLDONE $(date -Is) ==="
