#!/usr/bin/env bash
# REPRO campaign Wave 2 — T3AL, the whole GPU-side pipeline in one queued job.
#
# Stages, in order, all resume-safe (each skips work whose output exists):
#   1. CoCa ViT-L/14 features + captions on the 4 fps grid, val+test, four datasets
#   2. val sweep over the four frozen knob presets (variant `main` only)
#   3. preset selection by the rule frozen in idea-stage/repro_t3al/RUN_RECORD.md
#   4. the single test call, three seeds, then the shared evaluator
#
# Launched through gpu_queue.sh by t3al_launch.sh; never run directly on a busy card.
set -u
cd /home/jehc223/Retrieval-hate
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DS="HateMM,MHC,MHC_zh,HateClipSeg"
SEEDS="20250819 20250820 20250821"
OUT=idea-stage/repro_t3al
mkdir -p $OUT/sweep $OUT/eval

echo "[stage1] features+captions $(date -Is)"
python scripts/repro_campaign/extract_coca_4fps.py \
  --datasets "$DS" --splits val,test --fp16 --batch 64 || exit 1

echo "[stage2] val sweep $(date -Is)"
for P in A_thumos B_thumos_rescaled C_thumos_15steps D_anet; do
  python scripts/repro_campaign/run_t3al.py --datasets "$DS" --splits val \
    --preset "$P" --seed 20250819 --variants main \
    --out-dir "$OUT/sweep/$P" || exit 1
  python scripts/repro_campaign/eval_frame.py --method curves \
    --curve-dir "$OUT/sweep/$P" --variants main --method-name "T3AL" \
    --wave 2 --supervision label-free --split val \
    --out "$OUT/eval/val_$P.json" || exit 1
done

echo "[stage3] select preset $(date -Is)"
python scripts/repro_campaign/t3al_select.py || exit 1
PRESET=$(python -c "import json;print(json.load(open('$OUT/preset_chosen.json'))['preset'])")
echo "[stage3] chosen preset = $PRESET"

echo "[stage4] test, three seeds $(date -Is)"
for S in $SEEDS; do
  python scripts/repro_campaign/run_t3al.py --datasets "$DS" --splits test \
    --preset "$PRESET" --seed "$S" --hcs-classes \
    --variants main,mainq_sim,c1_hateful \
    --out-dir "$OUT/curves_s$S" || exit 1
done
ln -sfn curves_s20250819 "$OUT/curves"

for S in $SEEDS; do
  python scripts/repro_campaign/eval_frame.py --method curves \
    --curve-dir "$OUT/curves_s$S" \
    --variants main,mainq_sim,c0_normal,c1_hateful,c2_insulting,c3_sexual,c4_violence,c5_harm \
    --method-name "T3AL" --wave 2 --supervision label-free --split test \
    --out "$OUT/eval/test_s$S.json" || exit 1
done
python scripts/repro_campaign/t3al_aggregate.py
echo "[all-done] $(date -Is)"
