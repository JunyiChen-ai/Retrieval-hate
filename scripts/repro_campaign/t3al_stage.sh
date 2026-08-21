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
[ -n "$PRESET" ] || { echo "[stage3] no preset chosen"; exit 1; }
echo "[stage3] chosen preset = $PRESET"

# freeze §6 run metadata
python - "$PRESET" > "$OUT/run_meta.json" <<'PY'
import json, subprocess, sys, time, torch
print(json.dumps(dict(
    method="T3AL", wave=2, supervision="label-free",
    repo="benedettaliberatori/T3AL@dfbbbc1c",
    backbone="open_clip coca_ViT-L-14 / mscoco_finetuned_laion2B-s13B-b90k",
    preset=sys.argv[1], seeds=[20250819, 20250820, 20250821],
    feature_rate_fps=4.0, caption_rate_fps=1.0,
    git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip(),
    torch=torch.__version__,
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    started=time.strftime("%Y-%m-%dT%H:%M:%S%z")), indent=1))
PY

echo "[stage4] test, three seeds $(date -Is)"
# Each seed is scored as soon as it finishes, so an interrupted run still has a
# complete, aggregatable result for every seed that did complete.
for S in $SEEDS; do
  python scripts/repro_campaign/run_t3al.py --datasets "$DS" --splits test \
    --preset "$PRESET" --seed "$S" --hcs-classes \
    --variants main,mainq_sim,c1_hateful \
    --out-dir "$OUT/curves_s$S" || exit 1
  ln -sfn curves_s20250819 "$OUT/curves"
  python scripts/repro_campaign/eval_frame.py --method curves \
    --curve-dir "$OUT/curves_s$S" \
    --variants main,mainq_sim,c0_normal,c1_hateful,c2_insulting,c3_sexual,c4_violence,c5_harm \
    --method-name "T3AL" --wave 2 --supervision label-free --split test \
    --out "$OUT/eval/test_s$S.json" || exit 1
  python scripts/repro_campaign/t3al_aggregate.py || true
done
echo "[all-done] $(date -Is)"
