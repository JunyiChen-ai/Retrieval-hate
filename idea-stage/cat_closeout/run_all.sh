#!/usr/bin/env bash
# CAT close-out -- the single frozen run.  idea-stage/CAT_CLOSEOUT_FREEZE.md section 7.
#
# Leg A  MHC-ZH end-to-end re-extraction (reversed item order) + 20-seed A0/CAT/RAND
# Leg B  MHC-EN transport extraction + 30-seed A0/CAT/RAND
# Leg C  5x5 repeated stratified CV on train+dev, both datasets, A0/CAT
# Leg D  per-item read-out audit off the already-dumped R10-COMBO logits (CPU)
#
# A leg that halts on a belt stops that leg only; later independent legs still run.
set -uo pipefail
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1

CC=idea-stage/cat_closeout
GRID=idea-stage/r10_combo/run_combo_grid.sh     # run_grid.sh + --dump_head_scores
ZH_BASE="Qwen2.5-VL-7B-Instruct-LoRA_HF"
EN_BASE="Qwen2.5-VL-7B-Instruct-LoRA_HF"
A_SEEDS=$(seq -s, 1300 1319)
B_SEEDS=$(seq -s, 1400 1429)

echo "=== CAT CLOSE-OUT start $(date -Is) freeze=ae286c9 ==="
mkdir -p $CC/out

# ---------------------------------------------------------------- Leg A
echo "=== LEG A extraction (MHC_zh, reverse order) $(date -Is) ==="
python $CC/extract_cc.py --dataset MHC_zh --lora_dir logging/lora/MHC_zh \
  --out_model_base_tag "$ZH_BASE" --splits train,val,test --order reverse
A_EXTRACT=$?
echo "=== LEG A extraction rc=$A_EXTRACT $(date -Is) ==="

if [ $A_EXTRACT -eq 0 ]; then
  python $CC/build_cc_arms.py --dataset MHC_zh --base "$ZH_BASE" --prefix CCA --compare_tp
  if [ $? -eq 0 ]; then
    echo "=== LEG A grid $(date -Is) seeds=$A_SEEDS ==="
    bash $GRID logging/runs/cat_closeout/legA "A0:CCA-A0,CAT:CCA-CAT,RAND:CCA-RAND" \
      MHC_zh "$A_SEEDS" CCA
    python $CC/analyze_cc.py --logdir logging/runs/cat_closeout/legA/logs \
      --dataset MHC_zh --arms A0,CAT,RAND --seeds "$A_SEEDS" \
      --contrasts "CAT-A0,CAT-RAND,RAND-A0" --label "legA_MHC_zh_reextraction" \
      --out $CC/out/legA_MHC_zh.json
  else
    echo "=== LEG A HALTED at build (belt B1) ==="
  fi
fi

# ---------------------------------------------------------------- Leg B
echo "=== LEG B restore MHC-EN LoRA from B2 $(date -Is) ==="
mkdir -p logging/lora/MHC
/home/jehc223/.local/bin/rclone copy b2:junyi-data/RGCL_video/logs/lora/MHC \
  logging/lora/MHC --exclude "checkpoint-*/**" --transfers 4 --stats 30s
ls -la logging/lora/MHC

echo "=== LEG B extraction (MHC_EN, forward order) $(date -Is) ==="
python $CC/extract_cc.py --dataset MHC --lora_dir logging/lora/MHC \
  --out_model_base_tag "$EN_BASE" --splits train,val,test --order forward
B_EXTRACT=$?
echo "=== LEG B extraction rc=$B_EXTRACT $(date -Is) ==="

if [ $B_EXTRACT -eq 0 ]; then
  python $CC/build_cc_arms.py --dataset MHC --base "$EN_BASE" --prefix CCB
  if [ $? -eq 0 ]; then
    echo "=== LEG B grid $(date -Is) seeds=$B_SEEDS ==="
    bash $GRID logging/runs/cat_closeout/legB "A0:CCB-A0,CAT:CCB-CAT,RAND:CCB-RAND" \
      MHC "$B_SEEDS" CCB
    python $CC/analyze_cc.py --logdir logging/runs/cat_closeout/legB/logs \
      --dataset MHC --arms A0,CAT,RAND --seeds "$B_SEEDS" \
      --contrasts "CAT-A0,CAT-RAND,RAND-A0" --label "legB_MHC_EN_transport" \
      --out $CC/out/legB_MHC_EN.json
  else
    echo "=== LEG B HALTED at build (belt B1) ==="
  fi
fi

# ---------------------------------------------------------------- Leg C
for DS in MHC_zh HateMM; do
  echo "=== LEG C build CV cells $DS $(date -Is) ==="
  python $CC/build_cv.py --dataset $DS || continue
  CELLSEEDS=""
  for R in 0 1 2 3 4; do
    for F in 0 1 2 3 4; do
      S=$((1500 + 5*R + F))
      CELLSEEDS="${CELLSEEDS}${CELLSEEDS:+,}${S}"
      bash $GRID logging/runs/cat_closeout/legC_${DS} \
        "A0:CCCVr${R}f${F}-A0,CAT:CCCVr${R}f${F}-CAT" $DS "$S" CCCV
    done
    echo "PROGRESS legC $DS repeat $R done $(date -Is)"
  done
  python $CC/analyze_cc.py --logdir logging/runs/cat_closeout/legC_${DS}/logs \
    --dataset $DS --arms A0,CAT --seeds "$CELLSEEDS" --contrasts "CAT-A0" \
    --label "legC_cv_${DS}" --out $CC/out/legC_${DS}.json
  echo "=== LEG C $DS cleanup cell caches $(date -Is) ==="
  rm -f data/CLIP_Embedding/${DS}/*_CCCVr[0-4]f[0-4]-*.pt
done

# ---------------------------------------------------------------- verdict
echo "=== FROZEN VERDICT $(date -Is) ==="
python $CC/verdict.py --legA $CC/out/legA_MHC_zh.json --legB $CC/out/legB_MHC_EN.json \
  --legC_zh $CC/out/legC_MHC_zh.json --legC_hm $CC/out/legC_HateMM.json \
  --out $CC/out/verdict.json

# ---------------------------------------------------------------- Leg D
echo "=== LEG D per-item audit (CPU) $(date -Is) ==="
python $CC/audit_items.py --logdir logging/runs/r10_combo/zh/logs --dataset MHC_zh \
  --seeds "$(seq -s, 600 629)" --gt data/gt/MHC_zh/test.jsonl \
  --ocr data/OCR/MHC_zh_test/ocr_video.jsonl \
  --cc data/CLIP_Embedding/MHC_zh/test_seen_${ZH_BASE}-cc.pt \
  --out $CC/out/legD_MHC_zh.json
python $CC/audit_items.py --logdir logging/runs/r10_combo/hm/logs --dataset HateMM \
  --seeds "$(seq -s, 600 614)" --gt data/gt/HateMM/test.jsonl \
  --ocr data/OCR/HateMM/ocr_video_test.jsonl \
  --out $CC/out/legD_HateMM.json

echo "=== CAT CLOSE-OUT ALLDONE $(date -Is) ==="
