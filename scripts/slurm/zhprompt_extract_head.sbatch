#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=zhprompt
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# ZHPROMPT probe (prereg refine-logs/ZHPROMPT_PREREG.md) — ONE job, 8 CPU / 64 G /
# 1 A100 throughout (an 8-CPU job => the "never two concurrent 16-CPU jobs" wedge rule
# is trivially satisfied; and this is ONE submission, honouring the prereg budget).
#
# Stage A: re-extract the ZH dual-stream (img+text) with faithful CHINESE instruction
#          prompts (recon §3), on BOTH encoders, into DISTINCT -zhp caches:
#            Arm-F (frozen)  -> Qwen2.5-VL-7B-Instruct_HF-zhp        (paired vs 13115)
#            Arm-L (LoRA)    -> Qwen2.5-VL-7B-Instruct-LoRA_HF-zhp   (paired vs 13150; PRIMARY)
#          The Chinese strings are HARDCODED here (frame16/readout no-submit-time-typo
#          discipline); the extractor args DEFAULT to the byte-identical English
#          constants, so default == identity == the KS-parity guard.
# Stage B: 6 head runs (2 arms x 3 seeds) on the -zhp caches. run_one is BYTE-IDENTICAL
#          to enc3seed_zh_b3.sbatch (the runner that produced BOTH floors 13115/13150);
#          the ONLY per-arm variables are --model and the derived --exp_comment "_${MODEL}"
#          (=> distinct output dirs under the single group RAC_video_zhp; no clobber).
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# ----------------------------------------------------------------------------
# Frozen Chinese prompt strings (verbatim from ZHPROMPT_FORENSIC_RECON.md §3;
# faithful TRANSLATION only, no wording edits). Single-quoted => no shell
# interpolation. These are the ONLY manipulated variable vs the banked floors.
# ----------------------------------------------------------------------------
IMG_ZH='描述这段视频中的人物、符号、手势以及画面上的文字。'
TEXT_ZH='你正在分析一段可能包含仇恨或冒犯性内容的短视频。请结合视频画面以及所提供的标题和文字记录,概括其中的攻击对象、符号、语气,以及所传达的任何有害意图。'
TITLE_ZH='标题:'
TRANSCRIPT_ZH='文字记录:'
NONE_ZH='(无)'

DATASET=MHC_zh
LORA_DIR=logging/lora/MHC_zh
FROZEN_TAG=Qwen2.5-VL-7B-Instruct_HF-zhp
LORA_TAG=Qwen2.5-VL-7B-Instruct-LoRA_HF-zhp

echo "########## ZHPROMPT STAGE A: Chinese-prompt re-extraction ($DATASET) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- Arm-F: FROZEN encoder, Chinese prompts -> -zhp cache (never touches …_HF.pt).
echo "---------- Arm-F (frozen) -> ${FROZEN_TAG} ----------"
python src/utils/generate_VideoMLLM_embedding_HF.py \
    --dataset "$DATASET" \
    --out_model_tag "$FROZEN_TAG" \
    --num_frames 8 \
    --img_instruction "$IMG_ZH" \
    --text_instruction "$TEXT_ZH" \
    --title_label "$TITLE_ZH" \
    --transcript_label "$TRANSCRIPT_ZH" \
    --none_placeholder "$NONE_ZH" \
    --device cuda

# --- Arm-L: LoRA-merged encoder (deployed ZH adapter), Chinese prompts -> -zhp cache.
echo "---------- Arm-L (LoRA) -> ${LORA_TAG} ----------"
if [ ! -d "${LORA_DIR}" ]; then
  echo "[zhprompt] ERROR: LORA_DIR '${LORA_DIR}' is not a directory." >&2
  exit 2
fi
python src/utils/generate_VideoMLLM_embedding_lora_HF.py \
    --dataset "$DATASET" \
    --lora_dir "$LORA_DIR" \
    --out_model_tag "$LORA_TAG" \
    --num_frames 8 \
    --img_instruction "$IMG_ZH" \
    --text_instruction "$TEXT_ZH" \
    --title_label "$TITLE_ZH" \
    --transcript_label "$TRANSCRIPT_ZH" \
    --none_placeholder "$NONE_ZH" \
    --device cuda

# --- Shape sanity (DEV): abort BEFORE spending the 6 head runs if any -zhp cache is
#     malformed (protects the budgeted test reads). $0 GPU.
echo "---------- Stage A shape sanity ----------"
python - "$DATASET" "$FROZEN_TAG" "$LORA_TAG" <<'PY'
import sys, os, torch
ds, ftag, ltag = sys.argv[1], sys.argv[2], sys.argv[3]
root = os.path.join("data/CLIP_Embedding", ds)
bad = 0
for tag in (ftag, ltag):
    for split in ("train", "dev_seen", "test_seen"):
        p = os.path.join(root, f"{split}_{tag}.pt")
        if not os.path.exists(p):
            print(f"[MISSING] {p}"); bad += 1; continue
        o = torch.load(p, map_location="cpu")
        n = len(o["labels"]); img = o["img_feats"]; txt = o["text_feats"]
        ids = [x for sub in o["ids"] for x in sub]
        ok = (img.shape == (n, 3584) and txt.shape == (n, 3584) and len(ids) == n and n > 0)
        print(f"[{'OK' if ok else 'BAD'}] {p}: N={n} img={tuple(img.shape)} text={tuple(txt.shape)} ids={len(ids)}")
        if not ok:
            bad += 1
if bad:
    print(f"SHAPE_SANITY_FAIL ({bad} bad) -> aborting before head runs"); sys.exit(3)
print("SHAPE_SANITY_OK")
PY

# push derived -zhp .pt caches to B2 (videos NEVER leave the node; keep local for heads).
bash scripts/b2_push.sh "data/CLIP_Embedding/$DATASET" "embeddings/$DATASET" || true

echo ""
echo "########## ZHPROMPT STAGE B: 6 head runs (2 arms x 3 seeds) ##########"

GROUP_NAME=RAC_video_zhp
WARMUP=5

# config list: "DATASET MODEL SEED"  (2 arms x 3 seeds; distinct MODEL => distinct exp dir)
CONFIGS=(
  "MHC_zh $FROZEN_TAG 0"
  "MHC_zh $FROZEN_TAG 1"
  "MHC_zh $FROZEN_TAG 2"
  "MHC_zh $LORA_TAG 0"
  "MHC_zh $LORA_TAG 1"
  "MHC_zh $LORA_TAG 2"
)

run_one () {
  local DATASET="$1" MODEL="$2" SEED="$3"
  local RUNLOG="slurm/logs/enc3s_${DATASET}_${MODEL}_seed${SEED}_${SLURM_JOB_ID:-$$}.trainlog"
  echo ""
  echo "########## RUN: DATASET=$DATASET MODEL=$MODEL SEED=$SEED -> $RUNLOG ##########"
  python ./src/run_rac.py --batch_size 64 \
      --lr 0.0001 --epochs 30 --topk 20 --dataset "$DATASET" \
      --model "$MODEL" \
      --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
      --fusion_mode "align" \
      --hard_negatives_loss True --no_hard_negatives 1 \
      --final_eval False --seed "$SEED" --group_name "$GROUP_NAME" \
      --metric "cos" --loss "triplet" --batch_norm False \
      --hybrid_loss True --warmup "$WARMUP" \
      --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
      --lambda_seg 0 --seg_mode full --num_subclips 4 \
      --em_rounds 2 --consensus_topk 10 --consensus_margin 0.2 \
      --exp_comment "_${MODEL}" \
      --Faiss_GPU False --force False 2>&1 | tee "$RUNLOG"

  echo "-------- VAL-SELECTED (warmup>=5) + FINAL-EPOCH readout: $DATASET / $MODEL / seed$SEED --------"
  WARMUP="$WARMUP" python - "$RUNLOG" <<'PY'
import sys, re, os
log = open(sys.argv[1]).read()
val, test = {}, {}
vre = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
tre = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
for m in vre.finditer(log):
    e=int(m.group(1)); val[e]=tuple(float(x) for x in m.groups()[1:])
for m in tre.finditer(log):
    e=int(m.group(1)); test[e]=tuple(float(x) for x in m.groups()[1:])
if not val or not test:
    print("NO_PARSE"); sys.exit(0)
W=int(os.environ.get("WARMUP",5))
warm=[e for e in val if e>=W] or list(val.keys())
best=max(warm, key=lambda e:(val[e][3], val[e][4]))
a=test.get(best); fe=max(test); b=test[fe]
print(f"VALSEL epoch {best}: TEST F1 {a[0]:.4f} acc {a[3]:.4f} roc {a[4]:.4f}")
print(f"FINAL  epoch {fe}: TEST F1 {b[0]:.4f} acc {b[3]:.4f} roc {b[4]:.4f}")
print(f"RESULT_ROW\t{sys.argv[1]}\tvalsel\t{best}\t{a[0]:.4f}\t{a[3]:.4f}\t{a[4]:.4f}\tfinal\t{fe}\t{b[0]:.4f}\t{b[3]:.4f}\t{b[4]:.4f}")
PY
}

for cfg in "${CONFIGS[@]}"; do
  read -r D M S <<< "$cfg"
  run_one "$D" "$M" "$S"
done

echo ""
echo "======== zhprompt ALL DONE ($SLURM_JOB_ID) ========"
bash scripts/b2_push.sh logging "logs/zhprompt" || true
