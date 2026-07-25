#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=moka_eh
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# MOKA cell — stage 2 of 2 (refine-logs/MOKA_PREREG.md). ONE job, 8 CPU (so the family NEVER holds
# two 16-CPU jobs; this is submitted only AFTER lora_sft_moka.sbatch reports COMPLETED).
#
#   Stage A0  KS-MOKA-0b merge-drift probe ($0 test-touch — features carry no labels):
#             re-extract the ALREADY-BANKED generic ZH adapter through the UNMERGED path
#             (--no_merge) and report mean per-item cosine vs the banked merged cache.
#   Stage A1  MokA extraction (--moka: unmerged + routed + explicit lora_A_v load), 3 splits.
#   Stage S   shape sanity (aborts BEFORE the 3 budgeted test reads on any malformed cache).
#   Stage B   3 RGCL align-fusion head + kNN runs, paired within head-seed vs banked floor 13150.
#             The run_one() block below is BYTE-IDENTICAL to scripts/slurm/enc3seed_zh_b3.sbatch
#             (the runner that produced floor 13150).
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export MOKA_ROUTE_IMPL=dense
export MOKA_STRICT=1

DS=MHC_zh
BANKED_ADAPTER=logging/lora/MHC_zh
MOKA_ADAPTER=logging/lora/MHC_zh_moka
UM_TAG=Qwen2.5-VL-7B-Instruct-LoRA_HF-um
MOKA_TAG=Qwen2.5-VL-7B-Instruct-LoRA-moka_HF

for d in "$BANKED_ADAPTER" "$MOKA_ADAPTER"; do
  if [ ! -f "$d/adapter_model.safetensors" ]; then
    echo "[moka_eh] ABORT: missing adapter $d/adapter_model.safetensors" >&2; exit 2
  fi
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- Stage A0: KS-MOKA-0b — unmerged re-extraction of the BANKED generic adapter -------------
echo "######### Stage A0: KS-MOKA-0b unmerged re-extraction (banked generic ZH adapter) #########"
python src/utils/generate_VideoMLLM_embedding_lora_HF.py \
    --dataset "$DS" \
    --lora_dir "$BANKED_ADAPTER" \
    --out_model_tag "$UM_TAG" \
    --no_merge \
    --num_frames 8 \
    --device cuda

python - "$DS" "$UM_TAG" <<'PY0'
import sys, torch
ds, um = sys.argv[1], sys.argv[2]
worst = 1.0
for split in ("train", "dev_seen", "test_seen"):
    a = torch.load("data/CLIP_Embedding/{}/{}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt".format(ds, split), map_location="cpu")
    b = torch.load("data/CLIP_Embedding/{}/{}_{}.pt".format(ds, split, um), map_location="cpu")
    assert a["ids"][0] == b["ids"][0], "id order differs in {}".format(split)
    for stream in ("img_feats", "text_feats"):
        cos = torch.nn.functional.cosine_similarity(a[stream].float(), b[stream].float(), dim=1)
        print("[KS-MOKA-0b] {:9s} {:10s} mean cos {:.8f}  min cos {:.8f}".format(
            split, stream, cos.mean().item(), cos.min().item()), flush=True)
        worst = min(worst, cos.mean().item())
print("[KS-MOKA-0b] WORST mean per-item cosine over all 6 (split x stream) = {:.8f}".format(worst))
print("[KS-MOKA-0b] >= 0.9999 ? {}".format(worst >= 0.9999))
PY0

# ---- Stage A1: MokA extraction (unmerged + routed) -------------------------------------------
echo "######### Stage A1: MokA extraction (--moka) #########"
python src/utils/generate_VideoMLLM_embedding_lora_HF.py \
    --dataset "$DS" \
    --lora_dir "$MOKA_ADAPTER" \
    --out_model_tag "$MOKA_TAG" \
    --moka \
    --num_frames 8 \
    --device cuda

# ---- Stage S: shape sanity BEFORE any budgeted test read --------------------------------------
python - "$DS" "$MOKA_TAG" <<'PY1'
import sys, torch
ds, tag = sys.argv[1], sys.argv[2]
for split in ("train", "dev_seen", "test_seen"):
    p = "data/CLIP_Embedding/{}/{}_{}.pt".format(ds, split, tag)
    d = torch.load(p, map_location="cpu")
    n = d["img_feats"].shape[0]
    assert d["img_feats"].shape == d["text_feats"].shape == (n, 3584), (p, d["img_feats"].shape)
    assert len(d["ids"][0]) == n and n > 0
    assert torch.isfinite(d["img_feats"]).all() and torch.isfinite(d["text_feats"]).all(), p
    print("[shape-sanity] OK {} N={} nonzero_img={} nonzero_txt={}".format(
        p, n, int((d["img_feats"].abs().sum(1) > 0).sum()), int((d["text_feats"].abs().sum(1) > 0).sum())))
PY1

# ---- Stage B: 3 head-seeds on the MokA cache (paired vs banked floor 13150) --------------------
LORA=$MOKA_TAG
GROUP_NAME=RAC_video_moka
WARMUP=5

# config list: "DATASET MODEL SEED"
CONFIGS=(
  "MHC_zh $LORA 0"
  "MHC_zh $LORA 1"
  "MHC_zh $LORA 2"
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
echo "======== moka extract+head ALL DONE ($SLURM_JOB_ID) ========"
bash scripts/b2_push.sh "data/CLIP_Embedding/$DS" "embeddings/$DS" || true
bash scripts/b2_push.sh logging "logs/moka" || true
