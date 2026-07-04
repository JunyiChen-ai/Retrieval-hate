#!/bin/bash
#SBATCH --job-name=mhl_train
#SBATCH --partition=slurmpartition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/data/jehc223/RGCL/baselines/multihateloc_reimpl/slurm/logs/train_%j.out
#SBATCH --error=/data/jehc223/RGCL/baselines/multihateloc_reimpl/slurm/logs/train_%j.err

# Usage: sbatch train_eval.sbatch <MODALITIES> <SEED>
#   e.g. sbatch train_eval.sbatch vat 0
set -e
MODS=${1:-vat}
SEED=${2:-0}
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
cd /data/jehc223/RGCL/baselines/multihateloc_reimpl/src

RUN=/data/jehc223/RGCL/baselines/multihateloc_reimpl/results/${MODS}_seed${SEED}
echo "host=$(hostname) mods=$MODS seed=$SEED start=$(date)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv | head -3

python3 train.py --modalities "$MODS" --seed "$SEED" --out "$RUN" \
    --epochs 100 --batch 32 --lr 1e-4 --d_model 256 --k_div 3 --max_t 256

# evaluate best checkpoint at full length (localization needs full videos)
python3 eval.py --ckpt "$RUN/best.pt" --out "$RUN/eval_test" --split test --max_t 0
echo "end=$(date)"
