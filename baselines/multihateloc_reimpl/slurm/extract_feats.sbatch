#!/bin/bash
#SBATCH --job-name=mhl_feats
#SBATCH --partition=slurmpartition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=/data/jehc223/RGCL/baselines/multihateloc_reimpl/slurm/logs/feats_%j.out
#SBATCH --error=/data/jehc223/RGCL/baselines/multihateloc_reimpl/slurm/logs/feats_%j.err

set -e
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
cd /data/jehc223/RGCL/baselines/multihateloc_reimpl/src

echo "host=$(hostname) start=$(date)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv

# 2 shards, one per GPU, run concurrently
CUDA_VISIBLE_DEVICES=0 python3 extract_features.py --shard 0 --nshards 2 \
    > ../slurm/logs/feats_shard0.log 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python3 extract_features.py --shard 1 --nshards 2 \
    > ../slurm/logs/feats_shard1.log 2>&1 &
PID1=$!

wait $PID0; R0=$?
wait $PID1; R1=$?
echo "shard0 rc=$R0 shard1 rc=$R1 end=$(date)"

# count outputs
N=$(ls /data/jehc223/RGCL/data/multihateloc_feats/HateMM/*.npz 2>/dev/null | wc -l)
echo "total npz files: $N / 1083"
