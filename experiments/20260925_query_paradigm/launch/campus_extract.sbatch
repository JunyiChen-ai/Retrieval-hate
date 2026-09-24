#!/bin/bash
#SBATCH --job-name=qtl_extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/20260925_query_paradigm/extract/slurm_%j.out
# usage (campus, repo root): sbatch experiments/20260925_query_paradigm/launch/campus_extract.sbatch <corpus> <splits> [shard i/n]
# README section 2.1: query-tree answers with Qwen2.5-VL-7B (vLLM, env ~/miniconda3/envs/vlm)
cd "$HOME/Retrieval-hate"
export HF_HOME="$HOME/.cache/huggingface"
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
"$HOME/miniconda3/envs/vlm/bin/python" experiments/20260925_query_paradigm/extract_tree_answers.py \
  --corpus "$1" --splits "$2" --shard "${3:-0/1}" --gpu-mem 0.85
