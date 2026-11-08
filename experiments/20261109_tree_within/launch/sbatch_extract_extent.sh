#!/usr/bin/env bash
# Extent question (README section 8): No / Some / All answers for every queryable node of the query tree,
# Qwen2.5-VL-7B-Instruct, same frames, transcripts and per-dataset definitions as soft_both_p1
# (experiments/20260925_query_paradigm/extract_tree_soft.py --prompt p3). Jobs run one after another; each is
# <corpus>:<shard i/n>:<splits>. Videos already in the output file are skipped, so a cut job resumes when resubmitted.
# Submit on the target host:
#   ssh <host> 'cd ~/Retrieval-hate && mkdir -p runs/20261109_tree_within/extract && sbatch -p local-<node> -J ext_p3 \
#     -o runs/20261109_tree_within/extract/slurm_%j.out experiments/20261109_tree_within/launch/sbatch_extract_extent.sh \
#     hatemm:0/2:test,val hateclipseg:0/2:test,val dehate:0/2:test,val hatemm:0/2:train hateclipseg:0/2:train dehate:0/2:train'
# Output data/vlm_tree/<Corpus>/ext_both_p3.shard<i>of<n>.jsonl; logs runs/20261109_tree_within/extract/.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G
#SBATCH --time=1-00:00:00
set -euo pipefail
cd "$HOME/Retrieval-hate"
export HF_HOME="$HOME/.cache/huggingface"
L=runs/20261109_tree_within/extract
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
echo "${SLURM_JOB_ID:-none}" > "$L/job_$(hostname).id"
for job in "$@"; do
  IFS=: read -r c s sp <<< "$job"
  "$HOME/miniconda3/envs/vlm/bin/python" -u experiments/20260925_query_paradigm/extract_tree_soft.py \
    --corpus "$c" --splits "$sp" --shard "$s" --view both --prompt p3 --gpu-mem 0.85 \
    >> "$L/${c}_${s//\//of}_$(hostname).log" 2>&1
  echo "done $job"
done
echo ALL_DONE
