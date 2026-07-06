#!/usr/bin/env bash
# Pull the 9 val-selected archive-kNN heads (EN seeds 0-3, ZH seeds 0-4) from B2
# for the P2 neighbor-reranking experiment. Checkpoints live on B2 under
# logs/<local logging/ path minus 'logging/'>. ~40 MB each, 360 MB total.
# Deleted locally after the CPU gate/retrieval phase (p2_rerank_eval.py --mode collect).
set -euo pipefail
B2="b2:junyi-data/RGCL_video/logs"
ROOT="/data/jehc223/RGCL"

declare -A EN=( [0]="best_model_24_0.7875.pt" [1]="best_model_29_0.7875.pt" \
                [2]="best_model_21_0.8125.pt" [3]="best_model_27_0.7875.pt" )
declare -A ZH=( [0]="best_model_18_0.8717948717948718.pt" \
                [1]="best_model_23_0.8846153846153846.pt" \
                [2]="best_model_14_0.8846153846153846.pt" \
                [3]="best_model_17_0.8717948717948718.pt" \
                [4]="best_model_12_0.8717948717948718.pt" )

pull() {  # ds model seed file
    local ds="$1" model="$2" seed="$3" file="$4"
    local group; if [[ "$seed" == "0" ]]; then group="RAC_video_archive"; else group="RAC_video_archive_seeds"; fi
    local run="RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20__PseudoGold_positive_1_hard_negative_1_seed${seed}_hybrid_loss_${model}_arc-knn-a0.25"
    local rel="Retrieval/${ds}/${group}/${run}/ckpt/${file}"
    local dst="${ROOT}/logging/${rel}"
    if [[ -f "$dst" ]]; then echo "[have] $dst"; return; fi
    mkdir -p "$(dirname "$dst")"
    echo "[pull] ${ds} seed${seed} -> ${file}"
    rclone copyto "${B2}/${rel}" "$dst" --transfers 4
}

for s in 0 1 2 3;     do pull MHC    "Qwen2.5-VL-7B-Instruct_HF"      "$s" "${EN[$s]}"; done
for s in 0 1 2 3 4;   do pull MHC_zh "Qwen2.5-VL-7B-Instruct-LoRA_HF" "$s" "${ZH[$s]}"; done
echo "[p2_pull_ckpts] all done"
