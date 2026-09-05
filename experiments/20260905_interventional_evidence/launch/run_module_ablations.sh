#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
corpus=$1
seed=$2
run="runs/20260905_interventional_evidence/ablations/${corpus}/seed${seed}"
study="runs/20260905_interventional_evidence/${corpus}/seed${seed}"
python_bin="$HOME/miniconda3/envs/HateVideo/bin/python"
trial=$("$python_bin" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert len(s["trials"])==s["n_trials"] and all(t["state"] in ["COMPLETE","PRUNED"] for t in s["trials"]); print(s["best"]["number"])' "$study/study_summary.json")
config="$study/trial${trial}/hparams.json"
mkdir -p "$run"
echo "host=$(hostname) date=$(date -Is) locked_trial=$trial"
echo "$$" > "$run/run.pid"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
arms=(raw_verdict ordinary_attention additive_fusion full_input_only four_logits no_interaction dempster_fusion no_block)
pids=()
failed=0
for arm in "${arms[@]}"; do
  test ! -e "$run/$arm"
  mkdir "$run/$arm"
  "$python_bin" -u experiments/20260905_interventional_evidence/train.py --corpus "$corpus" --seed "$seed" --ablation "$arm" --config "$config" --out-dir "$run/$arm" --num-workers 2 > "$run/$arm/stdout.log" 2>&1 &
  pids+=("$!")
  if [[ ${#pids[@]} -eq 3 ]]; then
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ $failed -eq 0 ]] || exit 1
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed -eq 0 ]] || exit 1
"$python_bin" -c 'import json,pathlib,sys; (pathlib.Path(sys.argv[1])/"completion.json").write_text(json.dumps({"state":"ABLATIONS_FINISHED","expected_arms":8}))' "$run"
