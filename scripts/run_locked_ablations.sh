#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
experiment=$1
corpus=$2
seed=$3
shift 3
arms=("$@")
[[ ${#arms[@]} -gt 0 ]]
[[ "$experiment" =~ ^[0-9]{8}_[a-z0-9_]+$ ]]
[[ "$corpus" == hatemm || "$corpus" == hateclipseg ]]
[[ "$seed" =~ ^[0-9]+$ ]]
run="runs/${experiment}/ablations/${corpus}/seed${seed}"
study="runs/${experiment}/${corpus}/seed${seed}"
python_bin="$HOME/miniconda3/envs/HateVideo/bin/python"
trial=$("$python_bin" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert len(s["trials"])==s["n_trials"] and all(t["state"] in ["COMPLETE","PRUNED"] for t in s["trials"]); assert s["best"] is not None; print(s["best"]["number"])' "$study/study_summary.json")
config="$study/trial${trial}/hparams.json"
trainer="experiments/${experiment}/train.py"
if [[ ! -f "$trainer" ]]; then
  trainer="archive/experiments/${experiment}/train.py"
fi
test -f "$trainer"
mkdir -p "$run"
meta="$run"
if [[ -n "${ABLATION_CHAIN_NAME:-}" ]]; then
  [[ "$ABLATION_CHAIN_NAME" =~ ^[a-z0-9_]+$ ]]
  meta="$run/$ABLATION_CHAIN_NAME"
  mkdir -p "$meta"
fi
exec 9> "$run/launch.lock"
flock -n 9
echo "host=$(hostname) date=$(date -Is) locked_trial=$trial"
echo "$$" > "$meta/run.pid"
# Validate all destinations before starting any arm, never partly rerun a chain.
for arm in "${arms[@]}"; do
  [[ "$arm" =~ ^[a-z0-9_]+$ ]]
  test ! -e "$run/$arm"
done
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
pids=()
failed=0
for arm in "${arms[@]}"; do
  mkdir "$run/$arm"
  "$python_bin" -u "$trainer" --corpus "$corpus" --seed "$seed" --ablation "$arm" --config "$config" --out-dir "$run/$arm" --num-workers 2 > "$run/$arm/stdout.log" 2>&1 &
  pids+=("$!")
  if [[ ${#pids[@]} -eq 3 ]]; then
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ $failed -eq 0 ]] || exit 1
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed -eq 0 ]] || exit 1
"$python_bin" -c 'import json,pathlib,sys; (pathlib.Path(sys.argv[1])/"completion.json").write_text(json.dumps({"state":"ABLATIONS_FINISHED","expected_arms":len(sys.argv)-2,"arms":sys.argv[2:]}))' "$meta" "${arms[@]}"
