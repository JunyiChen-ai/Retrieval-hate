#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
run=runs/20260905_interventional_evidence/ablations/hateclipseg/seed234
config=runs/20260905_interventional_evidence/hateclipseg/seed234/trial18/hparams.json
mkdir -p "$run"
echo "host=$(hostname) date=$(date -Is) locked_trial=18"
echo "$$" > "$run/run.pid"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python_bin="$HOME/miniconda3/envs/HateVideo/bin/python"
arms=(raw_verdict ordinary_attention additive_fusion full_input_only four_logits no_interaction dempster_fusion no_block)
pids=()
failed=0
for arm in "${arms[@]}"; do
  # Never overwrite an old diagnostic run or silently repeat it.
  test ! -e "$run/$arm"
  mkdir "$run/$arm"
  "$python_bin" -u experiments/20260905_interventional_evidence/train.py --corpus hateclipseg --seed 234 --ablation "$arm" --config "$config" --out-dir "$run/$arm" --num-workers 2 > "$run/$arm/stdout.log" 2>&1 &
  pids+=("$!")
  if [[ ${#pids[@]} -eq 3 ]]; then
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ $failed -eq 0 ]] || exit 1
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed -eq 0 ]] || exit 1
"$python_bin" -c 'import json,pathlib; pathlib.Path("runs/20260905_interventional_evidence/ablations/hateclipseg/seed234/completion.json").write_text(json.dumps({"state":"ABLATIONS_FINISHED","expected_arms":8}))'
