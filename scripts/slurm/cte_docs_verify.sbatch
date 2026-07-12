#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --job-name=cte_docs_verify
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('/data/jehc223/RGCL')
state = json.loads((root / 'TARGET_STATE.json').read_text(encoding='utf-8'))
decision = json.loads((root / 'artifacts/cte/v1/C0_DECISION.json').read_text(encoding='utf-8'))
freeze = json.loads((root / 'artifacts/cte/v1/CONFIG_FREEZE.json').read_text(encoding='utf-8'))
raw = dict(decision)
stored = raw.pop('payload_sha256')
canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False)
assert hashlib.sha256(canonical.encode('utf-8')).hexdigest() == stored
assert decision['C0_DECISION'] == 'STOP'
assert decision['C1_unlocked'] is False
assert state['gate_3_c0_execution']['decision_payload_sha256'] == stored
assert state['gate_3_c0_execution']['C1_unlocked'] is False
assert state['gate_3_c0_execution']['segment_gold_exists'] is False
assert state['gate_3_c0_execution']['segment_gold_used'] is False
assert state['gate_3_c0_execution']['mllm_call_count'] == 0
assert not (root / 'artifacts/cte/v1/c1').exists()
for item in freeze['input_files']:
    assert hashlib.sha256((root / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
for path in (
    'refine-logs/cte/EXPERIMENT_PLAN.md',
    'refine-logs/cte/EXPERIMENT_TRACKER.md',
    'refine-logs/cte/EXPERIMENT_RESULTS.md',
    'research-wiki/experiments/exp-cte-c0.md',
):
    assert (root / path).is_file() and (root / path).stat().st_size > 0
print(json.dumps({
    'status': 'DOCS_VERIFY_GO',
    'C0_DECISION': decision['C0_DECISION'],
    'C1_unlocked': decision['C1_unlocked'],
    'mllm_call_count': decision['mllm_call_count'],
    'segment_gold_exists': decision['segment_gold_exists'],
    'c1_artifact_exists': (root / 'artifacts/cte/v1/c1').exists(),
}, sort_keys=True))
PY
