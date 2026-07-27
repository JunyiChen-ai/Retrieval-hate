cd /data/jehc223/RGCL || exit 1
source ~/.bashrc 2>/dev/null
conda activate HateVideo 2>/dev/null
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
LOG=scripts/analysis/.mechnov_diag2.log
: > "$LOG"
for d in zh en; do
  for try in 1 2 3; do
    python scripts/analysis/mechnov_pairverify_diag.py --datasets "$d" \
      --out scripts/analysis/mechnov_pairverify_diag_OUT.json >> "$LOG" 2>&1
    echo "[diag] $d try $try exit $?" >> "$LOG"
    python - <<PY >> "$LOG" 2>&1 && break
import json,sys
d=json.load(open("scripts/analysis/mechnov_pairverify_diag_OUT.json"))
sys.exit(0 if "$d" in d.get("datasets",{}) else 1)
PY
  done
done
echo "[diag] ALL DIAG DONE" >> "$LOG"
