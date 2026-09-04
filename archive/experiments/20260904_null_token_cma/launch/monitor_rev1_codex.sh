#!/usr/bin/env bash
# Completion notifier only: never starts, stops, or changes experiment code.
set -uo pipefail
cd /home/jehc223/Retrieval-hate || exit 1
MON=runs/20260904_null_token_cma/rev1/monitor_codex
mkdir -p "$MON"
exec 9>"$MON/monitor.lock"
flock -n 9 || exit 0
echo "$$" > "$MON/run.pid"
THREAD=01a06df5-3e92-79b0-be30-820db943e551
CODEX=/home/jehc223/.local/bin/codex
[[ -f "$MON/notification_sent" ]] && exit 0
echo "monitor started $(date -Is); remote chain PID 326400; interval 120s"
while true; do
  STATE=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 uoa-lab1 '
    RUN=/home/jehc223/Retrieval-hate/runs/20260904_null_token_cma/rev1
    CMD=$(ps -p 326400 -o args=)
    if [[ "$CMD" == *experiments/20260904_null_token_cma/launch/run_rev1_hatemm_lab1.sh* ]]; then
      echo RUNNING
    elif [[ -f "$RUN/REV1_hatemm_DONE" ]]; then
      echo CHAIN_FINISHED
    elif pgrep -f "[s]earch.py.*--out-root runs/20260904_null_token_cma/rev1|[t]rain.py.*runs/20260904_null_token_cma/rev1" >/dev/null; then
      echo CHILD_RUNNING
    else
      echo STOPPED_WITHOUT_DONE
    fi
  ')
  RC=$?
  if [[ $RC -ne 0 ]]; then
    echo "$(date -Is) SSH unavailable rc=$RC; retry, not terminal"
    sleep 120
    continue
  fi
  echo "$(date -Is) $STATE"
  case "$STATE" in
    RUNNING|CHILD_RUNNING) sleep 120; continue ;;
    CHAIN_FINISHED|STOPPED_WITHOUT_DONE) ;;
    *) sleep 120; continue ;;
  esac
  MESSAGE="实验监控通知：lab1 候选4空token修订1状态=$STATE，观察时间=$(date -Is)。日志位于 runs/20260904_null_token_cma/rev1/monitor_codex/run.log。请重新核验进程和两语料输出；DONE仅代表链结束，不证明所有trial成功。按用户已有授权，回传结果、检查完整性、依research iteration rules继续三模块novelty/整体novel paradigm/尽少方法超参数目标。不要重复启动旧实验；异常退出先诊断。"
  if "$CODEX" queue --thread "$THREAD" --message "$MESSAGE"; then
    echo "$(date -Is) $STATE" > "$MON/notification_sent"
    exit 0
  fi
  echo "$(date -Is) queue failed; retry in 120s"
  sleep 120
done
