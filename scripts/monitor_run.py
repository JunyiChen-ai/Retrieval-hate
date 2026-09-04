#!/usr/bin/env python3
"""Notify this Codex thread once a detached process group finishes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from datetime import datetime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--pgid', required=True, type=int)
    p.add_argument('--remote-run', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--thread', required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'monitor.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if (out / 'notification_sent').exists():
        return
    (out / 'run.pid').write_text(str(os.getpid()) + '\n')
    (out / 'config.json').write_text(json.dumps(vars(a), indent=2))
    # The launched job uses setsid, so descendants retain this process group.
    probe = (
        "import subprocess,json,pathlib; "
        "rows=subprocess.check_output(['ps','-eo','pgid=,stat=,args='],text=True).splitlines(); "
        f"alive=any(r.split(None,2)[0]=={str(a.pgid)!r} and not r.split(None,2)[1].startswith('Z') for r in rows if len(r.split(None,2))==3); "
        f"done=pathlib.Path({a.remote_run!r})/'completion.json'; "
        "print('RUNNING' if alive else ('OUTPUT_FINISHED' if done.exists() else 'STOPPED_WITHOUT_COMPLETION'))"
    )
    while True:
        try:
            command = (['python3', '-c', probe] if a.host == 'local' else
                       ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                        a.host, 'python3 -c ' + shlex.quote(probe)])
            result = subprocess.run(command,
                                    text=True, capture_output=True, timeout=45)
            state = result.stdout.strip() if result.returncode == 0 else 'SSH_UNAVAILABLE'
        except subprocess.TimeoutExpired:
            state = 'SSH_UNAVAILABLE'
        now = datetime.now().astimezone().isoformat()
        print(now, state, flush=True)
        if state in ['OUTPUT_FINISHED', 'STOPPED_WITHOUT_COMPLETION']:
            msg = (f'实验监控通知：{a.host} {a.remote_run} 状态={state}，观察时间={now}。'
                   '请核验进程与输出覆盖率，回传缓存/结果；完成标记不证明完整性。异常退出先诊断，不重复启动旧任务。'
                   '依已有授权与研究规则继续三模块novelty、整体novel paradigm、尽少方法超参数目标。')
            sent = subprocess.run([str(Path.home() / '.local/bin/codex'), 'queue',
                                   '--thread', a.thread, '--message', msg])
            if sent.returncode == 0:
                (out / 'notification_sent').write_text(now + ' ' + state + '\n')
                return
        time.sleep(120)


if __name__ == '__main__':
    main()
