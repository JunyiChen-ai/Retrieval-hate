#!/usr/bin/env python3
"""Poll a detached experiment's identity and children, then notify its Codex thread."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time


def probe(run):
    root = Path(run)
    identity = json.loads((root / 'identity.json').read_text())
    if identity['host'] != socket.gethostname() or str(root.resolve()) != identity['run']:
        raise RuntimeError('Host or run directory differs from the recorded launch identity')
    processes = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            args = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            pid, pgid, sid = int(p.name), int(fields[2]), int(fields[3])
            leader = pid == identity['pid'] and fields[19] == identity['start_ticks']
            child = (pgid == identity['pgid'] and sid == identity['sid']
                     and identity['repo'] in args and pid != identity['pid'])
            if leader or child:
                processes.append({'pid': pid, 'ppid': int(fields[1]), 'leader': leader})
        except (OSError, IndexError, ValueError):
            continue
    completion = None
    if (root / 'completion.json').exists():
        completion = json.loads((root / 'completion.json').read_text())
    state = 'RUNNING' if processes else (
        'OUTPUT_FINISHED' if completion and completion.get('status') == 'success' else
        'FAILED' if completion else 'STOPPED_WITHOUT_COMPLETION')
    return {'state': state, 'identity': identity, 'processes': processes, 'completion': completion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--host')
    ap.add_argument('--pgid', type=int)
    ap.add_argument('--remote-run', required=True)
    ap.add_argument('--out-dir')
    ap.add_argument('--thread')
    ap.add_argument('--interval', type=int, default=120)
    ap.add_argument('--continuation', default='核验实际进程与评测完整性，回传结果，继续已授权的 backbone novelty 自动迭代；异常先诊断，不重复启动旧任务。')
    a = ap.parse_args()
    if a.probe:
        print(json.dumps(probe(a.remote_run)))
        return
    if not (a.host and a.out_dir and a.thread):
        ap.error('--host, --out-dir and --thread are required for monitoring')
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'monitor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if (out / 'notification_sent').exists():
        return
    (out / 'run.pid').write_text(str(os.getpid()) + '\n')
    (out / 'config.json').write_text(json.dumps(vars(a), indent=2, ensure_ascii=False))
    # Standard layout: <repo>/runs/<experiment>/<corpus>/seed<seed>.
    remote = Path(a.remote_run)
    parts = remote.parts
    run_index = parts.index('runs')
    repo = Path(*parts[:run_index])
    command = ['python3', str(repo / 'scripts/monitor_run.py'), '--probe', '--remote-run', str(remote)]
    if a.host != 'local':
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', a.host, shlex.join(command)]
    binding = None
    if (out / 'binding.json').exists():
        binding = json.loads((out / 'binding.json').read_text())
    stopped_count = 0
    while True:
        now = datetime.now().astimezone().isoformat()
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=45)
            if result.returncode:
                raise RuntimeError(result.stderr.strip()[-800:] or 'probe failed')
            observation = json.loads(result.stdout)
            identity = observation['identity']
            if a.pgid is not None and identity['pgid'] != a.pgid:
                raise RuntimeError('Recorded process group differs from monitor configuration')
            if binding is None:
                binding = identity
                (out / 'binding.json').write_text(json.dumps(binding, indent=2))
            elif identity != binding:
                raise RuntimeError('Run identity changed; observation is not the bound run')
            state = observation['state']
            (out / 'last_check.json').write_text(json.dumps({'time': now, **observation}, indent=2))
            print(now, state, 'live_processes', len(observation['processes']), flush=True)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            print(now, 'OBSERVATION_UNAVAILABLE; will retry:', str(exc), flush=True)
            stopped_count = 0
            time.sleep(a.interval)
            continue
        if state == 'STOPPED_WITHOUT_COMPLETION':
            stopped_count += 1
        else:
            stopped_count = 0
        terminal = state in ('OUTPUT_FINISHED', 'FAILED') or stopped_count >= 3
        if terminal:
            message = (f'实验监控通知：{a.host} {a.remote_run}，状态={state}，时间={now}。'
                       f'主进程与同会话相关子进程均已结束。{a.continuation} '
                       '完成标记不能替代评测输出完整性核验。')
            try:
                sent = subprocess.run([str(Path.home() / '.local/bin/codex'), 'queue',
                                       '--thread', a.thread, '--message', message],
                                      text=True, capture_output=True, timeout=45)
                print(now, 'notification', sent.returncode, sent.stdout.strip(), sent.stderr.strip(), flush=True)
                if sent.returncode == 0:
                    (out / 'notification_sent').write_text(now + ' ' + state + '\n')
                    return
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(now, 'notification failed; will retry:', str(exc), flush=True)
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
