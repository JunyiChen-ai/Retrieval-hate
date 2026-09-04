"""Send periodic reminders to an existing Codex thread; no model polling.

Runtime state belongs in runs/. Stop with the same arguments plus --stop.
Requires the local Codex app-server and its queue command to remain available.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thread', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--interval', type=int, default=10800)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('interval must be positive')
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stop_file = root / 'STOP'
    if args.stop:
        stop_file.write_text(datetime.now().astimezone().isoformat() + '\n')
        print('Stop requested; monitor exits within 30 seconds.', flush=True)
        return
    lock = (root / 'monitor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Monitor already running.', flush=True)
        return
    if stop_file.exists():
        print('STOP exists; monitor remains stopped.', flush=True)
        return
    event = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: event.set())
    message = '请检查当前用户设定目标是否已完成 如未完成 请继续推进 如已完成或碰到硬阻塞无法推进 关掉该monitor并向用户报告'
    state_file = root / 'state.json'
    state = {'thread': args.thread, 'interval_seconds': args.interval,
             'message': message, 'next_send_epoch': time.time() + args.interval,
             'sent_count': 0}
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state['thread'] != args.thread or state['interval_seconds'] != args.interval:
            raise RuntimeError('Existing monitor configuration differs; use a separate output directory')
    def save():
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    (root / 'run.pid').write_text(str(os.getpid()) + '\n')
    save()
    print('Started; next reminder ' + datetime.fromtimestamp(state['next_send_epoch']).astimezone().isoformat(), flush=True)
    while not event.is_set() and not stop_file.exists():
        remaining = state['next_send_epoch'] - time.time()
        if remaining > 0:
            event.wait(min(30, remaining))
            continue
        try:
            result = subprocess.run(['/home/jehc223/.local/bin/codex', 'queue',
                                     '--thread', args.thread, '--message', message],
                                    text=True, capture_output=True, timeout=60)
            print(datetime.now().astimezone().isoformat(), result.stdout, result.stderr, flush=True)
            if result.returncode != 0:
                event.wait(60)
                continue
        except (OSError, subprocess.TimeoutExpired) as error:
            print('Queue unavailable; retry:', error, flush=True)
            event.wait(60)
            continue
        state['sent_count'] += 1
        state['last_send_epoch'] = time.time()
        state['next_send_epoch'] = state['last_send_epoch'] + args.interval
        save()
    state['stopped_at'] = datetime.now().astimezone().isoformat()
    save()
    print('Stopped.', flush=True)


if __name__ == '__main__':
    main()
