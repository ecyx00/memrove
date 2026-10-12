#!/usr/bin/env python3
"""Recall hook latency benchmark (not part of the unit suite: wall and CPU time flake under machine load).
Usage: python3 tests/v3_recall_benchmark.py   -> prints latency and CPU stats; exit 1 if child CPU p95 > 150 ms."""
import json
import os
import resource
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'template/.claude/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
import memrove_v3 as v3

LESSON = dict(kind='lesson', project='other', title='Stripe webhook retries', topics=['idempotency', 'webhook'],
              approach='dedupe by event id', result='unnecessary', why='stripe already retries safely', session='sess-9',
              text='Stripe webhook retries idempotency keys', origin_type='user_explicit', confidence=0.9,
              updated_at='2026-10-01T00:00:00Z')


def main():
    os.environ['MEMROVE_NO_LLM'] = '1'
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        vault, state, repo = root / 'vault', root / 'state', root / 'here'
        for p in (vault, repo / '.git'):
            p.mkdir(parents=True)
        v3.MemoryStore(state, vault)
        with sqlite3.connect(state / 'memory.sqlite3') as db:
            for i in range(2000):
                rec = dict(LESSON, id=f'r{i}', source=f'r{i}.md', revision=1, project=f'p{i % 40}', status='active', visibility='internal',
                           facts={}, supersedes=[], title=f'Topic {i} alpha{i % 97} beta{i % 31}', topics=[f'tag{i % 53}', f'area{i % 17}'],
                           approach=f'approach number {i} using tool{i % 23}', why=f'because reason {i}', text=f'note body {i} ' + 'filler ' * 30,
                           kind=('lesson', 'decision', 'rejected', 'session')[i % 4])
                db.execute('INSERT INTO records VALUES(?,?)', (rec['id'], json.dumps(rec, sort_keys=True)))
        payload = {'prompt': 'how do I wire stripe webhook retries with idempotency keys alpha5 beta5 tool5', 'cwd': str(repo),
                   'hook_event_name': 'UserPromptSubmit'}
        times, cpu = [], []
        for i in range(20):
            t, r0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_CHILDREN)
            r = subprocess.run([sys.executable, str(SCRIPTS / 'memrove_v3_hook.py'), '--vault', str(vault), '--state', str(state), '--harness', 'claude',
                                '--global'], input=json.dumps(dict(payload, session_id=f'x{i}')), text=True, capture_output=True, cwd=repo,
                               env=dict(os.environ, HOME=str(root), MEMROVE_V3_NO_SPAWN='1'))
            r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            if r.returncode:
                sys.exit(r.stderr)
            times.append((time.perf_counter() - t) * 1000)
            cpu.append((r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime) * 1000)
        times.sort()
        cpu.sort()
        print(f'wall ms: min={times[0]:.0f} median={statistics.median(times):.0f} p95={times[18]:.0f} max={times[-1]:.0f}')
        print(f'child CPU ms: median={statistics.median(cpu):.0f} p95={cpu[18]:.0f} max={cpu[-1]:.0f}')
        return 1 if cpu[18] > 150 else 0


if __name__ == '__main__':
    sys.exit(main())
