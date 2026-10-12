#!/usr/bin/env python3
"""E1: transcript cursor capture, global-mode guards, source-aware SessionStart, hook latency."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
import os as _os
_os.environ.setdefault("MEMROVE_NO_LLM", "1")  # tests never call the real Haiku CLI
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'template/.claude/scripts'
HOOK = SCRIPTS / 'memrove_v3_hook.py'
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True


def line(**kw):
    return json.dumps(kw, ensure_ascii=False) + '\n'


def user(text):
    return line(type='user', message={'content': text})


def tool(name, **inp):
    return line(type='assistant', message={'content': [{'type': 'tool_use', 'name': name, 'input': inp}]})


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v3-capture-')
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.vault, self.state, self.repo = root / 'vault', root / 'state', root / 'repo'
        for p in (self.vault, self.repo / '.git'):
            p.mkdir(parents=True)
        self.tp = root / 't.jsonl'
        self.env = dict(os.environ, MEMROVE_V3_NO_SPAWN='1', HOME=str(root))
        self.env.pop('CLAUDE_CODE_ENTRYPOINT', None)

    def hook(self, payload, extra=()):
        r = subprocess.run([sys.executable, str(HOOK), '--vault', str(self.vault), '--state', str(self.state),
                            '--harness', 'claude', '--global', *extra], input=json.dumps(payload), text=True, capture_output=True,
                           cwd=self.repo, env=self.env, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def event(self, name='SessionEnd'):
        return {'hook_event_name': name, 'session_id': 's1', 'transcript_path': str(self.tp), 'cwd': str(self.repo)}

    def test_thirty_quotes_one_run(self):
        from memrove_v3_sync import SyncEngine
        self.tp.write_text(''.join(user(f'Bundan sonra kural numarası {i} her zaman geçerli olsun.') for i in range(30)), encoding='utf-8')
        self.hook(self.event())
        from memrove_v3_hook import drain_queue
        drain_queue(self.vault, self.state)
        with SyncEngine(self.vault, self.state).store._connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM records WHERE json_extract(payload,'$.kind')='preference'").fetchone()[0], 30)

    def test_capture_cursor_and_records(self):
        from memrove_v3_sync import SyncEngine
        self.tp.write_text(
            user('Bundan sonra testleri her zaman önce yaz.') + user('Hayır, bunu yapma, sqlite yerine json istemiyorum.') +
            user('Hmm bu nasıl olur?') + user('password = hunter2hunter2 olsun her zaman') +
            tool('Edit', file_path='src/a.py') + tool('TodoWrite', todos=[{'content': 'ship E1', 'status': 'pending'}, {'content': 'old', 'status': 'completed'}]),
            encoding='utf-8')
        self.hook(self.event())
        self.hook(self.event())  # second event: nothing new
        from memrove_v3_hook import drain_queue
        self.assertEqual(drain_queue(self.vault, self.state)['failed'], 0)
        engine = SyncEngine(self.vault, self.state)
        with engine.store._connect() as db:
            recs = [json.loads(r[0]) for r in db.execute('SELECT payload FROM records')]
        by = {r['kind']: r for r in recs if r['kind'] != 'task'}
        self.assertEqual(sorted(by), ['preference', 'rejected', 'session'])
        self.assertEqual(by['rejected']['polarity'], 'neg')
        self.assertEqual(by['preference']['verified_by'], 'user_quote')
        self.assertEqual(by['preference']['project'], 'repo')
        self.assertTrue(all(isinstance(r['id'], str) for r in recs))
        self.assertFalse(any('hunter2' in json.dumps(r) for r in recs))  # secret sentence dropped
        self.assertEqual([r['title'] for r in recs if r['kind'] == 'task'], ['ship E1'])
        n = len(recs)
        with self.tp.open('a', encoding='utf-8') as f:  # incremental: only the new quote is read
            f.write(user('Karar verdim: postgres kullanacağız.'))
        self.hook(self.event('PreCompact'))
        drain_queue(self.vault, self.state)
        with engine.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM records').fetchone()[0], n + 1)

    def test_global_guards(self):
        self.tp.write_text(user('Her zaman türkçe cevap ver.'), encoding='utf-8')
        (self.repo / '.memrove').mkdir()
        (self.repo / '.memrove' / 'no-memory').write_text('')
        self.assertEqual(self.hook(self.event()).strip(), '{}')
        self.assertFalse((self.state / 'hook-queue').exists())
        (self.repo / '.memrove' / 'no-memory').unlink()
        self.tp.write_text(line(type='system', entrypoint='sdk-cli') + user('Her zaman türkçe cevap ver.'), encoding='utf-8')
        self.assertEqual(self.hook(self.event()).strip(), '{}')
        self.assertFalse((self.state / 'hook-queue').exists())

    def test_no_gap_warning_and_source_dispatch(self):
        (self.state).mkdir()
        (self.state / 'hook-health.json').write_text(json.dumps({'sync': {'status': 'ok', 'potential_missing_receipts': 3}}))
        out = self.hook({'hook_event_name': 'SessionStart', 'session_id': 's1', 'source': 'compact', 'cwd': str(self.repo)})
        self.assertNotIn('lack structured receipts', out)

    def test_hook_p95(self):
        times = []
        for i in range(15):
            t = time.perf_counter()
            self.hook(dict(self.event('Stop'), session_id='p%d' % i))
            times.append(time.perf_counter() - t)
        print('hook p95 ms (incl. interpreter start):', round(sorted(times)[13] * 1000))
        self.assertLess(sorted(times)[13], 0.6)


if __name__ == '__main__':
    unittest.main()
