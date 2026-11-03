#!/usr/bin/env python3
"""memrove pack: the as-of view hides later records and the excluded session in every arm; the card arm drafts once and reuses its cache."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_llm as llm  # noqa: E402
import memrove_v3_pack as pack  # noqa: E402

AS_OF = '2026-10-10T12:00:00+00:00'
OLD, LATE, SAME = 'Standing rule OLD says tests run in parallel always.', 'Standing rule LATE says never touch the vault again.', 'Standing rule SAME is from the excluded session.'
EXC = 'ffffffffffff-excluded'


def rec(i, sentence, day, session='aaaaaaaaaaaa-other', **kw):
    a = {'assessments': {'restate': {'state': 'assessed', 'answers': {'standing_rule': 'yes', 'scope': 'global', 'sentence': sentence}}}}
    return dict(id=i, kind='preference', origin_type='user_explicit', valid_from=day, session=session, project='repo', status='active', **a, **kw)


class PackTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='v3-pack-')
        self.addCleanup(tmp.cleanup)
        d = self.dir = Path(tmp.name)
        self.state, self.cache, self.tp = d / 'state', d / 'cache', d / 'projects'
        for p in (self.state, self.tp / 'repo-dir'):
            p.mkdir(parents=True)
        db = sqlite3.connect(self.state / 'memory.sqlite3')
        db.execute('CREATE TABLE records(id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        late_card = dict(id='c-late', kind='card', project='repo', status='active', valid_from='2026-10-12', did=['LATECARD'], session='bbbbbbbbbbbb-x')
        for r in (rec('p-old', OLD, '2026-10-01'), rec('p-late', LATE, '2026-10-11'), rec('p-same', SAME, '2026-10-02', session=EXC), late_card):
            db.execute('INSERT INTO records VALUES(?,?)', (r['id'], json.dumps(r)))
        db.commit()
        db.close()
        self.d_proj = d / 'repo'
        (self.d_proj / '.git').mkdir(parents=True)
        lines = [{'type': 'user', 'cwd': str(self.d_proj), 'timestamp': '2026-10-05T10:00:00.000Z', 'message': {'content': 'karar verdim, sqlite kullanalım'}},
                 {'type': 'assistant', 'timestamp': '2026-10-05T10:05:00.000Z', 'message': {'content': [{'type': 'text', 'text': 'ok'}]}}]
        (self.tp / 'repo-dir' / 'prev0123456789.jsonl').write_text('\n'.join(json.dumps(x) for x in lines), encoding='utf-8')
        (self.dir / 'vault').mkdir()
        self.calls = []
        llm.RUNNER = self.runner
        self.addCleanup(setattr, llm, 'RUNNER', llm._claude)

    def runner(self, system, payload, vault=None):
        self.calls.append(payload)
        return json.dumps({'did': ['PREVDID chose sqlite'], 'why': [], 'open': [], 'next': 'go'}), 0.01, None

    def block(self, arm):
        return pack.pack(self.state, arm, 'repo', 'p', AS_OF, EXC, cache=self.cache, transcripts=self.tp)

    def test_later_and_excluded_records_absent_in_every_arm(self):
        for arm in ('none', 'rules', 'card'):
            b = self.block(arm)
            for hidden in ('LATE', 'SAME', 'LATECARD'):
                self.assertNotIn(hidden, b, arm)
        self.assertEqual(self.block('none'), '')
        self.assertIn('OLD', self.block('rules'))

    def test_card_arm_drafts_previous_session_once(self):
        first = self.block('card')
        self.assertIn('PREVDID', first)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.block('card'), first)
        self.assertEqual(len(self.calls), 1)   # cache hit

    def test_cli_graph_arm_without_module_fails_for_that_arm_only(self):
        base = [sys.executable, str(ROOT / 'scripts/memrove_v3.py'), '--vault', str(self.dir / 'vault'), '--state', str(self.state), 'pack', '--project', 'repo',
                '--as-of', AS_OF, '--exclude-session', EXC, '--cache', str(self.cache), '--transcripts', str(self.tp)]
        ok = subprocess.run(base + ['--arm', 'rules'], capture_output=True, text=True)
        self.assertEqual((ok.returncode, 'OLD' in ok.stdout), (0, True), ok.stderr)
        try:
            import memrove_v3_pack_graph  # noqa: F401
        except ImportError:
            bad = subprocess.run(base + ['--arm', 'graph'], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            self.assertIn('graph arm unavailable', bad.stderr)


if __name__ == '__main__':
    unittest.main()
