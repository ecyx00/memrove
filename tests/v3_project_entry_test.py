#!/usr/bin/env python3
"""E73: `memrove project` answers where a project stands from stored facts: current vs superseded decisions, open loops, last end state, size cap."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
os.environ['MEMROVE_THREADS'] = '1'  # the legacy thread/card writers are off by default (topic heads); these tests cover them
import memrove_v3_project as P  # noqa: E402
import memrove_v3_threads as th  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402


class ProjectEntryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='memrove-e73-')
        self.addCleanup(self.tmp.cleanup)
        self.vault = Path(self.tmp.name) / 'vault'
        (self.vault / 'wiki/_sources/grab').mkdir(parents=True)
        (self.vault / 'README.md').write_text('hello', encoding='utf-8')
        self.comp = self.vault / '🔮 850-Companion'
        self.comp.mkdir()
        (self.comp / 'Threads.md').write_text('# Threads\n\n## Active Threads\n\n## Closed Threads\n', encoding='utf-8')
        (self.comp / 'Last-Session.md').write_text('# Son oturum\n\n', encoding='utf-8')
        self.state = Path(self.tmp.name) / 'state'
        self.engine = SyncEngine(self.vault, self.state)
        base = dict(source='README.md', session='abcdef1234567890', project='/work/alpha')
        ing = self.engine.store.ingest
        ing(dict(base, id='dec:old', kind='decision', text='Use SQLite for the index.', updated_at='2026-09-01'))
        ing(dict(base, id='dec:new', kind='decision', text='Use Postgres for the index.', updated_at='2026-10-01', supersedes=['dec:old']))
        ing(dict(base, id='dec:edge-old', kind='decision', text='Poll the queue every minute.', updated_at='2026-09-05'))
        ing(dict(base, id='dec:edge-new', kind='decision', text='Push queue events instead of polling.', updated_at='2026-10-02'))
        ing(dict(base, id='dec:other', kind='decision', text='Beta only decision.', updated_at='2026-10-03', project='/work/beta'))
        (self.vault / 'wiki/_sources/grab/loop.md').write_text('---\nproject: alpha\nopen: true\ntitle: Ship the importer\n---\nbody\n', encoding='utf-8')
        ing(dict(base, id='card:loop', kind='note', text='Ship the importer', source='wiki/_sources/grab/loop.md', updated_at='2026-10-04'))
        th.apply_card(self.comp / 'Last-Session.md', 'sess0001xyz', 'alpha', 'index', '2026-10-05 10:00',
                      {'outcome': 'Index moved to Postgres.', 'reasoning': 'x', 'next': 'write the migration'}, [])
        th.apply_threads(self.comp / 'Threads.md', self.state, 'alpha', '2026-10-05',
                         [{'title': 'Importer', 'context': 'c', 'did': [], 'decisions': [], 'open': ['wire cli'], 'status': 'active'}], [])
        with self.engine.store._connect() as db:
            import memrove_v3_graph as G
            G.add_edge(db, 'dec:edge-new', 'dec:edge-old', 'supersedes', 'push replaces polling', G.provenance('agent:test', 'agent'))

    def text(self, limit=5000, name='alpha'):
        with self.engine.store._connect() as db:
            return P.entry(db, self.vault, name, limit)

    def test_latest_decision_is_current_and_earlier_is_superseded(self):
        out = self.text()
        cur, gone = out.split('[SUPERSEDED')[0], out
        self.assertRegex(out, r'rec:dec:new 2026-10-01 Use Postgres')
        self.assertNotRegex(out, r'rec:dec:new[^\n]*SUPERSEDED')
        self.assertRegex(out, r'rec:dec:old[^\n]*SQLite[^\n]*\[SUPERSEDED by rec:dec:new\]')  # record field
        self.assertRegex(out, r'rec:dec:edge-old[^\n]*\[SUPERSEDED by rec:dec:edge-new\]')  # stored graph edge
        self.assertNotIn('Beta only', out)
        self.assertLess(out.index('rec:dec:new'), out.index('rec:dec:old'))
        self.assertTrue(cur and gone)

    def test_same_day_order_follows_full_timestamp(self):
        ing = self.engine.store.ingest
        base = dict(source='README.md', session='abcdef1234567890', project='/work/alpha', kind='decision')
        ing(dict(base, id='dec:a-late', text='Late same-day call.', recorded_at='2026-10-06T18:00:00+00:00'))
        ing(dict(base, id='dec:b-early', text='Early same-day call.', recorded_at='2026-10-06T08:00:00+00:00'))
        ing(dict(base, id='dec:c-dateonly', text='Date-only call.', valid_from='2026-10-06'))
        out = self.text(limit=9000)
        self.assertLess(out.index('rec:dec:a-late'), out.index('rec:dec:b-early'))
        self.assertLess(out.index('rec:dec:b-early'), out.index('rec:dec:c-dateonly'))

    def test_remember_supersedes_marks_the_older_record(self):
        from memrove_v3_context import remember
        r = remember(self.engine, 'decision', 'Resume the paused eval work.', project='alpha', user=True, supersedes=['rec:dec:edge-new'])
        out = self.text()
        self.assertRegex(out, r'rec:dec:edge-new[^\n]*\[SUPERSEDED by rec:' + r['id'])
        self.assertRegex(out, r'rec:' + r['id'] + r' [^\n]*Resume')
        with self.assertRaises(ValueError):
            remember(self.engine, 'decision', 'Other call.', project='alpha', supersedes=['nope-404'])

    def test_open_loop_thread_and_last_end_state(self):
        out = self.text()
        self.assertRegex(out, r'rec:card:loop 2026-10-04 Ship the importer')
        self.assertIn('th:t-', out)
        self.assertIn('Importer', out)
        self.assertIn('last session (end state):', out)
        self.assertIn('Index moved to Postgres.', out)
        self.assertIn('write the migration', out)

    def test_size_cap_and_unknown_project(self):
        self.assertLessEqual(len(self.text(limit=400).encode()), 400 + 80)
        self.assertLess(len(self.text().encode()), 6000)
        self.assertTrue(self.text(name='nope').startswith("no project 'nope'; projects: "))

    def test_cli_defaults_to_cwd_project(self):
        (Path(self.tmp.name) / 'alpha/.git').mkdir(parents=True)
        r = subprocess.run([sys.executable, str(ROOT / 'scripts/memrove_v3.py'), '--vault', str(self.vault), '--state', str(self.state), 'project'],
                           cwd=Path(self.tmp.name) / 'alpha', capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('project alpha', r.stdout)


if __name__ == '__main__':
    unittest.main()
