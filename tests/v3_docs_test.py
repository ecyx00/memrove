#!/usr/bin/env python3
"""E6: repo docs -> typed nodes, edges, stub notes, doctor metrics."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'template/.claude/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
import memrove_v3_docs as D  # noqa: E402
import memrove_v3_projections as P  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402

FILES = {
    'docs/state/2026-10-01-login-problem.md': '# Login redirect loop\n\nUsers bounce between /login and /home after the session cookie expires.\n\nSee `src/auth/session.py`.\n',
    'docs/state/2026-10-01-login-solution.md': '# Login redirect loop fixed\n\nCleared the stale session cookie in src/auth/session.py and added a guard.\n',
    'docs/state/2026-10-02-daylog.md': '# Day summary\n\nWorked on the login redirect loop and the platform structure; see [[architecture]].\n',
    'docs/architecture.md': '# Platform architecture\n\nThe auth module in src/auth handles session cookie and login flow.\n',
    'docs/rules.md': '# Coding rules\n\nAlways validate the session cookie before redirect. Login handlers must be idempotent.\n',
    'src/auth/session.py': 'x = 1\n',
}


class DocsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='memrove-e6-')
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.repo = base / 'shop'
        for rel, text in FILES.items():
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / rel).write_text(text, encoding='utf-8')
        self.vault = base / 'vault'
        self.vault.mkdir()
        self.engine = SyncEngine(self.vault, base / 'state')
        self.store = self.engine.store

    def rows(self):
        with self.store._connect() as db:
            return {r[0]: r for r in D.all_docs(db)}, D.all_edges(db)

    def test_kinds_edges_no_unlinked_and_incremental(self):
        t = time.monotonic()
        res = D.scan(self.store, self.repo)
        print('scan s=%.3f' % (time.monotonic() - t))
        self.assertEqual((res['docs'], res['changed']), (5, 5))
        docs, edges = self.rows()
        kinds = {Path(r[2]).name: r[5] for r in docs.values()}
        self.assertEqual(kinds['2026-10-01-login-problem.md'], 'problem')
        self.assertEqual(kinds['2026-10-01-login-solution.md'], 'solution')
        self.assertEqual(kinds['2026-10-02-daylog.md'], 'daylog')
        self.assertEqual(kinds['architecture.md'], 'reference')
        self.assertEqual(kinds['rules.md'], 'rule')
        self.assertIn(('doc:shop/docs/state/2026-10-01-login-solution.md', 'doc:shop/docs/state/2026-10-01-login-problem.md', 'solves'),
                      {e[:3] for e in edges})
        self.assertIn(('doc:shop/docs/state/2026-10-02-daylog.md', 'doc:shop/docs/architecture.md', 'mentions'), {e[:3] for e in edges})
        self.assertIn('area:src/auth', {e[1] for e in edges})
        self.assertIn('day:2026-10-01', {e[1] for e in edges})
        m = D.doctor(self.store)
        self.assertEqual((m['docs'], m['unlinked']), (5, 0), m)
        self.assertLessEqual(m['max_degree'], 12)
        again = D.scan(self.store, self.repo)
        self.assertEqual((again['changed'], again['removed']), (0, 0))
        (self.repo / 'docs/rules.md').unlink()
        self.assertEqual(D.scan(self.store, self.repo)['removed'], 1)

    def test_stub_notes_stable_and_hub_groups(self):
        D.scan(self.store, self.repo)
        P.project_graph(self.engine, budget_s=5)
        files = {p.relative_to(self.vault).as_posix(): p.read_bytes() for p in (self.vault / 'memrove').rglob('*.md')}
        stub = files['memrove/docs/shop/docs-state-2026-10-01-login-solution.md'].decode()
        self.assertIn('"generated": true', stub)
        self.assertIn('Source: `docs/state/2026-10-01-login-solution.md`', stub)
        self.assertIn('solves: [[memrove/docs/shop/docs-state-2026-10-01-login-problem]]', stub)
        hub = files['memrove/projects/shop.md'].decode()
        self.assertIn('## Docs: solution', hub)
        self.assertFalse((self.vault / 'docs').exists())  # never copied
        again = P.project_graph(self.engine, budget_s=5)
        self.assertEqual(again['written'], 0)
        self.assertEqual(files, {p.relative_to(self.vault).as_posix(): p.read_bytes() for p in (self.vault / 'memrove').rglob('*.md')})


if __name__ == '__main__':
    unittest.main()
