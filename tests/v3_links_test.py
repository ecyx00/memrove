#!/usr/bin/env python3
"""Graph edges (E3): derivation rules, degree cap, no orphans, incremental scan."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_links as links  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402


class LinksTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='memrove-links-')
        self.addCleanup(tmp.cleanup)
        self.vault, state = Path(tmp.name) / 'v', Path(tmp.name) / 's'
        self.vault.mkdir()
        self.engine = SyncEngine(self.vault, state)
        self.store, self.links = self.engine.store, self.engine.links

    def put(self, id, **fields):
        source = f'cap/{id}.md'
        (self.vault / source).parent.mkdir(exist_ok=True)
        (self.vault / source).write_text('x', encoding='utf-8')
        self.store.ingest(dict(fields, id=id, source=source, text=fields.pop('text', 'x')))

    def derive(self):
        with self.store._connect() as db:
            return self.links.derive_missing(db)

    def edges(self, id):
        return {(s, d, r) for s, d, r, _ in self.links.for_node(id)}

    def test_rules_and_no_orphans(self):
        self.put('s1', kind='session', project='p', session='s1', ts='2026-01-01', files=[f'src/d{i}/a.py' for i in range(12)])
        self.put('s2', kind='session', project='p', session='s2', ts='2026-01-02')
        self.put('d1', kind='decision', project='p', session='s2', key='db', polarity='pos')
        self.put('r1', kind='rejected', project='p', session='s2', key='db', polarity='neg')
        self.put('d2', kind='decision', project='p', session='s2', key='db2', polarity='pos', supersedes=['d1'])
        self.put('n1', kind='note', text='see [[d1]] and [[nope]] and [[cap/r1|alias]]')
        self.assertEqual(self.derive(), 6)
        for rid in ('d1', 'r1', 'd2', 's1', 's2', 'n1'):
            self.assertTrue(any(r == 'in_project' for _, _, r in self.edges(rid)), rid)
        for rid in ('d1', 'r1', 'd2'):
            self.assertIn((rid, 's2', 'from_session'), self.edges(rid))
        self.assertIn(('r1', 'd1', 'rejects'), self.edges('d1'))
        self.assertIn(('d2', 'd1', 'supersedes'), self.edges('d2'))
        self.assertIn(('s2', 's1', 'next'), self.edges('s2'))
        self.assertEqual(sum(r == 'touches' for _, _, r in self.edges('s1')), 8)
        self.assertEqual({d for _, d, r in self.edges('n1') if r == 'mentions'}, {'d1', 'r1'})
        self.assertEqual(self.derive(), 0)  # idempotent: nothing left to derive

    def test_degree_cap_keeps_heaviest(self):
        for i in range(20):
            self.links.add('hub', f'x{i}', 'mentions', w=i / 100)
        out = [d for s, d, r, w in self.links.for_node('hub') if s == 'hub']
        self.assertEqual(len(out), links.DEGREE_CAP)
        self.assertIn('x19', out)
        self.assertNotIn('x0', out)

    def test_expand_signature_both_directions(self):
        self.links.add('r1', 'd1', 'rejects')
        self.links.add('d2', 'd1', 'supersedes')
        self.links.add('d1', 'area:src', 'touches')
        self.links.add('d1', 'project:p', 'in_project')
        links.bind(self.store)
        got = links.expand(['d1'], ('rejects', 'supersedes', 'touches'), 12)  # exactly E2's call
        self.assertEqual({(i, r) for i, r, _ in got}, {('r1', 'rejects'), ('d2', 'supersedes'), ('area:src', 'touches')})
        self.assertEqual(len(links.expand(['d1'], ('rejects', 'supersedes', 'touches'), 1)), 1)
        self.assertEqual(links.expand([], ('rejects',), 5), [])

    def test_projection_edge_ids_accepts_for_node(self):
        import memrove_v3_projections as proj
        self.links.add('s1', 'area:src', 'touches')
        self.links.add('n1', 's1', 'mentions')
        links.bind(self.store)
        old = proj._links
        proj._links = links
        self.addCleanup(setattr, proj, '_links', old)
        self.assertEqual(sorted(proj._edge_ids('s1')), [('area:src', 'touches'), ('n1', 'mentions')])

    def test_incremental_scan(self):
        a = self.vault / 'a.md'
        a.write_text('---\nid: a\nkind: note\n---\nlinks to [[b]]\n', encoding='utf-8')
        (self.vault / 'b.md').write_text('---\nid: b\nkind: note\n---\nhi\n', encoding='utf-8')
        self.assertEqual(self.engine.sync(full=False)['indexed'], 2)
        self.assertIn(('a', 'b', 'mentions'), self.edges('a'))
        first = self.engine.sync(full=False)
        self.assertEqual((first['indexed'], first['deleted']), (0, 0))  # nothing re-read
        a.write_text('---\nid: a\nkind: note\n---\nno links now, more text\n', encoding='utf-8')
        self.assertEqual(self.engine.sync(full=False)['indexed'], 1)
        self.assertNotIn(('a', 'b', 'mentions'), self.edges('a'))
        (self.vault / 'b.md').unlink()
        self.assertEqual(self.engine.sync(full=False)['deleted'], 1)
        self.assertEqual(self.edges('b'), set())
        self.assertEqual(self.engine.sync()['indexed'], 1)  # full scan still reads everything


if __name__ == '__main__':
    unittest.main()
