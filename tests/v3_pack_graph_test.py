#!/usr/bin/env python3
"""graph_pack: as-of and session exclusion, successor over superseded, whole-line cap, abstain on flat activation."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_pack_graph as pg  # noqa: E402

AS_OF = '2026-10-10T12:00:00+00:00'


class PackGraphTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='v3-pack-graph-')
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name)
        self.db = sqlite3.connect(self.state / 'memory.sqlite3')
        self.db.execute('CREATE TABLE records(id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.db.execute('CREATE TABLE graph_nodes(id TEXT PRIMARY KEY, kind TEXT, name TEXT, why TEXT, prov TEXT, ts TEXT)')
        self.db.execute('CREATE TABLE graph_edges(src TEXT, dst TEXT, type TEXT, why TEXT, prov TEXT, ts TEXT, PRIMARY KEY(src,dst,type))')

    def rec(self, rid, text, kind='decision', at='2026-10-05T10:00:00+00:00', project='P', **kw):
        body = {'kind': kind, 'status': 'active', 'project': project, 'recorded_at': at, 'text': text, **kw}
        self.db.execute('INSERT INTO records VALUES (?,?)', (rid, json.dumps(body)))

    def edge(self, a, b, typ='relates', ts='2026-10-05T10:00:00+00:00'):
        self.db.execute('INSERT INTO graph_edges VALUES (?,?,?,?,?,?)', (a, b, typ, 'w', '{}', ts))

    def pack(self, prompt='zebra cache', cap=1500, exclude='none'):
        self.db.commit()
        return pg.graph_pack(self.state, 'P', prompt, AS_OF, exclude, cap)

    def peaked(self):
        self.rec('a', 'zebra cache decision alpha')
        for n in 'bcdefg':
            self.rec(n, f'filler decision number {n}ish content{n}')
        self.edge('a', 'b')
        self.edge('a', 'c', 'contradicts')

    def test_node_after_as_of_never_appears(self):
        self.peaked()
        self.rec('late', 'zebra cache later correction', at='2026-10-12T10:00:00+00:00')
        self.rec('late2', 'quokka dated edge only')
        self.edge('a', 'late', 'supersedes')
        self.edge('a', 'late2', 'relates', ts='2026-10-12T00:00:00+00:00')
        out = self.pack()
        self.assertIn('alpha', out)
        self.assertNotIn('later correction', out)
        self.assertNotIn('dated edge only', out)

    def test_excluded_session_is_invisible(self):
        self.peaked()
        self.rec('own', 'zebra cache own session fix', session='S1')
        self.edge('a', 'own')
        self.assertIn('own session fix', self.pack())
        self.assertNotIn('own session fix', self.pack(exclude='S1'))

    def test_superseded_ranks_below_successor(self):
        self.peaked()
        self.rec('old', 'zebra cache old choice', at='2026-10-04T10:00:00+00:00')
        self.rec('new', 'zebra cache new choice', at='2026-10-06T10:00:00+00:00')
        self.edge('old', 'new', 'supersedes')  # edge direction must not matter: the newer record wins
        out = self.pack()
        self.assertIn('new choice', out)
        # the demoted predecessor may fall out of the top three entirely; it must never lead
        self.assertLess(out.index('new choice'), out.find('old choice') % (len(out) + 1) if 'old choice' in out else len(out))

    def test_cap_keeps_whole_lines(self):
        self.peaked()
        full = self.pack()
        lines = full.splitlines()
        self.assertGreater(len(lines), 1)
        cap = len(lines[0]) + 5
        out = self.pack(cap=cap)
        self.assertLessEqual(len(out), cap)
        self.assertEqual(out, lines[0])

    def test_flat_activation_abstains(self):
        for n in 'abcdef':
            self.rec(n, f'unrelated note {n}ish word{n}')
        self.assertEqual(self.pack(prompt='zebra cache'), '')
        self.assertEqual(pg.graph_pack(self.state, 'P', 'zebra', AS_OF, 'none', 1500), '')


if __name__ == '__main__':
    unittest.main()
