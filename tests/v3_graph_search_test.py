#!/usr/bin/env python3
"""`memrove search` follows graph_edges: neighbour lines under a hit, edge order, the newer record first, a size cap; `show` lists edges."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402
import memrove_v3_docsearch as docsearch  # noqa: E402


class GraphSearchTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='v3-graph-search-')
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name)
        self.db = sqlite3.connect(self.state / 'memory.sqlite3')
        self.db.execute('CREATE TABLE records(id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.db.execute('CREATE TABLE graph_nodes(id TEXT PRIMARY KEY, kind TEXT, name TEXT, why TEXT, prov TEXT, at TEXT)')
        self.db.execute('CREATE TABLE graph_edges(src TEXT, dst TEXT, type TEXT, why TEXT, prov TEXT, at TEXT, PRIMARY KEY(src,dst,type))')
        patch = mock.patch.object(docsearch, 'repos', return_value=[])   # search walks every docs/*.md of the repos on this machine: none here
        patch.start()
        self.addCleanup(patch.stop)
        old = cs.GRAPH_SHARE   # the share is of the plain output, which is one short line here (no docs tail): give the order tests room
        cs.GRAPH_SHARE = 5.0
        self.addCleanup(setattr, cs, 'GRAPH_SHARE', old)
        chatlog.append(self.state, [{'kind': 'user', 'text': 'unrelated chatter', 'date': '2026-10-07T10:00:00+00:00'}])

    def rec(self, rid, text, **kw):
        body = {'kind': 'decision', 'status': 'active', 'source': 'notes/observed/decision/x.md', 'valid_from': '2026-10-05', 'text': text, **kw}
        self.db.execute('INSERT INTO records VALUES (?,?)', (rid, json.dumps(body)))

    def edge(self, src, dst, typ, why='because of the shared topic'):
        self.db.execute('INSERT INTO graph_edges VALUES (?,?,?,?,?,?)', (src, dst, typ, why, '{}', '2026-10-06'))

    def search(self, q='zebra', **kw):
        self.db.commit()
        return cs.search(self.state, q, **kw)

    def test_neighbour_lines_follow_the_hit(self):
        self.rec('a', 'zebra cache decision')
        self.rec('b', 'redis is the store', title='Redis store')
        self.edge('a', 'b', 'relates', 'both pick the cache store')
        lines = self.search().splitlines()
        self.assertTrue(lines[0].startswith('rec:a|'))
        self.assertEqual(lines[1], '  > relates rec:b Redis store - both pick the cache store')

    def test_order_and_cap_of_two_neighbours(self):
        self.rec('a', 'zebra cache decision')
        for n, t in (('r', 'relates'), ('p', 'contradicts'), ('s', 'answers')):
            self.rec(n, f'other {n}')
            self.edge('a', n, t)
        self.db.execute("INSERT INTO graph_nodes VALUES ('concept:cache','concept','Cache','w','{}','x')")
        self.edge('a', 'concept:cache', 'part_of')
        lines = [x for x in self.search().splitlines() if x.startswith('  >')]
        self.assertEqual([l.split()[1] for l in lines], ['contradicts', 'answers'])   # supersedes/contradicts/answers before part_of before relates, two at most

    def test_part_of_before_relates(self):
        self.rec('a', 'zebra cache decision')
        self.rec('r', 'other r')
        self.db.execute("INSERT INTO graph_nodes VALUES ('concept:cache','concept','Cache','w','{}','x')")
        self.edge('a', 'r', 'relates')
        self.edge('a', 'concept:cache', 'part_of')
        self.assertEqual([l.split()[1] for l in [x for x in self.search().splitlines() if x.startswith('  >')]], ['part_of', 'relates'])

    def test_newer_record_first_and_old_marked_replaced(self):
        self.rec('old', 'zebra cache uses memcached')
        self.rec('new', 'we moved the store to redis')
        self.edge('new', 'old', 'supersedes', 'redis replaced memcached')
        lines = [l for l in self.search().splitlines() if not l.startswith('  >')]
        self.assertTrue(lines[0].startswith('rec:new|'))
        self.assertTrue(lines[1].startswith('rec:old|') and '[replaced by rec:new]' in lines[1])

    def test_hidden_neighbour_is_not_shown(self):
        self.rec('a', 'zebra cache decision')
        self.rec('x', 'private one', visibility='private')
        self.rec('y', 'dropped one', status='dropped_llm')
        self.edge('a', 'x', 'relates')
        self.edge('a', 'y', 'relates')
        self.assertNotIn('  >', self.search())

    def test_added_text_is_capped_at_40_percent(self):
        cs.GRAPH_SHARE = 0.4
        for i in range(4):
            self.rec(f'h{i}', 'zebra cache decision ' + 'word ' * 30 + str(i))
            for j in range(2):
                self.rec(f'n{i}{j}', 'neighbour ' + 'long text ' * 20, title='T' * 70)
                self.edge(f'h{i}', f'n{i}{j}', 'relates', 'w' * 80)
        out = self.search()
        added = sum(len(l) + 1 for l in out.splitlines() if l.startswith('  >'))
        base = len(out) - added
        self.assertGreater(added, 0)
        self.assertLessEqual(added, 0.4 * base)

    def test_graph_never_drops_a_plain_hit(self):
        self.rec('old', 'zebra cache uses memcached')
        self.rec('new', 'we moved the store to redis')
        self.rec('c', 'zebra cache other')
        self.rec('n', 'neighbour', title='N')
        self.edge('new', 'old', 'supersedes')
        self.edge('c', 'n', 'relates')
        ids = lambda out: {l.split('|')[0] for l in out.splitlines() if l and not l[0].isspace() and '|' in l}
        with_graph = ids(self.search())
        cs_graph = cs.graph
        cs.graph = lambda state, rows, room, history=False: (rows, {})
        try:
            plain = ids(self.search())
        finally:
            cs.graph = cs_graph
        self.assertTrue(plain <= with_graph and with_graph - plain == {'rec:new'})

    def test_edges_of_lists_every_edge_with_why(self):
        import memrove_v3_neighbours as nb
        self.rec('a', 'zebra')
        self.rec('b', 'other', title='B title')
        self.edge('a', 'b', 'contradicts', 'they disagree')
        self.db.commit()
        for rid, d in (('a', 'out'), ('b', 'in')):
            e = nb.edges_of(self.state, rid)
            self.assertEqual((len(e), e[0]['type'], e[0]['dir'], e[0]['why']), (1, 'contradicts', d, 'they disagree'))


if __name__ == '__main__':
    unittest.main()
