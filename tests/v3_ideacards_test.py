#!/usr/bin/env python3
"""Idea cards: reply parsing, none handling, cursor resume after a failed batch, search_cards (BM25 only: no daemon in tests)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'template/.claude/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
os.environ['MEMROVE_MODELD_SOCK'] = '/nonexistent/modeld.sock'  # daemon down: vectors skipped, BM25 only
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_ideacards as ic  # noqa: E402
import memrove_v3_llm as llm  # noqa: E402

TEXTS = ['Hafıza geri çağırma BM25 ile olmaz, anlam aramalı', 'ne durumda', 'Subagent briefleri İngilizce, kartlar yazarken üretilsin', 'başla', 'Disk dar, scratch /tmp']


class Cards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        chatlog.append(self.state, [{'kind': 'user' if i != 3 else 'agent', 'text': t, 'date': '2026-11-03T00:00:00+03:00'} for i, t in enumerate(TEXTS)])
        self.addCleanup(setattr, llm, 'RUNNER', llm._claude)
        self.calls = []
        self.addCleanup(setattr, ic, 'embed_missing', ic.embed_missing)
        self.real_embed = ic.embed_missing
        ic.embed_missing = lambda *a, **k: 0  # no daemon spawn wait in tests

    def runner(self, fail_at=()):
        def run(system, payload, vault=None):
            self.calls.append(payload)
            if len(self.calls) in fail_at:
                return None, 0.0, 'boom'
            out = []
            for line in payload.splitlines():
                m = json.loads(line)
                none = m['t'] in ('ne durumda', 'başla')
                out.append({'i': m['i'], 'card': 'none' if none else 'Owner wants ' + ('semantic memory recall' if 'BM25' in m['t'] else 'English subagent briefs'),
                            'project': 'memrove', 'topics': ['recall', 'memory']})
            return json.dumps(out), 0.0, None
        return run

    def test_parse_none_and_junk(self):
        r = ic.parse([{'i': 1, 'card': 'None.', 'topics': []}, {'i': 2, 'card': 'An idea', 'project': 'X', 'topics': ['a', 5, 'b']}, {'i': 9, 'card': 'x'}, 'junk', {'i': 2, 'card': 'dup'}], {1, 2})
        self.assertEqual(r, {1: (None, '', ''), 2: ('An idea', 'x', 'a, b')})
        self.assertEqual(ic.parse('nope', {1}), {})

    def test_build_counts_and_search(self):
        llm.RUNNER = self.runner()
        st = ic.build(self.state, None, max_calls=5)
        self.assertEqual((st['messages'], st['cards'], st['none'], st['calls']), (4, 3, 1, 1))
        self.assertEqual(ic.search_cards(self.state, 'semantic recall for memory', 3)[0][0], 0)
        self.assertEqual(ic.search_cards(self.state, 'English briefs', 3)[0][0], 2)
        self.assertEqual(ic.search_cards(self.state, 'zzzz qqqq'), [])
        self.assertEqual(ic.build(self.state, None, max_calls=5)['calls'], 0)  # nothing pending: no call

    def test_cursor_resume(self):
        old = ic.BATCH_MSGS
        ic.BATCH_MSGS = 2
        self.addCleanup(setattr, ic, 'BATCH_MSGS', old)
        llm.RUNNER = self.runner(fail_at=(1,))
        st = ic.build(self.state, None, max_calls=5)  # call 1 fails, batch 2 stored, cursor stays before the hole
        self.assertEqual(st['messages'], 2)
        db = ic.connect(self.state)
        self.assertEqual(int(db.execute("SELECT value FROM meta WHERE key='cursor'").fetchone()[0]), -1)
        self.assertEqual([m['i'] for m in ic.pending(self.state, db)], [0, 1])
        llm.RUNNER = self.runner()
        self.assertEqual(ic.build(self.state, None, max_calls=5)['messages'], 2)
        self.assertEqual(int(db.execute("SELECT value FROM meta WHERE key='cursor'").fetchone()[0]), 4)
        self.assertEqual(ic.pending(self.state, db), [])

    def test_daily_cap(self):
        llm.RUNNER = self.runner()
        (self.state / 'llm-config.json').write_text('{"ideacards_cap": 0}')
        (self.state / 'chatlog').mkdir(exist_ok=True)
        self.assertEqual(ic.build(self.state, None)['stopped'], 'budget')
        self.assertEqual(self.calls, [])

    def test_usage_counted_when_loop_raises(self):
        llm.RUNNER = self.runner()
        self.addCleanup(setattr, ic, 'store', ic.store)
        ic.store = lambda *a, **k: 1 / 0
        with self.assertRaises(ZeroDivisionError):
            ic.build(self.state, None)
        self.assertEqual(ic._used(self.state), 1)

    def test_failed_batch_stops_blocking(self):
        old = ic.BATCH_MSGS
        ic.BATCH_MSGS = 2
        self.addCleanup(setattr, ic, 'BATCH_MSGS', old)
        good = self.runner()
        llm.RUNNER = lambda s, p, v=None: ('not json', 0.0, None) if '"i": 0,' in p else good(s, p, v)
        ic.build(self.state, None, max_calls=5)  # run 1: first batch unparseable (attempt 1), second batch carded
        ic.build(self.state, None, max_calls=5)  # run 2: attempt 2 stores "none"
        db = ic.connect(self.state)
        self.assertEqual(ic.pending(self.state, db), [])
        self.assertEqual(db.execute("SELECT card FROM cards WHERE id=0").fetchone()[0], 'none')
        self.assertNotEqual(db.execute("SELECT card FROM cards WHERE id=2").fetchone()[0], 'none')

    def test_vectors_follow_embedder_name(self):
        import array
        import memrove_v3_embed as emb
        real = emb.DaemonEmbedder
        self.addCleanup(setattr, emb, 'DaemonEmbedder', real)

        def fake(nm):
            class E:
                name = nm
                def __init__(self, *a, **k): pass
                def embed(self, texts): return [array.array('f', [1.0, 0.0, 0.0]) for _ in texts]
                embed_query = embed
            return E
        llm.RUNNER = self.runner()
        ic.build(self.state, None, max_calls=5)
        ic.embed_missing = self.real_embed
        emb.DaemonEmbedder = fake('A')
        self.assertEqual(ic.embed_missing(self.state, ic.connect(self.state), spawn=False), 3)
        db = ic.connect(self.state)
        self.assertEqual(len(ic._vectors(self.state, db, 'A')[0]), 3)
        self.assertEqual(ic._vectors(self.state, db, 'B')[0], [])  # other embedder: nothing to dot against
        emb.DaemonEmbedder = fake('B')
        self.assertEqual(ic.embed_missing(self.state, db, spawn=False), 3)  # stale vname -> re-embedded


if __name__ == '__main__':
    unittest.main()
