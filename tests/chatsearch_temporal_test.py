#!/usr/bin/env python3
"""R2 wired into chat search: a time expression in the query boosts messages dated inside its range (never filters), time words are not terms."""
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402

REF = date(2026, 11, 11)   # a Wednesday: "last week" = 2026-11-02..2026-11-08
BODY = "we ordered pizza with mushrooms and olives"


def build(state):
    old = {"kind": "user", "text": BODY, "date": "2026-06-01T10:00:00"}      # lower id: wins a plain tie
    new = {"kind": "user", "text": BODY, "date": "2026-11-04T10:00:00"}
    pad = [{"kind": "user", "text": f"unrelated note {i} about gardening", "date": "2026-10-01T10:00:00"} for i in range(6)]
    chatlog.append(state, [old, new] + pad, today="2026-11-11")


class ChatsearchTemporalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = self.tmp.name
        build(self.state)
        os.environ["MEMROVE_TEMPORAL"] = "1"

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("MEMROVE_TEMPORAL", None)

    def ids(self, query, **kw):
        return [r[0] for r in cs.merged(self.state, query, top=5, hybrid=False, **kw)]

    def test_in_range_message_ranks_first(self):
        os.environ["MEMROVE_TEMPORAL"] = "0"
        self.assertEqual(self.ids("pizza mushrooms last week", ref=REF)[:2], [0, 1])   # plain bm25: the tie keeps the old one first
        os.environ["MEMROVE_TEMPORAL"] = "1"
        self.assertEqual(self.ids("pizza mushrooms last week", ref=REF)[:2], [1, 0])

    def test_out_of_range_message_is_kept(self):
        self.assertIn(0, self.ids("pizza mushrooms last week", ref=REF))

    def test_time_words_are_not_terms(self):
        self.assertEqual(self.ids("pizza last week", ref=REF)[:1], [1])
        self.assertEqual(self.ids("last week", ref=REF)[:0], [])   # nothing but time words: falls back to the whole query, no crash

    def test_no_time_words_is_identical(self):
        q = "pizza mushrooms olives"
        a = cs.search(self.state, q)
        os.environ["MEMROVE_TEMPORAL"] = "0"
        self.assertEqual(a, cs.search(self.state, q))
        self.assertEqual(cs.merged(self.state, q, ref=REF), cs.merged(self.state, q, ref=None))


if __name__ == '__main__':
    unittest.main()
