#!/usr/bin/env python3
"""Chat-log lane of the auto-recall hook: strength gate, 300-token budget, per-session seen set, `id|date|snippet` lines."""
import itertools
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402

LINE = re.compile(r"^\d+\|\d{4}-\d\d-\d\d\|\S.*$")
UID = itertools.count()
QUERY = "which restaurant did Marta recommend for the anniversary dinner in Lisbon"


def msg(n, text, kind="user"):
    return {"kind": kind, "text": text, "date": "2023-03-0%d" % (n % 9 + 1), "k": "c:uuid%d" % next(UID)}


class ChatLane(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="chatlane-")
        self.addCleanup(tmp.cleanup)
        self.state = tmp.name
        words = "amber basalt cobalt dune ember fjord glacier harbor iris jasper kelp lagoon".split()
        items = [msg(i, f"{words[i]} {words[i]} planning. Marta recommended restaurant {words[i].title()}Tasca for our anniversary dinner in Lisbon. "
                        f"Then we talked about {words[i - 1]} {words[i]} trains. " + "Padding sentence about nothing at all. " * 25) for i in range(12)]
        items += [msg(20 + i, f"unrelated chatter {i} about gardening and compost") for i in range(20 + cs.LANE_RECENT)]   # the 12 above are older than the newest LANE_RECENT
        chatlog.append(self.state, items, today="2023-03-09")

    def lenient(self):   # the 12-message fixture has no rare words, so its bm25 sits under the real gate
        self.addCleanup(setattr, cs, "LANE_MIN", cs.LANE_MIN)
        cs.LANE_MIN = 0.0
        self.addCleanup(setattr, cs, "LANE_MIN_RARE", cs.LANE_MIN_RARE)
        cs.LANE_MIN_RARE = 0

    def test_locked_imports_db_does_not_stall_hook(self):
        import sqlite3
        import time
        self.lenient()
        path = chatlog.root(self.state) / "imports.db"
        lock = sqlite3.connect(path, isolation_level=None)
        self.addCleanup(lock.close)
        lock.execute("CREATE TABLE x(a)")
        lock.execute("BEGIN EXCLUSIVE")
        t = time.monotonic()
        try:
            cs.lane_text(self.state, QUERY, "s1")
        except sqlite3.Error:
            pass   # the hook's caller catches this
        self.assertLess(time.monotonic() - t, 2)

    def test_budget_never_above_300(self):
        self.lenient()
        text = cs.lane_text(self.state, QUERY, "s1")
        self.assertTrue(text)
        self.assertLessEqual(cs._tokens(text), cs.LANE_TOKENS)
        self.assertLessEqual(cs._tokens(cs.LANE_HEADER + "\n" + "\n".join(l for _, l in cs.lane(self.state, QUERY, budget=30))), 30 + cs._tokens(cs.LANE_HEADER))

    def test_weak_match_gives_no_lines(self):
        self.assertEqual(cs.lane(self.state, "zebra quantum"), [])
        self.assertTrue(cs.lane_candidates(self.state, QUERY))
        self.assertEqual(cs.lane_text(self.state, QUERY, "s0"), "")   # matches, but under the real gate
        self.assertEqual(cs.lane_text(self.state, "zebra quantum", "s1"), "")

    def test_seen_id_is_not_reinjected(self):
        self.lenient()
        first = cs.lane_text(self.state, QUERY, "s1")
        ids1 = {l.split("|")[0] for l in first.splitlines()[1:]}
        second = cs.lane_text(self.state, QUERY, "s1")
        ids2 = {l.split("|")[0] for l in second.splitlines()[1:]}
        self.assertTrue(ids1)
        self.assertFalse(ids1 & ids2)
        self.assertEqual({l.split("|")[0] for l in cs.lane_text(self.state, QUERY, "s2").splitlines()[1:]}, ids1)   # another session sees them again

    def test_every_line_is_a_pointer(self):
        self.lenient()
        lines = cs.lane_text(self.state, QUERY, "s1").splitlines()
        self.assertGreater(len(lines), 1)
        self.assertEqual(lines[0], cs.LANE_HEADER)
        for l in lines[1:]:
            self.assertRegex(l, LINE)
            self.assertLessEqual(len(l.split("|", 2)[2]), cs.LANE_SNIP + 6)

    def test_in_context_messages_are_excluded(self):
        self.lenient()
        ids = {i for i, _ in cs.lane(self.state, QUERY)}
        self.assertTrue(ids)
        tr = Path(self.state) / "t.jsonl"   # transcript naming the uuid of every hit: they are in the agent's context
        tr.write_text(" ".join(str(chatlog.get(self.state, i).get("k", "x")) for i in ids))
        self.assertFalse({i for i, _ in cs.lane(self.state, QUERY, transcript=str(tr))} & ids)
        # no transcript: the newest LANE_RECENT messages count as in context
        self.assertEqual(cs._in_context(self.state, [0, chatlog.count(self.state) - 1], None), {chatlog.count(self.state) - 1})

    def test_only_the_transcript_tail_is_read(self):
        self.lenient()
        ids = sorted({i for i, _ in cs.lane(self.state, QUERY)})
        self.assertGreater(len(ids), 1)
        old, new = (str(chatlog.get(self.state, i).get("k", "x")) for i in ids[:2])
        tr = Path(self.state) / "big.jsonl"
        tr.write_text(old + " " + "." * 500 + " " + new)
        self.addCleanup(setattr, cs, "LANE_TAIL", cs.LANE_TAIL)
        cs.LANE_TAIL = len(new) + 20
        cs._read.cache_clear()
        self.assertEqual(cs._in_context(self.state, ids[:2], str(tr)), {ids[1]})   # old whole-file read would also return ids[0]

    def test_rare_stem_gate(self):
        self.lenient()
        cs.LANE_MIN_RARE = 1
        self.addCleanup(setattr, cs, "LANE_RARE_DF", cs.LANE_RARE_DF)
        cs.LANE_RARE_DF = 5
        q = "Marta recommended restaurant anniversary dinner Lisbon"   # every word is in 12 of 72 messages: none is rare at df <= 5
        self.assertEqual(cs.lane(self.state, q), [])
        chatlog.append(self.state, [msg(3, "Marta recommended restaurant Zanzibarello anniversary dinner Lisbon")] + [msg(5, "gardening compost") for _ in range(45)])
        self.assertTrue(cs.lane(self.state, q + " Zanzibarello"))

    def test_chatter_prompt_is_skipped(self):
        self.lenient()
        self.assertEqual(cs.lane(self.state, "ok Marta dinner"), [])   # 2 words of 4+ letters
        self.assertTrue(cs.lane(self.state, "ok Marta dinner Lisbon"))


if __name__ == "__main__":
    unittest.main()
