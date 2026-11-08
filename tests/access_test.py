#!/usr/bin/env python3
"""cross-agent access (design 12.6): policy file, read scope by project, write deny, remote off, chat origin, malformed file."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
os.environ.setdefault("MEMROVE_NO_LLM", "1")
import memrove_core as core  # noqa: E402
import memrove_v3_access as access  # noqa: E402
import memrove_v3_chatlog as chatlog  # noqa: E402

DAY = "2026-10-07T10:00:00+00:00"


class AccessTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = self.t = Path(tmp.name).resolve()
        (t / "v").mkdir()
        self.state = t / "s"
        self.c = core.Core(t / "v", self.state)
        self.c.ingest({"text": "pelican alpha sqlite decision", "project": "P", "why": "w"})
        self.c.ingest({"text": "pelican beta sqlite decision", "project": "Q", "why": "w"})
        self.msgs = [m["i"] for m in chatlog.append(self.state, [
            {"kind": "user", "text": "pelican chat about sqlite in P", "date": DAY, "src": "claude", "proj": "P"},
            {"kind": "user", "text": "pelican chat about sqlite in Q", "date": DAY, "src": "codex", "proj": "Q"},
            {"kind": "user", "text": "pelican chat about sqlite nowhere", "date": DAY, "src": "codex"}])]

    def policy(self, obj):
        (self.state / "access.json").write_text(obj if isinstance(obj, str) else json.dumps(obj))

    def ids(self, **scope):
        return [h["id"] for h in self.c.recall("pelican sqlite", scope=scope, k=20)]

    def test_no_file_recall_unchanged(self):
        base = self.ids()
        self.assertEqual(len(base), 5)
        self.assertEqual(self.ids(agent="anyone"), base)

    def test_read_scope_drops_other_project_records_and_chat(self):
        self.policy({"agents": {"codex": {"read": ["P"]}}})
        hits = self.c.recall("pelican sqlite", scope={"agent": "codex"}, k=20)
        texts = " ".join(h["text"] for h in hits)
        self.assertIn("alpha", texts)
        self.assertIn("in P", texts)
        for bad in ("beta", "in Q", "nowhere"):
            self.assertNotIn(bad, texts)

    def test_write_false_is_denied_and_writes_nothing(self):
        self.policy({"agents": {"codex": {"write": False}}})
        with self.c.store._connect() as db:
            n = db.execute("SELECT count(*) FROM records").fetchone()[0]
        self.assertEqual(self.c.ingest("a new fact", scope={"agent": "codex"}), [{"id": None, "status": "denied"}])
        with self.c.store._connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM records").fetchone()[0], n)
        self.assertIsNotNone(self.c.ingest("a new fact", scope={"agent": "claude-code"})[0]["id"])

    def _relate(self, scope):
        calls = []
        self.c.relate_decider = lambda o, n: calls.append(1) or {"relation": "supports", "why": "f", "scores": {}}
        self.c.ingest({"text": "pelican sqlite older decision", "valid_from": "2026-01-01"})
        with mock.patch.dict(os.environ, {"MEMROVE_RELATE": "1", "MEMROVE_RELATE_REMOTE": "1"}):
            self.c.ingest({"text": "pelican sqlite newer decision confirmed", "valid_from": "2026-06-01"}, scope=scope)
        return len(calls)

    def test_remote_false_never_calls_decider(self):
        self.policy({"agents": {"codex": {"remote": False}}})
        self.assertEqual(self._relate({"agent": "codex"}), 0)

    def test_remote_default_still_calls_decider(self):
        self.policy({"agents": {"codex": {"remote": False}}})
        self.assertGreater(self._relate({"agent": "claude-code"}), 0)

    def test_chat_origin_is_agent_of_source(self):
        hits = {h["id"]: h for h in self.c.recall("pelican chat", k=20)}
        self.assertEqual(hits[self.msgs[1]]["origin"], "codex")
        self.assertEqual(hits[self.msgs[0]]["origin"], "claude-code")

    def test_session_scope_keeps_that_session_only(self):
        self.c.ingest({"text": "pelican sqlite session note", "project": "P"}, scope={"session": "s1"})
        hits = self.c.recall("pelican sqlite", scope={"session": "s1"}, k=20)
        self.assertEqual([h["text"].strip() for h in hits if h["arm"] == "record"], ["pelican sqlite session note"])

    def test_session_scope_keeps_chat_hits(self):
        self.c.ingest({"text": "pelican sqlite session note", "project": "P"}, scope={"session": "s1"})
        texts = " ".join(h["text"] for h in self.c.recall("pelican sqlite", scope={"session": "s1"}, k=20))
        self.assertIn("session note", texts)
        self.assertIn("chat about", texts)

    def test_no_agent_uses_policy_default(self):
        self.policy({"default": {"read": ["P"]}})
        texts = " ".join(h["text"] for h in self.c.recall("pelican sqlite", k=20))
        self.assertIn("alpha", texts)
        self.assertNotIn("beta", texts)
        self.assertNotIn("nowhere", texts)

    def test_recall_reads_each_day_file_once(self):
        chatlog.append(self.state, [{"kind": "user", "text": f"pelican sqlite older {n}", "date": DAY, "src": "codex", "proj": "P"} for n in range(4)], today="2026-10-01")
        real, calls, many = chatlog._read_all, [], chatlog.get_many
        def counted(*a):  # reads made by the per-hit fetch only (the search reads the log too)
            n = len(calls)
            try:
                return many(*a)
            finally:
                spent.append(len(calls) - n)
        spent = []
        with mock.patch.object(chatlog, "_read_all", lambda p, w=None: calls.append(p) or real(p, w)), mock.patch.object(chatlog, "get_many", counted):
            hits = self.c.recall("pelican sqlite", scope={"agent": "codex"}, agent="codex", k=20)
        self.assertGreaterEqual(len([h for h in hits if h["arm"] == "chat"]), 2)
        self.assertEqual(len(spent), 1)
        self.assertLessEqual(spent[0], len(chatlog._files(chatlog.root(self.state))))

    def test_malformed_keeps_writes_open_and_remote_off(self):
        self.policy("{not json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            pol = access.load(self.state)
        self.assertEqual(len(err.getvalue().strip().splitlines()), 1)
        self.assertTrue(access.can_write(pol, "codex"))
        self.assertTrue(access.can_read(pol, "codex", None))
        self.assertFalse(access.can_remote(pol, "claude-code"))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNotNone(self.c.ingest("still saved", scope={"agent": "codex"})[0]["id"])
            self.assertEqual(self._relate({"agent": "claude-code"}), 0)

    def test_project_less_readable_only_with_star(self):
        pol = {"default": {"read": ["P"]}}
        self.assertFalse(access.can_read(pol, "x", None))
        self.assertTrue(access.can_read({"default": {"read": ["*"]}}, "x", None))


if __name__ == "__main__":
    unittest.main()
