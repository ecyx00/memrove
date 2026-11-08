#!/usr/bin/env python3
"""core wiring of relate: opt-in via MEMROVE_RELATE=1, fake decider, failure never breaks ingest, supersedes only proposed."""
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

OLD = {"kind": "decision", "text": "use sqlite for the memory store backend", "why": "w", "valid_from": "2026-01-01"}
NEW = {"kind": "decision", "text": "sqlite memory store backend is confirmed by the benchmark", "why": "w", "valid_from": "2026-06-01"}


class CoreRelateTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = Path(tmp.name).resolve()
        (t / "v").mkdir()
        self.c = core.Core(t / "v", t / "s")
        self.calls = []
        self.c.relate_decider = self.decide
        self.rel = "supports"
        self.old = self.c.ingest(OLD)[0]["id"]

    def decide(self, old, new):
        self.calls.append((old["id"], new["id"]))
        if self.rel == "boom":
            raise RuntimeError("down")
        return {"relation": self.rel, "why": "fake", "scores": {}}

    def edges(self):
        with self.c.store._connect() as db:
            return db.execute("SELECT src,dst,type FROM graph_edges").fetchall()

    def test_on_writes_edge(self):
        with mock.patch.dict(os.environ, {"MEMROVE_RELATE": "1"}):
            new = self.c.ingest(NEW)[0]["id"]
        self.assertEqual(self.edges(), [(new, self.old, "supports")])

    def test_no_write_lock_held_while_deciding(self):
        import sqlite3
        self.c.ingest({**OLD, "text": "sqlite memory store backend for the old design", "valid_from": "2026-02-01"})
        locked = []

        def probe(old, new):  # a second writer must get in while a decider (network in real life) runs
            db = sqlite3.connect(str(self.c.store.database), timeout=0.1)
            try:
                db.execute("BEGIN IMMEDIATE")
                db.rollback()
                locked.append(False)
            except sqlite3.OperationalError:
                locked.append(True)
            finally:
                db.close()
            return {"relation": "supports", "why": "fake", "scores": {}}

        self.c.relate_decider = probe
        with mock.patch.dict(os.environ, {"MEMROVE_RELATE": "1"}):
            self.c.ingest(NEW)
        self.assertEqual(len(locked), 2)
        self.assertEqual(locked, [False, False])
        self.assertEqual(len(self.edges()), 2)

    def test_off_no_edge_no_call(self):
        with mock.patch.dict(os.environ, {"MEMROVE_RELATE": ""}):
            self.c.ingest(NEW)
        self.assertEqual((self.edges(), self.calls), ([], []))

    def test_decider_error_keeps_ingest(self):
        self.rel = "boom"
        with mock.patch.dict(os.environ, {"MEMROVE_RELATE": "1"}):
            r = self.c.ingest(NEW)[0]
        self.assertEqual(r["status"], "created")
        self.assertTrue(self.calls)
        self.assertEqual(self.edges(), [])

    def test_supersedes_is_proposed_only(self):
        self.rel = "supersedes"
        with mock.patch.dict(os.environ, {"MEMROVE_RELATE": "1"}):
            r = self.c.ingest(NEW)[0]
        self.assertEqual(self.edges(), [])
        self.assertEqual([(p["src"], p["dst"]) for p in r["proposals"]], [(r["id"], self.old)])

    def _ingest_on(self, item=NEW, **env):
        with mock.patch.dict(os.environ, {"MEMROVE_RELATE": "1", **env}):
            return self.c.ingest(item)[0]

    def test_tombstoned_and_closed_never_reach_decider(self):
        closed = self.c.ingest({**OLD, "text": "use sqlite for the memory store backend always"})[0]["id"]
        with self.c.store._connect() as db:
            p = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (closed,)).fetchone()[0])
            db.execute("UPDATE records SET payload=? WHERE id=?", (json.dumps({**p, "valid_to": "2026-02-01"}), closed))
        self.c.forget(self.old)
        self._ingest_on()
        self.assertEqual(self.calls, [])

    def test_remote_decider_needs_remote_flag(self):
        self.c.relate_decider = "jev"
        spy = mock.Mock(return_value={"relation": "supports", "why": "x", "scores": {}})
        with mock.patch.dict(core.relate.DECIDERS, {"jev": (True, spy)}):
            self._ingest_on()
            self.assertFalse(spy.called)
            self._ingest_on({**NEW, "text": NEW["text"] + " again"}, MEMROVE_RELATE_REMOTE="1")
            self.assertTrue(spy.called)

    def test_email_pair_is_skipped_for_remote(self):
        self.c.ingest({**OLD, "text": "use sqlite for the memory store backend, ask bob@example.com"})
        self.c.forget(self.old)
        self._ingest_on(MEMROVE_RELATE_REMOTE="1")
        self.assertEqual(self.calls, [])

    def test_supersedes_proposal_is_persisted(self):
        self.rel = "supersedes"
        r = self._ingest_on()
        line = json.loads((Path(self.c.state) / "reflect-proposals.jsonl").read_text().splitlines()[-1])
        self.assertEqual((line["src"], line["dst"], line["by"]), (r["id"], self.old, "relate"))


if __name__ == "__main__":
    unittest.main()
