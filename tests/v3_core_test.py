#!/usr/bin/env python3
"""core facade: bi-temporal migration, as-of, update closes (never deletes), hard forget survives re-import, feedback row, recall == chatsearch.merged."""
import json
import os
import sqlite3
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
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402
import memrove_v3_graph as graph  # noqa: E402


class CoreTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = self.t = Path(tmp.name).resolve()
        (t / "v").mkdir()
        self.state = t / "s"
        self.c = core.Core(t / "v", self.state)

    def test_migration_idempotent_and_backfills(self):
        db = sqlite3.connect(self.state / "memory.sqlite3")
        db.execute("DROP TABLE graph_edges")
        db.execute("CREATE TABLE graph_edges(src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL, why TEXT NOT NULL, prov TEXT NOT NULL, ts TEXT NOT NULL, PRIMARY KEY(src,dst,type))")
        db.execute("INSERT INTO graph_edges VALUES ('a','b','relates','w','{}','2026-01-02')")
        graph.migrate(db)
        one = db.execute("SELECT * FROM graph_edges").fetchall()
        graph.migrate(db)
        self.assertEqual(one, db.execute("SELECT * FROM graph_edges").fetchall())
        self.assertEqual(one[0][6:], ("2026-01-02", None, "2026-01-02"))  # valid_from, valid_to, recorded_at... column order below
        db.close()

    def test_update_closes_old_and_as_of(self):
        old = self.c.ingest({"kind": "decision", "text": "use postgres for the store", "why": "w", "valid_from": "2026-01-01"})[0]["id"]
        res = self.c.update(old, {"text": "use sqlite for the store", "why": "w", "valid_from": "2026-06-01"})
        self.assertEqual(res["closed"], old)
        self.assertEqual([i for i, _ in self.c.as_of("2026-03-01")], [old])
        self.assertEqual([i for i, _ in self.c.as_of("2026-07-01")], [res["id"]])
        with self.c.store._connect() as db:   # nothing deleted
            self.assertEqual(db.execute("SELECT count(*) FROM records").fetchone()[0], 2)
        self.assertEqual(self.c.expand(old)["record"]["valid_to"], "2026-06-01")

    def _edit_and_sync(self, rid):
        with self.c.store._connect() as db:
            src = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])["source"]
        f = self.t / "v" / src
        f.write_text(f.read_text() + "\nedited by hand\n")
        self.c.sync.sync()

    def test_valid_to_survives_file_edit(self):
        old = self.c.ingest({"kind": "decision", "text": "use postgres for the store", "why": "w", "valid_from": "2026-01-01"})[0]["id"]
        new = self.c.update(old, {"text": "use sqlite for the store", "why": "w", "valid_from": "2026-06-01"})["id"]
        self._edit_and_sync(old)
        self.assertEqual([i for i, _ in self.c.as_of("2026-07-01")], [new])
        self.assertEqual(self.c.expand(old)["record"]["valid_to"], "2026-06-01")

    def test_open_record_stays_open_after_file_edit(self):
        rid = self.c.ingest({"kind": "decision", "text": "use postgres for the store", "why": "w", "valid_from": "2026-01-01"})[0]["id"]
        self._edit_and_sync(rid)
        self.assertEqual([i for i, _ in self.c.as_of("2026-07-01")], [rid])
        self.assertIsNone(self.c.expand(rid)["record"].get("valid_to"))

    def test_hard_forget_removes_and_stays_gone(self):
        rid = self.c.ingest({"kind": "lesson", "text": "secret lesson about tokens", "why": "w"})[0]["id"]
        other = self.c.ingest({"kind": "lesson", "text": "other lesson", "why": "w"})[0]["id"]
        with self.c.store._connect() as db:
            db.execute("INSERT INTO graph_edges(src,dst,type,why,prov,ts) VALUES (?,?,?,?,?,?)", (rid, other, "relates", "w", "{}", "2026-01-01"))
        v = sqlite3.connect(self.state / "vectors.sqlite3")
        v.execute("CREATE TABLE vec(h TEXT, model TEXT, v BLOB, PRIMARY KEY(h, model)) WITHOUT ROWID")
        v.execute("CREATE TABLE link(id TEXT, model TEXT, h TEXT, PRIMARY KEY(id, model)) WITHOUT ROWID")
        v.execute("INSERT INTO vec VALUES ('h1','m',x'00')")
        v.execute("INSERT INTO link VALUES (?,?,?)", (rid, "m", "h1"))
        v.commit()
        v.close()
        r = self.c.forget(rid, "hard")
        self.assertEqual(r["deleted"]["graph_edges"], 1)
        self.assertEqual(r["deleted"]["vectors"], 2)
        self.c.ingest({"kind": "lesson", "text": "secret lesson about tokens", "why": "w"})   # re-import of the same content
        with self.c.store._connect() as db:
            self.assertEqual([x[0] for x in db.execute("SELECT id FROM records")], [other])
            self.assertEqual(db.execute("SELECT count(*) FROM forgotten").fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT count(*) FROM events WHERE record_id=?", (rid,)).fetchone()[0], 0)

    def test_feedback_one_row(self):
        with mock.patch.dict(os.environ, {"DECIDE_HOME": str(self.t / "d")}):
            os.environ.pop("MEMROVE_NO_DECIDE_LOG", None)
            self.c.feedback("obs-1", "wrong", "stale fact")
            lines = (self.t / "d/log/decisions.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["answer"], "wrong")
            with self.assertRaises(ValueError):
                self.c.feedback("obs-1", "bogus")

    def test_recall_matches_chatsearch(self):
        chatlog.append(self.state, [{"kind": "user", "text": f"we talked about banana plan {i} and the kiwi budget", "date": "2026-10-07T10:00:00+00:00"} for i in range(5)])
        want = [r[0] for r in cs.merged(str(self.state), "banana plan", top=3)]
        got = self.c.recall("banana plan", k=3)
        self.assertEqual([h["id"] for h in got], want)
        self.assertEqual(got[0]["evidence"][0]["kind"], "msg")


if __name__ == "__main__":
    unittest.main()
