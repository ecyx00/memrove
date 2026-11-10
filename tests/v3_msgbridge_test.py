#!/usr/bin/env python3
"""msgbridge: a record points at the owner message it came from; text and the older revision stay; edge `verified` is decided by code only.
Fixture text is made up."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
os.environ.setdefault("MEMROVE_NO_LLM", "1")
import memrove_core as core  # noqa: E402
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_graph as graph  # noqa: E402
import memrove_v3_msgbridge as bridge  # noqa: E402
import memrove_v3_recall as recall  # noqa: E402

SAID = "We keep the harbour ledger in plain files, never in a database."
U1, U2, U3 = ("aaaaaaaa-0000-0000-0000-00000000000%d" % n for n in (1, 2, 3))
PROV = {"by": "job:j5", "basis": "model", "at": "2026-10-01T00:00:00+00:00"}


class BridgeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = Path(tmp.name).resolve()
        (t / "v").mkdir()
        self.home, self.state = t / "home", t / "s"
        self.c = core.Core(t / "v", self.state)
        self.ctx = SimpleNamespace(state=self.state, engine=lambda: self.c.sync)
        chatlog.append(self.state, [
            {"kind": "agent", "text": f"As you said: {SAID}", "date": "2026-10-07T09:00:00+00:00", "src": "claude", "k": "c:a1"},
            {"kind": "user", "text": f"ok.  {SAID.upper()} thanks", "date": "2026-10-07T10:00:00+00:00", "src": "claude", "k": "c:" + U1},
            {"kind": "user", "text": "lighthouse keeper rota shifts tuesday evening swapped with harbour pilot", "date": "2026-10-08T10:00:00+00:00", "src": "claude", "k": "c:" + U2},
            {"kind": "user", "text": "something unrelated about gardening tools and weather", "date": "2026-10-09T10:00:00+00:00", "src": "claude", "k": "c:" + U3}])

    def rec(self, text, **kw):
        return self.c.ingest({"kind": "decision", "text": text, "why": "w", **kw})[0]["id"]

    def payload(self, rid):
        with self.c.store._connect() as db:
            return json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])

    def test_quote_stage_writes_message_id_keeps_text_and_old_revision(self):
        rid = self.rec(SAID, user=True)
        before = self.payload(rid)
        out = bridge.run(self.ctx, apply=True, llm=False, home=self.home)
        after = self.payload(rid)
        self.assertEqual((after["source_msg_id"], after["source_msg_stage"]), (1, "quote"))  # the owner's message, not the agent echo (id 0)
        start, end = after["source_quote_span"]
        self.assertEqual(chatlog.get(self.state, 1)["text"][start:end].casefold(), SAID.casefold())
        self.assertEqual(after["text"], before["text"])
        self.assertEqual(after["revision"], before["revision"] + 1)
        with self.c.store._connect() as db:
            old = [json.loads(r[0]) for r in db.execute("SELECT record FROM events WHERE record_id=? AND revision=?", (rid, before["revision"]))]
        self.assertTrue(old and "source_msg_id" not in old[0] and old[0]["text"] == before["text"])
        self.assertEqual(out["by_stage"], {"quote": 1})

    def test_provenance_apply_never_wipes_a_stamp_and_is_idempotent(self):
        from memrove_v3_projections import update_record
        rid = self.rec("keep the ledger in plain files")
        stamp = {"content_msg": f"transcript:s1#{U1}", "content_span": [0, 4], "authority_msg": f"transcript:s1#{U1}", "authority_verdict": "said",
                 "prov_stage": "W", "trigger_msg": f"transcript:s1#{U2}"}
        update_record(self.c.sync, rid, stamp)
        before = self.payload(rid)
        for _ in range(2):  # the model finds nothing both times
            bridge.run_provenance(self.ctx, apply=True, runner=lambda system, payload: (None, {}), home=self.home)
            after = self.payload(rid)
            self.assertEqual({k: after.get(k) for k in stamp}, stamp)
            self.assertEqual(after["revision"], before["revision"])

    def test_provenance_rerun_that_finds_another_pointer_keeps_the_stamp(self):
        from memrove_v3_projections import update_record
        rid = self.rec(SAID, user=True)
        stamp = {"content_msg": 3, "content_span": [0, 9], "authority_msg": 3, "authority_verdict": "said", "prov_stage": "W"}
        update_record(self.c.sync, rid, stamp)
        before = self.payload(rid)
        other = {"content_msg": 1, "content_span": [0, 5], "authority_msg": 1, "authority_verdict": "said", "prov_stage": "C+said", "source": "claude"}
        with unittest.mock.patch.object(bridge, "provenance", lambda *a, **k: ({rid: other}, {})):  # the rerun finds message 1, not 3
            bridge.run_provenance(self.ctx, apply=True, runner=lambda system, payload: (None, {}), home=self.home)
        after = self.payload(rid)
        self.assertEqual({k: after.get(k) for k in stamp}, stamp)
        self.assertEqual(after["revision"], before["revision"])

    def test_dry_writes_nothing(self):
        rid = self.rec(SAID, user=True)
        before = self.payload(rid)
        out = bridge.run(self.ctx, apply=False, llm=False, home=self.home)
        self.assertEqual(self.payload(rid), before)
        self.assertEqual((out["resolved"], out["records_written"]), (1, 0))

    def test_session_stage_uses_the_transcript_session(self):
        sid = "11111111-2222-3333-4444-555555555555"
        folder = self.home / ".claude/projects/p"
        folder.mkdir(parents=True)
        (folder / f"{sid}.jsonl").write_text(f'{{"uuid":"{U2}"}}\n{{"uuid":"{U1}"}}\n')
        rid = self.rec("lighthouse keeper rota shifts tuesday evening swapped with the harbour pilot")
        other = self.rec("something unrelated about gardening tools and weather")  # u3 is not in the session file
        for r in (rid, other):
            self.c._patch(r, "test", session=sid)
        bridge.run(self.ctx, apply=True, llm=False, home=self.home)
        self.assertEqual((self.payload(rid)["source_msg_id"], self.payload(rid)["source_msg_stage"]), (2, "session"))
        self.assertNotIn("source_msg_id", self.payload(other))

    def test_reresolved_record_drops_the_old_span(self):
        rid = self.rec(SAID, user=True)
        bridge.run(self.ctx, apply=True, llm=False, home=self.home)
        self.assertIn("source_quote_span", self.payload(rid))
        with self.c.store._connect() as db:
            p = self.payload(rid)
            p["source_msg_id"], p["source_msg_stage"] = 3, "session"   # moved to a message with no quote span
            db.execute("UPDATE records SET payload=? WHERE id=?", (json.dumps(p), rid))
        real = bridge.resolve
        bridge.resolve = lambda m, a: ({rid: dict(msg=3, span=None, stage="session", by="code")}, {})
        try:
            bridge.run(self.ctx, apply=True, llm=False, home=self.home)
        finally:
            bridge.resolve = real
        self.assertIsNone(self.payload(rid)["source_quote_span"])

    def test_session_stage_does_not_verify_edges(self):
        a = self.rec("lighthouse keeper rota shifts tuesday evening swapped with the harbour pilot")
        with self.c.store._connect() as db:
            graph.add_concept(db, "Ledger rules", "w", PROV)
            graph.add_edge(db, a, "concept:ledger-rules", "part_of", "w", PROV)
        found = {a: dict(msg=2, span=None, stage="session", by="code")}
        st = bridge.edge_status([(a, "concept:ledger-rules", "part_of")], {"concept:ledger-rules"}, found, None)
        self.assertEqual(list(st.values()), ["candidate"])

    def test_haiku_pick_must_be_a_shown_candidate_and_none_leaves_the_record(self):
        a = self.rec("harbour pilot duty changes")
        b = self.rec("gardening weather question")
        sent = []

        def runner(system, payload):
            sent.append(payload)
            ids = [i["id"] for i in json.loads(payload)["items"]]
            return {"items": [{"id": ids[0], "pick": 0}, {"id": ids[1], "pick": 99}]}, {"input_tokens": 5, "output_tokens": 2}
        for rid in (a, b):
            self.c._patch(rid, "test", valid_from="2026-10-08")
        out = bridge.run(self.ctx, apply=False, llm=True, runner=runner, home=self.home)
        self.assertEqual(out["haiku"]["calls"], 1)
        self.assertEqual(out["haiku"]["input_tokens"], 5)
        self.assertEqual(out["by_stage"].get("haiku", 0) + out["unresolved_with_candidates"], 2)
        self.assertLessEqual(out["by_stage"].get("haiku", 0), 1)  # the out-of-range pick was dropped

    def test_edge_status_is_decided_by_code(self):
        old, new = self.rec(SAID, user=True), self.rec("lighthouse keeper rota shifts tuesday evening swapped with harbour pilot", user=True)
        with self.c.store._connect() as db:
            graph.add_concept(db, "Ledger rules", "w", PROV)
            graph.add_edge(db, new, old, "supersedes", "newer", PROV)   # new (msg 2) is later than old (msg 1): passes
            graph.add_edge(db, old, new, "supersedes", "wrong way", PROV)
            graph.add_edge(db, old, "concept:ledger-rules", "part_of", "w", PROV)
            graph.add_edge(db, old, "concept:gone-away", "part_of", "w", PROV)   # dangling: repaired by name, then verifiable
            graph.add_edge(db, old, new, "relates", "w", PROV)
        out = bridge.run(self.ctx, apply=True, llm=False, home=self.home)
        with self.c.store._connect() as db:
            st = {(s, d, t): x for s, d, t, x in db.execute("SELECT src,dst,type,status FROM graph_edges")}
            node = db.execute("SELECT kind,name,why FROM graph_nodes WHERE id='concept:gone-away'").fetchone()
        self.assertEqual(st[(new, old, "supersedes")], "verified")
        self.assertEqual(st[(old, new, "supersedes")], "candidate")
        self.assertEqual(st[(old, "concept:ledger-rules", "part_of")], "verified")
        self.assertEqual(st[(old, "concept:gone-away", "part_of")], "candidate")  # a recreated concept proves nothing
        bridge.run(self.ctx, apply=True, llm=False, home=self.home)  # second run: the node now exists but is still the restored one
        with self.c.store._connect() as db:
            self.assertEqual(db.execute("SELECT status FROM graph_edges WHERE dst='concept:gone-away'").fetchone()[0], "candidate")
        self.assertEqual(st[(old, new, "relates")], "candidate")
        self.assertEqual(node, ("concept", "gone away", "restored"))
        self.assertEqual(out["edges"]["verified_by_type"]["supersedes"], [1, 2])

    def test_search_and_show_expose_the_message_id(self):
        rid = self.rec(SAID, user=True)
        bridge.run(self.ctx, apply=True, llm=False, home=self.home)
        hit = recall.search(self.c.store, "harbour ledger plain files")[0]
        self.assertEqual((hit["id"], hit["source_msg_id"]), (rid, 1))
        self.assertEqual(recall.show(self.c.store, rid)["source_msg_id"], 1)


if __name__ == "__main__":
    unittest.main()
