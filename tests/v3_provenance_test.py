#!/usr/bin/env python3
"""Provenance pilot verifier: a pointer is stored only when code can check it. Fixture text is made up."""
import hashlib
import json
import sqlite3
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
os.environ.setdefault("MEMROVE_NO_LLM", "1")
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_msgbridge as bridge  # noqa: E402

MSG = {
    1: {"kind": "agent", "text": "I propose the  harbour ledger stays in plain files from now on."},
    2: {"kind": "user", "text": "ok go"},
    3: {"kind": "agent", "text": "Done, the ledger is in plain files."},
    4: {"kind": "user", "text": "good"},
    9: {"kind": "user", "text": "message that belongs to another session entirely"},
}
IDS = {1, 2, 3, 4}
M = ["cccccccc-0000-0000-0000-00000000000%d" % n for n in range(4)]
SPAN = "the harbour ledger stays in plain files"


class VerifyTest(unittest.TestCase):
    def test_good_pointers_and_whitespace_case(self):
        v = bridge.verify(MSG, IDS, 1, SPAN.upper(), 2)
        self.assertEqual((v["content_msg"], v["authority_msg"]), (1, 2))
        s, e = v["content_span"]
        self.assertEqual(" ".join(MSG[1]["text"][s:e].split()).lower(), SPAN)

    def test_fake_span_rejected(self):
        self.assertEqual(bridge.verify(MSG, IDS, 1, "the harbour ledger moves into a database now", 2), {})

    def test_short_span_rejected(self):
        self.assertEqual(bridge.verify(MSG, IDS, 1, "plain files from now", 2), {})

    def test_authority_before_content_rejected(self):
        self.assertNotIn("authority_msg", bridge.verify(MSG, IDS, 3, "the ledger is in plain files", 2))

    def test_other_session_and_agent_authority_rejected(self):
        self.assertNotIn("authority_msg", bridge.verify(MSG, IDS, 1, SPAN, 9))
        self.assertNotIn("authority_msg", bridge.verify(MSG, IDS, 1, SPAN, 3))  # agent message is no authority
        self.assertEqual(bridge.verify(MSG, IDS, 9, SPAN, 4), {})  # content from another session

    def test_stage_a_last_owner_message_before_recorded_at(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t).resolve()
            sid = "bbbbbbbb-0000-0000-0000-000000000001"
            proj = t / "home/.claude/projects/p"
            proj.mkdir(parents=True)
            (proj / f"{sid}.jsonl").write_text("".join(json.dumps({"uuid": u}, separators=(",", ":")) + "\n" for u in M))
            rows = [("user", M[0], "2026-10-01T10:00:00+00:00"), ("agent", M[1], "2026-10-01T10:01:00+00:00"),
                    ("user", M[2], "2026-10-01T10:02:00+00:00"), ("user", M[3], "2026-10-01T10:09:00+00:00")]
            chatlog.append(t / "s", [{"kind": k, "text": f"text {u}", "date": d, "src": "claude", "k": "c:" + u} for k, u, d in rows])
            chat = bridge.Chat(t / "s", t / "home")
            for session in (sid, hashlib.sha256(sid.encode()).hexdigest()[:24]):
                self.assertEqual(chat.session({"session": session}), sid)
            rec = {"session": hashlib.sha256(sid.encode()).hexdigest()[:24], "recorded_at": "2026-10-01T10:05:00+00:00"}
            ids = chat.ids(sid)
            out, _ = bridge.provenance(chat, [("obs-x", rec)])
            self.assertEqual(out["obs-x"]["trigger_msg"], ids[2])  # m3: m2 is the agent, m4 is later
            self.assertNotIn("authority_msg", out["obs-x"])  # a position guess is never authority


def _fake_runner(span, answers=None):
    """Haiku stand-in. Content call: the message whose line holds `span`. Authority call: answers {rid: (turn index, verdict)} (default none)."""
    def run(system, payload):
        if system == bridge.ASYSTEM:
            items = json.loads(payload)["items"]
            return {"items": [{"id": it["id"], "pick": (answers or {}).get(it["id"], (None, "none"))[0], "verdict": (answers or {}).get(it["id"], (None, "none"))[1]} for it in items]}, {"input_tokens": 1}
        recs = json.loads(payload.split("\n\nRECORDS:\n")[1])["records"]
        ids = {ln.split("]")[0][1:]: ln for ln in payload.split("MESSAGES:\n")[1].split("\n\nRECORDS:")[0].splitlines()}
        cid = next(int(i) for i, ln in ids.items() if span in ln)
        return {"items": [{"id": r["id"], "content_msg": cid, "span": span} for r in recs]}, {"input_tokens": 1}
    return run


def _home(t, sid, rows):
    proj = t / "home/.claude/projects/p"
    proj.mkdir(parents=True)
    (proj / f"{sid}.jsonl").write_text("".join(json.dumps({"type": k, "uuid": u, "timestamp": d, "message": {"content": c}}) + "\n" for k, u, d, c in rows))
    (t / "s").mkdir()
    return bridge.Chat(t / "s", t / "home")


T = lambda text: [{"type": "text", "text": text}]


class FallbackTest(unittest.TestCase):
    SID = "dddddddd-0000-0000-0000-000000000001"
    SPAN = "the harbour ledger stays in plain files"
    ROWS = [("assistant", "u1", "2026-10-01T10:00:00Z", T("I propose the harbour ledger stays in plain files.")),
            ("user", "u2", "2026-10-01T10:01:00Z", [{"type": "tool_result", "content": "the harbour ledger stays in plain files"}]),
            ("user", "u3", "2026-10-01T10:02:00Z", "<system-reminder>hook says do something else</system-reminder>"),
            ("user", "u4", "2026-10-01T10:03:00Z", "/reload-plugins now please"),
            ("user", "u5", "2026-10-01T10:04:00Z", "Here is a note offered by a side agent: the quiet lantern rule holds for every deploy"),
            ("user", "u6", "2026-10-01T10:05:00Z", "ok go ahead with that")]

    def run_prov(self, recs, answers=None, span=None, rows=None):
        with tempfile.TemporaryDirectory() as t:
            chat = _home(Path(t).resolve(), self.SID, rows or self.ROWS)
            out, usage = bridge.provenance(chat, recs, {r for r, _ in recs}, _fake_runner(span or self.SPAN, answers))
            return chat, out, usage

    def test_agent_content_authority_is_the_approving_owner_turn_not_the_next_one(self):
        # next owner turns after u1 (injected u3, u5 skipped): u4, u6 -> index 1 = u6
        chat, out, _ = self.run_prov([("p-x", {"session": self.SID, "kind": "decision"})], {"p-x": (1, "approves")})
        o = out["p-x"]
        self.assertNotIn("authority_msg", o)  # agent-spoken content: a Haiku pick is a candidate, never authority
        self.assertEqual(o["authority_candidate"], f"transcript:{self.SID}#u6")
        self.assertEqual((o["authority_verdict"], o["prov_stage"], o["content_msg"]), ("approves", "C+approves", f"transcript:{self.SID}#u1"))

    def test_pick_outside_shown_turns_and_none_give_no_authority(self):
        _, out, _ = self.run_prov([("p-x", {"session": self.SID}), ("p-y", {"session": self.SID})], {"p-x": (7, "approves"), "p-y": (0, "none")})
        for k in out:
            self.assertNotIn("authority_msg", out[k])

    def test_rejected_is_stored_without_authority(self):
        _, out, _ = self.run_prov([("p-x", {"session": self.SID, "kind": "decision"})], {"p-x": (0, "rejects")})
        o = out["p-x"]
        self.assertEqual((o["rejection_candidate"], "authority_msg" in o, "authority_candidate" in o, o["prov_stage"]), (f"transcript:{self.SID}#u4", False, False, "C+rejected"))
        self.assertNotIn("owner_rejected", o)

    def test_lesson_note_need_no_authority_and_rejected_kind_rejection_is_supporting(self):
        recs = [("l-x", {"session": self.SID, "kind": "lesson"}), ("obs-n", {"session": self.SID, "kind": "note"}), ("p-r", {"session": self.SID, "kind": "rejected"})]
        _, out, _ = self.run_prov(recs, {"l-x": (1, "approves"), "obs-n": (0, "rejects"), "p-r": (0, "rejects")})
        for k in ("l-x", "obs-n"):
            self.assertEqual(out[k]["authority"], "not_required")
            self.assertTrue(not {"authority_msg", "authority_candidate"} & set(out[k]))
        self.assertIn("rejection_candidate", out["obs-n"])
        self.assertEqual((out["p-r"]["authority_verdict"], "rejection_candidate" in out["p-r"], "authority_msg" in out["p-r"]), ("rejects_supports", False, False))
        self.assertIn("authority_candidate", out["p-r"])

    def test_candidates_have_no_readers_and_apply_adds_edge_status_column(self):
        # no code path may read the unproven picks (ranking, demotion, closing, edge status): only the pass that writes them mentions them
        names = ("authority_candidate", "rejection_candidate")
        for f in list((ROOT / "template/.claude/scripts").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
            if f.name != "memrove_v3_msgbridge.py":
                self.assertFalse(any(n in f.read_text() for n in names), f.name)
        import memrove_v3_graph as graph
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE graph_edges(src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL, why TEXT NOT NULL, prov TEXT NOT NULL, ts TEXT NOT NULL, valid_from TEXT, valid_to TEXT, recorded_at TEXT, PRIMARY KEY(src,dst,type))")  # live shape: no status column
        for _ in range(2):  # idempotent
            graph.ensure(db)
        self.assertIn("status", [r[1] for r in db.execute("PRAGMA table_info(graph_edges)")])

    def test_injected_lines_are_never_owner_turns_and_content_there_is_relayed(self):
        chat, out, _ = self.run_prov([("q-x", {"session": self.SID, "text": "x"})], span="the quiet lantern rule holds for every deploy")
        self.assertEqual((out["q-x"]["prov_stage"], "authority_msg" in out["q-x"]), ("relayed", False))
        self.assertEqual([chat.msg[i]["kind"] for i in chat.ids(self.SID)], ["agent", "meta", "user", "meta", "user"])

    def test_owner_content_is_its_own_authority_without_a_call(self):
        chat, out, usage = self.run_prov([("p-x", {"session": self.SID})], span="ok go ahead with that", rows=self.ROWS)
        self.assertEqual((out["p-x"]["authority_verdict"], out["p-x"]["authority_msg"], out["p-x"]["prov_stage"]), ("said", out["p-x"]["content_msg"], "C+said"))
        self.assertEqual(usage["calls"], 1)  # the content call only

    def test_quote_wins_over_haiku_for_q(self):
        rows = [("user", "u1", "2026-10-01T10:00:00Z", "pasted earlier: the harbour ledger stays in plain files and a lot of other words around it"),
                ("user", "u2", "2026-10-01T10:01:00Z", "the harbour ledger stays in plain files")]
        chat, out, usage = self.run_prov([("q-x", {"session": self.SID, "text": "The harbour ledger stays in plain files"})], rows=rows)
        self.assertEqual((out["q-x"]["content_msg"], out["q-x"]["prov_stage"], usage.get("calls", 0)), (f"transcript:{self.SID}#u2", "quote+said", 0))

    def test_big_session_records_are_batched(self):
        sizes = []
        inner = _fake_runner(self.SPAN)
        def run(system, payload):
            if system == bridge.PSYSTEM:
                sizes.append(len(json.loads(payload.split("\n\nRECORDS:\n")[1])["records"]))
            return inner(system, payload)
        with tempfile.TemporaryDirectory() as t:
            chat = _home(Path(t).resolve(), self.SID, self.ROWS)
            recs = [(f"p-{n}", {"session": self.SID}) for n in range(bridge.REC_BATCH * 2 + 1)]
            out, _ = bridge.provenance(chat, recs, {r for r, _ in recs}, run)
        self.assertEqual(sizes, [bridge.REC_BATCH, bridge.REC_BATCH, 1])
        self.assertEqual(sum("content_msg" in o for o in out.values()), len(recs))

    def test_source_lost_and_obs_trigger_not_overwriting_authority(self):
        with tempfile.TemporaryDirectory() as t:
            chat = _home(Path(t).resolve(), self.SID, self.ROWS)
            rec = {"session": self.SID, "recorded_at": "2026-10-01T10:04:30Z"}
            out, _ = bridge.provenance(chat, [("p-x", {"session": "no-such-session"}), ("obs-y", rec)])
            self.assertEqual(out["p-x"]["prov_stage"], "source_lost")
            self.assertEqual((out["obs-y"]["trigger_msg"], "authority_msg" in out["obs-y"], out["obs-y"]["prov_stage"]), (f"transcript:{self.SID}#u4", False, "none"))

    def test_seed_is_reverified_and_a_bad_seed_falls_back_to_haiku(self):
        with tempfile.TemporaryDirectory() as t:
            chat = _home(Path(t).resolve(), self.SID, self.ROWS)
            text = chat.msg[chat.ids(self.SID)[0]]["text"]
            s = text.index("the harbour")
            ptr = f"transcript:{self.SID}#u1"
            recs = [("p-a", {"session": self.SID}), ("p-b", {"session": self.SID})]
            out, usage = bridge.provenance(chat, recs, {"p-a", "p-b"}, _fake_runner(self.SPAN), seed={"p-a": (ptr, s, s + len(self.SPAN)), "p-b": (ptr, 0, 3)})
            self.assertEqual((out["p-a"]["content_msg"], out["p-b"]["content_msg"]), (ptr, ptr))
            self.assertEqual(usage["calls"], 2)  # p-b content call + the authority call for both


class ImportsTest(unittest.TestCase):
    def test_imports_db_with_naive_dates_and_owner_filter(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t).resolve()
            (t / "s/chatlog").mkdir(parents=True)
            db = sqlite3.connect(t / "s/chatlog/imports.db")
            db.execute("create table x(id TEXT PRIMARY KEY, src, session, project, date, kind, text, j2 INTEGER DEFAULT 0)")
            db.executemany("insert into x(id,src,session,date,kind,text) values (?,?,?,?,?,?)", [
                ("m1", "chatgpt", "imp-1", "2026-10-01T10:00:00", "agent", "I propose the harbour ledger stays in plain files."),
                ("m2", "chatgpt", "imp-1", "2026-10-01T10:01:00+00:00", "user", "yes do that please"),
                ("m3", "chatgpt", "imp-1", "2026-10-01T10:02:00", "user", "This session is being continued from a previous conversation")])
            db.commit()
            db.close()
            chat = bridge.Chat(t / "s", t / "home")
            rec = {"session": "imp-1", "recorded_at": "2026-10-01T10:30:00+00:00"}
            rec["kind"] = "decision"
            out, _ = bridge.provenance(chat, [("obs-i", rec)], {"obs-i"}, _fake_runner("the harbour ledger stays in plain files", {"obs-i": (0, "acts_on")}))
            o = out["obs-i"]
            self.assertEqual((o["source"], o["content_msg"], o["authority_candidate"], o["trigger_msg"]), ("imports", "imports:imp-1#m1", "imports:imp-1#m2", "imports:imp-1#m2"))
            self.assertNotIn("authority_msg", o)
            self.assertEqual(chat.msg[chat.ids("imp-1")[2]]["kind"], "meta")


if __name__ == "__main__":
    unittest.main()
