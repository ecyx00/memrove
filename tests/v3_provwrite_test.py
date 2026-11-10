#!/usr/bin/env python3
"""Write-time provenance: remember stamps authority, --from is verified, extractor items carry or lack a pointer. Fixture text is made up."""
import json
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
import memrove_v3_llm as llm  # noqa: E402
import memrove_v3_prov as prov  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402

SID = "dddddddd-0000-0000-0000-000000000001"
U = ["dddddddd-1111-0000-0000-00000000000%d" % n for n in range(3)]
TEXT = ["we keep the harbour ledger in plain files from now on", "unrelated chatter here", "ok go ahead with it"]


def tline(u, t):
    return json.dumps({"type": "user", "uuid": u, "message": {"content": t}}, separators=(",", ":")) + "\n"


class RememberTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name).resolve()
        proj = self.t / "home/.claude/projects/p"
        proj.mkdir(parents=True)
        (proj / f"{SID}.jsonl").write_text("".join(tline(u, t) for u, t in zip(U, TEXT)))
        self.state = self.t / "s"
        chatlog.append(self.state, [{"kind": "user", "text": t, "date": "2026-10-01T10:0%d:00+00:00" % n, "src": "claude", "k": "c:" + u} for n, (u, t) in enumerate(zip(U, TEXT))])
        self.home = self.t / "home"

    def test_last_owner_message_is_a_trigger_not_authority(self):
        f = prov.remember_fields(self.state, SID, home=self.home)
        self.assertEqual(f["trigger_msg"], f"transcript:{SID}#{U[2]}")
        self.assertEqual(f["prov_stage"], "trigger-only")
        self.assertTrue(not {"content_msg", "authority_msg"} & set(f))

    def test_only_a_verified_from_owner_message_gives_authority(self):
        i = next(iter(chatlog.messages(self.state)))["i"]
        f = prov.remember_fields(self.state, SID, i, "keep the harbour ledger in plain files", home=self.home)
        self.assertEqual((f["authority_msg"], f["authority_verdict"], f["trigger_msg"]), (i, "said", f"transcript:{SID}#{U[2]}"))
        au = "dddddddd-2222-0000-0000-000000000009"
        with open(self.t / f"home/.claude/projects/p/{SID}.jsonl", "a") as fh:
            fh.write(tline(au, "x"))
        chatlog.append(self.state, [{"kind": "agent", "text": "I suggest we keep the harbour ledger in plain files always", "date": "2026-10-01T10:09:00+00:00", "src": "claude", "k": "c:" + au}])
        a = [m["i"] for m in chatlog.messages(self.state) if m["kind"] == "agent"][0]
        g = prov.remember_fields(self.state, SID, a, "keep the harbour ledger in plain files always", home=self.home)
        self.assertEqual(g["content_msg"], a)
        self.assertNotIn("authority_msg", g)

    def test_from_verified_and_fake_span_rejected(self):
        i = next(iter(chatlog.messages(self.state)))["i"]
        f = prov.remember_fields(self.state, SID, i, "keep the harbour ledger in plain files", home=self.home)
        self.assertEqual((f["content_msg"], f["prov_stage"]), (i, "W"))
        with self.assertRaises(ValueError):
            prov.remember_fields(self.state, SID, i, "move the harbour ledger into a database", home=self.home)


class ExtractorTest(unittest.TestCase):
    def run_note(self, cites):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        vault, state = root / "v", root / "s"
        vault.mkdir()
        state.mkdir()
        (root / "r/.git").mkdir(parents=True)
        tp = root / "t.jsonl"
        tp.write_text("".join(tline(u, t) for u, t in zip(U, TEXT)))
        engine = SyncEngine(vault, state)
        reply = {"did": ["x"], "decisions": ["Keep the ledger in plain files"], "rejected": [], "open": [], "cites": cites,
                 "outcomes": [{"situation": "ledger storage choice", "approach": "plain files", "result": "worked", "why": "simple", "reuse": "applies to any small store",
                               "evidence": "harbour ledger in plain files", "topics": [], "entities": []}]}
        llm.RUNNER = lambda system, payload, vault=None: (json.dumps(reply), 0.0, None)
        self.addCleanup(setattr, llm, "RUNNER", llm._claude)
        os.environ.pop("MEMROVE_NO_LLM", None)
        self.addCleanup(os.environ.__setitem__, "MEMROVE_NO_LLM", "1")
        ev = {"event": "SessionEnd", "session": SID, "transcript_path": str(tp), "project": "repo", "at": 1}
        llm.session_notes(engine, vault, state, [ev])
        with engine.store._connect() as db:
            rs = [json.loads(r[0]) for r in db.execute("SELECT payload FROM records")]
        return {r["kind"]: r for r in rs if r.get("origin") == "llm" and r["kind"] in ("decision", "lesson")}

    def test_verified_item_carries_pointer(self):
        span = "the harbour ledger in plain files"
        rs = self.run_note([{"kind": "decision", "n": 0, "msg": 1, "span": span}, {"kind": "lesson", "n": 0, "msg": 1, "span": span}])
        for k in ("decision", "lesson"):
            self.assertEqual((rs[k]["prov_stage"], rs[k]["content_msg"], rs[k]["authority_msg"]), ("W", f"transcript:{SID}#{U[0]}", f"transcript:{SID}#{U[0]}"))
            s, e = rs[k]["content_span"]
            self.assertEqual(TEXT[0][s:e], span)

    def test_non_verbatim_span_written_unverified(self):
        rs = self.run_note([{"kind": "decision", "n": 0, "msg": 1, "span": "ledger lives in a database from today"},
                            {"kind": "lesson", "n": 0, "msg": 2, "span": "the harbour ledger in plain files"}])  # span is not in message 2
        for k in ("decision", "lesson"):
            self.assertEqual(rs[k]["prov_stage"], "unverified")
            self.assertNotIn("content_msg", rs[k])

    def test_malformed_cites_never_stop_the_note(self):
        span = "the harbour ledger in plain files"
        rs = self.run_note([{"kind": "decision", "n": [0], "msg": 1, "span": span}, {"kind": "decision", "n": {"a": 0}, "msg": 1, "span": span},
                            {"kind": "decision", "msg": 1, "span": span}, {"kind": "decision", "n": "9" * 5000, "msg": 1, "span": span}, {"kind": "lesson", "n": "0", "msg": 1, "span": span}])
        self.assertEqual(rs["decision"]["prov_stage"], "unverified")  # list/dict/missing n dropped, the note and records still written
        self.assertEqual(rs["lesson"]["prov_stage"], "W")  # a digit string is accepted


if __name__ == "__main__":
    unittest.main()
