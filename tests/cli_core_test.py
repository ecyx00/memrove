#!/usr/bin/env python3
"""CLI wired to Core: remember marks evidence and honours the write policy, search --json honours the read policy, reflect --dry writes nothing."""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
os.environ["MEMROVE_NO_LLM"] = "1"
os.environ.pop("MEMROVE_AGENT", None)
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402
import memrove_v3_hook as hook  # noqa: E402

SAID = "Always keep the memrove store in SQLite, never postgres."
QUOTED = f'The owner said "{SAID}" today'  # evidence.verify looks for a quoted span


class CliCoreTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = self.t = Path(tmp.name).resolve()
        (t / "v").mkdir()
        self.state = t / "s"
        chatlog.append(self.state, [{"kind": "user", "text": f"ok. {SAID} thanks", "date": "2026-10-07T10:00:00+00:00", "src": "claude", "k": "c:1"}])

    def run_cli(self, *argv, agent=()):
        p = subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(self.t / "v"), "--state", str(self.state), *agent, *argv],
                           capture_output=True, text=True, env=os.environ)
        return p.returncode, p.stdout, p.stderr

    def payloads(self):
        if not (self.state / "memory.sqlite3").exists():
            return []
        with sqlite3.connect(self.state / "memory.sqlite3") as db:
            return [json.loads(p) for (p,) in db.execute("SELECT payload FROM records")]

    def policy(self, pol):
        (self.state / "access.json").write_text(json.dumps(pol))

    def remember(self, text, agent=()):
        return self.run_cli("remember", text, "--kind", "decision", "--why", "w", "--project", "alpha", "--cwd", str(self.t), agent=agent)

    def test_remember_marks_evidence_and_agent(self):
        code, out, _ = self.remember(QUOTED)
        self.assertEqual(code, 0)
        r = self.payloads()[0]
        self.assertEqual((r["id"], r["evidence_state"], r["agent"]), (out.strip(), "verified", "claude-code"))

    def test_remember_honours_write_deny(self):
        self.policy({"agents": {"codex": {"write": False}}})
        code, _, err = self.remember(QUOTED, ("--agent", "codex"))
        self.assertEqual(code, 1)
        self.assertIn("PermissionError", err)
        self.assertEqual(self.payloads(), [])
        self.assertEqual(self.remember(QUOTED)[0], 0)  # another agent still writes

    def test_search_text_is_the_chat_search_output_and_json_obeys_read_policy(self):
        import memrove_v3_chatsearch as cs
        self.remember(QUOTED)
        code, out, _ = self.run_cli("search", "SQLite postgres")
        self.assertEqual((code, out), (0, cs.search(self.state, "SQLite postgres") + "\n"))
        arms = lambda *a: {h["arm"] for h in json.loads(self.run_cli("search", "SQLite postgres", "--json", agent=a)[1])}
        self.assertIn("record", arms())
        self.policy({"agents": {"codex": {"read": ["beta"]}}})
        self.assertIn("record", arms())
        self.assertNotIn("record", arms("--agent", "codex"))

    def test_reflect_without_apply_writes_nothing(self):
        self.remember(QUOTED)
        self.assertEqual(self.run_cli("reflect", "--apply", "--dry")[0], 1)

        def snap():
            with sqlite3.connect(self.state / "memory.sqlite3") as db:
                return db.execute("SELECT count(*) FROM events").fetchone(), self.payloads(), (self.state / "reflect-proposals.jsonl").exists()
        before = snap()
        code, out, _ = self.run_cli("reflect", "--day", "2026-10-07")
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["dry"])
        self.assertEqual(snap(), before)

    def test_reflect_apply_closes_a_stale_record_and_default_does_not(self):
        import importlib.util
        import io
        from contextlib import redirect_stdout
        from unittest import mock
        import memrove_core as core
        import memrove_v3_llm as llm
        day = "2026-10-07"
        ids = [m["i"] for m in chatlog.append(self.state, [
            {"kind": "agent", "text": "ok, postgres then", "date": day + "T10:30:00+00:00", "src": "claude", "k": "c:2"},
            {"kind": "user", "text": "no, drop postgres, sqlite is final", "date": day + "T11:00:00+00:00", "src": "claude", "k": "c:3"}])]
        c = core.Core(self.t / "v", self.state)
        rid = c.ingest({"kind": "decision", "text": "store is postgres", "why": "w", "valid_from": day})[0]["id"]
        c._patch(rid, "test", egress="external_ok")
        answer = {"stale": [{"id": rid, "by": ids[1], "quote": "drop postgres, sqlite is final", "why": "replaced"}]}
        spec = importlib.util.spec_from_file_location("memrove_cli_apply", ROOT / "scripts/memrove_v3.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        argv = ["--vault", str(self.t / "v"), "--state", str(self.state), "reflect", "--day", day]

        def run(*extra):
            (self.state / "reflect-usage.json").unlink(missing_ok=True)
            with mock.patch.object(llm, "disabled", return_value=False), mock.patch.object(llm, "RUNNER", lambda s, p, vault=None: (json.dumps(answer), 0.0, None)), redirect_stdout(io.StringIO()) as o:
                self.assertEqual(cli.main(argv + list(extra)), 0)
            return json.loads(o.getvalue())
        closed = lambda: [r.get("valid_to") for r in self.payloads() if r["id"] == rid]
        self.assertEqual((run()["closed"], closed()), ([rid], [None]))   # would close, wrote nothing
        res = run("--apply")
        self.assertEqual((res["dry"], res["closed"], closed()), (False, [rid], [day]))

    def test_search_json_rejects_what_it_would_ignore(self):
        for extra in (["--since", "2026-01-01"], ["--until", "2026-12-01"], ["--history"]):
            self.assertEqual(self.run_cli("search", "SQLite", "--json", *extra)[0], 1, extra)
        self.assertEqual(self.run_cli("search", "", "--json")[0], 1)

    def test_remember_prints_relate_proposals(self):
        import importlib.util
        import io
        from contextlib import redirect_stderr, redirect_stdout
        from unittest import mock
        spec = importlib.util.spec_from_file_location("memrove_cli_under_test", ROOT / "scripts/memrove_v3.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        real = cli.Context.core

        def core(ctx):
            c = real(ctx)
            c.relate_decider = lambda old, new: {"relation": "supersedes", "why": "newer"}
            return c
        argv = ["--vault", str(self.t / "v"), "--state", str(self.state), "remember", "--kind", "decision", "--cwd", str(self.t), "--project", "alpha"]
        old = "use sqlite for the memrove store with five retries"
        with mock.patch.dict(os.environ, {"MEMROVE_RELATE": "1"}), mock.patch.object(cli.Context, "core", core), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(argv + [old, "--why", "w"]), 0)
            core_ = real(cli.Context(self.t / "v", self.state))
            core_._patch(self.payloads()[0]["id"], "test", valid_from="2026-01-01")
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(cli.main(argv + [old.replace("five", "three"), "--why", "w"]), 0)
        self.assertIn("relate proposal", err.getvalue())


class HookAccessTest(unittest.TestCase):
    def test_recall_hook_honours_the_read_policy_of_its_harness(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        state = tmp.name
        (Path(state) / "push.json").write_text(json.dumps({"recall_cards": True}))
        chatlog.append(state, [{"kind": "user", "text": "zanzibarello pineapple quokka configuration", "date": "2023-03-02", "k": "c:u1"}]
                       + [{"kind": "user", "text": "gardening compost %d" % i, "date": "2023-03-02", "k": "c:g%d" % i} for i in range(60)], today="2023-03-09")
        cs.sync(state)
        payload = {"hook_event_name": "UserPromptSubmit", "prompt": "zanzibarello pineapple quokka configuration question about the staging cluster", "session_id": "s1", "transcript_path": "", "cwd": state}
        os.environ.pop("MEMROVE_NO_RECALL", None)
        saved = (cs.LANE_MIN, cs.LANE_MIN_RARE)
        cs.LANE_MIN, cs.LANE_MIN_RARE = 0.0, 0
        self.addCleanup(lambda: setattr(cs, "LANE_MIN", saved[0]) or setattr(cs, "LANE_MIN_RARE", saved[1]))
        self.assertNotEqual(hook.recall_hook(state, payload), "{}")  # no access.json: open
        (Path(state) / "access.json").write_text(json.dumps({"agents": {"codex": {"read": []}}}))
        self.assertNotEqual(hook.recall_hook(state, {**payload, "session_id": "s2"}), "{}")  # a policy that names another agent leaves claude-code open
        self.assertEqual(hook.recall_hook(state, {**payload, "session_id": "s3"}, "codex"), "{}")

    def test_capture_advances_the_cursor_of_a_write_denied_agent(self):
        import memrove_v3_capture as capture
        from memrove_v3_sync import SyncEngine
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = Path(tmp.name)
        (t / "v").mkdir()
        tp = t / "t.jsonl"
        tp.write_text(json.dumps({"type": "user", "message": {"content": "Bundan sonra testleri her zaman once yaz."}}) + "\n", encoding="utf-8")
        engine = SyncEngine(t / "v", t / "s")
        (t / "s" / "access.json").write_text(json.dumps({"agents": {"claude-code": {"write": False}}}))
        self.assertEqual(capture.run(engine, {"event": "SessionEnd", "session": "s1", "transcript_path": str(tp), "harness": "claude"}), 0)
        cur = json.loads((t / "s" / "capture-cursors.json").read_text())["s1"]["cursor"]
        self.assertEqual(cur, tp.stat().st_size)


if __name__ == "__main__":
    unittest.main()
