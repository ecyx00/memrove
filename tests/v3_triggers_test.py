#!/usr/bin/env python3
"""Lesson triggers: PreToolUse shows (warn) or refuses (deny) a lesson when a tool call matches; survives sync rebuild; fast."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JIT = ROOT / "template/.claude/scripts/memrove_v3_jit.py"
CLI = ROOT / "scripts/memrove_v3.py"
LESSON = "Scout uses unittest, not pytest."


class TriggerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v3-trig-")).resolve()
        self.vault, self.state, self.proj = self.tmp / "Vault", self.tmp / "state", self.tmp / "work" / "scout"
        self.vault.mkdir()
        (self.proj / ".git").mkdir(parents=True)
        self.env = {**{k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "MEMROVE"))}, "MEMROVE_READER": "local"}   # the owner, without a harness env
        self.env.update(HOME=str(self.tmp), CLAUDE_CODE_SESSION_ID="s1", MEMROVE_NO_LLM="1")

    def cli(self, *a):
        r = subprocess.run([sys.executable, str(CLI), "--vault", str(self.vault), "--state", str(self.state), *a], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def seed(self, text=LESSON, *extra):
        return self.cli("remember", text, "--kind", "lesson", "--cwd", str(self.proj), "--trigger-tool", "Bash",
                        "--trigger", r"\bpytest\b", "--trigger-cwd", "/scout", *extra)

    def hook(self, command="uv run pytest tests/", cwd=None, sid="s1", agent=None):
        p = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(cwd or self.proj), "session_id": sid}
        if agent:
            p["agent_id"] = agent
        r = subprocess.run([sys.executable, str(JIT), "--state", str(self.state)], input=json.dumps(p), capture_output=True, text=True, env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_warn_scope_and_once(self):
        rid = self.seed()
        out = self.hook()["hookSpecificOutput"]
        self.assertEqual(out["hookEventName"], "PreToolUse")
        self.assertTrue(out["additionalContext"].startswith(f"memrove lesson {rid}"), out)
        self.assertIn(LESSON, out["additionalContext"])
        self.assertEqual(self.hook(), {})                        # once per session
        self.assertNotEqual(self.hook(agent="a1"), {})           # another agent in the same session
        self.assertEqual(self.hook(sid="s2", cwd=self.tmp), {})  # cwd outside scout
        self.assertEqual(self.hook(sid="s3", command="python -m unittest"), {})  # no match

    def test_deny(self):
        self.seed("Never run pytest in scout.", "--trigger-deny")
        for _ in range(2):  # deny is not once-only: the next call must be refused too
            out = self.hook()["hookSpecificOutput"]
            self.assertEqual(out["permissionDecision"], "deny")
            self.assertIn("Never run pytest", out["permissionDecisionReason"])

    def test_trigger_command_survives_rebuild(self):
        rid = self.cli("remember", LESSON, "--kind", "lesson", "--cwd", str(self.proj))
        self.assertEqual(self.hook(), {})
        self.cli("trigger", rid, "--tool", "Bash", "--match", r"\bpytest\b")
        self.assertIn(rid, self.cli("trigger", "--list"))
        self.assertNotEqual(self.hook(sid="s9"), {})
        (self.state / "memory.sqlite3").unlink()
        (self.state / "triggers.json").unlink()
        self.cli("sync")
        self.assertIn(rid, (self.state / "triggers.json").read_text())
        self.assertNotEqual(self.hook(sid="s10"), {})

    def test_corrupt_file(self):
        self.seed()
        (self.state / "triggers.json").write_text("{not json")
        self.assertEqual(self.hook(), {})

    def test_dead_lessons_do_not_fire(self):
        import sqlite3
        for i, (patch, fires) in enumerate([({"status": "retired"}, False), ({"verdict": "reject"}, False), ({"valid_to": "2000-01-01"}, False), ({"valid_to": "2999-01-01"}, True)]):
            rid = self.seed(f"Lesson number {i} about pytest.")
            with sqlite3.connect(self.state / "memory.sqlite3") as db:
                (p,) = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
                db.execute("UPDATE records SET payload=? WHERE id=?", (json.dumps({**json.loads(p), **patch}), rid))
            sys.path.insert(0, str(JIT.parent))
            import memrove_v3_triggers as trig
            trig.compile(self.vault, self.state)
            out = self.hook(sid=f"dead{i}")
            self.assertEqual(rid in json.dumps(out), fires, (patch, out))
            trig.set_trigger(self.vault, self.state, rid, None)

    def test_match_runs_on_raw_command(self):
        self.cli("remember", "Quote lesson.", "--kind", "lesson", "--cwd", str(self.proj), "--trigger-tool", "Bash", "--trigger", 'echo "x"')
        self.assertNotEqual(self.hook(command='echo "x" > f'), {})

    def test_latency(self):
        self.seed()
        times = []
        for _ in range(100):
            t = time.perf_counter()
            self.hook(command="ls -la", sid="lat")
            times.append((time.perf_counter() - t) * 1000)
        p95 = sorted(times)[94]
        print(f"hook p95 {p95:.1f} ms (100 runs)")
        self.assertLess(p95, 80)


if __name__ == "__main__":
    unittest.main()
