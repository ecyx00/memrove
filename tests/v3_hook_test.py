"""Hook adapter: queue, worker, synthetic-prompt filter, opt-outs and what the installer ships."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from v3_package_helpers import clean_environ, isolated_env

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "template/.claude/scripts"
HOOK = SCRIPTS / "memrove_v3_hook.py"
INSTALLER = ROOT / "scripts/install_v3.py"


class HookTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="memrove-hook-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.vault, self.state = self.root / "vault dir", self.root / "state dir"
        self.vault.mkdir()
        self.env = isolated_env(self.root / "home")
        env = clean_environ()
        env.start()
        self.addCleanup(env.stop)
        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        spec = importlib.util.spec_from_file_location("memrove_hook_under_test", HOOK)
        self.hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.hook)
        self.payload = {"hook_event_name": "SessionEnd", "session_id": "s1", "event_id": "e1",
                        "cwd": str(self.vault), "prompt": "CANARY_NEVER_PERSIST"}

    def seed(self):
        from memrove_v3_sync import SyncEngine, render
        note = self.vault / "notes/plan.md"
        note.parent.mkdir()
        note.write_text(render({"id": "plan", "kind": "fact"}, "Nebula calibration is owned by Reviewer.\n"), encoding="utf-8")
        SyncEngine(self.vault, self.state).sync()

    def queue(self):
        return list((self.state / "hook-queue").glob("*.json"))

    def run_hook(self, payload, extra=()):
        out = subprocess.run([sys.executable, str(HOOK), "--vault", str(self.vault), "--state", str(self.state),
                              "--harness", "claude", *extra], input=json.dumps(payload), text=True, encoding="utf-8",
                             capture_output=True, cwd=self.vault, env=self.env, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_handoff_protocol_is_pushed_at_session_start_not_subagent_start(self):
        self.seed()
        start = self.run_hook({"hook_event_name": "SessionStart", "session_id": "s1", "cwd": str(self.vault), "source": "startup"})
        self.assertIn("memrove card --did", start["hookSpecificOutput"]["additionalContext"])
        sub = self.run_hook({"hook_event_name": "SubagentStart", "session_id": "s1", "cwd": str(self.vault), "agent_type": "general-purpose"})
        self.assertNotIn("memrove card", json.dumps(sub))

    def test_queue_dedupes_and_stores_metadata_only(self):
        a = self.hook.enqueue_event(self.vault, self.state, self.payload, "codex")
        self.assertEqual(a, self.hook.enqueue_event(self.vault, self.state, self.payload, "codex"))
        files = self.queue()
        self.assertEqual(len(files), 1)
        self.assertNotIn("CANARY", files[0].read_text())

    def test_failed_drain_retries_and_acks_once(self):
        from memrove_v3_sync import SyncEngine
        self.seed()
        self.hook.enqueue_event(self.vault, self.state, self.payload, "codex")
        with patch.object(SyncEngine, "sync", side_effect=RuntimeError("boom")):
            failed = self.hook.drain_queue(self.vault, self.state)
        self.assertEqual((failed["processed"], failed["failed"], failed["pending"]), (0, 1, 1))
        self.assertEqual(self.hook.drain_queue(self.vault, self.state)["processed"], 1)
        self.hook.enqueue_event(self.vault, self.state, self.payload, "codex")
        self.assertEqual(self.hook.drain_queue(self.vault, self.state)["processed"], 0)

    def test_worker_syncs_incrementally(self):
        from memrove_v3_sync import SyncEngine
        self.seed()
        self.hook.enqueue_event(self.vault, self.state, self.payload, "codex")
        with patch.object(SyncEngine, "sync", autospec=True, wraps=SyncEngine.sync) as sync:
            self.hook.drain_queue(self.vault, self.state)
        self.assertIs(sync.call_args.kwargs.get("full"), False)

    def test_degraded_sync_is_acknowledged_and_flagged(self):
        self.seed()
        (self.vault / "notes/bad.md").write_text("---\nkey:\n  a:\n    b: c\n---\nx\n", encoding="utf-8")
        self.hook.enqueue_event(self.vault, self.state, self.payload, "codex")
        done = self.hook.drain_queue(self.vault, self.state)
        self.assertEqual((done["processed"], done["failed"], done["pending"]), (1, 0, 0))
        health = json.loads((self.state / "hook-health.json").read_text(encoding="utf-8"))
        self.assertEqual(health["sync"]["status"], "degraded")
        self.assertIn("needs attention", self.hook.health_warning(self.state))

    def companion_file(self, history):
        folder = self.vault / "Companion"
        folder.mkdir(exist_ok=True)
        path = folder / "Last-Session.md"
        path.write_text("# Last session\n\nNewest handoff.\n\n## Previous sessions\n\n" + history, encoding="utf-8")
        return path

    def test_session_end_compacts_an_over_limit_companion_file_into_the_archive(self):
        self.seed()
        old = "- 2026-01-01 OLD-FACT-MARKER\n" * 200
        path = self.companion_file(old)
        self.hook.enqueue_event(self.vault, self.state, self.payload, "codex")
        self.assertEqual(self.hook.drain_queue(self.vault, self.state)["failed"], 0)
        self.assertLessEqual(len(path.read_text(encoding="utf-8")), 3000)
        archive = "".join(p.read_text(encoding="utf-8") for p in (self.vault / "Companion/Archive").glob("Last-Session-*.md"))
        self.assertEqual(archive.count("OLD-FACT-MARKER"), 200)

    def test_threads_over_limit_evict_whole_old_threads_to_the_archive(self):
        self.seed()
        folder = self.vault / "Companion"
        folder.mkdir()
        body = "".join(f"### Thread {i}\n**Status:** active\n- 2026-01-{i % 28 + 1:02d} update {'x' * 300}\n\n" for i in range(60))
        (folder / "Threads.md").write_text("# Threads\n\n## Active Threads\n\n" + body + "## Closed Threads\n", encoding="utf-8")
        self.hook.enqueue_event(self.vault, self.state, self.payload, "codex")
        self.hook.drain_queue(self.vault, self.state)
        live = (folder / "Threads.md").read_text(encoding="utf-8")
        archive = "".join(p.read_text(encoding="utf-8") for p in (folder / "Archive").glob("Threads-*.md"))
        self.assertLessEqual(len(live), 8000)
        for i in range(60):
            self.assertEqual((f"### Thread {i}\n" in live) + archive.count(f"### Thread {i}\n"), 1, i)

    def test_under_limit_companion_file_is_untouched(self):
        self.seed()
        path = self.companion_file("- 2026-01-01 small\n")
        before = path.read_bytes()
        self.hook.enqueue_event(self.vault, self.state, self.payload, "codex")
        self.hook.drain_queue(self.vault, self.state)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.vault / "Companion/Archive").exists())

    def test_claude_md_over_budget_prints_one_line(self):
        md = self.root / "CLAUDE.md"
        md.write_text("x" * 7300)
        with patch.dict(os.environ, {"MEMROVE_CLAUDE_MD_BUDGET": "7200"}):
            line = self.hook.claude_md_warning(md)
            self.assertEqual(line, "CLAUDE.md is 100 bytes over budget: convert a rule to a check or remove one\n")
            md.write_text("x" * 7200)
            self.assertEqual(self.hook.claude_md_warning(md), "")
            self.assertEqual(self.hook.claude_md_warning(self.root / "missing.md"), "")

    def test_concurrent_drains_acknowledge_once(self):
        self.seed()
        self.hook.enqueue_event(self.vault, self.state, self.payload, "codex")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.hook.drain_queue(self.vault, self.state), range(2)))
        self.assertEqual(sum(r["processed"] for r in results), 1)
        self.assertEqual(self.queue(), [])

    def test_guards_skip_the_queue(self):
        for name in ("MEMROVE_V3_INTERNAL", "MEMROVE_V3_SKIP"):
            self.env[name] = "1"
            self.assertEqual(self.run_hook(self.payload), {})
            del self.env[name]
        self.assertEqual(self.queue(), [])
        for n, value in enumerate(("0", "false", ""), 1):  # only an exact 1 disables memory
            self.env["MEMROVE_V3_SKIP"] = value
            self.run_hook(dict(self.payload, event_id=f"x{n}"))
            self.assertEqual(len(self.queue()), n)

    def test_no_memory_marker_skips_the_session(self):
        (self.vault / ".memrove").mkdir()
        (self.vault / ".memrove/no-memory").write_text("")
        self.assertEqual(self.run_hook(self.payload), {})
        self.assertEqual(self.queue(), [])

    def test_synthetic_prompt_detection(self):
        synthetic = self.hook.is_synthetic_prompt
        for prefix in ("<task-notification> done", "  <agent-message id=1>", "<local-command-stdout> out",
                       "Another Claude session sent a message: hi"):
            self.assertTrue(synthetic({"prompt": prefix}), prefix)
        for origin in ("peer", {"kind": "coordinator"}):
            self.assertTrue(synthetic({"prompt": "plain", "origin": origin}))
        for kind in ("channel", "remote", "", "human"):  # unknown kinds fall back to the text
            self.assertFalse(synthetic({"prompt": "plain", "origin": {"kind": kind}}))
        self.assertFalse(synthetic({"prompt": "<task-notification>", "origin": "human"}))
        self.assertTrue(synthetic({"prompt": [{"type": "text", "text": "<task-notification> x"}]}))
        for empty in ({}, None, {"prompt": None}):
            self.assertFalse(synthetic(empty))
        with patch.dict(os.environ, {"MEMROVE_V3_FILTER_HARNESS_TURNS": "0"}):
            self.assertFalse(synthetic({"prompt": "<task-notification> x"}))

    def test_recall_hook_fails_open_and_skips_synthetic_turns(self):
        self.assertEqual(self.hook.recall_hook(self.state, {"prompt": "nebula"}), "{}")  # no store yet
        self.seed()
        with patch.dict(os.environ, {"MEMROVE_NO_RECALL": "1"}):
            self.assertEqual(self.hook.recall_hook(self.state, {"prompt": "nebula"}), "{}")
        self.assertEqual(self.hook.recall_hook(self.state, {"prompt": "<task-notification> nebula"}), "{}")

    def test_session_start_carries_the_notice(self):
        notice = self.hook.session_notice(self.state, {"source": "clear"})
        self.assertIn("lead, not evidence", notice)

    def test_compact_start_carries_the_owners_recent_words(self):
        def rec(kind, content, **kw):
            return json.dumps({"type": kind, "message": {"content": content}, **kw})
        lines = [rec("user", "first prompt " + "x" * 50),
                 rec("user", [{"type": "tool_result", "content": "TOOL_OUT"}]),
                 rec("user", "<system-reminder>REMINDER</system-reminder>"),
                 rec("user", "This session is being continued from a previous conversation: SUMMARY", isCompactSummary=True),
                 rec("user", "subagent words", isSidechain=True),
                 rec("assistant", "AGENT_REPLY"),
                 rec("user", "second " + "y" * 900),
                 rec("user", "third prompt")]
        path = self.root / "t.jsonl"
        path.write_text("\n".join(lines), encoding="utf-8")
        payload = {"source": "compact", "transcript_path": str(path), "cwd": str(self.vault)}
        text = self.hook.session_notice(self.state, payload)
        for gone in ("TOOL_OUT", "REMINDER", "SUMMARY", "subagent words", "AGENT_REPLY"):
            self.assertNotIn(gone, text)
        self.assertLess(text.index("first prompt"), text.index("second"))
        self.assertLess(text.index("second"), text.index("third prompt"))
        self.assertIn("second " + "y" * 693 + " [cut: 207 more chars]", text)  # 700 chars kept, then the marker
        self.assertNotIn("y" * 694, text)
        tail = self.hook.recent_words(path, budget=250)  # over budget: newest kept, oldest dropped
        self.assertIn("third prompt", tail)
        self.assertNotIn("first prompt", tail)
        self.assertLessEqual(len(tail), 250)
        self.assertNotIn("recent words", self.hook.session_notice(self.state, {**payload, "source": "resume"}))
        self.assertNotIn("recent words", self.hook.session_notice(self.state, {**payload, "source": "startup"}))

    def test_installer_ships_the_model_daemon_and_the_grab_panel(self):
        out = subprocess.run([sys.executable, str(INSTALLER), "--vault", str(self.vault), "--state", str(self.state)],
                             capture_output=True, text=True, encoding="utf-8", cwd=self.vault, env=self.env, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        for name in ("memrove_modeld.py", "memrove_v3_grab_panel.swift", "memrove_v3_hook.py"):
            self.assertTrue((self.vault / ".claude/scripts" / name).is_file(), name)

    def test_installed_hook_command_runs_in_a_path_with_spaces(self):
        subprocess.run([sys.executable, str(INSTALLER), "--vault", str(self.vault), "--state", str(self.state)],
                       capture_output=True, cwd=self.vault, env=self.env, timeout=60, check=True)
        self.seed()
        config = json.loads((self.vault / ".codex/hooks.json").read_text())
        hooks = [h for m in config["hooks"]["SessionStart"] for h in m.get("hooks", []) if "memrove_v3_hook.py" in str(h)]
        self.assertEqual(len(hooks), 1)
        command = [hooks[0]["command"], *hooks[0].get("args", [])] if hooks[0].get("args") else hooks[0]["command"]
        out = subprocess.run(command, shell=isinstance(command, str), input=json.dumps(dict(self.payload, hook_event_name="SessionStart")),
                             text=True, encoding="utf-8", capture_output=True, cwd=self.vault, env=self.env, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        json.loads(out.stdout)


if __name__ == "__main__":
    unittest.main()
