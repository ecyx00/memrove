#!/usr/bin/env python3
"""Fail-closed egress: an unknown label counts as device_only everywhere, and a CLI reader that is not Claude Code, Codex or a terminal
(opencode, Gemini, a script) sees only the records its tier may see. Stub RUNNER only."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_llm_test import Base, evaluator  # noqa: E402
import memrove_v3_trust as trust  # noqa: E402
import memrove_v3_restate as restate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/memrove_v3.py"
OWNER_KEYS = ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CODEX_THREAD_ID", "CODEX_SANDBOX", "MEMROVE_READER")


def cli_module():
    spec = importlib.util.spec_from_file_location("memrove_cli_under_test", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FailClosedTest(Base):
    def test_unknown_or_empty_label_counts_as_device_only(self):
        for label in ("extrenal_ok", "", None, "EXTERNAL_OK"):
            self.assertFalse(trust.visible_to(label, "paid"), label)
            self.assertFalse(trust.visible_to(label, "free"), label)
            self.assertTrue(trust.visible_to(label, "local"), label)
        self.assertTrue(trust.visible_to("external_ok", "free"))
        self.assertFalse(trust.visible_to("external_ok", "nonsense") and trust.visible_to("paid_ok", "nonsense"))   # unknown tier = free
        self.assertFalse(trust.visible_to("paid_ok", "nonsense"))

    def test_evaluator_refuses_an_item_without_a_label(self):
        self.reply["lesson_reusable"] = {"items": [{"id": "r1", "reusable": "yes"}]}
        for item in ({"id": "r1", "text": "t", "prev": "p", "revision": 1}, {"id": "r1", "text": "t", "prev": "p", "revision": 1, "egress": "typo_ok"}):
            (a,) = evaluator.evaluate(self.state, "lesson_reusable", [item], provider="haiku")
            self.assertEqual(a["state"], "not_assessed")
        self.assertEqual(self.calls, [])

    def test_restate_items_carry_the_record_label(self):
        it = restate.item_of({"id": "q-1", "text": "x", "project": "acme", "verified_by": "user_quote", "revision": 1}, [])
        self.assertEqual(it["egress"], "external_ok")
        self.assertEqual(restate.item_of({"id": "q-2", "text": "x", "project": "acme"}, [])["egress"], "device_only")

    def test_reader_tier(self):
        self.assertEqual(trust.reader_tier({"CLAUDECODE": "1"}, tty=False), "local")
        self.assertEqual(trust.reader_tier({"CODEX_THREAD_ID": "t"}, tty=False), "local")
        self.assertEqual(trust.reader_tier({}, tty=True), "local")
        self.assertEqual(trust.reader_tier({"OPENCODE_CLIENT": "cli"}, tty=False), "free")
        self.assertEqual(trust.reader_tier({"MEMROVE_READER": "paid", "CLAUDECODE": "1"}, tty=True), "paid")
        self.assertEqual(trust.reader_tier({"MEMROVE_READER": "lokal"}, tty=True), "free")   # a typo fails closed


class CliReaderTest(Base):
    def setUp(self):
        super().setUp()
        self.engine.note_create("notes/observed/open.md", "open rule about widgets", {"id": "obs-open", "kind": "decision", "egress": "external_ok"})
        self.engine.note_create("notes/observed/closed.md", "private rule about widgets", {"id": "obs-closed", "kind": "decision", "egress": "device_only"})
        self.engine.note_create("notes/observed/typo.md", "typo rule about widgets", {"id": "obs-typo", "kind": "decision", "egress": "extrenal_ok"})

    def run_cli(self, *argv, **env):
        base = {k: v for k, v in os.environ.items() if k not in OWNER_KEYS}
        return subprocess.run([sys.executable, str(CLI), "--vault", str(self.vault), "--state", str(self.state), *argv],
                              text=True, capture_output=True, env={**base, **env}, timeout=60)

    def test_free_reader_show_refuses_blocked_records(self):
        self.assertEqual(self.run_cli("show", "obs-open").returncode, 0)
        for rid in ("obs-closed", "obs-typo"):
            r = self.run_cli("show", rid)
            self.assertEqual(r.returncode, 1, r.stdout)
            self.assertIn("PermissionError", r.stderr)
            self.assertNotIn("private rule", r.stdout + r.stderr)

    def test_free_reader_gets_no_blocked_text_through_edges_or_nudges(self):
        import memrove_v3_neighbours as nb
        from unittest import mock
        mod = cli_module()
        edges = [{"type": "supports", "dir": "out", "id": "obs-closed", "summary": "private rule about widgets", "why": "w"},
                 {"type": "supports", "dir": "out", "id": "obs-open", "summary": "open rule about widgets", "why": "w"}]
        args = mock.Mock(id="obs-open", visible=mod._record_check(self.state, "free"))
        with mock.patch.object(nb, "edges_of", return_value=edges), mock.patch("builtins.print") as out:
            mod.cmd_show(args, mod.Context(self.vault, self.state))
        text = " ".join(str(c) for c in out.call_args_list)
        self.assertIn("open rule", text)
        self.assertNotIn("private rule", text)
        self.engine.note_create("notes/observed/acme.md", "private widget pricing tables stay secret forever", {"id": "obs-acme", "kind": "decision", "project": "acme", "egress": "device_only"})
        said = "widget pricing tables stay secret forever and ever"
        self.assertIn("may replace: rec:obs-acme", self.run_cli("remember", said, "--kind", "decision", "--project", "acme", MEMROVE_READER="local").stderr)   # the nudge fires for the owner
        r = self.run_cli("remember", said + " again", "--kind", "decision", "--project", "acme")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("private widget", r.stdout + r.stderr)

    def test_owner_harness_sees_everything_as_before(self):
        for env in ({"CLAUDECODE": "1"}, {"CODEX_THREAD_ID": "t"}, {"MEMROVE_READER": "local"}):
            r = self.run_cli("show", "obs-closed", **env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("private rule", r.stdout)

    def test_free_reader_cannot_run_unfiltered_commands(self):
        r = self.run_cli("project")
        self.assertEqual(r.returncode, 1)
        self.assertIn("MEMROVE_READER=local", r.stderr)

    def test_search_lines_are_an_allow_list(self):
        mod = cli_module()
        text = "\n".join(["head|2026-11-03|topic|docs/state/x.md|agent write-up", "rec:obs-open|2026-11-03|decision|open rule",
                          "  > supports obs-closed private - why", "rec:obs-closed|2026-11-03|decision|private rule",
                          "6361|2026-10-13 13:33|user|owner message", "x-c8bfda4ff151c0eb|2026-08-11 09:15|agent|agent reply", "",
                          "docs (open with memrove doc PATH):", "doc|2026-11-03|/x.md|Decision|agent write-up"])
        out = mod._reader_lines(text, lambda rid: rid == "obs-open").splitlines()
        self.assertEqual(out, ["rec:obs-open|2026-11-03|decision|open rule", "6361|2026-10-13 13:33|user|owner message",
                               "x-c8bfda4ff151c0eb|2026-08-11 09:15|agent|agent reply"])
        ok = mod._record_check(self.state, "free")
        self.assertEqual([ok(i) for i in ("obs-open", "rec:obs-open", "obs-closed", "obs-typo", "missing")], [True, True, False, False, False])


if __name__ == "__main__":
    unittest.main()
