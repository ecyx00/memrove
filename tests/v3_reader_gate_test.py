#!/usr/bin/env python3
"""Fail-closed egress: an unknown label counts as device_only everywhere, and a CLI reader that is not Claude Code, Codex or a terminal
(opencode, Gemini, a script) sees only the records its tier may see. Stub RUNNER only."""
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
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
        self.assertIn("neighbour rec:obs-acme", self.run_cli("remember", said, "--kind", "decision", "--project", "acme", MEMROVE_READER="local").stderr)   # the nudge fires for the owner
        r = self.run_cli("remember", said + " again", "--kind", "decision", "--project", "acme")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("private widget", r.stdout + r.stderr)

    def test_replaced_line_names_only_replacer_ids_the_reader_may_see(self):   # D12
        import memrove_v3_neighbours as nb
        from unittest import mock
        mod = cli_module()
        tail = "history, do not use as current."

        def shown(line, visible):
            with mock.patch.object(nb, "replaced", return_value=line), mock.patch("builtins.print") as out:
                mod.cmd_show(mock.Mock(id="obs-open", visible=visible), mod.Context(self.vault, self.state))
            return [c.args[0] for c in out.call_args_list if str(c.args[0]).startswith("REPLACED")]
        free = mod._record_check(self.state, "free")   # obs-open is visible, obs-closed is not
        self.assertEqual(shown(f"REPLACED by obs-closed, obs-open on 2026-11-15: {tail}", free), [f"REPLACED by obs-open on 2026-11-15: {tail}"])
        self.assertEqual(shown(f"REPLACED by obs-closed on 2026-11-15: {tail}", free), [f"REPLACED on 2026-11-15: {tail}"])
        self.assertEqual(shown(f"REPLACED by obs-closed on 2026-11-15: {tail}", None), [f"REPLACED by obs-closed on 2026-11-15: {tail}"])   # the owner's harness sees the id
        search = (f"REPLACED by obs-closed on 2026-11-15: {tail}\nrec:obs-open|2026-11-03|decision|open rule\n"
                  f"REPLACED by obs-open on 2026-11-15: {tail}\nrec:obs-closed|2026-11-03|decision|private rule")   # a boundary goes with its row, only when the row stays
        self.assertEqual(mod._reader_lines(search, free).splitlines(), [f"REPLACED on 2026-11-15: {tail}", "rec:obs-open|2026-11-03|decision|open rule"])

    def test_a_hidden_replacer_id_is_in_no_show_field_and_no_search_row_tag(self):   # M2
        self.engine.note_create("notes/observed/new.md", "newer open rule about widgets", {"id": "obs-new", "kind": "decision", "egress": "external_ok"})
        with sqlite3.connect(self.state / "memory.sqlite3") as db:   # obs-open is replaced by obs-closed (hidden from this reader) and by obs-new
            db.execute("UPDATE records SET payload=json_set(payload,'$.supersedes',json_array('obs-open')) WHERE id IN ('obs-closed','obs-new')")
        free, owner = self.run_cli("show", "obs-open"), self.run_cli("show", "obs-open", MEMROVE_READER="local")
        self.assertEqual(free.returncode, 0, free.stderr)
        self.assertNotIn("obs-closed", free.stdout + free.stderr)
        self.assertEqual(json.loads(free.stdout[free.stdout.index("{"):])["superseded_by"], ["obs-new"])   # visible ids stay
        self.assertEqual(json.loads(owner.stdout[owner.stdout.index("{"):])["superseded_by"], ["obs-closed", "obs-new"])   # the owner's harness sees both
        mod = cli_module()
        row = "rec:obs-open|2026-11-03|decision [replaced by rec:obs-closed]|open rule"
        self.assertEqual(mod._reader_lines(row, mod._record_check(self.state, "free")), "rec:obs-open|2026-11-03|decision [replaced]|open rule")
        self.assertEqual(mod._reader_lines(row.replace("obs-closed", "obs-new"), mod._record_check(self.state, "free")), row.replace("obs-closed", "obs-new"))

    def _replaced_by_hidden(self):
        """obs-open (visible) is replaced by obs-closed (hidden from a free reader)."""
        with sqlite3.connect(self.state / "memory.sqlite3") as db:
            db.execute("UPDATE records SET payload=json_set(payload,'$.supersedes',json_array('obs-open')) WHERE id='obs-closed'")

    def test_n1_every_replaced_tag_of_a_search_row_loses_a_hidden_id(self):
        from unittest import mock
        import io
        import memrove_v3_chatsearch as chat
        mod = cli_module()
        self.engine.note_create("notes/observed/new.md", "newer open rule about widgets", {"id": "obs-new", "kind": "decision", "egress": "external_ok"})
        row = "rec:obs-open|2026-11-03|decision [replaced by rec:obs-new] [replaced by rec:obs-closed]|open rule"   # a topic head and a graph edge both replace it
        for env, expect in (("free", "rec:obs-open|2026-11-03|decision [replaced by rec:obs-new] [replaced]|open rule"), ("local", row)):
            out = io.StringIO()
            args = mock.Mock(query="open", limit=None, since=None, until=None, history=False, json=False, records=False, visible=mod._record_check(self.state, "free") if env == "free" else None)
            with mock.patch.object(chat, "search", return_value=row), mock.patch("sys.stdout", out), mock.patch("sys.stderr", out):
                mod.cmd_search(args, mod.Context(self.vault, self.state))
            self.assertEqual(out.getvalue().strip(), expect)

    def test_n2_card_names_no_hidden_id_of_an_open_pair(self):
        import memrove_v3_suspense as S
        self._replaced_by_hidden()
        self.engine.note_create("notes/observed/new.md", "newer open rule about widgets", {"id": "obs-new", "kind": "decision", "egress": "external_ok"})
        with sqlite3.connect(self.state / "memory.sqlite3") as db:
            db.execute(S.SCHEMA)
            db.executemany("INSERT INTO suspense(new_id,nb_id,day,session,project) VALUES (?,?,'2026-11-15','s1','acme')", [("obs-closed", "obs-open"), ("obs-new", "obs-open")])
        argv = ("card", "--session", "s1", "--project", "acme", "--cwd", str(self.root))
        free, owner = self.run_cli(*argv), self.run_cli(*argv, MEMROVE_READER="local")
        self.assertEqual(free.returncode, 0, free.stderr)
        self.assertNotIn("obs-closed", free.stdout + free.stderr)
        self.assertIn("open pairs: 2 in project, 2 this session [obs-new>obs-open]", free.stdout)   # the count is the whole truth, the ids only the visible pairs
        self.assertIn("obs-closed>obs-open", owner.stdout)

    def test_n3_replaces_error_names_no_hidden_replacer(self):
        self._replaced_by_hidden()
        argv = ("remember", "A later rule about widgets", "--kind", "decision", "--project", "acme", "--replaces", "obs-open", "--cwd", str(self.root))
        free, owner = self.run_cli(*argv), self.run_cli(*argv, MEMROVE_READER="local")
        self.assertEqual(free.returncode, 0, free.stderr)
        self.assertIn("obs-open is already replaced;", free.stderr)
        self.assertNotIn("obs-closed", free.stdout + free.stderr)
        self.assertIn("obs-open is already replaced by obs-closed;", owner.stderr)

    def test_n4_edge_summary_of_a_replaced_neighbour_names_no_hidden_replacer(self):
        import memrove_v3_neighbours as nb
        from unittest import mock
        import io
        mod = cli_module()
        line = "REPLACED by obs-closed on 2026-11-15: history, do not use as current."
        edges = [{"type": "supports", "dir": "out", "id": "obs-open", "summary": line, "why": "w"}]
        for visible, expect in ((mod._record_check(self.state, "free"), "REPLACED on 2026-11-15: history, do not use as current."), (None, line)):
            out = io.StringIO()
            with mock.patch.object(nb, "edges_of", return_value=edges), mock.patch("sys.stdout", out), mock.patch("sys.stderr", out):
                mod.cmd_show(mock.Mock(id="obs-open", visible=visible), mod.Context(self.vault, self.state))
            self.assertEqual(json.loads(out.getvalue()[out.getvalue().index("{"):])["edges"][0]["summary"], expect)
            self.assertEqual("obs-closed" in out.getvalue(), visible is None)

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
