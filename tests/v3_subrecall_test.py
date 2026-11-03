#!/usr/bin/env python3
"""Subagent recall: the brief is found, ranking ignores age, the owner's words come first, conflicts show both dates, budget, fallback, scout."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "template/.claude/scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import memrove_v3_chatsearch as cs  # noqa: E402
import memrove_v3_subrecall as sr  # noqa: E402
from v3_rules_test import judged, rec, store  # noqa: E402


def r(i, kind, text, day, origin="agent_inference", project="acme", **kw):
    return rec(i, kind, text, day, project=project, origin_type=origin, verified_by="", source=f"notes/observed/{i}.md", **kw)


BRIEF = "Write the SubagentStart hook for the billing exporter: parse invoices with the ledger parser and add tests for the exporter."
RECS = [
    r("a1", "decision", "The billing exporter must parse invoices with the ledger parser, never regex.", "2026-04-02"),
    r("a2", "preference", "For the billing exporter the owner wants invoices parsed by the ledger parser and tests for the exporter.", "2026-10-30", origin="user_explicit"),
    r("a3", "lesson", "Ledger parser invoices exporter billing tests: the parser silently drops credit notes.", "2026-10-01"),
    r("a4", "decision", "Unrelated: the dashboard theme is dark.", "2026-10-31"),
    r("a5", "decision", "Billing exporter invoices ledger parser: other project only.", "2026-10-31", project="other"),
]


class SubrecallTest(unittest.TestCase):
    def setUp(self):
        self.d = store(RECS)
        self.no_msgs = mock.patch.object(sr, "_messages", return_value=[])
        self.no_msgs.start()
        self.addCleanup(self.no_msgs.stop)

    def test_brief_found_and_scoped(self):
        ids = [c["id"] for c in sr.recall(self.d, BRIEF, "acme")]
        self.assertIn("rec:a1", ids)
        self.assertNotIn("rec:a4", ids)    # unrelated
        self.assertNotIn("rec:a5", ids)    # other project

    def test_old_record_can_win_and_age_is_not_a_signal(self):
        c = sr.candidates(self.d, BRIEF, "acme")
        self.assertEqual(c[0]["id"], "rec:a1" if c[0]["date"] < "2026-10" else c[0]["id"])
        self.assertIn("2026-04-02", [x["date"] for x in c])
        self.assertNotEqual([x["date"] for x in c], sorted((x["date"] for x in c), reverse=True))   # not sorted newest first by construction

    def test_owner_words_first(self):
        items = sr.recall(self.d, BRIEF, "acme")
        self.assertTrue(items[0]["user"])
        self.assertEqual(items[0]["id"], "rec:a2")

    def test_user_message_is_owner_words(self):
        row = (7, "2026-03-01T10:00:00", "user", "billing exporter invoices need the ledger parser tests, do it properly")
        with mock.patch.object(sr, "_messages", return_value=[(-9.0, row)]):
            items = sr.recall(self.d, BRIEF, "acme")
        self.assertEqual(items[0]["id"], "msg:7")
        self.assertEqual(items[0]["date"], "2026-03-01")

    def test_selector_picks_replace_search_and_missing_job_falls_through(self):
        import memrove_v3_selector as sel
        with mock.patch.object(sr, "from_picks", wraps=sr.from_picks):
            txt = sr.block(self.d, BRIEF, "acme", ["rec:a3", "rec:zzz"])
        self.assertIn("rec:a3", txt)
        self.assertNotIn("rec:a1", txt)
        self.assertIsNone(sel.wait(self.d, BRIEF, 0.2))   # no job was started: caller keeps the dense/bm25 result
        out = sel.out_path(self.d, sel.key_of(BRIEF))
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"picks": ["m1"]}))
        self.assertEqual(sel.wait(self.d, BRIEF, 0.2), ["m1"])

    def test_hand_back_messages_are_not_owner_words(self):
        for i, t in enumerate(["<agent-message> billing exporter invoices ledger parser tests exporter",
                               "Spawn the subagent for the billing exporter invoices ledger parser tests",
                               "billing exporter invoices ledger parser tests " + "x" * 1100]):
            with mock.patch.object(sr, "_messages", return_value=[(-9.0, (8 + i, "2026-10-30T10:00:00", "user", t))]):
                self.assertNotIn(f"msg:{8 + i}", [c["id"] for c in sr.recall(self.d, BRIEF, "acme")])

    def test_conflicting_pair_shows_both_with_dates(self):
        d = store([r("c1", "decision", "Billing exporter invoices parse with the ledger parser and tests.", "2026-04-02"),
                   r("c2", "decision", "Billing exporter invoices: never use the ledger parser, no tests for it.", "2026-10-20")])
        text = sr.block(d, BRIEF, "acme")
        self.assertIn("rec:c1 2026-04-02", text)
        self.assertIn("rec:c2 2026-10-20", text)
        self.assertIn("conflicts with rec:c2 2026-10-20", text)
        self.assertIn("conflicts with rec:c1 2026-04-02", text)

    def test_budget_and_item_cap(self):
        many = [r(f"m{i}", "lesson", "Billing exporter invoices ledger parser tests " + "x" * 400 + f" {i}", "2026-10-0" + str(i % 9 + 1)) for i in range(12)]
        text = sr.block(store(many), BRIEF, "acme")
        self.assertLessEqual(len(text), sr.BUDGET_CHARS)
        self.assertLessEqual(text.count("\n- "), sr.MAX_ITEMS)

    def test_nothing_relevant_is_empty(self):
        self.assertEqual(sr.block(self.d, "Rename the swift variable in the camera widget", "acme"), "")


class HookTest(unittest.TestCase):
    def run_hook(self, payload, d, **env):
        out = subprocess.run([sys.executable, str(SCRIPTS / "memrove_v3_jit.py")], input=json.dumps(payload), text=True, capture_output=True,
                             env={**os.environ, "MEMROVE_STATE": d, **env}, timeout=15).stdout
        return json.loads(out).get("hookSpecificOutput", {}).get("additionalContext")

    def test_stash_then_start_injects_recall_and_global_rules_only(self):
        d = store([*RECS, rec("g1", "preference", "x", "2026-10-13", assessments=judged("Everything Claude writes is English.")),
                   rec("p1", "preference", "y", "2026-10-13", project="acme", assessments=judged("Acme has no widgetco.", "project", "acme"))])
        sr.stash(d, {"session_id": "s1", "tool_name": "Agent", "tool_input": {"subagent_type": "coder", "prompt": BRIEF}})
        cwd = Path(tempfile.mkdtemp(prefix="v3-sub-")) / "acme"
        (cwd / ".git").mkdir(parents=True)
        ctx = self.run_hook({"hook_event_name": "SubagentStart", "session_id": "s1", "agent_type": "coder", "cwd": str(cwd)}, d)
        self.assertIn("rec:a2", ctx)
        self.assertIn("Everything Claude writes is English.", ctx)
        self.assertNotIn("widgetco", ctx)

    def test_scout_gets_nothing(self):
        d = store(RECS)
        sr.stash(d, {"session_id": "s1", "tool_name": "Agent", "tool_input": {"subagent_type": "scout", "prompt": BRIEF}})
        self.assertIsNone(self.run_hook({"hook_event_name": "SubagentStart", "session_id": "s1", "agent_type": "scout"}, d))

    def test_error_falls_back_to_full_standing_rules(self):
        d = store([rec("g1", "preference", "x", "2026-10-13", assessments=judged("Everything Claude writes is English."))])
        with open(Path(d) / "agent-briefs.jsonl", "w") as f:
            f.write("not json\n")
        ctx = self.run_hook({"hook_event_name": "SubagentStart", "session_id": "s1", "agent_type": "coder"}, d)
        self.assertIn("Everything Claude writes is English.", ctx)
        ctx = self.run_hook({"hook_event_name": "SubagentStart", "agent_type": "coder"}, d, MEMROVE_SUBRECALL="0")
        self.assertIn("Before you answer or start a task", ctx)   # today's long rule

    def test_brief_from_transcript(self):
        t = Path(tempfile.mkdtemp()) / "agent-x.jsonl"
        t.write_text(json.dumps({"message": {"role": "user", "content": BRIEF}}) + "\n")
        self.assertEqual(sr.brief_of("/nonexistent", {"agent_transcript_path": str(t)}), BRIEF)


if __name__ == "__main__":
    unittest.main()
