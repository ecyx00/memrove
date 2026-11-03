#!/usr/bin/env python3
"""Stable-id imports: x-<hash> ids are the same on every run, a second import adds 0, search and zoom find them."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402
import memrove_v3_chattree as ct  # noqa: E402
import memrove_v3_imports as imports  # noqa: E402


def overview(home, sid="s1"):
    d = home / f'.gemini/antigravity/brain/{sid}/.system_generated/logs'
    d.mkdir(parents=True)
    rows = [
        {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "created_at": "2026-04-19T10:00:00Z",
         "content": "<USER_REQUEST>replan the FRONTEND_BUILD_PLAN for Svelte</USER_REQUEST>"},
        {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "created_at": "2026-04-19T10:00:05Z",
         "content": "narration only", "tool_calls": [{"name": "x"}]},
        {"step_index": 2, "source": "MODEL", "type": "PLANNER_RESPONSE", "created_at": "2026-04-19T10:00:09Z",
         "content": "The plan now targets Svelte 5 and Vite."},
    ]
    (d / 'overview.txt').write_text("".join(json.dumps(r) + "\n" for r in rows))


class Imports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home, self.state = Path(self.tmp.name) / 'home', Path(self.tmp.name) / 'state'
        overview(self.home)

    def ids(self):
        import sqlite3
        return sorted(r[0] for r in sqlite3.connect(self.state / 'chatlog/imports.db').execute("SELECT id FROM x"))

    def test_ids_stable_and_second_run_adds_zero(self):
        a = imports.run(self.state, self.home)["antigravity-overview"]
        first = self.ids()
        b = imports.run(self.state, self.home)["antigravity-overview"]
        self.assertEqual((a["added"], b["added"], len(first)), (2, 0, 2))   # lean: the user turn and the final answer
        self.assertEqual(first, self.ids())
        self.assertTrue(all(i.startswith("x-") and len(i) == 18 for i in first))
        self.assertEqual(imports.run(self.state, self.home, full=True)["antigravity-overview"]["added"], 1)   # the narration

    def test_skips_what_the_log_already_holds(self):
        self.state.mkdir(parents=True)
        (self.state / 'chatlog').mkdir()
        (self.state / 'chatlog/seen.txt').write_text(chatlog._sha("a:s1:0") + "\n")
        self.assertEqual(imports.run(self.state, self.home)["antigravity-overview"], {"read": 2, "skipped": 1, "added": 1})

    def test_extract_uses_the_session_project(self):
        import sqlite3
        from unittest import mock
        import memrove_v3_llm as llm
        imports.run(self.state, self.home)
        (self.state / 'memory.sqlite3').touch()
        sqlite3.connect(self.state / 'memory.sqlite3').execute("CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, payload TEXT)")
        db = sqlite3.connect(self.state / 'chatlog/imports.db')
        db.execute("UPDATE x SET project = 'fisai'"); db.commit()
        seen = []
        with mock.patch.object(llm, 'call', return_value={}), mock.patch.object(llm, '_thread_prompt', return_value=''), \
                mock.patch.object(llm, '_note', side_effect=lambda e, st, d, t, ev, *a: seen.append(ev)):
            imports.extract(None, self.home, self.state, link=False)
        self.assertEqual([e["project"] for e in seen], ["fisai"])

    def test_search_zoom_date(self):
        imports.run(self.state, self.home)
        hit = next(r for r in cs.hits(self.state, "Svelte 5 Vite") if r[0].startswith("x-"))
        self.assertIn("Svelte 5", hit[3])
        self.assertIn("Svelte 5", ct.zoom(self.state, hit[0], 1))
        self.assertTrue(ct.date_of(self.state, hit[0]).startswith("2026-04-19"))
        self.assertEqual(len([r for r in cs.hits(self.state, "", since="2026-04-19", until="2026-04-19") if r[0].startswith("x-")]), 2)


if __name__ == '__main__':
    unittest.main()
