#!/usr/bin/env python3
"""Cursor reader: IDE db (read-only) and agent-CLI transcript; real dates, sessions, kinds, projects, 0 on re-import."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_imports as imports  # noqa: E402


def fixture(home):
    user = home / 'Library/Application Support/Cursor/User'
    (user / 'globalStorage').mkdir(parents=True)
    proj = home / 'Desktop/Alpha'
    (proj / '.git').mkdir(parents=True)
    ws = user / 'workspaceStorage/w1'
    ws.mkdir(parents=True)
    (ws / 'workspace.json').write_text(json.dumps({"folder": "file://" + str(proj)}))
    w = sqlite3.connect(ws / 'state.vscdb')
    w.execute("CREATE TABLE ItemTable(key TEXT PRIMARY KEY, value BLOB)")
    w.execute("INSERT INTO ItemTable VALUES('composer.composerData', ?)", (json.dumps({"allComposers": [{"composerId": "c1"}]}),))
    w.commit()
    g = sqlite3.connect(user / 'globalStorage/state.vscdb')
    g.execute("CREATE TABLE cursorDiskKV(key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
    kv = [("composerData:c1", {"createdAt": 1755848939459}), ("composerData:c2", {"createdAt": 1778150447537})]
    b = lambda cid, bid, t, text, at=None, tool=False: (f"bubbleId:{cid}:{bid}", {"type": t, "text": text, "createdAt": at, **({"toolFormerData": {"x": 1}} if tool else {})})
    kv += [b("c1", "b1", 1, "set up the project", "2026-04-07T16:48:03.417Z"),
           b("c1", "b2", 2, "narration: reading files", "2026-04-07T16:48:04.000Z", tool=True),
           b("c1", "b3", 2, "", "2026-04-07T16:48:05.000Z"),
           b("c1", "b4", 2, "Done, the project runs.", "2026-04-07T16:48:09.000Z"),
           b("c2", "b5", 1, "old chat without bubble time"),   # falls back to the composer's createdAt
           b("c2", "b6", 2, "old answer")]
    g.executemany("INSERT INTO cursorDiskKV VALUES(?,?)", [(k, json.dumps(v)) for k, v in kv])
    g.commit()
    t = home / '.cursor/projects/-x/agent-transcripts'
    (t / 'c1').mkdir(parents=True)
    (t / 'c1/c1.jsonl').write_text('{"role":"user","message":{"content":[{"type":"text","text":"covered by the db"}]}}\n')
    (t / 'tr9').mkdir()
    rows = [{"role": "user", "message": {"content": [{"type": "text", "text": "<cursor_commands>big</cursor_commands><user_query>\nfix the bug\n</user_query>"}]}},
            {"role": "assistant", "message": {"content": [{"type": "text", "text": "looking"}, {"type": "tool_use", "name": "Read", "input": {}}]}},
            {"role": "assistant", "message": {"content": [{"type": "text", "text": "Fixed it."}]}}]
    (t / 'tr9/tr9.jsonl').write_text("".join(json.dumps(r) + "\n" for r in rows))


class Cursor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home, self.state = Path(self.tmp.name) / 'home', Path(self.tmp.name) / 'state'
        fixture(self.home)

    def rows(self, where="1"):
        return sqlite3.connect(self.state / 'chatlog/imports.db').execute(f"SELECT session, project, date, kind, text FROM x WHERE {where} ORDER BY session, date, kind DESC").fetchall()

    def test_cursor(self):
        a = imports.run(self.state, self.home, only="cursor")["cursor"]
        self.assertEqual(a["added"], 6)   # c1: user + final; c2: user + answer; tr9: user + final (the db-held transcript is skipped)
        rows = self.rows()
        c1 = [r for r in rows if r[0] == 'c1']
        self.assertEqual([(r[3], r[4]) for r in c1], [("user", "set up the project"), ("agent", "Done, the project runs.")])
        self.assertEqual({r[1] for r in c1}, {"Alpha"})
        self.assertTrue(c1[0][2].startswith("2026-04-07T") and c1[1][2] > c1[0][2])
        c2 = [r for r in rows if r[0] == 'c2']
        self.assertEqual(len(c2), 2)
        self.assertTrue(all(r[2].startswith("2026-05-07") for r in c2))   # composer createdAt 1778150447
        self.assertEqual({r[1] for r in c2}, {""})
        tr = [r for r in rows if r[0] == 'tr9']
        self.assertEqual([(r[3], r[4]) for r in tr], [("user", "fix the bug"), ("agent", "Fixed it.")])
        full = imports.run(self.state, self.home, only="cursor", full=True)["cursor"]
        self.assertEqual(full["added"], 2)   # the two narration messages
        self.assertEqual(imports.run(self.state, self.home, only="cursor", full=True)["cursor"]["added"], 0)

    def test_db_is_opened_read_only(self):
        db = self.home / 'Library/Application Support/Cursor/User/globalStorage/state.vscdb'
        db.chmod(0o444)
        before = db.read_bytes()
        imports.run(self.state, self.home, only="cursor")
        self.assertEqual(db.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
