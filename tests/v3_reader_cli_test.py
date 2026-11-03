#!/usr/bin/env python3
"""opencode and Copilot readers: real dates, sessions, kinds, project, the automation filter, and 0 added on re-import."""
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
import memrove_v3_reader_opencode as oc  # noqa: E402

MS = 1789215147000   # 2026-09-12 in local time; the exact day is asserted through the same conversion


def opencode(home, proj):
    p = home / '.local/share/opencode/opencode.db'
    p.parent.mkdir(parents=True)
    db = sqlite3.connect(p)
    db.executescript("""CREATE TABLE session(id, directory, parent_id, time_created);
        CREATE TABLE message(id, session_id, time_created, data);
        CREATE TABLE part(id, message_id, time_created, data);""")
    n = [0]

    def msg(sid, role, text, t):
        n[0] += 1
        mid = f"m{n[0]}"
        db.execute("INSERT INTO message VALUES(?,?,?,?)", (mid, sid, t, json.dumps({"role": role})))
        db.execute("INSERT INTO part VALUES(?,?,?,?)", (f"p{n[0]}", mid, t, json.dumps({"type": "text", "text": text})))

    for sid, cwd, parent, first in [("keep", str(proj), None, "fix the importer"), ("tmp", "/private/tmp/x", None, "hello"),
                                    ("child", str(proj), "keep", "sub"), ("json", str(proj), None, '"Task: triage items"')]:
        db.execute("INSERT INTO session VALUES(?,?,?,?)", (sid, cwd, parent, MS))
        msg(sid, "user", first, MS)
        msg(sid, "assistant", "looking", MS + 1000)      # narration
        msg(sid, "assistant", "done: importer fixed", MS + 2000)   # final
    db.commit()
    db.close()


def copilot(home):
    p = home / '.copilot/history-session-state/session_abc_1.json'
    p.parent.mkdir(parents=True)
    tl = [{"id": "u1", "type": "user", "timestamp": "2026-11-09T22:27:16.616Z", "text": "short", "expandedText": "bet model please"},
          {"id": "a1", "type": "copilot", "timestamp": "2026-11-09T22:27:24.000Z", "text": "narration"},
          {"id": "t1", "type": "tool_call_completed", "timestamp": "2026-11-09T22:27:25.000Z", "arguments": "{'path': '/Users/x/Desktop/bet/a.py'}"},
          {"id": "a2", "type": "copilot", "timestamp": "2026-11-09T22:27:30.000Z", "text": "model trained"}]
    p.write_text(json.dumps({"sessionId": "abc", "timeline": tl}))


class ReaderCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self.home, self.state = t / 'home', t / 'state'
        self.proj = t / 'work' / 'myproj'
        (self.proj / '.git').mkdir(parents=True)
        self.addCleanup(setattr, oc, 'TMP', oc.TMP)
        oc.TMP = ('/private/tmp/x',)   # the test dir itself lives under the system temp dir
        opencode(self.home, self.proj)
        copilot(self.home)

    def rows(self, src):
        db = sqlite3.connect(self.state / 'chatlog/imports.db')
        return db.execute("SELECT session, project, date, kind, text FROM x WHERE src = ? ORDER BY date, rowid", (src,)).fetchall()

    def test_opencode_filter_kinds_dates_project(self):
        a = imports.run(self.state, self.home, "opencode")["opencode"]
        self.assertEqual(a["added"], 2)   # lean: user + final answer of the one owner session; tmp, child and json-quoted dropped
        r = self.rows("opencode")
        self.assertEqual([(x[0], x[1], x[3], x[4]) for x in r], [("keep", "myproj", "user", "fix the importer"), ("keep", "myproj", "agent", "done: importer fixed")])
        self.assertTrue(r[0][2].startswith("2026-09-"))
        self.assertLess(r[0][2], r[1][2])   # each message keeps its own time
        self.assertEqual(imports.run(self.state, self.home, "opencode", full=True)["opencode"]["added"], 1)   # narration only in full

    def test_copilot(self):
        imports.run(self.state, self.home, "copilot")
        r = self.rows("copilot")
        self.assertEqual([(x[0], x[3], x[4]) for x in r], [("abc", "user", "bet model please"), ("abc", "agent", "model trained")])
        self.assertTrue(r[0][2].startswith("2026-11-09") or r[0][2].startswith("2026-11-10"))   # local zone

    def test_reimport_adds_zero(self):
        a = imports.run(self.state, self.home)
        b = imports.run(self.state, self.home)
        self.assertEqual((a["opencode"]["added"], a["copilot"]["added"]), (2, 2))
        self.assertEqual((b["opencode"]["added"], b["copilot"]["added"]), (0, 0))


if __name__ == '__main__':
    unittest.main()
