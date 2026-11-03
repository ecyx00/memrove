#!/usr/bin/env python3
"""ChatGPT and NotebookLM web-export readers: active branch only, final vs narration, projects, sessions, 0 added on re-import."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_imports as imports  # noqa: E402

T = 1789215147


def node(nid, parent, role, text, t, ctype="text", hidden=False):
    msg = {"author": {"role": role}, "create_time": t, "content": {"content_type": ctype, "parts": [text]}, "metadata": {"is_visually_hidden_from_conversation": hidden}}
    return nid, {"id": nid, "parent": parent, "message": msg}


def chatgpt_zip(path):
    # root -> u1 -> a1 (regenerated: a1x is the dead branch) -> u2 -> tool (hidden) -> a2 narration -> a3 final; u2 was also edited (u2x dead)
    nodes = dict([node("u1", None, "user", "first question", T), node("a1x", "u1", "assistant", "DEAD answer", T + 1),
                  node("a1", "u1", "assistant", "first answer", T + 2), node("u2x", "a1", "user", "DEAD edit", T + 3),
                  node("u2", "a1", "user", "second question", T + 4), node("tool", "u2", "tool", "ignored", T + 5),
                  node("a2", "tool", "assistant", "narration", T + 6), node("a3", "a2", "assistant", "final answer", T + 7),
                  node("sys", "a3", "assistant", "thoughts", T + 8, ctype="thoughts")])
    conv = {"conversation_id": "c1", "current_node": "a3", "mapping": nodes}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("conversations.json", json.dumps([conv]))
        z.writestr("user.json", "{}")
        z.writestr("blob.dat", b"\0" * 10)


def takeout_zip(path):
    b = "Takeout/NotebookLM/Nb/"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(b + "Nb metadata.json", json.dumps({"title": "Nb", "id": "nbid", "metadata": {"createTime": "2026-03-14T23:42:05.021369Z"}}))
        z.writestr(b + "Chat History/Chat Session - s1.html", "MODEL: <p>answer &amp; more</p>\nUSER: <p>question</p>\n")   # newest first
        z.writestr(b + "Notes/My note.html", "<p>my <b>own</b> words</p>")
        z.writestr(b + "Sources/page.html", "<p>third party</p>")
        z.writestr("Takeout/Gemini/gems.json", "{}")


class ReaderWeb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self.home, self.state = t / 'home', t / 'state'
        d = self.home / 'Desktop/web-chats'
        d.mkdir(parents=True)
        chatgpt_zip(d / 'a.zip')
        takeout_zip(d / 'takeout.zip')

    def rows(self, src):
        db = sqlite3.connect(self.state / 'chatlog/imports.db')
        return db.execute("SELECT session, project, kind, text FROM x WHERE src = ? ORDER BY date, rowid", (src,)).fetchall()

    def test_chatgpt_active_branch_only(self):
        self.assertEqual(imports.run(self.state, self.home, "chatgpt")["chatgpt"]["added"], 4)
        self.assertEqual(self.rows("chatgpt"), [("c1", "unknown", "user", "first question"), ("c1", "unknown", "agent", "first answer"),
                                                ("c1", "unknown", "user", "second question"), ("c1", "unknown", "agent", "final answer")])
        self.assertEqual(imports.run(self.state, self.home, "chatgpt", full=True)["chatgpt"]["added"], 1)   # narration only in full

    def test_notebooklm_chat_and_notes(self):
        self.assertEqual(imports.run(self.state, self.home, "notebooklm")["notebooklm"]["added"], 3)
        self.assertEqual(sorted(self.rows("notebooklm")), [("note:nbid", "Nb", "agent", "my own words"), ("s1", "Nb", "agent", "answer & more"), ("s1", "Nb", "user", "question")])

    def test_reimport_adds_zero(self):
        self.assertEqual(sum(v["added"] for v in imports.run(self.state, self.home).values() if v["added"]), 7)
        self.assertEqual(sum(v["added"] for v in imports.run(self.state, self.home).values()), 0)


if __name__ == '__main__':
    unittest.main()
