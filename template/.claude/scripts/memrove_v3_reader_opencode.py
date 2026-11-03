"""Reader: opencode (~/.local/share/opencode/opencode.db, read-only). One file holds every session, so each message names its own session and cwd.

Only sessions where the owner typed are read. Dropped: a cwd under a temp dir (scout, probes, smoke runs), child sessions
(parent_id set) and sessions whose first user text is a JSON-quoted string (a program's json.dumps prompt)."""
import json
import sqlite3
from pathlib import Path

import memrove_v3_chatlog as chatlog

NAME = "opencode"
TMP = ("/tmp", "/private/tmp", "/private/var/folders", "/var/folders")


def files(home):
    p = Path(home) / ".local/share/opencode/opencode.db"
    return [p] if p.exists() else []


def _texts(db, mid):
    parts = (json.loads(r[0]) for r in db.execute("SELECT data FROM part WHERE message_id = ? ORDER BY time_created, id", (mid,)))
    return [p["text"].strip() for p in parts if p.get("type") == "text" and not p.get("synthetic") and (p.get("text") or "").strip()]


def read(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = []
    try:
        for sid, cwd, parent in db.execute("SELECT id, directory, parent_id FROM session ORDER BY time_created").fetchall():
            if parent or (cwd or "").startswith(TMP):
                continue
            turns = []   # (role, time, id, texts)
            for mid, t, data in db.execute("SELECT id, time_created, data FROM message WHERE session_id = ? ORDER BY time_created, id", (sid,)):
                role = json.loads(data).get("role")
                if role in ("user", "assistant") and (texts := _texts(db, mid)):
                    turns.append((role, t, mid, texts))
            first = next((x[3][0] for x in turns if x[0] == "user"), None)
            if first is None or first.startswith('"'):
                continue
            for i, (role, t, mid, texts) in enumerate(turns):
                # the last assistant message before the next user message is the final answer; earlier ones are narration
                last = role == "assistant" and (i + 1 == len(turns) or turns[i + 1][0] == "user")
                out.append({"ts": chatlog._ts(t / 1000), "kind": "user" if role == "user" else "agent", "text": "\n".join(texts),
                            "k": f"o:{mid}", "final": role == "user" or last, "session": sid, "cwd": cwd})
    finally:
        db.close()
    return out
