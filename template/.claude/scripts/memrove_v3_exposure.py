"""Exposure log: what memrove showed the agent or the owner, one row per item, in the memory DB (table `exposure`).
Every give path calls `log`; it fails open (a log problem never costs a hook or a command) and never stores query or record text.
session = the same 24-char hash of the host session id as `injected` and the decision log (`key`), null when the host gave none."""
import hashlib
import os
import sqlite3
import time
from pathlib import Path

SCHEMA = "CREATE TABLE IF NOT EXISTS exposure(session TEXT, ts REAL NOT NULL, path TEXT NOT NULL, item_id TEXT NOT NULL, slot TEXT)"


def key(raw):
    """Host session id -> stored session key (None stays None)."""
    return hashlib.sha256(str(raw).encode()).hexdigest()[:24] if raw else None


def env_session():
    """Session of a CLI run: the id Claude Code (or the owner's script) puts in the environment, else None."""
    return key(os.environ.get("CLAUDE_CODE_SESSION_ID") or os.environ.get("MEMROVE_SESSION"))


def qhash(query):
    return "q:" + hashlib.sha1(str(query).encode("utf-8", "replace")).hexdigest()[:12]


def log(state, path, ids, slot=None, session=None):
    """path: rules|card|jit|recall|zoom|show|search; slot: the block/section, or the query hash for search. -> rows written."""
    ids = [str(i) for i in dict.fromkeys(ids or ()) if i]
    db = Path(state) / "memory.sqlite3" if state else None
    if not ids or not db or not db.is_file():
        return 0
    try:
        conn = sqlite3.connect(db, timeout=2)
        try:
            conn.execute(SCHEMA)
            now = time.time()
            conn.executemany("INSERT INTO exposure VALUES(?,?,?,?,?)", [(session, now, path, i, slot) for i in ids])
            conn.commit()
        finally:
            conn.close()
        return len(ids)
    except Exception:
        return 0
