"""Reader: Cursor chats. Two formats behind one NAME, told apart by file suffix.

- IDE: ~/Library/Application Support/Cursor/User/globalStorage/state.vscdb (cursorDiskKV: composerData:<cid> + bubbleId:<cid>:<bid>,
  opened read-only because Cursor may be running). Bubble type 1 = user, 2 = agent; a bubble carries its own createdAt (ISO) except
  in the oldest composers, which fall back to the composer's createdAt. Project: workspaceStorage/*/ lists a workspace's composer
  ids (ItemTable composer.composerData) and workspace.json its folder; composerData.workspaceIdentifier is the second source.
- Agent CLI: ~/.cursor/projects/<dir>/agent-transcripts/<sid>/<sid>.jsonl, rows {role, message.content[{type text|tool_use}]} with
  NO timestamps. Every transcript seen so far is also an IDE composer; files() lists only the ones the IDE db does not hold.
Messages carry their own "session" and "cwd" (the importer resolves cwd to the project name; "" when no folder is known)."""
import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

import memrove_v3_chatlog as chatlog

NAME = "cursor"
IDE = Path("Library/Application Support/Cursor/User/globalStorage/state.vscdb")


def _ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)


def _held(db_path):
    """Composer ids that have bubbles in the IDE db."""
    try:
        with _ro(db_path) as db:
            return {k.split(":")[1] for (k,) in db.execute("SELECT key FROM cursorDiskKV WHERE key LIKE 'bubbleId:%'")}
    except sqlite3.Error:
        return set()


def files(home):
    home = Path(home)
    db, out = home / IDE, []
    if db.exists():
        out.append(db)
    held = _held(db) if out else set()
    out += [p for p in sorted(home.glob(".cursor/projects/*/agent-transcripts/*/*.jsonl")) if p.stem not in held]
    return out


def read(path):
    return _read_ide(Path(path)) if Path(path).suffix == ".vscdb" else _read_transcript(Path(path))


def _turns(rows, session, cwd):
    """rows: [(ts, kind, text, has_tool, key)] in order -> msgs. Final = the last tool-free agent text of a turn."""
    out, turn = [], []

    def flush():
        last = next((m for m in reversed(turn) if m["kind"] == "agent" and not m["_tool"]), None)
        for m in turn:
            m["final"] = m["kind"] == "user" or m is last
            del m["_tool"]
        out.extend(turn)
        turn.clear()

    for ts, kind, text, tool, key in rows:
        if kind == "user":
            flush()
        turn.append({"ts": ts, "kind": kind, "text": text, "k": key, "session": session, "cwd": cwd, "_tool": tool})
    flush()
    return out


def _read_ide(path):
    ws = path.parents[1] / "workspaceStorage"   # .../User/globalStorage/state.vscdb -> .../User/workspaceStorage
    folder_of = {}
    for d in sorted(ws.glob("*/")):
        try:
            folder = json.loads((d / "workspace.json").read_text()).get("folder")
            with _ro(d / "state.vscdb") as w:
                r = w.execute("SELECT value FROM ItemTable WHERE key = 'composer.composerData'").fetchone()
            for c in json.loads(r[0]).get("allComposers", []) if r else []:
                folder_of[c["composerId"]] = unquote(folder.removeprefix("file://"))
        except (OSError, ValueError, sqlite3.Error, AttributeError, KeyError, TypeError):
            continue
    out = []
    with _ro(path) as db:
        composers = {}
        for k, v in db.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'"):
            try:
                composers[k.split(":", 1)[1]] = json.loads(v)
            except (TypeError, ValueError):
                pass
        by = {}
        for k, v in db.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'bubbleId:%' ORDER BY rowid"):
            _, cid, bid = k.split(":", 2)
            try:
                by.setdefault(cid, []).append((bid, json.loads(v)))
            except (TypeError, ValueError):
                pass
    for cid, bubbles in by.items():
        c = composers.get(cid, {})
        wi = c.get("workspaceIdentifier")
        folder = folder_of.get(cid) or (wi.get("uri", {}).get("fsPath") if isinstance(wi, dict) else None)
        base = chatlog._ts((c.get("createdAt") or 0) / 1000) if c.get("createdAt") else None   # epoch ms
        rows = []
        for bid, b in bubbles:
            text = (b.get("text") or "").strip()
            ts = chatlog._ts(b.get("createdAt")) or base
            if text and ts and b.get("type") in (1, 2):   # thinking-only and empty tool bubbles carry no text
                rows.append((ts, "user" if b["type"] == 1 else "agent", text, bool(b.get("toolFormerData")), f"cursor:{cid}:{bid}"))
        rows.sort(key=lambda r: r[0])   # stable: bubbles without their own time keep db order
        out += _turns(rows, cid, folder)
    return out


def _decode_dir(name, root=Path("/")):
    """'Users-ann-Desktop-My-App' -> /Users/ann/Desktop/'My App' by walking the disk (the dashes are ambiguous)."""
    toks = name.split("-")
    def walk(base, i):
        if i == len(toks):
            return base
        for j in range(len(toks), i, -1):
            for sep in ("-", " "):
                p = base / sep.join(toks[i:j])
                if p.exists() and (r := walk(p, j)):
                    return r
        return None
    return walk(root, 0)


def _read_transcript(path):
    st = path.stat()
    base = datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime)).astimezone()   # ponytail: rows have no time; birth time is stable when the file grows, so ids stay put; messages get +1 s each. Real times only exist in the IDE db.
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            r = json.loads(line)
            blocks = r["message"]["content"]
        except (ValueError, KeyError, TypeError):
            continue
        text = "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text").strip()
        if r.get("role") == "user":
            q = re.search(r"<user_query>\s*(.*?)\s*</user_query>", text, re.S)
            text = chatlog.clean_turn(q[1] if q else text)
        if text and r.get("role") in ("user", "assistant"):
            rows.append((base + timedelta(seconds=i), "user" if r["role"] == "user" else "agent", text,
                         any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks), f"cursor:{path.stem}:{i}"))
    folder = _decode_dir(path.parents[2].name)
    return _turns(rows, path.stem, folder)
