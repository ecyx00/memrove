"""Docs channel of `memrove search`: write-ups under <repo>/docs/{state,design,research,decisions} as handles, never bodies.

One heading section is one FTS5 row (bm25, the chat search's folding); a handle is `doc|date|path|heading|snippet`, the best section per file.
The index lives in <state>/chatlog/docs.db and is refreshed by file mtime on each search. 0 LLM tokens.
"""
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import memrove_v3_chatlog as chatlog
import memrove_v3_chatsearch as cs

SUBDIRS = ("state", "design", "research", "decisions")
ROOTS_ENV = "MEMROVE_DOC_REPOS"   # os.pathsep-separated repo dirs; default: none
SLOTS = 3
MAX_BYTES = 400_000
HEAD = re.compile(r"^#{1,4}\s+(.*)$")
DAY = re.compile(r"(\d{4}-\d{2}-\d{2})")


def repos():
    if os.environ.get(ROOTS_ENV):
        return [Path(p) for p in os.environ[ROOTS_ENV].split(os.pathsep) if p]
    return []  # nothing beyond the vault until the owner names repos


def files():
    for repo in repos():
        for sub in SUBDIRS:
            base = repo / "docs" / sub
            if base.is_dir():
                for dirpath, _, names in os.walk(base):
                    for n in names:
                        if n.endswith(".md"):
                            yield Path(dirpath) / n


def sections(text):
    """-> [(heading, body)]: text before the first heading is its own section named by the file."""
    out, head, buf, fence = [], "", [], False
    for line in text.splitlines():
        fence ^= line.lstrip().startswith("```")
        m = None if fence else HEAD.match(line)
        if m:
            out.append((head, "\n".join(buf)))
            head, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    out.append((head, "\n".join(buf)))
    return [(h, b.strip()) for h, b in out if h or b.strip()]


def _db(state):
    db = sqlite3.connect(chatlog.root(state) / "docs.db", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS f(path TEXT PRIMARY KEY, mtime REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS s(id INTEGER PRIMARY KEY, path TEXT, date TEXT, head TEXT, body TEXT)")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(head, body, tokenize='unicode61 remove_diacritics 2')")
    return db


def refresh(db):
    """Re-index files whose mtime changed, drop the vanished ones. -> files (re)indexed."""
    seen, n = {}, 0
    for p in files():
        try:
            seen[str(p)] = p.stat().st_mtime
        except OSError:
            pass
    known = dict(db.execute("SELECT path, mtime FROM f").fetchall())
    for path in set(known) - set(seen) | {p for p, m in seen.items() if known.get(p) != m}:
        for (rid,) in db.execute("SELECT id FROM s WHERE path = ?", (path,)).fetchall():
            db.execute("DELETE FROM fts WHERE rowid = ?", (rid,))
        db.execute("DELETE FROM s WHERE path = ?", (path,))
        db.execute("DELETE FROM f WHERE path = ?", (path,))
        if path not in seen:
            continue
        try:
            text = Path(path).read_text(errors="replace")[:MAX_BYTES]
        except OSError:
            continue
        m = DAY.match(Path(path).name)
        day = m.group(1) if m else datetime.fromtimestamp(seen[path]).strftime("%Y-%m-%d")
        name = Path(path).stem
        for head, body in sections(text):
            cur = db.execute("INSERT INTO s(path, date, head, body) VALUES (?,?,?,?)", (path, day, head, body))
            db.execute("INSERT INTO fts(rowid, head, body) VALUES (?,?,?)",
                       (cur.lastrowid, cs.index_text(f"{name} {head}"), cs.index_text(body)))
        db.execute("INSERT INTO f VALUES (?,?)", (path, seen[path]))
        n += 1
    db.commit()
    return n


def handles(state, query, slots=SLOTS):
    """-> [`doc|date|path|heading|snippet`], best section of each of the top `slots` files."""
    t = cs.terms(query)
    if not t:
        return []
    db = _db(state)
    try:
        refresh(db)
        rows = db.execute(
            "SELECT s.path, s.date, s.head, s.body FROM fts JOIN s ON s.id = fts.rowid WHERE fts MATCH ? "
            "ORDER BY bm25(fts, 4.0, 1.0) LIMIT 40", (" OR ".join(t),)).fetchall()
    finally:
        db.close()
    out, seen = [], set()
    for r in rows:
        if r["path"] in seen:
            continue
        seen.add(r["path"])
        snip = cs.snippet(r["body"], query, 120) if r["body"] else ""
        out.append(f"doc|{r['date']}|{r['path']}|{r['head'] or Path(r['path']).name}|{snip}")
        if len(out) == slots:
            break
    return out


def allowed(path):
    """The resolved path when it is a .md file under <repo>/docs/<SUBDIRS> of an indexed repo, else None."""
    p = Path(path).expanduser().resolve()
    if p.suffix != ".md" or not p.is_file():
        return None
    for repo in repos():
        for sub in SUBDIRS:
            if (repo / "docs" / sub).resolve() in p.parents:
                return p
    return None


def show(path, section=None):
    """The text of an indexed doc, or of one section of it (heading matched case-folded, exact first, then prefix)."""
    p = allowed(path)
    if p is None:
        raise ValueError("memrove doc opens only .md files under docs/{state,design,research,decisions} of the indexed repos")
    text = p.read_text(errors="replace")
    if not section:
        return cs.redact(text)
    want = cs.fold(section.strip().lstrip("# ").strip())
    secs = [(h, b) for h, b in sections(text) if h]
    pick = next((x for x in secs if cs.fold(x[0]) == want), None) or next((x for x in secs if cs.fold(x[0]).startswith(want)), None)
    if pick is None:
        raise ValueError("no section '%s'; headings: %s" % (section, "; ".join(h for h, _ in secs[:30])))
    return cs.redact(f"# {pick[0]}\n\n{pick[1]}")
