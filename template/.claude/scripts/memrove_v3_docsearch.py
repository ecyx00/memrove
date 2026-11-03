"""Docs channel of `memrove search`: write-ups under <repo>/docs/{state,audits,research,design,handoff,decisions} as handles, never bodies.

One heading section is one FTS5 row (bm25, the chat search's folding); a handle is `doc|date|path|heading|snippet`, the best section per file.
The index lives in <state>/chatlog/docs.db and is refreshed by file mtime on each search. 0 LLM tokens.
"""
import glob
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import memrove_v3_chatlog as chatlog
import memrove_v3_chatsearch as cs

SUBDIRS = ("state", "audits", "research", "design", "handoff", "decisions")
ROOTS_ENV = "MEMROVE_DOC_REPOS"   # os.pathsep-separated repo dirs or globs (~/Projects/*); default: none
SLOTS = 3
MAX_BYTES = 400_000
HEAD = re.compile(r"^#{1,4}\s+(.*)$")
DAY = re.compile(r"(\d{4}-\d{2}-\d{2})")


def repos():
    """A glob entry is expanded on every call, so a new repo with a docs/ folder is found without a config change."""
    out = []
    for p in os.environ.get(ROOTS_ENV, "").split(os.pathsep):
        p = os.path.expanduser(os.path.expandvars(p))
        if not p:
            continue
        if glob.has_magic(p):
            # a git worktree checkout has a `.git` file, not a directory: it is a copy of a repo already indexed
            out += [Path(m) for m in sorted(glob.glob(p)) if os.path.isdir(os.path.join(m, "docs")) and not os.path.isfile(os.path.join(m, ".git"))]
        else:
            out.append(Path(p))
    return list(dict.fromkeys(out))  # nothing beyond the vault until the owner names repos


def files():
    for repo in repos():
        for dirpath, dirs, names in os.walk(repo / "docs"):   # every .md under docs/, whatever its subfolder
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
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
    chatlog.root(state).mkdir(parents=True, exist_ok=True)
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
    if not db.in_transaction:
        db.execute("BEGIN IMMEDIATE")  # parallel searches refresh one at a time, not on the same stale file list
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
            db.execute("DELETE FROM fts WHERE rowid = ?", (cur.lastrowid,))  # an orphan fts row may hold a reused id
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
        if (repo / "docs").resolve() in p.parents:
            return p
    return None


def show(path, section=None):
    """The text of an indexed doc, or of one section of it (heading matched case-folded, exact first, then prefix)."""
    p = allowed(path)
    if p is None:
        raise ValueError("memrove doc opens only .md files under docs/{state,audits,research,design,handoff,decisions} of the indexed repos")
    text = p.read_text(errors="replace")
    if not section:
        return cs.redact(text)
    want = cs.fold(section.strip().lstrip("# ").strip())
    secs = [(h, b) for h, b in sections(text) if h]
    pick = next((x for x in secs if cs.fold(x[0]) == want), None) or next((x for x in secs if cs.fold(x[0]).startswith(want)), None)
    if pick is None:
        raise ValueError("no section '%s'; headings: %s" % (section, "; ".join(h for h, _ in secs[:30])))
    return cs.redact(f"# {pick[0]}\n\n{pick[1]}")


def _decision(path):
    """The front-matter `decision:` of a write-up cut to ~100 chars, else its first heading."""
    from memrove_v3_docs import front_matter
    try:
        text = Path(path).read_text(errors="replace")[:MAX_BYTES]
    except OSError:
        return ""
    d = front_matter(text).get("decision")
    if d:
        return "decision: " + " ".join(d.split())[:100]
    return next((h for h, _ in sections(text) if h), "")[:100]


def pointers(state, prompt, cwd=None, n=SLOTS, min_overlap=2):
    """-> up to `n` lines `path — decision: <cut>` of docs sharing >= `min_overlap` distinctive prompt words; pointers, never content.
    Words in over a fifth of the sections carry no signal and are dropped. Order: the cwd's repo, then more shared words, then newer."""
    words = [w for w in dict.fromkeys(cs.WORD.findall(cs.fold(prompt or ""))) if len(w) >= 4][:40]
    if not words:
        return []
    db = _db(state)
    try:
        refresh(db)
        total = db.execute("SELECT count(*) FROM s").fetchone()[0]
        if total < 1:
            return []
        keep = []
        for w in words:
            df = db.execute("SELECT count(*) FROM fts WHERE fts MATCH ?", (f'"{w}"',)).fetchone()[0]
            if 0 < df <= max(total // 5, 20):
                keep.append(w)
        if len(keep) < min_overlap:
            return []
        rows = db.execute("SELECT s.path, s.date, fts.head || ' ' || fts.body AS t FROM fts JOIN s ON s.id = fts.rowid WHERE fts MATCH ?",
                          (" OR ".join(f'"{w}"' for w in keep),)).fetchall()
    finally:
        db.close()
    hit = {}
    for r in rows:
        toks = set(cs.WORD.findall(r["t"]))
        e = hit.setdefault(r["path"], [r["date"], set()])
        e[1] |= {w for w in keep if w in toks}
    here = Path(cwd).resolve() if cwd else None
    home = [str(r / "docs") + os.sep for r in repos() if here and (here == r.resolve() or r.resolve() in here.parents)]
    ranked = sorted(((p, d, len(m)) for p, (d, m) in hit.items() if len(m) >= min_overlap),
                    key=lambda x: (not any(x[0].startswith(h) for h in home), -x[2], [-ord(c) for c in x[1]]))
    return [f"{p} — {_decision(p)}" for p, _, _ in ranked[:n]]
