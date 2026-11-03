"""Stable-id import of chat history the live chat log does not read. Ids are content hashes (x-<sha16>), so a re-import adds 0.

Storage: state/chatlog/imports.db (x table + fts), searched together with the chat log by memrove_v3_chatsearch and opened by
`memrove zoom x-...`. A new source is one memrove_v3_reader_<name>.py with NAME, files(home), read(path)."""
import hashlib
import importlib
import re
import sqlite3
import time
from pathlib import Path

import memrove_v3_chatlog as chatlog

OPEN = "x-"


def readers():
    here = Path(__file__).parent
    return {m.NAME: m for m in (importlib.import_module(p.stem) for p in sorted(here.glob("memrove_v3_reader_*.py")))}


def _db(state):
    path = chatlog.root(state) / "imports.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("CREATE TABLE IF NOT EXISTS x(id TEXT PRIMARY KEY, src, session, project, date, kind, text, j2 INTEGER DEFAULT 0)")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS xfts USING fts5(folded, tokenize='unicode61 remove_diacritics 2')")
    return db


def make_id(src, session, kind, ts, text):
    return OPEN + hashlib.sha256(f"{src}|{session}|{kind}|{ts.isoformat()}|{text}".encode("utf-8")).hexdigest()[:16]


def _session(path):
    p = Path(path)
    return p.parents[2].name if p.name == "overview.txt" else p.stem


def _project(cwd, state):
    if not cwd:
        return ""
    from memrove_v3_project import resolve
    return resolve(cwd, state)[0] or "unknown"


def run(state, home, only=None, full=False, dry_run=False):
    """-> {source: {"read", "skipped", "added"}}. Lean (default): user turns and final agent answers; full adds the narration."""
    import memrove_v3_chatsearch as cs
    seen = chatlog._seen(chatlog.root(state))
    db = None if dry_run else _db(state)
    out = {}
    try:
        for name, rd in readers().items():
            if only and name != only:
                continue
            st = out[name] = {"read": 0, "skipped": 0, "added": 0}
            for path in rd.files(home):
                sid = _session(path)
                for m in rd.read(path):
                    if m["kind"] == "agent" and m.get("final") is False and not full:
                        continue
                    st["read"] += 1
                    big = m["kind"] == "user" and len(m["text"].encode("utf-8")) >= chatlog.PASTE_MIN
                    if chatlog._sha(m["k"]) in seen or (big and "p:" + chatlog._sha(" ".join(m["text"].split()), 24) in seen):
                        st["skipped"] += 1
                        continue
                    text = chatlog.redact(m["text"], state)[0]
                    ms = m.get("session", sid)   # a reader whose file holds many sessions names each message's session and cwd
                    xid = make_id(name, ms, m["kind"], m["ts"], m["text"])
                    if dry_run:
                        st["added"] += 1
                        continue
                    cur = db.execute("INSERT OR IGNORE INTO x(id, src, session, project, date, kind, text) VALUES (?,?,?,?,?,?,?)",
                                     (xid, name, ms, m["project"] if "project" in m else _project(m.get("cwd"), state), m["ts"].isoformat(timespec="seconds"), m["kind"], text))
                    if cur.rowcount == 1:
                        db.execute("INSERT INTO xfts(rowid, folded) VALUES (?,?)", (cur.lastrowid, cs.index_text(text)))
                        st["added"] += 1
        if db:
            db.commit()
    finally:
        if db:
            db.close()
    return out


def scored(state, q, since, until, top, cand=200, prefix=0.6):
    """-> [(bm25 or None, (id, date, kind, text))] for chatsearch._scored; q = terms(query) or None."""
    path = chatlog.root(state) / "imports.db"
    if not path.exists():
        return []
    db = sqlite3.connect(path, timeout=30)
    try:
        where, args = [], []
        if q:
            where.append("xfts MATCH ?")
        if since or until:
            where.append("x.date != ''")
        if since:
            where.append("x.date >= ?")
            args.append(since)
        if until:
            where.append("substr(x.date, 1, ?) <= ?")
            args += [len(until), until]
        w = " AND ".join(where) or "1"
        if q:
            best = {}
            for expr, weight in ((q[0], 1.0), (q[1], prefix)):
                sql = f"SELECT x.id, x.date, x.kind, x.text, bm25(xfts) FROM xfts JOIN x ON x.rowid = xfts.rowid WHERE {w} ORDER BY bm25(xfts) LIMIT ?"
                for i, d, k, t, b in db.execute(sql, (expr, *args, cand)):
                    prev = best.get(i)
                    best[i] = ((prev[0] if prev else 0.0) + weight * b, (i, d, k, t))
            return sorted(best.values(), key=lambda p: p[0])[:top]
        rows = db.execute(f"SELECT id, date, kind, text FROM x WHERE {w} ORDER BY date DESC, id DESC LIMIT ?", (*args, top)).fetchall()
        return [(None, r) for r in rows]
    finally:
        db.close()


def get(state, xid):
    """-> {"kind", "date", "text"} or None."""
    path = chatlog.root(state) / "imports.db"
    if not path.exists():
        return None
    db = sqlite3.connect(path, timeout=30)
    try:
        r = db.execute("SELECT kind, date, text FROM x WHERE id = ?", (xid,)).fetchone()
    finally:
        db.close()
    return dict(zip(("kind", "date", "text"), r)) if r else None


def _retry(fn, tries=8):
    """The hook worker can hold the store's write lock for a moment: retry "database is locked" instead of losing a paid model answer."""
    for i in range(tries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or i == tries - 1:
                raise
            time.sleep(1 + 2 * i)


def extract(engine, vault, state, max_cost=1.0, link=True):
    """J2 over imported messages not yet extracted (j2 = 0), oldest first, one session at a time, then J5 links the new records.
    Stops when the spend reaches max_cost dollars. -> {"windows", "records", "cost", "stopped", "edges"}"""
    import memrove_v3_llm as llm
    from memrove_v3_capture import has_secret
    path = chatlog.root(state) / "imports.db"
    out = {"windows": 0, "records": 0, "cost": 0.0, "stopped": "", "edges": 0}
    if not path.exists():
        return out
    db = sqlite3.connect(path, timeout=30)
    rec_ids = lambda: {r[0] for r in sqlite3.connect(Path(state) / "memory.sqlite3").execute("SELECT id FROM records")}
    before = rec_ids()
    budget = {"left": 10 ** 6, "calls": 0, "cost": 0.0}
    streak = 0
    room = llm.MAX_J2_CHARS - llm.J2_ASSISTANT_CHARS
    try:
        for (sid,) in db.execute("SELECT session FROM x WHERE j2 = 0 GROUP BY session ORDER BY min(date)").fetchall():
            rows = db.execute("SELECT id, date, kind, text FROM x WHERE session = ? ORDER BY date, rowid", (sid,)).fetchall()
            named = [p for (p,) in db.execute("SELECT project FROM x WHERE session = ? AND project NOT IN ('', 'unknown') AND project IS NOT NULL", (sid,))]
            project = max(set(named), key=named.count) if named else "unknown"
            todo = [r for r in rows if db.execute("SELECT j2 FROM x WHERE id = ?", (r[0],)).fetchone()[0] == 0]
            while todo and not out["stopped"]:
                users, ids, last, used = [], [], [], 0
                while todo:   # one window: user turns until the budget is full, the last 3 agent answers as context
                    i, d, k, t = todo[0]
                    if k == "user" and users and used + len(llm._short(t, 600)) + 3 > room:
                        break
                    todo.pop(0)
                    ids.append(i)
                    if k == "user":
                        users.append(t)
                        used += len(llm._short(t, 600)) + 3
                    else:
                        last = (last + [llm._short(t, 300)])[-3:]
                text = "USER MESSAGES:\n" + "\n".join("- " + llm._short(u, 600) for u in users) + "\n\nLAST ASSISTANT MESSAGES:\n" + "\n---\n".join(last)
                if users and not has_secret(text):
                    if budget["cost"] >= max_cost:
                        out["stopped"] = "cost"
                        break   # the window stays j2 = 0
                    st = {}
                    ev = {"project": project, "day": rows[0][1][:10]}  # the session's most common reader project; Antigravity names none -> "unknown"
                    data = llm.call(state, vault, "j2", llm._thread_prompt(vault, ev, text), None, budget, status=st, meta={"session": sid[:12], "import": True})
                    if not isinstance(data, dict):
                        if st.get("reason", "failed") != "failed":
                            out["stopped"] = st["reason"]
                            break
                        out["failed"] = out.get("failed", 0) + 1   # one window the model could not answer: skip it (marked done), keep going
                        streak += 1
                        if streak >= 3:   # three in a row is an outage, not a poison window: stop and leave the rest j2 = 0
                            out["stopped"] = "failed"
                            break
                        db.executemany("UPDATE x SET j2 = 1 WHERE id = ?", [(i,) for i in ids])
                        db.commit()
                        continue
                    streak = 0
                    _retry(lambda: llm._note(engine, state, data, text, ev, sid, str(path), len(ids), vault))
                    out["windows"] += 1
                if out["stopped"]:
                    break
                db.executemany("UPDATE x SET j2 = 1 WHERE id = ?", [(i,) for i in ids])
                db.commit()
        out["cost"] = round(budget["cost"], 4)
        new = sorted(rec_ids() - before)
        out["records"] = len(new)
        if link and new:
            import memrove_v3_graph as graph
            res = _retry(lambda: graph.link_new(engine, vault, state, limit=len(new), ids=new, budget={"left": 10 ** 6, "calls": 0, "cost": 0.0}, daily=False,
                                 token_cap=200000, max_cost=max(0.0, max_cost - budget["cost"])))
            out["edges"] = res.get("edges", 0)
    finally:
        db.close()
    return out
