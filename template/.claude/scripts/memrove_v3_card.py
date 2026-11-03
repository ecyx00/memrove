"""Handoff card: one short record per session (kind "card", id c-<sid12>, rewritten in place) that the next SessionStart pushes.

Drafted by a background job after J2 from the transcript, only when the owner's own text holds a decision/rejection/preference cue
(no cue = chit-chat = no model call). The sources of a card are the session's own records, found by query, never named by the model.
Push: block() reads <state>/memory.sqlite3 read-only, never raises. Kill switch: MEMROVE_CARD=0. Stdlib only.
"""
import json
import os
from pathlib import Path
import sqlite3

CAP = 1500   # bytes of the pushed block
PROTOCOL = ('Handoff: when a meaningful piece of work ends (a decision, a correction or a finished change; not chit-chat, not every turn), '
            'write your card once before the final reply: memrove card --did "<what>" --why "<why>" --open "<left open>" --next "<first step next time>" '
            '(--did/--why/--open repeatable, 1-3 short items each).\n')
SYSTEM = ("You write a handoff card for the next agent in 8 lines at most. Input: the owner's decision cues, first and last messages, last assistant text. "
          'Reply with JSON only: {"skip": false, "did": [<=3 strings], "why": [<=2], "open": [<=2], "next": "one sentence"}. '
          'English, facts only, each string under 110 chars, no padding. {"skip": true} when nothing was decided or left open.')
SOURCE_KINDS = ("decision", "rejected", "preference", "lesson")


def _cid(sid):
    return "c-" + str(sid)[:12]


def read(path):
    """-> (human turns, cue quotes [(type, quote)], last assistant text) of a whole transcript."""
    from memrove_v3_capture import user_text, assistant_text, capture
    users, cues, last = [], [], ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            u = user_text(r)
            if u:
                users.append(u)
                cues += capture(u)
            elif r.get("type") == "assistant":
                last = assistant_text(r) or last
    return users, cues, last


def _fmt(meta):
    out = []
    for key in ("did", "why", "open"):
        out += [f"- {key}: {x}" for x in meta.get(key) or []]
    if meta.get("next"):
        out.append("- next: " + meta["next"])
    return "\n".join(out)


def upsert(engine, cid, meta, sources=()):
    """Create the card or rewrite it in place; add cites edges to its sources."""
    from memrove_v3_projections import update_record
    from memrove_v3_links import Links
    text = _fmt(meta)
    full = {**meta, "id": cid, "kind": "card", "status": "active"}
    try:
        engine.note_create(f"notes/cards/{cid}.md", text, full)
    except ValueError:   # exists: rewrite in place
        update_record(engine, cid, {**full, "text": text})
    links = Links(engine.store)
    for s in sources:
        links.add(cid, s, "cites", 0.8)


def _sources(engine, sid):
    with engine.store._connect() as db:
        rows = db.execute("SELECT id FROM records WHERE json_extract(payload,'$.session')=? AND json_extract(payload,'$.kind') IN (?,?,?,?) LIMIT 8",
                          (sid, *SOURCE_KINDS)).fetchall()
    return [r[0] for r in rows]


def _short(s, n=600):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _payload(users, cues, last):
    parts = ["CUES:\n" + "\n".join(f"- {t}: {_short(q, 300)}" for t, q in cues[:12]),
             "FIRST MESSAGES:\n" + "\n".join("- " + _short(u) for u in users[:3]),
             "LAST MESSAGES:\n" + "\n".join("- " + _short(u) for u in users[3:][-10:]),
             "LAST ASSISTANT TEXT:\n" + _short(last, 1200)]
    return "\n\n".join(parts)


def _clean(data, key, n):
    return [_short(x, 120) for x in data.get(key, [])[:n] if isinstance(x, str) and x.strip()] if isinstance(data.get(key), list) else []


def draft_all(engine, vault, state, events, root=None):
    """Worker job: a card per ended session with a cue. -> {"drafted": n}. Skips cards written by the agent and sessions unchanged since the last draft."""
    import memrove_v3_llm as llm
    import memrove_v3_queue as queue
    n = 0
    for e in sorted(events, key=lambda e: e.get("at", 0)):
        sid, tp = e.get("session"), e.get("transcript_path")
        if e.get("event") != "SessionEnd" or not sid or not tp or not os.path.isfile(tp) or queue.is_subagent(sid, tp):
            continue
        cid, size = _cid(sid), os.path.getsize(tp)
        with engine.store._connect() as db:
            row = db.execute("SELECT payload FROM records WHERE id=?", (cid,)).fetchone()
        old = json.loads(row[0]) if row else {}
        if old.get("origin") == "agent" or old.get("transcript_bytes") == size:
            continue
        users, cues, last = read(tp)
        if not cues:
            continue   # chit-chat: no model call
        data = llm.call(state, vault, "card", _payload(users, cues, last), root, system=SYSTEM, meta={"session": sid[:12], "users": len(users)})
        if not isinstance(data, dict) or data.get("skip"):
            continue
        meta = {"project": e.get("project") or "unknown", "session": sid, "origin": "llm", "origin_type": "agent_inference", "valid_from": e.get("day") or "",
                "did": _clean(data, "did", 3), "why": _clean(data, "why", 2), "open": _clean(data, "open", 2), "next": _short(data.get("next") or "", 200),
                "transcript_bytes": size}
        if not (meta["did"] or meta["open"] or meta["next"]):
            continue
        upsert(engine, cid, meta, _sources(engine, sid))
        n += 1
    return {"drafted": n}


def block(state, project=None, cap=CAP):
    """The newest card of the project as pushed text ('' when off, none or on any error). Over the cap: whole lines, then 'truncated, open <id>'."""
    if os.environ.get("MEMROVE_CARD") == "0" or not project:
        return ""
    try:
        db = sqlite3.connect(f"file:{Path(state) / 'memory.sqlite3'}?mode=ro", uri=True, timeout=2)
        try:
            row = db.execute("SELECT payload FROM records WHERE json_extract(payload,'$.kind')='card' AND json_extract(payload,'$.project')=? "
                             "AND json_extract(payload,'$.status')='active' ORDER BY json_extract(payload,'$.valid_from') DESC, "
                             "json_extract(payload,'$.updated_at') DESC LIMIT 1", (project,)).fetchone()
        finally:
            db.close()
        if not row:
            return ""
        r = json.loads(row[0])
        head = f"Last session in {project} ({r.get('valid_from', '')}), handoff card {r['id']}:\n"
        tail = f"truncated, open {r['id']}\n"
        lines = [l + "\n" for l in _fmt(r).split("\n") if l]
        body, used = "", len(head.encode())
        for i, l in enumerate(lines):
            if used + len(l.encode()) + (len(tail.encode()) if i < len(lines) - 1 else 0) > cap:
                room = cap - used - len(tail.encode())
                if not body and room > 20:   # a single huge first line: cut it on a character
                    body = l.encode()[:room - 1].decode("utf-8", "ignore").rstrip() + "…\n"
                return head + body + tail
            body += l
            used += len(l.encode())
        return head + body if body else ""
    except Exception:
        return ""
