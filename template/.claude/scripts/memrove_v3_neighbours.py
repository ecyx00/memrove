"""Link-graph neighbours for `memrove search` and `memrove show`: read-only over graph_edges, no model call.

`neighbours(state, ids)` -> {rec id: [line, ...]}: at most PER_HIT compact lines per hit, `> type handle summary - why`, edges that
replace, contradict or answer first, then part_of, then relates. `newer(state, ids)` -> {old id: newer live id} from supersedes edges.
`edges_of(state, id)` -> every edge of one record with its why (for `memrove show`). Missing tables or store -> nothing.
"""
import json
import sqlite3
from datetime import datetime, timezone

import memrove_v3_chatlog as chatlog

PER_HIT = 2
SUMMARY = 70
WHY = 50
ORDER = {"supersedes": 0, "contradicts": 0, "answers": 0, "part_of": 1, "relates": 2}
OFF = frozenset(("retired", "rejected", "dropped_llm", "deleted_by_user", "source_missing"))


def _clip(text, n):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _edges(db, rid):
    """-> [(type, direction, other id, why)], direction 'out' when rid is the source."""
    rows = db.execute("SELECT src,dst,type,why FROM graph_edges WHERE src=? OR dst=?", (rid, rid)).fetchall()
    return [(t, "out" if s == rid else "in", d if s == rid else s, w) for s, d, t, w in rows]


def replaced_line(db, rid):
    """The fixed boundary line for a record a live record replaces (its `supersedes` holds it), else ''. Ids and a date only, no record text."""
    from memrove_v3_nudge import GONE
    rows = [json.loads(p) for (p,) in db.execute("SELECT payload FROM records WHERE id<>? AND EXISTS (SELECT 1 FROM json_each(payload,'$.supersedes') j WHERE j.value=?)", (rid, rid))]
    live = sorted((str(r.get("valid_from") or r.get("created_at") or "")[:10], r["id"]) for r in rows if r.get("status") not in GONE)
    return f"REPLACED by {', '.join(i for _, i in live)} on {live[0][0]}: history, do not use as current." if live else ""


def _node(db, oid):
    """-> (handle, summary) of a live record or a concept, else None (missing, retired, private, expired)."""
    if oid.startswith("concept:"):
        row = db.execute("SELECT name FROM graph_nodes WHERE id=?", (oid,)).fetchone()
        return (oid, row[0]) if row else None
    row = db.execute("SELECT payload FROM records WHERE id=?", (oid,)).fetchone()
    if not row:
        return None
    r = json.loads(row[0])
    if r.get("status") in OFF or r.get("verdict") == "reject" or r.get("visibility") == "private":
        return None
    if str(r.get("valid_to") or "") and str(r["valid_to"]) <= datetime.now(timezone.utc).replace(tzinfo=None).isoformat():
        return None
    sup = getattr(db, "sup", None)   # ids some record's `supersedes` names, scanned once per _run (a bare connection checks every record)
    return "rec:" + oid, (sup is None or oid in sup) and replaced_line(db, oid) or _clip(r.get("title") or r.get("text") or r.get("approach"), SUMMARY)   # a replaced record shows the boundary line, not its text


def _label(t, d):
    return "replaced-by" if (t, d) == ("supersedes", "in") else t


class _DB(sqlite3.Connection):
    sup = None


def _run(state, fn):
    path = chatlog.root(state).parent / "memory.sqlite3"
    if not path.exists():
        return None
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30, factory=_DB)
        try:
            db.sup = {v for (v,) in db.execute("SELECT j.value FROM records, json_each(records.payload,'$.supersedes') j")}
            return fn(db)
        finally:
            db.close()
    except sqlite3.Error:   # no graph tables yet
        return None


def neighbours(state, ids, per_hit=PER_HIT, known=()):
    def run(db):
        out = {}
        for hit in ids:
            found = []
            for t, d, oid, why in sorted(_edges(db, hit.removeprefix("rec:")), key=lambda e: (ORDER.get(e[0], 3), e[1] != "in", e[2])):
                node = _node(db, oid)
                if node and node[0] not in ids and node[0] not in known and not (t == "supersedes" and d == "in"):   # a newer record is shown as a hit, not a neighbour
                    found.append(f"  > {_label(t, d)} {node[0]} {node[1]} - {_clip(why, WHY)}")
            if found:
                out[hit] = found[:per_hit]
        return out
    return _run(state, run) or {}


def newer(state, ids):
    def run(db):
        out = {}
        for hit in ids:
            for t, d, oid, _ in _edges(db, hit.removeprefix("rec:")):
                if (t, d) == ("supersedes", "in") and _node(db, oid):
                    out[hit] = "rec:" + oid
        return out
    return _run(state, run) or {}


def edges_of(state, rid):
    def run(db):
        out = []
        for t, d, oid, why in sorted(_edges(db, rid), key=lambda e: (ORDER.get(e[0], 3), e[2])):
            node = _node(db, oid)
            out.append({"type": t, "dir": d, "id": node[0] if node else oid, "summary": node[1] if node else "(gone)", "why": why})
        return out
    return _run(state, run) or []


def replaced(state, rid):
    """replaced_line of one record (for `memrove show`), '' when it has no live replacer or on any error."""
    return _run(state, lambda db: replaced_line(db, rid.removeprefix("rec:"))) or ""


def record_row(state, rid):
    """-> (id, valid_from, kind, text) of a record, as a search row."""
    def run(db):
        r = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid.removeprefix("rec:"),)).fetchone()[0])
        return rid, str(r.get("valid_from") or "")[:10], r.get("kind") or "", str(r.get("text") or r.get("title") or r.get("approach") or "")
    return _run(state, run)
