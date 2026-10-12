"""`memrove remember` nudge: open records of the same project that look like near-duplicates of the new text. Deterministic, no LLM, never raises."""
import contextlib
import json
import sqlite3
from pathlib import Path

KINDS = ("decision", "rejected", "preference")  # kinds that take part in supersede chains
GONE = ("retired", "rejected", "dropped_llm", "deleted_by_user", "source_missing", "superseded")


def nudge(state, project, text, exclude=None, n=3):
    """Up to n 'may replace: rec:…' lines; [] when nothing shares enough terms or on any error."""
    try:
        from memrove_v3_recall import terms
        path = Path(state) / "memory.sqlite3"
        mine = set(terms(text))
        if not mine or not path.exists():
            return []
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT payload FROM records WHERE json_extract(payload,'$.project')=? AND json_extract(payload,'$.kind') IN (?,?,?)", (project, *KINDS)).fetchall()
        recs = [json.loads(r) for (r,) in rows]
        gone = {o for p in recs for o in (p.get("supersedes") or [])}
        scored = []
        for p in recs:
            theirs = set(terms(p.get("text") or ""))
            if p["id"] == exclude or p["id"] in gone or p.get("status") in GONE or not theirs:
                continue
            shared = len(mine & theirs)
            if shared >= 3 and shared >= 0.4 * min(len(mine), len(theirs)):
                day = str(p.get("valid_from") or p.get("recorded_at") or "")[:10]
                scored.append((-shared / min(len(mine), len(theirs)), day, p["id"], " ".join(str(p["text"]).split())[:70]))
        return [f"may replace: rec:{i} ({d} {t}) (rerun with --supersedes {i})" for _, d, i, t in sorted(scored)[:n]]
    except Exception:
        return []


def topic_line(state, project, n=8):
    """'topics of <project> (reuse with --topic): a, b, c' (most recent first), or '' when it has none. Never raises."""
    try:
        from memrove_v3_heads import topics
        t = topics(state, project, n)
        return f"topics of {project} (reuse with --topic): " + ", ".join(t) if t else ""
    except Exception:
        return ""
