"""Topic heads: the newest entry per (project, topic) over decision/rejected records that carry a topic and over docs/state write-ups.
A deterministic view rebuilt on each call (no table, no LLM): the writer chose the key, so "current" is structural. Older entries on a key
render as replaced with a pointer to the head; nothing is deleted. An agent entry never replaces a --user record. Never raises."""
import contextlib
import json
import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

KINDS = ("decision", "rejected")
GONE = ("retired", "dropped_llm", "deleted_by_user", "source_missing")
DAY = re.compile(r"^(\d{4}-\d{2}-\d{2})-?(.*)$")
READ = 16000  # bytes of a write-up read for its front matter, title and verdict section
EXP = re.compile(r"^(e\d+)(?:-|$)")  # experiment prefix groups front-matter-less files: e87-arms, e87-job-set -> e87
VERDICT = re.compile(r"verdict|result|outcome|summary|decision|finding", re.I)
NEXT = re.compile(r"next|open|follow", re.I)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")


def _writeup_paths(project):
    import memrove_v3_docsearch as ds
    for repo in ds.repos():
        base = repo / "docs" / "state"
        if repo.name == project and base.is_dir():
            for p in sorted(base.rglob("*.md")):
                yield p, base


def _under(body, heading_rx, fallback=False):
    """First sentence or bullet under the first heading matching heading_rx (fallback: first non-heading line), cut to 200 chars."""
    lines, take = body.splitlines(), False
    for ln in lines:
        if ln.startswith("#"):
            take = bool(heading_rx.search(ln.lstrip("# ")))
            continue
        t = ln.strip().lstrip("-*0123456789.) ").strip()
        if take and t and not t.startswith(("```", "|")):
            return re.split(r"(?<=[.!?])\s", t, 1)[0][:200]
    if fallback:
        for ln in lines:
            t = ln.strip().lstrip("-*0123456789.) ").strip()
            if t and not t.startswith(("#", "```", "|")):
                return re.split(r"(?<=[.!?])\s", t, 1)[0][:200]
    return ""


def _writeup(path, project, base):
    import memrove_v3_docs as docs
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            text = f.read(READ)
        mtime = path.stat().st_mtime
    except OSError:
        return None
    fm = docs.front_matter(text)
    m = DAY.match(path.stem)
    day = m.group(1) if m else date.fromtimestamp(mtime).isoformat()
    body = re.sub(r"\A---\r?\n.*?\r?\n---[ \t]*\r?\n", "", text, flags=re.S)
    title = docs._title(path.name, body)
    stem = slug(m.group(2) if m else path.stem)
    topic = slug(fm.get("topic")) or (EXP.match(stem).group(1) if EXP.match(stem) else stem)
    return {"id": str(path), "ref": "docs/state/" + path.relative_to(base).as_posix(), "kind": "writeup", "project": project, "topic": topic,
            "keyed": bool(fm.get("topic")), "status": fm.get("status", ""), "decision": fm.get("decision") or _under(body, VERDICT, True) or title, "next": fm.get("next") or _under(body, NEXT),
            "supersedes": [fm["supersedes"]] if fm.get("supersedes") else [], "day": day, "ts": "", "user": False}


def _records(state):
    path = Path(state) / "memory.sqlite3"
    if not path.exists():
        return []
    with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)) as db:
        rows = db.execute("SELECT payload FROM records WHERE json_extract(payload,'$.kind') IN (?,?) AND json_extract(payload,'$.topic') IS NOT NULL", KINDS).fetchall()
    out = []
    for (raw,) in rows:
        r = json.loads(raw)
        if r.get("status") in GONE:
            continue
        why = " ".join(str(r.get("why") or "").split())
        out.append({"id": "rec:" + r["id"], "ref": "rec:" + r["id"], "kind": r["kind"], "project": r.get("project") or "", "topic": slug(r["topic"]), "keyed": True,
                    "status": "rejected" if r["kind"] == "rejected" else "", "decision": " ".join(str(r.get("text") or "").split()) + (f" (why: {why})" if why else ""),
                    "next": "", "supersedes": ["rec:" + s.removeprefix("rec:") for s in r.get("supersedes") or []],
                    "day": str(r.get("valid_from") or r.get("recorded_at") or "")[:10], "ts": str(r.get("recorded_at") or ""),
                    "user": r.get("origin_type") == "user_explicit"})
    return out


def entries(state, project=None):
    """Keyed entries of one project (or all): decision/rejected records with a topic + docs/state write-ups."""
    out = [e for e in _records(state) if project in (None, e["project"])]
    import memrove_v3_docsearch as ds
    names = {project} if project else {r.name for r in ds.repos()}
    for name in sorted(names):
        out += [e for e in (_writeup(p, name, b) for p, b in _writeup_paths(name)) if e]
    return out


def build(items):
    """-> (heads newest first, replaced {entry id: head/superseder id}). Pure."""
    order = lambda e: (e["day"], e["ts"], e["id"])
    groups, replaced = {}, {}
    for e in items:
        groups.setdefault((e["project"], e["topic"]), []).append(e)
    heads = []
    for g in groups.values():
        head = max([e for e in g if e["user"]] or g, key=order)  # an agent entry never replaces a --user one
        heads.append(head)
        replaced.update({e["id"]: head["id"] for e in g if e is not head and order(e) < order(head)})
    by_id = {e["id"]: e for e in items}
    for e in sorted(items, key=order):  # explicit supersedes: record ids, write-up paths or topics
        for target in e["supersedes"]:
            hit = [t for t in items if t["id"] == target or t["id"].endswith("/" + target.lstrip("/")) or (t["topic"] == slug(target) and t["project"] == e["project"])]
            for t in hit:
                if t is not e and not (t["user"] and not e["user"]) and not (t["topic"] == e["topic"] and order(t) > order(e)):
                    replaced[t["id"]] = e["id"]
    heads = [h for h in heads if h["id"] not in replaced]
    return sorted(heads, key=order, reverse=True), replaced


def heads(state, project=None):
    try:
        return build(entries(state, project))
    except Exception:
        return [], {}


def line(h):
    bits = [f"[{h['topic']}]"] + ([h["status"]] if h["status"] else []) + ([h["kind"]] if h["kind"] in KINDS else [])
    text = " ".join(h["decision"].split())
    text = text if len(text) <= 200 else text[:199].rstrip() + "…"
    return f"  {' '.join(bits)} {h['day']} {text}" + (f" | next: {' '.join(h['next'].split())[:100]}" if h["next"] else "") + f" | {h['ref']}"


def render(state, project, count=12):
    """Lines for `memrove project`: the newest `count` heads of the project. [] when none."""
    hs, _ = heads(state, project)
    return ["heads (newest entry per topic; older ones are replaced):"] + [line(h) for h in hs[:count]] if hs else []


def topics(state, project, n=8):
    """Existing topic slugs of a project, most recent first (for the remember nudge)."""
    hs, _ = heads(state, project)
    return [h["topic"] for h in hs if h["keyed"]][:n]


def match(state, query, n=2):
    """Heads whose words cover the query terms (>= 2 shared, or all when the query is short): [head]."""
    from memrove_v3_recall import terms
    q = set(terms(query))
    if not q:
        return []
    hs, _ = heads(state)
    scored = []
    for h in hs:
        shared = len(q & set(terms(f"{h['topic']} {h['decision']} {h['next']}")))
        if shared >= min(2, len(q)) and shared >= 0.5 * len(q):
            scored.append((-shared, h["day"], h))
    return [h for _, _, h in sorted(scored, key=lambda s: (s[0], s[1]))[:n]]


def topicless_recent(days=7, today=None):
    """-> (write-ups of the last `days` days, of those without a `topic` front-matter key), over every docs repo."""
    import memrove_v3_docsearch as ds
    since = ((today or date.today()) - timedelta(days=days)).isoformat()
    total = bare = 0
    for repo in ds.repos():
        for e in (_writeup(p, repo.name, b) for p, b in _writeup_paths(repo.name)):
            if e and e["day"] >= since:
                total += 1
                bare += not e["keyed"]
    return total, bare
