"""Deterministic transcript capture: byte-cursor read -> user quotes, open tasks, touched files -> v3 records.

Records are written through SyncEngine.note_create/task_create (validation -> intent -> commit -> readback).
Runs in the detached worker, never in the hook's own latency path. Stdlib only.
"""
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from memrove_v3_secrets import PATTERNS

MAX_FIRST_READ = 8 * 1024 * 1024
MAX_NEW = 200  # safety cap per run; sync is incremental now
TAG = re.compile(r"<(system-reminder|task-notification|command-[a-z]+|local-command-[a-z]+)>.*?</\1>", re.S)
# Cue regexes below mix Turkish and English: her zaman/asla/bundan sonra = always/never/from now on; yanlış/hayır/yapma/vazgeç/istemiyorum/olmaz = wrong/no/don't/give up/I don't want/no way; karar verdim/kararım/kullanalım/ile gidelim = I decided/my decision/let's use/let's go with; haberin olsun = for your information; tercih = preference. They match Turkish user text.
GLOBAL = re.compile(r"\b(her zaman|asla|bundan sonra|hicbir zaman|hiçbir zaman|always|never|from now on)\b", re.I)
CUES = (
    ("rejected", re.compile(r"\b(yanlış|yanlis|hayır|hayir|yapma|vazgeç|vazgec|istemiyorum|olmaz|don'?t|do not|stop (?:using|doing)|not what i|instead of|revert)\b", re.I)),
    ("decision", re.compile(r"\b(karar verdim|kararım|karar:|kullanalım|ile gidelim|let'?s go with|we(?:'ll| will) use|i decided|decided to)\b", re.I)),
    ("preference", re.compile(r"\b(her zaman|asla|bundan sonra|tercih|always|never|from now on|i prefer|prefer to|istiyorum ki|olsun)\b", re.I)),
)
DURABLE = re.compile(r"\b(her zaman|asla|hep|bundan sonra|haberin olsun|karar verdim|kararım[ıi]z|karar:|always|never|from now on)\b", re.I)
# ponytail: surface heuristics (Turkish suffix regexes), tuned on the 15-quote audit; revisit at >=50 real quotes
R1 = re.compile(r"^(?:(?:bu|şu|o)\s+\w+(?:sa|se)\b|(?:bunu|şunu|onu)\b)", re.I)  # deictic with unresolved referent
R2 = re.compile(r"\b(?:\w+(?:alım|elim)|hallet|yap|bak)\b", re.I)  # 1st-plural / imperative task directive
R3 = re.compile(r"\b(?:sıkboğaz|sıkıl|bunal|yorul)", re.I)  # mood verbs
WRITE_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")


def has_secret(text):
    return any(p.search(text) for p in PATTERNS)


def user_text(rec):
    if rec.get("type") != "user" or rec.get("isSidechain") or rec.get("isMeta") or rec.get("isCompactSummary"):
        return None
    c = (rec.get("message") or {}).get("content")
    if isinstance(c, list):
        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    if not isinstance(c, str):
        return None
    c = TAG.sub("", c).strip()
    return None if not c or c.startswith(("<", "[Request interrupted", "Caveat:")) else c


def sentences(text):
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = " ".join(s.split())
        if 8 <= len(s) <= 200 and not s.endswith("?"):
            yield s


def is_durable(quote):
    return bool(DURABLE.search(quote))


def noise(s):
    """R1 deictic-without-object, R2 task directive without durability marker, R3 mood verb."""
    return bool(R1.search(s) or R3.search(s) or (R2.search(s) and not DURABLE.search(s)))


def capture(text):
    """-> [(type, quote)] for sentences stating a decision/preference/rejection; secrets dropped."""
    out = []
    for s in sentences(text):
        if has_secret(s) or noise(s):
            continue
        for typ, rx in CUES:
            if rx.search(s):
                out.append((typ, s))
                break
    return out


def key_of(quote):
    from memrove_v3 import _tokens
    return "-".join(sorted(_tokens(quote), key=lambda t: (-len(t), t))[:3])


def tool_uses(rec):
    if rec.get("type") != "assistant":
        return
    for b in (rec.get("message") or {}).get("content") or []:
        if isinstance(b, dict) and b.get("type") == "tool_use":
            yield b.get("name"), b.get("input") or {}


def apply_tasks(rec, tasks):
    for name, i in tool_uses(rec):
        if name == "TodoWrite":
            tasks.clear()
            for k, t in enumerate(i.get("todos") or []):
                tasks[str(k)] = [t.get("content", ""), t.get("status", "pending")]
        elif name == "TaskCreate":
            tasks[f"c{len(tasks)}"] = [i.get("subject", ""), "pending"]
        elif name == "TaskUpdate" and str(i.get("taskId")) in tasks:
            tasks[str(i["taskId"])][1] = i.get("status", "pending")


def touched(rec, cwd):
    for name, i in tool_uses(rec):
        p = i.get("file_path") or i.get("notebook_path")
        if name in WRITE_TOOLS and isinstance(p, str):
            yield str(Path(p) if Path(p).is_absolute() else Path(cwd) / p)


def read_new(path, cursor):
    """Complete JSON lines after byte `cursor` (first read capped to the last 8 MB) -> (records, new_cursor)."""
    size = Path(path).stat().st_size
    if size < cursor:
        cursor = 0  # transcript was replaced
    if size == cursor:
        return [], cursor
    start = max(cursor, size - MAX_FIRST_READ) if cursor == 0 else cursor
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read()
    if start and cursor == 0:
        head, _, raw = raw.partition(b"\n")
        start += len(head) + 1
    raw = raw[: raw.rfind(b"\n") + 1]
    recs = []
    for line in raw.splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    return recs, start + len(raw)


def headless(path):
    try:
        with open(path, "rb") as f:
            return b'"entrypoint":"sdk-cli"' in f.read(16384).replace(b'": "', b'":"')
    except OSError:
        return False


def _sha(text, n):
    return hashlib.sha256(text.encode()).hexdigest()[:n]


EXTRACTOR_VERSION = "e10"


def origin_type_of(rec):
    """Provenance enum (design 12) for any record, old or new: stored value, else mapped from verified_by/origin/source."""
    if rec.get("origin_type"):
        return rec["origin_type"]
    if rec.get("verified_by") == "user_quote":
        return "user_explicit"
    if rec.get("verified_by") == "llm" or rec.get("origin") == "llm":
        return "agent_inference"
    if str(rec.get("source", "")).startswith(("docs/", "knowledge/")) or rec.get("kind") in ("doc", "reference"):
        return "derived"
    return None


# Turkish: yanlış hayır artık istemiyorum yapma olmaz vazgeç değil = wrong no anymore I don't want don't no way give up not.
OPPOSE_CUES = "yanlış hayır artık istemiyorum yapma olmaz vazgeç değil don't stop revert instead"


def same_subject(mine, key, r, theirs_text):
    """Shared subject test for opposition, corroboration and reassessment: equal key, or >=2 shared content tokens covering half the smaller side."""
    from memrove_v3 import _tokens
    theirs = _tokens(theirs_text) - _tokens(OPPOSE_CUES)
    shared = len(mine & theirs)
    return bool((key and r.get("key") == key) or (shared >= 2 and shared * 2 >= min(len(mine), len(theirs))))


def opposed(engine, project, key, polarity, own_id, quote="", same=False, kind=None):
    """Active records of the same project with the opposite polarity on the same subject -> ids.
    same=True: same polarity, user-quote records only (J2 corroboration).
    same=True with kind: active records of that kind (lesson corroboration), polarity ignored.
    Subject match: equal key, or >=2 shared content tokens covering half of the smaller side (cue words ignored)."""
    from memrove_v3 import _tokens
    mine = _tokens(quote) - _tokens(OPPOSE_CUES)
    if kind:
        cond, extra = "json_extract(payload,'$.kind')=?", kind
    else:
        cond = "json_extract(payload,'$.polarity') IS NOT NULL AND json_extract(payload,'$.polarity')" + ("=" if same else "!=") + "?"
        cond += " AND json_extract(payload,'$.verified_by')='user_quote'" if same else ""
        extra = polarity
    with engine.store._connect() as db:
        rows = [json.loads(r[0]) for r in db.execute("SELECT payload FROM records WHERE json_extract(payload,'$.project')=? AND " + cond +
                                                     " AND coalesce(json_extract(payload,'$.status'),'active') IN ('active','') AND json_extract(payload,'$.id')!=?",
                                                     (project, extra, own_id))]
    out = []
    for r in rows:
        if same_subject(mine, key, r, str(r.get("quote") or r.get("text"))):
            out.append(r["id"])
    return out


def count_used(engine, sid, recs):
    """Recall-injected record ids or note paths seen in new transcript text -> used_n + 1 each (once per scan).
    The recall block itself carries neither, so injection alone never counts as use."""
    try:
        with engine.store._connect() as db:
            ids = [r[0][7:] for r in db.execute("SELECT block_id FROM injected WHERE session=? AND block_id LIKE 'recall:%' AND block_id NOT LIKE 'recall:ent:%'", (sid,))]
            if not ids:
                return 0
            blob = "\n".join(json.dumps(r, ensure_ascii=False) for r in recs)
            from memrove_v3_projections import update_record
            n = 0
            for rid in ids:
                row = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
                rec = json.loads(row[0]) if row else {}
                if rec and ((len(rid) >= 6 and rid in blob) or (rec.get("source") and rec["source"] in blob)):
                    n += update_record(engine, rid, {"used_n": int(rec.get("used_n") or 0) + 1})
        return n
    except Exception:
        return 0  # measurement never costs the capture


def run(engine, event):
    """One queued event with transcript_path -> number of records written. Never raises on a bad transcript."""
    tp, sid = event.get("transcript_path"), event.get("session")
    if not tp or not sid or not os.path.isfile(tp):
        return 0
    state_file = engine.state / "capture-cursors.json"
    try:
        saved = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    cur = saved.get(sid, {"cursor": 0, "tasks": {}})
    recs, new_cursor = read_new(tp, cur["cursor"])
    turns = sum(1 for r in recs if r.get("type") == "assistant")
    if event.get("event") == "Stop" and turns < event.get("min_turns", 40):
        return 0  # Stop is throttled; cursor stays so the next trigger sees these turns
    count_used(engine, sid, recs)
    cwd, project = event.get("cwd") or ".", event.get("project") or "unknown"
    tasks, files, quotes, prev = cur.get("tasks", {}), [], [], ""
    for r in recs:
        apply_tasks(r, tasks)
        files.extend(p for p in touched(r, cwd) if p not in files)
        t = user_text(r)
        if t:
            quotes.extend((typ, q, prev) for typ, q in capture(t))
            prev = t
    written = 0
    today = date.today().isoformat()
    base = {"project": project, "session": sid, "verified_by": "user_quote", "status": "active", "origin_type": "user_explicit", "confidence": 1.0,
            "extractor_version": EXTRACTOR_VERSION, "valid_from": today}
    firsts = {}
    for typ, quote, p in quotes:
        firsts.setdefault((typ, quote), p)
    for (typ, quote), p in list(firsts.items())[:MAX_NEW]:
        rid = "q-" + _sha(project + typ + quote, 16)
        meta = dict(base, id=rid, kind=typ, key=key_of(quote), polarity="neg" if typ == "rejected" else "pos")
        old = opposed(engine, project, meta["key"], meta["polarity"], rid, quote)
        if old:  # a later opposite statement on the same key retires the earlier record (supersedes)
            meta["supersedes"] = old
        if is_durable(quote):
            meta["durable"] = True
        if p and not has_secret(p):
            meta["prev"] = " ".join(p.split())[:200]  # J1 judges a quote with the line before it
        try:
            engine.note_create(f"notes/captured/{rid}.md", quote, meta)
            written += 1
            if old:  # supersede closes the old record's validity window
                from memrove_v3_projections import update_record
                for oid in old:
                    update_record(engine, oid, {"valid_to": today})
        except ValueError:
            pass  # same quote already captured (source exists) or rejected by validation
    for k, (subject, status) in tasks.items():
        if status in ("completed", "done", "cancelled") or not subject.strip() or has_secret(subject):
            continue
        tid = "t-" + _sha(project + subject, 16)
        try:
            engine.task_create(f"tasks/{tid}.md", subject, {"id": tid, "title": subject[:120], "owner": "user", "project": project, "session": sid,
                                                            "status": "active" if status == "in_progress" else "inbox"})
            written += 1
        except ValueError:
            pass
    if files:
        rid = f"s-{sid[:12]}-{cur['cursor']}"
        body = "Files touched in session:\n" + "\n".join("- " + f for f in files[-60:])
        try:
            engine.note_create(f"notes/captured/{rid}.md", body, dict(base, id=rid, kind="session", files=files[-60:]))
            written += 1
        except ValueError:
            pass
    saved[sid] = {"cursor": new_cursor, "tasks": tasks}
    for old in list(saved)[:-200]:
        saved.pop(old)
    tmp = state_file.with_name(state_file.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, state_file)
    return written
