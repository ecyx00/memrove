"""Import past Codex / Antigravity sessions into the V3 store (`memrove import-sessions --harness codex|antigravity`).

Reads (never writes) the session logs. Only the user's own turns become records; model text, developer/system messages and injected
context (skills, environment, AGENTS.md, images) are dropped. The extraction is the live capture path's own pieces
(capture(), key_of, is_durable, has_secret, the `q-<sha(project+type+quote)>` id, note_create under notes/captured/), so an imported
record is the same record live capture would have written, and a quote live capture already took in the same project is not doubled.
Zero model calls and no J2 registration: the J-jobs (J1) pick the new quotes up through the normal worker under the existing daily cap.
Idempotent: same id -> note_create refuses a second time. Quotes are processed oldest first and carry their own date (valid_from);
a supersede only retires a record that is older than the quote.
"""
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CLEAN_TAG = re.compile(r"<(system-reminder|task-notification|command-[a-z]+|local-command-[a-z]+)>.*?</\1>", re.S)
INJECTED = ("<skill>", "<environment_context>", "# AGENTS.md instructions", "<turn_aborted>", "<image", "</image>", "<subagent", "<user_instructions>",
            "<permissions", "<collaboration_mode>", "[Request interrupted", "Caveat:")
REQUEST_MARK = "## My request for Codex:"
LONG_TURN = 700  # survey: turns over ~700 chars are structured briefs; from those only durable-marker quotes count
# Turkish: 'sen bir <role>' is the Turkish form of 'you are a <role>' (matches Turkish prompts).
ROLE_OPENER = re.compile(r"^\W*(you are|you're|act as|your (role|task|job) is|as an? [a-z][a-z \-]{2,40},|sen bir [^.\n]{0,60}(uzman|asistan|mühendis|ajan|mimar))", re.I)
STRUCT_LINE = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|\*\*[^*]+\*\*)")
ENGLISH = re.compile(r"\b(the|and|to|of|with|for|before|your|you|when|that|this)\b", re.I)
STRUCT_MIN, LINES_MAX = 3, 15  # >= 3 header/bullet lines, or more than 15 non-empty lines: a pasted document, not a sentence of the user's
J1_KINDS = ("decision", "preference", "rejected")
J1_BATCH, J2_WINDOW = 40, 12000 - 1100  # mirror memrove_v3_llm.MAX_J1 and MAX_J2_CHARS - J2_ASSISTANT_CHARS


def _sha(text, n=24):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _day(ts):
    """ISO string or epoch seconds -> YYYY-MM-DD (local day) or None."""
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts).date().isoformat()
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone().date().isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def _jsonl(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for n, raw in enumerate(f, 1):
                try:
                    r = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    yield n, r
    except OSError:
        return


def clean_turn(text):
    """Own words of one user message, or None when it is injected context."""
    text = CLEAN_TAG.sub("", text or "").strip()
    if REQUEST_MARK in text:
        text = text.split(REQUEST_MARK, 1)[1].strip()
    elif text.lstrip().startswith("# Files mentioned by the user"):
        return None
    return None if not text or text.startswith(INJECTED) else text


# ---- readers: each yields {"sid","path","cwd","project_paths","turns":[(text, day, line)]} --------------------------

def codex_sessions(home):
    root = Path(home) / ".codex"
    seen = set()
    for f in sorted((root / "sessions").rglob("*.jsonl")):
        s = {"sid": None, "path": str(f), "cwd": None, "paths": [], "turns": [], "skipped": None}
        day0 = None
        for n, r in _jsonl(f):
            p = r.get("payload") if isinstance(r.get("payload"), dict) else {}
            if r.get("type") == "session_meta":
                s["sid"], s["cwd"], day0 = p.get("id") or p.get("session_id"), p.get("cwd"), _day(p.get("timestamp"))
                src = p.get("source")
                if p.get("thread_source") == "subagent" or (isinstance(src, dict) and "subagent" in src):
                    s["skipped"] = "subagent"
            elif r.get("type") == "turn_context" and not s["cwd"]:
                s["cwd"] = p.get("cwd")
            elif r.get("type") == "response_item" and p.get("type") == "message" and p.get("role") == "user":
                parts = [clean_turn(b.get("text", "")) for b in p.get("content") or [] if isinstance(b, dict) and b.get("type") == "input_text"]
                t = "\n".join(x for x in parts if x)
                if t:
                    s["turns"].append((t, _day(r.get("timestamp")) or day0, n))
        s["sid"] = s["sid"] or f.stem
        seen.add(s["sid"])
        yield s
    # history.jsonl: prompts of sessions that have no rollout file left (no cwd -> project unknown)
    by = {}
    for n, r in _jsonl(root / "history.jsonl"):
        sid, t = r.get("session_id"), clean_turn(r.get("text"))
        if sid and sid not in seen and t:
            by.setdefault(sid, []).append((t, _day(r.get("ts")), n))
    for sid, turns in by.items():
        yield {"sid": sid, "path": str(root / "history.jsonl"), "cwd": None, "paths": [], "turns": turns, "skipped": None, "history_only": True}


def _arg_paths(args):
    for v in (args or {}).values():
        v = v.strip().strip('"') if isinstance(v, str) else ""
        if v.startswith("/") and "\n" not in v and len(v) < 400:
            yield v


def antigravity_sessions(home):
    brain = Path(home) / ".gemini" / "antigravity" / "brain"
    for d in sorted(brain.glob("*")):
        files = sorted((d / ".system_generated" / "logs").glob("transcript*.jsonl"))
        if not files:
            continue
        s = {"sid": d.name, "path": str(files[0]), "cwd": None, "paths": [], "turns": [], "skipped": None}
        steps = set()
        for f in files:
            for n, r in _jsonl(f):
                key = r.get("step_index")
                if key in steps:
                    continue
                steps.add(key)
                for c in r.get("tool_calls") or []:
                    s["paths"] += list(_arg_paths(c.get("args")))
                if r.get("source") == "USER_EXPLICIT" and r.get("type") == "USER_INPUT":
                    t = r.get("content") or ""
                    m = re.search(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", t, re.S)
                    t = clean_turn(m[1] if m else re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", "", t, flags=re.S))
                    if t:
                        s["turns"].append((t, _day(r.get("created_at")), n))
        s["turns"].sort(key=lambda x: x[1] or "")
        yield s


READERS = {"codex": codex_sessions, "antigravity": antigravity_sessions}


# ---- project -------------------------------------------------------------------------------------------------------

def project_of(s, state, vault):
    """cwd (Codex) through the live resolver; Antigravity has none, so a vote of the resolver over absolute tool-argument paths
    (>= 3 hits and a 60% share, like memrove_v3_project._pick); otherwise `unknown`."""
    from memrove_v3_project import resolve
    if s["cwd"]:
        return resolve(s["cwd"], state, vault)[0] or "unknown"
    votes = Counter()
    for p in s["paths"]:
        q = Path(p)
        name = resolve(q if q.is_dir() else q.parent, state, vault)[0]
        if name:
            votes[name] += 1
    if votes:
        name, n = votes.most_common(1)[0]
        if n >= 3 and n >= 0.6 * sum(votes.values()):
            return name
    return "unknown"


# ---- collect -------------------------------------------------------------------------------------------------------

def pasted_reason(turn):
    """Why a user turn is a pasted prompt/document and not the user's own words (None: it is theirs)."""
    lines = [x for x in turn.splitlines() if x.strip()]
    if ROLE_OPENER.search(turn):
        return "role-opener"
    if sum(1 for x in lines if STRUCT_LINE.match(x)) >= STRUCT_MIN:
        return "markdown-structure"
    if len(lines) > LINES_MAX:
        return "many-lines"
    return None


def english_paste(quote):
    """The user writes Turkish (98%); an ASCII-only quote with several English function words is pasted text."""
    return quote.isascii() and len(ENGLISH.findall(quote)) >= 2


def collect(harness, home, state, vault):
    """-> (quotes, stats). quotes: dicts for every capture() hit (first occurrence per project/type/quote, oldest first)."""
    import memrove_v3_capture as cap
    quotes, firsts, seen_in = [], {}, {}
    stats = {"before_filter": Counter(), "dropped": Counter(), "sessions": 0, "sessions_skipped": Counter(), "history_only_sessions": 0, "user_turns": 0, "turns_by_project": Counter(), "j2_windows": 0}
    for s in READERS[harness](home):
        if s["skipped"]:
            stats["sessions_skipped"][s["skipped"]] += 1
            continue
        if not s["turns"]:
            stats["sessions_skipped"]["no-user-turns"] += 1
            continue
        project = project_of(s, state, vault)
        stats["sessions"] += 1
        stats["history_only_sessions"] += bool(s.get("history_only"))
        stats["user_turns"] += len(s["turns"])
        stats["turns_by_project"][project] += len(s["turns"])
        used = 0
        for t, _, _ in s["turns"]:  # the J2 windowing a session would need if it were queued (informational, J2 is not used)
            w = min(len(" ".join(t.split())), 600) + 3
            if used and used + w > J2_WINDOW:
                stats["j2_windows"] += 1
                used = 0
            used += w
        stats["j2_windows"] += 1
        prev = ""
        for t, day, line in s["turns"]:
            hits = cap.capture(t)
            if len(t) > LONG_TURN:
                hits = [(typ, q) for typ, q in hits if cap.is_durable(q)]  # a long brief/log: its task constraints ("do not ...") are not standing preferences
            stats["before_filter"][harness] += len(hits)
            why = pasted_reason(t) if hits else None
            if why:
                stats["dropped"][why] += len(hits)
                hits = []
            for typ, q in hits:
                if english_paste(q):
                    stats["dropped"]["english-paste"] += 1
                    continue
                norm = " ".join(q.casefold().split())
                seen_in.setdefault(norm, set()).add(s["sid"])
                k = (project, typ, q)
                if k not in firsts or (day or "") < (firsts[k]["day"] or ""):
                    rec = {"project": project, "type": typ, "quote": q, "prev": prev, "day": day, "harness": harness, "session": _sha(s["sid"]),
                           "source_path": s["path"], "source_line": line, "norm": norm}
                    if k in firsts:
                        quotes[quotes.index(firsts[k])] = rec
                    else:
                        quotes.append(rec)
                    firsts[k] = rec
            prev = t
    repeated = [q for q in quotes if len(seen_in[q["norm"]]) >= 2]  # the same words in several sessions: a pasted prompt, not a remark
    stats["dropped"]["repeats-across-sessions"] += len(repeated)
    quotes = [q for q in quotes if len(seen_in[q["norm"]]) < 2]
    quotes.sort(key=lambda r: (r["day"] or "", r["source_line"]))
    return quotes, stats


def rid_of(q):
    return "q-" + hashlib.sha256((q["project"] + q["type"] + q["quote"]).encode()).hexdigest()[:16]


def _existing(state):
    """ids already in the store, read-only (a dry-run never opens the engine)."""
    import sqlite3
    db_path = Path(state) / "memory.sqlite3"
    if not db_path.is_file():
        return set()
    db = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
    try:
        return {r[0] for r in db.execute("SELECT id FROM records")}
    except sqlite3.Error:
        return set()
    finally:
        db.close()


# ---- run -----------------------------------------------------------------------------------------------------------

def run(harness, vault, state, dry_run=False, home=None, sample=5):
    import memrove_v3_capture as cap
    home = Path(home) if home else Path.home()
    vault, state = Path(vault).resolve(), Path(state)
    quotes, st = collect(harness, home, state, vault)
    have = _existing(state)
    fresh = [q for q in quotes if rid_of(q) not in have]
    by_kind, by_project = Counter(q["type"] for q in quotes), Counter(q["project"] for q in quotes)
    j1 = sum(1 for q in fresh if q["type"] in J1_KINDS)
    report = {"harness": harness, "dry_run": dry_run, "sessions": st["sessions"], "sessions_skipped": dict(st["sessions_skipped"]),
              "history_only_sessions": st["history_only_sessions"], "user_turns": st["user_turns"],
              "user_turns_by_project": dict(sorted(st["turns_by_project"].items())),
              "records": len(quotes), "by_kind": dict(sorted(by_kind.items())), "by_project": dict(sorted(by_project.items())),
              "by_kind_project": {f"{k}/{p}": n for (k, p), n in sorted(Counter((q["type"], q["project"]) for q in quotes).items())},
              "rules_durable_quotes": sum(cap.is_durable(q["quote"]) for q in quotes),
              "before_pasted_filter": st["before_filter"][harness], "dropped_as_pasted": dict(st["dropped"]),
              "already_in_store": len(quotes) - len(fresh), "awaits_j1": "imported records stay out of RULES/recall until J1 judged them durable",
              "j_jobs": {"calls_made_by_import": 0, "j1_calls_to_assess_new": math.ceil(j1 / J1_BATCH), "j1_batch": J1_BATCH,
                         "j2_windows_if_queued": st["j2_windows"], "j2_registered": False},
              "date_range": [min((q["day"] for q in quotes if q["day"]), default=None), max((q["day"] for q in quotes if q["day"]), default=None)],
              "sample": [{"kind": q["type"], "project": q["project"], "day": q["day"], "text": q["quote"][:100]} for q in quotes[:sample]]}
    if dry_run:
        return report
    from memrove_v3_sync import SyncEngine
    from memrove_v3_projections import update_record
    engine = SyncEngine(vault, state)
    created = exists = superseded = 0
    for q in quotes:
        rid, today = rid_of(q), q["day"] or datetime.now(timezone.utc).date().isoformat()
        meta = {"project": q["project"], "session": q["session"], "verified_by": "user_quote", "status": "active", "origin_type": "user_explicit",
                "confidence": 1.0, "extractor_version": cap.EXTRACTOR_VERSION, "valid_from": today, "id": rid, "kind": q["type"],
                "key": cap.key_of(q["quote"]), "polarity": "neg" if q["type"] == "rejected" else "pos",
                "imported_from": f"{harness}-session", "harness": harness, "source_path": q["source_path"], "source_line": q["source_line"]}
        older, newer = [], []
        with engine.store._connect() as db:
            for oid in cap.opposed(engine, q["project"], meta["key"], meta["polarity"], rid, q["quote"]):
                row = db.execute("SELECT payload FROM records WHERE id=?", (oid,)).fetchone()
                vf = str(json.loads(row[0]).get("valid_from", "")) if row else ""
                (older if vf < today else newer).append((oid, vf))
        if older:
            meta["supersedes"] = [o for o, _ in older]
        if newer:  # a newer opposite statement already exists: this one is history from the start
            meta["valid_to"] = min(v for _, v in newer)
        if cap.is_durable(q["quote"]):
            meta["durable"] = True
        if q["prev"] and not cap.has_secret(q["prev"]):
            meta["prev"] = " ".join(q["prev"].split())[:200]
        try:
            engine.note_create(f"notes/captured/{rid}.md", q["quote"], meta)
            created += 1
            for oid, _ in older:
                update_record(engine, oid, {"valid_to": today})
                superseded += 1
        except ValueError:
            exists += 1
    report.update(created=created, already_imported=exists, superseded=superseded)
    return report
