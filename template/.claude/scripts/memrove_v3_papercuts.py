"""Harness papercuts: small frictions (failed or retried tool call, misleading error, setup step) collected without model discipline.
Sources: deterministic scan of a session transcript and its subagents/*.jsonl, plus the J2 `papercuts` list (higher rank weight).
Store-only records (no per-record vault file), one per (project, signature); a recurrence in a new session bumps count. Projected to memrove/papercuts/<project|harness>.md."""
import hashlib
import json
from pathlib import Path
import re

def _dump(r):
    return json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


WEIGHT = {"det": 1.0, "j2": 2.0}
# expected, not frictions: user declined, interrupted
DECLINED = re.compile(r"doesn't want to proceed|user rejected|request interrupted|\[Request interrupted", re.I)
TESTRUN = re.compile(r"unittest|pytest|\btests?\b.*\.(?:sh|py)|(?:npm|pnpm|yarn|cargo|go|swift|make)\s+(?:run\s+)?test|jest|vitest")
NOMATCH = ("grep", "rg", "diff", "cmp", "test", "[")
ERRLINE = re.compile(r"error|exception|not found|no such|denied|invalid|failed|cannot|can't|unable|unrecognized|usage:|blocked|traceback|refused|timed out|fatal", re.I)
HARNESS = re.compile(r"<tool_use_error>|auto mode classifier|No such tool|hook (?:error|blocked)", re.I)


def _text(content):
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content if isinstance(content, str) else ""


def _excerpt(text):
    """Most telling line: the exception of a traceback, else the first line that is not the exit-code header."""
    lines = [l.strip() for l in re.sub(r"</?tool_use_error>", "", text).splitlines() if l.strip() and not re.fullmatch(r"Exit code \d+", l.strip())]
    if not lines:
        return ""
    if any(l.startswith("Traceback") for l in lines):
        return lines[-1][:140]
    return next((l for l in lines if ERRLINE.search(l)), lines[0])[:140]


def signature(tool, excerpt):
    s = re.sub(r"(?:/[\w.@+-]+)+|toolu_\w+|\b[0-9a-f]{7,}\b|\d+", lambda m: "<p>" if m.group(0).startswith("/") else "#", excerpt.lower())
    return f"{tool}: {' '.join(s.split())[:28]}"


def _expected(tool, inp, text):
    if DECLINED.search(text):
        return True
    if tool == "Bash":
        cmd = str(inp.get("command", "")).strip()
        first = re.split(r"\s+", re.sub(r"^(?:cd\s+\S+\s*(?:&&|;)\s*)", "", cmd))[0] if cmd else ""
        if not _excerpt(text) and text.startswith("Exit code"):
            return True  # silent non-zero: grep without a match, diff, test -f
        if text.startswith("Exit code 1\n") and not ERRLINE.search(text):
            return True  # exit 1 that says nothing error-like: a deliberate check, not a friction
        return bool(TESTRUN.search(cmd)) or first in NOMATCH and text.count("\n") < 2
    return False


def extract(transcript):
    """Deterministic scan of one transcript file and <stem>/subagents/*.jsonl -> [{tool, excerpt, retry, harness, source}]."""
    p = Path(transcript)
    files = [(p, "main")] + [(f, f.stem.removeprefix("agent-")) for f in sorted((p.with_suffix("") / "subagents").glob("*.jsonl"))]
    out = []
    for f, source in files:
        calls, errors = [], []  # calls: [(id, name, input)] in order; errors: (call index, text)
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            content = (rec.get("message") or {}).get("content")
            for b in content if isinstance(content, list) else []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    calls.append((b.get("id"), b.get("name"), b.get("input") or {}))
                elif b.get("type") == "tool_result" and b.get("is_error"):
                    idx = next((i for i, c in enumerate(calls) if c[0] == b.get("tool_use_id")), None)
                    if idx is not None:
                        errors.append((idx, _text(b.get("content"))))
        for idx, text in errors:
            _, tool, inp = calls[idx]
            if _expected(tool, inp, text):
                continue
            ex = _excerpt(text) or "(no message)"
            out.append({"tool": tool, "excerpt": ex, "retry": idx + 1 < len(calls) and calls[idx + 1][1] == tool, "source": source,
                        "harness": bool(HARNESS.search(text))})
    return out


def record(engine, project, sid, day, tool, excerpt, source, retry=False, harness=False, cause="", n=1, tp=None):
    """Create the papercut record, or bump count when this session has not reported it yet. -> True when new or bumped."""
    from memrove_v3_capture import has_secret
    from memrove_v3_projections import _slug
    from memrove_v3_trust import with_transcript
    if has_secret(excerpt + cause):
        return False
    sig = signature(tool, excerpt) if tool else re.sub(r"\W+", " ", excerpt.lower()).strip()[:100]
    scope_key = "harness" if harness else project
    rid = "pc-" + hashlib.sha256((scope_key + "|" + sig).encode()).hexdigest()[:16]
    with engine.store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
        if row:
            old = json.loads(row[0])
            seen = sorted(set(old.get("seen_sessions", [])) | {sid})
            rec = dict(old, seen_sessions=seen, count=len(seen), occ=dict(old.get("occ", {}), **{sid: n}), last_seen=day,  # occ per session: a re-scan does not double count
                       rank_weight=max(old.get("rank_weight", 1.0), WEIGHT[source]), retried=bool(old.get("retried") or retry), revision=old["revision"] + 1)
            if tp:
                rec["derived_from"] = with_transcript(old, tp)
            if rec == old:
                return False
            db.execute("UPDATE records SET payload=? WHERE id=?", (_dump(rec), rid))
            db.execute("INSERT INTO events(event_type,record_id,revision,record) VALUES ('update',?,?,?)", (rid, rec["revision"], _dump(rec)))
            return True
        rec = {"id": rid, "kind": "papercut", "scope": "technical", "origin": "papercut", "verified_by": "transcript" if source == "det" else "llm",
               "origin_type": "derived" if source == "det" else "agent_inference", "status": "active", "valid_from": day, "revision": 1, "visibility": "internal",
               "tool": tool or "", "excerpt": excerpt, "cause": cause, "retried": bool(retry), "harness": bool(harness), "signature": sig,
               "text": f"[papercut] {tool + ': ' if tool else ''}{excerpt}" + (f" ({cause})" if cause else ""),
               "source": f"memrove/papercuts/{'harness' if harness else _slug(project)}.md",  # the aggregate note; no per-record file
               "rank_weight": WEIGHT[source], "count": 1, "occ": {sid: n}, "seen_sessions": [sid], "last_seen": day, "derived_from": with_transcript({}, tp)}
        if not harness:
            rec["project"] = project
        db.execute("INSERT INTO records VALUES (?,?)", (rid, _dump(rec)))
        db.execute("INSERT INTO events(event_type,record_id,revision,record) VALUES ('ingest',?,?,?)", (rid, 1, _dump(rec)))
    return True


def collect(engine, events, state=None):
    """Deterministic pass over SessionEnd/PreCompact events. Idempotent per session. No model call."""
    n = 0
    for ev in events:
        sid, tp = ev.get("session"), ev.get("transcript_path")
        if ev.get("event") not in ("SessionEnd", "PreCompact") or not sid or not tp or not Path(tp).is_file():
            continue
        from datetime import date
        project, day = ev.get("project") or "unknown", ev.get("day") or date.today().isoformat()
        groups = {}
        for e in extract(tp):
            g = groups.setdefault((e["tool"], signature(e["tool"], e["excerpt"]), e["harness"]), dict(e, n=0))
            g["n"] += 1
            g["retry"] = g["retry"] or e["retry"]
        for g in groups.values():
            n += record(engine, project, sid, day, g["tool"], g["excerpt"], "det", g["retry"], g["harness"], n=g["n"], tp=tp)
    return n


def from_j2(engine, items, project, sid, day, tp=None):
    """J2 `papercuts`: strings 'doing -> what got in the way -> guessed cause'. At most 5."""
    n = 0
    for s in items[:5] if isinstance(items, list) else []:
        if isinstance(s, str) and len(s.strip()) >= 12:
            parts = [x.strip() for x in re.split(r"\s*(?:->|→)\s*", s.strip())]
            n += record(engine, project, sid, day, "", parts[1] if len(parts) > 1 else parts[0], "j2", cause=parts[2] if len(parts) > 2 else "", tp=tp)
    return n


def render(records, title):
    """Generated note body, sorted by count x weight, retried first on ties."""
    rows = sorted(records, key=lambda r: (-r.get("count", 1) * r.get("rank_weight", 1.0), -sum(r.get("occ", {}).values()), not r.get("retried"), r["id"]))
    lines = [f"# Papercuts: {title}", "", "Generated. Sorted by sessions seen x weight (J2 items weigh more). Retry = same tool called again right after.", ""]
    for r in rows:
        tag = ("retry, " if r.get("retried") else "") + ("J2" if r.get("rank_weight", 1.0) > 1 else "transcript")
        lines.append(f"- {r.get('count', 1)} sessions, {sum(r.get('occ', {}).values()) or 1} hits [{tag}] {r.get('tool') + ': ' if r.get('tool') else ''}{r.get('excerpt', '')}" + (f" ({r['cause']})" if r.get("cause") else "") + f" last {r.get('last_seen', '')}")
    return "\n".join(lines) + "\n"
