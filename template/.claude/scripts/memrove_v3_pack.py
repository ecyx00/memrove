"""Push replay: the SessionStart block of one arm, as memory looked at a past moment (`memrove pack`). Read-only on the store.

As-of view = a throwaway copy of the records that existed before the timestamp and do not belong to the excluded session:
a record counts by recorded_at (full time) else valid_from (a day: the whole day counts as not yet, so nothing from the case's day leaks);
a record with no date is dropped. A valid_to or restate_dup_of that points at or into the future is cleared. Rule restatements carry no
time of their own, so a restated sentence is taken as known from its record's date (ponytail: a later re-judgement is not undone).
Arms: none = "", rules = rules block, card = rules + card, graph = rules + card + graph pack (memrove_v3_pack_graph.graph_pack).
Card: the newest meaningful (decision cue) same-project transcript that ended before as-of, drafted once with the shipped card prompt
and cached by session id. Threads are not as-of capable (they live in the vault, not in records): left out. Stdlib only.
"""
from datetime import datetime, timezone
import importlib
import json
import os
import re
from pathlib import Path
import sqlite3
import tempfile

BUDGET = 1.0   # USD per cache directory, all drafts together
TS = re.compile(r'"timestamp": ?"([^"]+)"')
KINDS = ("preference", "decision", "rejected", "card")


def parse_time(s):
    """ISO string -> aware UTC datetime (naive input is taken as UTC)."""
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _before(r, as_of_raw, as_of):
    """True when the record existed before as-of; day-only dates need a strictly earlier day."""
    rec = r.get("recorded_at")
    if rec:
        try:
            return parse_time(rec) < as_of
        except ValueError:
            pass
    day = str(r.get("valid_from") or "")[:10]
    return bool(day) and day < str(as_of_raw)[:10]


def _is_session(r, sid):
    if not sid:
        return False
    mine = [str(r.get("session") or "")] + [str(x) for x in r.get("seen_sessions") or []]
    return any(m and (m.startswith(sid[:12]) or sid.startswith(m[:12])) for m in mine if len(m) >= 12)


def asof_state(state, as_of, exclude_session, dest, extra=()):
    """Write dest/memory.sqlite3 with the as-of records (rule and card kinds only) plus `extra` payloads -> dest."""
    t = parse_time(as_of)
    src = sqlite3.connect(f"file:{Path(state) / 'memory.sqlite3'}?mode=ro", uri=True, timeout=5)
    try:
        rows = [json.loads(p) for (p,) in src.execute(
            f"SELECT payload FROM records WHERE json_extract(payload,'$.kind') IN ({','.join('?' * len(KINDS))})", KINDS)]
    finally:
        src.close()
    keep = [r for r in rows if _before(r, as_of, t) and not _is_session(r, exclude_session)]
    ids = {r["id"] for r in keep}
    for r in keep:
        if r.get("restate_dup_of") and r["restate_dup_of"] not in ids:
            r["restate_dup_of"] = None
        if r.get("valid_to") and str(r["valid_to"])[:10] >= str(as_of)[:10]:
            r["valid_to"] = None
    out = sqlite3.connect(Path(dest) / "memory.sqlite3")
    out.execute("CREATE TABLE records(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    out.executemany("INSERT OR REPLACE INTO records VALUES(?,?)", [(r["id"], json.dumps(r)) for r in [*keep, *extra]])
    out.commit()
    out.close()
    return Path(dest)


def _index(root, cache, state):
    """-> [{sid, path, project, end}] of every non-subagent transcript; cached by path, size and mtime."""
    import memrove_v3_llm as llm
    f = Path(cache) / "transcripts.json"
    try:
        old = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    new, out = {}, []
    for p in sorted([*Path(root).glob("*.jsonl"), *Path(root).glob("*/*.jsonl")]):
        if "subagents" in p.parts:
            continue
        st = p.stat()
        key = f"{p}|{st.st_size}|{st.st_mtime_ns}"
        e = old.get(p.stem) if old.get(p.stem, {}).get("key") == key else None
        if not e:
            end = ""
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = TS.search(line)
                    end = m.group(1) if m else end
            e = {"key": key, "path": str(p), "project": llm._project_of(p, state), "end": end}
        new[p.stem] = e
        out.append({"sid": p.stem, **e})
    Path(cache).mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(new), encoding="utf-8")
    return out


def _spent(cache):
    try:
        return float((Path(cache) / "spend.json").read_text())
    except (OSError, ValueError):
        return 0.0


def draft(sid, path, project, day, state, cache, vault=None):
    """Card meta of one session, drafted once (cached by session id, a skip is cached too) -> dict or None. Stops at the budget."""
    import memrove_v3_card as card
    import memrove_v3_llm as llm
    f = Path(cache) / "cards" / f"{sid}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8")) or None
    users, cues, last = card.read(path)
    data = None
    if cues:   # no cue = chit-chat = no model call, same as the shipped hook
        if _spent(cache) >= BUDGET:
            raise RuntimeError(f"pack: draft budget ${BUDGET:.2f} spent ({cache}/spend.json)")
        text, cost, err = llm.RUNNER(card.SYSTEM, card._payload(users, cues, last), vault)
        Path(cache).mkdir(parents=True, exist_ok=True)
        (Path(cache) / "spend.json").write_text(str(_spent(cache) + (cost or 0)))
        if err:
            raise RuntimeError(f"pack: card draft failed: {err}")   # not cached: a rerun retries
        data = llm._parse(text)
    meta = None
    if isinstance(data, dict) and not data.get("skip"):
        meta = {"did": card._clean(data, "did", 3), "why": card._clean(data, "why", 2), "open": card._clean(data, "open", 2),
                "next": card._short(data.get("next") or "", 200)}
        meta = meta if (meta["did"] or meta["open"] or meta["next"]) else None
    if meta:
        meta.update(id="c-" + sid[:12], kind="card", status="active", project=project, session=sid, valid_from=day, origin="llm")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def previous_card(state, project, as_of, exclude_session, cache, transcripts, vault=None):
    """Draft of the newest same-project session that ended before as-of and had a decision cue, or None."""
    t = parse_time(as_of)
    rows = [e for e in _index(transcripts, cache, state)
            if e["project"] == project and e["end"] and parse_time(e["end"]) < t and not _is_session({"session": e["sid"]}, exclude_session)]
    for e in sorted(rows, key=lambda e: e["end"], reverse=True):
        meta = draft(e["sid"], e["path"], project, e["end"][:10], state, cache, vault)
        if meta:
            return meta
    return None


def pack(state, arm, project, prompt, as_of, exclude_session, cap=1500, cache=None, transcripts=None, vault=None):
    """The block of one arm as of `as_of`, without the excluded session."""
    if arm == "none":
        return ""
    import memrove_v3_card as card
    import memrove_v3_rules as rules
    cache = cache or Path.home() / ".cache" / "memrove-pack"
    transcripts = transcripts or Path.home() / ".claude" / "projects"
    graph = None
    if arm == "graph":
        try:
            graph = importlib.import_module('memrove_v3_pack_graph').graph_pack   # by name: the module ships separately
        except ImportError as exc:
            raise RuntimeError(f"pack: graph arm unavailable ({exc}); memrove_v3_pack_graph.py is not in the scripts directory") from exc
    parts = []
    with tempfile.TemporaryDirectory(prefix="memrove-asof-") as tmp:
        first = asof_state(state, as_of, exclude_session, tmp)
        parts.append(rules.block(first, project, cap=cap))
        if arm in ("card", "graph"):
            have = card.block(first, project, cap=cap)
            if not have:
                meta = previous_card(state, project, as_of, exclude_session, cache, transcripts, vault)
                if meta:
                    os.remove(first / "memory.sqlite3")
                    asof_state(state, as_of, exclude_session, tmp, extra=[meta])
                    have = card.block(first, project, cap=cap)
            parts.append(have)
    if graph:
        parts.append(graph(state, project, prompt, as_of, exclude_session, cap) or "")
    return "\n".join(p.rstrip("\n") for p in parts if p) + ("\n" if any(parts) else "")
