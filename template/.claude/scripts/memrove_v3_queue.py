"""J2 queue bookkeeping: which sessions still owe a window, poison-window retries and quarantine, the daily cap, the last error.
State lives in state/j2-queue.json (+ the session registry llm-sessions.json that llm.session_notes drains). Nothing here calls a model.
Read side for doctor and the SessionStart warning: status(state)."""
import json
import math
import os
import time
from datetime import date
from pathlib import Path

DEFAULT_CAP, DEFAULT_TRIES, DEFAULT_IDLE_S = 40, 3, 1800
# A PreCompact or idle sweep sends a session to J2 only once it owes at least J2_MIN_CHARS of user text (about a quarter of a window), or has
# been quiet for J2_TAIL_S (the tail is flushed then). SessionEnd always flushes. Without this a long session paid one call per tiny delta.
DEFAULT_MIN_CHARS, DEFAULT_TAIL_S = 3000, 4 * 3600
# Shares of the daily cap. J2 (threads, session cards: what SessionStart shows) may use all of it; every other job leaves J2's reserve untouched,
# J3/J4/evaluators also leave J1's, and J4 alone never takes more than its share (and run_jobs lets it run only when no J2 window is due). llm-config.json may override each as a 0..1 fraction.
DEFAULT_SHARES = {"j2_reserve": 0.5, "j1_reserve": 0.15, "j4_share": 0.10}


def config(state):
    """Daily call cap, tries per window before quarantine, idle seconds before a session without SessionEnd is swept.
    From state/llm-config.json {"daily_cap":..,"max_tries":..,"sweep_idle_s":..}; daily_cap also from env MEMROVE_LLM_DAILY_CAP."""
    cfg = {"daily_cap": DEFAULT_CAP, "max_tries": DEFAULT_TRIES, "sweep_idle_s": DEFAULT_IDLE_S, "j2_min_chars": DEFAULT_MIN_CHARS, "j2_tail_s": DEFAULT_TAIL_S, **DEFAULT_SHARES}
    try:
        file = json.loads((Path(state) / "llm-config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        file = {}
    for k in cfg:
        v = file.get(k) if isinstance(file, dict) else None
        if k == "daily_cap" and v is None and os.environ.get("MEMROVE_LLM_DAILY_CAP", "").isdigit():
            v = int(os.environ["MEMROVE_LLM_DAILY_CAP"])
        if k in DEFAULT_SHARES:
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1:
                cfg[k] = float(v)
        elif isinstance(v, int) and not isinstance(v, bool) and v >= 1:
            cfg[k] = v
    return cfg


def allowed(state, job, usage):
    """May `job` make one more call today? usage = {"calls": n, "by_job": {job: n}}. Priority: j2 > j1 > everything else (j3, j4, evaluators)."""
    cfg, by = config(state), usage.get("by_job") or {}
    cap, total = cfg["daily_cap"], usage.get("calls", 0)
    if job == "j2":
        return total < cap
    held = max(0, math.ceil(cap * cfg["j2_reserve"]) - by.get("j2", 0))
    if job != "j1":
        held += max(0, math.ceil(cap * cfg["j1_reserve"]) - by.get("j1", 0))
    if total + held >= cap:
        return False
    return job != "j4" or by.get("j4", 0) < math.ceil(cap * cfg["j4_share"])


def _read(path, default):
    try:
        v = json.loads(Path(path).read_text(encoding="utf-8"))
        return v if isinstance(v, type(default)) else default
    except (OSError, ValueError):
        return default


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _load(state):
    q = _read(Path(state) / "j2-queue.json", {})
    q.setdefault("tries", {})
    q.setdefault("quarantine", [])
    return q


def _save(state, q):
    q["tries"] = dict(list(q["tries"].items())[-200:])
    q["quarantine"] = q["quarantine"][-100:]
    _write(Path(state) / "j2-queue.json", q)


def record_error(state, where, exc):
    """Keep the last error (where, type, short message) for doctor and the SessionStart warning. Never raises."""
    try:
        q = _load(state)
        q["last_error"] = {"at": time.time(), "where": where, "error": type(exc).__name__ if isinstance(exc, BaseException) else "error",
                           "msg": str(exc)[:300]}
        _save(state, q)
    except Exception:
        pass


def window_failed(state, sid, cursor, to, project, err):
    """One failed J2 window. -> True when it just reached max_tries: it is quarantined (listed, skipped) and the caller moves the cursor past it."""
    q, key = _load(state), f"{sid}:{cursor}"
    q["tries"][key] = q["tries"].get(key, 0) + 1
    q["last_error"] = {"at": time.time(), "where": "j2", "error": (err or "no-usable-answer")[:300], "msg": f"session {sid[:12]} window @{cursor}"}
    __import__("memrove_v3_heartbeat").stamp(state, "j2_records", err or "no-usable-answer")
    gone = q["tries"][key] >= config(state)["max_tries"]
    if gone:
        q["quarantine"].append({"session": sid, "project": project, "from": cursor, "to": to, "tries": q["tries"].pop(key),
                                "err": err or "no-usable-answer", "at": time.time()})
    _save(state, q)
    return gone


def window_ok(state, sid, cursor):
    q = _load(state)
    q["tries"].pop(f"{sid}:{cursor}", None)
    q["last_ok"] = time.time()
    _save(state, q)
    __import__("memrove_v3_heartbeat").stamp(state, "j2_records")


def cap_hit(state):
    q = _load(state)
    q["cap_hit"] = {"day": date.today().isoformat(), "at": time.time()}
    _save(state, q)


def is_subagent(sid, path):
    """Subagent transcripts (<session>/subagents/agent-*.jsonl, or agent-* at the project root in older Claude Code) are never J2 sessions:
    their records are all isSidechain (no user messages), and their results land in the parent transcript."""
    p = str(path or "")
    return str(sid or "").startswith("agent-") or "/subagents/" in p or os.path.basename(p).startswith("agent-")


def same_transcript(reg, cursors):
    """One transcript can be registered under two ids (raw session uuid from older hooks, hashed id from newer ones), each with its own cursor,
    and both drain it. -> {path: [sids]}; every id of a path gets the furthest cursor so no window is sent twice. Mutates cursors."""
    groups = {}
    for sid, v in reg.items():
        if v.get("transcript_path"):
            groups.setdefault(v["transcript_path"], []).append(sid)
    for sids in groups.values():
        if len(sids) > 1:
            top = max(cursors.get(s, 0) for s in sids)
            for s in sids:
                if top:
                    cursors[s] = top
    return groups


def pending_chars(path, cursor, limit):
    """User text (each message cut to 600, +3) from `cursor` to the end of the transcript, stopping once it reaches `limit`."""
    from memrove_v3_capture import user_text
    n = 0
    try:
        if os.path.getsize(path) < cursor:
            cursor = 0
        with open(path, "rb") as f:
            f.seek(cursor)
            for raw in f:
                if not raw.endswith(b"\n"):
                    break
                try:
                    r = json.loads(raw)
                except ValueError:
                    continue
                u = user_text(r) if isinstance(r, dict) else None
                if u:
                    n += min(len(u), 600) + 3
                    if n >= limit:
                        break
    except OSError:
        pass
    return n


def ready(state, entry, cursor, now=None, flush=False):
    """May this session's pending text go to J2 now? flush (SessionEnd) always; otherwise enough text, or quiet long enough to flush the tail."""
    cfg, now = config(state), now or time.time()
    if flush or entry.get("ended"):
        return True
    return now - (entry.get("at") or 0) >= cfg["j2_tail_s"] or pending_chars(entry["transcript_path"], cursor, cfg["j2_min_chars"]) >= cfg["j2_min_chars"]


def register(state, events):
    """Every capture event with a transcript puts its session in the registry, whatever the worker is doing (locked, capped, no SessionEnd).
    `ended` is set by SessionEnd/PreCompact; the rest is picked up once the session has been idle for sweep_idle_s."""
    path = Path(state) / "llm-sessions.json"
    reg, changed = _read(path, {}), False
    for e in sorted(events, key=lambda e: e.get("at", 0)):
        if e.get("session") and e.get("transcript_path") and not is_subagent(e["session"], e["transcript_path"]):
            old = reg.get(e["session"], {})
            reg[e["session"]] = {"transcript_path": e["transcript_path"], "project": e.get("project") or old.get("project"), "day": e.get("day") or old.get("day"),
                                 "at": e.get("at"), "ended": e.get("event") == "SessionEnd" if e.get("event") else bool(old.get("ended"))}
            changed = True
    if changed:
        _write(path, dict(sorted(reg.items(), key=lambda kv: kv[1].get("at") or 0)[-200:]))


def _behind(state, now=None):
    """Registered sessions that owe J2 windows and are due (ended, or idle long enough) -> list of session ids."""
    now = now or time.time()
    reg, cur = _read(Path(state) / "llm-sessions.json", {}), _read(Path(state) / "llm-cursors.json", {})
    groups, out = same_transcript(reg, cur), []
    for sid, v in reg.items():
        tp = v.get("transcript_path")
        if not tp or is_subagent(sid, tp) or not os.path.isfile(tp) or cur.get(sid, 0) >= os.path.getsize(tp) or groups[tp][0] != sid:
            continue
        if due(state, v, now) and ready(state, v, cur.get(sid, 0), now):
            out.append(sid)
    return out


def due(state, entry, now=None):
    """Is a registry entry due for J2 (SessionEnd/PreCompact seen, or idle past the sweep threshold)?"""
    return bool(entry.get("ended")) or (now or time.time()) - (entry.get("at") or 0) >= config(state)["sweep_idle_s"]


def sweep_due(state):
    try:
        return bool(_behind(state))
    except Exception:
        return False


def status(state):
    """The one read side: {"pending": n, "quarantined": n, "quarantine": [...], "cap_used": n, "cap": n, "cap_hit_today": bool, "last_error": {...}|None, "last_ok": ts|None}."""
    state = Path(state)
    q, cfg, today = _load(state), config(state), date.today().isoformat()
    usage = _read(state / "llm-usage.json", {})
    return {"pending": len(_behind(state)), "quarantined": len(q["quarantine"]), "quarantine": q["quarantine"][-5:],
            "cap_used": usage.get("calls", 0) if usage.get("day") == today else 0, "cap": cfg["daily_cap"],
            "by_job": dict(usage.get("by_job") or {}) if usage.get("day") == today else {}, "j2_reserve": math.ceil(cfg["daily_cap"] * cfg["j2_reserve"]),
            "cap_hit_today": (q.get("cap_hit") or {}).get("day") == today, "last_error": q.get("last_error"), "last_ok": q.get("last_ok")}
