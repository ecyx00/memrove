#!/usr/bin/env python3
"""Executable memory: a record may carry a read-only shell `check` that says "still true". Run at write time (refuse on failure)
and by the SessionEnd worker (results cached in state/check-results.json). Nothing is ever superseded automatically."""
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

ALLOWED = {"test", "grep", "rg", "ls", "stat", "cat", "head", "jq", "hintmap", "claude", "memrove", "python3", "sqlite3"}
MEMROVE_WRITES = {"sync", "remember", "backfill", "init", "restore", "backup", "grab", "graph"}
TIMEOUT = 5
CACHE = "check-results.json"


def validate(cmd):
    """-> None if the command may run unattended, else the reason it may not. ponytail: first-word allowlist; python3 -c and
    claude/hintmap/memrove can still write if they try, upgrade to a sandbox if checks ever come from untrusted text."""
    if not isinstance(cmd, str) or not cmd.strip():
        return "empty check"
    if "\n" in cmd:
        return "newlines are not allowed"
    q = None  # bash quote state; backslashes outside single quotes are refused so this scan and shlex agree with bash
    for i, ch in enumerate(cmd):
        if q == "'":
            q = None if ch == "'" else q
        elif ch == "\\":
            return "backslashes are not allowed outside single quotes"
        elif ch == "`" or (ch == "$" and cmd[i + 1:i + 2] == "("):
            return "backticks and $(...) are not allowed"
        elif ch in "'\"":
            q = None if q == ch else (q or ch)
    lex = shlex.shlex(cmd, posix=False, punctuation_chars=True)  # non-posix keeps quotes, so a quoted ">" stays a word
    lex.whitespace_split = True
    try:
        toks = list(lex)
    except ValueError as exc:
        return f"cannot parse: {exc}"
    segs, cur = [], []
    for t in toks:
        if t in ("|", "||", "&&", ";"):
            segs.append(cur)
            cur = []
        elif all(c in lex.punctuation_chars for c in t):
            return "redirects, & and subshells are not allowed"
        else:
            cur.append(t)
    segs.append(cur)
    for seg in segs:
        words = shlex.split(" ".join(seg)) if seg else []
        if not words:
            return "empty command segment"
        head = words[0]
        if head not in ALLOWED:
            return f"{head!r} is not allowed; allowed: {', '.join(sorted(ALLOWED))}"
        if head == "python3" and words[1:2] != ["-c"]:
            return "python3 is allowed only as python3 -c"
        if head == "sqlite3" and not ({"-readonly", "-ro"} & set(words)):
            return "sqlite3 needs the -readonly flag"
        if head == "memrove" and MEMROVE_WRITES & set(words[1:2]):
            return f"memrove {words[1]} writes; checks are read-only"
    return None


def run(cmd, timeout=TIMEOUT):
    """-> (ok, why). Runs in the home directory, so checks use absolute or ~ paths."""
    bad = validate(cmd)
    if bad:
        return False, bad
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=Path.home(), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if p.returncode:
        return False, f"exit {p.returncode}: {(p.stderr or p.stdout).strip()[:200]}"
    return True, ""


def _active(store):
    """{id: (project, check)} of records with a check that nothing supersedes and that are not retired."""
    with store._connect() as db:
        gone = {v for (v,) in db.execute("SELECT DISTINCT j.value FROM records, json_each(payload,'$.supersedes') j")}
        rows = db.execute("SELECT id, payload FROM records WHERE json_extract(payload,'$.check') IS NOT NULL "
                          "AND coalesce(json_extract(payload,'$.status'),'active') NOT IN ('retired','rejected','dropped_llm','deleted_by_user','source_missing','superseded')").fetchall()
    out = {}
    for rid, raw in rows:
        r = json.loads(raw)
        if rid not in gone and r.get("check"):
            out[rid] = (r.get("project") or "", r["check"])
    return out


def _read(state):
    try:
        return json.loads((Path(state) / CACHE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def run_all(store, state, projects=None):
    """Worker job: run the checks of active records (of `projects`, or all) and cache {id: {ok, why, at, check}}."""
    active, cache = _active(store), _read(state)
    cache = {k: v for k, v in cache.items() if k in active}  # superseded or retired records leave the cache
    for rid, (project, cmd) in active.items():
        if projects is None or project in projects:
            ok, why = run(cmd)
            cache[rid] = {"ok": ok, "why": why, "at": time.time(), "check": cmd, "project": project}
    tmp = Path(state) / (CACHE + ".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(Path(state) / CACHE)
    return cache


def forget(state, ids):
    """Drop cached results of records that were just superseded, so no stale line outlives the link."""
    cache = _read(state)
    if any(cache.pop(i, None) for i in ids):
        (Path(state) / CACHE).write_text(json.dumps(cache), encoding="utf-8")


def failing(state):
    return {k: v for k, v in _read(state).items() if not v.get("ok")}


def notice(state):
    """SessionStart line: empty unless a cached check fails. Reads one small file."""
    n = len(failing(state))
    return f"{n} records may be stale: `memrove doctor`\n" if n else ""
