#!/usr/bin/env python3
"""Lesson triggers. The source of truth is the vault sidecar <vault>/.memrove-triggers.json {record id: spec}: records under
notes/observed are not projections (an edit to the store would not reach their Markdown), and a sidecar keyed by id survives
sync and a rebuild from Markdown. compile() joins it with the active records into <state>/triggers.json, the only file the
PreToolUse hook reads (memrove_v3_jit)."""
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def spec(tools, match, cwd=None, deny=False):
    """-> the JSON string stored on the record; raises ValueError on a bad regex or no tool."""
    tools = [t for t in (tools or []) if t.strip()]
    if not tools or not match:
        raise ValueError("a trigger needs at least one tool and a match regex")
    for rx in (match, cwd):
        if rx:
            re.compile(rx)
    return json.dumps({"tools": tools, "match": match, "cwd": cwd or "", "deny": bool(deny)}, ensure_ascii=False, separators=(",", ":"))


def sidecar(vault):
    return Path(vault) / ".memrove-triggers.json"


def load(vault):
    try:
        return json.loads(sidecar(vault).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save(vault, mapping):
    path = sidecar(vault)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(mapping, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def set_trigger(vault, state, rid, value):
    """Store (or with None remove) the spec of one record and recompile."""
    mapping = load(vault)
    mapping.pop(rid, None) if value is None else mapping.__setitem__(rid, value)
    save(vault, mapping)
    compile(vault, state)


def entries(vault, state):
    """Active records with a trigger -> list of compiled dicts."""
    from memrove_v3_context import is_dead  # one liveness rule for recall and triggers
    out, mapping = [], load(vault)
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    with sqlite3.connect(f"file:{Path(state) / 'memory.sqlite3'}?mode=ro", uri=True) as db:
        rows = [json.loads(p) for rid in mapping for (p,) in db.execute("SELECT payload FROM records WHERE id=?", (rid,))]
        gone = {s for (p,) in db.execute("SELECT payload FROM records WHERE payload LIKE '%\"supersedes\"%'") for s in (json.loads(p).get("supersedes") or [])}
    for r in rows:
        try:
            t = json.loads(mapping[r["id"]])
            if r.get("status") == "superseded" or is_dead(r, now) or r["id"] in gone:
                continue
            out.append({"id": r["id"], "date": r.get("valid_from") or "", "text": " ".join(str(r.get("text") or "").split()),
                        "tools": t["tools"], "match": t["match"], "cwd": t.get("cwd") or "", "deny": bool(t.get("deny"))})
        except Exception:
            continue
    return sorted(out, key=lambda e: e["id"])


def compile(vault, state):
    """Rewrite <state>/triggers.json atomically; never raises (a failed compile must not fail sync or remember)."""
    try:
        path = Path(state) / "triggers.json"
        data = json.dumps(entries(vault, state), ensure_ascii=False)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception:
        return False
