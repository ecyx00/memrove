"""Cross-agent access policy (design 12.6): what an agent may read, whether it may write, whether its text may go to a remote service.
<state>/access.json = {"default": {read, write, remote}, "agents": {name: {read, write, remote}}}. No file: everything open, as before.
A malformed file: one stderr line, read and write stay open (capture never breaks), remote is off for everyone (privacy fails closed). Stdlib only."""
import json
import sys
from pathlib import Path

OPEN = {"read": ["*"], "write": True, "remote": True}


def load(state):
    """-> policy dict, or None when state has no access.json (callers treat None as open)."""
    p = Path(state) / "access.json"
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        raw = None
    try:
        pol = json.loads(raw)
        if not isinstance(pol, dict) or not isinstance(pol.get("agents", {}), dict) or not isinstance(pol.get("default", {}), dict):
            raise ValueError
        return pol
    except (TypeError, ValueError):
        print("memrove: access.json unreadable, reads and writes stay open, remote is off", file=sys.stderr)
        return {"default": {**OPEN, "remote": False}, "agents": {}, "malformed": True}


def _rule(policy, agent):
    if policy is None:
        return OPEN
    r = {**OPEN, **policy.get("default", {}), **(policy.get("agents", {}).get(agent) or {})}
    if policy.get("malformed"):
        r["remote"] = False
    return r


def can_read(policy, agent, project):
    read = _rule(policy, agent).get("read")
    read = [read] if isinstance(read, str) else (read or [])
    return "*" in read or (bool(project) and project in read)


def can_write(policy, agent):
    return _rule(policy, agent).get("write") is not False


def can_remote(policy, agent):
    return _rule(policy, agent).get("remote") is True
