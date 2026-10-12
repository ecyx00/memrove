"""Kind registry (kinds.json next to this file) and record scope. A module adds a kind by editing the JSON, never the core.
Scope: project:<name> | user | technical | learning | personal. Authorization runs before candidate generation."""
import functools
import json
import re
from pathlib import Path

REGISTRY = Path(__file__).with_name("kinds.json")
ORIGINS = ("user_explicit", "agent_inference", "external_source", "derived")
SCOPE = re.compile(r"project:[^\s:]+|user|technical|learning|personal")
# Brief section 21. Overridden per client by <state>/client-scopes.json, e.g. {"claude_code": ["project", "user"]}.
CLIENT_DEFAULTS = {"claude_code": ["project", "user", "technical", "learning"]}


@functools.lru_cache(maxsize=1)
def registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def spec(kind):
    try:
        return registry()[kind]
    except (KeyError, TypeError):
        raise ValueError(f"unknown kind {kind!r}; add it to kinds.json") from None


def valid_scope(scope):
    return isinstance(scope, str) and bool(SCOPE.fullmatch(scope))


def effective_scope(record):
    """Stored scope, else the kind default ('project' -> project:<name>, or user when the record has no project)."""
    if record.get("scope"):
        return record["scope"]
    default = registry().get(record.get("kind"), {}).get("scope", "project")
    if default == "project":  # no project, or the unresolved label "unknown": user-level, so client scopes keep it recallable
        return f"project:{record['project']}" if record.get("project") and record["project"] != "unknown" else "user"
    return default


def client_scopes(state_dir, client="claude_code"):
    allowed = CLIENT_DEFAULTS.get(client, [])
    try:
        stored = json.loads((Path(state_dir) / "client-scopes.json").read_text(encoding="utf-8")).get(client)
        if isinstance(stored, list) and all(valid_scope(s) or s == "project" for s in stored):
            allowed = stored
    except (OSError, ValueError, AttributeError):
        pass
    return allowed


def in_scope(scope, allowed):
    """allowed tokens: exact scope, or 'project' for every project:<name>."""
    return scope in allowed or (scope.startswith("project:") and "project" in allowed)


def check_transition(kind, old, new):
    machine = registry().get(kind, {}).get("transitions")
    if not machine or old == new:
        return
    if old not in machine or new not in machine:
        raise ValueError(f"{kind} state must be one of {sorted(machine)}")
    if new not in machine[old]:
        raise ValueError(f"illegal {kind} transition {old} -> {new}")


def validate_record(record):
    """Registry rules for the store: scope grammar and, for kinds with a state machine, a legal status."""
    if "scope" in record and not valid_scope(record["scope"]):
        raise ValueError("invalid scope")
    states = registry().get(record.get("kind"), {}).get("states")
    if states and record.get("status") is not None and record["status"] not in states:
        raise ValueError(f"{record['kind']} status must be one of {states}")
