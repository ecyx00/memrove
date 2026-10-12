"""The companion folder: starter notes the user owns, and size limits for the two handoff files."""
import json
from pathlib import Path
import re

NAMES = ("Core.md", "Soul.md", "Rules.md", "Last-Session.md", "Threads.md", "Journal.md")
# Character limits for the two handoff files that are rewritten rather than appended to.
# 0 switches a limit off. Rules, identity and the journal grow by design and are only measured.
LIMITS = {"Last-Session.md": 3000, "Threads.md": 8000}
LIMIT_RANGE = (1000, 200000)
DEFAULT_DIRECTORY = "🔮 850-Companion"
STARTERS = {
    "Core.md": "# Thinking partner\n\nI am the user's thinking partner and second brain. We settle my identity and way of working together.\n\n"
               "## The user and how we work\nNot personalised yet. Learn the user's name, form of address, field and expectations by talking. "
               "Do not invent history.\n\n## Lasting preferences\nKeep the preferences the user states outright here, with the source they rest on.\n",
    "Rules.md": "# User corrections\n\nNo correction recorded yet. Record explicit user corrections with their date and scope, one or two lines per rule; "
                "long reasons and the story of the event go into a separate note. Never turn a passing request into a lasting rule.\n",
    "Last-Session.md": "# Last session\n\nNo work result recorded yet. memrove writes this file: after a session ends it appends one card headed "
                       "`## YYYY-MM-DD HH:MM · <label> · <session_id[:8]>` (result, reason, open step, source). Cards you write by hand are kept.\n",
    "Threads.md": "# Threads\n\n## Active threads\nNo open topic recorded yet.\n\n## Closed threads\n",
    "Journal.md": "# Journal\n\nObservations, lessons and open questions that come out of working together. Never present an inference as certain knowledge about the user.\n",
}


def directory(vault):
    """The one existing companion folder of the vault, a new default when there is none, None when it is ambiguous."""
    vault = Path(vault).resolve()
    named, holding = [], []
    for path in vault.iterdir():
        if path.name.startswith(".") or path.is_symlink() or not path.is_dir() or re.search(r"(?i)(archive|arşiv|arsiv)", path.name):
            continue
        if path.name == DEFAULT_DIRECTORY or path.name.casefold().endswith(("companion", "echo")):
            named.append(path)
        elif any((path / n).is_file() for n in ("Core.md", "Soul.md")):
            holding.append(path)
    found = named or holding
    if len(found) > 1:
        return None
    return found[0] if found else vault / DEFAULT_DIRECTORY


def initialize(vault, state):
    """Create the starter notes that are missing, once. They are user data: existing files are never touched."""
    from memrove_v3_sync import atomic
    state, vault = Path(state), Path(vault).resolve()
    marker = state / "companion-bootstrap.json"
    if marker.exists():
        return {"status": "existing"}
    target = directory(vault)
    if target is None or target.is_symlink():
        return {"status": "needs_attention", "reason": "Choose the existing companion directory; no notes created."}
    target.mkdir(parents=True, exist_ok=True)
    if not target.resolve().is_relative_to(vault):
        raise ValueError("Companion directory escapes vault")
    created = []
    for name, text in STARTERS.items():
        if name == "Core.md" and (target / "Soul.md").exists():
            continue  # a Soul already carries the identity
        try:
            with (target / name).open("x", encoding="utf-8") as out:
                out.write(text)
            created.append(name)
        except FileExistsError:
            pass
    state.mkdir(parents=True, exist_ok=True)
    atomic(marker, json.dumps({"schema": 1, "directory": target.relative_to(vault).as_posix()}))
    return {"status": "initialized", "created": created}


def check_limits(value):
    if not isinstance(value, dict) or set(value) - set(LIMITS) - {"schema"} or value.get("schema", 1) != 1:
        raise ValueError("companion limits accept only " + ", ".join(LIMITS))
    limits = {name: value.get(name, default) for name, default in LIMITS.items()}
    for name, number in limits.items():
        if type(number) is not int or not (number == 0 or LIMIT_RANGE[0] <= number <= LIMIT_RANGE[1]):
            raise ValueError(f"{name} limit must be 0 (off) or an integer between {LIMIT_RANGE[0]} and {LIMIT_RANGE[1]}")
    return limits


def read_limits(state):
    """(limits, valid). Kept in the runtime state, not in the vault preferences, so an older release still reads those.
    A damaged file yields the defaults and is never rewritten silently."""
    path = Path(state) / "companion-limits.json"
    if not path.exists() and not path.is_symlink():
        return dict(LIMITS), True
    try:
        if path.is_symlink():
            raise ValueError("symlink")
        return check_limits(json.loads(path.read_text(encoding="utf-8"))), True
    except (ValueError, OSError):
        return dict(LIMITS), False


def save_limits(state, changes):
    from memrove_v3_sync import atomic
    current, valid = read_limits(state)
    if not valid:
        raise ValueError("companion-limits.json in the runtime state is invalid; fix or remove it first")
    limits = check_limits(dict(current, **changes))
    atomic(Path(state) / "companion-limits.json", json.dumps(dict(schema=1, **limits), ensure_ascii=False, indent=2) + "\n")
    return limits


def size(path):
    """Characters as stored (not bytes, not UTF-16 units); line endings are not translated. Streams the file."""
    with Path(path).open(encoding="utf-8", errors="replace", newline="") as source:
        return sum(len(chunk) for chunk in iter(lambda: source.read(1 << 16), ""))


def hygiene(vault, state, names=NAMES, target=None):
    """Read-only size report of the companion files; a limited file says whether it is over."""
    vault = Path(vault).resolve()
    target = directory(vault) if target is None else target
    limits, valid = read_limits(state)
    report = {"status": "ok", "limits": limits, "limits_file": "ok" if valid else "invalid_defaults_used",
              "files": {}, "over_limit": []}
    if target is None:
        return dict(report, status="ambiguous_directory")
    report["directory"] = target.relative_to(vault).as_posix()
    for name in names:
        path = target / name
        if path.is_symlink() or not path.is_file():
            continue
        entry = report["files"][name] = {"chars": size(path)}
        if name in LIMITS:
            entry.update(limit=limits[name], over_limit=bool(limits[name]) and entry["chars"] > limits[name])
            if entry["over_limit"]:
                report["over_limit"].append(name)
    if report["over_limit"]:
        report["status"] = "over_limit"
    return report
