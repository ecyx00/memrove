"""Vault preferences: how often the background sync may start, and the secret filter. No models."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

FILE = ".memrove-preferences.json"
DEFAULTS = dict(auto_sync=True, interval_minutes=0, secret_filter=True)
PROFILES = {"normal": dict(auto_sync=True, interval_minutes=0),
            "economical": dict(auto_sync=True, interval_minutes=15),
            "manual": dict(auto_sync=False, interval_minutes=15)}
RETIRED = ("context_mode", "context_chars")  # written by older releases; accepted and dropped


def validate(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS) - set(RETIRED):
        raise ValueError("Unsupported preferences; use memrove.py preferences")
    prefs = dict(DEFAULTS, **{k: v for k, v in value.items() if k in DEFAULTS})
    for key in ("auto_sync", "secret_filter"):
        if type(prefs[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if type(prefs["interval_minutes"]) is not int or not 0 <= prefs["interval_minutes"] <= 1440:
        raise ValueError("interval_minutes must be an integer between 0 and 1440")
    return prefs


def preferences_path(vault):
    path = Path(vault) / FILE
    if path.is_symlink():
        raise ValueError("Preferences must be a regular vault-local file")
    return path


def read(vault):
    path = preferences_path(vault)
    return validate(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})


def save(vault, changes, profile=None):
    """Validate the stored file first (a malformed one is never overwritten), then write atomically.
    A profile never flips the secret filter: that is a separate safety choice."""
    current = read(vault)
    prefs = validate(dict(current, **(PROFILES[profile] if profile else {}), **changes))
    path = preferences_path(vault)
    fd, temp = tempfile.mkstemp(prefix=".memrove-preferences-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(prefs, out, ensure_ascii=False, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return prefs


def claim_check(state, settings, event, now=None):
    """May an event start the background worker? A session start always may; other events wait out interval_minutes,
    counted across clients through one small SQLite file. A pending hook error lifts the wait."""
    if not settings["auto_sync"]:
        return False
    if not settings["interval_minutes"]:
        return True
    now = time.time() if now is None else now
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(state / "check-cadence.sqlite3", timeout=1)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS cadence (id INTEGER PRIMARY KEY, started REAL)")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT started FROM cadence WHERE id=1").fetchone()
        waiting = row and 0 <= now - row[0] < settings["interval_minutes"] * 60
        if event != "SessionStart" and waiting and not (state / "hook-error.json").exists():
            return False
        db.execute("INSERT OR REPLACE INTO cadence VALUES (1, ?)", (now,))
    return True
