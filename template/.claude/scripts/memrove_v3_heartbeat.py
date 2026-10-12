"""Heartbeat per background writer: <state>/heartbeat/<writer>.json = {since, last_run, last_ok, last_error}.
A writer is unhealthy when it has had no success for STALE_S while new owner messages reached the chat log after that success
(or after its first run, when it never succeeded). No new messages = no work = healthy. Stamping never raises."""
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import time

STALE_S = 24 * 3600
WRITERS = ("chatlog", "chattree", "j2_records", "j2_threads", "grab_process")  # docsearch refresh is lazy (on query), not a background writer


def _path(state, writer):
    return Path(state) / "heartbeat" / f"{writer}.json"


def read(state, writer):
    try:
        return json.loads(_path(state, writer).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stamp(state, writer, error=None, now=None):
    """One run finished: error=None -> success. -> None, whatever happens."""
    try:
        now = time.time() if now is None else now
        beat = read(state, writer) or {}
        beat.setdefault("since", now)
        beat["last_run"] = now
        if error is None:
            beat["last_ok"] = now
        else:
            beat["last_error"] = {"at": now, "error": type(error).__name__ if isinstance(error, BaseException) else "error", "msg": str(error)[:200]}
        path = _path(state, writer)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(beat), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


@contextmanager
def guard(state, writer):
    """Stamp success when the block ends, the error (re-raised) when it fails."""
    try:
        yield
    except BaseException as exc:
        stamp(state, writer, exc)
        raise
    stamp(state, writer)


def _new_messages(state, since):
    """True when an owner message newer than `since` (epoch) is in the chat log. Reads only the day files that can hold one."""
    d = Path(state) / "chatlog" / "main"
    day = datetime.fromtimestamp(since).date().isoformat()
    for path in sorted(d.glob("*.jsonl"), reverse=True):
        if path.stem < day:
            break
        try:
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    r = json.loads(raw)
                    if r.get("kind") == "user" and datetime.fromisoformat(r["date"]).timestamp() > since:
                        return True
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
        except OSError:
            continue
    return False


def _install_time(state, create):
    """Epoch of the first look, kept in heartbeat/.since (written once, only when create=True: SessionStart, never doctor)."""
    path = Path(state) / "heartbeat" / ".since"
    try:
        return float(path.read_text())
    except (OSError, ValueError):
        pass
    if not create:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(time.time()))
        return float(path.read_text())
    except (OSError, ValueError):
        return None


def check(state, now=None, create=False):
    """-> [{writer, status: healthy|idle|unhealthy|unseen|never ran, age_h (since last_ok), last_error}] for every writer.
    A chat-driven writer with no heartbeat is 'never ran' once 24 h passed since .since and owner messages came after it."""
    now = time.time() if now is None else now
    since = _install_time(state, create)
    out = []
    for w in WRITERS:
        b = read(state, w)
        if not b:
            dead = since and w != "grab_process" and now - since > STALE_S and _new_messages(state, since)
            out.append({"writer": w, "status": "never ran" if dead else "unseen", "age_h": None, "last_error": None})
            continue
        ref = b.get("last_ok") or b.get("since") or now
        age = now - ref
        if age <= STALE_S:
            status = "healthy"
        elif w == "grab_process":  # driven by captured cards, not chat: stale only while its last run failed
            status = "unhealthy" if (b.get("last_error") or {}).get("at", 0) > b.get("last_ok", 0) else "idle"
        else:
            status = "unhealthy" if _new_messages(state, ref) else "idle"
        out.append({"writer": w, "status": status, "age_h": round(age / 3600, 1) if b.get("last_ok") else None, "last_error": b.get("last_error")})
    return out


def line(state, now=None):
    """The SessionStart line: empty unless some writer is unhealthy."""
    try:
        bad = [r["writer"] for r in check(state, now, create=True) if r["status"] in ("unhealthy", "never ran")]
    except Exception:
        return ""
    return ("Memrove: yazici sessiz: " + ", ".join(bad) + " (memrove doctor)\n") if bad else ""
