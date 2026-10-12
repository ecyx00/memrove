"""Chat log (memory core, slice 1): one append-only log of every user message and agent text reply, from Claude Code main
transcripts, Codex and Antigravity, in timestamp order. Design: Taelin's OptChat / UniiChat (log -> binary tree of 512-byte
summaries -> a cache-stable view). The tree and the view live in memrove_v3_chattree.

state/chatlog/
  main/YYYY-MM-DD.jsonl   {i, kind, text, size, date, src, k}   i = permanent id; the file is the local day of the WRITE
  cursors.json            per source file: byte offset of the first unread complete line
  seen.txt                one short hash per ingested source record (+ "p:" hashes of long user pastes)
  lock                    flock: one process owns the chat while it writes

Rules kept from the spec: only append, one write + flush + fsync per batch, never edit; text is cut only by splitting it into several
messages in a row; thoughts and tool traffic are not logged (v1). The user's words are never filtered (no English / pasted-prompt
heuristics); the one exception is an exact repeat of a long user paste. Secrets are redacted before anything is written.
"""
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import time

from memrove_v3_secrets import redact
from memrove_v3_sessions import clean_turn

MAX_MSG = 8000        # bytes of text per message; a longer text is split into several messages in a row
PASTE_MIN = 1000      # a user text this long that repeats exactly is a re-paste: logged once
KINDS = ("user", "agent")
TAG = re.compile(r"<(system-reminder|task-notification|command-[a-z]+|local-command-[a-z]+)>.*?</\1>", re.S)
COMMAND = re.compile(r"<command-name>\s*(.*?)\s*</command-name>(?:.*?<command-args>(.*?)</command-args>)?", re.S)
NOT_WORDS = ("[Request interrupted", "Caveat:", "<bash-stdout>", "<bash-stderr>")


def root(state):
    return Path(state) / "chatlog"


# ---- lock -----------------------------------------------------------------------------------------------------------

@contextmanager
def lock(state, wait=False):
    """Exclusive owner of the chat. Yields True when held, False when another process owns it (wait=False). The OS frees it on death."""
    d = root(state)
    d.mkdir(parents=True, exist_ok=True)
    f = open(d / "lock", "a+")
    got = False
    try:
        try:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            got = True
        except ImportError:  # Windows
            import msvcrt
            f.seek(0)
            while not got:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    got = True
                except OSError:
                    if not wait:
                        break
                    time.sleep(0.2)
        except OSError:
            got = False
        yield got
    finally:
        f.close()


# ---- log ------------------------------------------------------------------------------------------------------------

def _files(d):
    return sorted((Path(d) / "main").glob("*.jsonl"))


def _last_record(path):
    """Last valid JSON line of a file (a torn tail is skipped) or None."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        n = min(size, 1 << 17)
        while True:
            f.seek(size - n)
            lines = f.read(n).split(b"\n")
            if size > n:
                lines = lines[1:]  # first piece may be cut
            for raw in reversed(lines):
                try:
                    r = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(r, dict) and "i" in r:
                    return r
            if n >= size:
                return None
            n = min(size, n * 4)


def count(state):
    """Number of messages in the log = the next id."""
    for path in reversed(_files(root(state))):
        r = _last_record(path)
        if r:
            return r["i"] + 1
    return 0


def read_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            try:
                r = json.loads(raw)
            except ValueError:
                continue  # torn line from a crash
            if isinstance(r, dict) and "i" in r:
                yield r


def messages(state, start=0, stop=None):
    """Messages with start <= i < stop, in order."""
    files = _files(root(state))
    for n, path in enumerate(files):
        nxt = _first_id(files[n + 1]) if n + 1 < len(files) else None
        if nxt is not None and nxt <= start:
            continue
        for r in read_file(path):
            if r["i"] >= start and (stop is None or r["i"] < stop):
                yield r
            elif stop is not None and r["i"] >= stop:
                return


def _first_id(path):
    for r in read_file(path):
        return r["i"]
    return None


def get(state, i):
    for r in messages(state, i, i + 1):
        return r
    return None


def append(state, items, today=None):
    """Append [{kind, text, date, src?, k?}] under the caller's lock: ids are assigned in order, one write + flush + fsync.
    -> the stored records. A file that does not end in a newline (crash mid-write) gets one first."""
    if not items:
        return []
    d = root(state)
    (d / "main").mkdir(parents=True, exist_ok=True)
    nxt = count(state)
    path = d / "main" / ((today or datetime.now().date().isoformat()) + ".jsonl")
    out, blob = [], []
    for it in items:
        rec = {"i": nxt, "kind": it["kind"], "text": it["text"], "size": len((it["kind"] + ": " + it["text"]).encode("utf-8")),
               "date": it["date"]}
        for k in ("src", "k", "probe"):
            if it.get(k):
                rec[k] = it[k]
        blob.append(json.dumps(rec, ensure_ascii=False) + "\n")
        out.append(rec)
        nxt += 1
    with open(path, "ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell():
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write("".join(blob).encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    return out


# ---- splitting, redaction ---------------------------------------------------------------------------------------------

def split_text(text, limit=MAX_MSG):
    """Pieces of at most `limit` UTF-8 bytes whose concatenation is exactly `text`; breaks fall after a blank line, line or space when one is near."""
    out = []
    while len(text.encode("utf-8")) > limit:
        cut = text[:limit]
        while len(cut.encode("utf-8")) > limit:
            cut = cut[: max(1, int(len(cut) * limit / len(cut.encode("utf-8")) * 0.98))]
        floor = len(cut) * 3 // 4
        for sep in ("\n\n", "\n", " "):
            at = cut.rfind(sep)
            if at >= floor:
                cut = cut[: at + len(sep)]
                break
        out.append(cut)
        text = text[len(cut):]
    out.append(text)
    return out


def _sha(s, n=16):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


# ---- sources ----------------------------------------------------------------------------------------------------------

def _ts(value):
    """ISO string or epoch -> aware datetime (local zone) or None."""
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value).astimezone()
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone()
    except (ValueError, OSError, OverflowError):
        return None


def _lines(path, off):
    """(start, end, record) for every complete JSON line from byte `off`; a half-written last line waits for its writer."""
    try:
        size = Path(path).stat().st_size
        if size < off:
            off = 0  # replaced file: the seen-set keeps everything already logged from doubling
        with open(path, "rb") as f:
            f.seek(off)
            pos = off
            for raw in f:
                if not raw.endswith(b"\n"):
                    break
                start, pos = pos, pos + len(raw)
                try:
                    r = json.loads(raw)
                except ValueError:
                    r = None
                yield start, pos, r if isinstance(r, dict) else None
    except OSError:
        return


def _blocks(content, kinds=("text",)):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") in kinds and isinstance(b.get("text"), str))
    return ""


def user_words(content):
    """The user's own text of a Claude message body, or None when it is harness traffic. Slash commands keep their name and arguments;
    <pasted_content> and <bash-input> stay (the user's words); injected tags, interrupts and shell output go."""
    text = _blocks(content)
    m = COMMAND.match(text.lstrip())
    if m:
        return (m[1] + " " + (m[2] or "")).strip() or None
    text = TAG.sub("", text).strip()
    return None if not text or text.startswith(NOT_WORDS) else text


def is_probe(r):
    """True for the developer's own test runs: a headless/SDK entrypoint (`claude -p`, sdk-cli) or a cwd under /tmp."""
    cwd = r.get("cwd") if isinstance(r.get("cwd"), str) else ""
    return str(r.get("entrypoint") or "").startswith("sdk") or cwd.startswith(("/tmp/", "/private/tmp/")) or cwd in ("/tmp", "/private/tmp")


def claude_source(path, off):
    """-> (messages, new_offset). messages: {ts, kind, text, k}."""
    out, end, prev = [], off, None   # prev: the last assistant text, narration ("final": False) when a tool call follows it directly
    for _, end, r in _lines(path, off):
        if not r or r.get("isSidechain"):
            continue
        ts, t = r.get("timestamp"), r.get("type")
        if t == "assistant":
            body = (r.get("message") or {}).get("content")
            if prev and isinstance(body, list) and any(isinstance(b, dict) and b.get("type") == "tool_use" for b in body):
                prev["final"] = False
        if t != "assistant" or _blocks((r.get("message") or {}).get("content")).strip():
            prev = None
        if t == "user" and not (r.get("isMeta") or r.get("isCompactSummary")):
            text, kind, key = user_words((r.get("message") or {}).get("content")), "user", r.get("uuid")
        elif t == "attachment" and (r.get("attachment") or {}).get("type") == "queued_command" and ((r["attachment"].get("origin") or {}).get("kind") == "human"):
            a = r["attachment"]  # a message the user typed while the agent worked: it reaches the log only through this record
            text, kind, key, ts = user_words(a.get("prompt")), "user", a.get("delivery_id") or r.get("uuid"), a.get("timestamp") or ts
        elif t == "assistant" and (r.get("message") or {}).get("model") != "<synthetic>":
            text, kind, key = _blocks((r.get("message") or {}).get("content")).strip(), "agent", r.get("uuid")
        else:
            continue
        if text and key and _ts(ts):
            out.append({"ts": _ts(ts), "kind": kind, "text": text, "k": "c:" + str(key), **({"probe": True} if is_probe(r) else {})})
            prev = out[-1] if kind == "agent" else None
    return out, end


def codex_source(path, off):
    out, end = [], off
    for start, end, r in _lines(path, off):
        if not r:
            continue
        p = r.get("payload") if isinstance(r.get("payload"), dict) else {}
        if r.get("type") == "session_meta":
            src = p.get("source")
            if p.get("thread_source") == "subagent" or (isinstance(src, dict) and "subagent" in src):
                return [], 1 << 62  # a subagent rollout is not the user's chat; the oversized cursor restarts at line 1 and stops there again
        elif r.get("type") == "response_item" and p.get("type") == "message" and p.get("role") in ("user", "assistant"):
            if p["role"] == "user":
                parts = [clean_turn(b.get("text", "")) for b in p.get("content") or [] if isinstance(b, dict) and b.get("type") == "input_text"]
                text, kind = "\n".join(x for x in parts if x), "user"
            else:  # every assistant text, commentary included, like Claude's mid-turn text
                text, kind = _blocks(p.get("content"), ("output_text",)).strip(), "agent"
            if text and _ts(r.get("timestamp")):
                out.append({"ts": _ts(r["timestamp"]), "kind": kind, "text": text, "k": f"x:{Path(path).stem[-36:]}:{start}",
                            "final": p.get("phase") != "commentary"})
    return out, end


def antigravity_source(path, off):
    out, end = [], off
    sid = Path(path).parents[2].name
    for _, end, r in _lines(path, off):
        if not r:
            continue
        ts, key = r.get("created_at"), f"a:{sid}:{r.get('step_index')}"
        if r.get("source") == "USER_EXPLICIT" and r.get("type") == "USER_INPUT":
            t = r.get("content") or ""
            m = re.search(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", t, re.S)
            text, kind = clean_turn(m[1] if m else re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", "", t, flags=re.S)), "user"
        elif r.get("source") == "MODEL" and r.get("type") == "PLANNER_RESPONSE" and isinstance(r.get("content"), str):
            text, kind = r["content"].strip(), "agent"
        else:
            continue
        if text and _ts(ts):
            out.append({"ts": _ts(ts), "kind": kind, "text": text, "k": key, "final": not r.get("tool_calls")})
    return out, end


def source_cwd(src, path, scan=60):
    """The working directory a transcript was run in, from its first records; None when unknown (Antigravity names none)."""
    if src not in ("claude", "codex"):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for _, line in zip(range(scan), f):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                p = r.get("payload") if src == "codex" and isinstance(r, dict) else r
                if isinstance(p, dict) and isinstance(p.get("cwd"), str) and p["cwd"]:
                    return p["cwd"]
    except OSError:
        pass
    return None


def opted_out(src, path):
    """True when the transcript's project carries `.memrove/no-memory`: the hook's own convention (memrove_v3_hook.skip_session)."""
    cwd = source_cwd(src, path)
    if cwd is None:
        return False
    from memrove_v3_project import project_root
    try:
        return (project_root(cwd) / ".memrove" / "no-memory").exists()
    except OSError:
        return False


def source_files(home):
    home = Path(home)
    yield from (("claude", p, claude_source) for p in sorted(home.glob(".claude/projects/*/*.jsonl")))
    yield from (("codex", p, codex_source) for p in sorted(home.glob(".codex/sessions/**/*.jsonl")))
    yield from (("antigravity", p, antigravity_source) for p in sorted(home.glob(".gemini/antigravity/brain/*/.system_generated/logs/transcript*.jsonl")))


# ---- ingest -----------------------------------------------------------------------------------------------------------

def _load_json(path, default):
    try:
        v = json.loads(Path(path).read_text(encoding="utf-8"))
        return v if isinstance(v, type(default)) else default
    except (OSError, ValueError):
        return default


def write_json(path, data):
    """Atomic: temp file, fsync, rename."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _seen(d):
    path = d / "seen.txt"
    s = set(path.read_text(encoding="utf-8").split()) if path.exists() else set()
    for r in _tail(d, 200):  # a crash between the log write and seen.txt must not double a message
        if r.get("k"):
            s.add(_sha(r["k"]))
    return s


def _tail(d, n):
    for path in reversed(_files(d)):
        recs = list(read_file(path))
        if recs:
            return recs[-n:]
    return []


def ingest(state, home=None, since=None, limit=None, lean_before=None):
    """Read what the sources gained since the cursors, order it by timestamp, redact, split, append. Caller holds the lock.
    since: 'YYYY-MM-DD' - older messages and files untouched since then are skipped. limit: stop after that many source messages (tests, cache metric).
    lean_before: 'YYYY-MM-DD' - import mode for old history: before that day only the user's messages and the agent's final answers are kept (readers mark
    mid-turn narration "final": False); from that day on everything. The first value stays, like the floor. Dropped narration is counted per source in stats["lean_dropped"].
    -> stats {"src/kind": [messages, log messages], "repeat_pastes": n, "redacted": n, "new": n}"""
    home = Path(home) if home else Path.home()
    d = root(state)
    d.mkdir(parents=True, exist_ok=True)
    cursors, seen = _load_json(d / "cursors.json", {}), _seen(d)
    # The first --since of a chat stays its floor: ids are positions, so older history must never land after newer messages.
    conf = _load_json(d / "config.json", {})
    first = {k: v for k, v in (("since", since), ("lean_before", lean_before)) if v and k not in conf}
    if first:  # one write: two separate writes of a stale conf dropped the floor
        conf = {**conf, **first}
        write_json(d / "config.json", conf)
    since = conf.get("since") or since
    lean = _ts(conf["lean_before"] + "T00:00:00") if conf.get("lean_before") else None
    floor = _ts(since + "T00:00:00") if since else None
    cands, new_cursors = [], {}
    for n, (src, path, reader) in enumerate(source_files(home)):
        try:
            st = Path(path).stat()
        except OSError:
            continue
        off = cursors.get(str(path), 0)
        if st.st_size == off or (floor and st.st_mtime < floor.timestamp()):
            continue
        msgs, end = reader(path, off)
        new_cursors[str(path)] = end  # advances for an opted-out file too: it is read once, never logged
        if opted_out(src, path):
            continue
        cands += [dict(m, src=src, order=n) for m in msgs if not floor or m["ts"] >= floor]
    cands.sort(key=lambda m: (m["ts"], m["order"]))
    stats, items, fresh, pastes = {}, [], [], set()
    for m in cands:
        h = _sha(m["k"])
        if h in seen:
            continue
        fresh.append(h)
        if lean and m["ts"] < lean and m["kind"] == "agent" and m.get("final") is False:
            drop = stats.setdefault("lean_dropped", {})
            drop[m["src"]] = drop.get(m["src"], 0) + 1
            continue
        row = stats.setdefault(f"{m['src']}/{m['kind']}", [0, 0])
        if m["kind"] == "user" and len(m["text"].encode("utf-8")) >= PASTE_MIN:
            ph = "p:" + _sha(" ".join(m["text"].split()), 24)
            if ph in seen or ph in pastes:
                stats["repeat_pastes"] = stats.get("repeat_pastes", 0) + 1
                continue
            pastes.add(ph)
            fresh.append(ph)
        text, hits = redact(m["text"], state)
        stats["redacted"] = stats.get("redacted", 0) + hits
        pieces = split_text(text)
        row[0] += 1
        row[1] += len(pieces)
        for j, piece in enumerate(pieces):
            items.append({"kind": m["kind"], "text": piece, "date": m["ts"].isoformat(timespec="seconds"), "src": m["src"],
                          "k": m["k"] + (f"#{j}" if j else ""), **({"probe": True} if m.get("probe") else {})})
    if limit is not None:
        # keep whole source messages: cut the item list at a message boundary
        keep, seen_k = [], set()
        for it in items:
            base = it["k"].split("#")[0]
            if base not in seen_k and len(seen_k) >= limit:
                break
            seen_k.add(base)
            keep.append(it)
        items = keep
        new_cursors = {}  # a limited run is a measurement helper; the cursors stay where they were
        fresh = [_sha(k) for k in seen_k]
    append(state, items)
    if fresh:
        with open(d / "seen.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(fresh) + "\n")
            f.flush()
            os.fsync(f.fileno())
    if new_cursors:
        write_json(d / "cursors.json", {**cursors, **new_cursors})
    stats["new"] = len(items)
    return stats
