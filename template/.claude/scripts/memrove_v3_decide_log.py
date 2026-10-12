"""Decision log for the local decision model's eval sets: same JSONL schema as the external decide tool
(~/.local/share/memrove/decision-log/log/decisions.jsonl, append-only, outcomes as later lines, latest wins). Stdlib only,
no import of decide. Every public function fails open: a log problem never costs a hook, a worker or a command."""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

STATE_INLINE_MAX = 2000
TAIL_BYTES = 4_000_000  # lookups read only the newest part of the log
SECRETS = re.compile(
    r"sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}"
    r"|xox[abprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{30,}|Bearer\s+[A-Za-z0-9._\-]{16,}"
    r"|(?i:(?:api[_-]?key|secret|token|passwd|password)[\"']?\s*[:=]\s*[\"']?)[^\s\"',;]{8,}")


def off():
    """Off by switch, and under unittest unless the test points DECIDE_HOME at a temp dir (the suite must not write the real log)."""
    return os.environ.get("MEMROVE_NO_DECIDE_LOG") == "1" or ("unittest" in sys.modules and not os.environ.get("DECIDE_HOME"))


def path():
    return Path(os.environ.get("DECIDE_HOME") or Path.home() / ".local/share/memrove/decision-log") / "log" / "decisions.jsonl"


def _redact(o):
    if isinstance(o, str):
        return SECRETS.sub("[REDACTED]", o)
    if isinstance(o, list):
        return [_redact(x) for x in o]
    if isinstance(o, dict):
        return {k: _redact(v) for k, v in o.items()}
    return o


def _sha(s, n):
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:n]


def _append(rec):
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(_redact(rec), ensure_ascii=False) + "\n").encode()
    fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def log(use, question, options=None, answer=None, confidence=None, source="deterministic", state=None, ts=None):
    """One decision line -> its id, or None when nothing could be written."""
    if off():
        return None
    try:
        rec = {"use": use, "question": question, "options": options, "answer": answer,
               "confidence": confidence, "source": source, "ts": ts or time.time(), "outcome": None}
        s = state if isinstance(state, str) or state is None else json.dumps(state, ensure_ascii=False, sort_keys=True)
        if s is not None and len(s) > STATE_INLINE_MAX:
            rec["state_sha"], rec["state_ref"] = _sha(s, 40), None
        else:
            rec["state"], rec["state_ref"] = s, None
        rec["id"] = f"{use}-{_sha(use + str(rec['ts']) + question + str(answer) + (s or ''), 12)}"
        _append(rec)
        return rec["id"]
    except Exception:
        return None


def set_outcome(decision_id, outcome):
    if off():
        return
    try:
        _append({"id": decision_id, "outcome": outcome, "ts": time.time()})
    except Exception:
        pass


def _tail(use):
    """Decisions of one `use` from the newest part of the log, outcomes merged in. -> list of dicts (oldest first)."""
    p = path()
    by, order = {}, []
    with p.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - TAIL_BYTES))
        data = f.read().decode("utf-8", "replace")
    for line in data.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "use" in r:
            if r["use"] == use:
                by[r["id"]] = r
                order.append(r["id"])
        elif r.get("id") in by:
            by[r["id"]]["outcome"] = r["outcome"]
    return [by[i] for i in dict.fromkeys(order)]


def _state(r):
    try:
        s = json.loads(r.get("state") or "{}")
        return s if isinstance(s, dict) else {}
    except ValueError:
        return {}


def log_recall(prompt, session, project, candidates, injected, extra=None):
    """candidates: [{id, score?, gate, injected, s1?, s2?}]; injected: ids put into the prompt; extra (abstain, mode, embedder, stage2 ...) joins the state."""
    top = max((c.get("score") or 0 for c in candidates if c.get("injected")), default=None)
    return log("memrove.recall", prompt, options=candidates, answer=list(injected), confidence=top,
               state={"session": session, "project": project, **(extra or {})})


def log_capture_kind(text, kind, record_id, stage, source="model", confidence=None, project=None, session=None, options=None, **extra):
    """A kind was assigned to a record (J2 creation, J1 verdict, `memrove remember`)."""
    return log("memrove.capture_kind", text, options=options or ["decision", "rejected", "lesson", "preference", "note"], answer=kind,
               confidence=confidence, source=source,
               state={"record": record_id, "stage": stage, "project": project, "session": session, **extra})


def kind_corrected(text, record_id, new_kind, by):
    """Mark earlier capture_kind labels of this record (or this exact text) that said something else as corrected."""
    try:
        for r in _tail("memrove.capture_kind"):
            if (_state(r).get("record") == record_id or r.get("question") == text) and r.get("answer") != new_kind \
                    and (r.get("outcome") or {}).get("corrected_to") != new_kind:
                set_outcome(r["id"], {"corrected_to": new_kind, "by": by})
    except Exception:
        pass


RANK = {"ignored": 0, "opened": 1, "cited": 2, "used": 2}  # "used" = the old name of "cited" in earlier outcome lines
USE_SKIP = ("fs-", "rules:more", "ent:")  # always-loaded files, the rules overflow pointer and alias cards have no id the model could open or name
_SHOW = re.compile(r"\bshow\s+[\"']?([^\s\"';|&)]+)")


def log_inject(session, project, source, epoch, blocks, ts=None):
    """One session-start injection (RULES, THREADS, LAST SESSION ...); blocks: [{id, section}]. Same outcome flow as a recall decision."""
    return log("memrove.inject", f"session-start:{source}", options=blocks, answer=[b["id"] for b in blocks],
               state={"session": session, "project": project, "epoch": epoch, "source": source}, ts=ts)


def _iso(s):
    from datetime import datetime  # lazy: only the Stop worker needs it
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def scan_use(transcript, needles):
    """needles: {id: [lowercase strings that name it]} -> {id: [(ts, 'opened'|'cited')]} from assistant tool calls and text.
    opened = a `memrove show <id>` command (or a Read of the doc path); cited = id or key/title in assistant text or any tool input."""
    flat = {n: i for i, ns in needles.items() for n in ns if n}
    if not flat:
        return {}
    quick = re.compile("|".join(re.escape(n) for n in sorted(flat, key=len, reverse=True)))
    hits, last = {}, float("inf")  # a line without a timestamp counts as after the injection
    with open(transcript, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"assistant"' not in line:
                continue
            low = line.lower()
            if not quick.search(low):
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            last = _iso(msg.get("timestamp")) or last
            m = msg.get("message") if isinstance(msg.get("message"), dict) else msg
            if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
                continue
            for block in m["content"]:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    inp = block.get("input") or {}
                    cmd = str(inp.get("command") or "")
                    shown = {x.lower() for x in _SHOW.findall(cmd)} if "memrove" in cmd else set()
                    blob = json.dumps(inp, ensure_ascii=False).lower()
                    for i, ns in needles.items():
                        if any(i.lower() == s or (i.lower().startswith(s) and len(s) >= 6) for s in shown):
                            hits.setdefault(i, []).append((last, "opened"))
                        elif i.startswith("doc:") and block.get("name") == "Read" and any(n in blob for n in ns):
                            hits.setdefault(i, []).append((last, "opened"))
                        elif any(n in blob for n in ns):
                            hits.setdefault(i, []).append((last, "cited"))
                elif block.get("type") == "text":
                    t = str(block.get("text")).lower()
                    for i, ns in needles.items():
                        if any(n in t for n in ns):
                            hits.setdefault(i, []).append((last, "cited"))
    return hits


def card_status(hits, since):
    """Best signal at or after `since` (the injection time) -> 'cited' > 'opened' > 'ignored'."""
    best = "ignored"
    for ts, st in hits or ():
        if ts >= since - 1 and RANK[st] > RANK[best]:
            best = st
    return best


def needles_for(bid, extra=()):
    """What names a block id: the id itself, a th:/ls:/doc: suffix, and any key/title the caller knows (>= 10 chars, else too generic)."""
    out = {bid.lower()}
    if bid.split(":", 1)[0] in ("th", "ls"):
        out.add(bid.split(":", 1)[1].lower())
    elif bid.startswith("doc:"):
        out.add(bid[4:].lower())
    out.update(e.lower() for e in extra if e and len(e) >= 10)
    return sorted(out)


def resolve_session(session, transcript, lookup=None, credit=None):
    """Stop/SessionEnd worker: label this session's recall and session-start decisions per injected id (cited > opened > ignored).
    Only signals after the injection count. Latest outcome wins in the log, so a later Stop may upgrade an earlier `ignored`.
    lookup(id) -> extra needles (record key/title); credit(id, new_status, prior_status) is called once per id when a session first reaches a higher status."""
    try:
        todo = [r for u in ("memrove.recall", "memrove.inject") for r in _tail(u) if _state(r).get("session") == session and r.get("answer")]
        if not todo or not transcript or not Path(transcript).is_file():
            return 0
        ids = sorted({i for r in todo for i in r["answer"] if not i.startswith(USE_SKIP)})
        hits = scan_use(transcript, {i: needles_for(i, lookup(i) if lookup else ()) for i in ids})
        prior = {}
        for r in todo:
            for i, st in ((r.get("outcome") or {}).get("cards") or {}).items():
                prior[i] = max(prior.get(i, "ignored"), st, key=RANK.get)
        done, best = 0, dict(prior)
        for r in sorted(todo, key=lambda r: r.get("ts") or 0):
            cards = {i: card_status(hits.get(i), r["ts"]) if not i.startswith(USE_SKIP) else "ignored" for i in r["answer"]}
            new = {"status": max(cards.values(), key=RANK.get), "cards": cards}
            if (r.get("outcome") or {}) != new:
                set_outcome(r["id"], new)
                done += 1
            for i, st in cards.items():
                if RANK[st] > RANK[best.get(i, "ignored")]:
                    if credit:
                        credit(i, st, best.get(i, "ignored"))
                    best[i] = st
        return done
    except Exception:
        return 0
