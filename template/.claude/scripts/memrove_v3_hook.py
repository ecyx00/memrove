#!/usr/bin/env python3
"""Lifecycle hook for Claude Code and Codex. Hot path: print a short notice, queue metadata, start a detached worker.
Capture, sync and the model jobs run in that worker, never in the hook."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True  # a hook must not leave .pyc files in the vault
import time
import uuid

CAPTURE_EVENTS = {"Stop", "PreCompact", "SessionEnd"}
EVENTS = {"SessionStart", "UserPromptSubmit", "PostToolUse"} | CAPTURE_EVENTS
# Text the host itself puts in the prompt slot: nobody typed it, so it gets no recall.
AUTOMATED_PREFIXES = ("<task-notification>", "Another Claude session sent a message:", "<agent-message",
                      "<local-command-caveat>", "<command-name>", "<local-command-stdout>")
AUTOMATED_ORIGINS = frozenset({"task-notification", "peer", "coordinator"})
KEEP_TRANSCRIPTS = 300


def prompt_text(payload):
    """The prompt as a string; hosts may send null or a list of content blocks."""
    prompt = payload.get("prompt") if isinstance(payload, dict) else None
    if isinstance(prompt, list):
        prompt = "\n".join(p if isinstance(p, str) else str(p.get("text", "")) for p in prompt if isinstance(p, (str, dict)))
    return prompt if isinstance(prompt, str) else ""


def is_synthetic_prompt(payload):
    """True for turns no human typed (subagent results, peer messages, task notifications)."""
    if not isinstance(payload, dict) or os.environ.get("MEMROVE_V3_FILTER_HARNESS_TURNS") == "0":
        return False
    origin = payload.get("origin")
    kind = origin.get("kind") if isinstance(origin, dict) else origin
    if kind == "human":
        return False
    return kind in AUTOMATED_ORIGINS or prompt_text(payload).lstrip().startswith(AUTOMATED_PREFIXES)


def project_root(cwd, state=None):
    from memrove_v3_project import project_root as resolve
    return resolve(cwd, state)


def skip_session(payload):
    """Opt-outs: a .memrove/no-memory marker in the project, and headless sdk-cli runs."""
    if (project_root(payload.get("cwd")) / ".memrove" / "no-memory").exists():
        return True
    return is_sdk(payload) and not headless_recall()


def is_sdk(payload):
    if payload.get("entrypoint") == "sdk-cli" or os.environ.get("CLAUDE_CODE_ENTRYPOINT") == "sdk-cli":
        return True
    if payload.get("transcript_path"):
        from memrove_v3_capture import headless
        return headless(payload["transcript_path"])
    return False


def headless_recall():
    """Opt-in MEMROVE_HEADLESS_RECALL=1: a headless sdk-cli run gets read-side hook context only; main() queues nothing for it (ingest skips its transcript)."""
    return os.environ.get("MEMROVE_HEADLESS_RECALL") == "1"


def atomic(path, data):
    """Write JSON to a unique temp file next to path, flush it to disk, then rename it into place."""
    os.makedirs(path.parent, exist_ok=True)
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temp, "w", encoding="utf-8") as out:
        out.write(json.dumps(data, ensure_ascii=False))
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def note_error(state, exc, **extra):
    """Leave a trace for the doctor: hook-error.json holds the last failure and clears on the next clean drain."""
    atomic(Path(state) / "hook-error.json", {"at": time.time(), "error": type(exc).__name__, **extra})


def acknowledge(state, pending):
    """Move handled queue files to hook-done/. -> how many this call moved."""
    moved = 0
    os.makedirs(state / "hook-done", exist_ok=True)
    for path in pending:
        try:
            os.replace(path, state / "hook-done" / path.name)
            moved += 1
        except FileNotFoundError:
            pass  # another worker acknowledged it first
    return moved


def output_context(event, text):
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


def session_key(payload):
    return hashlib.sha256(str(payload.get("session_id", "unknown")).encode()).hexdigest()[:24]


def pastref_line(payload, state=None):
    """A one-line pointer for a prompt that refers to earlier work. Off unless MEMROVE_PASTREF=1 or <state>/pastref.on exists."""
    try:
        enabled = os.environ.get("MEMROVE_PASTREF") == "1" or (state and (Path(state) / "pastref.on").exists())
        if not enabled or is_synthetic_prompt(payload):
            return ""
        from memrove_v3_pastref import LINE, is_past_reference
        if not is_past_reference(prompt_text(payload)):
            return ""
        return LINE + (doc_pointers(payload, state) if os.environ.get("MEMROVE_PASTREF_POINTERS") == "1" else "")   # pointers cost 2.3 s against the 2 s hook timeout and were never opened
    except Exception:
        return ""


def doc_pointers(payload, state):
    """'\n'-led block of up to 3 `path — decision: ...` doc pointers (no content) for the prompt, or ''. Logged as a memrove.inject
    row (section `pointers`) so `memrove recall-stats` reports opened/cited/ignored for them."""
    try:
        import memrove_v3_decide_log as dl
        import memrove_v3_docsearch as ds
        from memrove_v3_project import sticky
        found = ds.pointers(state, prompt_text(payload), payload.get("cwd"))
        if not found:
            return ""
        session = session_key(payload)
        project = sticky(Path(state), session, payload.get("cwd") or ".", transcript=payload.get("transcript_path"))[0] or "unknown"
        dl.log_inject(session, project, "prompt", 0, [{"id": "doc:" + f.split(" — ")[0], "section": "pointers"} for f in found])
        return "\nDocs that may hold the answer (pointers only; open before acting):\n" + "\n".join(found)
    except Exception:
        return ""


def enqueue_event(vault, state, payload, harness) -> str:
    """Write one small JSON file per hook event into hook-queue/; the worker consumes it. -> queue key."""
    from memrove_v3_bridge import origin
    state = Path(state)  # the runtime directory, outside the vault
    event = {"event": payload.get("hook_event_name"), "harness": harness, "session": session_key(payload),
             "raw_session": str(payload.get("session_id", "unknown"))[:80]}
    event.update(origin(payload, harness) or {})
    if payload.get("no_memory") is True:
        event["no_memory"] = True
    project = None
    if payload.get("cwd"):  # the first resolvable event fixes the project of a session
        from memrove_v3_project import sticky
        project = sticky(state, event["session"], payload["cwd"], vault, payload.get("transcript_path"))[0]
    if payload.get("transcript_path") and event["event"] in CAPTURE_EVENTS:
        event.update(transcript_path=payload["transcript_path"], cwd=payload.get("cwd") or ".", project=project or "unknown")
    if isinstance(payload.get("prompt"), str) and payload["prompt"].strip():
        event["prompted"] = True  # the fact only, never the text
    key = hashlib.sha256(str(payload.get("event_id") or uuid.uuid4().hex).encode()).hexdigest()
    if not (state / "hook-queue" / f"{key}.json").exists() and not (state / "hook-done" / f"{key}.json").exists():
        atomic(state / "hook-queue" / f"{key}.json", dict(event, at=time.time()))
    return key


def _remember_transcripts(state, events):
    """session hash -> raw id and transcript path, so doctor can find the transcript of a session."""
    found = {e["session"]: {"raw": e.get("raw_session"), "path": e["transcript_path"]}
             for e in events if e.get("session") and e.get("transcript_path")}
    if found:
        path = Path(state) / "session-map.json"
        try:
            known = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            known = {}
        atomic(path, dict(list({**known, **found}.items())[-KEEP_TRANSCRIPTS:]))


def _label_sessions(engine, state, events):
    """Mark which recalled and injected records a finished session opened, cited or ignored."""
    ended = [e for e in events if e.get("event") in ("Stop", "SessionEnd") and e.get("transcript_path")]
    if not ended:
        return
    import memrove_v3_decide_log as decide_log
    from memrove_v3_outcomes import make_lookup, materialize
    from memrove_v3_projections import update_record
    lookup = make_lookup(state, engine.store._connect)

    def credit(rid, status, prior):
        with engine.store._connect() as db:
            row = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
        if row:
            update_record(engine, rid, {status + "_n": int(json.loads(row[0]).get(status + "_n") or 0) + 1})

    for e in ended:
        materialize(state, e.get("session"))
        decide_log.resolve_session(e.get("session"), e["transcript_path"], lookup=lambda i: lookup(i)["needles"], credit=credit)


def _model_jobs(engine, vault, state, events):
    """Projections, doc scan and the Haiku jobs; run at session end, before compaction, or when a sweep is due."""
    from memrove_v3_projections import ingest_verdicts, project_graph
    from memrove_v3_docs import scan as scan_docs
    from memrove_v3_project import sticky
    from memrove_v3_llm import run_jobs
    ingest_verdicts(engine)
    project_graph(engine)
    roots = {str(root) for e in events if e.get("cwd") and e.get("event") in ("SessionEnd", "PreCompact")
             for root in [sticky(state, e.get("session", ""), e["cwd"], vault)[1]] if root}
    for root in roots:
        scan_docs(engine.store, root)
    run_jobs(engine, vault, state, events, next(iter(roots)) if roots else None)
    project_graph(engine, budget_s=1.0)  # notes and proposals the jobs just wrote


def queued(state):
    return sorted(Path(state).joinpath("hook-queue").glob("*.json"))


def drain_queue(vault, state):
    """Worker: capture transcripts, sync Markdown, run jobs. Acknowledges queued events only after a clean sync."""
    state, pending = Path(state), queued(state)
    for path in list(pending):  # a poison event is set aside with a trace so it cannot block the ones behind it
        try:
            if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
                raise ValueError("event is not an object")
        except FileNotFoundError:
            pending.remove(path)
        except (OSError, ValueError) as exc:
            note_error(state, exc, quarantined=path.name)
            os.makedirs(state / "hook-bad", exist_ok=True)
            try:
                os.replace(path, state / "hook-bad" / path.name)
            except FileNotFoundError:
                pass
            pending.remove(path)
    fail = {"processed": 0, "failed": len(pending), "pending": len(pending)}
    try:
        from memrove_v3_sync import SyncEngine
        import memrove_v3_queue as queue
        from memrove_v3_capture import run as capture_run
        if (state / "v3-install.json").exists():
            from memrove_v3_companion import initialize
            initialize(vault, state)
        engine = SyncEngine(vault, state)
        events = sorted((json.loads(p.read_text(encoding="utf-8")) for p in pending if p.exists()), key=lambda e: e.get("at", 0))
        try:
            _label_sessions(engine, state, events)
        except Exception:  # noqa: BLE001 - labelling never costs the drain
            pass
        queue.register(state, events)  # before anything can fail: a session is never lost to a busy worker
        _remember_transcripts(state, events)
        for event in events:
            if event.get("transcript_path") and not event.get("no_memory"):
                try:
                    capture_run(engine, event)
                except Exception as exc:  # a bad transcript must not block the acknowledgement, but it leaves a trace
                    queue.record_error(state, "capture", exc)
        try:
            from memrove_v3_chattree import spawn  # chat memory is opt-in: only where state/chatlog/ exists
            spawn(state, vault)
        except Exception as exc:  # chat memory failing is logged, never fatal
            queue.record_error(state, "chatlog", exc)
        result = engine.sync(full=False)
        try:
            from memrove_v3_embed import refresh
            refresh(state)
        except Exception:  # vectors are a cache; recall still works without them
            pass
        if any(e.get("event") in ("SessionEnd", "PreCompact") for e in events) or queue.sweep_due(state):
            try:
                _model_jobs(engine, vault, state, events)
            except Exception as exc:  # the model jobs are best effort
                queue.record_error(state, "jobs", exc)
        if any(e.get("event") == "SessionEnd" for e in events):
            try:  # set point with an outflow: an over-limit companion file is archived verbatim, then trimmed
                from memrove_v3_compact import compact
                compact(vault, state)
            except Exception as exc:  # best effort; the doctor still reports an over-limit file
                queue.record_error(state, "compact", exc)
        if any(e.get("event") == "SessionEnd" for e in events):
            try:  # executable memory: re-run the checks of this session's projects; results are cached, nothing is superseded
                from memrove_v3_checks import run_all
                from memrove_v3_project import project_name
                run_all(engine.store, state, {project_name(e["cwd"], state, vault) for e in events if e.get("event") == "SessionEnd" and e.get("cwd")})
            except Exception as exc:  # best effort
                queue.record_error(state, "checks", exc)
        from memrove_v3_secrets import health
        result["secrets_redacted"] = health(state)["total"]
    except Exception as exc:
        note_error(state, exc)
        return fail
    atomic(state.joinpath("hook-health.json"), {"at": time.time(), "sync": result})
    # A degraded scan only left out invalid sources; a conflict means ownership is unclear, so nothing is acknowledged.
    if result.get("conflicts") or result.get("status") not in ("ok", "succeeded", "synced", "degraded"):
        return fail
    state.joinpath("hook-error.json").unlink(missing_ok=True)
    return {"processed": acknowledge(state, pending), "failed": 0, "pending": len(queued(state))}


def single_flight(state):
    """Non-blocking exclusive lock on <state>/hook-worker.lock. -> open file (keep it alive), or None when a worker already runs."""
    os.makedirs(state, exist_ok=True)
    lock = open(Path(state) / "hook-worker.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock
    except OSError:
        lock.close()
        return None


def run_worker(vault, state):
    """One heavy worker per state dir. A loser exits at once: its event is already in hook-queue/, and the holder
    re-drains until the queue is empty or a round makes no progress. After releasing, the queue is looked at again so
    an event that arrived between the last drain and the release is never stranded."""
    outcome = {"processed": 0, "failed": 0, "pending": 0}
    while True:
        lock = single_flight(state)
        if lock is None:
            return dict(outcome, deferred=True)
        gained = 0
        try:
            time.sleep(float(os.environ.get("MEMROVE_V3_TEST_HOLD", 0)))  # test knob: makes a short drain overlap its rivals
            while True:
                result = drain_queue(vault, state)
                gained += result["processed"]
                outcome["failed"], outcome["pending"] = result["failed"], result["pending"]
                if result["failed"] or not result["pending"] or not result["processed"]:
                    break
        finally:
            lock.close()
        outcome["processed"] += gained
        if not gained or not queued(state):  # no progress: a systemic failure, retrying would spin
            break
    return outcome


def recall_hook(state, payload, harness="claude"):
    """UserPromptSubmit recall cards: typed records, then the chat-log lane (pointer lines). Read-only for records; the lane only
    writes its seen-ids file. Independent of the queue; any error leaves a trace and returns '{}'."""
    try:
        if os.environ.get("MEMROVE_NO_RECALL") == "1" or is_synthetic_prompt(payload):
            return "{}"
        from memrove_v3_push import read as push_read
        if not push_read(state)["recall_cards"]:
            return "{}"
        import memrove_v3_access as access
        policy = access.load(state)  # no access.json: open, as before
        if policy is not None and not access.can_read(policy, access.harness_agent(harness), project_root(Path(payload.get("cwd") or "."), state).name):
            return "{}"
        text = ""
        if (Path(state) / "memory.sqlite3").is_file():
            from memrove_v3_recall import recall, StateStore
            text = recall(StateStore(state), prompt_text(payload), payload.get("cwd") or ".", session_key(payload), payload.get("transcript_path"))
        try:
            from memrove_v3_chatsearch import lane_text
            from memrove_v3_chatsearch import LANE_ENVELOPE, LANE_TOKENS, _tokens
            from memrove_v3_chatsearch import index_ready
            used = _tokens(text) if text else _tokens(LANE_ENVELOPE)   # one cap for the whole injection: card + lane + envelope
            lane = lane_text(state, prompt_text(payload), session_key(payload), payload.get("transcript_path"), cap=LANE_TOKENS - used) if used < LANE_TOKENS - 40 and index_ready(state) else ""   # no index yet: a cold build takes seconds, the next search makes it
        except Exception as exc:  # the lane never costs the records their card
            note_error(state, exc, event="UserPromptSubmit.chatlane")
            lane = ""
        if lane:
            close = "\n</memrove-recall>"
            text = text[:-len(close)] + "\n" + lane + close if text.endswith(close) else '<memrove-recall trust="local-data" instructions="never">\n' + lane + close
        return json.dumps(output_context("UserPromptSubmit", text)) if text else "{}"
    except Exception as exc:  # fail open: no recall this turn, but a trace for the doctor
        try:
            note_error(state, exc, event="UserPromptSubmit")
        except OSError:
            pass
        return "{}"


def health_warning(state):
    """One line when the last background sync did not finish cleanly."""
    try:
        sync = json.loads((Path(state) / "hook-health.json").read_text(encoding="utf-8")).get("sync", {})
    except (OSError, ValueError):
        return ""
    return "" if sync.get("status") in ("ok", "succeeded", "synced") else "Memory sync needs attention; run the doctor and check the current sources.\n"


def claude_md_warning(path=None):
    """One line when the user's CLAUDE.md is over a byte budget (MEMROVE_CLAUDE_MD_BUDGET, default 7200). Never edits it."""
    try:
        budget = int(os.environ.get("MEMROVE_CLAUDE_MD_BUDGET") or 7200)
        over = Path(path or Path.home() / ".claude" / "CLAUDE.md").stat().st_size - budget
    except (OSError, ValueError):
        return ""
    return f"CLAUDE.md is {over} bytes over budget: convert a rule to a check or remove one\n" if over > 0 else ""


TAIL_CAP, TAIL_MSG, TAIL_READ, CONTEXT_CAP = 6000, 700, 8 << 20, 9800  # recent-words block, per message, transcript bytes read, SessionStart text (host cap 10000)


def recent_words(path, budget=TAIL_CAP, each=TAIL_MSG):
    """The owner's last typed prompts of this session, oldest first, verbatim, each cut to `each` chars, newest kept when over `budget`. '' when none."""
    from memrove_v3_capture import has_secret, user_text
    try:
        with open(path, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - TAIL_READ))
            lines = f.read().decode("utf-8", "replace").splitlines()[1 if size > TAIL_READ else 0:]
    except OSError:
        return ""
    head, picked, used = "your recent words in this session (verbatim, oldest first; the compaction summary is lossy, these are not):\n", [], 0
    for ln in reversed(lines):
        try:
            rec = json.loads(ln)
            text = user_text(rec)
        except ValueError:
            continue
        origin = rec.get("origin")
        if (not text or (origin.get("kind") if isinstance(origin, dict) else origin) in AUTOMATED_ORIGINS
                or text.startswith(AUTOMATED_PREFIXES + ("This session is being continued",)) or has_secret(text)):
            continue
        item = "- " + (text if len(text) <= each else text[:each] + f" [cut: {len(text) - each} more chars]") + "\n"
        if used + len(item) + len(head) > budget:
            break
        picked.append(item)
        used += len(item)
    return head + "".join(reversed(picked)) if picked else ""


def session_notice(state, payload, reserve=0):
    """Text for SessionStart: heartbeat line plus the pull-memory instruction for this start source.
    `reserve` = chars the caller appends after; on compact the owner's recent words come before the heads view and the whole stays under CONTEXT_CAP."""
    from memrove_v3_heartbeat import line
    notice, source = line(state), payload.get("source")
    if source in ("startup", "compact"):
        try:
            from memrove_v3_chattree import instruction
            notice += instruction(state)
        except Exception:  # no chat memory here: nothing to announce
            pass
    elif source in ("resume", "clear"):
        from memrove_v3_jit import RULE
        notice += RULE + "\n"
    try:
        from memrove_v3_checks import notice as stale_notice
        notice += stale_notice(state)
    except Exception:  # a broken cache never costs the start notice
        pass
    if source == "compact" and payload.get("transcript_path"):
        try:
            notice += recent_words(payload["transcript_path"], min(TAIL_CAP, CONTEXT_CAP - reserve - len(notice)))
        except Exception:  # the tail never costs the start notice
            pass
    if source != "compact":  # heads are pull-able (memrove search); only a compaction loses context they help restore
        return notice
    from memrove_v3_heads import start_view  # the project's newest topic heads, inside the same view
    heads = start_view(state, payload.get("cwd") or ".")
    return notice + (heads if len(notice) + len(heads) + reserve <= CONTEXT_CAP else "")


def spawn_worker(vault, state, harness):
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--vault", str(vault), "--state", str(state),
                             "--harness", harness, "--worker"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)


def parse_args():
    ap = argparse.ArgumentParser(description="memrove lifecycle hook")
    for name, kind in (("--vault", Path), ("--state", Path)):
        ap.add_argument(name, required=True, type=kind)
    ap.add_argument("--harness", choices=("codex", "claude"), required=True)
    ap.add_argument("--event", help="event name, when the host does not put it in the payload")
    ap.add_argument("--global", action="store_true", dest="global_mode",
                    help="user-level install: project and opt-outs resolve from the payload cwd")
    ap.add_argument("--worker", action="store_true", help="run the detached queue worker")
    ap.add_argument("--drain-queue", action="store_true", help="process the queue once and exit")
    ap.add_argument("--metadata-only", action="store_true", help="queue lifecycle metadata and print nothing")
    return ap.parse_args()


def main():
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    args = parse_args()
    vault, state = args.vault.resolve(), args.state.resolve()
    if vault == state or vault in state.parents:
        raise ValueError("Runtime state must be outside the vault")
    os.umask(0o077)
    from memrove_v3_preferences import read, claim_check
    if args.worker or args.drain_queue:
        if args.worker and not read(vault)["auto_sync"]:
            print(json.dumps({"processed": 0, "failed": 0, "paused": True}))
            return
        outcome = run_worker(vault, state) if args.worker else drain_queue(vault, state)
        print(json.dumps(outcome))
        raise SystemExit(1 if outcome["failed"] else 0)
    event, notice = args.event, ""
    try:
        payload = json.loads(sys.stdin.read(1 << 20) or "{}")
        event = payload["hook_event_name"] = payload.get("hook_event_name", args.event)
        if (event not in EVENTS or os.environ.get("MEMROVE_V3_INTERNAL") or os.environ.get("MEMROVE_V3_SKIP") == "1"
                or os.environ.get("MEMROVE_INVOKED_BY") or payload.get("no_memory") is True or skip_session(payload)):
            print("{}")
            return
        read_only = headless_recall() and is_sdk(payload)  # reaching here with an sdk run means the flag let it through
        if read_only and event not in ("SessionStart", "UserPromptSubmit"):
            print("{}")
            return
        if event == "UserPromptSubmit" and args.global_mode and args.harness == "claude" and not args.metadata_only:
            out, line = recall_hook(state, payload, args.harness), pastref_line(payload, state)
            if line:
                cards = json.loads(out).get("hookSpecificOutput", {}).get("additionalContext", "")
                out = json.dumps(output_context(event, (cards + "\n" if cards else "") + line))
            print(out)
            return
        if not args.metadata_only:
            if event == "SessionStart":
                notice = health_warning(state) + claude_md_warning()
                if args.harness == "claude":
                    try:  # standing rules are pushed text (a rule works by being in context)
                        from memrove_v3_rules import block as rules_block
                        from memrove_v3_project import project_name
                        notice += rules_block(state, project_name(payload.get("cwd"), state, vault), cwd=payload.get("cwd"))
                    except Exception:
                        pass
                    try:  # the handoff card of the project's last session
                        from memrove_v3_card import block as card_block
                        from memrove_v3_project import project_name
                        notice += card_block(state, project_name(payload.get("cwd"), state, vault))
                        from memrove_v3_card import PROTOCOL
                        notice += PROTOCOL
                    except Exception:
                        pass
                notice = session_notice(state, payload, reserve=len(notice)) + notice
            elif event == "UserPromptSubmit":
                notice = pastref_line(payload, state)
        if read_only:
            print(json.dumps(output_context(event, notice)) if notice else "{}")
            return
        settings = read(vault)
        if settings["auto_sync"]:  # manual profile: no queueing, no worker
            enqueue_event(vault, state, payload, args.harness)
            if os.environ.get("MEMROVE_V3_NO_SPAWN") != "1" and (event in ("PreCompact", "SessionEnd") or claim_check(state, settings, event)):
                spawn_worker(vault, state, args.harness)
        print(json.dumps(output_context(event, notice)) if notice else "{}")
    except Exception as exc:
        note_error(state, exc)
        if not args.metadata_only and event in ("SessionStart", "UserPromptSubmit"):
            print(json.dumps(output_context(event, notice + "Memory sync failed; metadata stays queued. Run the doctor and check the current sources.")))
        else:
            print("{}")


if __name__ == "__main__":
    main()
