#!/usr/bin/env python3
"""Just-in-time memory rule: SubagentStart, SessionStart resume/clear and PreToolUse(AskUserQuestion). Stdlib only, never blocks.

Standalone command (settings entries for SubagentStart and PreToolUse) and a library for memrove_v3_hook (SessionStart).
"""
import json
import sys
from pathlib import Path

RULE = (
    "Before you answer or start a task that may build on earlier work, search first; a hit or a doc line is a lead, not evidence; "
    "open it with zoom or memrove doc and answer from what you opened. On one topic the newer statement wins, and the user's word beats an agent's. "
    "Start every docs/state write-up with front matter: topic (slug), status (open|done|rejected), decision (one sentence with why), next (one line)."
)
SUBAGENT_RULE = RULE + ' Without the MCP tools use the CLI: memrove search "<words>" / memrove zoom <id> <n>.'
SHORT_RULE = ('Search memory before you start (memrove search "<words>" / memrove zoom <id> <n>); a hit is a lead: open it before you rely on it. '
              "The newer statement wins and the owner's word beats an agent's. Start every docs/state write-up with front matter: topic (slug), status (open|done|rejected), decision (one sentence with why), next (one line).")
RECALL_SECONDS = 2
import os
SELECTOR_SECONDS = float(os.environ.get("MEMROVE_SELECTOR_WAIT") or 2)   # 6 once the SubagentStart hook timeout is 9 s (install_v3 sets it); the live settings still say 3 s
ASK_HINT = ("Before you ask the owner, check memory: `memrove search` for this decision; "
            "the owner should not have to repeat a settled answer.")
# One-lookup and UI agents (scout, Explore, ui-reviewer) get nothing.
SUBAGENT_TYPES = frozenset({"coder", "worker", "deep", "researcher", "general-purpose", "fork"})
WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit"})
RESUME_SOURCES = frozenset({"resume", "clear"})


def _context(event, text):
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


def _expose(state, payload, path, ids, slot):
    """Exposure log row per shown id; never raises."""
    try:
        import memrove_v3_exposure as ex
        ex.log(state, path, ids, slot, ex.key(payload.get("session_id")))
    except Exception:
        pass


def _rules(payload):
    """The global and session-project standing rules (short cap) for a subagent; '' when off or on any error. State: MEMROVE_STATE, else the newest memrove-v3 store."""
    try:
        import os
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from memrove_v3_rules import block
        state = os.environ.get("MEMROVE_STATE")
        if not state:
            base = Path.home() / "Library/Application Support/memrove-v3" if sys.platform == "darwin" else Path.home() / ".local/share/memrove-v3"
            found = sorted(base.glob("*/memory.sqlite3"), key=lambda p: p.stat().st_mtime)
            state = found[-1].parent if found else None
        from memrove_v3_project import project_name
        if not state:
            return ""
        shown = []
        text = block(state, project_name(payload.get("cwd"), state), ids=shown)
        _expose(state, payload, "jit", shown, "rules")
        return "\n" + text   # no CLAUDE.md filter: a subagent may not load the files the session loaded
    except Exception:
        return ""


def _recalled(payload, state):
    """SHORT_RULE + the 8 global rules + memory matching the agent's brief; '' (caller falls back to the full standing rules) on any error, timeout or when memory is off. MEMROVE_SUBRECALL=0 turns it off."""
    import os
    import signal
    if os.environ.get("MEMROVE_SUBRECALL") == "0" or os.environ.get("MEMROVE_RULES") == "0":
        return ""
    def late(*_):
        raise TimeoutError
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        state = state or os.environ.get("MEMROVE_STATE")
        if not state:
            return ""
        import memrove_v3_subrecall as sr
        from memrove_v3_project import project_name
        from memrove_v3_rules import short_block
        signal.signal(signal.SIGALRM, late)
        signal.alarm(RECALL_SECONDS)
        try:
            brief = sr.brief_of(state, payload)
            picks = None
            if brief and os.environ.get("MEMROVE_SELECTOR") == "1":   # opt-in: off until it beats dense+cards live
                import memrove_v3_selector as sel
                signal.alarm(int(SELECTOR_SECONDS) + 2)   # the selector job was started at PreToolUse(Agent); wait for it, then fall back to dense/bm25
                picks = sel.wait(state, brief, SELECTOR_SECONDS)
                if picks is None and payload.get("agent_id"):   # not ready: the picks go out on the agent's first tool call
                    (Path(state) / "selector" / f"late-{payload['agent_id']}").write_text(sel.key_of(brief))
            shown, rule_ids = [], []
            found = sr.block(state, brief, project_name(payload.get("cwd"), state), picks, ids=shown) if brief else ""
            rules = short_block(state, ids=rule_ids)
            _expose(state, payload, "jit", shown, "recall")
            _expose(state, payload, "jit", rule_ids, "rules")
        finally:
            signal.alarm(0)
        if os.environ.get("MEMROVE_RECALL_DEBUG"):
            with open(os.environ["MEMROVE_RECALL_DEBUG"], "a") as f:
                f.write(json.dumps({"keys": sorted(payload), "brief": brief[:200], "recall": found}, ensure_ascii=False) + "\n")
        return "\n".join(x for x in (SHORT_RULE, rules.rstrip("\n"), found) if x) if rules or found else ""
    except BaseException as e:
        if os.environ.get("MEMROVE_RECALL_DEBUG"):
            with open(os.environ["MEMROVE_RECALL_DEBUG"], "a") as f:
                f.write(json.dumps({"error": repr(e), "keys": sorted(payload)}) + "\n")
        return ""


def handle(payload, state=None):
    """The hook output dict for one payload, or None for no injection. Never a deny."""
    if not isinstance(payload, dict):
        return None
    event = payload.get("hook_event_name")
    if event == "SubagentStart":
        kind = str(payload.get("agent_type") or "").rsplit(":", 1)[-1]  # plugin agents arrive as plugin:name
        if kind not in SUBAGENT_TYPES:
            return None
        return _context(event, _recalled(payload, state) or SUBAGENT_RULE + _rules(payload))
    if event == "SessionStart":
        return _context(event, RULE) if payload.get("source") in RESUME_SOURCES else None
    if event == "PreToolUse":
        late = _late(payload, state)
        if late:
            return late
        if payload.get("tool_name") == "Agent" and state:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import memrove_v3_subrecall
            memrove_v3_subrecall.stash(state, payload)
        hit = _trigger(payload, state)
        if hit:
            return hit
        if payload.get("tool_name") == "AskUserQuestion":
            return _context(event, ASK_HINT)
        return None
    if event == "PostToolUse" and payload.get("tool_name") in WRITE_TOOLS:  # docs/state write-up without front matter: a one-line reminder
        path = (payload.get("tool_input") or {}).get("file_path") or ""
        if path.endswith(".md") and "/docs/state/" in path and not _opted_out(path):
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import memrove_v3_heads
            text = memrove_v3_heads.check(state, path)
            return _context(event, text) if text else None
    return None


def _late(payload, state):
    """Selector picks that were not ready at SubagentStart, delivered once on the subagent's first tool call (needs PreToolUse wired for that tool)."""
    try:
        agent = payload.get("agent_id")
        marker = Path(state) / "selector" / f"late-{agent}" if agent and state else None
        if not marker or not marker.exists():
            return None
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import memrove_v3_selector as sel
        import memrove_v3_subrecall as sr
        res = sel.out_path(state, marker.read_text().strip())
        if not res.exists():
            return None
        marker.unlink()
        picks = json.loads(res.read_text(encoding="utf-8")).get("picks")
        shown = []
        text = sr.block(state, "", None, picks, ids=shown) if picks else ""
        _expose(state, payload, "jit", shown, "late")
        return _context("PreToolUse", text.replace("Memory for this task", "More memory for this task", 1)) if text else None
    except Exception:
        return None


def _trigger(payload, state):
    """A lesson whose trigger matches this tool call, from <state>/triggers.json (never the DB). Warn once per session+agent; deny every time. Never modifies tool input."""
    try:
        import hashlib
        import re
        tool = payload.get("tool_name")
        entries = json.loads((Path(state) / "triggers.json").read_text(encoding="utf-8"))
        text = cwd = None
        for e in entries:
            if tool not in e["tools"]:
                continue
            ti = payload.get("tool_input")
            text = (ti["command"] if isinstance(ti, dict) and isinstance(ti.get("command"), str) else json.dumps(ti, ensure_ascii=False)) if text is None else text
            cwd = str(payload.get("cwd") or "") if cwd is None else cwd
            if not re.search(e["match"], text) or (e.get("cwd") and not re.search(e["cwd"], cwd)):
                continue
            msg = f"memrove lesson {e['id']} ({e['date']}): {e['text']}"
            if e.get("deny"):
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": msg}}
            who = hashlib.sha1(f"{payload.get('session_id')}\0{payload.get('agent_id') or ''}".encode()).hexdigest()
            seen = Path(state) / "triggers-seen" / who
            if seen.exists() and e["id"] in seen.read_text().split():
                continue
            seen.parent.mkdir(exist_ok=True)
            with seen.open("a") as f:
                f.write(e["id"] + "\n")
            return _context("PreToolUse", msg)
    except Exception:
        pass
    return None


def _opted_out(path):
    """True when a `.docs-guard-skip` marker sits in an ancestor of the file, below the home directory."""
    home = Path.home()
    for d in Path(path).parents:
        if d == home:
            return False
        if (d / ".docs-guard-skip").exists():
            return True
    return False


def main():
    state = sys.argv[sys.argv.index("--state") + 1] if "--state" in sys.argv[1:-1] else None
    try:
        out = handle(json.loads(sys.stdin.read(1_000_000) or "{}"), state)
    except Exception:
        out = None
    print(json.dumps(out) if out else "{}")


if __name__ == "__main__":
    main()
