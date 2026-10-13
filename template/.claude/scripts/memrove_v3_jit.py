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
ASK_HINT = ("Before you ask the owner, check memory: `memrove search` for this decision; "
            "the owner should not have to repeat a settled answer.")
# One-lookup and UI agents (scout, Explore, ui-reviewer) get nothing.
SUBAGENT_TYPES = frozenset({"coder", "worker", "deep", "researcher", "general-purpose", "fork"})
WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit"})
RESUME_SOURCES = frozenset({"resume", "clear"})


def _context(event, text):
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


def _rules(payload):
    """The global standing rules (short form) for a subagent; '' when off or on any error. State: MEMROVE_STATE, else the newest memrove-v3 store."""
    try:
        import os
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from memrove_v3_rules import block
        state = os.environ.get("MEMROVE_STATE")
        if not state:
            base = Path.home() / "Library/Application Support/memrove-v3" if sys.platform == "darwin" else Path.home() / ".local/share/memrove-v3"
            found = sorted(base.glob("*/memory.sqlite3"), key=lambda p: p.stat().st_mtime)
            state = found[-1].parent if found else None
        return ("\n" + block(state, global_only=True)) if state else ""
    except Exception:
        return ""


def handle(payload, state=None):
    """The hook output dict for one payload, or None for no injection. Never a deny."""
    if not isinstance(payload, dict):
        return None
    event = payload.get("hook_event_name")
    if event == "SubagentStart":
        kind = str(payload.get("agent_type") or "").rsplit(":", 1)[-1]  # plugin agents arrive as plugin:name
        return _context(event, SUBAGENT_RULE + _rules(payload)) if kind in SUBAGENT_TYPES else None
    if event == "SessionStart":
        return _context(event, RULE) if payload.get("source") in RESUME_SOURCES else None
    if event == "PreToolUse" and payload.get("tool_name") == "AskUserQuestion":
        return _context(event, ASK_HINT)
    if event == "PostToolUse" and payload.get("tool_name") in WRITE_TOOLS:  # docs/state write-up without front matter: a one-line reminder
        path = (payload.get("tool_input") or {}).get("file_path") or ""
        if path.endswith(".md") and "/docs/state/" in path:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import memrove_v3_heads
            text = memrove_v3_heads.check(state, path)
            return _context(event, text) if text else None
    return None


def main():
    state = sys.argv[sys.argv.index("--state") + 1] if "--state" in sys.argv[1:-1] else None
    try:
        out = handle(json.loads(sys.stdin.read(1_000_000) or "{}"), state)
    except Exception:
        out = None
    print(json.dumps(out) if out else "{}")


if __name__ == "__main__":
    main()
