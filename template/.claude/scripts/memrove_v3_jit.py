#!/usr/bin/env python3
"""Just-in-time memory rule: SubagentStart, SessionStart resume/clear and PreToolUse(AskUserQuestion). Stdlib only, never blocks.

Standalone command (settings entries for SubagentStart and PreToolUse) and a library for memrove_v3_hook (SessionStart).
"""
import json
import sys

RULE = (
    "Before you answer or start a task that may build on earlier work, search first; a hit or a doc line is a lead, not evidence; "
    "open it with zoom or memrove doc and answer from what you opened. On one topic the newer statement wins, and the user's word beats an agent's."
)
SUBAGENT_RULE = RULE + ' Without the MCP tools use the CLI: memrove search "<words>" / memrove zoom <id> <n>.'
ASK_HINT = ("Before you ask the owner, check memory: `memrove search` for this decision; "
            "the owner should not have to repeat a settled answer.")
# One-lookup and UI agents (scout, Explore, ui-reviewer) get nothing.
SUBAGENT_TYPES = frozenset({"coder", "worker", "deep", "researcher", "general-purpose", "fork"})
RESUME_SOURCES = frozenset({"resume", "clear"})


def _context(event, text):
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


def handle(payload):
    """The hook output dict for one payload, or None for no injection. Never a deny."""
    if not isinstance(payload, dict):
        return None
    event = payload.get("hook_event_name")
    if event == "SubagentStart":
        kind = str(payload.get("agent_type") or "").rsplit(":", 1)[-1]  # plugin agents arrive as plugin:name
        return _context(event, SUBAGENT_RULE) if kind in SUBAGENT_TYPES else None
    if event == "SessionStart":
        return _context(event, RULE) if payload.get("source") in RESUME_SOURCES else None
    if event == "PreToolUse" and payload.get("tool_name") == "AskUserQuestion":
        return _context(event, ASK_HINT)
    return None


def main():
    try:
        out = handle(json.loads(sys.stdin.read(1_000_000) or "{}"))
    except Exception:
        out = None
    print(json.dumps(out) if out else "{}")


if __name__ == "__main__":
    main()
