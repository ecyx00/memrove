"""Real-use metric: do the injected memory lines fire in live transcripts, and are they followed? Read-only, stdlib only.

Reads <projects>/*/*.jsonl (Claude Code transcripts) and <session>/subagents/agent-*.jsonl. Probes (sdk entrypoint,
cwd under /tmp) are skipped. A turn = one owner prompt up to the next; "followed" = the turn called memrove search, zoom,
doc or project (CLI via Bash, or the memrove-chat MCP tool) before the final answer.
"""
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import memrove_v3_pastref as pastref

LINES = (pastref.LINE, "This prompt refers to or continues earlier work")  # current text, then the pre-e80 text
CALL = re.compile(r"\bmemrove\s+(search|zoom|doc|project)\b")
MCP = re.compile(r"^mcp__memrove-chat__(search|zoom|doc|project)$")


def _records(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    yield r
    except OSError:
        return


def _probe(recs):
    return any(str(r.get("entrypoint") or "").startswith("sdk") or str(r.get("cwd") or "").startswith(("/tmp", "/private/tmp")) for r in recs)


def _called_memrove(r):
    if r.get("type") != "assistant":
        return False
    for b in (r.get("message") or {}).get("content") or []:
        if isinstance(b, dict) and b.get("type") == "tool_use":
            if MCP.match(b.get("name") or "") or (b.get("name") == "Bash" and CALL.search((b.get("input") or {}).get("command") or "")):
                return True
    return False


def _hook_text(r, event):
    a = r.get("attachment") or {}
    if r.get("type") == "attachment" and a.get("type") == "hook_additional_context" and a.get("hookEvent") == event:
        return " ".join(str(c) for c in a.get("content") or [])
    return ""


def _owner_prompt(r):
    if r.get("type") != "user" or r.get("isSidechain") or r.get("isMeta"):
        return False
    c = (r.get("message") or {}).get("content")
    if isinstance(c, list):
        return any(isinstance(b, dict) and b.get("type") == "text" for b in c) and not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c)
    return isinstance(c, str) and bool(c.strip())


def _day(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d")
    except (AttributeError, ValueError):
        return ""


def stats(projects, since=""):
    """-> {"days": {day: counts}, "subagents": {agent_type: {total, fired, called_memrove}}}; `since` = 'YYYY-MM-DD' lower bound."""
    days = defaultdict(lambda: dict.fromkeys(("with_line", "with_line_followed", "without_line", "without_line_followed"), 0))
    subs = defaultdict(lambda: dict.fromkeys(("total", "fired", "called_memrove"), 0))
    for path in sorted(Path(projects).glob("*/*.jsonl")):
        recs = list(_records(path))
        if _probe(recs):
            continue
        turn = None  # [day, with_line, followed]
        done = []
        for r in recs:
            if _owner_prompt(r):
                turn = [_day(r.get("timestamp")), False, False]
                done.append(turn)
            elif turn:
                if any(x in _hook_text(r, "UserPromptSubmit") for x in LINES):
                    turn[1] = True
                elif _called_memrove(r):
                    turn[2] = True
        for day, line, followed in done:
            if day and day >= since:
                e = days[day]
                e["with_line" if line else "without_line"] += 1
                e["with_line_followed" if line else "without_line_followed"] += followed
        for meta in (path.parent / path.stem / "subagents").glob("agent-*.meta.json"):
            try:
                kind = json.loads(meta.read_text(encoding="utf-8")).get("agentType") or "?"
            except (OSError, ValueError):
                kind = "?"
            arecs = list(_records(meta.with_name(meta.name.replace(".meta.json", ".jsonl"))))
            if since and not any(_day(r.get("timestamp")) >= since for r in arecs if r.get("timestamp")):
                continue
            s = subs[kind]
            s["total"] += 1
            s["fired"] += any("search first" in _hook_text(r, "SubagentStart") for r in arecs)  # memrove's rule, not another hook's hint
            s["called_memrove"] += any(_called_memrove(r) for r in arecs)
    return {"days": dict(sorted(days.items())), "subagents": dict(sorted(subs.items()))}


def _pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def render(rep):
    out = ["real use (live transcripts, probes excluded); followed = memrove search/zoom/doc/project in the same turn",
           f"  {'day':<11}{'line':>6}{'followed':>10}{'rate':>6}   {'no line':>8}{'followed':>10}{'rate':>6}"]
    for d, e in rep["days"].items():
        out.append(f"  {d:<11}{e['with_line']:>6}{e['with_line_followed']:>10}{_pct(e['with_line_followed'], e['with_line']):>6}   "
                   f"{e['without_line']:>8}{e['without_line_followed']:>10}{_pct(e['without_line_followed'], e['without_line']):>6}")
    out.append(f"\nsubagents: {'type':<18}{'total':>6}{'hint fired':>11}{'called memrove':>13}{'share':>7}")
    for k, s in rep["subagents"].items():
        out.append(f"           {k[:18]:<18}{s['total']:>6}{s['fired']:>11}{s['called_memrove']:>13}{_pct(s['called_memrove'], s['total']):>7}")
    return "\n".join(out)


if __name__ == "__main__":
    print(render(stats(Path.home() / ".claude/projects")))
