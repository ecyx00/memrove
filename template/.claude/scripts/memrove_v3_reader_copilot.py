"""Reader: GitHub Copilot CLI/VS Code chat (~/.copilot/history-session-state/session_*.json). The timeline holds the user and copilot
texts with real timestamps (chatMessages has none). session-state/ only holds checkpoints, no chat. No cwd is stored, so the project
comes from the most common directory in the tool-call paths."""
import json
import re
from collections import Counter
from pathlib import Path

import memrove_v3_chatlog as chatlog

NAME = "copilot"


def files(home):
    return sorted(Path(home).glob(".copilot/history-session-state/session_*.json"))


def read(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    tl, sid = d.get("timeline") or [], d.get("sessionId") or Path(path).stem
    dirs = Counter(str(Path(p).parent) for e in tl if e.get("type") == "tool_call_completed"
                   for p in re.findall(r"""['"]path['"]: ?['"](/Users/[^'"]+)""", str(e.get("arguments"))))
    cwd = dirs.most_common(1)[0][0] if dirs else None
    seq = [(i, e) for i, e in enumerate(tl) if e.get("type") in ("user", "copilot") and (e.get("expandedText") or e.get("text") or "").strip()]
    out = []
    for j, (i, e) in enumerate(seq):
        user = e["type"] == "user"
        text = (e.get("expandedText") or e["text"]).strip() if user else e["text"].strip()
        ts = chatlog._ts(e.get("timestamp"))
        if ts:
            out.append({"ts": ts, "kind": "user" if user else "agent", "text": text, "k": f"c:{sid}:{e.get('id') or i}",
                        "final": user or j + 1 == len(seq) or seq[j + 1][1]["type"] == "user", "session": sid, "cwd": cwd})
    return out
