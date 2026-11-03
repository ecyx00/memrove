"""Reader: ChatGPT account exports (zips under ~/Desktop/web-chats holding conversations*.json). The JSON is read straight from the zip.

The mapping is a tree (edits, regenerations): only the active branch is read, from current_node up to the root. Hidden and non-text
nodes (tool calls, thoughts, system context) are dropped. The export carries no project name -> project "unknown"."""
import json
import zipfile
from pathlib import Path

import memrove_v3_chatlog as chatlog

NAME = "chatgpt"


def _members(path):
    with zipfile.ZipFile(path) as z:
        return z, [n for n in z.namelist() if n.startswith("conversations") and n.endswith(".json") and "/" not in n]


def files(home):
    out = []
    for p in sorted((Path(home) / "Desktop/web-chats").glob("*.zip")):
        try:
            if _members(p)[1]:
                out.append(p)
        except zipfile.BadZipFile:
            pass
    return out


def _text(msg):
    c = msg.get("content") or {}
    if c.get("content_type") not in ("text", "multimodal_text") or (msg.get("metadata") or {}).get("is_visually_hidden_from_conversation"):
        return ""
    return "\n".join(p for p in c.get("parts") or [] if isinstance(p, str)).strip()


def _chain(conv):
    m, node, out = conv.get("mapping") or {}, conv.get("current_node"), []
    while node in m:
        out.append(m[node])
        node = m[node].get("parent")
    return out[::-1]


def read(path):
    out = []
    with zipfile.ZipFile(path) as z:
        for name in _members(path)[1]:
            for conv in json.loads(z.read(name)):
                turns = []
                for node in _chain(conv):
                    msg = node.get("message") or {}
                    role = (msg.get("author") or {}).get("role")
                    if role in ("user", "assistant") and msg.get("create_time") and (t := _text(msg)):
                        turns.append((role, msg["create_time"], node["id"], t))
                sid = conv.get("conversation_id") or conv.get("id")
                for i, (role, ts, nid, t) in enumerate(turns):
                    last = role == "assistant" and (i + 1 == len(turns) or turns[i + 1][0] == "user")   # later assistant text in a turn is the answer
                    out.append({"ts": chatlog._ts(ts), "kind": "user" if role == "user" else "agent", "text": t, "k": f"g:{sid}:{nid}",
                                "final": role == "user" or last, "session": sid, "project": "unknown"})
    return out
