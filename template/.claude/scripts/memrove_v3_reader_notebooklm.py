"""Reader: Google Takeout NotebookLM (zip under ~/Desktop/web-chats). Chat History/*.html and Notes/*.html per notebook; Sources/Artifacts skipped.

A chat file is "USER: <html>" / "MODEL: <html>" blocks, newest first, with no timestamps: time = the notebook's createTime + 1 s per
message in chronological order (stable, so ids do not change). A note is a model answer the owner saved (kind agent, not the owner's words), time = createTime.
Project = the notebook title."""
import html
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

NAME = "notebooklm"
BASE = "Takeout/NotebookLM/"


def files(home):
    out = []
    for p in sorted((Path(home) / "Desktop/web-chats").glob("*.zip")):
        try:
            with zipfile.ZipFile(p) as z:
                if any(n.startswith(BASE) for n in z.namelist()):
                    out.append(p)
        except zipfile.BadZipFile:
            pass
    return out


def _plain(s):
    s = re.sub(r"</(p|h\d|li|ul|ol|div|pre|tr)>|<br\s*/?>", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(re.sub(r"<[^>]+>", "", s))).strip()


def read(path):
    out = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for meta in (n for n in names if n.startswith(BASE) and n.endswith(" metadata.json") and n.count("/") == 3):
            d = json.loads(z.read(meta))
            folder = meta.rsplit("/", 1)[0] + "/"
            t0 = datetime.fromisoformat(d["metadata"]["createTime"].replace("Z", "+00:00"))
            title = d.get("title") or "unknown"
            for n in (x for x in names if x.startswith(folder) and x.endswith(".html")):
                rel = n[len(folder):]
                if rel.startswith("Notes/"):
                    out.append({"ts": t0, "kind": "agent", "text": _plain(z.read(n).decode("utf-8")), "k": f"n:{d['id']}:{rel}", "final": True,
                                "session": f"note:{d['id']}", "project": title})
                elif rel.startswith("Chat History/"):
                    sid = rel.rsplit(" - ", 1)[-1][:-5]
                    parts = re.split(r"(?m)^(USER|MODEL): ", z.read(n).decode("utf-8"))
                    msgs = list(zip(parts[1::2], parts[2::2]))[::-1]
                    for i, (who, body) in enumerate(msgs):
                        out.append({"ts": t0.fromtimestamp(t0.timestamp() + i, t0.tzinfo), "kind": "user" if who == "USER" else "agent",
                                    "text": _plain(body), "k": f"nb:{sid}:{i}", "final": True, "session": sid, "project": title})
    return [m for m in out if m["text"]]
