"""Standing-rules block: the owner's own active rules as pushed text (SessionStart, after compaction, resume; global-only at SubagentStart).

A rule works by being in context, not by being opened, so this block is pushed on purpose while recall cards stay off.
A line is the restate step's sentence (memrove_v3_restate: self-contained English, subject and scope named), never the raw quote; a record
without a standing restate judgement is not a rule. Global rules go to every session, a project rule only to sessions of that project.
Reads <state>/memory.sqlite3 read-only. Kill switch: MEMROVE_RULES=0. Stdlib only, never raises.
"""
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import sqlite3

CAP = 1500          # bytes of the whole block
SUB_CAP = 700       # global-only form for subagents
MAX_WORDS, MAX_CHARS = 40, 200
TURKISH = re.compile("[çğıöşüÇĞİÖŞÜ]")
BARE = re.compile(r"^(?:don'?t|do|no|never|always|not|stop|avoid)\b", re.I)   # an imperative or a negation has no subject
KINDS = ("preference", "decision", "rejected")
GENERIC = ("", "unknown", ".claude", Path.home().name)   # labels of sessions started in the home directory: no project
HEAD = "Standing rules (owner's own; binding; newest wins):\n"


def _when(r):
    return str(r.get("valid_from") or r.get("recorded_at") or r.get("updated_at") or "")   # the day the owner said it; updated_at moves with every re-assessment


def _own(r):
    return r.get("origin_type") == "user_explicit" or r.get("verified_by") == "user_quote"


def _live(r):
    return not (r.get("restate_dup_of") or r.get("visibility") == "private" or r.get("verdict") == "reject" or r.get("valid_to") or r.get("status") == "deleted_by_user")


def _restated(r):
    """-> (sentence, scope, project) when the restate step judged the record a standing rule with a usable sentence, else None."""
    a = ((r.get("assessments") or {}).get("restate") or {})
    ans = a.get("answers") or {}
    sent = " ".join(str(ans.get("sentence") or "").split())
    if a.get("state") != "assessed" or ans.get("standing_rule") != "yes" or not sentence_ok(sent):
        return None
    scope = ans.get("scope")
    if scope not in ("global", "project") or (scope == "project" and not str(ans.get("project") or "").strip()):
        return None
    return sent, scope, str(ans.get("project") or "").strip()


def sentence_ok(s):
    """A whole English sentence that names its subject: no cut, no Turkish, not an imperative or a bare negation."""
    n = len(s.split())
    return bool(4 <= n <= MAX_WORDS and len(s) <= MAX_CHARS and s[-1] in ".!)" and "…" not in s and not TURKISH.search(s) and not BARE.match(s))


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def rules(state, project=None, global_only=False):
    """-> [(when, project_tag, sentence, id)]: project and global rules take turns, most-repeated then newest first. Duplicates and reversals are marked by the restate step (restate_dup_of)."""
    path = Path(state) / "memory.sqlite3"
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    try:
        rows = [json.loads(p) for (p,) in db.execute(
            "SELECT payload FROM records WHERE json_extract(payload,'$.kind') IN (?,?,?)", KINDS)]
    finally:
        db.close()
    dup = {r["id"]: r["restate_dup_of"] for r in rows if r.get("restate_dup_of")}
    said = {}   # how often the owner said a rule: the records merged into it, followed to the root
    for i in dup:
        root, hops = i, 0
        while root in dup and hops < 20:
            root, hops = dup[root], hops + 1
        said[root] = said.get(root, 0) + 1
    rows = [(r, _restated(r)) for r in rows if _own(r) and _live(r)]
    rows = [(r, x) for r, x in rows if x]
    gone = {s for r, _ in rows for s in (r.get("supersedes") or [])}
    # global: scope global; a project rule only for a session of that project (never in the subagent form)
    mine = [(r, x) for r, x in rows if r["id"] not in gone and (x[1] == "global" or (not global_only and project and _norm(x[2]) == _norm(project)))]
    # project rules and global rules take turns, each list ordered by how often the owner said the rule, then newest first:
    # a project rule is delivered nowhere else, a global one reaches every session
    mine.sort(key=lambda t: (-said.get(t[0]["id"], 0), _neg(_when(t[0]))))
    lists = [[t for t in mine if t[1][1] != "global"], [t for t in mine if t[1][1] == "global"]]
    order = [t for pair in zip_longest(*lists) for t in pair if t]
    return [(_when(r), "" if x[1] == "global" else x[2], x[0], r["id"]) for r, x in order]


def _neg(when):
    """Sort key that puts the newer date first inside an ascending sort."""
    return [-ord(c) for c in when]


def block(state, project=None, global_only=False, cap=None):
    """The block text ('' when off, empty or on any error). Whole sentences only; the rest is left out when the cap is reached."""
    if os.environ.get("MEMROVE_RULES") == "0":
        return ""
    cap = cap or (SUB_CAP if global_only else CAP)
    try:
        items = rules(state, project, global_only)
    except Exception:
        return ""
    out, used = [], len(HEAD.encode())
    for _, tag, rule, _id in items:
        line = "- " + rule + "\n"
        if used + len(line.encode()) <= cap:
            out.append(line)
            used += len(line.encode())
    return HEAD + "".join(out) if out else ""
