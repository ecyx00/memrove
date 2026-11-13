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
import time

CAP = 6000          # bytes of the whole block, session and subagent alike: a rule works only in context, so the cap guards a runaway count, it does not ration rules (3000 left 7-20 rules of a busy project out)
MAX_WORDS, MAX_CHARS = 40, 200
TURKISH = re.compile("[çğıöşüÇĞİÖŞÜ]")
BARE = re.compile(r"^(?:don'?t|do|no|never|always|not|stop|avoid)\b", re.I)   # an imperative or a negation has no subject
KINDS = ("preference", "decision", "rejected")
GENERIC = ("", "unknown", ".claude", Path.home().name)   # labels of sessions started in the home directory: no project
LOADED_MIN = 3   # content words a sentence needs for the duplicate check between rules
FILLER = frozenset("the a an of to in on is are be and or for with that this it its as at by from do does owner wants want claude when only any all new own must should stay stays will may used use given give personal intelligence".split())
HEAD = "Standing rules (owner's own; binding; newest wins):\n"


def _when(r):
    return str(r.get("valid_from") or r.get("recorded_at") or r.get("updated_at") or "")   # the day the owner said it; updated_at moves with every re-assessment


def _ids(v):
    """supersedes as a list (the store may hold a JSON or Python-literal string)."""
    if isinstance(v, list):
        return v
    return re.findall(r"[\w-]+-[0-9a-f]{8,}", str(v or ""))


def _own(r):
    return r.get("origin_type") == "user_explicit" or r.get("verified_by") == "user_quote"


def _live(r):
    return not (r.get("restate_dup_of") or r.get("unverified") or  # unverified (Core.reflect: the messages do not support it) never becomes a standing rule
 r.get("visibility") == "private" or r.get("verdict") == "reject" or r.get("valid_to") or r.get("status") == "deleted_by_user")


def _restated(r):
    """-> (sentence, scope, project) when the restate step judged the record a standing rule with a usable sentence, else None."""
    a = ((r.get("assessments") or {}).get("restate") or {})
    ans = a.get("answers") or {}
    sent = " ".join(str(ans.get("sentence") or "").split())
    hand = str(r.get("rule_scope") or "").strip()   # set by the owner (memrove rule-scope): wins over the model's judgement
    if a.get("state") != "assessed" and hand and _own(r):   # an owner rule scoped by hand is a rule now, in its own words; it does not wait for the restate step
        a, ans, sent = {"state": "assessed"}, {"standing_rule": "yes"}, " ".join(str(r.get("text") or "").split())
    if a.get("state") != "assessed" or ans.get("standing_rule") != "yes" or not sentence_ok(sent):
        return None
    scope, proj = ans.get("scope"), str(ans.get("project") or "").strip()
    if hand:
        scope, proj = ("global", "") if hand == "global" else ("project", hand)
    if scope not in ("global", "project") or (scope == "project" and not proj):
        return None
    return sent, scope, proj


def sentence_ok(s):
    """A whole English sentence that names its subject: no cut, no Turkish, not an imperative or a bare negation."""
    n = len(s.split())
    return bool(4 <= n <= MAX_WORDS and len(s) <= MAX_CHARS and s[-1] in ".!)" and "…" not in s and not TURKISH.search(s) and not BARE.match(s))


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


NEG = "\0neg"
NEGATORS = frozenset("not no never without cannot cant dont doesnt isnt arent wont shouldnt mustnt nor avoid avoids stop stops".split())


def _stem(w):
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _words(s):
    """-> content-word stems, plus NEG when the text negates (a reversal must not match the line it reverses)."""
    toks = [t.replace("'", "") for t in re.findall(r"[a-z0-9']+", s.lower().replace("\u2019", "'"))]
    out = {_stem(t) for t in toks if t not in FILLER and t not in NEGATORS and len(t) > 1}
    if any(t in NEGATORS for t in toks):
        out.add(NEG)
    return out


def _flat(s):
    """lowercase, every run of whitespace/punctuation collapsed to one space."""
    return " ".join(re.sub(r"[\W_]+", " ", str(s).lower()).split())


def loaded_text(cwd):
    """-> the flattened text of the CLAUDE.md files a session in cwd loads: ~/.claude/CLAUDE.md, and CLAUDE.md from cwd up to home."""
    home, files = Path.home(), [Path.home() / ".claude" / "CLAUDE.md"]
    if cwd:
        d = Path(cwd).expanduser().resolve()
        while True:
            files.append(d / "CLAUDE.md")
            if d == home or d.parent == d or home not in d.parents:
                break
            d = d.parent
    out = []
    for f in dict.fromkeys(files):
        try:
            out.append(_flat(f.read_text()))
        except OSError:
            pass
    return " " + " ".join(out) + " "


def rules(state, project=None, global_only=False, cwd=None, dropped=None):
    """-> [(when, project_tag, sentence, id)]: project and global rules take turns, most-repeated then newest first. Duplicates and reversals are marked by the restate step (restate_dup_of)."""
    path = Path(state) / "memory.sqlite3"
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    try:
        rows = [json.loads(p) for (p,) in db.execute(
            "SELECT payload FROM records WHERE json_extract(payload,'$.kind') IN (?,?,?)", KINDS)]
    finally:
        db.close()
    for r in rows:   # a correction is never a duplicate of the rule it replaces, even when the merge step marked it so
        if r.get("restate_dup_of") in _ids(r.get("supersedes")):
            r["restate_dup_of"] = None
    dup = {r["id"]: r["restate_dup_of"] for r in rows if r.get("restate_dup_of")}
    said = {}   # how often the owner said a rule: the records merged into it, followed to the root
    for i in dup:
        root, hops = i, 0
        while root in dup and hops < 20:
            root, hops = dup[root], hops + 1
        said[root] = said.get(root, 0) + 1
    by_id = {r["id"]: r for r in rows}
    for r in rows:   # an owner correction (--supersedes) of a rule keeps the old rule's scope until the restate step judges it
        if not r.get("rule_scope") and ((r.get("assessments") or {}).get("restate") or {}).get("state") != "assessed" and _own(r):
            old = next((x for x in (by_id.get(s) for s in _ids(r.get("supersedes"))) if x and _restated(x)), None)
            if old:
                _, scope, proj = _restated(old)
                r["rule_scope"] = "global" if scope == "global" else proj
    rows = [(r, _restated(r)) for r in rows if _own(r) and _live(r)]
    rows = [(r, x) for r, x in rows if x]
    gone = {s for r, _ in rows for s in _ids(r.get("supersedes"))}   # a correction hides the old rule only once it is itself a restated rule: a pending one would leave a gap
    # global: scope global; a project rule only for a session of that project (never in the subagent form)
    mine = [(r, x) for r, x in rows if r["id"] not in gone and (x[1] == "global" or (not global_only and project and _norm(x[2]) == _norm(project)))]
    # project rules and global rules take turns, each list ordered by how often the owner said the rule, then newest first:
    # a project rule is delivered nowhere else, a global one reaches every session
    mine.sort(key=lambda t: (-said.get(t[0]["id"], 0), _neg(_when(t[0]))))
    lists = [[t for t in mine if t[1][1] != "global"], [t for t in mine if t[1][1] == "global"]]
    order = [t for pair in zip_longest(*lists) for t in pair if t]
    if cwd:   # a rule whose normalized text sits whole in a loaded CLAUDE.md needs no second push; its id goes to `dropped`
        text = loaded_text(cwd)
        gone_ids = [t for t in order if f" {_flat(t[1][0])} " in text]
        order = [t for t in order if t not in gone_ids]
        if dropped is not None:
            dropped.extend(t[0]["id"] for t in gone_ids)
    kept, seen = [], []   # the merge step misses some duplicates (two records restated to one sentence): one line per rule
    for t in order:
        w = _words(t[1][0])
        i = next((i for i, s in enumerate(seen) if _within(w, s) or _within(s, w)), None)
        if i is None:
            kept.append(t)
            seen.append(w)
        elif len(w) > len(seen[i]):   # the longer line says all the shorter one says: it takes the shorter one's place
            kept[i], seen[i] = t, w
    return [(_when(r), "" if x[1] == "global" else x[2], x[0], r["id"]) for r, x in kept]


def _within(a, b):
    """a says nothing b does not: every content word of a is in b, both negate or neither does. A one-word swap (Sonnet for Opus, 4B for 9B) is a different rule, so overlap ratios are not used."""
    return (NEG in a) == (NEG in b) and len(a - {NEG}) >= LOADED_MIN and a <= b


def _neg(when):
    """Sort key that puts the newer date first inside an ascending sort."""
    return [-ord(c) for c in when]


def block(state, project=None, global_only=False, cap=None, cwd=None, ids=None):
    """The block text ('' when off, empty or on any error). Whole sentences only; when the cap cuts, the last line counts the rest and names the command that lists them."""
    if os.environ.get("MEMROVE_RULES") == "0":
        return ""
    cap = cap or CAP
    try:
        dropped = []
        items = rules(state, project, global_only, cwd, dropped)
    except Exception:
        return ""
    if dropped:   # audit trail: every dropped rule has an exact normalized match in CLAUDE.md
        try:
            with open(Path(state) / "rules-dropped.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), "ids": dropped}) + "\n")
        except OSError:
            pass
    lines = ["- " + rule + "\n" for _, _, rule, _ in items]
    notice = lambda n: f"- ({n} more rule{'s' * (n > 1)} left out by the size cap; list them: memrove rules --left-out" + (f" --project '{project}'" if project else "") + ")\n"
    size = lambda xs: sum(len(x.encode()) for x in xs)
    k = len(lines)   # the longest prefix that fits, with the notice when something is left out
    while k and len(HEAD.encode()) + size(lines[:k]) + (len(notice(len(lines) - k).encode()) if k < len(lines) else 0) > cap:
        k -= 1
    if not k:
        return ""
    if ids is not None:  # the caller's exposure log: which rule ids went out
        ids.extend(i for *_, i in items[:k])
    return HEAD + "".join(lines[:k]) + (notice(len(lines) - k) if k < len(lines) else "")


def left_out(state, project=None, cwd=None):
    """-> [(id, project_tag, sentence)] the SessionStart block leaves out for this project (what `memrove rules --left-out` lists)."""
    try:   # the cap the last SessionStart used for this project (the hook gives rules the room other slots leave)
        cap = json.loads((Path(state) / "rules-cap.json").read_text(encoding="utf-8")).get(project or "")
    except (OSError, ValueError):
        cap = None
    shown = len([ln for ln in block(state, project, cap=cap, cwd=cwd).splitlines()[1:] if not ln.startswith("- (")])
    return [(i, tag, rule) for _, tag, rule, i in rules(state, project, cwd=cwd)[shown:]]


def short_block(state, n=8, ids=None):
    """The subagent form of the standing rules: the n most-repeated/newest global rules only (project rules reach an agent through recall). '' when off or on any error."""
    if os.environ.get("MEMROVE_RULES") == "0":
        return ""
    try:
        items = rules(state, None, global_only=True)[:n]
    except Exception:
        return ""
    if ids is not None:
        ids.extend(i for *_, i in items)
    return HEAD + "".join("- " + rule + "\n" for _, _, rule, _ in items) if items else ""
