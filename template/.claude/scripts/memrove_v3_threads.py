"""Context units for the second brain: Threads.md topics and Last-Session.md cards, written by the J2 job in the v2 format.

Only blocks carrying a memrove marker comment (`<!-- memrove:thread ... -->`, `<!-- memrove:card ... -->`) are ever rewritten; every other
byte of the user's files stays as it was. A thread is a readable document (status, context, short dated "done" list, decisions with
reasons, open items, evidence links) merged section by section, not a log. Atomic records stay as evidence and are linked from the thread.
"""
import hashlib
import json
import os
import re
from pathlib import Path
from datetime import timedelta
from urllib.parse import quote, unquote

THREAD_RE = re.compile(r'<!-- memrove:thread id=(t-[0-9a-f]+) project=([^\s>]*) -->')
CARD_RE = re.compile(r'<!-- memrove:card sid=([0-9a-zA-Z_-]+) project=([^\s>]*) -->')
# Active/Open headings; Aktif/Açık are the Turkish forms (Turkish vault files).
ACTIVE_RE = re.compile(r'(?im)^## (?:(?:Active|Open)(?: Threads)?|Aktif[^\n]*|A[çc][ıi]k(?:[ \t][^\n]*)?)[ \t]*$')  # same as companion.excerpt
# Closed: Kapan/Kapalı are the Turkish forms.
CLOSED_RE = re.compile(r'(?im)^## (?:Closed|Kapan|Kapalı)[^\n]*$')
LABEL_RE = re.compile(r'^\*\*([^*:\n]+):\*\*[ \t]*(.*)$')
SCALAR_T, LIST_T = ("Status", "Context"), ("Done", "Decisions", "Open items", "Completed", "Evidence")
SCALAR_C = ("Result", "Rationale", "Next step", "Source")
CAPS = {"Done": 8, "Decisions": 10, "Open items": 8, "Completed": 8, "Evidence": 12}
DORMANT_DAYS = 14  # a thread without a linked session for this long is hidden from injection (still searchable)
WIKI_RE = re.compile(r"\[\[[^\]]+\]\]")
STATUSES = ("active", "waiting", "closed")
INDEX = "threads-index.json"  # state dir: record id -> thread, read by recall (the hook has no vault path)


def enabled():
    """Via negativa (topic heads): the thread / session-card / Last-Session writers and their surfaces are off. MEMROVE_THREADS=1 turns them back on."""
    return os.environ.get("MEMROVE_THREADS") == "1"


def _proj(m):
    return unquote(m[2])


def _clean(s, n=240):
    s = " ".join(str(s).split())
    if len(s) <= n:
        return s
    cut = s[:n - 1]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > n * .6 else cut).rstrip(" ,;:") + "…"  # at a word boundary when one is near


def _tail(s, n):
    """Last <= n chars, starting at a word boundary (never mid-word)."""
    if len(s) <= n:
        return s
    t = s[-n:]
    cut = t.find(" ")
    return "…" + (t[cut + 1:] if 0 <= cut < len(t) - 1 else t)


def _words(s):
    return set(re.findall(r"\w{4,}", str(s).casefold()))


def _similar(a, b, ratio=.6):
    x, y = _words(a), _words(b)
    return bool(x and y) and len(x & y) / min(len(x), len(y)) >= ratio


def _day_of(d):
    from datetime import date
    try:
        return date.fromisoformat(d["Status"].split("·")[-1].strip())
    except ValueError:
        return None


def _read(path):
    try:
        t = Path(path).read_bytes().decode("utf-8")
    except OSError:
        return "", False
    return t.replace("\r\n", "\n"), "\r\n" in t


def _write(path, text, crlf):
    from memrove_v3_sync import atomic
    atomic(path, text.replace("\n", "\r\n") if crlf else text)


def parse(block, scalars, lists):
    """Block text -> dict(title, mark, <scalar>: str, <list>: [str], extra: [lines]). Unknown lines are kept in extra."""
    lines = block.rstrip("\n").split("\n")
    d = {"title": re.sub(r"^#+\s*", "", lines[0]), "mark": "", "extra": [], **{k: "" for k in scalars}, **{k: [] for k in lists}}
    cur = None
    for ln in lines[1:]:
        m = LABEL_RE.match(ln)
        if ln.startswith("<!-- memrove:") and not d["mark"]:
            d["mark"] = ln
        elif m and m[1] in scalars + lists:
            cur = m[1]
            if cur in scalars:
                d[cur] = m[2].strip()
            elif m[2].strip():
                d[cur].append(m[2].strip())
        elif ln.startswith("- ") and cur in lists:
            d[cur].append(ln[2:].strip())
        elif ln.strip():
            d["extra"].append(ln)
    return d


def render_thread(d):
    out = [f"### {d['title']}", d["mark"], f"**Status:** {d['Status']}", f"**Context:** {d['Context']}"]
    for k in LIST_T:
        out += [f"**{k}:**"] + [f"- {x}" for x in d[k]] if d[k] else []
    return "\n".join(out + d["extra"]) + "\n\n"


def render_card(d):
    out = [f"## {d['title']}", d["mark"]] + [f"**{k}:** {d[k]}" for k in SCALAR_C if d[k]]
    return "\n".join(out + d["extra"]) + "\n\n"


def _spans(text, lo, hi, hashes):
    """Spans (start, end) of blocks opened by `hashes` headings inside [lo, hi); a block ends at the next heading of the same or higher level."""
    opens = [m.start() for m in re.compile(rf"(?m)^{hashes} [^\n]*\n?").finditer(text, lo, hi)]
    stop = re.compile(rf"(?m)^#{{1,{len(hashes)}}} ")
    out = []
    for s in opens:
        m = stop.search(text, s + 1, hi)
        out.append((s, m.start() if m else hi))
    return out


def _regions(text):
    """(active_lo, active_hi, closed_lo, closed_hi); -1 when a section does not exist."""
    a, c = ACTIVE_RE.search(text), CLOSED_RE.search(text)

    def end(m):
        n = re.compile(r"(?m)^## ").search(text, m.end())
        return n.start() if n else len(text)
    return (a.end() + 1 if a else -1, end(a) if a else -1, c.end() + 1 if c else -1, end(c) if c else -1)


def threads(text):
    """All thread blocks -> [(start, end, region, parsed|None)]; parsed is None for blocks the user wrote."""
    al, ah, cl, ch = _regions(text)
    out = []
    for region, lo, hi in (("active", al, ah), ("closed", cl, ch)):
        if lo < 0:
            continue
        for s, e in _spans(text, lo, hi, "###"):
            d = parse(text[s:e], SCALAR_T, LIST_T)
            out.append((s, e, region, d if THREAD_RE.search(d["mark"]) else None, text[s:e]))
    return out


def _tid(project, title):
    return "t-" + hashlib.sha256((project + title.casefold()).encode()).hexdigest()[:8]


def _status_of(d):
    return (d["Status"].split("·")[0].strip() or "active").lower()


def active_for_prompt(path, project, limit=12):
    """Title + one-line state of the project's open memrove threads, for the J2 prompt. -> (text, {id: title})."""
    text, _ = _read(path)
    rows, ids = [], {}
    for *_, region, d, _raw in threads(text):
        if d is None or region != "active" or _proj(THREAD_RE.search(d["mark"])) != project or _status_of(d) == "closed":
            continue
        tid = THREAD_RE.search(d["mark"])[1]
        did = d["Done"][0] if d["Done"] else ""
        opn = "; ".join(d["Open items"][:5])
        rows.append(f"- {tid}: {d['title']} [{_status_of(d)}] {_clean(d['Context'], 140)}" + (f" | last: {_clean(did, 100)}" if did else "")
                    + (f" | open: {_clean(opn, 200)}" if opn else ""))
        ids[tid] = d["title"]
        if len(rows) >= limit:
            break
    return ("ACTIVE THREADS:\n" + "\n".join(rows)) if rows else "ACTIVE THREADS:\n(none)", ids


def _merge(d, upd, day, links):
    """Merge one thread update into the parsed thread (readable document, not a log)."""
    ctx = _clean(upd.get("context", ""), 300)
    if ctx and not d["Context"]:  # ponytail: context is only filled once; J2 sees 140 chars of it and could not merge a rewrite
        d["Context"] = ctx
    status = upd.get("status") if upd.get("status") in STATUSES else "active"
    d["Status"] = f"{status} · {day}"
    seen = {x.split(" · ", 1)[-1].casefold() for x in d["Done"]}
    new = [f"{day} · {_clean(x, 200)}" for x in upd.get("did", []) if isinstance(x, str) and x.strip() and _clean(x, 200).casefold() not in seen]
    d["Done"] = (new + d["Done"])[:CAPS["Done"]]
    for x in upd.get("decisions", []):
        what, why = (x.get("what"), x.get("why")) if isinstance(x, dict) else (x, "")
        if isinstance(what, str) and what.strip():
            line = _clean(what, 200) + (f" — {_clean(why, 200)}" if isinstance(why, str) and why.strip() else "")
            if not any(line.casefold() == y.casefold() for y in d["Decisions"]):
                d["Decisions"] = (d["Decisions"] + [line])[-CAPS["Decisions"]:]
    if isinstance(upd.get("open"), list):  # J2 saw the open items and returns the full current list
        now = [_clean(x, 200) for x in upd["open"] if isinstance(x, str) and x.strip()][:CAPS["Open items"]]
        gone = [x for x in d["Open items"] if not any(_similar(x, y) or x.casefold() == y.casefold() for y in now)]
        d["Completed"] = ([f"{day} · {x}" for x in gone] + d["Completed"])[:CAPS["Completed"]]  # finished items leave a trace
        d["Open items"] = now
    for ln in links:
        if ln not in d["Evidence"]:
            d["Evidence"] = (d["Evidence"] + [ln])[-CAPS["Evidence"]:]
    return status


def _index(state, project, d, extra_ids, vault_file):
    p = Path(state) / INDEX
    try:
        idx = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        idx = {}
    idx.setdefault("threads", {})
    idx.setdefault("records", {})
    tid = THREAD_RE.search(d["mark"])[1]
    idx["threads"][tid] = {"title": d["title"], "project": project, "status": _status_of(d), "day": d["Status"].split("·")[-1].strip(),
                           "context": _clean(d["Context"], 160), "file": str(vault_file)}
    for rid in extra_ids:
        idx["records"][rid] = tid
    p.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")


def apply_threads(path, state, project, day, updates, record_links):
    """updates: [{attach, title, context, did, decisions, open, status}] from J2. record_links: [(record_id, [[wikilink]], text)].
    Attaches to an open memrove thread of this project by id (or exact title), else opens one. -> [titles of the threads touched]."""
    text, crlf = _read(path)
    mine = {THREAD_RE.search(d["mark"])[1]: (s, e, region, d) for s, e, region, d, _ in threads(text)
            if d is not None and _proj(THREAD_RE.search(d["mark"])) == project}
    titles = {d["title"].casefold(): tid for tid, (_s, _e, region, d) in mine.items() if region == "active"}
    touched = []
    for upd in updates[:5]:
        if not isinstance(upd, dict):
            continue
        tid = upd.get("attach") if upd.get("attach") in mine and mine[upd.get("attach")][2] == "active" else titles.get(_clean(upd.get("title", ""), 80).casefold())
        links = []
        for rid, wiki, rtext in record_links:  # evidence goes to the thread whose update shares the most words with the record
            words = lambda s: set(re.findall(r"\w{4,}", str(s).casefold()))
            best = max(updates[:5], key=lambda u: len(words(rtext) & words(json.dumps(u.get("decisions", []), ensure_ascii=False) + " ".join(map(str, u.get("did", []))))) if isinstance(u, dict) else -1)
            if best is upd:
                links += wiki
        links = [x for x in links if x]
        if tid:
            _s, _e, _r, d = mine[tid]
        else:
            title = _clean(upd.get("title", ""), 80)
            if not title:
                continue
            tid = _tid(project, title)
            if tid in mine:  # same title came back after closing: reopen that thread
                _s, _e, _r, d = mine[tid]
            else:
                d = {"title": title, "mark": f"<!-- memrove:thread id={tid} project={quote(project, safe='')} -->", "extra": [], **{k: "" for k in SCALAR_T}, **{k: [] for k in LIST_T}}
        _merge(d, upd, day, links)
        text = _put_thread(text, tid, d, project)
        mine = {THREAD_RE.search(x["mark"])[1]: (s, e, region, x) for s, e, region, x, _ in threads(text) if x is not None and _proj(THREAD_RE.search(x["mark"])) == project}
        titles = {x["title"].casefold(): t for t, (_s, _e, region, x) in mine.items() if region == "active"}
        _index(state, project, d, [rid for rid, wiki, _t in record_links if any(w in d["Evidence"] for w in wiki)], path)
        touched.append(d["title"])
    if touched:
        _write(path, text, crlf)
    return touched


def _insert(text, pos, block):
    """Block right after a heading line (and the blank line that follows it, if any)."""
    if text[pos:pos + 1] == "\n":
        return text[:pos + 1] + block + text[pos + 1:]
    return text[:pos] + "\n" + block + text[pos:]


def _put_thread(text, tid, d, project):
    """Remove the thread's old block (if any) and put the rendered one first in Active (updated threads float up) or first in Closed."""
    for s, e, _region, x, _ in threads(text):
        if x is not None and THREAD_RE.search(x["mark"])[1] == tid:
            text = text[:s] + text[e:]
            break
    closed = _status_of(d) == "closed"
    al, ah, cl, ch = _regions(text)
    block = render_thread(d)
    if closed and cl < 0:
        return text.rstrip("\n") + "\n\n## Closed Threads\n\n" + block
    if closed:
        return _insert(text, cl, block)
    if al < 0:
        pos = CLOSED_RE.search(text)
        head = "## Active Threads\n\n" + block
        return text[:pos.start()] + head + text[pos.start():] if pos else text.rstrip("\n") + ("\n\n" if text.strip() else "") + head
    return _insert(text, al, block)


def apply_card(path, sid, project, label, ts, card, sources):
    """One Last-Session card per session, newest first: `## YYYY-MM-DD HH:MM · <label> · <sid8>`. A later window of the same session
    updates its own card (outcome grows to the last 600 chars, next step is replaced); other cards are never touched."""
    sid8 = sid[:8]
    text, crlf = _read(path)
    c = {k: _clean(card.get(a, ""), 500) for k, a in zip(SCALAR_C[:3], ("outcome", "reasoning", "next"))}
    old = None
    for m in re.finditer(r"(?m)^## [^\n]*\n(<!-- memrove:card sid=%s project=[^\n]*-->)\n" % re.escape(sid8), text):
        n = re.compile(r"(?m)^## ").search(text, m.end())
        old = (m.start(), n.start() if n else len(text))
    if old:
        d = parse(text[old[0]:old[1]], SCALAR_C, ())
        c["Result"] = _tail(d["Result"] if c["Result"] in d["Result"] else (d["Result"] + " " + c["Result"]).strip(), 600)
        c["Rationale"] = c["Rationale"] or d["Rationale"]
        c["Next step"] = c["Next step"] or d["Next step"]
        srcs = list(dict.fromkeys(WIKI_RE.findall(d["Source"]) + sources))  # whole wikilinks; titles may hold spaces
        d.update(c, Source=" ".join(srcs))
        text = text[:old[0]] + render_card(d) + text[old[1]:]
    else:
        if junk_card(label, c, sources):
            return False
        d = {"title": f"{ts} · {_clean(label, 60) or project} · {sid8}", "mark": f"<!-- memrove:card sid={sid8} project={quote(project, safe='')} -->", "extra": [],
             **c, "Source": " ".join(sources)}
        first = re.compile(r"(?m)^## ").search(text)
        block = render_card(d)
        text = (text[:first.start()] + block + text[first.start():]) if first else (text.rstrip("\n") + ("\n\n" if text.strip() else "") + block.rstrip("\n") + "\n")
    _write(path, text, crlf)
    return True


JUNK_LABEL = re.compile(r"(?i)\b(probe|confirmation|confirm|onay|ping|smoke|selftest)\b")
EMPTY = {"", "none", "null", "n/a", "yok", "-", "none."}


def junk_card(label, c, sources):
    """Probe, confirmation and chit-chat sessions: no outcome, a probe label, or neither a next step nor a touched thread."""
    out = c["Result"].strip().casefold()
    return out in EMPTY or bool(JUNK_LABEL.search(label or "")) or (
        not c["Next step"].strip() and not any(x.startswith("[[Threads#") for x in sources))


def _card_stamp(title):
    m = re.match(r"(\d{4}-\d{2}-\d{2})(?: (\d{2}:\d{2}))?", title)
    return (m[1] + " " + (m[2] or "00:00")) if m else ""


def _companion_text(vault, name):
    try:
        from memrove_v3_companion import directory
        d = directory(vault)
        return _read(d / name)[0] if d else ""
    except Exception:  # ponytail: a companion problem must never cost the session its context
        return ""


def _dormant(d, today=None):
    from datetime import date
    day = _day_of(d)
    return day is not None and (today or date.today()) - day > timedelta(days=DORMANT_DAYS)


def inject_items(vault, project, limit=12, today=None):
    """Open threads for SessionStart: [(id, full line, short line)], the id is the first token of both so `memrove show <id>` always opens it.
    The project's memrove threads (dormant and closed ones stay out), plus user-written threads that name the project."""
    out = []
    for *_, region, p, raw in threads(_companion_text(vault, "Threads.md")):
        if region != "active":
            continue
        if p is not None:
            m = THREAD_RE.search(p["mark"])
            if _proj(m) not in project or _status_of(p) == "closed" or _dormant(p, today):
                continue
            tid = f"th:{m[1]}"
            line = f"{tid} [{_status_of(p)}] {p['title']}: {_clean(p['Context'], 140)}"
            for k, n in (("Done", 1), ("Decisions", 1), ("Open items", 3)):
                if p[k]:
                    line += f"\n  {k}: " + " | ".join(_clean(x, 110) for x in p[k][:n])
            if p["Completed"]:
                line += f"\n  Completed: {len(p['Completed'])} (`memrove show {tid}`)"
            out.append((tid, line, f"{tid} [{_status_of(p)}] {_clean(p['title'], 80)}"))
        elif any(x.casefold() in raw.casefold() for x in project if len(x) > 2):
            tid = "th:u:" + hashlib.sha256(raw.encode()).hexdigest()[:8]
            title = raw.split("\n", 1)[0].lstrip("# ").strip()
            out.append((tid, f"{tid} " + _clean(raw.replace("\n", " "), 420), f"{tid} {_clean(title, 80)}"))
    return out[:limit]


def _cards(vault):
    """Last-Session.md cards -> [(stamp, text)], file order."""
    text = _companion_text(vault, "Last-Session.md")
    starts = [m.start() for m in re.finditer(r"(?m)^## ", text)] + [len(text)]
    return [(_card_stamp(text[s:e].split("\n", 1)[0][3:]), text[s:e]) for s, e in zip(starts, starts[1:])]


def _card_id(c):
    m = CARD_RE.search(c)
    return m[1] if m else hashlib.sha256(c.encode()).hexdigest()[:8]


def last_card(vault, project):
    """Newest card for the project (by its timestamp, not file position); a vault with no memrove card falls back to the file's first card
    (user-written, newest first). The line always carries the card's open step in full."""
    cards = _cards(vault)
    mine = [(st, c) for st, c in cards if (m := CARD_RE.search(c)) and _proj(m) in project]
    if mine:
        pick = max(mine, key=lambda x: x[0])[1]  # max keeps the first of equal stamps = file order
    elif cards and not any(CARD_RE.search(c) for _, c in cards):
        pick = cards[0][1]
    else:
        return []
    d = parse(pick, SCALAR_C, ())
    title = _clean(d["title"], 90)
    if not CARD_RE.search(pick):  # user-written card: no fields, show it clipped at a word boundary
        body = " ".join(l for l in pick.strip().split("\n")[1:] if not l.startswith("<!-- memrove:")).replace("**", "")
        return [(f"ls:{_card_id(pick)}", f"ls:{_card_id(pick)} {title}: " + _clean(body, 500), f"ls:{_card_id(pick)} {title}")]
    lid = f"ls:{_card_id(pick)}"
    line = f"{lid} {title}"
    if d["Result"]:
        line += "\n  Result: " + _clean(d["Result"], 240)
    if d["Next step"]:
        line += "\n  Next step: " + _clean(d["Next step"], 400)
    return [(lid, line, f"{lid} {title}")]


def thread_doc(vault, tid):
    """`memrove show th:<id>`: the thread as a readable dict from Threads.md (th:u:<hash> = a user-written block). None when absent."""
    tid = tid[3:] if tid.startswith("th:") else tid
    for *_, region, p, raw in threads(_companion_text(vault, "Threads.md")):
        if p is not None and THREAD_RE.search(p["mark"])[1] == tid:
            return {"id": "th:" + tid, "kind": "thread", "project": _proj(THREAD_RE.search(p["mark"])), "region": region,
                    **{k: v for k, v in {"title": p["title"], "Status": p["Status"], "Context": p["Context"], **{k: p[k] for k in LIST_T}}.items() if v}}
        if p is None and tid.startswith("u:") and hashlib.sha256(raw.encode()).hexdigest()[:8] == tid[2:]:
            return {"id": "th:" + tid, "kind": "thread", "region": region, "text": raw.strip()}
    return None


def card_doc(vault, cid):
    """`memrove show ls:<sid8>`: one Last-Session card, in full."""
    cid = cid[3:] if cid.startswith("ls:") else cid
    for _st, c in _cards(vault):
        if _card_id(c) == cid:
            d = parse(c, SCALAR_C, ())
            return {"id": "ls:" + cid, "kind": "last_session", "title": d["title"], **{k: d[k] for k in SCALAR_C if d[k]}} if CARD_RE.search(c) \
                else {"id": "ls:" + cid, "kind": "last_session", "text": c.strip()}
    return None


MAX_CARDS, CLOSED_DAYS = 20, 60


def _drop_thread(text, tid):
    for s, e, _region, x, _ in threads(text):
        if x is not None and THREAD_RE.search(x["mark"])[1] == tid:
            return text[:s] + text[e:]
    return text


def _index_edit(state, fn):
    p = Path(state) / INDEX
    try:
        idx = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    fn(idx)
    p.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")


def _union(a, b, cap):
    out = list(a)
    out += [x for x in b if not any(x.casefold() == y.casefold() for y in out)]
    return out[:cap]


def lifecycle(text, state, today=None):
    """Thread hygiene on Threads.md text. -> (new text, number of changes).
    1. Merge: the same work under two projects (>=2 shared title words, 60% of the shorter title) becomes one thread, kept under the project
       with the most memrove blocks (the established one; a tie goes to the newer thread). Its old id resolves to the survivor in the index.
    2. Close: dormant (no linked session for DORMANT_DAYS) and no open items -> Closed Threads, last day kept so the archive clock is right."""
    from datetime import date
    today, changed = today or date.today(), 0
    live = lambda: [(THREAD_RE.search(d["mark"])[1], _proj(THREAD_RE.search(d["mark"])), d) for *_, region, d, _r in threads(text)
                    if d is not None and region == "active" and _status_of(d) != "closed"]
    weight = {}
    for *_, d, _r in threads(text):
        if d is not None:
            q = _proj(THREAD_RE.search(d["mark"]))
            weight[q] = weight.get(q, 0) + 1
    def pair():
        rows = live()
        for tid, proj, d in rows:
            for tid2, proj2, d2 in rows:
                if proj2 != proj and tid2 != tid and (d["title"].casefold() == d2["title"].casefold()
                                                      or (len(_words(d["title"]) & _words(d2["title"])) >= 2 and _similar(d["title"], d2["title"]))):
                    first = (weight[proj], str(_day_of(d))) >= (weight[proj2], str(_day_of(d2)))
                    return (tid, d, tid2, d2) if first else (tid2, d2, tid, d)

    while (m := pair()) and changed < 20:
        ktid, kd, dtid, dd = m
        for k in ("Done", "Decisions", "Open items", "Completed", "Evidence"):
            kd[k] = _union(kd[k], dd[k], CAPS[k])
        kd["Context"] = kd["Context"] or dd["Context"]
        if (_day_of(dd) or date.min) > (_day_of(kd) or date.min):
            kd["Status"] = dd["Status"]
        text = _put_thread(_drop_thread(text, dtid), ktid, kd, _proj(THREAD_RE.search(kd["mark"])))
        _index_edit(state, lambda idx, a=dtid, b=ktid: (idx["threads"].pop(a, None), idx["records"].update({r: b for r, t in idx["records"].items() if t == a}),
                                                        idx.setdefault("alias", {}).update({a: b})))
        changed += 1
    gone = set()
    for tid, proj, d in live():
        if tid not in gone and _dormant(d, today) and not d["Open items"]:
            d["Status"] = f"closed · {d['Status'].split('·')[-1].strip()}"
            text = _put_thread(text, tid, d, proj)
            _index_edit(state, lambda idx, t=tid: idx["threads"].get(t, {}).update({"status": "closed"}))
            changed += 1
    return text, changed


def _archive(vault, target, name, pieces, month):
    """Append moved blocks verbatim to <companion>/Archive/<Stem>-YYYY-MM.md (memrove_v3_compact's convention), before they leave the live file."""
    from memrove_v3_compact import ARCHIVE_DIRECTORY
    arch = target / ARCHIVE_DIRECTORY / f"{Path(name).stem}-{month}.md"
    old = arch.read_text(encoding="utf-8") if arch.exists() else (
        f'---\n{{"kind": "note", "visibility": "private", "title": "{name} archive {month}"}}\n---\n# {name} archive: {month}\n\n')
    from memrove_v3_sync import atomic
    atomic(arch, old.rstrip("\n") + "\n\n" + "".join(pieces).replace("\r\n", "\n"))


def prune(vault, state, today=None, max_cards=MAX_CARDS, closed_days=CLOSED_DAYS):
    """memrove cards beyond `max_cards` per project and memrove threads closed longer than `closed_days` ago move to the compaction archive.
    User-written cards and threads are never moved. Takes the compaction lock; busy -> try next run. -> number of blocks moved"""
    from datetime import date, timedelta
    from memrove_v3_compact import _compact_lock
    from memrove_v3_companion import directory
    target = directory(vault)
    if target is None:
        return 0
    today = today or date.today()
    month, moved = today.strftime("%Y-%m"), 0
    try:
        with _compact_lock(vault, state):
            # Last-Session: file order is newest first
            text, crlf = _read(target / "Last-Session.md")
            seen, cut = {}, []
            starts = [m.start() for m in re.finditer(r"(?m)^## ", text)] + [len(text)]
            for s, e in zip(starts, starts[1:]):
                m = CARD_RE.search(text[s:e])
                if m:
                    seen[_proj(m)] = seen.get(_proj(m), 0) + 1
                    if seen[_proj(m)] > max_cards:
                        cut.append((s, e))
            if cut:
                _archive(vault, target, "Last-Session.md", [text[s:e] for s, e in cut], month)
                for s, e in reversed(cut):
                    text = text[:s] + text[e:]
                _write(target / "Last-Session.md", text, crlf)
                moved += len(cut)
            text, crlf = _read(target / "Threads.md")
            text2, n = lifecycle(text, state, today)
            if n:
                _write(target / "Threads.md", text2, crlf)
                text = text2
            cut = []
            for s, e, region, d, raw in threads(text):
                day = d["Status"].split("·")[-1].strip() if d else ""
                try:
                    old = region == "closed" and d is not None and date.fromisoformat(day) < today - timedelta(days=closed_days)
                except ValueError:
                    old = False
                if old:
                    cut.append((s, e, raw))
            if cut:
                _archive(vault, target, "Threads.md", [c[2] for c in cut], month)
                for s, e, _ in reversed(cut):
                    text = text[:s] + text[e:]
                _write(target / "Threads.md", text, crlf)
                moved += len(cut)
    except TimeoutError:
        pass
    return moved


def drop_owned(vault, state, project=None):
    """Rebuild step: remove memrove-marked threads and cards (of `project`, or all when None) from Threads.md / Last-Session.md and the recall index.
    User-written blocks and everything else stay byte-identical; Archive is untouched. -> number of blocks dropped"""
    from memrove_v3_companion import directory
    target = directory(vault)
    if target is None:
        return 0
    n = 0
    mine = lambda m: m and (project is None or _proj(m) == project)
    text, crlf = _read(target / "Threads.md")
    cut = [(s, e) for s, e, _r, d, _raw in threads(text) if d is not None and mine(THREAD_RE.search(d["mark"]))]
    for s, e in reversed(cut):
        text = text[:s] + text[e:]
    if cut:
        _write(target / "Threads.md", text, crlf)
    n += len(cut)
    text, crlf = _read(target / "Last-Session.md")
    starts = [m.start() for m in re.finditer(r"(?m)^## ", text)] + [len(text)]
    cut = [(s, e) for s, e in zip(starts, starts[1:]) if mine(CARD_RE.search(text[s:e]))]
    for s, e in reversed(cut):
        text = text[:s] + text[e:]
    if cut:
        _write(target / "Last-Session.md", text, crlf)
    n += len(cut)
    p = Path(state) / INDEX
    try:
        idx = json.loads(p.read_text(encoding="utf-8"))
        gone = {t for t, v in idx.get("threads", {}).items() if project is None or v.get("project") == project}
        idx["threads"] = {t: v for t, v in idx.get("threads", {}).items() if t not in gone}
        idx["records"] = {r: t for r, t in idx.get("records", {}).items() if t not in gone}
        p.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError):
        pass
    return n
