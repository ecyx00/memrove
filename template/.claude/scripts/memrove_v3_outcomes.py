"""Did an injected card or block get used? Reads the decision log (recall and session-start injections with the
per-id outcome the Stop worker wrote) and reports counts and rates; backfills outcomes for older injections from the
transcripts into a temp copy of the log. Stdlib only; read-only on transcripts and on the store."""
import json
import os
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import memrove_v3_decide_log as dl

STATUSES = ("opened", "cited", "ignored")
USES = ("memrove.recall", "memrove.inject")


def _prefix_kind(bid):
    for p, k in (("th:", "thread"), ("t-", "thread"), ("ls:", "last"), ("doc:", "doc"), ("ent:", "entity"), ("rules:", "rules"), ("rule:", "rules"), ("fs-", "file")):
        if bid.startswith(p):
            return k
    return ""


def make_lookup(state_dir, connect):
    """-> info(id) -> {kind, project, title, needles}; connect() opens the store read-only. Memoised; never raises."""
    try:
        idx = json.loads((Path(state_dir) / "threads-index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        idx = {}
    cache = {}

    def info(bid):
        if bid in cache:
            return cache[bid]
        out = {"kind": _prefix_kind(bid), "project": "", "title": "", "needles": []}
        try:
            t = (idx.get("threads") or {}).get(bid.split(":", 1)[1] if bid.startswith("th:") else bid)
            if t:
                out.update(kind="thread", project=t.get("project") or "", title=t.get("title") or "", needles=[t.get("title") or ""])
            else:
                with connect() as db:
                    row = db.execute("SELECT payload FROM records WHERE id=?", (bid,)).fetchone()
                if row:
                    r = json.loads(row[0])
                    title = r.get("title") or r.get("text") or r.get("approach") or ""
                    out.update(kind=r.get("kind") or "", project=r.get("project") or "", title=" ".join(str(title).split())[:80], needles=[r.get("key") or "", r.get("title") or ""])
        except Exception:
            pass
        cache[bid] = out
        return out
    return info


def _merged(since):
    """Recall and session-start decisions at or after `since`, outcomes merged in (latest wins)."""
    p = dl.path()
    by = {}
    if not p.is_file():
        return []
    with p.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("use") in USES:
                if (r.get("ts") or 0) >= since:
                    by[r["id"]] = r
            elif r.get("id") in by:
                by[r["id"]]["outcome"] = r.get("outcome")
    return list(by.values())


def _section(r, bid, sections, info):
    if r["use"] == "memrove.recall":
        return "entity" if bid.startswith("ent:") else "recall"
    if sections.get(bid):
        return sections[bid]
    k = _prefix_kind(bid) or info(bid)["kind"]  # logged before E34: no section on the row, so a preference record is taken to be a RULES line
    return {"rules": "rules", "preference": "rules", "thread": "threads", "last": "last", "doc": "docs", "file": "files"}.get(k, "records")


def stats(info, days=7, now=None):
    """-> dict: per-source / kind / project counts and rates, unresolved count, never-used records."""
    since = (now or time.time()) - days * 86400
    cells = []  # (source, id, status)
    unresolved = Counter()
    for r in _merged(since):
        sections = {o.get("id"): o.get("section") for o in (r.get("options") or []) if isinstance(o, dict) and r["use"] == "memrove.inject"}
        cards = (r.get("outcome") or {}).get("cards")
        for bid in r.get("answer") or []:
            src = _section(r, bid, sections, info)
            if src == "files" or bid.startswith(dl.USE_SKIP):
                continue  # always-loaded files and pointers have nothing the model could open or name
            if cards is None:
                unresolved[src] += 1
                continue
            st = cards.get(bid, "ignored")
            cells.append((src, bid, "cited" if st == "used" else st))

    def table(keyf):
        t = defaultdict(Counter)
        for src, bid, st in cells:
            t[keyf(src, bid)][st] += 1
        out = {}
        for k, c in sorted(t.items(), key=lambda kv: -sum(kv[1].values())):
            n = sum(c.values())
            out[k] = {"injected": n, **{s: c[s] for s in STATUSES}, **{s + "_rate": round(c[s] / n, 3) for s in STATUSES}}
        return out
    by_id = defaultdict(lambda: [0, 0, ""])
    for src, bid, st in cells:
        e = by_id[bid]
        e[0] += 1
        e[1] += st != "ignored"
        e[2] = e[2] or src
    never = sorted(((v[0], bid, v[2]) for bid, v in by_id.items() if v[1] == 0 and v[2] != "entity"), reverse=True)[:15]
    return {"days": days, "overall": table(lambda s, b: "all"),
            "by_source": table(lambda s, b: s),
            "by_kind": table(lambda s, b: info(b)["kind"] or "?"),
            "by_project": table(lambda s, b: info(b)["project"] or "?"),
            "unresolved": dict(unresolved),
            "never_used": [{"id": b, "source": s, "injected": n, "kind": info(b)["kind"], "project": info(b)["project"], "title": info(b)["title"]} for n, b, s in never]}


def render(rep):
    lines = [f"recall-stats, last {rep['days']} days (rates = share of injected ids with a resolved outcome)"]
    for name in ("overall", "by_source", "by_kind", "by_project"):
        lines.append(f"\n{name}:")
        lines.append(f"  {'':<14}{'injected':>9}{'opened':>8}{'cited':>7}{'ignored':>9}  opened%  cited%  ignored%")
        for k, v in rep[name].items():
            lines.append(f"  {k[:13]:<14}{v['injected']:>9}{v['opened']:>8}{v['cited']:>7}{v['ignored']:>9}  {v['opened_rate']*100:>6.1f}  {v['cited_rate']*100:>6.1f}  {v['ignored_rate']*100:>7.1f}")
    if rep["unresolved"]:
        lines.append(f"\nunresolved (no Stop/SessionEnd yet): {rep['unresolved']}")
    lines.append("\nnever used (most injected first):")
    lines += [f"  {x['injected']:>3}x {x['source']:<8} {x['id']}  [{x['kind']}/{x['project']}] {x['title']}" for x in rep["never_used"]]
    return "\n".join(lines)


def _inject_events(state, since, session=None):
    """(session, epoch, (ts, source, project), [[id, section]]) for session-start injections in inject-log.jsonl that have no decision row yet.
    Entries written before E34 carry no `blocks`; their ids come from the injected table (section unknown)."""
    have = {(_s(r).get("session"), _s(r).get("epoch")) for r in _merged(since) if r["use"] == "memrove.inject"}
    try:
        lines = (Path(state) / "inject-log.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    ents, legacy = {}, set()
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        k = (e.get("session"), e.get("epoch"))
        if (e.get("ts") or 0) >= since and k not in have and (session is None or k[0] == session):
            ents[k] = e
            if "blocks" not in e:
                legacy.add(k)
    if legacy:
        with sqlite3.connect(f"file:{Path(state) / 'memory.sqlite3'}?mode=ro", uri=True) as db:
            for s, ep, b in db.execute("SELECT session, epoch, block_id FROM injected"):
                if (s, ep) in legacy:
                    ents[(s, ep)].setdefault("blocks", []).append([b, ""])
    return [(s, ep, (e["ts"], e.get("source") or "startup", e.get("project") or ""), [x for x in e.get("blocks", []) if not x[0].startswith("recall:")])
            for (s, ep), e in ents.items()]


def materialize(state, session=None, days=3):
    """Write the missing memrove.inject decision rows (at the original injection time) -> count. Runs in the Stop worker and the backfill, never in the hook."""
    n = 0
    for sid, ep, (ts, source, project), blocks in _inject_events(state, time.time() - days * 86400, session):
        if blocks:
            n += bool(dl.log_inject(sid, project, source, ep, [{"id": b, "section": sec} for b, sec in blocks], ts=ts))
    return n


def _s(r):
    return dl._state(r)


def backfill(state, info, days=3, dest=None, live=False):
    """Run the detector over the logged injections of the last `days` days. Transcripts are only read. Outcomes go to a temp copy
    of the decision log (dest/decisions.jsonl) unless live=True. -> (decide_home, sessions resolved, injections synthesised)."""
    import shutil
    import tempfile
    since = time.time() - days * 86400
    src = dl.path()
    if live:
        home = os.environ.get("DECIDE_HOME")
    else:
        home = str(dest or tempfile.mkdtemp(prefix="memrove-usage-"))
        (Path(home) / "log").mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy(src, Path(home) / "log" / "decisions.jsonl")
    old = os.environ.get("DECIDE_HOME")
    if home:
        os.environ["DECIDE_HOME"] = home
    os.environ.pop("MEMROVE_NO_DECIDE_LOG", None)
    try:
        made = materialize(state, days=days)
        try:
            smap = json.loads((Path(state) / "session-map.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            smap = {}
        sessions = {_s(r).get("session") for r in _merged(since)}
        done = 0
        for sid in sessions:
            tp = (smap.get(sid) or {}).get("path")
            if tp and Path(tp).is_file():
                dl.resolve_session(sid, tp, lookup=lambda i: info(i)["needles"])
                done += 1
        return home, done, made
    finally:
        if old is None:
            os.environ.pop("DECIDE_HOME", None)
        else:
            os.environ["DECIDE_HOME"] = old

