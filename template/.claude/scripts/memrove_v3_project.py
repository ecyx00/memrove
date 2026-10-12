"""One project resolution for capture and recall: the nearest ancestor of cwd that is a git root or carries a project marker.
Never the home directory, never the vault, never an unmarked subdirectory (docs/, src/, ...). A nested repo is its own project
only because it is the nearest git root."""
import json
import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import quote

MARKERS = (".git", ".memrove", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Package.swift", "pom.xml", "Gemfile", "CLAUDE.md", "AGENTS.md")
WORKSPACES = ("Desktop", "Projects", "Developer", "code", "src")  # under home; state/project-config.json {"workspace_roots": [...]} adds more


def workspace_roots(state=None):
    home = Path.home().resolve()
    roots = {home / w for w in WORKSPACES}
    try:
        extra = json.loads((Path(state) / "project-config.json").read_text(encoding="utf-8")).get("workspace_roots", []) if state else []
    except (OSError, ValueError, AttributeError):
        extra = []
    roots |= {Path(x).expanduser().resolve() for x in extra if isinstance(x, str) and x}
    return roots


def resolve(cwd, state=None, vault=None):
    """-> (name, root): the nearest git root / marker directory above cwd; failing that, the ancestor that is a direct child of a
    workspace root (~/Desktop, ~/Projects, ... plus state/project-config.json). Else (None, None).
    Home, `/`, workspace roots themselves and the vault (and anything inside it) are never projects."""
    start = Path(cwd or ".").expanduser().resolve()
    home = Path.home().resolve()
    vault = Path(vault).expanduser().resolve() if vault else None
    if vault and (start == vault or vault in start.parents):
        return None, None
    roots = workspace_roots(state)
    chain = [p for p in (start, *start.parents)]
    for p in chain:
        if p == home or p == p.parent:
            break
        if p not in roots and any((p / m).exists() for m in MARKERS):
            return p.name, p
    for p in chain:
        if p.parent in roots and p != vault:
            return p.name, p
    return None, None


def project_root(cwd, state=None, vault=None):
    """Root directory of the project (the cwd itself when none resolves; use project_name for the label)."""
    return resolve(cwd, state, vault)[1] or Path(cwd or ".").expanduser().resolve()


def project_name(cwd, state=None, vault=None):
    return resolve(cwd, state, vault)[0] or "unknown"


UNRESOLVED = "unknown"  # no project by cwd or by touched paths; effective_scope maps it to user scope, which clients recall
CHUNK = 256 * 1024  # most new transcript bytes one hook-path call reads
_CD = re.compile(r"""\bcd\s+["']?(/[^\s"';&|)]+)""")


def _count(data, state, vault, hits):
    for raw in data.splitlines():
        if b"tool_use" not in raw:
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        content = (rec.get("message") or {}).get("content") if isinstance(rec, dict) else None
        for b in content if isinstance(content, list) else []:
            if not isinstance(b, dict) or b.get("type") != "tool_use" or not isinstance(b.get("input"), dict):
                continue
            inp, paths = b["input"], []
            for k in ("file_path", "path", "notebook_path"):
                if isinstance(inp.get(k), str) and inp[k].startswith("/"):
                    paths.append(inp[k] if k == "path" else os.path.dirname(inp[k]))
            if isinstance(inp.get("command"), str):
                paths += _CD.findall(inp["command"])
            for path in paths:
                name = resolve(path, state, vault)[0]
                if name and not name.startswith("."):  # ~/.claude and friends are configuration, not a project the user works on
                    hits[name] = hits.get(name, 0) + 1


def _pick(hits, min_hits=3, share=0.6):
    total = sum(hits.values())
    if total:
        top, n = max(hits.items(), key=lambda kv: (kv[1], kv[0]))
        if n >= min_hits and n >= share * total:
            return top
    return None


def infer(transcript, state=None, vault=None, session=None, max_bytes=8_000_000):
    """Project a session worked on, from the absolute paths its own tool calls touch (file_path/path/notebook_path inputs and `cd /abs`
    targets in Bash commands), each resolved with the cwd rule. -> name when one project has >= 3 hits and >= 60% of the hits that
    resolve to a project, else None. Deterministic. With `session` the scan is incremental: state/session-paths.json keeps a byte cursor
    and the running counts, and one call reads at most CHUNK new bytes (later events catch up), so the hook path stays cheap."""
    path = Path(state) / "session-paths.json" if (state and session) else None
    saved = {}
    if path:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
    cur = saved.get(session) or {"cursor": 0, "hits": {}}
    hits, cursor = dict(cur["hits"]), cur["cursor"] if path else 0
    try:
        with open(transcript, "rb") as f:
            if os.fstat(f.fileno()).st_size < cursor:
                hits, cursor = {}, 0  # transcript was replaced
            f.seek(cursor)
            data = f.read(CHUNK if path else max_bytes)
    except (OSError, TypeError):
        return None
    if path and not data.endswith(b"\n") and b"\n" in data:
        data = data[:data.rfind(b"\n") + 1]  # only whole lines; the rest is the next call's
    _count(data, state, vault, hits)
    if path and data:
        saved[session] = {"cursor": cursor + len(data), "hits": hits}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(dict(list(saved.items())[-200:])), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass
    return _pick(hits)


def sticky(state, session, cwd, vault=None, transcript=None):
    """Project of a session, resolved once from its first resolvable event and kept (state/session-projects.json).
    -> (name, root) or (None, None). When the cwd resolves to nothing (a session started from home) the project is inferred from the paths the
    transcript touched (infer); that root is None. Nothing found is not stored, so a later event may still resolve."""
    path = Path(state) / "session-projects.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    if session in saved:
        return saved[session]["name"], Path(saved[session]["root"]) if saved[session].get("root") else None
    name, root = resolve(cwd, state, vault)
    if not name and transcript:
        name = infer(transcript, state, vault, session)
    if name:
        saved[session] = {"name": name, "root": str(root) if root else None}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(dict(list(saved.items())[-500:])), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass  # the label is still right for this event; the checkpoint must not be lost over it
    return name, root


def relabel_plan(state, transcripts_dir=None):
    """Dry run: what re-resolving every record's project from its session's first cwd would change.
    -> {"changes": {"old -> new": count}, "unresolvable": n, "records": [{id, old, new}]}. Writes nothing."""
    state = Path(state)
    try:
        smap = json.loads((state / "session-map.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        smap = {}
    first_cwd, plan, changes, unresolvable = {}, [], {}, 0
    db = sqlite3.connect(f"file:{state / 'memory.sqlite3'}?mode=ro", uri=True, timeout=1)
    try:
        rows = [(i, json.loads(p)) for i, p in db.execute("SELECT id,payload FROM records")]
    finally:
        db.close()
    for rid, rec in rows:
        old, sid = rec.get("project"), rec.get("session")
        if not old or not sid:
            continue
        if sid not in first_cwd:
            first_cwd[sid] = _first_cwd((smap.get(sid) or {}).get("path"))
        if not first_cwd[sid]:
            unresolvable += 1
            continue
        new = project_name(first_cwd[sid], state)
        if new == "unknown":
            new = infer((smap.get(sid) or {}).get("path"), state) or UNRESOLVED
        if new != old:
            plan.append({"id": rid, "old": old, "new": new})
            key = f"{old} -> {new}"
            changes[key] = changes.get(key, 0) + 1
    return {"dry_run": True, "changes": changes, "unresolvable": unresolvable, "records": plan}


def _on_disk(name, state):
    """A project root with this directory name exists under a workspace root (git root or marker; ~/Projects/panel)."""
    return any((r / name).is_dir() and resolve(r / name, state)[0] == name for r in workspace_roots(state))


def _targets(state, known=None):
    """-> (moves, names): moves {record id: (old, new)} and names {stale project name: new}. A name is stale when none of its
    records resolves to it but some resolve elsewhere (docs, notes, home: home-directory subfolders, not projects). Records of a stale
    name that cannot be resolved follow the name's most common resolved target. Names with no evidence (their sessions left no
    transcript) are real projects as far as the data shows and stay. Records with relabeled_from are never moved again."""
    plan = relabel_plan(state)
    changed = {r["id"]: r for r in plan["records"]}
    db = sqlite3.connect(f"file:{Path(state) / 'memory.sqlite3'}?mode=ro", uri=True, timeout=1)
    try:
        rows = [(i, json.loads(p)) for i, p in db.execute("SELECT id,payload FROM records")]
    finally:
        db.close()
    tally, kept = {}, set()
    for rid, rec in rows:
        if rid in changed:
            n = tally.setdefault(rec["project"], {})
            n[changed[rid]["new"]] = n.get(changed[rid]["new"], 0) + 1
    smap = json.loads((Path(state) / "session-map.json").read_text(encoding="utf-8")) if (Path(state) / "session-map.json").exists() else {}
    for rid, rec in rows:  # resolvable and unchanged = the name is confirmed
        sid = rec.get("session")
        if rec.get("project") in tally and sid and rid not in changed and _first_cwd((smap.get(sid) or {}).get("path")):
            kept.add(rec["project"])
    names = {old: max(n.items(), key=lambda kv: (kv[1], kv[0]))[0] for old, n in tally.items() if old not in kept}
    names = {**(known or {}), **names}
    names = {k: v for k, v in names.items() if not _on_disk(k, state)}  # a project that exists on disk is never stale (home-cwd sessions still work on it)  # names stale in an earlier run keep applying to records that appear or fail later
    moves = {}
    for rid, rec in rows:
        if rec.get("relabeled_from") or not rec.get("project"):
            continue
        if rid in changed:
            if changed[rid]["new"] == "unknown" and _on_disk(rec["project"], state):
                continue
            moves[rid] = (rec["project"], changed[rid]["new"])
        elif rec["project"] in names:
            moves[rid] = (rec["project"], names[rec["project"]])
    return moves, names


def _marks(vault, fn):
    """Apply fn(regex_match_kind, project) -> project to every thread/card marker in Threads.md. -> number of markers changed."""
    import re
    import memrove_v3_threads as th
    from memrove_v3_companion import directory
    d = directory(vault)
    path = d / "Threads.md" if d else None
    if not path or not path.exists():
        return 0
    text, crlf = th._read(path)
    count = 0

    def sub(kind):
        def run(m):
            nonlocal count
            new = fn(kind, th._proj(m), m)
            if new == th._proj(m):
                return m[0]
            count += 1
            return m[0].replace("project=" + m[2], "project=" + quote(new, safe=""))
        return run
    out = th.CARD_RE.sub(sub("card"), th.THREAD_RE.sub(sub("thread"), text))
    if count:
        th._write(path, out, crlf)
    return count


def _move(engine, db, rid, changes):
    """update_record, plus the front matter of a source file the store does not own (notes/observed/...): sync re-reads the project
    from that file, so it must change first (update_record then recomputes source_sha256). Owned notes are rewritten by update_record."""
    from memrove_v3_projections import OWNED, update_record
    from memrove_v3_sync import atomic, parse, render
    src = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0]).get("source") or ""
    path = engine._path(src) if src else None
    if path and not src.startswith(OWNED) and path.is_file():
        meta, body = parse(path.read_text(encoding="utf-8"))
        if meta:
            meta.update({k: v for k, v in changes.items() if v is not None})
            for k in [k for k, v in changes.items() if v is None]:
                meta.pop(k, None)
            atomic(path, render(meta, body))
    return update_record(engine, rid, changes, db)


def relabel_apply(engine, state, vault, undo=False, only=None):
    """Move records (and the threads/cards of stale project names) to their resolved project through update_record, keeping the old
    project in the record's relabeled_from and in state/relabel-journal.json. undo=True restores from those. Idempotent.
    -> counts. Caller takes the backup first and runs engine.sync() after."""
    import memrove_v3_threads as th
    state = Path(state)
    jpath = state / "relabel-journal.json"
    try:
        journal = json.loads(jpath.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        journal = {}
    journal.setdefault("names", {})
    journal.setdefault("threads", {})
    moved = 0
    if undo:
        with engine.store._connect() as db:
            ids = [i for i, p in db.execute("SELECT id,payload FROM records") if json.loads(p).get("relabeled_from") and (not only or json.loads(p)["relabeled_from"] == only)]
            for rid in ids:
                old = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])["relabeled_from"]
                moved += _move(engine, db, rid, {"project": old, "relabeled_from": None})

        def revert(kind, proj, m):
            old = journal["threads"].get(m[1] if kind == "thread" else "card:" + m[1])
            return old if old and (not only or old == only) else proj
        marks = _marks(vault, revert)
        th._index_edit(state, lambda idx: [t.update(project=journal["threads"][tid]) for tid, t in idx.get("threads", {}).items()
                                           if tid in journal["threads"] and (not only or journal["threads"][tid] == only)])
        journal["threads"] = {k: v for k, v in journal["threads"].items() if only and v != only}
        journal["names"] = {k: v for k, v in journal["names"].items() if only and k != only}
        jpath.write_text(json.dumps(journal, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"undone_records": moved, "markers": marks}
    moves, names = _targets(state, journal["names"])
    with engine.store._connect() as db:
        for rid, (old, new) in moves.items():
            moved += _move(engine, db, rid, {"project": new, "relabeled_from": old})
    journal["names"].update(names)

    def move(kind, proj, m):
        new = names.get(proj)
        if new:
            journal["threads"].setdefault(m[1] if kind == "thread" else "card:" + m[1], proj)
        return new or proj
    marks = _marks(vault, move)

    def edit(idx):
        for tid, t in idx.get("threads", {}).items():
            if t.get("project") in names:
                journal["threads"].setdefault(tid, t["project"])
                t["project"] = names[t["project"]]
    th._index_edit(state, edit)
    jpath.write_text(json.dumps(journal, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"moved_records": moved, "stale_names": names, "markers": marks}


def _first_cwd(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for _, line in zip(range(50), f):
                try:
                    cwd = json.loads(line).get("cwd")
                except ValueError:
                    continue
                if cwd:
                    return cwd
    except OSError:
        pass
    return None


# ---------------------------------------------------------------- project entry (pull): `memrove project [name]`
def _t(text, n):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def entry(db, vault, project, limit=5000, decisions=6, old=3, state=None):
    """Where a project stands, in plain lines that each start with an id `memrove show <id>` opens: the latest decisions (the ones a later
    record or a supersedes edge replaced are marked), open loops (grab cards flagged open, open threads), the last session's end state.
    Built from stored facts only (no model call); `limit` bytes cap the whole text, each section keeps its own share."""
    import memrove_v3_graph as g
    import memrove_v3_threads as th
    nodes = g.load_nodes(db, vault)
    name = g.view_project(project)
    mine = [n for n in nodes.values() if n["project"] == name and n["kind"] != "concept"]
    if not mine:
        known = sorted({n["project"] for n in nodes.values() if n["project"] and n["project"] != g.GENERIC and n["kind"] != "concept"})
        return f"no project {project!r}; projects: " + ", ".join(known)
    newer = {}  # old id -> id of the record that replaced it (record field or a stored supersedes edge)
    for e in g.all_edges(db, nodes):
        if e["type"] == "supersedes" and not e["dst"].startswith(("concept:", "project:")):
            newer.setdefault(e["dst"], e["src"])
    order = lambda n: (n.get("ts") or n["day"], n["id"])  # full timestamp when the record has one; a date-only record sorts first within its day
    decs = sorted((n for n in mine if n["kind"] == "decision"), key=order, reverse=True)
    cur = [n for n in decs if n["id"] not in newer and n.get("status") != "superseded"]
    gone = [n for n in decs if n not in cur]
    out = [f"project {name}"]
    if state:
        import memrove_v3_heads as heads
        out += heads.render(state, name)
    out.append("decisions (latest first):" if decs else "decisions: none recorded")
    for n in cur[:decisions]:
        out.append(f"  rec:{n['id']} {n['day']} {_t(n['title'], 110)}")
    for n in gone[:old]:
        by = newer.get(n["id"])
        out.append(f"  rec:{n['id']} {n['day']} {_t(n['title'], 90)} [SUPERSEDED" + (f" by rec:{by}" if by else "") + "]")
    names = {name, *(str(project).split("/")[-1:] if project else [])}
    loops = sorted((n for n in mine if n["kind"] == "card" and n.get("open")), key=order, reverse=True)
    try:
        open_threads = th.inject_items(vault, names, limit=6) if th.enabled() else []
    except Exception:  # ponytail: companion files are best-effort, the rest still answers
        open_threads = []
    if loops or open_threads:
        out.append("open loops:")
        out += [f"  rec:{n['id']} {n['day']} {_t(n['title'], 110)}" for n in loops[:5]]
        out += [f"  {short}" for _id, _full, short in open_threads]
    try:
        last = th.last_card(vault, names) if th.enabled() else []
    except Exception:
        last = []
    if last:
        out.append("last session (end state):")
        out += ["  " + ln.strip() if i == 0 else "    " + ln.strip() for _id, full, _s in last[:1] for i, ln in enumerate(full.split("\n"))]
    return g._clip(out, limit)
