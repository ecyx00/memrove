"""The 7-verb core: ingest, recall, expand, reflect, update, forget, feedback. One facade over existing code; every adapter (CLI, MCP, hooks) calls this.
Stdlib only, no LLM call. Design: personal-intelligence/docs/design/2026-11-07-memrove-engine.md sections 5, 6, 12.
scope = {space, project, agent, session, tags}. Hit = {id, text, score, arm, evidence:[{kind, ref}], valid_from, valid_to, origin}.
Evidence is a list but only kind "msg" (a chat-log message id) is built: memrove stores no code and links no code (owner, 2026-11-07)."""
import contextlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memrove_v3_access as access  # noqa: E402
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402
import memrove_v3_evidence as evidence  # noqa: E402
import memrove_v3_graph as graph  # noqa: E402
import memrove_v3_outcomes as outcomes  # noqa: E402
import memrove_v3_relate as relate  # noqa: E402
import memrove_v3_secrets as secrets  # noqa: E402
from memrove_v3 import is_forgotten, keep_forgotten, text_hash  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402

DEFAULT_AGENT = "claude-code"  # design 12.6: hook-captured records with no source


def agent_of(r):
    """Which agent wrote a record: its own field, else the reader that imported it ("codex-session" -> codex), else the hook default."""
    return r.get("agent") or (str(r.get("imported_from") or "").removesuffix("-session") or DEFAULT_AGENT)


FILE_FIELDS = {"close": ("valid_to", "superseded_by"), "reflect": ("unverified", "reflect")}  # _patch event -> payload fields kept in the vault file
SRC_AGENT = {"claude": "claude-code"}  # chat-log `src` -> agent name; codex, antigravity, ... are their own names


def _today():
    return datetime.now(timezone.utc).date().isoformat()


def _del(db, sql, args):
    try:
        return db.execute(sql, args).rowcount
    except sqlite3.OperationalError:  # table not created in this store
        return 0


_LIVE_BAD = ("deleted_by_user", "retired", "merged")
_PRIVATE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|https?://\S*\?\S|/(?:Users|home)/\w|~/")


def _private(text, state):
    """True if text must not leave the machine: redaction changes it, or it holds an email, a URL with a query or a home path."""
    return secrets.redact(text, state)[0] != text or bool(_PRIVATE.search(text))


def relate_new(core, rec, agent=None):
    """Typed edges from a just-written live record to its 3 nearest older live ones (lexical overlap). Off unless MEMROVE_RELATE=1;
    decider "none" (nothing sent) unless MEMROVE_RELATE_REMOTE=1 too, then pairs with secrets/emails/query URLs/home paths are skipped. Never raises.
    -> supersedes proposals [{src, dst, why}], also appended to state/reflect-proposals.jsonl. core.relate_decider: name or callable."""
    if os.environ.get("MEMROVE_RELATE") != "1" or not rec.get("id"):
        return []
    remote = os.environ.get("MEMROVE_RELATE_REMOTE") == "1"
    if remote and not access.can_remote(access.load(core.state), agent or DEFAULT_AGENT):
        return []  # the owner asked for remote relate, the policy forbids this agent's text to leave: skip the pass, no decider runs
    live = lambda r: not r.get("valid_to") and r.get("status") not in _LIVE_BAD
    try:
        with core.store._connect() as db:  # read only: no write transaction may be open while a decider does network calls
            rows = [json.loads(p) for _, p in db.execute("SELECT id, payload FROM records WHERE id != ?", (rec["id"],))]
            me = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rec["id"],)).fetchone()[0])
        if not live(me):
            return []
        new = {"id": me["id"], "text": me["text"], "valid_from": me.get("valid_from")}
        older = [{"id": r["id"], "text": r["text"], "valid_from": r.get("valid_from")} for r in rows if live(r) and str(r.get("valid_from") or "") <= str(new["valid_from"] or "")]
        top = sorted(older, key=lambda o: relate.overlap(o["text"], new["text"])[0], reverse=True)[:3]  # shortcut: full scan, upgrade to FTS at ~10k records
        if remote:
            top = [o for o in top if not (_private(o["text"], core.state) or _private(new["text"], core.state))]
        decider = getattr(core, "relate_decider", None) or ("jev" if remote else "none")
        res = relate.relate(None, [(o, new) for o in top], decider=decider, remote_ok=remote, write=False)
        edges = [r for r in res if r["status"] == "edge"]
        if edges:  # one short write transaction, after all decisions
            name = decider if isinstance(decider, str) else getattr(decider, "__name__", "custom")
            with core.store._connect() as db:
                for r in edges:
                    relate.write_edge(db, r, new["valid_from"], name)
        props = [{"src": r["src"], "dst": r["dst"], "why": r["why"]} for r in res if r["status"] == "proposed"]
        if props:
            ts = datetime.now(timezone.utc).isoformat()
            with (Path(core.state) / "reflect-proposals.jsonl").open("a", encoding="utf-8") as f:
                for p in props:  # line shape of memrove_v3_reflect._propose: day, verdict, id (+ extras)
                    f.write(json.dumps({"day": ts[:10], "verdict": "supersedes", "id": p["src"], **p, "by": "relate", "ts": ts}, ensure_ascii=False) + "\n")
        return props
    except Exception as e:  # never raises; class name only, the text may hold a key
        print(f"memrove: relate failed: {type(e).__name__}", file=sys.stderr)
        return []


class Core:
    """Core(vault, state): the engine bound to one vault and one state dir (memory.sqlite3 lives in state)."""

    def __init__(self, vault, state):
        self.sync = SyncEngine(vault, state)
        self.store, self.state = self.sync.store, str(self.sync.state)
        with self.store._connect() as db:
            graph.ensure(db)  # idempotent bi-temporal migration
            self._backfill_agent(db)

    @staticmethod
    def _backfill_agent(db):
        """Write `agent` into records that lack it (derived by agent_of). Idempotent; no row is added or removed; no event (derived metadata, agent_of covers a rebuilt record)."""
        for rid, p in db.execute("SELECT id, payload FROM records WHERE json_extract(payload,'$.agent') IS NULL").fetchall():
            r = json.loads(p)
            r["agent"] = agent_of(r)
            db.execute("UPDATE records SET payload=? WHERE id=?", (json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")), rid))

    # ---- write
    def ingest(self, items, scope=None):
        """items: [str | {text, kind, why, topic, project, source, valid_from}] -> [{id, status}]. kind defaults to note."""
        from memrove_v3_context import remember
        scope = scope or {}
        agent = scope.get("agent") or DEFAULT_AGENT
        if not access.can_write(access.load(self.state), agent):
            return [{"id": None, "status": "denied"} for _ in ([items] if isinstance(items, (str, dict)) else items)]
        out = []
        for it in ([items] if isinstance(items, (str, dict)) else items):
            it = {"text": it} if isinstance(it, str) else dict(it)
            with self.store._connect() as db:
                if is_forgotten(db, {"text": it["text"]}):
                    out.append({"id": None, "status": "forgotten"})
                    continue
            out.append(remember(self.sync, it.get("kind", "note"), it["text"], why=it.get("why"), project=it.get("project") or scope.get("project"),
                                session=scope.get("session"), source=it.get("source", "memrove-core"), topic=it.get("topic"),
                                valid_from=it.get("valid_from"), agent=scope.get("agent") or DEFAULT_AGENT, user=bool(it.get("user")),
                                supersedes=it.get("supersedes") or (), check=it.get("check"), measured=it.get("measured")))
            if out[-1].get("id"):
                try:
                    evidence.mark(self, out[-1]["id"])  # shortcut: loads the whole chat log per call, pass a cached log= if ingest gets hot
                except Exception as e:  # a failed verdict must never fail the write; class name only
                    print(f"memrove: evidence mark failed: {type(e).__name__}", file=sys.stderr)
            out[-1]["proposals"] = relate_new(self, out[-1], agent)
        return out

    def _one(self, db, rid):
        rid = rid.removeprefix("rec:")
        rows = db.execute("SELECT id, payload FROM records WHERE id=? OR id LIKE ? LIMIT 2", (rid, rid + "%")).fetchall()
        exact = [r for r in rows if r[0] == rid]
        if len(rows) != 1 and not exact:
            raise ValueError(f"no unique record for {rid!r}")
        return (exact or rows)[0]

    def update(self, rid, new, supersedes=True):
        """Write `new` (str | dict as in ingest) and close the old record: old.valid_to = new.valid_from. Nothing is deleted."""
        from memrove_v3_context import remember
        new = {"text": new} if isinstance(new, str) else dict(new)
        with self.store._connect() as db:
            oid, payload = self._one(db, rid)
        old = json.loads(payload)
        start = new.get("valid_from") or _today()
        res = remember(self.sync, new.get("kind", old.get("kind", "note")), new["text"], why=new.get("why"), project=new.get("project") or old.get("project"),
                       source=new.get("source", "memrove-core"), topic=new.get("topic") or old.get("topic"), valid_from=start,
                       agent=new.get("agent") or old.get("agent") or DEFAULT_AGENT, supersedes=[oid] if supersedes else (), close=False)
        closes = supersedes and not old.get("valid_to") and res["id"] != oid and res.get("status") != "exists"  # same text = same record: never close what was just written
        if closes:
            self._patch(oid, "close", valid_to=start)
        return {**res, "closed": oid if closes else None, "proposals": relate_new(self, res)}

    def _patch(self, rid, event, **fields):
        """Set payload fields on a stored record and log one event. Returns the new payload; nothing is deleted."""
        with self.store._connect() as db:
            old = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])
            old.update(fields)
            if event in FILE_FIELDS:
                self._fields_to_file(old, FILE_FIELDS[event])
            body = json.dumps(old, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            db.execute("UPDATE records SET payload=? WHERE id=?", (body, rid))
            db.execute("INSERT INTO events(event_type,record_id,revision,record) VALUES (?,?,?,?)", (event, rid, old.get("revision", 1), body))
        return old

    def _fields_to_file(self, rec, keys):
        """A close or a reflect mark is also written into the source file's front matter, so a store rebuilt from the vault keeps it.
        rec["source_sha256"] follows the new bytes, so sync does not see a hand edit."""
        from memrove_v3_sync import _hash, atomic, parse, render
        try:
            path = self.sync._path(rec["source"], existing=True)
            meta, body = parse(path.read_text(encoding="utf-8"))
        except (KeyError, ValueError, OSError):
            return  # no file behind this record
        meta.update({k: rec[k] for k in keys if rec.get(k)})
        text = render(meta, body)
        atomic(path, text)
        rec["source_sha256"] = _hash(text)

    def _hide_text(self, texts):
        """Hard forget: hide every copy of the text that stays on disk (chat log, imports.db) from all readers by id; drop the derived rows
        (chat search index, idea cards). Nothing in the chat log, the imports or the vault is deleted. Text shorter than 8 characters is not matched."""
        want = [" ".join(t.split()).casefold() for t in texts if len(" ".join(t.split())) >= 8]
        hit = lambda t: any(w in " ".join(str(t).split()).casefold() for w in want)
        out = {"chat": 0, "imports": 0, "search": 0, "cards": 0}
        if not want:
            return out
        d = chatlog.root(self.state)
        chat = [r["i"] for f in chatlog._files(d) for r in chatlog.read_file(f) if hit(r.get("text"))]
        imp = []
        if (d / "imports.db").exists():
            with contextlib.closing(sqlite3.connect(d / "imports.db", timeout=30)) as db:
                imp = [i for i, t in db.execute("SELECT id, text FROM x") if hit(t)]
        chatlog.hide(self.state, chat, imp)
        out["chat"], out["imports"] = len(chat), len(imp)
        for name, table, fts, ids in (("search.db", "m", "fts", chat), (None, "cards", "cards_fts", chat)):
            path = d / name if name else Path(self.state) / "ideacards.sqlite3"
            if not ids or not path.exists():
                continue
            with contextlib.closing(sqlite3.connect(path, timeout=30)) as db:
                try:
                    for i in ids:
                        out["search" if name else "cards"] += db.execute(f"DELETE FROM {table} WHERE id=?", (i,)).rowcount
                        db.execute(f"DELETE FROM {fts} WHERE rowid=?", (i,))
                    db.commit()
                except sqlite3.OperationalError:
                    pass
        return out

    def forget(self, target, mode="tombstone"):
        """target: record id (prefix ok) or a scope dict (project/session/agent, at least one). tombstone keeps the row (status deleted_by_user, valid_to set).
        hard deletes the row, its events, graph edges/node, links, vectors and lists id + text hash in `forgotten`. -> {mode, ids, deleted: {table: n}}."""
        if mode not in ("tombstone", "hard"):
            raise ValueError("mode must be tombstone or hard")
        with self.store._connect() as db:
            if isinstance(target, dict):
                keys = [k for k in ("project", "session", "agent") if target.get(k)]
                if not keys:
                    raise ValueError("a forget scope needs project, session or agent")
                where = " AND ".join(f"json_extract(payload,'$.{k}')=?" for k in keys)
                rows = db.execute(f"SELECT id, payload FROM records WHERE {where}", [target[k] for k in keys]).fetchall()
            else:
                rows = [self._one(db, target)]
            n, texts, listed = {}, [], []
            if mode == "hard":
                db.execute("PRAGMA secure_delete=ON")  # freed pages are zeroed, not left with the text
            for rid, payload in rows:
                if mode == "tombstone":
                    self.sync._tombstone(db, rid, payload)
                    continue
                rec = json.loads(payload)
                texts.append(rec.get("text") or "")
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for h in ("id:" + rid, text_hash(rec.get("text"))):
                    db.execute("INSERT OR IGNORE INTO forgotten VALUES (?,?,?,?)", (h, rid, rec.get("source"), now))
                    listed.append([h, rid, rec.get("source"), now])
                for t, sql, a in (("events", "DELETE FROM events WHERE record_id=?", (rid,)), ("records", "DELETE FROM records WHERE id=?", (rid,)),
                                  ("markdown_sources", "DELETE FROM markdown_sources WHERE id=?", (rid,)),
                                  ("graph_edges", "DELETE FROM graph_edges WHERE src=? OR dst=?", (rid, rid)),
                                  ("graph_nodes", "DELETE FROM graph_nodes WHERE id=?", (rid,)),
                                  ("links", "DELETE FROM links WHERE src=? OR dst=?", (rid, rid))):
                    n[t] = n.get(t, 0) + _del(db, sql, a)
            ids = [r[0] for r in rows]
            if mode == "hard":
                keep_forgotten(self.state, listed)  # before the commit: if it fails nothing is deleted
        if mode == "hard":
            n["vectors"] = self._drop_vectors(ids)
            n["hidden"] = self._hide_text(texts)
        return {"mode": mode, "ids": ids, "deleted": n}

    def _drop_vectors(self, ids):
        path = Path(self.state) / "vectors.sqlite3"
        if not path.exists():
            return 0
        db = sqlite3.connect(path, timeout=30)
        try:
            c = sum(db.execute("DELETE FROM link WHERE id=?", (i,)).rowcount for i in ids)
            c += db.execute("DELETE FROM vec WHERE h NOT IN (SELECT h FROM link)").rowcount
            db.commit()
            return c
        finally:
            db.close()

    def feedback(self, rid, signal="used", reason=""):
        """signal: used | wrong | stale. Logs one row in the decision log; not used in ranking yet (design 12.7)."""
        return outcomes.feedback(rid, signal, reason)

    # ---- read
    def recall(self, query, scope=None, arms=None, k=8, budget_tokens=None, as_of=None, agent=None):
        """Chat log + records through chatsearch.merged (the 93.1 any@5 lane), as Hits in merged order. arms: subset of {"chat","record"}. as_of: ISO day. agent: only hits written by that agent (chat messages by their source). scope.session filters records only (chat rows carry no session id). scope.agent unset with an access.json: the policy default applies."""
        scope = scope or {}
        rows = cs.merged(self.state, query, top=k * 4 if agent else k, until=as_of)
        recs = {i.removeprefix("rec:") for i, *_ in rows if isinstance(i, str) and i.startswith("rec:")}
        meta = {}
        if recs:
            with self.store._connect() as db:
                for rid, p in db.execute(f"SELECT id, payload FROM records WHERE id IN ({','.join('?' * len(recs))})", list(recs)):
                    meta[rid] = json.loads(p)
        cids = {int(i) for i, *_ in rows if str(i).isdigit() and not (isinstance(i, str) and i.startswith("rec:"))}
        msgs = chatlog.get_many(self.state, cids) if cids else {}  # one read per day file, shared by policy, origin and _by_agent
        hits, used = [], 0
        reader, policy, sess = scope.get("agent"), access.load(self.state), scope.get("session")
        for rank, (i, date, kind, text) in enumerate(rows):
            r = meta.get(i.removeprefix("rec:")) if isinstance(i, str) else None
            arm = "record" if r is not None else "chat"
            m = None if r is not None else (msgs.get(int(i)) if str(i).isdigit() else None)
            if not access.can_read(policy, reader, (r if r is not None else m or {}).get("project" if r is not None else "proj")):
                continue
            if sess and r is not None and r.get("session") != sess:  # chat rows carry no session id: a session scope filters records only
                continue
            if (arms and arm not in arms) or (r is not None and scope.get("project") and r.get("project") != scope["project"]):
                continue
            if agent and not self._by_agent(r, m, agent):
                continue
            if len(hits) >= k:
                break
            used += len(text) // 4 + 1
            if budget_tokens and hits and used > budget_tokens:
                break
            hits.append({"id": i, "text": text, "score": 1.0 / (1 + rank), "arm": arm, "evidence": [] if r is not None else [{"kind": "msg", "ref": i}],
                         "valid_from": (r or {}).get("valid_from") or date, "valid_to": (r or {}).get("valid_to"),
                         "origin": (r or {}).get("agent") or (DEFAULT_AGENT if r is not None else (SRC_AGENT.get(m.get("src"), m.get("src")) if m and m.get("src") else kind))})
        return hits

    def _by_agent(self, r, m, agent):
        if r is not None:
            return agent_of(r) == agent
        # an import id has no source here (m is None): fail closed
        return bool(m) and SRC_AGENT.get(m.get("src"), m.get("src")) == agent

    def as_of(self, t, project=None, kind=None, agent=None):
        """Records true at ISO time t: valid_from <= t AND (valid_to IS NULL OR valid_to > t). -> [(id, payload dict)]."""
        q = ("SELECT id, payload FROM records WHERE json_extract(payload,'$.valid_from') <= ? "
             "AND (json_extract(payload,'$.valid_to') IS NULL OR json_extract(payload,'$.valid_to') > ?)")
        args = [t, t]
        for k, v in (("project", project), ("kind", kind)):
            if v:
                q += f" AND json_extract(payload,'$.{k}')=?"
                args.append(v)
        with self.store._connect() as db:
            rows = [(i, json.loads(p)) for i, p in db.execute(q, args)]
        return [(i, r) for i, r in rows if not agent or agent_of(r) == agent]

    def edges_as_of(self, rid, t):
        with self.store._connect() as db:
            return db.execute("SELECT src,dst,type FROM graph_edges WHERE (src=? OR dst=?) AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)",
                              (rid, rid, t, t)).fetchall()

    def expand(self, rid, depth=1):
        """A record (payload + graph neighbours to `depth` hops) or a chat message (id is an int / numeric string / x-import id)."""
        if isinstance(rid, int) or str(rid).isdigit():
            r = chatlog.get(self.state, int(rid))
            if r is None:
                raise ValueError(f"message {rid} not found")
            return {"id": int(rid), "arm": "chat", "text": r["text"], "kind": r["kind"], "date": r.get("date", "")}
        with self.store._connect() as db:
            oid, payload = self._one(db, str(rid))
            seen, front, edges = {oid}, [oid], []
            for _ in range(max(0, depth)):
                nxt = []
                for n in front:
                    for s, d, t in db.execute("SELECT src,dst,type FROM graph_edges WHERE (src=? OR dst=?) AND valid_to IS NULL", (n, n)):
                        edges.append({"src": s, "dst": d, "type": t})
                        nxt += [x for x in (s, d) if x not in seen]
                        seen.update((s, d))
                front = nxt
        return {"id": oid, "arm": "record", "record": json.loads(payload), "edges": edges}

    def reflect(self, scope=None, kind="project", day=None, cap_tokens=None, max_calls=None, dry=False, until=None, check_model=None):
        """kind: card (newest project card), tree (chat summary tree, part 1), project (task-start pack for the project),
        dream (idle pass over one day, Haiku via memrove_v3_reflect: marks unsupported records, merges duplicates, closes stale ones; never deletes) -> stats."""
        scope = scope or {}
        if kind == "dream":
            import memrove_v3_reflect as dream
            kw = {k: v for k, v in (("cap_tokens", cap_tokens), ("max_calls", max_calls)) if v}
            return dream.run(self, day, scope=scope, dry=dry, until=until, check_model=check_model, **kw)
        if kind == "settle":
            import memrove_v3_reflect as dream
            return dream.settle(self, day)
        if kind == "card":
            import memrove_v3_card as card
            return card.block(self.state, scope.get("project"))
        if kind == "tree":
            import memrove_v3_chattree as tree
            return tree.wake(self.state)
        if kind == "project":
            from memrove_v3_context import context_pack
            return context_pack(self.store, "", project=scope.get("project"))
        raise ValueError("kind must be card, tree, project or dream")
