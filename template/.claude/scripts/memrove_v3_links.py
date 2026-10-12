"""Canonical graph edges in v3's SQLite. Code derives edges; the model never writes them.

Node ids: record id as-is, 'project:<name>', 'area:<dir>', session id as-is.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import posixpath
import re
import sqlite3
from urllib.parse import unquote
import sys

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def _wikilink_metni(text):
    """Text without code spans and fences, where Obsidian makes no graph edges."""
    return _CODE.sub("", text)


def _hedef(raw):
    """Link target: drop the alias after `|` and any #heading or ^block part."""
    target = raw.split("|", 1)[0].rstrip("\\")
    return unquote(re.split(r"[#^]", target, maxsplit=1)[0].strip())


DEGREE_CAP = 12
MAX_AREAS = 8
WEIGHT = {'in_project': 1.0, 'from_session': 1.0, 'supersedes': 1.0, 'rejects': 0.9,
          'next': 0.8, 'mentions': 0.6, 'touches': 0.5}
SCHEMA = 'CREATE TABLE IF NOT EXISTS links(src TEXT NOT NULL, dst TEXT NOT NULL, rel TEXT NOT NULL, w REAL NOT NULL, ts TEXT NOT NULL, PRIMARY KEY(src,dst,rel))'
_TYPE = "coalesce(json_extract(payload,'$.type'),json_extract(payload,'$.kind'))"


def _type(rec):
    return rec.get('type') or rec.get('kind') or 'note'


def _node(rec):
    """Session node of a record: its session field, or its own id for a session record."""
    return rec.get('session') or (rec['id'] if _type(rec) == 'session' else None)


class Links:
    def __init__(self, store):
        self.store = store
        if not store.read_only:
            with store._connect() as db:
                db.execute(SCHEMA)

    @contextmanager
    def _db(self, db):
        if db is not None:
            yield db
        else:
            with self.store._connect() as conn:
                yield conn

    # -- public API ---------------------------------------------------------
    def add(self, src, dst, rel, w=None, db=None):
        if not src or not dst or src == dst:
            return
        w = WEIGHT.get(rel, 0.5) if w is None else float(w)
        ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
        with self._db(db) as c:
            c.execute('INSERT OR REPLACE INTO links VALUES (?,?,?,?,?)', (src, dst, rel, w, ts))
            # degree cap: keep the 12 heaviest out-edges, newest first on ties
            c.execute('DELETE FROM links WHERE src=? AND rowid NOT IN (SELECT rowid FROM links WHERE src=? '
                      'ORDER BY w DESC, ts DESC, rel, dst LIMIT ?)', (src, src, DEGREE_CAP))

    def expand(self, ids, rels, limit, db=None):
        """1 hop in both directions -> [(other_id, rel, w)], heaviest first, excluding ids themselves."""
        ids, rels = list(dict.fromkeys(ids)), list(rels)
        if not ids or not rels or limit <= 0:
            return []
        mark = lambda items: ','.join('?' * len(items))
        out = {}
        try:
            with self._db(db) as c:
                for here, there in (('src', 'dst'), ('dst', 'src')):
                    rows = c.execute(f'SELECT {there},rel,w FROM links WHERE {here} IN ({mark(ids)}) AND rel IN ({mark(rels)}) '
                                     f'ORDER BY w DESC, {there} LIMIT ?', (*ids, *rels, limit + len(ids) * DEGREE_CAP))
                    for other, rel, w in rows:
                        if other not in ids and w > out.get((other, rel), -1):
                            out[(other, rel)] = w
        except sqlite3.OperationalError:  # read-only runtime without links table yet
            return []
        ranked = sorted(((o, r, w) for (o, r), w in out.items()), key=lambda t: (-t[2], t[0], t[1]))
        return ranked[:limit]

    def for_node(self, id, db=None):
        """All edges touching a node -> [(src, dst, rel, w)]."""
        try:
            with self._db(db) as c:
                return c.execute('SELECT src,dst,rel,w FROM links WHERE src=? OR dst=? ORDER BY w DESC, rel, src, dst', (id, id)).fetchall()
        except sqlite3.OperationalError:
            return []

    # -- deterministic derivation --------------------------------------------
    def forget(self, ids, db=None):
        with self._db(db) as c:
            for id in ids:
                c.execute('DELETE FROM links WHERE src=? OR dst=?', (id, id))

    def name_map(self, db):
        """Lowercase source path (with and without .md, and bare stem) -> record id, for wikilink targets."""
        names = {}
        for id, payload in db.execute('SELECT id,payload FROM records'):
            names.setdefault(id.casefold(), id)
            source = json.loads(payload).get('source')
            if source:
                path = source[:-3] if source.lower().endswith('.md') else source
                for key in (path, posixpath.basename(path)):
                    names.setdefault(key.casefold(), id)
        return names

    def derive(self, rec, db, names=None):
        id, kind, sess = rec['id'], _type(rec), _node(rec)
        managed = ['in_project', 'from_session', 'supersedes', 'mentions'] + (['next', 'touches'] if kind == 'session' else [])
        db.execute(f'DELETE FROM links WHERE src=? AND rel IN ({",".join("?" * len(managed))})', (id, *managed))
        project = rec.get('project')
        self.add(id, 'project:' + (project or '_none'), 'in_project', db=db)
        if kind != 'session' and kind != 'note':
            self.add(id, sess or 'session:unknown', 'from_session', db=db)  # no orphans
        for old in rec.get('supersedes') or []:
            self.add(id, old, 'supersedes', db=db)
        if sess:
            self._touches(sess, rec.get('files') or [], db)
        if kind == 'session' and sess:
            self._next(rec, sess, db)
        self._rejects(rec, kind, db)
        if WIKILINK and kind == 'note' and isinstance(rec.get('text'), str):
            names = names if names is not None else self.name_map(db)
            for raw in WIKILINK.findall(_wikilink_metni(rec['text'])):
                target = _hedef(raw)
                dst = names.get(target.casefold()) or names.get(posixpath.basename(target).casefold()) if target else None
                if dst:
                    self.add(id, dst, 'mentions', db=db)

    def _touches(self, sess, files, db):
        have = {r[0] for r in db.execute("SELECT dst FROM links WHERE src=? AND rel='touches'", (sess,))}
        counts = {}
        for f in files:
            if isinstance(f, str) and f:
                d = posixpath.dirname(f.replace('\\', '/')) or '.'
                counts[d] = counts.get(d, 0) + 1
        for d in sorted(counts, key=lambda d: (-counts[d], d)):
            if len(have) >= MAX_AREAS:
                break
            self.add(sess, 'area:' + d, 'touches', db=db)
            have.add('area:' + d)

    def _next(self, rec, sess, db):
        ts = rec.get('ts') or rec.get('updated_at') or rec.get('started_at')
        if not ts:
            return
        best = None
        # ponytail: scans the project's session records; add an indexed column if a project has thousands
        for (payload,) in db.execute(f"SELECT payload FROM records WHERE {_TYPE}='session' AND json_extract(payload,'$.project') IS ?", (rec.get('project'),)):
            other = json.loads(payload)
            ots, onode = other.get('ts') or other.get('updated_at') or other.get('started_at'), _node(other)
            if ots and onode and onode != sess and (ots, onode) < (ts, sess) and (best is None or (ots, onode) > best[0]):
                best = ((ots, onode), onode)
        if best:
            self.add(sess, best[1], 'next', db=db)

    def _rejects(self, rec, kind, db):
        key, pol = rec.get('key'), rec.get('polarity')
        if kind not in ('decision', 'rejected') or not key or pol is None:
            return
        other_kind = 'decision' if kind == 'rejected' else 'rejected'
        for oid, payload in db.execute(f"SELECT id,payload FROM records WHERE {_TYPE}=? AND json_extract(payload,'$.key')=?", (other_kind, key)):
            opol = json.loads(payload).get('polarity')
            if opol is not None and opol != pol and oid != rec['id']:
                self.add(*((rec['id'], oid) if kind == 'rejected' else (oid, rec['id'])), 'rejects', db=db)

    def derive_missing(self, db):
        """Derive edges for records that have no in_project edge yet (e.g. written by capture, not by sync)."""
        rows = db.execute("SELECT payload FROM records WHERE id NOT IN (SELECT src FROM links WHERE rel='in_project')").fetchall()
        if rows:
            names = self.name_map(db)
            for (payload,) in rows:
                self.derive(json.loads(payload), db, names)
        return len(rows)


# -- module-level API: links.add / links.expand / links.for_node ---------------
# The caller binds the store once per process (bind(store)); without it expand() returns [].
_bound = None


def bind(store):
    global _bound
    _bound = Links(store)
    return _bound


def _links(store):
    return Links(store) if store is not None else _bound


def add(src, dst, rel, w=None, store=None):
    if _links(store):
        _links(store).add(src, dst, rel, w)


def expand(ids, rels, limit, store=None):
    l = _links(store)
    return l.expand(ids, rels, limit) if l else []


def for_node(id, store=None):
    l = _links(store)
    return l.for_node(id) if l else []
