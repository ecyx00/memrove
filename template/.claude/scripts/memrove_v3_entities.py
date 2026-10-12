"""E11: entity layer. Platforms/tools/services named in saved docs -> merged entity rows, a Haiku-compiled
profile (J4, cited per line), a stale chain from source-doc hashes, and match() for message-time recall (E12).

Entities live in table `entities(id, data JSON)` with data.kind == 'entity' (not in `records`: sync owns that table).
Edges doc -> entity use rel `about_entity` in the links table.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

KINDS = ("platform", "tool", "service", "concept", "project")
MAX_ALIASES, MAX_PER_DOC, MAX_J4, MAX_SRC_CHARS, MAX_WORDS, HEAD = 8, 5, 5, 12000, 400, 5
SECTIONS = ("what", "settings", "rules", "ui", "pitfalls")
MODEL_VERSION = "haiku"
REL = "about_entity"
SCHEMA = "CREATE TABLE IF NOT EXISTS entities(id TEXT PRIMARY KEY, data TEXT NOT NULL)"

STOP = frozenset("""app api web data test tests user users admin server client file files page pages code name list item items type
text view form button config setting settings option options error errors log logs key keys token tokens value values field fields
table tables rule rules docs doc note notes task tasks project projects system service tool tools platform module modules
one two new old all any the and for you not but are was has have with from this that what when then than into out
get set run use make add remove create update delete save load open close start stop home main base core util utils lib
version build deploy release main dev prod local cloud mobile desktop screen menu tab panel
ayar ayarlar ayari proje projeler sistem veri kural kurallar belge dosya sayfa ekran sunucu kullanici yonetici uygulama
hata liste form ogesi tur metin anahtar deger alan tablo gorev not notlar surum yayin yerel bulut ana temel
bir iki ben sen biz siz ama ile icin gibi daha cok her bu su ise veya ya ki mi mu""".split())


def fold(text):
    """Lowercase, ascii-folded (Turkish aware), non-alphanumerics -> single spaces."""
    # Fold Turkish dotted/dotless i.
    s = str(text).replace("İ", "i").replace("ı", "i")
    s = unicodedata.normalize("NFKD", s.casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def valid_alias(a):
    a = fold(a)
    return len(a.replace(" ", "")) >= 3 and a not in STOP


def _slug(name):
    return fold(name).replace(" ", "-") or "entity"


def _eid(name):
    return "entity:" + _slug(name)


def _keys(e):
    return {fold(x) for x in [e["name"], *e.get("aliases", [])] if fold(x)}


def _load(db):
    db.execute(SCHEMA)
    return {r[0]: json.loads(r[1]) for r in db.execute("SELECT id,data FROM entities")}


def _save(db, e):
    e["derived_from"] = list(e["sources"])  # same chain as `sources` under the name the record layer uses (memrove_v3_sync._mark_source_missing)
    db.execute("INSERT OR REPLACE INTO entities VALUES (?,?)", (e["id"], json.dumps(e, sort_keys=True, ensure_ascii=False)))


def doc_sha(row_data, mtime, size):
    return (row_data or {}).get("sha") or f"{mtime}:{size}"


def _doc_shas(db):
    try:
        return {r[0]: doc_sha(json.loads(r[3]), r[1], r[2]) for r in db.execute("SELECT id,mtime,size,data FROM docs")}
    except sqlite3.OperationalError:
        return {}


def record_doc_entities(store, doc_id, ents, db=None, lk=None):
    """Replace a doc's entity attachments by `ents` ([{name, aliases, kind}]), merging into existing entities. -> count attached."""
    import memrove_v3_links as links
    lk = lk or links.Links(store)
    own = db is None
    ctx = store._connect() if own else None
    db = ctx.__enter__() if own else db
    try:
        known = _load(db)
        db.execute("DELETE FROM links WHERE src=? AND rel=?", (doc_id, REL))
        for e in known.values():  # re-tagging: detach first, re-attach below
            if doc_id in e["sources"]:
                e["sources"].remove(doc_id)
        n = 0
        for raw in ents[:MAX_PER_DOC]:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not fold(raw["name"]):
                continue
            kind = raw.get("kind") if raw.get("kind") in KINDS else "concept"
            aliases = [a.strip() for a in raw.get("aliases") or [] if isinstance(a, str) and valid_alias(a)][:MAX_ALIASES]
            new = {"name": raw["name"].strip()[:80], "aliases": aliases, "kind": kind}
            nk = _keys(new)
            hit = next((e for e in known.values() if fold(e["name"]) in nk or fold(new["name"]) in _keys(e)
                        or (e["entity_kind"] == kind and {k for k in nk & _keys(e) if valid_alias(k)})), None)
            if hit is None:
                hit = {"id": _eid(new["name"]), "kind": "entity", "name": new["name"], "aliases": aliases, "entity_kind": kind,
                       "sources": [], "source_hashes": {}, "compiled_at": None, "stale": True, "profile": []}
                if hit["id"] in known:  # slug clash with an unmerged entity of another kind
                    hit["id"] += "-" + kind
                known[hit["id"]] = hit
            else:
                hit["aliases"] = sorted(set(hit["aliases"]) | set(aliases), key=str.casefold)[:MAX_ALIASES * 2]
            if doc_id not in hit["sources"]:
                hit["sources"] = sorted([*hit["sources"], doc_id])
            hit["stale"] = True
            hit.pop("status", None)  # a new live source revives a source_missing entity
            lk.add(doc_id, hit["id"], REL, 0.7, db=db)
            n += 1
        for e in list(known.values()):
            if e["sources"]:
                _save(db, e)
            else:
                db.execute("DELETE FROM entities WHERE id=?", (e["id"],))
                db.execute("DELETE FROM links WHERE dst=? AND rel=?", (e["id"], REL))
        return n
    finally:
        if own:
            ctx.__exit__(None, None, None)


def refresh_stale(store):
    """Source hash changed or source removed -> stale=true (the J4 queue). Entities left without live sources become status=source_missing (kept, never matched or projected). -> stale count."""
    with store._connect() as db:
        ents, shas, n = _load(db), _doc_shas(db), 0
        for e in ents.values():
            live = [s for s in e["sources"] if s in shas]
            changed = len(live) != len(e["sources"]) or any(e["source_hashes"].get(s) != shas[s] for s in live)
            if not live:  # every source doc vanished: keep the row for the audit trail, out of match() and the entity note
                if e.get("status") != "source_missing":
                    _save(db, dict(e, status="source_missing", stale=False))
                    db.execute("DELETE FROM links WHERE dst=? AND rel=?", (e["id"], REL))
                continue
            if changed or e.get("status") == "source_missing":
                e.pop("status", None)
                e["sources"], e["stale"] = live, True
                db.execute("DELETE FROM links WHERE dst=? AND rel=? AND src NOT IN (%s)" % ",".join("?" * len(live)), (e["id"], REL, *live))
                _save(db, e)
            n += bool(e["stale"])
    return n


def _clean_profile(data, sources):
    """J4 JSON -> [{s, t}] keeping only lines whose trailing [id] is a real source; <= MAX_WORDS words."""
    out, words = [], 0
    secs = data.get("sections") if isinstance(data, dict) else None
    if not isinstance(secs, dict):
        return out
    for s in SECTIONS:
        for t in secs.get(s) or []:
            m = re.search(r"\[([^\[\]]+)\]\s*$", t) if isinstance(t, str) else None
            if not m or m.group(1) not in sources:
                continue
            w = len(t.split())
            if words + w > MAX_WORDS:
                return out
            out.append({"s": s, "t": " ".join(t.split())})
            words += w
    return out


def _source_text(db, ids, root):
    rows = {r[0]: r for r in db.execute("SELECT id,project,path,title,summary FROM docs WHERE id IN (%s)" % ",".join("?" * len(ids)), ids)} if ids else {}
    per, parts = MAX_SRC_CHARS // max(1, len(rows)), []
    for i in ids:
        r = rows.get(i)
        if not r:
            continue
        text = ""
        for base in ([Path(root)] if root else []) + [Path()]:
            try:
                text = (base / r[2]).read_text(encoding="utf-8", errors="replace")
                break
            except OSError:
                continue
        text = text or f"{r[3]}\n{r[4]}"
        parts.append(f"[{i}]\n{text[:max(200, per - len(i) - 4)]}")
    return "\n\n".join(parts)[:MAX_SRC_CHARS + 4000]


def compile_entities(store, vault, state, root=None, limit=MAX_J4):
    """J4: stale entities -> cited Turkish profile via the gated call(). -> compiled count."""
    import memrove_v3_llm as llm
    refresh_stale(store)
    with store._connect() as db:
        todo = [e for e in _load(db).values() if e["stale"] and e.get("status") != "source_missing"][:limit]
        shas = _doc_shas(db)
    n = 0
    for e in todo:
        with store._connect() as db:
            text = _source_text(db, e["sources"], root)
        ev = hashlib.sha256(f'{e["name"]}\n{text}'.encode("utf-8")).hexdigest()[:16]  # the evidence the model would see
        if ev == e.get("evidence_hash"):  # nothing new since the last answered run (touched file, re-tag, empty profile): no call, just un-stale
            with store._connect() as db:
                cur = _load(db).get(e["id"])
                if cur:
                    _save(db, dict(cur, stale=False, source_hashes={s: shas[s] for s in cur["sources"] if s in shas}))
            continue
        payload = json.dumps({"entity": e["name"], "aliases": e["aliases"], "kind": e["entity_kind"], "documents": text}, ensure_ascii=False)
        st = {}
        data = llm.call(state, vault, "j4", payload, root, status=st)
        if data is None:
            if st.get("reason") == "failed":
                continue  # stays stale; retried next run
            break  # disabled, cap or reserved for J2/J1: the rest would hit the same wall
        profile = _clean_profile(data, set(e["sources"]))
        with store._connect() as db:
            cur = _load(db).get(e["id"])
            if not cur:
                continue
            cur["evidence_hash"] = ev  # answered: the same evidence is never asked again, even when the answer had no citable line
            if profile:
                cur.update(profile=profile, compiled_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), origin_type="derived", model_version=llm.MODEL)
            cur.update(stale=False, source_hashes={s: shas[s] for s in cur["sources"] if s in shas})
            _save(db, cur)
        n += bool(profile)
    return n


def all_entities(db):
    return [e for e in _load(db).values() if e.get("status") != "source_missing"]


# -- read API (E12) -----------------------------------------------------------------------
_CACHE = {}


def _stamp(store):
    db = Path(store.database)
    out = []
    for p in (db, Path(str(db) + "-wal")):
        try:
            st = p.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def _alias_map(store):
    """-> (compiled regex or None, {alias: (entity_id, head)}), rebuilt only when the db file changed. One SQL query."""
    key, stamp = str(store.database), _stamp(store)
    hit = _CACHE.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        with store._connect() as db:
            rows = [e for e in (json.loads(r[0]) for r in db.execute("SELECT data FROM entities")) if e.get("status") != "source_missing"]
    except sqlite3.OperationalError:
        rows = []
    owners = {}
    for e in rows:
        for a in _keys(e):
            if valid_alias(a):
                owners.setdefault(a, set()).add(e["id"])
    amap = {}
    for e in rows:
        head = "\n".join(p["t"] for p in e.get("profile", [])[:HEAD])
        for a in _keys(e):
            if a in owners and len(owners[a]) == 1:  # claimed by two entities -> ambiguous, never matched
                amap[a] = (e["id"], head)
    # aliases >=5 chars may carry a Turkish suffix tail (supabaseden, vercelin); shorter ones need strict word boundaries
    alt = lambda xs: "|".join(re.escape(a) for a in sorted(xs, key=len, reverse=True))
    long_, short = [a for a in amap if len(a) >= 5], [a for a in amap if len(a) < 5]
    parts = ([f"(?P<l>{alt(long_)})[a-z]{{0,5}}"] if long_ else []) + ([f"(?P<s>{alt(short)})"] if short else [])
    rx = re.compile(r"(?<![a-z0-9])(?:" + "|".join(parts) + r")(?![a-z0-9])") if parts else None
    _CACHE[key] = (stamp, (rx, amap))
    return rx, amap


def match(store, text):
    """text -> [(entity_id, alias, profile_head)], one per entity, in order of first appearance. Fail-open: [] on any error."""
    try:
        rx, amap = _alias_map(store)
        if not rx:
            return []
        out, seen = [], set()
        for m in rx.finditer(fold(text)):
            alias = m.group(m.lastgroup)
            eid, head = amap[alias]
            if eid not in seen:
                seen.add(eid)
                out.append((eid, alias, head))
        return out
    except Exception:
        return []


def note_path(entity_id):
    return f"memrove/entities/{entity_id.split(':', 1)[1]}.md"
