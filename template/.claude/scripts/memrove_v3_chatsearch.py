"""Search over the chat log: sqlite FTS5 (bm25) on the verbatim messages, one `id|date|kind|snippet` line per hit.

The model reads the tree's summaries; a minor topic can be reduced to a word there and one topic is scattered over branches that
only merge by time. Search finds the messages by their words; `zoom id 1` opens one whole.

state/chatlog/search.db   m(id, kind, date, text) + fts(text, folded); built incrementally from the append-only log (ids never change).
Turkish: dotless i is folded to i on both sides, unicode61 drops the other diacritics, and a long query word is cut to its stem
(`STEM` letters) and matched as a prefix, so "kararlarımızı" finds "karar", "kararı", "kararlar". A prefix-only match scores PREFIX
times a full-word match (`graphql` beats `graph*`). A hyphenated name is also indexed joined (`context-mode` -> `contextmode`) and a
query's adjacent words are tried as a phrase, so `context-mode`, `context mode` and `contextmode` meet.

One search for the agent: the records the owner or the agent saved on purpose (`memrove remember`: decision, rejected, lesson,
preference, note; stored as notes/observed/*) are ranked in the same list as `rec:<id>|date|kind|snippet`. Auto-captured, imported,
hub, `dropped_llm` and J1-pending records stay in the store but are not searched here. A record counts as of T only when its
`valid_from` day is before T's day (T a bare date: up to and including that day); one that is expired or replaced through
`supersedes` is hidden. `memrove show <id>` (or `zoom rec:<id> 1`) opens a record with its why.
"""
import functools
import hashlib
import json
import os
import re
from pathlib import Path
from datetime import date as _date, datetime, timezone
import sqlite3

import memrove_v3_chatlog as chatlog

REC_CACHE_VERSION = 1  # bump when _rec_body or index_text changes

STEM = 5          # a query word longer than this is cut to this many letters and matched as a prefix
TOP = 8           # hits printed
SNIPPET = 160     # characters of a hit's line
PREFIX = 0.6      # weight of a prefix-only match against a full-token match
CAND = 400        # candidates each of the two match passes keeps
VERSION = 2       # index layout; a different one is rebuilt
DENSE_POOL = 30   # hybrid: candidates each of the two lists contributes
POOL = 30         # bm25 candidates the optional rerank looks at
# Turkish dotless i folds to i.
FOLD = str.maketrans({"ı": "i", "İ": "i", "I": "i"})
WORD = re.compile(r"\w+", re.UNICODE)
HYPHENATED = re.compile(r"\w+(?:[-_]\w+)+", re.UNICODE)
REC_KINDS = ("decision", "rejected", "lesson", "preference", "note")
REC_SOURCES = ("notes/observed/", "notes/captured/", "notes/inferred/", "wiki/_concepts/")   # `memrove remember` (obs-), cue capture (q-), J2 (p-), learning-wiki concept pages
REC_WEIGHT = 2.5   # a record's bm25 is scaled up: the few hundred records give every term a lower idf than the thousands of messages do
OTHER = 0.6      # score factor of a q-/p- record against an obs- one
REC_SLOTS = 2      # records in one answer at most, so a hub of records cannot push the messages out
HYBRID_REC_SLOTS = 3   # records in a hybrid answer at most: the gate is top-3, so a record in the top 3 is only ever pushed out by 3 records above it
GRAPH_HITS = 3     # record hits that get neighbour lines
GRAPH_SHARE = 0.4  # neighbour lines and newer records add at most this share of the plain output
REC_OFF = frozenset(("retired", "rejected", "dropped_llm", "deleted_by_user", "source_missing"))


SECRET = re.compile(
    r"(?s:-----BEGIN[ A-Z]*PRIVATE KEY-----.*?(?:-----END[ A-Z]*PRIVATE KEY-----|$))"   # a private key block, to its end or to the end of the text
    r"|\b(?:sk|pk|rk)-[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_\w{20,}|\bxox[abprs]-[\w-]{10,}"   # OpenAI/Anthropic, GitHub, Slack
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|\bAIza[\w-]{30,}|\bKGAT_\w{16,}"                                              # AWS, Google, dataset-host
    r"|\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}|\bBearer\s+[\w.~+/=-]{20,}"                                           # JWT, bearer header
    r"|(?i:((?:key|secret|token|passw(?:or)?d)\w*\s*[:=]\s*[\"']?)([^\s\"'`,;)]{16,}))")                         # name = long value with a digit
REDACTED = "[redacted secret]"


def redact(text):
    """The text with the common secret shapes (API keys, tokens, private keys, `key = value`) replaced for display; the stored log is untouched."""
    def sub(m):
        if m.group(2) is None:
            return REDACTED
        return m.group(1) + REDACTED if any(c.isdigit() for c in m.group(2)) else m.group(0)
    return SECRET.sub(sub, text)


def index_text(text):
    """Folded text, plus each hyphenated name joined at the end so `contextmode` finds `context-mode`."""
    low = fold(text)
    return low + " " + " ".join(re.sub(r"[-_]", "", m) for m in HYPHENATED.findall(low)) if HYPHENATED.search(low) else low


def fold(text):
    return text.translate(FOLD).lower()


BUSY = 30.0   # seconds to wait for search.db's lock; lane_text lowers it (the hook has a 5 s timeout)


def _db(state):
    path = chatlog.root(state) / "search.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=BUSY)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS m(id INTEGER PRIMARY KEY, kind TEXT, date TEXT, text TEXT)")
    if db.execute("PRAGMA user_version").fetchone()[0] != VERSION:  # an older layout: rebuild from the log
        db.execute("DROP TABLE IF EXISTS fts")
        db.execute("DELETE FROM m")
        db.execute(f"PRAGMA user_version = {VERSION}")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(folded, tokenize='unicode61 remove_diacritics 2')")
    return db


def sync(state, db=None):
    """Index the messages the log gained since the last call (all of them after a log that shrank). -> messages added."""
    own = db is None
    db = db or _db(state)
    try:
        total = chatlog.count(state)
        last = db.execute("SELECT COALESCE(MAX(id), -1) FROM m").fetchone()[0]
        mark = db.execute("SELECT text FROM m WHERE id = ?", (last,)).fetchone() if last >= 0 else None
        same = mark is not None and next((r["text"] for r in chatlog.messages(state, last, last + 1)), None) == mark[0]
        if last >= total or (last >= 0 and not same):  # not the log this index was built from (a shorter or a replaced one)
            db.execute("DELETE FROM m")
            db.execute("DELETE FROM fts")
            last = -1
        rows = list(chatlog.messages(state, last + 1))
        db.executemany("INSERT INTO m VALUES (?,?,?,?)", [(r["i"], r["kind"], r.get("date", ""), r["text"]) for r in rows])
        db.executemany("INSERT INTO fts(rowid, folded) VALUES (?,?)", [(r["i"], index_text(r["text"])) for r in rows])
        db.commit()
        return len(rows)
    finally:
        if own:
            db.close()


def terms(query):
    """-> (exact, prefix) MATCH expressions over the folded query words, or None. exact: full tokens, adjacent pairs as a phrase.
    prefix: the long words cut to `stem*`."""
    words = list(dict.fromkeys(WORD.findall(fold(query))))
    seq = WORD.findall(fold(query))
    if not words:
        return None
    ex = [f'"{w}"' for w in words]
    ex += [f'"{a} {b}"' for a, b in zip(seq, seq[1:])]
    pre = [f'"{w[:STEM]}"*' if len(w) > STEM else f'"{w}"' for w in words]
    return " OR ".join(dict.fromkeys(ex)), " OR ".join(dict.fromkeys(pre))


def snippet(text, query, width=SNIPPET):
    """The text around the first query word (a stem match counts), one line, cut to width."""
    flat = redact(" ".join(text.split()))
    low = fold(flat)
    at = min((p for p in (low.find(w[:STEM]) for w in WORD.findall(fold(query))) if p >= 0), default=0)
    start = max(0, at - width // 4)
    cut = flat[start:start + width]
    return ("..." if start else "") + cut + ("..." if start + width < len(flat) else "")


def _bound(value, name):
    """An ISO date or datetime, or None. Messages keep their own ISO text, so bounds compare as text (a date is a prefix of its datetimes)."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} must be YYYY-MM-DD or an ISO datetime, got {value!r}") from None
    return value


T_IN = 1.5       # temporal boost: bm25 scale for a message dated inside the range the query names
T_FLOOR = 0.5    # ... and the floor for one far outside it (scale = 1/(1+days_out/T_DECAY), never below this)
T_DECAY = 60.0


def _when(query, ref, since, until):
    """-> (query without its time words, (since, until) date range) or (query, None). The range is a soft boost, not a filter: LongMemEval
    evidence dates sit around the event, so a range holds some evidence in 86% of questions and all of it in 32%. OFF by default (bench:
    docs/state/2026-11-07-temporal-wiring.md, no gain); MEMROVE_TEMPORAL=1 turns it on. Never with explicit since/until. shortcut: ref defaults to today; callers replaying old prompts pass the prompt's own date."""
    if since or until or os.environ.get("MEMROVE_TEMPORAL") != "1":
        return query, None
    import memrove_v3_temporal as temporal
    r = temporal.parse(query, ref or _date.today())
    if r is None:
        return query, None
    rest = query[:r[2][0]] + " " + query[r[2][1]:]
    return (rest if terms(rest) else query), (r[0], r[1])


def _scale(day, rng):
    """bm25 multiplier for a message dated `day` (bm25 is negative: more means better). Undated: 1."""
    try:
        d = _date.fromisoformat(str(day)[:10])
    except ValueError:
        return 1.0
    if rng[0] <= d <= rng[1]:
        return T_IN
    gap = (rng[0] - d).days if d < rng[0] else (d - rng[1]).days
    return max(T_FLOOR, 1.0 / (1.0 + gap / T_DECAY))


def hits(state, query, top=TOP, rerank=None, since=None, until=None):
    """-> [(id, date, kind, text)], best first. rerank: optional fn(query, [(id, text)]) -> ids reordered (fails open).
    since / until: ISO date or datetime on the message date, until inclusive (a bare date covers its whole day). With a range and
    no query word: the messages in the range, newest first."""
    return [row for _, row in _scored(state, query, top, rerank, since, until)]


def _scored(state, query, top, rerank, since, until, ref=None):
    """-> [(bm25 score or None for a range listing, (id, date, kind, text))], best first."""
    since, until = _bound(since, "since"), _bound(until, "until")
    qtext, rng = _when(query, ref, since, until)
    q = terms(qtext)
    if q is None and not (since or until):
        return []
    where, args = [], []
    if q:
        where.append("fts MATCH ?")
    if since or until:
        where.append("m.date != ''")
    if since:
        where.append("m.date >= ?")
        args.append(since)
    if until:
        where.append("substr(m.date, 1, ?) <= ?")
        args += [len(until), until]
    db = _db(state)
    try:
        sync(state, db)
        if q:
            score, day = {}, {}
            for expr, weight in ((q[0], 1.0), (q[1], PREFIX)):
                sql = f"SELECT m.id, bm25(fts), m.date FROM fts JOIN m ON m.id = fts.rowid WHERE {' AND '.join(where)} ORDER BY bm25(fts) LIMIT ?"
                for i, b, dt in db.execute(sql, (expr, *args, CAND)):
                    score[i] = score.get(i, 0.0) + weight * b
                    day[i] = dt
            if rng:
                score = {i: b * _scale(day[i], rng) for i, b in score.items()}
            keep = sorted(score, key=score.get)[:POOL if rerank else top]
            by_id = {r[0]: r for r in db.execute(f"SELECT id, date, kind, text FROM m WHERE id IN ({','.join('?' * len(keep))})", keep)}
            rows = [by_id[i] for i in keep]
            scores = [score[i] for i in keep]
        else:
            rows = db.execute(f"SELECT m.id, m.date, m.kind, m.text FROM m WHERE {' AND '.join(where)} ORDER BY m.date DESC, m.id DESC LIMIT ?",
                              (*args, top)).fetchall()
            scores = [None] * len(rows)
    finally:
        db.close()
    import memrove_v3_imports as imports   # stable-id imports (x-...): searched with the log, merged by score
    extra = imports.scored(state, q, since, until, top, CAND, PREFIX)
    if extra and rng:
        extra = [(sc * _scale(r[1], rng), r) for sc, r in extra]
    if extra:
        mine = list(zip(scores, rows))
        both = sorted(mine + extra, key=(lambda p: p[0]) if q else (lambda p: p[1][1]), reverse=not q)
        if q:
            both = both[:POOL if rerank else top]
        scores, rows = [p[0] for p in both], [p[1] for p in both]
    if rerank and rows and q:
        try:
            order = rerank(query, [(r[0], r[3]) for r in rows])
            ranked = sorted(zip(scores, rows), key=lambda p: order.index(p[1][0]) if p[1][0] in order else len(order))
            scores, rows = [p[0] for p in ranked], [p[1] for p in ranked]
        except Exception:  # fails open: the bm25 order stands
            pass
    return list(zip(scores, rows))[:top]


def _day_limit(until):
    """-> (day, inclusive): the day a record's valid_from is compared with. A datetime T excludes its own day (valid_from has no hour)."""
    return (until[:10], len(until) <= 10) if until else (None, True)


def _rec_body(r):
    keys = ("title", "text", "approach", "result", "situation", "why")
    if str(r.get("source") or "").startswith("wiki/"):   # a 26 KB wiki page must not lose bm25 to length: cap each field, title 3x
        return " ".join(str(r.get(k) or "")[:3000] for k in ("title", "title") + keys)
    return " ".join(str(r.get(k) or "") for k in keys)   # owner records rank as before 0153cc6


def _rec_cache_load(state):
    """{record id: [md5 of payload, the fields records() reads + folded text]} from state/records-cache.json; {} when absent or other version."""
    try:
        c = json.loads((Path(state) / "records-cache.json").read_text(encoding="utf-8"))
        return c["records"] if c.get("version") == REC_CACHE_VERSION else {}
    except (OSError, ValueError, KeyError, AttributeError):
        return {}


def _rec_cache_save(state, old, seen):
    """Write only when an entry changed or appeared (records dropped by the filters stay: a stale id is never read, its md5 must match)."""
    if all(old.get(k) == v for k, v in seen.items()):
        return
    try:
        tmp = Path(state) / f"records-cache.json.{os.getpid()}"
        tmp.write_text(json.dumps({"version": REC_CACHE_VERSION, "records": {**old, **seen}}), encoding="utf-8")
        os.replace(tmp, Path(state) / "records-cache.json")
    except OSError:
        pass


def records(state, query, top=TOP, since=None, until=None, session_notes=False, ref=None):
    """-> [(score, ("rec:<id>", valid_from, kind, text))]: the saved records matching the query, scored on the same bm25 scale as messages
    (exact and prefix passes weighted like `hits`, over a throwaway FTS table of the few hundred eligible records)."""
    qtext, rng = _when(query, ref, since, until)
    q = terms(qtext)
    live = _live(state, since, until, session_notes) if q is not None else []
    if not live:
        return []
    out = _bm25(live, q, CAND if rng else top)
    if rng:
        out = sorted(((sc * _scale(r[1], rng), r) for sc, r in out), key=lambda p: p[0])[:top]
    return out


def _live(state, since=None, until=None, session_notes=False):
    """-> [(id, record dict, day)]: the searchable records as of `until` (eligibility, scope, validity, supersedes)."""
    path = chatlog.root(state).parent / "memory.sqlite3"
    if not path.exists():
        return []
    until, since = _bound(until, "until"), _bound(since, "since")
    day, inclusive = _day_limit(until)
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() if not until else (until if len(until) > 10 else until + "T23:59:59")
    marks = ",".join("?" * len(REC_KINDS))
    j = lambda k: f"coalesce(json_extract(payload,'$.{k}'),'')"
    src = " OR ".join(j("source") + " LIKE ?" for _ in REC_SOURCES)
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        rows = db.execute(
            f"SELECT id, payload, {j('kind')}, {j('project')}, {j('scope')} FROM records WHERE {j('kind')} IN ({marks}) "
            f"AND (({src}) {'OR ' + j('source') + ' LIKE ' + repr('notes/sessions/%') if session_notes else ''}) "
            f"AND {j('status')} NOT IN ({','.join(repr(s) for s in sorted(REC_OFF))}) AND {j('verdict')}!='reject' AND {j('visibility')}!='private' "
            f"AND ({j('imported_from')} NOT LIKE '%-session' OR json_extract(payload,'$.assessments.j1.answers.durable')='yes')",
            (*REC_KINDS, *(x + "%" for x in REC_SOURCES))).fetchall()
        db.close()
    except sqlite3.Error:
        return []
    from memrove_v3_kinds import client_scopes, effective_scope, in_scope
    client = client_scopes(state)
    live, cache = [], _rec_cache_load(state)
    seen = {}
    for rid, payload, kind, project, scope in rows:
        if not in_scope(effective_scope({"kind": kind, "project": project, "scope": scope}), client):
            continue
        key = hashlib.md5(payload.encode("utf-8", "replace")).hexdigest()   # a changed payload hashes differently: re-parsed
        hit = cache.get(rid)
        if hit and hit[0] == key:
            r = hit[1]
        else:
            r = json.loads(payload)
            r = {**{k: r.get(k) for k in ("valid_from", "source", "valid_to", "supersedes", "kind", "text", "title", "approach")}, "folded": index_text(_rec_body(r))}
        seen[rid] = [key, r]
        when = str(r.get("valid_from") or "")[:10]
        if not when:  # session notes carry their day in the file name: notes/sessions/YYYY-MM-DD-...
            m = re.search(r"/(\d{4}-\d{2}-\d{2})-", str(r.get("source") or ""))
            when = m.group(1) if m else ""
        if (day and (not when or when > day or (when == day and not inclusive))) or (since and when < since[:10]):
            continue
        if str(r.get("valid_to") or "") and str(r["valid_to"]) <= now:
            continue
        live.append((rid, r, when))
    _rec_cache_save(state, cache, seen)
    gone = {s for _, r, _ in live for s in (r.get("supersedes") or [])}
    return [x for x in live if x[0] not in gone]


def _bm25(live, q, top):
    mem = sqlite3.connect(":memory:")
    mem.execute("CREATE VIRTUAL TABLE fts USING fts5(folded, tokenize='unicode61 remove_diacritics 2')")
    mem.executemany("INSERT INTO fts(rowid, folded) VALUES (?,?)", [(i, r["folded"]) for i, (_, r, _) in enumerate(live)])
    score = {}
    for expr, weight in ((q[0], 1.0), (q[1], PREFIX)):
        for i, b in mem.execute("SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?", (expr, CAND)):
            score[i] = score.get(i, 0.0) + weight * b
    mem.close()
    for i in score:   # bm25 is negative, lower is better: a cue-captured (q-) or inferred (p-) record ranks behind a saved (obs-) one
        if not str(live[i][1].get("source") or "").startswith(REC_SOURCES[0]):
            score[i] *= OTHER
    out = []
    for i in sorted(score, key=score.get)[:top]:
        rid, r, when = live[i]
        text = str(r.get("text") or r.get("title") or r.get("approach") or "")
        out.append((score[i], (f"rec:{rid}", when, r.get("kind") or "", text)))
    return out


def _dense(state, query, live):
    """-> record ids ("rec:<id>") of the live owner-saved (notes/observed) records by EmbeddingGemma 2 cosine to the query, best first. [] when the daemon or the vectors are missing."""
    import memrove_v3_embed as emb
    try:
        qv = emb.DaemonEmbedder(timeout=3).embed_query([query])[0]
    except (OSError, ValueError, KeyError):
        return []
    vecs = emb.VecStore(state, readonly=True).load(emb.DaemonEmbedder.name)
    sims = [(sum(a * b for a, b in zip(qv, vecs[rid])), rid) for rid, r, _ in live if rid in vecs and str(r.get("source") or "").startswith(REC_SOURCES[0])]
    return ["rec:" + rid for _, rid in sorted(sims, reverse=True)[:DENSE_POOL]]


TR_LETTERS = set("çğıöşüİÇĞÖŞÜ")
# Fixed a priori (not tuned on any eval set). Words that are also English words (de, da, ne, bu, var, ile) stay: they are rare in an English query.
TR_WORDS = {"ve", "bir", "bu", "için", "ile", "de", "da", "mi", "mı", "ne", "gibi", "ama", "çok", "değil", "yok", "var", "olarak", "neden", "nasıl"}


def is_turkish(query):
    """A Turkish-only letter, or one common Turkish function word. Records are English, so only a Turkish query needs the dense cross-lingual lane."""
    return any(c in TR_LETTERS for c in query) or any(w in TR_WORDS for w in re.findall(r"\w+", query.lower()))


def _hybrid(state, query, top, rerank, since, until, session_notes, ref=None):
    """qmd's fusion (RRF k=60, each list weight 2, +0.05 for a rank-1 and +0.02 for a rank 2-3 hit in either list) of the bm25 list above
    (messages and records) with the dense record list. Dense-only records enter as rows; messages keep their bm25 rank."""
    lane = merged(state, query, DENSE_POOL, rerank, since, until, session_notes, hybrid=False, ref=ref)
    live = _live(state, since, until, session_notes)
    rows = {r[0]: r for r in lane}
    by_id = {"rec:" + rid: ("rec:" + rid, when, r.get("kind") or "", str(r.get("text") or r.get("title") or r.get("approach") or "")) for rid, r, when in live}
    lists = [[r[0] for r in lane], _dense(state, query, live)]
    rule = os.environ.get("MEMROVE_HYBRID_RULE", "routed")   # routed (w2 for a Turkish query, bm25 for English), rrf (qmd), dense_first (dense top 3 then bm25), w2 (plain RRF, dense weight 2)
    if rule == "routed":
        rule = "w2"
    if rule == "dense_first":
        first = lists[1][:3]
        ids = first + [i for i in lists[0] if i not in first]
        return _cap_records([rows.get(i) or by_id[i] for i in ids])[:top]
    score = {}
    for li, ids in enumerate(lists):
        for rank, i in enumerate(ids):
            if rule == "w2":
                score[i] = score.get(i, 0.0) + (1, 2)[li] / (60 + rank + 1)
            else:
                score[i] = score.get(i, 0.0) + 2.0 / (60 + rank + 1) + (0.05 if rank == 0 else 0.02 if rank < 3 else 0.0)
    return _cap_records([rows.get(i) or by_id[i] for i in sorted(score, key=lambda i: -score[i])])[:top]


def _cap_records(rows):
    """Keep the first HYBRID_REC_SLOTS records in fused order; messages are never dropped."""
    n = 0
    out = []
    for r in rows:
        n += str(r[0]).startswith("rec:")
        if not str(r[0]).startswith("rec:") or n <= HYBRID_REC_SLOTS:
            out.append(r)
    return out


def merged(state, query, top=TOP, rerank=None, since=None, until=None, session_notes=False, hybrid=None, ref=None):
    """-> [(id, date, kind, text)]: messages and records in one list by bm25 score (records scaled by REC_WEIGHT, at most REC_SLOTS of them).
    A record's id is the string `rec:<id>`. hybrid (default on; env MEMROVE_HYBRID_SEARCH=0 turns it off): fuse that list with the EG2 dense record lane."""
    if hybrid is None:
        hybrid = os.environ.get("MEMROVE_HYBRID_SEARCH") != "0"
    if hybrid and query.strip() and (os.environ.get("MEMROVE_HYBRID_RULE", "routed") != "routed" or is_turkish(query)):
        return _hybrid(state, query, top, rerank, since, until, session_notes, ref)
    msgs = _scored(state, query, top, rerank, since, until, ref)
    recs = [(REC_WEIGHT * sc, row) for sc, row in records(state, query, REC_SLOTS, since, until, session_notes, ref)]
    if not recs:
        return [row for _, row in msgs]
    both = sorted(msgs + recs, key=lambda p: p[0])   # a reranked message list is not in score order; the rerank is an opt-in nobody ships yet
    return [row for _, row in both][:top]


def search(state, query, top=TOP, rerank=None, since=None, until=None, history=False):
    """The command's text: `head|...` lines (topic heads matching the query) first, then `id|date|kind|snippet` per hit, or a one-line note.
    Entries a newer head replaced stay, after the unreplaced hits and tagged `[replaced by REF]` (history=True: no tag, no reordering)."""
    import memrove_v3_docsearch as docsearch   # imported here: docsearch imports this module
    import memrove_v3_heads as heads
    rows = merged(state, query, top, rerank, since, until)
    docs = docsearch.handles(state, query)
    hs, replaced = heads.heads(state) if query.strip() or not history else ([], {})   # built once: it re-parses every write-up
    top_heads = heads.match(state, query, hs=hs) if query.strip() else []
    gone = {} if history else replaced
    shown = {h["id"] for h in top_heads}
    rows = [r for r in rows if r[0] not in shown]
    by = lambda d: next((v for k, v in gone.items() if k == d.split("|")[2] or k.endswith("/" + d.split("|")[2])), None)
    docs = [d for d in docs if not by(d)] + [f"{d} [replaced by {by(d)}]" for d in docs if by(d)]
    rows = [r for r in rows if r[0] not in gone] + [(i, d, f"{k} [replaced by {gone[i]}]", t) for i, d, k, t in rows if i in gone]
    head_lines = [f"head|{h['day']}|{h['topic']}{' ' + h['status'] if h['status'] else ''}|{h['ref']}|{snippet(h['decision'] + (' | next: ' + h['next'] if h['next'] else ''), query)}" for h in top_heads]
    line = lambda r: f"{r[0]}|{(r[1] or '')[:16].replace('T', ' ')}|{r[2]}|{snippet(r[3], query)}"
    tail = ("\n\ndocs (open with memrove doc PATH [--section HEADING]; Read also works):\n" + "\n".join(docs)) if docs else ""
    if not (rows or top_heads):
        return "no message matches" + tail
    plain = "\n".join(head_lines + [line(r) for r in rows]) + tail
    rows, under = graph(state, rows, GRAPH_SHARE * len(plain), history)
    return "\n".join(head_lines + [x for r in rows for x in [line(r), *under.get(r[0], [])]]) + tail


def graph(state, rows, room, history=False):
    """Link-graph additions to a hit list -> (rows, {rec id: [neighbour line]}). A hit that a newer record replaced through a supersedes
    edge is preceded by that record and tagged `[replaced by ...]`; each of the first GRAPH_HITS record hits gets up to 2 neighbour lines.
    Everything added fits in `room` characters. No edges, no store or no graph tables: rows unchanged."""
    import memrove_v3_neighbours as nb
    ids = [r[0] for r in rows if str(r[0]).startswith("rec:")]
    if not ids:
        return rows, {}
    out, used = [], 0
    newer = {} if history else nb.newer(state, ids)
    for r in rows:
        n = newer.get(r[0])
        if n and n not in ids and (new := nb.record_row(state, n)) and used + len(new[3]) <= room:
            used += len(new[3])
            out += [new, (r[0], r[1], f"{r[2]} [replaced by {n}]", r[3])]
            ids.append(n)
        else:
            out.append(r)
    top = [r[0] for r in out if str(r[0]).startswith("rec:") and "[replaced by" not in r[2]][:GRAPH_HITS]
    under = {}
    for hit, lines in nb.neighbours(state, top, known=set(ids)).items():
        for x in lines:
            if used + len(x) + 1 <= room:
                under.setdefault(hit, []).append(x)
                used += len(x) + 1
    return out, under


# ---- auto-recall lane: pointer lines for the UserPromptSubmit hook ----------------------------------------------------------
LANE_TOKENS = 300      # hard cap of the chat lane (estimated, see _tokens)
LANE_MIN = 14.0        # strength gate: a hit's summed -bm25 must reach this; below it the lane injects nothing (swept on LongMemEval-S)
LANE_POOL = 30         # candidates fetched (some are dropped as already in context)
LANE_RECENT = 40       # no transcript: this many newest log messages count as in context
LANE_RARE_DF = 36      # a query word is rare when at most this many log messages hold it (absolute: a share would loosen as the log grows)
LANE_MIN_RARE = 1      # a hit must hold this many rare query words
LANE_MIN_WORDS = 3     # the prompt must have this many words of 4+ letters, else it is chatter ("okey", "başla")
LANE_AGENT = float(os.environ.get("MEMROVE_LANE_AGENT", "0.7"))   # weight of an assistant message: long, so it matches more words than the owner's own
LANE_LINES = 4        # hits looked at after the gate and dedup, best first
LANE_SNIP = 120        # characters of a snippet at most
LANE_DUP = 0.8         # word-set overlap above which a hit repeats an earlier one
LANE_HEADER = "Chat log (`memrove zoom <id> 1` opens):"
_SENT = re.compile(r"(?<=[.!?\n])\s+")


def _tokens(s):
    """Token estimate, stdlib only: chars/3 rounded up. o200k_base runs at about 4 chars a token on English prose and 2.5-3 on ids and
    non-English, so the 300-token cap holds in real tokens too (bench/lme.py checks the real count with tiktoken)."""
    return -(-len(s) // 3)


def window(text, query, width=LANE_SNIP):
    """The 1-2 sentences holding the most query stems (not the message head), one line, cut to width."""
    stems = {w[:STEM] for w in WORD.findall(fold(query)) if len(w) > 2}
    flat = redact(text)
    sents = [s.strip() for s in _SENT.split(flat) if s.strip()] or [" ".join(flat.split())]
    hit = [sum(st in fold(s) for st in stems) for s in sents]
    i = max(range(len(sents)), key=lambda k: (hit[k], -k))
    out = " ".join(sents[i].split())
    if len(out) < width // 2 and i + 1 < len(sents):
        out += " " + " ".join(sents[i + 1].split())
    if len(out) > width:   # one long sentence: centre the cut on the first stem
        low = fold(out)
        at = min((p for p in (low.find(st) for st in stems) if p >= 0), default=0)
        start = max(0, at - width // 4)
        out = ("..." if start else "") + out[start:start + width] + "..."
    return out


LANE_TAIL = 4 << 20    # bytes of the transcript read: older turns are compacted out of context anyway, and a long session's file reaches 300 MB


@functools.lru_cache(maxsize=2)   # bench replays many prompts of one transcript; the hook is one call per process
def _read(path):
    with open(path, "rb") as f:
        f.seek(max(0, f.seek(0, os.SEEK_END) - LANE_TAIL))
        return f.read().decode("utf-8", errors="replace")


def _in_context(state, ids, transcript):
    """-> the ids the agent already has: messages of this session (their uuid is in the hook's transcript file) or, with no readable
    transcript, the LANE_RECENT newest messages of the log. The log has no session field; a message's `k` is its transcript uuid."""
    try:
        data = _read(transcript) if transcript else None
    except OSError:
        data = None
    if data is None:
        top = chatlog.count(state)
        return {i for i in ids if isinstance(i, int) and i >= top - LANE_RECENT}
    out = set()
    want = {i for i in ids if isinstance(i, int)}
    if not want:
        return out
    for r in chatlog.messages(state, min(want), max(want) + 1):   # one pass: a get() per id rescans the file each time
        k = str(r.get("k", "")).split("#")[0]
        if r["i"] in want and k.startswith("c:") and k[2:] in data:
            out.add(r["i"])
    return out


def _rare_stems(state, query):
    """Query stems that are rare in the log (document frequency <= LANE_RARE_DF messages): the words that make a match specific."""
    stems = list(dict.fromkeys(w[:STEM] if len(w) > STEM else w for w in WORD.findall(fold(query)) if len(w) > 2))[:24]
    db = _db(state)
    try:
        return [w for w in stems if db.execute("SELECT count(*) FROM fts WHERE fts MATCH ?", (f'"{w}"*',)).fetchone()[0] <= LANE_RARE_DF]
    finally:
        db.close()


def lane_candidates(state, query, transcript=None, upto=None, ref=None):
    """-> [(strength, id, date, text, rare)] best first: chat-log hits (never records) not already in the agent's context, strength = summed
    -bm25 (assistant x LANE_AGENT), rare = how many rare query stems the message holds. upto: ignore ids >= upto (bench replay)."""
    rows = [(-sc * (LANE_AGENT if r[2] == "agent" else 1.0), r[0], r[1], r[3]) for sc, r in _scored(state, query, LANE_POOL, None, None, None, ref) if sc is not None]
    if upto is not None:
        rows = [r for r in rows if not isinstance(r[1], int) or r[1] < upto]
    skip = _in_context(state, [r[1] for r in rows], transcript)
    rare = _rare_stems(state, _when(query, ref, None, None)[0])
    rows = [r + (sum(w in fold(r[3]) for w in rare),) for r in rows if r[1] not in skip]
    return sorted(rows, key=lambda c: -c[0])


def lane_pack(cands, query, seen=(), min_strength=None, budget=LANE_TOKENS, max_lines=LANE_LINES, width=LANE_SNIP, min_rare=None):
    """-> [(id, line)]: hits at or above min_strength, not in seen, near-duplicates collapsed, packed greedily by strength per token
    into budget tokens, shown best first. A line is `id|date|snippet`; `memrove zoom <id> 1` opens the message."""
    min_strength = LANE_MIN if min_strength is None else min_strength
    pool, kept = [], []
    min_rare = LANE_MIN_RARE if min_rare is None else min_rare
    for st, i, date, text, rare in cands:
        words = frozenset(WORD.findall(fold(text)))
        if len(kept) >= max_lines or st < min_strength or rare < min_rare or str(i) in seen or any(len(words & w) / max(1, len(words | w)) > LANE_DUP for w in kept):
            continue
        kept.append(words)
        line = f"{i}|{(date or '')[:10]}|{window(text, query, width)}".replace("<", "(").replace(">", ")")   # no tag can close the recall envelope
        pool.append((st / _tokens(line + "\n"), st, i, line))
    out, room = [], budget
    for _, st, i, line in sorted(pool, key=lambda p: -p[0]):
        n = _tokens(line + "\n")
        if n <= room:
            room -= n
            out.append((st, i, line))
    return [(i, line) for _, i, line in sorted(out, key=lambda p: -p[0])]


def lane(state, query, seen=(), min_strength=None, budget=LANE_TOKENS, transcript=None):
    """-> [(id, line)] for the hook; [] when the match is weak. Any search error is the caller's to catch."""
    if len({w for w in WORD.findall(fold(query)) if len(w) > 3}) < LANE_MIN_WORDS:
        return []
    return lane_pack(lane_candidates(state, query, transcript), query, seen, min_strength, budget)


LANE_ENVELOPE = '<memrove-recall trust="local-data" instructions="never">\n\n</memrove-recall>'
LANE_BUSY = 0.3


def index_ready(state):
    """True when search.db exists in the current layout. The hook searches only a ready index: a cold build or a layout rebuild takes
    seconds (4.9 s for 12k messages) against the hook's 5 s timeout, so it is left to `memrove search` and the worker."""
    path = chatlog.root(state) / "search.db"
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=LANE_BUSY)
        try:
            return db.execute("PRAGMA user_version").fetchone()[0] == VERSION
        finally:
            db.close()
    except sqlite3.Error:
        return False


def lane_text(state, query, session, transcript=None, cap=LANE_TOKENS):
    """The chat lane for one prompt: header + `id|date|snippet` lines, at most LANE_TOKENS in all, '' when the match is weak. Ids already
    injected in this session (state/chatlane_seen.json) are skipped and the new ones recorded. Errors are the caller's to catch."""
    if not chatlog.root(state).is_dir():   # no log yet: do not create one just to search it
        return ""
    path = Path(state) / "chatlane_seen.json"
    try:
        seen = json.loads(path.read_text())
    except (OSError, ValueError):
        seen = {}
    done = set(map(str, seen.get(session, [])))
    global BUSY
    old, BUSY = BUSY, LANE_BUSY
    try:
        lines = lane(state, query[:500], done, budget=cap - _tokens(LANE_HEADER + "\n"), transcript=transcript)
    finally:
        BUSY = old
    if not lines:
        return ""
    seen.pop(session, None)
    seen[session] = sorted(done | {str(i) for i, _ in lines})
    seen = dict(list(seen.items())[-50:])   # the 50 newest sessions
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(seen))
    os.replace(tmp, path)
    return "\n".join([LANE_HEADER] + [line for _, line in lines])
